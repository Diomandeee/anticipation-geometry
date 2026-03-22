# cc-anticipation: Implementation Guide

**Document Version**: 0.1.0
**Created**: 2025-12-26
**Parent**: [PROJECT_CHARTER.md](./PROJECT_CHARTER.md)

---

## 1. Module Structure

```
cc-anticipation/
├── Cargo.toml
├── src/
│   ├── lib.rs                    # Public API, re-exports
│   ├── types.rs                  # MotionWindow, AnticipationPacket, etc.
│   ├── config.rs                 # AnticiaptionConfig (frozen)
│   ├── kernel.rs                 # Main anticipation kernel
│   ├── features/
│   │   ├── mod.rs
│   │   ├── kinematics.rs         # Skeleton-based features
│   │   ├── latent_dynamics.rs    # LIM-RPS-based features
│   │   └── coordination.rs       # Cross-limb coherence
│   ├── embedding/
│   │   ├── mod.rs
│   │   ├── projection.rs         # Fixed random projection (v0)
│   │   └── encoder.rs            # Learned encoder (v1, stub)
│   ├── constraints/
│   │   ├── mod.rs
│   │   ├── balance.rs            # COM/support polygon
│   │   └── limits.rs             # Joint range, speed saturation
│   ├── scalars/
│   │   ├── mod.rs
│   │   ├── commitment.rs
│   │   ├── uncertainty.rs
│   │   ├── transition_pressure.rs
│   │   ├── recovery_margin.rs
│   │   ├── phase_stiffness.rs
│   │   ├── novelty.rs
│   │   └── stability.rs
│   ├── neighbors/
│   │   ├── mod.rs
│   │   └── dispersion.rs         # Continuation dispersion from HNSW
│   └── replay/
│       ├── mod.rs
│       └── harness.rs            # Deterministic replay validation
├── bindings/
│   └── python.rs                 # PyO3 bindings
├── tests/
│   ├── determinism.rs
│   ├── behavioral.rs
│   └── integration.rs
├── benches/
│   └── kernel_bench.rs
└── docs/
    ├── PROJECT_CHARTER.md
    ├── GLOSSARY.md
    ├── INVARIANTS.md
    ├── IMPLEMENTATION_GUIDE.md
    ├── CHECKLIST.md
    └── CONTINUATION.md
```

---

## 2. Core Types (types.rs)

```rust
//! Core types for cc-anticipation
//!
//! These types are FROZEN. Changes require schema version bump.

use std::collections::HashMap;

/// Schema version for AnticipationPacket
pub const SCHEMA_VERSION: &str = "0.1.0";

/// Quaternion (w, x, y, z) - unit normalized
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Quat {
    pub w: f32,
    pub x: f32,
    pub y: f32,
    pub z: f32,
}

/// 3D position
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Vec3 {
    pub x: f32,
    pub y: f32,
    pub z: f32,
}

/// Single skeleton frame (27 bones for mocopi)
#[derive(Clone, Debug)]
pub struct SkeletonFrame {
    pub timestamp: f64,
    pub root_position: Vec3,
    pub root_rotation: Quat,
    pub bone_rotations: [Quat; 27],  // Fixed size for mocopi
    pub valid: bool,
    pub source_seq: Option<u64>,
}

/// Single latent frame (LIM-RPS)
#[derive(Clone, Debug)]
pub struct LatentFrame {
    pub timestamp: f64,
    pub z: Vec<f32>,           // Latent vector (e.g., 25D)
    pub dz_dt: Option<Vec<f32>>,  // Optional pre-computed derivative
    pub valid: bool,
}

/// The canonical motion window - INPUT CONTRACT
#[derive(Clone, Debug)]
pub struct MotionWindow {
    pub window_id: String,
    pub t_start: f64,
    pub t_end: f64,
    pub fps: f32,
    pub skeleton_frames: Vec<SkeletonFrame>,
    pub latent_frames: Vec<LatentFrame>,
    pub coverage: f32,
    pub device_offsets: HashMap<String, f64>,
    pub dropped_reason: Option<String>,
}

/// The anticipation packet - OUTPUT CONTRACT
#[derive(Clone, Debug)]
pub struct AnticipationPacket {
    // Scalars (all 0-1 except transition_pressure which can be negative)
    pub commitment: f32,
    pub uncertainty: f32,
    pub transition_pressure: f32,
    pub recovery_margin: f32,
    pub phase_stiffness: f32,
    pub novelty: f32,
    pub stability: f32,

    // Vectors
    pub regime_embedding: Vec<f32>,     // 64-256 dims
    pub constraint_vector: Vec<f32>,    // ~8 dims
    pub derivative_summary: Vec<f32>,   // ~8 dims

    // Debug/Telemetry
    pub debug: Option<DebugTrace>,

    // Provenance
    pub window_id: String,
    pub timestamp: f64,
    pub schema_version: String,
}

#[derive(Clone, Debug)]
pub struct DebugTrace {
    pub raw_features: HashMap<String, f32>,
    pub neighbor_ids: Option<Vec<String>>,
    pub intermediate_scores: HashMap<String, f32>,
}
```

---

## 3. Configuration (config.rs)

```rust
//! Frozen configuration for cc-anticipation kernel

/// Configuration for the anticipation kernel
///
/// This config is FROZEN per version. Changes require version bump.
#[derive(Clone, Debug)]
pub struct AnticipationConfig {
    pub version: String,

    // Window parameters (must match aligner)
    pub fps: f32,                      // 50.0
    pub window_seconds: f32,           // 1.0 (50 frames)

    // Embedding
    pub regime_embedding_dim: usize,   // 64
    pub constraint_vector_dim: usize,  // 8
    pub derivative_summary_dim: usize, // 8

    // Neighbor search
    pub neighbor_k: usize,             // 10
    pub min_similarity: f32,           // 0.3

    // Thresholds
    pub min_coverage: f32,             // 0.90
    pub stillness_threshold: f32,      // 0.01 (m/s kinetic energy)

    // Smoothing
    pub transition_pressure_alpha: f32, // 0.3 (EMA coefficient)
    pub novelty_history_seconds: f32,   // 10.0

    // Debug
    pub emit_debug: bool,
}

impl Default for AnticipationConfig {
    fn default() -> Self {
        Self {
            version: "0.1.0".to_string(),
            fps: 50.0,
            window_seconds: 1.0,
            regime_embedding_dim: 64,
            constraint_vector_dim: 8,
            derivative_summary_dim: 8,
            neighbor_k: 10,
            min_similarity: 0.3,
            min_coverage: 0.90,
            stillness_threshold: 0.01,
            transition_pressure_alpha: 0.3,
            novelty_history_seconds: 10.0,
            emit_debug: true,
        }
    }
}
```

---

## 4. Kernel Implementation (kernel.rs)

```rust
//! Main anticipation kernel
//!
//! Converts MotionWindow → AnticipationPacket deterministically.

use crate::types::*;
use crate::config::AnticipationConfig;
use crate::features::{compute_kinematic_features, compute_latent_features};
use crate::embedding::project_to_regime_embedding;
use crate::constraints::compute_constraint_vector;
use crate::scalars::*;

/// The anticipation kernel - main entry point
pub struct AnticipationKernel {
    config: AnticipationConfig,

    // State for temporal computations
    prev_commitment: f32,
    prev_uncertainty: f32,
    prev_timestamp: Option<f64>,

    // Novelty history ring buffer
    regime_history: Vec<Vec<f32>>,
    regime_history_idx: usize,
    regime_history_len: usize,

    // Pre-allocated buffers (no allocation in hot path)
    feature_buffer: Vec<f32>,
    embedding_buffer: Vec<f32>,
}

impl AnticipationKernel {
    pub fn new(config: AnticipationConfig) -> Self {
        let history_len = (config.novelty_history_seconds * config.fps) as usize;

        Self {
            config: config.clone(),
            prev_commitment: 0.5,
            prev_uncertainty: 0.5,
            prev_timestamp: None,
            regime_history: vec![vec![0.0; config.regime_embedding_dim]; history_len],
            regime_history_idx: 0,
            regime_history_len: history_len,
            feature_buffer: vec![0.0; 128],  // Pre-allocate
            embedding_buffer: vec![0.0; config.regime_embedding_dim],
        }
    }

    /// Process a MotionWindow and emit an AnticipationPacket
    ///
    /// INVARIANT: Deterministic given same input
    /// INVARIANT: No heap allocation
    pub fn process(&mut self, window: &MotionWindow) -> Result<AnticipationPacket, String> {
        // Check coverage threshold
        if window.coverage < self.config.min_coverage {
            return Err(format!(
                "Coverage {} < min {}",
                window.coverage, self.config.min_coverage
            ));
        }

        // 1. Compute kinematic features from skeleton
        let kin_features = if !window.skeleton_frames.is_empty() {
            Some(compute_kinematic_features(&window.skeleton_frames, &self.config))
        } else {
            None
        };

        // 2. Compute latent dynamics from LIM-RPS
        let lat_features = if !window.latent_frames.is_empty() {
            Some(compute_latent_features(&window.latent_frames, &self.config))
        } else {
            None
        };

        // Must have at least one source
        if kin_features.is_none() && lat_features.is_none() {
            return Err("No skeleton or latent frames available".to_string());
        }

        // 3. Fuse into feature vector
        let fused_features = self.fuse_features(kin_features, lat_features);

        // 4. Project to regime embedding
        let regime_embedding = project_to_regime_embedding(
            &fused_features,
            self.config.regime_embedding_dim,
        );

        // 5. Compute constraint vector
        let constraint_vector = if let Some(ref kf) = kin_features {
            compute_constraint_vector(kf, &self.config)
        } else {
            vec![0.0; self.config.constraint_vector_dim]
        };

        // 6. Compute derivative summary
        let derivative_summary = self.compute_derivative_summary(&fused_features);

        // 7. Compute scalars
        let uncertainty = compute_uncertainty(
            &regime_embedding,
            None, // TODO: neighbor search in v1
            &self.config,
        );

        let commitment = compute_commitment(
            uncertainty,
            &constraint_vector,
            &derivative_summary,
            &self.config,
        );

        let dt = self.prev_timestamp
            .map(|prev| window.t_end - prev)
            .unwrap_or(1.0 / self.config.fps as f64) as f32;

        let transition_pressure = compute_transition_pressure(
            commitment,
            self.prev_commitment,
            uncertainty,
            self.prev_uncertainty,
            dt,
            self.config.transition_pressure_alpha,
        );

        let recovery_margin = compute_recovery_margin(&constraint_vector);

        let phase_stiffness = if let Some(ref kf) = kin_features {
            compute_phase_stiffness(kf)
        } else {
            0.5 // Neutral if no skeleton
        };

        let novelty = self.compute_novelty(&regime_embedding);

        let stability = if let Some(ref lf) = lat_features {
            compute_stability(lf)
        } else if let Some(ref kf) = kin_features {
            compute_stability_from_kinematics(kf)
        } else {
            0.5
        };

        // 8. Update state for next iteration
        self.prev_commitment = commitment;
        self.prev_uncertainty = uncertainty;
        self.prev_timestamp = Some(window.t_end);
        self.update_novelty_history(&regime_embedding);

        // 9. Build packet
        let debug = if self.config.emit_debug {
            Some(self.build_debug_trace(&fused_features, commitment, uncertainty))
        } else {
            None
        };

        Ok(AnticipationPacket {
            commitment,
            uncertainty,
            transition_pressure,
            recovery_margin,
            phase_stiffness,
            novelty,
            stability,
            regime_embedding,
            constraint_vector,
            derivative_summary,
            debug,
            window_id: window.window_id.clone(),
            timestamp: window.t_end,
            schema_version: SCHEMA_VERSION.to_string(),
        })
    }

    fn fuse_features(
        &self,
        kin: Option<KinematicFeatures>,
        lat: Option<LatentFeatures>,
    ) -> Vec<f32> {
        // Simple concatenation for v0
        let mut features = Vec::with_capacity(64);

        if let Some(k) = kin {
            features.extend(k.to_vec());
        }
        if let Some(l) = lat {
            features.extend(l.to_vec());
        }

        // Pad to fixed size
        features.resize(64, 0.0);
        features
    }

    fn compute_derivative_summary(&self, features: &[f32]) -> Vec<f32> {
        // Extract: kinetic_intensity, angular_intensity, jerk_energy,
        // cross_limb_coherence, and 4 more dimensions
        // For v0, just take first 8 features or pad
        let mut summary = vec![0.0; self.config.derivative_summary_dim];
        for (i, &f) in features.iter().take(self.config.derivative_summary_dim).enumerate() {
            summary[i] = f;
        }
        summary
    }

    fn compute_novelty(&self, embedding: &[f32]) -> f32 {
        // Distance from centroid of recent history
        let mut centroid = vec![0.0f32; embedding.len()];
        let mut count = 0usize;

        for hist in &self.regime_history {
            if hist.iter().any(|&x| x != 0.0) {
                for (i, &v) in hist.iter().enumerate() {
                    centroid[i] += v;
                }
                count += 1;
            }
        }

        if count == 0 {
            return 0.5; // Neutral if no history
        }

        for c in &mut centroid {
            *c /= count as f32;
        }

        // L2 distance normalized
        let dist: f32 = embedding.iter()
            .zip(centroid.iter())
            .map(|(&e, &c)| (e - c).powi(2))
            .sum::<f32>()
            .sqrt();

        // Normalize to 0-1 (assuming embedding is roughly unit normalized)
        (dist / 2.0).min(1.0).max(0.0)
    }

    fn update_novelty_history(&mut self, embedding: &[f32]) {
        for (i, &v) in embedding.iter().enumerate() {
            self.regime_history[self.regime_history_idx][i] = v;
        }
        self.regime_history_idx = (self.regime_history_idx + 1) % self.regime_history_len;
    }

    fn build_debug_trace(
        &self,
        features: &[f32],
        commitment: f32,
        uncertainty: f32,
    ) -> DebugTrace {
        let mut raw_features = std::collections::HashMap::new();
        for (i, &f) in features.iter().enumerate() {
            raw_features.insert(format!("f_{}", i), f);
        }

        let mut intermediate_scores = std::collections::HashMap::new();
        intermediate_scores.insert("commitment".to_string(), commitment);
        intermediate_scores.insert("uncertainty".to_string(), uncertainty);

        DebugTrace {
            raw_features,
            neighbor_ids: None,
            intermediate_scores,
        }
    }
}
```

---

## 5. Feature Computation (features/kinematics.rs)

```rust
//! Kinematic features from skeleton frames
//!
//! FROZEN: Feature computation logic must be deterministic.

use crate::types::*;
use crate::config::AnticipationConfig;

/// Kinematic features extracted from skeleton window
#[derive(Clone, Debug)]
pub struct KinematicFeatures {
    pub kinetic_intensity: f32,      // Total kinetic energy proxy
    pub angular_intensity: f32,       // Rotational energy
    pub jerk_energy: f32,             // Rate of acceleration change
    pub directional_persistence: f32, // How consistent is velocity direction
    pub cross_limb_coherence: f32,    // Left-right coordination
    pub torso_lead: f32,              // Does torso lead limbs?
    pub head_predict: f32,            // Does head predict torso turns?
    pub balance_margin: f32,          // COM to support distance
}

impl KinematicFeatures {
    pub fn to_vec(&self) -> Vec<f32> {
        vec![
            self.kinetic_intensity,
            self.angular_intensity,
            self.jerk_energy,
            self.directional_persistence,
            self.cross_limb_coherence,
            self.torso_lead,
            self.head_predict,
            self.balance_margin,
        ]
    }
}

/// Bone indices for mocopi 27-bone skeleton
pub mod bones {
    pub const HIPS: usize = 0;
    pub const SPINE: usize = 1;
    pub const CHEST: usize = 2;
    pub const HEAD: usize = 5;
    pub const LEFT_SHOULDER: usize = 11;
    pub const LEFT_ELBOW: usize = 12;
    pub const LEFT_WRIST: usize = 13;
    pub const RIGHT_SHOULDER: usize = 14;
    pub const RIGHT_ELBOW: usize = 15;
    pub const RIGHT_WRIST: usize = 16;
    pub const LEFT_HIP: usize = 17;
    pub const LEFT_KNEE: usize = 18;
    pub const LEFT_ANKLE: usize = 19;
    pub const RIGHT_HIP: usize = 20;
    pub const RIGHT_KNEE: usize = 21;
    pub const RIGHT_ANKLE: usize = 22;
}

/// Compute kinematic features from skeleton frames
///
/// INVARIANT: Deterministic given same input
pub fn compute_kinematic_features(
    frames: &[SkeletonFrame],
    config: &AnticipationConfig,
) -> KinematicFeatures {
    if frames.len() < 3 {
        return KinematicFeatures::default();
    }

    let dt = 1.0 / config.fps;
    let n = frames.len();

    // Compute velocities and accelerations using central differences
    let mut root_velocities = Vec::with_capacity(n - 2);
    let mut root_accelerations = Vec::with_capacity(n - 4);

    for i in 1..n-1 {
        if frames[i-1].valid && frames[i+1].valid {
            let v = vec3_sub(frames[i+1].root_position, frames[i-1].root_position);
            let v = vec3_scale(v, 1.0 / (2.0 * dt));
            root_velocities.push(v);
        }
    }

    // Kinetic intensity: RMS of root velocity
    let kinetic_intensity = if !root_velocities.is_empty() {
        let sum_sq: f32 = root_velocities.iter()
            .map(|v| v.x*v.x + v.y*v.y + v.z*v.z)
            .sum();
        (sum_sq / root_velocities.len() as f32).sqrt()
    } else {
        0.0
    };

    // Angular intensity: RMS of torso angular velocity
    let angular_intensity = compute_angular_intensity(frames, config);

    // Jerk energy: RMS of acceleration derivatives
    let jerk_energy = compute_jerk_energy(&root_velocities, dt);

    // Directional persistence: autocorrelation of velocity direction
    let directional_persistence = compute_directional_persistence(&root_velocities);

    // Cross-limb coherence: correlation of left/right wrist speeds
    let cross_limb_coherence = compute_cross_limb_coherence(frames, config);

    // Torso lead: does torso velocity lead limb velocity?
    let torso_lead = 0.5; // TODO: implement

    // Head predict: does head rotation predict torso turns?
    let head_predict = 0.5; // TODO: implement

    // Balance margin: COM to support polygon distance
    let balance_margin = compute_balance_margin(frames);

    KinematicFeatures {
        kinetic_intensity,
        angular_intensity,
        jerk_energy,
        directional_persistence,
        cross_limb_coherence,
        torso_lead,
        head_predict,
        balance_margin,
    }
}

fn vec3_sub(a: Vec3, b: Vec3) -> Vec3 {
    Vec3 { x: a.x - b.x, y: a.y - b.y, z: a.z - b.z }
}

fn vec3_scale(v: Vec3, s: f32) -> Vec3 {
    Vec3 { x: v.x * s, y: v.y * s, z: v.z * s }
}

fn compute_angular_intensity(frames: &[SkeletonFrame], config: &AnticipationConfig) -> f32 {
    // Compute angular velocity from torso rotation quaternions
    // Using quaternion derivative: ω ≈ 2 * d(q)/dt * q^-1
    let dt = 1.0 / config.fps;
    let mut angular_speeds = Vec::new();

    for i in 1..frames.len() {
        if frames[i-1].valid && frames[i].valid {
            let q1 = frames[i-1].bone_rotations[bones::CHEST];
            let q2 = frames[i].bone_rotations[bones::CHEST];

            // Approximate angular speed from quaternion difference
            let dq = quat_mul(q2, quat_conj(q1));
            let angle = 2.0 * dq.w.acos().abs();
            angular_speeds.push(angle / dt);
        }
    }

    if angular_speeds.is_empty() {
        return 0.0;
    }

    let sum_sq: f32 = angular_speeds.iter().map(|&a| a * a).sum();
    (sum_sq / angular_speeds.len() as f32).sqrt()
}

fn quat_mul(a: Quat, b: Quat) -> Quat {
    Quat {
        w: a.w*b.w - a.x*b.x - a.y*b.y - a.z*b.z,
        x: a.w*b.x + a.x*b.w + a.y*b.z - a.z*b.y,
        y: a.w*b.y - a.x*b.z + a.y*b.w + a.z*b.x,
        z: a.w*b.z + a.x*b.y - a.y*b.x + a.z*b.w,
    }
}

fn quat_conj(q: Quat) -> Quat {
    Quat { w: q.w, x: -q.x, y: -q.y, z: -q.z }
}

fn compute_jerk_energy(velocities: &[Vec3], dt: f32) -> f32 {
    if velocities.len() < 3 {
        return 0.0;
    }

    let mut jerk_sq_sum = 0.0f32;
    let mut count = 0;

    for i in 2..velocities.len() {
        // Second derivative of velocity = jerk
        let a1 = vec3_sub(velocities[i-1], velocities[i-2]);
        let a2 = vec3_sub(velocities[i], velocities[i-1]);
        let j = vec3_sub(a2, a1);
        let j = vec3_scale(j, 1.0 / (dt * dt));
        jerk_sq_sum += j.x*j.x + j.y*j.y + j.z*j.z;
        count += 1;
    }

    if count == 0 {
        return 0.0;
    }

    (jerk_sq_sum / count as f32).sqrt()
}

fn compute_directional_persistence(velocities: &[Vec3]) -> f32 {
    if velocities.len() < 2 {
        return 0.5;
    }

    let mut dot_sum = 0.0f32;
    let mut count = 0;

    for i in 1..velocities.len() {
        let v1 = velocities[i-1];
        let v2 = velocities[i];

        let mag1 = (v1.x*v1.x + v1.y*v1.y + v1.z*v1.z).sqrt();
        let mag2 = (v2.x*v2.x + v2.y*v2.y + v2.z*v2.z).sqrt();

        if mag1 > 1e-6 && mag2 > 1e-6 {
            let dot = (v1.x*v2.x + v1.y*v2.y + v1.z*v2.z) / (mag1 * mag2);
            dot_sum += dot;
            count += 1;
        }
    }

    if count == 0 {
        return 0.5;
    }

    // Map from [-1, 1] to [0, 1]
    (dot_sum / count as f32 + 1.0) / 2.0
}

fn compute_cross_limb_coherence(frames: &[SkeletonFrame], config: &AnticipationConfig) -> f32 {
    // Correlation of left and right wrist speeds
    let dt = 1.0 / config.fps;
    let mut left_speeds = Vec::new();
    let mut right_speeds = Vec::new();

    // TODO: Implement FK to get wrist positions from rotations
    // For now, return neutral
    0.5
}

fn compute_balance_margin(frames: &[SkeletonFrame]) -> f32 {
    // Approximate COM to support polygon distance
    // TODO: Implement with FK for ankle positions
    // For now, return neutral
    0.5
}

impl Default for KinematicFeatures {
    fn default() -> Self {
        Self {
            kinetic_intensity: 0.0,
            angular_intensity: 0.0,
            jerk_energy: 0.0,
            directional_persistence: 0.5,
            cross_limb_coherence: 0.5,
            torso_lead: 0.5,
            head_predict: 0.5,
            balance_margin: 0.5,
        }
    }
}
```

---

## 6. Integration with cc-core-rs

### 6.1 Shared Dependencies

```toml
# cc-anticipation/Cargo.toml
[dependencies]
cc-core-rs = { path = "../cc-core-rs" }

# Use existing primitives
# - LockFreeRingBuffer for novelty history
# - OneEuroFilter for scalar smoothing
# - SlewLimiter for transition_pressure
```

### 6.2 Ring Buffer for History

```rust
use cc_core_rs::LockFreeRingBuffer;

// Novelty history as ring buffer
let mut regime_history: LockFreeRingBuffer<[f32; 64]> =
    LockFreeRingBuffer::new(500); // 10 seconds at 50Hz
```

### 6.3 Smoothing for Temporal Scalars

```rust
use cc_core_rs::OneEuroFilter;

// Smooth transition_pressure to avoid jitter
let mut pressure_filter = OneEuroFilter::new(
    50.0,  // sample rate
    1.0,   // min cutoff
    0.007, // beta
    1.0,   // d_cutoff
);

let smoothed_pressure = pressure_filter.filter(raw_pressure);
```

---

## 7. Integration with rag_plusplus

### 7.1 HNSW for Neighbor Search

```rust
use rag_plusplus::HnswIndex;

// Query MotionPhrase library for similar regimes
let neighbors = hnsw_index.search(&regime_embedding, config.neighbor_k);

// Compute continuation dispersion
let uncertainty = compute_dispersion_from_neighbors(&neighbors);
```

### 7.2 OutcomeStats for Running Statistics

```rust
use rag_plusplus::OutcomeStats;

// Track running statistics for normalization
let mut kinetic_stats = OutcomeStats::new();
kinetic_stats.update(kinetic_intensity);

// Normalize to z-score
let normalized = (kinetic_intensity - kinetic_stats.mean()) / kinetic_stats.stddev();
```

---

## 8. Python Bindings (bindings/python.rs)

```rust
use pyo3::prelude::*;

#[pyclass]
pub struct PyAnticipationKernel {
    inner: AnticipationKernel,
}

#[pymethods]
impl PyAnticipationKernel {
    #[new]
    pub fn new(config: PyAnticipationConfig) -> Self {
        Self {
            inner: AnticipationKernel::new(config.into()),
        }
    }

    pub fn process(&mut self, window: &PyMotionWindow) -> PyResult<PyAnticipationPacket> {
        self.inner.process(&window.into())
            .map(|p| p.into())
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e))
    }
}

#[pymodule]
fn cc_anticipation(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<PyAnticipationKernel>()?;
    m.add_class::<PyAnticipationPacket>()?;
    m.add_class::<PyMotionWindow>()?;
    Ok(())
}
```

---

## 9. Testing Strategy

### 9.1 Determinism Tests

```rust
#[test]
fn test_deterministic_replay() {
    let config = AnticipationConfig::default();
    let window = load_test_window("session_001.bin");

    let mut kernel1 = AnticipationKernel::new(config.clone());
    let mut kernel2 = AnticipationKernel::new(config.clone());

    let packet1 = kernel1.process(&window).unwrap();
    let packet2 = kernel2.process(&window).unwrap();

    assert_eq!(packet1.commitment, packet2.commitment);
    assert_eq!(packet1.regime_embedding, packet2.regime_embedding);
    assert_eq!(packet1.window_id, packet2.window_id);
}
```

### 9.2 Behavioral Tests

```rust
#[test]
fn test_stillness_response() {
    let still_window = create_still_window(1.0); // 1 second stillness
    let mut kernel = AnticipationKernel::new(AnticipationConfig::default());

    let packet = kernel.process(&still_window).unwrap();

    assert!(packet.commitment < 0.2, "Stillness should have low commitment");
    assert!(packet.stability > 0.8, "Stillness should have high stability");
    assert!(packet.transition_pressure.abs() < 0.05, "Stillness should have ~0 pressure");
}

#[test]
fn test_commitment_before_peak() {
    let sweep_windows = load_sweep_sequence("turn_sequence.bin");
    let mut kernel = AnticipationKernel::new(AnticipationConfig::default());

    let packets: Vec<_> = sweep_windows.iter()
        .map(|w| kernel.process(w).unwrap())
        .collect();

    // Find kinetic peak
    let peak_idx = packets.iter()
        .enumerate()
        .max_by(|(_, a), (_, b)|
            a.debug.as_ref().unwrap().raw_features["kinetic_intensity"]
                .partial_cmp(&b.debug.as_ref().unwrap().raw_features["kinetic_intensity"])
                .unwrap()
        )
        .unwrap().0;

    // Commitment should rise BEFORE peak
    let pre_peak_commitment = packets[peak_idx.saturating_sub(5)].commitment;
    let at_peak_commitment = packets[peak_idx].commitment;

    assert!(pre_peak_commitment > 0.5, "Commitment should rise before peak");
}
```

---

## 10. Benchmark Strategy

```rust
use criterion::{criterion_group, criterion_main, Criterion};

fn kernel_benchmark(c: &mut Criterion) {
    let config = AnticipationConfig::default();
    let mut kernel = AnticipationKernel::new(config);
    let window = load_test_window("typical_motion.bin");

    c.bench_function("kernel_process", |b| {
        b.iter(|| kernel.process(&window))
    });
}

criterion_group!(benches, kernel_benchmark);
criterion_main!(benches);
```

**Target**: < 2ms per kernel tick on M1.

---

**Next Document**: [CHECKLIST.md](./CHECKLIST.md)

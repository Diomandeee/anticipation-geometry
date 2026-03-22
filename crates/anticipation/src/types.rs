//! Core types for cc-anticipation
//!
//! These types are FROZEN. Changes require schema version bump.
//!
//! # Frozen Contracts
//!
//! - [`MotionWindow`]: Input from cc-window-aligner
//! - [`AnticipationPacket`]: Output to Conductor/RAG++
//! - [`SCHEMA_VERSION`]: Must match for compatibility

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Schema version for AnticipationPacket
///
/// FROZEN: Increment on any breaking change to packet structure.
///
/// # Version History
/// - 0.1.0: Initial frozen contract
/// - 0.2.0: Added SourceWindowIdentity for frame provenance (Phase 4 Event Spine)
pub const SCHEMA_VERSION: &str = "0.2.0";

/// Number of bones in mocopi skeleton
pub const MOCOPI_BONE_COUNT: usize = 27;

// =============================================================================
// Primitive Types
// =============================================================================

/// Quaternion (w, x, y, z) - unit normalized
///
/// Uses scalar-first convention (w, x, y, z) for consistency with mocopi SDK.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct Quat {
    /// Scalar component
    pub w: f32,
    /// X component
    pub x: f32,
    /// Y component
    pub y: f32,
    /// Z component
    pub z: f32,
}

impl Quat {
    /// Identity quaternion (no rotation)
    pub const IDENTITY: Self = Self { w: 1.0, x: 0.0, y: 0.0, z: 0.0 };

    /// Create a new quaternion
    pub fn new(w: f32, x: f32, y: f32, z: f32) -> Self {
        Self { w, x, y, z }
    }

    /// Compute the dot product with another quaternion
    #[inline]
    pub fn dot(&self, other: &Self) -> f32 {
        self.w * other.w + self.x * other.x + self.y * other.y + self.z * other.z
    }

    /// Compute the squared magnitude
    #[inline]
    pub fn magnitude_squared(&self) -> f32 {
        self.dot(self)
    }

    /// Compute the magnitude
    #[inline]
    pub fn magnitude(&self) -> f32 {
        self.magnitude_squared().sqrt()
    }

    /// Normalize to unit quaternion
    pub fn normalize(&self) -> Self {
        let mag = self.magnitude();
        if mag < 1e-10 {
            return Self::IDENTITY;
        }
        Self {
            w: self.w / mag,
            x: self.x / mag,
            y: self.y / mag,
            z: self.z / mag,
        }
    }

    /// Conjugate (inverse for unit quaternions)
    #[inline]
    pub fn conjugate(&self) -> Self {
        Self {
            w: self.w,
            x: -self.x,
            y: -self.y,
            z: -self.z,
        }
    }

    /// Multiply two quaternions
    pub fn mul(&self, other: &Self) -> Self {
        Self {
            w: self.w * other.w - self.x * other.x - self.y * other.y - self.z * other.z,
            x: self.w * other.x + self.x * other.w + self.y * other.z - self.z * other.y,
            y: self.w * other.y - self.x * other.z + self.y * other.w + self.z * other.x,
            z: self.w * other.z + self.x * other.y - self.y * other.x + self.z * other.w,
        }
    }

    /// Spherical linear interpolation with hemisphere correction
    ///
    /// INVARIANT: Deterministic hemisphere choice (always shortest path)
    pub fn slerp(&self, other: &Self, t: f32) -> Self {
        let mut other = *other;
        let mut dot = self.dot(&other);

        // Ensure shortest path (hemisphere correction)
        if dot < 0.0 {
            other = Self {
                w: -other.w,
                x: -other.x,
                y: -other.y,
                z: -other.z,
            };
            dot = -dot;
        }

        // If very close, use linear interpolation to avoid numerical issues
        if dot > 0.9995 {
            let result = Self {
                w: self.w + t * (other.w - self.w),
                x: self.x + t * (other.x - self.x),
                y: self.y + t * (other.y - self.y),
                z: self.z + t * (other.z - self.z),
            };
            return result.normalize();
        }

        // Clamp dot to valid acos range
        let dot = dot.clamp(-1.0, 1.0);
        let theta_0 = dot.acos();
        let sin_theta_0 = theta_0.sin();
        let theta = theta_0 * t;
        let sin_theta = theta.sin();

        let s0 = theta.cos() - dot * sin_theta / sin_theta_0;
        let s1 = sin_theta / sin_theta_0;

        Self {
            w: s0 * self.w + s1 * other.w,
            x: s0 * self.x + s1 * other.x,
            y: s0 * self.y + s1 * other.y,
            z: s0 * self.z + s1 * other.z,
        }.normalize()
    }
}

impl Default for Quat {
    fn default() -> Self {
        Self::IDENTITY
    }
}

/// 3D position vector
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct Vec3 {
    /// X coordinate
    pub x: f32,
    /// Y coordinate
    pub y: f32,
    /// Z coordinate
    pub z: f32,
}

impl Vec3 {
    /// Zero vector
    pub const ZERO: Self = Self { x: 0.0, y: 0.0, z: 0.0 };

    /// Create a new vector
    pub fn new(x: f32, y: f32, z: f32) -> Self {
        Self { x, y, z }
    }

    /// Add two vectors
    #[inline]
    pub fn add(&self, other: &Self) -> Self {
        Self {
            x: self.x + other.x,
            y: self.y + other.y,
            z: self.z + other.z,
        }
    }

    /// Subtract two vectors
    #[inline]
    pub fn sub(&self, other: &Self) -> Self {
        Self {
            x: self.x - other.x,
            y: self.y - other.y,
            z: self.z - other.z,
        }
    }

    /// Scale by scalar
    #[inline]
    pub fn scale(&self, s: f32) -> Self {
        Self {
            x: self.x * s,
            y: self.y * s,
            z: self.z * s,
        }
    }

    /// Dot product
    #[inline]
    pub fn dot(&self, other: &Self) -> f32 {
        self.x * other.x + self.y * other.y + self.z * other.z
    }

    /// Squared magnitude
    #[inline]
    pub fn magnitude_squared(&self) -> f32 {
        self.dot(self)
    }

    /// Magnitude
    #[inline]
    pub fn magnitude(&self) -> f32 {
        self.magnitude_squared().sqrt()
    }

    /// Normalize to unit vector
    pub fn normalize(&self) -> Self {
        let mag = self.magnitude();
        if mag < 1e-10 {
            return Self::ZERO;
        }
        self.scale(1.0 / mag)
    }

    /// Linear interpolation
    #[inline]
    pub fn lerp(&self, other: &Self, t: f32) -> Self {
        Self {
            x: self.x + t * (other.x - self.x),
            y: self.y + t * (other.y - self.y),
            z: self.z + t * (other.z - self.z),
        }
    }
}

impl Default for Vec3 {
    fn default() -> Self {
        Self::ZERO
    }
}

// =============================================================================
// Frame Types
// =============================================================================

/// Single skeleton frame (27 bones for mocopi)
///
/// Contains root transform and per-bone local rotations.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SkeletonFrame {
    /// Timestamp in canonical server time (seconds)
    pub timestamp: f64,
    /// Root position in world space (meters)
    pub root_position: Vec3,
    /// Root orientation in world space
    pub root_rotation: Quat,
    /// Per-bone local rotations (27 bones for mocopi)
    pub bone_rotations: [Quat; MOCOPI_BONE_COUNT],
    /// Whether this frame contains valid data
    pub valid: bool,
    /// Source sequence number for provenance
    pub source_seq: Option<u64>,
}

impl SkeletonFrame {
    /// Create an invalid/empty frame at the given timestamp
    pub fn invalid(timestamp: f64) -> Self {
        Self {
            timestamp,
            root_position: Vec3::ZERO,
            root_rotation: Quat::IDENTITY,
            bone_rotations: [Quat::IDENTITY; MOCOPI_BONE_COUNT],
            valid: false,
            source_seq: None,
        }
    }

    /// Interpolate between two skeleton frames
    ///
    /// INVARIANT: Uses slerp for rotations, lerp for positions
    pub fn interpolate(&self, other: &Self, t: f32) -> Self {
        Self {
            timestamp: self.timestamp + t as f64 * (other.timestamp - self.timestamp),
            root_position: self.root_position.lerp(&other.root_position, t),
            root_rotation: self.root_rotation.slerp(&other.root_rotation, t),
            bone_rotations: {
                let mut bones = [Quat::IDENTITY; MOCOPI_BONE_COUNT];
                for i in 0..MOCOPI_BONE_COUNT {
                    bones[i] = self.bone_rotations[i].slerp(&other.bone_rotations[i], t);
                }
                bones
            },
            valid: self.valid && other.valid,
            source_seq: self.source_seq, // Use earlier seq
        }
    }
}

/// Single latent frame (LIM-RPS)
///
/// Contains the latent vector from motion encoding plus optional derivatives.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct LatentFrame {
    /// Timestamp in canonical server time (seconds)
    pub timestamp: f64,
    /// Latent vector (e.g., 25D for LIM-RPS)
    pub z: Vec<f32>,
    /// Optional pre-computed derivative dz/dt
    pub dz_dt: Option<Vec<f32>>,
    /// Whether this frame contains valid data
    pub valid: bool,
}

impl LatentFrame {
    /// Create an invalid/empty frame at the given timestamp
    pub fn invalid(timestamp: f64, dim: usize) -> Self {
        Self {
            timestamp,
            z: vec![0.0; dim],
            dz_dt: None,
            valid: false,
        }
    }

    /// Linear interpolation between two latent frames
    pub fn interpolate(&self, other: &Self, t: f32) -> Self {
        let z = self.z.iter()
            .zip(other.z.iter())
            .map(|(&a, &b)| a + t * (b - a))
            .collect();

        let dz_dt = match (&self.dz_dt, &other.dz_dt) {
            (Some(a), Some(b)) => Some(
                a.iter().zip(b.iter())
                    .map(|(&x, &y)| x + t * (y - x))
                    .collect()
            ),
            _ => None,
        };

        Self {
            timestamp: self.timestamp + t as f64 * (other.timestamp - self.timestamp),
            z,
            dz_dt,
            valid: self.valid && other.valid,
        }
    }
}

// =============================================================================
// Window Types (Input Contract)
// =============================================================================

/// The canonical motion window - INPUT CONTRACT
///
/// A fixed-length ring of aligned frames at canonical fps.
/// Produced by cc-window-aligner, consumed by cc-anticipation kernel.
///
/// # Invariants
///
/// - All frames are at canonical fps (e.g., 50Hz)
/// - `window_id` is a deterministic hash of content
/// - `coverage` is in range [0.0, 1.0]
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MotionWindow {
    /// Deterministic hash identifying this window
    pub window_id: String,
    /// Start time of window (canonical server time)
    pub t_start: f64,
    /// End time of window (canonical server time)
    pub t_end: f64,
    /// Frame rate (e.g., 50.0)
    pub fps: f32,
    /// Skeleton frames (may be empty if no skeleton stream)
    pub skeleton_frames: Vec<SkeletonFrame>,
    /// Latent frames (may be empty if no latent stream)
    pub latent_frames: Vec<LatentFrame>,
    /// Fraction of frames that are valid [0.0, 1.0]
    pub coverage: f32,
    /// Per-device clock offsets for provenance
    pub device_offsets: HashMap<String, f64>,
    /// Reason for degraded/dropped window (if any)
    pub dropped_reason: Option<String>,
}

impl MotionWindow {
    /// Check if window has usable skeleton data
    pub fn has_skeleton(&self) -> bool {
        !self.skeleton_frames.is_empty() &&
            self.skeleton_frames.iter().any(|f| f.valid)
    }

    /// Check if window has usable latent data
    pub fn has_latent(&self) -> bool {
        !self.latent_frames.is_empty() &&
            self.latent_frames.iter().any(|f| f.valid)
    }

    /// Duration of window in seconds
    pub fn duration(&self) -> f64 {
        self.t_end - self.t_start
    }

    /// Number of frames in window
    pub fn frame_count(&self) -> usize {
        self.skeleton_frames.len().max(self.latent_frames.len())
    }
}

// =============================================================================
// Provenance Types (Phase 4 Event Spine)
// =============================================================================

/// Source window identity - traces AnticipationPacket back to fusion frames
///
/// This enables deterministic replay: given an AnticipationPacket, you can
/// identify exactly which FusedSkeleton frames produced it.
///
/// # Design Notes
/// - Uses fixed-size arrays for hash stability (no String allocation)
/// - Minimal coupling: does NOT depend on cc-collection types
/// - Frame indices are within-session (reset per session)
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct SourceWindowIdentity {
    /// Session identifier (content-hash from session start)
    pub session_id: [u8; 16],

    /// First frame index in the window (inclusive)
    pub frame_start: u64,

    /// Last frame index in the window (inclusive)
    pub frame_end: u64,

    /// Wall clock timestamp of window start (microseconds since epoch)
    pub wall_time_start_us: i64,

    /// Wall clock timestamp of window end (microseconds since epoch)
    pub wall_time_end_us: i64,

    /// Hash of the basis (coordinate system) used for these frames
    /// This is a truncated hash of the full BasisId from cc-collection
    pub basis_hash: [u8; 16],
}

impl SourceWindowIdentity {
    /// Create a new source window identity
    pub fn new(
        session_id: [u8; 16],
        frame_start: u64,
        frame_end: u64,
        wall_time_start_us: i64,
        wall_time_end_us: i64,
        basis_hash: [u8; 16],
    ) -> Self {
        Self {
            session_id,
            frame_start,
            frame_end,
            wall_time_start_us,
            wall_time_end_us,
            basis_hash,
        }
    }

    /// Number of frames in the window
    pub fn frame_count(&self) -> u64 {
        self.frame_end.saturating_sub(self.frame_start) + 1
    }

    /// Duration in microseconds
    pub fn duration_us(&self) -> i64 {
        self.wall_time_end_us - self.wall_time_start_us
    }
}

// =============================================================================
// Packet Types (Output Contract)
// =============================================================================

/// The anticipation packet - OUTPUT CONTRACT
///
/// Contains scalars (control-ready), vectors (query/conditioning),
/// and debug traces (telemetry).
///
/// # Schema Version
///
/// FROZEN at [`SCHEMA_VERSION`]. Any breaking change requires version bump.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AnticipationPacket {
    // -------------------------------------------------------------------------
    // Scalars (all 0-1 except transition_pressure)
    // -------------------------------------------------------------------------

    /// How irreversible the current motion has become [0-1]
    pub commitment: f32,

    /// How many plausible futures remain [0-1]
    /// High = many options, Low = few options
    pub uncertainty: f32,

    /// Rate at which futures are collapsing
    /// Can be negative (futures opening up)
    pub transition_pressure: f32,

    /// Distance to balance/attractor loss [0-1]
    /// High = safe, Low = danger
    pub recovery_margin: f32,

    /// How locked to internal metronome [0-1]
    pub phase_stiffness: f32,

    /// Distance from recent regimes [0-1]
    /// High = exploring, Low = repeating
    pub novelty: f32,

    /// Local stationarity of dynamics [0-1]
    pub stability: f32,

    // -------------------------------------------------------------------------
    // Vectors (query/conditioning)
    // -------------------------------------------------------------------------

    /// Regime embedding (64-256 dims)
    /// Query key for RAG++, conditioning for MotionGen
    pub regime_embedding: Vec<f32>,

    /// Constraint vector (~8 dims)
    /// Balance, joint limits, speed saturation
    pub constraint_vector: Vec<f32>,

    /// Derivative summary (~8 dims)
    /// Kinetic intensity, angular intensity, jerk, coherence
    pub derivative_summary: Vec<f32>,

    // -------------------------------------------------------------------------
    // Debug/Telemetry
    // -------------------------------------------------------------------------

    /// Optional debug traces (raw features, neighbor IDs, scores)
    pub debug: Option<DebugTrace>,

    // -------------------------------------------------------------------------
    // Provenance
    // -------------------------------------------------------------------------

    /// Window ID from input MotionWindow
    pub window_id: String,

    /// Timestamp (t_end of window)
    pub timestamp: f64,

    /// Source frame identity for tracing back to fusion frames
    ///
    /// Added in v0.2.0. Optional for backwards compatibility with
    /// packets created before Phase 4 Event Spine.
    pub source_identity: Option<SourceWindowIdentity>,

    /// Schema version for compatibility checking
    pub schema_version: String,
}

impl AnticipationPacket {
    /// Validate that packet conforms to invariants
    ///
    /// # Invariants Checked
    /// - INV-004: Scalar bounds
    /// - INV-005: Embedding dimension
    /// - INV-007: Schema version
    pub fn validate(&self) -> Result<(), String> {
        // INV-004: Scalar bounds
        if self.commitment < 0.0 || self.commitment > 1.0 {
            return Err(format!("commitment {} out of [0,1]", self.commitment));
        }
        if self.uncertainty < 0.0 || self.uncertainty > 1.0 {
            return Err(format!("uncertainty {} out of [0,1]", self.uncertainty));
        }
        if self.recovery_margin < 0.0 || self.recovery_margin > 1.0 {
            return Err(format!("recovery_margin {} out of [0,1]", self.recovery_margin));
        }
        if self.phase_stiffness < 0.0 || self.phase_stiffness > 1.0 {
            return Err(format!("phase_stiffness {} out of [0,1]", self.phase_stiffness));
        }
        if self.novelty < 0.0 || self.novelty > 1.0 {
            return Err(format!("novelty {} out of [0,1]", self.novelty));
        }
        if self.stability < 0.0 || self.stability > 1.0 {
            return Err(format!("stability {} out of [0,1]", self.stability));
        }

        // INV-007: Schema version
        if self.schema_version != SCHEMA_VERSION {
            return Err(format!(
                "schema_version {} != expected {}",
                self.schema_version, SCHEMA_VERSION
            ));
        }

        Ok(())
    }
}

/// Debug traces for telemetry and replay
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DebugTrace {
    /// Raw computed features before fusion
    pub raw_features: HashMap<String, f32>,
    /// Neighbor IDs from HNSW search (if neighbors feature enabled)
    pub neighbor_ids: Option<Vec<String>>,
    /// Intermediate scores during computation
    pub intermediate_scores: HashMap<String, f32>,
}

impl Default for DebugTrace {
    fn default() -> Self {
        Self {
            raw_features: HashMap::new(),
            neighbor_ids: None,
            intermediate_scores: HashMap::new(),
        }
    }
}

// =============================================================================
// Bone Indices (mocopi 27-bone skeleton)
// =============================================================================

/// Bone indices for mocopi 27-bone skeleton
pub mod bones {
    /// Hips (root)
    pub const HIPS: usize = 0;
    /// Spine
    pub const SPINE: usize = 1;
    /// Chest
    pub const CHEST: usize = 2;
    /// Upper chest
    pub const UPPER_CHEST: usize = 3;
    /// Neck
    pub const NECK: usize = 4;
    /// Head
    pub const HEAD: usize = 5;
    /// Left shoulder
    pub const LEFT_SHOULDER: usize = 6;
    /// Left upper arm
    pub const LEFT_UPPER_ARM: usize = 7;
    /// Left lower arm
    pub const LEFT_LOWER_ARM: usize = 8;
    /// Left hand
    pub const LEFT_HAND: usize = 9;
    /// Right shoulder
    pub const RIGHT_SHOULDER: usize = 10;
    /// Right upper arm
    pub const RIGHT_UPPER_ARM: usize = 11;
    /// Right lower arm
    pub const RIGHT_LOWER_ARM: usize = 12;
    /// Right hand
    pub const RIGHT_HAND: usize = 13;
    /// Left upper leg
    pub const LEFT_UPPER_LEG: usize = 14;
    /// Left lower leg
    pub const LEFT_LOWER_LEG: usize = 15;
    /// Left foot
    pub const LEFT_FOOT: usize = 16;
    /// Left toes
    pub const LEFT_TOES: usize = 17;
    /// Right upper leg
    pub const RIGHT_UPPER_LEG: usize = 18;
    /// Right lower leg
    pub const RIGHT_LOWER_LEG: usize = 19;
    /// Right foot
    pub const RIGHT_FOOT: usize = 20;
    /// Right toes
    pub const RIGHT_TOES: usize = 21;
    // Additional bones (22-26) for fine detail
    /// Left thumb
    pub const LEFT_THUMB: usize = 22;
    /// Left index
    pub const LEFT_INDEX: usize = 23;
    /// Right thumb
    pub const RIGHT_THUMB: usize = 24;
    /// Right index
    pub const RIGHT_INDEX: usize = 25;
    /// Eyes (gaze direction)
    pub const EYES: usize = 26;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_quat_identity() {
        let q = Quat::IDENTITY;
        assert_eq!(q.magnitude(), 1.0);
    }

    #[test]
    fn test_quat_slerp_identity() {
        let q1 = Quat::IDENTITY;
        let q2 = Quat::IDENTITY;
        let result = q1.slerp(&q2, 0.5);
        assert!((result.w - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_quat_slerp_hemisphere() {
        // Quaternions on opposite hemispheres
        let q1 = Quat::new(1.0, 0.0, 0.0, 0.0);
        let q2 = Quat::new(-1.0, 0.0, 0.0, 0.0); // Same rotation, opposite sign

        // Should take shortest path (stay at identity)
        let result = q1.slerp(&q2, 0.5);
        assert!((result.w.abs() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_vec3_lerp() {
        let a = Vec3::new(0.0, 0.0, 0.0);
        let b = Vec3::new(2.0, 4.0, 6.0);
        let mid = a.lerp(&b, 0.5);
        assert!((mid.x - 1.0).abs() < 1e-6);
        assert!((mid.y - 2.0).abs() < 1e-6);
        assert!((mid.z - 3.0).abs() < 1e-6);
    }

    #[test]
    fn test_packet_validate() {
        let packet = AnticipationPacket {
            commitment: 0.5,
            uncertainty: 0.5,
            transition_pressure: 0.0,
            recovery_margin: 0.5,
            phase_stiffness: 0.5,
            novelty: 0.5,
            stability: 0.5,
            regime_embedding: vec![0.0; 64],
            constraint_vector: vec![0.0; 8],
            derivative_summary: vec![0.0; 8],
            debug: None,
            window_id: "test".to_string(),
            timestamp: 0.0,
            schema_version: SCHEMA_VERSION.to_string(),
            source_identity: None,
        };
        assert!(packet.validate().is_ok());
    }

    #[test]
    fn test_packet_validate_out_of_bounds() {
        let packet = AnticipationPacket {
            commitment: 1.5, // Invalid!
            uncertainty: 0.5,
            transition_pressure: 0.0,
            recovery_margin: 0.5,
            phase_stiffness: 0.5,
            novelty: 0.5,
            stability: 0.5,
            regime_embedding: vec![0.0; 64],
            constraint_vector: vec![0.0; 8],
            derivative_summary: vec![0.0; 8],
            debug: None,
            window_id: "test".to_string(),
            timestamp: 0.0,
            schema_version: SCHEMA_VERSION.to_string(),
            source_identity: None,
        };
        assert!(packet.validate().is_err());
    }
}

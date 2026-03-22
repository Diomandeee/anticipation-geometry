//! Python bindings for cc-anticipation
//!
//! Provides PyO3 bindings for the anticipation kernel and types.
//!
//! # Usage from Python
//!
//! ```python
//! from cc_anticipation import AnticipationKernel, AnticipationConfig, MotionWindow
//!
//! config = AnticipationConfig()
//! kernel = AnticipationKernel(config)
//!
//! # Create motion window from your data
//! window = MotionWindow(
//!     window_id="window_001",
//!     t_start=0.0,
//!     t_end=1.0,
//!     fps=50.0,
//!     skeleton_frames=frames,  # List of dicts
//!     coverage=0.98,
//! )
//!
//! # Process window
//! packet = kernel.process(window)
//! print(f"Commitment: {packet.commitment}")
//! print(f"Uncertainty: {packet.uncertainty}")
//! ```

use pyo3::prelude::*;
use pyo3::exceptions::PyValueError;
use std::collections::HashMap;

use crate::config::AnticipationConfig as RustConfig;
use crate::kernel::{AnticipationKernel as RustKernel, KernelError};
use crate::types::{
    AnticipationPacket as RustPacket,
    MotionWindow as RustWindow,
    SkeletonFrame as RustSkeletonFrame,
    LatentFrame as RustLatentFrame,
    DebugTrace as RustDebugTrace,
    Quat, Vec3, MOCOPI_BONE_COUNT,
};

// =============================================================================
// Configuration
// =============================================================================

/// Python wrapper for AnticipationConfig
#[pyclass(name = "AnticipationConfig")]
#[derive(Clone)]
pub struct PyAnticipationConfig {
    inner: RustConfig,
}

#[pymethods]
impl PyAnticipationConfig {
    #[new]
    #[pyo3(signature = (
        fps = 50.0,
        window_seconds = 1.0,
        regime_embedding_dim = 64,
        min_coverage = 0.90,
        emit_debug = true
    ))]
    fn new(
        fps: f32,
        window_seconds: f32,
        regime_embedding_dim: usize,
        min_coverage: f32,
        emit_debug: bool,
    ) -> Self {
        Self {
            inner: RustConfig {
                fps,
                window_seconds,
                regime_embedding_dim,
                min_coverage,
                emit_debug,
                ..Default::default()
            },
        }
    }

    /// Create default configuration
    #[staticmethod]
    fn default() -> Self {
        Self {
            inner: RustConfig::default(),
        }
    }

    /// Create low-latency configuration
    #[staticmethod]
    fn low_latency() -> Self {
        Self {
            inner: RustConfig::low_latency(),
        }
    }

    /// Create analysis configuration
    #[staticmethod]
    fn analysis() -> Self {
        Self {
            inner: RustConfig::analysis(),
        }
    }

    #[getter]
    fn fps(&self) -> f32 {
        self.inner.fps
    }

    #[getter]
    fn window_seconds(&self) -> f32 {
        self.inner.window_seconds
    }

    #[getter]
    fn regime_embedding_dim(&self) -> usize {
        self.inner.regime_embedding_dim
    }

    #[getter]
    fn min_coverage(&self) -> f32 {
        self.inner.min_coverage
    }

    #[getter]
    fn emit_debug(&self) -> bool {
        self.inner.emit_debug
    }

    fn __repr__(&self) -> String {
        format!(
            "AnticipationConfig(fps={}, window_seconds={}, regime_embedding_dim={}, min_coverage={})",
            self.inner.fps,
            self.inner.window_seconds,
            self.inner.regime_embedding_dim,
            self.inner.min_coverage
        )
    }
}

// =============================================================================
// Skeleton Frame
// =============================================================================

/// Python wrapper for SkeletonFrame
#[pyclass(name = "SkeletonFrame")]
#[derive(Clone)]
pub struct PySkeletonFrame {
    inner: RustSkeletonFrame,
}

#[pymethods]
impl PySkeletonFrame {
    #[new]
    #[pyo3(signature = (
        timestamp,
        root_position,
        root_rotation,
        bone_rotations,
        valid = true
    ))]
    fn new(
        timestamp: f64,
        root_position: [f32; 3],
        root_rotation: [f32; 4],
        bone_rotations: Vec<[f32; 4]>,
        valid: bool,
    ) -> PyResult<Self> {
        if bone_rotations.len() != MOCOPI_BONE_COUNT {
            return Err(PyValueError::new_err(format!(
                "Expected {} bone rotations, got {}",
                MOCOPI_BONE_COUNT,
                bone_rotations.len()
            )));
        }

        let mut bones = [Quat::IDENTITY; MOCOPI_BONE_COUNT];
        for (i, &[w, x, y, z]) in bone_rotations.iter().enumerate() {
            bones[i] = Quat::new(w, x, y, z);
        }

        Ok(Self {
            inner: RustSkeletonFrame {
                timestamp,
                root_position: Vec3::new(root_position[0], root_position[1], root_position[2]),
                root_rotation: Quat::new(
                    root_rotation[0],
                    root_rotation[1],
                    root_rotation[2],
                    root_rotation[3],
                ),
                bone_rotations: bones,
                valid,
                source_seq: None,
            },
        })
    }

    /// Create an identity/still frame at the given timestamp
    #[staticmethod]
    fn identity(timestamp: f64) -> Self {
        Self {
            inner: RustSkeletonFrame {
                timestamp,
                root_position: Vec3::new(0.0, 1.0, 0.0),
                root_rotation: Quat::IDENTITY,
                bone_rotations: [Quat::IDENTITY; MOCOPI_BONE_COUNT],
                valid: true,
                source_seq: None,
            },
        }
    }

    #[getter]
    fn timestamp(&self) -> f64 {
        self.inner.timestamp
    }

    #[getter]
    fn root_position(&self) -> [f32; 3] {
        [
            self.inner.root_position.x,
            self.inner.root_position.y,
            self.inner.root_position.z,
        ]
    }

    #[getter]
    fn valid(&self) -> bool {
        self.inner.valid
    }
}

// =============================================================================
// Latent Frame
// =============================================================================

/// Python wrapper for LatentFrame
#[pyclass(name = "LatentFrame")]
#[derive(Clone)]
pub struct PyLatentFrame {
    inner: RustLatentFrame,
}

#[pymethods]
impl PyLatentFrame {
    #[new]
    #[pyo3(signature = (timestamp, z, valid = true))]
    fn new(timestamp: f64, z: Vec<f32>, valid: bool) -> Self {
        Self {
            inner: RustLatentFrame {
                timestamp,
                z,
                dz_dt: None,
                valid,
            },
        }
    }

    #[getter]
    fn timestamp(&self) -> f64 {
        self.inner.timestamp
    }

    #[getter]
    fn z(&self) -> Vec<f32> {
        self.inner.z.clone()
    }

    #[getter]
    fn valid(&self) -> bool {
        self.inner.valid
    }
}

// =============================================================================
// Motion Window
// =============================================================================

/// Python wrapper for MotionWindow
#[pyclass(name = "MotionWindow")]
#[derive(Clone)]
pub struct PyMotionWindow {
    inner: RustWindow,
}

#[pymethods]
impl PyMotionWindow {
    #[new]
    #[pyo3(signature = (
        window_id,
        t_start,
        t_end,
        fps,
        skeleton_frames = None,
        latent_frames = None,
        coverage = 0.98
    ))]
    fn new(
        window_id: String,
        t_start: f64,
        t_end: f64,
        fps: f32,
        skeleton_frames: Option<Vec<PySkeletonFrame>>,
        latent_frames: Option<Vec<PyLatentFrame>>,
        coverage: f32,
    ) -> Self {
        Self {
            inner: RustWindow {
                window_id,
                t_start,
                t_end,
                fps,
                skeleton_frames: skeleton_frames
                    .map(|f| f.into_iter().map(|pf| pf.inner).collect())
                    .unwrap_or_default(),
                latent_frames: latent_frames
                    .map(|f| f.into_iter().map(|pf| pf.inner).collect())
                    .unwrap_or_default(),
                coverage,
                device_offsets: HashMap::new(),
                dropped_reason: None,
            },
        }
    }

    /// Create a test window with still skeleton frames
    #[staticmethod]
    fn still(duration_seconds: f32, start_time: f64) -> Self {
        let fps = 50.0;
        let frame_count = (duration_seconds * fps) as usize;

        let frames: Vec<RustSkeletonFrame> = (0..frame_count)
            .map(|i| RustSkeletonFrame {
                timestamp: start_time + i as f64 * (1.0 / fps as f64),
                root_position: Vec3::new(0.0, 1.0, 0.0),
                root_rotation: Quat::IDENTITY,
                bone_rotations: [Quat::IDENTITY; MOCOPI_BONE_COUNT],
                valid: true,
                source_seq: Some(i as u64),
            })
            .collect();

        Self {
            inner: RustWindow {
                window_id: format!("still_window_{}", start_time),
                t_start: start_time,
                t_end: start_time + duration_seconds as f64,
                fps,
                skeleton_frames: frames,
                latent_frames: vec![],
                coverage: 0.98,
                device_offsets: HashMap::new(),
                dropped_reason: None,
            },
        }
    }

    #[getter]
    fn window_id(&self) -> &str {
        &self.inner.window_id
    }

    #[getter]
    fn t_start(&self) -> f64 {
        self.inner.t_start
    }

    #[getter]
    fn t_end(&self) -> f64 {
        self.inner.t_end
    }

    #[getter]
    fn fps(&self) -> f32 {
        self.inner.fps
    }

    #[getter]
    fn coverage(&self) -> f32 {
        self.inner.coverage
    }

    #[getter]
    fn frame_count(&self) -> usize {
        self.inner.frame_count()
    }

    fn has_skeleton(&self) -> bool {
        self.inner.has_skeleton()
    }

    fn has_latent(&self) -> bool {
        self.inner.has_latent()
    }

    fn __repr__(&self) -> String {
        format!(
            "MotionWindow(id='{}', t=[{:.2}, {:.2}], fps={}, frames={}, coverage={:.2})",
            self.inner.window_id,
            self.inner.t_start,
            self.inner.t_end,
            self.inner.fps,
            self.inner.frame_count(),
            self.inner.coverage
        )
    }
}

// =============================================================================
// Debug Trace
// =============================================================================

/// Python wrapper for DebugTrace
#[pyclass(name = "DebugTrace")]
#[derive(Clone)]
pub struct PyDebugTrace {
    inner: RustDebugTrace,
}

#[pymethods]
impl PyDebugTrace {
    #[getter]
    fn raw_features(&self) -> HashMap<String, f32> {
        self.inner.raw_features.clone()
    }

    #[getter]
    fn neighbor_ids(&self) -> Option<Vec<String>> {
        self.inner.neighbor_ids.clone()
    }

    #[getter]
    fn intermediate_scores(&self) -> HashMap<String, f32> {
        self.inner.intermediate_scores.clone()
    }
}

// =============================================================================
// Anticipation Packet
// =============================================================================

/// Python wrapper for AnticipationPacket
#[pyclass(name = "AnticipationPacket")]
#[derive(Clone)]
pub struct PyAnticipationPacket {
    inner: RustPacket,
}

#[pymethods]
impl PyAnticipationPacket {
    // Scalar getters
    #[getter]
    fn commitment(&self) -> f32 {
        self.inner.commitment
    }

    #[getter]
    fn uncertainty(&self) -> f32 {
        self.inner.uncertainty
    }

    #[getter]
    fn transition_pressure(&self) -> f32 {
        self.inner.transition_pressure
    }

    #[getter]
    fn recovery_margin(&self) -> f32 {
        self.inner.recovery_margin
    }

    #[getter]
    fn phase_stiffness(&self) -> f32 {
        self.inner.phase_stiffness
    }

    #[getter]
    fn novelty(&self) -> f32 {
        self.inner.novelty
    }

    #[getter]
    fn stability(&self) -> f32 {
        self.inner.stability
    }

    // Vector getters
    #[getter]
    fn regime_embedding(&self) -> Vec<f32> {
        self.inner.regime_embedding.clone()
    }

    #[getter]
    fn constraint_vector(&self) -> Vec<f32> {
        self.inner.constraint_vector.clone()
    }

    #[getter]
    fn derivative_summary(&self) -> Vec<f32> {
        self.inner.derivative_summary.clone()
    }

    // Debug
    #[getter]
    fn debug(&self) -> Option<PyDebugTrace> {
        self.inner.debug.clone().map(|d| PyDebugTrace { inner: d })
    }

    // Provenance
    #[getter]
    fn window_id(&self) -> &str {
        &self.inner.window_id
    }

    #[getter]
    fn timestamp(&self) -> f64 {
        self.inner.timestamp
    }

    #[getter]
    fn schema_version(&self) -> &str {
        &self.inner.schema_version
    }

    /// Get all scalars as a dictionary
    fn scalars(&self) -> HashMap<String, f32> {
        let mut m = HashMap::new();
        m.insert("commitment".to_string(), self.inner.commitment);
        m.insert("uncertainty".to_string(), self.inner.uncertainty);
        m.insert("transition_pressure".to_string(), self.inner.transition_pressure);
        m.insert("recovery_margin".to_string(), self.inner.recovery_margin);
        m.insert("phase_stiffness".to_string(), self.inner.phase_stiffness);
        m.insert("novelty".to_string(), self.inner.novelty);
        m.insert("stability".to_string(), self.inner.stability);
        m
    }

    /// Validate packet against invariants
    fn validate(&self) -> PyResult<()> {
        self.inner
            .validate()
            .map_err(|e| PyValueError::new_err(e))
    }

    fn __repr__(&self) -> String {
        format!(
            "AnticipationPacket(commitment={:.3}, uncertainty={:.3}, transition_pressure={:.3}, window_id='{}')",
            self.inner.commitment,
            self.inner.uncertainty,
            self.inner.transition_pressure,
            self.inner.window_id
        )
    }
}

// =============================================================================
// Anticipation Kernel
// =============================================================================

/// Python wrapper for AnticipationKernel
#[pyclass(name = "AnticipationKernel")]
pub struct PyAnticipationKernel {
    inner: RustKernel,
}

#[pymethods]
impl PyAnticipationKernel {
    #[new]
    fn new(config: PyAnticipationConfig) -> Self {
        Self {
            inner: RustKernel::new(config.inner),
        }
    }

    /// Process a motion window and return an anticipation packet
    fn process(&mut self, window: &PyMotionWindow) -> PyResult<PyAnticipationPacket> {
        self.inner
            .process(&window.inner)
            .map(|p| PyAnticipationPacket { inner: p })
            .map_err(|e| match e {
                KernelError::InsufficientCoverage(coverage, min) => {
                    PyValueError::new_err(format!(
                        "Coverage {:.3} below minimum {:.3}",
                        coverage, min
                    ))
                }
                KernelError::NoMotionData => {
                    PyValueError::new_err("No skeleton or latent frames available")
                }
                KernelError::InvalidConfig(msg) => {
                    PyValueError::new_err(format!("Invalid config: {}", msg))
                }
            })
    }

    /// Reset kernel state
    fn reset(&mut self) {
        self.inner.reset();
    }

    fn __repr__(&self) -> String {
        format!(
            "AnticipationKernel(fps={}, window_seconds={})",
            self.inner.config().fps,
            self.inner.config().window_seconds
        )
    }
}

// =============================================================================
// Module
// =============================================================================

/// cc-anticipation Python module
#[pymodule]
fn cc_anticipation(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<PyAnticipationConfig>()?;
    m.add_class::<PySkeletonFrame>()?;
    m.add_class::<PyLatentFrame>()?;
    m.add_class::<PyMotionWindow>()?;
    m.add_class::<PyDebugTrace>()?;
    m.add_class::<PyAnticipationPacket>()?;
    m.add_class::<PyAnticipationKernel>()?;

    // Add schema version
    m.add("SCHEMA_VERSION", crate::types::SCHEMA_VERSION)?;
    m.add("MOCOPI_BONE_COUNT", MOCOPI_BONE_COUNT)?;

    Ok(())
}

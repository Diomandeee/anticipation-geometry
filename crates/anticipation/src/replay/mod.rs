//! Replay harness for deterministic validation
//!
//! This module provides tools for recording and replaying
//! kernel processing to verify determinism (INV-001).
//!
//! # Overview
//!
//! The replay system enables:
//! 1. Recording MotionWindow + AnticipationPacket pairs during live processing
//! 2. Saving traces to disk (JSON format)
//! 3. Replaying traces through the kernel
//! 4. Comparing outputs to verify determinism
//!
//! # Usage
//!
//! ## Recording
//!
//! ```ignore
//! use cc_anticipation::replay::TraceRecorder;
//!
//! let mut recorder = TraceRecorder::new("session_001");
//!
//! // During processing
//! let packet = kernel.process(&window)?;
//! recorder.record(&window, &packet);
//!
//! // Save to disk
//! recorder.save_to_file("trace.json")?;
//! ```
//!
//! ## Validation
//!
//! ```ignore
//! use cc_anticipation::replay::ReplayHarness;
//!
//! let harness = ReplayHarness::load_from_file("trace.json")?;
//! let result = harness.validate(&mut kernel)?;
//!
//! assert!(result.all_deterministic());
//! ```

use crate::config::AnticipationConfig;
use crate::kernel::AnticipationKernel;
use crate::types::{MotionWindow, AnticipationPacket, SkeletonFrame, LatentFrame, SCHEMA_VERSION};

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufReader, BufWriter, Write};
use std::path::Path;

/// A single trace entry: input window + expected output packet
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TraceEntry {
    /// Sequence number in the trace
    pub seq: u64,
    /// The input motion window
    pub window: SerializableWindow,
    /// The expected output packet
    pub expected: SerializablePacket,
    /// Timestamp when recorded
    pub recorded_at: f64,
}

/// Serializable version of MotionWindow
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SerializableWindow {
    pub window_id: String,
    pub t_start: f64,
    pub t_end: f64,
    pub fps: f32,
    pub skeleton_frames: Vec<SerializableSkeletonFrame>,
    pub latent_frames: Vec<SerializableLatentFrame>,
    pub coverage: f32,
}

/// Serializable skeleton frame
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SerializableSkeletonFrame {
    pub timestamp: f64,
    pub root_position: [f32; 3],
    pub root_rotation: [f32; 4],
    pub bone_rotations: Vec<[f32; 4]>,
    pub valid: bool,
    pub source_seq: Option<u64>,
}

/// Serializable latent frame
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SerializableLatentFrame {
    pub timestamp: f64,
    pub z: Vec<f32>,
    pub dz_dt: Option<Vec<f32>>,
    pub valid: bool,
}

/// Serializable version of AnticipationPacket
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SerializablePacket {
    pub commitment: f32,
    pub uncertainty: f32,
    pub transition_pressure: f32,
    pub recovery_margin: f32,
    pub phase_stiffness: f32,
    pub novelty: f32,
    pub stability: f32,
    pub regime_embedding: Vec<f32>,
    pub constraint_vector: Vec<f32>,
    pub derivative_summary: Vec<f32>,
    pub window_id: String,
    pub timestamp: f64,
    pub schema_version: String,
}

/// Trace file metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TraceMetadata {
    /// Session identifier
    pub session_id: String,
    /// Schema version used
    pub schema_version: String,
    /// Configuration used
    pub config: SerializableConfig,
    /// Number of entries
    pub entry_count: usize,
    /// Total duration in seconds
    pub duration_seconds: f64,
    /// Creation timestamp
    pub created_at: String,
}

/// Serializable config
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SerializableConfig {
    pub fps: f32,
    pub window_seconds: f32,
    pub regime_embedding_dim: usize,
    pub min_coverage: f32,
}

/// Complete trace file
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TraceFile {
    /// File format version
    pub format_version: String,
    /// Metadata
    pub metadata: TraceMetadata,
    /// Entries
    pub entries: Vec<TraceEntry>,
}

impl TraceFile {
    /// Format version
    pub const FORMAT_VERSION: &'static str = "1.0.0";
}

/// Trace recorder for capturing kernel processing
#[derive(Debug)]
pub struct TraceRecorder {
    session_id: String,
    config: AnticipationConfig,
    entries: Vec<TraceEntry>,
    start_time: Option<f64>,
}

impl TraceRecorder {
    /// Create a new trace recorder
    pub fn new(session_id: impl Into<String>, config: AnticipationConfig) -> Self {
        Self {
            session_id: session_id.into(),
            config,
            entries: Vec::new(),
            start_time: None,
        }
    }

    /// Record a window and its output packet
    pub fn record(&mut self, window: &MotionWindow, packet: &AnticipationPacket) {
        if self.start_time.is_none() {
            self.start_time = Some(window.t_start);
        }

        let entry = TraceEntry {
            seq: self.entries.len() as u64,
            window: window.into(),
            expected: packet.into(),
            recorded_at: window.t_end,
        };

        self.entries.push(entry);
    }

    /// Number of recorded entries
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether the recorder is empty
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Build the trace file
    pub fn build(&self) -> TraceFile {
        let duration = self.entries.last()
            .map(|e| e.recorded_at - self.start_time.unwrap_or(0.0))
            .unwrap_or(0.0);

        TraceFile {
            format_version: TraceFile::FORMAT_VERSION.to_string(),
            metadata: TraceMetadata {
                session_id: self.session_id.clone(),
                schema_version: SCHEMA_VERSION.to_string(),
                config: (&self.config).into(),
                entry_count: self.entries.len(),
                duration_seconds: duration,
                created_at: chrono_lite_now(),
            },
            entries: self.entries.clone(),
        }
    }

    /// Save trace to a file
    pub fn save_to_file(&self, path: impl AsRef<Path>) -> Result<(), TraceError> {
        let trace = self.build();
        let file = File::create(path).map_err(TraceError::Io)?;
        let mut writer = BufWriter::new(file);
        serde_json::to_writer_pretty(&mut writer, &trace).map_err(TraceError::Json)?;
        writer.flush().map_err(TraceError::Io)?;
        Ok(())
    }
}

/// Result of replaying a single entry
#[derive(Debug, Clone)]
pub struct ReplayResult {
    /// Sequence number
    pub seq: u64,
    /// Whether output matched expected
    pub deterministic: bool,
    /// Commitment difference
    pub diff_commitment: f32,
    /// Uncertainty difference
    pub diff_uncertainty: f32,
    /// Max embedding difference
    pub diff_embedding_max: f32,
    /// Error message if processing failed
    pub error: Option<String>,
}

/// Summary of replay validation
#[derive(Debug, Clone)]
pub struct ValidationSummary {
    /// Total entries
    pub total_entries: usize,
    /// Deterministic entries
    pub deterministic_count: usize,
    /// Non-deterministic entries
    pub non_deterministic_count: usize,
    /// Failed entries (processing error)
    pub failed_count: usize,
    /// Per-entry results
    pub results: Vec<ReplayResult>,
}

impl ValidationSummary {
    /// Whether all entries are deterministic
    pub fn all_deterministic(&self) -> bool {
        self.non_deterministic_count == 0 && self.failed_count == 0
    }

    /// Determinism rate (0.0 to 1.0)
    pub fn determinism_rate(&self) -> f32 {
        if self.total_entries == 0 {
            return 1.0;
        }
        self.deterministic_count as f32 / self.total_entries as f32
    }
}

/// Replay harness for validation
#[derive(Debug)]
pub struct ReplayHarness {
    trace: TraceFile,
}

impl ReplayHarness {
    /// Load a trace from file
    pub fn load_from_file(path: impl AsRef<Path>) -> Result<Self, TraceError> {
        let file = File::open(path).map_err(TraceError::Io)?;
        let reader = BufReader::new(file);
        let trace: TraceFile = serde_json::from_reader(reader).map_err(TraceError::Json)?;
        Ok(Self { trace })
    }

    /// Create from trace file
    pub fn new(trace: TraceFile) -> Self {
        Self { trace }
    }

    /// Get trace metadata
    pub fn metadata(&self) -> &TraceMetadata {
        &self.trace.metadata
    }

    /// Number of entries
    pub fn len(&self) -> usize {
        self.trace.entries.len()
    }

    /// Whether empty
    pub fn is_empty(&self) -> bool {
        self.trace.entries.is_empty()
    }

    /// Validate determinism by replaying through kernel
    ///
    /// Tolerance is the maximum allowed difference for floating point values.
    pub fn validate(&self, kernel: &mut AnticipationKernel, tolerance: f32) -> ValidationSummary {
        let mut results = Vec::with_capacity(self.trace.entries.len());
        let mut deterministic_count = 0;
        let mut failed_count = 0;

        // Reset kernel to fresh state
        kernel.reset();

        for entry in &self.trace.entries {
            let window: MotionWindow = (&entry.window).into();

            match kernel.process(&window) {
                Ok(actual) => {
                    let expected = &entry.expected;

                    // Compare outputs
                    let diff_commitment = (actual.commitment - expected.commitment).abs();
                    let diff_uncertainty = (actual.uncertainty - expected.uncertainty).abs();
                    let diff_embedding_max = actual.regime_embedding.iter()
                        .zip(expected.regime_embedding.iter())
                        .map(|(a, e)| (a - e).abs())
                        .fold(0.0f32, f32::max);

                    let deterministic = diff_commitment <= tolerance
                        && diff_uncertainty <= tolerance
                        && diff_embedding_max <= tolerance;

                    if deterministic {
                        deterministic_count += 1;
                    }

                    results.push(ReplayResult {
                        seq: entry.seq,
                        deterministic,
                        diff_commitment,
                        diff_uncertainty,
                        diff_embedding_max,
                        error: None,
                    });
                }
                Err(e) => {
                    failed_count += 1;
                    results.push(ReplayResult {
                        seq: entry.seq,
                        deterministic: false,
                        diff_commitment: f32::MAX,
                        diff_uncertainty: f32::MAX,
                        diff_embedding_max: f32::MAX,
                        error: Some(e.to_string()),
                    });
                }
            }
        }

        ValidationSummary {
            total_entries: self.trace.entries.len(),
            deterministic_count,
            non_deterministic_count: self.trace.entries.len() - deterministic_count - failed_count,
            failed_count,
            results,
        }
    }
}

/// Trace errors
#[derive(Debug)]
pub enum TraceError {
    /// IO error
    Io(std::io::Error),
    /// JSON serialization error
    Json(serde_json::Error),
}

impl std::fmt::Display for TraceError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TraceError::Io(e) => write!(f, "IO error: {}", e),
            TraceError::Json(e) => write!(f, "JSON error: {}", e),
        }
    }
}

impl std::error::Error for TraceError {}

// -------------------------------------------------------------------------
// Conversions
// -------------------------------------------------------------------------

impl From<&MotionWindow> for SerializableWindow {
    fn from(w: &MotionWindow) -> Self {
        Self {
            window_id: w.window_id.clone(),
            t_start: w.t_start,
            t_end: w.t_end,
            fps: w.fps,
            skeleton_frames: w.skeleton_frames.iter().map(Into::into).collect(),
            latent_frames: w.latent_frames.iter().map(Into::into).collect(),
            coverage: w.coverage,
        }
    }
}

impl From<&SerializableWindow> for MotionWindow {
    fn from(w: &SerializableWindow) -> Self {
        Self {
            window_id: w.window_id.clone(),
            t_start: w.t_start,
            t_end: w.t_end,
            fps: w.fps,
            skeleton_frames: w.skeleton_frames.iter().map(Into::into).collect(),
            latent_frames: w.latent_frames.iter().map(Into::into).collect(),
            coverage: w.coverage,
            device_offsets: HashMap::new(),
            dropped_reason: None,
        }
    }
}

impl From<&SkeletonFrame> for SerializableSkeletonFrame {
    fn from(f: &SkeletonFrame) -> Self {
        Self {
            timestamp: f.timestamp,
            root_position: [f.root_position.x, f.root_position.y, f.root_position.z],
            root_rotation: [f.root_rotation.x, f.root_rotation.y, f.root_rotation.z, f.root_rotation.w],
            bone_rotations: f.bone_rotations.iter()
                .map(|q| [q.x, q.y, q.z, q.w])
                .collect(),
            valid: f.valid,
            source_seq: f.source_seq,
        }
    }
}

impl From<&SerializableSkeletonFrame> for SkeletonFrame {
    fn from(f: &SerializableSkeletonFrame) -> Self {
        use crate::types::{Quat, Vec3, MOCOPI_BONE_COUNT};

        let mut bone_rotations = [Quat::IDENTITY; MOCOPI_BONE_COUNT];
        for (i, r) in f.bone_rotations.iter().enumerate() {
            if i < MOCOPI_BONE_COUNT {
                bone_rotations[i] = Quat::new(r[0], r[1], r[2], r[3]);
            }
        }

        Self {
            timestamp: f.timestamp,
            root_position: Vec3::new(f.root_position[0], f.root_position[1], f.root_position[2]),
            root_rotation: Quat::new(f.root_rotation[0], f.root_rotation[1], f.root_rotation[2], f.root_rotation[3]),
            bone_rotations,
            valid: f.valid,
            source_seq: f.source_seq,
        }
    }
}

impl From<&LatentFrame> for SerializableLatentFrame {
    fn from(f: &LatentFrame) -> Self {
        Self {
            timestamp: f.timestamp,
            z: f.z.clone(),
            dz_dt: f.dz_dt.clone(),
            valid: f.valid,
        }
    }
}

impl From<&SerializableLatentFrame> for LatentFrame {
    fn from(f: &SerializableLatentFrame) -> Self {
        Self {
            timestamp: f.timestamp,
            z: f.z.clone(),
            dz_dt: f.dz_dt.clone(),
            valid: f.valid,
        }
    }
}

impl From<&AnticipationPacket> for SerializablePacket {
    fn from(p: &AnticipationPacket) -> Self {
        Self {
            commitment: p.commitment,
            uncertainty: p.uncertainty,
            transition_pressure: p.transition_pressure,
            recovery_margin: p.recovery_margin,
            phase_stiffness: p.phase_stiffness,
            novelty: p.novelty,
            stability: p.stability,
            regime_embedding: p.regime_embedding.clone(),
            constraint_vector: p.constraint_vector.clone(),
            derivative_summary: p.derivative_summary.clone(),
            window_id: p.window_id.clone(),
            timestamp: p.timestamp,
            schema_version: p.schema_version.clone(),
        }
    }
}

impl From<&AnticipationConfig> for SerializableConfig {
    fn from(c: &AnticipationConfig) -> Self {
        Self {
            fps: c.fps,
            window_seconds: c.window_seconds,
            regime_embedding_dim: c.regime_embedding_dim,
            min_coverage: c.min_coverage,
        }
    }
}

/// Simple timestamp (no chrono dependency)
fn chrono_lite_now() -> String {
    // Use a placeholder - in production this would use actual timestamps
    "2025-12-26T00:00:00Z".to_string()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{Quat, Vec3, MOCOPI_BONE_COUNT};

    fn create_test_window(t: f64) -> MotionWindow {
        let config = AnticipationConfig::default();
        let frame_count = (config.window_seconds * config.fps) as usize;

        let frames: Vec<SkeletonFrame> = (0..frame_count)
            .map(|i| SkeletonFrame {
                timestamp: t + i as f64 * (1.0 / config.fps as f64),
                root_position: Vec3::new(0.0, 1.0, 0.0),
                root_rotation: Quat::IDENTITY,
                bone_rotations: [Quat::IDENTITY; MOCOPI_BONE_COUNT],
                valid: true,
                source_seq: Some(i as u64),
            })
            .collect();

        MotionWindow {
            window_id: format!("test_window_{}", t),
            t_start: t,
            t_end: t + config.window_seconds as f64,
            fps: config.fps,
            skeleton_frames: frames,
            latent_frames: vec![],
            coverage: 0.98,
            device_offsets: HashMap::new(),
            dropped_reason: None,
        }
    }

    #[test]
    fn test_trace_recorder() {
        let config = AnticipationConfig::default();
        let mut kernel = AnticipationKernel::new(config.clone());
        let mut recorder = TraceRecorder::new("test_session", config);

        // Record some windows
        for i in 0..5 {
            let window = create_test_window(i as f64);
            let packet = kernel.process(&window).unwrap();
            recorder.record(&window, &packet);
        }

        assert_eq!(recorder.len(), 5);

        let trace = recorder.build();
        assert_eq!(trace.metadata.entry_count, 5);
        assert_eq!(trace.metadata.session_id, "test_session");
    }

    #[test]
    fn test_replay_determinism() {
        let config = AnticipationConfig::default();
        let mut kernel = AnticipationKernel::new(config.clone());
        let mut recorder = TraceRecorder::new("determinism_test", config.clone());

        // Record
        for i in 0..10 {
            let window = create_test_window(i as f64);
            let packet = kernel.process(&window).unwrap();
            recorder.record(&window, &packet);
        }

        let trace = recorder.build();

        // Replay with fresh kernel
        let mut replay_kernel = AnticipationKernel::new(config);
        let harness = ReplayHarness::new(trace);
        let summary = harness.validate(&mut replay_kernel, 1e-6);

        assert!(summary.all_deterministic(),
            "Replay should be deterministic. Determinism rate: {}",
            summary.determinism_rate());
    }

    #[test]
    fn test_window_serialization_roundtrip() {
        let window = create_test_window(0.0);
        let serializable: SerializableWindow = (&window).into();
        let roundtrip: MotionWindow = (&serializable).into();

        assert_eq!(window.window_id, roundtrip.window_id);
        assert_eq!(window.t_start, roundtrip.t_start);
        assert_eq!(window.t_end, roundtrip.t_end);
        assert_eq!(window.fps, roundtrip.fps);
        assert_eq!(window.skeleton_frames.len(), roundtrip.skeleton_frames.len());
    }

    #[test]
    fn test_validation_summary() {
        let summary = ValidationSummary {
            total_entries: 10,
            deterministic_count: 8,
            non_deterministic_count: 1,
            failed_count: 1,
            results: vec![],
        };

        assert!(!summary.all_deterministic());
        assert!((summary.determinism_rate() - 0.8).abs() < 1e-6);
    }
}

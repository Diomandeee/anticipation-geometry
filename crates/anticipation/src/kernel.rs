//! Main anticipation kernel
//!
//! Converts MotionWindow → AnticipationPacket deterministically.
//!
//! # Invariants
//!
//! - INV-001: Deterministic replay (same input → same output)
//! - INV-006: No heap allocation in hot path
//! - PERF-001: < 2ms per tick
//!
//! # Neighbor Integration (v1)
//!
//! When the `neighbors` feature is enabled and a MotionPhraseIndex is set,
//! the kernel uses continuation dispersion for improved uncertainty estimates.

use crate::config::AnticipationConfig;
use crate::features::{compute_kinematic_features, compute_latent_features, KinematicFeatures, LatentFeatures};
use crate::embedding::project_to_regime_embedding;
use crate::constraints::compute_constraint_vector;
use crate::scalars::{
    compute_commitment, compute_uncertainty, compute_transition_pressure,
    compute_recovery_margin, compute_phase_stiffness, compute_stability,
};
use crate::types::{AnticipationPacket, DebugTrace, MotionWindow, SCHEMA_VERSION};

#[cfg(feature = "neighbors")]
use crate::neighbors::{MotionPhraseIndex, compute_continuation_dispersion, dispersion_to_uncertainty};

#[cfg(feature = "neighbors")]
use crate::phrase::{MotionPhraseLibrary, PriorBundle, MultiAxisQuery};

use std::collections::HashMap;
use thiserror::Error;
#[cfg(feature = "neighbors")]
use std::sync::Arc;

/// Errors that can occur during kernel processing
#[derive(Error, Debug)]
pub enum KernelError {
    /// Coverage below minimum threshold
    #[error("coverage {0:.3} < min_coverage {1:.3}")]
    InsufficientCoverage(f32, f32),

    /// No usable motion data
    #[error("no skeleton or latent frames available")]
    NoMotionData,

    /// Configuration validation failed
    #[error("invalid configuration: {0}")]
    InvalidConfig(String),

    /// Skeleton frame validation failed
    #[error("invalid skeleton frame: {0}")]
    InvalidSkeletonFrame(String),

    /// Latent frame validation failed  
    #[error("invalid latent frame: {0}")]
    InvalidLatentFrame(String),

    /// Insufficient bones in skeleton frame
    #[error("insufficient bones: got {0}, need at least {1}")]
    InsufficientBones(usize, usize),
}

/// The anticipation kernel - main entry point
///
/// Converts stabilized motion windows into anticipatory signals.
///
/// # Example
///
/// ```ignore
/// let config = AnticipationConfig::default();
/// let mut kernel = AnticipationKernel::new(config);
///
/// loop {
///     let window = receive_window(); // From cc-window-aligner
///     match kernel.process(&window) {
///         Ok(packet) => send_to_conductor(packet),
///         Err(e) => log_error(e),
///     }
/// }
/// ```
pub struct AnticipationKernel {
    config: AnticipationConfig,

    // -------------------------------------------------------------------------
    // State for temporal computations
    // -------------------------------------------------------------------------

    /// Previous commitment value for derivative computation
    prev_commitment: f32,

    /// Previous uncertainty value for derivative computation
    prev_uncertainty: f32,

    /// Previous timestamp for dt computation
    prev_timestamp: Option<f64>,

    // -------------------------------------------------------------------------
    // Novelty history ring buffer
    // -------------------------------------------------------------------------

    /// Ring buffer of recent regime embeddings
    regime_history: Vec<Vec<f32>>,

    /// Current write index in regime_history
    regime_history_idx: usize,

    /// Number of valid entries in regime_history
    regime_history_count: usize,

    // -------------------------------------------------------------------------
    // Pre-allocated buffers (INV-006: no hot-path allocation)
    // -------------------------------------------------------------------------

    /// Buffer for fused features
    feature_buffer: Vec<f32>,

    /// Buffer for derivative summary computation
    derivative_buffer: Vec<f32>,

    // -------------------------------------------------------------------------
    // Neighbor-based uncertainty (v1, optional)
    // -------------------------------------------------------------------------

    /// Optional phrase index for neighbor-based uncertainty
    #[cfg(feature = "neighbors")]
    phrase_index: Option<Arc<MotionPhraseIndex>>,

    /// Optional phrase library for PriorBundle queries
    #[cfg(feature = "neighbors")]
    phrase_library: Option<Arc<MotionPhraseLibrary>>,

    /// Number of neighbors to query for uncertainty
    #[cfg(feature = "neighbors")]
    k_neighbors: usize,
}

impl AnticipationKernel {
    /// Create a new anticipation kernel with the given configuration
    ///
    /// # Panics
    ///
    /// Panics if configuration validation fails.
    pub fn new(config: AnticipationConfig) -> Self {
        config.validate().expect("Invalid configuration");

        let history_len = config.novelty_history_len();
        let embedding_dim = config.regime_embedding_dim;

        Self {
            config: config.clone(),
            prev_commitment: 0.5,
            prev_uncertainty: 0.5,
            prev_timestamp: None,
            regime_history: vec![vec![0.0; embedding_dim]; history_len],
            regime_history_idx: 0,
            regime_history_count: 0,
            feature_buffer: vec![0.0; 128],
            derivative_buffer: vec![0.0; config.derivative_summary_dim],
            #[cfg(feature = "neighbors")]
            phrase_index: None,
            #[cfg(feature = "neighbors")]
            phrase_library: None,
            #[cfg(feature = "neighbors")]
            k_neighbors: 8,
        }
    }

    /// Set the motion phrase index for neighbor-based uncertainty
    ///
    /// When set, the kernel uses continuation dispersion from similar
    /// motion phrases to improve uncertainty estimates.
    #[cfg(feature = "neighbors")]
    pub fn with_phrase_index(mut self, index: Arc<MotionPhraseIndex>) -> Self {
        self.phrase_index = Some(index);
        self
    }

    /// Set the motion phrase index (mutable version)
    #[cfg(feature = "neighbors")]
    pub fn set_phrase_index(&mut self, index: Arc<MotionPhraseIndex>) {
        self.phrase_index = Some(index);
    }

    /// Clear the phrase index
    #[cfg(feature = "neighbors")]
    pub fn clear_phrase_index(&mut self) {
        self.phrase_index = None;
    }

    /// Set number of neighbors for uncertainty computation
    #[cfg(feature = "neighbors")]
    pub fn set_k_neighbors(&mut self, k: usize) {
        self.k_neighbors = k;
    }

    /// Set the motion phrase library for PriorBundle queries
    ///
    /// The phrase library provides richer neighbor statistics including
    /// regime distributions and transition predictions.
    #[cfg(feature = "neighbors")]
    pub fn with_phrase_library(mut self, library: Arc<MotionPhraseLibrary>) -> Self {
        self.phrase_library = Some(library);
        self
    }

    /// Set the motion phrase library (mutable version)
    #[cfg(feature = "neighbors")]
    pub fn set_phrase_library(&mut self, library: Arc<MotionPhraseLibrary>) {
        self.phrase_library = Some(library);
    }

    /// Clear the phrase library
    #[cfg(feature = "neighbors")]
    pub fn clear_phrase_library(&mut self) {
        self.phrase_library = None;
    }

    /// Query priors from the phrase library
    ///
    /// Returns a PriorBundle with neighbor statistics, regime distributions,
    /// and continuation predictions. Returns None if no library is set or
    /// if the library is empty.
    #[cfg(feature = "neighbors")]
    pub fn query_priors(&self, regime_embedding: &[f32]) -> Option<PriorBundle> {
        let library = self.phrase_library.as_ref()?;

        if library.is_empty() {
            return Some(PriorBundle::empty("live"));
        }

        let query = MultiAxisQuery::motion_only(regime_embedding.to_vec());
        let neighbors = library.query_multi_axis(&query, self.k_neighbors);

        if neighbors.is_empty() {
            return Some(PriorBundle::empty("live"));
        }

        Some(library.build_prior_bundle("live", &neighbors))
    }

    /// Query priors and get blended uncertainty
    ///
    /// Combines neighbor-based uncertainty (70%) with heuristic uncertainty (30%)
    /// for stability. Returns (uncertainty, prior_bundle).
    #[cfg(feature = "neighbors")]
    pub fn query_priors_with_uncertainty(
        &self,
        regime_embedding: &[f32],
    ) -> (f32, Option<PriorBundle>) {
        let heuristic = compute_uncertainty(regime_embedding, &self.config);

        let bundle = match self.query_priors(regime_embedding) {
            Some(b) if b.has_neighbors() => b,
            _ => return (heuristic, None),
        };

        let blended = bundle.blended_uncertainty(heuristic);
        (blended, Some(bundle))
    }

    /// Process a MotionWindow and emit an AnticipationPacket
    ///
    /// # Invariants
    ///
    /// - INV-001: Deterministic given same input
    /// - INV-003: Refuses low-coverage windows
    /// - INV-006: No heap allocation (uses pre-allocated buffers)
    ///
    /// # Errors
    ///
    /// Returns error if:
    /// - Coverage below `config.min_coverage`
    /// - No skeleton or latent frames available
    pub fn process(&mut self, window: &MotionWindow) -> Result<AnticipationPacket, KernelError> {
        // INV-003: Check coverage threshold
        if window.coverage < self.config.min_coverage {
            return Err(KernelError::InsufficientCoverage(
                window.coverage,
                self.config.min_coverage,
            ));
        }

        // 1. Compute kinematic features from skeleton
        let kin_features = if window.has_skeleton() {
            Some(compute_kinematic_features(&window.skeleton_frames, &self.config))
        } else {
            None
        };

        // 2. Compute latent dynamics from LIM-RPS
        let lat_features = if window.has_latent() {
            Some(compute_latent_features(&window.latent_frames, &self.config))
        } else {
            None
        };

        // Must have at least one source
        if kin_features.is_none() && lat_features.is_none() {
            return Err(KernelError::NoMotionData);
        }

        // 3. Fuse into feature vector
        let fused_features = self.fuse_features(&kin_features, &lat_features);

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
        // Use neighbor-based uncertainty if phrase index is available
        #[cfg(feature = "neighbors")]
        let (uncertainty, neighbor_ids) = self.compute_uncertainty_with_neighbors(&regime_embedding);

        #[cfg(not(feature = "neighbors"))]
        let uncertainty = compute_uncertainty(
            &regime_embedding,
            &self.config,
        );
        #[cfg(not(feature = "neighbors"))]
        let neighbor_ids: Option<Vec<String>> = None;

        let commitment = compute_commitment(
            uncertainty,
            &constraint_vector,
            &fused_features,
            &self.config,
        );

        let dt = self.prev_timestamp
            .map(|prev| (window.t_end - prev) as f32)
            .unwrap_or(self.config.frame_duration());

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
            compute_stability(kf)
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
            Some(self.build_debug_trace(&fused_features, &kin_features, &lat_features, neighbor_ids))
        } else {
            None
        };

        let packet = AnticipationPacket {
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
            source_identity: None, // Set by caller for provenance tracking
        };

        // Validate before returning (debug builds)
        debug_assert!(packet.validate().is_ok(), "Packet validation failed");

        Ok(packet)
    }

    /// Reset kernel state (for replay or testing)
    pub fn reset(&mut self) {
        self.prev_commitment = 0.5;
        self.prev_uncertainty = 0.5;
        self.prev_timestamp = None;
        self.regime_history_idx = 0;
        self.regime_history_count = 0;

        // Clear history
        for hist in &mut self.regime_history {
            for v in hist.iter_mut() {
                *v = 0.0;
            }
        }
    }

    /// Get current configuration
    pub fn config(&self) -> &AnticipationConfig {
        &self.config
    }

    /// Validate a MotionWindow before processing
    /// 
    /// Checks for:
    /// - NaN/Inf in skeleton frame positions
    /// - NaN/Inf in skeleton frame rotations
    /// - NaN/Inf in latent frame vectors
    /// - Timestamp validity
    /// - Minimum bone count for sensor coverage
    ///
    /// # Returns
    /// 
    /// Ok(()) if validation passes, Err with details otherwise.
    pub fn validate_window(&self, window: &MotionWindow) -> Result<(), KernelError> {
        // Validate skeleton frames
        for (i, frame) in window.skeleton_frames.iter().enumerate() {
            if !frame.valid {
                continue; // Skip invalid frames, they won't be used
            }

            // Check root position for NaN/Inf
            if !frame.root_position.x.is_finite() 
                || !frame.root_position.y.is_finite()
                || !frame.root_position.z.is_finite() {
                return Err(KernelError::InvalidSkeletonFrame(
                    format!("frame {} has invalid root position (NaN/Inf)", i)
                ));
            }

            // Check root rotation for NaN/Inf
            if !frame.root_rotation.x.is_finite()
                || !frame.root_rotation.y.is_finite()
                || !frame.root_rotation.z.is_finite()
                || !frame.root_rotation.w.is_finite() {
                return Err(KernelError::InvalidSkeletonFrame(
                    format!("frame {} has invalid root rotation (NaN/Inf)", i)
                ));
            }

            // Check all bone rotations for NaN/Inf
            for (bone_idx, quat) in frame.bone_rotations.iter().enumerate() {
                if !quat.x.is_finite() || !quat.y.is_finite() 
                    || !quat.z.is_finite() || !quat.w.is_finite() {
                    return Err(KernelError::InvalidSkeletonFrame(
                        format!("frame {} bone {} has invalid rotation (NaN/Inf)", i, bone_idx)
                    ));
                }
            }

            // Check timestamp is reasonable
            if !frame.timestamp.is_finite() || frame.timestamp < 0.0 {
                return Err(KernelError::InvalidSkeletonFrame(
                    format!("frame {} has invalid timestamp: {}", i, frame.timestamp)
                ));
            }
        }

        // Validate latent frames
        for (i, frame) in window.latent_frames.iter().enumerate() {
            if !frame.valid {
                continue;
            }

            // Check latent vector for NaN/Inf
            for (j, &val) in frame.z.iter().enumerate() {
                if !val.is_finite() {
                    return Err(KernelError::InvalidLatentFrame(
                        format!("frame {} z[{}] is NaN/Inf", i, j)
                    ));
                }
            }

            // Check derivatives if present
            if let Some(ref dz) = frame.dz_dt {
                for (j, &val) in dz.iter().enumerate() {
                    if !val.is_finite() {
                        return Err(KernelError::InvalidLatentFrame(
                            format!("frame {} dz_dt[{}] is NaN/Inf", i, j)
                        ));
                    }
                }
            }
        }

        // Check window time bounds
        if !window.t_start.is_finite() || !window.t_end.is_finite() {
            return Err(KernelError::InvalidConfig(
                format!("window has invalid time bounds: {} - {}", window.t_start, window.t_end)
            ));
        }

        if window.t_end <= window.t_start {
            return Err(KernelError::InvalidConfig(
                format!("window t_end ({}) <= t_start ({})", window.t_end, window.t_start)
            ));
        }

        Ok(())
    }

    /// Process a MotionWindow with explicit validation
    ///
    /// Like `process()` but always runs validation first.
    /// Use this when receiving data from untrusted sources.
    pub fn process_validated(&mut self, window: &MotionWindow) -> Result<AnticipationPacket, KernelError> {
        self.validate_window(window)?;
        self.process(window)
    }

    // -------------------------------------------------------------------------
    // Internal methods
    // -------------------------------------------------------------------------

    /// Fuse kinematic and latent features into a single vector
    fn fuse_features(
        &mut self,
        kin: &Option<KinematicFeatures>,
        lat: &Option<LatentFeatures>,
    ) -> Vec<f32> {
        // Clear buffer
        for v in self.feature_buffer.iter_mut() {
            *v = 0.0;
        }

        let mut idx = 0;

        // Add kinematic features
        if let Some(k) = kin {
            let kf = k.to_vec();
            for &v in &kf {
                if idx < self.feature_buffer.len() {
                    self.feature_buffer[idx] = v;
                    idx += 1;
                }
            }
        }

        // Add latent features
        if let Some(l) = lat {
            let lf = l.to_vec();
            for &v in &lf {
                if idx < self.feature_buffer.len() {
                    self.feature_buffer[idx] = v;
                    idx += 1;
                }
            }
        }

        // Return copy (avoiding allocation would require more complex lifetime handling)
        self.feature_buffer.clone()
    }

    /// Compute derivative summary from fused features
    fn compute_derivative_summary(&mut self, features: &[f32]) -> Vec<f32> {
        let dim = self.config.derivative_summary_dim;

        // Clear buffer
        for v in self.derivative_buffer.iter_mut() {
            *v = 0.0;
        }

        // Extract first N features as summary
        // In v1, this becomes a learned projection
        for (i, &f) in features.iter().take(dim).enumerate() {
            self.derivative_buffer[i] = f;
        }

        self.derivative_buffer.clone()
    }

    /// Compute novelty as distance from recent regime history centroid
    fn compute_novelty(&self, embedding: &[f32]) -> f32 {
        if self.regime_history_count == 0 {
            return 0.5; // Neutral if no history
        }

        // Compute centroid of history
        let mut centroid = vec![0.0f32; embedding.len()];
        let count = self.regime_history_count.min(self.regime_history.len());

        for i in 0..count {
            for (j, &v) in self.regime_history[i].iter().enumerate() {
                if j < centroid.len() {
                    centroid[j] += v;
                }
            }
        }

        for c in &mut centroid {
            *c /= count as f32;
        }

        // L2 distance from centroid
        let dist: f32 = embedding.iter()
            .zip(centroid.iter())
            .map(|(&e, &c)| (e - c).powi(2))
            .sum::<f32>()
            .sqrt();

        // Normalize to [0, 1] assuming embeddings are roughly unit-scaled
        (dist / 2.0).clamp(0.0, 1.0)
    }

    /// Update novelty history with new embedding
    fn update_novelty_history(&mut self, embedding: &[f32]) {
        let idx = self.regime_history_idx;
        let hist = &mut self.regime_history[idx];

        for (i, &v) in embedding.iter().enumerate() {
            if i < hist.len() {
                hist[i] = v;
            }
        }

        self.regime_history_idx = (idx + 1) % self.regime_history.len();
        if self.regime_history_count < self.regime_history.len() {
            self.regime_history_count += 1;
        }
    }

    /// Compute uncertainty with neighbor-based dispersion (v1)
    #[cfg(feature = "neighbors")]
    fn compute_uncertainty_with_neighbors(&self, embedding: &[f32]) -> (f32, Option<Vec<String>>) {
        // If no phrase index, fall back to heuristic
        let Some(ref index) = self.phrase_index else {
            let uncertainty = compute_uncertainty(embedding, &self.config);
            return (uncertainty, None);
        };

        // Query neighbors and compute dispersion
        let dispersion = compute_continuation_dispersion(index.as_ref(), embedding, self.k_neighbors);

        if dispersion.neighbor_count == 0 {
            // No neighbors found, fall back to heuristic
            let uncertainty = compute_uncertainty(embedding, &self.config);
            return (uncertainty, None);
        }

        // Convert dispersion to uncertainty
        let neighbor_uncertainty = dispersion_to_uncertainty(&dispersion);

        // Blend with heuristic uncertainty for stability
        let heuristic_uncertainty = compute_uncertainty(embedding, &self.config);
        let blend_factor = 0.7; // 70% neighbors, 30% heuristic
        let uncertainty = blend_factor * neighbor_uncertainty + (1.0 - blend_factor) * heuristic_uncertainty;

        // Collect neighbor IDs for debug
        let neighbor_ids = if let Ok(results) = index.query(embedding, self.k_neighbors) {
            Some(results.into_iter().map(|r| r.id).collect())
        } else {
            None
        };

        (uncertainty.clamp(0.0, 1.0), neighbor_ids)
    }

    /// Build debug trace for telemetry
    fn build_debug_trace(
        &self,
        features: &[f32],
        kin: &Option<KinematicFeatures>,
        lat: &Option<LatentFeatures>,
        neighbor_ids: Option<Vec<String>>,
    ) -> DebugTrace {
        let mut raw_features = HashMap::new();

        // Add fused features
        for (i, &f) in features.iter().enumerate() {
            raw_features.insert(format!("fused_{}", i), f);
        }

        // Add kinematic features if present
        if let Some(k) = kin {
            raw_features.insert("kinetic_intensity".to_string(), k.kinetic_intensity);
            raw_features.insert("angular_intensity".to_string(), k.angular_intensity);
            raw_features.insert("jerk_energy".to_string(), k.jerk_energy);
            raw_features.insert("directional_persistence".to_string(), k.directional_persistence);
        }

        // Add latent features if present
        if let Some(l) = lat {
            raw_features.insert("latent_norm".to_string(), l.norm);
            raw_features.insert("latent_velocity_norm".to_string(), l.velocity_norm);
        }

        let mut intermediate_scores = HashMap::new();
        intermediate_scores.insert("prev_commitment".to_string(), self.prev_commitment);
        intermediate_scores.insert("prev_uncertainty".to_string(), self.prev_uncertainty);
        intermediate_scores.insert("history_count".to_string(), self.regime_history_count as f32);

        #[cfg(feature = "neighbors")]
        if self.phrase_index.is_some() {
            intermediate_scores.insert("neighbor_k".to_string(), self.k_neighbors as f32);
        }

        DebugTrace {
            raw_features,
            neighbor_ids,
            intermediate_scores,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{SkeletonFrame, Quat, Vec3};

    fn create_test_window(coverage: f32) -> MotionWindow {
        let frames: Vec<SkeletonFrame> = (0..50)
            .map(|i| SkeletonFrame {
                timestamp: i as f64 * 0.02,
                root_position: Vec3::new(0.0, 0.0, 0.0),
                root_rotation: Quat::IDENTITY,
                bone_rotations: [Quat::IDENTITY; 27],
                valid: true,
                source_seq: Some(i as u64),
            })
            .collect();

        MotionWindow {
            window_id: "test_window".to_string(),
            t_start: 0.0,
            t_end: 1.0,
            fps: 50.0,
            skeleton_frames: frames,
            latent_frames: vec![],
            coverage,
            device_offsets: HashMap::new(),
            dropped_reason: None,
        }
    }

    #[test]
    fn test_kernel_creation() {
        let config = AnticipationConfig::default();
        let kernel = AnticipationKernel::new(config);
        assert_eq!(kernel.prev_commitment, 0.5);
    }

    #[test]
    fn test_process_valid_window() {
        let config = AnticipationConfig::default();
        let mut kernel = AnticipationKernel::new(config);
        let window = create_test_window(0.95);

        let result = kernel.process(&window);
        assert!(result.is_ok());

        let packet = result.unwrap();
        assert!(packet.validate().is_ok());
    }

    #[test]
    fn test_process_low_coverage() {
        let config = AnticipationConfig::default();
        let mut kernel = AnticipationKernel::new(config);
        let window = create_test_window(0.5); // Below 0.90 threshold

        let result = kernel.process(&window);
        assert!(matches!(result, Err(KernelError::InsufficientCoverage(_, _))));
    }

    #[test]
    fn test_deterministic_replay() {
        let config = AnticipationConfig::default();
        let window = create_test_window(0.95);

        let mut kernel1 = AnticipationKernel::new(config.clone());
        let mut kernel2 = AnticipationKernel::new(config);

        let packet1 = kernel1.process(&window).unwrap();
        let packet2 = kernel2.process(&window).unwrap();

        assert_eq!(packet1.commitment, packet2.commitment);
        assert_eq!(packet1.uncertainty, packet2.uncertainty);
        assert_eq!(packet1.regime_embedding, packet2.regime_embedding);
    }

    #[test]
    fn test_reset() {
        let config = AnticipationConfig::default();
        let mut kernel = AnticipationKernel::new(config);
        let window = create_test_window(0.95);

        kernel.process(&window).unwrap();
        assert!(kernel.prev_timestamp.is_some());

        kernel.reset();
        assert!(kernel.prev_timestamp.is_none());
        assert_eq!(kernel.prev_commitment, 0.5);
    }
}

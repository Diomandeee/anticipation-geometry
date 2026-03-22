//! Frozen configuration for cc-anticipation kernel
//!
//! Configuration is versioned and changes require careful consideration
//! to maintain replay determinism.

use serde::{Deserialize, Serialize};

/// Configuration for the anticipation kernel
///
/// This config is FROZEN per version. Changes require version bump
/// and may break replay determinism.
///
/// # Example
///
/// ```
/// use cc_anticipation::AnticipationConfig;
///
/// let config = AnticipationConfig::default();
/// assert_eq!(config.fps, 50.0);
/// assert_eq!(config.regime_embedding_dim, 64);
/// ```
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AnticipationConfig {
    /// Configuration version (for replay compatibility)
    pub version: String,

    // -------------------------------------------------------------------------
    // Window Parameters (must match aligner)
    // -------------------------------------------------------------------------

    /// Canonical frame rate in Hz (default: 50.0)
    pub fps: f32,

    /// Window duration in seconds (default: 1.0 = 50 frames at 50Hz)
    pub window_seconds: f32,

    // -------------------------------------------------------------------------
    // Embedding Dimensions
    // -------------------------------------------------------------------------

    /// Regime embedding dimension (default: 64)
    pub regime_embedding_dim: usize,

    /// Constraint vector dimension (default: 8)
    pub constraint_vector_dim: usize,

    /// Derivative summary dimension (default: 8)
    pub derivative_summary_dim: usize,

    // -------------------------------------------------------------------------
    // Neighbor Search (v1, requires "neighbors" feature)
    // -------------------------------------------------------------------------

    /// Number of neighbors to retrieve (default: 10)
    pub neighbor_k: usize,

    /// Minimum similarity threshold for neighbors (default: 0.3)
    pub min_similarity: f32,

    // -------------------------------------------------------------------------
    // Thresholds
    // -------------------------------------------------------------------------

    /// Minimum coverage to emit packet (default: 0.90)
    pub min_coverage: f32,

    /// Kinetic energy threshold for stillness detection (default: 0.01 m/s)
    pub stillness_threshold: f32,

    // -------------------------------------------------------------------------
    // Smoothing Parameters
    // -------------------------------------------------------------------------

    /// EMA coefficient for transition_pressure smoothing (default: 0.3)
    ///
    /// Higher = more responsive, Lower = more stable
    pub transition_pressure_alpha: f32,

    /// History duration for novelty computation in seconds (default: 10.0)
    pub novelty_history_seconds: f32,

    // -------------------------------------------------------------------------
    // Debug/Telemetry
    // -------------------------------------------------------------------------

    /// Whether to emit debug traces in packets (default: true)
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

impl AnticipationConfig {
    /// Create a new configuration with default values
    pub fn new() -> Self {
        Self::default()
    }

    /// Create configuration optimized for low latency
    ///
    /// Shorter history, faster response, less smoothing.
    pub fn low_latency() -> Self {
        Self {
            version: "0.1.0".to_string(),
            fps: 50.0,
            window_seconds: 0.5, // Shorter window
            regime_embedding_dim: 64,
            constraint_vector_dim: 8,
            derivative_summary_dim: 8,
            neighbor_k: 5,      // Fewer neighbors
            min_similarity: 0.4,
            min_coverage: 0.85, // More tolerant
            stillness_threshold: 0.01,
            transition_pressure_alpha: 0.5, // More responsive
            novelty_history_seconds: 5.0,   // Shorter history
            emit_debug: false,  // Skip debug overhead
        }
    }

    /// Create configuration optimized for analysis/replay
    ///
    /// Full debug output, strict thresholds.
    pub fn analysis() -> Self {
        Self {
            version: "0.1.0".to_string(),
            fps: 50.0,
            window_seconds: 2.0, // Longer window for stability
            regime_embedding_dim: 128, // Higher resolution
            constraint_vector_dim: 16,
            derivative_summary_dim: 16,
            neighbor_k: 20,     // More neighbors
            min_similarity: 0.2,
            min_coverage: 0.95, // Strict
            stillness_threshold: 0.005,
            transition_pressure_alpha: 0.2, // More stable
            novelty_history_seconds: 20.0,  // Longer history
            emit_debug: true,
        }
    }

    /// Compute number of frames in a window
    pub fn window_frame_count(&self) -> usize {
        (self.window_seconds * self.fps) as usize
    }

    /// Compute frame duration in seconds
    pub fn frame_duration(&self) -> f32 {
        1.0 / self.fps
    }

    /// Compute novelty history buffer size in frames
    pub fn novelty_history_len(&self) -> usize {
        (self.novelty_history_seconds * self.fps) as usize
    }

    /// Validate configuration
    pub fn validate(&self) -> Result<(), String> {
        if self.fps <= 0.0 {
            return Err("fps must be positive".to_string());
        }
        if self.window_seconds <= 0.0 {
            return Err("window_seconds must be positive".to_string());
        }
        if self.regime_embedding_dim == 0 {
            return Err("regime_embedding_dim must be positive".to_string());
        }
        if self.min_coverage < 0.0 || self.min_coverage > 1.0 {
            return Err("min_coverage must be in [0, 1]".to_string());
        }
        if self.transition_pressure_alpha < 0.0 || self.transition_pressure_alpha > 1.0 {
            return Err("transition_pressure_alpha must be in [0, 1]".to_string());
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = AnticipationConfig::default();
        assert_eq!(config.fps, 50.0);
        assert_eq!(config.window_frame_count(), 50);
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_low_latency_config() {
        let config = AnticipationConfig::low_latency();
        assert_eq!(config.window_frame_count(), 25); // 0.5s at 50Hz
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_analysis_config() {
        let config = AnticipationConfig::analysis();
        assert_eq!(config.regime_embedding_dim, 128);
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_invalid_config() {
        let config = AnticipationConfig {
            fps: -1.0,
            ..Default::default()
        };
        assert!(config.validate().is_err());
    }
}

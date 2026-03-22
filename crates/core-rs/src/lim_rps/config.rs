//! LIM-RPS Configuration.
//!
//! Configuration for the Lipschitz Implicit Map solver. These parameters
//! match the Python LIMRPSConfig exactly for cross-language compatibility.

use std::collections::HashMap;

/// Proximal operator mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ProxMode {
    /// L2 proximal (elementwise shrinkage)
    #[default]
    L2,
    /// Group L2 proximal (shrinks groups together)
    GroupL2,
    /// No proximal regularization
    None,
}

/// LIM-RPS solver configuration.
///
/// Matches the Python `LIMRPSConfig` class exactly for interoperability.
/// All defaults match the Python implementation.
#[derive(Debug, Clone)]
pub struct LimRpsConfig {
    // ========================================================================
    // Iteration parameters
    // ========================================================================

    /// Maximum number of fixed-point iterations.
    /// Default: 4 (sufficient for well-conditioned operators)
    pub max_iters: usize,

    /// Forward step size γ for fixed-point iteration.
    /// Smaller values = more stable, larger = faster convergence.
    /// Must satisfy γ * L < 1 for convergence (L = Lipschitz constant).
    /// Default: 0.5
    pub step_size: f32,

    /// Early stopping threshold.
    /// Stop when ||x_k - x_{k-1}|| < eps.
    /// Set to 0.0 to disable early stopping.
    /// Default: 1e-4
    pub early_stop_eps: f32,

    /// Minimum iterations before early stopping is checked.
    /// Default: 2
    pub early_stop_min_iters: usize,

    // ========================================================================
    // Operator architecture
    // ========================================================================

    /// Hidden dimension for the CrossModalOperator MLP.
    /// Default: 128
    pub hidden_dim: usize,

    /// Number of hidden layers in the operator.
    /// Default: 2
    pub num_layers: usize,

    /// Number of power iterations for spectral normalization.
    /// More iterations = tighter bound, but slower.
    /// Default: 1 (sufficient after warm-up)
    pub spectral_iters: usize,

    // ========================================================================
    // Proximal parameters
    // ========================================================================

    /// Proximal operator mode.
    /// Default: L2
    pub prox_mode: ProxMode,

    /// Proximal regularization strength τ.
    /// Weight for pulling toward encoder latents.
    /// Default: 0.05
    pub prox_tau: f32,

    // ========================================================================
    // Box constraints
    // ========================================================================

    /// Lower bound for box projection.
    /// Default: -10.0
    pub box_lower: f32,

    /// Upper bound for box projection.
    /// Default: 10.0
    pub box_upper: f32,

    /// Maximum state norm (for gradient clipping).
    /// Default: 10.0
    pub max_state_norm: f32,

    // ========================================================================
    // Optional features
    // ========================================================================

    /// Whether to use a learned metric for distance computation.
    /// Default: false
    pub use_metric: bool,

    /// Whether to use a learned step field (spatially varying step size).
    /// Default: false
    pub use_step_field: bool,

    /// Temporal consistency weight (for smoothing across frames).
    /// Default: 0.0 (disabled)
    pub temporal_lambda: f32,

    /// EMA smoothing factor (0 = disabled, 1 = no update).
    /// Default: 0.0
    pub ema_alpha: f32,
}

impl Default for LimRpsConfig {
    fn default() -> Self {
        Self {
            // Iteration
            max_iters: 4,
            step_size: 0.5,
            early_stop_eps: 1e-4,
            early_stop_min_iters: 2,

            // Architecture
            hidden_dim: 128,
            num_layers: 2,
            spectral_iters: 1,

            // Proximal
            prox_mode: ProxMode::L2,
            prox_tau: 0.05,

            // Box
            box_lower: -10.0,
            box_upper: 10.0,
            max_state_norm: 10.0,

            // Optional
            use_metric: false,
            use_step_field: false,
            temporal_lambda: 0.0,
            ema_alpha: 0.0,
        }
    }
}

impl LimRpsConfig {
    /// Create a new config with default values.
    pub fn new() -> Self {
        Self::default()
    }

    /// Builder pattern: set max iterations.
    pub fn with_max_iters(mut self, max_iters: usize) -> Self {
        self.max_iters = max_iters;
        self
    }

    /// Builder pattern: set step size.
    pub fn with_step_size(mut self, step_size: f32) -> Self {
        self.step_size = step_size;
        self
    }

    /// Builder pattern: set hidden dimension.
    pub fn with_hidden_dim(mut self, hidden_dim: usize) -> Self {
        self.hidden_dim = hidden_dim;
        self
    }

    /// Builder pattern: set number of layers.
    pub fn with_num_layers(mut self, num_layers: usize) -> Self {
        self.num_layers = num_layers;
        self
    }

    /// Builder pattern: set proximal mode.
    pub fn with_prox_mode(mut self, prox_mode: ProxMode) -> Self {
        self.prox_mode = prox_mode;
        self
    }

    /// Builder pattern: set proximal tau.
    pub fn with_prox_tau(mut self, prox_tau: f32) -> Self {
        self.prox_tau = prox_tau;
        self
    }

    /// Builder pattern: set box constraints.
    pub fn with_box_constraints(mut self, lower: f32, upper: f32) -> Self {
        self.box_lower = lower;
        self.box_upper = upper;
        self
    }

    /// Builder pattern: set temporal lambda.
    pub fn with_temporal_lambda(mut self, temporal_lambda: f32) -> Self {
        self.temporal_lambda = temporal_lambda;
        self
    }

    /// Builder pattern: enable EMA smoothing.
    pub fn with_ema(mut self, alpha: f32) -> Self {
        self.ema_alpha = alpha;
        self
    }

    /// Validate configuration.
    pub fn validate(&self) -> Result<(), String> {
        if self.step_size <= 0.0 || self.step_size > 1.0 {
            return Err(format!(
                "step_size must be in (0, 1], got {}",
                self.step_size
            ));
        }

        if self.box_lower >= self.box_upper {
            return Err(format!(
                "box_lower ({}) must be < box_upper ({})",
                self.box_lower, self.box_upper
            ));
        }

        if self.max_iters == 0 {
            return Err("max_iters must be > 0".to_string());
        }

        if self.hidden_dim == 0 {
            return Err("hidden_dim must be > 0".to_string());
        }

        if self.prox_tau < 0.0 {
            return Err(format!("prox_tau must be >= 0, got {}", self.prox_tau));
        }

        Ok(())
    }

    /// Convert to a flat dictionary for serialization.
    pub fn to_dict(&self) -> HashMap<String, f32> {
        let mut d = HashMap::new();
        d.insert("max_iters".to_string(), self.max_iters as f32);
        d.insert("step_size".to_string(), self.step_size);
        d.insert("early_stop_eps".to_string(), self.early_stop_eps);
        d.insert("hidden_dim".to_string(), self.hidden_dim as f32);
        d.insert("num_layers".to_string(), self.num_layers as f32);
        d.insert("spectral_iters".to_string(), self.spectral_iters as f32);
        d.insert("prox_tau".to_string(), self.prox_tau);
        d.insert("box_lower".to_string(), self.box_lower);
        d.insert("box_upper".to_string(), self.box_upper);
        d.insert("max_state_norm".to_string(), self.max_state_norm);
        d.insert("temporal_lambda".to_string(), self.temporal_lambda);
        d.insert("ema_alpha".to_string(), self.ema_alpha);
        d
    }

    /// Load from a flat dictionary.
    pub fn from_dict(d: &HashMap<String, f32>) -> Self {
        let mut config = Self::default();

        if let Some(&v) = d.get("max_iters") {
            config.max_iters = v as usize;
        }
        if let Some(&v) = d.get("step_size") {
            config.step_size = v;
        }
        if let Some(&v) = d.get("early_stop_eps") {
            config.early_stop_eps = v;
        }
        if let Some(&v) = d.get("hidden_dim") {
            config.hidden_dim = v as usize;
        }
        if let Some(&v) = d.get("num_layers") {
            config.num_layers = v as usize;
        }
        if let Some(&v) = d.get("spectral_iters") {
            config.spectral_iters = v as usize;
        }
        if let Some(&v) = d.get("prox_tau") {
            config.prox_tau = v;
        }
        if let Some(&v) = d.get("box_lower") {
            config.box_lower = v;
        }
        if let Some(&v) = d.get("box_upper") {
            config.box_upper = v;
        }
        if let Some(&v) = d.get("max_state_norm") {
            config.max_state_norm = v;
        }
        if let Some(&v) = d.get("temporal_lambda") {
            config.temporal_lambda = v;
        }
        if let Some(&v) = d.get("ema_alpha") {
            config.ema_alpha = v;
        }

        config
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = LimRpsConfig::default();
        assert_eq!(config.max_iters, 4);
        assert_eq!(config.step_size, 0.5);
        assert_eq!(config.hidden_dim, 128);
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_builder_pattern() {
        let config = LimRpsConfig::new()
            .with_max_iters(10)
            .with_step_size(0.3)
            .with_hidden_dim(256)
            .with_box_constraints(-5.0, 5.0);

        assert_eq!(config.max_iters, 10);
        assert_eq!(config.step_size, 0.3);
        assert_eq!(config.hidden_dim, 256);
        assert_eq!(config.box_lower, -5.0);
        assert_eq!(config.box_upper, 5.0);
    }

    #[test]
    fn test_validation() {
        let mut config = LimRpsConfig::default();

        // Invalid step size
        config.step_size = 0.0;
        assert!(config.validate().is_err());
        config.step_size = 0.5;

        // Invalid box constraints
        config.box_lower = 10.0;
        config.box_upper = 5.0;
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_serialization_roundtrip() {
        let config = LimRpsConfig::new()
            .with_max_iters(8)
            .with_hidden_dim(64);

        let dict = config.to_dict();
        let restored = LimRpsConfig::from_dict(&dict);

        assert_eq!(config.max_iters, restored.max_iters);
        assert_eq!(config.hidden_dim, restored.hidden_dim);
    }
}

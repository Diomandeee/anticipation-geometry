//! LIM-RPS Equilibrium Solver.
//!
//! The core fixed-point iteration solver for finding equilibrium latents.
//!
//! # Algorithm
//!
//! Given encoder latents `enc` and operator `B`, find equilibrium `z*`:
//!
//! ```text
//! z_0 = enc (or warm start)
//! for k = 0..K:
//!     z_{k+1} = prox(z_k - γ * B(z_k), enc)
//!     z_{k+1} = box_project(z_{k+1})
//! return z_K
//! ```
//!
//! The proximal step pulls toward encoder latents, ensuring the equilibrium
//! stays close to the raw encoder output while allowing cross-modal fusion.

use super::config::{LimRpsConfig, ProxMode};
use super::operator::{CrossModalOperator, LimRpsOperator};
use crate::proximal_ops::{
    box_project_inplace, clip_state_norm_inplace, ema_inplace, l2_distance,
    l2_prox_pull_inplace,
};

/// Result from LIM-RPS solver.
#[derive(Debug, Clone)]
pub struct LimRpsResult {
    /// Equilibrium latent z*.
    pub x_star: Vec<f32>,
    /// Per-iteration residuals ||z_k - z_{k-1}||.
    pub residuals: Vec<f32>,
    /// Number of iterations used.
    pub k_used: usize,
    /// Whether early stopping triggered.
    pub converged: bool,
    /// Final residual ||z_K - z_{K-1}||.
    pub final_residual: f32,
    /// Mean residual across iterations.
    pub mean_residual: f32,
}

impl Default for LimRpsResult {
    fn default() -> Self {
        Self {
            x_star: Vec::new(),
            residuals: Vec::new(),
            k_used: 0,
            converged: false,
            final_residual: f32::MAX,
            mean_residual: f32::MAX,
        }
    }
}

/// LIM-RPS Equilibrium Solver.
///
/// Wraps a CrossModalOperator and provides fixed-point iteration
/// to find equilibrium latents.
#[derive(Debug)]
pub struct LimRpsSolver {
    /// Configuration.
    config: LimRpsConfig,
    /// Cross-modal operator.
    operator: CrossModalOperator,
    /// Total latent dimension.
    total_dim: usize,
    /// Previous equilibrium (for temporal smoothing).
    z_prev: Option<Vec<f32>>,
}

impl LimRpsSolver {
    /// Create a new LIM-RPS solver.
    ///
    /// # Arguments
    /// * `total_dim` - Dimension of the latent space
    /// * `config` - Solver configuration
    pub fn new(total_dim: usize, config: LimRpsConfig) -> Self {
        let operator = CrossModalOperator::new(total_dim, &config);

        Self {
            config,
            operator,
            total_dim,
            z_prev: None,
        }
    }

    /// Create with a pre-built operator.
    pub fn with_operator(operator: CrossModalOperator, config: LimRpsConfig) -> Self {
        let total_dim = operator.input_dim();

        Self {
            config,
            operator,
            total_dim,
            z_prev: None,
        }
    }

    /// Solve for equilibrium latent.
    ///
    /// # Arguments
    /// * `encoder_latent` - Raw encoder output
    /// * `z_init` - Optional initial state (defaults to encoder_latent)
    ///
    /// # Returns
    /// LimRpsResult with equilibrium latent and diagnostics
    pub fn solve(&self, encoder_latent: &[f32], z_init: Option<&[f32]>) -> LimRpsResult {
        debug_assert_eq!(encoder_latent.len(), self.total_dim);

        let mut z = z_init.map(|v| v.to_vec()).unwrap_or_else(|| encoder_latent.to_vec());
        let mut z_prev_iter = vec![0.0; self.total_dim];
        let mut residuals = Vec::with_capacity(self.config.max_iters);
        let mut k_used = 0;
        let mut converged = false;

        for k in 0..self.config.max_iters {
            // Save previous state
            z_prev_iter.copy_from_slice(&z);

            // 1) Apply operator: b = B(z)
            let b = self.operator.forward(&z);

            // 2) Forward step: z = z - γ * b
            for (zi, &bi) in z.iter_mut().zip(b.iter()) {
                *zi -= self.config.step_size * bi;
            }

            // 3) Proximal step
            match self.config.prox_mode {
                ProxMode::L2 => {
                    if self.config.prox_tau > 0.0 {
                        l2_prox_pull_inplace(&mut z, encoder_latent, self.config.prox_tau);
                    }
                }
                ProxMode::GroupL2 => {
                    // For group L2, we'd need group definitions
                    // For now, fall back to elementwise L2
                    if self.config.prox_tau > 0.0 {
                        l2_prox_pull_inplace(&mut z, encoder_latent, self.config.prox_tau);
                    }
                }
                ProxMode::None => {}
            }

            // 4) Box projection
            box_project_inplace(&mut z, self.config.box_lower, self.config.box_upper);

            // 5) Clip state norm
            clip_state_norm_inplace(&mut z, self.config.max_state_norm);

            // 6) Optional EMA smoothing with iteration history
            if self.config.ema_alpha > 0.0 {
                ema_inplace(&mut z, &z_prev_iter, self.config.ema_alpha);
            }

            // 7) Compute residual
            let residual = l2_distance(&z, &z_prev_iter);
            residuals.push(residual);
            k_used = k + 1;

            // 8) Early stopping
            if self.config.early_stop_eps > 0.0
                && k >= self.config.early_stop_min_iters
                && residual < self.config.early_stop_eps
            {
                converged = true;
                break;
            }
        }

        // Temporal smoothing with previous equilibrium
        if self.config.temporal_lambda > 0.0 {
            if let Some(ref z_temporal_prev) = self.z_prev {
                for (zi, &pi) in z.iter_mut().zip(z_temporal_prev.iter()) {
                    *zi = (1.0 - self.config.temporal_lambda) * *zi
                        + self.config.temporal_lambda * pi;
                }
            }
        }

        // Compute summary stats
        let final_residual = *residuals.last().unwrap_or(&f32::MAX);
        let mean_residual = if residuals.is_empty() {
            f32::MAX
        } else {
            residuals.iter().sum::<f32>() / residuals.len() as f32
        };

        LimRpsResult {
            x_star: z,
            residuals,
            k_used,
            converged,
            final_residual,
            mean_residual,
        }
    }

    /// Solve with warm start from previous equilibrium.
    ///
    /// Uses the previous equilibrium as initial state and updates
    /// internal state for temporal consistency.
    pub fn solve_warm(&mut self, encoder_latent: &[f32]) -> LimRpsResult {
        let z_init = self.z_prev.as_deref();
        let result = self.solve(encoder_latent, z_init);

        // Update temporal state
        self.z_prev = Some(result.x_star.clone());

        result
    }

    /// Reset temporal state.
    pub fn reset(&mut self) {
        self.z_prev = None;
    }

    /// Solve batch of encoder latents.
    ///
    /// # Arguments
    /// * `encoder_batch` - Batch of encoder latents
    ///
    /// # Returns
    /// Vector of LimRpsResult, one per input
    pub fn solve_batch(&self, encoder_batch: &[Vec<f32>]) -> Vec<LimRpsResult> {
        encoder_batch
            .iter()
            .map(|enc| self.solve(enc, None))
            .collect()
    }

    /// Parallel batch solve using Rayon.
    #[cfg(feature = "rayon")]
    pub fn solve_batch_parallel(&self, encoder_batch: &[Vec<f32>]) -> Vec<LimRpsResult> {
        use rayon::prelude::*;

        encoder_batch
            .par_iter()
            .map(|enc| self.solve(enc, None))
            .collect()
    }

    /// Get reference to operator.
    pub fn operator(&self) -> &CrossModalOperator {
        &self.operator
    }

    /// Get mutable reference to operator.
    pub fn operator_mut(&mut self) -> &mut CrossModalOperator {
        &mut self.operator
    }

    /// Get configuration.
    pub fn config(&self) -> &LimRpsConfig {
        &self.config
    }

    /// Update configuration.
    pub fn set_config(&mut self, config: LimRpsConfig) {
        self.config = config;
    }

    /// Get total dimension.
    pub fn total_dim(&self) -> usize {
        self.total_dim
    }

    /// Load operator weights from state dict.
    pub fn load_operator_weights(&mut self, state_dict: &std::collections::HashMap<String, Vec<f32>>) {
        self.operator.load_state_dict(state_dict);
    }

    /// Export operator weights to state dict.
    pub fn operator_state_dict(&self) -> std::collections::HashMap<String, Vec<f32>> {
        self.operator.state_dict()
    }

    /// Refresh operator spectral normalization.
    pub fn refresh_spectral_state(&mut self, n_iters: usize) {
        self.operator.refresh_spectral_state(n_iters);
    }
}

/// Simple solver using generic operator trait.
pub fn solve_with_operator<O: LimRpsOperator>(
    encoder_latent: &[f32],
    z_init: Option<&[f32]>,
    operator: &O,
    config: &LimRpsConfig,
) -> LimRpsResult {
    let total_dim = encoder_latent.len();
    debug_assert_eq!(operator.input_dim(), total_dim);
    debug_assert_eq!(operator.output_dim(), total_dim);

    let mut z = z_init.map(|v| v.to_vec()).unwrap_or_else(|| encoder_latent.to_vec());
    let mut z_prev_iter = vec![0.0; total_dim];
    let mut residuals = Vec::with_capacity(config.max_iters);
    let mut k_used = 0;
    let mut converged = false;

    for k in 0..config.max_iters {
        z_prev_iter.copy_from_slice(&z);

        // Apply operator
        let b = operator.apply(&z);

        // Forward step
        for (zi, &bi) in z.iter_mut().zip(b.iter()) {
            *zi -= config.step_size * bi;
        }

        // Proximal step
        if config.prox_tau > 0.0 && config.prox_mode != ProxMode::None {
            l2_prox_pull_inplace(&mut z, encoder_latent, config.prox_tau);
        }

        // Box projection
        box_project_inplace(&mut z, config.box_lower, config.box_upper);

        // Clip state norm
        clip_state_norm_inplace(&mut z, config.max_state_norm);

        // EMA
        if config.ema_alpha > 0.0 {
            ema_inplace(&mut z, &z_prev_iter, config.ema_alpha);
        }

        // Residual
        let residual = l2_distance(&z, &z_prev_iter);
        residuals.push(residual);
        k_used = k + 1;

        // Early stopping
        if config.early_stop_eps > 0.0
            && k >= config.early_stop_min_iters
            && residual < config.early_stop_eps
        {
            converged = true;
            break;
        }
    }

    let final_residual = *residuals.last().unwrap_or(&f32::MAX);
    let mean_residual = if residuals.is_empty() {
        f32::MAX
    } else {
        residuals.iter().sum::<f32>() / residuals.len() as f32
    };

    LimRpsResult {
        x_star: z,
        residuals,
        k_used,
        converged,
        final_residual,
        mean_residual,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::operator::IdentityOperator;

    #[test]
    fn test_lim_rps_solver_new() {
        let config = LimRpsConfig::default();
        let solver = LimRpsSolver::new(16, config);

        assert_eq!(solver.total_dim(), 16);
    }

    #[test]
    fn test_lim_rps_solver_solve() {
        let config = LimRpsConfig::new()
            .with_max_iters(10)
            .with_hidden_dim(32);
        let solver = LimRpsSolver::new(8, config);

        let encoder = vec![1.0, 0.5, 0.0, -0.5, 0.3, -0.3, 0.1, -0.1];
        let result = solver.solve(&encoder, None);

        assert_eq!(result.x_star.len(), 8);
        assert!(result.k_used > 0);
        assert!(!result.residuals.is_empty());
    }

    #[test]
    fn test_lim_rps_solver_convergence() {
        let config = LimRpsConfig::new()
            .with_max_iters(50)
            .with_step_size(0.3)
            .with_prox_tau(0.1);
        let mut solver = LimRpsSolver::new(8, config);
        solver.refresh_spectral_state(50);

        let encoder = vec![1.0, 0.5, 0.0, -0.5, 0.3, -0.3, 0.1, -0.1];
        let result = solver.solve(&encoder, None);

        // Residuals should decrease
        for i in 1..result.residuals.len().min(5) {
            assert!(
                result.residuals[i] <= result.residuals[i - 1] * 1.5 + 0.01,
                "Residuals should generally decrease: {} > {}",
                result.residuals[i],
                result.residuals[i - 1]
            );
        }
    }

    #[test]
    fn test_lim_rps_solver_warm_start() {
        let config = LimRpsConfig::new().with_max_iters(5);
        let mut solver = LimRpsSolver::new(4, config);

        let encoder1 = vec![1.0, 0.0, 0.0, 0.0];
        let result1 = solver.solve_warm(&encoder1);

        // Second solve with similar input should start from previous
        let encoder2 = vec![1.1, 0.0, 0.0, 0.0];
        let result2 = solver.solve_warm(&encoder2);

        // With warm start, should converge faster (fewer effective changes)
        // Just verify it works without error
        assert_eq!(result1.x_star.len(), 4);
        assert_eq!(result2.x_star.len(), 4);
    }

    #[test]
    fn test_lim_rps_solver_batch() {
        let config = LimRpsConfig::new().with_max_iters(5);
        let solver = LimRpsSolver::new(4, config);

        let batch = vec![
            vec![1.0, 0.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0, 0.0],
            vec![0.0, 0.0, 1.0, 0.0],
        ];

        let results = solver.solve_batch(&batch);

        assert_eq!(results.len(), 3);
        for r in &results {
            assert_eq!(r.x_star.len(), 4);
        }
    }

    #[test]
    fn test_solve_with_identity_operator() {
        let config = LimRpsConfig::new()
            .with_max_iters(10)
            .with_prox_tau(0.1);
        let operator = IdentityOperator::new(4);

        let encoder = vec![1.0, 2.0, 3.0, 4.0];
        let result = solve_with_operator(&encoder, None, &operator, &config);

        // With identity operator and prox pull, should converge toward encoder
        assert_eq!(result.x_star.len(), 4);
    }

    #[test]
    fn test_early_stopping() {
        let config = LimRpsConfig::new()
            .with_max_iters(100)
            .with_prox_tau(0.5); // Strong pull = fast convergence
        let solver = LimRpsSolver::new(4, config);

        let encoder = vec![0.5, 0.5, 0.5, 0.5];
        let result = solver.solve(&encoder, Some(&encoder)); // Start at equilibrium

        // Should stop early since we start at equilibrium
        assert!(
            result.k_used < 100 || result.converged,
            "Should converge: k_used={}, converged={}",
            result.k_used,
            result.converged
        );
    }

    #[test]
    fn test_box_projection() {
        let config = LimRpsConfig::new()
            .with_max_iters(5)
            .with_box_constraints(-1.0, 1.0);
        let solver = LimRpsSolver::new(4, config);

        let encoder = vec![10.0, -10.0, 0.5, 0.0];
        let result = solver.solve(&encoder, None);

        // All values should be within box constraints
        for &v in &result.x_star {
            assert!(v >= -1.0 && v <= 1.0, "Value {} outside box [-1, 1]", v);
        }
    }

    #[test]
    fn test_temporal_smoothing() {
        let config = LimRpsConfig::new()
            .with_max_iters(5)
            .with_temporal_lambda(0.5);
        let mut solver = LimRpsSolver::new(4, config);

        let encoder1 = vec![1.0, 0.0, 0.0, 0.0];
        let result1 = solver.solve_warm(&encoder1);

        let encoder2 = vec![-1.0, 0.0, 0.0, 0.0];
        let result2 = solver.solve_warm(&encoder2);

        // With temporal smoothing, result2 should be influenced by result1
        // The first component should be somewhere between the two
        // (exact value depends on many factors)
        assert_eq!(result1.x_star.len(), 4);
        assert_eq!(result2.x_star.len(), 4);
    }
}

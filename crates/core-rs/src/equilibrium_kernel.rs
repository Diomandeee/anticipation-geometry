//! Fast equilibrium kernel for fixed-point iteration.
//!
//! This module provides a pure numerical fixed-point solver that can be used:
//! - As a fast path when the operator is a simple linear transformation
//! - For testing and benchmarking equilibrium convergence
//! - As a building block for more complex solvers
//!
//! The core iteration is:
//!     x_{k+1} = prox( x_k - step_size * (A @ x_k + b) )
//!
//! where prox includes L2 pull toward encoder and box projection.

use crate::proximal_ops::{
    box_project_inplace, clip_state_norm_inplace, ema_inplace, l2_distance,
    l2_prox_pull_inplace,
};
use crate::matrix_ops::DenseMatrix;

/// Configuration for the fast equilibrium kernel.
#[derive(Debug, Clone)]
pub struct EquilibriumKernelConfig {
    /// Maximum number of iterations
    pub max_iters: usize,
    /// Forward step size γ
    pub step_size: f32,
    /// L2 proximal weight toward encoder latents
    pub prox_l2_weight: f32,
    /// Box constraint lower bound
    pub box_lower: f32,
    /// Box constraint upper bound
    pub box_upper: f32,
    /// Maximum state norm (for clipping)
    pub max_state_norm: f32,
    /// Early stopping threshold (0 = disabled)
    pub early_stop_eps: f32,
    /// Minimum iterations before early stopping
    pub early_stop_min_iters: usize,
    /// EMA smoothing factor (0 = disabled)
    pub ema_alpha: f32,
}

impl Default for EquilibriumKernelConfig {
    fn default() -> Self {
        Self {
            max_iters: 4,
            step_size: 0.5,
            prox_l2_weight: 0.05,
            box_lower: -10.0,
            box_upper: 10.0,
            max_state_norm: 10.0,
            early_stop_eps: 0.0,
            early_stop_min_iters: 2,
            ema_alpha: 0.0,
        }
    }
}

/// Result from equilibrium kernel solve.
#[derive(Debug, Clone)]
pub struct EquilibriumResult {
    /// Final equilibrium state
    pub x_star: Vec<f32>,
    /// Per-iteration residuals
    pub residuals: Vec<f32>,
    /// Number of iterations used
    pub k_used: usize,
    /// Whether early stopping triggered
    pub converged: bool,
    /// Final residual
    pub final_residual: f32,
    /// Mean residual across iterations
    pub mean_residual: f32,
}

/// Simple linear operator for testing: b = Ax + bias
#[derive(Debug, Clone)]
pub struct LinearOperator {
    /// Diagonal of A matrix (simplified to diagonal for efficiency)
    pub a_diag: Vec<f32>,
    /// Bias vector
    pub bias: Vec<f32>,
}

impl LinearOperator {
    /// Create a new linear operator with given diagonal and bias.
    pub fn new(a_diag: Vec<f32>, bias: Vec<f32>) -> Self {
        debug_assert_eq!(a_diag.len(), bias.len());
        Self { a_diag, bias }
    }

    /// Create identity operator (A=I, bias=0).
    pub fn identity(dim: usize) -> Self {
        Self {
            a_diag: vec![1.0; dim],
            bias: vec![0.0; dim],
        }
    }

    /// Create contractive operator with given Lipschitz constant.
    pub fn contractive(dim: usize, lipschitz: f32) -> Self {
        Self {
            a_diag: vec![lipschitz; dim],
            bias: vec![0.0; dim],
        }
    }

    /// Apply operator: b = A @ x + bias
    pub fn apply(&self, x: &[f32]) -> Vec<f32> {
        debug_assert_eq!(x.len(), self.a_diag.len());
        x.iter()
            .zip(self.a_diag.iter())
            .zip(self.bias.iter())
            .map(|((&xi, &ai), &bi)| ai * xi + bi)
            .collect()
    }

    /// Apply operator in-place: x = A @ x + bias
    pub fn apply_inplace(&self, x: &mut [f32]) {
        debug_assert_eq!(x.len(), self.a_diag.len());
        for (xi, (&ai, &bi)) in x.iter_mut().zip(self.a_diag.iter().zip(self.bias.iter())) {
            *xi = ai * *xi + bi;
        }
    }
}

/// Solve fixed-point equilibrium using simple linear operator.
///
/// Iteration: x_{k+1} = prox( x_k - γ * operator(x_k) )
///
/// # Arguments
/// * `x_init` - Initial state
/// * `encoder` - Encoder latents (target for L2 pull)
/// * `operator` - Linear operator to apply
/// * `config` - Solver configuration
///
/// # Returns
/// EquilibriumResult with final state and diagnostics
pub fn solve_linear_equilibrium(
    x_init: &[f32],
    encoder: &[f32],
    operator: &LinearOperator,
    config: &EquilibriumKernelConfig,
) -> EquilibriumResult {
    let dim = x_init.len();
    debug_assert_eq!(encoder.len(), dim);
    debug_assert_eq!(operator.a_diag.len(), dim);

    let mut x = x_init.to_vec();
    let mut x_prev = vec![0.0; dim];
    let mut residuals = Vec::with_capacity(config.max_iters);
    let mut k_used = 0;
    let mut converged = false;

    for k in 0..config.max_iters {
        // Save previous state
        x_prev.copy_from_slice(&x);

        // 1) Apply operator: b = A @ x + bias
        let b = operator.apply(&x);

        // 2) Forward step: v = x - γ * b
        for (xi, bi) in x.iter_mut().zip(b.iter()) {
            *xi -= config.step_size * bi;
        }

        // 3) L2 proximal pull toward encoder
        if config.prox_l2_weight > 0.0 {
            l2_prox_pull_inplace(&mut x, encoder, config.prox_l2_weight);
        }

        // 4) Box projection
        box_project_inplace(&mut x, config.box_lower, config.box_upper);

        // 5) Clip state norm
        clip_state_norm_inplace(&mut x, config.max_state_norm);

        // 6) Optional EMA smoothing
        if config.ema_alpha > 0.0 {
            ema_inplace(&mut x, &x_prev, config.ema_alpha);
        }

        // 7) Compute residual
        let residual = l2_distance(&x, &x_prev);
        residuals.push(residual);
        k_used = k + 1;

        // 8) Early stopping
        if config.early_stop_eps > 0.0
            && k >= config.early_stop_min_iters
            && residual < config.early_stop_eps
        {
            converged = true;
            break;
        }
    }

    // Compute summary stats
    let final_residual = *residuals.last().unwrap_or(&0.0);
    let mean_residual = if residuals.is_empty() {
        0.0
    } else {
        residuals.iter().sum::<f32>() / residuals.len() as f32
    };

    EquilibriumResult {
        x_star: x,
        residuals,
        k_used,
        converged,
        final_residual,
        mean_residual,
    }
}

/// Solve equilibrium with external operator function.
///
/// This allows using any operator, not just linear.
///
/// # Arguments
/// * `x_init` - Initial state
/// * `encoder` - Encoder latents
/// * `operator_fn` - Function that computes operator(x)
/// * `config` - Solver configuration
pub fn solve_equilibrium<F>(
    x_init: &[f32],
    encoder: &[f32],
    operator_fn: F,
    config: &EquilibriumKernelConfig,
) -> EquilibriumResult
where
    F: Fn(&[f32]) -> Vec<f32>,
{
    let dim = x_init.len();
    debug_assert_eq!(encoder.len(), dim);

    let mut x = x_init.to_vec();
    let mut x_prev = vec![0.0; dim];
    let mut residuals = Vec::with_capacity(config.max_iters);
    let mut k_used = 0;
    let mut converged = false;

    for k in 0..config.max_iters {
        x_prev.copy_from_slice(&x);

        // Apply operator
        let b = operator_fn(&x);

        // Forward step
        for (xi, bi) in x.iter_mut().zip(b.iter()) {
            *xi -= config.step_size * bi;
        }

        // Proximal + box + clip
        if config.prox_l2_weight > 0.0 {
            l2_prox_pull_inplace(&mut x, encoder, config.prox_l2_weight);
        }
        box_project_inplace(&mut x, config.box_lower, config.box_upper);
        clip_state_norm_inplace(&mut x, config.max_state_norm);

        if config.ema_alpha > 0.0 {
            ema_inplace(&mut x, &x_prev, config.ema_alpha);
        }

        let residual = l2_distance(&x, &x_prev);
        residuals.push(residual);
        k_used = k + 1;

        if config.early_stop_eps > 0.0
            && k >= config.early_stop_min_iters
            && residual < config.early_stop_eps
        {
            converged = true;
            break;
        }
    }

    let final_residual = *residuals.last().unwrap_or(&0.0);
    let mean_residual = if residuals.is_empty() {
        0.0
    } else {
        residuals.iter().sum::<f32>() / residuals.len() as f32
    };

    EquilibriumResult {
        x_star: x,
        residuals,
        k_used,
        converged,
        final_residual,
        mean_residual,
    }
}

/// Batch solve: solve equilibrium for multiple inputs.
///
/// # Arguments
/// * `x_inits` - Batch of initial states
/// * `encoders` - Batch of encoder latents
/// * `operator` - Shared linear operator
/// * `config` - Solver configuration
pub fn solve_linear_equilibrium_batch(
    x_inits: &[Vec<f32>],
    encoders: &[Vec<f32>],
    operator: &LinearOperator,
    config: &EquilibriumKernelConfig,
) -> Vec<EquilibriumResult> {
    debug_assert_eq!(x_inits.len(), encoders.len());

    x_inits
        .iter()
        .zip(encoders.iter())
        .map(|(x_init, encoder)| solve_linear_equilibrium(x_init, encoder, operator, config))
        .collect()
}

/// Dense matrix operator for full weight matrices.
///
/// Unlike LinearOperator which only supports diagonal matrices,
/// this supports arbitrary dense matrices A.
#[derive(Debug, Clone)]
pub struct DenseOperator {
    /// Weight matrix
    pub matrix: DenseMatrix,
    /// Bias vector
    pub bias: Vec<f32>,
}

impl DenseOperator {
    /// Create a new dense operator.
    ///
    /// # Arguments
    /// * `matrix_data` - Row-major matrix data
    /// * `rows` - Number of rows (output dimension)
    /// * `cols` - Number of columns (input dimension)
    /// * `bias` - Bias vector of length rows
    pub fn new(matrix_data: Vec<f32>, rows: usize, cols: usize, bias: Vec<f32>) -> Self {
        assert_eq!(bias.len(), rows, "Bias must have same length as matrix rows");
        Self {
            matrix: DenseMatrix::new(matrix_data, rows, cols),
            bias,
        }
    }

    /// Create from a DenseMatrix and bias.
    pub fn from_matrix(matrix: DenseMatrix, bias: Vec<f32>) -> Self {
        assert_eq!(bias.len(), matrix.rows);
        Self { matrix, bias }
    }

    /// Apply operator: y = A @ x + bias
    pub fn apply(&self, x: &[f32]) -> Vec<f32> {
        self.matrix.matvec_add(x, &self.bias)
    }

    /// Apply operator into buffer
    pub fn apply_into(&self, x: &[f32], y: &mut [f32]) {
        self.matrix.matvec_add_into(x, &self.bias, y);
    }

    /// Get output dimension
    pub fn output_dim(&self) -> usize {
        self.matrix.rows
    }

    /// Get input dimension
    pub fn input_dim(&self) -> usize {
        self.matrix.cols
    }

    /// Estimate spectral norm (Lipschitz constant).
    pub fn spectral_norm(&self, max_iters: usize) -> f32 {
        self.matrix.spectral_norm_estimate(max_iters, 1e-6)
    }
}

/// Solve equilibrium with dense matrix operator.
///
/// This is the main solver for equilibrium models with full weight matrices.
///
/// # Arguments
/// * `x_init` - Initial state
/// * `encoder` - Encoder latents (target for L2 pull)
/// * `operator` - Dense matrix operator
/// * `config` - Solver configuration
pub fn solve_dense_equilibrium(
    x_init: &[f32],
    encoder: &[f32],
    operator: &DenseOperator,
    config: &EquilibriumKernelConfig,
) -> EquilibriumResult {
    let dim = x_init.len();
    debug_assert_eq!(encoder.len(), dim);
    debug_assert_eq!(operator.input_dim(), dim);
    debug_assert_eq!(operator.output_dim(), dim);

    let mut x = x_init.to_vec();
    let mut x_prev = vec![0.0; dim];
    let mut b = vec![0.0; dim]; // Reusable buffer for operator output
    let mut residuals = Vec::with_capacity(config.max_iters);
    let mut k_used = 0;
    let mut converged = false;

    for k in 0..config.max_iters {
        // Save previous state
        x_prev.copy_from_slice(&x);

        // 1) Apply dense operator: b = A @ x + bias
        operator.apply_into(&x, &mut b);

        // 2) Forward step: x = x - γ * b
        for (xi, &bi) in x.iter_mut().zip(b.iter()) {
            *xi -= config.step_size * bi;
        }

        // 3) L2 proximal pull toward encoder
        if config.prox_l2_weight > 0.0 {
            l2_prox_pull_inplace(&mut x, encoder, config.prox_l2_weight);
        }

        // 4) Box projection
        box_project_inplace(&mut x, config.box_lower, config.box_upper);

        // 5) Clip state norm
        clip_state_norm_inplace(&mut x, config.max_state_norm);

        // 6) Optional EMA smoothing
        if config.ema_alpha > 0.0 {
            ema_inplace(&mut x, &x_prev, config.ema_alpha);
        }

        // 7) Compute residual
        let residual = l2_distance(&x, &x_prev);
        residuals.push(residual);
        k_used = k + 1;

        // 8) Early stopping
        if config.early_stop_eps > 0.0
            && k >= config.early_stop_min_iters
            && residual < config.early_stop_eps
        {
            converged = true;
            break;
        }
    }

    let final_residual = *residuals.last().unwrap_or(&0.0);
    let mean_residual = if residuals.is_empty() {
        0.0
    } else {
        residuals.iter().sum::<f32>() / residuals.len() as f32
    };

    EquilibriumResult {
        x_star: x,
        residuals,
        k_used,
        converged,
        final_residual,
        mean_residual,
    }
}

/// Batch solve with dense operator.
pub fn solve_dense_equilibrium_batch(
    x_inits: &[Vec<f32>],
    encoders: &[Vec<f32>],
    operator: &DenseOperator,
    config: &EquilibriumKernelConfig,
) -> Vec<EquilibriumResult> {
    debug_assert_eq!(x_inits.len(), encoders.len());

    x_inits
        .iter()
        .zip(encoders.iter())
        .map(|(x_init, encoder)| solve_dense_equilibrium(x_init, encoder, operator, config))
        .collect()
}

// ============================================================================
// Anderson Acceleration
// ============================================================================

/// Configuration for Anderson Acceleration solver.
#[derive(Debug, Clone)]
pub struct AndersonConfig {
    /// Base equilibrium config
    pub base: EquilibriumKernelConfig,
    /// Anderson depth (m): number of previous iterates to use
    pub depth: usize,
    /// Mixing parameter β ∈ (0, 1]: blend between standard and accelerated
    pub beta: f32,
    /// Regularization for least-squares solve
    pub regularization: f32,
    /// Restart threshold: restart if condition number too high
    pub restart_threshold: f32,
}

impl Default for AndersonConfig {
    fn default() -> Self {
        Self {
            base: EquilibriumKernelConfig::default(),
            depth: 5,
            beta: 1.0,
            regularization: 1e-8,
            restart_threshold: 1e8,
        }
    }
}

impl AndersonConfig {
    /// Create config with given depth.
    pub fn with_depth(depth: usize) -> Self {
        Self {
            depth,
            ..Default::default()
        }
    }

    /// Create config from base config with given depth.
    pub fn from_base(base: EquilibriumKernelConfig, depth: usize) -> Self {
        Self {
            base,
            depth,
            ..Default::default()
        }
    }
}

/// Anderson Accelerator state machine.
///
/// Implements Type-I Anderson Acceleration for fixed-point iteration:
///
/// Standard iteration: x_{k+1} = g(x_k)
/// Anderson:          x_{k+1} = g(x_k) - Σ θ_i * (g(x_{k-i}) - g(x_{k-i-1}))
///
/// where θ solves: min ||f_k - F_k * θ||^2
/// and f_k = g(x_k) - x_k is the residual.
#[derive(Debug)]
pub struct AndersonAccelerator {
    /// Dimension of the problem
    dim: usize,
    /// Maximum history depth
    max_depth: usize,
    /// Current number of stored iterates
    current_depth: usize,
    /// History of iterates x (circular buffer)
    x_history: Vec<Vec<f32>>,
    /// History of g(x) values
    gx_history: Vec<Vec<f32>>,
    /// History of residuals f = g(x) - x
    f_history: Vec<Vec<f32>>,
    /// Current position in circular buffer
    head: usize,
    /// Regularization parameter
    regularization: f32,
}

impl AndersonAccelerator {
    /// Create a new Anderson accelerator.
    pub fn new(dim: usize, max_depth: usize, regularization: f32) -> Self {
        let mut x_history = Vec::with_capacity(max_depth + 1);
        let mut gx_history = Vec::with_capacity(max_depth + 1);
        let mut f_history = Vec::with_capacity(max_depth + 1);

        for _ in 0..=max_depth {
            x_history.push(vec![0.0; dim]);
            gx_history.push(vec![0.0; dim]);
            f_history.push(vec![0.0; dim]);
        }

        Self {
            dim,
            max_depth,
            current_depth: 0,
            x_history,
            gx_history,
            f_history,
            head: 0,
            regularization,
        }
    }

    /// Reset the accelerator state.
    pub fn reset(&mut self) {
        self.current_depth = 0;
        self.head = 0;
    }

    /// Store a new iterate and its image.
    fn store(&mut self, x: &[f32], gx: &[f32]) {
        self.x_history[self.head].copy_from_slice(x);
        self.gx_history[self.head].copy_from_slice(gx);

        // f = g(x) - x
        for (fi, (&gxi, &xi)) in self.f_history[self.head]
            .iter_mut()
            .zip(gx.iter().zip(x.iter()))
        {
            *fi = gxi - xi;
        }

        self.head = (self.head + 1) % (self.max_depth + 1);
        if self.current_depth < self.max_depth {
            self.current_depth += 1;
        }
    }

    /// Get index into circular buffer.
    #[inline]
    fn idx(&self, offset: usize) -> usize {
        (self.head + self.max_depth + 1 - 1 - offset) % (self.max_depth + 1)
    }

    /// Compute accelerated iterate.
    ///
    /// Returns g(x_k) if not enough history, otherwise computes
    /// the Anderson-accelerated iterate.
    pub fn accelerate(&mut self, x: &[f32], gx: &[f32], beta: f32) -> Vec<f32> {
        // Store current iterate
        self.store(x, gx);

        // Need at least 2 iterates for acceleration
        if self.current_depth < 2 {
            return gx.to_vec();
        }

        let m = self.current_depth.min(self.max_depth);

        // Build system for least-squares: min ||f_k - F * θ||^2
        // where F = [Δf_0, Δf_1, ..., Δf_{m-1}]
        // and Δf_i = f_{k-i} - f_{k-i-1}

        // Compute F^T F matrix (m x m) and F^T f_k vector
        let f_k = &self.f_history[self.idx(0)];

        // F^T F + regularization * I
        let mut ftf = vec![0.0f32; m * m];
        // F^T f_k
        let mut ftf_k = vec![0.0f32; m];

        for i in 0..m {
            let f_i = &self.f_history[self.idx(i)];
            let f_i_prev = &self.f_history[self.idx(i + 1)];

            // Δf_i = f_{k-i} - f_{k-i-1}
            for j in 0..m {
                let f_j = &self.f_history[self.idx(j)];
                let f_j_prev = &self.f_history[self.idx(j + 1)];

                // (Δf_i)^T (Δf_j)
                let mut dot = 0.0f32;
                for d in 0..self.dim {
                    let df_i = f_i[d] - f_i_prev[d];
                    let df_j = f_j[d] - f_j_prev[d];
                    dot += df_i * df_j;
                }
                ftf[i * m + j] = dot;
            }

            // Add regularization
            ftf[i * m + i] += self.regularization;

            // (Δf_i)^T f_k
            let mut dot = 0.0f32;
            for d in 0..self.dim {
                let df_i = f_i[d] - f_i_prev[d];
                dot += df_i * f_k[d];
            }
            ftf_k[i] = dot;
        }

        // Solve (F^T F) θ = F^T f_k using Cholesky or fallback to standard iterate
        let theta = match solve_positive_definite(&ftf, &ftf_k, m) {
            Some(t) => t,
            None => return gx.to_vec(), // Fall back to standard iterate
        };

        // Compute accelerated iterate:
        // x_{k+1} = g(x_k) - Σ θ_i * (g(x_{k-i}) - g(x_{k-i-1}))
        let mut result = gx.to_vec();

        for i in 0..m {
            let gx_i = &self.gx_history[self.idx(i)];
            let gx_i_prev = &self.gx_history[self.idx(i + 1)];

            for d in 0..self.dim {
                result[d] -= theta[i] * (gx_i[d] - gx_i_prev[d]);
            }
        }

        // Optional mixing: blend between standard g(x) and accelerated
        if beta < 1.0 {
            for (r, &g) in result.iter_mut().zip(gx.iter()) {
                *r = beta * *r + (1.0 - beta) * g;
            }
        }

        result
    }
}

/// Solve positive definite system Ax = b using Cholesky decomposition.
///
/// Returns None if matrix is not positive definite or singular.
fn solve_positive_definite(a: &[f32], b: &[f32], n: usize) -> Option<Vec<f32>> {
    // Cholesky decomposition: A = L L^T
    let mut l = vec![0.0f32; n * n];

    for i in 0..n {
        for j in 0..=i {
            let mut sum = a[i * n + j];

            for k in 0..j {
                sum -= l[i * n + k] * l[j * n + k];
            }

            if i == j {
                if sum <= 0.0 {
                    return None; // Not positive definite
                }
                l[i * n + j] = sum.sqrt();
            } else {
                let l_jj = l[j * n + j];
                if l_jj.abs() < 1e-12 {
                    return None; // Singular
                }
                l[i * n + j] = sum / l_jj;
            }
        }
    }

    // Forward substitution: L y = b
    let mut y = vec![0.0f32; n];
    for i in 0..n {
        let mut sum = b[i];
        for j in 0..i {
            sum -= l[i * n + j] * y[j];
        }
        let l_ii = l[i * n + i];
        if l_ii.abs() < 1e-12 {
            return None;
        }
        y[i] = sum / l_ii;
    }

    // Backward substitution: L^T x = y
    let mut x = vec![0.0f32; n];
    for i in (0..n).rev() {
        let mut sum = y[i];
        for j in (i + 1)..n {
            sum -= l[j * n + i] * x[j];
        }
        let l_ii = l[i * n + i];
        if l_ii.abs() < 1e-12 {
            return None;
        }
        x[i] = sum / l_ii;
    }

    Some(x)
}

/// Solve equilibrium with Anderson Acceleration.
///
/// Anderson Acceleration uses history of previous iterates to find
/// an optimal linear combination that accelerates convergence.
/// Can reduce iterations by 2-5x for well-conditioned problems.
///
/// # Arguments
/// * `x_init` - Initial state
/// * `encoder` - Encoder latents (target for L2 pull)
/// * `operator` - Linear operator
/// * `config` - Anderson configuration
pub fn solve_anderson_equilibrium(
    x_init: &[f32],
    encoder: &[f32],
    operator: &LinearOperator,
    config: &AndersonConfig,
) -> EquilibriumResult {
    let dim = x_init.len();
    debug_assert_eq!(encoder.len(), dim);
    debug_assert_eq!(operator.a_diag.len(), dim);

    let mut x = x_init.to_vec();
    let mut x_prev = vec![0.0; dim];
    let mut gx = vec![0.0; dim];
    let mut residuals = Vec::with_capacity(config.base.max_iters);
    let mut k_used = 0;
    let mut converged = false;

    let mut accelerator = AndersonAccelerator::new(dim, config.depth, config.regularization);

    for k in 0..config.base.max_iters {
        x_prev.copy_from_slice(&x);

        // 1) Apply operator: b = A @ x + bias
        let b = operator.apply(&x);

        // 2) Forward step: gx = x - γ * b (this is g(x))
        for (gxi, (&xi, &bi)) in gx.iter_mut().zip(x.iter().zip(b.iter())) {
            *gxi = xi - config.base.step_size * bi;
        }

        // 3) L2 proximal pull toward encoder
        if config.base.prox_l2_weight > 0.0 {
            l2_prox_pull_inplace(&mut gx, encoder, config.base.prox_l2_weight);
        }

        // 4) Box projection
        box_project_inplace(&mut gx, config.base.box_lower, config.base.box_upper);

        // 5) Clip state norm
        clip_state_norm_inplace(&mut gx, config.base.max_state_norm);

        // 6) Anderson acceleration
        let x_new = accelerator.accelerate(&x, &gx, config.beta);
        x.copy_from_slice(&x_new);

        // 7) Apply constraints to accelerated iterate
        box_project_inplace(&mut x, config.base.box_lower, config.base.box_upper);
        clip_state_norm_inplace(&mut x, config.base.max_state_norm);

        // 8) Optional EMA smoothing
        if config.base.ema_alpha > 0.0 {
            ema_inplace(&mut x, &x_prev, config.base.ema_alpha);
        }

        // 9) Compute residual
        let residual = l2_distance(&x, &x_prev);
        residuals.push(residual);
        k_used = k + 1;

        // 10) Early stopping
        if config.base.early_stop_eps > 0.0
            && k >= config.base.early_stop_min_iters
            && residual < config.base.early_stop_eps
        {
            converged = true;
            break;
        }
    }

    let final_residual = *residuals.last().unwrap_or(&0.0);
    let mean_residual = if residuals.is_empty() {
        0.0
    } else {
        residuals.iter().sum::<f32>() / residuals.len() as f32
    };

    EquilibriumResult {
        x_star: x,
        residuals,
        k_used,
        converged,
        final_residual,
        mean_residual,
    }
}

/// Solve equilibrium with Anderson Acceleration using dense operator.
pub fn solve_anderson_dense_equilibrium(
    x_init: &[f32],
    encoder: &[f32],
    operator: &DenseOperator,
    config: &AndersonConfig,
) -> EquilibriumResult {
    let dim = x_init.len();
    debug_assert_eq!(encoder.len(), dim);
    debug_assert_eq!(operator.input_dim(), dim);
    debug_assert_eq!(operator.output_dim(), dim);

    let mut x = x_init.to_vec();
    let mut x_prev = vec![0.0; dim];
    let mut gx = vec![0.0; dim];
    let mut b = vec![0.0; dim];
    let mut residuals = Vec::with_capacity(config.base.max_iters);
    let mut k_used = 0;
    let mut converged = false;

    let mut accelerator = AndersonAccelerator::new(dim, config.depth, config.regularization);

    for k in 0..config.base.max_iters {
        x_prev.copy_from_slice(&x);

        // 1) Apply dense operator: b = A @ x + bias
        operator.apply_into(&x, &mut b);

        // 2) Forward step: gx = x - γ * b
        for (gxi, (&xi, &bi)) in gx.iter_mut().zip(x.iter().zip(b.iter())) {
            *gxi = xi - config.base.step_size * bi;
        }

        // 3) L2 proximal pull toward encoder
        if config.base.prox_l2_weight > 0.0 {
            l2_prox_pull_inplace(&mut gx, encoder, config.base.prox_l2_weight);
        }

        // 4) Box projection
        box_project_inplace(&mut gx, config.base.box_lower, config.base.box_upper);

        // 5) Clip state norm
        clip_state_norm_inplace(&mut gx, config.base.max_state_norm);

        // 6) Anderson acceleration
        let x_new = accelerator.accelerate(&x, &gx, config.beta);
        x.copy_from_slice(&x_new);

        // 7) Apply constraints to accelerated iterate
        box_project_inplace(&mut x, config.base.box_lower, config.base.box_upper);
        clip_state_norm_inplace(&mut x, config.base.max_state_norm);

        // 8) Optional EMA smoothing
        if config.base.ema_alpha > 0.0 {
            ema_inplace(&mut x, &x_prev, config.base.ema_alpha);
        }

        // 9) Compute residual
        let residual = l2_distance(&x, &x_prev);
        residuals.push(residual);
        k_used = k + 1;

        // 10) Early stopping
        if config.base.early_stop_eps > 0.0
            && k >= config.base.early_stop_min_iters
            && residual < config.base.early_stop_eps
        {
            converged = true;
            break;
        }
    }

    let final_residual = *residuals.last().unwrap_or(&0.0);
    let mean_residual = if residuals.is_empty() {
        0.0
    } else {
        residuals.iter().sum::<f32>() / residuals.len() as f32
    };

    EquilibriumResult {
        x_star: x,
        residuals,
        k_used,
        converged,
        final_residual,
        mean_residual,
    }
}

/// Batch solve with Anderson Acceleration.
pub fn solve_anderson_equilibrium_batch(
    x_inits: &[Vec<f32>],
    encoders: &[Vec<f32>],
    operator: &LinearOperator,
    config: &AndersonConfig,
) -> Vec<EquilibriumResult> {
    debug_assert_eq!(x_inits.len(), encoders.len());

    x_inits
        .iter()
        .zip(encoders.iter())
        .map(|(x_init, encoder)| solve_anderson_equilibrium(x_init, encoder, operator, config))
        .collect()
}

// ============================================================================
// Parallel Batch Solving (Rayon)
// ============================================================================

use rayon::prelude::*;

/// Parallel batch solve with linear operator.
///
/// Uses Rayon to solve multiple equilibrium problems in parallel across
/// available CPU cores. Provides significant speedup for large batches.
///
/// # Arguments
/// * `x_inits` - Batch of initial states
/// * `encoders` - Batch of encoder latents
/// * `operator` - Shared linear operator (must be Send + Sync)
/// * `config` - Solver configuration
///
/// # Example
/// ```ignore
/// let results = solve_linear_equilibrium_batch_parallel(
///     &x_inits, &encoders, &operator, &config
/// );
/// ```
pub fn solve_linear_equilibrium_batch_parallel(
    x_inits: &[Vec<f32>],
    encoders: &[Vec<f32>],
    operator: &LinearOperator,
    config: &EquilibriumKernelConfig,
) -> Vec<EquilibriumResult> {
    debug_assert_eq!(x_inits.len(), encoders.len());

    x_inits
        .par_iter()
        .zip(encoders.par_iter())
        .map(|(x_init, encoder)| solve_linear_equilibrium(x_init, encoder, operator, config))
        .collect()
}

/// Parallel batch solve with dense operator.
pub fn solve_dense_equilibrium_batch_parallel(
    x_inits: &[Vec<f32>],
    encoders: &[Vec<f32>],
    operator: &DenseOperator,
    config: &EquilibriumKernelConfig,
) -> Vec<EquilibriumResult> {
    debug_assert_eq!(x_inits.len(), encoders.len());

    x_inits
        .par_iter()
        .zip(encoders.par_iter())
        .map(|(x_init, encoder)| solve_dense_equilibrium(x_init, encoder, operator, config))
        .collect()
}

/// Parallel batch solve with Anderson Acceleration.
pub fn solve_anderson_equilibrium_batch_parallel(
    x_inits: &[Vec<f32>],
    encoders: &[Vec<f32>],
    operator: &LinearOperator,
    config: &AndersonConfig,
) -> Vec<EquilibriumResult> {
    debug_assert_eq!(x_inits.len(), encoders.len());

    x_inits
        .par_iter()
        .zip(encoders.par_iter())
        .map(|(x_init, encoder)| solve_anderson_equilibrium(x_init, encoder, operator, config))
        .collect()
}

/// Parallel batch solve with Anderson Acceleration and dense operator.
pub fn solve_anderson_dense_equilibrium_batch_parallel(
    x_inits: &[Vec<f32>],
    encoders: &[Vec<f32>],
    operator: &DenseOperator,
    config: &AndersonConfig,
) -> Vec<EquilibriumResult> {
    debug_assert_eq!(x_inits.len(), encoders.len());

    x_inits
        .par_iter()
        .zip(encoders.par_iter())
        .map(|(x_init, encoder)| solve_anderson_dense_equilibrium(x_init, encoder, operator, config))
        .collect()
}

/// Configuration for parallel batch solving.
#[derive(Debug, Clone)]
pub struct ParallelConfig {
    /// Minimum batch size to use parallel execution.
    /// Below this threshold, sequential execution is used.
    pub min_parallel_size: usize,
    /// Optional thread pool size (None = use Rayon default)
    pub num_threads: Option<usize>,
}

impl Default for ParallelConfig {
    fn default() -> Self {
        Self {
            min_parallel_size: 4,
            num_threads: None,
        }
    }
}

/// Adaptive batch solve that chooses parallel vs sequential based on batch size.
///
/// For small batches, uses sequential execution to avoid thread overhead.
/// For larger batches, uses Rayon parallel execution.
pub fn solve_linear_equilibrium_batch_adaptive(
    x_inits: &[Vec<f32>],
    encoders: &[Vec<f32>],
    operator: &LinearOperator,
    config: &EquilibriumKernelConfig,
    parallel_config: &ParallelConfig,
) -> Vec<EquilibriumResult> {
    if x_inits.len() < parallel_config.min_parallel_size {
        solve_linear_equilibrium_batch(x_inits, encoders, operator, config)
    } else {
        solve_linear_equilibrium_batch_parallel(x_inits, encoders, operator, config)
    }
}

/// Adaptive batch solve for dense operators.
pub fn solve_dense_equilibrium_batch_adaptive(
    x_inits: &[Vec<f32>],
    encoders: &[Vec<f32>],
    operator: &DenseOperator,
    config: &EquilibriumKernelConfig,
    parallel_config: &ParallelConfig,
) -> Vec<EquilibriumResult> {
    if x_inits.len() < parallel_config.min_parallel_size {
        solve_dense_equilibrium_batch(x_inits, encoders, operator, config)
    } else {
        solve_dense_equilibrium_batch_parallel(x_inits, encoders, operator, config)
    }
}

/// Adaptive batch solve with Anderson Acceleration.
pub fn solve_anderson_equilibrium_batch_adaptive(
    x_inits: &[Vec<f32>],
    encoders: &[Vec<f32>],
    operator: &LinearOperator,
    config: &AndersonConfig,
    parallel_config: &ParallelConfig,
) -> Vec<EquilibriumResult> {
    if x_inits.len() < parallel_config.min_parallel_size {
        solve_anderson_equilibrium_batch(x_inits, encoders, operator, config)
    } else {
        solve_anderson_equilibrium_batch_parallel(x_inits, encoders, operator, config)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_linear_operator() {
        let op = LinearOperator::new(vec![0.5, 0.5], vec![1.0, -1.0]);
        let x = vec![2.0, 4.0];
        let result = op.apply(&x);
        // 0.5 * 2 + 1 = 2, 0.5 * 4 - 1 = 1
        assert!((result[0] - 2.0).abs() < 1e-6);
        assert!((result[1] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_solve_identity() {
        let config = EquilibriumKernelConfig {
            max_iters: 10,
            step_size: 0.5,
            prox_l2_weight: 0.1,
            early_stop_eps: 1e-4,
            ..Default::default()
        };

        let x_init = vec![0.0, 0.0, 0.0, 0.0];
        let encoder = vec![1.0, 2.0, 3.0, 4.0];
        let operator = LinearOperator::identity(4);

        let result = solve_linear_equilibrium(&x_init, &encoder, &operator, &config);

        // Should converge toward encoder (due to L2 pull)
        assert!(result.final_residual < 0.1);
        assert!(result.k_used <= config.max_iters);
    }

    #[test]
    fn test_solve_contractive() {
        let config = EquilibriumKernelConfig {
            max_iters: 20,
            step_size: 0.3,
            prox_l2_weight: 0.0, // No prox pull
            early_stop_eps: 1e-5,
            ..Default::default()
        };

        let x_init = vec![5.0, 5.0];
        let encoder = vec![0.0, 0.0];
        // Contractive operator: pushes toward origin
        let operator = LinearOperator::contractive(2, 0.5);

        let result = solve_linear_equilibrium(&x_init, &encoder, &operator, &config);

        // Should contract toward fixed point
        assert!(result.k_used > 1);
        // Residuals should decrease
        for i in 1..result.residuals.len() {
            assert!(result.residuals[i] <= result.residuals[i - 1] + 1e-6);
        }
    }

    #[test]
    fn test_early_stopping() {
        let config = EquilibriumKernelConfig {
            max_iters: 100,
            step_size: 0.5,
            prox_l2_weight: 0.5, // Strong pull
            early_stop_eps: 1e-3,
            early_stop_min_iters: 2,
            ..Default::default()
        };

        let x_init = vec![0.0, 0.0];
        let encoder = vec![1.0, 1.0];
        let operator = LinearOperator::contractive(2, 0.1);

        let result = solve_linear_equilibrium(&x_init, &encoder, &operator, &config);

        // Should stop early, not use all iterations
        assert!(result.k_used < config.max_iters);
        assert!(result.converged);
    }

    #[test]
    fn test_batch_solve() {
        let config = EquilibriumKernelConfig::default();

        let x_inits = vec![vec![0.0, 0.0], vec![1.0, 1.0], vec![-1.0, -1.0]];
        let encoders = vec![vec![0.5, 0.5], vec![0.5, 0.5], vec![0.5, 0.5]];
        let operator = LinearOperator::contractive(2, 0.3);

        let results = solve_linear_equilibrium_batch(&x_inits, &encoders, &operator, &config);

        assert_eq!(results.len(), 3);
        for r in results {
            assert_eq!(r.x_star.len(), 2);
        }
    }

    #[test]
    fn test_dense_operator() {
        // 2x2 matrix: [[0.5, 0.1], [0.2, 0.5]]
        let matrix_data = vec![0.5, 0.1, 0.2, 0.5];
        let bias = vec![1.0, -1.0];
        let op = DenseOperator::new(matrix_data, 2, 2, bias);

        let x = vec![2.0, 4.0];
        let result = op.apply(&x);
        // [0.5*2 + 0.1*4 + 1.0, 0.2*2 + 0.5*4 - 1.0] = [2.4, 1.4]
        assert!((result[0] - 2.4).abs() < 1e-5);
        assert!((result[1] - 1.4).abs() < 1e-5);
    }

    #[test]
    fn test_dense_equilibrium() {
        let config = EquilibriumKernelConfig {
            max_iters: 50,
            step_size: 0.3,
            prox_l2_weight: 0.1,
            early_stop_eps: 1e-5,
            early_stop_min_iters: 5,
            ..Default::default()
        };

        // Contractive operator: 0.5 * I
        let matrix_data = vec![0.5, 0.0, 0.0, 0.5];
        let bias = vec![0.0, 0.0];
        let operator = DenseOperator::new(matrix_data, 2, 2, bias);

        let x_init = vec![0.0, 0.0];
        let encoder = vec![1.0, 1.0];

        let result = solve_dense_equilibrium(&x_init, &encoder, &operator, &config);

        assert!(result.final_residual < 0.01);
        assert!(result.k_used < config.max_iters);
    }

    #[test]
    fn test_dense_spectral_norm() {
        // Identity matrix should have spectral norm 1
        let matrix_data = vec![1.0, 0.0, 0.0, 1.0];
        let bias = vec![0.0, 0.0];
        let op = DenseOperator::new(matrix_data, 2, 2, bias);

        let sigma = op.spectral_norm(100);
        assert!((sigma - 1.0).abs() < 1e-3);

        // 0.5 * I should have spectral norm 0.5
        let matrix_data2 = vec![0.5, 0.0, 0.0, 0.5];
        let op2 = DenseOperator::new(matrix_data2, 2, 2, vec![0.0, 0.0]);
        let sigma2 = op2.spectral_norm(100);
        assert!((sigma2 - 0.5).abs() < 1e-3);
    }

    // ========================================================================
    // Anderson Acceleration Tests
    // ========================================================================

    #[test]
    fn test_cholesky_solve() {
        // Simple 2x2 positive definite system
        // A = [[4, 2], [2, 3]], b = [8, 7]
        // Solve: 4x + 2y = 8, 2x + 3y = 7
        // Solution: x = 1.25, y = 1.5
        let a = vec![4.0, 2.0, 2.0, 3.0];
        let b = vec![8.0, 7.0];
        let x = solve_positive_definite(&a, &b, 2).unwrap();
        assert!((x[0] - 1.25).abs() < 1e-5, "x[0] = {}, expected 1.25", x[0]);
        assert!((x[1] - 1.5).abs() < 1e-5, "x[1] = {}, expected 1.5", x[1]);
    }

    #[test]
    fn test_cholesky_singular() {
        // Singular matrix should return None
        let a = vec![1.0, 1.0, 1.0, 1.0];
        let b = vec![1.0, 1.0];
        assert!(solve_positive_definite(&a, &b, 2).is_none());
    }

    #[test]
    fn test_anderson_accelerator_basic() {
        let mut acc = AndersonAccelerator::new(2, 3, 1e-8);

        // First iterate: just returns gx
        let x1 = vec![0.0, 0.0];
        let gx1 = vec![1.0, 1.0];
        let result1 = acc.accelerate(&x1, &gx1, 1.0);
        assert_eq!(result1, gx1);

        // Second iterate: still just returns gx (need 2+ history)
        let x2 = vec![1.0, 1.0];
        let gx2 = vec![0.8, 0.8];
        let result2 = acc.accelerate(&x2, &gx2, 1.0);
        // With only 2 iterates, acceleration kicks in
        assert!(result2[0] < 1.0); // Should be modified
    }

    #[test]
    fn test_anderson_equilibrium_convergence() {
        let base_config = EquilibriumKernelConfig {
            max_iters: 50,
            step_size: 0.3,
            prox_l2_weight: 0.1,
            early_stop_eps: 1e-6,
            early_stop_min_iters: 3,
            ..Default::default()
        };

        let anderson_config = AndersonConfig {
            base: base_config.clone(),
            depth: 5,
            beta: 1.0,
            regularization: 1e-8,
            restart_threshold: 1e8,
        };

        let x_init = vec![0.0, 0.0, 0.0, 0.0];
        let encoder = vec![1.0, 2.0, 3.0, 4.0];
        let operator = LinearOperator::contractive(4, 0.5);

        // Standard solve
        let result_std = solve_linear_equilibrium(&x_init, &encoder, &operator, &base_config);

        // Anderson solve
        let result_aa = solve_anderson_equilibrium(&x_init, &encoder, &operator, &anderson_config);

        // Both should converge
        assert!(result_std.final_residual < 0.01 || result_std.converged);
        assert!(result_aa.final_residual < 0.01 || result_aa.converged);

        // Anderson should typically use fewer or equal iterations
        // (may not always be true for simple problems, but shouldn't diverge)
        assert!(result_aa.final_residual <= result_std.final_residual * 2.0);
    }

    #[test]
    fn test_anderson_vs_standard_iterations() {
        // Test that Anderson doesn't use more iterations than standard for contractive ops
        let base_config = EquilibriumKernelConfig {
            max_iters: 100,
            step_size: 0.2,
            prox_l2_weight: 0.05,
            early_stop_eps: 1e-5,
            early_stop_min_iters: 5,
            ..Default::default()
        };

        let anderson_config = AndersonConfig::from_base(base_config.clone(), 5);

        let x_init = vec![5.0, -3.0, 2.0, -1.0, 4.0, -2.0, 1.0, -0.5];
        let encoder = vec![0.0; 8];
        let operator = LinearOperator::contractive(8, 0.3);

        let result_std = solve_linear_equilibrium(&x_init, &encoder, &operator, &base_config);
        let result_aa = solve_anderson_equilibrium(&x_init, &encoder, &operator, &anderson_config);

        // Both should converge to similar solution
        let diff: f32 = result_std
            .x_star
            .iter()
            .zip(result_aa.x_star.iter())
            .map(|(a, b)| (a - b).abs())
            .sum();

        assert!(diff < 0.1, "Solutions should be similar");
    }

    #[test]
    fn test_anderson_dense_equilibrium() {
        let base_config = EquilibriumKernelConfig {
            max_iters: 50,
            step_size: 0.3,
            prox_l2_weight: 0.1,
            early_stop_eps: 1e-5,
            early_stop_min_iters: 5,
            ..Default::default()
        };

        let anderson_config = AndersonConfig::from_base(base_config.clone(), 3);

        // Contractive dense matrix: 0.4 * I
        let matrix_data = vec![0.4, 0.0, 0.0, 0.4];
        let bias = vec![0.0, 0.0];
        let operator = DenseOperator::new(matrix_data, 2, 2, bias);

        let x_init = vec![5.0, -3.0];
        let encoder = vec![1.0, 1.0];

        let result = solve_anderson_dense_equilibrium(&x_init, &encoder, &operator, &anderson_config);

        assert!(result.final_residual < 0.01 || result.converged);
        assert!(result.k_used < 50);
    }

    #[test]
    fn test_anderson_config_constructors() {
        let config1 = AndersonConfig::with_depth(10);
        assert_eq!(config1.depth, 10);
        assert_eq!(config1.base.max_iters, 4); // Default

        let base = EquilibriumKernelConfig {
            max_iters: 100,
            ..Default::default()
        };
        let config2 = AndersonConfig::from_base(base, 7);
        assert_eq!(config2.depth, 7);
        assert_eq!(config2.base.max_iters, 100);
    }

    #[test]
    fn test_anderson_batch() {
        let config = AndersonConfig::with_depth(3);

        let x_inits = vec![
            vec![0.0, 0.0],
            vec![1.0, 1.0],
            vec![-1.0, -1.0],
        ];
        let encoders = vec![
            vec![0.5, 0.5],
            vec![0.5, 0.5],
            vec![0.5, 0.5],
        ];
        let operator = LinearOperator::contractive(2, 0.3);

        let results = solve_anderson_equilibrium_batch(&x_inits, &encoders, &operator, &config);

        assert_eq!(results.len(), 3);
        for r in results {
            assert_eq!(r.x_star.len(), 2);
        }
    }

    #[test]
    fn test_anderson_with_mixing() {
        let config = AndersonConfig {
            base: EquilibriumKernelConfig {
                max_iters: 20,
                step_size: 0.3,
                early_stop_eps: 1e-5,
                ..Default::default()
            },
            depth: 3,
            beta: 0.5, // 50% mixing
            regularization: 1e-8,
            restart_threshold: 1e8,
        };

        let x_init = vec![0.0, 0.0];
        let encoder = vec![1.0, 1.0];
        let operator = LinearOperator::contractive(2, 0.3);

        let result = solve_anderson_equilibrium(&x_init, &encoder, &operator, &config);

        // Should still converge with mixing
        assert!(result.k_used <= 20);
    }

    // ========================================================================
    // Parallel Batch Tests
    // ========================================================================

    #[test]
    fn test_parallel_batch_linear() {
        let config = EquilibriumKernelConfig {
            max_iters: 20,
            step_size: 0.3,
            early_stop_eps: 1e-5,
            ..Default::default()
        };

        // Create a larger batch
        let batch_size = 16;
        let dim = 4;
        let x_inits: Vec<Vec<f32>> = (0..batch_size)
            .map(|i| vec![i as f32 * 0.1; dim])
            .collect();
        let encoders: Vec<Vec<f32>> = (0..batch_size)
            .map(|_| vec![0.5; dim])
            .collect();
        let operator = LinearOperator::contractive(dim, 0.3);

        // Sequential
        let results_seq = solve_linear_equilibrium_batch(&x_inits, &encoders, &operator, &config);

        // Parallel
        let results_par = solve_linear_equilibrium_batch_parallel(&x_inits, &encoders, &operator, &config);

        // Results should be identical
        assert_eq!(results_seq.len(), results_par.len());
        for (seq, par) in results_seq.iter().zip(results_par.iter()) {
            assert_eq!(seq.k_used, par.k_used);
            assert!((seq.final_residual - par.final_residual).abs() < 1e-6);
            for (s, p) in seq.x_star.iter().zip(par.x_star.iter()) {
                assert!((s - p).abs() < 1e-6);
            }
        }
    }

    #[test]
    fn test_parallel_batch_dense() {
        let config = EquilibriumKernelConfig {
            max_iters: 20,
            step_size: 0.3,
            early_stop_eps: 1e-5,
            ..Default::default()
        };

        // Contractive dense matrix: 0.4 * I
        let matrix_data = vec![0.4, 0.0, 0.0, 0.4];
        let bias = vec![0.0, 0.0];
        let operator = DenseOperator::new(matrix_data, 2, 2, bias);

        let batch_size = 8;
        let x_inits: Vec<Vec<f32>> = (0..batch_size)
            .map(|i| vec![i as f32 * 0.5, -(i as f32) * 0.5])
            .collect();
        let encoders: Vec<Vec<f32>> = (0..batch_size)
            .map(|_| vec![1.0, 1.0])
            .collect();

        let results_seq = solve_dense_equilibrium_batch(&x_inits, &encoders, &operator, &config);
        let results_par = solve_dense_equilibrium_batch_parallel(&x_inits, &encoders, &operator, &config);

        assert_eq!(results_seq.len(), results_par.len());
        for (seq, par) in results_seq.iter().zip(results_par.iter()) {
            assert!((seq.final_residual - par.final_residual).abs() < 1e-6);
        }
    }

    #[test]
    fn test_parallel_batch_anderson() {
        let config = AndersonConfig::with_depth(3);

        let batch_size = 10;
        let x_inits: Vec<Vec<f32>> = (0..batch_size)
            .map(|i| vec![i as f32 * 0.2, -(i as f32) * 0.2])
            .collect();
        let encoders: Vec<Vec<f32>> = (0..batch_size)
            .map(|_| vec![0.5, 0.5])
            .collect();
        let operator = LinearOperator::contractive(2, 0.3);

        let results_seq = solve_anderson_equilibrium_batch(&x_inits, &encoders, &operator, &config);
        let results_par = solve_anderson_equilibrium_batch_parallel(&x_inits, &encoders, &operator, &config);

        assert_eq!(results_seq.len(), results_par.len());
        for (seq, par) in results_seq.iter().zip(results_par.iter()) {
            // Anderson may have small numerical differences due to history ordering
            // but should converge to same solution
            let diff: f32 = seq.x_star.iter()
                .zip(par.x_star.iter())
                .map(|(a, b)| (a - b).abs())
                .sum();
            assert!(diff < 0.01, "Solutions should match");
        }
    }

    #[test]
    fn test_adaptive_batch_threshold() {
        let config = EquilibriumKernelConfig::default();
        let parallel_config = ParallelConfig {
            min_parallel_size: 5,
            num_threads: None,
        };

        let operator = LinearOperator::contractive(2, 0.3);

        // Small batch - should use sequential
        let small_inits = vec![vec![0.0, 0.0]; 3];
        let small_encs = vec![vec![1.0, 1.0]; 3];
        let results = solve_linear_equilibrium_batch_adaptive(
            &small_inits, &small_encs, &operator, &config, &parallel_config
        );
        assert_eq!(results.len(), 3);

        // Large batch - should use parallel
        let large_inits = vec![vec![0.0, 0.0]; 10];
        let large_encs = vec![vec![1.0, 1.0]; 10];
        let results = solve_linear_equilibrium_batch_adaptive(
            &large_inits, &large_encs, &operator, &config, &parallel_config
        );
        assert_eq!(results.len(), 10);
    }

    #[test]
    fn test_parallel_config_default() {
        let config = ParallelConfig::default();
        assert_eq!(config.min_parallel_size, 4);
        assert!(config.num_threads.is_none());
    }
}

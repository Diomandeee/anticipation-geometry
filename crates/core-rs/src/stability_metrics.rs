//! Stability metrics for equilibrium solvers.
//!
//! High-performance implementations of convergence and stability analysis.

use crate::proximal_ops::{l2_distance, l2_norm};

/// Compute contraction ratio from consecutive residuals.
///
/// The contraction ratio r is defined as:
///     r = ||res[k+1]|| / ||res[k]||
///
/// A value r < 1 indicates contraction (convergence).
///
/// # Arguments
/// * `residuals` - Residual history (1D: per-iteration norms)
/// * `window_size` - Number of iterations to average over
///
/// # Returns
/// Mean contraction ratio over iterations
pub fn contraction_ratio(residuals: &[f32], window_size: usize) -> f32 {
    if residuals.len() < 2 {
        return 1.0;
    }

    // Compute ratios
    let mut ratios: Vec<f32> = Vec::with_capacity(residuals.len() - 1);
    for i in 0..residuals.len() - 1 {
        let ratio = residuals[i + 1] / (residuals[i] + 1e-8);
        ratios.push(ratio);
    }

    if ratios.is_empty() {
        return 1.0;
    }

    // Average over window
    if window_size > 1 && ratios.len() >= window_size {
        let mut windowed_means: Vec<f32> = Vec::new();
        for i in 0..=(ratios.len() - window_size) {
            let window_sum: f32 = ratios[i..i + window_size].iter().sum();
            windowed_means.push(window_sum / window_size as f32);
        }
        windowed_means.iter().sum::<f32>() / windowed_means.len() as f32
    } else {
        ratios.iter().sum::<f32>() / ratios.len() as f32
    }
}

/// Compute contraction ratio for batch residuals.
///
/// # Arguments
/// * `residuals` - 2D residuals [K iterations, B batch]
/// * `window_size` - Number of iterations to average over
pub fn contraction_ratio_batch(residuals: &[Vec<f32>], window_size: usize) -> f32 {
    if residuals.len() < 2 {
        return 1.0;
    }

    // Compute mean residual per iteration
    let res_norms: Vec<f32> = residuals
        .iter()
        .map(|batch| {
            if batch.is_empty() {
                0.0
            } else {
                batch.iter().sum::<f32>() / batch.len() as f32
            }
        })
        .collect();

    contraction_ratio(&res_norms, window_size)
}

/// Compute stability headroom.
///
/// Headroom = 1 - max(s · σ_max(J_B))
///
/// where s is the metric scale and σ_max is the operator's Lipschitz constant.
/// Positive headroom indicates guaranteed contraction.
///
/// # Arguments
/// * `operator_lipschitz` - Lipschitz constant of operator (should be ≤ 1)
/// * `metric_scale_max` - Maximum metric scale (None if no adaptive metric)
pub fn headroom(operator_lipschitz: f32, metric_scale_max: Option<f32>) -> f32 {
    match metric_scale_max {
        None => 1.0 - operator_lipschitz,
        Some(max_scale) => 1.0 - (max_scale * operator_lipschitz),
    }
}

/// Health check results for solver stability.
#[derive(Debug, Clone, PartialEq)]
pub struct StabilityHealthCheck {
    /// Contraction ratio < threshold
    pub is_contracting: bool,
    /// Final residual < threshold
    pub is_converged: bool,
    /// No NaN or Inf values
    pub is_stable: bool,
    /// Residuals decrease monotonically
    pub is_monotonic: bool,
}

impl StabilityHealthCheck {
    /// Create a new health check with all flags true.
    pub fn healthy() -> Self {
        Self {
            is_contracting: true,
            is_converged: true,
            is_stable: true,
            is_monotonic: true,
        }
    }

    /// Check if all health indicators are passing.
    pub fn all_ok(&self) -> bool {
        self.is_contracting && self.is_converged && self.is_stable && self.is_monotonic
    }
}

/// Perform health check on solver stability.
///
/// # Arguments
/// * `residuals` - Residual history (per-iteration norms)
/// * `contraction_threshold` - Maximum acceptable contraction ratio (default 0.95)
/// * `residual_threshold` - Maximum acceptable final residual (default 0.01)
pub fn stability_health_check(
    residuals: &[f32],
    contraction_threshold: f32,
    residual_threshold: f32,
) -> StabilityHealthCheck {
    if residuals.is_empty() {
        return StabilityHealthCheck::healthy();
    }

    // Compute contraction ratio
    let r = contraction_ratio(residuals, 2);

    // Final residual
    let final_res = residuals[residuals.len() - 1];

    // Check for NaN/Inf
    let is_stable = residuals.iter().all(|&v| v.is_finite());

    // Check monotonic decrease
    let is_monotonic = residuals
        .windows(2)
        .all(|w| w[1] <= w[0] + 1e-8); // Allow tiny tolerance

    StabilityHealthCheck {
        is_contracting: r < contraction_threshold,
        is_converged: final_res < residual_threshold,
        is_stable,
        is_monotonic,
    }
}

/// Estimate Lipschitz constant from input-output pairs.
///
/// Computes: max_{i,j} ||f(z_i) - f(z_j)|| / ||z_i - z_j||
///
/// # Arguments
/// * `z_sequence` - Input sequence, each element is a vector
/// * `output_sequence` - Output sequence, same structure as z_sequence
/// * `max_samples` - Maximum number of pairs to sample (for large N)
pub fn compute_lipschitz_estimate(
    z_sequence: &[Vec<f32>],
    output_sequence: &[Vec<f32>],
    max_samples: usize,
) -> f32 {
    let n = z_sequence.len();

    if n < 2 {
        return 0.0;
    }

    debug_assert_eq!(n, output_sequence.len());

    // Sample indices (uniform spacing for large N)
    let sample_size = n.min(max_samples);
    let step = if sample_size < n {
        (n - 1) as f32 / (sample_size - 1) as f32
    } else {
        1.0
    };

    let indices: Vec<usize> = (0..sample_size)
        .map(|i| ((i as f32) * step).round() as usize)
        .map(|i| i.min(n - 1))
        .collect();

    let mut max_ratio: f32 = 0.0;

    for i in 0..indices.len() {
        for j in (i + 1)..indices.len() {
            let idx_i = indices[i];
            let idx_j = indices[j];

            let input_dist = l2_distance(&z_sequence[idx_i], &z_sequence[idx_j]);
            let output_dist = l2_distance(&output_sequence[idx_i], &output_sequence[idx_j]);

            if input_dist > 1e-6 {
                let ratio = output_dist / input_dist;
                if ratio > max_ratio {
                    max_ratio = ratio;
                }
            }
        }
    }

    max_ratio
}

/// Convergence time estimation result.
#[derive(Debug, Clone)]
pub struct ConvergenceEstimate {
    /// Estimated iterations to reach target
    pub estimated_iters: usize,
    /// Whether target is already achieved
    pub already_converged: bool,
    /// Estimated contraction rate used
    pub contraction_rate: f32,
}

/// Estimate number of iterations needed to reach target residual.
///
/// Uses exponential decay model: res_final * r^k = target
/// Solving: k = log(target / res_final) / log(r)
///
/// # Arguments
/// * `residuals` - Residual history
/// * `target_residual` - Target residual value
/// * `contraction_rate` - Known rate (if None, estimate from residuals)
pub fn estimate_convergence_time(
    residuals: &[f32],
    target_residual: f32,
    contraction_rate: Option<f32>,
) -> ConvergenceEstimate {
    if residuals.is_empty() {
        return ConvergenceEstimate {
            estimated_iters: 0,
            already_converged: false,
            contraction_rate: 1.0,
        };
    }

    let final_res = residuals[residuals.len() - 1];

    // Check if already converged
    if final_res <= target_residual {
        return ConvergenceEstimate {
            estimated_iters: residuals.len(),
            already_converged: true,
            contraction_rate: contraction_ratio(residuals, 2),
        };
    }

    // Get or estimate contraction rate
    let r = match contraction_rate {
        Some(rate) => rate,
        None => {
            if residuals.len() >= 2 {
                contraction_ratio(residuals, 2)
            } else {
                return ConvergenceEstimate {
                    estimated_iters: 0,
                    already_converged: false,
                    contraction_rate: 1.0,
                };
            }
        }
    };

    // Can't converge if not contracting
    if r >= 1.0 || r <= 0.0 {
        return ConvergenceEstimate {
            estimated_iters: 0,
            already_converged: false,
            contraction_rate: r,
        };
    }

    // k = log(target / res_final) / log(r)
    let k_est = (target_residual / final_res).ln() / r.ln();
    let k_est = k_est.ceil().max(0.0) as usize;

    ConvergenceEstimate {
        estimated_iters: k_est,
        already_converged: false,
        contraction_rate: r,
    }
}

/// Compute change rates between consecutive states.
///
/// # Arguments
/// * `current` - Current state vector
/// * `previous` - Previous state vector
/// * `dt` - Time delta (for rate calculation)
///
/// # Returns
/// Change rate: ||current - previous|| / dt
pub fn compute_change_rate(current: &[f32], previous: &[f32], dt: f32) -> f32 {
    debug_assert_eq!(current.len(), previous.len());
    let dist = l2_distance(current, previous);
    dist / dt.max(1e-8)
}

/// Compute total energy of state (sum of norms).
pub fn compute_total_energy(states: &[&[f32]]) -> f32 {
    states.iter().map(|s| l2_norm(s)).sum()
}

/// Compute stability confidence from change rate.
///
/// stability = max(0, 1 - change_rate / scale)
pub fn compute_stability_confidence(change_rate: f32, scale: f32) -> f32 {
    (1.0 - change_rate / scale).max(0.0)
}

/// Full diagnostics for equilibrium state.
#[derive(Debug, Clone)]
pub struct EquilibriumDiagnostics {
    /// Distance to equilibrium (residual)
    pub distance_to_equilibrium: f32,
    /// Stability confidence [0, 1]
    pub stability_confidence: f32,
    /// Change rates per timescale
    pub change_rates: Vec<f32>,
    /// Total energy (sum of norms)
    pub total_energy: f32,
    /// Estimated Lipschitz constant
    pub lipschitz_estimate: f32,
    /// Health check results
    pub health: StabilityHealthCheck,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_contraction_ratio_converging() {
        // Decreasing residuals → ratio < 1
        let residuals = vec![1.0, 0.5, 0.25, 0.125];
        let r = contraction_ratio(&residuals, 2);
        assert!(r < 1.0);
        assert!((r - 0.5).abs() < 0.1);
    }

    #[test]
    fn test_contraction_ratio_diverging() {
        // Increasing residuals → ratio > 1
        let residuals = vec![0.1, 0.2, 0.4, 0.8];
        let r = contraction_ratio(&residuals, 2);
        assert!(r > 1.0);
    }

    #[test]
    fn test_headroom() {
        // Lipschitz = 0.8 → headroom = 0.2
        assert!((headroom(0.8, None) - 0.2).abs() < 1e-6);

        // With metric scale 1.2 and Lipschitz 0.8 → headroom = 1 - 0.96 = 0.04
        assert!((headroom(0.8, Some(1.2)) - 0.04).abs() < 1e-6);

        // Unstable: Lipschitz = 1.1 → headroom = -0.1
        assert!(headroom(1.1, None) < 0.0);
    }

    #[test]
    fn test_stability_health_check() {
        // Good case: converging
        let residuals = vec![1.0, 0.5, 0.25, 0.001];
        let health = stability_health_check(&residuals, 0.95, 0.01);
        assert!(health.is_contracting);
        assert!(health.is_converged);
        assert!(health.is_stable);
        assert!(health.is_monotonic);

        // Bad case: not converged
        let residuals2 = vec![1.0, 0.9, 0.8, 0.7];
        let health2 = stability_health_check(&residuals2, 0.95, 0.01);
        assert!(health2.is_contracting);
        assert!(!health2.is_converged); // final = 0.7 > 0.01
    }

    #[test]
    fn test_compute_lipschitz_estimate() {
        // Identity function: Lipschitz = 1
        let z_seq = vec![vec![0.0, 0.0], vec![1.0, 0.0], vec![2.0, 0.0]];
        let out_seq = z_seq.clone();
        let lip = compute_lipschitz_estimate(&z_seq, &out_seq, 100);
        assert!((lip - 1.0).abs() < 0.1);

        // Contractive function (0.5x): Lipschitz ≈ 0.5
        let z_seq2 = vec![vec![0.0, 0.0], vec![2.0, 0.0], vec![4.0, 0.0]];
        let out_seq2: Vec<Vec<f32>> = z_seq2
            .iter()
            .map(|v| v.iter().map(|&x| x * 0.5).collect())
            .collect();
        let lip2 = compute_lipschitz_estimate(&z_seq2, &out_seq2, 100);
        assert!((lip2 - 0.5).abs() < 0.1);
    }

    #[test]
    fn test_estimate_convergence_time() {
        // Already converged
        let residuals = vec![0.1, 0.01, 0.001];
        let est = estimate_convergence_time(&residuals, 0.01, None);
        assert!(est.already_converged);

        // Need more iterations
        let residuals2 = vec![1.0, 0.5, 0.25];
        let est2 = estimate_convergence_time(&residuals2, 0.001, None);
        assert!(!est2.already_converged);
        assert!(est2.estimated_iters > 0);
    }
}

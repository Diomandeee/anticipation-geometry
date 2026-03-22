//! Proximal operators for constrained optimization.
//!
//! High-performance implementations of:
//! - Box projection (L∞ constraints)
//! - L2 proximal operator (soft thresholding)
//! - Group L2 proximal operator
//! - EMA (exponentially weighted moving average)
//!
//! These are core building blocks used in equilibrium solvers.

/// Box projection: clamp values to [lower, upper].
///
/// Mathematically: P_{[l,u]}(x) = clamp(x, l, u)
#[inline]
pub fn box_project_scalar(x: f32, lower: f32, upper: f32) -> f32 {
    x.clamp(lower, upper)
}

/// Box projection for a vector (in-place).
#[inline]
pub fn box_project_inplace(x: &mut [f32], lower: f32, upper: f32) {
    for v in x.iter_mut() {
        *v = v.clamp(lower, upper);
    }
}

/// Box projection for a vector, returning new vector.
pub fn box_project(x: &[f32], lower: f32, upper: f32) -> Vec<f32> {
    x.iter().map(|&v| v.clamp(lower, upper)).collect()
}

/// L2 proximal operator (elementwise soft thresholding).
///
/// prox_τ(x) = sign(x) * max(|x| - τ, 0)
///           = x * max(1 - τ/|x|, 0)
#[inline]
pub fn l2_prox_scalar(x: f32, tau: f32) -> f32 {
    if tau <= 0.0 {
        return x;
    }
    let abs_x = x.abs();
    let scale = (1.0 - tau / (abs_x + 1e-8)).max(0.0);
    x * scale
}

/// L2 proximal operator for a vector (in-place).
pub fn l2_prox_inplace(x: &mut [f32], tau: f32) {
    if tau <= 0.0 {
        return;
    }
    for v in x.iter_mut() {
        let abs_v = v.abs();
        let scale = (1.0 - tau / (abs_v + 1e-8)).max(0.0);
        *v *= scale;
    }
}

/// L2 proximal operator for a vector, returning new vector.
pub fn l2_prox(x: &[f32], tau: f32) -> Vec<f32> {
    if tau <= 0.0 {
        return x.to_vec();
    }
    x.iter()
        .map(|&v| {
            let abs_v = v.abs();
            let scale = (1.0 - tau / (abs_v + 1e-8)).max(0.0);
            v * scale
        })
        .collect()
}

/// L2 proximal pull toward encoder latents.
///
/// v' = v + weight * (enc - v)
///    = (1 - weight) * v + weight * enc
#[inline]
pub fn l2_prox_pull_scalar(v: f32, enc: f32, weight: f32) -> f32 {
    v + weight * (enc - v)
}

/// L2 proximal pull for vectors (in-place).
pub fn l2_prox_pull_inplace(v: &mut [f32], enc: &[f32], weight: f32) {
    debug_assert_eq!(v.len(), enc.len());
    for (vi, &ei) in v.iter_mut().zip(enc.iter()) {
        *vi += weight * (ei - *vi);
    }
}

/// L2 proximal pull for vectors, returning new vector.
pub fn l2_prox_pull(v: &[f32], enc: &[f32], weight: f32) -> Vec<f32> {
    debug_assert_eq!(v.len(), enc.len());
    v.iter()
        .zip(enc.iter())
        .map(|(&vi, &ei)| vi + weight * (ei - vi))
        .collect()
}

/// Group L2 proximal operator.
///
/// For each group g: prox_τ(x_g) = (1 - τ/||x_g||)_+ * x_g
///
/// # Arguments
/// * `x` - Input vector
/// * `tau` - Threshold parameter
/// * `group_sizes` - Size of each group (must sum to x.len())
pub fn group_l2_prox(x: &[f32], tau: f32, group_sizes: &[usize]) -> Vec<f32> {
    if tau <= 0.0 {
        return x.to_vec();
    }

    let total: usize = group_sizes.iter().sum();
    debug_assert_eq!(x.len(), total, "Group sizes must sum to x.len()");

    let mut result = Vec::with_capacity(x.len());
    let mut start = 0;

    for &size in group_sizes {
        let end = start + size;
        let group = &x[start..end];

        // Compute L2 norm of group
        let norm: f32 = group.iter().map(|&v| v * v).sum::<f32>().sqrt();
        let scale = (1.0 - tau / (norm + 1e-8)).max(0.0);

        // Scale group
        for &v in group {
            result.push(v * scale);
        }

        start = end;
    }

    result
}

/// Exponentially weighted moving average (single step).
///
/// y[n] = α * y[n-1] + (1 - α) * x[n]
#[inline]
pub fn ema_scalar(x_new: f32, x_prev: f32, alpha: f32) -> f32 {
    alpha * x_prev + (1.0 - alpha) * x_new
}

/// EMA for vectors (in-place update of x_prev).
pub fn ema_inplace(x_prev: &mut [f32], x_new: &[f32], alpha: f32) {
    debug_assert_eq!(x_prev.len(), x_new.len());
    let one_minus_alpha = 1.0 - alpha;
    for (prev, &new) in x_prev.iter_mut().zip(x_new.iter()) {
        *prev = alpha * *prev + one_minus_alpha * new;
    }
}

/// EMA for vectors, returning new vector.
pub fn ema(x_prev: &[f32], x_new: &[f32], alpha: f32) -> Vec<f32> {
    debug_assert_eq!(x_prev.len(), x_new.len());
    let one_minus_alpha = 1.0 - alpha;
    x_prev
        .iter()
        .zip(x_new.iter())
        .map(|(&prev, &new)| alpha * prev + one_minus_alpha * new)
        .collect()
}

/// EMA for an entire array (forward pass).
pub fn ema_array(values: &[f32], alpha: f32, init: Option<f32>) -> Vec<f32> {
    if values.is_empty() {
        return Vec::new();
    }

    let mut result = Vec::with_capacity(values.len());
    let first = init.unwrap_or(values[0]);
    result.push(first);

    let one_minus_alpha = 1.0 - alpha;
    let mut prev = first;

    for &v in &values[1..] {
        let new = alpha * prev + one_minus_alpha * v;
        result.push(new);
        prev = new;
    }

    result
}

/// Compute L2 norm of a vector.
#[inline]
pub fn l2_norm(x: &[f32]) -> f32 {
    x.iter().map(|&v| v * v).sum::<f32>().sqrt()
}

/// Compute squared L2 norm of a vector.
#[inline]
pub fn l2_norm_squared(x: &[f32]) -> f32 {
    x.iter().map(|&v| v * v).sum()
}

/// Compute L2 distance between two vectors.
#[inline]
pub fn l2_distance(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b.iter())
        .map(|(&ai, &bi)| {
            let d = ai - bi;
            d * d
        })
        .sum::<f32>()
        .sqrt()
}

/// Clip state norm to max_norm (returns new vector).
///
/// If ||x|| > max_norm, return x * (max_norm / ||x||)
/// Otherwise return x unchanged.
pub fn clip_state_norm(x: &[f32], max_norm: f32) -> Vec<f32> {
    let norm = l2_norm(x);
    if norm <= max_norm {
        x.to_vec()
    } else {
        let scale = max_norm / norm;
        x.iter().map(|&v| v * scale).collect()
    }
}

/// Clip state norm (in-place).
pub fn clip_state_norm_inplace(x: &mut [f32], max_norm: f32) {
    let norm = l2_norm(x);
    if norm > max_norm {
        let scale = max_norm / norm;
        for v in x.iter_mut() {
            *v *= scale;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_box_project() {
        assert_eq!(box_project_scalar(0.5, 0.0, 1.0), 0.5);
        assert_eq!(box_project_scalar(-0.5, 0.0, 1.0), 0.0);
        assert_eq!(box_project_scalar(1.5, 0.0, 1.0), 1.0);

        let result = box_project(&[-1.0, 0.5, 1.5], 0.0, 1.0);
        assert_eq!(result, vec![0.0, 0.5, 1.0]);
    }

    #[test]
    fn test_l2_prox() {
        // With tau=0, should return input unchanged
        assert!((l2_prox_scalar(0.5, 0.0) - 0.5).abs() < 1e-6);

        // With large tau, should shrink toward zero
        let result = l2_prox_scalar(0.5, 0.3);
        assert!(result.abs() < 0.5);
        assert!(result > 0.0);
    }

    #[test]
    fn test_ema() {
        let result = ema_scalar(1.0, 0.0, 0.9);
        assert!((result - 0.1).abs() < 1e-6); // 0.9 * 0.0 + 0.1 * 1.0 = 0.1

        let result = ema_scalar(1.0, 0.0, 0.0);
        assert!((result - 1.0).abs() < 1e-6); // 0.0 * 0.0 + 1.0 * 1.0 = 1.0
    }

    #[test]
    fn test_l2_norm() {
        assert!((l2_norm(&[3.0, 4.0]) - 5.0).abs() < 1e-6);
        assert!((l2_norm(&[1.0, 0.0]) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_clip_state_norm() {
        let x = vec![3.0, 4.0]; // norm = 5
        let clipped = clip_state_norm(&x, 2.5);
        let norm = l2_norm(&clipped);
        assert!((norm - 2.5).abs() < 1e-5);

        // Should not clip if under max
        let x2 = vec![1.0, 1.0];
        let clipped2 = clip_state_norm(&x2, 10.0);
        assert_eq!(clipped2, x2);
    }
}

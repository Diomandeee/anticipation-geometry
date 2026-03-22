//! Spectral Normalization for Lipschitz-constrained layers.
//!
//! Implements power iteration to estimate and normalize by the spectral norm
//! (largest singular value) of weight matrices. This enforces a 1-Lipschitz
//! constraint on each layer.
//!
//! # Algorithm
//!
//! Power iteration estimates σ₁(W) by iterating:
//!   v = normalize(Wᵀ @ u)
//!   u = normalize(W @ v)
//!   σ = uᵀ @ W @ v
//!
//! Then W_normalized = W / σ has spectral norm 1.
//!
//! # References
//!
//! - Miyato et al., "Spectral Normalization for GANs" (2018)

/// Normalize a vector in-place, returning its original norm.
#[inline]
fn normalize_inplace(v: &mut [f32]) -> f32 {
    let norm: f32 = v.iter().map(|&x| x * x).sum::<f32>().sqrt();
    if norm > 1e-12 {
        for x in v.iter_mut() {
            *x /= norm;
        }
    }
    norm
}

/// Matrix-vector multiply: y = W @ x (W is row-major [rows × cols])
fn matvec(weight: &[f32], x: &[f32], rows: usize, cols: usize, y: &mut [f32]) {
    debug_assert_eq!(weight.len(), rows * cols);
    debug_assert_eq!(x.len(), cols);
    debug_assert_eq!(y.len(), rows);

    for i in 0..rows {
        let row_start = i * cols;
        y[i] = weight[row_start..row_start + cols]
            .iter()
            .zip(x.iter())
            .map(|(&w, &xi)| w * xi)
            .sum();
    }
}

/// Matrix-transpose-vector multiply: y = Wᵀ @ x (W is row-major [rows × cols])
fn matvec_transpose(weight: &[f32], x: &[f32], rows: usize, cols: usize, y: &mut [f32]) {
    debug_assert_eq!(weight.len(), rows * cols);
    debug_assert_eq!(x.len(), rows);
    debug_assert_eq!(y.len(), cols);

    y.fill(0.0);
    for i in 0..rows {
        let row_start = i * cols;
        for j in 0..cols {
            y[j] += weight[row_start + j] * x[i];
        }
    }
}

/// Estimate spectral norm via power iteration.
///
/// # Arguments
/// * `weight` - Row-major weight matrix [rows × cols]
/// * `rows` - Number of rows (output dimension)
/// * `cols` - Number of columns (input dimension)
/// * `u` - Power iteration vector [rows] (updated in-place)
/// * `n_iters` - Number of power iterations
///
/// # Returns
/// Estimated spectral norm σ₁(W)
pub fn spectral_norm_estimate(
    weight: &[f32],
    rows: usize,
    cols: usize,
    u: &mut [f32],
    n_iters: usize,
) -> f32 {
    debug_assert_eq!(weight.len(), rows * cols);
    debug_assert_eq!(u.len(), rows);

    let mut v = vec![0.0; cols];

    for _ in 0..n_iters {
        // v = normalize(Wᵀ @ u)
        matvec_transpose(weight, u, rows, cols, &mut v);
        normalize_inplace(&mut v);

        // u = normalize(W @ v)
        matvec(weight, &v, rows, cols, u);
        normalize_inplace(u);
    }

    // σ = uᵀ @ W @ v
    let mut wv = vec![0.0; rows];
    matvec(weight, &v, rows, cols, &mut wv);

    u.iter().zip(wv.iter()).map(|(&ui, &wi)| ui * wi).sum()
}

/// Compute spectrally normalized weights.
///
/// Returns W / σ₁(W), ensuring the output has spectral norm ≤ 1.
///
/// # Arguments
/// * `weight` - Row-major weight matrix [rows × cols]
/// * `rows` - Number of rows
/// * `cols` - Number of columns
/// * `u` - Power iteration vector (updated in-place for warm start)
/// * `n_iters` - Number of power iterations
///
/// # Returns
/// (normalized_weight, sigma) where normalized_weight = weight / sigma
pub fn spectral_normalize(
    weight: &[f32],
    rows: usize,
    cols: usize,
    u: &mut [f32],
    n_iters: usize,
) -> (Vec<f32>, f32) {
    let sigma = spectral_norm_estimate(weight, rows, cols, u, n_iters);

    // Avoid division by zero
    let sigma_safe = sigma.max(1e-12);

    let normalized: Vec<f32> = weight.iter().map(|&w| w / sigma_safe).collect();

    (normalized, sigma)
}

/// Dense layer with spectral normalization.
///
/// Maintains power iteration state for warm-starting between forward passes.
#[derive(Debug, Clone)]
pub struct SpectralNormDense {
    /// Raw weight matrix [out_features × in_features], row-major.
    pub weight: Vec<f32>,
    /// Optional bias vector [out_features].
    pub bias: Option<Vec<f32>>,
    /// Power iteration vector [out_features].
    pub u: Vec<f32>,
    /// Number of input features (columns).
    pub in_features: usize,
    /// Number of output features (rows).
    pub out_features: usize,
    /// Number of power iterations per forward pass.
    pub power_iters: usize,
    /// Cached spectral norm from last forward.
    cached_sigma: f32,
}

impl SpectralNormDense {
    /// Create a new spectral-normalized dense layer.
    ///
    /// Initializes with Xavier/Glorot uniform initialization scaled to have
    /// spectral norm approximately 1.
    pub fn new(in_features: usize, out_features: usize, bias: bool, power_iters: usize) -> Self {
        // Xavier initialization: scale ~ sqrt(2 / (in + out))
        // We scale to roughly unit spectral norm
        let scale = (2.0 / (in_features + out_features) as f32).sqrt();

        // Deterministic initialization for reproducibility
        let mut weight = Vec::with_capacity(out_features * in_features);
        for i in 0..(out_features * in_features) {
            // Simple LCG for deterministic pseudo-random
            let t = ((i as f32 * 0.618033988749895) % 1.0) * 2.0 - 1.0;
            weight.push(t * scale);
        }

        let bias_vec = if bias {
            Some(vec![0.0; out_features])
        } else {
            None
        };

        // Initialize u as uniform (will converge after a few iterations)
        let mut u = vec![1.0 / (out_features as f32).sqrt(); out_features];
        normalize_inplace(&mut u);

        Self {
            weight,
            bias: bias_vec,
            u,
            in_features,
            out_features,
            power_iters,
            cached_sigma: 1.0,
        }
    }

    /// Create from existing weights.
    pub fn from_weights(
        weight: Vec<f32>,
        bias: Option<Vec<f32>>,
        in_features: usize,
        out_features: usize,
        power_iters: usize,
    ) -> Self {
        assert_eq!(
            weight.len(),
            in_features * out_features,
            "Weight size mismatch"
        );
        if let Some(ref b) = bias {
            assert_eq!(b.len(), out_features, "Bias size mismatch");
        }

        let mut u = vec![1.0 / (out_features as f32).sqrt(); out_features];
        normalize_inplace(&mut u);

        Self {
            weight,
            bias,
            u,
            in_features,
            out_features,
            power_iters,
            cached_sigma: 1.0,
        }
    }

    /// Forward pass without updating spectral state.
    ///
    /// Uses cached sigma from last update.
    pub fn forward(&self, x: &[f32]) -> Vec<f32> {
        debug_assert_eq!(x.len(), self.in_features);

        let mut y = vec![0.0; self.out_features];

        // y = (W / σ) @ x + b
        let sigma_safe = self.cached_sigma.max(1e-12);
        for i in 0..self.out_features {
            let row_start = i * self.in_features;
            y[i] = self.weight[row_start..row_start + self.in_features]
                .iter()
                .zip(x.iter())
                .map(|(&w, &xi)| w * xi)
                .sum::<f32>()
                / sigma_safe;
        }

        if let Some(ref bias) = self.bias {
            for (yi, &bi) in y.iter_mut().zip(bias.iter()) {
                *yi += bi;
            }
        }

        y
    }

    /// Forward pass into buffer without updating spectral state.
    pub fn forward_into(&self, x: &[f32], y: &mut [f32]) {
        debug_assert_eq!(x.len(), self.in_features);
        debug_assert_eq!(y.len(), self.out_features);

        let sigma_safe = self.cached_sigma.max(1e-12);
        for i in 0..self.out_features {
            let row_start = i * self.in_features;
            y[i] = self.weight[row_start..row_start + self.in_features]
                .iter()
                .zip(x.iter())
                .map(|(&w, &xi)| w * xi)
                .sum::<f32>()
                / sigma_safe;
        }

        if let Some(ref bias) = self.bias {
            for (yi, &bi) in y.iter_mut().zip(bias.iter()) {
                *yi += bi;
            }
        }
    }

    /// Forward pass with spectral normalization update.
    ///
    /// Updates power iteration vectors and cached sigma.
    /// Use this during training or when weights have changed.
    pub fn forward_with_update(&mut self, x: &[f32]) -> Vec<f32> {
        self.update_spectral_state();
        self.forward(x)
    }

    /// Update spectral normalization state.
    ///
    /// Run power iteration to update u and cached_sigma.
    pub fn update_spectral_state(&mut self) {
        self.cached_sigma = spectral_norm_estimate(
            &self.weight,
            self.out_features,
            self.in_features,
            &mut self.u,
            self.power_iters,
        );
    }

    /// Refresh spectral state with extra iterations.
    ///
    /// Use after loading weights or for better accuracy.
    pub fn refresh_spectral_state(&mut self, n_iters: usize) {
        self.cached_sigma = spectral_norm_estimate(
            &self.weight,
            self.out_features,
            self.in_features,
            &mut self.u,
            n_iters,
        );
    }

    /// Get current estimated spectral norm.
    pub fn spectral_norm(&self) -> f32 {
        self.cached_sigma
    }

    /// Get normalized weights (W / σ).
    pub fn normalized_weight(&self) -> Vec<f32> {
        let sigma_safe = self.cached_sigma.max(1e-12);
        self.weight.iter().map(|&w| w / sigma_safe).collect()
    }

    /// Set weights from external source.
    pub fn set_weight(&mut self, weight: Vec<f32>) {
        assert_eq!(weight.len(), self.in_features * self.out_features);
        self.weight = weight;
        // Invalidate cache - will be updated on next forward_with_update
    }

    /// Set bias from external source.
    pub fn set_bias(&mut self, bias: Option<Vec<f32>>) {
        if let Some(ref b) = bias {
            assert_eq!(b.len(), self.out_features);
        }
        self.bias = bias;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_normalize_inplace() {
        let mut v = vec![3.0, 4.0];
        let norm = normalize_inplace(&mut v);
        assert!((norm - 5.0).abs() < 1e-6);
        assert!((v[0] - 0.6).abs() < 1e-6);
        assert!((v[1] - 0.8).abs() < 1e-6);
    }

    #[test]
    fn test_matvec() {
        // 2x3 matrix [[1,2,3], [4,5,6]]
        let w = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let x = vec![1.0, 1.0, 1.0];
        let mut y = vec![0.0; 2];

        matvec(&w, &x, 2, 3, &mut y);

        assert!((y[0] - 6.0).abs() < 1e-6); // 1+2+3
        assert!((y[1] - 15.0).abs() < 1e-6); // 4+5+6
    }

    #[test]
    fn test_matvec_transpose() {
        // Same 2x3 matrix
        let w = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let x = vec![1.0, 1.0];
        let mut y = vec![0.0; 3];

        matvec_transpose(&w, &x, 2, 3, &mut y);

        assert!((y[0] - 5.0).abs() < 1e-6); // 1+4
        assert!((y[1] - 7.0).abs() < 1e-6); // 2+5
        assert!((y[2] - 9.0).abs() < 1e-6); // 3+6
    }

    #[test]
    fn test_spectral_norm_identity() {
        // Identity matrix has spectral norm 1
        let w = vec![1.0, 0.0, 0.0, 1.0];
        let mut u = vec![0.7071, 0.7071];

        let sigma = spectral_norm_estimate(&w, 2, 2, &mut u, 10);

        assert!(
            (sigma - 1.0).abs() < 1e-3,
            "Identity should have σ=1, got {}",
            sigma
        );
    }

    #[test]
    fn test_spectral_norm_diagonal() {
        // Diagonal [2, 0.5] has spectral norm 2
        let w = vec![2.0, 0.0, 0.0, 0.5];
        let mut u = vec![0.7071, 0.7071];

        let sigma = spectral_norm_estimate(&w, 2, 2, &mut u, 20);

        assert!(
            (sigma - 2.0).abs() < 1e-2,
            "Diag [2, 0.5] should have σ=2, got {}",
            sigma
        );
    }

    #[test]
    fn test_spectral_normalize() {
        // After normalization, spectral norm should be ~1
        let w = vec![2.0, 0.0, 0.0, 2.0];
        let mut u = vec![0.7071, 0.7071];

        let (normalized, sigma) = spectral_normalize(&w, 2, 2, &mut u, 10);

        assert!((sigma - 2.0).abs() < 1e-2);

        // Check normalized has σ ≈ 1
        let mut u2 = vec![0.7071, 0.7071];
        let sigma_norm = spectral_norm_estimate(&normalized, 2, 2, &mut u2, 10);
        assert!(
            (sigma_norm - 1.0).abs() < 1e-2,
            "Normalized should have σ=1, got {}",
            sigma_norm
        );
    }

    #[test]
    fn test_spectral_norm_dense_forward() {
        let mut layer = SpectralNormDense::new(3, 2, true, 5);

        // Update spectral state
        layer.refresh_spectral_state(20);

        let x = vec![1.0, 0.0, 0.0];
        let y = layer.forward(&x);

        assert_eq!(y.len(), 2);
    }

    #[test]
    fn test_spectral_norm_dense_lipschitz() {
        // After spectral normalization, layer should be 1-Lipschitz
        let mut layer = SpectralNormDense::new(4, 4, false, 10);
        layer.refresh_spectral_state(50);

        // Test with two different inputs
        let x1 = vec![1.0, 0.0, 0.0, 0.0];
        let x2 = vec![0.0, 1.0, 0.0, 0.0];

        let y1 = layer.forward(&x1);
        let y2 = layer.forward(&x2);

        let input_dist: f32 = x1
            .iter()
            .zip(x2.iter())
            .map(|(&a, &b)| (a - b).powi(2))
            .sum::<f32>()
            .sqrt();

        let output_dist: f32 = y1
            .iter()
            .zip(y2.iter())
            .map(|(&a, &b)| (a - b).powi(2))
            .sum::<f32>()
            .sqrt();

        // 1-Lipschitz: ||f(x1) - f(x2)|| <= ||x1 - x2||
        assert!(
            output_dist <= input_dist * 1.1, // Small tolerance for numerical error
            "Layer should be 1-Lipschitz: output_dist={} > input_dist={}",
            output_dist,
            input_dist
        );
    }

    #[test]
    fn test_from_weights() {
        let weight = vec![1.0, 0.0, 0.0, 1.0];
        let bias = Some(vec![0.1, 0.2]);

        let layer = SpectralNormDense::from_weights(weight.clone(), bias.clone(), 2, 2, 5);

        assert_eq!(layer.weight, weight);
        assert_eq!(layer.bias, bias);
        assert_eq!(layer.in_features, 2);
        assert_eq!(layer.out_features, 2);
    }
}

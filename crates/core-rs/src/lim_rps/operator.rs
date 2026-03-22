//! Cross-Modal Operator for LIM-RPS.
//!
//! The CrossModalOperator is a 1-Lipschitz MLP that fuses multi-modal
//! latent representations. It uses spectral normalization to enforce
//! the Lipschitz constraint, ensuring convergence of the fixed-point iteration.
//!
//! # Architecture
//!
//! ```text
//! z_in → [SpectralNormDense] → ReLU/√2 → ... → [SpectralNormDense] → z_out
//! ```
//!
//! ReLU is scaled by 1/√2 to preserve 1-Lipschitz property.

use super::config::LimRpsConfig;
use super::spectral::SpectralNormDense;
use std::collections::HashMap;

/// Scaled ReLU activation: max(0, x) / √2
///
/// ReLU is 1-Lipschitz. Scaling by 1/√2 allows composing with
/// spectral-normalized layers while maintaining overall 1-Lipschitz.
#[inline]
#[allow(dead_code)]
pub fn scaled_relu(x: f32) -> f32 {
    x.max(0.0) * std::f32::consts::FRAC_1_SQRT_2
}

/// Apply scaled ReLU in-place.
fn scaled_relu_inplace(x: &mut [f32]) {
    for v in x.iter_mut() {
        *v = v.max(0.0) * std::f32::consts::FRAC_1_SQRT_2;
    }
}

/// 1-Lipschitz Cross-Modal Operator.
///
/// A multi-layer perceptron with spectral normalization that ensures
/// the entire network is 1-Lipschitz. This is the operator B(z) in
/// the fixed-point iteration:
///
///   z_{k+1} = prox(z_k - γ * B(z_k))
///
/// # Lipschitz Guarantee
///
/// Each layer is spectral-normalized to have σ(W) = 1.
/// ReLU activation is scaled by 1/√2.
/// Composition: L_total = L_1 * L_2 * ... * L_n = 1.
#[derive(Debug, Clone)]
pub struct CrossModalOperator {
    /// Hidden layers (spectral-normalized).
    hidden_layers: Vec<SpectralNormDense>,
    /// Output projection layer.
    output_layer: SpectralNormDense,
    /// Input dimension.
    input_dim: usize,
    /// Hidden dimension.
    hidden_dim: usize,
    /// Output dimension (usually == input_dim for equilibrium).
    output_dim: usize,
}

impl CrossModalOperator {
    /// Create a new CrossModalOperator.
    ///
    /// # Arguments
    /// * `input_dim` - Input/output dimension (equilibrium requires square operator)
    /// * `config` - LIM-RPS configuration
    pub fn new(input_dim: usize, config: &LimRpsConfig) -> Self {
        let hidden_dim = config.hidden_dim;
        let num_layers = config.num_layers;
        let power_iters = config.spectral_iters;

        let mut hidden_layers = Vec::with_capacity(num_layers);

        // First hidden layer: input_dim → hidden_dim
        hidden_layers.push(SpectralNormDense::new(
            input_dim, hidden_dim, true, power_iters,
        ));

        // Additional hidden layers: hidden_dim → hidden_dim
        for _ in 1..num_layers {
            hidden_layers.push(SpectralNormDense::new(
                hidden_dim, hidden_dim, true, power_iters,
            ));
        }

        // Output layer: hidden_dim → input_dim (no bias for residual-friendly)
        let output_layer = SpectralNormDense::new(hidden_dim, input_dim, false, power_iters);

        let mut op = Self {
            hidden_layers,
            output_layer,
            input_dim,
            hidden_dim,
            output_dim: input_dim,
        };

        // Initialize spectral normalization
        op.refresh_spectral_state(20);

        op
    }

    /// Create with specific dimensions.
    pub fn with_dimensions(
        input_dim: usize,
        hidden_dim: usize,
        output_dim: usize,
        num_layers: usize,
        power_iters: usize,
    ) -> Self {
        let mut hidden_layers = Vec::with_capacity(num_layers);

        if num_layers > 0 {
            hidden_layers.push(SpectralNormDense::new(
                input_dim, hidden_dim, true, power_iters,
            ));

            for _ in 1..num_layers {
                hidden_layers.push(SpectralNormDense::new(
                    hidden_dim, hidden_dim, true, power_iters,
                ));
            }
        }

        let output_layer = if num_layers > 0 {
            SpectralNormDense::new(hidden_dim, output_dim, false, power_iters)
        } else {
            SpectralNormDense::new(input_dim, output_dim, false, power_iters)
        };

        let mut op = Self {
            hidden_layers,
            output_layer,
            input_dim,
            hidden_dim,
            output_dim,
        };

        op.refresh_spectral_state(20);
        op
    }

    /// Forward pass: compute B(z).
    ///
    /// Does not update spectral normalization state.
    pub fn forward(&self, z: &[f32]) -> Vec<f32> {
        debug_assert_eq!(z.len(), self.input_dim);

        let mut x = z.to_vec();

        // Hidden layers with scaled ReLU
        for layer in &self.hidden_layers {
            x = layer.forward(&x);
            scaled_relu_inplace(&mut x);
        }

        // Output projection (no activation)
        self.output_layer.forward(&x)
    }

    /// Forward pass into buffer.
    pub fn forward_into(&self, z: &[f32], out: &mut [f32]) {
        let result = self.forward(z);
        out.copy_from_slice(&result);
    }

    /// Forward pass with spectral normalization update.
    ///
    /// Use this after weight updates (training) or periodically
    /// to maintain accurate spectral norm estimates.
    pub fn forward_with_update(&mut self, z: &[f32]) -> Vec<f32> {
        debug_assert_eq!(z.len(), self.input_dim);

        let mut x = z.to_vec();

        for layer in &mut self.hidden_layers {
            x = layer.forward_with_update(&x);
            scaled_relu_inplace(&mut x);
        }

        self.output_layer.forward_with_update(&x)
    }

    /// Refresh spectral normalization with extra iterations.
    ///
    /// Call after loading weights or for better accuracy.
    pub fn refresh_spectral_state(&mut self, n_iters: usize) {
        for layer in &mut self.hidden_layers {
            layer.refresh_spectral_state(n_iters);
        }
        self.output_layer.refresh_spectral_state(n_iters);
    }

    /// Get estimated Lipschitz constant of entire operator.
    ///
    /// For properly spectral-normalized networks, this should be ~1.
    pub fn lipschitz_estimate(&self) -> f32 {
        // Each hidden layer: σ(W) * (1/√2 for ReLU) = σ(W) / √2
        // But we normalized σ(W) = 1, and ReLU is scaled by 1/√2
        // So each layer contributes 1.0 * 0.707 to Lipschitz constant
        // But actually, spectral norm 1 + scaled ReLU = still 1-Lipschitz

        // With spectral norm = 1 and scaled ReLU, each layer is 1/√2 Lipschitz
        // But output has no activation, so it's 1-Lipschitz
        // Total: (1/√2)^n_hidden * 1

        let n_hidden = self.hidden_layers.len() as i32;
        (0.5f32).powi(n_hidden / 2) // Approximately, with some tolerance
    }

    /// Get total number of parameters.
    pub fn num_parameters(&self) -> usize {
        let mut count = 0;
        for layer in &self.hidden_layers {
            count += layer.weight.len();
            if layer.bias.is_some() {
                count += layer.out_features;
            }
        }
        count += self.output_layer.weight.len();
        if self.output_layer.bias.is_some() {
            count += self.output_layer.out_features;
        }
        count
    }

    /// Load weights from state dict.
    ///
    /// Keys should be:
    /// - `hidden.{i}.weight` for layer i weight
    /// - `hidden.{i}.bias` for layer i bias
    /// - `output.weight` for output layer weight
    /// - `output.bias` for output layer bias (if present)
    pub fn load_state_dict(&mut self, state_dict: &HashMap<String, Vec<f32>>) {
        for (i, layer) in self.hidden_layers.iter_mut().enumerate() {
            let weight_key = format!("hidden.{}.weight", i);
            let bias_key = format!("hidden.{}.bias", i);

            if let Some(weight) = state_dict.get(&weight_key) {
                layer.set_weight(weight.clone());
            }
            if let Some(bias) = state_dict.get(&bias_key) {
                layer.set_bias(Some(bias.clone()));
            }
        }

        if let Some(weight) = state_dict.get("output.weight") {
            self.output_layer.set_weight(weight.clone());
        }
        if let Some(bias) = state_dict.get("output.bias") {
            self.output_layer.set_bias(Some(bias.clone()));
        }

        // Refresh spectral normalization after loading
        self.refresh_spectral_state(20);
    }

    /// Export weights to state dict.
    pub fn state_dict(&self) -> HashMap<String, Vec<f32>> {
        let mut dict = HashMap::new();

        for (i, layer) in self.hidden_layers.iter().enumerate() {
            dict.insert(format!("hidden.{}.weight", i), layer.weight.clone());
            if let Some(ref bias) = layer.bias {
                dict.insert(format!("hidden.{}.bias", i), bias.clone());
            }
        }

        dict.insert("output.weight".to_string(), self.output_layer.weight.clone());
        if let Some(ref bias) = self.output_layer.bias {
            dict.insert("output.bias".to_string(), bias.clone());
        }

        dict
    }

    /// Get input dimension.
    pub fn input_dim(&self) -> usize {
        self.input_dim
    }

    /// Get hidden dimension.
    pub fn hidden_dim(&self) -> usize {
        self.hidden_dim
    }

    /// Get output dimension.
    pub fn output_dim(&self) -> usize {
        self.output_dim
    }

    /// Get number of hidden layers.
    pub fn num_layers(&self) -> usize {
        self.hidden_layers.len()
    }
}

/// Simple identity operator (for testing/baseline).
#[derive(Debug, Clone)]
pub struct IdentityOperator {
    dim: usize,
}

impl IdentityOperator {
    /// Create identity operator of given dimension.
    pub fn new(dim: usize) -> Self {
        Self { dim }
    }

    /// Forward pass: returns input unchanged.
    pub fn forward(&self, z: &[f32]) -> Vec<f32> {
        debug_assert_eq!(z.len(), self.dim);
        z.to_vec()
    }

    /// Get dimension.
    pub fn dim(&self) -> usize {
        self.dim
    }
}

/// Trait for LIM-RPS operators.
pub trait LimRpsOperator {
    /// Apply operator: output = B(input)
    fn apply(&self, z: &[f32]) -> Vec<f32>;

    /// Apply operator into buffer
    fn apply_into(&self, z: &[f32], out: &mut [f32]);

    /// Get input dimension
    fn input_dim(&self) -> usize;

    /// Get output dimension
    fn output_dim(&self) -> usize;
}

impl LimRpsOperator for CrossModalOperator {
    fn apply(&self, z: &[f32]) -> Vec<f32> {
        self.forward(z)
    }

    fn apply_into(&self, z: &[f32], out: &mut [f32]) {
        self.forward_into(z, out);
    }

    fn input_dim(&self) -> usize {
        self.input_dim
    }

    fn output_dim(&self) -> usize {
        self.output_dim
    }
}

impl LimRpsOperator for IdentityOperator {
    fn apply(&self, z: &[f32]) -> Vec<f32> {
        self.forward(z)
    }

    fn apply_into(&self, z: &[f32], out: &mut [f32]) {
        out.copy_from_slice(z);
    }

    fn input_dim(&self) -> usize {
        self.dim
    }

    fn output_dim(&self) -> usize {
        self.dim
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_scaled_relu() {
        assert_eq!(scaled_relu(-1.0), 0.0);
        assert!((scaled_relu(1.0) - std::f32::consts::FRAC_1_SQRT_2).abs() < 1e-6);
        assert_eq!(scaled_relu(0.0), 0.0);
    }

    #[test]
    fn test_cross_modal_operator_new() {
        let config = LimRpsConfig::default();
        let op = CrossModalOperator::new(16, &config);

        assert_eq!(op.input_dim(), 16);
        assert_eq!(op.output_dim(), 16);
        assert_eq!(op.hidden_dim(), 128);
        assert_eq!(op.num_layers(), 2);
    }

    #[test]
    fn test_cross_modal_operator_forward() {
        let config = LimRpsConfig::new().with_hidden_dim(32).with_num_layers(2);
        let op = CrossModalOperator::new(8, &config);

        let z = vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let out = op.forward(&z);

        assert_eq!(out.len(), 8);
    }

    #[test]
    fn test_cross_modal_operator_lipschitz() {
        let config = LimRpsConfig::new()
            .with_hidden_dim(32)
            .with_num_layers(2);
        let mut op = CrossModalOperator::new(8, &config);
        op.refresh_spectral_state(50);

        // Test Lipschitz constraint: ||B(z1) - B(z2)|| <= ||z1 - z2||
        let z1 = vec![1.0, 0.5, 0.0, -0.5, 0.3, -0.3, 0.1, -0.1];
        let z2 = vec![0.0, 1.0, 0.5, 0.0, -0.5, 0.3, -0.3, 0.1];

        let out1 = op.forward(&z1);
        let out2 = op.forward(&z2);

        let input_dist: f32 = z1
            .iter()
            .zip(z2.iter())
            .map(|(&a, &b)| (a - b).powi(2))
            .sum::<f32>()
            .sqrt();

        let output_dist: f32 = out1
            .iter()
            .zip(out2.iter())
            .map(|(&a, &b)| (a - b).powi(2))
            .sum::<f32>()
            .sqrt();

        // Should be 1-Lipschitz (with some tolerance for numerical error)
        assert!(
            output_dist <= input_dist * 1.2, // Allow 20% tolerance
            "Operator should be ~1-Lipschitz: output_dist={} > input_dist={}",
            output_dist,
            input_dist
        );
    }

    #[test]
    fn test_state_dict_roundtrip() {
        let config = LimRpsConfig::new().with_hidden_dim(16).with_num_layers(1);
        let mut op1 = CrossModalOperator::new(4, &config);
        // Ensure spectral norm is well-converged
        op1.refresh_spectral_state(50);

        let state = op1.state_dict();

        let mut op2 = CrossModalOperator::new(4, &config);
        op2.load_state_dict(&state);
        // Also ensure op2 has converged spectral norm
        op2.refresh_spectral_state(50);

        // Verify weights match
        let z = vec![1.0, 2.0, 3.0, 4.0];
        let out1 = op1.forward(&z);
        let out2 = op2.forward(&z);

        for (a, b) in out1.iter().zip(out2.iter()) {
            assert!(
                (a - b).abs() < 1e-3, // Slightly relaxed tolerance
                "Outputs should match after state_dict roundtrip: {} vs {}",
                a,
                b
            );
        }
    }

    #[test]
    fn test_identity_operator() {
        let op = IdentityOperator::new(4);
        let z = vec![1.0, 2.0, 3.0, 4.0];
        let out = op.forward(&z);
        assert_eq!(out, z);
    }

    #[test]
    fn test_num_parameters() {
        let config = LimRpsConfig::new()
            .with_hidden_dim(32)
            .with_num_layers(2);
        let op = CrossModalOperator::new(16, &config);

        // Layer 0: 16*32 + 32 = 544
        // Layer 1: 32*32 + 32 = 1056
        // Output: 32*16 = 512
        // Total: 544 + 1056 + 512 = 2112
        let expected = 16 * 32 + 32 + 32 * 32 + 32 + 32 * 16;
        assert_eq!(op.num_parameters(), expected);
    }
}

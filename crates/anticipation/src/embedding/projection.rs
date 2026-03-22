//! Fixed random projection for regime embedding (v0)
//!
//! FROZEN: Projection matrix is deterministic from seed.

use std::sync::LazyLock;

/// Projection seed for determinism
/// Chosen to be memorable and stable across versions
const PROJECTION_SEED: u64 = 0xCC_A7_1C_1A_7104_2025;

/// Maximum input dimension
const MAX_INPUT_DIM: usize = 128;

/// Maximum output dimension
const MAX_OUTPUT_DIM: usize = 256;

/// Pre-computed projection matrix (thread-safe, lazy initialization)
static PROJECTION_MATRIX: LazyLock<[[f32; MAX_OUTPUT_DIM]; MAX_INPUT_DIM]> = LazyLock::new(|| {
    generate_projection_matrix(PROJECTION_SEED)
});

/// Generate a deterministic random projection matrix
///
/// Uses a simple LCG for reproducibility (no external dependencies).
fn generate_projection_matrix(seed: u64) -> [[f32; MAX_OUTPUT_DIM]; MAX_INPUT_DIM] {
    let mut matrix = [[0.0f32; MAX_OUTPUT_DIM]; MAX_INPUT_DIM];
    let mut state = seed;

    // Simple LCG parameters (same as MINSTD)
    const A: u64 = 48271;
    const M: u64 = 2147483647;

    for row in matrix.iter_mut() {
        for col in row.iter_mut() {
            state = (A.wrapping_mul(state)) % M;
            // Map to [-1, 1] with Gaussian-like distribution approximation
            let u1 = state as f32 / M as f32;
            state = (A.wrapping_mul(state)) % M;
            let u2 = state as f32 / M as f32;

            // Box-Muller-like transform (simplified)
            let z = ((-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos())
                .clamp(-3.0, 3.0);

            // Scale for unit variance in output
            *col = z / (MAX_INPUT_DIM as f32).sqrt();
        }
    }

    matrix
}

/// Project fused features to regime embedding
///
/// # Arguments
///
/// * `features` - Input feature vector (up to 128 dims)
/// * `output_dim` - Desired output dimension (up to 256)
///
/// # Returns
///
/// Regime embedding of the specified dimension.
///
/// # Invariants
///
/// - FROZEN: Uses deterministic projection matrix
/// - Output is L2-normalized
pub fn project_to_regime_embedding(features: &[f32], output_dim: usize) -> Vec<f32> {
    let output_dim = output_dim.min(MAX_OUTPUT_DIM);
    let input_dim = features.len().min(MAX_INPUT_DIM);

    let mut output = vec![0.0f32; output_dim];

    // Matrix-vector multiply
    for (i, out) in output.iter_mut().enumerate() {
        let mut sum = 0.0f32;
        for (j, &feat) in features.iter().take(input_dim).enumerate() {
            sum += PROJECTION_MATRIX[j][i] * feat;
        }
        *out = sum;
    }

    // L2 normalize
    let norm: f32 = output.iter().map(|&x| x * x).sum::<f32>().sqrt();
    if norm > 1e-10 {
        for v in &mut output {
            *v /= norm;
        }
    }

    output
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_projection_determinism() {
        let features = vec![1.0, 2.0, 3.0, 4.0, 5.0];

        let emb1 = project_to_regime_embedding(&features, 64);
        let emb2 = project_to_regime_embedding(&features, 64);

        assert_eq!(emb1, emb2);
    }

    #[test]
    fn test_projection_dimension() {
        let features = vec![1.0; 32];

        let emb = project_to_regime_embedding(&features, 64);
        assert_eq!(emb.len(), 64);

        let emb = project_to_regime_embedding(&features, 128);
        assert_eq!(emb.len(), 128);
    }

    #[test]
    fn test_projection_normalized() {
        let features = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let emb = project_to_regime_embedding(&features, 64);

        let norm: f32 = emb.iter().map(|&x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_zero_input() {
        let features = vec![0.0; 32];
        let emb = project_to_regime_embedding(&features, 64);

        // Should be all zeros (or normalized zeros)
        let norm: f32 = emb.iter().map(|&x| x * x).sum::<f32>().sqrt();
        assert!(norm < 1e-6 || (norm - 1.0).abs() < 1e-6);
    }
}

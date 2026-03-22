//! Latent dynamics features from LIM-RPS stream
//!
//! FROZEN: Feature computation logic must be deterministic.

use crate::config::AnticipationConfig;
use crate::types::LatentFrame;

/// Latent dynamics features extracted from latent window
#[derive(Clone, Debug, Default)]
pub struct LatentFeatures {
    /// Norm of latent vector (activity level)
    pub norm: f32,
    /// Norm of velocity (rate of change)
    pub velocity_norm: f32,
    /// Norm of acceleration
    pub acceleration_norm: f32,
    /// Local predictability (linear fit residual)
    pub predictability: f32,
    /// Coherence of dimensions (cross-correlation)
    pub coherence: f32,
}

impl LatentFeatures {
    /// Convert to feature vector
    pub fn to_vec(&self) -> Vec<f32> {
        vec![
            self.norm,
            self.velocity_norm,
            self.acceleration_norm,
            self.predictability,
            self.coherence,
        ]
    }
}

/// Compute latent dynamics features from latent frames
///
/// # Invariants
///
/// - Deterministic given same input
/// - Uses central differences for derivatives
#[allow(unused_variables)]
pub fn compute_latent_features(
    frames: &[LatentFrame],
    config: &AnticipationConfig,
) -> LatentFeatures {
    if frames.is_empty() {
        return LatentFeatures::default();
    }

    let dt = 1.0 / config.fps;

    // Compute average norm
    let valid_frames: Vec<_> = frames.iter().filter(|f| f.valid).collect();
    if valid_frames.is_empty() {
        return LatentFeatures::default();
    }

    let norm = {
        let sum: f32 = valid_frames.iter()
            .map(|f| compute_vec_norm(&f.z))
            .sum();
        sum / valid_frames.len() as f32
    };

    // Compute velocities using central differences
    let mut velocities: Vec<Vec<f32>> = Vec::new();

    for i in 1..frames.len() - 1 {
        if frames[i - 1].valid && frames[i + 1].valid {
            let v: Vec<f32> = frames[i - 1].z.iter()
                .zip(frames[i + 1].z.iter())
                .map(|(&a, &b)| (b - a) / (2.0 * dt))
                .collect();
            velocities.push(v);
        }
    }

    let velocity_norm = if !velocities.is_empty() {
        let sum: f32 = velocities.iter()
            .map(|v| compute_vec_norm(v))
            .sum();
        sum / velocities.len() as f32
    } else {
        0.0
    };

    // Compute accelerations
    let mut accelerations: Vec<Vec<f32>> = Vec::new();

    for i in 1..velocities.len() {
        let a: Vec<f32> = velocities[i - 1].iter()
            .zip(velocities[i].iter())
            .map(|(&v1, &v2)| (v2 - v1) / dt)
            .collect();
        accelerations.push(a);
    }

    let acceleration_norm = if !accelerations.is_empty() {
        let sum: f32 = accelerations.iter()
            .map(|a| compute_vec_norm(a))
            .sum();
        sum / accelerations.len() as f32
    } else {
        0.0
    };

    // Predictability: fit linear model and measure residual
    let predictability = compute_predictability(&valid_frames);

    // Coherence: placeholder (dimension cross-correlation)
    let coherence = 0.5;

    LatentFeatures {
        norm,
        velocity_norm,
        acceleration_norm,
        predictability,
        coherence,
    }
}

/// Compute L2 norm of a vector
fn compute_vec_norm(v: &[f32]) -> f32 {
    v.iter().map(|&x| x * x).sum::<f32>().sqrt()
}

/// Compute predictability as inverse of linear fit residual
fn compute_predictability(frames: &[&LatentFrame]) -> f32 {
    if frames.len() < 3 {
        return 0.5;
    }

    // Simple linear extrapolation error
    let mut total_error = 0.0f32;
    let mut count = 0;

    for i in 2..frames.len() {
        let z0 = &frames[i - 2].z;
        let z1 = &frames[i - 1].z;
        let z2 = &frames[i].z;

        // Predict z2 from z0, z1 using linear extrapolation
        // z2_pred = 2*z1 - z0
        let error: f32 = z0.iter()
            .zip(z1.iter())
            .zip(z2.iter())
            .map(|((&a, &b), &c)| {
                let pred = 2.0 * b - a;
                (pred - c).powi(2)
            })
            .sum();

        total_error += error.sqrt();
        count += 1;
    }

    if count == 0 {
        return 0.5;
    }

    let avg_error = total_error / count as f32;

    // Convert to predictability: high error = low predictability
    // Normalize assuming typical error is around 1.0
    1.0 / (1.0 + avg_error)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_latent_frames(n: usize, dim: usize) -> Vec<LatentFrame> {
        (0..n)
            .map(|i| LatentFrame {
                timestamp: i as f64 * 0.02,
                z: vec![i as f32 * 0.1; dim], // Linear increase
                dz_dt: None,
                valid: true,
            })
            .collect()
    }

    #[test]
    fn test_latent_features() {
        let config = AnticipationConfig::default();
        let frames = create_test_latent_frames(50, 25);
        let features = compute_latent_features(&frames, &config);

        assert!(features.norm > 0.0);
        assert!(features.velocity_norm > 0.0);
        // High predictability for linear motion
        assert!(features.predictability > 0.5);
    }
}

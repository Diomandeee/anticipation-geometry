//! Scalar signal computation
//!
//! Computes the 7 scalar signals in the AnticipationPacket:
//! - commitment
//! - uncertainty
//! - transition_pressure
//! - recovery_margin
//! - phase_stiffness
//! - novelty
//! - stability

use crate::config::AnticipationConfig;
use crate::features::{KinematicFeatures, LatentFeatures};

/// Compute commitment score [0-1]
///
/// How irreversible the current motion has become.
/// High when: low uncertainty + high constraint proximity + directional momentum
#[allow(unused_variables)]
pub fn compute_commitment(
    uncertainty: f32,
    constraint_vector: &[f32],
    features: &[f32],
    config: &AnticipationConfig,
) -> f32 {
    // Base commitment from inverse uncertainty
    let base = 1.0 - uncertainty;

    // Constraint contribution (average of constraint proximities)
    let constraint_contrib = if !constraint_vector.is_empty() {
        constraint_vector.iter().sum::<f32>() / constraint_vector.len() as f32
    } else {
        0.0
    };

    // Directional persistence from features (if available)
    let directional = features.get(3).copied().unwrap_or(0.5);

    // Weighted combination
    let commitment = 0.4 * base + 0.3 * constraint_contrib + 0.3 * directional;

    commitment.clamp(0.0, 1.0)
}

/// Compute uncertainty score [0-1]
///
/// How many plausible futures remain.
/// In v0: heuristic based on feature variance
/// In v1: continuation dispersion from HNSW neighbors
#[allow(unused_variables)]
pub fn compute_uncertainty(
    regime_embedding: &[f32],
    config: &AnticipationConfig,
) -> f32 {
    // v0: Heuristic based on embedding entropy proxy
    // Higher variance in embedding = more uncertainty

    if regime_embedding.is_empty() {
        return 0.5;
    }

    // Compute variance of embedding dimensions
    let mean: f32 = regime_embedding.iter().sum::<f32>() / regime_embedding.len() as f32;
    let variance: f32 = regime_embedding.iter()
        .map(|&x| (x - mean).powi(2))
        .sum::<f32>() / regime_embedding.len() as f32;

    // Map variance to uncertainty
    // Higher variance = more diverse embedding = more uncertainty
    (variance * 10.0).clamp(0.0, 1.0)
}

/// Compute transition pressure
///
/// Rate at which futures are collapsing.
/// Formula: d(commitment)/dt - d(uncertainty)/dt, smoothed with EMA
pub fn compute_transition_pressure(
    commitment: f32,
    prev_commitment: f32,
    uncertainty: f32,
    prev_uncertainty: f32,
    dt: f32,
    alpha: f32,
) -> f32 {
    let d_commitment = (commitment - prev_commitment) / dt.max(0.001);
    let d_uncertainty = (uncertainty - prev_uncertainty) / dt.max(0.001);

    // Raw pressure: commitment rising + uncertainty falling
    let raw_pressure = d_commitment - d_uncertainty;

    // EMA smoothing (would need state, but for single call just return raw)
    // In practice, the kernel maintains this state
    raw_pressure * alpha

    // Note: transition_pressure is not clamped to [0,1] - can be negative
}

/// Compute recovery margin [0-1]
///
/// Distance to balance/attractor loss.
/// Derived from constraint vector.
pub fn compute_recovery_margin(constraint_vector: &[f32]) -> f32 {
    // Recovery margin is inverse of constraint pressure
    // When constraints are high, recovery margin is low

    if constraint_vector.is_empty() {
        return 0.5;
    }

    let avg_constraint = constraint_vector.iter().sum::<f32>() / constraint_vector.len() as f32;

    // Invert: high constraints = low margin
    (1.0 - avg_constraint).clamp(0.0, 1.0)
}

/// Compute phase stiffness [0-1]
///
/// How locked to internal metronome.
/// Derived from kinematic features.
pub fn compute_phase_stiffness(features: &KinematicFeatures) -> f32 {
    // High directional persistence + low jerk = phase locked
    let persistence_contrib = features.directional_persistence;
    let jerk_contrib = 1.0 / (1.0 + features.jerk_energy);

    (0.5 * persistence_contrib + 0.5 * jerk_contrib).clamp(0.0, 1.0)
}

/// Compute stability [0-1]
///
/// Local stationarity of dynamics.
/// Can be computed from kinematic or latent features.
pub fn compute_stability<T: StabilitySource>(source: &T) -> f32 {
    source.compute_stability()
}

/// Trait for computing stability from different feature types
pub trait StabilitySource {
    /// Compute stability score [0-1]
    fn compute_stability(&self) -> f32;
}

impl StabilitySource for KinematicFeatures {
    fn compute_stability(&self) -> f32 {
        // Low jerk + high persistence = stable
        let jerk_factor = 1.0 / (1.0 + self.jerk_energy);
        let persistence_factor = self.directional_persistence;

        (0.5 * jerk_factor + 0.5 * persistence_factor).clamp(0.0, 1.0)
    }
}

impl StabilitySource for LatentFeatures {
    fn compute_stability(&self) -> f32 {
        // High predictability + low acceleration = stable
        let pred_factor = self.predictability;
        let accel_factor = 1.0 / (1.0 + self.acceleration_norm);

        (0.5 * pred_factor + 0.5 * accel_factor).clamp(0.0, 1.0)
    }
}

/// Compute novelty is handled by the kernel (requires history)
pub fn compute_novelty(_embedding: &[f32], _history: &[Vec<f32>]) -> f32 {
    // This is actually implemented in kernel.rs with history state
    0.5
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_commitment_bounds() {
        let commitment = compute_commitment(
            0.5,
            &[0.5, 0.5, 0.5, 0.5],
            &[0.5, 0.5, 0.5, 0.5],
            &AnticipationConfig::default(),
        );
        assert!(commitment >= 0.0 && commitment <= 1.0);
    }

    #[test]
    fn test_commitment_high_uncertainty() {
        let commitment = compute_commitment(
            1.0, // Max uncertainty
            &[0.0; 8],
            &[0.0; 8],
            &AnticipationConfig::default(),
        );
        // High uncertainty should lead to lower commitment
        assert!(commitment < 0.5);
    }

    #[test]
    fn test_uncertainty_bounds() {
        let uncertainty = compute_uncertainty(
            &[0.1, 0.2, 0.3, 0.4],
            &AnticipationConfig::default(),
        );
        assert!(uncertainty >= 0.0 && uncertainty <= 1.0);
    }

    #[test]
    fn test_transition_pressure_rising() {
        let pressure = compute_transition_pressure(
            0.8,  // Current commitment (high)
            0.2,  // Previous commitment (low)
            0.3,  // Current uncertainty (low)
            0.7,  // Previous uncertainty (high)
            0.02, // dt
            0.3,  // alpha
        );
        // Rising commitment + falling uncertainty = positive pressure
        assert!(pressure > 0.0);
    }

    #[test]
    fn test_recovery_margin() {
        let high_constraint = compute_recovery_margin(&[0.9, 0.9, 0.9, 0.9]);
        let low_constraint = compute_recovery_margin(&[0.1, 0.1, 0.1, 0.1]);

        assert!(high_constraint < low_constraint);
    }
}

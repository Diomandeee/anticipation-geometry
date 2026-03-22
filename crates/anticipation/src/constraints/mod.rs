//! Constraint proximity computation
//!
//! Computes how close the body is to physical boundaries
//! where options vanish (balance, joint limits, speed saturation).

use crate::config::AnticipationConfig;
use crate::features::KinematicFeatures;

/// Compute constraint vector from kinematic features
///
/// # Output Dimensions
///
/// 0. Balance margin (COM to support distance)
/// 1. Joint limit proximity (average across joints)
/// 2. Speed saturation
/// 3. Angular momentum commitment
/// 4. Reserved
/// 5. Reserved
/// 6. Reserved
/// 7. Reserved
#[allow(unused_variables)]
pub fn compute_constraint_vector(
    features: &KinematicFeatures,
    config: &AnticipationConfig,
) -> Vec<f32> {
    let dim = config.constraint_vector_dim;
    let mut constraints = vec![0.0f32; dim];

    // Balance margin from features
    if dim > 0 {
        constraints[0] = features.balance_margin;
    }

    // Joint limit proximity (placeholder - would require FK)
    if dim > 1 {
        constraints[1] = 0.5; // Neutral
    }

    // Speed saturation: derived from kinetic intensity
    // Higher intensity = closer to speed limits
    if dim > 2 {
        // Assume max safe speed around 5 m/s
        constraints[2] = (features.kinetic_intensity / 5.0).clamp(0.0, 1.0);
    }

    // Angular momentum commitment: derived from angular intensity
    if dim > 3 {
        // Assume max rotation around 10 rad/s
        constraints[3] = (features.angular_intensity / 10.0).clamp(0.0, 1.0);
    }

    // Reserved dimensions stay at 0.0

    constraints
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_constraint_vector_dimension() {
        let config = AnticipationConfig::default();
        let features = KinematicFeatures::default();

        let constraints = compute_constraint_vector(&features, &config);
        assert_eq!(constraints.len(), config.constraint_vector_dim);
    }

    #[test]
    fn test_constraint_bounds() {
        let config = AnticipationConfig::default();
        let features = KinematicFeatures {
            kinetic_intensity: 10.0, // Very high
            angular_intensity: 20.0, // Very high
            ..Default::default()
        };

        let constraints = compute_constraint_vector(&features, &config);

        // Should be clamped to [0, 1]
        for &c in &constraints {
            assert!(c >= 0.0 && c <= 1.0);
        }
    }
}

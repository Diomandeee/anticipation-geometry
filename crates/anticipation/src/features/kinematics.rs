//! Kinematic features from skeleton frames
//!
//! FROZEN: Feature computation logic must be deterministic.

use crate::config::AnticipationConfig;
use crate::types::{SkeletonFrame, Vec3, bones};
use super::forward_kinematics::{compute_forward_kinematics, Keypoints};

/// Kinematic features extracted from skeleton window
#[derive(Clone, Debug, Default)]
pub struct KinematicFeatures {
    /// Total kinetic energy proxy (RMS of root velocity)
    pub kinetic_intensity: f32,
    /// Rotational energy (RMS of torso angular velocity)
    pub angular_intensity: f32,
    /// Rate of acceleration change (jerk)
    pub jerk_energy: f32,
    /// How consistent is velocity direction (autocorrelation)
    pub directional_persistence: f32,
    /// Left-right wrist speed correlation
    pub cross_limb_coherence: f32,
    /// Does torso velocity lead limb velocity?
    pub torso_lead: f32,
    /// Does head rotation predict torso turns?
    pub head_predict: f32,
    /// COM to support polygon distance
    pub balance_margin: f32,
}

impl KinematicFeatures {
    /// Convert to feature vector
    pub fn to_vec(&self) -> Vec<f32> {
        vec![
            self.kinetic_intensity,
            self.angular_intensity,
            self.jerk_energy,
            self.directional_persistence,
            self.cross_limb_coherence,
            self.torso_lead,
            self.head_predict,
            self.balance_margin,
        ]
    }
}

/// Compute kinematic features from skeleton frames
///
/// # Invariants
///
/// - Deterministic given same input
/// - Uses central differences for derivatives
pub fn compute_kinematic_features(
    frames: &[SkeletonFrame],
    config: &AnticipationConfig,
) -> KinematicFeatures {
    if frames.len() < 3 {
        return KinematicFeatures::default();
    }

    let dt = 1.0 / config.fps;

    // Compute root velocities using central differences
    let mut velocities: Vec<Vec3> = Vec::with_capacity(frames.len() - 2);

    for i in 1..frames.len() - 1 {
        if frames[i - 1].valid && frames[i + 1].valid {
            let v = frames[i + 1].root_position.sub(&frames[i - 1].root_position);
            let v = v.scale(1.0 / (2.0 * dt));
            velocities.push(v);
        }
    }

    // Kinetic intensity: RMS of root velocity
    let kinetic_intensity = if !velocities.is_empty() {
        let sum_sq: f32 = velocities.iter()
            .map(|v| v.magnitude_squared())
            .sum();
        (sum_sq / velocities.len() as f32).sqrt()
    } else {
        0.0
    };

    // Angular intensity from torso rotation
    let angular_intensity = compute_angular_intensity(frames, dt);

    // Jerk energy
    let jerk_energy = compute_jerk_energy(&velocities, dt);

    // Directional persistence
    let directional_persistence = compute_directional_persistence(&velocities);

    // Compute keypoints for FK-based features
    let keypoints: Vec<Keypoints> = frames.iter()
        .filter(|f| f.valid)
        .map(compute_forward_kinematics)
        .collect();

    // Cross-limb coherence: correlation of left/right wrist speeds
    let cross_limb_coherence = compute_cross_limb_coherence(&keypoints, dt);

    // Torso lead: placeholder for now
    let torso_lead = 0.5;

    // Head predict: placeholder for now
    let head_predict = 0.5;

    // Balance margin: average balance distance across window
    let balance_margin = compute_balance_margin(&keypoints);

    KinematicFeatures {
        kinetic_intensity,
        angular_intensity,
        jerk_energy,
        directional_persistence,
        cross_limb_coherence,
        torso_lead,
        head_predict,
        balance_margin,
    }
}

/// Compute angular intensity from torso rotation changes
fn compute_angular_intensity(frames: &[SkeletonFrame], dt: f32) -> f32 {
    let mut angular_speeds: Vec<f32> = Vec::new();

    for i in 1..frames.len() {
        if frames[i - 1].valid && frames[i].valid {
            let q1 = frames[i - 1].bone_rotations[bones::CHEST];
            let q2 = frames[i].bone_rotations[bones::CHEST];

            // Compute relative rotation
            let dq = q2.mul(&q1.conjugate());

            // Extract angle from quaternion (angle = 2 * acos(w))
            let angle = 2.0 * dq.w.abs().clamp(-1.0, 1.0).acos();
            let angular_speed = angle / dt;

            angular_speeds.push(angular_speed);
        }
    }

    if angular_speeds.is_empty() {
        return 0.0;
    }

    let sum_sq: f32 = angular_speeds.iter().map(|&a| a * a).sum();
    (sum_sq / angular_speeds.len() as f32).sqrt()
}

/// Compute jerk energy (rate of change of acceleration)
fn compute_jerk_energy(velocities: &[Vec3], dt: f32) -> f32 {
    if velocities.len() < 3 {
        return 0.0;
    }

    let mut jerk_sq_sum = 0.0f32;
    let mut count = 0;

    for i in 2..velocities.len() {
        // Acceleration at i-1 and i
        let a1 = velocities[i - 1].sub(&velocities[i - 2]);
        let a2 = velocities[i].sub(&velocities[i - 1]);

        // Jerk = change in acceleration
        let jerk = a2.sub(&a1).scale(1.0 / (dt * dt));
        jerk_sq_sum += jerk.magnitude_squared();
        count += 1;
    }

    if count == 0 {
        return 0.0;
    }

    (jerk_sq_sum / count as f32).sqrt()
}

/// Compute directional persistence (velocity autocorrelation)
fn compute_directional_persistence(velocities: &[Vec3]) -> f32 {
    if velocities.len() < 2 {
        return 0.5; // Neutral
    }

    let mut dot_sum = 0.0f32;
    let mut count = 0;

    for i in 1..velocities.len() {
        let v1 = &velocities[i - 1];
        let v2 = &velocities[i];

        let mag1 = v1.magnitude();
        let mag2 = v2.magnitude();

        if mag1 > 1e-6 && mag2 > 1e-6 {
            let dot = v1.dot(v2) / (mag1 * mag2);
            dot_sum += dot;
            count += 1;
        }
    }

    if count == 0 {
        return 0.5;
    }

    // Map from [-1, 1] to [0, 1]
    (dot_sum / count as f32 + 1.0) / 2.0
}

/// Compute cross-limb coherence (left-right wrist speed correlation)
///
/// High coherence = symmetric motion (both arms moving together)
/// Low coherence = asymmetric motion (arms moving independently)
fn compute_cross_limb_coherence(keypoints: &[Keypoints], dt: f32) -> f32 {
    if keypoints.len() < 2 {
        return 0.5;
    }

    // Compute wrist speeds
    let mut left_speeds: Vec<f32> = Vec::with_capacity(keypoints.len() - 1);
    let mut right_speeds: Vec<f32> = Vec::with_capacity(keypoints.len() - 1);

    for i in 1..keypoints.len() {
        let l_vel = keypoints[i].left_wrist().sub(&keypoints[i - 1].left_wrist());
        let r_vel = keypoints[i].right_wrist().sub(&keypoints[i - 1].right_wrist());

        left_speeds.push(l_vel.magnitude() / dt);
        right_speeds.push(r_vel.magnitude() / dt);
    }

    if left_speeds.is_empty() {
        return 0.5;
    }

    // Compute Pearson correlation coefficient
    let n = left_speeds.len() as f32;
    let mean_l: f32 = left_speeds.iter().sum::<f32>() / n;
    let mean_r: f32 = right_speeds.iter().sum::<f32>() / n;

    let mut cov = 0.0f32;
    let mut var_l = 0.0f32;
    let mut var_r = 0.0f32;

    for i in 0..left_speeds.len() {
        let dl = left_speeds[i] - mean_l;
        let dr = right_speeds[i] - mean_r;
        cov += dl * dr;
        var_l += dl * dl;
        var_r += dr * dr;
    }

    if var_l < 1e-10 || var_r < 1e-10 {
        return 0.5; // No variance = neutral
    }

    let correlation = cov / (var_l.sqrt() * var_r.sqrt());

    // Map from [-1, 1] to [0, 1]
    (correlation + 1.0) / 2.0
}

/// Compute balance margin from keypoint sequence
///
/// Returns a value in [0, 1] where:
/// - 1.0 = very stable (COM over support)
/// - 0.0 = very unstable (COM far from support)
fn compute_balance_margin(keypoints: &[Keypoints]) -> f32 {
    if keypoints.is_empty() {
        return 0.5;
    }

    // Average balance distance across all frames
    let total_distance: f32 = keypoints.iter()
        .map(|k| k.balance_distance())
        .sum();
    let avg_distance = total_distance / keypoints.len() as f32;

    // Convert distance to margin [0, 1]
    // Assume max "safe" distance is about 0.5m
    // Beyond that, very unstable
    let margin = 1.0 - (avg_distance / 0.5).clamp(0.0, 1.0);

    margin
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::Quat;

    fn create_test_frames(n: usize) -> Vec<SkeletonFrame> {
        (0..n)
            .map(|i| SkeletonFrame {
                timestamp: i as f64 * 0.02,
                root_position: Vec3::new(i as f32 * 0.1, 0.0, 0.0), // Moving in X
                root_rotation: Quat::IDENTITY,
                bone_rotations: [Quat::IDENTITY; 27],
                valid: true,
                source_seq: Some(i as u64),
            })
            .collect()
    }

    #[test]
    fn test_kinematic_features_moving() {
        let config = AnticipationConfig::default();
        let frames = create_test_frames(50);
        let features = compute_kinematic_features(&frames, &config);

        // Should have non-zero kinetic intensity for moving skeleton
        assert!(features.kinetic_intensity > 0.0);
        // High directional persistence for linear motion
        assert!(features.directional_persistence > 0.8);
    }

    #[test]
    fn test_kinematic_features_still() {
        let config = AnticipationConfig::default();
        let frames: Vec<SkeletonFrame> = (0..50)
            .map(|i| SkeletonFrame {
                timestamp: i as f64 * 0.02,
                root_position: Vec3::ZERO, // Not moving
                root_rotation: Quat::IDENTITY,
                bone_rotations: [Quat::IDENTITY; 27],
                valid: true,
                source_seq: Some(i as u64),
            })
            .collect();

        let features = compute_kinematic_features(&frames, &config);
        assert!(features.kinetic_intensity < 0.01);
    }
}

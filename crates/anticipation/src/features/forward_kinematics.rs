//! Forward Kinematics for mocopi 27-bone skeleton
//!
//! Computes world-space keypoint positions from root transform + bone rotations.
//!
//! # Bone Hierarchy
//!
//! The mocopi skeleton uses a standard humanoid hierarchy rooted at the hips.
//! Each bone's world rotation is computed by chaining local rotations from root.

use crate::types::{Quat, Vec3, SkeletonFrame, bones, MOCOPI_BONE_COUNT};

/// Bone rest lengths in meters (approximate adult proportions)
///
/// These are used for forward kinematics to compute keypoint positions.
/// Values are approximate and can be calibrated per-user.
#[rustfmt::skip]
pub const BONE_LENGTHS: [f32; MOCOPI_BONE_COUNT] = [
    0.00,  // 0: HIPS (root, no length)
    0.10,  // 1: SPINE
    0.15,  // 2: CHEST
    0.10,  // 3: UPPER_CHEST
    0.08,  // 4: NECK
    0.15,  // 5: HEAD
    0.12,  // 6: LEFT_SHOULDER
    0.28,  // 7: LEFT_UPPER_ARM
    0.25,  // 8: LEFT_LOWER_ARM
    0.08,  // 9: LEFT_HAND
    0.12,  // 10: RIGHT_SHOULDER
    0.28,  // 11: RIGHT_UPPER_ARM
    0.25,  // 12: RIGHT_LOWER_ARM
    0.08,  // 13: RIGHT_HAND
    0.40,  // 14: LEFT_UPPER_LEG
    0.38,  // 15: LEFT_LOWER_LEG
    0.08,  // 16: LEFT_FOOT
    0.10,  // 17: LEFT_TOES
    0.40,  // 18: RIGHT_UPPER_LEG
    0.38,  // 19: RIGHT_LOWER_LEG
    0.08,  // 20: RIGHT_FOOT
    0.10,  // 21: RIGHT_TOES
    0.03,  // 22: LEFT_THUMB
    0.05,  // 23: LEFT_INDEX
    0.03,  // 24: RIGHT_THUMB
    0.05,  // 25: RIGHT_INDEX
    0.00,  // 26: EYES (no length, orientation only)
];

/// Parent bone index for each bone (-1 for root)
#[rustfmt::skip]
pub const BONE_PARENTS: [i32; MOCOPI_BONE_COUNT] = [
    -1,  // 0: HIPS (root)
    0,   // 1: SPINE -> HIPS
    1,   // 2: CHEST -> SPINE
    2,   // 3: UPPER_CHEST -> CHEST
    3,   // 4: NECK -> UPPER_CHEST
    4,   // 5: HEAD -> NECK
    3,   // 6: LEFT_SHOULDER -> UPPER_CHEST
    6,   // 7: LEFT_UPPER_ARM -> LEFT_SHOULDER
    7,   // 8: LEFT_LOWER_ARM -> LEFT_UPPER_ARM
    8,   // 9: LEFT_HAND -> LEFT_LOWER_ARM
    3,   // 10: RIGHT_SHOULDER -> UPPER_CHEST
    10,  // 11: RIGHT_UPPER_ARM -> RIGHT_SHOULDER
    11,  // 12: RIGHT_LOWER_ARM -> RIGHT_UPPER_ARM
    12,  // 13: RIGHT_HAND -> RIGHT_LOWER_ARM
    0,   // 14: LEFT_UPPER_LEG -> HIPS
    14,  // 15: LEFT_LOWER_LEG -> LEFT_UPPER_LEG
    15,  // 16: LEFT_FOOT -> LEFT_LOWER_LEG
    16,  // 17: LEFT_TOES -> LEFT_FOOT
    0,   // 18: RIGHT_UPPER_LEG -> HIPS
    18,  // 19: RIGHT_LOWER_LEG -> RIGHT_UPPER_LEG
    19,  // 20: RIGHT_FOOT -> RIGHT_LOWER_LEG
    20,  // 21: RIGHT_TOES -> RIGHT_FOOT
    9,   // 22: LEFT_THUMB -> LEFT_HAND
    9,   // 23: LEFT_INDEX -> LEFT_HAND
    13,  // 24: RIGHT_THUMB -> RIGHT_HAND
    13,  // 25: RIGHT_INDEX -> RIGHT_HAND
    5,   // 26: EYES -> HEAD
];

/// Rest pose bone directions (local space, typically along Y or -Y)
///
/// These define the default direction each bone points before rotation.
#[rustfmt::skip]
pub const BONE_DIRECTIONS: [Vec3; MOCOPI_BONE_COUNT] = [
    Vec3 { x: 0.0, y: 1.0, z: 0.0 },   // HIPS
    Vec3 { x: 0.0, y: 1.0, z: 0.0 },   // SPINE
    Vec3 { x: 0.0, y: 1.0, z: 0.0 },   // CHEST
    Vec3 { x: 0.0, y: 1.0, z: 0.0 },   // UPPER_CHEST
    Vec3 { x: 0.0, y: 1.0, z: 0.0 },   // NECK
    Vec3 { x: 0.0, y: 1.0, z: 0.0 },   // HEAD
    Vec3 { x: -1.0, y: 0.0, z: 0.0 },  // LEFT_SHOULDER
    Vec3 { x: -1.0, y: 0.0, z: 0.0 },  // LEFT_UPPER_ARM
    Vec3 { x: -1.0, y: 0.0, z: 0.0 },  // LEFT_LOWER_ARM
    Vec3 { x: -1.0, y: 0.0, z: 0.0 },  // LEFT_HAND
    Vec3 { x: 1.0, y: 0.0, z: 0.0 },   // RIGHT_SHOULDER
    Vec3 { x: 1.0, y: 0.0, z: 0.0 },   // RIGHT_UPPER_ARM
    Vec3 { x: 1.0, y: 0.0, z: 0.0 },   // RIGHT_LOWER_ARM
    Vec3 { x: 1.0, y: 0.0, z: 0.0 },   // RIGHT_HAND
    Vec3 { x: 0.0, y: -1.0, z: 0.0 },  // LEFT_UPPER_LEG
    Vec3 { x: 0.0, y: -1.0, z: 0.0 },  // LEFT_LOWER_LEG
    Vec3 { x: 0.0, y: -1.0, z: 0.0 },  // LEFT_FOOT
    Vec3 { x: 0.0, y: 0.0, z: 1.0 },   // LEFT_TOES
    Vec3 { x: 0.0, y: -1.0, z: 0.0 },  // RIGHT_UPPER_LEG
    Vec3 { x: 0.0, y: -1.0, z: 0.0 },  // RIGHT_LOWER_LEG
    Vec3 { x: 0.0, y: -1.0, z: 0.0 },  // RIGHT_FOOT
    Vec3 { x: 0.0, y: 0.0, z: 1.0 },   // RIGHT_TOES
    Vec3 { x: -1.0, y: 0.0, z: 0.0 },  // LEFT_THUMB
    Vec3 { x: -1.0, y: 0.0, z: 0.0 },  // LEFT_INDEX
    Vec3 { x: 1.0, y: 0.0, z: 0.0 },   // RIGHT_THUMB
    Vec3 { x: 1.0, y: 0.0, z: 0.0 },   // RIGHT_INDEX
    Vec3 { x: 0.0, y: 0.0, z: 1.0 },   // EYES
];

/// Keypoint positions computed from forward kinematics
#[derive(Clone, Debug, Default)]
pub struct Keypoints {
    /// Position of each bone's end point in world space
    pub positions: [Vec3; MOCOPI_BONE_COUNT],
    /// World rotation of each bone
    pub rotations: [Quat; MOCOPI_BONE_COUNT],
}

impl Keypoints {
    /// Get position of a specific bone
    pub fn get(&self, bone: usize) -> Vec3 {
        self.positions[bone]
    }

    /// Get left wrist position
    pub fn left_wrist(&self) -> Vec3 {
        self.positions[bones::LEFT_HAND]
    }

    /// Get right wrist position
    pub fn right_wrist(&self) -> Vec3 {
        self.positions[bones::RIGHT_HAND]
    }

    /// Get left ankle position
    pub fn left_ankle(&self) -> Vec3 {
        self.positions[bones::LEFT_FOOT]
    }

    /// Get right ankle position
    pub fn right_ankle(&self) -> Vec3 {
        self.positions[bones::RIGHT_FOOT]
    }

    /// Get head position
    pub fn head(&self) -> Vec3 {
        self.positions[bones::HEAD]
    }

    /// Get hip/pelvis position (root)
    pub fn hips(&self) -> Vec3 {
        self.positions[bones::HIPS]
    }

    /// Approximate center of mass (weighted average of key joints)
    ///
    /// Uses simplified weights: torso heavy, limbs light.
    pub fn center_of_mass(&self) -> Vec3 {
        // Weights: hips=0.4, chest=0.3, head=0.1, limbs=0.05 each
        let hips = self.positions[bones::HIPS].scale(0.4);
        let chest = self.positions[bones::CHEST].scale(0.3);
        let head = self.positions[bones::HEAD].scale(0.1);
        let l_hand = self.positions[bones::LEFT_HAND].scale(0.05);
        let r_hand = self.positions[bones::RIGHT_HAND].scale(0.05);
        let l_foot = self.positions[bones::LEFT_FOOT].scale(0.05);
        let r_foot = self.positions[bones::RIGHT_FOOT].scale(0.05);

        hips.add(&chest).add(&head).add(&l_hand).add(&r_hand).add(&l_foot).add(&r_foot)
    }

    /// Approximate support polygon center (midpoint of feet)
    pub fn support_center(&self) -> Vec3 {
        let l_foot = self.positions[bones::LEFT_FOOT];
        let r_foot = self.positions[bones::RIGHT_FOOT];
        l_foot.lerp(&r_foot, 0.5)
    }

    /// Distance from COM projection to support center (balance proxy)
    ///
    /// Returns a value where:
    /// - 0.0 = COM directly over support center (stable)
    /// - Higher values = more off-balance
    pub fn balance_distance(&self) -> f32 {
        let com = self.center_of_mass();
        let support = self.support_center();

        // Project COM to ground plane (y=support.y)
        let com_projected = Vec3::new(com.x, support.y, com.z);

        com_projected.sub(&support).magnitude()
    }
}

/// Rotate a vector by a quaternion
///
/// v' = q * v * q^-1
pub fn rotate_vector(q: &Quat, v: &Vec3) -> Vec3 {
    // Quaternion-vector rotation: v' = q * (0, v) * q^-1
    // Optimized formula: v' = v + 2*w*(q_xyz × v) + 2*(q_xyz × (q_xyz × v))

    let qv = Vec3::new(q.x, q.y, q.z);
    let uv = cross(&qv, v);
    let uuv = cross(&qv, &uv);

    Vec3 {
        x: v.x + 2.0 * (q.w * uv.x + uuv.x),
        y: v.y + 2.0 * (q.w * uv.y + uuv.y),
        z: v.z + 2.0 * (q.w * uv.z + uuv.z),
    }
}

/// Cross product of two vectors
fn cross(a: &Vec3, b: &Vec3) -> Vec3 {
    Vec3 {
        x: a.y * b.z - a.z * b.y,
        y: a.z * b.x - a.x * b.z,
        z: a.x * b.y - a.y * b.x,
    }
}

/// Compute forward kinematics for a skeleton frame
///
/// Returns world-space positions and rotations for all bones.
///
/// # Algorithm
///
/// 1. Start at root (hips) with root_position and root_rotation
/// 2. Traverse hierarchy in order (parents before children)
/// 3. For each bone:
///    - World rotation = parent_world_rotation * local_rotation
///    - Position = parent_position + rotate(bone_direction * bone_length, parent_world_rotation)
pub fn compute_forward_kinematics(frame: &SkeletonFrame) -> Keypoints {
    let mut keypoints = Keypoints::default();

    // Root bone (hips)
    keypoints.positions[0] = frame.root_position;
    keypoints.rotations[0] = frame.root_rotation.mul(&frame.bone_rotations[0]);

    // Process bones in hierarchy order (parents before children)
    // The BONE_PARENTS array is structured so we can iterate in order
    for bone_idx in 1..MOCOPI_BONE_COUNT {
        let parent_idx = BONE_PARENTS[bone_idx];

        if parent_idx < 0 {
            // This shouldn't happen for non-root bones, but handle gracefully
            keypoints.positions[bone_idx] = frame.root_position;
            keypoints.rotations[bone_idx] = frame.bone_rotations[bone_idx];
            continue;
        }

        let parent_idx = parent_idx as usize;

        // Compute world rotation: parent_world * local
        let parent_world_rot = keypoints.rotations[parent_idx];
        let local_rot = frame.bone_rotations[bone_idx];
        keypoints.rotations[bone_idx] = parent_world_rot.mul(&local_rot);

        // Compute position: parent_pos + rotated offset
        let parent_pos = keypoints.positions[parent_idx];
        let bone_dir = BONE_DIRECTIONS[bone_idx];
        let bone_len = BONE_LENGTHS[bone_idx];

        // Offset in parent's rotated frame
        let offset = rotate_vector(&parent_world_rot, &bone_dir.scale(bone_len));
        keypoints.positions[bone_idx] = parent_pos.add(&offset);
    }

    keypoints
}

/// Compute keypoints for a sequence of frames
pub fn compute_keypoints_sequence(frames: &[SkeletonFrame]) -> Vec<Keypoints> {
    frames.iter()
        .filter(|f| f.valid)
        .map(compute_forward_kinematics)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_tpose_frame() -> SkeletonFrame {
        SkeletonFrame {
            timestamp: 0.0,
            root_position: Vec3::new(0.0, 1.0, 0.0), // Standing
            root_rotation: Quat::IDENTITY,
            bone_rotations: [Quat::IDENTITY; MOCOPI_BONE_COUNT],
            valid: true,
            source_seq: Some(0),
        }
    }

    #[test]
    fn test_tpose_keypoints() {
        let frame = create_tpose_frame();
        let keypoints = compute_forward_kinematics(&frame);

        // Hips should be at root position
        let hips = keypoints.hips();
        assert!((hips.x - 0.0).abs() < 1e-6);
        assert!((hips.y - 1.0).abs() < 1e-6);
        assert!((hips.z - 0.0).abs() < 1e-6);

        // Head should be above hips (spine + chest + upper_chest + neck + head)
        let head = keypoints.head();
        assert!(head.y > hips.y);

        // Left wrist should be to the left of center
        let l_wrist = keypoints.left_wrist();
        assert!(l_wrist.x < 0.0);

        // Right wrist should be to the right of center
        let r_wrist = keypoints.right_wrist();
        assert!(r_wrist.x > 0.0);
    }

    #[test]
    fn test_balance_distance_tpose() {
        let frame = create_tpose_frame();
        let keypoints = compute_forward_kinematics(&frame);

        // In T-pose, COM should be roughly over support
        let balance = keypoints.balance_distance();
        assert!(balance < 0.5, "T-pose should be balanced, got {}", balance);
    }

    #[test]
    fn test_rotate_vector() {
        // Rotate (1,0,0) by 90 degrees around Y axis
        // Should give (0,0,-1)
        let angle = std::f32::consts::FRAC_PI_2;
        let q = Quat::new(
            (angle / 2.0).cos(),
            0.0,
            (angle / 2.0).sin(),
            0.0,
        );
        let v = Vec3::new(1.0, 0.0, 0.0);
        let rotated = rotate_vector(&q, &v);

        assert!((rotated.x - 0.0).abs() < 1e-5, "x: {}", rotated.x);
        assert!((rotated.y - 0.0).abs() < 1e-5, "y: {}", rotated.y);
        assert!((rotated.z - (-1.0)).abs() < 1e-5, "z: {}", rotated.z);
    }

    #[test]
    fn test_com_calculation() {
        let frame = create_tpose_frame();
        let keypoints = compute_forward_kinematics(&frame);

        let com = keypoints.center_of_mass();

        // COM should be roughly in the middle horizontally
        assert!((com.x).abs() < 0.1);
        assert!((com.z).abs() < 0.1);
        // COM should be above ground
        assert!(com.y > 0.0);
    }
}

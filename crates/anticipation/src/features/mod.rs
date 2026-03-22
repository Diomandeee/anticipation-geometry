//! Feature computation from motion data
//!
//! This module computes kinematic and latent dynamics features
//! from skeleton and latent frames.

mod kinematics;
mod latent_dynamics;
pub mod forward_kinematics;

pub use kinematics::{KinematicFeatures, compute_kinematic_features};
pub use latent_dynamics::{LatentFeatures, compute_latent_features};
pub use forward_kinematics::{
    Keypoints, compute_forward_kinematics, compute_keypoints_sequence,
    rotate_vector, BONE_LENGTHS, BONE_PARENTS, BONE_DIRECTIONS,
};

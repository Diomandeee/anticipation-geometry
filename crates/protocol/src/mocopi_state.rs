//! # Mocopi State Payload
//!
//! JSON schema for Mocopi motion data transmission between relay and backend.
//! This defines the **data contract** for all Mocopi data passing through the system.
//!
//! ## Usage
//!
//! The relay sends `MocopiStateBatch` to the backend via HTTP POST.
//! The backend parses this and feeds into cc-collection for fusion.

use serde::{Deserialize, Serialize};

use crate::mocopi::{BoneTransform, MocopiKitType};

/// Motion control metrics computed from skeleton data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MotionControls {
    /// Hand energy [0-1] - velocity magnitude of hands
    pub hand_energy: f32,
    /// Foot energy [0-1] - velocity magnitude of feet
    pub foot_energy: f32,
    /// Combined raw energy [0-1] - overall motion intensity
    pub raw_energy: f32,
    /// Tension [0-1] - asymmetry between upper/lower body
    pub tension: f32,
    /// Stability [0-1] - inverse of total velocity
    pub stability: f32,
    /// Density [0-1] - ratio of moving limbs
    pub density: f32,
    /// Tempo nudge [-1, 1] - suggested tempo adjustment
    #[serde(default)]
    pub tempo_nudge: f32,
    /// Swing amount [0-1] - rhythmic swing factor
    #[serde(default)]
    pub swing_amount: f32,
    /// Follow vs lead [0-1] - 0=following beat, 1=leading
    #[serde(default)]
    pub follow_vs_lead: f32,
    /// Arm energy (pro kit) [0-1]
    #[serde(default)]
    pub arm_energy: f32,
    /// Thigh energy (pro kit) [0-1]
    #[serde(default)]
    pub thigh_energy: f32,
    /// Elbow energy (pro kit) [0-1]
    #[serde(default)]
    pub elbow_energy: f32,
}

impl Default for MotionControls {
    fn default() -> Self {
        Self {
            hand_energy: 0.0,
            foot_energy: 0.0,
            raw_energy: 0.0,
            tension: 0.0,
            stability: 1.0,
            density: 0.0,
            tempo_nudge: 0.0,
            swing_amount: 0.0,
            follow_vs_lead: 0.5,
            arm_energy: 0.0,
            thigh_energy: 0.0,
            elbow_energy: 0.0,
        }
    }
}

/// Extended bone transform with optional velocity
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BoneTransformWithVelocity {
    /// Bone ID (use MocopiBoneId for canonical reference)
    pub bone_id: u8,
    /// Position in meters (x, y, z)
    pub position: [f32; 3],
    /// Rotation as quaternion (w, x, y, z)
    pub rotation: [f32; 4],
    /// Velocity in m/s (computed from position delta)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub velocity: Option<[f32; 3]>,
}

impl From<BoneTransform> for BoneTransformWithVelocity {
    fn from(bt: BoneTransform) -> Self {
        Self {
            bone_id: bt.bone_id,
            position: bt.position,
            rotation: bt.rotation,
            velocity: None,
        }
    }
}

/// Single Mocopi state frame
///
/// This is the canonical format for a single skeleton frame.
/// Use this for both relay→backend and backend→visualization.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MocopiStateFrame {
    /// Session identifier
    pub session_id: String,
    /// Data source identifier (e.g., "relay-v5", "direct-udp")
    pub source: String,
    /// Timestamp in seconds (UTC)
    pub timestamp: f64,
    /// Frame index (monotonically increasing)
    pub frame_idx: u64,
    /// All bone transforms
    pub bones: Vec<BoneTransformWithVelocity>,
    /// Computed motion controls
    pub controls: MotionControls,
    /// Detected kit type
    pub kit_type: MocopiKitType,
    /// Total bone count in frame
    #[serde(default)]
    pub bone_count: usize,
    /// Combined motion energy (shorthand for controls.raw_energy)
    #[serde(default)]
    pub total_energy: f32,
    /// Stability confidence (shorthand for controls.stability)
    #[serde(default)]
    pub stability_confidence: f32,
}

impl MocopiStateFrame {
    /// Create empty frame
    pub fn new(session_id: impl Into<String>, source: impl Into<String>) -> Self {
        Self {
            session_id: session_id.into(),
            source: source.into(),
            timestamp: 0.0,
            frame_idx: 0,
            bones: Vec::new(),
            controls: MotionControls::default(),
            kit_type: MocopiKitType::Unknown,
            bone_count: 0,
            total_energy: 0.0,
            stability_confidence: 1.0,
        }
    }

    /// Check if frame is valid
    pub fn is_valid(&self) -> bool {
        !self.bones.is_empty()
            && self.bones.iter().all(|b| {
                b.position.iter().all(|v| v.is_finite())
                    && b.rotation.iter().all(|v| v.is_finite())
            })
    }

    /// Get bone by ID
    pub fn get_bone(&self, bone_id: u8) -> Option<&BoneTransformWithVelocity> {
        self.bones.iter().find(|b| b.bone_id == bone_id)
    }
}

/// Batch of Mocopi state frames for efficient transmission
///
/// The relay batches multiple frames and sends them in a single HTTP POST.
/// This reduces network overhead and improves throughput.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MocopiStateBatch {
    /// Frames in this batch
    pub batch: Vec<MocopiStateFrame>,
    /// Number of frames in batch
    pub batch_size: usize,
    /// Source identifier
    pub source: String,
    /// Batch timestamp (when batch was created)
    #[serde(default)]
    pub batch_timestamp: f64,
}

impl MocopiStateBatch {
    /// Create new batch
    pub fn new(source: impl Into<String>) -> Self {
        Self {
            batch: Vec::new(),
            batch_size: 0,
            source: source.into(),
            batch_timestamp: 0.0,
        }
    }

    /// Add frame to batch
    pub fn push(&mut self, frame: MocopiStateFrame) {
        self.batch.push(frame);
        self.batch_size = self.batch.len();
    }

    /// Check if batch is empty
    pub fn is_empty(&self) -> bool {
        self.batch.is_empty()
    }

    /// Get number of frames
    pub fn len(&self) -> usize {
        self.batch.len()
    }
}

/// Legacy format for backward compatibility
///
/// This matches the old Python relay format. Use only for parsing legacy data.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LegacyMocopiFrame {
    pub session_id: String,
    pub source: String,
    pub timestamp: f64,
    pub frame_idx: u64,
    /// Legacy: limb positions as name → [x, y, z]
    #[serde(default)]
    pub limb_positions: std::collections::HashMap<String, Vec<f32>>,
    /// Legacy: limb orientations as name → [qx, qy, qz, qw]
    #[serde(default)]
    pub limb_orientations: std::collections::HashMap<String, Vec<f32>>,
    /// Legacy: total energy
    #[serde(default)]
    pub total_energy: f32,
    /// Legacy: stability confidence
    #[serde(default)]
    pub stability_confidence: f32,
    /// Legacy: bone count
    #[serde(default)]
    pub bone_count: usize,
    /// Legacy: is pro kit
    #[serde(default)]
    pub is_pro_kit: bool,
    /// Legacy: sensor count
    #[serde(default)]
    pub sensor_count: usize,
    /// Legacy: controls object
    #[serde(default)]
    pub controls: Option<MotionControls>,
}

impl LegacyMocopiFrame {
    /// Convert to new format
    pub fn to_new_format(&self) -> MocopiStateFrame {
        use crate::mocopi::MocopiBoneId;

        let mut bones = Vec::new();

        // Convert legacy limb names to bone IDs
        let name_to_bone = |name: &str| -> Option<MocopiBoneId> {
            match name.to_lowercase().as_str() {
                "hip" | "root" => Some(MocopiBoneId::Root),
                "head" => Some(MocopiBoneId::Head),
                "left_hand" | "l_hand" | "left_wrist" | "l_wrist" => Some(MocopiBoneId::LeftWrist),
                "right_hand" | "r_hand" | "right_wrist" | "r_wrist" => {
                    Some(MocopiBoneId::RightWrist)
                }
                "left_foot" | "l_foot" | "left_ankle" | "l_ankle" => Some(MocopiBoneId::LeftAnkle),
                "right_foot" | "r_foot" | "right_ankle" | "r_ankle" => {
                    Some(MocopiBoneId::RightAnkle)
                }
                "left_upper_arm" | "l_shoulder" => Some(MocopiBoneId::LeftShoulder),
                "right_upper_arm" | "r_shoulder" => Some(MocopiBoneId::RightShoulder),
                "left_elbow" | "l_elbow" => Some(MocopiBoneId::LeftElbow),
                "right_elbow" | "r_elbow" => Some(MocopiBoneId::RightElbow),
                "left_thigh" | "l_hip" => Some(MocopiBoneId::LeftHip),
                "right_thigh" | "r_hip" => Some(MocopiBoneId::RightHip),
                _ => None,
            }
        };

        for (name, pos) in &self.limb_positions {
            if let Some(bone_id) = name_to_bone(name) {
                let position = if pos.len() >= 3 {
                    [pos[0], pos[1], pos[2]]
                } else {
                    [0.0, 0.0, 0.0]
                };

                let rotation = self
                    .limb_orientations
                    .get(name)
                    .map(|r| {
                        if r.len() >= 4 {
                            // Legacy is xyzw, convert to wxyz
                            [r[3], r[0], r[1], r[2]]
                        } else {
                            [1.0, 0.0, 0.0, 0.0]
                        }
                    })
                    .unwrap_or([1.0, 0.0, 0.0, 0.0]);

                bones.push(BoneTransformWithVelocity {
                    bone_id: bone_id.id(),
                    position,
                    rotation,
                    velocity: None,
                });
            }
        }

        MocopiStateFrame {
            session_id: self.session_id.clone(),
            source: self.source.clone(),
            timestamp: self.timestamp,
            frame_idx: self.frame_idx,
            bones,
            controls: self.controls.clone().unwrap_or_default(),
            kit_type: if self.is_pro_kit {
                MocopiKitType::Pro
            } else if self.sensor_count >= 6 {
                MocopiKitType::Base
            } else {
                MocopiKitType::Unknown
            },
            bone_count: self.bone_count,
            total_energy: self.total_energy,
            stability_confidence: self.stability_confidence,
        }
    }
}

/// Legacy batch format
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LegacyMocopiBatch {
    pub batch: Vec<LegacyMocopiFrame>,
    pub batch_size: usize,
    pub source: String,
}

impl LegacyMocopiBatch {
    /// Convert to new format
    pub fn to_new_format(&self) -> MocopiStateBatch {
        MocopiStateBatch {
            batch: self.batch.iter().map(|f| f.to_new_format()).collect(),
            batch_size: self.batch_size,
            source: self.source.clone(),
            batch_timestamp: 0.0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_frame_creation() {
        let frame = MocopiStateFrame::new("test-session", "test-source");
        assert_eq!(frame.session_id, "test-session");
        assert_eq!(frame.source, "test-source");
        assert!(!frame.is_valid()); // Empty is not valid
    }

    #[test]
    fn test_batch_operations() {
        let mut batch = MocopiStateBatch::new("test-relay");
        assert!(batch.is_empty());

        let frame = MocopiStateFrame::new("session", "relay");
        batch.push(frame);

        assert_eq!(batch.len(), 1);
        assert_eq!(batch.batch_size, 1);
    }

    #[test]
    fn test_legacy_conversion() {
        let mut legacy = LegacyMocopiFrame {
            session_id: "test".to_string(),
            source: "relay-v4".to_string(),
            timestamp: 1000.0,
            frame_idx: 42,
            limb_positions: std::collections::HashMap::new(),
            limb_orientations: std::collections::HashMap::new(),
            total_energy: 0.5,
            stability_confidence: 0.8,
            bone_count: 6,
            is_pro_kit: false,
            sensor_count: 6,
            controls: None,
        };

        legacy.limb_positions.insert("hip".to_string(), vec![0.0, 1.0, 0.0]);
        legacy.limb_orientations.insert("hip".to_string(), vec![0.0, 0.0, 0.0, 1.0]);

        let new_frame = legacy.to_new_format();
        assert_eq!(new_frame.session_id, "test");
        assert_eq!(new_frame.frame_idx, 42);
        assert_eq!(new_frame.kit_type, MocopiKitType::Base);
        assert!(!new_frame.bones.is_empty());
    }
}


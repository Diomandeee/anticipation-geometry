//! # Sony Mocopi Protocol Definitions
//!
//! Canonical bone ID mappings and data structures for Sony Mocopi motion capture.
//! This is the **single source of truth** for all Mocopi-related definitions
//! across the codebase.
//!
//! ## Bone ID Reference (Sony Mocopi SDK 2.1.0)
//!
//! The Mocopi skeleton hierarchy consists of 27 bones in a tree structure:
//!
//! ```text
//! root (0)
//! └── torso_1 (1)
//!     └── torso_2 (2)
//!         └── torso_3 (3)
//!             ├── neck (4)
//!             │   └── head (5)
//!             ├── l_collar (6)
//!             │   └── l_shoulder (7)
//!             │       └── l_elbow (8)
//!             │           └── l_wrist (9)
//!             │               └── l_hand (10)
//!             │                   └── l_fingers (11)
//!             ├── r_collar (12)
//!             │   └── r_shoulder (13)
//!             │       └── r_elbow (14)
//!             │           └── r_wrist (15)
//!             │               └── r_hand (16)
//!             │                   └── r_fingers (17)
//!             ├── l_hip (18)
//!             │   └── l_knee (19)
//!             │       └── l_ankle (20)
//!             │           └── l_foot (21)
//!             │               └── l_toe (22)
//!             └── r_hip (23)
//!                 └── r_knee (24)
//!                     └── r_ankle (25)
//!                         └── r_foot (26)
//!                             └── r_toe (27)
//! ```
//!
//! ## Base Kit Sensors (6 sensors)
//!
//! The base kit sensors are attached to:
//! - Root (hip) - bone 0
//! - Head - bone 5
//! - Left Wrist - bone 9
//! - Right Wrist - bone 15
//! - Left Ankle - bone 20
//! - Right Ankle - bone 25
//!
//! ## Pro Kit Additional Sensors (+6 sensors)
//!
//! The pro kit adds sensors at:
//! - Left Upper Arm (shoulder) - bone 7
//! - Right Upper Arm (shoulder) - bone 13
//! - Left Elbow - bone 8
//! - Right Elbow - bone 14
//! - Left Thigh (hip) - bone 18
//! - Right Thigh (hip) - bone 23

use serde::{Deserialize, Serialize};

/// Sony Mocopi bone IDs - canonical reference
///
/// These IDs match the Sony Mocopi SDK 2.1.0 bone enumeration.
/// Use this enum as the single source of truth for bone identification.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[repr(u8)]
pub enum MocopiBoneId {
    /// Root/Pelvis - primary anchor point
    Root = 0,
    /// Lower spine
    Torso1 = 1,
    /// Mid spine  
    Torso2 = 2,
    /// Upper spine/chest
    Torso3 = 3,
    /// Neck
    Neck = 4,
    /// Head (base kit sensor)
    Head = 5,
    /// Left collar bone
    LeftCollar = 6,
    /// Left shoulder (pro kit sensor)
    LeftShoulder = 7,
    /// Left elbow (pro kit sensor)
    LeftElbow = 8,
    /// Left wrist (base kit sensor)
    LeftWrist = 9,
    /// Left hand
    LeftHand = 10,
    /// Left fingers
    LeftFingers = 11,
    /// Right collar bone
    RightCollar = 12,
    /// Right shoulder (pro kit sensor)
    RightShoulder = 13,
    /// Right elbow (pro kit sensor)
    RightElbow = 14,
    /// Right wrist (base kit sensor)
    RightWrist = 15,
    /// Right hand
    RightHand = 16,
    /// Right fingers
    RightFingers = 17,
    /// Left hip/thigh (pro kit sensor)
    LeftHip = 18,
    /// Left knee
    LeftKnee = 19,
    /// Left ankle (base kit sensor)
    LeftAnkle = 20,
    /// Left foot
    LeftFoot = 21,
    /// Left toe
    LeftToe = 22,
    /// Right hip/thigh (pro kit sensor)
    RightHip = 23,
    /// Right knee
    RightKnee = 24,
    /// Right ankle (base kit sensor)
    RightAnkle = 25,
    /// Right foot
    RightFoot = 26,
    /// Right toe
    RightToe = 27,
}

impl MocopiBoneId {
    /// Total number of bones in full Mocopi skeleton
    pub const COUNT: usize = 28;

    /// Create from raw bone ID
    pub fn from_id(id: u8) -> Option<Self> {
        match id {
            0 => Some(Self::Root),
            1 => Some(Self::Torso1),
            2 => Some(Self::Torso2),
            3 => Some(Self::Torso3),
            4 => Some(Self::Neck),
            5 => Some(Self::Head),
            6 => Some(Self::LeftCollar),
            7 => Some(Self::LeftShoulder),
            8 => Some(Self::LeftElbow),
            9 => Some(Self::LeftWrist),
            10 => Some(Self::LeftHand),
            11 => Some(Self::LeftFingers),
            12 => Some(Self::RightCollar),
            13 => Some(Self::RightShoulder),
            14 => Some(Self::RightElbow),
            15 => Some(Self::RightWrist),
            16 => Some(Self::RightHand),
            17 => Some(Self::RightFingers),
            18 => Some(Self::LeftHip),
            19 => Some(Self::LeftKnee),
            20 => Some(Self::LeftAnkle),
            21 => Some(Self::LeftFoot),
            22 => Some(Self::LeftToe),
            23 => Some(Self::RightHip),
            24 => Some(Self::RightKnee),
            25 => Some(Self::RightAnkle),
            26 => Some(Self::RightFoot),
            27 => Some(Self::RightToe),
            _ => None,
        }
    }

    /// Get raw bone ID
    #[inline]
    pub fn id(self) -> u8 {
        self as u8
    }

    /// Get bone name
    pub fn name(self) -> &'static str {
        match self {
            Self::Root => "root",
            Self::Torso1 => "torso_1",
            Self::Torso2 => "torso_2",
            Self::Torso3 => "torso_3",
            Self::Neck => "neck",
            Self::Head => "head",
            Self::LeftCollar => "l_collar",
            Self::LeftShoulder => "l_shoulder",
            Self::LeftElbow => "l_elbow",
            Self::LeftWrist => "l_wrist",
            Self::LeftHand => "l_hand",
            Self::LeftFingers => "l_fingers",
            Self::RightCollar => "r_collar",
            Self::RightShoulder => "r_shoulder",
            Self::RightElbow => "r_elbow",
            Self::RightWrist => "r_wrist",
            Self::RightHand => "r_hand",
            Self::RightFingers => "r_fingers",
            Self::LeftHip => "l_hip",
            Self::LeftKnee => "l_knee",
            Self::LeftAnkle => "l_ankle",
            Self::LeftFoot => "l_foot",
            Self::LeftToe => "l_toe",
            Self::RightHip => "r_hip",
            Self::RightKnee => "r_knee",
            Self::RightAnkle => "r_ankle",
            Self::RightFoot => "r_foot",
            Self::RightToe => "r_toe",
        }
    }

    /// Check if this bone has a base kit sensor attached
    pub fn has_base_kit_sensor(self) -> bool {
        matches!(
            self,
            Self::Root | Self::Head | Self::LeftWrist | Self::RightWrist | Self::LeftAnkle | Self::RightAnkle
        )
    }

    /// Check if this bone has a pro kit sensor attached
    pub fn has_pro_kit_sensor(self) -> bool {
        matches!(
            self,
            Self::LeftShoulder | Self::RightShoulder | Self::LeftElbow | Self::RightElbow | Self::LeftHip | Self::RightHip
        )
    }

    /// Check if this bone has any sensor attached
    pub fn has_sensor(self) -> bool {
        self.has_base_kit_sensor() || self.has_pro_kit_sensor()
    }

    /// Get base kit sensor bone IDs (6 sensors)
    pub const fn base_kit_bones() -> [Self; 6] {
        [
            Self::Root,
            Self::Head,
            Self::LeftWrist,
            Self::RightWrist,
            Self::LeftAnkle,
            Self::RightAnkle,
        ]
    }

    /// Get pro kit additional sensor bone IDs (+6 sensors)
    pub const fn pro_kit_additional_bones() -> [Self; 6] {
        [
            Self::LeftShoulder,
            Self::RightShoulder,
            Self::LeftElbow,
            Self::RightElbow,
            Self::LeftHip,
            Self::RightHip,
        ]
    }

    /// Get all pro kit sensor bone IDs (12 sensors total)
    pub const fn pro_kit_all_bones() -> [Self; 12] {
        [
            Self::Root,
            Self::Head,
            Self::LeftWrist,
            Self::RightWrist,
            Self::LeftAnkle,
            Self::RightAnkle,
            Self::LeftShoulder,
            Self::RightShoulder,
            Self::LeftElbow,
            Self::RightElbow,
            Self::LeftHip,
            Self::RightHip,
        ]
    }

    /// Get parent bone in skeleton hierarchy
    pub fn parent(self) -> Option<Self> {
        match self {
            Self::Root => None,
            Self::Torso1 => Some(Self::Root),
            Self::Torso2 => Some(Self::Torso1),
            Self::Torso3 => Some(Self::Torso2),
            Self::Neck => Some(Self::Torso3),
            Self::Head => Some(Self::Neck),
            Self::LeftCollar => Some(Self::Torso3),
            Self::LeftShoulder => Some(Self::LeftCollar),
            Self::LeftElbow => Some(Self::LeftShoulder),
            Self::LeftWrist => Some(Self::LeftElbow),
            Self::LeftHand => Some(Self::LeftWrist),
            Self::LeftFingers => Some(Self::LeftHand),
            Self::RightCollar => Some(Self::Torso3),
            Self::RightShoulder => Some(Self::RightCollar),
            Self::RightElbow => Some(Self::RightShoulder),
            Self::RightWrist => Some(Self::RightElbow),
            Self::RightHand => Some(Self::RightWrist),
            Self::RightFingers => Some(Self::RightHand),
            Self::LeftHip => Some(Self::Root),
            Self::LeftKnee => Some(Self::LeftHip),
            Self::LeftAnkle => Some(Self::LeftKnee),
            Self::LeftFoot => Some(Self::LeftAnkle),
            Self::LeftToe => Some(Self::LeftFoot),
            Self::RightHip => Some(Self::Root),
            Self::RightKnee => Some(Self::RightHip),
            Self::RightAnkle => Some(Self::RightKnee),
            Self::RightFoot => Some(Self::RightAnkle),
            Self::RightToe => Some(Self::RightFoot),
        }
    }

    /// Iterate over all bone IDs
    pub fn all() -> impl Iterator<Item = Self> {
        (0..Self::COUNT as u8).filter_map(Self::from_id)
    }
}

/// Kit type detected from bone presence
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MocopiKitType {
    /// Base kit with 6 sensors
    Base,
    /// Pro kit with 12 sensors
    Pro,
    /// Unknown/partial configuration
    Unknown,
}

impl MocopiKitType {
    /// Detect kit type from set of present bone IDs
    pub fn detect(present_bones: &[u8]) -> Self {
        let has_base = MocopiBoneId::base_kit_bones()
            .iter()
            .all(|b| present_bones.contains(&b.id()));

        if !has_base {
            return Self::Unknown;
        }

        let has_pro = MocopiBoneId::pro_kit_additional_bones()
            .iter()
            .all(|b| present_bones.contains(&b.id()));

        if has_pro {
            Self::Pro
        } else {
            Self::Base
        }
    }

    /// Get expected sensor count
    pub fn sensor_count(self) -> usize {
        match self {
            Self::Base => 6,
            Self::Pro => 12,
            Self::Unknown => 0,
        }
    }
}

/// Single bone transform data
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct BoneTransform {
    /// Bone ID
    pub bone_id: u8,
    /// Position in meters (x, y, z)
    pub position: [f32; 3],
    /// Rotation as quaternion (w, x, y, z) - note: w first!
    pub rotation: [f32; 4],
}

impl BoneTransform {
    /// Create a new bone transform
    pub fn new(bone_id: u8, position: [f32; 3], rotation: [f32; 4]) -> Self {
        Self {
            bone_id,
            position,
            rotation,
        }
    }

    /// Get bone ID as enum
    pub fn bone(&self) -> Option<MocopiBoneId> {
        MocopiBoneId::from_id(self.bone_id)
    }

    /// Check if transform values are valid (not NaN or Inf)
    pub fn is_valid(&self) -> bool {
        self.position.iter().all(|v| v.is_finite())
            && self.rotation.iter().all(|v| v.is_finite())
    }

    /// Create from Mocopi TLV transform (qx, qy, qz, qw, px, py, pz)
    pub fn from_tlv_transform(bone_id: u8, tran: &[f32; 7]) -> Self {
        Self {
            bone_id,
            // TLV order: qx, qy, qz, qw, px, py, pz
            position: [tran[4], tran[5], tran[6]],
            // Convert to wxyz order
            rotation: [tran[3], tran[0], tran[1], tran[2]],
        }
    }
}

/// Complete Mocopi skeleton frame
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MocopiSkeletonFrame {
    /// Timestamp in seconds (from device)
    pub timestamp: f64,
    /// Frame index
    pub frame_idx: u64,
    /// All bone transforms
    pub bones: Vec<BoneTransform>,
    /// Detected kit type
    pub kit_type: MocopiKitType,
}

impl MocopiSkeletonFrame {
    /// Create empty frame
    pub fn new() -> Self {
        Self {
            timestamp: 0.0,
            frame_idx: 0,
            bones: Vec::new(),
            kit_type: MocopiKitType::Unknown,
        }
    }

    /// Get bone transform by ID
    pub fn get_bone(&self, bone_id: MocopiBoneId) -> Option<&BoneTransform> {
        self.bones.iter().find(|b| b.bone_id == bone_id.id())
    }

    /// Check if frame is valid
    pub fn is_valid(&self) -> bool {
        !self.bones.is_empty() && self.bones.iter().all(|b| b.is_valid())
    }

    /// Get sensor count
    pub fn sensor_count(&self) -> usize {
        self.bones
            .iter()
            .filter(|b| {
                MocopiBoneId::from_id(b.bone_id)
                    .map(|bid| bid.has_sensor())
                    .unwrap_or(false)
            })
            .count()
    }

    /// Detect and update kit type from bones
    pub fn detect_kit_type(&mut self) {
        let bone_ids: Vec<u8> = self.bones.iter().map(|b| b.bone_id).collect();
        self.kit_type = MocopiKitType::detect(&bone_ids);
    }
}

impl Default for MocopiSkeletonFrame {
    fn default() -> Self {
        Self::new()
    }
}

/// Mocopi TLV packet tags
pub mod tlv {
    /// Skeleton data container tag
    pub const SKDF: &[u8; 4] = b"skdf";
    /// Frame data tag
    pub const FRAM: &[u8; 4] = b"fram";
    /// Bone transforms tag
    pub const BTRS: &[u8; 4] = b"btrs";
    /// Single bone data tag
    pub const BTDT: &[u8; 4] = b"btdt";
    /// Bone ID tag
    pub const BNID: &[u8; 4] = b"bnid";
    /// Transform data tag
    pub const TRAN: &[u8; 4] = b"tran";
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bone_id_roundtrip() {
        for bone in MocopiBoneId::all() {
            let id = bone.id();
            let recovered = MocopiBoneId::from_id(id).unwrap();
            assert_eq!(bone, recovered);
        }
    }

    #[test]
    fn test_base_kit_bones() {
        let base = MocopiBoneId::base_kit_bones();
        assert_eq!(base.len(), 6);
        for bone in base {
            assert!(bone.has_base_kit_sensor());
        }
    }

    #[test]
    fn test_pro_kit_bones() {
        let pro = MocopiBoneId::pro_kit_all_bones();
        assert_eq!(pro.len(), 12);
        for bone in pro {
            assert!(bone.has_sensor());
        }
    }

    #[test]
    fn test_kit_detection() {
        // Base kit
        let base_ids: Vec<u8> = MocopiBoneId::base_kit_bones()
            .iter()
            .map(|b| b.id())
            .collect();
        assert_eq!(MocopiKitType::detect(&base_ids), MocopiKitType::Base);

        // Pro kit
        let pro_ids: Vec<u8> = MocopiBoneId::pro_kit_all_bones()
            .iter()
            .map(|b| b.id())
            .collect();
        assert_eq!(MocopiKitType::detect(&pro_ids), MocopiKitType::Pro);

        // Unknown (missing bones)
        let partial = vec![0, 5];
        assert_eq!(MocopiKitType::detect(&partial), MocopiKitType::Unknown);
    }

    #[test]
    fn test_bone_hierarchy() {
        // Head's parent is Neck
        assert_eq!(MocopiBoneId::Head.parent(), Some(MocopiBoneId::Neck));
        // Root has no parent
        assert_eq!(MocopiBoneId::Root.parent(), None);
        // Left wrist's parent is left elbow
        assert_eq!(
            MocopiBoneId::LeftWrist.parent(),
            Some(MocopiBoneId::LeftElbow)
        );
    }

    #[test]
    fn test_transform_validation() {
        let valid = BoneTransform::new(0, [0.0, 1.0, 0.0], [1.0, 0.0, 0.0, 0.0]);
        assert!(valid.is_valid());

        let invalid = BoneTransform::new(0, [f32::NAN, 1.0, 0.0], [1.0, 0.0, 0.0, 0.0]);
        assert!(!invalid.is_valid());
    }
}


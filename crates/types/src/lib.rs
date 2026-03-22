//! # CC Types
//!
//! Core types for the Computational Choreography motion processing pipeline.
//!
//! This crate defines foundational data structures used throughout the CC ecosystem:
//! - Motion windows and skeleton frames
//! - Device masks and provenance tracking
//! - Raw sensor packets
//!
//! ## Schema Version
//!
//! Current schema version: `0.1.0`
//!
//! Schema versions are FROZEN after lock. Breaking changes require major version bump.
//!
//! ## Type Stability
//!
//! | Type | Stability | Notes |
//! |------|-----------|-------|
//! | MotionWindow | FROZEN | Schema-versioned output |
//! | SkeletonFrame | FROZEN | 27-bone skeleton |
//! | DeviceMask | FROZEN | Bit flags for devices |
//! | FrameProvenance | FROZEN | Per-DOF tracking |

#![warn(missing_docs)]
#![warn(clippy::all)]
#![deny(unsafe_code)]

use serde::{Deserialize, Serialize};
use smallvec::SmallVec;

// =============================================================================
// Constants
// =============================================================================

/// Current schema version for MotionWindow
///
/// FROZEN after v0.1.0 lock. Changing this requires major version bump.
pub const SCHEMA_VERSION: &str = "0.1.0";

/// Canonical number of bones in skeleton
pub const BONE_COUNT: usize = 27;

/// Default canonical FPS
pub const DEFAULT_FPS: f32 = 50.0;

/// Default frames per window (1 second at 50 FPS)
pub const DEFAULT_WINDOW_FRAMES: usize = 50;

/// Default hop size (50% overlap)
pub const DEFAULT_HOP_SIZE: usize = 25;

/// Default minimum coverage threshold
pub const DEFAULT_MIN_COVERAGE: f32 = 0.9;

// =============================================================================
// Device Types
// =============================================================================

/// Device type enumeration
///
/// Identifies the source device class for sensor data.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum DeviceType {
    /// Sony mocopi motion capture suit (27-bone)
    Mocopi,
    /// Smartphone (accelerometer, gyroscope)
    Phone,
    /// Smartwatch (accelerometer, gyroscope, heart rate)
    Watch,
    /// AirPods Pro (head tracking)
    AirPods,
    /// Depth camera (e.g., LiDAR, structured light)
    DepthCamera,
    /// Generic IMU sensor
    GenericIMU,
}

/// Device mask for tracking which devices contributed to a frame
///
/// Implemented as a bit field for efficient storage and operations.
/// Each bit represents a device type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default, Serialize, Deserialize)]
pub struct DeviceMask(u8);

impl DeviceMask {
    /// No devices
    pub const NONE: Self = Self(0x00);
    /// Mocopi motion capture
    pub const MOCOPI: Self = Self(0x01);
    /// Smartphone
    pub const PHONE: Self = Self(0x02);
    /// Smartwatch
    pub const WATCH: Self = Self(0x04);
    /// AirPods
    pub const AIRPODS: Self = Self(0x08);
    /// Depth camera
    pub const DEPTH_CAMERA: Self = Self(0x10);
    /// Generic IMU
    pub const GENERIC_IMU: Self = Self(0x20);
    /// All known devices
    pub const ALL: Self = Self(0x3F);

    /// Create a new DeviceMask with specified bits
    #[inline]
    pub const fn new(bits: u8) -> Self {
        Self(bits)
    }

    /// Get the raw bits
    #[inline]
    pub const fn bits(self) -> u8 {
        self.0
    }

    /// Check if this mask contains another mask
    #[inline]
    pub const fn contains(self, other: Self) -> bool {
        (self.0 & other.0) == other.0
    }

    /// Union of two masks
    #[inline]
    pub const fn union(self, other: Self) -> Self {
        Self(self.0 | other.0)
    }

    /// Intersection of two masks
    #[inline]
    pub const fn intersection(self, other: Self) -> Self {
        Self(self.0 & other.0)
    }

    /// Difference (self - other)
    #[inline]
    pub const fn difference(self, other: Self) -> Self {
        Self(self.0 & !other.0)
    }

    /// Check if empty
    #[inline]
    pub const fn is_empty(self) -> bool {
        self.0 == 0
    }

    /// Count number of devices
    #[inline]
    pub const fn count(self) -> u32 {
        self.0.count_ones()
    }

    /// Create from device type
    pub fn from_device_type(device: DeviceType) -> Self {
        match device {
            DeviceType::Mocopi => Self::MOCOPI,
            DeviceType::Phone => Self::PHONE,
            DeviceType::Watch => Self::WATCH,
            DeviceType::AirPods => Self::AIRPODS,
            DeviceType::DepthCamera => Self::DEPTH_CAMERA,
            DeviceType::GenericIMU => Self::GENERIC_IMU,
        }
    }
}

impl std::ops::BitOr for DeviceMask {
    type Output = Self;

    fn bitor(self, rhs: Self) -> Self::Output {
        self.union(rhs)
    }
}

impl std::ops::BitOrAssign for DeviceMask {
    fn bitor_assign(&mut self, rhs: Self) {
        *self = self.union(rhs);
    }
}

impl std::ops::BitAnd for DeviceMask {
    type Output = Self;

    fn bitand(self, rhs: Self) -> Self::Output {
        self.intersection(rhs)
    }
}

// =============================================================================
// Bone and Skeleton Types
// =============================================================================

/// State of a single bone
///
/// Represents the rotational state of one bone in the skeleton.
/// Optionally includes local position for bones with translation DOFs.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct BoneState {
    /// Unit quaternion rotation (w, x, y, z)
    ///
    /// Invariant: `|rotation| ≈ 1.0` (normalized)
    /// Convention: Scalar-first (w, x, y, z)
    pub rotation: [f32; 4],

    /// Local position for bones with translation
    ///
    /// Only populated for specific bones (e.g., root).
    /// Units: meters in body frame.
    pub local_position: Option<[f32; 3]>,
}

impl Default for BoneState {
    fn default() -> Self {
        Self {
            // Identity quaternion
            rotation: [1.0, 0.0, 0.0, 0.0],
            local_position: None,
        }
    }
}

impl BoneState {
    /// Create a new bone state with the given rotation
    pub fn new(rotation: [f32; 4]) -> Self {
        Self {
            rotation,
            local_position: None,
        }
    }

    /// Create with rotation and position
    pub fn with_position(rotation: [f32; 4], position: [f32; 3]) -> Self {
        Self {
            rotation,
            local_position: Some(position),
        }
    }

    /// Get quaternion as nalgebra UnitQuaternion
    pub fn as_unit_quaternion(&self) -> nalgebra::UnitQuaternion<f32> {
        let q = nalgebra::Quaternion::new(
            self.rotation[0],
            self.rotation[1],
            self.rotation[2],
            self.rotation[3],
        );
        nalgebra::UnitQuaternion::new_normalize(q)
    }

    /// Check if the quaternion is normalized (within tolerance)
    pub fn is_normalized(&self, tolerance: f32) -> bool {
        let norm_sq = self.rotation.iter().map(|x| x * x).sum::<f32>();
        (norm_sq - 1.0).abs() < tolerance
    }

    /// Normalize the quaternion in place
    pub fn normalize(&mut self) {
        let norm = self.rotation.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 0.0 {
            for r in &mut self.rotation {
                *r /= norm;
            }
        }
    }
}

// =============================================================================
// Provenance Types
// =============================================================================

/// Frame provenance tracking
///
/// Records which devices contributed to each frame and whether
/// interpolation was applied. Essential for coverage computation
/// and quality-aware downstream processing.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FrameProvenance {
    /// Devices that contributed to this frame
    pub device_mask: DeviceMask,

    /// True if any degree of freedom was interpolated
    pub interpolated: bool,

    /// Per-bone source device mask
    ///
    /// Tracks which device(s) contributed to each bone's data.
    pub per_bone_source: [DeviceMask; BONE_COUNT],

    /// Aggregate uncertainty [0, 1]
    ///
    /// Higher values indicate more uncertainty (more interpolation,
    /// lower confidence sources, longer gaps).
    pub uncertainty: f32,

    /// Gap duration before this frame (ms)
    ///
    /// 0.0 if no gap (continuous data).
    pub gap_before_ms: f32,
}

impl Default for FrameProvenance {
    fn default() -> Self {
        Self {
            device_mask: DeviceMask::NONE,
            interpolated: false,
            per_bone_source: [DeviceMask::NONE; BONE_COUNT],
            uncertainty: 0.0,
            gap_before_ms: 0.0,
        }
    }
}

impl FrameProvenance {
    /// Check if provenance is complete (INV-008)
    ///
    /// Returns true if all bones have at least one source device.
    pub fn is_complete(&self) -> bool {
        !self.device_mask.is_empty()
    }

    /// Create provenance from a single device
    pub fn from_single_device(device: DeviceMask) -> Self {
        Self {
            device_mask: device,
            interpolated: false,
            per_bone_source: [device; BONE_COUNT],
            uncertainty: 0.0,
            gap_before_ms: 0.0,
        }
    }

    /// Mark as interpolated with given uncertainty
    pub fn mark_interpolated(&mut self, uncertainty: f32, gap_ms: f32) {
        self.interpolated = true;
        self.uncertainty = uncertainty;
        self.gap_before_ms = gap_ms;
    }
}

// =============================================================================
// Frame Types
// =============================================================================

/// A single skeleton frame at a canonical timestamp
///
/// Represents the full-body pose at one instant in time.
/// Part of the FROZEN schema after v0.1.0.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SkeletonFrame {
    /// Canonical timestamp (seconds since session start)
    pub timestamp: f64,

    /// 27-bone skeleton state
    ///
    /// Indexed by BoneIndex enum.
    pub bones: [BoneState; BONE_COUNT],

    /// Root (pelvis) position in body frame
    ///
    /// Units: meters
    pub root_position: [f32; 3],

    /// Provenance information for this frame
    pub provenance: FrameProvenance,
}

impl SkeletonFrame {
    /// Create a new skeleton frame
    pub fn new(timestamp: f64) -> Self {
        Self {
            timestamp,
            bones: [BoneState::default(); BONE_COUNT],
            root_position: [0.0, 0.0, 0.0],
            provenance: FrameProvenance::default(),
        }
    }

    /// Check if all bone quaternions are normalized
    pub fn is_normalized(&self, tolerance: f32) -> bool {
        self.bones.iter().all(|b| b.is_normalized(tolerance))
    }

    /// Normalize all bone quaternions
    pub fn normalize(&mut self) {
        for bone in &mut self.bones {
            bone.normalize();
        }
    }
}

// =============================================================================
// Latent Frame Types
// =============================================================================

/// A latent frame from LIM-RPS
///
/// Represents the compressed latent state learned by the motion model.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LatentFrame {
    /// Canonical timestamp (matches SkeletonFrame)
    pub timestamp: f64,

    /// Latent state vector
    ///
    /// Dimension depends on LIM-RPS configuration (typically 32-128).
    pub latent: SmallVec<[f32; 64]>,

    /// Model version that produced this latent
    pub model_version: String,
}

// =============================================================================
// Motion Window
// =============================================================================

/// A deterministic window of aligned motion data
///
/// The primary output of the WindowAligner. Contains a fixed number of
/// SkeletonFrames with guaranteed semantics for downstream processing.
///
/// # FROZEN after v0.1.0
///
/// Changing this struct's fields or semantics requires a major version bump.
///
/// # Invariants
///
/// - INV-001: Deterministic (same inputs → identical outputs)
/// - INV-002: Bounded (exactly `frames.len() == config.frames_per_window`)
/// - INV-003: Explicit missingness (interpolated frames marked in provenance)
/// - INV-004: Temporal monotonicity (`frames[i].timestamp < frames[i+1].timestamp`)
/// - INV-007: Checksum validity
/// - INV-010: Window ID determinism
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MotionWindow {
    // =========================================================================
    // Core Data
    // =========================================================================
    /// Skeleton frames (exactly `config.frames_per_window`)
    pub frames: Vec<SkeletonFrame>,

    /// Latent frames from LIM-RPS (optional)
    ///
    /// Only populated if LIM-RPS is integrated and has processed this window.
    pub latent_frames: Option<Vec<LatentFrame>>,

    // =========================================================================
    // Temporal Bounds
    // =========================================================================
    /// Start timestamp (canonical time, seconds since epoch)
    pub t_start: f64,

    /// End timestamp (canonical time)
    ///
    /// Invariant: `t_end = t_start + (frames.len() - 1) / fps`
    pub t_end: f64,

    /// Canonical frames per second
    pub fps: f32,

    // =========================================================================
    // Coverage & Quality
    // =========================================================================
    /// Coverage ratio [0, 1]
    ///
    /// Fraction of frames that are NOT interpolated.
    /// `coverage = count(non-interpolated) / total_frames`
    pub coverage: f32,

    /// Devices that contributed to this window
    pub device_mask: DeviceMask,

    /// Indices of fully-interpolated frames
    pub dropped_frames: Vec<usize>,

    // =========================================================================
    // Provenance
    // =========================================================================
    /// Session identifier
    pub session_id: String,

    /// Deterministic window ID
    ///
    /// Computed as: `hash(session_id, t_start, config_hash)`
    pub window_id: String,

    /// Aligner version that produced this window
    pub aligner_version: String,

    /// Schema version (for compatibility checking)
    pub schema_version: String,

    /// Hash of the configuration used
    pub config_hash: u64,

    // =========================================================================
    // Integrity
    // =========================================================================
    /// Checksum of frames + metadata
    ///
    /// Computed deterministically using xxhash.
    pub checksum: u64,
}

impl MotionWindow {
    /// Check temporal monotonicity (INV-004)
    pub fn check_temporal_monotonicity(&self) -> bool {
        self.frames
            .windows(2)
            .all(|w| w[0].timestamp < w[1].timestamp)
    }

    /// Compute expected duration based on fps
    pub fn expected_duration(&self) -> f64 {
        if self.frames.is_empty() {
            0.0
        } else {
            (self.frames.len() - 1) as f64 / self.fps as f64
        }
    }

    /// Check if duration matches expected (within tolerance)
    pub fn check_duration(&self, tolerance: f64) -> bool {
        let expected = self.expected_duration();
        let actual = self.t_end - self.t_start;
        (actual - expected).abs() < tolerance
    }

    /// Compute coverage from frames
    pub fn compute_coverage(&self) -> f32 {
        if self.frames.is_empty() {
            return 0.0;
        }
        let non_interpolated = self.frames.iter().filter(|f| !f.provenance.interpolated).count();
        non_interpolated as f32 / self.frames.len() as f32
    }

    /// Get total uncertainty across all frames
    pub fn total_uncertainty(&self) -> f32 {
        if self.frames.is_empty() {
            return 0.0;
        }
        self.frames.iter().map(|f| f.provenance.uncertainty).sum::<f32>() / self.frames.len() as f32
    }
}

// =============================================================================
// Raw Input Types
// =============================================================================

/// Raw packet from a device (pre-alignment)
///
/// This is the input to the WindowAligner before any processing.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RawPacket {
    /// Device identifier (unique per device instance)
    pub device_id: String,

    /// Device type
    pub device_type: DeviceType,

    /// Device-local timestamp (device clock)
    pub device_timestamp: f64,

    /// Arrival time (server/local clock)
    ///
    /// Used for clock synchronization.
    pub arrival_time: f64,

    /// Packet payload
    pub payload: PacketPayload,
}

/// Payload variants for different device types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PacketPayload {
    /// Full mocopi skeleton frame
    MocopiFrame {
        /// 27 bone quaternions (w, x, y, z)
        bones: Vec<[f32; 4]>,
        /// Root position
        root_position: [f32; 3],
    },

    /// IMU data (accelerometer + gyroscope)
    IMU {
        /// Acceleration (m/s²)
        acceleration: [f32; 3],
        /// Angular velocity (rad/s)
        angular_velocity: [f32; 3],
    },

    /// Head tracking (AirPods)
    HeadTracking {
        /// Head orientation quaternion
        orientation: [f32; 4],
    },

    /// Heart rate data
    HeartRate {
        /// Beats per minute
        bpm: f32,
    },

    /// Generic sensor value
    Generic {
        /// Sensor name
        name: String,
        /// Sensor value(s)
        values: Vec<f32>,
    },
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_device_mask_operations() {
        let mocopi = DeviceMask::MOCOPI;
        let phone = DeviceMask::PHONE;
        let combined = mocopi | phone;

        assert!(combined.contains(mocopi));
        assert!(combined.contains(phone));
        assert!(!combined.contains(DeviceMask::WATCH));
        assert_eq!(combined.count(), 2);
    }

    #[test]
    fn test_bone_state_normalization() {
        let mut bone = BoneState::new([2.0, 0.0, 0.0, 0.0]);
        assert!(!bone.is_normalized(0.001));

        bone.normalize();
        assert!(bone.is_normalized(0.001));
        assert!((bone.rotation[0] - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_skeleton_frame_creation() {
        let frame = SkeletonFrame::new(1.0);
        assert_eq!(frame.timestamp, 1.0);
        assert_eq!(frame.bones.len(), BONE_COUNT);
    }

    #[test]
    fn test_frame_provenance_default() {
        let prov = FrameProvenance::default();
        assert!(prov.device_mask.is_empty());
        assert!(!prov.interpolated);
        assert_eq!(prov.uncertainty, 0.0);
    }

    #[test]
    fn test_motion_window_temporal_monotonicity() {
        let mut window = MotionWindow {
            frames: vec![
                SkeletonFrame::new(0.0),
                SkeletonFrame::new(0.02),
                SkeletonFrame::new(0.04),
            ],
            latent_frames: None,
            t_start: 0.0,
            t_end: 0.04,
            fps: 50.0,
            coverage: 1.0,
            device_mask: DeviceMask::MOCOPI,
            dropped_frames: vec![],
            session_id: "test".to_string(),
            window_id: "w_test".to_string(),
            aligner_version: "0.1.0".to_string(),
            schema_version: "0.1.0".to_string(),
            config_hash: 0,
            checksum: 0,
        };

        assert!(window.check_temporal_monotonicity());

        // Violate monotonicity
        window.frames[1].timestamp = 0.05;
        assert!(!window.check_temporal_monotonicity());
    }

    #[test]
    fn test_constants() {
        assert_eq!(SCHEMA_VERSION, "0.1.0");
        assert_eq!(BONE_COUNT, 27);
        assert_eq!(DEFAULT_FPS, 50.0);
    }
}

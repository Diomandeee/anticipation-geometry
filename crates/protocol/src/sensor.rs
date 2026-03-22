//! Sensor input types
//!
//! Raw sensor data from motion capture devices (iPhones, Apple Watch, AirPods).

use serde::{Deserialize, Serialize};

/// Raw sensor frame from a single device at a single timestamp.
///
/// This represents the fundamental input to the system - what the body is doing
/// at a specific moment in time, as measured by inertial sensors.
///
/// ## Coordinate System
///
/// All vectors use the device's local coordinate system:
/// - X: Right (when holding phone upright)
/// - Y: Up (toward top of device)
/// - Z: Forward (out of screen)
///
/// ## Units
///
/// - Acceleration: m/s²
/// - Gyroscope: rad/s
/// - Magnetometer: µT (microtesla)
/// - Heart rate: BPM (beats per minute)
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct SensorFrame {
    /// Timestamp in microseconds since Unix epoch
    ///
    /// This is the authoritative timestamp for all sensor readings in this frame.
    /// Microsecond precision allows for accurate synchronization across devices.
    pub timestamp_us: i64,

    /// Device identifier (e.g., "left_hand", "right_hand", "body", "head")
    ///
    /// This identifies which physical device generated this frame, allowing
    /// multi-device fusion in LIM-RPS.
    pub device_id: String,

    /// Accelerometer reading [x, y, z] in m/s²
    ///
    /// Includes both motion acceleration AND gravity. To get linear acceleration
    /// (motion only), subtract the gravity vector.
    pub accel: [f32; 3],

    /// Gyroscope reading [x, y, z] in rad/s
    ///
    /// Angular velocity around each axis. Positive values follow right-hand rule.
    pub gyro: [f32; 3],

    /// Gravity vector [x, y, z] in m/s²
    ///
    /// Device's estimate of gravity direction. Should have magnitude close to 9.81.
    /// Use this to separate gravity from motion acceleration.
    pub gravity: [f32; 3],

    /// Device orientation quaternion [w, x, y, z]
    ///
    /// Represents the device's orientation in 3D space.
    /// Identity quaternion [1, 0, 0, 0] means device is upright, screen facing user.
    pub quaternion: [f32; 4],

    /// Optional magnetometer reading [x, y, z] in µT
    ///
    /// Magnetic field vector. Can be used for absolute heading, but prone to interference.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub magnetometer: Option<[f32; 3]>,

    /// Optional heart rate in BPM
    ///
    /// Only available from Apple Watch. Useful for detecting cardiovascular exertion
    /// and correlating with movement intensity.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub heart_rate: Option<f32>,

    /// Optional location data [latitude, longitude] in degrees
    ///
    /// GPS coordinates if available. Can be used for venue detection or outdoor performances.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub location: Option<[f64; 2]>,
}

impl SensorFrame {
    /// Create a new sensor frame with required fields
    pub fn new(timestamp_us: i64, device_id: impl Into<String>) -> Self {
        Self {
            timestamp_us,
            device_id: device_id.into(),
            accel: [0.0; 3],
            gyro: [0.0; 3],
            gravity: [0.0, -9.81, 0.0], // Default: gravity pointing down
            quaternion: [1.0, 0.0, 0.0, 0.0], // Identity quaternion
            magnetometer: None,
            heart_rate: None,
            location: None,
        }
    }

    /// Get linear acceleration (accel minus gravity)
    ///
    /// This isolates the motion component from total acceleration.
    pub fn linear_accel(&self) -> [f32; 3] {
        [
            self.accel[0] - self.gravity[0],
            self.accel[1] - self.gravity[1],
            self.accel[2] - self.gravity[2],
        ]
    }

    /// Get total acceleration magnitude in m/s²
    pub fn accel_magnitude(&self) -> f32 {
        (self.accel[0].powi(2) + self.accel[1].powi(2) + self.accel[2].powi(2)).sqrt()
    }

    /// Get linear acceleration magnitude in m/s²
    pub fn linear_accel_magnitude(&self) -> f32 {
        let la = self.linear_accel();
        (la[0].powi(2) + la[1].powi(2) + la[2].powi(2)).sqrt()
    }

    /// Get gyroscope magnitude (total angular velocity) in rad/s
    pub fn gyro_magnitude(&self) -> f32 {
        (self.gyro[0].powi(2) + self.gyro[1].powi(2) + self.gyro[2].powi(2)).sqrt()
    }

    /// Get device roll angle from quaternion (rotation around Z axis) in radians
    pub fn roll(&self) -> f32 {
        let [w, x, y, z] = self.quaternion;
        (2.0 * (w * z + x * y)).atan2(1.0 - 2.0 * (y.powi(2) + z.powi(2)))
    }

    /// Get device pitch angle from quaternion (rotation around X axis) in radians
    pub fn pitch(&self) -> f32 {
        let [w, x, y, z] = self.quaternion;
        (2.0 * (w * x + y * z)).asin()
    }

    /// Get device yaw angle from quaternion (rotation around Y axis) in radians
    pub fn yaw(&self) -> f32 {
        let [w, x, y, z] = self.quaternion;
        (2.0 * (w * y - z * x)).atan2(1.0 - 2.0 * (x.powi(2) + y.powi(2)))
    }

    /// Check if this frame has valid data
    ///
    /// Returns false if accelerometer or gyro readings are all zeros (sensor not working)
    pub fn is_valid(&self) -> bool {
        let accel_sum = self.accel.iter().map(|&x| x.abs()).sum::<f32>();
        let gyro_sum = self.gyro.iter().map(|&x| x.abs()).sum::<f32>();
        accel_sum > 0.01 || gyro_sum > 0.001
    }
}

impl Default for SensorFrame {
    fn default() -> Self {
        Self::new(0, "unknown")
    }
}

/// Aggregated sensor data from multiple devices at a single time slice
///
/// This represents a synchronized snapshot of all devices. The timestamp is the
/// "canonical" time - individual device frames may have slightly different timestamps
/// due to clock drift, which should be corrected before fusion.
#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq)]
pub struct MultiDeviceFrame {
    /// Canonical timestamp in microseconds (typically from "master" device or server)
    pub timestamp_us: i64,

    /// Left hand/arm device (phone in left pocket or hand)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub left: Option<SensorFrame>,

    /// Right hand/arm device (phone in right pocket or hand)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub right: Option<SensorFrame>,

    /// Body/torso device (phone in chest pocket or waist)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub body: Option<SensorFrame>,

    /// Head device (AirPods Pro with head tracking)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub head: Option<SensorFrame>,

    /// Wrist device (Apple Watch)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub watch: Option<SensorFrame>,
}

impl MultiDeviceFrame {
    /// Create a new multi-device frame with canonical timestamp
    pub fn new(timestamp_us: i64) -> Self {
        Self {
            timestamp_us,
            ..Default::default()
        }
    }

    /// Count how many devices have data in this frame
    pub fn device_count(&self) -> usize {
        let mut count = 0;
        if self.left.is_some() {
            count += 1;
        }
        if self.right.is_some() {
            count += 1;
        }
        if self.body.is_some() {
            count += 1;
        }
        if self.head.is_some() {
            count += 1;
        }
        if self.watch.is_some() {
            count += 1;
        }
        count
    }

    /// Check if frame has at least one valid device
    pub fn is_valid(&self) -> bool {
        self.device_count() > 0
    }

    /// Get all frames as a vector
    pub fn all_frames(&self) -> Vec<&SensorFrame> {
        let mut frames = Vec::new();
        if let Some(ref frame) = self.left {
            frames.push(frame);
        }
        if let Some(ref frame) = self.right {
            frames.push(frame);
        }
        if let Some(ref frame) = self.body {
            frames.push(frame);
        }
        if let Some(ref frame) = self.head {
            frames.push(frame);
        }
        if let Some(ref frame) = self.watch {
            frames.push(frame);
        }
        frames
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sensor_frame_creation() {
        let frame = SensorFrame::new(1234567890, "test_device");
        assert_eq!(frame.timestamp_us, 1234567890);
        assert_eq!(frame.device_id, "test_device");
        assert_eq!(frame.gravity, [0.0, -9.81, 0.0]);
        assert_eq!(frame.quaternion, [1.0, 0.0, 0.0, 0.0]);
    }

    #[test]
    fn test_linear_accel() {
        let mut frame = SensorFrame::new(0, "test");
        frame.accel = [1.0, -9.81, 0.5];
        frame.gravity = [0.0, -9.81, 0.0];

        let linear = frame.linear_accel();
        assert!((linear[0] - 1.0).abs() < 0.001);
        assert!(linear[1].abs() < 0.001);
        assert!((linear[2] - 0.5).abs() < 0.001);
    }

    #[test]
    fn test_magnitudes() {
        let mut frame = SensorFrame::new(0, "test");
        frame.accel = [3.0, 4.0, 0.0];
        frame.gyro = [0.1, 0.0, 0.0];

        assert!((frame.accel_magnitude() - 5.0).abs() < 0.001);
        assert!((frame.gyro_magnitude() - 0.1).abs() < 0.001);
    }

    #[test]
    fn test_validity() {
        let mut frame = SensorFrame::new(0, "test");
        assert!(!frame.is_valid()); // All zeros

        frame.accel = [0.1, 0.0, 0.0];
        assert!(frame.is_valid());
    }

    #[test]
    fn test_multi_device_frame() {
        let mut multi = MultiDeviceFrame::new(1000);
        assert_eq!(multi.device_count(), 0);
        assert!(!multi.is_valid());

        multi.left = Some(SensorFrame::new(1000, "left"));
        assert_eq!(multi.device_count(), 1);
        assert!(multi.is_valid());

        multi.right = Some(SensorFrame::new(1000, "right"));
        assert_eq!(multi.device_count(), 2);

        let frames = multi.all_frames();
        assert_eq!(frames.len(), 2);
    }
}

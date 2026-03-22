//! Device information and capabilities
//!
//! Types for describing devices in the network.

use serde::{Deserialize, Serialize};

/// Device role in the system
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum DeviceRole {
    /// Primary controller (usually the brain/compute device)
    Primary,

    /// Secondary sensor device (e.g., watch, phone)
    Secondary,

    /// Audio output device
    AudioOutput,

    /// Visualization display
    Display,

    /// Recording/logging device
    Recorder,

    /// Generic client
    Client,
}

impl DeviceRole {
    /// Get human-readable name
    pub fn name(&self) -> &'static str {
        match self {
            DeviceRole::Primary => "Primary",
            DeviceRole::Secondary => "Secondary",
            DeviceRole::AudioOutput => "Audio Output",
            DeviceRole::Display => "Display",
            DeviceRole::Recorder => "Recorder",
            DeviceRole::Client => "Client",
        }
    }

    /// Check if this role can send control commands
    pub fn can_control(&self) -> bool {
        matches!(self, DeviceRole::Primary | DeviceRole::Client)
    }

    /// Check if this role produces sensor data
    pub fn produces_sensors(&self) -> bool {
        matches!(self, DeviceRole::Primary | DeviceRole::Secondary)
    }

    /// Check if this role consumes audio
    pub fn consumes_audio(&self) -> bool {
        matches!(self, DeviceRole::AudioOutput | DeviceRole::Primary)
    }
}

impl Default for DeviceRole {
    fn default() -> Self {
        DeviceRole::Client
    }
}

/// Device capability
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum Capability {
    /// Can produce sensor frames
    SensorProducer,

    /// Can consume sensor frames
    SensorConsumer,

    /// Can produce control packets (cc-brain)
    ControlPacketProducer,

    /// Can consume control packets
    ControlPacketConsumer,

    /// Can produce audio
    AudioProducer,

    /// Can consume/play audio
    AudioConsumer,

    /// Can record sessions
    SessionRecorder,

    /// Can display visualization
    Visualizer,

    /// Can send control commands
    Controller,

    /// Supports MIDI input
    MidiInput,

    /// Supports MIDI output
    MidiOutput,

    /// Supports time synchronization (NTP/PTP)
    TimeSync,

    /// Has accelerometer
    Accelerometer,

    /// Has gyroscope
    Gyroscope,

    /// Has magnetometer
    Magnetometer,

    /// Has heart rate sensor
    HeartRate,

    /// Has GPS
    Gps,

    /// Custom capability
    Custom(String),
}

impl Capability {
    /// Get capability name
    pub fn name(&self) -> String {
        match self {
            Capability::SensorProducer => "Sensor Producer".to_string(),
            Capability::SensorConsumer => "Sensor Consumer".to_string(),
            Capability::ControlPacketProducer => "Control Packet Producer".to_string(),
            Capability::ControlPacketConsumer => "Control Packet Consumer".to_string(),
            Capability::AudioProducer => "Audio Producer".to_string(),
            Capability::AudioConsumer => "Audio Consumer".to_string(),
            Capability::SessionRecorder => "Session Recorder".to_string(),
            Capability::Visualizer => "Visualizer".to_string(),
            Capability::Controller => "Controller".to_string(),
            Capability::MidiInput => "MIDI Input".to_string(),
            Capability::MidiOutput => "MIDI Output".to_string(),
            Capability::TimeSync => "Time Sync".to_string(),
            Capability::Accelerometer => "Accelerometer".to_string(),
            Capability::Gyroscope => "Gyroscope".to_string(),
            Capability::Magnetometer => "Magnetometer".to_string(),
            Capability::HeartRate => "Heart Rate".to_string(),
            Capability::Gps => "GPS".to_string(),
            Capability::Custom(s) => s.clone(),
        }
    }
}

/// Device platform
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum DevicePlatform {
    /// iOS (iPhone, iPad, Apple Watch)
    Ios,

    /// macOS
    MacOs,

    /// Windows
    Windows,

    /// Linux
    Linux,

    /// Android
    Android,

    /// Web browser
    Web,

    /// Embedded device
    Embedded,

    /// Unknown/other
    Other(String),
}

impl DevicePlatform {
    /// Get platform name
    pub fn name(&self) -> String {
        match self {
            DevicePlatform::Ios => "iOS".to_string(),
            DevicePlatform::MacOs => "macOS".to_string(),
            DevicePlatform::Windows => "Windows".to_string(),
            DevicePlatform::Linux => "Linux".to_string(),
            DevicePlatform::Android => "Android".to_string(),
            DevicePlatform::Web => "Web".to_string(),
            DevicePlatform::Embedded => "Embedded".to_string(),
            DevicePlatform::Other(s) => s.clone(),
        }
    }
}

/// Device information
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct DeviceInfo {
    /// Unique device ID (UUID or hardware ID)
    pub device_id: String,

    /// Device name (user-friendly)
    pub name: String,

    /// Device role
    pub role: DeviceRole,

    /// Device platform
    pub platform: DevicePlatform,

    /// Device capabilities
    pub capabilities: Vec<Capability>,

    /// Device model (e.g., "iPhone 15 Pro", "MacBook Pro")
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,

    /// OS version
    #[serde(skip_serializing_if = "Option::is_none")]
    pub os_version: Option<String>,

    /// App version
    #[serde(skip_serializing_if = "Option::is_none")]
    pub app_version: Option<String>,

    /// Protocol version
    pub protocol_version: String,

    /// Network address (IP:port)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub network_address: Option<String>,

    /// Battery level (0.0 - 1.0)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub battery_level: Option<f32>,

    /// Is charging
    #[serde(skip_serializing_if = "Option::is_none")]
    pub is_charging: Option<bool>,

    /// Last seen timestamp (microseconds)
    pub last_seen_us: i64,

    /// Custom metadata
    #[serde(skip_serializing_if = "Option::is_none")]
    pub metadata: Option<std::collections::HashMap<String, String>>,
}

impl DeviceInfo {
    /// Create new device info
    pub fn new(
        device_id: impl Into<String>,
        name: impl Into<String>,
        role: DeviceRole,
        platform: DevicePlatform,
    ) -> Self {
        Self {
            device_id: device_id.into(),
            name: name.into(),
            role,
            platform,
            capabilities: Vec::new(),
            model: None,
            os_version: None,
            app_version: None,
            protocol_version: crate::PROTOCOL_VERSION.to_string(),
            network_address: None,
            battery_level: None,
            is_charging: None,
            last_seen_us: Self::now_us(),
            metadata: None,
        }
    }

    /// Add a capability
    pub fn add_capability(mut self, cap: Capability) -> Self {
        self.capabilities.push(cap);
        self
    }

    /// Set model
    pub fn model(mut self, model: impl Into<String>) -> Self {
        self.model = Some(model.into());
        self
    }

    /// Set OS version
    pub fn os_version(mut self, version: impl Into<String>) -> Self {
        self.os_version = Some(version.into());
        self
    }

    /// Set app version
    pub fn app_version(mut self, version: impl Into<String>) -> Self {
        self.app_version = Some(version.into());
        self
    }

    /// Set network address
    pub fn network_address(mut self, addr: impl Into<String>) -> Self {
        self.network_address = Some(addr.into());
        self
    }

    /// Set battery level
    pub fn battery_level(mut self, level: f32) -> Self {
        self.battery_level = Some(level.clamp(0.0, 1.0));
        self
    }

    /// Set charging state
    pub fn charging(mut self, charging: bool) -> Self {
        self.is_charging = Some(charging);
        self
    }

    /// Update last seen timestamp to now
    pub fn update_last_seen(&mut self) {
        self.last_seen_us = Self::now_us();
    }

    /// Get age since last seen (seconds)
    pub fn age_seconds(&self) -> f64 {
        let now = Self::now_us();
        (now - self.last_seen_us) as f64 / 1_000_000.0
    }

    /// Check if device has a capability
    pub fn has_capability(&self, cap: &Capability) -> bool {
        self.capabilities.contains(cap)
    }

    /// Check if device is online (seen within last 10 seconds)
    pub fn is_online(&self) -> bool {
        self.age_seconds() < 10.0
    }

    fn now_us() -> i64 {
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_micros() as i64
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_device_role() {
        let role = DeviceRole::Primary;
        assert_eq!(role.name(), "Primary");
        assert!(role.can_control());
        assert!(role.produces_sensors());
    }

    #[test]
    fn test_device_role_defaults() {
        let role = DeviceRole::default();
        assert_eq!(role, DeviceRole::Client);
    }

    #[test]
    fn test_capability_name() {
        let cap = Capability::SensorProducer;
        assert_eq!(cap.name(), "Sensor Producer");

        let custom = Capability::Custom("TestCap".to_string());
        assert_eq!(custom.name(), "TestCap");
    }

    #[test]
    fn test_device_platform() {
        let platform = DevicePlatform::Ios;
        assert_eq!(platform.name(), "iOS");

        let other = DevicePlatform::Other("Custom".to_string());
        assert_eq!(other.name(), "Custom");
    }

    #[test]
    fn test_device_info_creation() {
        let info = DeviceInfo::new(
            "device1",
            "Test Device",
            DeviceRole::Primary,
            DevicePlatform::Ios,
        );

        assert_eq!(info.device_id, "device1");
        assert_eq!(info.name, "Test Device");
        assert_eq!(info.role, DeviceRole::Primary);
        assert_eq!(info.protocol_version, crate::PROTOCOL_VERSION);
    }

    #[test]
    fn test_device_info_builder() {
        let info = DeviceInfo::new(
            "device1",
            "iPhone",
            DeviceRole::Secondary,
            DevicePlatform::Ios,
        )
        .model("iPhone 15 Pro")
        .os_version("iOS 17.0")
        .app_version("1.0.0")
        .network_address("192.168.1.100:8080")
        .battery_level(0.85)
        .charging(true)
        .add_capability(Capability::Accelerometer)
        .add_capability(Capability::Gyroscope);

        assert_eq!(info.model.as_deref(), Some("iPhone 15 Pro"));
        assert_eq!(info.battery_level, Some(0.85));
        assert_eq!(info.is_charging, Some(true));
        assert_eq!(info.capabilities.len(), 2);
    }

    #[test]
    fn test_device_has_capability() {
        let info = DeviceInfo::new(
            "device1",
            "Test",
            DeviceRole::Primary,
            DevicePlatform::Ios,
        )
        .add_capability(Capability::SensorProducer)
        .add_capability(Capability::Accelerometer);

        assert!(info.has_capability(&Capability::SensorProducer));
        assert!(info.has_capability(&Capability::Accelerometer));
        assert!(!info.has_capability(&Capability::MidiInput));
    }

    #[test]
    fn test_device_age() {
        let mut info = DeviceInfo::new(
            "device1",
            "Test",
            DeviceRole::Primary,
            DevicePlatform::Ios,
        );

        std::thread::sleep(std::time::Duration::from_millis(100));
        let age = info.age_seconds();
        assert!(age >= 0.1 && age < 0.5);

        info.update_last_seen();
        let new_age = info.age_seconds();
        assert!(new_age < 0.01);
    }

    #[test]
    fn test_device_is_online() {
        let info = DeviceInfo::new(
            "device1",
            "Test",
            DeviceRole::Primary,
            DevicePlatform::Ios,
        );
        assert!(info.is_online());
    }

    #[test]
    fn test_serialization() {
        let info = DeviceInfo::new(
            "device1",
            "Test",
            DeviceRole::Primary,
            DevicePlatform::Ios,
        )
        .add_capability(Capability::SensorProducer);

        let json = serde_json::to_string(&info).unwrap();
        let deserialized: DeviceInfo = serde_json::from_str(&json).unwrap();
        assert_eq!(info.device_id, deserialized.device_id);
    }
}

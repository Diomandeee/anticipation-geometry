//! Control messages
//!
//! Commands for controlling recording sessions, playback, and system state.

use serde::{Deserialize, Serialize};
use crate::strudel_ir::{PatternEdit, edit::EditSequence};

/// Control message type
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum ControlMessageType {
    /// Command (request action)
    Command,

    /// Query (request information)
    Query,

    /// Response to query
    Response,

    /// State notification
    Notification,
}

/// Control message
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ControlMessage {
    /// Message type
    pub message_type: ControlMessageType,

    /// Command/query payload
    pub command: Command,
}

impl ControlMessage {
    /// Create a command message
    pub fn command(cmd: Command) -> Self {
        Self {
            message_type: ControlMessageType::Command,
            command: cmd,
        }
    }

    /// Create a query message
    pub fn query(cmd: Command) -> Self {
        Self {
            message_type: ControlMessageType::Query,
            command: cmd,
        }
    }

    /// Create a response message
    pub fn response(cmd: Command) -> Self {
        Self {
            message_type: ControlMessageType::Response,
            command: cmd,
        }
    }

    /// Create a notification message
    pub fn notification(cmd: Command) -> Self {
        Self {
            message_type: ControlMessageType::Notification,
            command: cmd,
        }
    }
}

/// Command variants
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum Command {
    // === Session control ===

    /// Start a new recording session
    StartRecording {
        /// Session name
        session_name: String,
        /// Session metadata
        #[serde(skip_serializing_if = "Option::is_none")]
        metadata: Option<SessionMetadata>,
    },

    /// Stop current recording session
    StopRecording,

    /// Pause recording (resume-able)
    PauseRecording,

    /// Resume paused recording
    ResumeRecording,

    /// Cancel recording (discard data)
    CancelRecording,

    /// Save recorded session
    SaveRecording {
        /// Save location
        path: String,
    },

    // === Markers and annotations ===

    /// Add a marker at current time
    AddMarker {
        /// Marker label
        label: String,
        /// Optional marker type
        #[serde(skip_serializing_if = "Option::is_none")]
        marker_type: Option<String>,
    },

    /// Remove marker
    RemoveMarker {
        /// Marker ID
        marker_id: String,
    },

    // === Pattern editing (real-time) ===

    /// Apply a pattern edit
    ApplyEdit {
        edit: PatternEdit,
    },

    /// Apply multiple edits atomically
    ApplyEditSequence {
        sequence: EditSequence,
    },

    /// Undo last edit
    UndoEdit,

    /// Redo last undone edit
    RedoEdit,

    /// Clear all edits (reset to initial state)
    ResetEdits,

    // === Playback control ===

    /// Start playback
    StartPlayback {
        /// Playback position (seconds)
        #[serde(skip_serializing_if = "Option::is_none")]
        start_position_s: Option<f32>,
    },

    /// Stop playback
    StopPlayback,

    /// Pause playback
    PausePlayback,

    /// Seek to position
    Seek {
        /// Position in seconds
        position_s: f32,
    },

    /// Set playback speed
    SetPlaybackSpeed {
        /// Speed multiplier (1.0 = normal)
        speed: f32,
    },

    // === System control ===

    /// Set global tempo
    SetTempo {
        /// Tempo in BPM
        bpm: f32,
    },

    /// Set global key
    SetKey {
        /// Key (e.g., "C", "Am", "F#")
        key: String,
    },

    /// Set global time signature
    SetTimeSignature {
        /// Numerator (beats per bar)
        numerator: u32,
        /// Denominator (note value)
        denominator: u32,
    },

    /// Enable/disable metronome
    SetMetronome {
        enabled: bool,
        /// Volume (0.0 - 1.0)
        volume: f32,
    },

    /// Set coupling mode
    SetCouplingMode {
        mode: crate::CouplingMode,
    },

    /// Set latent field program
    SetLatentProgram {
        program_id: String,
    },

    // === Device control ===

    /// Calibrate sensors
    CalibrateSensors,

    /// Reset sensor fusion
    ResetSensorFusion,

    /// Set device role
    SetDeviceRole {
        role: crate::network::DeviceRole,
    },

    // === Query commands ===

    /// Query current session state
    QuerySessionState,

    /// Query current arrangement
    QueryArrangement,

    /// Query current latent state
    QueryLatentState,

    /// Query device status
    QueryDeviceStatus,

    // === Response commands ===

    /// Response: session state
    SessionState {
        state: SessionState,
    },

    /// Response: arrangement
    Arrangement {
        layers: Vec<crate::strudel_ir::Layer>,
    },

    /// Response: latent state
    LatentState {
        state: crate::LatentState,
    },

    /// Response: device status
    DeviceStatus {
        status: DeviceStatus,
    },

    // === Generic ===

    /// Success response
    Success {
        message: String,
    },

    /// Error response
    Error {
        code: u32,
        message: String,
    },
}

/// Session state
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum SessionState {
    Idle,
    Recording,
    Paused,
    Playing,
    Stopped,
}

impl Default for SessionState {
    fn default() -> Self {
        SessionState::Idle
    }
}

/// Session metadata
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct SessionMetadata {
    /// Artist/performer name
    #[serde(skip_serializing_if = "Option::is_none")]
    pub artist: Option<String>,

    /// Session title
    #[serde(skip_serializing_if = "Option::is_none")]
    pub title: Option<String>,

    /// Session tags
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tags: Option<Vec<String>>,

    /// BPM
    #[serde(skip_serializing_if = "Option::is_none")]
    pub bpm: Option<f32>,

    /// Key
    #[serde(skip_serializing_if = "Option::is_none")]
    pub key: Option<String>,

    /// Custom metadata
    #[serde(skip_serializing_if = "Option::is_none")]
    pub custom: Option<std::collections::HashMap<String, String>>,
}

impl SessionMetadata {
    /// Create empty metadata
    pub fn new() -> Self {
        Self {
            artist: None,
            title: None,
            tags: None,
            bpm: None,
            key: None,
            custom: None,
        }
    }

    /// Set artist
    pub fn artist(mut self, artist: impl Into<String>) -> Self {
        self.artist = Some(artist.into());
        self
    }

    /// Set title
    pub fn title(mut self, title: impl Into<String>) -> Self {
        self.title = Some(title.into());
        self
    }

    /// Add tag
    pub fn add_tag(mut self, tag: impl Into<String>) -> Self {
        self.tags.get_or_insert_with(Vec::new).push(tag.into());
        self
    }

    /// Set BPM
    pub fn bpm(mut self, bpm: f32) -> Self {
        self.bpm = Some(bpm);
        self
    }

    /// Set key
    pub fn key(mut self, key: impl Into<String>) -> Self {
        self.key = Some(key.into());
        self
    }
}

impl Default for SessionMetadata {
    fn default() -> Self {
        Self::new()
    }
}

/// Device status
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct DeviceStatus {
    /// Device is online
    pub online: bool,

    /// Battery level (0.0 - 1.0)
    pub battery_level: Option<f32>,

    /// Is charging
    pub is_charging: Option<bool>,

    /// CPU usage (0.0 - 1.0)
    pub cpu_usage: Option<f32>,

    /// Memory usage (bytes)
    pub memory_bytes: Option<u64>,

    /// Network latency (milliseconds)
    pub network_latency_ms: Option<f32>,

    /// Sensor frame rate (Hz)
    pub sensor_fps: Option<f32>,

    /// Control packet frame rate (Hz)
    pub control_fps: Option<f32>,
}

impl DeviceStatus {
    /// Create default online status
    pub fn online() -> Self {
        Self {
            online: true,
            battery_level: None,
            is_charging: None,
            cpu_usage: None,
            memory_bytes: None,
            network_latency_ms: None,
            sensor_fps: None,
            control_fps: None,
        }
    }

    /// Create offline status
    pub fn offline() -> Self {
        Self {
            online: false,
            battery_level: None,
            is_charging: None,
            cpu_usage: None,
            memory_bytes: None,
            network_latency_ms: None,
            sensor_fps: None,
            control_fps: None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_control_message_creation() {
        let cmd = Command::StartRecording {
            session_name: "test".to_string(),
            metadata: None,
        };
        let msg = ControlMessage::command(cmd);

        assert_eq!(msg.message_type, ControlMessageType::Command);
    }

    #[test]
    fn test_session_metadata_builder() {
        let metadata = SessionMetadata::new()
            .artist("Test Artist")
            .title("Test Session")
            .add_tag("experimental")
            .bpm(120.0)
            .key("Am");

        assert_eq!(metadata.artist.as_deref(), Some("Test Artist"));
        assert_eq!(metadata.bpm, Some(120.0));
        assert_eq!(metadata.tags.as_ref().unwrap().len(), 1);
    }

    #[test]
    fn test_device_status() {
        let status = DeviceStatus::online();
        assert!(status.online);

        let offline = DeviceStatus::offline();
        assert!(!offline.online);
    }

    #[test]
    fn test_session_state() {
        let state = SessionState::Recording;
        assert_eq!(state, SessionState::Recording);

        let default_state = SessionState::default();
        assert_eq!(default_state, SessionState::Idle);
    }

    #[test]
    fn test_command_serialization() {
        let cmd = Command::SetTempo { bpm: 120.0 };
        let json = serde_json::to_string(&cmd).unwrap();
        let deserialized: Command = serde_json::from_str(&json).unwrap();
        assert_eq!(cmd, deserialized);
    }
}

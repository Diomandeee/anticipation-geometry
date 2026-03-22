//! Network message types
//!
//! All messages flowing over the network between devices and components.
//! Designed for MessagePack serialization with compact encoding.

pub mod control;
pub mod data;
pub mod device;
pub mod sync;

pub use control::{ControlMessage, ControlMessageType};
pub use data::{DataMessage, DataMessageType};
pub use device::{DeviceInfo, DeviceRole, Capability, DevicePlatform};
pub use sync::{SyncMessage, TimeSyncRequest, TimeSyncResponse};

use serde::{Deserialize, Serialize};

/// Network message envelope
///
/// All messages are wrapped in this envelope for routing and versioning.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct NetworkMessage {
    /// Protocol version (for compatibility checking)
    pub protocol_version: String,

    /// Message ID (unique per message)
    pub message_id: u64,

    /// Timestamp when message was created (microseconds)
    pub timestamp_us: i64,

    /// Sender device ID
    pub sender_id: String,

    /// Target device ID (None = broadcast)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub target_id: Option<String>,

    /// Message payload
    pub payload: MessagePayload,

    /// Priority (higher = more urgent)
    #[serde(default)]
    pub priority: u8,

    /// Whether this message requires acknowledgment
    #[serde(default)]
    pub requires_ack: bool,

    /// Reply to message ID (for responses)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reply_to: Option<u64>,
}

impl NetworkMessage {
    /// Create a new network message
    pub fn new(
        sender_id: impl Into<String>,
        payload: MessagePayload,
    ) -> Self {
        Self {
            protocol_version: crate::PROTOCOL_VERSION.to_string(),
            message_id: Self::generate_id(),
            timestamp_us: Self::now_us(),
            sender_id: sender_id.into(),
            target_id: None,
            payload,
            priority: 0,
            requires_ack: false,
            reply_to: None,
        }
    }

    /// Create a broadcast message
    pub fn broadcast(
        sender_id: impl Into<String>,
        payload: MessagePayload,
    ) -> Self {
        Self::new(sender_id, payload)
    }

    /// Create a targeted message
    pub fn to(
        sender_id: impl Into<String>,
        target_id: impl Into<String>,
        payload: MessagePayload,
    ) -> Self {
        let mut msg = Self::new(sender_id, payload);
        msg.target_id = Some(target_id.into());
        msg
    }

    /// Create a reply to another message
    pub fn reply_to(
        sender_id: impl Into<String>,
        original: &NetworkMessage,
        payload: MessagePayload,
    ) -> Self {
        let mut msg = Self::new(sender_id, payload);
        msg.target_id = Some(original.sender_id.clone());
        msg.reply_to = Some(original.message_id);
        msg
    }

    /// Set priority
    pub fn priority(mut self, priority: u8) -> Self {
        self.priority = priority;
        self
    }

    /// Require acknowledgment
    pub fn require_ack(mut self) -> Self {
        self.requires_ack = true;
        self
    }

    /// Check if this message is a broadcast
    pub fn is_broadcast(&self) -> bool {
        self.target_id.is_none()
    }

    /// Check if this message is targeted
    pub fn is_targeted(&self) -> bool {
        self.target_id.is_some()
    }

    /// Check if this is a reply
    pub fn is_reply(&self) -> bool {
        self.reply_to.is_some()
    }

    /// Get age of this message in seconds
    pub fn age_seconds(&self) -> f64 {
        let now = Self::now_us();
        (now - self.timestamp_us) as f64 / 1_000_000.0
    }

    // Internal helpers
    fn generate_id() -> u64 {
        use std::sync::atomic::{AtomicU64, Ordering};
        static COUNTER: AtomicU64 = AtomicU64::new(1);
        COUNTER.fetch_add(1, Ordering::Relaxed)
    }

    fn now_us() -> i64 {
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_micros() as i64
    }
}

/// Message payload variants
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum MessagePayload {
    /// Control message (commands, state changes)
    Control(ControlMessage),

    /// Data message (sensor frames, audio, etc.)
    Data(DataMessage),

    /// Device management message
    Device(DeviceMessage),

    /// Time synchronization message
    Sync(SyncMessage),

    /// Acknowledgment
    Ack {
        /// Message ID being acknowledged
        ack_message_id: u64,
    },

    /// Error response
    Error {
        /// Error code
        code: u32,
        /// Error message
        message: String,
        /// Message ID that caused the error
        #[serde(skip_serializing_if = "Option::is_none")]
        error_message_id: Option<u64>,
    },

    /// Ping (for keepalive)
    Ping,

    /// Pong (response to ping)
    Pong,
}

impl MessagePayload {
    /// Get the type name of this payload
    pub fn type_name(&self) -> &'static str {
        match self {
            MessagePayload::Control(_) => "Control",
            MessagePayload::Data(_) => "Data",
            MessagePayload::Device(_) => "Device",
            MessagePayload::Sync(_) => "Sync",
            MessagePayload::Ack { .. } => "Ack",
            MessagePayload::Error { .. } => "Error",
            MessagePayload::Ping => "Ping",
            MessagePayload::Pong => "Pong",
        }
    }

    /// Check if this payload requires acknowledgment by default
    pub fn requires_ack_by_default(&self) -> bool {
        matches!(
            self,
            MessagePayload::Control(_) | MessagePayload::Device(_)
        )
    }
}

/// Device management message
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum DeviceMessage {
    /// Announce device presence
    Announce(DeviceInfo),

    /// Device going offline
    Goodbye,

    /// Request device info
    Query,

    /// Response to query
    Info(DeviceInfo),

    /// Update device capabilities
    UpdateCapabilities {
        capabilities: Vec<Capability>,
    },

    /// Update device role
    UpdateRole {
        role: DeviceRole,
    },
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_network_message_creation() {
        let payload = MessagePayload::Ping;
        let msg = NetworkMessage::new("device1", payload);

        assert_eq!(msg.sender_id, "device1");
        assert_eq!(msg.protocol_version, crate::PROTOCOL_VERSION);
        assert!(msg.is_broadcast());
        assert!(!msg.is_targeted());
        assert!(!msg.is_reply());
    }

    #[test]
    fn test_targeted_message() {
        let payload = MessagePayload::Pong;
        let msg = NetworkMessage::to("device1", "device2", payload);

        assert_eq!(msg.sender_id, "device1");
        assert_eq!(msg.target_id.as_deref(), Some("device2"));
        assert!(msg.is_targeted());
        assert!(!msg.is_broadcast());
    }

    #[test]
    fn test_reply_message() {
        let original = NetworkMessage::new("device1", MessagePayload::Ping);
        let reply = NetworkMessage::reply_to("device2", &original, MessagePayload::Pong);

        assert_eq!(reply.sender_id, "device2");
        assert_eq!(reply.target_id.as_deref(), Some("device1"));
        assert_eq!(reply.reply_to, Some(original.message_id));
        assert!(reply.is_reply());
    }

    #[test]
    fn test_message_builder() {
        let msg = NetworkMessage::new("device1", MessagePayload::Ping)
            .priority(10)
            .require_ack();

        assert_eq!(msg.priority, 10);
        assert!(msg.requires_ack);
    }

    #[test]
    fn test_message_age() {
        let msg = NetworkMessage::new("device1", MessagePayload::Ping);
        std::thread::sleep(std::time::Duration::from_millis(50));
        let age = msg.age_seconds();
        assert!(age >= 0.05 && age < 0.2);
    }

    #[test]
    fn test_payload_type_name() {
        assert_eq!(MessagePayload::Ping.type_name(), "Ping");
        assert_eq!(MessagePayload::Pong.type_name(), "Pong");
        assert_eq!(
            MessagePayload::Error {
                code: 1,
                message: "test".to_string(),
                error_message_id: None
            }
            .type_name(),
            "Error"
        );
    }

    #[test]
    fn test_serialization() {
        let msg = NetworkMessage::new("device1", MessagePayload::Ping);
        let json = serde_json::to_string(&msg).unwrap();
        let deserialized: NetworkMessage = serde_json::from_str(&json).unwrap();
        assert_eq!(msg.sender_id, deserialized.sender_id);
    }
}

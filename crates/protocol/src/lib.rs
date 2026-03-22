//! # cc-protocol
//!
//! Unified protocol definitions for Computational Choreography.
//!
//! This crate provides all message types and serialization for communication
//! between iOS app, desktop control center, Rust audio engine, and Python tools.
//!
//! ## Core Concepts
//!
//! The protocol implements the **dual-time contract**:
//! - **Latent time**: Continuous, body-relative (from embodied physics)
//! - **Execution time**: Quantized bar/beat grid (for audio scheduling)
//! - **Coherence gate**: Dynamic coupling between them
//!
//! ## Main Types
//!
//! - [`ControlPacket`] - The primary message flowing through the system
//! - [`SensorFrame`] - Raw sensor input
//! - [`LatentState`] - Embodied physics representation
//! - [`CoherenceMetrics`] - Dual-time system state
//! - [`SectionState`] - High-level state machine
//!
//! ## Strudel-IR
//!
//! Symbolic music representation based on Strudel/TidalCycles:
//! - [`Pattern`] - Musical pattern definition
//! - [`PatternEdit`] - Incremental modifications
//! - [`Layer`] - Instrument + pattern + FX
//!
//! ## Example
//!
//! ```rust,ignore
//! use cc_protocol::*;
//!
//! // Receive sensor data
//! let frame = SensorFrame { /* ... */ };
//!
//! // Process (in cc-brain)
//! let packet = brain.process(&frame);
//!
//! // Check coherence
//! match packet.coherence.coupling_mode {
//!     CouplingMode::HardLock => {
//!         // Can apply bar-aligned edits
//!         let edit = PatternEdit::new()
//!             .target("kick")
//!             .operation(EditOp::SetPattern(new_pattern))
//!             .quantize(Quantization::Bar);
//!     }
//!     CouplingMode::Free => {
//!         // Only continuous modulation
//!     }
//!     _ => {}
//! }
//! ```

// Core modules
pub mod sensor;
pub mod latent;
pub mod coherence;
pub mod section_state;
pub mod control_packet;
pub mod clock;
pub mod strudel_ir;
pub mod network;
pub mod mocopi;
pub mod mocopi_state;

// Serialization utilities
mod serialization;

// Conditional binding generation
#[cfg(feature = "swift-bindings")]
pub mod bindings;

// Generated Strudel API (from build.rs)
#[cfg(feature = "strudel-api")]
mod strudel_api_generated {
    include!(concat!(env!("OUT_DIR"), "/strudel_api_generated.rs"));
}

#[cfg(feature = "strudel-api")]
pub use strudel_api_generated::*;

// Re-exports for convenience
pub use sensor::{SensorFrame, MultiDeviceFrame};
pub use latent::{LatentState, LatentGeometry};
pub use coherence::{CoherenceMetrics, CouplingMode};
pub use section_state::{SectionState, SectionStateContext};
pub use control_packet::{ControlPacket, ControlCodes, RegionType};
pub use clock::{ExecutionClock, Quantization};
pub use strudel_ir::{Pattern, PatternEdit, PatternModifier, Layer, Effect, EffectType, FilterType, Note, Instrument, Waveform};
pub use network::{ControlMessage, DataMessage, DeviceInfo, DeviceRole, DevicePlatform, Capability, NetworkMessage, MessagePayload, TimeSyncRequest, TimeSyncResponse, SyncMessage};
pub use network::control::{Command, SessionMetadata};
pub use serialization::{serialize_msgpack, deserialize_msgpack, serialize_json, deserialize_json, Format, measure_size, BatchSerializer};
pub use mocopi::{MocopiBoneId, MocopiKitType, BoneTransform, MocopiSkeletonFrame};
pub use mocopi_state::{MocopiStateFrame, MocopiStateBatch, MotionControls, BoneTransformWithVelocity, LegacyMocopiFrame, LegacyMocopiBatch};

/// Protocol version for compatibility checking
pub const PROTOCOL_VERSION: &str = "0.1.0";

/// Result type for protocol operations
pub type Result<T> = std::result::Result<T, ProtocolError>;

/// Error types for protocol operations
#[derive(Debug, thiserror::Error)]
pub enum ProtocolError {
    #[error("Serialization error: {0}")]
    Serialization(String),

    #[error("Deserialization error: {0}")]
    Deserialization(String),

    #[error("Protocol version mismatch: expected {expected}, got {actual}")]
    VersionMismatch { expected: String, actual: String },

    #[error("Invalid message type: {0}")]
    InvalidMessageType(String),

    #[error("Invalid value: {0}")]
    InvalidValue(String),
}

impl From<serde_json::Error> for ProtocolError {
    fn from(e: serde_json::Error) -> Self {
        ProtocolError::Serialization(e.to_string())
    }
}

impl From<rmp_serde::encode::Error> for ProtocolError {
    fn from(e: rmp_serde::encode::Error) -> Self {
        ProtocolError::Serialization(e.to_string())
    }
}

impl From<rmp_serde::decode::Error> for ProtocolError {
    fn from(e: rmp_serde::decode::Error) -> Self {
        ProtocolError::Deserialization(e.to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_protocol_version() {
        assert!(!PROTOCOL_VERSION.is_empty());
        // Version should be semver-compatible
        let parts: Vec<&str> = PROTOCOL_VERSION.split('.').collect();
        assert_eq!(parts.len(), 3);
    }
}

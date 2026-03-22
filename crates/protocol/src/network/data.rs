//! Data messages
//!
//! Streaming data: sensor frames, control packets, audio chunks, etc.

use serde::{Deserialize, Serialize};
use crate::{SensorFrame, ControlPacket};

/// Data message type
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum DataMessageType {
    /// Real-time stream
    Stream,

    /// Batch/bulk transfer
    Batch,

    /// Response to query
    Response,
}

/// Data message
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct DataMessage {
    /// Message type
    pub message_type: DataMessageType,

    /// Data payload
    pub data: DataPayload,
}

impl DataMessage {
    /// Create a stream message
    pub fn stream(data: DataPayload) -> Self {
        Self {
            message_type: DataMessageType::Stream,
            data,
        }
    }

    /// Create a batch message
    pub fn batch(data: DataPayload) -> Self {
        Self {
            message_type: DataMessageType::Batch,
            data,
        }
    }

    /// Create a response message
    pub fn response(data: DataPayload) -> Self {
        Self {
            message_type: DataMessageType::Response,
            data,
        }
    }
}

/// Data payload variants
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum DataPayload {
    /// Single sensor frame
    SensorFrame(SensorFrame),

    /// Batch of sensor frames
    SensorBatch {
        frames: Vec<SensorFrame>,
        /// Sequence number (for ordering)
        sequence: u64,
    },

    /// Single control packet
    ControlPacket(Box<ControlPacket>),

    /// Batch of control packets
    ControlPacketBatch {
        packets: Vec<ControlPacket>,
        /// Sequence number
        sequence: u64,
    },

    /// Audio chunk
    AudioChunk {
        /// Sample rate (Hz)
        sample_rate: u32,
        /// Number of channels
        channels: u32,
        /// Audio samples (interleaved if multi-channel)
        samples: Vec<f32>,
        /// Timestamp (microseconds)
        timestamp_us: i64,
        /// Sequence number
        sequence: u64,
    },

    /// MIDI event
    MidiEvent {
        /// MIDI bytes
        bytes: Vec<u8>,
        /// Timestamp (microseconds)
        timestamp_us: i64,
    },

    /// MIDI batch
    MidiBatch {
        events: Vec<MidiEventData>,
        /// Sequence number
        sequence: u64,
    },

    /// Marker event
    Marker {
        /// Marker label
        label: String,
        /// Timestamp (microseconds)
        timestamp_us: i64,
        /// Marker type
        #[serde(skip_serializing_if = "Option::is_none")]
        marker_type: Option<String>,
    },

    /// Video frame metadata (actual image data sent separately)
    VideoFrameMetadata {
        /// Frame number
        frame_number: u64,
        /// Timestamp (microseconds)
        timestamp_us: i64,
        /// Width
        width: u32,
        /// Height
        height: u32,
        /// Format (e.g., "RGBA", "YUV420p")
        format: String,
        /// Size in bytes
        size_bytes: u64,
    },

    /// Binary blob (generic container)
    Blob {
        /// Blob type identifier
        blob_type: String,
        /// Binary data
        data: Vec<u8>,
        /// Timestamp (microseconds)
        timestamp_us: i64,
    },

    /// Trajectory data (for visualization)
    Trajectory {
        /// 2D or 3D points
        points: Vec<Vec<f32>>,
        /// Timestamps (microseconds)
        timestamps_us: Vec<i64>,
        /// Trajectory metadata
        #[serde(skip_serializing_if = "Option::is_none")]
        metadata: Option<TrajectoryMetadata>,
    },

    /// Session recording metadata
    SessionRecording {
        /// Session ID
        session_id: String,
        /// Start time (microseconds)
        start_time_us: i64,
        /// Duration (seconds)
        duration_s: f32,
        /// File paths
        files: Vec<String>,
        /// Session metadata
        #[serde(skip_serializing_if = "Option::is_none")]
        metadata: Option<crate::network::control::SessionMetadata>,
    },
}

/// MIDI event data
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct MidiEventData {
    /// MIDI bytes
    pub bytes: Vec<u8>,
    /// Timestamp (microseconds)
    pub timestamp_us: i64,
}

impl MidiEventData {
    /// Create a MIDI event
    pub fn new(bytes: Vec<u8>, timestamp_us: i64) -> Self {
        Self {
            bytes,
            timestamp_us,
        }
    }

    /// Get MIDI status byte
    pub fn status(&self) -> Option<u8> {
        self.bytes.first().copied()
    }

    /// Get MIDI channel (0-15)
    pub fn channel(&self) -> Option<u8> {
        self.status().map(|s| s & 0x0F)
    }

    /// Check if this is a note on message
    pub fn is_note_on(&self) -> bool {
        self.status().map(|s| (s & 0xF0) == 0x90).unwrap_or(false)
    }

    /// Check if this is a note off message
    pub fn is_note_off(&self) -> bool {
        self.status().map(|s| (s & 0xF0) == 0x80).unwrap_or(false)
    }

    /// Check if this is a control change message
    pub fn is_cc(&self) -> bool {
        self.status().map(|s| (s & 0xF0) == 0xB0).unwrap_or(false)
    }
}

/// Trajectory metadata
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct TrajectoryMetadata {
    /// Trajectory type
    pub trajectory_type: String,

    /// Color (RGB, 0.0 - 1.0)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub color: Option<[f32; 3]>,

    /// Line width
    #[serde(skip_serializing_if = "Option::is_none")]
    pub line_width: Option<f32>,

    /// Custom metadata
    #[serde(skip_serializing_if = "Option::is_none")]
    pub custom: Option<std::collections::HashMap<String, String>>,
}

impl TrajectoryMetadata {
    /// Create new trajectory metadata
    pub fn new(trajectory_type: impl Into<String>) -> Self {
        Self {
            trajectory_type: trajectory_type.into(),
            color: None,
            line_width: None,
            custom: None,
        }
    }

    /// Set color
    pub fn color(mut self, r: f32, g: f32, b: f32) -> Self {
        self.color = Some([r, g, b]);
        self
    }

    /// Set line width
    pub fn line_width(mut self, width: f32) -> Self {
        self.line_width = Some(width);
        self
    }
}

/// Helper for creating audio chunks
pub struct AudioChunkBuilder {
    sample_rate: u32,
    channels: u32,
    samples: Vec<f32>,
    timestamp_us: i64,
    sequence: u64,
}

impl AudioChunkBuilder {
    /// Create a new audio chunk builder
    pub fn new(sample_rate: u32, channels: u32) -> Self {
        Self {
            sample_rate,
            channels,
            samples: Vec::new(),
            timestamp_us: 0,
            sequence: 0,
        }
    }

    /// Set samples
    pub fn samples(mut self, samples: Vec<f32>) -> Self {
        self.samples = samples;
        self
    }

    /// Set timestamp
    pub fn timestamp_us(mut self, timestamp_us: i64) -> Self {
        self.timestamp_us = timestamp_us;
        self
    }

    /// Set sequence number
    pub fn sequence(mut self, sequence: u64) -> Self {
        self.sequence = sequence;
        self
    }

    /// Build the audio chunk payload
    pub fn build(self) -> DataPayload {
        DataPayload::AudioChunk {
            sample_rate: self.sample_rate,
            channels: self.channels,
            samples: self.samples,
            timestamp_us: self.timestamp_us,
            sequence: self.sequence,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_data_message_creation() {
        let payload = DataPayload::Marker {
            label: "test".to_string(),
            timestamp_us: 1000,
            marker_type: None,
        };
        let msg = DataMessage::stream(payload);

        assert_eq!(msg.message_type, DataMessageType::Stream);
    }

    #[test]
    fn test_sensor_batch() {
        let frames = vec![
            SensorFrame::default(),
            SensorFrame::default(),
        ];
        let payload = DataPayload::SensorBatch {
            frames,
            sequence: 1,
        };

        match payload {
            DataPayload::SensorBatch { frames, sequence } => {
                assert_eq!(frames.len(), 2);
                assert_eq!(sequence, 1);
            }
            _ => panic!("Wrong payload type"),
        }
    }

    #[test]
    fn test_audio_chunk_builder() {
        let chunk = AudioChunkBuilder::new(48000, 2)
            .samples(vec![0.1, 0.2, 0.3, 0.4])
            .timestamp_us(1000)
            .sequence(5)
            .build();

        match chunk {
            DataPayload::AudioChunk {
                sample_rate,
                channels,
                samples,
                timestamp_us,
                sequence,
            } => {
                assert_eq!(sample_rate, 48000);
                assert_eq!(channels, 2);
                assert_eq!(samples.len(), 4);
                assert_eq!(timestamp_us, 1000);
                assert_eq!(sequence, 5);
            }
            _ => panic!("Wrong payload type"),
        }
    }

    #[test]
    fn test_midi_event_data() {
        // Note on: 0x90 (channel 0) + note 60 + velocity 100
        let event = MidiEventData::new(vec![0x90, 60, 100], 1000);

        assert_eq!(event.status(), Some(0x90));
        assert_eq!(event.channel(), Some(0));
        assert!(event.is_note_on());
        assert!(!event.is_note_off());
        assert!(!event.is_cc());
    }

    #[test]
    fn test_midi_note_off() {
        // Note off: 0x80 (channel 0) + note 60 + velocity 0
        let event = MidiEventData::new(vec![0x80, 60, 0], 1000);

        assert!(event.is_note_off());
        assert!(!event.is_note_on());
    }

    #[test]
    fn test_midi_cc() {
        // CC: 0xB0 (channel 0) + controller 7 + value 127
        let event = MidiEventData::new(vec![0xB0, 7, 127], 1000);

        assert!(event.is_cc());
        assert!(!event.is_note_on());
        assert!(!event.is_note_off());
    }

    #[test]
    fn test_trajectory_metadata() {
        let meta = TrajectoryMetadata::new("latent_path")
            .color(1.0, 0.0, 0.0)
            .line_width(2.0);

        assert_eq!(meta.trajectory_type, "latent_path");
        assert_eq!(meta.color, Some([1.0, 0.0, 0.0]));
        assert_eq!(meta.line_width, Some(2.0));
    }

    #[test]
    fn test_trajectory_payload() {
        let payload = DataPayload::Trajectory {
            points: vec![vec![0.0, 0.0], vec![1.0, 1.0]],
            timestamps_us: vec![1000, 2000],
            metadata: None,
        };

        match payload {
            DataPayload::Trajectory { points, timestamps_us, .. } => {
                assert_eq!(points.len(), 2);
                assert_eq!(timestamps_us.len(), 2);
            }
            _ => panic!("Wrong payload type"),
        }
    }
}

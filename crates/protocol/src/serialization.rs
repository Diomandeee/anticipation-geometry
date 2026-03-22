//! Serialization utilities
//!
//! Helpers for MessagePack (compact network streaming) and JSON (debugging, logging).

use crate::{ProtocolError, Result};
use serde::{Deserialize, Serialize};

/// Serialize a value to MessagePack bytes
///
/// MessagePack is used for network streaming due to its compact size.
/// Uses named (map) format for forward/backward compatibility when struct fields change.
pub fn serialize_msgpack<T>(value: &T) -> Result<Vec<u8>>
where
    T: Serialize,
{
    rmp_serde::to_vec_named(value).map_err(ProtocolError::from)
}

/// Deserialize a value from MessagePack bytes
///
/// Supports both named (map) and compact (array) formats for compatibility.
pub fn deserialize_msgpack<'a, T>(bytes: &'a [u8]) -> Result<T>
where
    T: Deserialize<'a>,
{
    rmp_serde::from_slice(bytes).map_err(ProtocolError::from)
}

/// Serialize a value to JSON string
///
/// JSON is used for logging, debugging, and human-readable storage.
pub fn serialize_json<T>(value: &T) -> Result<String>
where
    T: Serialize,
{
    serde_json::to_string(value).map_err(ProtocolError::from)
}

/// Serialize a value to pretty JSON string
///
/// Useful for debugging and config files.
pub fn serialize_json_pretty<T>(value: &T) -> Result<String>
where
    T: Serialize,
{
    serde_json::to_string_pretty(value).map_err(ProtocolError::from)
}

/// Deserialize a value from JSON string
pub fn deserialize_json<'a, T>(json: &'a str) -> Result<T>
where
    T: Deserialize<'a>,
{
    serde_json::from_str(json).map_err(ProtocolError::from)
}

/// Serialize a value to JSON bytes
pub fn serialize_json_bytes<T>(value: &T) -> Result<Vec<u8>>
where
    T: Serialize,
{
    serde_json::to_vec(value).map_err(ProtocolError::from)
}

/// Deserialize a value from JSON bytes
pub fn deserialize_json_bytes<'a, T>(bytes: &'a [u8]) -> Result<T>
where
    T: Deserialize<'a>,
{
    serde_json::from_slice(bytes).map_err(ProtocolError::from)
}

/// Serialization format
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Format {
    /// MessagePack (binary, compact)
    MessagePack,
    /// JSON (text, human-readable)
    Json,
    /// JSON pretty-printed
    JsonPretty,
}

impl Format {
    /// Get file extension for this format
    pub fn extension(&self) -> &'static str {
        match self {
            Format::MessagePack => "msgpack",
            Format::Json | Format::JsonPretty => "json",
        }
    }

    /// Get MIME type for this format
    pub fn mime_type(&self) -> &'static str {
        match self {
            Format::MessagePack => "application/msgpack",
            Format::Json | Format::JsonPretty => "application/json",
        }
    }

    /// Check if format is binary
    pub fn is_binary(&self) -> bool {
        matches!(self, Format::MessagePack)
    }

    /// Check if format is text
    pub fn is_text(&self) -> bool {
        matches!(self, Format::Json | Format::JsonPretty)
    }
}

/// Serialize a value with specified format
pub fn serialize<T>(value: &T, format: Format) -> Result<Vec<u8>>
where
    T: Serialize,
{
    match format {
        Format::MessagePack => serialize_msgpack(value),
        Format::Json => serialize_json_bytes(value),
        Format::JsonPretty => {
            let json = serialize_json_pretty(value)?;
            Ok(json.into_bytes())
        }
    }
}

/// Deserialize a value with specified format
pub fn deserialize<'a, T>(bytes: &'a [u8], format: Format) -> Result<T>
where
    T: Deserialize<'a>,
{
    match format {
        Format::MessagePack => deserialize_msgpack(bytes),
        Format::Json | Format::JsonPretty => deserialize_json_bytes(bytes),
    }
}

/// Measure serialization size
pub fn measure_size<T>(value: &T) -> Result<SizeStats>
where
    T: Serialize,
{
    let msgpack_bytes = serialize_msgpack(value)?;
    let json_bytes = serialize_json_bytes(value)?;
    let json_pretty_bytes = serialize_json_pretty(value)?.into_bytes();

    Ok(SizeStats {
        msgpack_bytes: msgpack_bytes.len(),
        json_bytes: json_bytes.len(),
        json_pretty_bytes: json_pretty_bytes.len(),
    })
}

/// Serialization size statistics
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SizeStats {
    /// Size in MessagePack format
    pub msgpack_bytes: usize,
    /// Size in JSON format
    pub json_bytes: usize,
    /// Size in pretty JSON format
    pub json_pretty_bytes: usize,
}

impl SizeStats {
    /// Get compression ratio (JSON / MessagePack)
    pub fn compression_ratio(&self) -> f32 {
        if self.msgpack_bytes == 0 {
            return 1.0;
        }
        self.json_bytes as f32 / self.msgpack_bytes as f32
    }

    /// Get bytes saved by using MessagePack vs JSON
    pub fn bytes_saved(&self) -> i32 {
        self.json_bytes as i32 - self.msgpack_bytes as i32
    }

    /// Get percentage saved
    pub fn percent_saved(&self) -> f32 {
        if self.json_bytes == 0 {
            return 0.0;
        }
        (self.bytes_saved() as f32 / self.json_bytes as f32) * 100.0
    }

    /// Print human-readable size comparison
    pub fn print(&self) {
        println!("Serialization size comparison:");
        println!("  MessagePack: {} bytes", self.msgpack_bytes);
        println!("  JSON:        {} bytes", self.json_bytes);
        println!("  JSON pretty: {} bytes", self.json_pretty_bytes);
        println!("  Compression: {:.2}x", self.compression_ratio());
        println!("  Saved:       {} bytes ({:.1}%)", self.bytes_saved(), self.percent_saved());
    }
}

/// Batch serialization helper
///
/// Efficiently serialize multiple values into a single buffer.
pub struct BatchSerializer {
    format: Format,
    buffer: Vec<u8>,
}

impl BatchSerializer {
    /// Create a new batch serializer
    pub fn new(format: Format) -> Self {
        Self {
            format,
            buffer: Vec::new(),
        }
    }

    /// Add a value to the batch
    pub fn add<T>(&mut self, value: &T) -> Result<()>
    where
        T: Serialize,
    {
        let bytes = serialize(value, self.format)?;
        self.buffer.extend_from_slice(&bytes);
        Ok(())
    }

    /// Get the serialized buffer
    pub fn finish(self) -> Vec<u8> {
        self.buffer
    }

    /// Get current buffer size
    pub fn len(&self) -> usize {
        self.buffer.len()
    }

    /// Check if buffer is empty
    pub fn is_empty(&self) -> bool {
        self.buffer.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{SensorFrame, ControlPacket};

    #[test]
    fn test_msgpack_roundtrip() {
        let frame = SensorFrame::default();
        let bytes = serialize_msgpack(&frame).unwrap();
        let decoded: SensorFrame = deserialize_msgpack(&bytes).unwrap();
        assert_eq!(frame.device_id, decoded.device_id);
    }

    #[test]
    fn test_json_roundtrip() {
        let frame = SensorFrame::default();
        let json = serialize_json(&frame).unwrap();
        let decoded: SensorFrame = deserialize_json(&json).unwrap();
        assert_eq!(frame.device_id, decoded.device_id);
    }

    #[test]
    fn test_json_pretty() {
        let frame = SensorFrame::default();
        let pretty = serialize_json_pretty(&frame).unwrap();

        // Pretty JSON should have newlines
        assert!(pretty.contains('\n'));

        // Should still deserialize correctly
        let decoded: SensorFrame = deserialize_json(&pretty).unwrap();
        assert_eq!(frame.device_id, decoded.device_id);
    }

    #[test]
    fn test_format_properties() {
        assert_eq!(Format::MessagePack.extension(), "msgpack");
        assert_eq!(Format::Json.extension(), "json");
        assert_eq!(Format::MessagePack.mime_type(), "application/msgpack");
        assert!(Format::MessagePack.is_binary());
        assert!(Format::Json.is_text());
    }

    #[test]
    fn test_generic_serialize() {
        let frame = SensorFrame::default();

        let msgpack = serialize(&frame, Format::MessagePack).unwrap();
        let json = serialize(&frame, Format::Json).unwrap();

        // MessagePack should be smaller
        assert!(msgpack.len() < json.len());
    }

    #[test]
    fn test_generic_deserialize() {
        let frame = SensorFrame::default();

        let msgpack = serialize(&frame, Format::MessagePack).unwrap();
        let decoded: SensorFrame = deserialize(&msgpack, Format::MessagePack).unwrap();
        assert_eq!(frame.device_id, decoded.device_id);

        let json = serialize(&frame, Format::Json).unwrap();
        let decoded: SensorFrame = deserialize(&json, Format::Json).unwrap();
        assert_eq!(frame.device_id, decoded.device_id);
    }

    #[test]
    fn test_measure_size() {
        let frame = SensorFrame::default();
        let stats = measure_size(&frame).unwrap();

        assert!(stats.msgpack_bytes > 0);
        assert!(stats.json_bytes > 0);
        assert!(stats.json_pretty_bytes > 0);

        // Pretty JSON should be larger than compact JSON
        assert!(stats.json_pretty_bytes >= stats.json_bytes);

        // MessagePack should be smaller than JSON
        assert!(stats.msgpack_bytes < stats.json_bytes);
    }

    #[test]
    fn test_size_stats() {
        let stats = SizeStats {
            msgpack_bytes: 100,
            json_bytes: 200,
            json_pretty_bytes: 250,
        };

        assert_eq!(stats.compression_ratio(), 2.0);
        assert_eq!(stats.bytes_saved(), 100);
        assert_eq!(stats.percent_saved(), 50.0);
    }

    #[test]
    fn test_batch_serializer() {
        let mut batch = BatchSerializer::new(Format::MessagePack);
        assert!(batch.is_empty());

        let frame1 = SensorFrame::default();
        let frame2 = SensorFrame::default();

        batch.add(&frame1).unwrap();
        batch.add(&frame2).unwrap();

        assert!(!batch.is_empty());
        assert!(batch.len() > 0);

        let buffer = batch.finish();
        assert!(buffer.len() > 0);
    }

    #[test]
    fn test_control_packet_serialization() {
        let packet = ControlPacket::empty();

        let msgpack = serialize_msgpack(&packet).unwrap();
        let json = serialize_json(&packet).unwrap();

        // Verify MessagePack is more compact
        assert!(msgpack.len() < json.len());

        // Verify roundtrip
        let decoded: ControlPacket = deserialize_msgpack(&msgpack).unwrap();
        assert_eq!(packet.protocol_version, decoded.protocol_version);
    }

    #[test]
    fn test_compression_comparison() {
        let packet = ControlPacket::empty();
        let stats = measure_size(&packet).unwrap();

        println!("\nControlPacket size:");
        stats.print();

        // MessagePack should be significantly smaller
        assert!(stats.compression_ratio() > 1.0);
    }

    #[test]
    fn test_error_handling() {
        // Invalid MessagePack bytes
        let bad_bytes = vec![0xFF, 0xFF, 0xFF];
        let result: Result<SensorFrame> = deserialize_msgpack(&bad_bytes);
        assert!(result.is_err());

        // Invalid JSON
        let bad_json = "{invalid json}";
        let result: Result<SensorFrame> = deserialize_json(bad_json);
        assert!(result.is_err());
    }
}

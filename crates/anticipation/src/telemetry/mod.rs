//! Telemetry module for cc-anticipation
//!
//! Provides dashboard-friendly formatting of AnticipationPacket for streaming
//! over WebSocket. Matches the existing cc-mcs-headless message protocol.
//!
//! # Message Format
//!
//! ```json
//! {
//!   "type": "anticipation",
//!   "data": { ...AnticipationData... },
//!   "timestamp": 1735306800000
//! }
//! ```
//!
//! # Usage
//!
//! ```ignore
//! use cc_anticipation::telemetry::{AnticipationData, TelemetryMessage};
//!
//! let packet = kernel.process(&window)?;
//! let message = TelemetryMessage::from_packet(&packet);
//! let json = message.to_json()?;
//! websocket.send(json).await?;
//! ```

use serde::{Deserialize, Serialize};
use crate::AnticipationPacket;

/// Telemetry message envelope matching cc-mcs-headless WebSocket protocol.
///
/// Message types: 'sensor', 'moment', 'gesture', 'status', 'anticipation'
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TelemetryMessage {
    /// Message type discriminator
    #[serde(rename = "type")]
    pub msg_type: String,
    /// Payload data
    pub data: AnticipationData,
    /// Unix timestamp in milliseconds
    pub timestamp: u64,
}

impl TelemetryMessage {
    /// Create a telemetry message from an AnticipationPacket.
    #[must_use]
    pub fn from_packet(packet: &AnticipationPacket) -> Self {
        Self {
            msg_type: "anticipation".to_string(),
            data: AnticipationData::from_packet(packet),
            timestamp: (packet.timestamp * 1000.0) as u64,
        }
    }

    /// Serialize to JSON string.
    pub fn to_json(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string(self)
    }

    /// Serialize to pretty JSON string (for debugging).
    pub fn to_json_pretty(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string_pretty(self)
    }
}

/// Dashboard-friendly anticipation data.
///
/// Flattened and simplified for frontend consumption.
/// All fields use snake_case to match TypeScript conventions.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AnticipationData {
    // -------------------------------------------------------------------------
    // Core Scalars (0-1 range, dashboard-ready)
    // -------------------------------------------------------------------------

    /// How irreversible the current motion has become [0-1]
    pub commitment: f32,

    /// How many plausible futures remain [0-1]
    pub uncertainty: f32,

    /// Rate at which futures are collapsing (can be negative)
    pub transition_pressure: f32,

    /// Distance to balance/attractor loss [0-1]
    pub recovery_margin: f32,

    /// How locked to internal metronome [0-1]
    pub phase_stiffness: f32,

    /// Distance from recent regimes [0-1]
    pub novelty: f32,

    /// Local stationarity of dynamics [0-1]
    pub stability: f32,

    // -------------------------------------------------------------------------
    // Derived Scalars (computed for dashboard)
    // -------------------------------------------------------------------------

    /// Risk level combining multiple factors [0-1]
    /// high uncertainty + low recovery_margin + high transition_pressure = high risk
    pub risk_level: f32,

    /// Readiness for musical transition [0-1]
    /// high commitment + low uncertainty + positive pressure = ready
    pub transition_readiness: f32,

    /// Motion quality score [0-1]
    /// stability * (1 - uncertainty) * recovery_margin
    pub motion_quality: f32,

    // -------------------------------------------------------------------------
    // Regime State
    // -------------------------------------------------------------------------

    /// Regime label derived from scalars
    pub regime: String,

    /// Regime color for visualization (hex)
    pub regime_color: String,

    // -------------------------------------------------------------------------
    // Vectors (dimension-reduced for dashboard)
    // -------------------------------------------------------------------------

    /// Regime embedding summary (first 8 dims or PCA-reduced)
    pub embedding_summary: Vec<f32>,

    /// Constraint vector (~8 dims)
    pub constraint_vector: Vec<f32>,

    /// Derivative summary (~8 dims)
    pub derivative_summary: Vec<f32>,

    // -------------------------------------------------------------------------
    // Provenance
    // -------------------------------------------------------------------------

    /// Window ID from source data
    pub window_id: String,

    /// Schema version
    pub schema_version: String,
}

impl AnticipationData {
    /// Create from an AnticipationPacket.
    #[must_use]
    pub fn from_packet(packet: &AnticipationPacket) -> Self {
        // Compute derived scalars
        let risk_level = compute_risk_level(
            packet.uncertainty,
            packet.recovery_margin,
            packet.transition_pressure,
        );

        let transition_readiness = compute_transition_readiness(
            packet.commitment,
            packet.uncertainty,
            packet.transition_pressure,
        );

        let motion_quality = compute_motion_quality(
            packet.stability,
            packet.uncertainty,
            packet.recovery_margin,
        );

        // Derive regime from scalars
        let (regime, regime_color) = derive_regime(
            packet.commitment,
            packet.uncertainty,
            packet.stability,
            packet.novelty,
        );

        // Reduce embedding to first 8 dims for dashboard
        let embedding_summary: Vec<f32> = packet.regime_embedding
            .iter()
            .take(8)
            .copied()
            .collect();

        Self {
            commitment: packet.commitment,
            uncertainty: packet.uncertainty,
            transition_pressure: packet.transition_pressure,
            recovery_margin: packet.recovery_margin,
            phase_stiffness: packet.phase_stiffness,
            novelty: packet.novelty,
            stability: packet.stability,
            risk_level,
            transition_readiness,
            motion_quality,
            regime,
            regime_color,
            embedding_summary,
            constraint_vector: packet.constraint_vector.clone(),
            derivative_summary: packet.derivative_summary.clone(),
            window_id: packet.window_id.clone(),
            schema_version: packet.schema_version.clone(),
        }
    }
}

/// Compute risk level from uncertainty, recovery margin, and pressure.
fn compute_risk_level(uncertainty: f32, recovery_margin: f32, transition_pressure: f32) -> f32 {
    // High risk when: high uncertainty, low recovery, high positive pressure
    let uncertainty_factor = uncertainty;
    let recovery_factor = 1.0 - recovery_margin;
    let pressure_factor = transition_pressure.max(0.0).min(1.0);

    let risk = 0.4 * uncertainty_factor + 0.4 * recovery_factor + 0.2 * pressure_factor;
    risk.clamp(0.0, 1.0)
}

/// Compute transition readiness from commitment, uncertainty, and pressure.
fn compute_transition_readiness(commitment: f32, uncertainty: f32, transition_pressure: f32) -> f32 {
    // Ready when: high commitment, low uncertainty, positive pressure
    let commitment_factor = commitment;
    let certainty_factor = 1.0 - uncertainty;
    let pressure_factor = (transition_pressure + 1.0) / 2.0; // Map [-1,1] to [0,1]

    let readiness = 0.4 * commitment_factor + 0.4 * certainty_factor + 0.2 * pressure_factor;
    readiness.clamp(0.0, 1.0)
}

/// Compute motion quality from stability, uncertainty, and recovery margin.
fn compute_motion_quality(stability: f32, uncertainty: f32, recovery_margin: f32) -> f32 {
    let quality = stability * (1.0 - uncertainty * 0.5) * recovery_margin;
    quality.clamp(0.0, 1.0)
}

/// Derive regime label and color from scalars.
fn derive_regime(commitment: f32, uncertainty: f32, stability: f32, novelty: f32) -> (String, String) {
    // Decision tree for regime classification
    if stability < 0.3 {
        ("unstable".to_string(), "#ef4444".to_string()) // red
    } else if novelty > 0.7 {
        ("exploring".to_string(), "#8b5cf6".to_string()) // purple
    } else if uncertainty > 0.6 {
        if commitment > 0.5 {
            ("branching".to_string(), "#f59e0b".to_string()) // amber
        } else {
            ("floating".to_string(), "#06b6d4".to_string()) // cyan
        }
    } else if commitment > 0.7 {
        ("committed".to_string(), "#22c55e".to_string()) // green
    } else if uncertainty < 0.3 && stability > 0.7 {
        ("locked".to_string(), "#3b82f6".to_string()) // blue
    } else {
        ("neutral".to_string(), "#6b7280".to_string()) // gray
    }
}

/// Streaming buffer for dashboard telemetry.
///
/// Maintains a rolling window of recent packets for sparkline visualization.
#[derive(Debug, Clone)]
pub struct TelemetryBuffer {
    /// Maximum number of entries to keep
    max_size: usize,
    /// Rolling buffer of data points
    entries: Vec<TelemetryEntry>,
}

/// Single entry in telemetry buffer.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TelemetryEntry {
    /// Timestamp in seconds
    pub timestamp: f64,
    /// Core scalars for sparklines
    pub commitment: f32,
    pub uncertainty: f32,
    pub transition_pressure: f32,
    pub recovery_margin: f32,
    pub stability: f32,
    /// Derived scalars
    pub risk_level: f32,
    pub motion_quality: f32,
    /// Regime for color coding
    pub regime: String,
}

impl TelemetryBuffer {
    /// Create a new buffer with specified max size.
    #[must_use]
    pub fn new(max_size: usize) -> Self {
        Self {
            max_size,
            entries: Vec::with_capacity(max_size),
        }
    }

    /// Push a new packet into the buffer.
    pub fn push(&mut self, packet: &AnticipationPacket) {
        let data = AnticipationData::from_packet(packet);

        let entry = TelemetryEntry {
            timestamp: packet.timestamp,
            commitment: data.commitment,
            uncertainty: data.uncertainty,
            transition_pressure: data.transition_pressure,
            recovery_margin: data.recovery_margin,
            stability: data.stability,
            risk_level: data.risk_level,
            motion_quality: data.motion_quality,
            regime: data.regime,
        };

        self.entries.push(entry);

        // Trim to max size
        if self.entries.len() > self.max_size {
            self.entries.remove(0);
        }
    }

    /// Get all entries as a slice.
    #[must_use]
    pub fn entries(&self) -> &[TelemetryEntry] {
        &self.entries
    }

    /// Get entries as JSON for bulk transfer.
    pub fn to_json(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string(&self.entries)
    }

    /// Number of entries in buffer.
    #[must_use]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether buffer is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Clear all entries.
    pub fn clear(&mut self) {
        self.entries.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::SCHEMA_VERSION;

    fn create_test_packet() -> AnticipationPacket {
        AnticipationPacket {
            commitment: 0.6,
            uncertainty: 0.4,
            transition_pressure: 0.2,
            recovery_margin: 0.8,
            phase_stiffness: 0.5,
            novelty: 0.3,
            stability: 0.7,
            regime_embedding: vec![0.1; 64],
            constraint_vector: vec![0.5; 8],
            derivative_summary: vec![0.2; 8],
            debug: None,
            window_id: "test-window".to_string(),
            timestamp: 1735306800.123,
            schema_version: SCHEMA_VERSION.to_string(),
            source_identity: None,
        }
    }

    #[test]
    fn test_telemetry_message_from_packet() {
        let packet = create_test_packet();
        let message = TelemetryMessage::from_packet(&packet);

        assert_eq!(message.msg_type, "anticipation");
        assert_eq!(message.timestamp, 1735306800123);
        assert_eq!(message.data.commitment, 0.6);
        assert_eq!(message.data.uncertainty, 0.4);
    }

    #[test]
    fn test_telemetry_message_to_json() {
        let packet = create_test_packet();
        let message = TelemetryMessage::from_packet(&packet);
        let json = message.to_json().unwrap();

        assert!(json.contains("\"type\":\"anticipation\""));
        assert!(json.contains("\"commitment\":0.6"));
        assert!(json.contains("\"window_id\":\"test-window\""));
    }

    #[test]
    fn test_regime_derivation() {
        // Stable, committed
        let (regime, color) = derive_regime(0.8, 0.2, 0.8, 0.2);
        assert_eq!(regime, "committed");
        assert_eq!(color, "#22c55e");

        // Unstable
        let (regime, _) = derive_regime(0.5, 0.5, 0.2, 0.5);
        assert_eq!(regime, "unstable");

        // Exploring
        let (regime, _) = derive_regime(0.5, 0.5, 0.5, 0.8);
        assert_eq!(regime, "exploring");

        // Locked
        let (regime, _) = derive_regime(0.4, 0.2, 0.8, 0.2);
        assert_eq!(regime, "locked");
    }

    #[test]
    fn test_risk_level_computation() {
        // Low risk: low uncertainty, high recovery, low pressure
        let low_risk = compute_risk_level(0.1, 0.9, 0.1);
        assert!(low_risk < 0.2);

        // High risk: high uncertainty, low recovery, high pressure
        let high_risk = compute_risk_level(0.9, 0.1, 0.9);
        assert!(high_risk > 0.7);
    }

    #[test]
    fn test_telemetry_buffer() {
        let mut buffer = TelemetryBuffer::new(3);
        let packet = create_test_packet();

        buffer.push(&packet);
        assert_eq!(buffer.len(), 1);

        buffer.push(&packet);
        buffer.push(&packet);
        buffer.push(&packet);

        // Should trim to max size
        assert_eq!(buffer.len(), 3);

        let entries = buffer.entries();
        assert_eq!(entries.len(), 3);
        assert_eq!(entries[0].commitment, 0.6);
    }

    #[test]
    fn test_embedding_summary() {
        let mut packet = create_test_packet();
        packet.regime_embedding = (0..64).map(|i| i as f32 / 64.0).collect();

        let data = AnticipationData::from_packet(&packet);

        assert_eq!(data.embedding_summary.len(), 8);
        assert!((data.embedding_summary[0] - 0.0).abs() < 0.01);
        assert!((data.embedding_summary[7] - 7.0 / 64.0).abs() < 0.01);
    }
}

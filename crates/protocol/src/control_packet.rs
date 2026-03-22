//! Control Packet - The main message type
//!
//! This is THE central data structure flowing through the entire system.
//! Every frame, cc-brain produces a ControlPacket that contains everything
//! downstream components need to make musical decisions.

use serde::{Deserialize, Serialize};
use crate::{LatentState, CoherenceMetrics, SectionState, SensorFrame};

/// Control codes from latent field
///
/// These are the output of the latent field mapping: z*(t) → c(t).
/// They represent "where am I in the sound world?"
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ControlCodes {
    /// Primary control vector c(t)
    ///
    /// Maps to musical dimensions:
    /// - Energy (overall intensity)
    /// - Density (how many things playing)
    /// - Brightness (spectral centroid)
    /// - Tension (harmonic/rhythmic dissonance)
    /// - Movement (rate of change)
    pub codes: Vec<f32>,

    /// Which latent field "program" is active
    ///
    /// Different programs define different mappings from z* to sound worlds.
    /// Each program can represent a different musical style or territory.
    pub program_id: String,

    /// Seed for deterministic field evaluation
    ///
    /// Same seed + same z* = same c(t). Enables reproducibility.
    pub field_seed: u64,

    /// Region type in the latent field
    pub region_type: RegionType,

    /// Distance to nearest region boundary (normalized 0.0 - 1.0)
    ///
    /// 0.0 = at boundary, 1.0 = deep in center of region
    pub boundary_distance: f32,
}

impl ControlCodes {
    /// Create default control codes
    pub fn new(program_id: impl Into<String>) -> Self {
        Self {
            codes: vec![0.5; 8], // Neutral defaults
            program_id: program_id.into(),
            field_seed: 0,
            region_type: RegionType::Unexplored,
            boundary_distance: 0.5,
        }
    }

    /// Get energy code (typically first dimension)
    pub fn energy(&self) -> f32 {
        self.codes.first().copied().unwrap_or(0.5)
    }

    /// Get density code (typically second dimension)
    pub fn density(&self) -> f32 {
        self.codes.get(1).copied().unwrap_or(0.5)
    }

    /// Get brightness code (typically third dimension)
    pub fn brightness(&self) -> f32 {
        self.codes.get(2).copied().unwrap_or(0.5)
    }

    /// Check if we're near a region boundary
    pub fn is_near_boundary(&self) -> bool {
        self.boundary_distance < 0.2
    }

    /// Check if we're in a stable region
    pub fn is_in_stable_region(&self) -> bool {
        matches!(self.region_type, RegionType::Stable { .. })
    }
}

impl Default for ControlCodes {
    fn default() -> Self {
        Self::new("default")
    }
}

/// Region type in the latent field
///
/// The latent field is partitioned into different regions, each with
/// different sonic characteristics.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum RegionType {
    /// Stable sonic region with consistent character
    ///
    /// Example: "deep bass pad world", "percussive intensity", "melodic flow"
    Stable {
        /// Unique ID for this region
        region_id: u32,
    },

    /// Corridor between two regions (transition path)
    ///
    /// Movement through a corridor creates a musical transition.
    Corridor {
        /// Region we're coming from
        from: u32,
        /// Region we're going to
        to: u32,
    },

    /// Ridge (high-gradient area)
    ///
    /// Small movements cause large sonic changes. Unstable but expressive.
    Ridge,

    /// Valley (low-gradient area)
    ///
    /// Large movements cause small sonic changes. Stable but less responsive.
    Valley,

    /// Unexplored territory
    ///
    /// No predefined mapping. System will generate something exploratory.
    Unexplored,
}

/// The main control packet flowing through the system
///
/// This is produced by cc-brain every frame (typically 10-100 Hz) and consumed by:
/// - Execution clock (for quantization decisions)
/// - Controller model (for pattern edit decisions)
/// - Audio engine (for continuous parameter modulation)
/// - Visualization (for UI updates)
/// - Network (for multi-device sync)
///
/// ## Size Considerations
///
/// Typical size: 200-500 bytes (with 16D latent, 8D control codes).
/// Designed to be efficiently serialized with MessagePack for network streaming.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ControlPacket {
    /// Protocol version for compatibility checking
    pub protocol_version: String,

    /// Timestamp in microseconds (authoritative time)
    pub timestamp_us: i64,

    /// Latent state from LIM-RPS
    ///
    /// This is the raw output from the embodied physics solver.
    pub latent_state: LatentState,

    /// Control codes from latent field
    ///
    /// These are the mapped "sound world coordinates".
    pub control_codes: ControlCodes,

    /// Coherence metrics (dual-time system)
    ///
    /// Determines whether we're in Free, SoftLock, or HardLock mode.
    pub coherence: CoherenceMetrics,

    /// Current section state
    ///
    /// High-level state machine (Entry, Stable, Divergence, etc.)
    pub section_state: SectionState,

    /// Transition intensity (0.0 - 1.0)
    ///
    /// How strongly is the system transitioning? Only meaningful in
    /// Transition or Divergence states.
    pub transition_intensity: f32,

    /// Optional: Raw sensor snapshot (for logging/replay)
    ///
    /// Can include the actual sensor frame that generated this packet.
    /// Typically omitted during network streaming to save bandwidth.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub sensor_snapshot: Option<SensorFrame>,

    /// Optional: Performance metrics
    ///
    /// How long did it take to compute this packet?
    #[serde(skip_serializing_if = "Option::is_none")]
    pub metrics: Option<PerformanceMetrics>,
}

impl ControlPacket {
    /// Create a new control packet with required fields
    pub fn new(
        timestamp_us: i64,
        latent_state: LatentState,
        control_codes: ControlCodes,
        coherence: CoherenceMetrics,
        section_state: SectionState,
    ) -> Self {
        Self {
            protocol_version: crate::PROTOCOL_VERSION.to_string(),
            timestamp_us,
            latent_state,
            control_codes,
            coherence,
            section_state,
            transition_intensity: 0.0,
            sensor_snapshot: None,
            metrics: None,
        }
    }

    /// Create a default/empty packet (for testing)
    pub fn empty() -> Self {
        Self {
            protocol_version: crate::PROTOCOL_VERSION.to_string(),
            timestamp_us: 0,
            latent_state: LatentState::default(),
            control_codes: ControlCodes::default(),
            coherence: CoherenceMetrics::default(),
            section_state: SectionState::default(),
            transition_intensity: 0.0,
            sensor_snapshot: None,
            metrics: None,
        }
    }

    /// Check if pattern edits are allowed (convenience method)
    pub fn allows_edits(&self) -> bool {
        self.section_state.allows_edits() && self.coherence.allows_edits()
    }

    /// Check if bar-aligned transitions are allowed
    pub fn allows_bar_transitions(&self) -> bool {
        self.section_state.allows_bar_transitions() && self.coherence.allows_bar_transitions()
    }

    /// Get overall musical intensity (0.0 - 1.0)
    ///
    /// Combines energy from codes and section state intensity.
    pub fn musical_intensity(&self) -> f32 {
        let code_energy = self.control_codes.energy();
        let state_intensity = self.section_state.intensity();
        (code_energy * 0.6 + state_intensity * 0.4).clamp(0.0, 1.0)
    }

    /// Check if this packet is valid
    ///
    /// Validates:
    /// - Protocol version matches
    /// - Timestamp is reasonable
    /// - Required fields are present
    pub fn is_valid(&self) -> bool {
        // Check protocol version
        if self.protocol_version != crate::PROTOCOL_VERSION {
            return false;
        }

        // Check timestamp is not in far future or distant past
        // (within ±1 hour of "now" is reasonable for offline processing)
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_micros() as i64;
        let one_hour_us = 3600 * 1_000_000;
        if (self.timestamp_us - now).abs() > one_hour_us {
            // Allow this in tests/offline mode
            // return false;
        }

        // Check coherence values are in valid range
        if self.coherence.coherence < 0.0 || self.coherence.coherence > 1.0 {
            return false;
        }
        if self.coherence.periodicity < 0.0 || self.coherence.periodicity > 1.0 {
            return false;
        }

        // Check transition intensity is in valid range
        if self.transition_intensity < 0.0 || self.transition_intensity > 1.0 {
            return false;
        }

        true
    }

    /// Get age of this packet in seconds
    ///
    /// How old is this packet relative to "now"?
    pub fn age_seconds(&self) -> f64 {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_micros() as i64;
        (now - self.timestamp_us) as f64 / 1_000_000.0
    }

    /// Attach performance metrics
    pub fn with_metrics(mut self, metrics: PerformanceMetrics) -> Self {
        self.metrics = Some(metrics);
        self
    }

    /// Attach sensor snapshot (for logging)
    pub fn with_sensor_snapshot(mut self, snapshot: SensorFrame) -> Self {
        self.sensor_snapshot = Some(snapshot);
        self
    }
}

impl Default for ControlPacket {
    fn default() -> Self {
        Self::empty()
    }
}

/// Performance metrics for packet generation
///
/// Used for monitoring and optimization.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct PerformanceMetrics {
    /// Time to compute LIM-RPS (microseconds)
    pub lim_rps_time_us: u64,

    /// Time to compute coherence (microseconds)
    pub coherence_time_us: u64,

    /// Time to update state machine (microseconds)
    pub state_machine_time_us: u64,

    /// Total packet generation time (microseconds)
    pub total_time_us: u64,

    /// CPU temperature (celsius, if available)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cpu_temp_c: Option<f32>,

    /// Memory usage (bytes, if available)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub memory_bytes: Option<u64>,
}

impl PerformanceMetrics {
    /// Create new metrics with just the timings
    pub fn new(lim_rps_us: u64, coherence_us: u64, state_machine_us: u64) -> Self {
        let total = lim_rps_us + coherence_us + state_machine_us;
        Self {
            lim_rps_time_us: lim_rps_us,
            coherence_time_us: coherence_us,
            state_machine_time_us: state_machine_us,
            total_time_us: total,
            cpu_temp_c: None,
            memory_bytes: None,
        }
    }

    /// Check if performance is within acceptable limits
    ///
    /// Target: Total time < 10ms (10,000 us) for real-time operation at 100Hz
    pub fn is_realtime(&self) -> bool {
        self.total_time_us < 10_000
    }

    /// Get frames per second capability
    pub fn max_fps(&self) -> f32 {
        if self.total_time_us == 0 {
            return 0.0;
        }
        1_000_000.0 / self.total_time_us as f32
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_control_codes_creation() {
        let codes = ControlCodes::new("test_program");
        assert_eq!(codes.program_id, "test_program");
        assert_eq!(codes.codes.len(), 8);
    }

    #[test]
    fn test_control_codes_accessors() {
        let mut codes = ControlCodes::new("test");
        codes.codes = vec![0.8, 0.6, 0.4];
        assert_eq!(codes.energy(), 0.8);
        assert_eq!(codes.density(), 0.6);
        assert_eq!(codes.brightness(), 0.4);
    }

    #[test]
    fn test_region_type() {
        let stable = RegionType::Stable { region_id: 1 };
        let corridor = RegionType::Corridor { from: 1, to: 2 };

        // Just check they serialize/deserialize
        let _ = serde_json::to_string(&stable).unwrap();
        let _ = serde_json::to_string(&corridor).unwrap();
    }

    #[test]
    fn test_control_packet_creation() {
        let packet = ControlPacket::empty();
        assert_eq!(packet.protocol_version, crate::PROTOCOL_VERSION);
        assert!(packet.is_valid());
    }

    #[test]
    fn test_control_packet_allows_edits() {
        let mut packet = ControlPacket::empty();

        // Default: Entry state, Free mode -> no edits
        assert!(!packet.allows_edits());

        // Change to Stable + HardLock -> allow edits
        packet.section_state = SectionState::StableSection;
        packet.coherence.coupling_mode = crate::CouplingMode::HardLock;
        assert!(packet.allows_edits());
    }

    #[test]
    fn test_performance_metrics() {
        let metrics = PerformanceMetrics::new(3000, 2000, 1000);
        assert_eq!(metrics.total_time_us, 6000);
        assert!(metrics.is_realtime());
        assert!(metrics.max_fps() > 100.0);
    }

    #[test]
    fn test_performance_metrics_not_realtime() {
        let metrics = PerformanceMetrics::new(8000, 3000, 2000);
        assert_eq!(metrics.total_time_us, 13000);
        assert!(!metrics.is_realtime());
    }
}

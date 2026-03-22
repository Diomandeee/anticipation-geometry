//! Execution clock - quantized time for audio scheduling
//!
//! Implements the "execution time" side of the dual-time contract.
//! While latent time is continuous and body-relative, execution time
//! is quantized to bars/beats for safe, glitch-free audio scheduling.

use serde::{Deserialize, Serialize};
use crate::{ControlPacket, CouplingMode};

/// Quantization precision for musical events
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum Quantization {
    /// No quantization - execute immediately
    None,

    /// Quantize to nearest 16th note
    Sixteenth,

    /// Quantize to nearest 8th note
    Eighth,

    /// Quantize to nearest beat (quarter note)
    Beat,

    /// Quantize to nearest half bar (2 beats)
    HalfBar,

    /// Quantize to nearest full bar (4 beats)
    Bar,

    /// Quantize to nearest 2 bars (8 beats)
    TwoBars,

    /// Quantize to nearest 4 bars (16 beats)
    FourBars,
}

impl Quantization {
    /// Get quantization window size in beats
    pub fn beats(&self) -> f32 {
        match self {
            Quantization::None => 0.0,
            Quantization::Sixteenth => 0.25,
            Quantization::Eighth => 0.5,
            Quantization::Beat => 1.0,
            Quantization::HalfBar => 2.0,
            Quantization::Bar => 4.0,
            Quantization::TwoBars => 8.0,
            Quantization::FourBars => 16.0,
        }
    }

    /// Get quantization window size in seconds for given tempo
    pub fn seconds(&self, tempo_bpm: f32) -> f32 {
        if tempo_bpm <= 0.0 {
            return 0.0;
        }
        let beats_per_second = tempo_bpm / 60.0;
        self.beats() / beats_per_second
    }

    /// Get recommended quantization based on coupling mode
    ///
    /// Tighter coupling → more precise quantization allowed
    pub fn for_coupling_mode(mode: CouplingMode) -> Self {
        match mode {
            CouplingMode::Free => Quantization::None,
            CouplingMode::SoftLock => Quantization::Bar,
            CouplingMode::HardLock => Quantization::Beat,
        }
    }
}

/// Execution clock - quantized time for scheduling
///
/// This maintains the bar/beat grid that the audio engine uses to schedule events.
/// It is continuously updated based on ControlPackets from cc-brain, but maintains
/// its own phase and drift correction to ensure stable, glitch-free timing.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ExecutionClock {
    /// Current host time anchor (microseconds)
    ///
    /// This is the reference time for all calculations. Updated from ControlPacket.
    pub host_time_us: i64,

    /// Current bar number (0-indexed)
    ///
    /// None if grid is not active (Free mode).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub bar: Option<u32>,

    /// Current beat within bar (0-indexed, 0-3 for 4/4)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub beat: Option<u32>,

    /// Beat phase (0.0 - 1.0 within current beat)
    ///
    /// 0.0 = on the beat, 0.5 = halfway to next beat, etc.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub beat_phase: Option<f32>,

    /// Current tempo in BPM
    ///
    /// None if no tempo locked (Free mode).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tempo_bpm: Option<f32>,

    /// Coupling strength to latent time (0.0 - 1.0)
    ///
    /// 0.0 = free-running, no coupling
    /// 1.0 = hard-locked to detected periodicity
    pub coupling_strength: f32,

    /// Phase drift correction rate (per second)
    ///
    /// How quickly we correct phase errors to match detected tempo.
    /// Higher values = faster correction but more jitter.
    pub drift_correction_rate: f32,

    /// Time signature (beats per bar)
    ///
    /// Typically 4 for 4/4 time.
    pub beats_per_bar: u32,

    /// Accumulated phase error (for monitoring)
    ///
    /// Difference between where we think the beat is vs detected phase.
    pub phase_error: f32,

    /// Time since last tempo change (seconds)
    ///
    /// Used for hysteresis - don't change tempo too rapidly.
    pub tempo_stable_duration_s: f32,
}

impl ExecutionClock {
    /// Create a new execution clock
    pub fn new() -> Self {
        Self {
            host_time_us: 0,
            bar: None,
            beat: None,
            beat_phase: None,
            tempo_bpm: None,
            coupling_strength: 0.0,
            drift_correction_rate: 0.1, // 10% correction per second
            beats_per_bar: 4,
            phase_error: 0.0,
            tempo_stable_duration_s: 0.0,
        }
    }

    /// Update clock from a control packet
    ///
    /// This is called every time we receive a new ControlPacket from cc-brain.
    /// The clock adapts its behavior based on the coupling mode and detected tempo.
    pub fn update(&mut self, packet: &ControlPacket, dt: f32) {
        self.host_time_us = packet.timestamp_us;
        self.tempo_stable_duration_s += dt;

        match packet.coherence.coupling_mode {
            CouplingMode::Free => {
                // Release grid
                self.release_grid();
            }

            CouplingMode::SoftLock => {
                // Soft lock: allow drift
                self.coupling_strength = 0.3;
                self.update_soft_lock(packet, dt);
            }

            CouplingMode::HardLock => {
                // Hard lock: maintain phase
                self.coupling_strength = 0.9;
                self.update_hard_lock(packet, dt);
            }
        }
    }

    /// Release the grid (Free mode)
    fn release_grid(&mut self) {
        self.coupling_strength = 0.0;
        self.bar = None;
        self.beat = None;
        self.beat_phase = None;
        // Keep tempo_bpm for reference, but don't use it
    }

    /// Update in soft lock mode
    fn update_soft_lock(&mut self, packet: &ControlPacket, dt: f32) {
        if let Some(detected_tempo) = packet.coherence.tempo_bpm {
            // Update tempo with hysteresis
            if let Some(current_tempo) = self.tempo_bpm {
                // Only change tempo if difference is significant and stable
                let tempo_diff = (detected_tempo - current_tempo).abs();
                if tempo_diff > 5.0 && self.tempo_stable_duration_s > 2.0 {
                    self.tempo_bpm = Some(detected_tempo);
                    self.tempo_stable_duration_s = 0.0;
                }
            } else {
                // No current tempo, adopt detected tempo
                self.tempo_bpm = Some(detected_tempo);
                self.bar = Some(0);
                self.beat = Some(0);
                self.beat_phase = packet.coherence.phase;
                self.tempo_stable_duration_s = 0.0;
            }

            // Drift phase slowly toward detected phase
            if let (Some(detected_phase), Some(current_phase)) =
                (packet.coherence.phase, self.beat_phase)
            {
                let phase_diff = detected_phase - current_phase;
                let correction = phase_diff * self.drift_correction_rate * dt;
                let new_phase = (current_phase + correction).rem_euclid(1.0);
                self.beat_phase = Some(new_phase);
                self.phase_error = phase_diff;
            }

            // Advance beat phase based on tempo
            self.advance_phase(dt);
        }
    }

    /// Update in hard lock mode
    fn update_hard_lock(&mut self, packet: &ControlPacket, dt: f32) {
        if let Some(detected_tempo) = packet.coherence.tempo_bpm {
            // In hard lock, follow tempo more closely but still with some filtering
            if let Some(current_tempo) = self.tempo_bpm {
                // Exponential moving average
                let alpha = 0.1;
                self.tempo_bpm = Some(current_tempo * (1.0 - alpha) + detected_tempo * alpha);
            } else {
                self.tempo_bpm = Some(detected_tempo);
                self.bar = Some(0);
                self.beat = Some(0);
                self.beat_phase = packet.coherence.phase;
            }

            // Tight phase correction
            if let (Some(detected_phase), Some(current_phase)) =
                (packet.coherence.phase, self.beat_phase)
            {
                let phase_diff = detected_phase - current_phase;
                // Faster correction in hard lock
                let correction = phase_diff * (self.drift_correction_rate * 5.0) * dt;
                let new_phase = (current_phase + correction).rem_euclid(1.0);
                self.beat_phase = Some(new_phase);
                self.phase_error = phase_diff;
            }

            // Advance beat phase
            self.advance_phase(dt);
        }
    }

    /// Advance beat phase based on tempo and dt
    fn advance_phase(&mut self, dt: f32) {
        if let (Some(tempo), Some(phase)) = (self.tempo_bpm, self.beat_phase) {
            let beats_per_second = tempo / 60.0;
            let phase_delta = beats_per_second * dt;
            let new_phase = (phase + phase_delta).rem_euclid(1.0);

            // Check for beat crossing
            if new_phase < phase {
                // Crossed a beat boundary
                if let Some(beat) = self.beat {
                    let new_beat = (beat + 1) % self.beats_per_bar;
                    self.beat = Some(new_beat);

                    // Check for bar crossing
                    if new_beat == 0 {
                        if let Some(bar) = self.bar {
                            self.bar = Some(bar + 1);
                        }
                    }
                }
            }

            self.beat_phase = Some(new_phase);
        }
    }

    /// Get time until next quantization boundary in microseconds
    ///
    /// Returns None if grid is not active or quantization is None.
    pub fn time_until_quantize(&self, quant: Quantization) -> Option<i64> {
        match quant {
            Quantization::None => Some(0), // Execute immediately

            _ => {
                let tempo = self.tempo_bpm?;
                let phase = self.beat_phase?;
                let beat = self.beat?;

                let quant_beats = quant.beats();
                let beats_per_second = tempo / 60.0;
                let beat_duration_us = (1_000_000.0 / beats_per_second) as i64;

                // Calculate position within quantization window
                let current_beat_offset = beat as f32 + phase;
                let quant_position = (current_beat_offset / quant_beats).fract() * quant_beats;
                let beats_to_next = quant_beats - quant_position;

                Some((beats_to_next * beat_duration_us as f32) as i64)
            }
        }
    }

    /// Get absolute time of next quantization boundary
    pub fn next_quantize_time_us(&self, quant: Quantization) -> Option<i64> {
        let delta = self.time_until_quantize(quant)?;
        Some(self.host_time_us + delta)
    }

    /// Check if we're currently on a quantization boundary (within tolerance)
    ///
    /// Tolerance is 5% of beat duration.
    pub fn is_on_quantize(&self, quant: Quantization) -> bool {
        if let Some(time_until) = self.time_until_quantize(quant) {
            if let Some(tempo) = self.tempo_bpm {
                let beat_duration_us = (60_000_000.0 / tempo) as i64;
                let tolerance = beat_duration_us / 20; // 5%
                time_until.abs() < tolerance
            } else {
                false
            }
        } else {
            false
        }
    }

    /// Get current time in beats (fractional)
    pub fn current_beat_time(&self) -> Option<f32> {
        let beat = self.beat? as f32;
        let phase = self.beat_phase?;
        Some(beat + phase)
    }

    /// Get current time in bars (fractional)
    pub fn current_bar_time(&self) -> Option<f32> {
        let bar = self.bar? as f32;
        let beat_time = self.current_beat_time()?;
        Some(bar + beat_time / self.beats_per_bar as f32)
    }

    /// Check if clock is active (grid exists)
    pub fn is_active(&self) -> bool {
        self.tempo_bpm.is_some() && self.beat_phase.is_some()
    }

    /// Check if tempo is stable (hasn't changed recently)
    pub fn is_tempo_stable(&self, min_duration_s: f32) -> bool {
        self.is_active() && self.tempo_stable_duration_s >= min_duration_s
    }
}

impl Default for ExecutionClock {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{CoherenceMetrics, LatentState, ControlCodes, SectionState};

    fn make_test_packet(tempo_bpm: f32, phase: f32, mode: CouplingMode) -> ControlPacket {
        let mut packet = ControlPacket::empty();
        packet.coherence.tempo_bpm = Some(tempo_bpm);
        packet.coherence.phase = Some(phase);
        packet.coherence.coupling_mode = mode;
        packet.coherence.tempo_confidence = 0.8;
        packet
    }

    #[test]
    fn test_quantization_beats() {
        assert_eq!(Quantization::None.beats(), 0.0);
        assert_eq!(Quantization::Sixteenth.beats(), 0.25);
        assert_eq!(Quantization::Beat.beats(), 1.0);
        assert_eq!(Quantization::Bar.beats(), 4.0);
    }

    #[test]
    fn test_quantization_seconds() {
        let quant = Quantization::Beat;
        assert!((quant.seconds(120.0) - 0.5).abs() < 0.001); // 120 BPM = 2 beats/sec
        assert!((quant.seconds(60.0) - 1.0).abs() < 0.001);  // 60 BPM = 1 beat/sec
    }

    #[test]
    fn test_clock_creation() {
        let clock = ExecutionClock::new();
        assert!(!clock.is_active());
        assert_eq!(clock.coupling_strength, 0.0);
        assert_eq!(clock.beats_per_bar, 4);
    }

    #[test]
    fn test_clock_free_mode() {
        let mut clock = ExecutionClock::new();
        let packet = make_test_packet(120.0, 0.0, CouplingMode::Free);

        clock.update(&packet, 0.01);

        assert_eq!(clock.coupling_strength, 0.0);
        assert!(!clock.is_active());
    }

    #[test]
    fn test_clock_hard_lock() {
        let mut clock = ExecutionClock::new();
        let packet = make_test_packet(120.0, 0.0, CouplingMode::HardLock);

        clock.update(&packet, 0.01);

        assert!(clock.coupling_strength > 0.5);
        assert!(clock.is_active());
        assert_eq!(clock.tempo_bpm, Some(120.0));
    }

    #[test]
    fn test_clock_phase_advancement() {
        let mut clock = ExecutionClock::new();
        let packet = make_test_packet(120.0, 0.0, CouplingMode::HardLock);

        clock.update(&packet, 0.0); // Initialize
        assert_eq!(clock.beat, Some(0));
        assert_eq!(clock.beat_phase.unwrap(), 0.0);

        // Advance by 0.25 seconds = 0.5 beats at 120 BPM
        clock.update(&packet, 0.25);
        assert!(clock.beat_phase.unwrap() > 0.4);
        assert!(clock.beat_phase.unwrap() < 0.6);
    }

    #[test]
    fn test_time_until_quantize() {
        let mut clock = ExecutionClock::new();
        clock.tempo_bpm = Some(120.0);
        clock.beat = Some(0);
        clock.beat_phase = Some(0.5); // Halfway through beat 0

        // At 120 BPM, beat duration = 500ms = 500,000 us
        // We're at phase 0.5, so 250,000 us until next beat
        let time_until = clock.time_until_quantize(Quantization::Beat).unwrap();
        assert!((time_until - 250_000).abs() < 1000);
    }
}

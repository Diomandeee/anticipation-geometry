//! Coherence metrics and dual-time system
//!
//! Implements the **dual-time contract** from cc-protocol.md:
//! - Latent time (continuous, body-relative)
//! - Execution time (quantized, bar/beat grid)
//! - Coherence gate (dynamic coupling between them)

use serde::{Deserialize, Serialize};

/// Coupling mode between latent time and execution time
///
/// This determines how tightly the execution clock is locked to the detected
/// rhythmic structure in the body's movement.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum CouplingMode {
    /// No grid, continuous modulation only
    ///
    /// Used when:
    /// - No rhythmic structure detected
    /// - Movement is exploratory or chaotic
    /// - Periodicity is too weak to lock
    ///
    /// Behavior:
    /// - No bar/beat grid active
    /// - Pattern edits are disallowed
    /// - Only continuous parameter modulation (filters, FX, density)
    /// - Quantization windows are maximally wide
    Free,

    /// Loose grid, permissive quantization
    ///
    /// Used when:
    /// - Rhythmic structure is emerging but not stable
    /// - Periodicity is moderate (0.4 - 0.7)
    /// - Body is finding the groove
    ///
    /// Behavior:
    /// - Grid exists but has low rigidity
    /// - Phase can drift slowly to follow body
    /// - Edits allowed but only at coarse boundaries (bars, not beats)
    /// - Quantization windows are wide (~50-100ms)
    SoftLock,

    /// Strict grid, bar-aligned edits allowed
    ///
    /// Used when:
    /// - Strong rhythmic structure locked in
    /// - Periodicity is high (> 0.7)
    /// - Body and music are synchronized
    ///
    /// Behavior:
    /// - Grid is phase-locked to detected periodicity
    /// - Pattern edits can happen at bar boundaries
    /// - Quantization is strict (~10-20ms windows)
    /// - Phase corrections are smoothed to avoid clicks
    HardLock,
}

impl CouplingMode {
    /// Can we apply structural pattern edits in this mode?
    pub fn allows_edits(&self) -> bool {
        matches!(self, CouplingMode::SoftLock | CouplingMode::HardLock)
    }

    /// Can we apply bar-aligned section transitions?
    pub fn allows_bar_transitions(&self) -> bool {
        matches!(self, CouplingMode::HardLock)
    }

    /// Get quantization strictness (0.0 = loose, 1.0 = strict)
    pub fn quantization_strictness(&self) -> f32 {
        match self {
            CouplingMode::Free => 0.0,
            CouplingMode::SoftLock => 0.4,
            CouplingMode::HardLock => 0.9,
        }
    }

    /// Get coupling strength (0.0 = no coupling, 1.0 = full lock)
    pub fn coupling_strength(&self) -> f32 {
        match self {
            CouplingMode::Free => 0.0,
            CouplingMode::SoftLock => 0.3,
            CouplingMode::HardLock => 0.9,
        }
    }
}

impl Default for CouplingMode {
    fn default() -> Self {
        CouplingMode::Free
    }
}

/// Coherence metrics and periodicity detection
///
/// This is computed by the `CoherenceGate` in cc-brain and flows through
/// every `ControlPacket`. It determines the coupling mode and provides
/// estimates of tempo/phase when rhythmic structure is detected.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct CoherenceMetrics {
    /// Overall coherence level (0.0 = chaotic, 1.0 = perfectly coherent)
    ///
    /// Computed from:
    /// - Low divergence (trajectory not expanding/contracting wildly)
    /// - Low curvature variance (smooth curves, not jagged)
    /// - Stable velocity (consistent speed)
    pub coherence: f32,

    /// Strength of detected periodicity (0.0 = none, 1.0 = perfect repetition)
    ///
    /// Detected via autocorrelation of velocity magnitude in latent space.
    /// Values above 0.5 indicate usable rhythmic structure.
    pub periodicity: f32,

    /// Estimated tempo in BPM (only valid if periodicity > threshold)
    ///
    /// Extracted from peak in autocorrelation. Range is typically 60-200 BPM
    /// for danceable music. `None` if periodicity is too weak.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tempo_bpm: Option<f32>,

    /// Estimated phase within periodic cycle (0.0 - 1.0)
    ///
    /// 0.0 represents "the one" (downbeat), 0.25 is beat 2, 0.5 is beat 3, etc.
    /// `None` if tempo is not locked.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub phase: Option<f32>,

    /// Confidence in tempo/phase estimates (0.0 - 1.0)
    ///
    /// Based on:
    /// - Sharpness of autocorrelation peak
    /// - Stability of period over time
    /// - Agreement with previous tempo estimates
    pub tempo_confidence: f32,

    /// Current coupling mode (Free, SoftLock, HardLock)
    ///
    /// Determined from coherence and periodicity:
    /// - `coherence > 0.7 && periodicity > 0.7` → HardLock
    /// - `coherence > 0.4 && periodicity > 0.4` → SoftLock
    /// - Otherwise → Free
    pub coupling_mode: CouplingMode,

    /// Time spent in current coupling mode (seconds)
    ///
    /// Used for hysteresis - we don't want to flip modes too rapidly.
    /// A mode must be active for at least 0.5-1.0 seconds before transitioning.
    pub mode_duration_s: f32,
}

impl CoherenceMetrics {
    /// Create new coherence metrics with defaults
    pub fn new() -> Self {
        Self::default()
    }

    /// Check if we can apply pattern edits
    pub fn allows_edits(&self) -> bool {
        self.coupling_mode.allows_edits()
    }

    /// Check if we can do bar-aligned transitions
    pub fn allows_bar_transitions(&self) -> bool {
        self.coupling_mode.allows_bar_transitions()
    }

    /// Get beats per second (convenience for internal calculations)
    pub fn beats_per_second(&self) -> Option<f32> {
        self.tempo_bpm.map(|bpm| bpm / 60.0)
    }

    /// Check if tempo is stable enough to use
    pub fn has_stable_tempo(&self) -> bool {
        self.tempo_bpm.is_some() && self.tempo_confidence > 0.6
    }

    /// Update mode duration
    pub fn update_mode_duration(&mut self, dt: f32) {
        self.mode_duration_s += dt;
    }

    /// Reset mode duration (called when mode changes)
    pub fn reset_mode_duration(&mut self) {
        self.mode_duration_s = 0.0;
    }
}

impl Default for CoherenceMetrics {
    fn default() -> Self {
        Self {
            coherence: 0.0,
            periodicity: 0.0,
            tempo_bpm: None,
            phase: None,
            tempo_confidence: 0.0,
            coupling_mode: CouplingMode::Free,
            mode_duration_s: 0.0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_coupling_mode_allows_edits() {
        assert!(!CouplingMode::Free.allows_edits());
        assert!(CouplingMode::SoftLock.allows_edits());
        assert!(CouplingMode::HardLock.allows_edits());
    }

    #[test]
    fn test_coupling_mode_allows_bar_transitions() {
        assert!(!CouplingMode::Free.allows_bar_transitions());
        assert!(!CouplingMode::SoftLock.allows_bar_transitions());
        assert!(CouplingMode::HardLock.allows_bar_transitions());
    }

    #[test]
    fn test_coupling_mode_strengths() {
        assert_eq!(CouplingMode::Free.coupling_strength(), 0.0);
        assert!(CouplingMode::SoftLock.coupling_strength() > 0.0);
        assert!(CouplingMode::HardLock.coupling_strength() > 0.5);
    }

    #[test]
    fn test_coherence_metrics_default() {
        let metrics = CoherenceMetrics::default();
        assert_eq!(metrics.coherence, 0.0);
        assert_eq!(metrics.periodicity, 0.0);
        assert_eq!(metrics.coupling_mode, CouplingMode::Free);
        assert!(!metrics.allows_edits());
    }

    #[test]
    fn test_coherence_metrics_stable_tempo() {
        let mut metrics = CoherenceMetrics::default();
        assert!(!metrics.has_stable_tempo());

        metrics.tempo_bpm = Some(120.0);
        metrics.tempo_confidence = 0.5;
        assert!(!metrics.has_stable_tempo());

        metrics.tempo_confidence = 0.7;
        assert!(metrics.has_stable_tempo());
    }

    #[test]
    fn test_beats_per_second() {
        let mut metrics = CoherenceMetrics::default();
        assert_eq!(metrics.beats_per_second(), None);

        metrics.tempo_bpm = Some(120.0);
        assert_eq!(metrics.beats_per_second(), Some(2.0));

        metrics.tempo_bpm = Some(90.0);
        assert_eq!(metrics.beats_per_second(), Some(1.5));
    }

    #[test]
    fn test_mode_duration() {
        let mut metrics = CoherenceMetrics::default();
        assert_eq!(metrics.mode_duration_s, 0.0);

        metrics.update_mode_duration(0.1);
        assert!((metrics.mode_duration_s - 0.1).abs() < 0.001);

        metrics.reset_mode_duration();
        assert_eq!(metrics.mode_duration_s, 0.0);
    }
}

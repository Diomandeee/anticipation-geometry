//! Section state machine
//!
//! High-level state machine that tracks the performer's macro-dynamical regime.
//! This determines whether structural musical changes are allowed.

use serde::{Deserialize, Serialize};

/// High-level dynamical regime of the performance
///
/// This state machine captures the "big picture" of what's happening:
/// - Are we in a stable groove?
/// - Are we exploring/diverging?
/// - Are we transitioning between sections?
///
/// The state determines what musical operations are allowed. For example,
/// you can only apply bar-aligned pattern edits during StableSection or Resolution.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum SectionState {
    /// Initial state - no coherent structure yet
    ///
    /// Just started moving, or reset after a long pause.
    /// Waiting for enough motion history to determine state.
    Entry,

    /// Stable groove established
    ///
    /// Movement has settled into a periodic pattern with high coherence.
    /// This is the primary "performance" state where the system is locked in.
    ///
    /// Musical behavior:
    /// - Pattern edits allowed (if coupling mode permits)
    /// - Bar-aligned transitions possible
    /// - FX modulation active
    StableSection,

    /// High divergence - exploration phase
    ///
    /// Movement is chaotic, non-periodic, or rapidly changing.
    /// The system is exploring new territory.
    ///
    /// Musical behavior:
    /// - No structural edits
    /// - Continuous modulation only (filters, density, FX)
    /// - Sound follows the chaos
    Divergence,

    /// Actively transitioning between sections
    ///
    /// Detected a section boundary - moving from one stable state to another.
    /// This is a directed change, not random divergence.
    ///
    /// Musical behavior:
    /// - Section-level changes (key, tempo, energy)
    /// - Build-ups, breakdowns, drops
    /// - Large-scale FX sweeps
    Transition,

    /// Converging into new stable state
    ///
    /// After divergence or transition, settling into a new groove.
    /// Coherence is increasing, periodicity emerging.
    ///
    /// Musical behavior:
    /// - Pattern complexity can increase
    /// - Elements being added back in
    /// - Locking onto new tempo/key
    Resolution,

    /// Pause or very low energy
    ///
    /// Movement has essentially stopped. Waiting for re-activation.
    ///
    /// Musical behavior:
    /// - Ambient pads only
    /// - Gradual fade or silence
    /// - Ready to restart on motion
    Idle,
}

impl SectionState {
    /// Can we apply structural pattern edits in this state?
    pub fn allows_edits(&self) -> bool {
        matches!(
            self,
            SectionState::StableSection | SectionState::Resolution
        )
    }

    /// Can we apply bar-aligned section transitions?
    pub fn allows_bar_transitions(&self) -> bool {
        matches!(self, SectionState::StableSection)
    }

    /// Should we maintain strict timing in this state?
    pub fn requires_strict_timing(&self) -> bool {
        matches!(
            self,
            SectionState::StableSection | SectionState::Resolution
        )
    }

    /// Should we focus on continuous modulation rather than discrete events?
    pub fn prefers_continuous_modulation(&self) -> bool {
        matches!(
            self,
            SectionState::Divergence | SectionState::Transition
        )
    }

    /// Is this a "stable" state (not transient)?
    pub fn is_stable(&self) -> bool {
        matches!(
            self,
            SectionState::StableSection | SectionState::Idle
        )
    }

    /// Is this a "transient" state (temporary)?
    pub fn is_transient(&self) -> bool {
        matches!(
            self,
            SectionState::Divergence | SectionState::Transition | SectionState::Resolution
        )
    }

    /// Get a human-readable name
    pub fn name(&self) -> &'static str {
        match self {
            SectionState::Entry => "Entry",
            SectionState::StableSection => "Stable",
            SectionState::Divergence => "Divergence",
            SectionState::Transition => "Transition",
            SectionState::Resolution => "Resolution",
            SectionState::Idle => "Idle",
        }
    }

    /// Get a short name for logging
    pub fn short_name(&self) -> &'static str {
        match self {
            SectionState::Entry => "ENTRY",
            SectionState::StableSection => "STABLE",
            SectionState::Divergence => "DIVERG",
            SectionState::Transition => "TRANS",
            SectionState::Resolution => "RESOL",
            SectionState::Idle => "IDLE",
        }
    }

    /// Get typical duration hint (for UI)
    ///
    /// Returns approximate expected duration in seconds.
    /// None means indefinite (user-controlled).
    pub fn typical_duration_s(&self) -> Option<f32> {
        match self {
            SectionState::Entry => Some(2.0),
            SectionState::StableSection => None, // Can last entire performance
            SectionState::Divergence => Some(5.0),
            SectionState::Transition => Some(4.0),
            SectionState::Resolution => Some(3.0),
            SectionState::Idle => None,
        }
    }

    /// Get UI color hint (for visualization)
    ///
    /// Returns RGB in 0.0-1.0 range.
    pub fn color_rgb(&self) -> [f32; 3] {
        match self {
            SectionState::Entry => [0.5, 0.5, 0.5],        // Gray
            SectionState::StableSection => [0.2, 0.8, 0.4], // Green
            SectionState::Divergence => [0.9, 0.3, 0.3],    // Red
            SectionState::Transition => [0.9, 0.7, 0.2],    // Yellow
            SectionState::Resolution => [0.3, 0.6, 0.9],    // Blue
            SectionState::Idle => [0.3, 0.3, 0.4],          // Dark gray
        }
    }

    /// Get intensity hint (0.0 - 1.0)
    ///
    /// How "active" or "intense" is this state typically?
    pub fn intensity(&self) -> f32 {
        match self {
            SectionState::Entry => 0.3,
            SectionState::StableSection => 0.7,
            SectionState::Divergence => 0.9,
            SectionState::Transition => 0.8,
            SectionState::Resolution => 0.6,
            SectionState::Idle => 0.1,
        }
    }
}

impl Default for SectionState {
    fn default() -> Self {
        SectionState::Entry
    }
}

/// Extended section state with context
///
/// Includes additional information about the current section beyond just the enum.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct SectionStateContext {
    /// Current state
    pub state: SectionState,

    /// Time spent in current state (seconds)
    pub duration_s: f32,

    /// Transition intensity (0.0 - 1.0)
    ///
    /// How strongly is the system trying to transition?
    /// Only relevant in Transition or Divergence states.
    pub transition_intensity: f32,

    /// Stability metric (0.0 - 1.0)
    ///
    /// How stable is the current state?
    /// 1.0 = very stable, unlikely to change
    /// 0.0 = very unstable, likely to change soon
    pub stability: f32,

    /// Section ID (increments on each transition to StableSection)
    ///
    /// Useful for tracking "which part of the performance" we're in.
    pub section_id: u32,

    /// Previous state (for hysteresis)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub previous_state: Option<SectionState>,
}

impl SectionStateContext {
    /// Create new context in Entry state
    pub fn new() -> Self {
        Self {
            state: SectionState::Entry,
            duration_s: 0.0,
            transition_intensity: 0.0,
            stability: 0.5,
            section_id: 0,
            previous_state: None,
        }
    }

    /// Update duration
    pub fn update_duration(&mut self, dt: f32) {
        self.duration_s += dt;
    }

    /// Transition to a new state
    pub fn transition_to(&mut self, new_state: SectionState) {
        if new_state != self.state {
            self.previous_state = Some(self.state);
            self.state = new_state;
            self.duration_s = 0.0;

            // Increment section ID when entering StableSection
            if new_state == SectionState::StableSection {
                self.section_id += 1;
            }
        }
    }

    /// Check if state has been stable for a minimum duration
    pub fn is_stable_for(&self, min_duration_s: f32) -> bool {
        self.state.is_stable() && self.duration_s >= min_duration_s
    }

    /// Get transition intensity (only relevant in transient states)
    pub fn get_transition_intensity(&self) -> f32 {
        if self.state.is_transient() {
            self.transition_intensity
        } else {
            0.0
        }
    }
}

impl Default for SectionStateContext {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_section_state_allows_edits() {
        assert!(!SectionState::Entry.allows_edits());
        assert!(SectionState::StableSection.allows_edits());
        assert!(!SectionState::Divergence.allows_edits());
        assert!(!SectionState::Transition.allows_edits());
        assert!(SectionState::Resolution.allows_edits());
        assert!(!SectionState::Idle.allows_edits());
    }

    #[test]
    fn test_section_state_allows_bar_transitions() {
        assert!(!SectionState::Entry.allows_bar_transitions());
        assert!(SectionState::StableSection.allows_bar_transitions());
        assert!(!SectionState::Divergence.allows_bar_transitions());
    }

    #[test]
    fn test_section_state_stable() {
        assert!(!SectionState::Entry.is_stable());
        assert!(SectionState::StableSection.is_stable());
        assert!(!SectionState::Divergence.is_stable());
        assert!(SectionState::Idle.is_stable());
    }

    #[test]
    fn test_section_state_transient() {
        assert!(!SectionState::Entry.is_transient());
        assert!(!SectionState::StableSection.is_transient());
        assert!(SectionState::Divergence.is_transient());
        assert!(SectionState::Transition.is_transient());
        assert!(SectionState::Resolution.is_transient());
    }

    #[test]
    fn test_section_state_names() {
        assert_eq!(SectionState::StableSection.name(), "Stable");
        assert_eq!(SectionState::Divergence.short_name(), "DIVERG");
    }

    #[test]
    fn test_section_context_creation() {
        let ctx = SectionStateContext::new();
        assert_eq!(ctx.state, SectionState::Entry);
        assert_eq!(ctx.duration_s, 0.0);
        assert_eq!(ctx.section_id, 0);
    }

    #[test]
    fn test_section_context_transition() {
        let mut ctx = SectionStateContext::new();
        assert_eq!(ctx.state, SectionState::Entry);
        assert_eq!(ctx.section_id, 0);

        ctx.transition_to(SectionState::StableSection);
        assert_eq!(ctx.state, SectionState::StableSection);
        assert_eq!(ctx.previous_state, Some(SectionState::Entry));
        assert_eq!(ctx.section_id, 1);
        assert_eq!(ctx.duration_s, 0.0);

        ctx.update_duration(1.5);
        assert_eq!(ctx.duration_s, 1.5);

        ctx.transition_to(SectionState::Divergence);
        assert_eq!(ctx.state, SectionState::Divergence);
        assert_eq!(ctx.section_id, 1); // Doesn't increment
        assert_eq!(ctx.duration_s, 0.0); // Resets
    }

    #[test]
    fn test_section_context_stability_check() {
        let mut ctx = SectionStateContext::new();
        ctx.state = SectionState::StableSection;
        ctx.duration_s = 2.0;

        assert!(ctx.is_stable_for(1.0));
        assert!(!ctx.is_stable_for(3.0));
    }
}

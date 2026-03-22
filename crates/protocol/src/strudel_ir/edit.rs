//! Pattern edits and operations
//!
//! Commands for modifying layers and patterns during performance.

use serde::{Deserialize, Serialize};
use super::{Pattern, Layer};
use crate::clock::Quantization;

/// Target for a pattern edit
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum EditTarget {
    /// Target a specific layer by ID
    Layer(String),

    /// Target all layers matching a tag
    Tag(String),

    /// Target all layers
    All,
}

/// Operations that can be performed on patterns/layers
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum EditOperation {
    // === Layer-level operations ===

    /// Add a new layer to the arrangement
    AddLayer {
        layer: Layer,
    },

    /// Remove a layer
    RemoveLayer,

    /// Enable/disable a layer
    SetEnabled {
        enabled: bool,
    },

    /// Set layer gain
    SetGain {
        gain: f32,
    },

    /// Set layer pan
    SetPan {
        pan: f32,
    },

    // === Pattern operations ===

    /// Replace the entire pattern
    SetPattern {
        pattern: Pattern,
    },

    /// Morph between current and new pattern
    MorphPattern {
        target_pattern: Pattern,
        /// Morph amount: 0.0 = current, 1.0 = target
        amount: f32,
    },

    /// Change pattern speed
    SetSpeed {
        speed: f32,
    },

    /// Change pattern length
    SetLength {
        length: f32,
    },

    // === Effect operations ===

    /// Add an effect to the layer
    AddEffect {
        effect: super::effect::Effect,
    },

    /// Remove an effect by index
    RemoveEffect {
        index: usize,
    },

    /// Modify an effect parameter
    ModifyEffect {
        /// Effect index in the chain
        index: usize,
        /// Parameter name
        parameter: String,
        /// New value
        value: f32,
    },

    /// Clear all effects
    ClearEffects,

    // === Parameter operations ===

    /// Set a custom parameter
    SetParam {
        key: String,
        value: f32,
    },

    /// Increment a parameter
    IncrementParam {
        key: String,
        delta: f32,
    },

    // === Compound operations ===

    /// Apply multiple operations atomically
    Batch {
        operations: Vec<EditOperation>,
    },
}

/// A pattern edit command
///
/// This is what the controller model emits to change the music.
/// Each edit targets a layer (or group of layers) and specifies an operation.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct PatternEdit {
    /// What to modify
    pub target: EditTarget,

    /// What operation to perform
    pub operation: EditOperation,

    /// When to apply (quantization)
    pub quantization: Quantization,

    /// Crossfade duration in beats
    ///
    /// How long to blend from old to new state.
    /// 0.0 = instant change, 1.0 = crossfade over 1 beat.
    pub crossfade_beats: f32,

    /// Priority (higher = applied first if multiple edits on same layer)
    pub priority: u32,

    /// Optional condition: only apply if this evaluates to true
    ///
    /// Expression language TBD (could be simple predicates like "energy > 0.5")
    #[serde(skip_serializing_if = "Option::is_none")]
    pub condition: Option<String>,
}

impl PatternEdit {
    /// Create a new pattern edit targeting a layer
    pub fn new(layer_id: impl Into<String>) -> Self {
        Self {
            target: EditTarget::Layer(layer_id.into()),
            operation: EditOperation::SetEnabled { enabled: true },
            quantization: Quantization::None,
            crossfade_beats: 0.0,
            priority: 0,
            condition: None,
        }
    }

    /// Create edit targeting all layers
    pub fn all() -> Self {
        Self {
            target: EditTarget::All,
            operation: EditOperation::SetEnabled { enabled: true },
            quantization: Quantization::None,
            crossfade_beats: 0.0,
            priority: 0,
            condition: None,
        }
    }

    /// Create edit targeting layers by tag
    pub fn tag(tag: impl Into<String>) -> Self {
        Self {
            target: EditTarget::Tag(tag.into()),
            operation: EditOperation::SetEnabled { enabled: true },
            quantization: Quantization::None,
            crossfade_beats: 0.0,
            priority: 0,
            condition: None,
        }
    }

    /// Set the operation
    pub fn operation(mut self, op: EditOperation) -> Self {
        self.operation = op;
        self
    }

    /// Set quantization
    pub fn quantize(mut self, quant: Quantization) -> Self {
        self.quantization = quant;
        self
    }

    /// Set crossfade duration
    pub fn crossfade(mut self, beats: f32) -> Self {
        self.crossfade_beats = beats.max(0.0);
        self
    }

    /// Set priority
    pub fn priority(mut self, priority: u32) -> Self {
        self.priority = priority;
        self
    }

    /// Add condition
    pub fn when(mut self, condition: impl Into<String>) -> Self {
        self.condition = Some(condition.into());
        self
    }

    // === Convenience methods for common operations ===

    /// Set layer gain
    pub fn set_gain(mut self, gain: f32) -> Self {
        self.operation = EditOperation::SetGain { gain };
        self
    }

    /// Set layer pan
    pub fn set_pan(mut self, pan: f32) -> Self {
        self.operation = EditOperation::SetPan { pan };
        self
    }

    /// Enable layer
    pub fn enable(mut self) -> Self {
        self.operation = EditOperation::SetEnabled { enabled: true };
        self
    }

    /// Disable layer
    pub fn disable(mut self) -> Self {
        self.operation = EditOperation::SetEnabled { enabled: false };
        self
    }

    /// Replace pattern
    pub fn set_pattern(mut self, pattern: Pattern) -> Self {
        self.operation = EditOperation::SetPattern { pattern };
        self
    }

    /// Change pattern speed
    pub fn set_speed(mut self, speed: f32) -> Self {
        self.operation = EditOperation::SetSpeed { speed };
        self
    }

    /// Add effect
    pub fn add_effect(mut self, effect: super::effect::Effect) -> Self {
        self.operation = EditOperation::AddEffect { effect };
        self
    }

    /// Remove layer
    pub fn remove(mut self) -> Self {
        self.operation = EditOperation::RemoveLayer;
        self
    }

    /// Add new layer
    pub fn add_layer(mut self, layer: Layer) -> Self {
        self.operation = EditOperation::AddLayer { layer };
        self
    }
}

/// A sequence of pattern edits
///
/// Used for scripted changes or macro-level transitions.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct EditSequence {
    /// List of edits to apply
    pub edits: Vec<PatternEdit>,

    /// Whether to apply all edits atomically
    pub atomic: bool,

    /// Optional name/description
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
}

impl EditSequence {
    /// Create an empty sequence
    pub fn new() -> Self {
        Self {
            edits: Vec::new(),
            atomic: false,
            name: None,
        }
    }

    /// Create an atomic sequence (all-or-nothing)
    pub fn atomic() -> Self {
        Self {
            edits: Vec::new(),
            atomic: true,
            name: None,
        }
    }

    /// Add an edit to the sequence
    pub fn add(mut self, edit: PatternEdit) -> Self {
        self.edits.push(edit);
        self
    }

    /// Set name
    pub fn name(mut self, name: impl Into<String>) -> Self {
        self.name = Some(name.into());
        self
    }

    /// Get number of edits
    pub fn len(&self) -> usize {
        self.edits.len()
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.edits.is_empty()
    }
}

impl Default for EditSequence {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_edit_target() {
        let layer_target = EditTarget::Layer("kick".to_string());
        let tag_target = EditTarget::Tag("drums".to_string());
        let all_target = EditTarget::All;

        // Just verify they serialize
        let _ = serde_json::to_string(&layer_target).unwrap();
        let _ = serde_json::to_string(&tag_target).unwrap();
        let _ = serde_json::to_string(&all_target).unwrap();
    }

    #[test]
    fn test_pattern_edit_creation() {
        let edit = PatternEdit::new("kick");
        assert_eq!(edit.target, EditTarget::Layer("kick".to_string()));
        assert_eq!(edit.quantization, Quantization::None);
        assert_eq!(edit.crossfade_beats, 0.0);
    }

    #[test]
    fn test_pattern_edit_builder() {
        let edit = PatternEdit::new("bass")
            .set_gain(0.8)
            .quantize(Quantization::Bar)
            .crossfade(2.0)
            .priority(10)
            .when("energy > 0.5");

        assert_eq!(edit.quantization, Quantization::Bar);
        assert_eq!(edit.crossfade_beats, 2.0);
        assert_eq!(edit.priority, 10);
        assert_eq!(edit.condition.as_deref(), Some("energy > 0.5"));

        match edit.operation {
            EditOperation::SetGain { gain } => assert_eq!(gain, 0.8),
            _ => panic!("Wrong operation"),
        }
    }

    #[test]
    fn test_edit_all() {
        let edit = PatternEdit::all().disable();
        assert_eq!(edit.target, EditTarget::All);
        match edit.operation {
            EditOperation::SetEnabled { enabled } => assert!(!enabled),
            _ => panic!("Wrong operation"),
        }
    }

    #[test]
    fn test_edit_tag() {
        let edit = PatternEdit::tag("drums").set_gain(0.5);
        assert_eq!(edit.target, EditTarget::Tag("drums".to_string()));
    }

    #[test]
    fn test_edit_operation_types() {
        let ops = vec![
            EditOperation::SetGain { gain: 0.8 },
            EditOperation::SetPan { pan: -0.5 },
            EditOperation::SetEnabled { enabled: true },
            EditOperation::RemoveLayer,
            EditOperation::ClearEffects,
        ];

        for op in ops {
            // Just verify they serialize
            let _ = serde_json::to_string(&op).unwrap();
        }
    }

    #[test]
    fn test_edit_sequence() {
        let sequence = EditSequence::new()
            .add(PatternEdit::new("kick").set_gain(0.8))
            .add(PatternEdit::new("bass").set_gain(0.6))
            .name("drop");

        assert_eq!(sequence.len(), 2);
        assert_eq!(sequence.name.as_deref(), Some("drop"));
        assert!(!sequence.atomic);
    }

    #[test]
    fn test_atomic_sequence() {
        let sequence = EditSequence::atomic()
            .add(PatternEdit::new("kick").enable())
            .add(PatternEdit::new("bass").enable());

        assert!(sequence.atomic);
        assert_eq!(sequence.len(), 2);
    }

    #[test]
    fn test_batch_operation() {
        let batch = EditOperation::Batch {
            operations: vec![
                EditOperation::SetGain { gain: 0.8 },
                EditOperation::SetPan { pan: 0.0 },
            ],
        };

        match batch {
            EditOperation::Batch { operations } => {
                assert_eq!(operations.len(), 2);
            }
            _ => panic!("Wrong type"),
        }
    }

    #[test]
    fn test_morph_operation() {
        let target = Pattern::rest();
        let morph = EditOperation::MorphPattern {
            target_pattern: target,
            amount: 0.5,
        };

        match morph {
            EditOperation::MorphPattern { amount, .. } => {
                assert_eq!(amount, 0.5);
            }
            _ => panic!("Wrong type"),
        }
    }
}

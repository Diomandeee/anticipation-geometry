//! Strudel-IR: Symbolic music intermediate representation
//!
//! This module defines a symbolic music language inspired by Strudel/TidalCycles.
//! It serves as the bridge between body movement (ControlPacket) and audio output.
//!
//! ## Design Philosophy
//!
//! Strudel-IR is:
//! - **Platform-independent**: Same IR works on iOS, desktop, web
//! - **Serializable**: Can be stored, transmitted, and replayed
//! - **Expressive**: Captures musical intent at the right abstraction level
//! - **Type-safe**: Rust enums/structs prevent invalid music
//!
//! ## Architecture
//!
//! ```text
//! Controller Model → PatternEdit → Interpreter → Audio Events → Sound
//! ```
//!
//! The controller (transformer or rule-based) emits `PatternEdit` commands.
//! An interpreter (native AudioKit on iOS, cpal/rodio on desktop, Strudel.js on web)
//! converts these to actual audio events.
//!
//! ## Core Types
//!
//! - [`Pattern`] - A musical pattern (rhythm, melody, harmony)
//! - [`Layer`] - An instrument playing a pattern with FX
//! - [`PatternEdit`] - A command to modify a layer
//! - [`Effect`] - Audio effect (filter, delay, reverb, etc.)
//! - [`Note`] - A single musical note or event
//!
//! ## Example
//!
//! ```rust,ignore
//! use cc_protocol::strudel_ir::*;
//!
//! // Create a kick drum pattern
//! let kick_pattern = Pattern::sequence(vec![
//!     Some(Note::sample("bd", 0)),
//!     None,
//!     None,
//!     Some(Note::sample("bd", 0)),
//! ]);
//!
//! // Create a layer
//! let kick_layer = Layer::new("kick", Instrument::Sampler)
//!     .pattern(kick_pattern)
//!     .gain(0.8);
//!
//! // Apply an edit
//! let edit = PatternEdit::new("kick")
//!     .set_gain(0.6)
//!     .quantize(Quantization::Beat);
//! ```

pub mod pattern;
pub mod edit;
pub mod effect;

pub use pattern::{Pattern, PatternVariant, PatternModifier, Note, Instrument, Layer, Waveform};
pub use edit::{PatternEdit, EditOperation, EditTarget};
pub use effect::{Effect, EffectType, FilterType};

// Re-export Quantization from clock
pub use crate::clock::Quantization;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_module_imports() {
        // Just verify all types are accessible
        let _pattern = Pattern::rest();
        let _note = Note::new(60.0);
        let _edit = PatternEdit::new("test");
        let _effect = Effect::lpf(1000.0, 2.0);
    }
}

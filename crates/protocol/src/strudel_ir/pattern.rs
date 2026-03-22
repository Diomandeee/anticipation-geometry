//! Musical patterns and layers
//!
//! Core types for representing musical structure.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// A single musical note or event
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Note {
    /// MIDI note number or frequency in Hz
    ///
    /// If < 20, interpreted as MIDI note number (60 = C4).
    /// If >= 20, interpreted as frequency in Hz.
    pub pitch: f32,

    /// Velocity (0.0 - 1.0)
    ///
    /// Controls note intensity/loudness. Multiplied with layer gain.
    pub velocity: f32,

    /// Duration in beats
    ///
    /// How long the note should sound. Can be fractional.
    /// None = use default note length (typically 1 beat).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub duration: Option<f32>,

    /// Sample name (for samplers)
    ///
    /// e.g., "bd" for bass drum, "sd" for snare drum
    #[serde(skip_serializing_if = "Option::is_none")]
    pub sample: Option<String>,

    /// Sample index (for sample banks)
    ///
    /// Selects which variant of a sample to use.
    /// e.g., "bd:0", "bd:1", etc.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub sample_index: Option<u32>,
}

impl Note {
    /// Create a note from MIDI note number
    pub fn new(midi_note: f32) -> Self {
        Self {
            pitch: midi_note,
            velocity: 1.0,
            duration: None,
            sample: None,
            sample_index: None,
        }
    }

    /// Create a note from frequency in Hz
    pub fn from_freq(freq_hz: f32) -> Self {
        Self {
            pitch: freq_hz,
            velocity: 1.0,
            duration: None,
            sample: None,
            sample_index: None,
        }
    }

    /// Create a sample trigger (for drum sounds)
    pub fn sample(name: impl Into<String>, index: u32) -> Self {
        Self {
            pitch: 60.0, // Default middle C
            velocity: 1.0,
            duration: None,
            sample: Some(name.into()),
            sample_index: Some(index),
        }
    }

    /// Set velocity
    pub fn velocity(mut self, vel: f32) -> Self {
        self.velocity = vel.clamp(0.0, 1.0);
        self
    }

    /// Set duration in beats
    pub fn duration(mut self, dur: f32) -> Self {
        self.duration = Some(dur);
        self
    }

    /// Check if this is a sample trigger
    pub fn is_sample(&self) -> bool {
        self.sample.is_some()
    }

    /// Convert MIDI note to frequency in Hz
    ///
    /// The pitch field is interpreted as:
    /// - If <= 127: MIDI note number (0-127, where 60 = C4, 69 = A4)
    /// - If > 127: Already a frequency in Hz
    ///
    /// This threshold ensures all valid MIDI notes (0-127) are converted,
    /// while allowing direct frequency specification for higher values.
    pub fn to_freq(&self) -> f32 {
        if self.pitch > 127.0 {
            // Already a frequency (no MIDI note is > 127)
            self.pitch
        } else {
            // Convert MIDI to frequency: f = 440 * 2^((n-69)/12)
            440.0 * 2.0_f32.powf((self.pitch - 69.0) / 12.0)
        }
    }
}

/// Waveform for synthesis
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum Waveform {
    Sine,
    Saw,
    Square,
    Triangle,
    Noise,
    Pulse,
}

/// Instrument type
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum Instrument {
    /// Sample playback
    Sampler {
        /// Sample bank name (e.g., "808", "tr909")
        bank: Option<String>,
    },

    /// Synthesizer
    Synth {
        /// Patch/preset name
        patch: String,
        /// Waveform (if applicable)
        waveform: Option<Waveform>,
    },

    /// External MIDI device
    Midi {
        /// MIDI channel (1-16)
        channel: u8,
        /// Program change (0-127)
        program: Option<u8>,
    },

    /// Custom (user-defined)
    Custom {
        /// Type identifier
        type_id: String,
        /// Configuration parameters
        params: HashMap<String, f32>,
    },
}

impl Instrument {
    /// Create a sampler instrument
    pub fn sampler() -> Self {
        Instrument::Sampler { bank: None }
    }

    /// Create a sampler with specific bank
    pub fn sampler_bank(bank: impl Into<String>) -> Self {
        Instrument::Sampler {
            bank: Some(bank.into()),
        }
    }

    /// Create a synth instrument
    pub fn synth(patch: impl Into<String>) -> Self {
        Instrument::Synth {
            patch: patch.into(),
            waveform: None,
        }
    }

    /// Create a synth with waveform
    pub fn synth_wave(patch: impl Into<String>, wave: Waveform) -> Self {
        Instrument::Synth {
            patch: patch.into(),
            waveform: Some(wave),
        }
    }

    /// Create a MIDI instrument
    pub fn midi(channel: u8) -> Self {
        Instrument::Midi {
            channel: channel.clamp(1, 16),
            program: None,
        }
    }
}

/// Pattern variant - different ways to generate events
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum PatternVariant {
    /// Rest (silence)
    Rest,

    /// Explicit sequence of notes
    ///
    /// Each element is one step. None = rest, Some = play note.
    Sequence {
        steps: Vec<Option<Note>>,
    },

    /// Euclidean rhythm: (hits, steps)
    ///
    /// Distributes `hits` evenly across `steps` using Euclidean algorithm.
    /// Example: (3, 8) = X..X..X. pattern
    Euclidean {
        hits: u32,
        steps: u32,
        /// Optional rotation (default 0)
        rotation: Option<i32>,
    },

    /// Continuous signal (for parameter modulation)
    ///
    /// Generates a continuous waveform at given frequency.
    Signal {
        waveform: Waveform,
        /// Frequency in Hz
        frequency: f32,
    },

    /// Polymetric/polymeter pattern
    ///
    /// Multiple patterns playing at different rates simultaneously.
    Poly {
        patterns: Vec<Pattern>,
    },
}

/// A musical pattern
///
/// This is the core abstraction for rhythm, melody, and structure.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Pattern {
    /// Pattern variant
    pub variant: PatternVariant,

    /// Length in beats
    ///
    /// How long this pattern takes to complete one cycle.
    pub length: f32,

    /// Speed multiplier (1.0 = normal, 2.0 = double speed, 0.5 = half speed)
    pub speed: f32,

    /// Modifiers applied to this pattern
    #[serde(skip_serializing_if = "Vec::is_empty", default)]
    pub modifiers: Vec<PatternModifier>,
}

impl Pattern {
    /// Create a rest (silence) pattern
    pub fn rest() -> Self {
        Self {
            variant: PatternVariant::Rest,
            length: 1.0,
            speed: 1.0,
            modifiers: Vec::new(),
        }
    }

    /// Create a sequence pattern
    pub fn sequence(steps: Vec<Option<Note>>) -> Self {
        let length = steps.len() as f32;
        Self {
            variant: PatternVariant::Sequence { steps },
            length,
            speed: 1.0,
            modifiers: Vec::new(),
        }
    }

    /// Create a Euclidean rhythm pattern
    pub fn euclidean(hits: u32, steps: u32) -> Self {
        Self {
            variant: PatternVariant::Euclidean {
                hits,
                steps,
                rotation: None,
            },
            length: steps as f32,
            speed: 1.0,
            modifiers: Vec::new(),
        }
    }

    /// Create a Euclidean rhythm with rotation
    pub fn euclidean_rot(hits: u32, steps: u32, rotation: i32) -> Self {
        Self {
            variant: PatternVariant::Euclidean {
                hits,
                steps,
                rotation: Some(rotation),
            },
            length: steps as f32,
            speed: 1.0,
            modifiers: Vec::new(),
        }
    }

    /// Create a continuous signal pattern
    pub fn signal(waveform: Waveform, frequency: f32) -> Self {
        Self {
            variant: PatternVariant::Signal { waveform, frequency },
            length: 1.0,
            speed: 1.0,
            modifiers: Vec::new(),
        }
    }

    /// Create a polymetric pattern
    pub fn poly(patterns: Vec<Pattern>) -> Self {
        Self {
            variant: PatternVariant::Poly { patterns },
            length: 1.0,
            speed: 1.0,
            modifiers: Vec::new(),
        }
    }

    /// Set pattern speed
    pub fn speed(mut self, speed: f32) -> Self {
        self.speed = speed;
        self
    }

    /// Set pattern length
    pub fn length(mut self, length: f32) -> Self {
        self.length = length;
        self
    }

    /// Add a modifier
    pub fn with_modifier(mut self, modifier: PatternModifier) -> Self {
        self.modifiers.push(modifier);
        self
    }

    /// Apply fast modifier (speed up)
    pub fn fast(mut self, factor: f32) -> Self {
        self.modifiers.push(PatternModifier::Fast { factor });
        self
    }

    /// Apply slow modifier (slow down)
    pub fn slow(mut self, factor: f32) -> Self {
        self.modifiers.push(PatternModifier::Slow { factor });
        self
    }

    /// Apply reverse modifier
    pub fn rev(mut self) -> Self {
        self.modifiers.push(PatternModifier::Reverse);
        self
    }

    /// Apply degrade modifier (randomly skip events)
    pub fn degrade(mut self, probability: f32) -> Self {
        self.modifiers.push(PatternModifier::Degrade { probability });
        self
    }
}

/// Pattern modifiers (inspired by Strudel/Tidal)
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum PatternModifier {
    /// Speed up pattern
    Fast { factor: f32 },

    /// Slow down pattern
    Slow { factor: f32 },

    /// Reverse pattern
    Reverse,

    /// Add swing
    Swing { amount: f32 },

    /// Randomly skip events
    Degrade { probability: f32 },

    /// Change density (thin out or thicken)
    Density { factor: f32 },

    /// Rotate pattern
    Rotate { steps: i32 },

    /// Every nth time, apply a function
    Every {
        n: u32,
        /// Modifier to apply every nth cycle
        modifier: Box<PatternModifier>,
    },
}

/// A layer in the musical arrangement
///
/// A layer = instrument + pattern + effects + parameters
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Layer {
    /// Unique layer ID
    pub id: String,

    /// Instrument
    pub instrument: Instrument,

    /// Current pattern
    pub pattern: Pattern,

    /// Enabled state
    pub enabled: bool,

    /// Volume (0.0 - 1.0)
    pub gain: f32,

    /// Pan (-1.0 = left, 0.0 = center, 1.0 = right)
    pub pan: f32,

    /// Effects chain
    #[serde(skip_serializing_if = "Vec::is_empty", default)]
    pub effects: Vec<super::effect::Effect>,

    /// Custom parameters
    #[serde(skip_serializing_if = "HashMap::is_empty", default)]
    pub params: HashMap<String, f32>,
}

impl Layer {
    /// Create a new layer
    pub fn new(id: impl Into<String>, instrument: Instrument) -> Self {
        Self {
            id: id.into(),
            instrument,
            pattern: Pattern::rest(),
            enabled: true,
            gain: 1.0,
            pan: 0.0,
            effects: Vec::new(),
            params: HashMap::new(),
        }
    }

    /// Set pattern
    pub fn pattern(mut self, pattern: Pattern) -> Self {
        self.pattern = pattern;
        self
    }

    /// Set gain
    pub fn gain(mut self, gain: f32) -> Self {
        self.gain = gain.clamp(0.0, 2.0);
        self
    }

    /// Set pan
    pub fn pan(mut self, pan: f32) -> Self {
        self.pan = pan.clamp(-1.0, 1.0);
        self
    }

    /// Enable/disable layer
    pub fn enabled(mut self, enabled: bool) -> Self {
        self.enabled = enabled;
        self
    }

    /// Add effect
    pub fn add_effect(mut self, effect: super::effect::Effect) -> Self {
        self.effects.push(effect);
        self
    }

    /// Set custom parameter
    pub fn param(mut self, key: impl Into<String>, value: f32) -> Self {
        self.params.insert(key.into(), value);
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_note_creation() {
        let note = Note::new(60.0);
        assert_eq!(note.pitch, 60.0);
        assert_eq!(note.velocity, 1.0);
        assert!(!note.is_sample());
    }

    #[test]
    fn test_note_from_freq() {
        let note = Note::from_freq(440.0);
        assert_eq!(note.pitch, 440.0);
        assert_eq!(note.to_freq(), 440.0);
    }

    #[test]
    fn test_note_sample() {
        let note = Note::sample("bd", 0);
        assert!(note.is_sample());
        assert_eq!(note.sample.as_deref(), Some("bd"));
        assert_eq!(note.sample_index, Some(0));
    }

    #[test]
    fn test_note_to_freq() {
        let note = Note::new(69.0); // A4
        assert!((note.to_freq() - 440.0).abs() < 0.1);
    }

    #[test]
    fn test_pattern_rest() {
        let pattern = Pattern::rest();
        assert_eq!(pattern.length, 1.0);
        assert_eq!(pattern.speed, 1.0);
    }

    #[test]
    fn test_pattern_sequence() {
        let steps = vec![
            Some(Note::new(60.0)),
            None,
            Some(Note::new(64.0)),
            None,
        ];
        let pattern = Pattern::sequence(steps);
        assert_eq!(pattern.length, 4.0);
    }

    #[test]
    fn test_pattern_euclidean() {
        let pattern = Pattern::euclidean(3, 8);
        assert_eq!(pattern.length, 8.0);
        match pattern.variant {
            PatternVariant::Euclidean { hits, steps, .. } => {
                assert_eq!(hits, 3);
                assert_eq!(steps, 8);
            }
            _ => panic!("Wrong variant"),
        }
    }

    #[test]
    fn test_pattern_modifiers() {
        let pattern = Pattern::rest()
            .fast(2.0)
            .slow(0.5)
            .rev()
            .degrade(0.5);
        assert_eq!(pattern.modifiers.len(), 4);
    }

    #[test]
    fn test_instrument_creation() {
        let sampler = Instrument::sampler();
        assert!(matches!(sampler, Instrument::Sampler { .. }));

        let synth = Instrument::synth("sine");
        assert!(matches!(synth, Instrument::Synth { .. }));

        let midi = Instrument::midi(1);
        assert!(matches!(midi, Instrument::Midi { .. }));
    }

    #[test]
    fn test_layer_creation() {
        let layer = Layer::new("kick", Instrument::sampler())
            .gain(0.8)
            .pan(-0.2)
            .pattern(Pattern::euclidean(4, 16));

        assert_eq!(layer.id, "kick");
        assert_eq!(layer.gain, 0.8);
        assert_eq!(layer.pan, -0.2);
        assert!(layer.enabled);
    }

    #[test]
    fn test_layer_with_effects() {
        let layer = Layer::new("bass", Instrument::synth("saw"))
            .add_effect(super::super::Effect::lpf(1000.0, 2.0));
        assert_eq!(layer.effects.len(), 1);
    }
}

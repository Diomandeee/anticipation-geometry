//! Audio effects
//!
//! Definitions for filters, delays, reverbs, and other audio processing.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Filter type
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum FilterType {
    LowPass,
    HighPass,
    BandPass,
    Notch,
    AllPass,
    LowShelf,
    HighShelf,
    Peak,
}

/// Effect types
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum EffectType {
    /// Filter
    Filter {
        filter_type: FilterType,
        /// Cutoff frequency in Hz
        cutoff: f32,
        /// Resonance/Q (0.1 - 20.0)
        resonance: f32,
    },

    /// Reverb
    Reverb {
        /// Room size (0.0 - 1.0)
        room_size: f32,
        /// Damping (0.0 - 1.0)
        damping: f32,
        /// Wet/dry mix (0.0 - 1.0, where 1.0 = 100% wet)
        wet: f32,
    },

    /// Delay / Echo
    Delay {
        /// Delay time in seconds
        time_s: f32,
        /// Feedback amount (0.0 - 1.0)
        feedback: f32,
        /// Wet/dry mix
        wet: f32,
    },

    /// Distortion / Saturation
    Distortion {
        /// Drive amount (1.0 - 10.0+)
        drive: f32,
        /// Tone control (0.0 - 1.0)
        tone: f32,
    },

    /// Compressor
    Compressor {
        /// Threshold in dB
        threshold: f32,
        /// Ratio (1.0 - 20.0)
        ratio: f32,
        /// Attack time in milliseconds
        attack_ms: f32,
        /// Release time in milliseconds
        release_ms: f32,
    },

    /// Sidechain compression
    Sidechain {
        /// Source layer ID for sidechain signal
        source_layer: String,
        /// Sidechain amount (0.0 - 1.0)
        amount: f32,
    },

    /// Bit crusher
    BitCrush {
        /// Bit depth (1 - 16)
        bits: u32,
        /// Sample rate reduction factor (1.0 - 100.0)
        rate_reduction: f32,
    },

    /// Chorus
    Chorus {
        /// Rate in Hz
        rate: f32,
        /// Depth (0.0 - 1.0)
        depth: f32,
        /// Wet/dry mix
        wet: f32,
    },

    /// Phaser
    Phaser {
        /// Rate in Hz
        rate: f32,
        /// Depth (0.0 - 1.0)
        depth: f32,
        /// Feedback amount
        feedback: f32,
    },

    /// Auto-pan
    AutoPan {
        /// Rate in Hz
        rate: f32,
        /// Depth (0.0 - 1.0)
        depth: f32,
    },

    /// Parametric EQ (3-band)
    EQ {
        /// Low frequency gain (dB)
        low: f32,
        /// Mid frequency gain (dB)
        mid: f32,
        /// High frequency gain (dB)
        high: f32,
    },

    /// Custom effect (user-defined)
    Custom {
        /// Effect type identifier
        type_id: String,
        /// Parameters
        params: HashMap<String, f32>,
    },
}

/// An audio effect instance
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Effect {
    /// Effect type and parameters
    pub effect_type: EffectType,

    /// Enabled state
    pub enabled: bool,

    /// Bypass state (for A/B testing)
    pub bypassed: bool,

    /// Overall wet/dry mix (overrides effect-specific wet if present)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub mix: Option<f32>,
}

impl Effect {
    /// Create a new effect
    pub fn new(effect_type: EffectType) -> Self {
        Self {
            effect_type,
            enabled: true,
            bypassed: false,
            mix: None,
        }
    }

    /// Set enabled state
    pub fn enabled(mut self, enabled: bool) -> Self {
        self.enabled = enabled;
        self
    }

    /// Set bypass state
    pub fn bypass(mut self, bypassed: bool) -> Self {
        self.bypassed = bypassed;
        self
    }

    /// Set mix
    pub fn mix(mut self, mix: f32) -> Self {
        self.mix = Some(mix.clamp(0.0, 1.0));
        self
    }

    // === Convenience constructors ===

    /// Create a low-pass filter
    pub fn lpf(cutoff: f32, resonance: f32) -> Self {
        Self::new(EffectType::Filter {
            filter_type: FilterType::LowPass,
            cutoff,
            resonance,
        })
    }

    /// Create a high-pass filter
    pub fn hpf(cutoff: f32, resonance: f32) -> Self {
        Self::new(EffectType::Filter {
            filter_type: FilterType::HighPass,
            cutoff,
            resonance,
        })
    }

    /// Create a band-pass filter
    pub fn bpf(cutoff: f32, resonance: f32) -> Self {
        Self::new(EffectType::Filter {
            filter_type: FilterType::BandPass,
            cutoff,
            resonance,
        })
    }

    /// Create a reverb
    pub fn reverb(room_size: f32, damping: f32, wet: f32) -> Self {
        Self::new(EffectType::Reverb {
            room_size: room_size.clamp(0.0, 1.0),
            damping: damping.clamp(0.0, 1.0),
            wet: wet.clamp(0.0, 1.0),
        })
    }

    /// Create a simple reverb with default damping
    pub fn reverb_simple(room_size: f32, wet: f32) -> Self {
        Self::reverb(room_size, 0.5, wet)
    }

    /// Create a delay
    pub fn delay(time_s: f32, feedback: f32, wet: f32) -> Self {
        Self::new(EffectType::Delay {
            time_s,
            feedback: feedback.clamp(0.0, 1.0),
            wet: wet.clamp(0.0, 1.0),
        })
    }

    /// Create a distortion
    pub fn distort(drive: f32, tone: f32) -> Self {
        Self::new(EffectType::Distortion {
            drive: drive.max(1.0),
            tone: tone.clamp(0.0, 1.0),
        })
    }

    /// Create a compressor
    pub fn compress(threshold: f32, ratio: f32, attack_ms: f32, release_ms: f32) -> Self {
        Self::new(EffectType::Compressor {
            threshold,
            ratio: ratio.max(1.0),
            attack_ms: attack_ms.max(0.1),
            release_ms: release_ms.max(1.0),
        })
    }

    /// Create a simple compressor with default attack/release
    pub fn compress_simple(threshold: f32, ratio: f32) -> Self {
        Self::compress(threshold, ratio, 10.0, 100.0)
    }

    /// Create a sidechain compressor
    pub fn sidechain(source_layer: impl Into<String>, amount: f32) -> Self {
        Self::new(EffectType::Sidechain {
            source_layer: source_layer.into(),
            amount: amount.clamp(0.0, 1.0),
        })
    }

    /// Create a bit crusher
    pub fn bitcrush(bits: u32, rate_reduction: f32) -> Self {
        Self::new(EffectType::BitCrush {
            bits: bits.clamp(1, 16),
            rate_reduction: rate_reduction.max(1.0),
        })
    }

    /// Create a chorus
    pub fn chorus(rate: f32, depth: f32, wet: f32) -> Self {
        Self::new(EffectType::Chorus {
            rate,
            depth: depth.clamp(0.0, 1.0),
            wet: wet.clamp(0.0, 1.0),
        })
    }

    /// Create a phaser
    pub fn phaser(rate: f32, depth: f32, feedback: f32) -> Self {
        Self::new(EffectType::Phaser {
            rate,
            depth: depth.clamp(0.0, 1.0),
            feedback: feedback.clamp(0.0, 1.0),
        })
    }

    /// Create an auto-pan
    pub fn autopan(rate: f32, depth: f32) -> Self {
        Self::new(EffectType::AutoPan {
            rate,
            depth: depth.clamp(0.0, 1.0),
        })
    }

    /// Create a 3-band EQ
    pub fn eq(low: f32, mid: f32, high: f32) -> Self {
        Self::new(EffectType::EQ {
            low: low.clamp(-24.0, 24.0),
            mid: mid.clamp(-24.0, 24.0),
            high: high.clamp(-24.0, 24.0),
        })
    }
}

/// Effect preset (named collection of effects)
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct EffectPreset {
    /// Preset name
    pub name: String,

    /// Effects in order
    pub effects: Vec<Effect>,

    /// Optional description
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
}

impl EffectPreset {
    /// Create a new preset
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            effects: Vec::new(),
            description: None,
        }
    }

    /// Add an effect
    pub fn add(mut self, effect: Effect) -> Self {
        self.effects.push(effect);
        self
    }

    /// Set description
    pub fn description(mut self, desc: impl Into<String>) -> Self {
        self.description = Some(desc.into());
        self
    }

    // === Common preset constructors ===

    /// Create a "warm bass" preset
    pub fn warm_bass() -> Self {
        Self::new("Warm Bass")
            .add(Effect::lpf(800.0, 1.0))
            .add(Effect::distort(2.0, 0.6))
            .add(Effect::compress_simple(-12.0, 4.0))
            .description("Warm, saturated bass sound")
    }

    /// Create a "bright lead" preset
    pub fn bright_lead() -> Self {
        Self::new("Bright Lead")
            .add(Effect::hpf(200.0, 0.7))
            .add(Effect::chorus(0.5, 0.3, 0.4))
            .add(Effect::delay(0.375, 0.3, 0.2))
            .add(Effect::reverb_simple(0.6, 0.3))
            .description("Bright, spacious lead sound")
    }

    /// Create a "lo-fi" preset
    pub fn lofi() -> Self {
        Self::new("Lo-Fi")
            .add(Effect::bitcrush(8, 2.0))
            .add(Effect::lpf(4000.0, 0.7))
            .add(Effect::reverb_simple(0.8, 0.4))
            .description("Gritty, lo-fi character")
    }

    /// Create a "space" preset
    pub fn space() -> Self {
        Self::new("Space")
            .add(Effect::delay(0.5, 0.6, 0.5))
            .add(Effect::reverb(0.9, 0.3, 0.7))
            .add(Effect::chorus(0.2, 0.4, 0.3))
            .description("Deep, spacious ambience")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_filter_creation() {
        let lpf = Effect::lpf(1000.0, 2.0);
        match lpf.effect_type {
            EffectType::Filter { filter_type, cutoff, resonance } => {
                assert_eq!(filter_type, FilterType::LowPass);
                assert_eq!(cutoff, 1000.0);
                assert_eq!(resonance, 2.0);
            }
            _ => panic!("Wrong type"),
        }
    }

    #[test]
    fn test_reverb_creation() {
        let reverb = Effect::reverb(0.8, 0.5, 0.6);
        match reverb.effect_type {
            EffectType::Reverb { room_size, damping, wet } => {
                assert_eq!(room_size, 0.8);
                assert_eq!(damping, 0.5);
                assert_eq!(wet, 0.6);
            }
            _ => panic!("Wrong type"),
        }
    }

    #[test]
    fn test_delay_creation() {
        let delay = Effect::delay(0.5, 0.4, 0.5);
        match delay.effect_type {
            EffectType::Delay { time_s, feedback, wet } => {
                assert_eq!(time_s, 0.5);
                assert_eq!(feedback, 0.4);
                assert_eq!(wet, 0.5);
            }
            _ => panic!("Wrong type"),
        }
    }

    #[test]
    fn test_effect_builder() {
        let effect = Effect::lpf(1000.0, 2.0)
            .enabled(false)
            .bypass(true)
            .mix(0.5);

        assert!(!effect.enabled);
        assert!(effect.bypassed);
        assert_eq!(effect.mix, Some(0.5));
    }

    #[test]
    fn test_compressor() {
        let comp = Effect::compress(-12.0, 4.0, 10.0, 100.0);
        match comp.effect_type {
            EffectType::Compressor { threshold, ratio, attack_ms, release_ms } => {
                assert_eq!(threshold, -12.0);
                assert_eq!(ratio, 4.0);
                assert_eq!(attack_ms, 10.0);
                assert_eq!(release_ms, 100.0);
            }
            _ => panic!("Wrong type"),
        }
    }

    #[test]
    fn test_sidechain() {
        let sc = Effect::sidechain("kick", 0.8);
        match sc.effect_type {
            EffectType::Sidechain { source_layer, amount } => {
                assert_eq!(source_layer, "kick");
                assert_eq!(amount, 0.8);
            }
            _ => panic!("Wrong type"),
        }
    }

    #[test]
    fn test_bitcrush() {
        let bc = Effect::bitcrush(8, 2.0);
        match bc.effect_type {
            EffectType::BitCrush { bits, rate_reduction } => {
                assert_eq!(bits, 8);
                assert_eq!(rate_reduction, 2.0);
            }
            _ => panic!("Wrong type"),
        }
    }

    #[test]
    fn test_eq() {
        let eq = Effect::eq(-3.0, 0.0, 6.0);
        match eq.effect_type {
            EffectType::EQ { low, mid, high } => {
                assert_eq!(low, -3.0);
                assert_eq!(mid, 0.0);
                assert_eq!(high, 6.0);
            }
            _ => panic!("Wrong type"),
        }
    }

    #[test]
    fn test_preset_creation() {
        let preset = EffectPreset::new("Test")
            .add(Effect::lpf(1000.0, 1.0))
            .add(Effect::reverb_simple(0.5, 0.3))
            .description("Test preset");

        assert_eq!(preset.name, "Test");
        assert_eq!(preset.effects.len(), 2);
        assert_eq!(preset.description.as_deref(), Some("Test preset"));
    }

    #[test]
    fn test_warm_bass_preset() {
        let preset = EffectPreset::warm_bass();
        assert_eq!(preset.name, "Warm Bass");
        assert!(preset.effects.len() >= 2);
    }

    #[test]
    fn test_bright_lead_preset() {
        let preset = EffectPreset::bright_lead();
        assert_eq!(preset.name, "Bright Lead");
        assert!(preset.effects.len() >= 3);
    }

    #[test]
    fn test_serialization() {
        let effect = Effect::lpf(1000.0, 2.0).mix(0.5);
        let json = serde_json::to_string(&effect).unwrap();
        let deserialized: Effect = serde_json::from_str(&json).unwrap();
        assert_eq!(effect, deserialized);
    }
}

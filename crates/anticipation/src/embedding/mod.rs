//! Regime embedding computation
//!
//! This module projects fused features into a regime embedding space.
//!
//! # Versions
//!
//! - v0: Fixed random projection (deterministic, no learning)
//! - v1: Learned encoder trained on MotionPhrase library

mod projection;

pub use projection::project_to_regime_embedding;

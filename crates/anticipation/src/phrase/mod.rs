//! Motion Phrase Library
//!
//! HNSW-based retrieval infrastructure for neighbor-based uncertainty estimation.
//! Provides multi-index queries across motion, audio, joint, and transition axes.
//!
//! # Architecture
//!
//! ```text
//! Recording → PhraseExtractor → MotionPhrase records → MotionPhraseLibrary
//!                                                           ↓
//! Current regime_embedding → MultiAxisQuery → K neighbors → PriorBundle
//!                                                           ↓
//!                                     Kernel.compute_uncertainty_with_neighbors()
//! ```
//!
//! # Schema Versioning
//!
//! All phrase types are versioned for forward compatibility. Current version: 2
//! (v2 adds gesture label support)
//!
//! # Features
//!
//! This module requires the `neighbors` feature to be enabled.

mod types;
mod prior;
mod library;
mod fusion;
mod persistence;
mod extraction;

#[cfg(test)]
mod tests;

pub use types::*;
pub use prior::*;
pub use library::*;
pub use fusion::rrf_fuse;
pub use persistence::{save_library, load_library, LibraryError};
pub use extraction::{ExtractionConfig, PhraseExtractor, StubExtractor, ExtractionError};

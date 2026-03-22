//! Neighbor search for continuation dispersion (v1)
//!
//! This module integrates with rag_plusplus HNSW index
//! for computing uncertainty from continuation statistics.
//!
//! # Overview
//!
//! The anticipation kernel can optionally use a library of
//! previously observed motion phrases to improve uncertainty
//! estimates. When enabled:
//!
//! 1. Query the HNSW index with current regime_embedding
//! 2. Find K similar motion states from the library
//! 3. Examine how those states continued (their "continuations")
//! 4. High divergence in continuations = high uncertainty
//!
//! # Usage
//!
//! ```ignore
//! use cc_anticipation::neighbors::{MotionPhraseIndex, MotionPhraseIndexConfig};
//!
//! let config = MotionPhraseIndexConfig::default();
//! let mut index = MotionPhraseIndex::new(config);
//!
//! // Index phrases from a motion library
//! index.add_phrase(phrase_id, &embedding, continuation)?;
//!
//! // Query for continuation dispersion
//! let dispersion = compute_continuation_dispersion(&index, &query_embedding, 8);
//! let uncertainty = dispersion_to_uncertainty(&dispersion);
//! ```
//!
//! Requires the "neighbors" feature.

#[cfg(feature = "neighbors")]
mod dispersion;

#[cfg(feature = "neighbors")]
pub use dispersion::{
    MotionPhraseIndex,
    MotionPhraseIndexConfig,
    ContinuationData,
    DispersionResult,
    compute_continuation_dispersion,
    dispersion_to_uncertainty,
};

//! cc-anticipation: Anticipation kernel for Computational Choreography
//!
//! This crate converts stabilized motion state into actionable anticipatory signals.
//! It answers one question: "What futures are cheap vs expensive given what's happening now?"
//!
//! # Architecture Position
//!
//! ```text
//! Sensors → cc-mcs-headless → cc-window-aligner → [cc-anticipation] → Conductor
//!                                    ↓                    ↓
//!                             (MotionWindow)      (AnticipationPacket)
//! ```
//!
//! # Core Concepts
//!
//! - **MotionWindow**: Fixed-length ring of aligned frames at canonical fps (input)
//! - **AnticipationPacket**: Scalars + regime embedding + debug traces (output)
//! - **Commitment**: How irreversible the current motion has become
//! - **Uncertainty**: How many plausible futures remain
//! - **Transition Pressure**: Rate at which futures are collapsing
//!
//! # Example
//!
//! ```ignore
//! use cc_anticipation::{AnticipationKernel, AnticipationConfig, MotionWindow};
//!
//! let config = AnticipationConfig::default();
//! let mut kernel = AnticipationKernel::new(config);
//!
//! // Process a motion window (from cc-window-aligner)
//! let packet = kernel.process(&window)?;
//!
//! // Use scalars for control
//! if packet.transition_pressure > 0.7 {
//!     // Prepare for a change
//! }
//! ```
//!
//! # Features
//!
//! - `python`: Enable PyO3 bindings for Python integration
//! - `neighbors`: Enable HNSW neighbor search via rag_plusplus

#![warn(missing_docs)]
#![allow(clippy::needless_return)]

pub mod types;
pub mod config;
pub mod kernel;
pub mod features;
pub mod embedding;
pub mod constraints;
pub mod scalars;

#[cfg(feature = "neighbors")]
pub mod neighbors;

#[cfg(feature = "neighbors")]
pub mod phrase;

pub mod replay;
pub mod telemetry;

#[cfg(feature = "python")]
pub mod bindings;

// Re-exports for convenience
pub use types::{
    Quat, Vec3,
    SkeletonFrame, LatentFrame,
    MotionWindow, AnticipationPacket, DebugTrace,
    SCHEMA_VERSION,
};
pub use config::AnticipationConfig;
pub use kernel::AnticipationKernel;

// Feature computation re-exports
pub use features::{KinematicFeatures, LatentFeatures};
pub use embedding::project_to_regime_embedding;
pub use constraints::compute_constraint_vector;
pub use scalars::{
    compute_commitment, compute_uncertainty,
    compute_transition_pressure, compute_recovery_margin,
    compute_phase_stiffness, compute_novelty, compute_stability,
};

// Replay harness re-exports
pub use replay::{
    TraceRecorder, ReplayHarness, TraceFile, TraceEntry,
    ValidationSummary, TraceError,
};

// Telemetry re-exports
pub use telemetry::{
    TelemetryMessage, AnticipationData, TelemetryBuffer, TelemetryEntry,
};

// Phrase library re-exports (requires "neighbors" feature)
#[cfg(feature = "neighbors")]
pub use phrase::{
    MotionPhrase, PhraseEmbeddings, PhaseModel, OutcomeMetadata, PhraseProvenance,
    EmbeddingAxis, PHRASE_SCHEMA_VERSION,
    PriorBundle, AxisScores, WelfordAccumulator,
    MotionPhraseLibrary, LibraryConfig, LibraryMetadata, MultiAxisQuery, AxisWeights,
    ExtractionConfig, PhraseExtractor, StubExtractor, ExtractionError, LibraryError,
};

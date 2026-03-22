//! Motion phrase record types
//!
//! Frozen, versioned schema for motion phrases used in neighbor-based uncertainty.

use serde::{Deserialize, Serialize};

/// Current schema version for forward compatibility
/// NOTE: Bumped to 2 for gesture label support
pub const PHRASE_SCHEMA_VERSION: u32 = 2;

/// Default motion embedding dimension (regime embedding)
pub const DEFAULT_MOTION_DIM: usize = 64;
/// Default audio embedding dimension
pub const DEFAULT_AUDIO_DIM: usize = 32;
/// Default joint angle embedding dimension (14 joints * 3 Euler angles)
pub const DEFAULT_JOINT_DIM: usize = 42;
/// Default transition embedding dimension
pub const DEFAULT_TRANSITION_DIM: usize = 32;

/// Complete motion phrase record (frozen schema)
///
/// A phrase represents a temporal segment of motion with associated embeddings
/// and outcome metadata for neighbor-based retrieval.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MotionPhrase {
    /// Unique phrase identifier (UUID v4)
    pub phrase_id: String,
    /// Schema version for migration support
    pub schema_version: u32,

    // === Temporal bounds ===
    /// Start time in recording (seconds)
    pub t_start: f64,
    /// End time in recording (seconds)
    pub t_end: f64,
    /// Duration in seconds
    pub duration: f32,

    // === Phase model ===
    /// Tempo and beat alignment information
    pub phase_model: PhaseModel,

    // === Multi-axis embeddings ===
    /// Embedding vectors for different retrieval axes
    pub embeddings: PhraseEmbeddings,

    // === Outcome metadata ===
    /// What happened after this phrase (for continuation prediction)
    pub outcome: OutcomeMetadata,

    // === Provenance ===
    /// Recording and extraction metadata
    pub provenance: PhraseProvenance,

    // === Gesture Labels (v2) ===
    /// Optional gesture annotations for this phrase
    #[serde(default)]
    pub gesture_labels: Vec<GestureAnnotation>,
}

impl MotionPhrase {
    /// Create a new motion phrase with required fields
    pub fn new(
        phrase_id: String,
        t_start: f64,
        t_end: f64,
        motion_embedding: Vec<f32>,
    ) -> Self {
        Self {
            phrase_id,
            schema_version: PHRASE_SCHEMA_VERSION,
            t_start,
            t_end,
            duration: (t_end - t_start) as f32,
            phase_model: PhaseModel::default(),
            embeddings: PhraseEmbeddings {
                motion: motion_embedding,
                audio: None,
                joint: None,
                transition: None,
            },
            outcome: OutcomeMetadata::default(),
            provenance: PhraseProvenance::default(),
            gesture_labels: Vec::new(),
        }
    }

    /// Add a gesture annotation to this phrase
    pub fn add_gesture(&mut self, annotation: GestureAnnotation) {
        self.gesture_labels.push(annotation);
    }

    /// Check if this phrase has any gesture labels
    pub fn has_gestures(&self) -> bool {
        !self.gesture_labels.is_empty()
    }

    /// Get all gesture label IDs
    pub fn gesture_ids(&self) -> Vec<u32> {
        let mut ids: Vec<u32> = self.gesture_labels.iter().map(|a| a.label.id).collect();
        ids.sort_unstable();
        ids.dedup();
        ids
    }

    /// Validate the phrase schema and embedding dimensions
    pub fn validate(&self) -> Result<(), PhraseValidationError> {
        if self.schema_version > PHRASE_SCHEMA_VERSION {
            return Err(PhraseValidationError::UnsupportedVersion(self.schema_version));
        }
        if self.embeddings.motion.is_empty() {
            return Err(PhraseValidationError::EmptyMotionEmbedding);
        }
        if self.duration <= 0.0 {
            return Err(PhraseValidationError::InvalidDuration(self.duration));
        }
        Ok(())
    }
}

/// Phase/tempo model for the phrase
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct PhaseModel {
    /// Detected tempo in BPM
    pub tempo_bpm: f32,
    /// Beat grid alignment offset (seconds)
    pub beat_offset: f32,
    /// Number of beats in phrase
    pub beat_count: u32,
    /// Phase at start [0, 1]
    pub start_phase: f32,
    /// Phase at end [0, 1]
    pub end_phase: f32,
    /// Tempo confidence [0, 1]
    pub tempo_confidence: f32,
}

impl PhaseModel {
    /// Create a phase model with detected tempo
    pub fn with_tempo(tempo_bpm: f32, confidence: f32) -> Self {
        Self {
            tempo_bpm,
            tempo_confidence: confidence,
            ..Default::default()
        }
    }
}

/// Multi-axis embedding vectors
///
/// Each axis represents a different retrieval dimension:
/// - motion: Overall motion regime (primary)
/// - audio: Audio features aligned to motion
/// - joint: Joint angle configurations
/// - transition: Transition behavior patterns
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct PhraseEmbeddings {
    /// Motion regime embedding (64D) - primary axis, required
    pub motion: Vec<f32>,
    /// Audio feature embedding (32D) - optional
    pub audio: Option<Vec<f32>>,
    /// Joint angle summary (42D) - optional (14 joints * 3 Euler angles)
    pub joint: Option<Vec<f32>>,
    /// Transition embedding (32D) - for continuation prediction, optional
    pub transition: Option<Vec<f32>>,
}

impl PhraseEmbeddings {
    /// Create embeddings with motion only
    pub fn motion_only(motion: Vec<f32>) -> Self {
        Self {
            motion,
            audio: None,
            joint: None,
            transition: None,
        }
    }

    /// Get the embedding for a specific axis
    pub fn get_axis(&self, axis: EmbeddingAxis) -> Option<&[f32]> {
        match axis {
            EmbeddingAxis::Motion => Some(&self.motion),
            EmbeddingAxis::Audio => self.audio.as_deref(),
            EmbeddingAxis::Joint => self.joint.as_deref(),
            EmbeddingAxis::Transition => self.transition.as_deref(),
        }
    }
}

/// Embedding axis identifiers
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum EmbeddingAxis {
    /// Motion regime embedding
    Motion,
    /// Audio feature embedding
    Audio,
    /// Joint angle summary
    Joint,
    /// Transition behavior embedding
    Transition,
}

impl EmbeddingAxis {
    /// Get the default dimension for this axis
    pub fn default_dim(&self) -> usize {
        match self {
            EmbeddingAxis::Motion => DEFAULT_MOTION_DIM,
            EmbeddingAxis::Audio => DEFAULT_AUDIO_DIM,
            EmbeddingAxis::Joint => DEFAULT_JOINT_DIM,
            EmbeddingAxis::Transition => DEFAULT_TRANSITION_DIM,
        }
    }
}

/// Outcome statistics for this phrase
///
/// Captures what happened after this phrase ended, used for
/// continuation prediction and dispersion computation.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct OutcomeMetadata {
    /// What regime came next
    pub next_regime: String,
    /// Next phrase's motion embedding (for continuation dispersion)
    pub next_embedding: Vec<f32>,
    /// Delta commitment from this phrase to next
    pub delta_commitment: f32,
    /// Delta uncertainty from this phrase to next
    pub delta_uncertainty: f32,
    /// Transition success score [0, 1]
    pub transition_success: f32,
    /// Was this phrase part of a performance? (vs practice)
    pub is_performance: bool,
}

impl OutcomeMetadata {
    /// Create outcome with next regime information
    pub fn with_next(next_regime: String, next_embedding: Vec<f32>) -> Self {
        Self {
            next_regime,
            next_embedding,
            ..Default::default()
        }
    }
}

/// Recording provenance metadata
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct PhraseProvenance {
    /// Source recording ID
    pub recording_id: String,
    /// Performer ID (anonymized)
    pub performer_id: String,
    /// Extraction timestamp (Unix epoch millis)
    pub extracted_at: u64,
    /// Extraction algorithm version
    pub extractor_version: String,
}

impl PhraseProvenance {
    /// Create provenance with recording ID
    pub fn from_recording(recording_id: String, performer_id: String) -> Self {
        Self {
            recording_id,
            performer_id,
            extracted_at: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_millis() as u64)
                .unwrap_or(0),
            extractor_version: env!("CARGO_PKG_VERSION").to_string(),
        }
    }
}

/// Phrase validation errors
#[derive(Debug, Clone)]
pub enum PhraseValidationError {
    /// Schema version is newer than supported
    UnsupportedVersion(u32),
    /// Motion embedding is empty
    EmptyMotionEmbedding,
    /// Duration is invalid
    InvalidDuration(f32),
}

impl std::fmt::Display for PhraseValidationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::UnsupportedVersion(v) => {
                write!(f, "Unsupported schema version: {} (max: {})", v, PHRASE_SCHEMA_VERSION)
            }
            Self::EmptyMotionEmbedding => write!(f, "Motion embedding is empty"),
            Self::InvalidDuration(d) => write!(f, "Invalid duration: {}", d),
        }
    }
}

impl std::error::Error for PhraseValidationError {}

// =============================================================================
// GESTURE LABELS (v2)
// =============================================================================

/// Category of gesture.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum GestureCategory {
    /// Full-body gestures from Mokopi suit.
    FullBody,
    /// Hand gestures from iPhone/Watch sensors.
    Hand,
    /// Combined full-body + hand gesture.
    Hybrid,
}

impl Default for GestureCategory {
    fn default() -> Self {
        Self::FullBody
    }
}

/// A gesture label with unique identifier.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct GestureLabel {
    /// Unique gesture identifier.
    pub id: u32,
    /// Human-readable gesture name.
    pub name: String,
    /// Gesture category.
    pub category: GestureCategory,
}

impl GestureLabel {
    /// Creates a new gesture label.
    pub fn new(id: u32, name: impl Into<String>, category: GestureCategory) -> Self {
        Self {
            id,
            name: name.into(),
            category,
        }
    }

    /// Creates a full-body gesture label.
    pub fn full_body(id: u32, name: impl Into<String>) -> Self {
        Self::new(id, name, GestureCategory::FullBody)
    }

    /// Creates a hand gesture label.
    pub fn hand(id: u32, name: impl Into<String>) -> Self {
        Self::new(id, name, GestureCategory::Hand)
    }
}

/// A gesture annotation within a phrase.
///
/// Marks a specific time range within a phrase as containing a gesture.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GestureAnnotation {
    /// The gesture label.
    pub label: GestureLabel,
    /// Start time relative to phrase start (seconds).
    pub t_start: f32,
    /// End time relative to phrase start (seconds).
    pub t_end: f32,
    /// Confidence of the annotation (0-1).
    pub confidence: f32,
    /// Source of the annotation (manual, automatic, etc.).
    #[serde(default)]
    pub source: AnnotationSource,
}

impl GestureAnnotation {
    /// Creates a new gesture annotation.
    pub fn new(label: GestureLabel, t_start: f32, t_end: f32) -> Self {
        Self {
            label,
            t_start,
            t_end,
            confidence: 1.0,
            source: AnnotationSource::Manual,
        }
    }

    /// Duration of the annotation in seconds.
    pub fn duration(&self) -> f32 {
        self.t_end - self.t_start
    }

    /// Sets the confidence.
    pub fn with_confidence(mut self, confidence: f32) -> Self {
        self.confidence = confidence;
        self
    }

    /// Sets the source.
    pub fn with_source(mut self, source: AnnotationSource) -> Self {
        self.source = source;
        self
    }
}

/// Source of a gesture annotation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum AnnotationSource {
    /// Manual annotation by a user.
    #[default]
    Manual,
    /// Automatic annotation by a classifier.
    Automatic,
    /// Imported from external system.
    Imported,
    /// Corrected from automatic to manual.
    Corrected,
}

/// A motion phrase with labeled gestures.
///
/// This is a convenience wrapper that provides additional methods
/// for working with gesture-annotated phrases.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct LabeledMotionPhrase {
    /// The underlying motion phrase.
    pub phrase: MotionPhrase,
}

impl LabeledMotionPhrase {
    /// Creates a new labeled motion phrase.
    pub fn new(phrase: MotionPhrase) -> Self {
        Self { phrase }
    }

    /// Wraps an existing phrase.
    pub fn from_phrase(phrase: MotionPhrase) -> Self {
        Self { phrase }
    }

    /// Adds a gesture annotation.
    pub fn add_annotation(&mut self, annotation: GestureAnnotation) {
        self.phrase.gesture_labels.push(annotation);
    }

    /// Adds a gesture label at a specific time range.
    pub fn add_label(&mut self, label: GestureLabel, t_start: f32, t_end: f32) {
        self.add_annotation(GestureAnnotation::new(label, t_start, t_end));
    }

    /// Returns all gesture labels in this phrase.
    pub fn labels(&self) -> Vec<&GestureLabel> {
        self.phrase.gesture_labels.iter().map(|a| &a.label).collect()
    }

    /// Returns unique gesture label IDs in this phrase.
    pub fn label_ids(&self) -> Vec<u32> {
        let mut ids: Vec<u32> = self.phrase.gesture_labels.iter().map(|a| a.label.id).collect();
        ids.sort_unstable();
        ids.dedup();
        ids
    }

    /// Checks if this phrase contains a specific gesture.
    pub fn contains_gesture(&self, label_id: u32) -> bool {
        self.phrase.gesture_labels.iter().any(|a| a.label.id == label_id)
    }

    /// Returns annotations for a specific gesture.
    pub fn annotations_for(&self, label_id: u32) -> Vec<&GestureAnnotation> {
        self.phrase
            .gesture_labels
            .iter()
            .filter(|a| a.label.id == label_id)
            .collect()
    }

    /// Returns the dominant gesture (longest duration).
    pub fn dominant_gesture(&self) -> Option<&GestureLabel> {
        self.phrase
            .gesture_labels
            .iter()
            .max_by(|a, b| a.duration().partial_cmp(&b.duration()).unwrap_or(std::cmp::Ordering::Equal))
            .map(|a| &a.label)
    }

    /// Returns the phrase ID.
    pub fn phrase_id(&self) -> &str {
        &self.phrase.phrase_id
    }

    /// Returns the motion embedding.
    pub fn motion_embedding(&self) -> &[f32] {
        &self.phrase.embeddings.motion
    }
}

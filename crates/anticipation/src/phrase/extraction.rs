//! Phrase extraction pipeline hooks (stub for future implementation)
//!
//! These interfaces will be implemented when actual motion data is available.
//! For now, they provide the type signatures and placeholder implementations.

use std::path::Path;

use super::types::MotionPhrase;
use crate::types::MotionWindow;

/// Configuration for phrase extraction
#[derive(Clone, Debug)]
pub struct ExtractionConfig {
    /// Minimum phrase duration in seconds
    pub min_duration: f32,
    /// Maximum phrase duration in seconds
    pub max_duration: f32,
    /// Overlap ratio between adjacent phrases (0.0 to 0.9)
    pub overlap_ratio: f32,
    /// Tempo detection algorithm
    pub tempo_algorithm: TempoAlgorithm,
    /// Whether to compute audio embeddings
    pub compute_audio: bool,
    /// Whether to compute joint embeddings
    pub compute_joints: bool,
    /// Whether to compute transition embeddings
    pub compute_transitions: bool,
}

impl Default for ExtractionConfig {
    fn default() -> Self {
        Self {
            min_duration: 1.0,
            max_duration: 8.0,
            overlap_ratio: 0.5,
            tempo_algorithm: TempoAlgorithm::Autocorrelation,
            compute_audio: false,
            compute_joints: false,
            compute_transitions: true,
        }
    }
}

impl ExtractionConfig {
    /// Create config for fast extraction (minimal embeddings)
    pub fn fast() -> Self {
        Self {
            compute_audio: false,
            compute_joints: false,
            compute_transitions: false,
            ..Default::default()
        }
    }

    /// Create config for full extraction (all embeddings)
    pub fn full() -> Self {
        Self {
            compute_audio: true,
            compute_joints: true,
            compute_transitions: true,
            ..Default::default()
        }
    }
}

/// Tempo detection algorithm options
#[derive(Clone, Debug, Default)]
pub enum TempoAlgorithm {
    /// Autocorrelation-based tempo detection
    #[default]
    Autocorrelation,
    /// Beat tracker (requires audio)
    BeatTracker,
    /// Use externally provided tempo
    External(f32),
    /// Infer from motion periodicity
    MotionPeriodicity,
}

/// Extraction progress callback
pub type ProgressCallback = Box<dyn Fn(ExtractionProgress) + Send + Sync>;

/// Extraction progress information
#[derive(Clone, Debug)]
pub struct ExtractionProgress {
    /// Current window index
    pub current: usize,
    /// Total windows to process
    pub total: usize,
    /// Phrases extracted so far
    pub phrases_extracted: usize,
    /// Estimated completion fraction (0.0 to 1.0)
    pub progress: f32,
}

/// Trait for phrase extractors
///
/// Implement this trait to create custom extraction algorithms.
/// The default implementation is `StubExtractor` which returns empty results.
pub trait PhraseExtractor: Send + Sync {
    /// Extract phrases from a sequence of motion windows
    ///
    /// # Arguments
    ///
    /// * `windows` - Sequence of motion windows to process
    /// * `session_id` - Session identifier for provenance
    ///
    /// # Returns
    ///
    /// Vector of extracted motion phrases
    fn extract(&self, windows: &[MotionWindow], session_id: &str) -> Vec<MotionPhrase>;

    /// Extract phrases from a recording file
    ///
    /// # Arguments
    ///
    /// * `path` - Path to the recording file (e.g., .msgpack, .json)
    ///
    /// # Returns
    ///
    /// Vector of extracted motion phrases, or error
    fn extract_from_file(&self, path: &Path) -> Result<Vec<MotionPhrase>, ExtractionError>;

    /// Extract with progress callback
    fn extract_with_progress(
        &self,
        windows: &[MotionWindow],
        session_id: &str,
        _callback: ProgressCallback,
    ) -> Vec<MotionPhrase> {
        // Default: ignore progress callback
        self.extract(windows, session_id)
    }

    /// Get the configuration for this extractor
    fn config(&self) -> &ExtractionConfig;
}

/// Placeholder extractor that returns empty results
///
/// Use this when you need a PhraseExtractor but don't have real
/// extraction logic yet.
#[derive(Clone, Debug, Default)]
pub struct StubExtractor {
    config: ExtractionConfig,
}

impl StubExtractor {
    /// Create a new stub extractor
    pub fn new() -> Self {
        Self::default()
    }

    /// Create with custom config
    pub fn with_config(config: ExtractionConfig) -> Self {
        Self { config }
    }
}

impl PhraseExtractor for StubExtractor {
    fn extract(&self, _windows: &[MotionWindow], _session_id: &str) -> Vec<MotionPhrase> {
        // Stub: return empty
        Vec::new()
    }

    fn extract_from_file(&self, _path: &Path) -> Result<Vec<MotionPhrase>, ExtractionError> {
        // Stub: return empty
        Ok(Vec::new())
    }

    fn config(&self) -> &ExtractionConfig {
        &self.config
    }
}

/// Extraction errors
#[derive(Debug)]
pub enum ExtractionError {
    /// I/O error reading file
    IoError(std::io::Error),
    /// Parse error in recording format
    ParseError(String),
    /// Insufficient data for extraction
    InsufficientData {
        /// Windows provided
        provided: usize,
        /// Minimum required
        required: usize,
    },
    /// Invalid configuration
    InvalidConfig(String),
}

impl std::fmt::Display for ExtractionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::IoError(e) => write!(f, "I/O error: {}", e),
            Self::ParseError(e) => write!(f, "Parse error: {}", e),
            Self::InsufficientData { provided, required } => {
                write!(f, "Insufficient data: {} windows provided, {} required", provided, required)
            }
            Self::InvalidConfig(e) => write!(f, "Invalid config: {}", e),
        }
    }
}

impl std::error::Error for ExtractionError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::IoError(e) => Some(e),
            _ => None,
        }
    }
}

impl From<std::io::Error> for ExtractionError {
    fn from(e: std::io::Error) -> Self {
        Self::IoError(e)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_stub_extractor_returns_empty() {
        let extractor = StubExtractor::new();
        let phrases = extractor.extract(&[], "test");
        assert!(phrases.is_empty());
    }

    #[test]
    fn test_stub_extractor_file_returns_empty() {
        let extractor = StubExtractor::new();
        let result = extractor.extract_from_file(Path::new("/nonexistent"));
        assert!(result.unwrap().is_empty());
    }

    #[test]
    fn test_config_defaults() {
        let config = ExtractionConfig::default();
        assert_eq!(config.min_duration, 1.0);
        assert_eq!(config.max_duration, 8.0);
        assert_eq!(config.overlap_ratio, 0.5);
    }

    #[test]
    fn test_config_fast() {
        let config = ExtractionConfig::fast();
        assert!(!config.compute_audio);
        assert!(!config.compute_joints);
        assert!(!config.compute_transitions);
    }

    #[test]
    fn test_config_full() {
        let config = ExtractionConfig::full();
        assert!(config.compute_audio);
        assert!(config.compute_joints);
        assert!(config.compute_transitions);
    }
}

//! Library persistence
//!
//! Binary format for saving and loading MotionPhraseLibrary to disk.
//! Uses a custom format with JSON for metadata and phrases.

use std::fs::File;
use std::io::{Read, Write, BufReader, BufWriter};
use std::path::Path;

use serde::{Deserialize, Serialize};

use super::library::{MotionPhraseLibrary, LibraryConfig, LibraryMetadata};
use super::types::MotionPhrase;

/// Magic bytes for library files
const LIBRARY_MAGIC: &[u8; 4] = b"MPLB";

/// Current file format version
const FORMAT_VERSION: u32 = 1;

/// Library persistence error
#[derive(Debug)]
pub enum LibraryError {
    /// I/O error
    IoError(std::io::Error),
    /// JSON serialization error
    SerializationError(String),
    /// Invalid file format
    InvalidFormat,
    /// Unsupported format version
    UnsupportedVersion(u32),
    /// Validation error
    ValidationError(String),
    /// Index error
    IndexError(String),
}

impl std::fmt::Display for LibraryError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::IoError(e) => write!(f, "I/O error: {}", e),
            Self::SerializationError(e) => write!(f, "Serialization error: {}", e),
            Self::InvalidFormat => write!(f, "Invalid library file format"),
            Self::UnsupportedVersion(v) => {
                write!(f, "Unsupported format version: {} (max: {})", v, FORMAT_VERSION)
            }
            Self::ValidationError(e) => write!(f, "Validation error: {}", e),
            Self::IndexError(e) => write!(f, "Index error: {}", e),
        }
    }
}

impl std::error::Error for LibraryError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::IoError(e) => Some(e),
            _ => None,
        }
    }
}

impl From<std::io::Error> for LibraryError {
    fn from(e: std::io::Error) -> Self {
        Self::IoError(e)
    }
}

/// Serializable library wrapper (for JSON encoding)
#[derive(Serialize, Deserialize)]
struct SerializableLibrary {
    /// Format version
    version: u32,
    /// Library configuration
    config: LibraryConfig,
    /// Library metadata
    metadata: LibraryMetadata,
    /// All phrases
    phrases: Vec<MotionPhrase>,
}

/// Save library to binary format
///
/// Format:
/// - 4 bytes: Magic "MPLB"
/// - 4 bytes: Format version (u32 little-endian)
/// - 8 bytes: JSON payload length (u64 little-endian)
/// - N bytes: JSON payload (config + metadata + phrases)
///
/// Note: HNSW indices are rebuilt on load from phrases
pub fn save_library(library: &MotionPhraseLibrary, path: &Path) -> Result<(), LibraryError> {
    let file = File::create(path)?;
    let mut writer = BufWriter::new(file);

    // Write header
    writer.write_all(LIBRARY_MAGIC)?;
    writer.write_all(&FORMAT_VERSION.to_le_bytes())?;

    // Prepare serializable wrapper
    let serializable = SerializableLibrary {
        version: FORMAT_VERSION,
        config: library.config().clone(),
        metadata: library.stats().clone(),
        phrases: library.phrases().values().cloned().collect(),
    };

    // Serialize to JSON
    let json_bytes = serde_json::to_vec(&serializable)
        .map_err(|e| LibraryError::SerializationError(e.to_string()))?;

    // Write length and payload
    writer.write_all(&(json_bytes.len() as u64).to_le_bytes())?;
    writer.write_all(&json_bytes)?;

    writer.flush()?;
    Ok(())
}

/// Load library from binary format
pub fn load_library(path: &Path) -> Result<MotionPhraseLibrary, LibraryError> {
    let file = File::open(path)?;
    let mut reader = BufReader::new(file);

    // Verify magic
    let mut magic = [0u8; 4];
    reader.read_exact(&mut magic)?;
    if &magic != LIBRARY_MAGIC {
        return Err(LibraryError::InvalidFormat);
    }

    // Check version
    let mut version_bytes = [0u8; 4];
    reader.read_exact(&mut version_bytes)?;
    let version = u32::from_le_bytes(version_bytes);
    if version > FORMAT_VERSION {
        return Err(LibraryError::UnsupportedVersion(version));
    }

    // Read payload length
    let mut len_bytes = [0u8; 8];
    reader.read_exact(&mut len_bytes)?;
    let payload_len = u64::from_le_bytes(len_bytes) as usize;

    // Read payload
    let mut json_bytes = vec![0u8; payload_len];
    reader.read_exact(&mut json_bytes)?;

    // Deserialize
    let serializable: SerializableLibrary = serde_json::from_slice(&json_bytes)
        .map_err(|e| LibraryError::SerializationError(e.to_string()))?;

    // Rebuild library with indices
    let mut library = MotionPhraseLibrary::new(serializable.config);
    *library.metadata_mut() = serializable.metadata;

    // Rebuild indices by inserting phrases
    for phrase in serializable.phrases {
        library.insert(phrase)?;
    }

    Ok(library)
}

/// Check if a file is a valid library file
pub fn is_library_file(path: &Path) -> bool {
    let file = match File::open(path) {
        Ok(f) => f,
        Err(_) => return false,
    };
    let mut reader = BufReader::new(file);

    let mut magic = [0u8; 4];
    if reader.read_exact(&mut magic).is_err() {
        return false;
    }

    &magic == LIBRARY_MAGIC
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn test_save_load_empty_library() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test.mplb");

        // Create and save empty library
        let config = LibraryConfig::default();
        let library = MotionPhraseLibrary::new(config);
        save_library(&library, &path).unwrap();

        // Verify file exists and has magic bytes
        assert!(is_library_file(&path));

        // Load and verify
        let loaded = load_library(&path).unwrap();
        assert_eq!(loaded.len(), 0);
    }

    #[test]
    fn test_save_load_with_phrases() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test.mplb");

        // Create library with phrases
        let config = LibraryConfig::default();
        let mut library = MotionPhraseLibrary::new(config);

        for i in 0..5 {
            let embedding: Vec<f32> = (0..64).map(|j| (i * 7 + j) as f32 / 100.0).collect();
            let phrase = MotionPhrase::new(
                format!("phrase_{}", i),
                i as f64,
                (i + 1) as f64,
                embedding,
            );
            library.insert(phrase).unwrap();
        }

        assert_eq!(library.len(), 5);

        // Save
        save_library(&library, &path).unwrap();

        // Load and verify
        let loaded = load_library(&path).unwrap();
        assert_eq!(loaded.len(), 5);

        // Verify phrase data
        let phrase = loaded.get_phrase("phrase_0").unwrap();
        assert_eq!(phrase.phrase_id, "phrase_0");
        assert_eq!(phrase.t_start, 0.0);
    }

    #[test]
    fn test_invalid_format() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("invalid.mplb");

        // Write invalid magic bytes
        let mut file = File::create(&path).unwrap();
        file.write_all(b"XXXX").unwrap();

        let result = load_library(&path);
        assert!(matches!(result, Err(LibraryError::InvalidFormat)));
    }

    #[test]
    fn test_is_library_file() {
        let dir = tempdir().unwrap();

        // Valid library file
        let valid_path = dir.path().join("valid.mplb");
        let config = LibraryConfig::default();
        let library = MotionPhraseLibrary::new(config);
        save_library(&library, &valid_path).unwrap();
        assert!(is_library_file(&valid_path));

        // Invalid file
        let invalid_path = dir.path().join("invalid.txt");
        let mut file = File::create(&invalid_path).unwrap();
        file.write_all(b"not a library").unwrap();
        assert!(!is_library_file(&invalid_path));

        // Non-existent file
        let missing_path = dir.path().join("missing.mplb");
        assert!(!is_library_file(&missing_path));
    }
}

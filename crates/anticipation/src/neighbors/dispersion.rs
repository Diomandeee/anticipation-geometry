//! Continuation dispersion from neighbor statistics
//!
//! Uses HNSW index to find similar motion states and compute
//! how divergent their continuations are.
//!
//! High dispersion = uncertain (many different possible next states)
//! Low dispersion = committed (neighbors all continue similarly)

use rag_plusplus_core::{HNSWConfig, HNSWIndex, VectorIndex, DistanceType, SearchResult};
use std::collections::HashMap;

/// Configuration for the motion phrase index.
#[derive(Debug, Clone)]
pub struct MotionPhraseIndexConfig {
    /// Dimension of regime embeddings
    pub embedding_dim: usize,
    /// HNSW M parameter (connections per layer)
    pub hnsw_m: usize,
    /// HNSW ef_construction
    pub ef_construction: usize,
    /// HNSW ef_search
    pub ef_search: usize,
    /// Number of neighbors to consider for dispersion
    pub k_neighbors: usize,
}

impl Default for MotionPhraseIndexConfig {
    fn default() -> Self {
        Self {
            embedding_dim: 64,
            hnsw_m: 16,
            ef_construction: 200,
            ef_search: 64,
            k_neighbors: 8,
        }
    }
}

/// Continuation data for a motion phrase.
#[derive(Debug, Clone)]
pub struct ContinuationData {
    /// The next regime embedding after this phrase
    pub next_embedding: Vec<f32>,
    /// Change in commitment from this phrase to next
    pub delta_commitment: f32,
    /// Change in uncertainty from this phrase to next
    pub delta_uncertainty: f32,
    /// Time duration of the phrase
    pub duration_seconds: f32,
}

/// Index of motion phrases for continuation dispersion.
///
/// Stores regime embeddings and their continuations for
/// computing uncertainty from neighbor statistics.
#[derive(Debug)]
pub struct MotionPhraseIndex {
    /// HNSW index for regime embeddings
    index: HNSWIndex,
    /// Continuation data for each phrase
    continuations: HashMap<String, ContinuationData>,
    /// Configuration
    config: MotionPhraseIndexConfig,
}

impl MotionPhraseIndex {
    /// Create a new motion phrase index.
    #[must_use]
    pub fn new(config: MotionPhraseIndexConfig) -> Self {
        let hnsw_config = HNSWConfig::new(config.embedding_dim)
            .with_m(config.hnsw_m)
            .with_ef_construction(config.ef_construction)
            .with_ef_search(config.ef_search)
            .with_distance(DistanceType::Cosine);

        Self {
            index: HNSWIndex::new(hnsw_config),
            continuations: HashMap::new(),
            config,
        }
    }

    /// Add a motion phrase to the index.
    ///
    /// # Arguments
    ///
    /// * `phrase_id` - Unique identifier for the phrase
    /// * `embedding` - Regime embedding vector
    /// * `continuation` - What happened after this phrase
    pub fn add_phrase(
        &mut self,
        phrase_id: String,
        embedding: &[f32],
        continuation: ContinuationData,
    ) -> Result<(), String> {
        self.index
            .add(phrase_id.clone(), embedding)
            .map_err(|e| format!("Failed to add phrase: {}", e))?;
        self.continuations.insert(phrase_id, continuation);
        Ok(())
    }

    /// Query for similar phrases.
    ///
    /// Returns search results sorted by similarity.
    pub fn query(&self, embedding: &[f32], k: usize) -> Result<Vec<SearchResult>, String> {
        self.index
            .search(embedding, k)
            .map_err(|e| format!("Search failed: {}", e))
    }

    /// Get continuation data for a phrase.
    #[must_use]
    pub fn get_continuation(&self, phrase_id: &str) -> Option<&ContinuationData> {
        self.continuations.get(phrase_id)
    }

    /// Number of indexed phrases.
    #[must_use]
    pub fn len(&self) -> usize {
        self.index.len()
    }

    /// Whether the index is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.index.is_empty()
    }

    /// Configuration.
    #[must_use]
    pub fn config(&self) -> &MotionPhraseIndexConfig {
        &self.config
    }
}

/// Result of continuation dispersion computation.
#[derive(Debug, Clone)]
pub struct DispersionResult {
    /// Embedding dispersion: variance of neighbor embeddings
    pub embedding_dispersion: f32,
    /// Continuation dispersion: variance of next embeddings
    pub continuation_dispersion: f32,
    /// Mean change in commitment across neighbors
    pub mean_delta_commitment: f32,
    /// Std of commitment changes
    pub std_delta_commitment: f32,
    /// Mean change in uncertainty across neighbors
    pub mean_delta_uncertainty: f32,
    /// Number of neighbors found
    pub neighbor_count: usize,
    /// Mean similarity to neighbors
    pub mean_similarity: f32,
}

impl Default for DispersionResult {
    fn default() -> Self {
        Self {
            embedding_dispersion: 0.5,
            continuation_dispersion: 0.5,
            mean_delta_commitment: 0.0,
            std_delta_commitment: 0.0,
            mean_delta_uncertainty: 0.0,
            neighbor_count: 0,
            mean_similarity: 0.0,
        }
    }
}

/// Compute continuation dispersion from neighbor statistics.
///
/// # Arguments
///
/// * `index` - Motion phrase index
/// * `query_embedding` - Current regime embedding
/// * `k` - Number of neighbors to consider
///
/// # Returns
///
/// Dispersion statistics about neighbor continuations.
pub fn compute_continuation_dispersion(
    index: &MotionPhraseIndex,
    query_embedding: &[f32],
    k: usize,
) -> DispersionResult {
    // Handle empty index
    if index.is_empty() {
        return DispersionResult::default();
    }

    // Query neighbors
    let results = match index.query(query_embedding, k) {
        Ok(r) => r,
        Err(_) => return DispersionResult::default(),
    };

    if results.is_empty() {
        return DispersionResult::default();
    }

    // Collect continuation data
    let mut next_embeddings: Vec<&[f32]> = Vec::new();
    let mut delta_commitments: Vec<f32> = Vec::new();
    let mut delta_uncertainties: Vec<f32> = Vec::new();
    let mut similarities: Vec<f32> = Vec::new();

    for result in &results {
        if let Some(cont) = index.get_continuation(&result.id) {
            next_embeddings.push(&cont.next_embedding);
            delta_commitments.push(cont.delta_commitment);
            delta_uncertainties.push(cont.delta_uncertainty);
            similarities.push(result.score);
        }
    }

    let n = next_embeddings.len();
    if n == 0 {
        return DispersionResult::default();
    }

    // Compute embedding dispersion (variance of neighbor embeddings)
    let embedding_dispersion = compute_embedding_variance(&next_embeddings);

    // Compute continuation dispersion (variance of next embeddings)
    let continuation_dispersion = embedding_dispersion; // Same for now

    // Compute commitment statistics
    let (mean_delta_commitment, std_delta_commitment) = mean_std(&delta_commitments);
    let (mean_delta_uncertainty, _) = mean_std(&delta_uncertainties);
    let mean_similarity = similarities.iter().sum::<f32>() / n as f32;

    DispersionResult {
        embedding_dispersion,
        continuation_dispersion,
        mean_delta_commitment,
        std_delta_commitment,
        mean_delta_uncertainty,
        neighbor_count: n,
        mean_similarity,
    }
}

/// Compute variance of embeddings.
fn compute_embedding_variance(embeddings: &[&[f32]]) -> f32 {
    if embeddings.is_empty() {
        return 0.0;
    }

    let dim = embeddings[0].len();
    let n = embeddings.len() as f32;

    // Compute mean embedding
    let mut mean = vec![0.0f32; dim];
    for emb in embeddings {
        for (i, &v) in emb.iter().enumerate() {
            mean[i] += v / n;
        }
    }

    // Compute variance (sum of squared distances from mean)
    let mut variance = 0.0f32;
    for emb in embeddings {
        for (i, &v) in emb.iter().enumerate() {
            let diff = v - mean[i];
            variance += diff * diff;
        }
    }
    variance /= n * dim as f32;

    // Normalize to [0, 1] range (assuming embeddings are normalized)
    variance.sqrt().clamp(0.0, 1.0)
}

/// Compute mean and standard deviation.
fn mean_std(values: &[f32]) -> (f32, f32) {
    if values.is_empty() {
        return (0.0, 0.0);
    }

    let n = values.len() as f32;
    let mean = values.iter().sum::<f32>() / n;
    let variance = values.iter().map(|&v| (v - mean).powi(2)).sum::<f32>() / n;

    (mean, variance.sqrt())
}

/// Convert dispersion to uncertainty score.
///
/// Maps dispersion statistics to a scalar uncertainty in [0, 1].
pub fn dispersion_to_uncertainty(result: &DispersionResult) -> f32 {
    if result.neighbor_count == 0 {
        return 0.5; // No information
    }

    // Combine factors:
    // - High continuation dispersion = high uncertainty
    // - High std of commitment changes = high uncertainty
    // - Low similarity to neighbors = high uncertainty

    let dispersion_factor = result.continuation_dispersion;
    let commitment_variance_factor = result.std_delta_commitment.clamp(0.0, 1.0);
    let similarity_factor = 1.0 - result.mean_similarity;

    // Weighted combination
    let uncertainty = 0.5 * dispersion_factor
        + 0.3 * commitment_variance_factor
        + 0.2 * similarity_factor;

    uncertainty.clamp(0.0, 1.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_empty_index() {
        let config = MotionPhraseIndexConfig::default();
        let index = MotionPhraseIndex::new(config);
        let query = vec![0.0f32; 64];

        let result = compute_continuation_dispersion(&index, &query, 8);
        assert_eq!(result.neighbor_count, 0);
        assert_eq!(result.embedding_dispersion, 0.5);
    }

    #[test]
    fn test_add_and_query() {
        let config = MotionPhraseIndexConfig::default();
        let mut index = MotionPhraseIndex::new(config);

        // Add a phrase
        let embedding: Vec<f32> = (0..64).map(|i| (i as f32 / 64.0)).collect();
        let continuation = ContinuationData {
            next_embedding: embedding.clone(),
            delta_commitment: 0.1,
            delta_uncertainty: -0.05,
            duration_seconds: 1.0,
        };

        index.add_phrase("phrase_1".to_string(), &embedding, continuation).unwrap();
        assert_eq!(index.len(), 1);

        // Query
        let results = index.query(&embedding, 1).unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].id, "phrase_1");
    }

    #[test]
    fn test_dispersion_uniform_continuations() {
        let config = MotionPhraseIndexConfig::default();
        let mut index = MotionPhraseIndex::new(config);

        // Add phrases with identical continuations
        for i in 0..10 {
            let mut embedding: Vec<f32> = (0..64).map(|j| ((i * 7 + j) as f32 / 64.0) % 1.0).collect();
            // Normalize
            let norm: f32 = embedding.iter().map(|v| v * v).sum::<f32>().sqrt();
            for v in &mut embedding {
                *v /= norm;
            }

            let continuation = ContinuationData {
                next_embedding: vec![0.5; 64], // All same
                delta_commitment: 0.1,
                delta_uncertainty: 0.0,
                duration_seconds: 1.0,
            };

            index.add_phrase(format!("phrase_{}", i), &embedding, continuation).unwrap();
        }

        // Query
        let query = vec![0.1f32; 64];
        let result = compute_continuation_dispersion(&index, &query, 8);

        // Uniform continuations should have low dispersion
        assert!(result.continuation_dispersion < 0.3,
            "Expected low dispersion for uniform continuations, got {}",
            result.continuation_dispersion);
    }

    #[test]
    fn test_dispersion_to_uncertainty() {
        // No neighbors
        let result = DispersionResult::default();
        assert_eq!(dispersion_to_uncertainty(&result), 0.5);

        // High dispersion
        let high_dispersion = DispersionResult {
            embedding_dispersion: 0.9,
            continuation_dispersion: 0.9,
            mean_delta_commitment: 0.0,
            std_delta_commitment: 0.8,
            mean_delta_uncertainty: 0.0,
            neighbor_count: 5,
            mean_similarity: 0.5,
        };
        let u = dispersion_to_uncertainty(&high_dispersion);
        assert!(u > 0.6, "Expected high uncertainty for high dispersion, got {}", u);

        // Low dispersion
        let low_dispersion = DispersionResult {
            embedding_dispersion: 0.1,
            continuation_dispersion: 0.1,
            mean_delta_commitment: 0.0,
            std_delta_commitment: 0.1,
            mean_delta_uncertainty: 0.0,
            neighbor_count: 5,
            mean_similarity: 0.95,
        };
        let u = dispersion_to_uncertainty(&low_dispersion);
        assert!(u < 0.3, "Expected low uncertainty for low dispersion, got {}", u);
    }

    #[test]
    fn test_mean_std() {
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let (mean, std) = mean_std(&values);
        assert!((mean - 3.0).abs() < 1e-6);
        assert!((std - 1.414).abs() < 0.01);

        let empty: Vec<f32> = vec![];
        let (m, s) = mean_std(&empty);
        assert_eq!(m, 0.0);
        assert_eq!(s, 0.0);
    }
}

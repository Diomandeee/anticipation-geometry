//! MotionPhraseLibrary - Multi-index phrase retrieval
//!
//! High-level library abstraction over HNSW indices for motion phrase retrieval.
//! Supports multi-axis queries with RRF fusion.

use std::collections::HashMap;
use std::path::Path;

use rag_plusplus_core::{HNSWConfig, HNSWIndex, VectorIndex, DistanceType};
use serde::{Deserialize, Serialize};

use super::types::{MotionPhrase, DEFAULT_MOTION_DIM, DEFAULT_AUDIO_DIM, DEFAULT_JOINT_DIM, DEFAULT_TRANSITION_DIM};
use super::prior::{PriorBundle, AxisScores, StatsAccumulator};
use super::fusion::rrf_fuse;
use super::persistence::LibraryError;

/// Reranking strategy for query results
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub enum RerankStrategy {
    /// No reranking (keep original order)
    #[default]
    None,
    /// Rerank by outcome statistics (commitment, uncertainty)
    OutcomeWeighted,
    /// Boost more recent phrases
    Recency,
    /// Maximal Marginal Relevance for diversity
    MMR,
    /// Combine outcome and recency
    Composite,
}


/// Configuration for the phrase library
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct LibraryConfig {
    /// HNSW M parameter (connections per node)
    pub hnsw_m: usize,
    /// HNSW ef_construction parameter
    pub hnsw_ef_construction: usize,
    /// HNSW ef_search parameter
    pub hnsw_ef_search: usize,
    /// Motion embedding dimension
    pub motion_dim: usize,
    /// Audio embedding dimension
    pub audio_dim: usize,
    /// Joint embedding dimension
    pub joint_dim: usize,
    /// Transition embedding dimension
    pub transition_dim: usize,
    /// Number of neighbors to retrieve by default
    pub default_k: usize,
    /// RRF k parameter for score fusion
    pub rrf_k: f32,
    /// Reranking strategy (default: None)
    #[serde(default)]
    pub rerank_strategy: RerankStrategy,
    /// Weight for original score in reranking (0-1)
    #[serde(default = "default_original_weight")]
    pub rerank_original_weight: f32,
    /// Weight for outcome score in reranking (0-1)
    #[serde(default = "default_outcome_weight")]
    pub rerank_outcome_weight: f32,
    /// MMR lambda (0 = pure diversity, 1 = pure relevance)
    #[serde(default = "default_mmr_lambda")]
    pub rerank_mmr_lambda: f32,
}

fn default_original_weight() -> f32 { 0.5 }
fn default_outcome_weight() -> f32 { 0.3 }
fn default_mmr_lambda() -> f32 { 0.7 }

impl Default for LibraryConfig {
    fn default() -> Self {
        Self {
            hnsw_m: 16,
            hnsw_ef_construction: 200,
            hnsw_ef_search: 64,
            motion_dim: DEFAULT_MOTION_DIM,
            audio_dim: DEFAULT_AUDIO_DIM,
            joint_dim: DEFAULT_JOINT_DIM,
            transition_dim: DEFAULT_TRANSITION_DIM,
            default_k: 8,
            rrf_k: 60.0,
            rerank_strategy: RerankStrategy::None,
            rerank_original_weight: default_original_weight(),
            rerank_outcome_weight: default_outcome_weight(),
            rerank_mmr_lambda: default_mmr_lambda(),
        }
    }
}

impl LibraryConfig {
    /// Enable outcome-weighted reranking
    pub fn with_outcome_reranking(mut self) -> Self {
        self.rerank_strategy = RerankStrategy::OutcomeWeighted;
        self
    }

    /// Enable MMR reranking for diversity
    pub fn with_mmr_reranking(mut self, lambda: f32) -> Self {
        self.rerank_strategy = RerankStrategy::MMR;
        self.rerank_mmr_lambda = lambda;
        self
    }
}

/// Library metadata
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct LibraryMetadata {
    /// File format version
    pub version: u32,
    /// Number of phrases in library
    pub phrase_count: usize,
    /// Creation timestamp (Unix epoch millis)
    pub created_at: u64,
    /// Last update timestamp
    pub last_updated: u64,
    /// Number of unique performers
    pub performer_count: usize,
    /// Total duration of all phrases (seconds)
    pub total_duration_seconds: f64,
}

impl LibraryMetadata {
    /// Current format version
    pub const FORMAT_VERSION: u32 = 1;

    /// Create new metadata with current timestamp
    pub fn new() -> Self {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_millis() as u64)
            .unwrap_or(0);
        Self {
            version: Self::FORMAT_VERSION,
            created_at: now,
            last_updated: now,
            ..Default::default()
        }
    }
}

/// Multi-axis phrase library
///
/// Stores motion phrases with HNSW indices for each embedding axis.
/// Supports single-axis and multi-axis queries with RRF fusion.
///
/// # Reranking
///
/// When a reranking strategy is configured, results are reranked after RRF fusion:
/// - `OutcomeWeighted`: Prioritize phrases with better historical outcomes
/// - `MMR`: Diversify results while maintaining relevance
/// - `Composite`: Combine outcome and recency factors
pub struct MotionPhraseLibrary {
    config: LibraryConfig,
    /// Primary motion axis index
    motion_index: HNSWIndex,
    /// Audio axis index (optional)
    audio_index: Option<HNSWIndex>,
    /// Joint axis index (optional)
    joint_index: Option<HNSWIndex>,
    /// Transition axis index (optional)
    transition_index: Option<HNSWIndex>,
    /// Full phrase records by ID
    phrases: HashMap<String, MotionPhrase>,
    /// Library metadata
    metadata: LibraryMetadata,
}

impl MotionPhraseLibrary {
    /// Create empty library with config
    pub fn new(config: LibraryConfig) -> Self {
        let motion_hnsw = HNSWConfig::new(config.motion_dim)
            .with_m(config.hnsw_m)
            .with_ef_construction(config.hnsw_ef_construction)
            .with_ef_search(config.hnsw_ef_search)
            .with_distance(DistanceType::Cosine);

        Self {
            motion_index: HNSWIndex::new(motion_hnsw),
            audio_index: None,
            joint_index: None,
            transition_index: None,
            phrases: HashMap::new(),
            metadata: LibraryMetadata::new(),
            config,
        }
    }

    /// Load library from disk
    pub fn load(path: &Path) -> Result<Self, LibraryError> {
        super::persistence::load_library(path)
    }

    /// Save library to disk
    pub fn save(&self, path: &Path) -> Result<(), LibraryError> {
        super::persistence::save_library(self, path)
    }

    /// Add a phrase to the library
    pub fn insert(&mut self, phrase: MotionPhrase) -> Result<(), LibraryError> {
        // Validate phrase
        phrase.validate().map_err(|e| LibraryError::ValidationError(e.to_string()))?;

        // Add to motion index (required)
        self.motion_index
            .add(phrase.phrase_id.clone(), &phrase.embeddings.motion)
            .map_err(|e| LibraryError::IndexError(e.to_string()))?;

        // Add to optional indices if embeddings present
        if let Some(ref audio) = phrase.embeddings.audio {
            self.ensure_audio_index();
            if let Some(ref mut idx) = self.audio_index {
                idx.add(phrase.phrase_id.clone(), audio)
                    .map_err(|e| LibraryError::IndexError(e.to_string()))?;
            }
        }

        if let Some(ref joint) = phrase.embeddings.joint {
            self.ensure_joint_index();
            if let Some(ref mut idx) = self.joint_index {
                idx.add(phrase.phrase_id.clone(), joint)
                    .map_err(|e| LibraryError::IndexError(e.to_string()))?;
            }
        }

        if let Some(ref transition) = phrase.embeddings.transition {
            self.ensure_transition_index();
            if let Some(ref mut idx) = self.transition_index {
                idx.add(phrase.phrase_id.clone(), transition)
                    .map_err(|e| LibraryError::IndexError(e.to_string()))?;
            }
        }

        // Update metadata
        self.metadata.total_duration_seconds += phrase.duration as f64;
        self.metadata.phrase_count += 1;
        self.metadata.last_updated = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_millis() as u64)
            .unwrap_or(0);

        // Store phrase
        self.phrases.insert(phrase.phrase_id.clone(), phrase);

        Ok(())
    }

    /// Batch insert phrases
    pub fn insert_batch(&mut self, phrases: Vec<MotionPhrase>) -> Result<usize, LibraryError> {
        let mut count = 0;
        for phrase in phrases {
            self.insert(phrase)?;
            count += 1;
        }
        Ok(count)
    }

    /// Query for similar phrases using motion axis only
    pub fn query_motion(&self, embedding: &[f32], k: usize) -> Vec<(String, f32)> {
        match self.motion_index.search(embedding, k) {
            Ok(results) => results.into_iter().map(|r| (r.id, r.score)).collect(),
            Err(_) => Vec::new(),
        }
    }

    /// Multi-axis query with RRF fusion and optional reranking
    ///
    /// When a reranking strategy is configured (via LibraryConfig), results are
    /// reranked after RRF fusion. Strategies include:
    /// - `OutcomeWeighted`: Boost phrases with higher commitment scores
    /// - `MMR`: Diversify results using Maximal Marginal Relevance
    /// - `Composite`: Combine outcome and recency factors
    pub fn query_multi_axis(
        &self,
        query: &MultiAxisQuery,
        k: usize,
    ) -> Vec<(String, f32, AxisScores)> {
        let mut ranked_lists: Vec<Vec<(String, f32)>> = Vec::new();
        let mut weights: Vec<f32> = Vec::new();

        // Motion axis (always present)
        let motion_results = self.query_motion(&query.motion, k * 2);
        if !motion_results.is_empty() {
            ranked_lists.push(motion_results);
            weights.push(query.weights.as_ref().map(|w| w.motion).unwrap_or(1.0));
        }

        // Audio axis
        if let (Some(ref audio_emb), Some(ref audio_idx)) = (&query.audio, &self.audio_index) {
            if let Ok(results) = audio_idx.search(audio_emb, k * 2) {
                let audio_results: Vec<_> = results.into_iter().map(|r| (r.id, r.score)).collect();
                if !audio_results.is_empty() {
                    ranked_lists.push(audio_results);
                    weights.push(query.weights.as_ref().map(|w| w.audio).unwrap_or(1.0));
                }
            }
        }

        // Joint axis
        if let (Some(ref joint_emb), Some(ref joint_idx)) = (&query.joint, &self.joint_index) {
            if let Ok(results) = joint_idx.search(joint_emb, k * 2) {
                let joint_results: Vec<_> = results.into_iter().map(|r| (r.id, r.score)).collect();
                if !joint_results.is_empty() {
                    ranked_lists.push(joint_results);
                    weights.push(query.weights.as_ref().map(|w| w.joint).unwrap_or(1.0));
                }
            }
        }

        // Transition axis
        if let (Some(ref trans_emb), Some(ref trans_idx)) = (&query.transition, &self.transition_index) {
            if let Ok(results) = trans_idx.search(trans_emb, k * 2) {
                let trans_results: Vec<_> = results.into_iter().map(|r| (r.id, r.score)).collect();
                if !trans_results.is_empty() {
                    ranked_lists.push(trans_results);
                    weights.push(query.weights.as_ref().map(|w| w.transition).unwrap_or(1.0));
                }
            }
        }

        if ranked_lists.is_empty() {
            return Vec::new();
        }

        // Fuse with RRF
        let fused = rrf_fuse(&ranked_lists, &weights, self.config.rrf_k);

        // Build results with axis scores
        let mut results: Vec<_> = fused
            .into_iter()
            .map(|(id, score)| {
                let axis_scores = self.compute_axis_scores(&id, &ranked_lists);
                (id, score, axis_scores)
            })
            .collect();

        // Apply reranking if configured
        results = self.apply_reranking(results);

        // Return top k
        results.into_iter().take(k).collect()
    }

    /// Apply reranking strategy to query results
    ///
    /// Reranking strategies:
    /// - `None`: No reranking (pass through)
    /// - `OutcomeWeighted`: Boost by historical commitment scores
    /// - `Recency`: Boost more recent phrases
    /// - `MMR`: Maximal Marginal Relevance for diversity
    /// - `Composite`: Combined outcome + recency
    fn apply_reranking(
        &self,
        results: Vec<(String, f32, AxisScores)>,
    ) -> Vec<(String, f32, AxisScores)> {
        match self.config.rerank_strategy {
            RerankStrategy::None => results,
            RerankStrategy::OutcomeWeighted => {
                self.rerank_by_outcome(results)
            }
            RerankStrategy::Recency => {
                self.rerank_by_recency(results)
            }
            RerankStrategy::MMR => {
                self.rerank_mmr(results)
            }
            RerankStrategy::Composite => {
                self.rerank_composite(results)
            }
        }
    }

    /// Rerank by outcome (commitment) scores
    fn rerank_by_outcome(
        &self,
        mut results: Vec<(String, f32, AxisScores)>,
    ) -> Vec<(String, f32, AxisScores)> {
        let original_weight = self.config.rerank_original_weight;
        let outcome_weight = self.config.rerank_outcome_weight;

        for (phrase_id, score, _) in &mut results {
            if let Some(phrase) = self.phrases.get(phrase_id) {
                // Use commitment as outcome (higher = better)
                let outcome_score = phrase.outcome.delta_commitment.clamp(0.0, 1.0);
                *score = original_weight * *score + outcome_weight * outcome_score;
            }
        }

        // Sort by new score (descending)
        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        results
    }

    /// Rerank by recency (boost newer phrases)
    fn rerank_by_recency(
        &self,
        mut results: Vec<(String, f32, AxisScores)>,
    ) -> Vec<(String, f32, AxisScores)> {
        let original_weight = self.config.rerank_original_weight;
        let recency_weight = 1.0 - original_weight;

        // Get current timestamp
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_millis() as u64)
            .unwrap_or(0);

        let half_life_ms = 7.0 * 24.0 * 3600.0 * 1000.0; // 7 days in ms

        for (phrase_id, score, _) in &mut results {
            if let Some(phrase) = self.phrases.get(phrase_id) {
                // Use provenance.extracted_at as the phrase timestamp
                let age_ms = now.saturating_sub(phrase.provenance.extracted_at) as f64;
                let recency_score = (-age_ms / half_life_ms * std::f64::consts::LN_2).exp() as f32;
                *score = original_weight * *score + recency_weight * recency_score;
            }
        }

        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        results
    }

    /// Rerank using Maximal Marginal Relevance for diversity
    fn rerank_mmr(
        &self,
        results: Vec<(String, f32, AxisScores)>,
    ) -> Vec<(String, f32, AxisScores)> {
        if results.len() <= 1 {
            return results;
        }

        let lambda = self.config.rerank_mmr_lambda;
        let mut reranked = Vec::with_capacity(results.len());
        let mut remaining: Vec<_> = results;

        // Sort by original score first
        remaining.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        // Select first by pure relevance
        reranked.push(remaining.remove(0));

        // Select remaining by MMR
        while !remaining.is_empty() {
            let mut best_idx = 0;
            let mut best_mmr = f32::NEG_INFINITY;

            for (i, (cand_id, cand_score, _)) in remaining.iter().enumerate() {
                // Relevance term
                let relevance = *cand_score;

                // Diversity term (max similarity to already selected)
                let max_sim = reranked.iter()
                    .map(|(sel_id, _, _)| self.phrase_similarity(cand_id, sel_id))
                    .fold(0.0f32, f32::max);

                // MMR score: lambda * relevance - (1 - lambda) * max_similarity
                let mmr = lambda * relevance - (1.0 - lambda) * max_sim;

                if mmr > best_mmr {
                    best_mmr = mmr;
                    best_idx = i;
                }
            }

            reranked.push(remaining.remove(best_idx));
        }

        reranked
    }

    /// Composite reranking (outcome + recency)
    fn rerank_composite(
        &self,
        mut results: Vec<(String, f32, AxisScores)>,
    ) -> Vec<(String, f32, AxisScores)> {
        let original_weight = self.config.rerank_original_weight;
        let outcome_weight = self.config.rerank_outcome_weight;
        let recency_weight = 1.0 - original_weight - outcome_weight;

        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_millis() as u64)
            .unwrap_or(0);

        let half_life_ms = 7.0 * 24.0 * 3600.0 * 1000.0;

        for (phrase_id, score, _) in &mut results {
            if let Some(phrase) = self.phrases.get(phrase_id) {
                let outcome_score = phrase.outcome.delta_commitment.clamp(0.0, 1.0);
                // Use provenance.extracted_at as the phrase timestamp
                let age_ms = now.saturating_sub(phrase.provenance.extracted_at) as f64;
                let recency_score = (-age_ms / half_life_ms * std::f64::consts::LN_2).exp() as f32;

                *score = original_weight * *score
                    + outcome_weight * outcome_score
                    + recency_weight.max(0.0) * recency_score;
            }
        }

        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        results
    }

    /// Compute cosine similarity between two phrases (using motion embeddings)
    fn phrase_similarity(&self, id_a: &str, id_b: &str) -> f32 {
        let (phrase_a, phrase_b) = match (self.phrases.get(id_a), self.phrases.get(id_b)) {
            (Some(a), Some(b)) => (a, b),
            _ => return 0.0,
        };

        let emb_a = &phrase_a.embeddings.motion;
        let emb_b = &phrase_b.embeddings.motion;

        if emb_a.len() != emb_b.len() || emb_a.is_empty() {
            return 0.0;
        }

        // Cosine similarity
        let dot: f32 = emb_a.iter().zip(emb_b.iter()).map(|(a, b)| a * b).sum();
        let norm_a: f32 = emb_a.iter().map(|x| x * x).sum::<f32>().sqrt();
        let norm_b: f32 = emb_b.iter().map(|x| x * x).sum::<f32>().sqrt();

        if norm_a > 0.0 && norm_b > 0.0 {
            dot / (norm_a * norm_b)
        } else {
            0.0
        }
    }

    /// Build PriorBundle from query results
    ///
    /// Uses StatsAccumulator (OutcomeStats when `neighbors` feature enabled) for
    /// numerically stable statistics with optional confidence intervals.
    pub fn build_prior_bundle(
        &self,
        query_id: &str,
        neighbors: &[(String, f32, AxisScores)],
    ) -> PriorBundle {
        if neighbors.is_empty() {
            return PriorBundle::empty(query_id);
        }

        let mut bundle = PriorBundle::new(query_id);
        bundle.k = neighbors.len();

        // Compute statistics using StatsAccumulator (OutcomeStats when feature enabled)
        let mut similarity_acc = StatsAccumulator::new();
        let mut commitment_acc = StatsAccumulator::new();
        let mut uncertainty_acc = StatsAccumulator::new();

        // Collect next embeddings for dispersion
        let mut next_embeddings: Vec<&[f32]> = Vec::new();
        let mut regime_counts: HashMap<String, usize> = HashMap::new();

        for (phrase_id, similarity, axis_scores) in neighbors {
            similarity_acc.push(*similarity);

            if let Some(phrase) = self.phrases.get(phrase_id) {
                commitment_acc.push(phrase.outcome.delta_commitment);
                uncertainty_acc.push(phrase.outcome.delta_uncertainty);

                if !phrase.outcome.next_embedding.is_empty() {
                    next_embeddings.push(&phrase.outcome.next_embedding);
                }

                *regime_counts.entry(phrase.outcome.next_regime.clone()).or_default() += 1;
            }

            // Use the last axis scores (they should all be similar)
            bundle.axis_scores = axis_scores.clone();
        }

        // Populate basic statistics
        bundle.mean_similarity = similarity_acc.mean();
        bundle.commitment_mean = commitment_acc.mean();
        bundle.commitment_std = commitment_acc.std();
        bundle.uncertainty_mean = uncertainty_acc.mean();
        bundle.uncertainty_std = uncertainty_acc.std();

        // Populate confidence intervals (available when `neighbors` feature enabled)
        bundle.commitment_ci = commitment_acc.confidence_interval_90();
        bundle.uncertainty_ci = uncertainty_acc.confidence_interval_90();
        bundle.sample_count = commitment_acc.sample_count();

        // Compute dispersion from next embeddings
        bundle.dispersion = self.compute_embedding_dispersion(&next_embeddings);

        // Compute neighbor-based uncertainty (same formula as dispersion.rs)
        bundle.neighbor_uncertainty = self.dispersion_to_uncertainty(
            bundle.dispersion,
            bundle.commitment_std,
            bundle.mean_similarity,
        );

        // Build regime distribution
        let total_count = regime_counts.values().sum::<usize>() as f32;
        let mut regime_dist: Vec<_> = regime_counts
            .into_iter()
            .map(|(regime, count)| (regime, count as f32 / total_count))
            .collect();
        regime_dist.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        bundle.likely_next_regime = regime_dist.first().map(|(r, _)| r.clone()).unwrap_or_default();
        bundle.regime_distribution = regime_dist;

        bundle
    }

    /// Get phrase by ID
    pub fn get_phrase(&self, phrase_id: &str) -> Option<&MotionPhrase> {
        self.phrases.get(phrase_id)
    }

    /// Get library statistics
    pub fn stats(&self) -> &LibraryMetadata {
        &self.metadata
    }

    /// Get library config
    pub fn config(&self) -> &LibraryConfig {
        &self.config
    }

    /// Number of phrases
    pub fn len(&self) -> usize {
        self.phrases.len()
    }

    /// Whether library is empty
    pub fn is_empty(&self) -> bool {
        self.phrases.is_empty()
    }

    /// Get all phrases (for serialization)
    pub(crate) fn phrases(&self) -> &HashMap<String, MotionPhrase> {
        &self.phrases
    }

    /// Get mutable metadata (for deserialization)
    pub(crate) fn metadata_mut(&mut self) -> &mut LibraryMetadata {
        &mut self.metadata
    }

    // === Private helpers ===

    fn ensure_audio_index(&mut self) {
        if self.audio_index.is_none() {
            let config = HNSWConfig::new(self.config.audio_dim)
                .with_m(self.config.hnsw_m)
                .with_ef_construction(self.config.hnsw_ef_construction)
                .with_ef_search(self.config.hnsw_ef_search)
                .with_distance(DistanceType::Cosine);
            self.audio_index = Some(HNSWIndex::new(config));
        }
    }

    fn ensure_joint_index(&mut self) {
        if self.joint_index.is_none() {
            let config = HNSWConfig::new(self.config.joint_dim)
                .with_m(self.config.hnsw_m)
                .with_ef_construction(self.config.hnsw_ef_construction)
                .with_ef_search(self.config.hnsw_ef_search)
                .with_distance(DistanceType::Cosine);
            self.joint_index = Some(HNSWIndex::new(config));
        }
    }

    fn ensure_transition_index(&mut self) {
        if self.transition_index.is_none() {
            let config = HNSWConfig::new(self.config.transition_dim)
                .with_m(self.config.hnsw_m)
                .with_ef_construction(self.config.hnsw_ef_construction)
                .with_ef_search(self.config.hnsw_ef_search)
                .with_distance(DistanceType::Cosine);
            self.transition_index = Some(HNSWIndex::new(config));
        }
    }

    fn compute_axis_scores(&self, id: &str, ranked_lists: &[Vec<(String, f32)>]) -> AxisScores {
        let mut scores = AxisScores::default();

        // Simplified: just check presence in each list
        // In practice, could use rank or score from each list
        for (list_idx, list) in ranked_lists.iter().enumerate() {
            let score = list.iter()
                .find(|(pid, _)| pid == id)
                .map(|(_, s)| *s)
                .unwrap_or(0.0);

            match list_idx {
                0 => scores.motion = score,
                1 => scores.audio = score,
                2 => scores.joint = score,
                3 => scores.transition = score,
                _ => {}
            }
        }

        scores.fused = scores.total() / ranked_lists.len().max(1) as f32;
        scores
    }

    fn compute_embedding_dispersion(&self, embeddings: &[&[f32]]) -> f32 {
        if embeddings.is_empty() {
            return 0.5; // Default uncertainty
        }

        let dim = embeddings[0].len();
        if dim == 0 {
            return 0.5;
        }

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

        // Normalize to [0, 1] range
        variance.sqrt().clamp(0.0, 1.0)
    }

    fn dispersion_to_uncertainty(
        &self,
        dispersion: f32,
        commitment_std: f32,
        mean_similarity: f32,
    ) -> f32 {
        // Same formula as neighbors/dispersion.rs
        let dispersion_factor = dispersion;
        let commitment_variance_factor = commitment_std.clamp(0.0, 1.0);
        let similarity_factor = 1.0 - mean_similarity;

        let uncertainty = 0.5 * dispersion_factor
            + 0.3 * commitment_variance_factor
            + 0.2 * similarity_factor;

        uncertainty.clamp(0.0, 1.0)
    }
}

/// Multi-axis query input
#[derive(Clone, Debug)]
pub struct MultiAxisQuery {
    /// Motion embedding (required)
    pub motion: Vec<f32>,
    /// Audio embedding (optional)
    pub audio: Option<Vec<f32>>,
    /// Joint embedding (optional)
    pub joint: Option<Vec<f32>>,
    /// Transition embedding (optional)
    pub transition: Option<Vec<f32>>,
    /// Axis weights for fusion (default: equal)
    pub weights: Option<AxisWeights>,
}

impl MultiAxisQuery {
    /// Create a motion-only query
    pub fn motion_only(motion: Vec<f32>) -> Self {
        Self {
            motion,
            audio: None,
            joint: None,
            transition: None,
            weights: None,
        }
    }

    /// Create a query with custom weights
    pub fn with_weights(mut self, weights: AxisWeights) -> Self {
        self.weights = Some(weights);
        self
    }
}

/// Axis weights for multi-index fusion
#[derive(Clone, Debug, Default)]
pub struct AxisWeights {
    /// Motion axis weight
    pub motion: f32,
    /// Audio axis weight
    pub audio: f32,
    /// Joint axis weight
    pub joint: f32,
    /// Transition axis weight
    pub transition: f32,
}

impl AxisWeights {
    /// Create uniform weights
    pub fn uniform() -> Self {
        Self {
            motion: 1.0,
            audio: 1.0,
            joint: 1.0,
            transition: 1.0,
        }
    }

    /// Create motion-heavy weights
    pub fn motion_heavy() -> Self {
        Self {
            motion: 2.0,
            audio: 0.5,
            joint: 0.5,
            transition: 1.0,
        }
    }
}

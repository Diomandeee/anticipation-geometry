//! PriorBundle output type
//!
//! Output from RAG++ queries consumed by the Conductor for decision-making.
//!
//! When the `neighbors` feature is enabled, uses RAG++'s OutcomeStats for
//! numerically stable statistics with confidence intervals.

use serde::{Deserialize, Serialize};

#[cfg(feature = "neighbors")]
use rag_plusplus_core::OutcomeStats;

/// Bundle of priors from neighbor retrieval
///
/// This is the output of a multi-axis query to the MotionPhraseLibrary.
/// It contains aggregated statistics from the K nearest neighbors that
/// inform uncertainty estimation and continuation prediction.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PriorBundle {
    /// Query phrase ID (or "live" for real-time queries)
    pub query_id: String,
    /// Number of neighbors retrieved
    pub k: usize,
    /// Average similarity score of neighbors [0, 1]
    pub mean_similarity: f32,

    // === Continuation statistics ===
    /// Mean of delta_commitment across neighbors
    pub commitment_mean: f32,
    /// Standard deviation of delta_commitment
    pub commitment_std: f32,
    /// Mean of delta_uncertainty across neighbors
    pub uncertainty_mean: f32,
    /// Standard deviation of delta_uncertainty
    pub uncertainty_std: f32,

    // === Prototype curves ===
    /// Most common next regime (mode)
    pub likely_next_regime: String,
    /// Distribution over next regimes [(regime, probability)]
    pub regime_distribution: Vec<(String, f32)>,

    // === Dispersion metrics ===
    /// Embedding dispersion (variance) of neighbors
    pub dispersion: f32,
    /// Computed neighbor-based uncertainty [0, 1]
    pub neighbor_uncertainty: f32,

    // === Confidence intervals (from OutcomeStats) ===
    /// 90% confidence interval for commitment_mean (lower, upper)
    pub commitment_ci: Option<(f32, f32)>,
    /// 90% confidence interval for uncertainty_mean (lower, upper)
    pub uncertainty_ci: Option<(f32, f32)>,
    /// Sample count for statistics (may exceed k if merged with record stats)
    pub sample_count: u64,

    // === Selection scores (for multi-index) ===
    /// Per-axis contribution to final ranking
    pub axis_scores: AxisScores,
}

impl Default for PriorBundle {
    fn default() -> Self {
        Self {
            query_id: String::new(),
            k: 0,
            mean_similarity: 0.0,
            commitment_mean: 0.0,
            commitment_std: 0.0,
            uncertainty_mean: 0.0,
            uncertainty_std: 0.0,
            likely_next_regime: "unknown".to_string(),
            regime_distribution: Vec::new(),
            dispersion: 0.0,
            neighbor_uncertainty: 0.5, // Default to maximum uncertainty
            commitment_ci: None,
            uncertainty_ci: None,
            sample_count: 0,
            axis_scores: AxisScores::default(),
        }
    }
}

impl PriorBundle {
    /// Create a new prior bundle with query ID
    pub fn new(query_id: impl Into<String>) -> Self {
        Self {
            query_id: query_id.into(),
            ..Default::default()
        }
    }

    /// Create a prior bundle indicating no neighbors were found
    pub fn empty(query_id: impl Into<String>) -> Self {
        Self {
            query_id: query_id.into(),
            k: 0,
            neighbor_uncertainty: 1.0, // Maximum uncertainty when no neighbors
            ..Default::default()
        }
    }

    /// Check if this bundle has valid neighbor data
    pub fn has_neighbors(&self) -> bool {
        self.k > 0
    }

    /// Get the blended uncertainty (neighbor + heuristic)
    ///
    /// Uses 70% neighbor-based + 30% heuristic weight as documented
    /// in the dispersion module.
    pub fn blended_uncertainty(&self, heuristic_uncertainty: f32) -> f32 {
        if self.has_neighbors() {
            0.7 * self.neighbor_uncertainty + 0.3 * heuristic_uncertainty
        } else {
            heuristic_uncertainty
        }
    }

    /// Get the confidence in the continuation prediction
    ///
    /// Higher similarity + lower dispersion = higher confidence
    pub fn continuation_confidence(&self) -> f32 {
        if !self.has_neighbors() {
            return 0.0;
        }
        // High similarity and low dispersion means confident prediction
        let similarity_factor = self.mean_similarity;
        let dispersion_factor = 1.0 - self.dispersion.min(1.0);
        (similarity_factor * dispersion_factor).clamp(0.0, 1.0)
    }

    /// Get conservative commitment estimate using confidence interval lower bound
    ///
    /// Returns the lower bound of the 90% CI if available, otherwise the mean.
    /// This implements "optimistic pessimism" - we explore widely but estimate conservatively.
    pub fn commitment_conservative(&self) -> f32 {
        self.commitment_ci
            .map(|(lower, _)| lower)
            .unwrap_or(self.commitment_mean)
    }

    /// Get conservative uncertainty estimate using confidence interval upper bound
    ///
    /// Returns the upper bound of the 90% CI if available, otherwise the mean.
    /// For uncertainty, conservative means assuming higher uncertainty.
    pub fn uncertainty_conservative(&self) -> f32 {
        self.uncertainty_ci
            .map(|(_, upper)| upper)
            .unwrap_or(self.uncertainty_mean)
    }

    /// Check if confidence intervals are available
    pub fn has_confidence_intervals(&self) -> bool {
        self.commitment_ci.is_some() && self.uncertainty_ci.is_some()
    }
}

/// Per-axis contribution scores
///
/// Tracks how much each retrieval axis contributed to the final
/// neighbor selection. Useful for debugging and tuning weights.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct AxisScores {
    /// Motion axis score contribution
    pub motion: f32,
    /// Audio axis score contribution
    pub audio: f32,
    /// Joint axis score contribution
    pub joint: f32,
    /// Transition axis score contribution
    pub transition: f32,
    /// Final fused score (after RRF)
    pub fused: f32,
}

impl AxisScores {
    /// Create scores with only motion axis
    pub fn motion_only(score: f32) -> Self {
        Self {
            motion: score,
            fused: score,
            ..Default::default()
        }
    }

    /// Get the total contribution from all axes
    pub fn total(&self) -> f32 {
        self.motion + self.audio + self.joint + self.transition
    }

    /// Normalize scores to sum to 1
    pub fn normalized(&self) -> Self {
        let total = self.total();
        if total == 0.0 {
            return self.clone();
        }
        Self {
            motion: self.motion / total,
            audio: self.audio / total,
            joint: self.joint / total,
            transition: self.transition / total,
            fused: self.fused,
        }
    }
}

/// Running statistics accumulator using Welford's algorithm
///
/// Numerically stable computation of mean and variance in a single pass.
#[derive(Clone, Debug, Default)]
pub struct WelfordAccumulator {
    count: usize,
    mean: f64,
    m2: f64,
}

impl WelfordAccumulator {
    /// Create a new accumulator
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a value to the accumulator
    pub fn push(&mut self, value: f32) {
        let value = value as f64;
        self.count += 1;
        let delta = value - self.mean;
        self.mean += delta / self.count as f64;
        let delta2 = value - self.mean;
        self.m2 += delta * delta2;
    }

    /// Get the current count
    pub fn count(&self) -> usize {
        self.count
    }

    /// Get the mean (returns 0 if empty)
    pub fn mean(&self) -> f32 {
        self.mean as f32
    }

    /// Get the sample variance (returns 0 if count < 2)
    pub fn variance(&self) -> f32 {
        if self.count < 2 {
            return 0.0;
        }
        (self.m2 / (self.count - 1) as f64) as f32
    }

    /// Get the sample standard deviation
    pub fn std(&self) -> f32 {
        self.variance().sqrt()
    }
}

/// Statistics accumulator that uses OutcomeStats when `neighbors` feature is enabled
///
/// This provides a unified interface for building statistics with optional
/// confidence intervals. When the feature is disabled, falls back to WelfordAccumulator.
#[cfg(feature = "neighbors")]
#[derive(Clone, Debug)]
pub struct StatsAccumulator {
    inner: OutcomeStats,
}

#[cfg(feature = "neighbors")]
impl StatsAccumulator {
    /// Create a new accumulator
    pub fn new() -> Self {
        Self {
            inner: OutcomeStats::new(1),
        }
    }

    /// Add a value to the accumulator
    pub fn push(&mut self, value: f32) {
        self.inner.update(&[value]);
    }

    /// Get the current count
    pub fn count(&self) -> usize {
        self.inner.count() as usize
    }

    /// Get the mean (returns 0 if empty)
    pub fn mean(&self) -> f32 {
        self.inner.mean().map(|m| m[0]).unwrap_or(0.0)
    }

    /// Get the sample standard deviation
    pub fn std(&self) -> f32 {
        self.inner.std().map(|s| s[0]).unwrap_or(0.0)
    }

    /// Get the 90% confidence interval (None if count < 2)
    pub fn confidence_interval_90(&self) -> Option<(f32, f32)> {
        self.inner.confidence_interval(0.90).map(|(lower, upper)| {
            (lower[0], upper[0])
        })
    }

    /// Get the sample count as u64 (for PriorBundle)
    pub fn sample_count(&self) -> u64 {
        self.inner.count()
    }

    /// Merge with another accumulator (parallel Welford)
    pub fn merge(&self, other: &Self) -> Self {
        Self {
            inner: self.inner.merge(&other.inner),
        }
    }
}

#[cfg(feature = "neighbors")]
impl Default for StatsAccumulator {
    fn default() -> Self {
        Self::new()
    }
}

/// Fallback to WelfordAccumulator when `neighbors` feature is disabled
#[cfg(not(feature = "neighbors"))]
pub type StatsAccumulator = WelfordAccumulator;

/// Extension trait to add confidence interval methods to WelfordAccumulator
/// when used as StatsAccumulator fallback
#[cfg(not(feature = "neighbors"))]
impl WelfordAccumulator {
    /// Confidence intervals not available without `neighbors` feature
    pub fn confidence_interval_90(&self) -> Option<(f32, f32)> {
        None
    }

    /// Sample count as u64
    pub fn sample_count(&self) -> u64 {
        self.count as u64
    }

    /// Merge not fully supported without `neighbors` feature
    pub fn merge(&self, _other: &Self) -> Self {
        // Without full merge support, just return self
        // This is a limitation when the feature is disabled
        self.clone()
    }
}

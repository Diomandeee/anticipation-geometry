//! RRF Score Fusion
//!
//! Reciprocal Rank Fusion for combining ranked lists from multiple retrieval axes.

use std::collections::HashMap;

/// Reciprocal Rank Fusion for combining ranked lists
///
/// RRF combines multiple ranked lists by summing reciprocal ranks:
/// ```text
/// RRF_score(d) = Σ weight_i / (k + rank_i(d))
/// ```
///
/// Where k is typically 60 (default), and rank_i(d) is the 1-based rank
/// of document d in list i.
///
/// # Arguments
///
/// * `ranked_lists` - Slice of ranked lists, each containing (id, score) pairs
///   ordered by decreasing score
/// * `weights` - Weight for each list (default 1.0 if shorter than lists)
/// * `k` - RRF constant (typically 60)
///
/// # Returns
///
/// Fused ranked list sorted by RRF score (decreasing)
///
/// # References
///
/// Cormack, G. V., Clarke, C. L., & Buettcher, S. (2009).
/// "Reciprocal rank fusion outperforms condorcet and individual rank learning methods."
pub fn rrf_fuse(
    ranked_lists: &[Vec<(String, f32)>],
    weights: &[f32],
    k: f32,
) -> Vec<(String, f32)> {
    if ranked_lists.is_empty() {
        return Vec::new();
    }

    let mut scores: HashMap<String, f32> = HashMap::new();

    for (list_idx, list) in ranked_lists.iter().enumerate() {
        let weight = weights.get(list_idx).copied().unwrap_or(1.0);

        for (rank, (id, _score)) in list.iter().enumerate() {
            // RRF formula: weight / (k + rank + 1)
            // rank is 0-indexed, so add 1 to make it 1-indexed
            let rrf_score = weight / (k + rank as f32 + 1.0);
            *scores.entry(id.clone()).or_default() += rrf_score;
        }
    }

    // Sort by RRF score (descending)
    let mut results: Vec<_> = scores.into_iter().collect();
    results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    results
}

/// Combine scores using CombSUM
///
/// Simply sums the scores from all lists for each document.
/// Faster than RRF but less robust to score distribution differences.
pub fn combsum_fuse(
    ranked_lists: &[Vec<(String, f32)>],
    weights: &[f32],
) -> Vec<(String, f32)> {
    if ranked_lists.is_empty() {
        return Vec::new();
    }

    let mut scores: HashMap<String, f32> = HashMap::new();

    for (list_idx, list) in ranked_lists.iter().enumerate() {
        let weight = weights.get(list_idx).copied().unwrap_or(1.0);

        for (id, score) in list {
            *scores.entry(id.clone()).or_default() += weight * score;
        }
    }

    let mut results: Vec<_> = scores.into_iter().collect();
    results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    results
}

/// Combine scores using CombMNZ (Multiply by Number of Non-Zero)
///
/// Multiplies combined score by the number of lists containing the document.
/// Rewards documents that appear in multiple lists.
pub fn combmnz_fuse(
    ranked_lists: &[Vec<(String, f32)>],
    weights: &[f32],
) -> Vec<(String, f32)> {
    if ranked_lists.is_empty() {
        return Vec::new();
    }

    let mut scores: HashMap<String, (f32, usize)> = HashMap::new();

    for (list_idx, list) in ranked_lists.iter().enumerate() {
        let weight = weights.get(list_idx).copied().unwrap_or(1.0);

        for (id, score) in list {
            let entry = scores.entry(id.clone()).or_default();
            entry.0 += weight * score;
            entry.1 += 1;
        }
    }

    let mut results: Vec<_> = scores
        .into_iter()
        .map(|(id, (score, count))| (id, score * count as f32))
        .collect();
    results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    results
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rrf_fuse_basic() {
        let list1 = vec![
            ("a".to_string(), 0.9),
            ("b".to_string(), 0.8),
            ("c".to_string(), 0.7),
        ];
        let list2 = vec![
            ("b".to_string(), 0.95),
            ("a".to_string(), 0.85),
            ("d".to_string(), 0.75),
        ];

        let fused = rrf_fuse(&[list1, list2], &[1.0, 1.0], 60.0);

        // "b" should rank highest (rank 2 in list1 + rank 1 in list2)
        // RRF(b) = 1/(60+2) + 1/(60+1) = 0.0161 + 0.0164 = 0.0325
        // RRF(a) = 1/(60+1) + 1/(60+2) = 0.0164 + 0.0161 = 0.0325
        // They should be close, but b has slightly better combined rank
        assert!(fused.len() >= 2);
        // Both a and b should be in top 2
        let top_ids: Vec<_> = fused.iter().take(2).map(|(id, _)| id.as_str()).collect();
        assert!(top_ids.contains(&"a") && top_ids.contains(&"b"));
    }

    #[test]
    fn test_rrf_fuse_empty() {
        let fused = rrf_fuse(&[], &[], 60.0);
        assert!(fused.is_empty());
    }

    #[test]
    fn test_rrf_fuse_single_list() {
        let list = vec![
            ("a".to_string(), 0.9),
            ("b".to_string(), 0.8),
        ];

        let fused = rrf_fuse(&[list], &[1.0], 60.0);
        assert_eq!(fused.len(), 2);
        assert_eq!(fused[0].0, "a"); // Highest rank
    }

    #[test]
    fn test_rrf_fuse_weighted() {
        let list1 = vec![
            ("a".to_string(), 0.9),
        ];
        let list2 = vec![
            ("b".to_string(), 0.9),
        ];

        // Give list2 double weight
        let fused = rrf_fuse(&[list1, list2], &[1.0, 2.0], 60.0);

        // b should have higher score due to higher weight
        assert_eq!(fused[0].0, "b");
    }

    #[test]
    fn test_combsum_fuse() {
        let list1 = vec![
            ("a".to_string(), 0.9),
            ("b".to_string(), 0.5),
        ];
        let list2 = vec![
            ("b".to_string(), 0.8),
            ("a".to_string(), 0.4),
        ];

        let fused = combsum_fuse(&[list1, list2], &[1.0, 1.0]);

        // a: 0.9 + 0.4 = 1.3
        // b: 0.5 + 0.8 = 1.3
        assert_eq!(fused.len(), 2);
    }

    #[test]
    fn test_combmnz_fuse() {
        let list1 = vec![
            ("a".to_string(), 0.5),
        ];
        let list2 = vec![
            ("a".to_string(), 0.5),
        ];
        let list3 = vec![
            ("b".to_string(), 1.0),
        ];

        let fused = combmnz_fuse(&[list1, list2, list3], &[1.0, 1.0, 1.0]);

        // a: (0.5 + 0.5) * 2 = 2.0
        // b: 1.0 * 1 = 1.0
        assert_eq!(fused[0].0, "a");
        assert!(fused[0].1 > fused[1].1);
    }
}

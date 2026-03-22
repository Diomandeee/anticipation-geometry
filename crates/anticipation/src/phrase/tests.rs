//! Unit tests for phrase module

use super::*;

fn create_test_phrase(id: &str, embedding: Vec<f32>) -> MotionPhrase {
    let mut phrase = MotionPhrase::new(
        id.to_string(),
        0.0,
        2.0,
        embedding,
    );
    phrase.outcome = OutcomeMetadata::with_next(
        "idle".to_string(),
        vec![0.0; 64],
    );
    phrase.outcome.delta_commitment = 0.1;
    phrase.outcome.delta_uncertainty = -0.05;
    phrase
}

#[test]
fn test_motion_phrase_creation() {
    let embedding = vec![0.5; 64];
    let phrase = MotionPhrase::new(
        "test_phrase".to_string(),
        0.0,
        2.0,
        embedding,
    );

    assert_eq!(phrase.phrase_id, "test_phrase");
    assert_eq!(phrase.schema_version, PHRASE_SCHEMA_VERSION);
    assert_eq!(phrase.duration, 2.0);
    assert!(phrase.validate().is_ok());
}

#[test]
fn test_motion_phrase_validation() {
    // Empty embedding should fail
    let phrase = MotionPhrase::new(
        "test".to_string(),
        0.0,
        1.0,
        vec![],
    );
    assert!(phrase.validate().is_err());

    // Invalid duration should fail
    let mut phrase = MotionPhrase::new(
        "test".to_string(),
        0.0,
        0.0,
        vec![0.5; 64],
    );
    phrase.duration = 0.0;
    assert!(phrase.validate().is_err());
}

#[test]
fn test_library_insert_and_query() {
    let config = LibraryConfig::default();
    let mut library = MotionPhraseLibrary::new(config);

    // Insert test phrases with varied embeddings (not uniform to avoid cosine similarity quirks)
    let emb1: Vec<f32> = (0..64).map(|i| (i as f32 / 64.0)).collect();
    let emb2: Vec<f32> = (0..64).map(|i| ((63 - i) as f32 / 64.0)).collect(); // Reversed
    let phrase1 = create_test_phrase("p1", emb1.clone());
    let phrase2 = create_test_phrase("p2", emb2);
    library.insert(phrase1).unwrap();
    library.insert(phrase2).unwrap();

    assert_eq!(library.len(), 2);

    // Query with embedding close to p1
    let results = library.query_motion(&emb1, 2);

    assert_eq!(results.len(), 2);
    // p1 should be most similar to its own embedding
    assert_eq!(results[0].0, "p1");
}

#[test]
fn test_library_multi_axis_query() {
    let config = LibraryConfig::default();
    let mut library = MotionPhraseLibrary::new(config);

    // Insert phrases with different embeddings
    for i in 0..5 {
        let motion: Vec<f32> = (0..64).map(|j| (i * 10 + j) as f32 / 100.0).collect();
        let phrase = create_test_phrase(&format!("p{}", i), motion);
        library.insert(phrase).unwrap();
    }

    // Multi-axis query (motion only in this case)
    let query = MultiAxisQuery::motion_only(vec![0.5; 64]);
    let results = library.query_multi_axis(&query, 3);

    assert_eq!(results.len(), 3);
    // Each result should have axis scores
    for (id, score, axis_scores) in &results {
        assert!(!id.is_empty());
        assert!(*score >= 0.0);
        assert!(axis_scores.motion >= 0.0);
    }
}

#[test]
fn test_prior_bundle_construction() {
    let config = LibraryConfig::default();
    let mut library = MotionPhraseLibrary::new(config);

    // Insert diverse phrases
    for i in 0..10 {
        let embedding: Vec<f32> = (0..64).map(|j| (i + j) as f32 / 100.0).collect();
        let mut phrase = create_test_phrase(&format!("p{}", i), embedding);
        phrase.outcome.next_regime = if i % 2 == 0 { "idle" } else { "moving" }.to_string();
        phrase.outcome.delta_commitment = i as f32 * 0.1;
        library.insert(phrase).unwrap();
    }

    let query = MultiAxisQuery::motion_only(vec![0.5; 64]);
    let neighbors = library.query_multi_axis(&query, 5);
    let bundle = library.build_prior_bundle("test", &neighbors);

    assert_eq!(bundle.k, 5);
    assert!(bundle.dispersion >= 0.0);
    assert!(bundle.neighbor_uncertainty >= 0.0 && bundle.neighbor_uncertainty <= 1.0);
    assert!(!bundle.regime_distribution.is_empty());
}

#[test]
fn test_prior_bundle_empty() {
    let bundle = PriorBundle::empty("test");
    assert_eq!(bundle.k, 0);
    assert_eq!(bundle.neighbor_uncertainty, 1.0);
    assert!(!bundle.has_neighbors());
}

#[test]
fn test_prior_bundle_blended_uncertainty() {
    let mut bundle = PriorBundle::new("test");
    bundle.k = 5;
    bundle.neighbor_uncertainty = 0.3;

    let heuristic = 0.6;
    let blended = bundle.blended_uncertainty(heuristic);

    // 0.7 * 0.3 + 0.3 * 0.6 = 0.21 + 0.18 = 0.39
    assert!((blended - 0.39).abs() < 0.01);
}

#[test]
fn test_welford_accumulator() {
    let mut acc = WelfordAccumulator::new();

    acc.push(1.0);
    acc.push(2.0);
    acc.push(3.0);
    acc.push(4.0);
    acc.push(5.0);

    assert_eq!(acc.count(), 5);
    assert!((acc.mean() - 3.0).abs() < 0.01);
    // Sample variance of [1,2,3,4,5] = 2.5, std = sqrt(2.5) ≈ 1.58
    assert!((acc.std() - 1.58).abs() < 0.1);
}

#[test]
fn test_axis_scores() {
    let scores = AxisScores {
        motion: 0.8,
        audio: 0.6,
        joint: 0.4,
        transition: 0.2,
        fused: 0.5,
    };

    assert_eq!(scores.total(), 2.0);

    let normalized = scores.normalized();
    assert!((normalized.motion - 0.4).abs() < 0.01);
    assert!((normalized.audio - 0.3).abs() < 0.01);
}

#[test]
fn test_phrase_embeddings_get_axis() {
    let embeddings = PhraseEmbeddings {
        motion: vec![1.0; 64],
        audio: Some(vec![0.5; 32]),
        joint: None,
        transition: Some(vec![0.3; 32]),
    };

    assert!(embeddings.get_axis(EmbeddingAxis::Motion).is_some());
    assert!(embeddings.get_axis(EmbeddingAxis::Audio).is_some());
    assert!(embeddings.get_axis(EmbeddingAxis::Joint).is_none());
    assert!(embeddings.get_axis(EmbeddingAxis::Transition).is_some());
}

#[test]
fn test_embedding_axis_default_dim() {
    assert_eq!(EmbeddingAxis::Motion.default_dim(), 64);
    assert_eq!(EmbeddingAxis::Audio.default_dim(), 32);
    assert_eq!(EmbeddingAxis::Joint.default_dim(), 42);
    assert_eq!(EmbeddingAxis::Transition.default_dim(), 32);
}

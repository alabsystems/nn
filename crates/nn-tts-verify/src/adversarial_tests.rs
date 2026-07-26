// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

use super::*;

#[test]
fn test_english_confusion_sets_cover_voicing_pairs() {
    let sets = english_confusion_sets();
    let voicing_sets: Vec<_> = sets
        .iter()
        .filter(|s| s.category == ConfusionCategory::VoicingPair)
        .collect();
    assert_eq!(
        voicing_sets.len(),
        6,
        "expected 6 voicing pairs (p/b, t/d, k/g, f/v, s/z, θ/ð)"
    );
    for s in &voicing_sets {
        assert_eq!(
            s.token_ids.len(),
            2,
            "voicing pair {}: expected exactly 2 tokens",
            s.name
        );
        assert_eq!(s.labels.len(), 2);
    }
}

#[test]
fn test_english_confusion_sets_completeness() {
    let sets = english_confusion_sets();
    // We expect ~15 sets covering all categories.
    assert!(
        sets.len() >= 14,
        "expected at least 14 confusion sets, got {}",
        sets.len()
    );

    // Check each category is represented.
    let has_voicing = sets
        .iter()
        .any(|s| s.category == ConfusionCategory::VoicingPair);
    let has_manner = sets
        .iter()
        .any(|s| s.category == ConfusionCategory::MannerConfusion);
    let has_place = sets
        .iter()
        .any(|s| s.category == ConfusionCategory::PlaceConfusion);
    let has_vowel = sets
        .iter()
        .any(|s| s.category == ConfusionCategory::VowelProximity);
    let has_cross = sets
        .iter()
        .any(|s| s.category == ConfusionCategory::CrossLanguage);
    assert!(has_voicing, "missing VoicingPair sets");
    assert!(has_manner, "missing MannerConfusion sets");
    assert!(has_place, "missing PlaceConfusion sets");
    assert!(has_vowel, "missing VowelProximity sets");
    assert!(has_cross, "missing CrossLanguage sets");
}

#[test]
fn test_confusion_set_structural_invariants() {
    // Verify structural invariants of confusion sets:
    // 1. Every set has >= 2 tokens (singletons are meaningless).
    // 2. token_ids and labels have matching length.
    // 3. No duplicate token IDs within a set.
    let sets = english_confusion_sets();
    for s in &sets {
        assert!(
            s.token_ids.len() >= 2,
            "confusion set '{}' has {} tokens (need >= 2)",
            s.name,
            s.token_ids.len()
        );
        assert_eq!(
            s.token_ids.len(),
            s.labels.len(),
            "confusion set '{}': token_ids ({}) and labels ({}) length mismatch",
            s.name,
            s.token_ids.len(),
            s.labels.len()
        );
        // Check no duplicate token IDs within the set.
        let mut seen = std::collections::HashSet::new();
        for &tid in &s.token_ids {
            assert!(
                seen.insert(tid),
                "duplicate token {} in confusion set '{}'",
                tid,
                s.name
            );
        }
    }
}

#[test]
fn test_discover_confusion_sets_identical_embeddings() {
    // Two tokens with identical embeddings should form a confusion set.
    let embed_dim = 4;
    let vocab_size = 3;
    // Token 0 and 1 are identical; token 2 is different.
    let weights = vec![
        1.0, 0.0, 0.0, 0.0, // token 0
        1.0, 0.0, 0.0, 0.0, // token 1 (identical to 0)
        0.0, 1.0, 0.0, 0.0, // token 2 (orthogonal)
    ];

    let sets = discover_confusion_sets(&weights, vocab_size, embed_dim, 0.99, 5).unwrap();
    assert_eq!(
        sets.len(),
        1,
        "identical tokens should form exactly 1 confusion set"
    );
    assert!(sets[0].token_ids.contains(&0));
    assert!(sets[0].token_ids.contains(&1));
    assert!(!sets[0].token_ids.contains(&2));
}

#[test]
fn test_discover_confusion_sets_orthogonal() {
    // All tokens are orthogonal — no confusion sets.
    let embed_dim = 3;
    let vocab_size = 3;
    let weights = vec![
        1.0, 0.0, 0.0, // token 0
        0.0, 1.0, 0.0, // token 1
        0.0, 0.0, 1.0, // token 2
    ];

    let sets = discover_confusion_sets(&weights, vocab_size, embed_dim, 0.9, 5).unwrap();
    assert!(
        sets.is_empty(),
        "orthogonal embeddings should produce no confusion sets"
    );
}

#[test]
fn test_discover_confusion_sets_max_neighbors() {
    // Four identical embeddings with max_neighbors=2 → set should have at most 3 members.
    let embed_dim = 2;
    let vocab_size = 4;
    let weights = vec![
        1.0, 0.0, // token 0
        1.0, 0.0, // token 1
        1.0, 0.0, // token 2
        1.0, 0.0, // token 3
    ];

    let sets = discover_confusion_sets(&weights, vocab_size, embed_dim, 0.99, 2).unwrap();
    assert_eq!(sets.len(), 1);
    // Token 0 gets neighbors 1 and 2 (max=2), tokens 1 and 2 are visited.
    // Token 3 is also identical and visited transitively.
    assert!(
        sets[0].token_ids.len() <= 3,
        "max_neighbors=2 should limit set size"
    );
}

#[test]
fn test_discover_confusion_sets_empty_input() {
    let sets = discover_confusion_sets(&[], 0, 0, 0.9, 5).unwrap();
    assert!(sets.is_empty());
}

#[test]
fn test_discover_confusion_sets_invalid_size() {
    let result = discover_confusion_sets(&[1.0, 2.0], 3, 2, 0.9, 5);
    assert!(result.is_err());
}

#[test]
fn test_embedding_bounds_for_token_set_single_token() {
    // Single token → point bounds (lower == upper).
    let embed_dim = 3;
    let vocab_size = 2;
    let weights = vec![
        1.0, 2.0, 3.0, // token 0
        4.0, 5.0, 6.0, // token 1
    ];

    let (lower, upper) =
        embedding_bounds_for_token_set(&weights, vocab_size, embed_dim, &[0]).unwrap();
    assert_eq!(lower, vec![1.0, 2.0, 3.0]);
    assert_eq!(upper, vec![1.0, 2.0, 3.0]);
}

#[test]
fn test_embedding_bounds_for_token_set_wider_than_single() {
    // Two tokens → bounds span both embeddings.
    let embed_dim = 3;
    let vocab_size = 2;
    let weights = vec![
        1.0, 5.0, 3.0, // token 0
        4.0, 2.0, 6.0, // token 1
    ];

    let (lower, upper) =
        embedding_bounds_for_token_set(&weights, vocab_size, embed_dim, &[0, 1]).unwrap();
    assert_eq!(lower, vec![1.0, 2.0, 3.0]);
    assert_eq!(upper, vec![4.0, 5.0, 6.0]);

    // Verify wider than single token.
    let (lo_single, hi_single) =
        embedding_bounds_for_token_set(&weights, vocab_size, embed_dim, &[0]).unwrap();
    for d in 0..embed_dim {
        assert!(
            lower[d] <= lo_single[d],
            "multi-token lower should be <= single-token lower"
        );
        assert!(
            upper[d] >= hi_single[d],
            "multi-token upper should be >= single-token upper"
        );
    }
}

#[test]
fn test_embedding_bounds_empty_token_ids() {
    let result = embedding_bounds_for_token_set(&[1.0, 2.0], 1, 2, &[]);
    assert!(result.is_err());
}

#[test]
fn test_embedding_bounds_out_of_range() {
    let result = embedding_bounds_for_token_set(&[1.0, 2.0], 1, 2, &[5]);
    assert!(result.is_err());
}

#[test]
fn test_sequence_perturbation_bounds_no_perturbation() {
    // No perturbed positions → all point bounds.
    let embed_dim = 2;
    let vocab_size = 3;
    let weights = vec![
        1.0, 2.0, // token 0
        3.0, 4.0, // token 1
        5.0, 6.0, // token 2
    ];

    let (lower, upper) = sequence_perturbation_bounds(
        &weights,
        vocab_size,
        embed_dim,
        &[0, 1, 2],
        &[], // no perturbation
        &[], // no confusion sets
    )
    .unwrap();

    // All positions are fixed: lower == upper == embedding values.
    assert_eq!(lower, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    assert_eq!(upper, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
}

#[test]
fn test_sequence_perturbation_bounds_with_perturbation() {
    let embed_dim = 2;
    let vocab_size = 3;
    let weights = vec![
        1.0, 2.0, // token 0
        3.0, 4.0, // token 1
        5.0, 6.0, // token 2
    ];

    let cs = ConfusionSet {
        name: "test".into(),
        token_ids: vec![0, 2],
        labels: vec!["t0".into(), "t2".into()],
        category: ConfusionCategory::VoicingPair,
    };

    // Perturb position 0 (token 0, confused with token 2).
    let (lower, upper) = sequence_perturbation_bounds(
        &weights,
        vocab_size,
        embed_dim,
        &[0, 1],
        &[0], // perturb position 0
        &[cs],
    )
    .unwrap();

    // Position 0 perturbed: bounds span token 0 and token 2.
    assert_eq!(lower[0], 1.0); // min(1.0, 5.0)
    assert_eq!(upper[0], 5.0); // max(1.0, 5.0)
    assert_eq!(lower[1], 2.0); // min(2.0, 6.0)
    assert_eq!(upper[1], 6.0); // max(2.0, 6.0)

    // Position 1 fixed: point bounds.
    assert_eq!(lower[2], 3.0);
    assert_eq!(upper[2], 3.0);
    assert_eq!(lower[3], 4.0);
    assert_eq!(upper[3], 4.0);
}

#[test]
fn test_sequence_perturbation_token_not_in_confusion_set() {
    // Token at a perturbed position is not in any confusion set → treated as fixed.
    let embed_dim = 2;
    let vocab_size = 2;
    let weights = vec![
        1.0, 2.0, // token 0
        3.0, 4.0, // token 1
    ];

    let (lower, upper) = sequence_perturbation_bounds(
        &weights,
        vocab_size,
        embed_dim,
        &[0],
        &[0], // perturb position 0
        &[],  // no confusion sets → token 0 not found
    )
    .unwrap();

    // Falls back to point bounds since token not in any confusion set.
    assert_eq!(lower, vec![1.0, 2.0]);
    assert_eq!(upper, vec![1.0, 2.0]);
}

// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Token sampling and log-probability tests for Whisper decode.
//!
//! Extracted from `decode_tests.rs` for file size compliance.

use crate::decode::{compute_log_prob, sample_token};
use rand::rngs::StdRng;
use rand::SeedableRng;

// -- Sample token tests --

#[test]
fn test_sample_token_greedy() {
    let logits = [1.0_f32, 5.0, 2.0, 3.0];
    let (idx, log_prob) = sample_token(&logits, 0.0, None);
    assert_eq!(idx, 1, "should pick index 1 (highest logit)");
    assert!(log_prob.is_finite());
    assert!(log_prob <= 0.0, "log-prob should be non-positive");
}

#[test]
fn test_sample_token_temperature_no_rng() {
    let logits = [1.0_f32, 5.0, 2.0, 3.0];
    // Without RNG, temperature sampling falls back to argmax.
    let (idx, _) = sample_token(&logits, 0.5, None);
    assert_eq!(idx, 1);
}

#[test]
fn test_sample_token_temperature_with_rng() {
    // With uniform logits and RNG, sampling produces diverse tokens.
    let logits = [0.0_f32, 0.0, 0.0, 0.0];
    let mut rng = StdRng::seed_from_u64(42);
    let mut seen = std::collections::HashSet::new();
    for _ in 0..20 {
        let (idx, log_prob) = sample_token(&logits, 1.0, Some(&mut rng));
        assert!(idx < 4);
        assert!(log_prob.is_finite());
        seen.insert(idx);
    }
    assert!(
        seen.len() >= 2,
        "expected diverse samples from uniform distribution, got {seen:?}"
    );
}

#[test]
fn test_sample_token_same_seed_reproducible() {
    let logits = [1.0_f32, 2.0, 3.0, 4.0];
    let mut rng1 = StdRng::seed_from_u64(123);
    let mut rng2 = StdRng::seed_from_u64(123);
    let (idx1, lp1) = sample_token(&logits, 0.8, Some(&mut rng1));
    let (idx2, lp2) = sample_token(&logits, 0.8, Some(&mut rng2));
    assert_eq!(idx1, idx2, "same seed should produce same token");
    assert!((lp1 - lp2).abs() < 1e-6);
}

#[test]
fn test_sample_token_different_seeds_differ() {
    // With uniform logits, different seeds produce different sequences.
    let logits = [0.0_f32, 0.0, 0.0, 0.0];
    let mut results_a = Vec::new();
    let mut results_b = Vec::new();
    let mut rng_a = StdRng::seed_from_u64(1);
    let mut rng_b = StdRng::seed_from_u64(999);
    for _ in 0..10 {
        results_a.push(sample_token(&logits, 1.0, Some(&mut rng_a)).0);
        results_b.push(sample_token(&logits, 1.0, Some(&mut rng_b)).0);
    }
    assert_ne!(
        results_a, results_b,
        "different seeds should produce different sequences"
    );
}

// -- Log-prob computation tests --

#[test]
fn test_compute_log_prob_sum_to_one() {
    let logits = [1.0_f32, 2.0, 3.0];
    let probs: Vec<f32> = (0..3).map(|i| compute_log_prob(&logits, i).exp()).collect();
    let total: f32 = probs.iter().sum();
    assert!(
        (total - 1.0).abs() < 1e-5,
        "softmax probs should sum to 1: {total}"
    );
}

#[test]
fn test_compute_log_prob_ordering() {
    let logits = [1.0_f32, 3.0, 2.0];
    let lp0 = compute_log_prob(&logits, 0);
    let lp1 = compute_log_prob(&logits, 1);
    let lp2 = compute_log_prob(&logits, 2);
    assert!(lp1 > lp0, "highest logit should have highest log-prob");
    assert!(lp1 > lp2);
}

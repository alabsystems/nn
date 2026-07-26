// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! RoPE (Rotary Position Embedding) tests for the Qwen3 RoPECache (#4186).
//!
//! Covers: frequency computation, position-0 identity, norm preservation,
//! adjacent position divergence, dimension pairing, and try_new validation.

use crate::rope_cache::RoPECache;

// ---------------------------------------------------------------------------
// Frequency computation for known positions
// ---------------------------------------------------------------------------

#[test]
fn test_frequency_at_position_1_matches_reference() {
    // At position 1, angle[i] = 1 * theta[i] = 1 / base^(2i/dim)
    let head_dim = 8;
    let base = 10_000.0_f32;
    let cache = RoPECache::new(16, head_dim, base);
    let (cos, sin) = cache.get(1);

    for i in 0..head_dim / 2 {
        let theta = 1.0 / f64::from(base).powf((2 * i) as f64 / head_dim as f64);
        let expected_cos = theta.cos() as f32;
        let expected_sin = theta.sin() as f32;
        assert!(
            (cos[i] - expected_cos).abs() < 1e-6,
            "cos[{i}] at pos 1: expected {expected_cos}, got {}",
            cos[i]
        );
        assert!(
            (sin[i] - expected_sin).abs() < 1e-6,
            "sin[{i}] at pos 1: expected {expected_sin}, got {}",
            sin[i]
        );
    }
}

#[test]
fn test_frequency_at_large_position_matches_reference() {
    let head_dim = 128;
    let base = 1_000_000.0_f32;
    let cache = RoPECache::new(2048, head_dim, base);
    let pos = 1024;
    let (cos, sin) = cache.get(pos);

    // Spot check several frequency indices
    for i in [0, 16, 32, 48, 63] {
        let theta = 1.0 / f64::from(base).powf((2 * i) as f64 / head_dim as f64);
        let angle = pos as f64 * theta;
        let expected_cos = angle.cos() as f32;
        let expected_sin = angle.sin() as f32;
        assert!(
            (cos[i] - expected_cos).abs() < 1e-4,
            "cos[{i}] at pos {pos}: expected {expected_cos}, got {}",
            cos[i]
        );
        assert!(
            (sin[i] - expected_sin).abs() < 1e-4,
            "sin[{i}] at pos {pos}: expected {expected_sin}, got {}",
            sin[i]
        );
    }
}

#[test]
fn test_frequency_decreases_geometrically() {
    // theta[i] = 1/base^(2i/dim) decreases as i increases.
    // At position 1, sin values should generally decrease in magnitude
    // for early indices where angles are small (sin(x) ~ x for small x).
    let head_dim = 128;
    let base = 10_000.0_f32;
    let cache = RoPECache::new(16, head_dim, base);
    let (_, sin) = cache.get(1);

    // The first sin value (highest frequency) should be larger than the last
    // because theta[0]=1.0 > theta[63] ~ 1.155e-4, and sin(theta) ~ theta
    // for small theta.
    assert!(
        sin[0].abs() > sin[head_dim / 2 - 1].abs(),
        "first frequency should produce larger sin value: sin[0]={}, sin[last]={}",
        sin[0],
        sin[head_dim / 2 - 1]
    );
}

// ---------------------------------------------------------------------------
// RoPE application doesn't change vector magnitude (rotation invariant)
// ---------------------------------------------------------------------------

#[test]
fn test_apply_rope_preserves_magnitude_small_vector() {
    let head_dim = 4;
    let cache = RoPECache::new(64, head_dim, 10_000.0);

    let q = [3.0_f32, 4.0, 5.0, 6.0];
    let k = [1.0_f32, 2.0, 3.0, 4.0];

    let q_norm_before: f32 = q.iter().map(|x| x * x).sum::<f32>().sqrt();
    let k_norm_before: f32 = k.iter().map(|x| x * x).sum::<f32>().sqrt();

    // Apply RoPE at multiple positions and check norm preservation
    for pos in [1, 10, 50] {
        let mut q_copy = q;
        let mut k_copy = k;
        let (cos, sin) = cache.get(pos);
        RoPECache::apply_rope(&mut q_copy, &mut k_copy, cos, sin);

        let q_norm_after: f32 = q_copy.iter().map(|x| x * x).sum::<f32>().sqrt();
        let k_norm_after: f32 = k_copy.iter().map(|x| x * x).sum::<f32>().sqrt();

        assert!(
            (q_norm_before - q_norm_after).abs() < 1e-5,
            "q norm changed at pos {pos}: {q_norm_before} -> {q_norm_after}"
        );
        assert!(
            (k_norm_before - k_norm_after).abs() < 1e-5,
            "k norm changed at pos {pos}: {k_norm_before} -> {k_norm_after}"
        );
    }
}

#[test]
fn test_apply_rope_preserves_pair_norm_individually() {
    // Each (x[2i], x[2i+1]) pair undergoes a 2D rotation, preserving its L2 norm.
    let head_dim = 8;
    let cache = RoPECache::new(32, head_dim, 10_000.0);
    let half = head_dim / 2;

    let q: Vec<f32> = (1..=8).map(|i| i as f32).collect();
    let k: Vec<f32> = (10..=17).map(|i| i as f32 * 0.5).collect();

    for pos in [1, 7, 25] {
        let mut q_copy = q.clone();
        let mut k_copy = k.clone();
        let (cos, sin) = cache.get(pos);
        RoPECache::apply_rope(&mut q_copy, &mut k_copy, cos, sin);

        for i in 0..half {
            let orig_q_pair_norm = q[2 * i].hypot(q[2 * i + 1]);
            let rot_q_pair_norm =
                q_copy[2 * i].hypot(q_copy[2 * i + 1]);
            assert!(
                (orig_q_pair_norm - rot_q_pair_norm).abs() < 1e-5,
                "q pair {i} norm changed at pos {pos}: {orig_q_pair_norm} -> {rot_q_pair_norm}"
            );
        }
    }
}

// ---------------------------------------------------------------------------
// RoPE at position 0 is identity
// ---------------------------------------------------------------------------

#[test]
fn test_apply_rope_position_zero_is_identity_small() {
    let cache = RoPECache::new(16, 4, 10_000.0);
    let (cos, sin) = cache.get(0);

    let original = [1.0_f32, 2.0, 3.0, 4.0];
    let mut q = original;
    let mut k = original;
    RoPECache::apply_rope(&mut q, &mut k, cos, sin);

    for i in 0..4 {
        assert!(
            (q[i] - original[i]).abs() < 1e-7,
            "q[{i}] changed at pos 0: {} -> {}",
            original[i],
            q[i]
        );
        assert!(
            (k[i] - original[i]).abs() < 1e-7,
            "k[{i}] changed at pos 0: {} -> {}",
            original[i],
            k[i]
        );
    }
}

#[test]
fn test_apply_rope_position_zero_is_identity_large() {
    let head_dim = 128;
    let cache = RoPECache::new(16, head_dim, 1_000_000.0);
    let (cos, sin) = cache.get(0);

    let original: Vec<f32> = (0..head_dim).map(|i| (i as f32 + 1.0) * 0.7).collect();
    let mut q = original.clone();
    let mut k = original.clone();
    RoPECache::apply_rope(&mut q, &mut k, cos, sin);

    for i in 0..head_dim {
        assert!(
            (q[i] - original[i]).abs() < 1e-5,
            "q[{i}] changed at pos 0 with head_dim={head_dim}"
        );
    }
}

// ---------------------------------------------------------------------------
// Adjacent positions produce different rotations
// ---------------------------------------------------------------------------

#[test]
fn test_adjacent_positions_produce_different_results() {
    let head_dim = 8;
    let cache = RoPECache::new(64, head_dim, 10_000.0);

    let input = [1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];

    for pos in 0..10 {
        let (cos_a, sin_a) = cache.get(pos);
        let (cos_b, sin_b) = cache.get(pos + 1);

        let mut q_a = input;
        let mut k_a = input;
        RoPECache::apply_rope(&mut q_a, &mut k_a, cos_a, sin_a);

        let mut q_b = input;
        let mut k_b = input;
        RoPECache::apply_rope(&mut q_b, &mut k_b, cos_b, sin_b);

        // At least one element should differ
        let any_diff = q_a
            .iter()
            .zip(q_b.iter())
            .any(|(&a, &b)| (a - b).abs() > 1e-7);
        assert!(
            any_diff,
            "positions {pos} and {} should produce different rotations",
            pos + 1
        );
    }
}

#[test]
fn test_widely_separated_positions_differ_significantly() {
    let head_dim = 128;
    let cache = RoPECache::new(2048, head_dim, 10_000.0);

    let input: Vec<f32> = (0..head_dim).map(|i| (i as f32 + 1.0) * 0.1).collect();

    let (cos_0, sin_0) = cache.get(0);
    let (cos_1000, sin_1000) = cache.get(1000);

    let mut q_0 = input.clone();
    let mut k_0 = input.clone();
    RoPECache::apply_rope(&mut q_0, &mut k_0, cos_0, sin_0);

    let mut q_1000 = input.clone();
    let mut k_1000 = input;
    RoPECache::apply_rope(&mut q_1000, &mut k_1000, cos_1000, sin_1000);

    // Compute L2 distance between the two rotated vectors
    let dist_sq: f32 = q_0
        .iter()
        .zip(q_1000.iter())
        .map(|(&a, &b)| (a - b) * (a - b))
        .sum();
    let dist = dist_sq.sqrt();

    assert!(
        dist > 0.1,
        "positions 0 and 1000 should produce significantly different rotations, got L2={dist}"
    );
}

// ---------------------------------------------------------------------------
// Dimension pairing correctness
// ---------------------------------------------------------------------------

#[test]
fn test_dimension_pairing_consecutive_pairs() {
    // RoPECache uses consecutive pairing: (x[0], x[1]), (x[2], x[3]), etc.
    // Verify by checking that modifying only x[0] and x[1] changes only the first pair.
    let head_dim = 8;
    let cache = RoPECache::new(16, head_dim, 10_000.0);
    let (cos, sin) = cache.get(5);

    // Vector with only the first pair non-zero
    let mut q = [1.0_f32, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
    let mut k = [0.0_f32; 8];
    RoPECache::apply_rope(&mut q, &mut k, cos, sin);

    // Only elements 0 and 1 should be affected (the first pair)
    assert!(
        q[0].abs() > 1e-7 || q[1].abs() > 1e-7,
        "first pair should be non-zero after rotation"
    );
    // Elements 2-7 should remain zero (they started as zero pairs)
    for (i, &qi) in q.iter().enumerate().skip(2) {
        assert!(qi.abs() < 1e-7, "q[{i}] should remain ~zero, got {qi}");
    }
}

#[test]
fn test_dimension_pairing_each_pair_uses_correct_frequency() {
    // Pair i uses frequency theta[i] = 1/base^(2i/dim).
    // Verify by comparing apply_rope output with manual computation per pair.
    let head_dim = 6;
    let base = 100.0_f32;
    let pos = 3_usize;
    let cache = RoPECache::new(16, head_dim, base);
    let (cos, sin) = cache.get(pos);

    let q_in = [1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    let mut q = q_in;
    let mut k = [0.0_f32; 6];
    RoPECache::apply_rope(&mut q, &mut k, cos, sin);

    // Manually compute expected output for each pair
    for i in 0..3 {
        let theta = 1.0 / f64::from(base).powf((2 * i) as f64 / head_dim as f64);
        let angle = pos as f64 * theta;
        let c = angle.cos() as f32;
        let s = angle.sin() as f32;

        let x0 = q_in[2 * i];
        let x1 = q_in[2 * i + 1];
        let expected_0 = x0 * c - x1 * s;
        let expected_1 = x0 * s + x1 * c;

        assert!(
            (q[2 * i] - expected_0).abs() < 1e-5,
            "pair {i}, elem 0: expected {expected_0}, got {}",
            q[2 * i]
        );
        assert!(
            (q[2 * i + 1] - expected_1).abs() < 1e-5,
            "pair {i}, elem 1: expected {expected_1}, got {}",
            q[2 * i + 1]
        );
    }
}

#[test]
fn test_different_bases_produce_different_frequencies() {
    // Different base values should produce different rotations at the same position
    let head_dim = 8;
    let pos = 5;
    let input = [1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];

    let cache_10k = RoPECache::new(16, head_dim, 10_000.0);
    let cache_1m = RoPECache::new(16, head_dim, 1_000_000.0);

    let (cos_10k, sin_10k) = cache_10k.get(pos);
    let (cos_1m, sin_1m) = cache_1m.get(pos);

    let mut q_10k = input;
    let mut k_10k = input;
    RoPECache::apply_rope(&mut q_10k, &mut k_10k, cos_10k, sin_10k);

    let mut q_1m = input;
    let mut k_1m = input;
    RoPECache::apply_rope(&mut q_1m, &mut k_1m, cos_1m, sin_1m);

    let any_diff = q_10k
        .iter()
        .zip(q_1m.iter())
        .any(|(&a, &b)| (a - b).abs() > 1e-7);
    assert!(
        any_diff,
        "different bases should produce different rotations"
    );
}

// ---------------------------------------------------------------------------
// Cache accessor correctness
// ---------------------------------------------------------------------------

#[test]
fn test_cache_accessors_match_construction_params() {
    let cache = RoPECache::new(512, 128, 1_000_000.0);
    assert_eq!(cache.max_seq_len(), 512);
    assert_eq!(cache.head_dim(), 128);
    assert_eq!(cache.half_dim(), 64);
    assert!((cache.base() - 1_000_000.0).abs() < f32::EPSILON);
}

#[test]
fn test_get_range_consistency_with_individual_get() {
    let cache = RoPECache::new(64, 16, 10_000.0);
    let (cos_range, sin_range) = cache.get_range(5, 10);

    for i in 0..10 {
        let (cos_i, sin_i) = cache.get(5 + i);
        assert_eq!(cos_range[i].as_slice(), cos_i);
        assert_eq!(sin_range[i].as_slice(), sin_i);
    }
}

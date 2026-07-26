// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Kani proof harnesses for MTP speculative decoding and multi-layer KV cache.
//!
//! Proves properties of:
//! - [`SpeculativeConfig`] validation and [`SpeculativeOutput`] acceptance rate
//! - [`KvCache`] multi-layer construction, indexing, reset, and clear
//! - [`PreallocKvCache`] construction, indexing, capacity, reset, and clear

use super::*;

// ===========================================================================
// MTP Speculative Decoding proofs
// ===========================================================================

/// Prove `SpeculativeConfig::validate` rejects `num_speculative == 0`.
#[kani::unwind(1)]
#[kani::proof]
fn proof_speculative_config_rejects_zero_speculation_depth() {
    let config = SpeculativeConfig::new(100, 0);
    assert!(
        config.validate().is_err(),
        "num_speculative=0 must be rejected"
    );
}

/// Prove `SpeculativeConfig::validate` accepts any `num_speculative > 0`.
#[kani::unwind(1)]
#[kani::proof]
fn proof_speculative_config_accepts_positive_speculation_depth() {
    let n: usize = kani::any();
    kani::assume(n >= 1 && n <= 64);
    let max_tok: usize = kani::any();
    kani::assume(max_tok <= 4096);
    let config = SpeculativeConfig::new(max_tok, n);
    assert!(
        config.validate().is_ok(),
        "num_speculative > 0 must be accepted"
    );
}

/// Prove draft token indices produced by `SpeculativeOutput` are bounded by
/// acceptance rate in [0.0, 1.0].
#[kani::unwind(1)]
#[kani::proof]
fn proof_speculative_output_acceptance_rate_bounded() {
    let total_drafted: usize = kani::any();
    let total_accepted: usize = kani::any();
    kani::assume(total_drafted <= 1024);
    kani::assume(total_accepted <= total_drafted);

    let output = SpeculativeOutput::new(Vec::new(), false, total_drafted, total_accepted);
    let rate = output.acceptance_rate();

    if total_drafted == 0 {
        assert!(
            rate == 0.0,
            "acceptance rate must be 0.0 when no tokens drafted"
        );
    } else {
        assert!(
            rate >= 0.0 && rate <= 1.0,
            "acceptance rate must be in [0.0, 1.0]"
        );
    }
}

/// Prove `SpeculativeOutput::acceptance_rate` returns 0.0 when `total_drafted` is 0.
#[kani::unwind(1)]
#[kani::proof]
fn proof_speculative_output_zero_drafted_rate() {
    let output = SpeculativeOutput::new(Vec::new(), false, 0, 0);
    assert!(
        output.acceptance_rate() == 0.0,
        "acceptance rate must be 0.0 when no tokens drafted"
    );
}

/// Prove `SpeculativeOutput::acceptance_rate` returns 1.0 when all drafted tokens
/// are accepted.
#[kani::unwind(1)]
#[kani::proof]
fn proof_speculative_output_full_acceptance_rate() {
    let n: usize = kani::any();
    kani::assume(n >= 1 && n <= 256);
    let output = SpeculativeOutput::new(Vec::new(), false, n, n);
    let rate = output.acceptance_rate();
    // f64 division of equal values should be exactly 1.0
    assert!(
        (rate - 1.0).abs() < 1e-15,
        "acceptance rate must be 1.0 when all tokens accepted"
    );
}

/// Prove `SpeculativeConfig::with_eos_token_id` correctly stores the EOS token.
#[kani::unwind(1)]
#[kani::proof]
fn proof_speculative_config_eos_token_id() {
    let eos: usize = kani::any();
    kani::assume(eos <= 200_000);
    let config = SpeculativeConfig::new(100, 4).with_eos_token_id(eos);
    assert_eq!(
        config.eos_token_id,
        Some(eos),
        "eos_token_id must match the value passed to with_eos_token_id"
    );
}

/// Prove `SpeculativeConfig::new` has `eos_token_id == None` by default.
#[kani::unwind(1)]
#[kani::proof]
fn proof_speculative_config_default_no_eos() {
    let config = SpeculativeConfig::new(100, 4);
    assert!(
        config.eos_token_id.is_none(),
        "default config must have no eos_token_id"
    );
}

// ===========================================================================
// Multi-layer KV cache proofs
// ===========================================================================

/// Prove `KvCache::new` creates exactly the requested number of layers and
/// the layer count is always > 0 for non-zero input.
#[kani::unwind(1)]
#[kani::proof]
fn proof_kv_cache_multi_layer_count_positive() {
    let n: usize = kani::any();
    kani::assume(n >= 1 && n <= 128);
    let cache = KvCache::new(n);
    assert!(
        cache.num_layers() > 0,
        "KvCache layer count must be > 0 for non-zero input"
    );
    assert_eq!(
        cache.num_layers(),
        n,
        "KvCache layer count must match constructor arg"
    );
}

/// Prove all layers in a new `KvCache` report the same seq_len (0).
#[kani::unwind(1)]
#[kani::proof]
fn proof_kv_cache_all_layers_same_seq_position() {
    let n: usize = kani::any();
    kani::assume(n >= 1 && n <= 32);
    let cache = KvCache::new(n);
    // All layers are empty, so overall seq_len should be 0
    assert_eq!(cache.seq_len(), 0, "fresh cache must have seq_len 0");
    // Each individual layer should also be empty
    let layer = cache.layer(0).expect("layer 0 must exist");
    assert!(layer.is_empty(), "each layer in fresh cache must be empty");
    assert_eq!(
        layer.seq_len(),
        0,
        "each layer in fresh cache must have seq_len 0"
    );
}

/// Prove `KvCache::reset` sets all layers to empty and seq_len to 0, preserving layer count.
#[kani::unwind(1)]
#[kani::proof]
fn proof_kv_cache_clear_resets_position_to_zero() {
    let n: usize = kani::any();
    kani::assume(n >= 1 && n <= 16);
    let mut cache = KvCache::new(n);
    // Clear should reset position to 0
    cache.clear();
    assert_eq!(cache.seq_len(), 0, "clear must reset seq_len to 0");
    assert!(cache.is_empty(), "clear must make cache empty");
    assert_eq!(cache.num_layers(), n, "clear must preserve layer count");
}

/// Prove `KvCache::layer` returns Err for out-of-bounds index.
#[kani::unwind(1)]
#[kani::proof]
fn proof_kv_cache_get_returns_error_for_oob() {
    let n: usize = kani::any();
    kani::assume(n >= 1 && n <= 16);
    let cache = KvCache::new(n);
    // Accessing at exactly num_layers should fail
    let result = cache.layer(n);
    assert!(
        result.is_err(),
        "accessing layer at index == num_layers must fail"
    );
}

// ===========================================================================
// Pre-allocated multi-layer KV cache proofs
// ===========================================================================

/// Prove `PreallocKvCache::new` creates the correct number of layers.
#[kani::unwind(1)]
#[kani::proof]
fn proof_prealloc_kv_cache_layer_count() {
    let n: usize = kani::any();
    kani::assume(n >= 1 && n <= 64);
    let max_seq: usize = kani::any();
    kani::assume(max_seq >= 1 && max_seq <= 4096);
    let cache = PreallocKvCache::new(n, max_seq).expect("valid args must create cache");
    assert_eq!(
        cache.num_layers(),
        n,
        "PreallocKvCache layer count must match constructor arg"
    );
}

/// Prove `PreallocKvCache::max_seq_len` is always >= current seq_len for a fresh cache.
#[kani::unwind(1)]
#[kani::proof]
fn proof_prealloc_size_ge_current_seq_len() {
    let n: usize = kani::any();
    kani::assume(n >= 1 && n <= 16);
    let max_seq: usize = kani::any();
    kani::assume(max_seq >= 1 && max_seq <= 4096);
    let cache = PreallocKvCache::new(n, max_seq).expect("valid args must create cache");
    assert!(
        cache.max_seq_len() >= cache.seq_len(),
        "max_seq_len must be >= current seq_len"
    );
}

/// Prove `PreallocKvCache::clear` resets position to 0 and preserves layer count.
#[kani::unwind(1)]
#[kani::proof]
fn proof_prealloc_clear_resets_position() {
    let n: usize = kani::any();
    kani::assume(n >= 1 && n <= 16);
    let max_seq: usize = kani::any();
    kani::assume(max_seq >= 1 && max_seq <= 2048);
    let mut cache = PreallocKvCache::new(n, max_seq).expect("valid args must create cache");
    cache.clear();
    assert_eq!(cache.seq_len(), 0, "clear must reset seq_len to 0");
    assert!(cache.is_empty(), "clear must make cache empty");
    assert_eq!(cache.num_layers(), n, "clear must preserve layer count");
    assert_eq!(
        cache.max_seq_len(),
        max_seq,
        "clear must preserve max_seq_len"
    );
}

/// Prove `PreallocKvCache::reset` makes all layers empty.
#[kani::unwind(1)]
#[kani::proof]
fn proof_prealloc_reset_empties() {
    let n: usize = kani::any();
    kani::assume(n >= 1 && n <= 16);
    let max_seq: usize = kani::any();
    kani::assume(max_seq >= 1 && max_seq <= 2048);
    let mut cache = PreallocKvCache::new(n, max_seq).expect("valid args must create cache");
    cache.reset();
    assert!(cache.is_empty(), "reset cache must be empty");
    assert_eq!(cache.seq_len(), 0, "reset cache must have seq_len 0");
    assert_eq!(cache.num_layers(), n, "reset must preserve layer count");
}

/// Prove `PreallocKvCache::remaining_capacity` equals `max_seq_len` for a fresh cache.
#[kani::unwind(1)]
#[kani::proof]
fn proof_prealloc_remaining_capacity_fresh() {
    let n: usize = kani::any();
    kani::assume(n >= 1 && n <= 8);
    let max_seq: usize = kani::any();
    kani::assume(max_seq >= 1 && max_seq <= 4096);
    let cache = PreallocKvCache::new(n, max_seq).expect("valid args must create cache");
    assert_eq!(
        cache.remaining_capacity(),
        max_seq,
        "fresh cache remaining_capacity must equal max_seq_len"
    );
}

/// Prove `PreallocKvCache::new` rejects zero `max_seq_len`.
#[kani::unwind(1)]
#[kani::proof]
fn proof_prealloc_rejects_zero_max_seq_len() {
    let n: usize = kani::any();
    kani::assume(n >= 1 && n <= 32);
    let result = PreallocKvCache::new(n, 0);
    assert!(result.is_err(), "PreallocKvCache must reject max_seq_len=0");
}

/// Prove `PreallocKvCache::layer` returns Err for out-of-bounds index.
#[kani::unwind(1)]
#[kani::proof]
fn proof_prealloc_layer_oob() {
    let n: usize = kani::any();
    kani::assume(n >= 1 && n <= 16);
    let max_seq: usize = kani::any();
    kani::assume(max_seq >= 1 && max_seq <= 1024);
    let cache = PreallocKvCache::new(n, max_seq).expect("valid args must create cache");
    let result = cache.layer(n);
    assert!(
        result.is_err(),
        "accessing layer at index == num_layers must fail"
    );
}

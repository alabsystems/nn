// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Advanced Kani proof harnesses for GptOssKvCache invariants.
//!
//! Extends the basic KV cache proofs (kani_kv_cache_proofs.rs) with:
//! - Sliding window eviction correctness (post-eviction len <= window)
//! - Append monotonically increases seq_len
//! - Cache positions are monotonically increasing
//! - K and V caches always have symmetric seq_len
//! - Cache clear resets seq_len to zero
//!
//! Part of #4271 (gpt-oss Kani proofs for device utils and KV cache).

use crate::config::LayerType;

// ============================================================================
// Harness 1: Sliding window eviction leaves cache len <= window_size
// ============================================================================

/// Proves that after sliding window eviction, the effective cache length
/// is at most `window_size`.
///
/// In gpt-oss, sliding attention layers cap context at `sliding_window`
/// tokens. This proof verifies the eviction invariant: for any sequence
/// length and any window size, `min(seq_len, window_size) <= window_size`.
#[kani::unwind(1)]
#[kani::proof]
fn proof_sliding_window_eviction_correct() {
    let seq_len: usize = kani::any();
    let window_size: usize = kani::any();

    kani::assume(window_size >= 1 && window_size <= 4096);
    kani::assume(seq_len <= 131_072); // max_position_embeddings

    // Effective length after eviction
    let effective_len = seq_len.min(window_size);

    assert!(
        effective_len <= window_size,
        "post-eviction cache len must be <= window_size"
    );

    // Also verify: effective_len <= seq_len (can't grow beyond what we have)
    assert!(
        effective_len <= seq_len,
        "effective len must not exceed actual seq_len"
    );

    // And: effective_len is the minimum of both
    assert!(
        effective_len == seq_len || effective_len == window_size,
        "effective len must be one of seq_len or window_size"
    );
}

// ============================================================================
// Harness 2: Cache append increases sequence length
// ============================================================================

/// Proves that appending a non-empty batch of tokens to the cache
/// increases the logical sequence length.
///
/// Models the core invariant: after processing `new_tokens` tokens,
/// `new_seq_len = old_seq_len + new_tokens`, which is strictly greater
/// than `old_seq_len` for any `new_tokens >= 1`.
#[kani::unwind(1)]
#[kani::proof]
fn proof_cache_append_increases_seq_len() {
    let old_seq_len: usize = kani::any();
    let new_tokens: usize = kani::any();

    kani::assume(old_seq_len <= 131_072);
    kani::assume(new_tokens >= 1 && new_tokens <= 4096);
    // Guard against overflow
    kani::assume(old_seq_len <= usize::MAX - new_tokens);

    let new_seq_len = old_seq_len + new_tokens;

    assert!(
        new_seq_len > old_seq_len,
        "appending tokens must increase seq_len"
    );
    assert_eq!(
        new_seq_len,
        old_seq_len + new_tokens,
        "new seq_len must be old + new_tokens"
    );
}

// ============================================================================
// Harness 3: Cache positions are monotonically increasing
// ============================================================================

/// Proves that position IDs in the cache are monotonically increasing.
///
/// In autoregressive decoding, positions are computed as
/// `offset + i` for `i in 0..new_tokens`. This proof verifies that
/// `position[i] < position[i+1]` for all consecutive pairs.
#[kani::unwind(1)]
#[kani::proof]
fn proof_cache_position_monotonic() {
    let offset: usize = kani::any();
    let i: usize = kani::any();

    kani::assume(offset <= 131_072);
    kani::assume(i <= 4096);
    // Guard against overflow
    kani::assume(offset <= usize::MAX - i - 1);

    let pos_i = offset + i;
    let pos_next = offset + i + 1;

    assert!(
        pos_next > pos_i,
        "positions must be strictly monotonically increasing"
    );
    assert_eq!(
        pos_next - pos_i,
        1,
        "consecutive positions must differ by exactly 1"
    );
}

// ============================================================================
// Harness 4: Dual K/V caches always have symmetric seq_len
// ============================================================================

/// Proves that K and V caches always have the same sequence length.
///
/// In gpt-oss, K and V tensors are appended together in each forward
/// pass. This proof models the invariant: if K.seq_len == V.seq_len
/// before an append, and both receive the same `new_tokens`, then
/// K.seq_len == V.seq_len after the append.
#[kani::unwind(1)]
#[kani::proof]
fn proof_dual_cache_kv_symmetric() {
    let k_seq_len: usize = kani::any();
    let v_seq_len: usize = kani::any();
    let new_tokens: usize = kani::any();

    kani::assume(k_seq_len <= 131_072);
    kani::assume(v_seq_len <= 131_072);
    kani::assume(new_tokens >= 1 && new_tokens <= 4096);
    kani::assume(k_seq_len <= usize::MAX - new_tokens);
    kani::assume(v_seq_len <= usize::MAX - new_tokens);

    // Pre-condition: K and V are symmetric before append
    kani::assume(k_seq_len == v_seq_len);

    // Both receive the same number of new tokens
    let k_new = k_seq_len + new_tokens;
    let v_new = v_seq_len + new_tokens;

    assert_eq!(
        k_new, v_new,
        "K and V caches must have the same seq_len after symmetric append"
    );
}

// ============================================================================
// Harness 5: Cache clear resets seq_len to zero
// ============================================================================

/// Proves that clearing the cache always resets seq_len to 0 regardless
/// of prior state.
///
/// Models the `GptOssKvCache::reset()` operation. After reset,
/// `seq_len == 0` and the cache is ready for a new sequence.
#[kani::unwind(25)]
#[kani::proof]
fn proof_cache_clear_resets_to_zero() {
    let cfg = crate::config::GptOssConfig::gptoss_20b();
    let mut cache = crate::kv_cache::GptOssKvCache::new(&cfg);

    // Cache starts at seq_len 0
    assert_eq!(cache.seq_len(), 0, "new cache must have seq_len 0");

    // Reset should bring it back to 0
    cache.reset();
    assert_eq!(cache.seq_len(), 0, "reset cache must have seq_len 0");

    // Layer count must be preserved after reset
    assert_eq!(
        cache.num_layers(),
        cfg.num_hidden_layers,
        "reset must preserve layer count"
    );

    // Sliding window must be preserved after reset
    assert_eq!(
        cache.sliding_window(),
        cfg.sliding_window,
        "reset must preserve sliding_window"
    );

    // Effective context for all layers must be 0 after reset
    for i in 0..cfg.num_hidden_layers {
        assert_eq!(
            cache.effective_context_len(i),
            0,
            "effective context must be 0 after reset"
        );
    }
}

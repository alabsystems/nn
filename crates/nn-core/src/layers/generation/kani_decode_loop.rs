// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Kani proof harnesses for the decode loop (`DecodeContext`).
//!
//! Proves properties of `DecodeContext` bookkeeping:
//! - `remaining_capacity` + `seq_len` <= `max_seq_len`
//! - `is_full` iff remaining_capacity == 0
//! - `reset` and `clear` restore clean state

use super::*;

// --- DecodeContext capacity arithmetic ---

/// A minimal mock cache backend for Kani proofs.
/// Tracks seq_len and num_layers without DynTensor dependencies.
struct MockCache {
    seq_len: usize,
    num_layers: usize,
}

impl MockCache {
    fn new(num_layers: usize) -> Self {
        Self {
            seq_len: 0,
            num_layers,
        }
    }
}

impl KvCacheBackend for MockCache {
    fn layer_backend_mut(
        &mut self,
        index: usize,
    ) -> crate::Result<&mut dyn crate::layers::generation::KvCacheLayerBackend> {
        Err(crate::TensorError::InvalidShape(format!(
            "MockCache has no real layers (requested index {index})"
        )))
    }

    fn num_layers(&self) -> usize {
        self.num_layers
    }

    fn seq_len(&self) -> usize {
        self.seq_len
    }

    fn reset(&mut self) {
        self.seq_len = 0;
    }

    fn clear(&mut self) {
        self.seq_len = 0;
    }
}

/// Prove `remaining_capacity` + `seq_len` <= `max_seq_len` always holds.
/// The invariant uses `saturating_sub`, so remaining_capacity never underflows.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(8)]
fn proof_remaining_capacity_invariant() {
    let max_seq_len: usize = kani::any();
    kani::assume(max_seq_len >= 1 && max_seq_len <= 4096);

    let seq_len: usize = kani::any();
    kani::assume(seq_len <= max_seq_len + 10); // allow over-max to test saturation

    let mut cache = MockCache::new(1);
    cache.seq_len = seq_len;
    let ctx = DecodeContext::new(cache, max_seq_len);

    let remaining = ctx.remaining_capacity();
    // remaining_capacity uses saturating_sub, so it's always <= max_seq_len.
    assert!(
        remaining <= max_seq_len,
        "remaining_capacity must be <= max_seq_len"
    );
    // If seq_len <= max_seq_len, the sum is exact.
    if seq_len <= max_seq_len {
        assert_eq!(
            remaining + seq_len,
            max_seq_len,
            "remaining + seq_len must equal max_seq_len when within bounds"
        );
    } else {
        // Saturating subtraction: remaining == 0 when over.
        assert_eq!(
            remaining, 0,
            "remaining must be 0 when seq_len > max_seq_len"
        );
    }
}

/// Prove `is_full` iff `remaining_capacity == 0`.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(8)]
fn proof_is_full_iff_remaining_zero() {
    let max_seq_len: usize = kani::any();
    kani::assume(max_seq_len >= 1 && max_seq_len <= 4096);

    let seq_len: usize = kani::any();
    kani::assume(seq_len <= max_seq_len + 10);

    let mut cache = MockCache::new(1);
    cache.seq_len = seq_len;
    let ctx = DecodeContext::new(cache, max_seq_len);

    assert_eq!(
        ctx.is_full(),
        ctx.remaining_capacity() == 0,
        "is_full must be true iff remaining_capacity is 0"
    );
}

/// Prove `is_full` returns true when seq_len >= max_seq_len.
#[kani::unwind(1)]
#[kani::proof]
fn proof_is_full_at_max() {
    let max_seq_len: usize = kani::any();
    kani::assume(max_seq_len >= 1 && max_seq_len <= 4096);

    let mut cache = MockCache::new(1);
    cache.seq_len = max_seq_len;
    let ctx = DecodeContext::new(cache, max_seq_len);

    assert!(
        ctx.is_full(),
        "is_full must be true when seq_len == max_seq_len"
    );
}

/// Prove `is_full` returns false when seq_len < max_seq_len.
#[kani::unwind(1)]
#[kani::proof]
fn proof_not_full_below_max() {
    let max_seq_len: usize = kani::any();
    kani::assume(max_seq_len >= 2 && max_seq_len <= 4096);

    let seq_len: usize = kani::any();
    kani::assume(seq_len < max_seq_len);

    let mut cache = MockCache::new(1);
    cache.seq_len = seq_len;
    let ctx = DecodeContext::new(cache, max_seq_len);

    assert!(
        !ctx.is_full(),
        "is_full must be false when seq_len < max_seq_len"
    );
}

/// Prove `reset` zeroes the generated_count.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(8)]
fn proof_reset_clears_generated_count() {
    let max_seq_len: usize = kani::any();
    kani::assume(max_seq_len >= 1 && max_seq_len <= 4096);

    let mut cache = MockCache::new(1);
    cache.seq_len = 42;
    let mut ctx = DecodeContext::new(cache, max_seq_len);

    // Simulate some generation.
    ctx.generated_count = 42;

    ctx.reset();
    assert_eq!(
        ctx.generated_count(),
        0,
        "generated_count must be 0 after reset"
    );
    assert_eq!(ctx.seq_len(), 0, "seq_len must be 0 after reset");
}

/// Prove `clear` zeroes the generated_count and seq_len.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(8)]
fn proof_clear_clears_state() {
    let max_seq_len: usize = kani::any();
    kani::assume(max_seq_len >= 1 && max_seq_len <= 4096);

    let mut cache = MockCache::new(1);
    cache.seq_len = 100;
    let mut ctx = DecodeContext::new(cache, max_seq_len);
    ctx.generated_count = 50;

    ctx.clear();
    assert_eq!(
        ctx.generated_count(),
        0,
        "generated_count must be 0 after clear"
    );
    assert_eq!(ctx.seq_len(), 0, "seq_len must be 0 after clear");
}

/// Prove `max_seq_len` accessor returns the constructor argument.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(8)]
fn proof_max_seq_len_accessor() {
    let max_seq_len: usize = kani::any();
    kani::assume(max_seq_len <= 131072);

    let cache = MockCache::new(12);
    let ctx = DecodeContext::new(cache, max_seq_len);

    assert_eq!(
        ctx.max_seq_len(),
        max_seq_len,
        "max_seq_len must return constructor value"
    );
}

/// Prove `num_layers` delegates to the cache backend.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(8)]
fn proof_num_layers_delegates() {
    let num_layers: usize = kani::any();
    kani::assume(num_layers >= 1 && num_layers <= 128);

    let cache = MockCache::new(num_layers);
    let ctx = DecodeContext::new(cache, 2048);

    assert_eq!(
        ctx.num_layers(),
        num_layers,
        "num_layers must delegate to cache backend"
    );
}

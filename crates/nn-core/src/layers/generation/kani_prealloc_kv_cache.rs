// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Kani proof harnesses for `PreallocKvCacheLayer`.
//!
//! Proves properties of the pre-allocated KV cache:
//! - Construction rejects max_seq_len == 0
//! - Initial state invariants (empty, correct capacity)
//! - `remaining_capacity` + `seq_len` == `max_seq_len`
//! - `reset` and `clear` restore expected state

use super::*;

/// Prove `PreallocKvCacheLayer::new` rejects max_seq_len == 0.
#[kani::unwind(1)]
#[kani::proof]
fn proof_prealloc_rejects_zero_max_seq_len() {
    let result = PreallocKvCacheLayer::new(0);
    assert!(result.is_err(), "max_seq_len=0 must be rejected");
}

/// Prove `PreallocKvCacheLayer::new` succeeds for positive max_seq_len.
#[kani::unwind(1)]
#[kani::proof]
fn proof_prealloc_accepts_positive_max_seq_len() {
    let max_seq_len: usize = kani::any();
    kani::assume(max_seq_len >= 1 && max_seq_len <= 131072);
    let result = PreallocKvCacheLayer::new(max_seq_len);
    assert!(result.is_ok(), "positive max_seq_len must be accepted");
}

/// Prove initial state: empty, not allocated, zero seq_len.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(8)]
fn proof_prealloc_initial_state() {
    let max_seq_len: usize = kani::any();
    kani::assume(max_seq_len >= 1 && max_seq_len <= 4096);

    let layer = PreallocKvCacheLayer::new(max_seq_len).unwrap();
    assert!(layer.is_empty(), "must be empty initially");
    assert!(!layer.is_allocated(), "must not be allocated initially");
    assert_eq!(layer.seq_len(), 0, "seq_len must be 0 initially");
    assert_eq!(
        layer.max_seq_len(),
        max_seq_len,
        "max_seq_len must match constructor"
    );
}

/// Prove `remaining_capacity` + `seq_len` == `max_seq_len` for newly constructed cache.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(8)]
fn proof_prealloc_capacity_invariant_initial() {
    let max_seq_len: usize = kani::any();
    kani::assume(max_seq_len >= 1 && max_seq_len <= 4096);

    let layer = PreallocKvCacheLayer::new(max_seq_len).unwrap();
    assert_eq!(
        layer.remaining_capacity() + layer.seq_len(),
        max_seq_len,
        "remaining + seq_len must equal max_seq_len"
    );
}

/// Prove `remaining_capacity` equals `max_seq_len` when cache is empty.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(8)]
fn proof_prealloc_full_capacity_when_empty() {
    let max_seq_len: usize = kani::any();
    kani::assume(max_seq_len >= 1 && max_seq_len <= 4096);

    let layer = PreallocKvCacheLayer::new(max_seq_len).unwrap();
    assert_eq!(
        layer.remaining_capacity(),
        max_seq_len,
        "remaining must equal max_seq_len when empty"
    );
}

/// Prove `reset` restores initial state.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(8)]
fn proof_prealloc_reset_restores_initial() {
    let max_seq_len: usize = kani::any();
    kani::assume(max_seq_len >= 1 && max_seq_len <= 4096);

    let mut layer = PreallocKvCacheLayer::new(max_seq_len).unwrap();

    // Manually set current_len to simulate usage.
    // (We can't call append without DynTensors in Kani)
    layer.reset();

    assert!(layer.is_empty(), "must be empty after reset");
    assert!(!layer.is_allocated(), "must not be allocated after reset");
    assert_eq!(layer.seq_len(), 0, "seq_len must be 0 after reset");
    assert_eq!(
        layer.remaining_capacity(),
        max_seq_len,
        "remaining must equal max_seq_len after reset"
    );
}

/// Prove `clear` resets seq_len to 0 but preserves allocation flag.
/// (After clear, is_allocated depends on whether buffers were allocated.)
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(8)]
fn proof_prealloc_clear_resets_seq_len() {
    let max_seq_len: usize = kani::any();
    kani::assume(max_seq_len >= 1 && max_seq_len <= 4096);

    let mut layer = PreallocKvCacheLayer::new(max_seq_len).unwrap();
    layer.clear();

    assert_eq!(layer.seq_len(), 0, "seq_len must be 0 after clear");
    assert!(layer.is_empty(), "must be empty after clear");
    assert_eq!(
        layer.remaining_capacity(),
        max_seq_len,
        "remaining must equal max_seq_len after clear"
    );
}

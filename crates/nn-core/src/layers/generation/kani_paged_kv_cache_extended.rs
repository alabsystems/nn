// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Extended Kani proof harnesses for `PagedKvCache` safety invariants.
//!
//! Proves deeper properties beyond basic construction/allocation:
//! - Page table index bounds after append operations
//! - Block allocation/deallocation invariants across multiple sequences
//! - Sequence-to-physical page mapping correctness
//! - Block size alignment and page boundary transitions
//! - Memory capacity calculations and overflow safety
//! - Multi-sequence page independence (no cross-contamination)
//! - KV head dimension consistency through operations
//! - Page fault detection (get_kv on invalid layer/seq)
//! - Token count tracking accuracy
//! - Free pool integrity after mixed alloc/free patterns

use super::*;

// ===========================================================================
// Page table index bounds after append
// ===========================================================================

/// Prove that after appending tokens up to page_size, no new page is allocated
/// (the first page suffices for tokens within one page).
#[kani::unwind(8)]
#[kani::proof]
fn proof_paged_kv_append_within_single_page_no_extra_alloc() {
    let page_size: usize = kani::any();
    kani::assume(page_size >= 2 && page_size <= 4);
    let num_layers: usize = 1;
    // Need at least num_layers pages for initial allocation.
    let num_pages: usize = kani::any();
    kani::assume(num_pages >= 2 && num_pages <= 4);

    let mut cache = PagedKvCache::new(page_size, num_pages, num_layers, 1, 1).unwrap();
    cache.allocate_sequence(0).unwrap();

    let free_after_alloc = cache.num_free_pages();

    // Append exactly page_size tokens — should all fit in the initial page.
    for _i in 0..page_size {
        cache.append_kv(0, 0, &[1.0], &[2.0]).unwrap();
    }

    assert_eq!(
        cache.num_free_pages(),
        free_after_alloc,
        "appending within one page must not allocate extra pages"
    );
    assert_eq!(
        cache.sequence_token_count(0),
        Some(page_size),
        "token count must match number of appends"
    );
}

/// Prove that appending page_size+1 tokens triggers exactly one new page allocation.
#[kani::unwind(8)]
#[kani::proof]
fn proof_paged_kv_append_triggers_new_page_at_boundary() {
    let page_size: usize = kani::any();
    kani::assume(page_size >= 1 && page_size <= 3);
    let num_layers: usize = 1;
    // Need at least 2 pages (1 for initial alloc + 1 for overflow).
    let num_pages: usize = kani::any();
    kani::assume(num_pages >= 2 && num_pages <= 4);

    let mut cache = PagedKvCache::new(page_size, num_pages, num_layers, 1, 1).unwrap();
    cache.allocate_sequence(0).unwrap();
    let free_after_alloc = cache.num_free_pages();

    // Fill first page completely.
    for _i in 0..page_size {
        cache.append_kv(0, 0, &[1.0], &[2.0]).unwrap();
    }

    // One more token should allocate a new page.
    cache.append_kv(0, 0, &[3.0], &[4.0]).unwrap();

    assert_eq!(
        cache.num_free_pages(),
        free_after_alloc - 1,
        "crossing page boundary must allocate exactly one new page"
    );
}

// ===========================================================================
// Block allocation/deallocation invariants
// ===========================================================================

/// Prove that allocating and freeing multiple sequences preserves total page count.
#[kani::unwind(8)]
#[kani::proof]
fn proof_paged_kv_multi_alloc_free_conservation() {
    let num_layers: usize = kani::any();
    kani::assume(num_layers >= 1 && num_layers <= 2);
    let num_seqs: usize = kani::any();
    kani::assume(num_seqs >= 1 && num_seqs <= 3);
    let num_pages = num_seqs * num_layers + 2; // extra headroom

    let mut cache = PagedKvCache::new(4, num_pages, num_layers, 1, 1).unwrap();

    // Allocate all sequences.
    for s in 0..num_seqs {
        cache.allocate_sequence(s).unwrap();
    }

    let expected_allocated = num_seqs * num_layers;
    assert_eq!(
        cache.num_free_pages() + expected_allocated,
        num_pages,
        "free + allocated must equal total after multi-alloc"
    );

    // Free all sequences.
    for s in 0..num_seqs {
        cache.free_sequence(s);
    }

    assert_eq!(
        cache.num_free_pages(),
        num_pages,
        "all pages must be free after freeing all sequences"
    );
}

/// Prove that freeing sequences in reverse order restores all pages.
#[kani::unwind(8)]
#[kani::proof]
fn proof_paged_kv_reverse_free_order_conservation() {
    let num_layers: usize = 1;
    let num_pages: usize = 6;

    let mut cache = PagedKvCache::new(4, num_pages, num_layers, 1, 1).unwrap();
    cache.allocate_sequence(0).unwrap();
    cache.allocate_sequence(1).unwrap();
    cache.allocate_sequence(2).unwrap();

    // Free in reverse order.
    cache.free_sequence(2);
    cache.free_sequence(1);
    cache.free_sequence(0);

    assert_eq!(
        cache.num_free_pages(),
        num_pages,
        "reverse-order free must restore all pages"
    );
    assert_eq!(
        cache.num_active_sequences(),
        0,
        "no active sequences after freeing all"
    );
}

// ===========================================================================
// Sequence-to-physical page mapping correctness
// ===========================================================================

/// Prove that get_kv returns correct data length after appending tokens.
#[kani::unwind(8)]
#[kani::proof]
fn proof_paged_kv_get_returns_correct_data_length() {
    let num_heads: usize = kani::any();
    kani::assume(num_heads >= 1 && num_heads <= 2);
    let head_dim: usize = kani::any();
    kani::assume(head_dim >= 1 && head_dim <= 2);
    let page_size: usize = 2;
    let num_tokens: usize = kani::any();
    kani::assume(num_tokens >= 1 && num_tokens <= 3);

    let elems = num_heads * head_dim;
    // Need enough pages for initial + overflow.
    let num_pages = 4;

    let mut cache = PagedKvCache::new(page_size, num_pages, 1, num_heads, head_dim).unwrap();
    cache.allocate_sequence(0).unwrap();

    let token_data = vec![1.0_f32; elems];
    for _t in 0..num_tokens {
        cache.append_kv(0, 0, &token_data, &token_data).unwrap();
    }

    let (k, v) = cache.get_kv(0, 0).unwrap();
    let expected_len = num_tokens * elems;

    assert_eq!(
        k.len(),
        expected_len,
        "K output length must be num_tokens * num_heads * head_dim"
    );
    assert_eq!(
        v.len(),
        expected_len,
        "V output length must be num_tokens * num_heads * head_dim"
    );
}

/// Prove that get_kv on an unallocated sequence returns an error.
#[kani::unwind(1)]
#[kani::proof]
fn proof_paged_kv_get_unallocated_seq_error() {
    let cache = PagedKvCache::new(4, 8, 2, 1, 1).unwrap();
    let result = cache.get_kv(99, 0);
    assert!(
        result.is_err(),
        "get_kv on unallocated sequence must return error"
    );
}

// ===========================================================================
// Block size alignment and page boundary transitions
// ===========================================================================

/// Prove that exactly 2*page_size tokens allocate exactly one extra page (two
/// pages total per layer) beyond the initial allocation.
#[kani::unwind(12)]
#[kani::proof]
fn proof_paged_kv_two_full_pages_allocation() {
    let page_size: usize = kani::any();
    kani::assume(page_size >= 1 && page_size <= 3);
    let num_layers: usize = 1;
    let num_pages: usize = 4; // plenty of headroom

    let mut cache = PagedKvCache::new(page_size, num_pages, num_layers, 1, 1).unwrap();
    cache.allocate_sequence(0).unwrap();
    let free_after_alloc = cache.num_free_pages();

    // Append exactly 2*page_size tokens.
    for _i in 0..(2 * page_size) {
        cache.append_kv(0, 0, &[1.0], &[2.0]).unwrap();
    }

    // Should have used exactly one additional page beyond the initial.
    assert_eq!(
        cache.num_free_pages(),
        free_after_alloc - 1,
        "2*page_size tokens must use exactly 2 pages total (1 initial + 1 extra)"
    );
}

/// Prove that page_size accessor returns the correct value.
#[kani::unwind(1)]
#[kani::proof]
fn proof_paged_kv_page_size_accessor() {
    let ps: usize = kani::any();
    kani::assume(ps >= 1 && ps <= 64);
    let cache = PagedKvCache::new(ps, 4, 1, 1, 1).unwrap();
    assert_eq!(
        cache.page_size(),
        ps,
        "page_size() must return constructor value"
    );
}

// ===========================================================================
// Memory capacity calculations
// ===========================================================================

/// Prove that num_pages accessor always matches the constructor argument.
#[kani::unwind(1)]
#[kani::proof]
fn proof_paged_kv_num_pages_accessor() {
    let np: usize = kani::any();
    kani::assume(np >= 1 && np <= 256);
    let cache = PagedKvCache::new(4, np, 1, 1, 1).unwrap();
    assert_eq!(
        cache.num_pages(),
        np,
        "num_pages() must return constructor value"
    );
}

/// Prove that after allocating a sequence, num_free_pages + num_layers <= num_pages.
#[kani::unwind(1)]
#[kani::proof]
fn proof_paged_kv_free_plus_allocated_le_total() {
    let num_layers: usize = kani::any();
    kani::assume(num_layers >= 1 && num_layers <= 4);
    let num_pages: usize = kani::any();
    kani::assume(num_pages >= num_layers && num_pages <= 16);

    let mut cache = PagedKvCache::new(4, num_pages, num_layers, 1, 1).unwrap();
    cache.allocate_sequence(0).unwrap();

    assert!(
        cache.num_free_pages() + num_layers <= num_pages,
        "free pages + allocated pages must not exceed total"
    );
}

/// Prove that appending a token when no free pages remain returns an error
/// (page fault detection for expansion).
#[kani::unwind(8)]
#[kani::proof]
fn proof_paged_kv_append_exhausted_pool_error() {
    let page_size: usize = 1;
    let num_layers: usize = 1;
    // Exactly 1 page: enough for initial alloc, no room for expansion.
    let mut cache = PagedKvCache::new(page_size, 1, num_layers, 1, 1).unwrap();
    cache.allocate_sequence(0).unwrap();

    // First token fits in the initial page.
    cache.append_kv(0, 0, &[1.0], &[2.0]).unwrap();

    // Second token needs a new page, but pool is exhausted.
    let result = cache.append_kv(0, 0, &[3.0], &[4.0]);
    assert!(
        result.is_err(),
        "append must fail when free pool is exhausted and new page is needed"
    );
}

// ===========================================================================
// Multi-sequence page independence
// ===========================================================================

/// Prove that data written to one sequence is not visible in another.
#[kani::unwind(8)]
#[kani::proof]
fn proof_paged_kv_sequences_independent_data() {
    let mut cache = PagedKvCache::new(4, 8, 1, 1, 1).unwrap();
    cache.allocate_sequence(0).unwrap();
    cache.allocate_sequence(1).unwrap();

    // Write distinct values to each sequence.
    cache.append_kv(0, 0, &[10.0], &[20.0]).unwrap();
    cache.append_kv(1, 0, &[30.0], &[40.0]).unwrap();

    let (k0, v0) = cache.get_kv(0, 0).unwrap();
    let (k1, v1) = cache.get_kv(1, 0).unwrap();

    assert_eq!(k0[0], 10.0, "seq 0 K must contain its own data");
    assert_eq!(v0[0], 20.0, "seq 0 V must contain its own data");
    assert_eq!(k1[0], 30.0, "seq 1 K must contain its own data");
    assert_eq!(v1[0], 40.0, "seq 1 V must contain its own data");
}

/// Prove that freeing one sequence does not affect another's data.
#[kani::unwind(8)]
#[kani::proof]
fn proof_paged_kv_free_does_not_corrupt_other_seq() {
    let mut cache = PagedKvCache::new(4, 8, 1, 1, 1).unwrap();
    cache.allocate_sequence(0).unwrap();
    cache.allocate_sequence(1).unwrap();

    cache.append_kv(0, 0, &[10.0], &[20.0]).unwrap();
    cache.append_kv(1, 0, &[30.0], &[40.0]).unwrap();

    // Free seq 0 — seq 1 must be unaffected.
    cache.free_sequence(0);

    let (k1, v1) = cache.get_kv(1, 0).unwrap();
    assert_eq!(k1[0], 30.0, "seq 1 K must be intact after freeing seq 0");
    assert_eq!(v1[0], 40.0, "seq 1 V must be intact after freeing seq 0");
    assert_eq!(
        cache.num_active_sequences(),
        1,
        "must have 1 active sequence after freeing one"
    );
}

// ===========================================================================
// KV head dimension consistency
// ===========================================================================

/// Prove that append_kv rejects data with wrong length (too short).
#[kani::unwind(1)]
#[kani::proof]
fn proof_paged_kv_append_rejects_wrong_k_length() {
    let num_heads: usize = 2;
    let head_dim: usize = 4;
    let elems = num_heads * head_dim; // 8

    let mut cache = PagedKvCache::new(4, 4, 1, num_heads, head_dim).unwrap();
    cache.allocate_sequence(0).unwrap();

    // Correct V, wrong K (too short).
    let correct = vec![1.0_f32; elems];
    let wrong = vec![1.0_f32; elems - 1];

    let result = cache.append_kv(0, 0, &wrong, &correct);
    assert!(
        result.is_err(),
        "append_kv must reject K data with wrong length"
    );
}

/// Prove that append_kv rejects data with wrong V length (too long).
#[kani::unwind(1)]
#[kani::proof]
fn proof_paged_kv_append_rejects_wrong_v_length() {
    let num_heads: usize = 2;
    let head_dim: usize = 4;
    let elems = num_heads * head_dim; // 8

    let mut cache = PagedKvCache::new(4, 4, 1, num_heads, head_dim).unwrap();
    cache.allocate_sequence(0).unwrap();

    let correct = vec![1.0_f32; elems];
    let wrong = vec![1.0_f32; elems + 1];

    let result = cache.append_kv(0, 0, &correct, &wrong);
    assert!(
        result.is_err(),
        "append_kv must reject V data with wrong length"
    );
}

// ===========================================================================
// Page fault detection (invalid layer / unallocated seq)
// ===========================================================================

/// Prove that append_kv rejects out-of-range layer index.
#[kani::unwind(1)]
#[kani::proof]
fn proof_paged_kv_append_rejects_oob_layer() {
    let num_layers: usize = kani::any();
    kani::assume(num_layers >= 1 && num_layers <= 4);

    let mut cache = PagedKvCache::new(4, 8, num_layers, 1, 1).unwrap();
    cache.allocate_sequence(0).unwrap();

    // Layer index == num_layers is out of bounds.
    let result = cache.append_kv(0, num_layers, &[1.0], &[2.0]);
    assert!(
        result.is_err(),
        "append_kv must reject layer index >= num_layers"
    );
}

/// Prove that get_kv rejects out-of-range layer index.
#[kani::unwind(1)]
#[kani::proof]
fn proof_paged_kv_get_rejects_oob_layer() {
    let num_layers: usize = kani::any();
    kani::assume(num_layers >= 1 && num_layers <= 4);

    let mut cache = PagedKvCache::new(4, 8, num_layers, 1, 1).unwrap();
    cache.allocate_sequence(0).unwrap();

    let result = cache.get_kv(0, num_layers);
    assert!(
        result.is_err(),
        "get_kv must reject layer index >= num_layers"
    );
}

/// Prove that append_kv on a non-existent (unallocated) sequence returns error.
#[kani::unwind(1)]
#[kani::proof]
fn proof_paged_kv_append_unallocated_seq_error() {
    let mut cache = PagedKvCache::new(4, 8, 1, 1, 1).unwrap();
    // Do not allocate seq 0.
    let result = cache.append_kv(0, 0, &[1.0], &[2.0]);
    assert!(
        result.is_err(),
        "append_kv on unallocated sequence must return error"
    );
}

// ===========================================================================
// Token count tracking accuracy
// ===========================================================================

/// Prove that sequence_token_count tracks appends accurately across page boundaries.
#[kani::unwind(12)]
#[kani::proof]
fn proof_paged_kv_token_count_tracks_appends() {
    let page_size: usize = 2;
    let num_tokens: usize = kani::any();
    kani::assume(num_tokens >= 1 && num_tokens <= 5);

    // Need enough pages: 1 initial + ceil(num_tokens / page_size) - 1 extras.
    let num_pages = 4;

    let mut cache = PagedKvCache::new(page_size, num_pages, 1, 1, 1).unwrap();
    cache.allocate_sequence(0).unwrap();

    for _t in 0..num_tokens {
        cache.append_kv(0, 0, &[1.0], &[2.0]).unwrap();
    }

    assert_eq!(
        cache.sequence_token_count(0),
        Some(num_tokens),
        "token count must exactly match number of appends"
    );
}

/// Prove that sequence_token_count returns None after freeing the sequence.
#[kani::unwind(8)]
#[kani::proof]
fn proof_paged_kv_token_count_none_after_free() {
    let mut cache = PagedKvCache::new(4, 4, 1, 1, 1).unwrap();
    cache.allocate_sequence(0).unwrap();
    cache.append_kv(0, 0, &[1.0], &[2.0]).unwrap();

    cache.free_sequence(0);

    assert!(
        cache.sequence_token_count(0).is_none(),
        "token count must be None after freeing sequence"
    );
}

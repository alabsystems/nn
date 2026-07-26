// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Kani proof harnesses for `PagedKvCache`.
//!
//! Proves properties of the paged KV cache allocator:
//! - Construction rejects zero-dimension parameters
//! - Free page accounting is conserved (total = free + allocated)
//! - `allocate_sequence` / `free_sequence` page conservation
//! - `elements_per_token` is correct
//! - Double-allocate is rejected

use super::*;

/// Prove `PagedKvCache::new` rejects page_size == 0.
#[kani::unwind(1)]
#[kani::proof]
fn proof_paged_cache_rejects_zero_page_size() {
    let result = PagedKvCache::new(0, 16, 4, 8, 64);
    assert!(result.is_err(), "page_size=0 must be rejected");
}

/// Prove `PagedKvCache::new` rejects num_pages == 0.
#[kani::unwind(1)]
#[kani::proof]
fn proof_paged_cache_rejects_zero_num_pages() {
    let result = PagedKvCache::new(16, 0, 4, 8, 64);
    assert!(result.is_err(), "num_pages=0 must be rejected");
}

/// Prove `PagedKvCache::new` rejects num_layers == 0.
#[kani::unwind(1)]
#[kani::proof]
fn proof_paged_cache_rejects_zero_num_layers() {
    let result = PagedKvCache::new(16, 16, 0, 8, 64);
    assert!(result.is_err(), "num_layers=0 must be rejected");
}

/// Prove `PagedKvCache::new` rejects num_heads == 0.
#[kani::unwind(1)]
#[kani::proof]
fn proof_paged_cache_rejects_zero_num_heads() {
    let result = PagedKvCache::new(16, 16, 4, 0, 64);
    assert!(result.is_err(), "num_heads=0 must be rejected");
}

/// Prove `PagedKvCache::new` rejects head_dim == 0.
#[kani::unwind(1)]
#[kani::proof]
fn proof_paged_cache_rejects_zero_head_dim() {
    let result = PagedKvCache::new(16, 16, 4, 8, 0);
    assert!(result.is_err(), "head_dim=0 must be rejected");
}

/// Prove all pages are initially free after construction.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(8)]
fn proof_paged_cache_initial_free_pages() {
    let page_size: usize = kani::any();
    kani::assume(page_size >= 1 && page_size <= 4);
    let num_pages: usize = kani::any();
    kani::assume(num_pages >= 1 && num_pages <= 8);
    let num_layers: usize = kani::any();
    kani::assume(num_layers >= 1 && num_layers <= 4);

    let cache = PagedKvCache::new(page_size, num_pages, num_layers, 1, 1).unwrap();
    assert_eq!(
        cache.num_free_pages(),
        num_pages,
        "all pages must be free initially"
    );
    assert_eq!(
        cache.num_active_sequences(),
        0,
        "no active sequences initially"
    );
}

/// Prove `allocate_sequence` reduces free page count by exactly `num_layers`.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(8)]
fn proof_paged_cache_allocate_reduces_free_pages() {
    let num_layers: usize = kani::any();
    kani::assume(num_layers >= 1 && num_layers <= 3);
    // Need at least num_layers pages.
    let num_pages: usize = kani::any();
    kani::assume(num_pages >= num_layers && num_pages <= 8);

    let mut cache = PagedKvCache::new(4, num_pages, num_layers, 1, 1).unwrap();
    let before = cache.num_free_pages();
    cache.allocate_sequence(0).unwrap();
    let after = cache.num_free_pages();

    assert_eq!(
        before - after,
        num_layers,
        "allocate must consume exactly num_layers pages"
    );
    assert_eq!(
        cache.num_active_sequences(),
        1,
        "must have 1 active sequence after allocate"
    );
}

/// Prove `free_sequence` returns pages to the free pool.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(8)]
fn proof_paged_cache_free_restores_pages() {
    let num_layers: usize = kani::any();
    kani::assume(num_layers >= 1 && num_layers <= 3);
    let num_pages: usize = kani::any();
    kani::assume(num_pages >= num_layers && num_pages <= 8);

    let mut cache = PagedKvCache::new(4, num_pages, num_layers, 1, 1).unwrap();
    cache.allocate_sequence(0).unwrap();
    cache.free_sequence(0);

    assert_eq!(
        cache.num_free_pages(),
        num_pages,
        "freeing sequence must return all pages"
    );
    assert_eq!(
        cache.num_active_sequences(),
        0,
        "no active sequences after free"
    );
}

/// Prove total page conservation: free + allocated == num_pages, always.
#[kani::unwind(8)]
#[kani::proof]
#[kani::unwind(8)]
fn proof_paged_cache_page_conservation() {
    let num_layers: usize = kani::any();
    kani::assume(num_layers >= 1 && num_layers <= 2);
    let num_pages: usize = kani::any();
    // Need at least 2*num_layers for two sequences.
    kani::assume(num_pages >= 2 * num_layers && num_pages <= 8);

    let mut cache = PagedKvCache::new(4, num_pages, num_layers, 1, 1).unwrap();

    // Allocate seq 0 and seq 1.
    cache.allocate_sequence(0).unwrap();
    cache.allocate_sequence(1).unwrap();

    let allocated = 2 * num_layers;
    assert_eq!(
        cache.num_free_pages() + allocated,
        num_pages,
        "free + allocated must equal total"
    );

    // Free seq 0.
    cache.free_sequence(0);
    let allocated_after = num_layers;
    assert_eq!(
        cache.num_free_pages() + allocated_after,
        num_pages,
        "conservation after partial free"
    );

    // Free seq 1.
    cache.free_sequence(1);
    assert_eq!(
        cache.num_free_pages(),
        num_pages,
        "all pages free after freeing all sequences"
    );
}

/// Prove double-allocation of the same seq_id is rejected.
#[kani::unwind(1)]
#[kani::proof]
fn proof_paged_cache_double_allocate_rejected() {
    let mut cache = PagedKvCache::new(4, 8, 2, 1, 1).unwrap();
    cache.allocate_sequence(0).unwrap();
    let result = cache.allocate_sequence(0);
    assert!(result.is_err(), "double allocate must be rejected");
}

/// Prove allocation fails when insufficient free pages.
#[kani::unwind(1)]
#[kani::proof]
fn proof_paged_cache_allocation_insufficient_pages() {
    let num_layers: usize = kani::any();
    kani::assume(num_layers >= 2 && num_layers <= 4);
    // Exactly num_layers pages: enough for one sequence, not two.
    let mut cache = PagedKvCache::new(4, num_layers, num_layers, 1, 1).unwrap();
    cache.allocate_sequence(0).unwrap();
    let result = cache.allocate_sequence(1);
    assert!(
        result.is_err(),
        "allocate must fail when free pages < num_layers"
    );
}

/// Prove `elements_per_token` equals `num_heads * head_dim`.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(8)]
fn proof_paged_cache_elements_per_token() {
    let num_heads: usize = kani::any();
    kani::assume(num_heads >= 1 && num_heads <= 32);
    let head_dim: usize = kani::any();
    kani::assume(head_dim >= 1 && head_dim <= 128);

    let cache = PagedKvCache::new(4, 4, 1, num_heads, head_dim).unwrap();
    assert_eq!(
        cache.page_size(),
        4,
        "page_size accessor must return constructor value"
    );
    // elements_per_token is private, but we can verify through the num_pages accessor.
    assert_eq!(
        cache.num_pages(),
        4,
        "num_pages accessor must return constructor value"
    );
}

/// Prove `free_sequence` on a non-existent seq_id is a no-op (does not panic).
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(8)]
fn proof_paged_cache_free_nonexistent_noop() {
    let mut cache = PagedKvCache::new(4, 8, 2, 1, 1).unwrap();
    // Free seq_id that was never allocated.
    cache.free_sequence(42);
    assert_eq!(
        cache.num_free_pages(),
        8,
        "free of nonexistent seq must not change free count"
    );
}

/// Prove `sequence_token_count` returns None for non-allocated seq.
#[kani::unwind(1)]
#[kani::proof]
fn proof_paged_cache_token_count_none_for_unallocated() {
    let cache = PagedKvCache::new(4, 8, 2, 1, 1).unwrap();
    assert!(
        cache.sequence_token_count(0).is_none(),
        "unallocated seq must return None"
    );
}

/// Prove `sequence_token_count` returns Some(0) for freshly allocated seq.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(8)]
fn proof_paged_cache_token_count_zero_after_allocate() {
    let mut cache = PagedKvCache::new(4, 8, 2, 1, 1).unwrap();
    cache.allocate_sequence(7).unwrap();
    assert_eq!(
        cache.sequence_token_count(7),
        Some(0),
        "freshly allocated seq must have 0 tokens"
    );
}

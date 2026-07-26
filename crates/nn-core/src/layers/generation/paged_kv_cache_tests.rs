// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Tests for [`PagedKvCache`].

use super::PagedKvCache;

/// Helper: create K data for a token. Fills with `token_id as f32` for easy verification.
fn make_kv_data(token_id: usize, num_heads: usize, head_dim: usize) -> Vec<f32> {
    vec![token_id as f32; num_heads * head_dim]
}

#[test]
fn test_paged_kv_cache_new_validates_zero_params() {
    assert!(PagedKvCache::new(0, 100, 4, 8, 64).is_err());
    assert!(PagedKvCache::new(16, 0, 4, 8, 64).is_err());
    assert!(PagedKvCache::new(16, 100, 0, 8, 64).is_err());
    assert!(PagedKvCache::new(16, 100, 4, 0, 64).is_err());
    assert!(PagedKvCache::new(16, 100, 4, 8, 0).is_err());
}

#[test]
fn test_paged_kv_cache_new_valid() {
    let cache = PagedKvCache::new(16, 100, 4, 8, 64).expect("valid params");
    assert_eq!(cache.num_free_pages(), 100);
    assert_eq!(cache.num_pages(), 100);
    assert_eq!(cache.page_size(), 16);
    assert_eq!(cache.num_active_sequences(), 0);
}

#[test]
fn test_paged_kv_cache_allocate_and_free_single_sequence() {
    let num_layers = 2;
    let mut cache = PagedKvCache::new(4, 20, num_layers, 2, 4).expect("valid");

    // Allocate uses num_layers pages.
    cache.allocate_sequence(0).expect("alloc seq 0");
    assert_eq!(cache.num_free_pages(), 20 - num_layers);
    assert_eq!(cache.num_active_sequences(), 1);
    assert_eq!(cache.sequence_token_count(0), Some(0));

    // Free returns all pages.
    cache.free_sequence(0);
    assert_eq!(cache.num_free_pages(), 20);
    assert_eq!(cache.num_active_sequences(), 0);
    assert_eq!(cache.sequence_token_count(0), None);
}

#[test]
fn test_paged_kv_cache_duplicate_allocate_errors() {
    let mut cache = PagedKvCache::new(4, 20, 2, 2, 4).expect("valid");
    cache.allocate_sequence(0).expect("first alloc");
    assert!(cache.allocate_sequence(0).is_err());
}

#[test]
fn test_paged_kv_cache_three_sequences_different_lengths() {
    let page_size = 4;
    let num_layers = 2;
    let num_heads = 2;
    let head_dim = 3;
    // Need enough pages: 3 seqs × 2 layers × (up to 3 pages) = 18 max, use 30.
    let mut cache =
        PagedKvCache::new(page_size, 30, num_layers, num_heads, head_dim).expect("valid");

    let initial_free = cache.num_free_pages();

    // Allocate 3 sequences.
    cache.allocate_sequence(10).expect("alloc seq 10");
    cache.allocate_sequence(20).expect("alloc seq 20");
    cache.allocate_sequence(30).expect("alloc seq 30");
    assert_eq!(cache.num_active_sequences(), 3);
    // Each allocation takes num_layers pages.
    assert_eq!(cache.num_free_pages(), initial_free - 3 * num_layers);

    // Seq 10: append 2 tokens (fits in 1 page per layer).
    for token_id in 0..2 {
        let k = make_kv_data(token_id + 100, num_heads, head_dim);
        let v = make_kv_data(token_id + 200, num_heads, head_dim);
        for layer in 0..num_layers {
            cache.append_kv(10, layer, &k, &v).expect("append seq 10");
        }
    }
    assert_eq!(cache.sequence_token_count(10), Some(2));

    // Seq 20: append 6 tokens (needs 2 pages per layer: 4 + 2).
    for token_id in 0..6 {
        let k = make_kv_data(token_id + 300, num_heads, head_dim);
        let v = make_kv_data(token_id + 400, num_heads, head_dim);
        for layer in 0..num_layers {
            cache.append_kv(20, layer, &k, &v).expect("append seq 20");
        }
    }
    assert_eq!(cache.sequence_token_count(20), Some(6));

    // Seq 30: append 10 tokens (needs 3 pages per layer: 4 + 4 + 2).
    for token_id in 0..10 {
        let k = make_kv_data(token_id + 500, num_heads, head_dim);
        let v = make_kv_data(token_id + 600, num_heads, head_dim);
        for layer in 0..num_layers {
            cache.append_kv(30, layer, &k, &v).expect("append seq 30");
        }
    }
    assert_eq!(cache.sequence_token_count(30), Some(10));

    // Verify data integrity for seq 10 (2 tokens).
    for layer in 0..num_layers {
        let (k, v) = cache.get_kv(10, layer).expect("get seq 10");
        let elems = num_heads * head_dim;
        assert_eq!(k.len(), 2 * elems);
        assert_eq!(v.len(), 2 * elems);
        // Token 0 K values should all be (0 + 100) = 100.0
        assert!(k[..elems].iter().all(|&x| (x - 100.0).abs() < 1e-6));
        // Token 1 K values should all be (1 + 100) = 101.0
        assert!(k[elems..2 * elems]
            .iter()
            .all(|&x| (x - 101.0).abs() < 1e-6));
        // Token 0 V values should all be (0 + 200) = 200.0
        assert!(v[..elems].iter().all(|&x| (x - 200.0).abs() < 1e-6));
    }

    // Verify data integrity for seq 20 (6 tokens, crosses page boundary at token 4).
    for layer in 0..num_layers {
        let (k, v) = cache.get_kv(20, layer).expect("get seq 20");
        let elems = num_heads * head_dim;
        assert_eq!(k.len(), 6 * elems);
        for token_id in 0..6 {
            let start = token_id * elems;
            let expected_k = (token_id + 300) as f32;
            let expected_v = (token_id + 400) as f32;
            assert!(
                k[start..start + elems]
                    .iter()
                    .all(|&x| (x - expected_k).abs() < 1e-6),
                "seq 20 K mismatch at token {token_id}"
            );
            assert!(
                v[start..start + elems]
                    .iter()
                    .all(|&x| (x - expected_v).abs() < 1e-6),
                "seq 20 V mismatch at token {token_id}"
            );
        }
    }

    // Verify data integrity for seq 30 (10 tokens, 3 pages per layer).
    for layer in 0..num_layers {
        let (k, v) = cache.get_kv(30, layer).expect("get seq 30");
        let elems = num_heads * head_dim;
        assert_eq!(k.len(), 10 * elems);
        for token_id in 0..10 {
            let start = token_id * elems;
            let expected_k = (token_id + 500) as f32;
            let expected_v = (token_id + 600) as f32;
            assert!(
                k[start..start + elems]
                    .iter()
                    .all(|&x| (x - expected_k).abs() < 1e-6),
                "seq 30 K mismatch at token {token_id}"
            );
            assert!(
                v[start..start + elems]
                    .iter()
                    .all(|&x| (x - expected_v).abs() < 1e-6),
                "seq 30 V mismatch at token {token_id}"
            );
        }
    }

    // Free seq 20, verify page reclamation.
    let free_before = cache.num_free_pages();
    cache.free_sequence(20);
    let free_after = cache.num_free_pages();
    assert_eq!(cache.num_active_sequences(), 2);
    // Seq 20 had 2 pages per layer (initial + 1 expansion) × 2 layers = 4 pages returned.
    // Initial allocation gave 1 page/layer, then 1 expansion page per layer = 2×2 = 4.
    assert!(
        free_after > free_before,
        "freeing seq 20 should return pages"
    );

    // Remaining sequences still work.
    let (k, _) = cache.get_kv(10, 0).expect("seq 10 still valid");
    assert_eq!(k.len(), 2 * num_heads * head_dim);
    let (k, _) = cache.get_kv(30, 0).expect("seq 30 still valid");
    assert_eq!(k.len(), 10 * num_heads * head_dim);

    // Free remaining sequences.
    cache.free_sequence(10);
    cache.free_sequence(30);
    assert_eq!(cache.num_active_sequences(), 0);
    assert_eq!(cache.num_free_pages(), 30);
}

#[test]
fn test_paged_kv_cache_append_wrong_data_length() {
    let mut cache = PagedKvCache::new(4, 10, 1, 2, 3).expect("valid");
    cache.allocate_sequence(0).expect("alloc");
    // Expected length is 2 * 3 = 6, pass 5.
    let short = vec![1.0; 5];
    let correct = vec![1.0; 6];
    assert!(cache.append_kv(0, 0, &short, &correct).is_err());
    assert!(cache.append_kv(0, 0, &correct, &short).is_err());
}

#[test]
fn test_paged_kv_cache_layer_out_of_range() {
    let mut cache = PagedKvCache::new(4, 10, 2, 2, 3).expect("valid");
    cache.allocate_sequence(0).expect("alloc");
    let data = vec![1.0; 6];
    assert!(cache.append_kv(0, 2, &data, &data).is_err());
    assert!(cache.get_kv(0, 2).is_err());
}

#[test]
fn test_paged_kv_cache_unallocated_sequence_errors() {
    let cache = PagedKvCache::new(4, 10, 2, 2, 3).expect("valid");
    assert!(cache.get_kv(99, 0).is_err());
}

#[test]
fn test_paged_kv_cache_insufficient_pages_for_allocation() {
    // Only 1 page, but 2 layers needed.
    let mut cache = PagedKvCache::new(4, 1, 2, 2, 3).expect("valid");
    assert!(cache.allocate_sequence(0).is_err());
}

#[test]
fn test_paged_kv_cache_page_expansion_exhaustion() {
    // 2 pages total, 1 layer → 1 page for allocation, 1 spare.
    // page_size=2 → after 2 tokens, expansion needs a new page (uses the spare).
    // After 4 tokens, expansion needs another → should fail.
    let num_heads = 1;
    let head_dim = 2;
    let mut cache = PagedKvCache::new(2, 2, 1, num_heads, head_dim).expect("valid");
    cache.allocate_sequence(0).expect("alloc");
    assert_eq!(cache.num_free_pages(), 1);

    let data = vec![1.0; num_heads * head_dim];
    // Token 0, 1 fit in first page.
    cache.append_kv(0, 0, &data, &data).expect("token 0");
    cache.append_kv(0, 0, &data, &data).expect("token 1");
    // Token 2 needs expansion → uses last free page.
    cache.append_kv(0, 0, &data, &data).expect("token 2");
    assert_eq!(cache.num_free_pages(), 0);
    // Token 3 fits in second page.
    cache.append_kv(0, 0, &data, &data).expect("token 3");
    // Token 4 needs a third page → should fail.
    assert!(cache.append_kv(0, 0, &data, &data).is_err());
}

#[test]
fn test_paged_kv_cache_free_and_reuse_pages() {
    let num_layers = 1;
    let num_heads = 1;
    let head_dim = 2;
    let mut cache = PagedKvCache::new(4, 4, num_layers, num_heads, head_dim).expect("valid");

    // Allocate seq 0, fill with data.
    cache.allocate_sequence(0).expect("alloc 0");
    let data_a = vec![42.0; num_heads * head_dim];
    cache.append_kv(0, 0, &data_a, &data_a).expect("append");
    assert_eq!(cache.num_free_pages(), 3);

    // Free seq 0.
    cache.free_sequence(0);
    assert_eq!(cache.num_free_pages(), 4);

    // Allocate seq 1 — reuses freed page.
    cache.allocate_sequence(1).expect("alloc 1");
    let data_b = vec![99.0; num_heads * head_dim];
    cache.append_kv(1, 0, &data_b, &data_b).expect("append 1");

    // Verify seq 1 has its own data (not seq 0's stale data).
    let (k, v) = cache.get_kv(1, 0).expect("get seq 1");
    assert!(k.iter().all(|&x| (x - 99.0).abs() < 1e-6));
    assert!(v.iter().all(|&x| (x - 99.0).abs() < 1e-6));
}

#[test]
fn test_paged_kv_cache_empty_sequence_get_kv() {
    let mut cache = PagedKvCache::new(4, 10, 2, 2, 3).expect("valid");
    cache.allocate_sequence(0).expect("alloc");
    let (k, v) = cache.get_kv(0, 0).expect("get empty");
    assert!(k.is_empty());
    assert!(v.is_empty());
}

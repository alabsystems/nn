// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! KV cache initialization, append, retrieval, sequence length tracking,
//! and memory layout tests for Qwen3 (#4186).

use crate::test_utils::tiny_config;
use crate::Qwen3Model;
use nn_core::layers::kv_cache::KvCache;
use nn_core::var_builder::VarBuilder;
use nn_core::{DType, Device};

// ---------------------------------------------------------------------------
// KV cache initialization
// ---------------------------------------------------------------------------

#[test]
fn test_kv_cache_new_correct_layer_count() {
    let cache = KvCache::new(12);
    assert_eq!(cache.num_layers(), 12);
}

#[test]
fn test_kv_cache_new_is_empty() {
    let cache = KvCache::new(4);
    assert!(cache.is_empty());
    assert_eq!(cache.seq_len(), 0);
}

#[test]
fn test_kv_cache_new_from_model_matches_config() {
    let cfg = tiny_config(); // 2 layers
    let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
    let model = Qwen3Model::load(&vb, cfg.clone()).unwrap();
    let cache = model.new_cache();
    assert_eq!(cache.num_layers(), cfg.num_hidden_layers);
    assert!(cache.is_empty());
}

#[test]
fn test_kv_cache_new_zero_layers() {
    let cache = KvCache::new(0);
    assert_eq!(cache.num_layers(), 0);
    assert!(cache.is_empty());
    assert_eq!(cache.seq_len(), 0);
}

// ---------------------------------------------------------------------------
// Cache append and retrieval
// ---------------------------------------------------------------------------

#[test]
fn test_kv_cache_single_token_append_via_forward() {
    let cfg = tiny_config();
    let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
    let model = Qwen3Model::load(&vb, cfg).unwrap();
    let mut cache = model.new_cache();

    model.forward_cached(&[0], &[0], Some(&mut cache)).unwrap();
    assert_eq!(cache.seq_len(), 1);
    assert!(!cache.is_empty());
}

#[test]
fn test_kv_cache_multi_token_append_via_forward() {
    let cfg = tiny_config();
    let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
    let model = Qwen3Model::load(&vb, cfg).unwrap();
    let mut cache = model.new_cache();

    // Prefill with 4 tokens
    model
        .forward_cached(&[0, 1, 2, 3], &[0, 1, 2, 3], Some(&mut cache))
        .unwrap();
    assert_eq!(cache.seq_len(), 4);
}

#[test]
fn test_kv_cache_incremental_append_accumulates() {
    let cfg = tiny_config();
    let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
    let model = Qwen3Model::load(&vb, cfg).unwrap();
    let mut cache = model.new_cache();

    for i in 0..5 {
        model.forward_cached(&[i], &[i], Some(&mut cache)).unwrap();
        assert_eq!(
            cache.seq_len(),
            i + 1,
            "after {n} appends, seq_len should be {n}",
            n = i + 1
        );
    }
}

#[test]
fn test_kv_cache_prefill_then_decode_sequence() {
    let cfg = tiny_config();
    let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
    let model = Qwen3Model::load(&vb, cfg.clone()).unwrap();
    let mut cache = model.new_cache();

    // Prefill
    let prefill_logits = model
        .forward_cached(&[0, 1, 2], &[0, 1, 2], Some(&mut cache))
        .unwrap();
    assert_eq!(prefill_logits.dims(), &[1, 3, cfg.vocab_size]);
    assert_eq!(cache.seq_len(), 3);

    // Decode single tokens
    for step in 3..7 {
        let decode_logits = model
            .forward_cached(&[step % cfg.vocab_size], &[step], Some(&mut cache))
            .unwrap();
        assert_eq!(decode_logits.dims(), &[1, 1, cfg.vocab_size]);
        assert_eq!(cache.seq_len(), step + 1);
    }
}

// ---------------------------------------------------------------------------
// Cache with different sequence lengths
// ---------------------------------------------------------------------------

#[test]
fn test_kv_cache_long_sequence_stability() {
    let cfg = tiny_config(); // max_position_embeddings = 64
    let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
    let model = Qwen3Model::load(&vb, cfg.clone()).unwrap();
    let mut cache = model.new_cache();

    // Process tokens up to near max position
    for i in 0..32 {
        let logits = model
            .forward_cached(&[i % cfg.vocab_size], &[i], Some(&mut cache))
            .unwrap();
        let vals = logits.to_flat_vec::<f32>().unwrap();
        assert!(
            vals.iter().all(|v| v.is_finite()),
            "all logits should be finite at step {i}"
        );
    }
    assert_eq!(cache.seq_len(), 32);
}

#[test]
fn test_kv_cache_different_prefill_sizes() {
    let cfg = tiny_config();
    let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
    let model = Qwen3Model::load(&vb, cfg.clone()).unwrap();

    // Test prefill with various sizes
    for prefill_len in [1, 2, 4, 8, 16] {
        let mut cache = model.new_cache();
        let ids: Vec<usize> = (0..prefill_len).collect();
        let positions: Vec<usize> = (0..prefill_len).collect();

        let logits = model
            .forward_cached(&ids, &positions, Some(&mut cache))
            .unwrap();
        assert_eq!(logits.dims(), &[1, prefill_len, cfg.vocab_size]);
        assert_eq!(cache.seq_len(), prefill_len);
    }
}

// ---------------------------------------------------------------------------
// Cache memory layout and layer access
// ---------------------------------------------------------------------------

#[test]
fn test_kv_cache_layer_access_valid() {
    let cache = KvCache::new(4);
    // All layers should be accessible
    for i in 0..4 {
        assert!(cache.layer(i).is_ok(), "layer {i} should be accessible");
    }
}

#[test]
fn test_kv_cache_layer_access_out_of_bounds() {
    let cache = KvCache::new(4);
    assert!(cache.layer(4).is_err(), "layer 4 should be out of bounds");
    assert!(
        cache.layer(100).is_err(),
        "layer 100 should be out of bounds"
    );
}

#[test]
fn test_kv_cache_all_layers_populated_after_forward() {
    let cfg = tiny_config(); // 2 layers
    let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
    let model = Qwen3Model::load(&vb, cfg).unwrap();
    let mut cache = model.new_cache();

    model
        .forward_cached(&[0, 1], &[0, 1], Some(&mut cache))
        .unwrap();

    // Both layers should be non-empty after forward
    for i in 0..2 {
        let layer = cache.layer(i).unwrap();
        assert!(
            !layer.is_empty(),
            "layer {i} should be populated after forward"
        );
        assert_eq!(layer.seq_len(), 2, "layer {i} should have seq_len=2");
    }
}

#[test]
fn test_kv_cache_reset_clears_all_layers() {
    let cfg = tiny_config();
    let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
    let model = Qwen3Model::load(&vb, cfg).unwrap();
    let mut cache = model.new_cache();

    // Populate cache
    model
        .forward_cached(&[0, 1, 2], &[0, 1, 2], Some(&mut cache))
        .unwrap();
    assert_eq!(cache.seq_len(), 3);

    // Reset
    cache.reset();
    assert!(cache.is_empty());
    assert_eq!(cache.seq_len(), 0);
}

#[test]
fn test_kv_cache_clear_preserves_structure() {
    let cfg = tiny_config();
    let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
    let model = Qwen3Model::load(&vb, cfg).unwrap();
    let mut cache = model.new_cache();

    // Populate cache
    model
        .forward_cached(&[0, 1, 2], &[0, 1, 2], Some(&mut cache))
        .unwrap();
    assert_eq!(cache.seq_len(), 3);

    // Clear should empty all layers but preserve the layer count
    cache.clear();
    assert!(cache.is_empty());
    assert_eq!(cache.seq_len(), 0);
    assert_eq!(cache.num_layers(), 2);
}

#[test]
fn test_kv_cache_reuse_after_clear() {
    let cfg = tiny_config();
    let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
    let model = Qwen3Model::load(&vb, cfg.clone()).unwrap();
    let mut cache = model.new_cache();

    // First use
    model
        .forward_cached(&[0, 1], &[0, 1], Some(&mut cache))
        .unwrap();
    assert_eq!(cache.seq_len(), 2);

    // Clear and reuse
    cache.clear();
    assert_eq!(cache.seq_len(), 0);

    // Reuse with different input
    let logits = model
        .forward_cached(&[5, 6, 7], &[0, 1, 2], Some(&mut cache))
        .unwrap();
    assert_eq!(logits.dims(), &[1, 3, cfg.vocab_size]);
    assert_eq!(cache.seq_len(), 3);
}

// ---------------------------------------------------------------------------
// Cache mismatch detection
// ---------------------------------------------------------------------------

#[test]
fn test_kv_cache_layer_mismatch_detected() {
    let cfg = tiny_config(); // 2 layers
    let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
    let model = Qwen3Model::load(&vb, cfg).unwrap();

    // Create cache with wrong number of layers
    let mut wrong_cache = KvCache::new(5);
    let err = model.forward_cached(&[0], &[0], Some(&mut wrong_cache));
    assert!(err.is_err(), "mismatched cache layer count should error");
}

#[test]
fn test_kv_cache_layer_mismatch_error_includes_counts() {
    let cfg = tiny_config(); // 2 layers
    let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
    let model = Qwen3Model::load(&vb, cfg).unwrap();

    let mut wrong_cache = KvCache::new(10);
    let err = model
        .forward_cached(&[0], &[0], Some(&mut wrong_cache))
        .unwrap_err();
    let msg = format!("{err:?}");
    assert!(
        msg.contains("10") && msg.contains("2"),
        "error should mention both cache (10) and model (2) layer counts: {msg}"
    );
}

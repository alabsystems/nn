#![allow(deprecated)]
// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! KV cache, causal mask, edge case, and YaRN scaling tests.
//! forward_from_embeddings tests extracted to `qwen3_tests_cache_embeddings.rs`.

use super::*;
use crate::test_utils::tiny_config;
use nn_core::var_builder::VarBuilder;
use nn_core::{DType, Device};

// -- KV cache tests -----------------------------------------------------------

#[test]
fn test_forward_cached_none_matches_forward() {
    // forward_cached(ids, pos, None) should produce the same result as forward(ids, pos)
    let cfg = tiny_config();
    let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
    let model = Qwen3Model::load(&vb, cfg).unwrap();

    // Single token (zero weights produce valid output for seq_len=1)
    let logits_plain = model.forward(&[42], &[0]).unwrap();
    let logits_cached = model.forward_cached(&[42], &[0], None).unwrap();
    assert_eq!(logits_plain.dims(), logits_cached.dims());
    assert_eq!(
        logits_plain.to_flat_vec::<f32>().unwrap(),
        logits_cached.to_flat_vec::<f32>().unwrap()
    );
}

#[test]
fn test_forward_cached_wrong_layer_count() {
    let cfg = tiny_config(); // 2 layers
    let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
    let model = Qwen3Model::load(&vb, cfg).unwrap();

    let mut cache = KvCache::new(5); // wrong: 5 != 2
    let result = model.forward_cached(&[0], &[0], Some(&mut cache));
    assert!(result.is_err());
    let msg = format!("{:?}", result.unwrap_err());
    assert!(
        msg.contains("5") && msg.contains("2"),
        "error should mention layer count mismatch: {msg}"
    );
}

#[test]
fn test_new_cache_layer_count() {
    let cfg = tiny_config();
    let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
    let model = Qwen3Model::load(&vb, cfg).unwrap();
    let cache = model.new_cache();
    assert_eq!(cache.num_layers(), 2);
    assert!(cache.is_empty());
}

#[test]
fn test_forward_cached_single_token_populates_cache() {
    let cfg = tiny_config();
    let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
    let model = Qwen3Model::load(&vb, cfg).unwrap();
    let mut cache = model.new_cache();

    // First token: cache should be populated
    let logits = model.forward_cached(&[42], &[0], Some(&mut cache)).unwrap();
    assert_eq!(logits.dims(), &[1, 1, 100]);
    assert_eq!(cache.seq_len(), 1);
    assert!(!cache.is_empty());
}

#[test]
fn test_forward_cached_incremental_grows_cache() {
    let cfg = tiny_config();
    let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
    let model = Qwen3Model::load(&vb, cfg).unwrap();
    let mut cache = model.new_cache();

    // Step 1: first token
    model.forward_cached(&[10], &[0], Some(&mut cache)).unwrap();
    assert_eq!(cache.seq_len(), 1);

    // Step 2: second token
    model.forward_cached(&[20], &[1], Some(&mut cache)).unwrap();
    assert_eq!(cache.seq_len(), 2);

    // Step 3: third token
    let logits = model.forward_cached(&[30], &[2], Some(&mut cache)).unwrap();
    assert_eq!(cache.seq_len(), 3);
    assert_eq!(logits.dims(), &[1, 1, 100]);
}

#[test]
fn test_causal_mask_with_offset_shape() {
    // 1 new token, 5 total (4 cached + 1 new)
    let mask = causal_mask_with_offset(1, 5, DType::F32, &Device::Cpu).unwrap();
    assert_eq!(mask.dims(), &[1, 1, 1, 5]);

    // The single new token can attend to all 5 positions (it's the last one)
    let data = mask.to_flat_vec::<f32>().unwrap();
    assert_eq!(data, &[0.0, 0.0, 0.0, 0.0, 0.0]);
}

#[test]
fn test_causal_mask_with_offset_two_new_tokens() {
    // 2 new tokens, 4 total (2 cached + 2 new)
    let mask = causal_mask_with_offset(2, 4, DType::F32, &Device::Cpu).unwrap();
    assert_eq!(mask.dims(), &[1, 1, 2, 4]);
    let data = mask.to_flat_vec::<f32>().unwrap();
    // Row 0 (abs pos 2): can attend to positions 0,1,2 but not 3
    assert_eq!(data[0], 0.0); // pos 0
    assert_eq!(data[1], 0.0); // pos 1
    assert_eq!(data[2], 0.0); // pos 2
    assert!(data[3].is_infinite() && data[3] < 0.0); // pos 3
                                                     // Row 1 (abs pos 3): can attend to all 4 positions
    assert_eq!(data[4], 0.0);
    assert_eq!(data[5], 0.0);
    assert_eq!(data[6], 0.0);
    assert_eq!(data[7], 0.0);
}

#[test]
fn test_causal_mask_with_offset_no_offset_matches_original() {
    // When new_tokens == total_tokens (no cache), should match causal_mask
    let mask_orig = causal_mask(3, DType::F32, &Device::Cpu).unwrap();
    let mask_offset = causal_mask_with_offset(3, 3, DType::F32, &Device::Cpu).unwrap();
    assert_eq!(
        mask_orig.to_flat_vec::<f32>().unwrap(),
        mask_offset.to_flat_vec::<f32>().unwrap()
    );
}

// forward_from_embeddings and forward_from_embeddings_with_hidden tests
// extracted to qwen3_tests_cache_embeddings.rs
#[path = "qwen3_tests_cache_embeddings.rs"]
mod embeddings;

// -- Proof coverage: edge case tests ------------------------------------------

#[test]
fn test_causal_mask_zero_tokens() {
    // 0 tokens: nn-core rejects — generating a mask for 0 tokens is a programmer error
    let result = causal_mask(0, DType::F32, &Device::Cpu);
    assert!(result.is_err());
}

#[test]
fn test_causal_mask_with_offset_zero_new_tokens() {
    // 0 new tokens: nn-core rejects — no query tokens means no mask needed
    let result = causal_mask_with_offset(0, 5, DType::F32, &Device::Cpu);
    assert!(result.is_err());
}

#[test]
fn test_causal_mask_single_token_is_all_zeros() {
    // When new_tokens=1 (autoregressive decoding), the single query can attend
    // to all prior positions. The mask is all-zeros for any total_tokens value.
    // This justifies skipping mask allocation when seq_len == 1 (returns None).
    for total in [1, 5, 10, 100] {
        let mask = causal_mask_with_offset(1, total, DType::F32, &Device::Cpu).unwrap();
        let data = mask.to_flat_vec::<f32>().unwrap();
        assert!(
            data.iter().all(|&v| v == 0.0),
            "single-token mask should be all-zeros for total_tokens={total}"
        );
    }
}

#[test]
fn test_config_validation_zero_hidden_size() {
    let mut cfg = tiny_config();
    cfg.hidden_size = 0;
    assert!(cfg.validate().is_err());
}

#[test]
fn test_config_validation_zero_intermediate_size() {
    let mut cfg = tiny_config();
    cfg.intermediate_size = 0;
    assert!(cfg.validate().is_err());
}

#[test]
fn test_config_validation_zero_kv_heads() {
    let mut cfg = tiny_config();
    cfg.num_key_value_heads = 0;
    assert!(cfg.validate().is_err());
}

#[test]
fn test_repeat_kv_wrong_rank_error() {
    // repeat_kv expects 4D input — 3D should fail
    let x = DynTensor::ones(&[1, 2, 128], DType::F32, &Device::Cpu).unwrap();
    let result = repeat_kv(&x, 2);
    assert!(result.is_err(), "3D input to repeat_kv should fail");
}

#[test]
fn test_model_forward_empty_input() {
    // Empty input (no tokens) should produce empty output
    let cfg = tiny_config();
    let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
    let model = Qwen3Model::load(&vb, cfg).unwrap();
    let result = model.forward(&[], &[]);
    // Empty input may succeed or fail — just verify no panic
    if let Ok(logits) = result {
        assert_eq!(logits.dims()[1], 0);
    }
}

// -- YaRN scaling integration (#1230) -----------------------------------------

#[test]
fn test_model_load_with_yarn_scaling() {
    use nn_core::layers::YarnScaling;
    let mut cfg = tiny_config();
    cfg.max_position_embeddings = 256;
    cfg.rope_scaling = Some(YarnScaling::new(4.0, 1.0, 32.0, 1.0, 64));
    let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
    let model = Qwen3Model::load(&vb, cfg).unwrap();
    let logits = model.forward(&[42], &[0]).unwrap();
    assert_eq!(logits.dims(), &[1, 1, 100]);
}

// -- AC6: Multi-token equivalence: one-shot vs incremental with KV cache ------
#[test]
fn test_kv_cache_multi_token_equivalence() {
    // Processing [A, B, C] in one shot should match [A]→[B]→[C] incrementally
    // for the final token's logits.
    let cfg = tiny_config();
    let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
    let model = Qwen3Model::load(&vb, cfg).unwrap();

    // One-shot: process all 3 tokens at once (no cache)
    let logits_oneshot = model.forward(&[10, 20, 30], &[0, 1, 2]).unwrap();
    // Extract last token's logits [1, 1, vocab]
    let last_oneshot = logits_oneshot.narrow(1, 2, 1).unwrap();
    let oneshot_vec = last_oneshot.to_flat_vec::<f32>().unwrap();

    // Incremental: process token by token with cache
    let mut cache = model.new_cache();
    model.forward_cached(&[10], &[0], Some(&mut cache)).unwrap();
    model.forward_cached(&[20], &[1], Some(&mut cache)).unwrap();
    let logits_incr = model.forward_cached(&[30], &[2], Some(&mut cache)).unwrap();
    let incr_vec = logits_incr.to_flat_vec::<f32>().unwrap();

    // Logits for token C should match between one-shot and incremental.
    assert_eq!(oneshot_vec.len(), incr_vec.len());
    for (i, (&a, &b)) in oneshot_vec.iter().zip(incr_vec.iter()).enumerate() {
        assert!(
            (a - b).abs() < 1e-5,
            "logit mismatch at index {i}: oneshot={a}, incremental={b}"
        );
    }
}

// -- AC7: Batch > 1 with cache (forward_from_embeddings accepts arbitrary batch) -

#[test]
fn test_kv_cache_batch_gt_1_via_embeddings() {
    let cfg = tiny_config();
    let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
    let model = Qwen3Model::load(&vb, cfg.clone()).unwrap();
    let mut cache = model.new_cache();

    // Batch=2, seq_len=1: two different sequences processed in parallel.
    let emb = DynTensor::ones(&[2, 1, cfg.hidden_size], DType::F32, &Device::Cpu).unwrap();
    let logits = model
        .forward_from_embeddings(&emb, &[0], Some(&mut cache))
        .unwrap();
    assert_eq!(logits.dims(), &[2, 1, cfg.vocab_size]);
    assert_eq!(cache.seq_len(), 1);

    // Step 2: append another token for both sequences.
    let emb2 = DynTensor::ones(&[2, 1, cfg.hidden_size], DType::F32, &Device::Cpu).unwrap();
    let logits2 = model
        .forward_from_embeddings(&emb2, &[1], Some(&mut cache))
        .unwrap();
    assert_eq!(logits2.dims(), &[2, 1, cfg.vocab_size]);
    assert_eq!(cache.seq_len(), 2);
    let arr = logits2.as_cpu_f32().unwrap();
    assert!(arr.iter().all(|v| v.is_finite()), "batch>1 cache NaN/Inf");
}

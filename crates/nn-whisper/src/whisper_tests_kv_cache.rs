#![allow(deprecated)]
// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! KV cache tests for Whisper self-attention and cross-attention.

use crate::test_utils::tiny_config;
use crate::WhisperModel;
use nn_core::dyn_tensor::DynTensor;
use nn_core::test_utils::cpu;
use nn_core::{DType, VarBuilder};

#[test]
fn test_self_attn_kv_cache_multi_step_decode() {
    // Multi-step autoregressive decode: each step decodes 1 token with advancing offset.
    let config = tiny_config();
    let vb = VarBuilder::zeros(DType::F32, &cpu());
    let mut model = WhisperModel::load(&vb, config.clone()).unwrap();

    let encoder_out = DynTensor::zeros(&[1, 8, config.d_model], DType::F32, &cpu()).unwrap();

    // Step 0: first token, flush cache.
    let t0 = DynTensor::new(&[0.0], &[1, 1], &cpu()).unwrap();
    let logits0 = model.decode(&t0, &encoder_out, true, 0).unwrap();
    assert_eq!(logits0.dims(), &[1, 1, config.vocab_size]);

    // Step 1: second token, offset=1.
    let t1 = DynTensor::new(&[1.0], &[1, 1], &cpu()).unwrap();
    let logits1 = model.decode(&t1, &encoder_out, false, 1).unwrap();
    assert_eq!(logits1.dims(), &[1, 1, config.vocab_size]);

    // Step 2: third token, offset=2.
    let t2 = DynTensor::new(&[2.0], &[1, 1], &cpu()).unwrap();
    let logits2 = model.decode(&t2, &encoder_out, false, 2).unwrap();
    assert_eq!(logits2.dims(), &[1, 1, config.vocab_size]);
}

#[test]
fn test_self_attn_kv_cache_flush_resets() {
    // Flushing the cache on a subsequent step should give the same result
    // as a fresh first step (since self-attention KV cache is cleared).
    let config = tiny_config();
    let vb = VarBuilder::zeros(DType::F32, &cpu());
    let mut model = WhisperModel::load(&vb, config.clone()).unwrap();

    let encoder_out = DynTensor::zeros(&[1, 8, config.d_model], DType::F32, &cpu()).unwrap();
    let t0 = DynTensor::new(&[5.0], &[1, 1], &cpu()).unwrap();

    // First decode with flush.
    let logits_first = model.decode(&t0, &encoder_out, true, 0).unwrap();

    // Decode a second token to populate cache.
    let t1 = DynTensor::new(&[6.0], &[1, 1], &cpu()).unwrap();
    let _logits1 = model.decode(&t1, &encoder_out, false, 1).unwrap();

    // Flush again with same token as first step — should match logits_first.
    let logits_reflush = model.decode(&t0, &encoder_out, true, 0).unwrap();

    assert_eq!(
        logits_first.to_flat_vec::<f32>().unwrap(),
        logits_reflush.to_flat_vec::<f32>().unwrap(),
        "flushing cache should produce identical output to first step"
    );
}

#[test]
fn test_self_attn_kv_cache_incremental_vs_batch() {
    // Compare incremental 1-token-at-a-time decode vs a single 3-token batch decode.
    // With zero weights, both should produce the same last-token logits.
    let config = tiny_config();
    let vb = VarBuilder::zeros(DType::F32, &cpu());

    let encoder_out = DynTensor::zeros(&[1, 8, config.d_model], DType::F32, &cpu()).unwrap();

    // Batch decode: all 3 tokens at once.
    let mut model_batch = WhisperModel::load(&vb, config.clone()).unwrap();
    let tokens_batch = DynTensor::new(&[0.0, 1.0, 2.0], &[1, 3], &cpu()).unwrap();
    let logits_batch = model_batch
        .decode(&tokens_batch, &encoder_out, true, 0)
        .unwrap();
    // Last token logits from batch: [1, 3, vocab] -> row 2.
    let batch_flat = logits_batch.to_flat_vec::<f32>().unwrap();
    let vocab = config.vocab_size;
    let last_token_batch = &batch_flat[2 * vocab..3 * vocab];

    // Incremental decode: 1 token at a time.
    let mut model_incr = WhisperModel::load(&vb, config).unwrap();
    let t0 = DynTensor::new(&[0.0], &[1, 1], &cpu()).unwrap();
    let _l0 = model_incr.decode(&t0, &encoder_out, true, 0).unwrap();
    let t1 = DynTensor::new(&[1.0], &[1, 1], &cpu()).unwrap();
    let _l1 = model_incr.decode(&t1, &encoder_out, false, 1).unwrap();
    let t2 = DynTensor::new(&[2.0], &[1, 1], &cpu()).unwrap();
    let logits_incr = model_incr.decode(&t2, &encoder_out, false, 2).unwrap();
    let incr_flat = logits_incr.to_flat_vec::<f32>().unwrap();

    assert_eq!(
        last_token_batch.len(),
        incr_flat.len(),
        "last-token logit count mismatch"
    );

    // With zero weights, outputs are degenerate but should match exactly.
    for (i, (&b, &inc)) in last_token_batch.iter().zip(incr_flat.iter()).enumerate() {
        assert!(
            (b - inc).abs() < 1e-5,
            "logit mismatch at index {i}: batch={b}, incremental={inc}"
        );
    }
}

#[test]
fn test_self_attn_kv_cache_reset() {
    // reset_kv_cache should allow a fresh decode sequence.
    let config = tiny_config();
    let vb = VarBuilder::zeros(DType::F32, &cpu());
    let mut model = WhisperModel::load(&vb, config.clone()).unwrap();

    let encoder_out = DynTensor::zeros(&[1, 8, config.d_model], DType::F32, &cpu()).unwrap();
    let t0 = DynTensor::new(&[0.0], &[1, 1], &cpu()).unwrap();

    // Decode a few tokens.
    model.decode(&t0, &encoder_out, true, 0).unwrap();
    let t1 = DynTensor::new(&[1.0], &[1, 1], &cpu()).unwrap();
    model.decode(&t1, &encoder_out, false, 1).unwrap();

    // Reset and start fresh.
    model.reset_kv_cache();

    // Should succeed without errors — fresh decode after reset.
    let logits = model.decode(&t0, &encoder_out, true, 0).unwrap();
    assert_eq!(logits.dims(), &[1, 1, config.vocab_size]);
}

#[test]
fn test_encoder_output_shape_stride2_downsample() {
    // Conv2 has stride=2, so output seq_len = ceil(input_len / 2).
    // For input len 16 with pad=1: (16 + 2*1 - 3)/2 + 1 = 8.
    let config = tiny_config();
    let vb = VarBuilder::zeros(DType::F32, &cpu());
    let mut model = WhisperModel::load(&vb, config.clone()).unwrap();

    let mel = DynTensor::zeros(&[1, config.num_mel_bins, 16], DType::F32, &cpu()).unwrap();

    let out = model.encode(&mel).unwrap();
    // After conv1 (stride=1, pad=1): L=16
    // After conv2 (stride=2, pad=1): L = (16 + 2 - 3)/2 + 1 = 8
    assert_eq!(out.dim(1).unwrap(), 8);
}

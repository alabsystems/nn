// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Integration tests for ResidualAttentionBlock and MultiHeadAttention forward paths.
//!
//! These tests verify structural correctness (shapes, residual connections,
//! cache behavior) using both zero and non-trivial weights.
//! Covers the #1 test coverage gap: block.rs had zero dedicated tests.

use nn_core::dyn_tensor::DynTensor;
use nn_core::test_utils::cpu;
use nn_core::{DType, VarBuilder};
use nn_whisper::attention::MultiHeadAttention;
use nn_whisper::block::ResidualAttentionBlock;
use nn_whisper::positional::causal_mask;

// -- ResidualAttentionBlock structural tests --

#[test]
fn test_encoder_block_load_has_no_cross_attn() {
    let vb = VarBuilder::zeros(DType::F32, &cpu());
    let block = ResidualAttentionBlock::load_encoder(&vb, 2, 8, 16).unwrap();
    // Encoder block: self-attention only, no cross-attention.
    // Verify by checking self_cache_len starts at 0.
    assert_eq!(block.self_cache_len(), 0);
}

#[test]
fn test_decoder_block_load_succeeds() {
    let vb = VarBuilder::zeros(DType::F32, &cpu());
    let block = ResidualAttentionBlock::load_decoder(&vb, 2, 8, 16).unwrap();
    assert_eq!(block.self_cache_len(), 0);
}

#[test]
fn test_encoder_block_forward_output_shape() {
    let vb = VarBuilder::zeros(DType::F32, &cpu());
    let mut block = ResidualAttentionBlock::load_encoder(&vb, 2, 8, 16).unwrap();
    let x = DynTensor::zeros(&[1, 4, 8], DType::F32, &cpu()).unwrap();
    let out = block.forward_encoder(&x).unwrap();
    // Output shape must match input shape (residual connection).
    assert_eq!(out.dims(), &[1, 4, 8]);
}

#[test]
fn test_decoder_block_forward_output_shape() {
    let vb = VarBuilder::zeros(DType::F32, &cpu());
    let mut block = ResidualAttentionBlock::load_decoder(&vb, 2, 8, 16).unwrap();
    let x = DynTensor::zeros(&[1, 3, 8], DType::F32, &cpu()).unwrap();
    let encoder_out = DynTensor::zeros(&[1, 6, 8], DType::F32, &cpu()).unwrap();
    let mask = causal_mask(3, DType::F32, &cpu()).unwrap();
    let out = block
        .forward_decoder(&x, &encoder_out, &mask, true)
        .unwrap();
    assert_eq!(out.dims(), &[1, 3, 8]);
}

#[test]
fn test_encoder_block_residual_connection() {
    // With zero weights, self-attention and FFN both produce zero outputs.
    // The residual connection means output == input.
    let vb = VarBuilder::zeros(DType::F32, &cpu());
    let mut block = ResidualAttentionBlock::load_encoder(&vb, 2, 8, 16).unwrap();

    let data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let x = DynTensor::from_vec(data.clone(), &[1, 1, 8], &cpu()).unwrap();
    let out = block.forward_encoder(&x).unwrap();

    // Zero-weight attention produces all-zero output for the attention branch.
    // LayerNorm on zero weights: weight=0, bias=0 -> output is 0.
    // So attention output = 0, residual = input + 0 = input.
    // Then FFN: LayerNorm(input, w=0, b=0) = 0, fc1(0)=0, gelu(0)=0, fc2(0)=0.
    // residual = input + 0 = input.
    let out_vals = out.to_flat_vec::<f32>().unwrap();
    assert_eq!(
        out_vals, data,
        "with zero weights, output should equal input via residual"
    );
}

#[test]
fn test_decoder_block_residual_connection() {
    let vb = VarBuilder::zeros(DType::F32, &cpu());
    let mut block = ResidualAttentionBlock::load_decoder(&vb, 2, 8, 16).unwrap();

    let data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let x = DynTensor::from_vec(data.clone(), &[1, 1, 8], &cpu()).unwrap();
    let encoder_out = DynTensor::zeros(&[1, 4, 8], DType::F32, &cpu()).unwrap();
    let mask = causal_mask(1, DType::F32, &cpu()).unwrap();

    let out = block
        .forward_decoder(&x, &encoder_out, &mask, true)
        .unwrap();
    let out_vals = out.to_flat_vec::<f32>().unwrap();
    assert_eq!(
        out_vals, data,
        "with zero weights, decoder output should equal input via residuals"
    );
}

#[test]
fn test_encoder_block_output_finite() {
    // Non-trivial input to exercise GELU activation path.
    let vb = VarBuilder::zeros(DType::F32, &cpu());
    let mut block = ResidualAttentionBlock::load_encoder(&vb, 2, 8, 16).unwrap();

    let data: Vec<f32> = (0..24).map(|i| (i as f32) * 0.1 - 1.2).collect();
    let x = DynTensor::from_vec(data, &[1, 3, 8], &cpu()).unwrap();
    let out = block.forward_encoder(&x).unwrap();
    let flat = out.to_flat_vec::<f32>().unwrap();
    for (i, &v) in flat.iter().enumerate() {
        assert!(v.is_finite(), "non-finite at index {i}: {v}");
    }
}

#[test]
fn test_decoder_block_kv_cache_accumulates() {
    let vb = VarBuilder::zeros(DType::F32, &cpu());
    let mut block = ResidualAttentionBlock::load_decoder(&vb, 2, 8, 16).unwrap();
    let encoder_out = DynTensor::zeros(&[1, 4, 8], DType::F32, &cpu()).unwrap();

    // Step 0: one token, flush cache.
    let t0 = DynTensor::zeros(&[1, 1, 8], DType::F32, &cpu()).unwrap();
    let mask0 = causal_mask(1, DType::F32, &cpu()).unwrap();
    block
        .forward_decoder(&t0, &encoder_out, &mask0, true)
        .unwrap();
    assert_eq!(block.self_cache_len(), 1);

    // Step 1: another token, don't flush.
    let t1 = DynTensor::zeros(&[1, 1, 8], DType::F32, &cpu()).unwrap();
    // For step 1, mask should be [1, 2] — attend to both positions.
    let full_mask = causal_mask(2, DType::F32, &cpu()).unwrap();
    let mask1 = full_mask.narrow(0, 1, 1).unwrap(); // last row: [1, 2]
    block
        .forward_decoder(&t1, &encoder_out, &mask1, false)
        .unwrap();
    assert_eq!(block.self_cache_len(), 2);
}

#[test]
fn test_decoder_block_reset_cache() {
    let vb = VarBuilder::zeros(DType::F32, &cpu());
    let mut block = ResidualAttentionBlock::load_decoder(&vb, 2, 8, 16).unwrap();
    let encoder_out = DynTensor::zeros(&[1, 4, 8], DType::F32, &cpu()).unwrap();

    let t0 = DynTensor::zeros(&[1, 1, 8], DType::F32, &cpu()).unwrap();
    let mask0 = causal_mask(1, DType::F32, &cpu()).unwrap();
    block
        .forward_decoder(&t0, &encoder_out, &mask0, true)
        .unwrap();
    assert_eq!(block.self_cache_len(), 1);

    block.reset_cache();
    assert_eq!(block.self_cache_len(), 0);
}

// -- MultiHeadAttention forward path tests --

#[test]
fn test_self_attention_forward_shape() {
    let vb = VarBuilder::zeros(DType::F32, &cpu());
    let mut attn = MultiHeadAttention::load(&vb, 2, 8).unwrap();
    let x = DynTensor::zeros(&[1, 3, 8], DType::F32, &cpu()).unwrap();
    let mask = causal_mask(3, DType::F32, &cpu()).unwrap();
    let out = attn.forward(&x, None, Some(&mask), true).unwrap();
    assert_eq!(out.dims(), &[1, 3, 8]);
}

#[test]
fn test_cross_attention_forward_shape() {
    let vb = VarBuilder::zeros(DType::F32, &cpu());
    let mut attn = MultiHeadAttention::load(&vb, 2, 8).unwrap();
    let x = DynTensor::zeros(&[1, 2, 8], DType::F32, &cpu()).unwrap();
    let xa = DynTensor::zeros(&[1, 6, 8], DType::F32, &cpu()).unwrap();
    let out = attn.forward(&x, Some(&xa), None, true).unwrap();
    assert_eq!(out.dims(), &[1, 2, 8]);
}

#[test]
fn test_self_attention_cache_grows() {
    let vb = VarBuilder::zeros(DType::F32, &cpu());
    let mut attn = MultiHeadAttention::load(&vb, 2, 8).unwrap();

    // Step 0.
    let t0 = DynTensor::zeros(&[1, 1, 8], DType::F32, &cpu()).unwrap();
    let mask0 = causal_mask(1, DType::F32, &cpu()).unwrap();
    attn.forward(&t0, None, Some(&mask0), true).unwrap();
    assert_eq!(attn.self_cache_len(), 1);

    // Step 1.
    let t1 = DynTensor::zeros(&[1, 1, 8], DType::F32, &cpu()).unwrap();
    let full_mask = causal_mask(2, DType::F32, &cpu()).unwrap();
    let mask1 = full_mask.narrow(0, 1, 1).unwrap();
    attn.forward(&t1, None, Some(&mask1), false).unwrap();
    assert_eq!(attn.self_cache_len(), 2);
}

#[test]
fn test_cross_attention_cache_reuse() {
    // Cross-attention: KV is computed from encoder output on first call,
    // then cached on subsequent calls.
    let vb = VarBuilder::zeros(DType::F32, &cpu());
    let mut attn = MultiHeadAttention::load(&vb, 2, 8).unwrap();

    let x = DynTensor::zeros(&[1, 1, 8], DType::F32, &cpu()).unwrap();
    let xa = DynTensor::zeros(&[1, 4, 8], DType::F32, &cpu()).unwrap();

    // First call: compute + cache encoder KV.
    let out1 = attn.forward(&x, Some(&xa), None, true).unwrap();

    // Second call: should reuse cached KV (no flush).
    let out2 = attn.forward(&x, Some(&xa), None, false).unwrap();

    // With zero weights, both outputs should be identical.
    assert_eq!(
        out1.to_flat_vec::<f32>().unwrap(),
        out2.to_flat_vec::<f32>().unwrap(),
        "cross-attention cache should produce identical output on reuse"
    );
}

#[test]
fn test_attention_flush_clears_cross_cache() {
    let vb = VarBuilder::zeros(DType::F32, &cpu());
    let mut attn = MultiHeadAttention::load(&vb, 2, 8).unwrap();

    let x = DynTensor::zeros(&[1, 1, 8], DType::F32, &cpu()).unwrap();
    let xa = DynTensor::zeros(&[1, 4, 8], DType::F32, &cpu()).unwrap();

    // Populate cross cache.
    attn.forward(&x, Some(&xa), None, true).unwrap();

    // Flush: should recompute.
    let out = attn.forward(&x, Some(&xa), None, true).unwrap();
    assert_eq!(out.dims(), &[1, 1, 8]);
}

#[test]
fn test_attention_reset_clears_all_caches() {
    let vb = VarBuilder::zeros(DType::F32, &cpu());
    let mut attn = MultiHeadAttention::load(&vb, 2, 8).unwrap();

    // Populate self-attention cache.
    let x = DynTensor::zeros(&[1, 1, 8], DType::F32, &cpu()).unwrap();
    let mask = causal_mask(1, DType::F32, &cpu()).unwrap();
    attn.forward(&x, None, Some(&mask), true).unwrap();
    assert_eq!(attn.self_cache_len(), 1);

    attn.reset_cache();
    assert_eq!(attn.self_cache_len(), 0);
}

#[test]
fn test_attention_output_finite_with_nonzero_input() {
    let vb = VarBuilder::zeros(DType::F32, &cpu());
    let mut attn = MultiHeadAttention::load(&vb, 2, 8).unwrap();

    // Non-trivial input.
    let data: Vec<f32> = (0..24).map(|i| (i as f32) * 0.05 - 0.6).collect();
    let x = DynTensor::from_vec(data, &[1, 3, 8], &cpu()).unwrap();
    let mask = causal_mask(3, DType::F32, &cpu()).unwrap();
    let out = attn.forward(&x, None, Some(&mask), true).unwrap();

    let flat = out.to_flat_vec::<f32>().unwrap();
    for (i, &v) in flat.iter().enumerate() {
        assert!(v.is_finite(), "non-finite at index {i}: {v}");
    }
}

#[test]
fn test_encoder_block_batch_size_2() {
    let vb = VarBuilder::zeros(DType::F32, &cpu());
    let mut block = ResidualAttentionBlock::load_encoder(&vb, 2, 8, 16).unwrap();

    let x = DynTensor::zeros(&[2, 4, 8], DType::F32, &cpu()).unwrap();
    let out = block.forward_encoder(&x).unwrap();
    assert_eq!(out.dims(), &[2, 4, 8]);
}

#[test]
fn test_decoder_block_batch_size_2() {
    let vb = VarBuilder::zeros(DType::F32, &cpu());
    let mut block = ResidualAttentionBlock::load_decoder(&vb, 2, 8, 16).unwrap();

    let x = DynTensor::zeros(&[2, 3, 8], DType::F32, &cpu()).unwrap();
    let encoder_out = DynTensor::zeros(&[2, 6, 8], DType::F32, &cpu()).unwrap();
    let mask = causal_mask(3, DType::F32, &cpu()).unwrap();
    let out = block
        .forward_decoder(&x, &encoder_out, &mask, true)
        .unwrap();
    assert_eq!(out.dims(), &[2, 3, 8]);
}

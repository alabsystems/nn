// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

use super::*;
use nn_core::dyn_tensor::DynTensor;
use nn_core::{DType, Device};

#[test]
fn test_check_output_finite_rejects_nan() {
    let nan_tensor =
        DynTensor::from_vec(vec![1.0, f32::NAN, 3.0], &[1, 1, 3], &Device::Cpu).unwrap();
    let err = check_output_finite(&nan_tensor, "MultiHeadAttention").unwrap_err();
    let msg = format!("{err:?}");
    assert!(
        msg.contains("NonFiniteData"),
        "expected NonFiniteData, got: {msg}"
    );
}

#[test]
fn test_check_output_finite_accepts_valid() {
    let valid = DynTensor::from_vec(vec![1.0, 2.0, 3.0], &[1, 1, 3], &Device::Cpu).unwrap();
    check_output_finite(&valid, "MultiHeadAttention").unwrap();
}

#[test]
fn test_cross_attention_batch_mismatch_error() {
    use nn_core::DType;
    let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
    let n_heads = 2;
    let d_model = 4;
    let mut attn = MultiHeadAttention::load(&vb, n_heads, d_model).unwrap();

    // Decoder input: batch=1
    let x = DynTensor::zeros(&[1, 3, d_model], DType::F32, &Device::Cpu).unwrap();
    // Encoder output: batch=2 (mismatch)
    let xa = DynTensor::zeros(&[2, 8, d_model], DType::F32, &Device::Cpu).unwrap();

    let err = attn.forward(&x, Some(&xa), None, true).unwrap_err();
    let msg = format!("{err}");
    assert!(
        msg.contains("encoder batch size (2) != decoder batch size (1)"),
        "expected batch mismatch error, got: {msg}"
    );
}

#[test]
fn test_load_zero_heads_returns_error() {
    use nn_core::DType;
    let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
    let result = MultiHeadAttention::load(&vb, 0, 64);
    assert!(result.is_err());
    let msg = format!("{}", result.err().unwrap());
    assert!(
        msg.contains("n_heads must be > 0"),
        "expected n_heads error, got: {msg}"
    );
}

#[test]
fn test_load_indivisible_d_model_returns_error() {
    use nn_core::DType;
    let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
    let result = MultiHeadAttention::load(&vb, 3, 64);
    assert!(result.is_err());
    let msg = format!("{}", result.err().unwrap());
    assert!(
        msg.contains("d_model (64) must be divisible by n_heads (3)"),
        "expected divisibility error, got: {msg}"
    );
}

#[test]
fn test_cross_attention_stale_cache_detected() {
    use nn_core::DType;
    let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
    let n_heads = 2;
    let d_model = 4;
    let mut attn = MultiHeadAttention::load(&vb, n_heads, d_model).unwrap();

    let x = DynTensor::zeros(&[1, 1, d_model], DType::F32, &Device::Cpu).unwrap();

    // First call with encoder output of seq_len=8 — populates cache.
    let enc_a = DynTensor::zeros(&[1, 8, d_model], DType::F32, &Device::Cpu).unwrap();
    attn.forward(&x, Some(&enc_a), None, true).unwrap();

    // Second call with different seq_len=12 WITHOUT flushing — should error.
    let enc_b = DynTensor::zeros(&[1, 12, d_model], DType::F32, &Device::Cpu).unwrap();
    let err = attn.forward(&x, Some(&enc_b), None, false).unwrap_err();
    let msg = format!("{err}");
    assert!(
        msg.contains("cross-attention KV cache seq_len"),
        "expected stale cache error, got: {msg}"
    );
}

#[test]
fn test_cross_attention_flush_allows_different_encoder() {
    use nn_core::DType;
    let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
    let n_heads = 2;
    let d_model = 4;
    let mut attn = MultiHeadAttention::load(&vb, n_heads, d_model).unwrap();

    let x = DynTensor::zeros(&[1, 1, d_model], DType::F32, &Device::Cpu).unwrap();

    // First call with seq_len=8.
    let enc_a = DynTensor::zeros(&[1, 8, d_model], DType::F32, &Device::Cpu).unwrap();
    attn.forward(&x, Some(&enc_a), None, true).unwrap();

    // Second call with seq_len=12 WITH flushing — should succeed.
    let enc_b = DynTensor::zeros(&[1, 12, d_model], DType::F32, &Device::Cpu).unwrap();
    attn.forward(&x, Some(&enc_b), None, true).unwrap();
}

// -- Flash Attention optimization tests (#2981) --

/// Self-attention with seq_len=1 and causal mask activates the mask-skip path.
/// The causal mask at any position P for seq_len=1 is all-zeros (every cached
/// position is visible), so passing None to sdpa is semantically equivalent.
#[test]
fn test_self_attn_seq1_with_mask_produces_correct_output() {
    let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
    let n_heads = 2;
    let d_model = 8;
    let mut attn = MultiHeadAttention::load(&vb, n_heads, d_model).unwrap();

    // Build causal mask [4, 4] like the decoder does.
    let mask = crate::positional::causal_mask(4, DType::F32, &Device::Cpu).unwrap();

    // First step: seq_len=4, flush=true → sdpa_causal path (S_q == S_kv).
    let x0 = DynTensor::zeros(&[1, 4, d_model], DType::F32, &Device::Cpu).unwrap();
    let mask0 = mask.narrow(0, 0, 4).unwrap().narrow(1, 0, 4).unwrap();
    let out0 = attn.forward(&x0, None, Some(&mask0), true).unwrap();
    assert_eq!(out0.dims(), &[1, 4, d_model]);
    check_output_finite(&out0, "test").unwrap();

    // Subsequent step: seq_len=1, offset=3 → mask-skip path.
    let x1 = DynTensor::zeros(&[1, 1, d_model], DType::F32, &Device::Cpu).unwrap();
    // Slice mask row 3, columns 0..5 (total_kv = 4 cached + 1 new = 5).
    // Use a fresh 5-wide mask since the original is only 4×4.
    let big_mask = crate::positional::causal_mask(8, DType::F32, &Device::Cpu).unwrap();
    let mask1 = big_mask.narrow(0, 4, 1).unwrap().narrow(1, 0, 5).unwrap();
    let out1 = attn.forward(&x1, None, Some(&mask1), false).unwrap();
    assert_eq!(out1.dims(), &[1, 1, d_model]);
    check_output_finite(&out1, "test").unwrap();
}

/// Multi-step autoregressive decode exercises the seq_len=1 optimization
/// on every step after the initial prompt.
#[test]
fn test_multi_step_decode_shapes_consistent() {
    let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
    let n_heads = 2;
    let d_model = 8;
    let max_pos = 16;
    let mut attn = MultiHeadAttention::load(&vb, n_heads, d_model).unwrap();

    let mask = crate::positional::causal_mask(max_pos, DType::F32, &Device::Cpu).unwrap();

    // Initial prompt: seq_len=3
    let prompt = DynTensor::zeros(&[1, 3, d_model], DType::F32, &Device::Cpu).unwrap();
    let m = mask.narrow(0, 0, 3).unwrap().narrow(1, 0, 3).unwrap();
    let out = attn.forward(&prompt, None, Some(&m), true).unwrap();
    assert_eq!(out.dims(), &[1, 3, d_model]);

    // 5 autoregressive steps: seq_len=1 each
    for step in 0..5 {
        let offset = 3 + step;
        let total_kv = offset + 1;
        let x = DynTensor::zeros(&[1, 1, d_model], DType::F32, &Device::Cpu).unwrap();
        let m = mask
            .narrow(0, offset, 1)
            .unwrap()
            .narrow(1, 0, total_kv)
            .unwrap();
        let out = attn.forward(&x, None, Some(&m), false).unwrap();
        assert_eq!(out.dims(), &[1, 1, d_model], "step {step}");
        check_output_finite(&out, "test").unwrap();
    }
}

/// The no-cache masked path uses sdpa_causal when mask is provided.
#[test]
fn test_no_cache_masked_uses_sdpa_causal() {
    let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
    let n_heads = 2;
    let d_model = 8;
    let attn = MultiHeadAttention::load(&vb, n_heads, d_model).unwrap();

    let x = DynTensor::zeros(&[1, 6, d_model], DType::F32, &Device::Cpu).unwrap();
    let mask = crate::positional::causal_mask(6, DType::F32, &Device::Cpu).unwrap();

    let out = attn
        .forward_self_attn_no_cache_masked(&x, Some(&mask))
        .unwrap();
    assert_eq!(out.dims(), &[1, 6, d_model]);
    check_output_finite(&out, "test").unwrap();

    // Without mask should also work (passes through to sdpa with None).
    let out_no_mask = attn.forward_self_attn_no_cache_masked(&x, None).unwrap();
    assert_eq!(out_no_mask.dims(), &[1, 6, d_model]);
    check_output_finite(&out_no_mask, "test").unwrap();
}

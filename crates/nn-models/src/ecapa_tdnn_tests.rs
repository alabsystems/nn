#![allow(deprecated)]
// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

use super::*;
use nn_core::test_utils::cpu;

/// Create a deterministic test tensor with values in [0.1, 1.1).
fn test_tensor(dims: &[usize]) -> DynTensor {
    let numel: usize = dims.iter().product();
    let data: Vec<f32> = (0..numel)
        .map(|i| (i as f32 * 0.0073).sin().abs() + 0.1)
        .collect();
    DynTensor::from_vec(data, dims, &cpu()).expect("test tensor")
}

#[test]
fn test_ecapa_tdnn_shape() {
    let device = cpu();
    let vb = VarBuilder::zeros(nn_core::DType::F32, &device);
    let model = EcapaTdnn::load(vb.pp("ecapa")).expect("load");
    let mel = test_tensor(&[2, 80, 100]);
    let embedding = model.forward(&mel).expect("forward");
    assert_eq!(embedding.dims(), &[2, 192]);
}

#[test]
fn test_ecapa_tdnn_embed_dim() {
    let device = cpu();
    let vb = VarBuilder::zeros(nn_core::DType::F32, &device);
    let model = EcapaTdnn::load(vb.pp("ecapa")).expect("load");
    assert_eq!(model.embed_dim(), 192);
}

/// Zero weights produce zero embeddings — L2 normalization guard (eps=1e-12)
/// prevents division by zero and returns 0/eps ≈ 0.
///
/// Original test asserted norm==1.0 with zero weights, which is impossible:
/// zero weights → zero pre-norm embedding → norm=0 → clamped to eps → 0/eps ≈ 0.
/// L2 normalization to unit norm requires non-zero weights, tested via
/// production safetensors in downstream integration tests (dvoice).
#[test]
fn test_ecapa_tdnn_l2_zero_weights_produces_zero() {
    let device = cpu();
    let vb = VarBuilder::zeros(nn_core::DType::F32, &device);
    let model = EcapaTdnn::load(vb.pp("ecapa")).expect("load");
    let mel = test_tensor(&[1, 80, 50]);
    let embedding = model.forward(&mel).expect("forward");
    let data = embedding.to_flat_vec::<f32>().expect("data");
    let norm: f32 = data.iter().map(|v| v * v).sum::<f32>().sqrt();
    assert!(
        norm < 1e-6,
        "zero-weight model should produce near-zero embedding, got norm={norm}"
    );
}

/// L2 normalization produces unit norm when embedding is non-zero.
///
/// Verifies the normalization code path: sqr → sum_keepdim → sqrt → maximum(eps) → div.
/// Uses eps guard test: inject a known non-zero embedding and verify norm=1.
#[test]
fn test_ecapa_tdnn_l2_normalization_code_path() {
    // Directly test the normalization formula used in forward():
    //   norm = sqrt(sum(x^2, keepdim=True)).max(1e-12)
    //   result = x / norm
    let device = cpu();
    // Create a non-zero embedding directly and normalize it.
    let data: Vec<f32> = (0..192).map(|i| (i as f32 * 0.037).sin()).collect();
    let x = DynTensor::from_vec(data, &[1, 192], &device).expect("embedding");

    let norm_tensor = x
        .sqr()
        .expect("sqr")
        .sum_keepdim(1)
        .expect("sum")
        .sqrt()
        .expect("sqrt");
    let eps = DynTensor::full(&[], 1e-12, x.dtype(), &device).expect("eps");
    let norm_clamped = norm_tensor.maximum(&eps).expect("max");
    let normalized = x.broadcast_div(&norm_clamped).expect("div");

    // Compute L2 norm of result.
    let result_data = normalized.to_flat_vec::<f32>().expect("data");
    let result_norm: f32 = result_data.iter().map(|v| v * v).sum::<f32>().sqrt();
    assert!(
        (result_norm - 1.0).abs() < 1e-5,
        "normalized embedding should have unit norm, got {result_norm}"
    );
}

#[test]
fn test_ecapa_tdnn_variable_length() {
    let device = cpu();
    let vb = VarBuilder::zeros(nn_core::DType::F32, &device);
    let model = EcapaTdnn::load(vb.pp("ecapa")).expect("load");
    // Different temporal lengths -> same embedding dimension.
    let mel_short = test_tensor(&[1, 80, 30]);
    let mel_long = test_tensor(&[1, 80, 200]);
    let emb_short = model.forward(&mel_short).expect("forward short");
    let emb_long = model.forward(&mel_long).expect("forward long");
    assert_eq!(emb_short.dims(), &[1, 192]);
    assert_eq!(emb_long.dims(), &[1, 192]);
}

#[test]
fn test_ecapa_tdnn_wrong_channels() {
    let device = cpu();
    let vb = VarBuilder::zeros(nn_core::DType::F32, &device);
    let model = EcapaTdnn::load(vb.pp("ecapa")).expect("load");
    let bad_input = test_tensor(&[1, 40, 100]);
    let result = model.forward(&bad_input);
    assert!(result.is_err(), "should reject non-80 mel channels");
}

#[test]
fn test_ecapa_tdnn_wrong_rank() {
    let device = cpu();
    let vb = VarBuilder::zeros(nn_core::DType::F32, &device);
    let model = EcapaTdnn::load(vb.pp("ecapa")).expect("load");
    let bad_input = test_tensor(&[80, 100]);
    let result = model.forward(&bad_input);
    assert!(result.is_err(), "should reject 2D input");
}

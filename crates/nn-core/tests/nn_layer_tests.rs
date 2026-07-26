// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Comprehensive integration tests for nn layers: Linear, Conv1d, Conv2d,
//! LayerNorm, BatchNorm, GroupNorm, RmsNorm, Embedding, MultiHeadAttention,
//! Dropout, LSTM, Sequential, Activation, InstanceNorm, SwiGlu.
//!
//! These tests exercise forward-pass shape correctness, numerical properties,
//! and error-path validation from *outside* the crate (integration test),
//! ensuring the public API surface works as documented.

use nn_core::layers::{
    Activation, BatchNorm, Conv1d, Conv1dConfig, Conv2d, Conv2dConfig, Dropout, Embedding,
    GroupNorm, InstanceNorm, LayerNorm, Linear, Module, MultiHeadAttention, RmsNorm, Sequential,
    SwiGlu,
};
use nn_core::{DType, Device, DynTensor};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn cpu() -> Device {
    Device::Cpu
}

fn ones(shape: &[usize]) -> DynTensor {
    DynTensor::ones(shape, DType::F32, &cpu()).unwrap()
}

fn zeros(shape: &[usize]) -> DynTensor {
    DynTensor::zeros(shape, DType::F32, &cpu()).unwrap()
}

fn randn_like(shape: &[usize]) -> DynTensor {
    // Deterministic pseudo-random data for reproducible tests.
    let numel: usize = shape.iter().product();
    let mut data = Vec::with_capacity(numel);
    let mut state: u64 = 42;
    for _ in 0..numel {
        // Simple xorshift64 -> map to [-1, 1].
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        let val = (state as f32) / (u64::MAX as f32) * 2.0 - 1.0;
        data.push(val);
    }
    DynTensor::from_vec(data, shape, &cpu()).unwrap()
}

fn full(shape: &[usize], val: f64) -> DynTensor {
    DynTensor::full(shape, val, DType::F32, &cpu()).unwrap()
}

// ===========================================================================
// Linear
// ===========================================================================

#[test]
fn test_nn_layer_linear_forward_shape() {
    // weight [out=4, in=3], input [B=2, 3] -> output [2, 4]
    let w = ones(&[4, 3]);
    let linear = Linear::new(w, None).unwrap();
    let x = ones(&[2, 3]);
    let y = linear.forward(&x).unwrap();
    assert_eq!(y.dims(), &[2, 4]);
}

#[test]
fn test_nn_layer_linear_weight_application() {
    // Identity-like weight: only first input feature passes through.
    let w = DynTensor::from_vec(vec![1.0, 0.0, 0.0, 0.0, 0.0, 1.0], &[2, 3], &cpu()).unwrap();
    let linear = Linear::new(w, None).unwrap();
    let x = DynTensor::from_vec(vec![7.0, 8.0, 9.0], &[1, 3], &cpu()).unwrap();
    let y = linear.forward(&x).unwrap();
    let vals = y.to_flat_vec::<f32>().unwrap();
    assert!((vals[0] - 7.0).abs() < 1e-5, "got {}", vals[0]);
    assert!((vals[1] - 9.0).abs() < 1e-5, "got {}", vals[1]);
}

#[test]
fn test_nn_layer_linear_with_bias() {
    let w = DynTensor::from_vec(vec![1.0, 0.0, 0.0, 1.0], &[2, 2], &cpu()).unwrap();
    let b = DynTensor::from_vec(vec![100.0, 200.0], &[2], &cpu()).unwrap();
    let linear = Linear::new(w, Some(b)).unwrap();
    let x = DynTensor::from_vec(vec![3.0, 4.0], &[1, 2], &cpu()).unwrap();
    let y = linear.forward(&x).unwrap();
    let vals = y.to_flat_vec::<f32>().unwrap();
    assert!((vals[0] - 103.0).abs() < 1e-5);
    assert!((vals[1] - 204.0).abs() < 1e-5);
}

#[test]
fn test_nn_layer_linear_no_bias_variant() {
    let w = ones(&[2, 3]);
    let linear = Linear::new(w, None).unwrap();
    assert!(linear.bias().is_none());
    assert_eq!(linear.out_features(), 2);
    assert_eq!(linear.in_features(), 3);
}

#[test]
fn test_nn_layer_linear_3d_input() {
    // [B, S, in_features] -> [B, S, out_features]
    let w = ones(&[5, 4]);
    let linear = Linear::new(w, None).unwrap();
    let x = ones(&[2, 3, 4]);
    let y = linear.forward(&x).unwrap();
    assert_eq!(y.dims(), &[2, 3, 5]);
}

// ===========================================================================
// Conv1d
// ===========================================================================

#[test]
fn test_nn_layer_conv1d_output_shape_default() {
    // weight [out=2, in=1, kernel=3], input [B=1, C=1, L=10]
    // output L = (10 - 3) / 1 + 1 = 8
    let w = ones(&[2, 1, 3]);
    let conv = Conv1d::new(w, None, Conv1dConfig::default()).unwrap();
    let x = ones(&[1, 1, 10]);
    let y = conv.forward(&x).unwrap();
    assert_eq!(y.dims(), &[1, 2, 8]);
}

#[test]
fn test_nn_layer_conv1d_with_stride() {
    // stride=2, kernel=3, input L=10 -> output L = (10 - 3) / 2 + 1 = 4
    let w = ones(&[1, 1, 3]);
    let cfg = Conv1dConfig::new(0, 2, 1);
    let conv = Conv1d::new(w, None, cfg).unwrap();
    let x = ones(&[1, 1, 10]);
    let y = conv.forward(&x).unwrap();
    assert_eq!(y.dims(), &[1, 1, 4]);
}

#[test]
fn test_nn_layer_conv1d_with_padding() {
    // padding=1, kernel=3, stride=1, input L=5 -> output L = (5 + 2*1 - 3) / 1 + 1 = 5
    let w = ones(&[1, 1, 3]);
    let cfg = Conv1dConfig::new(1, 1, 1);
    let conv = Conv1d::new(w, None, cfg).unwrap();
    let x = ones(&[1, 1, 5]);
    let y = conv.forward(&x).unwrap();
    assert_eq!(y.dims(), &[1, 1, 5]);
}

#[test]
fn test_nn_layer_conv1d_with_dilation() {
    // dilation=2, kernel=3, effective kernel = 3 + (3-1)*(2-1) = 5
    // input L=10 -> output L = (10 - 5) / 1 + 1 = 6
    let w = ones(&[1, 1, 3]);
    let cfg = Conv1dConfig::new(0, 1, 2);
    let conv = Conv1d::new(w, None, cfg).unwrap();
    let x = ones(&[1, 1, 10]);
    let y = conv.forward(&x).unwrap();
    assert_eq!(y.dims(), &[1, 1, 6]);
}

#[test]
fn test_nn_layer_conv1d_with_bias() {
    let w = DynTensor::from_vec(vec![1.0, 1.0, 1.0], &[1, 1, 3], &cpu()).unwrap();
    let b = DynTensor::from_vec(vec![100.0], &[1], &cpu()).unwrap();
    let conv = Conv1d::new(w, Some(b), Conv1dConfig::default()).unwrap();
    let x = DynTensor::from_vec(vec![1.0, 1.0, 1.0, 1.0, 1.0], &[1, 1, 5], &cpu()).unwrap();
    let y = conv.forward(&x).unwrap();
    let vals = y.to_flat_vec::<f32>().unwrap();
    // Each output = sum of 3 ones + bias 100 = 103
    for v in &vals {
        assert!((*v - 103.0).abs() < 1e-5, "expected 103.0, got {v}");
    }
}

// ===========================================================================
// Conv2d
// ===========================================================================

#[test]
fn test_nn_layer_conv2d_output_shape_default() {
    // weight [out=4, in=3, kH=3, kW=3], input [B=1, C=3, H=8, W=8]
    // output H = (8 - 3) / 1 + 1 = 6, W = 6
    let w = ones(&[4, 3, 3, 3]);
    let conv = Conv2d::new(w, None, Conv2dConfig::default()).unwrap();
    let x = ones(&[1, 3, 8, 8]);
    let y = conv.forward(&x).unwrap();
    assert_eq!(y.dims(), &[1, 4, 6, 6]);
}

#[test]
fn test_nn_layer_conv2d_with_padding_stride() {
    // padding=1, stride=2, kernel=3x3, input 8x8
    // output = (8 + 2 - 3) / 2 + 1 = 4
    let w = ones(&[2, 1, 3, 3]);
    let cfg = Conv2dConfig::new(1, 2, 1);
    let conv = Conv2d::new(w, None, cfg).unwrap();
    let x = ones(&[1, 1, 8, 8]);
    let y = conv.forward(&x).unwrap();
    assert_eq!(y.dims(), &[1, 2, 4, 4]);
}

#[test]
fn test_nn_layer_conv2d_with_bias() {
    let w = ones(&[2, 1, 1, 1]); // 1x1 convolution
    let b = DynTensor::from_vec(vec![10.0, 20.0], &[2], &cpu()).unwrap();
    let conv = Conv2d::new(w, Some(b), Conv2dConfig::default()).unwrap();
    let x = DynTensor::from_vec(vec![5.0, 5.0, 5.0, 5.0], &[1, 1, 2, 2], &cpu()).unwrap();
    let y = conv.forward(&x).unwrap();
    assert_eq!(y.dims(), &[1, 2, 2, 2]);
    let vals = y.to_flat_vec::<f32>().unwrap();
    // Channel 0: 5*1 + 10 = 15, Channel 1: 5*1 + 20 = 25
    for v in &vals[..4] {
        assert!((*v - 15.0).abs() < 1e-5, "ch0 expected 15, got {v}");
    }
    for v in &vals[4..] {
        assert!((*v - 25.0).abs() < 1e-5, "ch1 expected 25, got {v}");
    }
}

#[test]
fn test_nn_layer_conv2d_rejects_3d_weight() {
    let w = ones(&[2, 1, 3]); // 3D, not 4D
    let err = Conv2d::new(w, None, Conv2dConfig::default()).unwrap_err();
    assert!(
        err.to_string().contains("rank") || err.to_string().contains("Rank"),
        "expected rank error, got: {err}"
    );
}

// ===========================================================================
// LayerNorm
// ===========================================================================

#[test]
fn test_nn_layer_layer_norm_shape_preserved() {
    let weight = ones(&[8]);
    let bias = zeros(&[8]);
    let ln = LayerNorm::new(weight, bias, 1e-5).unwrap();
    let x = randn_like(&[2, 3, 8]);
    let y = ln.forward(&x).unwrap();
    assert_eq!(y.dims(), &[2, 3, 8]);
}

#[test]
fn test_nn_layer_layer_norm_output_mean_near_zero() {
    let dim = 64;
    let weight = ones(&[dim]);
    let bias = zeros(&[dim]);
    let ln = LayerNorm::new(weight, bias, 1e-5).unwrap();
    let x = randn_like(&[1, dim]);
    let y = ln.forward(&x).unwrap();
    let vals = y.to_flat_vec::<f32>().unwrap();
    let mean: f32 = vals.iter().sum::<f32>() / vals.len() as f32;
    assert!(
        mean.abs() < 0.01,
        "LayerNorm output mean should be near 0, got {mean}"
    );
}

#[test]
fn test_nn_layer_layer_norm_output_variance_near_one() {
    let dim = 64;
    let weight = ones(&[dim]);
    let bias = zeros(&[dim]);
    let ln = LayerNorm::new(weight, bias, 1e-5).unwrap();
    let x = randn_like(&[1, dim]);
    let y = ln.forward(&x).unwrap();
    let vals = y.to_flat_vec::<f32>().unwrap();
    let mean: f32 = vals.iter().sum::<f32>() / vals.len() as f32;
    let var: f32 = vals.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / vals.len() as f32;
    assert!(
        (var - 1.0).abs() < 0.1,
        "LayerNorm output variance should be near 1, got {var}"
    );
}

#[test]
fn test_nn_layer_layer_norm_with_affine() {
    // weight=2, bias=5, input [1, -1] -> normalized [1, -1] -> *2 + 5 = [7, 3]
    let weight = full(&[2], 2.0);
    let bias = full(&[2], 5.0);
    let ln = LayerNorm::new(weight, bias, 0.0).unwrap();
    let x = DynTensor::from_vec(vec![1.0, -1.0], &[1, 2], &cpu()).unwrap();
    let y = ln.forward(&x).unwrap();
    let vals = y.to_flat_vec::<f32>().unwrap();
    assert!(
        (vals[0] - 7.0).abs() < 1e-4,
        "expected 7.0, got {}",
        vals[0]
    );
    assert!(
        (vals[1] - 3.0).abs() < 1e-4,
        "expected 3.0, got {}",
        vals[1]
    );
}

// ===========================================================================
// BatchNorm (inference mode)
// ===========================================================================

#[test]
fn test_nn_layer_batch_norm_inference_shape_preserved() {
    let c = 4;
    let mean = zeros(&[c]);
    let var = ones(&[c]);
    let bn = BatchNorm::new(mean, var, None, None, 1e-5).unwrap();
    let x = randn_like(&[2, c, 8]);
    let y = bn.forward(&x).unwrap();
    assert_eq!(y.dims(), &[2, c, 8]);
}

#[test]
fn test_nn_layer_batch_norm_known_values() {
    // running_mean=0, running_var=1, eps=0 -> identity normalization
    let mean = zeros(&[2]);
    let var = ones(&[2]);
    let bn = BatchNorm::new(mean, var, None, None, 0.0).unwrap();
    let x = DynTensor::from_vec(vec![3.0, 7.0], &[1, 2], &cpu()).unwrap();
    let y = bn.forward(&x).unwrap();
    let vals = y.to_flat_vec::<f32>().unwrap();
    assert!(
        (vals[0] - 3.0).abs() < 1e-5,
        "expected 3.0, got {}",
        vals[0]
    );
    assert!(
        (vals[1] - 7.0).abs() < 1e-5,
        "expected 7.0, got {}",
        vals[1]
    );
}

#[test]
fn test_nn_layer_batch_norm_with_affine() {
    // running_mean=0, running_var=1, weight=2, bias=10 -> y = 2*x + 10
    let mean = zeros(&[1]);
    let var = ones(&[1]);
    let weight = full(&[1], 2.0);
    let bias = full(&[1], 10.0);
    let bn = BatchNorm::new(mean, var, Some(weight), Some(bias), 0.0).unwrap();
    let x = DynTensor::from_vec(vec![5.0], &[1, 1], &cpu()).unwrap();
    let y = bn.forward(&x).unwrap();
    let vals = y.to_flat_vec::<f32>().unwrap();
    assert!(
        (vals[0] - 20.0).abs() < 1e-4,
        "expected 20.0, got {}",
        vals[0]
    );
}

#[test]
fn test_nn_layer_batch_norm_4d_input() {
    let c = 3;
    let mean = zeros(&[c]);
    let var = ones(&[c]);
    let bn = BatchNorm::new(mean, var, None, None, 1e-5).unwrap();
    let x = ones(&[2, c, 4, 4]);
    let y = bn.forward(&x).unwrap();
    assert_eq!(y.dims(), &[2, c, 4, 4]);
}

// ===========================================================================
// GroupNorm
// ===========================================================================

#[test]
fn test_nn_layer_group_norm_output_shape() {
    let c = 8;
    let groups = 4;
    let w = ones(&[c]);
    let b = zeros(&[c]);
    let gn = GroupNorm::new(groups, c, w, b, 1e-5).unwrap();
    let x = randn_like(&[2, c, 16]);
    let y = gn.forward(&x).unwrap();
    assert_eq!(y.dims(), &[2, c, 16]);
}

#[test]
fn test_nn_layer_group_norm_constant_input() {
    // Constant input -> normalized to 0, then affine: bias only
    let c = 4;
    let w = ones(&[c]);
    let b = full(&[c], 42.0);
    let gn = GroupNorm::new(2, c, w, b, 1e-5).unwrap();
    let x = full(&[1, c, 3], 5.0);
    let y = gn.forward(&x).unwrap();
    let vals = y.to_flat_vec::<f32>().unwrap();
    for v in &vals {
        assert!(
            (*v - 42.0).abs() < 0.1,
            "constant input should normalize to 0, then bias=42, got {v}"
        );
    }
}

#[test]
fn test_nn_layer_group_norm_rejects_non_divisible() {
    let w = ones(&[5]);
    let b = zeros(&[5]);
    let err = GroupNorm::new(3, 5, w, b, 1e-5).unwrap_err();
    let msg = err.to_string();
    assert!(
        msg.contains("divisible") || msg.contains("Divisible") || msg.contains("num_channels"),
        "expected divisibility error, got: {msg}"
    );
}

#[test]
fn test_nn_layer_group_norm_4d_input() {
    let c = 4;
    let w = ones(&[c]);
    let b = zeros(&[c]);
    let gn = GroupNorm::new(2, c, w, b, 1e-5).unwrap();
    let x = randn_like(&[1, c, 3, 3]);
    let y = gn.forward(&x).unwrap();
    assert_eq!(y.dims(), &[1, c, 3, 3]);
}

// ===========================================================================
// RmsNorm
// ===========================================================================

#[test]
fn test_nn_layer_rms_norm_output_shape() {
    let dim = 16;
    let weight = ones(&[dim]);
    let rn = RmsNorm::new(weight, 1e-5).unwrap();
    let x = randn_like(&[2, 4, dim]);
    let y = rn.forward(&x).unwrap();
    assert_eq!(y.dims(), &[2, 4, dim]);
}

#[test]
fn test_nn_layer_rms_norm_normalization_property() {
    // For unit weight, RMS-normalized output should have RMS close to 1.
    let dim = 64;
    let weight = ones(&[dim]);
    let rn = RmsNorm::new(weight, 1e-6).unwrap();
    let x = randn_like(&[1, dim]);
    let y = rn.forward(&x).unwrap();
    let vals = y.to_flat_vec::<f32>().unwrap();
    let mean_sq: f32 = vals.iter().map(|v| v * v).sum::<f32>() / vals.len() as f32;
    let rms = mean_sq.sqrt();
    assert!(
        (rms - 1.0).abs() < 0.1,
        "RMS of RmsNorm output should be near 1, got {rms}"
    );
}

#[test]
fn test_nn_layer_rms_norm_with_weight_scaling() {
    // weight = 2.0 -> output scaled by 2x
    let weight = full(&[4], 2.0);
    let rn = RmsNorm::new(weight, 1e-5).unwrap();
    let x = DynTensor::from_vec(vec![1.0, 1.0, 1.0, 1.0], &[1, 4], &cpu()).unwrap();
    let y = rn.forward(&x).unwrap();
    let vals = y.to_flat_vec::<f32>().unwrap();
    // RMS of [1,1,1,1] = 1.0, so output = 2.0 * 1.0 / 1.0 = 2.0
    for v in &vals {
        assert!((*v - 2.0).abs() < 1e-4, "expected 2.0, got {v}");
    }
}

#[test]
fn test_nn_layer_rms_norm_rejects_rank0() {
    let weight = ones(&[1]);
    let rn = RmsNorm::new(weight, 1e-5).unwrap();
    let scalar = DynTensor::from_vec(vec![1.0], &[], &cpu()).unwrap();
    assert!(rn.forward(&scalar).is_err());
}

// ===========================================================================
// Embedding
// ===========================================================================

#[test]
fn test_nn_layer_embedding_output_shape_1d() {
    // vocab=10, dim=4, input [3] -> output [3, 4]
    let weight = randn_like(&[10, 4]);
    let emb = Embedding::new(weight).unwrap();
    let ids = DynTensor::from_vec_u32(vec![0, 5, 9], &[3], &cpu()).unwrap();
    let y = emb.forward(&ids).unwrap();
    assert_eq!(y.dims(), &[3, 4]);
}

#[test]
fn test_nn_layer_embedding_output_shape_2d() {
    // vocab=10, dim=4, input [B=2, S=3] -> output [2, 3, 4]
    let weight = randn_like(&[10, 4]);
    let emb = Embedding::new(weight).unwrap();
    let ids = DynTensor::from_vec_u32(vec![0, 1, 2, 3, 4, 5], &[2, 3], &cpu()).unwrap();
    let y = emb.forward(&ids).unwrap();
    assert_eq!(y.dims(), &[2, 3, 4]);
}

#[test]
fn test_nn_layer_embedding_correct_lookup() {
    let weight =
        DynTensor::from_vec(vec![10.0, 11.0, 20.0, 21.0, 30.0, 31.0], &[3, 2], &cpu()).unwrap();
    let emb = Embedding::new(weight).unwrap();
    let ids = DynTensor::from_vec_u32(vec![2, 0], &[2], &cpu()).unwrap();
    let y = emb.forward(&ids).unwrap();
    let vals = y.to_flat_vec::<f32>().unwrap();
    assert!((vals[0] - 30.0).abs() < 1e-6); // id 2
    assert!((vals[1] - 31.0).abs() < 1e-6);
    assert!((vals[2] - 10.0).abs() < 1e-6); // id 0
    assert!((vals[3] - 11.0).abs() < 1e-6);
}

#[test]
fn test_nn_layer_embedding_i64_indices() {
    let weight = randn_like(&[5, 3]);
    let emb = Embedding::new(weight).unwrap();
    let ids = DynTensor::from_vec_i64(vec![0, 2, 4], &[3], &cpu()).unwrap();
    let y = emb.forward(&ids).unwrap();
    assert_eq!(y.dims(), &[3, 3]);
}

#[test]
fn test_nn_layer_embedding_rejects_out_of_range() {
    let weight = ones(&[3, 2]);
    let emb = Embedding::new(weight).unwrap();
    let ids = DynTensor::from_vec_u32(vec![10], &[1], &cpu()).unwrap();
    assert!(emb.forward(&ids).is_err());
}

#[test]
fn test_nn_layer_embedding_rejects_non_2d_weight() {
    let w = ones(&[3, 2, 1]); // 3D
    assert!(Embedding::new(w).is_err());
}

// ===========================================================================
// MultiHeadAttention
// ===========================================================================

#[test]
fn test_nn_layer_mha_output_shape() {
    // dim=8, num_heads=2, num_kv_heads=2 -> head_dim=4
    let dim = 8;
    let num_heads = 2;
    let num_kv_heads = 2;
    let q_proj = Linear::new(ones(&[dim, dim]), None).unwrap();
    let k_proj = Linear::new(ones(&[dim, dim]), None).unwrap();
    let v_proj = Linear::new(ones(&[dim, dim]), None).unwrap();
    let out_proj = Linear::new(ones(&[dim, dim]), None).unwrap();
    let mha =
        MultiHeadAttention::new(q_proj, k_proj, v_proj, out_proj, num_heads, num_kv_heads).unwrap();
    // Input [B=1, S=4, D=8]
    let x = randn_like(&[1, 4, dim]);
    let y = mha.forward(&x, None, None, None, 0).unwrap();
    assert_eq!(y.dims(), &[1, 4, dim]);
}

#[test]
fn test_nn_layer_mha_gqa_output_shape() {
    // GQA: num_heads=4, num_kv_heads=2 -> 2 groups
    let dim = 16;
    let num_heads = 4;
    let num_kv_heads = 2;
    let head_dim = dim / num_heads; // 4
    let kv_dim = num_kv_heads * head_dim; // 8
    let q_proj = Linear::new(ones(&[dim, dim]), None).unwrap();
    let k_proj = Linear::new(ones(&[kv_dim, dim]), None).unwrap();
    let v_proj = Linear::new(ones(&[kv_dim, dim]), None).unwrap();
    let out_proj = Linear::new(ones(&[dim, dim]), None).unwrap();
    let mha =
        MultiHeadAttention::new(q_proj, k_proj, v_proj, out_proj, num_heads, num_kv_heads).unwrap();
    let x = randn_like(&[1, 3, dim]);
    let y = mha.forward(&x, None, None, None, 0).unwrap();
    assert_eq!(y.dims(), &[1, 3, dim]);
}

#[test]
fn test_nn_layer_mha_rejects_zero_heads() {
    let dim = 8;
    let q = Linear::new(ones(&[dim, dim]), None).unwrap();
    let k = Linear::new(ones(&[dim, dim]), None).unwrap();
    let v = Linear::new(ones(&[dim, dim]), None).unwrap();
    let o = Linear::new(ones(&[dim, dim]), None).unwrap();
    assert!(MultiHeadAttention::new(q, k, v, o, 0, 1).is_err());
}

#[test]
fn test_nn_layer_mha_rejects_non_divisible_heads() {
    let dim = 8;
    let q = Linear::new(ones(&[dim, dim]), None).unwrap();
    let k = Linear::new(ones(&[dim, dim]), None).unwrap();
    let v = Linear::new(ones(&[dim, dim]), None).unwrap();
    let o = Linear::new(ones(&[dim, dim]), None).unwrap();
    // num_heads=3 is not divisible by num_kv_heads=2
    assert!(MultiHeadAttention::new(q, k, v, o, 3, 2).is_err());
}

// ===========================================================================
// Dropout
// ===========================================================================

#[test]
fn test_nn_layer_dropout_identity_in_eval() {
    let d = Dropout::new(0.5);
    let x = DynTensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2], &cpu()).unwrap();
    let y = d.forward(&x).unwrap();
    assert_eq!(y.dims(), &[2, 2]);
    let vals = y.to_flat_vec::<f32>().unwrap();
    assert_eq!(vals, vec![1.0, 2.0, 3.0, 4.0]);
}

#[test]
fn test_nn_layer_dropout_shape_preserved() {
    let d = Dropout::new(0.9);
    let x = randn_like(&[3, 4, 5]);
    let y = d.forward(&x).unwrap();
    assert_eq!(y.dims(), &[3, 4, 5]);
}

#[test]
fn test_nn_layer_dropout_zero_and_one_prob() {
    let _ = Dropout::new(0.0);
    let _ = Dropout::new(1.0);
}

// ===========================================================================
// LSTM
// ===========================================================================

#[test]
fn test_nn_layer_lstm_single_step_shape() {
    use nn_core::layers::Lstm;
    let input_size = 4;
    let hidden_size = 3;
    let four_h = 4 * hidden_size;
    let w_ih = zeros(&[four_h, input_size]);
    let w_hh = zeros(&[four_h, hidden_size]);
    let lstm = Lstm::new(w_ih, w_hh, None, None, hidden_size).unwrap();
    let x = randn_like(&[1, input_size]);
    let (output, state) = lstm.forward(&x, None).unwrap();
    assert_eq!(output.dims(), &[1, hidden_size]);
    assert_eq!(state.h.dims(), &[1, hidden_size]);
    assert_eq!(state.c.dims(), &[1, hidden_size]);
}

#[test]
fn test_nn_layer_lstm_with_bias() {
    use nn_core::layers::Lstm;
    let input_size = 2;
    let hidden_size = 2;
    let four_h = 4 * hidden_size;
    let w_ih = zeros(&[four_h, input_size]);
    let w_hh = zeros(&[four_h, hidden_size]);
    let b_ih = zeros(&[four_h]);
    let b_hh = zeros(&[four_h]);
    let lstm = Lstm::new(w_ih, w_hh, Some(b_ih), Some(b_hh), hidden_size).unwrap();
    let x = ones(&[1, input_size]);
    let (output, state) = lstm.forward(&x, None).unwrap();
    assert_eq!(output.dims(), &[1, hidden_size]);
    assert_eq!(state.h.dims(), &[1, hidden_size]);
}

#[test]
fn test_nn_layer_lstm_rejects_zero_hidden() {
    use nn_core::layers::Lstm;
    let w_ih = zeros(&[4, 2]);
    let w_hh = zeros(&[4, 1]);
    assert!(Lstm::new(w_ih, w_hh, None, None, 0).is_err());
}

// ===========================================================================
// InstanceNorm
// ===========================================================================

#[test]
fn test_nn_layer_instance_norm_shape_preserved() {
    let inorm = InstanceNorm::new(1e-5).unwrap();
    let x = randn_like(&[2, 3, 16]);
    let y = inorm.forward(&x).unwrap();
    assert_eq!(y.dims(), &[2, 3, 16]);
}

#[test]
fn test_nn_layer_instance_norm_rejects_2d() {
    let inorm = InstanceNorm::new(1e-5).unwrap();
    let x = ones(&[2, 3]); // rank 2, needs >= 3
    assert!(inorm.forward(&x).is_err());
}

// ===========================================================================
// Sequential
// ===========================================================================

#[test]
fn test_nn_layer_sequential_chain() {
    let mut seq = Sequential::new();
    seq.add(Activation::Relu);
    seq.add(Dropout::new(0.0));
    assert_eq!(seq.len(), 2);
    let x = DynTensor::from_vec(vec![-1.0, 2.0, -3.0, 4.0], &[4], &cpu()).unwrap();
    let y = seq.forward(&x).unwrap();
    let vals = y.to_flat_vec::<f32>().unwrap();
    assert_eq!(vals, vec![0.0, 2.0, 0.0, 4.0]);
}

#[test]
fn test_nn_layer_sequential_empty_is_identity() {
    let seq = Sequential::new();
    assert!(seq.is_empty());
    let x = DynTensor::from_vec(vec![1.0, 2.0, 3.0], &[3], &cpu()).unwrap();
    let y = seq.forward(&x).unwrap();
    assert_eq!(y.to_flat_vec::<f32>().unwrap(), vec![1.0, 2.0, 3.0]);
}

#[test]
fn test_nn_layer_sequential_with_closures() {
    let mut seq = Sequential::new();
    seq.add_fn(DynTensor::relu);
    seq.add_fn(DynTensor::neg);
    let x = DynTensor::from_vec(vec![-5.0, 10.0], &[2], &cpu()).unwrap();
    let y = seq.forward(&x).unwrap();
    let vals = y.to_flat_vec::<f32>().unwrap();
    // relu(-5, 10) = (0, 10), neg = (0, -10)
    assert_eq!(vals, vec![0.0, -10.0]);
}

// ===========================================================================
// Activation
// ===========================================================================

#[test]
fn test_nn_layer_activation_relu() {
    let x = DynTensor::from_vec(vec![-2.0, -1.0, 0.0, 1.0, 2.0], &[5], &cpu()).unwrap();
    let y = Activation::Relu.forward(&x).unwrap();
    assert_eq!(
        y.to_flat_vec::<f32>().unwrap(),
        vec![0.0, 0.0, 0.0, 1.0, 2.0]
    );
}

#[test]
fn test_nn_layer_activation_sigmoid_at_zero() {
    let x = DynTensor::from_vec(vec![0.0], &[1], &cpu()).unwrap();
    let y = Activation::Sigmoid.forward(&x).unwrap();
    let val = y.to_flat_vec::<f32>().unwrap()[0];
    assert!(
        (val - 0.5).abs() < 1e-6,
        "sigmoid(0) should be 0.5, got {val}"
    );
}

#[test]
fn test_nn_layer_activation_gelu_shape() {
    let x = randn_like(&[2, 3, 4]);
    let y = Activation::Gelu.forward(&x).unwrap();
    assert_eq!(y.dims(), &[2, 3, 4]);
}

#[test]
fn test_nn_layer_activation_silu_shape() {
    let x = randn_like(&[4, 8]);
    let y = Activation::Silu.forward(&x).unwrap();
    assert_eq!(y.dims(), &[4, 8]);
}

#[test]
fn test_nn_layer_activation_tanh_bounded() {
    let x = DynTensor::from_vec(vec![-100.0, 0.0, 100.0], &[3], &cpu()).unwrap();
    let y = Activation::Tanh.forward(&x).unwrap();
    let vals = y.to_flat_vec::<f32>().unwrap();
    for v in &vals {
        assert!(
            *v >= -1.0 && *v <= 1.0,
            "tanh output should be in [-1, 1], got {v}"
        );
    }
}

#[test]
fn test_nn_layer_activation_leaky_relu() {
    let x = DynTensor::from_vec(vec![-10.0, 0.0, 10.0], &[3], &cpu()).unwrap();
    let y = Activation::LeakyRelu(0.01).forward(&x).unwrap();
    let vals = y.to_flat_vec::<f32>().unwrap();
    assert!((vals[0] - (-0.1)).abs() < 1e-5, "got {}", vals[0]);
    assert!((vals[1]).abs() < 1e-5);
    assert!((vals[2] - 10.0).abs() < 1e-5);
}

// ===========================================================================
// SwiGlu
// ===========================================================================

#[test]
fn test_nn_layer_swiglu_output_shape() {
    let dim = 8;
    let ff_dim = 16;
    let w_gate = Linear::new(ones(&[ff_dim, dim]), None).unwrap();
    let w_up = Linear::new(ones(&[ff_dim, dim]), None).unwrap();
    let w_down = Linear::new(ones(&[dim, ff_dim]), None).unwrap();
    let swiglu = SwiGlu::new(w_gate, w_up, w_down).unwrap();
    let x = randn_like(&[2, 4, dim]);
    let y = swiglu.forward(&x).unwrap();
    assert_eq!(y.dims(), &[2, 4, dim]);
}

#[test]
fn test_nn_layer_swiglu_rejects_mismatched_gate_up() {
    let w_gate = Linear::new(ones(&[16, 8]), None).unwrap();
    let w_up = Linear::new(ones(&[32, 8]), None).unwrap(); // mismatch: 32 != 16
    let w_down = Linear::new(ones(&[8, 16]), None).unwrap();
    assert!(SwiGlu::new(w_gate, w_up, w_down).is_err());
}

// ===========================================================================
// Cross-layer composition (integration)
// ===========================================================================

#[test]
fn test_nn_layer_linear_then_layer_norm() {
    let dim = 8;
    let w = ones(&[dim, dim]);
    let linear = Linear::new(w, None).unwrap();
    let ln_w = ones(&[dim]);
    let ln_b = zeros(&[dim]);
    let ln = LayerNorm::new(ln_w, ln_b, 1e-5).unwrap();

    let x = randn_like(&[2, dim]);
    let h = linear.forward(&x).unwrap();
    let y = ln.forward(&h).unwrap();
    assert_eq!(y.dims(), &[2, dim]);
}

#[test]
fn test_nn_layer_embedding_then_linear() {
    let vocab = 5;
    let dim = 4;
    let emb_weight = randn_like(&[vocab, dim]);
    let emb = Embedding::new(emb_weight).unwrap();
    let linear_w = ones(&[dim, dim]);
    let linear = Linear::new(linear_w, None).unwrap();

    let ids = DynTensor::from_vec_u32(vec![0, 2, 4], &[1, 3], &cpu()).unwrap();
    let embedded = emb.forward(&ids).unwrap();
    assert_eq!(embedded.dims(), &[1, 3, dim]);
    let y = linear.forward(&embedded).unwrap();
    assert_eq!(y.dims(), &[1, 3, dim]);
}

#[test]
fn test_nn_layer_apply_sugar() {
    let w = DynTensor::from_vec(vec![1.0, 0.0, 0.0, 1.0], &[2, 2], &cpu()).unwrap();
    let linear = Linear::new(w, None).unwrap();
    let x = DynTensor::from_vec(vec![3.0, 7.0], &[1, 2], &cpu()).unwrap();
    let y = x.apply(&linear).unwrap();
    let vals = y.to_flat_vec::<f32>().unwrap();
    assert!((vals[0] - 3.0).abs() < 1e-6);
    assert!((vals[1] - 7.0).abs() < 1e-6);
}

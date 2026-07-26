// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Comprehensive tests for DynTensor neural network layers.
//!
//! Covers: Linear, Embedding, RmsNorm, LayerNorm, Conv1d,
//! Activation functions, and Dropout — focusing on shape contracts,
//! numerical correctness, and edge-case validation.

#![allow(deprecated)]

use crate::dyn_tensor::DynTensor;
use crate::layers::*;
use crate::var_builder::VarBuilder;
use crate::{DType, Device, Result};

fn cpu() -> Device {
    Device::Cpu
}

// ===========================================================================
// Linear — forward shape, bias/no-bias, weight dims, batched, VarBuilder
// ===========================================================================

#[test]
fn test_linear_forward_output_shape_matches_out_features() {
    // weight [out=3, in=4], input [B=2, 4] -> output [2, 3]
    let w = DynTensor::zeros(&[3, 4], DType::F32, &cpu()).unwrap();
    let lin = Linear::new(w, None).unwrap();
    let x = DynTensor::zeros(&[2, 4], DType::F32, &cpu()).unwrap();
    let y = lin.forward(&x).unwrap();
    assert_eq!(y.dims(), &[2, 3]);
}

#[test]
fn test_linear_no_bias_output_is_matmul() {
    // Identity-like weight: [[1,0],[0,1],[1,1]]
    let w = DynTensor::from_vec(vec![1.0, 0.0, 0.0, 1.0, 1.0, 1.0], &[3, 2], &cpu()).unwrap();
    let lin = Linear::new(w, None).unwrap();
    let x = DynTensor::from_vec(vec![3.0, 7.0], &[1, 2], &cpu()).unwrap();
    let y = lin.forward(&x).unwrap();
    let vals = y.to_flat_vec::<f32>().unwrap();
    // y = x @ W^T = [3,7] @ [[1,0,1],[0,1,1]] = [3, 7, 10]
    assert!((vals[0] - 3.0).abs() < 1e-5);
    assert!((vals[1] - 7.0).abs() < 1e-5);
    assert!((vals[2] - 10.0).abs() < 1e-5);
}

#[test]
fn test_linear_with_bias_adds_correctly() {
    let w = DynTensor::from_vec(vec![1.0, 0.0, 0.0, 1.0], &[2, 2], &cpu()).unwrap();
    let b = DynTensor::from_vec(vec![100.0, -50.0], &[2], &cpu()).unwrap();
    let lin = Linear::new(w, Some(b)).unwrap();
    let x = DynTensor::from_vec(vec![5.0, 3.0], &[1, 2], &cpu()).unwrap();
    let y = lin.forward(&x).unwrap();
    let vals = y.to_flat_vec::<f32>().unwrap();
    // y = [5,3] + [100,-50] = [105, -47]
    assert!((vals[0] - 105.0).abs() < 1e-5);
    assert!((vals[1] - (-47.0)).abs() < 1e-5);
}

#[test]
fn test_linear_batched_3d_input() {
    // [B=2, S=3, D=4] -> [B=2, S=3, out=2]
    let w = DynTensor::zeros(&[2, 4], DType::F32, &cpu()).unwrap();
    let lin = Linear::new(w, None).unwrap();
    let x = DynTensor::zeros(&[2, 3, 4], DType::F32, &cpu()).unwrap();
    let y = lin.forward(&x).unwrap();
    assert_eq!(y.dims(), &[2, 3, 2]);
}

#[test]
fn test_linear_weight_dimensions_reported_correctly() {
    let w = DynTensor::zeros(&[16, 32], DType::F32, &cpu()).unwrap();
    let b = DynTensor::zeros(&[16], DType::F32, &cpu()).unwrap();
    let lin = Linear::new(w, Some(b)).unwrap();
    assert_eq!(lin.in_features(), 32);
    assert_eq!(lin.out_features(), 16);
    assert_eq!(lin.weight().dims(), &[16, 32]);
    assert!(lin.bias().is_some());
    assert_eq!(lin.bias().unwrap().dims(), &[16]);
}

#[test]
fn test_linear_via_var_builder_zeros() {
    let vb = VarBuilder::zeros(DType::F32, &cpu());
    let lin = linear(8, 4, &vb).unwrap();
    assert_eq!(lin.in_features(), 8);
    assert_eq!(lin.out_features(), 4);
    assert!(lin.bias().is_some());
}

#[test]
fn test_linear_no_bias_via_var_builder() {
    let vb = VarBuilder::zeros(DType::F32, &cpu());
    let lin = linear_no_bias(8, 4, &vb).unwrap();
    assert_eq!(lin.in_features(), 8);
    assert_eq!(lin.out_features(), 4);
    assert!(lin.bias().is_none());
}

#[test]
fn test_linear_single_element_input() {
    let w = DynTensor::from_vec(vec![2.0], &[1, 1], &cpu()).unwrap();
    let b = DynTensor::from_vec(vec![3.0], &[1], &cpu()).unwrap();
    let lin = Linear::new(w, Some(b)).unwrap();
    let x = DynTensor::from_vec(vec![5.0], &[1, 1], &cpu()).unwrap();
    let y = lin.forward(&x).unwrap();
    let val = y.to_flat_vec::<f32>().unwrap()[0];
    assert!((val - 13.0).abs() < 1e-5); // 5*2 + 3
}

// ===========================================================================
// Embedding — lookup output shape, vocab bounds, batch lookup, multi-dim
// ===========================================================================

#[test]
fn test_embedding_output_shape_matches_embed_dim() {
    // vocab=5, embed_dim=8
    let w = DynTensor::zeros(&[5, 8], DType::F32, &cpu()).unwrap();
    let emb = Embedding::new(w).unwrap();
    let result = emb.forward_ids(&[0, 1, 2]).unwrap();
    assert_eq!(result.dims(), &[3, 8]);
}

#[test]
fn test_embedding_single_id_lookup() {
    let w = DynTensor::from_vec(vec![10.0, 20.0, 30.0, 40.0, 50.0, 60.0], &[3, 2], &cpu()).unwrap();
    let emb = Embedding::new(w).unwrap();
    let result = emb.forward_ids(&[1]).unwrap();
    assert_eq!(result.dims(), &[1, 2]);
    let vals = result.to_flat_vec::<f32>().unwrap();
    assert!((vals[0] - 30.0).abs() < 1e-6);
    assert!((vals[1] - 40.0).abs() < 1e-6);
}

#[test]
fn test_embedding_vocab_boundary_last_index() {
    // vocab_size=4, accessing index 3 (last valid)
    let w = DynTensor::from_vec(
        vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 99.0, 88.0],
        &[4, 2],
        &cpu(),
    )
    .unwrap();
    let emb = Embedding::new(w).unwrap();
    let result = emb.forward_ids(&[3]).unwrap();
    let vals = result.to_flat_vec::<f32>().unwrap();
    assert!((vals[0] - 99.0).abs() < 1e-6);
    assert!((vals[1] - 88.0).abs() < 1e-6);
}

#[test]
fn test_embedding_vocab_out_of_bounds() {
    let w = DynTensor::zeros(&[4, 2], DType::F32, &cpu()).unwrap();
    let emb = Embedding::new(w).unwrap();
    assert!(emb.forward_ids(&[4]).is_err()); // 4 >= vocab_size=4
    assert!(emb.forward_ids(&[100]).is_err());
}

#[test]
fn test_embedding_batch_2d_u32_input() {
    // Input shape [B=2, S=3] -> output [2, 3, embed_dim=4]
    let w = DynTensor::zeros(&[10, 4], DType::F32, &cpu()).unwrap();
    let emb = Embedding::new(w).unwrap();
    let ids = DynTensor::from_vec_u32(vec![0, 1, 2, 3, 4, 5], &[2, 3], &cpu()).unwrap();
    let result = emb.forward(&ids).unwrap();
    assert_eq!(result.dims(), &[2, 3, 4]);
}

#[test]
fn test_embedding_i64_input() {
    let w = DynTensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2], &cpu()).unwrap();
    let emb = Embedding::new(w).unwrap();
    let ids = DynTensor::from_vec_i64(vec![0, 1], &[2], &cpu()).unwrap();
    let result = emb.forward(&ids).unwrap();
    assert_eq!(result.dims(), &[2, 2]);
    let vals = result.to_flat_vec::<f32>().unwrap();
    assert!((vals[0] - 1.0).abs() < 1e-6);
    assert!((vals[2] - 3.0).abs() < 1e-6);
}

#[test]
fn test_embedding_via_var_builder() {
    let vb = VarBuilder::zeros(DType::F32, &cpu());
    let emb = embedding(100, 64, &vb).unwrap();
    assert_eq!(emb.weight().dims(), &[100, 64]);
}

// ===========================================================================
// RmsNorm — output shape preservation, eps, normalized output properties
// ===========================================================================

#[test]
fn test_rms_norm_preserves_shape_2d() {
    let weight = DynTensor::ones(&[8], DType::F32, &cpu()).unwrap();
    let norm = RmsNorm::new(weight, 1e-5).unwrap();
    let x = DynTensor::zeros(&[4, 8], DType::F32, &cpu()).unwrap();
    let y = norm.forward(&x).unwrap();
    assert_eq!(y.dims(), &[4, 8]);
}

#[test]
fn test_rms_norm_preserves_shape_3d() {
    let weight = DynTensor::ones(&[16], DType::F32, &cpu()).unwrap();
    let norm = RmsNorm::new(weight, 1e-5).unwrap();
    let x = DynTensor::zeros(&[2, 4, 16], DType::F32, &cpu()).unwrap();
    let y = norm.forward(&x).unwrap();
    assert_eq!(y.dims(), &[2, 4, 16]);
}

#[test]
fn test_rms_norm_unit_input_normalizes_to_one() {
    // Input [1,1,1,1]: RMS = 1, normed = [1,1,1,1] * weight
    let weight = DynTensor::ones(&[4], DType::F32, &cpu()).unwrap();
    let norm = RmsNorm::new(weight, 1e-8).unwrap();
    let x = DynTensor::from_vec(vec![1.0, 1.0, 1.0, 1.0], &[1, 4], &cpu()).unwrap();
    let y = norm.forward(&x).unwrap();
    let vals = y.to_flat_vec::<f32>().unwrap();
    for v in &vals {
        assert!((*v - 1.0).abs() < 0.01, "expected ~1.0, got {v}");
    }
}

#[test]
fn test_rms_norm_scaling_by_weight() {
    // Input [1,1], weight [2,3]: RMS([1,1])=1, output = [2,3]
    let weight = DynTensor::from_vec(vec![2.0, 3.0], &[2], &cpu()).unwrap();
    let norm = RmsNorm::new(weight, 1e-8).unwrap();
    let x = DynTensor::from_vec(vec![1.0, 1.0], &[1, 2], &cpu()).unwrap();
    let y = norm.forward(&x).unwrap();
    let vals = y.to_flat_vec::<f32>().unwrap();
    assert!((vals[0] - 2.0).abs() < 0.01);
    assert!((vals[1] - 3.0).abs() < 0.01);
}

#[test]
fn test_rms_norm_eps_zero_works() {
    let weight = DynTensor::ones(&[3], DType::F32, &cpu()).unwrap();
    // eps=0 is valid (but numerically fragile)
    let norm = RmsNorm::new(weight, 0.0).unwrap();
    let x = DynTensor::from_vec(vec![1.0, 2.0, 3.0], &[1, 3], &cpu()).unwrap();
    let y = norm.forward(&x).unwrap();
    let vals = y.to_flat_vec::<f32>().unwrap();
    for v in &vals {
        assert!(v.is_finite());
    }
}

#[test]
fn test_rms_norm_via_var_builder() {
    let vb = VarBuilder::zeros(DType::F32, &cpu());
    let norm = rms_norm(32, 1e-5, &vb).unwrap();
    assert_eq!(norm.weight().dims(), &[32]);
}

// ===========================================================================
// LayerNorm — shape preservation, affine parameters, eps
// ===========================================================================

#[test]
fn test_layer_norm_preserves_shape_2d() {
    let w = DynTensor::ones(&[8], DType::F32, &cpu()).unwrap();
    let b = DynTensor::zeros(&[8], DType::F32, &cpu()).unwrap();
    let ln = LayerNorm::new(w, b, 1e-5).unwrap();
    let x = DynTensor::zeros(&[4, 8], DType::F32, &cpu()).unwrap();
    let y = ln.forward(&x).unwrap();
    assert_eq!(y.dims(), &[4, 8]);
}

#[test]
fn test_layer_norm_output_has_zero_mean_unit_var() {
    // With weight=1 and bias=0, LayerNorm output should have approximately
    // zero mean and unit variance along the last dimension.
    let w = DynTensor::ones(&[4], DType::F32, &cpu()).unwrap();
    let b = DynTensor::zeros(&[4], DType::F32, &cpu()).unwrap();
    let ln = LayerNorm::new(w, b, 1e-5).unwrap();
    let x = DynTensor::from_vec(vec![1.0, 2.0, 4.0, 8.0], &[1, 4], &cpu()).unwrap();
    let y = ln.forward(&x).unwrap();
    let vals = y.to_flat_vec::<f32>().unwrap();
    let mean: f32 = vals.iter().sum::<f32>() / vals.len() as f32;
    assert!(mean.abs() < 0.01, "mean should be ~0, got {mean}");
    let var: f32 = vals.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / vals.len() as f32;
    assert!((var - 1.0).abs() < 0.1, "variance should be ~1, got {var}");
}

#[test]
fn test_layer_norm_affine_transform() {
    // weight=2, bias=10 on normalized [1,-1] -> [2*1+10, 2*(-1)+10] = [12, 8]
    let w = DynTensor::from_vec(vec![2.0, 2.0], &[2], &cpu()).unwrap();
    let b = DynTensor::from_vec(vec![10.0, 10.0], &[2], &cpu()).unwrap();
    let ln = LayerNorm::new(w, b, 0.0).unwrap();
    let x = DynTensor::from_vec(vec![1.0, -1.0], &[1, 2], &cpu()).unwrap();
    let y = ln.forward(&x).unwrap();
    let vals = y.to_flat_vec::<f32>().unwrap();
    // Normalized [1,-1] with mean=0 var=1 stays [1,-1], then affine: [12, 8]
    assert!((vals[0] - 12.0).abs() < 1e-4);
    assert!((vals[1] - 8.0).abs() < 1e-4);
}

#[test]
fn test_layer_norm_batched_4d() {
    // [B=2, H=2, S=3, D=4] -> same shape
    let w = DynTensor::ones(&[4], DType::F32, &cpu()).unwrap();
    let b = DynTensor::zeros(&[4], DType::F32, &cpu()).unwrap();
    let ln = LayerNorm::new(w, b, 1e-5).unwrap();
    let x =
        DynTensor::from_vec((0..48).map(|i| i as f32).collect(), &[2, 2, 3, 4], &cpu()).unwrap();
    let y = ln.forward(&x).unwrap();
    assert_eq!(y.dims(), &[2, 2, 3, 4]);
}

#[test]
fn test_layer_norm_via_var_builder() {
    let vb = VarBuilder::zeros(DType::F32, &cpu());
    let cfg = LayerNormConfig { eps: 1e-6 };
    let ln = layer_norm(64, cfg, &vb).unwrap();
    assert_eq!(ln.weight().dims(), &[64]);
    assert_eq!(ln.bias().dims(), &[64]);
}

#[test]
fn test_layer_norm_rank0_rejected() {
    let w = DynTensor::ones(&[1], DType::F32, &cpu()).unwrap();
    let b = DynTensor::zeros(&[1], DType::F32, &cpu()).unwrap();
    let ln = LayerNorm::new(w, b, 1e-5).unwrap();
    let scalar = DynTensor::from_vec(vec![1.0], &[], &cpu()).unwrap();
    assert!(ln.forward(&scalar).is_err());
}

// ===========================================================================
// Conv1d — output length, stride, padding, dilation, groups
// ===========================================================================

#[test]
fn test_conv1d_output_length_no_padding_stride1() {
    // input_len=5, kernel=3, stride=1, padding=0, dilation=1
    // output_len = (5 + 2*0 - 1*(3-1) - 1)/1 + 1 = 3
    let w = DynTensor::zeros(&[1, 1, 3], DType::F32, &cpu()).unwrap();
    let conv = Conv1d::new(w, None, Conv1dConfig::default()).unwrap();
    let x = DynTensor::zeros(&[1, 1, 5], DType::F32, &cpu()).unwrap();
    let y = conv.forward(&x).unwrap();
    assert_eq!(y.dims(), &[1, 1, 3]);
}

#[test]
fn test_conv1d_output_length_with_padding() {
    // input_len=5, kernel=3, padding=1 -> output_len = (5+2-2-1)/1+1 = 5
    let w = DynTensor::zeros(&[1, 1, 3], DType::F32, &cpu()).unwrap();
    let cfg = Conv1dConfig::new(1, 1, 1); // padding=1, stride=1, dilation=1
    let conv = Conv1d::new(w, None, cfg).unwrap();
    let x = DynTensor::zeros(&[1, 1, 5], DType::F32, &cpu()).unwrap();
    let y = conv.forward(&x).unwrap();
    assert_eq!(y.dims(), &[1, 1, 5]);
}

#[test]
fn test_conv1d_output_length_with_stride() {
    // input_len=8, kernel=3, stride=2, padding=0 -> output_len = (8-3)/2+1 = 3
    let w = DynTensor::zeros(&[1, 1, 3], DType::F32, &cpu()).unwrap();
    let cfg = Conv1dConfig::new(0, 2, 1);
    let conv = Conv1d::new(w, None, cfg).unwrap();
    let x = DynTensor::zeros(&[1, 1, 8], DType::F32, &cpu()).unwrap();
    let y = conv.forward(&x).unwrap();
    assert_eq!(y.dims(), &[1, 1, 3]);
}

#[test]
fn test_conv1d_output_length_with_dilation() {
    // input_len=10, kernel=3, dilation=2 -> effective_kernel=5
    // output_len = (10-5)/1+1 = 6
    let w = DynTensor::zeros(&[1, 1, 3], DType::F32, &cpu()).unwrap();
    let cfg = Conv1dConfig::new(0, 1, 2); // padding=0, stride=1, dilation=2
    let conv = Conv1d::new(w, None, cfg).unwrap();
    let x = DynTensor::zeros(&[1, 1, 10], DType::F32, &cpu()).unwrap();
    let y = conv.forward(&x).unwrap();
    assert_eq!(y.dims(), &[1, 1, 6]);
}

#[test]
fn test_conv1d_groups_depthwise() {
    // groups=2 on 2 input channels: weight [2, 1, 3]
    let w = DynTensor::from_vec(vec![1.0; 6], &[2, 1, 3], &cpu()).unwrap();
    let cfg = Conv1dConfig::default().with_groups(2);
    let conv = Conv1d::new(w, None, cfg).unwrap();
    let x = DynTensor::from_vec(vec![1.0; 12], &[1, 2, 6], &cpu()).unwrap();
    let y = conv.forward(&x).unwrap();
    // out_channels=2, output_len = (6-3)/1+1 = 4
    assert_eq!(y.dims(), &[1, 2, 4]);
}

#[test]
fn test_conv1d_with_bias_output() {
    // kernel=1 with weight=1, bias=5: y = x*1 + 5
    let w = DynTensor::from_vec(vec![1.0], &[1, 1, 1], &cpu()).unwrap();
    let b = DynTensor::from_vec(vec![5.0], &[1], &cpu()).unwrap();
    let conv = Conv1d::new(w, Some(b), Conv1dConfig::default()).unwrap();
    let x = DynTensor::from_vec(vec![3.0, 7.0], &[1, 1, 2], &cpu()).unwrap();
    let y = conv.forward(&x).unwrap();
    let vals = y.to_flat_vec::<f32>().unwrap();
    assert!((vals[0] - 8.0).abs() < 1e-5); // 3+5
    assert!((vals[1] - 12.0).abs() < 1e-5); // 7+5
}

#[test]
fn test_conv1d_multi_channel_output() {
    // in=1, out=3, kernel=1 -> 3 output channels
    let w = DynTensor::from_vec(vec![1.0, 2.0, 3.0], &[3, 1, 1], &cpu()).unwrap();
    let conv = Conv1d::new(w, None, Conv1dConfig::default()).unwrap();
    let x = DynTensor::from_vec(vec![10.0], &[1, 1, 1], &cpu()).unwrap();
    let y = conv.forward(&x).unwrap();
    assert_eq!(y.dims(), &[1, 3, 1]);
    let vals = y.to_flat_vec::<f32>().unwrap();
    assert!((vals[0] - 10.0).abs() < 1e-5);
    assert!((vals[1] - 20.0).abs() < 1e-5);
    assert!((vals[2] - 30.0).abs() < 1e-5);
}

#[test]
fn test_conv1d_batched_input() {
    let w = DynTensor::zeros(&[2, 1, 3], DType::F32, &cpu()).unwrap();
    let conv = Conv1d::new(w, None, Conv1dConfig::default()).unwrap();
    // batch of 4
    let x = DynTensor::zeros(&[4, 1, 8], DType::F32, &cpu()).unwrap();
    let y = conv.forward(&x).unwrap();
    // out_len = (8-3)/1+1 = 6
    assert_eq!(y.dims(), &[4, 2, 6]);
}

#[test]
fn test_conv1d_via_var_builder() {
    let vb = VarBuilder::zeros(DType::F32, &cpu());
    let cfg = Conv1dConfig::new(1, 1, 1);
    let c = conv1d(4, 8, 3, cfg, &vb).unwrap();
    assert_eq!(c.weight().dims(), &[8, 4, 3]);
    assert!(c.bias().is_some());
}

#[test]
fn test_conv1d_no_bias_via_var_builder() {
    let vb = VarBuilder::zeros(DType::F32, &cpu());
    let cfg = Conv1dConfig::default();
    let c = conv1d_no_bias(4, 8, 3, cfg, &vb).unwrap();
    assert_eq!(c.weight().dims(), &[8, 4, 3]);
    assert!(c.bias().is_none());
}

#[test]
fn test_conv1d_config_builder_chain() {
    let cfg = Conv1dConfig::default()
        .with_padding(2)
        .with_stride(3)
        .with_dilation(4)
        .with_groups(5);
    assert_eq!(cfg.padding, 2);
    assert_eq!(cfg.stride, 3);
    assert_eq!(cfg.dilation, 4);
    assert_eq!(cfg.groups, 5);
}

#[test]
fn test_conv1d_weight_rank_error() {
    let w = DynTensor::zeros(&[4, 4], DType::F32, &cpu()).unwrap();
    assert!(Conv1d::new(w, None, Conv1dConfig::default()).is_err());
}

#[test]
fn test_conv1d_groups_zero_error() {
    let w = DynTensor::zeros(&[1, 1, 3], DType::F32, &cpu()).unwrap();
    let cfg = Conv1dConfig::default().with_groups(0);
    assert!(Conv1d::new(w, None, cfg).is_err());
}

// ===========================================================================
// Activation functions — shape preservation, known values
// ===========================================================================

#[test]
fn test_activation_relu_preserves_shape() {
    let x = DynTensor::zeros(&[2, 3, 4], DType::F32, &cpu()).unwrap();
    let y = Activation::Relu.forward(&x).unwrap();
    assert_eq!(y.dims(), &[2, 3, 4]);
}

#[test]
fn test_activation_gelu_preserves_shape() {
    let x = DynTensor::zeros(&[5, 10], DType::F32, &cpu()).unwrap();
    let y = Activation::Gelu.forward(&x).unwrap();
    assert_eq!(y.dims(), &[5, 10]);
}

#[test]
fn test_activation_silu_preserves_shape() {
    let x = DynTensor::zeros(&[3, 4], DType::F32, &cpu()).unwrap();
    let y = Activation::Silu.forward(&x).unwrap();
    assert_eq!(y.dims(), &[3, 4]);
}

#[test]
fn test_activation_sigmoid_output_in_0_1() {
    let x = DynTensor::from_vec(vec![-10.0, -1.0, 0.0, 1.0, 10.0], &[5], &cpu()).unwrap();
    let y = Activation::Sigmoid.forward(&x).unwrap();
    let vals = y.to_flat_vec::<f32>().unwrap();
    for v in &vals {
        assert!(*v >= 0.0 && *v <= 1.0, "sigmoid output {v} not in [0,1]");
    }
    // sigmoid(-10) close to 0
    assert!(vals[0] < 0.001);
    // sigmoid(10) close to 1
    assert!(vals[4] > 0.999);
}

#[test]
fn test_activation_tanh_output_in_neg1_1() {
    let x = DynTensor::from_vec(vec![-10.0, -1.0, 0.0, 1.0, 10.0], &[5], &cpu()).unwrap();
    let y = Activation::Tanh.forward(&x).unwrap();
    let vals = y.to_flat_vec::<f32>().unwrap();
    for v in &vals {
        assert!(*v >= -1.0 && *v <= 1.0, "tanh output {v} not in [-1,1]");
    }
}

#[test]
fn test_activation_relu_zeroes_negatives() {
    let x = DynTensor::from_vec(vec![-3.0, -2.0, -1.0, 0.0, 1.0, 2.0], &[6], &cpu()).unwrap();
    let y = Activation::Relu.forward(&x).unwrap();
    let vals = y.to_flat_vec::<f32>().unwrap();
    assert_eq!(vals[0], 0.0);
    assert_eq!(vals[1], 0.0);
    assert_eq!(vals[2], 0.0);
    assert_eq!(vals[3], 0.0);
    assert_eq!(vals[4], 1.0);
    assert_eq!(vals[5], 2.0);
}

#[test]
fn test_activation_gelu_known_values() {
    let x = DynTensor::from_vec(vec![0.0, 1.0, -1.0], &[3], &cpu()).unwrap();
    let y = Activation::Gelu.forward(&x).unwrap();
    let vals = y.to_flat_vec::<f32>().unwrap();
    // GELU(0) = 0
    assert!(vals[0].abs() < 1e-6);
    // GELU(1) approx 0.8412
    assert!((vals[1] - 0.8412).abs() < 0.01);
    // GELU(-1) approx -0.1588
    assert!((vals[2] - (-0.1588)).abs() < 0.01);
}

#[test]
fn test_activation_silu_known_values() {
    let x = DynTensor::from_vec(vec![0.0, 1.0, -1.0], &[3], &cpu()).unwrap();
    let y = Activation::Silu.forward(&x).unwrap();
    let vals = y.to_flat_vec::<f32>().unwrap();
    // SiLU(0) = 0
    assert!(vals[0].abs() < 1e-6);
    // SiLU(1) = sigmoid(1) approx 0.7311
    assert!((vals[1] - 0.7311).abs() < 0.01);
    // SiLU(-1) = -sigmoid(-1) approx -0.2689
    assert!((vals[2] - (-0.2689)).abs() < 0.01);
}

#[test]
fn test_activation_elu_negative_region() {
    let x = DynTensor::from_vec(vec![-2.0, -1.0, 0.0, 1.0], &[4], &cpu()).unwrap();
    let y = Activation::Elu(1.0).forward(&x).unwrap();
    let vals = y.to_flat_vec::<f32>().unwrap();
    // ELU(x) = alpha*(exp(x)-1) for x < 0
    assert!(
        vals[0] < 0.0 && vals[0] > -1.0,
        "ELU(-2) should be in (-1, 0)"
    );
    assert!((vals[2] - 0.0).abs() < 1e-6);
    assert!((vals[3] - 1.0).abs() < 1e-6);
}

#[test]
fn test_activation_leaky_relu_negative_slope() {
    let x = DynTensor::from_vec(vec![-10.0, 0.0, 10.0], &[3], &cpu()).unwrap();
    let y = Activation::LeakyRelu(0.01).forward(&x).unwrap();
    let vals = y.to_flat_vec::<f32>().unwrap();
    assert!((vals[0] - (-0.1)).abs() < 1e-5); // -10 * 0.01
    assert!((vals[1] - 0.0).abs() < 1e-5);
    assert!((vals[2] - 10.0).abs() < 1e-5);
}

// ===========================================================================
// Dropout — shape preservation, identity at inference
// ===========================================================================

#[test]
fn test_dropout_preserves_shape_4d() {
    let d = Dropout::new(0.5);
    let x = DynTensor::zeros(&[2, 3, 4, 5], DType::F32, &cpu()).unwrap();
    let y = d.forward(&x).unwrap();
    assert_eq!(y.dims(), &[2, 3, 4, 5]);
}

#[test]
fn test_dropout_is_identity_at_inference() {
    let d = Dropout::new(0.99);
    let x = DynTensor::from_vec(vec![1.5, -2.3, 0.0, 42.0, -100.0, 3.14], &[2, 3], &cpu()).unwrap();
    let y = d.forward(&x).unwrap();
    let x_vals = x.to_flat_vec::<f32>().unwrap();
    let y_vals = y.to_flat_vec::<f32>().unwrap();
    assert_eq!(x_vals, y_vals, "dropout should be identity at inference");
}

#[test]
fn test_dropout_zero_probability() {
    let d = Dropout::new(0.0);
    let x = DynTensor::from_vec(vec![1.0, 2.0, 3.0], &[3], &cpu()).unwrap();
    let y = d.forward(&x).unwrap();
    assert_eq!(y.to_flat_vec::<f32>().unwrap(), vec![1.0, 2.0, 3.0]);
}

#[test]
fn test_dropout_preserves_dtype() {
    let d = Dropout::new(0.5);
    let x = DynTensor::zeros(&[4], DType::F32, &cpu()).unwrap();
    let y = d.forward(&x).unwrap();
    assert_eq!(y.dtype(), DType::F32);
}

// ===========================================================================
// VarBuilder integration — verifies layer construction from VarBuilder
// ===========================================================================

#[test]
fn test_var_builder_linear_forward_roundtrip() {
    let vb = VarBuilder::zeros(DType::F32, &cpu());
    let lin = linear(4, 2, &vb).unwrap();
    let x = DynTensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[1, 4], &cpu()).unwrap();
    let y = lin.forward(&x).unwrap();
    assert_eq!(y.dims(), &[1, 2]);
    // All-zero weights and bias: output should be zero
    let vals = y.to_flat_vec::<f32>().unwrap();
    for v in &vals {
        assert!(v.abs() < 1e-6, "zero-weight linear should produce zeros");
    }
}

#[test]
fn test_var_builder_conv1d_forward_roundtrip() {
    let vb = VarBuilder::zeros(DType::F32, &cpu());
    let cfg = Conv1dConfig::new(1, 1, 1);
    let c = conv1d(2, 4, 3, cfg, &vb).unwrap();
    let x = DynTensor::from_vec(vec![1.0; 12], &[1, 2, 6], &cpu()).unwrap();
    let y = c.forward(&x).unwrap();
    // output_len = (6 + 2*1 - 1*(3-1) - 1)/1 + 1 = 6
    assert_eq!(y.dims(), &[1, 4, 6]);
}

#[test]
fn test_var_builder_embedding_forward_roundtrip() {
    let vb = VarBuilder::zeros(DType::F32, &cpu());
    let emb = embedding(10, 8, &vb).unwrap();
    let ids = DynTensor::from_vec_u32(vec![0, 5, 9], &[3], &cpu()).unwrap();
    let y = emb.forward(&ids).unwrap();
    assert_eq!(y.dims(), &[3, 8]);
}

#[test]
fn test_var_builder_rms_norm_forward_roundtrip() {
    let vb = VarBuilder::zeros(DType::F32, &cpu());
    let norm = rms_norm(16, 1e-5, &vb).unwrap();
    let x = DynTensor::from_vec(vec![1.0; 16], &[1, 16], &cpu()).unwrap();
    let y = norm.forward(&x).unwrap();
    assert_eq!(y.dims(), &[1, 16]);
}

// ===========================================================================
// Sequential — compose multiple layers
// ===========================================================================

#[test]
fn test_sequential_linear_then_activation() {
    let w = DynTensor::from_vec(vec![1.0, 0.0, 0.0, 1.0], &[2, 2], &cpu()).unwrap();
    let b = DynTensor::from_vec(vec![-1.0, -1.0], &[2], &cpu()).unwrap();
    let lin = Linear::new(w, Some(b)).unwrap();

    let mut seq = Sequential::new();
    seq.add(lin);
    seq.add(Activation::Relu);

    let x = DynTensor::from_vec(vec![0.5, 2.0], &[1, 2], &cpu()).unwrap();
    let y = seq.forward(&x).unwrap();
    let vals = y.to_flat_vec::<f32>().unwrap();
    // linear: [0.5-1, 2-1] = [-0.5, 1.0], relu: [0, 1.0]
    assert!((vals[0] - 0.0).abs() < 1e-5);
    assert!((vals[1] - 1.0).abs() < 1e-5);
}

#[test]
fn test_sequential_empty_is_identity() {
    let seq = Sequential::new();
    let x = DynTensor::from_vec(vec![1.0, 2.0, 3.0], &[3], &cpu()).unwrap();
    let y = seq.forward(&x).unwrap();
    assert_eq!(y.to_flat_vec::<f32>().unwrap(), vec![1.0, 2.0, 3.0]);
}

// ===========================================================================
// Module trait — apply() sugar
// ===========================================================================

#[test]
fn test_apply_chains_modules() {
    let w = DynTensor::from_vec(vec![2.0], &[1, 1], &cpu()).unwrap();
    let lin = Linear::new(w, None).unwrap();
    let x = DynTensor::from_vec(vec![5.0], &[1, 1], &cpu()).unwrap();
    let y = x.apply(&lin).unwrap();
    let val = y.to_flat_vec::<f32>().unwrap()[0];
    assert!((val - 10.0).abs() < 1e-5);
}

#[test]
fn test_closure_as_module() {
    let double = |x: &DynTensor| -> Result<DynTensor> {
        let two = DynTensor::from_vec(vec![2.0], &[1], &cpu()).unwrap();
        x.broadcast_mul(&two)
    };
    let x = DynTensor::from_vec(vec![3.0, 4.0], &[2], &cpu()).unwrap();
    let y = x.apply(&double).unwrap();
    let vals = y.to_flat_vec::<f32>().unwrap();
    assert!((vals[0] - 6.0).abs() < 1e-5);
    assert!((vals[1] - 8.0).abs() < 1e-5);
}

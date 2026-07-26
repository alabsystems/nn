// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Extended nn layer configuration and interaction tests.
//!
//! Covers config defaults, output size calculations, layer construction
//! validation, forward-pass shape contracts, and cross-layer composition.
//!
//! Part of #4186.

use crate::dyn_tensor::DynTensor;
use crate::module::ModuleT;
use crate::layers::{
    Activation, BatchNorm, BatchNormConfig, Conv1d, Conv1dConfig, Conv2d, Conv2dConfig, Dropout,
    Embedding, LayerNorm, LayerNormConfig, Linear, MaxPool1d, MaxPool2d, Module, Pool1dConfig,
    Pool2dConfig, Sequential,
};
use crate::{conv1d_out_len, conv2d_out_len, DType, Device};

// =============================================================================
// 1. Conv1dConfig / Conv2dConfig — output size calculations
// =============================================================================

#[test]
fn test_conv1d_out_len_no_padding_stride1() {
    // input=10, kernel=3, padding=0, stride=1, dilation=1
    // out = (10 + 0 - 3)/1 + 1 = 8
    let out = conv1d_out_len(10, 3, 0, 1, 1).unwrap();
    assert_eq!(out, 8);
}

#[test]
fn test_conv1d_out_len_with_padding() {
    // input=10, kernel=3, padding=1, stride=1, dilation=1
    // out = (10 + 2 - 3)/1 + 1 = 10 (same-padding)
    let out = conv1d_out_len(10, 3, 1, 1, 1).unwrap();
    assert_eq!(out, 10);
}

#[test]
fn test_conv1d_out_len_stride2() {
    // input=10, kernel=3, padding=0, stride=2, dilation=1
    // out = (10 - 3)/2 + 1 = 4
    let out = conv1d_out_len(10, 3, 0, 2, 1).unwrap();
    assert_eq!(out, 4);
}

#[test]
fn test_conv1d_out_len_dilation2() {
    // input=10, kernel=3, padding=0, stride=1, dilation=2
    // effective_k = (3-1)*2 + 1 = 5
    // out = (10 - 5)/1 + 1 = 6
    let out = conv1d_out_len(10, 3, 0, 1, 2).unwrap();
    assert_eq!(out, 6);
}

#[test]
fn test_conv1d_out_len_combined() {
    // input=16, kernel=5, padding=2, stride=2, dilation=1
    // out = (16 + 4 - 5)/2 + 1 = 8
    let out = conv1d_out_len(16, 5, 2, 2, 1).unwrap();
    assert_eq!(out, 8);
}

#[test]
fn test_conv1d_out_len_large_dilation_stride() {
    // input=32, kernel=3, padding=4, stride=4, dilation=3
    // effective_k = (3-1)*3 + 1 = 7
    // padded = 32 + 8 = 40
    // out = (40 - 7)/4 + 1 = 33/4 + 1 = 8 + 1 = 9
    let out = conv1d_out_len(32, 3, 4, 4, 3).unwrap();
    assert_eq!(out, 9);
}

#[test]
fn test_conv2d_out_len_no_padding() {
    // input_h=8, kernel=3, padding=0, stride=1, dilation=1
    // out = (8 - 3)/1 + 1 = 6
    let out = conv2d_out_len(8, 3, 0, 1, 1).unwrap();
    assert_eq!(out, 6);
}

#[test]
fn test_conv2d_out_len_same_padding() {
    // input=7, kernel=3, padding=1, stride=1, dilation=1
    // out = (7 + 2 - 3)/1 + 1 = 7
    let out = conv2d_out_len(7, 3, 1, 1, 1).unwrap();
    assert_eq!(out, 7);
}

#[test]
fn test_conv2d_out_len_stride_and_dilation() {
    // input=16, kernel=3, padding=1, stride=2, dilation=2
    // effective_k = (3-1)*2 + 1 = 5
    // padded = 16 + 2 = 18
    // out = (18 - 5)/2 + 1 = 7
    let out = conv2d_out_len(16, 3, 1, 2, 2).unwrap();
    assert_eq!(out, 7);
}

// -- Conv config defaults and builder pattern --------------------------------

#[test]
fn test_conv1d_config_defaults() {
    let cfg = Conv1dConfig::default();
    assert_eq!(cfg.padding, 0);
    assert_eq!(cfg.stride, 1);
    assert_eq!(cfg.dilation, 1);
    assert_eq!(cfg.groups, 1);
}

#[test]
fn test_conv1d_config_builder() {
    let cfg = Conv1dConfig::new(2, 4, 1).with_groups(8);
    assert_eq!(cfg.padding, 2);
    assert_eq!(cfg.stride, 4);
    assert_eq!(cfg.dilation, 1);
    assert_eq!(cfg.groups, 8);
}

#[test]
fn test_conv2d_config_defaults() {
    let cfg = Conv2dConfig::default();
    assert_eq!(cfg.padding, 0);
    assert_eq!(cfg.stride, 1);
    assert_eq!(cfg.dilation, 1);
    assert_eq!(cfg.groups, 1);
}

#[test]
fn test_conv2d_config_builder() {
    let cfg = Conv2dConfig::new(1, 2, 3).with_groups(4);
    assert_eq!(cfg.padding, 1);
    assert_eq!(cfg.stride, 2);
    assert_eq!(cfg.dilation, 3);
    assert_eq!(cfg.groups, 4);
}

// =============================================================================
// 2. Pool1dConfig / Pool2dConfig — output size calculations
// =============================================================================

#[test]
fn test_pool1d_config_defaults() {
    let cfg = Pool1dConfig::new(4);
    assert_eq!(cfg.kernel_size, 4);
    assert_eq!(cfg.stride, 4, "stride should default to kernel_size");
    assert_eq!(cfg.padding, 0);
}

#[test]
fn test_pool1d_config_custom_stride() {
    let cfg = Pool1dConfig::new(3).with_stride(2).with_padding(1);
    assert_eq!(cfg.kernel_size, 3);
    assert_eq!(cfg.stride, 2);
    assert_eq!(cfg.padding, 1);
}

#[test]
fn test_pool2d_config_defaults() {
    let cfg = Pool2dConfig::new(2);
    assert_eq!(cfg.kernel_size, 2);
    assert_eq!(cfg.stride, 2, "stride should default to kernel_size");
    assert_eq!(cfg.padding, 0);
}

#[test]
fn test_pool2d_config_custom_stride() {
    let cfg = Pool2dConfig::new(3).with_stride(1).with_padding(1);
    assert_eq!(cfg.kernel_size, 3);
    assert_eq!(cfg.stride, 1);
    assert_eq!(cfg.padding, 1);
}

#[test]
fn test_pool1d_output_shape() {
    // input [B=1, C=2, L=10], kernel=3, stride=3, padding=0
    // out_len = (10 - 3)/3 + 1 = 3
    let x = DynTensor::full(&[1, 2, 10], 1.0, DType::F32, &Device::Cpu).unwrap();
    let pool = MaxPool1d::new(Pool1dConfig::new(3)).unwrap();
    let y = pool.forward(&x).unwrap();
    assert_eq!(y.dims(), &[1, 2, 3]);
}

#[test]
fn test_pool1d_custom_stride_output_shape() {
    // input [1, 1, 8], kernel=3, stride=2, padding=0
    // out_len = (8 - 3)/2 + 1 = 3
    let x = DynTensor::full(&[1, 1, 8], 1.0, DType::F32, &Device::Cpu).unwrap();
    let pool = MaxPool1d::new(Pool1dConfig::new(3).with_stride(2)).unwrap();
    let y = pool.forward(&x).unwrap();
    assert_eq!(y.dims(), &[1, 1, 3]);
}

#[test]
fn test_pool2d_output_shape() {
    // input [1, 1, 6, 6], kernel=2, stride=2 -> [1, 1, 3, 3]
    let x = DynTensor::full(&[1, 1, 6, 6], 1.0, DType::F32, &Device::Cpu).unwrap();
    let pool = MaxPool2d::new(Pool2dConfig::new(2)).unwrap();
    let y = pool.forward(&x).unwrap();
    assert_eq!(y.dims(), &[1, 1, 3, 3]);
}

#[test]
fn test_pool2d_with_padding_output_shape() {
    // input [1, 1, 5, 5], kernel=3, stride=1, padding=1
    // out = (5 + 2 - 3)/1 + 1 = 5 (same-padding)
    let x = DynTensor::full(&[1, 1, 5, 5], 1.0, DType::F32, &Device::Cpu).unwrap();
    let pool = MaxPool2d::new(Pool2dConfig::new(3).with_stride(1).with_padding(1)).unwrap();
    let y = pool.forward(&x).unwrap();
    assert_eq!(y.dims(), &[1, 1, 5, 5]);
}

#[test]
fn test_pool_rejects_zero_kernel() {
    let err = MaxPool1d::new(Pool1dConfig::new(0));
    assert!(err.is_err(), "kernel_size=0 should be rejected");
}

// =============================================================================
// 3. LayerNormConfig — epsilon and elementwise_affine defaults
// =============================================================================

#[test]
fn test_layer_norm_config_default() {
    let cfg = LayerNormConfig::default();
    assert!(
        (cfg.eps - 1e-5).abs() < 1e-12,
        "default eps should be 1e-5, got {}",
        cfg.eps
    );
}

#[test]
fn test_layer_norm_config_custom_eps() {
    let cfg = LayerNormConfig::new(1e-12);
    assert!(
        (cfg.eps - 1e-12).abs() < 1e-18,
        "custom eps should be 1e-12, got {}",
        cfg.eps
    );
}

#[test]
fn test_layer_norm_forward_preserves_shape() {
    let d = 8;
    let weight = DynTensor::ones(&[d], DType::F32, &Device::Cpu).unwrap();
    let bias = DynTensor::zeros(&[d], DType::F32, &Device::Cpu).unwrap();
    let ln = LayerNorm::new(weight, bias, 1e-5).unwrap();
    let x = DynTensor::from_vec(
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        &[1, d],
        &Device::Cpu,
    )
    .unwrap();
    let y = ln.forward(&x).unwrap();
    assert_eq!(y.dims(), &[1, d]);
}

#[test]
fn test_layer_norm_rejects_mismatched_weight_bias() {
    let weight = DynTensor::ones(&[4], DType::F32, &Device::Cpu).unwrap();
    let bias = DynTensor::zeros(&[8], DType::F32, &Device::Cpu).unwrap();
    let result = LayerNorm::new(weight, bias, 1e-5);
    assert!(result.is_err(), "mismatched weight/bias shapes should fail");
}

// =============================================================================
// 4. BatchNormConfig — momentum, epsilon, track_running_stats defaults
// =============================================================================

#[test]
fn test_batch_norm_config_default() {
    let cfg = BatchNormConfig::default();
    assert!((cfg.eps - 1e-5).abs() < 1e-12, "default eps should be 1e-5");
    assert!(
        (cfg.momentum - 0.1).abs() < 1e-12,
        "default momentum should be 0.1"
    );
    assert!(cfg.remove_mean, "default remove_mean should be true");
    assert!(cfg.affine, "default affine should be true");
}

#[test]
fn test_batch_norm_config_builder() {
    let cfg = BatchNormConfig::new(1e-3)
        .with_momentum(0.01)
        .with_remove_mean(false)
        .with_affine(false);
    assert!((cfg.eps - 1e-3).abs() < 1e-12);
    assert!((cfg.momentum - 0.01).abs() < 1e-12);
    assert!(!cfg.remove_mean);
    assert!(!cfg.affine);
}

#[test]
fn test_batch_norm_forward_preserves_shape() {
    let channels = 3;
    let mean = DynTensor::zeros(&[channels], DType::F32, &Device::Cpu).unwrap();
    let var = DynTensor::ones(&[channels], DType::F32, &Device::Cpu).unwrap();
    let bn = BatchNorm::new(mean, var, None, None, 1e-5).unwrap();
    let x = DynTensor::ones(&[2, channels, 4], DType::F32, &Device::Cpu).unwrap();
    let y = bn.forward(&x).unwrap();
    assert_eq!(y.dims(), &[2, channels, 4]);
}

#[test]
fn test_batch_norm_rejects_rank1_input() {
    let mean = DynTensor::zeros(&[4], DType::F32, &Device::Cpu).unwrap();
    let var = DynTensor::ones(&[4], DType::F32, &Device::Cpu).unwrap();
    let bn = BatchNorm::new(mean, var, None, None, 1e-5).unwrap();
    let x = DynTensor::ones(&[4], DType::F32, &Device::Cpu).unwrap();
    let result = bn.forward(&x);
    assert!(
        result.is_err(),
        "rank-1 input should be rejected by BatchNorm"
    );
}

// =============================================================================
// 5. Linear — weight/bias shape validation
// =============================================================================

#[test]
fn test_linear_weight_shape() {
    let in_features = 8;
    let out_features = 4;
    let weight = DynTensor::ones(&[out_features, in_features], DType::F32, &Device::Cpu).unwrap();
    let bias = DynTensor::zeros(&[out_features], DType::F32, &Device::Cpu).unwrap();
    let layer = Linear::new(weight, Some(bias)).unwrap();
    assert_eq!(layer.weight().dims(), &[out_features, in_features]);
    assert_eq!(layer.in_features(), in_features);
    assert_eq!(layer.out_features(), out_features);
}

#[test]
fn test_linear_no_bias() {
    let weight = DynTensor::ones(&[4, 8], DType::F32, &Device::Cpu).unwrap();
    let layer = Linear::new(weight, None).unwrap();
    assert!(layer.bias().is_none());
}

#[test]
fn test_linear_rejects_wrong_rank() {
    let weight = DynTensor::ones(&[4], DType::F32, &Device::Cpu).unwrap();
    let result = Linear::new(weight, None);
    assert!(result.is_err(), "1D weight should be rejected");
}

#[test]
fn test_linear_rejects_mismatched_bias() {
    let weight = DynTensor::ones(&[4, 8], DType::F32, &Device::Cpu).unwrap();
    let bias = DynTensor::zeros(&[8], DType::F32, &Device::Cpu).unwrap(); // should be [4]
    let result = Linear::new(weight, Some(bias));
    assert!(result.is_err(), "bias dim must match out_features");
}

#[test]
fn test_linear_forward_shape() {
    let in_f = 8;
    let out_f = 4;
    let weight = DynTensor::ones(&[out_f, in_f], DType::F32, &Device::Cpu).unwrap();
    let layer = Linear::new(weight, None).unwrap();
    // [batch=2, in_f] -> [batch=2, out_f]
    let x = DynTensor::ones(&[2, in_f], DType::F32, &Device::Cpu).unwrap();
    let y = layer.forward(&x).unwrap();
    assert_eq!(y.dims(), &[2, out_f]);
}

#[test]
fn test_linear_forward_3d_input() {
    // Batched linear: [B, S, in_f] -> [B, S, out_f]
    let in_f = 4;
    let out_f = 6;
    let weight = DynTensor::ones(&[out_f, in_f], DType::F32, &Device::Cpu).unwrap();
    let layer = Linear::new(weight, None).unwrap();
    let x = DynTensor::ones(&[2, 3, in_f], DType::F32, &Device::Cpu).unwrap();
    let y = layer.forward(&x).unwrap();
    assert_eq!(y.dims(), &[2, 3, out_f]);
}

// =============================================================================
// 6. Embedding — lookup shape [batch, seq, dim]
// =============================================================================

#[test]
fn test_embedding_output_shape_1d() {
    let vocab = 10;
    let dim = 16;
    let weight = DynTensor::ones(&[vocab, dim], DType::F32, &Device::Cpu).unwrap();
    let emb = Embedding::new(weight).unwrap();
    let result = emb.forward_ids(&[0, 3, 7]).unwrap();
    assert_eq!(result.dims(), &[3, dim]);
}

#[test]
fn test_embedding_output_shape_2d() {
    // Forward with [batch=2, seq=3] integer indices -> [2, 3, dim]
    let vocab = 10;
    let dim = 8;
    let weight = DynTensor::ones(&[vocab, dim], DType::F32, &Device::Cpu).unwrap();
    let emb = Embedding::new(weight).unwrap();
    let ids = DynTensor::from_vec_u32(vec![0, 1, 2, 3, 4, 5], &[2, 3], &Device::Cpu).unwrap();
    let y = emb.forward(&ids).unwrap();
    assert_eq!(y.dims(), &[2, 3, dim]);
}

#[test]
fn test_embedding_rejects_wrong_weight_rank() {
    let weight = DynTensor::ones(&[10], DType::F32, &Device::Cpu).unwrap();
    let result = Embedding::new(weight);
    assert!(result.is_err(), "1D weight should be rejected");
}

#[test]
fn test_embedding_out_of_range() {
    let weight = DynTensor::ones(&[5, 4], DType::F32, &Device::Cpu).unwrap();
    let emb = Embedding::new(weight).unwrap();
    let result = emb.forward_ids(&[5]); // vocab=5, index 5 is out of range
    assert!(result.is_err(), "index >= vocab_size should fail");
}

#[test]
fn test_embedding_weight_accessor() {
    let weight = DynTensor::ones(&[8, 4], DType::F32, &Device::Cpu).unwrap();
    let emb = Embedding::new(weight).unwrap();
    assert_eq!(emb.weight().dims(), &[8, 4]);
    assert_eq!(emb.embeddings().dims(), &[8, 4]);
}

// =============================================================================
// 7. Dropout — inference mode identity behavior
// =============================================================================

#[test]
fn test_dropout_is_identity_in_forward() {
    let d = Dropout::new(0.5);
    let x = DynTensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2], &Device::Cpu).unwrap();
    let y = d.forward(&x).unwrap();
    assert_eq!(y.to_flat_vec::<f32>().unwrap(), vec![1.0, 2.0, 3.0, 4.0]);
}

#[test]
fn test_dropout_forward_t_train_mode_still_identity() {
    // nn is inference-only; even forward_t(train=true) is identity
    let d = Dropout::new(0.5);
    let x = DynTensor::from_vec(vec![5.0, 6.0], &[2], &Device::Cpu).unwrap();
    let y_train = d.forward_t(&x, true).unwrap();
    let y_eval = d.forward_t(&x, false).unwrap();
    assert_eq!(
        y_train.to_flat_vec::<f32>().unwrap(),
        y_eval.to_flat_vec::<f32>().unwrap(),
        "train and eval should be identical for inference-only dropout"
    );
}

#[test]
fn test_dropout_preserves_shape_multidim() {
    let d = Dropout::new(0.1);
    let x = DynTensor::ones(&[3, 4, 5], DType::F32, &Device::Cpu).unwrap();
    let y = d.forward(&x).unwrap();
    assert_eq!(y.dims(), &[3, 4, 5]);
}

// =============================================================================
// 8. Sequential — forward pass chains through multiple layers
// =============================================================================

#[test]
fn test_sequential_chains_closures() {
    let mut seq = Sequential::new();
    seq.add_fn(|x| x.mul_scalar(2.0));
    seq.add_fn(|x| x.add_scalar(1.0));
    assert_eq!(seq.len(), 2);
    assert!(!seq.is_empty());
    let x = DynTensor::from_vec(vec![1.0, 2.0, 3.0], &[3], &Device::Cpu).unwrap();
    let y = seq.forward(&x).unwrap();
    // (x * 2) + 1 = [3.0, 5.0, 7.0]
    assert_eq!(y.to_flat_vec::<f32>().unwrap(), vec![3.0, 5.0, 7.0]);
}

#[test]
fn test_sequential_with_module() {
    let mut seq = Sequential::new();
    seq.add(Activation::Relu);
    seq.add(Dropout::new(0.0)); // identity
    seq.add(Activation::Sigmoid);
    assert_eq!(seq.len(), 3);
    let x = DynTensor::from_vec(vec![-1.0, 0.0, 1.0], &[3], &Device::Cpu).unwrap();
    let y = seq.forward(&x).unwrap();
    let vals = y.to_flat_vec::<f32>().unwrap();
    // relu(-1, 0, 1) = (0, 0, 1), sigmoid(0, 0, 1) = (0.5, 0.5, ~0.731)
    assert!((vals[0] - 0.5).abs() < 1e-4);
    assert!((vals[1] - 0.5).abs() < 1e-4);
    assert!((vals[2] - 0.7311).abs() < 0.01);
}

#[test]
fn test_sequential_empty_is_identity() {
    let seq = Sequential::new();
    assert!(seq.is_empty());
    let x = DynTensor::from_vec(vec![42.0], &[1], &Device::Cpu).unwrap();
    let y = seq.forward(&x).unwrap();
    assert_eq!(y.to_flat_vec::<f32>().unwrap(), vec![42.0]);
}

#[test]
fn test_sequential_single_layer() {
    let mut seq = Sequential::new();
    seq.add_fn(DynTensor::neg);
    let x = DynTensor::from_vec(vec![1.0, -2.0], &[2], &Device::Cpu).unwrap();
    let y = seq.forward(&x).unwrap();
    assert_eq!(y.to_flat_vec::<f32>().unwrap(), vec![-1.0, 2.0]);
}

// =============================================================================
// 9. Activation functions — enum variant coverage
// =============================================================================

#[test]
fn test_activation_enum_relu() {
    let a = Activation::Relu;
    let x = DynTensor::from_vec(vec![-2.0, 0.0, 3.0], &[3], &Device::Cpu).unwrap();
    let y = a.forward(&x).unwrap();
    assert_eq!(y.to_flat_vec::<f32>().unwrap(), vec![0.0, 0.0, 3.0]);
}

#[test]
fn test_activation_enum_gelu() {
    let a = Activation::Gelu;
    let x = DynTensor::from_vec(vec![0.0], &[1], &Device::Cpu).unwrap();
    let y = a.forward(&x).unwrap();
    // GELU(0) = 0
    assert!(y.to_scalar::<f32>().unwrap().abs() < 1e-6);
}

#[test]
fn test_activation_enum_silu() {
    let a = Activation::Silu;
    let x = DynTensor::from_vec(vec![0.0], &[1], &Device::Cpu).unwrap();
    let y = a.forward(&x).unwrap();
    // SiLU(0) = 0 * sigmoid(0) = 0
    assert!(y.to_scalar::<f32>().unwrap().abs() < 1e-6);
}

#[test]
fn test_activation_enum_sigmoid() {
    let a = Activation::Sigmoid;
    let x = DynTensor::from_vec(vec![0.0], &[1], &Device::Cpu).unwrap();
    let y = a.forward(&x).unwrap();
    assert!((y.to_scalar::<f32>().unwrap() - 0.5).abs() < 1e-6);
}

#[test]
fn test_activation_enum_tanh() {
    let a = Activation::Tanh;
    let x = DynTensor::from_vec(vec![0.0], &[1], &Device::Cpu).unwrap();
    let y = a.forward(&x).unwrap();
    assert!(y.to_scalar::<f32>().unwrap().abs() < 1e-6);
}

#[test]
fn test_activation_enum_elu() {
    let a = Activation::Elu(1.0);
    let x = DynTensor::from_vec(vec![-1.0, 0.0, 1.0], &[3], &Device::Cpu).unwrap();
    let y = a.forward(&x).unwrap();
    let vals = y.to_flat_vec::<f32>().unwrap();
    // ELU(-1, alpha=1) = e^{-1} - 1 ~ -0.6321
    assert!((vals[0] - (-0.6321)).abs() < 0.001);
    assert_eq!(vals[1], 0.0);
    assert_eq!(vals[2], 1.0);
}

#[test]
fn test_activation_enum_leaky_relu() {
    let a = Activation::LeakyRelu(0.01);
    let x = DynTensor::from_vec(vec![-10.0, 0.0, 10.0], &[3], &Device::Cpu).unwrap();
    let y = a.forward(&x).unwrap();
    let vals = y.to_flat_vec::<f32>().unwrap();
    assert!((vals[0] - (-0.1)).abs() < 1e-5);
    assert_eq!(vals[1], 0.0);
    assert_eq!(vals[2], 10.0);
}

#[test]
fn test_activation_is_copy_and_debug() {
    let a = Activation::Relu;
    let b = a; // Copy
    assert_eq!(a, b); // PartialEq
    let _ = format!("{a:?}"); // Debug
}

// =============================================================================
// 10. Module trait — forward/forward_t and training mode
// =============================================================================

#[test]
fn test_module_trait_forward() {
    // Any implementor of Module can be called via forward()
    let layer = Activation::Relu;
    let x = DynTensor::from_vec(vec![-1.0, 1.0], &[2], &Device::Cpu).unwrap();
    let y = Module::forward(&layer, &x).unwrap();
    assert_eq!(y.to_flat_vec::<f32>().unwrap(), vec![0.0, 1.0]);
}

#[test]
fn test_module_t_blanket_impl() {
    // ModuleT blanket impl ignores train flag
    let layer = Activation::Sigmoid;
    let x = DynTensor::from_vec(vec![0.0], &[1], &Device::Cpu).unwrap();
    let y_train = layer.forward_t(&x, true).unwrap();
    let y_eval = layer.forward_t(&x, false).unwrap();
    assert_eq!(
        y_train.to_flat_vec::<f32>().unwrap(),
        y_eval.to_flat_vec::<f32>().unwrap()
    );
}

#[test]
fn test_closure_as_module() {
    let layer = |x: &DynTensor| x.mul_scalar(3.0);
    let x = DynTensor::from_vec(vec![1.0, 2.0], &[2], &Device::Cpu).unwrap();
    let y = layer.forward(&x).unwrap();
    assert_eq!(y.to_flat_vec::<f32>().unwrap(), vec![3.0, 6.0]);
}

#[test]
fn test_option_none_module_is_identity() {
    let none_module: Option<&Activation> = None;
    let x = DynTensor::from_vec(vec![1.0, 2.0], &[2], &Device::Cpu).unwrap();
    let y = none_module.forward(&x).unwrap();
    assert_eq!(y.to_flat_vec::<f32>().unwrap(), vec![1.0, 2.0]);
}

#[test]
fn test_option_some_module_delegates() {
    let act = Activation::Relu;
    let some_module = Some(&act);
    let x = DynTensor::from_vec(vec![-1.0, 2.0], &[2], &Device::Cpu).unwrap();
    let y = some_module.forward(&x).unwrap();
    assert_eq!(y.to_flat_vec::<f32>().unwrap(), vec![0.0, 2.0]);
}

// =============================================================================
// Cross-layer interaction: Conv1d -> Linear shape validation
// =============================================================================

#[test]
fn test_conv1d_construction_and_forward() {
    let out_ch = 4;
    let in_ch = 2;
    let kernel = 3;
    let weight = DynTensor::ones(&[out_ch, in_ch, kernel], DType::F32, &Device::Cpu).unwrap();
    let bias = DynTensor::zeros(&[out_ch], DType::F32, &Device::Cpu).unwrap();
    let cfg = Conv1dConfig::new(1, 1, 1); // padding=1 for same-size
    let conv = Conv1d::new(weight, Some(bias), cfg).unwrap();
    assert_eq!(conv.weight().dims(), &[out_ch, in_ch, kernel]);
    assert!(conv.bias().is_some());

    // Forward: [B=1, in_ch, L=8] -> [B=1, out_ch, L=8] (same padding with k=3, p=1, s=1, d=1)
    let x = DynTensor::ones(&[1, in_ch, 8], DType::F32, &Device::Cpu).unwrap();
    let y = conv.forward(&x).unwrap();
    assert_eq!(y.dims(), &[1, out_ch, 8]);
}

#[test]
fn test_conv2d_construction_and_forward() {
    let out_ch = 4;
    let in_ch = 3;
    let kh = 3;
    let kw = 3;
    let weight = DynTensor::ones(&[out_ch, in_ch, kh, kw], DType::F32, &Device::Cpu).unwrap();
    let cfg = Conv2dConfig::new(1, 1, 1); // same padding
    let conv = Conv2d::new(weight, None, cfg).unwrap();
    assert!(conv.bias().is_none());

    // Forward: [1, 3, 8, 8] -> [1, 4, 8, 8]
    let x = DynTensor::ones(&[1, in_ch, 8, 8], DType::F32, &Device::Cpu).unwrap();
    let y = conv.forward(&x).unwrap();
    assert_eq!(y.dims(), &[1, out_ch, 8, 8]);
}

#[test]
fn test_conv1d_rejects_wrong_weight_rank() {
    let weight = DynTensor::ones(&[4, 2], DType::F32, &Device::Cpu).unwrap();
    let result = Conv1d::new(weight, None, Conv1dConfig::default());
    assert!(result.is_err(), "2D weight should be rejected for Conv1d");
}

#[test]
fn test_conv1d_rejects_zero_groups() {
    let weight = DynTensor::ones(&[4, 2, 3], DType::F32, &Device::Cpu).unwrap();
    let cfg = Conv1dConfig::default().with_groups(0);
    let result = Conv1d::new(weight, None, cfg);
    assert!(result.is_err(), "groups=0 should be rejected");
}

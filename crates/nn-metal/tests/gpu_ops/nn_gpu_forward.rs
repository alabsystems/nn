// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! GPU forward tests for nn layers (#1081).
//!
//! Each test constructs an nn layer with GPU weights, runs forward on a GPU
//! input tensor, and compares the result against the same layer run on CPU
//! within a tolerance. This validates that the Metal DynTensor backend
//! correctly supports matmul, broadcast ops, reductions, and activations
//! as composed by nn layer forward passes.
//!
//! Activation, shape, and conv op tests are in `nn_gpu_forward_ops.rs`.
//! LSTM tests are in `nn_gpu_forward_lstm.rs`.

use super::test_utils::{assert_gpu_cpu_close, gpu_init};
use nn_core::dyn_tensor::DynTensor;
use nn_core::layers::{Embedding, GroupNorm, LayerNorm, Linear, Module, RmsNorm};
use nn_core::Device;

const TOL: f32 = 1e-4;

fn init() {
    gpu_init();
}

fn assert_close(gpu_result: &DynTensor, cpu_result: &DynTensor, label: &str) {
    assert_gpu_cpu_close(gpu_result, cpu_result, TOL, label);
}

// -- AC1: Linear GPU forward --------------------------------------------------

#[test]
fn test_linear_no_bias_gpu() {
    init();
    let w_data = vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0];
    let x_data = vec![3.0, 5.0, 7.0];

    // CPU reference
    let w_cpu = DynTensor::new(&w_data, &[2, 3], &Device::Cpu).unwrap();
    let x_cpu = DynTensor::new(&x_data, &[1, 3], &Device::Cpu).unwrap();
    let linear_cpu = Linear::new(w_cpu, None).unwrap();
    let y_cpu = linear_cpu.forward(&x_cpu).unwrap();

    // GPU
    let w_gpu = DynTensor::new(&w_data, &[2, 3], &Device::metal()).unwrap();
    let x_gpu = DynTensor::new(&x_data, &[1, 3], &Device::metal()).unwrap();
    let linear_gpu = Linear::new(w_gpu, None).unwrap();
    let y_gpu = linear_gpu.forward(&x_gpu).unwrap();

    assert_eq!(y_gpu.device(), Device::metal());
    assert_eq!(y_gpu.dims(), &[1, 2]);
    assert_close(&y_gpu, &y_cpu, "linear_no_bias");
}

#[test]
fn test_linear_with_bias_gpu() {
    init();
    let w_data = vec![1.0, 0.0, 0.0, 1.0];
    let b_data = vec![10.0, 20.0];
    let x_data = vec![1.0, 2.0];

    // CPU reference
    let w_cpu = DynTensor::new(&w_data, &[2, 2], &Device::Cpu).unwrap();
    let b_cpu = DynTensor::new(&b_data, &[2], &Device::Cpu).unwrap();
    let x_cpu = DynTensor::new(&x_data, &[1, 2], &Device::Cpu).unwrap();
    let linear_cpu = Linear::new(w_cpu, Some(b_cpu)).unwrap();
    let y_cpu = linear_cpu.forward(&x_cpu).unwrap();

    // GPU
    let w_gpu = DynTensor::new(&w_data, &[2, 2], &Device::metal()).unwrap();
    let b_gpu = DynTensor::new(&b_data, &[2], &Device::metal()).unwrap();
    let x_gpu = DynTensor::new(&x_data, &[1, 2], &Device::metal()).unwrap();
    let linear_gpu = Linear::new(w_gpu, Some(b_gpu)).unwrap();
    let y_gpu = linear_gpu.forward(&x_gpu).unwrap();

    assert_eq!(y_gpu.device(), Device::metal());
    assert_close(&y_gpu, &y_cpu, "linear_with_bias");
}

#[test]
fn test_linear_batch_gpu() {
    init();
    let w_data = vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0];
    let x_data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];

    // CPU reference
    let w_cpu = DynTensor::new(&w_data, &[4, 3], &Device::Cpu).unwrap();
    let x_cpu = DynTensor::new(&x_data, &[2, 3], &Device::Cpu).unwrap();
    let linear_cpu = Linear::new(w_cpu, None).unwrap();
    let y_cpu = linear_cpu.forward(&x_cpu).unwrap();

    // GPU
    let w_gpu = DynTensor::new(&w_data, &[4, 3], &Device::metal()).unwrap();
    let x_gpu = DynTensor::new(&x_data, &[2, 3], &Device::metal()).unwrap();
    let linear_gpu = Linear::new(w_gpu, None).unwrap();
    let y_gpu = linear_gpu.forward(&x_gpu).unwrap();

    assert_eq!(y_gpu.dims(), &[2, 4]);
    assert_close(&y_gpu, &y_cpu, "linear_batch");
}

// -- AC2: LayerNorm GPU forward -----------------------------------------------

#[test]
fn test_layer_norm_gpu() {
    init();
    let dim = 4;
    let w_data = vec![1.0; dim];
    let b_data = vec![0.0; dim];
    let x_data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let eps = 1e-5;

    // CPU reference
    let w_cpu = DynTensor::new(&w_data, &[dim], &Device::Cpu).unwrap();
    let b_cpu = DynTensor::new(&b_data, &[dim], &Device::Cpu).unwrap();
    let x_cpu = DynTensor::new(&x_data, &[2, dim], &Device::Cpu).unwrap();
    let ln_cpu = LayerNorm::new(w_cpu, b_cpu, eps).unwrap();
    let y_cpu = ln_cpu.forward(&x_cpu).unwrap();

    // GPU
    let w_gpu = DynTensor::new(&w_data, &[dim], &Device::metal()).unwrap();
    let b_gpu = DynTensor::new(&b_data, &[dim], &Device::metal()).unwrap();
    let x_gpu = DynTensor::new(&x_data, &[2, dim], &Device::metal()).unwrap();
    let ln_gpu = LayerNorm::new(w_gpu, b_gpu, eps).unwrap();
    let y_gpu = ln_gpu.forward(&x_gpu).unwrap();

    assert_eq!(y_gpu.device(), Device::metal());
    assert_eq!(y_gpu.dims(), &[2, dim]);
    assert_close(&y_gpu, &y_cpu, "layer_norm");
}

#[test]
fn test_layer_norm_affine_gpu() {
    init();
    let dim = 3;
    let w_data = vec![2.0, 0.5, 1.0];
    let b_data = vec![0.1, -0.2, 0.3];
    let x_data = vec![1.0, 5.0, 3.0];
    let eps = 1e-5;

    // CPU reference
    let w_cpu = DynTensor::new(&w_data, &[dim], &Device::Cpu).unwrap();
    let b_cpu = DynTensor::new(&b_data, &[dim], &Device::Cpu).unwrap();
    let x_cpu = DynTensor::new(&x_data, &[1, dim], &Device::Cpu).unwrap();
    let ln_cpu = LayerNorm::new(w_cpu, b_cpu, eps).unwrap();
    let y_cpu = ln_cpu.forward(&x_cpu).unwrap();

    // GPU
    let w_gpu = DynTensor::new(&w_data, &[dim], &Device::metal()).unwrap();
    let b_gpu = DynTensor::new(&b_data, &[dim], &Device::metal()).unwrap();
    let x_gpu = DynTensor::new(&x_data, &[1, dim], &Device::metal()).unwrap();
    let ln_gpu = LayerNorm::new(w_gpu, b_gpu, eps).unwrap();
    let y_gpu = ln_gpu.forward(&x_gpu).unwrap();

    assert_close(&y_gpu, &y_cpu, "layer_norm_affine");
}

// -- AC2: GroupNorm GPU forward -----------------------------------------------

#[test]
fn test_group_norm_gpu() {
    init();
    let num_groups = 2;
    let num_channels = 4;
    let spatial = 3;
    let w_data = vec![1.0; num_channels];
    let b_data = vec![0.0; num_channels];
    // [1, 4, 3] input — 1 batch, 4 channels, 3 spatial
    let x_data: Vec<f32> = (0..num_channels * spatial)
        .map(|i| (i as f32) * 0.5 + 1.0)
        .collect();
    let eps = 1e-5;

    // CPU reference
    let w_cpu = DynTensor::new(&w_data, &[num_channels], &Device::Cpu).unwrap();
    let b_cpu = DynTensor::new(&b_data, &[num_channels], &Device::Cpu).unwrap();
    let x_cpu = DynTensor::new(&x_data, &[1, num_channels, spatial], &Device::Cpu).unwrap();
    let gn_cpu = GroupNorm::new(num_groups, num_channels, w_cpu, b_cpu, eps).unwrap();
    let y_cpu = gn_cpu.forward(&x_cpu).unwrap();

    // GPU
    let w_gpu = DynTensor::new(&w_data, &[num_channels], &Device::metal()).unwrap();
    let b_gpu = DynTensor::new(&b_data, &[num_channels], &Device::metal()).unwrap();
    let x_gpu = DynTensor::new(&x_data, &[1, num_channels, spatial], &Device::metal()).unwrap();
    let gn_gpu = GroupNorm::new(num_groups, num_channels, w_gpu, b_gpu, eps).unwrap();
    let y_gpu = gn_gpu.forward(&x_gpu).unwrap();

    assert_eq!(y_gpu.device(), Device::metal());
    assert_eq!(y_gpu.dims(), &[1, num_channels, spatial]);
    assert_close(&y_gpu, &y_cpu, "group_norm");
}

// -- AC3: Broadcast binary ops on GPU -----------------------------------------

#[test]
fn test_broadcast_add_different_shapes_gpu() {
    init();
    // [2, 3] + [3] => broadcast add
    let a_data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let b_data = vec![10.0, 20.0, 30.0];

    // CPU reference
    let a_cpu = DynTensor::new(&a_data, &[2, 3], &Device::Cpu).unwrap();
    let b_cpu = DynTensor::new(&b_data, &[3], &Device::Cpu).unwrap();
    let y_cpu = a_cpu.broadcast_add(&b_cpu).unwrap();

    // GPU
    let a_gpu = DynTensor::new(&a_data, &[2, 3], &Device::metal()).unwrap();
    let b_gpu = DynTensor::new(&b_data, &[3], &Device::metal()).unwrap();
    let y_gpu = a_gpu.broadcast_add(&b_gpu).unwrap();

    assert_eq!(y_gpu.dims(), &[2, 3]);
    assert_close(&y_gpu, &y_cpu, "broadcast_add");
}

#[test]
fn test_broadcast_mul_different_shapes_gpu() {
    init();
    // [2, 1, 3] * [1, 4, 1] => [2, 4, 3]
    let a_data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let b_data = vec![10.0, 20.0, 30.0, 40.0];

    // CPU reference
    let a_cpu = DynTensor::new(&a_data, &[2, 1, 3], &Device::Cpu).unwrap();
    let b_cpu = DynTensor::new(&b_data, &[1, 4, 1], &Device::Cpu).unwrap();
    let y_cpu = a_cpu.broadcast_mul(&b_cpu).unwrap();

    // GPU
    let a_gpu = DynTensor::new(&a_data, &[2, 1, 3], &Device::metal()).unwrap();
    let b_gpu = DynTensor::new(&b_data, &[1, 4, 1], &Device::metal()).unwrap();
    let y_gpu = a_gpu.broadcast_mul(&b_gpu).unwrap();

    assert_eq!(y_gpu.dims(), &[2, 4, 3]);
    assert_close(&y_gpu, &y_cpu, "broadcast_mul");
}

#[test]
fn test_broadcast_sub_scalar_gpu() {
    init();
    // [2, 3] - [1] => broadcast sub with scalar
    let a_data = vec![10.0, 20.0, 30.0, 40.0, 50.0, 60.0];
    let b_data = vec![5.0];

    // CPU reference
    let a_cpu = DynTensor::new(&a_data, &[2, 3], &Device::Cpu).unwrap();
    let b_cpu = DynTensor::new(&b_data, &[1], &Device::Cpu).unwrap();
    let y_cpu = a_cpu.broadcast_sub(&b_cpu).unwrap();

    // GPU
    let a_gpu = DynTensor::new(&a_data, &[2, 3], &Device::metal()).unwrap();
    let b_gpu = DynTensor::new(&b_data, &[1], &Device::metal()).unwrap();
    let y_gpu = a_gpu.broadcast_sub(&b_gpu).unwrap();

    assert_eq!(y_gpu.dims(), &[2, 3]);
    assert_close(&y_gpu, &y_cpu, "broadcast_sub_scalar");
}

// -- AC5: Embedding GPU forward (#1287) ----------------------------------------

#[test]
fn test_embedding_gpu() {
    init();
    // Vocab=5, dim=3 weight table
    let w_data = vec![
        0.1, 0.2, 0.3, // row 0
        0.4, 0.5, 0.6, // row 1
        0.7, 0.8, 0.9, // row 2
        1.0, 1.1, 1.2, // row 3
        1.3, 1.4, 1.5, // row 4
    ];

    // CPU reference
    let w_cpu = DynTensor::new(&w_data, &[5, 3], &Device::Cpu).unwrap();
    let emb_cpu = Embedding::new(w_cpu).unwrap();
    let ids_cpu = DynTensor::from_vec_u32(vec![0, 2, 4], &[3], &Device::Cpu).unwrap();
    let y_cpu = emb_cpu.forward(&ids_cpu).unwrap();

    // GPU: weight on Metal, IDs on CPU (Embedding transfers IDs internally)
    let w_gpu = DynTensor::new(&w_data, &[5, 3], &Device::metal()).unwrap();
    let emb_gpu = Embedding::new(w_gpu).unwrap();
    let ids_gpu = DynTensor::from_vec_u32(vec![0, 2, 4], &[3], &Device::Cpu).unwrap();
    let y_gpu = emb_gpu.forward(&ids_gpu).unwrap();

    assert_eq!(y_gpu.dims(), &[3, 3]);
    assert_close(&y_gpu, &y_cpu, "embedding");
}

#[test]
fn test_embedding_2d_ids_gpu() {
    init();
    let w_data: Vec<f32> = (0..20).map(|i| i as f32 * 0.1).collect();

    // CPU reference: [B=2, S=3] IDs
    let w_cpu = DynTensor::new(&w_data, &[5, 4], &Device::Cpu).unwrap();
    let emb_cpu = Embedding::new(w_cpu).unwrap();
    let ids_cpu = DynTensor::from_vec_u32(vec![0, 1, 2, 3, 4, 0], &[2, 3], &Device::Cpu).unwrap();
    let y_cpu = emb_cpu.forward(&ids_cpu).unwrap();

    // GPU
    let w_gpu = DynTensor::new(&w_data, &[5, 4], &Device::metal()).unwrap();
    let emb_gpu = Embedding::new(w_gpu).unwrap();
    let ids_gpu = DynTensor::from_vec_u32(vec![0, 1, 2, 3, 4, 0], &[2, 3], &Device::Cpu).unwrap();
    let y_gpu = emb_gpu.forward(&ids_gpu).unwrap();

    assert_eq!(y_gpu.dims(), &[2, 3, 4]); // [B, S, embed_dim]
    assert_close(&y_gpu, &y_cpu, "embedding_2d_ids");
}

// -- AC6: RmsNorm GPU forward (#1287) ------------------------------------------

#[test]
fn test_rms_norm_gpu() {
    init();
    let dim = 4;
    let w_data = vec![1.0; dim];
    let x_data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let eps = 1e-5;

    // CPU reference
    let w_cpu = DynTensor::new(&w_data, &[dim], &Device::Cpu).unwrap();
    let x_cpu = DynTensor::new(&x_data, &[2, dim], &Device::Cpu).unwrap();
    let rn_cpu = RmsNorm::new(w_cpu, eps).unwrap();
    let y_cpu = rn_cpu.forward(&x_cpu).unwrap();

    // GPU
    let w_gpu = DynTensor::new(&w_data, &[dim], &Device::metal()).unwrap();
    let x_gpu = DynTensor::new(&x_data, &[2, dim], &Device::metal()).unwrap();
    let rn_gpu = RmsNorm::new(w_gpu, eps).unwrap();
    let y_gpu = rn_gpu.forward(&x_gpu).unwrap();

    assert_eq!(y_gpu.device(), Device::metal());
    assert_eq!(y_gpu.dims(), &[2, dim]);
    assert_close(&y_gpu, &y_cpu, "rms_norm");
}

#[test]
fn test_rms_norm_with_weight_gpu() {
    init();
    let w_data = vec![2.0, 0.5, 1.0];
    let x_data = vec![3.0, 4.0, 5.0];
    let eps = 1e-5;

    // CPU reference
    let w_cpu = DynTensor::new(&w_data, &[3], &Device::Cpu).unwrap();
    let x_cpu = DynTensor::new(&x_data, &[1, 3], &Device::Cpu).unwrap();
    let rn_cpu = RmsNorm::new(w_cpu, eps).unwrap();
    let y_cpu = rn_cpu.forward(&x_cpu).unwrap();

    // GPU
    let w_gpu = DynTensor::new(&w_data, &[3], &Device::metal()).unwrap();
    let x_gpu = DynTensor::new(&x_data, &[1, 3], &Device::metal()).unwrap();
    let rn_gpu = RmsNorm::new(w_gpu, eps).unwrap();
    let y_gpu = rn_gpu.forward(&x_gpu).unwrap();

    assert_eq!(y_gpu.device(), Device::metal());
    assert_close(&y_gpu, &y_cpu, "rms_norm_with_weight");
}

// -- AC7: LSTM GPU forward tests are in nn_gpu_forward_lstm.rs ----------------

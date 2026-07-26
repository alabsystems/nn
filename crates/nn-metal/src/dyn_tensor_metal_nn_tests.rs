#![allow(deprecated)]
// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! GPU nn layer forward tests — verifies nn layers (Linear, LayerNorm,
//! GroupNorm, etc.) produce correct results on Metal GPU tensors.
//! Part of #1081: zero GPU coverage for nn forward paths.
//!
//! Composite tests (LSTM, Embedding, multi-layer models) are in
//! `dyn_tensor_metal_nn_tests_composite.rs` (#1115 split).

use nn_core::dyn_tensor::DynTensor;
use nn_core::layers::{Conv2d, Conv2dConfig, GroupNorm, LayerNorm, Linear};
use nn_core::{DType, Device};

use crate::test_common::{assert_close, assert_gpu_matches_cpu, init};

#[path = "dyn_tensor_metal_nn_tests_composite.rs"]
mod composite;

#[path = "dyn_tensor_metal_nn_tests_model.rs"]
mod model;

#[path = "dyn_tensor_metal_nn_tests_decoder.rs"]
mod decoder;

#[path = "dyn_tensor_metal_nn_tests_qwen3.rs"]
mod qwen3;

#[path = "dyn_tensor_metal_nn_tests_parity.rs"]
mod parity;

// -- AC1: Linear GPU forward tests --------------------------------------------

#[test]
fn test_gpu_linear_no_bias() {
    // Identity-like weight: selects first 2 of 3 input features
    assert_gpu_matches_cpu(
        |dev| {
            let w = DynTensor::new(&[1.0, 0.0, 0.0, 0.0, 1.0, 0.0], &[2, 3], dev).unwrap();
            let layer = Linear::new(w, None).unwrap();
            let x = DynTensor::new(&[3.0, 5.0, 7.0], &[1, 3], dev).unwrap();
            (Box::new(layer), x)
        },
        1e-5,
        "linear_no_bias",
    );
}

#[test]
fn test_gpu_linear_with_bias() {
    assert_gpu_matches_cpu(
        |dev| {
            let w = DynTensor::new(&[1.0, 2.0, 0.5, 3.0], &[2, 2], dev).unwrap();
            let b = DynTensor::new(&[10.0, 20.0], &[2], dev).unwrap();
            let layer = Linear::new(w, Some(b)).unwrap();
            let x = DynTensor::new(&[1.0, 2.0], &[1, 2], dev).unwrap();
            (Box::new(layer), x)
        },
        1e-5,
        "linear_with_bias",
    );
}

#[test]
fn test_gpu_linear_batched() {
    // Batch of 2 inputs through a 3->2 linear layer
    assert_gpu_matches_cpu(
        |dev| {
            let w = DynTensor::new(&[1.0, 0.0, 0.0, 0.0, 1.0, 0.0], &[2, 3], dev).unwrap();
            let b = DynTensor::new(&[0.5, -0.5], &[2], dev).unwrap();
            let layer = Linear::new(w, Some(b)).unwrap();
            let x = DynTensor::new(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], dev).unwrap();
            (Box::new(layer), x)
        },
        1e-5,
        "linear_batched",
    );
}

// -- AC2: LayerNorm / GroupNorm GPU forward tests -----------------------------

#[test]
fn test_gpu_layer_norm_known_values() {
    // Input [1, -1]: mean=0, var=1, eps=0 -> normed = [1, -1]
    assert_gpu_matches_cpu(
        |dev| {
            let w = DynTensor::ones(&[2], DType::F32, dev).unwrap();
            let b = DynTensor::zeros(&[2], DType::F32, dev).unwrap();
            let layer = LayerNorm::new(w, b, 0.0).unwrap();
            let x = DynTensor::new(&[1.0, -1.0], &[1, 2], dev).unwrap();
            (Box::new(layer), x)
        },
        1e-5,
        "layer_norm_known",
    );
}

#[test]
fn test_gpu_layer_norm_with_affine() {
    // Input [3, 1]: mean=2, var=1, normed=[1, -1]. weight=2 bias=5 -> [7, 3]
    assert_gpu_matches_cpu(
        |dev| {
            let w = DynTensor::full(&[2], 2.0, DType::F32, dev).unwrap();
            let b = DynTensor::full(&[2], 5.0, DType::F32, dev).unwrap();
            let layer = LayerNorm::new(w, b, 0.0).unwrap();
            let x = DynTensor::new(&[3.0, 1.0], &[1, 2], dev).unwrap();
            (Box::new(layer), x)
        },
        1e-5,
        "layer_norm_affine",
    );
}

#[test]
fn test_gpu_layer_norm_batched() {
    // Batch of 3, hidden_size=4
    assert_gpu_matches_cpu(
        |dev| {
            let w = DynTensor::ones(&[4], DType::F32, dev).unwrap();
            let b = DynTensor::zeros(&[4], DType::F32, dev).unwrap();
            let layer = LayerNorm::new(w, b, 1e-5).unwrap();
            let x = DynTensor::new(
                &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 0.1, 0.2, 0.3, 0.4],
                &[3, 4],
                dev,
            )
            .unwrap();
            (Box::new(layer), x)
        },
        1e-4,
        "layer_norm_batched",
    );
}

#[test]
fn test_gpu_group_norm_single_group() {
    // 1 group = LayerNorm-like behavior over channels
    assert_gpu_matches_cpu(
        |dev| {
            let w = DynTensor::ones(&[4], DType::F32, dev).unwrap();
            let b = DynTensor::zeros(&[4], DType::F32, dev).unwrap();
            let layer = GroupNorm::new(1, 4, w, b, 1e-5).unwrap();
            let x =
                DynTensor::new(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], &[1, 4, 2], dev).unwrap();
            (Box::new(layer), x)
        },
        1e-4,
        "group_norm_single_group",
    );
}

#[test]
fn test_gpu_group_norm_multi_group() {
    // 2 groups of 2 channels each
    assert_gpu_matches_cpu(
        |dev| {
            let w = DynTensor::full(&[4], 2.0, DType::F32, dev).unwrap();
            let b = DynTensor::full(&[4], 1.0, DType::F32, dev).unwrap();
            let layer = GroupNorm::new(2, 4, w, b, 1e-5).unwrap();
            let x = DynTensor::new(
                &[
                    1.0, 2.0, 3.0, 4.0, 10.0, 20.0, 30.0, 40.0, 100.0, 200.0, 300.0, 400.0,
                ],
                &[1, 4, 3],
                dev,
            )
            .unwrap();
            (Box::new(layer), x)
        },
        1e-3,
        "group_norm_multi_group",
    );
}

// -- AC3: Broadcast binary op tests -------------------------------------------

#[test]
fn test_gpu_broadcast_add_scalar() {
    init();
    let a = DynTensor::new(&[1.0, 2.0, 3.0, 4.0], &[2, 2], &Device::metal()).unwrap();
    let s = DynTensor::new(&[10.0], &[1], &Device::metal()).unwrap();
    let r = a.broadcast_add(&s).unwrap();
    assert_eq!(r.device(), Device::metal());
    assert_eq!(r.dims(), &[2, 2]);
    let vals = r
        .to_device(&Device::Cpu)
        .unwrap()
        .to_flat_vec::<f32>()
        .unwrap();
    assert_close(
        &vals,
        &[11.0, 12.0, 13.0, 14.0],
        1e-6,
        "broadcast_add_scalar",
    );
}

#[test]
fn test_gpu_broadcast_mul_row_vector() {
    init();
    // [2, 3] * [1, 3] -> element-wise scale per column
    let a = DynTensor::new(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], &Device::metal()).unwrap();
    let scale = DynTensor::new(&[10.0, 100.0, 1000.0], &[1, 3], &Device::metal()).unwrap();
    let r = a.broadcast_mul(&scale).unwrap();
    assert_eq!(r.device(), Device::metal());
    assert_eq!(r.dims(), &[2, 3]);
    let vals = r
        .to_device(&Device::Cpu)
        .unwrap()
        .to_flat_vec::<f32>()
        .unwrap();
    assert_close(
        &vals,
        &[10.0, 200.0, 3000.0, 40.0, 500.0, 6000.0],
        1e-3,
        "broadcast_mul_row",
    );
}

#[test]
fn test_gpu_broadcast_sub_column_vector() {
    init();
    // [2, 3] - [2, 1] -> subtract per row
    let a = DynTensor::new(
        &[10.0, 20.0, 30.0, 40.0, 50.0, 60.0],
        &[2, 3],
        &Device::metal(),
    )
    .unwrap();
    let bias = DynTensor::new(&[1.0, 2.0], &[2, 1], &Device::metal()).unwrap();
    let r = a.broadcast_sub(&bias).unwrap();
    assert_eq!(r.device(), Device::metal());
    let vals = r
        .to_device(&Device::Cpu)
        .unwrap()
        .to_flat_vec::<f32>()
        .unwrap();
    assert_close(
        &vals,
        &[9.0, 19.0, 29.0, 38.0, 48.0, 58.0],
        1e-5,
        "broadcast_sub_column",
    );
}

#[test]
fn test_gpu_broadcast_div() {
    init();
    let a = DynTensor::new(&[10.0, 20.0, 30.0, 40.0], &[2, 2], &Device::metal()).unwrap();
    let d = DynTensor::new(&[2.0, 5.0], &[1, 2], &Device::metal()).unwrap();
    let r = a.broadcast_div(&d).unwrap();
    assert_eq!(r.device(), Device::metal());
    let vals = r
        .to_device(&Device::Cpu)
        .unwrap()
        .to_flat_vec::<f32>()
        .unwrap();
    assert_close(&vals, &[5.0, 4.0, 15.0, 8.0], 1e-5, "broadcast_div");
}

// -- AC4: Unary op tests (sigmoid, gelu, tanh) --------------------------------

#[test]
fn test_gpu_sigmoid() {
    init();
    let x = DynTensor::new(&[0.0, 1.0, -1.0, 10.0], &[4], &Device::metal()).unwrap();
    let r = x.sigmoid().unwrap();
    assert_eq!(r.device(), Device::metal());
    let vals = r
        .to_device(&Device::Cpu)
        .unwrap()
        .to_flat_vec::<f32>()
        .unwrap();
    assert_close(
        &vals,
        &[0.5, 0.7310586, 0.26894143, 0.9999546],
        1e-4,
        "sigmoid",
    );
}

#[test]
fn test_gpu_gelu() {
    init();
    let x = DynTensor::new(&[0.0, 1.0, -1.0, 2.0], &[4], &Device::metal()).unwrap();
    let r = x.gelu().unwrap();
    assert_eq!(r.device(), Device::metal());
    let vals = r
        .to_device(&Device::Cpu)
        .unwrap()
        .to_flat_vec::<f32>()
        .unwrap();
    let cpu_x = DynTensor::new(&[0.0, 1.0, -1.0, 2.0], &[4], &Device::Cpu).unwrap();
    let cpu_r = cpu_x.gelu().unwrap().to_flat_vec::<f32>().unwrap();
    assert_close(&vals, &cpu_r, 1e-4, "gelu");
}

#[test]
fn test_gpu_tanh() {
    init();
    let x = DynTensor::new(&[0.0, 1.0, -1.0, 3.0], &[4], &Device::metal()).unwrap();
    let r = x.tanh().unwrap();
    assert_eq!(r.device(), Device::metal());
    let vals = r
        .to_device(&Device::Cpu)
        .unwrap()
        .to_flat_vec::<f32>()
        .unwrap();
    assert_close(
        &vals,
        &[0.0, 0.7615942, -0.7615942, 0.9950548],
        1e-4,
        "tanh",
    );
}

#[test]
fn test_gpu_sigmoid_2d() {
    init();
    // 2D tensor to verify GPU dispatch handles multi-dim correctly
    let x = DynTensor::new(&[0.0, 1.0, -1.0, 2.0, -2.0, 0.5], &[2, 3], &Device::metal()).unwrap();
    let r = x.sigmoid().unwrap();
    assert_eq!(r.device(), Device::metal());
    assert_eq!(r.dims(), &[2, 3]);
    let vals = r
        .to_device(&Device::Cpu)
        .unwrap()
        .to_flat_vec::<f32>()
        .unwrap();
    let cpu_x = DynTensor::new(&[0.0, 1.0, -1.0, 2.0, -2.0, 0.5], &[2, 3], &Device::Cpu).unwrap();
    let cpu_r = cpu_x.sigmoid().unwrap().to_flat_vec::<f32>().unwrap();
    assert_close(&vals, &cpu_r, 1e-4, "sigmoid_2d");
}

// -- Conv2d GPU dispatch tests (#1291) ----------------------------------------

#[test]
fn test_gpu_conv2d_basic() {
    assert_gpu_matches_cpu(
        |dev| {
            let w = DynTensor::new(&[0.5, -0.3], &[2, 1, 1, 1], dev).unwrap();
            let layer = Conv2d::new(w, None, Conv2dConfig::default()).unwrap();
            let x = DynTensor::new(
                &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
                &[1, 1, 3, 3],
                dev,
            )
            .unwrap();
            (Box::new(layer), x)
        },
        1e-5,
        "conv2d_basic",
    );
}

#[test]
fn test_gpu_conv2d_with_padding() {
    assert_gpu_matches_cpu(
        |dev| {
            let w = DynTensor::new(
                &[1.0, 0.0, -1.0, 2.0, 0.0, -2.0, 1.0, 0.0, -1.0],
                &[1, 1, 3, 3],
                dev,
            )
            .unwrap();
            let mut cfg = Conv2dConfig::default();
            cfg.padding = 1;
            let layer = Conv2d::new(w, None, cfg).unwrap();
            let x = DynTensor::new(
                &[
                    1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0,
                    15.0, 16.0,
                ],
                &[1, 1, 4, 4],
                dev,
            )
            .unwrap();
            (Box::new(layer), x)
        },
        1e-4,
        "conv2d_padding",
    );
}

#[test]
fn test_gpu_conv2d_groups() {
    assert_gpu_matches_cpu(
        |dev| {
            let w = DynTensor::new(
                &[1.0, 1.0, 1.0, 1.0, -1.0, -1.0, -1.0, -1.0],
                &[2, 1, 2, 2],
                dev,
            )
            .unwrap();
            let mut cfg = Conv2dConfig::default();
            cfg.groups = 2;
            let layer = Conv2d::new(w, None, cfg).unwrap();
            let x = DynTensor::new(
                &[
                    1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                    1.0, 1.0,
                ],
                &[1, 2, 3, 3],
                dev,
            )
            .unwrap();
            (Box::new(layer), x)
        },
        1e-5,
        "conv2d_depthwise",
    );
}

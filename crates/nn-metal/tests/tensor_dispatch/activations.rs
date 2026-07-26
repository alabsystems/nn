// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Activation dispatch tests: standalone activation ops through Metal tensor dispatch.
//!
//! Extracted from `tensor_dispatch.rs` for the 500-line limit (#783 F1).
//!
//! Coverage: GLU, Sigmoid, GELU, ReLU, Tanh.

use super::test_utils::{assert_within_budget, glu_ref, metal_setup, rand_f32_vec};
use nn_dsl::{tensor_block_builder::TensorBlockBuilder, ScalarType};
use nn_metal::execute_tensor_dispatch;
use std::collections::HashMap;

// ===========================================================================
// GLU via execute_tensor_dispatch (#670)
// ===========================================================================

/// GLU dispatch: Narrow×2 + Sigmoid + BinaryMul executes through Metal dispatch.
///
/// Part of #670 AC3.
#[test]
fn test_glu_tensor_dispatch() {
    let cache = metal_setup();

    let (channels_2x, time) = (8, 16);
    let half = channels_2x / 2;

    let mut b = TensorBlockBuilder::new("glu_dispatch");
    let x = b.add_input("x", &[channels_2x, time]);
    let glu = b.add_glu(x, 0, &[channels_2x, time]).expect("even dim");
    let kernel = b.build(glu).expect("valid graph");

    // CPU reference
    let x_data = rand_f32_vec(0x6100_0001, channels_2x * time, -3.0, 3.0);
    let cpu_out = glu_ref(&x_data, channels_2x, time);

    // GPU via execute_tensor_dispatch
    let mut inputs = HashMap::new();
    inputs.insert("x", x_data);

    let gpu_out =
        execute_tensor_dispatch(&cache, &kernel, ScalarType::F32, &inputs).expect("GLU dispatch");

    assert_eq!(
        gpu_out.len(),
        half * time,
        "GLU output should have C/2 * T = {} elements, got {}",
        half * time,
        gpu_out.len()
    );
    assert_within_budget("glu_dispatch", &gpu_out, &cpu_out);
}

// ===========================================================================
// Sigmoid standalone via execute_tensor_dispatch (#676)
// ===========================================================================

/// Sigmoid dispatch: single Sigmoid op executes through Metal dispatch.
#[test]
fn test_sigmoid_tensor_dispatch() {
    let cache = metal_setup();

    let shape = [4_usize, 16];
    let total = shape[0] * shape[1];

    let mut b = TensorBlockBuilder::new("sigmoid_dispatch");
    let x = b.add_input("x", &shape);
    let sig = b.add_sigmoid(x, &shape);
    let kernel = b.build(sig).expect("valid graph");

    // CPU reference: sigmoid(x) = 1 / (1 + exp(-x))
    let x_data = rand_f32_vec(0x5160_0001, total, -5.0, 5.0);
    let cpu_out: Vec<f32> = x_data.iter().map(|&v| 1.0 / (1.0 + (-v).exp())).collect();

    let mut inputs = HashMap::new();
    inputs.insert("x", x_data);

    let gpu_out = execute_tensor_dispatch(&cache, &kernel, ScalarType::F32, &inputs)
        .expect("sigmoid dispatch");

    assert_eq!(gpu_out.len(), total, "sigmoid output length");
    assert_within_budget("sigmoid_dispatch", &gpu_out, &cpu_out);
}

// ===========================================================================
// GELU standalone via execute_tensor_dispatch (#676)
// ===========================================================================

/// GELU dispatch: single GELU op (tanh approximation) executes through Metal dispatch.
#[test]
fn test_gelu_tensor_dispatch() {
    let cache = metal_setup();

    let shape = [4_usize, 16];
    let total = shape[0] * shape[1];

    let mut b = TensorBlockBuilder::new("gelu_dispatch");
    let x = b.add_input("x", &shape);
    let gelu = b.add_gelu(x, &shape);
    let kernel = b.build(gelu).expect("valid graph");

    // CPU reference: GELU exp-based form (matches scalar kernel and MSL codegen after #679).
    // 0.5 * x * (2.0 - 2.0 / (exp(2 * inner) + 1.0))
    let x_data = rand_f32_vec(0x6E10_0001, total, -5.0, 5.0);
    let cpu_out: Vec<f32> = x_data
        .iter()
        .map(|&v| {
            let x = f64::from(v);
            let inner = 0.7978845608028654_f64 * (x + 0.044715 * x * x * x);
            let e2 = (2.0 * inner).exp();
            (0.5 * x * (2.0 - 2.0 / (e2 + 1.0))) as f32
        })
        .collect();

    let mut inputs = HashMap::new();
    inputs.insert("x", x_data);

    let gpu_out =
        execute_tensor_dispatch(&cache, &kernel, ScalarType::F32, &inputs).expect("gelu dispatch");

    assert_eq!(gpu_out.len(), total, "gelu output length");
    assert_within_budget("gelu_dispatch", &gpu_out, &cpu_out);
}

// ===========================================================================
// ConvTranspose1d via execute_tensor_dispatch (#676)
// ===========================================================================

/// ConvTranspose1d dispatch: standalone transposed convolution on Metal.
///
/// Demucs decoder pattern: in_ch=4, out_ch=2, kernel=3, stride=2, pad=1.
/// Uses small dimensions for tractable CI time.
#[test]
fn test_conv_transpose_1d_tensor_dispatch() {
    let cache = metal_setup();

    let (in_ch, out_ch, kernel_size, in_len, stride, padding) = (4, 2, 3, 8, 2, 1);
    let out_len = (in_len - 1) * stride + kernel_size - 2 * padding;

    let kernel = nn_dsl::conv_transpose_1d::build_conv_transpose_1d(
        "conv_transpose_test",
        in_ch,
        out_ch,
        kernel_size,
        in_len,
        stride,
        padding,
        1,
        1,
        false,
        0,
    )
    .expect("build conv_transpose_1d");

    let x_data = rand_f32_vec(0xCE_0001, in_ch * in_len, -1.0, 1.0);
    let w_data = rand_f32_vec(0xCE_0002, in_ch * out_ch * kernel_size, -0.5, 0.5);

    let cpu_out = super::test_utils::conv_transpose_1d_ref(
        &x_data,
        &w_data,
        None,
        in_ch,
        out_ch,
        kernel_size,
        in_len,
        stride,
        padding,
    );

    let mut inputs = HashMap::new();
    inputs.insert("data", x_data);
    inputs.insert("weight", w_data);

    let gpu_out = execute_tensor_dispatch(&cache, &kernel, ScalarType::F32, &inputs)
        .expect("conv_transpose_1d dispatch");

    assert_eq!(
        gpu_out.len(),
        out_ch * out_len,
        "conv_transpose_1d output length"
    );
    assert_within_budget("conv_transpose_1d", &gpu_out, &cpu_out);
}

// ===========================================================================
// ReLU standalone via execute_tensor_dispatch (#761 D1)
// ===========================================================================

/// ReLU dispatch: single ReLU op (max(x, 0)) executes through Metal tensor dispatch.
#[test]
fn test_relu_tensor_dispatch() {
    let cache = metal_setup();

    let shape = [4_usize, 16];
    let total = shape[0] * shape[1];

    let mut b = TensorBlockBuilder::new("relu_dispatch");
    let x = b.add_input("x", &shape);
    let relu = b.add_relu(x, &shape);
    let kernel = b.build(relu).expect("valid graph");

    let x_data = rand_f32_vec(0xAE10_0001, total, -10.0, 10.0);
    let cpu_out: Vec<f32> = x_data.iter().map(|&v| v.max(0.0)).collect();

    let mut inputs = HashMap::new();
    inputs.insert("x", x_data);

    let gpu_out =
        execute_tensor_dispatch(&cache, &kernel, ScalarType::F32, &inputs).expect("relu dispatch");

    assert_eq!(gpu_out.len(), total, "relu output length");
    assert_within_budget("relu_dispatch", &gpu_out, &cpu_out);
}

// ===========================================================================
// Tanh standalone via execute_tensor_dispatch (#761 D1)
// ===========================================================================

/// Tanh dispatch: single tanh op executes through Metal tensor dispatch.
#[test]
fn test_tanh_tensor_dispatch() {
    let cache = metal_setup();

    let shape = [4_usize, 16];
    let total = shape[0] * shape[1];

    let mut b = TensorBlockBuilder::new("tanh_dispatch");
    let x = b.add_input("x", &shape);
    let tanh_out = b.add_tanh(x, &shape);
    let kernel = b.build(tanh_out).expect("valid graph");

    let x_data = rand_f32_vec(0xBE10_0001, total, -5.0, 5.0);
    let cpu_out: Vec<f32> = x_data.iter().map(|&v| v.tanh()).collect();

    let mut inputs = HashMap::new();
    inputs.insert("x", x_data);

    let gpu_out =
        execute_tensor_dispatch(&cache, &kernel, ScalarType::F32, &inputs).expect("tanh dispatch");

    assert_eq!(gpu_out.len(), total, "tanh output length");
    assert_within_budget("tanh_dispatch", &gpu_out, &cpu_out);
}

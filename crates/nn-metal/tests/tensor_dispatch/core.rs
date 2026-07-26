// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Integration tests for `execute_tensor_dispatch` — the multi-step tensor
//! dispatch executor.
//!
//! Validates that the reusable executor in `nn_metal::tensor_dispatch`
//! produces GPU output matching Rust CPU reference implementations within
//! the precision budget.
//!
//! # Coverage
//!
//! - K5 RMSNorm: reduce → broadcast → elementwise chains (3 inputs, 12 nodes)
//! - K2 InstanceNorm: reduce → broadcast → elementwise (2 inputs, 12 nodes)
//! - Conv1d: no bias, with bias, dvoice pattern (stride=4, pad=2)
//!
//! Activation and ConvTranspose1d tests extracted to `tensor_dispatch_activations.rs`.
//!
//! See #315 for tracking.

use super::test_utils::{assert_within_budget, conv1d_ref, metal_setup, rand_f32_vec};
use nn_dsl::{
    build_instance_norm_decomposed, build_rms_norm_decomposed, instance_norm_ref, rms_norm_ref,
    ScalarType,
};
use nn_metal::execute_tensor_dispatch;
use std::collections::HashMap;

// ===========================================================================
// K5 RMSNorm via execute_tensor_dispatch
// ===========================================================================

#[test]
fn test_k5_rms_norm_tensor_dispatch() {
    let cache = metal_setup();

    let (n, hidden) = (4, 32);
    let total = n * hidden;
    let eps = 1e-5_f32;

    let kernel = build_rms_norm_decomposed(n, hidden).expect("build K5 RMSNorm");

    // CPU reference
    let x_data = rand_f32_vec(0xE5F6_A7B8, total, -2.0, 2.0);
    let weight = vec![1.0_f32; hidden];
    let cpu_out = rms_norm_ref(&x_data, &weight, n, hidden, eps).expect("rms_norm_ref");

    // GPU via execute_tensor_dispatch (named inputs)
    let mut inputs = HashMap::new();
    inputs.insert("x", x_data);
    inputs.insert("eps", vec![eps]);
    inputs.insert("weight", weight);

    let gpu_out = execute_tensor_dispatch(&cache, &kernel, ScalarType::F32, &inputs)
        .expect("execute_tensor_dispatch K5");

    assert_eq!(gpu_out.len(), cpu_out.len(), "K5 output length mismatch");
    assert_within_budget("rms_norm_dispatch", &gpu_out, &cpu_out);
}

// ===========================================================================
// K2 InstanceNorm via execute_tensor_dispatch
// ===========================================================================

#[test]
fn test_k2_instance_norm_tensor_dispatch() {
    let cache = metal_setup();

    let (b, c, t) = (1, 4, 32);
    let total = b * c * t;
    let eps = 1e-5_f32;

    let kernel = build_instance_norm_decomposed(b, c, t).expect("build K2 InstanceNorm");

    // CPU reference
    let x_data = rand_f32_vec(0xA1B2_C3D4, total, -3.0, 3.0);
    let cpu_out = instance_norm_ref(&x_data, b, c, t, eps).expect("instance_norm_ref");

    // GPU via execute_tensor_dispatch (named inputs)
    let mut inputs = HashMap::new();
    inputs.insert("x", x_data);
    inputs.insert("eps", vec![eps]);

    let gpu_out = execute_tensor_dispatch(&cache, &kernel, ScalarType::F32, &inputs)
        .expect("execute_tensor_dispatch K2");

    assert_eq!(gpu_out.len(), cpu_out.len(), "K2 output length mismatch");
    assert_within_budget("instance_norm_dispatch", &gpu_out, &cpu_out);
}

// ===========================================================================
// Conv1d via execute_tensor_dispatch
// ===========================================================================

/// Conv1d without bias — basic config matching test_build_conv1d_stride_padding
/// (dvoice pattern: in_ch=1, out_ch=48, kernel=8, stride=4, padding=2).
/// Uses a smaller in_length for faster CI.
#[test]
fn test_conv1d_tensor_dispatch_no_bias() {
    let cache = metal_setup();

    let (in_ch, out_ch, kernel_size, in_len, stride, padding) = (2, 3, 3, 8, 1, 0);
    let out_len = (in_len + 2 * padding - kernel_size) / stride + 1;

    let kernel = nn_dsl::conv1d::build_conv1d(
        "conv1d_test",
        in_ch,
        out_ch,
        kernel_size,
        in_len,
        stride,
        padding,
        false,
    )
    .expect("build conv1d");

    let x_data = rand_f32_vec(0xC0_0001, in_ch * in_len, -1.0, 1.0);
    let w_data = rand_f32_vec(0xC0_0002, out_ch * in_ch * kernel_size, -0.5, 0.5);

    let cpu_out = conv1d_ref(
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
        .expect("conv1d dispatch no bias");

    assert_eq!(gpu_out.len(), out_ch * out_len, "conv1d output length");
    assert_within_budget("conv1d_no_bias", &gpu_out, &cpu_out);
}

/// Conv1d with bias — verifies bias is added correctly per output channel.
#[test]
fn test_conv1d_tensor_dispatch_with_bias() {
    let cache = metal_setup();

    let (in_ch, out_ch, kernel_size, in_len, stride, padding) = (1, 2, 3, 8, 1, 0);
    let out_len = (in_len + 2 * padding - kernel_size) / stride + 1;

    let kernel = nn_dsl::conv1d::build_conv1d(
        "conv1d_bias",
        in_ch,
        out_ch,
        kernel_size,
        in_len,
        stride,
        padding,
        true,
    )
    .expect("build conv1d with bias");

    let x_data = rand_f32_vec(0xB1_0001, in_ch * in_len, -1.0, 1.0);
    let w_data = rand_f32_vec(0xB1_0002, out_ch * in_ch * kernel_size, -0.3, 0.3);
    let b_data = rand_f32_vec(0xB1_0003, out_ch, -0.1, 0.1);

    let cpu_out = conv1d_ref(
        &x_data,
        &w_data,
        Some(&b_data),
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
    inputs.insert("bias", b_data);

    let gpu_out = execute_tensor_dispatch(&cache, &kernel, ScalarType::F32, &inputs)
        .expect("conv1d dispatch with bias");

    assert_eq!(gpu_out.len(), out_ch * out_len, "conv1d+bias output length");
    assert_within_budget("conv1d_with_bias", &gpu_out, &cpu_out);
}

/// Conv1d dvoice Demucs pattern: stride=4, padding=2.
///
/// Matches the first encoder layer: in_ch=1, out_ch=48, kernel=8, stride=4, pad=2.
/// Uses in_len=64 for tractable CI time.
#[test]
fn test_conv1d_tensor_dispatch_dvoice_pattern() {
    let cache = metal_setup();

    let (in_ch, out_ch, kernel_size, in_len, stride, padding) = (1, 48, 8, 64, 4, 2);
    let out_len = (in_len + 2 * padding - kernel_size) / stride + 1;

    let kernel = nn_dsl::conv1d::build_conv1d(
        "conv1d_dvoice",
        in_ch,
        out_ch,
        kernel_size,
        in_len,
        stride,
        padding,
        false,
    )
    .expect("build conv1d dvoice");

    let x_data = rand_f32_vec(0xDA_0001, in_ch * in_len, -1.0, 1.0);
    let w_data = rand_f32_vec(0xDA_0002, out_ch * in_ch * kernel_size, -0.2, 0.2);

    let cpu_out = conv1d_ref(
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
        .expect("conv1d dispatch dvoice pattern");

    assert_eq!(
        gpu_out.len(),
        out_ch * out_len,
        "dvoice conv1d output length"
    );
    assert_within_budget("conv1d_dvoice", &gpu_out, &cpu_out);
}

// Activation dispatch tests (GLU, Sigmoid, GELU, ReLU, Tanh) and ConvTranspose1d
// extracted to tensor_dispatch_activations.rs for the 500-line limit (#783 F1).

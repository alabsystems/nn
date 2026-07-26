// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Differential dispatch tests for causal Conv1d (ZeroPad1d + Conv1d):
//! GPU output matches CPU reference implementation within precision budget.
//!
//! Part of #589 AC4.

use super::test_utils::{assert_within_budget, causal_conv1d_ref, metal_setup, rand_f32_vec};
use nn_dsl::ScalarType;
use nn_metal::execute_tensor_dispatch;
use std::collections::HashMap;

/// Causal Conv1d dispatch: ZeroPad1d + Conv1d(padding=0) on Metal.
///
/// Basic config: in_ch=2, out_ch=3, kernel=3, stride=1, dilation=1.
/// Verifies GPU output matches CPU reference for the pad-then-conv decomposition.
/// Part of #589 AC4.
#[test]
fn test_causal_conv1d_tensor_dispatch_basic() {
    let cache = metal_setup();

    let (in_ch, out_ch, kernel_size, in_len, stride, dilation, groups) = (2, 3, 3, 8, 1, 1, 1);
    let pad_left = (kernel_size - 1) * dilation;
    let eff_kernel = dilation * (kernel_size - 1) + 1;
    let out_len = (in_len + pad_left - eff_kernel) / stride + 1;

    let kernel = nn_dsl::build_causal_conv1d(
        "causal_conv1d_basic",
        in_ch,
        out_ch,
        kernel_size,
        in_len,
        stride,
        dilation,
        groups,
        false,
    )
    .expect("build causal conv1d");

    let x_data = rand_f32_vec(0xCA01_0001, in_ch * in_len, -1.0, 1.0);
    let w_data = rand_f32_vec(0xCA01_0002, out_ch * in_ch * kernel_size, -0.5, 0.5);

    let cpu_out = causal_conv1d_ref(
        &x_data,
        &w_data,
        None,
        in_ch,
        out_ch,
        kernel_size,
        in_len,
        stride,
        dilation,
    );

    let mut inputs = HashMap::new();
    inputs.insert("data", x_data);
    inputs.insert("weight", w_data);

    let gpu_out = execute_tensor_dispatch(&cache, &kernel, ScalarType::F32, &inputs)
        .expect("causal conv1d dispatch basic");

    assert_eq!(
        gpu_out.len(),
        out_ch * out_len,
        "causal conv1d output length"
    );
    assert_within_budget("causal_conv1d_basic", &gpu_out, &cpu_out);
}

/// Causal Conv1d dispatch: dilated, dvoice VocoderResBlock pattern.
///
/// Config: in_ch=4, out_ch=4, kernel=3, dilation=3 (ResBlock dilation sequence).
/// Part of #589 AC4.
#[test]
fn test_causal_conv1d_tensor_dispatch_dilated() {
    let cache = metal_setup();

    let (in_ch, out_ch, kernel_size, in_len, stride, dilation, groups) = (4, 4, 3, 32, 1, 3, 1);
    let pad_left = (kernel_size - 1) * dilation;
    let eff_kernel = dilation * (kernel_size - 1) + 1;
    let out_len = (in_len + pad_left - eff_kernel) / stride + 1;

    let kernel = nn_dsl::build_causal_conv1d(
        "causal_conv1d_dilated",
        in_ch,
        out_ch,
        kernel_size,
        in_len,
        stride,
        dilation,
        groups,
        false,
    )
    .expect("build causal conv1d dilated");

    let x_data = rand_f32_vec(0xCA03_0001, in_ch * in_len, -1.0, 1.0);
    let w_data = rand_f32_vec(0xCA03_0002, out_ch * in_ch * kernel_size, -0.4, 0.4);

    let cpu_out = causal_conv1d_ref(
        &x_data,
        &w_data,
        None,
        in_ch,
        out_ch,
        kernel_size,
        in_len,
        stride,
        dilation,
    );

    let mut inputs = HashMap::new();
    inputs.insert("data", x_data);
    inputs.insert("weight", w_data);

    let gpu_out = execute_tensor_dispatch(&cache, &kernel, ScalarType::F32, &inputs)
        .expect("causal conv1d dispatch dilated");

    assert_eq!(
        gpu_out.len(),
        out_ch * out_len,
        "causal conv1d dilated output length"
    );
    assert_within_budget("causal_conv1d_dilated", &gpu_out, &cpu_out);
}

/// Causal Conv1d dispatch: with bias.
///
/// Part of #589 AC4.
#[test]
fn test_causal_conv1d_tensor_dispatch_with_bias() {
    let cache = metal_setup();

    let (in_ch, out_ch, kernel_size, in_len, stride, dilation, groups) = (2, 3, 3, 8, 1, 1, 1);
    let pad_left = (kernel_size - 1) * dilation;
    let eff_kernel = dilation * (kernel_size - 1) + 1;
    let out_len = (in_len + pad_left - eff_kernel) / stride + 1;

    let kernel = nn_dsl::build_causal_conv1d(
        "causal_conv1d_bias",
        in_ch,
        out_ch,
        kernel_size,
        in_len,
        stride,
        dilation,
        groups,
        true,
    )
    .expect("build causal conv1d with bias");

    let x_data = rand_f32_vec(0xCA02_0001, in_ch * in_len, -1.0, 1.0);
    let w_data = rand_f32_vec(0xCA02_0002, out_ch * in_ch * kernel_size, -0.3, 0.3);
    let b_data = rand_f32_vec(0xCA02_0003, out_ch, -0.1, 0.1);

    let cpu_out = causal_conv1d_ref(
        &x_data,
        &w_data,
        Some(&b_data),
        in_ch,
        out_ch,
        kernel_size,
        in_len,
        stride,
        dilation,
    );

    let mut inputs = HashMap::new();
    inputs.insert("data", x_data);
    inputs.insert("weight", w_data);
    inputs.insert("bias", b_data);

    let gpu_out = execute_tensor_dispatch(&cache, &kernel, ScalarType::F32, &inputs)
        .expect("causal conv1d dispatch with bias");

    assert_eq!(
        gpu_out.len(),
        out_ch * out_len,
        "causal conv1d+bias output length"
    );
    assert_within_budget("causal_conv1d_bias", &gpu_out, &cpu_out);
}

// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Cross-backend contract tests for Conv2d tensor kernel:
//! GPU output within NY verified bounds.
//!
//! Uses `tensor_kernel_to_graph` + `propagate_ibp` for bounds and
//! `execute_tensor_dispatch` for GPU execution.
//!
//! Part of #779, Part of #793.

use super::test_utils::{assert_gpu_within_bounds, metal_setup, rand_f32_vec};

use nn_dsl::conv2d::{build_conv2d, build_conv2d_full};
use nn_dsl::ScalarType;
use nn_metal::execute_tensor_dispatch;
use nn_verify::{tensor_kernel_to_graph, BoundedTensor, TensorParamBinding};
use ndarray::{ArrayD, IxDyn};
use std::collections::HashMap;

fn constant_tensor(shape: &[usize], data: Vec<f32>) -> TensorParamBinding {
    TensorParamBinding::ConstantTensor(
        ArrayD::from_shape_vec(IxDyn(shape), data).expect("shape/data length mismatch"),
    )
}

// ---------------------------------------------------------------------------
// Shared verification helpers
// ---------------------------------------------------------------------------

/// Prove IBP bounds for a Conv2d tensor kernel, returning (proved_lo, proved_hi).
fn prove_conv2d_bounds(
    def: &nn_dsl::TensorKernelDef,
    bindings: &[TensorParamBinding],
    in_ch: usize,
    in_h: usize,
    in_w: usize,
) -> (ArrayD<f32>, ArrayD<f32>) {
    let graph = tensor_kernel_to_graph(def, bindings).expect("conv2d graph");
    let lower_in = ArrayD::from_elem(IxDyn(&[in_ch, in_h, in_w]), -1.0f32);
    let upper_in = ArrayD::from_elem(IxDyn(&[in_ch, in_h, in_w]), 1.0f32);
    let input_bounds = BoundedTensor::new(lower_in, upper_in).expect("input bounds");
    let output_bounds = graph.propagate_ibp(&input_bounds).expect("IBP propagation");
    let (lo, hi) = output_bounds.lower_upper();
    assert!(
        lo.iter().all(|v| v.is_finite()),
        "proved lower must be finite"
    );
    assert!(
        hi.iter().all(|v| v.is_finite()),
        "proved upper must be finite"
    );
    (lo.clone(), hi.clone())
}

// ===========================================================================
// Conv2d contract tests
// ===========================================================================

/// Conv2d contract: no bias, 3×3 kernel, stride=1, pad=0.
/// Part of #779, Part of #793.
#[test]
fn test_conv2d_gpu_output_within_verified_bounds_no_bias() {
    let (in_ch, out_ch, kh, kw, in_h, in_w, stride_h, stride_w, pad_h, pad_w) =
        (2, 3, 3, 3, 6, 6, 1, 1, 0, 0);
    let out_h = (in_h + 2 * pad_h - kh) / stride_h + 1; // 4
    let out_w = (in_w + 2 * pad_w - kw) / stride_w + 1; // 4

    let def = build_conv2d(
        "conv2d_contract",
        in_ch,
        out_ch,
        kh,
        kw,
        in_h,
        in_w,
        stride_h,
        stride_w,
        pad_h,
        pad_w,
        false,
    )
    .expect("build conv2d");

    let weight_data = rand_f32_vec(0xC2D0_0001, out_ch * in_ch * kh * kw, -0.5, 0.5);
    let bindings = vec![
        TensorParamBinding::Variable,
        constant_tensor(&[out_ch, in_ch, kh, kw], weight_data.clone()),
    ];

    let (proved_lo, proved_hi) = prove_conv2d_bounds(&def, &bindings, in_ch, in_h, in_w);
    assert_eq!(
        proved_lo.shape(),
        &[out_ch, out_h, out_w],
        "output bounds shape"
    );

    let cache = metal_setup();
    let data = rand_f32_vec(0xC2D0_0002, in_ch * in_h * in_w, -1.0, 1.0);
    let mut inputs = HashMap::new();
    inputs.insert("data", data);
    inputs.insert("weight", weight_data);

    let gpu_out = execute_tensor_dispatch(&cache, &def, ScalarType::F32, &inputs)
        .expect("conv2d GPU dispatch");
    assert_eq!(gpu_out.len(), out_ch * out_h * out_w, "output length");

    assert_gpu_within_bounds("conv2d", &gpu_out, &proved_lo, &proved_hi);
}

/// Conv2d contract: with bias, 3×3 kernel.
/// Part of #779, Part of #793.
#[test]
fn test_conv2d_gpu_output_within_verified_bounds_with_bias() {
    let (in_ch, out_ch, kh, kw, in_h, in_w, stride_h, stride_w, pad_h, pad_w) =
        (1, 2, 3, 3, 8, 8, 1, 1, 0, 0);
    let out_h = (in_h + 2 * pad_h - kh) / stride_h + 1; // 6
    let out_w = (in_w + 2 * pad_w - kw) / stride_w + 1; // 6

    let def = build_conv2d(
        "conv2d_contract_bias",
        in_ch,
        out_ch,
        kh,
        kw,
        in_h,
        in_w,
        stride_h,
        stride_w,
        pad_h,
        pad_w,
        true,
    )
    .expect("build conv2d with bias");

    let weight_data = rand_f32_vec(0xB2A5_0001, out_ch * in_ch * kh * kw, -0.3, 0.3);
    let bias_data = rand_f32_vec(0xB2A5_0002, out_ch, -0.1, 0.1);
    let bindings = vec![
        TensorParamBinding::Variable,
        constant_tensor(&[out_ch, in_ch, kh, kw], weight_data.clone()),
        constant_tensor(&[out_ch], bias_data.clone()),
    ];

    let (proved_lo, proved_hi) = prove_conv2d_bounds(&def, &bindings, in_ch, in_h, in_w);
    assert_eq!(proved_lo.shape(), &[out_ch, out_h, out_w]);

    let cache = metal_setup();
    let data = rand_f32_vec(0xB2A5_0003, in_ch * in_h * in_w, -1.0, 1.0);
    let mut inputs = HashMap::new();
    inputs.insert("data", data);
    inputs.insert("weight", weight_data);
    inputs.insert("bias", bias_data);

    let gpu_out = execute_tensor_dispatch(&cache, &def, ScalarType::F32, &inputs)
        .expect("conv2d+bias GPU dispatch");
    assert_eq!(gpu_out.len(), out_ch * out_h * out_w);

    assert_gpu_within_bounds("conv2d+bias", &gpu_out, &proved_lo, &proved_hi);
}

/// Conv2d contract: Demucs spectral decoder pattern (3×3, stride=1, pad=1).
/// Same-padding preserves spatial dimensions.
/// Part of #779, Part of #793.
#[test]
fn test_conv2d_gpu_output_within_verified_bounds_demucs_pattern() {
    let (in_ch, out_ch, kh, kw, in_h, in_w, stride_h, stride_w, pad_h, pad_w) =
        (4, 8, 3, 3, 8, 8, 1, 1, 1, 1);
    let out_h = (in_h + 2 * pad_h - kh) / stride_h + 1; // 8
    let out_w = (in_w + 2 * pad_w - kw) / stride_w + 1; // 8

    let def = build_conv2d(
        "conv2d_contract_demucs",
        in_ch,
        out_ch,
        kh,
        kw,
        in_h,
        in_w,
        stride_h,
        stride_w,
        pad_h,
        pad_w,
        false,
    )
    .expect("build conv2d demucs");

    let weight_data = rand_f32_vec(0xDE50_0001, out_ch * in_ch * kh * kw, -0.2, 0.2);
    let bindings = vec![
        TensorParamBinding::Variable,
        constant_tensor(&[out_ch, in_ch, kh, kw], weight_data.clone()),
    ];

    let (proved_lo, proved_hi) = prove_conv2d_bounds(&def, &bindings, in_ch, in_h, in_w);
    assert_eq!(
        proved_lo.shape(),
        &[out_ch, out_h, out_w],
        "same-padding shape"
    );

    let cache = metal_setup();
    let data = rand_f32_vec(0xDE50_0002, in_ch * in_h * in_w, -1.0, 1.0);
    let mut inputs = HashMap::new();
    inputs.insert("data", data);
    inputs.insert("weight", weight_data);

    let gpu_out = execute_tensor_dispatch(&cache, &def, ScalarType::F32, &inputs)
        .expect("demucs conv2d GPU dispatch");
    assert_eq!(gpu_out.len(), out_ch * out_h * out_w);

    assert_gpu_within_bounds("demucs conv2d", &gpu_out, &proved_lo, &proved_hi);
}

/// Conv2d contract: dilated convolution (dilation=2, kernel=3×3).
/// Verifies Metal dilation bake-in matches NY expand_dilated_kernel_2d.
/// Part of #779, Part of #793.
#[test]
fn test_conv2d_gpu_output_within_verified_bounds_dilated() {
    let (in_ch, out_ch, kh, kw, in_h, in_w, stride_h, stride_w, pad_h, pad_w) =
        (2, 2, 3, 3, 10, 10, 1, 1, 0, 0);
    let (dilation_h, dilation_w, groups) = (2, 2, 1);
    // effective kernel: 2*(3-1)+1 = 5
    let eff_kh = dilation_h * (kh - 1) + 1;
    let eff_kw = dilation_w * (kw - 1) + 1;
    let out_h = (in_h + 2 * pad_h - eff_kh) / stride_h + 1; // 6
    let out_w = (in_w + 2 * pad_w - eff_kw) / stride_w + 1; // 6

    let def = build_conv2d_full(
        "conv2d_contract_dilated",
        in_ch,
        out_ch,
        kh,
        kw,
        in_h,
        in_w,
        stride_h,
        stride_w,
        pad_h,
        pad_w,
        dilation_h,
        dilation_w,
        groups,
        false,
    )
    .expect("build conv2d dilated");

    let weight_data = rand_f32_vec(0xD2A0_0001, out_ch * in_ch * kh * kw, -0.4, 0.4);
    let bindings = vec![
        TensorParamBinding::Variable,
        constant_tensor(&[out_ch, in_ch, kh, kw], weight_data.clone()),
    ];

    let (proved_lo, proved_hi) = prove_conv2d_bounds(&def, &bindings, in_ch, in_h, in_w);
    assert_eq!(
        proved_lo.shape(),
        &[out_ch, out_h, out_w],
        "dilated output shape"
    );

    let cache = metal_setup();
    let data = rand_f32_vec(0xD2A0_0002, in_ch * in_h * in_w, -1.0, 1.0);
    let mut inputs = HashMap::new();
    inputs.insert("data", data);
    inputs.insert("weight", weight_data);

    let gpu_out = execute_tensor_dispatch(&cache, &def, ScalarType::F32, &inputs)
        .expect("dilated conv2d GPU dispatch");
    assert_eq!(
        gpu_out.len(),
        out_ch * out_h * out_w,
        "dilated output length"
    );

    assert_gpu_within_bounds("dilated conv2d", &gpu_out, &proved_lo, &proved_hi);
}

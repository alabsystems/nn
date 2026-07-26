// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Cross-backend contract tests for ConvTranspose1d tensor kernel:
//! GPU output within NY verified bounds.
//!
//! Follows the Conv1d contract test pattern (contract_conv1d.rs):
//! `tensor_kernel_to_graph` + `propagate_ibp` for bounds,
//! `execute_tensor_dispatch` for GPU execution.
//!
//! Part of #635.

use super::test_utils::{assert_gpu_within_bounds, metal_setup, rand_f32_vec};

use nn_dsl::conv_transpose_1d::build_conv_transpose_1d;
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

/// Prove IBP bounds for a ConvTranspose1d tensor kernel.
fn prove_conv_transpose_1d_bounds(
    def: &nn_dsl::TensorKernelDef,
    bindings: &[TensorParamBinding],
    in_ch: usize,
    in_len: usize,
) -> (ArrayD<f32>, ArrayD<f32>) {
    let graph = tensor_kernel_to_graph(def, bindings).expect("conv_transpose_1d graph");
    let lower_in = ArrayD::from_elem(IxDyn(&[in_ch, in_len]), -1.0f32);
    let upper_in = ArrayD::from_elem(IxDyn(&[in_ch, in_len]), 1.0f32);
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
// ConvTranspose1d contract tests
// ===========================================================================

/// ConvTranspose1d contract: no bias, in_ch=2, out_ch=3, kernel=3, stride=2, pad=1.
/// out_len = (4-1)*2 + 3 - 2*1 = 7.
/// Part of #635.
#[test]
fn test_conv_transpose_1d_gpu_output_within_verified_bounds_no_bias() {
    let (in_ch, out_ch, kernel_size, in_len, stride, padding) = (2, 3, 3, 4, 2, 1);
    let out_len = (in_len - 1) * stride + kernel_size - 2 * padding;

    let def = build_conv_transpose_1d(
        "ct1d_contract",
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

    // Weight layout: [in_channels, out_channels, kernel_size]
    let weight_data = rand_f32_vec(0xC71D_0001, in_ch * out_ch * kernel_size, -0.5, 0.5);
    let bindings = vec![
        TensorParamBinding::Variable,
        constant_tensor(&[in_ch, out_ch, kernel_size], weight_data.clone()),
    ];

    let (proved_lo, proved_hi) = prove_conv_transpose_1d_bounds(&def, &bindings, in_ch, in_len);
    assert_eq!(proved_lo.shape(), &[out_ch, out_len], "output bounds shape");

    let cache = metal_setup();
    let data = rand_f32_vec(0xC71D_0002, in_ch * in_len, -1.0, 1.0);
    let mut inputs = HashMap::new();
    inputs.insert("data", data);
    inputs.insert("weight", weight_data);

    let gpu_out = execute_tensor_dispatch(&cache, &def, ScalarType::F32, &inputs)
        .expect("conv_transpose_1d GPU dispatch");
    assert_eq!(gpu_out.len(), out_ch * out_len, "output length");

    assert_gpu_within_bounds("conv_transpose_1d", &gpu_out, &proved_lo, &proved_hi);
}

/// ConvTranspose1d contract: with bias, in_ch=1, out_ch=2, kernel=3, stride=1, pad=0.
/// out_len = (6-1)*1 + 3 - 0 = 8.
/// Part of #635.
#[test]
fn test_conv_transpose_1d_gpu_output_within_verified_bounds_with_bias() {
    let (in_ch, out_ch, kernel_size, in_len, stride, padding) = (1, 2, 3, 6, 1, 0);
    let out_len = (in_len - 1) * stride + kernel_size - 2 * padding;

    let def = build_conv_transpose_1d(
        "ct1d_contract_bias",
        in_ch,
        out_ch,
        kernel_size,
        in_len,
        stride,
        padding,
        1,
        1,
        true,
        0,
    )
    .expect("build conv_transpose_1d with bias");

    let weight_data = rand_f32_vec(0xC71D_0003, in_ch * out_ch * kernel_size, -0.3, 0.3);
    let bias_data = rand_f32_vec(0xC71D_0004, out_ch, -0.1, 0.1);
    let bindings = vec![
        TensorParamBinding::Variable,
        constant_tensor(&[in_ch, out_ch, kernel_size], weight_data.clone()),
        constant_tensor(&[out_ch], bias_data.clone()),
    ];

    let (proved_lo, proved_hi) = prove_conv_transpose_1d_bounds(&def, &bindings, in_ch, in_len);
    assert_eq!(proved_lo.shape(), &[out_ch, out_len]);

    let cache = metal_setup();
    let data = rand_f32_vec(0xC71D_0005, in_ch * in_len, -1.0, 1.0);
    let mut inputs = HashMap::new();
    inputs.insert("data", data);
    inputs.insert("weight", weight_data);
    inputs.insert("bias", bias_data);

    let gpu_out = execute_tensor_dispatch(&cache, &def, ScalarType::F32, &inputs)
        .expect("conv_transpose_1d+bias GPU dispatch");
    assert_eq!(gpu_out.len(), out_ch * out_len);

    assert_gpu_within_bounds("conv_transpose_1d+bias", &gpu_out, &proved_lo, &proved_hi);
}

/// ConvTranspose1d contract: Demucs decoder pattern (stride=4, kernel=8).
/// Config: in_ch=96, out_ch=48, kernel=8, stride=4, pad=0, in_len=16.
/// out_len = (16-1)*4 + 8 - 0 = 68.
/// Part of #635.
#[test]
fn test_conv_transpose_1d_gpu_output_within_verified_bounds_demucs() {
    let (in_ch, out_ch, kernel_size, in_len, stride, padding) = (96, 48, 8, 16, 4, 0);
    let out_len = (in_len - 1) * stride + kernel_size - 2 * padding;

    let def = build_conv_transpose_1d(
        "ct1d_contract_demucs",
        in_ch,
        out_ch,
        kernel_size,
        in_len,
        stride,
        padding,
        1,
        1,
        true,
        0,
    )
    .expect("build conv_transpose_1d demucs");

    let weight_data = rand_f32_vec(0xC71D_0006, in_ch * out_ch * kernel_size, -0.1, 0.1);
    let bias_data = rand_f32_vec(0xC71D_0007, out_ch, -0.05, 0.05);
    let bindings = vec![
        TensorParamBinding::Variable,
        constant_tensor(&[in_ch, out_ch, kernel_size], weight_data.clone()),
        constant_tensor(&[out_ch], bias_data.clone()),
    ];

    let (proved_lo, proved_hi) = prove_conv_transpose_1d_bounds(&def, &bindings, in_ch, in_len);
    assert_eq!(proved_lo.shape(), &[out_ch, out_len]);

    let cache = metal_setup();
    let data = rand_f32_vec(0xC71D_0008, in_ch * in_len, -1.0, 1.0);
    let mut inputs = HashMap::new();
    inputs.insert("data", data);
    inputs.insert("weight", weight_data);
    inputs.insert("bias", bias_data);

    let gpu_out = execute_tensor_dispatch(&cache, &def, ScalarType::F32, &inputs)
        .expect("demucs conv_transpose_1d GPU dispatch");
    assert_eq!(gpu_out.len(), out_ch * out_len);

    assert_gpu_within_bounds("demucs conv_transpose_1d", &gpu_out, &proved_lo, &proved_hi);
}

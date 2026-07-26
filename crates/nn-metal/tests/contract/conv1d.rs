// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Cross-backend contract tests for Conv1d tensor kernel:
//! GPU output within NY verified bounds.
//!
//! Unlike scalar contract tests (contract.rs, contract_norm.rs) which use
//! `VerifyRequest` + `dispatch_elementwise`, Conv1d is a tensor-level op using
//! `tensor_kernel_to_graph` + `propagate_ibp` for bounds and
//! `execute_tensor_dispatch` for GPU execution.
//!
//! Part of #615.

use super::test_utils::{assert_gpu_within_bounds, metal_setup, rand_f32_vec};

use nn_dsl::conv1d::{build_conv1d, build_conv1d_full};
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

/// Prove IBP bounds for a Conv1d tensor kernel, returning (proved_lo, proved_hi).
fn prove_conv1d_bounds(
    def: &nn_dsl::TensorKernelDef,
    bindings: &[TensorParamBinding],
    in_ch: usize,
    in_len: usize,
) -> (ArrayD<f32>, ArrayD<f32>) {
    let graph = tensor_kernel_to_graph(def, bindings).expect("conv1d graph");
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
// Conv1d contract tests
// ===========================================================================

/// Conv1d contract: no bias, in_ch=2, out_ch=3, kernel=2, stride=1, pad=0.
/// Part of #615.
#[test]
fn test_conv1d_gpu_output_within_verified_bounds_no_bias() {
    let (in_ch, out_ch, kernel_size, in_len, stride, padding) = (2, 3, 2, 6, 1, 0);
    let out_len = (in_len + 2 * padding - kernel_size) / stride + 1;

    let def = build_conv1d(
        "conv1d_contract",
        in_ch,
        out_ch,
        kernel_size,
        in_len,
        stride,
        padding,
        false,
    )
    .expect("build conv1d");

    let weight_data = rand_f32_vec(0xC0DE_0001, out_ch * in_ch * kernel_size, -0.5, 0.5);
    let bindings = vec![
        TensorParamBinding::Variable,
        constant_tensor(&[out_ch, in_ch, kernel_size], weight_data.clone()),
    ];

    let (proved_lo, proved_hi) = prove_conv1d_bounds(&def, &bindings, in_ch, in_len);
    assert_eq!(proved_lo.shape(), &[out_ch, out_len], "output bounds shape");

    let cache = metal_setup();
    let data = rand_f32_vec(0xC0DE_0002, in_ch * in_len, -1.0, 1.0);
    let mut inputs = HashMap::new();
    inputs.insert("data", data);
    inputs.insert("weight", weight_data);

    let gpu_out = execute_tensor_dispatch(&cache, &def, ScalarType::F32, &inputs)
        .expect("conv1d GPU dispatch");
    assert_eq!(gpu_out.len(), out_ch * out_len, "output length");

    assert_gpu_within_bounds("conv1d", &gpu_out, &proved_lo, &proved_hi);
}

/// Conv1d contract: with bias, in_ch=1, out_ch=2, kernel=3, stride=1, pad=0.
/// Part of #615.
#[test]
fn test_conv1d_gpu_output_within_verified_bounds_with_bias() {
    let (in_ch, out_ch, kernel_size, in_len, stride, padding) = (1, 2, 3, 8, 1, 0);
    let out_len = (in_len + 2 * padding - kernel_size) / stride + 1;

    let def = build_conv1d(
        "conv1d_contract_bias",
        in_ch,
        out_ch,
        kernel_size,
        in_len,
        stride,
        padding,
        true,
    )
    .expect("build conv1d with bias");

    let weight_data = rand_f32_vec(0xB1A5_0001, out_ch * in_ch * kernel_size, -0.3, 0.3);
    let bias_data = rand_f32_vec(0xB1A5_0002, out_ch, -0.1, 0.1);
    let bindings = vec![
        TensorParamBinding::Variable,
        constant_tensor(&[out_ch, in_ch, kernel_size], weight_data.clone()),
        constant_tensor(&[out_ch], bias_data.clone()),
    ];

    let (proved_lo, proved_hi) = prove_conv1d_bounds(&def, &bindings, in_ch, in_len);
    assert_eq!(proved_lo.shape(), &[out_ch, out_len]);

    let cache = metal_setup();
    let data = rand_f32_vec(0xB1A5_0003, in_ch * in_len, -1.0, 1.0);
    let mut inputs = HashMap::new();
    inputs.insert("data", data);
    inputs.insert("weight", weight_data);
    inputs.insert("bias", bias_data);

    let gpu_out = execute_tensor_dispatch(&cache, &def, ScalarType::F32, &inputs)
        .expect("conv1d+bias GPU dispatch");
    assert_eq!(gpu_out.len(), out_ch * out_len);

    assert_gpu_within_bounds("conv1d+bias", &gpu_out, &proved_lo, &proved_hi);
}

/// Conv1d contract: dvoice encoder pattern (stride + padding).
/// Config: in_ch=1, out_ch=48, kernel=8, stride=4, pad=2, in_len=64.
/// Part of #615.
#[test]
fn test_conv1d_gpu_output_within_verified_bounds_dvoice_pattern() {
    let (in_ch, out_ch, kernel_size, in_len, stride, padding) = (1, 48, 8, 64, 4, 2);
    let out_len = (in_len + 2 * padding - kernel_size) / stride + 1;

    let def = build_conv1d(
        "conv1d_contract_dv",
        in_ch,
        out_ch,
        kernel_size,
        in_len,
        stride,
        padding,
        false,
    )
    .expect("build conv1d dvoice");

    let weight_data = rand_f32_vec(0xDA50_0001, out_ch * in_ch * kernel_size, -0.2, 0.2);
    let bindings = vec![
        TensorParamBinding::Variable,
        constant_tensor(&[out_ch, in_ch, kernel_size], weight_data.clone()),
    ];

    let (proved_lo, proved_hi) = prove_conv1d_bounds(&def, &bindings, in_ch, in_len);
    assert_eq!(proved_lo.shape(), &[out_ch, out_len]);

    let cache = metal_setup();
    let data = rand_f32_vec(0xDA50_0002, in_ch * in_len, -1.0, 1.0);
    let mut inputs = HashMap::new();
    inputs.insert("data", data);
    inputs.insert("weight", weight_data);

    let gpu_out = execute_tensor_dispatch(&cache, &def, ScalarType::F32, &inputs)
        .expect("dvoice conv1d GPU dispatch");
    assert_eq!(gpu_out.len(), out_ch * out_len);

    assert_gpu_within_bounds("dvoice conv1d", &gpu_out, &proved_lo, &proved_hi);
}

/// Conv1d contract: dilated convolution (dilation=2, dvoice DConv pattern).
/// Verifies Metal dilation bake-in matches NY expand_dilated_kernel.
/// Part of #636.
#[test]
fn test_conv1d_gpu_output_within_verified_bounds_dilated() {
    let (in_ch, out_ch, kernel_size, in_len, stride, padding, dilation, groups) =
        (4, 4, 3, 32, 1, 0, 2, 1);
    // effective_kernel = dilation * (kernel_size - 1) + 1 = 2*(3-1)+1 = 5
    let effective_kernel = dilation * (kernel_size - 1) + 1;
    let out_len = (in_len + 2 * padding - effective_kernel) / stride + 1;

    let def = build_conv1d_full(
        "conv1d_contract_dilated",
        in_ch,
        out_ch,
        kernel_size,
        in_len,
        stride,
        padding,
        dilation,
        groups,
        false,
    )
    .expect("build conv1d dilated");

    let weight_data = rand_f32_vec(0xD11A_0001, out_ch * in_ch * kernel_size, -0.4, 0.4);
    let bindings = vec![
        TensorParamBinding::Variable,
        constant_tensor(&[out_ch, in_ch, kernel_size], weight_data.clone()),
    ];

    let (proved_lo, proved_hi) = prove_conv1d_bounds(&def, &bindings, in_ch, in_len);
    assert_eq!(
        proved_lo.shape(),
        &[out_ch, out_len],
        "dilated output shape"
    );

    let cache = metal_setup();
    let data = rand_f32_vec(0xD11A_0002, in_ch * in_len, -1.0, 1.0);
    let mut inputs = HashMap::new();
    inputs.insert("data", data);
    inputs.insert("weight", weight_data);

    let gpu_out = execute_tensor_dispatch(&cache, &def, ScalarType::F32, &inputs)
        .expect("dilated conv1d GPU dispatch");
    assert_eq!(gpu_out.len(), out_ch * out_len, "dilated output length");

    assert_gpu_within_bounds("dilated conv1d", &gpu_out, &proved_lo, &proved_hi);
}

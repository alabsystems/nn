// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Cross-backend contract tests for causal Conv1d tensor kernel:
//! GPU output within NY verified bounds.
//!
//! Uses `build_causal_conv1d` (ZeroPad1d + Conv1d(padding=0) decomposition)
//! and verifies GPU output falls within IBP-propagated bounds from NY.
//!
//! The graph translation converts ZeroPad1d into a LinearLayer that maps
//! `[in_ch, in_len]` to `[in_ch, padded_len]` with zeros in the pad regions.
//! The graph's NETWORK_INPUT shape is `[in_ch, in_len]` (unpadded), so input
//! bounds must match that shape. The LinearLayer handles padding internally.
//!
//! Part of #589 AC5.

use super::test_utils::{assert_gpu_within_bounds, metal_setup, rand_f32_vec};

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

/// Prove IBP bounds for a causal Conv1d tensor kernel.
///
/// Input bounds have shape `[in_ch, in_len]` (unpadded). The graph handles
/// ZeroPad1d internally via a LinearLayer that maps `[in_ch, in_len]` to
/// `[in_ch, padded_len]` with zeros in the pad region.
fn prove_causal_conv1d_bounds(
    def: &nn_dsl::TensorKernelDef,
    bindings: &[TensorParamBinding],
    in_ch: usize,
    in_len: usize,
) -> (ArrayD<f32>, ArrayD<f32>) {
    let graph = tensor_kernel_to_graph(def, bindings).expect("causal conv1d graph");

    // Input bounds at unpadded shape — the ZeroPad1d LinearLayer handles padding.
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
// Causal Conv1d contract tests
// ===========================================================================

/// Causal Conv1d contract: kernel=3, dilation=1 (dvoice ResBlock base layer).
/// Part of #589 AC5.
#[test]
fn test_causal_conv1d_gpu_within_bounds_k3_d1() {
    let (in_ch, out_ch, kernel_size, in_len, stride, dilation, groups) = (2, 3, 3, 8, 1, 1, 1);
    let pad_left = (kernel_size - 1) * dilation;
    let eff_kernel = dilation * (kernel_size - 1) + 1;
    let out_len = (in_len + pad_left - eff_kernel) / stride + 1;

    let def = nn_dsl::build_causal_conv1d(
        "causal_contract_k3d1",
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

    let weight_data = rand_f32_vec(0xCC01_0001, out_ch * in_ch * kernel_size, -0.5, 0.5);
    let bindings = vec![
        TensorParamBinding::Variable,
        constant_tensor(&[out_ch, in_ch, kernel_size], weight_data.clone()),
    ];

    let (proved_lo, proved_hi) = prove_causal_conv1d_bounds(&def, &bindings, in_ch, in_len);
    assert_eq!(proved_lo.shape(), &[out_ch, out_len], "output bounds shape");

    let cache = metal_setup();
    let data = rand_f32_vec(0xCC01_0002, in_ch * in_len, -1.0, 1.0);
    let mut inputs = HashMap::new();
    inputs.insert("data", data);
    inputs.insert("weight", weight_data);

    let gpu_out = execute_tensor_dispatch(&cache, &def, ScalarType::F32, &inputs)
        .expect("causal conv1d GPU dispatch k3d1");
    assert_eq!(gpu_out.len(), out_ch * out_len, "output length");

    assert_gpu_within_bounds("causal_conv1d_k3d1", &gpu_out, &proved_lo, &proved_hi);
}

/// Causal Conv1d contract: kernel=3, dilation=3 (dvoice ResBlock dilated layer).
/// Part of #589 AC5.
#[test]
fn test_causal_conv1d_gpu_within_bounds_k3_d3() {
    let (in_ch, out_ch, kernel_size, in_len, stride, dilation, groups) = (4, 4, 3, 32, 1, 3, 1);
    let pad_left = (kernel_size - 1) * dilation;
    let eff_kernel = dilation * (kernel_size - 1) + 1;
    let out_len = (in_len + pad_left - eff_kernel) / stride + 1;

    let def = nn_dsl::build_causal_conv1d(
        "causal_contract_k3d3",
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

    let weight_data = rand_f32_vec(0xCC03_0001, out_ch * in_ch * kernel_size, -0.4, 0.4);
    let bindings = vec![
        TensorParamBinding::Variable,
        constant_tensor(&[out_ch, in_ch, kernel_size], weight_data.clone()),
    ];

    let (proved_lo, proved_hi) = prove_causal_conv1d_bounds(&def, &bindings, in_ch, in_len);
    assert_eq!(
        proved_lo.shape(),
        &[out_ch, out_len],
        "dilated bounds shape"
    );

    let cache = metal_setup();
    let data = rand_f32_vec(0xCC03_0002, in_ch * in_len, -1.0, 1.0);
    let mut inputs = HashMap::new();
    inputs.insert("data", data);
    inputs.insert("weight", weight_data);

    let gpu_out = execute_tensor_dispatch(&cache, &def, ScalarType::F32, &inputs)
        .expect("causal conv1d GPU dispatch k3d3");
    assert_eq!(gpu_out.len(), out_ch * out_len, "dilated output length");

    assert_gpu_within_bounds("causal_conv1d_k3d3", &gpu_out, &proved_lo, &proved_hi);
}

/// Causal Conv1d contract: kernel=3, dilation=5 (dvoice ResBlock third dilation).
/// Part of #589 AC5.
#[test]
fn test_causal_conv1d_gpu_within_bounds_k3_d5() {
    let (in_ch, out_ch, kernel_size, in_len, stride, dilation, groups) = (4, 4, 3, 32, 1, 5, 1);
    let pad_left = (kernel_size - 1) * dilation;
    let eff_kernel = dilation * (kernel_size - 1) + 1;
    let out_len = (in_len + pad_left - eff_kernel) / stride + 1;

    let def = nn_dsl::build_causal_conv1d(
        "causal_contract_k3d5",
        in_ch,
        out_ch,
        kernel_size,
        in_len,
        stride,
        dilation,
        groups,
        false,
    )
    .expect("build causal conv1d dilation=5");

    let weight_data = rand_f32_vec(0xCC05_0001, out_ch * in_ch * kernel_size, -0.3, 0.3);
    let bindings = vec![
        TensorParamBinding::Variable,
        constant_tensor(&[out_ch, in_ch, kernel_size], weight_data.clone()),
    ];

    let (proved_lo, proved_hi) = prove_causal_conv1d_bounds(&def, &bindings, in_ch, in_len);
    assert_eq!(
        proved_lo.shape(),
        &[out_ch, out_len],
        "dilation=5 bounds shape"
    );

    let cache = metal_setup();
    let data = rand_f32_vec(0xCC05_0002, in_ch * in_len, -1.0, 1.0);
    let mut inputs = HashMap::new();
    inputs.insert("data", data);
    inputs.insert("weight", weight_data);

    let gpu_out = execute_tensor_dispatch(&cache, &def, ScalarType::F32, &inputs)
        .expect("causal conv1d GPU dispatch k3d5");
    assert_eq!(gpu_out.len(), out_ch * out_len, "dilation=5 output length");

    assert_gpu_within_bounds("causal_conv1d_k3d5", &gpu_out, &proved_lo, &proved_hi);
}

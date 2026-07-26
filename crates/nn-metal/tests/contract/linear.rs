// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Cross-backend contract tests for Linear (fully-connected) tensor kernel:
//! GPU output within NY verified bounds.
//!
//! Tests the full pipeline: IR → dispatch plan → MSL codegen → Metal execution,
//! verified against NY IBP bounds from `tensor_kernel_to_graph`.
//!
//! Part of #730 (Direction 4+5).

use super::test_utils::{assert_gpu_within_bounds, metal_setup, rand_f32_vec};

use nn_dsl::linear::{build_linear, build_linear_batched};
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

/// Prove IBP bounds for a Linear tensor kernel, returning (proved_lo, proved_hi).
fn prove_linear_bounds(
    def: &nn_dsl::TensorKernelDef,
    bindings: &[TensorParamBinding],
    input_shape: &[usize],
) -> (ArrayD<f32>, ArrayD<f32>) {
    let graph = tensor_kernel_to_graph(def, bindings).expect("linear graph");
    let lower_in = ArrayD::from_elem(IxDyn(input_shape), -1.0f32);
    let upper_in = ArrayD::from_elem(IxDyn(input_shape), 1.0f32);
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
// Linear contract tests
// ===========================================================================

/// Linear contract: no bias, in_features=4, out_features=3.
/// Part of #730.
#[test]
fn test_linear_gpu_output_within_verified_bounds_no_bias() {
    let (in_features, out_features) = (4, 3);

    let def =
        build_linear("linear_contract", in_features, out_features, false).expect("build linear");

    let weight_data = rand_f32_vec(0x11AE_0001, out_features * in_features, -0.5, 0.5);
    let bindings = vec![
        TensorParamBinding::Variable,
        constant_tensor(&[out_features, in_features], weight_data.clone()),
    ];

    let (proved_lo, proved_hi) = prove_linear_bounds(&def, &bindings, &[in_features]);
    assert_eq!(proved_lo.shape(), &[out_features], "output bounds shape");

    let cache = metal_setup();
    let data = rand_f32_vec(0x11AE_0002, in_features, -1.0, 1.0);
    let mut inputs = HashMap::new();
    inputs.insert("data", data);
    inputs.insert("weight", weight_data);

    let gpu_out = execute_tensor_dispatch(&cache, &def, ScalarType::F32, &inputs)
        .expect("linear GPU dispatch");
    assert_eq!(gpu_out.len(), out_features, "output length");

    assert_gpu_within_bounds("linear", &gpu_out, &proved_lo, &proved_hi);
}

/// Linear contract: with bias, in_features=4, out_features=3.
/// Part of #730.
#[test]
fn test_linear_gpu_output_within_verified_bounds_with_bias() {
    let (in_features, out_features) = (4, 3);

    let def = build_linear("linear_contract_bias", in_features, out_features, true)
        .expect("build linear with bias");

    let weight_data = rand_f32_vec(0xB1A5_1001, out_features * in_features, -0.3, 0.3);
    let bias_data = rand_f32_vec(0xB1A5_1002, out_features, -0.1, 0.1);
    let bindings = vec![
        TensorParamBinding::Variable,
        constant_tensor(&[out_features, in_features], weight_data.clone()),
        constant_tensor(&[out_features], bias_data.clone()),
    ];

    let (proved_lo, proved_hi) = prove_linear_bounds(&def, &bindings, &[in_features]);
    assert_eq!(proved_lo.shape(), &[out_features]);

    let cache = metal_setup();
    let data = rand_f32_vec(0xB1A5_1003, in_features, -1.0, 1.0);
    let mut inputs = HashMap::new();
    inputs.insert("data", data);
    inputs.insert("weight", weight_data);
    inputs.insert("bias", bias_data);

    let gpu_out = execute_tensor_dispatch(&cache, &def, ScalarType::F32, &inputs)
        .expect("linear+bias GPU dispatch");
    assert_eq!(gpu_out.len(), out_features);

    assert_gpu_within_bounds("linear+bias", &gpu_out, &proved_lo, &proved_hi);
}

/// Linear contract: dvoice-representative dimensions (768 → 3072, Qwen3 MLP).
/// Part of #730.
#[test]
fn test_linear_gpu_output_within_verified_bounds_dvoice_dims() {
    let (in_features, out_features) = (64, 128);
    let batch_size = 4;

    let def = build_linear_batched(
        "linear_contract_dv",
        batch_size,
        in_features,
        out_features,
        false,
    )
    .expect("build batched linear");

    let weight_data = rand_f32_vec(0xDA50_1001, out_features * in_features, -0.2, 0.2);
    let bindings = vec![
        TensorParamBinding::Variable,
        constant_tensor(&[out_features, in_features], weight_data.clone()),
    ];

    let (proved_lo, proved_hi) = prove_linear_bounds(&def, &bindings, &[batch_size, in_features]);
    assert_eq!(proved_lo.shape(), &[batch_size, out_features]);

    let cache = metal_setup();
    let data = rand_f32_vec(0xDA50_1002, batch_size * in_features, -1.0, 1.0);
    let mut inputs = HashMap::new();
    inputs.insert("data", data);
    inputs.insert("weight", weight_data);

    let gpu_out = execute_tensor_dispatch(&cache, &def, ScalarType::F32, &inputs)
        .expect("dvoice linear GPU dispatch");
    assert_eq!(gpu_out.len(), batch_size * out_features);

    assert_gpu_within_bounds("dvoice linear", &gpu_out, &proved_lo, &proved_hi);
}

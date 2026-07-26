// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Cross-backend contract tests for the spectral decoder rewrite sub-def:
//! GPU output of a BinaryAdd → Reshape → Conv2d(3×3) → Reshape → GLU chain
//! within NY composed IBP bounds.
//!
//! The rewrite sub-def is the core of the Demucs spectral decoder:
//! `skip_add → Conv2d(3×3, s=1, p=1) → GLU` operating on `[C, F*T]` flattened
//! tensors with internal reshape to `[C, F, T]` for the Conv2d.
//!
//! This tests the composed operation on Metal GPU and verifies output falls
//! within NY proved bounds.
//!
//! Uses single-variable mode: data = Variable, skip = ConstantTensor(zeros).
//! Multi-variable stacking hits a shape mismatch with the Reshape ops in the
//! graph (the stacking dimension interacts with the reshape from 2D→3D).
//! Same approach as `compose_demucs_decoder_block.rs`.
//!
//! Part of #779 Phase B.

use super::test_utils::{assert_gpu_within_bounds, metal_setup, rand_f32_vec};

use nn_dsl::tensor_block_builder::TensorBlockBuilder;
use nn_dsl::tensor_ir::TensorKernelDef;
use nn_dsl::ScalarType;
use nn_metal::execute_tensor_dispatch;
use nn_verify::{tensor_kernel_to_graph, BoundedTensor, TensorParamBinding};
use ndarray::{ArrayD, IxDyn};
use std::collections::HashMap;

// ---------------------------------------------------------------------------
// Rewrite sub-def builder
// ---------------------------------------------------------------------------

/// Build the spectral decoder rewrite sub-def:
/// BinaryAdd(data, skip) → Reshape[C, F, T] → Conv2d(3×3, s=1, p=1) → Reshape[2C, F*T] → GLU.
///
/// Inputs: "data" [C, F*T], "skip" [C, F*T], "rw_weight" [2C, C, 3, 3], "rw_bias" [2C].
/// Output: [C, F*T].
fn build_spectral_rewrite(channels: usize, freq: usize, time: usize) -> TensorKernelDef {
    let doubled = channels * 2;
    let ft = freq * time;
    let k = 3;
    let pad = 1;

    // Conv2d(3×3, s=1, p=1) preserves spatial dims.
    let out_f = freq + 2 * pad - k + 1;
    let out_t = time + 2 * pad - k + 1;
    let out_ft = out_f * out_t;

    let mut b = TensorBlockBuilder::new("spectral_rewrite_contract");

    let data = b.add_input("data", &[channels, ft]);
    let skip = b.add_input("skip", &[channels, ft]);
    let rw_weight = b.add_input("rw_weight", &[doubled, channels, k, k]);
    let rw_bias = b.add_input("rw_bias", &[doubled]);

    // Skip add: [C, F*T].
    let x = b.add_binary_add(data, skip, &[channels, ft]);

    // Reshape to [C, F, T] for Conv2d.
    let x_3d = b.add_reshape(x, &[channels, freq, time]);

    // Conv2d(C → 2C, k=3×3, s=1, p=1): preserves F and T.
    let conv_out = b.add_conv2d(
        x_3d,
        rw_weight,
        Some(rw_bias),
        1,
        1,
        pad,
        pad,
        &[doubled, out_f, out_t],
    );

    // Reshape to [2C, F*T] for GLU.
    let conv_flat = b.add_reshape(conv_out, &[doubled, out_ft]);

    // GLU: [2C, F*T] → [C, F*T].
    let glu_out = b
        .add_glu(conv_flat, 0, &[doubled, out_ft])
        .expect("even dim");

    b.build(glu_out).expect("valid graph")
}

// ---------------------------------------------------------------------------
// Verification helpers
// ---------------------------------------------------------------------------

/// Prove IBP bounds for the rewrite sub-def with single-variable mode.
///
/// data = Variable, skip = ConstantTensor(zeros).
/// Input bounds: [C, F*T] for data.
/// Returns (proved_lower, proved_upper) arrays over output shape [C, F*T].
fn prove_rewrite_bounds(
    def: &TensorKernelDef,
    bindings: &[TensorParamBinding],
    channels: usize,
    ft: usize,
    input_lo: f32,
    input_hi: f32,
) -> (ArrayD<f32>, ArrayD<f32>) {
    let graph = tensor_kernel_to_graph(def, bindings).expect("rewrite graph must build");

    // Single-variable input bounds: [C, F*T].
    let lower_in = ArrayD::from_elem(IxDyn(&[channels, ft]), input_lo);
    let upper_in = ArrayD::from_elem(IxDyn(&[channels, ft]), input_hi);
    let input_bounds = BoundedTensor::new(lower_in, upper_in).expect("valid input bounds");

    let output_bounds = graph
        .propagate_ibp(&input_bounds)
        .expect("IBP through spectral rewrite");
    let (lo, hi) = output_bounds.lower_upper();

    assert!(
        lo.iter().all(|v| v.is_finite()),
        "proved lower must be finite"
    );
    assert!(
        hi.iter().all(|v| v.is_finite()),
        "proved upper must be finite"
    );
    for (l, u) in lo.iter().zip(hi.iter()) {
        assert!(l <= u, "lower {l} must be <= upper {u}");
    }

    (lo.clone(), hi.clone())
}

// ===========================================================================
// Spectral decoder rewrite GPU contract tests
// ===========================================================================

/// Small rewrite sub-def: BinaryAdd(2, 4*4) → Conv2d(2→4, 3×3, p=1) → GLU.
/// Exercises the full reshape + Conv2d + GLU chain at minimal dimensions.
/// Part of #779.
#[test]
fn test_spectral_rewrite_gpu_within_bounds_small() {
    let (channels, freq, time) = (2, 4, 4);
    let ft = freq * time;
    let doubled = channels * 2;

    let def = build_spectral_rewrite(channels, freq, time);

    let weight_data = rand_f32_vec(0x5DE0_0001, doubled * channels * 3 * 3, -0.3, 0.3);
    let bias_data = rand_f32_vec(0x5DE0_0002, doubled, -0.1, 0.1);
    let skip_data = ArrayD::from_elem(IxDyn(&[channels, ft]), 0.0f32);
    let bindings = vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(skip_data),
        TensorParamBinding::ConstantTensor(
            ArrayD::from_shape_vec(IxDyn(&[doubled, channels, 3, 3]), weight_data.clone())
                .expect("weight shape"),
        ),
        TensorParamBinding::ConstantTensor(
            ArrayD::from_shape_vec(IxDyn(&[doubled]), bias_data.clone()).expect("bias shape"),
        ),
    ];

    let (proved_lo, proved_hi) = prove_rewrite_bounds(&def, &bindings, channels, ft, -1.0, 1.0);
    assert_eq!(proved_lo.shape(), &[channels, ft], "output bounds shape");

    // Run on Metal GPU with random data and zero skip.
    let cache = metal_setup();
    let data = rand_f32_vec(0x5DE0_0003, channels * ft, -1.0, 1.0);
    let skip = vec![0.0f32; channels * ft];
    let mut inputs = HashMap::new();
    inputs.insert("data", data);
    inputs.insert("skip", skip);
    inputs.insert("rw_weight", weight_data);
    inputs.insert("rw_bias", bias_data);

    let gpu_out = execute_tensor_dispatch(&cache, &def, ScalarType::F32, &inputs)
        .expect("spectral rewrite GPU dispatch");
    assert_eq!(gpu_out.len(), channels * ft, "output length");

    assert_gpu_within_bounds("spectral_rewrite_small", &gpu_out, &proved_lo, &proved_hi);
}

/// Demucs-scale rewrite: Conv2d(48→96, 3×3, p=1) on [48, 4, 8] spatial.
/// Tests at production channel counts where precision accumulation matters.
/// Part of #779.
#[test]
fn test_spectral_rewrite_gpu_within_bounds_demucs_scale() {
    let (channels, freq, time) = (48, 4, 8);
    let ft = freq * time;
    let doubled = channels * 2;

    let def = build_spectral_rewrite(channels, freq, time);

    // Small weights to prevent IBP blow-up at Demucs scale.
    let weight_data = rand_f32_vec(0x5DA5_0001, doubled * channels * 3 * 3, -0.03, 0.03);
    let bias_data = rand_f32_vec(0x5DA5_0002, doubled, -0.01, 0.01);
    let skip_data = ArrayD::from_elem(IxDyn(&[channels, ft]), 0.0f32);
    let bindings = vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(skip_data),
        TensorParamBinding::ConstantTensor(
            ArrayD::from_shape_vec(IxDyn(&[doubled, channels, 3, 3]), weight_data.clone())
                .expect("weight shape"),
        ),
        TensorParamBinding::ConstantTensor(
            ArrayD::from_shape_vec(IxDyn(&[doubled]), bias_data.clone()).expect("bias shape"),
        ),
    ];

    let (proved_lo, proved_hi) = prove_rewrite_bounds(&def, &bindings, channels, ft, -1.0, 1.0);
    assert_eq!(proved_lo.shape(), &[channels, ft]);

    // Vacuous widening guard.
    let max_width = proved_lo
        .iter()
        .zip(proved_hi.iter())
        .map(|(l, u)| u - l)
        .fold(0.0f32, f32::max);
    assert!(
        max_width < 200.0,
        "IBP width {max_width} exceeds threshold — possible vacuous widening"
    );

    let cache = metal_setup();
    let data = rand_f32_vec(0x5DA5_0003, channels * ft, -1.0, 1.0);
    let skip = vec![0.0f32; channels * ft];
    let mut inputs = HashMap::new();
    inputs.insert("data", data);
    inputs.insert("skip", skip);
    inputs.insert("rw_weight", weight_data);
    inputs.insert("rw_bias", bias_data);

    let gpu_out = execute_tensor_dispatch(&cache, &def, ScalarType::F32, &inputs)
        .expect("demucs-scale spectral rewrite GPU dispatch");
    assert_eq!(gpu_out.len(), channels * ft);

    assert_gpu_within_bounds("spectral_rewrite_demucs", &gpu_out, &proved_lo, &proved_hi);
}

/// Non-square spatial: Conv2d(4→8, 3×3, p=1) on [4, 6, 3] (freq > time).
/// Exercises asymmetric reshape dimensions that differ from square test cases.
/// Part of #779.
#[test]
fn test_spectral_rewrite_gpu_within_bounds_non_square() {
    let (channels, freq, time) = (4, 6, 3);
    let ft = freq * time;
    let doubled = channels * 2;

    let def = build_spectral_rewrite(channels, freq, time);

    let weight_data = rand_f32_vec(0x5D60_0001, doubled * channels * 3 * 3, -0.3, 0.3);
    let bias_data = rand_f32_vec(0x5D60_0002, doubled, -0.1, 0.1);
    let skip_data = ArrayD::from_elem(IxDyn(&[channels, ft]), 0.0f32);
    let bindings = vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(skip_data),
        TensorParamBinding::ConstantTensor(
            ArrayD::from_shape_vec(IxDyn(&[doubled, channels, 3, 3]), weight_data.clone())
                .expect("weight shape"),
        ),
        TensorParamBinding::ConstantTensor(
            ArrayD::from_shape_vec(IxDyn(&[doubled]), bias_data.clone()).expect("bias shape"),
        ),
    ];

    let (proved_lo, proved_hi) = prove_rewrite_bounds(&def, &bindings, channels, ft, -1.0, 1.0);
    assert_eq!(proved_lo.shape(), &[channels, ft]);

    let cache = metal_setup();
    let data = rand_f32_vec(0x5D60_0003, channels * ft, -1.0, 1.0);
    let skip = vec![0.0f32; channels * ft];
    let mut inputs = HashMap::new();
    inputs.insert("data", data);
    inputs.insert("skip", skip);
    inputs.insert("rw_weight", weight_data);
    inputs.insert("rw_bias", bias_data);

    let gpu_out = execute_tensor_dispatch(&cache, &def, ScalarType::F32, &inputs)
        .expect("non-square spectral rewrite GPU dispatch");
    assert_eq!(gpu_out.len(), channels * ft);

    assert_gpu_within_bounds(
        "spectral_rewrite_nonsquare",
        &gpu_out,
        &proved_lo,
        &proved_hi,
    );
}

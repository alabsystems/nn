// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Integration tests: ConvTranspose1d decoder composition for dvoice Demucs.
//!
//! Validates that a multi-op `TensorKernelDef` representing a Demucs temporal
//! decoder fragment with ConvTranspose1d (upsampling) chains through
//! `tensor_kernel_to_graph` and produces a NY `GraphNetwork` where
//! IBP and CROWN bounds propagate end-to-end.
//!
//! Decoder pattern tested:
//!   BinaryAdd(skip) → ConvTranspose1d(upsample) → Narrow(center_trim) → GLU → GELU
//!
//! Part of #695. Extends `compose_decoder_chain.rs` which uses Conv1d (not ConvTranspose1d).

use super::common::{assert_bounds_valid, assert_crown_tighter_than_ibp, conv_transpose_out_len};
use nn_dsl::tensor_block_builder::TensorBlockBuilder;
use nn_verify::{
    propagate_with_crown_fallback, tensor_kernel_to_graph, BoundedTensor, PropMethod,
    TensorParamBinding,
};
use ndarray::{ArrayD, IxDyn};

// ---------------------------------------------------------------------------
// Builder helpers
// ---------------------------------------------------------------------------

/// Build a Demucs decoder fragment using ConvTranspose1d for upsampling.
///
/// Shape flow (C_in channels, C_out = C_in/2, T_in time steps):
///   data:[C_in, T_in] + skip:[C_in, T_in] → BinaryAdd → [C_in, T_in]
///   ConvTranspose1d(C_in → 2*C_out, k, stride, pad) → [2*C_out, T_up]
///   Narrow(center_trim, axis=1, trim each side) → [2*C_out, T_trimmed]
///   GLU(axis=0): Narrow[0..C_out] * Sigmoid(Narrow[C_out..2*C_out]) → [C_out, T_trimmed]
///   GELU → [C_out, T_trimmed]
///
/// `trim` is the number of time steps to remove from each side after ConvTranspose1d.
/// This models the center_trim pattern used in Demucs to remove boundary artifacts.
fn build_conv_transpose_decoder(
    c_in: usize,
    c_out: usize,
    t_in: usize,
    kernel_size: usize,
    stride: usize,
    padding: usize,
    trim: usize,
) -> nn_dsl::tensor_ir::TensorKernelDef {
    let doubled_out = c_out * 2;
    let t_up = conv_transpose_out_len(t_in, stride, kernel_size, padding);
    let t_trimmed = t_up - 2 * trim;

    let mut b = TensorBlockBuilder::new("demucs_decoder_conv_transpose");

    // Inputs: data (variable), skip (variable), conv_transpose weight (constant)
    let data = b.add_input("data", &[c_in, t_in]);
    let skip = b.add_input("skip", &[c_in, t_in]);
    let weight = b.add_input("weight", &[c_in, doubled_out, kernel_size]);

    // Step 1: Skip connection add
    let added = b.add_binary_add(data, skip, &[c_in, t_in]);

    // Step 2: ConvTranspose1d (upsample: C_in → 2*C_out)
    let deconv = b.add_conv_transpose_1d(
        added,
        weight,
        None,
        stride,
        padding,
        1,
        1,
        0, // output_padding
        &[doubled_out, t_up],
    );

    // Step 3: Center trim (Narrow on time axis to remove boundary artifacts)
    let trimmed = b.add_narrow(deconv, 1, trim, t_trimmed, &[doubled_out, t_trimmed]);

    // Step 4: GLU decomposition (Narrow + Sigmoid + BinaryMul on channel axis)
    let glu_out = b
        .add_glu(trimmed, 0, &[doubled_out, t_trimmed])
        .expect("even dim");

    // Step 5: GELU activation
    let gelu_out = b.add_gelu(glu_out, &[c_out, t_trimmed]);

    b.build(gelu_out).expect("valid graph")
}

/// Create BoundedTensor input for 2 variable inputs of shape [C, T].
/// Multi-variable inputs are stacked along axis 0: shape = [2, C, T].
fn multi_var_input(channels: usize, length: usize, lo: f32, hi: f32) -> BoundedTensor {
    let lower = ArrayD::from_elem(IxDyn(&[2, channels, length]), lo);
    let upper = ArrayD::from_elem(IxDyn(&[2, channels, length]), hi);
    BoundedTensor::new(lower, upper).expect("valid bounds")
}

// ---------------------------------------------------------------------------
// AC1: Decoder composition using ConvTranspose1d
// ---------------------------------------------------------------------------

/// ConvTranspose1d decoder fragment builds a valid graph.
#[test]
fn test_conv_transpose_decoder_graph_builds() {
    // Small test: C_in=4, C_out=2, T=8, k=3, stride=2, pad=0, trim=1
    // ConvTranspose1d output: (8-1)*2 + 3 - 0 = 17
    // After trim=1: 17 - 2 = 15
    let c_in = 4;
    let c_out = 2;
    let t_in = 8;
    let def = build_conv_transpose_decoder(c_in, c_out, t_in, 3, 2, 0, 1);

    let weight = ArrayD::from_elem(IxDyn(&[c_in, c_out * 2, 3]), 0.1f32);
    let bindings = vec![
        TensorParamBinding::Variable,
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(weight),
    ];

    let graph =
        tensor_kernel_to_graph(&def, &bindings).expect("conv_transpose decoder graph must build");
    assert!(
        graph.num_nodes() >= 5,
        "graph should have nodes for add, conv_transpose, narrow, glu components, gelu"
    );
}

// ---------------------------------------------------------------------------
// AC2: Center_trim pattern (ConvTranspose1d → Narrow) tested
// ---------------------------------------------------------------------------

/// ConvTranspose1d → Narrow (center_trim) produces correct output dimensions.
#[test]
fn test_conv_transpose_center_trim_ibp() {
    // Demucs-like: C_in=8, C_out=4, T=16, k=8, stride=4, pad=2, trim=2
    // ConvTranspose1d: (16-1)*4 + 8 - 4 = 64
    // After trim=2: 64 - 4 = 60
    // GLU: [8, 60] → [4, 60]
    // GELU: [4, 60]
    let c_in = 8;
    let c_out = 4;
    let t_in = 16;
    let def = build_conv_transpose_decoder(c_in, c_out, t_in, 8, 4, 2, 2);

    let weight = ArrayD::from_elem(IxDyn(&[c_in, c_out * 2, 8]), 0.05f32);
    let bindings = vec![
        TensorParamBinding::Variable,
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(weight),
    ];

    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");
    let input = multi_var_input(c_in, t_in, -1.0, 1.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through conv_transpose + center_trim");
    assert_bounds_valid(&output);
}

/// Center_trim with zero trim (no cropping) still works.
#[test]
fn test_conv_transpose_no_trim_ibp() {
    // No trim: ConvTranspose1d output used directly by GLU
    let c_in = 4;
    let c_out = 2;
    let t_in = 8;
    // ConvTranspose1d: (8-1)*2 + 4 - 2 = 16
    // No trim → GLU on [4, 16] → [2, 16]
    let def = build_conv_transpose_decoder(c_in, c_out, t_in, 4, 2, 1, 0);

    let weight = ArrayD::from_elem(IxDyn(&[c_in, c_out * 2, 4]), 0.1f32);
    let bindings = vec![
        TensorParamBinding::Variable,
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(weight),
    ];

    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");
    let input = multi_var_input(c_in, t_in, -0.5, 0.5);

    let output = graph.propagate_ibp(&input).expect("IBP no trim");
    assert_bounds_valid(&output);
}

// ---------------------------------------------------------------------------
// AC3: IBP + CROWN bounds propagation
// ---------------------------------------------------------------------------

/// IBP bounds propagate through full ConvTranspose1d decoder chain.
#[test]
fn test_conv_transpose_decoder_ibp_propagates() {
    let c_in = 8;
    let c_out = 4;
    let t_in = 16;
    let def = build_conv_transpose_decoder(c_in, c_out, t_in, 8, 4, 2, 2);

    let weight = ArrayD::from_elem(IxDyn(&[c_in, c_out * 2, 8]), 0.05f32);
    let bindings = vec![
        TensorParamBinding::Variable,
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(weight),
    ];

    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");
    let input = multi_var_input(c_in, t_in, -1.0, 1.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through conv_transpose decoder");
    assert_bounds_valid(&output);

    // Vacuous widening guard
    let (lo, hi) = output.lower_upper();
    let max_width = lo
        .iter()
        .zip(hi.iter())
        .map(|(l, u)| u - l)
        .fold(0.0f32, f32::max);
    assert!(
        max_width < 100.0,
        "IBP width {max_width} exceeds threshold — possible vacuous widening"
    );
}

/// CROWN propagation through ConvTranspose1d decoder chain.
#[test]
fn test_conv_transpose_decoder_crown_propagates() {
    let c_in = 4;
    let c_out = 2;
    let t_in = 8;
    let def = build_conv_transpose_decoder(c_in, c_out, t_in, 4, 2, 1, 1);

    let weight = ArrayD::from_elem(IxDyn(&[c_in, c_out * 2, 4]), 0.1f32);
    let bindings = vec![
        TensorParamBinding::Variable,
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(weight),
    ];

    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");
    let input = multi_var_input(c_in, t_in, -1.0, 1.0);

    let (method, output, fallback_reason) =
        propagate_with_crown_fallback(&graph, &input).expect("CROWN/fallback");
    assert_bounds_valid(&output);

    match method {
        PropMethod::Crown => assert!(fallback_reason.is_none()),
        PropMethod::Ibp => {
            // IBP fallback is acceptable for complex composition.
            assert!(fallback_reason.is_some());
        }
        _ => panic!("unexpected PropMethod variant"),
    }
}

/// CROWN bounds at least as tight as IBP for ConvTranspose1d decoder.
#[test]
fn test_conv_transpose_decoder_crown_tighter_than_ibp() {
    let c_in = 4;
    let c_out = 2;
    let t_in = 8;
    let def = build_conv_transpose_decoder(c_in, c_out, t_in, 3, 2, 0, 1);

    let weight = ArrayD::from_elem(IxDyn(&[c_in, c_out * 2, 3]), 0.1f32);
    let bindings = vec![
        TensorParamBinding::Variable,
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(weight),
    ];

    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");
    let input = multi_var_input(c_in, t_in, -1.0, 1.0);

    // IBP
    let ibp_output = graph.propagate_ibp(&input).expect("IBP");

    // CROWN (with fallback)
    let (_, crown_output, _) =
        propagate_with_crown_fallback(&graph, &input).expect("CROWN/fallback");

    assert_crown_tighter_than_ibp(&crown_output, &ibp_output);
}

// ---------------------------------------------------------------------------
// AC4: Dvoice-scale tests (48, 96 channels)
// ---------------------------------------------------------------------------

/// Dvoice 48-channel decoder: ConvTranspose1d(48→24, k=8, s=4, pad=2) + trim.
#[test]
fn test_conv_transpose_decoder_dvoice_48ch() {
    let c_in = 48;
    let c_out = 24;
    let t_in = 16;
    let kernel_size = 8;
    let stride = 4;
    let padding = 2;
    let trim = 2;
    // ConvTranspose1d: (16-1)*4 + 8 - 4 = 64, after trim=2: 60

    let def = build_conv_transpose_decoder(c_in, c_out, t_in, kernel_size, stride, padding, trim);

    let weight = ArrayD::from_elem(IxDyn(&[c_in, c_out * 2, kernel_size]), 0.01f32);
    let bindings = vec![
        TensorParamBinding::Variable,
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(weight),
    ];

    let graph = tensor_kernel_to_graph(&def, &bindings).expect("dvoice 48ch decoder graph");
    let input = multi_var_input(c_in, t_in, -1.0, 1.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through dvoice 48ch decoder");
    assert_bounds_valid(&output);

    let (lo, hi) = output.lower_upper();
    let max_width = lo
        .iter()
        .zip(hi.iter())
        .map(|(l, u)| u - l)
        .fold(0.0f32, f32::max);
    assert!(
        max_width < 200.0,
        "IBP width {max_width} exceeds threshold at 48ch scale"
    );
}

/// Dvoice 96-channel decoder: ConvTranspose1d(96→48, k=8, s=4, pad=2) + trim.
#[test]
fn test_conv_transpose_decoder_dvoice_96ch() {
    let c_in = 96;
    let c_out = 48;
    let t_in = 8;
    let kernel_size = 8;
    let stride = 4;
    let padding = 2;
    let trim = 2;
    // ConvTranspose1d: (8-1)*4 + 8 - 4 = 32, after trim=2: 28

    let def = build_conv_transpose_decoder(c_in, c_out, t_in, kernel_size, stride, padding, trim);

    let weight = ArrayD::from_elem(IxDyn(&[c_in, c_out * 2, kernel_size]), 0.01f32);
    let bindings = vec![
        TensorParamBinding::Variable,
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(weight),
    ];

    let graph = tensor_kernel_to_graph(&def, &bindings).expect("dvoice 96ch decoder graph");
    let input = multi_var_input(c_in, t_in, -1.0, 1.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through dvoice 96ch decoder");
    assert_bounds_valid(&output);
}

/// Dvoice-scale CROWN propagation: 48-channel ConvTranspose1d decoder.
#[test]
fn test_conv_transpose_decoder_dvoice_48ch_crown() {
    let c_in = 48;
    let c_out = 24;
    let t_in = 16;
    let kernel_size = 8;
    let stride = 4;
    let padding = 2;
    let trim = 2;

    let def = build_conv_transpose_decoder(c_in, c_out, t_in, kernel_size, stride, padding, trim);

    let weight = ArrayD::from_elem(IxDyn(&[c_in, c_out * 2, kernel_size]), 0.01f32);
    let bindings = vec![
        TensorParamBinding::Variable,
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(weight),
    ];

    let graph = tensor_kernel_to_graph(&def, &bindings).expect("dvoice 48ch decoder graph");
    let input = multi_var_input(c_in, t_in, -1.0, 1.0);

    let (_, output, _) =
        propagate_with_crown_fallback(&graph, &input).expect("dvoice CROWN/fallback");
    assert_bounds_valid(&output);
}

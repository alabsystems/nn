// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Integration tests: tensor-level decoder composition for dvoice Demucs.
//!
//! Validates that a multi-op `TensorKernelDef` representing a Demucs temporal
//! decoder fragment chains through `tensor_kernel_to_graph` and produces a
//! single NY `GraphNetwork` where IBP bounds propagate end-to-end.
//!
//! Decoder pattern tested: BinaryAdd(skip) → Conv1d → GLU(Narrow+Sigmoid+BinaryMul) → GELU
//!
//! Part of #652. Complements the encoder composition tests in `compose_tensor_chain.rs`.

use super::common;
use nn_dsl::tensor_block_builder::TensorBlockBuilder;
use nn_verify::{
    propagate_with_crown_fallback, tensor_kernel_to_graph, PropMethod, TensorParamBinding,
};
use ndarray::{ArrayD, IxDyn};

// ---------------------------------------------------------------------------
// Decoder fragment builder
// ---------------------------------------------------------------------------

/// Build a Demucs decoder fragment: BinaryAdd → Conv1d → GLU → GELU.
///
/// Shape flow (channels=48, T=16):
///   data:[C, T] + skip:[C, T] → BinaryAdd → [C, T]
///   Conv1d(C→2C, k, s=1, pad) → [2C, T]
///   GLU(axis=0): Narrow[0..C] * Sigmoid(Narrow[C..2C]) → [C, T]
///   GELU → [C, T]
///
/// Returns (TensorKernelDef, num_variable_inputs).
fn build_decoder_fragment(
    channels: usize,
    length: usize,
    kernel_size: usize,
    padding: usize,
) -> nn_dsl::tensor_ir::TensorKernelDef {
    let doubled = channels * 2;
    let out_length = length + 2 * padding - kernel_size + 1;

    let mut b = TensorBlockBuilder::new("demucs_decoder_fragment");

    // Inputs: data (variable), skip (variable), conv weight (constant)
    let data = b.add_input("data", &[channels, length]);
    let skip = b.add_input("skip", &[channels, length]);
    let weight = b.add_input("weight", &[doubled, channels, kernel_size]);

    // Step 1: Skip connection add
    let added = b.add_binary_add(data, skip, &[channels, length]);

    // Step 2: Conv1d (channels → 2*channels for GLU split)
    let conv_out = b.add_conv1d(added, weight, None, 1, padding, &[doubled, out_length]);

    // Step 3: GLU decomposition (Narrow + Sigmoid + BinaryMul)
    let glu_out = b
        .add_glu(conv_out, 0, &[doubled, out_length])
        .expect("even dim");

    // Step 4: GELU activation
    let gelu_out = b.add_gelu(glu_out, &[channels, out_length]);

    b.build(gelu_out).expect("valid graph")
}

// ---------------------------------------------------------------------------
// IBP tests
// ---------------------------------------------------------------------------

/// Decoder fragment builds a valid NY graph.
#[test]
fn test_decoder_fragment_graph_builds() {
    let def = build_decoder_fragment(4, 16, 3, 1);
    def.validate().expect("decoder fragment should validate");

    let weight = ArrayD::from_elem(IxDyn(&[8, 4, 3]), 0.1f32);
    let bindings = vec![
        TensorParamBinding::Variable,
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(weight),
    ];

    let graph = tensor_kernel_to_graph(&def, &bindings).expect("decoder fragment graph must build");
    assert!(
        graph.num_nodes() >= 4,
        "graph should have at least 4 nodes (add, conv, glu, gelu)"
    );
}

/// IBP bounds propagate through the full decoder fragment.
#[test]
fn test_decoder_fragment_ibp_bounds_propagate() {
    let channels = 4;
    let length = 16;
    let def = build_decoder_fragment(channels, length, 3, 1);

    let weight = ArrayD::from_elem(IxDyn(&[8, 4, 3]), 0.1f32);
    let bindings = vec![
        TensorParamBinding::Variable,
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(weight),
    ];

    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");
    let input = common::uniform_bounds(&[2, channels, length], 1.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through decoder fragment");
    common::assert_bounds_valid(&output);
}

/// Dvoice-scale parameters: Conv1d(48→96, k=3, stride=1, pad=1).
#[test]
fn test_decoder_fragment_dvoice_params_ibp() {
    let channels = 48;
    let length = 16;
    let kernel_size = 3;
    let padding = 1;
    let doubled = channels * 2;

    let def = build_decoder_fragment(channels, length, kernel_size, padding);

    let weight = ArrayD::from_elem(IxDyn(&[doubled, channels, kernel_size]), 0.01f32);
    let bindings = vec![
        TensorParamBinding::Variable,
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(weight),
    ];

    let graph = tensor_kernel_to_graph(&def, &bindings).expect("dvoice decoder graph");
    let input = common::uniform_bounds(&[2, channels, length], 1.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through dvoice decoder");
    common::assert_bounds_valid(&output);

    // Vacuous widening guard: IBP width should not blow up.
    // With small weights (0.01), bounds should be reasonable.
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

// ---------------------------------------------------------------------------
// CROWN tests
// ---------------------------------------------------------------------------

/// CROWN propagation through the decoder fragment.
#[test]
fn test_decoder_fragment_crown_propagates() {
    let channels = 4;
    let length = 8;
    let def = build_decoder_fragment(channels, length, 3, 1);

    let weight = ArrayD::from_elem(IxDyn(&[8, 4, 3]), 0.1f32);
    let bindings = vec![
        TensorParamBinding::Variable,
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(weight),
    ];

    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");
    let input = common::uniform_bounds(&[2, channels, length], 1.0);

    let (method, output, fallback_reason) =
        propagate_with_crown_fallback(&graph, &input).expect("CROWN/fallback");
    common::assert_bounds_valid(&output);

    match method {
        PropMethod::Crown => assert!(fallback_reason.is_none()),
        PropMethod::Ibp => {
            // IBP fallback is acceptable for the decoder fragment.
            assert!(fallback_reason.is_some());
        }
        _ => panic!("unexpected PropMethod variant"),
    }
}

/// CROWN bounds should be at least as tight as IBP.
#[test]
fn test_decoder_fragment_crown_tighter_than_ibp() {
    let channels = 4;
    let length = 8;
    let def = build_decoder_fragment(channels, length, 3, 1);

    let weight = ArrayD::from_elem(IxDyn(&[8, 4, 3]), 0.1f32);
    let bindings = vec![
        TensorParamBinding::Variable,
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(weight),
    ];

    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");
    let input = common::uniform_bounds(&[2, channels, length], 1.0);

    // IBP
    let ibp_output = graph.propagate_ibp(&input).expect("IBP");

    // CROWN (with fallback)
    let (_, crown_output, _) =
        propagate_with_crown_fallback(&graph, &input).expect("CROWN/fallback");

    common::assert_crown_tighter_than_ibp(&crown_output, &ibp_output);
}

/// Dvoice-scale CROWN propagation: full 48-channel decoder fragment.
#[test]
fn test_decoder_fragment_dvoice_scale_crown() {
    let channels = 48;
    let length = 16;
    let kernel_size = 3;
    let padding = 1;
    let doubled = channels * 2;

    let def = build_decoder_fragment(channels, length, kernel_size, padding);

    let weight = ArrayD::from_elem(IxDyn(&[doubled, channels, kernel_size]), 0.01f32);
    let bindings = vec![
        TensorParamBinding::Variable,
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(weight),
    ];

    let graph = tensor_kernel_to_graph(&def, &bindings).expect("dvoice decoder graph");
    let input = common::uniform_bounds(&[2, channels, length], 1.0);

    let (_, output, _) =
        propagate_with_crown_fallback(&graph, &input).expect("dvoice CROWN/fallback");
    common::assert_bounds_valid(&output);
}

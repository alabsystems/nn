// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Named-binding `model_to_graph_network()` composition tests.
//!
//! Validates the named-binding API (`model_to_graph_network`) against a
//! Kokoro-scale decoder block: Conv1d → ReLU → InstanceNorm → Conv1d.
//! Uses `HashMap<&str, TensorParamBinding>` instead of positional vectors,
//! demonstrating the ergonomic advantage for models with many constant inputs.
//!
//! Part of #2039.

use std::collections::HashMap;

use super::common::{assert_bounds_valid, assert_crown_tighter_when_not_fallback, uniform_bounds};
use nn_dsl::tensor_block_builder::TensorBlockBuilder;
use nn_verify::{model_to_graph_network, TensorParamBinding};
use ndarray::{ArrayD, IxDyn};

// ---------------------------------------------------------------------------
// Small-scale dimensions for NY tractability
// ---------------------------------------------------------------------------

const IN_CHANNELS: usize = 4;
const MID_CHANNELS: usize = 4;
const OUT_CHANNELS: usize = 4;
const TIME: usize = 8;
const KERNEL_SIZE: usize = 3;
const PADDING: usize = 1;
const WEIGHT_MAG: f32 = 0.001;
const EPS: f32 = 1e-5;

// ---------------------------------------------------------------------------
// Builder
// ---------------------------------------------------------------------------

/// Build a Conv1d → ReLU → InstanceNorm → Conv1d block.
///
/// Named inputs:
/// - "features": Variable [IN_CHANNELS, TIME]
/// - "conv1_weight": ConstantTensor [MID_CHANNELS, IN_CHANNELS, KERNEL_SIZE]
/// - "conv1_bias": ConstantTensor [MID_CHANNELS]
/// - "eps": ConstantScalar (1e-5)
/// - "conv2_weight": ConstantTensor [OUT_CHANNELS, MID_CHANNELS, KERNEL_SIZE]
/// - "conv2_bias": ConstantTensor [OUT_CHANNELS]
fn build_decoder_block() -> nn_dsl::tensor_ir::TensorKernelDef {
    let mid_shape = [MID_CHANNELS, TIME];
    let out_shape = [OUT_CHANNELS, TIME];

    let mut b = TensorBlockBuilder::new("decoder_block");

    // Variable input
    let features = b.add_input("features", &[IN_CHANNELS, TIME]);

    // Conv1d #1 weights (named constants)
    let conv1_w = b.add_input("conv1_weight", &[MID_CHANNELS, IN_CHANNELS, KERNEL_SIZE]);
    let conv1_b = b.add_input("conv1_bias", &[MID_CHANNELS]);

    // InstanceNorm eps
    let eps = b.add_input("eps", &[1]);

    // Conv1d #2 weights (named constants)
    let conv2_w = b.add_input("conv2_weight", &[OUT_CHANNELS, MID_CHANNELS, KERNEL_SIZE]);
    let conv2_b = b.add_input("conv2_bias", &[OUT_CHANNELS]);

    // Forward: Conv1d → ReLU → InstanceNorm → Conv1d
    let x = b.add_conv1d(features, conv1_w, Some(conv1_b), 1, PADDING, &mid_shape);
    let x = b.add_relu(x, &mid_shape);
    let x = b.add_instance_norm(x, eps, 1, None, None, &mid_shape);
    let x = b.add_conv1d(x, conv2_w, Some(conv2_b), 1, PADDING, &out_shape);

    b.build(x).expect("valid decoder block")
}

/// Build named bindings map for the decoder block.
fn decoder_bindings() -> HashMap<&'static str, TensorParamBinding> {
    let mut bindings = HashMap::new();

    // "features" defaults to Variable (not in map)

    // Conv1d #1 weights
    bindings.insert(
        "conv1_weight",
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[MID_CHANNELS, IN_CHANNELS, KERNEL_SIZE]),
            WEIGHT_MAG,
        )),
    );
    bindings.insert(
        "conv1_bias",
        TensorParamBinding::ConstantTensor(ArrayD::zeros(IxDyn(&[MID_CHANNELS]))),
    );

    // InstanceNorm eps
    bindings.insert("eps", TensorParamBinding::ConstantScalar(EPS));

    // Conv1d #2 weights
    bindings.insert(
        "conv2_weight",
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[OUT_CHANNELS, MID_CHANNELS, KERNEL_SIZE]),
            WEIGHT_MAG,
        )),
    );
    bindings.insert(
        "conv2_bias",
        TensorParamBinding::ConstantTensor(ArrayD::zeros(IxDyn(&[OUT_CHANNELS]))),
    );

    bindings
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

/// Named-binding API builds a NY graph from the decoder block.
#[test]
fn test_model_to_graph_builds() {
    let def = build_decoder_block();
    let bindings = decoder_bindings();

    let graph = model_to_graph_network(&def, &bindings).expect("decoder block graph");
    // Conv1d + ReLU + InstanceNorm + Conv1d = at least 4 NY nodes.
    assert!(
        graph.num_nodes() >= 4,
        "decoder block needs >= 4 NY nodes, got {}",
        graph.num_nodes()
    );
}

/// IBP propagation through the named-binding decoder block produces finite bounds.
#[test]
fn test_model_to_graph_ibp_propagates() {
    let def = build_decoder_block();
    let bindings = decoder_bindings();
    let graph = model_to_graph_network(&def, &bindings).expect("graph");

    let input = uniform_bounds(&[IN_CHANNELS, TIME], 10.0);
    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through decoder block");

    assert_eq!(output.lower_upper().0.shape(), &[OUT_CHANNELS, TIME]);
    assert_bounds_valid(&output);
}

/// CROWN propagation through the named-binding decoder block.
#[test]
fn test_model_to_graph_crown_propagates() {
    let def = build_decoder_block();
    let bindings = decoder_bindings();
    let graph = model_to_graph_network(&def, &bindings).expect("graph");

    let input = uniform_bounds(&[IN_CHANNELS, TIME], 10.0);

    let (_method, output, _fallback) = assert_crown_tighter_when_not_fallback(&graph, &input);

    assert_eq!(output.lower_upper().0.shape(), &[OUT_CHANNELS, TIME]);
    assert_bounds_valid(&output);
}

/// Omitting "features" from the bindings map defaults it to Variable —
/// the graph still builds and matches the explicit-Variable path.
#[test]
fn test_model_to_graph_missing_binding_defaults_to_variable() {
    let def = build_decoder_block();

    // Provide all constants but omit "features" — it should default to Variable.
    let partial_bindings = decoder_bindings();
    // decoder_bindings() already omits "features", so this is the same as the
    // full named path. Verify explicitly that inserting Variable is equivalent.
    let mut explicit_bindings = decoder_bindings();
    explicit_bindings.insert("features", TensorParamBinding::Variable);

    let partial_graph =
        model_to_graph_network(&def, &partial_bindings).expect("partial bindings graph");
    let explicit_graph =
        model_to_graph_network(&def, &explicit_bindings).expect("explicit variable graph");

    assert_eq!(partial_graph.num_nodes(), explicit_graph.num_nodes());
}

/// Named bindings produce equivalent results to positional bindings.
#[test]
fn test_named_vs_positional_equivalence() {
    let def = build_decoder_block();

    // Named path
    let named = decoder_bindings();
    let named_graph = model_to_graph_network(&def, &named).expect("named graph");

    // Positional path — must match the add_input() order exactly
    let positional = vec![
        TensorParamBinding::Variable, // features
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[MID_CHANNELS, IN_CHANNELS, KERNEL_SIZE]),
            WEIGHT_MAG,
        )),
        TensorParamBinding::ConstantTensor(ArrayD::zeros(IxDyn(&[MID_CHANNELS]))),
        TensorParamBinding::ConstantScalar(EPS),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[OUT_CHANNELS, MID_CHANNELS, KERNEL_SIZE]),
            WEIGHT_MAG,
        )),
        TensorParamBinding::ConstantTensor(ArrayD::zeros(IxDyn(&[OUT_CHANNELS]))),
    ];
    let positional_graph =
        nn_verify::tensor_kernel_to_graph(&def, &positional).expect("positional graph");

    // Both graphs should have the same structure.
    assert_eq!(named_graph.num_nodes(), positional_graph.num_nodes());

    // Both should produce the same IBP bounds.
    let input = uniform_bounds(&[IN_CHANNELS, TIME], 5.0);
    let named_out = named_graph.propagate_ibp(&input).expect("named IBP");
    let positional_out = positional_graph
        .propagate_ibp(&input)
        .expect("positional IBP");

    let (n_lo, n_hi) = named_out.lower_upper();
    let (p_lo, p_hi) = positional_out.lower_upper();

    for (&nl, &pl) in n_lo.iter().zip(p_lo.iter()) {
        assert!(
            (nl - pl).abs() < 1e-6,
            "lower mismatch: named={nl}, positional={pl}"
        );
    }
    for (&nh, &ph) in n_hi.iter().zip(p_hi.iter()) {
        assert!(
            (nh - ph).abs() < 1e-6,
            "upper mismatch: named={nh}, positional={ph}"
        );
    }
}

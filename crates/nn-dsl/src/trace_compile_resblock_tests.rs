// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Tests for `compile_fused_adain_resblock`.
//!
//! Verifies that `FusedAdainResBlock` compiles to a single `Dispatch` step
//! with correct kernel name, weights, and output shape. Also covers error
//! paths: non-finite eps and wrong-rank inputs.
//!
//! Part of #2459.

use nn_core::dyn_tensor::trace::{
    ComputationGraph, KokoroFusedOp, ResBlockActivation, TraceNode, TraceOp, WeightRef,
};
use nn_core::DType;

use crate::trace_compile::{compile_trace, CompiledStep};

// -- Helpers ------------------------------------------------------------------

fn graph_from_nodes(nodes: Vec<TraceNode>) -> ComputationGraph {
    ComputationGraph::from_nodes(nodes)
}

fn input_node(id: u64, shape: &[usize]) -> TraceNode {
    TraceNode::new(
        id,
        format!("input_{id}"),
        TraceOp::Input,
        vec![],
        shape.to_vec(),
        DType::F32,
    )
}

/// Make a weight reference with deterministic data matching the shape.
fn weight(shape: &[usize]) -> WeightRef {
    let n: usize = shape.iter().product();
    // Fill with small positive values to avoid NaN in downstream ops.
    let data: Vec<f32> = (0..n).map(|i| 0.01 * (i as f32 + 1.0)).collect();
    WeightRef::new(data, shape.to_vec()).expect("valid weight")
}

/// Build a standard resblock graph: [1, c, t] input + [1, style_dim] style → resblock.
fn build_resblock_graph(
    name: &str,
    c: usize,
    t: usize,
    k: usize,
    style_dim: usize,
    activation: ResBlockActivation,
    dilation: usize,
    padding: usize,
    eps: f64,
    residual_scale: f64,
    output_shape: Vec<usize>,
) -> ComputationGraph {
    let op = TraceOp::KokoroFused(KokoroFusedOp::FusedAdainResBlock {
        activation,
        adain1_weight: weight(&[2 * c, style_dim]),
        adain1_bias: weight(&[2 * c]),
        adain2_weight: weight(&[2 * c, style_dim]),
        adain2_bias: weight(&[2 * c]),
        conv1_weight: weight(&[c, c, k]),
        conv1_bias: weight(&[c]),
        conv1_dilation: dilation,
        conv1_padding: padding,
        conv2_weight: weight(&[c, c, k]),
        conv2_bias: weight(&[c]),
        conv2_padding: padding,
        eps,
        residual_scale,
    });
    graph_from_nodes(vec![
        input_node(0, &[1, c, t]),
        input_node(1, &[1, style_dim]),
        TraceNode::new(2, name.into(), op, vec![0, 1], output_shape, DType::F32),
    ])
}

/// Find the single `Dispatch` step in compiled output. Returns `(kernel, weight_data)`.
fn get_dispatch(
    steps: &[CompiledStep],
) -> (
    &crate::trace_compile::CompiledKernel,
    &std::collections::HashMap<String, WeightRef>,
) {
    let d = steps
        .iter()
        .find(|s| matches!(s, CompiledStep::Dispatch { .. }))
        .expect("dispatch");
    match d {
        CompiledStep::Dispatch {
            kernel,
            weight_data,
            ..
        } => (kernel, weight_data),
        _ => unreachable!(),
    }
}

// -- Kokoro Generator ResBlock (Snake activation) ----------------------------

/// A minimal FusedAdainResBlock with Snake activation should compile to
/// exactly one `Dispatch` step (beyond the two input passthroughs).
#[test]
fn test_compile_resblock_snake_single_dispatch() {
    let (c, t, k) = (4, 16, 3);
    let activation = ResBlockActivation::Snake {
        alpha1: weight(&[1, c, 1]),
        alpha2: weight(&[1, c, 1]),
    };
    let output_shape = vec![1, c, t];
    let graph = build_resblock_graph(
        "resblock_snake",
        c,
        t,
        k,
        8,
        activation,
        1,
        1,
        1e-5,
        1.0,
        output_shape.clone(),
    );

    let steps = compile_trace(&graph).expect("snake resblock should compile");
    let dispatch_count = steps
        .iter()
        .filter(|s| matches!(s, CompiledStep::Dispatch { .. }))
        .count();
    assert_eq!(dispatch_count, 1, "ResBlock should compile to 1 dispatch");

    let (kernel, weight_data) = get_dispatch(&steps);
    assert_eq!(kernel.name(), "fused_adain_resblock");
    assert_eq!(kernel.output_shape(), Some(output_shape.as_slice()));

    for key in [
        "adain1_w", "adain1_b", "adain2_w", "adain2_b", "conv1_w", "conv1_b", "conv2_w", "conv2_b",
        "alpha1", "alpha2", "eps", "ones",
    ] {
        assert!(weight_data.contains_key(key), "missing weight key: {key}");
    }
    // Snake: residual_scale == 1.0 should not add a weight
    assert!(!weight_data.contains_key("residual_scale"));
}

// -- Kokoro F0 ResBlock (LeakyRelu activation) --------------------------------

/// F0 AdainResBlk1d uses LeakyRelu and a non-1.0 residual scale.
/// Uses dilation=1, padding=1 (length-preserving for k=3) since
/// the shared helper uses same padding for both convs and the residual
/// add requires matching shapes (x + conv2_out).
#[test]
fn test_compile_resblock_leaky_relu_with_residual_scale() {
    let (c, t, k) = (8, 32, 3);
    let padding = (k - 1) / 2; // symmetric, length-preserving for stride=1, dilation=1
    let inv_sqrt2: f64 = 1.0 / f64::from(2.0_f32).sqrt();
    let activation = ResBlockActivation::LeakyRelu { slope: 0.2 };
    let output_shape = vec![1, c, t];

    let graph = build_resblock_graph(
        "resblock_leaky_relu",
        c,
        t,
        k,
        16,
        activation,
        1,
        padding,
        1e-5,
        inv_sqrt2,
        output_shape.clone(),
    );
    let steps = compile_trace(&graph).expect("leaky_relu resblock should compile");

    let dispatch_count = steps
        .iter()
        .filter(|s| matches!(s, CompiledStep::Dispatch { .. }))
        .count();
    assert_eq!(
        dispatch_count, 1,
        "F0 ResBlock should compile to 1 dispatch"
    );

    let (kernel, weight_data) = get_dispatch(&steps);
    assert_eq!(kernel.name(), "fused_adain_resblock");
    assert_eq!(kernel.output_shape(), Some(output_shape.as_slice()));

    // LeakyRelu: no alpha weights, but residual_scale is present
    assert!(
        !weight_data.contains_key("alpha1"),
        "LeakyRelu should not have alpha weights"
    );
    assert!(
        weight_data.contains_key("residual_scale"),
        "non-1.0 residual_scale should produce a weight"
    );
    for key in ["adain1_w", "adain1_b", "conv1_w", "conv1_b"] {
        assert!(weight_data.contains_key(key), "missing key: {key}");
    }
}

// -- Error paths --------------------------------------------------------------

/// Non-finite eps should be rejected.
#[test]
fn test_compile_resblock_non_finite_eps() {
    let activation = ResBlockActivation::LeakyRelu { slope: 0.2 };
    let graph = build_resblock_graph(
        "resblock_nan_eps",
        4,
        16,
        3,
        8,
        activation,
        1,
        1,
        f64::NAN,
        1.0,
        vec![1, 4, 16],
    );
    let err = compile_trace(&graph).unwrap_err();
    let msg = format!("{err}");
    assert!(
        msg.contains("NonFiniteConstant") || msg.contains("non-finite") || msg.contains("eps"),
        "expected non-finite eps error, got: {msg}"
    );
}

/// Rank-2 input (missing batch dim) should be rejected.
#[test]
fn test_compile_resblock_wrong_rank() {
    let c = 4;
    let t = 16;
    let k = 3;

    let op = TraceOp::KokoroFused(KokoroFusedOp::FusedAdainResBlock {
        activation: ResBlockActivation::LeakyRelu { slope: 0.2 },
        adain1_weight: weight(&[2 * c, 8]),
        adain1_bias: weight(&[2 * c]),
        adain2_weight: weight(&[2 * c, 8]),
        adain2_bias: weight(&[2 * c]),
        conv1_weight: weight(&[c, c, k]),
        conv1_bias: weight(&[c]),
        conv1_dilation: 1,
        conv1_padding: 1,
        conv2_weight: weight(&[c, c, k]),
        conv2_bias: weight(&[c]),
        conv2_padding: 1,
        eps: 1e-5,
        residual_scale: 1.0,
    });

    let graph = graph_from_nodes(vec![
        input_node(0, &[c, t]), // rank 2, not rank 3
        input_node(1, &[1, 8]),
        TraceNode::new(
            2,
            "resblock_rank2".into(),
            op,
            vec![0, 1],
            vec![c, t],
            DType::F32,
        ),
    ]);

    let err = compile_trace(&graph).unwrap_err();
    let msg = format!("{err}");
    assert!(
        msg.contains("rank") || msg.contains("UnsupportedTraceOp"),
        "expected rank error, got: {msg}"
    );
}

/// Negative eps should be rejected.
#[test]
fn test_compile_resblock_negative_eps() {
    let activation = ResBlockActivation::LeakyRelu { slope: 0.2 };
    let graph = build_resblock_graph(
        "resblock_neg_eps",
        4,
        16,
        3,
        8,
        activation,
        1,
        1,
        -1e-5,
        1.0,
        vec![1, 4, 16],
    );
    let err = compile_trace(&graph).unwrap_err();
    let msg = format!("{err}");
    assert!(
        msg.contains("NonFiniteConstant") || msg.contains("eps"),
        "expected eps validation error, got: {msg}"
    );
}

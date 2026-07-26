// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Constant-fold correctness tests for AdaIN1d and RmsNorm tensor translations.
//!
//! Validates that constant inputs produce correct analytical values instead of
//! the hardcoded 0.0 that was present before #661.
//!
//! Each test verifies both structural folding (num_nodes == 1) AND numerical
//! correctness (output value matches analytical expectation). Per #425, tests
//! that only check structure without checking values cannot catch computation bugs.

use nn_dsl::tensor_ir::{TensorKernelDef, TensorNode, TensorNodeId, TensorOpKind};
use nn_verify::{tensor_kernel_to_graph, BoundedTensor, TensorParamBinding};
use ndarray::{ArrayD, IxDyn};

/// Extract the constant value from a 1-node constant-fold graph by running IBP
/// with a dummy input. The constant output is independent of input bounds.
fn extract_constant(graph: &nn_verify::GraphNetwork, shape: &[usize]) -> f32 {
    let lower = ArrayD::from_elem(IxDyn(shape), 0.0f32);
    let upper = ArrayD::from_elem(IxDyn(shape), 0.0f32);
    let input = BoundedTensor::new(lower, upper).expect("dummy bounds");
    let output = graph.propagate_ibp(&input).expect("IBP on constant graph");
    let (lo, hi) = output.lower_upper();
    // For a constant graph, lo == hi for all elements.
    // Use flat view to index regardless of dimensionality.
    let lo_flat = lo.as_slice().expect("contiguous lower");
    let hi_flat = hi.as_slice().expect("contiguous upper");
    let val = lo_flat[0];
    assert!(
        (val - hi_flat[0]).abs() < 1e-6,
        "constant graph should have lo==hi, got lo={} hi={}",
        val,
        hi_flat[0]
    );
    val
}

fn mk_input(id: usize, name: &str, shape: Vec<usize>) -> TensorNode {
    TensorNode::new(
        TensorNodeId::new(id),
        TensorOpKind::Input {
            name: name.to_string(),
            shape: shape.clone(),
        },
        shape,
    )
}

// --- AC3: AdaIN1d constant-fold returns beta, not 0.0 ---

#[test]
fn test_adain1d_constant_input_returns_beta_not_zero() {
    // AdaIN1d(constant_input) = gamma * InstanceNorm(constant) + beta = gamma * 0 + beta = beta.
    // With beta=7.5, the constant-fold must return 7.5, not 0.0.
    let shape = vec![2, 4, 16];
    let def = TensorKernelDef::new(
        "adain1d_const_fold",
        vec![
            mk_input(0, "x", shape.clone()),
            mk_input(1, "eps", vec![1]),
            mk_input(2, "style_gamma", vec![4]),
            mk_input(3, "style_beta", vec![4]),
            TensorNode::new(
                TensorNodeId::new(4),
                TensorOpKind::AdaIN1d {
                    input: TensorNodeId::new(0),
                    eps: TensorNodeId::new(1),
                    axis: 2,
                    style_gamma: TensorNodeId::new(2),
                    style_beta: TensorNodeId::new(3),
                },
                shape.clone(),
            ),
        ],
        TensorNodeId::new(4),
    );
    let bindings = vec![
        TensorParamBinding::ConstantScalar(5.0),  // constant input
        TensorParamBinding::ConstantScalar(1e-5), // eps
        TensorParamBinding::ConstantScalar(2.0),  // style_gamma (irrelevant for constant fold)
        TensorParamBinding::ConstantScalar(7.5),  // style_beta — this is the expected output
    ];
    let graph =
        tensor_kernel_to_graph(&def, &bindings).expect("constant-input AdaIN1d should translate");

    // Structural: constant fold should produce a single-node graph.
    assert_eq!(
        graph.num_nodes(),
        1,
        "constant input should fold to 1 graph node, got {}",
        graph.num_nodes()
    );

    // Numerical correctness: output must be beta=7.5, not the old hardcoded 0.0.
    let val = extract_constant(&graph, &shape);
    assert!(
        (val - 7.5).abs() < 1e-6,
        "AdaIN1d constant-fold should return beta=7.5, got {val}"
    );
}

#[test]
fn test_adain1d_constant_input_zero_beta_returns_zero() {
    // With beta=0.0, the constant-fold should return 0.0 (which happens to be
    // the old behavior — this test ensures we didn't break the zero-beta case).
    let shape = vec![2, 4, 16];
    let def = TensorKernelDef::new(
        "adain1d_const_fold_zero_beta",
        vec![
            mk_input(0, "x", shape.clone()),
            mk_input(1, "eps", vec![1]),
            mk_input(2, "style_gamma", vec![4]),
            mk_input(3, "style_beta", vec![4]),
            TensorNode::new(
                TensorNodeId::new(4),
                TensorOpKind::AdaIN1d {
                    input: TensorNodeId::new(0),
                    eps: TensorNodeId::new(1),
                    axis: 2,
                    style_gamma: TensorNodeId::new(2),
                    style_beta: TensorNodeId::new(3),
                },
                shape.clone(),
            ),
        ],
        TensorNodeId::new(4),
    );
    let bindings = vec![
        TensorParamBinding::ConstantScalar(3.0),
        TensorParamBinding::ConstantScalar(1e-5),
        TensorParamBinding::ConstantScalar(1.0),
        TensorParamBinding::ConstantScalar(0.0), // beta=0
    ];
    let graph = tensor_kernel_to_graph(&def, &bindings)
        .expect("constant-input AdaIN1d with zero beta should translate");

    assert_eq!(
        graph.num_nodes(),
        1,
        "constant input should fold to 1 graph node, got {}",
        graph.num_nodes()
    );

    // Numerical: output must be 0.0 (beta=0).
    let val = extract_constant(&graph, &shape);
    assert!(
        val.abs() < 1e-6,
        "AdaIN1d constant-fold with beta=0 should return 0.0, got {val}"
    );
}

// --- AC4: RmsNorm constant-fold returns correct analytical value ---

#[test]
fn test_rms_norm_constant_input_returns_correct_value() {
    // RmsNorm(c) = c / sqrt(c² + eps) * weight.
    // For c=3.0, eps=1e-5, weight=2.0:
    //   rms = sqrt(9.0 + 1e-5) ≈ 3.0000017
    //   output = 3.0 / 3.0000017 * 2.0 ≈ 1.9999989
    // The old code returned 0.0, which is incorrect.
    let shape = vec![2, 8];
    let def = TensorKernelDef::new(
        "rms_norm_const_fold",
        vec![
            mk_input(0, "x", shape.clone()),
            mk_input(1, "eps", vec![1]),
            mk_input(2, "weight", vec![8]),
            TensorNode::new(
                TensorNodeId::new(3),
                TensorOpKind::RmsNorm {
                    input: TensorNodeId::new(0),
                    eps: TensorNodeId::new(1),
                    axis: 1,
                    weight: TensorNodeId::new(2),
                },
                shape.clone(),
            ),
        ],
        TensorNodeId::new(3),
    );
    let bindings = vec![
        TensorParamBinding::ConstantScalar(3.0),  // constant input
        TensorParamBinding::ConstantScalar(1e-5), // eps
        TensorParamBinding::ConstantScalar(2.0),  // weight
    ];
    let graph =
        tensor_kernel_to_graph(&def, &bindings).expect("constant-input RmsNorm should translate");

    // Structural: constant fold should produce a single-node graph.
    assert_eq!(
        graph.num_nodes(),
        1,
        "constant input should fold to 1 graph node, got {}",
        graph.num_nodes()
    );

    // Numerical: output = 3.0 / sqrt(9.0 + 1e-5) * 2.0 ≈ 2.0
    let expected = 3.0_f32 / (9.0_f32 + 1e-5_f32).sqrt() * 2.0;
    let val = extract_constant(&graph, &shape);
    assert!(
        (val - expected).abs() < 1e-4,
        "RmsNorm constant-fold should return ~{expected}, got {val}"
    );
}

#[test]
fn test_rms_norm_constant_input_negative_value() {
    // For c=-4.0, eps=1e-5, weight=1.0:
    //   rms = sqrt(16.0 + 1e-5) ≈ 4.0000013
    //   output = -4.0 / 4.0000013 * 1.0 ≈ -0.9999997
    // The old code returned 0.0, missing the sign and weight.
    let shape = vec![2, 8];
    let def = TensorKernelDef::new(
        "rms_norm_const_fold_neg",
        vec![
            mk_input(0, "x", shape.clone()),
            mk_input(1, "eps", vec![1]),
            mk_input(2, "weight", vec![8]),
            TensorNode::new(
                TensorNodeId::new(3),
                TensorOpKind::RmsNorm {
                    input: TensorNodeId::new(0),
                    eps: TensorNodeId::new(1),
                    axis: 1,
                    weight: TensorNodeId::new(2),
                },
                shape.clone(),
            ),
        ],
        TensorNodeId::new(3),
    );
    let bindings = vec![
        TensorParamBinding::ConstantScalar(-4.0), // negative constant input
        TensorParamBinding::ConstantScalar(1e-5),
        TensorParamBinding::ConstantScalar(1.0),
    ];
    let graph = tensor_kernel_to_graph(&def, &bindings)
        .expect("constant-input RmsNorm with negative input should translate");

    assert_eq!(
        graph.num_nodes(),
        1,
        "constant input should fold to 1 graph node, got {}",
        graph.num_nodes()
    );

    // Numerical: output = -4.0 / sqrt(16.0 + 1e-5) * 1.0 ≈ -1.0
    let expected = -4.0_f32 / (16.0_f32 + 1e-5_f32).sqrt() * 1.0;
    let val = extract_constant(&graph, &shape);
    assert!(
        (val - expected).abs() < 1e-4,
        "RmsNorm constant-fold for negative input should return ~{expected}, got {val}"
    );
}

#[test]
fn test_rms_norm_constant_input_zero_returns_zero() {
    // For c=0.0, eps=1e-5, weight=5.0:
    //   rms = sqrt(0.0 + 1e-5) ≈ 0.00316
    //   output = 0.0 / 0.00316 * 5.0 = 0.0
    // Zero input should still return zero (matches old behavior for this case).
    let shape = vec![2, 8];
    let def = TensorKernelDef::new(
        "rms_norm_const_fold_zero",
        vec![
            mk_input(0, "x", shape.clone()),
            mk_input(1, "eps", vec![1]),
            mk_input(2, "weight", vec![8]),
            TensorNode::new(
                TensorNodeId::new(3),
                TensorOpKind::RmsNorm {
                    input: TensorNodeId::new(0),
                    eps: TensorNodeId::new(1),
                    axis: 1,
                    weight: TensorNodeId::new(2),
                },
                shape.clone(),
            ),
        ],
        TensorNodeId::new(3),
    );
    let bindings = vec![
        TensorParamBinding::ConstantScalar(0.0), // zero input
        TensorParamBinding::ConstantScalar(1e-5),
        TensorParamBinding::ConstantScalar(5.0),
    ];
    let graph = tensor_kernel_to_graph(&def, &bindings)
        .expect("constant-input RmsNorm with zero input should translate");

    assert_eq!(
        graph.num_nodes(),
        1,
        "constant input should fold to 1 graph node, got {}",
        graph.num_nodes()
    );

    // Numerical: 0.0 / anything * weight = 0.0
    let val = extract_constant(&graph, &shape);
    assert!(
        val.abs() < 1e-6,
        "RmsNorm constant-fold for zero input should return 0.0, got {val}"
    );
}

// --- Non-uniform ConstantTensor rejection tests (#755) ---

#[test]
fn test_rms_norm_constant_input_nonuniform_weight_rejected() {
    // Constant input + non-uniform ConstantTensor weight cannot be folded
    // to a single scalar — the translation rejects this degenerate case.
    let shape = vec![2, 8];
    let weight_data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let def = TensorKernelDef::new(
        "rms_norm_nonuniform_weight",
        vec![
            mk_input(0, "x", shape.clone()),
            mk_input(1, "eps", vec![1]),
            mk_input(2, "weight", vec![8]),
            TensorNode::new(
                TensorNodeId::new(3),
                TensorOpKind::RmsNorm {
                    input: TensorNodeId::new(0),
                    eps: TensorNodeId::new(1),
                    axis: 1,
                    weight: TensorNodeId::new(2),
                },
                shape,
            ),
        ],
        TensorNodeId::new(3),
    );
    let err = tensor_kernel_to_graph(
        &def,
        &[
            TensorParamBinding::ConstantScalar(3.0),
            TensorParamBinding::ConstantScalar(1e-5),
            TensorParamBinding::ConstantTensor(
                ArrayD::from_shape_vec(IxDyn(&[8]), weight_data).unwrap(),
            ),
        ],
    )
    .expect_err("non-uniform weight with constant input should be rejected");
    let msg = format!("{err:?}");
    assert!(
        msg.contains("non-uniform weight"),
        "unexpected error: {msg}"
    );
}

#[test]
fn test_adain1d_constant_input_nonuniform_beta_rejected() {
    // Constant input + non-uniform ConstantTensor beta cannot be folded
    // to a single scalar — the translation rejects this degenerate case.
    let shape = vec![2, 4, 16];
    let beta_data = vec![0.1, 0.2, 0.3, 0.4];
    let def = TensorKernelDef::new(
        "adain1d_nonuniform_beta",
        vec![
            mk_input(0, "x", shape.clone()),
            mk_input(1, "eps", vec![1]),
            mk_input(2, "style_gamma", vec![4]),
            mk_input(3, "style_beta", vec![4]),
            TensorNode::new(
                TensorNodeId::new(4),
                TensorOpKind::AdaIN1d {
                    input: TensorNodeId::new(0),
                    eps: TensorNodeId::new(1),
                    axis: 2,
                    style_gamma: TensorNodeId::new(2),
                    style_beta: TensorNodeId::new(3),
                },
                shape,
            ),
        ],
        TensorNodeId::new(4),
    );
    let err = tensor_kernel_to_graph(
        &def,
        &[
            TensorParamBinding::ConstantScalar(5.0),
            TensorParamBinding::ConstantScalar(1e-5),
            TensorParamBinding::ConstantScalar(2.0),
            TensorParamBinding::ConstantTensor(
                ArrayD::from_shape_vec(IxDyn(&[4]), beta_data).unwrap(),
            ),
        ],
    )
    .expect_err("non-uniform beta with constant input should be rejected");
    let msg = format!("{err:?}");
    assert!(msg.contains("non-uniform beta"), "unexpected error: {msg}");
}

#[test]
fn test_instance_norm_affine_constant_input_nonuniform_beta_rejected() {
    // Constant input + non-uniform ConstantTensor beta cannot be folded
    // to a single scalar — the translation rejects this degenerate case.
    let shape = vec![2, 4, 16];
    let beta_data = vec![0.1, 0.2, 0.3, 0.4];
    let def = TensorKernelDef::new(
        "instance_norm_nonuniform_beta",
        vec![
            mk_input(0, "x", shape.clone()),
            mk_input(1, "eps", vec![1]),
            mk_input(2, "gamma", vec![4]),
            mk_input(3, "beta", vec![4]),
            TensorNode::new(
                TensorNodeId::new(4),
                TensorOpKind::InstanceNorm1d {
                    input: TensorNodeId::new(0),
                    eps: TensorNodeId::new(1),
                    axis: 2,
                    gamma: Some(TensorNodeId::new(2)),
                    beta: Some(TensorNodeId::new(3)),
                },
                shape,
            ),
        ],
        TensorNodeId::new(4),
    );
    let err = tensor_kernel_to_graph(
        &def,
        &[
            TensorParamBinding::ConstantScalar(1.0),
            TensorParamBinding::ConstantScalar(1e-5),
            TensorParamBinding::ConstantScalar(1.0),
            TensorParamBinding::ConstantTensor(
                ArrayD::from_shape_vec(IxDyn(&[4]), beta_data).unwrap(),
            ),
        ],
    )
    .expect_err("non-uniform beta with constant input should be rejected");
    let msg = format!("{err:?}");
    assert!(msg.contains("non-uniform beta"), "unexpected error: {msg}");
}

// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! InstanceNorm1d rank guard and error path tests.
//! Extracted from graph_translate_tensor.rs (#356).
//! Affine variant tests split to graph_translate_tensor_norm_affine.rs (#423).

use nn_dsl::tensor_ir::{TensorKernelDef, TensorNode, TensorNodeId, TensorOpKind};
use nn_verify::{tensor_kernel_to_graph, TensorParamBinding};

// --- InstanceNorm1d rank guard regression tests (#154) ---

#[test]
fn test_instance_norm_rank1_returns_error() {
    let def = TensorKernelDef::new(
        "instance_norm_rank1",
        vec![
            TensorNode::new(
                TensorNodeId::new(0),
                TensorOpKind::Input {
                    name: "x".to_string(),
                    shape: vec![8],
                },
                vec![8],
            ),
            TensorNode::new(
                TensorNodeId::new(1),
                TensorOpKind::Input {
                    name: "eps".to_string(),
                    shape: vec![1],
                },
                vec![1],
            ),
            TensorNode::new(
                TensorNodeId::new(2),
                TensorOpKind::InstanceNorm1d {
                    input: TensorNodeId::new(0),
                    eps: TensorNodeId::new(1),
                    gamma: None,
                    beta: None,
                    axis: 0,
                },
                vec![8],
            ),
        ],
        TensorNodeId::new(2),
    );
    let bindings = vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantScalar(1e-5),
    ];
    let err = tensor_kernel_to_graph(&def, &bindings)
        .expect_err("rank-1 InstanceNorm1d must be rejected");
    let err_msg = format!("{err:?}");
    assert!(
        err_msg.contains("RankTooLow") || err_msg.contains("rank"),
        "error should indicate rank problem, got: {err_msg}"
    );
}

// --- InstanceNorm1d error path tests (#187 AC4) ---

#[test]
fn test_instance_norm_non_last_axis_rejected() {
    let def = TensorKernelDef::new(
        "instance_norm_bad_axis",
        vec![
            TensorNode::new(
                TensorNodeId::new(0),
                TensorOpKind::Input {
                    name: "x".to_string(),
                    shape: vec![4, 8, 16],
                },
                vec![4, 8, 16],
            ),
            TensorNode::new(
                TensorNodeId::new(1),
                TensorOpKind::Input {
                    name: "eps".to_string(),
                    shape: vec![1],
                },
                vec![1],
            ),
            TensorNode::new(
                TensorNodeId::new(2),
                TensorOpKind::InstanceNorm1d {
                    input: TensorNodeId::new(0),
                    eps: TensorNodeId::new(1),
                    gamma: None,
                    beta: None,
                    axis: 0,
                },
                vec![4, 8, 16],
            ),
        ],
        TensorNodeId::new(2),
    );
    let bindings = vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantScalar(1e-5),
    ];
    let err = tensor_kernel_to_graph(&def, &bindings)
        .expect_err("non-last-axis InstanceNorm1d must be rejected");
    let err_msg = format!("{err:?}");
    assert!(
        err_msg.contains("not the last axis") || err_msg.contains("axis"),
        "error should indicate axis mismatch, got: {err_msg}"
    );
}

#[test]
fn test_instance_norm_variable_eps_rejected() {
    let def = TensorKernelDef::new(
        "instance_norm_var_eps",
        vec![
            TensorNode::new(
                TensorNodeId::new(0),
                TensorOpKind::Input {
                    name: "x".to_string(),
                    shape: vec![2, 4, 16],
                },
                vec![2, 4, 16],
            ),
            TensorNode::new(
                TensorNodeId::new(1),
                TensorOpKind::Input {
                    name: "eps".to_string(),
                    shape: vec![1],
                },
                vec![1],
            ),
            TensorNode::new(
                TensorNodeId::new(2),
                TensorOpKind::InstanceNorm1d {
                    input: TensorNodeId::new(0),
                    eps: TensorNodeId::new(1),
                    gamma: None,
                    beta: None,
                    axis: 2,
                },
                vec![2, 4, 16],
            ),
        ],
        TensorNodeId::new(2),
    );
    let bindings = vec![TensorParamBinding::Variable, TensorParamBinding::Variable];
    let err = tensor_kernel_to_graph(&def, &bindings)
        .expect_err("variable eps for InstanceNorm1d must be rejected");
    let err_msg = format!("{err}");
    assert!(
        err_msg.contains("eps must be constant") || err_msg.contains("eps"),
        "error should indicate eps must be constant, got: {err_msg}"
    );
}

#[test]
fn test_instance_norm_middle_axis_rejected() {
    let def = TensorKernelDef::new(
        "instance_norm_mid_axis",
        vec![
            TensorNode::new(
                TensorNodeId::new(0),
                TensorOpKind::Input {
                    name: "x".to_string(),
                    shape: vec![2, 8, 32],
                },
                vec![2, 8, 32],
            ),
            TensorNode::new(
                TensorNodeId::new(1),
                TensorOpKind::Input {
                    name: "eps".to_string(),
                    shape: vec![1],
                },
                vec![1],
            ),
            TensorNode::new(
                TensorNodeId::new(2),
                TensorOpKind::InstanceNorm1d {
                    input: TensorNodeId::new(0),
                    eps: TensorNodeId::new(1),
                    gamma: None,
                    beta: None,
                    axis: 1,
                },
                vec![2, 8, 32],
            ),
        ],
        TensorNodeId::new(2),
    );
    let bindings = vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantScalar(1e-5),
    ];
    let err = tensor_kernel_to_graph(&def, &bindings)
        .expect_err("middle-axis InstanceNorm1d must be rejected");
    let err_msg = format!("{err:?}");
    assert!(
        err_msg.contains("not the last axis") || err_msg.contains("axis"),
        "error should indicate axis mismatch, got: {err_msg}"
    );
}

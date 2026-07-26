// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Affine InstanceNorm1d NY translation tests (#302).
//!
//! Split from graph_translate_tensor_norm.rs (#423):
//! - Constant gamma/beta translation and IBP propagation
//! - Variable gamma/beta rejection
//! - Constant-input affine folding

use nn_dsl::tensor_ir::{TensorKernelDef, TensorNode, TensorNodeId, TensorOpKind};
use nn_verify::{tensor_kernel_to_graph, BoundedTensor, TensorParamBinding};
use ndarray::{arr1, ArrayD, IxDyn};

#[test]
fn test_instance_norm_affine_constant_gamma_beta_translates() {
    let shape = vec![2, 4, 16];
    let mk_input = |id, name: &str, s: Vec<usize>| {
        TensorNode::new(
            TensorNodeId::new(id),
            TensorOpKind::Input {
                name: name.to_string(),
                shape: s.clone(),
            },
            s,
        )
    };
    let def = TensorKernelDef::new(
        "instance_norm_affine",
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
                shape.clone(),
            ),
        ],
        TensorNodeId::new(4),
    );
    let bindings = vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantScalar(1e-5),
        TensorParamBinding::ConstantScalar(2.0),
        TensorParamBinding::ConstantScalar(10.0),
    ];
    let graph = tensor_kernel_to_graph(&def, &bindings)
        .expect("affine InstanceNorm with constant gamma/beta should translate");

    let lower = ArrayD::from_elem(IxDyn(&shape), -1.0f32);
    let upper = ArrayD::from_elem(IxDyn(&shape), 1.0f32);
    let input = BoundedTensor::new(lower, upper).expect("input bounds");
    let output = graph.propagate_ibp(&input).expect("IBP propagation");
    let (lo, hi) = output.lower_upper();

    assert!(
        lo.iter().all(|v| v.is_finite()),
        "lower bounds must be finite"
    );
    assert!(
        hi.iter().all(|v| v.is_finite()),
        "upper bounds must be finite"
    );
    let hi_max = hi.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    assert!(
        hi_max > 10.0,
        "with beta=10, upper bound should exceed 10.0, got {hi_max}"
    );
}

#[test]
fn test_instance_norm_affine_variable_gamma_rejected() {
    let def = TensorKernelDef::new(
        "instance_norm_var_gamma",
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
                TensorOpKind::Input {
                    name: "gamma".to_string(),
                    shape: vec![4],
                },
                vec![4],
            ),
            TensorNode::new(
                TensorNodeId::new(3),
                TensorOpKind::Input {
                    name: "beta".to_string(),
                    shape: vec![4],
                },
                vec![4],
            ),
            TensorNode::new(
                TensorNodeId::new(4),
                TensorOpKind::InstanceNorm1d {
                    input: TensorNodeId::new(0),
                    eps: TensorNodeId::new(1),
                    axis: 2,
                    gamma: Some(TensorNodeId::new(2)),
                    beta: Some(TensorNodeId::new(3)),
                },
                vec![2, 4, 16],
            ),
        ],
        TensorNodeId::new(4),
    );
    let bindings = vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantScalar(1e-5),
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantScalar(0.0),
    ];
    let err = tensor_kernel_to_graph(&def, &bindings)
        .expect_err("variable gamma for affine InstanceNorm must be rejected");
    let err_msg = format!("{err}");
    assert!(
        err_msg.contains("gamma must be constant"),
        "error should indicate gamma must be constant, got: {err_msg}"
    );
}

#[test]
fn test_instance_norm_affine_variable_beta_rejected() {
    let def = TensorKernelDef::new(
        "instance_norm_var_beta",
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
                TensorOpKind::Input {
                    name: "gamma".to_string(),
                    shape: vec![4],
                },
                vec![4],
            ),
            TensorNode::new(
                TensorNodeId::new(3),
                TensorOpKind::Input {
                    name: "beta".to_string(),
                    shape: vec![4],
                },
                vec![4],
            ),
            TensorNode::new(
                TensorNodeId::new(4),
                TensorOpKind::InstanceNorm1d {
                    input: TensorNodeId::new(0),
                    eps: TensorNodeId::new(1),
                    axis: 2,
                    gamma: Some(TensorNodeId::new(2)),
                    beta: Some(TensorNodeId::new(3)),
                },
                vec![2, 4, 16],
            ),
        ],
        TensorNodeId::new(4),
    );
    let bindings = vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantScalar(1e-5),
        TensorParamBinding::ConstantScalar(1.0),
        TensorParamBinding::Variable,
    ];
    let err = tensor_kernel_to_graph(&def, &bindings)
        .expect_err("variable beta for affine InstanceNorm must be rejected");
    let err_msg = format!("{err}");
    assert!(
        err_msg.contains("beta must be constant"),
        "error should indicate beta must be constant, got: {err_msg}"
    );
}

#[test]
fn test_instance_norm_affine_constant_input_returns_beta() {
    let def = TensorKernelDef::new(
        "instance_norm_const_in_affine",
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
                TensorOpKind::Input {
                    name: "gamma".to_string(),
                    shape: vec![4],
                },
                vec![4],
            ),
            TensorNode::new(
                TensorNodeId::new(3),
                TensorOpKind::Input {
                    name: "beta".to_string(),
                    shape: vec![4],
                },
                vec![4],
            ),
            TensorNode::new(
                TensorNodeId::new(4),
                TensorOpKind::InstanceNorm1d {
                    input: TensorNodeId::new(0),
                    eps: TensorNodeId::new(1),
                    axis: 2,
                    gamma: Some(TensorNodeId::new(2)),
                    beta: Some(TensorNodeId::new(3)),
                },
                vec![2, 4, 16],
            ),
        ],
        TensorNodeId::new(4),
    );
    let bindings = vec![
        TensorParamBinding::ConstantScalar(5.0),
        TensorParamBinding::ConstantScalar(1e-5),
        TensorParamBinding::ConstantScalar(2.0),
        TensorParamBinding::ConstantScalar(10.0),
    ];
    let graph = tensor_kernel_to_graph(&def, &bindings)
        .expect("constant input affine InstanceNorm should translate");

    assert_eq!(
        graph.num_nodes(),
        1,
        "constant input should fold to 1 graph node, got {}",
        graph.num_nodes()
    );
}

/// InstanceNorm1d with per-channel WeightTensor gamma/beta should translate
/// and produce finite IBP bounds, matching the pattern from AdaIN1d (#662).
#[test]
fn test_instance_norm_affine_weight_tensor_gamma_beta_translates() {
    let shape = vec![2, 4, 16];
    let mk_input = |id, name: &str, s: Vec<usize>| {
        TensorNode::new(
            TensorNodeId::new(id),
            TensorOpKind::Input {
                name: name.to_string(),
                shape: s.clone(),
            },
            s,
        )
    };
    let def = TensorKernelDef::new(
        "instance_norm_weight_affine",
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
                shape.clone(),
            ),
        ],
        TensorNodeId::new(4),
    );
    // Per-channel gamma and beta via ConstantTensor (WeightTensor path).
    let gamma_arr = arr1(&[1.0_f32, 2.0, 0.5, 3.0]).into_dyn();
    let beta_arr = arr1(&[0.0_f32, 10.0, -5.0, 7.0]).into_dyn();
    let bindings = vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantScalar(1e-5),
        TensorParamBinding::ConstantTensor(gamma_arr),
        TensorParamBinding::ConstantTensor(beta_arr),
    ];
    let graph = tensor_kernel_to_graph(&def, &bindings)
        .expect("InstanceNorm with WeightTensor gamma/beta should translate");

    let lower = ArrayD::from_elem(IxDyn(&shape), -1.0f32);
    let upper = ArrayD::from_elem(IxDyn(&shape), 1.0f32);
    let input = BoundedTensor::new(lower, upper).expect("input bounds");
    let output = graph.propagate_ibp(&input).expect("IBP propagation");
    let (lo, hi) = output.lower_upper();

    assert!(
        lo.iter().all(|v| v.is_finite()),
        "lower bounds must be finite"
    );
    assert!(
        hi.iter().all(|v| v.is_finite()),
        "upper bounds must be finite"
    );
}

/// InstanceNorm1d WeightTensor with wrong shape must be rejected.
#[test]
fn test_instance_norm_affine_weight_tensor_wrong_shape_rejected() {
    let shape = vec![2, 4, 16];
    let mk_input = |id, name: &str, s: Vec<usize>| {
        TensorNode::new(
            TensorNodeId::new(id),
            TensorOpKind::Input {
                name: name.to_string(),
                shape: s.clone(),
            },
            s,
        )
    };
    let def = TensorKernelDef::new(
        "instance_norm_bad_weight_shape",
        vec![
            mk_input(0, "x", shape.clone()),
            mk_input(1, "eps", vec![1]),
            mk_input(2, "gamma", vec![8]),
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
    // gamma has 8 elements but num_channels is 4 → shape mismatch.
    let gamma_arr = arr1(&[1.0_f32, 2.0, 0.5, 3.0, 1.0, 2.0, 0.5, 3.0]).into_dyn();
    let beta_arr = arr1(&[0.0_f32, 10.0, -5.0, 7.0]).into_dyn();
    let bindings = vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantScalar(1e-5),
        TensorParamBinding::ConstantTensor(gamma_arr),
        TensorParamBinding::ConstantTensor(beta_arr),
    ];
    let err = tensor_kernel_to_graph(&def, &bindings)
        .expect_err("mismatched gamma shape should be rejected");
    let err_msg = format!("{err}");
    assert!(
        err_msg.contains("shape") || err_msg.contains("mismatch"),
        "error should mention shape problem, got: {err_msg}"
    );
}

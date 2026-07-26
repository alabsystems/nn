// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Integration tests for NormBoundsMode::ForwardMode (#744).
//!
//! Verifies that forward-mode IBP through normalization layers produces
//! tighter bounds than conservative mode for the pathological case:
//! high-variance input centers with small perturbation radius. This is the
//! typical real-model scenario (feature activations spread across a wide range,
//! perturbed by a small epsilon for verification).
//!
//! With uniform bounds (same lo/hi for all elements), conservative mode
//! can actually be tighter because all elements have identical uncertainty.
//! The degeneration (~1e10x wider than scalar) occurs when element-wise
//! bounds differ, amplifying the conservative mean/variance estimates.
//!
//! NormBoundsMode config wiring and full-pipeline tests extracted to
//! `graph_translate_tensor_norm_forward_mode_pipeline.rs` (#1402).

use nn_dsl::tensor_ir::{TensorKernelDef, TensorNode, TensorNodeId, TensorOpKind};
use nn_verify::{
    tensor_kernel_to_graph, tensor_kernel_to_graph_with_norm_mode, BoundedTensor, NormBoundsMode,
    TensorParamBinding,
};
use ndarray::{ArrayD, IxDyn};

/// Build an InstanceNorm1d kernel def: input [C=2, T=4], eps=1e-5, axis=last.
fn instance_norm_kernel() -> TensorKernelDef {
    TensorKernelDef::new(
        "instance_norm_test",
        vec![
            TensorNode::new(
                TensorNodeId::new(0),
                TensorOpKind::Input {
                    name: "x".to_string(),
                    shape: vec![2, 4],
                },
                vec![2, 4],
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
                vec![2, 4],
            ),
        ],
        TensorNodeId::new(2),
    )
}

/// Build a LayerNorm kernel def: input [B=1, T=4], weight/bias=[4], eps=1e-5, axis=last.
fn layer_norm_kernel() -> TensorKernelDef {
    TensorKernelDef::new(
        "layer_norm_test",
        vec![
            TensorNode::new(
                TensorNodeId::new(0),
                TensorOpKind::Input {
                    name: "x".to_string(),
                    shape: vec![1, 4],
                },
                vec![1, 4],
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
                    name: "weight".to_string(),
                    shape: vec![4],
                },
                vec![4],
            ),
            TensorNode::new(
                TensorNodeId::new(3),
                TensorOpKind::Input {
                    name: "bias".to_string(),
                    shape: vec![4],
                },
                vec![4],
            ),
            TensorNode::new(
                TensorNodeId::new(4),
                TensorOpKind::LayerNorm {
                    input: TensorNodeId::new(0),
                    eps: TensorNodeId::new(1),
                    weight: TensorNodeId::new(2),
                    bias: TensorNodeId::new(3),
                    axis: 1,
                },
                vec![1, 4],
            ),
        ],
        TensorNodeId::new(4),
    )
}

/// Build an RmsNorm kernel def: input [B=1, T=4], weight=[4], eps=1e-5, axis=last.
fn rms_norm_kernel() -> TensorKernelDef {
    TensorKernelDef::new(
        "rms_norm_test",
        vec![
            TensorNode::new(
                TensorNodeId::new(0),
                TensorOpKind::Input {
                    name: "x".to_string(),
                    shape: vec![1, 4],
                },
                vec![1, 4],
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
                    name: "weight".to_string(),
                    shape: vec![4],
                },
                vec![4],
            ),
            TensorNode::new(
                TensorNodeId::new(3),
                TensorOpKind::RmsNorm {
                    input: TensorNodeId::new(0),
                    eps: TensorNodeId::new(1),
                    weight: TensorNodeId::new(2),
                    axis: 1,
                },
                vec![1, 4],
            ),
        ],
        TensorNodeId::new(3),
    )
}

/// Propagate bounds through a graph and return the max output width.
fn propagate_width(graph: &nn_verify::GraphNetwork, input: &BoundedTensor) -> f32 {
    let output = graph.propagate_ibp(input).expect("IBP propagation");
    output.max_width()
}

/// Create high-variance input bounds with small perturbation radius.
/// Centers: [0, 5, 10, 15, -8, -3, 3, 8], radius: 0.05
/// This is the pattern where forward mode is tighter: high center-point
/// variance means the Jacobian-based first-order term dominates the
/// second-order remainder.
fn high_variance_instance_norm_input() -> BoundedTensor {
    let r = 0.05;
    let lower = ArrayD::from_shape_vec(
        IxDyn(&[2, 4]),
        vec![
            0.0 - r,
            5.0 - r,
            10.0 - r,
            15.0 - r,
            -8.0 - r,
            -3.0 - r,
            3.0 - r,
            8.0 - r,
        ],
    )
    .expect("valid lower");
    let upper = ArrayD::from_shape_vec(
        IxDyn(&[2, 4]),
        vec![
            0.0 + r,
            5.0 + r,
            10.0 + r,
            15.0 + r,
            -8.0 + r,
            -3.0 + r,
            3.0 + r,
            8.0 + r,
        ],
    )
    .expect("valid upper");
    BoundedTensor::new(lower, upper).expect("valid bounded tensor")
}

/// Create high-variance input bounds for LayerNorm/RmsNorm: [B=1, T=4].
fn high_variance_norm_input() -> BoundedTensor {
    let r = 0.05;
    let lower = ArrayD::from_shape_vec(IxDyn(&[1, 4]), vec![0.0 - r, 5.0 - r, 10.0 - r, 15.0 - r])
        .expect("valid lower");
    let upper = ArrayD::from_shape_vec(IxDyn(&[1, 4]), vec![0.0 + r, 5.0 + r, 10.0 + r, 15.0 + r])
        .expect("valid upper");
    BoundedTensor::new(lower, upper).expect("valid bounded tensor")
}

#[test]
fn test_instance_norm_forward_mode_tighter_than_conservative() {
    let kernel = instance_norm_kernel();
    let bindings = vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantScalar(1e-5),
    ];
    let input = high_variance_instance_norm_input();

    let graph_conservative =
        tensor_kernel_to_graph(&kernel, &bindings).expect("conservative graph");
    let graph_forward =
        tensor_kernel_to_graph_with_norm_mode(&kernel, &bindings, NormBoundsMode::ForwardMode)
            .expect("forward-mode graph");

    let width_conservative = propagate_width(&graph_conservative, &input);
    let width_forward = propagate_width(&graph_forward, &input);

    assert!(
        width_conservative.is_finite(),
        "conservative bounds should be finite"
    );
    assert!(
        width_forward.is_finite(),
        "forward-mode bounds should be finite"
    );
    // Forward mode should be tighter for high-variance, small-perturbation inputs.
    assert!(
        width_forward <= width_conservative + 1e-3,
        "forward-mode ({width_forward}) should be no wider than conservative ({width_conservative})"
    );
    // Quantitative check: conservative IBP degenerates by orders of magnitude
    // for high-variance, small-perturbation inputs. Forward mode should produce
    // bounds at least 10x tighter (dvoice #744 reports ~1e10x degeneration;
    // 10x is a conservative threshold that catches misconfiguration).
    if width_conservative > 1.0 && width_forward > 0.0 {
        let ratio = width_conservative / width_forward;
        assert!(
            ratio >= 10.0,
            "forward-mode should be >=10x tighter than conservative for high-variance inputs, \
             got ratio={ratio:.1}x (conservative={width_conservative:.2}, forward={width_forward:.2})"
        );
    }
}

#[test]
fn test_layer_norm_forward_mode_tighter_than_conservative() {
    let kernel = layer_norm_kernel();
    let gamma = ndarray::arr1(&[1.0f32, 1.0, 1.0, 1.0]);
    let beta = ndarray::arr1(&[0.0f32, 0.0, 0.0, 0.0]);
    let bindings = vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantScalar(1e-5),
        TensorParamBinding::ConstantTensor(gamma.into_dyn()),
        TensorParamBinding::ConstantTensor(beta.into_dyn()),
    ];
    let input = high_variance_norm_input();

    let graph_conservative =
        tensor_kernel_to_graph(&kernel, &bindings).expect("conservative graph");
    let graph_forward =
        tensor_kernel_to_graph_with_norm_mode(&kernel, &bindings, NormBoundsMode::ForwardMode)
            .expect("forward-mode graph");

    let width_conservative = propagate_width(&graph_conservative, &input);
    let width_forward = propagate_width(&graph_forward, &input);

    assert!(
        width_conservative.is_finite(),
        "conservative bounds should be finite"
    );
    assert!(
        width_forward.is_finite(),
        "forward-mode bounds should be finite"
    );
    assert!(
        width_forward <= width_conservative + 1e-3,
        "forward-mode ({width_forward}) should be no wider than conservative ({width_conservative})"
    );
    // Quantitative check: see dvoice #744.
    if width_conservative > 1.0 && width_forward > 0.0 {
        let ratio = width_conservative / width_forward;
        assert!(
            ratio >= 10.0,
            "LayerNorm forward-mode should be >=10x tighter, \
             got ratio={ratio:.1}x (conservative={width_conservative:.2}, forward={width_forward:.2})"
        );
    }
}

#[test]
fn test_rms_norm_forward_mode_tighter_than_conservative() {
    let kernel = rms_norm_kernel();
    let weight = ndarray::arr1(&[1.0f32, 1.0, 1.0, 1.0]);
    let bindings = vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantScalar(1e-5),
        TensorParamBinding::ConstantTensor(weight.into_dyn()),
    ];
    let input = high_variance_norm_input();

    let graph_conservative =
        tensor_kernel_to_graph(&kernel, &bindings).expect("conservative graph");
    let graph_forward =
        tensor_kernel_to_graph_with_norm_mode(&kernel, &bindings, NormBoundsMode::ForwardMode)
            .expect("forward-mode graph");

    let width_conservative = propagate_width(&graph_conservative, &input);
    let width_forward = propagate_width(&graph_forward, &input);

    assert!(
        width_conservative.is_finite(),
        "conservative bounds should be finite"
    );
    assert!(
        width_forward.is_finite(),
        "forward-mode bounds should be finite"
    );
    assert!(
        width_forward <= width_conservative + 1e-3,
        "forward-mode ({width_forward}) should be no wider than conservative ({width_conservative})"
    );
    // Quantitative check: see dvoice #744.
    if width_conservative > 1.0 && width_forward > 0.0 {
        let ratio = width_conservative / width_forward;
        assert!(
            ratio >= 10.0,
            "RmsNorm forward-mode should be >=10x tighter, \
             got ratio={ratio:.1}x (conservative={width_conservative:.2}, forward={width_forward:.2})"
        );
    }
}

// NormBoundsMode config wiring and full-pipeline tests extracted to
// graph_translate_tensor_norm_forward_mode_pipeline.rs (#1402).
#[path = "graph_translate_tensor_norm_forward_mode_pipeline.rs"]
mod pipeline_tests;

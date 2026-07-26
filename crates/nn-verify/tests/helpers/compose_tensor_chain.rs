// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Integration tests: tensor-level composition of Conv1d + Snake + InstanceNorm1d.
//!
//! Validates that a multi-op `TensorKernelDef` chains through `tensor_kernel_to_graph`
//! and produces a single NY `GraphNetwork` where IBP and CROWN bounds
//! propagate end-to-end.
//!
//! The dvoice Demucs encoder block is: Conv1d -> Snake activation -> InstanceNorm.
//! Each op works individually. These tests prove they compose correctly.
//!
//! Two-layer composition tests are in `compose_tensor_chain_two_layer.rs`.

use super::common::{assert_bounds_valid, assert_crown_tighter_than_ibp, uniform_bounds};
use nn_dsl::adain::build_snake_scalar_kernel;
use nn_dsl::tensor_ir::{
    BroadcastAlignment, TensorKernelDef, TensorNode, TensorNodeId, TensorOpKind,
};
use nn_verify::{
    propagate_with_crown_fallback, tensor_kernel_to_graph, PropMethod, TensorParamBinding,
};
use ndarray::{ArrayD, IxDyn};

// ---------------------------------------------------------------------------
// Shared reference implementations for soundness tests
// ---------------------------------------------------------------------------

/// Snake activation reference: x + sin(x)^2 with alpha=1.
fn snake_ref(x: f32) -> f32 {
    x + (x.sin()).powi(2)
}

/// InstanceNorm reference: (x - mean) / std, computed over the spatial dimension.
fn instance_norm_ref(vals: &[f32]) -> Vec<f32> {
    let n = vals.len() as f32;
    let mean = vals.iter().sum::<f32>() / n;
    let var = vals.iter().map(|&v| (v - mean).powi(2)).sum::<f32>() / n;
    let std = (var + 1e-5_f32).sqrt();
    vals.iter().map(|&v| (v - mean) / std).collect()
}

// ---------------------------------------------------------------------------
// Single-block Demucs encoder builder
// ---------------------------------------------------------------------------

/// Build a Demucs encoder block as a multi-op TensorKernelDef.
///
/// Nodes: data, weight, alpha, eps, Conv1d, Broadcast(alpha), Snake, InstanceNorm1d.
fn build_demucs_block(
    in_channels: usize,
    out_channels: usize,
    kernel_size: usize,
    in_length: usize,
    stride: usize,
    padding: usize,
) -> TensorKernelDef {
    let out_length = (in_length + 2 * padding - kernel_size) / stride + 1;
    let out_shape = vec![out_channels, out_length];
    let snake_kernel = build_snake_scalar_kernel().expect("snake kernel must build");

    let nodes = vec![
        TensorNode::new(
            TensorNodeId::new(0),
            TensorOpKind::Input {
                name: "data".to_string(),
                shape: vec![in_channels, in_length],
            },
            vec![in_channels, in_length],
        ),
        TensorNode::new(
            TensorNodeId::new(1),
            TensorOpKind::Input {
                name: "weight".to_string(),
                shape: vec![out_channels, in_channels, kernel_size],
            },
            vec![out_channels, in_channels, kernel_size],
        ),
        TensorNode::new(
            TensorNodeId::new(2),
            TensorOpKind::Input {
                name: "alpha".to_string(),
                shape: vec![1],
            },
            vec![1],
        ),
        TensorNode::new(
            TensorNodeId::new(3),
            TensorOpKind::Input {
                name: "eps".to_string(),
                shape: vec![1],
            },
            vec![1],
        ),
        TensorNode::new(
            TensorNodeId::new(4),
            TensorOpKind::Conv1d {
                input: TensorNodeId::new(0),
                weight: TensorNodeId::new(1),
                bias: None,
                stride,
                padding,
                dilation: 1,
                groups: 1,
            },
            out_shape.clone(),
        ),
        TensorNode::new(
            TensorNodeId::new(5),
            TensorOpKind::Broadcast {
                input: TensorNodeId::new(2),
                target_shape: out_shape.clone(),
                alignment: BroadcastAlignment::Right,
            },
            out_shape.clone(),
        ),
        TensorNode::new(
            TensorNodeId::new(6),
            TensorOpKind::Elementwise {
                kernel: snake_kernel,
                inputs: vec![TensorNodeId::new(4), TensorNodeId::new(5)],
            },
            out_shape.clone(),
        ),
        TensorNode::new(
            TensorNodeId::new(7),
            TensorOpKind::InstanceNorm1d {
                input: TensorNodeId::new(6),
                eps: TensorNodeId::new(3),
                axis: 1,
                gamma: None,
                beta: None,
            },
            out_shape,
        ),
    ];

    TensorKernelDef::new("demucs_block", nodes, TensorNodeId::new(7))
}

// ---------------------------------------------------------------------------
// IBP tests
// ---------------------------------------------------------------------------

/// Multi-op TensorKernelDef translates into a valid NY GraphNetwork.
#[test]
fn test_demucs_block_graph_builds() {
    let def = build_demucs_block(1, 4, 3, 16, 1, 1);
    assert_eq!(def.nodes.last().unwrap().shape, vec![4, 16]);
    assert_eq!(def.nodes.len(), 8);

    let weight = ArrayD::from_elem(IxDyn(&[4, 1, 3]), 0.1f32);
    let bindings = vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(weight),
        TensorParamBinding::ConstantScalar(1.0),
        TensorParamBinding::ConstantScalar(1e-5),
    ];

    let graph = tensor_kernel_to_graph(&def, &bindings).expect("Demucs block graph must build");
    assert!(graph.num_nodes() >= 3);
}

/// IBP bounds propagate through Conv1d + Snake + InstanceNorm1d.
#[test]
fn test_demucs_block_ibp_bounds_propagate() {
    let def = build_demucs_block(1, 4, 3, 16, 1, 1);

    let weight = ArrayD::from_elem(IxDyn(&[4, 1, 3]), 0.1f32);
    let bindings = vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(weight),
        TensorParamBinding::ConstantScalar(1.0),
        TensorParamBinding::ConstantScalar(1e-5),
    ];

    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");

    let input = uniform_bounds(&[1, 16], 1.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through Demucs block");
    let (lo, _hi) = output.lower_upper();

    assert_eq!(lo.shape(), &[4, 16]);
    assert_bounds_valid(&output);
}

/// Dvoice-scale: Conv1d(1->48, k=8, stride=4, pad=2) + Snake + InstanceNorm.
#[test]
fn test_demucs_block_dvoice_scale_ibp() {
    let def = build_demucs_block(1, 48, 8, 64, 4, 2);
    assert_eq!(def.nodes.last().unwrap().shape, vec![48, 16]);

    let weight = ArrayD::from_elem(IxDyn(&[48, 1, 8]), 0.01f32);
    let bindings = vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(weight),
        TensorParamBinding::ConstantScalar(1.0),
        TensorParamBinding::ConstantScalar(1e-5),
    ];

    let graph = tensor_kernel_to_graph(&def, &bindings).expect("dvoice graph");

    let input = uniform_bounds(&[1, 64], 1.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through dvoice block");
    let (lo, _hi) = output.lower_upper();

    assert_eq!(lo.shape(), &[48, 16]);
    assert_bounds_valid(&output);
}

/// Concrete forward pass lies within IBP bounds for the composed chain.
#[test]
fn test_demucs_block_ibp_soundness_concrete() {
    let weight_data = vec![0.3f32, -0.5, 0.2, 0.4]; // [2, 1, 2]
    let weight = ArrayD::from_shape_vec(IxDyn(&[2, 1, 2]), weight_data).unwrap();

    let def = build_demucs_block(1, 2, 2, 4, 1, 0);
    assert_eq!(def.nodes.last().unwrap().shape, vec![2, 3]);

    let bindings = vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(weight),
        TensorParamBinding::ConstantScalar(1.0),
        TensorParamBinding::ConstantScalar(1e-5),
    ];

    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");

    let input = uniform_bounds(&[1, 4], 1.0);

    let output = graph.propagate_ibp(&input).expect("IBP");
    let (lo, hi) = output.lower_upper();

    // Manual forward for x=[0.5, -0.3, 0.8, -0.1]:
    // Conv1d [2,3]: oc0=[0.30, -0.49, 0.29], oc1=[-0.02, 0.26, 0.12]
    let conv_oc0 = [0.30f32, -0.49, 0.29];
    let conv_oc1 = [-0.02f32, 0.26, 0.12];

    let snake_oc0: Vec<f32> = conv_oc0.iter().map(|&x| snake_ref(x)).collect();
    let snake_oc1: Vec<f32> = conv_oc1.iter().map(|&x| snake_ref(x)).collect();

    let norm_oc0 = instance_norm_ref(&snake_oc0);
    let norm_oc1 = instance_norm_ref(&snake_oc1);

    for (t, val) in norm_oc0.iter().enumerate() {
        assert!(lo[[0, t]] <= *val + 0.01, "oc0 t={t}: lo > fwd");
        assert!(hi[[0, t]] >= *val - 0.01, "oc0 t={t}: hi < fwd");
    }
    for (t, val) in norm_oc1.iter().enumerate() {
        assert!(lo[[1, t]] <= *val + 0.01, "oc1 t={t}: lo > fwd");
        assert!(hi[[1, t]] >= *val - 0.01, "oc1 t={t}: hi < fwd");
    }
}

// ---------------------------------------------------------------------------
// CROWN tests
// ---------------------------------------------------------------------------

/// CROWN propagation through the full composed chain (Conv1d + Snake + InstanceNorm).
///
/// CROWN backward requires pre-activation bounds for nonlinear layers (Snake,
/// InstanceNorm). The graph-level CROWN first runs IBP forward, then backward.
/// This test verifies CROWN either succeeds with tighter bounds or falls back
/// to IBP with valid bounds.
#[test]
fn test_demucs_block_crown_propagates() {
    let def = build_demucs_block(1, 4, 3, 16, 1, 1);

    let weight = ArrayD::from_elem(IxDyn(&[4, 1, 3]), 0.1f32);
    let bindings = vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(weight),
        TensorParamBinding::ConstantScalar(1.0),
        TensorParamBinding::ConstantScalar(1e-5),
    ];

    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");

    let input = uniform_bounds(&[1, 16], 1.0);

    let (method, output, fallback_reason) =
        propagate_with_crown_fallback(&graph, &input).expect("CROWN/fallback");

    let (lo, _hi) = output.lower_upper();
    assert_eq!(lo.shape(), &[4, 16]);
    assert_bounds_valid(&output);

    // Record which method succeeded for diagnostic clarity.
    match method {
        PropMethod::Crown => assert!(fallback_reason.is_none()),
        PropMethod::Ibp => {
            // IBP fallback is acceptable — InstanceNorm may not support
            // CROWN backward in all NY versions.
            assert!(fallback_reason.is_some());
        }
        _ => panic!("unexpected PropMethod variant"),
    }
}

/// CROWN bounds should be at least as tight as IBP for the composed chain.
#[test]
fn test_demucs_block_crown_tighter_than_ibp() {
    let def = build_demucs_block(1, 2, 3, 8, 1, 1);

    let weight = ArrayD::from_elem(IxDyn(&[2, 1, 3]), 0.1f32);
    let bindings = vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(weight),
        TensorParamBinding::ConstantScalar(1.0),
        TensorParamBinding::ConstantScalar(1e-5),
    ];

    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");

    let input = uniform_bounds(&[1, 8], 1.0);

    // IBP
    let ibp_output = graph.propagate_ibp(&input).expect("IBP");

    // CROWN (with fallback)
    let (method, crown_output, _) =
        propagate_with_crown_fallback(&graph, &input).expect("CROWN/fallback");

    assert_crown_tighter_than_ibp(&crown_output, &ibp_output);

    if method == PropMethod::Ibp {
        let (crown_lo, crown_hi) = crown_output.lower_upper();
        let (ibp_lo, ibp_hi) = ibp_output.lower_upper();
        let tol = 1e-4;
        for ((&cl, &il), (&cu, &iu)) in crown_lo
            .iter()
            .zip(ibp_lo.iter())
            .zip(crown_hi.iter().zip(ibp_hi.iter()))
        {
            assert!((cl - il).abs() < tol, "fallback lo mismatch: {cl} vs {il}");
            assert!((cu - iu).abs() < tol, "fallback hi mismatch: {cu} vs {iu}");
        }
    }
}

/// CROWN soundness: concrete forward pass lies within CROWN/fallback bounds.
#[test]
fn test_demucs_block_crown_soundness_concrete() {
    let weight_data = vec![0.3f32, -0.5, 0.2, 0.4]; // [2, 1, 2]
    let weight = ArrayD::from_shape_vec(IxDyn(&[2, 1, 2]), weight_data).unwrap();

    let def = build_demucs_block(1, 2, 2, 4, 1, 0);
    let bindings = vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(weight),
        TensorParamBinding::ConstantScalar(1.0),
        TensorParamBinding::ConstantScalar(1e-5),
    ];

    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");

    let input = uniform_bounds(&[1, 4], 1.0);

    let (_, output, _) = propagate_with_crown_fallback(&graph, &input).expect("CROWN/fallback");
    let (lo, hi) = output.lower_upper();

    // Same concrete forward as IBP soundness test.
    let conv_oc0 = [0.30f32, -0.49, 0.29];
    let conv_oc1 = [-0.02f32, 0.26, 0.12];

    let snake_oc0: Vec<f32> = conv_oc0.iter().map(|&x| snake_ref(x)).collect();
    let snake_oc1: Vec<f32> = conv_oc1.iter().map(|&x| snake_ref(x)).collect();

    let norm_oc0 = instance_norm_ref(&snake_oc0);
    let norm_oc1 = instance_norm_ref(&snake_oc1);

    for (t, val) in norm_oc0.iter().enumerate() {
        assert!(lo[[0, t]] <= *val + 0.01, "oc0 t={t}: CROWN lo > fwd");
        assert!(hi[[0, t]] >= *val - 0.01, "oc0 t={t}: CROWN hi < fwd");
    }
    for (t, val) in norm_oc1.iter().enumerate() {
        assert!(lo[[1, t]] <= *val + 0.01, "oc1 t={t}: CROWN lo > fwd");
        assert!(hi[[1, t]] >= *val - 0.01, "oc1 t={t}: CROWN hi < fwd");
    }
}

/// Dvoice-scale CROWN propagation: Conv1d(1->48, k=8, stride=4, pad=2).
#[test]
fn test_demucs_block_dvoice_scale_crown() {
    let def = build_demucs_block(1, 48, 8, 64, 4, 2);

    let weight = ArrayD::from_elem(IxDyn(&[48, 1, 8]), 0.01f32);
    let bindings = vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(weight),
        TensorParamBinding::ConstantScalar(1.0),
        TensorParamBinding::ConstantScalar(1e-5),
    ];

    let graph = tensor_kernel_to_graph(&def, &bindings).expect("dvoice graph");

    let input = uniform_bounds(&[1, 64], 1.0);

    let (_, output, _) =
        propagate_with_crown_fallback(&graph, &input).expect("dvoice CROWN/fallback");
    let (lo, _hi) = output.lower_upper();

    assert_eq!(lo.shape(), &[48, 16]);
    assert_bounds_valid(&output);
}

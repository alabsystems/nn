// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Integration tests for `certify_auto_fusion_from_graph()` — the certificate
//! emission pipeline.
//!
//! Verifies that CROWN-conclusive fusion proofs produce valid, serializable
//! `FusionEquivalenceCertificate`s alongside the verification results.
//!
//! Part of #2127 (Wave 4: Auto fusion verification — AC 3: certificates).

use nn_core::dyn_tensor::trace::{ComputationGraph, TraceNode, TraceOp};
use nn_core::DType;
use nn_verify::certify_auto_fusion_from_graph;

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

fn op_node(id: u64, op: TraceOp, inputs: &[u64], shape: &[usize]) -> TraceNode {
    TraceNode::new(
        id,
        format!("{}_{id}", op.canonical_name()),
        op,
        inputs.to_vec(),
        shape.to_vec(),
        DType::F32,
    )
}

/// Certify Exp → Relu chain — produces exactly one certificate.
#[test]
fn test_certify_exp_relu_one_certificate() {
    let graph = ComputationGraph::from_nodes(vec![
        input_node(0, &[1, 4, 8]),
        op_node(1, TraceOp::Exp, &[0], &[1, 4, 8]),
        op_node(2, TraceOp::Relu, &[1], &[1, 4, 8]),
    ]);

    let result = certify_auto_fusion_from_graph(&graph, &[(1.0, 1.0)], 1e-5, 512)
        .expect("certify should succeed");

    assert_eq!(result.verification.chains_detected, 1);
    assert_eq!(result.verification.conclusive_count, 1);
    assert_eq!(
        result.certificates.len(),
        1,
        "one conclusive → one certificate"
    );

    let cert = &result.certificates[0];
    assert!(
        cert.proves_equivalence(),
        "certificate should prove equivalence"
    );
    assert_eq!(cert.dimension, 512);
    assert_eq!(cert.epsilon, 1e-5);
    assert_eq!(cert.variable_bounds.len(), 1);
    assert_eq!(cert.variable_bounds[0], (1.0, 1.0));
}

/// Certificate validates and serializes to JSON.
#[test]
fn test_certify_certificate_serialization() {
    let graph = ComputationGraph::from_nodes(vec![
        input_node(0, &[1, 4, 8]),
        op_node(1, TraceOp::Exp, &[0], &[1, 4, 8]),
        op_node(2, TraceOp::Relu, &[1], &[1, 4, 8]),
    ]);

    let result = certify_auto_fusion_from_graph(&graph, &[(0.5, 0.5)], 1e-5, 256)
        .expect("certify should succeed");

    assert!(!result.certificates.is_empty());
    let cert = &result.certificates[0];

    // Validate internal consistency.
    cert.validate().expect("certificate should be valid");

    // Serialize to JSON and back.
    let json = cert.to_json().expect("JSON serialization");
    let deser: nn_verify::FusionEquivalenceCertificate =
        serde_json::from_str(&json).expect("JSON deserialization");
    assert_eq!(deser.dimension, 256);
    assert!(deser.proves_equivalence());
}

/// No fusible chains → no certificates, empty result.
#[test]
fn test_certify_no_chains_no_certificates() {
    let graph = ComputationGraph::from_nodes(vec![input_node(0, &[2, 4])]);

    let result =
        certify_auto_fusion_from_graph(&graph, &[], 1e-5, 512).expect("certify should succeed");

    assert_eq!(result.verification.chains_detected, 0);
    assert!(result.certificates.is_empty());
}

/// Multiple disjoint chains produce multiple certificates.
#[test]
fn test_certify_multiple_chains_multiple_certificates() {
    let shape = &[1, 4, 8];
    let graph = ComputationGraph::from_nodes(vec![
        input_node(0, shape),
        op_node(1, TraceOp::Exp, &[0], shape),
        op_node(2, TraceOp::Relu, &[1], shape),
        input_node(10, shape),
        op_node(11, TraceOp::Sigmoid, &[10], shape),
        op_node(12, TraceOp::Mul, &[11, 10], shape),
    ]);

    let result = certify_auto_fusion_from_graph(&graph, &[(0.5, 0.5)], 1e-5, 768)
        .expect("certify should succeed");

    assert_eq!(result.verification.chains_detected, 2);
    assert_eq!(
        result.certificates.len(),
        result.verification.conclusive_count,
        "certificate count matches conclusive count"
    );
    for cert in &result.certificates {
        assert!(cert.proves_equivalence());
        assert_eq!(cert.dimension, 768);
        cert.validate().expect("certificate should be valid");
    }
}

/// Certificate records the correct sequential kernel names from the spec.
#[test]
fn test_certify_sequential_kernel_names() {
    let graph = ComputationGraph::from_nodes(vec![
        input_node(0, &[1, 4, 8]),
        op_node(1, TraceOp::Exp, &[0], &[1, 4, 8]),
        op_node(2, TraceOp::Relu, &[1], &[1, 4, 8]),
    ]);

    let result = certify_auto_fusion_from_graph(&graph, &[(1.0, 1.0)], 1e-5, 512)
        .expect("certify should succeed");

    let cert = &result.certificates[0];
    // The sequential names should be the first and second kernel names
    // from the auto-generated spec (not empty).
    assert!(
        !cert.sequential_names.0.is_empty(),
        "first name should not be empty"
    );
    assert!(
        !cert.sequential_names.1.is_empty(),
        "second name should not be empty"
    );
    assert!(
        !cert.fused_kernel_name.is_empty(),
        "fused name should not be empty"
    );
}

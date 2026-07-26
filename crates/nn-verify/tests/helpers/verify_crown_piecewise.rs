// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! CROWN method selection tests for piecewise activations (#40 Phase 2).
//!
//! Verifies CROWN linear relaxation is selected for specialized layers
//! (ReLU, ClipLayer, LeakyReLU). Decomposition and fallback tests split
//! to verify_crown_piecewise_decomp.rs (#423).

use nn_dsl::ir::{BinOpKind, CompareOpKind, MinMaxKind};
use nn_dsl::{IRNode, IRNodeKind, KernelDef, NodeId, Param, ScalarType};
use nn_verify::{scalar_input_bounds, VerifyConfig, VerifyRequest};

// --- Kernel IR helpers ---

fn relu_kernel_ir() -> KernelDef {
    KernelDef::new(
        "relu",
        vec![Param::new("x", ScalarType::F32)],
        ScalarType::F32,
        vec![
            IRNode::new(NodeId::new(0), IRNodeKind::Param(0)),
            IRNode::new(NodeId::new(1), IRNodeKind::Literal(0.0)),
            IRNode::new(
                NodeId::new(2),
                IRNodeKind::MinMax {
                    op: MinMaxKind::Max,
                    lhs: NodeId::new(0),
                    rhs: NodeId::new(1),
                },
            ),
        ],
        NodeId::new(2),
    )
}

fn clamp_kernel_ir() -> KernelDef {
    KernelDef::new(
        "clamp",
        vec![Param::new("x", ScalarType::F32)],
        ScalarType::F32,
        vec![
            IRNode::new(NodeId::new(0), IRNodeKind::Param(0)),
            IRNode::new(NodeId::new(1), IRNodeKind::Literal(-1.0)),
            IRNode::new(NodeId::new(2), IRNodeKind::Literal(1.0)),
            IRNode::new(
                NodeId::new(3),
                IRNodeKind::Clamp {
                    input: NodeId::new(0),
                    min: NodeId::new(1),
                    max: NodeId::new(2),
                },
            ),
        ],
        NodeId::new(3),
    )
}

fn leaky_relu_kernel_ir() -> KernelDef {
    KernelDef::new(
        "leaky_relu",
        vec![Param::new("x", ScalarType::F32)],
        ScalarType::F32,
        vec![
            IRNode::new(NodeId::new(0), IRNodeKind::Param(0)),
            IRNode::new(NodeId::new(1), IRNodeKind::Literal(0.0)),
            IRNode::new(
                NodeId::new(2),
                IRNodeKind::Compare {
                    op: CompareOpKind::Gt,
                    lhs: NodeId::new(0),
                    rhs: NodeId::new(1),
                },
            ),
            IRNode::new(NodeId::new(3), IRNodeKind::Literal(0.01)),
            IRNode::new(
                NodeId::new(4),
                IRNodeKind::BinOp {
                    op: BinOpKind::Mul,
                    lhs: NodeId::new(3),
                    rhs: NodeId::new(0),
                },
            ),
            IRNode::new(
                NodeId::new(5),
                IRNodeKind::Select {
                    cond: NodeId::new(2),
                    then_val: NodeId::new(0),
                    else_val: NodeId::new(4),
                },
            ),
        ],
        NodeId::new(5),
    )
}

// --- CROWN method selection tests ---

#[test]
fn test_relu_crown_propagation_succeeds() {
    // ReLU maps to ReLULayer which has native CROWN linear relaxation.
    let kernel = relu_kernel_ir();
    let input_bounds = scalar_input_bounds(-5.0, 5.0).expect("bounds");
    let config = VerifyConfig::with_threshold(0.0).expect("valid threshold");

    let result = VerifyRequest::new(&kernel)
        .constant_params(&[])
        .input_bounds(&input_bounds)
        .config(config)
        .verify_bounds()
        .expect("CROWN propagation should succeed for ReLU");

    // CROWN-family (Crown or the strictly-tighter AlphaCrown) — escalation
    // tries alpha-CROWN first and reports AlphaCrown on success, so assert the
    // family via is_tight() rather than `== Crown` (#3344). A None fallback
    // reason confirms it did NOT fall back to IBP.
    assert!(
        result.method.is_tight(),
        "ReLU should use a CROWN-family method (not fall back to IBP): \
         method = {:?}, fallback reason = {:?}",
        result.method,
        result.crown_fallback_reason,
    );
    assert!(result.is_finite, "ReLU bounds should be finite");
    // relu(x) with x in [-5, 5] → [0, 5]. Check both sides for tightness.
    assert!(
        result.output_lower >= -0.01,
        "relu lower should be >= 0, got {}",
        result.output_lower
    );
    assert!(
        result.output_lower <= 0.01,
        "relu lower should be tight near 0, got {}",
        result.output_lower
    );
    assert!(
        result.output_upper >= 4.99,
        "relu upper should be >= 5, got {}",
        result.output_upper
    );
    assert!(
        result.output_upper <= 5.01,
        "relu upper should be tight near 5, got {}",
        result.output_upper
    );
    assert!(result.crown_fallback_reason.is_none());
}

#[test]
fn test_clamp_crown_propagation_succeeds() {
    // ClipLayer has CROWN linear relaxation via clip_linear_relaxation.
    let kernel = clamp_kernel_ir();
    let input_bounds = scalar_input_bounds(-5.0, 5.0).expect("bounds");
    let config = VerifyConfig::with_threshold(0.0).expect("valid threshold");

    let result = VerifyRequest::new(&kernel)
        .constant_params(&[])
        .input_bounds(&input_bounds)
        .config(config)
        .verify_bounds()
        .expect("CROWN propagation should succeed for Clamp");

    // CROWN-family (Crown or AlphaCrown) — see is_tight() note above (#3344).
    assert!(
        result.method.is_tight(),
        "Clamp should use a CROWN-family method: method = {:?}, fallback reason = {:?}",
        result.method,
        result.crown_fallback_reason,
    );
    assert!(result.is_finite);
    assert!(
        result.output_lower >= -1.01,
        "clamp lower should be >= -1, got {}",
        result.output_lower
    );
    assert!(
        result.output_upper <= 1.01,
        "clamp upper should be <= 1, got {}",
        result.output_upper
    );
    assert!(result.crown_fallback_reason.is_none());
}

#[test]
fn test_leaky_relu_crown_propagation_succeeds() {
    // LeakyReLULayer has CROWN with pre-activation bounds.
    let kernel = leaky_relu_kernel_ir();
    let input_bounds = scalar_input_bounds(-10.0, 10.0).expect("bounds");
    let config = VerifyConfig::with_threshold(0.0).expect("valid threshold");

    let result = VerifyRequest::new(&kernel)
        .constant_params(&[])
        .input_bounds(&input_bounds)
        .config(config)
        .verify_bounds()
        .expect("CROWN propagation should succeed for LeakyReLU");

    // CROWN-family (Crown or AlphaCrown) — see is_tight() note above (#3344).
    assert!(
        result.method.is_tight(),
        "LeakyReLU should use a CROWN-family method: method = {:?}, fallback reason = {:?}",
        result.method,
        result.crown_fallback_reason,
    );
    assert!(result.is_finite);
    // leaky_relu(x, 0.01) with x in [-10, 10] → [-0.1, 10]. Check both sides.
    assert!(
        result.output_lower <= -0.09,
        "leaky_relu lower should be <= -0.1, got {}",
        result.output_lower
    );
    assert!(
        result.output_lower >= -0.11,
        "leaky_relu lower should be tight near -0.1, got {}",
        result.output_lower
    );
    assert!(
        result.output_upper >= 9.99,
        "leaky_relu upper should be >= 10, got {}",
        result.output_upper
    );
    assert!(
        result.output_upper <= 10.01,
        "leaky_relu upper should be tight near 10, got {}",
        result.output_upper
    );
    assert!(result.crown_fallback_reason.is_none());
}

// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! CROWN decomposition and fallback tests for piecewise activations (#40 Phase 2).
//!
//! Split from verify_crown_piecewise.rs (#423):
//! - ReLU decomposition CROWN vs IBP tightness
//! - min decomposition CROWN propagation
//! - Variable-variable MinMax IBP fallback
//! - Generic Select → WhereLayer fallback

use nn_dsl::ir::{BinOpKind, CompareOpKind, MinMaxKind};
use nn_dsl::{IRNode, IRNodeKind, KernelDef, NodeId, Param, ScalarType};
use nn_verify::{scalar_input_bounds, ParamBinding, PropMethod, VerifyConfig, VerifyRequest};

fn minmax_kernel_ir(op: MinMaxKind, constant: f64) -> KernelDef {
    let name = match op {
        MinMaxKind::Max => format!("max_x_{constant}"),
        MinMaxKind::Min => format!("min_x_{constant}"),
        _ => "minmax".to_string(),
    };
    KernelDef::new(
        name,
        vec![Param::new("x", ScalarType::F32)],
        ScalarType::F32,
        vec![
            IRNode::new(NodeId::new(0), IRNodeKind::Param(0)),
            IRNode::new(NodeId::new(1), IRNodeKind::Literal(constant)),
            IRNode::new(
                NodeId::new(2),
                IRNodeKind::MinMax {
                    op,
                    lhs: NodeId::new(0),
                    rhs: NodeId::new(1),
                },
            ),
        ],
        NodeId::new(2),
    )
}

// --- ReLU decomposition CROWN vs IBP tightness ---

#[test]
fn test_relu_decomposition_crown_vs_ibp_tightness() {
    // max(x, 3.0) decomposes to SubConstant(3) → ReLU → AddConstant(3).
    let kernel = minmax_kernel_ir(MinMaxKind::Max, 3.0);
    let input_bounds = scalar_input_bounds(1.0, 5.0).expect("bounds");

    let ibp_result = VerifyRequest::new(&kernel)
        .constant_params(&[])
        .input_bounds(&input_bounds)
        .config(VerifyConfig::with_threshold(1e10).expect("valid threshold"))
        .verify_bounds()
        .expect("IBP");
    assert_eq!(ibp_result.method, PropMethod::Ibp);

    let crown_result = VerifyRequest::new(&kernel)
        .constant_params(&[])
        .input_bounds(&input_bounds)
        .config(VerifyConfig::with_threshold(0.0).expect("valid threshold"))
        .verify_bounds()
        .expect("CROWN");

    // CROWN-family (Crown or the strictly-tighter AlphaCrown). Escalation tries
    // alpha-CROWN first and reports AlphaCrown on success; assert the family via
    // is_tight() rather than `== Crown` (#3344). The width assertion below is the
    // soundness/tightness check: CROWN must be no looser than IBP.
    assert!(
        crown_result.method.is_tight(),
        "max(x,3) via relu decomposition should support a CROWN-family method: \
         method = {:?}, fallback = {:?}",
        crown_result.method,
        crown_result.crown_fallback_reason,
    );
    // max(x,3) with x in [1,5] → [3, 5]. Check both sides for tightness.
    assert!(
        crown_result.output_lower >= 2.99,
        "max(x,3) lower should be >= 3, got {}",
        crown_result.output_lower
    );
    assert!(
        crown_result.output_lower <= 3.01,
        "max(x,3) lower should be tight near 3, got {}",
        crown_result.output_lower
    );
    assert!(
        crown_result.output_upper >= 4.99,
        "max(x,3) upper should be >= 5, got {}",
        crown_result.output_upper
    );
    assert!(
        crown_result.output_upper <= 5.01,
        "max(x,3) upper should be tight near 5, got {}",
        crown_result.output_upper
    );
    assert!(
        crown_result.output_width <= ibp_result.output_width + 1e-6,
        "CROWN width {} should be <= IBP width {}",
        crown_result.output_width,
        ibp_result.output_width,
    );
}

#[test]
fn test_min_decomposition_crown_succeeds() {
    // min(x, 2.0) = 2 - relu(2 - x).
    let kernel = minmax_kernel_ir(MinMaxKind::Min, 2.0);
    let input_bounds = scalar_input_bounds(-1.0, 5.0).expect("bounds");
    let config = VerifyConfig::with_threshold(0.0).expect("valid threshold");

    let result = VerifyRequest::new(&kernel)
        .constant_params(&[])
        .input_bounds(&input_bounds)
        .config(config)
        .verify_bounds()
        .expect("CROWN propagation should succeed for min decomposition");

    // CROWN-family (Crown or AlphaCrown) — see is_tight() note above (#3344).
    assert!(
        result.method.is_tight(),
        "min(x,2) decomposition should support a CROWN-family method: \
         method = {:?}, fallback = {:?}",
        result.method,
        result.crown_fallback_reason,
    );
    assert!(result.is_finite);
    // min(x, 2) with x in [-1, 5] → [-1, 2]. Check both sides for tightness.
    assert!(
        result.output_lower <= -0.99,
        "min(x,2) lower should be <= -1, got {}",
        result.output_lower
    );
    assert!(
        result.output_lower >= -1.01,
        "min(x,2) lower should be tight near -1, got {}",
        result.output_lower
    );
    assert!(
        result.output_upper >= 1.99,
        "min(x,2) upper should be >= 2, got {}",
        result.output_upper
    );
    assert!(
        result.output_upper <= 2.01,
        "min(x,2) upper should be tight near 2, got {}",
        result.output_upper
    );
}

// --- Variable-variable MinMax: expected IBP fallback ---

#[test]
fn test_var_var_max_crown_fallback_documented() {
    // max(x, y) with both variable uses MaxBinaryLayer.
    // MaxBinaryLayer has IBP but may lack CROWN backward dispatch.
    let kernel = KernelDef::new(
        "max_xy",
        vec![
            Param::new("x", ScalarType::F32),
            Param::new("y", ScalarType::F32),
        ],
        ScalarType::F32,
        vec![
            IRNode::new(NodeId::new(0), IRNodeKind::Param(0)),
            IRNode::new(NodeId::new(1), IRNodeKind::Param(1)),
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
    );

    let bindings = vec![ParamBinding::Variable, ParamBinding::Variable];
    let config = VerifyConfig::with_threshold(0.0).expect("valid threshold");

    let result = VerifyRequest::new(&kernel)
        .bindings(&bindings)
        .variable_bounds(&[(-5.0, 5.0), (-5.0, 5.0)])
        .config(config)
        .verify_bounds()
        .expect("verification should succeed via IBP or CROWN");

    assert!(result.is_finite, "bounds should be finite");
    // max(x, y) with both in [-5, 5] → [-5, 5]
    assert!(result.output_lower <= -4.99);
    assert!(result.output_upper >= 4.99);

    // Document method provenance: MaxBinaryLayer may lack CROWN backward
    // dispatch, so IBP fallback with reason is the expected path.
    if result.method == PropMethod::Ibp {
        assert!(
            result.crown_fallback_reason.is_some(),
            "IBP fallback must document CROWN failure reason for var-var max"
        );
    }
}

// --- Generic Select → WhereLayer fallback (#40 Finding B) ---

/// `if x > 0 { x + 1 } else { x - 1 }` — non-pattern-matched Select (no ReLU/LeakyReLU match).
fn generic_select_kernel_ir() -> KernelDef {
    KernelDef::new(
        "generic_select",
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
            IRNode::new(NodeId::new(3), IRNodeKind::Literal(1.0)),
            IRNode::new(
                NodeId::new(4),
                IRNodeKind::BinOp {
                    op: BinOpKind::Add,
                    lhs: NodeId::new(0),
                    rhs: NodeId::new(3),
                },
            ),
            IRNode::new(
                NodeId::new(5),
                IRNodeKind::BinOp {
                    op: BinOpKind::Sub,
                    lhs: NodeId::new(0),
                    rhs: NodeId::new(3),
                },
            ),
            IRNode::new(
                NodeId::new(6),
                IRNodeKind::Select {
                    cond: NodeId::new(2),
                    then_val: NodeId::new(4),
                    else_val: NodeId::new(5),
                },
            ),
        ],
        NodeId::new(6),
    )
}

#[test]
fn test_select_generic_where_fallback_crown() {
    let kernel = generic_select_kernel_ir();
    let input_bounds = scalar_input_bounds(-5.0, 5.0).expect("bounds");
    // Force CROWN with threshold=0.0
    let config = VerifyConfig::with_threshold(0.0).expect("valid threshold");

    let result = VerifyRequest::new(&kernel)
        .constant_params(&[])
        .input_bounds(&input_bounds)
        .config(config)
        .verify_bounds()
        .expect("verification should succeed via WhereLayer fallback");

    assert!(
        result.is_finite,
        "WhereLayer fallback bounds should be finite"
    );
    // if x > 0 { x+1 } else { x-1 } with x in [-5, 5]:
    // true branch: x+1 for x in (0,5] → [1, 6]
    // false branch: x-1 for x in [-5,0] → [-6, -1]
    // Sound bounds must contain [-6, 6]. IBP WhereLayer takes
    // min/max over then/else branches, producing tight [-6, 6].
    assert!(
        result.output_lower <= -5.99,
        "generic select lower should be <= -6, got {}",
        result.output_lower
    );
    assert!(
        result.output_upper >= 5.99,
        "generic select upper should be >= 6, got {}",
        result.output_upper
    );

    // WhereLayer may lack CROWN backward dispatch, causing IBP fallback.
    // Document the provenance: if IBP, fallback reason must be recorded.
    if result.method == PropMethod::Ibp {
        assert!(
            result.crown_fallback_reason.is_some(),
            "IBP fallback for generic Select must document CROWN failure reason"
        );
    }
}

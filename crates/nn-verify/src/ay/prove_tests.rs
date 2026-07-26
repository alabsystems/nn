// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Tests for ay SMT prove module.

use super::*;
use crate::test_helpers::bounds;
use crate::verify_input::ScalarInputBounds;
use nn_dsl::ir::{BinOpKind, IRNode, IRNodeKind, NodeId, Param, ScalarType};
use nn_dsl::test_kernels::snake_kernel;

#[test]
fn test_verify_snake_translates() {
    let kernel = snake_kernel();
    let result = verify_kernel_smt(&kernel, &[1.0], bounds(-10.0, 10.0)).unwrap();
    assert_eq!(result.solver, "ay-direct");
    assert_eq!(result.encoding, SmtEncodingKind::UfApprox);
    assert_eq!(result.property, "output_bounded");
    // #2640: Snake now routes to ay NRA solver via ALL logic auto-detection.
    assert_ne!(
        result.outcome,
        SmtOutcome::Unexecuted,
        "snake should reach NRA solver (#2640), got: {:?} (detail: {:?})",
        result.outcome,
        result.detail,
    );
}

#[test]
fn test_verify_simple_add_translates() {
    let kernel = KernelDef::new(
        "add_one",
        vec![Param::new("x", ScalarType::F32)],
        ScalarType::F32,
        vec![
            IRNode::new(NodeId::new(0), IRNodeKind::Param(0)),
            IRNode::new(NodeId::new(1), IRNodeKind::Literal(1.0)),
            IRNode::new(
                NodeId::new(2),
                IRNodeKind::BinOp {
                    op: BinOpKind::Add,
                    lhs: NodeId::new(0),
                    rhs: NodeId::new(1),
                },
            ),
        ],
        NodeId::new(2),
    );

    let result = verify_kernel_smt(&kernel, &[], bounds(-10.0, 10.0)).unwrap();
    assert_eq!(result.encoding, SmtEncodingKind::Exact);
    // add_one is not in the analytical match arms, so heuristic ±1e6 fallback
    // is used. Since #385, heuristic bounds produce Unexecuted to prevent
    // vacuous proofs. Use verify_kernel_smt_with_bounds for meaningful proofs.
    assert_eq!(result.solver, "ay");
    assert_eq!(
        result.outcome,
        SmtOutcome::Unexecuted,
        "heuristic bounds should produce Unexecuted (#385), got: {:?} (detail: {:?})",
        result.outcome,
        result.detail,
    );
    let detail = result.detail.as_deref().unwrap_or("");
    assert!(
        detail.contains("heuristic"),
        "detail should mention heuristic fallback, got: {detail}"
    );
}

#[test]
fn test_snake_smt2_output() {
    let kernel = snake_kernel();
    let smt2 = kernel_to_smt2(&kernel, &[1.0], bounds(-10.0, 10.0)).unwrap();
    // Should contain SMT-LIB2 structure
    assert!(smt2.contains("set-logic"));
    assert!(smt2.contains("declare-const"));
    assert!(smt2.contains("declare-fun"));
    assert!(smt2.contains("sin_approx"));
    assert!(smt2.contains("check-sat"));
    // Verify input bounds are asserted: x >= -10, x <= 10
    assert!(
        smt2.contains("-10.0"),
        "SMT-LIB2 should contain the lower input bound -10.0, got: {smt2}"
    );
    assert!(
        smt2.contains("10.0"),
        "SMT-LIB2 should contain the upper input bound 10.0, got: {smt2}"
    );
    // Verify the sin range axioms are present
    assert!(
        smt2.contains("-1.0"),
        "SMT-LIB2 should contain sin range lower bound -1, got: {smt2}"
    );
}

#[test]
fn test_smt_status_record_serialization() {
    let record = SmtStatusRecord {
        solver: "ay".to_string(),
        encoding: SmtEncodingKind::UfApprox,
        property: "output_bounded".to_string(),
        outcome: SmtOutcome::Proven,
        detail: None,
        bounds_source: BoundsSource::Heuristic,
        expected_bounds: None,
        proof_alethe: None,
        proof_verdict: None,
    };
    let json = serde_json::to_string_pretty(&record).unwrap();
    assert!(json.contains("\"uf_approx\""));
    assert!(json.contains("\"proven\""));

    let roundtrip: SmtStatusRecord = serde_json::from_str(&json).unwrap();
    assert_eq!(roundtrip, record);
}

#[test]
fn test_smt_outcome_unexecuted_roundtrip() {
    let record = SmtStatusRecord {
        solver: "ay".to_string(),
        encoding: SmtEncodingKind::Exact,
        property: "output_bounded".to_string(),
        outcome: SmtOutcome::Unexecuted,
        detail: Some("Phase A: solver not invoked".to_string()),
        bounds_source: BoundsSource::Heuristic,
        expected_bounds: None,
        proof_alethe: None,
        proof_verdict: None,
    };
    let json = serde_json::to_string_pretty(&record).unwrap();
    assert!(json.contains("\"unexecuted\""));
    let roundtrip: SmtStatusRecord = serde_json::from_str(&json).unwrap();
    assert_eq!(roundtrip.outcome, SmtOutcome::Unexecuted);
}

#[test]
fn test_smt_outcome_execution_failed_roundtrip() {
    let record = SmtStatusRecord {
        solver: "ay".to_string(),
        encoding: SmtEncodingKind::Exact,
        property: "output_bounded".to_string(),
        outcome: SmtOutcome::ExecutionFailed,
        detail: Some("direct execution failed: needs fallback".to_string()),
        bounds_source: BoundsSource::Heuristic,
        expected_bounds: None,
        proof_alethe: None,
        proof_verdict: None,
    };
    let json = serde_json::to_string_pretty(&record).unwrap();
    assert!(json.contains("\"execution_failed\""));
    let roundtrip: SmtStatusRecord = serde_json::from_str(&json).unwrap();
    assert_eq!(roundtrip.outcome, SmtOutcome::ExecutionFailed);
}

#[test]
fn test_verify_with_explicit_bounds() {
    let kernel = KernelDef::new(
        "add_one",
        vec![Param::new("x", ScalarType::F32)],
        ScalarType::F32,
        vec![
            IRNode::new(NodeId::new(0), IRNodeKind::Param(0)),
            IRNode::new(NodeId::new(1), IRNodeKind::Literal(1.0)),
            IRNode::new(
                NodeId::new(2),
                IRNodeKind::BinOp {
                    op: BinOpKind::Add,
                    lhs: NodeId::new(0),
                    rhs: NodeId::new(1),
                },
            ),
        ],
        NodeId::new(2),
    );

    // Explicit output bounds: for add_one with x in [-10, 10], output is [-9, 11]
    let result =
        verify_kernel_smt_with_bounds(&kernel, &[], bounds(-10.0, 10.0), Some((-9.0, 11.0)))
            .unwrap();
    assert_eq!(result.encoding, SmtEncodingKind::Exact);
    assert_eq!(result.solver, "ay-direct");
    // ay#5357 + ay#5605 fixed: QF_LRA solver correctly handles all linear kernels.
    assert_eq!(
        result.outcome,
        SmtOutcome::Proven,
        "add_one with explicit bounds must reach Proven, got: {:?} (detail: {:?})",
        result.outcome,
        result.detail,
    );
    // #383: explicit bounds → BoundsSource::CallerProvided.
    assert_eq!(
        result.bounds_source,
        BoundsSource::CallerProvided,
        "verify_kernel_smt_with_bounds(Some(...)) should produce CallerProvided"
    );
    assert_eq!(
        result.expected_bounds,
        Some((-9.0, 11.0)),
        "expected_bounds should match caller-provided values"
    );
}

#[test]
fn test_smt2_with_explicit_bounds() {
    let kernel = KernelDef::new(
        "double",
        vec![Param::new("x", ScalarType::F32)],
        ScalarType::F32,
        vec![
            IRNode::new(NodeId::new(0), IRNodeKind::Param(0)),
            IRNode::new(NodeId::new(1), IRNodeKind::Literal(2.0)),
            IRNode::new(
                NodeId::new(2),
                IRNodeKind::BinOp {
                    op: BinOpKind::Mul,
                    lhs: NodeId::new(0),
                    rhs: NodeId::new(1),
                },
            ),
        ],
        NodeId::new(2),
    );

    let smt2 = kernel_to_smt2_with_bounds(&kernel, &[], bounds(-5.0, 5.0), (-10.0, 10.0)).unwrap();
    assert!(smt2.contains("check-sat"));
    // Verify input bounds appear in the SMT-LIB2
    assert!(
        smt2.contains("-5.0"),
        "explicit input bound -5 should appear in SMT-LIB2"
    );
    assert!(
        smt2.contains("5.0"),
        "explicit input bound 5 should appear in SMT-LIB2"
    );
    // Verify expected output bounds appear (negated property: output < lower OR output > upper).
    // After SMT_QUANTIZATION_MARGIN widening, -10.0 becomes -10.0001 encoded as
    // (/ (- 10000100) 1000000), so check for the widened numerator.
    assert!(
        smt2.contains("10000100"),
        "expected widened output bound numerator should appear in SMT-LIB2"
    );
}

#[test]
fn test_kernel_status_smt_roundtrip() {
    use crate::soundness_compat::VerificationSoundnessMode;
    use crate::status::{
        InputBoundsRecord, KernelStatus, OutputBoundsRecord, ParamInputRecord, VerifyOutcome,
    };
    use crate::verify_types::PropMethod;

    let status = KernelStatus {
        status: VerifyOutcome::Verified,
        method: PropMethod::Ibp,
        input_bounds: InputBoundsRecord {
            variable_inputs: vec![ParamInputRecord {
                param_index: 0,
                lower: -10.0,
                upper: 10.0,
            }],
            constant_params: vec![1.0],
            input_shape: Some(vec![1]),
            input_range: Some((-10.0, 10.0)),
        },
        output_bounds: OutputBoundsRecord {
            lower: -10.0,
            upper: 11.0,
            tensor_lower: None,
            tensor_upper: None,
            shape: None,
            is_infeasible: false,
        },
        output_width: 21.0,
        crown_error: None,
        soundness_mode: VerificationSoundnessMode::Sound,
        smt: Some(SmtStatusRecord {
            solver: "ay".to_string(),
            encoding: SmtEncodingKind::UfApprox,
            property: "output_bounded".to_string(),
            outcome: SmtOutcome::Proven,
            detail: None,
            bounds_source: BoundsSource::Analytical,
            expected_bounds: Some((-11.0, 11.0)),
            proof_alethe: None,
            proof_verdict: None,
        }),
        crown_coverage: None,
        ibp_comparison_width: None,
        crown_ibp_ratio: None,
        weight_artifact: None,
        soundness_justification: None,
        stale: false,
        stale_reason: None,
        proof_strength: None,
    };

    let json = serde_json::to_string_pretty(&status).unwrap();
    assert!(json.contains("\"smt\""));
    assert!(json.contains("\"uf_approx\""));

    let roundtrip: KernelStatus = serde_json::from_str(&json).unwrap();
    assert_eq!(roundtrip.smt.as_ref().unwrap().outcome, SmtOutcome::Proven);
}

#[test]
fn test_verify_inverted_input_bounds_rejected() {
    // input_lower=10 > input_upper=-10: should fail, not produce a vacuous proof.
    let err = ScalarInputBounds::new(10.0, -10.0).unwrap_err();
    assert!(
        matches!(err, VerifyError::InvalidInputBounds { .. }),
        "inverted bounds should produce InvalidInputBounds, got: {err:?}"
    );
}

#[test]
fn test_verify_nan_input_bounds_rejected() {
    let err = ScalarInputBounds::new(f32::NAN, 10.0).unwrap_err();
    assert!(
        matches!(err, VerifyError::InvalidInputBounds { .. }),
        "NaN bounds should produce InvalidInputBounds, got: {err:?}"
    );
}

#[test]
fn test_verify_inverted_output_bounds_rejected() {
    let kernel = KernelDef::new(
        "add_one",
        vec![Param::new("x", ScalarType::F32)],
        ScalarType::F32,
        vec![
            IRNode::new(NodeId::new(0), IRNodeKind::Param(0)),
            IRNode::new(NodeId::new(1), IRNodeKind::Literal(1.0)),
            IRNode::new(
                NodeId::new(2),
                IRNodeKind::BinOp {
                    op: BinOpKind::Add,
                    lhs: NodeId::new(0),
                    rhs: NodeId::new(1),
                },
            ),
        ],
        NodeId::new(2),
    );
    // Inverted output bounds (10.0, -10.0) should fail, not produce a vacuous proof.
    let err = verify_kernel_smt_with_bounds(&kernel, &[], bounds(-10.0, 10.0), Some((10.0, -10.0)))
        .unwrap_err();
    let msg = err.to_string();
    assert!(
        msg.contains("inverted") || msg.contains("lower"),
        "inverted output bounds should produce a clear error, got: {msg}"
    );
}

#[test]
fn test_kernel_status_legacy_no_smt() {
    let json = r#"{
        "status": "verified",
        "method": "IBP",
        "input_bounds": {
            "variable_inputs": [{"param_index": 0, "lower": -1.0, "upper": 1.0}],
            "constant_params": []
        },
        "output_bounds": {"lower": -1.0, "upper": 1.0},
        "output_width": 2.0
    }"#;
    let status: crate::status::KernelStatus = serde_json::from_str(json).unwrap();
    assert!(status.smt.is_none());
}

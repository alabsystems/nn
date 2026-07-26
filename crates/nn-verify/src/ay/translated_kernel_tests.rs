// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Tests for `TranslatedKernel` incremental push/pop verification.

use crate::status::SmtEncodingKind;
use crate::test_helpers::bounds;
use crate::ay::{verify_kernel_smt_with_bounds, TranslatedKernel};
use nn_dsl::ir::{BinOpKind, IRNode, IRNodeKind, KernelDef, NodeId, Param, ScalarType};
use nn_dsl::test_kernels::snake_kernel;

/// Build a simple `x + 1` kernel (exact, linear — suitable for direct execution).
fn add_one_kernel() -> KernelDef {
    KernelDef::new(
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
    )
}

#[test]
fn test_translated_kernel_from_add_one() {
    let kernel = add_one_kernel();
    let tk = TranslatedKernel::from_kernel(&kernel, &[], bounds(-10.0, 10.0)).unwrap();
    assert_eq!(tk.kernel_name(), "add_one");
    assert_eq!(tk.encoding(), SmtEncodingKind::Exact);
    assert!(!tk.uses_nonlinear());
}

#[test]
fn test_translated_kernel_from_snake() {
    let kernel = snake_kernel();
    let tk = TranslatedKernel::from_kernel(&kernel, &[1.0], bounds(-10.0, 10.0)).unwrap();
    assert_eq!(tk.kernel_name(), "snake");
    assert_eq!(tk.encoding(), SmtEncodingKind::UfApprox);
}

#[test]
fn test_check_output_bounded_matches_single_shot_add_one() {
    let kernel = add_one_kernel();
    let input_bounds = bounds(-10.0, 10.0);
    let expected_bounds = (-9.0, 11.0);

    // Single-shot path.
    let single_shot =
        verify_kernel_smt_with_bounds(&kernel, &[], input_bounds, Some(expected_bounds)).unwrap();

    // Incremental path.
    let mut tk = TranslatedKernel::from_kernel(&kernel, &[], input_bounds).unwrap();
    let incremental = tk.check_output_bounded(expected_bounds).unwrap();

    assert_eq!(
        single_shot.outcome, incremental.outcome,
        "outcome mismatch: single_shot={:?}, incremental={:?}",
        single_shot, incremental
    );
    assert_eq!(single_shot.encoding, incremental.encoding);
    assert_eq!(single_shot.bounds_source, incremental.bounds_source);
    assert_eq!(single_shot.expected_bounds, incremental.expected_bounds);
}

#[test]
fn test_check_output_bounded_matches_single_shot_snake() {
    let kernel = snake_kernel();
    let input_bounds = bounds(-10.0, 10.0);
    let expected_bounds = (-10.0, 11.0);

    // Single-shot path.
    let single_shot =
        verify_kernel_smt_with_bounds(&kernel, &[1.0], input_bounds, Some(expected_bounds))
            .unwrap();

    // Incremental path.
    let mut tk = TranslatedKernel::from_kernel(&kernel, &[1.0], input_bounds).unwrap();
    let incremental = tk.check_output_bounded(expected_bounds).unwrap();

    assert_eq!(
        single_shot.outcome, incremental.outcome,
        "outcome mismatch: single_shot={:?}, incremental={:?}",
        single_shot, incremental
    );
    assert_eq!(single_shot.encoding, incremental.encoding);
}

#[test]
fn test_reuse_same_translation_multiple_bounds() {
    let kernel = add_one_kernel();
    let mut tk = TranslatedKernel::from_kernel(&kernel, &[], bounds(-10.0, 10.0)).unwrap();

    // First check: tight bounds.
    let r1 = tk.check_output_bounded((-9.0, 11.0)).unwrap();

    // Second check: wider bounds.
    let r2 = tk.check_output_bounded((-100.0, 100.0)).unwrap();

    // Both should succeed (same kernel, just different bound widths).
    // The exact outcome depends on ay solver state, but both should be
    // valid SmtStatusRecords with matching encoding.
    assert_eq!(r1.encoding, SmtEncodingKind::Exact);
    assert_eq!(r2.encoding, SmtEncodingKind::Exact);

    // The kernel is reusable after both checks.
    assert_eq!(tk.kernel_name(), "add_one");
}

#[test]
fn test_reuse_snake_translation_different_bounds() {
    let kernel = snake_kernel();
    let mut tk = TranslatedKernel::from_kernel(&kernel, &[1.0], bounds(-10.0, 10.0)).unwrap();

    // Check with analytical bounds.
    let r1 = tk.check_output_bounded((-10.0, 11.0)).unwrap();
    // Check with wider bounds.
    let r2 = tk.check_output_bounded((-100.0, 200.0)).unwrap();
    // Check with tight bounds.
    let r3 = tk.check_output_bounded((-10.0, 10.5)).unwrap();

    // All should produce valid results with UfApprox encoding.
    assert_eq!(r1.encoding, SmtEncodingKind::UfApprox);
    assert_eq!(r2.encoding, SmtEncodingKind::UfApprox);
    assert_eq!(r3.encoding, SmtEncodingKind::UfApprox);
}

#[test]
fn test_check_output_bounded_rejects_nan() {
    let kernel = add_one_kernel();
    let mut tk = TranslatedKernel::from_kernel(&kernel, &[], bounds(-10.0, 10.0)).unwrap();
    let err = tk.check_output_bounded((f64::NAN, 10.0));
    assert!(err.is_err(), "NaN lower bound should be rejected");
}

#[test]
fn test_check_output_bounded_rejects_inf() {
    let kernel = add_one_kernel();
    let mut tk = TranslatedKernel::from_kernel(&kernel, &[], bounds(-10.0, 10.0)).unwrap();
    let err = tk.check_output_bounded((-10.0, f64::INFINITY));
    assert!(err.is_err(), "infinite upper bound should be rejected");
}

#[test]
fn test_check_output_bounded_rejects_inverted() {
    let kernel = add_one_kernel();
    let mut tk = TranslatedKernel::from_kernel(&kernel, &[], bounds(-10.0, 10.0)).unwrap();
    let err = tk.check_output_bounded((10.0, -10.0));
    assert!(err.is_err(), "inverted bounds should be rejected");
}

#[test]
fn test_kernel_still_usable_after_error() {
    let kernel = add_one_kernel();
    let mut tk = TranslatedKernel::from_kernel(&kernel, &[], bounds(-10.0, 10.0)).unwrap();

    // Trigger an error (inverted bounds).
    let _ = tk.check_output_bounded((10.0, -10.0));

    // Kernel should still be usable after the error.
    let result = tk.check_output_bounded((-9.0, 11.0));
    assert!(result.is_ok(), "kernel should be reusable after error");
}

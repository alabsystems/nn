// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! SMT-related status tests: SmtContradiction downgrade, SMT outcome preservation,
//! and SmtContradiction JSON serialization roundtrip.

use super::status_test_helpers::{scalar_output_bounds, single_input_bounds};
use super::*;
use crate::verify_input::ScalarInputBounds;

/// When ay finds a counterexample, `record_smt` must downgrade the
/// `VerifyOutcome` from `Verified` to `SmtContradiction` so that
/// consumers checking status alone see the contradiction (#393).
#[test]
fn test_record_smt_counterexample_downgrades_verified_to_smt_contradiction() {
    let mut status = VerifyStatus::default();
    let result = KernelVerification {
        kernel_name: "contradicted".to_string(),
        method: PropMethod::Ibp,
        output_lower: -5.0,
        output_upper: 5.0,
        output_width: 10.0,
        is_finite: true,
        crown_fallback_reason: None,
        soundness_mode: VerificationSoundnessMode::Sound,
        output_tensor: None,
    };

    status
        .record(
            &result,
            ScalarInputBounds::new(-10.0, 10.0).expect("valid test bounds"),
            &[1.0],
            None,
        )
        .expect("record");
    assert_eq!(
        status.kernels["contradicted"].status,
        VerifyOutcome::Verified,
        "should start as Verified"
    );

    let smt_record = SmtStatusRecord {
        solver: "ay-direct".to_string(),
        encoding: SmtEncodingKind::Exact,
        property: "output_bounded".to_string(),
        outcome: SmtOutcome::Counterexample,
        detail: Some("direct execution: SAT (counterexample: {x: 1.0})".to_string()),
        bounds_source: BoundsSource::Heuristic,
        expected_bounds: None,
        proof_alethe: None,
        proof_verdict: None,
    };
    status
        .record_smt("contradicted", smt_record)
        .expect("record_smt");

    assert_eq!(
        status.kernels["contradicted"].status,
        VerifyOutcome::SmtContradiction,
        "Verified must be downgraded to SmtContradiction after counterexample"
    );
    assert_eq!(
        status.kernels["contradicted"]
            .smt
            .as_ref()
            .expect("smt field should be set after record_smt")
            .outcome,
        SmtOutcome::Counterexample,
    );

    // History entry must also be downgraded
    let history = status.history_for("contradicted").expect("history exists");
    assert_eq!(history.len(), 1);
    assert_eq!(
        history[0].status,
        VerifyOutcome::SmtContradiction,
        "history entry must also be downgraded"
    );
}

/// Non-counterexample SMT outcomes (Proven, Unknown, Unexecuted) must NOT
/// change the VerifyOutcome.
#[test]
fn test_record_smt_proven_preserves_verified_status() {
    let mut status = VerifyStatus::default();
    let result = KernelVerification {
        kernel_name: "confirmed".to_string(),
        method: PropMethod::Ibp,
        output_lower: -5.0,
        output_upper: 5.0,
        output_width: 10.0,
        is_finite: true,
        crown_fallback_reason: None,
        soundness_mode: VerificationSoundnessMode::Sound,
        output_tensor: None,
    };

    status
        .record(
            &result,
            ScalarInputBounds::new(-10.0, 10.0).expect("valid test bounds"),
            &[1.0],
            None,
        )
        .expect("record");

    let smt_record = SmtStatusRecord {
        solver: "ay-direct".to_string(),
        encoding: SmtEncodingKind::Exact,
        property: "output_bounded".to_string(),
        outcome: SmtOutcome::Proven,
        detail: Some("direct execution: UNSAT".to_string()),
        bounds_source: BoundsSource::Heuristic,
        expected_bounds: None,
        proof_alethe: None,
        proof_verdict: None,
    };
    status
        .record_smt("confirmed", smt_record)
        .expect("record_smt");

    assert_eq!(
        status.kernels["confirmed"].status,
        VerifyOutcome::Verified,
        "Proven SMT result should not change Verified status"
    );
}

/// `ExecutionFailed` means direct execution could not evaluate the kernel.
/// This must NOT downgrade `Verified` to `SmtContradiction` (#436).
#[test]
fn test_record_smt_execution_failed_preserves_verified_status() {
    let mut status = VerifyStatus::default();
    let result = KernelVerification {
        kernel_name: "exec_failed".to_string(),
        method: PropMethod::Ibp,
        output_lower: -5.0,
        output_upper: 5.0,
        output_width: 10.0,
        is_finite: true,
        crown_fallback_reason: None,
        soundness_mode: VerificationSoundnessMode::Sound,
        output_tensor: None,
    };

    status
        .record(
            &result,
            ScalarInputBounds::new(-10.0, 10.0).expect("valid test bounds"),
            &[],
            None,
        )
        .expect("record");

    let smt_record = SmtStatusRecord {
        solver: "ay-direct".to_string(),
        encoding: SmtEncodingKind::Exact,
        property: "output_bounded".to_string(),
        outcome: SmtOutcome::ExecutionFailed,
        detail: Some("direct execution failed: needs fallback".to_string()),
        bounds_source: BoundsSource::Heuristic,
        expected_bounds: None,
        proof_alethe: None,
        proof_verdict: None,
    };
    status
        .record_smt("exec_failed", smt_record)
        .expect("record_smt");

    assert_eq!(
        status.kernels["exec_failed"].status,
        VerifyOutcome::Verified,
        "ExecutionFailed must not downgrade Verified status"
    );
    assert_eq!(
        status.kernels["exec_failed"]
            .smt
            .as_ref()
            .expect("smt field")
            .outcome,
        SmtOutcome::ExecutionFailed,
    );
}

/// `Unknown` means the solver ran but could not decide. Must NOT downgrade (#436).
#[test]
fn test_record_smt_unknown_preserves_verified_status() {
    let mut status = VerifyStatus::default();
    let result = KernelVerification {
        kernel_name: "unknown_result".to_string(),
        method: PropMethod::Ibp,
        output_lower: -5.0,
        output_upper: 5.0,
        output_width: 10.0,
        is_finite: true,
        crown_fallback_reason: None,
        soundness_mode: VerificationSoundnessMode::Sound,
        output_tensor: None,
    };

    status
        .record(
            &result,
            ScalarInputBounds::new(-10.0, 10.0).expect("valid test bounds"),
            &[],
            None,
        )
        .expect("record");

    let smt_record = SmtStatusRecord {
        solver: "ay".to_string(),
        encoding: SmtEncodingKind::UfApprox,
        property: "output_bounded".to_string(),
        outcome: SmtOutcome::Unknown,
        detail: Some("solver: timeout".to_string()),
        bounds_source: BoundsSource::Heuristic,
        expected_bounds: None,
        proof_alethe: None,
        proof_verdict: None,
    };
    status
        .record_smt("unknown_result", smt_record)
        .expect("record_smt");

    assert_eq!(
        status.kernels["unknown_result"].status,
        VerifyOutcome::Verified,
        "Unknown must not downgrade Verified status"
    );
    assert_eq!(
        status.kernels["unknown_result"]
            .smt
            .as_ref()
            .expect("smt field")
            .outcome,
        SmtOutcome::Unknown,
    );
}

/// `Unexecuted` means the solver was never invoked. Must NOT downgrade (#436).
#[test]
fn test_record_smt_unexecuted_preserves_verified_status() {
    let mut status = VerifyStatus::default();
    let result = KernelVerification {
        kernel_name: "unexecuted_smt".to_string(),
        method: PropMethod::Ibp,
        output_lower: -5.0,
        output_upper: 5.0,
        output_width: 10.0,
        is_finite: true,
        crown_fallback_reason: None,
        soundness_mode: VerificationSoundnessMode::Sound,
        output_tensor: None,
    };

    status
        .record(
            &result,
            ScalarInputBounds::new(-10.0, 10.0).expect("valid test bounds"),
            &[],
            None,
        )
        .expect("record");

    let smt_record = SmtStatusRecord {
        solver: "ay".to_string(),
        encoding: SmtEncodingKind::UfApprox,
        property: "output_bounded".to_string(),
        outcome: SmtOutcome::Unexecuted,
        detail: Some("Phase A: SMT-LIB2 generated but solver not invoked".to_string()),
        bounds_source: BoundsSource::Heuristic,
        expected_bounds: None,
        proof_alethe: None,
        proof_verdict: None,
    };
    status
        .record_smt("unexecuted_smt", smt_record)
        .expect("record_smt");

    assert_eq!(
        status.kernels["unexecuted_smt"].status,
        VerifyOutcome::Verified,
        "Unexecuted must not downgrade Verified status"
    );
    assert_eq!(
        status.kernels["unexecuted_smt"]
            .smt
            .as_ref()
            .expect("smt field")
            .outcome,
        SmtOutcome::Unexecuted,
    );
}

/// `SmtStatusRecord::execution_failed` produces the expected record (#481).
#[test]
fn test_execution_failed_constructor_fields() {
    let record = SmtStatusRecord::execution_failed("pipeline error: boom");
    assert_eq!(record.solver, "ay");
    assert_eq!(record.encoding, SmtEncodingKind::UfApprox);
    assert_eq!(record.property, "pipeline_failure");
    assert_eq!(record.outcome, SmtOutcome::ExecutionFailed);
    assert_eq!(record.detail.as_deref(), Some("pipeline error: boom"),);
    assert_eq!(record.bounds_source, BoundsSource::Heuristic);
    assert!(record.expected_bounds.is_none());
}

/// Partial-persist scenario: NY succeeds but ay fails.
/// After recording `execution_failed`, the entry's smt field is
/// `Some(ExecutionFailed)` — distinguishable from `None` (never attempted) (#481).
#[test]
fn test_partial_persist_execution_failed_distinguishable_from_none() {
    let mut status = VerifyStatus::default();
    let result = KernelVerification {
        kernel_name: "partial_kernel".to_string(),
        method: PropMethod::Ibp,
        output_lower: -5.0,
        output_upper: 5.0,
        output_width: 10.0,
        is_finite: true,
        crown_fallback_reason: None,
        soundness_mode: VerificationSoundnessMode::Sound,
        output_tensor: None,
    };

    // Step 1: NY records successfully
    status
        .record(
            &result,
            ScalarInputBounds::new(-10.0, 10.0).expect("valid test bounds"),
            &[1.0],
            None,
        )
        .expect("record");

    // After record(), smt is None — "never attempted"
    assert!(
        status.kernels["partial_kernel"].smt.is_none(),
        "smt should be None before ay runs"
    );

    // Step 2: ay fails — caller records explicit failure
    let failure = SmtStatusRecord::execution_failed("pipeline error: ay encode failed");
    status
        .record_smt("partial_kernel", failure)
        .expect("record_smt");

    // After record_smt, smt is Some(ExecutionFailed) — distinguishable from None
    let smt = status.kernels["partial_kernel"]
        .smt
        .as_ref()
        .expect("smt should be Some after execution_failed");
    assert_eq!(smt.outcome, SmtOutcome::ExecutionFailed);
    assert!(smt
        .detail
        .as_ref()
        .expect("detail should contain failure reason")
        .contains("ay encode failed"));

    // Status remains Verified (ExecutionFailed does not downgrade)
    assert_eq!(
        status.kernels["partial_kernel"].status,
        VerifyOutcome::Verified,
    );
}

/// SmtContradiction serializes as `"smt_contradiction"` in JSON.
#[test]
fn test_smt_contradiction_json_roundtrip() {
    let mut status = VerifyStatus::default();
    status.kernels.insert(
        "test".to_string(),
        KernelStatus {
            status: VerifyOutcome::SmtContradiction,
            method: PropMethod::Ibp,
            input_bounds: single_input_bounds(-1.0, 1.0, vec![]),
            output_bounds: scalar_output_bounds(-1.0, 1.0),
            output_width: 2.0,
            crown_error: None,
            soundness_mode: VerificationSoundnessMode::Sound,
            smt: Some(SmtStatusRecord {
                solver: "ay".to_string(),
                encoding: SmtEncodingKind::Exact,
                property: "output_bounded".to_string(),
                outcome: SmtOutcome::Counterexample,
                detail: None,
                bounds_source: BoundsSource::Heuristic,
                expected_bounds: None,
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
        },
    );

    let json = serde_json::to_string_pretty(&status).expect("serialize");
    assert!(
        json.contains("smt_contradiction"),
        "SmtContradiction should serialize as smt_contradiction, got:\n{json}"
    );

    let deserialized: VerifyStatus = serde_json::from_str(&json).expect("deserialize");
    assert_eq!(
        deserialized.kernels["test"].status,
        VerifyOutcome::SmtContradiction
    );
}

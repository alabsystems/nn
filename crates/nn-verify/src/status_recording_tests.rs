// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Unit tests for status_recording.rs: record, record_failure, record_smt,
//! and run_count write-path functions.
//!
//! Covers the #247 bug scenario (record_smt with missing history), history
//! truncation at MAX_HISTORY_PER_KERNEL, and round-trip correctness.

use ny_core::VerificationSoundnessMode;

use super::*;
use crate::status_smt::{BoundsSource, SmtEncodingKind, SmtOutcome, SmtStatusRecord};
use crate::verify_input::ScalarInputBounds;
use crate::verify_types::PropMethod;

/// Build a minimal `KernelVerification` for testing.
fn stub_verification(name: &str, lower: f32, upper: f32) -> KernelVerification {
    KernelVerification {
        kernel_name: name.to_string(),
        method: PropMethod::Ibp,
        output_lower: lower,
        output_upper: upper,
        output_width: upper - lower,
        is_finite: lower.is_finite() && upper.is_finite(),
        crown_fallback_reason: None,
        soundness_mode: VerificationSoundnessMode::Heuristic,
        output_tensor: None,
    }
}

/// Build a minimal `SmtStatusRecord` for testing.
fn stub_smt_record(outcome: SmtOutcome) -> SmtStatusRecord {
    SmtStatusRecord {
        solver: "ay".to_string(),
        encoding: SmtEncodingKind::UfApprox,
        property: "bound_check".to_string(),
        outcome,
        detail: None,
        bounds_source: BoundsSource::Analytical,
        expected_bounds: Some((-1.0, 1.0)),
        proof_alethe: None,
        proof_verdict: None,
    }
}

// ======================== record() ========================

#[test]
fn test_record_creates_entry_and_history() {
    let mut status = VerifyStatus::default();
    let result = stub_verification("snake", -1.0, 1.0);
    let bounds = ScalarInputBounds::new(-1.0, 1.0).unwrap();

    status
        .record(&result, bounds, &[1.0_f32], None)
        .expect("record should succeed");

    // Latest entry
    let entry = status.kernel("snake").expect("kernel entry should exist");
    // is_finite=true, crown_fallback_reason=None → Verified
    assert_eq!(entry.status, VerifyOutcome::Verified);
    assert_eq!(entry.method, PropMethod::Ibp);

    // History
    assert_eq!(status.run_count("snake"), 1);
    let hist = status.history_for("snake").expect("history should exist");
    assert_eq!(hist.len(), 1);
    assert_eq!(hist[0].method, PropMethod::Ibp);
}

#[test]
fn test_record_with_status_key_override() {
    let mut status = VerifyStatus::default();
    let result = stub_verification("snake", -1.0, 1.0);
    let bounds = ScalarInputBounds::new(-1.0, 1.0).unwrap();

    status
        .record(&result, bounds, &[1.0_f32], Some("snake_alpha_1.0"))
        .expect("record with status_key should succeed");

    // Should be stored under the overridden key, not kernel_name
    assert!(status.kernel("snake").is_none());
    assert!(status.kernel("snake_alpha_1.0").is_some());
    assert_eq!(status.run_count("snake_alpha_1.0"), 1);
}

#[test]
fn test_record_verified_status_when_finite_no_fallback() {
    let mut status = VerifyStatus::default();
    let result = KernelVerification {
        kernel_name: "test_kernel".to_string(),
        method: PropMethod::Crown,
        output_lower: -0.5,
        output_upper: 0.5,
        output_width: 1.0,
        is_finite: true,
        crown_fallback_reason: None,
        soundness_mode: VerificationSoundnessMode::Sound,
        output_tensor: None,
    };
    let bounds = ScalarInputBounds::new(-1.0, 1.0).unwrap();

    status
        .record(&result, bounds, &[0.5_f32], None)
        .expect("record should succeed");

    let entry = status.kernel("test_kernel").unwrap();
    assert_eq!(entry.status, VerifyOutcome::Verified);
    assert_eq!(entry.soundness_mode, VerificationSoundnessMode::Sound);
}

#[test]
fn test_record_ibp_fallback_status() {
    let mut status = VerifyStatus::default();
    let result = KernelVerification {
        kernel_name: "fallback_kernel".to_string(),
        method: PropMethod::Ibp,
        output_lower: -0.5,
        output_upper: 0.5,
        output_width: 1.0,
        is_finite: true,
        crown_fallback_reason: Some("CROWN failed: unsupported layer".to_string()),
        soundness_mode: VerificationSoundnessMode::Heuristic,
        output_tensor: None,
    };
    let bounds = ScalarInputBounds::new(-1.0, 1.0).unwrap();

    status
        .record(&result, bounds, &[], None)
        .expect("record should succeed");

    let entry = status.kernel("fallback_kernel").unwrap();
    assert_eq!(entry.status, VerifyOutcome::IbpFallback);
    assert!(entry.crown_error.is_some());
}

// ======================== record_failure() ========================

#[test]
fn test_record_failure_creates_failed_entry() {
    let mut status = VerifyStatus::default();
    let bounds = ScalarInputBounds::new(-1.0, 1.0).unwrap();

    status
        .record_failure("bad_kernel", PropMethod::Ibp, bounds, &[1.0_f32])
        .expect("record_failure should succeed");

    let entry = status.kernel("bad_kernel").expect("should have entry");
    assert_eq!(entry.status, VerifyOutcome::Failed);
    assert_eq!(entry.output_bounds.lower, 0.0);
    assert_eq!(entry.output_bounds.upper, 0.0);
    assert_eq!(status.run_count("bad_kernel"), 1);
}

#[test]
fn test_record_failure_rejects_non_finite_constant() {
    let mut status = VerifyStatus::default();
    let bounds = ScalarInputBounds::new(-1.0, 1.0).unwrap();

    let err = status
        .record_failure("bad_kernel", PropMethod::Ibp, bounds, &[f32::NAN])
        .unwrap_err();
    let msg = err.to_string();
    assert!(
        msg.contains("non-finite") || msg.contains("NaN"),
        "NaN constant param should be rejected, got: {msg}"
    );
}

// ======================== record_smt() ========================

#[test]
fn test_record_smt_attaches_to_existing_entry() {
    let mut status = VerifyStatus::default();
    let result = stub_verification("snake", -1.0, 1.0);
    let bounds = ScalarInputBounds::new(-1.0, 1.0).unwrap();
    status.record(&result, bounds, &[1.0_f32], None).unwrap();

    let smt = stub_smt_record(SmtOutcome::Proven);
    status.record_smt("snake", smt).unwrap();

    let entry = status.kernel("snake").unwrap();
    let smt_record = entry.smt.as_ref().expect("smt field should be populated");
    assert_eq!(smt_record.outcome, SmtOutcome::Proven);

    // History entry should also be updated
    let hist = status.history_for("snake").unwrap();
    let last = hist.last().unwrap();
    assert!(last.smt.is_some());
    assert_eq!(last.smt.as_ref().unwrap().outcome, SmtOutcome::Proven);
}

#[test]
fn test_record_smt_missing_entry_returns_error() {
    // Per #669: record_smt on non-existent entry returns Err (was Ok(false))
    let mut status = VerifyStatus::default();
    let smt = stub_smt_record(SmtOutcome::Proven);

    let err = status
        .record_smt("nonexistent_kernel", smt)
        .expect_err("record_smt should error for missing kernel");
    assert!(
        err.to_string().contains("no kernel entry"),
        "error should mention missing entry: {err}"
    );
}

#[test]
fn test_record_smt_counterexample_downgrades_status() {
    let mut status = VerifyStatus::default();
    // Record an initially-verified kernel
    let result = KernelVerification {
        kernel_name: "contradicted".to_string(),
        method: PropMethod::Crown,
        output_lower: -0.5,
        output_upper: 0.5,
        output_width: 1.0,
        is_finite: true,
        crown_fallback_reason: None,
        soundness_mode: VerificationSoundnessMode::Heuristic,
        output_tensor: None,
    };
    let bounds = ScalarInputBounds::new(-1.0, 1.0).unwrap();
    status.record(&result, bounds, &[], None).unwrap();
    assert_eq!(
        status.kernel("contradicted").unwrap().status,
        VerifyOutcome::Verified
    );

    // SMT finds counterexample — should downgrade to SmtContradiction
    let smt = stub_smt_record(SmtOutcome::Counterexample);
    status.record_smt("contradicted", smt).unwrap();

    let entry = status.kernel("contradicted").unwrap();
    assert_eq!(entry.status, VerifyOutcome::SmtContradiction);

    // History should also be downgraded
    let hist = status.history_for("contradicted").unwrap();
    assert_eq!(hist.last().unwrap().status, VerifyOutcome::SmtContradiction);
}

// ======================== run_count() ========================

#[test]
fn test_run_count_zero_for_unknown_kernel() {
    let status = VerifyStatus::default();
    assert_eq!(status.run_count("never_recorded"), 0);
}

#[test]
fn test_run_count_accumulates() {
    let mut status = VerifyStatus::default();
    let result = stub_verification("multi_run", -1.0, 1.0);
    let bounds = ScalarInputBounds::new(-1.0, 1.0).unwrap();

    for _ in 0..5 {
        status.record(&result, bounds, &[1.0_f32], None).unwrap();
    }
    assert_eq!(status.run_count("multi_run"), 5);
}

// ======================== history truncation ========================

#[test]
fn test_history_truncated_at_max() {
    use super::status_recording::MAX_HISTORY_PER_KERNEL;

    let mut status = VerifyStatus::default();
    let result = stub_verification("overflow_kernel", -1.0, 1.0);
    let bounds = ScalarInputBounds::new(-1.0, 1.0).unwrap();

    // Record more than MAX_HISTORY_PER_KERNEL times
    for _ in 0..(MAX_HISTORY_PER_KERNEL + 5) {
        status.record(&result, bounds, &[1.0_f32], None).unwrap();
    }

    let hist = status.history_for("overflow_kernel").unwrap();
    assert_eq!(
        hist.len(),
        MAX_HISTORY_PER_KERNEL,
        "history should be truncated to {} entries, got {}",
        MAX_HISTORY_PER_KERNEL,
        hist.len()
    );
}

// ======================== #560: partial-write detection ========================

#[test]
fn test_record_smt_partial_write_when_history_missing() {
    // Reproduce #560: kernel entry exists but history is missing (legacy
    // deserialized status file or manually constructed state). record_smt
    // should NOT return Ok(true) when history was not updated.
    //
    // Construct via JSON deserialization to simulate a legacy status file
    // that has kernels but no history.
    let json = r#"{
        "kernels": {
            "legacy_kernel": {
                "status": "verified",
                "method": "IBP",
                "input_bounds": {
                    "variable_inputs": [{"param_index": 0, "lower": -1.0, "upper": 1.0}],
                    "constant_params": [1.0]
                },
                "output_bounds": { "lower": -0.5, "upper": 0.5 },
                "output_width": 1.0
            }
        }
    }"#;
    let mut status: VerifyStatus = serde_json::from_str(json).expect("valid JSON");

    // Verify precondition: kernel exists but history does NOT.
    assert!(status.kernel("legacy_kernel").is_some());
    assert!(status.history_for("legacy_kernel").is_none());

    let smt = stub_smt_record(SmtOutcome::Proven);
    // Per #669: record_smt returns Ok(()) — partial writes (entry updated,
    // history missing) are logged to stderr, not signaled via return value.
    status
        .record_smt("legacy_kernel", smt)
        .expect("should not error");

    // Verify the kernel entry WAS updated (the partial write).
    let entry = status.kernel("legacy_kernel").unwrap();
    assert!(entry.smt.is_some(), "smt field should be populated");
    assert_eq!(entry.smt.as_ref().unwrap().outcome, SmtOutcome::Proven);

    // Verify history was NOT updated (the partial part).
    assert!(
        status.history_for("legacy_kernel").is_none(),
        "history should still be missing — partial write did not create it"
    );
}

// ======================== record_pipeline() vacuous width ========================

#[test]
fn test_record_pipeline_tight_bounds_are_verified() {
    let mut status = VerifyStatus::default();
    status
        .record_pipeline(
            "tight_pipeline",
            PropMethod::Ibp,
            -1.0,
            1.0,
            -0.5,
            0.5,
            &[4],
            VerificationSoundnessMode::Heuristic,
            Some(&[1, 4]),
        )
        .expect("record_pipeline");

    let entry = status.kernel("tight_pipeline").unwrap();
    assert_eq!(
        entry.status,
        VerifyOutcome::Verified,
        "tight bounds (width=1.0) should be Verified"
    );
}

#[test]
fn test_record_pipeline_vacuous_bounds_are_bounds_computed() {
    let mut status = VerifyStatus::default();
    // Width = 2e10 — vacuously wide, matches the kokoro_chained_norm entries.
    status
        .record_pipeline(
            "vacuous_pipeline",
            PropMethod::Crown,
            -1.0,
            1.0,
            -1e10,
            1e10,
            &[4, 16],
            VerificationSoundnessMode::Heuristic,
            Some(&[4, 16]),
        )
        .expect("record_pipeline");

    let entry = status.kernel("vacuous_pipeline").unwrap();
    assert_eq!(
        entry.status,
        VerifyOutcome::BoundsComputed,
        "vacuous bounds (width=2e10) must be BoundsComputed, not Verified"
    );
}

#[test]
fn test_record_pipeline_boundary_width_is_verified() {
    let mut status = VerifyStatus::default();
    // Width = 999_999 — just under threshold, should be Verified.
    let half = 999_999.0 / 2.0;
    status
        .record_pipeline(
            "boundary_pipeline",
            PropMethod::Ibp,
            -1.0,
            1.0,
            -half,
            half,
            &[8],
            VerificationSoundnessMode::Sound,
            None,
        )
        .expect("record_pipeline");

    let entry = status.kernel("boundary_pipeline").unwrap();
    assert_eq!(
        entry.status,
        VerifyOutcome::Verified,
        "width just under 1e6 should still be Verified"
    );
}

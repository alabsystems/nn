// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0
#![cfg(feature = "ay-smt")]

//! Integration test: verify kernels via ay SMT path and persist results.
//!
//! Tests both exact-encoding (direct execution) and UF-approximation
//! (Phase A translation only) paths. Verifies that `record_smt` correctly
//! attaches SMT results to existing NY verification entries in
//! `VerifyStatus`.

use nn_verify::{
    scalar_input_bounds, verify_kernel_smt, verify_kernel_smt_with_bounds, ScalarInputBounds,
    SmtEncodingKind, SmtOutcome, VerifyRequest, VerifyStatus,
};

use super::common;

/// Scale kernel: pure arithmetic → exact encoding in ay.
fn scale_kernel() -> nn_dsl::ir::KernelDef {
    common::parse_kernel("fn scale(x: f32) -> f32 { x * 2.0 + 1.0 }")
}

/// Clamp kernel: pure arithmetic with clamp → exact encoding in ay.
fn clamp_kernel() -> nn_dsl::ir::KernelDef {
    common::parse_kernel("fn clamped(x: f32) -> f32 { x.clamp(-1.0, 1.0) }")
}

// --- End-to-end ay verification tests ---

#[test]
fn test_ay_exact_kernel_encoding_correct_bounds() {
    // Scale kernel: f(x) = 2x + 1, x in [-5, 5] → output in [-9, 11].
    //
    // ay#5357 fix landed (6aac039): strict inequality model validation fixed.
    // ay#5605 fix landed: real_mul incompleteness resolved — ay-direct now
    // correctly handles multiplication by constants. Strict Proven assertion.
    let kernel = scale_kernel();
    let result = verify_kernel_smt_with_bounds(
        &kernel,
        &[],
        ScalarInputBounds::new(-5.0, 5.0).expect("valid test bounds"),
        Some((-9.0, 11.0)),
    )
    .unwrap();

    assert_eq!(result.encoding, SmtEncodingKind::Exact);
    assert_eq!(
        result.solver, "ay-direct",
        "exact kernel should use direct execution"
    );
    assert_eq!(
        result.outcome,
        SmtOutcome::Proven,
        "ay#5605 fixed: scale kernel (2x+1) must reach Proven, got: {:?} (detail: {:?})",
        result.outcome,
        result.detail,
    );
}

#[test]
fn test_ay_exact_kernel_detects_tight_bounds() {
    // Scale kernel: f(x) = 2x + 1, x in [-5, 5] but bounds [-1, 1] too tight.
    // f(5) = 11 > 1 → should find counterexample.
    //
    // ay#5357 fix landed (6aac039): ay-direct should now find counterexamples
    // for exact linear kernels with too-tight bounds.
    // Tightened from `Counterexample | Unknown` to `Counterexample`.
    let kernel = scale_kernel();
    let result = verify_kernel_smt_with_bounds(
        &kernel,
        &[],
        ScalarInputBounds::new(-5.0, 5.0).expect("valid test bounds"),
        Some((-1.0, 1.0)),
    )
    .unwrap();

    assert_eq!(result.encoding, SmtEncodingKind::Exact);
    assert_eq!(result.solver, "ay-direct");
    assert_eq!(
        result.outcome,
        SmtOutcome::Counterexample,
        "ay#5357 fixed: too-tight bounds must find Counterexample, got: {:?} (detail: {:?})",
        result.outcome,
        result.detail,
    );

    // Counterexample result must include detail with witness.
    assert!(
        result.detail.is_some(),
        "Counterexample result must include detail with witness"
    );
    assert!(
        !result.detail.as_ref().unwrap().is_empty(),
        "Counterexample detail must not be empty"
    );
}

#[test]
fn test_ay_clamp_kernel_encoding_bounded_output() {
    // Clamp kernel: f(x) = clamp(x, -1, 1), x in [-100, 100] → output in [-1, 1].
    //
    // ay#5357 fix landed (6aac039): ay-direct should now prove clamp kernel
    // bounds. Tightened from `Proven | Unknown` to `Proven`.
    let kernel = clamp_kernel();
    let result = verify_kernel_smt_with_bounds(
        &kernel,
        &[],
        ScalarInputBounds::new(-100.0, 100.0).expect("valid test bounds"),
        Some((-1.0, 1.0)),
    )
    .unwrap();

    assert_eq!(result.encoding, SmtEncodingKind::Exact);
    assert_eq!(result.solver, "ay-direct");
    assert_eq!(
        result.outcome,
        SmtOutcome::Proven,
        "ay#5357 fixed: clamp kernel must reach Proven, got: {:?} (detail: {:?})",
        result.outcome,
        result.detail,
    );

    // Proven result must include expected_bounds metadata.
    assert!(
        result.expected_bounds.is_some(),
        "Proven result must include expected_bounds metadata"
    );
    let (lo, hi) = result.expected_bounds.unwrap();
    assert!(lo < hi, "Proven bounds must have lo < hi, got ({lo}, {hi})");
}

#[test]
fn test_ay_snake_uf_approximation_unexecuted() {
    // Snake kernel uses sin (transcendental) → UF approximation.
    // Phase A: SMT-LIB2 generated but solver not invoked.
    let kernel = common::snake_kernel();
    let result = verify_kernel_smt(
        &kernel,
        &[1.0],
        ScalarInputBounds::new(-10.0, 10.0).expect("valid test bounds"),
    )
    .unwrap();

    assert_eq!(result.encoding, SmtEncodingKind::UfApprox);
    // After ay rev bump (d4021eb4), ay direct execution handles UF programs.
    // Accept either Proven (ay solved it) or Unexecuted (old behavior).
    assert!(
        matches!(result.outcome, SmtOutcome::Proven | SmtOutcome::Unexecuted),
        "UF kernel should be Proven or Unexecuted, got: {:?}",
        result.outcome
    );
}

/// Per-model status file path for snake kernel persistence (#2577).
fn status_file_path() -> std::path::PathBuf {
    let manifest_dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
    let workspace_root = manifest_dir
        .parent()
        .and_then(|p| p.parent())
        .expect("workspace root");
    // Snake is a shared kernel.
    nn_verify::model_status_path(workspace_root, "shared")
}

#[test]
fn test_ay_smt_persisted_alongside_gamma_crown() {
    // Full pipeline: NY IBP → ay SMT → persist both.
    let kernel = common::snake_kernel();
    let (x_lo, x_hi) = (-10.0f32, 10.0f32);
    let status_path = status_file_path();
    let bounds = ScalarInputBounds::new(x_lo, x_hi).expect("valid test bounds");

    // Lock held across full load-modify-save cycle (#482 TOCTOU fix).
    let mut locked = VerifyStatus::load_locked(&status_path).expect("load_locked");
    let pre_existing = locked.status.kernel_count();

    // Step 1: NY IBP verification.
    let input_bounds = scalar_input_bounds(x_lo, x_hi).expect("input bounds");
    let ibp_result = VerifyRequest::new(&kernel)
        .constant_params(&[1.0])
        .input_bounds(&input_bounds)
        .verify_bounds()
        .expect("IBP verification");
    assert!(ibp_result.is_finite, "IBP bounds should be finite");
    locked
        .status
        .record(&ibp_result, bounds, &[1.0], None)
        .expect("record IBP");

    let entry = locked
        .status
        .kernel("snake")
        .expect("snake entry after IBP");
    assert!(entry.smt.is_none(), "no SMT result before ay verification");

    // Step 2: ay SMT verification.
    let smt_result =
        verify_kernel_smt(&kernel, &[1.0], bounds).expect("ay SMT verification should succeed");
    assert_eq!(smt_result.encoding, SmtEncodingKind::UfApprox);

    // Step 3: Attach SMT result to existing kernel entry.
    locked
        .status
        .record_smt("snake", smt_result)
        .expect("record_smt");

    // Step 4: Verify combined entry has both IBP and SMT results.
    let entry = locked.status.kernel("snake").expect("snake entry exists");
    assert_eq!(entry.status, nn_verify::VerifyOutcome::Verified);
    let smt = entry.smt.as_ref().expect("SMT result should be attached");
    assert_eq!(smt.encoding, SmtEncodingKind::UfApprox);
    // After ay rev bump, ay may solve UF programs directly (Proven).
    assert!(
        matches!(smt.outcome, SmtOutcome::Proven | SmtOutcome::Unexecuted),
        "UF kernel SMT should be Proven or Unexecuted, got: {:?}",
        smt.outcome
    );

    // Step 5: Persistence round-trip (lock held across full cycle, #482).
    locked.save().expect("save");
    drop(locked);

    // Locked validation read to avoid TOCTOU race with parallel tests (#537).
    let validation = VerifyStatus::load_locked(&status_path).expect("load_locked validation");

    // Merge must not destroy pre-existing entries (#450 regression guard).
    assert!(
        validation.status.kernel_count() >= pre_existing,
        "must not destroy pre-existing entries: had {pre_existing}, now {}",
        validation.status.kernel_count()
    );
    let loaded_entry = validation
        .status
        .kernel("snake")
        .expect("snake entry round-trips");
    let loaded_smt = loaded_entry.smt.as_ref().expect("SMT field should persist");
    assert_eq!(loaded_smt.encoding, SmtEncodingKind::UfApprox);
    assert!(
        matches!(
            loaded_smt.outcome,
            SmtOutcome::Proven | SmtOutcome::Unexecuted
        ),
        "loaded UF kernel SMT should be Proven or Unexecuted, got: {:?}",
        loaded_smt.outcome
    );
}

#[test]
fn test_ay_record_smt_returns_false_for_unknown_kernel() {
    let mut status = VerifyStatus::default();
    // Get a real SmtStatusRecord from verification of a simple kernel.
    let scale = scale_kernel();
    let smt_result = verify_kernel_smt_with_bounds(
        &scale,
        &[],
        ScalarInputBounds::new(-5.0, 5.0).expect("valid test bounds"),
        Some((-9.0, 11.0)),
    )
    .unwrap();
    // Per #669: record_smt returns Err when no kernel entry exists.
    let err = status
        .record_smt("nonexistent_kernel", smt_result)
        .expect_err("record_smt should error for missing kernel");
    assert!(err.to_string().contains("no kernel entry"));
}

#[test]
fn test_ay_exact_and_uf_combined_status() {
    // Verify two kernels: one exact, one UF. Both persisted in same VerifyStatus.
    let scale = scale_kernel();
    let snake = common::snake_kernel();
    let mut status = VerifyStatus::default();

    // Record NY results for both.
    let scale_bounds = scalar_input_bounds(-5.0, 5.0).expect("scale bounds");
    let scale_ibp = VerifyRequest::new(&scale)
        .constant_params(&[])
        .input_bounds(&scale_bounds)
        .verify_bounds()
        .expect("scale IBP");
    status
        .record(
            &scale_ibp,
            ScalarInputBounds::new(-5.0, 5.0).expect("valid test bounds"),
            &[],
            None,
        )
        .expect("record scale");

    let snake_bounds = scalar_input_bounds(-10.0, 10.0).expect("snake bounds");
    let snake_ibp = VerifyRequest::new(&snake)
        .constant_params(&[1.0])
        .input_bounds(&snake_bounds)
        .verify_bounds()
        .expect("snake IBP");
    status
        .record(
            &snake_ibp,
            ScalarInputBounds::new(-10.0, 10.0).expect("valid test bounds"),
            &[1.0],
            None,
        )
        .expect("record snake");

    // Record ay results for both.
    let scale_smt = verify_kernel_smt_with_bounds(
        &scale,
        &[],
        ScalarInputBounds::new(-5.0, 5.0).expect("valid test bounds"),
        Some((-9.0, 11.0)),
    )
    .unwrap();
    let snake_smt = verify_kernel_smt(
        &snake,
        &[1.0],
        ScalarInputBounds::new(-10.0, 10.0).expect("valid test bounds"),
    )
    .unwrap();

    status
        .record_smt("scale", scale_smt)
        .expect("record scale smt");
    status
        .record_smt("snake", snake_smt)
        .expect("record snake smt");

    // Verify different encoding kinds.
    let scale_entry = status.kernel("scale").expect("scale entry");
    assert_eq!(
        scale_entry.smt.as_ref().unwrap().encoding,
        SmtEncodingKind::Exact
    );

    let snake_entry = status.kernel("snake").expect("snake entry");
    assert_eq!(
        snake_entry.smt.as_ref().unwrap().encoding,
        SmtEncodingKind::UfApprox
    );
}

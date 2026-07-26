// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Fusion recording tests (#803 AC1 + AC4).
//!
//! Tests for `verify_fusion_and_record()` — persisting fusion verification
//! results to `VerifyStatus`. Extracted from `fusion_equivalence.rs`.

use nn_dsl::{
    build_adain_scalar_kernel, build_adain_snake_fused_kernel, build_snake_scalar_kernel,
};
use nn_verify::{
    verify_fusion_and_record, verify_fusion_and_record_with_config, FusionSpec, NormBoundsMode,
    PropMethod, VerifyConfig, VerifyOutcome, VerifyStatus,
};

/// Representative input bounds for the dvoice Kokoro decoder.
const DVOICE_BOUNDS: [(f32, f32); 7] = [
    (-10.0, 10.0), // x: audio features after encoder
    (-5.0, 5.0),   // mu: channel mean
    (0.001, 10.0), // var: channel variance (positive)
    (0.1, 5.0),    // gamma: style scale
    (-3.0, 3.0),   // beta: style shift
    (0.01, 100.0), // alpha: snake activation parameter
    (1e-5, 1e-5),  // eps: constant epsilon (point interval)
];

#[test]
fn test_fusion_verify_and_record_point_inputs() {
    let fused = build_adain_snake_fused_kernel().expect("fused kernel");
    let adain = build_adain_scalar_kernel().expect("adain kernel");
    let snake = build_snake_scalar_kernel().expect("snake kernel");

    let point_bounds = [
        (1.0, 1.0),
        (0.0, 0.0),
        (1.0, 1.0),
        (1.0, 1.0),
        (0.0, 0.0),
        (1.0, 1.0),
        (1e-5, 1e-5),
    ];

    let spec = FusionSpec::new(&fused, &adain, &snake, 7, &[0, 1, 2, 3, 4, 6], &[0, 5], 0)
        .expect("valid fusion spec");

    let mut status = VerifyStatus::default();
    // Use 2e-6 epsilon to accommodate CROWN relaxation error (~1.19e-7 for f32)
    // on nonlinear ops (sin, pow, rsqrt) in the AdaIN+Snake fusion.
    let result = verify_fusion_and_record(&mut status, &spec, &point_bounds, 2e-6, None)
        .expect("fusion verify-and-record should succeed");

    // Point inputs → CROWN proves near-zero diff within epsilon → Verified.
    assert!(result.fusion.within_epsilon);
    assert!(result.fusion.is_conclusive());

    // AC1: Result is recorded in status with fusion_ prefix.
    // Fused kernel name is "adain_snake" (from KernelDef), so key is "fusion_adain_snake".
    let entry = status
        .kernel("fusion_adain_snake")
        .expect("fusion entry should exist in status");
    assert_eq!(entry.status, VerifyOutcome::Verified);
    assert_eq!(entry.method, PropMethod::Crown);

    // AC4: is_conclusive() is reflected via status (Verified = CROWN conclusive).
    assert_eq!(entry.output_width, result.fusion.max_abs_diff);
}

#[test]
fn test_fusion_record_with_custom_status_key() {
    let fused = build_adain_snake_fused_kernel().expect("fused kernel");
    let adain = build_adain_scalar_kernel().expect("adain kernel");
    let snake = build_snake_scalar_kernel().expect("snake kernel");

    let point_bounds = [
        (1.0, 1.0),
        (0.0, 0.0),
        (1.0, 1.0),
        (1.0, 1.0),
        (0.0, 0.0),
        (1.0, 1.0),
        (1e-5, 1e-5),
    ];

    let spec = FusionSpec::new(&fused, &adain, &snake, 7, &[0, 1, 2, 3, 4, 6], &[0, 5], 0)
        .expect("valid fusion spec");

    let mut status = VerifyStatus::default();
    let _result = verify_fusion_and_record(
        &mut status,
        &spec,
        &point_bounds,
        1e-10,
        Some("fusion_adain_snake_point"),
    )
    .expect("fusion verify-and-record should succeed");

    // Custom key should be used instead of default.
    assert!(status.kernel("fusion_adain_snake_point").is_some());
    assert!(status.kernel("fusion_adain_snake").is_none());
}

#[test]
fn test_fusion_record_dvoice_bounds() {
    let fused = build_adain_snake_fused_kernel().expect("fused kernel");
    let adain = build_adain_scalar_kernel().expect("adain kernel");
    let snake = build_snake_scalar_kernel().expect("snake kernel");

    let spec = FusionSpec::new(&fused, &adain, &snake, 7, &[0, 1, 2, 3, 4, 6], &[0, 5], 0)
        .expect("valid fusion spec");

    let mut status = VerifyStatus::default();
    let result = verify_fusion_and_record(
        &mut status,
        &spec,
        &DVOICE_BOUNDS,
        1e6,
        Some("fusion_adain_snake_dvoice"),
    )
    .expect("fusion verify-and-record should succeed");

    let entry = status
        .kernel("fusion_adain_snake_dvoice")
        .expect("dvoice fusion entry should exist");

    // Diff bounds should be finite.
    assert!(entry.output_bounds.lower.is_finite());
    assert!(entry.output_bounds.upper.is_finite());

    // Soundness provenance persisted (AC4).
    assert_eq!(entry.soundness_mode, result.fusion.soundness_mode);

    // History should have exactly 1 entry.
    assert_eq!(status.run_count("fusion_adain_snake_dvoice"), 1);
}

/// CROWN-conclusive results that exceed epsilon should record as BoundsComputed,
/// not Failed (#2225). This tests the recording logic fix that distinguishes
/// "CROWN succeeded with bounded diff" from "IBP fallback, vacuous bounds."
#[test]
fn test_fusion_record_crown_conclusive_exceeded_epsilon_is_bounds_computed() {
    let fused = build_adain_snake_fused_kernel().expect("fused kernel");
    let adain = build_adain_scalar_kernel().expect("adain kernel");
    let snake = build_snake_scalar_kernel().expect("snake kernel");

    let spec = FusionSpec::new(&fused, &adain, &snake, 7, &[0, 1, 2, 3, 4, 6], &[0, 5], 0)
        .expect("valid fusion spec");

    let mut status = VerifyStatus::default();
    // Use tight epsilon (1e-4) with wide bounds so CROWN diff exceeds epsilon
    // but CROWN itself succeeds (conclusive with bounded diff).
    let result = verify_fusion_and_record(
        &mut status,
        &spec,
        &DVOICE_BOUNDS,
        1e-4,
        Some("fusion_adain_exceeded"),
    )
    .expect("fusion verify-and-record should succeed");

    // CROWN should succeed (conclusive) but diff exceeds 1e-4.
    assert!(result.fusion.is_conclusive(), "CROWN should be conclusive");
    assert!(!result.fusion.within_epsilon, "diff should exceed epsilon");

    let entry = status
        .kernel("fusion_adain_exceeded")
        .expect("entry should exist");
    assert_eq!(
        entry.status,
        VerifyOutcome::BoundsComputed,
        "CROWN-conclusive exceeded-epsilon should be BoundsComputed, not Failed"
    );
    assert_eq!(entry.method, PropMethod::Crown);
}

/// `verify_fusion_and_record_with_config` passes VerifyConfig through to the
/// underlying verify function (#2225).
#[test]
fn test_fusion_record_with_config_propagates() {
    let fused = build_adain_snake_fused_kernel().expect("fused kernel");
    let adain = build_adain_scalar_kernel().expect("adain kernel");
    let snake = build_snake_scalar_kernel().expect("snake kernel");

    let point_bounds = [
        (1.0, 1.0),
        (0.0, 0.0),
        (1.0, 1.0),
        (1.0, 1.0),
        (0.0, 0.0),
        (1.0, 1.0),
        (1e-5, 1e-5),
    ];

    let spec = FusionSpec::new(&fused, &adain, &snake, 7, &[0, 1, 2, 3, 4, 6], &[0, 5], 0)
        .expect("valid fusion spec");

    let config = VerifyConfig::default().with_norm_mode(NormBoundsMode::ForwardMode);
    let mut status = VerifyStatus::default();
    let result = verify_fusion_and_record_with_config(
        &mut status,
        &spec,
        &point_bounds,
        2e-6,
        Some("fusion_adain_with_config"),
        &config,
    )
    .expect("fusion verify-and-record with config should succeed");

    assert!(result.fusion.is_conclusive());

    let entry = status
        .kernel("fusion_adain_with_config")
        .expect("entry should exist");
    assert_eq!(entry.status, VerifyOutcome::Verified);
    assert_eq!(entry.method, PropMethod::Crown);
}

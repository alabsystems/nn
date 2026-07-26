// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Per-kernel verification tests: IBP-only and full pipeline (NY + ay).
//!
//! Split from `kernel_pipeline_verify.rs` to stay under 500 lines (#533).

use super::*;

// --- K8 SiLU-Mul: x=Variable, up=Constant ---

#[test]
fn test_silu_mul_verify_and_record() {
    let kernel = build_silu_mul_kernel().expect("build K8 SiLU-Mul");
    let mut status = VerifyStatus::default();
    let result = verify_and_record(&mut status, &kernel, &[2.0], -5.0, 5.0);
    assert!(status.has_kernel("silu_mul"));
    // silu_mul(x, up=2) = silu(x) * 2. For x in [-5,5]:
    // silu(-5)*2 ≈ -0.067, silu(5)*2 ≈ 9.93.
    // IBP may overestimate but must contain the true range.
    assert_sound_bounds(&result, -0.07, 9.9, 100.0);
}

// --- K3 AdaIN: x=Variable, (mu, var, gamma, beta, eps)=Constants ---

#[test]
fn test_adain_verify_and_record() {
    let kernel = build_adain_scalar_kernel().expect("build K3 AdaIN");
    let mut status = VerifyStatus::default();
    // adain(x, mu=0, var=1, gamma=1, beta=0, eps=1e-5):
    // ≈ x * rsqrt(1.00001) ≈ x * 0.999995, so bounds ≈ [-5, 5]
    let result = verify_and_record(&mut status, &kernel, &[0.0, 1.0, 1.0, 0.0, 1e-5], -5.0, 5.0);
    assert!(status.has_kernel("adain"));
    // True range is [-4.999975, 4.999975] due to rsqrt(1.00001).
    assert_sound_bounds(&result, -4.99, 4.99, 50.0);
}

// --- K6 RoPE cos: x0=Variable, (x1, freq)=Constants ---

#[test]
fn test_rope_cos_verify_and_record() {
    let kernel = build_rope_cos_kernel().expect("build K6 rope_cos");
    let mut status = VerifyStatus::default();
    // rope_cos(x0, x1=1.0, freq=0.5) = x0*cos(0.5) - 1.0*sin(0.5)
    // = 0.8776*x0 - 0.4794. For x0 in [-10, 10]:
    // min ≈ 0.8776*(-10) - 0.4794 = -9.255, max ≈ 0.8776*10 - 0.4794 = 8.297
    let result = verify_and_record(&mut status, &kernel, &[1.0, 0.5], -10.0, 10.0);
    assert!(status.has_kernel("rope_cos"));
    // True range: [-9.2553, 8.2959]. Use inner values as soundness targets.
    assert_sound_bounds(&result, -9.25, 8.29, 50.0);
}

// --- K6 RoPE sin: x0=Variable, (x1, freq)=Constants ---

#[test]
fn test_rope_sin_verify_and_record() {
    let kernel = build_rope_sin_kernel().expect("build K6 rope_sin");
    let mut status = VerifyStatus::default();
    // rope_sin(x0, x1=1.0, freq=0.5) = x0*sin(0.5) + 1.0*cos(0.5)
    // = 0.4794*x0 + 0.8776. For x0 in [-10, 10]:
    // min ≈ 0.4794*(-10) + 0.8776 = -3.917, max ≈ 0.4794*10 + 0.8776 = 5.672
    let result = verify_and_record(&mut status, &kernel, &[1.0, 0.5], -10.0, 10.0);
    assert!(status.has_kernel("rope_sin"));
    // True range: [-3.9164, 5.6716]. Use inner values as soundness targets.
    assert_sound_bounds(&result, -3.91, 5.67, 50.0);
}

// --- Unified pipeline: NY + ay in one call (#397) ---

#[test]
fn test_verify_and_record_full_snake() {
    let kernel = common::snake_kernel();
    let mut status = VerifyStatus::default();
    let bounds = ScalarInputBounds::new(-10.0, 10.0).unwrap();

    let result =
        verify_and_record_full(&mut status, &kernel, &[1.0], bounds, None).expect("full pipeline");

    // NY result recorded
    assert!(status.has_kernel("snake"));
    assert!(result.gamma_crown.is_finite);
    assert!(
        result.cross_verified,
        "finite NY bounds should enable cross-verification (#428)"
    );

    // SMT result recorded (should have smt field populated)
    let entry = status.kernel("snake").unwrap();
    assert!(
        entry.smt.is_some(),
        "SMT result must be attached after full pipeline"
    );

    // JSON roundtrip: save, reload, verify SMT fields survive (#437)
    let tmp_path = std::env::temp_dir().join("nn_test_snake_smt_roundtrip.json");
    status.save(&tmp_path).expect("save status to temp file");

    let loaded = VerifyStatus::load(&tmp_path).expect("load status from temp file");
    let loaded_entry = loaded
        .kernel("snake")
        .expect("snake must exist after roundtrip");
    let loaded_smt = loaded_entry
        .smt
        .as_ref()
        .expect("SMT result must survive JSON roundtrip");

    // Snake uses UfApprox encoding (non-linear: sin, powi) → solver never invoked → Unexecuted
    assert_eq!(
        loaded_smt.outcome,
        nn_verify::SmtOutcome::Unexecuted,
        "snake SMT outcome should be Unexecuted after roundtrip, got {:?}",
        loaded_smt.outcome,
    );
    assert_eq!(
        loaded_smt.encoding,
        nn_verify::SmtEncodingKind::UfApprox,
        "snake SMT encoding should be UfApprox after roundtrip"
    );
    assert!(
        !loaded_smt.solver.is_empty(),
        "solver field must not be empty after roundtrip"
    );
    assert_eq!(
        loaded_smt.property, "output_bounded",
        "property must survive roundtrip"
    );

    // Cleanup
    let _ = std::fs::remove_file(&tmp_path);
}

#[test]
fn test_verify_and_record_full_silu_mul() {
    let kernel = build_silu_mul_kernel().expect("build K8 SiLU-Mul");
    let mut status = VerifyStatus::default();
    let bounds = ScalarInputBounds::new(-5.0, 5.0).unwrap();

    let result =
        verify_and_record_full(&mut status, &kernel, &[2.0], bounds, None).expect("full pipeline");

    assert!(status.has_kernel("silu_mul"));
    assert!(result.gamma_crown.is_finite);

    let entry = status.kernel("silu_mul").unwrap();
    assert!(
        entry.smt.is_some(),
        "SMT result must be attached after full pipeline"
    );
}

// --- Full pipeline (NY + ay) per kernel base type (#529) ---

#[test]
fn test_full_pipeline_adain() {
    let kernel = build_adain_scalar_kernel().expect("build adain");
    let mut status = VerifyStatus::default();
    // adain(x, mu=0, var=1, gamma=1, beta=0, eps=1e-5) — identity normalization
    let bounds = ScalarInputBounds::new(-5.0, 5.0).unwrap();

    let result = verify_and_record_full(
        &mut status,
        &kernel,
        &[0.0, 1.0, 1.0, 0.0, 1e-5],
        bounds,
        None,
    )
    .expect("adain full pipeline");

    assert!(result.gamma_crown.is_finite);
    assert!(result.cross_verified);
    let entry = status.kernel("adain").unwrap();
    assert!(entry.smt.is_some(), "adain must have SMT result");
}

#[test]
fn test_full_pipeline_adain_snake() {
    let kernel = build_adain_snake_fused_kernel().expect("build adain_snake");
    let mut status = VerifyStatus::default();
    // adain_snake(x, mu=0, var=1, gamma=1, beta=0, alpha=1, eps=1e-5) — fused kernel
    let bounds = ScalarInputBounds::new(-5.0, 5.0).unwrap();

    let result = verify_and_record_full(
        &mut status,
        &kernel,
        &[0.0, 1.0, 1.0, 0.0, 1.0, 1e-5],
        bounds,
        None,
    )
    .expect("adain_snake full pipeline");

    assert!(result.gamma_crown.is_finite);
    assert!(result.cross_verified);
    let entry = status.kernel("adain_snake").unwrap();
    assert!(entry.smt.is_some(), "adain_snake must have SMT result");
}

#[test]
fn test_full_pipeline_rope_cos() {
    let kernel = build_rope_cos_kernel().expect("build rope_cos");
    let mut status = VerifyStatus::default();
    // rope_cos(x0, x1=1.0, freq=0.5)
    let bounds = ScalarInputBounds::new(-10.0, 10.0).unwrap();

    let result = verify_and_record_full(&mut status, &kernel, &[1.0, 0.5], bounds, None)
        .expect("rope_cos full pipeline");

    assert!(result.gamma_crown.is_finite);
    assert!(result.cross_verified);
    let entry = status.kernel("rope_cos").unwrap();
    assert!(entry.smt.is_some(), "rope_cos must have SMT result");
}

#[test]
fn test_full_pipeline_rope_sin() {
    let kernel = build_rope_sin_kernel().expect("build rope_sin");
    let mut status = VerifyStatus::default();
    // rope_sin(x0, x1=1.0, freq=0.5)
    let bounds = ScalarInputBounds::new(-10.0, 10.0).unwrap();

    let result = verify_and_record_full(&mut status, &kernel, &[1.0, 0.5], bounds, None)
        .expect("rope_sin full pipeline");

    assert!(result.gamma_crown.is_finite);
    assert!(result.cross_verified);
    let entry = status.kernel("rope_sin").unwrap();
    assert!(entry.smt.is_some(), "rope_sin must have SMT result");
}

#[test]
fn test_full_pipeline_layer_norm() {
    let kernel = build_layer_norm_scalar_kernel().expect("build layer_norm");
    let mut status = VerifyStatus::default();
    // layer_norm(x, mu=0, var=1, eps=1e-5, gamma=1, beta=0)
    let bounds = ScalarInputBounds::new(-5.0, 5.0).unwrap();

    let result = verify_and_record_full(
        &mut status,
        &kernel,
        &[0.0, 1.0, 1e-5, 1.0, 0.0],
        bounds,
        None,
    )
    .expect("layer_norm full pipeline");

    assert!(result.gamma_crown.is_finite);
    assert!(result.cross_verified);
    let entry = status.kernel("layer_norm_scalar").unwrap();
    assert!(entry.smt.is_some(), "layer_norm must have SMT result");
}

#[test]
fn test_full_pipeline_rms_norm() {
    let kernel = build_rms_norm_scalar_kernel().expect("build rms_norm");
    let mut status = VerifyStatus::default();
    // rms_norm(x, rms_weight=1.0, gamma=1.0)
    let bounds = ScalarInputBounds::new(-5.0, 5.0).unwrap();

    let result = verify_and_record_full(&mut status, &kernel, &[1.0, 1.0], bounds, None)
        .expect("rms_norm full pipeline");

    assert!(result.gamma_crown.is_finite);
    assert!(result.cross_verified);
    let entry = status.kernel("rms_norm_scalar").unwrap();
    assert!(entry.smt.is_some(), "rms_norm must have SMT result");
}

#[test]
fn test_full_pipeline_instance_norm() {
    let kernel = build_instance_norm_scalar_kernel().expect("build instance_norm");
    let mut status = VerifyStatus::default();
    // instance_norm(x, mu=0, var=1, eps=1e-5)
    let bounds = ScalarInputBounds::new(-5.0, 5.0).unwrap();

    let result = verify_and_record_full(&mut status, &kernel, &[0.0, 1.0, 1e-5], bounds, None)
        .expect("instance_norm full pipeline");

    assert!(result.gamma_crown.is_finite);
    assert!(result.cross_verified);
    let entry = status.kernel("instance_norm_scalar").unwrap();
    assert!(entry.smt.is_some(), "instance_norm must have SMT result");
}

#[test]
fn test_full_pipeline_instance_norm_affine() {
    let kernel = build_instance_norm_affine_scalar_kernel().expect("build instance_norm_affine");
    let mut status = VerifyStatus::default();
    // instance_norm_affine(x, mu=0, var=1, eps=1e-5, gamma=1, beta=0)
    let bounds = ScalarInputBounds::new(-5.0, 5.0).unwrap();

    let result = verify_and_record_full(
        &mut status,
        &kernel,
        &[0.0, 1.0, 1e-5, 1.0, 0.0],
        bounds,
        None,
    )
    .expect("instance_norm_affine full pipeline");

    assert!(result.gamma_crown.is_finite);
    assert!(result.cross_verified);
    let entry = status.kernel("instance_norm_affine_scalar").unwrap();
    assert!(
        entry.smt.is_some(),
        "instance_norm_affine must have SMT result"
    );
}

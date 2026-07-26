// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Advanced verification tests: multi-variable, CROWN escalation, status_key,
//! and cross_verified invariants.
//!
//! Split from `kernel_pipeline_verify.rs` to stay under 500 lines (#533).

use super::*;

// --- K8 SiLU-Mul: multi-variable path (both x and up are Variable) ---

#[test]
fn test_silu_mul_multi_variable_bounds() {
    let kernel = build_silu_mul_kernel().expect("build K8 SiLU-Mul");
    let bindings = [ParamBinding::Variable, ParamBinding::Variable];
    let variable_bounds = [(-5.0f32, 5.0f32), (-3.0f32, 3.0f32)];

    let result = VerifyRequest::new(&kernel)
        .bindings(&bindings)
        .variable_bounds(&variable_bounds)
        .verify_bounds()
        .expect("silu_mul multi-variable IBP should succeed");

    assert!(
        result.is_finite,
        "silu_mul multi-variable bounds not finite"
    );
    // silu_mul(x, up) with x in [-5,5], up in [-3,3].
    // silu(5)*3 ≈ 14.90, silu(5)*(-3) ≈ -14.90.
    // IBP must contain the true range; width should be reasonable.
    // Use conservative inner bounds that are definitely achievable.
    assert_sound_bounds(&result, -14.89, 14.89, 200.0);
}

// --- Multi-variable unified pipeline (#411) ---

#[test]
fn test_verify_and_record_full_multi_silu_mul() {
    let kernel = build_silu_mul_kernel().expect("build K8 SiLU-Mul");
    let mut status = VerifyStatus::default();

    // silu_mul(x, up) — both as Variable
    let bindings = [ParamBinding::Variable, ParamBinding::Variable];
    let variable_bounds = [(-5.0f32, 5.0f32), (-3.0f32, 3.0f32)];

    let result =
        verify_and_record_full_multi(&mut status, &kernel, &bindings, &variable_bounds, None)
            .expect("multi-variable full pipeline");

    // NY result recorded
    assert!(status.has_kernel("silu_mul"));
    assert!(result.gamma_crown.is_finite);

    // SMT result recorded (should have smt field populated)
    let entry = status.kernel("silu_mul").unwrap();
    assert!(
        entry.smt.is_some(),
        "SMT result must be attached after multi-variable full pipeline"
    );

    // Verify the input_bounds record has 2 variable inputs (not 1)
    assert_eq!(
        entry.input_bounds.variable_inputs.len(),
        2,
        "multi-variable pipeline should record 2 variable inputs"
    );
    assert_eq!(entry.input_bounds.variable_inputs[0].param_index, 0);
    assert_eq!(entry.input_bounds.variable_inputs[1].param_index, 1);
}

// --- status_key creates distinct entries without overwriting (#521, #526) ---

#[test]
fn test_status_key_creates_distinct_entry() {
    let kernel = common::snake_kernel();
    let mut status = VerifyStatus::default();
    let bounds = ScalarInputBounds::new(-10.0, 10.0).unwrap();

    // Record with default key (None) — uses kernel.name ("snake")
    verify_and_record_full(&mut status, &kernel, &[1.0], bounds, None).expect("full pipeline None");
    assert!(status.has_kernel("snake"), "default key should be 'snake'");

    // Record with override key — creates a separate entry
    verify_and_record_full(&mut status, &kernel, &[2.0], bounds, Some("snake_alpha_2"))
        .expect("full pipeline Some");
    assert!(
        status.has_kernel("snake_alpha_2"),
        "override key should create 'snake_alpha_2'"
    );

    // Both entries exist (not overwritten)
    assert!(
        status.has_kernel("snake"),
        "original 'snake' entry must not be overwritten"
    );
    assert!(
        status.kernel_count() >= 2,
        "expected at least 2 entries, got {}",
        status.kernel_count()
    );

    // Verify entries are independently recorded with correct data
    let default_entry = status.kernel("snake").unwrap();
    let override_entry = status.kernel("snake_alpha_2").unwrap();
    assert_eq!(default_entry.status, nn_verify::VerifyOutcome::Verified);
    assert_eq!(override_entry.status, nn_verify::VerifyOutcome::Verified);
    // Both should have SMT results (from full pipeline)
    assert!(default_entry.smt.is_some(), "default entry should have SMT");
    assert!(
        override_entry.smt.is_some(),
        "override entry should have SMT"
    );
}

#[test]
fn test_status_key_multi_variable_creates_distinct_entry() {
    let kernel = build_silu_mul_kernel().expect("build K8 SiLU-Mul");
    let mut status = VerifyStatus::default();

    let bindings = [ParamBinding::Variable, ParamBinding::Variable];
    let variable_bounds = [(-5.0f32, 5.0f32), (-3.0f32, 3.0f32)];

    // Record with default key (None)
    verify_and_record_full_multi(&mut status, &kernel, &bindings, &variable_bounds, None)
        .expect("multi None");
    assert!(status.has_kernel("silu_mul"));

    // Record with override key
    verify_and_record_full_multi(
        &mut status,
        &kernel,
        &bindings,
        &variable_bounds,
        Some("silu_mul_wide"),
    )
    .expect("multi Some");
    assert!(status.has_kernel("silu_mul_wide"));

    // Both entries exist
    assert!(status.has_kernel("silu_mul"));
    assert!(status.kernel_count() >= 2);
}

// --- cross_verified field reflects NY finiteness (#428) ---

#[test]
fn test_pipeline_cross_verified_true_on_finite_bounds() {
    // Positive case: normal bounds produce finite NY output → cross_verified true.
    let kernel = build_silu_mul_kernel().expect("build K8 SiLU-Mul");
    let mut status = VerifyStatus::default();
    let bounds = ScalarInputBounds::new(-5.0, 5.0).unwrap();

    let result =
        verify_and_record_full(&mut status, &kernel, &[2.0], bounds, None).expect("full pipeline");

    assert!(result.gamma_crown.is_finite);
    assert!(
        result.cross_verified,
        "finite NY bounds should enable cross-verification (#428)"
    );
}

#[test]
fn test_pipeline_cross_verified_field_tracks_is_finite() {
    // Verify that cross_verified == gamma_crown.is_finite on a representative kernel.
    // The non-finite path (cross_verified=false) requires input magnitudes beyond ay's
    // real encoding limit (~9.2e12), so full pipeline end-to-end tests only exercise
    // the finite/true path. This test verifies the logical invariant holds.
    let kernel = common::snake_kernel();
    let mut status = VerifyStatus::default();
    let bounds = ScalarInputBounds::new(-10.0, 10.0).unwrap();

    let result =
        verify_and_record_full(&mut status, &kernel, &[1.0], bounds, None).expect("full pipeline");

    assert_eq!(
        result.cross_verified, result.gamma_crown.is_finite,
        "cross_verified must equal gamma_crown.is_finite"
    );
}

// --- TranslatedKernel integration in pipeline (#719) ---
//
// AC2: The pipeline (verify_and_record_full) uses TranslatedKernel internally
// for the ay step. Verify the SMT result uses CallerProvided bounds source
// and produces the same outcome as a direct TranslatedKernel call.

#[test]
fn test_pipeline_smt_uses_translated_kernel() {
    use nn_verify::{BoundsSource, SmtEncodingKind, TranslatedKernel};

    // Use add_one (exact linear) — most predictable for comparing outcomes.
    let kernel = {
        use nn_dsl::ir::*;
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
    };
    let mut status = VerifyStatus::default();
    let bounds = ScalarInputBounds::new(-10.0, 10.0).unwrap();

    // Run through the full pipeline.
    let result =
        verify_and_record_full(&mut status, &kernel, &[], bounds, None).expect("full pipeline");

    // NY should produce finite bounds for add_one.
    assert!(result.gamma_crown.is_finite);
    assert!(result.cross_verified);

    // SMT result should use CallerProvided bounds source — the hallmark of
    // TranslatedKernel (not Analytical or Heuristic).
    assert_eq!(
        result.smt.bounds_source,
        BoundsSource::CallerProvided,
        "pipeline SMT step should use CallerProvided (TranslatedKernel path)"
    );
    assert_eq!(result.smt.encoding, SmtEncodingKind::Exact);

    // Cross-check: direct TranslatedKernel call with same NY bounds.
    let expected = (
        f64::from(result.gamma_crown.output_lower),
        f64::from(result.gamma_crown.output_upper),
    );
    let mut tk = TranslatedKernel::from_kernel(&kernel, &[], bounds).unwrap();
    let direct = tk.check_output_bounded(expected).unwrap();

    assert_eq!(
        result.smt.outcome, direct.outcome,
        "pipeline and direct TranslatedKernel should produce identical outcome"
    );
    assert_eq!(result.smt.encoding, direct.encoding);
    assert_eq!(result.smt.bounds_source, direct.bounds_source);
    assert_eq!(result.smt.expected_bounds, direct.expected_bounds);
}

// --- CROWN escalation path tests (#529 AC2) ---
//
// These verify that the IBP → CROWN escalation path works for kernels beyond
// snake. Using a low threshold (5.0) forces CROWN when IBP width exceeds it.

#[test]
fn test_crown_escalation_silu_mul() {
    let kernel = build_silu_mul_kernel().expect("build silu_mul");
    let input_bounds = scalar_input_bounds(-5.0, 5.0).expect("input bounds");
    let crown_config = VerifyConfig::with_threshold(5.0).expect("valid threshold");

    let result = VerifyRequest::new(&kernel)
        .constant_params(&[2.0])
        .input_bounds(&input_bounds)
        .config(crown_config)
        .verify_bounds()
        .expect("CROWN escalation should succeed for silu_mul");

    assert!(result.is_finite, "CROWN result must be finite");
    // CROWN may or may not be triggered depending on IBP width; either method is valid.
    // The important thing is the pipeline doesn't panic/error.
}

#[test]
fn test_crown_escalation_adain() {
    let kernel = build_adain_scalar_kernel().expect("build adain");
    let input_bounds = scalar_input_bounds(-5.0, 5.0).expect("input bounds");
    let crown_config = VerifyConfig::with_threshold(5.0).expect("valid threshold");

    // adain(x, mu=0, var=1, gamma=1, beta=0, eps=1e-5)
    let result = VerifyRequest::new(&kernel)
        .constant_params(&[0.0, 1.0, 1.0, 0.0, 1e-5])
        .input_bounds(&input_bounds)
        .config(crown_config)
        .verify_bounds()
        .expect("CROWN escalation should succeed for adain");

    assert!(result.is_finite, "CROWN result must be finite");
}

// --- Multi-variable full pipeline for RoPE (#529 AC3) ---
//
// RoPE kernels naturally have 2 variable inputs (x0, x1) with freq as constant.

#[test]
fn test_full_pipeline_rope_cos_multi_variable() {
    let kernel = build_rope_cos_kernel().expect("build rope_cos");
    let mut status = VerifyStatus::default();

    // rope_cos(x0=Variable, x1=Variable, freq=0.5)
    let bindings = [
        ParamBinding::Variable,
        ParamBinding::Variable,
        ParamBinding::Constant(0.5),
    ];
    let variable_bounds = [(-10.0f32, 10.0f32), (-5.0f32, 5.0f32)];

    let result =
        verify_and_record_full_multi(&mut status, &kernel, &bindings, &variable_bounds, None)
            .expect("rope_cos multi-variable full pipeline");

    assert!(result.gamma_crown.is_finite);
    assert!(result.cross_verified);

    let entry = status.kernel("rope_cos").unwrap();
    assert!(entry.smt.is_some(), "rope_cos multi must have SMT result");
    assert_eq!(
        entry.input_bounds.variable_inputs.len(),
        2,
        "rope_cos multi should record 2 variable inputs"
    );
}

#[test]
fn test_full_pipeline_rope_sin_multi_variable() {
    let kernel = build_rope_sin_kernel().expect("build rope_sin");
    let mut status = VerifyStatus::default();

    // rope_sin(x0=Variable, x1=Variable, freq=0.5)
    let bindings = [
        ParamBinding::Variable,
        ParamBinding::Variable,
        ParamBinding::Constant(0.5),
    ];
    let variable_bounds = [(-10.0f32, 10.0f32), (-5.0f32, 5.0f32)];

    let result =
        verify_and_record_full_multi(&mut status, &kernel, &bindings, &variable_bounds, None)
            .expect("rope_sin multi-variable full pipeline");

    assert!(result.gamma_crown.is_finite);
    assert!(result.cross_verified);

    let entry = status.kernel("rope_sin").unwrap();
    assert!(entry.smt.is_some(), "rope_sin multi must have SMT result");
    assert_eq!(
        entry.input_bounds.variable_inputs.len(),
        2,
        "rope_sin multi should record 2 variable inputs"
    );
}

// --- _with_config variants (#843) ---
//
// Verify that the scalar pipeline _with_config variants accept and use a
// custom VerifyConfig, producing the same results as the default variants.

#[test]
fn test_verify_and_record_full_with_config_snake() {
    let kernel = common::snake_kernel();
    let mut status = VerifyStatus::default();
    let bounds = ScalarInputBounds::new(-10.0, 10.0).unwrap();
    let config = VerifyConfig::with_threshold(5.0).expect("valid threshold");

    let result = verify_and_record_full_with_config(
        &mut status,
        &kernel,
        &[1.0],
        bounds,
        Some("snake_custom_config"),
        &config,
    )
    .expect("full pipeline with custom config");

    assert!(status.has_kernel("snake_custom_config"));
    assert!(result.gamma_crown.is_finite);
    assert!(
        result.cross_verified,
        "finite NY bounds should enable cross-verification"
    );
    let entry = status.kernel("snake_custom_config").unwrap();
    assert!(
        entry.smt.is_some(),
        "SMT result must be attached after _with_config pipeline"
    );
}

#[test]
fn test_verify_and_record_full_multi_with_config_silu_mul() {
    let kernel = build_silu_mul_kernel().expect("build K8 SiLU-Mul");
    let mut status = VerifyStatus::default();
    let bindings = [ParamBinding::Variable, ParamBinding::Variable];
    let variable_bounds = [(-5.0f32, 5.0f32), (-3.0f32, 3.0f32)];
    let config = VerifyConfig::with_threshold(5.0).expect("valid threshold");

    let result = verify_and_record_full_multi_with_config(
        &mut status,
        &kernel,
        &bindings,
        &variable_bounds,
        Some("silu_mul_custom_config"),
        &config,
    )
    .expect("multi-variable full pipeline with custom config");

    assert!(status.has_kernel("silu_mul_custom_config"));
    assert!(result.gamma_crown.is_finite);
    let entry = status.kernel("silu_mul_custom_config").unwrap();
    assert!(
        entry.smt.is_some(),
        "SMT result must be attached after multi _with_config pipeline"
    );
    assert_eq!(
        entry.input_bounds.variable_inputs.len(),
        2,
        "multi-variable _with_config should record 2 variable inputs"
    );
}

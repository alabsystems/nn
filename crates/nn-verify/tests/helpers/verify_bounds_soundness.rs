// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Soundness provenance tests (#189, #194) for kernel bounds verification.
//!
//! Core bounds tests: verify_bounds.rs
//! Multi-variable tests: verify_bounds_multi.rs
//! Validation/edge-case tests: verify_bounds_validation.rs

use super::common::{exp_kernel, snake_kernel, unary_fn_kernel};
use nn_dsl::ir::UnaryFnKind;
use nn_verify::{
    scalar_input_bounds, Bound, PropMethod, VerificationSoundnessMode, VerifyConfig, VerifyRequest,
};

// ---------------------------------------------------------------------------
// Soundness provenance tests (#189)
// ---------------------------------------------------------------------------

#[test]
fn test_ibp_snake_soundness_mode_is_sound() {
    // IBP on scalar kernel ops (sin, cos, exp) does not use heuristic
    // relaxations — soundness_mode must be Sound. This verifies AC1: the
    // soundness_mode field is extracted from graph content, not hardcoded.
    let kernel = snake_kernel();
    let input_bounds = scalar_input_bounds(-5.0, 5.0).expect("bounds");
    let config = VerifyConfig::with_threshold(1e10).expect("valid threshold");

    let result = VerifyRequest::new(&kernel)
        .constant_params(&[1.0])
        .input_bounds(&input_bounds)
        .config(config)
        .verify_bounds()
        .expect("IBP verification should pass");

    assert_eq!(result.method, PropMethod::Ibp);
    assert_eq!(
        result.soundness_mode,
        VerificationSoundnessMode::Sound,
        "IBP on scalar kernel ops should report Sound provenance"
    );
}

#[test]
fn test_crown_snake_soundness_mode_is_sound() {
    // Native SnakeLayer uses analytical CROWN linear relaxation (not sampling-
    // based SinLayer). Its CROWN backward is provably sound, so the soundness
    // mode must be Sound. Before #338, snake was decomposed through SinLayer
    // (sampling-based, Heuristic). Native SnakeLayer (#338 AC1) fixed this.
    let kernel = snake_kernel();
    let input_bounds = scalar_input_bounds(-5.0, 5.0).expect("bounds");
    // Threshold 0.0 forces CROWN escalation.
    let config = VerifyConfig::with_threshold(0.0).expect("valid threshold");

    let result = VerifyRequest::new(&kernel)
        .constant_params(&[1.0])
        .input_bounds(&input_bounds)
        .config(config)
        .verify_bounds()
        .expect("CROWN verification should pass");

    // CROWN may succeed or fall back to IBP. Either way, native SnakeLayer
    // reports Sound (IBP is always sound; SnakeLayer CROWN is analytical).
    assert_eq!(
        result.soundness_mode,
        VerificationSoundnessMode::Sound,
        "Native SnakeLayer CROWN is analytically sound"
    );
}

#[test]
fn test_require_sound_bounds_path_accepts_native_snake() {
    // With native SnakeLayer (#338), CROWN is analytically sound (not heuristic).
    // require_sound=true should now PASS for snake kernels, since the native
    // layer's CROWN backward is provably sound. Before #338, this would reject
    // with SoundnessRequired because decomposed SinLayer used sampling.
    let kernel = snake_kernel();
    let input_bounds = scalar_input_bounds(-5.0, 5.0).expect("bounds");
    let config = VerifyConfig::with_threshold(0.0)
        .expect("valid threshold")
        .with_require_sound(true);

    let result = VerifyRequest::new(&kernel)
        .constant_params(&[1.0])
        .input_bounds(&input_bounds)
        .config(config)
        .verify_bounds()
        .expect("require_sound=true should pass with native SnakeLayer (Sound)");

    assert_eq!(
        result.soundness_mode,
        VerificationSoundnessMode::Sound,
        "Native SnakeLayer CROWN is sound; require_sound should accept"
    );
}

#[test]
fn test_require_sound_bounds_path_passes_for_ibp() {
    // require_sound=true with a high threshold (IBP only) should pass since
    // IBP never uses heuristic relaxations.
    let kernel = snake_kernel();
    let input_bounds = scalar_input_bounds(-5.0, 5.0).expect("bounds");
    let config = VerifyConfig::with_threshold(1e10)
        .expect("valid threshold")
        .with_require_sound(true);

    let result = VerifyRequest::new(&kernel)
        .constant_params(&[1.0])
        .input_bounds(&input_bounds)
        .config(config)
        .verify_bounds()
        .expect("require_sound=true should pass when IBP result is Sound");

    assert_eq!(result.method, PropMethod::Ibp);
    assert_eq!(result.soundness_mode, VerificationSoundnessMode::Sound);
}

// ---------------------------------------------------------------------------
// SqrtNegativeDomain + AC3 path agreement tests (#194)
// ---------------------------------------------------------------------------

#[test]
fn test_sqrt_negative_domain_bounds_path_reports_heuristic() {
    // AC1 (#194): sqrt(x) with x ∈ [-5, 5] includes negative-domain inputs.
    // NY clamps sqrt(x<0) to 0 (a heuristic). The bounds path must
    // classify this as Heuristic, not Sound.
    let kernel = unary_fn_kernel(UnaryFnKind::Sqrt);
    let input_bounds = scalar_input_bounds(-5.0, 5.0).expect("bounds");
    let config = VerifyConfig::with_threshold(1e10).expect("valid threshold");

    let result = VerifyRequest::new(&kernel)
        .constant_params(&[])
        .input_bounds(&input_bounds)
        .config(config)
        .verify_bounds()
        .expect("IBP verification should pass");

    assert_eq!(result.method, PropMethod::Ibp);
    assert_eq!(
        result.soundness_mode,
        VerificationSoundnessMode::Heuristic,
        "sqrt(x) with x ∈ [-5, 5] must report Heuristic (SqrtNegativeDomain)"
    );
}

#[test]
fn test_sqrt_positive_domain_bounds_path_reports_sound() {
    // Complementary to the negative-domain test: sqrt(x) with x ∈ [1, 5] has
    // no negative-domain inputs, so no SqrtNegativeDomain heuristic applies.
    let kernel = unary_fn_kernel(UnaryFnKind::Sqrt);
    let input_bounds = scalar_input_bounds(1.0, 5.0).expect("bounds");
    let config = VerifyConfig::with_threshold(1e10).expect("valid threshold");

    let result = VerifyRequest::new(&kernel)
        .constant_params(&[])
        .input_bounds(&input_bounds)
        .config(config)
        .verify_bounds()
        .expect("IBP verification should pass");

    assert_eq!(result.method, PropMethod::Ibp);
    assert_eq!(
        result.soundness_mode,
        VerificationSoundnessMode::Sound,
        "sqrt(x) with x ∈ [1, 5] must report Sound (no negative domain)"
    );
}

#[test]
fn test_sqrt_negative_domain_bounds_and_spec_paths_agree() {
    // AC3 (#194): bounds path and spec path must agree on soundness classification
    // for the same kernel with the same input bounds. Using sqrt(x) with negative
    // domain — both should classify as Heuristic.
    let kernel = unary_fn_kernel(UnaryFnKind::Sqrt);
    let input_bounds = scalar_input_bounds(-5.0, 5.0).expect("bounds");
    let config = VerifyConfig::with_threshold(1e10).expect("valid threshold");

    // Bounds path
    let bounds_result = VerifyRequest::new(&kernel)
        .constant_params(&[])
        .input_bounds(&input_bounds)
        .config(config.clone())
        .verify_bounds()
        .expect("bounds verification should pass");
    let bounds_soundness = bounds_result.soundness_mode;

    // Spec path — use wide output spec so verification succeeds
    let output_spec = vec![Bound::new(-100.0, 100.0)];
    let spec_result = VerifyRequest::new(&kernel)
        .constant_params(&[])
        .input_bounds(&input_bounds)
        .required_output_bounds(&output_spec)
        .config(config)
        .verify_spec()
        .expect("spec verification should pass");
    let spec_soundness = spec_result.result.provenance().mode();

    assert_eq!(
        bounds_soundness, spec_soundness,
        "bounds path ({bounds_soundness:?}) and spec path ({spec_soundness:?}) \
         must agree on soundness for sqrt(x) with x ∈ [-5, 5]"
    );
    // Both should be Heuristic for this case
    assert_eq!(bounds_soundness, VerificationSoundnessMode::Heuristic);
}

#[test]
fn test_sound_kernel_bounds_and_spec_paths_agree() {
    // AC3 (#194): complementary test — for a kernel with no heuristics (exp(x)),
    // both paths should agree on Sound classification.
    let kernel = exp_kernel();
    let input_bounds = scalar_input_bounds(-5.0, 5.0).expect("bounds");
    let config = VerifyConfig::with_threshold(1e10).expect("valid threshold");

    // Bounds path
    let bounds_result = VerifyRequest::new(&kernel)
        .constant_params(&[])
        .input_bounds(&input_bounds)
        .config(config.clone())
        .verify_bounds()
        .expect("bounds verification should pass");
    let bounds_soundness = bounds_result.soundness_mode;

    // Spec path — use wide output spec
    let output_spec = vec![Bound::new(0.0, 200.0)];
    let spec_result = VerifyRequest::new(&kernel)
        .constant_params(&[])
        .input_bounds(&input_bounds)
        .required_output_bounds(&output_spec)
        .config(config)
        .verify_spec()
        .expect("spec verification should pass");
    let spec_soundness = spec_result.result.provenance().mode();

    assert_eq!(
        bounds_soundness, spec_soundness,
        "bounds path ({bounds_soundness:?}) and spec path ({spec_soundness:?}) \
         must agree on soundness for exp(x)"
    );
    assert_eq!(bounds_soundness, VerificationSoundnessMode::Sound);
}

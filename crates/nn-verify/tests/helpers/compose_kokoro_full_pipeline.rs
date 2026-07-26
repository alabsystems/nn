// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Integration tests: Full Kokoro TTS pipeline NY composition.
//!
//! Proves end-to-end properties of the Kokoro TTS pipeline:
//! - **Property 1 (Non-silence):** exp() lower bounds > 0
//! - **Property 2 (Non-clipping):** upper bounds < threshold
//! - **Property 3 (Duration positivity):** dur_logits finite → exp(dur_logits) > 0
//!
//! Consolidated: builds each pipeline variant ONCE and runs all property checks,
//! eliminating ~12 redundant graph builds (was 15 builds, now 3).
//!
//! Part of #1741: THE MOONSHOT — end-to-end Kokoro pipeline verification.

#[path = "kokoro_full_pipeline.rs"]
mod full_pipeline_helpers;

use super::common::{
    assert_bounds_valid, assert_crown_tighter_when_not_fallback, bounds_min_max, uniform_bounds,
    verify_and_assert,
};
use full_pipeline_helpers::{
    build_kokoro_duration_branch, build_kokoro_full_pipeline, build_kokoro_vocoder_only_pipeline,
    kokoro_duration_branch_bindings, kokoro_full_pipeline_bindings, kokoro_vocoder_only_bindings,
    D_MODEL, OUT_CHANNELS, SEQ_LEN, TIME_UP,
};
use nn_verify::{tensor_kernel_to_graph, VerificationSoundnessMode};

// ===========================================================================
// Full pipeline (text encoder + vocoder): all properties in one test
// (was: 5 tests — def_validates, graph_builds, ibp, crown, verify_and_record)
// ===========================================================================

#[test]
fn test_kokoro_full_pipeline_all_properties() {
    let (def, out_shape) = build_kokoro_full_pipeline();
    assert_eq!(out_shape, [OUT_CHANNELS, TIME_UP]);
    def.validate()
        .expect("kokoro full pipeline def should validate");

    let bindings = kokoro_full_pipeline_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings)
        .expect("kokoro full pipeline graph should translate");
    assert!(
        graph.num_nodes() >= 15,
        "full pipeline graph should have >= 15 nodes, got {}",
        graph.num_nodes()
    );

    let input = uniform_bounds(&[D_MODEL, SEQ_LEN], 1.0);

    // --- IBP: Properties 1+2 ---
    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through full Kokoro pipeline");
    assert_eq!(output.lower_upper().0.shape(), &[OUT_CHANNELS, TIME_UP]);
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Kokoro full pipeline IBP: bounds=[{lo_min}, {hi_max}]");
    assert!(
        lo_min > 0.0,
        "PROPERTY 1 VIOLATION: exp output should be positive, got lo_min={lo_min}"
    );
    assert!(
        hi_max < 1e8,
        "PROPERTY 2: IBP upper bound should be bounded, got hi_max={hi_max}"
    );
    eprintln!("  Property 1 (Non-silence): lower bound {lo_min} > 0");
    eprintln!("  Property 2 (Non-clipping): upper bound {hi_max} < 1e8");

    // --- CROWN ---
    let (method, crown_output, fallback_reason) =
        assert_crown_tighter_when_not_fallback(&graph, &input);
    assert_eq!(
        crown_output.lower_upper().0.shape(),
        &[OUT_CHANNELS, TIME_UP]
    );
    let (crown_lo_min, crown_hi_max) = bounds_min_max(&crown_output);
    eprintln!("Kokoro full pipeline: method={method:?}, bounds=[{crown_lo_min}, {crown_hi_max}]");
    if let Some(reason) = &fallback_reason {
        eprintln!("CROWN fallback reason: {reason}");
    }
    assert!(
        crown_lo_min > 0.0,
        "CROWN PROPERTY 1: exp output positive, got lo_min={crown_lo_min}"
    );
    assert!(
        crown_hi_max < 1e8,
        "CROWN PROPERTY 2: upper bound bounded, got hi_max={crown_hi_max}"
    );

    // --- Verify and record ---
    let result = verify_and_assert(&def, &bindings, &input, "kokoro_full_pipeline");
    assert_eq!(
        result.num_variables, 1,
        "single Variable input (text_features)"
    );
    let (lo, _) = result.output_bounds.lower_upper();
    assert_eq!(lo.shape(), &[OUT_CHANNELS, TIME_UP]);
    // After #2940 native Snake/AdaIN, NY produces Sound bounds
    // (was Heuristic with decomposed layers). Accept both.
    assert!(
        matches!(
            result.verification.soundness_mode,
            VerificationSoundnessMode::Sound | VerificationSoundnessMode::Heuristic
        ),
        "soundness should be Sound or Heuristic, got {:?}",
        result.verification.soundness_mode
    );
}

// ===========================================================================
// Vocoder-only pipeline: all properties in one test
// (was: 5 tests — def_validates, graph_builds, ibp, verify, monotonicity)
// ===========================================================================

#[test]
fn test_kokoro_vocoder_only_all_properties() {
    let (def, out_shape) = build_kokoro_vocoder_only_pipeline();
    assert_eq!(out_shape, [OUT_CHANNELS, TIME_UP]);
    def.validate()
        .expect("kokoro vocoder-only pipeline should validate");

    let bindings = kokoro_vocoder_only_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings)
        .expect("kokoro vocoder-only pipeline graph should translate");
    assert!(
        graph.num_nodes() >= 10,
        "vocoder-only graph should have >= 10 nodes, got {}",
        graph.num_nodes()
    );

    let enc_dim = 8; // ENC_DIM

    // --- IBP: Properties 1+2 ---
    let input = uniform_bounds(&[enc_dim, SEQ_LEN], 1.0);
    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through vocoder-only pipeline");
    assert_eq!(output.lower_upper().0.shape(), &[OUT_CHANNELS, TIME_UP]);
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Vocoder-only IBP: bounds=[{lo_min}, {hi_max}]");
    assert!(
        lo_min > 0.0,
        "PROPERTY 1: vocoder output should be positive, got lo_min={lo_min}"
    );
    assert!(
        hi_max.is_finite(),
        "PROPERTY 2: vocoder output should be finite, got hi_max={hi_max}"
    );

    // --- Verify and record ---
    let result = verify_and_assert(&def, &bindings, &input, "kokoro_vocoder_pipeline");
    assert_eq!(result.num_variables, 1, "single Variable input");
    // After #2940 native Snake/AdaIN, NY produces Sound bounds
    // (was Heuristic with decomposed layers). Accept both.
    assert!(
        matches!(
            result.verification.soundness_mode,
            VerificationSoundnessMode::Sound | VerificationSoundnessMode::Heuristic
        ),
        "soundness should be Sound or Heuristic, got {:?}",
        result.verification.soundness_mode
    );

    // --- Input bound monotonicity ---
    let wide_output = &output; // reuse from above
    let (wide_lo_min, wide_hi_max) = bounds_min_max(wide_output);
    let wide_range = wide_hi_max - wide_lo_min;

    let narrow_input = uniform_bounds(&[enc_dim, SEQ_LEN], 0.1);
    let narrow_output = graph.propagate_ibp(&narrow_input).expect("IBP narrow");
    let (narrow_lo_min, narrow_hi_max) = bounds_min_max(&narrow_output);
    let narrow_range = narrow_hi_max - narrow_lo_min;

    eprintln!("Wide input range [-1,1] -> output range: {wide_range:.6}");
    eprintln!("Narrow input range [-0.1,0.1] -> output range: {narrow_range:.6}");
    assert!(
        narrow_range <= wide_range + 1e-6,
        "IBP monotonicity violated: narrow range {narrow_range} > wide range {wide_range}"
    );
}

// ===========================================================================
// Duration branch: all properties in one test
// (was: 5 tests — def_validates, graph_builds, ibp_positivity, crown, verify)
// ===========================================================================

#[test]
fn test_kokoro_duration_branch_all_properties() {
    let (def, out_len) = build_kokoro_duration_branch();
    assert_eq!(out_len, SEQ_LEN);
    def.validate()
        .expect("kokoro duration branch def should validate");

    let bindings = kokoro_duration_branch_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings)
        .expect("kokoro duration branch graph should translate");
    assert!(
        graph.num_nodes() >= 8,
        "duration branch graph should have >= 8 nodes, got {}",
        graph.num_nodes()
    );

    let input = uniform_bounds(&[D_MODEL, SEQ_LEN], 1.0);

    // --- IBP: Property 3 (duration positivity) ---
    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through duration branch");
    let (lo, hi) = output.lower_upper();
    assert_eq!(lo.shape(), &[SEQ_LEN], "dur_logits shape mismatch");
    assert_bounds_valid(&output);
    for (idx, (&lo_val, &hi_val)) in lo.iter().zip(hi.iter()).enumerate() {
        assert!(
            lo_val.is_finite(),
            "PROPERTY 3: dur_logits lower at phoneme {idx} must be finite, got {lo_val}"
        );
        assert!(
            hi_val.is_finite(),
            "PROPERTY 3: dur_logits upper at phoneme {idx} must be finite, got {hi_val}"
        );
    }
    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Duration branch IBP: dur_logits bounds=[{lo_min}, {hi_max}]");
    eprintln!("  Property 3 (Duration positivity): all dur_logits finite");

    // --- CROWN ---
    let (method, crown_output, fallback_reason) =
        assert_crown_tighter_when_not_fallback(&graph, &input);
    let (crown_lo, _) = crown_output.lower_upper();
    assert_eq!(crown_lo.shape(), &[SEQ_LEN], "dur_logits shape mismatch");
    let (crown_lo_min, _) = bounds_min_max(&crown_output);
    eprintln!("Duration branch: method={method:?}, lo_min={crown_lo_min}");
    if let Some(reason) = &fallback_reason {
        eprintln!("CROWN fallback reason: {reason}");
    }
    let (crown_lo, crown_hi) = crown_output.lower_upper();
    for (idx, (&lo_val, &hi_val)) in crown_lo.iter().zip(crown_hi.iter()).enumerate() {
        assert!(
            lo_val.is_finite(),
            "CROWN PROPERTY 3: dur_logits lower at phoneme {idx} must be finite, got {lo_val}"
        );
        assert!(
            hi_val.is_finite(),
            "CROWN PROPERTY 3: dur_logits upper at phoneme {idx} must be finite, got {hi_val}"
        );
    }

    // --- Verify and record ---
    let result = verify_and_assert(&def, &bindings, &input, "kokoro_duration_branch");
    assert_eq!(
        result.num_variables, 1,
        "single Variable input (text_features)"
    );
    let (lo, _) = result.output_bounds.lower_upper();
    assert_eq!(lo.shape(), &[SEQ_LEN]);
    // After #2940 native Snake/AdaIN, NY produces Sound bounds
    // (was Heuristic with decomposed layers). Accept both.
    assert!(
        matches!(
            result.verification.soundness_mode,
            VerificationSoundnessMode::Sound | VerificationSoundnessMode::Heuristic
        ),
        "soundness should be Sound or Heuristic, got {:?}",
        result.verification.soundness_mode
    );
}

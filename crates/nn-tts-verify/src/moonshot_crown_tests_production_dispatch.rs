// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Integration tests: production dispatch plans × P8 implementation correctness.
//!
//! Connects the actual Kokoro-82M and Silero VAD dispatch plan builders with
//! `analyze_dispatch_plan` to produce real P8 (implementation correctness)
//! ay coverage metrics. All prior P8 tests used synthetic dispatch steps —
//! these tests verify the ay coverage fractions for production models.
//!
//! Part of #1741. Updated for #2917 (binary_add/mul ay coverage).

use super::*;
use crate::kokoro_dispatch::{
    build_kokoro_dispatch_plan, build_kokoro_dispatch_plan_default,
    TOTAL_EXPECTED_STEPS as KOKORO_TOTAL_STEPS,
};
use crate::moonshot::VerificationLevel;
use crate::silero_vad_dispatch::{
    build_silero_vad_dispatch_plan_default, TOTAL_EXPECTED_STEPS as VAD_TOTAL_STEPS,
};

// ---------------------------------------------------------------------------
// Kokoro-82M vocoder
// ---------------------------------------------------------------------------

#[test]
fn test_kokoro_dispatch_plan_p8_analysis() {
    let (steps, _t_final) = build_kokoro_dispatch_plan_default();
    assert_eq!(steps.len(), KOKORO_TOTAL_STEPS);

    let evidence = analyze_dispatch_plan(&steps);

    // Kokoro has 181 total steps. Metadata-only steps (none expected in
    // Kokoro's current plan) are excluded from the denominator.
    assert!(
        evidence.total_steps > 0,
        "should have numerical steps to analyze"
    );

    // Kokoro uses: Conv1d (unproven), ConvTranspose1d (unproven),
    // Linear/AdaIN (unproven), Sigmoid/Snake (ay-proven), Tanh (ay-proven),
    // BinaryAdd (ay-proven via scalar "add" bounds, #2917).
    //
    // Proven categories: sigmoid, tanh_act, add
    // Unproven categories: conv1d, conv_transpose1d, linear
    assert!(
        evidence.proven_steps > 0,
        "Kokoro should have ay-proven activation steps"
    );
    assert!(
        !evidence.all_proven,
        "Kokoro has conv/linear ops without ay proofs"
    );

    // Verify specific proven categories are present
    assert!(
        evidence.proven_categories.contains(&"sigmoid".to_string()),
        "sigmoid should be ay-proven (Snake/LeakyReLU), got: {:?}",
        evidence.proven_categories
    );
    assert!(
        evidence.proven_categories.contains(&"tanh_act".to_string()),
        "tanh_act should be ay-proven (exp/sin), got: {:?}",
        evidence.proven_categories
    );
    assert!(
        evidence.proven_categories.contains(&"add".to_string()),
        "add should be ay-proven (BinaryAdd, #2917), got: {:?}",
        evidence.proven_categories
    );

    // Verify unproven categories are present
    assert!(
        evidence.unproven_categories.contains(&"conv1d".to_string()),
        "conv1d should be unproven, got: {:?}",
        evidence.unproven_categories
    );
    assert!(
        evidence
            .unproven_categories
            .contains(&"conv_transpose1d".to_string()),
        "conv_transpose1d should be unproven, got: {:?}",
        evidence.unproven_categories
    );
    assert!(
        evidence.unproven_categories.contains(&"linear".to_string()),
        "linear should be unproven, got: {:?}",
        evidence.unproven_categories
    );
}

#[test]
fn test_kokoro_p8_verification_level() {
    let (steps, _) = build_kokoro_dispatch_plan_default();
    let evidence = analyze_dispatch_plan(&steps);
    let result = check_implementation_correctness(&evidence);

    assert_eq!(result.property_index, 7, "P8 is index 7");
    assert!(
        !result.proven,
        "Kokoro not fully ay-proven (has conv/linear)"
    );

    // Kokoro's ay coverage fraction (#2917):
    // 51 Sigmoid (Snake/LeakyReLU) + 2 Tanh (exp/sin) + 26 BinaryAdd = 79 proven
    // out of 181 total numerical steps = 43.6%
    // This is < 50%, so should be Empirical.
    let fraction = evidence.proven_steps as f64 / evidence.total_steps as f64;
    assert!(
        fraction > 0.4 && fraction < 0.5,
        "fraction should be ~43.6%, got {fraction:.3}"
    );

    // With ~44% coverage (sigmoid+tanh+add), Kokoro is below 50% threshold
    assert_eq!(
        result.level,
        VerificationLevel::Empirical,
        "Kokoro should be Empirical at ~44% ay coverage (fraction={fraction:.3})"
    );
}

#[test]
fn test_kokoro_exact_proven_step_count() {
    let (steps, _) = build_kokoro_dispatch_plan_default();
    let evidence = analyze_dispatch_plan(&steps);

    // From kokoro_dispatch_tests.rs:
    // 51 Sigmoid + 2 Tanh + 26 BinaryAdd = 79 ay-proven (#2917)
    // Total numerical: 181 (no metadata-only steps in Kokoro plan)
    assert_eq!(
        evidence.total_steps, 181,
        "all 181 Kokoro steps are numerical"
    );
    assert_eq!(
        evidence.proven_steps, 79,
        "51 Sigmoid + 2 Tanh + 26 BinaryAdd = 79 proven steps"
    );
}

#[test]
fn test_kokoro_p8_explanation_contains_details() {
    let (steps, _) = build_kokoro_dispatch_plan_default();
    let evidence = analyze_dispatch_plan(&steps);
    let result = check_implementation_correctness(&evidence);

    assert!(
        result.explanation.contains("79/181"),
        "explanation should show 79/181 fraction, got: {}",
        result.explanation
    );
    assert!(
        result.explanation.contains("sigmoid"),
        "explanation should mention sigmoid"
    );
    assert!(
        result.explanation.contains("conv1d"),
        "explanation should mention conv1d gap"
    );
}

// ---------------------------------------------------------------------------
// Silero VAD
// ---------------------------------------------------------------------------

#[test]
fn test_silero_vad_dispatch_plan_p8_analysis() {
    let steps = build_silero_vad_dispatch_plan_default();
    assert_eq!(steps.len(), VAD_TOTAL_STEPS);

    let evidence = analyze_dispatch_plan(&steps);

    assert!(
        evidence.total_steps > 0,
        "should have numerical steps to analyze"
    );

    // Silero VAD uses: Conv1d (unproven), Relu (ay-proven),
    // Linear (unproven), Sigmoid (ay-proven), Tanh (ay-proven),
    // BinaryAdd (ay-proven #2917), BinaryMul (ay-proven #2917),
    // Reduce (unproven).
    assert!(
        evidence.proven_steps > 0,
        "Silero VAD should have ay-proven activation steps"
    );
    assert!(
        !evidence.all_proven,
        "Silero VAD has conv/linear/reduce ops without ay proofs"
    );

    // Proven categories: add, mul, relu, sigmoid, tanh_act
    assert!(
        evidence.proven_categories.contains(&"relu".to_string()),
        "relu should be ay-proven, got: {:?}",
        evidence.proven_categories
    );
    assert!(
        evidence.proven_categories.contains(&"sigmoid".to_string()),
        "sigmoid should be ay-proven, got: {:?}",
        evidence.proven_categories
    );
    assert!(
        evidence.proven_categories.contains(&"tanh_act".to_string()),
        "tanh_act should be ay-proven, got: {:?}",
        evidence.proven_categories
    );
    assert!(
        evidence.proven_categories.contains(&"add".to_string()),
        "add should be ay-proven (BinaryAdd, #2917), got: {:?}",
        evidence.proven_categories
    );
    assert!(
        evidence.proven_categories.contains(&"mul".to_string()),
        "mul should be ay-proven (BinaryMul, #2917), got: {:?}",
        evidence.proven_categories
    );

    // Unproven: conv1d, linear, reduce
    assert!(
        evidence.unproven_categories.contains(&"conv1d".to_string()),
        "conv1d should be unproven"
    );
    assert!(
        evidence.unproven_categories.contains(&"linear".to_string()),
        "linear should be unproven"
    );
}

#[test]
fn test_silero_vad_exact_proven_step_count() {
    let steps = build_silero_vad_dispatch_plan_default();
    let evidence = analyze_dispatch_plan(&steps);

    // Silero VAD: 24 total steps, all numerical (no metadata-only)
    // Proven (#2917): 5 Relu + 4 Sigmoid + 2 Tanh + 2 BinaryAdd + 3 BinaryMul = 16
    // Unproven: 4 Conv1d + 1 Reduce + 3 Linear = 8 unproven
    assert_eq!(evidence.total_steps, 24, "all 24 VAD steps are numerical");
    assert_eq!(
        evidence.proven_steps, 16,
        "5 Relu + 4 Sigmoid + 2 Tanh + 2 BinaryAdd + 3 BinaryMul = 16 proven steps"
    );
}

#[test]
fn test_silero_vad_p8_verification_level() {
    let steps = build_silero_vad_dispatch_plan_default();
    let evidence = analyze_dispatch_plan(&steps);
    let result = check_implementation_correctness(&evidence);

    assert_eq!(result.property_index, 7, "P8 is index 7");
    assert!(!result.proven, "Silero VAD not fully ay-proven");

    // 16/24 = 66.7% >= 50%, so CrownPartial (#2917)
    let fraction = evidence.proven_steps as f64 / evidence.total_steps as f64;
    assert!(
        fraction > 0.6 && fraction < 0.7,
        "VAD fraction should be ~66.7%, got {fraction:.3}"
    );
    assert_eq!(
        result.level,
        VerificationLevel::CrownPartial,
        "Silero VAD should be CrownPartial at ~67% ay coverage (#2917)"
    );
}

// ---------------------------------------------------------------------------
// Cross-model comparison
// ---------------------------------------------------------------------------

#[test]
fn test_silero_vad_higher_ay_fraction_than_kokoro() {
    let (kokoro_steps, _) = build_kokoro_dispatch_plan_default();
    let vad_steps = build_silero_vad_dispatch_plan_default();

    let kokoro_ev = analyze_dispatch_plan(&kokoro_steps);
    let vad_ev = analyze_dispatch_plan(&vad_steps);

    let kokoro_frac = kokoro_ev.proven_steps as f64 / kokoro_ev.total_steps as f64;
    let vad_frac = vad_ev.proven_steps as f64 / vad_ev.total_steps as f64;

    // Silero VAD has higher ay coverage because its architecture has
    // proportionally more activation + binary ops (LSTM gates: sigmoid+tanh,
    // binary add/mul for cell updates) relative to its total op count
    // than Kokoro (which is dominated by Conv1d/ConvTranspose1d/Linear).
    assert!(
        vad_frac > kokoro_frac,
        "VAD ay fraction ({vad_frac:.3}) should exceed Kokoro ({kokoro_frac:.3})"
    );
}

#[test]
fn test_both_models_share_proven_categories() {
    let (kokoro_steps, _) = build_kokoro_dispatch_plan_default();
    let vad_steps = build_silero_vad_dispatch_plan_default();

    let kokoro_ev = analyze_dispatch_plan(&kokoro_steps);
    let vad_ev = analyze_dispatch_plan(&vad_steps);

    // Both models should have sigmoid and add in their proven categories
    assert!(kokoro_ev.proven_categories.contains(&"sigmoid".to_string()));
    assert!(vad_ev.proven_categories.contains(&"sigmoid".to_string()));
    assert!(kokoro_ev.proven_categories.contains(&"add".to_string()));
    assert!(vad_ev.proven_categories.contains(&"add".to_string()));

    // VAD has relu (encoder activations), Kokoro does not (uses Snake/Sigmoid)
    assert!(vad_ev.proven_categories.contains(&"relu".to_string()));
}

// ---------------------------------------------------------------------------
// Stability: ay coverage is deterministic across seq_len
// ---------------------------------------------------------------------------

#[test]
fn test_kokoro_ay_coverage_invariant_across_seq_len() {
    let (steps_10, _) = build_kokoro_dispatch_plan(10);
    let (steps_200, _) = build_kokoro_dispatch_plan(200);

    let ev_10 = analyze_dispatch_plan(&steps_10);
    let ev_200 = analyze_dispatch_plan(&steps_200);

    // Same topology → same step counts → same ay fraction
    assert_eq!(ev_10.total_steps, ev_200.total_steps);
    assert_eq!(ev_10.proven_steps, ev_200.proven_steps);
    assert_eq!(ev_10.proven_categories, ev_200.proven_categories);
    assert_eq!(ev_10.unproven_categories, ev_200.unproven_categories);
}

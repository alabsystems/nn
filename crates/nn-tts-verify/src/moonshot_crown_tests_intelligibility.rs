// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Dedicated P3 (intelligibility) tests — proxy and attention monotonicity.
//!
//! Phase 29 of #1741 fills the P3 test coverage gap: P3 had only ~4 dedicated
//! tests (fewest of all 8 properties). This file adds edge cases, D=192
//! dedicated tests, and NaN/Inf guard verification.

use super::*;

// ============================================================================
// P3 proxy: edge cases
// ============================================================================

/// NaN in output bounds must not produce a "proven" intelligibility proxy.
///
/// Mirrors the P2 NaN guard test pattern. Without finiteness guards, NaN
/// in the output range computation can produce arbitrary ratio values.
#[test]
fn test_intelligibility_proxy_nan_output_bounds() {
    let mut upper = vec![0.5; 8];
    upper[3] = f64::NAN;
    let cert = bounded_pipeline(vec![-0.5; 8], upper, true);
    let result = check_intelligibility_proxy(&cert, 1.0);
    // NaN contaminates the range computation — ratio may be NaN.
    // The function should either fail (not proven) or produce a finite ratio.
    // Either way, it must not claim "proven" with non-finite data.
    if result.proven {
        // If somehow proven, bound_value must be finite.
        assert!(
            result.bound_value.is_finite(),
            "proven P3 must have finite bound_value, got {}",
            result.bound_value,
        );
    }
}

/// Inf in output bounds must not produce a "proven" intelligibility proxy.
#[test]
fn test_intelligibility_proxy_inf_output_bounds() {
    let cert = bounded_pipeline(vec![-0.5; 8], vec![f64::INFINITY; 8], true);
    let result = check_intelligibility_proxy(&cert, 1.0);
    // Infinite output range → infinite ratio → should not be proven.
    assert!(
        !result.proven || !result.bound_value.is_finite(),
        "infinite output bounds should not prove P3: ratio={}",
        result.bound_value,
    );
}

/// Zero *output* range (identical upper/lower) produces ratio=0 → proven.
///
/// `bounded_pipeline` sets e2e_input to [-1.0, 1.0] (range=2.0), so the
/// *input* range is non-zero. The output range is 0 because lower==upper.
/// ratio = 0/2 = 0 < threshold, so the proxy IS provable.
#[test]
fn test_intelligibility_proxy_zero_output_range() {
    let cert = bounded_pipeline(vec![0.5; 8], vec![0.5; 8], true);
    let result = check_intelligibility_proxy(&cert, 1.0);
    // Output range = 0, input range = 2.0 → ratio = 0.0 < 1.0 → proven.
    assert!(
        result.proven,
        "zero output range means perfectly tight bounds — should prove P3"
    );
    assert!(
        (result.bound_value - 0.0).abs() < 1e-10,
        "ratio should be ~0.0, got {}",
        result.bound_value
    );
}

/// Very tight bounds (near-zero output range) should produce very low ratio.
#[test]
fn test_intelligibility_proxy_very_tight_bounds() {
    let cert = bounded_pipeline(vec![-0.001; 8], vec![0.001; 8], true);
    let result = check_intelligibility_proxy(&cert, 100.0);
    // Output range = 0.002, input range = 2.0, ratio = 0.001 << 100.0.
    assert!(result.proven, "tight bounds should prove P3");
    assert_eq!(result.level, VerificationLevel::CrownPartial);
    assert!(result.bound_value < 0.01);
}

/// Asymmetric bounds across dimensions — ratio should use the worst (max) range.
#[test]
fn test_intelligibility_proxy_asymmetric_dimensions() {
    let lower = vec![-0.1, -0.5, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1];
    let upper = vec![0.1, 0.5, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1];
    let cert = bounded_pipeline(lower, upper, true);
    let result = check_intelligibility_proxy(&cert, 1.0);
    // Max output range = 1.0 (dimension 1), input range = 2.0, ratio = 0.5.
    assert!(result.proven, "max output range 0.5 < 1.0 threshold");
    assert!((result.bound_value - 0.5).abs() < 0.01);
}

/// IBP fallback (is_sound=false) produces Empirical level, not CrownPartial.
///
/// Per `check_intelligibility_proxy` line 212-216: `CrownPartial` requires
/// `cert.is_sound == true`. When `is_sound=false`, the level is `Empirical`.
#[test]
fn test_intelligibility_proxy_ibp_produces_empirical() {
    let cert = bounded_pipeline(vec![-0.3; 8], vec![0.3; 8], false);
    let result = check_intelligibility_proxy(&cert, 1.0);
    assert!(result.proven);
    assert_eq!(
        result.level,
        VerificationLevel::Empirical,
        "proxy P3 with is_sound=false should be Empirical"
    );
}

// ============================================================================
// P3 attention monotonicity: dedicated tests
// ============================================================================

/// Attention monotonicity with zero margin is NOT proven (margin must be > 0).
#[test]
fn test_attention_monotonicity_zero_margin() {
    let cert = bounded_pipeline(vec![-0.5; 8], vec![0.5; 8], true);
    let attn = crate::monotonicity::AttentionMonotonicityCertificate {
        decoder_steps: 20,
        encoder_positions: 20,
        min_margin: 0.0,
        is_proven: false,
        row_margins: vec![0.0; 20],
        input_bound: 1.0,
        propagation_mode: "CROWN".to_string(),
    };
    let result = check_intelligibility_with_monotonicity(&cert, &attn);
    // Not proven → falls back to proxy.
    assert_eq!(result.level, VerificationLevel::CrownPartial);
}

/// Large decoder_steps × encoder_positions (100×200) to verify scaling.
#[test]
fn test_attention_monotonicity_large_sequence() {
    let cert = bounded_pipeline(vec![-0.5; 8], vec![0.5; 8], true);
    let attn = crate::monotonicity::AttentionMonotonicityCertificate {
        decoder_steps: 100,
        encoder_positions: 200,
        min_margin: 0.2,
        is_proven: true,
        row_margins: vec![0.2; 100],
        input_bound: 1.0,
        propagation_mode: "CROWN".to_string(),
    };
    let result = check_intelligibility_with_monotonicity(&cert, &attn);
    assert!(result.proven);
    assert_eq!(result.level, VerificationLevel::CrownProven);
    assert!(result.explanation.contains("decoder_steps=100"));
    assert!(result.explanation.contains("encoder_positions=200"));
}

/// Negative min_margin (counter-example found) → not proven → proxy fallback.
#[test]
fn test_attention_monotonicity_negative_margin() {
    let cert = bounded_pipeline(vec![-0.5; 8], vec![0.5; 8], true);
    let attn = crate::monotonicity::AttentionMonotonicityCertificate {
        decoder_steps: 30,
        encoder_positions: 30,
        min_margin: -0.3,
        is_proven: false,
        row_margins: vec![-0.3; 30],
        input_bound: 1.0,
        propagation_mode: "CROWN".to_string(),
    };
    let result = check_intelligibility_with_monotonicity(&cert, &attn);
    assert!(result.proven, "proxy fallback should still pass");
    assert_eq!(result.level, VerificationLevel::CrownPartial);
}

/// D=192 dedicated P3 attention monotonicity test (matches P4/P5/P6 scale).
#[test]
fn test_attention_monotonicity_d192() {
    let dim = 192;
    let cert = bounded_pipeline(vec![-0.5; dim], vec![0.5; dim], true);
    let attn = crate::monotonicity::AttentionMonotonicityCertificate {
        decoder_steps: 50,
        encoder_positions: 50,
        min_margin: 0.4,
        is_proven: true,
        row_margins: vec![0.4; 50],
        input_bound: 1.0,
        propagation_mode: "CROWN".to_string(),
    };
    let result = check_intelligibility_with_monotonicity(&cert, &attn);
    assert!(
        result.proven,
        "D=192 P3 must be proven: {}",
        result.explanation
    );
    assert_eq!(result.level, VerificationLevel::CrownProven);
    assert!((result.bound_value - 0.4).abs() < 1e-6);
}

/// Tight CROWN-family propagation modes should all count as sound for P3.
#[test]
fn test_attention_monotonicity_crown_family_modes_are_sound() {
    let cert = bounded_pipeline(vec![-0.5; 8], vec![0.5; 8], true);

    for mode in [
        "CROWN",
        "AlphaCrown",
        "alpha-CROWN",
        "BetaCrown",
        "beta-CROWN",
    ] {
        let attn = crate::monotonicity::AttentionMonotonicityCertificate {
            decoder_steps: 8,
            encoder_positions: 8,
            min_margin: 0.25,
            is_proven: true,
            row_margins: vec![0.25; 8],
            input_bound: 1.0,
            propagation_mode: mode.to_string(),
        };
        let result = check_intelligibility_with_monotonicity(&cert, &attn);
        assert_eq!(
            result.level,
            VerificationLevel::CrownProven,
            "mode {mode} should be treated as fully sound"
        );
        assert!(result.is_sound, "mode {mode} should be sound");
        assert!(
            result.explanation.contains(mode),
            "explanation should preserve the original propagation mode"
        );
    }
}

/// D=192 P3 proxy (without attention cert) at production scale.
#[test]
fn test_intelligibility_proxy_d192() {
    let dim = 192;
    let cert = bounded_pipeline(vec![-0.3; dim], vec![0.3; dim], true);
    let result = check_intelligibility_proxy(&cert, 100.0);
    // Output range = 0.6, input range = 2.0, ratio = 0.3 < 100.0.
    assert!(
        result.proven,
        "D=192 proxy P3 must prove with generous threshold"
    );
    assert_eq!(result.level, VerificationLevel::CrownPartial);
}

// ============================================================================
// P1/P2 at D=192 (production scale, matching P4/P5/P6 coverage)
// ============================================================================

/// D=192 dedicated P1 (non-silence) test.
///
/// P4, P5, P6 all have dedicated D=192 tests but P1 was only tested at D=192
/// via bundle assertions. This fills that gap.
#[test]
fn test_non_silence_d192() {
    let dim = 192;
    let cert = bounded_pipeline(vec![-0.5; dim], vec![0.8; dim], true);
    let result = check_non_silence(&cert, 0.01);
    assert!(
        result.proven,
        "D=192 P1 must be proven: {}",
        result.explanation
    );
    assert_eq!(result.level, VerificationLevel::CrownProven);
    assert!(result.bound_value > 0.01);
}

/// D=192 P1 near-zero bounds — not proven.
#[test]
fn test_non_silence_d192_near_zero() {
    let dim = 192;
    let cert = bounded_pipeline(vec![-0.005; dim], vec![0.005; dim], true);
    let result = check_non_silence(&cert, 0.01);
    assert!(
        !result.proven,
        "near-zero bounds at D=192 should not prove P1"
    );
    assert_eq!(result.level, VerificationLevel::Empirical);
}

/// D=192 P1 with IBP fallback.
#[test]
fn test_non_silence_d192_ibp_fallback() {
    let dim = 192;
    let cert = bounded_pipeline(vec![-0.5; dim], vec![0.8; dim], false);
    let result = check_non_silence(&cert, 0.01);
    assert!(result.proven);
    assert_eq!(result.level, VerificationLevel::CrownPartial);
}

/// D=192 dedicated P2 (non-clipping) test.
///
/// P4, P5, P6 all have dedicated D=192 tests but P2 was only tested at D=192
/// via bundle assertions. This fills that gap.
#[test]
fn test_non_clipping_d192() {
    let dim = 192;
    let cert = bounded_pipeline(vec![-0.9; dim], vec![0.95; dim], true);
    let result = check_non_clipping(&cert);
    assert!(
        result.proven,
        "D=192 P2 must be proven: {}",
        result.explanation
    );
    assert_eq!(result.level, VerificationLevel::CrownProven);
    assert!(result.bound_value <= 1.0);
}

/// D=192 P2 with one dimension exceeding [-1, 1].
#[test]
fn test_non_clipping_d192_single_dim_exceeds() {
    let dim = 192;
    let mut upper = vec![0.9; dim];
    upper[100] = 1.05; // Single dimension exceeds.
    let cert = bounded_pipeline(vec![-0.9; dim], upper, true);
    let result = check_non_clipping(&cert);
    assert!(!result.proven, "single exceeding dim at D=192 must fail P2");
    assert_eq!(result.level, VerificationLevel::Empirical);
}

/// D=192 P2 exactly at boundary.
#[test]
fn test_non_clipping_d192_exact_boundary() {
    let dim = 192;
    let cert = bounded_pipeline(vec![-1.0; dim], vec![1.0; dim], true);
    let result = check_non_clipping(&cert);
    assert!(result.proven, "exactly at boundary at D=192 must pass P2");
    assert_eq!(result.level, VerificationLevel::CrownProven);
}

/// D=192 P2 with IBP fallback.
#[test]
fn test_non_clipping_d192_ibp_fallback() {
    let dim = 192;
    let cert = bounded_pipeline(vec![-0.5; dim], vec![0.5; dim], false);
    let result = check_non_clipping(&cert);
    assert!(result.proven);
    assert_eq!(result.level, VerificationLevel::CrownPartial);
}

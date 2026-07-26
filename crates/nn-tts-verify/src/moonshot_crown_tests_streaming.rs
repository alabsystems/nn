// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! P6 (streaming safety) property tests, bundle tests, and D=192 production tests.

use super::*;

// ---------------------------------------------------------------------------
// Property 6: Streaming safety
// ---------------------------------------------------------------------------

#[test]
fn test_streaming_safety_proven_tight_bounds() {
    // Tight output bounds [-0.1, 0.1] with 240-sample crossfade.
    // Range = 0.2, alpha_step = 1/239 ≈ 0.00418.
    // max_click_bound = 0.2 * 0.00418 ≈ 0.000837 < 0.3 threshold.
    let cert = bounded_pipeline(vec![-0.1; 8], vec![0.1; 8], true);
    let result = check_streaming_safety(&cert, 240, 0.3);
    assert!(result.proven);
    assert_eq!(result.level, VerificationLevel::CrownProven);
    assert!(result.bound_value < 0.01); // very small click bound
    assert!(result.explanation.contains("PROVEN"));
    assert_eq!(result.property_index, 5);
    assert_eq!(
        result.property_name,
        "Streaming-safe (bounded chunk discontinuity)"
    );
}

#[test]
fn test_streaming_safety_proven_with_default_crossfade() {
    // Output bounds [-0.5, 0.5], range = 1.0.
    // alpha_step = 1/239 ≈ 0.00418.
    // max_click_bound = 1.0 * 0.00418 ≈ 0.00418 < 0.3.
    let cert = bounded_pipeline(vec![-0.5; 8], vec![0.5; 8], true);
    let result = check_streaming_safety(&cert, 240, 0.3);
    assert!(result.proven);
    assert_eq!(result.level, VerificationLevel::CrownProven);
}

#[test]
fn test_streaming_safety_fails_wide_bounds() {
    // Very wide output bounds [-100, 100], range = 200.
    // alpha_step = 1/239 ≈ 0.00418.
    // max_click_bound = 200 * 0.00418 ≈ 0.837 > 0.3.
    let cert = bounded_pipeline(vec![-100.0; 8], vec![100.0; 8], true);
    let result = check_streaming_safety(&cert, 240, 0.3);
    assert!(!result.proven);
    assert_eq!(result.level, VerificationLevel::Empirical);
    assert!(result.bound_value > 0.3);
}

#[test]
fn test_streaming_safety_fails_short_crossfade() {
    // Output bounds [-0.5, 0.5], range = 1.0.
    // Short crossfade: 2 samples → alpha_step = 1.0.
    // max_click_bound = 1.0 * 1.0 = 1.0 > 0.3.
    let cert = bounded_pipeline(vec![-0.5; 8], vec![0.5; 8], true);
    let result = check_streaming_safety(&cert, 2, 0.3);
    assert!(!result.proven);
    assert_eq!(result.level, VerificationLevel::Empirical);
}

#[test]
fn test_streaming_safety_ibp_fallback() {
    let cert = bounded_pipeline(vec![-0.1; 8], vec![0.1; 8], false);
    let result = check_streaming_safety(&cert, 240, 0.3);
    assert!(result.proven);
    assert_eq!(result.level, VerificationLevel::CrownPartial);
}

#[test]
fn test_streaming_safety_single_sample_crossfade() {
    // Degenerate: crossfade_samples = 1 → alpha_step = 1.0.
    let cert = bounded_pipeline(vec![-0.1; 8], vec![0.1; 8], true);
    let result = check_streaming_safety(&cert, 1, 0.3);
    // Range = 0.2, step = 1.0, bound = 0.2 < 0.3.
    assert!(result.proven);
}

// ---------------------------------------------------------------------------
// Bundle tests with streaming (P6)
// ---------------------------------------------------------------------------

#[test]
fn test_bundle_includes_streaming() {
    let cert = bounded_pipeline(vec![-0.3; 8], vec![0.3; 8], true);
    let bundle = verify_properties_from_pipeline(&cert, 64);
    // Now includes P0-P2 and P5 (streaming) = 4 results.
    assert_eq!(bundle.results.len(), 4);
    assert!(bundle.all_proven);
    // Verify the 4th result is streaming safety.
    assert_eq!(bundle.results[3].property_index, 5);
    assert!(bundle.results[3].proven);
}

#[test]
fn test_bundle_with_custom_streaming() {
    let cert = bounded_pipeline(vec![-0.3; 8], vec![0.3; 8], true);
    let bundle = verify_properties_from_pipeline_with_streaming(&cert, 64, 120, 0.1);
    assert_eq!(bundle.results.len(), 4);
    // With shorter crossfade (120 samples), alpha_step = 1/119 ≈ 0.0084.
    // Range = 0.6, bound = 0.6 * 0.0084 ≈ 0.00504 < 0.1.
    assert!(bundle.results[3].proven);
}

#[test]
fn test_bundle_display_includes_p6() {
    let cert = bounded_pipeline(vec![-0.3; 8], vec![0.3; 8], true);
    let bundle = verify_properties_from_pipeline(&cert, 64);
    let s = format!("{bundle}");
    assert!(s.contains("P6:"));
    assert!(s.contains("Streaming-safe"));
    assert!(s.contains("4/4 proven"));
}

// ---------------------------------------------------------------------------
// Production-dimension D=192 tests (#1741 Property 6 gap)
// ---------------------------------------------------------------------------

#[test]
fn test_streaming_safety_d192_production() {
    // D=192 matches Kokoro decoder hidden dimension.
    // Output bounds [-0.5, 0.5], range = 1.0.
    // crossfade = 240 samples, alpha_step = 1/239 ≈ 0.00418.
    // max_click_bound = 1.0 * 0.00418 ≈ 0.00418 < 0.3 threshold.
    let cert = bounded_pipeline(vec![-0.5; 192], vec![0.5; 192], true);
    let result = check_streaming_safety(&cert, 240, 0.3);
    assert!(result.proven, "streaming safety must pass at D=192");
    assert_eq!(result.level, VerificationLevel::CrownProven);
    assert!(result.bound_value < 0.01);
}

#[test]
fn test_all_moonshot_properties_d192() {
    // Full moonshot property bundle at production dimension D=192.
    // With tight output bounds [-0.3, 0.3], all 4 properties should pass.
    let cert = bounded_pipeline(vec![-0.3; 192], vec![0.3; 192], true);
    let bundle = verify_properties_from_pipeline(&cert, 192);
    assert_eq!(bundle.verification_dim, 192);
    assert_eq!(bundle.results.len(), 4);
    assert!(
        bundle.all_proven,
        "all 4 properties must pass at D=192: {bundle}"
    );
}

#[test]
fn test_moonshot_from_stages_d192() {
    // Three-stage pipeline at D=192, bridging through verify_moonshot_from_stages.
    let dim = 192;
    let stages = vec![
        VerifiedStage {
            name: "encoder".to_string(),
            input_lower: vec![-1.0; dim],
            input_upper: vec![1.0; dim],
            output_lower: vec![-0.8; dim],
            output_upper: vec![0.8; dim],
            input_shape: vec![1, dim],
            output_shape: vec![1, dim],
            method: "CROWN".to_string(),
            is_sound: true,
        },
        VerifiedStage {
            name: "prosody".to_string(),
            input_lower: vec![-1.0; dim],
            input_upper: vec![1.0; dim],
            output_lower: vec![-0.5; dim],
            output_upper: vec![0.5; dim],
            input_shape: vec![1, dim],
            output_shape: vec![1, dim],
            method: "CROWN".to_string(),
            is_sound: true,
        },
        VerifiedStage {
            name: "decoder".to_string(),
            input_lower: vec![-1.0; dim],
            input_upper: vec![1.0; dim],
            output_lower: vec![-0.4; dim],
            output_upper: vec![0.4; dim],
            input_shape: vec![1, dim],
            output_shape: vec![1, dim],
            method: "CROWN".to_string(),
            is_sound: true,
        },
    ];

    let bundle = verify_moonshot_from_stages(&stages, dim).expect("valid stages");
    assert_eq!(bundle.verification_dim, 192);
    assert!(
        bundle.all_proven,
        "3-stage D=192 moonshot must pass: {bundle}"
    );
    // Streaming safety: range=0.8, step=1/239≈0.00418, bound≈0.00335 < 0.3
    assert!(bundle.results[3].proven);
    assert!(bundle.results[3].bound_value < 0.01);
}

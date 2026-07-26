// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! P5 (temporal boundedness) property tests, bundle tests, and D=192 production tests.

use super::*;

// ---------------------------------------------------------------------------
// Property 5: Temporal boundedness
// ---------------------------------------------------------------------------

/// Helper to construct a TimingCertificate from a bounded pipeline.
pub(super) fn timing_certificate(
    dim: usize,
    out_lower: Vec<f64>,
    out_upper: Vec<f64>,
    is_sound: bool,
    worst_case_time_us: f64,
    timing_bound_us: f64,
) -> (PipelineCertificate, TimingCertificate) {
    let cert = bounded_pipeline(out_lower, out_upper, is_sound);
    let timing_met = worst_case_time_us <= timing_bound_us;
    let timing_cert = TimingCertificate {
        bounds_cert: cert.clone(),
        cost_profiles: vec![crate::cost_model::LayerCostProfile {
            layer_name: "test_layer".to_string(),
            flops: 1_000_000,
            memory_bytes: 4 * dim as u64,
            estimated_time_us: worst_case_time_us,
            measured_time_us: None,
        }],
        worst_case_time_us,
        total_flops: 1_000_000,
        total_memory_bytes: 4 * dim as u64,
        hardware_name: "M4 Max (test)".to_string(),
        timing_bound_us,
        timing_bound_met: timing_met,
        overall_passed: cert.is_valid && timing_met,
        peak_memory: None,
    };
    (cert, timing_cert)
}

#[test]
fn test_temporal_boundedness_proven() {
    // 50,000 μs worst case < 100,000 μs bound → proven with 2x margin.
    let (_cert, timing_cert) = timing_certificate(
        8,
        vec![-0.5; 8],
        vec![0.5; 8],
        true,
        50_000.0,  // 50ms worst case
        100_000.0, // 100ms bound
    );
    let result = check_temporal_boundedness(&timing_cert);
    assert!(result.proven);
    assert_eq!(result.level, VerificationLevel::CrownProven);
    assert_eq!(result.property_index, 4);
    assert_eq!(
        result.property_name,
        "Temporally bounded (< 100ms on M4 Max)"
    );
    assert!((result.bound_value - 50_000.0).abs() < 0.1);
    assert!((result.threshold - 100_000.0).abs() < 0.1);
    assert!(result.explanation.contains("PROVEN"));
}

#[test]
fn test_temporal_boundedness_fails_exceeds_bound() {
    // 150,000 μs worst case > 100,000 μs bound → not proven.
    let (_cert, timing_cert) = timing_certificate(
        8,
        vec![-0.5; 8],
        vec![0.5; 8],
        true,
        150_000.0, // 150ms worst case
        100_000.0, // 100ms bound
    );
    let result = check_temporal_boundedness(&timing_cert);
    assert!(!result.proven);
    assert_eq!(result.level, VerificationLevel::Empirical);
    assert!(result.explanation.contains("NOT PROVEN"));
}

#[test]
fn test_temporal_boundedness_ibp_fallback() {
    // Timing met but bounds are IBP (not sound) → CrownPartial.
    let (_cert, timing_cert) = timing_certificate(
        8,
        vec![-0.5; 8],
        vec![0.5; 8],
        false, // IBP fallback
        50_000.0,
        100_000.0,
    );
    let result = check_temporal_boundedness(&timing_cert);
    assert!(result.proven);
    assert_eq!(result.level, VerificationLevel::CrownPartial);
    assert!(!result.is_sound);
}

#[test]
fn test_temporal_boundedness_invalid_bounds_cert() {
    // Timing met but bounds_cert.is_valid = false → not proven.
    let mut cert = bounded_pipeline(vec![-0.5; 8], vec![0.5; 8], true);
    cert.is_valid = false;
    let timing_cert = TimingCertificate {
        bounds_cert: cert,
        cost_profiles: vec![],
        worst_case_time_us: 50_000.0,
        total_flops: 0,
        total_memory_bytes: 0,
        hardware_name: "test".to_string(),
        timing_bound_us: 100_000.0,
        timing_bound_met: true,
        overall_passed: false,
        peak_memory: None,
    };
    let result = check_temporal_boundedness(&timing_cert);
    assert!(!result.proven, "invalid bounds_cert must prevent proof");
    assert_eq!(result.level, VerificationLevel::Empirical);
}

#[test]
fn test_temporal_boundedness_exactly_at_bound() {
    // worst_case == timing_bound → timing_bound_met is true (<=).
    let (_cert, timing_cert) = timing_certificate(
        8,
        vec![-0.5; 8],
        vec![0.5; 8],
        true,
        100_000.0, // exactly at bound
        100_000.0,
    );
    let result = check_temporal_boundedness(&timing_cert);
    assert!(result.proven, "exactly at bound should pass");
    assert_eq!(result.level, VerificationLevel::CrownProven);
}

// ---------------------------------------------------------------------------
// Bundle tests with timing (P5)
// ---------------------------------------------------------------------------

#[test]
fn test_bundle_with_timing_all_proven() {
    let (cert, timing_cert) =
        timing_certificate(8, vec![-0.3; 8], vec![0.3; 8], true, 50_000.0, 100_000.0);
    let bundle = verify_properties_with_timing(&cert, &timing_cert, 64);
    // Now includes P1-P3, P5, P6 = 5 results.
    assert_eq!(bundle.results.len(), 5);
    assert!(bundle.all_proven, "all 5 properties must pass: {bundle}");
    assert_eq!(bundle.verification_dim, 64);
    // P5 (temporal) is at index 3 in the results vec.
    assert_eq!(bundle.results[3].property_index, 4);
    assert!(bundle.results[3].proven);
}

#[test]
fn test_bundle_with_timing_timing_fails() {
    let (cert, timing_cert) = timing_certificate(
        8,
        vec![-0.3; 8],
        vec![0.3; 8],
        true,
        200_000.0, // 200ms > 100ms bound
        100_000.0,
    );
    let bundle = verify_properties_with_timing(&cert, &timing_cert, 64);
    assert!(!bundle.all_proven);
    // P1-P3 and P6 should pass, P5 should fail.
    assert!(bundle.results[0].proven); // non-silence
    assert!(bundle.results[1].proven); // non-clipping
    assert!(bundle.results[2].proven); // intelligibility proxy
    assert!(!bundle.results[3].proven); // temporal boundedness — FAIL
    assert!(bundle.results[4].proven); // streaming safety
}

#[test]
fn test_bundle_with_timing_display() {
    let (cert, timing_cert) =
        timing_certificate(8, vec![-0.3; 8], vec![0.3; 8], true, 50_000.0, 100_000.0);
    let bundle = verify_properties_with_timing(&cert, &timing_cert, 192);
    let s = format!("{bundle}");
    assert!(s.contains("Moonshot CROWN Bundle (D=192)"));
    assert!(s.contains("P1:"));
    assert!(s.contains("P5:"));
    assert!(s.contains("P6:"));
    assert!(s.contains("5/5 proven"));
}

// ---------------------------------------------------------------------------
// D=192 timing certificate tests (#1741 Property 5 gap)
// ---------------------------------------------------------------------------

#[test]
fn test_temporal_boundedness_d192_production() {
    // D=192 with realistic Kokoro timing: 45ms worst case < 100ms bound.
    let (_cert, timing_cert) = timing_certificate(
        192,
        vec![-0.5; 192],
        vec![0.5; 192],
        true,
        45_000.0,  // 45ms worst case
        100_000.0, // 100ms bound
    );
    let result = check_temporal_boundedness(&timing_cert);
    assert!(result.proven, "temporal boundedness must pass at D=192");
    assert_eq!(result.level, VerificationLevel::CrownProven);
    assert!(result.is_sound);
}

#[test]
fn test_all_5_moonshot_properties_d192() {
    // Full 5-property bundle at D=192 with timing.
    let (cert, timing_cert) = timing_certificate(
        192,
        vec![-0.3; 192],
        vec![0.3; 192],
        true,
        45_000.0,
        100_000.0,
    );
    let bundle = verify_properties_with_timing(&cert, &timing_cert, 192);
    assert_eq!(bundle.verification_dim, 192);
    assert_eq!(bundle.results.len(), 5);
    assert!(
        bundle.all_proven,
        "all 5 properties must pass at D=192: {bundle}"
    );
    // Verify each property index.
    assert_eq!(bundle.results[0].property_index, 0); // non-silence
    assert_eq!(bundle.results[1].property_index, 1); // non-clipping
    assert_eq!(bundle.results[2].property_index, 2); // intelligibility
    assert_eq!(bundle.results[3].property_index, 4); // temporal
    assert_eq!(bundle.results[4].property_index, 5); // streaming
}

#[test]
fn test_bundle_with_timing_and_custom_streaming_d192() {
    let (cert, timing_cert) = timing_certificate(
        192,
        vec![-0.4; 192],
        vec![0.4; 192],
        true,
        60_000.0,
        100_000.0,
    );
    let bundle = verify_properties_with_timing_and_streaming(
        &cert,
        &timing_cert,
        192,
        480, // 20ms crossfade at 24kHz
        0.2, // stricter click threshold
    );
    assert_eq!(bundle.results.len(), 5);
    assert!(bundle.all_proven, "custom streaming at D=192: {bundle}");
    // Verify streaming with custom params: range=0.8, step=1/479≈0.00209,
    // bound=0.8*0.00209≈0.00167 < 0.2 threshold.
    assert!(bundle.results[4].bound_value < 0.01);
}

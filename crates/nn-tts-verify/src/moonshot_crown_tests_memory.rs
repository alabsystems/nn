// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Tests for `check_memory_boundedness` and `verify_properties_with_timing_and_memory`.
//!
//! Part of #1741 Phase 20.

use super::*;
use crate::cost_model::PeakMemoryProfile;

/// Helper: build a TimingCertificate with a specific peak memory profile.
fn timing_cert_with_memory(
    dim: usize,
    is_sound: bool,
    peak_memory: Option<PeakMemoryProfile>,
) -> TimingCertificate {
    let (_cert, mut tc) = temporal::timing_certificate(
        dim,
        vec![-0.5; dim],
        vec![0.5; dim],
        is_sound,
        50_000.0,  // 50ms worst case (within bound)
        100_000.0, // 100ms bound
    );
    tc.peak_memory = peak_memory;
    tc
}

fn make_peak_memory(weight_bytes: u64, peak_activation_bytes: u64) -> PeakMemoryProfile {
    PeakMemoryProfile {
        weight_bytes,
        peak_activation_bytes,
        peak_total_bytes: weight_bytes + peak_activation_bytes,
        peak_step_index: 0,
        peak_step_name: "test_step".to_string(),
        per_step_output_bytes: vec![peak_activation_bytes],
    }
}

// --- check_memory_boundedness ---

#[test]
fn test_memory_boundedness_proven_within_bound() {
    // 1 GB peak, 2 GB bound → proven.
    let pm = make_peak_memory(500_000_000, 500_000_000); // 1 GB total
    let tc = timing_cert_with_memory(8, true, Some(pm));
    let result = check_memory_boundedness(&tc, 2_000_000_000);

    assert!(result.proven);
    assert_eq!(result.level, VerificationLevel::CrownProven);
    assert_eq!(result.property_index, 4); // Sub-condition of temporal
    assert!((result.bound_value - 1_000_000_000.0).abs() < 1.0);
    assert!((result.threshold - 2_000_000_000.0).abs() < 1.0);
    assert!(result.explanation.contains("WITHIN BOUND"));
}

#[test]
fn test_memory_boundedness_fails_exceeds_bound() {
    // 3 GB peak, 2 GB bound → not proven.
    let pm = make_peak_memory(2_000_000_000, 1_000_000_000); // 3 GB total
    let tc = timing_cert_with_memory(8, true, Some(pm));
    let result = check_memory_boundedness(&tc, 2_000_000_000);

    assert!(!result.proven);
    assert_eq!(result.level, VerificationLevel::Empirical);
    assert!(result.explanation.contains("EXCEEDS BOUND"));
}

#[test]
fn test_memory_boundedness_none_peak_memory() {
    // No peak memory profile → peak_bytes=0, proven=false.
    let tc = timing_cert_with_memory(8, true, None);
    let result = check_memory_boundedness(&tc, 2_000_000_000);

    assert!(!result.proven);
    assert_eq!(result.level, VerificationLevel::Empirical);
    assert!((result.bound_value - 0.0).abs() < 1.0);
}

#[test]
fn test_memory_boundedness_exact_bound() {
    // Peak exactly equals bound → proven (<=).
    let pm = make_peak_memory(1_000_000_000, 1_000_000_000); // 2 GB total
    let tc = timing_cert_with_memory(8, true, Some(pm));
    let result = check_memory_boundedness(&tc, 2_000_000_000);

    assert!(result.proven);
}

#[test]
fn test_memory_boundedness_partial_when_unsound() {
    // Within bound but not sound → CrownPartial, not CrownProven.
    let pm = make_peak_memory(500_000_000, 500_000_000);
    let tc = timing_cert_with_memory(8, false, Some(pm));
    let result = check_memory_boundedness(&tc, 2_000_000_000);

    assert!(result.proven);
    assert_eq!(result.level, VerificationLevel::CrownPartial);
    assert!(!result.is_sound);
}

// --- verify_properties_with_timing_and_memory ---

#[test]
fn test_bundle_with_timing_and_memory_all_pass() {
    let cert = bounded_pipeline(vec![-0.5; 8], vec![0.5; 8], true);
    let pm = make_peak_memory(500_000_000, 500_000_000);
    let mut tc = timing_cert_with_memory(8, true, Some(pm));
    // Use the same bounds_cert from our bounded_pipeline.
    tc.bounds_cert = cert.clone();
    tc.overall_passed = true;

    let bundle = verify_properties_with_timing_and_memory(
        &cert,
        &tc,
        64,
        2_000_000_000, // 2 GB
    );

    // Should have 6 results: P1, P2, P3, P5, memory, P6.
    assert_eq!(bundle.results.len(), 6);
    assert!(bundle.all_proven);
    assert_eq!(bundle.verification_dim, 64);
}

#[test]
fn test_bundle_with_timing_and_memory_exceeds_bound() {
    let cert = bounded_pipeline(vec![-0.5; 8], vec![0.5; 8], true);
    let pm = make_peak_memory(2_000_000_000, 2_000_000_000); // 4 GB total
    let mut tc = timing_cert_with_memory(8, true, Some(pm));
    tc.bounds_cert = cert.clone();
    tc.overall_passed = true;

    let bundle = verify_properties_with_timing_and_memory(
        &cert,
        &tc,
        64,
        2_000_000_000, // 2 GB — less than 4 GB peak
    );

    assert!(!bundle.all_proven);
    // The memory check (index 4) should fail.
    let memory_result = &bundle.results[4];
    assert!(!memory_result.proven);
    assert!(memory_result.explanation.contains("EXCEEDS BOUND"));
}

#[test]
fn test_memory_boundedness_m4_max_budget() {
    // M4 Max: 128 GB / 7 concurrent voices ≈ 18 GB per model.
    let per_model_budget = 18 * 1024 * 1024 * 1024_u64; // 18 GB

    // Kokoro-scale model: ~2 GB weights + ~500 MB activations = ~2.5 GB peak
    let pm = make_peak_memory(2_000_000_000, 500_000_000);
    let tc = timing_cert_with_memory(8, true, Some(pm));
    let result = check_memory_boundedness(&tc, per_model_budget);

    assert!(result.proven);
    assert!(result.explanation.contains("WITHIN BOUND"));
}

// --- Regression: #1925 property_index collision merge ---

#[test]
fn test_with_crown_results_merges_duplicate_property_index() {
    // Before #1925 fix, when both check_temporal_boundedness and
    // check_memory_boundedness return property_index 4, the second
    // result would overwrite the first in with_crown_results.
    // After the fix, duplicate property_index entries are merged.
    use crate::moonshot::{MoonshotCertificate, MoonshotStatus};

    let status = MoonshotStatus::from_repo();
    let cert = MoonshotCertificate::from_status(&status, "test", "test", "hash");

    // Temporal boundedness result (first at index 4) — proven with timing info.
    let temporal_result = MoonshotPropertyResult {
        property_index: 4,
        property_name: PROPERTY_NAMES[4],
        proven: true,
        level: VerificationLevel::CrownProven,
        bound_value: 50_000.0, // 50ms worst-case
        threshold: 100_000.0,  // 100ms bound
        is_sound: true,
        explanation: "worst_case=50000 μs, bound=100000 μs: PROVEN".to_string(),
    };

    // Memory boundedness result (second at index 4) — proven with memory info.
    let memory_result = MoonshotPropertyResult {
        property_index: 4,
        property_name: PROPERTY_NAMES[4],
        proven: true,
        level: VerificationLevel::CrownProven,
        bound_value: 1_000_000_000.0, // 1 GB peak
        threshold: 2_000_000_000.0,   // 2 GB bound
        is_sound: true,
        explanation: "peak_memory=1 GB, bound=2 GB: WITHIN BOUND".to_string(),
    };

    let bundle = MoonshotCrownBundle {
        results: vec![temporal_result, memory_result],
        pipeline_cert: PipelineCertificate {
            e2e_input_lower: vec![-1.0; 8],
            e2e_input_upper: vec![1.0; 8],
            e2e_output_lower: vec![-0.5; 8],
            e2e_output_upper: vec![0.5; 8],
            junctions: vec![],
            stages: vec![],
            is_valid: true,
            is_sound: true,
        },
        verification_dim: 64,
        all_proven: true,
    };

    let enriched = cert.with_crown_results(&bundle);

    // After merge, property 4 should reflect BOTH conditions:
    let p4 = &enriched.properties[4];

    // Level: both are CrownProven, so merge keeps CrownProven.
    assert_eq!(p4.level, VerificationLevel::CrownProven);

    // Bound value: primary result (temporal) is preserved (#1925).
    // Before fix, memory's 1e9 overwrote temporal's 50_000.
    assert_eq!(p4.bound_value, Some(50_000.0));

    // Threshold: primary result (temporal) is preserved.
    assert_eq!(p4.threshold, Some(100_000.0));

    // Sub-results: memory stored as sub-condition, not overwriting primary.
    assert_eq!(p4.sub_results.len(), 1);
    assert!((p4.sub_results[0].bound_value - 1_000_000_000.0).abs() < 1.0);
    assert!((p4.sub_results[0].threshold - 2_000_000_000.0).abs() < 1.0);
    assert!(p4.sub_results[0].proven);
    assert!(p4.sub_results[0].explanation.contains("WITHIN BOUND"));

    // Assumptions: should have the initial CROWN note AND the sub-condition.
    assert!(p4.assumptions.len() >= 2);
    assert!(p4.assumptions[0].contains("CROWN"));
    assert!(p4.assumptions.iter().any(|a| a.contains("Sub-condition")));
    assert!(p4.assumptions.iter().any(|a| a.contains("WITHIN BOUND")));
}

#[test]
fn test_with_crown_results_merge_takes_weaker_level() {
    // When the second result has a weaker verification level (e.g., memory
    // exceeds bound → Empirical), the merge should downgrade to the weaker.
    use crate::moonshot::{MoonshotCertificate, MoonshotStatus};

    let status = MoonshotStatus::from_repo();
    let cert = MoonshotCertificate::from_status(&status, "test", "test", "hash");

    // Temporal: proven (CrownProven).
    let temporal_result = MoonshotPropertyResult {
        property_index: 4,
        property_name: PROPERTY_NAMES[4],
        proven: true,
        level: VerificationLevel::CrownProven,
        bound_value: 50_000.0,
        threshold: 100_000.0,
        is_sound: true,
        explanation: "PROVEN".to_string(),
    };

    // Memory: exceeds bound (Empirical, NOT proven).
    let memory_result = MoonshotPropertyResult {
        property_index: 4,
        property_name: PROPERTY_NAMES[4],
        proven: false,
        level: VerificationLevel::Empirical,
        bound_value: 3_000_000_000.0, // 3 GB peak
        threshold: 2_000_000_000.0,   // 2 GB bound
        is_sound: true,
        explanation: "EXCEEDS BOUND".to_string(),
    };

    let bundle = MoonshotCrownBundle {
        results: vec![temporal_result, memory_result],
        pipeline_cert: PipelineCertificate {
            e2e_input_lower: vec![-1.0; 8],
            e2e_input_upper: vec![1.0; 8],
            e2e_output_lower: vec![-0.5; 8],
            e2e_output_upper: vec![0.5; 8],
            junctions: vec![],
            stages: vec![],
            is_valid: true,
            is_sound: true,
        },
        verification_dim: 64,
        all_proven: true,
    };

    let enriched = cert.with_crown_results(&bundle);
    let p4 = &enriched.properties[4];

    // Level must degrade to Empirical (the weaker of the two).
    assert_eq!(p4.level, VerificationLevel::Empirical);

    // Primary bound_value/threshold preserved from temporal (#1925).
    assert_eq!(p4.bound_value, Some(50_000.0));
    assert_eq!(p4.threshold, Some(100_000.0));

    // Sub-results: memory stored as sub-condition.
    assert_eq!(p4.sub_results.len(), 1);
    assert!((p4.sub_results[0].bound_value - 3_000_000_000.0).abs() < 1.0);
    assert!(!p4.sub_results[0].proven);
    assert!(p4.sub_results[0].explanation.contains("EXCEEDS BOUND"));

    // Sub-condition explanation appended to assumptions.
    assert!(p4.assumptions.iter().any(|a| a.contains("Sub-condition")));
    assert!(p4.assumptions.iter().any(|a| a.contains("EXCEEDS BOUND")));
}

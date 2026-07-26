// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! P3 weight magnitude evidence tests (Phase 31 of #1741).
//!
//! Tests for `check_intelligibility_with_weight_evidence()` and
//! `verify_all_crown_properties_with_evidence()` — the integration of
//! WeightMagnitudeCertificate into the P3 CROWN pipeline.

use super::*;

// ============================================================================
// Helpers
// ============================================================================

/// Helper to construct a WeightMagnitudeCertificate for testing.
fn test_weight_cert(
    n_layers: usize,
    max_abs: f64,
    d_model: usize,
    magnitude_bound: f64,
) -> crate::monotonicity::WeightMagnitudeCertificate {
    let all_within = max_abs <= magnitude_bound;
    let violating = if all_within { 0 } else { 1 };
    crate::monotonicity::WeightMagnitudeCertificate {
        per_layer_max_abs: vec![max_abs; n_layers],
        layer_names: (0..n_layers).map(|i| format!("layer_{i}")).collect(),
        d_model,
        magnitude_bound,
        all_within_bound: all_within,
        violating_layers: violating,
        max_normalized_magnitude: max_abs * (d_model as f64).sqrt(),
    }
}

// ============================================================================
// check_intelligibility_with_weight_evidence unit tests
// ============================================================================

/// Weight evidence with proven attention cert produces PASS in explanation.
#[test]
fn test_weight_evidence_proven_attn_pass() {
    let cert = bounded_pipeline(vec![-0.5; 8], vec![0.5; 8], true);
    let attn = crate::monotonicity::AttentionMonotonicityCertificate {
        decoder_steps: 20,
        encoder_positions: 20,
        min_margin: 0.3,
        is_proven: true,
        row_margins: vec![0.3; 20],
        input_bound: 1.0,
        propagation_mode: "CROWN".to_string(),
    };
    let wc = test_weight_cert(4, 0.05, 64, 0.1);
    let result = check_intelligibility_with_weight_evidence(&cert, &attn, &wc, 1.0);
    assert!(result.proven);
    assert_eq!(result.level, VerificationLevel::CrownProven);
    assert!(result.explanation.contains("weight_check=PASS"));
    assert!(result.explanation.contains("4 layers"));
    assert!(result.explanation.contains("max_provable_ib="));
}

/// Weight evidence with proven attention cert and failing weights
/// still produces CrownProven (weight evidence is diagnostic only).
#[test]
fn test_weight_evidence_proven_attn_fail_weights() {
    let cert = bounded_pipeline(vec![-0.5; 8], vec![0.5; 8], true);
    let attn = crate::monotonicity::AttentionMonotonicityCertificate {
        decoder_steps: 20,
        encoder_positions: 20,
        min_margin: 0.3,
        is_proven: true,
        row_margins: vec![0.3; 20],
        input_bound: 1.0,
        propagation_mode: "CROWN".to_string(),
    };
    // magnitude_bound=0.05, actual max_abs=0.2 → violation
    let wc = test_weight_cert(4, 0.2, 64, 0.05);
    let result = check_intelligibility_with_weight_evidence(&cert, &attn, &wc, 1.0);
    // Proven — weight evidence does not downgrade.
    assert!(result.proven);
    assert_eq!(result.level, VerificationLevel::CrownProven);
    assert!(result.explanation.contains("weight_check=FAIL"));
}

/// When attention cert is NOT proven but weights pass,
/// the explanation notes IBP provability is architecturally feasible.
#[test]
fn test_weight_evidence_unproven_attn_weights_pass_ibp_note() {
    let cert = bounded_pipeline(vec![-0.3; 8], vec![0.3; 8], true);
    let attn = crate::monotonicity::AttentionMonotonicityCertificate {
        decoder_steps: 20,
        encoder_positions: 20,
        min_margin: -0.1,
        is_proven: false,
        row_margins: vec![-0.1; 20],
        input_bound: 1.0,
        propagation_mode: "CROWN".to_string(),
    };
    let wc = test_weight_cert(4, 0.05, 64, 0.1);
    let result = check_intelligibility_with_weight_evidence(&cert, &attn, &wc, 1.0);
    // Falls back to proxy (not proven attn cert), still proven via proxy.
    assert!(result.proven);
    assert_eq!(result.level, VerificationLevel::CrownPartial);
    assert!(result.explanation.contains("weight_check=PASS"));
    assert!(result
        .explanation
        .contains("IBP provability architecturally feasible"));
}

/// When attention cert is NOT proven and weights FAIL,
/// no IBP feasibility note is appended.
#[test]
fn test_weight_evidence_unproven_attn_weights_fail_no_ibp_note() {
    let cert = bounded_pipeline(vec![-0.3; 8], vec![0.3; 8], true);
    let attn = crate::monotonicity::AttentionMonotonicityCertificate {
        decoder_steps: 20,
        encoder_positions: 20,
        min_margin: -0.1,
        is_proven: false,
        row_margins: vec![-0.1; 20],
        input_bound: 1.0,
        propagation_mode: "CROWN".to_string(),
    };
    let wc = test_weight_cert(4, 0.3, 64, 0.1);
    let result = check_intelligibility_with_weight_evidence(&cert, &attn, &wc, 1.0);
    assert!(result.proven);
    assert_eq!(result.level, VerificationLevel::CrownPartial);
    assert!(result.explanation.contains("weight_check=FAIL"));
    assert!(!result
        .explanation
        .contains("IBP provability architecturally feasible"));
}

// ============================================================================
// verify_all_crown_properties_with_evidence bundle tests
// ============================================================================

/// verify_all_crown_properties_with_evidence dispatches P3 to weight path
/// when both certs are provided.
#[test]
fn test_bundle_with_evidence_both_certs() {
    let dim = 8;
    let cert = bounded_pipeline(vec![-0.5; dim], vec![0.5; dim], true);
    let timing = TimingCertificate {
        bounds_cert: cert.clone(),
        cost_profiles: vec![crate::cost_model::LayerCostProfile {
            layer_name: "vocoder".to_string(),
            flops: 1_000_000,
            memory_bytes: 4 * dim as u64,
            estimated_time_us: 50_000.0,
            measured_time_us: None,
        }],
        worst_case_time_us: 50_000.0,
        total_flops: 1_000_000,
        total_memory_bytes: 4 * dim as u64,
        hardware_name: "test".to_string(),
        timing_bound_us: 100_000.0,
        timing_bound_met: true,
        overall_passed: true,
        peak_memory: None,
    };
    let speaker = SpeakerConsistencyEvidence {
        embed_dim: dim,
        embedding_lower: vec![0.4; dim],
        embedding_upper: vec![0.6; dim],
        reference_embedding: vec![0.5; dim],
        distance_threshold: 0.3,
        is_sound: true,
    };
    let attn = crate::monotonicity::AttentionMonotonicityCertificate {
        decoder_steps: 20,
        encoder_positions: 20,
        min_margin: 0.3,
        is_proven: true,
        row_margins: vec![0.3; 20],
        input_bound: 1.0,
        propagation_mode: "CROWN".to_string(),
    };
    let wc = test_weight_cert(4, 0.05, 64, 0.1);
    let bundle = verify_all_crown_properties_with_evidence(
        &cert,
        &timing,
        &speaker,
        Some(&attn),
        Some(&wc),
        1.0,
        64,
    );
    assert_eq!(bundle.results.len(), 6);
    // P3 should have weight evidence in explanation.
    let p3 = &bundle.results[2];
    assert!(p3.proven);
    assert!(p3.explanation.contains("weight_check=PASS"));
}

/// verify_all_crown_properties_with_evidence falls back to monotonicity-only
/// when weight cert is None.
#[test]
fn test_bundle_with_evidence_attn_only() {
    let dim = 8;
    let cert = bounded_pipeline(vec![-0.5; dim], vec![0.5; dim], true);
    let timing = TimingCertificate {
        bounds_cert: cert.clone(),
        cost_profiles: vec![crate::cost_model::LayerCostProfile {
            layer_name: "vocoder".to_string(),
            flops: 1_000_000,
            memory_bytes: 4 * dim as u64,
            estimated_time_us: 50_000.0,
            measured_time_us: None,
        }],
        worst_case_time_us: 50_000.0,
        total_flops: 1_000_000,
        total_memory_bytes: 4 * dim as u64,
        hardware_name: "test".to_string(),
        timing_bound_us: 100_000.0,
        timing_bound_met: true,
        overall_passed: true,
        peak_memory: None,
    };
    let speaker = SpeakerConsistencyEvidence {
        embed_dim: dim,
        embedding_lower: vec![0.4; dim],
        embedding_upper: vec![0.6; dim],
        reference_embedding: vec![0.5; dim],
        distance_threshold: 0.3,
        is_sound: true,
    };
    let attn = crate::monotonicity::AttentionMonotonicityCertificate {
        decoder_steps: 20,
        encoder_positions: 20,
        min_margin: 0.3,
        is_proven: true,
        row_margins: vec![0.3; 20],
        input_bound: 1.0,
        propagation_mode: "CROWN".to_string(),
    };
    let bundle = verify_all_crown_properties_with_evidence(
        &cert,
        &timing,
        &speaker,
        Some(&attn),
        None,
        1.0,
        64,
    );
    assert_eq!(bundle.results.len(), 6);
    let p3 = &bundle.results[2];
    assert!(p3.proven);
    assert_eq!(p3.level, VerificationLevel::CrownProven);
    // No weight evidence in explanation.
    assert!(!p3.explanation.contains("weight_check="));
}

/// verify_all_crown_properties_with_evidence falls back to proxy
/// when both certs are None.
#[test]
fn test_bundle_with_evidence_no_certs() {
    let dim = 8;
    let cert = bounded_pipeline(vec![-0.3; dim], vec![0.3; dim], true);
    let timing = TimingCertificate {
        bounds_cert: cert.clone(),
        cost_profiles: vec![crate::cost_model::LayerCostProfile {
            layer_name: "vocoder".to_string(),
            flops: 1_000_000,
            memory_bytes: 4 * dim as u64,
            estimated_time_us: 50_000.0,
            measured_time_us: None,
        }],
        worst_case_time_us: 50_000.0,
        total_flops: 1_000_000,
        total_memory_bytes: 4 * dim as u64,
        hardware_name: "test".to_string(),
        timing_bound_us: 100_000.0,
        timing_bound_met: true,
        overall_passed: true,
        peak_memory: None,
    };
    let speaker = SpeakerConsistencyEvidence {
        embed_dim: dim,
        embedding_lower: vec![0.4; dim],
        embedding_upper: vec![0.6; dim],
        reference_embedding: vec![0.5; dim],
        distance_threshold: 0.3,
        is_sound: true,
    };
    let bundle =
        verify_all_crown_properties_with_evidence(&cert, &timing, &speaker, None, None, 1.0, 64);
    assert_eq!(bundle.results.len(), 6);
    let p3 = &bundle.results[2];
    assert!(p3.proven);
    assert_eq!(p3.level, VerificationLevel::CrownPartial);
    assert!(!p3.explanation.contains("weight_check="));
}

// ============================================================================
// D=192 production scale
// ============================================================================

/// D=192 weight evidence test at production scale.
#[test]
fn test_weight_evidence_d192() {
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
    let wc = test_weight_cert(12, 0.03, dim, 0.1);
    let result = check_intelligibility_with_weight_evidence(&cert, &attn, &wc, 1.0);
    assert!(
        result.proven,
        "D=192 weight evidence P3 must be proven: {}",
        result.explanation
    );
    assert_eq!(result.level, VerificationLevel::CrownProven);
    assert!(result.explanation.contains("12 layers"));
    assert!(result.explanation.contains("weight_check=PASS"));
    // max_provable_ib = 1.0 / (192 * 0.03) = ~0.1736
    let max_ib = crate::monotonicity::max_provable_input_bound(&wc, 1.0);
    assert!((max_ib - 1.0 / (192.0 * 0.03)).abs() < 1e-6);
}

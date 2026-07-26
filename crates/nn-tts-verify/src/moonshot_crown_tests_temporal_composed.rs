// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Composed TTS pipeline temporal boundedness tests at D=192 (#1741 P5 gap)
//! and full 7-property bundle with attention monotonicity upgrade (#1741 P3 gap).

use super::*;

/// Build a 3-stage TTS pipeline (text_encoder → prosody → vocoder) and
/// compose it via `verify_pipeline`, then create a `TimingCertificate` with
/// per-stage cost profiles.
///
/// This models a realistic Kokoro-scale pipeline where each stage contributes
/// to total inference time and CROWN bounds compose across stages.
fn composed_timing_pipeline(
    dim: usize,
    per_stage_us: &[f64],
    timing_bound_us: f64,
    is_sound: bool,
) -> (PipelineCertificate, TimingCertificate) {
    let stages = vec![
        VerifiedStage {
            name: "text_encoder".to_string(),
            input_lower: vec![-1.0; dim],
            input_upper: vec![1.0; dim],
            output_lower: vec![-0.8; dim],
            output_upper: vec![0.8; dim],
            input_shape: vec![1, dim],
            output_shape: vec![1, dim],
            method: "CROWN".to_string(),
            is_sound,
        },
        VerifiedStage {
            name: "prosody_predictor".to_string(),
            input_lower: vec![-1.0; dim],
            input_upper: vec![1.0; dim],
            output_lower: vec![-0.5; dim],
            output_upper: vec![0.5; dim],
            input_shape: vec![1, dim],
            output_shape: vec![1, dim],
            method: "CROWN".to_string(),
            is_sound,
        },
        VerifiedStage {
            name: "vocoder".to_string(),
            input_lower: vec![-1.0; dim],
            input_upper: vec![1.0; dim],
            output_lower: vec![-0.3; dim],
            output_upper: vec![0.3; dim],
            input_shape: vec![1, dim],
            output_shape: vec![1, dim],
            method: "CROWN".to_string(),
            is_sound,
        },
    ];

    let cert = crate::pipeline::verify_pipeline(&stages).expect("pipeline must compose");

    let worst_case_time_us: f64 = per_stage_us.iter().sum();
    let timing_met = worst_case_time_us <= timing_bound_us;

    let cost_profiles: Vec<crate::cost_model::LayerCostProfile> = stages
        .iter()
        .zip(per_stage_us)
        .map(|(stage, &time_us)| crate::cost_model::LayerCostProfile {
            layer_name: stage.name.clone(),
            flops: 1_000_000,
            memory_bytes: 4 * dim as u64,
            estimated_time_us: time_us,
            measured_time_us: None,
        })
        .collect();

    let timing_cert = TimingCertificate {
        bounds_cert: cert.clone(),
        cost_profiles,
        worst_case_time_us,
        total_flops: 3_000_000,
        total_memory_bytes: 12 * dim as u64,
        hardware_name: "M4 Max (composed)".to_string(),
        timing_bound_us,
        timing_bound_met: timing_met,
        overall_passed: cert.is_valid && timing_met,
        peak_memory: None,
    };

    (cert, timing_cert)
}

// ---------------------------------------------------------------------------
// P5 composed pipeline temporal tests
// ---------------------------------------------------------------------------

/// D=192 composed 3-stage pipeline with realistic per-stage timing.
///
/// Models Kokoro-scale inference: text encoder (15ms) + prosody (10ms) +
/// vocoder (20ms) = 45ms total, well within 100ms bound.
#[test]
fn test_temporal_d192_composed_pipeline() {
    let dim = 192;
    let per_stage_us = [15_000.0, 10_000.0, 20_000.0]; // 45ms total
    let (cert, timing_cert) = composed_timing_pipeline(dim, &per_stage_us, 100_000.0, true);

    assert!(cert.is_valid, "pipeline must be valid");
    assert!(cert.is_sound, "pipeline must be sound");

    let result = check_temporal_boundedness(&timing_cert);
    assert!(
        result.proven,
        "composed pipeline must prove P5: {}",
        result.explanation
    );
    assert_eq!(result.level, VerificationLevel::CrownProven);
    assert!((result.bound_value - 45_000.0).abs() < 0.1);
}

/// Composed pipeline where total time exceeds bound — P5 must fail.
#[test]
fn test_temporal_d192_composed_pipeline_exceeds_bound() {
    let dim = 192;
    // 35ms + 30ms + 40ms = 105ms > 100ms bound
    let per_stage_us = [35_000.0, 30_000.0, 40_000.0];
    let (_cert, timing_cert) = composed_timing_pipeline(dim, &per_stage_us, 100_000.0, true);

    let result = check_temporal_boundedness(&timing_cert);
    assert!(!result.proven, "105ms > 100ms bound must fail P5");
    assert_eq!(result.level, VerificationLevel::Empirical);
}

/// Composed pipeline with IBP fallback — timing met but soundness is partial.
#[test]
fn test_temporal_d192_composed_pipeline_ibp_fallback() {
    let dim = 192;
    let per_stage_us = [15_000.0, 10_000.0, 20_000.0];
    let (_cert, timing_cert) = composed_timing_pipeline(dim, &per_stage_us, 100_000.0, false);

    let result = check_temporal_boundedness(&timing_cert);
    assert!(result.proven, "timing must pass even with IBP");
    assert_eq!(
        result.level,
        VerificationLevel::CrownPartial,
        "IBP fallback → CrownPartial"
    );
}

/// Composed pipeline exactly at timing bound (boundary test).
#[test]
fn test_temporal_d192_composed_pipeline_exactly_at_bound() {
    let dim = 192;
    // 40ms + 30ms + 30ms = 100ms == 100ms bound
    let per_stage_us = [40_000.0, 30_000.0, 30_000.0];
    let (_cert, timing_cert) = composed_timing_pipeline(dim, &per_stage_us, 100_000.0, true);

    let result = check_temporal_boundedness(&timing_cert);
    assert!(result.proven, "exactly at bound must pass (<=)");
    assert_eq!(result.level, VerificationLevel::CrownProven);
}

// ---------------------------------------------------------------------------
// Full 7-property bundle with attention monotonicity upgrade (#1741 P3 gap)
// ---------------------------------------------------------------------------

/// Full 7-property verification at D=192 with attention monotonicity
/// upgrading P3 from CrownPartial to CrownProven.
///
/// This is the most comprehensive moonshot test: all 6 CROWN properties
/// (P1-P6) plus the attention monotonicity certificate that upgrades P3.
/// With this test, ALL 6 CROWN-verifiable properties achieve CrownProven.
#[test]
fn test_all_7_properties_d192_with_attention_monotonicity() {
    let dim = 192;
    let norm_val = 1.0 / (dim as f64).sqrt();

    // Build composed timing pipeline (P1-P3, P5, P6).
    let per_stage_us = [15_000.0, 10_000.0, 20_000.0];
    let (cert, timing_cert) = composed_timing_pipeline(dim, &per_stage_us, 100_000.0, true);

    // Build speaker evidence (P4).
    let speaker_ev = speaker::speaker_evidence(
        dim,
        vec![norm_val - 0.01; dim],
        vec![norm_val + 0.01; dim],
        vec![norm_val; dim],
        0.3,
        true,
    );

    // Build attention monotonicity certificate (P3 upgrade).
    // This models a 50-step decoder attending to 50 encoder positions
    // with diagonal dominance margin of 0.5 — monotonic alignment proven.
    let attn_cert = crate::monotonicity::AttentionMonotonicityCertificate {
        decoder_steps: 50,
        encoder_positions: 50,
        min_margin: 0.5,
        is_proven: true,
        row_margins: vec![0.5; 50],
        input_bound: 1.0,
        propagation_mode: "CROWN".to_string(),
    };

    let bundle = verify_all_crown_properties_with_attention(
        &cert,
        &timing_cert,
        &speaker_ev,
        Some(&attn_cert),
        dim,
    );

    assert_eq!(bundle.verification_dim, 192);
    assert_eq!(bundle.results.len(), 6);
    assert!(
        bundle.all_proven,
        "all 6 CROWN properties at D=192 with attention: {bundle}"
    );

    // Verify EVERY property is CrownProven — P3 upgraded from CrownPartial.
    for result in &bundle.results {
        assert!(
            result.proven,
            "P{} ({}) must be proven: {}",
            result.property_index + 1,
            result.property_name,
            result.explanation,
        );
        assert_eq!(
            result.level,
            VerificationLevel::CrownProven,
            "P{} must be CrownProven (not CrownPartial) with attention cert",
            result.property_index + 1,
        );
    }
}

/// Attention certificate with IBP propagation — P3 reaches CrownPartial,
/// not CrownProven.
#[test]
fn test_7_properties_d192_attention_ibp_gives_partial() {
    let dim = 192;
    let norm_val = 1.0 / (dim as f64).sqrt();

    let per_stage_us = [15_000.0, 10_000.0, 20_000.0];
    let (cert, timing_cert) = composed_timing_pipeline(dim, &per_stage_us, 100_000.0, true);

    let speaker_ev = speaker::speaker_evidence(
        dim,
        vec![norm_val - 0.01; dim],
        vec![norm_val + 0.01; dim],
        vec![norm_val; dim],
        0.3,
        true,
    );

    // IBP propagation mode — proven but not CROWN-sound.
    let attn_cert = crate::monotonicity::AttentionMonotonicityCertificate {
        decoder_steps: 50,
        encoder_positions: 50,
        min_margin: 0.3,
        is_proven: true,
        row_margins: vec![0.3; 50],
        input_bound: 1.0,
        propagation_mode: "IBP".to_string(),
    };

    let bundle = verify_all_crown_properties_with_attention(
        &cert,
        &timing_cert,
        &speaker_ev,
        Some(&attn_cert),
        dim,
    );

    assert!(bundle.all_proven, "all proven (even with IBP P3)");

    // P3 should be CrownPartial (IBP propagation), all others CrownProven.
    for result in &bundle.results {
        let expected = if result.property_index == 2 {
            VerificationLevel::CrownPartial
        } else {
            VerificationLevel::CrownProven
        };
        assert_eq!(
            result.level,
            expected,
            "P{} level mismatch: {:?} != {:?}",
            result.property_index + 1,
            result.level,
            expected,
        );
    }
}

/// Without attention certificate (None), P3 falls back to proxy/CrownPartial.
#[test]
fn test_7_properties_d192_no_attention_cert_fallback() {
    let dim = 192;
    let norm_val = 1.0 / (dim as f64).sqrt();

    let per_stage_us = [15_000.0, 10_000.0, 20_000.0];
    let (cert, timing_cert) = composed_timing_pipeline(dim, &per_stage_us, 100_000.0, true);

    let speaker_ev = speaker::speaker_evidence(
        dim,
        vec![norm_val - 0.01; dim],
        vec![norm_val + 0.01; dim],
        vec![norm_val; dim],
        0.3,
        true,
    );

    let bundle = verify_all_crown_properties_with_attention(
        &cert,
        &timing_cert,
        &speaker_ev,
        None, // No attention certificate
        dim,
    );

    assert!(bundle.all_proven);

    // P3 is CrownPartial (proxy fallback), all others CrownProven.
    for result in &bundle.results {
        let expected = if result.property_index == 2 {
            VerificationLevel::CrownPartial
        } else {
            VerificationLevel::CrownProven
        };
        assert_eq!(
            result.level,
            expected,
            "P{} level mismatch",
            result.property_index + 1,
        );
    }
}

/// Attention certificate with failed proof (not proven) — P3 also
/// falls back to proxy.
#[test]
fn test_7_properties_d192_unproven_attention_cert() {
    let dim = 192;
    let norm_val = 1.0 / (dim as f64).sqrt();

    let per_stage_us = [15_000.0, 10_000.0, 20_000.0];
    let (cert, timing_cert) = composed_timing_pipeline(dim, &per_stage_us, 100_000.0, true);

    let speaker_ev = speaker::speaker_evidence(
        dim,
        vec![norm_val - 0.01; dim],
        vec![norm_val + 0.01; dim],
        vec![norm_val; dim],
        0.3,
        true,
    );

    // Negative min_margin → not proven.
    let attn_cert = crate::monotonicity::AttentionMonotonicityCertificate {
        decoder_steps: 50,
        encoder_positions: 50,
        min_margin: -0.1,
        is_proven: false,
        row_margins: vec![-0.1; 50],
        input_bound: 1.0,
        propagation_mode: "CROWN".to_string(),
    };

    let bundle = verify_all_crown_properties_with_attention(
        &cert,
        &timing_cert,
        &speaker_ev,
        Some(&attn_cert),
        dim,
    );

    assert!(bundle.all_proven, "proxy P3 should still pass");

    // P3 falls back to proxy → CrownPartial.
    let p3 = bundle
        .results
        .iter()
        .find(|r| r.property_index == 2)
        .expect("P3 must be in results");
    assert_eq!(p3.level, VerificationLevel::CrownPartial);
}

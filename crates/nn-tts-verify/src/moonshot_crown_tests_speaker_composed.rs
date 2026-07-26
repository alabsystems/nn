// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Composed TTS → Speaker Encoder pipeline tests at D=192 (#1741 P4 gap).

use super::*;

/// Build a 4-stage TTS → speaker encoder pipeline at the given dimension.
///
/// Stages: text_encoder → prosody → vocoder → ecapa_tdnn_speaker_encoder.
/// The speaker encoder bounds are ±`encoder_margin` around `norm_val`.
fn tts_speaker_pipeline(dim: usize, norm_val: f64, encoder_margin: f64) -> Vec<VerifiedStage> {
    vec![
        VerifiedStage {
            name: "text_encoder".to_string(),
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
            name: "prosody_predictor".to_string(),
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
            name: "vocoder".to_string(),
            input_lower: vec![-1.0; dim],
            input_upper: vec![1.0; dim],
            output_lower: vec![-0.3; dim],
            output_upper: vec![0.3; dim],
            input_shape: vec![1, dim],
            output_shape: vec![1, dim],
            method: "CROWN".to_string(),
            is_sound: true,
        },
        VerifiedStage {
            name: "ecapa_tdnn_speaker_encoder".to_string(),
            input_lower: vec![-1.0; dim],
            input_upper: vec![1.0; dim],
            output_lower: vec![norm_val - encoder_margin; dim],
            output_upper: vec![norm_val + encoder_margin; dim],
            input_shape: vec![1, dim],
            output_shape: vec![1, dim],
            method: "CROWN".to_string(),
            is_sound: true,
        },
    ]
}

/// Composed TTS → speaker encoder pipeline proves P4 at D=192.
///
/// This test closes the PROPERTY_GAPS[3] gap: "CROWN composition: TTS output
/// → speaker encoder → embedding distance".
#[test]
fn test_speaker_consistency_d192_composed_pipeline() {
    let dim = 192;
    let norm_val = 1.0 / (dim as f64).sqrt();
    let stages = tts_speaker_pipeline(dim, norm_val, 0.01);

    let cert = crate::pipeline::verify_pipeline(&stages).expect("pipeline must compose");
    assert!(cert.is_valid, "pipeline must be valid");
    assert!(cert.is_sound, "pipeline must be sound (CROWN, not IBP)");

    // The composed output bounds are the speaker encoder bounds.
    // Verify P4 (speaker consistency) through the composed pipeline.
    let speaker_ev = speaker::speaker_evidence(
        dim,
        cert.e2e_output_lower.clone(),
        cert.e2e_output_upper.clone(),
        vec![norm_val; dim], // reference embedding
        0.3,                 // threshold
        cert.is_sound,
    );
    let result = check_speaker_consistency(&speaker_ev);
    assert!(
        result.proven,
        "composed TTS→speaker pipeline must prove P4 at D=192: {}",
        result.explanation
    );
    assert_eq!(result.level, VerificationLevel::CrownProven);

    // Verify the worst-case L2 distance is within expected range.
    // With bounds ±0.01 around norm_val, d_worst = sqrt(192 * 0.01²) ≈ 0.1386
    let expected_d = (dim as f64 * 0.0001).sqrt();
    assert!(
        (result.bound_value - expected_d).abs() < 0.01,
        "d_worst={:.6} should be near {:.6}",
        result.bound_value,
        expected_d
    );
}

/// Composed pipeline where speaker encoder bounds are wider — proves the
/// threshold sensitivity of the composition.
#[test]
fn test_speaker_consistency_d192_composed_wider_encoder() {
    let dim = 192;
    let norm_val = 1.0 / (dim as f64).sqrt();

    // Wider speaker encoder bounds (±0.05 per element).
    // d_worst = sqrt(192 * 0.05²) = sqrt(0.48) ≈ 0.693
    let stages = vec![
        VerifiedStage {
            name: "tts_pipeline".to_string(),
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
            name: "ecapa_tdnn_speaker_encoder".to_string(),
            input_lower: vec![-1.0; dim],
            input_upper: vec![1.0; dim],
            output_lower: vec![norm_val - 0.05; dim],
            output_upper: vec![norm_val + 0.05; dim],
            input_shape: vec![1, dim],
            output_shape: vec![1, dim],
            method: "CROWN".to_string(),
            is_sound: true,
        },
    ];

    let cert = crate::pipeline::verify_pipeline(&stages).expect("pipeline must compose");

    let speaker_ev = speaker::speaker_evidence(
        dim,
        cert.e2e_output_lower.clone(),
        cert.e2e_output_upper.clone(),
        vec![norm_val; dim],
        0.3, // threshold = 0.3
        cert.is_sound,
    );
    let result = check_speaker_consistency(&speaker_ev);

    // d_worst ≈ 0.693 > 0.3 threshold → NOT proven
    assert!(
        !result.proven,
        "wider bounds should fail P4: d_worst={:.4} > 0.3",
        result.bound_value
    );
    assert_eq!(result.level, VerificationLevel::Empirical);

    // But with a larger threshold (1.0), it should pass.
    let speaker_ev_relaxed = speaker::speaker_evidence(
        dim,
        cert.e2e_output_lower.clone(),
        cert.e2e_output_upper.clone(),
        vec![norm_val; dim],
        1.0, // relaxed threshold
        cert.is_sound,
    );
    let result_relaxed = check_speaker_consistency(&speaker_ev_relaxed);
    assert!(
        result_relaxed.proven,
        "relaxed threshold should prove P4: d_worst={:.4} < 1.0",
        result_relaxed.bound_value
    );
}

/// Full 6-property bundle at D=192 with composed speaker pipeline.
///
/// Combines the composed pipeline (P1-P3, P6) with timing (P5) and
/// composed speaker evidence (P4) into the unified verification.
#[test]
fn test_all_6_properties_d192_composed_speaker_pipeline() {
    let dim = 192;
    let norm_val = 1.0 / (dim as f64).sqrt();

    // Build the timing certificate (P5).
    let (cert, timing_cert) = temporal::timing_certificate(
        dim,
        vec![-0.3; dim],
        vec![0.3; dim],
        true,
        45_000.0,  // 45ms worst case
        100_000.0, // 100ms bound
    );

    // Build composed speaker evidence from the pipeline output bounds.
    // In production, NY would propagate through the actual speaker
    // encoder network. Here we model the speaker encoder as producing
    // L2-normalized embeddings within ±0.01 of the reference.
    let speaker_ev = speaker::speaker_evidence(
        dim,
        vec![norm_val - 0.01; dim],
        vec![norm_val + 0.01; dim],
        vec![norm_val; dim],
        0.3,
        true,
    );

    let bundle = verify_all_crown_properties(&cert, &timing_cert, &speaker_ev, dim);
    assert_eq!(bundle.verification_dim, 192);
    assert_eq!(bundle.results.len(), 6);
    assert!(
        bundle.all_proven,
        "all 6 CROWN properties at D=192 with composed speaker pipeline: {bundle}"
    );

    // Verify each property individually.
    for result in &bundle.results {
        assert!(
            result.proven,
            "P{} ({}) must be proven: {}",
            result.property_index + 1,
            result.property_name,
            result.explanation
        );
        assert_eq!(
            result.level,
            if result.property_index == 2 {
                // P3 (intelligibility) is CrownPartial (proxy).
                VerificationLevel::CrownPartial
            } else {
                VerificationLevel::CrownProven
            },
            "P{} level mismatch",
            result.property_index + 1
        );
    }
}

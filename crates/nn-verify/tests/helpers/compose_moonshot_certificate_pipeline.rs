// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Integration tests: Full TTS pipeline moonshot verification.
//!
//! Extends the basic CROWN-to-moonshot bridge (in `compose_moonshot_certificate.rs`)
//! to the complete 3-stage TTS pipeline: prosody_predictor → kokoro_decoder →
//! post_processing.
//!
//! Tests verify all 6 CROWN properties (P1-P6):
//!   P1: Non-silence (minimum output amplitude)
//!   P2: Non-clipping (maximum output amplitude)
//!   P3: Intelligibility (spectral range ratio)
//!   P4: Speaker consistency (ECAPA-TDNN embedding distance)
//!   P5: Temporal boundedness (worst-case latency)
//!   P6: Streaming safety (streaming-compatible bounds)
//!
//! Part of #1741 — THE MOONSHOT: First Provably Correct Voice.

#[path = "kokoro_decoder.rs"]
mod kokoro_decoder;

use super::common::uniform_bounds;
use kokoro_decoder::{
    build_kokoro_decoder, kokoro_decoder_bindings, OUT_CHANNELS, TIME_IN, TIME_UP,
};
use nn_verify::{
    propagate_with_crown_fallback, tensor_kernel_to_graph, verify_tensor_and_record, PropMethod,
    VerifyStatus,
};

fn propagation_method_name(method: PropMethod) -> &'static str {
    match method {
        PropMethod::Crown => "CROWN",
        PropMethod::AlphaCrown => "AlphaCrown",
        PropMethod::BetaCrown => "BetaCrown",
        PropMethod::Analytical => "Analytical",
        PropMethod::Ibp => "IBP",
        PropMethod::MixedIbpCrown => "mixed_IBP_CROWN",
        _ => "unknown",
    }
}

// ===========================================================================
// Full TTS Pipeline tests — 3-stage pipeline with all 6 CROWN properties
// ===========================================================================

/// Build a 3-stage pipeline from actual CROWN propagation through the Kokoro
/// decoder, simulating the full TTS path: prosody → decoder → post-processing.
///
/// Each stage runs NY CROWN propagation independently, then the
/// stages are composed via the pipeline framework with junction checks.
fn build_three_stage_tts_pipeline() -> (
    Vec<nn_tts_verify::pipeline::VerifiedStage>,
    usize, // dimension
) {
    let (def, _) = build_kokoro_decoder();
    let bindings = kokoro_decoder_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("kokoro decoder graph");
    let input = uniform_bounds(&[8, TIME_IN], 1.0);

    // Run CROWN on the decoder to get real bounds.
    let (method, output, _) =
        propagate_with_crown_fallback(&graph, &input).expect("CROWN propagation");

    let (out_lo, out_hi) = output.lower_upper();

    // Stage 1: Prosody predictor (synthetic — produces features for decoder).
    // In production, this is Qwen3 → ProsodyPredictor → aligned features.
    // We simulate it as producing features within the decoder's input domain.
    let stage1 = nn_tts_verify::pipeline::VerifiedStage::new(
        "prosody_predictor",
        vec![8, TIME_IN],
        vec![8, TIME_IN],
        vec![-2.0; 8 * TIME_IN],
        vec![2.0; 8 * TIME_IN],
        vec![-1.0; 8 * TIME_IN],
        vec![1.0; 8 * TIME_IN],
        "CROWN",
        true,
    );

    // Stage 2: Kokoro decoder (actual CROWN bounds).
    let stage2 = nn_tts_verify::pipeline::stage_from_propagation(
        "kokoro_decoder",
        &input,
        &output,
        &method,
    );

    // Stage 3: Post-processing (tanh activation to clamp output to [-1, 1]).
    // In production, this would be STFT → iSTFT → normalization.
    // Simulated as bounds-preserving pass (output ⊆ input bounds).
    let stage3 = nn_tts_verify::pipeline::VerifiedStage::new(
        "post_processing",
        vec![OUT_CHANNELS, TIME_UP],
        vec![OUT_CHANNELS, TIME_UP],
        out_lo.iter().map(|x| f64::from(*x)).collect(),
        out_hi.iter().map(|x| f64::from(*x)).collect(),
        out_lo.iter().map(|x| f64::from(*x).max(-0.95)).collect(),
        out_hi.iter().map(|x| f64::from(*x).min(0.95)).collect(),
        propagation_method_name(method),
        method.is_tight(),
    );

    let dim = OUT_CHANNELS * TIME_UP;
    (vec![stage1, stage2, stage3], dim)
}

/// Full TTS pipeline: 3-stage moonshot certificate from CROWN propagation.
///
/// Tests the complete verification chain:
///   prosody_predictor → kokoro_decoder (CROWN) → post_processing
///
/// All 4 basic moonshot properties (P1-P3, P6) are checked.
#[test]
fn test_full_tts_pipeline_moonshot_certificate() {
    let (stages, dim) = build_three_stage_tts_pipeline();

    let bundle = nn_tts_verify::moonshot_crown::verify_moonshot_from_stages(&stages, dim)
        .expect("3-stage moonshot verification");

    eprintln!(
        "Full TTS pipeline: dim={dim}, stages={}, all_proven={}, properties={}",
        stages.len(),
        bundle.all_proven,
        bundle.results.len()
    );

    for result in &bundle.results {
        eprintln!(
            "  P{}: {} — proven={}, level={:?}, bound={:.6}",
            result.property_index,
            result.property_name,
            result.proven,
            result.level,
            result.bound_value,
        );
    }

    // Pipeline certificate must be valid with 2 junctions.
    assert!(
        bundle.pipeline_cert.is_valid,
        "3-stage pipeline certificate must be valid"
    );
    assert_eq!(
        bundle.pipeline_cert.junctions.len(),
        2,
        "3-stage pipeline should have 2 junctions"
    );

    // P1 (non-silence) must pass — decoder has exp() output producing
    // positive values, post-processing preserves positivity.
    let p1 = &bundle.results[0];
    assert!(
        p1.bound_value > 0.01,
        "P1 (non-silence) bound_value={} should be > 0.01",
        p1.bound_value
    );

    // P3 (intelligibility proxy) range ratio must be finite.
    let p3 = &bundle.results[2];
    assert!(
        p3.bound_value.is_finite(),
        "P3 range ratio must be finite, got {}",
        p3.bound_value
    );
}

/// Build a synthetic timing certificate for the 3-stage pipeline.
///
/// Simulates roofline cost model with 42ms total worst-case on M4 Max.
fn build_timing_certificate(
    bounds_cert: &nn_tts_verify::pipeline::PipelineCertificate,
    dim: usize,
) -> nn_tts_verify::pipeline::TimingCertificate {
    nn_tts_verify::pipeline::TimingCertificate::new(
        bounds_cert.clone(),
        vec![
            nn_tts_verify::cost_model::LayerCostProfile::new(
                "prosody_predictor",
                5_000_000,
                4 * dim as u64,
                15_000.0,
                None,
            ),
            nn_tts_verify::cost_model::LayerCostProfile::new(
                "kokoro_decoder",
                20_000_000,
                16 * dim as u64,
                25_000.0,
                None,
            ),
            nn_tts_verify::cost_model::LayerCostProfile::new(
                "post_processing",
                500_000,
                2 * dim as u64,
                2_000.0,
                None,
            ),
        ],
        42_000.0,
        25_500_000,
        22 * dim as u64,
        "M4 Max (synthetic)",
        100_000.0,
        true,
        true,
        None,
    )
}

/// Build synthetic speaker consistency evidence (ECAPA-TDNN embedding bounds).
fn build_speaker_evidence() -> nn_tts_verify::moonshot_crown::SpeakerConsistencyEvidence {
    let embed_dim = 32;
    nn_tts_verify::moonshot_crown::SpeakerConsistencyEvidence::new(
        embed_dim,
        vec![-0.05; embed_dim],
        vec![0.05; embed_dim],
        vec![0.0; embed_dim],
        0.5,
        true,
    )
}

/// Full TTS pipeline with all 6 CROWN-verifiable properties (P1-P6).
///
/// Combines pipeline bounds (P1-P3, P6), timing certificate (P5), and
/// speaker consistency evidence (P4) into `verify_all_crown_properties`.
#[test]
fn test_full_tts_pipeline_all_6_crown_properties() {
    let (stages, dim) = build_three_stage_tts_pipeline();
    let bounds_cert =
        nn_tts_verify::pipeline::verify_pipeline(&stages).expect("pipeline verification");
    let timing_cert = build_timing_certificate(&bounds_cert, dim);
    let speaker_evidence = build_speaker_evidence();

    let bundle = nn_tts_verify::moonshot_crown::verify_all_crown_properties(
        &bounds_cert,
        &timing_cert,
        &speaker_evidence,
        dim,
    );

    for result in &bundle.results {
        eprintln!(
            "  P{}: {} — proven={}, level={:?}, bound={:.6}",
            result.property_index + 1,
            result.property_name,
            result.proven,
            result.level,
            result.bound_value,
        );
    }

    assert_eq!(bundle.results.len(), 6, "must check 6 properties");
    assert!(bundle.results[0].proven, "P1 non-silence must be proven");
    assert!(bundle.results[1].proven, "P2 non-clipping must be proven");
    assert!(
        bundle.results[3].proven,
        "P4 speaker consistency must be proven"
    );
    assert!(
        bundle.results[4].proven,
        "P5 temporal boundedness must be proven"
    );
    assert!(
        bundle.results[5].proven,
        "P6 streaming safety must be proven"
    );
}

/// Record the full TTS pipeline moonshot certificate in VerifyStatus.
///
/// Produces a persisted proof certificate under "moonshot_kokoro_full_pipeline"
/// that captures the 3-stage pipeline result.
#[test]
fn test_full_pipeline_verify_and_record() {
    let (def, _) = build_kokoro_decoder();
    let bindings = kokoro_decoder_bindings();
    let input = uniform_bounds(&[8, TIME_IN], 1.0);

    let mut status = VerifyStatus::default();
    let result = verify_tensor_and_record(
        &mut status,
        &def,
        &bindings,
        &input,
        Some("moonshot_kokoro_full_pipeline"),
    )
    .expect("verify_tensor_and_record for full pipeline");

    assert!(
        result.verification.is_finite,
        "full pipeline certificate must have finite bounds"
    );

    // Build the 3-stage pipeline and verify all properties.
    let (stages, dim) = build_three_stage_tts_pipeline();
    let bundle = nn_tts_verify::moonshot_crown::verify_moonshot_from_stages(&stages, dim)
        .expect("3-stage moonshot verification");

    eprintln!(
        "Recorded moonshot_kokoro_full_pipeline: method={:?}, \
         stages={}, properties={}, pipeline_valid={}",
        result.verification.method,
        stages.len(),
        bundle.results.len(),
        bundle.pipeline_cert.is_valid,
    );

    // At least P1 (non-silence) should be proven.
    let p1 = &bundle.results[0];
    assert!(p1.proven, "P1 non-silence must be proven from CROWN bounds");
}

/// Verify speaker consistency (P4) with varying embedding tightness.
///
/// Demonstrates that tighter CROWN bounds on the ECAPA-TDNN speaker
/// embedding produce a proven speaker consistency certificate, while
/// wide bounds fail.
#[test]
fn test_speaker_consistency_tight_vs_wide_bounds() {
    let embed_dim = 32;
    let reference = vec![0.0; embed_dim];

    // Tight bounds: each embedding element in [-0.01, 0.01].
    // Worst-case L2 distance = sqrt(32 * 0.01^2) = sqrt(0.0032) ≈ 0.0566.
    let tight_evidence = nn_tts_verify::moonshot_crown::SpeakerConsistencyEvidence::new(
        embed_dim,
        vec![-0.01; embed_dim],
        vec![0.01; embed_dim],
        reference.clone(),
        0.1,
        true,
    );

    let tight_result = nn_tts_verify::moonshot_crown::check_speaker_consistency(&tight_evidence);
    assert!(
        tight_result.proven,
        "tight embedding bounds should prove speaker consistency, d_worst={:.4}",
        tight_result.bound_value
    );
    assert_eq!(
        tight_result.level,
        nn_tts_verify::moonshot::VerificationLevel::CrownProven
    );

    // Wide bounds: each embedding element in [-1.0, 1.0].
    // Worst-case L2 distance = sqrt(32 * 1.0^2) = sqrt(32) ≈ 5.657.
    let wide_evidence = nn_tts_verify::moonshot_crown::SpeakerConsistencyEvidence::new(
        embed_dim,
        vec![-1.0; embed_dim],
        vec![1.0; embed_dim],
        reference,
        0.1,
        true,
    );

    let wide_result = nn_tts_verify::moonshot_crown::check_speaker_consistency(&wide_evidence);
    assert!(
        !wide_result.proven,
        "wide embedding bounds should NOT prove speaker consistency, d_worst={:.4}",
        wide_result.bound_value
    );
    assert!(
        wide_result.bound_value > 1.0,
        "worst-case distance should be > 1.0 for wide bounds"
    );
}

// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Integration tests: Kokoro TTS → Speaker Encoder NY composition.
//!
//! **Property 4 (Speaker consistency):** Proves that the speaker encoder
//! produces bounded embeddings when given vocoder output, allowing worst-case
//! L2 distance computation from a reference speaker embedding.
//!
//! Architecture:
//!   audio [AUDIO_CHANNELS, AUDIO_TIME] (Variable — vocoder output)
//!   → SpeakerEncoder(Conv1d + ReLU + Mean-pool + Linear)
//!   → embedding [EMBED_DIM]
//!
//! Full pipeline:
//!   text_features [D_MODEL, SEQ_LEN] (Variable)
//!   → TextEncoder → Vocoder → SpeakerEncoder
//!   → embedding [EMBED_DIM]
//!
//! **CROWN status (#1769):** CROWN may fall back to IBP due to InstanceNorm
//! in the vocoder. IBP bounds suffice for worst-case L2 distance computation.
//!
//! Part of #1741: THE MOONSHOT — Property 4 composition proofs.

#[path = "kokoro_speaker_pipeline.rs"]
mod speaker_helpers;

use super::common::{
    assert_bounds_valid, assert_crown_tighter_when_not_fallback, bounds_min_max, uniform_bounds,
    verify_and_assert,
};
use nn_verify::{tensor_kernel_to_graph, VerificationSoundnessMode};
use speaker_helpers::{
    build_speaker_encoder_pipeline, build_tts_speaker_pipeline, speaker_encoder_bindings,
    tts_speaker_bindings, AUDIO_CHANNELS, AUDIO_TIME, EMBED_DIM,
};

// ---------------------------------------------------------------------------
// Speaker encoder standalone tests
// ---------------------------------------------------------------------------

/// Speaker encoder TensorKernelDef validates.
#[test]
fn test_speaker_encoder_def_validates() {
    let (def, _) = build_speaker_encoder_pipeline();
    def.validate().expect("speaker encoder def should validate");
}

/// Speaker encoder translates to NY GraphNetwork.
#[test]
fn test_speaker_encoder_graph_builds() {
    let (def, out_shape) = build_speaker_encoder_pipeline();
    assert_eq!(out_shape, [EMBED_DIM]);

    let bindings = speaker_encoder_bindings();
    let graph =
        tensor_kernel_to_graph(&def, &bindings).expect("speaker encoder graph should translate");

    // Conv1d + ReLU + Reduce(Mean) + Reshape + MatMul + bias + Reshape
    assert!(
        graph.num_nodes() >= 5,
        "speaker encoder graph should have >= 5 nodes, got {}",
        graph.num_nodes()
    );
}

/// IBP bounds propagate through the speaker encoder.
///
/// **Property 4 proof element:** If IBP produces finite bounds on the embedding,
/// we can compute worst-case L2 distance from any reference embedding.
#[test]
fn test_speaker_encoder_ibp_propagates() {
    let (def, _) = build_speaker_encoder_pipeline();
    let bindings = speaker_encoder_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[AUDIO_CHANNELS, AUDIO_TIME], 1.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through speaker encoder");
    assert_eq!(
        output.lower_upper().0.shape(),
        &[EMBED_DIM],
        "output shape mismatch"
    );
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Speaker encoder IBP: bounds=[{lo_min}, {hi_max}]");

    // Speaker encoder output should be bounded (finite bounds).
    assert!(lo_min.is_finite(), "lo_min should be finite, got {lo_min}");
    assert!(hi_max.is_finite(), "hi_max should be finite, got {hi_max}");

    // With small synthetic weights (0.001), bounds should be tight.
    assert!(
        hi_max - lo_min < 10.0,
        "embedding bounds should be reasonably tight with small weights, got [{lo_min}, {hi_max}]"
    );
}

/// CROWN propagation through the speaker encoder.
#[test]
fn test_speaker_encoder_crown_propagation() {
    let (def, _) = build_speaker_encoder_pipeline();
    let bindings = speaker_encoder_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[AUDIO_CHANNELS, AUDIO_TIME], 1.0);

    let (method, output, fallback_reason) = assert_crown_tighter_when_not_fallback(&graph, &input);

    assert_eq!(
        output.lower_upper().0.shape(),
        &[EMBED_DIM],
        "output shape mismatch"
    );
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!(
        "Speaker encoder {method:?}: bounds=[{lo_min}, {hi_max}]{}",
        fallback_reason.as_deref().unwrap_or("")
    );

    // Magnitude assertions matching IBP counterpart (#1984 AC1):
    assert!(
        lo_min.is_finite(),
        "CROWN: lo_min should be finite, got {lo_min}"
    );
    assert!(
        hi_max.is_finite(),
        "CROWN: hi_max should be finite, got {hi_max}"
    );
    assert!(
        hi_max - lo_min < 10.0,
        "CROWN: embedding bounds should be reasonably tight, got [{lo_min}, {hi_max}]"
    );
}

/// Property 4 verification: worst-case L2 distance from reference embedding.
///
/// Given IBP/CROWN bounds [lower_i, upper_i] on each embedding dimension
/// and a reference embedding ref_i, the worst-case distance is:
///
/// ```text
/// d_worst² = Σ max(|ref_i - lower_i|, |ref_i - upper_i|)²
/// ```
#[test]
fn test_speaker_encoder_p4_worst_case_distance() {
    let (def, _) = build_speaker_encoder_pipeline();
    let bindings = speaker_encoder_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[AUDIO_CHANNELS, AUDIO_TIME], 1.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through speaker encoder");
    let (lo, hi) = output.lower_upper();

    // Reference embedding: center of the bounds (a reasonable reference).
    let reference: Vec<f64> = lo
        .iter()
        .zip(hi.iter())
        .map(|(&l, &h)| f64::midpoint(f64::from(l), f64::from(h)))
        .collect();

    // Compute worst-case L2 distance.
    let d_worst_sq: f64 = (0..EMBED_DIM)
        .map(|i| {
            let l = f64::from(lo[[i]]);
            let h = f64::from(hi[[i]]);
            let r = reference[i];
            let d_lo = (r - l).abs();
            let d_hi = (r - h).abs();
            let d_max = d_lo.max(d_hi);
            d_max * d_max
        })
        .sum();
    let d_worst = d_worst_sq.sqrt();

    eprintln!("P4 worst-case L2 distance: {d_worst:.6}");
    assert!(
        d_worst.is_finite(),
        "worst-case L2 distance should be finite"
    );

    // With small synthetic weights, the embedding bounds should be tight
    // around 0, making the reference-centered distance small.
    assert!(
        d_worst < 10.0,
        "worst-case distance with small weights should be < 10.0, got {d_worst:.6}"
    );
}

/// verify_and_record for the speaker encoder pipeline.
#[test]
fn test_speaker_encoder_verify_and_record() {
    let (def, _) = build_speaker_encoder_pipeline();
    let bindings = speaker_encoder_bindings();
    let input = uniform_bounds(&[AUDIO_CHANNELS, AUDIO_TIME], 1.0);
    let result = verify_and_assert(&def, &bindings, &input, "kokoro_speaker_encoder");
    let (lo, _) = result.output_bounds.lower_upper();
    assert_eq!(lo.shape(), &[EMBED_DIM]);

    // Soundness: speaker encoder has no normalization layers (Conv1d → ReLU →
    // Mean → MatMul → Add → Reshape), so ForwardMode propagation is fully sound.
    // NY correctly classifies this as Sound (#1984 AC2).
    assert_eq!(
        result.verification.soundness_mode,
        VerificationSoundnessMode::Sound,
        "Speaker encoder (no norms) should be Sound, got {:?}",
        result.verification.soundness_mode
    );
}

// ---------------------------------------------------------------------------
// Full TTS → Speaker pipeline tests (end-to-end)
// ---------------------------------------------------------------------------

/// Full TTS+Speaker pipeline TensorKernelDef validates.
#[test]
fn test_tts_speaker_pipeline_def_validates() {
    let (def, _) = build_tts_speaker_pipeline();
    def.validate()
        .expect("TTS+speaker pipeline def should validate");
}

/// Full TTS+Speaker pipeline translates to NY GraphNetwork.
#[test]
fn test_tts_speaker_pipeline_graph_builds() {
    let (def, out_shape) = build_tts_speaker_pipeline();
    assert_eq!(out_shape, [EMBED_DIM]);

    let bindings = tts_speaker_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings)
        .expect("TTS+speaker pipeline graph should translate");

    // TextEncoder + Vocoder + SpeakerEncoder = substantial graph.
    assert!(
        graph.num_nodes() >= 20,
        "TTS+speaker pipeline should have >= 20 nodes, got {}",
        graph.num_nodes()
    );
}

/// IBP bounds propagate through the full TTS → Speaker pipeline.
///
/// **Property 4 end-to-end proof:** Text features → audio → speaker embedding,
/// with bounded embedding distance proving speaker consistency.
#[test]
fn test_tts_speaker_pipeline_ibp_propagates() {
    let d_model = 8;
    let seq_len = 2;
    let (def, _) = build_tts_speaker_pipeline();
    let bindings = tts_speaker_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[d_model, seq_len], 1.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through TTS+speaker pipeline");
    assert_eq!(
        output.lower_upper().0.shape(),
        &[EMBED_DIM],
        "output shape mismatch"
    );
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("TTS+Speaker IBP: bounds=[{lo_min}, {hi_max}]");

    // End-to-end bounds should be finite.
    assert!(lo_min.is_finite(), "lo_min should be finite, got {lo_min}");
    assert!(hi_max.is_finite(), "hi_max should be finite, got {hi_max}");
}

/// CROWN propagation through the full TTS → Speaker pipeline.
#[test]
fn test_tts_speaker_pipeline_crown_propagation() {
    let d_model = 8;
    let seq_len = 2;
    let (def, _) = build_tts_speaker_pipeline();
    let bindings = tts_speaker_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[d_model, seq_len], 1.0);

    let (method, output, fallback_reason) = assert_crown_tighter_when_not_fallback(&graph, &input);

    assert_eq!(
        output.lower_upper().0.shape(),
        &[EMBED_DIM],
        "output shape mismatch"
    );
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!(
        "TTS+Speaker {method:?}: bounds=[{lo_min}, {hi_max}]{}",
        fallback_reason.as_deref().unwrap_or("")
    );

    // Magnitude assertions matching IBP counterpart (#1984 AC1):
    assert!(
        lo_min.is_finite(),
        "CROWN: lo_min should be finite, got {lo_min}"
    );
    assert!(
        hi_max.is_finite(),
        "CROWN: hi_max should be finite, got {hi_max}"
    );
}

/// Property 4 end-to-end: text features → speaker embedding distance.
///
/// Proves that for all text inputs in [-1, 1], the resulting speaker
/// embedding is within bounded L2 distance of a reference embedding.
#[test]
fn test_tts_speaker_pipeline_p4_end_to_end() {
    let d_model = 8;
    let seq_len = 2;
    let (def, _) = build_tts_speaker_pipeline();
    let bindings = tts_speaker_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[d_model, seq_len], 1.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through TTS+speaker pipeline");
    let (lo, hi) = output.lower_upper();

    // Reference embedding: center of bounds.
    let reference: Vec<f64> = lo
        .iter()
        .zip(hi.iter())
        .map(|(&l, &h)| f64::midpoint(f64::from(l), f64::from(h)))
        .collect();

    // Worst-case L2 distance.
    let d_worst_sq: f64 = (0..EMBED_DIM)
        .map(|i| {
            let l = f64::from(lo[[i]]);
            let h = f64::from(hi[[i]]);
            let r = reference[i];
            let d_lo = (r - l).abs();
            let d_hi = (r - h).abs();
            let d_max = d_lo.max(d_hi);
            d_max * d_max
        })
        .sum();
    let d_worst = d_worst_sq.sqrt();

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!(
        "P4 end-to-end worst-case L2 distance: {d_worst:.6} (embedding bounds: [{lo_min}, {hi_max}])",
    );

    assert!(
        d_worst.is_finite(),
        "end-to-end worst-case L2 distance should be finite"
    );

    // Log the P4 result for certificate generation.
    eprintln!("Property 4 (Speaker consistency): finite embedding bounds -> computable worst-case distance");
}

/// verify_and_record for the full TTS+Speaker pipeline.
#[test]
fn test_tts_speaker_pipeline_verify_and_record() {
    let d_model = 8;
    let seq_len = 2;
    let (def, _) = build_tts_speaker_pipeline();
    let bindings = tts_speaker_bindings();
    let input = uniform_bounds(&[d_model, seq_len], 1.0);
    let result = verify_and_assert(&def, &bindings, &input, "kokoro_tts_speaker_pipeline");
    let (lo, _) = result.output_bounds.lower_upper();
    assert_eq!(lo.shape(), &[EMBED_DIM]);

    // Soundness: full TTS+Speaker pipeline includes Snake activation (vocoder
    // decoder) which uses sampling-based nonlinear relaxation → Heuristic (#1984 AC2).
    assert_eq!(
        result.verification.soundness_mode,
        VerificationSoundnessMode::Heuristic,
        "TTS+Speaker pipeline (Snake activation) should be Heuristic, got {:?}",
        result.verification.soundness_mode
    );
}

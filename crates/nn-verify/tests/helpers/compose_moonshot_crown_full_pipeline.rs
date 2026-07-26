// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Integration tests: 5-stage fully CROWN-backed moonshot pipeline.
//!
//! Extends `compose_moonshot_crown_prosody.rs` (3-stage) by adding the
//! Kokoro ProsodyPredictor (duration model) as a real CROWN-backed stage.
//!
//! The production Kokoro TTS pipeline has parallel branches:
//!   - F0EnergyPredictor computes pitch/energy from text+style
//!   - ProsodyPredictor computes duration logits from text+style
//!   - Both condition the decoder through style/alignment
//!
//! The sequential pipeline framework composes stages linearly, so we model
//! the parallel-to-sequential conversion as:
//!
//! ```text
//! Stage 1: F0EnergyPredictor (CROWN)
//!   flat_input [24] → shared Conv1d → F0 + Energy heads → [2]
//!
//! Stage 2: F0-to-prosody adapter (analytical)
//!   F0 output [2] → expand to prosody input domain [12]
//!
//! Stage 3: ProsodyPredictor (CROWN)
//!   flat_input [12] → Conv1d → AdaLayerNorm → LSTM → Linear → [1]
//!
//! Stage 4: Duration-to-decoder adapter (analytical)
//!   Duration [1] → expand to decoder input domain [8, 4]
//!
//! **CROWN status (#1769):** CROWN falls back to IBP across all configurations
//! due to NY alpha selection (R1-927). Bounds are structurally valid
//! but not CROWN-tightened. CROWN-specific tightness assertions are skipped.
//!
//! Stage 5: Kokoro decoder (CROWN)
//!   [8, 4] → Conv1d → ConvTranspose1d → ResBlock → Conv1d → Exp → [4, 8]
//! ```
//!
//! Part of #1741 — THE MOONSHOT: First Provably Correct Voice.
//! Part of #1739 — Provable Computational Boundedness.

use super::common;

#[path = "kokoro_decoder.rs"]
mod kokoro_decoder;

#[path = "kokoro_f0_energy.rs"]
mod kokoro_f0_energy;

#[path = "kokoro_prosody.rs"]
mod kokoro_prosody;

#[path = "pipeline_attention.rs"]
mod pipeline_attention;

#[path = "pipeline_full_crown.rs"]
mod pipeline_full_crown;

use common::{assert_bounds_valid, bounds_min_max, uniform_bounds};
use kokoro_f0_energy::{build_kokoro_f0_energy, kokoro_f0_energy_bindings};
use kokoro_prosody::{build_kokoro_prosody_single_block, kokoro_prosody_bindings};
use nn_verify::{propagate_with_crown_fallback, tensor_kernel_to_graph};
use pipeline_attention::{
    build_attention_certificate_for_pipeline, build_synthetic_speaker, build_synthetic_timing,
};
use pipeline_full_crown::{build_full_crown_pipeline, max_abs_bound};

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

/// CROWN propagation through ProsodyPredictor produces valid bounds.
#[test]
fn test_prosody_predictor_crown_produces_valid_bounds() {
    let (prosody_def, _) = build_kokoro_prosody_single_block();
    let prosody_bindings = kokoro_prosody_bindings();
    let prosody_graph =
        tensor_kernel_to_graph(&prosody_def, &prosody_bindings).expect("ProsodyPredictor graph");
    let prosody_input = uniform_bounds(&[kokoro_prosody::FLAT_INPUT_SIZE], 1.0);

    let (method, output, _) = propagate_with_crown_fallback(&prosody_graph, &prosody_input)
        .expect("Prosody CROWN propagation");

    assert_bounds_valid(&output);

    let (lo, hi) = output.lower_upper();
    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!(
        "ProsodyPredictor CROWN: method={method:?}, output_len={}, \
         bounds=[{lo_min:.6}, {hi_max:.6}]",
        lo.len(),
    );

    assert!(lo.iter().all(|x| x.is_finite()), "lower bounds finite");
    assert!(hi.iter().all(|x| x.is_finite()), "upper bounds finite");
}

/// 5-stage pipeline moonshot certificate with 3 real CROWN stages.
///
/// This is the most complete moonshot pipeline test:
///   Stage 1: F0EnergyPredictor (CROWN)
///   Stage 2: F0-to-prosody adapter (analytical)
///   Stage 3: ProsodyPredictor  (CROWN)
///   Stage 4: Duration-to-decoder adapter (analytical)
///   Stage 5: Kokoro decoder    (CROWN)
#[test]
fn test_full_crown_pipeline_certificate() {
    let (stages, dim, f0_m, prosody_m, dec_m) = build_full_crown_pipeline();

    let bundle = nn_tts_verify::moonshot_crown::verify_moonshot_from_stages(&stages, dim)
        .expect("5-stage moonshot verification");

    eprintln!(
        "5-stage pipeline: dim={dim}, stages={}, all_proven={}, \
         f0={f0_m:?}, prosody={prosody_m:?}, decoder={dec_m:?}",
        stages.len(),
        bundle.all_proven,
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

    // 5-stage pipeline: 4 junctions.
    assert!(bundle.pipeline_cert.is_valid, "pipeline must be valid");
    assert_eq!(
        bundle.pipeline_cert.junctions.len(),
        4,
        "5-stage pipeline should have 4 junctions"
    );

    // P1 (non-silence) bound must be non-zero.
    let p1 = &bundle.results[0];
    assert!(
        p1.bound_value > 0.0,
        "P1 bound_value={} should be > 0.0",
        p1.bound_value
    );
}

/// All 6 properties on the full CROWN pipeline.
#[test]
fn test_full_pipeline_all_6_properties() {
    let (stages, dim, _, _, _) = build_full_crown_pipeline();

    let bounds_cert =
        nn_tts_verify::pipeline::verify_pipeline(&stages).expect("pipeline verification");

    let timing_cert = build_synthetic_timing(&bounds_cert, dim);
    let speaker_evidence = build_synthetic_speaker();

    let bundle = nn_tts_verify::moonshot_crown::verify_all_crown_properties(
        &bounds_cert,
        &timing_cert,
        &speaker_evidence,
        dim,
    );

    eprintln!(
        "5-stage all 6 properties: checked={}, all_proven={}",
        bundle.results.len(),
        bundle.all_proven
    );

    for result in &bundle.results {
        eprintln!(
            "  P{}: {} — proven={}, level={:?}",
            result.property_index + 1,
            result.property_name,
            result.proven,
            result.level,
        );
    }

    assert_eq!(bundle.results.len(), 6, "must check 6 properties");
    assert!(bundle.results[3].proven, "P4 speaker consistency proven");
    assert!(bundle.results[4].proven, "P5 temporal boundedness proven");
}

/// Both CROWN models contribute to the adapter bounds.
#[test]
fn test_both_crown_models_inform_adapter() {
    // Get F0 bounds.
    let (f0_def, _) = build_kokoro_f0_energy();
    let f0_bindings = kokoro_f0_energy_bindings();
    let f0_graph = tensor_kernel_to_graph(&f0_def, &f0_bindings).expect("F0EnergyPredictor graph");
    let f0_input = uniform_bounds(&[kokoro_f0_energy::FLAT_INPUT_SIZE], 1.0);
    let (_, f0_output, _) = propagate_with_crown_fallback(&f0_graph, &f0_input).expect("F0 CROWN");
    let (f0_lo, f0_hi) = f0_output.lower_upper();
    let f0_lo_s = f0_lo.as_slice().expect("contiguous");
    let f0_hi_s = f0_hi.as_slice().expect("contiguous");

    // Get prosody (duration) bounds.
    let (prosody_def, _) = build_kokoro_prosody_single_block();
    let prosody_bindings = kokoro_prosody_bindings();
    let prosody_graph =
        tensor_kernel_to_graph(&prosody_def, &prosody_bindings).expect("ProsodyPredictor graph");
    let prosody_input = uniform_bounds(&[kokoro_prosody::FLAT_INPUT_SIZE], 1.0);
    let (_, prosody_output, _) =
        propagate_with_crown_fallback(&prosody_graph, &prosody_input).expect("Prosody CROWN");
    let (dur_lo, dur_hi) = prosody_output.lower_upper();
    let dur_lo_s = dur_lo.as_slice().expect("contiguous");
    let dur_hi_s = dur_hi.as_slice().expect("contiguous");

    let f0_max = max_abs_bound(f0_lo_s, f0_hi_s);
    let dur_max = max_abs_bound(dur_lo_s, dur_hi_s);
    let combined = f0_max.max(dur_max);

    eprintln!(
        "Adapter bound sources: f0_max={f0_max:.6}, dur_max={dur_max:.6}, \
         combined={combined:.6}"
    );

    assert!(f0_max.is_finite(), "F0 max bound must be finite");
    assert!(dur_max.is_finite(), "Duration max bound must be finite");
    assert!(combined.is_finite(), "Combined bound must be finite");
    assert!(combined >= f0_max, "combined >= f0_max");
    assert!(combined >= dur_max, "combined >= dur_max");
}

/// 5-stage pipeline with attention monotonicity upgrades P3 to CrownPartial.
///
/// This is Phase 10: wiring the attention monotonicity proofs from Phase 22-23
/// (W4) into the full CROWN-backed moonshot pipeline. The attention certificate
/// provides a real diagonal dominance proof instead of the range-ratio proxy.
#[test]
fn test_full_pipeline_all_6_with_attention() {
    let (stages, dim, _, _, _) = build_full_crown_pipeline();

    let bounds_cert =
        nn_tts_verify::pipeline::verify_pipeline(&stages).expect("pipeline verification");

    let timing_cert = build_synthetic_timing(&bounds_cert, dim);
    let speaker_evidence = build_synthetic_speaker();

    // Build attention certificate from PE-aware cross-attention scores.
    let attn_cert = build_attention_certificate_for_pipeline(0.01);
    assert!(
        attn_cert.is_proven,
        "attention certificate must be proven at input_bound=0.01"
    );

    let bundle = nn_tts_verify::moonshot_crown::verify_all_crown_properties_with_attention(
        &bounds_cert,
        &timing_cert,
        &speaker_evidence,
        Some(&attn_cert),
        dim,
    );

    eprintln!(
        "5-stage + attention: checked={}, all_proven={}",
        bundle.results.len(),
        bundle.all_proven
    );

    for result in &bundle.results {
        eprintln!(
            "  P{}: {} — proven={}, level={:?}",
            result.property_index + 1,
            result.property_name,
            result.proven,
            result.level,
        );
    }

    assert_eq!(bundle.results.len(), 6, "must check 6 properties");

    // P3 must be upgraded from proxy to real diagonal dominance proof.
    let p3 = &bundle.results[2];
    assert!(p3.proven, "P3 must be proven with attention certificate");
    assert_eq!(
        p3.level,
        nn_tts_verify::moonshot::VerificationLevel::CrownPartial,
        "P3 should be CrownPartial with IBP attention cert, got {:?}",
        p3.level
    );
    assert!(
        p3.explanation.contains("diagonal dominance"),
        "P3 must reference diagonal dominance: {}",
        p3.explanation
    );

    // Other properties still work.
    assert!(bundle.results[0].proven, "P1 non-silence");
    assert!(bundle.results[3].proven, "P4 speaker consistency");
    assert!(bundle.results[4].proven, "P5 temporal boundedness");
}

/// 5-stage pipeline with multi-head softmax attention via weight-space margins.
///
/// Bridges Phase 23's softmax-inclusive attention proofs into the moonshot
/// pipeline via `from_multi_head_weight_margins()`.
#[test]
fn test_full_pipeline_with_multi_head_attention() {
    let (stages, dim, _, _, _) = build_full_crown_pipeline();

    let bounds_cert =
        nn_tts_verify::pipeline::verify_pipeline(&stages).expect("pipeline verification");

    let timing_cert = build_synthetic_timing(&bounds_cert, dim);
    let speaker_evidence = build_synthetic_speaker();

    // Simulate 2-head softmax attention weight margins (Phase 23 style).
    // Per-head per-row margins: all positive → monotonicity proven in all heads.
    let head0_margins = vec![0.15, 0.12, 0.18, 0.10]; // 4 decoder steps
    let head1_margins = vec![0.20, 0.08, 0.14, 0.11];

    let attn_cert = nn_tts_verify::monotonicity::from_multi_head_weight_margins(
        &[head0_margins, head1_margins],
        4, // decoder_steps
        4, // encoder_positions
        1.0,
        "IBP",
    )
    .expect("valid margins should not fail");

    assert!(attn_cert.is_proven, "multi-head certificate must be proven");

    let bundle = nn_tts_verify::moonshot_crown::verify_all_crown_properties_with_attention(
        &bounds_cert,
        &timing_cert,
        &speaker_evidence,
        Some(&attn_cert),
        dim,
    );

    // P3 must be upgraded.
    let p3 = &bundle.results[2];
    assert!(p3.proven, "P3 must be proven with multi-head certificate");
    assert_eq!(
        p3.level,
        nn_tts_verify::moonshot::VerificationLevel::CrownPartial,
    );

    // Minimum margin should be 0.08 (head1, step1).
    assert!(
        (attn_cert.min_margin - 0.08).abs() < 1e-10,
        "min_margin={}, expected 0.08",
        attn_cert.min_margin
    );
}

/// Record the 5-stage pipeline in VerifyStatus.
#[test]
fn test_full_pipeline_verify_and_record() {
    use nn_verify::{verify_tensor_and_record, VerifyStatus};

    // Record F0EnergyPredictor.
    let (f0_def, _) = build_kokoro_f0_energy();
    let f0_bindings = kokoro_f0_energy_bindings();
    let f0_input = uniform_bounds(&[kokoro_f0_energy::FLAT_INPUT_SIZE], 1.0);

    let mut status = VerifyStatus::default();
    let f0_result = verify_tensor_and_record(
        &mut status,
        &f0_def,
        &f0_bindings,
        &f0_input,
        Some("moonshot_full_f0_energy"),
    )
    .expect("record F0EnergyPredictor");

    assert!(f0_result.verification.is_finite);

    // Record ProsodyPredictor.
    let (prosody_def, _) = build_kokoro_prosody_single_block();
    let prosody_bindings = kokoro_prosody_bindings();
    let prosody_input = uniform_bounds(&[kokoro_prosody::FLAT_INPUT_SIZE], 1.0);

    let prosody_result = verify_tensor_and_record(
        &mut status,
        &prosody_def,
        &prosody_bindings,
        &prosody_input,
        Some("moonshot_full_prosody_predictor"),
    )
    .expect("record ProsodyPredictor");

    assert!(prosody_result.verification.is_finite);

    // Build and verify full 5-stage pipeline.
    let (stages, dim, _, _, _) = build_full_crown_pipeline();
    let bundle = nn_tts_verify::moonshot_crown::verify_moonshot_from_stages(&stages, dim)
        .expect("5-stage moonshot from stages");

    eprintln!(
        "Recorded 5-stage pipeline: f0={:?}, prosody={:?}, \
         pipeline_valid={}, properties={}",
        f0_result.verification.method,
        prosody_result.verification.method,
        bundle.pipeline_cert.is_valid,
        bundle.results.len(),
    );

    assert!(bundle.pipeline_cert.is_valid);
}

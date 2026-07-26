// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Integration tests: Attention monotonicity integrated into moonshot pipeline.
//!
//! This test bridges the gap identified in `compose_cross_attention_monotonicity.rs`
//! where the comment "NOT YET INTEGRATED into MoonshotCrownBundle pipeline" was
//! noted. Now, `verify_all_crown_properties_with_attention()` accepts an
//! `AttentionMonotonicityCertificate` and upgrades Property 3 (Intelligibility)
//! from proxy-based CrownPartial to real CrownProven (diagonal dominance proof).
//!
//! Test architecture:
//!   1. Build 3-stage TTS pipeline (prosody → decoder → post-processing)
//!   2. Build PE-aware attention score graph (Q=Variable, K=Constant)
//!   3. Run IBP on attention graph → score bounds
//!   4. Feed score bounds into `interpret_attention_monotonicity()` → certificate
//!   5. Feed pipeline + timing + speaker + attention into
//!      `verify_all_crown_properties_with_attention()`
//!   6. Assert P3 achieves CrownPartial (IBP) or CrownProven (CROWN) instead
//!      of Empirical
//!
//! Part of #1741 — THE MOONSHOT: First Provably Correct Voice.

#[path = "kokoro_decoder.rs"]
mod kokoro_decoder;

use super::common::uniform_bounds;
use kokoro_decoder::{
    build_kokoro_decoder, kokoro_decoder_bindings, OUT_CHANNELS, TIME_IN, TIME_UP,
};
use nn_dsl::tensor_block_builder::TensorBlockBuilder;
use nn_tts_verify::moonshot::VerificationLevel;
use nn_verify::{
    propagate_with_crown_fallback, tensor_kernel_to_graph, PropMethod, TensorParamBinding,
};
use ndarray::{ArrayD, IxDyn};

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

// ---------------------------------------------------------------------------
// Pipeline builder (reused from compose_moonshot_certificate_pipeline.rs)
// ---------------------------------------------------------------------------

/// Build a 3-stage TTS pipeline from actual CROWN propagation.
fn build_three_stage_tts_pipeline() -> (
    Vec<nn_tts_verify::pipeline::VerifiedStage>,
    usize, // dimension
) {
    let (def, _) = build_kokoro_decoder();
    let bindings = kokoro_decoder_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("kokoro decoder graph");
    let input = uniform_bounds(&[8, TIME_IN], 1.0);

    let (method, output, _) =
        propagate_with_crown_fallback(&graph, &input).expect("CROWN propagation");

    let (out_lo, out_hi) = output.lower_upper();

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

    let stage2 = nn_tts_verify::pipeline::stage_from_propagation(
        "kokoro_decoder",
        &input,
        &output,
        &method,
    );

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

/// Build synthetic timing certificate.
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

/// Build synthetic speaker consistency evidence.
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

// ---------------------------------------------------------------------------
// Attention monotonicity builder (PE-aware cross-attention scores)
// ---------------------------------------------------------------------------

/// Build a PE-aware attention score graph and produce a monotonicity certificate.
///
/// Architecture: Scores = (hidden + PE) @ K^T / sqrt(d)
///   PE contributes a constant diagonally-dominant term.
///   hidden is the Variable with tiny bounds.
///
/// Returns `(certificate, propagation_mode)`.
fn build_attention_certificate(
    input_bound: f32,
) -> (
    nn_tts_verify::monotonicity::AttentionMonotonicityCertificate,
    String,
) {
    let t = 4; // decoder/encoder positions
    let d = 8; // embedding dimension

    let mut b = TensorBlockBuilder::new("moonshot_attn_integration");
    let hidden = b.add_input("hidden", &[t, d]);
    let pe = b.add_input("pe", &[t, d]);
    let k = b.add_input("key", &[t, d]);

    // Q = hidden + PE
    let q = b.add_binary_add(hidden, pe, &[t, d]);

    // Scores = Q @ K^T / sqrt(d) → [T, T]
    let scale = 1.0 / (d as f32).sqrt();
    let scores = b.add_matmul(q, k, true, Some(scale), &[t, t]);
    let def = b.build(scores).expect("valid attention graph");

    // Identity-like K: each position has large value in distinct columns.
    let k_scale = 3.0;
    let cols_per = d / t;
    let mut k_data = vec![0.0f32; t * d];
    for pos in 0..t {
        for c in 0..cols_per {
            k_data[pos * d + pos * cols_per + c] = k_scale;
        }
    }
    let k_tensor = ArrayD::from_shape_vec(IxDyn(&[t, d]), k_data).expect("K");

    // PE = K (same structure) → diagonally dominant constant component.
    let bindings = vec![
        TensorParamBinding::Variable,                         // hidden
        TensorParamBinding::ConstantTensor(k_tensor.clone()), // pe = K
        TensorParamBinding::ConstantTensor(k_tensor),         // key = K
    ];

    let input = uniform_bounds(&[t, d], input_bound);
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");
    let output = graph.propagate_ibp(&input).expect("IBP propagation");

    let (lo, hi) = output.lower_upper();
    let score_lower: Vec<f32> = lo.iter().copied().collect();
    let score_upper: Vec<f32> = hi.iter().copied().collect();

    let mode = "IBP".to_string();
    let cert = nn_tts_verify::monotonicity::interpret_attention_monotonicity(
        &score_lower,
        &score_upper,
        t,
        t,
        f64::from(input_bound),
        &mode,
    )
    .expect("valid monotonicity certificate");

    (cert, mode)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

/// Verify that `verify_all_crown_properties_with_attention` produces
/// CrownPartial for P3 when given a proven IBP attention certificate.
///
/// This is the core integration test: the full moonshot pipeline (6 properties)
/// with a real attention monotonicity certificate replacing the proxy check.
#[test]
fn test_moonshot_all_6_with_attention_certificate() {
    let (stages, dim) = build_three_stage_tts_pipeline();
    let bounds_cert =
        nn_tts_verify::pipeline::verify_pipeline(&stages).expect("pipeline verification");
    let timing_cert = build_timing_certificate(&bounds_cert, dim);
    let speaker_evidence = build_speaker_evidence();
    let (attn_cert, _mode) = build_attention_certificate(0.01);

    // Pre-condition: attention certificate is proven.
    assert!(
        attn_cert.is_proven,
        "attention certificate must be proven for this test"
    );

    let bundle = nn_tts_verify::moonshot_crown::verify_all_crown_properties_with_attention(
        &bounds_cert,
        &timing_cert,
        &speaker_evidence,
        Some(&attn_cert),
        dim,
    );

    assert_eq!(bundle.results.len(), 6, "must check all 6 properties");

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

    // P3 (intelligibility): with proven attention certificate using IBP,
    // the level should be CrownPartial (IBP-proven monotonicity).
    let p3 = &bundle.results[2];
    assert!(p3.proven, "P3 must be proven with attention certificate");
    assert_eq!(
        p3.level,
        VerificationLevel::CrownPartial,
        "P3 should be CrownPartial with IBP attention certificate, got {:?}",
        p3.level
    );
    assert!(
        p3.bound_value > 0.0,
        "P3 bound_value (min_margin) should be positive, got {}",
        p3.bound_value
    );
    assert!(
        p3.explanation.contains("diagonal dominance"),
        "P3 explanation should reference diagonal dominance, got: {}",
        p3.explanation
    );

    // Other properties should still work as before.
    assert!(bundle.results[0].proven, "P1 non-silence must be proven");
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

/// Verify that `verify_all_crown_properties_with_attention` falls back
/// to the proxy check when no attention certificate is provided (`None`).
///
/// This confirms backwards compatibility with the non-attention path.
#[test]
fn test_moonshot_with_attention_none_falls_back_to_proxy() {
    let (stages, dim) = build_three_stage_tts_pipeline();
    let bounds_cert =
        nn_tts_verify::pipeline::verify_pipeline(&stages).expect("pipeline verification");
    let timing_cert = build_timing_certificate(&bounds_cert, dim);
    let speaker_evidence = build_speaker_evidence();

    // Without attention certificate — should produce same result as
    // verify_all_crown_properties.
    let bundle_with_attn =
        nn_tts_verify::moonshot_crown::verify_all_crown_properties_with_attention(
            &bounds_cert,
            &timing_cert,
            &speaker_evidence,
            None, // no attention certificate
            dim,
        );

    let bundle_without = nn_tts_verify::moonshot_crown::verify_all_crown_properties(
        &bounds_cert,
        &timing_cert,
        &speaker_evidence,
        dim,
    );

    assert_eq!(bundle_with_attn.results.len(), bundle_without.results.len());

    // P3 should have identical results when no attention cert is provided.
    let p3_with = &bundle_with_attn.results[2];
    let p3_without = &bundle_without.results[2];

    assert_eq!(
        p3_with.proven, p3_without.proven,
        "P3 proven should match when no attention cert"
    );
    assert_eq!(
        p3_with.level, p3_without.level,
        "P3 level should match when no attention cert"
    );
    assert!(
        (p3_with.bound_value - p3_without.bound_value).abs() < 1e-6,
        "P3 bound_value should match: {} vs {}",
        p3_with.bound_value,
        p3_without.bound_value
    );
}

/// Verify that a non-proven attention certificate falls back to proxy.
///
/// When the attention certificate has `is_proven == false` (e.g., wide
/// input bounds that prevent diagonal dominance proof), P3 should use
/// the proxy check, not the monotonicity path.
#[test]
fn test_moonshot_non_proven_attention_falls_back() {
    let (stages, dim) = build_three_stage_tts_pipeline();
    let bounds_cert =
        nn_tts_verify::pipeline::verify_pipeline(&stages).expect("pipeline verification");
    let timing_cert = build_timing_certificate(&bounds_cert, dim);
    let speaker_evidence = build_speaker_evidence();

    // Wide input bounds → diagonal dominance NOT provable.
    let (wide_cert, _) = build_attention_certificate(5.0);
    assert!(
        !wide_cert.is_proven,
        "wide input bounds should prevent diagonal dominance proof"
    );

    let bundle = nn_tts_verify::moonshot_crown::verify_all_crown_properties_with_attention(
        &bounds_cert,
        &timing_cert,
        &speaker_evidence,
        Some(&wide_cert),
        dim,
    );

    // P3 should NOT reference diagonal dominance — it fell back to proxy.
    let p3 = &bundle.results[2];
    assert!(
        !p3.explanation.contains("diagonal dominance"),
        "non-proven attention cert should fall back to proxy, got: {}",
        p3.explanation
    );
}

/// Verify P3 level distinction: sound CROWN-family modes vs IBP mode.
///
/// When `propagation_mode` is in the sound CROWN family and `is_proven == true`,
/// P3 should achieve CrownProven. When mode is "IBP", P3 should be CrownPartial.
#[test]
fn test_p3_level_sound_crown_family_vs_ibp_attention_mode() {
    let (stages, dim) = build_three_stage_tts_pipeline();
    let bounds_cert =
        nn_tts_verify::pipeline::verify_pipeline(&stages).expect("pipeline verification");
    let timing_cert = build_timing_certificate(&bounds_cert, dim);
    let speaker_evidence = build_speaker_evidence();

    // Get a proven IBP certificate and check its level.
    let (ibp_cert, _) = build_attention_certificate(0.01);
    assert!(ibp_cert.is_proven);
    assert_eq!(ibp_cert.propagation_mode, "IBP");

    let ibp_bundle = nn_tts_verify::moonshot_crown::verify_all_crown_properties_with_attention(
        &bounds_cert,
        &timing_cert,
        &speaker_evidence,
        Some(&ibp_cert),
        dim,
    );
    assert_eq!(
        ibp_bundle.results[2].level,
        VerificationLevel::CrownPartial,
        "IBP attention cert should produce CrownPartial for P3"
    );

    for crown_mode in ["CROWN", "AlphaCrown", "BetaCrown"] {
        let crown_cert = nn_tts_verify::monotonicity::AttentionMonotonicityCertificate {
            decoder_steps: ibp_cert.decoder_steps,
            encoder_positions: ibp_cert.encoder_positions,
            min_margin: ibp_cert.min_margin,
            is_proven: true,
            row_margins: ibp_cert.row_margins.clone(),
            input_bound: ibp_cert.input_bound,
            propagation_mode: crown_mode.to_string(),
        };

        let crown_bundle =
            nn_tts_verify::moonshot_crown::verify_all_crown_properties_with_attention(
                &bounds_cert,
                &timing_cert,
                &speaker_evidence,
                Some(&crown_cert),
                dim,
            );
        assert_eq!(
            crown_bundle.results[2].level,
            VerificationLevel::CrownProven,
            "{crown_mode} attention cert should produce CrownProven for P3"
        );

        eprintln!(
            "P3 IBP level: {:?}, {crown_mode} level: {:?}",
            ibp_bundle.results[2].level, crown_bundle.results[2].level
        );
    }
}

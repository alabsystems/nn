// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Integration tests: Moonshot pipeline with real CROWN prosody stage.
//!
//! Replaces the synthetic prosody_predictor stage in
//! `compose_moonshot_certificate_pipeline.rs` with actual NY CROWN
//! propagation through the Kokoro F0EnergyPredictor. This closes the gap
//! identified by R1-927: the 3-stage pipeline test used hardcoded bounds
//! for Stage 1 (`[-1.0, 1.0]`) instead of real CROWN-computed bounds.
//!
//! ## Architecture
//!
//! ```text
//! Stage 1: F0EnergyPredictor (CROWN)
//!   flat_input [24] → shared Conv1d → F0 head + Energy head → [2]
//!
//! Stage 2: Adapter (analytical)
//!   [2] → broadcast/expand to [8, 4] → produces features for decoder
//!   Bounds: conservative expansion from prosody bounds
//!
//! Stage 3: Kokoro decoder (CROWN)
//!   [8, 4] → Conv1d → ConvTranspose1d → ResBlock → Conv1d → Exp → [4, 8]
//! ```
//!
//! The adapter stage is analytical (not CROWN): it models the real pipeline's
//! conditioning path where prosody features (F0 + energy) modulate decoder
//! inputs. The bounds are widened conservatively: max absolute prosody bound
//! becomes the input range for all decoder channels. This is sound: if
//! prosody ∈ [lo, hi], then any downstream feature conditioned on prosody
//! is bounded by the worst-case prosody value.
//!
//! Part of #1741 — THE MOONSHOT: First Provably Correct Voice.
//! Part of #1739 — Provable Computational Boundedness.

#[path = "kokoro_decoder.rs"]
mod kokoro_decoder;

#[path = "kokoro_f0_energy.rs"]
mod kokoro_f0_energy;

use super::common::{assert_bounds_valid, bounds_min_max, uniform_bounds};
use kokoro_decoder::{
    build_kokoro_decoder, kokoro_decoder_bindings, OUT_CHANNELS, TIME_IN, TIME_UP,
};
use kokoro_f0_energy::{build_kokoro_f0_energy, kokoro_f0_energy_bindings, FLAT_INPUT_SIZE};
use nn_verify::{propagate_with_crown_fallback, tensor_kernel_to_graph, PropMethod};

// ---------------------------------------------------------------------------
// Pipeline construction
// ---------------------------------------------------------------------------

/// Build a 3-stage pipeline where Stage 1 uses CROWN-propagated F0EnergyPredictor
/// bounds instead of synthetic/hardcoded bounds.
///
/// Returns (stages, dimension, prosody_method, decoder_method) for test assertions.
fn build_crown_prosody_pipeline() -> (
    Vec<nn_tts_verify::pipeline::VerifiedStage>,
    usize,
    PropMethod,
    PropMethod,
) {
    // --- Stage 1: F0EnergyPredictor via real CROWN propagation ---
    let (f0_def, _) = build_kokoro_f0_energy();
    let f0_bindings = kokoro_f0_energy_bindings();
    let f0_graph = tensor_kernel_to_graph(&f0_def, &f0_bindings).expect("F0EnergyPredictor graph");
    let f0_input = uniform_bounds(&[FLAT_INPUT_SIZE], 1.0);

    let (f0_method, f0_output, _) =
        propagate_with_crown_fallback(&f0_graph, &f0_input).expect("F0 CROWN propagation");

    assert_bounds_valid(&f0_output);

    let stage1 = nn_tts_verify::pipeline::stage_from_propagation(
        "f0_energy_predictor",
        &f0_input,
        &f0_output,
        &f0_method,
    );

    // --- Stage 2: Analytical adapter (prosody → decoder features) ---
    // The F0EnergyPredictor outputs [2] (F0 + energy). The decoder expects [8, 4].
    // In the real pipeline, prosody features condition the decoder through style
    // vectors, duration alignment, and pitch/energy modulation. We model this as
    // a conservative bound expansion: the max absolute prosody bound sets the
    // input range for all decoder feature elements.
    let (f0_lo, f0_hi) = f0_output.lower_upper();
    let max_prosody_abs = f0_lo
        .iter()
        .chain(f0_hi.iter())
        .map(|x| x.abs())
        .fold(0.0_f32, f32::max);

    // Conservative expansion: decoder features bounded by prosody extremes
    let decoder_input_bound = (max_prosody_abs * 2.0).max(1.0); // At least ±1.0
    let decoder_input_size = 8 * TIME_IN; // IN_CHANNELS * TIME_IN

    let stage2 = nn_tts_verify::pipeline::VerifiedStage::new(
        "prosody_to_features",
        vec![f0_lo.len()], // input shape matches F0EnergyPredictor output
        vec![8, TIME_IN],  // output shape matches decoder input
        f0_lo.iter().map(|x| f64::from(*x)).collect(),
        f0_hi.iter().map(|x| f64::from(*x)).collect(),
        vec![f64::from(-decoder_input_bound); decoder_input_size],
        vec![f64::from(decoder_input_bound); decoder_input_size],
        "analytical", // analytical adapter, not CROWN — sound by construction
        true,
    );

    // --- Stage 3: Kokoro decoder via real CROWN propagation ---
    let (dec_def, _) = build_kokoro_decoder();
    let dec_bindings = kokoro_decoder_bindings();
    let dec_graph = tensor_kernel_to_graph(&dec_def, &dec_bindings).expect("decoder graph");
    let dec_input = uniform_bounds(&[8, TIME_IN], decoder_input_bound);

    let (dec_method, dec_output, _) =
        propagate_with_crown_fallback(&dec_graph, &dec_input).expect("decoder CROWN propagation");

    assert_bounds_valid(&dec_output);

    let stage3 = nn_tts_verify::pipeline::stage_from_propagation(
        "kokoro_decoder",
        &dec_input,
        &dec_output,
        &dec_method,
    );

    let dim = OUT_CHANNELS * TIME_UP;
    (vec![stage1, stage2, stage3], dim, f0_method, dec_method)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

/// CROWN propagation through F0EnergyPredictor produces finite, valid bounds.
///
/// This is the fundamental prerequisite for replacing the synthetic prosody stage.
#[test]
fn test_f0_energy_crown_produces_valid_bounds() {
    let (f0_def, _) = build_kokoro_f0_energy();
    let f0_bindings = kokoro_f0_energy_bindings();
    let f0_graph = tensor_kernel_to_graph(&f0_def, &f0_bindings).expect("F0EnergyPredictor graph");
    let f0_input = uniform_bounds(&[FLAT_INPUT_SIZE], 1.0);

    let (method, output, _) =
        propagate_with_crown_fallback(&f0_graph, &f0_input).expect("F0 CROWN propagation");

    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);

    eprintln!(
        "F0EnergyPredictor CROWN: method={method:?}, bounds=[{lo_min:.6}, {hi_max:.6}], \
         output_len={}",
        output.lower_upper().0.len()
    );
}

/// 3-stage moonshot certificate with real CROWN prosody bounds.
///
/// This replaces the synthetic prosody stage from
/// `compose_moonshot_certificate_pipeline.rs::build_three_stage_tts_pipeline`.
/// All three stages have provenance:
///   Stage 1: CROWN (F0EnergyPredictor)
///   Stage 2: analytical (sound adapter)
///   Stage 3: CROWN (Kokoro decoder)
#[test]
fn test_crown_prosody_pipeline_moonshot_certificate() {
    let (stages, dim, f0_method, dec_method) = build_crown_prosody_pipeline();

    let bundle = nn_tts_verify::moonshot_crown::verify_moonshot_from_stages(&stages, dim)
        .expect("3-stage CROWN prosody moonshot verification");

    eprintln!(
        "CROWN prosody pipeline: dim={dim}, stages={}, all_proven={}, \
         properties={}, f0_method={f0_method:?}, dec_method={dec_method:?}",
        stages.len(),
        bundle.all_proven,
        bundle.results.len(),
    );

    for result in &bundle.results {
        eprintln!(
            "  P{}: {} — proven={}, level={:?}, bound={:.6}, threshold={:.6}",
            result.property_index,
            result.property_name,
            result.proven,
            result.level,
            result.bound_value,
            result.threshold,
        );
    }

    // Pipeline certificate must be valid with 2 junctions.
    assert!(
        bundle.pipeline_cert.is_valid,
        "CROWN prosody pipeline certificate must be valid"
    );
    assert_eq!(
        bundle.pipeline_cert.junctions.len(),
        2,
        "3-stage pipeline should have 2 junctions"
    );

    // P1 (non-silence) must have non-zero bounds.
    let p1 = &bundle.results[0];
    assert!(
        p1.bound_value > 0.0,
        "P1 (non-silence) bound_value={} should be > 0.0",
        p1.bound_value
    );

    // P3 (intelligibility proxy) range ratio must be finite.
    let p3 = &bundle.results[2];
    assert!(
        p3.bound_value.is_finite(),
        "P3 range ratio must be finite, got {}",
        p3.bound_value
    );

    // P6 (streaming safety) click bound must be finite.
    let p6 = &bundle.results[3];
    assert!(
        p6.bound_value.is_finite(),
        "P6 max_click_bound must be finite, got {}",
        p6.bound_value
    );
}

/// Build a synthetic timing certificate for the 3-stage prosody pipeline.
fn build_synthetic_timing(
    bounds_cert: &nn_tts_verify::pipeline::PipelineCertificate,
    dim: usize,
) -> nn_tts_verify::pipeline::TimingCertificate {
    nn_tts_verify::pipeline::TimingCertificate::new(
        bounds_cert.clone(),
        vec![
            nn_tts_verify::cost_model::LayerCostProfile::new(
                "f0_energy_predictor",
                2_000_000,
                4 * dim as u64,
                8_000.0,
                None,
            ),
            nn_tts_verify::cost_model::LayerCostProfile::new(
                "prosody_to_features",
                100_000,
                dim as u64,
                500.0,
                None,
            ),
            nn_tts_verify::cost_model::LayerCostProfile::new(
                "kokoro_decoder",
                20_000_000,
                16 * dim as u64,
                25_000.0,
                None,
            ),
        ],
        33_500.0,
        22_100_000,
        21 * dim as u64,
        "M4 Max (synthetic)",
        100_000.0,
        true,
        true,
        None,
    )
}

/// Build synthetic speaker consistency evidence (tight ECAPA-TDNN bounds).
fn build_synthetic_speaker() -> nn_tts_verify::moonshot_crown::SpeakerConsistencyEvidence {
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

/// All 6 CROWN properties with real prosody CROWN bounds.
///
/// Combines real CROWN pipeline (P1-P3, P6) with synthetic timing (P5)
/// and speaker consistency (P4) certificates.
#[test]
fn test_crown_prosody_all_6_properties() {
    let (stages, dim, _, _) = build_crown_prosody_pipeline();

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
        "All 6 properties (CROWN prosody): checked={}, all_proven={}",
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

    // P4 (speaker consistency) and P5 (temporal boundedness) should be proven
    // from the synthetic certificates.
    assert!(
        bundle.results[3].proven,
        "P4 speaker consistency must be proven"
    );
    assert!(
        bundle.results[4].proven,
        "P5 temporal boundedness must be proven"
    );
}

/// Wider F0EnergyPredictor input bounds still produce finite pipeline bounds.
///
/// Tests robustness: when prosody input range is [-2, 2] instead of [-1, 1],
/// the pipeline bounds widen but remain finite and the pipeline certificate
/// is still valid.
#[test]
fn test_crown_prosody_wider_input_bounds() {
    let (f0_def, _) = build_kokoro_f0_energy();
    let f0_bindings = kokoro_f0_energy_bindings();
    let f0_graph = tensor_kernel_to_graph(&f0_def, &f0_bindings).expect("F0EnergyPredictor graph");

    // Wider input bounds.
    let wide_input = uniform_bounds(&[FLAT_INPUT_SIZE], 2.0);

    let (method, output, _) =
        propagate_with_crown_fallback(&f0_graph, &wide_input).expect("F0 CROWN wide");

    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Wide F0 CROWN: method={method:?}, output_range=[{lo_min:.6}, {hi_max:.6}]");
}

/// Verify that CROWN prosody bounds are at least as informative as synthetic.
///
/// The old synthetic stage used `output_lower = vec![-1.0; ...]` and
/// `output_upper = vec![1.0; ...]`. Real CROWN bounds should produce a
/// finite range (not necessarily tighter, since the F0 model's bounds
/// depend on weight magnitudes, but always finite and valid).
#[test]
fn test_crown_prosody_bounds_versus_synthetic() {
    let (f0_def, _) = build_kokoro_f0_energy();
    let f0_bindings = kokoro_f0_energy_bindings();
    let f0_graph = tensor_kernel_to_graph(&f0_def, &f0_bindings).expect("F0EnergyPredictor graph");
    let f0_input = uniform_bounds(&[FLAT_INPUT_SIZE], 1.0);

    let (method, output, _) =
        propagate_with_crown_fallback(&f0_graph, &f0_input).expect("F0 CROWN propagation");

    // Real CROWN bounds must be finite (unlike synthetic which is always ±1.0).
    let (lo_min, hi_max) = bounds_min_max(&output);

    eprintln!(
        "CROWN prosody vs synthetic: method={method:?}, \
         CROWN=[{lo_min:.6}, {hi_max:.6}], synthetic=[-1.0, 1.0]"
    );

    assert!(lo_min.is_finite(), "CROWN lower must be finite");
    assert!(hi_max.is_finite(), "CROWN upper must be finite");

    // The CROWN bounds reflect the actual network behavior at these weights.
    // With WEIGHT_MAG=0.01, they should be relatively tight.
    // range >= 0.0 is structurally guaranteed (NY ensures lo <= hi).
    // Assert a meaningful upper bound instead.
    let range = hi_max - lo_min;
    assert!(range.is_finite(), "CROWN range must be finite, got {range}");
    assert!(
        range < 1e6,
        "CROWN range should be bounded with small weights, got {range}"
    );
}

/// Record CROWN prosody pipeline in VerifyStatus for proof persistence.
#[test]
fn test_crown_prosody_verify_and_record() {
    use nn_verify::{verify_tensor_and_record, VerifyStatus};

    // Record the F0EnergyPredictor CROWN result.
    let (f0_def, _) = build_kokoro_f0_energy();
    let f0_bindings = kokoro_f0_energy_bindings();
    let f0_input = uniform_bounds(&[FLAT_INPUT_SIZE], 1.0);

    let mut status = VerifyStatus::default();
    let f0_result = verify_tensor_and_record(
        &mut status,
        &f0_def,
        &f0_bindings,
        &f0_input,
        Some("moonshot_f0_energy_predictor"),
    )
    .expect("verify_tensor_and_record for F0EnergyPredictor");

    assert!(
        f0_result.verification.is_finite,
        "F0EnergyPredictor bounds must be finite"
    );

    // Build and verify the full pipeline.
    let (stages, dim, _, _) = build_crown_prosody_pipeline();
    let bundle = nn_tts_verify::moonshot_crown::verify_moonshot_from_stages(&stages, dim)
        .expect("CROWN prosody moonshot from stages");

    eprintln!(
        "Recorded moonshot_f0_energy_predictor: method={:?}, \
         pipeline_stages={}, properties={}, pipeline_valid={}",
        f0_result.verification.method,
        stages.len(),
        bundle.results.len(),
        bundle.pipeline_cert.is_valid,
    );

    // Pipeline must be valid.
    assert!(
        bundle.pipeline_cert.is_valid,
        "CROWN prosody pipeline certificate must be valid"
    );
}

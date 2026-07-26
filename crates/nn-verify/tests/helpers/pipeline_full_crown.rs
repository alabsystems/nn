// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! 5-stage fully CROWN-backed pipeline builder for moonshot integration tests.
//!
//! Extracted from `compose_moonshot_crown_full_pipeline.rs` to keep the
//! test file under 500 lines.
//!
//! Part of #1741 — THE MOONSHOT: First Provably Correct Voice.

use super::common::{assert_bounds_valid, uniform_bounds};
use super::kokoro_decoder::{
    build_kokoro_decoder, kokoro_decoder_bindings, OUT_CHANNELS, TIME_IN, TIME_UP,
};
use super::kokoro_f0_energy::{build_kokoro_f0_energy, kokoro_f0_energy_bindings};
use super::kokoro_prosody::{build_kokoro_prosody_single_block, kokoro_prosody_bindings};
use nn_verify::{propagate_with_crown_fallback, tensor_kernel_to_graph, PropMethod};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

pub(crate) fn max_abs_bound(lo: &[f32], hi: &[f32]) -> f32 {
    lo.iter()
        .chain(hi.iter())
        .map(|x| x.abs())
        .fold(0.0_f32, f32::max)
}

// ---------------------------------------------------------------------------
// 5-stage pipeline construction
// ---------------------------------------------------------------------------

/// Build a 5-stage pipeline with 3 real CROWN stages + 2 analytical adapters.
///
/// The pipeline chains: F0 → adapt → ProsodyPredictor → adapt → Decoder.
/// Both adapters are analytical (sound by construction): they conservatively
/// expand bounds from the source domain to the target domain.
pub(crate) fn build_full_crown_pipeline() -> (
    Vec<nn_tts_verify::pipeline::VerifiedStage>,
    usize,
    PropMethod,
    PropMethod,
    PropMethod,
) {
    // --- Stage 1: F0EnergyPredictor via real CROWN propagation ---
    let (f0_def, _) = build_kokoro_f0_energy();
    let f0_bindings = kokoro_f0_energy_bindings();
    let f0_graph = tensor_kernel_to_graph(&f0_def, &f0_bindings).expect("F0EnergyPredictor graph");
    let f0_input = uniform_bounds(&[super::kokoro_f0_energy::FLAT_INPUT_SIZE], 1.0);

    let (f0_method, f0_output, _) =
        propagate_with_crown_fallback(&f0_graph, &f0_input).expect("F0 CROWN propagation");
    assert_bounds_valid(&f0_output);

    let stage1 = nn_tts_verify::pipeline::stage_from_propagation(
        "f0_energy_predictor",
        &f0_input,
        &f0_output,
        &f0_method,
    );

    // --- Stage 2: F0-to-prosody adapter ---
    // Maps F0 output [2] → prosody input domain [12] (FLAT_INPUT_SIZE for ProsodyPredictor).
    // In production, both models consume text+style independently. Here we model
    // the sequential dependency: F0 output informs the prosody input range.
    let (f0_lo, f0_hi) = f0_output.lower_upper();
    let f0_lo_slice = f0_lo.as_slice().expect("contiguous f0_lo");
    let f0_hi_slice = f0_hi.as_slice().expect("contiguous f0_hi");
    let f0_max = max_abs_bound(f0_lo_slice, f0_hi_slice);
    let prosody_bound = (f0_max * 2.0).max(1.0);
    let prosody_flat_size = super::kokoro_prosody::FLAT_INPUT_SIZE;

    let stage2 = nn_tts_verify::pipeline::VerifiedStage::new(
        "f0_to_prosody_adapter",
        vec![f0_lo_slice.len()],
        vec![prosody_flat_size],
        f0_lo_slice.iter().map(|x| f64::from(*x)).collect(),
        f0_hi_slice.iter().map(|x| f64::from(*x)).collect(),
        vec![f64::from(-prosody_bound); prosody_flat_size],
        vec![f64::from(prosody_bound); prosody_flat_size],
        "analytical",
        true,
    );

    // --- Stage 3: ProsodyPredictor via real CROWN propagation ---
    let (prosody_def, _) = build_kokoro_prosody_single_block();
    let prosody_bindings = kokoro_prosody_bindings();
    let prosody_graph =
        tensor_kernel_to_graph(&prosody_def, &prosody_bindings).expect("ProsodyPredictor graph");
    let prosody_input = uniform_bounds(&[prosody_flat_size], prosody_bound);

    let (prosody_method, prosody_output, _) =
        propagate_with_crown_fallback(&prosody_graph, &prosody_input)
            .expect("Prosody CROWN propagation");
    assert_bounds_valid(&prosody_output);

    let stage3 = nn_tts_verify::pipeline::stage_from_propagation(
        "prosody_predictor",
        &prosody_input,
        &prosody_output,
        &prosody_method,
    );

    // --- Stage 4: Duration-to-decoder adapter ---
    // Maps duration output [1] → decoder input [8, 4].
    // Combines duration logit bounds with F0-derived bounds for decoder features.
    let (dur_lo, dur_hi) = prosody_output.lower_upper();
    let dur_lo_slice = dur_lo.as_slice().expect("contiguous dur_lo");
    let dur_hi_slice = dur_hi.as_slice().expect("contiguous dur_hi");
    let dur_max = max_abs_bound(dur_lo_slice, dur_hi_slice);
    let combined_max = f0_max.max(dur_max);
    let decoder_input_bound = (combined_max * 2.0).max(1.0);
    let decoder_input_size = 8 * TIME_IN;

    let stage4 = nn_tts_verify::pipeline::VerifiedStage::new(
        "duration_to_decoder_adapter",
        vec![dur_lo_slice.len()],
        vec![8, TIME_IN],
        dur_lo_slice.iter().map(|x| f64::from(*x)).collect(),
        dur_hi_slice.iter().map(|x| f64::from(*x)).collect(),
        vec![f64::from(-decoder_input_bound); decoder_input_size],
        vec![f64::from(decoder_input_bound); decoder_input_size],
        "analytical",
        true,
    );

    // --- Stage 5: Kokoro decoder via real CROWN propagation ---
    let (dec_def, _) = build_kokoro_decoder();
    let dec_bindings = kokoro_decoder_bindings();
    let dec_graph = tensor_kernel_to_graph(&dec_def, &dec_bindings).expect("decoder graph");
    let dec_input = uniform_bounds(&[8, TIME_IN], decoder_input_bound);

    let (dec_method, dec_output, _) =
        propagate_with_crown_fallback(&dec_graph, &dec_input).expect("decoder CROWN propagation");
    assert_bounds_valid(&dec_output);

    let stage5 = nn_tts_verify::pipeline::stage_from_propagation(
        "kokoro_decoder",
        &dec_input,
        &dec_output,
        &dec_method,
    );

    let dim = OUT_CHANNELS * TIME_UP;
    (
        vec![stage1, stage2, stage3, stage4, stage5],
        dim,
        f0_method,
        prosody_method,
        dec_method,
    )
}

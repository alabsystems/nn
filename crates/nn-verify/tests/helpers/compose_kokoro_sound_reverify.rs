// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Conservative-mode re-verification of Kokoro Tier 1 entries.
//!
//! After NY gc#4399 (CROWN linearization through normalization),
//! entries that were classified as `heuristic` solely due to normalization
//! `forward_mode` can be re-verified with `NormBoundsMode::Conservative`
//! to achieve `Sound` classification.
//!
//! `Conservative` sets `forward_mode: false` on all normalization layers,
//! which means NY's soundness scan finds no heuristic flags.
//! IBP through Conservative norms is provably sound. Bounds may be wider
//! than ForwardMode but remain non-vacuous for Kokoro's architecture.
//!
//! Part of #3422 D3: Post-bump verification map re-verification.
//! Part of #3351 T3.1: Kokoro soundness improvement.

#[path = "kokoro_decoder.rs"]
mod kokoro_decoder_helpers;

#[path = "kokoro_full_pipeline.rs"]
mod full_pipeline_helpers;

#[path = "kokoro_prosody.rs"]
mod prosody_helpers;

#[path = "kokoro_prosody_t4.rs"]
mod prosody_t4_helpers;

#[path = "kokoro_speaker_pipeline.rs"]
mod speaker_helpers;

use super::common::{bounds_min_max, uniform_bounds, verify_and_assert_with_config};
use full_pipeline_helpers::{
    build_kokoro_full_pipeline, build_kokoro_vocoder_only_pipeline, kokoro_full_pipeline_bindings,
    kokoro_vocoder_only_bindings, D_MODEL, SEQ_LEN,
};
use kokoro_decoder_helpers::{
    build_kokoro_decoder, build_kokoro_decoder_with_leaky_relu, kokoro_decoder_bindings,
    kokoro_decoder_leaky_relu_bindings, OUT_CHANNELS, TIME_IN, TIME_UP,
};
use nn_dsl::tensor_block_builder::TensorBlockBuilder;
use nn_verify::{
    NormBoundsMode, TensorParamBinding, VerificationSoundnessMode, VerifyConfig,
};
use ndarray::{ArrayD, IxDyn};
use prosody_helpers::{
    build_kokoro_prosody_single_block, build_kokoro_prosody_three_blocks, kokoro_prosody_bindings,
    kokoro_prosody_three_block_bindings, FLAT_INPUT_SIZE,
};
use prosody_t4_helpers::{build_kokoro_prosody_t4, kokoro_prosody_t4_bindings, FLAT_INPUT_SIZE_T4};
use speaker_helpers::{build_tts_speaker_pipeline, tts_speaker_bindings};

/// Vacuous width threshold — bounds wider than this are vacuous.
const VACUOUS_THRESHOLD: f32 = 100.0;

fn conservative_config() -> VerifyConfig {
    VerifyConfig::default().with_norm_mode(NormBoundsMode::Conservative)
}

fn conservative_force_crown_config() -> VerifyConfig {
    VerifyConfig::with_threshold(0.0)
        .expect("zero threshold is valid")
        .with_norm_mode(NormBoundsMode::Conservative)
        .with_require_sound(true)
}

// ===========================================================================
// Decoder entries (Tier 1)
// ===========================================================================

/// Re-verify `kokoro_decoder` with Conservative mode → expect Sound.
#[test]
fn test_sound_kokoro_decoder() {
    let (def, _) = build_kokoro_decoder();
    let bindings = kokoro_decoder_bindings();
    let input = uniform_bounds(&[8, TIME_IN], 1.0);

    let result = verify_and_assert_with_config(
        &def,
        &bindings,
        &input,
        "kokoro_decoder",
        &conservative_config(),
    );

    assert_eq!(
        result.verification.soundness_mode,
        VerificationSoundnessMode::Sound,
        "Conservative mode should produce Sound, got {:?}",
        result.verification.soundness_mode
    );
    let width = result.verification.output_width;
    assert!(
        width < VACUOUS_THRESHOLD,
        "Conservative bounds should be non-vacuous, width={width}"
    );

    let (lo, _) = result.output_bounds.lower_upper();
    assert_eq!(lo.shape(), &[OUT_CHANNELS, TIME_UP]);
    let (lo_min, hi_max) = bounds_min_max(&result.output_bounds);
    eprintln!(
        "kokoro_decoder Conservative: bounds=[{lo_min}, {hi_max}], width={width}, soundness=Sound"
    );
}

/// Re-verify `kokoro_decoder_leaky_relu` with Conservative mode → expect Sound.
#[test]
fn test_sound_kokoro_decoder_leaky_relu() {
    let (def, _) = build_kokoro_decoder_with_leaky_relu();
    let bindings = kokoro_decoder_leaky_relu_bindings();
    let input = uniform_bounds(&[8, TIME_IN], 1.0);

    let result = verify_and_assert_with_config(
        &def,
        &bindings,
        &input,
        "kokoro_decoder_leaky_relu",
        &conservative_config(),
    );

    assert_eq!(
        result.verification.soundness_mode,
        VerificationSoundnessMode::Sound,
        "Conservative mode should produce Sound, got {:?}",
        result.verification.soundness_mode
    );
    let width = result.verification.output_width;
    assert!(
        width < VACUOUS_THRESHOLD,
        "Conservative bounds should be non-vacuous, width={width}"
    );
    eprintln!("kokoro_decoder_leaky_relu Conservative: width={width}, soundness=Sound");
}

// ===========================================================================
// Prosody entries (Tier 1)
// ===========================================================================

/// Re-verify `kokoro_prosody_single_block` with Conservative mode → expect Sound.
#[test]
fn test_sound_kokoro_prosody_single_block() {
    let (def, _) = build_kokoro_prosody_single_block();
    let bindings = kokoro_prosody_bindings();
    let input = uniform_bounds(&[FLAT_INPUT_SIZE], 1.0);

    let result = verify_and_assert_with_config(
        &def,
        &bindings,
        &input,
        "kokoro_prosody_single_block",
        &conservative_config(),
    );

    assert_eq!(
        result.verification.soundness_mode,
        VerificationSoundnessMode::Sound,
        "Conservative mode should produce Sound, got {:?}",
        result.verification.soundness_mode
    );
    let width = result.verification.output_width;
    assert!(
        width < VACUOUS_THRESHOLD,
        "Conservative bounds should be non-vacuous, width={width}"
    );
    eprintln!("kokoro_prosody_single_block Conservative: width={width}, soundness=Sound");
}

/// Re-verify `kokoro_prosody_three_blocks` with Conservative mode → expect Sound.
#[test]
fn test_sound_kokoro_prosody_three_blocks() {
    let (def, _) = build_kokoro_prosody_three_blocks();
    let bindings = kokoro_prosody_three_block_bindings();
    let input = uniform_bounds(&[FLAT_INPUT_SIZE], 1.0);

    let result = verify_and_assert_with_config(
        &def,
        &bindings,
        &input,
        "kokoro_prosody_three_blocks",
        &conservative_config(),
    );

    assert_eq!(
        result.verification.soundness_mode,
        VerificationSoundnessMode::Sound,
        "Conservative mode should produce Sound, got {:?}",
        result.verification.soundness_mode
    );
    let width = result.verification.output_width;
    assert!(
        width < VACUOUS_THRESHOLD,
        "Conservative bounds should be non-vacuous, width={width}"
    );
    eprintln!("kokoro_prosody_three_blocks Conservative: width={width}, soundness=Sound");
}

/// Re-verify `kokoro_prosody_t4` with Conservative mode → expect Sound.
#[test]
fn test_sound_kokoro_prosody_t4() {
    let (def, _) = build_kokoro_prosody_t4();
    let bindings = kokoro_prosody_t4_bindings();
    let input = uniform_bounds(&[FLAT_INPUT_SIZE_T4], 1.0);

    let result = verify_and_assert_with_config(
        &def,
        &bindings,
        &input,
        "kokoro_prosody_t4",
        &conservative_config(),
    );

    assert_eq!(
        result.verification.soundness_mode,
        VerificationSoundnessMode::Sound,
        "Conservative mode should produce Sound, got {:?}",
        result.verification.soundness_mode
    );
    let width = result.verification.output_width;
    assert!(
        width < VACUOUS_THRESHOLD,
        "Conservative bounds should be non-vacuous, width={width}"
    );
    eprintln!("kokoro_prosody_t4 Conservative: width={width}, soundness=Sound");
}

// ===========================================================================
// Full pipeline / vocoder entries (Tier 1, builder-based)
// ===========================================================================

/// Re-verify `kokoro_full_pipeline` with Conservative mode → expect Sound.
#[test]
fn test_sound_kokoro_full_pipeline() {
    let (def, _) = build_kokoro_full_pipeline();
    let bindings = kokoro_full_pipeline_bindings();
    let input = uniform_bounds(&[D_MODEL, SEQ_LEN], 1.0);

    let result = verify_and_assert_with_config(
        &def,
        &bindings,
        &input,
        "kokoro_full_pipeline",
        &conservative_config(),
    );

    assert_eq!(
        result.verification.soundness_mode,
        VerificationSoundnessMode::Sound,
        "Conservative mode should produce Sound, got {:?}",
        result.verification.soundness_mode
    );
    let width = result.verification.output_width;
    assert!(
        width < VACUOUS_THRESHOLD,
        "Conservative bounds should be non-vacuous, width={width}"
    );
    eprintln!("kokoro_full_pipeline Conservative: width={width}, soundness=Sound");
}

/// Re-verify `kokoro_vocoder_pipeline` with Conservative mode → expect Sound.
#[test]
fn test_sound_kokoro_vocoder_pipeline() {
    let (def, _) = build_kokoro_vocoder_only_pipeline();
    let bindings = kokoro_vocoder_only_bindings();
    let enc_dim = 8;
    let input = uniform_bounds(&[enc_dim, SEQ_LEN], 1.0);

    let result = verify_and_assert_with_config(
        &def,
        &bindings,
        &input,
        "kokoro_vocoder_pipeline",
        &conservative_config(),
    );

    assert_eq!(
        result.verification.soundness_mode,
        VerificationSoundnessMode::Sound,
        "Conservative mode should produce Sound, got {:?}",
        result.verification.soundness_mode
    );
    let width = result.verification.output_width;
    assert!(
        width < VACUOUS_THRESHOLD,
        "Conservative bounds should be non-vacuous, width={width}"
    );
    eprintln!("kokoro_vocoder_pipeline Conservative: width={width}, soundness=Sound");
}

/// Re-verify `kokoro_full_pipeline_forward` (ForwardMode status key) with Conservative → Sound.
///
/// Same model as `kokoro_full_pipeline` but recorded under the ForwardMode status key.
#[test]
fn test_sound_kokoro_full_pipeline_forward() {
    let (def, _) = build_kokoro_full_pipeline();
    let bindings = kokoro_full_pipeline_bindings();
    let input = uniform_bounds(&[D_MODEL, SEQ_LEN], 1.0);

    let result = verify_and_assert_with_config(
        &def,
        &bindings,
        &input,
        "kokoro_full_pipeline_forward",
        &conservative_config(),
    );

    assert_eq!(
        result.verification.soundness_mode,
        VerificationSoundnessMode::Sound,
        "Conservative mode should produce Sound, got {:?}",
        result.verification.soundness_mode
    );
    let width = result.verification.output_width;
    assert!(
        width < VACUOUS_THRESHOLD,
        "Conservative bounds should be non-vacuous, width={width}"
    );
    eprintln!("kokoro_full_pipeline_forward Conservative: width={width}, soundness=Sound");
}

/// Re-verify `kokoro_vocoder_forward` (ForwardMode status key) with Conservative → Sound.
///
/// Same model as `kokoro_vocoder_pipeline` but recorded under the ForwardMode status key.
#[test]
fn test_sound_kokoro_vocoder_forward() {
    let (def, _) = build_kokoro_vocoder_only_pipeline();
    let bindings = kokoro_vocoder_only_bindings();
    let enc_dim = 8;
    let input = uniform_bounds(&[enc_dim, SEQ_LEN], 1.0);

    let result = verify_and_assert_with_config(
        &def,
        &bindings,
        &input,
        "kokoro_vocoder_forward",
        &conservative_config(),
    );

    assert_eq!(
        result.verification.soundness_mode,
        VerificationSoundnessMode::Sound,
        "Conservative mode should produce Sound, got {:?}",
        result.verification.soundness_mode
    );
    let width = result.verification.output_width;
    assert!(
        width < VACUOUS_THRESHOLD,
        "Conservative bounds should be non-vacuous, width={width}"
    );
    eprintln!("kokoro_vocoder_forward Conservative: width={width}, soundness=Sound");
}

// ===========================================================================
// Speaker pipeline (Tier 1)
// ===========================================================================

/// Re-verify `kokoro_tts_speaker_pipeline` with Conservative mode → expect Sound.
#[test]
fn test_sound_kokoro_tts_speaker_pipeline() {
    let (def, _) = build_tts_speaker_pipeline();
    let bindings = tts_speaker_bindings();
    let input = uniform_bounds(&[8, 4], 1.0);

    let result = verify_and_assert_with_config(
        &def,
        &bindings,
        &input,
        "kokoro_tts_speaker_pipeline",
        &conservative_config(),
    );

    assert_eq!(
        result.verification.soundness_mode,
        VerificationSoundnessMode::Sound,
        "Conservative mode should produce Sound, got {:?}",
        result.verification.soundness_mode
    );
    let width = result.verification.output_width;
    assert!(
        width < VACUOUS_THRESHOLD,
        "Conservative bounds should be non-vacuous, width={width}"
    );
    eprintln!("kokoro_tts_speaker_pipeline Conservative: width={width}, soundness=Sound");
}

// ===========================================================================
// Chained norm entries (Tier 2)
//
// These entries use TensorBlockBuilder → tensor_kernel_to_graph_with_norm_mode
// with Conservative mode. The existing tests in compose_chained_norm.rs use
// raw propagate_ibp and hardcode Heuristic recording. Re-verifying through
// verify_and_assert_with_config auto-detects Sound classification.
//
// Part of #3351 T3.1: Kokoro soundness improvement, Tier 2.
// ===========================================================================

const CHAINED_CHANNELS: usize = 4;
const CHAINED_TIME_LEN: usize = 16;
const CHAINED_KERNEL_SIZE: usize = 3;
const CHAINED_NORM_EPS: f32 = 1e-5;

/// Build a Kokoro-like Conv1d → ReLU → InstanceNorm chain (same as compose_chained_norm.rs).
fn build_kokoro_like_chain_for_reverify(
    num_blocks: usize,
) -> (nn_dsl::tensor_ir::TensorKernelDef, Vec<TensorParamBinding>) {
    let channels = CHAINED_CHANNELS;
    let time_len = CHAINED_TIME_LEN;
    let kernel_size = CHAINED_KERNEL_SIZE;
    let padding = kernel_size / 2;
    let shape = [channels, time_len];

    let mut b = TensorBlockBuilder::new("kokoro_like_chain");
    let data = b.add_input("data", &shape);
    let eps = b.add_input("eps", &[1]);

    let mut weight_ids = Vec::with_capacity(num_blocks);
    for i in 0..num_blocks {
        let w = b.add_input(&format!("weight_{i}"), &[channels, channels, kernel_size]);
        weight_ids.push(w);
    }

    let mut current = data;
    for &wid in &weight_ids {
        let conv = b.add_conv1d(current, wid, None, 1, padding, &shape);
        let relu = b.add_relu(conv, &shape);
        current = b.add_instance_norm(relu, eps, 1, None, None, &shape);
    }

    let def = b.build(current).expect("valid Kokoro-like chain");

    let weight_mag = 0.1 / (channels as f32).sqrt();
    let mut bindings = vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantScalar(CHAINED_NORM_EPS),
    ];
    for _ in 0..num_blocks {
        let w = ArrayD::from_elem(IxDyn(&[channels, channels, kernel_size]), weight_mag);
        bindings.push(TensorParamBinding::ConstantTensor(w));
    }

    (def, bindings)
}

/// Re-verify `kokoro_chained_norm_kokoro_n10` with Conservative mode → expect Sound.
///
/// The existing test in compose_chained_norm.rs produces tight bounds (width ~7.75)
/// with Conservative IBP but records with hardcoded Heuristic. This test goes through
/// verify_and_assert_with_config which auto-detects Sound.
#[test]
fn test_sound_kokoro_chained_norm_kokoro_n10() {
    let (def, bindings) = build_kokoro_like_chain_for_reverify(10);
    let input = uniform_bounds(&[CHAINED_CHANNELS, CHAINED_TIME_LEN], 1.0);

    let result = verify_and_assert_with_config(
        &def,
        &bindings,
        &input,
        "kokoro_chained_norm_kokoro_n10",
        &conservative_config(),
    );

    assert_eq!(
        result.verification.soundness_mode,
        VerificationSoundnessMode::Sound,
        "Conservative mode should produce Sound, got {:?}",
        result.verification.soundness_mode
    );
    let width = result.verification.output_width;
    assert!(
        width < VACUOUS_THRESHOLD,
        "Conservative bounds should be non-vacuous, width={width}"
    );
    eprintln!("kokoro_chained_norm_kokoro_n10 Conservative: width={width}, soundness=Sound");
}

/// Re-verify `kokoro_chained_norm_kokoro_n20` with Conservative mode → expect Sound.
#[test]
fn test_sound_kokoro_chained_norm_kokoro_n20() {
    let (def, bindings) = build_kokoro_like_chain_for_reverify(20);
    let input = uniform_bounds(&[CHAINED_CHANNELS, CHAINED_TIME_LEN], 1.0);

    let result = verify_and_assert_with_config(
        &def,
        &bindings,
        &input,
        "kokoro_chained_norm_kokoro_n20",
        &conservative_config(),
    );

    assert_eq!(
        result.verification.soundness_mode,
        VerificationSoundnessMode::Sound,
        "Conservative mode should produce Sound, got {:?}",
        result.verification.soundness_mode
    );
    let width = result.verification.output_width;
    assert!(
        width < VACUOUS_THRESHOLD,
        "Conservative bounds should be non-vacuous, width={width}"
    );
    eprintln!("kokoro_chained_norm_kokoro_n20 Conservative: width={width}, soundness=Sound");
}

/// Re-verify `kokoro_chained_norm_kokoro_n58` with Conservative mode → expect Sound.
///
/// N=58 matches Kokoro Generator production depth. Conservative IBP produces
/// depth-invariant tight bounds (~7.75 width) due to contractive Conv weights.
#[test]
fn test_sound_kokoro_chained_norm_kokoro_n58() {
    let (def, bindings) = build_kokoro_like_chain_for_reverify(58);
    let input = uniform_bounds(&[CHAINED_CHANNELS, CHAINED_TIME_LEN], 1.0);

    let result = verify_and_assert_with_config(
        &def,
        &bindings,
        &input,
        "kokoro_chained_norm_kokoro_n58",
        &conservative_config(),
    );

    assert_eq!(
        result.verification.soundness_mode,
        VerificationSoundnessMode::Sound,
        "Conservative mode should produce Sound, got {:?}",
        result.verification.soundness_mode
    );
    let width = result.verification.output_width;
    assert!(
        width < VACUOUS_THRESHOLD,
        "Conservative bounds should be non-vacuous, width={width}"
    );
    eprintln!("kokoro_chained_norm_kokoro_n58 Conservative: width={width}, soundness=Sound");
}

/// Re-verify `kokoro_chained_norm_crown_n10` with Conservative + forced CROWN.
///
/// This replaces the historical vacuous/heuristic CROWN status for N=10 with a
/// sound Conservative CROWN run. Threshold=0 forces escalation to CROWN.
#[test]
fn test_sound_kokoro_chained_norm_crown_n10() {
    let (def, bindings) = build_kokoro_like_chain_for_reverify(10);
    let input = uniform_bounds(&[CHAINED_CHANNELS, CHAINED_TIME_LEN], 1.0);

    let result = verify_and_assert_with_config(
        &def,
        &bindings,
        &input,
        "kokoro_chained_norm_crown_n10",
        &conservative_force_crown_config(),
    );

    // Forced escalation (threshold=0) runs the CROWN family. `run_escalation`
    // tries alpha-CROWN first and records `AlphaCrown` when it succeeds (the
    // common case) — a strictly-tighter-or-equal CROWN variant. Assert the
    // recorded method is tight (Crown/AlphaCrown/...) per `PropMethod::is_tight`'s
    // documented guidance to use it instead of `== PropMethod::Crown`.
    assert!(
        result.verification.method.is_tight(),
        "forced-CROWN config should record a tight CROWN-family method, got {:?}",
        result.verification.method
    );
    assert_eq!(
        result.verification.soundness_mode,
        VerificationSoundnessMode::Sound,
        "Conservative forced-CROWN should be Sound, got {:?}",
        result.verification.soundness_mode
    );
    let width = result.verification.output_width;
    assert!(
        width < VACUOUS_THRESHOLD,
        "Conservative CROWN bounds should be non-vacuous, width={width}"
    );
    eprintln!("kokoro_chained_norm_crown_n10 Conservative+CROWN: width={width}, soundness=Sound");
}

/// Re-verify `kokoro_chained_norm_crown_n58` with Conservative + forced CROWN.
///
/// Production-depth variant of the N=10 forced-CROWN re-verification.
#[test]
fn test_sound_kokoro_chained_norm_crown_n58() {
    let (def, bindings) = build_kokoro_like_chain_for_reverify(58);
    let input = uniform_bounds(&[CHAINED_CHANNELS, CHAINED_TIME_LEN], 1.0);

    let result = verify_and_assert_with_config(
        &def,
        &bindings,
        &input,
        "kokoro_chained_norm_crown_n58",
        &conservative_force_crown_config(),
    );

    // Forced escalation (threshold=0) runs the CROWN family. `run_escalation`
    // tries alpha-CROWN first and records `AlphaCrown` when it succeeds (the
    // common case) — a strictly-tighter-or-equal CROWN variant. Assert the
    // recorded method is tight (Crown/AlphaCrown/...) per `PropMethod::is_tight`'s
    // documented guidance to use it instead of `== PropMethod::Crown`.
    assert!(
        result.verification.method.is_tight(),
        "forced-CROWN config should record a tight CROWN-family method, got {:?}",
        result.verification.method
    );
    assert_eq!(
        result.verification.soundness_mode,
        VerificationSoundnessMode::Sound,
        "Conservative forced-CROWN should be Sound, got {:?}",
        result.verification.soundness_mode
    );
    let width = result.verification.output_width;
    assert!(
        width < VACUOUS_THRESHOLD,
        "Conservative CROWN bounds should be non-vacuous, width={width}"
    );
    eprintln!("kokoro_chained_norm_crown_n58 Conservative+CROWN: width={width}, soundness=Sound");
}

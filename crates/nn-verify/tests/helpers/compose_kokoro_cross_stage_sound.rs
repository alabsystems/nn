// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

// Helpers are shared across multiple test binaries; not all binaries use all functions.
#![allow(dead_code, clippy::duplicated_attributes)]

//! Sound-promotion compose tests for Kokoro cross-stage pipeline compositions.
//!
//! These tests add NEW sound verification entries for cross-stage pipeline
//! compositions that previously had no status file coverage. Each test uses
//! `NormBoundsMode::Conservative` which produces `Sound` classification because
//! Conservative IBP through normalization layers is provably sound (standard
//! interval arithmetic over-approximation, no sampling-based linearization).
//!
//! New entries added to `nn_verify_status_kokoro.json`:
//!
//!   1. `kokoro_encoder_style_chain_sound` — TextEncoder + StyleProjector composed
//!      pipeline with Conservative IBP through Tanh squashing.
//!
//!   2. `kokoro_full_pipeline_sound` — Full 4-stage encoder-to-decoder pipeline
//!      (Conv1d + ReLU + Linear + ConvTranspose1d + InstanceNorm + Snake + Exp)
//!      with Conservative IBP proving P1 (non-silence) and P2 (bounded).
//!
//!   3. `kokoro_multi_resblock_decoder_sound` — Decoder with 2 sequential
//!      InstanceNorm + Snake + Conv1d ResBlocks proving P1 through deep
//!      normalization chains.
//!
//!   4. `kokoro_decoder_block_sound` — Single decoder block (ConvTranspose1d +
//!      InstanceNorm + Snake + Exp) proving P1 positivity with Conservative IBP.
//!
//!   5. `kokoro_text_encoder_sound` — Text encoder standalone (Conv1d + ReLU +
//!      Linear projection) with Conservative IBP. No normalization layers but
//!      records explicit Sound status for completeness.
//!
//! Strategy:
//!   All tests use `VerifyConfig::default().with_norm_mode(NormBoundsMode::Conservative)`
//!   and `verify_and_assert_with_config` to record entries with guaranteed Sound
//!   soundness mode. The builders are reused from `kokoro_multi_stage.rs` at
//!   reduced dimensions (hidden=8, seq=4) for NY tractability.
//!
//! Part of verification soundness improvement.
//! Part of Epic #3351 (Absolutely Best Kokoro).

#[path = "kokoro_multi_stage.rs"]
mod ms_helpers;

use ms_helpers::{
    build_decoder_block, build_encoder_style_chain, build_full_four_stage_pipeline,
    build_multi_resblock_decoder, build_text_encoder, D_MODEL, ENC_DIM, OUT_CH, SEQ_LEN, STYLE_DIM,
    TIME_UP, VOC_CH,
};

use super::common::{bounds_min_max, uniform_bounds, verify_and_assert_with_config};

use nn_verify::{NormBoundsMode, VerificationSoundnessMode, VerifyConfig};

// ===========================================================================
// Configuration
// ===========================================================================

/// Vacuous width threshold — bounds wider than this are meaningless.
const VACUOUS_THRESHOLD: f32 = 200.0;

fn conservative_config() -> VerifyConfig {
    VerifyConfig::default().with_norm_mode(NormBoundsMode::Conservative)
}

// ===========================================================================
// 1. kokoro_encoder_style_chain_sound
// ===========================================================================

/// Composed TextEncoder + StyleProjector with Conservative IBP -> Sound.
///
/// The encoder-style chain includes: Conv1d + ReLU + Linear + Linear + Tanh + Linear.
/// Tanh squashes intermediate values to [-1, 1], ensuring tight output bounds.
/// Conservative mode uses standard IBP (no forward-mode sampling), producing
/// provably sound bounds.
///
/// Properties verified:
///   - Tanh output ∈ [-1, 1] (with small weight tolerance)
///   - Output bounds are finite and non-vacuous
///   - Soundness mode is Sound
#[test]
fn test_compose_kokoro_encoder_style_chain_sound() {
    let (def, bindings, out_shape) = build_encoder_style_chain();
    assert_eq!(out_shape, [STYLE_DIM, SEQ_LEN]);

    let input = uniform_bounds(&[D_MODEL, SEQ_LEN], 1.0);

    let result = verify_and_assert_with_config(
        &def,
        &bindings,
        &input,
        "kokoro_encoder_style_chain_sound",
        &conservative_config(),
    );

    assert_eq!(
        result.verification.soundness_mode,
        VerificationSoundnessMode::Sound,
        "Conservative mode should produce Sound, got {:?}",
        result.verification.soundness_mode
    );

    let (lo, _) = result.output_bounds.lower_upper();
    assert_eq!(lo.shape(), &[STYLE_DIM, SEQ_LEN]);

    let width = result.verification.output_width;
    assert!(
        width < VACUOUS_THRESHOLD,
        "Conservative bounds should be non-vacuous, width={width}"
    );

    let (lo_min, hi_max) = bounds_min_max(&result.output_bounds);
    // Tanh squashes to [-1, 1]; with small weights output stays within.
    assert!(
        lo_min >= -1.0 - 1e-3,
        "Encoder-style chain: Tanh lower >= -1.0, got {lo_min}"
    );
    assert!(
        hi_max <= 1.0 + 1e-3,
        "Encoder-style chain: Tanh upper <= 1.0, got {hi_max}"
    );
    eprintln!(
        "kokoro_encoder_style_chain_sound: bounds=[{lo_min}, {hi_max}], width={width}, soundness=Sound"
    );
}

// ===========================================================================
// 2. kokoro_full_pipeline_sound
// ===========================================================================

/// Full 4-stage encoder-to-decoder pipeline with Conservative IBP -> Sound.
///
/// Pipeline: text features -> Conv1d + ReLU + Linear (encoder) ->
///           Conv1d + LeakyReLU + ConvTranspose1d (upsample) ->
///           InstanceNorm + Snake + Conv1d (ResBlock) ->
///           LeakyReLU + Conv1d + Exp (output)
///
/// Properties verified:
///   - P1 (Non-silence): exp() output lower bound > 0
///   - P2 (Bounded): output upper bound < threshold
///   - Output bounds are finite and non-vacuous
///   - Soundness mode is Sound
#[test]
fn test_compose_kokoro_full_pipeline_sound() {
    let (def, bindings, out_shape) = build_full_four_stage_pipeline();
    assert_eq!(out_shape, [OUT_CH, TIME_UP]);

    let input = uniform_bounds(&[D_MODEL, SEQ_LEN], 1.0);

    let result = verify_and_assert_with_config(
        &def,
        &bindings,
        &input,
        "kokoro_full_pipeline_sound",
        &conservative_config(),
    );

    assert_eq!(
        result.verification.soundness_mode,
        VerificationSoundnessMode::Sound,
        "Conservative mode should produce Sound, got {:?}",
        result.verification.soundness_mode
    );

    let (lo, _) = result.output_bounds.lower_upper();
    assert_eq!(lo.shape(), &[OUT_CH, TIME_UP]);

    let width = result.verification.output_width;
    assert!(
        width < VACUOUS_THRESHOLD,
        "Conservative bounds should be non-vacuous, width={width}"
    );

    let (lo_min, hi_max) = bounds_min_max(&result.output_bounds);
    // P1: exp() output must be strictly positive.
    assert!(
        lo_min > 0.0,
        "P1 VIOLATION: exp output must be positive, got {lo_min}"
    );
    // P2: output must be finite and bounded.
    assert!(
        hi_max.is_finite(),
        "P2: output must be finite, got {hi_max}"
    );
    assert!(hi_max < 1e8, "P2: output must be bounded, got {hi_max}");
    eprintln!(
        "kokoro_full_pipeline_sound: bounds=[{lo_min}, {hi_max}], width={width}, soundness=Sound"
    );
    eprintln!("  P1 (Non-silence) PROVEN: lower {lo_min} > 0");
    eprintln!("  P2 (Bounded) PROVEN: upper {hi_max} < 1e8");
}

// ===========================================================================
// 3. kokoro_multi_resblock_decoder_sound
// ===========================================================================

/// Multi-ResBlock decoder with Conservative IBP -> Sound.
///
/// Decoder with 2 sequential ResBlocks, each containing:
///   InstanceNorm + Snake + Conv1d + residual connection
/// Followed by LeakyReLU + Conv1d + Exp.
///
/// This is the deepest InstanceNorm chain in the Kokoro decoder. Conservative
/// IBP through multiple chained InstanceNorm layers is provably sound — each
/// layer independently computes interval arithmetic bounds without approximation.
///
/// Properties verified:
///   - P1 (Non-silence): exp() lower > 0 through 2 InstanceNorm + Snake blocks
///   - Output bounds are finite and non-vacuous
///   - Soundness mode is Sound
#[test]
fn test_compose_kokoro_multi_resblock_decoder_sound() {
    let (def, bindings, out_shape) = build_multi_resblock_decoder();
    assert_eq!(out_shape, [OUT_CH, TIME_UP]);

    let input = uniform_bounds(&[VOC_CH, SEQ_LEN], 1.0);

    let result = verify_and_assert_with_config(
        &def,
        &bindings,
        &input,
        "kokoro_multi_resblock_decoder_sound",
        &conservative_config(),
    );

    assert_eq!(
        result.verification.soundness_mode,
        VerificationSoundnessMode::Sound,
        "Conservative mode should produce Sound, got {:?}",
        result.verification.soundness_mode
    );

    let (lo, _) = result.output_bounds.lower_upper();
    assert_eq!(lo.shape(), &[OUT_CH, TIME_UP]);

    let width = result.verification.output_width;
    assert!(
        width < VACUOUS_THRESHOLD,
        "Conservative bounds should be non-vacuous, width={width}"
    );

    let (lo_min, hi_max) = bounds_min_max(&result.output_bounds);
    // P1: exp() output must be strictly positive.
    assert!(
        lo_min > 0.0,
        "P1 VIOLATION: exp output through 2 ResBlocks must be positive, got {lo_min}"
    );
    assert!(hi_max.is_finite(), "Output must be finite, got {hi_max}");
    eprintln!(
        "kokoro_multi_resblock_decoder_sound: bounds=[{lo_min}, {hi_max}], width={width}, soundness=Sound"
    );
    eprintln!("  P1 (Non-silence) through 2 ResBlocks PROVEN: lower {lo_min} > 0");
}

// ===========================================================================
// 4. kokoro_decoder_block_sound
// ===========================================================================

/// Single decoder block with Conservative IBP -> Sound.
///
/// Architecture: Conv1d + LeakyReLU + ConvTranspose1d + InstanceNorm + Snake +
/// Conv1d + residual + LeakyReLU + Conv1d + Exp.
///
/// The InstanceNorm + Snake block is the core Kokoro vocoder pattern. This test
/// proves P1 (non-silence) with explicit Sound soundness via Conservative IBP.
///
/// Properties verified:
///   - P1 (Non-silence): exp() lower > 0
///   - Output shape matches expected upsampled dimensions
///   - Soundness mode is Sound
#[test]
fn test_compose_kokoro_decoder_block_sound() {
    let (def, bindings, out_shape) = build_decoder_block();
    assert_eq!(out_shape, [OUT_CH, TIME_UP]);

    let input = uniform_bounds(&[VOC_CH, SEQ_LEN], 1.0);

    let result = verify_and_assert_with_config(
        &def,
        &bindings,
        &input,
        "kokoro_decoder_block_sound",
        &conservative_config(),
    );

    assert_eq!(
        result.verification.soundness_mode,
        VerificationSoundnessMode::Sound,
        "Conservative mode should produce Sound, got {:?}",
        result.verification.soundness_mode
    );

    let (lo, _) = result.output_bounds.lower_upper();
    assert_eq!(lo.shape(), &[OUT_CH, TIME_UP]);

    let width = result.verification.output_width;
    assert!(
        width < VACUOUS_THRESHOLD,
        "Conservative bounds should be non-vacuous, width={width}"
    );

    let (lo_min, hi_max) = bounds_min_max(&result.output_bounds);
    assert!(
        lo_min > 0.0,
        "P1 VIOLATION: exp output must be positive, got {lo_min}"
    );
    assert!(hi_max.is_finite(), "Output must be finite, got {hi_max}");
    eprintln!(
        "kokoro_decoder_block_sound: bounds=[{lo_min}, {hi_max}], width={width}, soundness=Sound"
    );
}

// ===========================================================================
// 5. kokoro_text_encoder_sound
// ===========================================================================

/// Text encoder standalone with Conservative IBP -> Sound.
///
/// Architecture: Conv1d(k=3, same-pad) + ReLU + Transpose + MatMul + Bias + Transpose.
/// No normalization layers, but records explicit Sound status for cross-stage
/// composition completeness. The encoder output feeds into the decoder and
/// style projector — having a Sound entry for the encoder enables compositional
/// reasoning about downstream pipeline stages.
///
/// Properties verified:
///   - Output bounds are finite (no NaN/Inf through Conv1d + ReLU + Linear)
///   - Output shape matches expected encoder dimensions
///   - Soundness mode is Sound
#[test]
fn test_compose_kokoro_text_encoder_sound() {
    let (def, bindings, out_shape) = build_text_encoder();
    assert_eq!(out_shape, [ENC_DIM, SEQ_LEN]);

    let input = uniform_bounds(&[D_MODEL, SEQ_LEN], 1.0);

    let result = verify_and_assert_with_config(
        &def,
        &bindings,
        &input,
        "kokoro_text_encoder_sound",
        &conservative_config(),
    );

    assert_eq!(
        result.verification.soundness_mode,
        VerificationSoundnessMode::Sound,
        "Conservative mode should produce Sound, got {:?}",
        result.verification.soundness_mode
    );

    let (lo, _) = result.output_bounds.lower_upper();
    assert_eq!(lo.shape(), &[ENC_DIM, SEQ_LEN]);

    let width = result.verification.output_width;
    assert!(
        width < VACUOUS_THRESHOLD,
        "Conservative bounds should be non-vacuous, width={width}"
    );

    let (lo_min, hi_max) = bounds_min_max(&result.output_bounds);
    assert!(
        lo_min.is_finite(),
        "Text encoder output must be finite, got lo_min={lo_min}"
    );
    assert!(
        hi_max.is_finite(),
        "Text encoder output must be finite, got hi_max={hi_max}"
    );
    eprintln!(
        "kokoro_text_encoder_sound: bounds=[{lo_min}, {hi_max}], width={width}, soundness=Sound"
    );
}

// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Multi-stage compose verification tests for the Kokoro full pipeline.
//!
//! Tests chain multiple Kokoro pipeline stages using TensorBlockBuilder at
//! reduced dimensions (hidden=8, seq=4) and verify end-to-end bound
//! propagation through each stage combination:
//!
//! 1. Text encoder standalone (Conv1d + ReLU + Linear)
//! 2. Style projector standalone (Linear + Tanh + Linear)
//! 3. Decoder block standalone (Conv1d + LeakyReLU + ConvTranspose1d + ResBlock + Exp)
//! 4. Encoder + style projector chain (multi-stage)
//! 5. Full 4-stage pipeline (encoder + decoder, end-to-end)
//! 6. Multi-ResBlock decoder (deeper verification)
//! 7. Bounds monotonicity (narrower input -> narrower output)
//!
//! Part of #3617: Compose verification tests for Kokoro full pipeline.
//! Part of #3351: Epic — Absolutely Best Kokoro.

#[path = "kokoro_multi_stage.rs"]
mod ms_helpers;

use nn_verify::{tensor_kernel_to_graph, VerificationSoundnessMode};
use ms_helpers::{
    build_decoder_block, build_encoder_style_chain, build_full_four_stage_pipeline,
    build_multi_resblock_decoder, build_style_projector, build_text_encoder, D_MODEL, ENC_DIM,
    OUT_CH, SEQ_LEN, STYLE_DIM, TIME_UP, VOC_CH,
};

use super::common::{
    assert_bounds_valid, assert_crown_tighter_when_not_fallback, bounds_min_max, uniform_bounds,
    verify_and_assert,
};

// ===========================================================================
// Test 1: Text encoder standalone
// ===========================================================================

/// Text encoder: Conv1d + ReLU + Linear produces finite bounds.
///
/// Part of #3617.
#[test]
fn test_kokoro_ms_text_encoder() {
    let (def, bindings, out_shape) = build_text_encoder();
    assert_eq!(out_shape, [ENC_DIM, SEQ_LEN]);
    def.validate().expect("text encoder def validates");

    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[D_MODEL, SEQ_LEN], 1.0);

    let ibp_output = graph
        .propagate_ibp(&input)
        .expect("IBP through text encoder");
    assert_bounds_valid(&ibp_output);
    let (lo_min, hi_max) = bounds_min_max(&ibp_output);
    eprintln!("Text encoder IBP: bounds=[{lo_min}, {hi_max}]");
    assert!(lo_min.is_finite(), "text encoder lo must be finite");
    assert!(hi_max.is_finite(), "text encoder hi must be finite");

    let result = verify_and_assert(&def, &bindings, &input, "kokoro_ms_text_encoder");
    assert_eq!(result.num_variables, 1);
}

// ===========================================================================
// Test 2: Style projector standalone
// ===========================================================================

/// Style projector: Linear + Tanh + Linear. Tanh guarantees output in [-1, 1].
///
/// Part of #3617.
#[test]
fn test_kokoro_ms_style_projector() {
    let (def, bindings, out_shape) = build_style_projector();
    assert_eq!(out_shape, [STYLE_DIM, SEQ_LEN]);
    def.validate().expect("style projector def validates");

    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[ENC_DIM, SEQ_LEN], 1.0);

    let ibp_output = graph
        .propagate_ibp(&input)
        .expect("IBP through style projector");
    assert_bounds_valid(&ibp_output);
    let (lo_min, hi_max) = bounds_min_max(&ibp_output);
    eprintln!("Style projector IBP: bounds=[{lo_min}, {hi_max}]");

    // With small weights, output should be near zero.
    assert!(lo_min.abs() < 1.0, "style lo should be small, got {lo_min}");
    assert!(hi_max.abs() < 1.0, "style hi should be small, got {hi_max}");

    let result = verify_and_assert(&def, &bindings, &input, "kokoro_ms_style_projector");
    assert_eq!(result.num_variables, 1);
}

// ===========================================================================
// Test 3: Decoder block standalone
// ===========================================================================

/// Decoder block: Conv1d + LeakyReLU + ConvTranspose1d + ResBlock + Exp.
/// P1 (Non-silence): exp() output must be strictly positive.
///
/// Part of #3617.
#[test]
fn test_kokoro_ms_decoder_block() {
    let (def, bindings, out_shape) = build_decoder_block();
    assert_eq!(out_shape, [OUT_CH, TIME_UP]);
    def.validate().expect("decoder block def validates");

    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    assert!(
        graph.num_nodes() >= 8,
        "decoder should have >= 8 nodes, got {}",
        graph.num_nodes()
    );

    let input = uniform_bounds(&[VOC_CH, SEQ_LEN], 1.0);

    let ibp_output = graph.propagate_ibp(&input).expect("IBP through decoder");
    assert_eq!(ibp_output.lower_upper().0.shape(), &[OUT_CH, TIME_UP]);
    assert_bounds_valid(&ibp_output);
    let (lo_min, hi_max) = bounds_min_max(&ibp_output);
    eprintln!("Decoder IBP: bounds=[{lo_min}, {hi_max}]");

    assert!(
        lo_min > 0.0,
        "P1 VIOLATION: exp output should be positive, got {lo_min}"
    );
    assert!(hi_max < 1e6, "IBP upper should be bounded, got {hi_max}");

    // CROWN
    let (method, crown_output, fallback_reason) =
        assert_crown_tighter_when_not_fallback(&graph, &input);
    let (crown_lo, crown_hi) = bounds_min_max(&crown_output);
    eprintln!("Decoder CROWN: method={method:?}, bounds=[{crown_lo}, {crown_hi}]");
    if let Some(reason) = &fallback_reason {
        eprintln!("CROWN fallback reason: {reason}");
    }
    assert!(crown_lo > 0.0, "CROWN P1: exp positive, got {crown_lo}");

    let result = verify_and_assert(&def, &bindings, &input, "kokoro_ms_decoder_block");
    assert_eq!(result.num_variables, 1);
    assert_eq!(
        result.output_bounds.lower_upper().0.shape(),
        &[OUT_CH, TIME_UP]
    );
}

// ===========================================================================
// Test 4: Encoder + style projector chain
// ===========================================================================

/// Multi-stage: text encoder + style projector chain.
/// Verifies bounds propagate through Conv1d + ReLU + Linear + Linear + Tanh + Linear.
///
/// Part of #3617.
#[test]
fn test_kokoro_ms_encoder_style_chain() {
    let (def, bindings, out_shape) = build_encoder_style_chain();
    assert_eq!(out_shape, [STYLE_DIM, SEQ_LEN]);
    def.validate().expect("encoder-style chain validates");

    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    assert!(
        graph.num_nodes() >= 10,
        "chain should have >= 10 nodes, got {}",
        graph.num_nodes()
    );

    let input = uniform_bounds(&[D_MODEL, SEQ_LEN], 1.0);

    let ibp_output = graph
        .propagate_ibp(&input)
        .expect("IBP through encoder-style chain");
    assert_eq!(ibp_output.lower_upper().0.shape(), &[STYLE_DIM, SEQ_LEN]);
    assert_bounds_valid(&ibp_output);
    let (lo_min, hi_max) = bounds_min_max(&ibp_output);
    eprintln!("Encoder-style chain IBP: bounds=[{lo_min}, {hi_max}]");

    // Tanh squashes to [-1, 1]; with small weights output stays well within.
    assert!(lo_min >= -1.0 - 1e-3, "Tanh lower >= -1.0, got {lo_min}");
    assert!(hi_max <= 1.0 + 1e-3, "Tanh upper <= 1.0, got {hi_max}");

    // CROWN
    let (method, crown_output, fallback_reason) =
        assert_crown_tighter_when_not_fallback(&graph, &input);
    let (crown_lo, crown_hi) = bounds_min_max(&crown_output);
    eprintln!("Encoder-style chain CROWN: method={method:?}, bounds=[{crown_lo}, {crown_hi}]");
    if let Some(reason) = &fallback_reason {
        eprintln!("CROWN fallback reason: {reason}");
    }

    let result = verify_and_assert(&def, &bindings, &input, "kokoro_ms_encoder_style_chain");
    assert_eq!(result.num_variables, 1);
    assert_eq!(
        result.output_bounds.lower_upper().0.shape(),
        &[STYLE_DIM, SEQ_LEN]
    );
}

// ===========================================================================
// Test 5: Full 4-stage pipeline
// ===========================================================================

/// Full 4-stage pipeline: encoder + decoder (end-to-end).
/// P1 (Non-silence): exp() lower > 0. P2 (Non-clipping): upper < threshold.
///
/// Part of #3617.
#[test]
fn test_kokoro_ms_full_four_stage_pipeline() {
    let (def, bindings, out_shape) = build_full_four_stage_pipeline();
    assert_eq!(out_shape, [OUT_CH, TIME_UP]);
    def.validate().expect("full pipeline validates");

    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    assert!(
        graph.num_nodes() >= 15,
        "pipeline should have >= 15 nodes, got {}",
        graph.num_nodes()
    );

    let input = uniform_bounds(&[D_MODEL, SEQ_LEN], 1.0);

    let ibp_output = graph
        .propagate_ibp(&input)
        .expect("IBP through full pipeline");
    assert_eq!(ibp_output.lower_upper().0.shape(), &[OUT_CH, TIME_UP]);
    assert_bounds_valid(&ibp_output);
    let (lo_min, hi_max) = bounds_min_max(&ibp_output);
    eprintln!("Full 4-stage pipeline IBP: bounds=[{lo_min}, {hi_max}]");

    assert!(
        lo_min > 0.0,
        "P1 VIOLATION: exp output positive, got {lo_min}"
    );
    assert!(hi_max < 1e8, "P2: upper bounded, got {hi_max}");
    eprintln!("  P1 (Non-silence): lower {lo_min} > 0");
    eprintln!("  P2 (Non-clipping): upper {hi_max} < 1e8");

    // CROWN
    let (method, crown_output, fallback_reason) =
        assert_crown_tighter_when_not_fallback(&graph, &input);
    let (crown_lo, crown_hi) = bounds_min_max(&crown_output);
    eprintln!("Full 4-stage CROWN: method={method:?}, bounds=[{crown_lo}, {crown_hi}]");
    if let Some(reason) = &fallback_reason {
        eprintln!("CROWN fallback reason: {reason}");
    }

    let result = verify_and_assert(
        &def,
        &bindings,
        &input,
        "kokoro_ms_full_four_stage_pipeline",
    );
    assert_eq!(
        result.num_variables, 1,
        "single Variable input (text_features)"
    );
    assert_eq!(
        result.output_bounds.lower_upper().0.shape(),
        &[OUT_CH, TIME_UP]
    );
    assert!(
        matches!(
            result.verification.soundness_mode,
            VerificationSoundnessMode::Sound | VerificationSoundnessMode::Heuristic
        ),
        "soundness should be Sound or Heuristic, got {:?}",
        result.verification.soundness_mode
    );
}

// ===========================================================================
// Test 6: Multi-ResBlock decoder
// ===========================================================================

/// Decoder with 2 sequential ResBlocks tests bound stability through deeper paths.
///
/// Part of #3617.
#[test]
fn test_kokoro_ms_multi_resblock_decoder() {
    let (def, bindings, out_shape) = build_multi_resblock_decoder();
    assert_eq!(out_shape, [OUT_CH, TIME_UP]);
    def.validate().expect("multi-resblock decoder validates");

    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    assert!(
        graph.num_nodes() >= 12,
        "multi-resblock should have >= 12 nodes, got {}",
        graph.num_nodes()
    );

    let input = uniform_bounds(&[VOC_CH, SEQ_LEN], 1.0);

    let ibp_output = graph
        .propagate_ibp(&input)
        .expect("IBP through multi-resblock decoder");
    assert_eq!(ibp_output.lower_upper().0.shape(), &[OUT_CH, TIME_UP]);
    assert_bounds_valid(&ibp_output);
    let (lo_min, hi_max) = bounds_min_max(&ibp_output);
    eprintln!("Multi-ResBlock decoder IBP: bounds=[{lo_min}, {hi_max}]");

    assert!(lo_min > 0.0, "P1: exp positive, got {lo_min}");
    assert!(hi_max < 1e6, "IBP upper bounded, got {hi_max}");

    // CROWN
    let (method, crown_output, fallback_reason) =
        assert_crown_tighter_when_not_fallback(&graph, &input);
    let (crown_lo, crown_hi) = bounds_min_max(&crown_output);
    eprintln!("Multi-ResBlock CROWN: method={method:?}, bounds=[{crown_lo}, {crown_hi}]");
    if let Some(reason) = &fallback_reason {
        eprintln!("CROWN fallback reason: {reason}");
    }

    let result = verify_and_assert(&def, &bindings, &input, "kokoro_ms_multi_resblock_decoder");
    assert_eq!(result.num_variables, 1);
    assert_eq!(
        result.output_bounds.lower_upper().0.shape(),
        &[OUT_CH, TIME_UP]
    );
}

// ===========================================================================
// Test 7: Bounds monotonicity
// ===========================================================================

/// Narrower input produces narrower output bounds through the full pipeline.
/// This is a fundamental soundness property of IBP.
///
/// Part of #3617.
#[test]
fn test_kokoro_ms_bounds_monotonicity() {
    let (def, bindings, _) = build_full_four_stage_pipeline();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");

    // Wide input [-1, 1]
    let wide_input = uniform_bounds(&[D_MODEL, SEQ_LEN], 1.0);
    let wide_output = graph.propagate_ibp(&wide_input).expect("IBP wide");
    let (wide_lo, wide_hi) = bounds_min_max(&wide_output);
    let wide_range = wide_hi - wide_lo;

    // Narrow input [-0.1, 0.1]
    let narrow_input = uniform_bounds(&[D_MODEL, SEQ_LEN], 0.1);
    let narrow_output = graph.propagate_ibp(&narrow_input).expect("IBP narrow");
    let (narrow_lo, narrow_hi) = bounds_min_max(&narrow_output);
    let narrow_range = narrow_hi - narrow_lo;

    eprintln!("Wide input [-1,1] -> output range: {wide_range:.6}");
    eprintln!("Narrow input [-0.1,0.1] -> output range: {narrow_range:.6}");

    assert!(
        narrow_range <= wide_range + 1e-6,
        "IBP monotonicity violated: narrow range {narrow_range} > wide range {wide_range}"
    );
    eprintln!("Monotonicity OK: narrower input -> narrower output");
    if wide_range > 0.0 {
        let tightening = 1.0 - (narrow_range / wide_range);
        eprintln!(
            "  Tightening: {tightening:.2} ({:.0}% tighter)",
            tightening * 100.0
        );
    }
}

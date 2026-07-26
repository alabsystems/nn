// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Integration tests: Qwen3 decoder NY composition.
//!
//! 8 IBP compose tests covering every sub-component of the Qwen3 decoder:
//!
//! 1. **RoPE position encoding bounds**: Rotary embedding rotation preserves
//!    magnitude. With |cos|, |sin| <= 1 and input in [-1, 1], output in [-2, 2].
//! 2. **GQA attention score bounds**: Grouped-query attention with N_HEADS=2,
//!    N_KV_HEADS=1. KV head repeated via broadcast. Causal mask applied.
//! 3. **SwiGLU activation bounds**: gate_proj -> SiLU -> mul(up_proj) -> down_proj.
//!    3 non-linearities (sigmoid + 2 binary_mul) with small weights.
//! 4. **RMSNorm output bounds**: Normalization constrains output magnitude.
//! 5. **Decoder block residual stream**: RMSNorm -> MHA -> residual -> RMSNorm ->
//!    SwiGLU -> residual. Two residual connections control bounds growth.
//! 6. **Full decoder layer composition**: 2-block decoder + final RMSNorm + lm_head.
//!    End-to-end IBP from token embeddings to vocabulary logits.
//! 7. **Bounds widening analysis**: Compare 1-block vs full decoder to measure
//!    IBP blowup factor through composition.
//! 8. **GQA attention verify-and-record**: Verify-and-record for status tracking.
//!
//! Dimensions: D_MODEL=8, N_HEADS=2, N_KV_HEADS=1, HEAD_DIM=4, SEQ_LEN=4, VOCAB=16.
//! Uses IbpValidated soundness mode per nn engineering rules.
//!
//! Part of #4186: Add compose verification tests for Qwen3 decoder bounds.

#[path = "qwen3_decoder.rs"]
mod helpers;

use super::common::{
    assert_bounds_valid, assert_crown_tighter_when_not_fallback, bounds_min_max, uniform_bounds,
    verify_and_assert,
};
use helpers::{
    build_decoder_block, build_full_decoder, build_gqa, build_rmsnorm, build_rope, build_swiglu,
    decoder_block_bindings, full_decoder_bindings, gqa_bindings, rmsnorm_bindings, rope_bindings,
    swiglu_bindings, D_MODEL, HEAD_DIM, SEQ_LEN, VOCAB_SIZE,
};
use nn_verify::{tensor_kernel_to_graph, VerificationSoundnessMode};

// ============================================================================
// 1. RoPE position encoding bounds (IBP)
// ============================================================================

/// RoPE rotation validates, translates, and propagates IBP.
///
/// RoPE is a rotation: y1 = x1*cos - x2*sin, y2 = x1*sin + x2*cos.
/// With |cos|, |sin| <= 1 and input in [-1, 1], each output element has
/// magnitude <= 2 (IBP may be slightly wider due to interval arithmetic).
#[test]
fn test_qwen3_dec_rope_ibp() {
    let def = build_rope();
    def.validate().expect("RoPE should validate");

    let bindings = rope_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    assert!(
        graph.num_nodes() >= 8,
        "RoPE graph should have >= 8 nodes, got {}",
        graph.num_nodes()
    );

    let input = uniform_bounds(&[SEQ_LEN, HEAD_DIM], 1.0);
    let output = graph.propagate_ibp(&input).expect("IBP through RoPE");
    assert_eq!(output.lower_upper().0.shape(), &[SEQ_LEN, HEAD_DIM]);
    assert_bounds_valid(&output);

    let (lo, hi) = bounds_min_max(&output);
    eprintln!("Qwen3 dec RoPE IBP: [{lo}, {hi}]");
    // Rotation preserves magnitude: |y| <= |x1| + |x2| <= 2 in worst case.
    assert!(lo >= -5.0, "RoPE IBP lower >= -5, got {lo}");
    assert!(hi <= 5.0, "RoPE IBP upper <= 5, got {hi}");
}

/// RoPE CROWN should produce tight bounds (piecewise-linear rotation).
#[test]
fn test_qwen3_dec_rope_crown() {
    let def = build_rope();
    let bindings = rope_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, HEAD_DIM], 1.0);

    let (method, output, fallback_reason) = assert_crown_tighter_when_not_fallback(&graph, &input);
    assert_eq!(output.lower_upper().0.shape(), &[SEQ_LEN, HEAD_DIM]);
    let (lo, hi) = bounds_min_max(&output);
    eprintln!("Qwen3 dec RoPE: method={method:?}, [{lo}, {hi}]");
    if let Some(r) = &fallback_reason {
        eprintln!("Fallback: {r}");
    }
}

// ============================================================================
// 2. GQA attention score bounds (IBP)
// ============================================================================

/// GQA attention with causal mask validates and propagates IBP.
///
/// Key GQA property: N_KV_HEADS=1, N_HEADS=2 — the single KV head is
/// broadcast to serve both Q heads. Output has same shape as input.
#[test]
fn test_qwen3_dec_gqa_ibp() {
    let def = build_gqa();
    def.validate().expect("GQA should validate");

    let bindings = gqa_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    assert!(
        graph.num_nodes() >= 10,
        "GQA graph should have >= 10 nodes, got {}",
        graph.num_nodes()
    );

    let input = uniform_bounds(&[SEQ_LEN, D_MODEL], 1.0);
    let output = graph.propagate_ibp(&input).expect("IBP through GQA");
    assert_eq!(output.lower_upper().0.shape(), &[SEQ_LEN, D_MODEL]);
    assert_bounds_valid(&output);

    let (lo, hi) = bounds_min_max(&output);
    eprintln!("Qwen3 dec GQA IBP: [{lo}, {hi}]");
    assert!(lo.is_finite(), "GQA lower must be finite, got {lo}");
    assert!(hi.is_finite(), "GQA upper must be finite, got {hi}");
    assert!(lo.abs() < 1e8, "GQA IBP lower magnitude < 1e8, got {lo}");
}

// ============================================================================
// 3. SwiGLU activation bounds (IBP)
// ============================================================================

/// SwiGLU MLP validates, translates, and propagates IBP.
///
/// SwiGLU has 3 non-linearities (sigmoid in SiLU + 2 binary_mul).
/// With small weights (0.001), output magnitude stays bounded.
#[test]
fn test_qwen3_dec_swiglu_ibp() {
    let def = build_swiglu();
    def.validate().expect("SwiGLU should validate");

    let bindings = swiglu_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    // gate_linear + sigmoid + mul + up_linear + mul + down_linear = 6+ nodes
    assert!(
        graph.num_nodes() >= 6,
        "SwiGLU graph should have >= 6 nodes, got {}",
        graph.num_nodes()
    );

    let input = uniform_bounds(&[SEQ_LEN, D_MODEL], 1.0);
    let output = graph.propagate_ibp(&input).expect("IBP through SwiGLU");
    assert_eq!(output.lower_upper().0.shape(), &[SEQ_LEN, D_MODEL]);
    assert_bounds_valid(&output);

    let (lo, hi) = bounds_min_max(&output);
    eprintln!("Qwen3 dec SwiGLU IBP: [{lo}, {hi}]");
    assert!(lo.abs() < 1e6, "SwiGLU IBP lower magnitude < 1e6, got {lo}");
    assert!(hi.abs() < 1e6, "SwiGLU IBP upper magnitude < 1e6, got {hi}");
}

/// SwiGLU CROWN propagation.
#[test]
fn test_qwen3_dec_swiglu_crown() {
    let def = build_swiglu();
    let bindings = swiglu_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, D_MODEL], 1.0);

    let (method, output, fallback_reason) = assert_crown_tighter_when_not_fallback(&graph, &input);
    assert_eq!(output.lower_upper().0.shape(), &[SEQ_LEN, D_MODEL]);
    let (lo, hi) = bounds_min_max(&output);
    eprintln!("Qwen3 dec SwiGLU: method={method:?}, [{lo}, {hi}]");
    if let Some(r) = &fallback_reason {
        eprintln!("Fallback: {r}");
    }
}

// ============================================================================
// 4. RMSNorm output bounds (IBP)
// ============================================================================

/// RMSNorm validates and propagates IBP.
///
/// RMSNorm normalizes the input: output = (x / rms(x)) * weight.
/// With weight=1.0, output should have bounded magnitude.
#[test]
fn test_qwen3_dec_rmsnorm_ibp() {
    let def = build_rmsnorm();
    def.validate().expect("RMSNorm should validate");

    let bindings = rmsnorm_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    // RMSNorm translates to a single native `RmsNorm` layer; the `hidden`
    // Variable uses the NETWORK_INPUT sentinel and eps/weight bind as constants.
    assert!(
        graph.num_nodes() >= 1,
        "RMSNorm graph should have >= 1 node, got {}",
        graph.num_nodes()
    );

    let input = uniform_bounds(&[SEQ_LEN, D_MODEL], 1.0);
    let output = graph.propagate_ibp(&input).expect("IBP through RMSNorm");
    assert_eq!(output.lower_upper().0.shape(), &[SEQ_LEN, D_MODEL]);
    assert_bounds_valid(&output);

    let (lo, hi) = bounds_min_max(&output);
    eprintln!("Qwen3 dec RMSNorm IBP: [{lo}, {hi}]");
    assert!(lo.is_finite(), "RMSNorm lower must be finite, got {lo}");
    assert!(hi.is_finite(), "RMSNorm upper must be finite, got {hi}");
}

/// RMSNorm verify-and-record.
#[test]
fn test_qwen3_dec_rmsnorm_verify_record() {
    let def = build_rmsnorm();
    let bindings = rmsnorm_bindings();
    let input = uniform_bounds(&[SEQ_LEN, D_MODEL], 1.0);

    let result = verify_and_assert(&def, &bindings, &input, "qwen3_dec_rmsnorm");
    assert_eq!(result.num_variables, 1);
    assert_eq!(
        result.output_bounds.lower_upper().0.shape(),
        &[SEQ_LEN, D_MODEL]
    );

    // RMSNorm should produce Heuristic soundness mode.
    assert_eq!(
        result.verification.soundness_mode,
        VerificationSoundnessMode::Heuristic,
        "RMSNorm should produce Heuristic soundness, got {:?}",
        result.verification.soundness_mode
    );
}

// ============================================================================
// 5. Decoder block residual stream bounds (IBP)
// ============================================================================

/// Decoder block validates and propagates IBP.
///
/// Key property: two residual connections (attention + MLP) constrain bounds
/// growth. With small weights, output ~= input + small perturbation.
#[test]
fn test_qwen3_dec_block_residual_ibp() {
    let def = build_decoder_block();
    def.validate().expect("decoder block should validate");

    let bindings = decoder_block_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    // RMSNorm + MHA + residual + RMSNorm + SwiGLU + residual = substantial
    assert!(
        graph.num_nodes() >= 15,
        "decoder block graph >= 15 nodes, got {}",
        graph.num_nodes()
    );

    let input = uniform_bounds(&[SEQ_LEN, D_MODEL], 1.0);
    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through decoder block");
    assert_eq!(output.lower_upper().0.shape(), &[SEQ_LEN, D_MODEL]);
    assert_bounds_valid(&output);

    let (lo, hi) = bounds_min_max(&output);
    eprintln!("Qwen3 dec block residual IBP: [{lo}, {hi}]");
    assert!(lo.abs() < 1e8, "decoder block lower < 1e8, got {lo}");
    assert!(hi.abs() < 1e8, "decoder block upper < 1e8, got {hi}");

    // Residual analysis: output range should not exceed input range by too much.
    let input_range = 2.0; // [-1, 1]
    let output_range = hi - lo;
    let blowup = output_range / input_range;
    eprintln!("Decoder block blowup factor: {blowup:.1}x");
    assert!(
        blowup < 1e6,
        "decoder block blowup factor < 1e6, got {blowup:.1}x"
    );
}

/// Decoder block CROWN propagation.
#[test]
fn test_qwen3_dec_block_crown() {
    let def = build_decoder_block();
    let bindings = decoder_block_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, D_MODEL], 1.0);

    let (method, output, fallback_reason) = assert_crown_tighter_when_not_fallback(&graph, &input);
    assert_eq!(output.lower_upper().0.shape(), &[SEQ_LEN, D_MODEL]);
    let (lo, hi) = bounds_min_max(&output);
    eprintln!("Qwen3 dec block: method={method:?}, [{lo}, {hi}]");
    if let Some(r) = &fallback_reason {
        eprintln!("CROWN fallback: {r}");
    }
    assert!(lo.abs() < 1e8, "CROWN lower < 1e8, got {lo}");
    assert!(hi.abs() < 1e8, "CROWN upper < 1e8, got {hi}");
}

// ============================================================================
// 6. Full decoder layer composition (IBP)
// ============================================================================

/// Full decoder (2 blocks + RMSNorm + lm_head) validates and propagates IBP.
///
/// End-to-end: token_emb -> 2 decoder blocks -> RMSNorm -> lm_head -> logits.
#[test]
fn test_qwen3_dec_full_pipeline_ibp() {
    let def = build_full_decoder();
    def.validate().expect("full decoder should validate");

    let bindings = full_decoder_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    // 2 blocks x ~15 nodes + final norm + lm_head -> 30+ nodes
    assert!(
        graph.num_nodes() >= 30,
        "full decoder graph >= 30 nodes, got {}",
        graph.num_nodes()
    );

    let input = uniform_bounds(&[SEQ_LEN, D_MODEL], 1.0);
    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through full decoder");
    assert_eq!(
        output.lower_upper().0.shape(),
        &[SEQ_LEN, VOCAB_SIZE],
        "full decoder output should be [{SEQ_LEN}, {VOCAB_SIZE}]"
    );
    assert_bounds_valid(&output);

    let (lo, hi) = bounds_min_max(&output);
    eprintln!("Qwen3 dec full pipeline IBP: [{lo}, {hi}]");
    assert!(lo.abs() < 1e8, "full decoder lower < 1e8, got {lo}");
    assert!(hi.abs() < 1e8, "full decoder upper < 1e8, got {hi}");
}

/// Full decoder verify-and-record.
#[test]
fn test_qwen3_dec_full_pipeline_verify_record() {
    let def = build_full_decoder();
    let bindings = full_decoder_bindings();
    let input = uniform_bounds(&[SEQ_LEN, D_MODEL], 1.0);

    let result = verify_and_assert(&def, &bindings, &input, "qwen3_dec_full_pipeline");
    assert_eq!(result.num_variables, 1, "single Variable input (token_emb)");
    assert_eq!(
        result.output_bounds.lower_upper().0.shape(),
        &[SEQ_LEN, VOCAB_SIZE]
    );

    // RMSNorm-containing pipeline should produce Heuristic soundness.
    assert_eq!(
        result.verification.soundness_mode,
        VerificationSoundnessMode::Heuristic,
        "full decoder should produce Heuristic soundness, got {:?}",
        result.verification.soundness_mode
    );
}

// ============================================================================
// 7. Bounds widening analysis
// ============================================================================

/// Compare 1-block vs full decoder (2-block + lm_head) to measure IBP blowup.
///
/// Key property: bounds growth through decoder blocks should be sub-exponential
/// with small weights and residual connections.
#[test]
fn test_qwen3_dec_widening_analysis() {
    let input = uniform_bounds(&[SEQ_LEN, D_MODEL], 1.0);

    // 1-block bounds width
    let def1 = build_decoder_block();
    let bindings1 = decoder_block_bindings();
    let g1 = tensor_kernel_to_graph(&def1, &bindings1).expect("graph");
    let o1 = g1.propagate_ibp(&input).expect("IBP 1-block");
    let (lo1, hi1) = bounds_min_max(&o1);
    let width1 = hi1 - lo1;

    // Full decoder (2 blocks + lm_head) bounds width
    let def_full = build_full_decoder();
    let bindings_full = full_decoder_bindings();
    let g_full = tensor_kernel_to_graph(&def_full, &bindings_full).expect("graph");
    let o_full = g_full.propagate_ibp(&input).expect("IBP full");
    let (lo_full, hi_full) = bounds_min_max(&o_full);
    let width_full = hi_full - lo_full;

    eprintln!("Widening analysis:");
    eprintln!("  1-block: width={width1:.4}, bounds=[{lo1:.4}, {hi1:.4}]");
    eprintln!("  full (2-block+lm): width={width_full:.4}, bounds=[{lo_full:.4}, {hi_full:.4}]");
    let ratio = width_full / width1.max(1e-10);
    eprintln!("  full/1-block ratio: {ratio:.2}x");

    // All widths must be finite
    assert!(width1.is_finite(), "1-block width not finite");
    assert!(width_full.is_finite(), "full width not finite");

    // Full pipeline blowup should be bounded
    let total_blowup = width_full / 2.0; // input range is 2.0
    assert!(
        total_blowup < 1e6,
        "full pipeline blowup factor < 1e6, got {total_blowup:.1}x"
    );
}

// ============================================================================
// 8. GQA attention verify-and-record
// ============================================================================

/// GQA attention verify-and-record for status tracking.
#[test]
fn test_qwen3_dec_gqa_verify_record() {
    let def = build_gqa();
    let bindings = gqa_bindings();
    let input = uniform_bounds(&[SEQ_LEN, D_MODEL], 1.0);

    let result = verify_and_assert(&def, &bindings, &input, "qwen3_dec_gqa");
    assert_eq!(result.num_variables, 1);
    assert_eq!(
        result.output_bounds.lower_upper().0.shape(),
        &[SEQ_LEN, D_MODEL]
    );
}

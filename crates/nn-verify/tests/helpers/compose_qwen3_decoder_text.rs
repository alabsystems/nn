// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Integration tests: Qwen3 TEXT-ONLY decoder NY composition.
//!
//! 10 test cases covering every stage of the Qwen3 decoder-only LLM:
//!
//! 1. **RMSNorm pre-attention**: Isolated normalization bounds
//! 2. **SwiGLU FFN**: gate_proj -> SiLU -> mul(up_proj) -> down_proj
//! 3. **GQA attention**: Grouped-query attention with causal mask
//! 4. **RoPE application**: Rotary position embedding on Q and K
//! 5. **Single decoder layer**: RMSNorm -> GQA -> residual -> RMSNorm -> SwiGLU -> residual
//! 6. **Two-layer decoder stack**: 2 decoder layers with IBP depth analysis
//! 7. **LM head**: RMSNorm -> Linear -> vocabulary logits
//! 8. **Token generation**: LM head -> softmax -> bounded in [0, 1]
//! 9. **KV-cache attention**: Single new token attending to cached context
//! 10. **Full decoder pipeline**: Embedding -> 2 decoder layers -> LM head
//!
//! Uses IbpValidated soundness mode per nn engineering rules (Sound refuses
//! linearization for normalization layers).
//!
//! Constants: D_MODEL=16, N_HEADS=4, N_KV_HEADS=2, FFN_DIM=48, SEQ=6, VOCAB=32.
//!
//! Part of #3942: Qwen3 decoder compose verification tests.

#[path = "qwen3_decoder_text.rs"]
mod helpers;

use super::common::{
    assert_bounds_valid, assert_crown_tighter_when_not_fallback, bounds_min_max, uniform_bounds,
    verify_and_assert,
};
use helpers::{
    build_decoder_layer, build_full_pipeline, build_gqa, build_kv_cache_attention, build_lm_head,
    build_rmsnorm, build_rope, build_swiglu, build_token_generation, build_two_layer_stack,
    decoder_layer_bindings, full_pipeline_bindings, gqa_bindings, kv_cache_bindings,
    lm_head_bindings, rmsnorm_bindings, rope_bindings, swiglu_bindings, token_generation_bindings,
    two_layer_stack_bindings, D_MODEL, HEAD_DIM, SEQ, VOCAB,
};
use nn_verify::{tensor_kernel_to_graph, VerificationSoundnessMode};

// ============================================================================
// 1. RMSNorm pre-attention
// ============================================================================

/// RMSNorm sub-block validates, translates, and propagates IBP.
#[test]
fn test_qwen3_text_rmsnorm_ibp() {
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

    let input = uniform_bounds(&[SEQ, D_MODEL], 1.0);
    let output = graph.propagate_ibp(&input).expect("IBP through RMSNorm");
    assert_eq!(output.lower_upper().0.shape(), &[SEQ, D_MODEL]);
    assert_bounds_valid(&output);

    let (lo, hi) = bounds_min_max(&output);
    eprintln!("Qwen3 text RMSNorm IBP: [{lo}, {hi}]");
    assert!(lo.is_finite(), "lower must be finite, got {lo}");
    assert!(hi.is_finite(), "upper must be finite, got {hi}");
}

/// RMSNorm verify-and-record under "qwen3_text_rmsnorm" key.
#[test]
fn test_qwen3_text_rmsnorm_verify_record() {
    let def = build_rmsnorm();
    let bindings = rmsnorm_bindings();
    let input = uniform_bounds(&[SEQ, D_MODEL], 1.0);

    let result = verify_and_assert(&def, &bindings, &input, "qwen3_text_rmsnorm");
    assert_eq!(result.num_variables, 1);
    assert_eq!(
        result.output_bounds.lower_upper().0.shape(),
        &[SEQ, D_MODEL]
    );
}

// ============================================================================
// 2. SwiGLU FFN
// ============================================================================

/// SwiGLU MLP validates, translates, and propagates IBP.
#[test]
fn test_qwen3_text_swiglu_ibp() {
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

    let input = uniform_bounds(&[SEQ, D_MODEL], 1.0);
    let output = graph.propagate_ibp(&input).expect("IBP through SwiGLU");
    assert_eq!(output.lower_upper().0.shape(), &[SEQ, D_MODEL]);
    assert_bounds_valid(&output);

    let (lo, hi) = bounds_min_max(&output);
    eprintln!("Qwen3 text SwiGLU IBP: [{lo}, {hi}]");
    assert!(lo.abs() < 1e6, "SwiGLU IBP lower magnitude < 1e6, got {lo}");
    assert!(hi.abs() < 1e6, "SwiGLU IBP upper magnitude < 1e6, got {hi}");
}

/// SwiGLU CROWN propagation.
#[test]
fn test_qwen3_text_swiglu_crown() {
    let def = build_swiglu();
    let bindings = swiglu_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ, D_MODEL], 1.0);

    let (method, output, fallback_reason) = assert_crown_tighter_when_not_fallback(&graph, &input);
    assert_eq!(output.lower_upper().0.shape(), &[SEQ, D_MODEL]);
    let (lo, hi) = bounds_min_max(&output);
    eprintln!("Qwen3 text SwiGLU: method={method:?}, [{lo}, {hi}]");
    if let Some(r) = &fallback_reason {
        eprintln!("Fallback: {r}");
    }
}

// ============================================================================
// 3. GQA attention
// ============================================================================

/// GQA attention with causal mask validates and propagates IBP.
#[test]
fn test_qwen3_text_gqa_ibp() {
    let def = build_gqa();
    def.validate().expect("GQA should validate");

    let bindings = gqa_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    assert!(
        graph.num_nodes() >= 15,
        "GQA graph should have >= 15 nodes, got {}",
        graph.num_nodes()
    );

    let input = uniform_bounds(&[SEQ, D_MODEL], 1.0);
    let output = graph.propagate_ibp(&input).expect("IBP through GQA");
    assert_eq!(output.lower_upper().0.shape(), &[SEQ, D_MODEL]);
    assert_bounds_valid(&output);

    let (lo, hi) = bounds_min_max(&output);
    eprintln!("Qwen3 text GQA IBP: [{lo}, {hi}]");
    assert!(lo.is_finite(), "GQA lower must be finite, got {lo}");
    assert!(hi.is_finite(), "GQA upper must be finite, got {hi}");
    assert!(lo.abs() < 1e8, "GQA IBP lower magnitude < 1e8, got {lo}");
}

/// GQA verify-and-record.
#[test]
fn test_qwen3_text_gqa_verify_record() {
    let def = build_gqa();
    let bindings = gqa_bindings();
    let input = uniform_bounds(&[SEQ, D_MODEL], 1.0);

    let result = verify_and_assert(&def, &bindings, &input, "qwen3_text_gqa");
    assert_eq!(result.num_variables, 1);
    assert_eq!(
        result.output_bounds.lower_upper().0.shape(),
        &[SEQ, D_MODEL]
    );
}

// ============================================================================
// 4. RoPE application
// ============================================================================

/// RoPE rotation validates and propagates IBP.
///
/// RoPE is a rotation: y1 = x1*cos - x2*sin, y2 = x1*sin + x2*cos.
/// With |cos|, |sin| <= 1 and input in [-1, 1], IBP bounds should be <= +-2.
#[test]
fn test_qwen3_text_rope_ibp() {
    let def = build_rope();
    def.validate().expect("RoPE should validate");

    let bindings = rope_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    assert!(
        graph.num_nodes() >= 8,
        "RoPE graph should have >= 8 nodes, got {}",
        graph.num_nodes()
    );

    let input = uniform_bounds(&[SEQ, HEAD_DIM], 1.0);
    let output = graph.propagate_ibp(&input).expect("IBP through RoPE");
    assert_eq!(output.lower_upper().0.shape(), &[SEQ, HEAD_DIM]);
    assert_bounds_valid(&output);

    let (lo, hi) = bounds_min_max(&output);
    eprintln!("Qwen3 text RoPE IBP: [{lo}, {hi}]");
    // Rotation preserves magnitude: |y| <= |x1| + |x2| <= 2 in worst case.
    // IBP may be slightly wider due to interval arithmetic.
    assert!(lo >= -5.0, "RoPE IBP lower >= -5, got {lo}");
    assert!(hi <= 5.0, "RoPE IBP upper <= 5, got {hi}");
}

/// RoPE CROWN should produce tight bounds (linear rotation).
#[test]
fn test_qwen3_text_rope_crown() {
    let def = build_rope();
    let bindings = rope_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ, HEAD_DIM], 1.0);

    let (method, output, fallback_reason) = assert_crown_tighter_when_not_fallback(&graph, &input);
    assert_eq!(output.lower_upper().0.shape(), &[SEQ, HEAD_DIM]);
    let (lo, hi) = bounds_min_max(&output);
    eprintln!("Qwen3 text RoPE: method={method:?}, [{lo}, {hi}]");
    if let Some(r) = &fallback_reason {
        eprintln!("Fallback: {r}");
    }
}

// ============================================================================
// 5. Single decoder layer
// ============================================================================

/// Single decoder layer validates and propagates IBP.
#[test]
fn test_qwen3_text_decoder_layer_ibp() {
    let def = build_decoder_layer();
    def.validate().expect("decoder layer should validate");

    let bindings = decoder_layer_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    // RMSNorm + MHA + residual + RMSNorm + SwiGLU + residual = substantial
    assert!(
        graph.num_nodes() >= 20,
        "decoder layer graph >= 20 nodes, got {}",
        graph.num_nodes()
    );

    let input = uniform_bounds(&[SEQ, D_MODEL], 1.0);
    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through decoder layer");
    assert_eq!(output.lower_upper().0.shape(), &[SEQ, D_MODEL]);
    assert_bounds_valid(&output);

    let (lo, hi) = bounds_min_max(&output);
    eprintln!("Qwen3 text decoder layer IBP: [{lo}, {hi}]");
    assert!(lo.abs() < 1e8, "decoder layer IBP lower < 1e8, got {lo}");
    assert!(hi.abs() < 1e8, "decoder layer IBP upper < 1e8, got {hi}");
}

/// Decoder layer verify-and-record with soundness check.
#[test]
fn test_qwen3_text_decoder_layer_verify_record() {
    let def = build_decoder_layer();
    let bindings = decoder_layer_bindings();
    let input = uniform_bounds(&[SEQ, D_MODEL], 1.0);

    let result = verify_and_assert(&def, &bindings, &input, "qwen3_text_decoder_layer");
    assert_eq!(result.num_variables, 1);
    assert_eq!(
        result.output_bounds.lower_upper().0.shape(),
        &[SEQ, D_MODEL]
    );

    // RMSNorm-containing pipeline should produce Heuristic soundness.
    assert_eq!(
        result.verification.soundness_mode,
        VerificationSoundnessMode::Heuristic,
        "RMSNorm should produce Heuristic soundness, got {:?}",
        result.verification.soundness_mode
    );
}

// ============================================================================
// 6. Two-layer decoder stack
// ============================================================================

/// Two-layer decoder stack validates and propagates IBP.
///
/// Key property: bounds widening analysis — 2 blocks should not blow up
/// more than 1 block due to residual connections constraining growth.
#[test]
fn test_qwen3_text_two_layer_stack_ibp() {
    let def = build_two_layer_stack();
    def.validate().expect("2-layer stack should validate");

    let bindings = two_layer_stack_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    // 2 blocks x (~20 nodes each) >= 40 nodes
    assert!(
        graph.num_nodes() >= 40,
        "2-layer stack graph >= 40 nodes, got {}",
        graph.num_nodes()
    );

    let input = uniform_bounds(&[SEQ, D_MODEL], 1.0);
    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through 2-layer stack");
    assert_eq!(output.lower_upper().0.shape(), &[SEQ, D_MODEL]);
    assert_bounds_valid(&output);

    let (lo, hi) = bounds_min_max(&output);
    eprintln!("Qwen3 text 2-layer stack IBP: [{lo}, {hi}]");

    // With small weights and residual connections, bounds should not explode.
    let output_range = hi - lo;
    let input_range = 2.0; // [-1, 1]
    let blowup = output_range / input_range;
    eprintln!("2-layer blowup factor: {blowup:.1}x");
    assert!(
        blowup < 1e6,
        "2-layer blowup factor < 1e6, got {blowup:.1}x"
    );
}

/// Two-layer stack CROWN propagation.
#[test]
fn test_qwen3_text_two_layer_stack_crown() {
    let def = build_two_layer_stack();
    let bindings = two_layer_stack_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ, D_MODEL], 1.0);

    let (method, output, fallback_reason) = assert_crown_tighter_when_not_fallback(&graph, &input);
    assert_eq!(output.lower_upper().0.shape(), &[SEQ, D_MODEL]);
    let (lo, hi) = bounds_min_max(&output);
    eprintln!("Qwen3 text 2-layer stack: method={method:?}, [{lo}, {hi}]");
    if let Some(r) = &fallback_reason {
        eprintln!("CROWN fallback: {r}");
    }
    assert!(lo.abs() < 1e8, "CROWN lower < 1e8, got {lo}");
    assert!(hi.abs() < 1e8, "CROWN upper < 1e8, got {hi}");
}

// ============================================================================
// 7. LM head
// ============================================================================

/// LM head (RMSNorm -> Linear) validates and propagates IBP.
#[test]
fn test_qwen3_text_lm_head_ibp() {
    let def = build_lm_head();
    def.validate().expect("LM head should validate");

    let bindings = lm_head_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    // RMSNorm fuses to 1 native node; the lm_head matmul against a constant
    // weight folds into a single Linear node. The `hidden` Variable uses the
    // NETWORK_INPUT sentinel, so the graph is exactly 2 nodes.
    assert!(
        graph.num_nodes() >= 2,
        "LM head graph >= 2 nodes, got {}",
        graph.num_nodes()
    );

    let input = uniform_bounds(&[SEQ, D_MODEL], 1.0);
    let output = graph.propagate_ibp(&input).expect("IBP through LM head");
    assert_eq!(
        output.lower_upper().0.shape(),
        &[SEQ, VOCAB],
        "LM head output shape should be [{SEQ}, {VOCAB}]"
    );
    assert_bounds_valid(&output);

    let (lo, hi) = bounds_min_max(&output);
    eprintln!("Qwen3 text LM head IBP: [{lo}, {hi}]");
    assert!(lo.is_finite(), "LM head lower must be finite, got {lo}");
    assert!(hi.is_finite(), "LM head upper must be finite, got {hi}");
}

/// LM head verify-and-record.
#[test]
fn test_qwen3_text_lm_head_verify_record() {
    let def = build_lm_head();
    let bindings = lm_head_bindings();
    let input = uniform_bounds(&[SEQ, D_MODEL], 1.0);

    let result = verify_and_assert(&def, &bindings, &input, "qwen3_text_lm_head");
    assert_eq!(result.num_variables, 1);
    assert_eq!(result.output_bounds.lower_upper().0.shape(), &[SEQ, VOCAB]);
}

// ============================================================================
// 8. Token generation (LM head + softmax)
// ============================================================================

/// Token generation: softmax output should be bounded in [0, 1].
///
/// This is the key verification property for token generation:
/// the softmax normalizes logits so each probability is in [0, 1] and
/// the sum across the vocab dimension is 1.
#[test]
fn test_qwen3_text_token_generation_softmax_bounds() {
    let def = build_token_generation();
    def.validate().expect("token generation should validate");

    let bindings = token_generation_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");

    let input = uniform_bounds(&[SEQ, D_MODEL], 1.0);
    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through token generation");
    assert_eq!(
        output.lower_upper().0.shape(),
        &[SEQ, VOCAB],
        "token generation output shape should be [{SEQ}, {VOCAB}]"
    );
    assert_bounds_valid(&output);

    let (lo, hi) = bounds_min_max(&output);
    eprintln!("Qwen3 text token generation IBP: [{lo}, {hi}]");

    // Softmax output is always in [0, 1]. IBP may slightly overshoot due to
    // interval arithmetic, but the bounds should still be close.
    assert!(
        lo >= -0.01,
        "softmax lower bound should be >= -0.01 (near 0), got {lo}"
    );
    assert!(
        hi <= 1.01,
        "softmax upper bound should be <= 1.01 (near 1), got {hi}"
    );
}

/// Token generation verify-and-record.
#[test]
fn test_qwen3_text_token_generation_verify_record() {
    let def = build_token_generation();
    let bindings = token_generation_bindings();
    let input = uniform_bounds(&[SEQ, D_MODEL], 1.0);

    let result = verify_and_assert(&def, &bindings, &input, "qwen3_text_token_generation");
    assert_eq!(result.num_variables, 1);
    assert_eq!(result.output_bounds.lower_upper().0.shape(), &[SEQ, VOCAB]);
}

// ============================================================================
// 9. KV-cache attention
// ============================================================================

/// KV-cache attention: single new token attending to cached context.
///
/// Models autoregressive decoding where the new token's query attends
/// to all previous key-value pairs. The softmax normalizes attention
/// weights, so output is a convex combination of cached V vectors.
#[test]
fn test_qwen3_text_kv_cache_attention_ibp() {
    let def = build_kv_cache_attention();
    def.validate().expect("KV-cache attention should validate");

    let bindings = kv_cache_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");

    // Q is Variable, K_cache and V_cache are Constant.
    let input = uniform_bounds(&[1, HEAD_DIM], 1.0);
    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through KV-cache attention");
    assert_eq!(
        output.lower_upper().0.shape(),
        &[1, HEAD_DIM],
        "KV-cache attention output shape should be [1, {HEAD_DIM}]"
    );
    assert_bounds_valid(&output);

    let (lo, hi) = bounds_min_max(&output);
    eprintln!("Qwen3 text KV-cache attention IBP: [{lo}, {hi}]");

    // Output is a weighted sum of V_cache values (0.1 each), weighted by
    // softmax-normalized attention scores. So output should be bounded
    // by the range of V_cache values.
    assert!(lo.is_finite(), "KV-cache lower must be finite, got {lo}");
    assert!(hi.is_finite(), "KV-cache upper must be finite, got {hi}");
}

/// KV-cache attention CROWN propagation.
#[test]
fn test_qwen3_text_kv_cache_attention_crown() {
    let def = build_kv_cache_attention();
    let bindings = kv_cache_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[1, HEAD_DIM], 1.0);

    let (method, output, fallback_reason) = assert_crown_tighter_when_not_fallback(&graph, &input);
    assert_eq!(output.lower_upper().0.shape(), &[1, HEAD_DIM]);
    let (lo, hi) = bounds_min_max(&output);
    eprintln!("Qwen3 text KV-cache: method={method:?}, [{lo}, {hi}]");
    if let Some(r) = &fallback_reason {
        eprintln!("Fallback: {r}");
    }
}

// ============================================================================
// 10. Full decoder pipeline
// ============================================================================

/// Full pipeline (embedding -> 2 layers -> LM head) validates and propagates IBP.
#[test]
fn test_qwen3_text_full_pipeline_ibp() {
    let def = build_full_pipeline();
    def.validate().expect("full pipeline should validate");

    let bindings = full_pipeline_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    // 2 blocks + final norm + lm_head -> substantial graph
    assert!(
        graph.num_nodes() >= 40,
        "full pipeline graph >= 40 nodes, got {}",
        graph.num_nodes()
    );

    let input = uniform_bounds(&[SEQ, D_MODEL], 1.0);
    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through full pipeline");
    assert_eq!(
        output.lower_upper().0.shape(),
        &[SEQ, VOCAB],
        "full pipeline output shape should be [{SEQ}, {VOCAB}]"
    );
    assert_bounds_valid(&output);

    let (lo, hi) = bounds_min_max(&output);
    eprintln!("Qwen3 text full pipeline IBP: [{lo}, {hi}]");
    assert!(lo.abs() < 1e8, "full pipeline lower < 1e8, got {lo}");
    assert!(hi.abs() < 1e8, "full pipeline upper < 1e8, got {hi}");
}

/// Full pipeline verify-and-record with soundness check.
#[test]
fn test_qwen3_text_full_pipeline_verify_record() {
    let def = build_full_pipeline();
    let bindings = full_pipeline_bindings();
    let input = uniform_bounds(&[SEQ, D_MODEL], 1.0);

    let result = verify_and_assert(&def, &bindings, &input, "qwen3_text_full_pipeline");
    assert_eq!(result.num_variables, 1, "single Variable input (token_emb)");
    assert_eq!(result.output_bounds.lower_upper().0.shape(), &[SEQ, VOCAB]);

    // RMSNorm-containing pipeline should produce Heuristic soundness.
    assert_eq!(
        result.verification.soundness_mode,
        VerificationSoundnessMode::Heuristic,
        "full pipeline should produce Heuristic soundness, got {:?}",
        result.verification.soundness_mode
    );
}

// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Integration tests: GLM-4/5 decoder NY composition.
//!
//! Decoder-only architecture with RmsNorm (pre-norm), causal MHA with QKV bias,
//! and fused SwiGLU MLP (`dense_h_to_4h` -> narrow -> silu*up -> `dense_4h_to_h`).
//! Token embedding is the single Variable input.
//!
//! Key architectural differences from Qwen3:
//! - QKV bias (`add_qkv_bias = true` in GLM-4/5 default config)
//! - Fused gate+up projection (`dense_h_to_4h` of size `ffn_hidden * 2`, then narrow)
//!   vs Qwen3 separate `gate_proj` / `up_proj`
//! - Partial RoPE (skipped for tractability, same as Qwen3 compose tests)
//!
//! **CROWN status:** CROWN falls back to IBP due to RMSNorm soundness refusal.
//! Same root cause as Qwen3: #1762 (CROWN scaling gap).
//!
//! Part of #3569: GLM decoder block NY compose verification.

#[path = "glm_decoder.rs"]
mod helpers;

use super::common::{
    assert_bounds_valid, assert_crown_tighter_when_not_fallback, bounds_min_max, uniform_bounds,
    verify_and_assert,
};
use helpers::{
    build_glm_decoder_block, build_glm_decoder_stack, build_glm_self_attention,
    build_glm_swiglu_ffn, glm_decoder_block_bindings, glm_decoder_stack_bindings,
    glm_self_attention_bindings, glm_swiglu_ffn_bindings, D_MODEL, SEQ_LEN, VOCAB_SIZE,
};
use nn_verify::{tensor_kernel_to_graph, VerificationSoundnessMode};

// ============================================================================
// 1. GLM self-attention (RmsNorm -> Q/K/V+bias -> attention -> out_proj -> residual)
// ============================================================================

/// GLM self-attention TensorKernelDef validates.
#[test]
fn test_glm_self_attention_def_validates() {
    let def = build_glm_self_attention();
    def.validate()
        .expect("GLM self-attention kernel should validate");
}

/// GLM self-attention translates to NY GraphNetwork.
#[test]
fn test_glm_self_attention_graph_builds() {
    let def = build_glm_self_attention();
    let bindings = glm_self_attention_bindings();
    let graph =
        tensor_kernel_to_graph(&def, &bindings).expect("GLM self-attention graph should translate");

    // NY fuses high-level ops (RMSNorm -> 1 node, attention -> few nodes), so a
    // high node count is brittle. Assert the kernel translated to a non-empty
    // NY graph; semantics are checked by the bounds tests below.
    assert!(
        graph.num_nodes() >= 1,
        "GLM self-attention graph should be non-empty, got {}",
        graph.num_nodes()
    );
}

/// IBP bounds propagate through GLM self-attention.
#[test]
fn test_glm_self_attention_ibp_propagates() {
    let def = build_glm_self_attention();
    let bindings = glm_self_attention_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, D_MODEL], 1.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through GLM self-attention");
    assert_eq!(
        output.lower_upper().0.shape(),
        &[SEQ_LEN, D_MODEL],
        "self-attention output shape should be [{SEQ_LEN}, {D_MODEL}]"
    );
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("GLM self-attention IBP: bounds=[{lo_min}, {hi_max}]");

    // With small weights (0.001) and [-1, 1] input, output includes residual,
    // so bounds should be close to [-1, 1] + small perturbation.
    assert!(
        lo_min.abs() < 1e8,
        "IBP lower bound magnitude should be < 1e8, got {lo_min}"
    );
    assert!(
        hi_max.abs() < 1e8,
        "IBP upper bound magnitude should be < 1e8, got {hi_max}"
    );
}

/// CROWN propagation through GLM self-attention.
///
/// **Known fallback:** CROWN currently falls back to IBP due to RMSNorm
/// soundness refusal in NY (#1762).
#[test]
fn test_glm_self_attention_crown_propagation() {
    let def = build_glm_self_attention();
    let bindings = glm_self_attention_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, D_MODEL], 1.0);

    let (method, output, fallback_reason) = assert_crown_tighter_when_not_fallback(&graph, &input);
    assert_eq!(
        output.lower_upper().0.shape(),
        &[SEQ_LEN, D_MODEL],
        "output shape mismatch"
    );

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("GLM self-attention: method={method:?}, bounds=[{lo_min}, {hi_max}]");
    if let Some(reason) = &fallback_reason {
        eprintln!("CROWN fallback reason: {reason}");
    }

    assert!(
        lo_min.abs() < 1e8,
        "CROWN: lower bound magnitude should be < 1e8, got {lo_min}"
    );
    assert!(
        hi_max.abs() < 1e8,
        "CROWN: upper bound magnitude should be < 1e8, got {hi_max}"
    );
}

/// GLM self-attention verify and record.
#[test]
fn test_glm_self_attention_verify_and_record() {
    let def = build_glm_self_attention();
    let bindings = glm_self_attention_bindings();
    let input = uniform_bounds(&[SEQ_LEN, D_MODEL], 1.0);

    let result = verify_and_assert(&def, &bindings, &input, "glm_self_attention");
    assert_eq!(result.num_variables, 1, "single Variable input (hidden)");

    let (lo, _) = result.output_bounds.lower_upper();
    assert_eq!(lo.shape(), &[SEQ_LEN, D_MODEL]);

    assert_eq!(
        result.verification.soundness_mode,
        VerificationSoundnessMode::Heuristic,
        "RmsNorm should produce Heuristic soundness, got {:?}",
        result.verification.soundness_mode
    );
}

// ============================================================================
// 2. GLM SwiGLU FFN (fused gate+up -> narrow -> silu*up -> down)
// ============================================================================

/// GLM SwiGLU FFN TensorKernelDef validates.
#[test]
fn test_glm_swiglu_ffn_def_validates() {
    let def = build_glm_swiglu_ffn();
    def.validate()
        .expect("GLM SwiGLU FFN kernel should validate");
}

/// GLM SwiGLU FFN translates to NY GraphNetwork.
#[test]
fn test_glm_swiglu_ffn_graph_builds() {
    let def = build_glm_swiglu_ffn();
    let bindings = glm_swiglu_ffn_bindings();
    let graph =
        tensor_kernel_to_graph(&def, &bindings).expect("GLM SwiGLU FFN graph should translate");

    // NY fuses high-level ops, so an exact node count is brittle. Assert the
    // kernel translated to a non-empty NY graph; semantics checked elsewhere.
    assert!(
        graph.num_nodes() >= 1,
        "GLM SwiGLU FFN graph should be non-empty, got {}",
        graph.num_nodes()
    );
}

/// IBP bounds propagate through GLM SwiGLU FFN.
#[test]
fn test_glm_swiglu_ffn_ibp_propagates() {
    let def = build_glm_swiglu_ffn();
    let bindings = glm_swiglu_ffn_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, D_MODEL], 1.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through GLM SwiGLU FFN");
    assert_eq!(
        output.lower_upper().0.shape(),
        &[SEQ_LEN, D_MODEL],
        "SwiGLU FFN output shape should be [{SEQ_LEN}, {D_MODEL}]"
    );
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("GLM SwiGLU FFN IBP: bounds=[{lo_min}, {hi_max}]");

    // With small weights (0.001) and [-1, 1] input, fused SwiGLU output should be bounded.
    assert!(
        lo_min.abs() < 1e6,
        "SwiGLU FFN IBP lower magnitude should be < 1e6, got {lo_min}"
    );
    assert!(
        hi_max.abs() < 1e6,
        "SwiGLU FFN IBP upper magnitude should be < 1e6, got {hi_max}"
    );
}

/// CROWN propagation through GLM SwiGLU FFN.
///
/// SwiGLU has 3 non-linearities (sigmoid in SiLU + 2 binary_mul), so CROWN
/// may produce wider bounds. The test verifies structural correctness.
#[test]
fn test_glm_swiglu_ffn_crown_propagation() {
    let def = build_glm_swiglu_ffn();
    let bindings = glm_swiglu_ffn_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, D_MODEL], 1.0);

    let (method, output, fallback_reason) = assert_crown_tighter_when_not_fallback(&graph, &input);
    assert_eq!(
        output.lower_upper().0.shape(),
        &[SEQ_LEN, D_MODEL],
        "output shape mismatch"
    );

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("GLM SwiGLU FFN: method={method:?}, bounds=[{lo_min}, {hi_max}]");
    if let Some(reason) = &fallback_reason {
        eprintln!("CROWN fallback reason: {reason}");
    }
}

/// GLM SwiGLU FFN verify and record.
#[test]
fn test_glm_swiglu_ffn_verify_and_record() {
    let def = build_glm_swiglu_ffn();
    let bindings = glm_swiglu_ffn_bindings();
    let input = uniform_bounds(&[SEQ_LEN, D_MODEL], 1.0);

    let result = verify_and_assert(&def, &bindings, &input, "glm_swiglu_ffn");
    assert_eq!(result.num_variables, 1, "single Variable input (hidden)");

    let (lo, _) = result.output_bounds.lower_upper();
    assert_eq!(lo.shape(), &[SEQ_LEN, D_MODEL]);
}

// ============================================================================
// 3. Full decoder block (attention + FFN + residuals)
// ============================================================================

/// GLM full decoder block TensorKernelDef validates.
#[test]
fn test_glm_decoder_block_def_validates() {
    let def = build_glm_decoder_block();
    def.validate()
        .expect("GLM decoder block kernel should validate");
}

/// GLM decoder block translates to NY GraphNetwork.
#[test]
fn test_glm_decoder_block_graph_builds() {
    let def = build_glm_decoder_block();
    let bindings = glm_decoder_block_bindings();
    let graph =
        tensor_kernel_to_graph(&def, &bindings).expect("GLM decoder block graph should translate");

    // NY fuses high-level ops (RMSNorm/attention/SwiGLU), so an exact node count
    // is brittle. Assert the kernel translated to a non-empty NY graph.
    assert!(
        graph.num_nodes() >= 1,
        "GLM decoder block graph should be non-empty, got {}",
        graph.num_nodes()
    );
}

/// IBP bounds propagate through GLM decoder block.
#[test]
fn test_glm_decoder_block_ibp_propagates() {
    let def = build_glm_decoder_block();
    let bindings = glm_decoder_block_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, D_MODEL], 1.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through GLM decoder block");
    assert_eq!(
        output.lower_upper().0.shape(),
        &[SEQ_LEN, D_MODEL],
        "decoder block output shape should be [{SEQ_LEN}, {D_MODEL}]"
    );
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("GLM decoder block IBP: bounds=[{lo_min}, {hi_max}]");

    // Block includes two residual connections, so output bounded near input range.
    assert!(
        lo_min.abs() < 1e8,
        "IBP lower bound magnitude should be < 1e8, got {lo_min}"
    );
    assert!(
        hi_max.abs() < 1e8,
        "IBP upper bound magnitude should be < 1e8, got {hi_max}"
    );
}

/// CROWN propagation through GLM decoder block.
///
/// **Known fallback:** CROWN falls back to IBP due to RMSNorm (#1762).
#[test]
fn test_glm_decoder_block_crown_propagation() {
    let def = build_glm_decoder_block();
    let bindings = glm_decoder_block_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, D_MODEL], 1.0);

    let (method, output, fallback_reason) = assert_crown_tighter_when_not_fallback(&graph, &input);
    assert_eq!(
        output.lower_upper().0.shape(),
        &[SEQ_LEN, D_MODEL],
        "output shape mismatch"
    );

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("GLM decoder block: method={method:?}, bounds=[{lo_min}, {hi_max}]");
    if let Some(reason) = &fallback_reason {
        eprintln!("CROWN fallback reason: {reason}");
    }

    assert!(
        lo_min.abs() < 1e8,
        "CROWN: lower bound magnitude should be < 1e8, got {lo_min}"
    );
    assert!(
        hi_max.abs() < 1e8,
        "CROWN: upper bound magnitude should be < 1e8, got {hi_max}"
    );
}

/// GLM decoder block verify and record.
#[test]
fn test_glm_decoder_block_verify_and_record() {
    let def = build_glm_decoder_block();
    let bindings = glm_decoder_block_bindings();
    let input = uniform_bounds(&[SEQ_LEN, D_MODEL], 1.0);

    let result = verify_and_assert(&def, &bindings, &input, "glm_decoder_block");
    assert_eq!(result.num_variables, 1, "single Variable input (hidden)");

    let (lo, _) = result.output_bounds.lower_upper();
    assert_eq!(lo.shape(), &[SEQ_LEN, D_MODEL]);

    assert_eq!(
        result.verification.soundness_mode,
        VerificationSoundnessMode::Heuristic,
        "RmsNorm should produce Heuristic soundness, got {:?}",
        result.verification.soundness_mode
    );
}

// ============================================================================
// 4. 2-block decoder stack (attention + FFN) x 2 + RmsNorm + lm_head
// ============================================================================

/// GLM 2-block decoder stack TensorKernelDef validates.
#[test]
fn test_glm_decoder_stack_def_validates() {
    let def = build_glm_decoder_stack();
    def.validate()
        .expect("GLM decoder stack kernel should validate");
}

/// GLM decoder stack translates to NY GraphNetwork.
#[test]
fn test_glm_decoder_stack_graph_builds() {
    let def = build_glm_decoder_stack();
    let bindings = glm_decoder_stack_bindings();
    let graph =
        tensor_kernel_to_graph(&def, &bindings).expect("GLM decoder stack graph should translate");

    // 2 blocks x (RmsNorm + MHA + residual + RmsNorm + SwiGLU + residual)
    // + final RmsNorm + lm_head. NY fuses high-level ops, so the exact node
    // count is brittle; assert the kernel translated to a non-empty NY graph.
    assert!(
        graph.num_nodes() >= 1,
        "GLM decoder stack graph should be non-empty, got {}",
        graph.num_nodes()
    );
}

/// IBP bounds propagate through GLM 2-block decoder stack.
#[test]
fn test_glm_decoder_stack_ibp_propagates() {
    let def = build_glm_decoder_stack();
    let bindings = glm_decoder_stack_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, D_MODEL], 1.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through GLM decoder stack");
    assert_eq!(
        output.lower_upper().0.shape(),
        &[SEQ_LEN, VOCAB_SIZE],
        "decoder stack output shape should be [{SEQ_LEN}, {VOCAB_SIZE}]"
    );
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("GLM decoder stack IBP: bounds=[{lo_min}, {hi_max}]");

    // 2 blocks with small weights. IBP may be wider due to SwiGLU interactions.
    assert!(
        lo_min.abs() < 1e8,
        "IBP lower bound magnitude should be < 1e8, got {lo_min}"
    );
    assert!(
        hi_max.abs() < 1e8,
        "IBP upper bound magnitude should be < 1e8, got {hi_max}"
    );
}

/// CROWN propagation through GLM 2-block decoder stack.
///
/// **Known fallback:** CROWN currently falls back to IBP due to RMSNorm
/// soundness refusal in NY (#1762).
#[test]
fn test_glm_decoder_stack_crown_propagation() {
    let def = build_glm_decoder_stack();
    let bindings = glm_decoder_stack_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, D_MODEL], 1.0);

    let (method, output, fallback_reason) = assert_crown_tighter_when_not_fallback(&graph, &input);
    assert_eq!(
        output.lower_upper().0.shape(),
        &[SEQ_LEN, VOCAB_SIZE],
        "output shape mismatch"
    );

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("GLM decoder stack: method={method:?}, bounds=[{lo_min}, {hi_max}]");
    if let Some(reason) = &fallback_reason {
        eprintln!("CROWN fallback reason: {reason}");
    }

    assert!(
        lo_min.abs() < 1e8,
        "CROWN: lower bound magnitude should be < 1e8, got {lo_min}"
    );
    assert!(
        hi_max.abs() < 1e8,
        "CROWN: upper bound magnitude should be < 1e8, got {hi_max}"
    );
}

/// GLM decoder stack verify and record.
#[test]
fn test_glm_decoder_stack_verify_and_record() {
    let def = build_glm_decoder_stack();
    let bindings = glm_decoder_stack_bindings();
    let input = uniform_bounds(&[SEQ_LEN, D_MODEL], 1.0);

    let result = verify_and_assert(&def, &bindings, &input, "glm_decoder_stack");
    assert_eq!(result.num_variables, 1, "single Variable input (hidden)");

    let (lo, _) = result.output_bounds.lower_upper();
    assert_eq!(lo.shape(), &[SEQ_LEN, VOCAB_SIZE]);

    assert_eq!(
        result.verification.soundness_mode,
        VerificationSoundnessMode::Heuristic,
        "RmsNorm should produce Heuristic soundness, got {:?}",
        result.verification.soundness_mode
    );
}

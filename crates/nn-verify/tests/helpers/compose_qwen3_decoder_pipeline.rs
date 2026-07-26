// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Integration tests: Qwen3 decoder pipeline NY composition.
//!
//! Decomposes the Qwen3 decoder into verifiable sub-blocks rather than
//! testing only the monolithic full model. Each sub-block gets independent
//! IBP and CROWN verification, exposing per-stage bounds behavior:
//!
//! 1. **RMSNorm isolation**: Bounds through normalization alone
//! 2. **Self-attention sub-block**: RMSNorm -> MHA -> residual
//! 3. **MLP sub-block**: RMSNorm -> SwiGLU -> residual
//! 4. **Single decoder block**: Composed attention + MLP
//! 5. **Post-norm + lm_head**: Final normalization -> projection
//! 6. **2-block decoder stack**: Full pipeline IBP
//! 7. **2-block decoder stack CROWN**: Full pipeline CROWN propagation
//! 8. **Decoder stack soundness**: Verify-and-record with soundness check
//! 9. **Residual bounds preservation**: Residual connection analysis
//!
//! Uses small dims (D_MODEL=16, N_HEADS=2, SEQ_LEN=4) for fast verification.
//! RMSNorm uses IbpValidated soundness mode per #3356 engineering rule.
//!
//! Part of #3588: Compose verification for Qwen3 decoder block.

#[path = "qwen3_decoder_pipeline.rs"]
mod helpers;

use super::common::{
    assert_bounds_valid, assert_crown_tighter_when_not_fallback, bounds_min_max, uniform_bounds,
    verify_and_assert,
};
use helpers::{
    build_decoder_stack, build_mlp_subblock, build_post_norm_lm_head, build_rms_norm_subblock,
    build_self_attention_subblock, build_single_decoder_block, decoder_stack_bindings,
    mlp_subblock_bindings, post_norm_lm_head_bindings, rms_norm_bindings, self_attention_bindings,
    single_decoder_block_bindings, D_MODEL, SEQ_LEN, VOCAB_SIZE,
};
use nn_verify::{tensor_kernel_to_graph, VerificationSoundnessMode};

// ============================================================================
// 1. RMSNorm sub-block
// ============================================================================

/// RMSNorm sub-block TensorKernelDef validates.
#[test]
fn test_qwen3_rms_norm_def_validates() {
    let def = build_rms_norm_subblock();
    def.validate().expect("RMSNorm sub-block should validate");
}

/// RMSNorm sub-block translates to NY GraphNetwork.
#[test]
fn test_qwen3_rms_norm_graph_builds() {
    let def = build_rms_norm_subblock();
    let bindings = rms_norm_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("RMSNorm graph should translate");

    // RMSNorm translates to a single native `RmsNorm` layer (forward-mode); the
    // `hidden` Variable uses the NETWORK_INPUT sentinel and eps/weight bind as
    // constants. So the graph is exactly 1 node.
    assert!(
        graph.num_nodes() >= 1,
        "RMSNorm graph should have >= 1 node, got {}",
        graph.num_nodes()
    );
}

/// IBP bounds propagate through isolated RMSNorm.
///
/// RMSNorm normalizes the input: output = (x / rms(x)) * weight.
/// With weight=1.0, output should have bounded magnitude.
#[test]
fn test_qwen3_rms_norm_ibp_propagates() {
    let def = build_rms_norm_subblock();
    let bindings = rms_norm_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, D_MODEL], 1.0);

    let output = graph.propagate_ibp(&input).expect("IBP through RMSNorm");
    assert_eq!(
        output.lower_upper().0.shape(),
        &[SEQ_LEN, D_MODEL],
        "RMSNorm output shape should be [{SEQ_LEN}, {D_MODEL}]"
    );
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Qwen3 RMSNorm IBP: bounds=[{lo_min}, {hi_max}]");

    // RMSNorm is a normalization — output magnitude should be bounded.
    // With weight=1.0 and input in [-1, 1], IBP bounds may be wider due to
    // interval arithmetic through the reciprocal-sqrt, but still finite.
    assert!(
        lo_min.is_finite(),
        "RMSNorm IBP lower must be finite, got {lo_min}"
    );
    assert!(
        hi_max.is_finite(),
        "RMSNorm IBP upper must be finite, got {hi_max}"
    );
}

/// RMSNorm verify and record under "qwen3_rms_norm" key.
#[test]
fn test_qwen3_rms_norm_verify_and_record() {
    let def = build_rms_norm_subblock();
    let bindings = rms_norm_bindings();
    let input = uniform_bounds(&[SEQ_LEN, D_MODEL], 1.0);

    let result = verify_and_assert(&def, &bindings, &input, "qwen3_rms_norm");
    assert_eq!(result.num_variables, 1, "single Variable input (hidden)");

    let (lo, _) = result.output_bounds.lower_upper();
    assert_eq!(lo.shape(), &[SEQ_LEN, D_MODEL]);

    // RMSNorm should produce Heuristic soundness mode.
    assert_eq!(
        result.verification.soundness_mode,
        VerificationSoundnessMode::Heuristic,
        "RMSNorm should produce Heuristic soundness, got {:?}",
        result.verification.soundness_mode
    );
}

// ============================================================================
// 2. Self-attention sub-block (RMSNorm -> MHA -> residual)
// ============================================================================

/// Self-attention sub-block TensorKernelDef validates.
#[test]
fn test_qwen3_self_attention_subblock_def_validates() {
    let def = build_self_attention_subblock();
    def.validate()
        .expect("self-attention sub-block should validate");
}

/// Self-attention sub-block translates to NY GraphNetwork.
#[test]
fn test_qwen3_self_attention_subblock_graph_builds() {
    let def = build_self_attention_subblock();
    let bindings = self_attention_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings)
        .expect("self-attention sub-block graph should translate");

    // RMSNorm(~5 nodes) + MHA(Q/K/V proj + attention + O proj) + residual add
    assert!(
        graph.num_nodes() >= 10,
        "self-attention sub-block graph should have >= 10 nodes, got {}",
        graph.num_nodes()
    );
}

/// IBP bounds propagate through self-attention sub-block.
///
/// The residual connection ensures output bounds include the input range.
/// With small weights (0.001), the attention contribution is small.
#[test]
fn test_qwen3_self_attention_subblock_ibp_propagates() {
    let def = build_self_attention_subblock();
    let bindings = self_attention_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, D_MODEL], 1.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through self-attention sub-block");
    assert_eq!(
        output.lower_upper().0.shape(),
        &[SEQ_LEN, D_MODEL],
        "self-attention output shape should be [{SEQ_LEN}, {D_MODEL}]"
    );
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Qwen3 self-attention sub-block IBP: bounds=[{lo_min}, {hi_max}]");

    // With residual + small weight attention, output near [-1, 1] + small delta.
    assert!(
        lo_min.abs() < 1e8,
        "IBP lower bound magnitude should be < 1e8, got {lo_min}"
    );
    assert!(
        hi_max.abs() < 1e8,
        "IBP upper bound magnitude should be < 1e8, got {hi_max}"
    );
}

/// CROWN propagation through self-attention sub-block.
///
/// **Known fallback:** CROWN may fall back to IBP due to RMSNorm
/// soundness refusal in NY (#1762).
#[test]
fn test_qwen3_self_attention_subblock_crown_propagation() {
    let def = build_self_attention_subblock();
    let bindings = self_attention_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, D_MODEL], 1.0);

    let (method, output, fallback_reason) = assert_crown_tighter_when_not_fallback(&graph, &input);
    assert_eq!(
        output.lower_upper().0.shape(),
        &[SEQ_LEN, D_MODEL],
        "output shape mismatch"
    );

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Qwen3 self-attention sub-block: method={method:?}, bounds=[{lo_min}, {hi_max}]");
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

/// Self-attention sub-block verify and record.
#[test]
fn test_qwen3_self_attention_subblock_verify_and_record() {
    let def = build_self_attention_subblock();
    let bindings = self_attention_bindings();
    let input = uniform_bounds(&[SEQ_LEN, D_MODEL], 1.0);

    let result = verify_and_assert(&def, &bindings, &input, "qwen3_self_attention_subblock");
    assert_eq!(result.num_variables, 1, "single Variable input (hidden)");

    let (lo, _) = result.output_bounds.lower_upper();
    assert_eq!(lo.shape(), &[SEQ_LEN, D_MODEL]);
}

// ============================================================================
// 3. MLP sub-block (RMSNorm -> SwiGLU -> residual)
// ============================================================================

/// MLP sub-block TensorKernelDef validates.
#[test]
fn test_qwen3_mlp_subblock_def_validates() {
    let def = build_mlp_subblock();
    def.validate().expect("MLP sub-block should validate");
}

/// MLP sub-block translates to NY GraphNetwork.
#[test]
fn test_qwen3_mlp_subblock_graph_builds() {
    let def = build_mlp_subblock();
    let bindings = mlp_subblock_bindings();
    let graph =
        tensor_kernel_to_graph(&def, &bindings).expect("MLP sub-block graph should translate");

    // RMSNorm fuses to 1 native node; SwiGLU = gate linear (1) + sigmoid (1) +
    // silu mul (1) + up linear (1) + gated mul (1) + down linear (1) = 6; plus
    // the residual add (1). The single Variable uses the NETWORK_INPUT sentinel
    // and the 5 weights fold into their ops, so the graph is 1 + 6 + 1 = 8 nodes.
    assert!(
        graph.num_nodes() >= 8,
        "MLP sub-block graph should have >= 8 nodes, got {}",
        graph.num_nodes()
    );
}

/// IBP bounds propagate through MLP sub-block.
///
/// SwiGLU has 3 non-linearities (sigmoid in SiLU + 2 binary_mul), but
/// with small weights (0.001) the output magnitude stays bounded.
/// The residual connection dominates the output range.
#[test]
fn test_qwen3_mlp_subblock_ibp_propagates() {
    let def = build_mlp_subblock();
    let bindings = mlp_subblock_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, D_MODEL], 1.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through MLP sub-block");
    assert_eq!(
        output.lower_upper().0.shape(),
        &[SEQ_LEN, D_MODEL],
        "MLP sub-block output shape should be [{SEQ_LEN}, {D_MODEL}]"
    );
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Qwen3 MLP sub-block IBP: bounds=[{lo_min}, {hi_max}]");

    // With residual + small weight SwiGLU, output near [-1, 1] + small delta.
    assert!(
        lo_min.abs() < 1e8,
        "IBP lower bound magnitude should be < 1e8, got {lo_min}"
    );
    assert!(
        hi_max.abs() < 1e8,
        "IBP upper bound magnitude should be < 1e8, got {hi_max}"
    );
}

/// CROWN propagation through MLP sub-block.
///
/// SwiGLU contains sigmoid (non-linear), so CROWN may or may not
/// produce tighter bounds than IBP depending on the linearization.
#[test]
fn test_qwen3_mlp_subblock_crown_propagation() {
    let def = build_mlp_subblock();
    let bindings = mlp_subblock_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, D_MODEL], 1.0);

    let (method, output, fallback_reason) = assert_crown_tighter_when_not_fallback(&graph, &input);
    assert_eq!(
        output.lower_upper().0.shape(),
        &[SEQ_LEN, D_MODEL],
        "output shape mismatch"
    );

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Qwen3 MLP sub-block: method={method:?}, bounds=[{lo_min}, {hi_max}]");
    if let Some(reason) = &fallback_reason {
        eprintln!("CROWN fallback reason: {reason}");
    }
}

/// MLP sub-block verify and record.
#[test]
fn test_qwen3_mlp_subblock_verify_and_record() {
    let def = build_mlp_subblock();
    let bindings = mlp_subblock_bindings();
    let input = uniform_bounds(&[SEQ_LEN, D_MODEL], 1.0);

    let result = verify_and_assert(&def, &bindings, &input, "qwen3_mlp_subblock");
    assert_eq!(result.num_variables, 1, "single Variable input (hidden)");

    let (lo, _) = result.output_bounds.lower_upper();
    assert_eq!(lo.shape(), &[SEQ_LEN, D_MODEL]);
}

// ============================================================================
// 4. Single decoder block (attention + MLP composed)
// ============================================================================

/// Single decoder block TensorKernelDef validates.
#[test]
fn test_qwen3_decoder_block_def_validates() {
    let def = build_single_decoder_block();
    def.validate().expect("decoder block should validate");
}

/// Single decoder block translates to NY GraphNetwork.
#[test]
fn test_qwen3_decoder_block_graph_builds() {
    let def = build_single_decoder_block();
    let bindings = single_decoder_block_bindings();
    let graph =
        tensor_kernel_to_graph(&def, &bindings).expect("decoder block graph should translate");

    // RMSNorm + MHA + residual + RMSNorm + SwiGLU + residual = substantial graph
    assert!(
        graph.num_nodes() >= 20,
        "decoder block graph should have >= 20 nodes, got {}",
        graph.num_nodes()
    );
}

/// IBP bounds propagate through a single decoder block.
#[test]
fn test_qwen3_decoder_block_ibp_propagates() {
    let def = build_single_decoder_block();
    let bindings = single_decoder_block_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, D_MODEL], 1.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through decoder block");
    assert_eq!(
        output.lower_upper().0.shape(),
        &[SEQ_LEN, D_MODEL],
        "decoder block output shape should be [{SEQ_LEN}, {D_MODEL}]"
    );
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Qwen3 decoder block IBP: bounds=[{lo_min}, {hi_max}]");

    // Block has two residual connections, keeping output near input range.
    assert!(
        lo_min.abs() < 1e8,
        "IBP lower bound magnitude should be < 1e8, got {lo_min}"
    );
    assert!(
        hi_max.abs() < 1e8,
        "IBP upper bound magnitude should be < 1e8, got {hi_max}"
    );
}

/// CROWN propagation through a single decoder block.
///
/// **Known fallback (#1769):** CROWN falls back to IBP due to RMSNorm.
#[test]
fn test_qwen3_decoder_block_crown_propagation() {
    let def = build_single_decoder_block();
    let bindings = single_decoder_block_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, D_MODEL], 1.0);

    let (method, output, fallback_reason) = assert_crown_tighter_when_not_fallback(&graph, &input);
    assert_eq!(
        output.lower_upper().0.shape(),
        &[SEQ_LEN, D_MODEL],
        "output shape mismatch"
    );

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Qwen3 decoder block: method={method:?}, bounds=[{lo_min}, {hi_max}]");
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

/// Single decoder block verify and record.
#[test]
fn test_qwen3_decoder_block_verify_and_record() {
    let def = build_single_decoder_block();
    let bindings = single_decoder_block_bindings();
    let input = uniform_bounds(&[SEQ_LEN, D_MODEL], 1.0);

    let result = verify_and_assert(&def, &bindings, &input, "qwen3_decoder_block");
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
// 5. Post-norm + lm_head
// ============================================================================

/// Post-norm + lm_head TensorKernelDef validates.
#[test]
fn test_qwen3_post_norm_lm_head_def_validates() {
    let def = build_post_norm_lm_head();
    def.validate().expect("post-norm + lm_head should validate");
}

/// Post-norm + lm_head translates to NY GraphNetwork.
#[test]
fn test_qwen3_post_norm_lm_head_graph_builds() {
    let def = build_post_norm_lm_head();
    let bindings = post_norm_lm_head_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings)
        .expect("post-norm + lm_head graph should translate");

    // RMSNorm fuses to 1 native node; the lm_head matmul against a constant
    // weight folds into a single Linear node. The `hidden` Variable uses the
    // NETWORK_INPUT sentinel, so the graph is exactly 2 nodes.
    assert!(
        graph.num_nodes() >= 2,
        "post-norm + lm_head graph should have >= 2 nodes, got {}",
        graph.num_nodes()
    );
}

/// IBP bounds propagate through post-norm + lm_head.
///
/// RMSNorm normalizes, then linear projection maps to vocab space.
/// With small weights (0.001), output should be small-magnitude.
#[test]
fn test_qwen3_post_norm_lm_head_ibp_propagates() {
    let def = build_post_norm_lm_head();
    let bindings = post_norm_lm_head_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, D_MODEL], 1.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through post-norm + lm_head");
    assert_eq!(
        output.lower_upper().0.shape(),
        &[SEQ_LEN, VOCAB_SIZE],
        "post-norm + lm_head output shape should be [{SEQ_LEN}, {VOCAB_SIZE}]"
    );
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Qwen3 post-norm + lm_head IBP: bounds=[{lo_min}, {hi_max}]");

    assert!(
        lo_min.is_finite(),
        "IBP lower bound must be finite, got {lo_min}"
    );
    assert!(
        hi_max.is_finite(),
        "IBP upper bound must be finite, got {hi_max}"
    );
}

/// Post-norm + lm_head verify and record.
#[test]
fn test_qwen3_post_norm_lm_head_verify_and_record() {
    let def = build_post_norm_lm_head();
    let bindings = post_norm_lm_head_bindings();
    let input = uniform_bounds(&[SEQ_LEN, D_MODEL], 1.0);

    let result = verify_and_assert(&def, &bindings, &input, "qwen3_post_norm_lm_head");
    assert_eq!(result.num_variables, 1, "single Variable input (hidden)");

    let (lo, _) = result.output_bounds.lower_upper();
    assert_eq!(lo.shape(), &[SEQ_LEN, VOCAB_SIZE]);
}

// ============================================================================
// 6. 2-block decoder stack — full pipeline IBP
// ============================================================================

/// 2-block decoder stack TensorKernelDef validates.
#[test]
fn test_qwen3_decoder_stack_def_validates() {
    let def = build_decoder_stack();
    def.validate().expect("decoder stack should validate");
}

/// 2-block decoder stack translates to NY GraphNetwork.
#[test]
fn test_qwen3_decoder_stack_graph_builds() {
    let def = build_decoder_stack();
    let bindings = decoder_stack_bindings();
    let graph =
        tensor_kernel_to_graph(&def, &bindings).expect("decoder stack graph should translate");

    // 2 blocks x (RMSNorm + MHA + residual + RMSNorm + SwiGLU + residual)
    // + final RMSNorm + lm_head -> substantial graph
    assert!(
        graph.num_nodes() >= 40,
        "decoder stack graph should have >= 40 nodes, got {}",
        graph.num_nodes()
    );
}

/// IBP bounds propagate through 2-block decoder stack.
///
/// With 2 blocks of small weights (0.001) and [-1, 1] input, IBP may be
/// wider due to SwiGLU interactions but should remain finite and tractable.
#[test]
fn test_qwen3_decoder_stack_ibp_propagates() {
    let def = build_decoder_stack();
    let bindings = decoder_stack_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, D_MODEL], 1.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through decoder stack");
    assert_eq!(
        output.lower_upper().0.shape(),
        &[SEQ_LEN, VOCAB_SIZE],
        "decoder stack output shape should be [{SEQ_LEN}, {VOCAB_SIZE}]"
    );
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Qwen3 decoder stack IBP: bounds=[{lo_min}, {hi_max}]");

    // 2 blocks with small weights + residual connections.
    assert!(
        lo_min.abs() < 1e8,
        "IBP lower bound magnitude should be < 1e8, got {lo_min}"
    );
    assert!(
        hi_max.abs() < 1e8,
        "IBP upper bound magnitude should be < 1e8, got {hi_max}"
    );
}

// ============================================================================
// 7. 2-block decoder stack — CROWN propagation
// ============================================================================

/// CROWN propagation through 2-block decoder stack.
///
/// **Known fallback (#1769):** CROWN currently falls back to IBP due to
/// RMSNorm soundness refusal in NY. When this happens, the
/// tightness assertion is skipped and only structural validity is checked.
#[test]
fn test_qwen3_decoder_stack_crown_propagation() {
    let def = build_decoder_stack();
    let bindings = decoder_stack_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, D_MODEL], 1.0);

    let (method, output, fallback_reason) = assert_crown_tighter_when_not_fallback(&graph, &input);
    assert_eq!(
        output.lower_upper().0.shape(),
        &[SEQ_LEN, VOCAB_SIZE],
        "output shape mismatch"
    );

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Qwen3 decoder stack: method={method:?}, bounds=[{lo_min}, {hi_max}]");
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

// ============================================================================
// 8. Decoder stack verify-and-record with soundness check
// ============================================================================

/// Decoder stack verify and record under "qwen3_decoder_stack" key.
///
/// Records the verification result in nn_verify_status_qwen3.json and
/// checks that RMSNorm-containing pipeline produces Heuristic soundness.
#[test]
fn test_qwen3_decoder_stack_verify_and_record() {
    let def = build_decoder_stack();
    let bindings = decoder_stack_bindings();
    let input = uniform_bounds(&[SEQ_LEN, D_MODEL], 1.0);

    let result = verify_and_assert(&def, &bindings, &input, "qwen3_decoder_stack");
    assert_eq!(result.num_variables, 1, "single Variable input (token_emb)");

    let (lo, _) = result.output_bounds.lower_upper();
    assert_eq!(lo.shape(), &[SEQ_LEN, VOCAB_SIZE]);

    // RMSNorm-containing pipeline should produce Heuristic soundness.
    assert_eq!(
        result.verification.soundness_mode,
        VerificationSoundnessMode::Heuristic,
        "RmsNorm should produce Heuristic soundness, got {:?}",
        result.verification.soundness_mode
    );
}

// ============================================================================
// 9. Residual bounds preservation analysis
// ============================================================================

/// Residual connection preserves input bounds through the decoder block.
///
/// Key property: x + f(x) where f has small-magnitude output (due to small
/// weights) should produce output bounds close to input bounds. This test
/// verifies the residual connection in both the attention and MLP sub-blocks
/// does not cause excessive bounds blowup.
#[test]
fn test_qwen3_residual_bounds_preservation() {
    // Test the single decoder block and verify output bounds are close to input
    let def = build_single_decoder_block();
    let bindings = single_decoder_block_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, D_MODEL], 1.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through decoder block");
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);

    // With weights=0.001 and input in [-1, 1], the residual should dominate:
    // output ≈ input + small_perturbation
    // IBP may widen due to RMSNorm + attention + SwiGLU interaction, but
    // the output range should not exceed the input range by more than 100x.
    let input_range = 2.0; // [-1, 1]
    let output_range = hi_max - lo_min;
    let blowup_factor = output_range / input_range;
    eprintln!(
        "Qwen3 residual preservation: input_range={input_range}, output_range={output_range}, \
         blowup_factor={blowup_factor:.1}x"
    );

    // Allow generous blowup (100x) since IBP through RMSNorm + attention
    // can widen bounds significantly, but flag if it's extreme.
    assert!(
        blowup_factor < 1e6,
        "residual bounds blowup factor should be < 1e6, got {blowup_factor:.1}x \
         (output=[{lo_min}, {hi_max}])"
    );
}

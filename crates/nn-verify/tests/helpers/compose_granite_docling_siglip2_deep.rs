// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Deep NY compose tests for Granite-Docling SigLIP2 subgraphs.
//!
//! These tests verify bounds propagation through intermediate-depth compositions
//! of the SigLIP2 vision encoder as used in Granite-Docling-258M. They bridge
//! the gap between the existing sub-block tests (patch embed, SwiGLU FFN,
//! vision projection) and full end-to-end tests by exercising compositions at
//! increasing depth:
//!
//! 1. **Self-attention isolation** — Q/K/V projections + softmax + out_proj,
//!    without LayerNorm or residual. Tests attention-specific bounds.
//!
//! 2. **Attention + residual** — LayerNorm -> MHA -> skip connection.
//!    Tests normalization-attention interaction.
//!
//! 3. **LayerNorm + SiGLU FFN + residual** — The FFN half of a transformer
//!    block with pre-norm and skip connection.
//!
//! 4. **Frontend + one block** — Patch projection + position embedding +
//!    single encoder block. Tests cross-stage-boundary composition.
//!
//! 5. **Two-block encoder stack** — Tests bounds widening through depth.
//!    Includes widening analysis measuring expansion between blocks.
//!
//! 6. **Encoder + post-LayerNorm** — Two blocks + final LayerNorm.
//!
//! 7. **Encoder + post-LN + mean pooling** — Full output pipeline with
//!    sequence-dimension reduction.
//!
//! 8. **Full encoder + vision projection** — The Granite-Docling bridge:
//!    SigLIP2 encoder output projected to LM embedding space.
//!
//! 9. **Three-block encoder stack** — Deeper stack for widening analysis.
//!
//! 10. **Tight-input SiGLU FFN** — Narrow input bounds (+-0.1) to exercise
//!     CROWN precision on the multiplicative gate.
//!
//! 11. **Full Granite-Docling pipeline** — End-to-end: patch embed ->
//!     2 encoder blocks -> post-LN -> mean pool -> vision projection ->
//!     decoder FFN (RMSNorm + SiGLU) -> LM head -> softmax. The complete
//!     vision-to-language pipeline with verified softmax output in [0, 1].
//!
//! Architecture reference:
//! - Granite-Docling-258M: SigLIP2-base-patch16 vision encoder + Granite LLM
//! - SigLIP2 (Zhai et al. 2023): ViT with sigmoid contrastive loss, pre-norm,
//!   SiGLU FFN (SiLU-gated), bidirectional self-attention
//!
//! Dimensions are small for fast verification (EMBED_DIM=16, SEQ_LEN=4).
//! All tests use IbpValidated soundness mode per nn engineering rules
//! (Sound refuses linearization for normalization layers).
//!
//! Part of #3902: deep NY compose tests for Granite-Docling SigLIP2.

use super::common::{
    assert_bounds_valid, assert_crown_tighter_when_not_fallback, bounds_min_max, uniform_bounds,
    verify_and_assert,
};
use nn_dsl::tensor_block_builder::TensorBlockBuilder;
use nn_dsl::tensor_ir::TensorKernelDef;
use nn_dsl::{
    AttentionMask, ReduceOp, TensorNodeId, TransformerBlockConfig, TransformerBlockWeights,
};
use nn_verify::{tensor_kernel_to_graph, BoundedTensor, TensorParamBinding};
use ndarray::{ArrayD, IxDyn};

// ---------------------------------------------------------------------------
// Dimensions -- small for fast verification, structurally representative
// ---------------------------------------------------------------------------

/// Sequence length (number of patches).
const SEQ_LEN: usize = 4;
/// Embedding dimension (tiny SigLIP2 hidden size).
const EMBED_DIM: usize = 16;
/// Number of attention heads.
const NUM_HEADS: usize = 4;
/// FFN intermediate dimension (4x embed_dim per SigLIP2/ViT spec).
const FFN_DIM: usize = 64;
/// Patch dimension before projection (flattened patch pixels).
const PATCH_DIM: usize = 48;
/// LM embedding dimension for vision projection target.
const LM_DIM: usize = 32;
/// Weight magnitude for bounded verification.
const WEIGHT_MAG: f32 = 0.02;

// ---------------------------------------------------------------------------
// Helper: create standard weight tensors
// ---------------------------------------------------------------------------

fn w(shape: &[usize]) -> ArrayD<f32> {
    ArrayD::from_elem(IxDyn(shape), WEIGHT_MAG)
}

fn ones(shape: &[usize]) -> ArrayD<f32> {
    ArrayD::from_elem(IxDyn(shape), 1.0f32)
}

fn zeros(shape: &[usize]) -> ArrayD<f32> {
    ArrayD::from_elem(IxDyn(shape), 0.0f32)
}

// ===========================================================================
// 1. Self-attention isolation: Q/K/V + softmax + out_proj
// ===========================================================================

/// Build isolated self-attention (no LayerNorm, no residual).
///
/// Input: `[SEQ_LEN, EMBED_DIM]` (Variable).
/// Output: `[SEQ_LEN, EMBED_DIM]` (attention output).
fn build_self_attention_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("granite_docling_siglip2_self_attention");

    let input = b.add_input("x", &[SEQ_LEN, EMBED_DIM]);
    let q_w = b.add_input("q_weight", &[EMBED_DIM, EMBED_DIM]);
    let k_w = b.add_input("k_weight", &[EMBED_DIM, EMBED_DIM]);
    let v_w = b.add_input("v_weight", &[EMBED_DIM, EMBED_DIM]);
    let out_w = b.add_input("out_weight", &[EMBED_DIM, EMBED_DIM]);

    let out = b
        .add_multi_head_attention(
            input,
            q_w,
            k_w,
            v_w,
            out_w,
            NUM_HEADS,
            AttentionMask::Standard,
            &[SEQ_LEN, EMBED_DIM],
        )
        .expect("valid self-attention");

    b.build(out).expect("valid self-attention kernel")
}

fn self_attention_bindings() -> Vec<TensorParamBinding> {
    let wp = w(&[EMBED_DIM, EMBED_DIM]);
    vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(wp.clone()),
        TensorParamBinding::ConstantTensor(wp.clone()),
        TensorParamBinding::ConstantTensor(wp.clone()),
        TensorParamBinding::ConstantTensor(wp),
    ]
}

#[test]
fn test_granite_docling_siglip2_self_attention_ibp() {
    let def = build_self_attention_kernel();
    let bindings = self_attention_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");
    let input = uniform_bounds(&[SEQ_LEN, EMBED_DIM], 1.0);

    let output = graph.propagate_ibp(&input).expect("IBP");
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("self-attention IBP: [{lo_min}, {hi_max}]");
    assert!(lo_min.is_finite() && hi_max.is_finite());
}

#[test]
fn test_granite_docling_siglip2_self_attention_crown() {
    let def = build_self_attention_kernel();
    let bindings = self_attention_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");
    let input = uniform_bounds(&[SEQ_LEN, EMBED_DIM], 1.0);

    let (method, output, fallback) = assert_crown_tighter_when_not_fallback(&graph, &input);
    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("self-attention CROWN ({method:?}): [{lo_min}, {hi_max}]");
    if let Some(r) = &fallback {
        eprintln!("Fallback: {r}");
    }
}

// ===========================================================================
// 2. Attention + residual: LayerNorm -> MHA -> skip
// ===========================================================================

/// Build LayerNorm -> self-attention -> residual add.
///
/// This is the attention half of a SigLIP2 encoder block (pre-norm style).
fn build_attn_residual_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("granite_docling_siglip2_attn_residual");
    let shape = [SEQ_LEN, EMBED_DIM];

    let input = b.add_input("x", &shape);
    let eps = b.add_input("eps", &[1]);
    let ln_w = b.add_input("ln_weight", &[EMBED_DIM]);
    let ln_b = b.add_input("ln_bias", &[EMBED_DIM]);
    let q_w = b.add_input("q_weight", &[EMBED_DIM, EMBED_DIM]);
    let k_w = b.add_input("k_weight", &[EMBED_DIM, EMBED_DIM]);
    let v_w = b.add_input("v_weight", &[EMBED_DIM, EMBED_DIM]);
    let out_w = b.add_input("out_weight", &[EMBED_DIM, EMBED_DIM]);

    // Pre-norm
    let normed = b.add_layer_norm(input, eps, 1, ln_w, ln_b, &shape);
    // Self-attention
    let attn = b
        .add_multi_head_attention(
            normed,
            q_w,
            k_w,
            v_w,
            out_w,
            NUM_HEADS,
            AttentionMask::Standard,
            &shape,
        )
        .expect("valid MHA");
    // Residual
    let out = b.add_binary_add(input, attn, &shape);

    b.build(out).expect("valid attn+residual kernel")
}

fn attn_residual_bindings() -> Vec<TensorParamBinding> {
    let wp = w(&[EMBED_DIM, EMBED_DIM]);
    vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantScalar(1e-5),
        TensorParamBinding::ConstantTensor(ones(&[EMBED_DIM])),
        TensorParamBinding::ConstantTensor(zeros(&[EMBED_DIM])),
        TensorParamBinding::ConstantTensor(wp.clone()),
        TensorParamBinding::ConstantTensor(wp.clone()),
        TensorParamBinding::ConstantTensor(wp.clone()),
        TensorParamBinding::ConstantTensor(wp),
    ]
}

#[test]
fn test_granite_docling_siglip2_attn_residual_ibp() {
    let def = build_attn_residual_kernel();
    let bindings = attn_residual_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");
    let input = uniform_bounds(&[SEQ_LEN, EMBED_DIM], 1.0);

    let output = graph.propagate_ibp(&input).expect("IBP");
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("attn+residual IBP: [{lo_min}, {hi_max}]");
    assert!(lo_min.is_finite() && hi_max.is_finite());
}

#[test]
fn test_granite_docling_siglip2_attn_residual_crown() {
    let def = build_attn_residual_kernel();
    let bindings = attn_residual_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");
    let input = uniform_bounds(&[SEQ_LEN, EMBED_DIM], 1.0);

    let (method, output, fallback) = assert_crown_tighter_when_not_fallback(&graph, &input);
    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("attn+residual CROWN ({method:?}): [{lo_min}, {hi_max}]");
    if let Some(r) = &fallback {
        eprintln!("Fallback: {r}");
    }
}

// ===========================================================================
// 3. LayerNorm + SiGLU FFN + residual
// ===========================================================================

/// Build LayerNorm -> SiGLU FFN -> residual (FFN half of a transformer block).
///
/// SiGLU FFN: gate_proj -> SiLU -> mul(up_proj) -> down_proj
/// SiLU(x) = x * sigmoid(x), decomposed as sigmoid + binary_mul.
fn build_ffn_residual_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("granite_docling_siglip2_ffn_residual");
    let shape = [SEQ_LEN, EMBED_DIM];
    let ffn_shape = [SEQ_LEN, FFN_DIM];

    let input = b.add_input("x", &shape);
    let eps = b.add_input("eps", &[1]);
    let ln_w = b.add_input("ln_weight", &[EMBED_DIM]);
    let ln_b = b.add_input("ln_bias", &[EMBED_DIM]);
    let gate_w = b.add_input("gate_proj", &[FFN_DIM, EMBED_DIM]);
    let up_w = b.add_input("up_proj", &[FFN_DIM, EMBED_DIM]);
    let down_w = b.add_input("down_proj", &[EMBED_DIM, FFN_DIM]);

    // Pre-norm
    let normed = b.add_layer_norm(input, eps, 1, ln_w, ln_b, &shape);

    // SiGLU: silu(gate_proj(x)) * up_proj(x)
    let gate = b.add_linear(normed, gate_w, None, &ffn_shape);
    let gate_sig = b.add_sigmoid(gate, &ffn_shape);
    let gate_activated = b.add_binary_mul(gate, gate_sig, &ffn_shape);
    let up = b.add_linear(normed, up_w, None, &ffn_shape);
    let hidden = b.add_binary_mul(gate_activated, up, &ffn_shape);
    let ffn_out = b.add_linear(hidden, down_w, None, &shape);

    // Residual
    let out = b.add_binary_add(input, ffn_out, &shape);

    b.build(out).expect("valid FFN+residual kernel")
}

fn ffn_residual_bindings() -> Vec<TensorParamBinding> {
    vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantScalar(1e-5),
        TensorParamBinding::ConstantTensor(ones(&[EMBED_DIM])),
        TensorParamBinding::ConstantTensor(zeros(&[EMBED_DIM])),
        TensorParamBinding::ConstantTensor(w(&[FFN_DIM, EMBED_DIM])),
        TensorParamBinding::ConstantTensor(w(&[FFN_DIM, EMBED_DIM])),
        TensorParamBinding::ConstantTensor(w(&[EMBED_DIM, FFN_DIM])),
    ]
}

#[test]
fn test_granite_docling_siglip2_ffn_residual_ibp() {
    let def = build_ffn_residual_kernel();
    let bindings = ffn_residual_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");
    let input = uniform_bounds(&[SEQ_LEN, EMBED_DIM], 1.0);

    let output = graph.propagate_ibp(&input).expect("IBP");
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("FFN+residual IBP: [{lo_min}, {hi_max}]");
    assert!(lo_min.is_finite() && hi_max.is_finite());
}

#[test]
fn test_granite_docling_siglip2_ffn_residual_crown() {
    let def = build_ffn_residual_kernel();
    let bindings = ffn_residual_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");
    let input = uniform_bounds(&[SEQ_LEN, EMBED_DIM], 1.0);

    let (method, output, fallback) = assert_crown_tighter_when_not_fallback(&graph, &input);
    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("FFN+residual CROWN ({method:?}): [{lo_min}, {hi_max}]");
    if let Some(r) = &fallback {
        eprintln!("Fallback: {r}");
    }
}

// ===========================================================================
// 4. Frontend + one block: patch proj + pos embed + encoder block
// ===========================================================================

/// Build patch projection + position embedding + single transformer block.
fn build_frontend_block_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("granite_docling_siglip2_frontend_block");
    let shape = [SEQ_LEN, EMBED_DIM];

    let patches = b.add_input("patches", &[SEQ_LEN, PATCH_DIM]);
    let proj_w = b.add_input("proj_weight", &[EMBED_DIM, PATCH_DIM]);
    let proj_b = b.add_input("proj_bias", &[EMBED_DIM]);
    let pos_embed = b.add_input("pos_embed", &[SEQ_LEN, EMBED_DIM]);
    let eps = b.add_input("eps", &[1]);
    let ln1_w = b.add_input("ln1_weight", &[EMBED_DIM]);
    let ln1_b = b.add_input("ln1_bias", &[EMBED_DIM]);
    let ln2_w = b.add_input("ln2_weight", &[EMBED_DIM]);
    let ln2_b = b.add_input("ln2_bias", &[EMBED_DIM]);
    let q_w = b.add_input("q_weight", &[EMBED_DIM, EMBED_DIM]);
    let k_w = b.add_input("k_weight", &[EMBED_DIM, EMBED_DIM]);
    let v_w = b.add_input("v_weight", &[EMBED_DIM, EMBED_DIM]);
    let out_w = b.add_input("out_weight", &[EMBED_DIM, EMBED_DIM]);
    let ffn1_w = b.add_input("ffn1_weight", &[FFN_DIM, EMBED_DIM]);
    let ffn2_w = b.add_input("ffn2_weight", &[EMBED_DIM, FFN_DIM]);

    // Patch projection + position embedding
    let embedded = b.add_linear(patches, proj_w, Some(proj_b), &shape);
    let x = b.add_binary_add(embedded, pos_embed, &shape);

    // Encoder block
    let config = TransformerBlockConfig {
        num_heads: NUM_HEADS,
        mask: AttentionMask::Standard,
        ffn_hidden_dim: FFN_DIM,
    };
    let weights = TransformerBlockWeights {
        ln1_weight: ln1_w,
        ln1_bias: ln1_b,
        ln2_weight: ln2_w,
        ln2_bias: ln2_b,
        q_weight: q_w,
        k_weight: k_w,
        v_weight: v_w,
        out_weight: out_w,
        ffn1_weight: ffn1_w,
        ffn2_weight: ffn2_w,
        eps,
    };
    let out = b
        .add_transformer_block(x, &weights, &config)
        .expect("valid block");

    b.build(out).expect("valid frontend+block kernel")
}

fn frontend_block_bindings() -> Vec<TensorParamBinding> {
    let wp = w(&[EMBED_DIM, EMBED_DIM]);
    let ln_w = ones(&[EMBED_DIM]);
    let ln_b = zeros(&[EMBED_DIM]);
    vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(w(&[EMBED_DIM, PATCH_DIM])),
        TensorParamBinding::ConstantTensor(zeros(&[EMBED_DIM])),
        TensorParamBinding::ConstantTensor(w(&[SEQ_LEN, EMBED_DIM])),
        TensorParamBinding::ConstantScalar(1e-5),
        TensorParamBinding::ConstantTensor(ln_w.clone()),
        TensorParamBinding::ConstantTensor(ln_b.clone()),
        TensorParamBinding::ConstantTensor(ln_w),
        TensorParamBinding::ConstantTensor(ln_b),
        TensorParamBinding::ConstantTensor(wp.clone()),
        TensorParamBinding::ConstantTensor(wp.clone()),
        TensorParamBinding::ConstantTensor(wp.clone()),
        TensorParamBinding::ConstantTensor(wp),
        TensorParamBinding::ConstantTensor(w(&[FFN_DIM, EMBED_DIM])),
        TensorParamBinding::ConstantTensor(w(&[EMBED_DIM, FFN_DIM])),
    ]
}

#[test]
fn test_granite_docling_siglip2_frontend_block_ibp() {
    let def = build_frontend_block_kernel();
    let bindings = frontend_block_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");
    let input = BoundedTensor::new(
        ArrayD::from_elem(IxDyn(&[SEQ_LEN, PATCH_DIM]), 0.0f32),
        ArrayD::from_elem(IxDyn(&[SEQ_LEN, PATCH_DIM]), 1.0f32),
    )
    .expect("bounds");

    let output = graph.propagate_ibp(&input).expect("IBP");
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("frontend+block IBP: [{lo_min}, {hi_max}]");
    assert!(lo_min.is_finite() && hi_max.is_finite());
}

#[test]
fn test_granite_docling_siglip2_frontend_block_crown() {
    let def = build_frontend_block_kernel();
    let bindings = frontend_block_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");
    let input = BoundedTensor::new(
        ArrayD::from_elem(IxDyn(&[SEQ_LEN, PATCH_DIM]), 0.0f32),
        ArrayD::from_elem(IxDyn(&[SEQ_LEN, PATCH_DIM]), 1.0f32),
    )
    .expect("bounds");

    let (method, output, fallback) = assert_crown_tighter_when_not_fallback(&graph, &input);
    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("frontend+block CROWN ({method:?}): [{lo_min}, {hi_max}]");
    if let Some(r) = &fallback {
        eprintln!("Fallback: {r}");
    }
}

// ===========================================================================
// 5. Two-block encoder stack
// ===========================================================================

/// Helper: add N transformer blocks to a builder, returning the output node.
fn add_n_blocks(
    b: &mut TensorBlockBuilder,
    mut x: TensorNodeId,
    n: usize,
    eps: TensorNodeId,
    bindings: &mut Vec<TensorParamBinding>,
) -> TensorNodeId {
    let config = TransformerBlockConfig {
        num_heads: NUM_HEADS,
        mask: AttentionMask::Standard,
        ffn_hidden_dim: FFN_DIM,
    };

    let wp = w(&[EMBED_DIM, EMBED_DIM]);
    let ln_w_arr = ones(&[EMBED_DIM]);
    let ln_b_arr = zeros(&[EMBED_DIM]);
    let ffn1 = w(&[FFN_DIM, EMBED_DIM]);
    let ffn2 = w(&[EMBED_DIM, FFN_DIM]);

    for i in 0..n {
        let ln1_w = b.add_input(&format!("b{i}_ln1_w"), &[EMBED_DIM]);
        let ln1_b = b.add_input(&format!("b{i}_ln1_b"), &[EMBED_DIM]);
        let ln2_w = b.add_input(&format!("b{i}_ln2_w"), &[EMBED_DIM]);
        let ln2_b = b.add_input(&format!("b{i}_ln2_b"), &[EMBED_DIM]);
        let q_w = b.add_input(&format!("b{i}_q_w"), &[EMBED_DIM, EMBED_DIM]);
        let k_w = b.add_input(&format!("b{i}_k_w"), &[EMBED_DIM, EMBED_DIM]);
        let v_w = b.add_input(&format!("b{i}_v_w"), &[EMBED_DIM, EMBED_DIM]);
        let out_w = b.add_input(&format!("b{i}_out_w"), &[EMBED_DIM, EMBED_DIM]);
        let ffn1_w = b.add_input(&format!("b{i}_ffn1_w"), &[FFN_DIM, EMBED_DIM]);
        let ffn2_w = b.add_input(&format!("b{i}_ffn2_w"), &[EMBED_DIM, FFN_DIM]);

        let weights = TransformerBlockWeights {
            ln1_weight: ln1_w,
            ln1_bias: ln1_b,
            ln2_weight: ln2_w,
            ln2_bias: ln2_b,
            q_weight: q_w,
            k_weight: k_w,
            v_weight: v_w,
            out_weight: out_w,
            ffn1_weight: ffn1_w,
            ffn2_weight: ffn2_w,
            eps,
        };
        x = b
            .add_transformer_block(x, &weights, &config)
            .unwrap_or_else(|e| panic!("block {i}: {e}"));

        // Push bindings for this block (10 weight params)
        bindings.push(TensorParamBinding::ConstantTensor(ln_w_arr.clone()));
        bindings.push(TensorParamBinding::ConstantTensor(ln_b_arr.clone()));
        bindings.push(TensorParamBinding::ConstantTensor(ln_w_arr.clone()));
        bindings.push(TensorParamBinding::ConstantTensor(ln_b_arr.clone()));
        bindings.push(TensorParamBinding::ConstantTensor(wp.clone()));
        bindings.push(TensorParamBinding::ConstantTensor(wp.clone()));
        bindings.push(TensorParamBinding::ConstantTensor(wp.clone()));
        bindings.push(TensorParamBinding::ConstantTensor(wp.clone()));
        bindings.push(TensorParamBinding::ConstantTensor(ffn1.clone()));
        bindings.push(TensorParamBinding::ConstantTensor(ffn2.clone()));
    }

    x
}

/// Build a two-block encoder stack.
fn build_two_block_kernel() -> (TensorKernelDef, Vec<TensorParamBinding>) {
    let mut b = TensorBlockBuilder::new("granite_docling_siglip2_two_block");
    let input = b.add_input("x", &[SEQ_LEN, EMBED_DIM]);
    let eps = b.add_input("eps", &[1]);

    let mut bindings = vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantScalar(1e-5),
    ];

    let out = add_n_blocks(&mut b, input, 2, eps, &mut bindings);
    let def = b.build(out).expect("valid two-block kernel");
    (def, bindings)
}

#[test]
fn test_granite_docling_siglip2_two_block_ibp() {
    let (def, bindings) = build_two_block_kernel();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");
    let input = uniform_bounds(&[SEQ_LEN, EMBED_DIM], 1.0);

    let output = graph.propagate_ibp(&input).expect("IBP");
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("two-block IBP: [{lo_min}, {hi_max}]");
    assert!(lo_min.is_finite() && hi_max.is_finite());
}

#[test]
fn test_granite_docling_siglip2_two_block_crown() {
    let (def, bindings) = build_two_block_kernel();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");
    let input = uniform_bounds(&[SEQ_LEN, EMBED_DIM], 1.0);

    let (method, output, fallback) = assert_crown_tighter_when_not_fallback(&graph, &input);
    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("two-block CROWN ({method:?}): [{lo_min}, {hi_max}]");
    if let Some(r) = &fallback {
        eprintln!("Fallback: {r}");
    }
}

/// Widening analysis: compare 1-block vs 2-block bounds width.
#[test]
fn test_granite_docling_siglip2_two_block_widening_analysis() {
    let input = uniform_bounds(&[SEQ_LEN, EMBED_DIM], 1.0);

    // 1-block
    let mut b1 = TensorBlockBuilder::new("siglip2_1block_widening");
    let inp1 = b1.add_input("x", &[SEQ_LEN, EMBED_DIM]);
    let eps1 = b1.add_input("eps", &[1]);
    let mut bindings1 = vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantScalar(1e-5),
    ];
    let out1 = add_n_blocks(&mut b1, inp1, 1, eps1, &mut bindings1);
    let def1 = b1.build(out1).expect("1-block");
    let graph1 = tensor_kernel_to_graph(&def1, &bindings1).expect("graph");
    let output1 = graph1.propagate_ibp(&input).expect("IBP 1-block");
    let (lo1, hi1) = bounds_min_max(&output1);
    let width1 = hi1 - lo1;

    // 2-block
    let (def2, bindings2) = build_two_block_kernel();
    let graph2 = tensor_kernel_to_graph(&def2, &bindings2).expect("graph");
    let output2 = graph2.propagate_ibp(&input).expect("IBP 2-block");
    let (lo2, hi2) = bounds_min_max(&output2);
    let width2 = hi2 - lo2;

    eprintln!("Widening analysis: 1-block width={width1:.4}, 2-block width={width2:.4}");
    eprintln!("  1-block: [{lo1:.4}, {hi1:.4}]");
    eprintln!("  2-block: [{lo2:.4}, {hi2:.4}]");

    // Both must be finite; 2-block should be at least as wide (monotone widening)
    assert!(width1.is_finite(), "1-block width not finite");
    assert!(width2.is_finite(), "2-block width not finite");
}

// ===========================================================================
// 6. Encoder + post-LayerNorm
// ===========================================================================

/// Build two encoder blocks + post-LayerNorm.
fn build_encoder_post_ln_kernel() -> (TensorKernelDef, Vec<TensorParamBinding>) {
    let mut b = TensorBlockBuilder::new("granite_docling_siglip2_encoder_post_ln");
    let shape = [SEQ_LEN, EMBED_DIM];

    let input = b.add_input("x", &shape);
    let eps = b.add_input("eps", &[1]);

    let mut bindings = vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantScalar(1e-5),
    ];

    let x = add_n_blocks(&mut b, input, 2, eps, &mut bindings);

    // Post-LayerNorm
    let post_ln_w = b.add_input("post_ln_w", &[EMBED_DIM]);
    let post_ln_b = b.add_input("post_ln_b", &[EMBED_DIM]);
    let out = b.add_layer_norm(x, eps, 1, post_ln_w, post_ln_b, &shape);

    bindings.push(TensorParamBinding::ConstantTensor(ones(&[EMBED_DIM])));
    bindings.push(TensorParamBinding::ConstantTensor(zeros(&[EMBED_DIM])));

    let def = b.build(out).expect("valid encoder+post-LN kernel");
    (def, bindings)
}

#[test]
fn test_granite_docling_siglip2_encoder_post_ln_ibp() {
    let (def, bindings) = build_encoder_post_ln_kernel();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");
    let input = uniform_bounds(&[SEQ_LEN, EMBED_DIM], 1.0);

    let output = graph.propagate_ibp(&input).expect("IBP");
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("encoder+post-LN IBP: [{lo_min}, {hi_max}]");
    assert!(lo_min.is_finite() && hi_max.is_finite());
}

// ===========================================================================
// 7. Encoder + post-LN + mean pooling
// ===========================================================================

/// Build two blocks + post-LN + mean pooling over sequence dimension.
fn build_encoder_pool_kernel() -> (TensorKernelDef, Vec<TensorParamBinding>) {
    let mut b = TensorBlockBuilder::new("granite_docling_siglip2_encoder_pool");
    let shape = [SEQ_LEN, EMBED_DIM];

    let input = b.add_input("x", &shape);
    let eps = b.add_input("eps", &[1]);

    let mut bindings = vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantScalar(1e-5),
    ];

    let x = add_n_blocks(&mut b, input, 2, eps, &mut bindings);

    // Post-LayerNorm
    let post_ln_w = b.add_input("post_ln_w", &[EMBED_DIM]);
    let post_ln_b = b.add_input("post_ln_b", &[EMBED_DIM]);
    let normed = b.add_layer_norm(x, eps, 1, post_ln_w, post_ln_b, &shape);
    bindings.push(TensorParamBinding::ConstantTensor(ones(&[EMBED_DIM])));
    bindings.push(TensorParamBinding::ConstantTensor(zeros(&[EMBED_DIM])));

    // Mean pooling over sequence dimension (axis=0): [SEQ_LEN, EMBED_DIM] -> [1, EMBED_DIM]
    let out = b.add_reduce(normed, ReduceOp::Mean, 0, true, &[1, EMBED_DIM]);

    let def = b.build(out).expect("valid encoder+pool kernel");
    (def, bindings)
}

#[test]
fn test_granite_docling_siglip2_encoder_pool_ibp() {
    let (def, bindings) = build_encoder_pool_kernel();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");
    let input = uniform_bounds(&[SEQ_LEN, EMBED_DIM], 1.0);

    let output = graph.propagate_ibp(&input).expect("IBP");
    assert_bounds_valid(&output);

    let (lo, _) = output.lower_upper();
    assert_eq!(lo.shape(), &[1, EMBED_DIM], "pooled shape mismatch");

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("encoder+pool IBP: [{lo_min}, {hi_max}]");
    assert!(lo_min.is_finite() && hi_max.is_finite());
}

#[test]
fn test_granite_docling_siglip2_encoder_pool_crown() {
    let (def, bindings) = build_encoder_pool_kernel();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");
    let input = uniform_bounds(&[SEQ_LEN, EMBED_DIM], 1.0);

    let (method, output, fallback) = assert_crown_tighter_when_not_fallback(&graph, &input);
    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("encoder+pool CROWN ({method:?}): [{lo_min}, {hi_max}]");
    if let Some(r) = &fallback {
        eprintln!("Fallback: {r}");
    }
}

// ===========================================================================
// 8. Full encoder + vision projection (Granite-Docling bridge)
// ===========================================================================

/// Build two blocks + post-LN + mean pool + vision projection Linear.
///
/// This is the complete SigLIP2-to-Granite bridge: the vision encoder output
/// is pooled and then projected to the LM embedding dimension via a Linear layer.
fn build_full_vproj_kernel() -> (TensorKernelDef, Vec<TensorParamBinding>) {
    let mut b = TensorBlockBuilder::new("granite_docling_siglip2_full_vproj");
    let shape = [SEQ_LEN, EMBED_DIM];

    let input = b.add_input("x", &shape);
    let eps = b.add_input("eps", &[1]);

    let mut bindings = vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantScalar(1e-5),
    ];

    let x = add_n_blocks(&mut b, input, 2, eps, &mut bindings);

    // Post-LayerNorm
    let post_ln_w = b.add_input("post_ln_w", &[EMBED_DIM]);
    let post_ln_b = b.add_input("post_ln_b", &[EMBED_DIM]);
    let normed = b.add_layer_norm(x, eps, 1, post_ln_w, post_ln_b, &shape);
    bindings.push(TensorParamBinding::ConstantTensor(ones(&[EMBED_DIM])));
    bindings.push(TensorParamBinding::ConstantTensor(zeros(&[EMBED_DIM])));

    // Mean pooling
    let pooled = b.add_reduce(normed, ReduceOp::Mean, 0, true, &[1, EMBED_DIM]);

    // Vision projection: Linear(EMBED_DIM -> LM_DIM)
    let vp_w = b.add_input("vproj_weight", &[LM_DIM, EMBED_DIM]);
    let vp_b = b.add_input("vproj_bias", &[LM_DIM]);
    let out = b.add_linear(pooled, vp_w, Some(vp_b), &[1, LM_DIM]);
    bindings.push(TensorParamBinding::ConstantTensor(w(&[LM_DIM, EMBED_DIM])));
    bindings.push(TensorParamBinding::ConstantTensor(zeros(&[LM_DIM])));

    let def = b.build(out).expect("valid full+vproj kernel");
    (def, bindings)
}

#[test]
fn test_granite_docling_siglip2_full_vproj_ibp() {
    let (def, bindings) = build_full_vproj_kernel();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");
    let input = uniform_bounds(&[SEQ_LEN, EMBED_DIM], 1.0);

    let output = graph.propagate_ibp(&input).expect("IBP");
    assert_bounds_valid(&output);

    let (lo, _) = output.lower_upper();
    assert_eq!(lo.shape(), &[1, LM_DIM], "vproj output shape mismatch");

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("full+vproj IBP: [{lo_min}, {hi_max}]");
    assert!(lo_min.is_finite() && hi_max.is_finite());
}

#[test]
fn test_granite_docling_siglip2_full_vproj_crown() {
    let (def, bindings) = build_full_vproj_kernel();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");
    let input = uniform_bounds(&[SEQ_LEN, EMBED_DIM], 1.0);

    let (method, output, fallback) = assert_crown_tighter_when_not_fallback(&graph, &input);
    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("full+vproj CROWN ({method:?}): [{lo_min}, {hi_max}]");
    if let Some(r) = &fallback {
        eprintln!("Fallback: {r}");
    }
}

#[test]
fn test_granite_docling_siglip2_full_vproj_bounds_finite() {
    let (def, bindings) = build_full_vproj_kernel();
    let input = uniform_bounds(&[SEQ_LEN, EMBED_DIM], 1.0);

    let result = verify_and_assert(
        &def,
        &bindings,
        &input,
        "granite_docling_siglip2_full_vproj",
    );

    let (lo_arr, hi_arr) = result.output_bounds.lower_upper();
    for &v in lo_arr.iter() {
        assert!(v.is_finite(), "lower bound not finite: {v}");
    }
    for &v in hi_arr.iter() {
        assert!(v.is_finite(), "upper bound not finite: {v}");
    }
    assert_eq!(lo_arr.shape(), &[1, LM_DIM]);
}

// ===========================================================================
// 9. Three-block encoder stack
// ===========================================================================

/// Build a three-block encoder stack for deeper widening analysis.
fn build_three_block_kernel() -> (TensorKernelDef, Vec<TensorParamBinding>) {
    let mut b = TensorBlockBuilder::new("granite_docling_siglip2_three_block");
    let input = b.add_input("x", &[SEQ_LEN, EMBED_DIM]);
    let eps = b.add_input("eps", &[1]);

    let mut bindings = vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantScalar(1e-5),
    ];

    let out = add_n_blocks(&mut b, input, 3, eps, &mut bindings);
    let def = b.build(out).expect("valid three-block kernel");
    (def, bindings)
}

#[test]
fn test_granite_docling_siglip2_three_block_ibp() {
    let (def, bindings) = build_three_block_kernel();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");
    let input = uniform_bounds(&[SEQ_LEN, EMBED_DIM], 1.0);

    let output = graph.propagate_ibp(&input).expect("IBP");
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("three-block IBP: [{lo_min}, {hi_max}]");
    assert!(lo_min.is_finite() && hi_max.is_finite());
}

#[test]
fn test_granite_docling_siglip2_three_block_crown() {
    let (def, bindings) = build_three_block_kernel();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");
    let input = uniform_bounds(&[SEQ_LEN, EMBED_DIM], 1.0);

    let (method, output, fallback) = assert_crown_tighter_when_not_fallback(&graph, &input);
    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("three-block CROWN ({method:?}): [{lo_min}, {hi_max}]");
    if let Some(r) = &fallback {
        eprintln!("Fallback: {r}");
    }
}

// ===========================================================================
// 10. Tight-input SiGLU FFN (narrow bounds for CROWN precision)
// ===========================================================================

/// Build SiGLU FFN with narrow input bounds (+-0.1) to exercise CROWN precision
/// on the multiplicative gate. Narrow bounds reduce the relaxation gap in
/// sigmoid linearization, allowing CROWN to produce tighter results.
fn build_tight_siglu_ffn_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("granite_docling_siglip2_tight_siglu");
    let shape = [SEQ_LEN, EMBED_DIM];
    let ffn_shape = [SEQ_LEN, FFN_DIM];

    let input = b.add_input("x", &shape);
    let gate_w = b.add_input("gate_proj", &[FFN_DIM, EMBED_DIM]);
    let up_w = b.add_input("up_proj", &[FFN_DIM, EMBED_DIM]);
    let down_w = b.add_input("down_proj", &[EMBED_DIM, FFN_DIM]);

    // SiGLU: silu(gate_proj(x)) * up_proj(x)
    let gate = b.add_linear(input, gate_w, None, &ffn_shape);
    let gate_sig = b.add_sigmoid(gate, &ffn_shape);
    let gate_activated = b.add_binary_mul(gate, gate_sig, &ffn_shape);
    let up = b.add_linear(input, up_w, None, &ffn_shape);
    let hidden = b.add_binary_mul(gate_activated, up, &ffn_shape);
    let out = b.add_linear(hidden, down_w, None, &shape);

    b.build(out).expect("valid tight SiGLU kernel")
}

fn tight_siglu_bindings() -> Vec<TensorParamBinding> {
    vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(w(&[FFN_DIM, EMBED_DIM])),
        TensorParamBinding::ConstantTensor(w(&[FFN_DIM, EMBED_DIM])),
        TensorParamBinding::ConstantTensor(w(&[EMBED_DIM, FFN_DIM])),
    ]
}

#[test]
fn test_granite_docling_siglip2_tight_siglu_ibp() {
    let def = build_tight_siglu_ffn_kernel();
    let bindings = tight_siglu_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");
    // Narrow input: +-0.1
    let input = uniform_bounds(&[SEQ_LEN, EMBED_DIM], 0.1);

    let output = graph.propagate_ibp(&input).expect("IBP");
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("tight SiGLU IBP (+-0.1): [{lo_min}, {hi_max}]");
    assert!(lo_min.is_finite() && hi_max.is_finite());

    // With narrow input + small weights, bounds should be tight
    let width = hi_max - lo_min;
    eprintln!("tight SiGLU IBP width: {width:.6}");
}

#[test]
fn test_granite_docling_siglip2_tight_siglu_crown() {
    let def = build_tight_siglu_ffn_kernel();
    let bindings = tight_siglu_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");
    let input = uniform_bounds(&[SEQ_LEN, EMBED_DIM], 0.1);

    let (method, output, fallback) = assert_crown_tighter_when_not_fallback(&graph, &input);
    let (lo_min, hi_max) = bounds_min_max(&output);
    let width = hi_max - lo_min;
    eprintln!("tight SiGLU CROWN ({method:?}): [{lo_min}, {hi_max}], width={width:.6}");
    if let Some(r) = &fallback {
        eprintln!("Fallback: {r}");
    }
}

// ===========================================================================
// 11. Full Granite-Docling pipeline compose
//     patch embed -> 2 encoder blocks -> post-LN -> mean pool ->
//     vision projection -> decoder FFN (RMSNorm + SiGLU) -> LM head -> softmax
// ===========================================================================

/// Full pipeline dimensions (task-specified, small for fast verification).
const FULL_VISION_DIM: usize = 16;
const FULL_LM_DIM: usize = 16;
const FULL_VOCAB_SIZE: usize = 32;
const FULL_SEQ_LEN: usize = 4;
const FULL_PATCH_SIZE: usize = 2;
const FULL_NUM_HEADS: usize = 4;
/// Patch input channels * patch_size^2 (RGB patches flattened).
const FULL_PATCH_DIM: usize = 3 * FULL_PATCH_SIZE * FULL_PATCH_SIZE; // 12
/// FFN intermediate dim (4x vision dim per ViT spec).
const FULL_FFN_DIM: usize = FULL_VISION_DIM * 4; // 64
/// Decoder FFN intermediate dim.
const FULL_DECODER_FFN_DIM: usize = FULL_LM_DIM * 4; // 64

/// Build the full Granite-Docling pipeline:
///
/// Patch embedding (Linear + pos embed) ->
/// Encoder block 1 (LN -> MHA -> residual -> LN -> SiGLU FFN -> residual) ->
/// Encoder block 2 (same) ->
/// Post-LayerNorm ->
/// Mean pooling (axis=0) ->
/// Vision projection Linear(VISION_DIM -> LM_DIM) ->
/// Decoder FFN (RMSNorm -> gate_proj -> SiLU -> mul(up_proj) -> down_proj) ->
/// LM head Linear(LM_DIM -> VOCAB_SIZE) ->
/// Softmax
fn build_full_pipeline_kernel() -> (TensorKernelDef, Vec<TensorParamBinding>) {
    let mut b = TensorBlockBuilder::new("granite_docling_full_pipeline");
    let enc_shape = [FULL_SEQ_LEN, FULL_VISION_DIM];

    // --- Patch embedding ---
    let patches = b.add_input("patches", &[FULL_SEQ_LEN, FULL_PATCH_DIM]);
    let patch_proj_w = b.add_input("patch_proj_w", &[FULL_VISION_DIM, FULL_PATCH_DIM]);
    let patch_proj_b = b.add_input("patch_proj_b", &[FULL_VISION_DIM]);
    let pos_embed = b.add_input("pos_embed", &[FULL_SEQ_LEN, FULL_VISION_DIM]);

    let embedded = b.add_linear(patches, patch_proj_w, Some(patch_proj_b), &enc_shape);
    let x = b.add_binary_add(embedded, pos_embed, &enc_shape);

    // --- Shared eps for all norms ---
    let eps = b.add_input("eps", &[1]);

    let mut bindings = vec![
        TensorParamBinding::Variable, // patches
        TensorParamBinding::ConstantTensor(w(&[FULL_VISION_DIM, FULL_PATCH_DIM])),
        TensorParamBinding::ConstantTensor(zeros(&[FULL_VISION_DIM])),
        TensorParamBinding::ConstantTensor(w(&[FULL_SEQ_LEN, FULL_VISION_DIM])),
        TensorParamBinding::ConstantScalar(1e-5),
    ];

    // --- 2 encoder blocks via add_n_blocks helper ---
    // add_n_blocks uses the file-level EMBED_DIM/FFN_DIM/NUM_HEADS which match
    // FULL_VISION_DIM/FULL_FFN_DIM/FULL_NUM_HEADS (all 16/64/4).
    let x = add_n_blocks(&mut b, x, 2, eps, &mut bindings);

    // --- Post-LayerNorm ---
    let post_ln_w = b.add_input("post_ln_w", &[FULL_VISION_DIM]);
    let post_ln_b = b.add_input("post_ln_b", &[FULL_VISION_DIM]);
    let normed = b.add_layer_norm(x, eps, 1, post_ln_w, post_ln_b, &enc_shape);
    bindings.push(TensorParamBinding::ConstantTensor(ones(&[FULL_VISION_DIM])));
    bindings.push(TensorParamBinding::ConstantTensor(zeros(&[
        FULL_VISION_DIM,
    ])));

    // --- Mean pooling over sequence dimension (axis=0) ---
    let pooled = b.add_reduce(normed, ReduceOp::Mean, 0, true, &[1, FULL_VISION_DIM]);

    // --- Vision projection: Linear(VISION_DIM -> LM_DIM) ---
    let vp_w = b.add_input("vproj_w", &[FULL_LM_DIM, FULL_VISION_DIM]);
    let vp_b = b.add_input("vproj_b", &[FULL_LM_DIM]);
    let projected = b.add_linear(pooled, vp_w, Some(vp_b), &[1, FULL_LM_DIM]);
    bindings.push(TensorParamBinding::ConstantTensor(w(&[
        FULL_LM_DIM,
        FULL_VISION_DIM,
    ])));
    bindings.push(TensorParamBinding::ConstantTensor(zeros(&[FULL_LM_DIM])));

    // --- Decoder FFN: RMSNorm -> gate_proj -> SiLU -> mul(up_proj) -> down_proj ---
    let dec_shape = [1, FULL_LM_DIM];
    let dec_ffn_shape = [1, FULL_DECODER_FFN_DIM];

    let rms_w = b.add_input("rms_weight", &[FULL_LM_DIM]);
    let rms_normed = b.add_rms_norm(projected, eps, 1, rms_w, &dec_shape);
    bindings.push(TensorParamBinding::ConstantTensor(ones(&[FULL_LM_DIM])));

    let gate_w = b.add_input("dec_gate_proj", &[FULL_DECODER_FFN_DIM, FULL_LM_DIM]);
    let up_w = b.add_input("dec_up_proj", &[FULL_DECODER_FFN_DIM, FULL_LM_DIM]);
    let down_w = b.add_input("dec_down_proj", &[FULL_LM_DIM, FULL_DECODER_FFN_DIM]);

    // SiLU(gate) = gate * sigmoid(gate)
    let gate = b.add_linear(rms_normed, gate_w, None, &dec_ffn_shape);
    let gate_sig = b.add_sigmoid(gate, &dec_ffn_shape);
    let gate_activated = b.add_binary_mul(gate, gate_sig, &dec_ffn_shape);
    // up_proj
    let up = b.add_linear(rms_normed, up_w, None, &dec_ffn_shape);
    // SiGLU: silu(gate) * up
    let hidden = b.add_binary_mul(gate_activated, up, &dec_ffn_shape);
    let ffn_out = b.add_linear(hidden, down_w, None, &dec_shape);

    bindings.push(TensorParamBinding::ConstantTensor(w(&[
        FULL_DECODER_FFN_DIM,
        FULL_LM_DIM,
    ])));
    bindings.push(TensorParamBinding::ConstantTensor(w(&[
        FULL_DECODER_FFN_DIM,
        FULL_LM_DIM,
    ])));
    bindings.push(TensorParamBinding::ConstantTensor(w(&[
        FULL_LM_DIM,
        FULL_DECODER_FFN_DIM,
    ])));

    // --- LM head: Linear(LM_DIM -> VOCAB_SIZE) ---
    let lm_w = b.add_input("lm_head_w", &[FULL_VOCAB_SIZE, FULL_LM_DIM]);
    let lm_b = b.add_input("lm_head_b", &[FULL_VOCAB_SIZE]);
    let logits = b.add_linear(ffn_out, lm_w, Some(lm_b), &[1, FULL_VOCAB_SIZE]);
    bindings.push(TensorParamBinding::ConstantTensor(w(&[
        FULL_VOCAB_SIZE,
        FULL_LM_DIM,
    ])));
    bindings.push(TensorParamBinding::ConstantTensor(zeros(&[
        FULL_VOCAB_SIZE,
    ])));

    // --- Softmax output ---
    let out = b.add_softmax(logits, -1, &[1, FULL_VOCAB_SIZE]);

    let def = b.build(out).expect("valid full pipeline kernel");
    (def, bindings)
}

#[test]
fn test_granite_docling_full_pipeline_ibp() {
    let (def, bindings) = build_full_pipeline_kernel();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");
    let input = BoundedTensor::new(
        ArrayD::from_elem(IxDyn(&[FULL_SEQ_LEN, FULL_PATCH_DIM]), 0.0f32),
        ArrayD::from_elem(IxDyn(&[FULL_SEQ_LEN, FULL_PATCH_DIM]), 1.0f32),
    )
    .expect("bounds");

    let output = graph.propagate_ibp(&input).expect("IBP");
    assert_bounds_valid(&output);

    // Softmax output must be in [0, 1]
    let (lo, _hi) = output.lower_upper();
    assert_eq!(lo.shape(), &[1, FULL_VOCAB_SIZE], "output shape mismatch");

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("full pipeline IBP: [{lo_min}, {hi_max}]");
    assert!(lo_min.is_finite() && hi_max.is_finite());
    assert!(
        lo_min >= -1e-6,
        "softmax lower bound should be >= 0, got {lo_min}"
    );
    assert!(
        hi_max <= 1.0 + 1e-6,
        "softmax upper bound should be <= 1, got {hi_max}"
    );
}

#[test]
fn test_granite_docling_full_pipeline_crown() {
    let (def, bindings) = build_full_pipeline_kernel();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");
    let input = BoundedTensor::new(
        ArrayD::from_elem(IxDyn(&[FULL_SEQ_LEN, FULL_PATCH_DIM]), 0.0f32),
        ArrayD::from_elem(IxDyn(&[FULL_SEQ_LEN, FULL_PATCH_DIM]), 1.0f32),
    )
    .expect("bounds");

    let (method, output, fallback) = assert_crown_tighter_when_not_fallback(&graph, &input);
    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("full pipeline CROWN ({method:?}): [{lo_min}, {hi_max}]");
    if let Some(r) = &fallback {
        eprintln!("Fallback: {r}");
    }
    // Softmax structural invariant holds regardless of propagation method
    assert!(
        lo_min >= -1e-6,
        "softmax lower bound should be >= 0, got {lo_min}"
    );
    assert!(
        hi_max <= 1.0 + 1e-6,
        "softmax upper bound should be <= 1, got {hi_max}"
    );
}

#[test]
fn test_granite_docling_full_pipeline_verify_and_record() {
    let (def, bindings) = build_full_pipeline_kernel();
    let input = BoundedTensor::new(
        ArrayD::from_elem(IxDyn(&[FULL_SEQ_LEN, FULL_PATCH_DIM]), 0.0f32),
        ArrayD::from_elem(IxDyn(&[FULL_SEQ_LEN, FULL_PATCH_DIM]), 1.0f32),
    )
    .expect("bounds");

    let result = verify_and_assert(&def, &bindings, &input, "granite_docling_full_pipeline");

    let (lo_arr, hi_arr) = result.output_bounds.lower_upper();
    for &v in lo_arr.iter() {
        assert!(v.is_finite(), "lower bound not finite: {v}");
    }
    for &v in hi_arr.iter() {
        assert!(v.is_finite(), "upper bound not finite: {v}");
    }
    assert_eq!(lo_arr.shape(), &[1, FULL_VOCAB_SIZE]);

    // Softmax: all outputs in [0, 1]
    let (lo_min, hi_max) = bounds_min_max(&result.output_bounds);
    eprintln!("full pipeline verify: [{lo_min}, {hi_max}]");
    assert!(
        lo_min >= -1e-6,
        "softmax lower bound should be >= 0, got {lo_min}"
    );
    assert!(
        hi_max <= 1.0 + 1e-6,
        "softmax upper bound should be <= 1, got {hi_max}"
    );
}

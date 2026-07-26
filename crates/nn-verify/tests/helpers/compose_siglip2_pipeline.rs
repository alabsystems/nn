// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Pipeline-level NY composition tests for SigLIP2 vision encoder.
//!
//! These tests verify bounds propagation through multi-stage pipeline
//! compositions that reflect the actual SigLIP2 inference pipeline:
//!
//! 1. **Linear patch projection** — isolated Linear projection stage
//! 2. **Post-LayerNorm** — LayerNorm on encoder output (SigLIP2-specific)
//! 3. **Mean pooling** — reduce over sequence dimension (final output stage)
//! 4. **Multi-block stacking (3 blocks)** — bounds widening through deep stacks
//! 5. **Attention sub-block isolation** — MHA without FFN for targeted verification
//! 6. **Narrow-bounds SiGLU FFN** — SiGLU with tight input for precision analysis
//! 7. **Position embed + single block** — composition across stage boundary
//! 8. **Full pipeline with post-norm + mean pool** — end-to-end with all stages
//! 9. **Post-LayerNorm CROWN** — CROWN tightness through normalization
//! 10. **Multi-block CROWN** — CROWN through deep stacked blocks
//! 11. **Attention sub-block CROWN** — CROWN through MHA without FFN
//! 12. **Narrow SiGLU FFN CROWN** — CROWN precision on multiplicative gating
//! 13. **Position embed + block CROWN** — cross-stage CROWN propagation
//! 14. **Full pipeline CROWN** — end-to-end CROWN with all stages including pool
//! 15. **Head projection** — isolated final Linear projection sub-block
//! 16. **SiGLU FFN with residual** — full FFN sub-block (LN + SiGLU + residual)
//! 17. **Two-block widening analysis** — bounds expansion measurement between blocks
//! 18. **Mean pool CROWN** — CROWN through mean reduction
//!
//! Dimensions are small for fast verification (D=16, S=4, FFN=64, heads=4).
//!
//! Part of #3583: SigLIP2 pipeline compose verification.

use super::common::{
    assert_bounds_valid, assert_bounds_width, assert_crown_tighter_when_not_fallback,
    bounds_min_max, uniform_bounds, verify_and_assert,
};
use nn_dsl::tensor_block_builder::TensorBlockBuilder;
use nn_dsl::tensor_ir::TensorKernelDef;
use nn_dsl::{AttentionMask, ReduceOp, TransformerBlockConfig, TransformerBlockWeights};
use nn_verify::{
    tensor_kernel_to_graph, BoundedTensor, TensorParamBinding, VerificationSoundnessMode,
};
use ndarray::{ArrayD, IxDyn};

// ---------------------------------------------------------------------------
// Dimensions — small for fast verification, structurally representative
// ---------------------------------------------------------------------------

/// Number of image patches (2x2 grid).
const NUM_PATCHES: usize = 4;
/// Embedding/hidden dimension (tiny for fast bounds propagation).
const EMBED_DIM: usize = 16;
/// Number of attention heads.
const NUM_HEADS: usize = 4;
/// FFN intermediate dimension (4x embed_dim).
const FFN_DIM: usize = 64;
/// Patch input dimension (flattened pixel patch before linear projection).
const PATCH_DIM: usize = 48;
/// Head projection output dimension.
const HEAD_DIM: usize = 8;

// ---------------------------------------------------------------------------
// Builder helpers
// ---------------------------------------------------------------------------

/// Build an isolated linear patch projection kernel.
///
/// Input: `[NUM_PATCHES, PATCH_DIM]` (Variable, flattened patches).
/// Output: `[NUM_PATCHES, EMBED_DIM]`.
///
/// Models the patch embedding's Linear projection only (without Conv2d spatial
/// extraction). This isolates the linear transformation bounds behavior.
fn build_linear_patch_projection_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("siglip2_pipeline_linear_patch_proj");

    let patches = b.add_input("patches", &[NUM_PATCHES, PATCH_DIM]);
    let proj_w = b.add_input("proj_weight", &[EMBED_DIM, PATCH_DIM]);
    let proj_b = b.add_input("proj_bias", &[EMBED_DIM]);

    let out = b.add_linear(patches, proj_w, Some(proj_b), &[NUM_PATCHES, EMBED_DIM]);

    b.build(out).expect("valid linear patch projection kernel")
}

fn linear_patch_projection_bindings() -> Vec<TensorParamBinding> {
    vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[EMBED_DIM, PATCH_DIM]),
            0.02f32,
        )),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[EMBED_DIM]), 0.0f32)),
    ]
}

/// Build a post-LayerNorm kernel (SigLIP2 applies LayerNorm after all encoder blocks).
///
/// Input: `[NUM_PATCHES, EMBED_DIM]` (Variable, encoder output).
/// Output: `[NUM_PATCHES, EMBED_DIM]`.
fn build_post_layernorm_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("siglip2_pipeline_post_layernorm");

    let input = b.add_input("x", &[NUM_PATCHES, EMBED_DIM]);
    let eps = b.add_input("eps", &[1]);
    let ln_w = b.add_input("ln_weight", &[EMBED_DIM]);
    let ln_b = b.add_input("ln_bias", &[EMBED_DIM]);

    let out = b.add_layer_norm(input, eps, 1, ln_w, ln_b, &[NUM_PATCHES, EMBED_DIM]);

    b.build(out).expect("valid post-LayerNorm kernel")
}

fn post_layernorm_bindings() -> Vec<TensorParamBinding> {
    vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantScalar(1e-6),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[EMBED_DIM]), 1.0f32)),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[EMBED_DIM]), 0.0f32)),
    ]
}

/// Build a mean pooling kernel (reduces sequence dimension).
///
/// Input: `[NUM_PATCHES, EMBED_DIM]` (Variable, sequence of patch embeddings).
/// Output: `[1, EMBED_DIM]` (mean-pooled, keepdim=true).
///
/// SigLIP2 uses mean pooling (no CLS token) to produce the image-level
/// representation.
fn build_mean_pool_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("siglip2_pipeline_mean_pool");

    let input = b.add_input("x", &[NUM_PATCHES, EMBED_DIM]);

    let out = b.add_reduce(input, ReduceOp::Mean, 0, true, &[1, EMBED_DIM]);

    b.build(out).expect("valid mean pool kernel")
}

fn mean_pool_bindings() -> Vec<TensorParamBinding> {
    vec![TensorParamBinding::Variable]
}

/// Build a 3-block stacked encoder kernel for multi-block bounds propagation.
///
/// Input: `[NUM_PATCHES, EMBED_DIM]` (Variable).
/// Output: `[NUM_PATCHES, EMBED_DIM]`.
///
/// Tests how bounds widen through multiple stacked transformer blocks.
/// Uses GELU FFN (standard `add_transformer_block`) for all 3 blocks.
fn build_three_block_stack_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("siglip2_pipeline_three_block_stack");

    let input = b.add_input("x", &[NUM_PATCHES, EMBED_DIM]);
    let eps = b.add_input("eps", &[1]);

    let config = TransformerBlockConfig {
        num_heads: NUM_HEADS,
        mask: AttentionMask::Standard,
        ffn_hidden_dim: FFN_DIM,
    };

    // Allocate all 3 blocks' weights
    let mut x = input;
    for i in 0..3 {
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
    }

    b.build(x).expect("valid 3-block stack kernel")
}

fn three_block_stack_bindings() -> Vec<TensorParamBinding> {
    let w_proj = ArrayD::from_elem(IxDyn(&[EMBED_DIM, EMBED_DIM]), 0.02f32);
    let w_ffn1 = ArrayD::from_elem(IxDyn(&[FFN_DIM, EMBED_DIM]), 0.02f32);
    let w_ffn2 = ArrayD::from_elem(IxDyn(&[EMBED_DIM, FFN_DIM]), 0.02f32);
    let ln_w = ArrayD::from_elem(IxDyn(&[EMBED_DIM]), 1.0f32);
    let ln_b = ArrayD::from_elem(IxDyn(&[EMBED_DIM]), 0.0f32);

    let mut bindings = vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantScalar(1e-6),
    ];

    // 3 blocks x 10 weights each
    for _ in 0..3 {
        bindings.push(TensorParamBinding::ConstantTensor(ln_w.clone()));
        bindings.push(TensorParamBinding::ConstantTensor(ln_b.clone()));
        bindings.push(TensorParamBinding::ConstantTensor(ln_w.clone()));
        bindings.push(TensorParamBinding::ConstantTensor(ln_b.clone()));
        bindings.push(TensorParamBinding::ConstantTensor(w_proj.clone()));
        bindings.push(TensorParamBinding::ConstantTensor(w_proj.clone()));
        bindings.push(TensorParamBinding::ConstantTensor(w_proj.clone()));
        bindings.push(TensorParamBinding::ConstantTensor(w_proj.clone()));
        bindings.push(TensorParamBinding::ConstantTensor(w_ffn1.clone()));
        bindings.push(TensorParamBinding::ConstantTensor(w_ffn2.clone()));
    }

    bindings
}

/// Build an attention sub-block kernel in isolation (no FFN).
///
/// Input: `[NUM_PATCHES, EMBED_DIM]` (Variable).
/// Output: `[NUM_PATCHES, EMBED_DIM]`.
///
/// Architecture: LayerNorm -> MHA -> residual.
/// Isolates the attention mechanism for targeted bounds analysis.
fn build_attention_subblock_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("siglip2_pipeline_attention_subblock");

    let input = b.add_input("x", &[NUM_PATCHES, EMBED_DIM]);
    let eps = b.add_input("eps", &[1]);
    let ln_w = b.add_input("ln_weight", &[EMBED_DIM]);
    let ln_b = b.add_input("ln_bias", &[EMBED_DIM]);
    let q_w = b.add_input("q_weight", &[EMBED_DIM, EMBED_DIM]);
    let k_w = b.add_input("k_weight", &[EMBED_DIM, EMBED_DIM]);
    let v_w = b.add_input("v_weight", &[EMBED_DIM, EMBED_DIM]);
    let out_w = b.add_input("out_weight", &[EMBED_DIM, EMBED_DIM]);

    // LayerNorm
    let normed = b.add_layer_norm(input, eps, 1, ln_w, ln_b, &[NUM_PATCHES, EMBED_DIM]);

    // Multi-head self-attention (bidirectional, no causal mask)
    let attn_out = b
        .add_multi_head_attention(
            normed,
            q_w,
            k_w,
            v_w,
            out_w,
            NUM_HEADS,
            AttentionMask::Standard,
            &[NUM_PATCHES, EMBED_DIM],
        )
        .expect("valid MHA");

    // Residual connection
    let out = b.add_binary_add(input, attn_out, &[NUM_PATCHES, EMBED_DIM]);

    b.build(out).expect("valid attention sub-block kernel")
}

fn attention_subblock_bindings() -> Vec<TensorParamBinding> {
    let w_proj = ArrayD::from_elem(IxDyn(&[EMBED_DIM, EMBED_DIM]), 0.02f32);

    vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantScalar(1e-6),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[EMBED_DIM]), 1.0f32)),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[EMBED_DIM]), 0.0f32)),
        TensorParamBinding::ConstantTensor(w_proj.clone()),
        TensorParamBinding::ConstantTensor(w_proj.clone()),
        TensorParamBinding::ConstantTensor(w_proj.clone()),
        TensorParamBinding::ConstantTensor(w_proj),
    ]
}

/// Build a SiGLU FFN kernel with narrow-bounds analysis.
///
/// Same structure as compose_siglip2.rs SiGLU FFN but with smaller dimensions
/// (D=16, FFN=64) for pipeline-level composition. Tests bound widening
/// behavior of the multiplicative gating under tighter input ranges.
///
/// Input: `[NUM_PATCHES, EMBED_DIM]` (Variable).
/// Output: `[NUM_PATCHES, EMBED_DIM]`.
fn build_narrow_siglu_ffn_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("siglip2_pipeline_narrow_siglu_ffn");

    let input = b.add_input("x", &[NUM_PATCHES, EMBED_DIM]);
    let gate_w = b.add_input("gate_weight", &[FFN_DIM, EMBED_DIM]);
    let gate_b = b.add_input("gate_bias", &[FFN_DIM]);
    let up_w = b.add_input("up_weight", &[FFN_DIM, EMBED_DIM]);
    let up_b = b.add_input("up_bias", &[FFN_DIM]);
    let down_w = b.add_input("down_weight", &[EMBED_DIM, FFN_DIM]);
    let down_b = b.add_input("down_bias", &[EMBED_DIM]);

    // Gate branch: Linear -> SiLU(x) = x * sigmoid(x)
    let gate = b.add_linear(input, gate_w, Some(gate_b), &[NUM_PATCHES, FFN_DIM]);
    let gate_sig = b.add_sigmoid(gate, &[NUM_PATCHES, FFN_DIM]);
    let gate_activated = b.add_binary_mul(gate, gate_sig, &[NUM_PATCHES, FFN_DIM]);

    // Up branch: Linear
    let up = b.add_linear(input, up_w, Some(up_b), &[NUM_PATCHES, FFN_DIM]);

    // Multiplicative gating
    let hidden = b.add_binary_mul(gate_activated, up, &[NUM_PATCHES, FFN_DIM]);

    // Down projection
    let out = b.add_linear(hidden, down_w, Some(down_b), &[NUM_PATCHES, EMBED_DIM]);

    b.build(out).expect("valid narrow SiGLU FFN kernel")
}

fn narrow_siglu_ffn_bindings() -> Vec<TensorParamBinding> {
    vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[FFN_DIM, EMBED_DIM]),
            0.02f32,
        )),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[FFN_DIM]), 0.0f32)),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[FFN_DIM, EMBED_DIM]),
            0.02f32,
        )),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[FFN_DIM]), 0.0f32)),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[EMBED_DIM, FFN_DIM]),
            0.02f32,
        )),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[EMBED_DIM]), 0.0f32)),
    ]
}

/// Build a position embedding + single encoder block composition.
///
/// Input: `[NUM_PATCHES, EMBED_DIM]` (Variable, pre-embedded patches).
/// Output: `[NUM_PATCHES, EMBED_DIM]`.
///
/// Tests the stage boundary between embedding and encoder: pos_embed addition
/// followed by one transformer block.
fn build_pos_embed_plus_block_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("siglip2_pipeline_pos_embed_block");

    let input = b.add_input("x", &[NUM_PATCHES, EMBED_DIM]);
    let pos_embed = b.add_input("pos_embed", &[NUM_PATCHES, EMBED_DIM]);
    let eps = b.add_input("eps", &[1]);

    // Position embedding addition
    let x = b.add_binary_add(input, pos_embed, &[NUM_PATCHES, EMBED_DIM]);

    // Single encoder block
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

    b.build(out).expect("valid pos_embed + block kernel")
}

fn pos_embed_plus_block_bindings() -> Vec<TensorParamBinding> {
    let w_proj = ArrayD::from_elem(IxDyn(&[EMBED_DIM, EMBED_DIM]), 0.02f32);
    let w_ffn1 = ArrayD::from_elem(IxDyn(&[FFN_DIM, EMBED_DIM]), 0.02f32);
    let w_ffn2 = ArrayD::from_elem(IxDyn(&[EMBED_DIM, FFN_DIM]), 0.02f32);

    vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[NUM_PATCHES, EMBED_DIM]),
            0.01f32,
        )),
        TensorParamBinding::ConstantScalar(1e-6),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[EMBED_DIM]), 1.0f32)),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[EMBED_DIM]), 0.0f32)),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[EMBED_DIM]), 1.0f32)),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[EMBED_DIM]), 0.0f32)),
        TensorParamBinding::ConstantTensor(w_proj.clone()),
        TensorParamBinding::ConstantTensor(w_proj.clone()),
        TensorParamBinding::ConstantTensor(w_proj.clone()),
        TensorParamBinding::ConstantTensor(w_proj),
        TensorParamBinding::ConstantTensor(w_ffn1),
        TensorParamBinding::ConstantTensor(w_ffn2),
    ]
}

/// Build the full SigLIP2 pipeline: proj + pos_embed + 2 blocks + post-LN + mean pool.
///
/// Input: `[NUM_PATCHES, PATCH_DIM]` (Variable, flattened patch pixels).
/// Output: `[1, EMBED_DIM]` (mean-pooled image representation).
///
/// This is the most comprehensive pipeline test, including every stage:
/// 1. Linear patch projection: [S, P] -> [S, D]
/// 2. Position embedding addition: [S, D] + [S, D]
/// 3. Encoder block 1: [S, D] -> [S, D]
/// 4. Encoder block 2: [S, D] -> [S, D]
/// 5. Post-LayerNorm: [S, D] -> [S, D]
/// 6. Mean pooling: [S, D] -> [1, D]
fn build_full_pipeline_with_pool_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("siglip2_pipeline_full_with_pool");

    let patches = b.add_input("patches", &[NUM_PATCHES, PATCH_DIM]);
    let proj_w = b.add_input("proj_weight", &[EMBED_DIM, PATCH_DIM]);
    let proj_b = b.add_input("proj_bias", &[EMBED_DIM]);
    let pos_embed = b.add_input("pos_embed", &[NUM_PATCHES, EMBED_DIM]);
    let eps = b.add_input("eps", &[1]);

    // Stage 1: Linear patch projection
    let embedded = b.add_linear(patches, proj_w, Some(proj_b), &[NUM_PATCHES, EMBED_DIM]);

    // Stage 2: Position embedding
    let x = b.add_binary_add(embedded, pos_embed, &[NUM_PATCHES, EMBED_DIM]);

    // Stage 3-4: Encoder blocks
    let config = TransformerBlockConfig {
        num_heads: NUM_HEADS,
        mask: AttentionMask::Standard,
        ffn_hidden_dim: FFN_DIM,
    };

    let mut curr = x;
    for i in 0..2 {
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

        curr = b
            .add_transformer_block(curr, &weights, &config)
            .unwrap_or_else(|e| panic!("block {i}: {e}"));
    }

    // Stage 5: Post-LayerNorm
    let post_ln_w = b.add_input("post_ln_w", &[EMBED_DIM]);
    let post_ln_b = b.add_input("post_ln_b", &[EMBED_DIM]);
    let normed = b.add_layer_norm(
        curr,
        eps,
        1,
        post_ln_w,
        post_ln_b,
        &[NUM_PATCHES, EMBED_DIM],
    );

    // Stage 6: Mean pooling over sequence dimension
    let out = b.add_reduce(normed, ReduceOp::Mean, 0, true, &[1, EMBED_DIM]);

    b.build(out).expect("valid full pipeline with pool kernel")
}

fn full_pipeline_with_pool_bindings() -> Vec<TensorParamBinding> {
    let proj_w = ArrayD::from_elem(IxDyn(&[EMBED_DIM, PATCH_DIM]), 0.02f32);
    let proj_b = ArrayD::from_elem(IxDyn(&[EMBED_DIM]), 0.0f32);
    let pos_embed = ArrayD::from_elem(IxDyn(&[NUM_PATCHES, EMBED_DIM]), 0.01f32);
    let w_proj = ArrayD::from_elem(IxDyn(&[EMBED_DIM, EMBED_DIM]), 0.02f32);
    let w_ffn1 = ArrayD::from_elem(IxDyn(&[FFN_DIM, EMBED_DIM]), 0.02f32);
    let w_ffn2 = ArrayD::from_elem(IxDyn(&[EMBED_DIM, FFN_DIM]), 0.02f32);
    let ln_w = ArrayD::from_elem(IxDyn(&[EMBED_DIM]), 1.0f32);
    let ln_b = ArrayD::from_elem(IxDyn(&[EMBED_DIM]), 0.0f32);

    let mut bindings = vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(proj_w),
        TensorParamBinding::ConstantTensor(proj_b),
        TensorParamBinding::ConstantTensor(pos_embed),
        TensorParamBinding::ConstantScalar(1e-6),
    ];

    // 2 blocks x 10 weights each
    for _ in 0..2 {
        bindings.push(TensorParamBinding::ConstantTensor(ln_w.clone()));
        bindings.push(TensorParamBinding::ConstantTensor(ln_b.clone()));
        bindings.push(TensorParamBinding::ConstantTensor(ln_w.clone()));
        bindings.push(TensorParamBinding::ConstantTensor(ln_b.clone()));
        bindings.push(TensorParamBinding::ConstantTensor(w_proj.clone()));
        bindings.push(TensorParamBinding::ConstantTensor(w_proj.clone()));
        bindings.push(TensorParamBinding::ConstantTensor(w_proj.clone()));
        bindings.push(TensorParamBinding::ConstantTensor(w_proj.clone()));
        bindings.push(TensorParamBinding::ConstantTensor(w_ffn1.clone()));
        bindings.push(TensorParamBinding::ConstantTensor(w_ffn2.clone()));
    }

    // Post-LayerNorm weights
    bindings.push(TensorParamBinding::ConstantTensor(ln_w));
    bindings.push(TensorParamBinding::ConstantTensor(ln_b));

    bindings
}

// ===========================================================================
// Tests
// ===========================================================================

// ---------------------------------------------------------------------------
// 1. Linear patch projection — IBP bounds through the projection stage
// ---------------------------------------------------------------------------

/// Linear patch projection IBP produces bounded output with [0, 1] patch input.
///
/// Linear layers are exact in IBP (no activation linearization needed), so
/// output width should be proportional to weight magnitude * input range.
#[test]
fn test_siglip2_pipeline_linear_patch_proj_ibp() {
    let def = build_linear_patch_projection_kernel();
    let bindings = linear_patch_projection_bindings();
    let input = BoundedTensor::new(
        ArrayD::from_elem(IxDyn(&[NUM_PATCHES, PATCH_DIM]), 0.0f32),
        ArrayD::from_elem(IxDyn(&[NUM_PATCHES, PATCH_DIM]), 1.0f32),
    )
    .expect("bounds");

    let result = verify_and_assert(
        &def,
        &bindings,
        &input,
        "siglip2_pipeline_linear_patch_proj",
    );
    assert_bounds_valid(&result.output_bounds);

    let (lo, _hi) = result.output_bounds.lower_upper();
    assert_eq!(lo.shape(), &[NUM_PATCHES, EMBED_DIM]);

    // Linear layer with w=0.02, input in [0,1]: output in [0, 0.02*48] = [0, 0.96].
    // IBP should be tight for linear layers.
    let (lo_min, hi_max) = bounds_min_max(&result.output_bounds);
    eprintln!("siglip2_pipeline linear patch proj IBP: [{lo_min:.4}, {hi_max:.4}]");
    assert_bounds_width(&result.output_bounds, 50.0, "linear patch proj");
}

// ---------------------------------------------------------------------------
// 2. Post-LayerNorm — IBP bounds after normalization
// ---------------------------------------------------------------------------

/// Post-LayerNorm IBP produces finite, valid bounds.
///
/// LayerNorm normalizes to near-zero mean and unit variance, but IBP
/// over-approximates due to the non-linear mean/variance computation.
/// With IbpValidated mode, bounds should remain finite.
#[test]
fn test_siglip2_pipeline_post_layernorm_ibp() {
    let def = build_post_layernorm_kernel();
    let bindings = post_layernorm_bindings();
    let input = uniform_bounds(&[NUM_PATCHES, EMBED_DIM], 1.0);

    let result = verify_and_assert(&def, &bindings, &input, "siglip2_pipeline_post_layernorm");
    assert_bounds_valid(&result.output_bounds);

    let (lo, _) = result.output_bounds.lower_upper();
    assert_eq!(lo.shape(), &[NUM_PATCHES, EMBED_DIM]);

    // LayerNorm produces Heuristic soundness mode
    assert_eq!(
        result.verification.soundness_mode,
        VerificationSoundnessMode::Heuristic,
        "post-LayerNorm should produce Heuristic mode"
    );

    let (lo_min, hi_max) = bounds_min_max(&result.output_bounds);
    eprintln!("siglip2_pipeline post-LayerNorm IBP: [{lo_min:.4}, {hi_max:.4}]");
}

// ---------------------------------------------------------------------------
// 3. Mean pooling — reduce dimension bounds
// ---------------------------------------------------------------------------

/// Mean pooling IBP correctly reduces sequence dimension.
///
/// Mean reduction over axis 0: output lower = mean of element-wise lowers
/// (since mean is monotone). Output shape should be [1, EMBED_DIM].
#[test]
fn test_siglip2_pipeline_mean_pool_ibp() {
    let def = build_mean_pool_kernel();
    let bindings = mean_pool_bindings();
    let input = uniform_bounds(&[NUM_PATCHES, EMBED_DIM], 1.0);

    let result = verify_and_assert(&def, &bindings, &input, "siglip2_pipeline_mean_pool");
    assert_bounds_valid(&result.output_bounds);

    let (lo, _hi) = result.output_bounds.lower_upper();
    assert_eq!(
        lo.shape(),
        &[1, EMBED_DIM],
        "mean pool output should be [1, {EMBED_DIM}]"
    );

    // Mean of uniform [-1, 1] bounds: each output element is mean of NUM_PATCHES
    // elements, all with same bounds [-1, 1], so output should be in [-1, 1].
    let (lo_min, hi_max) = bounds_min_max(&result.output_bounds);
    eprintln!("siglip2_pipeline mean pool IBP: [{lo_min:.4}, {hi_max:.4}]");
    assert!(
        lo_min >= -1.0 - 1e-4,
        "mean pool lower should be >= -1.0, got {lo_min}"
    );
    assert!(
        hi_max <= 1.0 + 1e-4,
        "mean pool upper should be <= 1.0, got {hi_max}"
    );
}

// ---------------------------------------------------------------------------
// 4. Multi-block stacking (3 blocks) — bounds propagation through depth
// ---------------------------------------------------------------------------

/// Three stacked transformer blocks produce finite IBP bounds.
///
/// Bounds widen through each block due to non-linear activations (GELU) and
/// normalization layers. With small (0.02) weights, widening should be bounded.
#[test]
fn test_siglip2_pipeline_three_block_stack_ibp() {
    let def = build_three_block_stack_kernel();
    let bindings = three_block_stack_bindings();
    let input = uniform_bounds(&[NUM_PATCHES, EMBED_DIM], 0.5);

    let result = verify_and_assert(
        &def,
        &bindings,
        &input,
        "siglip2_pipeline_three_block_stack",
    );
    assert_bounds_valid(&result.output_bounds);

    let (lo, _) = result.output_bounds.lower_upper();
    assert_eq!(lo.shape(), &[NUM_PATCHES, EMBED_DIM]);

    let (lo_min, hi_max) = bounds_min_max(&result.output_bounds);
    eprintln!("siglip2_pipeline 3-block stack IBP: [{lo_min:.4}, {hi_max:.4}]");

    // All bounds must be finite (not blown up through 3 blocks)
    assert!(
        lo_min.is_finite() && hi_max.is_finite(),
        "3-block stack bounds must be finite: [{lo_min}, {hi_max}]"
    );
}

// ---------------------------------------------------------------------------
// 5. Attention sub-block isolation — MHA bounds without FFN
// ---------------------------------------------------------------------------

/// Attention sub-block (LayerNorm + MHA + residual) IBP produces valid bounds.
///
/// Isolates the attention mechanism: LayerNorm normalizes input, MHA produces
/// attention-weighted output, residual adds original input. Tests the softmax
/// attention bounds without the complication of FFN activation.
#[test]
fn test_siglip2_pipeline_attention_subblock_ibp() {
    let def = build_attention_subblock_kernel();
    let bindings = attention_subblock_bindings();
    let input = uniform_bounds(&[NUM_PATCHES, EMBED_DIM], 1.0);

    let result = verify_and_assert(
        &def,
        &bindings,
        &input,
        "siglip2_pipeline_attention_subblock",
    );
    assert_bounds_valid(&result.output_bounds);

    let (lo, _) = result.output_bounds.lower_upper();
    assert_eq!(lo.shape(), &[NUM_PATCHES, EMBED_DIM]);

    let (lo_min, hi_max) = bounds_min_max(&result.output_bounds);
    eprintln!("siglip2_pipeline attention sub-block IBP: [{lo_min:.4}, {hi_max:.4}]");

    // Residual connection preserves input range, attention adds bounded shift
    assert!(
        lo_min.is_finite(),
        "attention sub-block lower must be finite"
    );
    assert!(
        hi_max.is_finite(),
        "attention sub-block upper must be finite"
    );
}

// ---------------------------------------------------------------------------
// 6. Narrow-bounds SiGLU FFN — tighter input for precision analysis
// ---------------------------------------------------------------------------

/// SiGLU FFN with narrow input bounds [-0.1, 0.1] for precision analysis.
///
/// With tight input bounds, the SiGLU multiplicative gating should produce
/// tight output bounds. sigmoid(small_x) ~ 0.5, so SiLU(x) ~ 0.5*x.
/// The multiplicative gate should not blow up bounds for small inputs.
#[test]
fn test_siglip2_pipeline_narrow_siglu_ffn_ibp() {
    let def = build_narrow_siglu_ffn_kernel();
    let bindings = narrow_siglu_ffn_bindings();
    // Narrow input: [-0.1, 0.1] (post-normalization scale)
    let input = uniform_bounds(&[NUM_PATCHES, EMBED_DIM], 0.1);

    let result = verify_and_assert(&def, &bindings, &input, "siglip2_pipeline_narrow_siglu_ffn");
    assert_bounds_valid(&result.output_bounds);

    let (lo, _) = result.output_bounds.lower_upper();
    assert_eq!(lo.shape(), &[NUM_PATCHES, EMBED_DIM]);

    let (lo_min, hi_max) = bounds_min_max(&result.output_bounds);
    eprintln!("siglip2_pipeline narrow SiGLU FFN IBP: [{lo_min:.4}, {hi_max:.4}]");

    // With small (0.02) weights and [-0.1, 0.1] input, output should be small
    assert_bounds_width(&result.output_bounds, 10.0, "narrow SiGLU FFN");
}

// ---------------------------------------------------------------------------
// 7. Position embedding + single block — stage boundary composition
// ---------------------------------------------------------------------------

/// Position embedding followed by one encoder block: cross-stage bounds.
///
/// Tests that bounds from embedding stage flow correctly through the first
/// encoder block. The pos_embed offset (0.01) should propagate as a shift.
#[test]
fn test_siglip2_pipeline_pos_embed_plus_block_ibp() {
    let def = build_pos_embed_plus_block_kernel();
    let bindings = pos_embed_plus_block_bindings();
    let input = uniform_bounds(&[NUM_PATCHES, EMBED_DIM], 1.0);

    let result = verify_and_assert(&def, &bindings, &input, "siglip2_pipeline_pos_embed_block");
    assert_bounds_valid(&result.output_bounds);

    let (lo, _) = result.output_bounds.lower_upper();
    assert_eq!(lo.shape(), &[NUM_PATCHES, EMBED_DIM]);

    let (lo_min, hi_max) = bounds_min_max(&result.output_bounds);
    eprintln!("siglip2_pipeline pos_embed + block IBP: [{lo_min:.4}, {hi_max:.4}]");

    // Heuristic mode due to LayerNorm in the transformer block
    assert_eq!(
        result.verification.soundness_mode,
        VerificationSoundnessMode::Heuristic,
        "pos_embed + block should produce Heuristic mode"
    );
}

// ---------------------------------------------------------------------------
// 8. Full pipeline with post-norm + mean pool — end-to-end output bounds
// ---------------------------------------------------------------------------

/// Full SigLIP2 pipeline (proj + pos + 2 blocks + post-LN + mean pool) IBP.
///
/// This is the most comprehensive test: all 6 pipeline stages composed.
/// Verifies that bounds propagate through the entire inference pipeline
/// and produce finite, valid output at the mean-pooled image representation.
#[test]
fn test_siglip2_pipeline_full_with_pool_ibp() {
    let def = build_full_pipeline_with_pool_kernel();
    let bindings = full_pipeline_with_pool_bindings();
    let input = BoundedTensor::new(
        ArrayD::from_elem(IxDyn(&[NUM_PATCHES, PATCH_DIM]), 0.0f32),
        ArrayD::from_elem(IxDyn(&[NUM_PATCHES, PATCH_DIM]), 1.0f32),
    )
    .expect("bounds");

    let result = verify_and_assert(&def, &bindings, &input, "siglip2_pipeline_full_with_pool");
    assert_bounds_valid(&result.output_bounds);

    let (lo_arr, hi_arr) = result.output_bounds.lower_upper();
    assert_eq!(
        lo_arr.shape(),
        &[1, EMBED_DIM],
        "full pipeline output should be [1, {EMBED_DIM}]"
    );

    // All elements must be finite (no NaN/Inf through any stage)
    for &v in lo_arr.iter() {
        assert!(v.is_finite(), "full pipeline lower bound not finite: {v}");
    }
    for &v in hi_arr.iter() {
        assert!(v.is_finite(), "full pipeline upper bound not finite: {v}");
    }

    let (lo_min, hi_max) = bounds_min_max(&result.output_bounds);
    eprintln!("siglip2_pipeline full pipeline IBP: [{lo_min:.4}, {hi_max:.4}]");
}

// ---------------------------------------------------------------------------
// 9. Post-LayerNorm CROWN — tightness through normalization
// ---------------------------------------------------------------------------

/// CROWN bounds through post-LayerNorm for tightness analysis.
///
/// CROWN linearizes the normalization, which may produce tighter bounds
/// than IBP for well-conditioned inputs. With IbpValidated mode, CROWN
/// should at least not blow up.
#[test]
fn test_siglip2_pipeline_post_layernorm_crown() {
    let def = build_post_layernorm_kernel();
    let bindings = post_layernorm_bindings();
    let input = uniform_bounds(&[NUM_PATCHES, EMBED_DIM], 1.0);

    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");
    let (method, output, fallback) = assert_crown_tighter_when_not_fallback(&graph, &input);
    assert_bounds_valid(&output);

    let (lo, _) = output.lower_upper();
    assert_eq!(lo.shape(), &[NUM_PATCHES, EMBED_DIM]);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("siglip2_pipeline post-LayerNorm CROWN ({method:?}): [{lo_min:.4}, {hi_max:.4}]");
    if let Some(reason) = &fallback {
        eprintln!("Fallback reason: {reason}");
    }
}

// ---------------------------------------------------------------------------
// 10. Multi-block CROWN — CROWN through 3 stacked blocks
// ---------------------------------------------------------------------------

/// CROWN propagation through 3 stacked transformer blocks.
///
/// Deep CROWN through multiple blocks with normalization is challenging:
/// each LayerNorm adds linearization error. This test verifies that CROWN
/// either succeeds with tighter bounds or gracefully falls back to IBP.
#[test]
fn test_siglip2_pipeline_three_block_stack_crown() {
    let def = build_three_block_stack_kernel();
    let bindings = three_block_stack_bindings();
    let input = uniform_bounds(&[NUM_PATCHES, EMBED_DIM], 0.5);

    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");
    let (method, output, fallback) = assert_crown_tighter_when_not_fallback(&graph, &input);
    assert_bounds_valid(&output);

    let (lo, _) = output.lower_upper();
    assert_eq!(lo.shape(), &[NUM_PATCHES, EMBED_DIM]);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("siglip2_pipeline 3-block stack CROWN ({method:?}): [{lo_min:.4}, {hi_max:.4}]");
    if let Some(reason) = &fallback {
        eprintln!("Fallback reason: {reason}");
    }

    // Regardless of method, bounds must be finite
    assert!(
        lo_min.is_finite() && hi_max.is_finite(),
        "3-block stack CROWN bounds must be finite: [{lo_min}, {hi_max}]"
    );
}

// ---------------------------------------------------------------------------
// 11. Attention sub-block CROWN — CROWN through MHA without FFN
// ---------------------------------------------------------------------------

/// CROWN propagation through the attention sub-block (LayerNorm + MHA + residual).
///
/// LayerNorm requires heuristic linearization in CROWN. The softmax in MHA
/// is also linearized. This test verifies that CROWN at least produces valid
/// finite bounds through the isolated attention path.
#[test]
fn test_siglip2_pipeline_attention_subblock_crown() {
    let def = build_attention_subblock_kernel();
    let bindings = attention_subblock_bindings();
    let input = uniform_bounds(&[NUM_PATCHES, EMBED_DIM], 1.0);

    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");
    let (method, output, fallback) = assert_crown_tighter_when_not_fallback(&graph, &input);
    assert_bounds_valid(&output);

    let (lo, _) = output.lower_upper();
    assert_eq!(lo.shape(), &[NUM_PATCHES, EMBED_DIM]);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!(
        "siglip2_pipeline attention sub-block CROWN ({method:?}): [{lo_min:.4}, {hi_max:.4}]"
    );
    if let Some(reason) = &fallback {
        eprintln!("Fallback reason: {reason}");
    }

    assert!(
        lo_min.is_finite(),
        "attention sub-block CROWN lower must be finite"
    );
    assert!(
        hi_max.is_finite(),
        "attention sub-block CROWN upper must be finite"
    );
}

// ---------------------------------------------------------------------------
// 12. Narrow SiGLU FFN CROWN — precision analysis on multiplicative gating
// ---------------------------------------------------------------------------

/// CROWN through narrow-bounds SiGLU FFN for precision analysis.
///
/// With tight input bounds [-0.1, 0.1], sigmoid is near-linear (slope ~0.25
/// around zero), so CROWN linearization should be relatively tight. The
/// multiplicative gating (BinaryMul of two bounded quantities) may still
/// cause some width expansion.
#[test]
fn test_siglip2_pipeline_narrow_siglu_ffn_crown() {
    let def = build_narrow_siglu_ffn_kernel();
    let bindings = narrow_siglu_ffn_bindings();
    let input = uniform_bounds(&[NUM_PATCHES, EMBED_DIM], 0.1);

    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");
    let (method, output, fallback) = assert_crown_tighter_when_not_fallback(&graph, &input);
    assert_bounds_valid(&output);

    let (lo, _) = output.lower_upper();
    assert_eq!(lo.shape(), &[NUM_PATCHES, EMBED_DIM]);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("siglip2_pipeline narrow SiGLU FFN CROWN ({method:?}): [{lo_min:.4}, {hi_max:.4}]");
    if let Some(reason) = &fallback {
        eprintln!("Fallback reason: {reason}");
    }
}

// ---------------------------------------------------------------------------
// 13. Position embed + block CROWN — cross-stage CROWN propagation
// ---------------------------------------------------------------------------

/// CROWN propagation through position embedding followed by one encoder block.
///
/// Tests cross-stage CROWN: the additive pos_embed offset propagates through
/// the CROWN linearization of LayerNorm and softmax in the encoder block.
/// LayerNorm causes Heuristic soundness mode.
#[test]
fn test_siglip2_pipeline_pos_embed_plus_block_crown() {
    let def = build_pos_embed_plus_block_kernel();
    let bindings = pos_embed_plus_block_bindings();
    let input = uniform_bounds(&[NUM_PATCHES, EMBED_DIM], 1.0);

    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");
    let (method, output, fallback) = assert_crown_tighter_when_not_fallback(&graph, &input);
    assert_bounds_valid(&output);

    let (lo, _) = output.lower_upper();
    assert_eq!(lo.shape(), &[NUM_PATCHES, EMBED_DIM]);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("siglip2_pipeline pos_embed + block CROWN ({method:?}): [{lo_min:.4}, {hi_max:.4}]");
    if let Some(reason) = &fallback {
        eprintln!("Fallback reason: {reason}");
    }

    assert!(
        lo_min.is_finite() && hi_max.is_finite(),
        "pos_embed + block CROWN bounds must be finite: [{lo_min}, {hi_max}]"
    );
}

// ---------------------------------------------------------------------------
// 14. Full pipeline CROWN — end-to-end with all stages including pool
// ---------------------------------------------------------------------------

/// CROWN propagation through the full SigLIP2 pipeline (all 6 stages).
///
/// This is the most demanding CROWN test: projection + pos_embed + 2 encoder
/// blocks (each with LayerNorm + MHA + LayerNorm + GELU FFN) + post-LayerNorm +
/// mean pooling. CROWN must linearize multiple normalization and activation
/// layers. Bounds may be wider than IBP due to accumulated linearization error.
#[test]
fn test_siglip2_pipeline_full_with_pool_crown() {
    let def = build_full_pipeline_with_pool_kernel();
    let bindings = full_pipeline_with_pool_bindings();
    let input = BoundedTensor::new(
        ArrayD::from_elem(IxDyn(&[NUM_PATCHES, PATCH_DIM]), 0.0f32),
        ArrayD::from_elem(IxDyn(&[NUM_PATCHES, PATCH_DIM]), 1.0f32),
    )
    .expect("bounds");

    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");
    let (method, output, fallback) = assert_crown_tighter_when_not_fallback(&graph, &input);
    assert_bounds_valid(&output);

    let (lo_arr, hi_arr) = output.lower_upper();
    assert_eq!(
        lo_arr.shape(),
        &[1, EMBED_DIM],
        "full pipeline CROWN output should be [1, {EMBED_DIM}]"
    );

    // All bounds must be finite through the full pipeline
    for &v in lo_arr.iter() {
        assert!(v.is_finite(), "full pipeline CROWN lower not finite: {v}");
    }
    for &v in hi_arr.iter() {
        assert!(v.is_finite(), "full pipeline CROWN upper not finite: {v}");
    }

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("siglip2_pipeline full pipeline CROWN ({method:?}): [{lo_min:.4}, {hi_max:.4}]");
    if let Some(reason) = &fallback {
        eprintln!("Fallback reason: {reason}");
    }
}

// ---------------------------------------------------------------------------
// 15. Head projection sub-block — isolated final Linear projection
// ---------------------------------------------------------------------------

/// Build an isolated head projection kernel (final stage before output).
///
/// Input: `[NUM_PATCHES, EMBED_DIM]` (Variable, encoder output).
/// Output: `[NUM_PATCHES, HEAD_DIM]`.
///
/// SigLIP2 projects the encoder output to a lower-dimensional head space
/// for contrastive matching.
fn build_head_projection_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("siglip2_pipeline_head_projection");

    let input = b.add_input("x", &[NUM_PATCHES, EMBED_DIM]);
    let head_w = b.add_input("head_weight", &[HEAD_DIM, EMBED_DIM]);
    let head_b = b.add_input("head_bias", &[HEAD_DIM]);

    let out = b.add_linear(input, head_w, Some(head_b), &[NUM_PATCHES, HEAD_DIM]);

    b.build(out).expect("valid head projection kernel")
}

fn head_projection_bindings() -> Vec<TensorParamBinding> {
    vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[HEAD_DIM, EMBED_DIM]),
            0.02f32,
        )),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[HEAD_DIM]), 0.0f32)),
    ]
}

/// Head projection IBP produces tight bounds for linear transform.
///
/// Linear projection is exact in IBP (no activation linearization). With
/// w=0.02 and [-1, 1] input, output width should be proportional to
/// 2 * 0.02 * EMBED_DIM = 0.64.
#[test]
fn test_siglip2_pipeline_head_projection_ibp() {
    let def = build_head_projection_kernel();
    let bindings = head_projection_bindings();
    let input = uniform_bounds(&[NUM_PATCHES, EMBED_DIM], 1.0);

    let result = verify_and_assert(&def, &bindings, &input, "siglip2_pipeline_head_projection");
    assert_bounds_valid(&result.output_bounds);

    let (lo, _) = result.output_bounds.lower_upper();
    assert_eq!(lo.shape(), &[NUM_PATCHES, HEAD_DIM]);

    let (lo_min, hi_max) = bounds_min_max(&result.output_bounds);
    eprintln!("siglip2_pipeline head projection IBP: [{lo_min:.4}, {hi_max:.4}]");

    // Linear with small weights: bounds should be tight
    assert_bounds_width(&result.output_bounds, 10.0, "head projection");
}

// ---------------------------------------------------------------------------
// 16. SiGLU FFN with residual — full FFN sub-block as in the real architecture
// ---------------------------------------------------------------------------

/// Build a SiGLU FFN sub-block with residual connection (as in the real SigLIP2).
///
/// Input: `[NUM_PATCHES, EMBED_DIM]` (Variable).
/// Output: `[NUM_PATCHES, EMBED_DIM]`.
///
/// Architecture: LayerNorm -> SiGLU FFN -> residual.
/// This tests the FFN path in context (with normalization and residual),
/// not the isolated FFN.
fn build_siglu_ffn_with_residual_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("siglip2_pipeline_siglu_ffn_residual");

    let input = b.add_input("x", &[NUM_PATCHES, EMBED_DIM]);
    let eps = b.add_input("eps", &[1]);
    let ln_w = b.add_input("ln_weight", &[EMBED_DIM]);
    let ln_b = b.add_input("ln_bias", &[EMBED_DIM]);
    let gate_w = b.add_input("gate_weight", &[FFN_DIM, EMBED_DIM]);
    let gate_b = b.add_input("gate_bias", &[FFN_DIM]);
    let up_w = b.add_input("up_weight", &[FFN_DIM, EMBED_DIM]);
    let up_b = b.add_input("up_bias", &[FFN_DIM]);
    let down_w = b.add_input("down_weight", &[EMBED_DIM, FFN_DIM]);
    let down_b = b.add_input("down_bias", &[EMBED_DIM]);

    // LayerNorm before FFN (pre-norm architecture)
    let normed = b.add_layer_norm(input, eps, 1, ln_w, ln_b, &[NUM_PATCHES, EMBED_DIM]);

    // SiGLU gate branch: Linear -> SiLU(x) = x * sigmoid(x)
    let gate = b.add_linear(normed, gate_w, Some(gate_b), &[NUM_PATCHES, FFN_DIM]);
    let gate_sig = b.add_sigmoid(gate, &[NUM_PATCHES, FFN_DIM]);
    let gate_activated = b.add_binary_mul(gate, gate_sig, &[NUM_PATCHES, FFN_DIM]);

    // Up branch: Linear
    let up = b.add_linear(normed, up_w, Some(up_b), &[NUM_PATCHES, FFN_DIM]);

    // Multiplicative gating
    let hidden = b.add_binary_mul(gate_activated, up, &[NUM_PATCHES, FFN_DIM]);

    // Down projection
    let ffn_out = b.add_linear(hidden, down_w, Some(down_b), &[NUM_PATCHES, EMBED_DIM]);

    // Residual connection: x + ffn_out
    let out = b.add_binary_add(input, ffn_out, &[NUM_PATCHES, EMBED_DIM]);

    b.build(out).expect("valid SiGLU FFN with residual kernel")
}

fn siglu_ffn_with_residual_bindings() -> Vec<TensorParamBinding> {
    vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantScalar(1e-6),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[EMBED_DIM]), 1.0f32)),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[EMBED_DIM]), 0.0f32)),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[FFN_DIM, EMBED_DIM]),
            0.02f32,
        )),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[FFN_DIM]), 0.0f32)),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[FFN_DIM, EMBED_DIM]),
            0.02f32,
        )),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[FFN_DIM]), 0.0f32)),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[EMBED_DIM, FFN_DIM]),
            0.02f32,
        )),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[EMBED_DIM]), 0.0f32)),
    ]
}

/// SiGLU FFN with residual IBP: LayerNorm + SiGLU + residual produces valid bounds.
///
/// The residual connection keeps output bounds anchored near the input range.
/// LayerNorm normalizes before FFN, and the residual adds the original input.
#[test]
fn test_siglip2_pipeline_siglu_ffn_residual_ibp() {
    let def = build_siglu_ffn_with_residual_kernel();
    let bindings = siglu_ffn_with_residual_bindings();
    let input = uniform_bounds(&[NUM_PATCHES, EMBED_DIM], 1.0);

    let result = verify_and_assert(
        &def,
        &bindings,
        &input,
        "siglip2_pipeline_siglu_ffn_residual",
    );
    assert_bounds_valid(&result.output_bounds);

    let (lo, _) = result.output_bounds.lower_upper();
    assert_eq!(lo.shape(), &[NUM_PATCHES, EMBED_DIM]);

    // Heuristic mode due to LayerNorm
    assert_eq!(
        result.verification.soundness_mode,
        VerificationSoundnessMode::Heuristic,
        "SiGLU FFN + residual with LayerNorm should produce Heuristic mode"
    );

    let (lo_min, hi_max) = bounds_min_max(&result.output_bounds);
    eprintln!("siglip2_pipeline SiGLU FFN + residual IBP: [{lo_min:.4}, {hi_max:.4}]");
}

// ---------------------------------------------------------------------------
// 17. Two-block widening analysis — measure bounds expansion between blocks
// ---------------------------------------------------------------------------

/// Build a 2-block stack for bounds widening measurement.
fn build_two_block_stack_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("siglip2_pipeline_two_block_stack");

    let input = b.add_input("x", &[NUM_PATCHES, EMBED_DIM]);
    let eps = b.add_input("eps", &[1]);

    let config = TransformerBlockConfig {
        num_heads: NUM_HEADS,
        mask: AttentionMask::Standard,
        ffn_hidden_dim: FFN_DIM,
    };

    let mut x = input;
    for i in 0..2 {
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
    }

    b.build(x).expect("valid 2-block stack kernel")
}

fn two_block_stack_bindings() -> Vec<TensorParamBinding> {
    let w_proj = ArrayD::from_elem(IxDyn(&[EMBED_DIM, EMBED_DIM]), 0.02f32);
    let w_ffn1 = ArrayD::from_elem(IxDyn(&[FFN_DIM, EMBED_DIM]), 0.02f32);
    let w_ffn2 = ArrayD::from_elem(IxDyn(&[EMBED_DIM, FFN_DIM]), 0.02f32);
    let ln_w = ArrayD::from_elem(IxDyn(&[EMBED_DIM]), 1.0f32);
    let ln_b = ArrayD::from_elem(IxDyn(&[EMBED_DIM]), 0.0f32);

    let mut bindings = vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantScalar(1e-6),
    ];

    // 2 blocks x 10 weights each
    for _ in 0..2 {
        bindings.push(TensorParamBinding::ConstantTensor(ln_w.clone()));
        bindings.push(TensorParamBinding::ConstantTensor(ln_b.clone()));
        bindings.push(TensorParamBinding::ConstantTensor(ln_w.clone()));
        bindings.push(TensorParamBinding::ConstantTensor(ln_b.clone()));
        bindings.push(TensorParamBinding::ConstantTensor(w_proj.clone()));
        bindings.push(TensorParamBinding::ConstantTensor(w_proj.clone()));
        bindings.push(TensorParamBinding::ConstantTensor(w_proj.clone()));
        bindings.push(TensorParamBinding::ConstantTensor(w_proj.clone()));
        bindings.push(TensorParamBinding::ConstantTensor(w_ffn1.clone()));
        bindings.push(TensorParamBinding::ConstantTensor(w_ffn2.clone()));
    }

    bindings
}

/// Two-block widening analysis: compare single-block vs two-block bounds.
///
/// Verifies that adding a second encoder block does not cause bounds to
/// blow up catastrophically. The widening ratio (2-block width / 1-block width)
/// should be bounded — with residual connections and small weights, the
/// expansion per block is limited.
#[test]
fn test_siglip2_pipeline_two_block_widening_analysis() {
    // Single block bounds
    let def1 = build_attention_subblock_kernel();
    let bindings1 = attention_subblock_bindings();
    let input = uniform_bounds(&[NUM_PATCHES, EMBED_DIM], 1.0);
    let graph1 = tensor_kernel_to_graph(&def1, &bindings1).expect("single block graph");
    let output1 = graph1.propagate_ibp(&input).expect("single block IBP");
    let (lo1_min, hi1_max) = bounds_min_max(&output1);
    let width1 = hi1_max - lo1_min;

    // Two-block bounds
    let def2 = build_two_block_stack_kernel();
    let bindings2 = two_block_stack_bindings();
    let graph2 = tensor_kernel_to_graph(&def2, &bindings2).expect("two block graph");
    let output2 = graph2.propagate_ibp(&input).expect("two block IBP");
    assert_bounds_valid(&output2);
    let (lo2_min, hi2_max) = bounds_min_max(&output2);
    let width2 = hi2_max - lo2_min;

    let expansion = if width1 > 1e-10 { width2 / width1 } else { 1.0 };
    eprintln!(
        "siglip2_pipeline widening analysis: 1-block=[{lo1_min:.4}, {hi1_max:.4}] width={width1:.4}, \
         2-block=[{lo2_min:.4}, {hi2_max:.4}] width={width2:.4}, expansion={expansion:.2}x"
    );

    // Both must be finite
    assert!(
        lo2_min.is_finite() && hi2_max.is_finite(),
        "2-block bounds must be finite: [{lo2_min}, {hi2_max}]"
    );

    // Expansion should not be catastrophic (< 1000x per block)
    assert!(
        expansion < 1000.0,
        "2-block expansion {expansion:.2}x exceeds 1000x threshold"
    );
}

// ---------------------------------------------------------------------------
// 18. Mean pool CROWN — CROWN through mean reduction
// ---------------------------------------------------------------------------

/// CROWN propagation through mean pooling reduction.
///
/// Mean reduction is a linear operation (weighted sum with 1/N coefficients),
/// so CROWN should handle it exactly (no linearization error). This test
/// verifies that CROWN bounds through mean pooling are at least as tight
/// as IBP bounds.
#[test]
fn test_siglip2_pipeline_mean_pool_crown() {
    let def = build_mean_pool_kernel();
    let bindings = mean_pool_bindings();
    let input = uniform_bounds(&[NUM_PATCHES, EMBED_DIM], 1.0);

    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");
    let (method, output, fallback) = assert_crown_tighter_when_not_fallback(&graph, &input);
    assert_bounds_valid(&output);

    let (lo, _) = output.lower_upper();
    assert_eq!(
        lo.shape(),
        &[1, EMBED_DIM],
        "mean pool CROWN output should be [1, {EMBED_DIM}]"
    );

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("siglip2_pipeline mean pool CROWN ({method:?}): [{lo_min:.4}, {hi_max:.4}]");
    if let Some(reason) = &fallback {
        eprintln!("Fallback reason: {reason}");
    }

    // Mean of [-1, 1] should stay within [-1, 1] (plus small epsilon)
    assert!(
        lo_min >= -1.0 - 1e-4,
        "mean pool CROWN lower should be >= -1.0, got {lo_min}"
    );
    assert!(
        hi_max <= 1.0 + 1e-4,
        "mean pool CROWN upper should be <= 1.0, got {hi_max}"
    );
}

// ===========================================================================
// ViT encoder compose blocks (hidden=8, seq=4, heads=2)
//
// These use minimal dimensions per #3583 specification for fast, targeted
// verification of the three canonical ViT encoder sub-blocks:
// 1. patch_embed — isolated linear patch projection
// 2. single_vit_block — manually decomposed transformer block (no composite)
// 3. multi_block_stack — 2-block stacked encoder
// ===========================================================================

/// Hidden dimension for ViT compose blocks.
const VIT_HIDDEN: usize = 8;
/// Sequence length (number of patches) for ViT compose blocks.
const VIT_SEQ: usize = 4;
/// Number of attention heads for ViT compose blocks.
const VIT_HEADS: usize = 2;
/// FFN intermediate dimension (4x hidden).
const VIT_FFN: usize = VIT_HIDDEN * 4; // 32
/// Flattened patch pixel dimension (input to patch_embed linear).
const VIT_PATCH_DIM: usize = 24;
/// Small weight magnitude for bounded verification.
const VIT_W_MAG: f32 = 0.02;

// ---------------------------------------------------------------------------
// 19. patch_embed — Linear patch projection (hidden=8, seq=4)
// ---------------------------------------------------------------------------

/// Build a patch embedding kernel: Linear projection from flattened patches.
///
/// Input: `[VIT_SEQ, VIT_PATCH_DIM]` (Variable, flattened pixel patches).
/// Output: `[VIT_SEQ, VIT_HIDDEN]`.
///
/// Models the patch embedding as a single Linear layer (simulating Conv2d
/// with kernel_size = stride = patch_size, flattened to 2D).
fn build_vit_patch_embed_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("siglip2_vit_patch_embed");

    let patches = b.add_input("patches", &[VIT_SEQ, VIT_PATCH_DIM]);
    let proj_w = b.add_input("proj_weight", &[VIT_HIDDEN, VIT_PATCH_DIM]);
    let proj_b = b.add_input("proj_bias", &[VIT_HIDDEN]);

    let out = b.add_linear(patches, proj_w, Some(proj_b), &[VIT_SEQ, VIT_HIDDEN]);

    b.build(out).expect("valid vit patch_embed kernel")
}

fn vit_patch_embed_bindings() -> Vec<TensorParamBinding> {
    vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[VIT_HIDDEN, VIT_PATCH_DIM]),
            VIT_W_MAG,
        )),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[VIT_HIDDEN]), 0.0f32)),
    ]
}

/// IBP bounds propagation through isolated patch embedding linear projection.
///
/// With uniform input in [-1, 1] and small weights (0.02), the output bounds
/// should be well-contained. Linear is a pure affine operation so IBP is exact.
#[test]
fn test_siglip2_vit_patch_embed_ibp() {
    let def = build_vit_patch_embed_kernel();
    let bindings = vit_patch_embed_bindings();
    let input = uniform_bounds(&[VIT_SEQ, VIT_PATCH_DIM], 1.0);

    let result = verify_and_assert(&def, &bindings, &input, "siglip2_vit_patch_embed");
    assert_bounds_valid(&result.output_bounds);

    let (lo, _) = result.output_bounds.lower_upper();
    assert_eq!(
        lo.shape(),
        &[VIT_SEQ, VIT_HIDDEN],
        "patch_embed output should be [{VIT_SEQ}, {VIT_HIDDEN}]"
    );

    let (lo_min, hi_max) = bounds_min_max(&result.output_bounds);
    eprintln!("siglip2_vit patch_embed IBP: [{lo_min:.4}, {hi_max:.4}]");

    // Linear with w_mag=0.02, input_dim=24, input in [-1,1]:
    // max output magnitude = 24 * 0.02 * 1.0 = 0.48 per element
    assert_bounds_width(&result.output_bounds, 2.0, "vit_patch_embed");
}

/// CROWN propagation through patch embedding. Linear is exact for CROWN.
#[test]
fn test_siglip2_vit_patch_embed_crown() {
    let def = build_vit_patch_embed_kernel();
    let bindings = vit_patch_embed_bindings();
    let input = uniform_bounds(&[VIT_SEQ, VIT_PATCH_DIM], 1.0);

    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");
    let (method, output, fallback) = assert_crown_tighter_when_not_fallback(&graph, &input);
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("siglip2_vit patch_embed CROWN ({method:?}): [{lo_min:.4}, {hi_max:.4}]");
    if let Some(reason) = &fallback {
        eprintln!("Fallback reason: {reason}");
    }
}

// ---------------------------------------------------------------------------
// 20. single_vit_block — Manual ViT transformer block decomposition
// ---------------------------------------------------------------------------

/// Build a single ViT transformer block with manual op-by-op construction.
///
/// This explicitly constructs the standard pre-norm ViT block WITHOUT using
/// the `add_transformer_block` composite, to verify each step independently:
///
/// 1. LayerNorm (pre-attention)
/// 2. Linear Q, Linear K, Linear V projections
/// 3. Multi-head self-attention (bidirectional)
/// 4. Linear output projection
/// 5. Residual connection (input + attn_out)
/// 6. LayerNorm (pre-FFN)
/// 7. Linear FFN up-projection
/// 8. GELU activation
/// 9. Linear FFN down-projection
/// 10. Residual connection (residual1 + ffn_out)
///
/// Input: `[VIT_SEQ, VIT_HIDDEN]` (Variable).
/// Output: `[VIT_SEQ, VIT_HIDDEN]`.
fn build_vit_single_block_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("siglip2_vit_single_block");

    let shape = [VIT_SEQ, VIT_HIDDEN];
    let ffn_shape = [VIT_SEQ, VIT_FFN];

    // Input
    let input = b.add_input("x", &shape);
    let eps = b.add_input("eps", &[1]);

    // LayerNorm 1 weights
    let ln1_w = b.add_input("ln1_weight", &[VIT_HIDDEN]);
    let ln1_b = b.add_input("ln1_bias", &[VIT_HIDDEN]);

    // Attention weights
    let q_w = b.add_input("q_weight", &[VIT_HIDDEN, VIT_HIDDEN]);
    let k_w = b.add_input("k_weight", &[VIT_HIDDEN, VIT_HIDDEN]);
    let v_w = b.add_input("v_weight", &[VIT_HIDDEN, VIT_HIDDEN]);
    let out_w = b.add_input("out_weight", &[VIT_HIDDEN, VIT_HIDDEN]);

    // LayerNorm 2 weights
    let ln2_w = b.add_input("ln2_weight", &[VIT_HIDDEN]);
    let ln2_b = b.add_input("ln2_bias", &[VIT_HIDDEN]);

    // FFN weights
    let ffn1_w = b.add_input("ffn1_weight", &[VIT_FFN, VIT_HIDDEN]);
    let ffn2_w = b.add_input("ffn2_weight", &[VIT_HIDDEN, VIT_FFN]);

    // Step 1: Pre-attention LayerNorm
    let normed = b.add_layer_norm(input, eps, 1, ln1_w, ln1_b, &shape);

    // Steps 2-4: Multi-head self-attention (Linear Q,K,V → attention → Linear proj)
    let attn_out = b
        .add_multi_head_attention(
            normed,
            q_w,
            k_w,
            v_w,
            out_w,
            VIT_HEADS,
            AttentionMask::Standard,
            &shape,
        )
        .expect("valid MHA");

    // Step 5: First residual connection
    let residual1 = b.add_binary_add(input, attn_out, &shape);

    // Step 6: Pre-FFN LayerNorm
    let normed2 = b.add_layer_norm(residual1, eps, 1, ln2_w, ln2_b, &shape);

    // Step 7: FFN up-projection
    let ffn1 = b.add_linear(normed2, ffn1_w, None, &ffn_shape);

    // Step 8: GELU activation
    let act = b.add_gelu(ffn1, &ffn_shape);

    // Step 9: FFN down-projection
    let ffn2 = b.add_linear(act, ffn2_w, None, &shape);

    // Step 10: Second residual connection
    let out = b.add_binary_add(residual1, ffn2, &shape);

    b.build(out).expect("valid vit single_block kernel")
}

fn vit_single_block_bindings() -> Vec<TensorParamBinding> {
    let w_proj = ArrayD::from_elem(IxDyn(&[VIT_HIDDEN, VIT_HIDDEN]), VIT_W_MAG);
    let w_ffn1 = ArrayD::from_elem(IxDyn(&[VIT_FFN, VIT_HIDDEN]), VIT_W_MAG);
    let w_ffn2 = ArrayD::from_elem(IxDyn(&[VIT_HIDDEN, VIT_FFN]), VIT_W_MAG);
    let ln_w = ArrayD::from_elem(IxDyn(&[VIT_HIDDEN]), 1.0f32);
    let ln_b = ArrayD::from_elem(IxDyn(&[VIT_HIDDEN]), 0.0f32);

    vec![
        TensorParamBinding::Variable,                       // x
        TensorParamBinding::ConstantScalar(1e-6),           // eps
        TensorParamBinding::ConstantTensor(ln_w.clone()),   // ln1_weight
        TensorParamBinding::ConstantTensor(ln_b.clone()),   // ln1_bias
        TensorParamBinding::ConstantTensor(w_proj.clone()), // q_weight
        TensorParamBinding::ConstantTensor(w_proj.clone()), // k_weight
        TensorParamBinding::ConstantTensor(w_proj.clone()), // v_weight
        TensorParamBinding::ConstantTensor(w_proj),         // out_weight
        TensorParamBinding::ConstantTensor(ln_w),           // ln2_weight
        TensorParamBinding::ConstantTensor(ln_b),           // ln2_bias
        TensorParamBinding::ConstantTensor(w_ffn1),         // ffn1_weight
        TensorParamBinding::ConstantTensor(w_ffn2),         // ffn2_weight
    ]
}

/// IBP bounds through a single ViT block with manual decomposition.
///
/// Verifies the full pre-norm transformer block: LayerNorm -> MHA -> residual
/// -> LayerNorm -> Linear -> GELU -> Linear -> residual. With small weights
/// (0.02) and hidden=8, bounds should remain finite and well-contained.
#[test]
fn test_siglip2_vit_single_block_ibp() {
    let def = build_vit_single_block_kernel();
    let bindings = vit_single_block_bindings();
    let input = uniform_bounds(&[VIT_SEQ, VIT_HIDDEN], 1.0);

    let result = verify_and_assert(&def, &bindings, &input, "siglip2_vit_single_block");
    assert_bounds_valid(&result.output_bounds);

    let (lo, _) = result.output_bounds.lower_upper();
    assert_eq!(
        lo.shape(),
        &[VIT_SEQ, VIT_HIDDEN],
        "single_vit_block output should be [{VIT_SEQ}, {VIT_HIDDEN}]"
    );

    let (lo_min, hi_max) = bounds_min_max(&result.output_bounds);
    eprintln!("siglip2_vit single_block IBP: [{lo_min:.4}, {hi_max:.4}]");

    // Residual connections keep output close to input range.
    // With small weights, bounds should not blow up catastrophically.
    assert!(
        lo_min.is_finite() && hi_max.is_finite(),
        "single_vit_block IBP bounds must be finite: [{lo_min}, {hi_max}]"
    );
}

/// CROWN through a single ViT block. Tests tightening through LayerNorm,
/// attention, GELU, and residual connections.
#[test]
fn test_siglip2_vit_single_block_crown() {
    let def = build_vit_single_block_kernel();
    let bindings = vit_single_block_bindings();
    let input = uniform_bounds(&[VIT_SEQ, VIT_HIDDEN], 1.0);

    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");
    let (method, output, fallback) = assert_crown_tighter_when_not_fallback(&graph, &input);
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("siglip2_vit single_block CROWN ({method:?}): [{lo_min:.4}, {hi_max:.4}]");
    if let Some(reason) = &fallback {
        eprintln!("Fallback reason: {reason}");
    }
}

// ---------------------------------------------------------------------------
// 21. multi_block_stack — 2-block ViT encoder (hidden=8, seq=4, heads=2)
// ---------------------------------------------------------------------------

/// Build a 2-block stacked ViT encoder at minimal dimensions.
///
/// Input: `[VIT_SEQ, VIT_HIDDEN]` (Variable).
/// Output: `[VIT_SEQ, VIT_HIDDEN]`.
///
/// Each block is constructed via `add_transformer_block` composite (which
/// internally decomposes to LayerNorm -> MHA -> residual -> LayerNorm ->
/// Linear -> GELU -> Linear -> residual). Two blocks test bounds widening
/// through depth.
fn build_vit_multi_block_stack_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("siglip2_vit_multi_block_stack");

    let input = b.add_input("x", &[VIT_SEQ, VIT_HIDDEN]);
    let eps = b.add_input("eps", &[1]);

    let config = TransformerBlockConfig {
        num_heads: VIT_HEADS,
        mask: AttentionMask::Standard,
        ffn_hidden_dim: VIT_FFN,
    };

    let mut x = input;
    for i in 0..2 {
        let ln1_w = b.add_input(&format!("b{i}_ln1_w"), &[VIT_HIDDEN]);
        let ln1_b = b.add_input(&format!("b{i}_ln1_b"), &[VIT_HIDDEN]);
        let ln2_w = b.add_input(&format!("b{i}_ln2_w"), &[VIT_HIDDEN]);
        let ln2_b = b.add_input(&format!("b{i}_ln2_b"), &[VIT_HIDDEN]);
        let q_w = b.add_input(&format!("b{i}_q_w"), &[VIT_HIDDEN, VIT_HIDDEN]);
        let k_w = b.add_input(&format!("b{i}_k_w"), &[VIT_HIDDEN, VIT_HIDDEN]);
        let v_w = b.add_input(&format!("b{i}_v_w"), &[VIT_HIDDEN, VIT_HIDDEN]);
        let out_w = b.add_input(&format!("b{i}_out_w"), &[VIT_HIDDEN, VIT_HIDDEN]);
        let ffn1_w = b.add_input(&format!("b{i}_ffn1_w"), &[VIT_FFN, VIT_HIDDEN]);
        let ffn2_w = b.add_input(&format!("b{i}_ffn2_w"), &[VIT_HIDDEN, VIT_FFN]);

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
    }

    b.build(x).expect("valid vit multi_block_stack kernel")
}

fn vit_multi_block_stack_bindings() -> Vec<TensorParamBinding> {
    let w_proj = ArrayD::from_elem(IxDyn(&[VIT_HIDDEN, VIT_HIDDEN]), VIT_W_MAG);
    let w_ffn1 = ArrayD::from_elem(IxDyn(&[VIT_FFN, VIT_HIDDEN]), VIT_W_MAG);
    let w_ffn2 = ArrayD::from_elem(IxDyn(&[VIT_HIDDEN, VIT_FFN]), VIT_W_MAG);
    let ln_w = ArrayD::from_elem(IxDyn(&[VIT_HIDDEN]), 1.0f32);
    let ln_b = ArrayD::from_elem(IxDyn(&[VIT_HIDDEN]), 0.0f32);

    let mut bindings = vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantScalar(1e-6),
    ];

    // 2 blocks x 10 weights each
    for _ in 0..2 {
        bindings.push(TensorParamBinding::ConstantTensor(ln_w.clone()));
        bindings.push(TensorParamBinding::ConstantTensor(ln_b.clone()));
        bindings.push(TensorParamBinding::ConstantTensor(ln_w.clone()));
        bindings.push(TensorParamBinding::ConstantTensor(ln_b.clone()));
        bindings.push(TensorParamBinding::ConstantTensor(w_proj.clone()));
        bindings.push(TensorParamBinding::ConstantTensor(w_proj.clone()));
        bindings.push(TensorParamBinding::ConstantTensor(w_proj.clone()));
        bindings.push(TensorParamBinding::ConstantTensor(w_proj.clone()));
        bindings.push(TensorParamBinding::ConstantTensor(w_ffn1.clone()));
        bindings.push(TensorParamBinding::ConstantTensor(w_ffn2.clone()));
    }

    bindings
}

/// IBP bounds through a 2-block stacked ViT encoder.
///
/// Tests how bounds widen through two consecutive transformer blocks at
/// minimal dimensions (hidden=8, seq=4, heads=2). Residual connections
/// limit per-block expansion.
#[test]
fn test_siglip2_vit_multi_block_stack_ibp() {
    let def = build_vit_multi_block_stack_kernel();
    let bindings = vit_multi_block_stack_bindings();
    let input = uniform_bounds(&[VIT_SEQ, VIT_HIDDEN], 1.0);

    let result = verify_and_assert(&def, &bindings, &input, "siglip2_vit_multi_block_stack");
    assert_bounds_valid(&result.output_bounds);

    let (lo, _) = result.output_bounds.lower_upper();
    assert_eq!(
        lo.shape(),
        &[VIT_SEQ, VIT_HIDDEN],
        "multi_block_stack output should be [{VIT_SEQ}, {VIT_HIDDEN}]"
    );

    let (lo_min, hi_max) = bounds_min_max(&result.output_bounds);
    eprintln!("siglip2_vit multi_block_stack IBP: [{lo_min:.4}, {hi_max:.4}]");

    assert!(
        lo_min.is_finite() && hi_max.is_finite(),
        "multi_block_stack IBP bounds must be finite: [{lo_min}, {hi_max}]"
    );
}

/// CROWN through 2-block stacked ViT encoder.
///
/// Tests CROWN tightening through deep composition. With two blocks, CROWN
/// must propagate linear relaxations through LayerNorm, attention, and GELU
/// in both blocks. The tightening benefit accumulates across depth.
#[test]
fn test_siglip2_vit_multi_block_stack_crown() {
    let def = build_vit_multi_block_stack_kernel();
    let bindings = vit_multi_block_stack_bindings();
    let input = uniform_bounds(&[VIT_SEQ, VIT_HIDDEN], 1.0);

    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");
    let (method, output, fallback) = assert_crown_tighter_when_not_fallback(&graph, &input);
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("siglip2_vit multi_block_stack CROWN ({method:?}): [{lo_min:.4}, {hi_max:.4}]");
    if let Some(reason) = &fallback {
        eprintln!("Fallback reason: {reason}");
    }
}

/// Widening analysis: single ViT block vs 2-block stack (hidden=8).
///
/// Measures how much bounds expand from 1 block to 2 blocks at minimal
/// dimensions. Complements the D=16 widening analysis above with smaller
/// dims to test whether expansion characteristics are dimension-invariant.
#[test]
fn test_siglip2_vit_block_widening_1v2() {
    // Single block
    let def1 = build_vit_single_block_kernel();
    let bindings1 = vit_single_block_bindings();
    let input = uniform_bounds(&[VIT_SEQ, VIT_HIDDEN], 1.0);
    let graph1 = tensor_kernel_to_graph(&def1, &bindings1).expect("single block graph");
    let output1 = graph1.propagate_ibp(&input).expect("single block IBP");
    let (lo1_min, hi1_max) = bounds_min_max(&output1);
    let width1 = hi1_max - lo1_min;

    // Two blocks
    let def2 = build_vit_multi_block_stack_kernel();
    let bindings2 = vit_multi_block_stack_bindings();
    let graph2 = tensor_kernel_to_graph(&def2, &bindings2).expect("two block graph");
    let output2 = graph2.propagate_ibp(&input).expect("two block IBP");
    assert_bounds_valid(&output2);
    let (lo2_min, hi2_max) = bounds_min_max(&output2);
    let width2 = hi2_max - lo2_min;

    let expansion = if width1 > 1e-10 { width2 / width1 } else { 1.0 };
    eprintln!(
        "siglip2_vit widening (D=8): 1-block=[{lo1_min:.4}, {hi1_max:.4}] width={width1:.4}, \
         2-block=[{lo2_min:.4}, {hi2_max:.4}] width={width2:.4}, expansion={expansion:.2}x"
    );

    assert!(
        lo2_min.is_finite() && hi2_max.is_finite(),
        "2-block bounds must be finite: [{lo2_min}, {hi2_max}]"
    );

    // Expansion should not be catastrophic
    assert!(
        expansion < 1000.0,
        "2-block expansion {expansion:.2}x exceeds 1000x threshold"
    );
}

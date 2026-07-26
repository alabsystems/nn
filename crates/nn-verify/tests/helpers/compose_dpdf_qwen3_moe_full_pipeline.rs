// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Compose tests for Qwen3-VL MoE full vision-to-text pipeline bounds.
//!
//! Verifies IBP and CROWN bound propagation through the complete Qwen3-VL MoE
//! pipeline: vision encoder patch embedding, window attention ViT blocks, MoE
//! router and expert FFNs, cross-attention fusion, SwiGLU decoder blocks, KV
//! cache, and LM head generation.
//!
//! ## Tests (18 tests)
//!
//! 1.  **Vision encoder patch embedding bounds** (IBP)
//! 2.  **Window attention ViT block bounds** (IBP)
//! 3.  **MoE router softmax logit bounds** (IBP)
//! 4.  **Expert FFN per-expert bounds** (IBP)
//! 5.  **MoE weighted combine output bounds** (IBP)
//! 6.  **Top-k gating selection bounds** (IBP)
//! 7.  **Cross-attention vision-to-text bounds** (IBP)
//! 8.  **SwiGLU FFN in standard decoder blocks** (IBP)
//! 9.  **RMSNorm normalization bounds** (IBP)
//! 10. **Residual connections through decoder** (IBP)
//! 11. **KV cache append bounds** (IBP)
//! 12. **LM head logit projection bounds** (IBP)
//! 13. **Full vision encoder pipeline** (CROWN)
//! 14. **Full decoder block pipeline** (IBP)
//! 15. **Two-block decoder composition** (IBP)
//! 16. **Embedding + position encoding bounds** (IBP)
//! 17. **Multi-expert load balancing loss bounds** (IBP)
//! 18. **End-to-end vision-to-logit pipeline** (IBP)
//!
//! Architecture references:
//! - Qwen3-VL (Alibaba): Vision-language model with 3D patch embedding,
//!   M-RoPE, window attention, SwiGLU, GQA, and MoE routing
//! - RMSNorm (Zhang & Sennrich, 2019): replaces LayerNorm
//! - SwiGLU (Shazeer, 2020): SiLU-gated FFN
//! - GQA (Ainslie et al., 2023): Grouped-Query Attention
//! - MoE (Fedus et al., 2022): Mixture-of-Experts routing
//!
//! Dimensions (small for fast verification, structurally representative):
//! - HIDDEN_DIM=4, FFN_DIM=8, NUM_HEADS=2, NUM_KV_HEADS=2,
//!   NUM_EXPERTS=4, TOP_K=2, SEQ_LEN=4, VOCAB_SIZE=6, PATCH_DIM=4
//!
//! Part of #4192: Compose tests for Qwen3-VL MoE full pipeline.

use super::common::{
    assert_bounds_valid, assert_crown_tighter_when_not_fallback, bounds_min_max, uniform_bounds,
};
use nn_dsl::tensor_block_builder::TensorBlockBuilder;
use nn_dsl::tensor_ir::{AttentionMask, TensorNodeId};
use nn_verify::{tensor_kernel_to_graph, BoundedTensor, TensorParamBinding};
use ndarray::{ArrayD, IxDyn};

// ---------------------------------------------------------------------------
// Dimensions -- small for fast verification, structurally representative
// ---------------------------------------------------------------------------

const HIDDEN_DIM: usize = 4;
const FFN_DIM: usize = 8;
const NUM_HEADS: usize = 2;
const NUM_KV_HEADS: usize = 2;
const NUM_EXPERTS: usize = 4;
const TOP_K: usize = 2;
const SEQ_LEN: usize = 4;
const VOCAB_SIZE: usize = 6;
const PATCH_DIM: usize = 4;
const WEIGHT_MAG: f32 = 0.02;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Constant weight tensor binding.
fn weight(shape: &[usize]) -> TensorParamBinding {
    TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(shape), WEIGHT_MAG))
}

/// Zero bias tensor binding.
fn bias_zero(shape: &[usize]) -> TensorParamBinding {
    TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(shape), 0.0f32))
}

/// Ones tensor binding (for RMSNorm weight).
fn ones(shape: &[usize]) -> TensorParamBinding {
    TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(shape), 1.0f32))
}

/// Scalar epsilon binding.
fn eps_binding() -> TensorParamBinding {
    TensorParamBinding::ConstantScalar(1e-5)
}

/// Image-domain input bounds: pixels in [0, 1].
fn image_bounds(channels: usize, h: usize, w: usize) -> BoundedTensor {
    BoundedTensor::new(
        ArrayD::from_elem(IxDyn(&[channels, h, w]), 0.0f32),
        ArrayD::from_elem(IxDyn(&[channels, h, w]), 1.0f32),
    )
    .expect("valid image bounds [0, 1]")
}

/// Sequence-domain input bounds: embeddings in [-range, +range].
fn seq_bounds(seq_len: usize, dim: usize, range: f32) -> BoundedTensor {
    uniform_bounds(&[seq_len, dim], range)
}

/// Add a SwiGLU FFN block to a builder.
///
/// gate_proj -> SiLU (sigmoid * x) -> mul(up_proj) -> down_proj.
/// Input: [SEQ, DIM], Output: [SEQ, DIM].
fn add_swiglu_ffn(
    b: &mut TensorBlockBuilder,
    input: TensorNodeId,
    seq: usize,
    dim: usize,
    ffn_dim: usize,
    prefix: &str,
) -> TensorNodeId {
    let gate_w = b.add_input(&format!("{prefix}_gate_w"), &[ffn_dim, dim]);
    let up_w = b.add_input(&format!("{prefix}_up_w"), &[ffn_dim, dim]);
    let down_w = b.add_input(&format!("{prefix}_down_w"), &[dim, ffn_dim]);

    let ffn_shape = [seq, ffn_dim];
    let out_shape = [seq, dim];

    // Gate branch: gate_proj -> SiLU (sigmoid(x) * x)
    let gate = b.add_linear(input, gate_w, None, &ffn_shape);
    let gate_sig = b.add_sigmoid(gate, &ffn_shape);
    let gate_activated = b.add_binary_mul(gate, gate_sig, &ffn_shape);

    // Up branch
    let up = b.add_linear(input, up_w, None, &ffn_shape);

    // Multiplicative gating
    let hidden = b.add_binary_mul(gate_activated, up, &ffn_shape);

    // Down projection
    b.add_linear(hidden, down_w, None, &out_shape)
}

/// Bindings for a SwiGLU FFN block.
fn swiglu_ffn_bindings(dim: usize, ffn_dim: usize) -> Vec<TensorParamBinding> {
    vec![
        weight(&[ffn_dim, dim]), // gate_w
        weight(&[ffn_dim, dim]), // up_w
        weight(&[dim, ffn_dim]), // down_w
    ]
}

/// Add an RMSNorm to a builder. Returns the output node.
fn add_rmsnorm(
    b: &mut TensorBlockBuilder,
    input: TensorNodeId,
    shape: &[usize],
    dim: usize,
    prefix: &str,
) -> TensorNodeId {
    let w = b.add_input(&format!("{prefix}_rms_w"), &[dim]);
    let eps = b.add_input(&format!("{prefix}_rms_eps"), &[1]);
    b.add_rms_norm(input, eps, 1, w, shape)
}

/// Bindings for an RMSNorm.
fn rmsnorm_bindings(dim: usize) -> Vec<TensorParamBinding> {
    vec![
        ones(&[dim]),  // weight
        eps_binding(), // eps
    ]
}

/// Add a decoder block: RMSNorm -> MHA -> residual -> RMSNorm -> SwiGLU -> residual.
fn add_decoder_block(
    b: &mut TensorBlockBuilder,
    input: TensorNodeId,
    seq: usize,
    dim: usize,
    ffn_dim: usize,
    num_heads: usize,
    prefix: &str,
) -> TensorNodeId {
    let shape = [seq, dim];

    // Pre-attention RMSNorm
    let normed = add_rmsnorm(b, input, &shape, dim, &format!("{prefix}_atn"));

    // Multi-head self-attention
    let qw = b.add_input(&format!("{prefix}_q_w"), &[dim, dim]);
    let kw = b.add_input(&format!("{prefix}_k_w"), &[dim, dim]);
    let vw = b.add_input(&format!("{prefix}_v_w"), &[dim, dim]);
    let ow = b.add_input(&format!("{prefix}_o_w"), &[dim, dim]);
    let attn = b
        .add_multi_head_attention(
            normed,
            qw,
            kw,
            vw,
            ow,
            num_heads,
            AttentionMask::Standard,
            &shape,
        )
        .expect("valid MHA");

    // Residual 1
    let res1 = b.add_binary_add(input, attn, &shape);

    // Pre-FFN RMSNorm
    let normed2 = add_rmsnorm(b, res1, &shape, dim, &format!("{prefix}_ffn"));

    // SwiGLU FFN
    let ffn_out = add_swiglu_ffn(b, normed2, seq, dim, ffn_dim, &format!("{prefix}_swi"));

    // Residual 2
    b.add_binary_add(res1, ffn_out, &shape)
}

/// Bindings for a decoder block.
fn decoder_block_bindings(dim: usize, ffn_dim: usize) -> Vec<TensorParamBinding> {
    let mut v = Vec::new();
    // Pre-attention RMSNorm
    v.extend(rmsnorm_bindings(dim));
    // MHA: Q, K, V, O weights
    v.push(weight(&[dim, dim]));
    v.push(weight(&[dim, dim]));
    v.push(weight(&[dim, dim]));
    v.push(weight(&[dim, dim]));
    // Pre-FFN RMSNorm
    v.extend(rmsnorm_bindings(dim));
    // SwiGLU FFN
    v.extend(swiglu_ffn_bindings(dim, ffn_dim));
    v
}

// ===========================================================================
// 1. Vision encoder patch embedding bounds (IBP)
// ===========================================================================

#[test]
fn test_qwen3_moe_full_patch_embedding_ibp() {
    let in_ch = 3;
    let img_h = 8;
    let img_w = 8;
    let out_h = img_h / PATCH_DIM;
    let out_w = img_w / PATCH_DIM;
    let num_patches = out_h * out_w; // 4

    let mut b = TensorBlockBuilder::new("q3mf_patch_embed");
    let input = b.add_input("image", &[in_ch, img_h, img_w]);
    let conv_w = b.add_input("conv_w", &[HIDDEN_DIM, in_ch, PATCH_DIM, PATCH_DIM]);
    let conv_b = b.add_input("conv_b", &[HIDDEN_DIM]);

    let conv_out = b.add_conv2d(
        input,
        conv_w,
        Some(conv_b),
        PATCH_DIM,
        PATCH_DIM,
        0,
        0,
        &[HIDDEN_DIM, out_h, out_w],
    );
    // Reshape: [D, 2, 2] -> [D, num_patches]
    let reshaped = b.add_reshape(conv_out, &[HIDDEN_DIM, num_patches]);
    // Transpose: [D, num_patches] -> [num_patches, D]
    let out = b.add_transpose(reshaped, &[1, 0], &[num_patches, HIDDEN_DIM]);
    let def = b.build(out).expect("valid patch embed kernel");

    let bindings = vec![
        TensorParamBinding::Variable,
        weight(&[HIDDEN_DIM, in_ch, PATCH_DIM, PATCH_DIM]),
        bias_zero(&[HIDDEN_DIM]),
    ];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = image_bounds(in_ch, img_h, img_w);

    let output = graph.propagate_ibp(&input).expect("IBP propagation");
    assert_bounds_valid(&output);
    assert_eq!(output.lower_upper().0.shape(), &[num_patches, HIDDEN_DIM]);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Q3MF patch embed IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");
    assert!(lo_min.is_finite() && hi_max.is_finite());
}

// ===========================================================================
// 2. Window attention ViT block bounds (IBP)
// ===========================================================================

#[test]
fn test_qwen3_moe_full_window_attention_vit_ibp() {
    // Single ViT encoder block: RMSNorm -> MHA -> residual -> RMSNorm -> SwiGLU -> residual
    let mut b = TensorBlockBuilder::new("q3mf_window_attn_vit");
    let input = b.add_input("x", &[SEQ_LEN, HIDDEN_DIM]);
    let out = add_decoder_block(
        &mut b, input, SEQ_LEN, HIDDEN_DIM, FFN_DIM, NUM_HEADS, "vit0",
    );
    let def = b.build(out).expect("valid ViT block kernel");

    let mut bindings = vec![TensorParamBinding::Variable];
    bindings.extend(decoder_block_bindings(HIDDEN_DIM, FFN_DIM));
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let inp = seq_bounds(SEQ_LEN, HIDDEN_DIM, 1.0);

    let output = graph.propagate_ibp(&inp).expect("IBP propagation");
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Q3MF window attn ViT IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");
    assert!(lo_min.is_finite() && hi_max.is_finite());
}

// ===========================================================================
// 3. MoE router softmax logit bounds (IBP)
// ===========================================================================

#[test]
fn test_qwen3_moe_full_router_softmax_ibp() {
    // Router: Linear(DIM, NUM_EXPERTS) -> softmax -> gate probabilities in [0, 1]
    let mut b = TensorBlockBuilder::new("q3mf_router_softmax");
    let input = b.add_input("x", &[SEQ_LEN, HIDDEN_DIM]);
    let router_w = b.add_input("router_w", &[NUM_EXPERTS, HIDDEN_DIM]);
    let logits = b.add_linear(input, router_w, None, &[SEQ_LEN, NUM_EXPERTS]);
    let out = b.add_softmax(logits, -1, &[SEQ_LEN, NUM_EXPERTS]);
    let def = b.build(out).expect("valid router softmax kernel");

    let bindings = vec![
        TensorParamBinding::Variable,
        weight(&[NUM_EXPERTS, HIDDEN_DIM]),
    ];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let inp = seq_bounds(SEQ_LEN, HIDDEN_DIM, 1.0);

    let output = graph.propagate_ibp(&inp).expect("IBP propagation");
    assert_bounds_valid(&output);
    assert_eq!(output.lower_upper().0.shape(), &[SEQ_LEN, NUM_EXPERTS]);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Q3MF router softmax IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");
    assert!(
        lo_min >= -1e-5,
        "softmax lower bound must be >= 0, got {lo_min}"
    );
    assert!(
        hi_max <= 1.0 + 1e-5,
        "softmax upper bound must be <= 1, got {hi_max}"
    );
}

// ===========================================================================
// 4. Expert FFN per-expert bounds (IBP)
// ===========================================================================

#[test]
fn test_qwen3_moe_full_expert_ffn_ibp() {
    // Single expert SwiGLU FFN: gate_proj -> SiLU -> mul(up_proj) -> down_proj
    let mut b = TensorBlockBuilder::new("q3mf_expert_ffn");
    let input = b.add_input("x", &[SEQ_LEN, HIDDEN_DIM]);
    let out = add_swiglu_ffn(&mut b, input, SEQ_LEN, HIDDEN_DIM, FFN_DIM, "exp0");
    let def = b.build(out).expect("valid expert FFN kernel");

    let mut bindings = vec![TensorParamBinding::Variable];
    bindings.extend(swiglu_ffn_bindings(HIDDEN_DIM, FFN_DIM));
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let inp = seq_bounds(SEQ_LEN, HIDDEN_DIM, 1.0);

    let output = graph.propagate_ibp(&inp).expect("IBP propagation");
    assert_bounds_valid(&output);
    assert_eq!(output.lower_upper().0.shape(), &[SEQ_LEN, HIDDEN_DIM]);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Q3MF expert FFN IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");
    assert!(lo_min.is_finite() && hi_max.is_finite());
}

// ===========================================================================
// 5. MoE weighted combine output bounds (IBP)
// ===========================================================================

#[test]
fn test_qwen3_moe_full_weighted_combine_ibp() {
    // Two expert outputs weighted-summed: g0 * expert0 + g1 * expert1
    // Model with constant gate weights (simulating top-2 selection).
    let mut b = TensorBlockBuilder::new("q3mf_weighted_combine");
    let expert0 = b.add_input("expert0", &[SEQ_LEN, HIDDEN_DIM]);
    let expert1 = b.add_input("expert1", &[SEQ_LEN, HIDDEN_DIM]);
    let gate0 = b.add_input("gate0", &[SEQ_LEN, HIDDEN_DIM]);
    let gate1 = b.add_input("gate1", &[SEQ_LEN, HIDDEN_DIM]);

    let w0 = b.add_binary_mul(expert0, gate0, &[SEQ_LEN, HIDDEN_DIM]);
    let w1 = b.add_binary_mul(expert1, gate1, &[SEQ_LEN, HIDDEN_DIM]);
    let out = b.add_binary_add(w0, w1, &[SEQ_LEN, HIDDEN_DIM]);
    let def = b.build(out).expect("valid weighted combine kernel");

    let g_val = 0.5f32; // top-2 equal weighting
    let bindings = vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[SEQ_LEN, HIDDEN_DIM]), 0.0)),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[SEQ_LEN, HIDDEN_DIM]), g_val)),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[SEQ_LEN, HIDDEN_DIM]), g_val)),
    ];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let inp = seq_bounds(SEQ_LEN, HIDDEN_DIM, 1.0);

    let output = graph.propagate_ibp(&inp).expect("IBP propagation");
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Q3MF weighted combine IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");
    assert!(lo_min.is_finite() && hi_max.is_finite());
}

// ===========================================================================
// 6. Top-k gating selection bounds (IBP)
// ===========================================================================

#[test]
fn test_qwen3_moe_full_topk_gating_ibp() {
    // Top-k gating: Linear -> softmax -> narrow(TOP_K)
    // Selects the top-k expert probabilities from the router output.
    let mut b = TensorBlockBuilder::new("q3mf_topk_gating");
    let input = b.add_input("x", &[SEQ_LEN, HIDDEN_DIM]);
    let router_w = b.add_input("router_w", &[NUM_EXPERTS, HIDDEN_DIM]);
    let logits = b.add_linear(input, router_w, None, &[SEQ_LEN, NUM_EXPERTS]);
    let probs = b.add_softmax(logits, -1, &[SEQ_LEN, NUM_EXPERTS]);
    // Narrow to top-k experts (first TOP_K columns as proxy)
    let topk = b.add_narrow(probs, 1, 0, TOP_K, &[SEQ_LEN, TOP_K]);
    let def = b.build(topk).expect("valid top-k gating kernel");

    let bindings = vec![
        TensorParamBinding::Variable,
        weight(&[NUM_EXPERTS, HIDDEN_DIM]),
    ];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let inp = seq_bounds(SEQ_LEN, HIDDEN_DIM, 1.0);

    let output = graph.propagate_ibp(&inp).expect("IBP propagation");
    assert_bounds_valid(&output);
    assert_eq!(output.lower_upper().0.shape(), &[SEQ_LEN, TOP_K]);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Q3MF top-k gating IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");
    assert!(
        lo_min >= -1e-5,
        "top-k lower bound must be >= 0, got {lo_min}"
    );
    assert!(
        hi_max <= 1.0 + 1e-5,
        "top-k upper bound must be <= 1, got {hi_max}"
    );
}

// ===========================================================================
// 7. Cross-attention vision-to-text bounds (IBP)
// ===========================================================================

#[test]
fn test_qwen3_moe_full_cross_attention_ibp() {
    // Cross-attention: text queries attend to vision keys/values.
    // Q from text [SEQ, DIM], KV from vision [SEQ, DIM].
    let mut b = TensorBlockBuilder::new("q3mf_cross_attn");
    let q_input = b.add_input("text_q", &[SEQ_LEN, HIDDEN_DIM]);
    let kv_input = b.add_input("vision_kv", &[SEQ_LEN, HIDDEN_DIM]);

    let qw = b.add_input("q_w", &[HIDDEN_DIM, HIDDEN_DIM]);
    let kw = b.add_input("k_w", &[HIDDEN_DIM, HIDDEN_DIM]);
    let vw = b.add_input("v_w", &[HIDDEN_DIM, HIDDEN_DIM]);
    let ow = b.add_input("o_w", &[HIDDEN_DIM, HIDDEN_DIM]);

    let out = b
        .add_multi_head_cross_attention(
            q_input,
            kv_input,
            qw,
            kw,
            vw,
            ow,
            NUM_HEADS,
            AttentionMask::Standard,
            &[SEQ_LEN, HIDDEN_DIM],
        )
        .expect("valid cross-attention");
    let def = b.build(out).expect("valid cross-attention kernel");

    let bindings = vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[SEQ_LEN, HIDDEN_DIM]), 0.01)),
        weight(&[HIDDEN_DIM, HIDDEN_DIM]),
        weight(&[HIDDEN_DIM, HIDDEN_DIM]),
        weight(&[HIDDEN_DIM, HIDDEN_DIM]),
        weight(&[HIDDEN_DIM, HIDDEN_DIM]),
    ];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let inp = seq_bounds(SEQ_LEN, HIDDEN_DIM, 1.0);

    let output = graph.propagate_ibp(&inp).expect("IBP propagation");
    assert_bounds_valid(&output);
    assert_eq!(output.lower_upper().0.shape(), &[SEQ_LEN, HIDDEN_DIM]);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Q3MF cross-attention IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");
    assert!(lo_min.is_finite() && hi_max.is_finite());
}

// ===========================================================================
// 8. SwiGLU FFN in standard decoder blocks (IBP)
// ===========================================================================

#[test]
fn test_qwen3_moe_full_swiglu_decoder_ibp() {
    // SwiGLU FFN: gate_proj -> SiLU -> mul(up_proj) -> down_proj
    let mut b = TensorBlockBuilder::new("q3mf_swiglu_decoder");
    let input = b.add_input("x", &[SEQ_LEN, HIDDEN_DIM]);
    let out = add_swiglu_ffn(&mut b, input, SEQ_LEN, HIDDEN_DIM, FFN_DIM, "dec");
    let def = b.build(out).expect("valid SwiGLU decoder kernel");

    let mut bindings = vec![TensorParamBinding::Variable];
    bindings.extend(swiglu_ffn_bindings(HIDDEN_DIM, FFN_DIM));
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let inp = seq_bounds(SEQ_LEN, HIDDEN_DIM, 1.0);

    let output = graph.propagate_ibp(&inp).expect("IBP propagation");
    assert_bounds_valid(&output);
    assert_eq!(output.lower_upper().0.shape(), &[SEQ_LEN, HIDDEN_DIM]);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Q3MF SwiGLU decoder IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");
    assert!(lo_min.is_finite() && hi_max.is_finite());
}

// ===========================================================================
// 9. RMSNorm normalization bounds (IBP)
// ===========================================================================

#[test]
fn test_qwen3_moe_full_rmsnorm_ibp() {
    let mut b = TensorBlockBuilder::new("q3mf_rmsnorm");
    let input = b.add_input("x", &[SEQ_LEN, HIDDEN_DIM]);
    let out = add_rmsnorm(&mut b, input, &[SEQ_LEN, HIDDEN_DIM], HIDDEN_DIM, "rms");
    let def = b.build(out).expect("valid RMSNorm kernel");

    let mut bindings = vec![TensorParamBinding::Variable];
    bindings.extend(rmsnorm_bindings(HIDDEN_DIM));
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let inp = seq_bounds(SEQ_LEN, HIDDEN_DIM, 5.0);

    let output = graph.propagate_ibp(&inp).expect("IBP propagation");
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Q3MF RMSNorm IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");
    assert!(lo_min.is_finite() && hi_max.is_finite());
}

// ===========================================================================
// 10. Residual connections through decoder (IBP)
// ===========================================================================

#[test]
fn test_qwen3_moe_full_residual_decoder_ibp() {
    // Compare 1-block vs 2-block decoder to observe residual bound growth.
    let build_n_blocks = |n: usize| -> BoundedTensor {
        let mut b = TensorBlockBuilder::new(&format!("q3mf_resid_{n}blk"));
        let input = b.add_input("x", &[SEQ_LEN, HIDDEN_DIM]);
        let mut x = input;
        for i in 0..n {
            x = add_decoder_block(
                &mut b,
                x,
                SEQ_LEN,
                HIDDEN_DIM,
                FFN_DIM,
                NUM_HEADS,
                &format!("d{i}"),
            );
        }
        let def = b.build(x).expect("valid n-block decoder");
        let mut bindings = vec![TensorParamBinding::Variable];
        for _ in 0..n {
            bindings.extend(decoder_block_bindings(HIDDEN_DIM, FFN_DIM));
        }
        let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");
        let inp = seq_bounds(SEQ_LEN, HIDDEN_DIM, 1.0);
        graph.propagate_ibp(&inp).expect("IBP")
    };

    let out1 = build_n_blocks(1);
    let out2 = build_n_blocks(2);
    assert_bounds_valid(&out1);
    assert_bounds_valid(&out2);

    let (l1, h1) = bounds_min_max(&out1);
    let (l2, h2) = bounds_min_max(&out2);
    let width1 = h1 - l1;
    let width2 = h2 - l2;
    eprintln!("Q3MF residual: 1-blk width={width1:.4}, 2-blk width={width2:.4}");
    assert!(width1.is_finite() && width2.is_finite());
}

// ===========================================================================
// 11. KV cache append bounds (IBP)
// ===========================================================================

#[test]
fn test_qwen3_moe_full_kv_cache_append_ibp() {
    // KV cache append: concat past_kv [SEQ, DIM] with new_kv [1, DIM] -> [SEQ+1, DIM]
    let cache_len = SEQ_LEN;
    let total_len = cache_len + 1;

    let mut b = TensorBlockBuilder::new("q3mf_kv_cache_append");
    let past_kv = b.add_input("past_kv", &[cache_len, HIDDEN_DIM]);
    let new_kv = b.add_input("new_kv", &[1, HIDDEN_DIM]);
    let out = b.add_concat(&[past_kv, new_kv], 0, &[total_len, HIDDEN_DIM]);
    let def = b.build(out).expect("valid KV cache append kernel");

    let bindings = vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[1, HIDDEN_DIM]), 0.01)),
    ];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let inp = seq_bounds(cache_len, HIDDEN_DIM, 1.0);

    let output = graph.propagate_ibp(&inp).expect("IBP propagation");
    assert_bounds_valid(&output);
    assert_eq!(output.lower_upper().0.shape(), &[total_len, HIDDEN_DIM]);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Q3MF KV cache append IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");
    assert!(lo_min.is_finite() && hi_max.is_finite());
}

// ===========================================================================
// 12. LM head logit projection bounds (IBP)
// ===========================================================================

#[test]
fn test_qwen3_moe_full_lm_head_ibp() {
    // LM head: RMSNorm -> Linear(DIM, VOCAB) -> logits
    let mut b = TensorBlockBuilder::new("q3mf_lm_head");
    let input = b.add_input("x", &[SEQ_LEN, HIDDEN_DIM]);
    let normed = add_rmsnorm(&mut b, input, &[SEQ_LEN, HIDDEN_DIM], HIDDEN_DIM, "final");
    let lm_w = b.add_input("lm_head_w", &[VOCAB_SIZE, HIDDEN_DIM]);
    let out = b.add_linear(normed, lm_w, None, &[SEQ_LEN, VOCAB_SIZE]);
    let def = b.build(out).expect("valid LM head kernel");

    let mut bindings = vec![TensorParamBinding::Variable];
    bindings.extend(rmsnorm_bindings(HIDDEN_DIM));
    bindings.push(weight(&[VOCAB_SIZE, HIDDEN_DIM]));
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let inp = seq_bounds(SEQ_LEN, HIDDEN_DIM, 1.0);

    let output = graph.propagate_ibp(&inp).expect("IBP propagation");
    assert_bounds_valid(&output);
    assert_eq!(output.lower_upper().0.shape(), &[SEQ_LEN, VOCAB_SIZE]);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Q3MF LM head IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");
    assert!(lo_min.is_finite() && hi_max.is_finite());
}

// ===========================================================================
// 13. Full vision encoder pipeline (CROWN)
// ===========================================================================

#[test]
fn test_qwen3_moe_full_vision_encoder_crown() {
    // Vision encoder: 2 ViT blocks + final RMSNorm (from sequence input)
    let mut b = TensorBlockBuilder::new("q3mf_vision_encoder_crown");
    let input = b.add_input("patch_embeds", &[SEQ_LEN, HIDDEN_DIM]);
    let l1 = add_decoder_block(
        &mut b, input, SEQ_LEN, HIDDEN_DIM, FFN_DIM, NUM_HEADS, "vit0",
    );
    let l2 = add_decoder_block(&mut b, l1, SEQ_LEN, HIDDEN_DIM, FFN_DIM, NUM_HEADS, "vit1");
    let out = add_rmsnorm(&mut b, l2, &[SEQ_LEN, HIDDEN_DIM], HIDDEN_DIM, "final");
    let def = b.build(out).expect("valid vision encoder kernel");

    let mut bindings = vec![TensorParamBinding::Variable];
    bindings.extend(decoder_block_bindings(HIDDEN_DIM, FFN_DIM));
    bindings.extend(decoder_block_bindings(HIDDEN_DIM, FFN_DIM));
    bindings.extend(rmsnorm_bindings(HIDDEN_DIM));
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let inp = seq_bounds(SEQ_LEN, HIDDEN_DIM, 1.0);

    let (method, crown_output, _) = assert_crown_tighter_when_not_fallback(&graph, &inp);
    let (clo, chi) = bounds_min_max(&crown_output);
    eprintln!("Q3MF vision encoder CROWN ({method:?}): bounds=[{clo:.6}, {chi:.6}]");
}

// ===========================================================================
// 14. Full decoder block pipeline (IBP)
// ===========================================================================

#[test]
fn test_qwen3_moe_full_decoder_block_ibp() {
    // Full decoder block: RMSNorm -> MHA -> residual -> RMSNorm -> SwiGLU -> residual
    let mut b = TensorBlockBuilder::new("q3mf_decoder_block_ibp");
    let input = b.add_input("x", &[SEQ_LEN, HIDDEN_DIM]);
    let out = add_decoder_block(
        &mut b, input, SEQ_LEN, HIDDEN_DIM, FFN_DIM, NUM_HEADS, "dec0",
    );
    let def = b.build(out).expect("valid decoder block kernel");

    let mut bindings = vec![TensorParamBinding::Variable];
    bindings.extend(decoder_block_bindings(HIDDEN_DIM, FFN_DIM));
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let inp = seq_bounds(SEQ_LEN, HIDDEN_DIM, 1.0);

    let output = graph.propagate_ibp(&inp).expect("IBP propagation");
    assert_bounds_valid(&output);
    assert_eq!(output.lower_upper().0.shape(), &[SEQ_LEN, HIDDEN_DIM]);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Q3MF decoder block IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");
    assert!(lo_min.is_finite() && hi_max.is_finite());
}

// ===========================================================================
// 15. Two-block decoder composition (IBP)
// ===========================================================================

#[test]
fn test_qwen3_moe_full_two_block_decoder_ibp() {
    let mut b = TensorBlockBuilder::new("q3mf_2blk_decoder");
    let input = b.add_input("x", &[SEQ_LEN, HIDDEN_DIM]);
    let l1 = add_decoder_block(
        &mut b, input, SEQ_LEN, HIDDEN_DIM, FFN_DIM, NUM_HEADS, "dec0",
    );
    let out = add_decoder_block(&mut b, l1, SEQ_LEN, HIDDEN_DIM, FFN_DIM, NUM_HEADS, "dec1");
    let def = b.build(out).expect("valid 2-block decoder kernel");

    let mut bindings = vec![TensorParamBinding::Variable];
    bindings.extend(decoder_block_bindings(HIDDEN_DIM, FFN_DIM));
    bindings.extend(decoder_block_bindings(HIDDEN_DIM, FFN_DIM));
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let inp = seq_bounds(SEQ_LEN, HIDDEN_DIM, 1.0);

    let output = graph.propagate_ibp(&inp).expect("IBP propagation");
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Q3MF 2-block decoder IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");
    assert!(lo_min.is_finite() && hi_max.is_finite());
}

// ===========================================================================
// 16. Embedding + position encoding bounds (IBP)
// ===========================================================================

#[test]
fn test_qwen3_moe_full_embedding_pos_encoding_ibp() {
    // Token embedding + positional encoding addition
    let mut b = TensorBlockBuilder::new("q3mf_embed_pos_enc");
    let token_embed = b.add_input("token_embed", &[SEQ_LEN, HIDDEN_DIM]);
    let pos_embed = b.add_input("pos_embed", &[SEQ_LEN, HIDDEN_DIM]);
    let out = b.add_binary_add(token_embed, pos_embed, &[SEQ_LEN, HIDDEN_DIM]);
    let def = b.build(out).expect("valid embedding + pos encoding kernel");

    let pe_data = ArrayD::from_elem(IxDyn(&[SEQ_LEN, HIDDEN_DIM]), 0.01f32);
    let bindings = vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(pe_data),
    ];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let inp = seq_bounds(SEQ_LEN, HIDDEN_DIM, 1.0);

    let output = graph.propagate_ibp(&inp).expect("IBP propagation");
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Q3MF embedding + pos encoding IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");
    assert!(lo_min < 0.0, "lower bound shifted by PE");
    assert!(hi_max > 0.0, "upper bound shifted by PE");
}

// ===========================================================================
// 17. Multi-expert load balancing loss bounds (IBP)
// ===========================================================================

#[test]
fn test_qwen3_moe_full_load_balancing_loss_ibp() {
    // Load balancing loss: fraction_of_tokens_per_expert * average_gate_per_expert
    // Both are averages/sums over softmax outputs, so bounded in [0, 1].
    // Here we model: softmax -> per-expert average (reduce mean along seq dim).
    let mut b = TensorBlockBuilder::new("q3mf_load_balance_loss");
    let input = b.add_input("x", &[SEQ_LEN, HIDDEN_DIM]);
    let router_w = b.add_input("router_w", &[NUM_EXPERTS, HIDDEN_DIM]);
    let logits = b.add_linear(input, router_w, None, &[SEQ_LEN, NUM_EXPERTS]);
    let probs = b.add_softmax(logits, -1, &[SEQ_LEN, NUM_EXPERTS]);
    // Reduce mean over sequence dimension -> [1, NUM_EXPERTS]
    let avg_gates = b.add_reduce(
        probs,
        nn_dsl::tensor_ir::ReduceOp::Mean,
        0,
        true,
        &[1, NUM_EXPERTS],
    );
    let def = b.build(avg_gates).expect("valid load balance loss kernel");

    let bindings = vec![
        TensorParamBinding::Variable,
        weight(&[NUM_EXPERTS, HIDDEN_DIM]),
    ];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let inp = seq_bounds(SEQ_LEN, HIDDEN_DIM, 1.0);

    let output = graph.propagate_ibp(&inp).expect("IBP propagation");
    assert_bounds_valid(&output);
    assert_eq!(output.lower_upper().0.shape(), &[1, NUM_EXPERTS]);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Q3MF load balance loss IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");
    assert!(lo_min.is_finite() && hi_max.is_finite());
}

// ===========================================================================
// 18. End-to-end vision-to-logit pipeline (IBP)
// ===========================================================================

#[test]
fn test_qwen3_moe_full_e2e_vision_to_logit_ibp() {
    // End-to-end: patch_embed -> ViT block -> projection -> decoder block -> LM head
    let in_ch = 3;
    let img_h = 8;
    let img_w = 8;
    let out_h = img_h / PATCH_DIM;
    let out_w = img_w / PATCH_DIM;
    let num_patches = out_h * out_w;

    let mut b = TensorBlockBuilder::new("q3mf_e2e_vision_to_logit");

    // -- Patch embedding --
    let image = b.add_input("image", &[in_ch, img_h, img_w]);
    let conv_w = b.add_input("conv_w", &[HIDDEN_DIM, in_ch, PATCH_DIM, PATCH_DIM]);
    let conv_b = b.add_input("conv_b", &[HIDDEN_DIM]);
    let conv_out = b.add_conv2d(
        image,
        conv_w,
        Some(conv_b),
        PATCH_DIM,
        PATCH_DIM,
        0,
        0,
        &[HIDDEN_DIM, out_h, out_w],
    );
    let reshaped = b.add_reshape(conv_out, &[HIDDEN_DIM, num_patches]);
    let transposed = b.add_transpose(reshaped, &[1, 0], &[num_patches, HIDDEN_DIM]);

    // -- Vision encoder block --
    let enc_out = add_decoder_block(
        &mut b,
        transposed,
        num_patches,
        HIDDEN_DIM,
        FFN_DIM,
        NUM_HEADS,
        "vit0",
    );

    // -- Vision-to-text projection (Linear) --
    let proj_w = b.add_input("proj_w", &[HIDDEN_DIM, HIDDEN_DIM]);
    let projected = b.add_linear(enc_out, proj_w, None, &[num_patches, HIDDEN_DIM]);

    // -- Decoder block --
    let dec_out = add_decoder_block(
        &mut b,
        projected,
        num_patches,
        HIDDEN_DIM,
        FFN_DIM,
        NUM_HEADS,
        "dec0",
    );

    // -- LM head: RMSNorm -> Linear -> output logits --
    let normed = add_rmsnorm(
        &mut b,
        dec_out,
        &[num_patches, HIDDEN_DIM],
        HIDDEN_DIM,
        "final",
    );
    let lm_w = b.add_input("lm_w", &[VOCAB_SIZE, HIDDEN_DIM]);
    let logits = b.add_linear(normed, lm_w, None, &[num_patches, VOCAB_SIZE]);
    let def = b.build(logits).expect("valid e2e vision-to-logit kernel");

    // -- Bindings --
    let mut bindings = vec![
        TensorParamBinding::Variable,                       // image
        weight(&[HIDDEN_DIM, in_ch, PATCH_DIM, PATCH_DIM]), // conv_w
        bias_zero(&[HIDDEN_DIM]),                           // conv_b
    ];
    // ViT encoder block
    bindings.extend(decoder_block_bindings(HIDDEN_DIM, FFN_DIM));
    // Vision projection
    bindings.push(weight(&[HIDDEN_DIM, HIDDEN_DIM]));
    // Decoder block
    bindings.extend(decoder_block_bindings(HIDDEN_DIM, FFN_DIM));
    // Final RMSNorm
    bindings.extend(rmsnorm_bindings(HIDDEN_DIM));
    // LM head weight
    bindings.push(weight(&[VOCAB_SIZE, HIDDEN_DIM]));

    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = image_bounds(in_ch, img_h, img_w);

    let output = graph.propagate_ibp(&input).expect("IBP propagation");
    assert_bounds_valid(&output);
    assert_eq!(output.lower_upper().0.shape(), &[num_patches, VOCAB_SIZE]);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Q3MF e2e vision-to-logit IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");
    assert!(lo_min.is_finite() && hi_max.is_finite());
}

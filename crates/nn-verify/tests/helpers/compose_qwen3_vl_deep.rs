// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Deep NY compose tests for Qwen3-VL subgraphs.
//!
//! These tests verify bounds propagation through intermediate-depth compositions
//! of the Qwen3-VL vision-language model. They bridge the gap between the
//! existing sub-block tests (patch embedding, window attention, MoE routing,
//! SwiGLU FFN) and full end-to-end tests by exercising compositions at
//! increasing depth:
//!
//! 1. **Window attention + M-RoPE** -- Local self-attention with rotary
//!    position encoding applied to Q/K before attention (IBP + CROWN).
//!
//! 2. **Full vision encoder block** -- RMSNorm -> Attention -> residual ->
//!    RMSNorm -> SwiGLU FFN -> residual. One ViT block (IBP + CROWN).
//!
//! 3. **2-block vision stack** -- Depth composition with widening analysis.
//!    Compares 1-block vs 2-block IBP bounds width (IBP).
//!
//! 4. **Vision-language projection** -- Linear mapping from vision encoder
//!    features to LM embedding space with verify-and-record (IBP + CROWN).
//!
//! 5. **SwiGLU decoder + GQA** -- Full decoder layer: RMSNorm -> causal
//!    GQA attention -> residual -> RMSNorm -> SwiGLU FFN -> residual (IBP).
//!
//! 6. **MoE routing analysis** -- Softmax expert gate with CROWN
//!    linearization. Verifies probability bounds in [0, 1] (IBP + CROWN).
//!
//! 7. **Full VLM pipeline** -- Patch embedding -> 2 encoder blocks ->
//!    vision projection -> decoder FFN. End-to-end compose (IBP).
//!
//! 8. **3D patch embedding** -- Conv2d with temporal dimension modeled
//!    as an extra input channel (IBP).
//!
//! Architecture references:
//! - Qwen2-VL / Qwen3-VL (Alibaba): Vision-language model with 3D patch
//!   embedding, M-RoPE, window attention, SwiGLU, GQA, and optional MoE
//! - RMSNorm (Zhang & Sennrich, 2019): replaces LayerNorm in Qwen
//! - SwiGLU (Shazeer, 2020): SiLU-gated FFN in Qwen decoder
//! - GQA (Ainslie et al., 2023): Grouped-Query Attention in Qwen decoder
//! - MoE (Fedus et al., 2022): Mixture-of-Experts routing
//!
//! Dimensions are small for fast verification (HIDDEN_DIM=16, SEQ_LEN=4).
//! All tests use IbpValidated soundness mode per nn engineering rules
//! (Sound refuses linearization for normalization layers).

use super::common::{
    assert_bounds_valid, assert_crown_tighter_when_not_fallback, bounds_min_max, uniform_bounds,
    verify_and_assert,
};
use nn_dsl::tensor_block_builder::TensorBlockBuilder;
use nn_dsl::tensor_ir::{AttentionMask, TensorKernelDef};
use nn_verify::{tensor_kernel_to_graph, BoundedTensor, TensorParamBinding};
use ndarray::{ArrayD, IxDyn};

// ---------------------------------------------------------------------------
// Dimensions -- small for fast verification, structurally representative
// ---------------------------------------------------------------------------

/// Sequence length (number of patches / tokens).
const SEQ_LEN: usize = 4;
/// Hidden dimension (tiny Qwen3-VL hidden size).
const HIDDEN_DIM: usize = 16;
/// Number of attention heads.
const NUM_HEADS: usize = 4;
/// Head dimension = HIDDEN_DIM / NUM_HEADS.
const HEAD_DIM: usize = HIDDEN_DIM / NUM_HEADS; // 4
/// Window size for windowed attention.
const WINDOW_SIZE: usize = 2;
/// FFN intermediate dimension (4x hidden_dim).
const FFN_DIM: usize = 64;
/// Number of KV heads for GQA (half of NUM_HEADS).
const NUM_KV_HEADS: usize = 2;
/// KV dimension = NUM_KV_HEADS * HEAD_DIM.
const KV_DIM: usize = NUM_KV_HEADS * HEAD_DIM; // 8
/// Number of MoE experts.
const NUM_EXPERTS: usize = 4;
/// Patch size for Conv2d patch embedding.
const PATCH_SIZE: usize = 4;
/// Image spatial size (square).
const IMG_SIZE: usize = 8;
/// Grid size = IMG_SIZE / PATCH_SIZE.
const GRID_SIZE: usize = IMG_SIZE / PATCH_SIZE; // 2
/// Number of patches = GRID_SIZE * GRID_SIZE.
const NUM_PATCHES: usize = GRID_SIZE * GRID_SIZE; // 4
/// Input channels (RGB).
const IN_CHANNELS: usize = 3;
/// Temporal frames for 3D patch embedding.
const TEMPORAL_FRAMES: usize = 2;
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

/// Image-domain input bounds: pixels in [0, 1].
fn image_bounds_01(shape: &[usize]) -> BoundedTensor {
    BoundedTensor::new(
        ArrayD::from_elem(IxDyn(shape), 0.0f32),
        ArrayD::from_elem(IxDyn(shape), 1.0f32),
    )
    .expect("valid image bounds [0, 1]")
}

/// Build M-RoPE cos/sin tensors for a given seq_len and head_dim.
fn build_mrope_cos_sin(seq_len: usize, head_dim: usize) -> (ArrayD<f32>, ArrayD<f32>) {
    let n = seq_len * head_dim;
    let section_size = head_dim / 3;
    let mut cos_data = Vec::with_capacity(n);
    let mut sin_data = Vec::with_capacity(n);
    for t in 0..seq_len {
        for d in 0..head_dim {
            let base = if d < section_size {
                10000.0_f64
            } else if d < 2 * section_size {
                5000.0_f64
            } else {
                5000.0_f64
            };
            let freq =
                (t as f64) / base.powf(2.0 * (d % section_size.max(1)) as f64 / head_dim as f64);
            cos_data.push(freq.cos() as f32);
            sin_data.push(freq.sin() as f32);
        }
    }
    let cos = ArrayD::from_shape_vec(IxDyn(&[seq_len, head_dim]), cos_data).expect("valid cos");
    let sin = ArrayD::from_shape_vec(IxDyn(&[seq_len, head_dim]), sin_data).expect("valid sin");
    (cos, sin)
}

// ===========================================================================
// Helper: build a single vision encoder block subgraph
// ===========================================================================

/// Adds one vision encoder block to a TensorBlockBuilder.
///
/// Block structure: RMSNorm -> Attention -> residual -> RMSNorm -> SwiGLU FFN -> residual.
///
/// Returns the output node ID and pushes weight bindings into `bindings`.
fn add_vision_encoder_block(
    b: &mut TensorBlockBuilder,
    input: nn_dsl::TensorNodeId,
    block_idx: usize,
    bindings: &mut Vec<TensorParamBinding>,
) -> nn_dsl::TensorNodeId {
    let shape = [SEQ_LEN, HIDDEN_DIM];
    let ffn_shape = [SEQ_LEN, FFN_DIM];
    let scale = 1.0 / (HEAD_DIM as f32).sqrt();

    // Pre-attention RMSNorm
    let norm1_eps = b.add_input(&format!("b{block_idx}_norm1_eps"), &[1]);
    let norm1_w = b.add_input(&format!("b{block_idx}_norm1_w"), &[HIDDEN_DIM]);
    let normed1 = b.add_rms_norm(input, norm1_eps, 1, norm1_w, &shape);

    // Self-attention (Q/K/V projections + softmax attention + out projection)
    let q_w = b.add_input(&format!("b{block_idx}_q_w"), &[HIDDEN_DIM, HIDDEN_DIM]);
    let k_w = b.add_input(&format!("b{block_idx}_k_w"), &[HIDDEN_DIM, HIDDEN_DIM]);
    let v_w = b.add_input(&format!("b{block_idx}_v_w"), &[HIDDEN_DIM, HIDDEN_DIM]);
    let out_w = b.add_input(&format!("b{block_idx}_out_w"), &[HIDDEN_DIM, HIDDEN_DIM]);

    let q = b.add_linear(normed1, q_w, None, &shape);
    let k = b.add_linear(normed1, k_w, None, &shape);
    let v = b.add_linear(normed1, v_w, None, &shape);
    let attn = b.add_attention(q, k, v, AttentionMask::Standard, Some(scale), &shape);
    let attn_out = b.add_linear(attn, out_w, None, &shape);

    // Residual after attention
    let res1 = b.add_binary_add(input, attn_out, &shape);

    // Pre-FFN RMSNorm
    let norm2_eps = b.add_input(&format!("b{block_idx}_norm2_eps"), &[1]);
    let norm2_w = b.add_input(&format!("b{block_idx}_norm2_w"), &[HIDDEN_DIM]);
    let normed2 = b.add_rms_norm(res1, norm2_eps, 1, norm2_w, &shape);

    // SwiGLU FFN: silu(gate_proj(x)) * up_proj(x) -> down_proj
    let gate_w = b.add_input(&format!("b{block_idx}_gate_w"), &[FFN_DIM, HIDDEN_DIM]);
    let up_w = b.add_input(&format!("b{block_idx}_up_w"), &[FFN_DIM, HIDDEN_DIM]);
    let down_w = b.add_input(&format!("b{block_idx}_down_w"), &[HIDDEN_DIM, FFN_DIM]);

    let gate = b.add_linear(normed2, gate_w, None, &ffn_shape);
    let gate_sig = b.add_sigmoid(gate, &ffn_shape);
    let gate_act = b.add_binary_mul(gate, gate_sig, &ffn_shape);
    let up = b.add_linear(normed2, up_w, None, &ffn_shape);
    let hidden = b.add_binary_mul(gate_act, up, &ffn_shape);
    let ffn_out = b.add_linear(hidden, down_w, None, &shape);

    // Residual after FFN
    let out = b.add_binary_add(res1, ffn_out, &shape);

    // Push bindings for this block (11 weight params)
    let norm_w_arr = ones(&[HIDDEN_DIM]);
    let qkvo = w(&[HIDDEN_DIM, HIDDEN_DIM]);
    bindings.push(TensorParamBinding::ConstantScalar(1e-5)); // norm1_eps
    bindings.push(TensorParamBinding::ConstantTensor(norm_w_arr.clone())); // norm1_w
    bindings.push(TensorParamBinding::ConstantTensor(qkvo.clone())); // q_w
    bindings.push(TensorParamBinding::ConstantTensor(qkvo.clone())); // k_w
    bindings.push(TensorParamBinding::ConstantTensor(qkvo.clone())); // v_w
    bindings.push(TensorParamBinding::ConstantTensor(qkvo)); // out_w
    bindings.push(TensorParamBinding::ConstantScalar(1e-5)); // norm2_eps
    bindings.push(TensorParamBinding::ConstantTensor(norm_w_arr)); // norm2_w
    bindings.push(TensorParamBinding::ConstantTensor(w(&[
        FFN_DIM, HIDDEN_DIM,
    ]))); // gate_w
    bindings.push(TensorParamBinding::ConstantTensor(w(&[
        FFN_DIM, HIDDEN_DIM,
    ]))); // up_w
    bindings.push(TensorParamBinding::ConstantTensor(w(&[
        HIDDEN_DIM, FFN_DIM,
    ]))); // down_w

    out
}

// ===========================================================================
// 1. Window attention + M-RoPE: local self-attention with RoPE
// ===========================================================================

/// Build windowed self-attention with M-RoPE applied to Q/K.
///
/// Input: `[SEQ_LEN, HIDDEN_DIM]` (Variable).
/// Output: `[SEQ_LEN, HIDDEN_DIM]`.
///
/// M-RoPE applies cos/sin multiplication to Q and K before attention.
/// Window attention is modeled using standard attention on window-sized
/// sequences (WINDOW_SIZE=2).
fn build_window_attn_mrope_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("qwen3_vl_deep_window_attn_mrope");

    // Use WINDOW_SIZE for local attention scope
    let win_shape = [WINDOW_SIZE, HIDDEN_DIM];

    let input = b.add_input("x", &win_shape);
    let q_w = b.add_input("q_weight", &[HIDDEN_DIM, HIDDEN_DIM]);
    let k_w = b.add_input("k_weight", &[HIDDEN_DIM, HIDDEN_DIM]);
    let v_w = b.add_input("v_weight", &[HIDDEN_DIM, HIDDEN_DIM]);
    let out_w = b.add_input("out_weight", &[HIDDEN_DIM, HIDDEN_DIM]);
    let cos_pe = b.add_input("cos_mrope", &[WINDOW_SIZE, HIDDEN_DIM]);
    let sin_pe = b.add_input("sin_mrope", &[WINDOW_SIZE, HIDDEN_DIM]);

    // Q/K projections
    let q = b.add_linear(input, q_w, None, &win_shape);
    let k = b.add_linear(input, k_w, None, &win_shape);
    let v = b.add_linear(input, v_w, None, &win_shape);

    // Apply M-RoPE to Q: q_rot = q * cos + q * sin (simplified rotation)
    let q_cos = b.add_binary_mul(q, cos_pe, &win_shape);
    let q_sin = b.add_binary_mul(q, sin_pe, &win_shape);
    let q_rope = b.add_binary_add(q_cos, q_sin, &win_shape);

    // Apply M-RoPE to K: k_rot = k * cos + k * sin
    let k_cos = b.add_binary_mul(k, cos_pe, &win_shape);
    let k_sin = b.add_binary_mul(k, sin_pe, &win_shape);
    let k_rope = b.add_binary_add(k_cos, k_sin, &win_shape);

    // Attention on rotated Q/K with V
    let scale = 1.0 / (HEAD_DIM as f32).sqrt();
    let attn = b.add_attention(
        q_rope,
        k_rope,
        v,
        AttentionMask::Standard,
        Some(scale),
        &win_shape,
    );
    let out = b.add_linear(attn, out_w, None, &win_shape);

    // Residual
    let result = b.add_binary_add(input, out, &win_shape);

    b.build(result)
        .expect("valid window attention + M-RoPE kernel")
}

fn window_attn_mrope_bindings() -> Vec<TensorParamBinding> {
    let wp = w(&[HIDDEN_DIM, HIDDEN_DIM]);
    let (cos_arr, sin_arr) = build_mrope_cos_sin(WINDOW_SIZE, HIDDEN_DIM);
    vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(wp.clone()),
        TensorParamBinding::ConstantTensor(wp.clone()),
        TensorParamBinding::ConstantTensor(wp.clone()),
        TensorParamBinding::ConstantTensor(wp),
        TensorParamBinding::ConstantTensor(cos_arr),
        TensorParamBinding::ConstantTensor(sin_arr),
    ]
}

#[test]
fn test_qwen3_vl_deep_window_attn_mrope_ibp() {
    let def = build_window_attn_mrope_kernel();
    let bindings = window_attn_mrope_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");
    let input = uniform_bounds(&[WINDOW_SIZE, HIDDEN_DIM], 1.0);

    let output = graph.propagate_ibp(&input).expect("IBP");
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("window attn + M-RoPE IBP: [{lo_min}, {hi_max}]");
    assert!(lo_min.is_finite() && hi_max.is_finite());
}

#[test]
fn test_qwen3_vl_deep_window_attn_mrope_crown() {
    let def = build_window_attn_mrope_kernel();
    let bindings = window_attn_mrope_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");
    let input = uniform_bounds(&[WINDOW_SIZE, HIDDEN_DIM], 1.0);

    let (method, output, fallback) = assert_crown_tighter_when_not_fallback(&graph, &input);
    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("window attn + M-RoPE CROWN ({method:?}): [{lo_min}, {hi_max}]");
    if let Some(r) = &fallback {
        eprintln!("Fallback: {r}");
    }
}

// ===========================================================================
// 2. Full vision encoder block: RMSNorm -> Attention -> FFN with residuals
// ===========================================================================

/// Build a full vision encoder block with small dimensions.
///
/// Input: `[SEQ_LEN, HIDDEN_DIM]` (Variable).
/// Output: `[SEQ_LEN, HIDDEN_DIM]`.
fn build_full_vision_encoder_block_kernel() -> (TensorKernelDef, Vec<TensorParamBinding>) {
    let mut b = TensorBlockBuilder::new("qwen3_vl_deep_vision_encoder_block");
    let input = b.add_input("x", &[SEQ_LEN, HIDDEN_DIM]);
    let mut bindings = vec![TensorParamBinding::Variable];

    let out = add_vision_encoder_block(&mut b, input, 0, &mut bindings);
    let def = b.build(out).expect("valid vision encoder block kernel");
    (def, bindings)
}

#[test]
fn test_qwen3_vl_deep_vision_encoder_block_ibp() {
    let (def, bindings) = build_full_vision_encoder_block_kernel();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);

    let output = graph.propagate_ibp(&input).expect("IBP");
    assert_bounds_valid(&output);

    let (lo, _) = output.lower_upper();
    assert_eq!(lo.shape(), &[SEQ_LEN, HIDDEN_DIM], "shape mismatch");

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("vision encoder block IBP: [{lo_min}, {hi_max}]");
    assert!(lo_min.is_finite() && hi_max.is_finite());
}

#[test]
fn test_qwen3_vl_deep_vision_encoder_block_crown() {
    let (def, bindings) = build_full_vision_encoder_block_kernel();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);

    let (method, output, fallback) = assert_crown_tighter_when_not_fallback(&graph, &input);
    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("vision encoder block CROWN ({method:?}): [{lo_min}, {hi_max}]");
    if let Some(r) = &fallback {
        eprintln!("Fallback: {r}");
    }
}

// ===========================================================================
// 3. 2-block vision stack with widening analysis
// ===========================================================================

/// Build a 2-block vision encoder stack.
fn build_two_block_vision_stack() -> (TensorKernelDef, Vec<TensorParamBinding>) {
    let mut b = TensorBlockBuilder::new("qwen3_vl_deep_two_block_vision");
    let input = b.add_input("x", &[SEQ_LEN, HIDDEN_DIM]);
    let mut bindings = vec![TensorParamBinding::Variable];

    let x = add_vision_encoder_block(&mut b, input, 0, &mut bindings);
    let out = add_vision_encoder_block(&mut b, x, 1, &mut bindings);

    let def = b.build(out).expect("valid 2-block vision stack kernel");
    (def, bindings)
}

#[test]
fn test_qwen3_vl_deep_two_block_vision_ibp() {
    let (def, bindings) = build_two_block_vision_stack();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);

    let output = graph.propagate_ibp(&input).expect("IBP");
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("2-block vision stack IBP: [{lo_min}, {hi_max}]");
    assert!(lo_min.is_finite() && hi_max.is_finite());
}

/// Widening analysis: compare 1-block vs 2-block IBP bounds width.
#[test]
fn test_qwen3_vl_deep_two_block_widening_analysis() {
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);

    // 1-block
    let (def1, bindings1) = build_full_vision_encoder_block_kernel();
    let graph1 = tensor_kernel_to_graph(&def1, &bindings1).expect("graph");
    let output1 = graph1.propagate_ibp(&input).expect("IBP 1-block");
    let (lo1, hi1) = bounds_min_max(&output1);
    let width1 = hi1 - lo1;

    // 2-block
    let (def2, bindings2) = build_two_block_vision_stack();
    let graph2 = tensor_kernel_to_graph(&def2, &bindings2).expect("graph");
    let output2 = graph2.propagate_ibp(&input).expect("IBP 2-block");
    let (lo2, hi2) = bounds_min_max(&output2);
    let width2 = hi2 - lo2;

    eprintln!("Widening analysis: 1-block width={width1:.4}, 2-block width={width2:.4}");
    eprintln!("  1-block: [{lo1:.4}, {hi1:.4}]");
    eprintln!("  2-block: [{lo2:.4}, {hi2:.4}]");

    // Both must be finite
    assert!(width1.is_finite(), "1-block width not finite");
    assert!(width2.is_finite(), "2-block width not finite");
}

// ===========================================================================
// 4. Vision-language projection: Linear mapping vision -> LM space
// ===========================================================================

/// Build vision-language projection: encoder output -> Linear(HIDDEN_DIM, LM_DIM).
///
/// Input: `[SEQ_LEN, HIDDEN_DIM]` (Variable, encoder output).
/// Output: `[SEQ_LEN, LM_DIM]`.
fn build_vision_language_projection_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("qwen3_vl_deep_vision_lang_proj");

    let input = b.add_input("vision_features", &[SEQ_LEN, HIDDEN_DIM]);
    let proj_w = b.add_input("proj_weight", &[LM_DIM, HIDDEN_DIM]);
    let proj_b = b.add_input("proj_bias", &[LM_DIM]);

    let out = b.add_linear(input, proj_w, Some(proj_b), &[SEQ_LEN, LM_DIM]);

    b.build(out)
        .expect("valid vision-language projection kernel")
}

fn vision_language_projection_bindings() -> Vec<TensorParamBinding> {
    vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(w(&[LM_DIM, HIDDEN_DIM])),
        TensorParamBinding::ConstantTensor(zeros(&[LM_DIM])),
    ]
}

#[test]
fn test_qwen3_vl_deep_vision_lang_proj_ibp() {
    let def = build_vision_language_projection_kernel();
    let bindings = vision_language_projection_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);

    let output = graph.propagate_ibp(&input).expect("IBP");
    assert_bounds_valid(&output);

    let (lo, _) = output.lower_upper();
    assert_eq!(
        lo.shape(),
        &[SEQ_LEN, LM_DIM],
        "projection output shape mismatch"
    );

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("vision-lang projection IBP: [{lo_min}, {hi_max}]");
    assert!(lo_min.is_finite() && hi_max.is_finite());
}

#[test]
fn test_qwen3_vl_deep_vision_lang_proj_crown() {
    let def = build_vision_language_projection_kernel();
    let bindings = vision_language_projection_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);

    let (method, output, fallback) = assert_crown_tighter_when_not_fallback(&graph, &input);
    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("vision-lang projection CROWN ({method:?}): [{lo_min}, {hi_max}]");
    if let Some(r) = &fallback {
        eprintln!("Fallback: {r}");
    }
}

#[test]
fn test_qwen3_vl_deep_vision_lang_proj_verify_and_record() {
    let def = build_vision_language_projection_kernel();
    let bindings = vision_language_projection_bindings();
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);

    let result = verify_and_assert(&def, &bindings, &input, "qwen3_vl_deep_vision_lang_proj");
    let (lo, _) = result.output_bounds.lower_upper();
    assert_eq!(lo.shape(), &[SEQ_LEN, LM_DIM]);
}

// ===========================================================================
// 5. SwiGLU decoder + GQA: full decoder layer
// ===========================================================================

/// Build a full decoder layer: RMSNorm -> causal GQA -> residual ->
/// RMSNorm -> SwiGLU FFN -> residual.
///
/// Input: `[SEQ_LEN, HIDDEN_DIM]` (Variable).
/// Output: `[SEQ_LEN, HIDDEN_DIM]`.
fn build_swiglu_decoder_gqa_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("qwen3_vl_deep_decoder_gqa_swiglu");

    let input = b.add_input("x", &[SEQ_LEN, HIDDEN_DIM]);
    let shape = [SEQ_LEN, HIDDEN_DIM];
    let kv_shape = [SEQ_LEN, KV_DIM];
    let ffn_shape = [SEQ_LEN, FFN_DIM];

    // Pre-attention RMSNorm
    let norm1_eps = b.add_input("norm1_eps", &[1]);
    let norm1_w = b.add_input("norm1_weight", &[HIDDEN_DIM]);
    let normed1 = b.add_rms_norm(input, norm1_eps, 1, norm1_w, &shape);

    // GQA: Q projects to full dim, K/V to KV_DIM
    let q_w = b.add_input("q_weight", &[KV_DIM, HIDDEN_DIM]);
    let k_w = b.add_input("k_weight", &[KV_DIM, HIDDEN_DIM]);
    let v_w = b.add_input("v_weight", &[KV_DIM, HIDDEN_DIM]);
    let out_up_w = b.add_input("out_up_weight", &[HIDDEN_DIM, KV_DIM]);

    let q = b.add_linear(normed1, q_w, None, &kv_shape);
    let k = b.add_linear(normed1, k_w, None, &kv_shape);
    let v = b.add_linear(normed1, v_w, None, &kv_shape);

    let scale = 1.0 / (HEAD_DIM as f32).sqrt();
    let attn = b.add_attention(q, k, v, AttentionMask::Causal, Some(scale), &kv_shape);

    // Project back to hidden dim
    let attn_out = b.add_linear(attn, out_up_w, None, &shape);

    // Residual after attention
    let res1 = b.add_binary_add(input, attn_out, &shape);

    // Pre-FFN RMSNorm
    let norm2_eps = b.add_input("norm2_eps", &[1]);
    let norm2_w = b.add_input("norm2_weight", &[HIDDEN_DIM]);
    let normed2 = b.add_rms_norm(res1, norm2_eps, 1, norm2_w, &shape);

    // SwiGLU FFN
    let gate_w = b.add_input("gate_weight", &[FFN_DIM, HIDDEN_DIM]);
    let up_w = b.add_input("up_weight", &[FFN_DIM, HIDDEN_DIM]);
    let down_w = b.add_input("down_weight", &[HIDDEN_DIM, FFN_DIM]);

    let gate = b.add_linear(normed2, gate_w, None, &ffn_shape);
    let gate_sig = b.add_sigmoid(gate, &ffn_shape);
    let gate_act = b.add_binary_mul(gate, gate_sig, &ffn_shape);
    let up = b.add_linear(normed2, up_w, None, &ffn_shape);
    let hidden = b.add_binary_mul(gate_act, up, &ffn_shape);
    let ffn_out = b.add_linear(hidden, down_w, None, &shape);

    // Residual after FFN
    let out = b.add_binary_add(res1, ffn_out, &shape);

    b.build(out).expect("valid decoder GQA + SwiGLU kernel")
}

fn swiglu_decoder_gqa_bindings() -> Vec<TensorParamBinding> {
    vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantScalar(1e-5), // norm1_eps
        TensorParamBinding::ConstantTensor(ones(&[HIDDEN_DIM])), // norm1_w
        TensorParamBinding::ConstantTensor(w(&[KV_DIM, HIDDEN_DIM])), // q_w
        TensorParamBinding::ConstantTensor(w(&[KV_DIM, HIDDEN_DIM])), // k_w
        TensorParamBinding::ConstantTensor(w(&[KV_DIM, HIDDEN_DIM])), // v_w
        TensorParamBinding::ConstantTensor(w(&[HIDDEN_DIM, KV_DIM])), // out_up_w
        TensorParamBinding::ConstantScalar(1e-5), // norm2_eps
        TensorParamBinding::ConstantTensor(ones(&[HIDDEN_DIM])), // norm2_w
        TensorParamBinding::ConstantTensor(w(&[FFN_DIM, HIDDEN_DIM])), // gate_w
        TensorParamBinding::ConstantTensor(w(&[FFN_DIM, HIDDEN_DIM])), // up_w
        TensorParamBinding::ConstantTensor(w(&[HIDDEN_DIM, FFN_DIM])), // down_w
    ]
}

#[test]
fn test_qwen3_vl_deep_decoder_gqa_swiglu_ibp() {
    let def = build_swiglu_decoder_gqa_kernel();
    let bindings = swiglu_decoder_gqa_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);

    let output = graph.propagate_ibp(&input).expect("IBP");
    assert_bounds_valid(&output);

    let (lo, _) = output.lower_upper();
    assert_eq!(
        lo.shape(),
        &[SEQ_LEN, HIDDEN_DIM],
        "decoder output shape mismatch"
    );

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("decoder GQA + SwiGLU IBP: [{lo_min}, {hi_max}]");
    assert!(lo_min.is_finite() && hi_max.is_finite());
}

// ===========================================================================
// 6. MoE routing analysis: softmax expert gate with CROWN
// ===========================================================================

/// Build MoE routing: Linear(HIDDEN_DIM -> NUM_EXPERTS) -> softmax.
///
/// Input: `[SEQ_LEN, HIDDEN_DIM]` (Variable).
/// Output: `[SEQ_LEN, NUM_EXPERTS]` (routing probabilities in [0, 1]).
fn build_moe_routing_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("qwen3_vl_deep_moe_routing");

    let input = b.add_input("hidden", &[SEQ_LEN, HIDDEN_DIM]);
    let router_w = b.add_input("router_weight", &[NUM_EXPERTS, HIDDEN_DIM]);

    let logits = b.add_linear(input, router_w, None, &[SEQ_LEN, NUM_EXPERTS]);
    let probs = b.add_softmax(logits, 1, &[SEQ_LEN, NUM_EXPERTS]);

    b.build(probs).expect("valid MoE routing kernel")
}

fn moe_routing_bindings() -> Vec<TensorParamBinding> {
    vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(w(&[NUM_EXPERTS, HIDDEN_DIM])),
    ]
}

#[test]
fn test_qwen3_vl_deep_moe_routing_ibp() {
    let def = build_moe_routing_kernel();
    let bindings = moe_routing_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);

    let output = graph.propagate_ibp(&input).expect("IBP");
    assert_bounds_valid(&output);

    let (lo, _) = output.lower_upper();
    assert_eq!(
        lo.shape(),
        &[SEQ_LEN, NUM_EXPERTS],
        "MoE routing shape mismatch"
    );

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("MoE routing IBP: [{lo_min}, {hi_max}]");

    // Softmax codomain is (0, 1)
    let eps = 1e-6;
    assert!(
        lo_min >= 0.0 - eps,
        "softmax lower must be >= 0, got {lo_min}"
    );
    assert!(
        hi_max <= 1.0 + eps,
        "softmax upper must be <= 1, got {hi_max}"
    );
}

#[test]
fn test_qwen3_vl_deep_moe_routing_crown() {
    let def = build_moe_routing_kernel();
    let bindings = moe_routing_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);

    let (method, output, fallback) = assert_crown_tighter_when_not_fallback(&graph, &input);
    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("MoE routing CROWN ({method:?}): [{lo_min}, {hi_max}]");
    if let Some(r) = &fallback {
        eprintln!("Fallback: {r}");
    }

    // Softmax codomain still in (0, 1)
    let eps = 1e-6;
    assert!(
        lo_min >= 0.0 - eps,
        "CROWN softmax lower must be >= 0, got {lo_min}"
    );
    assert!(
        hi_max <= 1.0 + eps,
        "CROWN softmax upper must be <= 1, got {hi_max}"
    );
}

// ===========================================================================
// 7. Full VLM pipeline: patch embed -> 2 encoder blocks -> proj -> decoder FFN
// ===========================================================================

/// Build full VLM pipeline compose.
///
/// Input: `[IN_CHANNELS, IMG_SIZE, IMG_SIZE]` (Variable, pixels [0, 1]).
/// Output: `[NUM_PATCHES, HIDDEN_DIM]` (decoder FFN output).
///
/// Pipeline:
///   Conv2d patch embed -> [NUM_PATCHES, HIDDEN_DIM]
///   -> 2 encoder blocks
///   -> Linear vision projection [NUM_PATCHES, LM_DIM]
///   -> SwiGLU decoder FFN [NUM_PATCHES, LM_DIM]
fn build_full_vlm_pipeline_kernel() -> (TensorKernelDef, Vec<TensorParamBinding>) {
    let mut b = TensorBlockBuilder::new("qwen3_vl_deep_full_vlm_pipeline");
    let shape = [NUM_PATCHES, HIDDEN_DIM];
    let proj_shape = [NUM_PATCHES, LM_DIM];
    let ffn_proj_shape = [NUM_PATCHES, FFN_DIM];

    // --- Patch embedding ---
    let image = b.add_input("image", &[IN_CHANNELS, IMG_SIZE, IMG_SIZE]);
    let patch_w = b.add_input(
        "patch_weight",
        &[HIDDEN_DIM, IN_CHANNELS, PATCH_SIZE, PATCH_SIZE],
    );
    let patch_b = b.add_input("patch_bias", &[HIDDEN_DIM]);

    let conv_out = b.add_conv2d(
        image,
        patch_w,
        Some(patch_b),
        PATCH_SIZE,
        PATCH_SIZE,
        0,
        0,
        &[HIDDEN_DIM, GRID_SIZE, GRID_SIZE],
    );
    let reshaped = b.add_reshape(conv_out, &[HIDDEN_DIM, NUM_PATCHES]);
    let x = b.add_transpose(reshaped, &[1, 0], &shape);

    let mut bindings = vec![
        TensorParamBinding::Variable, // image
        TensorParamBinding::ConstantTensor(w(&[HIDDEN_DIM, IN_CHANNELS, PATCH_SIZE, PATCH_SIZE])),
        TensorParamBinding::ConstantTensor(zeros(&[HIDDEN_DIM])),
    ];

    // --- 2 encoder blocks ---
    let x = add_vision_encoder_block(&mut b, x, 0, &mut bindings);
    let x = add_vision_encoder_block(&mut b, x, 1, &mut bindings);

    // --- Vision projection: Linear(HIDDEN_DIM -> LM_DIM) ---
    let vp_w = b.add_input("vproj_weight", &[LM_DIM, HIDDEN_DIM]);
    let vp_b = b.add_input("vproj_bias", &[LM_DIM]);
    let projected = b.add_linear(x, vp_w, Some(vp_b), &proj_shape);
    bindings.push(TensorParamBinding::ConstantTensor(w(&[LM_DIM, HIDDEN_DIM])));
    bindings.push(TensorParamBinding::ConstantTensor(zeros(&[LM_DIM])));

    // --- Decoder SwiGLU FFN on projected features ---
    let dec_gate_w = b.add_input("dec_gate_w", &[FFN_DIM, LM_DIM]);
    let dec_up_w = b.add_input("dec_up_w", &[FFN_DIM, LM_DIM]);
    let dec_down_w = b.add_input("dec_down_w", &[LM_DIM, FFN_DIM]);

    let gate = b.add_linear(projected, dec_gate_w, None, &ffn_proj_shape);
    let gate_sig = b.add_sigmoid(gate, &ffn_proj_shape);
    let gate_act = b.add_binary_mul(gate, gate_sig, &ffn_proj_shape);
    let up = b.add_linear(projected, dec_up_w, None, &ffn_proj_shape);
    let hidden = b.add_binary_mul(gate_act, up, &ffn_proj_shape);
    let ffn_out = b.add_linear(hidden, dec_down_w, None, &proj_shape);

    // Residual around decoder FFN
    let out = b.add_binary_add(projected, ffn_out, &proj_shape);

    bindings.push(TensorParamBinding::ConstantTensor(w(&[FFN_DIM, LM_DIM])));
    bindings.push(TensorParamBinding::ConstantTensor(w(&[FFN_DIM, LM_DIM])));
    bindings.push(TensorParamBinding::ConstantTensor(w(&[LM_DIM, FFN_DIM])));

    let def = b.build(out).expect("valid full VLM pipeline kernel");
    (def, bindings)
}

#[test]
fn test_qwen3_vl_deep_full_vlm_pipeline_ibp() {
    let (def, bindings) = build_full_vlm_pipeline_kernel();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");
    let input = image_bounds_01(&[IN_CHANNELS, IMG_SIZE, IMG_SIZE]);

    let output = graph.propagate_ibp(&input).expect("IBP");
    assert_bounds_valid(&output);

    let (lo, _) = output.lower_upper();
    assert_eq!(
        lo.shape(),
        &[NUM_PATCHES, LM_DIM],
        "VLM pipeline output shape mismatch"
    );

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("full VLM pipeline IBP (image [0,1]): [{lo_min}, {hi_max}]");
    assert!(lo_min.is_finite() && hi_max.is_finite());
}

// ===========================================================================
// 8. 3D patch embedding: Conv2d with temporal dimension
// ===========================================================================

/// Build a 3D patch embedding modeled with temporal-expanded channels.
///
/// Qwen3-VL uses Conv3D(3, D, (t, P, P), stride=(t, P, P)) for video.
/// Since TensorBlockBuilder lacks Conv3D, we model the temporal dimension
/// by treating TEMPORAL_FRAMES * IN_CHANNELS as input channels to Conv2D:
///   Conv2d(TEMPORAL_FRAMES * 3, D, P, stride=P).
///
/// Input: `[TEMPORAL_FRAMES * IN_CHANNELS, IMG_SIZE, IMG_SIZE]` (Variable).
/// Output: `[NUM_PATCHES, HIDDEN_DIM]` after reshape and transpose.
fn build_3d_patch_embedding_kernel() -> TensorKernelDef {
    let temporal_channels = TEMPORAL_FRAMES * IN_CHANNELS; // 6
    let mut b = TensorBlockBuilder::new("qwen3_vl_deep_3d_patch_embed");

    let input = b.add_input("video_frames", &[temporal_channels, IMG_SIZE, IMG_SIZE]);
    let weight = b.add_input(
        "patch3d_weight",
        &[HIDDEN_DIM, temporal_channels, PATCH_SIZE, PATCH_SIZE],
    );
    let bias = b.add_input("patch3d_bias", &[HIDDEN_DIM]);

    // Conv2d: [6, 8, 8] -> [D, 2, 2]
    let conv_out = b.add_conv2d(
        input,
        weight,
        Some(bias),
        PATCH_SIZE,
        PATCH_SIZE,
        0,
        0,
        &[HIDDEN_DIM, GRID_SIZE, GRID_SIZE],
    );

    // Reshape: [D, 2, 2] -> [D, NUM_PATCHES]
    let reshaped = b.add_reshape(conv_out, &[HIDDEN_DIM, NUM_PATCHES]);

    // Transpose: [D, NUM_PATCHES] -> [NUM_PATCHES, D]
    let out = b.add_transpose(reshaped, &[1, 0], &[NUM_PATCHES, HIDDEN_DIM]);

    b.build(out).expect("valid 3D patch embedding kernel")
}

fn patch3d_bindings() -> Vec<TensorParamBinding> {
    let temporal_channels = TEMPORAL_FRAMES * IN_CHANNELS;
    vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(w(&[
            HIDDEN_DIM,
            temporal_channels,
            PATCH_SIZE,
            PATCH_SIZE,
        ])),
        TensorParamBinding::ConstantTensor(zeros(&[HIDDEN_DIM])),
    ]
}

#[test]
fn test_qwen3_vl_deep_3d_patch_embed_ibp() {
    let temporal_channels = TEMPORAL_FRAMES * IN_CHANNELS;
    let def = build_3d_patch_embedding_kernel();
    let bindings = patch3d_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");
    let input = image_bounds_01(&[temporal_channels, IMG_SIZE, IMG_SIZE]);

    let output = graph.propagate_ibp(&input).expect("IBP");
    assert_bounds_valid(&output);

    let (lo, _) = output.lower_upper();
    assert_eq!(
        lo.shape(),
        &[NUM_PATCHES, HIDDEN_DIM],
        "3D patch embed output shape mismatch"
    );

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("3D patch embedding IBP (video [0,1]): [{lo_min}, {hi_max}]");
    assert!(lo_min.is_finite() && hi_max.is_finite());
    assert!(
        lo_min < hi_max,
        "bounds must be non-degenerate: [{lo_min}, {hi_max}]"
    );
}

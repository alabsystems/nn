// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Extended compose tests for Qwen3-VL vision encoder pipeline bounds (#4231).
//!
//! Supplements `compose_dpdf_qwen3_vl_vision_encoder.rs` (25 tests) with:
//! - CROWN variants for stages that only had IBP coverage
//! - Verify-and-record tests for key pipeline stages
//! - Window attention (Qwen3-VL specific local attention)
//! - RMSNorm isolation tests
//! - Temporal patch embedding bounds (3D patch proxy)
//!
//! ## Tests (15 tests)
//!
//! 1.  **RoPE Q/K projection** (CROWN) -- CROWN linearization through RoPE
//! 2.  **Vision projection** (CROWN) -- CROWN through Linear encoder->LM mapping
//! 3.  **2-block encoder stack** (CROWN) -- deep stack CROWN linearization
//! 4.  **RMSNorm isolation** (CROWN) -- standalone RMSNorm CROWN bounds
//! 5.  **Window attention** (IBP) -- local window self-attention bounds
//! 6.  **Window attention** (CROWN) -- CROWN through windowed attention
//! 7.  **Temporal patch embedding** (IBP) -- 3D patch temporal dim proxy
//! 8.  **Image norm + patch embed** (CROWN) -- normalization + conv CROWN
//! 9.  **Patch embed Conv2d verify-and-record** -- records IBP/CROWN result
//! 10. **Encoder block verify-and-record** -- records full block result
//! 11. **GQA attention verify-and-record** -- records GQA result
//! 12. **SwiGLU FFN verify-and-record** -- records SwiGLU result
//! 13. **Full pipeline verify-and-record** -- records end-to-end result
//! 14. **Global avg pool + projection** (CROWN) -- pool then project CROWN
//! 15. **Position embed + encoder block** (CROWN) -- position inject + block
//!
//! Part of #4231: Qwen3-VL vision encoder pipeline compose tests.

use super::common::{
    assert_bounds_valid, assert_crown_tighter_when_not_fallback, bounds_min_max, uniform_bounds,
    verify_and_assert,
};
use nn_dsl::tensor_block_builder::TensorBlockBuilder;
use nn_dsl::tensor_ir::{AttentionMask, ReduceOp};
use nn_verify::{tensor_kernel_to_graph, BoundedTensor, TensorParamBinding};
use ndarray::{ArrayD, IxDyn};

// ---------------------------------------------------------------------------
// Dimensions -- same as main file for consistency
// ---------------------------------------------------------------------------

const IMG_SIZE: usize = 8;
const PATCH_SIZE: usize = 4;
const IN_CHANNELS: usize = 3;
const GRID_SIZE: usize = IMG_SIZE / PATCH_SIZE; // 2
const SEQ_LEN: usize = GRID_SIZE * GRID_SIZE; // 4
const HIDDEN_DIM: usize = 16;
const FFN_DIM: usize = 32;
const NUM_HEADS: usize = 4;
const HEAD_DIM: usize = HIDDEN_DIM / NUM_HEADS; // 4
const NUM_KV_HEADS: usize = 2;
const KV_DIM: usize = NUM_KV_HEADS * HEAD_DIM; // 8
const LM_DIM: usize = 32;
const WEIGHT_MAG: f32 = 0.02;
const WINDOW_SIZE: usize = 2; // window attention uses 2-token windows

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn w(shape: &[usize]) -> ArrayD<f32> {
    ArrayD::from_elem(IxDyn(shape), WEIGHT_MAG)
}

fn ones_arr(shape: &[usize]) -> ArrayD<f32> {
    ArrayD::from_elem(IxDyn(shape), 1.0f32)
}

fn zeros_arr(shape: &[usize]) -> ArrayD<f32> {
    ArrayD::from_elem(IxDyn(shape), 0.0f32)
}

fn weight(shape: &[usize]) -> TensorParamBinding {
    TensorParamBinding::ConstantTensor(w(shape))
}

fn ones_bind(shape: &[usize]) -> TensorParamBinding {
    TensorParamBinding::ConstantTensor(ones_arr(shape))
}

fn zeros_bind(shape: &[usize]) -> TensorParamBinding {
    TensorParamBinding::ConstantTensor(zeros_arr(shape))
}

fn eps_bind() -> TensorParamBinding {
    TensorParamBinding::ConstantScalar(1e-5)
}

/// Sequence-domain input bounds: embeddings in [-range, +range].
fn seq_bounds(seq_len: usize, dim: usize, range: f32) -> BoundedTensor {
    uniform_bounds(&[seq_len, dim], range)
}

/// Build M-RoPE cos/sin tensors for a given seq_len and dim.
fn build_mrope_cos_sin(seq_len: usize, dim: usize) -> (ArrayD<f32>, ArrayD<f32>) {
    let n = seq_len * dim;
    let section_size = dim / 3;
    let mut cos_data = Vec::with_capacity(n);
    let mut sin_data = Vec::with_capacity(n);
    for t in 0..seq_len {
        for d in 0..dim {
            let base = if d < section_size {
                10000.0_f64
            } else {
                5000.0_f64
            };
            let freq = (t as f64) / base.powf(2.0 * (d % section_size.max(1)) as f64 / dim as f64);
            cos_data.push(freq.cos() as f32);
            sin_data.push(freq.sin() as f32);
        }
    }
    let cos = ArrayD::from_shape_vec(IxDyn(&[seq_len, dim]), cos_data).expect("valid cos");
    let sin = ArrayD::from_shape_vec(IxDyn(&[seq_len, dim]), sin_data).expect("valid sin");
    (cos, sin)
}

/// Add a single Qwen3-VL vision encoder block to a builder.
fn add_vision_encoder_block(
    b: &mut TensorBlockBuilder,
    input: nn_dsl::TensorNodeId,
    block_idx: usize,
    bindings: &mut Vec<TensorParamBinding>,
) -> nn_dsl::TensorNodeId {
    let shape = [SEQ_LEN, HIDDEN_DIM];
    let ffn_shape = [SEQ_LEN, FFN_DIM];
    let scale = 1.0 / (HEAD_DIM as f32).sqrt();

    let norm1_eps = b.add_input(&format!("b{block_idx}_norm1_eps"), &[1]);
    let norm1_w = b.add_input(&format!("b{block_idx}_norm1_w"), &[HIDDEN_DIM]);
    let normed1 = b.add_rms_norm(input, norm1_eps, 1, norm1_w, &shape);

    let q_w = b.add_input(&format!("b{block_idx}_q_w"), &[HIDDEN_DIM, HIDDEN_DIM]);
    let k_w = b.add_input(&format!("b{block_idx}_k_w"), &[HIDDEN_DIM, HIDDEN_DIM]);
    let v_w = b.add_input(&format!("b{block_idx}_v_w"), &[HIDDEN_DIM, HIDDEN_DIM]);
    let out_w = b.add_input(&format!("b{block_idx}_out_w"), &[HIDDEN_DIM, HIDDEN_DIM]);

    let q = b.add_linear(normed1, q_w, None, &shape);
    let k = b.add_linear(normed1, k_w, None, &shape);
    let v = b.add_linear(normed1, v_w, None, &shape);
    let attn = b.add_attention(q, k, v, AttentionMask::Standard, Some(scale), &shape);
    let attn_out = b.add_linear(attn, out_w, None, &shape);

    let res1 = b.add_binary_add(input, attn_out, &shape);

    let norm2_eps = b.add_input(&format!("b{block_idx}_norm2_eps"), &[1]);
    let norm2_w = b.add_input(&format!("b{block_idx}_norm2_w"), &[HIDDEN_DIM]);
    let normed2 = b.add_rms_norm(res1, norm2_eps, 1, norm2_w, &shape);

    let gate_w = b.add_input(&format!("b{block_idx}_gate_w"), &[FFN_DIM, HIDDEN_DIM]);
    let up_w = b.add_input(&format!("b{block_idx}_up_w"), &[FFN_DIM, HIDDEN_DIM]);
    let down_w = b.add_input(&format!("b{block_idx}_down_w"), &[HIDDEN_DIM, FFN_DIM]);

    let gate = b.add_linear(normed2, gate_w, None, &ffn_shape);
    let gate_sig = b.add_sigmoid(gate, &ffn_shape);
    let gate_act = b.add_binary_mul(gate, gate_sig, &ffn_shape);
    let up = b.add_linear(normed2, up_w, None, &ffn_shape);
    let hidden = b.add_binary_mul(gate_act, up, &ffn_shape);
    let ffn_out = b.add_linear(hidden, down_w, None, &shape);

    let out = b.add_binary_add(res1, ffn_out, &shape);

    let qkvo = w(&[HIDDEN_DIM, HIDDEN_DIM]);
    bindings.push(eps_bind());
    bindings.push(TensorParamBinding::ConstantTensor(ones_arr(&[HIDDEN_DIM])));
    bindings.push(TensorParamBinding::ConstantTensor(qkvo.clone()));
    bindings.push(TensorParamBinding::ConstantTensor(qkvo.clone()));
    bindings.push(TensorParamBinding::ConstantTensor(qkvo.clone()));
    bindings.push(TensorParamBinding::ConstantTensor(qkvo));
    bindings.push(eps_bind());
    bindings.push(TensorParamBinding::ConstantTensor(ones_arr(&[HIDDEN_DIM])));
    bindings.push(weight(&[FFN_DIM, HIDDEN_DIM]));
    bindings.push(weight(&[FFN_DIM, HIDDEN_DIM]));
    bindings.push(weight(&[HIDDEN_DIM, FFN_DIM]));

    out
}

// ===========================================================================
// 1. RoPE Q/K projection (CROWN)
// ===========================================================================

#[test]
fn test_qwen3_vl_ve_ext_rope_qk_crown() {
    let mut b = TensorBlockBuilder::new("qwen3_vl_ve_ext_rope_qk_crown");
    let input = b.add_input("x", &[SEQ_LEN, HIDDEN_DIM]);
    let q_w = b.add_input("q_w", &[HIDDEN_DIM, HIDDEN_DIM]);
    let k_w = b.add_input("k_w", &[HIDDEN_DIM, HIDDEN_DIM]);
    let cos_pe = b.add_input("cos", &[SEQ_LEN, HIDDEN_DIM]);
    let sin_pe = b.add_input("sin", &[SEQ_LEN, HIDDEN_DIM]);

    let shape = [SEQ_LEN, HIDDEN_DIM];

    let q = b.add_linear(input, q_w, None, &shape);
    let k = b.add_linear(input, k_w, None, &shape);

    let q_cos = b.add_binary_mul(q, cos_pe, &shape);
    let q_sin = b.add_binary_mul(q, sin_pe, &shape);
    let q_rope = b.add_binary_add(q_cos, q_sin, &shape);

    let k_cos = b.add_binary_mul(k, cos_pe, &shape);
    let k_sin = b.add_binary_mul(k, sin_pe, &shape);
    let k_rope = b.add_binary_add(k_cos, k_sin, &shape);

    let out = b.add_binary_add(q_rope, k_rope, &shape);
    let def = b.build(out).expect("valid RoPE Q/K kernel");

    let (cos_arr, sin_arr) = build_mrope_cos_sin(SEQ_LEN, HIDDEN_DIM);
    let bindings = vec![
        TensorParamBinding::Variable,
        weight(&[HIDDEN_DIM, HIDDEN_DIM]),
        weight(&[HIDDEN_DIM, HIDDEN_DIM]),
        TensorParamBinding::ConstantTensor(cos_arr),
        TensorParamBinding::ConstantTensor(sin_arr),
    ];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");
    let input = seq_bounds(SEQ_LEN, HIDDEN_DIM, 0.5);

    let (method, crown_out, _) = assert_crown_tighter_when_not_fallback(&graph, &input);
    let (clo, chi) = bounds_min_max(&crown_out);
    eprintln!("RoPE Q/K projection CROWN ({method:?}): [{clo:.6}, {chi:.6}]");
}

// ===========================================================================
// 2. Vision projection (CROWN)
// ===========================================================================

#[test]
fn test_qwen3_vl_ve_ext_vision_projection_crown() {
    let mut b = TensorBlockBuilder::new("qwen3_vl_ve_ext_vision_proj_crown");
    let input = b.add_input("encoder_out", &[SEQ_LEN, HIDDEN_DIM]);
    let proj_w = b.add_input("proj_w", &[LM_DIM, HIDDEN_DIM]);
    let proj_b = b.add_input("proj_b", &[LM_DIM]);

    let out = b.add_linear(input, proj_w, Some(proj_b), &[SEQ_LEN, LM_DIM]);
    let def = b.build(out).expect("valid vision projection kernel");

    let bindings = vec![
        TensorParamBinding::Variable,
        weight(&[LM_DIM, HIDDEN_DIM]),
        zeros_bind(&[LM_DIM]),
    ];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");
    let input = seq_bounds(SEQ_LEN, HIDDEN_DIM, 0.5);

    let (method, crown_out, _) = assert_crown_tighter_when_not_fallback(&graph, &input);
    let (clo, chi) = bounds_min_max(&crown_out);
    eprintln!("Vision projection CROWN ({method:?}): [{clo:.6}, {chi:.6}]");
    assert_eq!(crown_out.lower_upper().0.shape(), &[SEQ_LEN, LM_DIM]);
}

// ===========================================================================
// 3. 2-block encoder stack (CROWN)
// ===========================================================================

#[test]
fn test_qwen3_vl_ve_ext_2block_stack_crown() {
    let mut b = TensorBlockBuilder::new("qwen3_vl_ve_ext_2block_crown");
    let input = b.add_input("x", &[SEQ_LEN, HIDDEN_DIM]);
    let mut bindings = vec![TensorParamBinding::Variable];
    let mid = add_vision_encoder_block(&mut b, input, 0, &mut bindings);
    let out = add_vision_encoder_block(&mut b, mid, 1, &mut bindings);
    let def = b.build(out).expect("valid 2-block stack kernel");

    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");
    let input = seq_bounds(SEQ_LEN, HIDDEN_DIM, 0.3);

    let (method, crown_out, _) = assert_crown_tighter_when_not_fallback(&graph, &input);
    let (clo, chi) = bounds_min_max(&crown_out);
    eprintln!("2-block encoder stack CROWN ({method:?}): [{clo:.6}, {chi:.6}]");
}

// ===========================================================================
// 4. RMSNorm isolation (CROWN)
// ===========================================================================

#[test]
fn test_qwen3_vl_ve_ext_rmsnorm_isolation_crown() {
    let mut b = TensorBlockBuilder::new("qwen3_vl_ve_ext_rmsnorm_crown");
    let input = b.add_input("x", &[SEQ_LEN, HIDDEN_DIM]);
    let eps = b.add_input("eps", &[1]);
    let norm_w = b.add_input("weight", &[HIDDEN_DIM]);

    let out = b.add_rms_norm(input, eps, 1, norm_w, &[SEQ_LEN, HIDDEN_DIM]);
    let def = b.build(out).expect("valid RMSNorm kernel");

    let bindings = vec![
        TensorParamBinding::Variable,
        eps_bind(),
        ones_bind(&[HIDDEN_DIM]),
    ];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");
    let input = seq_bounds(SEQ_LEN, HIDDEN_DIM, 0.5);

    let (method, crown_out, _) = assert_crown_tighter_when_not_fallback(&graph, &input);
    let (clo, chi) = bounds_min_max(&crown_out);
    eprintln!("RMSNorm isolation CROWN ({method:?}): [{clo:.6}, {chi:.6}]");
    assert!(clo.is_finite() && chi.is_finite());
}

// ===========================================================================
// 5. Window attention (IBP)
// ===========================================================================

#[test]
fn test_qwen3_vl_ve_ext_window_attention_ibp() {
    // Qwen3-VL uses window attention in early ViT layers. Modeled as
    // attention over WINDOW_SIZE tokens (local scope) projected back out.
    let win_shape = [WINDOW_SIZE, HIDDEN_DIM];
    let scale = 1.0 / (HEAD_DIM as f32).sqrt();

    let mut b = TensorBlockBuilder::new("qwen3_vl_ve_ext_window_attn");
    let input = b.add_input("window_tokens", &win_shape);
    let q_w = b.add_input("q_w", &[HIDDEN_DIM, HIDDEN_DIM]);
    let k_w = b.add_input("k_w", &[HIDDEN_DIM, HIDDEN_DIM]);
    let v_w = b.add_input("v_w", &[HIDDEN_DIM, HIDDEN_DIM]);
    let out_w = b.add_input("out_w", &[HIDDEN_DIM, HIDDEN_DIM]);

    let q = b.add_linear(input, q_w, None, &win_shape);
    let k = b.add_linear(input, k_w, None, &win_shape);
    let v = b.add_linear(input, v_w, None, &win_shape);
    let attn = b.add_attention(q, k, v, AttentionMask::Standard, Some(scale), &win_shape);
    let out = b.add_linear(attn, out_w, None, &win_shape);
    let def = b.build(out).expect("valid window attention kernel");

    let qkvo = w(&[HIDDEN_DIM, HIDDEN_DIM]);
    let bindings = vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(qkvo.clone()),
        TensorParamBinding::ConstantTensor(qkvo.clone()),
        TensorParamBinding::ConstantTensor(qkvo.clone()),
        TensorParamBinding::ConstantTensor(qkvo),
    ];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");
    let input = seq_bounds(WINDOW_SIZE, HIDDEN_DIM, 1.0);

    let output = graph.propagate_ibp(&input).expect("IBP");
    assert_bounds_valid(&output);
    assert_eq!(output.lower_upper().0.shape(), &[WINDOW_SIZE, HIDDEN_DIM]);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Window attention IBP: [{lo_min:.6}, {hi_max:.6}]");
    assert!(lo_min.is_finite() && hi_max.is_finite());
}

// ===========================================================================
// 6. Window attention (CROWN)
// ===========================================================================

#[test]
fn test_qwen3_vl_ve_ext_window_attention_crown() {
    let win_shape = [WINDOW_SIZE, HIDDEN_DIM];
    let scale = 1.0 / (HEAD_DIM as f32).sqrt();

    let mut b = TensorBlockBuilder::new("qwen3_vl_ve_ext_window_attn_crown");
    let input = b.add_input("window_tokens", &win_shape);
    let q_w = b.add_input("q_w", &[HIDDEN_DIM, HIDDEN_DIM]);
    let k_w = b.add_input("k_w", &[HIDDEN_DIM, HIDDEN_DIM]);
    let v_w = b.add_input("v_w", &[HIDDEN_DIM, HIDDEN_DIM]);
    let out_w = b.add_input("out_w", &[HIDDEN_DIM, HIDDEN_DIM]);

    let q = b.add_linear(input, q_w, None, &win_shape);
    let k = b.add_linear(input, k_w, None, &win_shape);
    let v = b.add_linear(input, v_w, None, &win_shape);
    let attn = b.add_attention(q, k, v, AttentionMask::Standard, Some(scale), &win_shape);
    let out = b.add_linear(attn, out_w, None, &win_shape);
    let def = b.build(out).expect("valid window attention kernel");

    let qkvo = w(&[HIDDEN_DIM, HIDDEN_DIM]);
    let bindings = vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(qkvo.clone()),
        TensorParamBinding::ConstantTensor(qkvo.clone()),
        TensorParamBinding::ConstantTensor(qkvo.clone()),
        TensorParamBinding::ConstantTensor(qkvo),
    ];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");
    let input = seq_bounds(WINDOW_SIZE, HIDDEN_DIM, 0.5);

    let (method, crown_out, _) = assert_crown_tighter_when_not_fallback(&graph, &input);
    let (clo, chi) = bounds_min_max(&crown_out);
    eprintln!("Window attention CROWN ({method:?}): [{clo:.6}, {chi:.6}]");
}

// ===========================================================================
// 7. Temporal patch embedding (IBP) -- 3D patch proxy
// ===========================================================================

#[test]
fn test_qwen3_vl_ve_ext_temporal_patch_embed_ibp() {
    // Qwen3-VL uses 3D patch embedding (temporal + spatial). We model the
    // temporal dimension as a separate Conv2d over a temporal-spatial slice,
    // then project and sum with the spatial patch embedding output.
    let out_h = IMG_SIZE / PATCH_SIZE;
    let out_w = IMG_SIZE / PATCH_SIZE;
    let shape = [SEQ_LEN, HIDDEN_DIM];

    let mut b = TensorBlockBuilder::new("qwen3_vl_ve_ext_temporal_patch");
    // Spatial frame input
    let frame = b.add_input("frame", &[IN_CHANNELS, IMG_SIZE, IMG_SIZE]);
    let conv_w = b.add_input(
        "spatial_proj_w",
        &[HIDDEN_DIM, IN_CHANNELS, PATCH_SIZE, PATCH_SIZE],
    );
    let conv_b = b.add_input("spatial_proj_b", &[HIDDEN_DIM]);

    let spatial_conv = b.add_conv2d(
        frame,
        conv_w,
        Some(conv_b),
        PATCH_SIZE,
        PATCH_SIZE,
        0,
        0,
        &[HIDDEN_DIM, out_h, out_w],
    );
    let flat = b.add_reshape(spatial_conv, &[HIDDEN_DIM, SEQ_LEN]);
    let spatial_tokens = b.add_transpose(flat, &[1, 0], &shape);

    // Temporal embedding: a constant tensor representing the temporal patch
    // projection from an adjacent frame (small constant offset).
    let temporal_emb = b.add_input("temporal_emb", &shape);
    let combined = b.add_binary_add(spatial_tokens, temporal_emb, &shape);

    // RMSNorm after combined embedding
    let eps = b.add_input("norm_eps", &[1]);
    let norm_w = b.add_input("norm_w", &[HIDDEN_DIM]);
    let out = b.add_rms_norm(combined, eps, 1, norm_w, &shape);
    let def = b.build(out).expect("valid temporal patch kernel");

    let temporal_const = ArrayD::from_elem(IxDyn(&shape), 0.05f32);
    let bindings = vec![
        TensorParamBinding::Variable,
        weight(&[HIDDEN_DIM, IN_CHANNELS, PATCH_SIZE, PATCH_SIZE]),
        zeros_bind(&[HIDDEN_DIM]),
        TensorParamBinding::ConstantTensor(temporal_const),
        eps_bind(),
        ones_bind(&[HIDDEN_DIM]),
    ];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");
    let input = BoundedTensor::new(
        ArrayD::from_elem(IxDyn(&[IN_CHANNELS, IMG_SIZE, IMG_SIZE]), 0.0f32),
        ArrayD::from_elem(IxDyn(&[IN_CHANNELS, IMG_SIZE, IMG_SIZE]), 1.0f32),
    )
    .expect("valid image bounds");

    let output = graph.propagate_ibp(&input).expect("IBP");
    assert_bounds_valid(&output);
    assert_eq!(output.lower_upper().0.shape(), &shape);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Temporal patch embedding IBP: [{lo_min:.6}, {hi_max:.6}]");
    assert!(lo_min.is_finite() && hi_max.is_finite());
}

// ===========================================================================
// 8. Image normalization + patch embed (CROWN)
// ===========================================================================

#[test]
fn test_qwen3_vl_ve_ext_img_norm_patch_embed_crown() {
    let out_h = IMG_SIZE / PATCH_SIZE;
    let out_w = IMG_SIZE / PATCH_SIZE;
    let img_shape = [IN_CHANNELS, IMG_SIZE, IMG_SIZE];

    // ImageNet normalization constants
    let mean_vals = [0.485_f32, 0.456, 0.406];
    let inv_std_vals = [1.0 / 0.229_f32, 1.0 / 0.224, 1.0 / 0.225];
    let flat = IN_CHANNELS * IMG_SIZE * IMG_SIZE;
    let mut neg_mean_data = Vec::with_capacity(flat);
    let mut inv_std_data = Vec::with_capacity(flat);
    for c in 0..IN_CHANNELS {
        for _ in 0..(IMG_SIZE * IMG_SIZE) {
            neg_mean_data.push(-mean_vals[c]);
            inv_std_data.push(inv_std_vals[c]);
        }
    }
    let neg_mean_arr =
        ArrayD::from_shape_vec(IxDyn(&img_shape), neg_mean_data).expect("valid neg_mean");
    let inv_std_arr =
        ArrayD::from_shape_vec(IxDyn(&img_shape), inv_std_data).expect("valid inv_std");

    let mut b = TensorBlockBuilder::new("qwen3_vl_ve_ext_norm_patch_crown");
    let input = b.add_input("image", &img_shape);
    let neg_mean = b.add_input("neg_mean", &img_shape);
    let inv_std = b.add_input("inv_std", &img_shape);

    // (image + (-mean)) * inv_std
    let centered = b.add_binary_add(input, neg_mean, &img_shape);
    let normalized = b.add_binary_mul(centered, inv_std, &img_shape);

    // Patch embedding Conv2d
    let conv_w = b.add_input("proj_w", &[HIDDEN_DIM, IN_CHANNELS, PATCH_SIZE, PATCH_SIZE]);
    let conv_b = b.add_input("proj_b", &[HIDDEN_DIM]);
    let conv = b.add_conv2d(
        normalized,
        conv_w,
        Some(conv_b),
        PATCH_SIZE,
        PATCH_SIZE,
        0,
        0,
        &[HIDDEN_DIM, out_h, out_w],
    );
    let flat_node = b.add_reshape(conv, &[HIDDEN_DIM, SEQ_LEN]);
    let out = b.add_transpose(flat_node, &[1, 0], &[SEQ_LEN, HIDDEN_DIM]);
    let def = b.build(out).expect("valid img norm + patch kernel");

    let bindings = vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(neg_mean_arr),
        TensorParamBinding::ConstantTensor(inv_std_arr),
        weight(&[HIDDEN_DIM, IN_CHANNELS, PATCH_SIZE, PATCH_SIZE]),
        zeros_bind(&[HIDDEN_DIM]),
    ];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");
    // Tighter bounds for CROWN stability
    let input = BoundedTensor::new(
        ArrayD::from_elem(IxDyn(&img_shape), 0.3f32),
        ArrayD::from_elem(IxDyn(&img_shape), 0.7f32),
    )
    .expect("valid bounds");

    let (method, crown_out, _) = assert_crown_tighter_when_not_fallback(&graph, &input);
    let (clo, chi) = bounds_min_max(&crown_out);
    eprintln!("Image norm + patch embed CROWN ({method:?}): [{clo:.6}, {chi:.6}]");
    assert_eq!(crown_out.lower_upper().0.shape(), &[SEQ_LEN, HIDDEN_DIM]);
}

// ===========================================================================
// 9. Patch embed Conv2d verify-and-record
// ===========================================================================

#[test]
fn test_qwen3_vl_ve_ext_patch_embed_verify_and_record() {
    let out_h = IMG_SIZE / PATCH_SIZE;
    let out_w = IMG_SIZE / PATCH_SIZE;

    let mut b = TensorBlockBuilder::new("qwen3_vl_ve_ext_patch_embed_rec");
    let input = b.add_input("image", &[IN_CHANNELS, IMG_SIZE, IMG_SIZE]);
    let conv_w = b.add_input("proj_w", &[HIDDEN_DIM, IN_CHANNELS, PATCH_SIZE, PATCH_SIZE]);
    let conv_b = b.add_input("proj_b", &[HIDDEN_DIM]);
    let conv = b.add_conv2d(
        input,
        conv_w,
        Some(conv_b),
        PATCH_SIZE,
        PATCH_SIZE,
        0,
        0,
        &[HIDDEN_DIM, out_h, out_w],
    );
    let flat = b.add_reshape(conv, &[HIDDEN_DIM, SEQ_LEN]);
    let out = b.add_transpose(flat, &[1, 0], &[SEQ_LEN, HIDDEN_DIM]);
    let def = b.build(out).expect("valid patch embed kernel");

    let bindings = vec![
        TensorParamBinding::Variable,
        weight(&[HIDDEN_DIM, IN_CHANNELS, PATCH_SIZE, PATCH_SIZE]),
        zeros_bind(&[HIDDEN_DIM]),
    ];
    let input = BoundedTensor::new(
        ArrayD::from_elem(IxDyn(&[IN_CHANNELS, IMG_SIZE, IMG_SIZE]), 0.0f32),
        ArrayD::from_elem(IxDyn(&[IN_CHANNELS, IMG_SIZE, IMG_SIZE]), 1.0f32),
    )
    .expect("valid image bounds");

    let result = verify_and_assert(&def, &bindings, &input, "qwen3_vl_ve_ext::patch_embed");
    let (lo, _) = result.output_bounds.lower_upper();
    assert_eq!(lo.shape(), &[SEQ_LEN, HIDDEN_DIM]);
    eprintln!(
        "Patch embed verify-and-record: mode={:?}",
        result.verification.soundness_mode
    );
}

// ===========================================================================
// 10. Encoder block verify-and-record
// ===========================================================================

#[test]
fn test_qwen3_vl_ve_ext_encoder_block_verify_and_record() {
    let mut b = TensorBlockBuilder::new("qwen3_vl_ve_ext_enc_block_rec");
    let input = b.add_input("x", &[SEQ_LEN, HIDDEN_DIM]);
    let mut bindings = vec![TensorParamBinding::Variable];
    let out = add_vision_encoder_block(&mut b, input, 0, &mut bindings);
    let def = b.build(out).expect("valid encoder block kernel");

    let input = seq_bounds(SEQ_LEN, HIDDEN_DIM, 1.0);

    let result = verify_and_assert(&def, &bindings, &input, "qwen3_vl_ve_ext::encoder_block");
    let (lo, _) = result.output_bounds.lower_upper();
    assert_eq!(lo.shape(), &[SEQ_LEN, HIDDEN_DIM]);
    eprintln!(
        "Encoder block verify-and-record: mode={:?}",
        result.verification.soundness_mode
    );
}

// ===========================================================================
// 11. GQA attention verify-and-record
// ===========================================================================

#[test]
fn test_qwen3_vl_ve_ext_gqa_verify_and_record() {
    // GQA: project Q/K/V to KV_DIM so q_d == k_d == KV_DIM (QK^T contracts over
    // the head dim), then lift the attention output back to HIDDEN_DIM via out_w.
    let mut b = TensorBlockBuilder::new("qwen3_vl_ve_ext_gqa_rec");
    let input = b.add_input("x", &[SEQ_LEN, HIDDEN_DIM]);
    let q_w = b.add_input("q_w", &[KV_DIM, HIDDEN_DIM]);
    let k_w = b.add_input("k_w", &[KV_DIM, HIDDEN_DIM]);
    let v_w = b.add_input("v_w", &[KV_DIM, HIDDEN_DIM]);
    let out_w = b.add_input("out_w", &[HIDDEN_DIM, KV_DIM]);

    let q_shape = [SEQ_LEN, HIDDEN_DIM];
    let kv_shape = [SEQ_LEN, KV_DIM];

    let q = b.add_linear(input, q_w, None, &kv_shape);
    let k = b.add_linear(input, k_w, None, &kv_shape);
    let v = b.add_linear(input, v_w, None, &kv_shape);
    let scale = 1.0 / (HEAD_DIM as f32).sqrt();
    let attn = b.add_attention(q, k, v, AttentionMask::Standard, Some(scale), &kv_shape);
    let out = b.add_linear(attn, out_w, None, &q_shape);
    let def = b.build(out).expect("valid GQA kernel");

    let bindings = vec![
        TensorParamBinding::Variable,
        weight(&[KV_DIM, HIDDEN_DIM]),
        weight(&[KV_DIM, HIDDEN_DIM]),
        weight(&[KV_DIM, HIDDEN_DIM]),
        weight(&[HIDDEN_DIM, KV_DIM]),
    ];
    let input = seq_bounds(SEQ_LEN, HIDDEN_DIM, 1.0);

    let result = verify_and_assert(&def, &bindings, &input, "qwen3_vl_ve_ext::gqa_attention");
    let (lo, _) = result.output_bounds.lower_upper();
    assert_eq!(lo.shape(), &[SEQ_LEN, HIDDEN_DIM]);
    eprintln!(
        "GQA attention verify-and-record: mode={:?}",
        result.verification.soundness_mode
    );
}

// ===========================================================================
// 12. SwiGLU FFN verify-and-record
// ===========================================================================

#[test]
fn test_qwen3_vl_ve_ext_swiglu_verify_and_record() {
    let mut b = TensorBlockBuilder::new("qwen3_vl_ve_ext_swiglu_rec");
    let input = b.add_input("x", &[SEQ_LEN, HIDDEN_DIM]);
    let gate_w = b.add_input("gate_w", &[FFN_DIM, HIDDEN_DIM]);
    let up_w = b.add_input("up_w", &[FFN_DIM, HIDDEN_DIM]);
    let down_w = b.add_input("down_w", &[HIDDEN_DIM, FFN_DIM]);

    let shape = [SEQ_LEN, HIDDEN_DIM];
    let ffn_shape = [SEQ_LEN, FFN_DIM];

    let gate = b.add_linear(input, gate_w, None, &ffn_shape);
    let gate_sig = b.add_sigmoid(gate, &ffn_shape);
    let gate_act = b.add_binary_mul(gate, gate_sig, &ffn_shape);
    let up = b.add_linear(input, up_w, None, &ffn_shape);
    let hidden = b.add_binary_mul(gate_act, up, &ffn_shape);
    let out = b.add_linear(hidden, down_w, None, &shape);
    let def = b.build(out).expect("valid SwiGLU kernel");

    let bindings = vec![
        TensorParamBinding::Variable,
        weight(&[FFN_DIM, HIDDEN_DIM]),
        weight(&[FFN_DIM, HIDDEN_DIM]),
        weight(&[HIDDEN_DIM, FFN_DIM]),
    ];
    let input = seq_bounds(SEQ_LEN, HIDDEN_DIM, 1.0);

    let result = verify_and_assert(&def, &bindings, &input, "qwen3_vl_ve_ext::swiglu_ffn");
    let (lo, _) = result.output_bounds.lower_upper();
    assert_eq!(lo.shape(), &[SEQ_LEN, HIDDEN_DIM]);
    eprintln!(
        "SwiGLU FFN verify-and-record: mode={:?}",
        result.verification.soundness_mode
    );
}

// ===========================================================================
// 13. Full pipeline verify-and-record
// ===========================================================================

#[test]
fn test_qwen3_vl_ve_ext_full_pipeline_verify_and_record() {
    let out_h = IMG_SIZE / PATCH_SIZE;
    let out_w = IMG_SIZE / PATCH_SIZE;

    let mut b = TensorBlockBuilder::new("qwen3_vl_ve_ext_full_rec");
    let img = b.add_input("image", &[IN_CHANNELS, IMG_SIZE, IMG_SIZE]);
    let conv_w = b.add_input("proj_w", &[HIDDEN_DIM, IN_CHANNELS, PATCH_SIZE, PATCH_SIZE]);
    let conv_b = b.add_input("proj_b", &[HIDDEN_DIM]);

    let conv = b.add_conv2d(
        img,
        conv_w,
        Some(conv_b),
        PATCH_SIZE,
        PATCH_SIZE,
        0,
        0,
        &[HIDDEN_DIM, out_h, out_w],
    );
    let flat = b.add_reshape(conv, &[HIDDEN_DIM, SEQ_LEN]);
    let tokens = b.add_transpose(flat, &[1, 0], &[SEQ_LEN, HIDDEN_DIM]);

    let pe_eps = b.add_input("pe_norm_eps", &[1]);
    let pe_norm_w = b.add_input("pe_norm_w", &[HIDDEN_DIM]);
    let normed = b.add_rms_norm(tokens, pe_eps, 1, pe_norm_w, &[SEQ_LEN, HIDDEN_DIM]);

    let mut bindings = vec![
        TensorParamBinding::Variable,
        weight(&[HIDDEN_DIM, IN_CHANNELS, PATCH_SIZE, PATCH_SIZE]),
        zeros_bind(&[HIDDEN_DIM]),
        eps_bind(),
        ones_bind(&[HIDDEN_DIM]),
    ];
    let enc_out = add_vision_encoder_block(&mut b, normed, 0, &mut bindings);

    let vp_w = b.add_input("vp_w", &[LM_DIM, HIDDEN_DIM]);
    let vp_b = b.add_input("vp_b", &[LM_DIM]);
    let out = b.add_linear(enc_out, vp_w, Some(vp_b), &[SEQ_LEN, LM_DIM]);
    bindings.push(weight(&[LM_DIM, HIDDEN_DIM]));
    bindings.push(zeros_bind(&[LM_DIM]));

    let def = b.build(out).expect("valid full pipeline kernel");
    let input = BoundedTensor::new(
        ArrayD::from_elem(IxDyn(&[IN_CHANNELS, IMG_SIZE, IMG_SIZE]), 0.0f32),
        ArrayD::from_elem(IxDyn(&[IN_CHANNELS, IMG_SIZE, IMG_SIZE]), 1.0f32),
    )
    .expect("valid image bounds");

    let result = verify_and_assert(&def, &bindings, &input, "qwen3_vl_ve_ext::full_pipeline");
    let (lo, _) = result.output_bounds.lower_upper();
    assert_eq!(lo.shape(), &[SEQ_LEN, LM_DIM]);
    eprintln!(
        "Full pipeline verify-and-record: mode={:?}",
        result.verification.soundness_mode
    );
}

// ===========================================================================
// 14. Global avg pool + projection (CROWN)
// ===========================================================================

#[test]
fn test_qwen3_vl_ve_ext_gap_projection_crown() {
    // Global average pooling over sequence dim, then linear projection.
    // Tests CROWN through reduce-mean + linear composition.
    let mut b = TensorBlockBuilder::new("qwen3_vl_ve_ext_gap_proj_crown");
    let input = b.add_input("encoder_out", &[SEQ_LEN, HIDDEN_DIM]);

    let pooled = b.add_reduce(input, ReduceOp::Mean, 0, false, &[HIDDEN_DIM]);

    // Project from HIDDEN_DIM to LM_DIM
    let proj_w = b.add_input("proj_w", &[LM_DIM, HIDDEN_DIM]);
    let proj_b = b.add_input("proj_b", &[LM_DIM]);
    let out = b.add_linear(pooled, proj_w, Some(proj_b), &[LM_DIM]);
    let def = b.build(out).expect("valid GAP + projection kernel");

    let bindings = vec![
        TensorParamBinding::Variable,
        weight(&[LM_DIM, HIDDEN_DIM]),
        zeros_bind(&[LM_DIM]),
    ];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");
    let input = seq_bounds(SEQ_LEN, HIDDEN_DIM, 0.5);

    let (method, crown_out, _) = assert_crown_tighter_when_not_fallback(&graph, &input);
    let (clo, chi) = bounds_min_max(&crown_out);
    eprintln!("GAP + projection CROWN ({method:?}): [{clo:.6}, {chi:.6}]");
    assert_eq!(crown_out.lower_upper().0.shape(), &[LM_DIM]);
}

// ===========================================================================
// 15. Position embed + encoder block (CROWN)
// ===========================================================================

#[test]
fn test_qwen3_vl_ve_ext_pos_embed_encoder_crown() {
    // Position embedding addition followed by a full encoder block.
    // Tests CROWN through additive position injection + block composition.
    let shape = [SEQ_LEN, HIDDEN_DIM];

    let mut b = TensorBlockBuilder::new("qwen3_vl_ve_ext_pos_enc_block_crown");
    let input = b.add_input("tokens", &shape);
    let pos_emb = b.add_input("pos_embed", &shape);

    // Add position embedding (constant)
    let positioned = b.add_binary_add(input, pos_emb, &shape);

    let mut bindings = vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&shape), 0.01f32)),
    ];
    let out = add_vision_encoder_block(&mut b, positioned, 0, &mut bindings);
    let def = b.build(out).expect("valid pos + block kernel");

    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");
    let input = seq_bounds(SEQ_LEN, HIDDEN_DIM, 0.3);

    let (method, crown_out, _) = assert_crown_tighter_when_not_fallback(&graph, &input);
    let (clo, chi) = bounds_min_max(&crown_out);
    eprintln!("Position embed + encoder block CROWN ({method:?}): [{clo:.6}, {chi:.6}]");
}

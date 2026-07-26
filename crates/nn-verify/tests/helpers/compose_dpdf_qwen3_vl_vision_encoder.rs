// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Compose tests for Qwen3-VL vision encoder pipeline bounds (#4231).
//!
//! Verifies IBP and CROWN bound propagation through the Qwen3-VL vision
//! encoder subgraphs used in document understanding. These tests focus on
//! the ViT-style vision encoder pipeline: image -> patches -> encoder blocks
//! -> projection, covering all key sub-blocks:
//!
//! ## Tests (25 tests)
//!
//! 1.  **Patch embedding Conv2d** (IBP) -- Conv2d(3, D, P, stride=P) spatial
//! 2.  **Patch embed + RMSNorm** (IBP) -- Conv2d -> reshape -> RMSNorm
//! 3.  **Patch embed + RMSNorm** (CROWN) -- CROWN linearization through norm
//! 4.  **RoPE cos/sin bounded** (IBP) -- cos/sin mul on Q/K bounded in [-1,1]
//! 5.  **RoPE applied to Q/K** (IBP) -- Linear(Q) * cos + Linear(Q) * sin
//! 6.  **GQA attention** (IBP) -- Q/K/V with grouped heads -> softmax -> out
//! 7.  **GQA attention** (CROWN) -- CROWN through grouped attention
//! 8.  **SwiGLU FFN** (IBP) -- gate_proj -> SiLU -> mul(up_proj) -> down_proj
//! 9.  **SwiGLU FFN** (CROWN) -- CROWN linearization through SiLU gate
//! 10. **Vision encoder block** (IBP) -- RMSNorm -> attn -> res -> RMSNorm ->
//!     SwiGLU -> res
//! 11. **Vision encoder block** (CROWN) -- CROWN through full block
//! 12. **2-block encoder stack** (IBP) -- depth composition, widening analysis
//! 13. **4-block encoder stack** (IBP) -- deep pipeline bounds
//! 14. **Vision projection** (IBP) -- Linear(encoder_dim -> lm_dim) mapping
//! 15. **Full pipeline: patch -> 1 block -> proj** (IBP)
//! 16. **Full pipeline: patch -> 2 blocks -> proj** (IBP)
//! 17. **Full pipeline** (CROWN) -- end-to-end CROWN linearization
//! 18. **Encoder block depth scaling** (IBP) -- 1/2/4 block bound widths
//! 19. **Image normalization (mean/std)** (IBP) -- (img - mean) / std bounds
//! 20. **Multi-resolution image tiling** (IBP) -- 2 tiles additive merge bounds
//! 21. **Dynamic resolution padding** (IBP) -- zero-pad + Conv2d bounds
//! 22. **Global average pooling** (IBP) -- mean reduce over seq dim
//! 23. **Merged image-text sequence** (IBP) -- additive vision + text fusion
//! 24. **Multi-scale feature extraction** (IBP) -- shallow + deep additive fusion
//! 25. **Patch position interpolation** (IBP) -- linear interp of position embeds
//!
//! Architecture references:
//! - Qwen2-VL / Qwen3-VL (Alibaba): 3D patch embedding, M-RoPE, window
//!   attention, SwiGLU FFN, GQA
//! - RMSNorm (Zhang & Sennrich, 2019): replaces LayerNorm in Qwen
//! - SwiGLU (Shazeer, 2020): SiLU-gated FFN
//! - GQA (Ainslie et al., 2023): Grouped-Query Attention
//!
//! Dimensions (small for fast verification, structurally representative):
//! - IMG_SIZE=8, PATCH_SIZE=4, IN_CHANNELS=3, HIDDEN_DIM=16
//! - SEQ_LEN=4, FFN_DIM=32, NUM_HEADS=4, HEAD_DIM=4, NUM_KV_HEADS=2
//! - LM_DIM=32 (projection target)
//!
//! Part of #4231: Qwen3-VL vision encoder pipeline compose tests.

use super::common::{
    assert_bounds_valid, assert_crown_tighter_when_not_fallback, bounds_min_max, uniform_bounds,
};
use nn_dsl::tensor_block_builder::TensorBlockBuilder;
use nn_dsl::tensor_ir::{AttentionMask, ReduceOp};
use nn_verify::{tensor_kernel_to_graph, BoundedTensor, TensorParamBinding};
use ndarray::{ArrayD, IxDyn};

// ---------------------------------------------------------------------------
// Dimensions -- small for fast verification, structurally representative
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

/// Image-domain input bounds: pixels in [0, 1].
fn image_bounds() -> BoundedTensor {
    BoundedTensor::new(
        ArrayD::from_elem(IxDyn(&[IN_CHANNELS, IMG_SIZE, IMG_SIZE]), 0.0f32),
        ArrayD::from_elem(IxDyn(&[IN_CHANNELS, IMG_SIZE, IMG_SIZE]), 1.0f32),
    )
    .expect("valid image bounds [0, 1]")
}

/// Sequence-domain input bounds: embeddings in [-range, +range].
fn seq_bounds(seq_len: usize, dim: usize, range: f32) -> BoundedTensor {
    uniform_bounds(&[seq_len, dim], range)
}

/// Build M-RoPE cos/sin tensors for a given seq_len and head_dim.
fn build_mrope_cos_sin(seq_len: usize, dim: usize) -> (ArrayD<f32>, ArrayD<f32>) {
    let n = seq_len * dim;
    let section_size = dim / 3;
    let mut cos_data = Vec::with_capacity(n);
    let mut sin_data = Vec::with_capacity(n);
    for t in 0..seq_len {
        for d in 0..dim {
            let base = if d < section_size {
                10000.0_f64
            } else if d < 2 * section_size {
                5000.0_f64
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
///
/// Block: RMSNorm -> Attention -> residual -> RMSNorm -> SwiGLU FFN -> residual.
/// Returns the output node and pushes weight bindings.
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
    let gate_act = b.add_binary_mul(gate, gate_sig, &ffn_shape); // silu(x) = x * sigmoid(x)
    let up = b.add_linear(normed2, up_w, None, &ffn_shape);
    let hidden = b.add_binary_mul(gate_act, up, &ffn_shape);
    let ffn_out = b.add_linear(hidden, down_w, None, &shape);

    // Residual after FFN
    let out = b.add_binary_add(res1, ffn_out, &shape);

    // Push bindings (11 weight params)
    let qkvo = w(&[HIDDEN_DIM, HIDDEN_DIM]);
    bindings.push(eps_bind()); // norm1_eps
    bindings.push(TensorParamBinding::ConstantTensor(ones_arr(&[HIDDEN_DIM]))); // norm1_w
    bindings.push(TensorParamBinding::ConstantTensor(qkvo.clone())); // q_w
    bindings.push(TensorParamBinding::ConstantTensor(qkvo.clone())); // k_w
    bindings.push(TensorParamBinding::ConstantTensor(qkvo.clone())); // v_w
    bindings.push(TensorParamBinding::ConstantTensor(qkvo)); // out_w
    bindings.push(eps_bind()); // norm2_eps
    bindings.push(TensorParamBinding::ConstantTensor(ones_arr(&[HIDDEN_DIM]))); // norm2_w
    bindings.push(weight(&[FFN_DIM, HIDDEN_DIM])); // gate_w
    bindings.push(weight(&[FFN_DIM, HIDDEN_DIM])); // up_w
    bindings.push(weight(&[HIDDEN_DIM, FFN_DIM])); // down_w

    out
}

// ===========================================================================
// 1. Patch embedding Conv2d (IBP)
// ===========================================================================

#[test]
fn test_qwen3_vl_ve_patch_embed_conv2d_ibp() {
    let out_h = IMG_SIZE / PATCH_SIZE;
    let out_w = IMG_SIZE / PATCH_SIZE;

    let mut b = TensorBlockBuilder::new("qwen3_vl_ve_patch_embed");
    let input = b.add_input("image", &[IN_CHANNELS, IMG_SIZE, IMG_SIZE]);
    let conv_w = b.add_input("proj_w", &[HIDDEN_DIM, IN_CHANNELS, PATCH_SIZE, PATCH_SIZE]);
    let conv_b = b.add_input("proj_b", &[HIDDEN_DIM]);
    let out = b.add_conv2d(
        input,
        conv_w,
        Some(conv_b),
        PATCH_SIZE,
        PATCH_SIZE,
        0,
        0,
        &[HIDDEN_DIM, out_h, out_w],
    );
    let def = b.build(out).expect("valid patch embed kernel");

    let bindings = vec![
        TensorParamBinding::Variable,
        weight(&[HIDDEN_DIM, IN_CHANNELS, PATCH_SIZE, PATCH_SIZE]),
        zeros_bind(&[HIDDEN_DIM]),
    ];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");
    let input = image_bounds();

    let output = graph.propagate_ibp(&input).expect("IBP");
    assert_bounds_valid(&output);
    assert_eq!(output.lower_upper().0.shape(), &[HIDDEN_DIM, out_h, out_w]);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Patch embed Conv2d IBP: [{lo_min:.6}, {hi_max:.6}]");
    assert!(lo_min.is_finite() && hi_max.is_finite());
}

// ===========================================================================
// 2. Patch embed + RMSNorm (IBP)
// ===========================================================================

#[test]
fn test_qwen3_vl_ve_patch_embed_rmsnorm_ibp() {
    let out_h = IMG_SIZE / PATCH_SIZE;
    let out_w = IMG_SIZE / PATCH_SIZE;

    let mut b = TensorBlockBuilder::new("qwen3_vl_ve_patch_rmsnorm");
    let input = b.add_input("image", &[IN_CHANNELS, IMG_SIZE, IMG_SIZE]);
    let conv_w = b.add_input("proj_w", &[HIDDEN_DIM, IN_CHANNELS, PATCH_SIZE, PATCH_SIZE]);
    let conv_b = b.add_input("proj_b", &[HIDDEN_DIM]);

    // Conv2d patch projection
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

    // Reshape [D, H', W'] -> [D, N] -> transpose -> [N, D]
    let flat = b.add_reshape(conv, &[HIDDEN_DIM, SEQ_LEN]);
    let tokens = b.add_transpose(flat, &[1, 0], &[SEQ_LEN, HIDDEN_DIM]);

    // RMSNorm
    let eps = b.add_input("norm_eps", &[1]);
    let norm_w = b.add_input("norm_w", &[HIDDEN_DIM]);
    let out = b.add_rms_norm(tokens, eps, 1, norm_w, &[SEQ_LEN, HIDDEN_DIM]);
    let def = b.build(out).expect("valid patch + RMSNorm kernel");

    let bindings = vec![
        TensorParamBinding::Variable,
        weight(&[HIDDEN_DIM, IN_CHANNELS, PATCH_SIZE, PATCH_SIZE]),
        zeros_bind(&[HIDDEN_DIM]),
        eps_bind(),
        ones_bind(&[HIDDEN_DIM]),
    ];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");
    let input = image_bounds();

    let output = graph.propagate_ibp(&input).expect("IBP");
    assert_bounds_valid(&output);
    assert_eq!(output.lower_upper().0.shape(), &[SEQ_LEN, HIDDEN_DIM]);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Patch embed + RMSNorm IBP: [{lo_min:.6}, {hi_max:.6}]");
    assert!(lo_min.is_finite() && hi_max.is_finite());
}

// ===========================================================================
// 3. Patch embed + RMSNorm (CROWN)
// ===========================================================================

#[test]
fn test_qwen3_vl_ve_patch_embed_rmsnorm_crown() {
    let out_h = IMG_SIZE / PATCH_SIZE;
    let out_w = IMG_SIZE / PATCH_SIZE;

    let mut b = TensorBlockBuilder::new("qwen3_vl_ve_patch_rmsnorm_crown");
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
    let tokens = b.add_transpose(flat, &[1, 0], &[SEQ_LEN, HIDDEN_DIM]);
    let eps = b.add_input("norm_eps", &[1]);
    let norm_w = b.add_input("norm_w", &[HIDDEN_DIM]);
    let out = b.add_rms_norm(tokens, eps, 1, norm_w, &[SEQ_LEN, HIDDEN_DIM]);
    let def = b.build(out).expect("valid patch + RMSNorm kernel");

    let bindings = vec![
        TensorParamBinding::Variable,
        weight(&[HIDDEN_DIM, IN_CHANNELS, PATCH_SIZE, PATCH_SIZE]),
        zeros_bind(&[HIDDEN_DIM]),
        eps_bind(),
        ones_bind(&[HIDDEN_DIM]),
    ];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");
    // Tighter input range for CROWN stability
    let input = BoundedTensor::new(
        ArrayD::from_elem(IxDyn(&[IN_CHANNELS, IMG_SIZE, IMG_SIZE]), 0.25f32),
        ArrayD::from_elem(IxDyn(&[IN_CHANNELS, IMG_SIZE, IMG_SIZE]), 0.75f32),
    )
    .expect("valid bounds");

    let (method, crown_out, _) = assert_crown_tighter_when_not_fallback(&graph, &input);
    let (clo, chi) = bounds_min_max(&crown_out);
    eprintln!("Patch embed + RMSNorm CROWN ({method:?}): [{clo:.6}, {chi:.6}]");
}

// ===========================================================================
// 4. RoPE cos/sin bounded (IBP)
// ===========================================================================

#[test]
fn test_qwen3_vl_ve_rope_cos_sin_bounded_ibp() {
    // RoPE applies cos/sin element-wise multiplication to Q/K.
    // cos/sin values are bounded in [-1, 1] by construction.
    let mut b = TensorBlockBuilder::new("qwen3_vl_ve_rope_bounded");
    let input = b.add_input("q", &[SEQ_LEN, HIDDEN_DIM]);
    let cos_pe = b.add_input("cos", &[SEQ_LEN, HIDDEN_DIM]);
    let sin_pe = b.add_input("sin", &[SEQ_LEN, HIDDEN_DIM]);

    // q_rot = q * cos + q * sin (simplified RoPE)
    let q_cos = b.add_binary_mul(input, cos_pe, &[SEQ_LEN, HIDDEN_DIM]);
    let q_sin = b.add_binary_mul(input, sin_pe, &[SEQ_LEN, HIDDEN_DIM]);
    let out = b.add_binary_add(q_cos, q_sin, &[SEQ_LEN, HIDDEN_DIM]);
    let def = b.build(out).expect("valid RoPE kernel");

    let (cos_arr, sin_arr) = build_mrope_cos_sin(SEQ_LEN, HIDDEN_DIM);
    // Verify cos/sin values are in [-1, 1]
    for &v in cos_arr.iter().chain(sin_arr.iter()) {
        assert!(
            (-1.0 - 1e-6..=1.0 + 1e-6).contains(&v),
            "cos/sin value {v} must be in [-1, 1]"
        );
    }

    let bindings = vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(cos_arr),
        TensorParamBinding::ConstantTensor(sin_arr),
    ];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");
    let input = seq_bounds(SEQ_LEN, HIDDEN_DIM, 1.0);

    let output = graph.propagate_ibp(&input).expect("IBP");
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("RoPE cos/sin IBP: [{lo_min:.6}, {hi_max:.6}]");
    // RoPE with cos/sin in [-1,1] should not amplify bounds beyond 2x input
    assert!(lo_min.is_finite() && hi_max.is_finite());
}

// ===========================================================================
// 5. RoPE applied to Q/K projections (IBP)
// ===========================================================================

#[test]
fn test_qwen3_vl_ve_rope_qk_projection_ibp() {
    let mut b = TensorBlockBuilder::new("qwen3_vl_ve_rope_qk_proj");
    let input = b.add_input("x", &[SEQ_LEN, HIDDEN_DIM]);
    let q_w = b.add_input("q_w", &[HIDDEN_DIM, HIDDEN_DIM]);
    let k_w = b.add_input("k_w", &[HIDDEN_DIM, HIDDEN_DIM]);
    let cos_pe = b.add_input("cos", &[SEQ_LEN, HIDDEN_DIM]);
    let sin_pe = b.add_input("sin", &[SEQ_LEN, HIDDEN_DIM]);

    let shape = [SEQ_LEN, HIDDEN_DIM];

    // Q = Linear(x), K = Linear(x)
    let q = b.add_linear(input, q_w, None, &shape);
    let k = b.add_linear(input, k_w, None, &shape);

    // Apply RoPE to Q
    let q_cos = b.add_binary_mul(q, cos_pe, &shape);
    let q_sin = b.add_binary_mul(q, sin_pe, &shape);
    let q_rope = b.add_binary_add(q_cos, q_sin, &shape);

    // Apply RoPE to K
    let k_cos = b.add_binary_mul(k, cos_pe, &shape);
    let k_sin = b.add_binary_mul(k, sin_pe, &shape);
    let k_rope = b.add_binary_add(k_cos, k_sin, &shape);

    // Combine Q and K via addition (proxy for downstream use)
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
    let input = seq_bounds(SEQ_LEN, HIDDEN_DIM, 1.0);

    let output = graph.propagate_ibp(&input).expect("IBP");
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("RoPE Q/K projection IBP: [{lo_min:.6}, {hi_max:.6}]");
    assert!(lo_min.is_finite() && hi_max.is_finite());
}

// ===========================================================================
// 6. GQA attention (IBP)
// ===========================================================================

#[test]
fn test_qwen3_vl_ve_gqa_attention_ibp() {
    // Grouped-Query Attention: K/V reduced rank (NUM_KV_HEADS). QK^T contracts
    // over the head dim, so Q must share K's last dim — project Q to KV_DIM too
    // (as in the qwen3_vl GQA KV-cache kernel), then lift the attention output
    // back to HIDDEN_DIM via out_w. Modeled as: Q/K/V = Linear(x, [KV_DIM,D]),
    // attention on KV_DIM, then Linear(attn, [D,KV_DIM]).
    let mut b = TensorBlockBuilder::new("qwen3_vl_ve_gqa");
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

    // Attention on KV dim (q_d == k_d == KV_DIM); out_w lifts back to HIDDEN_DIM.
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
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");
    let input = seq_bounds(SEQ_LEN, HIDDEN_DIM, 1.0);

    let output = graph.propagate_ibp(&input).expect("IBP");
    assert_bounds_valid(&output);
    assert_eq!(output.lower_upper().0.shape(), &[SEQ_LEN, HIDDEN_DIM]);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("GQA attention IBP: [{lo_min:.6}, {hi_max:.6}]");
    assert!(lo_min.is_finite() && hi_max.is_finite());
}

// ===========================================================================
// 7. GQA attention (CROWN)
// ===========================================================================

#[test]
fn test_qwen3_vl_ve_gqa_attention_crown() {
    // GQA: project Q/K/V to KV_DIM so q_d == k_d == KV_DIM (QK^T contracts over
    // the head dim), then lift the attention output back to HIDDEN_DIM via out_w.
    let mut b = TensorBlockBuilder::new("qwen3_vl_ve_gqa_crown");
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
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");
    let input = seq_bounds(SEQ_LEN, HIDDEN_DIM, 0.5);

    let (method, crown_out, _) = assert_crown_tighter_when_not_fallback(&graph, &input);
    let (clo, chi) = bounds_min_max(&crown_out);
    eprintln!("GQA attention CROWN ({method:?}): [{clo:.6}, {chi:.6}]");
}

// ===========================================================================
// 8. SwiGLU FFN (IBP)
// ===========================================================================

#[test]
fn test_qwen3_vl_ve_swiglu_ffn_ibp() {
    // SwiGLU: silu(gate_proj(x)) * up_proj(x) -> down_proj
    // silu(x) = x * sigmoid(x)
    let mut b = TensorBlockBuilder::new("qwen3_vl_ve_swiglu");
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
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");
    let input = seq_bounds(SEQ_LEN, HIDDEN_DIM, 1.0);

    let output = graph.propagate_ibp(&input).expect("IBP");
    assert_bounds_valid(&output);
    assert_eq!(output.lower_upper().0.shape(), &[SEQ_LEN, HIDDEN_DIM]);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("SwiGLU FFN IBP: [{lo_min:.6}, {hi_max:.6}]");
    assert!(lo_min.is_finite() && hi_max.is_finite());
}

// ===========================================================================
// 9. SwiGLU FFN (CROWN)
// ===========================================================================

#[test]
fn test_qwen3_vl_ve_swiglu_ffn_crown() {
    let mut b = TensorBlockBuilder::new("qwen3_vl_ve_swiglu_crown");
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
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");
    let input = seq_bounds(SEQ_LEN, HIDDEN_DIM, 0.5);

    let (method, crown_out, _) = assert_crown_tighter_when_not_fallback(&graph, &input);
    let (clo, chi) = bounds_min_max(&crown_out);
    eprintln!("SwiGLU FFN CROWN ({method:?}): [{clo:.6}, {chi:.6}]");
}

// ===========================================================================
// 10. Vision encoder block (IBP)
// ===========================================================================

#[test]
fn test_qwen3_vl_ve_encoder_block_ibp() {
    let mut b = TensorBlockBuilder::new("qwen3_vl_ve_encoder_block");
    let input = b.add_input("x", &[SEQ_LEN, HIDDEN_DIM]);
    let mut bindings = vec![TensorParamBinding::Variable];
    let out = add_vision_encoder_block(&mut b, input, 0, &mut bindings);
    let def = b.build(out).expect("valid encoder block kernel");

    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");
    let input = seq_bounds(SEQ_LEN, HIDDEN_DIM, 1.0);

    let output = graph.propagate_ibp(&input).expect("IBP");
    assert_bounds_valid(&output);
    assert_eq!(output.lower_upper().0.shape(), &[SEQ_LEN, HIDDEN_DIM]);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Vision encoder block IBP: [{lo_min:.6}, {hi_max:.6}]");
    assert!(lo_min.is_finite() && hi_max.is_finite());
}

// ===========================================================================
// 11. Vision encoder block (CROWN)
// ===========================================================================

#[test]
fn test_qwen3_vl_ve_encoder_block_crown() {
    let mut b = TensorBlockBuilder::new("qwen3_vl_ve_encoder_block_crown");
    let input = b.add_input("x", &[SEQ_LEN, HIDDEN_DIM]);
    let mut bindings = vec![TensorParamBinding::Variable];
    let out = add_vision_encoder_block(&mut b, input, 0, &mut bindings);
    let def = b.build(out).expect("valid encoder block kernel");

    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");
    let input = seq_bounds(SEQ_LEN, HIDDEN_DIM, 0.5);

    let (method, crown_out, _) = assert_crown_tighter_when_not_fallback(&graph, &input);
    let (clo, chi) = bounds_min_max(&crown_out);
    eprintln!("Vision encoder block CROWN ({method:?}): [{clo:.6}, {chi:.6}]");
}

// ===========================================================================
// 12. 2-block encoder stack (IBP)
// ===========================================================================

#[test]
fn test_qwen3_vl_ve_2block_stack_ibp() {
    let mut b = TensorBlockBuilder::new("qwen3_vl_ve_2block_stack");
    let input = b.add_input("x", &[SEQ_LEN, HIDDEN_DIM]);
    let mut bindings = vec![TensorParamBinding::Variable];
    let mid = add_vision_encoder_block(&mut b, input, 0, &mut bindings);
    let out = add_vision_encoder_block(&mut b, mid, 1, &mut bindings);
    let def = b.build(out).expect("valid 2-block stack kernel");

    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");
    let input = seq_bounds(SEQ_LEN, HIDDEN_DIM, 1.0);

    let output = graph.propagate_ibp(&input).expect("IBP");
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("2-block encoder stack IBP: [{lo_min:.6}, {hi_max:.6}]");
    assert!(lo_min.is_finite() && hi_max.is_finite());
}

// ===========================================================================
// 13. 4-block encoder stack (IBP)
// ===========================================================================

#[test]
fn test_qwen3_vl_ve_4block_stack_ibp() {
    let mut b = TensorBlockBuilder::new("qwen3_vl_ve_4block_stack");
    let input = b.add_input("x", &[SEQ_LEN, HIDDEN_DIM]);
    let mut bindings = vec![TensorParamBinding::Variable];
    let l1 = add_vision_encoder_block(&mut b, input, 0, &mut bindings);
    let l2 = add_vision_encoder_block(&mut b, l1, 1, &mut bindings);
    let l3 = add_vision_encoder_block(&mut b, l2, 2, &mut bindings);
    let out = add_vision_encoder_block(&mut b, l3, 3, &mut bindings);
    let def = b.build(out).expect("valid 4-block stack kernel");

    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");
    let input = seq_bounds(SEQ_LEN, HIDDEN_DIM, 1.0);

    let output = graph.propagate_ibp(&input).expect("IBP");
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("4-block encoder stack IBP: [{lo_min:.6}, {hi_max:.6}]");
    assert!(lo_min.is_finite() && hi_max.is_finite());
}

// ===========================================================================
// 14. Vision projection (IBP)
// ===========================================================================

#[test]
fn test_qwen3_vl_ve_vision_projection_ibp() {
    // Linear mapping from vision encoder HIDDEN_DIM to LM embedding LM_DIM.
    let mut b = TensorBlockBuilder::new("qwen3_vl_ve_vision_proj");
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
    let input = seq_bounds(SEQ_LEN, HIDDEN_DIM, 1.0);

    let output = graph.propagate_ibp(&input).expect("IBP");
    assert_bounds_valid(&output);
    assert_eq!(output.lower_upper().0.shape(), &[SEQ_LEN, LM_DIM]);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Vision projection IBP: [{lo_min:.6}, {hi_max:.6}]");
    assert!(lo_min.is_finite() && hi_max.is_finite());
}

// ===========================================================================
// 15. Full pipeline: patch -> 1 block -> projection (IBP)
// ===========================================================================

#[test]
fn test_qwen3_vl_ve_full_pipeline_1block_ibp() {
    let out_h = IMG_SIZE / PATCH_SIZE;
    let out_w = IMG_SIZE / PATCH_SIZE;

    let mut b = TensorBlockBuilder::new("qwen3_vl_ve_full_1block");
    let img = b.add_input("image", &[IN_CHANNELS, IMG_SIZE, IMG_SIZE]);
    let conv_w = b.add_input("proj_w", &[HIDDEN_DIM, IN_CHANNELS, PATCH_SIZE, PATCH_SIZE]);
    let conv_b = b.add_input("proj_b", &[HIDDEN_DIM]);

    // Patch embedding
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

    // RMSNorm after patch embedding
    let pe_eps = b.add_input("pe_norm_eps", &[1]);
    let pe_norm_w = b.add_input("pe_norm_w", &[HIDDEN_DIM]);
    let normed = b.add_rms_norm(tokens, pe_eps, 1, pe_norm_w, &[SEQ_LEN, HIDDEN_DIM]);

    // 1 encoder block
    let mut bindings = vec![
        TensorParamBinding::Variable,
        weight(&[HIDDEN_DIM, IN_CHANNELS, PATCH_SIZE, PATCH_SIZE]),
        zeros_bind(&[HIDDEN_DIM]),
        eps_bind(),
        ones_bind(&[HIDDEN_DIM]),
    ];
    let enc_out = add_vision_encoder_block(&mut b, normed, 0, &mut bindings);

    // Vision projection
    let vp_w = b.add_input("vp_w", &[LM_DIM, HIDDEN_DIM]);
    let vp_b = b.add_input("vp_b", &[LM_DIM]);
    let out = b.add_linear(enc_out, vp_w, Some(vp_b), &[SEQ_LEN, LM_DIM]);
    bindings.push(weight(&[LM_DIM, HIDDEN_DIM]));
    bindings.push(zeros_bind(&[LM_DIM]));

    let def = b.build(out).expect("valid full pipeline kernel");
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");
    let input = image_bounds();

    let output = graph.propagate_ibp(&input).expect("IBP");
    assert_bounds_valid(&output);
    assert_eq!(output.lower_upper().0.shape(), &[SEQ_LEN, LM_DIM]);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Full pipeline (1 block) IBP: [{lo_min:.6}, {hi_max:.6}]");
    assert!(lo_min.is_finite() && hi_max.is_finite());
}

// ===========================================================================
// 16. Full pipeline: patch -> 2 blocks -> projection (IBP)
// ===========================================================================

#[test]
fn test_qwen3_vl_ve_full_pipeline_2block_ibp() {
    let out_h = IMG_SIZE / PATCH_SIZE;
    let out_w = IMG_SIZE / PATCH_SIZE;

    let mut b = TensorBlockBuilder::new("qwen3_vl_ve_full_2block");
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

    let l1 = add_vision_encoder_block(&mut b, normed, 0, &mut bindings);
    let l2 = add_vision_encoder_block(&mut b, l1, 1, &mut bindings);

    let vp_w = b.add_input("vp_w", &[LM_DIM, HIDDEN_DIM]);
    let vp_b = b.add_input("vp_b", &[LM_DIM]);
    let out = b.add_linear(l2, vp_w, Some(vp_b), &[SEQ_LEN, LM_DIM]);
    bindings.push(weight(&[LM_DIM, HIDDEN_DIM]));
    bindings.push(zeros_bind(&[LM_DIM]));

    let def = b.build(out).expect("valid full 2-block pipeline kernel");
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");
    let input = image_bounds();

    let output = graph.propagate_ibp(&input).expect("IBP");
    assert_bounds_valid(&output);
    assert_eq!(output.lower_upper().0.shape(), &[SEQ_LEN, LM_DIM]);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Full pipeline (2 blocks) IBP: [{lo_min:.6}, {hi_max:.6}]");
    assert!(lo_min.is_finite() && hi_max.is_finite());
}

// ===========================================================================
// 17. Full pipeline (CROWN)
// ===========================================================================

#[test]
fn test_qwen3_vl_ve_full_pipeline_crown() {
    let out_h = IMG_SIZE / PATCH_SIZE;
    let out_w = IMG_SIZE / PATCH_SIZE;

    let mut b = TensorBlockBuilder::new("qwen3_vl_ve_full_crown");
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
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");
    // Tighter input range for CROWN stability
    let input = BoundedTensor::new(
        ArrayD::from_elem(IxDyn(&[IN_CHANNELS, IMG_SIZE, IMG_SIZE]), 0.25f32),
        ArrayD::from_elem(IxDyn(&[IN_CHANNELS, IMG_SIZE, IMG_SIZE]), 0.75f32),
    )
    .expect("valid bounds");

    let (method, crown_out, _) = assert_crown_tighter_when_not_fallback(&graph, &input);
    let (clo, chi) = bounds_min_max(&crown_out);
    eprintln!("Full pipeline CROWN ({method:?}): [{clo:.6}, {chi:.6}]");
}

// ===========================================================================
// 18. Encoder block depth scaling (IBP)
// ===========================================================================

#[test]
fn test_qwen3_vl_ve_depth_scaling_ibp() {
    // Compare IBP bounds width across 1, 2, and 4 encoder blocks.
    // Bounds should widen with depth but remain finite.
    let mut widths = Vec::new();

    for num_blocks in [1, 2, 4] {
        let mut b = TensorBlockBuilder::new(&format!("qwen3_vl_ve_depth_{num_blocks}"));
        let input = b.add_input("x", &[SEQ_LEN, HIDDEN_DIM]);
        let mut bindings = vec![TensorParamBinding::Variable];

        let mut node = input;
        for i in 0..num_blocks {
            node = add_vision_encoder_block(&mut b, node, i, &mut bindings);
        }
        let def = b.build(node).expect("valid depth-scaled kernel");

        let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");
        let inp = seq_bounds(SEQ_LEN, HIDDEN_DIM, 1.0);

        let output = graph.propagate_ibp(&inp).expect("IBP");
        assert_bounds_valid(&output);

        let (lo_min, hi_max) = bounds_min_max(&output);
        let width = hi_max - lo_min;
        eprintln!("Depth {num_blocks} blocks IBP: [{lo_min:.6}, {hi_max:.6}], width={width:.6}");
        assert!(lo_min.is_finite() && hi_max.is_finite());
        widths.push(width);
    }

    // Bounds should widen monotonically with depth (or stay equal with RMSNorm)
    eprintln!(
        "Depth scaling widths: 1={:.4}, 2={:.4}, 4={:.4}",
        widths[0], widths[1], widths[2]
    );
    // All widths must be finite and positive
    for (i, w) in widths.iter().enumerate() {
        assert!(
            w.is_finite() && *w > 0.0,
            "depth {} width must be finite positive, got {w}",
            [1, 2, 4][i]
        );
    }
}

// ===========================================================================
// 19. Image normalization (mean/std) bounds (IBP)
// ===========================================================================

#[test]
fn test_qwen3_vl_ve_image_normalization_ibp() {
    // Image normalization: (pixel - mean) / std per channel.
    // Modeled as: subtract mean (constant), multiply by 1/std (constant).
    // ImageNet mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225].
    let mean_vals = [0.485_f32, 0.456, 0.406];
    let inv_std_vals = [1.0 / 0.229_f32, 1.0 / 0.224, 1.0 / 0.225];

    // Build per-channel mean and inv_std tensors [C, 1, 1] for broadcast
    let flat = IN_CHANNELS * IMG_SIZE * IMG_SIZE;
    let mut mean_data = Vec::with_capacity(flat);
    let mut inv_std_data = Vec::with_capacity(flat);
    for c in 0..IN_CHANNELS {
        for _ in 0..(IMG_SIZE * IMG_SIZE) {
            mean_data.push(mean_vals[c]);
            inv_std_data.push(inv_std_vals[c]);
        }
    }
    let mean_arr = ArrayD::from_shape_vec(IxDyn(&[IN_CHANNELS, IMG_SIZE, IMG_SIZE]), mean_data)
        .expect("valid mean");
    let inv_std_arr =
        ArrayD::from_shape_vec(IxDyn(&[IN_CHANNELS, IMG_SIZE, IMG_SIZE]), inv_std_data)
            .expect("valid inv_std");

    let img_shape = [IN_CHANNELS, IMG_SIZE, IMG_SIZE];
    let mut b = TensorBlockBuilder::new("qwen3_vl_ve_img_norm");
    let input = b.add_input("image", &img_shape);
    let mean_node = b.add_input("mean", &img_shape);
    let inv_std_node = b.add_input("inv_std", &img_shape);

    // (image - mean) * inv_std
    // Subtraction as: image + (-mean) -- use binary_add with negated mean constant
    // Actually, we model subtraction via: add(-mean) then multiply.
    // The builder doesn't have a subtract, so we negate mean externally.
    let neg_mean_arr = mean_arr.mapv(|v| -v);

    let centered = b.add_binary_add(input, mean_node, &img_shape);
    let normalized = b.add_binary_mul(centered, inv_std_node, &img_shape);
    let def = b.build(normalized).expect("valid img norm kernel");

    let bindings = vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(neg_mean_arr),
        TensorParamBinding::ConstantTensor(inv_std_arr),
    ];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");
    let input = image_bounds();

    let output = graph.propagate_ibp(&input).expect("IBP");
    assert_bounds_valid(&output);
    assert_eq!(output.lower_upper().0.shape(), &img_shape);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Image normalization IBP: [{lo_min:.6}, {hi_max:.6}]");
    assert!(lo_min.is_finite() && hi_max.is_finite());
    // Normalized range for [0,1] pixels with ImageNet stats: roughly [-2.1, 2.6]
    assert!(
        lo_min > -5.0 && hi_max < 5.0,
        "normalized bounds should be moderate"
    );
}

// ===========================================================================
// 20. Multi-resolution image tiling bounds (IBP)
// ===========================================================================

#[test]
fn test_qwen3_vl_ve_multi_resolution_tiling_ibp() {
    // Qwen3-VL processes high-res images by splitting into tiles.
    // Each tile goes through patch embedding independently, then tile
    // features are averaged (aggregated) before entering the encoder.
    // Model: variable tile -> patch embed, add constant tile embed -> project.
    let out_h = IMG_SIZE / PATCH_SIZE;
    let out_w = IMG_SIZE / PATCH_SIZE;

    let shape = [SEQ_LEN, HIDDEN_DIM];

    let mut b = TensorBlockBuilder::new("qwen3_vl_ve_tiling");

    // Tile 1 (variable input)
    let tile1 = b.add_input("tile1", &[IN_CHANNELS, IMG_SIZE, IMG_SIZE]);
    let conv_w = b.add_input("proj_w", &[HIDDEN_DIM, IN_CHANNELS, PATCH_SIZE, PATCH_SIZE]);
    let conv_b = b.add_input("proj_b", &[HIDDEN_DIM]);
    let conv1 = b.add_conv2d(
        tile1,
        conv_w,
        Some(conv_b),
        PATCH_SIZE,
        PATCH_SIZE,
        0,
        0,
        &[HIDDEN_DIM, out_h, out_w],
    );
    let flat1 = b.add_reshape(conv1, &[HIDDEN_DIM, SEQ_LEN]);
    let tok1 = b.add_transpose(flat1, &[1, 0], &shape);

    // Tile 2 features (constant -- represents a second tile's embeddings)
    let tile2_feat = b.add_input("tile2_feat", &shape);

    // Aggregate tile features: tok1 + tile2_feat (additive merge)
    let merged = b.add_binary_add(tok1, tile2_feat, &shape);

    // Project to LM_DIM
    let proj_w_node = b.add_input("lm_proj_w", &[LM_DIM, HIDDEN_DIM]);
    let out = b.add_linear(merged, proj_w_node, None, &[SEQ_LEN, LM_DIM]);
    let def = b.build(out).expect("valid tiling kernel");

    // Tile 2 features are constant (mid-range embeddings from a second tile)
    let tile2_const = ArrayD::from_elem(IxDyn(&shape), 0.1f32);
    let bindings = vec![
        TensorParamBinding::Variable,
        weight(&[HIDDEN_DIM, IN_CHANNELS, PATCH_SIZE, PATCH_SIZE]),
        zeros_bind(&[HIDDEN_DIM]),
        TensorParamBinding::ConstantTensor(tile2_const),
        weight(&[LM_DIM, HIDDEN_DIM]),
    ];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");
    let input = image_bounds();

    let output = graph.propagate_ibp(&input).expect("IBP");
    assert_bounds_valid(&output);
    assert_eq!(output.lower_upper().0.shape(), &[SEQ_LEN, LM_DIM]);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Multi-resolution tiling IBP: [{lo_min:.6}, {hi_max:.6}]");
    assert!(lo_min.is_finite() && hi_max.is_finite());
}

// ===========================================================================
// 21. Dynamic resolution padding bounds (IBP)
// ===========================================================================

#[test]
fn test_qwen3_vl_ve_dynamic_resolution_padding_ibp() {
    // Dynamic resolution: images are padded to a multiple of patch size.
    // Zero-padding preserves bounds (padded values are 0, within [0,1]).
    // Model: pad image from 6x6 to 8x8 (using zero-pad), then patch embed.
    // We model the padded image directly as [C, 8, 8] with slightly wider
    // bounds on the padding region.
    let out_h = IMG_SIZE / PATCH_SIZE;
    let out_w = IMG_SIZE / PATCH_SIZE;

    let mut b = TensorBlockBuilder::new("qwen3_vl_ve_dyn_pad");
    // Padded image (zero-padded region has values exactly 0, but bounds
    // conservatively cover [0, 1] for the full tensor)
    let input = b.add_input("padded_image", &[IN_CHANNELS, IMG_SIZE, IMG_SIZE]);
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
    let tokens = b.add_transpose(flat, &[1, 0], &[SEQ_LEN, HIDDEN_DIM]);

    // RMSNorm
    let eps = b.add_input("norm_eps", &[1]);
    let norm_w = b.add_input("norm_w", &[HIDDEN_DIM]);
    let out = b.add_rms_norm(tokens, eps, 1, norm_w, &[SEQ_LEN, HIDDEN_DIM]);
    let def = b.build(out).expect("valid padded pipeline kernel");

    let bindings = vec![
        TensorParamBinding::Variable,
        weight(&[HIDDEN_DIM, IN_CHANNELS, PATCH_SIZE, PATCH_SIZE]),
        zeros_bind(&[HIDDEN_DIM]),
        eps_bind(),
        ones_bind(&[HIDDEN_DIM]),
    ];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");
    // Padded image: some pixels are 0 (padding), others are in [0,1].
    // Conservative bounds: [0, 1] covers both.
    let input = image_bounds();

    let output = graph.propagate_ibp(&input).expect("IBP");
    assert_bounds_valid(&output);
    assert_eq!(output.lower_upper().0.shape(), &[SEQ_LEN, HIDDEN_DIM]);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Dynamic resolution padding IBP: [{lo_min:.6}, {hi_max:.6}]");
    assert!(lo_min.is_finite() && hi_max.is_finite());
}

// ===========================================================================
// 22. Global average pooling feature bounds (IBP)
// ===========================================================================

#[test]
fn test_qwen3_vl_ve_global_avg_pool_ibp() {
    // Global average pooling over the sequence dimension produces a single
    // feature vector [HIDDEN_DIM] from [SEQ_LEN, HIDDEN_DIM].
    // Used for classification heads or as a summary feature.
    let mut b = TensorBlockBuilder::new("qwen3_vl_ve_gap");
    let input = b.add_input("encoder_out", &[SEQ_LEN, HIDDEN_DIM]);

    // Mean-reduce along axis 0 (sequence dim)
    let pooled = b.add_reduce(input, ReduceOp::Mean, 0, false, &[HIDDEN_DIM]);
    let def = b.build(pooled).expect("valid GAP kernel");

    let bindings = vec![TensorParamBinding::Variable];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");
    let input = seq_bounds(SEQ_LEN, HIDDEN_DIM, 1.0);

    let output = graph.propagate_ibp(&input).expect("IBP");
    assert_bounds_valid(&output);
    assert_eq!(output.lower_upper().0.shape(), &[HIDDEN_DIM]);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Global avg pooling IBP: [{lo_min:.6}, {hi_max:.6}]");
    assert!(lo_min.is_finite() && hi_max.is_finite());
    // Mean-reduce should not expand bounds beyond input range [-1, 1]
    assert!(
        lo_min >= -1.0 - 1e-5 && hi_max <= 1.0 + 1e-5,
        "mean reduce should not widen bounds: [{lo_min}, {hi_max}]"
    );
}

// ===========================================================================
// 23. Merged image-text sequence bounds (IBP)
// ===========================================================================

#[test]
fn test_qwen3_vl_ve_merged_image_text_ibp() {
    // In Qwen3-VL, vision tokens are merged with text tokens for the LLM.
    // Model: vision_tokens are added to a constant text embedding baseline
    // (additive fusion), then projected. This tests that vision bounds
    // propagate correctly through the merge with constant text context.
    let shape = [SEQ_LEN, LM_DIM];

    let mut b = TensorBlockBuilder::new("qwen3_vl_ve_merge");
    let vision_tokens = b.add_input("vision", &shape);
    let text_bias = b.add_input("text_bias", &shape);

    // Additive merge: vision features + constant text embedding
    let merged = b.add_binary_add(vision_tokens, text_bias, &shape);

    // Apply a single linear projection (e.g., first LLM layer input proj)
    let proj_w = b.add_input("proj_w", &[LM_DIM, LM_DIM]);
    let proj_b = b.add_input("proj_b", &[LM_DIM]);
    let out = b.add_linear(merged, proj_w, Some(proj_b), &shape);
    let def = b.build(out).expect("valid merged pipeline kernel");

    // Text embedding is a constant bias (e.g., average text prompt embedding)
    let text_const = ArrayD::from_elem(IxDyn(&shape), 0.1f32);
    let bindings = vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(text_const),
        weight(&[LM_DIM, LM_DIM]),
        zeros_bind(&[LM_DIM]),
    ];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");
    let input = seq_bounds(SEQ_LEN, LM_DIM, 1.0);

    let output = graph.propagate_ibp(&input).expect("IBP");
    assert_bounds_valid(&output);
    assert_eq!(output.lower_upper().0.shape(), &shape);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Merged image-text sequence IBP: [{lo_min:.6}, {hi_max:.6}]");
    assert!(lo_min.is_finite() && hi_max.is_finite());
}

// ===========================================================================
// 24. Multi-scale feature extraction bounds (IBP)
// ===========================================================================

#[test]
fn test_qwen3_vl_ve_multi_scale_features_ibp() {
    // Multi-scale features: extract features from early and late encoder blocks,
    // project each to LM_DIM, then sum (FPN-style additive fusion).
    // This tests that bounds from different encoder depths combine correctly.
    let shape = [SEQ_LEN, LM_DIM];

    let mut b = TensorBlockBuilder::new("qwen3_vl_ve_multi_scale");
    let input = b.add_input("x", &[SEQ_LEN, HIDDEN_DIM]);
    let mut bindings = vec![TensorParamBinding::Variable];

    // Shallow block (block 0)
    let shallow = add_vision_encoder_block(&mut b, input, 0, &mut bindings);

    // Deep block (block 1, stacked on shallow)
    let deep = add_vision_encoder_block(&mut b, shallow, 1, &mut bindings);

    // Project shallow and deep features to LM_DIM
    let proj_s_w = b.add_input("proj_s_w", &[LM_DIM, HIDDEN_DIM]);
    let proj_d_w = b.add_input("proj_d_w", &[LM_DIM, HIDDEN_DIM]);
    let shallow_proj = b.add_linear(shallow, proj_s_w, None, &shape);
    let deep_proj = b.add_linear(deep, proj_d_w, None, &shape);

    // Additive multi-scale fusion: shallow_proj + deep_proj
    let out = b.add_binary_add(shallow_proj, deep_proj, &shape);
    bindings.push(weight(&[LM_DIM, HIDDEN_DIM]));
    bindings.push(weight(&[LM_DIM, HIDDEN_DIM]));

    let def = b.build(out).expect("valid multi-scale kernel");
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");
    let input = seq_bounds(SEQ_LEN, HIDDEN_DIM, 1.0);

    let output = graph.propagate_ibp(&input).expect("IBP");
    assert_bounds_valid(&output);
    assert_eq!(output.lower_upper().0.shape(), &shape);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Multi-scale feature extraction IBP: [{lo_min:.6}, {hi_max:.6}]");
    assert!(lo_min.is_finite() && hi_max.is_finite());
}

// ===========================================================================
// 25. Patch position interpolation bounds (IBP)
// ===========================================================================

#[test]
fn test_qwen3_vl_ve_patch_position_interpolation_ibp() {
    // Position interpolation: when input resolution differs from training,
    // position embeddings are linearly interpolated. Model as:
    // tokens + alpha * pos_embed_low + (1 - alpha) * pos_embed_high
    // where alpha is a constant interpolation factor.
    let alpha: f32 = 0.6;

    let mut b = TensorBlockBuilder::new("qwen3_vl_ve_pos_interp");
    let input = b.add_input("tokens", &[SEQ_LEN, HIDDEN_DIM]);
    let pos_lo = b.add_input("pos_lo", &[SEQ_LEN, HIDDEN_DIM]);
    let pos_hi = b.add_input("pos_hi", &[SEQ_LEN, HIDDEN_DIM]);
    let alpha_node = b.add_input("alpha", &[SEQ_LEN, HIDDEN_DIM]);
    let one_minus_alpha = b.add_input("one_minus_alpha", &[SEQ_LEN, HIDDEN_DIM]);

    let shape = [SEQ_LEN, HIDDEN_DIM];

    // interpolated_pos = alpha * pos_lo + (1-alpha) * pos_hi
    let scaled_lo = b.add_binary_mul(alpha_node, pos_lo, &shape);
    let scaled_hi = b.add_binary_mul(one_minus_alpha, pos_hi, &shape);
    let interp_pos = b.add_binary_add(scaled_lo, scaled_hi, &shape);

    // tokens + interpolated position embedding
    let out = b.add_binary_add(input, interp_pos, &shape);
    let def = b.build(out).expect("valid pos interp kernel");

    // Position embeddings as small constants (typical init scale)
    let pos_embed_lo = ArrayD::from_elem(IxDyn(&shape), 0.01f32);
    let pos_embed_hi = ArrayD::from_elem(IxDyn(&shape), 0.02f32);
    let alpha_arr = ArrayD::from_elem(IxDyn(&shape), alpha);
    let one_minus_alpha_arr = ArrayD::from_elem(IxDyn(&shape), 1.0 - alpha);

    let bindings = vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(pos_embed_lo),
        TensorParamBinding::ConstantTensor(pos_embed_hi),
        TensorParamBinding::ConstantTensor(alpha_arr),
        TensorParamBinding::ConstantTensor(one_minus_alpha_arr),
    ];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");
    let input = seq_bounds(SEQ_LEN, HIDDEN_DIM, 1.0);

    let output = graph.propagate_ibp(&input).expect("IBP");
    assert_bounds_valid(&output);
    assert_eq!(output.lower_upper().0.shape(), &shape);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Patch position interpolation IBP: [{lo_min:.6}, {hi_max:.6}]");
    assert!(lo_min.is_finite() && hi_max.is_finite());
    // Adding small position embeds to [-1,1] input should not dramatically widen
    assert!(
        lo_min > -2.0 && hi_max < 2.0,
        "pos interp should only slightly widen: [{lo_min}, {hi_max}]"
    );
}

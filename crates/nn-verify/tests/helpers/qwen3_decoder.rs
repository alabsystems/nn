// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Builder helpers for Qwen3 decoder NY composition tests.
//!
//! Architecture: token_emb -> N x (RmsNorm + GQA + residual, RmsNorm + SwiGLU + residual)
//! -> RmsNorm -> lm_head.
//!
//! Key Qwen3 features modeled:
//! - RmsNorm (not LayerNorm) for pre-norm
//! - SwiGLU MLP: `down_proj(silu(gate_proj(x)) * up_proj(x))`
//! - SiLU = sigmoid(x) * x, decomposed as sigmoid + binary_mul
//! - GQA (n_kv_heads < n_heads): K/V projected with fewer heads, repeated
//! - RoPE half-split rotation on Q and K
//! - Causal self-attention
//!
//! Dimensions: D_MODEL=8, N_HEADS=2, N_KV_HEADS=1, HEAD_DIM=4, SEQ_LEN=4.
//!
//! Part of #4186: Add compose verification tests for Qwen3 decoder bounds.

use nn_dsl::tensor_block_builder::TensorBlockBuilder;
use nn_dsl::tensor_ir::{TensorKernelDef, TensorNodeId};
use nn_dsl::AttentionMask;
use nn_verify::TensorParamBinding;
use ndarray::{ArrayD, IxDyn};

// ---------------------------------------------------------------------------
// Small-scale dimensions for NY tractability
// ---------------------------------------------------------------------------

/// Model dimension (production: 4096 for Qwen3-8B).
pub(super) const D_MODEL: usize = 8;

/// Number of query attention heads (production: 32 for Qwen3-8B).
pub(super) const N_HEADS: usize = 2;

/// Number of key/value attention heads for GQA (production: 4 for Qwen3-8B).
pub(super) const N_KV_HEADS: usize = 1;

/// Per-head dimension: D_MODEL / N_HEADS.
pub(super) const HEAD_DIM: usize = D_MODEL / N_HEADS; // 4

/// Half of head dimension (for RoPE rotation pairs).
pub(super) const HALF_DIM: usize = HEAD_DIM / 2; // 2

/// GQA repetition factor: N_HEADS / N_KV_HEADS.
pub(super) const GQA_REP: usize = N_HEADS / N_KV_HEADS; // 2

/// FFN intermediate dimension (production: ~2.67x d_model).
pub(super) const INTERMEDIATE_SIZE: usize = D_MODEL * 3; // 24

/// Vocabulary size (production: 151936 for Qwen3).
pub(super) const VOCAB_SIZE: usize = 16;

/// Sequence length (production: up to 131072 with YaRN).
pub(super) const SEQ_LEN: usize = 4;

/// KV projection dimension: N_KV_HEADS * HEAD_DIM.
const KV_DIM: usize = N_KV_HEADS * HEAD_DIM; // 4

/// Weight magnitude for small-scale test weights.
const WEIGHT_MAG: f32 = 0.001;

fn w(shape: &[usize]) -> ArrayD<f32> {
    ArrayD::from_elem(IxDyn(shape), WEIGHT_MAG)
}

fn ones(shape: &[usize]) -> ArrayD<f32> {
    ArrayD::from_elem(IxDyn(shape), 1.0f32)
}

// ---------------------------------------------------------------------------
// RoPE cos/sin tables
// ---------------------------------------------------------------------------

fn rope_cos_table() -> ArrayD<f32> {
    let mut data = vec![0.0f32; SEQ_LEN * HALF_DIM];
    for pos in 0..SEQ_LEN {
        for i in 0..HALF_DIM {
            let theta = (pos as f64) / 10000.0_f64.powf(2.0 * i as f64 / HEAD_DIM as f64);
            data[pos * HALF_DIM + i] = theta.cos() as f32;
        }
    }
    ArrayD::from_shape_vec(IxDyn(&[SEQ_LEN, HALF_DIM]), data).expect("valid cos table")
}

fn rope_sin_table() -> ArrayD<f32> {
    let mut data = vec![0.0f32; SEQ_LEN * HALF_DIM];
    for pos in 0..SEQ_LEN {
        for i in 0..HALF_DIM {
            let theta = (pos as f64) / 10000.0_f64.powf(2.0 * i as f64 / HEAD_DIM as f64);
            data[pos * HALF_DIM + i] = theta.sin() as f32;
        }
    }
    ArrayD::from_shape_vec(IxDyn(&[SEQ_LEN, HALF_DIM]), data).expect("valid sin table")
}

// ---------------------------------------------------------------------------
// 1. RoPE position encoding
// ---------------------------------------------------------------------------

/// RoPE half-split rotation on a single head's vector.
///
/// Input: `[SEQ_LEN, HEAD_DIM]` (Variable).
/// Output: `[SEQ_LEN, HEAD_DIM]`.
///
/// y1 = x1*cos - x2*sin, y2 = x1*sin + x2*cos
pub(super) fn build_rope() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("qwen3_dec_rope");

    let full = [SEQ_LEN, HEAD_DIM];
    let half = [SEQ_LEN, HALF_DIM];

    let input = b.add_input("x", &full);
    let cos = b.add_input("cos", &half);
    let sin = b.add_input("sin", &half);
    let neg_one = b.add_input("neg_one", &half);

    let output = rope_inline(&mut b, input, cos, sin, neg_one, &full, &half);
    b.build(output).expect("valid RoPE graph")
}

pub(super) fn rope_bindings() -> Vec<TensorParamBinding> {
    vec![
        TensorParamBinding::Variable, // x
        TensorParamBinding::ConstantTensor(rope_cos_table()),
        TensorParamBinding::ConstantTensor(rope_sin_table()),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[SEQ_LEN, HALF_DIM]), -1.0f32)),
    ]
}

// ---------------------------------------------------------------------------
// 2. GQA attention score bounds
// ---------------------------------------------------------------------------

/// Grouped-query attention with causal mask and KV head expansion.
///
/// Input: `[SEQ_LEN, D_MODEL]` (Variable).
/// Output: `[SEQ_LEN, D_MODEL]`.
///
/// Q has N_HEADS heads, K/V have N_KV_HEADS heads. K/V repeated GQA_REP times.
pub(super) fn build_gqa() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("qwen3_dec_gqa");

    let shape = [SEQ_LEN, D_MODEL];
    let input = b.add_input("hidden", &shape);
    let q_w = b.add_input("q_proj", &[D_MODEL, D_MODEL]);
    let k_w = b.add_input("k_proj", &[KV_DIM, D_MODEL]);
    let v_w = b.add_input("v_proj", &[KV_DIM, D_MODEL]);
    let o_w = b.add_input("o_proj", &[D_MODEL, D_MODEL]);

    // Q/K/V projections
    let q = b.add_linear(input, q_w, None, &[SEQ_LEN, D_MODEL]);
    let k = b.add_linear(input, k_w, None, &[SEQ_LEN, KV_DIM]);
    let v = b.add_linear(input, v_w, None, &[SEQ_LEN, KV_DIM]);

    // Reshape to multi-head layout
    let q = b.add_reshape(q, &[SEQ_LEN, N_HEADS, HEAD_DIM]);
    let k = b.add_reshape(k, &[SEQ_LEN, N_KV_HEADS, HEAD_DIM]);
    let v = b.add_reshape(v, &[SEQ_LEN, N_KV_HEADS, HEAD_DIM]);

    // GQA repeat: [S, KV_H, HD] -> [S, KV_H, 1, HD] -> broadcast -> [S, H, HD]
    let k = b.add_reshape(k, &[SEQ_LEN, N_KV_HEADS, 1, HEAD_DIM]);
    let k = b.add_broadcast(k, &[SEQ_LEN, N_KV_HEADS, GQA_REP, HEAD_DIM]);
    let k = b.add_reshape(k, &[SEQ_LEN, N_HEADS, HEAD_DIM]);

    let v = b.add_reshape(v, &[SEQ_LEN, N_KV_HEADS, 1, HEAD_DIM]);
    let v = b.add_broadcast(v, &[SEQ_LEN, N_KV_HEADS, GQA_REP, HEAD_DIM]);
    let v = b.add_reshape(v, &[SEQ_LEN, N_HEADS, HEAD_DIM]);

    // Transpose to [H, S, HD] for per-head attention
    let q = b.add_transpose(q, &[1, 0, 2], &[N_HEADS, SEQ_LEN, HEAD_DIM]);
    let k = b.add_transpose(k, &[1, 0, 2], &[N_HEADS, SEQ_LEN, HEAD_DIM]);
    let v = b.add_transpose(v, &[1, 0, 2], &[N_HEADS, SEQ_LEN, HEAD_DIM]);

    // Causal attention with scaling
    let scale = 1.0 / (HEAD_DIM as f32).sqrt();
    let attn = b.add_attention(
        q,
        k,
        v,
        AttentionMask::Causal,
        Some(scale),
        &[N_HEADS, SEQ_LEN, HEAD_DIM],
    );

    // Transpose back + reshape + output projection
    let attn = b.add_transpose(attn, &[1, 0, 2], &[SEQ_LEN, N_HEADS, HEAD_DIM]);
    let attn = b.add_reshape(attn, &[SEQ_LEN, D_MODEL]);
    let output = b.add_linear(attn, o_w, None, &shape);

    b.build(output).expect("valid GQA graph")
}

pub(super) fn gqa_bindings() -> Vec<TensorParamBinding> {
    vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(w(&[D_MODEL, D_MODEL])),
        TensorParamBinding::ConstantTensor(w(&[KV_DIM, D_MODEL])),
        TensorParamBinding::ConstantTensor(w(&[KV_DIM, D_MODEL])),
        TensorParamBinding::ConstantTensor(w(&[D_MODEL, D_MODEL])),
    ]
}

// ---------------------------------------------------------------------------
// 3. SwiGLU activation
// ---------------------------------------------------------------------------

/// SwiGLU MLP: silu(gate_proj(x)) * up_proj(x) -> down_proj.
///
/// Input: `[SEQ_LEN, D_MODEL]` (Variable).
/// Output: `[SEQ_LEN, D_MODEL]`.
pub(super) fn build_swiglu() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("qwen3_dec_swiglu");
    let shape = [SEQ_LEN, D_MODEL];
    let inter = [SEQ_LEN, INTERMEDIATE_SIZE];

    let input = b.add_input("hidden", &shape);
    let gate_w = b.add_input("gate_w", &[INTERMEDIATE_SIZE, D_MODEL]);
    let up_w = b.add_input("up_w", &[INTERMEDIATE_SIZE, D_MODEL]);
    let down_w = b.add_input("down_w", &[D_MODEL, INTERMEDIATE_SIZE]);

    let out = swiglu_inline(&mut b, input, gate_w, up_w, down_w, &shape, &inter);
    b.build(out).expect("valid SwiGLU graph")
}

pub(super) fn swiglu_bindings() -> Vec<TensorParamBinding> {
    vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(w(&[INTERMEDIATE_SIZE, D_MODEL])),
        TensorParamBinding::ConstantTensor(w(&[INTERMEDIATE_SIZE, D_MODEL])),
        TensorParamBinding::ConstantTensor(w(&[D_MODEL, INTERMEDIATE_SIZE])),
    ]
}

// ---------------------------------------------------------------------------
// 4. RMSNorm output
// ---------------------------------------------------------------------------

/// RMSNorm: x / rms(x) * weight.
///
/// Input: `[SEQ_LEN, D_MODEL]` (Variable).
/// Output: `[SEQ_LEN, D_MODEL]`.
pub(super) fn build_rmsnorm() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("qwen3_dec_rmsnorm");
    let shape = [SEQ_LEN, D_MODEL];

    let input = b.add_input("hidden", &shape);
    let eps = b.add_input("eps", &[1]);
    let weight = b.add_input("norm_w", &[D_MODEL]);
    let output = b.add_rms_norm(input, eps, 1, weight, &shape);
    b.build(output).expect("valid RMSNorm graph")
}

pub(super) fn rmsnorm_bindings() -> Vec<TensorParamBinding> {
    vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantScalar(1e-5),
        TensorParamBinding::ConstantTensor(ones(&[D_MODEL])),
    ]
}

// ---------------------------------------------------------------------------
// 5. Decoder block residual stream
// ---------------------------------------------------------------------------

/// Single Qwen3 decoder block with residual connections.
///
/// Pre-norm structure:
///   RMSNorm -> MHA(causal) -> + residual -> RMSNorm -> SwiGLU -> + residual
///
/// Input: `[SEQ_LEN, D_MODEL]` (Variable).
/// Output: `[SEQ_LEN, D_MODEL]`.
pub(super) fn build_decoder_block() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("qwen3_dec_block");
    let shape = [SEQ_LEN, D_MODEL];
    let inter = [SEQ_LEN, INTERMEDIATE_SIZE];

    let input = b.add_input("hidden", &shape);
    let eps = b.add_input("eps", &[1]);

    // Attention sub-block weights
    let attn_ln_w = b.add_input("attn_ln_w", &[D_MODEL]);
    let q_w = b.add_input("q_w", &[D_MODEL, D_MODEL]);
    let k_w = b.add_input("k_w", &[D_MODEL, D_MODEL]);
    let v_w = b.add_input("v_w", &[D_MODEL, D_MODEL]);
    let o_w = b.add_input("o_w", &[D_MODEL, D_MODEL]);

    // MLP sub-block weights
    let mlp_ln_w = b.add_input("mlp_ln_w", &[D_MODEL]);
    let gate_w = b.add_input("gate_w", &[INTERMEDIATE_SIZE, D_MODEL]);
    let up_w = b.add_input("up_w", &[INTERMEDIATE_SIZE, D_MODEL]);
    let down_w = b.add_input("down_w", &[D_MODEL, INTERMEDIATE_SIZE]);

    let output = decoder_block_inline(
        &mut b, input, eps, attn_ln_w, q_w, k_w, v_w, o_w, mlp_ln_w, gate_w, up_w, down_w, &shape,
        &inter,
    );
    b.build(output).expect("valid decoder block graph")
}

pub(super) fn decoder_block_bindings() -> Vec<TensorParamBinding> {
    let mut bindings = Vec::new();
    bindings.push(TensorParamBinding::Variable);
    bindings.push(TensorParamBinding::ConstantScalar(1e-5));
    push_block_bindings(&mut bindings);
    bindings
}

// ---------------------------------------------------------------------------
// 6. Full decoder layer composition
// ---------------------------------------------------------------------------

/// Two decoder blocks + final RMSNorm + LM head projection.
///
/// Input: `[SEQ_LEN, D_MODEL]` (Variable).
/// Output: `[SEQ_LEN, VOCAB_SIZE]`.
pub(super) fn build_full_decoder() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("qwen3_dec_full");
    let shape = [SEQ_LEN, D_MODEL];
    let inter = [SEQ_LEN, INTERMEDIATE_SIZE];
    let out_shape = [SEQ_LEN, VOCAB_SIZE];

    let input = b.add_input("token_emb", &shape);
    let eps = b.add_input("eps", &[1]);

    // 2 decoder blocks
    let mut current = input;
    for i in 0..2 {
        current = add_decoder_block(&mut b, current, eps, i, &shape, &inter);
    }

    // Final RMSNorm
    let ln_final_w = b.add_input("ln_final_w", &[D_MODEL]);
    let normed = b.add_rms_norm(current, eps, 1, ln_final_w, &shape);

    // LM head: [SEQ_LEN, D_MODEL] x [D_MODEL, VOCAB_SIZE]
    let lm_head_w = b.add_input("lm_head_w", &[D_MODEL, VOCAB_SIZE]);
    let logits = b.add_matmul(normed, lm_head_w, false, None, &out_shape);

    b.build(logits).expect("valid full decoder graph")
}

pub(super) fn full_decoder_bindings() -> Vec<TensorParamBinding> {
    let mut bindings = Vec::new();
    bindings.push(TensorParamBinding::Variable); // token_emb
    bindings.push(TensorParamBinding::ConstantScalar(1e-5)); // eps
                                                             // 2 decoder blocks
    for _ in 0..2 {
        push_block_bindings(&mut bindings);
    }
    // ln_final_w
    bindings.push(TensorParamBinding::ConstantTensor(ones(&[D_MODEL])));
    // lm_head_w
    bindings.push(TensorParamBinding::ConstantTensor(w(&[
        D_MODEL, VOCAB_SIZE,
    ])));
    bindings
}

// ===========================================================================
// Internal helpers
// ===========================================================================

/// SwiGLU MLP inline: silu(gate_proj(x)) * up_proj(x) -> down_proj.
fn swiglu_inline(
    b: &mut TensorBlockBuilder,
    input: TensorNodeId,
    gate_w: TensorNodeId,
    up_w: TensorNodeId,
    down_w: TensorNodeId,
    shape: &[usize],
    inter: &[usize],
) -> TensorNodeId {
    let gate = b.add_linear(input, gate_w, None, inter);
    let gate_sig = b.add_sigmoid(gate, inter);
    let gate_silu = b.add_binary_mul(gate, gate_sig, inter);
    let up = b.add_linear(input, up_w, None, inter);
    let gated = b.add_binary_mul(gate_silu, up, inter);
    b.add_linear(gated, down_w, None, shape)
}

/// RoPE rotation inline.
fn rope_inline(
    b: &mut TensorBlockBuilder,
    input: TensorNodeId,
    cos: TensorNodeId,
    sin: TensorNodeId,
    neg_one: TensorNodeId,
    full: &[usize],
    half: &[usize],
) -> TensorNodeId {
    let half_dim = half[half.len() - 1];
    let x1 = b.add_narrow(input, 1, 0, half_dim, half);
    let x2 = b.add_narrow(input, 1, half_dim, half_dim, half);

    // y1 = x1*cos - x2*sin
    let x1_cos = b.add_binary_mul(x1, cos, half);
    let x2_neg = b.add_binary_mul(x2, neg_one, half);
    let x2_neg_sin = b.add_binary_mul(x2_neg, sin, half);
    let y1 = b.add_binary_add(x1_cos, x2_neg_sin, half);

    // y2 = x1*sin + x2*cos
    let x1_sin = b.add_binary_mul(x1, sin, half);
    let x2_cos = b.add_binary_mul(x2, cos, half);
    let y2 = b.add_binary_add(x1_sin, x2_cos, half);

    b.add_concat(&[y1, y2], 1, full)
}

/// Build a decoder block inline (pre-norm RMSNorm -> MHA -> residual -> RMSNorm -> SwiGLU -> residual).
fn decoder_block_inline(
    b: &mut TensorBlockBuilder,
    input: TensorNodeId,
    eps: TensorNodeId,
    attn_ln_w: TensorNodeId,
    q_w: TensorNodeId,
    k_w: TensorNodeId,
    v_w: TensorNodeId,
    o_w: TensorNodeId,
    mlp_ln_w: TensorNodeId,
    gate_w: TensorNodeId,
    up_w: TensorNodeId,
    down_w: TensorNodeId,
    shape: &[usize],
    inter: &[usize],
) -> TensorNodeId {
    // Sub-block 1: RMSNorm -> MHA(causal) -> residual
    let attn_normed = b.add_rms_norm(input, eps, 1, attn_ln_w, shape);
    let attn_out = b
        .add_multi_head_attention(
            attn_normed,
            q_w,
            k_w,
            v_w,
            o_w,
            N_HEADS,
            AttentionMask::Causal,
            shape,
        )
        .expect("qwen3 decoder self-attention");
    let residual1 = b.add_binary_add(input, attn_out, shape);

    // Sub-block 2: RMSNorm -> SwiGLU -> residual
    let mlp_normed = b.add_rms_norm(residual1, eps, 1, mlp_ln_w, shape);
    let mlp_out = swiglu_inline(b, mlp_normed, gate_w, up_w, down_w, shape, inter);
    b.add_binary_add(residual1, mlp_out, shape)
}

/// Add a decoder block with auto-named weight inputs and push bindings.
fn add_decoder_block(
    b: &mut TensorBlockBuilder,
    input: TensorNodeId,
    eps: TensorNodeId,
    layer_idx: usize,
    shape: &[usize],
    inter: &[usize],
) -> TensorNodeId {
    let pfx = format!("L{layer_idx}");
    let attn_ln_w = b.add_input(&format!("{pfx}_attn_ln_w"), &[D_MODEL]);
    let q_w = b.add_input(&format!("{pfx}_q_w"), &[D_MODEL, D_MODEL]);
    let k_w = b.add_input(&format!("{pfx}_k_w"), &[D_MODEL, D_MODEL]);
    let v_w = b.add_input(&format!("{pfx}_v_w"), &[D_MODEL, D_MODEL]);
    let o_w = b.add_input(&format!("{pfx}_o_w"), &[D_MODEL, D_MODEL]);
    let mlp_ln_w = b.add_input(&format!("{pfx}_mlp_ln_w"), &[D_MODEL]);
    let gate_w = b.add_input(&format!("{pfx}_gate_w"), &[INTERMEDIATE_SIZE, D_MODEL]);
    let up_w = b.add_input(&format!("{pfx}_up_w"), &[INTERMEDIATE_SIZE, D_MODEL]);
    let down_w = b.add_input(&format!("{pfx}_down_w"), &[D_MODEL, INTERMEDIATE_SIZE]);

    decoder_block_inline(
        b, input, eps, attn_ln_w, q_w, k_w, v_w, o_w, mlp_ln_w, gate_w, up_w, down_w, shape, inter,
    )
}

/// Push bindings for one decoder block.
fn push_block_bindings(bindings: &mut Vec<TensorParamBinding>) {
    // attn_ln_w
    bindings.push(TensorParamBinding::ConstantTensor(ones(&[D_MODEL])));
    // q, k, v, o
    for _ in 0..4 {
        bindings.push(TensorParamBinding::ConstantTensor(w(&[D_MODEL, D_MODEL])));
    }
    // mlp_ln_w
    bindings.push(TensorParamBinding::ConstantTensor(ones(&[D_MODEL])));
    // gate_w, up_w
    for _ in 0..2 {
        bindings.push(TensorParamBinding::ConstantTensor(w(&[
            INTERMEDIATE_SIZE,
            D_MODEL,
        ])));
    }
    // down_w
    bindings.push(TensorParamBinding::ConstantTensor(w(&[
        D_MODEL,
        INTERMEDIATE_SIZE,
    ])));
}

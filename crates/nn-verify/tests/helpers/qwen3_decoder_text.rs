// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Builder helpers for Qwen3 TEXT-ONLY decoder NY composition tests.
//!
//! Provides TensorBlockBuilder graphs for 10 verifiable sub-blocks of the
//! Qwen3 decoder-only LLM (text generation, not VL):
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
//! Uses dimensions: D_MODEL=16, N_HEADS=4, N_KV_HEADS=2, FFN_DIM=48, SEQ=6, VOCAB=32.
//! These differ from the existing qwen3_decoder.rs (D_MODEL=8) and
//! qwen3_decoder_pipeline.rs (D_MODEL=16, N_HEADS=2) to test with GQA ratio=2
//! and larger FFN.
//!
//! Part of #3942: Qwen3 decoder compose verification tests.

use nn_dsl::tensor_block_builder::TensorBlockBuilder;
use nn_dsl::tensor_ir::{TensorKernelDef, TensorNodeId};
use nn_dsl::AttentionMask;
use nn_verify::TensorParamBinding;
use ndarray::{ArrayD, IxDyn};

// ---------------------------------------------------------------------------
// Dimensions — structurally representative with GQA ratio = 2
// ---------------------------------------------------------------------------

/// Model dimension (production: 4096 for Qwen3-8B).
pub(super) const D_MODEL: usize = 16;

/// Number of query attention heads (production: 32 for Qwen3-8B).
pub(super) const N_HEADS: usize = 4;

/// Number of key/value attention heads (GQA: fewer than Q heads).
pub(super) const N_KV_HEADS: usize = 2;

/// Per-head dimension: D_MODEL / N_HEADS.
pub(super) const HEAD_DIM: usize = D_MODEL / N_HEADS; // 4

/// Half of head dimension (RoPE rotation pairs).
pub(super) const HALF_DIM: usize = HEAD_DIM / 2; // 2

/// GQA repetition factor: each KV head serves this many Q heads.
pub(super) const GQA_REP: usize = N_HEADS / N_KV_HEADS; // 2

/// FFN intermediate dimension (production: ~2.67x d_model).
pub(super) const FFN_DIM: usize = 48;

/// Vocabulary size (production: 151936 for Qwen3).
pub(super) const VOCAB: usize = 32;

/// Sequence length.
pub(super) const SEQ: usize = 6;

/// KV projection dimension: N_KV_HEADS * HEAD_DIM.
const KV_DIM: usize = N_KV_HEADS * HEAD_DIM; // 8

/// Weight magnitude for small-scale test weights.
const W: f32 = 0.001;

// ---------------------------------------------------------------------------
// 1. RMSNorm pre-attention
// ---------------------------------------------------------------------------

/// Isolated RMSNorm: `[SEQ, D_MODEL]` -> `[SEQ, D_MODEL]`.
pub(super) fn build_rmsnorm() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("qwen3_text_rmsnorm");
    let shape = [SEQ, D_MODEL];
    let input = b.add_input("hidden", &shape);
    let eps = b.add_input("eps", &[1]);
    let weight = b.add_input("norm_w", &[D_MODEL]);
    let output = b.add_rms_norm(input, eps, 1, weight, &shape);
    b.build(output).expect("valid RMSNorm graph")
}

pub(super) fn rmsnorm_bindings() -> Vec<TensorParamBinding> {
    vec![
        TensorParamBinding::Variable, // hidden
        TensorParamBinding::ConstantScalar(1e-5),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[D_MODEL]), 1.0f32)),
    ]
}

// ---------------------------------------------------------------------------
// 2. SwiGLU FFN
// ---------------------------------------------------------------------------

/// SwiGLU MLP: `[SEQ, D_MODEL]` -> `[SEQ, D_MODEL]`.
///
/// SwiGLU(x) = down_proj(silu(gate_proj(x)) * up_proj(x))
pub(super) fn build_swiglu() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("qwen3_text_swiglu");
    let shape = [SEQ, D_MODEL];
    let inter = [SEQ, FFN_DIM];

    let input = b.add_input("hidden", &shape);
    let gate_w = b.add_input("gate_w", &[FFN_DIM, D_MODEL]);
    let up_w = b.add_input("up_w", &[FFN_DIM, D_MODEL]);
    let down_w = b.add_input("down_w", &[D_MODEL, FFN_DIM]);

    let out = swiglu_inline(&mut b, input, gate_w, up_w, down_w, &shape, &inter);
    b.build(out).expect("valid SwiGLU graph")
}

pub(super) fn swiglu_bindings() -> Vec<TensorParamBinding> {
    vec![
        TensorParamBinding::Variable, // hidden
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[FFN_DIM, D_MODEL]), W)),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[FFN_DIM, D_MODEL]), W)),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[D_MODEL, FFN_DIM]), W)),
    ]
}

// ---------------------------------------------------------------------------
// 3. GQA attention
// ---------------------------------------------------------------------------

/// Grouped-query attention with causal mask.
///
/// Input: `[SEQ, D_MODEL]` (Variable).
/// Output: `[SEQ, D_MODEL]`.
///
/// Q has N_HEADS heads, K/V have N_KV_HEADS heads (each KV head repeated
/// GQA_REP times to match Q).
pub(super) fn build_gqa() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("qwen3_text_gqa");

    let shape = [SEQ, D_MODEL];
    let input = b.add_input("hidden", &shape);
    let q_w = b.add_input("q_proj", &[D_MODEL, D_MODEL]);
    let k_w = b.add_input("k_proj", &[KV_DIM, D_MODEL]);
    let v_w = b.add_input("v_proj", &[KV_DIM, D_MODEL]);
    let o_w = b.add_input("o_proj", &[D_MODEL, D_MODEL]);

    // Q/K/V projections
    let q = b.add_linear(input, q_w, None, &[SEQ, D_MODEL]);
    let k = b.add_linear(input, k_w, None, &[SEQ, KV_DIM]);
    let v = b.add_linear(input, v_w, None, &[SEQ, KV_DIM]);

    // Reshape to multi-head layout
    let q = b.add_reshape(q, &[SEQ, N_HEADS, HEAD_DIM]);
    let k = b.add_reshape(k, &[SEQ, N_KV_HEADS, HEAD_DIM]);
    let v = b.add_reshape(v, &[SEQ, N_KV_HEADS, HEAD_DIM]);

    // Repeat K/V heads to match Q: [S, KV_H, HD] -> [S, KV_H, 1, HD] -> broadcast -> [S, H, HD]
    let k = b.add_reshape(k, &[SEQ, N_KV_HEADS, 1, HEAD_DIM]);
    let k = b.add_broadcast(k, &[SEQ, N_KV_HEADS, GQA_REP, HEAD_DIM]);
    let k = b.add_reshape(k, &[SEQ, N_HEADS, HEAD_DIM]);

    let v = b.add_reshape(v, &[SEQ, N_KV_HEADS, 1, HEAD_DIM]);
    let v = b.add_broadcast(v, &[SEQ, N_KV_HEADS, GQA_REP, HEAD_DIM]);
    let v = b.add_reshape(v, &[SEQ, N_HEADS, HEAD_DIM]);

    // Transpose to [H, S, HD] for per-head attention
    let q = b.add_transpose(q, &[1, 0, 2], &[N_HEADS, SEQ, HEAD_DIM]);
    let k = b.add_transpose(k, &[1, 0, 2], &[N_HEADS, SEQ, HEAD_DIM]);
    let v = b.add_transpose(v, &[1, 0, 2], &[N_HEADS, SEQ, HEAD_DIM]);

    // Causal attention
    let scale = 1.0 / (HEAD_DIM as f32).sqrt();
    let attn = b.add_attention(
        q,
        k,
        v,
        AttentionMask::Causal,
        Some(scale),
        &[N_HEADS, SEQ, HEAD_DIM],
    );

    // Transpose back + reshape
    let attn = b.add_transpose(attn, &[1, 0, 2], &[SEQ, N_HEADS, HEAD_DIM]);
    let attn = b.add_reshape(attn, &[SEQ, D_MODEL]);

    // Output projection
    let output = b.add_linear(attn, o_w, None, &shape);
    b.build(output).expect("valid GQA graph")
}

pub(super) fn gqa_bindings() -> Vec<TensorParamBinding> {
    vec![
        TensorParamBinding::Variable, // hidden
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[D_MODEL, D_MODEL]), W)),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[KV_DIM, D_MODEL]), W)),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[KV_DIM, D_MODEL]), W)),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[D_MODEL, D_MODEL]), W)),
    ]
}

// ---------------------------------------------------------------------------
// 4. RoPE application
// ---------------------------------------------------------------------------

/// RoPE rotation on a single head's Q or K vector.
///
/// Input: `[SEQ, HEAD_DIM]` (Variable).
/// Output: `[SEQ, HEAD_DIM]`.
pub(super) fn build_rope() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("qwen3_text_rope");

    let full = [SEQ, HEAD_DIM];
    let half = [SEQ, HALF_DIM];

    let input = b.add_input("x", &full);
    let cos = b.add_input("cos", &half);
    let sin = b.add_input("sin", &half);
    let neg_one = b.add_input("neg_one", &half);

    let output = rope_inline(&mut b, input, cos, sin, neg_one, &full, &half);
    b.build(output).expect("valid RoPE graph")
}

pub(super) fn rope_bindings() -> Vec<TensorParamBinding> {
    let mut cos_data = vec![0.0f32; SEQ * HALF_DIM];
    let mut sin_data = vec![0.0f32; SEQ * HALF_DIM];
    for pos in 0..SEQ {
        for i in 0..HALF_DIM {
            let theta = (pos as f64) / 10000.0_f64.powf(2.0 * i as f64 / HEAD_DIM as f64);
            cos_data[pos * HALF_DIM + i] = theta.cos() as f32;
            sin_data[pos * HALF_DIM + i] = theta.sin() as f32;
        }
    }
    vec![
        TensorParamBinding::Variable, // x
        TensorParamBinding::ConstantTensor(
            ArrayD::from_shape_vec(IxDyn(&[SEQ, HALF_DIM]), cos_data).expect("cos"),
        ),
        TensorParamBinding::ConstantTensor(
            ArrayD::from_shape_vec(IxDyn(&[SEQ, HALF_DIM]), sin_data).expect("sin"),
        ),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[SEQ, HALF_DIM]), -1.0f32)),
    ]
}

// ---------------------------------------------------------------------------
// 5. Single decoder layer
// ---------------------------------------------------------------------------

/// Single Qwen3 decoder layer.
///
/// Input: `[SEQ, D_MODEL]` (Variable).
/// Output: `[SEQ, D_MODEL]`.
///
/// RMSNorm -> MHA(causal) -> residual -> RMSNorm -> SwiGLU -> residual.
pub(super) fn build_decoder_layer() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("qwen3_text_decoder_layer");
    let shape = [SEQ, D_MODEL];
    let inter = [SEQ, FFN_DIM];

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
    let gate_w = b.add_input("gate_w", &[FFN_DIM, D_MODEL]);
    let up_w = b.add_input("up_w", &[FFN_DIM, D_MODEL]);
    let down_w = b.add_input("down_w", &[D_MODEL, FFN_DIM]);

    let output = decoder_layer_inline(
        &mut b, input, eps, attn_ln_w, q_w, k_w, v_w, o_w, mlp_ln_w, gate_w, up_w, down_w, &shape,
        &inter,
    );
    b.build(output).expect("valid decoder layer graph")
}

pub(super) fn decoder_layer_bindings() -> Vec<TensorParamBinding> {
    let mut bindings = Vec::new();
    bindings.push(TensorParamBinding::Variable); // hidden
    bindings.push(TensorParamBinding::ConstantScalar(1e-5)); // eps
                                                             // attn_ln_w
    bindings.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
        IxDyn(&[D_MODEL]),
        1.0f32,
    )));
    // q, k, v, o
    for _ in 0..4 {
        bindings.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[D_MODEL, D_MODEL]),
            W,
        )));
    }
    // mlp_ln_w
    bindings.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
        IxDyn(&[D_MODEL]),
        1.0f32,
    )));
    // gate_w, up_w
    for _ in 0..2 {
        bindings.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[FFN_DIM, D_MODEL]),
            W,
        )));
    }
    // down_w
    bindings.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
        IxDyn(&[D_MODEL, FFN_DIM]),
        W,
    )));
    bindings
}

// ---------------------------------------------------------------------------
// 6. Two-layer decoder stack
// ---------------------------------------------------------------------------

/// Two Qwen3 decoder layers stacked.
///
/// Input: `[SEQ, D_MODEL]` (Variable).
/// Output: `[SEQ, D_MODEL]`.
pub(super) fn build_two_layer_stack() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("qwen3_text_two_layer_stack");
    let shape = [SEQ, D_MODEL];
    let inter = [SEQ, FFN_DIM];

    let input = b.add_input("hidden", &shape);
    let eps = b.add_input("eps", &[1]);

    let mut current = input;
    for i in 0..2 {
        current = add_decoder_layer_block(&mut b, current, eps, i, &shape, &inter);
    }

    b.build(current).expect("valid 2-layer stack graph")
}

pub(super) fn two_layer_stack_bindings() -> Vec<TensorParamBinding> {
    let mut bindings = Vec::new();
    bindings.push(TensorParamBinding::Variable); // hidden
    bindings.push(TensorParamBinding::ConstantScalar(1e-5)); // eps
    for _ in 0..2 {
        push_layer_bindings(&mut bindings);
    }
    bindings
}

// ---------------------------------------------------------------------------
// 7. LM head (RMSNorm -> Linear -> logits)
// ---------------------------------------------------------------------------

/// LM head: RMSNorm -> Linear projection to vocabulary.
///
/// Input: `[SEQ, D_MODEL]` (Variable).
/// Output: `[SEQ, VOCAB]`.
pub(super) fn build_lm_head() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("qwen3_text_lm_head");
    let shape = [SEQ, D_MODEL];
    let out = [SEQ, VOCAB];

    let input = b.add_input("hidden", &shape);
    let eps = b.add_input("eps", &[1]);
    let norm_w = b.add_input("norm_w", &[D_MODEL]);
    let lm_w = b.add_input("lm_w", &[D_MODEL, VOCAB]);

    let normed = b.add_rms_norm(input, eps, 1, norm_w, &shape);
    let logits = b.add_matmul(normed, lm_w, false, None, &out);
    b.build(logits).expect("valid LM head graph")
}

pub(super) fn lm_head_bindings() -> Vec<TensorParamBinding> {
    vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantScalar(1e-5),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[D_MODEL]), 1.0f32)),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[D_MODEL, VOCAB]), W)),
    ]
}

// ---------------------------------------------------------------------------
// 8. Token generation (LM head + softmax)
// ---------------------------------------------------------------------------

/// Token generation: LM head -> softmax -> probability distribution.
///
/// Input: `[SEQ, D_MODEL]` (Variable).
/// Output: `[SEQ, VOCAB]` (probabilities in [0, 1], summing to 1).
pub(super) fn build_token_generation() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("qwen3_text_token_generation");
    let shape = [SEQ, D_MODEL];
    let out = [SEQ, VOCAB];

    let input = b.add_input("hidden", &shape);
    let eps = b.add_input("eps", &[1]);
    let norm_w = b.add_input("norm_w", &[D_MODEL]);
    let lm_w = b.add_input("lm_w", &[D_MODEL, VOCAB]);

    let normed = b.add_rms_norm(input, eps, 1, norm_w, &shape);
    let logits = b.add_matmul(normed, lm_w, false, None, &out);
    // Softmax along last axis (vocab dimension)
    let probs = b.add_softmax(logits, -1, &out);
    b.build(probs).expect("valid token generation graph")
}

pub(super) fn token_generation_bindings() -> Vec<TensorParamBinding> {
    lm_head_bindings() // Same inputs as LM head
}

// ---------------------------------------------------------------------------
// 9. KV-cache attention (single new token attending to cached context)
// ---------------------------------------------------------------------------

/// KV-cache attention: a single new token queries against cached K/V context.
///
/// Models the autoregressive decoding step where:
/// - Q: `[1, D_MODEL]` (new token)
/// - K_cache: `[SEQ, D_MODEL]` (constant cached keys)
/// - V_cache: `[SEQ, D_MODEL]` (constant cached values)
///
/// This is a simplified single-head version for tractable verification.
/// The new query attends to all cached positions (no causal mask needed
/// since all cached positions precede the new token).
///
/// Output: `[1, D_MODEL]`.
pub(super) fn build_kv_cache_attention() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("qwen3_text_kv_cache_attn");

    let q_shape = [1, HEAD_DIM];
    let kv_shape = [SEQ, HEAD_DIM];
    let out_shape = [1, HEAD_DIM];

    // New token query (Variable), cached K/V (Constant — previously computed)
    let q = b.add_input("q_new", &q_shape);
    let k_cache = b.add_input("k_cache", &kv_shape);
    let v_cache = b.add_input("v_cache", &kv_shape);

    // Scaled dot-product attention: softmax(Q @ K^T / sqrt(d)) @ V
    // Q: [1, HD], K: [SEQ, HD], V: [SEQ, HD] -> out: [1, HD]
    let scale = 1.0 / (HEAD_DIM as f32).sqrt();
    let attn = b.add_attention(
        q,
        k_cache,
        v_cache,
        AttentionMask::Standard,
        Some(scale),
        &out_shape,
    );

    b.build(attn).expect("valid KV-cache attention graph")
}

pub(super) fn kv_cache_bindings() -> Vec<TensorParamBinding> {
    vec![
        TensorParamBinding::Variable, // q_new [1, HEAD_DIM]
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[SEQ, HEAD_DIM]), 0.1f32)),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[SEQ, HEAD_DIM]), 0.1f32)),
    ]
}

// ---------------------------------------------------------------------------
// 10. Full decoder pipeline (Embedding -> 2 layers -> LM head)
// ---------------------------------------------------------------------------

/// Full decoder pipeline: token_emb -> 2 decoder layers -> RMSNorm -> lm_head.
///
/// Input: `[SEQ, D_MODEL]` (Variable — continuous relaxation of token embeddings).
/// Output: `[SEQ, VOCAB]`.
pub(super) fn build_full_pipeline() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("qwen3_text_full_pipeline");
    let shape = [SEQ, D_MODEL];
    let inter = [SEQ, FFN_DIM];
    let out = [SEQ, VOCAB];

    let input = b.add_input("token_emb", &shape);
    let eps = b.add_input("eps", &[1]);

    // 2 decoder layers
    let mut current = input;
    for i in 0..2 {
        current = add_decoder_layer_block(&mut b, current, eps, i, &shape, &inter);
    }

    // Final RMSNorm + lm_head
    let ln_final_w = b.add_input("ln_final_w", &[D_MODEL]);
    let normed = b.add_rms_norm(current, eps, 1, ln_final_w, &shape);
    let lm_w = b.add_input("lm_w", &[D_MODEL, VOCAB]);
    let logits = b.add_matmul(normed, lm_w, false, None, &out);

    b.build(logits).expect("valid full pipeline graph")
}

pub(super) fn full_pipeline_bindings() -> Vec<TensorParamBinding> {
    let mut bindings = Vec::new();
    bindings.push(TensorParamBinding::Variable); // token_emb
    bindings.push(TensorParamBinding::ConstantScalar(1e-5)); // eps
                                                             // 2 decoder layers
    for _ in 0..2 {
        push_layer_bindings(&mut bindings);
    }
    // ln_final_w
    bindings.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
        IxDyn(&[D_MODEL]),
        1.0f32,
    )));
    // lm_w
    bindings.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
        IxDyn(&[D_MODEL, VOCAB]),
        W,
    )));
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

    let x1_cos = b.add_binary_mul(x1, cos, half);
    let x2_neg = b.add_binary_mul(x2, neg_one, half);
    let x2_neg_sin = b.add_binary_mul(x2_neg, sin, half);
    let y1 = b.add_binary_add(x1_cos, x2_neg_sin, half);

    let x1_sin = b.add_binary_mul(x1, sin, half);
    let x2_cos = b.add_binary_mul(x2, cos, half);
    let y2 = b.add_binary_add(x1_sin, x2_cos, half);

    b.add_concat(&[y1, y2], 1, full)
}

/// Build a decoder layer inline (adds weight inputs with layer prefix).
fn decoder_layer_inline(
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
        .expect("decoder layer self-attention");
    let residual1 = b.add_binary_add(input, attn_out, shape);

    // Sub-block 2: RMSNorm -> SwiGLU -> residual
    let mlp_normed = b.add_rms_norm(residual1, eps, 1, mlp_ln_w, shape);
    let mlp_out = swiglu_inline(b, mlp_normed, gate_w, up_w, down_w, shape, inter);
    b.add_binary_add(residual1, mlp_out, shape)
}

/// Add a decoder layer block with auto-named weight inputs.
fn add_decoder_layer_block(
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
    let gate_w = b.add_input(&format!("{pfx}_gate_w"), &[FFN_DIM, D_MODEL]);
    let up_w = b.add_input(&format!("{pfx}_up_w"), &[FFN_DIM, D_MODEL]);
    let down_w = b.add_input(&format!("{pfx}_down_w"), &[D_MODEL, FFN_DIM]);

    decoder_layer_inline(
        b, input, eps, attn_ln_w, q_w, k_w, v_w, o_w, mlp_ln_w, gate_w, up_w, down_w, shape, inter,
    )
}

/// Push bindings for one decoder layer: attn_ln_w + Q/K/V/O + mlp_ln_w + gate/up/down.
fn push_layer_bindings(bindings: &mut Vec<TensorParamBinding>) {
    // attn_ln_w
    bindings.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
        IxDyn(&[D_MODEL]),
        1.0f32,
    )));
    // q, k, v, o
    for _ in 0..4 {
        bindings.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[D_MODEL, D_MODEL]),
            W,
        )));
    }
    // mlp_ln_w
    bindings.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
        IxDyn(&[D_MODEL]),
        1.0f32,
    )));
    // gate_w, up_w
    for _ in 0..2 {
        bindings.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[FFN_DIM, D_MODEL]),
            W,
        )));
    }
    // down_w
    bindings.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
        IxDyn(&[D_MODEL, FFN_DIM]),
        W,
    )));
}

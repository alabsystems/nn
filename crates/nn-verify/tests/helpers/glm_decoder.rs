// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Builder helpers for GLM-4/5 decoder NY composition tests.
//!
//! Architecture: Embedding -> N x (RmsNorm + MHA + residual, RmsNorm + SwiGLU + residual)
//! -> RmsNorm -> Linear (output_layer).
//!
//! Key GLM-4/5 differences from Qwen3 (see `crates/nn-glm5/src/layers.rs`):
//! - Fused QKV projection: single `query_key_value` weight, split after projection
//! - SwiGLU MLP: fused `dense_h_to_4h` (size `ffn_hidden * 2`), narrowed into gate+up
//! - QKV bias (`add_qkv_bias = true` by default)
//! - Partial RoPE (first `head_dim/2` dims rotated) -- skipped for tractability
//! - GQA via `multi_query_group_num` -- simplified to full heads here
//!
//! Simplifications for NY tractability:
//! - RoPE skipped (constant positional embedding approximation)
//! - GQA simplified (num_kv_heads == num_attention_heads)
//! - Fused QKV decomposed into separate Q/K/V linear projections
//!   (NY sees the same data flow; weight fusion is a storage detail)
//! - QKV bias included (GLM default: `add_qkv_bias = true`)
//!
//! Part of #3569: GLM decoder NY compose verification.

use nn_dsl::tensor_block_builder::TensorBlockBuilder;
use nn_dsl::tensor_ir::{TensorKernelDef, TensorNodeId};
use nn_dsl::AttentionMask;
use nn_verify::TensorParamBinding;
use ndarray::{ArrayD, IxDyn};

// ---------------------------------------------------------------------------
// Small-scale dimensions for NY tractability
// ---------------------------------------------------------------------------

/// Sequence length (production: up to 8192 for GLM-4-9B).
pub(super) const SEQ_LEN: usize = 4;

/// Model hidden dimension (production: 4096 for GLM-4-9B).
pub(super) const D_MODEL: usize = 32;

/// Number of attention heads (production: 32 for GLM-4-9B).
pub(super) const N_HEADS: usize = 4;

/// Per-head dimension.
pub(super) const HEAD_DIM: usize = D_MODEL / N_HEADS; // 8

/// FFN intermediate dimension (production: 13696 for GLM-4-9B).
/// GLM SwiGLU uses fused gate+up of size `FFN_DIM * 2`.
pub(super) const FFN_DIM: usize = 128;

/// Vocabulary size (production: 151552 for GLM-4-9B).
pub(super) const VOCAB_SIZE: usize = 16;

/// Number of decoder layers for test model.
const N_LAYERS: usize = 2;

/// Weight magnitude for small-scale test weights.
const WEIGHT_MAG: f32 = 0.001;

// ---------------------------------------------------------------------------
// Self-attention sub-graph builder
// ---------------------------------------------------------------------------

/// Build a GLM self-attention sub-graph.
///
/// Input: `[SEQ_LEN, D_MODEL]`.
/// Output: `[SEQ_LEN, D_MODEL]`.
///
/// GLM attention pipeline:
///   1. RmsNorm(input)
///   2. Q = Linear(normed) + bias   [S, D_MODEL]
///   3. K = Linear(normed) + bias   [S, D_MODEL]
///   4. V = Linear(normed) + bias   [S, D_MODEL]
///   5. Attention(Q, K, V, causal)  [S, D_MODEL]
///   6. out_proj = Linear(attn)     [S, D_MODEL]
///   7. residual = input + out_proj [S, D_MODEL]
///
/// Note: In production GLM, Q/K/V are fused into one projection then split.
/// For NY verification, separate projections are equivalent in data flow.
pub(super) fn build_glm_self_attention() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("glm_self_attention");

    let shape = [SEQ_LEN, D_MODEL];

    // Inputs
    let input = b.add_input("hidden", &shape);
    let eps = b.add_input("eps", &[1]);
    let ln_w = b.add_input("input_layernorm_w", &[D_MODEL]);

    // Q/K/V projection weights + biases (GLM has QKV bias)
    let q_w = b.add_input("q_w", &[D_MODEL, D_MODEL]);
    let q_b = b.add_input("q_b", &[D_MODEL]);
    let k_w = b.add_input("k_w", &[D_MODEL, D_MODEL]);
    let k_b = b.add_input("k_b", &[D_MODEL]);
    let v_w = b.add_input("v_w", &[D_MODEL, D_MODEL]);
    let v_b = b.add_input("v_b", &[D_MODEL]);
    let o_w = b.add_input("o_w", &[D_MODEL, D_MODEL]);

    // 1. Pre-norm
    let normed = b.add_rms_norm(input, eps, 1, ln_w, &shape);

    // 2-4. Q/K/V projections with bias
    let q = b.add_linear(normed, q_w, Some(q_b), &shape);
    let k = b.add_linear(normed, k_w, Some(k_b), &shape);
    let v = b.add_linear(normed, v_w, Some(v_b), &shape);

    // 5. Multi-head causal attention
    let attn = b.add_attention(
        q,
        k,
        v,
        AttentionMask::Causal,
        Some(1.0 / (HEAD_DIM as f32).sqrt()),
        &shape,
    );

    // 6. Output projection
    let out_proj = b.add_linear(attn, o_w, None, &shape);

    // 7. Residual connection
    let output = b.add_binary_add(input, out_proj, &shape);

    b.build(output).expect("valid GLM self-attention sub-graph")
}

/// Bindings for GLM self-attention: hidden=Variable, rest=Constant.
pub(super) fn glm_self_attention_bindings() -> Vec<TensorParamBinding> {
    let mut bindings = Vec::new();

    // hidden: Variable [SEQ_LEN, D_MODEL]
    bindings.push(TensorParamBinding::Variable);

    // eps: scalar constant
    bindings.push(TensorParamBinding::ConstantScalar(1e-5));

    // input_layernorm weight [D_MODEL]
    bindings.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
        IxDyn(&[D_MODEL]),
        1.0f32,
    )));

    // Q weight + bias, K weight + bias, V weight + bias
    for _ in 0..3 {
        bindings.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[D_MODEL, D_MODEL]),
            WEIGHT_MAG,
        )));
        bindings.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[D_MODEL]),
            0.0f32,
        )));
    }

    // Output projection weight [D_MODEL, D_MODEL]
    bindings.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
        IxDyn(&[D_MODEL, D_MODEL]),
        WEIGHT_MAG,
    )));

    bindings
}

// ---------------------------------------------------------------------------
// SwiGLU MLP sub-graph builder (GLM-style fused gate+up)
// ---------------------------------------------------------------------------

/// Build a GLM SwiGLU MLP sub-graph.
///
/// Input: `[SEQ_LEN, D_MODEL]`.
/// Output: `[SEQ_LEN, D_MODEL]`.
///
/// GLM SwiGLU MLP pipeline (mirrors `Glm5MLP::forward`):
///   1. intermediate = dense_h_to_4h(x)  [S, FFN_DIM * 2]
///   2. gate = narrow(intermediate, -1, 0, FFN_DIM)    [S, FFN_DIM]
///   3. up   = narrow(intermediate, -1, FFN_DIM, FFN_DIM) [S, FFN_DIM]
///   4. gate_sig = sigmoid(gate)          [S, FFN_DIM]
///   5. gate_silu = gate * gate_sig       [S, FFN_DIM]  -- SiLU
///   6. gated = gate_silu * up            [S, FFN_DIM]
///   7. out = dense_4h_to_h(gated)        [S, D_MODEL]
pub(super) fn build_glm_swiglu_ffn() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("glm_swiglu_ffn");

    let in_shape = [SEQ_LEN, D_MODEL];
    let fused_shape = [SEQ_LEN, FFN_DIM * 2];
    let inter_shape = [SEQ_LEN, FFN_DIM];

    // Inputs
    let input = b.add_input("hidden", &in_shape);
    let h_to_4h_w = b.add_input("dense_h_to_4h_w", &[FFN_DIM * 2, D_MODEL]);
    let h4_to_h_w = b.add_input("dense_4h_to_h_w", &[D_MODEL, FFN_DIM]);

    // 1. Fused projection to FFN_DIM * 2
    let intermediate = b.add_linear(input, h_to_4h_w, None, &fused_shape);

    // 2-3. Split into gate and up via narrow
    let gate = b.add_narrow(intermediate, 1, 0, FFN_DIM, &inter_shape);
    let up = b.add_narrow(intermediate, 1, FFN_DIM, FFN_DIM, &inter_shape);

    // 4. sigmoid(gate)
    let gate_sig = b.add_sigmoid(gate, &inter_shape);

    // 5. SiLU: gate * sigmoid(gate)
    let gate_silu = b.add_binary_mul(gate, gate_sig, &inter_shape);

    // 6. gated = gate_silu * up
    let gated = b.add_binary_mul(gate_silu, up, &inter_shape);

    // 7. down projection
    let out = b.add_linear(gated, h4_to_h_w, None, &in_shape);

    b.build(out).expect("valid GLM SwiGLU FFN sub-graph")
}

/// Bindings for GLM SwiGLU FFN: hidden=Variable, weights=Constant.
pub(super) fn glm_swiglu_ffn_bindings() -> Vec<TensorParamBinding> {
    vec![
        TensorParamBinding::Variable, // hidden [SEQ_LEN, D_MODEL]
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[FFN_DIM * 2, D_MODEL]),
            WEIGHT_MAG,
        )), // dense_h_to_4h_w
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[D_MODEL, FFN_DIM]),
            WEIGHT_MAG,
        )), // dense_4h_to_h_w
    ]
}

// ---------------------------------------------------------------------------
// Full decoder block (attention + FFN + residuals)
// ---------------------------------------------------------------------------

/// Weights for a single GLM decoder block.
struct GlmBlockWeights {
    // Attention sub-block
    attn_ln_w: TensorNodeId,
    q_w: TensorNodeId,
    q_b: TensorNodeId,
    k_w: TensorNodeId,
    k_b: TensorNodeId,
    v_w: TensorNodeId,
    v_b: TensorNodeId,
    o_w: TensorNodeId,
    // MLP sub-block
    mlp_ln_w: TensorNodeId,
    h_to_4h_w: TensorNodeId,
    h4_to_h_w: TensorNodeId,
    // Shared eps
    eps: TensorNodeId,
}

fn add_block_weights(
    b: &mut TensorBlockBuilder,
    layer_idx: usize,
    eps: TensorNodeId,
) -> GlmBlockWeights {
    let pfx = format!("layer{layer_idx}");
    GlmBlockWeights {
        attn_ln_w: b.add_input(&format!("{pfx}_attn_ln_w"), &[D_MODEL]),
        q_w: b.add_input(&format!("{pfx}_qw"), &[D_MODEL, D_MODEL]),
        q_b: b.add_input(&format!("{pfx}_qb"), &[D_MODEL]),
        k_w: b.add_input(&format!("{pfx}_kw"), &[D_MODEL, D_MODEL]),
        k_b: b.add_input(&format!("{pfx}_kb"), &[D_MODEL]),
        v_w: b.add_input(&format!("{pfx}_vw"), &[D_MODEL, D_MODEL]),
        v_b: b.add_input(&format!("{pfx}_vb"), &[D_MODEL]),
        o_w: b.add_input(&format!("{pfx}_ow"), &[D_MODEL, D_MODEL]),
        mlp_ln_w: b.add_input(&format!("{pfx}_mlp_ln_w"), &[D_MODEL]),
        h_to_4h_w: b.add_input(&format!("{pfx}_h_to_4h_w"), &[FFN_DIM * 2, D_MODEL]),
        h4_to_h_w: b.add_input(&format!("{pfx}_4h_to_h_w"), &[D_MODEL, FFN_DIM]),
        eps,
    }
}

/// Build a GLM SwiGLU FFN within a decoder block using pre-allocated weights.
fn build_fused_swiglu(
    b: &mut TensorBlockBuilder,
    input: TensorNodeId,
    h_to_4h_w: TensorNodeId,
    h4_to_h_w: TensorNodeId,
) -> TensorNodeId {
    let in_shape = [SEQ_LEN, D_MODEL];
    let fused_shape = [SEQ_LEN, FFN_DIM * 2];
    let inter_shape = [SEQ_LEN, FFN_DIM];

    let intermediate = b.add_linear(input, h_to_4h_w, None, &fused_shape);
    let gate = b.add_narrow(intermediate, 1, 0, FFN_DIM, &inter_shape);
    let up = b.add_narrow(intermediate, 1, FFN_DIM, FFN_DIM, &inter_shape);
    let gate_sig = b.add_sigmoid(gate, &inter_shape);
    let gate_silu = b.add_binary_mul(gate, gate_sig, &inter_shape);
    let gated = b.add_binary_mul(gate_silu, up, &inter_shape);
    b.add_linear(gated, h4_to_h_w, None, &in_shape)
}

/// Build a single GLM decoder block.
///
/// Pre-norm structure (matches `Glm5DecoderLayer::forward`):
/// 1. RmsNorm -> Q/K/V(+bias) -> causal attention -> out_proj -> + residual
/// 2. RmsNorm -> SwiGLU MLP (fused gate+up, narrow, silu*up, down) -> + residual
fn build_decoder_block(
    b: &mut TensorBlockBuilder,
    input: TensorNodeId,
    w: &GlmBlockWeights,
) -> TensorNodeId {
    let shape = [SEQ_LEN, D_MODEL];

    // --- Sub-block 1: Causal self-attention with QKV bias ---
    let attn_normed = b.add_rms_norm(input, w.eps, 1, w.attn_ln_w, &shape);

    let q = b.add_linear(attn_normed, w.q_w, Some(w.q_b), &shape);
    let k = b.add_linear(attn_normed, w.k_w, Some(w.k_b), &shape);
    let v = b.add_linear(attn_normed, w.v_w, Some(w.v_b), &shape);

    let attn_out = b.add_attention(
        q,
        k,
        v,
        AttentionMask::Causal,
        Some(1.0 / (HEAD_DIM as f32).sqrt()),
        &shape,
    );
    let out_proj = b.add_linear(attn_out, w.o_w, None, &shape);
    let residual1 = b.add_binary_add(input, out_proj, &shape);

    // --- Sub-block 2: SwiGLU MLP (fused gate+up, narrow, silu) ---
    let mlp_normed = b.add_rms_norm(residual1, w.eps, 1, w.mlp_ln_w, &shape);
    let mlp_out = build_fused_swiglu(b, mlp_normed, w.h_to_4h_w, w.h4_to_h_w);
    b.add_binary_add(residual1, mlp_out, &shape)
}

/// Build a single GLM decoder block as a standalone `TensorKernelDef`.
///
/// Used for focused per-block verification.
pub(super) fn build_glm_decoder_block() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("glm_decoder_block");

    let input = b.add_input("hidden", &[SEQ_LEN, D_MODEL]);
    let eps = b.add_input("eps", &[1]);
    let w = add_block_weights(&mut b, 0, eps);
    let output = build_decoder_block(&mut b, input, &w);

    b.build(output).expect("valid GLM decoder block sub-graph")
}

/// Bindings for a single GLM decoder block: hidden=Variable, rest=Constant.
#[allow(clippy::vec_init_then_push)]
pub(super) fn glm_decoder_block_bindings() -> Vec<TensorParamBinding> {
    let mut bindings = Vec::new();

    // hidden: Variable [SEQ_LEN, D_MODEL]
    bindings.push(TensorParamBinding::Variable);

    // eps: scalar constant
    bindings.push(TensorParamBinding::ConstantScalar(1e-5));

    // Per-block weights:
    //   attn_ln_w [D_MODEL]
    bindings.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
        IxDyn(&[D_MODEL]),
        1.0f32,
    )));
    //   q_w, q_b, k_w, k_b, v_w, v_b
    for _ in 0..3 {
        bindings.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[D_MODEL, D_MODEL]),
            WEIGHT_MAG,
        )));
        bindings.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[D_MODEL]),
            0.0f32,
        )));
    }
    //   o_w [D_MODEL, D_MODEL]
    bindings.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
        IxDyn(&[D_MODEL, D_MODEL]),
        WEIGHT_MAG,
    )));
    //   mlp_ln_w [D_MODEL]
    bindings.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
        IxDyn(&[D_MODEL]),
        1.0f32,
    )));
    //   h_to_4h_w [FFN_DIM * 2, D_MODEL]
    bindings.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
        IxDyn(&[FFN_DIM * 2, D_MODEL]),
        WEIGHT_MAG,
    )));
    //   4h_to_h_w [D_MODEL, FFN_DIM]
    bindings.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
        IxDyn(&[D_MODEL, FFN_DIM]),
        WEIGHT_MAG,
    )));

    bindings
}

// ---------------------------------------------------------------------------
// 2-block decoder stack
// ---------------------------------------------------------------------------

/// Build a 2-block GLM decoder stack as a `TensorKernelDef`.
///
/// Architecture: hidden -> 2 x DecoderBlock -> RmsNorm -> lm_head.
pub(super) fn build_glm_decoder_stack() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("glm_decoder_stack");

    let input = b.add_input("hidden", &[SEQ_LEN, D_MODEL]);
    let eps = b.add_input("eps", &[1]);

    // 2 decoder blocks
    let mut block_weights = Vec::new();
    for i in 0..N_LAYERS {
        block_weights.push(add_block_weights(&mut b, i, eps));
    }
    let mut current = input;
    for w in &block_weights {
        current = build_decoder_block(&mut b, current, w);
    }

    // Final RmsNorm
    let final_ln_w = b.add_input("ln_final_w", &[D_MODEL]);
    let normed = b.add_rms_norm(current, eps, 1, final_ln_w, &[SEQ_LEN, D_MODEL]);

    // Output projection (lm_head)
    let lm_head_w = b.add_input("lm_head_w", &[D_MODEL, VOCAB_SIZE]);
    let logits = b.add_matmul(normed, lm_head_w, false, None, &[SEQ_LEN, VOCAB_SIZE]);

    b.build(logits).expect("valid GLM decoder stack sub-graph")
}

/// Bindings for 2-block GLM decoder stack: hidden=Variable, rest=Constant.
#[allow(clippy::vec_init_then_push)]
pub(super) fn glm_decoder_stack_bindings() -> Vec<TensorParamBinding> {
    let mut bindings = Vec::new();

    // hidden: Variable [SEQ_LEN, D_MODEL]
    bindings.push(TensorParamBinding::Variable);

    // eps: scalar constant
    bindings.push(TensorParamBinding::ConstantScalar(1e-5));

    // N_LAYERS decoder blocks
    for _ in 0..N_LAYERS {
        // attn_ln_w [D_MODEL]
        bindings.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[D_MODEL]),
            1.0f32,
        )));
        // q_w + q_b, k_w + k_b, v_w + v_b
        for _ in 0..3 {
            bindings.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
                IxDyn(&[D_MODEL, D_MODEL]),
                WEIGHT_MAG,
            )));
            bindings.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
                IxDyn(&[D_MODEL]),
                0.0f32,
            )));
        }
        // o_w [D_MODEL, D_MODEL]
        bindings.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[D_MODEL, D_MODEL]),
            WEIGHT_MAG,
        )));
        // mlp_ln_w [D_MODEL]
        bindings.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[D_MODEL]),
            1.0f32,
        )));
        // h_to_4h_w [FFN_DIM * 2, D_MODEL]
        bindings.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[FFN_DIM * 2, D_MODEL]),
            WEIGHT_MAG,
        )));
        // 4h_to_h_w [D_MODEL, FFN_DIM]
        bindings.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[D_MODEL, FFN_DIM]),
            WEIGHT_MAG,
        )));
    }

    // Final RmsNorm weight
    bindings.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
        IxDyn(&[D_MODEL]),
        1.0f32,
    )));

    // lm_head weight [D_MODEL, VOCAB_SIZE]
    bindings.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
        IxDyn(&[D_MODEL, VOCAB_SIZE]),
        WEIGHT_MAG,
    )));

    bindings
}

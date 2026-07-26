// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Builder helpers for Qwen3 RoPE + GQA NY composition tests.
//!
//! Provides focused sub-graph builders for components that the existing
//! `qwen3_decoder.rs` explicitly skips for tractability:
//!
//! - **RoPE (Rotary Position Embeddings)**: half-split rotation decomposed as
//!   `y1 = x1*cos - x2*sin`, `y2 = x1*sin + x2*cos`, concatenated.
//! - **GQA (Grouped Query Attention)**: KV head expansion via reshape to
//!   match Q heads before attention.
//! - **Combined RoPE + scaled dot-product attention**: end-to-end single-head
//!   attention with positional encoding applied.
//! - **SwiGLU MLP**: `down_proj(silu(gate_proj(x)) * up_proj(x))` isolated
//!   from the full decoder block for focused verification.
//!
//! Part of #3560: Qwen3 RoPE + GQA NY compose verification.

use nn_dsl::tensor_block_builder::TensorBlockBuilder;
use nn_dsl::tensor_ir::TensorKernelDef;
use nn_verify::TensorParamBinding;
use ndarray::{ArrayD, IxDyn};

// ---------------------------------------------------------------------------
// Dimensions — small for fast verification, structurally representative
// ---------------------------------------------------------------------------

/// Sequence length.
pub(super) const SEQ_LEN: usize = 4;
/// Full hidden dimension.
pub(super) const HIDDEN_DIM: usize = 32;
/// Number of Q attention heads.
pub(super) const NUM_HEADS: usize = 4;
/// Number of KV attention heads (GQA: fewer than Q heads).
pub(super) const NUM_KV_HEADS: usize = 2;
/// Per-head dimension.
pub(super) const HEAD_DIM: usize = HIDDEN_DIM / NUM_HEADS; // 8
/// Half of head dimension (used in RoPE half-split).
pub(super) const HALF_DIM: usize = HEAD_DIM / 2; // 4
/// GQA repetition factor: each KV head serves this many Q heads.
pub(super) const GQA_REP: usize = NUM_HEADS / NUM_KV_HEADS; // 2
/// FFN intermediate dimension (SwiGLU).
pub(super) const INTERMEDIATE: usize = 64;
/// Weight magnitude for small-scale test weights.
const WEIGHT_MAG: f32 = 0.02;

// ---------------------------------------------------------------------------
// 1. RoPE (Rotary Position Embeddings) half-split sub-graph
// ---------------------------------------------------------------------------

/// Build a RoPE half-split rotation sub-graph.
///
/// Input: `[SEQ_LEN, HEAD_DIM]` (one head's Q or K vector).
/// cos, sin: `[SEQ_LEN, HALF_DIM]` constant positional embeddings.
///
/// Decomposition (matching HuggingFace `rotate_half`):
///   x1 = narrow(x, axis=1, start=0, length=HALF_DIM)   -> [S, HALF_DIM]
///   x2 = narrow(x, axis=1, start=HALF_DIM, length=HALF_DIM) -> [S, HALF_DIM]
///   y1 = x1 * cos + (x2 * neg_one) * sin               -> [S, HALF_DIM]
///       = x1 * cos - x2 * sin
///   y2 = x1 * sin + x2 * cos                            -> [S, HALF_DIM]
///   output = concat([y1, y2], axis=1)                    -> [S, HEAD_DIM]
pub(super) fn build_rope_half_split() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("qwen3_rope_half_split");

    let half_shape = [SEQ_LEN, HALF_DIM];
    let full_shape = [SEQ_LEN, HEAD_DIM];

    // Inputs
    let input = b.add_input("q_head", &full_shape);
    let cos = b.add_input("cos", &half_shape);
    let sin = b.add_input("sin", &half_shape);
    let neg_one = b.add_input("neg_one", &half_shape); // constant -1.0

    // Split input into halves along last dim
    let x1 = b.add_narrow(input, 1, 0, HALF_DIM, &half_shape);
    let x2 = b.add_narrow(input, 1, HALF_DIM, HALF_DIM, &half_shape);

    // y1 = x1 * cos - x2 * sin
    //    = x1 * cos + (x2 * neg_one) * sin
    let x1_cos = b.add_binary_mul(x1, cos, &half_shape);
    let x2_neg = b.add_binary_mul(x2, neg_one, &half_shape);
    let x2_neg_sin = b.add_binary_mul(x2_neg, sin, &half_shape);
    let y1 = b.add_binary_add(x1_cos, x2_neg_sin, &half_shape);

    // y2 = x1 * sin + x2 * cos
    let x1_sin = b.add_binary_mul(x1, sin, &half_shape);
    let x2_cos = b.add_binary_mul(x2, cos, &half_shape);
    let y2 = b.add_binary_add(x1_sin, x2_cos, &half_shape);

    // Concatenate halves back
    let output = b.add_concat(&[y1, y2], 1, &full_shape);

    b.build(output).expect("valid RoPE half-split sub-graph")
}

/// Bindings for RoPE half-split: input=Variable, cos/sin/neg_one=Constant.
pub(super) fn rope_bindings() -> Vec<TensorParamBinding> {
    let half_shape = &[SEQ_LEN, HALF_DIM];

    // Precompute cos/sin for positions 0..SEQ_LEN with head_dim frequencies.
    let mut cos_data = vec![0.0f32; SEQ_LEN * HALF_DIM];
    let mut sin_data = vec![0.0f32; SEQ_LEN * HALF_DIM];
    for t in 0..SEQ_LEN {
        for i in 0..HALF_DIM {
            let freq = (t as f64) / 10000.0_f64.powf(2.0 * i as f64 / HEAD_DIM as f64);
            cos_data[t * HALF_DIM + i] = freq.cos() as f32;
            sin_data[t * HALF_DIM + i] = freq.sin() as f32;
        }
    }

    vec![
        TensorParamBinding::Variable, // q_head [SEQ_LEN, HEAD_DIM]
        TensorParamBinding::ConstantTensor(
            ArrayD::from_shape_vec(IxDyn(half_shape), cos_data).expect("cos"),
        ),
        TensorParamBinding::ConstantTensor(
            ArrayD::from_shape_vec(IxDyn(half_shape), sin_data).expect("sin"),
        ),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(half_shape), -1.0f32)),
    ]
}

// ---------------------------------------------------------------------------
// 2. GQA (Grouped Query Attention) KV expansion sub-graph
// ---------------------------------------------------------------------------

/// Build a GQA KV expansion sub-graph.
///
/// Input: `[NUM_KV_HEADS, SEQ_LEN, HEAD_DIM]` (KV tensor with fewer heads).
/// Output: `[NUM_HEADS, SEQ_LEN, HEAD_DIM]` (expanded to match Q heads).
///
/// Decomposition of `repeat_kv`:
///   reshape [KV_HEADS, SEQ, HD] -> [KV_HEADS, 1, SEQ, HD]  (unsqueeze)
///   ...but TensorBlockBuilder doesn't have expand/repeat. So we decompose
///   differently: reshape to [KV_HEADS, SEQ * HD] -> tile via concat ->
///   reshape to [NUM_HEADS, SEQ, HD].
///
/// Simplified approach for NY: explicit concat of repeated heads.
///   For each KV head, repeat it GQA_REP times via Concat along axis 0.
///   This is mathematically equivalent to repeat_kv.
///
/// Actually, the simplest NY-friendly decomposition:
///   Narrow each KV head -> concat GQA_REP copies -> produces expanded heads.
///   Input: [NUM_KV_HEADS * SEQ_LEN, HEAD_DIM] (flattened head dim)
///   Each KV head block is [SEQ_LEN, HEAD_DIM].
///   Repeat each block GQA_REP times -> [NUM_HEADS * SEQ_LEN, HEAD_DIM].
///   Reshape -> [NUM_HEADS, SEQ_LEN, HEAD_DIM].
pub(super) fn build_gqa_expand() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("qwen3_gqa_expand");

    let kv_flat_shape = [NUM_KV_HEADS * SEQ_LEN, HEAD_DIM];
    let head_block = [SEQ_LEN, HEAD_DIM];
    let q_flat_shape = [NUM_HEADS * SEQ_LEN, HEAD_DIM];
    let output_shape = [NUM_HEADS, SEQ_LEN, HEAD_DIM];

    // Input: flattened KV heads [KV_HEADS * SEQ, HD]
    let input = b.add_input("kv_heads", &kv_flat_shape);

    // Extract each KV head and repeat GQA_REP times
    let mut expanded_blocks = Vec::new();
    for kv_idx in 0..NUM_KV_HEADS {
        let start = kv_idx * SEQ_LEN;
        let head_slice = b.add_narrow(input, 0, start, SEQ_LEN, &head_block);
        // Repeat this head GQA_REP times
        for _ in 0..GQA_REP {
            expanded_blocks.push(head_slice);
        }
    }

    // Concat all expanded head blocks along axis 0
    let expanded = b.add_concat(&expanded_blocks, 0, &q_flat_shape);

    // Reshape to [NUM_HEADS, SEQ_LEN, HEAD_DIM]
    let output = b.add_reshape(expanded, &output_shape);

    b.build(output).expect("valid GQA expand sub-graph")
}

/// Bindings for GQA expansion: input=Variable.
pub(super) fn gqa_expand_bindings() -> Vec<TensorParamBinding> {
    vec![TensorParamBinding::Variable] // kv_heads [KV_HEADS * SEQ, HD]
}

// ---------------------------------------------------------------------------
// 3. Combined RoPE + single-head attention sub-graph
// ---------------------------------------------------------------------------

/// Build a combined RoPE + attention sub-graph for one head.
///
/// Input: `[SEQ_LEN, HEAD_DIM]` (one attention head's Q/K/V pre-projection).
///
/// Pipeline:
///   1. RoPE on Q: narrow → rotate → concat
///   2. RoPE on K: narrow → rotate → concat (shares cos/sin)
///   3. Scaled dot-product: attention(Q_rot, K_rot, V) with scale=1/sqrt(HD)
///
/// Output: `[SEQ_LEN, HEAD_DIM]`.
pub(super) fn build_rope_attention_head() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("qwen3_rope_attention_head");

    let full_shape = [SEQ_LEN, HEAD_DIM];
    let half_shape = [SEQ_LEN, HALF_DIM];

    // Inputs: Q, K, V for one head + positional cos/sin
    let q = b.add_input("q_head", &full_shape);
    let k = b.add_input("k_head", &full_shape);
    let v = b.add_input("v_head", &full_shape);
    let cos = b.add_input("cos", &half_shape);
    let sin = b.add_input("sin", &half_shape);
    let neg_one = b.add_input("neg_one", &half_shape);

    // Apply RoPE to Q
    let q_rot = apply_rope_inline(&mut b, q, cos, sin, neg_one, &full_shape, &half_shape);

    // Apply RoPE to K
    let k_rot = apply_rope_inline(&mut b, k, cos, sin, neg_one, &full_shape, &half_shape);

    // Scaled dot-product attention: softmax(Q_rot @ K_rot^T / sqrt(HD)) @ V
    let scale = 1.0 / (HEAD_DIM as f32).sqrt();
    let attn = b.add_attention(
        q_rot,
        k_rot,
        v,
        nn_dsl::AttentionMask::Causal,
        Some(scale),
        &full_shape,
    );

    b.build(attn)
        .expect("valid RoPE + attention head sub-graph")
}

/// Helper: inline RoPE rotation (reuses cos/sin/neg_one nodes).
fn apply_rope_inline(
    b: &mut TensorBlockBuilder,
    input: nn_dsl::tensor_ir::TensorNodeId,
    cos: nn_dsl::tensor_ir::TensorNodeId,
    sin: nn_dsl::tensor_ir::TensorNodeId,
    neg_one: nn_dsl::tensor_ir::TensorNodeId,
    full_shape: &[usize],
    half_shape: &[usize],
) -> nn_dsl::tensor_ir::TensorNodeId {
    let x1 = b.add_narrow(input, 1, 0, HALF_DIM, half_shape);
    let x2 = b.add_narrow(input, 1, HALF_DIM, HALF_DIM, half_shape);

    let x1_cos = b.add_binary_mul(x1, cos, half_shape);
    let x2_neg = b.add_binary_mul(x2, neg_one, half_shape);
    let x2_neg_sin = b.add_binary_mul(x2_neg, sin, half_shape);
    let y1 = b.add_binary_add(x1_cos, x2_neg_sin, half_shape);

    let x1_sin = b.add_binary_mul(x1, sin, half_shape);
    let x2_cos = b.add_binary_mul(x2, cos, half_shape);
    let y2 = b.add_binary_add(x1_sin, x2_cos, half_shape);

    b.add_concat(&[y1, y2], 1, full_shape)
}

/// Bindings for RoPE + attention head: q,k,v=Variable, cos/sin/neg_one=Constant.
pub(super) fn rope_attention_bindings() -> Vec<TensorParamBinding> {
    let half_shape = &[SEQ_LEN, HALF_DIM];

    let mut cos_data = vec![0.0f32; SEQ_LEN * HALF_DIM];
    let mut sin_data = vec![0.0f32; SEQ_LEN * HALF_DIM];
    for t in 0..SEQ_LEN {
        for i in 0..HALF_DIM {
            let freq = (t as f64) / 10000.0_f64.powf(2.0 * i as f64 / HEAD_DIM as f64);
            cos_data[t * HALF_DIM + i] = freq.cos() as f32;
            sin_data[t * HALF_DIM + i] = freq.sin() as f32;
        }
    }

    vec![
        TensorParamBinding::Variable, // q_head
        TensorParamBinding::Variable, // k_head
        TensorParamBinding::Variable, // v_head
        TensorParamBinding::ConstantTensor(
            ArrayD::from_shape_vec(IxDyn(half_shape), cos_data).expect("cos"),
        ),
        TensorParamBinding::ConstantTensor(
            ArrayD::from_shape_vec(IxDyn(half_shape), sin_data).expect("sin"),
        ),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(half_shape), -1.0f32)),
    ]
}

// ---------------------------------------------------------------------------
// 4. SwiGLU MLP sub-graph (isolated from decoder block)
// ---------------------------------------------------------------------------

/// Build a SwiGLU MLP sub-graph.
///
/// Input: `[SEQ_LEN, HIDDEN_DIM]`.
/// Output: `[SEQ_LEN, HIDDEN_DIM]`.
///
/// SwiGLU(x) = down_proj(silu(gate_proj(x)) * up_proj(x))
/// SiLU(x) = x * sigmoid(x)
///
/// Decomposition:
///   gate = gate_proj(x)           [S, INTERMEDIATE]
///   gate_sig = sigmoid(gate)      [S, INTERMEDIATE]
///   gate_silu = gate * gate_sig   [S, INTERMEDIATE]   -- SiLU
///   up = up_proj(x)               [S, INTERMEDIATE]
///   gated = gate_silu * up        [S, INTERMEDIATE]   -- gated activation
///   out = down_proj(gated)        [S, HIDDEN_DIM]
pub(super) fn build_swiglu_mlp() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("qwen3_swiglu_mlp");

    let in_shape = [SEQ_LEN, HIDDEN_DIM];
    let inter_shape = [SEQ_LEN, INTERMEDIATE];

    // Inputs
    let input = b.add_input("hidden", &in_shape);
    let gate_w = b.add_input("gate_proj_w", &[INTERMEDIATE, HIDDEN_DIM]);
    let up_w = b.add_input("up_proj_w", &[INTERMEDIATE, HIDDEN_DIM]);
    let down_w = b.add_input("down_proj_w", &[HIDDEN_DIM, INTERMEDIATE]);

    // gate_proj(x) -> [S, INTERMEDIATE]
    let gate = b.add_linear(input, gate_w, None, &inter_shape);

    // sigmoid(gate) -> [S, INTERMEDIATE]
    let gate_sig = b.add_sigmoid(gate, &inter_shape);

    // SiLU: gate * sigmoid(gate) -> [S, INTERMEDIATE]
    let gate_silu = b.add_binary_mul(gate, gate_sig, &inter_shape);

    // up_proj(x) -> [S, INTERMEDIATE]
    let up = b.add_linear(input, up_w, None, &inter_shape);

    // gate_silu * up -> [S, INTERMEDIATE]
    let gated = b.add_binary_mul(gate_silu, up, &inter_shape);

    // down_proj(gated) -> [S, HIDDEN_DIM]
    let out = b.add_linear(gated, down_w, None, &in_shape);

    b.build(out).expect("valid SwiGLU MLP sub-graph")
}

/// Bindings for SwiGLU MLP: hidden=Variable, weights=Constant.
pub(super) fn swiglu_bindings() -> Vec<TensorParamBinding> {
    vec![
        TensorParamBinding::Variable, // hidden [SEQ_LEN, HIDDEN_DIM]
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[INTERMEDIATE, HIDDEN_DIM]),
            WEIGHT_MAG,
        )), // gate_proj_w
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[INTERMEDIATE, HIDDEN_DIM]),
            WEIGHT_MAG,
        )), // up_proj_w
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[HIDDEN_DIM, INTERMEDIATE]),
            WEIGHT_MAG,
        )), // down_proj_w
    ]
}

// ---------------------------------------------------------------------------
// 5. Q/K/V linear projection sub-graph
// ---------------------------------------------------------------------------

/// Build isolated Q/K/V linear projection sub-graph.
///
/// Input: `[SEQ_LEN, HIDDEN_DIM]` (Variable).
/// Output: `[SEQ_LEN, HIDDEN_DIM]` (concatenated Q, K, V projections summed —
///   simplified to a single linear projection for bounds verification).
///
/// Verifies that linear projection preserves IBP bounds: if input is in
/// `[-r, r]`, output is bounded by `||W|| * r + |b|`.
pub(super) fn build_qkv_projection() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("qwen3_qkv_projection");

    let shape = [SEQ_LEN, HIDDEN_DIM];

    let input = b.add_input("hidden", &shape);
    let q_w = b.add_input("q_proj_w", &[HIDDEN_DIM, HIDDEN_DIM]);
    let k_w = b.add_input("k_proj_w", &[HIDDEN_DIM, HIDDEN_DIM]);
    let v_w = b.add_input("v_proj_w", &[HIDDEN_DIM, HIDDEN_DIM]);

    // Q, K, V projections: each [SEQ_LEN, HIDDEN_DIM]
    let q = b.add_linear(input, q_w, None, &shape);
    let k = b.add_linear(input, k_w, None, &shape);
    let v = b.add_linear(input, v_w, None, &shape);

    // Combine Q + K + V to produce a single output for bounds analysis.
    // This verifies all three projection paths contribute bounded outputs.
    let qk = b.add_binary_add(q, k, &shape);
    let output = b.add_binary_add(qk, v, &shape);

    b.build(output).expect("valid Q/K/V projection sub-graph")
}

/// Bindings for Q/K/V projection: input=Variable, weights=Constant.
pub(super) fn qkv_projection_bindings() -> Vec<TensorParamBinding> {
    vec![
        TensorParamBinding::Variable, // hidden [SEQ_LEN, HIDDEN_DIM]
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[HIDDEN_DIM, HIDDEN_DIM]),
            WEIGHT_MAG,
        )), // q_proj_w
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[HIDDEN_DIM, HIDDEN_DIM]),
            WEIGHT_MAG,
        )), // k_proj_w
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[HIDDEN_DIM, HIDDEN_DIM]),
            WEIGHT_MAG,
        )), // v_proj_w
    ]
}

// ---------------------------------------------------------------------------
// 6. Attention score computation: Q @ K^T / sqrt(d) -> softmax
// ---------------------------------------------------------------------------

/// Build attention score sub-graph: scaled dot-product with softmax.
///
/// Input: Q `[SEQ_LEN, HEAD_DIM]` (Variable), K `[SEQ_LEN, HEAD_DIM]` (Variable).
/// Output: `[SEQ_LEN, SEQ_LEN]` (attention weights after softmax, bounded [0, 1]).
///
/// Decomposes: scores = softmax(Q @ K^T / sqrt(HEAD_DIM))
/// Key property: softmax output is always in [0, 1] and rows sum to 1.
pub(super) fn build_attention_scores() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("qwen3_attention_scores");

    let qk_shape = [SEQ_LEN, HEAD_DIM];
    let score_shape = [SEQ_LEN, SEQ_LEN];

    let q = b.add_input("q", &qk_shape);
    let k = b.add_input("k", &qk_shape);

    // Scaled dot-product: Q @ K^T / sqrt(HEAD_DIM)
    let scale = 1.0 / (HEAD_DIM as f32).sqrt();
    let scores = b.add_matmul(q, k, true, Some(scale), &score_shape);

    // Softmax along last axis: produces attention weights in [0, 1]
    let attn_weights = b.add_softmax(scores, 1, &score_shape);

    b.build(attn_weights)
        .expect("valid attention scores sub-graph")
}

/// Bindings for attention scores: Q=Variable, K=Variable.
pub(super) fn attention_scores_bindings() -> Vec<TensorParamBinding> {
    vec![
        TensorParamBinding::Variable, // q [SEQ_LEN, HEAD_DIM]
        TensorParamBinding::Variable, // k [SEQ_LEN, HEAD_DIM]
    ]
}

// ---------------------------------------------------------------------------
// 7. Attention weighted sum: softmax_weights @ V
// ---------------------------------------------------------------------------

/// Build attention weighted sum sub-graph.
///
/// Input: attention weights `[SEQ_LEN, SEQ_LEN]` (Variable, representing
///   softmax output), V `[SEQ_LEN, HEAD_DIM]` (Variable).
/// Output: `[SEQ_LEN, HEAD_DIM]`.
///
/// Key property: if attention weights are in [0, 1] (from softmax) and
/// V is in [-r, r], then the output is bounded by [-r, r] since the
/// weighted sum is a convex combination.
pub(super) fn build_attention_weighted_sum() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("qwen3_attention_weighted_sum");

    let weight_shape = [SEQ_LEN, SEQ_LEN];
    let v_shape = [SEQ_LEN, HEAD_DIM];

    let attn_weights = b.add_input("attn_weights", &weight_shape);
    let v = b.add_input("v", &v_shape);

    // Weighted sum: attn_weights @ V -> [SEQ_LEN, HEAD_DIM]
    let output = b.add_matmul(attn_weights, v, false, None, &v_shape);

    b.build(output)
        .expect("valid attention weighted sum sub-graph")
}

/// Bindings for attention weighted sum: weights=Variable, V=Variable.
pub(super) fn attention_weighted_sum_bindings() -> Vec<TensorParamBinding> {
    vec![
        TensorParamBinding::Variable, // attn_weights [SEQ_LEN, SEQ_LEN]
        TensorParamBinding::Variable, // v [SEQ_LEN, HEAD_DIM]
    ]
}

// ---------------------------------------------------------------------------
// 8. Causal mask application: add mask to scores before softmax
// ---------------------------------------------------------------------------

/// Build causal mask application sub-graph.
///
/// Input: raw attention scores `[SEQ_LEN, SEQ_LEN]` (Variable).
/// Output: `[SEQ_LEN, SEQ_LEN]` (masked scores with -1e9 at causal positions,
///   then softmax applied).
///
/// Causal mask: for position i, positions j > i are masked with -1e9.
/// After softmax, masked positions get weight ~0.
pub(super) fn build_causal_mask_attention() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("qwen3_causal_mask_attention");

    let score_shape = [SEQ_LEN, SEQ_LEN];

    let scores = b.add_input("scores", &score_shape);
    let mask = b.add_input("causal_mask", &score_shape);

    // Apply mask: scores + mask (mask has 0 for valid, -1e9 for invalid)
    let masked = b.add_binary_add(scores, mask, &score_shape);

    // Softmax normalizes masked scores — masked positions get ~0 weight
    let output = b.add_softmax(masked, 1, &score_shape);

    b.build(output)
        .expect("valid causal mask attention sub-graph")
}

/// Build a causal mask tensor: 0 for valid positions (j <= i), -1e9 for masked.
pub(super) fn build_causal_mask_tensor() -> ArrayD<f32> {
    let mask_value = -1e9_f32;
    let mut data = vec![0.0f32; SEQ_LEN * SEQ_LEN];
    for i in 0..SEQ_LEN {
        for j in 0..SEQ_LEN {
            if j > i {
                data[i * SEQ_LEN + j] = mask_value;
            }
        }
    }
    ArrayD::from_shape_vec(IxDyn(&[SEQ_LEN, SEQ_LEN]), data).expect("valid causal mask")
}

/// Bindings for causal mask attention: scores=Variable, mask=Constant.
pub(super) fn causal_mask_bindings() -> Vec<TensorParamBinding> {
    vec![
        TensorParamBinding::Variable, // scores [SEQ_LEN, SEQ_LEN]
        TensorParamBinding::ConstantTensor(build_causal_mask_tensor()), // causal_mask
    ]
}

// ---------------------------------------------------------------------------
// 9. Full attention block: projection + RoPE + GQA + causal attention + output
// ---------------------------------------------------------------------------

/// Build a full Qwen3 attention block sub-graph.
///
/// Input: `[SEQ_LEN, HIDDEN_DIM]` (Variable).
/// Output: `[SEQ_LEN, HIDDEN_DIM]`.
///
/// Pipeline:
///   1. Q/K/V linear projections
///   2. Multi-head attention with causal masking (uses `add_multi_head_attention`)
///   3. Output projection
///   4. Residual connection (input + attn_output)
///
/// This is the complete attention block composition, verifying that all
/// sub-components chain together while maintaining bounded outputs.
pub(super) fn build_full_attention_block() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("qwen3_full_attention_block");

    let shape = [SEQ_LEN, HIDDEN_DIM];

    let input = b.add_input("hidden", &shape);
    let q_w = b.add_input("q_proj_w", &[HIDDEN_DIM, HIDDEN_DIM]);
    let k_w = b.add_input("k_proj_w", &[HIDDEN_DIM, HIDDEN_DIM]);
    let v_w = b.add_input("v_proj_w", &[HIDDEN_DIM, HIDDEN_DIM]);
    let out_w = b.add_input("out_proj_w", &[HIDDEN_DIM, HIDDEN_DIM]);

    // Multi-head attention: projects Q/K/V, computes attention, projects output
    let attn_out = b
        .add_multi_head_attention(
            input,
            q_w,
            k_w,
            v_w,
            out_w,
            NUM_HEADS,
            nn_dsl::AttentionMask::Causal,
            &shape,
        )
        .expect("valid multi-head attention");

    // Residual connection: input + attention output
    let output = b.add_binary_add(input, attn_out, &shape);

    b.build(output)
        .expect("valid full attention block sub-graph")
}

/// Bindings for full attention block: input=Variable, weights=Constant.
pub(super) fn full_attention_block_bindings() -> Vec<TensorParamBinding> {
    let proj_w = ArrayD::from_elem(IxDyn(&[HIDDEN_DIM, HIDDEN_DIM]), WEIGHT_MAG);
    vec![
        TensorParamBinding::Variable,                       // hidden
        TensorParamBinding::ConstantTensor(proj_w.clone()), // q_proj_w
        TensorParamBinding::ConstantTensor(proj_w.clone()), // k_proj_w
        TensorParamBinding::ConstantTensor(proj_w.clone()), // v_proj_w
        TensorParamBinding::ConstantTensor(proj_w),         // out_proj_w
    ]
}

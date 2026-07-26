// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Builder helpers for Qwen3 decoder pipeline NY composition tests.
//!
//! Decomposes the Qwen3 decoder pipeline into verifiable sub-blocks:
//!
//! 1. **RMSNorm sub-block**: Isolated RMSNorm for normalization bounds
//! 2. **Self-attention sub-block**: RMSNorm -> MHA(causal) -> residual
//! 3. **MLP sub-block**: RMSNorm -> SwiGLU -> residual
//! 4. **Single decoder block**: attention + MLP sub-blocks composed
//! 5. **Post-norm + lm_head**: Final RMSNorm -> linear projection
//! 6. **2-block decoder stack**: Full pipeline with stacked blocks
//!
//! Uses the same architecture as `qwen3_decoder.rs` but exposes each
//! sub-block as a separately verifiable TensorKernelDef. This allows
//! per-sub-block IBP and CROWN verification, catching issues that the
//! full-model test may obscure (e.g., bounds blowup in a specific layer).
//!
//! Part of #3588: Compose verification for Qwen3 decoder block.

use nn_dsl::tensor_block_builder::TensorBlockBuilder;
use nn_dsl::tensor_ir::{TensorKernelDef, TensorNodeId};
use nn_dsl::AttentionMask;
use nn_verify::TensorParamBinding;
use ndarray::{ArrayD, IxDyn};

// ---------------------------------------------------------------------------
// Small-scale dimensions for NY tractability
// ---------------------------------------------------------------------------

/// Model dimension (production: 4096 for Qwen3-8B).
pub(super) const D_MODEL: usize = 16;

/// Number of attention heads (production: 32 for Qwen3-8B).
pub(super) const N_HEADS: usize = 2;

/// Per-head dimension: D_MODEL / N_HEADS.
pub(super) const HEAD_DIM: usize = D_MODEL / N_HEADS; // 8

/// FFN intermediate dimension (production: ~2.67x d_model for Qwen3).
pub(super) const INTERMEDIATE_SIZE: usize = D_MODEL * 2; // 32

/// Vocabulary size.
pub(super) const VOCAB_SIZE: usize = 16;

/// Sequence length.
pub(super) const SEQ_LEN: usize = 4;

/// Weight magnitude for small-scale test weights.
const WEIGHT_MAG: f32 = 0.001;

// ---------------------------------------------------------------------------
// 1. RMSNorm sub-block
// ---------------------------------------------------------------------------

/// Build an isolated RMSNorm sub-graph.
///
/// Input: `[SEQ_LEN, D_MODEL]` (Variable).
/// Output: `[SEQ_LEN, D_MODEL]`.
///
/// Tests normalization bounds propagation in isolation.
pub(super) fn build_rms_norm_subblock() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("qwen3_rms_norm");

    let shape = [SEQ_LEN, D_MODEL];

    let input = b.add_input("hidden", &shape);
    let eps = b.add_input("eps", &[1]);
    let weight = b.add_input("norm_weight", &[D_MODEL]);

    let output = b.add_rms_norm(input, eps, 1, weight, &shape);

    b.build(output).expect("valid RMSNorm sub-graph")
}

/// Bindings for RMSNorm sub-block: hidden=Variable, eps+weight=Constant.
pub(super) fn rms_norm_bindings() -> Vec<TensorParamBinding> {
    vec![
        TensorParamBinding::Variable, // hidden [SEQ_LEN, D_MODEL]
        TensorParamBinding::ConstantScalar(1e-5),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[D_MODEL]), 1.0f32)),
    ]
}

// ---------------------------------------------------------------------------
// 2. Self-attention sub-block (RMSNorm -> MHA -> residual)
// ---------------------------------------------------------------------------

/// Build the self-attention sub-block.
///
/// Input: `[SEQ_LEN, D_MODEL]` (Variable).
/// Output: `[SEQ_LEN, D_MODEL]`.
///
/// Pipeline: RMSNorm(input) -> MHA(causal) -> + input (residual).
pub(super) fn build_self_attention_subblock() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("qwen3_self_attention_subblock");

    let shape = [SEQ_LEN, D_MODEL];

    // Inputs
    let input = b.add_input("hidden", &shape);
    let eps = b.add_input("eps", &[1]);
    let attn_ln_w = b.add_input("attn_ln_w", &[D_MODEL]);
    let q_w = b.add_input("q_proj_w", &[D_MODEL, D_MODEL]);
    let k_w = b.add_input("k_proj_w", &[D_MODEL, D_MODEL]);
    let v_w = b.add_input("v_proj_w", &[D_MODEL, D_MODEL]);
    let o_w = b.add_input("o_proj_w", &[D_MODEL, D_MODEL]);

    // RMSNorm
    let normed = b.add_rms_norm(input, eps, 1, attn_ln_w, &shape);

    // Multi-head attention (causal)
    let attn_out = b
        .add_multi_head_attention(
            normed,
            q_w,
            k_w,
            v_w,
            o_w,
            N_HEADS,
            AttentionMask::Causal,
            &shape,
        )
        .expect("qwen3 self-attention");

    // Residual connection
    let output = b.add_binary_add(input, attn_out, &shape);

    b.build(output)
        .expect("valid self-attention sub-block graph")
}

/// Bindings for self-attention sub-block.
pub(super) fn self_attention_bindings() -> Vec<TensorParamBinding> {
    let mut bindings = Vec::new();
    bindings.push(TensorParamBinding::Variable); // hidden
    bindings.push(TensorParamBinding::ConstantScalar(1e-5)); // eps
    bindings.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
        IxDyn(&[D_MODEL]),
        1.0f32,
    ))); // attn_ln_w
         // Q, K, V, O projections
    for _ in 0..4 {
        bindings.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[D_MODEL, D_MODEL]),
            WEIGHT_MAG,
        )));
    }
    bindings
}

// ---------------------------------------------------------------------------
// 3. MLP sub-block (RMSNorm -> SwiGLU -> residual)
// ---------------------------------------------------------------------------

/// Build the MLP sub-block.
///
/// Input: `[SEQ_LEN, D_MODEL]` (Variable).
/// Output: `[SEQ_LEN, D_MODEL]`.
///
/// Pipeline: RMSNorm(input) -> SwiGLU(gate, up, down) -> + input (residual).
///
/// SwiGLU(x) = down_proj(silu(gate_proj(x)) * up_proj(x))
/// SiLU(x) = x * sigmoid(x)
pub(super) fn build_mlp_subblock() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("qwen3_mlp_subblock");

    let shape = [SEQ_LEN, D_MODEL];
    let inter_shape = [SEQ_LEN, INTERMEDIATE_SIZE];

    // Inputs
    let input = b.add_input("hidden", &shape);
    let eps = b.add_input("eps", &[1]);
    let mlp_ln_w = b.add_input("mlp_ln_w", &[D_MODEL]);
    let gate_w = b.add_input("gate_proj_w", &[INTERMEDIATE_SIZE, D_MODEL]);
    let up_w = b.add_input("up_proj_w", &[INTERMEDIATE_SIZE, D_MODEL]);
    let down_w = b.add_input("down_proj_w", &[D_MODEL, INTERMEDIATE_SIZE]);

    // RMSNorm
    let normed = b.add_rms_norm(input, eps, 1, mlp_ln_w, &shape);

    // SwiGLU
    let mlp_out = build_swiglu_inline(&mut b, normed, gate_w, up_w, down_w, &shape, &inter_shape);

    // Residual connection
    let output = b.add_binary_add(input, mlp_out, &shape);

    b.build(output).expect("valid MLP sub-block graph")
}

/// Bindings for MLP sub-block.
pub(super) fn mlp_subblock_bindings() -> Vec<TensorParamBinding> {
    vec![
        TensorParamBinding::Variable,             // hidden
        TensorParamBinding::ConstantScalar(1e-5), // eps
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[D_MODEL]), 1.0f32)), // mlp_ln_w
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[INTERMEDIATE_SIZE, D_MODEL]),
            WEIGHT_MAG,
        )), // gate_proj_w
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[INTERMEDIATE_SIZE, D_MODEL]),
            WEIGHT_MAG,
        )), // up_proj_w
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[D_MODEL, INTERMEDIATE_SIZE]),
            WEIGHT_MAG,
        )), // down_proj_w
    ]
}

// ---------------------------------------------------------------------------
// 4. Single decoder block (attention + MLP sub-blocks)
// ---------------------------------------------------------------------------

/// Build a single Qwen3 decoder block.
///
/// Input: `[SEQ_LEN, D_MODEL]` (Variable).
/// Output: `[SEQ_LEN, D_MODEL]`.
///
/// Pre-norm structure:
/// 1. RmsNorm -> MHA(causal) -> + residual
/// 2. RmsNorm -> SwiGLU MLP -> + residual
pub(super) fn build_single_decoder_block() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("qwen3_decoder_block");

    let shape = [SEQ_LEN, D_MODEL];
    let inter_shape = [SEQ_LEN, INTERMEDIATE_SIZE];

    // Inputs
    let input = b.add_input("hidden", &shape);
    let eps = b.add_input("eps", &[1]);

    // Attention sub-block weights
    let attn_ln_w = b.add_input("attn_ln_w", &[D_MODEL]);
    let q_w = b.add_input("q_proj_w", &[D_MODEL, D_MODEL]);
    let k_w = b.add_input("k_proj_w", &[D_MODEL, D_MODEL]);
    let v_w = b.add_input("v_proj_w", &[D_MODEL, D_MODEL]);
    let o_w = b.add_input("o_proj_w", &[D_MODEL, D_MODEL]);

    // MLP sub-block weights
    let mlp_ln_w = b.add_input("mlp_ln_w", &[D_MODEL]);
    let gate_w = b.add_input("gate_proj_w", &[INTERMEDIATE_SIZE, D_MODEL]);
    let up_w = b.add_input("up_proj_w", &[INTERMEDIATE_SIZE, D_MODEL]);
    let down_w = b.add_input("down_proj_w", &[D_MODEL, INTERMEDIATE_SIZE]);

    // Sub-block 1: RMSNorm -> MHA -> residual
    let attn_normed = b.add_rms_norm(input, eps, 1, attn_ln_w, &shape);
    let attn_out = b
        .add_multi_head_attention(
            attn_normed,
            q_w,
            k_w,
            v_w,
            o_w,
            N_HEADS,
            AttentionMask::Causal,
            &shape,
        )
        .expect("qwen3 self-attention");
    let residual1 = b.add_binary_add(input, attn_out, &shape);

    // Sub-block 2: RMSNorm -> SwiGLU -> residual
    let mlp_normed = b.add_rms_norm(residual1, eps, 1, mlp_ln_w, &shape);
    let mlp_out = build_swiglu_inline(
        &mut b,
        mlp_normed,
        gate_w,
        up_w,
        down_w,
        &shape,
        &inter_shape,
    );
    let output = b.add_binary_add(residual1, mlp_out, &shape);

    b.build(output).expect("valid decoder block graph")
}

/// Bindings for a single decoder block.
pub(super) fn single_decoder_block_bindings() -> Vec<TensorParamBinding> {
    let mut bindings = Vec::new();
    bindings.push(TensorParamBinding::Variable); // hidden
    bindings.push(TensorParamBinding::ConstantScalar(1e-5)); // eps

    // Attention sub-block: attn_ln_w, q_w, k_w, v_w, o_w
    bindings.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
        IxDyn(&[D_MODEL]),
        1.0f32,
    )));
    for _ in 0..4 {
        bindings.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[D_MODEL, D_MODEL]),
            WEIGHT_MAG,
        )));
    }

    // MLP sub-block: mlp_ln_w, gate_w, up_w, down_w
    bindings.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
        IxDyn(&[D_MODEL]),
        1.0f32,
    )));
    bindings.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
        IxDyn(&[INTERMEDIATE_SIZE, D_MODEL]),
        WEIGHT_MAG,
    )));
    bindings.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
        IxDyn(&[INTERMEDIATE_SIZE, D_MODEL]),
        WEIGHT_MAG,
    )));
    bindings.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
        IxDyn(&[D_MODEL, INTERMEDIATE_SIZE]),
        WEIGHT_MAG,
    )));

    bindings
}

// ---------------------------------------------------------------------------
// 5. Post-norm + lm_head projection
// ---------------------------------------------------------------------------

/// Build the post-norm + lm_head projection sub-graph.
///
/// Input: `[SEQ_LEN, D_MODEL]` (Variable).
/// Output: `[SEQ_LEN, VOCAB_SIZE]`.
///
/// Pipeline: RMSNorm(input) -> Linear(lm_head).
pub(super) fn build_post_norm_lm_head() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("qwen3_post_norm_lm_head");

    let shape = [SEQ_LEN, D_MODEL];
    let out_shape = [SEQ_LEN, VOCAB_SIZE];

    let input = b.add_input("hidden", &shape);
    let eps = b.add_input("eps", &[1]);
    let norm_w = b.add_input("ln_final_w", &[D_MODEL]);
    let lm_head_w = b.add_input("lm_head_w", &[D_MODEL, VOCAB_SIZE]);

    let normed = b.add_rms_norm(input, eps, 1, norm_w, &shape);
    let output = b.add_matmul(normed, lm_head_w, false, None, &out_shape);

    b.build(output)
        .expect("valid post-norm + lm_head sub-graph")
}

/// Bindings for post-norm + lm_head.
pub(super) fn post_norm_lm_head_bindings() -> Vec<TensorParamBinding> {
    vec![
        TensorParamBinding::Variable,             // hidden
        TensorParamBinding::ConstantScalar(1e-5), // eps
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[D_MODEL]), 1.0f32)), // norm_w
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[D_MODEL, VOCAB_SIZE]),
            WEIGHT_MAG,
        )), // lm_head_w
    ]
}

// ---------------------------------------------------------------------------
// 6. 2-block decoder stack (full pipeline)
// ---------------------------------------------------------------------------

/// Build a 2-block Qwen3 decoder stack with post-norm + lm_head.
///
/// Input: `[SEQ_LEN, D_MODEL]` (Variable).
/// Output: `[SEQ_LEN, VOCAB_SIZE]`.
///
/// Pipeline: token_emb -> decoder_block_0 -> decoder_block_1 -> RMSNorm -> lm_head.
pub(super) fn build_decoder_stack() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("qwen3_decoder_stack");

    let shape = [SEQ_LEN, D_MODEL];
    let inter_shape = [SEQ_LEN, INTERMEDIATE_SIZE];
    let out_shape = [SEQ_LEN, VOCAB_SIZE];

    // Variable input: token embeddings
    let input = b.add_input("token_emb", &shape);

    // Shared epsilon
    let eps = b.add_input("eps", &[1]);

    // Two decoder blocks
    let mut current = input;
    for i in 0..2 {
        current = add_decoder_block_inline(&mut b, current, eps, i, &shape, &inter_shape);
    }

    // Final RMSNorm + lm_head
    let ln_final_w = b.add_input("ln_final_w", &[D_MODEL]);
    let normed = b.add_rms_norm(current, eps, 1, ln_final_w, &shape);
    let lm_head_w = b.add_input("lm_head_w", &[D_MODEL, VOCAB_SIZE]);
    let output = b.add_matmul(normed, lm_head_w, false, None, &out_shape);

    b.build(output).expect("valid decoder stack graph")
}

/// Bindings for the 2-block decoder stack.
pub(super) fn decoder_stack_bindings() -> Vec<TensorParamBinding> {
    let mut bindings = Vec::new();
    bindings.push(TensorParamBinding::Variable); // token_emb
    bindings.push(TensorParamBinding::ConstantScalar(1e-5)); // eps

    // 2 decoder blocks
    for _ in 0..2 {
        // Attention sub-block: attn_ln_w, q_w, k_w, v_w, o_w
        bindings.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[D_MODEL]),
            1.0f32,
        )));
        for _ in 0..4 {
            bindings.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
                IxDyn(&[D_MODEL, D_MODEL]),
                WEIGHT_MAG,
            )));
        }

        // MLP sub-block: mlp_ln_w, gate_w, up_w, down_w
        bindings.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[D_MODEL]),
            1.0f32,
        )));
        bindings.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[INTERMEDIATE_SIZE, D_MODEL]),
            WEIGHT_MAG,
        )));
        bindings.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[INTERMEDIATE_SIZE, D_MODEL]),
            WEIGHT_MAG,
        )));
        bindings.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[D_MODEL, INTERMEDIATE_SIZE]),
            WEIGHT_MAG,
        )));
    }

    // Final RMSNorm weight
    bindings.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
        IxDyn(&[D_MODEL]),
        1.0f32,
    )));

    // lm_head weight
    bindings.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
        IxDyn(&[D_MODEL, VOCAB_SIZE]),
        WEIGHT_MAG,
    )));

    bindings
}

// ---------------------------------------------------------------------------
// Inline SwiGLU builder (shared by sub-blocks and full blocks)
// ---------------------------------------------------------------------------

/// Build SwiGLU MLP inline within a TensorBlockBuilder.
///
/// SwiGLU(x) = down_proj(silu(gate_proj(x)) * up_proj(x))
/// SiLU(x) = x * sigmoid(x)
fn build_swiglu_inline(
    b: &mut TensorBlockBuilder,
    input: TensorNodeId,
    gate_w: TensorNodeId,
    up_w: TensorNodeId,
    down_w: TensorNodeId,
    shape: &[usize],
    inter_shape: &[usize],
) -> TensorNodeId {
    // gate_proj(x) -> [S, intermediate]
    let gate = b.add_linear(input, gate_w, None, inter_shape);
    // sigmoid(gate) -> [S, intermediate]
    let gate_sig = b.add_sigmoid(gate, inter_shape);
    // SiLU: gate * sigmoid(gate) -> [S, intermediate]
    let gate_silu = b.add_binary_mul(gate, gate_sig, inter_shape);
    // up_proj(x) -> [S, intermediate]
    let up = b.add_linear(input, up_w, None, inter_shape);
    // gate_silu * up -> [S, intermediate]
    let gated = b.add_binary_mul(gate_silu, up, inter_shape);
    // down_proj(gated) -> [S, d_model]
    b.add_linear(gated, down_w, None, shape)
}

/// Build a full decoder block inline within a TensorBlockBuilder.
///
/// Adds all weight inputs with a layer prefix, then builds:
/// RMSNorm -> MHA -> residual -> RMSNorm -> SwiGLU -> residual.
fn add_decoder_block_inline(
    b: &mut TensorBlockBuilder,
    input: TensorNodeId,
    eps: TensorNodeId,
    layer_idx: usize,
    shape: &[usize],
    inter_shape: &[usize],
) -> TensorNodeId {
    let pfx = format!("layer{layer_idx}");

    // Attention sub-block weights
    let attn_ln_w = b.add_input(&format!("{pfx}_attn_ln_w"), &[D_MODEL]);
    let q_w = b.add_input(&format!("{pfx}_q_w"), &[D_MODEL, D_MODEL]);
    let k_w = b.add_input(&format!("{pfx}_k_w"), &[D_MODEL, D_MODEL]);
    let v_w = b.add_input(&format!("{pfx}_v_w"), &[D_MODEL, D_MODEL]);
    let o_w = b.add_input(&format!("{pfx}_o_w"), &[D_MODEL, D_MODEL]);

    // MLP sub-block weights
    let mlp_ln_w = b.add_input(&format!("{pfx}_mlp_ln_w"), &[D_MODEL]);
    let gate_w = b.add_input(&format!("{pfx}_gate_w"), &[INTERMEDIATE_SIZE, D_MODEL]);
    let up_w = b.add_input(&format!("{pfx}_up_w"), &[INTERMEDIATE_SIZE, D_MODEL]);
    let down_w = b.add_input(&format!("{pfx}_down_w"), &[D_MODEL, INTERMEDIATE_SIZE]);

    // Sub-block 1: RMSNorm -> MHA -> residual
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
        .expect("decoder block self-attention");
    let residual1 = b.add_binary_add(input, attn_out, shape);

    // Sub-block 2: RMSNorm -> SwiGLU -> residual
    let mlp_normed = b.add_rms_norm(residual1, eps, 1, mlp_ln_w, shape);
    let mlp_out = build_swiglu_inline(b, mlp_normed, gate_w, up_w, down_w, shape, inter_shape);
    b.add_binary_add(residual1, mlp_out, shape)
}

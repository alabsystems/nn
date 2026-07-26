// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Builder functions for Demucs transformer `TensorKernelDef`s.
//!
//! Backend-agnostic — constructs self-attention layers, cross-attention layers,
//! channel bridges, and input LayerNorms. Each component is a single
//! `TensorKernelDef` that can be dispatched via any backend.
//!
//! Extracted from `nn-metal` as part of #860.

use std::collections::HashMap;

use nn_dsl::tensor_block_builder::TensorBlockBuilder;
use nn_dsl::tensor_ir::{TensorIRError, TensorIRLayerError, TensorKernelDef, TensorNodeId};
use nn_dsl::AttentionMask;

use crate::demucs_transformer_constants::{
    FFN_HIDDEN_DIM, LAYER_NORM_EPS, NUM_HEADS, TRANSFORMER_DIM,
};
use crate::demucs_transformer_validate;
use crate::demucs_transformer_weights::{LayerNormWeights, TransformerLayerWeights};

use super::TransformerBuildError;

// ---------------------------------------------------------------------------
// Channel bridge (Conv1d, kernel=1)
// ---------------------------------------------------------------------------

/// Build a Conv1d channel bridge def (kernel=1, stride=1, padding=0).
///
/// Input: `[in_ch, seq_len]`, Output: `[out_ch, seq_len]`.
pub fn build_channel_bridge_def(
    name: &str,
    in_ch: usize,
    out_ch: usize,
    seq_len: usize,
) -> Result<(TensorKernelDef, HashMap<String, Vec<f32>>), TransformerBuildError> {
    let mut b = TensorBlockBuilder::new(name);

    let data = b.add_input(nn_dsl::input_names::DATA, &[in_ch, seq_len]);
    let weight = b.add_input("conv_weight", &[out_ch, in_ch, 1]);
    let bias = b.add_input("conv_bias", &[out_ch]);

    let out = b.add_conv1d(data, weight, Some(bias), 1, 0, &[out_ch, seq_len]);

    let wmap = HashMap::new();
    Ok((b.build(out)?, wmap))
}

/// Build weight map for a Conv1d bridge.
pub fn build_conv1d_weight_map(weight: &[f32], bias: &[f32]) -> HashMap<String, Vec<f32>> {
    let mut map = HashMap::new();
    map.insert("conv_weight".to_string(), weight.to_vec());
    map.insert("conv_bias".to_string(), bias.to_vec());
    map
}

// ---------------------------------------------------------------------------
// Input LayerNorm
// ---------------------------------------------------------------------------

/// Build a standalone LayerNorm def operating on `[seq_len, D]`.
///
/// Normalizes along axis=1 (the D dimension).
pub fn build_layer_norm_def(
    name: &str,
    seq_len: usize,
    ln_weights: &LayerNormWeights,
) -> Result<(TensorKernelDef, HashMap<String, Vec<f32>>), TransformerBuildError> {
    let d = TRANSFORMER_DIM;
    let mut b = TensorBlockBuilder::new(name);

    let data = b.add_input(nn_dsl::input_names::DATA, &[seq_len, d]);
    let eps = b.add_input("eps", &[1]);
    let gamma = b.add_input("ln_weight", &[d]);
    let beta = b.add_input("ln_bias", &[d]);

    let out = b.add_layer_norm(data, eps, 1, gamma, beta, &[seq_len, d]);

    let mut wmap = HashMap::new();
    wmap.insert("eps".to_string(), vec![LAYER_NORM_EPS]);
    wmap.insert("ln_weight".to_string(), ln_weights.weight.clone());
    wmap.insert("ln_bias".to_string(), ln_weights.bias.clone());

    Ok((b.build(out)?, wmap))
}

// ---------------------------------------------------------------------------
// Self-attention transformer layer
// ---------------------------------------------------------------------------

/// Build a self-attention transformer layer def.
///
/// Structure: `x += gamma_1 * MHA(LN1(x)); x += gamma_2 * FFN(LN2(x)); x = LN_out(x)`
///
/// Input: `[seq_len, D]` via "data". Output: `[seq_len, D]`.
pub fn build_self_attention_layer_def(
    name: &str,
    seq_len: usize,
    layer_weights: &TransformerLayerWeights,
) -> Result<(TensorKernelDef, HashMap<String, Vec<f32>>), TensorIRError> {
    let w = match layer_weights {
        TransformerLayerWeights::SelfAttention(w) => w,
        TransformerLayerWeights::CrossAttention(_) => {
            return Err(TensorIRLayerError::MhaZeroHeads.into());
        }
    };

    let d = TRANSFORMER_DIM;
    let ffn = FFN_HIDDEN_DIM;
    let shape = [seq_len, d];
    let ffn_shape = [seq_len, ffn];

    let mut b = TensorBlockBuilder::new(name);

    let data = b.add_input(nn_dsl::input_names::DATA, &shape);

    let ln1_eps = b.add_input("ln1_eps", &[1]);
    let ln1_gamma = b.add_input("ln1_weight", &[d]);
    let ln1_beta = b.add_input("ln1_bias", &[d]);
    let ln2_eps = b.add_input("ln2_eps", &[1]);
    let ln2_gamma = b.add_input("ln2_weight", &[d]);
    let ln2_beta = b.add_input("ln2_bias", &[d]);
    let lnout_eps = b.add_input("lnout_eps", &[1]);
    let lnout_gamma = b.add_input("lnout_weight", &[d]);
    let lnout_beta = b.add_input("lnout_bias", &[d]);

    let q_w = b.add_input("q_weight", &[d, d]);
    let k_w = b.add_input("k_weight", &[d, d]);
    let v_w = b.add_input("v_weight", &[d, d]);
    let out_w = b.add_input("out_weight", &[d, d]);

    let ffn1_w = b.add_input("ffn_linear1_weight", &[ffn, d]);
    let ffn1_b = b.add_input("ffn_linear1_bias", &[ffn]);
    let ffn2_w = b.add_input("ffn_linear2_weight", &[d, ffn]);
    let ffn2_b = b.add_input("ffn_linear2_bias", &[d]);

    let gamma_1 = b.add_input("gamma_1", &[d]);
    let gamma_2 = b.add_input("gamma_2", &[d]);

    // LN1(x) → MHA → gamma_1 * result → residual
    let normed1 = b.add_layer_norm(data, ln1_eps, 1, ln1_gamma, ln1_beta, &shape);
    let attn = b.add_multi_head_attention(
        normed1,
        q_w,
        k_w,
        v_w,
        out_w,
        NUM_HEADS,
        AttentionMask::Standard,
        &shape,
    )?;
    let gamma_1_bc = b.add_broadcast(gamma_1, &shape);
    let scaled_attn = b.add_binary_mul(attn, gamma_1_bc, &shape);
    let residual1 = b.add_binary_add(data, scaled_attn, &shape);

    // LN2(residual1) → FFN → gamma_2 * result → residual
    let normed2 = b.add_layer_norm(residual1, ln2_eps, 1, ln2_gamma, ln2_beta, &shape);
    let ffn1 = b.add_linear(normed2, ffn1_w, Some(ffn1_b), &ffn_shape);
    let ffn_act = b.add_gelu(ffn1, &ffn_shape);
    let ffn2 = b.add_linear(ffn_act, ffn2_w, Some(ffn2_b), &shape);
    let gamma_2_bc = b.add_broadcast(gamma_2, &shape);
    let scaled_ffn = b.add_binary_mul(ffn2, gamma_2_bc, &shape);
    let residual2 = b.add_binary_add(residual1, scaled_ffn, &shape);

    let out = b.add_layer_norm(residual2, lnout_eps, 1, lnout_gamma, lnout_beta, &shape);

    let wmap = demucs_transformer_validate::build_self_attention_weight_map(w);
    Ok((b.build(out)?, wmap))
}

// ---------------------------------------------------------------------------
// Cross-attention transformer layer
// ---------------------------------------------------------------------------

/// Build a cross-attention transformer layer def.
///
/// Two inputs: "data" `[q_seq, D]` (own branch) and "cross" `[kv_seq, D]`
/// (other branch). Output: `[q_seq, D]`.
pub fn build_cross_attention_layer_def(
    name: &str,
    q_seq_len: usize,
    kv_seq_len: usize,
    layer_weights: &TransformerLayerWeights,
) -> Result<(TensorKernelDef, HashMap<String, Vec<f32>>), TensorIRError> {
    let w = match layer_weights {
        TransformerLayerWeights::CrossAttention(w) => w,
        TransformerLayerWeights::SelfAttention(_) => {
            return Err(TensorIRLayerError::MhaZeroHeads.into());
        }
    };

    let d = TRANSFORMER_DIM;
    let ffn = FFN_HIDDEN_DIM;
    let h = NUM_HEADS;
    let head_dim = d / h;
    let q_shape = [q_seq_len, d];
    let kv_shape = [kv_seq_len, d];
    let ffn_shape = [q_seq_len, ffn];

    let mut b = TensorBlockBuilder::new(name);

    let data = b.add_input(nn_dsl::input_names::DATA, &q_shape);
    let cross = b.add_input("cross", &kv_shape);

    let ln1_eps = b.add_input("ln1_eps", &[1]);
    let ln1_gamma = b.add_input("ln1_weight", &[d]);
    let ln1_beta = b.add_input("ln1_bias", &[d]);
    let ln2_eps = b.add_input("ln2_eps", &[1]);
    let ln2_gamma = b.add_input("ln2_weight", &[d]);
    let ln2_beta = b.add_input("ln2_bias", &[d]);
    let ln3_eps = b.add_input("ln3_eps", &[1]);
    let ln3_gamma = b.add_input("ln3_weight", &[d]);
    let ln3_beta = b.add_input("ln3_bias", &[d]);
    let lnout_eps = b.add_input("lnout_eps", &[1]);
    let lnout_gamma = b.add_input("lnout_weight", &[d]);
    let lnout_beta = b.add_input("lnout_bias", &[d]);

    let q_w = b.add_input("q_weight", &[d, d]);
    let k_w = b.add_input("k_weight", &[d, d]);
    let v_w = b.add_input("v_weight", &[d, d]);
    let out_w = b.add_input("out_weight", &[d, d]);

    let ffn1_w = b.add_input("ffn_linear1_weight", &[ffn, d]);
    let ffn1_b = b.add_input("ffn_linear1_bias", &[ffn]);
    let ffn2_w = b.add_input("ffn_linear2_weight", &[d, ffn]);
    let ffn2_b = b.add_input("ffn_linear2_bias", &[d]);

    let gamma_1 = b.add_input("gamma_1", &[d]);
    let gamma_2 = b.add_input("gamma_2", &[d]);

    // Q = LN1(data), K/V = LN2(cross)
    let normed_q = b.add_layer_norm(data, ln1_eps, 1, ln1_gamma, ln1_beta, &q_shape);
    let normed_kv = b.add_layer_norm(cross, ln2_eps, 1, ln2_gamma, ln2_beta, &kv_shape);

    let cross_attn = build_cross_mha(
        &mut b, normed_q, normed_kv, q_w, k_w, v_w, out_w, h, head_dim, q_seq_len, kv_seq_len, d,
    );

    let gamma_1_bc = b.add_broadcast(gamma_1, &q_shape);
    let scaled_attn = b.add_binary_mul(cross_attn, gamma_1_bc, &q_shape);
    let residual1 = b.add_binary_add(data, scaled_attn, &q_shape);

    let normed3 = b.add_layer_norm(residual1, ln3_eps, 1, ln3_gamma, ln3_beta, &q_shape);
    let ffn1 = b.add_linear(normed3, ffn1_w, Some(ffn1_b), &ffn_shape);
    let ffn_act = b.add_gelu(ffn1, &ffn_shape);
    let ffn2 = b.add_linear(ffn_act, ffn2_w, Some(ffn2_b), &q_shape);
    let gamma_2_bc = b.add_broadcast(gamma_2, &q_shape);
    let scaled_ffn = b.add_binary_mul(ffn2, gamma_2_bc, &q_shape);
    let residual2 = b.add_binary_add(residual1, scaled_ffn, &q_shape);

    let out = b.add_layer_norm(residual2, lnout_eps, 1, lnout_gamma, lnout_beta, &q_shape);

    let wmap = demucs_transformer_validate::build_cross_attention_weight_map(w);
    Ok((b.build(out)?, wmap))
}

/// Build cross-attention from primitives: separate Q and K/V sources.
fn build_cross_mha(
    b: &mut TensorBlockBuilder,
    q_input: TensorNodeId,
    kv_input: TensorNodeId,
    q_w: TensorNodeId,
    k_w: TensorNodeId,
    v_w: TensorNodeId,
    out_w: TensorNodeId,
    num_heads: usize,
    head_dim: usize,
    q_seq: usize,
    kv_seq: usize,
    d: usize,
) -> TensorNodeId {
    let q_proj = b.add_linear(q_input, q_w, None, &[q_seq, d]);
    let k_proj = b.add_linear(kv_input, k_w, None, &[kv_seq, d]);
    let v_proj = b.add_linear(kv_input, v_w, None, &[kv_seq, d]);

    let q_r = b.add_reshape(q_proj, &[q_seq, num_heads, head_dim]);
    let k_r = b.add_reshape(k_proj, &[kv_seq, num_heads, head_dim]);
    let v_r = b.add_reshape(v_proj, &[kv_seq, num_heads, head_dim]);

    let q_t = b.add_transpose(q_r, &[1, 0, 2], &[num_heads, q_seq, head_dim]);
    let k_t = b.add_transpose(k_r, &[1, 0, 2], &[num_heads, kv_seq, head_dim]);
    let v_t = b.add_transpose(v_r, &[1, 0, 2], &[num_heads, kv_seq, head_dim]);

    let scale = 1.0 / (head_dim as f32).sqrt();
    let attn = b.add_attention(
        q_t,
        k_t,
        v_t,
        AttentionMask::Standard,
        Some(scale),
        &[num_heads, q_seq, head_dim],
    );

    let attn_back = b.add_transpose(attn, &[1, 0, 2], &[q_seq, num_heads, head_dim]);
    let attn_flat = b.add_reshape(attn_back, &[q_seq, d]);

    b.add_linear(attn_flat, out_w, None, &[q_seq, d])
}

#[cfg(test)]
#[path = "demucs_transformer_builders_tests.rs"]
mod tests;

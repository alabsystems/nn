// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Transformer block composite builder for `TensorBlockBuilder`.
//!
//! Decomposes a standard pre-norm transformer block into existing primitives:
//! LayerNorm → MHA → BinaryAdd(residual) → LayerNorm → Linear → GELU → Linear
//! → BinaryAdd(residual). See `designs/2026-03-02-transformer-verification-plan.md`
//! Phase C and #811.

use crate::tensor_ir::{TensorIRError, TensorIRLayerError, TensorNodeId};
use crate::AttentionMask;

use super::TensorBlockBuilder;

#[cfg(test)]
#[path = "tensor_block_builder_transformer_tests.rs"]
mod tests;

#[cfg(test)]
#[path = "tensor_block_builder_cross_attn_tests.rs"]
mod cross_attn_tests;

/// Configuration for a single transformer block.
///
/// Controls multi-head attention head count, masking strategy, LayerNorm
/// epsilon, and FFN intermediate width. Used by `add_transformer_block()`.
#[derive(Clone, Copy, Debug)]
pub struct TransformerBlockConfig {
    /// Number of attention heads. `model_dim` must be divisible by this.
    pub num_heads: usize,
    /// Attention mask type (Standard or Causal).
    pub mask: AttentionMask,
    /// FFN intermediate (hidden) dimension. Must be > 0.
    pub ffn_hidden_dim: usize,
}

/// Weight inputs for a single transformer block.
///
/// All weight tensors must be created via `add_input()` before calling
/// `add_transformer_block()`. Shape constraints:
/// - `ln1_weight`, `ln1_bias`, `ln2_weight`, `ln2_bias`: `[D]`
/// - `q_weight`, `k_weight`, `v_weight`, `out_weight`: `[D, D]`
/// - `ffn1_weight`: `[ffn_hidden_dim, D]`
/// - `ffn2_weight`: `[D, ffn_hidden_dim]`
/// - `eps`: `[1]` (scalar constant)
#[derive(Clone, Copy, Debug)]
pub struct TransformerBlockWeights {
    /// LayerNorm 1 (pre-attention) scale parameter `[D]`.
    pub ln1_weight: TensorNodeId,
    /// LayerNorm 1 (pre-attention) bias parameter `[D]`.
    pub ln1_bias: TensorNodeId,
    /// LayerNorm 2 (pre-FFN) scale parameter `[D]`.
    pub ln2_weight: TensorNodeId,
    /// LayerNorm 2 (pre-FFN) bias parameter `[D]`.
    pub ln2_bias: TensorNodeId,
    /// Query projection weight `[D, D]`.
    pub q_weight: TensorNodeId,
    /// Key projection weight `[D, D]`.
    pub k_weight: TensorNodeId,
    /// Value projection weight `[D, D]`.
    pub v_weight: TensorNodeId,
    /// Output projection weight `[D, D]`.
    pub out_weight: TensorNodeId,
    /// FFN first linear weight `[ffn_hidden_dim, D]`.
    pub ffn1_weight: TensorNodeId,
    /// FFN second linear weight `[D, ffn_hidden_dim]`.
    pub ffn2_weight: TensorNodeId,
    /// LayerNorm epsilon constant `[1]`.
    pub eps: TensorNodeId,
}

/// Configuration for a cross-attention transformer block.
///
/// Cross-attention projects Q from one input and K/V from another.
/// The output shape matches the Q input shape `[T_q, D]`.
#[derive(Clone, Copy, Debug)]
pub struct CrossAttentionBlockConfig {
    /// Number of attention heads. `model_dim` must be divisible by this.
    pub num_heads: usize,
    /// Attention mask type (Standard or Causal).
    pub mask: AttentionMask,
    /// FFN intermediate (hidden) dimension. Must be > 0.
    pub ffn_hidden_dim: usize,
}

/// Weight inputs for a cross-attention transformer block.
///
/// Cross-attention has 4 LayerNorms:
/// - `ln1`: pre-attention normalization on Q input
/// - `ln2`: pre-attention normalization on KV input (cross branch)
/// - `ln3`: pre-FFN normalization
/// - `ln_out`: output normalization
///
/// Shape constraints: same as `TransformerBlockWeights` except `ln2` is applied
/// to the KV input sequence.
#[derive(Clone, Copy, Debug)]
pub struct CrossAttentionBlockWeights {
    /// LayerNorm 1 (pre-attention, Q branch) scale `[D]`.
    pub ln1_weight: TensorNodeId,
    /// LayerNorm 1 (pre-attention, Q branch) bias `[D]`.
    pub ln1_bias: TensorNodeId,
    /// LayerNorm 2 (pre-attention, KV branch) scale `[D]`.
    pub ln2_weight: TensorNodeId,
    /// LayerNorm 2 (pre-attention, KV branch) bias `[D]`.
    pub ln2_bias: TensorNodeId,
    /// LayerNorm 3 (pre-FFN) scale `[D]`.
    pub ln3_weight: TensorNodeId,
    /// LayerNorm 3 (pre-FFN) bias `[D]`.
    pub ln3_bias: TensorNodeId,
    /// Output LayerNorm scale `[D]`.
    pub ln_out_weight: TensorNodeId,
    /// Output LayerNorm bias `[D]`.
    pub ln_out_bias: TensorNodeId,
    /// Query projection weight `[D, D]`.
    pub q_weight: TensorNodeId,
    /// Key projection weight `[D, D]`.
    pub k_weight: TensorNodeId,
    /// Value projection weight `[D, D]`.
    pub v_weight: TensorNodeId,
    /// Output projection weight `[D, D]`.
    pub out_weight: TensorNodeId,
    /// FFN first linear weight `[ffn_hidden_dim, D]`.
    pub ffn1_weight: TensorNodeId,
    /// FFN second linear weight `[D, ffn_hidden_dim]`.
    pub ffn2_weight: TensorNodeId,
    /// LayerNorm epsilon constant `[1]`.
    pub eps: TensorNodeId,
}

impl TensorBlockBuilder {
    /// Add a pre-norm transformer block. Returns the output node ID.
    ///
    /// Decomposes into:
    /// 1. `LayerNorm(input)` → attention pre-normalization
    /// 2. `MHA(normed, Q, K, V, out)` → multi-head self-attention
    /// 3. `BinaryAdd(input, attn)` → first residual connection
    /// 4. `LayerNorm(residual1)` → FFN pre-normalization
    /// 5. `Linear(normed2, ffn1_w)` → FFN up-projection
    /// 6. `GELU(ffn1)` → FFN activation
    /// 7. `Linear(act, ffn2_w)` → FFN down-projection
    /// 8. `BinaryAdd(residual1, ffn2)` → second residual connection
    ///
    /// Input shape: `[T, D]`. Output shape: `[T, D]`.
    /// `D` must be divisible by `config.num_heads`.
    ///
    /// Maps to NY via composed LayerNormLayer + LinearLayer +
    /// TransposeLayer + SelfAttentionLayer + GeluLayer chain.
    pub fn add_transformer_block(
        &mut self,
        input: TensorNodeId,
        weights: &TransformerBlockWeights,
        config: &TransformerBlockConfig,
    ) -> Result<TensorNodeId, TensorIRError> {
        // Validate config
        if config.num_heads == 0 {
            return Err(TensorIRLayerError::TransformerZeroHeads.into());
        }
        if config.ffn_hidden_dim == 0 {
            return Err(TensorIRLayerError::TransformerZeroFfnDim.into());
        }

        // Validate input shape: must be [T, D]
        let input_shape = self.nodes[input.index()].shape.clone();
        if input_shape.len() != 2 {
            return Err(TensorIRLayerError::TransformerInputRankInvalid {
                rank: input_shape.len(),
            }
            .into());
        }
        let seq_len = input_shape[0];
        let model_dim = input_shape[1];

        if !model_dim.is_multiple_of(config.num_heads) {
            return Err(TensorIRLayerError::MhaHeadDimNotDivisible {
                model_dim,
                num_heads: config.num_heads,
            }
            .into());
        }

        let shape = [seq_len, model_dim];
        let ffn_shape = [seq_len, config.ffn_hidden_dim];

        // Pre-norm: LayerNorm → MHA → residual
        let normed = self.add_layer_norm(
            input,
            weights.eps,
            1, // normalize over last axis
            weights.ln1_weight,
            weights.ln1_bias,
            &shape,
        );

        let attn = self.add_multi_head_attention(
            normed,
            weights.q_weight,
            weights.k_weight,
            weights.v_weight,
            weights.out_weight,
            config.num_heads,
            config.mask,
            &shape,
        )?;

        let residual1 = self.add_binary_add(input, attn, &shape);

        // Pre-norm: LayerNorm → FFN → residual
        let normed2 = self.add_layer_norm(
            residual1,
            weights.eps,
            1,
            weights.ln2_weight,
            weights.ln2_bias,
            &shape,
        );

        let ffn1 = self.add_linear(normed2, weights.ffn1_weight, None, &ffn_shape);
        let act = self.add_gelu(ffn1, &ffn_shape);
        let ffn2 = self.add_linear(act, weights.ffn2_weight, None, &shape);

        Ok(self.add_binary_add(residual1, ffn2, &shape))
    }

    /// Add a pre-norm cross-attention transformer block. Returns the output node ID.
    ///
    /// Cross-attention: Q is projected from `q_input`, K/V from `kv_input`.
    /// Used in encoder-decoder architectures and cross-domain transformers
    /// (HTDemucs temporal↔spectral).
    ///
    /// Decomposes into:
    /// 1. `LayerNorm(q_input)` → Q pre-normalization
    /// 2. `LayerNorm(kv_input)` → KV pre-normalization
    /// 3. `CrossMHA(normed_q, normed_kv)` → cross-attention
    /// 4. `BinaryAdd(q_input, attn)` → first residual (on Q branch)
    /// 5. `LayerNorm(residual1)` → FFN pre-normalization
    /// 6. `Linear → GELU → Linear` → FFN
    /// 7. `BinaryAdd(residual1, ffn)` → second residual
    /// 8. `LayerNorm(residual2)` → output normalization
    ///
    /// `q_input` shape: `[T_q, D]`. `kv_input` shape: `[T_kv, D]`.
    /// Output shape: `[T_q, D]`.
    pub fn add_cross_attention_transformer_block(
        &mut self,
        q_input: TensorNodeId,
        kv_input: TensorNodeId,
        weights: &CrossAttentionBlockWeights,
        config: &CrossAttentionBlockConfig,
    ) -> Result<TensorNodeId, TensorIRError> {
        if config.num_heads == 0 {
            return Err(TensorIRLayerError::TransformerZeroHeads.into());
        }
        if config.ffn_hidden_dim == 0 {
            return Err(TensorIRLayerError::TransformerZeroFfnDim.into());
        }

        let q_shape = self.nodes[q_input.index()].shape.clone();
        let kv_shape = self.nodes[kv_input.index()].shape.clone();
        if q_shape.len() != 2 {
            return Err(TensorIRLayerError::TransformerInputRankInvalid {
                rank: q_shape.len(),
            }
            .into());
        }
        if kv_shape.len() != 2 {
            return Err(TensorIRLayerError::TransformerInputRankInvalid {
                rank: kv_shape.len(),
            }
            .into());
        }
        let q_seq = q_shape[0];
        let model_dim = q_shape[1];

        if kv_shape[1] != model_dim {
            return Err(TensorIRLayerError::MhaHeadDimNotDivisible {
                model_dim: kv_shape[1],
                num_heads: config.num_heads,
            }
            .into());
        }

        if !model_dim.is_multiple_of(config.num_heads) {
            return Err(TensorIRLayerError::MhaHeadDimNotDivisible {
                model_dim,
                num_heads: config.num_heads,
            }
            .into());
        }

        let shape = [q_seq, model_dim];
        let kv_norm_shape = [kv_shape[0], model_dim];
        let ffn_shape = [q_seq, config.ffn_hidden_dim];

        // LN1(q_input) → Q source, LN2(kv_input) → K/V source
        let normed_q = self.add_layer_norm(
            q_input,
            weights.eps,
            1,
            weights.ln1_weight,
            weights.ln1_bias,
            &shape,
        );
        let normed_kv = self.add_layer_norm(
            kv_input,
            weights.eps,
            1,
            weights.ln2_weight,
            weights.ln2_bias,
            &kv_norm_shape,
        );

        let attn = self.add_multi_head_cross_attention(
            normed_q,
            normed_kv,
            weights.q_weight,
            weights.k_weight,
            weights.v_weight,
            weights.out_weight,
            config.num_heads,
            config.mask,
            &shape,
        )?;

        let residual1 = self.add_binary_add(q_input, attn, &shape);

        // LN3 → FFN → residual
        let normed3 = self.add_layer_norm(
            residual1,
            weights.eps,
            1,
            weights.ln3_weight,
            weights.ln3_bias,
            &shape,
        );
        let ffn1 = self.add_linear(normed3, weights.ffn1_weight, None, &ffn_shape);
        let act = self.add_gelu(ffn1, &ffn_shape);
        let ffn2 = self.add_linear(act, weights.ffn2_weight, None, &shape);
        let residual2 = self.add_binary_add(residual1, ffn2, &shape);

        // Output LayerNorm
        Ok(self.add_layer_norm(
            residual2,
            weights.eps,
            1,
            weights.ln_out_weight,
            weights.ln_out_bias,
            &shape,
        ))
    }
}

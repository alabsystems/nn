// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Composite attention MHA builder methods for `TensorBlockBuilder`.
//!
//! Extracted from `tensor_block_builder_ops.rs` to stay under the 500-line limit.
//! Contains `add_multi_head_attention` (self-attention) and
//! `add_multi_head_cross_attention` (cross-attention). Both decompose into:
//! Linear→Reshape→Transpose→Attention→Transpose→Reshape→Linear.
//!
//! Part of #779 Phase D, #823 extraction.

use super::*;
use crate::tensor_ir::{TensorIRError, TensorIRLayerError};
use crate::AttentionMask;

impl TensorBlockBuilder {
    /// Add a multi-head attention composite op. Returns output node ID.
    ///
    /// Decomposes into: Linear(Q,K,V) → Reshape → Transpose → Attention →
    /// Transpose → Reshape → Linear(out).
    ///
    /// Input shape: `[T, D]`. All weight shapes: `[D, D]`. Output: `output_shape`.
    /// `D` must be divisible by `num_heads`. Scale is `1/sqrt(head_dim)`.
    ///
    /// Supports `AttentionMask::Standard` and `AttentionMask::Causal`.
    /// Maps to NY via composed LinearLayer + TransposeLayer + SelfAttentionLayer.
    pub fn add_multi_head_attention(
        &mut self,
        input: TensorNodeId,
        q_weight: TensorNodeId,
        k_weight: TensorNodeId,
        v_weight: TensorNodeId,
        out_weight: TensorNodeId,
        num_heads: usize,
        mask: AttentionMask,
        output_shape: &[usize],
    ) -> Result<TensorNodeId, TensorIRError> {
        if num_heads == 0 {
            return Err(TensorIRLayerError::MhaZeroHeads.into());
        }

        let input_shape = self.nodes[input.index()].shape.clone();
        if input_shape.len() != 2 {
            return Err(TensorIRLayerError::MhaInputRankInvalid {
                rank: input_shape.len(),
            }
            .into());
        }
        let seq_len = input_shape[0];
        let model_dim = input_shape[1];

        if !model_dim.is_multiple_of(num_heads) {
            return Err(TensorIRLayerError::MhaHeadDimNotDivisible {
                model_dim,
                num_heads,
            }
            .into());
        }
        let head_dim = model_dim / num_heads;

        // Project Q, K, V: [T, D] → [T, D]
        let proj_shape = [seq_len, model_dim];
        let q = self.add_linear(input, q_weight, None, &proj_shape);
        let k = self.add_linear(input, k_weight, None, &proj_shape);
        let v = self.add_linear(input, v_weight, None, &proj_shape);

        // Reshape to [T, H, head_dim]
        let reshaped = [seq_len, num_heads, head_dim];
        let q = self.add_reshape(q, &reshaped);
        let k = self.add_reshape(k, &reshaped);
        let v = self.add_reshape(v, &reshaped);

        // Transpose to [H, T, head_dim] for per-head attention
        let transposed = [num_heads, seq_len, head_dim];
        let q = self.add_transpose(q, &[1, 0, 2], &transposed);
        let k = self.add_transpose(k, &[1, 0, 2], &transposed);
        let v = self.add_transpose(v, &[1, 0, 2], &transposed);

        // Per-head attention with scale = 1/sqrt(head_dim)
        let scale = 1.0 / (head_dim as f32).sqrt();
        let attn = self.add_attention(q, k, v, mask, Some(scale), &transposed);

        // Transpose back to [T, H, head_dim]
        let attn = self.add_transpose(attn, &[1, 0, 2], &reshaped);

        // Reshape to [T, D]
        let attn = self.add_reshape(attn, &proj_shape);

        // Output projection
        Ok(self.add_linear(attn, out_weight, None, output_shape))
    }

    /// Add a multi-head cross-attention composite op. Returns output node ID.
    ///
    /// Like `add_multi_head_attention` but with separate Q and KV inputs.
    /// Q is projected from `q_input`, K and V are projected from `kv_input`.
    /// This is the core cross-attention pattern used in encoder-decoder
    /// transformers (e.g., HTDemucs bottleneck: temporal queries spectral).
    ///
    /// Q input shape: `[T_q, D]`. KV input shape: `[T_kv, D]`.
    /// Output shape follows Q: `output_shape` should be `[T_q, D]`.
    /// `D` must be divisible by `num_heads`. Scale is `1/sqrt(head_dim)`.
    pub fn add_multi_head_cross_attention(
        &mut self,
        q_input: TensorNodeId,
        kv_input: TensorNodeId,
        q_weight: TensorNodeId,
        k_weight: TensorNodeId,
        v_weight: TensorNodeId,
        out_weight: TensorNodeId,
        num_heads: usize,
        mask: AttentionMask,
        output_shape: &[usize],
    ) -> Result<TensorNodeId, TensorIRError> {
        if num_heads == 0 {
            return Err(TensorIRLayerError::MhaZeroHeads.into());
        }

        let q_shape = self.nodes[q_input.index()].shape.clone();
        let kv_shape = self.nodes[kv_input.index()].shape.clone();

        if q_shape.len() != 2 {
            return Err(TensorIRLayerError::MhaInputRankInvalid {
                rank: q_shape.len(),
            }
            .into());
        }
        if kv_shape.len() != 2 {
            return Err(TensorIRLayerError::MhaInputRankInvalid {
                rank: kv_shape.len(),
            }
            .into());
        }

        let q_seq = q_shape[0];
        let model_dim = q_shape[1];
        let kv_seq = kv_shape[0];
        let kv_dim = kv_shape[1];

        if kv_dim != model_dim {
            return Err(TensorIRLayerError::MhaHeadDimNotDivisible {
                model_dim: kv_dim,
                num_heads: model_dim,
            }
            .into());
        }

        if !model_dim.is_multiple_of(num_heads) {
            return Err(TensorIRLayerError::MhaHeadDimNotDivisible {
                model_dim,
                num_heads,
            }
            .into());
        }
        let head_dim = model_dim / num_heads;

        // Project Q from q_input, K/V from kv_input
        let q_proj_shape = [q_seq, model_dim];
        let kv_proj_shape = [kv_seq, model_dim];
        let q = self.add_linear(q_input, q_weight, None, &q_proj_shape);
        let k = self.add_linear(kv_input, k_weight, None, &kv_proj_shape);
        let v = self.add_linear(kv_input, v_weight, None, &kv_proj_shape);

        // Reshape to [T, H, head_dim]
        let q_reshaped = [q_seq, num_heads, head_dim];
        let kv_reshaped = [kv_seq, num_heads, head_dim];
        let q = self.add_reshape(q, &q_reshaped);
        let k = self.add_reshape(k, &kv_reshaped);
        let v = self.add_reshape(v, &kv_reshaped);

        // Transpose to [H, T, head_dim]
        let q_transposed = [num_heads, q_seq, head_dim];
        let kv_transposed = [num_heads, kv_seq, head_dim];
        let q = self.add_transpose(q, &[1, 0, 2], &q_transposed);
        let k = self.add_transpose(k, &[1, 0, 2], &kv_transposed);
        let v = self.add_transpose(v, &[1, 0, 2], &kv_transposed);

        // Per-head attention: output shape is [H, T_q, head_dim]
        let scale = 1.0 / (head_dim as f32).sqrt();
        let attn = self.add_attention(q, k, v, mask, Some(scale), &q_transposed);

        // Transpose back to [T_q, H, head_dim]
        let attn = self.add_transpose(attn, &[1, 0, 2], &q_reshaped);

        // Reshape to [T_q, D]
        let attn = self.add_reshape(attn, &q_proj_shape);

        // Output projection
        Ok(self.add_linear(attn, out_weight, None, output_shape))
    }
}

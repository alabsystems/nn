// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Structural builder methods for `TensorBlockBuilder`.
//!
//! Extracted from `tensor_block_builder.rs` to stay under the 500-line limit.
//! Contains `add_rms_norm`, `add_stack`, `add_concat`, `add_softmax`,
//! `add_matmul`, `add_embedding`, `add_layer_norm`, `add_attention`,
//! `add_axis_select`, `add_transpose`, `add_gated_delta_net`, and `add_lstm`.
//!
//! Conv2d builder methods: `tensor_block_builder_conv2d.rs`
//! Composite attention methods: `tensor_block_builder_cross_attn.rs`

use super::*;
use crate::AttentionMask;

impl TensorBlockBuilder {
    /// Add an RMS normalization op. Returns output node ID.
    ///
    /// Normalizes input along `axis` using root mean square, then scales by `weight`.
    /// `eps` must be a scalar tensor `[1]`. Used in Transformer models (Qwen3, Llama).
    /// Maps to NY `RmsNormLayer`.
    pub fn add_rms_norm(
        &mut self,
        input: TensorNodeId,
        eps: TensorNodeId,
        axis: usize,
        weight: TensorNodeId,
        out_shape: &[usize],
    ) -> TensorNodeId {
        let id = self.alloc_id();
        self.nodes.push(TensorNode::new(
            id,
            TensorOpKind::RmsNorm {
                input,
                eps,
                axis,
                weight,
            },
            out_shape.to_vec(),
        ));
        id
    }

    /// Add a stack op that concatenates tensors along a new axis. Returns output node ID.
    ///
    /// All inputs must have the same shape. The output gains a new dimension at `axis`
    /// with size equal to the number of inputs. Used for RoPE pair assembly and
    /// multi-head attention head concatenation.
    pub fn add_stack(
        &mut self,
        inputs: &[TensorNodeId],
        axis: usize,
        out_shape: &[usize],
    ) -> TensorNodeId {
        let id = self.alloc_id();
        self.nodes.push(TensorNode::new(
            id,
            TensorOpKind::Stack {
                inputs: inputs.to_vec(),
                axis,
            },
            out_shape.to_vec(),
        ));
        id
    }

    /// Add a concat op that joins tensors along an existing axis. Returns output node ID.
    ///
    /// All inputs must have the same shape except at `axis`, where the output
    /// dimension equals the sum of input dimensions. Used for head merging in
    /// multi-head attention and KV cache concatenation.
    pub fn add_concat(
        &mut self,
        inputs: &[TensorNodeId],
        axis: usize,
        out_shape: &[usize],
    ) -> TensorNodeId {
        let id = self.alloc_id();
        self.nodes.push(TensorNode::new(
            id,
            TensorOpKind::Concat {
                inputs: inputs.to_vec(),
                axis,
            },
            out_shape.to_vec(),
        ));
        id
    }

    /// Add a softmax normalization op along one axis. Returns output node ID.
    ///
    /// Softmax normalizes the input so values sum to 1.0 along `axis`.
    /// `axis` uses Python-style indexing (negative values count from the end).
    /// Used in attention layers (query-key dot product → softmax → value weighting).
    /// Maps to NY `SoftmaxLayer` with IBP bound propagation.
    pub fn add_softmax(
        &mut self,
        input: TensorNodeId,
        axis: i32,
        out_shape: &[usize],
    ) -> TensorNodeId {
        let id = self.alloc_id();
        self.nodes.push(TensorNode::new(
            id,
            TensorOpKind::Softmax { input, axis },
            out_shape.to_vec(),
        ));
        id
    }

    pub fn add_log_softmax(
        &mut self,
        input: TensorNodeId,
        axis: i32,
        out_shape: &[usize],
    ) -> TensorNodeId {
        let id = self.alloc_id();
        self.nodes.push(TensorNode::new(
            id,
            TensorOpKind::LogSoftmax { input, axis },
            out_shape.to_vec(),
        ));
        id
    }

    /// Add a binary matrix multiplication op. Returns output node ID.
    ///
    /// Both inputs are bounded variables (unlike Linear which has a fixed weight).
    /// Left shape: `[*, M, K]`. Right shape: `[*, K, N]` (or `[*, N, K]` if
    /// `transpose_right`). Output shape: `[*, M, N]`.
    ///
    /// Set `transpose_right=true` for attention scores (`Q @ K^T`).
    /// Set `scale` to `Some(1.0 / sqrt(d_k))` for scaled dot-product attention.
    /// Maps to NY `MatMulLayer` with McCormick bilinear relaxation.
    pub fn add_matmul(
        &mut self,
        left: TensorNodeId,
        right: TensorNodeId,
        transpose_right: bool,
        scale: Option<f32>,
        out_shape: &[usize],
    ) -> TensorNodeId {
        let id = self.alloc_id();
        self.nodes.push(TensorNode::new(
            id,
            TensorOpKind::MatMul {
                left,
                right,
                transpose_right,
                scale,
            },
            out_shape.to_vec(),
        ));
        id
    }

    /// Add an embedding lookup op. Returns output node ID.
    ///
    /// Selects rows from a weight table using integer indices.
    /// Input shape: `[*]` (any dimensions, treated as row indices).
    /// Weight shape: `[num_embeddings, embedding_dim]`.
    /// Output shape: `[*, embedding_dim]`.
    ///
    /// Used for token/phoneme embeddings in transformer models (Qwen3-TTS, Kokoro).
    /// Maps to NY `GatherLayer(axis=0)` with IBP bound propagation.
    pub fn add_embedding(
        &mut self,
        input: TensorNodeId,
        weight: TensorNodeId,
        out_shape: &[usize],
    ) -> TensorNodeId {
        let id = self.alloc_id();
        self.nodes.push(TensorNode::new(
            id,
            TensorOpKind::Embedding { input, weight },
            out_shape.to_vec(),
        ));
        id
    }

    /// Add an index-select op (gather slices along `dim` using 1-D indices).
    ///
    /// Input shape: `[S0, ..., S_dim, ..., S_n]`.
    /// Indices shape: `[K]` (1-D).
    /// Output shape: `[S0, ..., K, ..., S_n]`.
    pub fn add_index_select(
        &mut self,
        input: TensorNodeId,
        indices: TensorNodeId,
        dim: usize,
        out_shape: &[usize],
    ) -> TensorNodeId {
        let id = self.alloc_id();
        self.nodes.push(TensorNode::new(
            id,
            TensorOpKind::IndexSelect {
                input,
                indices,
                dim,
            },
            out_shape.to_vec(),
        ));
        id
    }

    /// Add a gather op (N-D index lookup along one axis).
    ///
    /// Output shape == indices shape.
    pub fn add_gather(
        &mut self,
        input: TensorNodeId,
        indices: TensorNodeId,
        dim: usize,
        out_shape: &[usize],
    ) -> TensorNodeId {
        let id = self.alloc_id();
        self.nodes.push(TensorNode::new(
            id,
            TensorOpKind::Gather {
                input,
                indices,
                dim,
            },
            out_shape.to_vec(),
        ));
        id
    }

    /// Add a LayerNorm op. Returns output node ID.
    ///
    /// Normalizes input along `axis`, then scales by `weight` (gamma) and shifts
    /// by `bias` (beta). `eps` must be a scalar tensor `[1]`.
    /// Maps to NY `LayerNormLayer`.
    pub fn add_layer_norm(
        &mut self,
        input: TensorNodeId,
        eps: TensorNodeId,
        axis: usize,
        weight: TensorNodeId,
        bias: TensorNodeId,
        out_shape: &[usize],
    ) -> TensorNodeId {
        let id = self.alloc_id();
        self.nodes.push(TensorNode::new(
            id,
            TensorOpKind::LayerNorm {
                input,
                eps,
                axis,
                weight,
                bias,
            },
            out_shape.to_vec(),
        ));
        id
    }

    /// Add a monolithic self-attention op. Returns output node ID.
    ///
    /// Q `[*, T, D]`, K `[*, T_kv, D]`, V `[*, T_kv, D_v]`, output `[*, T, D_v]`.
    /// Scale `None` = auto `1/sqrt(D)`. Maps to NY `SelfAttentionLayer`.
    pub fn add_attention(
        &mut self,
        q: TensorNodeId,
        k: TensorNodeId,
        v: TensorNodeId,
        mask: AttentionMask,
        scale: Option<f32>,
        out_shape: &[usize],
    ) -> TensorNodeId {
        let id = self.alloc_id();
        self.nodes.push(TensorNode::new(
            id,
            TensorOpKind::Attention {
                q,
                k,
                v,
                mask,
                scale,
            },
            out_shape.to_vec(),
        ));
        id
    }

    /// Add an axis-select op that picks one index along an axis. Returns output node ID.
    ///
    /// Reduces rank by 1 (the selected axis is removed). Used for RoPE pair
    /// splitting and selecting specific heads in multi-head attention.
    pub fn add_axis_select(
        &mut self,
        input: TensorNodeId,
        axis: usize,
        index: usize,
        out_shape: &[usize],
    ) -> TensorNodeId {
        let id = self.alloc_id();
        self.nodes.push(TensorNode::new(
            id,
            TensorOpKind::AxisSelect { input, axis, index },
            out_shape.to_vec(),
        ));
        id
    }

    /// Add a transpose (axis permutation) op. Returns output node ID.
    ///
    /// `axes` must be a valid permutation of `[0, 1, ..., rank-1]` where
    /// `rank` is the number of dimensions in the input tensor. For example,
    /// `axes = [1, 0, 2]` permutes a `[A, B, C]` tensor to `[B, A, C]`.
    ///
    /// Used in multi-head attention: reshape `[T, H, D]` then transpose to
    /// `[H, T, D]` for per-head computation. Maps to NY `TransposeLayer`.
    pub fn add_transpose(
        &mut self,
        input: TensorNodeId,
        axes: &[usize],
        out_shape: &[usize],
    ) -> TensorNodeId {
        let id = self.alloc_id();
        self.nodes.push(TensorNode::new(
            id,
            TensorOpKind::Transpose {
                input,
                axes: axes.to_vec(),
            },
            out_shape.to_vec(),
        ));
        id
    }

    /// Add a Gated DeltaNet cell op (single time-step). Returns output node ID.
    ///
    /// Monolithic variant for NY verification. For Metal dispatch,
    /// use `decompose_gated_delta_net` which breaks this into primitives.
    pub fn add_gated_delta_net(
        &mut self,
        q: TensorNodeId,
        k: TensorNodeId,
        v: TensorNodeId,
        state: TensorNodeId,
        gate: TensorNodeId,
        beta: TensorNodeId,
        scale: f32,
        out_shape: &[usize],
    ) -> TensorNodeId {
        let id = self.alloc_id();
        self.nodes.push(TensorNode::new(
            id,
            TensorOpKind::GatedDeltaNet {
                q,
                k,
                v,
                state,
                gate,
                beta,
                scale,
            },
            out_shape.to_vec(),
        ));
        id
    }

    /// Add an LSTM cell op (single time-step). Returns output node ID.
    pub fn add_lstm(
        &mut self,
        input: TensorNodeId,
        hidden_state: TensorNodeId,
        cell_state: TensorNodeId,
        weight_ih: TensorNodeId,
        weight_hh: TensorNodeId,
        bias: Option<TensorNodeId>,
        out_shape: &[usize],
    ) -> TensorNodeId {
        let id = self.alloc_id();
        self.nodes.push(TensorNode::new(
            id,
            TensorOpKind::Lstm {
                input,
                hidden_state,
                cell_state,
                weight_ih,
                weight_hh,
                bias,
            },
            out_shape.to_vec(),
        ));
        id
    }
}

// Conv2d builder methods extracted to tensor_block_builder_conv2d.rs
#[path = "tensor_block_builder_conv2d.rs"]
mod conv2d_ops;

// Kani proof harnesses for composite tensor op builders (MatMul, Attention,
// Embedding, LSTM, Transpose, Gated DeltaNet). Part of #729 dvoice epic.
#[cfg(kani)]
#[path = "tensor_block_builder_ops_kani.rs"]
mod kani_builders;

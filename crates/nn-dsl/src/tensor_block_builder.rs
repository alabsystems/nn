// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Builder for constructing multi-op `TensorKernelDef` blocks.
//!
//! Reduces boilerplate when constructing neural network blocks like
//! the Demucs encoder (Conv1d + Snake + InstanceNorm). Node IDs and
//! shapes are managed automatically.
//!
//! # Example
//!
//! ```
//! use nn_dsl::tensor_block_builder::TensorBlockBuilder;
//! use nn_dsl::adain::build_snake_scalar_kernel;
//!
//! let snake = build_snake_scalar_kernel().expect("snake kernel");
//! let mut b = TensorBlockBuilder::new("demucs_enc");
//! let data = b.add_input("data", &[1, 64]);
//! let weight = b.add_input("weight", &[48, 1, 8]);
//! let alpha = b.add_input("alpha", &[1]);
//! let eps = b.add_input("eps", &[1]);
//!
//! let conv = b.add_conv1d(data, weight, None, 4, 2, &[48, 16]);
//! let alpha_bc = b.add_broadcast(alpha, &[48, 16]);
//! let act = b.add_elementwise(snake, &[conv, alpha_bc], &[48, 16]);
//! let norm = b.add_instance_norm(act, eps, 1, None, None, &[48, 16]);
//!
//! let def = b.build(norm).expect("valid graph");
//! assert_eq!(def.name, "demucs_enc");
//! ```

use crate::ir::KernelDef;
use crate::tensor_ir::{
    BroadcastAlignment, ReduceOp, TensorIRError, TensorIRLayerError, TensorKernelDef, TensorNode,
    TensorNodeId, TensorOpKind,
};

// Convolution and linear builder methods (add_conv1d, add_conv1d_full,
// add_conv_transpose_1d, add_linear) extracted to stay under the 500-line
// limit (Part of #1575).
#[path = "tensor_block_builder_conv.rs"]
mod conv_builders;

/// Builder for multi-op `TensorKernelDef` blocks.
#[derive(Debug)]
pub struct TensorBlockBuilder {
    name: String,
    nodes: Vec<TensorNode>,
    next_id: usize,
}

impl TensorBlockBuilder {
    /// Create a new builder with the given block name.
    #[must_use]
    pub fn new(name: &str) -> Self {
        Self {
            name: name.to_string(),
            nodes: Vec::new(),
            next_id: 0,
        }
    }

    /// Add an input node. Returns the node ID for wiring.
    pub fn add_input(&mut self, name: &str, shape: &[usize]) -> TensorNodeId {
        let id = self.alloc_id();
        self.nodes.push(TensorNode::new(
            id,
            TensorOpKind::Input {
                name: name.to_string(),
                shape: shape.to_vec(),
            },
            shape.to_vec(),
        ));
        id
    }

    /// Add an elementwise scalar kernel op. Returns output node ID.
    pub fn add_elementwise(
        &mut self,
        kernel: KernelDef,
        inputs: &[TensorNodeId],
        out_shape: &[usize],
    ) -> TensorNodeId {
        let id = self.alloc_id();
        self.nodes.push(TensorNode::new(
            id,
            TensorOpKind::Elementwise {
                kernel,
                inputs: inputs.to_vec(),
            },
            out_shape.to_vec(),
        ));
        id
    }

    /// Add a broadcast op (right-aligned, NumPy-style). Returns output node ID.
    pub fn add_broadcast(&mut self, input: TensorNodeId, target_shape: &[usize]) -> TensorNodeId {
        self.add_broadcast_aligned(input, target_shape, BroadcastAlignment::Right)
    }

    /// Add a broadcast op (left-aligned, per-channel). Returns output node ID.
    ///
    /// Use this for per-channel parameters like gamma/beta in normalization
    /// layers, where `[C]` broadcasts to `[C, T]` by aligning the channel
    /// dimension to the leftmost axis.
    pub fn add_broadcast_left(
        &mut self,
        input: TensorNodeId,
        target_shape: &[usize],
    ) -> TensorNodeId {
        self.add_broadcast_aligned(input, target_shape, BroadcastAlignment::Left)
    }

    fn add_broadcast_aligned(
        &mut self,
        input: TensorNodeId,
        target_shape: &[usize],
        alignment: BroadcastAlignment,
    ) -> TensorNodeId {
        let id = self.alloc_id();
        self.nodes.push(TensorNode::new(
            id,
            TensorOpKind::Broadcast {
                input,
                target_shape: target_shape.to_vec(),
                alignment,
            },
            target_shape.to_vec(),
        ));
        id
    }

    /// Add a reduce op (Sum or Mean) along one axis. Returns output node ID.
    ///
    /// When `keepdim` is `true`, the reduced axis is retained with size 1.
    /// When `keepdim` is `false`, the reduced axis is removed from the output shape.
    pub fn add_reduce(
        &mut self,
        input: TensorNodeId,
        op: ReduceOp,
        axis: usize,
        keepdim: bool,
        out_shape: &[usize],
    ) -> TensorNodeId {
        let id = self.alloc_id();
        self.nodes.push(TensorNode::new(
            id,
            TensorOpKind::Reduce {
                op,
                input,
                axis,
                keepdim,
            },
            out_shape.to_vec(),
        ));
        id
    }

    /// Add a binary addition op (skip connection). Returns output node ID.
    pub fn add_binary_add(
        &mut self,
        left: TensorNodeId,
        right: TensorNodeId,
        out_shape: &[usize],
    ) -> TensorNodeId {
        let id = self.alloc_id();
        self.nodes.push(TensorNode::new(
            id,
            TensorOpKind::BinaryAdd { left, right },
            out_shape.to_vec(),
        ));
        id
    }

    /// Add a binary multiplication op (element-wise, for GLU). Returns output node ID.
    pub fn add_binary_mul(
        &mut self,
        left: TensorNodeId,
        right: TensorNodeId,
        out_shape: &[usize],
    ) -> TensorNodeId {
        let id = self.alloc_id();
        self.nodes.push(TensorNode::new(
            id,
            TensorOpKind::BinaryMul { left, right },
            out_shape.to_vec(),
        ));
        id
    }

    /// Add a LayerScale op: `x * broadcast(scale)`.
    ///
    /// LayerScale is a per-channel learned scaling used in DConv blocks.
    /// `scale` should be a `[C]` tensor; it is left-broadcast to match `input`'s
    /// shape `[C, T]` (or higher rank), then element-wise multiplied.
    ///
    /// Returns the output node ID with the same shape as `input`.
    pub fn add_layer_scale(
        &mut self,
        input: TensorNodeId,
        scale: TensorNodeId,
        out_shape: &[usize],
    ) -> TensorNodeId {
        let scale_bc = self.add_broadcast_left(scale, out_shape);
        self.add_binary_mul(input, scale_bc, out_shape)
    }

    /// Add a narrow (slice) op along one axis. Returns output node ID.
    pub fn add_narrow(
        &mut self,
        input: TensorNodeId,
        axis: usize,
        start: usize,
        length: usize,
        out_shape: &[usize],
    ) -> TensorNodeId {
        let id = self.alloc_id();
        self.nodes.push(TensorNode::new(
            id,
            TensorOpKind::Narrow {
                input,
                axis,
                start,
                length,
            },
            out_shape.to_vec(),
        ));
        id
    }

    /// Add a GLU (Gated Linear Unit) decomposition along one axis.
    ///
    /// Splits the input in half along `axis`, applies sigmoid to the gate half,
    /// and multiplies: `narrow(data) * sigmoid(narrow(gate))`.
    ///
    /// The input's `axis` dimension must be even (2 * `half`). The output shape
    /// has `shape[axis] = half`.
    ///
    /// Returns the final BinaryMul output node ID.
    pub fn add_glu(
        &mut self,
        input: TensorNodeId,
        axis: usize,
        input_shape: &[usize],
    ) -> Result<TensorNodeId, TensorIRError> {
        let full_dim = input_shape[axis];
        if !full_dim.is_multiple_of(2) {
            return Err(TensorIRLayerError::GluOddDimension {
                axis,
                dim: full_dim,
            }
            .into());
        }
        let half = full_dim / 2;

        // Output shape: input shape with axis dimension halved.
        let mut out_shape: Vec<usize> = input_shape.to_vec();
        out_shape[axis] = half;

        // data = narrow(input, axis, start=0, length=half)
        let data = self.add_narrow(input, axis, 0, half, &out_shape);
        // gate = narrow(input, axis, start=half, length=half)
        let gate = self.add_narrow(input, axis, half, half, &out_shape);
        // gate_sig = sigmoid(gate)
        let gate_sig = self.add_sigmoid(gate, &out_shape);
        // output = data * sigmoid(gate)
        Ok(self.add_binary_mul(data, gate_sig, &out_shape))
    }

    /// Finalize into a `TensorKernelDef` with the given output node.
    ///
    /// Runs structural validation to catch shape mismatches, zero-dimension
    /// shapes, and invalid node references. Returns an error if the graph
    /// is invalid. (#792 AC1, #658 AC3)
    pub fn build(self, output: TensorNodeId) -> Result<TensorKernelDef, TensorIRError> {
        let def = TensorKernelDef::new(&self.name, self.nodes, output);
        def.validate()?;
        Ok(def)
    }

    fn alloc_id(&mut self) -> TensorNodeId {
        let id = TensorNodeId::new(self.next_id);
        self.next_id += 1;
        id
    }
}

// Normalization builder methods (add_instance_norm, add_group_norm_g1) extracted
// to stay under the 500-line limit (Part of #723).
#[path = "tensor_block_builder_norm.rs"]
mod norm_builders;

// Structural builder methods (add_rms_norm, add_stack, add_axis_select) extracted
// to stay under the 500-line limit (Part of #740).
#[path = "tensor_block_builder_ops.rs"]
mod ops_builders;

// Activation and padding builder methods (add_sigmoid, add_gelu, add_relu,
// add_tanh, add_zero_pad_1d) extracted to stay under the 500-line limit.
#[path = "tensor_block_builder_activations.rs"]
mod activation_builders;

// Transformer block composite builder (add_transformer_block, TransformerBlockConfig).
// Pre-norm: LayerNorm → MHA → residual → LayerNorm → FFN → residual. Part of #811.
#[path = "tensor_block_builder_transformer.rs"]
mod transformer_builders;
pub use transformer_builders::{
    CrossAttentionBlockConfig, CrossAttentionBlockWeights, TransformerBlockConfig,
    TransformerBlockWeights,
};

// Composite attention MHA builders (add_multi_head_attention, add_multi_head_cross_attention).
// Extracted from tensor_block_builder_ops.rs. Part of #779, #823.
#[path = "tensor_block_builder_cross_attn.rs"]
mod cross_attn_builders;

#[cfg(test)]
#[path = "tensor_block_builder_tests.rs"]
mod tests;

#[cfg(test)]
#[path = "tensor_block_builder_tests_extended.rs"]
mod tests_extended;

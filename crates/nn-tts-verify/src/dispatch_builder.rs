// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Ergonomic builder for `DispatchStep` plans.
//!
//! Replaces the repeated pattern of `NodeAlloc::next()` + struct literal push
//! with 1-line method calls. Subsumes the duplicated `NodeAlloc` struct.
//!
//! Part of #2218 (Kokoro epic).
//! Design: `designs/2026-03-19-dispatch-builder-dedup.md` (D1).

use nn_dsl::ir::ScalarType;
use nn_dsl::tensor_ir::{ReduceOp, TensorNodeId};
use nn_dsl::{Conv1dParams, ConvTranspose1dParams, DispatchStep};

/// Builder that owns a `Vec<DispatchStep>` and allocates `TensorNodeId`s.
///
/// Each method allocates the necessary node IDs internally and pushes
/// the constructed step. All methods use `ScalarType::F32` (the only
/// dtype used in existing dispatch plans).
pub(crate) struct DispatchBuilder {
    steps: Vec<DispatchStep>,
    next_id: usize,
}

impl DispatchBuilder {
    #[cfg(test)]
    pub(crate) fn new() -> Self {
        Self {
            steps: Vec::new(),
            next_id: 0,
        }
    }

    pub(crate) fn with_capacity(cap: usize) -> Self {
        Self {
            steps: Vec::with_capacity(cap),
            next_id: 0,
        }
    }

    /// Total number of node IDs allocated so far.
    pub(crate) fn node_count(&self) -> usize {
        self.next_id
    }

    /// Consume the builder and return the constructed steps.
    pub(crate) fn into_steps(self) -> Vec<DispatchStep> {
        self.steps
    }

    /// Allocate a single node ID without pushing a step.
    ///
    /// Use for edge-case manual step construction (e.g., strided Conv1d
    /// where `total_elements != out_ch * t_len`).
    pub(crate) fn alloc_node(&mut self) -> TensorNodeId {
        self.next()
    }

    /// Push a pre-built `DispatchStep`.
    ///
    /// Use for edge-case manual step construction.
    pub(crate) fn push_step(&mut self, step: DispatchStep) {
        self.steps.push(step);
    }

    fn next(&mut self) -> TensorNodeId {
        let id = self.next_id;
        self.next_id += 1;
        TensorNodeId::new(id)
    }

    // -- Step builders --------------------------------------------------------

    /// Push a `DispatchStep::Linear`. Allocates 4 nodes (input, weight, bias, output).
    pub(crate) fn linear(
        &mut self,
        name: impl Into<String>,
        in_features: usize,
        out_features: usize,
        batch_size: usize,
    ) -> &mut Self {
        let (input, weight, bias, output) = (self.next(), self.next(), self.next(), self.next());
        self.steps.push(DispatchStep::Linear {
            kernel_name: name.into(),
            dtype: ScalarType::F32,
            input,
            weight,
            bias: Some(bias),
            output,
            in_features,
            out_features,
            batch_size,
            total_elements: batch_size * out_features,
        });
        self
    }

    /// Push a `DispatchStep::Conv1d`. Allocates 4 nodes (input, weight, bias, output).
    pub(crate) fn conv1d(
        &mut self,
        name: impl Into<String>,
        in_ch: usize,
        out_ch: usize,
        kernel: usize,
        t_len: usize,
        stride: usize,
        padding: usize,
        dilation: usize,
    ) -> &mut Self {
        let (input, weight, bias, output) = (self.next(), self.next(), self.next(), self.next());
        self.steps.push(DispatchStep::Conv1d(Conv1dParams::new(
            name.into(),
            ScalarType::F32,
            input,
            weight,
            Some(bias),
            output,
            in_ch,
            out_ch,
            kernel,
            t_len,
            out_ch * t_len,
            stride,
            padding,
            dilation,
            1, // groups
        )));
        self
    }

    /// Push a `DispatchStep::ConvTranspose1d`. Allocates 4 nodes.
    pub(crate) fn conv_transpose1d(
        &mut self,
        name: impl Into<String>,
        in_ch: usize,
        out_ch: usize,
        kernel: usize,
        t_len: usize,
        stride: usize,
        padding: usize,
    ) -> &mut Self {
        let (input, weight, bias, output) = (self.next(), self.next(), self.next(), self.next());
        let t_out = (t_len - 1) * stride + kernel - 2 * padding;
        self.steps
            .push(DispatchStep::ConvTranspose1d(ConvTranspose1dParams::new(
                name.into(),
                ScalarType::F32,
                input,
                weight,
                Some(bias),
                output,
                in_ch,
                out_ch,
                kernel,
                t_len,
                out_ch * t_out,
                stride,
                padding,
                1, // dilation
                1, // groups
                0, // output_padding
            )));
        self
    }

    /// Push a `DispatchStep::Sigmoid`. Allocates 2 nodes (input, output).
    pub(crate) fn sigmoid(&mut self, name: impl Into<String>, total_elements: usize) -> &mut Self {
        let (input, output) = (self.next(), self.next());
        self.steps.push(DispatchStep::Sigmoid {
            kernel_name: name.into(),
            dtype: ScalarType::F32,
            input,
            output,
            total_elements,
        });
        self
    }

    /// Push a `DispatchStep::Tanh`. Allocates 2 nodes (input, output).
    pub(crate) fn tanh(&mut self, name: impl Into<String>, total_elements: usize) -> &mut Self {
        let (input, output) = (self.next(), self.next());
        self.steps.push(DispatchStep::Tanh {
            kernel_name: name.into(),
            dtype: ScalarType::F32,
            input,
            output,
            total_elements,
        });
        self
    }

    /// Push a `DispatchStep::Gelu`. Allocates 2 nodes (input, output).
    pub(crate) fn gelu(&mut self, name: impl Into<String>, total_elements: usize) -> &mut Self {
        let (input, output) = (self.next(), self.next());
        self.steps.push(DispatchStep::Gelu {
            kernel_name: name.into(),
            dtype: ScalarType::F32,
            input,
            output,
            total_elements,
        });
        self
    }

    /// Push a `DispatchStep::BinaryAdd`. Allocates 3 nodes (left, right, output).
    pub(crate) fn binary_add(
        &mut self,
        name: impl Into<String>,
        total_elements: usize,
    ) -> &mut Self {
        let (left, right, output) = (self.next(), self.next(), self.next());
        self.steps.push(DispatchStep::BinaryAdd {
            kernel_name: name.into(),
            dtype: ScalarType::F32,
            left,
            right,
            output,
            total_elements,
            broadcast: None,
        });
        self
    }

    /// Push a `DispatchStep::BinaryMul`. Allocates 3 nodes (left, right, output).
    pub(crate) fn binary_mul(
        &mut self,
        name: impl Into<String>,
        total_elements: usize,
    ) -> &mut Self {
        let (left, right, output) = (self.next(), self.next(), self.next());
        self.steps.push(DispatchStep::BinaryMul {
            kernel_name: name.into(),
            dtype: ScalarType::F32,
            left,
            right,
            output,
            total_elements,
            broadcast: None,
        });
        self
    }

    /// Push a `DispatchStep::MatMul`. Allocates 3 nodes (left, right, output).
    pub(crate) fn matmul(
        &mut self,
        name: impl Into<String>,
        m: usize,
        k: usize,
        n: usize,
        batch_size: usize,
        transpose_right: bool,
        broadcast_right: bool,
        scale: Option<f32>,
    ) -> &mut Self {
        let (left, right, output) = (self.next(), self.next(), self.next());
        self.steps.push(DispatchStep::MatMul {
            kernel_name: name.into(),
            dtype: ScalarType::F32,
            left,
            right,
            output,
            m,
            k,
            n,
            batch_size,
            transpose_right,
            broadcast_right,
            scale,
            total_elements: batch_size * m * n,
        });
        self
    }

    /// Push a `DispatchStep::Softmax`. Allocates 2 nodes (input, output).
    pub(crate) fn softmax(
        &mut self,
        name: impl Into<String>,
        axis_size: usize,
        outer_size: usize,
    ) -> &mut Self {
        let (input, output) = (self.next(), self.next());
        self.steps.push(DispatchStep::Softmax {
            kernel_name: name.into(),
            dtype: ScalarType::F32,
            input,
            output,
            axis: 1, // all existing callers use axis=1 (last dim of [outer, axis])
            axis_size,
            outer_size,
        });
        self
    }

    /// Push a `DispatchStep::Relu`. Allocates 2 nodes (input, output).
    pub(crate) fn relu(&mut self, name: impl Into<String>, total_elements: usize) -> &mut Self {
        let (input, output) = (self.next(), self.next());
        self.steps.push(DispatchStep::Relu {
            kernel_name: name.into(),
            dtype: ScalarType::F32,
            input,
            output,
            total_elements,
        });
        self
    }

    /// Push a `DispatchStep::Reduce`. Allocates 2 nodes (input, output).
    pub(crate) fn reduce(
        &mut self,
        name: impl Into<String>,
        op: ReduceOp,
        reduce_dim: usize,
        outer_size: usize,
    ) -> &mut Self {
        let (input, output) = (self.next(), self.next());
        self.steps.push(DispatchStep::Reduce {
            kernel_name: name.into(),
            op,
            dtype: ScalarType::F32,
            input,
            output,
            reduce_dim,
            outer_size,
            keepdim: false,
        });
        self
    }

    /// Push a `DispatchStep::Embedding`. Allocates 3 nodes (input, weight, output).
    pub(crate) fn embedding(
        &mut self,
        name: impl Into<String>,
        embedding_dim: usize,
        num_indices: usize,
    ) -> &mut Self {
        let (input, weight, output) = (self.next(), self.next(), self.next());
        self.steps.push(DispatchStep::Embedding {
            kernel_name: name.into(),
            dtype: ScalarType::F32,
            input,
            weight,
            output,
            embedding_dim,
            num_indices,
            total_elements: num_indices * embedding_dim,
        });
        self
    }
}

#[cfg(test)]
#[path = "dispatch_builder_tests.rs"]
mod tests;

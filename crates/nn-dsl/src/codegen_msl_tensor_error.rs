// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Error types for tensor-level MSL dispatch planning/codegen.
//!
//! Extracted from `codegen_msl_tensor.rs` to stay under the 500-line limit.

use crate::ir::IRError;
use crate::lower::LowerError;
use crate::tensor_ir::{TensorIRError, TensorNodeId};
use thiserror::Error;

/// Errors from tensor-level MSL dispatch planning/codegen.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum TensorMSLCodegenError {
    #[error(
        "reduce node {node_id:?} uses axis {axis} for shape {shape:?}, but tensor MSL currently supports only last-axis reductions"
    )]
    NonLastAxisReduce {
        node_id: TensorNodeId,
        axis: usize,
        shape: Vec<usize>,
    },

    #[error(
        "softmax node {node_id:?} uses axis {axis} for shape {shape:?}, but tensor MSL currently supports only last-axis softmax"
    )]
    NonLastAxisSoftmax {
        node_id: TensorNodeId,
        axis: usize,
        shape: Vec<usize>,
    },

    #[error("shape product overflow: {shape:?}")]
    ShapeProductOverflow { shape: Vec<usize> },

    #[error("emit_stack_kernel called with n_inputs=0 — produces invalid MSL")]
    EmptyStack,

    #[error("Metal buffer limit exceeded: kernel requires {required} buffers but Metal allows at most {max} (buffer indices 0..={max_index})")]
    BufferLimitExceeded {
        required: usize,
        max: usize,
        max_index: usize,
    },

    #[error("stride value {value} exceeds u32::MAX ({max}) — MSL uint is 32-bit")]
    StrideExceedsU32 { value: usize, max: u32 },

    #[error("invalid convolution parameter: {0}")]
    InvalidParameter(String),

    #[error("axis {axis} out of bounds for shape with rank {rank}")]
    AxisOutOfBounds { axis: usize, rank: usize },

    #[error("invalid tensor node reference {node_id:?} (graph has {graph_len} nodes)")]
    InvalidNodeRef {
        node_id: TensorNodeId,
        graph_len: usize,
    },

    #[error("unexpanded norm op '{op_name}' at node {node_id:?} — expand_norm_ops() should have decomposed this before dispatch planning")]
    UnexpandedNormOp {
        node_id: TensorNodeId,
        op_name: &'static str,
    },

    #[error("index dim {dim} out of bounds for rank {rank}")]
    InvalidDim { dim: usize, rank: usize },

    #[error("index dim {dim} has size 0 — OOB clamp would underflow")]
    EmptyDim { dim: usize },

    #[error("unsupported op '{op_name}' for MSL codegen: {reason}")]
    UnsupportedOp {
        op_name: &'static str,
        reason: &'static str,
    },

    #[error("tensor IR validation failed: {0}")]
    TensorIrValidation(#[from] TensorIRError),

    #[error("scalar kernel IR error during elementwise MSL emission: {0}")]
    ScalarCodegen(#[from] IRError),

    #[error("failed to build scalar kernel for elementwise op: {0}")]
    ScalarKernelBuild(#[from] LowerError),
}

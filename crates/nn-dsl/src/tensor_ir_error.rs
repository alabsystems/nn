// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Error types for tensor IR construction and validation.
//!
//! Extracted from `tensor_ir.rs` to keep that file under the 500-line limit.
//! All variants are re-exported via `tensor_ir.rs` -- consumer imports are unchanged.
//!
//! Layer-specific validation errors (Conv1d, LSTM, etc.) live in
//! `TensorIRLayerError` and are wrapped by `Layer(#[from] TensorIRLayerError)`.

use thiserror::Error;

use crate::ir::IRError;
use crate::kernel_error::KernelError;

use super::{TensorIRConvError, TensorIRLayerError, TensorNodeId};

/// Errors from tensor IR construction or validation.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum TensorIRError {
    #[error("tensor node {0:?} references out-of-bounds node")]
    InvalidNodeRef(TensorNodeId),

    #[error("tensor node {0:?} has forward or self reference to {1:?}")]
    ForwardRef(TensorNodeId, TensorNodeId),

    #[error("tensor node at index {expected_index} has id {found:?} (must equal TensorNodeId::new({expected_index}))")]
    MismatchedNodeId {
        found: TensorNodeId,
        expected_index: usize,
    },

    #[error("reduce axis {axis} out of bounds for input shape {shape:?}")]
    ReduceAxisOutOfBounds { axis: usize, shape: Vec<usize> },

    #[error(
        "reshape product mismatch: input has {input_product} elements, target has {target_product}"
    )]
    ReshapeProductMismatch {
        input_product: usize,
        target_product: usize,
    },

    #[error("axis_select axis {axis} out of bounds for shape {shape:?}")]
    AxisSelectOutOfBounds { axis: usize, shape: Vec<usize> },

    #[error("axis_select index {index} >= dimension size {dim} at axis {axis}")]
    AxisSelectIndexOutOfBounds {
        index: usize,
        dim: usize,
        axis: usize,
    },

    #[error("axis 0 is reserved in tensor verification path for op {op}")]
    AxisZeroReserved { op: &'static str },

    #[error("stack inputs have different shapes: expected {expected:?}, found {found:?}")]
    StackShapeMismatch {
        expected: Vec<usize>,
        found: Vec<usize>,
    },

    #[error("stack axis {axis} out of bounds for rank {rank}; valid range is 1..={rank}")]
    StackAxisOutOfBounds { axis: usize, rank: usize },

    #[error("stack must have at least one input")]
    EmptyStack,

    #[error("input tensor has empty dimension: shape {0:?}")]
    EmptyDimension(Vec<usize>),

    #[error("elementwise kernel expects {expected} params but got {got} tensor inputs")]
    ElementwiseParamMismatch { expected: usize, got: usize },

    #[error(
        "elementwise inputs have different shapes: expected {expected:?}, found {found:?} at input {index}"
    )]
    ElementwiseShapeMismatch {
        expected: Vec<usize>,
        found: Vec<usize>,
        index: usize,
    },

    #[error("broadcast target shape {target:?} is incompatible with input shape {input:?}")]
    IncompatibleBroadcast {
        input: Vec<usize>,
        target: Vec<usize>,
    },

    #[error(
        "broadcast {input:?} -> {target:?} is ambiguous: both left and right alignment match; specify alignment explicitly"
    )]
    AmbiguousBroadcast {
        input: Vec<usize>,
        target: Vec<usize>,
    },

    #[error("tensor graph has no nodes")]
    EmptyGraph,

    #[error("inner scalar kernel validation failed: {0}")]
    ScalarIrValidation(#[from] IRError),

    #[error("kernel dimension validation failed: {0}")]
    KernelValidation(#[from] KernelError),

    #[error("scalar kernel build failed: {0}")]
    ScalarKernelBuild(String),

    /// A `TraceOp` variant that `compile_trace()` cannot lower to a
    /// `TensorKernelDef`.  Returned for ops that have no TensorBlockBuilder
    /// mapping (e.g., PixelShuffle, Upsample2d, custom ops).
    #[error("unsupported trace op for compilation: {name}")]
    UnsupportedTraceOp { name: String },

    /// Transpose dims out of bounds for the given rank.
    #[error("transpose dim {dim0} or {dim1} out of bounds for rank {ndim}")]
    TransposeDimOutOfBounds {
        dim0: usize,
        dim1: usize,
        ndim: usize,
    },

    /// Permute axes are invalid (wrong length, out of bounds, or duplicates).
    #[error("invalid permute axes {axes:?} for rank {ndim}: {reason}")]
    InvalidPermuteAxes {
        axes: Vec<usize>,
        ndim: usize,
        reason: String,
    },

    /// A constant value (e.g., eps) is non-finite after f32 cast.
    #[error("non-finite constant '{name}': {value}")]
    NonFiniteConstant { name: String, value: f64 },

    /// A trace node references an input that does not exist in the computation
    /// graph. This indicates a malformed graph (e.g., from incomplete tracing
    /// or a bug in trace recording).
    #[error(
        "node '{node_name}' references missing input at index {input_idx} (input_id={input_id})"
    )]
    MissingInputNode {
        node_name: String,
        input_idx: usize,
        input_id: u64,
    },

    /// Softmax dim exceeds i32 range (Metal/GPU dispatch uses i32 for dim).
    #[error("softmax dim {dim} exceeds i32 range")]
    SoftmaxDimOverflow { dim: usize },

    /// Shape product overflows `usize`, indicating a malformed or unreasonably
    /// large model shape from the traced computation graph.
    #[error("shape product overflow: {shape:?}")]
    ShapeOverflow { shape: Vec<usize> },

    /// Weight data validation error (e.g., data length doesn't match shape product).
    #[error("weight data error: {0}")]
    WeightData(#[from] nn_core::TensorError),

    /// Layer-specific validation errors (Conv1d, LSTM, etc.).
    ///
    /// Never matched individually by callers of `TensorIRError` -- only
    /// constructed and auto-converted via the `#[from]` impl.
    #[error(transparent)]
    Layer(#[from] TensorIRLayerError),
}

/// Convenience conversion: `TensorIRConvError` → `TensorIRLayerError::Conv` → `TensorIRError::Layer`.
///
/// Allows `TensorIRConvError::Foo.into()` to produce `TensorIRError` directly,
/// which conv shape inference functions rely on.
impl From<TensorIRConvError> for TensorIRError {
    fn from(e: TensorIRConvError) -> Self {
        Self::Layer(TensorIRLayerError::from(e))
    }
}

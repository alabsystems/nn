// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Error types for tensor dispatch execution.

use nn_dsl::TensorNodeId;

use crate::error::MetalError;

/// Errors from tensor dispatch execution.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum TensorDispatchError {
    /// Underlying Metal backend error.
    #[error("Metal error: {0}")]
    Metal(#[from] MetalError),

    /// MSL code generation failed.
    #[error("DSL codegen error: {0}")]
    Codegen(#[from] nn_dsl::TensorMSLCodegenError),

    /// Kernel IR construction or validation error.
    #[error("DSL IR error: {0}")]
    Ir(#[from] nn_dsl::ir::IRError),

    /// A dispatch step references a node whose buffer was never allocated.
    #[error("missing buffer for node {0:?} — step references a node that was not produced")]
    MissingBuffer(TensorNodeId),

    /// A named input expected by the kernel was not provided by the caller.
    #[error("missing input '{name}' — expected by tensor kernel but not provided")]
    MissingInput { name: String },

    /// Attempted to extract a scalar `KernelDef` from a non-elementwise node.
    #[error(
        "node {node_id:?} is not an Elementwise node — cannot extract inner KernelDef for MSL"
    )]
    NotElementwise { node_id: TensorNodeId },

    /// Dispatch step variant not yet implemented.
    #[error("unsupported dispatch step variant: {step:?}")]
    UnsupportedStep { step: String },

    /// Output shape dimensions overflow `usize` when multiplied.
    #[error("shape product overflow: dimensions {shape:?} overflow usize")]
    ShapeOverflow { shape: Vec<usize> },

    /// Node index exceeds the kernel's node count.
    #[error("node index {index} out of bounds (kernel has {len} nodes)")]
    NodeIndexOutOfBounds { index: usize, len: usize },

    /// The `dtype` parameter does not match the element type's `scalar_type()`.
    ///
    /// This indicates a caller bug: the generic type parameter `E` implies one
    /// scalar type, but the `dtype` argument specifies a different one. MSL
    /// codegen would produce kernels for `dtype` while the CPU side reads
    /// buffers as `E`, causing silent data corruption.
    #[error("dtype mismatch: caller element type implies {expected:?} but dtype parameter is {actual:?}")]
    DtypeMismatch {
        expected: nn_dsl::ir::ScalarType,
        actual: nn_dsl::ir::ScalarType,
    },

    /// A referenced kernel node was not found in the expected context.
    #[error("kernel node {id:?} not found in {context}")]
    KernelNodeNotFound {
        id: TensorNodeId,
        context: &'static str,
    },

    /// A `DispatchInput::Gpu` buffer is smaller than the kernel node's expected size.
    ///
    /// This prevents silent GPU out-of-bounds reads when a previous dispatch
    /// produced a buffer of unexpected size (due to shape computation bugs,
    /// dispatch plan errors, or caller errors).
    #[error("GPU buffer size mismatch for input '{name}': expected >= {expected} bytes, got {actual} bytes")]
    BufferSizeMismatch {
        name: String,
        expected: usize,
        actual: usize,
    },
}

impl From<TensorDispatchError> for nn_core::TensorError {
    fn from(e: TensorDispatchError) -> Self {
        // Forward inner MetalError directly to avoid double "Metal error:" prefix.
        if let TensorDispatchError::Metal(inner) = e {
            return inner.into();
        }
        let msg = e.to_string();
        Self::backend_failure_with_source(
            nn_core::BackendDomain::Metal,
            nn_core::BackendErrorKind::DispatchFailed,
            msg,
            e,
        )
    }
}

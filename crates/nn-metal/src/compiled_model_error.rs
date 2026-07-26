// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Error type for compiled model execution.

use nn_core::{BackendDomain, BackendErrorKind, DType, TensorError};

/// Error during compiled model execution.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum CompiledModelError {
    /// Wrong number of input tensors provided.
    #[error("expected {expected} inputs, got {got}")]
    InputCountMismatch { expected: usize, got: usize },

    /// A dispatch step failed.
    #[error("dispatch step {step_idx} failed: {reason}")]
    DispatchFailed { step_idx: usize, reason: String },

    /// The compiled plan is empty.
    #[error("compiled plan is empty")]
    EmptyPlan,

    /// Trace compilation failed.
    #[error("trace compilation failed")]
    CompileFailed(#[from] nn_dsl::TensorIRError),

    /// Weight upload to GPU failed.
    #[error("weight upload for step {step_idx} '{name}' failed: {reason}")]
    WeightUploadFailed {
        step_idx: usize,
        name: String,
        reason: String,
    },

    /// An input DynTensor is not on GPU.
    #[error("input {index} is not a GPU tensor")]
    InputNotGpu { index: usize },

    /// Input tensor shape does not match the traced graph.
    #[error("input {index} shape mismatch: expected {expected:?}, got {got:?}")]
    ShapeMismatch {
        index: usize,
        expected: Vec<usize>,
        got: Vec<usize>,
    },

    /// Input tensor dtype does not match the traced graph.
    #[error("input {index} dtype mismatch: expected {expected:?}, got {got:?}")]
    DtypeMismatch {
        index: usize,
        expected: DType,
        got: DType,
    },

    /// Graph has compiled steps but no output node.
    #[error("graph has compiled steps but no output node")]
    MissingOutputNode,

    /// Invalid configuration (e.g., conflicting options).
    #[error("invalid config: {reason}")]
    InvalidConfig { reason: String },

    /// Verification of the compiled model graph failed.
    #[cfg(feature = "verify")]
    #[error("verification failed: {0}")]
    VerifyFailed(#[from] nn_verify::VerifyError),

    /// Certification of the compiled model graph failed.
    #[cfg(feature = "verify")]
    #[error("certification failed: {0}")]
    CertifyFailed(#[from] nn_verify::CertifyError),
}

impl From<CompiledModelError> for TensorError {
    fn from(e: CompiledModelError) -> Self {
        let msg = e.to_string();
        Self::backend_failure_with_source(
            BackendDomain::Metal,
            BackendErrorKind::Other,
            msg,
            e,
        )
    }
}

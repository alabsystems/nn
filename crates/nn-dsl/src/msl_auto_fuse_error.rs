// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Error types for fused MSL codegen.
//!
//! Extracted from `msl_auto_fuse.rs` for the 500-line limit.
//! Part of #3518.

/// Errors from fused MSL codegen.
#[derive(Debug, Clone, thiserror::Error)]
#[non_exhaustive]
pub enum FusedMslError {
    /// KernelDef IR validation failed.
    #[error("IR validation failed: {0}")]
    IrValidation(String),

    /// Input shape count does not match kernel parameter count.
    #[error("shape count ({shapes}) != param count ({params})")]
    ShapeParamMismatch { shapes: usize, params: usize },

    /// Buffer count exceeds Metal hardware limit.
    #[error("buffer count ({required}) exceeds Metal limit ({max})")]
    BufferLimitExceeded { required: usize, max: usize },

    /// Shape stride computation overflowed.
    #[error("shape overflow in {context}")]
    ShapeOverflow { context: String },

    /// Invalid node reference in IR.
    #[error("invalid node reference: index {0}")]
    InvalidNodeRef(usize),

    /// Invalid parameter reference in IR.
    #[error("invalid param reference: index {0}")]
    InvalidParamRef(usize),

    /// MSL codegen helper error.
    #[error("MSL codegen error: {0}")]
    MslCodegen(String),
}

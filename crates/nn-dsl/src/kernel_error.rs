// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Kernel construction and runtime validation errors.
//!
//! `KernelError` covers the 11 error variants that arise from kernel scalar
//! functions, reference implementations, and bounds validation — as opposed
//! to AST-to-IR lowering errors which remain in [`LowerError`](crate::lower::LowerError).
//!
//! Part of #584 (LowerError domain split).

use thiserror::Error;

/// Errors from kernel construction, scalar evaluation, and bounds validation.
///
/// These variants are used by the 9+ scalar kernel functions, `kernel_util.rs`
/// helpers, and bounds functions. They are distinct from AST-lowering errors
/// (`UnsupportedBinOp`, `SelfParam`, etc.) which only arise during
/// `Lowerer::lower_fn`.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum KernelError {
    #[error("invalid dimension: {name} must be positive, got {value}")]
    InvalidDimension { name: &'static str, value: usize },
    #[error("shape mismatch: expected length {expected}, got {got}")]
    ShapeMismatch { expected: usize, got: usize },
    #[error("dimension overflow: {dims}")]
    DimensionOverflow { dims: String },
    #[error("invalid eps: {value} (must be strictly positive)")]
    InvalidEps { value: f32 },
    #[error("dimension {name}={value} exceeds f32 precision limit (2^24 = 16777216)")]
    DimensionExceedsF32Precision { name: &'static str, value: usize },
    #[error("bound contains non-finite value: {value}")]
    NonFiniteBound { value: f32 },
    #[error("input contains non-finite value: {name}={value}")]
    NonFiniteInput { name: &'static str, value: f32 },
    #[error("output contains non-finite value: {name}={value}")]
    NonFiniteOutput { name: &'static str, value: f32 },
    #[error("slice contains non-finite value: {name}[{index}]={value}")]
    NonFiniteSliceElement {
        name: &'static str,
        index: usize,
        value: f32,
    },
    #[error("output slice contains non-finite value at index {index}: {value}")]
    NonFiniteSliceOutput { index: usize, value: f32 },
    #[error("inverted bounds: lower ({lower}) > upper ({upper})")]
    InvertedBounds { lower: f32, upper: f32 },
    #[error("invalid parameter: {name} must be {constraint}, got {value}")]
    InvalidParam {
        name: &'static str,
        constraint: &'static str,
        value: f32,
    },
}

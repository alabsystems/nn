// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Error types for nn-autodiff.

/// Errors from automatic differentiation operations.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum AutodiffError {
    /// The loss tensor must be a scalar (numel == 1) for backward().
    #[error("backward() requires scalar loss, got shape {shape:?}")]
    NonScalarLoss { shape: Vec<usize> },

    /// The loss value is non-finite (NaN or Inf).
    /// A non-finite loss produces garbage gradients for all variables, so
    /// backward() rejects early to avoid wasting computation.
    #[error("backward() requires finite loss value, got NaN or Inf")]
    NonFiniteLoss,

    /// An op variant does not have a backward rule implemented.
    #[error("no backward rule for op: {0}")]
    UnsupportedBackward(String),

    /// Tensor operation error from nn-core.
    #[error(transparent)]
    Tensor(#[from] nn_core::TensorError),

    /// Shape mismatch (e.g., VarMap retrieval with wrong dims).
    #[error("shape mismatch: expected {expected:?}, got {got:?}")]
    ShapeMismatch {
        expected: Vec<usize>,
        got: Vec<usize>,
    },

    /// Invalid dropout probability (must be in [0, 1)).
    #[error("dropout probability must be in [0, 1), got {p}")]
    Dropout { p: f64 },

    /// Tensor data is not contiguous (required for pool operations).
    #[error("{op}: tensor not contiguous")]
    NotContiguous { op: &'static str },

    /// Input tensor has wrong rank for the operation.
    #[error("{op}: expected {expected}D input, got {actual}D")]
    WrongInputRank {
        op: &'static str,
        expected: usize,
        actual: usize,
    },

    /// Invalid layer configuration parameter.
    #[error("{op}: {reason}")]
    InvalidConfig { op: &'static str, reason: String },

    /// DType mismatch (e.g., VarMap retrieval with wrong dtype).
    #[error("dtype mismatch for '{name}': expected {expected:?}, got {got:?}")]
    DTypeMismatch {
        name: String,
        expected: nn_core::DType,
        got: nn_core::DType,
    },

    /// Checkpoint contains non-finite (NaN/Inf) values.
    #[error("tensor '{name}' contains {count} non-finite (NaN/Inf) values")]
    NonFiniteCheckpoint { name: String, count: usize },

    /// MatMul backward requires operands with rank >= 2.
    #[error("MatMul backward requires rank >= 2, got a.rank={rank_a}, b.rank={rank_b}")]
    MatMulRankTooLow { rank_a: usize, rank_b: usize },

    /// Empty sequence passed to sequential operation (e.g., LSTM forward_seq).
    #[error("{op}: empty sequence (seq_len=0)")]
    EmptySequence { op: &'static str },

    /// RwLock was poisoned (a thread panicked while holding the lock).
    #[error("Var RwLock poisoned: {context}")]
    LockPoisoned { context: &'static str },

    /// Flat index exceeds u32::MAX in pooling index tensor.
    #[error("{op}: flat index {index} exceeds u32::MAX ({max})")]
    IndexOverflow {
        op: &'static str,
        index: usize,
        max: u32,
    },

    /// Backward input contains NaN/Inf, which would produce silently wrong
    /// gradients (e.g., Maximum/Minimum backward drops gradient for NaN
    /// elements because all IEEE 754 comparisons return false for NaN).
    #[error("{op} backward: input contains non-finite values")]
    NonFiniteBackwardInput { op: &'static str },
}

/// Convenience result type for autodiff operations.
pub type Result<T> = std::result::Result<T, AutodiffError>;

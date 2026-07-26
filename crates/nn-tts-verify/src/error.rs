// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Error types for TTS audio verification.

use thiserror::Error;

/// Structured error kinds for codec embedding algebra operations.
#[derive(Error, Debug)]
#[non_exhaustive]
pub enum CodecAlgebraKind {
    /// A required parameter was zero or invalid.
    #[error("{param} must be > 0")]
    InvalidParam { param: &'static str },
    /// An input was empty when at least one element was required.
    #[error("{what}")]
    EmptyInput { what: &'static str },
    /// Codebook tensor has wrong rank.
    #[error("codebook must be 2D, got rank {rank}")]
    RankMismatch { rank: usize },
    /// Codebook shape does not match the expected dimensions.
    #[error("codebook {index} shape mismatch")]
    CodebookShapeMismatch { index: usize },
    /// Number of token levels does not match the space configuration.
    #[error("expected {expected} levels, got {got}")]
    LevelCount { expected: usize, got: usize },
    /// Token sequence lengths differ across levels or from expected.
    #[error("level {level}: expected {expected} tokens, got {got}")]
    SequenceLengthMismatch {
        level: usize,
        expected: usize,
        got: usize,
    },
    /// A token index exceeds the vocabulary size.
    #[error("token {token} >= vocab_size {vocab_size}")]
    TokenOutOfRange { token: u32, vocab_size: usize },
    /// Interpolation alpha is not in [0.0, 1.0].
    #[error("alpha must be in [0.0, 1.0], got {alpha}")]
    AlphaOutOfRange { alpha: f32 },
    /// Embedding tensor has the wrong shape.
    #[error("embedding must be [seq_len, {expected_dim}]")]
    EmbeddingShape { expected_dim: usize },
}

/// Structured error kinds for DSP computation failures.
#[derive(Error, Debug)]
#[non_exhaustive]
pub enum DspErrorKind {
    /// A required DSP parameter was zero or invalid.
    #[error("{param}")]
    InvalidParam { param: &'static str },
    /// Not enough samples for the requested operation.
    #[error("insufficient samples for {operation}: need {needed}, got {got}")]
    InsufficientSamples {
        operation: &'static str,
        needed: usize,
        got: usize,
    },
    /// An input was empty when data was required.
    #[error("{what}")]
    EmptyInput { what: &'static str },
    /// A size or dimension constraint was violated.
    #[error("{what}: expected {expected}, got {got}")]
    SizeMismatch {
        what: &'static str,
        expected: usize,
        got: usize,
    },
    /// A general DSP computation error.
    #[error("{what}")]
    Computation { what: &'static str },
}

/// Structured error kinds for invalid configuration or parameters.
#[derive(Error, Debug)]
#[non_exhaustive]
pub enum InvalidConfigKind {
    /// Parameter is non-finite (NaN or Inf).
    #[error("{param} must be finite")]
    NonFinite { param: &'static str },
    /// Parameter must be finite and positive.
    #[error("{param} must be finite and positive")]
    NonPositive { param: &'static str },
    /// Lower bound is not less than upper bound for a range parameter.
    #[error("{param}: lower bound must be less than upper")]
    RangeInverted { param: &'static str },
    /// A general configuration constraint was violated.
    #[error("{what}")]
    Constraint { what: &'static str },
}

/// Errors from TTS quality verification.
#[derive(Error, Debug)]
#[non_exhaustive]
pub enum TtsVerifyError {
    /// Input audio is empty (zero samples).
    #[error("Empty audio input")]
    EmptyInput,

    /// Sample rate is zero or invalid.
    #[error("Invalid sample rate: {0}")]
    InvalidSampleRate(u32),

    /// Reference and candidate audio have different lengths.
    #[error("Length mismatch: candidate={candidate}, reference={reference}")]
    LengthMismatch { candidate: usize, reference: usize },

    /// Input audio contains NaN or Inf values.
    #[error("Non-finite audio samples: {count} NaN/Inf values")]
    NonFiniteInput { count: usize },

    /// DSP computation failed (FFT, autocorrelation, windowing, etc.).
    #[error("DSP error: {0}")]
    Dsp(DspErrorKind),

    /// Pipeline requires at least 2 stages for composition verification.
    #[error("Pipeline requires at least 2 stages, got {count}")]
    InsufficientStages {
        /// Number of stages provided.
        count: usize,
    },

    /// Codec embedding operation failed.
    #[error("Codec algebra error: {0}")]
    CodecAlgebra(CodecAlgebraKind),

    /// Invalid configuration or parameter.
    #[error("Invalid config: {0}")]
    InvalidConfig(InvalidConfigKind),

    /// An operation failed with a source error.
    ///
    /// Used when wrapping errors from CROWN propagation, bounds construction,
    /// or other operations that carry their own error type.
    #[error("{context}")]
    OperationFailed {
        /// Static description of what was being attempted.
        context: &'static str,
        /// The underlying error.
        #[source]
        source: Box<dyn std::error::Error + Send + Sync>,
    },

    /// Array dimensions do not match expected shape.
    #[error("Dimension mismatch in {context}: expected {expected}, got {actual}")]
    DimensionMismatch {
        /// Expected number of elements.
        expected: usize,
        /// Actual number of elements.
        actual: usize,
        /// Description of which input was wrong.
        context: &'static str,
    },

    /// Underlying tensor operation failed.
    #[error("Tensor error: {0}")]
    Tensor(#[from] nn_core::TensorError),

    /// Verification failed and the rejection policy is [`Reject`].
    ///
    /// The embedded [`Certificate`] contains per-check details so the caller
    /// can inspect which hard bounds or quality metrics failed.
    ///
    /// [`Reject`]: crate::config::RejectionPolicy::Reject
    /// [`Certificate`]: crate::certificate::Certificate
    ///
    /// Part of #3781, #3760.
    #[error("Verification rejected: certificate overall_passed=false")]
    VerificationRejected {
        /// The full certificate with per-check results.
        ///
        /// Boxed to keep `TtsVerifyError` small: the certificate is far larger
        /// than every other variant, so storing it inline would bloat all
        /// `Result<_, TtsVerifyError>` returns.
        cert: Box<crate::certificate::Certificate>,
    },
}

/// Validate that a floating-point parameter is finite (rejects NaN and Inf).
pub fn validate_finite(value: f64, name: &'static str) -> Result<(), TtsVerifyError> {
    if !value.is_finite() {
        return Err(TtsVerifyError::InvalidConfig(
            InvalidConfigKind::NonFinite { param: name },
        ));
    }
    Ok(())
}

/// Validate that a floating-point parameter is finite and positive.
pub fn validate_finite_positive(value: f64, name: &'static str) -> Result<(), TtsVerifyError> {
    if !value.is_finite() || value <= 0.0 {
        return Err(TtsVerifyError::InvalidConfig(
            InvalidConfigKind::NonPositive { param: name },
        ));
    }
    Ok(())
}

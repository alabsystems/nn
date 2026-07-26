// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Error types for Silero VAD model execution.
//!
//! Extracted from `silero_vad.rs` to keep the main module under 500 lines.

use crate::tensor_dispatch::TensorDispatchError;

/// Errors from Silero VAD execution.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum SileroVadError {
    /// STFT computation failed.
    #[error("STFT error: {0}")]
    Stft(#[from] crate::stft::StftError),

    /// Metal tensor dispatch failed.
    #[error("dispatch error: {0}")]
    Dispatch(#[from] TensorDispatchError),

    /// Weight tensor has wrong size.
    #[error("weight {name} size mismatch: expected {expected}, got {actual}")]
    WeightSize {
        name: &'static str,
        expected: usize,
        actual: usize,
    },

    /// Audio chunk has wrong length.
    #[error("audio length mismatch: expected {expected}, got {actual}")]
    AudioLength { expected: usize, actual: usize },

    /// Internal output has unexpected length.
    #[error("output length mismatch in {stage}: expected {expected}, got {actual}")]
    OutputLength {
        stage: &'static str,
        expected: usize,
        actual: usize,
    },

    /// Audio chunk contains non-finite values (NaN or Inf).
    #[error("audio contains {count} non-finite value(s) (first at index {first_index})")]
    NonFiniteAudio { count: usize, first_index: usize },

    /// Combined LSTM bias (bias_ih + bias_hh) contains non-finite values.
    #[error("LSTM combined bias has {count} non-finite value(s) (first at index {first_index})")]
    NonFiniteBias { count: usize, first_index: usize },

    /// Encoder block produced non-finite output (NaN or Inf propagation).
    #[error("encoder block {block} output has {count} non-finite value(s)")]
    NonFiniteEncoder { block: usize, count: usize },

    /// Output probability is non-finite (NaN or Inf from LSTM/output stage).
    #[error("output probability is non-finite: {value}")]
    NonFiniteOutput { value: f32 },

    /// LSTM output (h_new or c_new) contains non-finite values.
    ///
    /// Indicates numerical instability in LSTM gate activations (e.g., from
    /// extreme encoder output, GPU precision issues, or weight corruption).
    /// Caught immediately after LSTM dispatch to prevent NaN propagation
    /// across streaming chunks.
    #[error("LSTM output has {count} non-finite value(s)")]
    NonFiniteLstmState { count: usize },

    /// Incoming streaming state contains non-finite values.
    ///
    /// `SileroVadState` has public fields, so callers can construct states
    /// with NaN/Inf. This check prevents corrupted state from silently
    /// poisoning the LSTM and all subsequent chunks.
    #[error("incoming state {field} has {count} non-finite value(s)")]
    NonFiniteInputState { field: &'static str, count: usize },

    /// Streaming state has wrong dimensions.
    #[error("state {field} length mismatch: expected {expected}, got {actual}")]
    StateDimension {
        field: &'static str,
        expected: usize,
        actual: usize,
    },

    /// Tensor IR construction error (e.g., invalid LSTM dimensions).
    #[error("tensor IR error: {0}")]
    TensorIR(#[from] nn_dsl::tensor_ir::TensorIRError),

    /// Weight loading error (missing tensor or byte alignment).
    #[error("weight load error: {0}")]
    WeightLoad(#[from] crate::safetensors::WeightError),

    /// Weight tensor contains NaN or Inf values.
    #[error("weight tensor '{name}' has {count} non-finite value(s)")]
    NonFiniteWeight { name: String, count: usize },

    /// SVAD binary format error during weight loading.
    #[error("SVAD format error: {0}")]
    SvadFormat(String),

    /// Safetensors format error during weight loading.
    #[error("safetensors format error: {0}")]
    SafetensorsFormat(String),

    /// I/O error during weight file reading.
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    /// GPU buffer allocation failed during weight upload.
    #[error("GPU buffer allocation failed: {0}")]
    GpuBufferAlloc(String),

    /// Conv dimension computation error (from builder).
    #[error("conv dim error: {0}")]
    ConvDim(#[from] nn_models::DemucsBuilderError),
}

impl From<SileroVadError> for nn_core::TensorError {
    fn from(e: SileroVadError) -> Self {
        let msg = e.to_string();
        Self::backend_failure_with_source(
            nn_core::BackendDomain::Metal,
            nn_core::BackendErrorKind::Other,
            msg,
            e,
        )
    }
}

// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Error type for [`CompiledKokoro`](super::CompiledKokoro) operations.
//!
//! Extracted from `compiled_kokoro.rs` for 450-line compliance.

use nn_core::TensorError;
use nn_dsl::{TensorIRError, TensorMSLCodegenError};
use nn_models::kokoro_error::KokoroError;
use nn_tts_verify::TtsVerifyError;

/// Error during CompiledKokoro operations.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum CompiledKokoroError {
    /// Speed must be positive and finite.
    #[error("invalid speed {value}: must be positive and finite")]
    InvalidSpeed { value: f32 },

    /// Segment compilation failed.
    #[error("segment '{segment}' compilation failed: {source}")]
    SegmentCompileFailed {
        segment: &'static str,
        #[source]
        source: Box<TensorError>,
    },

    /// Segment execution failed.
    #[error("segment '{segment}' execution failed: {source}")]
    SegmentExecutionFailed {
        segment: &'static str,
        #[source]
        source: Box<TensorError>,
    },

    /// Segment produced fewer outputs than expected.
    #[error("segment '{segment}' output count mismatch: expected {expected}, got {actual}")]
    OutputCountMismatch {
        segment: &'static str,
        expected: usize,
        actual: usize,
    },

    /// Weight loading failed.
    #[error("weight loading failed: {source}")]
    WeightLoadFailed {
        #[source]
        source: Box<TensorError>,
    },

    /// Audio quality verification could not be executed.
    #[error("audio verification failed: {source}")]
    VerificationFailed {
        #[source]
        source: Box<TtsVerifyError>,
    },

    /// MSL pre-compilation failed for a segment.
    #[error("precompile '{segment}' (shape={shape_key}): {source}")]
    PrecompileFailed {
        segment: &'static str,
        shape_key: usize,
        #[source]
        source: Box<TensorError>,
    },

    /// Trace-to-IR compilation failed during pre-compilation.
    #[error("precompile '{segment}' (shape={shape_key}): IR compile failed: {source}")]
    PrecompileCompileFailed {
        segment: &'static str,
        shape_key: usize,
        #[source]
        source: Box<TensorIRError>,
    },

    /// MSL code generation failed during pre-compilation.
    #[error("precompile '{segment}' (shape={shape_key}): MSL codegen failed: {source}")]
    PrecompileMslCodegenFailed {
        segment: &'static str,
        shape_key: usize,
        #[source]
        source: Box<TensorMSLCodegenError>,
    },

    /// GPU iSTFT computation failed.
    #[error("GPU iSTFT failed")]
    GpuIstftFailed {
        #[source]
        source: Box<TensorError>,
    },

    /// IO error during pre-compilation.
    #[error("precompile IO: {0}")]
    PrecompileIo(#[from] std::io::Error),

    /// Tracing was not active when `record_input` was called.
    #[error("record_input: tracing not active")]
    TracingNotActive,

    /// Segment cache lookup failed after ensure call.
    #[error("segment '{segment}' cache miss after ensure")]
    SegmentCacheMiss { segment: &'static str },

    /// STFT/iSTFT basis initialization failed.
    #[error("{component} basis init failed: {source}")]
    BasisInitFailed {
        component: &'static str,
        #[source]
        source: Box<TensorError>,
    },

    /// Model configuration is invalid.
    #[error("invalid Kokoro config: {source}")]
    InvalidConfig {
        #[source]
        source: Box<KokoroError>,
    },

    /// Input validation failed (length mismatches, empty inputs, etc.).
    #[error("invalid input: {0}")]
    InvalidInput(String),

    /// Model weights were released via `release_model_weights()`.
    /// New input shapes cannot be compiled after release (#3079).
    #[error("model weights released — cannot compile new shapes (call precompile_shapes first)")]
    WeightsReleased,

    /// `release_model_weights()` requires sole ownership of shared state.
    /// Other instances created by `clone_dispatch()` must be dropped first.
    #[error("cannot release weights: shared state has multiple owners (drop clone_dispatch instances first)")]
    SharedOwnership,

    /// PeepholeConfig loading failed (IO or JSON parse error).
    /// Part of #3828 Phase 2B.
    #[error("peephole config load failed: {0}")]
    ConfigLoad(String),

    /// Deployment certificate generation failed.
    /// Part of #4254.
    #[cfg(feature = "verify")]
    #[error("deployment certificate generation failed: {reason}")]
    CertificateGenerationFailed {
        /// Human-readable reason for the failure.
        reason: String,
    },

    /// An underlying tensor operation failed.
    #[error(transparent)]
    Tensor(Box<TensorError>),
}

impl From<TensorError> for CompiledKokoroError {
    fn from(e: TensorError) -> Self {
        Self::Tensor(Box::new(e))
    }
}

impl From<KokoroError> for CompiledKokoroError {
    fn from(e: KokoroError) -> Self {
        match e {
            KokoroError::InvalidSpeed { value } => Self::InvalidSpeed { value },
            KokoroError::InvalidInput(msg) => Self::InvalidInput(msg),
            other => Self::InvalidConfig {
                source: Box::new(other),
            },
        }
    }
}

impl From<CompiledKokoroError> for TensorError {
    fn from(e: CompiledKokoroError) -> Self {
        match e {
            CompiledKokoroError::Tensor(te) => *te,
            CompiledKokoroError::SegmentCompileFailed { source, .. }
            | CompiledKokoroError::SegmentExecutionFailed { source, .. }
            | CompiledKokoroError::WeightLoadFailed { source }
            | CompiledKokoroError::PrecompileFailed { source, .. }
            | CompiledKokoroError::GpuIstftFailed { source }
            | CompiledKokoroError::BasisInitFailed { source, .. } => *source,
            other => Self::Unsupported(other.to_string()),
        }
    }
}

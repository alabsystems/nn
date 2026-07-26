// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Error and weight types for the Demucs spectral encoder.
//!
//! Extracted from `demucs_spectral_encoder.rs` for the 500-line limit.
//!
//! Part of #902 (code-health extraction).

use std::borrow::Cow;

use crate::TensorDispatchError;

// ---------------------------------------------------------------------------
// Error type
// ---------------------------------------------------------------------------

/// Errors from Demucs spectral encoder construction or forward pass.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum DemucsSpectralEncoderError {
    /// Weight tensor has wrong element count.
    #[error("weight '{name}' expected {expected} elements, got {actual}")]
    WeightSize {
        name: Cow<'static, str>,
        expected: usize,
        actual: usize,
    },

    /// GPU dispatch failed.
    #[error("dispatch error: {0}")]
    Dispatch(#[from] TensorDispatchError),

    /// Dimension mismatch (construction or dispatch output).
    #[error("stage '{stage}' expected {expected} elements, got {actual}")]
    DimMismatch {
        stage: String,
        expected: usize,
        actual: usize,
    },

    /// Non-finite values in input data.
    #[error("non-finite input at block {block}: {count} NaN/Inf values")]
    NonFiniteInput { block: usize, count: usize },

    /// Frequency embedding dimension mismatch.
    #[error("freq embedding: {field} {actual} != expected {expected}")]
    FreqEmbMismatch {
        field: &'static str,
        expected: usize,
        actual: usize,
    },

    /// Tensor IR construction error.
    #[error("tensor IR error: {0}")]
    TensorIr(#[from] nn_dsl::TensorIRError),
}

impl From<nn_models::DemucsBuilderError> for DemucsSpectralEncoderError {
    fn from(e: nn_models::DemucsBuilderError) -> Self {
        match e {
            nn_models::DemucsBuilderError::WeightSize {
                name,
                expected,
                actual,
            } => Self::WeightSize {
                name,
                expected,
                actual,
            },
            nn_models::DemucsBuilderError::InvalidConvDim { msg } => Self::DimMismatch {
                stage: msg.into_owned(),
                expected: 0,
                actual: 0,
            },
            nn_models::DemucsBuilderError::BlockCountMismatch {
                context,
                expected,
                actual,
            } => Self::WeightSize {
                name: context,
                expected,
                actual,
            },
            nn_models::DemucsBuilderError::Conv1dOutLen(e) => Self::DimMismatch {
                stage: format!("conv1d output length: {e}"),
                expected: 0,
                actual: 0,
            },
            _ => Self::DimMismatch {
                stage: e.to_string(),
                expected: 0,
                actual: 0,
            },
        }
    }
}

impl From<DemucsSpectralEncoderError> for nn_core::TensorError {
    fn from(e: DemucsSpectralEncoderError) -> Self {
        let msg = e.to_string();
        Self::backend_failure_with_source(
            nn_core::BackendDomain::Metal,
            nn_core::BackendErrorKind::Other,
            msg,
            e,
        )
    }
}

// ---------------------------------------------------------------------------
// Weight types (re-exported from nn-models)
// ---------------------------------------------------------------------------

pub use nn_models::demucs_spectral_weights::DemucsSpectralEncoderWeights;

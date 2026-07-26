// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Error types for the gpt-oss-20b (Chroma Context-1) model.

use nn_core::TensorError;

/// Errors from gpt-oss model construction or forward pass.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum GptOssError {
    /// Configuration validation failure (zero heads, non-finite eps, etc.).
    #[error("invalid config: {reason}")]
    InvalidConfig { reason: String },

    /// Forward pass input validation failure (mismatched input_ids/positions,
    /// wrong hidden_size for pre-computed embeddings).
    #[error("invalid input: {reason}")]
    InvalidInput { reason: String },

    /// KV cache layer count does not match model layer count.
    #[error("cache mismatch: cache has {cache_layers} layers, model has {model_layers}")]
    CacheMismatch {
        cache_layers: usize,
        model_layers: usize,
    },

    /// Non-finite values (NaN/Inf) detected in model output.
    #[error("non-finite output ({stage}): {count} NaN/Inf values")]
    NonFiniteOutput { stage: &'static str, count: usize },

    /// Weight loading error (missing tensor, shape mismatch).
    #[error("weight load: {reason}")]
    WeightLoad { reason: String },

    /// Passthrough for underlying tensor errors.
    #[error(transparent)]
    Tensor(#[from] TensorError),
}

impl From<GptOssError> for TensorError {
    fn from(e: GptOssError) -> Self {
        match e {
            GptOssError::Tensor(te) => te,
            other => {
                let msg = other.to_string();
                Self::backend_failure_with_source(
                    nn_core::BackendDomain::Device,
                    nn_core::BackendErrorKind::Other,
                    msg,
                    other,
                )
            }
        }
    }
}

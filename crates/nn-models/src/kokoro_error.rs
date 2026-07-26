// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Error types for Kokoro TTS model.
//!
//! Follows the defense-in-depth pattern established by HTDemucs (#929, #941, #958)
//! and SileroVad (#941, #944).

use thiserror::Error;

// Re-export so callers can match on inner iSTFT error variants.
pub use crate::kokoro_istft::KokoroIstftError;

/// Errors specific to Kokoro TTS forward paths.
#[derive(Error, Debug)]
#[non_exhaustive]
pub enum KokoroError {
    /// Invalid speed parameter (must be > 0.0 and finite).
    #[error("Invalid speed: {value} (must be finite and > 0.0)")]
    InvalidSpeed { value: f32 },

    /// Invalid input dimensions or values.
    #[error("{0}")]
    InvalidInput(String),

    /// Invalid configuration parameter.
    #[error("Invalid config: {field} — {reason}")]
    InvalidConfig { field: &'static str, reason: String },

    /// Non-finite values (NaN/Inf) detected in an intermediate tensor.
    #[error("Non-finite intermediate at {stage}: {count} NaN/Inf values")]
    NonFiniteIntermediate { stage: &'static str, count: usize },

    /// SourceModule weights missing — required for frequency-domain harmonic source.
    #[error("SourceModule weights not found in safetensors (required for STFT-domain excitation)")]
    MissingSourceModule,

    /// iSTFT bin count mismatch between decoder output and n_fft config.
    #[error("decoder output has {actual} bins, expected {expected} for n_fft={n_fft}")]
    IstftBinMismatch {
        actual: usize,
        expected: usize,
        n_fft: usize,
    },

    /// iSTFT spectrogram array not contiguous after standard-layout conversion.
    #[error("iSTFT spectrogram array not contiguous after standard-layout conversion")]
    IstftArrayLayout,

    /// iSTFT waveform reconstruction failed.
    #[error("iSTFT reconstruction failed: {0}")]
    IstftFailed(#[source] KokoroIstftError),

    /// Generator config vectors have mismatched lengths.
    #[error(
        "Generator config mismatch: {field} has length {actual}, expected {expected} \
         (must match {reference_field})"
    )]
    GeneratorConfigMismatch {
        field: &'static str,
        reference_field: &'static str,
        expected: usize,
        actual: usize,
    },

    /// Underlying tensor operation failed.
    #[error(transparent)]
    Tensor(#[from] nn_core::TensorError),
}

impl KokoroError {
    /// Convert to `TensorError` for use in contexts that require `TensorError`.
    ///
    /// Extracts the inner `TensorError` from the `Tensor` variant, or wraps
    /// non-tensor variants as `TensorError::Unsupported`. Needed for
    /// `trace_graph` closures which return `Result<_, TensorError>`.
    pub fn into_tensor_error(self) -> nn_core::TensorError {
        match self {
            Self::Tensor(te) => te,
            other => nn_core::TensorError::Unsupported(other.to_string()),
        }
    }
}

/// Enables `?` on `Result<T, KokoroError>` in functions returning `Result<T, TensorError>`.
///
/// Extracts the inner `TensorError` from `KokoroError::Tensor`, or wraps
/// non-tensor variants as `TensorError::Unsupported`. This allows callers
/// (trace_graph closures, verification test helpers) to use `?` directly
/// without explicit `map_err`.
impl From<KokoroError> for nn_core::TensorError {
    fn from(e: KokoroError) -> Self {
        e.into_tensor_error()
    }
}

/// Check a DynTensor for NaN/Inf values, returning `NonFiniteIntermediate` on failure.
///
/// Delegates to [`nn_core::layers::check_output_finite`] which uses GPU-native
/// [`GpuBackend::count_non_finite`] when available, avoiding `to_flat_vec::<f32>()`
/// CPU round-trips on Metal tensors.
pub(crate) fn check_tensor_finite(
    tensor: &nn_core::dyn_tensor::DynTensor,
    stage: &'static str,
) -> Result<(), KokoroError> {
    nn_core::layers::check_output_finite(tensor, stage).map_err(|e| match e {
        nn_core::TensorError::NonFiniteData { count, .. } => {
            KokoroError::NonFiniteIntermediate { stage, count }
        }
        other => KokoroError::Tensor(other),
    })
}

/// Validate that speed is positive and finite.
///
/// Shared validation for both [`KokoroModel::forward()`](super::kokoro_tts::KokoroModel)
/// and [`CompiledKokoro::synthesize()`](nn_metal compiled path).
pub fn validate_speed(speed: f32) -> Result<(), KokoroError> {
    if !speed.is_finite() || speed <= 0.0 {
        return Err(KokoroError::InvalidSpeed { value: speed });
    }
    Ok(())
}

/// Validate that Generator config vectors have consistent lengths.
///
/// `upsample_kernel_sizes` must match `upsample_rates` in length.
/// `resblock_dilations` must match `resblock_kernel_sizes` in length.
/// Prevents index-out-of-bounds panics in [`Generator::load`](super::kokoro_decoder::Generator).
pub(crate) fn validate_generator_config(
    config: &super::kokoro_tts::KokoroConfig,
) -> Result<(), KokoroError> {
    let num_ups = config.upsample_rates.len();
    if config.upsample_kernel_sizes.len() != num_ups {
        return Err(KokoroError::GeneratorConfigMismatch {
            field: "upsample_kernel_sizes",
            reference_field: "upsample_rates",
            expected: num_ups,
            actual: config.upsample_kernel_sizes.len(),
        });
    }
    let num_rk = config.resblock_kernel_sizes.len();
    if config.resblock_dilations.len() != num_rk {
        return Err(KokoroError::GeneratorConfigMismatch {
            field: "resblock_dilations",
            reference_field: "resblock_kernel_sizes",
            expected: num_rk,
            actual: config.resblock_dilations.len(),
        });
    }
    Ok(())
}

/// Maximum log-magnitude value before exp() to prevent overflow.
/// exp(88.0) ≈ 1.65e38, exp(88.7) ≈ 3.4e38 (near f32::MAX ≈ 3.4e38).
pub(crate) const LOG_MAG_CLAMP_MAX: f64 = 88.0;

#[cfg(test)]
#[path = "kokoro_error_tests.rs"]
mod tests;

// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Tests for `From<ModelError> for TensorError` impls added in #880.
//!
//! Each test verifies:
//! 1. `.into()` produces `BackendFailure { Metal|Cpu, ... }` (domain matches computation site)
//! 2. `?` operator works in a function returning `nn_core::Result<T>`

use crate::demucs_spectral_decoder::DemucsSpectralDecoderError;
use crate::demucs_spectral_encoder::DemucsSpectralEncoderError;
use crate::demucs_temporal_decoder::DemucsTemporalDecoderError;
use crate::demucs_temporal_encoder::DemucsTemporalEncoderError;
use crate::demucs_transformer::DemucsTransformerError;
use crate::htdemucs::HTDemucsError;
use crate::silero_vad::SileroVadError;
use crate::stft::StftError;

/// Helper: verify a TensorError is BackendFailure with the expected domain.
fn assert_backend_failure(
    err: nn_core::TensorError,
    expected_domain: nn_core::BackendDomain,
    expected_substring: &str,
) {
    match err {
        nn_core::TensorError::BackendFailure {
            domain, message, ..
        } => {
            assert_eq!(domain, expected_domain);
            assert!(
                message.contains(expected_substring),
                "expected message containing {expected_substring:?}, got: {message}"
            );
        }
        other => panic!("expected BackendFailure, got: {other:?}"),
    }
}

/// Convenience: assert Metal-domain BackendFailure.
fn assert_metal_backend_failure(err: nn_core::TensorError, expected_substring: &str) {
    assert_backend_failure(err, nn_core::BackendDomain::Metal, expected_substring);
}

/// Helper: simulate `?` propagation in a function returning `nn_core::Result<T>`.
fn propagate_htdemucs() -> nn_core::Result<()> {
    let err: Result<(), HTDemucsError> = Err(HTDemucsError::AudioTooShort {
        actual: 10,
        minimum: 100,
    });
    err?;
    Ok(())
}

fn propagate_silero_vad() -> nn_core::Result<()> {
    let err: Result<(), SileroVadError> = Err(SileroVadError::AudioLength {
        expected: 512,
        actual: 256,
    });
    err?;
    Ok(())
}

fn propagate_stft() -> nn_core::Result<()> {
    let err: Result<(), StftError> = Err(StftError::FreqsMismatch {
        expected: 129,
        actual: 64,
    });
    err?;
    Ok(())
}

fn propagate_temporal_encoder() -> nn_core::Result<()> {
    let err: Result<(), DemucsTemporalEncoderError> =
        Err(DemucsTemporalEncoderError::NonFiniteInput { block: 0, count: 3 });
    err?;
    Ok(())
}

fn propagate_spectral_encoder() -> nn_core::Result<()> {
    let err: Result<(), DemucsSpectralEncoderError> =
        Err(DemucsSpectralEncoderError::NonFiniteInput { block: 1, count: 5 });
    err?;
    Ok(())
}

fn propagate_temporal_decoder() -> nn_core::Result<()> {
    let err: Result<(), DemucsTemporalDecoderError> =
        Err(DemucsTemporalDecoderError::NonFiniteInput { block: 2, count: 7 });
    err?;
    Ok(())
}

fn propagate_spectral_decoder() -> nn_core::Result<()> {
    let err: Result<(), DemucsSpectralDecoderError> =
        Err(DemucsSpectralDecoderError::NonFiniteInput { block: 3, count: 9 });
    err?;
    Ok(())
}

fn propagate_transformer() -> nn_core::Result<()> {
    let err: Result<(), DemucsTransformerError> = Err(DemucsTransformerError::NonFiniteInput {
        layer: 4,
        count: 11,
    });
    err?;
    Ok(())
}

// --- Into tests (AC1–AC4) ---

#[test]
fn test_htdemucs_error_into_tensor_error() {
    let err = HTDemucsError::AudioTooShort {
        actual: 10,
        minimum: 100,
    };
    let tensor_err: nn_core::TensorError = err.into();
    assert_metal_backend_failure(tensor_err, "audio temporal dim 10 < minimum 100");
}

#[test]
fn test_silero_vad_error_into_tensor_error() {
    let err = SileroVadError::AudioLength {
        expected: 512,
        actual: 256,
    };
    let tensor_err: nn_core::TensorError = err.into();
    assert_metal_backend_failure(tensor_err, "audio length mismatch");
}

#[test]
fn test_stft_error_into_tensor_error() {
    let err = StftError::FreqsMismatch {
        expected: 129,
        actual: 64,
    };
    let tensor_err: nn_core::TensorError = err.into();
    // STFT runs on CPU (moved from nn-metal to nn-models in #860).
    assert_backend_failure(tensor_err, nn_core::BackendDomain::Cpu, "n_freqs mismatch");
}

#[test]
fn test_temporal_encoder_error_into_tensor_error() {
    let err = DemucsTemporalEncoderError::NonFiniteInput { block: 0, count: 3 };
    let tensor_err: nn_core::TensorError = err.into();
    assert_metal_backend_failure(tensor_err, "non-finite input at block 0");
}

#[test]
fn test_spectral_encoder_error_into_tensor_error() {
    let err = DemucsSpectralEncoderError::NonFiniteInput { block: 1, count: 5 };
    let tensor_err: nn_core::TensorError = err.into();
    assert_metal_backend_failure(tensor_err, "non-finite input at block 1");
}

#[test]
fn test_temporal_decoder_error_into_tensor_error() {
    let err = DemucsTemporalDecoderError::NonFiniteInput { block: 2, count: 7 };
    let tensor_err: nn_core::TensorError = err.into();
    assert_metal_backend_failure(tensor_err, "non-finite input at block 2");
}

#[test]
fn test_spectral_decoder_error_into_tensor_error() {
    let err = DemucsSpectralDecoderError::NonFiniteInput { block: 3, count: 9 };
    let tensor_err: nn_core::TensorError = err.into();
    assert_metal_backend_failure(tensor_err, "non-finite input at block 3");
}

#[test]
fn test_transformer_error_into_tensor_error() {
    let err = DemucsTransformerError::NonFiniteInput {
        layer: 4,
        count: 11,
    };
    let tensor_err: nn_core::TensorError = err.into();
    assert_metal_backend_failure(tensor_err, "non-finite input at layer 4");
}

// --- Question-mark propagation tests (AC5) ---

#[test]
fn test_htdemucs_error_question_mark_propagation() {
    let result = propagate_htdemucs();
    assert!(result.is_err());
    assert_metal_backend_failure(result.unwrap_err(), "audio temporal dim");
}

#[test]
fn test_silero_vad_error_question_mark_propagation() {
    let result = propagate_silero_vad();
    assert!(result.is_err());
    assert_metal_backend_failure(result.unwrap_err(), "audio length mismatch");
}

#[test]
fn test_stft_error_question_mark_propagation() {
    let result = propagate_stft();
    assert!(result.is_err());
    // STFT runs on CPU (moved from nn-metal to nn-models in #860).
    assert_backend_failure(
        result.unwrap_err(),
        nn_core::BackendDomain::Cpu,
        "n_freqs mismatch",
    );
}

#[test]
fn test_temporal_encoder_question_mark_propagation() {
    let result = propagate_temporal_encoder();
    assert!(result.is_err());
    assert_metal_backend_failure(result.unwrap_err(), "non-finite input at block 0");
}

#[test]
fn test_spectral_encoder_question_mark_propagation() {
    let result = propagate_spectral_encoder();
    assert!(result.is_err());
    assert_metal_backend_failure(result.unwrap_err(), "non-finite input at block 1");
}

#[test]
fn test_temporal_decoder_question_mark_propagation() {
    let result = propagate_temporal_decoder();
    assert!(result.is_err());
    assert_metal_backend_failure(result.unwrap_err(), "non-finite input at block 2");
}

#[test]
fn test_spectral_decoder_question_mark_propagation() {
    let result = propagate_spectral_decoder();
    assert!(result.is_err());
    assert_metal_backend_failure(result.unwrap_err(), "non-finite input at block 3");
}

#[test]
fn test_transformer_question_mark_propagation() {
    let result = propagate_transformer();
    assert!(result.is_err());
    assert_metal_backend_failure(result.unwrap_err(), "non-finite input at layer 4");
}

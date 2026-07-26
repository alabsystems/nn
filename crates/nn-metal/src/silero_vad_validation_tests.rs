// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Weight validation, input validation, state dimension, and output validation
//! tests for [`SileroVad`]. Extracted from `silero_vad_tests.rs` (#839).
//!
//! State dimension and NaN/Inf validation tests are in
//! `silero_vad_state_validation_tests.rs`.

#[path = "silero_vad_state_validation_tests.rs"]
mod state_validation;

use super::super::*;
use super::zero_weights;

/// Verify all 14 weight tensors reject wrong sizes via table-driven test.
#[test]
fn test_all_weight_tensors_reject_wrong_size() {
    let cases: Vec<(&str, Box<dyn Fn(&mut SileroVadWeights)>)> = vec![
        (
            "stft_basis",
            Box::new(|w: &mut SileroVadWeights| w.stft_basis = vec![0.0; 7]),
        ),
        (
            "encoder_0_weight",
            Box::new(|w: &mut SileroVadWeights| w.enc_weights[0] = vec![0.0; 7]),
        ),
        (
            "encoder_1_weight",
            Box::new(|w: &mut SileroVadWeights| w.enc_weights[1] = vec![0.0; 7]),
        ),
        (
            "encoder_2_weight",
            Box::new(|w: &mut SileroVadWeights| w.enc_weights[2] = vec![0.0; 7]),
        ),
        (
            "encoder_3_weight",
            Box::new(|w: &mut SileroVadWeights| w.enc_weights[3] = vec![0.0; 7]),
        ),
        (
            "encoder_0_bias",
            Box::new(|w: &mut SileroVadWeights| w.enc_biases[0] = vec![0.0; 7]),
        ),
        (
            "encoder_1_bias",
            Box::new(|w: &mut SileroVadWeights| w.enc_biases[1] = vec![0.0; 7]),
        ),
        (
            "encoder_2_bias",
            Box::new(|w: &mut SileroVadWeights| w.enc_biases[2] = vec![0.0; 7]),
        ),
        (
            "encoder_3_bias",
            Box::new(|w: &mut SileroVadWeights| w.enc_biases[3] = vec![0.0; 7]),
        ),
        (
            "lstm_weight_ih",
            Box::new(|w: &mut SileroVadWeights| w.lstm_weight_ih = vec![0.0; 7]),
        ),
        (
            "lstm_weight_hh",
            Box::new(|w: &mut SileroVadWeights| w.lstm_weight_hh = vec![0.0; 7]),
        ),
        (
            "lstm_bias_ih",
            Box::new(|w: &mut SileroVadWeights| w.lstm_bias_ih = vec![0.0; 7]),
        ),
        (
            "lstm_bias_hh",
            Box::new(|w: &mut SileroVadWeights| w.lstm_bias_hh = vec![0.0; 7]),
        ),
        (
            "output_weight",
            Box::new(|w: &mut SileroVadWeights| w.output_weight = vec![0.0; 7]),
        ),
        (
            "output_bias",
            Box::new(|w: &mut SileroVadWeights| w.output_bias = vec![0.0; 7]),
        ),
    ];

    for (expected_name, mutator) in &cases {
        let mut w = zero_weights();
        mutator(&mut w);
        let err = SileroVad::new(w).unwrap_err();
        match &err {
            SileroVadError::WeightSize { name, .. } => {
                assert_eq!(
                    *name, *expected_name,
                    "wrong tensor: expected {expected_name}, got {name}"
                );
            }
            other => panic!("expected WeightSize for {expected_name}, got {other:?}"),
        }
    }
}

#[test]
fn test_forward_rejects_wrong_audio_length() {
    let model = SileroVad::new(zero_weights()).unwrap();
    let backend = match crate::metal_backend::MetalBackend::init() {
        Ok(b) => b,
        Err(_) => return,
    };
    let cache = PipelineCache::new(backend.context().clone());
    let state = SileroVadState::zero();

    let err = model.forward(&cache, &[0.0f32; 256], &state).unwrap_err();
    assert!(matches!(
        err,
        SileroVadError::AudioLength {
            expected: 512,
            actual: 256
        }
    ));

    let err = model.forward(&cache, &[0.0f32; 1024], &state).unwrap_err();
    assert!(matches!(
        err,
        SileroVadError::AudioLength {
            expected: 512,
            actual: 1024
        }
    ));

    let err = model.forward(&cache, &[], &state).unwrap_err();
    assert!(matches!(
        err,
        SileroVadError::AudioLength {
            expected: 512,
            actual: 0
        }
    ));
}

#[test]
fn test_forward_rejects_nan_audio() {
    let model = SileroVad::new(zero_weights()).unwrap();
    let backend = match crate::metal_backend::MetalBackend::init() {
        Ok(b) => b,
        Err(_) => return,
    };
    let cache = PipelineCache::new(backend.context().clone());
    let state = SileroVadState::zero();

    // Single NaN at index 10.
    let mut audio = vec![0.0f32; CHUNK_SIZE];
    audio[10] = f32::NAN;
    let err = model.forward(&cache, &audio, &state).unwrap_err();
    assert!(matches!(
        err,
        SileroVadError::NonFiniteAudio {
            count: 1,
            first_index: 10
        }
    ));

    // +Infinity at index 0.
    audio[10] = 0.0;
    audio[0] = f32::INFINITY;
    let err = model.forward(&cache, &audio, &state).unwrap_err();
    assert!(matches!(
        err,
        SileroVadError::NonFiniteAudio {
            count: 1,
            first_index: 0
        }
    ));

    // -Infinity at last index.
    audio[0] = 0.0;
    audio[CHUNK_SIZE - 1] = f32::NEG_INFINITY;
    let err = model.forward(&cache, &audio, &state).unwrap_err();
    assert!(matches!(
        err,
        SileroVadError::NonFiniteAudio {
            count: 1,
            first_index: 511
        }
    ));

    // Multiple non-finite values.
    audio[0] = f32::NAN;
    audio[100] = f32::INFINITY;
    // audio[511] is already NEG_INFINITY from above.
    let err = model.forward(&cache, &audio, &state).unwrap_err();
    assert!(
        matches!(
            err,
            SileroVadError::NonFiniteAudio {
                count: 3,
                first_index: 0
            }
        ),
        "expected 3 non-finite values starting at 0, got {err:?}",
    );
}

/// AC1 (#789): Combined LSTM bias overflow detection.
///
/// When bias_ih and bias_hh both contain values near f32::MAX/2,
/// their sum overflows to infinity. `SileroVad::new()` must reject this.
#[test]
fn test_lstm_bias_overflow_detected() {
    let mut w = zero_weights();
    // bias_ih[0] + bias_hh[0] = MAX + MAX = +inf (IEEE 754 overflow)
    w.lstm_bias_ih[0] = f32::MAX;
    w.lstm_bias_hh[0] = f32::MAX;
    let err = SileroVad::new(w).unwrap_err();
    assert!(
        matches!(
            err,
            SileroVadError::NonFiniteBias {
                count: 1,
                first_index: 0
            }
        ),
        "expected NonFiniteBias at index 0, got {err:?}",
    );
}

/// AC1 (#789): NaN in bias_ih caught at construction (weight finiteness check).
///
/// Previously caught at the combined-bias check (NonFiniteBias). After #966,
/// NaN in individual bias tensors is caught earlier by validate_weight().
#[test]
fn test_lstm_bias_nan_detected() {
    let mut w = zero_weights();
    w.lstm_bias_ih[100] = f32::NAN;
    let err = SileroVad::new(w).unwrap_err();
    assert!(
        matches!(err, SileroVadError::NonFiniteWeight { ref name, count: 1 } if name == "lstm_bias_ih"),
        "expected NonFiniteWeight for lstm_bias_ih, got {err:?}",
    );
}

/// AC6 (#929): NaN in LSTM weights caught at construction (fail-fast, #966).
///
/// After #966, NaN weights are rejected during `SileroVad::new()` rather
/// than propagating through LSTM gates at runtime. This is strictly better:
/// the error is caught earlier with a clearer message.
#[test]
fn test_new_rejects_nan_lstm_weight_ih() {
    let mut w = zero_weights();
    w.lstm_weight_ih[0] = f32::NAN;
    let err = SileroVad::new(w).unwrap_err();
    assert!(
        matches!(err, SileroVadError::NonFiniteWeight { ref name, count: 1 } if name == "lstm_weight_ih"),
        "expected NonFiniteWeight for lstm_weight_ih, got {err:?}",
    );
}

/// AC6 (#929): NaN in LSTM weights also caught at construction (#966).
///
/// Duplicate of `test_new_rejects_nan_lstm_weight_ih` — both exercise the same
/// construction-time path. Retained as a regression guard for the forward_gpu
/// entry point which originally had a separate NaN detection path.
#[test]
fn test_new_rejects_nan_lstm_weight_ih_gpu_path() {
    let mut w = zero_weights();
    w.lstm_weight_ih[0] = f32::NAN;
    let err = SileroVad::new(w).unwrap_err();
    assert!(
        matches!(err, SileroVadError::NonFiniteWeight { ref name, count: 1 } if name == "lstm_weight_ih"),
        "expected NonFiniteWeight for lstm_weight_ih, got {err:?}",
    );
}

/// W4-321 AC4: NaN in output_bias caught at construction (fail-fast, #966).
///
/// Previously NaN propagated through the output stage (ReLU + Linear + Sigmoid)
/// and was caught at the NonFiniteOutput guard. After #966, construction-time
/// validation catches it immediately with a descriptive error.
#[test]
fn test_new_rejects_nan_output_bias() {
    let mut w = zero_weights();
    w.output_bias = vec![f32::NAN];
    let err = SileroVad::new(w).unwrap_err();
    assert!(
        matches!(err, SileroVadError::NonFiniteWeight { ref name, count: 1 } if name == "output_bias"),
        "expected NonFiniteWeight for output_bias, got {err:?}",
    );
}

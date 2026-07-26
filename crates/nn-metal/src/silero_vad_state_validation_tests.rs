// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! State dimension and NaN/Inf validation tests for [`SileroVad`] forward paths.
//!
//! Extracted from `silero_vad_validation_tests.rs` for 500-line compliance (#1306).

use super::super::super::*;
use super::super::zero_weights;

/// W4-321 AC3: State dimension validation rejects wrong-size context.
#[test]
fn test_forward_rejects_wrong_context_size() {
    let model = SileroVad::new(zero_weights()).unwrap();
    let backend = match crate::metal_backend::MetalBackend::init() {
        Ok(b) => b,
        Err(_) => return,
    };
    let cache = PipelineCache::new(backend.context().clone());
    let audio = vec![0.0f32; CHUNK_SIZE];

    // Wrong context length: 32 instead of AUDIO_CONTEXT_SIZE (64).
    let bad_state = SileroVadState {
        context: vec![0.0f32; 32],
        h_state: vec![0.0f32; LSTM_HIDDEN_SIZE],
        c_state: vec![0.0f32; LSTM_HIDDEN_SIZE],
    };
    let err = model.forward(&cache, &audio, &bad_state).unwrap_err();
    assert!(
        matches!(
            err,
            SileroVadError::StateDimension {
                field: "context",
                expected: 64,
                actual: 32
            }
        ),
        "expected StateDimension for context, got {err:?}",
    );
}

/// W4-321 AC3: State dimension validation rejects wrong-size h_state.
#[test]
fn test_forward_rejects_wrong_h_state_size() {
    let model = SileroVad::new(zero_weights()).unwrap();
    let backend = match crate::metal_backend::MetalBackend::init() {
        Ok(b) => b,
        Err(_) => return,
    };
    let cache = PipelineCache::new(backend.context().clone());
    let audio = vec![0.0f32; CHUNK_SIZE];

    // Wrong h_state length: 64 instead of LSTM_HIDDEN_SIZE (128).
    let bad_state = SileroVadState {
        context: vec![0.0f32; AUDIO_CONTEXT_SIZE],
        h_state: vec![0.0f32; 64],
        c_state: vec![0.0f32; LSTM_HIDDEN_SIZE],
    };
    let err = model.forward(&cache, &audio, &bad_state).unwrap_err();
    assert!(
        matches!(
            err,
            SileroVadError::StateDimension {
                field: "h_state",
                expected: 128,
                actual: 64
            }
        ),
        "expected StateDimension for h_state, got {err:?}",
    );
}

/// W4-321 AC3: State dimension validation rejects wrong-size c_state.
#[test]
fn test_forward_rejects_wrong_c_state_size() {
    let model = SileroVad::new(zero_weights()).unwrap();
    let backend = match crate::metal_backend::MetalBackend::init() {
        Ok(b) => b,
        Err(_) => return,
    };
    let cache = PipelineCache::new(backend.context().clone());
    let audio = vec![0.0f32; CHUNK_SIZE];

    // Wrong c_state length: 0 instead of LSTM_HIDDEN_SIZE (128).
    let bad_state = SileroVadState {
        context: vec![0.0f32; AUDIO_CONTEXT_SIZE],
        h_state: vec![0.0f32; LSTM_HIDDEN_SIZE],
        c_state: Vec::new(),
    };
    let err = model.forward(&cache, &audio, &bad_state).unwrap_err();
    assert!(
        matches!(
            err,
            SileroVadError::StateDimension {
                field: "c_state",
                expected: 128,
                actual: 0
            }
        ),
        "expected StateDimension for c_state, got {err:?}",
    );
}

/// AC5 (#929): NaN in incoming h_state is rejected at entry.
#[test]
fn test_forward_rejects_nan_h_state() {
    let model = SileroVad::new(zero_weights()).unwrap();
    let backend = match crate::metal_backend::MetalBackend::init() {
        Ok(b) => b,
        Err(_) => return,
    };
    let cache = PipelineCache::new(backend.context().clone());
    let audio = vec![0.0f32; CHUNK_SIZE];

    let mut bad_state = SileroVadState::zero();
    bad_state.h_state[50] = f32::NAN;
    bad_state.h_state[100] = f32::INFINITY;
    let err = model.forward(&cache, &audio, &bad_state).unwrap_err();
    assert!(
        matches!(
            err,
            SileroVadError::NonFiniteInputState {
                field: "h_state",
                count: 2
            }
        ),
        "expected NonFiniteInputState for h_state, got {err:?}",
    );
}

/// AC5 (#929): NaN in incoming c_state is rejected at entry.
#[test]
fn test_forward_rejects_nan_c_state() {
    let model = SileroVad::new(zero_weights()).unwrap();
    let backend = match crate::metal_backend::MetalBackend::init() {
        Ok(b) => b,
        Err(_) => return,
    };
    let cache = PipelineCache::new(backend.context().clone());
    let audio = vec![0.0f32; CHUNK_SIZE];

    let mut bad_state = SileroVadState::zero();
    bad_state.c_state[0] = f32::NEG_INFINITY;
    let err = model.forward(&cache, &audio, &bad_state).unwrap_err();
    assert!(
        matches!(
            err,
            SileroVadError::NonFiniteInputState {
                field: "c_state",
                count: 1
            }
        ),
        "expected NonFiniteInputState for c_state, got {err:?}",
    );
}

/// AC5 (#929): NaN in incoming context is rejected at entry.
#[test]
fn test_forward_rejects_nan_context() {
    let model = SileroVad::new(zero_weights()).unwrap();
    let backend = match crate::metal_backend::MetalBackend::init() {
        Ok(b) => b,
        Err(_) => return,
    };
    let cache = PipelineCache::new(backend.context().clone());
    let audio = vec![0.0f32; CHUNK_SIZE];

    let mut bad_state = SileroVadState::zero();
    bad_state.context[10] = f32::NAN;
    let err = model.forward(&cache, &audio, &bad_state).unwrap_err();
    assert!(
        matches!(
            err,
            SileroVadError::NonFiniteInputState {
                field: "context",
                count: 1
            }
        ),
        "expected NonFiniteInputState for context, got {err:?}",
    );
}

/// AC5 (#929): forward_gpu() also validates incoming state finiteness.
#[test]
fn test_forward_gpu_rejects_nan_state() {
    let model = SileroVad::new(zero_weights()).unwrap();
    let backend = match crate::metal_backend::MetalBackend::init() {
        Ok(b) => b,
        Err(_) => return,
    };
    let cache = PipelineCache::new(backend.context().clone());
    let audio = vec![0.0f32; CHUNK_SIZE];

    let mut bad_state = SileroVadState::zero();
    bad_state.h_state[0] = f32::NAN;
    let err = model.forward_gpu(&cache, &audio, &bad_state).unwrap_err();
    assert!(
        matches!(
            err,
            SileroVadError::NonFiniteInputState {
                field: "h_state",
                count: 1
            }
        ),
        "expected NonFiniteInputState for h_state in forward_gpu, got {err:?}",
    );
}

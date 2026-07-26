// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Tests for [`super::SileroVad`] model struct and forward pass.

use nn_dsl::lstm_decomposed::build_lstm_cell_decomposed_dual;

use super::*;

pub(super) fn zero_weights() -> SileroVadWeights {
    SileroVadWeights {
        stft_basis: vec![0.0; 258 * 256],
        enc_weights: [
            vec![0.0; 128 * 129 * 3],
            vec![0.0; 64 * 128 * 3],
            vec![0.0; 64 * 64 * 3],
            vec![0.0; 128 * 64 * 3],
        ],
        enc_biases: [vec![0.0; 128], vec![0.0; 64], vec![0.0; 64], vec![0.0; 128]],
        lstm_weight_ih: vec![0.0; 512 * 128],
        lstm_weight_hh: vec![0.0; 512 * 128],
        lstm_bias_ih: vec![0.0; 512],
        lstm_bias_hh: vec![0.0; 512],
        output_weight: vec![0.0; 128],
        output_bias: vec![0.0; 1],
    }
}

#[test]
fn test_model_construction_validates() {
    let model = SileroVad::new(zero_weights());
    assert!(model.is_ok(), "construction failed: {:?}", model.err());
}

#[test]
fn test_model_rejects_wrong_weight_size() {
    let mut w = zero_weights();
    w.stft_basis = vec![0.0; 100];
    let err = SileroVad::new(w).unwrap_err();
    assert!(
        matches!(
            err,
            SileroVadError::WeightSize {
                name: "stft_basis",
                ..
            }
        ),
        "expected WeightSize for stft_basis, got {err:?}",
    );
}

#[test]
fn test_encoder_temporal_progression() {
    assert_eq!(conv1d_output_len(640, 256, 128, 0).unwrap(), 4); // STFT
    assert_eq!(conv1d_output_len(4, 3, 1, 1).unwrap(), 4); // Enc0
    assert_eq!(conv1d_output_len(4, 3, 2, 1).unwrap(), 2); // Enc1
    assert_eq!(conv1d_output_len(2, 3, 2, 1).unwrap(), 1); // Enc2
    assert_eq!(conv1d_output_len(1, 3, 1, 1).unwrap(), 1); // Enc3
}

#[test]
fn test_encoder_block_def_validates() {
    let def = build_encoder_block_def(&ENCODER_BLOCKS[0], 4, 4).expect("valid graph");
    assert!(def.validate().is_ok(), "{:?}", def.validate());
}

#[test]
fn test_lstm_dual_def_validates() {
    // Production uses build_lstm_cell_decomposed_dual (dual h_new+c_new output),
    // not build_lstm_cell_decomposed (single h_new output).
    let def = build_lstm_cell_decomposed_dual(128, 128, 1, true).expect("valid LSTM dims");
    assert!(def.validate().is_ok(), "{:?}", def.validate());
}

#[test]
fn test_output_def_validates() {
    let def = build_output_def().expect("valid graph");
    assert!(def.validate().is_ok(), "{:?}", def.validate());
}

#[test]
fn test_all_kernel_defs_validate() {
    let model = SileroVad::new(zero_weights()).unwrap();
    for (i, def) in model.enc_defs.iter().enumerate() {
        assert!(
            def.validate().is_ok(),
            "enc block {i}: {:?}",
            def.validate()
        );
    }
    assert!(model.lstm_def.validate().is_ok());
    assert!(model.output_def.validate().is_ok());
}

#[test]
fn test_lstm_bias_combination() {
    let mut w = zero_weights();
    w.lstm_bias_ih = vec![1.0; 512];
    w.lstm_bias_hh = vec![2.0; 512];
    let model = SileroVad::new(w).unwrap();
    assert!(model.lstm_bias.iter().all(|&v| (v - 3.0).abs() < 1e-7));
}

#[test]
fn test_forward_zero_weights_on_metal() {
    // Zero weights → all intermediates zero → sigmoid(0) = 0.5.
    let model = SileroVad::new(zero_weights()).unwrap();

    let backend = match crate::metal_backend::MetalBackend::init() {
        Ok(b) => b,
        Err(_) => return, // Skip on non-Metal platforms
    };
    let cache = PipelineCache::new(backend.context().clone());
    let audio = vec![0.0f32; CHUNK_SIZE];
    let state = SileroVadState::zero();
    let output = model.forward(&cache, &audio, &state).expect("forward pass");

    // sigmoid(0) = 0.5 for the final output layer
    assert!(
        (output.probability - 0.5).abs() < 0.01,
        "expected ~0.5 for zero weights, got {}",
        output.probability,
    );
    // Verify state has correct dimensions.
    assert_eq!(output.state.h_state.len(), LSTM_HIDDEN_SIZE);
    assert_eq!(output.state.c_state.len(), LSTM_HIDDEN_SIZE);
    assert_eq!(output.state.context.len(), AUDIO_CONTEXT_SIZE);
}

#[test]
fn test_state_zero_initialization() {
    let state = SileroVadState::zero();
    assert_eq!(state.h_state.len(), LSTM_HIDDEN_SIZE);
    assert_eq!(state.c_state.len(), LSTM_HIDDEN_SIZE);
    assert_eq!(state.context.len(), AUDIO_CONTEXT_SIZE);
    assert!(state.h_state.iter().all(|&v| v == 0.0));
    assert!(state.c_state.iter().all(|&v| v == 0.0));
    assert!(state.context.iter().all(|&v| v == 0.0));
}

#[test]
fn test_multi_chunk_state_propagation() {
    let model = SileroVad::new(zero_weights()).unwrap();

    let backend = match crate::metal_backend::MetalBackend::init() {
        Ok(b) => b,
        Err(_) => return,
    };
    let cache = PipelineCache::new(backend.context().clone());
    let audio = vec![0.0f32; CHUNK_SIZE];
    let state0 = SileroVadState::zero();

    let out1 = model.forward(&cache, &audio, &state0).expect("chunk 1");
    let out2_stateful = model
        .forward(&cache, &audio, &out1.state)
        .expect("chunk 2 stateful");
    let out2_stateless = model
        .forward(&cache, &audio, &state0)
        .expect("chunk 2 stateless");

    // With zero weights, LSTM degenerates (all gates = sigmoid(0) = 0.5,
    // g = tanh(0) = 0, so c_new = 0.5*c + 0 = 0.5*c). Starting from c=0,
    // state doesn't differentiate. We verify plumbing + dimensions.
    assert_eq!(out2_stateful.state.h_state.len(), LSTM_HIDDEN_SIZE);
    assert_eq!(out2_stateful.state.c_state.len(), LSTM_HIDDEN_SIZE);
    assert_eq!(out2_stateful.state.context.len(), AUDIO_CONTEXT_SIZE);
    assert_eq!(out2_stateless.state.h_state.len(), LSTM_HIDDEN_SIZE);
    assert_eq!(out2_stateless.state.c_state.len(), LSTM_HIDDEN_SIZE);
    assert_eq!(out2_stateless.state.context.len(), AUDIO_CONTEXT_SIZE);

    assert!((0.0..=1.0).contains(&out2_stateful.probability));
    assert!((0.0..=1.0).contains(&out2_stateless.probability));
}

/// Parity test: `forward()` and `forward_gpu()` must produce identical output.
///
/// Both paths run the same kernels with the same weights. The only difference
/// is that `forward_gpu()` keeps encoder intermediate results on GPU between
/// stages, while `forward()` reads back to CPU after each stage. The final
/// numerical result must be bit-exact because the same Metal kernels execute
/// in both cases — the difference is only in the data transport path.
///
/// Part of #895 — AC3.
#[test]
fn test_forward_gpu_parity() {
    let model = SileroVad::new(zero_weights()).unwrap();

    let backend = match crate::metal_backend::MetalBackend::init() {
        Ok(b) => b,
        Err(_) => return, // Skip on non-Metal platforms
    };
    let cache = PipelineCache::new(backend.context().clone());
    let audio = vec![0.0f32; CHUNK_SIZE];
    let state = SileroVadState::zero();

    let cpu_out = model
        .forward(&cache, &audio, &state)
        .expect("forward (CPU round-trip)");
    let gpu_out = model
        .forward_gpu(&cache, &audio, &state)
        .expect("forward_gpu (buffer-to-buffer)");

    // Probability must be bit-exact — same kernels, same data.
    assert_eq!(
        cpu_out.probability, gpu_out.probability,
        "probability mismatch: forward={}, forward_gpu={}",
        cpu_out.probability, gpu_out.probability,
    );

    // LSTM state must match.
    assert_eq!(
        cpu_out.state.h_state, gpu_out.state.h_state,
        "h_state mismatch"
    );
    assert_eq!(
        cpu_out.state.c_state, gpu_out.state.c_state,
        "c_state mismatch"
    );
    assert_eq!(
        cpu_out.state.context, gpu_out.state.context,
        "context mismatch"
    );
}

/// Multi-chunk parity: forward_gpu() streaming state propagation matches forward().
#[test]
fn test_forward_gpu_multi_chunk_parity() {
    let model = SileroVad::new(zero_weights()).unwrap();

    let backend = match crate::metal_backend::MetalBackend::init() {
        Ok(b) => b,
        Err(_) => return,
    };
    let cache = PipelineCache::new(backend.context().clone());
    let audio = vec![0.1f32; CHUNK_SIZE]; // Non-zero audio for state differentiation

    let mut cpu_state = SileroVadState::zero();
    let mut gpu_state = SileroVadState::zero();

    for chunk_idx in 0..3 {
        let cpu_out = model.forward(&cache, &audio, &cpu_state).expect("forward");
        let gpu_out = model
            .forward_gpu(&cache, &audio, &gpu_state)
            .expect("forward_gpu");

        assert_eq!(
            cpu_out.probability, gpu_out.probability,
            "chunk {chunk_idx}: probability mismatch",
        );
        assert_eq!(
            cpu_out.state.h_state, gpu_out.state.h_state,
            "chunk {chunk_idx}: h_state mismatch",
        );
        assert_eq!(
            cpu_out.state.c_state, gpu_out.state.c_state,
            "chunk {chunk_idx}: c_state mismatch",
        );

        cpu_state = cpu_out.state;
        gpu_state = gpu_out.state;
    }
}

/// process_gpu() convenience wrapper parity.
#[test]
fn test_process_gpu_parity() {
    let model = SileroVad::new(zero_weights()).unwrap();

    let backend = match crate::metal_backend::MetalBackend::init() {
        Ok(b) => b,
        Err(_) => return,
    };
    let cache = PipelineCache::new(backend.context().clone());
    let audio = vec![0.0f32; CHUNK_SIZE];

    let mut cpu_state = SileroVadState::zero();
    let mut gpu_state = SileroVadState::zero();

    let cpu_prob = model
        .process(&cache, &audio, &mut cpu_state)
        .expect("process");
    let gpu_prob = model
        .process_gpu(&cache, &audio, &mut gpu_state)
        .expect("process_gpu");

    assert_eq!(
        cpu_prob, gpu_prob,
        "process vs process_gpu probability mismatch"
    );
    assert_eq!(
        cpu_state.h_state, gpu_state.h_state,
        "process state h mismatch"
    );
    assert_eq!(
        cpu_state.c_state, gpu_state.c_state,
        "process state c mismatch"
    );
}

// Varied-weight GPU parity tests extracted to separate file (#895).
#[path = "silero_vad_tests_gpu_parity.rs"]
mod gpu_parity;

// Validation tests extracted to separate file (#839).
#[path = "silero_vad_validation_tests.rs"]
mod validation;

// LSTM numeric correctness tests extracted to separate file (#825).
#[path = "silero_vad_lstm_numeric_tests.rs"]
mod lstm_numeric;

// SVAD binary parser robustness tests (P1-64 performance_proofs).
#[path = "silero_vad_svad_parser_tests.rs"]
mod svad_parser;

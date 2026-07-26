// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Varied-weight GPU/CPU parity tests for SileroVad (#895).
//!
//! Extracted from `silero_vad_tests.rs` to keep the parent under 500 lines.
//! These tests use non-zero, non-degenerate weights to exercise the full
//! computation graph — unlike zero-weight tests where sigmoid(0) = 0.5
//! masks many classes of bugs.

use super::super::*;

/// Generate non-zero, non-degenerate weights with per-element variation.
/// Uses sinusoidal variation so that different weight tensors have distinct
/// values (no symmetry), giving the encoder, LSTM, and output layer
/// non-trivial computation that exercises the full data path.
fn varied_weights() -> SileroVadWeights {
    fn fill(n: usize, base: f32, scale: f32) -> Vec<f32> {
        (0..n)
            .map(|i| base + scale * ((i as f32) * 0.017 - 0.3).sin())
            .collect()
    }
    SileroVadWeights {
        stft_basis: fill(258 * 256, 0.01, 0.005),
        enc_weights: [
            fill(128 * 129 * 3, 0.02, 0.01),
            fill(64 * 128 * 3, 0.015, 0.008),
            fill(64 * 64 * 3, 0.01, 0.006),
            fill(128 * 64 * 3, 0.012, 0.007),
        ],
        enc_biases: [
            fill(128, 0.1, 0.05),
            fill(64, 0.08, 0.04),
            fill(64, 0.06, 0.03),
            fill(128, 0.09, 0.045),
        ],
        lstm_weight_ih: fill(512 * 128, 0.03, 0.02),
        lstm_weight_hh: fill(512 * 128, 0.025, 0.015),
        lstm_bias_ih: fill(512, 0.2, 0.1),
        lstm_bias_hh: fill(512, 0.15, 0.08),
        output_weight: fill(128, 0.1, 0.05),
        output_bias: vec![0.05],
    }
}

/// GPU/CPU parity with non-zero, varied weights.
///
/// Zero-weight parity tests are weak: all intermediates are zero, sigmoid(0) = 0.5,
/// and many bugs (wrong axis, transposed weights, fused-vs-decomposed divergence)
/// are masked. Varied weights produce non-degenerate activations that exercise the
/// full computation graph.
#[test]
fn test_forward_gpu_parity_varied_weights() {
    let model = SileroVad::new(varied_weights()).unwrap();

    let backend = match crate::metal_backend::MetalBackend::init() {
        Ok(b) => b,
        Err(_) => return,
    };
    let cache = PipelineCache::new(backend.context().clone());
    let audio = vec![0.1f32; CHUNK_SIZE];
    let state = SileroVadState::zero();

    let cpu_out = model.forward(&cache, &audio, &state).expect("forward");
    let gpu_out = model
        .forward_gpu(&cache, &audio, &state)
        .expect("forward_gpu");

    // Probability must not be degenerate (0.5 = zero-weight case).
    assert!(
        (cpu_out.probability - 0.5).abs() > 0.001,
        "probability {} is degenerate (≈0.5); varied weights should produce non-trivial output",
        cpu_out.probability,
    );

    // Bit-exact parity between CPU-roundtrip and GPU-resident paths.
    assert_eq!(
        cpu_out.probability, gpu_out.probability,
        "probability mismatch: forward={}, forward_gpu={}",
        cpu_out.probability, gpu_out.probability,
    );
    assert_eq!(
        cpu_out.state.h_state, gpu_out.state.h_state,
        "h_state mismatch with varied weights"
    );
    assert_eq!(
        cpu_out.state.c_state, gpu_out.state.c_state,
        "c_state mismatch with varied weights"
    );
    assert_eq!(
        cpu_out.state.context, gpu_out.state.context,
        "context mismatch with varied weights"
    );
}

/// Multi-chunk GPU/CPU parity with varied weights.
///
/// Verifies that LSTM state propagation through non-zero weights produces
/// differentiated probabilities across chunks and that GPU/CPU paths remain
/// parity-exact throughout the streaming sequence.
#[test]
fn test_forward_gpu_multi_chunk_parity_varied() {
    let model = SileroVad::new(varied_weights()).unwrap();

    let backend = match crate::metal_backend::MetalBackend::init() {
        Ok(b) => b,
        Err(_) => return,
    };
    let cache = PipelineCache::new(backend.context().clone());
    let audio = vec![0.1f32; CHUNK_SIZE];

    let mut cpu_state = SileroVadState::zero();
    let mut gpu_state = SileroVadState::zero();
    let mut probabilities = Vec::new();

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

        probabilities.push(cpu_out.probability);
        cpu_state = cpu_out.state;
        gpu_state = gpu_out.state;
    }

    // With varied weights, probabilities should differ across chunks
    // (LSTM state evolves non-trivially). If all 3 are identical, the
    // LSTM state isn't actually propagating through non-zero weights.
    let all_same = probabilities
        .windows(2)
        .all(|w| (w[0] - w[1]).abs() < 1e-10);
    assert!(
        !all_same,
        "all 3 chunk probabilities are identical ({probabilities:?}); LSTM state may not be propagating",
    );
}

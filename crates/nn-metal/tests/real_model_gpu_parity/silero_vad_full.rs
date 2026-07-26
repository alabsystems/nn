// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Silero VAD full model forward: CPU vs GPU parity with real weights.
//!
//! Gated on `SILERO_VAD_WEIGHTS` env var. When set, loads real safetensors
//! weights, runs forward on both CPU and GPU dispatch paths, and compares
//! the speech probability and LSTM state element-wise.

use super::test_utils::gpu_init;

/// Helper: resolve Silero VAD weights path from env var or default location.
/// Returns `None` if weights are unavailable (test should skip).
fn silero_weights_path() -> Option<std::path::PathBuf> {
    let env_path = std::env::var("SILERO_VAD_WEIGHTS").ok();
    let path = env_path.map(std::path::PathBuf::from).unwrap_or_else(|| {
        let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .and_then(|p| p.parent())
            .expect("project root")
            .to_path_buf();
        root.join("models/silero_vad/silero_vad_16k.safetensors")
    });

    if path.exists() {
        Some(path)
    } else {
        eprintln!(
            "SKIP: Silero VAD weights not found at {}. \
             Set SILERO_VAD_WEIGHTS env var to enable.",
            path.display()
        );
        None
    }
}

/// Full Silero VAD forward: CPU dispatch vs GPU dispatch with real weights.
///
/// Loads the production Silero VAD model, runs a single chunk of silence
/// through both `forward()` (CPU round-trip) and `forward_gpu()` (GPU-resident),
/// and verifies that probability and LSTM state match within tolerance.
#[test]
fn test_silero_vad_cpu_vs_gpu() {
    gpu_init();

    let weights_path = match silero_weights_path() {
        Some(p) => p,
        None => return,
    };

    let model = nn_metal::SileroVad::load_safetensors(&weights_path).expect("load silero weights");
    let _backend = nn_metal::MetalBackend::init().expect("Metal backend");
    let cache = nn_metal::PipelineCache::new_global().expect("pipeline cache");

    // Test with silence (common production input).
    let silence = vec![0.0f32; 512];
    let state = nn_metal::SileroVadState::zero();

    let cpu_out = model
        .forward(&cache, &silence, &state)
        .expect("CPU forward");
    let gpu_out = model
        .forward_gpu(&cache, &silence, &state)
        .expect("GPU forward");

    // Probability must be in valid range.
    assert!(
        (0.0..=1.0).contains(&cpu_out.probability),
        "CPU probability {} outside [0, 1]",
        cpu_out.probability
    );
    assert!(
        (0.0..=1.0).contains(&gpu_out.probability),
        "GPU probability {} outside [0, 1]",
        gpu_out.probability
    );

    // CPU vs GPU parity on probability.
    let prob_diff = (cpu_out.probability - gpu_out.probability).abs();
    assert!(
        prob_diff <= 1e-5,
        "probability mismatch: cpu={}, gpu={}, diff={prob_diff}",
        cpu_out.probability,
        gpu_out.probability,
    );

    // LSTM h_state parity.
    assert_eq!(
        cpu_out.state.h_state.len(),
        gpu_out.state.h_state.len(),
        "h_state length mismatch"
    );
    for (i, (c, g)) in cpu_out
        .state
        .h_state
        .iter()
        .zip(gpu_out.state.h_state.iter())
        .enumerate()
    {
        let diff = (c - g).abs();
        assert!(diff <= 1e-5, "h_state[{i}]: cpu={c}, gpu={g}, diff={diff}");
    }

    // LSTM c_state parity.
    assert_eq!(
        cpu_out.state.c_state.len(),
        gpu_out.state.c_state.len(),
        "c_state length mismatch"
    );
    for (i, (c, g)) in cpu_out
        .state
        .c_state
        .iter()
        .zip(gpu_out.state.c_state.iter())
        .enumerate()
    {
        let diff = (c - g).abs();
        assert!(diff <= 1e-5, "c_state[{i}]: cpu={c}, gpu={g}, diff={diff}");
    }
}

/// Multi-chunk streaming: CPU vs GPU parity with real weights.
///
/// Runs 5 chunks through both paths and verifies that the accumulated
/// LSTM state divergence stays bounded. Catches bugs in state propagation
/// that only manifest over multiple recurrent steps.
#[test]
fn test_silero_vad_cpu_vs_gpu_streaming() {
    gpu_init();

    let weights_path = match silero_weights_path() {
        Some(p) => p,
        None => return,
    };

    let model = nn_metal::SileroVad::load_safetensors(&weights_path).expect("load silero weights");
    let _backend = nn_metal::MetalBackend::init().expect("Metal backend");
    let cache = nn_metal::PipelineCache::new_global().expect("pipeline cache");

    let mut cpu_state = nn_metal::SileroVadState::zero();
    let mut gpu_state = nn_metal::SileroVadState::zero();

    // Alternate between silence and synthetic audio to exercise varied paths.
    let chunks: Vec<Vec<f32>> = (0..5)
        .map(|i| {
            (0..512)
                .map(|j| ((i * 512 + j) as f32 * 0.001).sin() * 0.1)
                .collect()
        })
        .collect();

    for (i, chunk) in chunks.iter().enumerate() {
        let cpu_prob = model
            .process(&cache, chunk, &mut cpu_state)
            .unwrap_or_else(|e| panic!("CPU chunk {i}: {e}"));
        let gpu_prob = model
            .process_gpu(&cache, chunk, &mut gpu_state)
            .unwrap_or_else(|e| panic!("GPU chunk {i}: {e}"));

        let prob_diff = (cpu_prob - gpu_prob).abs();
        assert!(
            prob_diff <= 1e-4,
            "chunk {i}: cpu_prob={cpu_prob}, gpu_prob={gpu_prob}, diff={prob_diff}"
        );

        // Check h_state divergence stays bounded.
        let max_h_diff: f32 = cpu_state
            .h_state
            .iter()
            .zip(gpu_state.h_state.iter())
            .map(|(c, g)| (c - g).abs())
            .fold(0.0f32, f32::max);
        assert!(
            max_h_diff <= 1e-4,
            "chunk {i}: max h_state diff={max_h_diff} exceeds tolerance"
        );
    }
}

/// Silero VAD with speech-like audio: CPU vs GPU parity.
///
/// Uses a 440Hz sine wave as a crude speech proxy to ensure the model
/// produces non-degenerate probabilities and that CPU/GPU agree on them.
#[test]
fn test_silero_vad_cpu_vs_gpu_speech_like() {
    gpu_init();

    let weights_path = match silero_weights_path() {
        Some(p) => p,
        None => return,
    };

    let model = nn_metal::SileroVad::load_safetensors(&weights_path).expect("load silero weights");
    let _backend = nn_metal::MetalBackend::init().expect("Metal backend");
    let cache = nn_metal::PipelineCache::new_global().expect("pipeline cache");

    // 440Hz sine wave at 16kHz sample rate, amplitude 0.5.
    let audio: Vec<f32> = (0..512)
        .map(|i| 0.5 * (2.0 * std::f32::consts::PI * 440.0 * i as f32 / 16000.0).sin())
        .collect();

    let state = nn_metal::SileroVadState::zero();

    let cpu_out = model.forward(&cache, &audio, &state).expect("CPU forward");
    let gpu_out = model
        .forward_gpu(&cache, &audio, &state)
        .expect("GPU forward");

    let prob_diff = (cpu_out.probability - gpu_out.probability).abs();
    assert!(
        prob_diff <= 1e-5,
        "speech-like: cpu={}, gpu={}, diff={prob_diff}",
        cpu_out.probability,
        gpu_out.probability,
    );
}

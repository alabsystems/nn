// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! End-to-end Silero VAD test with real weights.
//!
//! Loads real model weights from `models/silero_vad/silero_vad_16k.safetensors`
//! and optionally compares output against PyTorch reference probabilities from
//! `models/silero_vad/test_probs_16k.bin`.
//!
//! When weights are absent the test prints a skip message and returns Ok
//! (graceful skip via early return, not test-level skip attributes).
//!
//! Zero-weight contract tests are in `silero_vad_e2e_contract.rs`.
//!
//! Part of #761 — Direction 5b (real-weight end-to-end test).

use std::path::Path;

use nn_metal::{
    MetalBackend, PipelineCache, SileroVad, SileroVadState, SileroVadWeights, WeightMap,
};

/// Number of new audio samples per chunk (32ms at 16kHz).
const CHUNK_SIZE: usize = 512;

/// Project root (two levels up from `crates/nn-metal/`).
fn project_root() -> std::path::PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(|p| p.parent())
        .expect("project root")
        .to_path_buf()
}

/// Read a binary file as a Vec<f32> using alignment-safe decoding.
fn read_f32_bin(path: &Path) -> Vec<f32> {
    let bytes = std::fs::read(path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    assert!(
        bytes.len().is_multiple_of(4),
        "{}: byte length {} not aligned to f32",
        path.display(),
        bytes.len(),
    );
    bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

/// Extract f32 data from a WeightMap tensor by name.
fn extract_f32(wm: &WeightMap, name: &str) -> Vec<f32> {
    let bytes = wm
        .tensor_data(name)
        .unwrap_or_else(|e| panic!("tensor '{name}': {e}"));
    assert!(
        bytes.len().is_multiple_of(4),
        "tensor '{name}' byte length {} not aligned to f32 (4 bytes)",
        bytes.len(),
    );
    bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

/// Load `SileroVadWeights` from an opened `WeightMap`.
fn load_weights(wm: &WeightMap) -> SileroVadWeights {
    SileroVadWeights::new(
        extract_f32(wm, "stft_forward_basis_buffer"),
        [
            extract_f32(wm, "encoder_0_weight"),
            extract_f32(wm, "encoder_1_weight"),
            extract_f32(wm, "encoder_2_weight"),
            extract_f32(wm, "encoder_3_weight"),
        ],
        [
            extract_f32(wm, "encoder_0_bias"),
            extract_f32(wm, "encoder_1_bias"),
            extract_f32(wm, "encoder_2_bias"),
            extract_f32(wm, "encoder_3_bias"),
        ],
        extract_f32(wm, "decoder_rnn_weight_ih"),
        extract_f32(wm, "decoder_rnn_weight_hh"),
        extract_f32(wm, "decoder_rnn_bias_ih"),
        extract_f32(wm, "decoder_rnn_bias_hh"),
        extract_f32(wm, "decoder_output_weight"),
        extract_f32(wm, "decoder_output_bias"),
    )
}

/// Full e2e test: load real weights → forward pass → verify output range.
///
/// Gracefully skips when `models/silero_vad/silero_vad_16k.safetensors` is absent.
#[test]
fn e2e_silero_vad_real_weights() {
    let weights_path = project_root().join("models/silero_vad/silero_vad_16k.safetensors");
    if !weights_path.exists() {
        eprintln!(
            "SKIP: Silero VAD weights not found at {}",
            weights_path.display()
        );
        return;
    }

    let backend = MetalBackend::init().expect("Metal backend required");
    let ctx = backend.context().clone();

    // SAFETY: weight file is a valid safetensors file, context is initialized,
    // file outlives the WeightMap (dropped in this scope).
    let wm = unsafe { WeightMap::load(&weights_path, &ctx).expect("load safetensors") };
    assert_eq!(wm.tensor_count(), 15, "expected 15 tensors in Silero VAD");

    let weights = load_weights(&wm);
    let model = SileroVad::new(weights).expect("model construction");

    let cache = PipelineCache::new(ctx);

    // Forward pass with silence (all zeros) — sigmoid(f(0)) should be in [0, 1].
    // forward() now accepts 512 new samples and prepends context from state.
    let silence = vec![0.0f32; 512];
    let state = SileroVadState::zero();
    let out = model
        .forward(&cache, &silence, &state)
        .expect("forward (silence)");
    assert!(
        (0.0..=1.0).contains(&out.probability),
        "silence probability {} outside [0, 1]",
        out.probability,
    );

    // Forward pass with deterministic noise, carrying state from silence chunk.
    let mut noise = vec![0.0f32; 512];
    let mut rng_state: u64 = 42;
    for v in &mut noise {
        // xorshift64
        rng_state ^= rng_state << 13;
        rng_state ^= rng_state >> 7;
        rng_state ^= rng_state << 17;
        *v = ((rng_state as f32) / (u64::MAX as f32)) * 0.2 - 0.1;
    }
    let out_noise = model
        .forward(&cache, &noise, &out.state)
        .expect("forward (noise)");
    assert!(
        (0.0..=1.0).contains(&out_noise.probability),
        "noise probability {} outside [0, 1]",
        out_noise.probability,
    );

    eprintln!(
        "Silero VAD e2e: silence_prob={:.6}, noise_prob={:.6}",
        out.probability, out_noise.probability,
    );
}

/// Compare against PyTorch reference probabilities if available.
///
/// Loads `test_audio_16k.bin` (5 chunks × 512 f32 samples) and
/// `test_probs_16k.bin` (5 f32 probabilities) generated by the converter
/// with `--test-reference`.
///
/// Now uses streaming state (carrying h/c across chunks) to match PyTorch
/// behavior. All chunks should match closely.
#[test]
fn e2e_silero_vad_reference_comparison() {
    let model_dir = project_root().join("models/silero_vad");
    let weights_path = model_dir.join("silero_vad_16k.safetensors");
    let audio_path = model_dir.join("test_audio_16k.bin");
    let probs_path = model_dir.join("test_probs_16k.bin");

    if !weights_path.exists() || !audio_path.exists() || !probs_path.exists() {
        eprintln!("SKIP: Reference data not found. Generate with the Silero VAD converter");
        return;
    }

    let backend = MetalBackend::init().expect("Metal backend required");
    let ctx = backend.context().clone();

    // SAFETY: weight file is valid safetensors, context initialized, file outlives WeightMap.
    let wm = unsafe { WeightMap::load(&weights_path, &ctx).expect("load safetensors") };
    let weights = load_weights(&wm);
    let model = SileroVad::new(weights).expect("model construction");
    let cache = PipelineCache::new(ctx);

    // Load reference audio and probabilities.
    let audio = read_f32_bin(&audio_path);
    let ref_probs = read_f32_bin(&probs_path);

    assert_eq!(audio.len(), 5 * 512, "expected 5 chunks of 512 samples");
    assert_eq!(ref_probs.len(), 5, "expected 5 reference probabilities");

    // Run chunks with streaming state (matching PyTorch behavior).
    // forward() now accepts 512 new samples and prepends context from state
    // internally, matching PyTorch's internal context management.
    let mut state = SileroVadState::zero();
    for (i, chunk) in audio.chunks(512).enumerate() {
        let out = model
            .forward(&cache, chunk, &state)
            .unwrap_or_else(|e| panic!("forward chunk {i}: {e}"));

        assert!(
            (0.0..=1.0).contains(&out.probability),
            "chunk {i}: probability {} outside [0, 1]",
            out.probability,
        );

        let delta = (out.probability - ref_probs[i]).abs();
        eprintln!(
            "chunk {i}: nn={:.6}, pytorch={:.6}, delta={:.6}",
            out.probability, ref_probs[i], delta,
        );

        // Tolerance assertion: nn output must match PyTorch reference within 1e-4.
        // This catches GPU/CPU numerical drift while allowing f32 rounding differences.
        assert!(
            delta < 1e-4,
            "chunk {i}: nn={:.6} differs from pytorch={:.6} by {:.6} (> 1e-4 tolerance)",
            out.probability,
            ref_probs[i],
            delta,
        );

        // Carry state to next chunk (streaming).
        state = out.state;
    }
}

/// Integration test for `SileroVad::load()` + `process()` convenience API.
///
/// Exercises the dvoice-friendly API surface: load from path in one call,
/// then run multi-chunk streaming via `process(&mut state)`.
///
/// Part of #885 — AC4 (load + multi-chunk processing with state carry).
#[test]
fn e2e_load_and_process_convenience_api() {
    let weights_path = project_root().join("models/silero_vad/silero_vad_16k.safetensors");
    if !weights_path.exists() {
        eprintln!(
            "SKIP: Silero VAD weights not found at {}",
            weights_path.display()
        );
        return;
    }

    let _backend = MetalBackend::init().expect("Metal backend required");

    // AC1: SileroVad::load(path) — single-call construction from safetensors.
    // SAFETY: weight file is a valid safetensors file and is not modified.
    let model = unsafe { SileroVad::load(&weights_path).expect("load from path") };

    let cache = PipelineCache::new_global().expect("pipeline cache");
    let mut state = SileroVadState::zero();

    // AC2+AC4: process() with mutable state carry across 4 chunks.
    // Chunk 0-1: silence, Chunk 2-3: deterministic noise.
    let silence = vec![0.0f32; CHUNK_SIZE];
    let mut noise = vec![0.0f32; CHUNK_SIZE];
    let mut rng: u64 = 123;
    for v in &mut noise {
        rng ^= rng << 13;
        rng ^= rng >> 7;
        rng ^= rng << 17;
        *v = ((rng as f32) / (u64::MAX as f32)) * 0.2 - 0.1;
    }

    let chunks: [&[f32]; 4] = [&silence, &silence, &noise, &noise];
    let mut probs = Vec::with_capacity(4);

    for (i, chunk) in chunks.iter().enumerate() {
        let prob = model
            .process(&cache, chunk, &mut state)
            .unwrap_or_else(|e| panic!("process chunk {i}: {e}"));
        assert!(
            (0.0..=1.0).contains(&prob),
            "chunk {i}: probability {prob} outside [0, 1]",
        );
        probs.push(prob);
    }

    // State should have been updated across all 4 chunks.
    // After processing, LSTM state vectors should be non-zero (silence pushes
    // through encoder → LSTM → output, producing non-trivial hidden state).
    assert_eq!(state.h_state.len(), 128, "h_state dimension");
    assert_eq!(state.c_state.len(), 128, "c_state dimension");
    assert_eq!(state.context.len(), 64, "context dimension");

    // Context should contain the last 64 samples of the final noise chunk.
    assert_eq!(
        &state.context[..],
        &noise[noise.len() - 64..],
        "context should be last 64 samples of final chunk",
    );

    eprintln!(
        "load+process e2e: probs={:?}",
        probs.iter().map(|p| format!("{p:.6}")).collect::<Vec<_>>(),
    );
}

/// `get_probabilities()` with real weights. Part of #839.
#[test]
fn e2e_get_probabilities_real_weights() {
    let weights_path = project_root().join("models/silero_vad/silero_vad_16k.safetensors");
    if !weights_path.exists() {
        eprintln!(
            "SKIP: Silero VAD weights not found at {}",
            weights_path.display()
        );
        return;
    }

    let model = SileroVad::load_safetensors(&weights_path).expect("load safetensors");
    let _backend = MetalBackend::init().expect("Metal backend");
    let cache = PipelineCache::new_global().expect("pipeline cache");

    // 10 chunks of silence (320ms) — should produce consistently low probabilities.
    let audio = vec![0.0f32; 10 * CHUNK_SIZE];
    let probs = model
        .get_probabilities(&cache, &audio)
        .expect("get_probabilities");
    assert_eq!(probs.len(), 10);
    for (i, &p) in probs.iter().enumerate() {
        assert!(
            (0.0..=1.0).contains(&p),
            "chunk {i}: probability {p} outside [0, 1]",
        );
    }
    eprintln!(
        "get_probabilities (silence): {:?}",
        probs.iter().map(|p| format!("{p:.4}")).collect::<Vec<_>>(),
    );
}

/// Integration test for the fully safe `SileroVad::load_safetensors()` path.
///
/// Identical to `e2e_load_and_process_convenience_api` but uses
/// `load_safetensors()` (no `unsafe`, no mmap, no Metal context at load time).
/// Verifies parity with the mmap-based `load()` path.
///
/// Part of #839 — dvoice integration: safe loading API.
#[test]
fn e2e_load_safetensors_safe_api() {
    let weights_path = project_root().join("models/silero_vad/silero_vad_16k.safetensors");
    if !weights_path.exists() {
        eprintln!(
            "SKIP: Silero VAD weights not found at {}",
            weights_path.display()
        );
        return;
    }

    // Key difference: no MetalBackend::init() needed before load_safetensors().
    // The backend is only needed for dispatch (forward/process), not weight loading.
    let model = SileroVad::load_safetensors(&weights_path).expect("safe load from safetensors");

    // Now init backend + cache for dispatch.
    let _backend = MetalBackend::init().expect("Metal backend required");
    let cache = PipelineCache::new_global().expect("pipeline cache");
    let mut state = SileroVadState::zero();

    // Process silence chunks — should produce low speech probabilities.
    let silence = vec![0.0f32; CHUNK_SIZE];
    for i in 0..2 {
        let prob = model
            .process(&cache, &silence, &mut state)
            .unwrap_or_else(|e| panic!("process silence chunk {i}: {e}"));
        assert!(
            (0.0..=1.0).contains(&prob),
            "silence chunk {i}: probability {prob} outside [0, 1]",
        );
    }

    // Compare with mmap-based load() to verify weight parity.
    // SAFETY: Weight file is a valid safetensors file and is not modified during the test.
    let model_mmap = unsafe { SileroVad::load(&weights_path).expect("mmap load") };
    let mut state_mmap = SileroVadState::zero();
    let mut state_safe = SileroVadState::zero();

    for i in 0..2 {
        let prob_mmap = model_mmap
            .process(&cache, &silence, &mut state_mmap)
            .unwrap_or_else(|e| panic!("mmap process chunk {i}: {e}"));
        let prob_safe = model
            .process(&cache, &silence, &mut state_safe)
            .unwrap_or_else(|e| panic!("safe process chunk {i}: {e}"));
        assert!(
            (prob_mmap - prob_safe).abs() < 1e-6,
            "chunk {i}: mmap prob {prob_mmap} != safe prob {prob_safe}",
        );
    }

    eprintln!("load_safetensors e2e: safe path produces identical results to mmap path");
}

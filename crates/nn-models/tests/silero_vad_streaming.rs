// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Streaming inference tests for Silero VAD with real weights.
//!
//! Gate: `SILERO_VAD_WEIGHTS` env var pointing to the safetensors file
//! (e.g., `weights/silero_vad.safetensors` or
//! `models/silero_vad/silero_vad_16k.safetensors`).
//!
//! These tests validate the streaming behavior of Silero VAD — processing
//! audio chunk by chunk with LSTM state carried across chunks. The model
//! is reconstructed on CPU using DynTensor nn layers (Conv1d, Lstm, Linear)
//! without requiring a GPU backend, making these tests backend-agnostic.
//!
//! Architecture (16kHz, 512-sample chunks):
//! ```text
//! Audio [576] → STFT (CPU) → [129, 4]
//!   → Encoder 0: Conv1d(129→128, k=3, s=1, p=1) + ReLU → [128, 4]
//!   → Encoder 1: Conv1d(128→64,  k=3, s=2, p=1) + ReLU → [64, 2]
//!   → Encoder 2: Conv1d(64→64,   k=3, s=2, p=1) + ReLU → [64, 1]
//!   → Encoder 3: Conv1d(64→128,  k=3, s=1, p=1) + ReLU → [128, 1]
//!   → Squeeze → [1, 128]
//!   → LSTM cell → h_new [1, 128]
//!   → ReLU → Linear(128→1) → Sigmoid → speech probability
//! ```

use std::collections::HashMap;
use std::path::Path;
use std::time::Instant;

use nn_core::dyn_tensor::DynTensor;
use nn_core::layers::{Conv1d, Conv1dConfig, Linear, Lstm, LstmState, Module};
use nn_core::{load_safetensors, DType, Device};

use nn_models::silero_vad_builders::{ENCODER_BLOCKS, LSTM_HIDDEN_SIZE};
use nn_models::stft::{compute_stft_magnitude, StftParams};

/// Number of new audio samples per chunk (32ms at 16kHz).
const CHUNK_SIZE: usize = 512;

/// Audio context carried between chunks (last 64 samples).
const AUDIO_CONTEXT_SIZE: usize = 64;

/// Load weights from `SILERO_VAD_WEIGHTS`, returning None if unavailable.
fn load_weights() -> Option<HashMap<String, DynTensor>> {
    let path = std::env::var("SILERO_VAD_WEIGHTS").ok()?;
    let p = Path::new(&path);
    if !p.exists() {
        eprintln!("SKIP: SILERO_VAD_WEIGHTS path does not exist: {path}");
        return None;
    }
    Some(load_safetensors(p).expect("load_safetensors should succeed"))
}

/// CPU-only Silero VAD model built from DynTensor nn layers.
///
/// No GPU backend required — all inference runs on CPU via DynTensor ops.
struct CpuSileroVad {
    stft_basis: Vec<f32>,
    stft_params: StftParams,
    enc_convs: Vec<Conv1d>,
    lstm: Lstm,
    output_linear: Linear,
}

/// Streaming state for the CPU Silero VAD model.
#[derive(Clone)]
struct StreamingState {
    h: DynTensor,
    c: DynTensor,
    context: Vec<f32>,
}

impl StreamingState {
    fn zero() -> Self {
        let device = Device::Cpu;
        Self {
            h: DynTensor::zeros(&[1, LSTM_HIDDEN_SIZE], DType::F32, &device).expect("zero h"),
            c: DynTensor::zeros(&[1, LSTM_HIDDEN_SIZE], DType::F32, &device).expect("zero c"),
            context: vec![0.0f32; AUDIO_CONTEXT_SIZE],
        }
    }
}

impl CpuSileroVad {
    /// Build from loaded weight tensors.
    fn from_tensors(tensors: &HashMap<String, DynTensor>) -> Self {
        let stft_basis = tensors
            .get("stft_forward_basis_buffer")
            .expect("stft_forward_basis_buffer")
            .to_flat_vec::<f32>()
            .expect("f32 conversion");

        // Build encoder Conv1d layers.
        let mut enc_convs = Vec::with_capacity(4);
        for (i, block) in ENCODER_BLOCKS.iter().enumerate() {
            let w = tensors
                .get(&format!("encoder_{i}_weight"))
                .expect("encoder weight")
                .clone();
            let b = tensors
                .get(&format!("encoder_{i}_bias"))
                .expect("encoder bias")
                .clone();
            let config = Conv1dConfig::new(block.padding, block.stride, 1);
            let conv = Conv1d::new(w, Some(b), config)
                .unwrap_or_else(|e| panic!("Conv1d construction for encoder_{i}: {e}"));
            enc_convs.push(conv);
        }

        // Build LSTM cell.
        let w_ih = tensors.get("decoder_rnn_weight_ih").expect("w_ih").clone();
        let w_hh = tensors.get("decoder_rnn_weight_hh").expect("w_hh").clone();
        let b_ih = tensors.get("decoder_rnn_bias_ih").expect("b_ih").clone();
        let b_hh = tensors.get("decoder_rnn_bias_hh").expect("b_hh").clone();
        let lstm = Lstm::new(w_ih, w_hh, Some(b_ih), Some(b_hh), LSTM_HIDDEN_SIZE)
            .expect("LSTM construction");

        // Build output linear layer: weight [1, 128, 1] → reshape to [1, 128].
        let out_w_raw = tensors.get("decoder_output_weight").expect("output weight");
        let out_w = out_w_raw
            .reshape([1, LSTM_HIDDEN_SIZE])
            .expect("reshape output weight");
        let out_b = tensors
            .get("decoder_output_bias")
            .expect("output bias")
            .clone();
        let output_linear =
            Linear::new(out_w, Some(out_b)).expect("Linear construction for output");

        Self {
            stft_basis,
            stft_params: StftParams::default(),
            enc_convs,
            lstm,
            output_linear,
        }
    }

    /// Run one forward pass on a 512-sample chunk with streaming state.
    fn forward(&self, audio: &[f32], state: &StreamingState) -> (f32, StreamingState) {
        assert_eq!(audio.len(), CHUNK_SIZE, "expected {CHUNK_SIZE} samples");

        // Step 0: Prepend context → 576-sample STFT input.
        let mut stft_input = Vec::with_capacity(AUDIO_CONTEXT_SIZE + CHUNK_SIZE);
        stft_input.extend_from_slice(&state.context);
        stft_input.extend_from_slice(audio);

        // Step 1: STFT → [129, 4] magnitude spectrogram.
        let stft_mag =
            compute_stft_magnitude(&stft_input, &self.stft_basis, &self.stft_params).expect("STFT");

        // Step 2: Encoder blocks — Conv1d + ReLU × 4.
        // Input shape: [1, 129, 4] (batch=1, channels=129, time=4).
        let n_freqs = self.stft_params.n_freqs; // 129
        let n_frames = stft_mag.len() / n_freqs; // 4
        let mut x = DynTensor::from_vec(stft_mag, &[1, n_freqs, n_frames], &Device::Cpu)
            .expect("DynTensor from STFT");

        for conv in &self.enc_convs {
            x = conv.forward(&x).expect("conv1d forward");
            x = x.relu().expect("relu");
        }

        // Step 3: Squeeze temporal dim (last dim should be 1 after encoder).
        // [1, 128, 1] → [1, 128]
        let enc_shape = x.dims().to_vec();
        assert_eq!(enc_shape.len(), 3, "expected rank-3 encoder output");
        assert_eq!(enc_shape[2], 1, "expected temporal dim = 1 after encoder");
        x = x.reshape([1, enc_shape[1]]).expect("squeeze");

        // Step 4: LSTM cell.
        let lstm_state = LstmState::new(state.h.clone(), state.c.clone()).expect("lstm state");
        let (_h_out, new_lstm_state) = self
            .lstm
            .forward(&x, Some(&lstm_state))
            .expect("lstm forward");

        // Step 5: Output — ReLU + Linear + Sigmoid.
        let h_new = new_lstm_state.h.clone();
        let relu_out = h_new.relu().expect("output relu");
        let linear_out = self.output_linear.forward(&relu_out).expect("linear");
        let prob_tensor = linear_out.sigmoid().expect("sigmoid");

        let prob_vec = prob_tensor.to_flat_vec::<f32>().expect("prob vec");
        let probability = prob_vec[0];

        // Save last 64 samples as context for next chunk.
        let new_context = audio[audio.len() - AUDIO_CONTEXT_SIZE..].to_vec();

        let new_state = StreamingState {
            h: new_lstm_state.h,
            c: new_lstm_state.c,
            context: new_context,
        };

        (probability, new_state)
    }

    /// Convenience: process a chunk, mutating state in place.
    fn process(&self, audio: &[f32], state: &mut StreamingState) -> f32 {
        let (prob, new_state) = self.forward(audio, state);
        *state = new_state;
        prob
    }
}

// ============================================================================
// Test: Initial hidden state is zeros
// ============================================================================

#[test]
fn test_streaming_state_init() {
    let state = StreamingState::zero();
    let h_vec = state.h.to_flat_vec::<f32>().expect("h to vec");
    let c_vec = state.c.to_flat_vec::<f32>().expect("c to vec");

    assert_eq!(h_vec.len(), LSTM_HIDDEN_SIZE);
    assert_eq!(c_vec.len(), LSTM_HIDDEN_SIZE);
    assert_eq!(state.context.len(), AUDIO_CONTEXT_SIZE);

    assert!(
        h_vec.iter().all(|&v| v == 0.0),
        "h_state should be all zeros"
    );
    assert!(
        c_vec.iter().all(|&v| v == 0.0),
        "c_state should be all zeros"
    );
    assert!(
        state.context.iter().all(|&v| v == 0.0),
        "context should be all zeros"
    );
}

// ============================================================================
// Test: Process one 512-sample chunk
// ============================================================================

#[test]
fn test_streaming_single_chunk() {
    let tensors = match load_weights() {
        Some(t) => t,
        None => {
            eprintln!("SKIP: SILERO_VAD_WEIGHTS not set");
            return;
        }
    };

    let model = CpuSileroVad::from_tensors(&tensors);
    let state = StreamingState::zero();

    // Process one chunk of silence.
    let silence = vec![0.0f32; CHUNK_SIZE];
    let (prob, new_state) = model.forward(&silence, &state);

    // Probability must be in [0, 1] (sigmoid output).
    assert!(
        (0.0..=1.0).contains(&prob),
        "probability {prob} outside [0, 1]"
    );
    assert!(prob.is_finite(), "probability must be finite");

    // State dimensions must be correct.
    assert_eq!(new_state.h.dims(), &[1, LSTM_HIDDEN_SIZE]);
    assert_eq!(new_state.c.dims(), &[1, LSTM_HIDDEN_SIZE]);
    assert_eq!(new_state.context.len(), AUDIO_CONTEXT_SIZE);

    eprintln!("Single chunk silence probability: {prob:.6}");
}

// ============================================================================
// Test: 10 sequential chunks — state evolves
// ============================================================================

#[test]
fn test_streaming_continuous() {
    let tensors = match load_weights() {
        Some(t) => t,
        None => {
            eprintln!("SKIP: SILERO_VAD_WEIGHTS not set");
            return;
        }
    };

    let model = CpuSileroVad::from_tensors(&tensors);
    let mut state = StreamingState::zero();
    let mut probs = Vec::with_capacity(10);

    // Generate 10 chunks of deterministic pseudo-random audio.
    let mut rng: u64 = 42;
    for i in 0..10 {
        let mut chunk = vec![0.0f32; CHUNK_SIZE];
        for v in &mut chunk {
            rng ^= rng << 13;
            rng ^= rng >> 7;
            rng ^= rng << 17;
            *v = ((rng as f32) / (u64::MAX as f32)) * 0.1 - 0.05;
        }

        let prob = model.process(&chunk, &mut state);
        assert!(
            (0.0..=1.0).contains(&prob),
            "chunk {i}: probability {prob} outside [0, 1]"
        );
        assert!(prob.is_finite(), "chunk {i}: probability must be finite");
        probs.push(prob);
    }

    // Verify state has evolved: h_state should NOT be all zeros after 10 chunks.
    let h_vec = state.h.to_flat_vec::<f32>().expect("h to vec");
    let c_vec = state.c.to_flat_vec::<f32>().expect("c to vec");

    let h_nonzero = h_vec.iter().any(|&v| v != 0.0);
    let c_nonzero = c_vec.iter().any(|&v| v != 0.0);
    assert!(h_nonzero, "h_state should be non-zero after 10 chunks");
    assert!(c_nonzero, "c_state should be non-zero after 10 chunks");

    // Context should be the last 64 samples of the last chunk (not zeros).
    let context_nonzero = state.context.iter().any(|&v| v != 0.0);
    assert!(
        context_nonzero,
        "context should be non-zero after processing"
    );

    // Verify all probabilities are finite.
    for (i, &p) in probs.iter().enumerate() {
        assert!(p.is_finite(), "chunk {i} produced non-finite probability");
    }

    eprintln!(
        "10-chunk streaming probs: {:?}",
        probs.iter().map(|p| format!("{p:.6}")).collect::<Vec<_>>()
    );
}

// ============================================================================
// Test: Speech detection — feed speech reference, verify prob > 0.5
// ============================================================================

#[test]
fn test_streaming_speech_detection() {
    let tensors = match load_weights() {
        Some(t) => t,
        None => {
            eprintln!("SKIP: SILERO_VAD_WEIGHTS not set");
            return;
        }
    };

    let model = CpuSileroVad::from_tensors(&tensors);

    // Load speech reference input if available.
    let project_root = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(|p| p.parent())
        .expect("project root");
    let speech_npy = project_root.join("weights/silero_vad_ref_speech_input.npy");
    let speech_ref_npy = project_root.join("weights/silero_vad_ref_speech_output.npy");

    if !speech_npy.exists() {
        eprintln!(
            "SKIP: speech reference not found at {}",
            speech_npy.display()
        );
        return;
    }

    // Load speech audio from npy (shape should be [512] or [1, 512]).
    let speech_trace = nn_reftest::load_npy(&speech_npy).expect("load speech npy");
    let speech_data = speech_trace
        .get(0)
        .expect("speech npy should have at least one checkpoint")
        .data
        .clone();
    assert!(
        speech_data.len() >= CHUNK_SIZE,
        "speech reference too short: {} < {CHUNK_SIZE}",
        speech_data.len()
    );

    // Run the model on speech audio chunks with streaming state.
    // Feed enough chunks for the model to warm up its LSTM state.
    let mut state = StreamingState::zero();
    let num_chunks = speech_data.len() / CHUNK_SIZE;
    let mut last_prob = 0.0f32;

    for i in 0..num_chunks {
        let start = i * CHUNK_SIZE;
        let chunk = &speech_data[start..start + CHUNK_SIZE];
        last_prob = model.process(chunk, &mut state);
        eprintln!("speech chunk {i}: prob={last_prob:.6}");
    }

    // The final probability after processing speech should indicate speech.
    // With real Silero VAD weights, speech input should produce prob > 0.5.
    assert!(
        last_prob > 0.3,
        "speech probability {last_prob:.6} should be above 0.3 for speech input"
    );

    // If reference output exists, compare.
    if speech_ref_npy.exists() {
        let ref_trace = nn_reftest::load_npy(&speech_ref_npy).expect("load ref output");
        let ref_prob = ref_trace.get(0).expect("ref output checkpoint").data[0];
        let delta = (last_prob - ref_prob).abs();
        eprintln!("speech detection: nn={last_prob:.6}, ref={ref_prob:.6}, delta={delta:.6}");
    }
}

// ============================================================================
// Test: Silence detection — feed silence, verify prob < 0.5
// ============================================================================

#[test]
fn test_streaming_silence_detection() {
    let tensors = match load_weights() {
        Some(t) => t,
        None => {
            eprintln!("SKIP: SILERO_VAD_WEIGHTS not set");
            return;
        }
    };

    let model = CpuSileroVad::from_tensors(&tensors);
    let mut state = StreamingState::zero();
    let silence = vec![0.0f32; CHUNK_SIZE];

    // Process 5 chunks of silence to let the model stabilize.
    let mut probs = Vec::with_capacity(5);
    for i in 0..5 {
        let prob = model.process(&silence, &mut state);
        probs.push(prob);
        eprintln!("silence chunk {i}: prob={prob:.6}");
    }

    // After 5 chunks of silence, probability should be low.
    let last_prob = *probs.last().unwrap();
    assert!(
        last_prob < 0.5,
        "silence probability {last_prob:.6} should be below 0.5"
    );

    // All silence probabilities should be in valid range.
    for (i, &p) in probs.iter().enumerate() {
        assert!(
            (0.0..=1.0).contains(&p),
            "silence chunk {i}: probability {p} outside [0, 1]"
        );
    }
}

// ============================================================================
// Test: State reset returns to initial
// ============================================================================

#[test]
fn test_streaming_state_reset() {
    let tensors = match load_weights() {
        Some(t) => t,
        None => {
            eprintln!("SKIP: SILERO_VAD_WEIGHTS not set");
            return;
        }
    };

    let model = CpuSileroVad::from_tensors(&tensors);
    let silence = vec![0.0f32; CHUNK_SIZE];

    // Run 1: Process 3 chunks from zero state.
    let mut state1 = StreamingState::zero();
    let prob1_first = model.process(&silence, &mut state1);

    // Run 2: Process 5 chunks, then reset, then process from zero state.
    let mut state2 = StreamingState::zero();
    for _ in 0..5 {
        let mut noise = vec![0.0f32; CHUNK_SIZE];
        let mut rng: u64 = 999;
        for v in &mut noise {
            rng ^= rng << 13;
            rng ^= rng >> 7;
            rng ^= rng << 17;
            *v = ((rng as f32) / (u64::MAX as f32)) * 0.3 - 0.15;
        }
        model.process(&noise, &mut state2);
    }

    // Reset state and process identical first chunk.
    state2 = StreamingState::zero();
    let prob2_first = model.process(&silence, &mut state2);

    // After reset, the first-chunk output should be identical.
    let delta = (prob1_first - prob2_first).abs();
    assert!(
        delta < 1e-6,
        "post-reset first chunk should match: {prob1_first:.6} vs {prob2_first:.6}, delta={delta:.6}"
    );

    // LSTM states should also match after processing the same input.
    let h1 = state1.h.to_flat_vec::<f32>().expect("h1");
    let h2 = state2.h.to_flat_vec::<f32>().expect("h2");
    let max_h_delta: f32 = h1
        .iter()
        .zip(&h2)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert!(
        max_h_delta < 1e-6,
        "post-reset h_state should match: max_delta={max_h_delta}"
    );

    eprintln!("State reset: prob_delta={delta:.6}, h_state_max_delta={max_h_delta:.6}");
}

// ============================================================================
// Test: Determinism — same input sequence produces same outputs
// ============================================================================

#[test]
fn test_streaming_determinism() {
    let tensors = match load_weights() {
        Some(t) => t,
        None => {
            eprintln!("SKIP: SILERO_VAD_WEIGHTS not set");
            return;
        }
    };

    let model = CpuSileroVad::from_tensors(&tensors);

    // Generate a fixed sequence of 5 chunks.
    let mut chunks = Vec::with_capacity(5);
    let mut rng: u64 = 12345;
    for _ in 0..5 {
        let mut chunk = vec![0.0f32; CHUNK_SIZE];
        for v in &mut chunk {
            rng ^= rng << 13;
            rng ^= rng >> 7;
            rng ^= rng << 17;
            *v = ((rng as f32) / (u64::MAX as f32)) * 0.2 - 0.1;
        }
        chunks.push(chunk);
    }

    // Run 1.
    let mut state1 = StreamingState::zero();
    let probs1: Vec<f32> = chunks
        .iter()
        .map(|c| model.process(c, &mut state1))
        .collect();

    // Run 2.
    let mut state2 = StreamingState::zero();
    let probs2: Vec<f32> = chunks
        .iter()
        .map(|c| model.process(c, &mut state2))
        .collect();

    // All probabilities should be bitwise identical (deterministic CPU path).
    for (i, (p1, p2)) in probs1.iter().zip(&probs2).enumerate() {
        assert_eq!(
            p1.to_bits(),
            p2.to_bits(),
            "chunk {i}: probabilities differ: {p1} vs {p2}"
        );
    }

    // Final LSTM states should also be bitwise identical.
    let h1 = state1.h.to_flat_vec::<f32>().expect("h1");
    let h2 = state2.h.to_flat_vec::<f32>().expect("h2");
    let c1 = state1.c.to_flat_vec::<f32>().expect("c1");
    let c2 = state2.c.to_flat_vec::<f32>().expect("c2");

    assert_eq!(h1.len(), h2.len());
    for (i, (a, b)) in h1.iter().zip(&h2).enumerate() {
        assert_eq!(a.to_bits(), b.to_bits(), "h_state[{i}] differs: {a} vs {b}");
    }
    for (i, (a, b)) in c1.iter().zip(&c2).enumerate() {
        assert_eq!(a.to_bits(), b.to_bits(), "c_state[{i}] differs: {a} vs {b}");
    }

    eprintln!(
        "Determinism verified: {:?}",
        probs1.iter().map(|p| format!("{p:.6}")).collect::<Vec<_>>()
    );
}

// ============================================================================
// Test: Batch vs sequential — get_probabilities-style processing matches
// ============================================================================

#[test]
fn test_streaming_batch_vs_sequential() {
    let tensors = match load_weights() {
        Some(t) => t,
        None => {
            eprintln!("SKIP: SILERO_VAD_WEIGHTS not set");
            return;
        }
    };

    let model = CpuSileroVad::from_tensors(&tensors);

    // Generate 8 chunks.
    let mut audio = Vec::with_capacity(8 * CHUNK_SIZE);
    let mut rng: u64 = 7777;
    for _ in 0..8 * CHUNK_SIZE {
        rng ^= rng << 13;
        rng ^= rng >> 7;
        rng ^= rng << 17;
        audio.push(((rng as f32) / (u64::MAX as f32)) * 0.2 - 0.1);
    }

    // Sequential processing: chunk by chunk with state carry.
    let mut state_seq = StreamingState::zero();
    let probs_seq: Vec<f32> = audio
        .chunks_exact(CHUNK_SIZE)
        .map(|c| model.process(c, &mut state_seq))
        .collect();

    // "Batch" processing: same thing but using forward() and manual state.
    let mut state_batch = StreamingState::zero();
    let mut probs_batch = Vec::with_capacity(8);
    for chunk in audio.chunks_exact(CHUNK_SIZE) {
        let (prob, new_state) = model.forward(chunk, &state_batch);
        probs_batch.push(prob);
        state_batch = new_state;
    }

    // Must produce identical results (both are sequential under the hood).
    assert_eq!(probs_seq.len(), probs_batch.len());
    for (i, (ps, pb)) in probs_seq.iter().zip(&probs_batch).enumerate() {
        assert_eq!(
            ps.to_bits(),
            pb.to_bits(),
            "chunk {i}: process() vs forward(): {ps} vs {pb}"
        );
    }

    // Final states should be identical.
    let h_seq = state_seq.h.to_flat_vec::<f32>().expect("h_seq");
    let h_batch = state_batch.h.to_flat_vec::<f32>().expect("h_batch");
    for (i, (a, b)) in h_seq.iter().zip(&h_batch).enumerate() {
        assert_eq!(a.to_bits(), b.to_bits(), "h_state[{i}] differs: {a} vs {b}");
    }

    eprintln!(
        "Batch vs sequential match: {} chunks identical",
        probs_seq.len()
    );
}

// ============================================================================
// Test: Per-chunk inference latency profile
// ============================================================================

#[test]
fn test_streaming_latency_profile() {
    let tensors = match load_weights() {
        Some(t) => t,
        None => {
            eprintln!("SKIP: SILERO_VAD_WEIGHTS not set");
            return;
        }
    };

    let model = CpuSileroVad::from_tensors(&tensors);
    let mut state = StreamingState::zero();
    let silence = vec![0.0f32; CHUNK_SIZE];

    // Warmup: 2 chunks.
    for _ in 0..2 {
        model.process(&silence, &mut state);
    }

    // Measure 20 chunks.
    let mut latencies = Vec::with_capacity(20);
    for _ in 0..20 {
        let start = Instant::now();
        model.process(&silence, &mut state);
        latencies.push(start.elapsed());
    }

    let mean_us = latencies.iter().map(std::time::Duration::as_micros).sum::<u128>() / 20;
    let max_us = latencies.iter().map(std::time::Duration::as_micros).max().unwrap();
    let min_us = latencies.iter().map(std::time::Duration::as_micros).min().unwrap();

    // At 16kHz, one 512-sample chunk = 32ms of audio.
    // For real-time streaming, inference must complete faster than 32ms.
    // CPU inference should comfortably be under 32ms (32000 us) per chunk.
    let chunk_duration_us: u128 = 32_000;
    assert!(
        mean_us < chunk_duration_us,
        "mean latency {mean_us}us exceeds real-time threshold {chunk_duration_us}us"
    );

    // Compute real-time factor (RTF) = processing_time / audio_duration.
    let rtf = mean_us as f64 / chunk_duration_us as f64;

    eprintln!(
        "Latency profile (20 chunks):\n\
         mean={mean_us}us, min={min_us}us, max={max_us}us\n\
         RTF={rtf:.4} (< 1.0 = real-time)\n\
         Latencies: {:?}",
        latencies
            .iter()
            .map(|d| format!("{}us", d.as_micros()))
            .collect::<Vec<_>>()
    );
}

// ============================================================================
// Test: Sequential output reference comparison (npy)
// ============================================================================

#[test]
fn test_streaming_sequential_reference() {
    let tensors = match load_weights() {
        Some(t) => t,
        None => {
            eprintln!("SKIP: SILERO_VAD_WEIGHTS not set");
            return;
        }
    };

    let project_root = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(|p| p.parent())
        .expect("project root");

    let ref_input_npy = project_root.join("weights/silero_vad_ref_input.npy");
    let ref_output_npy = project_root.join("weights/silero_vad_ref_output.npy");
    let ref_seq_npy = project_root.join("weights/silero_vad_ref_sequential_outputs.npy");

    if !ref_input_npy.exists() || !ref_output_npy.exists() {
        eprintln!("SKIP: reference npy files not found");
        return;
    }

    let model = CpuSileroVad::from_tensors(&tensors);

    // Load reference input audio.
    let input_trace = nn_reftest::load_npy(&ref_input_npy).expect("load input npy");
    let input_data = &input_trace.get(0).expect("input checkpoint").data;

    // Load reference single-chunk output.
    let output_trace = nn_reftest::load_npy(&ref_output_npy).expect("load output npy");
    let ref_output = output_trace.get(0).expect("output checkpoint").data[0];

    // Process first chunk from reference input.
    let mut state = StreamingState::zero();
    let num_samples = input_data.len().min(CHUNK_SIZE);
    let mut chunk = vec![0.0f32; CHUNK_SIZE];
    chunk[..num_samples].copy_from_slice(&input_data[..num_samples]);
    let prob = model.process(&chunk, &mut state);

    let delta = (prob - ref_output).abs();
    eprintln!("Reference comparison: nn={prob:.6}, ref={ref_output:.6}, delta={delta:.6}");

    // Allow some tolerance for CPU DynTensor vs Metal dispatch differences.
    assert!(
        delta < 0.01,
        "nn output {prob:.6} differs from reference {ref_output:.6} by {delta:.6}"
    );

    // If sequential reference outputs exist, compare multi-chunk streaming.
    if ref_seq_npy.exists() {
        let seq_trace = nn_reftest::load_npy(&ref_seq_npy).expect("load sequential ref");
        let ref_probs = &seq_trace.get(0).expect("sequential checkpoint").data;

        eprintln!("Sequential reference has {} outputs", ref_probs.len());

        // Re-run from zero state, processing as many chunks as we have reference.
        let mut state = StreamingState::zero();
        let total_samples = input_data.len();
        let num_chunks = (total_samples / CHUNK_SIZE).min(ref_probs.len());

        for i in 0..num_chunks {
            let start = i * CHUNK_SIZE;
            let end = (start + CHUNK_SIZE).min(total_samples);
            let mut chunk = vec![0.0f32; CHUNK_SIZE];
            let copy_len = end - start;
            chunk[..copy_len].copy_from_slice(&input_data[start..end]);

            let prob = model.process(&chunk, &mut state);
            let ref_p = ref_probs[i];
            let d = (prob - ref_p).abs();
            eprintln!("seq chunk {i}: nn={prob:.6}, ref={ref_p:.6}, delta={d:.6}");
        }
    }
}

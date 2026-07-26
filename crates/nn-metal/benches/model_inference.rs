// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Criterion benchmarks for real-weight model inference on Metal GPU.
//!
//! Measures RTF (real-time factor) and throughput for production models:
//!
//! A. **Whisper Encoder** (gated on `WHISPER_WEIGHTS`):
//!    - Encode 30s of mel features (full encoder forward pass)
//!    - Reports time per forward pass and RTF
//!
//! B. **Silero VAD** (gated on `SILERO_VAD_WEIGHTS` or local model path):
//!    - Process 1 second of audio in 512-sample chunks
//!    - Reports time per chunk and chunks per second
//!
//! C. **Kokoro TTS** (gated on `KOKORO_WEIGHTS`):
//!    - Full forward (magnitude + phase) and forward_audio (PCM)
//!    - Reports RTF for audio generation
//!
//! Each benchmark group skips gracefully if weights are not available.
//!
//! Run:
//!   WHISPER_WEIGHTS=/path/to/whisper-tiny \
//!   KOKORO_WEIGHTS=./nn/weights/kokoro_v1_0.safetensors \
//!   cargo bench -p nn-metal --bench model_inference

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use std::hint::black_box;
use criterion::{criterion_group, criterion_main, Criterion};
use nn_core::dyn_tensor::DynTensor;
use nn_core::var_builder::VarBuilder;
use nn_core::{DType, Device};

// ---------------------------------------------------------------------------
// Weight-loading helpers
// ---------------------------------------------------------------------------

/// Convert a safetensors tensor view to a CPU DynTensor.
fn convert_st_tensor(
    view: &safetensors::tensor::TensorView<'_>,
    name: &str,
    device: &Device,
) -> DynTensor {
    let shape: Vec<usize> = view.shape().to_vec();
    let numel: usize = shape.iter().product();
    match view.dtype() {
        safetensors::Dtype::F32 => {
            let floats: Vec<f32> = view
                .data()
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect();
            assert_eq!(floats.len(), numel, "F32 count mismatch for {name}");
            DynTensor::new(&floats, &shape, device).unwrap()
        }
        safetensors::Dtype::F16 => {
            let floats: Vec<f32> = view
                .data()
                .chunks_exact(2)
                .map(|c| half::f16::from_le_bytes([c[0], c[1]]).to_f32())
                .collect();
            assert_eq!(floats.len(), numel, "F16 count mismatch for {name}");
            DynTensor::new(&floats, &shape, device).unwrap()
        }
        safetensors::Dtype::BF16 => {
            let floats: Vec<f32> = view
                .data()
                .chunks_exact(2)
                .map(|c| half::bf16::from_le_bytes([c[0], c[1]]).to_f32())
                .collect();
            assert_eq!(floats.len(), numel, "BF16 count mismatch for {name}");
            DynTensor::new(&floats, &shape, device).unwrap()
        }
        safetensors::Dtype::I64 => {
            let ints: Vec<i64> = view
                .data()
                .chunks_exact(8)
                .map(|c| i64::from_le_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]]))
                .collect();
            assert_eq!(ints.len(), numel, "I64 count mismatch for {name}");
            DynTensor::from_vec_i64(ints, &shape, device).unwrap()
        }
        dt => panic!("unsupported dtype {dt:?} for tensor {name}"),
    }
}

/// Load all tensors from a safetensors file into a HashMap.
fn load_safetensors_map(path: &Path) -> HashMap<String, DynTensor> {
    let data = std::fs::read(path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
    let tensors = safetensors::SafeTensors::deserialize(&data)
        .unwrap_or_else(|e| panic!("parse {}: {e}", path.display()));
    let device = Device::Cpu;
    let mut map = HashMap::new();
    for name in tensors.names() {
        let view = tensors.tensor(name).unwrap();
        map.insert(name.to_string(), convert_st_tensor(&view, name, &device));
    }
    map
}

// ---------------------------------------------------------------------------
// A. Whisper Encoder Benchmark
// ---------------------------------------------------------------------------

fn bench_whisper_encoder(c: &mut Criterion) {
    let Some(weights_dir) = std::env::var("WHISPER_WEIGHTS").ok().map(PathBuf::from) else {
        eprintln!(
            "SKIP whisper_encoder: WHISPER_WEIGHTS not set. \
             Set to whisper-tiny weights directory to enable."
        );
        return;
    };

    let st_path = weights_dir.join("model.safetensors");
    if !st_path.exists() {
        eprintln!("SKIP whisper_encoder: {} not found", st_path.display());
        return;
    }

    eprintln!("Loading Whisper model from {}...", st_path.display());
    let config = nn_whisper::WhisperConfig::whisper_tiny();
    let mut model =
        nn_whisper::WhisperModel::load_safetensors(&st_path, config).expect("load whisper model");

    // 30s of audio = mel spectrogram [1, 80, 3000]
    let mel_data: Vec<f32> = (0..80 * 3000)
        .map(|i| ((i as f32) * 0.013).sin() * 0.5)
        .collect();
    let mel =
        DynTensor::from_vec(mel_data, &[1, 80, 3000], &Device::Cpu).expect("create mel tensor");

    let mut group = c.benchmark_group("whisper_encoder");
    // Model benchmarks are slower -- reduce sample size.
    group.sample_size(10);

    group.bench_function("encode_30s_mel_tiny", |bencher| {
        bencher.iter(|| {
            model.reset_kv_cache();
            let enc_out = model.encode(black_box(&mel)).unwrap();
            black_box(enc_out);
        });
    });

    group.finish();

    // Report RTF after benchmark.
    let start = std::time::Instant::now();
    model.reset_kv_cache();
    let _enc = model.encode(&mel).unwrap();
    let elapsed = start.elapsed();
    let audio_duration_s = 30.0;
    let rtf = elapsed.as_secs_f64() / audio_duration_s;
    eprintln!(
        "Whisper encoder: {:.3}ms for 30s audio, RTF={:.4}",
        elapsed.as_secs_f64() * 1000.0,
        rtf
    );
}

// ---------------------------------------------------------------------------
// B. Silero VAD Benchmark
// ---------------------------------------------------------------------------

fn bench_silero_vad(c: &mut Criterion) {
    // Try env var first, then local path.
    let weights_path = std::env::var("SILERO_VAD_WEIGHTS")
        .ok()
        .map(PathBuf::from)
        .or_else(|| {
            let local = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                .parent()
                .and_then(|p| p.parent())
                .map(|root| root.join("models/silero_vad/silero_vad_16k.safetensors"))
                .unwrap_or_default();
            if local.exists() {
                Some(local)
            } else {
                None
            }
        });

    let Some(weights_path) = weights_path else {
        eprintln!(
            "SKIP silero_vad: SILERO_VAD_WEIGHTS not set and \
             models/silero_vad/silero_vad_16k.safetensors not found."
        );
        return;
    };

    if !weights_path.exists() {
        eprintln!("SKIP silero_vad: {} not found", weights_path.display());
        return;
    }

    eprintln!("Loading Silero VAD from {}...", weights_path.display());
    let backend = nn_metal::MetalBackend::init().expect("Metal backend");
    let ctx = backend.context().clone();
    nn_metal::register_metal_dyn_backend();

    // SAFETY: weight file is valid safetensors, context is initialized.
    let wm = unsafe { nn_metal::WeightMap::load(&weights_path, &ctx).expect("load weights") };

    // Extract weight data (same pattern as silero_vad_e2e.rs).
    let extract = |name: &str| -> Vec<f32> {
        let bytes = wm
            .tensor_data(name)
            .unwrap_or_else(|e| panic!("tensor '{name}': {e}"));
        bytes
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect()
    };

    let weights = nn_metal::SileroVadWeights::new(
        extract("stft_forward_basis_buffer"),
        [
            extract("encoder_0_weight"),
            extract("encoder_1_weight"),
            extract("encoder_2_weight"),
            extract("encoder_3_weight"),
        ],
        [
            extract("encoder_0_bias"),
            extract("encoder_1_bias"),
            extract("encoder_2_bias"),
            extract("encoder_3_bias"),
        ],
        extract("decoder_rnn_weight_ih"),
        extract("decoder_rnn_weight_hh"),
        extract("decoder_rnn_bias_ih"),
        extract("decoder_rnn_bias_hh"),
        extract("decoder_output_weight"),
        extract("decoder_output_bias"),
    );
    let model = nn_metal::SileroVad::new(weights).expect("SileroVad::new");
    let cache = nn_metal::PipelineCache::new(ctx);

    // 1 second of audio at 16kHz = ~31 chunks of 512 samples.
    let num_chunks = 31usize;
    let chunks: Vec<Vec<f32>> = (0..num_chunks)
        .map(|chunk_idx| {
            (0..512)
                .map(|i| {
                    let t = (chunk_idx * 512 + i) as f32 / 16000.0;
                    (440.0 * t * std::f32::consts::TAU).sin() * 0.3
                        + (880.0 * t * std::f32::consts::TAU).sin() * 0.1
                })
                .collect()
        })
        .collect();

    let mut group = c.benchmark_group("silero_vad");
    group.sample_size(20);

    // Benchmark: single chunk forward pass
    group.bench_function("forward_single_chunk_512", |bencher| {
        let state = nn_metal::SileroVadState::zero();
        bencher.iter(|| {
            let out = model
                .forward(&cache, black_box(&chunks[0]), &state)
                .unwrap();
            let _ = black_box(out);
        });
    });

    // Benchmark: process 1 second of audio (streaming, carrying state)
    group.bench_function("forward_1s_streaming_31_chunks", |bencher| {
        bencher.iter(|| {
            let mut state = nn_metal::SileroVadState::zero();
            for chunk in &chunks {
                let out = model.forward(&cache, black_box(chunk), &state).unwrap();
                state = out.state;
            }
            let _ = black_box(state);
        });
    });

    group.finish();

    // Report throughput.
    let start = std::time::Instant::now();
    let mut state = nn_metal::SileroVadState::zero();
    for chunk in &chunks {
        let out = model.forward(&cache, chunk, &state).unwrap();
        state = out.state;
    }
    let elapsed = start.elapsed();
    let chunks_per_sec = num_chunks as f64 / elapsed.as_secs_f64();
    let audio_sec = num_chunks as f64 * 512.0 / 16000.0;
    let rtf = elapsed.as_secs_f64() / audio_sec;
    eprintln!(
        "Silero VAD: {:.3}ms for {:.2}s audio ({} chunks), \
         {:.0} chunks/s, RTF={:.4}",
        elapsed.as_secs_f64() * 1000.0,
        audio_sec,
        num_chunks,
        chunks_per_sec,
        rtf,
    );
}

// ---------------------------------------------------------------------------
// C. Kokoro TTS Benchmark
// ---------------------------------------------------------------------------

fn bench_kokoro_tts(c: &mut Criterion) {
    let Some(weights_path) = std::env::var("KOKORO_WEIGHTS").ok().map(PathBuf::from) else {
        eprintln!(
            "SKIP kokoro_tts: KOKORO_WEIGHTS not set. \
             Set to kokoro_v1_0.safetensors path to enable."
        );
        return;
    };

    if !weights_path.exists() {
        eprintln!("SKIP kokoro_tts: {} not found", weights_path.display());
        return;
    }

    eprintln!("Loading Kokoro model from {}...", weights_path.display());
    let weight_map = load_safetensors_map(&weights_path);
    let vb = VarBuilder::from_tensors(weight_map, DType::F32, &Device::Cpu);
    let config = nn_models::kokoro_tts::KokoroConfig::default();
    let model = nn_models::kokoro_tts::KokoroModel::load(&vb, &config).expect("KokoroModel::load");

    // Synthetic inputs: short phoneme sequence.
    let seq_len = 16;
    let input_ids_data: Vec<u32> = (1..=seq_len as u32).collect();
    let input_ids = DynTensor::from_vec_u32(input_ids_data, &[1, seq_len], &Device::Cpu).unwrap();

    // Style embedding: [1, 256] = [1, 2 * style_dim].
    let style_len = 2 * config.style_dim;
    let style_data: Vec<f32> = (0..style_len)
        .map(|i| (i as f32 * 0.7 + 0.3).sin() * 0.5)
        .collect();
    let style = DynTensor::new(&style_data, &[1, style_len], &Device::Cpu).unwrap();

    let mut group = c.benchmark_group("kokoro_tts");
    group.sample_size(10);

    // Benchmark: full forward (magnitude + phase), no iSTFT
    group.bench_function("forward_full_seq16", |bencher| {
        bencher.iter(|| {
            let (mag, phase) = model
                .forward(black_box(&input_ids), black_box(&style), 1.0)
                .unwrap();
            black_box((mag, phase));
        });
    });

    // Benchmark: full forward_audio (includes iSTFT -> PCM)
    group.bench_function("forward_audio_seq16", |bencher| {
        bencher.iter(|| {
            let audio = model
                .forward_audio(black_box(&input_ids), black_box(&style), 1.0)
                .unwrap();
            black_box(audio);
        });
    });

    // Benchmark with longer sequence for more realistic RTF measurement
    let long_seq_len = 64;
    let long_ids_data: Vec<u32> = (1..=long_seq_len as u32).map(|i| (i % 177) + 1).collect();
    let long_ids =
        DynTensor::from_vec_u32(long_ids_data, &[1, long_seq_len], &Device::Cpu).unwrap();

    group.bench_function("forward_audio_seq64", |bencher| {
        bencher.iter(|| {
            let audio = model
                .forward_audio(black_box(&long_ids), black_box(&style), 1.0)
                .unwrap();
            black_box(audio);
        });
    });

    group.finish();

    // Report RTF for audio generation.
    let start = std::time::Instant::now();
    let audio = model.forward_audio(&input_ids, &style, 1.0).unwrap();
    let elapsed = start.elapsed();
    let audio_samples = audio.dims()[2];
    let audio_duration_s = audio_samples as f64 / 24000.0; // Kokoro outputs at 24kHz
    let rtf = elapsed.as_secs_f64() / audio_duration_s;
    eprintln!(
        "Kokoro TTS (seq16): {:.3}ms for {:.3}s audio ({} samples), RTF={:.4}",
        elapsed.as_secs_f64() * 1000.0,
        audio_duration_s,
        audio_samples,
        rtf,
    );

    // Longer sequence RTF
    let start = std::time::Instant::now();
    let audio = model.forward_audio(&long_ids, &style, 1.0).unwrap();
    let elapsed = start.elapsed();
    let audio_samples = audio.dims()[2];
    let audio_duration_s = audio_samples as f64 / 24000.0;
    let rtf = elapsed.as_secs_f64() / audio_duration_s;
    eprintln!(
        "Kokoro TTS (seq64): {:.3}ms for {:.3}s audio ({} samples), RTF={:.4}",
        elapsed.as_secs_f64() * 1000.0,
        audio_duration_s,
        audio_samples,
        rtf,
    );
}

// ---------------------------------------------------------------------------
// Criterion harness
// ---------------------------------------------------------------------------

criterion_group!(
    benches,
    bench_whisper_encoder,
    bench_silero_vad,
    bench_kokoro_tts,
);
criterion_main!(benches);

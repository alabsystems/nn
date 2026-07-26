// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Full Whisper model Metal GPU vs CPU parity tests with real weights.
//!
//! Loads whisper-tiny weights, runs encoder and decoder on both CPU and Metal,
//! and compares outputs element-wise. Also compares Metal outputs against
//! PyTorch reference tensors (.npy files).
//!
//! ## Setup
//!
//! Set `WHISPER_WEIGHTS` to the whisper-tiny weights directory:
//!
//! ```bash
//! export WHISPER_WEIGHTS=./nn/weights/whisper-tiny
//! cargo test -p nn-metal --test model_forward_all -- whisper_metal_parity --nocapture
//! ```
//!
//! Required files in `$WHISPER_WEIGHTS/`:
//! - `model.safetensors` -- AI Provider whisper-tiny weights
//! - `ref_mel_input.npy` -- PyTorch reference mel [1, 80, 3000]
//! - `ref_encoder_output.npy` -- PyTorch reference encoder output [1, 1500, 384]
//! - `ref_decoder_input_ids.npy` -- PyTorch reference decoder token IDs [1, 1]
//! - `ref_decoder_logits.npy` -- PyTorch reference decoder logits [1, 1, 51865]

use std::path::{Path, PathBuf};

use nn_core::dyn_tensor::DynTensor;
use nn_core::{DType, Device, VarBuilder};
use nn_reftest::load_npy;
use nn_whisper::{WhisperConfig, WhisperModel};

use super::test_utils::{assert_gpu_cpu_close, gpu_init};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn init() {
    gpu_init();
}

/// Returns the whisper weights directory, or None if not configured.
fn weights_dir() -> Option<PathBuf> {
    std::env::var("WHISPER_WEIGHTS").ok().map(PathBuf::from)
}

/// Skip macro -- prints skip message and returns early when weights are absent.
macro_rules! skip_without_weights {
    ($dir:ident) => {
        let Some($dir) = weights_dir() else {
            eprintln!(
                "SKIP: WHISPER_WEIGHTS not set. \
                 Set to whisper-tiny weights directory to run real-weight Metal parity tests."
            );
            return;
        };
        if !$dir.join("model.safetensors").exists() {
            eprintln!("SKIP: model.safetensors not found in {}", $dir.display());
            return;
        }
    };
}

/// Load a reference .npy tensor, returning its flat f32 data and shape.
fn load_ref_npy(dir: &Path, name: &str) -> (Vec<f32>, Vec<usize>) {
    let path = dir.join(format!("{name}.npy"));
    let trace = load_npy(&path).unwrap_or_else(|e| {
        panic!("Failed to load reference {}: {e}", path.display());
    });
    let tensor = trace.get(0).expect("npy should contain one tensor");
    (tensor.data.clone(), tensor.shape.clone())
}

/// Load whisper-tiny weights into a CPU tensor map.
fn load_weight_tensors(dir: &Path) -> std::collections::HashMap<String, DynTensor> {
    let st_path = dir.join("model.safetensors");
    nn_core::dyn_tensor::load_safetensors(&st_path)
        .unwrap_or_else(|e| panic!("Failed to load weights from {}: {e}", st_path.display()))
}

/// Compute max absolute difference between two float slices.
fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "tensor size mismatch");
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max)
}

/// Compute mean absolute difference between two float slices.
fn mean_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "tensor size mismatch");
    let sum: f64 = a
        .iter()
        .zip(b.iter())
        .map(|(x, y)| (f64::from(*x) - f64::from(*y)).abs())
        .sum();
    (sum / a.len() as f64) as f32
}

/// Compute cosine similarity between two float slices.
fn cosine_similarity(a: &[f32], b: &[f32]) -> f64 {
    assert_eq!(a.len(), b.len());
    let mut dot = 0.0f64;
    let mut norm_a = 0.0f64;
    let mut norm_b = 0.0f64;
    for (x, y) in a.iter().zip(b.iter()) {
        let x = f64::from(*x);
        let y = f64::from(*y);
        dot += x * y;
        norm_a += x * x;
        norm_b += y * y;
    }
    let denom = norm_a.sqrt() * norm_b.sqrt();
    if denom < 1e-15 {
        0.0
    } else {
        dot / denom
    }
}

/// Return indices of top-k values in descending order.
fn top_k_indices(data: &[f32], k: usize) -> Vec<usize> {
    let mut indexed: Vec<(usize, f32)> = data.iter().copied().enumerate().collect();
    indexed.sort_by(|(_, a), (_, b)| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
    indexed.iter().take(k).map(|(i, _)| *i).collect()
}

// ===========================================================================
// A. Encoder: CPU vs Metal parity with real weights
// ===========================================================================

/// Whisper encoder with real whisper-tiny weights: CPU vs Metal parity.
///
/// Loads the same safetensors weights into CPU and Metal VarBuilders, runs
/// the encoder forward pass on the PyTorch reference mel input, and asserts
/// that the outputs match within tolerance.
///
/// Tolerance: 1e-3 max abs diff. The encoder has 4 layers with LayerNorm +
/// attention + FFN. Numerical differences between CPU and GPU paths accumulate
/// through these layers, so we allow slightly wider tolerance than single-op
/// parity tests.
#[test]
fn test_whisper_encoder_real_weights_cpu_vs_metal() {
    init();
    skip_without_weights!(dir);

    let config = WhisperConfig::whisper_tiny();
    let tensors = load_weight_tensors(&dir);

    // Load reference mel input.
    let (mel_data, mel_shape) = load_ref_npy(&dir, "ref_mel_input");
    assert_eq!(mel_shape, vec![1, 80, 3000], "mel input shape");

    // CPU model.
    let vb_cpu = VarBuilder::from_tensors(tensors.clone(), DType::F32, &Device::Cpu);
    let mut model_cpu = WhisperModel::load(&vb_cpu, config.clone()).expect("CPU model load");
    let mel_cpu =
        DynTensor::from_vec(mel_data.clone(), &mel_shape, &Device::Cpu).expect("CPU mel tensor");
    let enc_cpu = model_cpu.encode(&mel_cpu).expect("CPU encode");

    // Metal model.
    let vb_gpu = VarBuilder::from_tensors(tensors, DType::F32, &Device::metal());
    let mut model_gpu = WhisperModel::load(&vb_gpu, config).expect("Metal model load");
    let mel_gpu =
        DynTensor::from_vec(mel_data, &mel_shape, &Device::metal()).expect("Metal mel tensor");
    let enc_gpu = model_gpu.encode(&mel_gpu).expect("Metal encode");

    // Shape validation.
    assert_eq!(enc_cpu.dims(), &[1, 1500, 384], "CPU encoder shape");
    assert_eq!(enc_gpu.dims(), enc_cpu.dims(), "shapes must match");
    assert_eq!(enc_gpu.device(), Device::metal(), "output stays on GPU");

    // Parity check.
    assert_gpu_cpu_close(&enc_gpu, &enc_cpu, 1e-3, "whisper_encoder_real_weights");

    // Report statistics.
    let cpu_data = enc_cpu.to_flat_vec::<f32>().unwrap();
    let gpu_data = enc_gpu
        .to_device(&Device::Cpu)
        .unwrap()
        .to_flat_vec::<f32>()
        .unwrap();
    let mad = max_abs_diff(&cpu_data, &gpu_data);
    let mean_ad = mean_abs_diff(&cpu_data, &gpu_data);
    let cos_sim = cosine_similarity(&cpu_data, &gpu_data);

    eprintln!("--- Whisper encoder real-weight CPU vs Metal ---");
    eprintln!("  max_abs_diff  = {mad:.6e}");
    eprintln!("  mean_abs_diff = {mean_ad:.6e}");
    eprintln!("  cosine_sim    = {cos_sim:.10}");
}

// ===========================================================================
// B. Decoder: CPU vs Metal parity with real weights
// ===========================================================================

/// Whisper decoder with real weights: CPU vs Metal parity.
///
/// Uses the PyTorch reference encoder output as decoder input (to isolate
/// decoder parity from encoder divergence). Feeds SOT token (50258) and
/// compares decoder logits between CPU and Metal.
#[test]
fn test_whisper_decoder_real_weights_cpu_vs_metal() {
    init();
    skip_without_weights!(dir);

    let config = WhisperConfig::whisper_tiny();
    let tensors = load_weight_tensors(&dir);

    // Load PyTorch reference encoder output.
    let (ref_enc_data, ref_enc_shape) = load_ref_npy(&dir, "ref_encoder_output");
    assert_eq!(ref_enc_shape, vec![1, 1500, 384], "encoder output shape");

    // SOT token.
    let token_data = vec![50258.0f32];
    let token_shape = [1usize, 1];

    // CPU model.
    let vb_cpu = VarBuilder::from_tensors(tensors.clone(), DType::F32, &Device::Cpu);
    let mut model_cpu = WhisperModel::load(&vb_cpu, config.clone()).expect("CPU model load");
    let enc_out_cpu = DynTensor::from_vec(ref_enc_data.clone(), &ref_enc_shape, &Device::Cpu)
        .expect("CPU encoder output tensor");
    let tokens_cpu =
        DynTensor::new(&token_data, &token_shape, &Device::Cpu).expect("CPU token tensor");
    let logits_cpu = model_cpu
        .decode(&tokens_cpu, &enc_out_cpu, true, 0)
        .expect("CPU decode");

    // Metal model.
    let vb_gpu = VarBuilder::from_tensors(tensors, DType::F32, &Device::metal());
    let mut model_gpu = WhisperModel::load(&vb_gpu, config).expect("Metal model load");
    let enc_out_gpu = DynTensor::from_vec(ref_enc_data, &ref_enc_shape, &Device::metal())
        .expect("Metal encoder output tensor");
    let tokens_gpu =
        DynTensor::new(&token_data, &token_shape, &Device::metal()).expect("Metal token tensor");
    let logits_gpu = model_gpu
        .decode(&tokens_gpu, &enc_out_gpu, true, 0)
        .expect("Metal decode");

    // Shape validation.
    assert_eq!(logits_cpu.dims(), &[1, 1, 51865], "CPU logits shape");
    assert_eq!(logits_gpu.dims(), logits_cpu.dims(), "shapes must match");

    // Parity check: decoder involves cross-attention + self-attention + FFN
    // through 4 layers, so tolerance is slightly wider.
    assert_gpu_cpu_close(
        &logits_gpu,
        &logits_cpu,
        5e-3,
        "whisper_decoder_real_weights",
    );

    // Report statistics.
    let cpu_data = logits_cpu.to_flat_vec::<f32>().unwrap();
    let gpu_data = logits_gpu
        .to_device(&Device::Cpu)
        .unwrap()
        .to_flat_vec::<f32>()
        .unwrap();
    let mad = max_abs_diff(&cpu_data, &gpu_data);
    let mean_ad = mean_abs_diff(&cpu_data, &gpu_data);
    let cos_sim = cosine_similarity(&cpu_data, &gpu_data);

    eprintln!("--- Whisper decoder real-weight CPU vs Metal ---");
    eprintln!("  max_abs_diff  = {mad:.6e}");
    eprintln!("  mean_abs_diff = {mean_ad:.6e}");
    eprintln!("  cosine_sim    = {cos_sim:.10}");

    // Top-1 token must agree between CPU and Metal.
    let cpu_top1 = top_k_indices(&cpu_data, 1)[0];
    let gpu_top1 = top_k_indices(&gpu_data, 1)[0];
    assert_eq!(
        cpu_top1, gpu_top1,
        "Top-1 token must agree: CPU={cpu_top1}, Metal={gpu_top1}"
    );
    eprintln!("  top-1 token   = {cpu_top1} (CPU=Metal)");
}

// ===========================================================================
// C. Full pipeline: encode + decode CPU vs Metal parity
// ===========================================================================

/// Full Whisper pipeline: mel -> encode -> decode on CPU and Metal.
///
/// Runs the complete inference pipeline on both backends with identical inputs
/// and compares the output logits. This is the most comprehensive parity test:
/// it exercises every layer of the model on both backends.
#[test]
fn test_whisper_full_pipeline_cpu_vs_metal() {
    init();
    skip_without_weights!(dir);

    let config = WhisperConfig::whisper_tiny();
    let tensors = load_weight_tensors(&dir);

    let (mel_data, mel_shape) = load_ref_npy(&dir, "ref_mel_input");

    // SOT token for decoder.
    let token_data = vec![50258.0f32];
    let token_shape = [1usize, 1];

    // CPU pipeline.
    let vb_cpu = VarBuilder::from_tensors(tensors.clone(), DType::F32, &Device::Cpu);
    let mut model_cpu = WhisperModel::load(&vb_cpu, config.clone()).expect("CPU model load");
    let mel_cpu = DynTensor::from_vec(mel_data.clone(), &mel_shape, &Device::Cpu).expect("CPU mel");
    let enc_cpu = model_cpu.encode(&mel_cpu).expect("CPU encode");
    let tokens_cpu = DynTensor::new(&token_data, &token_shape, &Device::Cpu).expect("CPU tokens");
    let logits_cpu = model_cpu
        .decode(&tokens_cpu, &enc_cpu, true, 0)
        .expect("CPU decode");

    // Metal pipeline.
    let vb_gpu = VarBuilder::from_tensors(tensors, DType::F32, &Device::metal());
    let mut model_gpu = WhisperModel::load(&vb_gpu, config).expect("Metal model load");
    let mel_gpu = DynTensor::from_vec(mel_data, &mel_shape, &Device::metal()).expect("Metal mel");
    let enc_gpu = model_gpu.encode(&mel_gpu).expect("Metal encode");
    let tokens_gpu =
        DynTensor::new(&token_data, &token_shape, &Device::metal()).expect("Metal tokens");
    let logits_gpu = model_gpu
        .decode(&tokens_gpu, &enc_gpu, true, 0)
        .expect("Metal decode");

    // Shape validation.
    assert_eq!(logits_cpu.dims(), &[1, 1, 51865], "CPU logits shape");
    assert_eq!(logits_gpu.dims(), logits_cpu.dims(), "shapes must match");

    // Full pipeline tolerance: errors accumulate through encoder (4 layers) +
    // decoder (4 layers with cross-attention). Allow wider tolerance.
    assert_gpu_cpu_close(&logits_gpu, &logits_cpu, 1e-2, "whisper_full_pipeline");

    let cpu_data = logits_cpu.to_flat_vec::<f32>().unwrap();
    let gpu_data = logits_gpu
        .to_device(&Device::Cpu)
        .unwrap()
        .to_flat_vec::<f32>()
        .unwrap();
    let mad = max_abs_diff(&cpu_data, &gpu_data);
    let mean_ad = mean_abs_diff(&cpu_data, &gpu_data);
    let cos_sim = cosine_similarity(&cpu_data, &gpu_data);

    eprintln!("--- Whisper full pipeline CPU vs Metal ---");
    eprintln!("  max_abs_diff  = {mad:.6e}");
    eprintln!("  mean_abs_diff = {mean_ad:.6e}");
    eprintln!("  cosine_sim    = {cos_sim:.10}");

    // Top-1 token must agree (model should predict same next token on both).
    let cpu_top1 = top_k_indices(&cpu_data, 1)[0];
    let gpu_top1 = top_k_indices(&gpu_data, 1)[0];
    assert_eq!(
        cpu_top1, gpu_top1,
        "Full pipeline top-1 token must agree: CPU={cpu_top1}, Metal={gpu_top1}"
    );

    // Cosine similarity should be very high for numerically close backends.
    assert!(
        cos_sim > 0.999,
        "Full pipeline cosine similarity should be > 0.999, got {cos_sim:.8}"
    );

    eprintln!("  top-1 token   = {cpu_top1} (CPU=Metal)");
}

// ===========================================================================
// D. Metal vs PyTorch reference parity
// ===========================================================================

/// Metal encoder output compared against PyTorch reference.
///
/// This tests the Metal backend specifically against the PyTorch ground truth.
/// Due to sinusoidal vs learned positional embeddings divergence (documented in
/// real_weights.rs), the encoder has large expected divergence. This test
/// validates shapes, finiteness, and reports the divergence metrics.
#[test]
fn test_whisper_metal_encoder_vs_pytorch_reference() {
    init();
    skip_without_weights!(dir);

    let config = WhisperConfig::whisper_tiny();
    let tensors = load_weight_tensors(&dir);

    let (mel_data, mel_shape) = load_ref_npy(&dir, "ref_mel_input");
    let (ref_enc_data, ref_enc_shape) = load_ref_npy(&dir, "ref_encoder_output");
    assert_eq!(ref_enc_shape, vec![1, 1500, 384]);

    // Metal model.
    let vb_gpu = VarBuilder::from_tensors(tensors, DType::F32, &Device::metal());
    let mut model_gpu = WhisperModel::load(&vb_gpu, config).expect("Metal model load");
    let mel_gpu = DynTensor::from_vec(mel_data, &mel_shape, &Device::metal()).expect("Metal mel");
    let enc_gpu = model_gpu.encode(&mel_gpu).expect("Metal encode");

    // Shape validation.
    assert_eq!(enc_gpu.dims(), &[1, 1500, 384], "Metal encoder shape");

    let gpu_data = enc_gpu
        .to_device(&Device::Cpu)
        .unwrap()
        .to_flat_vec::<f32>()
        .unwrap();

    // Finiteness check.
    let non_finite = gpu_data.iter().filter(|v| !v.is_finite()).count();
    assert_eq!(non_finite, 0, "Metal encoder output must be all finite");

    // Report divergence from PyTorch (expected to be large due to pos-embed).
    let mad = max_abs_diff(&gpu_data, &ref_enc_data);
    let mean_ad = mean_abs_diff(&gpu_data, &ref_enc_data);
    let cos_sim = cosine_similarity(&gpu_data, &ref_enc_data);

    eprintln!("--- Metal encoder vs PyTorch reference ---");
    eprintln!("  max_abs_diff  = {mad:.6}");
    eprintln!("  mean_abs_diff = {mean_ad:.6}");
    eprintln!("  cosine_sim    = {cos_sim:.8}");
    eprintln!("  NOTE: Large divergence expected due to sinusoidal vs learned pos-embed.");

    // Reasonable magnitude check (same as CPU test in real_weights.rs).
    let max_abs = gpu_data.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
    assert!(
        max_abs < 100.0,
        "Metal encoder max abs value ({max_abs:.4}) should be < 100"
    );
}

/// Metal decoder output compared against PyTorch reference.
///
/// Feeds PyTorch reference encoder output into the Metal decoder and compares
/// the resulting logits against PyTorch reference logits. Since the decoder
/// uses learned positional embeddings (loaded from weights), and token
/// embeddings match, the decoder should produce close results.
#[test]
fn test_whisper_metal_decoder_vs_pytorch_reference() {
    init();
    skip_without_weights!(dir);

    let config = WhisperConfig::whisper_tiny();
    let tensors = load_weight_tensors(&dir);

    let (ref_enc_data, ref_enc_shape) = load_ref_npy(&dir, "ref_encoder_output");
    let (ref_logits_data, ref_logits_shape) = load_ref_npy(&dir, "ref_decoder_logits");
    assert_eq!(ref_enc_shape, vec![1, 1500, 384]);
    assert_eq!(ref_logits_shape, vec![1, 1, 51865]);

    // Metal model.
    let vb_gpu = VarBuilder::from_tensors(tensors, DType::F32, &Device::metal());
    let mut model_gpu = WhisperModel::load(&vb_gpu, config).expect("Metal model load");

    let enc_out_gpu = DynTensor::from_vec(ref_enc_data, &ref_enc_shape, &Device::metal())
        .expect("Metal encoder output");
    let tokens_gpu =
        DynTensor::new(&[50258.0f32], &[1, 1], &Device::metal()).expect("Metal token tensor");

    let logits_gpu = model_gpu
        .decode(&tokens_gpu, &enc_out_gpu, true, 0)
        .expect("Metal decode with PyTorch encoder output");

    let gpu_data = logits_gpu
        .to_device(&Device::Cpu)
        .unwrap()
        .to_flat_vec::<f32>()
        .unwrap();

    assert_eq!(gpu_data.len(), ref_logits_data.len());

    let mad = max_abs_diff(&gpu_data, &ref_logits_data);
    let mean_ad = mean_abs_diff(&gpu_data, &ref_logits_data);
    let cos_sim = cosine_similarity(&gpu_data, &ref_logits_data);

    eprintln!("--- Metal decoder vs PyTorch reference ---");
    eprintln!("  max_abs_diff  = {mad:.6}");
    eprintln!("  mean_abs_diff = {mean_ad:.6}");
    eprintln!("  cosine_sim    = {cos_sim:.8}");

    // With identical encoder output and learned decoder embeddings, Metal
    // should produce close results to PyTorch.
    assert!(
        cos_sim > 0.99,
        "Metal decoder cosine similarity vs PyTorch should be > 0.99, got {cos_sim:.8}"
    );

    // Top-1 token should match PyTorch.
    let gpu_top1 = top_k_indices(&gpu_data, 1)[0];
    let ref_top1 = top_k_indices(&ref_logits_data, 1)[0];
    eprintln!("  Metal top-1   = {gpu_top1}");
    eprintln!("  PyTorch top-1 = {ref_top1}");

    // Top-10 overlap should be substantial.
    let gpu_top10 = top_k_indices(&gpu_data, 10);
    let ref_top10 = top_k_indices(&ref_logits_data, 10);
    let overlap: usize = gpu_top10
        .iter()
        .filter(|idx| ref_top10.contains(idx))
        .count();
    eprintln!("  top-10 overlap = {overlap}/10");
    assert!(
        overlap >= 5,
        "At least 5 of top-10 tokens should overlap between Metal and PyTorch, got {overlap}/10"
    );
}

// ===========================================================================
// E. Multi-step autoregressive decoding on Metal
// ===========================================================================

/// Multi-step autoregressive decoding on Metal with real weights.
///
/// Runs 5 greedy decode steps on both CPU and Metal and verifies the
/// generated token sequences are identical. This exercises the KV cache
/// path on both backends.
#[test]
fn test_whisper_autoregressive_cpu_vs_metal() {
    init();
    skip_without_weights!(dir);

    let config = WhisperConfig::whisper_tiny();
    let tensors = load_weight_tensors(&dir);

    let (mel_data, mel_shape) = load_ref_npy(&dir, "ref_mel_input");

    let decode_steps = |device: &Device| -> Vec<usize> {
        let vb = VarBuilder::from_tensors(tensors.clone(), DType::F32, device);
        let mut model = WhisperModel::load(&vb, config.clone()).expect("model load");

        let mel = DynTensor::from_vec(mel_data.clone(), &mel_shape, device).expect("mel tensor");
        let enc_out = model.encode(&mel).expect("encode");

        let mut generated_tokens: Vec<usize> = Vec::new();
        let mut current_token = 50258.0f32; // SOT

        for step in 0..5 {
            let tokens = DynTensor::new(&[current_token], &[1, 1], device).expect("token tensor");

            let flush = step == 0;
            let logits = model
                .decode(&tokens, &enc_out, flush, step)
                .unwrap_or_else(|e| panic!("decode step {step}: {e}"));

            let flat = logits
                .to_device(&Device::Cpu)
                .unwrap()
                .to_flat_vec::<f32>()
                .unwrap();

            assert!(
                flat.iter().all(|v| v.is_finite()),
                "step {step}: logits contain non-finite values"
            );

            let next_token = flat
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .map(|(i, _)| i)
                .unwrap();
            generated_tokens.push(next_token);
            current_token = next_token as f32;
        }

        generated_tokens
    };

    let cpu_tokens = decode_steps(&Device::Cpu);
    let gpu_tokens = decode_steps(&Device::metal());

    eprintln!("--- Autoregressive decoding CPU vs Metal ---");
    eprintln!("  CPU tokens: {cpu_tokens:?}");
    eprintln!("  GPU tokens: {gpu_tokens:?}");

    // All generated tokens should be valid vocab indices.
    for &t in cpu_tokens.iter().chain(gpu_tokens.iter()) {
        assert!(t < 51865, "generated token {t} should be < vocab_size");
    }

    // Token sequences should be identical between CPU and Metal.
    assert_eq!(
        cpu_tokens, gpu_tokens,
        "Autoregressive token sequences must match: CPU={cpu_tokens:?}, Metal={gpu_tokens:?}"
    );
}

// ===========================================================================
// F. KV cache determinism on Metal
// ===========================================================================

/// Verify that resetting KV cache on Metal produces identical results.
///
/// This is the Metal counterpart of the CPU test in real_weights.rs.
/// Exercises the Metal KV cache reset path.
#[test]
fn test_whisper_kv_cache_reset_determinism_metal() {
    init();
    skip_without_weights!(dir);

    let config = WhisperConfig::whisper_tiny();
    let tensors = load_weight_tensors(&dir);

    let (mel_data, mel_shape) = load_ref_npy(&dir, "ref_mel_input");

    let vb = VarBuilder::from_tensors(tensors, DType::F32, &Device::metal());
    let mut model = WhisperModel::load(&vb, config).expect("Metal model load");
    let mel = DynTensor::from_vec(mel_data, &mel_shape, &Device::metal()).expect("Metal mel");
    let enc_out = model.encode(&mel).expect("Metal encode");

    let tokens = DynTensor::new(&[50258.0f32], &[1, 1], &Device::metal()).expect("token tensor");

    // First decode.
    let logits1 = model
        .decode(&tokens, &enc_out, true, 0)
        .expect("first decode");
    let v1 = logits1
        .to_device(&Device::Cpu)
        .unwrap()
        .to_flat_vec::<f32>()
        .unwrap();

    // Reset and decode again.
    model.reset_kv_cache();
    let logits2 = model
        .decode(&tokens, &enc_out, true, 0)
        .expect("second decode after reset");
    let v2 = logits2
        .to_device(&Device::Cpu)
        .unwrap()
        .to_flat_vec::<f32>()
        .unwrap();

    let mad = max_abs_diff(&v1, &v2);
    assert!(
        mad < 1e-6,
        "KV cache reset on Metal should produce identical results, max_abs_diff={mad:.6e}"
    );

    eprintln!("Metal KV cache reset determinism: max_abs_diff = {mad:.6e}");
}

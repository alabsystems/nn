// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Metal GPU vs CPU forward parity tests for Whisper tiny.
//!
//! These tests load real AI Provider Whisper-tiny weights, run forward passes on
//! both CPU and Metal GPU, and verify the outputs match within tolerance.
//!
//! ## Setup
//!
//! Set `WHISPER_WEIGHTS` to the directory containing `model.safetensors` and
//! reference `.npy` files:
//!
//! ```bash
//! export WHISPER_WEIGHTS=./nn/weights/whisper-tiny
//! cargo test -p nn-whisper --test metal_parity -- --nocapture
//! ```
//!
//! Tests skip gracefully when `WHISPER_WEIGHTS` is unset or Metal is unavailable.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use nn_core::dyn_tensor::DynTensor;
use nn_core::{DType, Device, VarBuilder};
use nn_metal::{register_metal_dyn_backend, MetalBackend};
use nn_reftest::load_npy;
use nn_whisper::{WhisperConfig, WhisperModel};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

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
                 Set to whisper-tiny weights directory to run metal parity tests."
            );
            return;
        };
    };
}

/// Initialize Metal backend. Returns false if Metal is unavailable.
fn init_metal() -> bool {
    match MetalBackend::init() {
        Ok(_) => {
            register_metal_dyn_backend();
            true
        }
        Err(e) => {
            eprintln!("SKIP: Metal not available: {e}");
            false
        }
    }
}

/// Skip macro for Metal availability.
macro_rules! skip_without_metal {
    () => {
        if !init_metal() {
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

/// Load safetensors into a HashMap of CPU DynTensors.
///
/// Replicates the private `load_safetensors_vb` logic from nn-whisper so we
/// can create VarBuilders targeting arbitrary devices.
fn load_safetensors_tensors(dir: &Path) -> HashMap<String, DynTensor> {
    let st_path = dir.join("model.safetensors");
    let data = std::fs::read(&st_path)
        .unwrap_or_else(|e| panic!("Failed to read {}: {e}", st_path.display()));
    let st = safetensors::SafeTensors::deserialize(&data)
        .unwrap_or_else(|e| panic!("Failed to parse safetensors: {e}"));

    let mut tensors = HashMap::new();
    for (name, view) in st.tensors() {
        let float_data: Vec<f32> = match view.dtype() {
            safetensors::Dtype::F32 => view
                .data()
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect(),
            safetensors::Dtype::BF16 => view
                .data()
                .chunks_exact(2)
                .map(|c| f32::from_bits(u32::from(u16::from_le_bytes([c[0], c[1]])) << 16))
                .collect(),
            safetensors::Dtype::F16 => view
                .data()
                .chunks_exact(2)
                .map(|c| half::f16::from_le_bytes([c[0], c[1]]).to_f32())
                .collect(),
            _ => continue,
        };
        let shape: Vec<usize> = view.shape().to_vec();
        let tensor = DynTensor::new(&float_data, &shape, &Device::Cpu)
            .unwrap_or_else(|e| panic!("Failed to create tensor {name}: {e}"));
        tensors.insert(name.clone(), tensor);
    }
    tensors
}

/// Load whisper-tiny model on a specified device.
fn load_model_on_device(dir: &Path, device: &Device) -> WhisperModel {
    let tensors = load_safetensors_tensors(dir);
    let vb = VarBuilder::from_tensors(tensors, DType::F32, device);
    let config = WhisperConfig::whisper_tiny();
    WhisperModel::load(&vb, config)
        .unwrap_or_else(|e| panic!("Failed to load model on {device}: {e}"))
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

/// Assert two vectors are close and print diagnostics.
fn assert_parity(label: &str, cpu: &[f32], gpu: &[f32], min_cosine: f64) {
    assert_eq!(cpu.len(), gpu.len(), "{label}: length mismatch");

    let cos_sim = cosine_similarity(cpu, gpu);
    let mad = max_abs_diff(cpu, gpu);
    let mean_ad = mean_abs_diff(cpu, gpu);

    eprintln!("{label}:");
    eprintln!("  elements      = {}", cpu.len());
    eprintln!("  cosine_sim    = {cos_sim:.8}");
    eprintln!("  max_abs_diff  = {mad:.6e}");
    eprintln!("  mean_abs_diff = {mean_ad:.6e}");

    // Check all values are finite.
    let cpu_nonfinite = cpu.iter().filter(|v| !v.is_finite()).count();
    let gpu_nonfinite = gpu.iter().filter(|v| !v.is_finite()).count();
    assert_eq!(
        cpu_nonfinite, 0,
        "{label}: CPU output has {cpu_nonfinite} non-finite values"
    );
    assert_eq!(
        gpu_nonfinite, 0,
        "{label}: GPU output has {gpu_nonfinite} non-finite values"
    );

    assert!(
        cos_sim > min_cosine,
        "{label}: cosine similarity {cos_sim:.8} below threshold {min_cosine}"
    );

    eprintln!("  PASS: cosine_sim={cos_sim:.8} > {min_cosine}");
}

// ===========================================================================
// A. Encoder CPU vs Metal
// ===========================================================================

#[test]
fn test_encoder_cpu_vs_metal() {
    skip_without_weights!(dir);
    skip_without_metal!();

    let (mel_data, mel_shape) = load_ref_npy(&dir, "ref_mel_input");
    assert_eq!(mel_shape, vec![1, 80, 3000], "mel input shape");

    // CPU forward.
    let mut cpu_model = load_model_on_device(&dir, &Device::Cpu);
    let mel_cpu = DynTensor::from_vec(mel_data.clone(), &mel_shape, &Device::Cpu)
        .expect("create CPU mel tensor");
    let enc_cpu = cpu_model.encode(&mel_cpu).expect("CPU encoder forward");
    let cpu_data = enc_cpu.to_flat_vec::<f32>().unwrap();

    // Metal forward.
    let mut gpu_model = load_model_on_device(&dir, &Device::metal());
    let mel_gpu = DynTensor::from_vec(mel_data, &mel_shape, &Device::Cpu)
        .expect("create mel tensor")
        .to_device(&Device::metal())
        .expect("transfer mel to Metal");
    let enc_gpu = gpu_model.encode(&mel_gpu).expect("Metal encoder forward");
    let gpu_data = enc_gpu
        .to_device(&Device::Cpu)
        .expect("transfer encoder output to CPU")
        .to_flat_vec::<f32>()
        .unwrap();

    assert_eq!(
        enc_cpu.dims(),
        enc_gpu.dims(),
        "encoder output shape mismatch"
    );
    eprintln!("Encoder output shape: {:?}", enc_cpu.dims());

    assert_parity("Encoder CPU vs Metal", &cpu_data, &gpu_data, 0.9999);
}

// ===========================================================================
// B. Decoder CPU vs Metal
// ===========================================================================

#[test]
fn test_decoder_cpu_vs_metal() {
    skip_without_weights!(dir);
    skip_without_metal!();

    // Load reference encoder output from PyTorch to isolate decoder parity.
    // Using PyTorch encoder output avoids compounding encoder divergence.
    let (ref_enc_data, ref_enc_shape) = load_ref_npy(&dir, "ref_encoder_output");
    assert_eq!(ref_enc_shape, vec![1, 1500, 384]);

    // CPU forward.
    let mut cpu_model = load_model_on_device(&dir, &Device::Cpu);
    let enc_cpu = DynTensor::from_vec(ref_enc_data.clone(), &ref_enc_shape, &Device::Cpu)
        .expect("create CPU encoder output");
    let tokens_cpu = DynTensor::new(&[50258.0], &[1, 1], &Device::Cpu).expect("CPU token tensor");
    let logits_cpu = cpu_model
        .decode(&tokens_cpu, &enc_cpu, true, 0)
        .expect("CPU decoder forward");
    let cpu_data = logits_cpu.to_flat_vec::<f32>().unwrap();

    // Metal forward.
    let mut gpu_model = load_model_on_device(&dir, &Device::metal());
    let enc_gpu = DynTensor::from_vec(ref_enc_data, &ref_enc_shape, &Device::Cpu)
        .expect("create encoder output")
        .to_device(&Device::metal())
        .expect("transfer encoder output to Metal");
    let tokens_gpu = DynTensor::new(&[50258.0], &[1, 1], &Device::Cpu)
        .expect("token tensor")
        .to_device(&Device::metal())
        .expect("transfer tokens to Metal");
    let logits_gpu = gpu_model
        .decode(&tokens_gpu, &enc_gpu, true, 0)
        .expect("Metal decoder forward");
    let gpu_data = logits_gpu
        .to_device(&Device::Cpu)
        .expect("transfer logits to CPU")
        .to_flat_vec::<f32>()
        .unwrap();

    assert_eq!(
        logits_cpu.dims(),
        logits_gpu.dims(),
        "decoder output shape mismatch"
    );
    eprintln!("Decoder logits shape: {:?}", logits_cpu.dims());

    assert_parity("Decoder CPU vs Metal", &cpu_data, &gpu_data, 0.9999);

    // Verify top-1 token matches.
    let cpu_argmax = cpu_data
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .map(|(i, _)| i)
        .unwrap();
    let gpu_argmax = gpu_data
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .map(|(i, _)| i)
        .unwrap();
    eprintln!("  CPU argmax = {cpu_argmax}, GPU argmax = {gpu_argmax}");
    assert_eq!(
        cpu_argmax, gpu_argmax,
        "Top-1 token should match between CPU and Metal"
    );
}

// ===========================================================================
// C. Attention Layer Isolation (single encoder block)
// ===========================================================================

#[test]
fn test_attention_cpu_vs_metal() {
    // Tests a smaller forward pass to isolate attention-layer parity.
    // Uses a short mel input (fewer frames) to reduce compute.
    skip_without_weights!(dir);
    skip_without_metal!();

    let (mel_data, mel_shape) = load_ref_npy(&dir, "ref_mel_input");

    // Use a narrow slice: first 200 frames (out of 3000) to keep it fast.
    // After conv2 (stride 2), this yields 100 timesteps through attention.
    let narrow_frames = 200;
    let mel_bins = mel_shape[1]; // 80
    let mut narrow_data = Vec::with_capacity(mel_bins * narrow_frames);
    for bin in 0..mel_bins {
        let start = bin * mel_shape[2];
        narrow_data.extend_from_slice(&mel_data[start..start + narrow_frames]);
    }
    let narrow_shape = vec![1, mel_bins, narrow_frames];

    // CPU forward.
    let mut cpu_model = load_model_on_device(&dir, &Device::Cpu);
    let mel_cpu = DynTensor::from_vec(narrow_data.clone(), &narrow_shape, &Device::Cpu)
        .expect("create CPU narrow mel");
    let enc_cpu = cpu_model
        .encode(&mel_cpu)
        .expect("CPU encoder forward (narrow)");
    let cpu_data = enc_cpu.to_flat_vec::<f32>().unwrap();

    // Metal forward.
    let mut gpu_model = load_model_on_device(&dir, &Device::metal());
    let mel_gpu = DynTensor::from_vec(narrow_data, &narrow_shape, &Device::Cpu)
        .expect("create narrow mel")
        .to_device(&Device::metal())
        .expect("transfer narrow mel to Metal");
    let enc_gpu = gpu_model
        .encode(&mel_gpu)
        .expect("Metal encoder forward (narrow)");
    let gpu_data = enc_gpu
        .to_device(&Device::Cpu)
        .expect("transfer to CPU")
        .to_flat_vec::<f32>()
        .unwrap();

    assert_eq!(
        enc_cpu.dims(),
        enc_gpu.dims(),
        "narrow encoder output shape mismatch"
    );
    eprintln!("Attention test encoder output shape: {:?}", enc_cpu.dims());

    assert_parity(
        "Attention CPU vs Metal (narrow encoder)",
        &cpu_data,
        &gpu_data,
        0.9999,
    );
}

// ===========================================================================
// D. Full Pipeline CPU vs Metal
// ===========================================================================

#[test]
fn test_full_pipeline_cpu_vs_metal() {
    // End-to-end: mel -> encode -> decode -> logits on both CPU and Metal.
    skip_without_weights!(dir);
    skip_without_metal!();

    let (mel_data, mel_shape) = load_ref_npy(&dir, "ref_mel_input");

    // --- CPU pipeline ---
    let mut cpu_model = load_model_on_device(&dir, &Device::Cpu);
    let mel_cpu =
        DynTensor::from_vec(mel_data.clone(), &mel_shape, &Device::Cpu).expect("CPU mel tensor");
    let enc_cpu = cpu_model.encode(&mel_cpu).expect("CPU encode");
    let tokens_cpu = DynTensor::new(&[50258.0], &[1, 1], &Device::Cpu).expect("CPU token tensor");
    let logits_cpu = cpu_model
        .decode(&tokens_cpu, &enc_cpu, true, 0)
        .expect("CPU decode");
    let cpu_logits = logits_cpu.to_flat_vec::<f32>().unwrap();
    let cpu_enc_data = enc_cpu.to_flat_vec::<f32>().unwrap();

    // --- Metal pipeline ---
    let mut gpu_model = load_model_on_device(&dir, &Device::metal());
    let mel_gpu = DynTensor::from_vec(mel_data, &mel_shape, &Device::Cpu)
        .expect("mel tensor")
        .to_device(&Device::metal())
        .expect("mel to Metal");
    let enc_gpu = gpu_model.encode(&mel_gpu).expect("Metal encode");
    let tokens_gpu = DynTensor::new(&[50258.0], &[1, 1], &Device::Cpu)
        .expect("token tensor")
        .to_device(&Device::metal())
        .expect("tokens to Metal");
    let logits_gpu = gpu_model
        .decode(&tokens_gpu, &enc_gpu, true, 0)
        .expect("Metal decode");
    let gpu_logits = logits_gpu
        .to_device(&Device::Cpu)
        .expect("logits to CPU")
        .to_flat_vec::<f32>()
        .unwrap();
    let gpu_enc_data = enc_gpu
        .to_device(&Device::Cpu)
        .expect("enc to CPU")
        .to_flat_vec::<f32>()
        .unwrap();

    // Check encoder parity first.
    assert_parity(
        "Full pipeline: encoder",
        &cpu_enc_data,
        &gpu_enc_data,
        0.9999,
    );

    // Check decoder parity (end-to-end, including any encoder divergence).
    assert_parity(
        "Full pipeline: decoder logits",
        &cpu_logits,
        &gpu_logits,
        0.999,
    );

    // Verify both produce the same argmax token.
    let cpu_argmax = cpu_logits
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .map(|(i, _)| i)
        .unwrap();
    let gpu_argmax = gpu_logits
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .map(|(i, _)| i)
        .unwrap();
    eprintln!("Full pipeline: CPU argmax={cpu_argmax}, GPU argmax={gpu_argmax}");
    assert_eq!(
        cpu_argmax, gpu_argmax,
        "Full pipeline: top-1 token should match between CPU and Metal"
    );
}

// ===========================================================================
// E. Multi-step Autoregressive Decode CPU vs Metal
// ===========================================================================

#[test]
fn test_autoregressive_decode_cpu_vs_metal() {
    // Multi-step greedy decode on both CPU and Metal, compare token sequences.
    skip_without_weights!(dir);
    skip_without_metal!();

    // Use PyTorch encoder output to isolate decoder parity.
    let (ref_enc_data, ref_enc_shape) = load_ref_npy(&dir, "ref_encoder_output");

    let mut cpu_model = load_model_on_device(&dir, &Device::Cpu);
    let mut gpu_model = load_model_on_device(&dir, &Device::metal());

    let enc_cpu = DynTensor::from_vec(ref_enc_data.clone(), &ref_enc_shape, &Device::Cpu)
        .expect("CPU encoder output");
    let enc_gpu = DynTensor::from_vec(ref_enc_data, &ref_enc_shape, &Device::Cpu)
        .expect("encoder output")
        .to_device(&Device::metal())
        .expect("encoder output to Metal");

    let num_steps = 5;
    let mut cpu_tokens = vec![50258u32]; // SOT
    let mut gpu_tokens = vec![50258u32];

    for step in 0..num_steps {
        let flush = step == 0;
        let last_cpu_tok = *cpu_tokens.last().unwrap() as f32;
        let last_gpu_tok = *gpu_tokens.last().unwrap() as f32;

        // CPU step.
        let tok_cpu = DynTensor::new(&[last_cpu_tok], &[1, 1], &Device::Cpu).expect("CPU token");
        let logits_cpu = cpu_model
            .decode(&tok_cpu, &enc_cpu, flush, step)
            .unwrap_or_else(|e| panic!("CPU decode step {step}: {e}"));
        let cpu_logits_flat = logits_cpu.to_flat_vec::<f32>().unwrap();

        // Metal step.
        let tok_gpu = DynTensor::new(&[last_gpu_tok], &[1, 1], &Device::Cpu)
            .expect("token")
            .to_device(&Device::metal())
            .expect("token to Metal");
        let logits_gpu = gpu_model
            .decode(&tok_gpu, &enc_gpu, flush, step)
            .unwrap_or_else(|e| panic!("Metal decode step {step}: {e}"));
        let gpu_logits_flat = logits_gpu
            .to_device(&Device::Cpu)
            .expect("logits to CPU")
            .to_flat_vec::<f32>()
            .unwrap();

        // Compare logits at each step.
        let cos_sim = cosine_similarity(&cpu_logits_flat, &gpu_logits_flat);
        let mad = max_abs_diff(&cpu_logits_flat, &gpu_logits_flat);
        eprintln!("Step {step}: cosine_sim={cos_sim:.8}, max_abs_diff={mad:.6e}");
        assert!(
            cos_sim > 0.999,
            "Step {step}: cosine_sim={cos_sim:.8} below 0.999 threshold"
        );

        // Greedy argmax.
        let cpu_next = cpu_logits_flat
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(i, _)| i as u32)
            .unwrap();
        let gpu_next = gpu_logits_flat
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(i, _)| i as u32)
            .unwrap();

        cpu_tokens.push(cpu_next);
        gpu_tokens.push(gpu_next);
    }

    eprintln!("CPU tokens: {cpu_tokens:?}");
    eprintln!("GPU tokens: {gpu_tokens:?}");
    assert_eq!(
        cpu_tokens, gpu_tokens,
        "Greedy decode should produce identical token sequences on CPU and Metal"
    );
}

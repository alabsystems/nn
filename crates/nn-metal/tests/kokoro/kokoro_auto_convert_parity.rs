// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Kokoro auto-converter parity test (#4276).
//!
//! Validates that the compiled GPU pipeline (CompiledKokoro) produces output
//! matching the manually-built CPU pipeline (KokoroModel) when given identical
//! weights and synthetic inputs.
//!
//! This test requires only KOKORO_WEIGHTS (no KOKORO_REFERENCE needed). It
//! builds both paths from the same safetensors file and compares audio output.
//!
//! # Architecture
//!
//! The full Kokoro model cannot be auto-converted as a single torch.export
//! graph because `length_regulate` performs CPU readback mid-forward (dynamic
//! repeat based on predicted durations). Instead, nn uses a hand-built
//! multi-segment CompiledKokoro that mirrors the KokoroModel architecture.
//!
//! This test proves the compiled pipeline is functionally equivalent to the
//! manual builder by comparing end-to-end audio output:
//!
//! 1. Load production weights into KokoroModel (CPU, manual builder)
//! 2. Load same weights into CompiledKokoro (GPU, compiled pipeline)
//! 3. Feed identical synthetic inputs through both
//! 4. Compare PCM audio output within tolerance
//!
//! # Acceptance Criteria
//!
//! - Audio length: equal (same phonemes -> same frame count)
//! - Cosine similarity: > 0.99
//! - Max-abs difference: < 0.05
//!
//! Part of #4276 (Kokoro auto-converter parity test).

use std::collections::HashMap;
use std::path::Path;

use nn_core::dyn_tensor::DynTensor;
use nn_core::{DType, Device, VarBuilder};
use nn_metal::compiled_kokoro::CompiledKokoro;
use nn_models::kokoro_tts::{KokoroConfig, KokoroModel};

// -- Env-var gating -----------------------------------------------------------

fn require_weights() -> Option<String> {
    super::kokoro_test_env::require_kokoro_weights("Kokoro auto-converter parity test skipped.")
}

// -- Helpers ------------------------------------------------------------------

fn cpu() -> Device {
    Device::Cpu
}

/// Build synthetic input tensors for Kokoro inference.
///
/// Returns `(input_ids, style, speed)`:
/// - `input_ids`: `[1, 8]` -- short phoneme sequence
/// - `style`: `[1, 256]` -- voice embedding (2 * style_dim=128)
/// - `speed`: 1.0
fn synthetic_inputs() -> (DynTensor, DynTensor, f32) {
    let input_ids = DynTensor::from_vec(
        vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
        &[1, 8],
        &cpu(),
    )
    .expect("input_ids");
    let style = DynTensor::full(&[1, 256], 0.01, DType::F32, &cpu()).expect("style");
    (input_ids, style, 1.0)
}

/// Load safetensors file into a HashMap<String, DynTensor> on CPU.
fn load_safetensors_to_map(path: &Path) -> HashMap<String, DynTensor> {
    let data = std::fs::read(path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
    let tensors = safetensors::SafeTensors::deserialize(&data)
        .unwrap_or_else(|e| panic!("parse {}: {e}", path.display()));
    let device = Device::Cpu;
    let mut map = HashMap::new();
    for name in tensors.names() {
        let view = tensors.tensor(name).unwrap();
        let shape: Vec<usize> = view.shape().to_vec();
        let numel: usize = shape.iter().product();
        let tensor = match view.dtype() {
            safetensors::Dtype::F32 => {
                let floats: Vec<f32> = view
                    .data()
                    .chunks_exact(4)
                    .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                    .collect();
                assert_eq!(floats.len(), numel, "F32 count mismatch for {name}");
                DynTensor::new(&floats, &shape, &device).unwrap()
            }
            safetensors::Dtype::F16 => {
                let floats: Vec<f32> = view
                    .data()
                    .chunks_exact(2)
                    .map(|c| half::f16::from_le_bytes([c[0], c[1]]).to_f32())
                    .collect();
                assert_eq!(floats.len(), numel, "F16 count mismatch for {name}");
                DynTensor::new(&floats, &shape, &device).unwrap()
            }
            safetensors::Dtype::BF16 => {
                let floats: Vec<f32> = view
                    .data()
                    .chunks_exact(2)
                    .map(|c| half::bf16::from_le_bytes([c[0], c[1]]).to_f32())
                    .collect();
                assert_eq!(floats.len(), numel, "BF16 count mismatch for {name}");
                DynTensor::new(&floats, &shape, &device).unwrap()
            }
            dt => panic!("unsupported dtype {dt:?} for tensor {name}"),
        };
        map.insert(name.to_string(), tensor);
    }
    map
}

fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
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
    if denom < 1e-12 {
        0.0
    } else {
        (dot / denom) as f32
    }
}

fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max)
}

// -- Tests --------------------------------------------------------------------

/// Auto-converter parity: CompiledKokoro (GPU compiled) vs KokoroModel (CPU manual).
///
/// Both paths load from the same safetensors file, run forward with identical
/// synthetic inputs, and compare PCM audio output.
///
/// This is the core parity test for #4276: it proves the compiled GPU pipeline
/// (the target of the auto-converter) produces equivalent output to the
/// manually-built CPU model.
///
/// Known divergences absorbed by tolerance:
/// - Rounding: CPU banker's rounding vs GPU add(0.5)+floor
/// - iSTFT: CPU kokoro_istft (no center) vs GPU IstftGpuBasis (center=true)
/// - Float ordering: GPU FTZ, different reduction order in matmul/softmax
#[test]
fn test_auto_convert_parity_synthetic_inputs() {
    let weights_path = match require_weights() {
        Some(p) => p,
        None => return,
    };

    super::test_utils::gpu_init();
    let cache = super::test_utils::metal_setup();

    let (input_ids, style, speed) = synthetic_inputs();

    // --- CPU path: KokoroModel (manual builder) ---
    let weight_map = load_safetensors_to_map(Path::new(&weights_path));
    let vb = VarBuilder::from_tensors(weight_map, DType::F32, &Device::Cpu);
    let config = KokoroConfig::default();
    let cpu_model = KokoroModel::load(&vb, &config).expect("KokoroModel::load failed");
    let cpu_audio_tensor = cpu_model
        .forward_audio(&input_ids, &style, speed)
        .expect("CPU forward_audio failed");
    let cpu_audio = cpu_audio_tensor
        .to_flat_vec::<f32>()
        .expect("CPU audio to f32");

    // --- GPU compiled path: CompiledKokoro ---
    // Use Warn rejection policy: synthetic phoneme IDs [0..7] may produce
    // click artifacts with production weights. This is expected for arbitrary
    // input, not a quality bug. Part of #4262.
    let mut hb = nn_tts_verify::HardBoundsConfig::default();
    hb.rejection_policy = nn_tts_verify::RejectionPolicy::Warn;

    // SAFETY: safetensors file is not modified while alive.
    let mut compiled = unsafe {
        CompiledKokoro::load_with_hard_bounds(&weights_path, hb)
            .expect("CompiledKokoro::load failed")
    };

    let (gpu_audio_tensor, cert) = compiled
        .synthesize(&input_ids, &style, speed, &cache)
        .expect("GPU synthesize failed");

    assert!(
        cert.overall_passed,
        "GPU TTS hard bounds failed: {:?}",
        cert.hard_bounds
            .iter()
            .filter(|b| !b.passed)
            .collect::<Vec<_>>()
    );

    let gpu_audio = gpu_audio_tensor
        .to_device(&Device::Cpu)
        .expect("GPU to CPU transfer")
        .to_flat_vec::<f32>()
        .expect("GPU audio to f32");

    // --- Compare ---
    let min_len = cpu_audio.len().min(gpu_audio.len());
    assert!(min_len > 0, "both audio outputs must be non-empty");

    let cpu_slice = &cpu_audio[..min_len];
    let gpu_slice = &gpu_audio[..min_len];

    let len_ratio = cpu_audio.len() as f64 / gpu_audio.len() as f64;
    let cos_sim = cosine_similarity(cpu_slice, gpu_slice);
    let max_diff = max_abs_diff(cpu_slice, gpu_slice);

    eprintln!(
        "Auto-converter parity (synthetic): cpu_len={}, gpu_len={}, ratio={:.4}, \
         cosine={:.6}, max_abs={:.6}",
        cpu_audio.len(),
        gpu_audio.len(),
        len_ratio,
        cos_sim,
        max_diff,
    );

    // AC: Audio length -- equal (same phonemes -> same frame count).
    assert_eq!(
        cpu_audio.len(),
        gpu_audio.len(),
        "Audio length mismatch: CPU={} vs GPU={}",
        cpu_audio.len(),
        gpu_audio.len(),
    );

    // AC: Cosine similarity > 0.99.
    assert!(
        cos_sim > 0.99,
        "Cosine similarity {cos_sim:.6} below 0.99 threshold"
    );

    // AC: Max-abs difference < 0.05.
    assert!(
        max_diff < 0.05,
        "Max-abs difference {max_diff:.6} exceeds 0.05 threshold"
    );
}

/// Auto-converter parity: both paths produce finite, non-zero audio.
///
/// A weaker check that exercises the basic pipeline without tight numerical
/// comparison. Catches gross failures (NaN, zero output, wrong shape).
#[test]
fn test_auto_convert_parity_output_sanity() {
    let weights_path = match require_weights() {
        Some(p) => p,
        None => return,
    };

    super::test_utils::gpu_init();
    let cache = super::test_utils::metal_setup();

    let (input_ids, style, speed) = synthetic_inputs();

    // --- CPU path ---
    let weight_map = load_safetensors_to_map(Path::new(&weights_path));
    let vb = VarBuilder::from_tensors(weight_map, DType::F32, &Device::Cpu);
    let config = KokoroConfig::default();
    let cpu_model = KokoroModel::load(&vb, &config).expect("KokoroModel::load");
    let cpu_audio = cpu_model
        .forward_audio(&input_ids, &style, speed)
        .expect("CPU forward_audio")
        .to_flat_vec::<f32>()
        .expect("CPU f32");

    // --- GPU path ---
    let mut hb = nn_tts_verify::HardBoundsConfig::default();
    hb.rejection_policy = nn_tts_verify::RejectionPolicy::Warn;
    let mut compiled = unsafe {
        CompiledKokoro::load_with_hard_bounds(&weights_path, hb).expect("CompiledKokoro::load")
    };
    let (gpu_tensor, _cert) = compiled
        .synthesize(&input_ids, &style, speed, &cache)
        .expect("GPU synthesize");
    let gpu_audio = gpu_tensor
        .to_device(&Device::Cpu)
        .expect("GPU to CPU")
        .to_flat_vec::<f32>()
        .expect("GPU f32");

    // Sanity: non-empty.
    assert!(!cpu_audio.is_empty(), "CPU audio is empty");
    assert!(!gpu_audio.is_empty(), "GPU audio is empty");

    // Sanity: no NaN.
    assert!(
        cpu_audio.iter().all(|v| v.is_finite()),
        "CPU audio contains non-finite values"
    );
    assert!(
        gpu_audio.iter().all(|v| v.is_finite()),
        "GPU audio contains non-finite values"
    );

    // Sanity: not all zeros (model actually ran).
    let cpu_energy: f64 = cpu_audio.iter().map(|v| f64::from(*v).powi(2)).sum();
    let gpu_energy: f64 = gpu_audio.iter().map(|v| f64::from(*v).powi(2)).sum();
    assert!(
        cpu_energy > 1e-6,
        "CPU audio is near-zero (energy={cpu_energy:.6e})"
    );
    assert!(
        gpu_energy > 1e-6,
        "GPU audio is near-zero (energy={gpu_energy:.6e})"
    );

    // Sanity: audio is in reasonable range [-2, 2] (production audio in [-1, 1]).
    let cpu_max = cpu_audio.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
    let gpu_max = gpu_audio.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
    assert!(
        cpu_max < 2.0,
        "CPU audio out of range: max_abs={cpu_max:.4}"
    );
    assert!(
        gpu_max < 2.0,
        "GPU audio out of range: max_abs={gpu_max:.4}"
    );

    eprintln!(
        "Auto-converter sanity: cpu_len={}, gpu_len={}, cpu_energy={:.4e}, gpu_energy={:.4e}",
        cpu_audio.len(),
        gpu_audio.len(),
        cpu_energy,
        gpu_energy,
    );
}

/// Documents the auto-converter gap for full end-to-end Kokoro conversion.
///
/// Full Kokoro cannot be auto-converted as a single torch.export graph because
/// `length_regulate` performs CPU readback mid-forward (dynamic repeat based on
/// predicted durations). The model requires 5 segments with CPU orchestration.
///
/// Current coverage:
/// - Per-segment auto-conversion: nn-import/tests/import/kokoro_converter_parity.rs
/// - Cross-path parity (synthetic): test_auto_convert_parity_synthetic_inputs (above)
/// - Cross-path parity (reference): compiled_model/kokoro_cross_path_parity.rs
/// - L3 PyTorch parity: kokoro/kokoro_l3_parity.rs
///
/// Missing for full auto-converter E2E:
/// - Multi-segment composition API in nn-import
/// - Or fixed-length length_regulate variant (no CPU readback)
#[test]
fn test_auto_convert_gap_documented() {
    eprintln!("== Kokoro Auto-Converter Parity Coverage ==");
    eprintln!();
    eprintln!("1. Per-segment import+compile: nn-import/tests/import/kokoro_converter_parity.rs");
    eprintln!(
        "2. Cross-path parity (synthetic): this file (test_auto_convert_parity_synthetic_inputs)"
    );
    eprintln!("3. Cross-path parity (reference): compiled_model/kokoro_cross_path_parity.rs");
    eprintln!("4. L3 PyTorch parity: kokoro/kokoro_l3_parity.rs");
    eprintln!();
    eprintln!("Gap: full E2E auto-converter (single torch.export graph).");
    eprintln!("Reason: length_regulate does CPU readback mid-forward.");
    eprintln!("Needed: multi-segment composition API OR fixed-length variant.");
}

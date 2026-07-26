// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Cross-path parity test: CompiledKokoro (GPU) vs KokoroModel (CPU).
//!
//! Proves that the two Rust audio generation paths produce matching output
//! when given identical inputs. Catches GPU pipeline regressions (like #2928)
//! that diverge from the CPU reference path.
//!
//! Gated on environment variables:
//!   - `KOKORO_WEIGHTS`: path to kokoro_v1_0.safetensors model weights
//!   - `KOKORO_REFERENCE`: path to kokoro_reference.safetensors (for input data)
//!
//! Acceptance criteria:
//!   - Audio length: equal (same phoneme count → same frame count)
//!   - Cosine similarity: > 0.99
//!   - Max-abs difference: < 0.05 (absorbs rounding + float ordering + GPU FTZ)
//!
//! Part of #2218 (Kokoro epic), #2981 (F16 mixed precision).

use std::collections::HashMap;
use std::path::Path;

use nn_core::dyn_tensor::DynTensor;
use nn_core::{DType, Device, VarBuilder};
use nn_metal::compiled_kokoro::CompiledKokoro;
use nn_models::kokoro_tts::{KokoroConfig, KokoroModel};

// -- Env-var gating -----------------------------------------------------------

fn require_paths() -> Option<(String, String)> {
    super::kokoro_test_env::require_kokoro_weights_and_reference(
        "compiled vs CPU Kokoro parity not run.",
    )
}

// -- Safetensors loading (CPU) ------------------------------------------------

fn convert_tensor(
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

fn load_safetensors_to_map(path: &Path) -> HashMap<String, DynTensor> {
    let data = std::fs::read(path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
    let tensors = safetensors::SafeTensors::deserialize(&data)
        .unwrap_or_else(|e| panic!("parse {}: {e}", path.display()));
    let device = Device::Cpu;
    let mut map = HashMap::new();
    for name in tensors.names() {
        let view = tensors.tensor(name).unwrap();
        map.insert(name.to_string(), convert_tensor(&view, name, &device));
    }
    map
}

// -- Helper: cosine similarity ------------------------------------------------

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

// -- Test ---------------------------------------------------------------------

/// Cross-path parity: CompiledKokoro (GPU compiled) vs KokoroModel (CPU).
///
/// Loads both models from the same weights, runs with identical inputs from
/// the reference file, and compares PCM audio output.
///
/// Known divergences absorbed by tolerance:
/// - Rounding: CPU banker's rounding vs GPU add(0.5)+floor
/// - iSTFT: CPU kokoro_istft (no center) vs GPU IstftGpuBasis (center=true)
/// - Float ordering: GPU FTZ, different reduction order
#[test]
fn test_compiled_vs_cpu_kokoro_parity() {
    let (weights_path, reference_path) = match require_paths() {
        Some(p) => p,
        None => return,
    };

    super::test_utils::gpu_init();
    let cache = super::test_utils::metal_setup();

    // Load reference data for inputs.
    let ref_trace =
        nn_reftest::load_safetensors(&reference_path).expect("load reference safetensors");
    let input_ids_ref = ref_trace
        .get_by_name("input_ids")
        .expect("missing 'input_ids'");
    let style_ref = ref_trace.get_by_name("style").expect("missing 'style'");
    let speed_ref = ref_trace.get_by_name("speed").expect("missing 'speed'");

    let input_ids = DynTensor::from_vec(
        input_ids_ref.data.clone(),
        &input_ids_ref.shape,
        &Device::Cpu,
    )
    .expect("build input_ids");
    let style = DynTensor::from_vec(style_ref.data.clone(), &style_ref.shape, &Device::Cpu)
        .expect("build style");
    let speed = speed_ref.data[0];

    // --- CPU path: KokoroModel ---
    let weight_map = load_safetensors_to_map(Path::new(&weights_path));
    let vb = VarBuilder::from_tensors(weight_map, DType::F32, &Device::Cpu);
    let config = KokoroConfig::default();
    let cpu_model = KokoroModel::load(&vb, &config).expect("load KokoroModel");
    let cpu_audio_tensor = cpu_model
        .forward_audio(&input_ids, &style, speed)
        .expect("CPU forward_audio");
    let cpu_audio = cpu_audio_tensor
        .to_flat_vec::<f32>()
        .expect("CPU audio to f32");

    // --- GPU compiled path: CompiledKokoro ---
    // Use Warn policy: reference tokens may produce click artifacts with
    // production weights that fail the no_clicks hard bound. Part of #4262.
    let mut hb = nn_tts_verify::HardBoundsConfig::default();
    hb.rejection_policy = nn_tts_verify::RejectionPolicy::Warn;
    // SAFETY: safetensors file is not modified while alive.
    let mut compiled = unsafe {
        CompiledKokoro::load_with_hard_bounds(&weights_path, hb).expect("load CompiledKokoro")
    };
    let (gpu_audio_tensor, cert) = compiled
        .synthesize(&input_ids, &style, speed, &cache)
        .expect("GPU synthesize");

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
        .expect("GPU to CPU")
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
        "Cross-path parity: cpu_len={}, gpu_len={}, ratio={:.4}, cosine={:.6}, max_abs={:.6}",
        cpu_audio.len(),
        gpu_audio.len(),
        len_ratio,
        cos_sim,
        max_diff,
    );

    // AC: Audio length — equal (same phonemes → same frames).
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

/// F16 mixed-precision cross-path parity (#2981): CompiledKokoro F16 vs F32.
///
/// Compares the F16 mixed-precision GPU pipeline against the F32 GPU baseline.
/// Both paths use identical compiled dispatch plans; the only difference is
/// F16 vs F32 precision. This isolates F16 quantization error from other
/// GPU-vs-CPU divergences tested above.
///
/// Wider tolerances than F32-vs-CPU:
///   - Cosine similarity: > 0.95 (F16 has ~3 decimal digits of precision)
///   - Max-abs difference: < 0.15 (accumulated F16 rounding through ~180 steps)
#[test]
fn test_compiled_f16_vs_f32_kokoro_parity() {
    let (weights_path, reference_path) = match require_paths() {
        Some(p) => p,
        None => return,
    };

    super::test_utils::gpu_init();
    let cache = super::test_utils::metal_setup();

    // Load reference data for inputs.
    let ref_trace =
        nn_reftest::load_safetensors(&reference_path).expect("load reference safetensors");
    let input_ids_ref = ref_trace
        .get_by_name("input_ids")
        .expect("missing 'input_ids'");
    let style_ref = ref_trace.get_by_name("style").expect("missing 'style'");
    let speed_ref = ref_trace.get_by_name("speed").expect("missing 'speed'");

    let input_ids = DynTensor::from_vec(
        input_ids_ref.data.clone(),
        &input_ids_ref.shape,
        &Device::Cpu,
    )
    .expect("build input_ids");
    let style = DynTensor::from_vec(style_ref.data.clone(), &style_ref.shape, &Device::Cpu)
        .expect("build style");
    let speed = speed_ref.data[0];

    // Use Warn policy: reference tokens may produce click artifacts with
    // production weights that fail the no_clicks hard bound. Part of #4262.
    let mut hb_f32 = nn_tts_verify::HardBoundsConfig::default();
    hb_f32.rejection_policy = nn_tts_verify::RejectionPolicy::Warn;
    let hb_f16 = hb_f32.clone();

    // --- F32 GPU baseline ---
    // SAFETY: safetensors file is not modified while alive.
    let mut f32_compiled = unsafe {
        CompiledKokoro::load_with_hard_bounds(&weights_path, hb_f32)
            .expect("load CompiledKokoro F32")
    };
    let (f32_audio_tensor, f32_cert) = f32_compiled
        .synthesize(&input_ids, &style, speed, &cache)
        .expect("F32 GPU synthesize");

    assert!(
        f32_cert.overall_passed,
        "F32 GPU TTS hard bounds failed: {:?}",
        f32_cert
            .hard_bounds
            .iter()
            .filter(|b| !b.passed)
            .collect::<Vec<_>>()
    );

    let f32_audio = f32_audio_tensor
        .to_device(&Device::Cpu)
        .expect("F32 GPU to CPU")
        .to_flat_vec::<f32>()
        .expect("F32 audio to f32");

    // --- F16 mixed-precision GPU ---
    // SAFETY: safetensors file is not modified while alive.
    let mut f16_compiled = unsafe {
        CompiledKokoro::load_with_hard_bounds(&weights_path, hb_f16)
            .expect("load CompiledKokoro F16")
    }
    .with_autocast();
    let (f16_audio_tensor, f16_cert) = f16_compiled
        .synthesize(&input_ids, &style, speed, &cache)
        .expect("F16 GPU synthesize");

    assert!(
        f16_cert.overall_passed,
        "F16 GPU TTS hard bounds failed: {:?}",
        f16_cert
            .hard_bounds
            .iter()
            .filter(|b| !b.passed)
            .collect::<Vec<_>>()
    );

    let f16_audio = f16_audio_tensor
        .to_device(&Device::Cpu)
        .expect("F16 GPU to CPU")
        .to_flat_vec::<f32>()
        .expect("F16 audio to f32");

    // --- Compare F16 vs F32 ---
    let min_len = f32_audio.len().min(f16_audio.len());
    assert!(min_len > 0, "both audio outputs must be non-empty");

    let f32_slice = &f32_audio[..min_len];
    let f16_slice = &f16_audio[..min_len];

    let len_ratio = f32_audio.len() as f64 / f16_audio.len() as f64;
    let cos_sim = cosine_similarity(f32_slice, f16_slice);
    let max_diff = max_abs_diff(f32_slice, f16_slice);

    eprintln!(
        "F16 vs F32 parity: f32_len={}, f16_len={}, ratio={:.4}, cosine={:.6}, max_abs={:.6}",
        f32_audio.len(),
        f16_audio.len(),
        len_ratio,
        cos_sim,
        max_diff,
    );

    // AC: Audio length must be equal (same inputs → same frame count).
    assert_eq!(
        f32_audio.len(),
        f16_audio.len(),
        "Audio length mismatch: F32={} vs F16={}",
        f32_audio.len(),
        f16_audio.len(),
    );

    // AC: Cosine similarity > 0.95.
    // F16 has ~3 decimal digits of precision; accumulated rounding through
    // ~180 compiled steps widens divergence vs the F32-vs-CPU 0.99 threshold.
    assert!(
        cos_sim > 0.95,
        "F16 vs F32 cosine similarity {cos_sim:.6} below 0.95 threshold"
    );

    // AC: Max-abs difference < 0.15.
    // Wider than F32-vs-CPU 0.05 due to F16 quantization in normalization
    // layers, matmul accumulation, and fused ResBlock chains.
    assert!(
        max_diff < 0.15,
        "F16 vs F32 max-abs difference {max_diff:.6} exceeds 0.15 threshold"
    );
}

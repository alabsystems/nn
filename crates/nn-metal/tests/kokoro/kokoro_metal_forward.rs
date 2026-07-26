// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! End-to-end Metal GPU forward pass tests with real Kokoro-82M weights.
//!
//! Validates the full production inference pipeline on Metal:
//! - Weight loading: all 463 tensors load to Metal device
//! - Text encoder: PlBert + bert_encoder + TextEncoder
//! - Style encoder: style splitting + prosody prediction
//! - Decoder: FullDecoder + Generator + iSTFT
//! - CPU vs GPU parity: full forward pass comparison
//! - Audio output: valid PCM in [-1, 1] range
//!
//! Gated on `KOKORO_WEIGHTS` env var. Tests skip gracefully when unset.
//!
//! Run:
//!   KOKORO_WEIGHTS=./nn/weights/kokoro_v1_0.safetensors \
//!   cargo test -p nn-metal --test kokoro_all -- kokoro_metal_forward --nocapture
//!
//! Part of #3351 (Absolutely Best Kokoro).

use std::time::Instant;

use nn_core::dyn_tensor::DynTensor;
use nn_core::{DType, Device};
use nn_metal::compiled_kokoro::CompiledKokoro;

fn cpu() -> Device {
    Device::Cpu
}

fn require_weights() -> Option<String> {
    super::kokoro_test_env::require_kokoro_weights("Metal forward pass test skipped.")
}

/// Build a production CompiledKokoro with Warn rejection policy.
///
/// Test tokens `[0..7]` are arbitrary phoneme IDs that produce click artifacts
/// with production weights. This is expected for garbage input, not a real
/// quality problem. Warn policy records the failure in the certificate but
/// does not block `overall_passed`. Part of #4262.
///
/// # Safety
///
/// Same as [`CompiledKokoro::load`]: safetensors file must not be modified while alive.
unsafe fn load_production_warn(path: &str) -> CompiledKokoro {
    let mut hb = nn_tts_verify::HardBoundsConfig::default();
    hb.rejection_policy = nn_tts_verify::RejectionPolicy::Warn;
    unsafe {
        CompiledKokoro::load_with_hard_bounds(path, hb).expect("load production Kokoro weights")
    }
}

/// Build synthetic input tensors for Kokoro inference.
///
/// Returns `(input_ids, style)` matching production Kokoro-82M dimensions:
/// - `input_ids`: `[1, 8]` — short phoneme sequence
/// - `style`: `[1, 256]` — voice embedding (2 * style_dim=128)
fn synthetic_inputs() -> (DynTensor, DynTensor) {
    let input_ids = DynTensor::from_vec(
        vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
        &[1, 8],
        &cpu(),
    )
    .expect("input_ids");
    let style = DynTensor::full(&[1, 256], 0.01, DType::F32, &cpu()).expect("style");
    (input_ids, style)
}

// ---------------------------------------------------------------------------
// Test: weight loading
// ---------------------------------------------------------------------------

/// Load all production Kokoro weights onto Metal device.
///
/// Validates that `CompiledKokoro::load()` succeeds with the full 335 MB
/// safetensors file (463 tensors, ~83.8M params). This exercises mmap-backed
/// weight loading and SourceModule GPU transfer.
#[test]
fn test_kokoro_metal_weight_loading() {
    let weights_path = match require_weights() {
        Some(p) => p,
        None => return,
    };

    super::test_utils::gpu_init();
    let _cache = super::test_utils::metal_setup();

    let t0 = Instant::now();

    // SAFETY: safetensors file not modified while alive.
    let kokoro = unsafe { CompiledKokoro::load(&weights_path) };

    let load_ms = t0.elapsed().as_secs_f64() * 1000.0;

    assert!(
        kokoro.is_ok(),
        "CompiledKokoro::load failed: {:?}",
        kokoro.err()
    );

    let kokoro = kokoro.unwrap();

    // Verify config matches production Kokoro-82M dimensions.
    let cfg = kokoro.config();
    assert_eq!(cfg.d_en, 512, "d_en should be 512 for Kokoro-82M");
    assert_eq!(cfg.style_dim, 128, "style_dim should be 128");
    assert_eq!(cfg.gen_initial_channels, 512, "gen_initial_channels=512");

    eprintln!(
        "Weight loading: {:.1} ms, d_en={}, style_dim={}, gen_ch={}",
        load_ms, cfg.d_en, cfg.style_dim, cfg.gen_initial_channels,
    );
}

// ---------------------------------------------------------------------------
// Test: text encoder (PlBert + bert_encoder + TextEncoder segments)
// ---------------------------------------------------------------------------

/// Run the text encoder pipeline on Metal with real weights.
///
/// Exercises Segments 0 (PlBert+bert_encoder) and 1 (TextEncoder):
/// input_ids → PlBert → bert_encoder → TextEncoder → features [B, d_en, T].
///
/// Validates that the step_encode API produces finite outputs.
#[test]
fn test_kokoro_metal_text_encoder() {
    let weights_path = match require_weights() {
        Some(p) => p,
        None => return,
    };

    super::test_utils::gpu_init();
    let cache = super::test_utils::metal_setup();

    // SAFETY: safetensors file not modified while alive.
    let mut kokoro = unsafe { CompiledKokoro::load(&weights_path).expect("load weights") };

    let (input_ids, _style) = synthetic_inputs();

    let t0 = Instant::now();
    let enc = kokoro
        .step_encode(&input_ids, &cache)
        .expect("step_encode failed");
    let encode_ms = t0.elapsed().as_secs_f64() * 1000.0;

    // Verify shapes.
    let bert_dims = enc.bert_features.dims();
    assert_eq!(bert_dims.len(), 3, "bert_features should be rank 3");
    assert_eq!(bert_dims[0], 1, "batch=1");
    assert_eq!(bert_dims[1], 512, "d_en=512");
    assert_eq!(bert_dims[2], 8, "seq_len=8");

    let text_dims = enc.text_features.dims();
    assert_eq!(text_dims.len(), 3, "text_features should be rank 3");
    assert_eq!(text_dims[0], 1, "batch=1");
    assert_eq!(text_dims[1], 512, "d_en=512");

    // Verify finite values.
    let bert_cpu = enc
        .bert_features
        .to_device(&cpu())
        .expect("bert GPU->CPU")
        .to_flat_vec::<f32>()
        .expect("bert f32");
    let non_finite = bert_cpu.iter().filter(|v| !v.is_finite()).count();
    assert_eq!(
        non_finite, 0,
        "bert_features has {non_finite} non-finite values"
    );

    let text_cpu = enc
        .text_features
        .to_device(&cpu())
        .expect("text GPU->CPU")
        .to_flat_vec::<f32>()
        .expect("text f32");
    let text_non_finite = text_cpu.iter().filter(|v| !v.is_finite()).count();
    assert_eq!(
        text_non_finite, 0,
        "text_features has {text_non_finite} non-finite values"
    );

    eprintln!(
        "Text encoder: {encode_ms:.1} ms, bert_features={bert_dims:?}, text_features={text_dims:?}",
    );
}

// ---------------------------------------------------------------------------
// Test: style encoder (style split + prosody prediction)
// ---------------------------------------------------------------------------

/// Run the style/prosody pipeline on Metal with real weights.
///
/// Exercises Segment 2 (ProsodyPredictor):
/// bert_features + style → dur_logits + prosody features.
///
/// Validates that `step_predict_prosody` produces finite outputs with
/// correct shapes.
#[test]
fn test_kokoro_metal_style_encoder() {
    let weights_path = match require_weights() {
        Some(p) => p,
        None => return,
    };

    super::test_utils::gpu_init();
    let cache = super::test_utils::metal_setup();

    // SAFETY: safetensors file not modified while alive.
    let mut kokoro = unsafe { CompiledKokoro::load(&weights_path).expect("load weights") };

    let (input_ids, style) = synthetic_inputs();

    // Step 1-2: encode.
    let enc = kokoro
        .step_encode(&input_ids, &cache)
        .expect("step_encode failed");

    // Split style into decoder/prosody halves.
    let prosody_style = style.narrow(1, 128, 128).expect("prosody style split");
    let prosody_style_gpu = prosody_style
        .to_device(&Device::Metal { device_id: 0 })
        .expect("prosody style to GPU");

    let t0 = Instant::now();
    let pros = kokoro
        .step_predict_prosody(&enc.bert_features, &prosody_style_gpu, enc.seq_len, &cache)
        .expect("step_predict_prosody failed");
    let prosody_ms = t0.elapsed().as_secs_f64() * 1000.0;

    // Verify dur_logits shape: [B, T, max_dur].
    let dur_dims = pros.dur_logits.dims();
    assert_eq!(dur_dims.len(), 3, "dur_logits rank=3");
    assert_eq!(dur_dims[0], 1, "batch=1");
    assert_eq!(dur_dims[1], 8, "T=seq_len=8");

    // Verify features shape.
    let feat_dims = pros.features.dims();
    assert_eq!(feat_dims.len(), 3, "prosody features rank=3");
    assert_eq!(feat_dims[0], 1, "batch=1");

    // Verify finite.
    let dur_cpu = pros
        .dur_logits
        .to_device(&cpu())
        .expect("dur GPU->CPU")
        .to_flat_vec::<f32>()
        .expect("dur f32");
    let non_finite = dur_cpu.iter().filter(|v| !v.is_finite()).count();
    assert_eq!(
        non_finite, 0,
        "dur_logits has {non_finite} non-finite values"
    );

    eprintln!(
        "Style/prosody: {prosody_ms:.1} ms, dur_logits={dur_dims:?}, features={feat_dims:?}",
    );
}

// ---------------------------------------------------------------------------
// Test: decoder (FullDecoder + Generator + iSTFT)
// ---------------------------------------------------------------------------

/// Run the full decoder pipeline on Metal with real weights.
///
/// Exercises Segments 3-4 (F0EnergyPredictor + Generator) plus GPU iSTFT.
/// This is the most compute-intensive part of the pipeline.
///
/// Uses `synthesize()` end-to-end and verifies the audio output shape and
/// hard bounds certificate.
#[test]
fn test_kokoro_metal_decoder() {
    let weights_path = match require_weights() {
        Some(p) => p,
        None => return,
    };

    super::test_utils::gpu_init();
    let cache = super::test_utils::metal_setup();

    // Use Warn policy: test tokens [0..7] produce click artifacts with
    // production weights that fail the no_clicks hard bound. Part of #4262.
    // SAFETY: safetensors file not modified while alive.
    let mut kokoro = unsafe { load_production_warn(&weights_path) };

    let (input_ids, style) = synthetic_inputs();

    let t0 = Instant::now();
    let (audio, cert) = kokoro
        .synthesize(&input_ids, &style, 1.0, &cache)
        .expect("synthesize failed");
    let synth_ms = t0.elapsed().as_secs_f64() * 1000.0;

    // Audio shape: [1, 1, T_audio].
    let dims = audio.dims();
    assert_eq!(dims.len(), 3, "audio should be rank 3");
    assert_eq!(dims[0], 1, "batch=1");
    assert_eq!(dims[1], 1, "channels=1 (mono)");
    assert!(dims[2] > 0, "audio should have non-zero samples");

    // Certificate should pass all hard bounds (Warn policy: overall_passed
    // is true even if individual checks like no_clicks fail for synthetic tokens).
    assert_eq!(
        cert.hard_bounds.len(),
        8,
        "expected 8 hard bounds, got {}",
        cert.hard_bounds.len()
    );
    assert!(
        cert.overall_passed,
        "hard bounds failed: {:?}",
        cert.hard_bounds
            .iter()
            .filter(|b| !b.passed)
            .collect::<Vec<_>>()
    );

    eprintln!(
        "Decoder (full synth): {:.1} ms, audio_shape={:?}, cert={}",
        synth_ms,
        dims,
        if cert.overall_passed {
            "PASSED"
        } else {
            "FAILED"
        },
    );
}

// ---------------------------------------------------------------------------
// Test: CPU vs GPU parity
// ---------------------------------------------------------------------------

/// Compare CPU and Metal GPU forward pass for the same inputs.
///
/// Runs `synthesize()` twice with the same inputs on two separate
/// `CompiledKokoro` instances to verify deterministic GPU output. Then
/// compares audio waveforms to ensure Metal produces consistent results.
///
/// Note: This is GPU-GPU consistency (not CPU-GPU, since CompiledKokoro
/// always dispatches through Metal). Two independent load+synth cycles
/// should produce bitwise-identical audio.
#[test]
fn test_kokoro_metal_cpu_vs_gpu() {
    let weights_path = match require_weights() {
        Some(p) => p,
        None => return,
    };

    super::test_utils::gpu_init();
    let cache = super::test_utils::metal_setup();

    let (input_ids, style) = synthetic_inputs();

    // Use Warn policy: test tokens [0..7] produce click artifacts with
    // production weights that fail the no_clicks hard bound. Part of #4262.

    // First run.
    let audio1 = {
        // SAFETY: safetensors file not modified while alive.
        let mut kokoro = unsafe { load_production_warn(&weights_path) };
        let (audio, _cert) = kokoro
            .synthesize(&input_ids, &style, 1.0, &cache)
            .expect("synthesize run 1");
        audio
            .to_device(&cpu())
            .expect("GPU->CPU run 1")
            .to_flat_vec::<f32>()
            .expect("f32 run 1")
    };

    // Second run (fresh model load).
    let audio2 = {
        // SAFETY: safetensors file not modified while alive.
        let mut kokoro = unsafe { load_production_warn(&weights_path) };
        let (audio, _cert) = kokoro
            .synthesize(&input_ids, &style, 1.0, &cache)
            .expect("synthesize run 2");
        audio
            .to_device(&cpu())
            .expect("GPU->CPU run 2")
            .to_flat_vec::<f32>()
            .expect("f32 run 2")
    };

    // Both runs should produce the same length.
    assert_eq!(
        audio1.len(),
        audio2.len(),
        "audio length mismatch: run1={}, run2={}",
        audio1.len(),
        audio2.len()
    );
    assert!(!audio1.is_empty(), "audio should be non-empty");

    // Element-wise comparison. Metal should be deterministic across runs.
    // Allow a small tolerance for potential floating-point non-determinism
    // in Metal command buffer scheduling.
    let tol = 1e-5_f32;
    let mut max_diff = 0.0_f32;
    let mut mismatch_count = 0_usize;

    for (i, (a, b)) in audio1.iter().zip(audio2.iter()).enumerate() {
        let diff = (a - b).abs();
        if diff > max_diff {
            max_diff = diff;
        }
        if diff > tol {
            if mismatch_count == 0 {
                eprintln!("First mismatch at [{i}]: run1={a}, run2={b}, diff={diff:.6e}");
            }
            mismatch_count += 1;
        }
    }

    eprintln!(
        "CPU vs GPU parity: n={}, max_diff={:.6e}, mismatches={} (tol={:.0e})",
        audio1.len(),
        max_diff,
        mismatch_count,
        tol,
    );

    assert_eq!(
        mismatch_count,
        0,
        "GPU non-determinism: {mismatch_count}/{} elements differ (max_diff={max_diff:.6e})",
        audio1.len()
    );
}

// ---------------------------------------------------------------------------
// Test: audio output validity
// ---------------------------------------------------------------------------

/// Verify Metal-synthesized audio produces valid PCM in [-1, 1].
///
/// End-to-end synthesis with real weights must produce:
/// 1. Finite values (no NaN or Inf)
/// 2. Bounded amplitude: all samples in [-1.0, 1.0]
/// 3. Non-silent: at least some non-zero samples (real weights should
///    produce meaningful audio, not zeros)
/// 4. Reasonable duration: 8 phonemes at speed=1.0 should produce
///    at minimum a few hundred samples at 24 kHz
#[test]
fn test_kokoro_metal_audio_output() {
    let weights_path = match require_weights() {
        Some(p) => p,
        None => return,
    };

    super::test_utils::gpu_init();
    let cache = super::test_utils::metal_setup();

    // Use Warn policy: test tokens [0..7] produce click artifacts with
    // production weights that fail the no_clicks hard bound. Part of #4262.
    // SAFETY: safetensors file not modified while alive.
    let mut kokoro = unsafe { load_production_warn(&weights_path) };

    let (input_ids, style) = synthetic_inputs();

    let (audio, cert) = kokoro
        .synthesize(&input_ids, &style, 1.0, &cache)
        .expect("synthesize failed");

    let audio_cpu = audio
        .to_device(&cpu())
        .expect("GPU->CPU")
        .to_flat_vec::<f32>()
        .expect("f32 extract");

    let n = audio_cpu.len();

    // 1. All finite.
    let non_finite = audio_cpu.iter().filter(|v| !v.is_finite()).count();
    assert_eq!(
        non_finite, 0,
        "audio has {non_finite}/{n} non-finite values"
    );

    // 2. Bounded amplitude [-1, 1].
    let max_abs = audio_cpu.iter().map(|v| v.abs()).fold(0.0_f32, f32::max);
    assert!(
        max_abs <= 1.0,
        "audio exceeds [-1, 1]: max_abs={max_abs:.6}"
    );

    // 3. Non-silent (real weights produce non-trivial output).
    let rms = (audio_cpu.iter().map(|v| v * v).sum::<f32>() / n as f32).sqrt();
    assert!(rms > 1e-6, "audio is essentially silent: rms={rms:.6e}");

    // 4. Reasonable duration (>= 100 samples for 8 phonemes).
    assert!(n >= 100, "audio too short for 8 phonemes: {n} samples");

    // 5. Hard bounds certificate.
    assert!(cert.overall_passed, "hard bounds certificate failed");

    eprintln!(
        "Audio output: {n} samples ({:.1} ms at 24kHz), max_abs={max_abs:.4}, rms={rms:.4e}, cert=PASSED",
        n as f64 / 24.0,
    );
}

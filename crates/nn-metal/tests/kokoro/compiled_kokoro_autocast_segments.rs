// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Per-segment F16 autocast verification for all Kokoro segments.
//!
//! Confirms that F16 autocast produces finite output for each pipeline stage:
//! PlBert, TextEncoder, ProsodyPredictor, F0EnergyPredictor, and Generator.
//! All segments receive the same autocast policy via `compile_with_shared()`.
//!
//! Also includes an SNR comparison between autocast and F32 baseline to verify
//! audio quality preservation (target: SNR > 35 dB).
//!
//! Part of #4269 (F16 autocast expansion beyond generator).

use nn_core::dyn_tensor::DynTensor;
use nn_core::{DType, Device};

use super::kokoro_test_weights::{build_kokoro_mini, mini_test_config};

fn cpu() -> Device {
    Device::Cpu
}

/// Helper: assert all values in a tensor are finite (no NaN/Inf).
fn assert_finite(t: &DynTensor, label: &str) {
    let vals = t
        .to_device(&cpu())
        .unwrap_or_else(|e| panic!("{label}: to_device(cpu) failed: {e}"))
        .to_flat_vec::<f32>()
        .unwrap_or_else(|e| panic!("{label}: to_flat_vec failed: {e}"));
    let non_finite = vals.iter().filter(|v| !v.is_finite()).count();
    assert_eq!(
        non_finite,
        0,
        "{label}: expected all finite, found {non_finite} non-finite out of {}",
        vals.len()
    );
}

/// Helper: compute signal-to-noise ratio in dB between reference and test.
///
/// SNR = 10 * log10(sum(ref^2) / sum((ref - test)^2))
fn snr_db(reference: &[f32], test: &[f32]) -> f64 {
    assert_eq!(reference.len(), test.len(), "SNR: length mismatch");
    let signal_power: f64 = reference.iter().map(|&x| f64::from(x) * f64::from(x)).sum();
    let noise_power: f64 = reference
        .iter()
        .zip(test.iter())
        .map(|(&r, &t)| {
            let d = f64::from(r) - f64::from(t);
            d * d
        })
        .sum();
    if noise_power < 1e-30 {
        return 120.0; // effectively identical
    }
    10.0 * (signal_power / noise_power).log10()
}

// -- Per-segment autocast tests (synthetic weights) ---------------------------

/// Segment 0 (PlBert + bert_encoder): autocast produces finite bert_features.
///
/// PlBert is a 12-layer ALBERT transformer with embedding + attention.
/// Compute-dominant ops (embedding, linear, attention) use F16;
/// accumulate ops (layer_norm, softmax) stay F32.
#[test]
fn test_autocast_seg_plbert_finite() {
    let (kokoro, cache) = build_kokoro_mini();
    let mut kokoro = kokoro.with_autocast();

    let input_ids = DynTensor::from_vec(vec![1.0, 2.0, 3.0], &[1, 3], &cpu()).unwrap();

    let encode = kokoro
        .step_encode(&input_ids, &cache)
        .expect("step_encode with autocast failed");

    assert_finite(&encode.bert_features, "autocast plbert bert_features");
    eprintln!(
        "PlBert autocast: bert_features shape {:?}, all finite",
        encode.bert_features.dims()
    );
}

/// Segment 1 (TextEncoder): autocast produces finite text_features.
///
/// TextEncoder uses its own Embedding + Linear + attention layers.
/// All compute-dominant ops benefit from F16 autocast.
#[test]
fn test_autocast_seg_text_encoder_finite() {
    let (kokoro, cache) = build_kokoro_mini();
    let mut kokoro = kokoro.with_autocast();

    let input_ids = DynTensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[1, 4], &cpu()).unwrap();

    let encode = kokoro
        .step_encode(&input_ids, &cache)
        .expect("step_encode for text encoder check");

    assert_finite(&encode.text_features, "autocast text_encoder text_features");
    eprintln!(
        "TextEncoder autocast: text_features shape {:?}, all finite",
        encode.text_features.dims()
    );
}

/// Segment 2 (ProsodyPredictor): autocast produces finite dur_logits and features.
///
/// ProsodyPredictor contains Linear + LSTM layers. LSTM stays F32
/// (sigmoid/tanh saturation at F16 range, per compiled_model_builder.rs D6).
/// Linear layers get F16 autocast. The key test: LSTM accumulation
/// correctness is preserved alongside F16 linear layers.
#[test]
fn test_autocast_seg_prosody_finite() {
    let (kokoro, cache) = build_kokoro_mini();
    let mut kokoro = kokoro.with_autocast();
    let cfg = mini_test_config();

    let input_ids = DynTensor::from_vec(vec![1.0, 2.0, 3.0], &[1, 3], &cpu()).unwrap();
    let style = DynTensor::new(
        &super::test_utils::rand_f32_vec(200, 2 * cfg.style_dim, -0.1, 0.1),
        &[1, 2 * cfg.style_dim],
        &cpu(),
    )
    .unwrap();

    let encode = kokoro
        .step_encode(&input_ids, &cache)
        .expect("step_encode for prosody");

    let styles = kokoro.split_style(&style).expect("split_style");
    let prosody = kokoro
        .step_predict_prosody(
            &encode.bert_features,
            &styles.prosody_style,
            encode.seq_len,
            &cache,
        )
        .expect("step_predict_prosody with autocast");

    assert_finite(&prosody.dur_logits, "autocast prosody dur_logits");
    assert_finite(&prosody.features, "autocast prosody features");
    eprintln!(
        "ProsodyPredictor autocast: dur_logits shape {:?}, features shape {:?}, all finite (LSTM stays F32)",
        prosody.dur_logits.dims(),
        prosody.features.dims()
    );
}

/// Segment 3 (F0EnergyPredictor): autocast produces finite f0 and energy.
///
/// F0EnergyPredictor has similar structure to ProsodyPredictor (Linear + LSTM).
/// LSTM stays F32 automatically; Linear layers get F16.
/// Tested via full pipeline because step_predict_f0_energy requires
/// outputs from regulate (which needs prosody + length_regulate).
#[test]
fn test_autocast_seg_f0_energy_finite() {
    let (kokoro, cache) = build_kokoro_mini();
    let mut kokoro = kokoro.with_autocast();
    let cfg = mini_test_config();

    let input_ids = DynTensor::from_vec(vec![1.0, 2.0, 3.0], &[1, 3], &cpu()).unwrap();
    let style = DynTensor::new(
        &super::test_utils::rand_f32_vec(300, 2 * cfg.style_dim, -0.1, 0.1),
        &[1, 2 * cfg.style_dim],
        &cpu(),
    )
    .unwrap();

    // Full pipeline exercises all segments including F0Energy.
    let (audio, _cert) = kokoro
        .synthesize(&input_ids, &style, 1.0, &cache)
        .expect("autocast synthesize for F0Energy path");

    assert_finite(&audio, "autocast full pipeline audio (F0Energy included)");
    eprintln!(
        "F0EnergyPredictor autocast: pipeline audio shape {:?}, all finite (LSTM stays F32)",
        audio.dims()
    );
}

/// Generator (Segment 4): autocast produces finite audio.
///
/// Generator has ~35 FusedResBlocks with Conv1d + ConvTranspose1d layers.
/// These are the heaviest compute ops and benefit most from F16 autocast.
/// This was the original F16 autocast target (#3766).
#[test]
fn test_autocast_seg_generator_finite() {
    let (kokoro, cache) = build_kokoro_mini();
    let mut kokoro = kokoro.with_autocast();
    let cfg = mini_test_config();

    let input_ids = DynTensor::from_vec(vec![1.0, 2.0], &[1, 2], &cpu()).unwrap();
    let style = DynTensor::new(
        &super::test_utils::rand_f32_vec(400, 2 * cfg.style_dim, -0.1, 0.1),
        &[1, 2 * cfg.style_dim],
        &cpu(),
    )
    .unwrap();

    let (audio, _cert) = kokoro
        .synthesize(&input_ids, &style, 1.0, &cache)
        .expect("autocast synthesize for Generator path");

    assert_finite(&audio, "autocast generator audio");
    eprintln!(
        "Generator autocast: audio shape {:?}, all finite",
        audio.dims()
    );
}

// -- SNR comparison tests -----------------------------------------------------

/// Full pipeline SNR: autocast vs F32 baseline with synthetic weights.
///
/// Autocast should preserve audio quality. With synthetic (zero) weights,
/// both paths produce near-silent audio. The test confirms no catastrophic
/// divergence between autocast and F32 output.
#[test]
fn test_autocast_vs_f32_snr_synthetic() {
    let cfg = mini_test_config();
    let input_ids = DynTensor::from_vec(vec![1.0, 2.0, 3.0], &[1, 3], &cpu()).unwrap();
    let style = DynTensor::new(
        &super::test_utils::rand_f32_vec(400, 2 * cfg.style_dim, -0.1, 0.1),
        &[1, 2 * cfg.style_dim],
        &cpu(),
    )
    .unwrap();

    // F32 baseline.
    let (kokoro_f32, cache) = build_kokoro_mini();
    let mut kokoro_f32 = kokoro_f32;
    let (audio_f32, _) = kokoro_f32
        .synthesize(&input_ids, &style, 1.0, &cache)
        .expect("F32 synthesize");

    // Autocast.
    let (kokoro_ac, _) = build_kokoro_mini();
    let mut kokoro_ac = kokoro_ac.with_autocast();
    let (audio_ac, _) = kokoro_ac
        .synthesize(&input_ids, &style, 1.0, &cache)
        .expect("autocast synthesize");

    // Both must produce same shape.
    assert_eq!(
        audio_f32.dims(),
        audio_ac.dims(),
        "autocast vs F32 shape mismatch"
    );

    let samples_f32 = audio_f32.to_flat_vec::<f32>().expect("f32 samples");
    let samples_ac = audio_ac.to_flat_vec::<f32>().expect("ac samples");

    // Check max absolute difference.
    let max_diff: f32 = samples_f32
        .iter()
        .zip(samples_ac.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f32, f32::max);

    eprintln!(
        "Autocast vs F32 synthetic: {} samples, max_diff={max_diff:.6e}",
        samples_f32.len()
    );

    // No catastrophic divergence (no NaN, no huge diffs).
    assert!(
        max_diff < 1.0,
        "autocast vs F32 max diff {max_diff} exceeds 1.0 -- catastrophic divergence"
    );
}

/// Production-weight SNR comparison: autocast vs F32.
///
/// Requires `KOKORO_WEIGHTS` env var. Measures SNR in dB between F32 baseline
/// and autocast output. Target: SNR > 35 dB (acceptance criteria from #4269).
#[test]
fn test_autocast_vs_f32_snr_production() {
    use nn_metal::compiled_kokoro::CompiledKokoro;

    let weights_path = match super::kokoro_test_env::require_kokoro_weights(
        "production SNR comparison not enforced.",
    ) {
        Some(path) => path,
        None => return,
    };

    super::test_utils::gpu_init();
    let cache = super::test_utils::metal_setup();

    let input_ids =
        DynTensor::from_vec_i64(vec![0_i64, 1, 2, 3, 4, 5, 6, 7], &[1, 8], &cpu()).unwrap();
    let style = DynTensor::full(&[1, 256], 0.01, DType::F32, &cpu()).unwrap();

    // Use Warn policy: test tokens [0..7] produce click artifacts with
    // production weights that fail the no_clicks hard bound. Part of #4262.
    let mut hb_f32 = nn_tts_verify::HardBoundsConfig::default();
    hb_f32.rejection_policy = nn_tts_verify::RejectionPolicy::Warn;
    let hb_ac = hb_f32.clone();

    // F32 baseline.
    // SAFETY: safetensors file not modified while alive.
    let mut f32_kokoro =
        unsafe { CompiledKokoro::load_with_hard_bounds(&weights_path, hb_f32).expect("load F32") };
    let (audio_f32, _) = f32_kokoro
        .synthesize(&input_ids, &style, 1.0, &cache)
        .expect("F32 synthesize");

    // Autocast.
    let mut ac_kokoro = unsafe {
        CompiledKokoro::load_with_hard_bounds(&weights_path, hb_ac).expect("load autocast")
    }
    .with_autocast();
    let (audio_ac, _) = ac_kokoro
        .synthesize(&input_ids, &style, 1.0, &cache)
        .expect("autocast synthesize");

    assert_eq!(
        audio_f32.dims(),
        audio_ac.dims(),
        "production autocast vs F32 shape mismatch"
    );

    let samples_f32 = audio_f32.to_flat_vec::<f32>().expect("f32 samples");
    let samples_ac = audio_ac.to_flat_vec::<f32>().expect("ac samples");

    let snr = snr_db(&samples_f32, &samples_ac);
    let max_diff: f32 = samples_f32
        .iter()
        .zip(samples_ac.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f32, f32::max);

    eprintln!(
        "Production autocast vs F32: {} samples, SNR={snr:.1} dB, max_diff={max_diff:.6e}",
        samples_f32.len()
    );

    // Acceptance criteria from #4269: SNR > 35 dB.
    assert!(
        snr > 35.0,
        "production autocast SNR {snr:.1} dB < 35.0 dB threshold"
    );
}

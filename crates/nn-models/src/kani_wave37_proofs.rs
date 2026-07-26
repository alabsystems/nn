// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Kani proof harnesses for nn-models (Wave 37).
//!
//! Covers:
//! - Qwen3-VL generation config validation (temperature, top_p)
//! - STFT/iSTFT parameter consistency invariants
//! - Demucs architecture constant relationships
//! - Kokoro streaming types duration/config safety
//! - Kokoro config validation completeness
//! - Kokoro error validate_speed IEEE 754 coverage
//! - Kokoro signal constant consistency
//! - IstftBasis Hann window properties (bounded, symmetric)
//! - Demucs shared channels_at_depth monotonicity
//! - conv1d_output_len arithmetic invariants
//!
//! Part of #4298.

// ── Qwen3-VL generation config invariants ────────────────────────────────

/// Qwen3VLGenerationConfig::validate accepts valid temperature.
#[kani::proof]
fn prove_qwen3_gen_config_valid_temp_passes() {
    let temp: f64 = kani::any();
    kani::assume(temp >= 0.0 && temp <= 100.0 && temp.is_finite());
    let cfg = crate::qwen3_vl::generate::Qwen3VLGenerationConfig {
        max_new_tokens: 1,
        temperature: temp,
        top_p: None,
        eos_token_id: None,
    };
    assert!(cfg.validate().is_ok());
}

/// Qwen3VLGenerationConfig::validate rejects negative temperature.
#[kani::proof]
fn prove_qwen3_gen_config_negative_temp_rejected() {
    let temp: f64 = kani::any();
    kani::assume(temp < 0.0 && temp.is_finite());
    let cfg = crate::qwen3_vl::generate::Qwen3VLGenerationConfig {
        max_new_tokens: 1,
        temperature: temp,
        top_p: None,
        eos_token_id: None,
    };
    assert!(cfg.validate().is_err());
}

/// Qwen3VLGenerationConfig::validate rejects NaN temperature.
#[kani::proof]
fn prove_qwen3_gen_config_nan_temp_rejected() {
    let cfg = crate::qwen3_vl::generate::Qwen3VLGenerationConfig {
        max_new_tokens: 1,
        temperature: f64::NAN,
        top_p: None,
        eos_token_id: None,
    };
    assert!(cfg.validate().is_err());
}

/// Qwen3VLGenerationConfig::validate rejects infinite temperature.
#[kani::proof]
fn prove_qwen3_gen_config_inf_temp_rejected() {
    let cfg = crate::qwen3_vl::generate::Qwen3VLGenerationConfig {
        max_new_tokens: 1,
        temperature: f64::INFINITY,
        top_p: None,
        eos_token_id: None,
    };
    assert!(cfg.validate().is_err());
}

/// Qwen3VLGenerationConfig::validate rejects top_p = 0.
#[kani::proof]
fn prove_qwen3_gen_config_top_p_zero_rejected() {
    let cfg = crate::qwen3_vl::generate::Qwen3VLGenerationConfig {
        max_new_tokens: 1,
        temperature: 1.0,
        top_p: Some(0.0),
        eos_token_id: None,
    };
    assert!(cfg.validate().is_err());
}

/// Qwen3VLGenerationConfig::validate accepts top_p in (0, 1].
#[kani::proof]
fn prove_qwen3_gen_config_valid_top_p_passes() {
    let p: f64 = kani::any();
    kani::assume(p > 0.0 && p <= 1.0 && p.is_finite());
    let cfg = crate::qwen3_vl::generate::Qwen3VLGenerationConfig {
        max_new_tokens: 1,
        temperature: 0.0,
        top_p: Some(p),
        eos_token_id: None,
    };
    assert!(cfg.validate().is_ok());
}

/// Qwen3VLGenerationConfig::validate rejects top_p > 1.
#[kani::proof]
fn prove_qwen3_gen_config_top_p_gt1_rejected() {
    let p: f64 = kani::any();
    kani::assume(p > 1.0 && p.is_finite());
    let cfg = crate::qwen3_vl::generate::Qwen3VLGenerationConfig {
        max_new_tokens: 1,
        temperature: 0.0,
        top_p: Some(p),
        eos_token_id: None,
    };
    assert!(cfg.validate().is_err());
}

// ── STFT / iSTFT parameter invariants ─────────────────────────────────

/// StftParams::new produces consistent n_freqs = n_fft / 2 + 1.
#[kani::proof]
fn prove_stft_params_n_freqs_consistency() {
    let n_fft: usize = kani::any();
    kani::assume(n_fft >= 2 && n_fft <= 8192 && n_fft % 2 == 0);
    let hop: usize = kani::any();
    kani::assume(hop >= 1 && hop <= n_fft);
    let params = crate::stft::StftParams::new(n_fft, hop);
    assert_eq!(params.n_freqs, n_fft / 2 + 1);
    assert_eq!(params.pad_right, n_fft / 4);
}

/// StftParams::default produces self-consistent values.
#[kani::proof]
fn prove_stft_params_default_consistent() {
    let d = crate::stft::StftParams::default();
    assert_eq!(d.n_fft, 256);
    assert_eq!(d.hop_length, 128);
    assert_eq!(d.n_freqs, d.n_fft / 2 + 1);
    assert_eq!(d.pad_right, d.n_fft / 4);
}

/// IstftParams::new rejects zero n_fft.
#[kani::proof]
fn prove_istft_params_zero_nfft_rejected() {
    let result = crate::istft::IstftParams::new(0, 1, false, false);
    assert!(result.is_err());
}

/// IstftParams::new rejects odd n_fft.
#[kani::proof]
fn prove_istft_params_odd_nfft_rejected() {
    let n: usize = kani::any();
    kani::assume(n >= 1 && n <= 4096 && n % 2 == 1);
    let result = crate::istft::IstftParams::new(n, 1, false, false);
    assert!(result.is_err());
}

/// IstftParams::new rejects zero hop_length.
#[kani::proof]
fn prove_istft_params_zero_hop_rejected() {
    let n: usize = kani::any();
    kani::assume(n >= 2 && n <= 4096 && n % 2 == 0);
    let result = crate::istft::IstftParams::new(n, 0, false, false);
    assert!(result.is_err());
}

/// IstftParams::new accepts valid even n_fft with nonzero hop.
#[kani::proof]
fn prove_istft_params_valid_accepted() {
    let n: usize = kani::any();
    kani::assume(n >= 2 && n <= 4096 && n % 2 == 0);
    let hop: usize = kani::any();
    kani::assume(hop >= 1 && hop <= n);
    let result = crate::istft::IstftParams::new(n, hop, false, false);
    assert!(result.is_ok());
    let p = result.unwrap();
    assert_eq!(p.n_fft, n);
    assert_eq!(p.hop_length, hop);
}

/// IstftParams::default produces valid parameters.
#[kani::proof]
fn prove_istft_params_default_valid() {
    let d = crate::istft::IstftParams::default();
    assert_eq!(d.n_fft, 4096);
    assert_eq!(d.hop_length, 1024);
    assert!(d.n_fft % 2 == 0);
    assert!(d.hop_length > 0);
    assert!(d.normalized);
    assert!(d.center);
}

// ── Demucs architecture constant relationships ───────────────────────

/// Demucs transformer constants are internally consistent.
#[kani::proof]
fn prove_demucs_transformer_constants_consistent() {
    use crate::demucs_transformer_constants::*;
    assert_eq!(
        FFN_HIDDEN_DIM,
        (TRANSFORMER_DIM as f64 * FFN_HIDDEN_SCALE) as usize
    );
    assert!(TRANSFORMER_DIM > 0);
    assert!(BOTTLENECK_DIM > 0);
    assert!(NUM_HEADS > 0);
    assert!(NUM_LAYERS > 0);
    // Transformer dim must be divisible by number of heads for multi-head attention.
    assert_eq!(TRANSFORMER_DIM % NUM_HEADS, 0);
}

/// Demucs shared constants: spectral input/output channel relationships.
#[kani::proof]
fn prove_demucs_spectral_channel_relationships() {
    use crate::demucs_shared::*;
    // Spectral input: stereo (2) x complex (2) = 4
    assert_eq!(SPECTRAL_INPUT_CHANNELS, AUDIO_CHANNELS * 2);
    // Spectral output: 4 sources x stereo (2) x complex (2) = 16
    assert_eq!(SPECTRAL_OUTPUT_CHANNELS, 4 * AUDIO_CHANNELS * 2);
    // Freq embedding dim matches channels_at_depth(0)
    assert_eq!(SPECTRAL_FREQ_EMB_DIM, channels_at_depth(0));
    // Padding = kernel / 4
    assert_eq!(SPECTRAL_CONV_PADDING, SPECTRAL_KERNEL_SIZE / 4);
    assert_eq!(TEMPORAL_CONV_PADDING, TEMPORAL_KERNEL_SIZE / 4);
}

/// channels_at_depth is monotonically non-decreasing (GROWTH >= 1).
#[kani::proof]
fn prove_channels_at_depth_monotonic() {
    let d: usize = kani::any();
    kani::assume(d <= 10);
    let c0 = crate::demucs_shared::channels_at_depth(d);
    let c1 = crate::demucs_shared::channels_at_depth(d + 1);
    // With GROWTH = 2.0 >= 1.0, channels are non-decreasing.
    assert!(c1 >= c0);
}

/// channels_at_depth(0) = BASE_CHANNELS.
#[kani::proof]
fn prove_channels_at_depth_zero_is_base() {
    assert_eq!(
        crate::demucs_shared::channels_at_depth(0),
        crate::demucs_shared::BASE_CHANNELS,
    );
}

/// channels_at_depth produces power-of-two multiples of BASE_CHANNELS.
#[kani::proof]
fn prove_channels_at_depth_power_of_two() {
    let d: usize = kani::any();
    kani::assume(d <= 10);
    let c = crate::demucs_shared::channels_at_depth(d);
    let base = crate::demucs_shared::BASE_CHANNELS;
    // With GROWTH = 2.0: channels_at_depth(d) = 48 * 2^d
    assert_eq!(c, base * (1 << d));
}

// ── Kokoro streaming types invariants ─────────────────────────────────

/// AudioChunk::is_empty <=> len == 0.
#[kani::proof]
fn prove_audio_chunk_empty_len_consistent() {
    let len: usize = kani::any();
    kani::assume(len <= 16);
    let pcm = vec![0.0f32; len];
    let chunk = crate::kokoro_streaming::AudioChunk::new(pcm, 1, 0, 0, 1, true);
    assert_eq!(chunk.is_empty(), chunk.len() == 0);
}

/// AudioChunk::duration_secs is non-negative.
#[kani::proof]
fn prove_audio_chunk_duration_non_negative() {
    let len: usize = kani::any();
    kani::assume(len <= 1000);
    let pcm = vec![0.0f32; len];
    let chunk = crate::kokoro_streaming::AudioChunk::new(pcm, 1, 0, 0, 1, true);
    assert!(chunk.duration_secs() >= 0.0);
}

/// AudioChunk::duration_secs for stereo divides by 2.
#[kani::proof]
fn prove_audio_chunk_stereo_duration() {
    let pcm = vec![0.0f32; 48000]; // 48000 interleaved samples
    let mono = crate::kokoro_streaming::AudioChunk::new(pcm.clone(), 1, 0, 0, 1, true);
    let stereo = crate::kokoro_streaming::AudioChunk::new(pcm, 2, 0, 0, 1, true);
    // Stereo duration should be half of mono (same sample count, double channels).
    let mono_dur = mono.duration_secs();
    let stereo_dur = stereo.duration_secs();
    assert!((stereo_dur * 2.0 - mono_dur).abs() < 1e-10);
}

/// KokoroStreamConfig::new rejects zero crossfade.
#[kani::proof]
fn prove_stream_config_zero_cf_rejected() {
    let result = crate::kokoro_streaming::KokoroStreamConfig::new(0);
    assert!(result.is_err());
}

/// KokoroStreamConfig::new accepts positive crossfade.
#[kani::proof]
fn prove_stream_config_positive_cf_accepted() {
    let cf: usize = kani::any();
    kani::assume(cf >= 1 && cf <= 10000);
    let result = crate::kokoro_streaming::KokoroStreamConfig::new(cf);
    assert!(result.is_ok());
}

/// KokoroStreamConfig::default passes validation.
#[kani::proof]
fn prove_stream_config_default_valid() {
    let cfg = crate::kokoro_streaming::KokoroStreamConfig::default();
    assert!(cfg.validate().is_ok());
    assert!(cfg.crossfade_samples > 0);
    assert_eq!(cfg.crossfade_samples, 960); // 40ms at 24kHz
}

/// crossfade_duration_secs is positive for valid configs.
#[kani::proof]
fn prove_crossfade_duration_positive() {
    let cf: usize = kani::any();
    kani::assume(cf >= 1 && cf <= 10000);
    let cfg = crate::kokoro_streaming::KokoroStreamConfig::new(cf).unwrap();
    let dur = cfg.crossfade_duration_secs();
    assert!(dur > 0.0);
    assert!(dur.is_finite());
}

/// concatenate_chunks on empty input returns empty vec.
#[kani::proof]
fn prove_concatenate_chunks_empty() {
    let result = crate::kokoro_streaming::concatenate_chunks(&[]);
    assert!(result.is_empty());
}

// ── Kokoro config validation invariants ───────────────────────────────

/// KokoroConfig::default() passes validation.
#[kani::proof]
fn prove_kokoro_config_default_valid() {
    let cfg = crate::kokoro_tts::KokoroConfig::default();
    assert!(cfg.validate().is_ok());
}

/// KokoroConfig::validate rejects d_en = 0.
#[kani::proof]
fn prove_kokoro_config_zero_den_rejected() {
    let mut cfg = crate::kokoro_tts::KokoroConfig::default();
    cfg.d_en = 0;
    assert!(cfg.validate().is_err());
}

/// KokoroConfig::validate rejects style_dim = 0.
#[kani::proof]
fn prove_kokoro_config_zero_style_dim_rejected() {
    let mut cfg = crate::kokoro_tts::KokoroConfig::default();
    cfg.style_dim = 0;
    assert!(cfg.validate().is_err());
}

/// KokoroConfig::validate rejects max_dur = 0.
#[kani::proof]
fn prove_kokoro_config_zero_max_dur_rejected() {
    let mut cfg = crate::kokoro_tts::KokoroConfig::default();
    cfg.max_dur = 0;
    assert!(cfg.validate().is_err());
}

/// KokoroConfig::validate rejects n_fft not divisible by 4.
#[kani::proof]
fn prove_kokoro_config_nfft_not_div4_rejected() {
    let mut cfg = crate::kokoro_tts::KokoroConfig::default();
    let nfft: usize = kani::any();
    kani::assume(nfft > 0 && nfft <= 256 && nfft % 4 != 0);
    cfg.n_fft = nfft;
    assert!(cfg.validate().is_err());
}

/// KokoroConfig::validate rejects n_fft = 0.
#[kani::proof]
fn prove_kokoro_config_nfft_zero_rejected() {
    let mut cfg = crate::kokoro_tts::KokoroConfig::default();
    cfg.n_fft = 0;
    assert!(cfg.validate().is_err());
}

/// KokoroConfig::validate rejects empty upsample_rates.
#[kani::proof]
fn prove_kokoro_config_empty_upsample_rejected() {
    let mut cfg = crate::kokoro_tts::KokoroConfig::default();
    cfg.upsample_rates = vec![];
    assert!(cfg.validate().is_err());
}

// ── Kokoro error validate_speed IEEE 754 ──────────────────────────────

/// validate_speed accepts positive finite speeds.
#[kani::proof]
fn prove_validate_speed_positive_finite_ok() {
    let speed: f32 = kani::any();
    kani::assume(speed > 0.0 && speed.is_finite());
    assert!(crate::kokoro_error::validate_speed(speed).is_ok());
}

/// validate_speed rejects zero.
#[kani::proof]
fn prove_validate_speed_zero_rejected() {
    assert!(crate::kokoro_error::validate_speed(0.0).is_err());
}

/// validate_speed rejects negative.
#[kani::proof]
fn prove_validate_speed_negative_rejected() {
    let speed: f32 = kani::any();
    kani::assume(speed < 0.0 && speed.is_finite());
    assert!(crate::kokoro_error::validate_speed(speed).is_err());
}

/// validate_speed rejects NaN.
#[kani::proof]
fn prove_validate_speed_nan_rejected() {
    assert!(crate::kokoro_error::validate_speed(f32::NAN).is_err());
}

/// validate_speed rejects positive infinity.
#[kani::proof]
fn prove_validate_speed_pos_inf_rejected() {
    assert!(crate::kokoro_error::validate_speed(f32::INFINITY).is_err());
}

/// validate_speed rejects negative infinity.
#[kani::proof]
fn prove_validate_speed_neg_inf_rejected() {
    assert!(crate::kokoro_error::validate_speed(f32::NEG_INFINITY).is_err());
}

// ── Kokoro signal constants ───────────────────────────────────────────

/// Kokoro signal constants are self-consistent.
#[kani::proof]
fn prove_kokoro_signal_constants_consistent() {
    use crate::kokoro_tts::{KOKORO_HOP_LENGTH, KOKORO_N_BINS, KOKORO_N_FFT, KOKORO_SAMPLE_RATE};
    assert_eq!(KOKORO_N_BINS, KOKORO_N_FFT / 2 + 1);
    assert!(KOKORO_HOP_LENGTH > 0);
    assert!(KOKORO_N_FFT > 0);
    assert!(KOKORO_SAMPLE_RATE > 0);
    // hop divides n_fft evenly for Kokoro parameters
    assert_eq!(KOKORO_N_FFT % KOKORO_HOP_LENGTH, 0);
}

// ── IstftBasis construction invariants ────────────────────────────────

/// IstftBasis::new rejects zero n_fft.
#[kani::proof]
fn prove_istft_basis_zero_nfft_rejected() {
    let params = crate::istft::IstftParams {
        n_fft: 0,
        hop_length: 1,
        normalized: false,
        center: false,
    };
    assert!(crate::istft::IstftBasis::new(params).is_err());
}

/// IstftBasis::new rejects odd n_fft.
#[kani::proof]
fn prove_istft_basis_odd_nfft_rejected() {
    let params = crate::istft::IstftParams {
        n_fft: 3,
        hop_length: 1,
        normalized: false,
        center: false,
    };
    assert!(crate::istft::IstftBasis::new(params).is_err());
}

/// IstftBasis::new with valid small params: n_bins is consistent.
#[kani::proof]
fn prove_istft_basis_n_bins_consistent() {
    let n: usize = kani::any();
    kani::assume(n >= 2 && n <= 32 && n % 2 == 0);
    let params = crate::istft::IstftParams {
        n_fft: n,
        hop_length: 1,
        normalized: false,
        center: false,
    };
    let basis = crate::istft::IstftBasis::new(params).unwrap();
    assert_eq!(basis.n_bins(), n / 2 + 1);
    // Window length = n_fft
    assert_eq!(basis.window().len(), n);
    // Cos/sin basis length = n_bins * n_fft
    let n_bins = n / 2 + 1;
    assert_eq!(basis.cos_basis().len(), n_bins * n);
    assert_eq!(basis.sin_basis().len(), n_bins * n);
}

/// Hann window values are in [0, 1] for small n_fft.
#[kani::proof]
fn prove_istft_hann_window_bounded() {
    let n: usize = kani::any();
    kani::assume(n >= 2 && n <= 32 && n % 2 == 0);
    let params = crate::istft::IstftParams {
        n_fft: n,
        hop_length: 1,
        normalized: false,
        center: false,
    };
    let basis = crate::istft::IstftBasis::new(params).unwrap();
    for &w in basis.window() {
        assert!(w >= 0.0);
        assert!(w <= 1.0);
        assert!(w.is_finite());
    }
}

/// Hann window is symmetric: w[k] == w[N-k] for periodic window.
#[kani::proof]
fn prove_istft_hann_window_symmetric() {
    let n: usize = kani::any();
    kani::assume(n >= 4 && n <= 32 && n % 2 == 0);
    let params = crate::istft::IstftParams {
        n_fft: n,
        hop_length: 1,
        normalized: false,
        center: false,
    };
    let basis = crate::istft::IstftBasis::new(params).unwrap();
    let window = basis.window();
    for k in 1..n {
        let mirror = n - k;
        let diff = (window[k] - window[mirror]).abs();
        assert!(diff < 1e-5, "Hann symmetry violated at k={}", k);
    }
}

// ── Demucs shared: conv1d_output_len boundary cases ──────────────────

/// conv1d_output_len with stride=1, padding=0 produces input - kernel + 1.
#[kani::proof]
fn prove_conv1d_output_len_no_padding() {
    let input: usize = kani::any();
    let kernel: usize = kani::any();
    kani::assume(input >= 1 && input <= 256);
    kani::assume(kernel >= 1 && kernel <= input);
    let result = crate::demucs_shared::conv1d_output_len(input, kernel, 1, 0);
    if let Ok(out) = result {
        assert_eq!(out, input - kernel + 1);
    }
}

/// conv1d_output_len with full padding preserves length (stride=1).
#[kani::proof]
fn prove_conv1d_output_len_same_padding() {
    let input: usize = kani::any();
    kani::assume(input >= 1 && input <= 256);
    let kernel: usize = kani::any();
    kani::assume(kernel >= 1 && kernel <= 64 && kernel % 2 == 1);
    let padding = (kernel - 1) / 2;
    let result = crate::demucs_shared::conv1d_output_len(input, kernel, 1, padding);
    if let Ok(out) = result {
        assert_eq!(out, input);
    }
}

// ── Kokoro error LOG_MAG_CLAMP_MAX safety ─────────────────────────────

/// LOG_MAG_CLAMP_MAX is safe: exp(LOG_MAG_CLAMP_MAX) does not overflow f32.
#[kani::proof]
fn prove_log_mag_clamp_max_safe() {
    let val = crate::kokoro_error::LOG_MAG_CLAMP_MAX;
    let result = (val as f32).exp();
    assert!(result.is_finite());
    assert!(result > 0.0);
}

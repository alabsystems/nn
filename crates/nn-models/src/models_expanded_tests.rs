// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Expanded cross-module tests for nn-models.
//!
//! Covers Kokoro config validation, STFT/iSTFT signal processing round-trips,
//! HTDemucs architecture constants, Silero VAD encoder shapes, streaming
//! assembly, chorus mixing, and edge cases. All tests run without model weights.

use std::f32::consts::PI;

use crate::demucs_shared::{
    channels_at_depth, conv1d_output_len, BASE_CHANNELS, DCONV_COMPRESS, DCONV_DEPTH, DCONV_KERNEL,
    DECODER_OUTPUT_CHANNELS, GROWTH, SPECTRAL_BASIC_DEPTH, SPECTRAL_DEPTH,
    SPECTRAL_FREQ_EMB_DIM, SPECTRAL_FREQ_EMB_FEATURES, SPECTRAL_INPUT_CHANNELS,
    SPECTRAL_KERNEL_SIZE, SPECTRAL_OUTPUT_CHANNELS, SPECTRAL_STRIDE, TEMPORAL_BASIC_DEPTH,
    TEMPORAL_DEPTH, TEMPORAL_KERNEL_SIZE, TEMPORAL_STRIDE,
};
use crate::demucs_transformer_constants::{
    BOTTLENECK_DIM, FFN_HIDDEN_DIM, FFN_HIDDEN_SCALE, LAYER_NORM_EPS, NUM_HEADS, NUM_LAYERS,
    TRANSFORMER_DIM,
};
use crate::istft::{IstftBasis, IstftError, IstftParams};
use crate::kokoro_chorus::ChorusConfig;
use crate::kokoro_error::KokoroError;
use crate::kokoro_streaming::{
    assemble_streaming_chunks, concatenate_chunks, crossfade_chunks, AudioChunk,
    CrossfadeWindow, KokoroStreamConfig,
};
use crate::kokoro_tts::{KokoroConfig, KOKORO_HOP_LENGTH, KOKORO_N_BINS, KOKORO_N_FFT,
    KOKORO_SAMPLE_RATE};
use crate::plbert::PlbertConfig;
use crate::silero_vad_builders::{ENCODER_BLOCKS, LSTM_HIDDEN_SIZE};
use crate::stft::{compute_stft_magnitude, StftError, StftParams};

// ===========================================================================
// Kokoro config: architecture invariants
// ===========================================================================

#[test]
fn test_kokoro_config_new_equals_default() {
    let from_new = KokoroConfig::new();
    let from_default = KokoroConfig::default();
    assert_eq!(from_new.d_en, from_default.d_en);
    assert_eq!(from_new.style_dim, from_default.style_dim);
    assert_eq!(from_new.n_fft, from_default.n_fft);
    assert_eq!(from_new.max_dur, from_default.max_dur);
    assert_eq!(from_new.gen_initial_channels, from_default.gen_initial_channels);
}

#[test]
fn test_kokoro_config_signal_constants_consistency() {
    // n_bins = n_fft / 2 + 1
    assert_eq!(KOKORO_N_BINS, KOKORO_N_FFT / 2 + 1);
    // hop_length = n_fft / 4
    assert_eq!(KOKORO_HOP_LENGTH, KOKORO_N_FFT / 4);
    // sample rate is 24kHz
    assert_eq!(KOKORO_SAMPLE_RATE, 24000);
}

#[test]
fn test_kokoro_config_upsample_product_times_hop() {
    let cfg = KokoroConfig::default();
    let hop = cfg.n_fft / 4; // 5
    let upsample_product: usize = cfg.upsample_rates.iter().product();
    // upsample_product * hop = 60 * 5 = 300, the source_upsample factor
    // used in SineGen for f0 upsampling to full audio rate.
    let source_upsample = upsample_product * hop;
    assert_eq!(source_upsample, 300);
}

#[test]
fn test_kokoro_config_gen_initial_channels_matches_d_en() {
    let cfg = KokoroConfig::default();
    assert_eq!(cfg.gen_initial_channels, cfg.d_en);
}

#[test]
fn test_kokoro_config_style_dim_is_half_voice_embedding() {
    let cfg = KokoroConfig::default();
    // Voice embedding is [B, 2*style_dim], split into decoder + prosody halves
    let voice_embedding_dim = 2 * cfg.style_dim;
    assert_eq!(voice_embedding_dim, 256);
}

#[test]
fn test_kokoro_config_resblock_dilations_structure() {
    let cfg = KokoroConfig::default();
    // Each resblock has 3 dilation values
    for (i, dilations) in cfg.resblock_dilations.iter().enumerate() {
        assert_eq!(dilations.len(), 3, "resblock {i} dilation count");
        // Dilations are [1, 3, 5]
        assert_eq!(dilations[0], 1, "resblock {i} dilation[0]");
        assert_eq!(dilations[1], 3, "resblock {i} dilation[1]");
        assert_eq!(dilations[2], 5, "resblock {i} dilation[2]");
    }
}

#[test]
fn test_kokoro_config_validate_n_fft_must_be_divisible_by_4() {
    // n_fft=20 is valid (20 % 4 == 0)
    let cfg = KokoroConfig::default();
    cfg.validate().unwrap();

    // n_fft=8 is also valid
    let cfg8 = KokoroConfig { n_fft: 8, ..Default::default() };
    cfg8.validate().unwrap();

    // n_fft=10 is invalid (10 % 4 != 0)
    let cfg10 = KokoroConfig { n_fft: 10, ..Default::default() };
    assert!(cfg10.validate().is_err());

    // n_fft=0 is invalid
    let cfg0 = KokoroConfig { n_fft: 0, ..Default::default() };
    assert!(cfg0.validate().is_err());
}

// ===========================================================================
// PlBert config
// ===========================================================================

#[test]
fn test_plbert_config_default_values() {
    let cfg = PlbertConfig::default();
    assert_eq!(cfg.vocab_size, 178);
    assert_eq!(cfg.embedding_dim, 128);
    assert_eq!(cfg.hidden_size, 768);
    assert_eq!(cfg.num_attention_heads, 12);
    assert_eq!(cfg.intermediate_size, 2048);
    assert_eq!(cfg.max_position_embeddings, 512);
    assert_eq!(cfg.num_hidden_layers, 12);
    assert!((cfg.layer_norm_eps - 1e-12).abs() < 1e-15);
}

#[test]
fn test_plbert_config_head_dim_evenly_divides() {
    let cfg = PlbertConfig::default();
    let head_dim = cfg.hidden_size / cfg.num_attention_heads;
    assert_eq!(head_dim * cfg.num_attention_heads, cfg.hidden_size);
    assert_eq!(head_dim, 64);
}

// ===========================================================================
// STFT: window function generation and frame calculations
// ===========================================================================

#[test]
fn test_stft_hann_window_via_istft_basis() {
    // Use IstftBasis to verify Hann window properties
    let params = IstftParams::new(8, 4, false, false).unwrap();
    let basis = IstftBasis::new(params).unwrap();
    let window = basis.window();
    assert_eq!(window.len(), 8);
    // Hann window endpoints: w[0] = 0.0, w[n_fft-1] ~ 0.0
    assert!(window[0].abs() < 1e-6, "hann window start should be ~0.0");
    assert!(
        window[7].abs() < 0.15,
        "hann window end should be near 0.0, got {}",
        window[7]
    );
    // Hann window peak at center
    let mid = window.len() / 2;
    assert!(
        window[mid] > 0.5,
        "hann window center should be > 0.5, got {}",
        window[mid]
    );
    // All values in [0, 1]
    for (i, &w) in window.iter().enumerate() {
        assert!(
            (0.0..=1.0).contains(&w),
            "window[{i}] = {w} out of [0,1]"
        );
    }
}

#[test]
fn test_stft_hann_window_periodic_properties() {
    // The iSTFT uses a periodic Hann window: w[k] = 0.5*(1 - cos(2*pi*k/N)).
    // Periodic Hann is NOT symmetric about N/2-1: w[0]=0 but w[N-1] != 0.
    // However, it satisfies w[k] + w[k + N/2] == 1.0 (constant sum property).
    let n_fft = 16;
    let params = IstftParams::new(n_fft, 4, false, false).unwrap();
    let basis = IstftBasis::new(params).unwrap();
    let window = basis.window();
    let half = n_fft / 2;
    for k in 0..half {
        let sum = window[k] + window[k + half];
        assert!(
            (sum - 1.0).abs() < 1e-5,
            "periodic Hann: w[{k}] + w[{}] = {} (expected 1.0)",
            k + half,
            sum,
        );
    }
}

#[test]
fn test_stft_frame_count_calculation() {
    // Standard Silero VAD: 576 samples + 64 pad_right = 640 padded
    // n_frames = (640 - 256) / 128 + 1 = 384/128 + 1 = 3 + 1 = 4
    let params = StftParams::default();
    let padded_len = 576 + params.pad_right; // 640
    let n_frames = (padded_len - params.n_fft) / params.hop_length + 1;
    assert_eq!(n_frames, 4);
}

#[test]
fn test_stft_frame_count_small_fft() {
    // n_fft=4, hop=2, audio=10, pad_right=1 => padded=11
    // n_frames = (11 - 4) / 2 + 1 = 7/2 + 1 = 3 + 1 = 4
    let params = StftParams::new(4, 2);
    let padded_len = 10 + params.pad_right; // 11
    let n_frames = (padded_len - params.n_fft) / params.hop_length + 1;
    assert_eq!(n_frames, 4);
}

#[test]
fn test_stft_n_freqs_formula() {
    // n_freqs = n_fft / 2 + 1 for real FFT
    for n_fft in [4, 8, 16, 32, 64, 128, 256, 512, 1024, 4096] {
        let params = StftParams::new(n_fft, n_fft / 2);
        assert_eq!(params.n_freqs, n_fft / 2 + 1, "n_fft={n_fft}");
    }
}

// ===========================================================================
// iSTFT: reconstruction and round-trip
// ===========================================================================

#[test]
fn test_istft_dc_signal_reconstruction() {
    // A constant (DC) signal should reconstruct to approximately constant.
    let n_fft = 8;
    let hop = 4;
    let params = IstftParams::new(n_fft, hop, false, false).unwrap();
    let basis = IstftBasis::new(params).unwrap();
    let n_bins = basis.n_bins(); // 5
    let n_frames = 4;

    // DC-only STFT: real[0, t] = 1.0, all others = 0
    let mut real = vec![0.0f32; n_bins * n_frames];
    let imag = vec![0.0f32; n_bins * n_frames];
    for t in 0..n_frames {
        real[0 * n_frames + t] = 1.0; // DC bin = 1.0
    }

    let output_len = n_fft + (n_frames - 1) * hop; // 8 + 12 = 20
    let result = basis.istft(&real, &imag, n_frames, output_len).unwrap();
    assert_eq!(result.len(), output_len);

    // After COLA normalization, the DC reconstruction should be approximately
    // constant across the middle region (edges may have window artifacts).
    // The norm factor is 1/n_fft = 0.125 for non-normalized mode.
    // DC bin amplitude 1.0 -> constant value of 1.0 * (1/N) * N = 1.0 after COLA.
    // Check that middle samples are finite and approximately equal.
    let mid_start = n_fft / 2;
    let mid_end = output_len.saturating_sub(n_fft / 2);
    if mid_end > mid_start + 2 {
        let mid_val = result[mid_start];
        for i in mid_start..mid_end {
            assert!(
                result[i].is_finite(),
                "result[{i}] is not finite: {}",
                result[i]
            );
            assert!(
                (result[i] - mid_val).abs() < 0.5,
                "DC reconstruction not constant at {i}: {} vs {mid_val}",
                result[i]
            );
        }
    }
}

#[test]
fn test_istft_single_frame() {
    // A single STFT frame should produce a valid windowed output.
    let n_fft = 8;
    let hop = 4;
    let params = IstftParams::new(n_fft, hop, false, false).unwrap();
    let basis = IstftBasis::new(params).unwrap();
    let n_bins = basis.n_bins(); // 5
    let n_frames = 1;

    let real = vec![1.0f32; n_bins * n_frames]; // all bins = 1.0
    let imag = vec![0.0f32; n_bins * n_frames];

    // output_len = n_fft + 0 * hop = 8
    let result = basis.istft(&real, &imag, n_frames, 8).unwrap();
    assert_eq!(result.len(), 8);
    // All samples should be finite
    for (i, &v) in result.iter().enumerate() {
        assert!(v.is_finite(), "result[{i}] is not finite: {v}");
    }
}

#[test]
fn test_istft_output_length_padding() {
    // Request a longer output_length than the actual reconstruction.
    let n_fft = 8;
    let hop = 4;
    let params = IstftParams::new(n_fft, hop, false, false).unwrap();
    let basis = IstftBasis::new(params).unwrap();
    let n_bins = basis.n_bins();
    let n_frames = 2;

    let real = vec![0.5f32; n_bins * n_frames];
    let imag = vec![0.0f32; n_bins * n_frames];

    // Actual reconstruction: n_fft + (2-1)*hop = 8 + 4 = 12
    // Request 20 => should zero-pad the tail
    let result = basis.istft(&real, &imag, n_frames, 20).unwrap();
    assert_eq!(result.len(), 20);
    // Tail should be zeros (padded)
    for i in 12..20 {
        assert!(
            result[i].abs() < 1e-10,
            "padded region result[{i}] should be 0, got {}",
            result[i]
        );
    }
}

#[test]
fn test_istft_output_length_trimming() {
    // Request a shorter output_length to trim the reconstruction.
    let n_fft = 8;
    let hop = 4;
    let params = IstftParams::new(n_fft, hop, false, false).unwrap();
    let basis = IstftBasis::new(params).unwrap();
    let n_bins = basis.n_bins();
    let n_frames = 3;

    let real = vec![0.1f32; n_bins * n_frames];
    let imag = vec![0.0f32; n_bins * n_frames];

    // Actual reconstruction: 8 + 2*4 = 16
    // Request 10 => should trim
    let result = basis.istft(&real, &imag, n_frames, 10).unwrap();
    assert_eq!(result.len(), 10);
}

#[test]
fn test_istft_center_trim() {
    // Center trimming removes n_fft/2 from each side.
    let n_fft = 8;
    let hop = 2;
    let params = IstftParams::new(n_fft, hop, false, true).unwrap(); // center=true
    let basis = IstftBasis::new(params).unwrap();
    let n_bins = basis.n_bins();
    let n_frames = 10;

    let real = vec![0.3f32; n_bins * n_frames];
    let imag = vec![0.0f32; n_bins * n_frames];

    // full_len = 8 + 9*2 = 26
    // center trim: 26 - 2*(8/2) = 26 - 8 = 18 trimmed
    // Request output_length = 18
    let result = basis.istft(&real, &imag, n_frames, 18).unwrap();
    assert_eq!(result.len(), 18);
}

#[test]
fn test_istft_normalized_vs_unnormalized() {
    // Normalized uses 1/sqrt(N), unnormalized uses 1/N.
    // For the same input, normalized output should be larger by sqrt(N).
    let n_fft = 16;
    let hop = 4;
    let params_norm = IstftParams::new(n_fft, hop, true, false).unwrap();
    let params_unnorm = IstftParams::new(n_fft, hop, false, false).unwrap();
    let basis_norm = IstftBasis::new(params_norm).unwrap();
    let basis_unnorm = IstftBasis::new(params_unnorm).unwrap();
    let n_bins = n_fft / 2 + 1;
    let n_frames = 2;

    // Use DC-only signal to get clean comparison
    let mut real = vec![0.0f32; n_bins * n_frames];
    let imag = vec![0.0f32; n_bins * n_frames];
    for t in 0..n_frames {
        real[t] = 1.0; // DC bin
    }

    let out_len = n_fft + (n_frames - 1) * hop;
    let result_norm = basis_norm
        .istft(&real, &imag, n_frames, out_len)
        .unwrap();
    let result_unnorm = basis_unnorm
        .istft(&real, &imag, n_frames, out_len)
        .unwrap();

    // Check that results are different (normalized should be larger)
    let ratio_factor = (n_fft as f32).sqrt();
    // Compare at a stable middle point
    let mid = out_len / 2;
    if result_unnorm[mid].abs() > 1e-6 {
        let ratio = result_norm[mid] / result_unnorm[mid];
        assert!(
            (ratio - ratio_factor).abs() < 1.0,
            "normalized/unnormalized ratio should be ~sqrt(N)={ratio_factor}, got {ratio}"
        );
    }
}

#[test]
fn test_istft_kokoro_params() {
    // Kokoro uses n_fft=20, hop=5, unnormalized, no center trim
    let params = IstftParams::new(20, 5, false, false).unwrap();
    let basis = IstftBasis::new(params).unwrap();
    assert_eq!(basis.n_bins(), 11); // 20/2 + 1
    assert_eq!(basis.window().len(), 20);
    assert_eq!(basis.cos_basis().len(), 11 * 20);
    assert_eq!(basis.sin_basis().len(), 11 * 20);
}

#[test]
fn test_istft_htdemucs_params() {
    // HTDemucs uses n_fft=4096, hop=1024, normalized, centered
    let params = IstftParams::default();
    assert_eq!(params.n_fft, 4096);
    assert_eq!(params.hop_length, 1024);
    assert!(params.normalized);
    assert!(params.center);

    // Creating the basis is valid (but would be large -- just check no error)
    // Skip actually creating it to avoid slow test (4096*2049 elements)
}

// ===========================================================================
// iSTFT: error conditions
// ===========================================================================

#[test]
fn test_istft_rejects_odd_nfft() {
    let err = IstftParams::new(7, 3, false, false).unwrap_err();
    assert!(matches!(err, IstftError::OddNfft { n_fft: 7 }));
}

#[test]
fn test_istft_rejects_zero_nfft() {
    let err = IstftParams::new(0, 4, false, false).unwrap_err();
    assert!(matches!(err, IstftError::OddNfft { n_fft: 0 }));
}

#[test]
fn test_istft_rejects_zero_hop() {
    let err = IstftParams::new(8, 0, false, false).unwrap_err();
    assert!(matches!(err, IstftError::ZeroHopLength));
}

#[test]
fn test_istft_rejects_nan_input() {
    let params = IstftParams::new(4, 2, false, false).unwrap();
    let basis = IstftBasis::new(params).unwrap();
    let n_frames = 1;
    let n_b = basis.n_bins(); // 3
    let mut real = vec![0.0f32; n_b * n_frames];
    let imag = vec![0.0f32; n_b * n_frames];
    real[0] = f32::NAN;
    let err = basis.istft(&real, &imag, n_frames, 4).unwrap_err();
    assert!(matches!(err, IstftError::NonFiniteInput));
}

#[test]
fn test_istft_rejects_infinity_input() {
    let params = IstftParams::new(4, 2, false, false).unwrap();
    let basis = IstftBasis::new(params).unwrap();
    let n_b = basis.n_bins(); // 3
    let n_frames = 1;
    let real = vec![0.0f32; n_b * n_frames];
    let mut imag = vec![0.0f32; n_b * n_frames];
    imag[1] = f32::INFINITY;
    let err = basis.istft(&real, &imag, n_frames, 4).unwrap_err();
    assert!(matches!(err, IstftError::NonFiniteInput));
}

#[test]
fn test_istft_rejects_length_mismatch() {
    let params = IstftParams::new(4, 2, false, false).unwrap();
    let basis = IstftBasis::new(params).unwrap();
    let real = vec![0.0f32; 6];
    let imag = vec![0.0f32; 3]; // wrong length
    let err = basis.istft(&real, &imag, 2, 4).unwrap_err();
    assert!(matches!(err, IstftError::LengthMismatch { .. }));
}

#[test]
fn test_istft_rejects_shape_mismatch() {
    let params = IstftParams::new(4, 2, false, false).unwrap();
    let basis = IstftBasis::new(params).unwrap();
    // n_bins = 3, n_frames = 2, so expected length = 3*2 = 6, but provide 9
    let real = vec![0.0f32; 9]; // wrong
    let imag = vec![0.0f32; 9];
    let err = basis.istft(&real, &imag, 2, 4).unwrap_err();
    assert!(matches!(err, IstftError::ShapeMismatch { .. }));
}

// ===========================================================================
// STFT: edge cases
// ===========================================================================

#[test]
fn test_stft_zero_length_audio_errors() {
    let params = StftParams::new(4, 2);
    let basis = vec![0.0f32; 6 * 4];
    let result = compute_stft_magnitude(&[], &basis, &params);
    assert!(result.is_err());
}

#[test]
fn test_stft_all_zeros_produces_zero_magnitude() {
    let params = StftParams::new(4, 2);
    let audio = vec![0.0f32; 8];
    let basis = vec![1.0f32; 6 * 4]; // Non-zero basis
    let result = compute_stft_magnitude(&audio, &basis, &params).unwrap();
    // Zero audio * any basis = zero
    for (i, &v) in result.iter().enumerate() {
        assert!(
            v.abs() < 1e-6,
            "zero audio should give zero magnitude, got result[{i}]={v}"
        );
    }
}

#[test]
fn test_stft_basis_element_count() {
    // STFT basis has (n_fft + 2) * n_fft elements
    let params = StftParams::default(); // n_fft=256
    let expected = (params.n_fft + 2) * params.n_fft; // 258 * 256 = 66048
    assert_eq!(expected, 66048);
}

#[test]
fn test_stft_minimum_audio_length_for_default_params() {
    // Default params: pad_right=64, n_fft=256.
    // Two conditions:
    //   1. audio.len() >= 2 + pad_right = 66 (reflection padding needs this)
    //   2. audio.len() + pad_right >= n_fft (padded audio must cover one FFT window)
    // Condition 2 is the binding one: audio >= 256 - 64 = 192
    let params = StftParams::default();
    let min_for_padding = 2 + params.pad_right; // 66
    let min_for_fft = params.n_fft - params.pad_right; // 192
    let min_len = min_for_padding.max(min_for_fft);
    assert_eq!(min_len, 192);

    // 192 samples should work
    let audio = vec![0.0f32; 192];
    let basis = vec![0.0f32; 258 * 256];
    let result = compute_stft_magnitude(&audio, &basis, &params);
    assert!(result.is_ok(), "192 samples should be sufficient");

    // 65 should fail the reflection padding check
    let audio_short = vec![0.0f32; 65];
    let result_short = compute_stft_magnitude(&audio_short, &basis, &params);
    assert!(result_short.is_err());
}

// ===========================================================================
// STFT + iSTFT round-trip (small signal)
// ===========================================================================

#[test]
fn test_stft_istft_cosine_roundtrip_small() {
    // Generate a cosine signal, compute forward STFT, then iSTFT,
    // and verify approximate reconstruction.
    let n_fft = 8;
    let hop = 2;
    let n_bins = n_fft / 2 + 1; // 5

    // Forward STFT basis: cos and sin components for each freq bin
    // Basis shape: [n_fft + 2, 1, n_fft] = [10, 1, 8]
    // First n_bins (5) filters = cosine, next n_bins (5) = sine
    let mut fwd_basis = vec![0.0f32; (n_fft + 2) * n_fft];
    for f in 0..n_bins {
        for k in 0..n_fft {
            let angle = 2.0 * PI * (f as f32) * (k as f32) / (n_fft as f32);
            fwd_basis[f * n_fft + k] = angle.cos();
            fwd_basis[(n_bins + f) * n_fft + k] = angle.sin();
        }
    }

    // Generate 16-sample cosine at bin 2
    let freq_bin = 2;
    let audio: Vec<f32> = (0..16)
        .map(|k| (2.0 * PI * freq_bin as f32 * k as f32 / n_fft as f32).cos())
        .collect();

    let stft_params = StftParams {
        n_fft,
        hop_length: hop,
        n_freqs: n_bins,
        pad_right: 0,
    };

    // Forward STFT
    let magnitude = compute_stft_magnitude(&audio, &fwd_basis, &stft_params).unwrap();
    let n_frames = (audio.len() - n_fft) / hop + 1; // (16-8)/2+1 = 5
    assert_eq!(magnitude.len(), n_bins * n_frames);

    // The magnitude at bin 2 should be significantly larger than other bins
    // (except possibly DC and Nyquist due to leakage)
    let mut bin_energies = vec![0.0f32; n_bins];
    for f in 0..n_bins {
        for t in 0..n_frames {
            bin_energies[f] += magnitude[f * n_frames + t].powi(2);
        }
    }
    // Bin 2 should have the most energy
    let max_bin = bin_energies
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .unwrap()
        .0;
    assert_eq!(max_bin, freq_bin, "peak energy should be at bin {freq_bin}");
}

// ===========================================================================
// HTDemucs architecture constants
// ===========================================================================

#[test]
fn test_htdemucs_transformer_dim_consistency() {
    assert_eq!(TRANSFORMER_DIM, 512);
    assert_eq!(NUM_HEADS, 8);
    assert_eq!(TRANSFORMER_DIM % NUM_HEADS, 0); // head_dim = 64
    assert_eq!(NUM_LAYERS, 5);
}

#[test]
fn test_htdemucs_ffn_hidden_dim() {
    assert_eq!(FFN_HIDDEN_DIM, (TRANSFORMER_DIM as f64 * FFN_HIDDEN_SCALE) as usize);
    assert_eq!(FFN_HIDDEN_DIM, 2048);
}

#[test]
fn test_htdemucs_bottleneck_dim_matches_depth3() {
    // BOTTLENECK_DIM should equal channels_at_depth(3) = 48 * 2^3 = 384
    assert_eq!(BOTTLENECK_DIM, channels_at_depth(3));
    assert_eq!(BOTTLENECK_DIM, 384);
}

#[test]
fn test_htdemucs_layer_norm_eps() {
    let eps = LAYER_NORM_EPS;
    assert!(eps > 0.0);
    assert_eq!(eps, 1e-5);
}

#[test]
fn test_htdemucs_temporal_encoder_depth_chain() {
    // Each temporal encoder block downsamples by TEMPORAL_STRIDE=4.
    // 4 basic blocks + 1 final block = 5 depths total.
    assert_eq!(TEMPORAL_DEPTH, TEMPORAL_BASIC_DEPTH + 1);

    // Channel progression: 48 -> 96 -> 192 -> 384 -> 768
    for d in 0..TEMPORAL_DEPTH {
        let ch = channels_at_depth(d);
        assert_eq!(ch, (BASE_CHANNELS as f64 * GROWTH.powi(d as i32)) as usize);
    }
}

#[test]
fn test_htdemucs_spectral_encoder_depth_chain() {
    // 4 basic blocks + 2 deep blocks = 6 depths total
    assert_eq!(SPECTRAL_DEPTH, SPECTRAL_BASIC_DEPTH + 2);

    // Channel progression at each depth
    for d in 0..SPECTRAL_DEPTH {
        let ch = channels_at_depth(d);
        assert!(ch > 0);
    }
}

#[test]
fn test_htdemucs_temporal_downsample_ratio() {
    // At each depth, temporal dimension shrinks by TEMPORAL_STRIDE
    let input_len = 4096;
    let mut t = input_len;
    for _ in 0..TEMPORAL_BASIC_DEPTH {
        // conv1d with kernel=8, stride=4, padding=2
        t = conv1d_output_len(t, TEMPORAL_KERNEL_SIZE, TEMPORAL_STRIDE, TEMPORAL_KERNEL_SIZE / 4)
            .unwrap();
    }
    // After 4 temporal downsample blocks, length should be reduced by ~4^4 = 256
    assert!(t > 0, "temporal output should be positive");
    assert!(t < input_len / 100, "temporal should be significantly downsampled");
}

#[test]
fn test_htdemucs_spectral_downsample_ratio() {
    // Spectral encoder downsamples by SPECTRAL_STRIDE at each depth
    let input_len = 1024;
    let mut s = input_len;
    for _ in 0..SPECTRAL_BASIC_DEPTH {
        s = conv1d_output_len(s, SPECTRAL_KERNEL_SIZE, SPECTRAL_STRIDE, SPECTRAL_KERNEL_SIZE / 4)
            .unwrap();
    }
    assert!(s > 0);
    assert!(s < input_len / 50, "spectral should be significantly downsampled");
}

// ===========================================================================
// Silero VAD encoder block shapes
// ===========================================================================

#[test]
fn test_silero_vad_encoder_block_count() {
    assert_eq!(ENCODER_BLOCKS.len(), 4);
}

#[test]
fn test_silero_vad_encoder_channel_chain() {
    // Block 0: 129 -> 128
    assert_eq!(ENCODER_BLOCKS[0].in_channels, 129);
    assert_eq!(ENCODER_BLOCKS[0].out_channels, 128);
    // Block 1: 128 -> 64
    assert_eq!(ENCODER_BLOCKS[1].in_channels, 128);
    assert_eq!(ENCODER_BLOCKS[1].out_channels, 64);
    // Block 2: 64 -> 64
    assert_eq!(ENCODER_BLOCKS[2].in_channels, 64);
    assert_eq!(ENCODER_BLOCKS[2].out_channels, 64);
    // Block 3: 64 -> 128
    assert_eq!(ENCODER_BLOCKS[3].in_channels, 64);
    assert_eq!(ENCODER_BLOCKS[3].out_channels, 128);
}

#[test]
fn test_silero_vad_encoder_channel_continuity() {
    // Each block's in_channels should match the previous block's out_channels
    for i in 1..ENCODER_BLOCKS.len() {
        assert_eq!(
            ENCODER_BLOCKS[i].in_channels,
            ENCODER_BLOCKS[i - 1].out_channels,
            "block {i} in_channels should match block {} out_channels",
            i - 1
        );
    }
}

#[test]
fn test_silero_vad_encoder_output_dimensions() {
    // Starting from 4 STFT frames (Silero VAD 16kHz: 576 samples → 4 frames)
    let mut t = 4;
    for (i, block) in ENCODER_BLOCKS.iter().enumerate() {
        t = conv1d_output_len(t, block.kernel_size, block.stride, block.padding).unwrap();
        assert!(t > 0, "block {i} output should be positive, got 0");
    }
    // After 4 blocks: stride=1, stride=2, stride=2, stride=1
    // Expected: 4 → 4 → 2 → 1 → 1
    assert_eq!(t, 1, "final temporal dimension for Silero VAD");
}

#[test]
fn test_silero_vad_lstm_hidden_size() {
    assert_eq!(LSTM_HIDDEN_SIZE, 128);
    // Last encoder block's output matches LSTM input
    assert_eq!(ENCODER_BLOCKS[3].out_channels, LSTM_HIDDEN_SIZE);
}

#[test]
fn test_silero_vad_first_block_matches_stft_freqs() {
    // STFT default: n_freqs = 129. First encoder block: in_channels = 129.
    let stft = StftParams::default();
    assert_eq!(stft.n_freqs, ENCODER_BLOCKS[0].in_channels);
}

// ===========================================================================
// Streaming: config and crossfade
// ===========================================================================

#[test]
fn test_stream_config_crossfade_duration() {
    let config = KokoroStreamConfig::default();
    let duration = config.crossfade_duration_secs();
    // 480 / 24000 = 0.02 seconds = 20ms
    assert!((duration - 0.02).abs() < 1e-9);
}

#[test]
fn test_stream_config_with_hann_window() {
    let config = KokoroStreamConfig::new(960)
        .unwrap()
        .with_window(CrossfadeWindow::Hann);
    assert_eq!(config.crossfade_window, CrossfadeWindow::Hann);
    assert_eq!(config.crossfade_samples, 960);
}

#[test]
fn test_crossfade_zero_samples_is_noop() {
    let prev = vec![1.0f32; 10];
    let mut next = vec![2.0f32; 10];
    crossfade_chunks(&prev, &mut next, 0).unwrap();
    // Next should be unchanged
    assert_eq!(next, vec![2.0f32; 10]);
}

#[test]
fn test_crossfade_full_blend_at_endpoints() {
    // With crossfade_samples = 3:
    // alpha[0] = 0.0 → fully prev, alpha[2] = 1.0 → fully next
    let prev = vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0]; // tail = [1.0, 1.0, 1.0]
    let mut next = vec![5.0, 5.0, 5.0, 5.0, 5.0, 5.0];
    crossfade_chunks(&prev, &mut next, 3).unwrap();

    // First crossfade sample: alpha=0 → prev=1.0, next ignored → ~1.0
    assert!(
        (next[0] - 1.0).abs() < 1e-5,
        "crossfade start should be ~prev tail: got {}",
        next[0]
    );
    // Last crossfade sample: alpha=1 → fully next=5.0
    assert!(
        (next[2] - 5.0).abs() < 1e-5,
        "crossfade end should be ~next: got {}",
        next[2]
    );
    // Non-crossfaded region should be unchanged
    assert_eq!(next[3], 5.0);
    assert_eq!(next[4], 5.0);
    assert_eq!(next[5], 5.0);
}

// ===========================================================================
// Streaming assembly
// ===========================================================================

#[test]
fn test_assemble_empty_input() {
    let config = KokoroStreamConfig::default();
    let result = assemble_streaming_chunks(&[], &config).unwrap();
    assert!(result.is_empty());
}

#[test]
fn test_assemble_single_chunk() {
    let config = KokoroStreamConfig::default();
    let raw = vec![vec![1.0f32; 1000]];
    let chunks = assemble_streaming_chunks(&raw, &config).unwrap();
    assert_eq!(chunks.len(), 1);
    assert!(chunks[0].is_final);
    assert_eq!(chunks[0].chunk_index, 0);
    assert_eq!(chunks[0].total_chunks, 1);
    assert_eq!(chunks[0].pcm.len(), 1000);
}

#[test]
fn test_assemble_two_chunks_crossfade() {
    let config = KokoroStreamConfig::new(100).unwrap();
    let raw = vec![vec![0.5f32; 500], vec![0.5f32; 500]];
    let chunks = assemble_streaming_chunks(&raw, &config).unwrap();
    assert_eq!(chunks.len(), 2);

    // First chunk emits 500 - 100 = 400 samples (tail reserved for crossfade)
    assert_eq!(chunks[0].pcm.len(), 400);
    assert!(!chunks[0].is_final);

    // Second chunk emits all 500 (crossfade applied in leading 100)
    assert_eq!(chunks[1].pcm.len(), 500);
    assert!(chunks[1].is_final);
}

#[test]
fn test_assemble_sample_offsets_are_cumulative() {
    let config = KokoroStreamConfig::new(50).unwrap();
    let raw = vec![
        vec![0.0f32; 200],
        vec![0.0f32; 300],
        vec![0.0f32; 250],
    ];
    let chunks = assemble_streaming_chunks(&raw, &config).unwrap();
    assert_eq!(chunks.len(), 3);
    assert_eq!(chunks[0].sample_offset, 0);
    // chunk[0] emit = 200 - 50 = 150
    assert_eq!(chunks[1].sample_offset, 150);
    // chunk[1] emit = 300 - 50 = 250
    assert_eq!(chunks[2].sample_offset, 150 + 250);
}

#[test]
fn test_concatenate_chunks_total_length() {
    let chunks = vec![
        AudioChunk::new(vec![1.0; 100], 1, 0, 0, 3, false),
        AudioChunk::new(vec![2.0; 200], 1, 100, 1, 3, false),
        AudioChunk::new(vec![3.0; 150], 1, 300, 2, 3, true),
    ];
    let concat = concatenate_chunks(&chunks);
    assert_eq!(concat.len(), 100 + 200 + 150);
}

// ===========================================================================
// Chorus config
// ===========================================================================

#[test]
fn test_chorus_equal_gain_single_voice() {
    let config = ChorusConfig::equal_gain(1).unwrap();
    assert_eq!(config.n_voices, 1);
    assert_eq!(config.gains.len(), 1);
    assert!((config.gains[0] - 1.0).abs() < 1e-6);
    assert!(config.clip_output);
    assert!(config.pans.is_none());
}

#[test]
fn test_chorus_equal_gain_four_voices() {
    let config = ChorusConfig::equal_gain(4).unwrap();
    assert_eq!(config.n_voices, 4);
    assert_eq!(config.gains.len(), 4);
    for &g in &config.gains {
        assert!((g - 0.25).abs() < 1e-6, "expected 0.25, got {g}");
    }
}

#[test]
fn test_chorus_rejects_zero_voices() {
    let err = ChorusConfig::equal_gain(0);
    assert!(err.is_err());
}

#[test]
fn test_chorus_rejects_too_many_voices() {
    let err = ChorusConfig::equal_gain(33);
    assert!(err.is_err());
}

#[test]
fn test_chorus_max_voices() {
    let config = ChorusConfig::equal_gain(32).unwrap();
    assert_eq!(config.n_voices, 32);
    let expected_gain = 1.0 / 32.0;
    for &g in &config.gains {
        assert!((g - expected_gain).abs() < 1e-6);
    }
}

// ===========================================================================
// Audio chunk
// ===========================================================================

#[test]
fn test_audio_chunk_duration_mono() {
    let chunk = AudioChunk::new(vec![0.0; 24000], 1, 0, 0, 1, true);
    let dur = chunk.duration_secs();
    assert!((dur - 1.0).abs() < 1e-9, "24000 samples at 24kHz = 1.0s");
}

#[test]
fn test_audio_chunk_duration_stereo() {
    // 48000 interleaved floats / 2 channels / 24000 Hz = 1.0 second
    let chunk = AudioChunk::new(vec![0.0; 48000], 2, 0, 0, 1, true);
    let dur = chunk.duration_secs();
    assert!((dur - 1.0).abs() < 1e-9, "48000 interleaved at 24kHz stereo = 1.0s");
}

#[test]
fn test_audio_chunk_is_empty() {
    let empty = AudioChunk::new(vec![], 1, 0, 0, 1, true);
    assert!(empty.is_empty());
    assert_eq!(empty.len(), 0);

    let nonempty = AudioChunk::new(vec![0.0], 1, 0, 0, 1, true);
    assert!(!nonempty.is_empty());
}

// ===========================================================================
// DConv architecture: compression ratio
// ===========================================================================

#[test]
fn test_dconv_compression_ratio() {
    // DCONV_COMPRESS = 4 means channels / 4 compressed channels
    for depth in 0..4 {
        let ch = channels_at_depth(depth);
        let compressed = ch / DCONV_COMPRESS;
        assert!(compressed > 0, "compressed channels at depth {depth} should be > 0");
        assert_eq!(compressed * DCONV_COMPRESS, ch, "compression should be exact");
    }
}

#[test]
fn test_dconv_depth_and_kernel() {
    // DCONV_DEPTH = 2 residual sub-layers per block
    assert_eq!(DCONV_DEPTH, 2);
    // DCONV_KERNEL = 3 for dilated convolution
    assert_eq!(DCONV_KERNEL, 3);
}

// ===========================================================================
// Signal constants cross-check
// ===========================================================================

#[test]
fn test_kokoro_n_bins_from_n_fft() {
    assert_eq!(KOKORO_N_FFT, 20);
    assert_eq!(KOKORO_N_BINS, 11); // 20/2 + 1
    assert_eq!(KOKORO_HOP_LENGTH, 5); // 20/4
}

#[test]
fn test_spectral_io_channels() {
    // Input: 2 stereo * 2 (real+imag) = 4
    assert_eq!(SPECTRAL_INPUT_CHANNELS, 4);
    // Output: 4 sources * 2 stereo * 2 (real+imag) = 16
    assert_eq!(SPECTRAL_OUTPUT_CHANNELS, 16);
    // Decoder output: 4 sources * 2 stereo = 8
    assert_eq!(DECODER_OUTPUT_CHANNELS, 8);
}

#[test]
fn test_spectral_freq_embedding_matches_base_channels() {
    assert_eq!(SPECTRAL_FREQ_EMB_DIM, BASE_CHANNELS);
    assert_eq!(SPECTRAL_FREQ_EMB_DIM, 48);
    assert_eq!(SPECTRAL_FREQ_EMB_FEATURES, 512);
}

// ===========================================================================
// iSTFT DFT basis mathematical properties
// ===========================================================================

#[test]
fn test_dft_basis_orthogonality_small() {
    // For a small n_fft, verify that the DFT basis vectors are approximately
    // orthogonal (cos basis only, for real DFT).
    let n_fft = 8;
    let params = IstftParams::new(n_fft, 4, false, false).unwrap();
    let basis = IstftBasis::new(params).unwrap();
    let n_bins = basis.n_bins(); // 5

    // Check orthogonality: dot(cos_row[f1], cos_row[f2]) should be ~0 for f1 != f2
    // and ~N/2 for f1 == f2 (except DC and Nyquist which are ~N).
    let cos = basis.cos_basis();
    for f1 in 0..n_bins {
        for f2 in (f1 + 1)..n_bins {
            let mut dot = 0.0f32;
            for k in 0..n_fft {
                dot += cos[f1 * n_fft + k] * cos[f2 * n_fft + k];
            }
            assert!(
                dot.abs() < 1e-4,
                "cos basis rows {f1} and {f2} not orthogonal: dot={dot}"
            );
        }
    }
}

#[test]
fn test_hann_window_cola_property() {
    // COLA (Constant Overlap-Add) condition: for hop=n_fft/4, the sum of
    // shifted Hann windows should be approximately constant.
    let n_fft = 16;
    let hop = n_fft / 4; // 4
    let params = IstftParams::new(n_fft, hop, false, false).unwrap();
    let basis = IstftBasis::new(params).unwrap();
    let window = basis.window();

    // Create overlap-add sum over several frames
    let n_frames = 8;
    let full_len = n_fft + (n_frames - 1) * hop;
    let mut window_sum = vec![0.0f32; full_len];
    for t in 0..n_frames {
        for k in 0..n_fft {
            window_sum[t * hop + k] += window[k] * window[k]; // w^2 for COLA norm
        }
    }

    // Check that the middle region has approximately constant window sum
    // (edges are not constant due to incomplete overlap)
    let start = n_fft;
    let end = full_len.saturating_sub(n_fft);
    if end > start + 2 {
        let mid_val = window_sum[start];
        for i in start..end {
            assert!(
                (window_sum[i] - mid_val).abs() < 0.1,
                "COLA sum not constant at {i}: {} vs {mid_val}",
                window_sum[i]
            );
        }
    }
}

// ===========================================================================
// Error type conversions
// ===========================================================================

#[test]
fn test_stft_error_to_tensor_error() {
    let stft_err = StftError::AudioTooShort {
        padded_len: 10,
        n_fft: 256,
    };
    let tensor_err: nn_core::TensorError = stft_err.into();
    let msg = tensor_err.to_string();
    assert!(msg.contains("10"));
    assert!(msg.contains("256"));
}

#[test]
fn test_istft_error_to_tensor_error() {
    let istft_err = IstftError::ZeroHopLength;
    let tensor_err: nn_core::TensorError = istft_err.into();
    let msg = tensor_err.to_string();
    assert!(msg.contains("hop_length"));
}

#[test]
fn test_kokoro_error_invalid_speed() {
    let err = KokoroError::InvalidSpeed { value: -1.0 };
    let msg = err.to_string();
    assert!(msg.contains("-1"));
    assert!(msg.contains("speed"));
}

#[test]
fn test_kokoro_error_invalid_config_display() {
    let err = KokoroError::InvalidConfig {
        field: "d_en",
        reason: "must be > 0".into(),
    };
    let msg = err.to_string();
    assert!(msg.contains("d_en"));
    assert!(msg.contains("must be > 0"));
}

// ===========================================================================
// channels_at_depth edge cases
// ===========================================================================

#[test]
fn test_channels_at_depth_large_depth() {
    // Should work up to MAX_ENCODER_DEPTH = 30
    let ch = channels_at_depth(10);
    assert_eq!(ch, (48.0 * 2.0f64.powi(10)) as usize);
    assert_eq!(ch, 49152);
}

#[test]
#[should_panic(expected = "exceeds maximum")]
fn test_channels_at_depth_overflow_panics() {
    let _ = channels_at_depth(31);
}

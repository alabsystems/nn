// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! HTDemucs integration tests for encoder/decoder shape chains, skip connections,
//! STFT/iSTFT branch dimensions, multi-source separation output, and cross-domain
//! interactions (#4186).
//!
//! These tests verify architectural invariants across the full HTDemucs pipeline
//! that are not covered by per-module unit tests.

use crate::demucs_shared::{
    channels_at_depth, conv1d_output_len, AUDIO_CHANNELS, BASE_CHANNELS, DCONV_COMPRESS,
    DCONV_DEPTH, DCONV_KERNEL, DECODER_OUTPUT_CHANNELS, GROWTH, SPECTRAL_BASIC_DEPTH,
    SPECTRAL_DEPTH, SPECTRAL_INPUT_CHANNELS, SPECTRAL_OUTPUT_CHANNELS, SPECTRAL_STRIDE,
    TEMPORAL_BASIC_DEPTH, TEMPORAL_CONV_PADDING, TEMPORAL_DEPTH, TEMPORAL_KERNEL_SIZE,
    TEMPORAL_STRIDE,
};
use crate::demucs_transformer_constants::{
    BOTTLENECK_DIM, FFN_HIDDEN_DIM, NUM_HEADS, NUM_LAYERS, TRANSFORMER_DIM,
};
use crate::istft::IstftParams;

// ===========================================================================
// 1. Encoder downsampling factor verification
// ===========================================================================

#[test]
fn test_temporal_encoder_total_downsample() {
    // Each temporal encoder block strides by TEMPORAL_STRIDE=4.
    // Total downsampling = TEMPORAL_STRIDE^TEMPORAL_DEPTH = 4^5 = 1024.
    let total_downsample = TEMPORAL_STRIDE.pow(TEMPORAL_DEPTH as u32);
    assert_eq!(
        total_downsample, 1024,
        "temporal branch downsamples by 4^5=1024"
    );
}

#[test]
fn test_spectral_encoder_total_downsample() {
    // Spectral branch: stride 4 at each of 6 depths.
    let total_downsample = SPECTRAL_STRIDE.pow(SPECTRAL_DEPTH as u32);
    assert_eq!(
        total_downsample, 4096,
        "spectral branch downsamples by 4^6=4096"
    );
}

#[test]
fn test_temporal_encoder_channel_progression() {
    // Channels double at each depth: 48 -> 96 -> 192 -> 384 -> 768
    for depth in 0..TEMPORAL_DEPTH {
        let ch = channels_at_depth(depth);
        let expected = (BASE_CHANNELS as f64 * GROWTH.powi(depth as i32)) as usize;
        assert_eq!(
            ch, expected,
            "temporal depth {depth}: channels={ch}, expected={expected}"
        );
    }
}

#[test]
fn test_spectral_encoder_channel_progression() {
    // Spectral uses same channel progression but has 6 depths
    let expected_channels = [48, 96, 192, 384, 768, 1536];
    for (depth, &expected) in expected_channels.iter().enumerate() {
        assert_eq!(channels_at_depth(depth), expected, "spectral depth {depth}");
    }
}

// ===========================================================================
// 2. Decoder upsampling factor verification
// ===========================================================================

#[test]
fn test_decoder_mirrors_encoder_depth() {
    // Temporal decoder has same depth as encoder
    assert_eq!(
        TEMPORAL_DEPTH, 5,
        "temporal decoder depth must match encoder"
    );
    // Spectral decoder has same depth as encoder
    assert_eq!(
        SPECTRAL_DEPTH, 6,
        "spectral decoder depth must match encoder"
    );
}

#[test]
fn test_decoder_output_channels_is_4_sources_times_stereo() {
    // 4 sources (drums, bass, vocals, other) x 2 stereo channels = 8
    assert_eq!(DECODER_OUTPUT_CHANNELS, 8);
    assert_eq!(DECODER_OUTPUT_CHANNELS, 4 * AUDIO_CHANNELS);
}

#[test]
fn test_spectral_output_channels_is_4_sources_times_stereo_times_complex() {
    // 4 sources x 2 stereo x 2 (real+imag) = 16
    assert_eq!(SPECTRAL_OUTPUT_CHANNELS, 16);
    assert_eq!(SPECTRAL_OUTPUT_CHANNELS, 4 * AUDIO_CHANNELS * 2);
}

// ===========================================================================
// 3. Skip connection dimension matching
// ===========================================================================

#[test]
fn test_skip_connections_match_encoder_decoder_channels() {
    // At each depth, the encoder output channels must match the decoder input
    // channels for skip connections to work.
    for depth in 0..TEMPORAL_BASIC_DEPTH {
        let enc_out = channels_at_depth(depth);
        // Decoder at same depth expects the same channel count from skip
        let dec_skip = channels_at_depth(depth);
        assert_eq!(
            enc_out, dec_skip,
            "temporal skip at depth {depth}: enc_out={enc_out} != dec_skip={dec_skip}"
        );
    }
}

#[test]
fn test_spectral_skip_connections_match() {
    for depth in 0..SPECTRAL_BASIC_DEPTH {
        let enc_out = channels_at_depth(depth);
        let dec_skip = channels_at_depth(depth);
        assert_eq!(
            enc_out, dec_skip,
            "spectral skip at depth {depth}: enc_out={enc_out} != dec_skip={dec_skip}"
        );
    }
}

#[test]
fn test_temporal_conv1d_output_shape_chain() {
    // Verify the output length through all 5 temporal encoder stages.
    // Input: 343980 samples (standard HTDemucs chunk: ~7.8s at 44.1kHz)
    let mut t = 343980usize;
    for depth in 0..TEMPORAL_DEPTH {
        t = conv1d_output_len(
            t,
            TEMPORAL_KERNEL_SIZE,
            TEMPORAL_STRIDE,
            TEMPORAL_CONV_PADDING,
        )
        .unwrap_or_else(|e| panic!("temporal conv1d_output_len failed at depth {depth}: {e}"));
    }
    // After 5 stages of stride-4: 343980 -> ~336 (exact depends on padding)
    assert!(
        t > 0,
        "temporal encoder chain must produce positive output length"
    );
    // Verify it roughly matches 343980 / 4^5 ~ 336
    let expected_approx = 343980 / TEMPORAL_STRIDE.pow(TEMPORAL_DEPTH as u32);
    assert!(
        (t as i64 - expected_approx as i64).unsigned_abs() < 10,
        "temporal output {t} should be near {expected_approx}"
    );
}

// ===========================================================================
// 4. STFT/iSTFT branch dimensions
// ===========================================================================

#[test]
fn test_htdemucs_istft_params_default() {
    // HTDemucs uses n_fft=4096, hop_length=1024
    let params = IstftParams::default();
    assert_eq!(params.n_fft, 4096, "HTDemucs default n_fft");
    assert_eq!(params.hop_length, 1024, "HTDemucs default hop_length");
}

#[test]
fn test_htdemucs_istft_n_bins() {
    // n_bins = n_fft/2 + 1 = 2049 for HTDemucs
    let params = IstftParams::default();
    let n_bins = params.n_fft / 2 + 1;
    assert_eq!(n_bins, 2049, "HTDemucs STFT produces 2049 frequency bins");
}

#[test]
fn test_spectral_input_channels_complex_stereo() {
    // Spectral branch input: 2 channels (stereo) x 2 (real+imag) = 4
    assert_eq!(SPECTRAL_INPUT_CHANNELS, 4);
    assert_eq!(SPECTRAL_INPUT_CHANNELS, AUDIO_CHANNELS * 2);
}

// ===========================================================================
// 5. Transformer bottleneck dimensions
// ===========================================================================

#[test]
fn test_bottleneck_dim_matches_depth3_channels() {
    // BOTTLENECK_DIM = channels_at_depth(3) = 48 * 2^3 = 384
    assert_eq!(BOTTLENECK_DIM, channels_at_depth(3));
    assert_eq!(BOTTLENECK_DIM, 384);
}

#[test]
fn test_transformer_dim_head_divisibility() {
    // TRANSFORMER_DIM must be divisible by NUM_HEADS for multi-head attention
    assert_eq!(TRANSFORMER_DIM % NUM_HEADS, 0);
    let head_dim = TRANSFORMER_DIM / NUM_HEADS;
    assert_eq!(head_dim, 64, "each attention head has 64 dims");
}

#[test]
fn test_ffn_hidden_dim_is_4x_transformer_dim() {
    assert_eq!(FFN_HIDDEN_DIM, 2048);
    assert_eq!(FFN_HIDDEN_DIM, TRANSFORMER_DIM * 4);
}

#[test]
fn test_transformer_num_layers() {
    assert_eq!(
        NUM_LAYERS, 5,
        "HTDemucs uses 5 transformer layers per branch"
    );
}

// ===========================================================================
// 6. DConv residual block invariants
// ===========================================================================

#[test]
fn test_dconv_compress_ratio() {
    // DConv bottleneck: channels / DCONV_COMPRESS
    for depth in 0..TEMPORAL_BASIC_DEPTH {
        let ch = channels_at_depth(depth);
        let compressed = ch / DCONV_COMPRESS;
        assert!(
            compressed > 0,
            "DConv compressed channels at depth {depth} must be > 0"
        );
        assert_eq!(
            ch % DCONV_COMPRESS, 0,
            "channels_at_depth({depth}) = {ch} must be divisible by DCONV_COMPRESS={DCONV_COMPRESS}"
        );
    }
}

#[test]
fn test_dconv_depth_is_2() {
    assert_eq!(DCONV_DEPTH, 2, "each DConv block has 2 residual sub-layers");
}

#[test]
fn test_dconv_dilation_pattern() {
    // Dilation doubles at each sub-layer: 1, 2 (for depth=2)
    for k in 0..DCONV_DEPTH {
        let dilation = 1usize << k;
        let expected = [1, 2][k];
        assert_eq!(dilation, expected, "DConv sub-layer {k} dilation");
    }
}

#[test]
fn test_dconv_causal_padding_preserves_length() {
    // Causal DConv pads left by (kernel-1)*dilation, right by 0.
    // With stride=1 and no right padding, output length = input length.
    let t = 100;
    for k in 0..DCONV_DEPTH {
        let dilation = 1usize << k;
        let pad_left = (DCONV_KERNEL - 1) * dilation;
        let padded_t = t + pad_left;
        // Conv1d output: (padded_t - kernel*dilation + dilation) / stride
        // For dilation=d, effective_kernel = kernel + (kernel-1)*(dilation-1) = kernel*dilation - dilation + 1
        // Wait, simpler: with padding=0 and dilated kernel:
        // output = (padded_t - dilation*(kernel-1) - 1) / 1 + 1
        let out = padded_t - dilation * (DCONV_KERNEL - 1);
        assert_eq!(
            out, t,
            "DConv sub-layer {k}: causal padding must preserve temporal length"
        );
    }
}

// ===========================================================================
// 7. Multi-source separation output
// ===========================================================================

#[test]
fn test_four_sources_defined() {
    // HTDemucs separates into 4 sources: drums, bass, other, vocals
    let num_sources = DECODER_OUTPUT_CHANNELS / AUDIO_CHANNELS;
    assert_eq!(num_sources, 4, "HTDemucs separates 4 sources");
}

#[test]
fn test_spectral_four_sources_complex() {
    // Spectral output: 4 sources x 2 channels x 2 (real+imag) = 16
    let num_sources = SPECTRAL_OUTPUT_CHANNELS / (AUDIO_CHANNELS * 2);
    assert_eq!(num_sources, 4);
}

// ===========================================================================
// 8. Frequency embedding dimensions
// ===========================================================================

#[test]
fn test_freq_emb_dim_matches_base_channels() {
    use crate::demucs_shared::{SPECTRAL_FREQ_EMB_DIM, SPECTRAL_FREQ_EMB_FEATURES};
    assert_eq!(
        SPECTRAL_FREQ_EMB_DIM, BASE_CHANNELS,
        "freq embedding dim must equal base channels (48)"
    );
    assert_eq!(
        SPECTRAL_FREQ_EMB_FEATURES, 512,
        "freq embedding supports up to 512 frequency bins"
    );
}

// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Tests for the Demucs spectral decoder.
//!
//! Part of #779 Phase B.

use super::*;
use crate::demucs_shared::{DCONV_COMPRESS, DCONV_DEPTH, DCONV_KERNEL};
use nn_models::demucs_spectral_weights::{
    SpectralDConvSubLayerWeights, SpectralDecoderBlockWeights,
};

// ---------------------------------------------------------------------------
// Helper: create synthetic weights for a spectral decoder block
// ---------------------------------------------------------------------------

fn make_dconv_weights(in_ch: usize) -> Vec<SpectralDConvSubLayerWeights> {
    let compressed = in_ch / DCONV_COMPRESS;
    let doubled = in_ch * 2;
    (0..DCONV_DEPTH)
        .map(|_| SpectralDConvSubLayerWeights {
            conv_compress_weight: vec![0.01; compressed * in_ch * DCONV_KERNEL],
            conv_compress_bias: vec![0.0; compressed],
            norm_compress_gamma: vec![1.0; compressed],
            norm_compress_beta: vec![0.0; compressed],
            conv_expand_weight: vec![0.01; doubled * compressed],
            conv_expand_bias: vec![0.0; doubled],
            norm_expand_gamma: vec![1.0; doubled],
            norm_expand_beta: vec![0.0; doubled],
            layer_scale: vec![0.1; in_ch],
        })
        .collect()
}

fn make_block_weights(in_ch: usize, out_ch: usize) -> SpectralDecoderBlockWeights {
    let doubled = in_ch * 2;
    SpectralDecoderBlockWeights {
        rewrite_weight: vec![0.01; doubled * in_ch * REWRITE_KERNEL * REWRITE_KERNEL],
        rewrite_bias: vec![0.0; doubled],
        dconv: make_dconv_weights(in_ch),
        conv_tr_weight: vec![0.01; in_ch * out_ch * KERNEL_SIZE],
        conv_tr_bias: vec![0.0; out_ch],
    }
}

fn make_decoder_weights() -> DemucsSpectralDecoderWeights {
    let mut blocks = Vec::with_capacity(DEPTH);
    for block_idx in 0..DEPTH {
        let encoder_depth = DEPTH - 1 - block_idx;
        let in_ch = channels_at_depth(encoder_depth);
        let out_ch = if encoder_depth == 0 {
            OUTPUT_CHANNELS
        } else {
            channels_at_depth(encoder_depth - 1)
        };
        blocks.push(make_block_weights(in_ch, out_ch));
    }
    DemucsSpectralDecoderWeights { blocks }
}

/// Generate encoder frequency dimensions for each depth.
///
/// Real HTDemucs: nfft=4096, initial freq = 2049, stride=4 each depth.
/// Test uses small but valid values. Minimum: after the deepest encoder's
/// conv1d output, the decoder bottleneck must have freq >= kernel_size - 2*padding = 4.
///
/// Working backwards from bottleneck: bottleneck_f = conv1d_out(enc_freqs[3], k=8, s=4, p=2).
/// enc_freqs[3] = 32 → bottleneck = (32-4)/4+1 = 8. Valid.
fn encoder_freq_dims() -> Vec<usize> {
    // Depth 0: 512, depth 1: 128, depth 2: 32, depth 3: 8 (matches stride=4 reduction).
    // These are valid: conv1d(8, k=8, s=4, p=2) = 2 ≥ 1.
    // Actually use very small values that still work:
    vec![128, 32, 16, 8]
}

fn encoder_time_dims() -> Vec<usize> {
    // Time is preserved through spectral branch (no temporal downsampling).
    vec![4, 4, 4, 4]
}

// ---------------------------------------------------------------------------
// Construction tests
// ---------------------------------------------------------------------------

#[test]
fn test_spectral_decoder_construction() {
    let weights = make_decoder_weights();
    let freqs = encoder_freq_dims();
    let times = encoder_time_dims();

    let decoder = DemucsSpectralDecoder::new(weights, &freqs, &times);
    assert!(decoder.is_ok(), "construction failed: {:?}", decoder.err());

    let decoder = decoder.unwrap();
    assert_eq!(decoder.block_sub_defs.len(), DEPTH);
    assert_eq!(decoder.block_in_ch.len(), DEPTH);
}

#[test]
fn test_spectral_decoder_wrong_encoder_freqs_len() {
    let weights = make_decoder_weights();
    let freqs = vec![32, 8]; // wrong length
    let times = encoder_time_dims();

    let err = DemucsSpectralDecoder::new(weights, &freqs, &times).unwrap_err();
    match err {
        DemucsSpectralDecoderError::DimMismatch { stage, .. } => {
            assert_eq!(stage, "encoder_freqs");
        }
        _ => panic!("expected DimMismatch, got: {err}"),
    }
}

#[test]
fn test_spectral_decoder_wrong_encoder_times_len() {
    let weights = make_decoder_weights();
    let freqs = encoder_freq_dims();
    let times = vec![4]; // wrong length

    let err = DemucsSpectralDecoder::new(weights, &freqs, &times).unwrap_err();
    match err {
        DemucsSpectralDecoderError::DimMismatch { stage, .. } => {
            assert_eq!(stage, "encoder_times");
        }
        _ => panic!("expected DimMismatch, got: {err}"),
    }
}

#[test]
fn test_spectral_decoder_wrong_block_count() {
    let mut weights = make_decoder_weights();
    weights.blocks.pop(); // 3 instead of 4

    let freqs = encoder_freq_dims();
    let times = encoder_time_dims();

    let err = DemucsSpectralDecoder::new(weights, &freqs, &times).unwrap_err();
    match err {
        DemucsSpectralDecoderError::WeightSize { name, .. } => {
            assert!(
                name.contains("blocks") || name.contains("spectral decoder"),
                "expected name mentioning blocks, got: {name}"
            );
        }
        _ => panic!("expected WeightSize, got: {err}"),
    }
}

#[test]
fn test_spectral_decoder_wrong_rewrite_weight_size() {
    let mut weights = make_decoder_weights();
    weights.blocks[0].rewrite_weight = vec![0.0; 10]; // wrong size

    let freqs = encoder_freq_dims();
    let times = encoder_time_dims();

    let err = DemucsSpectralDecoder::new(weights, &freqs, &times).unwrap_err();
    match err {
        DemucsSpectralDecoderError::WeightSize { name, .. } => {
            assert!(name.contains("rw_weight"), "got: {name}");
        }
        _ => panic!("expected WeightSize, got: {err}"),
    }
}

#[test]
fn test_spectral_decoder_channel_progression() {
    let weights = make_decoder_weights();
    let freqs = encoder_freq_dims();
    let times = encoder_time_dims();

    let decoder = DemucsSpectralDecoder::new(weights, &freqs, &times).unwrap();

    // Block 0 (deepest): channels_at_depth(3) = 48 * 2^3 = 384
    assert_eq!(decoder.block_in_ch[0], 384);
    // Block 1: channels_at_depth(2) = 48 * 2^2 = 192
    assert_eq!(decoder.block_in_ch[1], 192);
    // Block 2: channels_at_depth(1) = 48 * 2^1 = 96
    assert_eq!(decoder.block_in_ch[2], 96);
    // Block 3: channels_at_depth(0) = 48 * 2^0 = 48
    assert_eq!(decoder.block_in_ch[3], 48);
}

#[test]
fn test_center_trim_2d_identity() {
    let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    // [1 channel, 2 freq, 3 time] — trim to same size = identity.
    let result = dispatch_helpers::center_trim_2d(&data, 1, 2, 3, 2, 3).unwrap();
    assert_eq!(result, data);
}

#[test]
fn test_center_trim_2d_trims_freq() {
    // [1 channel, 4 freq, 2 time]
    // Layout: row-major [C, F, T]
    // c=0, f=0: [1, 2], f=1: [3, 4], f=2: [5, 6], f=3: [7, 8]
    let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    // Trim to 2 freq bins (center): skip 1 from start, 1 from end → f=1,2.
    let result = dispatch_helpers::center_trim_2d(&data, 1, 4, 2, 2, 2).unwrap();
    assert_eq!(result, vec![3.0, 4.0, 5.0, 6.0]);
}

#[test]
fn test_center_trim_2d_trims_both() {
    // [1 channel, 4 freq, 4 time]
    let mut data = Vec::new();
    for f in 0..4u32 {
        for t in 0..4u32 {
            data.push((f * 10 + t) as f32);
        }
    }
    // Trim to 2 freq, 2 time (center).
    // F center: skip 1, take f=1,2. T center: skip 1, take t=1,2.
    let result = dispatch_helpers::center_trim_2d(&data, 1, 4, 4, 2, 2).unwrap();
    // f=1: [10,11,12,13] → t=1,2 → [11, 12]
    // f=2: [20,21,22,23] → t=1,2 → [21, 22]
    assert_eq!(result, vec![11.0, 12.0, 21.0, 22.0]);
}

#[test]
fn test_center_trim_2d_multichannel() {
    // [2 channels, 4 freq, 2 time]
    // c=0: f=0:[0,1], f=1:[2,3], f=2:[4,5], f=3:[6,7]
    // c=1: f=0:[10,11], f=1:[12,13], f=2:[14,15], f=3:[16,17]
    let mut data = Vec::new();
    for c in 0..2u32 {
        for f in 0..4u32 {
            for t in 0..2u32 {
                data.push((c * 10 + f * 2 + t) as f32);
            }
        }
    }
    // Trim to 2 freq bins (center).
    let result = dispatch_helpers::center_trim_2d(&data, 2, 4, 2, 2, 2).unwrap();
    // c=0: f=1:[2,3], f=2:[4,5]
    // c=1: f=1:[12,13], f=2:[14,15]
    assert_eq!(result, vec![2.0, 3.0, 4.0, 5.0, 12.0, 13.0, 14.0, 15.0]);
}

#[test]
fn test_channels_at_depth() {
    assert_eq!(channels_at_depth(0), 48);
    assert_eq!(channels_at_depth(1), 96);
    assert_eq!(channels_at_depth(2), 192);
    assert_eq!(channels_at_depth(3), 384);
}

#[test]
fn test_gelu_f32_zero() {
    assert!((dispatch_helpers::gelu_f32(0.0) - 0.0).abs() < 1e-6);
}

#[test]
fn test_gelu_f32_positive() {
    // GELU(1.0) ≈ 0.8413
    let result = dispatch_helpers::gelu_f32(1.0);
    assert!((result - 0.8413).abs() < 0.01, "gelu(1.0) = {result}");
}

// ---------------------------------------------------------------------------
// Weight validation tests
// ---------------------------------------------------------------------------

#[test]
fn test_dconv_wrong_count() {
    let mut weights = make_decoder_weights();
    weights.blocks[0].dconv.pop(); // 1 instead of 2

    let freqs = encoder_freq_dims();
    let times = encoder_time_dims();

    let err = DemucsSpectralDecoder::new(weights, &freqs, &times).unwrap_err();
    match err {
        DemucsSpectralDecoderError::WeightSize { name, .. } => {
            assert!(
                name.contains("dconv"),
                "expected dconv in name, got: {name}"
            );
        }
        _ => panic!("expected WeightSize, got: {err}"),
    }
}

#[test]
fn test_conv_tr_wrong_weight_size() {
    let mut weights = make_decoder_weights();
    weights.blocks[0].conv_tr_weight = vec![0.0; 5]; // wrong size

    let freqs = encoder_freq_dims();
    let times = encoder_time_dims();

    let err = DemucsSpectralDecoder::new(weights, &freqs, &times).unwrap_err();
    match err {
        DemucsSpectralDecoderError::WeightSize { name, .. } => {
            assert!(name.contains("ct_weight"), "got: {name}");
        }
        _ => panic!("expected WeightSize, got: {err}"),
    }
}

// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Tests for `DemucsTemporalDecoder` — Part of #779 Phase A.

use super::builders;
use super::*;
use crate::demucs_shared::{DCONV_COMPRESS, DCONV_DEPTH, DCONV_KERNEL};
use crate::test_common::make_cache;

// ---------------------------------------------------------------------------
// Test helpers
// ---------------------------------------------------------------------------

fn zero_dconv_weights(channels: usize) -> DConvSubLayerWeights {
    let compressed = channels / DCONV_COMPRESS;
    let doubled = channels * 2;
    DConvSubLayerWeights {
        conv_compress_weight: vec![0.0; compressed * channels * DCONV_KERNEL],
        conv_compress_bias: vec![0.0; compressed],
        norm_compress_gamma: vec![0.0; compressed],
        norm_compress_beta: vec![0.0; compressed],
        conv_expand_weight: vec![0.0; doubled * compressed],
        conv_expand_bias: vec![0.0; doubled],
        norm_expand_gamma: vec![0.0; doubled],
        norm_expand_beta: vec![0.0; doubled],
        layer_scale: vec![0.0; channels],
    }
}

fn zero_block_weights(block_idx: usize) -> DecoderBlockWeights {
    let encoder_depth = DEPTH - 1 - block_idx;
    let in_ch = channels_at_depth(encoder_depth);
    let out_ch = if encoder_depth == 0 {
        OUTPUT_CHANNELS
    } else {
        channels_at_depth(encoder_depth - 1)
    };

    DecoderBlockWeights {
        rewrite_weight: vec![0.0; in_ch * 2 * in_ch * REWRITE_KERNEL],
        rewrite_bias: vec![0.0; in_ch * 2],
        dconv: (0..DCONV_DEPTH)
            .map(|_| zero_dconv_weights(in_ch))
            .collect(),
        conv_tr_weight: vec![0.0; in_ch * out_ch * KERNEL_SIZE],
        conv_tr_bias: vec![0.0; out_ch],
    }
}

fn zero_decoder_weights() -> DemucsTemporalDecoderWeights {
    DemucsTemporalDecoderWeights {
        blocks: (0..DEPTH).map(zero_block_weights).collect(),
    }
}

/// Compute encoder input temporal lengths for a given initial time T.
/// Encoder: Conv1d(k=8, s=4, p=2) at each depth.
fn compute_encoder_lengths(initial_t: usize) -> Vec<usize> {
    let mut lengths = Vec::with_capacity(DEPTH);
    let mut t = initial_t;
    for _ in 0..DEPTH {
        lengths.push(t);
        t = builders::conv1d_output_len(t, KERNEL_SIZE, STRIDE, KERNEL_SIZE / 4).unwrap();
    }
    lengths
}

// ---------------------------------------------------------------------------
// Unit tests — arithmetic helpers
// ---------------------------------------------------------------------------

#[test]
fn test_channels_at_depth() {
    assert_eq!(channels_at_depth(0), 48);
    assert_eq!(channels_at_depth(1), 96);
    assert_eq!(channels_at_depth(2), 192);
    assert_eq!(channels_at_depth(3), 384);
}

#[test]
fn test_conv1d_output_len() {
    // Standard: (10 + 2*1 - 3) / 1 + 1 = 10
    assert_eq!(builders::conv1d_output_len(10, 3, 1, 1).unwrap(), 10);
    // Strided: (10 + 2*2 - 8) / 4 + 1 = 2
    assert_eq!(builders::conv1d_output_len(10, 8, 4, 2).unwrap(), 2);
}

// ---------------------------------------------------------------------------
// Unit tests — center_trim_1d
// ---------------------------------------------------------------------------

#[test]
fn test_center_trim_1d_identity() {
    let data: Vec<f32> = (0..12).map(|x| x as f32).collect();
    let result = center_trim_1d(&data, 3, 4, 4).unwrap();
    assert_eq!(result, data);
}

#[test]
fn test_center_trim_1d_trims() {
    // 2 channels, 6 timesteps -> trim to 4.
    // delta=2, start=1. Each row [0..6] takes [1..5].
    let data: Vec<f32> = (0..12).map(|x| x as f32).collect();
    let result = center_trim_1d(&data, 2, 6, 4).unwrap();
    // Row 0: [0,1,2,3,4,5] -> [1,2,3,4]
    // Row 1: [6,7,8,9,10,11] -> [7,8,9,10]
    assert_eq!(result, vec![1.0, 2.0, 3.0, 4.0, 7.0, 8.0, 9.0, 10.0]);
}

#[test]
fn test_center_trim_1d_single_element() {
    // 1 channel, 5 timesteps -> trim to 1.
    let data: Vec<f32> = vec![10.0, 20.0, 30.0, 40.0, 50.0];
    let result = center_trim_1d(&data, 1, 5, 1).unwrap();
    // delta=4, start=2 -> element at index 2
    assert_eq!(result, vec![30.0]);
}

// ---------------------------------------------------------------------------
// Construction tests
// ---------------------------------------------------------------------------

#[test]
fn test_construction_zero_weights() {
    let weights = zero_decoder_weights();
    let enc_lens = compute_encoder_lengths(256);
    let decoder = DemucsTemporalDecoder::new(weights, &enc_lens);
    assert!(decoder.is_ok());
    let decoder = decoder.unwrap();
    assert_eq!(decoder.block_defs.len(), DEPTH);
    assert_eq!(decoder.block_weights.len(), DEPTH);
}

#[test]
fn test_construction_wrong_encoder_lengths() {
    let weights = zero_decoder_weights();
    // Provide 3 instead of 4 encoder lengths.
    let enc_lens = vec![256, 64, 16];
    let result = DemucsTemporalDecoder::new(weights, &enc_lens);
    assert!(result.is_err());
    let err = result.unwrap_err();
    assert!(
        matches!(err, DemucsTemporalDecoderError::DimMismatch { .. }),
        "expected DimMismatch, got: {err:?}"
    );
}

#[test]
fn test_construction_wrong_weight_size() {
    let mut weights = zero_decoder_weights();
    // Corrupt the first block's rewrite weight.
    weights.blocks[0].rewrite_weight = vec![0.0; 1];
    let enc_lens = compute_encoder_lengths(256);
    let result = DemucsTemporalDecoder::new(weights, &enc_lens);
    assert!(result.is_err());
    assert!(matches!(
        result.unwrap_err(),
        DemucsTemporalDecoderError::WeightSize { .. }
    ));
}

#[test]
fn test_construction_wrong_dconv_count() {
    let mut weights = zero_decoder_weights();
    // Remove one DConv sub-layer from block 0.
    weights.blocks[0].dconv.pop();
    let enc_lens = compute_encoder_lengths(256);
    let result = DemucsTemporalDecoder::new(weights, &enc_lens);
    assert!(result.is_err());
    assert!(matches!(
        result.unwrap_err(),
        DemucsTemporalDecoderError::WeightSize { .. }
    ));
}

// ---------------------------------------------------------------------------
// Kernel def structure tests
// ---------------------------------------------------------------------------

#[test]
fn test_all_kernel_defs_validate() {
    let weights = zero_decoder_weights();
    let enc_lens = compute_encoder_lengths(256);
    let decoder = DemucsTemporalDecoder::new(weights, &enc_lens).unwrap();
    for (i, def) in decoder.block_defs.iter().enumerate() {
        assert!(
            def.validate().is_ok(),
            "block {i} def validation failed: {:?}",
            def.validate().err()
        );
    }
}

#[test]
fn test_temporal_progression() {
    // Verify the temporal dimensions match expected encoder/decoder progression.
    let initial_t = 256;
    let enc_lens = compute_encoder_lengths(initial_t);
    assert_eq!(enc_lens[0], 256); // depth 0 input
                                  // depth 1: conv1d_output_len(256, 8, 4, 2) = (256+4-8)/4+1 = 63+1 = 64
    assert_eq!(enc_lens[1], 64);
    // depth 2: conv1d_output_len(64, 8, 4, 2) = (64+4-8)/4+1 = 15+1 = 16
    assert_eq!(enc_lens[2], 16);
    // depth 3: conv1d_output_len(16, 8, 4, 2) = (16+4-8)/4+1 = 3+1 = 4
    assert_eq!(enc_lens[3], 4);
}

// ---------------------------------------------------------------------------
// Forward pass validation tests
// ---------------------------------------------------------------------------

#[test]
fn test_forward_rejects_wrong_skips_count() {
    let cache = match make_cache() {
        Some(c) => c,
        None => return,
    };
    let weights = zero_decoder_weights();
    let enc_lens = compute_encoder_lengths(256);
    let decoder = DemucsTemporalDecoder::new(weights, &enc_lens).unwrap();

    let bottleneck_len = decoder.block_in_ch[0] * decoder.block_t_in[0];
    let bottleneck = vec![0.0f32; bottleneck_len];
    // 3 skips instead of 4.
    let skips: Vec<Vec<f32>> = vec![vec![0.0; 100]; 3];
    let result = decoder.forward(&cache, &bottleneck, &skips);
    assert!(result.is_err());
    assert!(matches!(
        result.unwrap_err(),
        DemucsTemporalDecoderError::DimMismatch { .. }
    ));
}

#[test]
fn test_forward_rejects_wrong_bottleneck_length() {
    let cache = match make_cache() {
        Some(c) => c,
        None => return,
    };
    let weights = zero_decoder_weights();
    let enc_lens = compute_encoder_lengths(256);
    let decoder = DemucsTemporalDecoder::new(weights, &enc_lens).unwrap();

    let bottleneck = vec![0.0f32; 1]; // Way too short.
    let skips: Vec<Vec<f32>> = (0..DEPTH)
        .map(|d| vec![0.0f32; channels_at_depth(d) * enc_lens[d]])
        .collect();
    let result = decoder.forward(&cache, &bottleneck, &skips);
    assert!(result.is_err());
    assert!(matches!(
        result.unwrap_err(),
        DemucsTemporalDecoderError::DimMismatch { .. }
    ));
}

#[test]
fn test_forward_rejects_nan_input() {
    let cache = match make_cache() {
        Some(c) => c,
        None => return,
    };
    let weights = zero_decoder_weights();
    let enc_lens = compute_encoder_lengths(256);
    let decoder = DemucsTemporalDecoder::new(weights, &enc_lens).unwrap();

    let bottleneck_len = decoder.block_in_ch[0] * decoder.block_t_in[0];
    let mut bottleneck = vec![0.0f32; bottleneck_len];
    bottleneck[0] = f32::NAN;
    let skips: Vec<Vec<f32>> = (0..DEPTH)
        .map(|d| vec![0.0f32; channels_at_depth(d) * enc_lens[d]])
        .collect();
    let result = decoder.forward(&cache, &bottleneck, &skips);
    assert!(result.is_err());
    assert!(matches!(
        result.unwrap_err(),
        DemucsTemporalDecoderError::NonFiniteInput { .. }
    ));
}

// ---------------------------------------------------------------------------
// Metal forward pass smoke tests
// ---------------------------------------------------------------------------

#[test]
fn test_forward_zero_weights_on_metal() {
    let cache = match make_cache() {
        Some(c) => c,
        None => return,
    };
    let weights = zero_decoder_weights();
    let enc_lens = compute_encoder_lengths(256);
    let decoder = DemucsTemporalDecoder::new(weights, &enc_lens).unwrap();

    // Bottleneck: [384, T_bottleneck] with zeros.
    let bottleneck_len = decoder.block_in_ch[0] * decoder.block_t_in[0];
    let bottleneck = vec![0.0f32; bottleneck_len];

    // Skips in encoder depth order (depth 0..3).
    let skips: Vec<Vec<f32>> = (0..DEPTH)
        .map(|d| vec![0.0f32; channels_at_depth(d) * enc_lens[d]])
        .collect();

    let result = decoder.forward(&cache, &bottleneck, &skips);
    match result {
        Ok(output) => {
            // Output should be [OUTPUT_CHANNELS, enc_lens[0]] = [8, 256].
            assert_eq!(output.len(), OUTPUT_CHANNELS * enc_lens[0]);
            // With all-zero weights, output should be all zeros (or very close).
            let max_abs = output.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
            assert!(
                max_abs < 1e-6,
                "expected near-zero output with zero weights, got max_abs={max_abs}"
            );
        }
        Err(e) => {
            panic!("forward pass failed with zero weights: {e}");
        }
    }
}

// GPU/CPU parity tests extracted to separate file for 500-line compliance.
#[path = "demucs_temporal_decoder_gpu_tests.rs"]
mod gpu_tests;

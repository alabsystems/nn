// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! GPU/CPU parity tests for `DemucsTemporalDecoder` — Part of #1391.

use super::super::builders;
use super::super::*;
use crate::demucs_shared::{DCONV_COMPRESS, DCONV_DEPTH, DCONV_KERNEL};
use crate::test_common::make_cache;

// ---------------------------------------------------------------------------
// Seeded weight generators for GPU parity testing
// ---------------------------------------------------------------------------

/// Generate deterministic non-zero DConv weights for parity testing.
fn seeded_dconv_weights(channels: usize, seed: u32) -> DConvSubLayerWeights {
    let compressed = channels / DCONV_COMPRESS;
    let doubled = channels * 2;
    let make = |len: usize, s: u32| -> Vec<f32> {
        (0..len)
            .map(|i| ((i as f32 + s as f32) * 0.001).sin() * 0.01)
            .collect()
    };
    DConvSubLayerWeights {
        conv_compress_weight: make(compressed * channels * DCONV_KERNEL, seed),
        conv_compress_bias: make(compressed, seed + 1),
        norm_compress_gamma: vec![1.0; compressed],
        norm_compress_beta: make(compressed, seed + 2),
        conv_expand_weight: make(doubled * compressed, seed + 3),
        conv_expand_bias: make(doubled, seed + 4),
        norm_expand_gamma: vec![1.0; doubled],
        norm_expand_beta: make(doubled, seed + 5),
        layer_scale: make(channels, seed + 6),
    }
}

fn seeded_block_weights(block_idx: usize) -> DecoderBlockWeights {
    let encoder_depth = DEPTH - 1 - block_idx;
    let in_ch = channels_at_depth(encoder_depth);
    let out_ch = if encoder_depth == 0 {
        OUTPUT_CHANNELS
    } else {
        channels_at_depth(encoder_depth - 1)
    };
    let seed = (block_idx as u32) * 100;
    let make = |len: usize, s: u32| -> Vec<f32> {
        (0..len)
            .map(|i| ((i as f32 + s as f32) * 0.001).sin() * 0.01)
            .collect()
    };

    DecoderBlockWeights {
        rewrite_weight: make(in_ch * 2 * in_ch * REWRITE_KERNEL, seed),
        rewrite_bias: make(in_ch * 2, seed + 10),
        dconv: (0..DCONV_DEPTH)
            .map(|d| seeded_dconv_weights(in_ch, seed + 20 + d as u32 * 10))
            .collect(),
        conv_tr_weight: make(in_ch * out_ch * KERNEL_SIZE, seed + 50),
        conv_tr_bias: make(out_ch, seed + 60),
    }
}

fn seeded_decoder_weights() -> DemucsTemporalDecoderWeights {
    DemucsTemporalDecoderWeights {
        blocks: (0..DEPTH).map(seeded_block_weights).collect(),
    }
}

/// Compute encoder input temporal lengths for a given initial time T.
fn compute_encoder_lengths(initial_t: usize) -> Vec<usize> {
    let mut lengths = Vec::with_capacity(DEPTH);
    let mut t = initial_t;
    for _ in 0..DEPTH {
        lengths.push(t);
        t = builders::conv1d_output_len(t, KERNEL_SIZE, STRIDE, KERNEL_SIZE / 4).unwrap();
    }
    lengths
}

fn zero_decoder_weights() -> DemucsTemporalDecoderWeights {
    super::zero_decoder_weights()
}

// ---------------------------------------------------------------------------
// GPU/CPU parity tests
// ---------------------------------------------------------------------------

#[test]
fn test_forward_gpu_zero_weights_parity() {
    let cache = match make_cache() {
        Some(c) => c,
        None => return,
    };
    let weights = zero_decoder_weights();
    let enc_lens = compute_encoder_lengths(256);
    let decoder = DemucsTemporalDecoder::new(weights, &enc_lens).unwrap();

    let bottleneck_len = decoder.block_in_ch[0] * decoder.block_t_in[0];
    let bottleneck = vec![0.0f32; bottleneck_len];
    let skips: Vec<Vec<f32>> = (0..DEPTH)
        .map(|d| vec![0.0f32; channels_at_depth(d) * enc_lens[d]])
        .collect();

    let cpu_out = decoder.forward(&cache, &bottleneck, &skips).unwrap();
    let gpu_out = decoder.forward_gpu(&cache, &bottleneck, &skips).unwrap();

    assert_eq!(cpu_out.len(), gpu_out.len());
    let max_err = cpu_out
        .iter()
        .zip(gpu_out.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert!(
        max_err < 1e-5,
        "GPU/CPU parity failed with zero weights: max_err={max_err}"
    );
}

#[test]
fn test_forward_gpu_seeded_weights_parity() {
    let cache = match make_cache() {
        Some(c) => c,
        None => return,
    };
    let weights = seeded_decoder_weights();
    let enc_lens = compute_encoder_lengths(256);
    let decoder = DemucsTemporalDecoder::new(weights, &enc_lens).unwrap();

    let bottleneck_len = decoder.block_in_ch[0] * decoder.block_t_in[0];
    let bottleneck: Vec<f32> = (0..bottleneck_len)
        .map(|i| (i as f32 * 0.001).sin() * 0.1)
        .collect();
    let skips: Vec<Vec<f32>> = (0..DEPTH)
        .map(|d| {
            let len = channels_at_depth(d) * enc_lens[d];
            (0..len).map(|i| (i as f32 * 0.002).cos() * 0.05).collect()
        })
        .collect();

    let cpu_out = decoder.forward(&cache, &bottleneck, &skips).unwrap();
    let gpu_out = decoder.forward_gpu(&cache, &bottleneck, &skips).unwrap();

    assert_eq!(cpu_out.len(), gpu_out.len());
    let max_err = cpu_out
        .iter()
        .zip(gpu_out.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert!(
        max_err < 1e-4,
        "GPU/CPU parity failed with seeded weights: max_err={max_err}"
    );
}

#[test]
fn test_forward_gpu_rejects_nan_input() {
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

    let result = decoder.forward_gpu(&cache, &bottleneck, &skips);
    assert!(result.is_err());
    assert!(matches!(
        result.unwrap_err(),
        DemucsTemporalDecoderError::NonFiniteInput { .. }
    ));
}

#[test]
fn test_forward_gpu_persistent_weights_reuse() {
    let cache = match make_cache() {
        Some(c) => c,
        None => return,
    };
    let weights = seeded_decoder_weights();
    let enc_lens = compute_encoder_lengths(256);
    let decoder = DemucsTemporalDecoder::new(weights, &enc_lens).unwrap();

    let bottleneck_len = decoder.block_in_ch[0] * decoder.block_t_in[0];
    let bottleneck: Vec<f32> = (0..bottleneck_len)
        .map(|i| (i as f32 * 0.001).sin() * 0.1)
        .collect();
    let skips: Vec<Vec<f32>> = (0..DEPTH)
        .map(|d| {
            let len = channels_at_depth(d) * enc_lens[d];
            (0..len).map(|i| (i as f32 * 0.002).cos() * 0.05).collect()
        })
        .collect();

    // Run forward_gpu twice — second call should reuse cached GPU weight buffers.
    let out1 = decoder.forward_gpu(&cache, &bottleneck, &skips).unwrap();
    let out2 = decoder.forward_gpu(&cache, &bottleneck, &skips).unwrap();

    assert_eq!(out1.len(), out2.len());
    let max_err = out1
        .iter()
        .zip(out2.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert!(
        max_err == 0.0,
        "second forward_gpu call produced different output: max_err={max_err}"
    );
}

/// Parity: forward_gpu() under NanCheckPolicy::Skip (scoped path) must
/// produce the same results as forward_gpu() under the default policy
/// (unscoped path).
///
/// W4's commit 07dda965 added GpuScope batching that activates only when
/// NanCheckPolicy::Skip is set. This test verifies the scoped code path
/// produces identical output, exercising forward_gpu_scoped() which was
/// previously untested.
#[test]
fn test_forward_gpu_scoped_parity() {
    use nn_core::layers::{with_nan_check_policy, NanCheckPolicy};

    let cache = match make_cache() {
        Some(c) => c,
        None => return,
    };
    let weights = seeded_decoder_weights();
    let enc_lens = compute_encoder_lengths(256);
    let decoder = DemucsTemporalDecoder::new(weights, &enc_lens).unwrap();

    let bottleneck_len = decoder.block_in_ch[0] * decoder.block_t_in[0];
    let bottleneck: Vec<f32> = (0..bottleneck_len)
        .map(|i| (i as f32 * 0.001).sin() * 0.1)
        .collect();
    let skips: Vec<Vec<f32>> = (0..DEPTH)
        .map(|d| {
            let len = channels_at_depth(d) * enc_lens[d];
            (0..len).map(|i| (i as f32 * 0.002).cos() * 0.05).collect()
        })
        .collect();

    // Unscoped path (default NanCheckPolicy).
    let unscoped_out = decoder
        .forward_gpu(&cache, &bottleneck, &skips)
        .expect("forward_gpu (unscoped)");

    // Scoped path: wrap in NanCheckPolicy::Skip to trigger with_gpu_scope.
    let scoped_out = with_nan_check_policy(NanCheckPolicy::Skip, || {
        decoder.forward_gpu(&cache, &bottleneck, &skips)
    })
    .expect("forward_gpu (scoped)");

    assert_eq!(
        unscoped_out.len(),
        scoped_out.len(),
        "output length mismatch"
    );
    let max_err = unscoped_out
        .iter()
        .zip(scoped_out.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert!(
        max_err < 1e-4,
        "scoped/unscoped paths diverge: max_err={max_err}"
    );
}

/// Verify `forward_gpu_with_skips` produces identical output to `forward_gpu`.
///
/// Creates GPU skip buffers from CPU skip data (simulating what the encoder's
/// `forward_gpu()` produces), passes them to `forward_gpu_with_skips()`, and
/// compares against `forward_gpu()` which uses CPU-uploaded skips.
#[test]
fn test_forward_gpu_with_skips_parity() {
    let cache = match make_cache() {
        Some(c) => c,
        None => return,
    };
    let weights = seeded_decoder_weights();
    let enc_lens = compute_encoder_lengths(256);
    let decoder = DemucsTemporalDecoder::new(weights, &enc_lens).expect("valid decoder");

    let bottleneck_len = decoder.block_in_ch[0] * decoder.block_t_in[0];
    let bottleneck: Vec<f32> = (0..bottleneck_len)
        .map(|i| (i as f32 * 0.001).sin() * 0.1)
        .collect();
    let skips: Vec<Vec<f32>> = (0..DEPTH)
        .map(|d| {
            let len = channels_at_depth(d) * enc_lens[d];
            (0..len).map(|i| (i as f32 * 0.002).cos() * 0.05).collect()
        })
        .collect();

    // Upload skips to GPU buffers (mimicking encoder forward_gpu output).
    let ctx = cache.context();
    let skips_gpu: Vec<MetalBuffer> = skips
        .iter()
        .map(|skip_data| {
            <f32 as MetalElement>::create_buffer(ctx, skip_data).expect("GPU buffer upload")
        })
        .collect();

    // forward_gpu: uses CPU-uploaded skips.
    let gpu_out = decoder
        .forward_gpu(&cache, &bottleneck, &skips)
        .expect("forward_gpu");
    // forward_gpu_with_skips: uses GPU-resident skips.
    let gpu_skip_out = decoder
        .forward_gpu_with_skips(&cache, &bottleneck, &skips, &skips_gpu)
        .expect("forward_gpu_with_skips");

    assert_eq!(gpu_out.len(), gpu_skip_out.len(), "output length mismatch");
    let max_err = gpu_out
        .iter()
        .zip(gpu_skip_out.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert!(
        max_err == 0.0,
        "forward_gpu_with_skips differs from forward_gpu: max_err={max_err}"
    );
}

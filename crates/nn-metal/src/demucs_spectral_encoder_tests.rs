// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Tests for the Demucs spectral encoder.
//!
//! Part of #831 — spectral encoder.

use super::*;

use crate::test_common::make_cache;
use nn_models::demucs_spectral_weights::{
    SpectralEncDConvSubLayerWeights, SpectralEncoderBlockWeights,
};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Create synthetic weights for a single DConv sub-layer.
fn make_dconv_weights(channels: usize) -> SpectralEncDConvSubLayerWeights {
    let compressed = channels / crate::demucs_shared::DCONV_COMPRESS; // DCONV_COMPRESS = 4
    SpectralEncDConvSubLayerWeights {
        conv_compress_weight: vec![0.01; compressed * channels * 3],
        conv_compress_bias: vec![0.0; compressed],
        norm_compress_gamma: vec![1.0; compressed],
        norm_compress_beta: vec![0.0; compressed],
        conv_expand_weight: vec![0.01; channels * 2 * compressed],
        conv_expand_bias: vec![0.0; channels * 2],
        norm_expand_gamma: vec![1.0; channels * 2],
        norm_expand_beta: vec![0.0; channels * 2],
        layer_scale: vec![0.01; channels],
    }
}

/// Create synthetic weights for a single encoder block.
fn make_block_weights(block_idx: usize) -> SpectralEncoderBlockWeights {
    let in_ch = if block_idx == 0 {
        INPUT_CHANNELS
    } else {
        channels_at_depth(block_idx - 1)
    };
    let out_ch = channels_at_depth(block_idx);

    SpectralEncoderBlockWeights {
        conv_weight: vec![0.01; out_ch * in_ch * KERNEL_SIZE],
        conv_bias: vec![0.0; out_ch],
        dconv: vec![make_dconv_weights(out_ch), make_dconv_weights(out_ch)],
        rewrite_weight: vec![0.01; out_ch * 2 * out_ch],
        rewrite_bias: vec![0.0; out_ch * 2],
    }
}

/// Create synthetic weights for the full encoder.
fn make_encoder_weights(with_freq_emb: bool) -> DemucsSpectralEncoderWeights {
    DemucsSpectralEncoderWeights {
        blocks: (0..DEPTH).map(make_block_weights).collect(),
        freq_emb_weight: if with_freq_emb {
            Some(vec![0.01; FREQ_EMB_FEATURES * FREQ_EMB_DIM])
        } else {
            None
        },
    }
}

// ---------------------------------------------------------------------------
// Construction tests
// ---------------------------------------------------------------------------

#[test]
fn test_constructor_validates_dimensions() {
    // F=2048, T=10 should succeed.
    let weights = make_encoder_weights(false);
    let enc = DemucsSpectralEncoder::new(weights, 2048, 10);
    assert!(enc.is_ok(), "constructor should succeed: {enc:?}");

    let enc = enc.unwrap();
    assert_eq!(enc.block_sub_defs.len(), DEPTH);
    assert_eq!(enc.block_out_ch, vec![48, 96, 192, 384]);
    assert_eq!(enc.block_f_in, vec![2048, 512, 128, 32]);
    assert_eq!(enc.block_f_out, vec![512, 128, 32, 8]);
}

#[test]
fn test_constructor_wrong_block_count() {
    let mut weights = make_encoder_weights(false);
    weights.blocks.pop();
    let err = DemucsSpectralEncoder::new(weights, 2048, 10);
    assert!(err.is_err());
}

#[test]
fn test_constructor_wrong_conv_weight_size() {
    let mut weights = make_encoder_weights(false);
    weights.blocks[0].conv_weight = vec![0.0; 10]; // wrong size
    let err = DemucsSpectralEncoder::new(weights, 2048, 10);
    assert!(err.is_err());
}

#[test]
fn test_constructor_with_freq_embedding() {
    let weights = make_encoder_weights(true);
    let enc = DemucsSpectralEncoder::new(weights, 2048, 10);
    assert!(enc.is_ok());
    assert!(enc.unwrap().freq_emb_weight.is_some());
}

#[test]
fn test_forward_input_validation() {
    let Some(cache) = make_cache() else { return };
    let weights = make_encoder_weights(false);
    let enc = DemucsSpectralEncoder::new(weights, 2048, 10).unwrap();

    // Wrong input length should fail.
    let bad_input = vec![0.0f32; 100];
    let result = enc.forward(&cache, &bad_input);
    assert!(result.is_err());
}

#[test]
fn test_forward_nonfinite_detection() {
    let Some(cache) = make_cache() else { return };
    let weights = make_encoder_weights(false);
    let enc = DemucsSpectralEncoder::new(weights, 512, 4).unwrap();

    let mut input = vec![0.5f32; INPUT_CHANNELS * 512 * 4];
    input[42] = f32::NAN;
    let result = enc.forward(&cache, &input);
    assert!(result.is_err());
    let err_msg = format!("{}", result.unwrap_err());
    assert!(
        err_msg.contains("non-finite"),
        "expected non-finite error, got: {err_msg}"
    );
}

#[test]
fn test_freq_emb_application() {
    // Test that freq embedding modifies data when present.
    let weights = make_encoder_weights(true);
    let enc = DemucsSpectralEncoder::new(weights, 2048, 10).unwrap();

    let channels = 48;
    let freq_out = 512;
    let time_len = 10;
    let data = vec![0.0f32; channels * freq_out * time_len];

    let result = enc
        .apply_freq_emb(&data, channels, freq_out, time_len)
        .unwrap();

    // With non-zero embedding weights and FREQ_EMB_SCALE=2.0, result should differ.
    let sum: f32 = result.iter().sum();
    assert!(sum.abs() > 0.0, "freq embedding should modify data");
}

#[test]
fn test_freq_emb_skipped_when_absent() {
    let weights = make_encoder_weights(false);
    let enc = DemucsSpectralEncoder::new(weights, 2048, 10).unwrap();

    let channels = 48;
    let freq_out = 512;
    let time_len = 10;
    let data = vec![1.0f32; channels * freq_out * time_len];

    let result = enc
        .apply_freq_emb(&data, channels, freq_out, time_len)
        .unwrap();
    assert_eq!(result, data, "without freq emb, data should be unchanged");
}

// ---------------------------------------------------------------------------
// Forward pass shape tests (requires Metal dispatch)
// ---------------------------------------------------------------------------

#[test]
fn test_forward_full_encoder_shapes() {
    let Some(cache) = make_cache() else { return };

    // Use f=512, t=4 for tractable dispatch count.
    let f_in = 512;
    let t = 4;
    let weights = make_encoder_weights(false);
    let enc = DemucsSpectralEncoder::new(weights, f_in, t);
    assert!(enc.is_ok(), "constructor with f=512, t=4: {enc:?}");
    let enc = enc.unwrap();

    let input = vec![0.1f32; INPUT_CHANNELS * f_in * t];
    let output = enc.forward(&cache, &input);
    assert!(output.is_ok(), "forward pass failed: {output:?}");

    let out = output.unwrap();
    assert_eq!(
        out.skips.len(),
        DEPTH,
        "should have {DEPTH} skip connections"
    );
    assert_eq!(out.freq_dims.len(), DEPTH, "should have {DEPTH} freq dims");
    assert_eq!(out.time_dim, t);

    // Verify skip shapes match expected channel × freq × time.
    for (d, skip) in out.skips.iter().enumerate() {
        let expected_ch = channels_at_depth(d);
        let expected_f = out.freq_dims[d];
        let expected_len = expected_ch * expected_f * t;
        assert_eq!(
            skip.len(),
            expected_len,
            "skip[{d}] length mismatch: expected {expected_ch}×{expected_f}×{t} = {expected_len}, got {}",
            skip.len()
        );
    }

    // Bottleneck should equal last skip.
    assert_eq!(out.bottleneck.len(), out.skips[DEPTH - 1].len());
}

// ---------------------------------------------------------------------------
// Dispatch overhead benchmark (#832)
// ---------------------------------------------------------------------------

/// Benchmark spectral encoder dispatch overhead at multiple scales.
///
/// Measures wall-clock time for the full forward pass and estimates
/// the number of GPU round-trips. Results are printed to stdout for
/// inclusion in #832 performance analysis.
///
/// Part of #832 — spectral dispatch performance characterization.
#[cfg(feature = "bench")]
#[test]
fn bench_spectral_encoder_dispatch_overhead() {
    let cache = match make_cache() {
        Some(c) => c,
        None => {
            eprintln!("Metal not available, skipping benchmark");
            return;
        }
    };

    // Test at multiple (f, t) scales.
    // Scale 1: small test dimensions (f=256, t=4) — minimum viable for 4 depth blocks
    //          (f=64 panics: depth 3 gets f_in=1, padded=5 < kernel_size=8)
    // Scale 2: medium test dimensions (f=512, t=4) — used in existing tests
    // Scale 3: production-adjacent (f=2048, t=8) — closer to real HTDemucs
    let scales: &[(usize, usize, &str)] = &[
        (256, 4, "small"),
        (512, 4, "medium"),
        (2048, 8, "production-adjacent"),
    ];

    for &(f_in, t, label) in scales {
        let weights = make_encoder_weights(false);
        let enc = match DemucsSpectralEncoder::new(weights, f_in, t) {
            Ok(e) => e,
            Err(e) => {
                eprintln!("[{label}] f={f_in}, t={t}: constructor failed: {e}");
                continue;
            }
        };

        let input = vec![0.1f32; INPUT_CHANNELS * f_in * t];

        // Warm-up run (populates Metal pipeline caches).
        let _ = enc.forward(&cache, &input);

        // Timed run.
        let start = std::time::Instant::now();
        let result = enc.forward(&cache, &input);
        let elapsed = start.elapsed();

        // Count expected GPU round-trips per block:
        // Sub-def 1 (Conv+GELU per time step): T dispatches
        // Sub-def 2 (DConv per freq bin): F_out dispatches
        // Sub-def 3 (Rewrite+GLU per time step): T dispatches
        // Total per block: 2*T + F_out
        let mut total_dispatches = 0usize;
        for d in 0..DEPTH {
            let f_out = enc.block_f_out[d];
            total_dispatches += 2 * t + f_out;
        }

        match result {
            Ok(_) => {
                let per_dispatch_us = elapsed.as_micros() as f64 / total_dispatches as f64;
                eprintln!(
                    "[{label}] f={f_in}, t={t}: {elapsed:?} wall, \
                     {total_dispatches} GPU dispatches, \
                     {per_dispatch_us:.1}us/dispatch"
                );
            }
            Err(e) => {
                eprintln!("[{label}] f={f_in}, t={t}: forward failed: {e}");
            }
        }
    }
}

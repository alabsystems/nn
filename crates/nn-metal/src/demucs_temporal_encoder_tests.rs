// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Unit tests for the Demucs temporal encoder.
//!
//! Tests verify weight validation, block building, and forward pass logic.
//! GPU dispatch tests are in `tests/integration_smoke_demucs_encoder.rs`.
//!
//! Part of #779 — Phase E.

use super::*;
use crate::demucs_shared::GROUP_NORM_EPS;
use crate::demucs_test_common::{make_encoder_block_weights, make_encoder_weights, DEPTH};
use crate::test_common::make_cache;

// ---------------------------------------------------------------------------
// Weight validation tests
// ---------------------------------------------------------------------------

#[test]
fn test_encoder_weight_validation_accepts_valid() {
    let weights = make_encoder_weights();
    let result = DemucsTemporalEncoder::new(weights, 256);
    assert!(
        result.is_ok(),
        "valid weights should be accepted: {result:?}"
    );
}

#[test]
fn test_encoder_weight_validation_wrong_block_count() {
    let weights = DemucsTemporalEncoderWeights { blocks: vec![] };
    let err = DemucsTemporalEncoder::new(weights, 256).unwrap_err();
    assert!(
        matches!(err, DemucsTemporalEncoderError::WeightSize { .. }),
        "wrong block count: {err}"
    );
}

#[test]
fn test_encoder_weight_validation_wrong_conv_weight_size() {
    let mut weights = make_encoder_weights();
    weights.blocks[0].conv_weight = vec![0.0; 10]; // wrong size
    let err = DemucsTemporalEncoder::new(weights, 256).unwrap_err();
    assert!(
        matches!(err, DemucsTemporalEncoderError::WeightSize { .. }),
        "wrong conv weight: {err}"
    );
}

#[test]
fn test_encoder_weight_validation_wrong_dconv_count() {
    let mut weights = make_encoder_weights();
    weights.blocks[0].dconv.pop(); // remove one sub-layer
    let err = DemucsTemporalEncoder::new(weights, 256).unwrap_err();
    assert!(
        matches!(err, DemucsTemporalEncoderError::WeightSize { .. }),
        "wrong dconv count: {err}"
    );
}

// ---------------------------------------------------------------------------
// Block builder tests
// ---------------------------------------------------------------------------

#[test]
fn test_encoder_block_def_builds() {
    // Block 0: AUDIO_CHANNELS(2) → channels_at_depth(0)=48, padded_t=256
    let result = builders::build_encoder_block_def(0, 2, 48, 256);
    assert!(result.is_ok(), "block 0 should build: {result:?}");
    let def = result.unwrap();
    assert!(def.nodes.len() > 5, "should have multiple nodes");
}

#[test]
fn test_encoder_block_def_deep_block_builds() {
    // Block 3: channels_at_depth(2)=192 → channels_at_depth(3)=384, padded_t=4
    let result = builders::build_encoder_block_def(3, 192, 384, 4);
    assert!(result.is_ok(), "block 3 should build: {result:?}");
}

#[test]
fn test_encoder_constructor_builds_all_blocks() {
    let weights = make_encoder_weights();
    let encoder = DemucsTemporalEncoder::new(weights, 256).unwrap();
    assert_eq!(encoder.block_defs.len(), DEPTH, "4 block defs");
    assert_eq!(encoder.block_weights.len(), DEPTH, "4 weight maps");
    assert_eq!(encoder.block_out_ch.len(), DEPTH, "4 channel counts");
}

#[test]
fn test_encoder_channel_progression() {
    let weights = make_encoder_weights();
    let encoder = DemucsTemporalEncoder::new(weights, 256).unwrap();
    assert_eq!(encoder.block_out_ch, vec![48, 96, 192, 384]);
}

// ---------------------------------------------------------------------------
// Shape arithmetic tests
// ---------------------------------------------------------------------------

#[test]
fn test_conv1d_out_len_standard() {
    // padded_t=256, kernel=8, stride=4, padding=2
    // (256 + 2*2 - 8) / 4 + 1 = 252/4 + 1 = 64
    let out = builders::conv1d_out_len(256).unwrap();
    assert_eq!(out, 64, "256 → 64 after stride-4 Conv1d");
}

#[test]
fn test_conv1d_out_len_small() {
    // padded_t=8
    // (8 + 4 - 8) / 4 + 1 = 4/4 + 1 = 2
    let out = builders::conv1d_out_len(8).unwrap();
    assert_eq!(out, 2, "8 → 2 after stride-4 Conv1d");
}

#[test]
fn test_stride_padding_is_correct() {
    // t_in=255 (not multiple of 4): padded to 256
    // t_in=256 (multiple of 4): unchanged
    // t_in=257: padded to 260
    assert_eq!(255 + (STRIDE - 255 % STRIDE), 256);
    assert_eq!(256 % STRIDE, 0); // no padding needed
    assert_eq!(257 + (STRIDE - 257 % STRIDE), 260);
}

// ---------------------------------------------------------------------------
// Forward pass dimension tests (no GPU, just validation)
// ---------------------------------------------------------------------------

#[test]
fn test_encoder_debug_format() {
    let weights = make_encoder_weights();
    let encoder = DemucsTemporalEncoder::new(weights, 256).unwrap();
    let debug = format!("{encoder:?}");
    assert!(
        debug.contains("DemucsTemporalEncoder"),
        "debug format: {debug}"
    );
    assert!(debug.contains("block_out_ch"), "debug format: {debug}");
}

#[test]
fn test_encoder_audio_alignment_check_formula() {
    // Audio length must be multiple of AUDIO_CHANNELS (2).
    // 513 is odd → not channel-aligned; 512 is even → OK.
    assert_ne!(513 % AUDIO_CHANNELS, 0, "513 must fail alignment");
    assert_eq!(512 % AUDIO_CHANNELS, 0, "512 must pass alignment");
    assert_eq!(256 % AUDIO_CHANNELS, 0, "256 must pass alignment");
    assert_ne!(1 % AUDIO_CHANNELS, 0, "1 must fail alignment");
}

#[test]
fn test_encoder_weight_map_has_expected_keys() {
    let in_ch = 2;
    let out_ch = 48;
    let block = make_encoder_block_weights(in_ch, out_ch);
    let map = builders::build_encoder_weight_map(&block);

    // Check key categories exist
    assert!(map.contains_key("conv_weight"), "must have conv_weight");
    assert!(map.contains_key("conv_bias"), "must have conv_bias");
    assert!(map.contains_key("rw_weight"), "must have rw_weight");
    assert!(map.contains_key("rw_bias"), "must have rw_bias");
    assert!(map.contains_key("dc0_cw"), "must have dc0_cw");
    assert!(map.contains_key("dc1_cw"), "must have dc1_cw");
    assert!(map.contains_key("dc0_eps"), "must have dc0_eps");
    assert_eq!(map["dc0_eps"], vec![GROUP_NORM_EPS], "eps value");
}

// ---------------------------------------------------------------------------
// Metal GPU dispatch tests
// ---------------------------------------------------------------------------

/// Smoke test: forward() with small weights runs to completion on Metal.
/// Uses stride-aligned t=256 (256 % 4 == 0) so no stride padding is needed.
#[test]
fn test_forward_on_metal() {
    let cache = match make_cache() {
        Some(c) => c,
        None => return, // Skip on non-Metal platforms
    };
    let weights = make_encoder_weights();
    let encoder = DemucsTemporalEncoder::new(weights, 256).unwrap();

    // Audio: [AUDIO_CHANNELS=2, T=256] → flat len = 512
    let audio = vec![0.0f32; AUDIO_CHANNELS * 256];
    let result = encoder.forward(&cache, &audio);
    match result {
        Ok(out) => {
            assert_eq!(out.skips.len(), DEPTH, "4 skip connections");
            assert_eq!(out.input_lengths.len(), DEPTH, "4 input lengths");
            // Bottleneck is the last block output.
            let last_ch = channels_at_depth(DEPTH - 1);
            assert!(
                out.bottleneck.len().is_multiple_of(last_ch),
                "bottleneck len {} not multiple of channels {}",
                out.bottleneck.len(),
                last_ch,
            );
        }
        Err(e) => panic!("forward failed: {e}"),
    }
}

/// Parity: forward() and forward_gpu() must produce bit-exact results.
/// Both use the same Metal kernels with the same data — only the dispatch
/// path differs (CPU round-trip vs buffer-to-buffer).
#[test]
fn test_forward_gpu_parity() {
    let cache = match make_cache() {
        Some(c) => c,
        None => return,
    };
    let weights = make_encoder_weights();
    let encoder = DemucsTemporalEncoder::new(weights, 256).unwrap();

    let audio = vec![0.1f32; AUDIO_CHANNELS * 256];
    let cpu_out = encoder.forward(&cache, &audio).expect("forward (CPU)");
    let gpu_out = encoder
        .forward_gpu(&cache, &audio)
        .expect("forward_gpu (buffer-to-buffer)");

    // Bottleneck must match.
    assert_eq!(
        cpu_out.bottleneck.len(),
        gpu_out.bottleneck.len(),
        "bottleneck length mismatch"
    );
    assert_eq!(
        cpu_out.bottleneck, gpu_out.bottleneck,
        "bottleneck values differ"
    );

    // Skip connections must match.
    assert_eq!(cpu_out.skips.len(), gpu_out.skips.len(), "skip count");
    for (i, (cpu_skip, gpu_skip)) in cpu_out.skips.iter().zip(gpu_out.skips.iter()).enumerate() {
        assert_eq!(
            cpu_skip,
            gpu_skip,
            "skip {i} mismatch: cpu len={}, gpu len={}",
            cpu_skip.len(),
            gpu_skip.len(),
        );
    }

    // Input lengths must match (derived from same initial_t).
    assert_eq!(cpu_out.input_lengths, gpu_out.input_lengths);
}

/// Parity with non-stride-aligned audio (T=255, not multiple of 4).
/// This exercises the stride-padding fallback in forward_gpu().
#[test]
fn test_forward_gpu_parity_non_aligned() {
    let cache = match make_cache() {
        Some(c) => c,
        None => return,
    };
    // T=255: not a multiple of STRIDE=4. Block 0 needs stride padding.
    let weights = make_encoder_weights();
    let encoder = DemucsTemporalEncoder::new(weights, 255).unwrap();

    let audio = vec![0.05f32; AUDIO_CHANNELS * 255];
    let cpu_out = encoder.forward(&cache, &audio).expect("forward (CPU)");
    let gpu_out = encoder
        .forward_gpu(&cache, &audio)
        .expect("forward_gpu (non-aligned)");

    assert_eq!(
        cpu_out.bottleneck, gpu_out.bottleneck,
        "bottleneck values differ for non-aligned T"
    );
    for (i, (cpu_skip, gpu_skip)) in cpu_out.skips.iter().zip(gpu_out.skips.iter()).enumerate() {
        assert_eq!(cpu_skip, gpu_skip, "skip {i} mismatch (non-aligned T)");
    }
    assert_eq!(cpu_out.input_lengths, gpu_out.input_lengths);
}

// ---------------------------------------------------------------------------
// GpuScope scoped path parity (#1985)
// ---------------------------------------------------------------------------

/// Parity: forward_gpu() under NanCheckPolicy::Skip (scoped path) must
/// produce the same results as forward_gpu() under the default policy
/// (unscoped path).
///
/// W4's commit 07dda965 added GpuScope batching that activates only when
/// NanCheckPolicy::Skip is set. This test verifies the scoped code path
/// produces identical output, exercising forward_gpu_scoped() which was
/// previously untested (all existing parity tests use the default policy
/// and thus exercise only forward_gpu_unscoped()).
#[test]
fn test_forward_gpu_scoped_parity() {
    use nn_core::layers::{with_nan_check_policy, NanCheckPolicy};

    let cache = match make_cache() {
        Some(c) => c,
        None => return,
    };
    let weights = make_encoder_weights();
    let encoder = DemucsTemporalEncoder::new(weights, 256).unwrap();

    let audio = vec![0.1f32; AUDIO_CHANNELS * 256];

    // Unscoped path (default NanCheckPolicy).
    let unscoped_out = encoder
        .forward_gpu(&cache, &audio)
        .expect("forward_gpu (unscoped)");

    // Scoped path: wrap in NanCheckPolicy::Skip to trigger with_gpu_scope.
    let scoped_out =
        with_nan_check_policy(NanCheckPolicy::Skip, || encoder.forward_gpu(&cache, &audio))
            .expect("forward_gpu (scoped)");

    // Bottleneck must be bit-exact.
    assert_eq!(
        unscoped_out.bottleneck.len(),
        scoped_out.bottleneck.len(),
        "bottleneck length mismatch"
    );
    assert_eq!(
        unscoped_out.bottleneck, scoped_out.bottleneck,
        "bottleneck values differ between scoped and unscoped paths"
    );

    // Skip connections must be bit-exact.
    assert_eq!(
        unscoped_out.skips.len(),
        scoped_out.skips.len(),
        "skip count"
    );
    for (i, (us, sc)) in unscoped_out
        .skips
        .iter()
        .zip(scoped_out.skips.iter())
        .enumerate()
    {
        assert_eq!(us, sc, "skip {i} mismatch between scoped and unscoped");
    }

    // Input lengths must match.
    assert_eq!(unscoped_out.input_lengths, scoped_out.input_lengths);
}

// ---------------------------------------------------------------------------
// Benchmark: forward() vs forward_gpu() dispatch overhead
// ---------------------------------------------------------------------------

/// Compare wall-clock time of forward() (CPU round-trip) vs forward_gpu()
/// (buffer-to-buffer). Runs at 3 temporal scales. Prints timing to stderr.
///
/// Not a correctness test — parity is verified above. This measures dispatch
/// overhead savings from eliminating GPU↔CPU round-trips between blocks.
#[cfg(feature = "bench")]
#[test]
fn bench_forward_vs_forward_gpu() {
    let cache = match make_cache() {
        Some(c) => c,
        None => {
            eprintln!("Metal not available, skipping benchmark");
            return;
        }
    };

    // Test at multiple scales: T must be multiple of STRIDE^DEPTH (4^4 = 256 min).
    // Scale 1: T=256 (minimum stride-aligned through 4 blocks)
    // Scale 2: T=1024 (typical production chunk)
    // Scale 3: T=4096 (large chunk)
    let scales: &[(usize, &str)] = &[(256, "small"), (1024, "medium"), (4096, "large")];
    let warmup_runs = 1;
    let timed_runs = 1;

    for &(t, label) in scales {
        let weights = make_encoder_weights();
        let encoder = match DemucsTemporalEncoder::new(weights, t) {
            Ok(e) => e,
            Err(e) => {
                eprintln!("[{label}] T={t}: constructor failed: {e}");
                continue;
            }
        };

        let audio = vec![0.1f32; AUDIO_CHANNELS * t];

        // Warm up both paths.
        for _ in 0..warmup_runs {
            let _ = encoder.forward(&cache, &audio);
            let _ = encoder.forward_gpu(&cache, &audio);
        }

        // Time CPU round-trip path.
        let cpu_start = std::time::Instant::now();
        for _ in 0..timed_runs {
            let _ = encoder.forward(&cache, &audio);
        }
        let cpu_elapsed = cpu_start.elapsed();

        // Time GPU buffer-to-buffer path.
        let gpu_start = std::time::Instant::now();
        for _ in 0..timed_runs {
            let _ = encoder.forward_gpu(&cache, &audio);
        }
        let gpu_elapsed = gpu_start.elapsed();

        let cpu_avg_us = cpu_elapsed.as_micros() as f64 / f64::from(timed_runs);
        let gpu_avg_us = gpu_elapsed.as_micros() as f64 / f64::from(timed_runs);
        let speedup = cpu_avg_us / gpu_avg_us;
        let savings_us = cpu_avg_us - gpu_avg_us;

        eprintln!(
            "[{label}] T={t}: cpu={cpu_avg_us:.0}us, gpu={gpu_avg_us:.0}us, \
             speedup={speedup:.2}x, savings={savings_us:.0}us/call \
             ({DEPTH} blocks, {timed_runs} runs)"
        );
    }
}

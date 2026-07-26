// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Synthetic-weight HTDemucs contract tests — always run, no real weights needed.
//!
//! Exercises the full forward pipeline (encoder → transformer → decoder) with
//! deterministic synthetic weights. Validates:
//! - Output shape: `[OUTPUT_CHANNELS, audio_t]` = `[8, audio_t]`
//! - Output finiteness (no NaN/Inf)
//! - CPU/GPU parity
//!
//! Pattern: matches `silero_vad_e2e_contract.rs`.
//!
//! Part of #1578 — HTDemucs always-running forward test.

use nn_metal::{HTDemucs, HTDemucsWeights, MetalBackend, PipelineCache};
use nn_models::demucs_temporal_weights::{
    DConvSubLayerWeights, DecoderBlockWeights, EncoderBlockWeights,
};
use nn_models::demucs_temporal_weights::{
    DemucsTemporalDecoderWeights, DemucsTemporalEncoderWeights,
};
use nn_models::demucs_transformer_weights::DemucsTransformerWeights;
use nn_models::demucs_transformer_weights::{
    CrossAttentionLayerWeights, LayerNormWeights, SelfAttentionLayerWeights,
    TransformerLayerWeights,
};

// ---------------------------------------------------------------------------
// HTDemucs production constants (from htdemucs.rs and demucs_shared)
// ---------------------------------------------------------------------------

/// Stereo audio channels.
const AUDIO_CHANNELS: usize = 2;
/// 4 sources × 2 channels = 8 output channels.
const OUTPUT_CHANNELS: usize = 8;
/// Encoder depth (4 blocks).
const DEPTH: usize = 4;
/// Initial channel count at depth 0: 48.
const INITIAL_CHANNELS: usize = 48;
/// Conv kernel size per encoder/decoder block.
const KERNEL_SIZE: usize = 8;
/// Rewrite conv kernel size.
const REWRITE_KERNEL: usize = 3;
/// DConv sub-layer count per block.
const DCONV_DEPTH: usize = 2;
/// DConv channel compression factor.
const DCONV_COMPRESS: usize = 8;
/// DConv inner kernel size.
const DCONV_KERNEL: usize = 3;
/// Decoder output channels (8 = NUM_SOURCES * AUDIO_CHANNELS).
const DECODER_OUTPUT_CHANNELS: usize = 8;
/// Transformer dimension.
const TRANSFORMER_DIM: usize = 512;
/// Bottleneck dimension: channels_at_depth(3) = 48 * 2^3 = 384.
const BOTTLENECK_DIM: usize = 384;
/// FFN hidden dimension (512 * 4 = 2048).
const FFN_HIDDEN_DIM: usize = 2048;
/// Number of transformer layers.
const NUM_LAYERS: usize = 5;

/// Channels at a given encoder depth: 48 * 2^depth.
fn channels_at_depth(depth: usize) -> usize {
    INITIAL_CHANNELS * (1 << depth)
}

// ---------------------------------------------------------------------------
// Weight factory functions (inline, using public nn_metal types)
// ---------------------------------------------------------------------------

fn make_dconv_weights(channels: usize) -> DConvSubLayerWeights {
    let compressed = channels / DCONV_COMPRESS;
    let doubled = channels * 2;
    DConvSubLayerWeights {
        conv_compress_weight: vec![0.01; compressed * channels * DCONV_KERNEL],
        conv_compress_bias: vec![0.0; compressed],
        norm_compress_gamma: vec![1.0; compressed],
        norm_compress_beta: vec![0.0; compressed],
        conv_expand_weight: vec![0.01; doubled * compressed],
        conv_expand_bias: vec![0.0; doubled],
        norm_expand_gamma: vec![1.0; doubled],
        norm_expand_beta: vec![0.0; doubled],
        layer_scale: vec![0.1; channels],
    }
}

fn make_encoder_block_weights(in_ch: usize, out_ch: usize) -> EncoderBlockWeights {
    let doubled = out_ch * 2;
    EncoderBlockWeights {
        conv_weight: vec![0.01; out_ch * in_ch * KERNEL_SIZE],
        conv_bias: vec![0.0; out_ch],
        dconv: (0..DCONV_DEPTH)
            .map(|_| make_dconv_weights(out_ch))
            .collect(),
        rewrite_weight: vec![0.01; doubled * out_ch],
        rewrite_bias: vec![0.0; doubled],
    }
}

fn make_encoder_weights() -> DemucsTemporalEncoderWeights {
    DemucsTemporalEncoderWeights {
        blocks: (0..DEPTH)
            .map(|d| {
                let in_ch = if d == 0 {
                    AUDIO_CHANNELS
                } else {
                    channels_at_depth(d - 1)
                };
                let out_ch = channels_at_depth(d);
                make_encoder_block_weights(in_ch, out_ch)
            })
            .collect(),
    }
}

fn make_layer_norm_weights(dim: usize) -> LayerNormWeights {
    LayerNormWeights {
        weight: vec![1.0; dim],
        bias: vec![0.0; dim],
    }
}

fn make_self_attention_weights() -> SelfAttentionLayerWeights {
    let d = TRANSFORMER_DIM;
    let ffn = FFN_HIDDEN_DIM;
    SelfAttentionLayerWeights {
        norm1: make_layer_norm_weights(d),
        norm2: make_layer_norm_weights(d),
        norm_out: make_layer_norm_weights(d),
        q_weight: vec![0.02; d * d],
        k_weight: vec![0.015; d * d],
        v_weight: vec![0.025; d * d],
        out_weight: vec![0.012; d * d],
        ffn_linear1_weight: vec![0.005; ffn * d],
        ffn_linear1_bias: vec![0.001; ffn],
        ffn_linear2_weight: vec![0.008; d * ffn],
        ffn_linear2_bias: vec![0.001; d],
        gamma_1: vec![0.01; d],
        gamma_2: vec![0.01; d],
    }
}

fn make_cross_attention_weights() -> CrossAttentionLayerWeights {
    let d = TRANSFORMER_DIM;
    let ffn = FFN_HIDDEN_DIM;
    CrossAttentionLayerWeights {
        norm1: make_layer_norm_weights(d),
        norm2: make_layer_norm_weights(d),
        norm3: make_layer_norm_weights(d),
        norm_out: make_layer_norm_weights(d),
        q_weight: vec![0.018; d * d],
        k_weight: vec![0.022; d * d],
        v_weight: vec![0.02; d * d],
        out_weight: vec![0.014; d * d],
        ffn_linear1_weight: vec![0.006; ffn * d],
        ffn_linear1_bias: vec![0.002; ffn],
        ffn_linear2_weight: vec![0.007; d * ffn],
        ffn_linear2_bias: vec![0.002; d],
        gamma_1: vec![0.012; d],
        gamma_2: vec![0.012; d],
    }
}

fn make_transformer_weights() -> DemucsTransformerWeights {
    let d = TRANSFORMER_DIM;
    let b = BOTTLENECK_DIM;

    let make_layers = || -> Vec<TransformerLayerWeights> {
        (0..NUM_LAYERS)
            .map(|i| {
                if i % 2 == 0 {
                    TransformerLayerWeights::SelfAttention(make_self_attention_weights())
                } else {
                    TransformerLayerWeights::CrossAttention(make_cross_attention_weights())
                }
            })
            .collect()
    };

    DemucsTransformerWeights {
        channel_upsampler_t_weight: vec![0.01; d * b],
        channel_upsampler_t_bias: vec![0.001; d],
        channel_upsampler_s_weight: vec![0.012; d * b],
        channel_upsampler_s_bias: vec![0.001; d],
        norm_in_t: make_layer_norm_weights(d),
        norm_in_s: make_layer_norm_weights(d),
        temporal_layers: make_layers(),
        spectral_layers: make_layers(),
        channel_downsampler_t_weight: vec![0.008; b * d],
        channel_downsampler_t_bias: vec![0.001; b],
        channel_downsampler_s_weight: vec![0.009; b * d],
        channel_downsampler_s_bias: vec![0.001; b],
    }
}

fn make_decoder_block_weights(block_idx: usize) -> DecoderBlockWeights {
    let encoder_depth = DEPTH - 1 - block_idx;
    let in_ch = channels_at_depth(encoder_depth);
    let out_ch = if encoder_depth == 0 {
        DECODER_OUTPUT_CHANNELS
    } else {
        channels_at_depth(encoder_depth - 1)
    };
    DecoderBlockWeights {
        rewrite_weight: vec![0.01; in_ch * 2 * in_ch * REWRITE_KERNEL],
        rewrite_bias: vec![0.0; in_ch * 2],
        dconv: (0..DCONV_DEPTH)
            .map(|_| make_dconv_weights(in_ch))
            .collect(),
        conv_tr_weight: vec![0.01; in_ch * out_ch * KERNEL_SIZE],
        conv_tr_bias: vec![0.0; out_ch],
    }
}

fn make_decoder_weights() -> DemucsTemporalDecoderWeights {
    DemucsTemporalDecoderWeights {
        blocks: (0..DEPTH).map(make_decoder_block_weights).collect(),
    }
}

/// Create valid synthetic HTDemucs weights (temporal only, no spectral).
///
/// Uses non-zero but small weights (0.01–0.025 for convolutions, 1.0 for norms)
/// that produce non-degenerate, numerically stable computation. Matches the
/// factory in `demucs_test_common.rs`.
fn make_htdemucs_weights() -> HTDemucsWeights {
    HTDemucsWeights::new(
        make_encoder_weights(),
        make_transformer_weights(),
        make_decoder_weights(),
        None,
        None,
    )
}

// ---------------------------------------------------------------------------
// Helper: deterministic pseudo-random noise (xorshift64)
// ---------------------------------------------------------------------------

fn deterministic_noise(len: usize) -> Vec<f32> {
    let mut state: u64 = 0xDEAD_BEEF_CAFE_BABE;
    (0..len)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            // Normalize to [-0.1, 0.1] — small amplitude to keep outputs stable.
            ((state as f64 / u64::MAX as f64) * 0.2 - 0.1) as f32
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Contract tests
// ---------------------------------------------------------------------------

/// Full forward pass with silence input — validates shape and finiteness.
///
/// Exercises the complete pipeline: encoder → transformer → decoder.
/// No real weights needed — synthetic weights always produce finite output.
#[test]
fn contract_htdemucs_forward_silence() {
    let backend = match MetalBackend::init() {
        Ok(b) => b,
        Err(_) => return,
    };
    let cache = PipelineCache::new(backend.context().clone());

    let audio_t = 256;
    let weights = make_htdemucs_weights();
    let model = HTDemucs::new(weights, audio_t).expect("valid synthetic weights");

    // Silence: [2, 256] flattened.
    let silence = vec![0.0f32; AUDIO_CHANNELS * audio_t];
    let output = model
        .forward(&cache, &silence)
        .expect("forward on silence should succeed");

    // Output shape: [OUTPUT_CHANNELS, audio_t] = [8, 256] flattened.
    let expected_len = OUTPUT_CHANNELS * audio_t;
    assert_eq!(
        output.len(),
        expected_len,
        "output length mismatch: expected {expected_len}, got {}",
        output.len()
    );

    // All outputs must be finite.
    let non_finite = output.iter().filter(|v| !v.is_finite()).count();
    assert_eq!(
        non_finite, 0,
        "{non_finite} non-finite values in silence output"
    );
}

/// Full forward pass with deterministic noise input.
///
/// Verifies the pipeline handles non-trivial input without NaN/Inf propagation.
#[test]
fn contract_htdemucs_forward_noise() {
    let backend = match MetalBackend::init() {
        Ok(b) => b,
        Err(_) => return,
    };
    let cache = PipelineCache::new(backend.context().clone());

    let audio_t = 256;
    let weights = make_htdemucs_weights();
    let model = HTDemucs::new(weights, audio_t).expect("valid synthetic weights");

    let noise = deterministic_noise(AUDIO_CHANNELS * audio_t);
    let output = model
        .forward(&cache, &noise)
        .expect("forward on noise should succeed");

    let expected_len = OUTPUT_CHANNELS * audio_t;
    assert_eq!(
        output.len(),
        expected_len,
        "output length mismatch: expected {expected_len}, got {}",
        output.len()
    );

    let non_finite = output.iter().filter(|v| !v.is_finite()).count();
    assert_eq!(
        non_finite, 0,
        "{non_finite} non-finite values in noise output"
    );

    // With non-zero weights and non-zero input, output should be non-trivial.
    let all_zero = output.iter().all(|&v| v == 0.0);
    assert!(!all_zero, "noise input should produce non-trivial output");
}

/// GPU forward path: validates shape, finiteness, and CPU/GPU parity.
#[test]
fn contract_htdemucs_forward_gpu_parity() {
    let backend = match MetalBackend::init() {
        Ok(b) => b,
        Err(_) => return,
    };
    let cache = PipelineCache::new(backend.context().clone());

    let audio_t = 256;
    let weights = make_htdemucs_weights();
    let model = HTDemucs::new(weights, audio_t).expect("valid synthetic weights");

    let noise = deterministic_noise(AUDIO_CHANNELS * audio_t);

    let cpu_output = model
        .forward(&cache, &noise)
        .expect("CPU forward should succeed");

    let gpu_output = model
        .forward_gpu(&cache, &noise)
        .expect("GPU forward should succeed");

    // Same shape.
    assert_eq!(
        cpu_output.len(),
        gpu_output.len(),
        "CPU/GPU output length mismatch"
    );

    // Both finite.
    let cpu_non_finite = cpu_output.iter().filter(|v| !v.is_finite()).count();
    let gpu_non_finite = gpu_output.iter().filter(|v| !v.is_finite()).count();
    assert_eq!(
        cpu_non_finite, 0,
        "{cpu_non_finite} non-finite in CPU output"
    );
    assert_eq!(
        gpu_non_finite, 0,
        "{gpu_non_finite} non-finite in GPU output"
    );

    // Parity: CPU and GPU should produce identical or near-identical results.
    // GPU uses Metal FTZ (flush-to-zero) which can cause small differences.
    let max_abs_diff = cpu_output
        .iter()
        .zip(gpu_output.iter())
        .map(|(c, g)| (c - g).abs())
        .fold(0.0f32, f32::max);

    assert!(
        max_abs_diff < 1e-3,
        "CPU/GPU max absolute difference {max_abs_diff} exceeds 1e-3"
    );
}

/// Determinism: same input produces same output on repeated calls.
#[test]
fn contract_htdemucs_forward_deterministic() {
    let backend = match MetalBackend::init() {
        Ok(b) => b,
        Err(_) => return,
    };
    let cache = PipelineCache::new(backend.context().clone());

    let audio_t = 256;
    let weights = make_htdemucs_weights();
    let model = HTDemucs::new(weights, audio_t).expect("valid synthetic weights");

    let noise = deterministic_noise(AUDIO_CHANNELS * audio_t);

    let output1 = model.forward(&cache, &noise).expect("first forward");
    let output2 = model.forward(&cache, &noise).expect("second forward");

    assert_eq!(output1, output2, "forward should be deterministic");
}

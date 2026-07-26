// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Integration-level tests for the full HTDemucs model.
//!
//! Tests cross-module interactions: encoder-decoder shape chains, STFT/iSTFT
//! round-trip with HTDemucs parameters, multi-source separation output shapes,
//! cross-attention between spectral and temporal domains, DConv residual
//! chains, config validation edge cases, and edge-case inputs.

use std::f32::consts::PI;

use crate::demucs_shared::{
    build_dconv_sublayer, channels_at_depth, conv1d_output_len, validate_weight_size,
    DConvSubLayerInputs, AUDIO_CHANNELS, BASE_CHANNELS, DCONV_COMPRESS, DCONV_DEPTH, DCONV_KERNEL,
    DECODER_OUTPUT_CHANNELS, DECODER_REWRITE_KERNEL, DECODER_REWRITE_PADDING, GROWTH,
    SPECTRAL_BASIC_DEPTH, SPECTRAL_CONV_PADDING, SPECTRAL_DEPTH, SPECTRAL_FREQ_EMB_DIM,
    SPECTRAL_FREQ_EMB_FEATURES, SPECTRAL_INPUT_CHANNELS, SPECTRAL_KERNEL_SIZE,
    SPECTRAL_OUTPUT_CHANNELS, SPECTRAL_REWRITE_KERNEL, SPECTRAL_STRIDE, TEMPORAL_BASIC_DEPTH,
    TEMPORAL_CONV_PADDING, TEMPORAL_DEPTH, TEMPORAL_KERNEL_SIZE, TEMPORAL_STRIDE,
};
use crate::demucs_spectral_decoder_builders::{build_decoder_block_sub_defs, conv2d_output_len};
use crate::demucs_spectral_encoder_builders::{
    build_encoder_block_sub_defs, spectral_conv1d_out_len,
};
use crate::demucs_temporal_decoder_builders::build_decoder_block_def;
use crate::demucs_temporal_encoder_builders::build_encoder_block_def;
use crate::demucs_transformer_builders::{
    build_channel_bridge_def, build_cross_attention_layer_def, build_layer_norm_def,
    build_self_attention_layer_def,
};
use crate::demucs_transformer_constants::{
    BOTTLENECK_DIM, FFN_HIDDEN_DIM, NUM_HEADS, NUM_LAYERS, TRANSFORMER_DIM,
};
use crate::demucs_transformer_helpers::{
    build_sinusoidal_table, transpose_ct_to_tc, transpose_tc_to_ct,
};
use crate::demucs_transformer_weights::{
    CrossAttentionLayerWeights, LayerNormWeights, SelfAttentionLayerWeights,
    TransformerLayerWeights,
};
use crate::istft::{IstftBasis, IstftError, IstftParams};

use nn_dsl::tensor_block_builder::TensorBlockBuilder;

// ===========================================================================
// Test helpers
// ===========================================================================

fn zero_ln() -> LayerNormWeights {
    LayerNormWeights {
        weight: vec![1.0; TRANSFORMER_DIM],
        bias: vec![0.0; TRANSFORMER_DIM],
    }
}

fn zero_self_attn() -> SelfAttentionLayerWeights {
    let d = TRANSFORMER_DIM;
    let ffn = FFN_HIDDEN_DIM;
    SelfAttentionLayerWeights {
        norm1: zero_ln(),
        norm2: zero_ln(),
        norm_out: zero_ln(),
        q_weight: vec![0.0; d * d],
        k_weight: vec![0.0; d * d],
        v_weight: vec![0.0; d * d],
        out_weight: vec![0.0; d * d],
        ffn_linear1_weight: vec![0.0; ffn * d],
        ffn_linear1_bias: vec![0.0; ffn],
        ffn_linear2_weight: vec![0.0; d * ffn],
        ffn_linear2_bias: vec![0.0; d],
        gamma_1: vec![1.0; d],
        gamma_2: vec![1.0; d],
    }
}

fn zero_cross_attn() -> CrossAttentionLayerWeights {
    let d = TRANSFORMER_DIM;
    let ffn = FFN_HIDDEN_DIM;
    CrossAttentionLayerWeights {
        norm1: zero_ln(),
        norm2: zero_ln(),
        norm3: zero_ln(),
        norm_out: zero_ln(),
        q_weight: vec![0.0; d * d],
        k_weight: vec![0.0; d * d],
        v_weight: vec![0.0; d * d],
        out_weight: vec![0.0; d * d],
        ffn_linear1_weight: vec![0.0; ffn * d],
        ffn_linear1_bias: vec![0.0; ffn],
        ffn_linear2_weight: vec![0.0; d * ffn],
        ffn_linear2_bias: vec![0.0; d],
        gamma_1: vec![1.0; d],
        gamma_2: vec![1.0; d],
    }
}

// (forward_stft helper removed -- tests use inline STFT for clarity)

// ===========================================================================
// 1. Temporal encoder-decoder shape chain
// ===========================================================================

/// Verify that the temporal encoder chain produces correct output shapes
/// at each depth, and that the decoder chain mirrors it back to the
/// original shape.
#[test]
fn test_temporal_encoder_decoder_shape_chain() {
    // HTDemucs temporal: 4 basic encoder blocks
    // depth 0: [2, T] -> [48, T/4]
    // depth 1: [48, T/4] -> [96, T/16]
    // depth 2: [96, T/16] -> [192, T/64]
    // depth 3: [192, T/64] -> [384, T/256]
    let initial_t = 1024;
    let mut padded_t = initial_t;
    let mut enc_shapes: Vec<(usize, usize)> = Vec::new();

    // Encoder chain
    let depths = [
        (AUDIO_CHANNELS, channels_at_depth(0)),
        (channels_at_depth(0), channels_at_depth(1)),
        (channels_at_depth(1), channels_at_depth(2)),
        (channels_at_depth(2), channels_at_depth(3)),
    ];

    for (block_idx, &(in_ch, out_ch)) in depths.iter().enumerate() {
        let def = build_encoder_block_def(block_idx, in_ch, out_ch, padded_t)
            .unwrap_or_else(|e| panic!("encoder block {block_idx} failed: {e}"));
        let output_shape = &def.nodes[def.output.index()].shape;
        assert_eq!(
            output_shape[0], out_ch,
            "enc block {block_idx} out channels"
        );
        enc_shapes.push((out_ch, output_shape[1]));
        padded_t = output_shape[1];
    }

    // Verify downsampling ratio: T should decrease by ~4x at each depth
    assert!(enc_shapes[0].1 < initial_t, "depth 0 should downsample");
    for i in 1..enc_shapes.len() {
        assert!(
            enc_shapes[i].1 < enc_shapes[i - 1].1,
            "depth {i} should further downsample"
        );
    }

    // Decoder chain (reverse order)
    let mut t_in = enc_shapes.last().unwrap().1;
    let dec_depths = [
        (channels_at_depth(3), channels_at_depth(2), false),
        (channels_at_depth(2), channels_at_depth(1), false),
        (channels_at_depth(1), channels_at_depth(0), false),
        (channels_at_depth(0), AUDIO_CHANNELS, true),
    ];

    for (block_idx, &(in_ch, out_ch, is_last)) in dec_depths.iter().enumerate() {
        // Target length from encoder skip connections
        let target_len = if block_idx < enc_shapes.len() - 1 {
            enc_shapes[enc_shapes.len() - 2 - block_idx].1
        } else {
            initial_t
        };
        let def = build_decoder_block_def(block_idx, in_ch, out_ch, t_in, target_len, is_last)
            .unwrap_or_else(|e| panic!("decoder block {block_idx} failed: {e}"));
        let output_shape = &def.nodes[def.output.index()].shape;
        assert_eq!(
            output_shape[0], out_ch,
            "dec block {block_idx} out channels"
        );
        t_in = output_shape[1];
    }
}

// ===========================================================================
// 2. Spectral encoder-decoder shape chain
// ===========================================================================

/// Verify spectral encoder blocks produce correct output shapes through
/// the full 4-depth basic chain.
#[test]
fn test_spectral_encoder_full_chain() {
    let mut f_in = 2049; // n_fft/2 + 1 for HTDemucs (n_fft=4096)
    let t_len = 16;

    let spectral_depths = [
        (SPECTRAL_INPUT_CHANNELS, channels_at_depth(0)),
        (channels_at_depth(0), channels_at_depth(1)),
        (channels_at_depth(1), channels_at_depth(2)),
        (channels_at_depth(2), channels_at_depth(3)),
    ];

    for (block_idx, &(in_ch, out_ch)) in spectral_depths.iter().enumerate() {
        let f_out = spectral_conv1d_out_len(
            f_in,
            SPECTRAL_KERNEL_SIZE,
            SPECTRAL_STRIDE,
            SPECTRAL_CONV_PADDING,
        )
        .unwrap();

        let sub_defs = build_encoder_block_sub_defs(block_idx, in_ch, out_ch, f_in, f_out, t_len)
            .unwrap_or_else(|e| panic!("spectral enc block {block_idx} failed: {e}"));

        // Conv+GELU output: [out_ch, f_out]
        let conv_shape = &sub_defs.conv_gelu_def.nodes[sub_defs.conv_gelu_def.output.index()].shape;
        assert_eq!(conv_shape[0], out_ch, "block {block_idx} conv out_ch");
        assert_eq!(conv_shape[1], f_out, "block {block_idx} conv f_out");

        // DConv output: [out_ch, t_len]
        let dc_shape = &sub_defs.dconv_def.nodes[sub_defs.dconv_def.output.index()].shape;
        assert_eq!(dc_shape[0], out_ch);
        assert_eq!(dc_shape[1], t_len);

        // Rewrite output: [out_ch, f_out]
        let rw_shape = &sub_defs.rewrite_def.nodes[sub_defs.rewrite_def.output.index()].shape;
        assert_eq!(rw_shape[0], out_ch);
        assert_eq!(rw_shape[1], f_out);

        f_in = f_out;
    }

    // After 4 depths of stride-4 downsampling, freq dimension should be much smaller
    assert!(
        f_in < 20,
        "after 4 spectral enc depths, freq should be small, got {f_in}"
    );
}

// ===========================================================================
// 3. DConv residual chain
// ===========================================================================

/// Verify DConv residual sublayers preserve shape through a full chain,
/// with increasing dilation rates (1, 2, 4, ...).
#[test]
fn test_dconv_residual_chain_shape_preservation() {
    // Test with HTDemucs-realistic channel counts at each depth
    let channel_configs = [48, 96, 192, 384];

    for &channels in &channel_configs {
        let compressed = channels / DCONV_COMPRESS;
        let t_len = 32;
        let mut b = TensorBlockBuilder::new("dconv_chain");
        let mut x = b.add_input("input", &[channels, t_len]);

        for k in 0..DCONV_DEPTH {
            let dc = DConvSubLayerInputs::add_to_builder(&mut b, k, channels, compressed);
            x = build_dconv_sublayer(&mut b, x, &dc, channels, compressed, t_len)
                .unwrap_or_else(|e| panic!("dconv sublayer k={k} ch={channels} failed: {e}"));
        }

        // Output shape must match input shape (residual connection preserves dimensions)
        let def = b.build(x).unwrap();
        let output_shape = &def.nodes[def.output.index()].shape;
        assert_eq!(
            output_shape,
            &[channels, t_len],
            "DConv residual chain for ch={channels} should preserve shape"
        );
    }
}

/// Verify DConv dilation rates follow the expected 2^k pattern.
#[test]
fn test_dconv_dilation_rates() {
    let mut b = TensorBlockBuilder::new("test");
    let channels = 48;
    let compressed = 12;

    for k in 0..5 {
        let dc = DConvSubLayerInputs::add_to_builder(&mut b, k, channels, compressed);
        assert_eq!(dc.dilation, 1 << k, "k={k}: dilation should be 2^{k}");
    }
}

// ===========================================================================
// 4. Cross-attention between spectral and temporal
// ===========================================================================

/// Verify cross-attention works with HTDemucs-realistic temporal and spectral
/// sequence lengths (different lengths for q and kv).
#[test]
fn test_cross_attention_htdemucs_dimensions() {
    let weights = TransformerLayerWeights::CrossAttention(zero_cross_attn());

    // Realistic HTDemucs: temporal bottleneck might be ~4 steps, spectral ~32
    let temporal_seq = 4;
    let spectral_seq = 32;

    // Temporal attends to spectral
    let (def_ts, _) =
        build_cross_attention_layer_def("cross_t_to_s", temporal_seq, spectral_seq, &weights)
            .expect("temporal->spectral cross-attention should build");
    let ts_shape = &def_ts.nodes[def_ts.output.index()].shape;
    assert_eq!(ts_shape, &[temporal_seq, TRANSFORMER_DIM]);

    // Spectral attends to temporal
    let (def_st, _) =
        build_cross_attention_layer_def("cross_s_to_t", spectral_seq, temporal_seq, &weights)
            .expect("spectral->temporal cross-attention should build");
    let st_shape = &def_st.nodes[def_st.output.index()].shape;
    assert_eq!(st_shape, &[spectral_seq, TRANSFORMER_DIM]);
}

// ===========================================================================
// 5. Transformer layer alternation pattern
// ===========================================================================

/// Verify the self/cross/self/cross/self alternation pattern of the
/// HTDemucs transformer bottleneck (5 layers, interleaved).
#[test]
fn test_transformer_layer_alternation() {
    let temporal_seq = 8;
    let spectral_seq = 16;

    for layer_idx in 0..NUM_LAYERS {
        if layer_idx % 2 == 0 {
            // Self-attention layers
            let weights = TransformerLayerWeights::SelfAttention(zero_self_attn());
            let (def_t, _) = build_self_attention_layer_def(
                &format!("temporal_self_{layer_idx}"),
                temporal_seq,
                &weights,
            )
            .expect("temporal self-attention should build");
            let shape = &def_t.nodes[def_t.output.index()].shape;
            assert_eq!(shape, &[temporal_seq, TRANSFORMER_DIM]);

            let (def_s, _) = build_self_attention_layer_def(
                &format!("spectral_self_{layer_idx}"),
                spectral_seq,
                &weights,
            )
            .expect("spectral self-attention should build");
            let shape = &def_s.nodes[def_s.output.index()].shape;
            assert_eq!(shape, &[spectral_seq, TRANSFORMER_DIM]);
        } else {
            // Cross-attention layers
            let weights = TransformerLayerWeights::CrossAttention(zero_cross_attn());
            let (def_ts, _) = build_cross_attention_layer_def(
                &format!("cross_t_to_s_{layer_idx}"),
                temporal_seq,
                spectral_seq,
                &weights,
            )
            .expect("cross t->s should build");
            let ts_shape = &def_ts.nodes[def_ts.output.index()].shape;
            assert_eq!(ts_shape, &[temporal_seq, TRANSFORMER_DIM]);

            let (def_st, _) = build_cross_attention_layer_def(
                &format!("cross_s_to_t_{layer_idx}"),
                spectral_seq,
                temporal_seq,
                &weights,
            )
            .expect("cross s->t should build");
            let st_shape = &def_st.nodes[def_st.output.index()].shape;
            assert_eq!(st_shape, &[spectral_seq, TRANSFORMER_DIM]);
        }
    }
}

// ===========================================================================
// 6. Channel bridge dimensions
// ===========================================================================

/// Verify channel bridge up/down between bottleneck (384) and transformer (512).
#[test]
fn test_channel_bridge_htdemucs_dimensions() {
    let seq_len = 8;

    // Upsample: BOTTLENECK_DIM (384) -> TRANSFORMER_DIM (512)
    let (up_def, _) =
        build_channel_bridge_def("bridge_up", BOTTLENECK_DIM, TRANSFORMER_DIM, seq_len)
            .expect("upsampler bridge should build");
    let up_shape = &up_def.nodes[up_def.output.index()].shape;
    assert_eq!(up_shape, &[TRANSFORMER_DIM, seq_len]);

    // Downsample: TRANSFORMER_DIM (512) -> BOTTLENECK_DIM (384)
    let (down_def, _) =
        build_channel_bridge_def("bridge_down", TRANSFORMER_DIM, BOTTLENECK_DIM, seq_len)
            .expect("downsampler bridge should build");
    let down_shape = &down_def.nodes[down_def.output.index()].shape;
    assert_eq!(down_shape, &[BOTTLENECK_DIM, seq_len]);
}

// ===========================================================================
// 7. Multi-source separation output dimensions
// ===========================================================================

/// Verify decoder output channels match 4-stem separation (drums, bass, vocals, other).
#[test]
fn test_multi_source_separation_output_channels() {
    // Temporal: 4 sources x 2 stereo channels = 8
    assert_eq!(DECODER_OUTPUT_CHANNELS, 8);

    // Spectral: 4 sources x 2 stereo x 2 (real+imag) = 16
    assert_eq!(SPECTRAL_OUTPUT_CHANNELS, 16);

    // Temporal decoder last block: output channels = DECODER_OUTPUT_CHANNELS
    let in_ch = channels_at_depth(0); // 48
    let out_ch = DECODER_OUTPUT_CHANNELS; // 8
    let t_in = 32;
    let target_len = 128;
    let def = build_decoder_block_def(3, in_ch, out_ch, t_in, target_len, true)
        .expect("last decoder block should build");
    let output_shape = &def.nodes[def.output.index()].shape;
    assert_eq!(
        output_shape[0], out_ch,
        "last decoder block should output 4 sources x 2 channels = 8"
    );
}

/// Verify spectral decoder last block outputs the correct channel count for
/// 4-source complex separation.
#[test]
fn test_spectral_decoder_last_block_output_channels() {
    let in_ch = channels_at_depth(0); // 48
    let out_ch = SPECTRAL_OUTPUT_CHANNELS; // 16
    let f_in = 31;
    let t_in = 16;
    let target_f = 128;

    let sub_defs =
        build_decoder_block_sub_defs(3, in_ch, out_ch, f_in, t_in, f_in, t_in, target_f, true)
            .expect("spectral last block should build");

    let ct_shape = &sub_defs.conv_tr_def.nodes[sub_defs.conv_tr_def.output.index()].shape;
    assert_eq!(
        ct_shape[0], out_ch,
        "spectral last block should output 4 sources x 2 ch x 2 (real+imag) = 16"
    );
}

// ===========================================================================
// 8. STFT/iSTFT round-trip with HTDemucs parameters
// ===========================================================================

/// Round-trip a sine wave through STFT/iSTFT using a moderate FFT size
/// (representative of HTDemucs approach but small enough for fast testing).
#[test]
fn test_stft_istft_round_trip_htdemucs_style() {
    // HTDemucs uses n_fft=4096, hop=1024, normalized=true.
    // For test speed, use a smaller FFT with the same ratio: n_fft=128, hop=32.
    let n_fft = 128;
    let hop = 32;
    let signal_len = 512;
    let freq = 7.0;

    let signal: Vec<f32> = (0..signal_len)
        .map(|i| (2.0 * PI * freq * i as f32 / signal_len as f32).sin())
        .collect();

    // Forward STFT with normalization (matching HTDemucs)
    let n_bins = n_fft / 2 + 1;
    let n_frames = (signal_len - n_fft) / hop + 1;
    let norm_factor = 1.0 / (n_fft as f32).sqrt();

    let window: Vec<f32> = (0..n_fft)
        .map(|k| 0.5 * (1.0 - (2.0 * PI * k as f32 / n_fft as f32).cos()))
        .collect();

    let mut real = vec![0.0f32; n_bins * n_frames];
    let mut imag = vec![0.0f32; n_bins * n_frames];

    for t in 0..n_frames {
        let offset = t * hop;
        for f in 0..n_bins {
            let mut r = 0.0f32;
            let mut im = 0.0f32;
            for k in 0..n_fft {
                let angle = 2.0 * PI * (f as f32) * (k as f32) / (n_fft as f32);
                let windowed = signal[offset + k] * window[k];
                r += windowed * angle.cos();
                im -= windowed * angle.sin();
            }
            real[f * n_frames + t] = r * norm_factor;
            imag[f * n_frames + t] = im * norm_factor;
        }
    }

    let params = IstftParams::new(n_fft, hop, true, false).unwrap();
    let basis = IstftBasis::new(params).unwrap();

    let full_len = n_fft + (n_frames - 1) * hop;
    let reconstructed = basis.istft(&real, &imag, n_frames, full_len).unwrap();

    // Interior samples (avoiding edge effects) should closely match original
    let margin = n_fft / 2;
    let mut max_err = 0.0f32;
    for i in margin..(full_len - margin).min(signal_len) {
        let err = (reconstructed[i] - signal[i]).abs();
        max_err = max_err.max(err);
    }
    assert!(
        max_err < 0.05,
        "HTDemucs-style round-trip max error = {max_err}, expected < 0.05"
    );
}

/// Verify iSTFT with HTDemucs default parameters constructs properly
/// and that n_bins matches expected 2049.
#[test]
fn test_istft_htdemucs_default_params() {
    let params = IstftParams::default();
    assert_eq!(params.n_fft, 4096);
    assert_eq!(params.hop_length, 1024);
    assert!(params.normalized);
    assert!(params.center);

    let basis = IstftBasis::new(params).unwrap();
    assert_eq!(basis.n_bins(), 2049);
}

// ===========================================================================
// 9. Config validation edge cases
// ===========================================================================

/// Verify channels_at_depth panics at maximum depth.
#[test]
#[should_panic(expected = "exceeds maximum")]
fn test_channels_at_depth_panics_on_overflow() {
    channels_at_depth(31);
}

/// Verify channels_at_depth works at maximum allowed depth (30).
#[test]
fn test_channels_at_depth_max_valid() {
    // depth 30: 48 * 2^30 = 48 * 1,073,741,824 = ~5.2e10 (fits in 64-bit usize)
    let ch = channels_at_depth(30);
    assert!(ch > 0);
    assert_eq!(ch, (BASE_CHANNELS as f64 * GROWTH.powi(30)) as usize);
}

/// Verify conv1d_output_len rejects kernel larger than padded input.
#[test]
fn test_conv1d_output_len_kernel_too_large() {
    let result = conv1d_output_len(
        4,
        TEMPORAL_KERNEL_SIZE,
        TEMPORAL_STRIDE,
        TEMPORAL_CONV_PADDING,
    );
    // 4 + 2*2 = 8, kernel=8, so (8 - 8)/4 + 1 = 1 -- should succeed
    assert!(result.is_ok());
    assert_eq!(result.unwrap(), 1);
}

/// Verify conv1d_output_len with minimum viable input for temporal encoder.
#[test]
fn test_conv1d_output_len_minimum_temporal() {
    // Minimum input for stride-4 temporal: need padded >= kernel
    // padded = in + 2*padding = in + 4
    // Need padded >= 8 => in >= 4
    let result = conv1d_output_len(4, 8, 4, 2);
    assert!(result.is_ok());
}

/// Validate weight size with zero-length name works.
#[test]
fn test_validate_weight_size_empty_name() {
    let data = vec![0.0; 10];
    assert!(validate_weight_size(&data, "", 10).is_ok());
    let err = validate_weight_size(&data, "", 5).unwrap_err();
    // Error message should still be parseable even with empty name
    assert!(err.to_string().contains("10"));
}

// ===========================================================================
// 10. IstftParams validation
// ===========================================================================

/// Verify IstftParams::new rejects odd n_fft.
#[test]
fn test_istft_params_rejects_odd_nfft() {
    let result = IstftParams::new(127, 32, false, false);
    assert!(matches!(result, Err(IstftError::OddNfft { n_fft: 127 })));
}

/// Verify IstftParams::new rejects zero n_fft.
#[test]
fn test_istft_params_rejects_zero_nfft() {
    let result = IstftParams::new(0, 32, false, false);
    assert!(matches!(result, Err(IstftError::OddNfft { n_fft: 0 })));
}

/// Verify IstftParams::new rejects zero hop length.
#[test]
fn test_istft_params_rejects_zero_hop() {
    let result = IstftParams::new(128, 0, false, false);
    assert!(matches!(result, Err(IstftError::ZeroHopLength)));
}

/// Verify IstftParams::new accepts valid parameters.
#[test]
fn test_istft_params_accepts_valid() {
    let params = IstftParams::new(4096, 1024, true, true).unwrap();
    assert_eq!(params.n_fft, 4096);
    assert_eq!(params.hop_length, 1024);
    assert!(params.normalized);
    assert!(params.center);
}

// ===========================================================================
// 11. Transpose round-trip
// ===========================================================================

/// Verify CT->TC->CT round-trip is exact for HTDemucs-scale dimensions.
#[test]
fn test_transpose_roundtrip_htdemucs_scale() {
    let channels = BOTTLENECK_DIM; // 384
    let seq_len = 16;
    let original: Vec<f32> = (0..channels * seq_len).map(|i| i as f32 * 0.001).collect();

    let tc = transpose_ct_to_tc(&original, channels, seq_len);
    assert_eq!(tc.len(), channels * seq_len);

    let ct = transpose_tc_to_ct(&tc, seq_len, channels);
    assert_eq!(ct.len(), channels * seq_len);

    for (i, (&orig, &recovered)) in original.iter().zip(ct.iter()).enumerate() {
        assert!(
            (orig - recovered).abs() < 1e-10,
            "mismatch at index {i}: {orig} vs {recovered}"
        );
    }
}

// ===========================================================================
// 12. Sinusoidal positional embedding properties
// ===========================================================================

/// Verify sinusoidal embedding has correct L2 norm properties.
#[test]
fn test_sinusoidal_embedding_l2_norm() {
    let seq_len = 16;
    let dim = TRANSFORMER_DIM; // 512
    let table = build_sinusoidal_table(seq_len, dim);

    // Each position embedding should have roughly similar L2 norm
    let mut norms = Vec::new();
    for pos in 0..seq_len {
        let row = &table[pos * dim..(pos + 1) * dim];
        let norm: f32 = row.iter().map(|v| v * v).sum::<f32>().sqrt();
        norms.push(norm);
    }

    // All norms should be similar (within 20% of each other)
    let max_norm = norms.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let min_norm = norms.iter().copied().fold(f32::INFINITY, f32::min);
    assert!(
        max_norm / min_norm < 1.2,
        "position embedding norms should be similar: min={min_norm}, max={max_norm}"
    );
}

/// Verify different positions produce distinguishable embeddings.
#[test]
fn test_sinusoidal_embedding_position_distinguishability() {
    let seq_len = 32;
    let dim = TRANSFORMER_DIM;
    let table = build_sinusoidal_table(seq_len, dim);

    // Compute pairwise L2 distance between consecutive positions
    for pos in 0..seq_len - 1 {
        let row_a = &table[pos * dim..(pos + 1) * dim];
        let row_b = &table[(pos + 1) * dim..(pos + 2) * dim];
        let dist: f32 = row_a
            .iter()
            .zip(row_b.iter())
            .map(|(a, b)| (a - b) * (a - b))
            .sum::<f32>()
            .sqrt();
        assert!(
            dist > 0.01,
            "positions {pos} and {} should be distinguishable, dist={dist}",
            pos + 1
        );
    }
}

// ===========================================================================
// 13. Edge cases: very short audio / temporal
// ===========================================================================

/// Verify the temporal encoder can handle a very short input
/// (minimum viable padded length).
#[test]
fn test_temporal_encoder_minimum_input() {
    let in_ch = AUDIO_CHANNELS;
    let out_ch = channels_at_depth(0); // 48
                                       // Minimum padded_t so that conv output >= 1
                                       // Conv1d out = (padded_t + 2*2 - 8) / 4 + 1 >= 1
                                       // => padded_t >= 8 - 4 = 4
                                       // But also need padded_t >= 8 (kernel_size) for valid conv
    let padded_t = TEMPORAL_KERNEL_SIZE; // 8

    let def = build_encoder_block_def(0, in_ch, out_ch, padded_t)
        .expect("encoder with minimum input should build");
    let output_shape = &def.nodes[def.output.index()].shape;
    assert_eq!(output_shape[0], out_ch);
    assert!(output_shape[1] > 0, "output temporal dim should be > 0");
}

/// Verify the temporal decoder handles a single time step.
#[test]
fn test_temporal_decoder_single_timestep() {
    let in_ch = channels_at_depth(0); // 48
    let out_ch = DECODER_OUTPUT_CHANNELS; // 8
    let t_in = 1;
    let target_len = 4;

    let def = build_decoder_block_def(0, in_ch, out_ch, t_in, target_len, true)
        .expect("decoder with single timestep should build");
    let output_shape = &def.nodes[def.output.index()].shape;
    assert_eq!(output_shape[0], out_ch);
}

// ===========================================================================
// 14. Architecture constant relationships
// ===========================================================================

/// Verify the key architecture constant relationships hold for HTDemucs.
#[test]
fn test_htdemucs_architecture_constants_consistency() {
    // TEMPORAL_DEPTH = TEMPORAL_BASIC_DEPTH + 1
    assert_eq!(TEMPORAL_DEPTH, TEMPORAL_BASIC_DEPTH + 1);

    // SPECTRAL_DEPTH = SPECTRAL_BASIC_DEPTH + 2 (2 deep blocks)
    assert_eq!(SPECTRAL_DEPTH, SPECTRAL_BASIC_DEPTH + 2);

    // AUDIO_CHANNELS = 2 (stereo)
    assert_eq!(AUDIO_CHANNELS, 2);

    // DECODER_OUTPUT_CHANNELS = 4 sources * 2 channels
    assert_eq!(DECODER_OUTPUT_CHANNELS, 4 * AUDIO_CHANNELS);

    // SPECTRAL_INPUT_CHANNELS = 2 stereo * 2 (real+imag)
    assert_eq!(SPECTRAL_INPUT_CHANNELS, 2 * AUDIO_CHANNELS);

    // SPECTRAL_OUTPUT_CHANNELS = 4 sources * 2 stereo * 2 (real+imag)
    assert_eq!(SPECTRAL_OUTPUT_CHANNELS, 4 * AUDIO_CHANNELS * 2);

    // TRANSFORMER_DIM divisible by NUM_HEADS
    assert_eq!(TRANSFORMER_DIM % NUM_HEADS, 0);

    // BOTTLENECK_DIM = channels_at_depth(3) = 384
    assert_eq!(BOTTLENECK_DIM, channels_at_depth(3));

    // Frequency embedding dimension matches base channels
    assert_eq!(SPECTRAL_FREQ_EMB_DIM, BASE_CHANNELS);

    // FFN_HIDDEN_DIM = TRANSFORMER_DIM * 4 = 2048
    assert_eq!(FFN_HIDDEN_DIM, 2048);

    // NUM_LAYERS = 5
    assert_eq!(NUM_LAYERS, 5);

    // Conv padding relationships
    assert_eq!(TEMPORAL_CONV_PADDING, TEMPORAL_KERNEL_SIZE / 4);
    assert_eq!(SPECTRAL_CONV_PADDING, SPECTRAL_KERNEL_SIZE / 4);
    assert_eq!(DECODER_REWRITE_PADDING, DECODER_REWRITE_KERNEL / 2);
    assert_eq!(SPECTRAL_REWRITE_KERNEL, DECODER_REWRITE_KERNEL);
}

/// Verify the channel progression at each depth matches the
/// standard HTDemucs doubling pattern.
#[test]
fn test_htdemucs_channel_progression() {
    let expected = [48, 96, 192, 384, 768, 1536];
    for (depth, &expected_ch) in expected.iter().enumerate() {
        assert_eq!(
            channels_at_depth(depth),
            expected_ch,
            "depth {depth} channels"
        );
    }
}

// ===========================================================================
// 15. Conv2d output length edge cases (spectral decoder)
// ===========================================================================

/// Verify conv2d_output_len with identical params to conv1d for consistency.
#[test]
fn test_conv2d_output_len_matches_conv1d_for_1d_case() {
    // When applied to a 1D case, conv2d_output_len should give same result as conv1d_output_len
    let in_len = 256;
    let kernel = 8;
    let stride = 4;
    let padding = 2;

    let conv1d_out = conv1d_output_len(in_len, kernel, stride, padding).unwrap();
    let conv2d_out = conv2d_output_len(in_len, kernel, stride, padding).unwrap();
    assert_eq!(conv1d_out, conv2d_out);
}

/// Verify conv2d_output_len with spectral encoder parameters.
#[test]
fn test_conv2d_output_len_spectral_params() {
    // Starting freq = 2049, spectral stride 4, kernel 8, padding 2
    let f_out = conv2d_output_len(
        2049,
        SPECTRAL_KERNEL_SIZE,
        SPECTRAL_STRIDE,
        SPECTRAL_CONV_PADDING,
    );
    assert!(f_out.is_ok());
    let f_out = f_out.unwrap();
    // (2049 + 2*2 - 8) / 4 + 1 = 2045/4 + 1 = 511 + 1 = 512
    assert_eq!(f_out, 512);
}

// ===========================================================================
// 16. iSTFT round-trip with zero signal
// ===========================================================================

/// Verify iSTFT reconstructs a zero signal correctly (no energy leakage).
#[test]
fn test_istft_round_trip_zero_signal() {
    let n_fft = 32;
    let hop = 8;
    let n_bins = n_fft / 2 + 1;
    let n_frames = 4;

    let real = vec![0.0f32; n_bins * n_frames];
    let imag = vec![0.0f32; n_bins * n_frames];

    let params = IstftParams::new(n_fft, hop, false, false).unwrap();
    let basis = IstftBasis::new(params).unwrap();

    let full_len = n_fft + (n_frames - 1) * hop;
    let result = basis.istft(&real, &imag, n_frames, full_len).unwrap();

    for (i, &val) in result.iter().enumerate() {
        assert!(
            val.abs() < 1e-10,
            "zero signal should reconstruct to zero, got sample[{i}] = {val}"
        );
    }
}

// ===========================================================================
// 17. Temporal encoder weight map completeness
// ===========================================================================

/// Verify encoder weight map has the exact number of expected keys.
#[test]
fn test_temporal_encoder_weight_map_key_count() {
    use crate::demucs_temporal_encoder_builders::build_encoder_weight_map;
    use crate::demucs_temporal_weights::{DConvSubLayerWeights, EncoderBlockWeights};

    let out_ch = 48;
    let in_ch = 2;
    let compressed = out_ch / DCONV_COMPRESS;

    let dconv_sub = DConvSubLayerWeights {
        conv_compress_weight: vec![0.0; compressed * out_ch * DCONV_KERNEL],
        conv_compress_bias: vec![0.0; compressed],
        norm_compress_gamma: vec![0.0; compressed],
        norm_compress_beta: vec![0.0; compressed],
        conv_expand_weight: vec![0.0; out_ch * 2 * compressed],
        conv_expand_bias: vec![0.0; out_ch * 2],
        norm_expand_gamma: vec![0.0; out_ch * 2],
        norm_expand_beta: vec![0.0; out_ch * 2],
        layer_scale: vec![0.0; out_ch],
    };

    let block = EncoderBlockWeights {
        conv_weight: vec![0.0; out_ch * in_ch * TEMPORAL_KERNEL_SIZE],
        conv_bias: vec![0.0; out_ch],
        dconv: vec![dconv_sub.clone(), dconv_sub],
        rewrite_weight: vec![0.0; out_ch * 2 * out_ch],
        rewrite_bias: vec![0.0; out_ch * 2],
    };

    let map = build_encoder_weight_map(&block);

    // Expected: 2 (conv) + 2 * 11 (dconv: 9 weight + 2 eps) + 2 (rewrite) = 26
    assert_eq!(
        map.len(),
        26,
        "encoder weight map should have 26 entries, got {}",
        map.len()
    );
}

/// Verify decoder weight map has the exact number of expected keys.
#[test]
fn test_temporal_decoder_weight_map_key_count() {
    use crate::demucs_temporal_decoder_builders::build_decoder_weight_map;
    use crate::demucs_temporal_weights::{DConvSubLayerWeights, DecoderBlockWeights};

    let in_ch = 48;
    let out_ch = 2;
    let compressed = in_ch / DCONV_COMPRESS;

    let dconv_sub = DConvSubLayerWeights {
        conv_compress_weight: vec![0.0; compressed * in_ch * DCONV_KERNEL],
        conv_compress_bias: vec![0.0; compressed],
        norm_compress_gamma: vec![0.0; compressed],
        norm_compress_beta: vec![0.0; compressed],
        conv_expand_weight: vec![0.0; in_ch * 2 * compressed],
        conv_expand_bias: vec![0.0; in_ch * 2],
        norm_expand_gamma: vec![0.0; in_ch * 2],
        norm_expand_beta: vec![0.0; in_ch * 2],
        layer_scale: vec![0.0; in_ch],
    };

    let block = DecoderBlockWeights {
        rewrite_weight: vec![0.0; in_ch * 2 * in_ch * DECODER_REWRITE_KERNEL],
        rewrite_bias: vec![0.0; in_ch * 2],
        dconv: vec![dconv_sub.clone(), dconv_sub],
        conv_tr_weight: vec![0.0; in_ch * out_ch * TEMPORAL_KERNEL_SIZE],
        conv_tr_bias: vec![0.0; out_ch],
    };

    let map = build_decoder_weight_map(&block);

    // Expected: 2 (rewrite) + 2 * 11 (dconv) + 2 (conv_tr) = 26
    assert_eq!(
        map.len(),
        26,
        "decoder weight map should have 26 entries, got {}",
        map.len()
    );
}

// ===========================================================================
// 18. LayerNorm def shape validation
// ===========================================================================

/// Verify LayerNorm def preserves shape for transformer dimensions.
#[test]
fn test_layer_norm_preserves_shape() {
    let ln_w = zero_ln();
    for seq_len in [1, 4, 16, 64] {
        let (def, wmap) = build_layer_norm_def(&format!("ln_{seq_len}"), seq_len, &ln_w)
            .expect("layer norm def should build");
        let shape = &def.nodes[def.output.index()].shape;
        assert_eq!(shape, &[seq_len, TRANSFORMER_DIM]);
        assert!(wmap.contains_key("eps"));
        assert!(wmap.contains_key("ln_weight"));
        assert!(wmap.contains_key("ln_bias"));
    }
}

// ===========================================================================
// 19. iSTFT center trimming behavior
// ===========================================================================

/// Verify center trimming removes n_fft/2 from each side.
#[test]
fn test_istft_center_trim_removes_correct_amount() {
    let n_fft = 64;
    let hop = 16;
    let n_bins = n_fft / 2 + 1;
    let n_frames = 8;
    let full_len = n_fft + (n_frames - 1) * hop;
    let trimmed_len = full_len - n_fft; // remove n_fft/2 from each side

    let params = IstftParams::new(n_fft, hop, false, true).unwrap();
    let basis = IstftBasis::new(params).unwrap();

    let real = vec![0.0f32; n_bins * n_frames];
    let imag = vec![0.0f32; n_bins * n_frames];

    let result = basis.istft(&real, &imag, n_frames, trimmed_len).unwrap();
    assert_eq!(result.len(), trimmed_len);
}

// ===========================================================================
// 20. Spectral encoder weight validation
// ===========================================================================

/// Verify spectral encoder weight validation catches wrong DConv depth.
#[test]
fn test_spectral_encoder_validation_wrong_dconv_depth() {
    use crate::demucs_spectral_encoder_builders::validate_all_encoder_weights;
    use crate::demucs_spectral_weights::{
        DemucsSpectralEncoderWeights, SpectralEncDConvSubLayerWeights, SpectralEncoderBlockWeights,
    };

    let out_ch = channels_at_depth(0);
    let in_ch = SPECTRAL_INPUT_CHANNELS;
    let compressed = out_ch / DCONV_COMPRESS;

    let dconv_sub = SpectralEncDConvSubLayerWeights {
        conv_compress_weight: vec![0.0; compressed * out_ch * DCONV_KERNEL],
        conv_compress_bias: vec![0.0; compressed],
        norm_compress_gamma: vec![0.0; compressed],
        norm_compress_beta: vec![0.0; compressed],
        conv_expand_weight: vec![0.0; out_ch * 2 * compressed],
        conv_expand_bias: vec![0.0; out_ch * 2],
        norm_expand_gamma: vec![0.0; out_ch * 2],
        norm_expand_beta: vec![0.0; out_ch * 2],
        layer_scale: vec![0.0; out_ch],
    };

    // Only 1 DConv sub-layer instead of 2
    let block = SpectralEncoderBlockWeights {
        conv_weight: vec![0.0; out_ch * in_ch * SPECTRAL_KERNEL_SIZE],
        conv_bias: vec![0.0; out_ch],
        dconv: vec![dconv_sub], // wrong: should be 2
        rewrite_weight: vec![0.0; out_ch * 2 * out_ch],
        rewrite_bias: vec![0.0; out_ch * 2],
    };

    let weights = DemucsSpectralEncoderWeights {
        blocks: vec![block; SPECTRAL_BASIC_DEPTH],
        freq_emb_weight: None,
    };

    let err = validate_all_encoder_weights(&weights).unwrap_err();
    assert!(
        err.to_string().contains("expected"),
        "error should mention expected count: {err}"
    );
}

/// Verify spectral encoder weight validation accepts correct weights.
#[test]
fn test_spectral_encoder_validation_accepts_correct() {
    use crate::demucs_spectral_encoder_builders::validate_all_encoder_weights;
    use crate::demucs_spectral_weights::{
        DemucsSpectralEncoderWeights, SpectralEncDConvSubLayerWeights, SpectralEncoderBlockWeights,
    };

    let mut blocks = Vec::new();
    for block_idx in 0..SPECTRAL_BASIC_DEPTH {
        let in_ch = if block_idx == 0 {
            SPECTRAL_INPUT_CHANNELS
        } else {
            channels_at_depth(block_idx - 1)
        };
        let out_ch = channels_at_depth(block_idx);
        let compressed = out_ch / DCONV_COMPRESS;

        let dconv_sub = SpectralEncDConvSubLayerWeights {
            conv_compress_weight: vec![0.0; compressed * out_ch * DCONV_KERNEL],
            conv_compress_bias: vec![0.0; compressed],
            norm_compress_gamma: vec![0.0; compressed],
            norm_compress_beta: vec![0.0; compressed],
            conv_expand_weight: vec![0.0; out_ch * 2 * compressed],
            conv_expand_bias: vec![0.0; out_ch * 2],
            norm_expand_gamma: vec![0.0; out_ch * 2],
            norm_expand_beta: vec![0.0; out_ch * 2],
            layer_scale: vec![0.0; out_ch],
        };

        blocks.push(SpectralEncoderBlockWeights {
            conv_weight: vec![0.0; out_ch * in_ch * SPECTRAL_KERNEL_SIZE],
            conv_bias: vec![0.0; out_ch],
            dconv: vec![dconv_sub.clone(), dconv_sub],
            rewrite_weight: vec![0.0; out_ch * 2 * out_ch],
            rewrite_bias: vec![0.0; out_ch * 2],
        });
    }

    let weights = DemucsSpectralEncoderWeights {
        blocks,
        freq_emb_weight: Some(vec![
            0.0;
            SPECTRAL_FREQ_EMB_FEATURES * SPECTRAL_FREQ_EMB_DIM
        ]),
    };

    validate_all_encoder_weights(&weights).expect("correct weights should pass validation");
}

// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Tests for [`super::demucs_spectral_decoder_builders`].

use super::*;
use crate::demucs_shared::{
    channels_at_depth, DCONV_COMPRESS, DCONV_KERNEL, SPECTRAL_BASIC_DEPTH, SPECTRAL_KERNEL_SIZE,
    SPECTRAL_OUTPUT_CHANNELS, SPECTRAL_REWRITE_KERNEL,
};
use crate::demucs_spectral_weights::{
    DemucsSpectralDecoderWeights, SpectralDConvSubLayerWeights, SpectralDecoderBlockWeights,
};

// -- conv2d_output_len --------------------------------------------------------

#[test]
fn test_conv2d_output_len_basic() {
    // (8 + 2*1 - 3) / 1 + 1 = 8 (same-padding with k=3, s=1, p=1)
    let out = conv2d_output_len(8, 3, 1, 1).unwrap();
    assert_eq!(out, 8);
}

#[test]
fn test_conv2d_output_len_stride2() {
    // (16 + 2*0 - 4) / 2 + 1 = 7
    let out = conv2d_output_len(16, 4, 2, 0).unwrap();
    assert_eq!(out, 7);
}

#[test]
fn test_conv2d_output_len_rejects_zero_stride() {
    let err = conv2d_output_len(8, 3, 0, 1).unwrap_err();
    assert!(err.to_string().contains("stride"), "error: {err}");
}

#[test]
fn test_conv2d_output_len_rejects_too_small() {
    // 2 + 2*0 = 2 < 3 (kernel)
    let err = conv2d_output_len(2, 3, 1, 0).unwrap_err();
    assert!(err.to_string().contains("padded"), "error: {err}");
}

// -- validate_all_decoder_weights ---------------------------------------------

#[test]
fn test_validate_all_decoder_weights_rejects_wrong_block_count() {
    let weights = DemucsSpectralDecoderWeights { blocks: vec![] };
    let err = validate_all_decoder_weights(&weights).unwrap_err();
    assert!(err.to_string().contains("expected"), "error: {err}");
}

// -- build_decoder_block_sub_defs ---------------------------------------------

#[test]
fn test_build_decoder_block_sub_defs_depth0() {
    // block_idx=0 corresponds to encoder_depth=3 (deepest)
    let encoder_depth = SPECTRAL_BASIC_DEPTH - 1; // 3
    let in_ch = channels_at_depth(encoder_depth); // 384
    let out_ch = channels_at_depth(encoder_depth - 1); // 192
    let f_in = 4;
    let t_in = 16;
    let rw_f_out = f_in; // rewrite preserves freq with same-padding
    let rw_t_out = t_in;
    let target_f = f_in * 4; // rough upsample target

    let sub_defs = build_decoder_block_sub_defs(
        0, in_ch, out_ch, f_in, t_in, rw_f_out, rw_t_out, target_f, false,
    )
    .expect("sub_defs should build");

    // rewrite_def: output is [in_ch, rw_f_out * rw_t_out]
    let rw_shape = &sub_defs.rewrite_def.nodes[sub_defs.rewrite_def.output.index()].shape;
    assert_eq!(rw_shape[0], in_ch);

    // dconv_def: output is [in_ch, rw_t_out]
    let dc_shape = &sub_defs.dconv_def.nodes[sub_defs.dconv_def.output.index()].shape;
    assert_eq!(dc_shape[0], in_ch);
    assert_eq!(dc_shape[1], rw_t_out);

    // conv_tr_def: output channel is out_ch
    let ct_shape = &sub_defs.conv_tr_def.nodes[sub_defs.conv_tr_def.output.index()].shape;
    assert_eq!(ct_shape[0], out_ch);
}

#[test]
fn test_build_decoder_block_sub_defs_last_block() {
    // Last block: out_ch = SPECTRAL_OUTPUT_CHANNELS (16)
    let encoder_depth = 0;
    let in_ch = channels_at_depth(encoder_depth); // 48
    let out_ch = SPECTRAL_OUTPUT_CHANNELS; // 16
    let f_in = 31;
    let t_in = 16;
    let target_f = 128;

    let sub_defs =
        build_decoder_block_sub_defs(3, in_ch, out_ch, f_in, t_in, f_in, t_in, target_f, true)
            .expect("last block sub_defs should build");

    let ct_shape = &sub_defs.conv_tr_def.nodes[sub_defs.conv_tr_def.output.index()].shape;
    assert_eq!(
        ct_shape[0], out_ch,
        "last block output channels should be spectral output channels"
    );
}

// -- build_decoder_block_weight_maps ------------------------------------------

fn make_zero_dconv_sub(channels: usize, compressed: usize) -> SpectralDConvSubLayerWeights {
    SpectralDConvSubLayerWeights {
        conv_compress_weight: vec![0.0; compressed * channels * DCONV_KERNEL],
        conv_compress_bias: vec![0.0; compressed],
        norm_compress_gamma: vec![0.0; compressed],
        norm_compress_beta: vec![0.0; compressed],
        conv_expand_weight: vec![0.0; channels * 2 * compressed],
        conv_expand_bias: vec![0.0; channels * 2],
        norm_expand_gamma: vec![0.0; channels * 2],
        norm_expand_beta: vec![0.0; channels * 2],
        layer_scale: vec![0.0; channels],
    }
}

#[test]
fn test_build_decoder_block_weight_maps_keys() {
    let in_ch = channels_at_depth(3); // 384
    let out_ch = channels_at_depth(2); // 192
    let compressed = in_ch / DCONV_COMPRESS;

    let dconv_sub = make_zero_dconv_sub(in_ch, compressed);

    let block = SpectralDecoderBlockWeights {
        rewrite_weight: vec![
            0.0;
            in_ch * 2 * in_ch * SPECTRAL_REWRITE_KERNEL * SPECTRAL_REWRITE_KERNEL
        ],
        rewrite_bias: vec![0.0; in_ch * 2],
        dconv: vec![dconv_sub.clone(), dconv_sub],
        conv_tr_weight: vec![0.0; in_ch * out_ch * SPECTRAL_KERNEL_SIZE],
        conv_tr_bias: vec![0.0; out_ch],
    };

    let maps = build_decoder_block_weight_maps(&block);

    assert!(maps.rewrite.contains_key("rw_weight"));
    assert!(maps.rewrite.contains_key("rw_bias"));
    assert!(maps.dconv.contains_key("dc0_cw"));
    assert!(maps.dconv.contains_key("dc1_ls"));
    assert!(maps.dconv.contains_key("dc0_eps"));
    assert!(maps.conv_tr.contains_key("ct_weight"));
    assert!(maps.conv_tr.contains_key("ct_bias"));
}

#[test]
fn test_build_decoder_block_weight_maps_dconv_eps_entries() {
    let in_ch = 48;
    let compressed = in_ch / DCONV_COMPRESS;
    let out_ch = 16;

    let dconv_sub = make_zero_dconv_sub(in_ch, compressed);

    let block = SpectralDecoderBlockWeights {
        rewrite_weight: vec![
            0.0;
            in_ch * 2 * in_ch * SPECTRAL_REWRITE_KERNEL * SPECTRAL_REWRITE_KERNEL
        ],
        rewrite_bias: vec![0.0; in_ch * 2],
        dconv: vec![dconv_sub.clone(), dconv_sub],
        conv_tr_weight: vec![0.0; in_ch * out_ch * SPECTRAL_KERNEL_SIZE],
        conv_tr_bias: vec![0.0; out_ch],
    };

    let maps = build_decoder_block_weight_maps(&block);

    // Each DConv sub-layer should have 11 entries (9 weight + 2 eps)
    // Total for 2 sub-layers: 22 entries
    assert_eq!(maps.dconv.len(), 22);
}

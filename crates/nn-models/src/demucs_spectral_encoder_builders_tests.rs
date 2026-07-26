// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Tests for [`super::demucs_spectral_encoder_builders`].

use super::*;
use crate::demucs_shared::{
    channels_at_depth, SPECTRAL_CONV_PADDING, SPECTRAL_INPUT_CHANNELS, SPECTRAL_KERNEL_SIZE,
    SPECTRAL_STRIDE,
};

// -- spectral_conv1d_out_len --------------------------------------------------

#[test]
fn test_spectral_conv1d_out_len_basic() {
    // (256 + 2*2 - 8) / 4 + 1 = 252/4 + 1 = 64
    let out = spectral_conv1d_out_len(
        256,
        SPECTRAL_KERNEL_SIZE,
        SPECTRAL_STRIDE,
        SPECTRAL_CONV_PADDING,
    )
    .unwrap();
    assert_eq!(out, 64);
}

// -- build_encoder_block_sub_defs ---------------------------------------------

#[test]
fn test_build_encoder_block_sub_defs_depth0() {
    let in_ch = SPECTRAL_INPUT_CHANNELS; // 4
    let out_ch = channels_at_depth(0); // 48
    let f_in = 128;
    let f_out = spectral_conv1d_out_len(
        f_in,
        SPECTRAL_KERNEL_SIZE,
        SPECTRAL_STRIDE,
        SPECTRAL_CONV_PADDING,
    )
    .unwrap();
    let t_len = 16;

    let sub_defs = build_encoder_block_sub_defs(0, in_ch, out_ch, f_in, f_out, t_len)
        .expect("sub_defs should build");

    // conv_gelu_def output: [out_ch, f_out]
    let conv_shape = &sub_defs.conv_gelu_def.nodes[sub_defs.conv_gelu_def.output.index()].shape;
    assert_eq!(conv_shape[0], out_ch);
    assert_eq!(conv_shape[1], f_out);

    // dconv_def output: [out_ch, t_len]
    let dconv_shape = &sub_defs.dconv_def.nodes[sub_defs.dconv_def.output.index()].shape;
    assert_eq!(dconv_shape[0], out_ch);
    assert_eq!(dconv_shape[1], t_len);

    // rewrite_def output: [out_ch, f_out]
    let rw_shape = &sub_defs.rewrite_def.nodes[sub_defs.rewrite_def.output.index()].shape;
    assert_eq!(rw_shape[0], out_ch);
    assert_eq!(rw_shape[1], f_out);
}

#[test]
fn test_build_encoder_block_sub_defs_depth1() {
    let in_ch = channels_at_depth(0); // 48
    let out_ch = channels_at_depth(1); // 96
    let f_in = 31;
    let f_out = spectral_conv1d_out_len(
        f_in,
        SPECTRAL_KERNEL_SIZE,
        SPECTRAL_STRIDE,
        SPECTRAL_CONV_PADDING,
    )
    .unwrap();
    let t_len = 16;

    let sub_defs = build_encoder_block_sub_defs(1, in_ch, out_ch, f_in, f_out, t_len)
        .expect("sub_defs should build");

    let conv_shape = &sub_defs.conv_gelu_def.nodes[sub_defs.conv_gelu_def.output.index()].shape;
    assert_eq!(conv_shape[0], out_ch);
}

// -- validate_all_encoder_weights ---------------------------------------------

#[test]
fn test_validate_all_encoder_weights_rejects_wrong_block_count() {
    use crate::demucs_spectral_weights::DemucsSpectralEncoderWeights;

    let weights = DemucsSpectralEncoderWeights {
        blocks: vec![], // wrong count
        freq_emb_weight: None,
    };
    let err = validate_all_encoder_weights(&weights).unwrap_err();
    let msg = err.to_string();
    assert!(
        msg.contains("expected"),
        "error should mention expected: {msg}"
    );
}

// -- build_encoder_block_weight_maps ------------------------------------------

#[test]
fn test_build_encoder_block_weight_maps_keys() {
    use crate::demucs_shared::DCONV_KERNEL;
    use crate::demucs_spectral_weights::{
        SpectralEncDConvSubLayerWeights, SpectralEncoderBlockWeights,
    };

    let out_ch = channels_at_depth(0); // 48
    let in_ch = SPECTRAL_INPUT_CHANNELS; // 4
    let compressed = out_ch / 4;

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

    let block = SpectralEncoderBlockWeights {
        conv_weight: vec![0.0; out_ch * in_ch * SPECTRAL_KERNEL_SIZE],
        conv_bias: vec![0.0; out_ch],
        dconv: vec![dconv_sub.clone(), dconv_sub],
        rewrite_weight: vec![0.0; out_ch * 2 * out_ch],
        rewrite_bias: vec![0.0; out_ch * 2],
    };

    let maps = build_encoder_block_weight_maps(&block);

    assert!(maps.conv_gelu.contains_key("conv_weight"));
    assert!(maps.conv_gelu.contains_key("conv_bias"));
    assert!(maps.dconv.contains_key("dc0_cw"));
    assert!(maps.dconv.contains_key("dc1_ls"));
    assert!(maps.rewrite.contains_key("rw_weight"));
    assert!(maps.rewrite.contains_key("rw_bias"));
}

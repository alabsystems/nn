// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Tests for [`super::demucs_temporal_encoder_builders`].

use super::*;
use crate::demucs_shared::{channels_at_depth, TEMPORAL_KERNEL_SIZE};

// -- conv1d_out_len -----------------------------------------------------------

#[test]
fn test_conv1d_out_len_depth0() {
    // depth0: in_ch=2 (stereo), padded_t=256
    // (256 + 2*2 - 8) / 4 + 1 = 252/4 + 1 = 64
    assert_eq!(conv1d_out_len(256).unwrap(), 64);
}

#[test]
fn test_conv1d_out_len_small() {
    // Minimum viable: padded_t = KERNEL_SIZE (8)
    // (8 + 2*2 - 8) / 4 + 1 = 2
    assert_eq!(conv1d_out_len(TEMPORAL_KERNEL_SIZE).unwrap(), 2);
}

// -- build_encoder_block_def --------------------------------------------------

#[test]
fn test_build_encoder_block_depth0_succeeds() {
    let in_ch = 2; // stereo input
    let out_ch = channels_at_depth(0); // 48
    let padded_t = 256;
    let def =
        build_encoder_block_def(0, in_ch, out_ch, padded_t).expect("encoder block 0 should build");
    assert_eq!(def.name, "demucs_enc_block0");
    // Output shape: [out_ch, conv_t_out]
    let output_shape = &def.nodes[def.output.index()].shape;
    assert_eq!(output_shape[0], out_ch);
}

#[test]
fn test_build_encoder_block_depth1_succeeds() {
    let in_ch = channels_at_depth(0); // 48
    let out_ch = channels_at_depth(1); // 96
    let padded_t = 128;
    let def =
        build_encoder_block_def(1, in_ch, out_ch, padded_t).expect("encoder block 1 should build");
    assert_eq!(def.name, "demucs_enc_block1");
    let output_shape = &def.nodes[def.output.index()].shape;
    assert_eq!(output_shape[0], out_ch);
}

#[test]
fn test_build_encoder_block_all_depths() {
    // Build all 4 encoder blocks matching HTDemucs architecture
    let depths = [(2, 48), (48, 96), (96, 192), (192, 384)];
    let mut padded_t = 256;
    for (block_idx, &(in_ch, out_ch)) in depths.iter().enumerate() {
        let def = build_encoder_block_def(block_idx, in_ch, out_ch, padded_t)
            .unwrap_or_else(|_| panic!("encoder block {block_idx} should build"));
        let output_shape = &def.nodes[def.output.index()].shape;
        assert_eq!(output_shape[0], out_ch, "block {block_idx} out_ch");
        // Next block's padded_t = this block's temporal output
        padded_t = output_shape[1];
    }
}

#[test]
fn test_build_encoder_block_node_count() {
    let in_ch = 2;
    let out_ch = 48;
    let padded_t = 64;
    let def = build_encoder_block_def(0, in_ch, out_ch, padded_t).expect("build should succeed");
    // Should have multiple nodes (inputs, conv, gelu, dconv, rewrite, glu)
    assert!(
        def.nodes.len() > 10,
        "expected many nodes, got {}",
        def.nodes.len()
    );
}

// -- build_encoder_weight_map -------------------------------------------------

#[test]
fn test_build_encoder_weight_map_has_expected_keys() {
    use crate::demucs_shared::{DCONV_DEPTH, DCONV_KERNEL};
    use crate::demucs_temporal_weights::{DConvSubLayerWeights, EncoderBlockWeights};

    let in_ch = 2;
    let out_ch = 48;
    let compressed = out_ch / 4; // DCONV_COMPRESS = 4

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

    // Must have conv_weight, conv_bias, rw_weight, rw_bias
    assert!(map.contains_key("conv_weight"));
    assert!(map.contains_key("conv_bias"));
    assert!(map.contains_key("rw_weight"));
    assert!(map.contains_key("rw_bias"));

    // Must have DConv keys for each sub-layer
    for k in 0..DCONV_DEPTH {
        assert!(map.contains_key(&format!("dc{k}_cw")));
        assert!(map.contains_key(&format!("dc{k}_ls")));
        assert!(map.contains_key(&format!("dc{k}_eps")));
    }
}

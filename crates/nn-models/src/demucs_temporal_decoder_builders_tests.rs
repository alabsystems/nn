// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Tests for [`super::demucs_temporal_decoder_builders`].

use super::*;
use crate::demucs_shared::channels_at_depth;

// -- build_decoder_block_def --------------------------------------------------

#[test]
fn test_build_decoder_block_last_succeeds() {
    // Last block: in_ch=48, out_ch=2 (stereo), is_last=true (no GELU)
    let in_ch = channels_at_depth(0); // 48
    let out_ch = 2;
    let t_in = 63;
    let target_len = 256;
    let def = build_decoder_block_def(3, in_ch, out_ch, t_in, target_len, true)
        .expect("decoder last block should build");
    assert_eq!(def.name, "demucs_dec_block3");
    let output_shape = &def.nodes[def.output.index()].shape;
    assert_eq!(output_shape[0], out_ch);
}

#[test]
fn test_build_decoder_block_non_last_succeeds() {
    // Non-last block: in_ch=96, out_ch=48, is_last=false (includes GELU)
    let in_ch = channels_at_depth(1); // 96
    let out_ch = channels_at_depth(0); // 48
    let t_in = 16;
    let target_len = 63;
    let def = build_decoder_block_def(2, in_ch, out_ch, t_in, target_len, false)
        .expect("decoder non-last block should build");
    assert_eq!(def.name, "demucs_dec_block2");
}

#[test]
fn test_build_decoder_block_has_skip_input() {
    let in_ch = 48;
    let out_ch = 2;
    let def =
        build_decoder_block_def(0, in_ch, out_ch, 32, 128, true).expect("build should succeed");
    // Should have at least a "data" and "skip" input
    let input_count = def
        .nodes
        .iter()
        .filter(|n| matches!(n.kind, nn_dsl::tensor_ir::TensorOpKind::Input { .. }))
        .count();
    // 2 variable inputs (data + skip) + conv/dconv/rewrite weights
    assert!(
        input_count >= 2,
        "expected at least 2 inputs, got {input_count}"
    );
}

#[test]
fn test_build_decoder_all_depths() {
    // Decoder runs in reverse depth order: 384→192→96→48→2
    let depths = [
        (384, 192, false),
        (192, 96, false),
        (96, 48, false),
        (48, 2, true),
    ];
    let mut t_in = 4; // smallest temporal dim at bottleneck
    for (block_idx, &(in_ch, out_ch, is_last)) in depths.iter().enumerate() {
        let target_len = t_in * 4; // approximate upsample target
        let def = build_decoder_block_def(block_idx, in_ch, out_ch, t_in, target_len, is_last)
            .unwrap_or_else(|_| panic!("decoder block {block_idx} should build"));
        let output_shape = &def.nodes[def.output.index()].shape;
        assert_eq!(output_shape[0], out_ch, "block {block_idx} out_ch");
        t_in = output_shape[1];
    }
}

// -- build_decoder_weight_map -------------------------------------------------

#[test]
fn test_build_decoder_weight_map_has_expected_keys() {
    use crate::demucs_shared::{DCONV_DEPTH, DCONV_KERNEL, TEMPORAL_KERNEL_SIZE};
    use crate::demucs_temporal_weights::{DConvSubLayerWeights, DecoderBlockWeights};

    let in_ch = 48;
    let out_ch = 2;
    let compressed = in_ch / 4;

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
        rewrite_weight: vec![0.0; in_ch * 2 * in_ch * 3],
        rewrite_bias: vec![0.0; in_ch * 2],
        dconv: vec![dconv_sub.clone(), dconv_sub],
        conv_tr_weight: vec![0.0; in_ch * out_ch * TEMPORAL_KERNEL_SIZE],
        conv_tr_bias: vec![0.0; out_ch],
    };

    let map = build_decoder_weight_map(&block);

    assert!(map.contains_key("rw_weight"));
    assert!(map.contains_key("ct_weight"));
    assert!(map.contains_key("ct_bias"));
    for k in 0..DCONV_DEPTH {
        assert!(map.contains_key(&format!("dc{k}_cw")));
    }
}

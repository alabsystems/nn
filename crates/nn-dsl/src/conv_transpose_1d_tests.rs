// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

use super::*;

#[test]
fn test_basic_conv_transpose_1d_builds() {
    // in_ch=4, out_ch=2, k=3, in_len=8, stride=2, pad=1, dilation=1, groups=1
    // out_len = (8-1)*2 - 2*1 + 1*(3-1) + 0 + 1 = 14 - 2 + 2 + 1 = 15
    let def =
        build_conv_transpose_1d("test", 4, 2, 3, 8, 2, 1, 1, 1, false, 0).expect("should build");
    assert_eq!(def.nodes.len(), 3); // data, weight, conv_transpose
    assert_eq!(def.nodes.last().unwrap().shape, vec![2, 15]);
}

#[test]
fn test_conv_transpose_1d_with_bias() {
    let def = build_conv_transpose_1d("test_bias", 4, 2, 3, 8, 2, 1, 1, 1, true, 0)
        .expect("should build");
    assert_eq!(def.nodes.len(), 4); // data, weight, bias, conv_transpose
    assert_eq!(def.nodes.last().unwrap().shape, vec![2, 15]);
}

#[test]
fn test_conv_transpose_1d_kokoro_upsample() {
    // Kokoro Generator: stride=10, kernel=20
    // out_len = (8-1)*10 - 0 + 1*(20-1) + 0 + 1 = 70 + 19 + 1 = 90
    let def = build_conv_transpose_1d("kokoro", 512, 256, 20, 8, 10, 0, 1, 1, true, 0)
        .expect("should build");
    assert_eq!(def.nodes.last().unwrap().shape, vec![256, 90]);
}

#[test]
fn test_conv_transpose_1d_demucs_decoder() {
    // Demucs decoder: stride=4, kernel=8
    // out_len = (16-1)*4 - 0 + 1*(8-1) + 0 + 1 = 60 + 7 + 1 = 68
    let def = build_conv_transpose_1d("demucs", 96, 48, 8, 16, 4, 0, 1, 1, true, 0)
        .expect("should build");
    assert_eq!(def.nodes.last().unwrap().shape, vec![48, 68]);
}

#[test]
fn test_conv_transpose_1d_zero_stride_rejected() {
    let result = build_conv_transpose_1d("test", 4, 2, 3, 8, 0, 0, 1, 1, false, 0);
    assert!(result.is_err());
}

#[test]
fn test_conv_transpose_1d_excessive_padding_rejected() {
    // out_len = (2-1)*1 - 2*10 + 1*(2-1) + 0 + 1 = 1 - 20 + 1 + 1 = negative
    let result = build_conv_transpose_1d("test", 4, 2, 2, 2, 1, 10, 1, 1, false, 0);
    assert!(result.is_err());
}

#[test]
fn test_conv_transpose_1d_validates() {
    let def =
        build_conv_transpose_1d("test_val", 4, 2, 3, 8, 2, 1, 1, 1, true, 0).expect("should build");
    def.validate().expect("should validate");
}

#[test]
fn test_conv_transpose_1d_dilation() {
    // dilation=2, kernel=3: effective kernel = 2*(3-1)+1 = 5
    // out_len = (4-1)*1 - 0 + 2*(3-1) + 0 + 1 = 3 + 4 + 1 = 8
    let def = build_conv_transpose_1d("test_dil", 4, 2, 3, 4, 1, 0, 2, 1, false, 0)
        .expect("should build");
    assert_eq!(def.nodes.last().unwrap().shape, vec![2, 8]);
}

#[test]
fn test_conv_transpose_1d_groups() {
    // groups=2: in_ch=4, out_ch_per_group=2 (so out_ch=4), kernel=3
    // out_len = (8-1)*1 - 0 + 1*(3-1) + 0 + 1 = 7 + 2 + 1 = 10
    let def = build_conv_transpose_1d("test_grp", 4, 4, 3, 8, 1, 0, 1, 2, false, 0)
        .expect("should build");
    assert_eq!(def.nodes.last().unwrap().shape, vec![4, 10]);
    // Weight shape should be [4, 2, 3] (out_ch_per_group=2)
    assert_eq!(def.nodes[1].shape, vec![4, 2, 3]);
}

#[test]
fn test_conv_transpose_1d_zero_dilation_rejected() {
    let result = build_conv_transpose_1d("test", 4, 2, 3, 8, 1, 0, 0, 1, false, 0);
    assert!(result.is_err());
}

#[test]
fn test_conv_transpose_1d_zero_groups_rejected() {
    let result = build_conv_transpose_1d("test", 4, 2, 3, 8, 1, 0, 1, 0, false, 0);
    assert!(result.is_err());
}

#[test]
fn test_conv_transpose_1d_output_padding() {
    // stride=2, output_padding=1: disambiguates output length
    // out_len = (4-1)*2 - 0 + 1*(3-1) + 1 + 1 = 6 + 2 + 1 + 1 = 10
    let def = build_conv_transpose_1d("test_op", 2, 1, 3, 4, 2, 0, 1, 1, false, 1)
        .expect("should build with output_padding=1");
    assert_eq!(def.nodes.last().unwrap().shape, vec![1, 10]);
}

#[test]
fn test_conv_transpose_1d_output_padding_ge_stride_rejected() {
    // output_padding=2, stride=2 → invalid (must be < stride)
    let result = build_conv_transpose_1d("test", 2, 1, 3, 4, 2, 0, 1, 1, false, 2);
    assert!(result.is_err());
}

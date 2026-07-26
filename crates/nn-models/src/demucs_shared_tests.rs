// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Tests for [`super::demucs_shared`] — conv1d_output_len, channels_at_depth,
//! validate_weight_size, DConvSubLayerInputs, build_dconv_sublayer.

use super::*;

// -- conv1d_output_len --------------------------------------------------------

#[test]
fn test_conv1d_output_len_htdemucs_temporal() {
    // HTDemucs temporal: kernel=8, stride=4, padding=2
    // (256 + 2*2 - 8) / 4 + 1 = 252/4 + 1 = 64
    assert_eq!(conv1d_output_len(256, 8, 4, 2).unwrap(), 64);
}

#[test]
fn test_conv1d_output_len_kernel_equals_input() {
    // kernel == input+2*padding: output should be 1 (single window)
    assert_eq!(conv1d_output_len(8, 8, 1, 0).unwrap(), 1);
}

#[test]
fn test_conv1d_output_len_unit_stride() {
    // stride=1, no padding: output = in - kernel + 1
    assert_eq!(conv1d_output_len(16, 3, 1, 0).unwrap(), 14);
}

#[test]
fn test_conv1d_output_len_with_large_padding() {
    // Padding larger than half kernel (same-padding-like)
    assert_eq!(conv1d_output_len(100, 7, 1, 3).unwrap(), 100);
}

#[test]
fn test_conv1d_output_len_err_on_zero_output() {
    // kernel > in + 2*padding → zero or negative output → returns Err
    assert!(conv1d_output_len(1, 10, 1, 0).is_err());
}

// -- channels_at_depth --------------------------------------------------------

#[test]
fn test_channels_at_depth_0() {
    // depth 0: BASE_CHANNELS * 2^0 = 48
    assert_eq!(channels_at_depth(0), 48);
}

#[test]
fn test_channels_at_depth_1() {
    // depth 1: 48 * 2^1 = 96
    assert_eq!(channels_at_depth(1), 96);
}

#[test]
fn test_channels_at_depth_2() {
    // depth 2: 48 * 2^2 = 192
    assert_eq!(channels_at_depth(2), 192);
}

#[test]
fn test_channels_at_depth_3() {
    // depth 3: 48 * 2^3 = 384
    assert_eq!(channels_at_depth(3), 384);
}

#[test]
fn test_channels_at_depth_matches_htdemucs_architecture() {
    // Full HTDemucs has 4 depths: 48, 96, 192, 384
    let expected = [48, 96, 192, 384];
    for (d, &exp) in expected.iter().enumerate() {
        assert_eq!(channels_at_depth(d), exp, "depth {d}");
    }
}

// -- validate_weight_size -----------------------------------------------------

#[test]
fn test_validate_weight_size_correct() {
    let data = vec![0.0; 100];
    assert!(validate_weight_size(&data, "test", 100).is_ok());
}

#[test]
fn test_validate_weight_size_too_small() {
    let data = vec![0.0; 50];
    let err = validate_weight_size(&data, "nn_weight", 100).unwrap_err();
    let msg = err.to_string();
    assert!(msg.contains("nn_weight"));
    assert!(msg.contains("100"));
    assert!(msg.contains("50"));
}

#[test]
fn test_validate_weight_size_too_large() {
    let data = vec![0.0; 200];
    let err = validate_weight_size(&data, "big", 100).unwrap_err();
    let msg = err.to_string();
    assert!(msg.contains("big"));
    assert!(msg.contains("200"));
}

#[test]
fn test_validate_weight_size_empty() {
    let data: Vec<f32> = vec![];
    assert!(validate_weight_size(&data, "empty", 0).is_ok());
}

#[test]
fn test_validate_weight_size_empty_expected_nonzero() {
    let data: Vec<f32> = vec![];
    assert!(validate_weight_size(&data, "w", 10).is_err());
}

// -- Architecture constants ---------------------------------------------------

#[test]
fn test_temporal_constants_consistency() {
    // padding = kernel_size / 4 for both encoder and decoder
    assert_eq!(TEMPORAL_CONV_PADDING, TEMPORAL_KERNEL_SIZE / 4);
    assert_eq!(TEMPORAL_CONV_TR_PADDING, TEMPORAL_KERNEL_SIZE / 4);
}

#[test]
fn test_spectral_constants_consistency() {
    assert_eq!(SPECTRAL_CONV_PADDING, SPECTRAL_KERNEL_SIZE / 4);
    assert_eq!(SPECTRAL_CONV_TR_PADDING, SPECTRAL_KERNEL_SIZE / 4);
    // Spectral input: 2 stereo × 2 (real+imag) = 4
    assert_eq!(SPECTRAL_INPUT_CHANNELS, 4);
    // Spectral output: 4 sources × 2 stereo × 2 (real+imag) = 16
    assert_eq!(SPECTRAL_OUTPUT_CHANNELS, 16);
}

#[test]
fn test_decoder_output_channels() {
    // 4 sources × 2 stereo = 8
    assert_eq!(DECODER_OUTPUT_CHANNELS, 8);
}

#[test]
fn test_dconv_constants() {
    // Use runtime values to avoid clippy::assertions_on_constants
    let compress = DCONV_COMPRESS;
    let depth = DCONV_DEPTH;
    let kernel = DCONV_KERNEL;
    let eps = GROUP_NORM_EPS;
    assert!(compress > 0);
    assert!(depth > 0);
    assert!(kernel > 0);
    assert!(eps > 0.0);
}

// -- DConvSubLayerInputs ------------------------------------------------------

#[test]
fn test_dconv_sublayer_inputs_add_to_builder() {
    let mut b = TensorBlockBuilder::new("test_dconv_sublayer");
    let channels = 96;
    let compressed = channels / DCONV_COMPRESS; // 24
    let dc = DConvSubLayerInputs::add_to_builder(&mut b, 0, channels, compressed);
    // Dilation for k=0: 2^0 = 1
    assert_eq!(dc.dilation, 1);
}

#[test]
fn test_dconv_sublayer_inputs_dilation_scaling() {
    let mut b = TensorBlockBuilder::new("test");
    let channels = 48;
    let compressed = 6;
    // k=0 → dilation=1, k=1 → dilation=2, k=2 → dilation=4
    for k in 0..3 {
        let dc = DConvSubLayerInputs::add_to_builder(&mut b, k, channels, compressed);
        assert_eq!(dc.dilation, 1 << k, "k={k}");
    }
}

// -- build_dconv_sublayer (graph construction) ---------------------------------

#[test]
fn test_build_dconv_sublayer_succeeds() {
    let mut b = TensorBlockBuilder::new("test");
    let channels = 48;
    let compressed = channels / DCONV_COMPRESS;
    let t_len = 64;
    let input = b.add_input("input", &[channels, t_len]);
    let dc = DConvSubLayerInputs::add_to_builder(&mut b, 0, channels, compressed);
    let result = build_dconv_sublayer(&mut b, input, &dc, channels, compressed, t_len);
    assert!(
        result.is_ok(),
        "build_dconv_sublayer failed: {:?}",
        result.err()
    );
}

#[test]
fn test_build_dconv_sublayer_multiple_layers() {
    let mut b = TensorBlockBuilder::new("test");
    let channels = 96;
    let compressed = channels / DCONV_COMPRESS;
    let t_len = 32;
    let mut x = b.add_input("input", &[channels, t_len]);
    for k in 0..DCONV_DEPTH {
        let dc = DConvSubLayerInputs::add_to_builder(&mut b, k, channels, compressed);
        x = build_dconv_sublayer(&mut b, x, &dc, channels, compressed, t_len)
            .expect("build_dconv_sublayer failed");
    }
    // Successfully built 2 stacked DConv sub-layers
}

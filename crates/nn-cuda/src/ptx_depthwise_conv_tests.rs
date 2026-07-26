// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Tests for PTX depthwise conv2d kernel generation.

use super::*;

// ---------------------------------------------------------------------------
// Output size computation
// ---------------------------------------------------------------------------

#[test]
fn test_depthwise_output_size_basic() {
    // 4x4 input, 3x3 kernel, stride 1, no padding -> 2
    assert_eq!(depthwise_conv2d_output_size(4, 3, 1, 0), Some(2));
}

#[test]
fn test_depthwise_output_size_with_padding() {
    // 4x4 input, 3x3 kernel, stride 1, padding 1 -> 4 (same padding)
    assert_eq!(depthwise_conv2d_output_size(4, 3, 1, 1), Some(4));
}

#[test]
fn test_depthwise_output_size_stride2() {
    // 8 input, 3 kernel, stride 2, padding 1 -> (8+2-3)/2+1 = 4
    assert_eq!(depthwise_conv2d_output_size(8, 3, 2, 1), Some(4));
}

#[test]
fn test_depthwise_output_size_1x1_kernel() {
    // 1x1 kernel = pointwise, output = input / stride
    assert_eq!(depthwise_conv2d_output_size(7, 1, 1, 0), Some(7));
}

#[test]
fn test_depthwise_output_size_kernel_larger_than_input() {
    assert_eq!(depthwise_conv2d_output_size(3, 5, 1, 0), None);
}

#[test]
fn test_depthwise_output_size_stride_larger_than_kernel() {
    // 10 input, 2 kernel, stride 3 -> (10-2)/3+1 = 3
    assert_eq!(depthwise_conv2d_output_size(10, 2, 3, 0), Some(3));
}

// ---------------------------------------------------------------------------
// Config validation
// ---------------------------------------------------------------------------

#[test]
fn test_depthwise_config_default() {
    let c = PtxDepthwiseConv2dConfig::default();
    assert_eq!(c.channels, 32);
    assert_eq!(c.kernel_h, 3);
    assert_eq!(c.kernel_w, 3);
    assert_eq!(c.stride_h, 1);
    assert_eq!(c.stride_w, 1);
    assert_eq!(c.padding_h, 0);
    assert_eq!(c.padding_w, 0);
    assert!(!c.use_bias);
    assert!(c.validate().is_ok());
}

#[test]
fn test_depthwise_config_empty_name() {
    let c = PtxDepthwiseConv2dConfig::new("", 32, 3, 3);
    assert!(c.validate().is_err());
}

#[test]
fn test_depthwise_config_zero_channels() {
    let c = PtxDepthwiseConv2dConfig::new("dw", 0, 3, 3);
    assert!(c.validate().is_err());
}

#[test]
fn test_depthwise_config_zero_kernel() {
    let c = PtxDepthwiseConv2dConfig::new("dw", 32, 0, 3);
    assert!(c.validate().is_err());
}

#[test]
fn test_depthwise_config_zero_stride() {
    let c = PtxDepthwiseConv2dConfig::new("dw", 32, 3, 3).with_stride(0, 1);
    assert!(c.validate().is_err());
}

#[test]
fn test_depthwise_config_block_too_large() {
    let c = PtxDepthwiseConv2dConfig::new("dw", 32, 3, 3).with_block_size(2048);
    assert!(c.validate().is_err());
}

#[test]
fn test_depthwise_config_builder_chain() {
    let c = PtxDepthwiseConv2dConfig::new("dw_conv", 64, 5, 5)
        .with_stride(2, 2)
        .with_padding(2, 2)
        .with_bias(true)
        .with_block_size(128)
        .with_sm_target("sm_90");
    assert_eq!(c.channels, 64);
    assert_eq!(c.kernel_h, 5);
    assert_eq!(c.stride_h, 2);
    assert_eq!(c.padding_h, 2);
    assert!(c.use_bias);
    assert_eq!(c.block_size, 128);
    assert_eq!(c.sm_target, "sm_90");
    assert!(c.validate().is_ok());
}

// ---------------------------------------------------------------------------
// Reference implementation
// ---------------------------------------------------------------------------

#[test]
fn test_depthwise_conv2d_reference_no_padding() {
    // 1 batch, 1 channel, 3x3 input, 2x2 kernel, stride 1, no padding -> 2x2 output
    let input = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
    // weight[0, :, :] = [[1, 0], [0, 1]]
    let weight = vec![1.0, 0.0, 0.0, 1.0];
    let config = PtxDepthwiseConv2dConfig::new("dw", 1, 2, 2);
    let out = depthwise_conv2d_reference(&input, &weight, None, &config, 1, 3, 3);
    assert_eq!(out.len(), 4);
    // out[0,0] = 1*1 + 2*0 + 4*0 + 5*1 = 6
    assert!((out[0] - 6.0).abs() < 1e-6);
    // out[0,1] = 2*1 + 3*0 + 5*0 + 6*1 = 8
    assert!((out[1] - 8.0).abs() < 1e-6);
    // out[1,0] = 4*1 + 5*0 + 7*0 + 8*1 = 12
    assert!((out[2] - 12.0).abs() < 1e-6);
    // out[1,1] = 5*1 + 6*0 + 8*0 + 9*1 = 14
    assert!((out[3] - 14.0).abs() < 1e-6);
}

#[test]
fn test_depthwise_conv2d_reference_with_bias() {
    let input = vec![1.0, 2.0, 3.0, 4.0]; // 1x1x2x2
    let weight = vec![1.0]; // 1x1x1 (1 channel, 1x1 kernel)
    let bias = vec![10.0];
    let config = PtxDepthwiseConv2dConfig::new("dw", 1, 1, 1);
    let out = depthwise_conv2d_reference(&input, &weight, Some(&bias), &config, 1, 2, 2);
    assert_eq!(out.len(), 4);
    assert!((out[0] - 11.0).abs() < 1e-6);
    assert!((out[1] - 12.0).abs() < 1e-6);
    assert!((out[2] - 13.0).abs() < 1e-6);
    assert!((out[3] - 14.0).abs() < 1e-6);
}

#[test]
fn test_depthwise_conv2d_reference_multichannel() {
    // 1 batch, 2 channels, 2x2 input, 1x1 kernel
    let input = vec![
        1.0, 2.0, 3.0, 4.0, // ch 0
        10.0, 20.0, 30.0, 40.0, // ch 1
    ];
    let weight = vec![2.0, 3.0]; // ch0 weight=2, ch1 weight=3
    let config = PtxDepthwiseConv2dConfig::new("dw", 2, 1, 1);
    let out = depthwise_conv2d_reference(&input, &weight, None, &config, 1, 2, 2);
    assert_eq!(out.len(), 8);
    // ch0: *2
    assert!((out[0] - 2.0).abs() < 1e-6);
    assert!((out[1] - 4.0).abs() < 1e-6);
    // ch1: *3
    assert!((out[4] - 30.0).abs() < 1e-6);
    assert!((out[5] - 60.0).abs() < 1e-6);
}

#[test]
fn test_depthwise_conv2d_reference_with_padding() {
    // 1 batch, 1 channel, 3x3 input, 3x3 kernel, pad 1 -> 3x3 output
    let input: Vec<f32> = (1..=9).map(|x| x as f32).collect();
    let weight = vec![0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0]; // identity filter
    let config = PtxDepthwiseConv2dConfig::new("dw", 1, 3, 3).with_padding(1, 1);
    let out = depthwise_conv2d_reference(&input, &weight, None, &config, 1, 3, 3);
    assert_eq!(out.len(), 9);
    // With identity filter and same-padding, output = input
    for i in 0..9 {
        assert!((out[i] - input[i]).abs() < 1e-6, "mismatch at idx {i}");
    }
}

// ---------------------------------------------------------------------------
// PTX structural checks
// ---------------------------------------------------------------------------

fn assert_dw_ptx_structure(ptx: &str, kernel_name: &str) {
    assert!(ptx.contains(".version 6.5"), "must contain PTX version");
    assert!(ptx.contains(".target sm_80"), "must contain SM target");
    assert!(
        ptx.contains(".address_size 64"),
        "must have 64-bit addressing"
    );
    assert!(
        ptx.contains(&format!(".visible .entry {kernel_name}")),
        "must have entry point: {kernel_name}"
    );
    assert!(ptx.contains("param_input"), "must have input param");
    assert!(ptx.contains("param_weight"), "must have weight param");
    assert!(ptx.contains("param_output"), "must have output param");
    assert!(
        ptx.contains("param_batch_size"),
        "must have batch_size param"
    );
    assert!(ptx.contains("ret;"), "must have ret instruction");
}

#[test]
fn test_depthwise_conv2d_ptx_structure() {
    let config = PtxDepthwiseConv2dConfig::new("dw_3x3", 32, 3, 3);
    let ptx = generate_depthwise_conv2d_ptx(&config).unwrap();
    assert_dw_ptx_structure(&ptx, "dw_3x3");
    assert!(ptx.contains("DepthwiseConv2d"));
}

#[test]
fn test_depthwise_conv2d_ptx_with_bias() {
    let config = PtxDepthwiseConv2dConfig::new("dw_bias", 32, 3, 3).with_bias(true);
    let ptx = generate_depthwise_conv2d_ptx(&config).unwrap();
    assert_dw_ptx_structure(&ptx, "dw_bias");
    assert!(ptx.contains("param_bias"), "must have bias param");
    assert!(ptx.contains("Add bias"), "must contain bias addition");
}

#[test]
fn test_depthwise_conv2d_ptx_no_bias() {
    let config = PtxDepthwiseConv2dConfig::new("dw_nobias", 32, 3, 3);
    let ptx = generate_depthwise_conv2d_ptx(&config).unwrap();
    assert!(!ptx.contains("param_bias"));
}

#[test]
fn test_depthwise_conv2d_ptx_has_grid_stride_loop() {
    let config = PtxDepthwiseConv2dConfig::new("dw", 32, 3, 3);
    let ptx = generate_depthwise_conv2d_ptx(&config).unwrap();
    assert!(ptx.contains("DW_LOOP:"));
    assert!(ptx.contains("DW_EXIT:"));
}

#[test]
fn test_depthwise_conv2d_ptx_has_kernel_loops() {
    let config = PtxDepthwiseConv2dConfig::new("dw", 32, 3, 3);
    let ptx = generate_depthwise_conv2d_ptx(&config).unwrap();
    assert!(ptx.contains("DW_KH_LOOP:"));
    assert!(ptx.contains("DW_KW_LOOP:"));
    assert!(ptx.contains("DW_KH_DONE:"));
    assert!(ptx.contains("DW_KW_DONE:"));
}

#[test]
fn test_depthwise_conv2d_ptx_has_fma() {
    let config = PtxDepthwiseConv2dConfig::new("dw", 32, 3, 3);
    let ptx = generate_depthwise_conv2d_ptx(&config).unwrap();
    assert!(ptx.contains("fma.rn.f32"), "must use fused multiply-add");
}

#[test]
fn test_depthwise_conv2d_ptx_is_pure_ptx() {
    let config = PtxDepthwiseConv2dConfig::new("dw", 32, 3, 3);
    let ptx = generate_depthwise_conv2d_ptx(&config).unwrap();
    assert!(!ptx.contains("__global__"));
    assert!(!ptx.contains("#include"));
}

#[test]
fn test_depthwise_conv2d_ptx_header_info() {
    let config = PtxDepthwiseConv2dConfig::new("dw", 64, 5, 5)
        .with_stride(2, 2)
        .with_padding(2, 2);
    let ptx = generate_depthwise_conv2d_ptx(&config).unwrap();
    assert!(ptx.contains("channels=64"));
    assert!(ptx.contains("kernel=5x5"));
    assert!(ptx.contains("stride=2x2"));
    assert!(ptx.contains("pad=2x2"));
}

#[test]
fn test_depthwise_conv2d_ptx_custom_sm_target() {
    let config = PtxDepthwiseConv2dConfig::new("dw", 32, 3, 3).with_sm_target("sm_90");
    let ptx = generate_depthwise_conv2d_ptx(&config).unwrap();
    assert!(ptx.contains(".target sm_90"));
}

#[test]
fn test_depthwise_conv2d_ptx_custom_block_size() {
    let config = PtxDepthwiseConv2dConfig::new("dw", 32, 3, 3).with_block_size(128);
    let ptx = generate_depthwise_conv2d_ptx(&config).unwrap();
    assert!(ptx.contains(".reqntid 128"));
}

// ---------------------------------------------------------------------------
// Edge cases
// ---------------------------------------------------------------------------

#[test]
fn test_depthwise_conv2d_ptx_1x1_kernel() {
    let config = PtxDepthwiseConv2dConfig::new("dw_1x1", 32, 1, 1);
    let ptx = generate_depthwise_conv2d_ptx(&config).unwrap();
    assert_dw_ptx_structure(&ptx, "dw_1x1");
    assert!(ptx.contains("kernel=1x1"));
}

#[test]
fn test_depthwise_conv2d_different_configs_different_ptx() {
    let config_a = PtxDepthwiseConv2dConfig::new("dw", 32, 3, 3);
    let config_b = PtxDepthwiseConv2dConfig::new("dw", 64, 5, 5);
    let ptx_a = generate_depthwise_conv2d_ptx(&config_a).unwrap();
    let ptx_b = generate_depthwise_conv2d_ptx(&config_b).unwrap();
    assert_ne!(ptx_a, ptx_b);
}

#[test]
fn test_depthwise_conv2d_bias_vs_no_bias_different_ptx() {
    let config_a = PtxDepthwiseConv2dConfig::new("dw", 32, 3, 3);
    let config_b = PtxDepthwiseConv2dConfig::new("dw", 32, 3, 3).with_bias(true);
    let ptx_a = generate_depthwise_conv2d_ptx(&config_a).unwrap();
    let ptx_b = generate_depthwise_conv2d_ptx(&config_b).unwrap();
    assert_ne!(ptx_a, ptx_b);
}

// ---------------------------------------------------------------------------
// Launch config
// ---------------------------------------------------------------------------

#[test]
fn test_depthwise_launch_config_basic() {
    // batch=1, channels=32, output=14x14 -> 1*32*14*14 = 6272
    let cfg = ptx_depthwise_conv2d_launch_config(1, 32, 14, 14);
    assert_eq!(cfg.block.x, 256);
    assert_eq!(cfg.grid.x, 25); // ceil(6272/256) = 24.5 -> 25
}

#[test]
fn test_depthwise_launch_config_small() {
    let cfg = ptx_depthwise_conv2d_launch_config(1, 1, 1, 1);
    assert_eq!(cfg.grid.x, 1);
}

#[test]
fn test_depthwise_launch_config_1d() {
    let cfg = ptx_depthwise_conv2d_launch_config(2, 64, 7, 7);
    assert_eq!(cfg.grid.y, 1);
    assert_eq!(cfg.grid.z, 1);
    assert_eq!(cfg.block.y, 1);
    assert_eq!(cfg.block.z, 1);
}

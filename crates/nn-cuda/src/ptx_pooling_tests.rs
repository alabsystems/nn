// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Tests for PTX pooling kernel generation (max pool, avg pool, adaptive avg pool).

use super::*;

// ---------------------------------------------------------------------------
// Output size computation
// ---------------------------------------------------------------------------

#[test]
fn test_pool2d_output_size_basic() {
    // 4x4 input, 2x2 kernel, stride 2, no padding -> 2
    assert_eq!(pool2d_output_size(4, 2, 2, 0), Some(2));
}

#[test]
fn test_pool2d_output_size_with_padding() {
    // 4x4 input, 3x3 kernel, stride 1, padding 1 -> (4+2-3)/1+1 = 4
    assert_eq!(pool2d_output_size(4, 3, 1, 1), Some(4));
}

#[test]
fn test_pool2d_output_size_stride_larger_than_kernel() {
    // 8 input, 2 kernel, stride 3, no padding -> (8-2)/3+1 = 3
    assert_eq!(pool2d_output_size(8, 2, 3, 0), Some(3));
}

#[test]
fn test_pool2d_output_size_1x1_kernel() {
    // 7 input, 1 kernel, stride 1, no padding -> 7
    assert_eq!(pool2d_output_size(7, 1, 1, 0), Some(7));
}

#[test]
fn test_pool2d_output_size_kernel_larger_than_input() {
    // 3 input, 5 kernel -> None
    assert_eq!(pool2d_output_size(3, 5, 1, 0), None);
}

#[test]
fn test_pool2d_output_size_kernel_equals_input() {
    // Global pooling: 7 input, 7 kernel -> 1
    assert_eq!(pool2d_output_size(7, 7, 1, 0), Some(1));
}

#[test]
fn test_pool2d_output_size_typical_resnet() {
    // ResNet: 112x112 -> maxpool 3x3, stride 2, pad 1 -> 56
    assert_eq!(pool2d_output_size(112, 3, 2, 1), Some(56));
}

// ---------------------------------------------------------------------------
// Config validation
// ---------------------------------------------------------------------------

#[test]
fn test_pool_config_default() {
    let c = PtxPool2dConfig::default();
    assert_eq!(c.kernel_h, 2);
    assert_eq!(c.kernel_w, 2);
    assert_eq!(c.stride_h, 2);
    assert_eq!(c.stride_w, 2);
    assert_eq!(c.pad_h, 0);
    assert_eq!(c.pad_w, 0);
    assert!(c.validate().is_ok());
}

#[test]
fn test_pool_config_empty_name() {
    let c = PtxPool2dConfig::new("", 2, 2);
    assert!(c.validate().is_err());
}

#[test]
fn test_pool_config_zero_kernel() {
    let c = PtxPool2dConfig::new("pool", 0, 2);
    assert!(c.validate().is_err());
}

#[test]
fn test_pool_config_zero_stride() {
    let c = PtxPool2dConfig::new("pool", 2, 2).with_stride(0, 0);
    assert!(c.validate().is_err());
}

#[test]
fn test_pool_config_block_too_large() {
    let c = PtxPool2dConfig::new("pool", 2, 2).with_block_size(2048);
    assert!(c.validate().is_err());
}

#[test]
fn test_adaptive_config_empty_name() {
    let c = PtxAdaptiveAvgPool2dConfig::new("", 1, 1);
    assert!(c.validate().is_err());
}

#[test]
fn test_adaptive_config_zero_output() {
    let c = PtxAdaptiveAvgPool2dConfig::new("apool", 0, 1);
    assert!(c.validate().is_err());
}

// ---------------------------------------------------------------------------
// Max pool 2D reference
// ---------------------------------------------------------------------------

#[test]
fn test_max_pool2d_reference_2x2_stride2() {
    // 1x1x4x4 input, 2x2 kernel, stride 2, no padding -> 1x1x2x2
    #[rustfmt::skip]
    let input = vec![
        1.0, 2.0, 3.0, 4.0,
        5.0, 6.0, 7.0, 8.0,
        9.0, 10.0, 11.0, 12.0,
        13.0, 14.0, 15.0, 16.0,
    ];
    let config = PtxPool2dConfig::new("mp", 2, 2);
    let out = max_pool2d_reference(&input, 1, 1, 4, 4, &config);
    assert_eq!(out.len(), 4);
    assert_eq!(out[0], 6.0); // max(1,2,5,6)
    assert_eq!(out[1], 8.0); // max(3,4,7,8)
    assert_eq!(out[2], 14.0); // max(9,10,13,14)
    assert_eq!(out[3], 16.0); // max(11,12,15,16)
}

#[test]
fn test_max_pool2d_reference_3x3_stride1_pad1() {
    // 1x1x3x3 input, 3x3 kernel, stride 1, pad 1 -> 1x1x3x3
    let input: Vec<f32> = (1..=9).map(|x| x as f32).collect();
    let config = PtxPool2dConfig::new("mp", 3, 3)
        .with_stride(1, 1)
        .with_padding(1, 1);
    let out = max_pool2d_reference(&input, 1, 1, 3, 3, &config);
    assert_eq!(out.len(), 9);
    // Center element sees all 9 values, max = 9
    assert_eq!(out[4], 9.0);
}

// ---------------------------------------------------------------------------
// Avg pool 2D reference
// ---------------------------------------------------------------------------

#[test]
fn test_avg_pool2d_reference_2x2_stride2() {
    #[rustfmt::skip]
    let input = vec![
        1.0, 2.0, 3.0, 4.0,
        5.0, 6.0, 7.0, 8.0,
        9.0, 10.0, 11.0, 12.0,
        13.0, 14.0, 15.0, 16.0,
    ];
    let config = PtxPool2dConfig::new("ap", 2, 2);
    let out = avg_pool2d_reference(&input, 1, 1, 4, 4, &config);
    assert_eq!(out.len(), 4);
    assert!((out[0] - 3.5).abs() < 1e-6); // avg(1,2,5,6) = 14/4
    assert!((out[1] - 5.5).abs() < 1e-6); // avg(3,4,7,8) = 22/4
    assert!((out[2] - 11.5).abs() < 1e-6); // avg(9,10,13,14) = 46/4
    assert!((out[3] - 13.5).abs() < 1e-6); // avg(11,12,15,16) = 54/4
}

// ---------------------------------------------------------------------------
// Adaptive avg pool 2D reference
// ---------------------------------------------------------------------------

#[test]
fn test_adaptive_avg_pool2d_reference_global() {
    // Global average pooling: 1x1x4x4 -> 1x1x1x1
    let input: Vec<f32> = (1..=16).map(|x| x as f32).collect();
    let out = adaptive_avg_pool2d_reference(&input, 1, 1, 4, 4, 1, 1);
    assert_eq!(out.len(), 1);
    let expected = (1..=16).sum::<i32>() as f32 / 16.0; // 8.5
    assert!((out[0] - expected).abs() < 1e-6);
}

#[test]
fn test_adaptive_avg_pool2d_reference_2x2() {
    #[rustfmt::skip]
    let input = vec![
        1.0, 2.0, 3.0, 4.0,
        5.0, 6.0, 7.0, 8.0,
        9.0, 10.0, 11.0, 12.0,
        13.0, 14.0, 15.0, 16.0,
    ];
    let out = adaptive_avg_pool2d_reference(&input, 1, 1, 4, 4, 2, 2);
    assert_eq!(out.len(), 4);
    // Top-left: rows 0..2, cols 0..2 -> avg(1,2,5,6) = 3.5
    assert!((out[0] - 3.5).abs() < 1e-6);
}

// ---------------------------------------------------------------------------
// PTX structural checks — max pool 2D
// ---------------------------------------------------------------------------

fn assert_pool_ptx_structure(ptx: &str, kernel_name: &str) {
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
    assert!(ptx.contains("param_output"), "must have output param");
    assert!(
        ptx.contains("param_batch_size"),
        "must have batch_size param"
    );
    assert!(ptx.contains("param_channels"), "must have channels param");
    assert!(ptx.contains("ret;"), "must have ret instruction");
}

#[test]
fn test_max_pool2d_ptx_structure() {
    let config = PtxPool2dConfig::new("maxpool_2x2", 2, 2);
    let ptx = generate_max_pool2d_ptx(&config).unwrap();
    assert_pool_ptx_structure(&ptx, "maxpool_2x2");
    assert!(ptx.contains("MaxPool2d"), "header should mention MaxPool2d");
    assert!(ptx.contains("max.f32"), "must use max.f32 instruction");
}

#[test]
fn test_avg_pool2d_ptx_structure() {
    let config = PtxPool2dConfig::new("avgpool_2x2", 2, 2);
    let ptx = generate_avg_pool2d_ptx(&config).unwrap();
    assert_pool_ptx_structure(&ptx, "avgpool_2x2");
    assert!(ptx.contains("AvgPool2d"), "header should mention AvgPool2d");
    assert!(ptx.contains("div.rn.f32"), "must divide for average");
}

#[test]
fn test_adaptive_avg_pool2d_ptx_structure() {
    let config = PtxAdaptiveAvgPool2dConfig::new("adaptive_avg", 1, 1);
    let ptx = generate_adaptive_avg_pool2d_ptx(&config).unwrap();
    assert!(ptx.contains(".version 6.5"));
    assert!(ptx.contains(".visible .entry adaptive_avg"));
    assert!(ptx.contains("AdaptiveAvgPool2d"));
    assert!(ptx.contains("div.rn.f32"), "must divide for average");
    assert!(ptx.contains("ret;"));
}

// ---------------------------------------------------------------------------
// PTX content checks
// ---------------------------------------------------------------------------

#[test]
fn test_max_pool2d_ptx_has_grid_stride_loop() {
    let config = PtxPool2dConfig::new("mp", 2, 2);
    let ptx = generate_max_pool2d_ptx(&config).unwrap();
    assert!(ptx.contains("POOL_LOOP:"));
    assert!(ptx.contains("POOL_EXIT:"));
}

#[test]
fn test_max_pool2d_ptx_has_kernel_window_loops() {
    let config = PtxPool2dConfig::new("mp", 3, 3);
    let ptx = generate_max_pool2d_ptx(&config).unwrap();
    assert!(ptx.contains("KH_LOOP:"));
    assert!(ptx.contains("KW_LOOP:"));
    assert!(ptx.contains("KH_DONE:"));
    assert!(ptx.contains("KW_DONE:"));
}

#[test]
fn test_pool2d_ptx_has_bounds_check() {
    let config = PtxPool2dConfig::new("mp", 3, 3).with_padding(1, 1);
    let ptx = generate_max_pool2d_ptx(&config).unwrap();
    assert!(ptx.contains("setp.ge.u32"), "must have bounds check");
}

#[test]
fn test_pool2d_ptx_is_pure_ptx() {
    let config = PtxPool2dConfig::new("mp", 2, 2);
    let ptx = generate_max_pool2d_ptx(&config).unwrap();
    assert!(!ptx.contains("__global__"));
    assert!(!ptx.contains("#include"));
}

#[test]
fn test_pool2d_ptx_header_contains_config() {
    let config = PtxPool2dConfig::new("mp", 3, 3)
        .with_stride(2, 2)
        .with_padding(1, 1);
    let ptx = generate_max_pool2d_ptx(&config).unwrap();
    assert!(ptx.contains("kernel=3x3"));
    assert!(ptx.contains("stride=2x2"));
    assert!(ptx.contains("pad=1x1"));
}

#[test]
fn test_pool2d_different_kinds_produce_different_ptx() {
    let config = PtxPool2dConfig::new("pool", 2, 2);
    let max_ptx = generate_max_pool2d_ptx(&config).unwrap();
    let avg_ptx = generate_avg_pool2d_ptx(&config).unwrap();
    assert_ne!(max_ptx, avg_ptx, "max vs avg should produce different PTX");
}

#[test]
fn test_pool2d_custom_sm_target() {
    let config = PtxPool2dConfig::new("mp", 2, 2).with_sm_target("sm_90");
    let ptx = generate_max_pool2d_ptx(&config).unwrap();
    assert!(ptx.contains(".target sm_90"));
}

#[test]
fn test_pool2d_custom_block_size() {
    let config = PtxPool2dConfig::new("mp", 2, 2).with_block_size(128);
    let ptx = generate_max_pool2d_ptx(&config).unwrap();
    assert!(ptx.contains(".reqntid 128"));
}

// ---------------------------------------------------------------------------
// Adaptive pool specific
// ---------------------------------------------------------------------------

#[test]
fn test_adaptive_pool2d_ptx_bakes_output_size() {
    let config = PtxAdaptiveAvgPool2dConfig::new("apool", 7, 7);
    let ptx = generate_adaptive_avg_pool2d_ptx(&config).unwrap();
    assert!(ptx.contains("output=7x7"));
}

#[test]
fn test_adaptive_pool2d_ptx_has_grid_stride() {
    let config = PtxAdaptiveAvgPool2dConfig::new("apool", 1, 1);
    let ptx = generate_adaptive_avg_pool2d_ptx(&config).unwrap();
    assert!(ptx.contains("APOOL_LOOP:"));
    assert!(ptx.contains("APOOL_EXIT:"));
}

// ---------------------------------------------------------------------------
// Launch config
// ---------------------------------------------------------------------------

#[test]
fn test_pool2d_launch_config_basic() {
    // batch=1, channels=64, output=7x7 -> 1*64*7*7 = 3136
    let cfg = ptx_pool2d_launch_config(1, 64, 7, 7);
    assert_eq!(cfg.block.x, 256);
    assert_eq!(cfg.grid.x, 13); // ceil(3136/256)
}

#[test]
fn test_pool2d_launch_config_small() {
    let cfg = ptx_pool2d_launch_config(1, 1, 1, 1);
    assert_eq!(cfg.grid.x, 1);
}

#[test]
fn test_pool2d_launch_config_1d() {
    let cfg = ptx_pool2d_launch_config(2, 32, 14, 14);
    assert_eq!(cfg.grid.y, 1);
    assert_eq!(cfg.grid.z, 1);
    assert_eq!(cfg.block.y, 1);
    assert_eq!(cfg.block.z, 1);
}

// ---------------------------------------------------------------------------
// Edge cases
// ---------------------------------------------------------------------------

#[test]
fn test_max_pool2d_1x1_kernel() {
    // 1x1 pool = identity
    let config = PtxPool2dConfig::new("mp_1x1", 1, 1).with_stride(1, 1);
    let ptx = generate_max_pool2d_ptx(&config).unwrap();
    assert_pool_ptx_structure(&ptx, "mp_1x1");
    assert!(ptx.contains("kernel=1x1"));
}

#[test]
fn test_avg_pool2d_1x1_kernel() {
    let config = PtxPool2dConfig::new("ap_1x1", 1, 1).with_stride(1, 1);
    let ptx = generate_avg_pool2d_ptx(&config).unwrap();
    assert_pool_ptx_structure(&ptx, "ap_1x1");
}

#[test]
fn test_max_pool2d_reference_1x1_is_identity() {
    let input = vec![1.0, 2.0, 3.0, 4.0];
    let config = PtxPool2dConfig::new("mp", 1, 1).with_stride(1, 1);
    let out = max_pool2d_reference(&input, 1, 1, 2, 2, &config);
    assert_eq!(out, input);
}

#[test]
fn test_avg_pool2d_reference_1x1_is_identity() {
    let input = vec![1.0, 2.0, 3.0, 4.0];
    let config = PtxPool2dConfig::new("ap", 1, 1).with_stride(1, 1);
    let out = avg_pool2d_reference(&input, 1, 1, 2, 2, &config);
    assert_eq!(out, input);
}

#[test]
fn test_max_pool2d_reference_global() {
    // Global max pool: kernel = input size
    let input: Vec<f32> = (1..=16).map(|x| x as f32).collect();
    let config = PtxPool2dConfig::new("mp", 4, 4).with_stride(4, 4);
    let out = max_pool2d_reference(&input, 1, 1, 4, 4, &config);
    assert_eq!(out.len(), 1);
    assert_eq!(out[0], 16.0);
}

#[test]
fn test_max_pool2d_reference_batched() {
    // 2 batches, 1 channel, 2x2 input, 2x2 kernel -> 2x1x1x1
    let input = vec![1.0, 2.0, 3.0, 4.0, 10.0, 20.0, 30.0, 40.0];
    let config = PtxPool2dConfig::new("mp", 2, 2);
    let out = max_pool2d_reference(&input, 2, 1, 2, 2, &config);
    assert_eq!(out.len(), 2);
    assert_eq!(out[0], 4.0);
    assert_eq!(out[1], 40.0);
}

#[test]
fn test_adaptive_avg_pool2d_reference_same_size() {
    // When output_h == h_in, should be identity
    let input = vec![1.0, 2.0, 3.0, 4.0];
    let out = adaptive_avg_pool2d_reference(&input, 1, 1, 2, 2, 2, 2);
    assert_eq!(out, input);
}

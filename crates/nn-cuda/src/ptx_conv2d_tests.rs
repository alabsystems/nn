// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Extended tests for PTX conv2d kernel generation.
//!
//! Covers non-square kernels, stride/padding combos, output size formula,
//! builder chaining, constant ranges, large kernels, 1x1 edge cases,
//! and PTX structural invariants.

use super::*;

// ---------------------------------------------------------------------------
// Constants validation
// ---------------------------------------------------------------------------

#[test]
fn test_constants_are_valid_ranges() {
    assert!(PTX_CONV2D_BLOCK_W >= PTX_CONV2D_MIN_BLOCK);
    assert!(PTX_CONV2D_BLOCK_W <= PTX_CONV2D_MAX_BLOCK);
    assert!(PTX_CONV2D_BLOCK_H >= PTX_CONV2D_MIN_BLOCK);
    assert!(PTX_CONV2D_BLOCK_H <= PTX_CONV2D_MAX_BLOCK);
    assert!(PTX_CONV2D_MIN_BLOCK > 0);
    assert!(PTX_CONV2D_MAX_BLOCK >= PTX_CONV2D_MIN_BLOCK);
    // Default block product must not exceed CUDA thread limit.
    assert!(PTX_CONV2D_BLOCK_W * PTX_CONV2D_BLOCK_H <= 1024);
}

#[test]
fn test_min_max_block_constants_values() {
    assert_eq!(PTX_CONV2D_MIN_BLOCK, 4);
    assert_eq!(PTX_CONV2D_MAX_BLOCK, 32);
    assert_eq!(PTX_CONV2D_BLOCK_W, 16);
    assert_eq!(PTX_CONV2D_BLOCK_H, 16);
}

// ---------------------------------------------------------------------------
// Config construction & builder chaining
// ---------------------------------------------------------------------------

#[test]
fn test_config_new_sets_defaults_correctly() {
    let c = PtxConv2dConfig::new("nn_kernel", 5, 7);
    assert_eq!(c.kernel_name, "nn_kernel");
    assert_eq!(c.kernel_h, 5);
    assert_eq!(c.kernel_w, 7);
    assert_eq!(c.stride_h, 1);
    assert_eq!(c.stride_w, 1);
    assert_eq!(c.pad_h, 0);
    assert_eq!(c.pad_w, 0);
    assert_eq!(c.dilation_h, 1);
    assert_eq!(c.dilation_w, 1);
    assert!(!c.use_bias);
    assert_eq!(c.block_w, PTX_CONV2D_BLOCK_W);
    assert_eq!(c.block_h, PTX_CONV2D_BLOCK_H);
    assert_eq!(c.sm_target, "sm_80");
}

#[test]
fn test_config_builder_chaining() {
    let c = PtxConv2dConfig::new("chained", 5, 5)
        .with_stride(2, 3)
        .with_padding(1, 2)
        .with_dilation(2, 1)
        .with_bias(true)
        .with_block_size(8, 8)
        .with_sm_target("sm_90");
    assert_eq!(c.kernel_name, "chained");
    assert_eq!(c.kernel_h, 5);
    assert_eq!(c.kernel_w, 5);
    assert_eq!(c.stride_h, 2);
    assert_eq!(c.stride_w, 3);
    assert_eq!(c.pad_h, 1);
    assert_eq!(c.pad_w, 2);
    assert_eq!(c.dilation_h, 2);
    assert_eq!(c.dilation_w, 1);
    assert!(c.use_bias);
    assert_eq!(c.block_h, 8);
    assert_eq!(c.block_w, 8);
    assert_eq!(c.sm_target, "sm_90");
    assert!(c.validate().is_ok());
}

#[test]
fn test_config_builder_override_stride_both_dims() {
    let c = PtxConv2dConfig::new("k", 3, 3).with_stride(4, 2);
    assert_eq!(c.stride_h, 4);
    assert_eq!(c.stride_w, 2);
}

#[test]
fn test_config_builder_override_padding_both_dims() {
    let c = PtxConv2dConfig::new("k", 3, 3).with_padding(3, 5);
    assert_eq!(c.pad_h, 3);
    assert_eq!(c.pad_w, 5);
}

#[test]
fn test_config_builder_override_dilation_both_dims() {
    let c = PtxConv2dConfig::new("k", 3, 3).with_dilation(3, 4);
    assert_eq!(c.dilation_h, 3);
    assert_eq!(c.dilation_w, 4);
}

// ---------------------------------------------------------------------------
// Validation edge cases
// ---------------------------------------------------------------------------

#[test]
fn test_validate_zero_kernel_w() {
    let c = PtxConv2dConfig::new("k", 3, 0);
    assert!(c.validate().is_err());
}

#[test]
fn test_validate_zero_stride_w() {
    let c = PtxConv2dConfig::new("k", 3, 3).with_stride(1, 0);
    assert!(c.validate().is_err());
}

#[test]
fn test_validate_zero_dilation_w() {
    let c = PtxConv2dConfig::new("k", 3, 3).with_dilation(1, 0);
    assert!(c.validate().is_err());
}

#[test]
fn test_validate_block_h_below_min() {
    let c = PtxConv2dConfig::new("k", 3, 3).with_block_size(3, 16);
    assert!(c.validate().is_err());
}

#[test]
fn test_validate_block_w_below_min() {
    let c = PtxConv2dConfig::new("k", 3, 3).with_block_size(16, 3);
    assert!(c.validate().is_err());
}

#[test]
fn test_validate_block_h_above_max() {
    let c = PtxConv2dConfig::new("k", 3, 3).with_block_size(33, 16);
    assert!(c.validate().is_err());
}

#[test]
fn test_validate_block_w_above_max() {
    let c = PtxConv2dConfig::new("k", 3, 3).with_block_size(16, 33);
    assert!(c.validate().is_err());
}

#[test]
fn test_validate_min_block_size_passes() {
    let c =
        PtxConv2dConfig::new("k", 3, 3).with_block_size(PTX_CONV2D_MIN_BLOCK, PTX_CONV2D_MIN_BLOCK);
    assert!(c.validate().is_ok());
}

#[test]
fn test_validate_max_block_size_passes() {
    let c =
        PtxConv2dConfig::new("k", 3, 3).with_block_size(PTX_CONV2D_MAX_BLOCK, PTX_CONV2D_MAX_BLOCK);
    // 32 * 32 = 1024 <= 1024, should pass.
    assert!(c.validate().is_ok());
}

#[test]
fn test_validate_large_kernel_passes() {
    let c = PtxConv2dConfig::new("k", 11, 11);
    assert!(c.validate().is_ok());
}

#[test]
fn test_validate_kernel_1x1_passes() {
    let c = PtxConv2dConfig::new("k", 1, 1);
    assert!(c.validate().is_ok());
}

// ---------------------------------------------------------------------------
// Non-square kernels
// ---------------------------------------------------------------------------

#[test]
fn test_non_square_kernel_3x5() {
    let c = PtxConv2dConfig::new("conv_3x5", 3, 5);
    assert!(c.validate().is_ok());
    assert!(!c.is_pointwise());
    assert_eq!(c.effective_kernel_h(), 3);
    assert_eq!(c.effective_kernel_w(), 5);
    // tile_h = (16-1)*1 + 3 = 18, tile_w = (16-1)*1 + 5 = 20
    assert_eq!(c.input_tile_h(), 18);
    assert_eq!(c.input_tile_w(), 20);
    assert_eq!(c.shared_memory_bytes(), 18 * 20 * 4);
}

#[test]
fn test_non_square_kernel_1x7() {
    let c = PtxConv2dConfig::new("conv_1x7", 1, 7);
    assert!(!c.is_pointwise()); // kW > 1, so not pointwise
    assert_eq!(c.effective_kernel_h(), 1);
    assert_eq!(c.effective_kernel_w(), 7);
    // tile_h = (16-1)*1 + 1 = 16, tile_w = (16-1)*1 + 7 = 22
    assert_eq!(c.input_tile_h(), 16);
    assert_eq!(c.input_tile_w(), 22);
}

#[test]
fn test_non_square_kernel_7x1() {
    let c = PtxConv2dConfig::new("conv_7x1", 7, 1);
    assert!(!c.is_pointwise());
    assert_eq!(c.effective_kernel_h(), 7);
    assert_eq!(c.effective_kernel_w(), 1);
    // tile_h = (16-1)*1 + 7 = 22, tile_w = (16-1)*1 + 1 = 16
    assert_eq!(c.input_tile_h(), 22);
    assert_eq!(c.input_tile_w(), 16);
}

#[test]
fn test_non_square_kernel_ptx_generation() {
    let c = PtxConv2dConfig::new("conv_3x5", 3, 5);
    let ptx = emit_ptx_conv2d(&c).unwrap();
    assert!(ptx.contains(".visible .entry conv_3x5"));
    assert!(ptx.contains(".shared .align 4 .f32 input_tile["));
    // tile_size = 18 * 20 = 360
    assert!(
        ptx.contains("input_tile[360]"),
        "non-square 3x5 should produce tile[360]"
    );
    assert!(ptx.contains("fma.rn.f32"));
}

#[test]
fn test_non_square_kernel_1x7_ptx_uses_shared_memory() {
    let c = PtxConv2dConfig::new("conv_1x7", 1, 7);
    let ptx = emit_ptx_conv2d(&c).unwrap();
    // 1x7 is NOT pointwise, so should use shared memory.
    assert!(ptx.contains(".shared .align 4"));
    assert!(ptx.contains("bar.sync"));
}

#[test]
fn test_non_square_kernels_produce_different_ptx() {
    let ptx_3x5 = emit_ptx_conv2d(&PtxConv2dConfig::new("conv", 3, 5)).unwrap();
    let ptx_5x3 = emit_ptx_conv2d(&PtxConv2dConfig::new("conv", 5, 3)).unwrap();
    assert_ne!(ptx_3x5, ptx_5x3, "3x5 and 5x3 must produce different PTX");
}

// ---------------------------------------------------------------------------
// Square kernels: 5x5 and 7x7
// ---------------------------------------------------------------------------

#[test]
fn test_square_kernel_5x5_tile() {
    let c = PtxConv2dConfig::new("conv_5x5", 5, 5);
    // tile_h = (16-1)*1 + 5 = 20, tile_w = 20
    assert_eq!(c.input_tile_h(), 20);
    assert_eq!(c.input_tile_w(), 20);
    assert_eq!(c.shared_memory_bytes(), 20 * 20 * 4);
}

#[test]
fn test_square_kernel_7x7_tile() {
    let c = PtxConv2dConfig::new("conv_7x7", 7, 7);
    // tile_h = (16-1)*1 + 7 = 22, tile_w = 22
    assert_eq!(c.input_tile_h(), 22);
    assert_eq!(c.input_tile_w(), 22);
    assert_eq!(c.shared_memory_bytes(), 22 * 22 * 4);
}

#[test]
fn test_square_kernel_5x5_ptx_generation() {
    let c = PtxConv2dConfig::new("conv_5x5", 5, 5).with_padding(2, 2);
    let ptx = emit_ptx_conv2d(&c).unwrap();
    assert!(ptx.contains(".visible .entry conv_5x5"));
    assert!(ptx.contains("input_tile[400]")); // 20 * 20
    assert!(ptx.contains("fma.rn.f32"));
    assert!(ptx.contains("bar.sync"));
}

#[test]
fn test_square_kernel_7x7_ptx_generation() {
    let c = PtxConv2dConfig::new("conv_7x7", 7, 7).with_padding(3, 3);
    let ptx = emit_ptx_conv2d(&c).unwrap();
    assert!(ptx.contains(".visible .entry conv_7x7"));
    assert!(ptx.contains("input_tile[484]")); // 22 * 22
    assert!(ptx.contains("fma.rn.f32"));
}

// ---------------------------------------------------------------------------
// Stride and padding variations
// ---------------------------------------------------------------------------

#[test]
fn test_stride_2_padding_1_3x3() {
    let c = PtxConv2dConfig::new("conv_s2p1", 3, 3)
        .with_stride(2, 2)
        .with_padding(1, 1);
    assert!(c.validate().is_ok());
    // tile_h = (16-1)*2 + 3 = 33, tile_w = 33
    assert_eq!(c.input_tile_h(), 33);
    assert_eq!(c.input_tile_w(), 33);
    let ptx = emit_ptx_conv2d(&c).unwrap();
    assert!(ptx.contains("input_tile[1089]")); // 33 * 33
}

#[test]
fn test_asymmetric_stride() {
    let c = PtxConv2dConfig::new("conv_asym_s", 3, 3).with_stride(2, 1);
    assert!(c.validate().is_ok());
    // tile_h = (16-1)*2 + 3 = 33, tile_w = (16-1)*1 + 3 = 18
    assert_eq!(c.input_tile_h(), 33);
    assert_eq!(c.input_tile_w(), 18);
    let ptx = emit_ptx_conv2d(&c).unwrap();
    assert!(!ptx.is_empty());
    assert!(ptx.contains("fma.rn.f32"));
}

#[test]
fn test_asymmetric_padding() {
    let c = PtxConv2dConfig::new("conv_asym_p", 3, 5).with_padding(1, 2);
    assert!(c.validate().is_ok());
    let ptx = emit_ptx_conv2d(&c).unwrap();
    assert!(ptx.contains(".visible .entry conv_asym_p"));
}

#[test]
fn test_large_padding() {
    let c = PtxConv2dConfig::new("conv_big_pad", 3, 3).with_padding(10, 10);
    assert!(c.validate().is_ok());
    let ptx = emit_ptx_conv2d(&c).unwrap();
    assert!(!ptx.is_empty());
}

#[test]
fn test_stride_4_kernel_7x7() {
    let c = PtxConv2dConfig::new("conv_s4k7", 7, 7)
        .with_stride(4, 4)
        .with_padding(3, 3);
    assert!(c.validate().is_ok());
    // tile_h = (16-1)*4 + 7 = 67, tile_w = 67
    assert_eq!(c.input_tile_h(), 67);
    assert_eq!(c.input_tile_w(), 67);
    let ptx = emit_ptx_conv2d(&c).unwrap();
    assert!(ptx.contains("input_tile[4489]")); // 67 * 67
}

// ---------------------------------------------------------------------------
// Dilation
// ---------------------------------------------------------------------------

#[test]
fn test_dilation_effective_kernel_non_square() {
    let c = PtxConv2dConfig::new("k", 3, 5).with_dilation(2, 3);
    // effective_h = (3-1)*2 + 1 = 5
    // effective_w = (5-1)*3 + 1 = 13
    assert_eq!(c.effective_kernel_h(), 5);
    assert_eq!(c.effective_kernel_w(), 13);
}

#[test]
fn test_dilation_tile_size() {
    let c = PtxConv2dConfig::new("k", 3, 3).with_dilation(3, 3);
    // effective = (3-1)*3 + 1 = 7 each
    // tile_h = (16-1)*1 + 7 = 22, tile_w = 22
    assert_eq!(c.input_tile_h(), 22);
    assert_eq!(c.input_tile_w(), 22);
}

#[test]
fn test_dilation_with_stride_tile_size() {
    let c = PtxConv2dConfig::new("k", 3, 3)
        .with_dilation(2, 2)
        .with_stride(2, 2);
    // effective = (3-1)*2+1 = 5 each
    // tile_h = (16-1)*2 + 5 = 35, tile_w = 35
    assert_eq!(c.input_tile_h(), 35);
    assert_eq!(c.input_tile_w(), 35);
}

#[test]
fn test_dilation_ptx_contains_comment() {
    let c = PtxConv2dConfig::new("conv_d2", 3, 3).with_dilation(2, 2);
    let ptx = emit_ptx_conv2d(&c).unwrap();
    assert!(ptx.contains("dilation: 2x2"));
}

// ---------------------------------------------------------------------------
// 1x1 pointwise edge cases
// ---------------------------------------------------------------------------

#[test]
fn test_1x1_with_bias() {
    let c = PtxConv2dConfig::new("conv_1x1_bias", 1, 1).with_bias(true);
    let ptx = emit_ptx_conv2d(&c).unwrap();
    assert!(ptx.contains("param_bias"));
    assert!(!ptx.contains(".shared"));
    assert!(!ptx.contains("bar.sync"));
}

#[test]
fn test_1x1_with_stride() {
    let c = PtxConv2dConfig::new("conv_1x1_s2", 1, 1).with_stride(2, 2);
    assert!(c.is_pointwise());
    assert_eq!(c.shared_memory_bytes(), 0);
    let ptx = emit_ptx_conv2d(&c).unwrap();
    assert!(!ptx.is_empty());
    assert!(ptx.contains("fma.rn.f32"));
    assert!(!ptx.contains(".shared"));
}

#[test]
fn test_1x1_with_padding() {
    let c = PtxConv2dConfig::new("conv_1x1_p1", 1, 1).with_padding(1, 1);
    assert!(c.is_pointwise());
    let ptx = emit_ptx_conv2d(&c).unwrap();
    assert!(!ptx.is_empty());
}

#[test]
fn test_1x1_effective_kernel_ignores_dilation() {
    // 1x1 with dilation should still be effective 1x1.
    let c = PtxConv2dConfig::new("k", 1, 1).with_dilation(4, 4);
    assert_eq!(c.effective_kernel_h(), 1); // (1-1)*4 + 1 = 1
    assert_eq!(c.effective_kernel_w(), 1);
    assert!(c.is_pointwise());
}

#[test]
fn test_1x1_pointwise_comment() {
    let c = PtxConv2dConfig::new("conv_1x1", 1, 1);
    let ptx = emit_ptx_conv2d(&c).unwrap();
    assert!(
        ptx.contains("pointwise"),
        "1x1 kernel should have 'pointwise' in comment"
    );
}

// ---------------------------------------------------------------------------
// Bias variations
// ---------------------------------------------------------------------------

#[test]
fn test_bias_3x3_ptx_has_bias_load() {
    let c = PtxConv2dConfig::new("conv_b", 3, 3)
        .with_padding(1, 1)
        .with_bias(true);
    let ptx = emit_ptx_conv2d(&c).unwrap();
    assert!(ptx.contains("param_bias"));
    assert!(ptx.contains("Add bias"));
}

#[test]
fn test_no_bias_ptx_has_no_bias_section() {
    let c = PtxConv2dConfig::new("conv_nb", 3, 3).with_bias(false);
    let ptx = emit_ptx_conv2d(&c).unwrap();
    assert!(!ptx.contains("param_bias"));
    assert!(!ptx.contains("Add bias"));
}

#[test]
fn test_bias_changes_ptx_output() {
    let ptx_no_bias =
        emit_ptx_conv2d(&PtxConv2dConfig::new("conv", 3, 3).with_bias(false)).unwrap();
    let ptx_bias = emit_ptx_conv2d(&PtxConv2dConfig::new("conv", 3, 3).with_bias(true)).unwrap();
    assert_ne!(ptx_no_bias, ptx_bias);
    // Bias version should be longer (has extra instructions).
    assert!(ptx_bias.len() > ptx_no_bias.len());
}

// ---------------------------------------------------------------------------
// Output size formula: (input_size + 2*pad - effective_kernel) / stride + 1
// (tested via launch config which depends on H_out/W_out)
// ---------------------------------------------------------------------------

/// Local helper to compute conv2d output size (returns raw usize, not Option).
fn local_conv2d_output_size(
    input_size: usize,
    pad: usize,
    kernel: usize,
    stride: usize,
    dilation: usize,
) -> usize {
    let effective_kernel = (kernel - 1) * dilation + 1;
    (input_size + 2 * pad - effective_kernel) / stride + 1
}

#[test]
fn test_output_size_formula_basic() {
    // 32x32 input, 3x3 kernel, pad=1, stride=1 -> 32x32
    assert_eq!(local_conv2d_output_size(32, 1, 3, 1, 1), 32);
}

#[test]
fn test_output_size_formula_no_padding() {
    // 32x32 input, 3x3 kernel, pad=0, stride=1 -> 30x30
    assert_eq!(local_conv2d_output_size(32, 0, 3, 1, 1), 30);
}

#[test]
fn test_output_size_formula_stride_2() {
    // 32x32 input, 3x3 kernel, pad=1, stride=2 -> 16x16
    assert_eq!(local_conv2d_output_size(32, 1, 3, 2, 1), 16);
}

#[test]
fn test_output_size_formula_5x5_pad2() {
    // 32x32 input, 5x5 kernel, pad=2, stride=1 -> 32x32
    assert_eq!(local_conv2d_output_size(32, 2, 5, 1, 1), 32);
}

#[test]
fn test_output_size_formula_7x7_stride2_pad3() {
    // 224 input, 7x7 kernel, pad=3, stride=2 -> 112 (like ResNet stem)
    assert_eq!(local_conv2d_output_size(224, 3, 7, 2, 1), 112);
}

#[test]
fn test_output_size_formula_1x1() {
    // 1x1 kernel, pad=0, stride=1 -> same size
    assert_eq!(local_conv2d_output_size(64, 0, 1, 1, 1), 64);
}

#[test]
fn test_output_size_formula_1x1_stride2() {
    // 1x1 kernel, pad=0, stride=2 -> halved
    assert_eq!(local_conv2d_output_size(64, 0, 1, 2, 1), 32);
}

#[test]
fn test_output_size_formula_dilation() {
    // 3x3 kernel, dilation=2 -> effective 5x5
    // 32 input, pad=2, stride=1 -> (32 + 4 - 5)/1 + 1 = 32
    assert_eq!(local_conv2d_output_size(32, 2, 3, 1, 2), 32);
}

#[test]
fn test_output_size_formula_dilation_no_pad() {
    // 3x3 kernel, dilation=2 -> effective 5x5
    // 32 input, pad=0, stride=1 -> (32 + 0 - 5)/1 + 1 = 28
    assert_eq!(local_conv2d_output_size(32, 0, 3, 1, 2), 28);
}

// ---------------------------------------------------------------------------
// Launch config
// ---------------------------------------------------------------------------

#[test]
fn test_launch_config_exact_divisible() {
    let c = PtxConv2dConfig::new("conv", 3, 3);
    // H_out=16, W_out=16 -> grid = [1, 1, batch*c_out]
    let (grid, block) = ptx_conv2d_launch_config(16, 16, 1, 1, &c);
    assert_eq!(grid, [1, 1, 1]);
    assert_eq!(block, [16, 16, 1]);
}

#[test]
fn test_launch_config_round_up() {
    let c = PtxConv2dConfig::new("conv", 3, 3);
    // H_out=17, W_out=17 -> grid_x = ceil(17/16)=2, grid_y = ceil(17/16)=2
    let (grid, _) = ptx_conv2d_launch_config(17, 17, 1, 1, &c);
    assert_eq!(grid, [2, 2, 1]);
}

#[test]
fn test_launch_config_single_element() {
    let c = PtxConv2dConfig::new("conv", 3, 3);
    let (grid, _) = ptx_conv2d_launch_config(1, 1, 1, 1, &c);
    assert_eq!(grid, [1, 1, 1]);
}

#[test]
fn test_launch_config_large_batch() {
    let c = PtxConv2dConfig::new("conv", 3, 3);
    let (grid, _) = ptx_conv2d_launch_config(32, 32, 8, 64, &c);
    assert_eq!(grid[2], 8 * 64);
}

#[test]
fn test_launch_config_block_always_positive() {
    let c = PtxConv2dConfig::new("conv", 3, 3).with_block_size(4, 4);
    let (grid, block) = ptx_conv2d_launch_config(100, 100, 1, 1, &c);
    assert!(grid[0] > 0);
    assert!(grid[1] > 0);
    assert!(grid[2] > 0);
    assert_eq!(block, [4, 4, 1]);
}

#[test]
fn test_launch_config_matches_resnet_stem() {
    // ResNet stem: 224->112 via 7x7 stride 2 pad 3, out_channels=64
    let c = PtxConv2dConfig::new("conv", 7, 7)
        .with_stride(2, 2)
        .with_padding(3, 3);
    let h_out = local_conv2d_output_size(224, 3, 7, 2, 1); // 112
    let w_out = local_conv2d_output_size(224, 3, 7, 2, 1); // 112
    assert_eq!(h_out, 112);
    assert_eq!(w_out, 112);
    let (grid, block) = ptx_conv2d_launch_config(h_out, w_out, 1, 64, &c);
    assert_eq!(grid[0], 7); // ceil(112/16)
    assert_eq!(grid[1], 7); // ceil(112/16)
    assert_eq!(grid[2], 64);
    assert_eq!(block, [16, 16, 1]);
}

// ---------------------------------------------------------------------------
// threads_per_block
// ---------------------------------------------------------------------------

#[test]
fn test_threads_per_block_default() {
    let c = PtxConv2dConfig::default();
    assert_eq!(c.threads_per_block(), 256);
}

#[test]
fn test_threads_per_block_custom() {
    let c = PtxConv2dConfig::new("k", 3, 3).with_block_size(8, 32);
    assert_eq!(c.threads_per_block(), 256);
}

#[test]
fn test_threads_per_block_min() {
    let c =
        PtxConv2dConfig::new("k", 3, 3).with_block_size(PTX_CONV2D_MIN_BLOCK, PTX_CONV2D_MIN_BLOCK);
    assert_eq!(
        c.threads_per_block(),
        PTX_CONV2D_MIN_BLOCK * PTX_CONV2D_MIN_BLOCK
    );
}

// ---------------------------------------------------------------------------
// PTX structural: thread indexing via tid/ctaid
// ---------------------------------------------------------------------------

#[test]
fn test_ptx_conv2d_uses_thread_block_indices() {
    let ptx = emit_ptx_conv2d_default("conv").unwrap();
    // PTX uses %tid (threadIdx) and %ctaid (blockIdx)
    assert!(ptx.contains("%tid.x"), "must use threadIdx.x");
    assert!(ptx.contains("%tid.y"), "must use threadIdx.y");
    assert!(ptx.contains("%ctaid.x"), "must use blockIdx.x");
    assert!(ptx.contains("%ctaid.y"), "must use blockIdx.y");
    assert!(ptx.contains("%ctaid.z"), "must use blockIdx.z");
}

#[test]
fn test_ptx_conv2d_1x1_uses_thread_block_indices() {
    let c = PtxConv2dConfig::new("conv1x1", 1, 1);
    let ptx = emit_ptx_conv2d(&c).unwrap();
    assert!(ptx.contains("%tid.x"));
    assert!(ptx.contains("%tid.y"));
    assert!(ptx.contains("%ctaid.x"));
    assert!(ptx.contains("%ctaid.y"));
    assert!(ptx.contains("%ctaid.z"));
}

// ---------------------------------------------------------------------------
// PTX structural: register declarations
// ---------------------------------------------------------------------------

#[test]
fn test_ptx_register_declarations() {
    let ptx = emit_ptx_conv2d_default("conv").unwrap();
    assert!(ptx.contains(".reg .u32  %r<32>"));
    assert!(ptx.contains(".reg .f32  %f<8>"));
    assert!(ptx.contains(".reg .u64  %rd<12>"));
    assert!(ptx.contains(".reg .pred %p<6>"));
}

// ---------------------------------------------------------------------------
// PTX structural: loop labels
// ---------------------------------------------------------------------------

#[test]
fn test_ptx_conv2d_general_has_kernel_loops() {
    let ptx = emit_ptx_conv2d_default("conv").unwrap();
    assert!(ptx.contains("IC_LOOP:"));
    assert!(ptx.contains("IC_DONE:"));
    assert!(ptx.contains("KH_LOOP:"));
    assert!(ptx.contains("KH_DONE:"));
    assert!(ptx.contains("KW_LOOP:"));
    assert!(ptx.contains("KW_DONE:"));
    assert!(ptx.contains("TILE_LOAD_LOOP:"));
    assert!(ptx.contains("TILE_LOAD_DONE:"));
}

#[test]
fn test_ptx_conv2d_1x1_has_ic_loop_only() {
    let c = PtxConv2dConfig::new("conv1x1", 1, 1);
    let ptx = emit_ptx_conv2d(&c).unwrap();
    assert!(ptx.contains("IC_LOOP:"));
    assert!(ptx.contains("IC_DONE:"));
    // Should NOT have KH/KW loops or tile loading.
    assert!(!ptx.contains("KH_LOOP:"));
    assert!(!ptx.contains("KW_LOOP:"));
    assert!(!ptx.contains("TILE_LOAD_LOOP:"));
}

// ---------------------------------------------------------------------------
// PTX structural: KERNEL_EXIT label and ret
// ---------------------------------------------------------------------------

#[test]
fn test_ptx_conv2d_has_kernel_exit() {
    let ptx = emit_ptx_conv2d_default("conv").unwrap();
    assert!(ptx.contains("KERNEL_EXIT:"));
    assert!(ptx.contains("ret;"));
}

#[test]
fn test_ptx_conv2d_1x1_has_kernel_exit() {
    let c = PtxConv2dConfig::new("conv1x1", 1, 1);
    let ptx = emit_ptx_conv2d(&c).unwrap();
    assert!(ptx.contains("KERNEL_EXIT:"));
    assert!(ptx.contains("ret;"));
}

// ---------------------------------------------------------------------------
// PTX output is non-empty for all valid configs
// ---------------------------------------------------------------------------

#[test]
fn test_ptx_non_empty_for_various_configs() {
    let configs = vec![
        PtxConv2dConfig::new("c1", 1, 1),
        PtxConv2dConfig::new("c2", 3, 3),
        PtxConv2dConfig::new("c3", 5, 5).with_padding(2, 2),
        PtxConv2dConfig::new("c4", 7, 7)
            .with_stride(2, 2)
            .with_padding(3, 3),
        PtxConv2dConfig::new("c5", 3, 5).with_dilation(2, 1),
        PtxConv2dConfig::new("c6", 1, 1)
            .with_bias(true)
            .with_stride(2, 2),
        PtxConv2dConfig::new("c7", 3, 3).with_block_size(8, 8),
        PtxConv2dConfig::new("c8", 11, 11),
    ];
    for config in &configs {
        let ptx = emit_ptx_conv2d(config).unwrap();
        assert!(
            !ptx.is_empty(),
            "PTX must be non-empty for config {:?}",
            config.kernel_name
        );
        assert!(
            ptx.len() > 100,
            "PTX must be substantial (got {} bytes) for {}",
            ptx.len(),
            config.kernel_name
        );
    }
}

// ---------------------------------------------------------------------------
// Custom block size affects reqntid and shared memory
// ---------------------------------------------------------------------------

#[test]
fn test_custom_block_4x4() {
    let c = PtxConv2dConfig::new("conv_4x4b", 3, 3).with_block_size(4, 4);
    let ptx = emit_ptx_conv2d(&c).unwrap();
    assert!(ptx.contains(".reqntid 4, 4"));
    // tile_h = (4-1)*1 + 3 = 6, tile_w = 6, size = 36
    assert!(ptx.contains("input_tile[36]"));
}

#[test]
fn test_custom_block_32x4() {
    let c = PtxConv2dConfig::new("conv_32x4b", 3, 3).with_block_size(32, 4);
    assert!(c.validate().is_ok());
    let ptx = emit_ptx_conv2d(&c).unwrap();
    // Note: reqntid uses (block_w, block_h) = (4, 32) order in PTX.
    // Looking at emit code: .reqntid {block_w}, {block_h}
    assert!(ptx.contains(".reqntid 4, 32"));
}

// ---------------------------------------------------------------------------
// SM target variations
// ---------------------------------------------------------------------------

#[test]
fn test_sm_targets() {
    for target in &["sm_70", "sm_75", "sm_80", "sm_86", "sm_90"] {
        let c = PtxConv2dConfig::new("conv", 3, 3).with_sm_target(target);
        let ptx = emit_ptx_conv2d(&c).unwrap();
        assert!(
            ptx.contains(&format!(".target {target}")),
            "PTX must contain target {target}"
        );
    }
}

// ---------------------------------------------------------------------------
// emit_ptx_conv2d_default specifics
// ---------------------------------------------------------------------------

#[test]
fn test_emit_default_produces_valid_ptx() {
    let ptx = emit_ptx_conv2d_default("test_default").unwrap();
    assert!(ptx.contains(".visible .entry test_default"));
    assert!(ptx.contains(".version"));
    assert!(ptx.contains(".target"));
    assert!(ptx.contains("ret;"));
}

#[test]
fn test_emit_default_uses_3x3_pad1() {
    let ptx = emit_ptx_conv2d_default("conv").unwrap();
    // 3x3 with pad=1 on block 16: tile = 18x18 = 324
    assert!(ptx.contains("input_tile[324]"));
    assert!(ptx.contains("pad: 1x1"));
}

// ---------------------------------------------------------------------------
// Error paths
// ---------------------------------------------------------------------------

#[test]
fn test_emit_rejects_invalid_config() {
    let c = PtxConv2dConfig::new("", 3, 3);
    assert!(emit_ptx_conv2d(&c).is_err());
}

#[test]
fn test_emit_rejects_zero_kernel() {
    let c = PtxConv2dConfig::new("k", 0, 0);
    assert!(emit_ptx_conv2d(&c).is_err());
}

#[test]
fn test_emit_rejects_zero_stride() {
    let c = PtxConv2dConfig::new("k", 3, 3).with_stride(0, 0);
    assert!(emit_ptx_conv2d(&c).is_err());
}

#[test]
fn test_emit_rejects_zero_dilation() {
    let c = PtxConv2dConfig::new("k", 3, 3).with_dilation(0, 0);
    assert!(emit_ptx_conv2d(&c).is_err());
}

#[test]
fn test_emit_rejects_block_out_of_range() {
    let c = PtxConv2dConfig::new("k", 3, 3).with_block_size(2, 2);
    assert!(emit_ptx_conv2d(&c).is_err());
}

// ---------------------------------------------------------------------------
// Stride > kernel_size edge cases
// ---------------------------------------------------------------------------

#[test]
fn test_stride_greater_than_kernel_size() {
    // stride=4 with 3x3 kernel: output pixels skip over kernel-size gaps.
    let c = PtxConv2dConfig::new("conv_stride_gt_k", 3, 3).with_stride(4, 4);
    assert!(c.validate().is_ok());
    // tile_h = (16-1)*4 + 3 = 63, tile_w = 63
    assert_eq!(c.input_tile_h(), 63);
    assert_eq!(c.input_tile_w(), 63);
    let ptx = emit_ptx_conv2d(&c).unwrap();
    assert!(ptx.contains("input_tile[3969]")); // 63 * 63
    assert!(ptx.contains("fma.rn.f32"));
}

#[test]
fn test_stride_much_larger_than_kernel() {
    // stride=8, kernel=1x1: huge stride with pointwise.
    let c = PtxConv2dConfig::new("conv_s8_k1", 1, 1).with_stride(8, 8);
    assert!(c.is_pointwise());
    assert_eq!(c.shared_memory_bytes(), 0);
    let ptx = emit_ptx_conv2d(&c).unwrap();
    assert!(ptx.contains("pointwise"));
    assert!(!ptx.contains(".shared"));
}

#[test]
fn test_stride_equals_kernel_size() {
    // stride=3 with 3x3 kernel: non-overlapping.
    let c = PtxConv2dConfig::new("conv_s_eq_k", 3, 3).with_stride(3, 3);
    assert!(c.validate().is_ok());
    // tile_h = (16-1)*3 + 3 = 48, tile_w = 48
    assert_eq!(c.input_tile_h(), 48);
    assert_eq!(c.input_tile_w(), 48);
}

// ---------------------------------------------------------------------------
// Large dilation edge cases
// ---------------------------------------------------------------------------

#[test]
fn test_large_dilation_value() {
    // dilation=8 on 3x3: effective = (3-1)*8 + 1 = 17
    let c = PtxConv2dConfig::new("conv_large_dil", 3, 3).with_dilation(8, 8);
    assert_eq!(c.effective_kernel_h(), 17);
    assert_eq!(c.effective_kernel_w(), 17);
    assert!(c.validate().is_ok());
    let ptx = emit_ptx_conv2d(&c).unwrap();
    // tile_h = (16-1)*1 + 17 = 32, tile_w = 32, tile_size = 1024
    assert_eq!(c.input_tile_h(), 32);
    assert_eq!(c.input_tile_w(), 32);
    assert!(ptx.contains("input_tile[1024]"));
}

#[test]
fn test_asymmetric_dilation_large() {
    // dilation_h=1, dilation_w=5 on 3x3
    let c = PtxConv2dConfig::new("k", 3, 3).with_dilation(1, 5);
    assert_eq!(c.effective_kernel_h(), 3); // (3-1)*1 + 1 = 3
    assert_eq!(c.effective_kernel_w(), 11); // (3-1)*5 + 1 = 11
    assert_eq!(c.input_tile_h(), 18); // (16-1)*1 + 3 = 18
    assert_eq!(c.input_tile_w(), 26); // (16-1)*1 + 11 = 26
}

// ---------------------------------------------------------------------------
// Combined stride + dilation + padding
// ---------------------------------------------------------------------------

#[test]
fn test_combined_stride_dilation_padding() {
    let c = PtxConv2dConfig::new("conv_combo", 3, 3)
        .with_stride(2, 2)
        .with_dilation(3, 3)
        .with_padding(3, 3);
    assert!(c.validate().is_ok());
    // effective = (3-1)*3 + 1 = 7
    assert_eq!(c.effective_kernel_h(), 7);
    assert_eq!(c.effective_kernel_w(), 7);
    // tile_h = (16-1)*2 + 7 = 37, tile_w = 37
    assert_eq!(c.input_tile_h(), 37);
    assert_eq!(c.input_tile_w(), 37);
    let ptx = emit_ptx_conv2d(&c).unwrap();
    assert!(!ptx.is_empty());
    assert!(ptx.contains("dilation: 3x3"));
    assert!(ptx.contains("stride: 2x2"));
    assert!(ptx.contains("pad: 3x3"));
}

// ---------------------------------------------------------------------------
// Output dim formula with stride > kernel
// ---------------------------------------------------------------------------

#[test]
fn test_output_size_stride_greater_than_kernel() {
    // 64 input, 3x3 kernel, pad=0, stride=4 -> (64 + 0 - 3)/4 + 1 = 16
    assert_eq!(local_conv2d_output_size(64, 0, 3, 4, 1), 16);
}

#[test]
fn test_output_size_stride_equals_kernel() {
    // 27 input, 3x3 kernel, pad=0, stride=3 -> (27 + 0 - 3)/3 + 1 = 9
    assert_eq!(local_conv2d_output_size(27, 0, 3, 3, 1), 9);
}

#[test]
fn test_output_size_large_dilation() {
    // 3x3 kernel, dilation=4 -> effective 9x9
    // 64 input, pad=4, stride=1 -> (64 + 8 - 9)/1 + 1 = 64
    assert_eq!(local_conv2d_output_size(64, 4, 3, 1, 4), 64);
}

#[test]
fn test_output_size_minimum_1() {
    // Smallest output: 1x1
    // kernel=3, pad=0, stride=1 -> input must be 3 for output=1
    assert_eq!(local_conv2d_output_size(3, 0, 3, 1, 1), 1);
}

// ---------------------------------------------------------------------------
// Launch config with stride > kernel
// ---------------------------------------------------------------------------

#[test]
fn test_launch_config_stride_greater_than_kernel() {
    let c = PtxConv2dConfig::new("conv", 3, 3).with_stride(4, 4);
    let h_out = local_conv2d_output_size(64, 0, 3, 4, 1); // 16
    let w_out = local_conv2d_output_size(64, 0, 3, 4, 1); // 16
    assert_eq!(h_out, 16);
    assert_eq!(w_out, 16);
    let (grid, block) = ptx_conv2d_launch_config(h_out, w_out, 1, 32, &c);
    assert_eq!(grid, [1, 1, 32]); // ceil(16/16)=1
    assert_eq!(block, [16, 16, 1]);
}

// ---------------------------------------------------------------------------
// PTX structural: comment content reflects config
// ---------------------------------------------------------------------------

#[test]
fn test_ptx_comment_reflects_stride_padding() {
    let c = PtxConv2dConfig::new("conv_s3p2", 5, 5)
        .with_stride(3, 3)
        .with_padding(2, 2);
    let ptx = emit_ptx_conv2d(&c).unwrap();
    assert!(ptx.contains("stride: 3x3"));
    assert!(ptx.contains("pad: 2x2"));
    assert!(ptx.contains("kernel: 5x5"));
}

#[test]
fn test_ptx_comment_reflects_shared_memory_bytes() {
    let c = PtxConv2dConfig::new("conv_shmem", 3, 3);
    let ptx = emit_ptx_conv2d(&c).unwrap();
    let expected_bytes = c.shared_memory_bytes();
    assert!(ptx.contains(&format!("shared memory: {expected_bytes} bytes")));
}

// ---------------------------------------------------------------------------
// Clone on config
// ---------------------------------------------------------------------------

#[test]
fn test_config_clone() {
    let c1 = PtxConv2dConfig::new("conv_clone", 5, 5)
        .with_stride(2, 2)
        .with_padding(2, 2)
        .with_dilation(2, 2)
        .with_bias(true)
        .with_block_size(8, 8)
        .with_sm_target("sm_90");
    let c2 = c1.clone();
    assert_eq!(c1.kernel_name, c2.kernel_name);
    assert_eq!(c1.kernel_h, c2.kernel_h);
    assert_eq!(c1.kernel_w, c2.kernel_w);
    assert_eq!(c1.stride_h, c2.stride_h);
    assert_eq!(c1.stride_w, c2.stride_w);
    assert_eq!(c1.pad_h, c2.pad_h);
    assert_eq!(c1.pad_w, c2.pad_w);
    assert_eq!(c1.dilation_h, c2.dilation_h);
    assert_eq!(c1.dilation_w, c2.dilation_w);
    assert_eq!(c1.use_bias, c2.use_bias);
    assert_eq!(c1.block_h, c2.block_h);
    assert_eq!(c1.block_w, c2.block_w);
    assert_eq!(c1.sm_target, c2.sm_target);
}

// ---------------------------------------------------------------------------
// Debug on config
// ---------------------------------------------------------------------------

#[test]
fn test_config_debug_format() {
    let c = PtxConv2dConfig::new("test_debug", 3, 3);
    let debug = format!("{c:?}");
    assert!(debug.contains("PtxConv2dConfig"));
    assert!(debug.contains("test_debug"));
    assert!(debug.contains("kernel_h: 3"));
    assert!(debug.contains("kernel_w: 3"));
}

// ---------------------------------------------------------------------------
// with_channels constructor
// ---------------------------------------------------------------------------

#[test]
fn test_with_channels_constructor() {
    let c = PtxConv2dConfig::with_channels("conv_ch", 3, 16, 3, 3);
    assert_eq!(c.in_channels, 3);
    assert_eq!(c.out_channels, 16);
    assert_eq!(c.kernel_h, 3);
    assert_eq!(c.kernel_w, 3);
    assert_eq!(c.groups, 1);
    assert_eq!(c.stride_h, 1);
    assert_eq!(c.pad_h, 0);
    assert!(c.validate().is_ok());
}

#[test]
fn test_with_channels_defaults_match_new() {
    let c1 = PtxConv2dConfig::new("k", 5, 5);
    let c2 = PtxConv2dConfig::with_channels("k", 0, 0, 5, 5);
    assert_eq!(c1.stride_h, c2.stride_h);
    assert_eq!(c1.stride_w, c2.stride_w);
    assert_eq!(c1.pad_h, c2.pad_h);
    assert_eq!(c1.pad_w, c2.pad_w);
    assert_eq!(c1.dilation_h, c2.dilation_h);
    assert_eq!(c1.dilation_w, c2.dilation_w);
    assert_eq!(c1.groups, c2.groups);
    assert_eq!(c1.use_bias, c2.use_bias);
    assert_eq!(c1.block_h, c2.block_h);
    assert_eq!(c1.block_w, c2.block_w);
    assert_eq!(c1.sm_target, c2.sm_target);
}

// ---------------------------------------------------------------------------
// Groups configuration & validation
// ---------------------------------------------------------------------------

#[test]
fn test_config_with_groups_builder() {
    let c = PtxConv2dConfig::with_channels("conv_g", 64, 64, 3, 3).with_groups(4);
    assert_eq!(c.groups, 4);
    assert!(c.validate().is_ok());
}

#[test]
fn test_config_groups_zero_rejected() {
    let c = PtxConv2dConfig::with_channels("k", 16, 16, 3, 3).with_groups(0);
    assert!(c.validate().is_err());
}

#[test]
fn test_config_groups_not_divisible_in_channels() {
    let c = PtxConv2dConfig::with_channels("k", 10, 12, 3, 3).with_groups(3);
    // 10 % 3 != 0
    assert!(c.validate().is_err());
}

#[test]
fn test_config_groups_not_divisible_out_channels() {
    let c = PtxConv2dConfig::with_channels("k", 12, 10, 3, 3).with_groups(3);
    // 10 % 3 != 0
    assert!(c.validate().is_err());
}

#[test]
fn test_config_groups_both_divisible() {
    let c = PtxConv2dConfig::with_channels("k", 12, 24, 3, 3).with_groups(4);
    assert!(c.validate().is_ok());
}

#[test]
fn test_config_depthwise() {
    let c = PtxConv2dConfig::with_channels("dw", 32, 32, 3, 3).with_groups(32);
    assert!(c.validate().is_ok());
    assert!(c.is_depthwise());
    assert!(c.has_groups());
    assert_eq!(c.in_channels_per_group(), 1);
    assert_eq!(c.out_channels_per_group(), 1);
}

#[test]
fn test_config_not_depthwise_when_groups_1() {
    let c = PtxConv2dConfig::with_channels("k", 32, 32, 3, 3);
    assert!(!c.is_depthwise());
    assert!(!c.has_groups());
}

#[test]
fn test_config_not_depthwise_when_channels_differ() {
    let c = PtxConv2dConfig::with_channels("k", 32, 64, 3, 3).with_groups(4);
    assert!(!c.is_depthwise());
    assert!(c.has_groups());
}

#[test]
fn test_channels_per_group() {
    let c = PtxConv2dConfig::with_channels("k", 64, 128, 3, 3).with_groups(4);
    assert_eq!(c.in_channels_per_group(), 16);
    assert_eq!(c.out_channels_per_group(), 32);
}

#[test]
fn test_channels_per_group_zero_channels() {
    let c = PtxConv2dConfig::new("k", 3, 3);
    assert_eq!(c.in_channels_per_group(), 0);
    assert_eq!(c.out_channels_per_group(), 0);
}

// ---------------------------------------------------------------------------
// conv2d_output_size
// ---------------------------------------------------------------------------

#[test]
fn test_output_size_no_padding_no_stride() {
    // 8x8 input, 3x3 kernel, stride 1, no padding, no dilation
    // out = (8 + 0 - 3) / 1 + 1 = 6
    assert_eq!(conv2d_output_size(8, 3, 1, 0, 1), Some(6));
}

#[test]
fn test_output_size_with_padding() {
    // 8x8 input, 3x3 kernel, pad=1
    // out = (8 + 2 - 3) / 1 + 1 = 8
    assert_eq!(conv2d_output_size(8, 3, 1, 1, 1), Some(8));
}

#[test]
fn test_output_size_with_stride() {
    // 8x8 input, 3x3 kernel, stride=2, pad=1
    // out = (8 + 2 - 3) / 2 + 1 = 4
    assert_eq!(conv2d_output_size(8, 3, 2, 1, 1), Some(4));
}

#[test]
fn test_output_size_with_dilation() {
    // 8x8 input, 3x3 kernel, dilation=2
    // effective_k = 2*(3-1) + 1 = 5
    // out = (8 + 0 - 5) / 1 + 1 = 4
    assert_eq!(conv2d_output_size(8, 3, 1, 0, 2), Some(4));
}

#[test]
fn test_output_size_1x1_same_as_input() {
    assert_eq!(conv2d_output_size(32, 1, 1, 0, 1), Some(32));
}

#[test]
fn test_output_size_too_small_input() {
    // 2x2 input, 5x5 kernel, no padding => invalid
    assert_eq!(conv2d_output_size(2, 5, 1, 0, 1), None);
}

#[test]
fn test_output_size_kernel_equals_input() {
    // 5x5 input, 5x5 kernel => 1x1 output
    assert_eq!(conv2d_output_size(5, 5, 1, 0, 1), Some(1));
}

#[test]
fn test_output_size_stride_2_exact() {
    // 7x7 input, 3x3 kernel, stride=2, pad=0
    // out = (7 - 3) / 2 + 1 = 3
    assert_eq!(conv2d_output_size(7, 3, 2, 0, 1), Some(3));
}

// ---------------------------------------------------------------------------
// CPU reference: known-value tests
// ---------------------------------------------------------------------------

#[test]
fn test_reference_1x1_identity_kernel() {
    // 1x1 conv with weight=1.0, no bias: output = input
    let config = PtxConv2dConfig::with_channels("ref", 1, 1, 1, 1);
    let input = vec![1.0, 2.0, 3.0, 4.0]; // [1, 1, 2, 2]
    let weight = vec![1.0]; // [1, 1, 1, 1]
    let output = conv2d_reference(&input, &weight, None, &config, 1, 2, 2);
    assert_eq!(output, vec![1.0, 2.0, 3.0, 4.0]);
}

#[test]
fn test_reference_1x1_scaling() {
    // 1x1 conv with weight=2.0: output = 2 * input
    let config = PtxConv2dConfig::with_channels("ref", 1, 1, 1, 1);
    let input = vec![1.0, 2.0, 3.0, 4.0]; // [1, 1, 2, 2]
    let weight = vec![2.0];
    let output = conv2d_reference(&input, &weight, None, &config, 1, 2, 2);
    assert_eq!(output, vec![2.0, 4.0, 6.0, 8.0]);
}

#[test]
fn test_reference_1x1_with_bias() {
    let config = PtxConv2dConfig::with_channels("ref", 1, 1, 1, 1);
    let input = vec![1.0, 2.0, 3.0, 4.0];
    let weight = vec![1.0];
    let bias = vec![10.0];
    let output = conv2d_reference(&input, &weight, Some(&bias), &config, 1, 2, 2);
    assert_eq!(output, vec![11.0, 12.0, 13.0, 14.0]);
}

#[test]
fn test_reference_3x3_no_padding() {
    // 4x4 input, 3x3 kernel, stride 1, no padding => 2x2 output
    // Input: all 1.0, weight: all 1.0
    // Each output pixel = sum of 3x3 window of 1s = 9
    let config = PtxConv2dConfig::with_channels("ref", 1, 1, 3, 3);
    let input = vec![1.0f32; 16]; // [1, 1, 4, 4]
    let weight = vec![1.0f32; 9]; // [1, 1, 3, 3]
    let output = conv2d_reference(&input, &weight, None, &config, 1, 4, 4);
    assert_eq!(output.len(), 4); // 2x2
    for v in &output {
        assert!((v - 9.0).abs() < 1e-6, "expected 9.0, got {v}");
    }
}

#[test]
fn test_reference_3x3_with_padding() {
    // 4x4 input, 3x3 kernel, pad=1 => 4x4 output
    let config = PtxConv2dConfig::with_channels("ref", 1, 1, 3, 3).with_padding(1, 1);
    let input = vec![1.0f32; 16]; // [1, 1, 4, 4]
    let weight = vec![1.0f32; 9]; // [1, 1, 3, 3]
    let output = conv2d_reference(&input, &weight, None, &config, 1, 4, 4);
    assert_eq!(output.len(), 16); // 4x4

    // Center pixels see full 3x3 window of 1s = 9
    // output[1*4+1], output[1*4+2], output[2*4+1], output[2*4+2]
    assert!((output[5] - 9.0).abs() < 1e-6, "center pixel should be 9.0");
    assert!((output[6] - 9.0).abs() < 1e-6);
    assert!((output[9] - 9.0).abs() < 1e-6);
    assert!((output[10] - 9.0).abs() < 1e-6);

    // Corner pixel (0,0) sees 2x2 window = 4
    assert!(
        (output[0] - 4.0).abs() < 1e-6,
        "corner should be 4.0, got {}",
        output[0]
    );
}

#[test]
fn test_reference_stride_2() {
    // 4x4 input, 1x1 kernel, stride=2 => 2x2 output (picks every other pixel)
    let config = PtxConv2dConfig::with_channels("ref", 1, 1, 1, 1).with_stride(2, 2);
    // Input values: row-major 1..16
    let input: Vec<f32> = (1..=16).map(|x| x as f32).collect();
    let weight = vec![1.0];
    let output = conv2d_reference(&input, &weight, None, &config, 1, 4, 4);
    // Output should pick [0,0], [0,2], [2,0], [2,2] = 1, 3, 9, 11
    assert_eq!(output.len(), 4);
    assert!((output[0] - 1.0).abs() < 1e-6);
    assert!((output[1] - 3.0).abs() < 1e-6);
    assert!((output[2] - 9.0).abs() < 1e-6);
    assert!((output[3] - 11.0).abs() < 1e-6);
}

#[test]
fn test_reference_dilation_2() {
    // 5x5 input, 3x3 kernel, dilation=2 => effective 5x5 kernel => 1x1 output
    let config = PtxConv2dConfig::with_channels("ref", 1, 1, 3, 3).with_dilation(2, 2);
    let input = vec![1.0f32; 25]; // [1, 1, 5, 5]
    let weight = vec![1.0f32; 9]; // [1, 1, 3, 3]
    let output = conv2d_reference(&input, &weight, None, &config, 1, 5, 5);
    // effective kernel = (3-1)*2+1 = 5, output = (5-5)/1+1 = 1
    assert_eq!(output.len(), 1);
    // 3x3 dilated kernel on all-1s picks 9 elements = 9.0
    assert!((output[0] - 9.0).abs() < 1e-6);
}

#[test]
fn test_reference_multi_channel_input() {
    // 2 input channels, 1 output channel, 1x1 kernel
    let config = PtxConv2dConfig::with_channels("ref", 2, 1, 1, 1);
    // Input [1, 2, 2, 2]: ch0=1.0, ch1=2.0
    let input = vec![1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0]; // [1, 2, 2, 2]
                                                              // Weight [1, 2, 1, 1]: w0=1.0, w1=0.5
    let weight = vec![1.0, 0.5]; // [1, 2, 1, 1]
    let output = conv2d_reference(&input, &weight, None, &config, 1, 2, 2);
    // Each output = 1.0 * 1.0 + 2.0 * 0.5 = 2.0
    assert_eq!(output.len(), 4);
    for v in &output {
        assert!((v - 2.0).abs() < 1e-6);
    }
}

#[test]
fn test_reference_multi_output_channel() {
    // 1 input channel, 2 output channels, 1x1 kernel
    let config = PtxConv2dConfig::with_channels("ref", 1, 2, 1, 1);
    let input = vec![3.0f32; 4]; // [1, 1, 2, 2], all 3.0
                                 // Weight [2, 1, 1, 1]: oc0 w=1.0, oc1 w=2.0
    let weight = vec![1.0, 2.0];
    let output = conv2d_reference(&input, &weight, None, &config, 1, 2, 2);
    assert_eq!(output.len(), 8); // [1, 2, 2, 2]
                                 // oc0: 3*1 = 3, oc1: 3*2 = 6
    for i in 0..4 {
        assert!((output[i] - 3.0).abs() < 1e-6, "oc0[{i}]");
    }
    for i in 4..8 {
        assert!((output[i] - 6.0).abs() < 1e-6, "oc1[{i}]");
    }
}

#[test]
fn test_reference_batch_size_2() {
    let config = PtxConv2dConfig::with_channels("ref", 1, 1, 1, 1);
    // batch=2, each [1, 1, 2, 2]
    let input = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let weight = vec![1.0];
    let output = conv2d_reference(&input, &weight, None, &config, 2, 2, 2);
    assert_eq!(output, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
}

// ---------------------------------------------------------------------------
// CPU reference: grouped convolution
// ---------------------------------------------------------------------------

#[test]
fn test_reference_groups_2() {
    // 4 in, 4 out, groups=2 => 2 in per group, 2 out per group
    // Weight: [4, 2, 1, 1]
    let config = PtxConv2dConfig::with_channels("ref", 4, 4, 1, 1).with_groups(2);

    // Input: [1, 4, 1, 1] => [ch0=1, ch1=2, ch2=3, ch3=4]
    let input = vec![1.0, 2.0, 3.0, 4.0];
    // Weight: [4, 2, 1, 1]
    // group0: oc0 reads ic0,ic1; oc1 reads ic0,ic1
    // group1: oc2 reads ic2,ic3; oc3 reads ic2,ic3
    let weight = vec![
        1.0, 0.0, // oc0: w[ic0]=1, w[ic1]=0 => output=1
        0.0, 1.0, // oc1: w[ic0]=0, w[ic1]=1 => output=2
        1.0, 0.0, // oc2: w[ic2]=1, w[ic3]=0 => output=3
        0.0, 1.0, // oc3: w[ic2]=0, w[ic3]=1 => output=4
    ];
    let output = conv2d_reference(&input, &weight, None, &config, 1, 1, 1);
    assert_eq!(output.len(), 4);
    assert!((output[0] - 1.0).abs() < 1e-6, "oc0={}", output[0]);
    assert!((output[1] - 2.0).abs() < 1e-6, "oc1={}", output[1]);
    assert!((output[2] - 3.0).abs() < 1e-6, "oc2={}", output[2]);
    assert!((output[3] - 4.0).abs() < 1e-6, "oc3={}", output[3]);
}

#[test]
fn test_reference_depthwise_3x3() {
    // Depthwise: 3 in, 3 out, groups=3 => 1 in per group, 1 out per group
    // Each output channel only looks at its own input channel
    let config = PtxConv2dConfig::with_channels("dw", 3, 3, 3, 3)
        .with_groups(3)
        .with_padding(1, 1);

    assert!(config.is_depthwise());

    // Input: [1, 3, 3, 3] all 1.0
    let input = vec![1.0f32; 27];
    // Weight: [3, 1, 3, 3]
    // ch0 filter = all 1.0, ch1 filter = all 2.0, ch2 filter = all 0.5
    let mut weight = vec![0.0f32; 27];
    for i in 0..9 {
        weight[i] = 1.0; // oc0
        weight[9 + i] = 2.0; // oc1
        weight[18 + i] = 0.5; // oc2
    }
    let output = conv2d_reference(&input, &weight, None, &config, 1, 3, 3);
    assert_eq!(output.len(), 27); // [1, 3, 3, 3]

    // Center pixel of each channel: full 3x3 visible
    // oc0 center: 9*1.0*1.0 = 9
    assert!((output[4] - 9.0).abs() < 1e-6, "oc0 center={}", output[4]);
    // oc1 center: 9*2.0*1.0 = 18
    assert!(
        (output[9 + 4] - 18.0).abs() < 1e-6,
        "oc1 center={}",
        output[13]
    );
    // oc2 center: 9*0.5*1.0 = 4.5
    assert!(
        (output[18 + 4] - 4.5).abs() < 1e-6,
        "oc2 center={}",
        output[22]
    );
}

#[test]
fn test_reference_depthwise_1x1() {
    // Depthwise 1x1: each channel scaled independently
    let config = PtxConv2dConfig::with_channels("dw1", 4, 4, 1, 1).with_groups(4);
    assert!(config.is_depthwise());

    // Input: [1, 4, 2, 2]
    let input: Vec<f32> = (1..=16).map(|x| x as f32).collect();
    // Weight: [4, 1, 1, 1] = [2.0, 0.5, 1.0, 3.0]
    let weight = vec![2.0, 0.5, 1.0, 3.0];
    let output = conv2d_reference(&input, &weight, None, &config, 1, 2, 2);
    assert_eq!(output.len(), 16);

    // oc0: [1,2,3,4] * 2 = [2,4,6,8]
    assert!((output[0] - 2.0).abs() < 1e-6);
    assert!((output[3] - 8.0).abs() < 1e-6);
    // oc1: [5,6,7,8] * 0.5 = [2.5,3,3.5,4]
    assert!((output[4] - 2.5).abs() < 1e-6);
    // oc2: [9,10,11,12] * 1 = [9,10,11,12]
    assert!((output[8] - 9.0).abs() < 1e-6);
    // oc3: [13,14,15,16] * 3 = [39,42,45,48]
    assert!((output[12] - 39.0).abs() < 1e-6);
    assert!((output[15] - 48.0).abs() < 1e-6);
}

// ---------------------------------------------------------------------------
// CPU reference: various kernel sizes (5x5, 7x7)
// ---------------------------------------------------------------------------

#[test]
fn test_reference_5x5_all_ones() {
    let config = PtxConv2dConfig::with_channels("ref5", 1, 1, 5, 5);
    let input = vec![1.0f32; 25]; // [1, 1, 5, 5]
    let weight = vec![1.0f32; 25]; // [1, 1, 5, 5]
    let output = conv2d_reference(&input, &weight, None, &config, 1, 5, 5);
    assert_eq!(output.len(), 1); // (5-5)/1+1 = 1
    assert!((output[0] - 25.0).abs() < 1e-6);
}

#[test]
fn test_reference_7x7_with_padding() {
    // 8x8 input, 7x7 kernel, pad=3 => output 8x8
    let config = PtxConv2dConfig::with_channels("ref7", 1, 1, 7, 7).with_padding(3, 3);
    let input = vec![1.0f32; 64]; // [1, 1, 8, 8]
    let weight = vec![1.0f32; 49]; // [1, 1, 7, 7]
    let output = conv2d_reference(&input, &weight, None, &config, 1, 8, 8);
    assert_eq!(output.len(), 64); // same size

    // Center pixel (3,3): full 7x7 visible = 49
    let center = output[3 * 8 + 3];
    assert!((center - 49.0).abs() < 1e-6, "7x7 center={center}");
}

// ---------------------------------------------------------------------------
// PTX generation with groups
// ---------------------------------------------------------------------------

#[test]
fn test_ptx_conv2d_groups_generates_valid_ptx() {
    let c = PtxConv2dConfig::with_channels("conv_g4", 64, 128, 3, 3)
        .with_groups(4)
        .with_padding(1, 1);
    let ptx = emit_ptx_conv2d(&c).unwrap();
    assert!(ptx.contains(".visible .entry conv_g4"));
    assert!(ptx.contains(".version 6.5"));
    assert!(ptx.contains("fma.rn.f32"));
    assert!(ptx.contains("ret;"));
}

#[test]
fn test_ptx_conv2d_depthwise_generates_valid_ptx() {
    let c = PtxConv2dConfig::with_channels("conv_dw", 32, 32, 3, 3)
        .with_groups(32)
        .with_padding(1, 1);
    assert!(c.is_depthwise());
    let ptx = emit_ptx_conv2d(&c).unwrap();
    assert!(ptx.contains(".visible .entry conv_dw"));
    assert!(ptx.contains("ret;"));
}

#[test]
fn test_ptx_conv2d_groups_1_same_as_no_groups() {
    // groups=1 with channels should produce same PTX structure as groups=1 without
    let c1 = PtxConv2dConfig::with_channels("conv", 64, 128, 3, 3)
        .with_groups(1)
        .with_padding(1, 1);
    let c2 = PtxConv2dConfig::with_channels("conv", 64, 128, 3, 3).with_padding(1, 1);
    let ptx1 = emit_ptx_conv2d(&c1).unwrap();
    let ptx2 = emit_ptx_conv2d(&c2).unwrap();
    assert_eq!(ptx1, ptx2, "groups=1 should match no-groups default");
}

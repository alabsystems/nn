// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Tests for PTX conv1d kernel generation.

use super::*;

// ---------------------------------------------------------------------------
// Config validation
// ---------------------------------------------------------------------------

#[test]
fn test_config_default() {
    let c = PtxConv1dConfig::default();
    assert_eq!(c.kernel_name, "ptx_conv1d_f32");
    assert_eq!(c.in_channels, 1);
    assert_eq!(c.out_channels, 1);
    assert_eq!(c.kernel_size, 3);
    assert_eq!(c.stride, 1);
    assert_eq!(c.padding, 0);
    assert_eq!(c.dilation, 1);
    assert_eq!(c.groups, 1);
    assert!(!c.use_bias);
    assert_eq!(c.block_size, 256);
    assert_eq!(c.sm_target, "sm_80");
    assert!(c.validate().is_ok());
}

#[test]
fn test_config_new_basic() {
    let c = PtxConv1dConfig::new("nn_conv", 64, 128, 3);
    assert_eq!(c.in_channels, 64);
    assert_eq!(c.out_channels, 128);
    assert_eq!(c.kernel_size, 3);
    assert!(c.validate().is_ok());
}

#[test]
fn test_config_empty_name_rejected() {
    let c = PtxConv1dConfig::new("", 64, 128, 3);
    assert!(c.validate().is_err());
}

#[test]
fn test_config_zero_in_channels_rejected() {
    let c = PtxConv1dConfig::new("conv", 0, 128, 3);
    assert!(c.validate().is_err());
}

#[test]
fn test_config_zero_out_channels_rejected() {
    let c = PtxConv1dConfig::new("conv", 64, 0, 3);
    assert!(c.validate().is_err());
}

#[test]
fn test_config_zero_kernel_size_rejected() {
    let c = PtxConv1dConfig::new("conv", 64, 128, 0);
    assert!(c.validate().is_err());
}

#[test]
fn test_config_kernel_too_large_rejected() {
    let c = PtxConv1dConfig::new("conv", 64, 128, PTX_CONV1D_MAX_KERNEL + 1);
    assert!(c.validate().is_err());
}

#[test]
fn test_config_zero_stride_rejected() {
    let c = PtxConv1dConfig::new("conv", 64, 128, 3).with_stride(0);
    assert!(c.validate().is_err());
}

#[test]
fn test_config_zero_dilation_rejected() {
    let c = PtxConv1dConfig::new("conv", 64, 128, 3).with_dilation(0);
    assert!(c.validate().is_err());
}

#[test]
fn test_config_zero_groups_rejected() {
    let c = PtxConv1dConfig::new("conv", 64, 128, 3).with_groups(0);
    assert!(c.validate().is_err());
}

#[test]
fn test_config_in_channels_not_divisible_by_groups() {
    let c = PtxConv1dConfig::new("conv", 64, 128, 3).with_groups(3);
    assert!(c.validate().is_err());
}

#[test]
fn test_config_out_channels_not_divisible_by_groups() {
    let c = PtxConv1dConfig::new("conv", 64, 127, 3).with_groups(2);
    assert!(c.validate().is_err());
}

#[test]
fn test_config_zero_block_size_rejected() {
    let c = PtxConv1dConfig::new("conv", 64, 128, 3).with_block_size(0);
    assert!(c.validate().is_err());
}

#[test]
fn test_config_block_size_too_large_rejected() {
    let c = PtxConv1dConfig::new("conv", 64, 128, 3).with_block_size(2048);
    assert!(c.validate().is_err());
}

#[test]
fn test_config_valid_groups() {
    let c = PtxConv1dConfig::new("conv", 64, 128, 3).with_groups(4);
    assert!(c.validate().is_ok());
    assert_eq!(c.in_channels_per_group(), 16);
    assert_eq!(c.out_channels_per_group(), 32);
}

#[test]
fn test_config_depthwise() {
    // Depthwise separable: groups == in_channels == out_channels
    let c = PtxConv1dConfig::new("dw_conv", 64, 64, 3).with_groups(64);
    assert!(c.validate().is_ok());
    assert_eq!(c.in_channels_per_group(), 1);
    assert_eq!(c.out_channels_per_group(), 1);
}

#[test]
fn test_config_effective_kernel_size() {
    let c = PtxConv1dConfig::new("conv", 64, 128, 3).with_dilation(2);
    assert_eq!(c.effective_kernel_size(), 5); // (3-1)*2 + 1
}

#[test]
fn test_config_effective_kernel_size_no_dilation() {
    let c = PtxConv1dConfig::new("conv", 64, 128, 5);
    assert_eq!(c.effective_kernel_size(), 5); // (5-1)*1 + 1
}

#[test]
fn test_config_builder_chain() {
    let c = PtxConv1dConfig::new("conv", 64, 128, 3)
        .with_stride(2)
        .with_padding(1)
        .with_dilation(1)
        .with_groups(1)
        .with_bias(true)
        .with_block_size(128)
        .with_sm_target("sm_90");
    assert_eq!(c.stride, 2);
    assert_eq!(c.padding, 1);
    assert_eq!(c.dilation, 1);
    assert_eq!(c.groups, 1);
    assert!(c.use_bias);
    assert_eq!(c.block_size, 128);
    assert_eq!(c.sm_target, "sm_90");
    assert!(c.validate().is_ok());
}

// ---------------------------------------------------------------------------
// Output length calculation
// ---------------------------------------------------------------------------

#[test]
fn test_output_length_basic() {
    // L_in=10, K=3, stride=1, pad=0, dilation=1 -> (10+0-3)/1+1 = 8
    assert_eq!(conv1d_output_length(10, 3, 1, 0, 1), Some(8));
}

#[test]
fn test_output_length_with_padding() {
    // L_in=10, K=3, stride=1, pad=1, dilation=1 -> (10+2-3)/1+1 = 10
    assert_eq!(conv1d_output_length(10, 3, 1, 1, 1), Some(10));
}

#[test]
fn test_output_length_with_stride() {
    // L_in=10, K=3, stride=2, pad=0, dilation=1 -> (10+0-3)/2+1 = 4
    assert_eq!(conv1d_output_length(10, 3, 2, 0, 1), Some(4));
}

#[test]
fn test_output_length_with_dilation() {
    // L_in=10, K=3, stride=1, pad=0, dilation=2
    // effective_k = 2*(3-1)+1 = 5
    // (10+0-5)/1+1 = 6
    assert_eq!(conv1d_output_length(10, 3, 1, 0, 2), Some(6));
}

#[test]
fn test_output_length_stride_and_padding() {
    // L_in=16, K=3, stride=2, pad=1 -> (16+2-3)/2+1 = 8
    assert_eq!(conv1d_output_length(16, 3, 2, 1, 1), Some(8));
}

#[test]
fn test_output_length_kernel_equals_input() {
    // L_in=5, K=5, stride=1, pad=0 -> (5-5)/1+1 = 1
    assert_eq!(conv1d_output_length(5, 5, 1, 0, 1), Some(1));
}

#[test]
fn test_output_length_kernel_larger_than_input_no_padding() {
    // L_in=3, K=5, stride=1, pad=0 -> padded=3 < effective_k=5 -> None
    assert_eq!(conv1d_output_length(3, 5, 1, 0, 1), None);
}

#[test]
fn test_output_length_kernel_1() {
    // L_in=10, K=1, stride=1, pad=0 -> 10
    assert_eq!(conv1d_output_length(10, 1, 1, 0, 1), Some(10));
}

#[test]
fn test_output_length_kokoro_like() {
    // Typical Kokoro conv: L_in=4096, K=7, stride=1, pad=3
    // (4096+6-7)/1+1 = 4096
    assert_eq!(conv1d_output_length(4096, 7, 1, 3, 1), Some(4096));
}

#[test]
fn test_output_length_demucs_like() {
    // Demucs encoder conv: L_in=44100, K=8, stride=4, pad=0
    // (44100-8)/4+1 = 11024
    assert_eq!(conv1d_output_length(44100, 8, 4, 0, 1), Some(11024));
}

// ---------------------------------------------------------------------------
// PTX structural checks
// ---------------------------------------------------------------------------

fn assert_common_ptx_structure(ptx: &str, kernel_name: &str) {
    assert!(ptx.contains(".version 6.5"), "must contain PTX version");
    assert!(ptx.contains(".target sm_80"), "must contain SM target");
    assert!(
        ptx.contains(".address_size 64"),
        "must declare 64-bit addressing"
    );
    assert!(
        ptx.contains(&format!(".visible .entry {kernel_name}")),
        "must declare visible entry point: {kernel_name}"
    );
    assert!(ptx.contains("param_input"), "must have input pointer param");
    assert!(
        ptx.contains("param_weight"),
        "must have weight pointer param"
    );
    assert!(
        ptx.contains("param_output"),
        "must have output pointer param"
    );
    assert!(
        ptx.contains("param_batch_size"),
        "must have batch_size param"
    );
    assert!(ptx.contains("param_length_in"), "must have length_in param");
    assert!(
        ptx.contains("param_length_out"),
        "must have length_out param"
    );
    assert!(ptx.contains("ret;"), "must contain ret instruction");
}

#[test]
fn test_ptx_structure_basic() {
    let config = PtxConv1dConfig::new("conv1d_k3", 64, 128, 3);
    let ptx = emit_ptx_conv1d(&config).unwrap();
    assert_common_ptx_structure(&ptx, "conv1d_k3");
}

#[test]
fn test_ptx_structure_with_bias() {
    let config = PtxConv1dConfig::new("conv1d_bias", 64, 128, 3).with_bias(true);
    let ptx = emit_ptx_conv1d(&config).unwrap();
    assert_common_ptx_structure(&ptx, "conv1d_bias");
    assert!(
        ptx.contains("param_bias"),
        "must have bias param when bias enabled"
    );
    assert!(ptx.contains("Add bias"), "must contain bias addition code");
}

#[test]
fn test_ptx_structure_no_bias() {
    let config = PtxConv1dConfig::new("conv1d_nobias", 64, 128, 3);
    let ptx = emit_ptx_conv1d(&config).unwrap();
    assert!(
        !ptx.contains("param_bias"),
        "must not have bias param when bias disabled"
    );
}

#[test]
fn test_ptx_contains_grid_stride_loop() {
    let config = PtxConv1dConfig::new("conv1d", 64, 128, 3);
    let ptx = emit_ptx_conv1d(&config).unwrap();
    assert!(
        ptx.contains("CONV1D_LOOP:"),
        "must have grid-stride loop label"
    );
    assert!(ptx.contains("CONV1D_EXIT:"), "must have kernel exit label");
    assert!(
        ptx.contains("%nctaid.x"),
        "must read gridDim.x for stride computation"
    );
}

#[test]
fn test_ptx_contains_ic_and_k_loops() {
    let config = PtxConv1dConfig::new("conv1d", 64, 128, 3);
    let ptx = emit_ptx_conv1d(&config).unwrap();
    assert!(
        ptx.contains("IC_LOOP:"),
        "must have input-channel loop label"
    );
    assert!(ptx.contains("IC_DONE:"), "must have IC loop done label");
    assert!(
        ptx.contains("K_LOOP:"),
        "must have kernel-position loop label"
    );
    assert!(ptx.contains("K_DONE:"), "must have K loop done label");
}

#[test]
fn test_ptx_contains_bounds_check() {
    let config = PtxConv1dConfig::new("conv1d", 64, 128, 3).with_padding(1);
    let ptx = emit_ptx_conv1d(&config).unwrap();
    assert!(
        ptx.contains("setp.ge.u32"),
        "must have bounds check via setp"
    );
    assert!(
        ptx.contains("in_pos < length_in"),
        "must check input position bounds"
    );
}

#[test]
fn test_ptx_contains_fma() {
    let config = PtxConv1dConfig::new("conv1d", 64, 128, 3);
    let ptx = emit_ptx_conv1d(&config).unwrap();
    assert!(
        ptx.contains("fma.rn.f32"),
        "must use fused multiply-add instruction"
    );
}

#[test]
fn test_ptx_contains_global_loads_stores() {
    let config = PtxConv1dConfig::new("conv1d", 64, 128, 3);
    let ptx = emit_ptx_conv1d(&config).unwrap();
    assert!(
        ptx.contains("ld.global.f32"),
        "must load from global memory"
    );
    assert!(ptx.contains("st.global.f32"), "must store to global memory");
}

#[test]
fn test_ptx_is_pure_ptx_not_cuda_cpp() {
    let config = PtxConv1dConfig::new("conv1d", 64, 128, 3);
    let ptx = emit_ptx_conv1d(&config).unwrap();
    assert!(
        !ptx.contains("__global__"),
        "must not contain CUDA C++ __global__"
    );
    assert!(!ptx.contains("#include"), "must not contain C++ #include");
    assert!(
        !ptx.contains("__shared__"),
        "must not contain CUDA C++ __shared__"
    );
    assert!(
        !ptx.contains("__syncthreads"),
        "must not contain CUDA C++ __syncthreads"
    );
    assert!(
        !ptx.contains("__restrict__"),
        "must not contain C++ __restrict__"
    );
}

// ---------------------------------------------------------------------------
// Groups > 1
// ---------------------------------------------------------------------------

#[test]
fn test_ptx_groups_2() {
    let config = PtxConv1dConfig::new("conv_g2", 64, 128, 3).with_groups(2);
    let ptx = emit_ptx_conv1d(&config).unwrap();
    assert_common_ptx_structure(&ptx, "conv_g2");
    // ic_per_group = 32, oc_per_group = 64
    assert!(ptx.contains("groups=2"), "header should mention groups=2");
}

#[test]
fn test_ptx_depthwise() {
    let config = PtxConv1dConfig::new("dw_conv", 32, 32, 5).with_groups(32);
    let ptx = emit_ptx_conv1d(&config).unwrap();
    assert_common_ptx_structure(&ptx, "dw_conv");
    assert!(ptx.contains("groups=32"), "header should mention groups=32");
}

// ---------------------------------------------------------------------------
// Dilation > 1
// ---------------------------------------------------------------------------

#[test]
fn test_ptx_dilation_2() {
    let config = PtxConv1dConfig::new("conv_d2", 64, 128, 3).with_dilation(2);
    let ptx = emit_ptx_conv1d(&config).unwrap();
    assert_common_ptx_structure(&ptx, "conv_d2");
    assert!(
        ptx.contains("dilation=2"),
        "header should mention dilation=2"
    );
}

#[test]
fn test_ptx_dilation_4() {
    let config = PtxConv1dConfig::new("conv_d4", 64, 128, 3)
        .with_dilation(4)
        .with_padding(4);
    let ptx = emit_ptx_conv1d(&config).unwrap();
    assert_common_ptx_structure(&ptx, "conv_d4");
    assert!(
        ptx.contains("dilation=4"),
        "header should mention dilation=4"
    );
}

// ---------------------------------------------------------------------------
// Various configs produce valid PTX
// ---------------------------------------------------------------------------

#[test]
fn test_ptx_kernel_size_1() {
    // Pointwise (1x1) conv1d
    let config = PtxConv1dConfig::new("pw_conv", 64, 128, 1);
    let ptx = emit_ptx_conv1d(&config).unwrap();
    assert_common_ptx_structure(&ptx, "pw_conv");
    assert!(ptx.contains("kernel=1"));
}

#[test]
fn test_ptx_kernel_size_7() {
    let config = PtxConv1dConfig::new("conv_k7", 64, 128, 7).with_padding(3);
    let ptx = emit_ptx_conv1d(&config).unwrap();
    assert_common_ptx_structure(&ptx, "conv_k7");
}

#[test]
fn test_ptx_stride_4() {
    let config = PtxConv1dConfig::new("conv_s4", 1, 48, 8).with_stride(4);
    let ptx = emit_ptx_conv1d(&config).unwrap();
    assert_common_ptx_structure(&ptx, "conv_s4");
    assert!(ptx.contains("stride=4"));
}

#[test]
fn test_ptx_custom_name() {
    let config = PtxConv1dConfig::new("nn_custom_conv1d", 32, 64, 5);
    let ptx = emit_ptx_conv1d(&config).unwrap();
    assert!(ptx.contains(".entry nn_custom_conv1d"));
}

#[test]
fn test_ptx_custom_sm_target() {
    let config = PtxConv1dConfig::new("conv_sm90", 64, 128, 3).with_sm_target("sm_90");
    let ptx = emit_ptx_conv1d(&config).unwrap();
    assert!(ptx.contains(".target sm_90"));
}

#[test]
fn test_ptx_custom_block_size() {
    let config = PtxConv1dConfig::new("conv_b128", 64, 128, 3).with_block_size(128);
    let ptx = emit_ptx_conv1d(&config).unwrap();
    assert!(ptx.contains(".reqntid 128"));
}

#[test]
fn test_ptx_default_block_size() {
    let config = PtxConv1dConfig::new("conv", 64, 128, 3);
    let ptx = emit_ptx_conv1d(&config).unwrap();
    assert!(ptx.contains(".reqntid 256"));
}

// ---------------------------------------------------------------------------
// Header contains config info
// ---------------------------------------------------------------------------

#[test]
fn test_ptx_header_contains_config_info() {
    let config = PtxConv1dConfig::new("conv", 64, 128, 3)
        .with_stride(2)
        .with_padding(1)
        .with_dilation(1)
        .with_groups(1);
    let ptx = emit_ptx_conv1d(&config).unwrap();
    assert!(ptx.contains("in_ch=64"), "header should have in_channels");
    assert!(
        ptx.contains("out_ch=128"),
        "header should have out_channels"
    );
    assert!(ptx.contains("kernel=3"), "header should have kernel_size");
    assert!(ptx.contains("stride=2"), "header should have stride");
    assert!(ptx.contains("pad=1"), "header should have padding");
    assert!(ptx.contains("dilation=1"), "header should have dilation");
    assert!(ptx.contains("groups=1"), "header should have groups");
    assert!(ptx.contains("block=256"), "header should have block_size");
}

// ---------------------------------------------------------------------------
// Convenience API
// ---------------------------------------------------------------------------

#[test]
fn test_emit_default() {
    let ptx = emit_ptx_conv1d_default(64, 128, 3, 1, 0).unwrap();
    assert!(ptx.contains(".entry ptx_conv1d_f32"));
    assert!(ptx.contains("in_ch=64"));
    assert!(ptx.contains("out_ch=128"));
    assert!(ptx.contains("kernel=3"));
}

#[test]
fn test_emit_default_with_stride_and_pad() {
    let ptx = emit_ptx_conv1d_default(1, 48, 8, 4, 0).unwrap();
    assert!(ptx.contains("stride=4"));
    assert!(ptx.contains("kernel=8"));
}

// ---------------------------------------------------------------------------
// Launch config
// ---------------------------------------------------------------------------

#[test]
fn test_launch_config_basic() {
    // batch=1, out_ch=128, output_length=100 -> total=12800
    let cfg = ptx_conv1d_launch_config(1, 128, 100);
    assert_eq!(cfg.block.x, 256);
    assert_eq!(cfg.grid.x, 50); // ceil(12800/256) = 50
    assert_eq!(cfg.shared_mem_bytes, 0);
}

#[test]
fn test_launch_config_small() {
    // batch=1, out_ch=1, output_length=10 -> total=10
    let cfg = ptx_conv1d_launch_config(1, 1, 10);
    assert_eq!(cfg.grid.x, 1); // ceil(10/256) = 1
    assert_eq!(cfg.block.x, 256);
}

#[test]
fn test_launch_config_batched() {
    // batch=4, out_ch=128, output_length=256 -> total=131072
    let cfg = ptx_conv1d_launch_config(4, 128, 256);
    assert_eq!(cfg.grid.x, 512); // ceil(131072/256) = 512
    assert_eq!(cfg.block.x, 256);
}

#[test]
fn test_launch_config_not_multiple() {
    // batch=1, out_ch=64, output_length=100 -> total=6400
    let cfg = ptx_conv1d_launch_config(1, 64, 100);
    assert_eq!(cfg.grid.x, 25); // ceil(6400/256) = 25
}

#[test]
fn test_launch_config_1d_grid() {
    // Verify launch config is 1D (y=1, z=1)
    let cfg = ptx_conv1d_launch_config(2, 64, 512);
    assert_eq!(cfg.grid.y, 1);
    assert_eq!(cfg.grid.z, 1);
    assert_eq!(cfg.block.y, 1);
    assert_eq!(cfg.block.z, 1);
}

// ---------------------------------------------------------------------------
// Instruction set coverage
// ---------------------------------------------------------------------------

#[test]
fn test_ptx_instruction_set_coverage() {
    let config = PtxConv1dConfig::new("conv1d", 64, 128, 3).with_padding(1);
    let ptx = emit_ptx_conv1d(&config).unwrap();
    let expected_instructions = [
        "ld.param",      // parameter loading
        "mov.u32",       // register moves
        "mad.lo.u32",    // multiply-add for indexing
        "mul.lo.u32",    // multiply for offsets
        "mul.wide.u32",  // widening multiply for byte offsets
        "add.u32",       // integer addition
        "sub.u32",       // subtraction (padding)
        "div.u32",       // division (decompose global_idx)
        "rem.u32",       // remainder (decompose global_idx)
        "add.u64",       // 64-bit pointer arithmetic
        "ld.global.f32", // load f32 from global memory
        "st.global.f32", // store f32 to global memory
        "fma.rn.f32",    // fused multiply-add
        "setp.ge.u32",   // bounds checks
        "bra",           // branch
        "ret",           // return
    ];
    for instr in &expected_instructions {
        assert!(
            ptx.contains(instr),
            "Conv1d PTX must contain instruction: {instr}"
        );
    }
}

// ---------------------------------------------------------------------------
// Different configs produce different PTX
// ---------------------------------------------------------------------------

#[test]
fn test_different_channels_produce_different_ptx() {
    let ptx_a = emit_ptx_conv1d_default(32, 64, 3, 1, 0).unwrap();
    let ptx_b = emit_ptx_conv1d_default(64, 128, 3, 1, 0).unwrap();
    assert_ne!(
        ptx_a, ptx_b,
        "different channel counts should produce different PTX"
    );
}

#[test]
fn test_different_kernel_sizes_produce_different_ptx() {
    let ptx_a = emit_ptx_conv1d_default(64, 128, 3, 1, 1).unwrap();
    let ptx_b = emit_ptx_conv1d_default(64, 128, 5, 1, 2).unwrap();
    assert_ne!(
        ptx_a, ptx_b,
        "different kernel sizes should produce different PTX"
    );
}

#[test]
fn test_bias_vs_no_bias_produce_different_ptx() {
    let config_no_bias = PtxConv1dConfig::new("conv", 64, 128, 3);
    let config_bias = PtxConv1dConfig::new("conv", 64, 128, 3).with_bias(true);
    let ptx_a = emit_ptx_conv1d(&config_no_bias).unwrap();
    let ptx_b = emit_ptx_conv1d(&config_bias).unwrap();
    assert_ne!(ptx_a, ptx_b, "bias vs no-bias should produce different PTX");
}

// ---------------------------------------------------------------------------
// Grid-stride advance
// ---------------------------------------------------------------------------

#[test]
fn test_ptx_grid_stride_advance() {
    let config = PtxConv1dConfig::new("conv", 64, 128, 3);
    let ptx = emit_ptx_conv1d(&config).unwrap();
    // Grid-stride: global_idx += grid_stride (stored in %r8)
    assert!(
        ptx.contains("add.u32       %r6, %r6, %r8"),
        "grid-stride loop must advance global_idx by grid_stride"
    );
}

// ---------------------------------------------------------------------------
// Accumulator initialization
// ---------------------------------------------------------------------------

#[test]
fn test_ptx_accumulator_initialized_to_zero() {
    let config = PtxConv1dConfig::new("conv", 64, 128, 3);
    let ptx = emit_ptx_conv1d(&config).unwrap();
    let zero_hex = format_ptx_float(0.0);
    assert!(
        ptx.contains(&format!("mov.f32       %f0, {zero_hex}")),
        "accumulator must be initialized to 0.0"
    );
}

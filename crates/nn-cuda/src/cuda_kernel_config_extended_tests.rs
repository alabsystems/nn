// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Extended tests for CUDA kernel launch configuration, grid/block dimensions,
//! shared memory calculation, register allocation patterns, and PTX instruction
//! generation across all PTX kernel modules.
//!
//! Covers:
//! 1. CudaDim3 and CudaLaunchConfig construction and edge cases
//! 2. Softmax config: warp/shared-memory thresholds, block sizing
//! 3. Matmul config: tile validation, shared memory, threads-per-block
//! 4. LayerNorm / RMSNorm config: warp boundary, shared memory
//! 5. Activation config: per-activation PTX correctness, alpha parameter
//! 6. Conv1d / Conv2d launch config: grid dimension calculation
//! 7. Elementwise launch config: grid-stride loop coverage
//! 8. Reduction launch config: row-per-block, shared memory
//! 9. Embedding launch config: total elements mapping
//! 10. Structural PTX validation: instructions, registers, entry points

use crate::codegen_ptx::{
    cuda_type, format_ptx_float, ptx_prelude, ptx_type, ptx_type_bytes, safe_ptx_uint,
    DEFAULT_SM_TARGET, PTX_BLOCK_SIZE, PTX_VERSION, WARP_SIZE,
};
use crate::cuda_ffi::{sm_target, CudaDim3, CudaLaunchConfig, CudaMemcpyKind};
use crate::cuda_validation::{
    validate_numerical, validate_ptx_structure, CudaValidationSuite, ErrorStats, ValidationResult,
};
use crate::ptx_activations::{
    emit_ptx_activation_default, gelu_reference, mish_reference,
    ptx_activation_launch_config, silu_reference, snake_reference, PtxActivation,
    PtxActivationConfig,
};
use crate::ptx_conv1d::{
    conv1d_output_length, ptx_conv1d_launch_config, PtxConv1dConfig, PTX_CONV1D_BLOCK_SIZE,
};
use crate::ptx_conv2d::{
    conv2d_output_size, ptx_conv2d_launch_config, PtxConv2dConfig, PTX_CONV2D_BLOCK_H,
    PTX_CONV2D_BLOCK_W,
};
use crate::ptx_elementwise::{
    add_reference, exp_reference, mul_reference, neg_reference,
    ptx_elementwise_launch_config, scalar_mul_reference, sqrt_reference,
    ELEMENTWISE_BLOCK_SIZE,
};
use crate::ptx_embedding::{
    embedding_reference, ptx_embedding_launch_config, PtxEmbeddingConfig, EMBEDDING_BLOCK_SIZE,
};
use crate::ptx_emit::{
    elementwise_launch_config, emit_elementwise_kernel, emit_matmul_kernel, emit_reduction_kernel,
    emit_softmax_kernel, matmul_launch_config, reduction_launch_config, ReductionOp,
};
use crate::ptx_layernorm::{emit_ptx_layernorm, ptx_layernorm_launch_config, PtxLayerNormConfig};
use crate::ptx_matmul::{
    emit_ptx_matmul, ptx_matmul_launch_config, PtxMatmulConfig, PTX_MATMUL_MAX_TILE, PTX_MATMUL_MIN_TILE, PTX_MATMUL_TILE_SIZE,
};
use crate::ptx_reduce::{
    argmax_reference, argmin_reference, max_reference, mean_reference, ptx_reduce_launch_config,
    sum_reference, REDUCE_BLOCK_SIZE,
};
use crate::ptx_residual::{
    residual_add_launch_config, residual_add_layernorm_launch_config,
    residual_add_relu_launch_config, RESIDUAL_BLOCK_SIZE,
};
use crate::ptx_rmsnorm::{emit_ptx_rmsnorm, ptx_rmsnorm_launch_config, PtxRmsNormConfig};
use crate::ptx_rope::{ptx_rope_launch_config, PtxRopeConfig, ROPE_BLOCK_SIZE};
use crate::ptx_softmax::{
    emit_ptx_softmax, log_softmax_reference, ptx_softmax_launch_config, softmax_reference,
    PtxSoftmaxConfig, SOFTMAX_BLOCK_SIZE,
};
use crate::ptx_transpose::{
    ptx_batch_transpose_launch_config, ptx_transpose_launch_config,
    transpose_reference, TRANSPOSE_BLOCK_SIZE,
};

use crate::ptx_cast::CAST_BLOCK_SIZE;
use crate::ptx_gather::GATHER_BLOCK_SIZE;
use crate::ptx_gemv::GEMV_BLOCK_SIZE;
use crate::ptx_instancenorm::INSTANCENORM_BLOCK_SIZE;
use crate::ptx_linear::LINEAR_BLOCK_SIZE;
use crate::ptx_pad::PAD_BLOCK_SIZE;
use crate::ptx_quantize::QUANTIZE_BLOCK_SIZE;
use crate::ptx_tensor_ops::TENSOR_OPS_BLOCK_SIZE;
use crate::ptx_upsample::UPSAMPLE_BLOCK_SIZE;
use crate::ptx_where::WHERE_BLOCK_SIZE;

use nn_dsl::ScalarType;

// =========================================================================
// Section 1: CudaDim3 construction and edge cases
// =========================================================================

#[test]
fn test_cuda_dim3_d1_total_is_x() {
    for x in [1, 32, 256, 1024, u32::MAX] {
        let d = CudaDim3::d1(x);
        assert_eq!(d.total(), u64::from(x));
        assert_eq!(d.y, 1);
        assert_eq!(d.z, 1);
    }
}

#[test]
fn test_cuda_dim3_d2_total_is_xy() {
    let d = CudaDim3::d2(16, 16);
    assert_eq!(d.total(), 256);
    let d = CudaDim3::d2(32, 8);
    assert_eq!(d.total(), 256);
}

#[test]
fn test_cuda_dim3_new_total_is_xyz() {
    let d = CudaDim3::new(4, 8, 2);
    assert_eq!(d.total(), 64);
    let d = CudaDim3::new(1, 1, 1);
    assert_eq!(d.total(), 1);
}

#[test]
fn test_cuda_dim3_large_product_no_overflow() {
    let d = CudaDim3::new(u32::MAX, 1, 1);
    assert_eq!(d.total(), u64::from(u32::MAX));
    let d = CudaDim3::new(65535, 65535, 1);
    assert_eq!(d.total(), 65535u64 * 65535u64);
}

// =========================================================================
// Section 2: CudaLaunchConfig factory methods
// =========================================================================

#[test]
fn test_launch_config_elementwise_exact_multiple() {
    let cfg = CudaLaunchConfig::for_elementwise(2048, 256);
    assert_eq!(cfg.grid.x, 8);
    assert_eq!(cfg.block.x, 256);
    assert_eq!(cfg.shared_mem_bytes, 0);
}

#[test]
fn test_launch_config_elementwise_non_multiple() {
    let cfg = CudaLaunchConfig::for_elementwise(1025, 256);
    assert_eq!(cfg.grid.x, 5); // ceil(1025/256) = 5
}

#[test]
fn test_launch_config_elementwise_single_element() {
    let cfg = CudaLaunchConfig::for_elementwise(1, 256);
    assert_eq!(cfg.grid.x, 1);
}

#[test]
fn test_launch_config_reduction_shared_mem_is_4_per_thread() {
    let cfg = CudaLaunchConfig::for_reduction(64, 128);
    assert_eq!(cfg.grid.x, 64);
    assert_eq!(cfg.block.x, 128);
    assert_eq!(cfg.shared_mem_bytes, 128 * 4);
}

#[test]
fn test_launch_config_matmul_2d_grid() {
    let cfg = CudaLaunchConfig::for_matmul(256, 512, 16, 16);
    assert_eq!(cfg.grid.x, 32); // ceil(512/16)
    assert_eq!(cfg.grid.y, 16); // ceil(256/16)
    assert_eq!(cfg.block.x, 16);
    assert_eq!(cfg.block.y, 16);
    assert_eq!(cfg.shared_mem_bytes, 0);
}

#[test]
fn test_launch_config_matmul_non_tile_multiple() {
    let cfg = CudaLaunchConfig::for_matmul(100, 100, 16, 16);
    assert_eq!(cfg.grid.x, 7); // ceil(100/16)
    assert_eq!(cfg.grid.y, 7);
}

#[test]
fn test_launch_config_batched_3d_grid() {
    let cfg = CudaLaunchConfig::for_batched(4, 8, 2, 256);
    assert_eq!(cfg.grid.x, 4);
    assert_eq!(cfg.grid.y, 8);
    assert_eq!(cfg.grid.z, 2);
    assert_eq!(cfg.block.x, 256);
}

// =========================================================================
// Section 3: Softmax config warp/shared-memory thresholds
// =========================================================================

#[test]
fn test_softmax_config_dim_1_is_warp_only() {
    let cfg = PtxSoftmaxConfig::new("sm1", 1);
    assert!(cfg.is_warp_only());
    assert_eq!(cfg.num_warps(), 1);
    assert_eq!(cfg.shared_memory_bytes(), 0);
    assert_eq!(cfg.block_size(), 32); // rounded up to warp
}

#[test]
fn test_softmax_config_dim_32_is_warp_only() {
    let cfg = PtxSoftmaxConfig::new("sm32", 32);
    assert!(cfg.is_warp_only());
    assert_eq!(cfg.block_size(), 32);
}

#[test]
fn test_softmax_config_dim_33_is_multi_warp() {
    let cfg = PtxSoftmaxConfig::new("sm33", 33);
    assert!(!cfg.is_warp_only());
    assert_eq!(cfg.block_size(), 64); // ceil(33/32)*32 = 64
    assert_eq!(cfg.num_warps(), 2);
    assert!(cfg.shared_memory_bytes() > 0);
}

#[test]
fn test_softmax_config_dim_256_max_block() {
    let cfg = PtxSoftmaxConfig::new("sm256", 256);
    assert_eq!(cfg.block_size(), 256);
    assert_eq!(cfg.num_warps(), 8);
    assert_eq!(cfg.shared_memory_bytes(), 8 * 4);
}

#[test]
fn test_softmax_config_dim_1024_caps_at_256() {
    let cfg = PtxSoftmaxConfig::new("sm1024", 1024);
    assert_eq!(cfg.block_size(), 256); // capped at MAX_BLOCK_SIZE
    assert_eq!(cfg.num_warps(), 8);
}

#[test]
fn test_softmax_launch_config_grid_is_num_rows() {
    let (grid, block) = ptx_softmax_launch_config(64, 128);
    assert_eq!(grid, [64, 1, 1]);
    assert_eq!(block[0], 128); // dim=128 -> block_size=128
}

#[test]
fn test_softmax_config_validate_rejects_zero_dim() {
    let cfg = PtxSoftmaxConfig::new("bad", 0);
    assert!(cfg.validate().is_err());
}

#[test]
fn test_softmax_config_validate_rejects_empty_name() {
    let cfg = PtxSoftmaxConfig::new("", 64);
    assert!(cfg.validate().is_err());
}

#[test]
fn test_softmax_ptx_generation_entry_point() {
    let cfg = PtxSoftmaxConfig::new("test_sm", 64);
    let ptx = emit_ptx_softmax(&cfg).unwrap();
    assert!(ptx.contains(".entry test_sm"));
    assert!(ptx.contains(".version"));
    assert!(ptx.contains(".target sm_80"));
}

#[test]
fn test_log_softmax_ptx_has_lg2_instruction() {
    let cfg = PtxSoftmaxConfig::new_log("test_lsm", 64);
    let ptx = emit_ptx_softmax(&cfg).unwrap();
    assert!(ptx.contains("lg2.approx.f32"));
}

// =========================================================================
// Section 4: Matmul config: tile validation and shared memory
// =========================================================================

#[test]
fn test_matmul_config_default_tile_16() {
    let cfg = PtxMatmulConfig::new("mm");
    assert_eq!(cfg.tile_size, PTX_MATMUL_TILE_SIZE);
    assert_eq!(cfg.tile_size, 16);
    assert!(cfg.validate().is_ok());
}

#[test]
fn test_matmul_config_shared_memory_bytes() {
    let cfg = PtxMatmulConfig::new("mm").with_tile_size(16);
    assert_eq!(cfg.shared_memory_bytes(), 2 * 16 * 16 * 4);
    assert_eq!(cfg.shared_memory_bytes(), 2048);
}

#[test]
fn test_matmul_config_shared_memory_max_tile() {
    let cfg = PtxMatmulConfig::new("mm32").with_tile_size(32);
    assert_eq!(cfg.shared_memory_bytes(), 8192);
}

#[test]
fn test_matmul_config_threads_per_block() {
    let cfg = PtxMatmulConfig::new("mm");
    assert_eq!(cfg.threads_per_block(), 256); // 16*16
    let cfg = PtxMatmulConfig::new("mm32").with_tile_size(32);
    assert_eq!(cfg.threads_per_block(), 1024); // 32*32
}

#[test]
fn test_matmul_config_validate_rejects_tile_too_small() {
    let cfg = PtxMatmulConfig::new("mm").with_tile_size(2);
    assert!(cfg.validate().is_err());
}

#[test]
fn test_matmul_config_validate_rejects_tile_too_large() {
    let cfg = PtxMatmulConfig::new("mm").with_tile_size(64);
    assert!(cfg.validate().is_err());
}

#[test]
fn test_matmul_config_validate_rejects_empty_name() {
    let cfg = PtxMatmulConfig {
        kernel_name: String::new(),
        tile_size: 16,
        sm_target: "sm_80".to_string(),
    };
    assert!(cfg.validate().is_err());
}

#[test]
fn test_matmul_launch_config_grid_block_dimensions() {
    let (grid, block) = ptx_matmul_launch_config(128, 256, 16);
    assert_eq!(grid, [16, 8, 1]); // ceil(256/16)=16, ceil(128/16)=8
    assert_eq!(block, [16, 16, 1]);
}

#[test]
fn test_matmul_launch_config_non_tile_multiple() {
    let (grid, _block) = ptx_matmul_launch_config(17, 33, 16);
    assert_eq!(grid[0], 3); // ceil(33/16)
    assert_eq!(grid[1], 2); // ceil(17/16)
}

#[test]
fn test_matmul_config_with_sm_target() {
    let cfg = PtxMatmulConfig::new("mm").with_sm_target("sm_90");
    assert_eq!(cfg.sm_target, "sm_90");
}

#[test]
fn test_matmul_ptx_generation_contains_shared_memory() {
    let cfg = PtxMatmulConfig::new("test_mm");
    let ptx = emit_ptx_matmul(&cfg).unwrap();
    assert!(ptx.contains(".shared"));
    assert!(ptx.contains(".entry test_mm"));
}

// =========================================================================
// Section 5: LayerNorm / RMSNorm config
// =========================================================================

#[test]
fn test_layernorm_config_block_size_warp_boundary() {
    let cfg = PtxLayerNormConfig::new("ln", 32, 1e-5);
    assert_eq!(cfg.block_size(), 32);
    let cfg = PtxLayerNormConfig::new("ln", 33, 1e-5);
    assert_eq!(cfg.block_size(), 64);
    let cfg = PtxLayerNormConfig::new("ln", 768, 1e-5);
    assert_eq!(cfg.block_size(), 256); // capped at 256
}

#[test]
fn test_rmsnorm_config_block_size_warp_boundary() {
    let cfg = PtxRmsNormConfig::new("rms", 32, 1e-6);
    assert_eq!(cfg.block_size(), 32);
    assert!(cfg.is_warp_only());
    let cfg = PtxRmsNormConfig::new("rms", 64, 1e-6);
    assert_eq!(cfg.block_size(), 64);
    assert_eq!(cfg.num_warps(), 2);
    assert!(!cfg.is_warp_only());
}

#[test]
fn test_rmsnorm_shared_memory_multi_warp() {
    let cfg = PtxRmsNormConfig::new("rms", 128, 1e-6);
    assert_eq!(cfg.num_warps(), 4);
    assert_eq!(cfg.shared_memory_bytes(), 4 * 4); // 4 warps * 4 bytes
}

#[test]
fn test_rmsnorm_shared_memory_warp_only() {
    let cfg = PtxRmsNormConfig::new("rms", 16, 1e-6);
    assert!(cfg.is_warp_only());
    assert_eq!(cfg.shared_memory_bytes(), 0);
}

#[test]
fn test_layernorm_launch_config_grid_is_num_rows() {
    let (grid, block) = ptx_layernorm_launch_config(32, 768);
    assert_eq!(grid, [32, 1, 1]);
    assert_eq!(block[0], 256); // 768 rounds to 256 cap
}

#[test]
fn test_rmsnorm_launch_config_grid_is_num_rows() {
    let (grid, block) = ptx_rmsnorm_launch_config(16, 256);
    assert_eq!(grid, [16, 1, 1]);
    assert_eq!(block[0], 256);
}

#[test]
fn test_layernorm_ptx_entry_point() {
    let cfg = PtxLayerNormConfig::new("test_ln", 128, 1e-5);
    let ptx = emit_ptx_layernorm(&cfg).unwrap();
    assert!(ptx.contains(".entry test_ln"));
}

#[test]
fn test_rmsnorm_ptx_entry_point() {
    let cfg = PtxRmsNormConfig::new("test_rms", 128, 1e-6);
    let ptx = emit_ptx_rmsnorm(&cfg).unwrap();
    assert!(ptx.contains(".entry test_rms"));
}

// =========================================================================
// Section 6: Activation config and per-activation PTX
// =========================================================================

#[test]
fn test_activation_config_default_block_size() {
    let cfg = PtxActivationConfig::new("act", PtxActivation::Silu);
    assert_eq!(cfg.block_size, 256);
}

#[test]
fn test_activation_config_custom_block_size() {
    let cfg = PtxActivationConfig::new("act", PtxActivation::Gelu).with_block_size(128);
    assert_eq!(cfg.block_size, 128);
}

#[test]
fn test_activation_config_validate_rejects_empty_name() {
    let cfg = PtxActivationConfig::new("", PtxActivation::Silu);
    assert!(cfg.validate().is_err());
}

#[test]
fn test_activation_config_validate_rejects_zero_block() {
    let cfg = PtxActivationConfig::new("act", PtxActivation::Silu).with_block_size(0);
    assert!(cfg.validate().is_err());
}

#[test]
fn test_snake_requires_alpha() {
    assert!(PtxActivation::Snake.requires_alpha());
    assert!(!PtxActivation::Silu.requires_alpha());
    assert!(!PtxActivation::Gelu.requires_alpha());
    assert!(!PtxActivation::GeluFast.requires_alpha());
    assert!(!PtxActivation::Mish.requires_alpha());
}

#[test]
fn test_activation_names() {
    assert_eq!(PtxActivation::Gelu.name(), "gelu");
    assert_eq!(PtxActivation::GeluFast.name(), "gelu_fast");
    assert_eq!(PtxActivation::Silu.name(), "silu");
    assert_eq!(PtxActivation::Mish.name(), "mish");
    assert_eq!(PtxActivation::Snake.name(), "snake");
}

#[test]
fn test_each_activation_generates_valid_ptx() {
    for act in [
        PtxActivation::Gelu,
        PtxActivation::GeluFast,
        PtxActivation::Silu,
        PtxActivation::Mish,
        PtxActivation::Snake,
    ] {
        let name = format!("test_{}", act.name());
        let ptx = emit_ptx_activation_default(&name, act).unwrap();
        assert!(
            ptx.contains(&format!(".entry {name}")),
            "{act:?} missing .entry"
        );
        assert!(ptx.contains(".version"), "{act:?} missing .version");
        assert!(ptx.contains(".reg"), "{act:?} missing register decls");
    }
}

#[test]
fn test_snake_ptx_has_alpha_param() {
    let ptx = emit_ptx_activation_default("snake_k", PtxActivation::Snake).unwrap();
    assert!(
        ptx.contains("param_alpha"),
        "Snake kernel must accept alpha parameter"
    );
}

#[test]
fn test_activation_launch_config_grid_stride() {
    let (grid, block) = ptx_activation_launch_config(1024, 256);
    assert_eq!(grid, [4, 1, 1]);
    assert_eq!(block, [256, 1, 1]);
}

#[test]
fn test_activation_launch_config_non_multiple() {
    let (grid, _block) = ptx_activation_launch_config(257, 256);
    assert_eq!(grid, [2, 1, 1]); // ceil(257/256)
}

// =========================================================================
// Section 7: Conv1d / Conv2d launch config
// =========================================================================

#[test]
fn test_conv1d_output_length_no_padding() {
    let l = conv1d_output_length(100, 3, 1, 0, 1).unwrap();
    assert_eq!(l, 98);
}

#[test]
fn test_conv1d_output_length_with_padding() {
    let l = conv1d_output_length(100, 3, 1, 1, 1).unwrap();
    assert_eq!(l, 100); // same padding
}

#[test]
fn test_conv1d_output_length_stride_2() {
    let l = conv1d_output_length(100, 3, 2, 0, 1).unwrap();
    assert_eq!(l, 49); // (100 - 3) / 2 + 1
}

#[test]
fn test_conv1d_launch_config_total_elements() {
    let cfg = ptx_conv1d_launch_config(2, 64, 50);
    // total = 2*64*50 = 6400
    let expected_grid = (6400u64.div_ceil(PTX_CONV1D_BLOCK_SIZE as u64)) as u32;
    assert_eq!(cfg.grid.x, expected_grid);
    assert_eq!(cfg.block.x, PTX_CONV1D_BLOCK_SIZE as u32);
}

#[test]
fn test_conv2d_output_size_basic() {
    // dim=28, kernel=3, stride=1, pad=0, dilation=1
    let h = conv2d_output_size(28, 3, 1, 0, 1).unwrap();
    assert_eq!(h, 26);
}

#[test]
fn test_conv2d_output_size_same_padding() {
    let h = conv2d_output_size(28, 3, 1, 1, 1).unwrap();
    assert_eq!(h, 28);
}

#[test]
fn test_conv2d_launch_config_3d_grid() {
    let config = PtxConv2dConfig::new("c2d", 3, 3);
    let (grid, block) = ptx_conv2d_launch_config(26, 26, 2, 64, &config);
    // grid_z = batch * c_out = 2*64 = 128
    assert_eq!(grid[2], 128);
    assert_eq!(block[0], config.block_w);
    assert_eq!(block[1], config.block_h);
}

// =========================================================================
// Section 8: Elementwise launch config
// =========================================================================

#[test]
fn test_elementwise_launch_config_exact() {
    let (grid, block) = elementwise_launch_config(PTX_BLOCK_SIZE);
    assert_eq!(grid, 1);
    assert_eq!(block, PTX_BLOCK_SIZE);
}

#[test]
fn test_elementwise_launch_config_large() {
    let (grid, block) = elementwise_launch_config(1_000_000);
    assert_eq!(block, PTX_BLOCK_SIZE);
    assert_eq!(grid, 1_000_000usize.div_ceil(PTX_BLOCK_SIZE));
}

#[test]
fn test_ptx_elementwise_launch_config_returns_tuple() {
    let (grid, block) = ptx_elementwise_launch_config(2048);
    assert_eq!(block[0], ELEMENTWISE_BLOCK_SIZE);
    assert_eq!(grid[0], 8); // ceil(2048/256)
}

// =========================================================================
// Section 9: Reduction launch config
// =========================================================================

#[test]
fn test_reduction_launch_config_one_block_per_row() {
    let (num_blocks, block_size) = reduction_launch_config(32, 128);
    assert_eq!(num_blocks, 32);
    assert_eq!(block_size, 128);
}

#[test]
fn test_reduction_launch_config_small_row() {
    let (num_blocks, block_size) = reduction_launch_config(10, 8);
    assert_eq!(num_blocks, 10);
    assert_eq!(block_size, 8);
}

#[test]
fn test_ptx_reduce_launch_config_default() {
    let (grid, block) = ptx_reduce_launch_config();
    assert_eq!(grid, [1, 1, 1]);
    assert_eq!(block[0], REDUCE_BLOCK_SIZE as usize);
}

// =========================================================================
// Section 10: Embedding launch config
// =========================================================================

#[test]
fn test_embedding_launch_config_total_elements() {
    let config = PtxEmbeddingConfig::new(50000, 768);
    let (grid, block) = ptx_embedding_launch_config(32, &config);
    // total = 32 * 768 = 24576
    let expected_grid = (24576u64.div_ceil(u64::from(EMBEDDING_BLOCK_SIZE))) as u32;
    assert_eq!(grid, expected_grid);
    assert_eq!(block, EMBEDDING_BLOCK_SIZE);
}

// =========================================================================
// Section 11: Transpose and RoPE launch configs
// =========================================================================

#[test]
fn test_transpose_launch_config_square() {
    let (grid, block) = ptx_transpose_launch_config(256, 256);
    assert_eq!(grid[0], 256 / TRANSPOSE_BLOCK_SIZE);
    assert_eq!(grid[1], 256 / TRANSPOSE_BLOCK_SIZE);
    assert_eq!(block[0], TRANSPOSE_BLOCK_SIZE);
    assert_eq!(block[1], TRANSPOSE_BLOCK_SIZE);
}

#[test]
fn test_batch_transpose_launch_config_includes_batch() {
    let (grid, _block) = ptx_batch_transpose_launch_config(4, 64, 64);
    assert_eq!(grid[2], 4); // batch dimension
}

#[test]
fn test_rope_launch_config_returns_tuple() {
    let config = PtxRopeConfig::new(32, 128);
    let (grid, block) = ptx_rope_launch_config(32, &config);
    // total_pairs = 32 * (128/2) = 2048
    let expected_grid = (2048u64.div_ceil(u64::from(ROPE_BLOCK_SIZE))) as u32;
    assert_eq!(grid, expected_grid);
    assert_eq!(block, ROPE_BLOCK_SIZE);
}

// =========================================================================
// Section 12: Block size constants are warp-aligned
// =========================================================================

#[test]
fn test_all_block_sizes_are_warp_aligned() {
    let block_sizes: Vec<(&str, u32)> = vec![
        ("PTX_BLOCK_SIZE", PTX_BLOCK_SIZE as u32),
        ("SOFTMAX_BLOCK_SIZE", SOFTMAX_BLOCK_SIZE),
        ("ELEMENTWISE_BLOCK_SIZE", ELEMENTWISE_BLOCK_SIZE),
        ("REDUCE_BLOCK_SIZE", REDUCE_BLOCK_SIZE),
        ("EMBEDDING_BLOCK_SIZE", EMBEDDING_BLOCK_SIZE),
        // TRANSPOSE_BLOCK_SIZE is a 2D tile dimension (TILE x TILE), so the
        // effective threads-per-block is its square (16x16 = 256), which is
        // what must be warp-aligned -- not the per-axis tile dimension itself.
        (
            "TRANSPOSE_BLOCK_SIZE^2",
            TRANSPOSE_BLOCK_SIZE * TRANSPOSE_BLOCK_SIZE,
        ),
        ("ROPE_BLOCK_SIZE", ROPE_BLOCK_SIZE),
        ("RESIDUAL_BLOCK_SIZE", RESIDUAL_BLOCK_SIZE),
        ("GATHER_BLOCK_SIZE", GATHER_BLOCK_SIZE),
        ("WHERE_BLOCK_SIZE", WHERE_BLOCK_SIZE),
        ("CAST_BLOCK_SIZE", CAST_BLOCK_SIZE),
        ("QUANTIZE_BLOCK_SIZE", QUANTIZE_BLOCK_SIZE),
        ("PAD_BLOCK_SIZE", PAD_BLOCK_SIZE),
        ("UPSAMPLE_BLOCK_SIZE", UPSAMPLE_BLOCK_SIZE),
        ("INSTANCENORM_BLOCK_SIZE", INSTANCENORM_BLOCK_SIZE),
        ("TENSOR_OPS_BLOCK_SIZE", TENSOR_OPS_BLOCK_SIZE),
        ("GEMV_BLOCK_SIZE", GEMV_BLOCK_SIZE),
        ("LINEAR_BLOCK_SIZE", LINEAR_BLOCK_SIZE),
        ("PTX_CONV1D_BLOCK_SIZE", PTX_CONV1D_BLOCK_SIZE as u32),
    ];

    for (name, size) in &block_sizes {
        assert!(
            *size % (WARP_SIZE as u32) == 0,
            "{name}={size} is not a multiple of WARP_SIZE={WARP_SIZE}"
        );
        assert!(*size > 0, "{name} must be > 0");
        assert!(*size <= 1024, "{name}={size} exceeds max block size 1024");
    }
}

// =========================================================================
// Section 13: PTX codegen helper functions
// =========================================================================

#[test]
fn test_format_ptx_float_special_values() {
    assert_eq!(format_ptx_float(f32::INFINITY), "0x7F800000");
    assert_eq!(format_ptx_float(f32::NEG_INFINITY), "0xFF800000");
    assert_eq!(format_ptx_float(f32::NAN), "0x7FC00000");
}

#[test]
fn test_format_ptx_float_known_values() {
    assert_eq!(format_ptx_float(0.0), "0f00000000");
    assert_eq!(format_ptx_float(1.0), "0f3F800000");
    assert_eq!(format_ptx_float(-1.0), "0fBF800000");
    assert_eq!(format_ptx_float(0.5), "0f3F000000");
}

#[test]
fn test_ptx_type_mapping_completeness() {
    assert_eq!(ptx_type(ScalarType::F32).unwrap(), ".f32");
    assert_eq!(ptx_type(ScalarType::F16).unwrap(), ".f16");
    assert_eq!(ptx_type(ScalarType::BF16).unwrap(), ".b16");
}

#[test]
fn test_ptx_type_bytes_correctness() {
    assert_eq!(ptx_type_bytes(ScalarType::F32).unwrap(), 4);
    assert_eq!(ptx_type_bytes(ScalarType::F16).unwrap(), 2);
    assert_eq!(ptx_type_bytes(ScalarType::BF16).unwrap(), 2);
}

#[test]
fn test_cuda_type_mapping_correctness() {
    assert_eq!(cuda_type(ScalarType::F32).unwrap(), "float");
    assert_eq!(cuda_type(ScalarType::F16).unwrap(), "__half");
    assert_eq!(cuda_type(ScalarType::BF16).unwrap(), "__nv_bfloat16");
}

#[test]
fn test_safe_ptx_uint_boundary() {
    assert!(safe_ptx_uint(0).is_ok());
    assert!(safe_ptx_uint(u32::MAX as usize).is_ok());
    assert!(safe_ptx_uint(u32::MAX as usize + 1).is_err());
}

#[test]
fn test_ptx_prelude_all_sm_targets() {
    for target in [
        sm_target::SM_70,
        sm_target::SM_75,
        sm_target::SM_80,
        sm_target::SM_86,
        sm_target::SM_89,
        sm_target::SM_90,
        sm_target::SM_100,
    ] {
        let prelude = ptx_prelude(target);
        assert!(prelude.contains(&format!(".target {target}")));
        assert!(prelude.contains(".address_size 64"));
    }
}

// =========================================================================
// Section 14: Reference function correctness
// =========================================================================

#[test]
fn test_softmax_reference_sums_to_one() {
    let input = vec![1.0, 2.0, 3.0, 4.0];
    let output = softmax_reference(&input);
    let sum: f32 = output.iter().sum();
    assert!((sum - 1.0).abs() < 1e-5, "softmax must sum to 1, got {sum}");
}

#[test]
fn test_softmax_reference_monotonically_increasing() {
    let input = vec![1.0, 2.0, 3.0, 4.0];
    let output = softmax_reference(&input);
    for i in 1..output.len() {
        assert!(output[i] > output[i - 1], "softmax should preserve order");
    }
}

#[test]
fn test_log_softmax_reference_all_negative() {
    let input = vec![1.0, 2.0, 3.0];
    let output = log_softmax_reference(&input);
    for &v in &output {
        assert!(v < 0.0, "log-softmax values must be negative, got {v}");
    }
}

#[test]
fn test_silu_reference_zero_at_zero() {
    let val = silu_reference(0.0);
    assert!((val - 0.0).abs() < 1e-6);
}

#[test]
fn test_gelu_reference_near_zero_at_negative() {
    let val = gelu_reference(-3.0);
    assert!(val.abs() < 0.01, "gelu(-3) should be near 0, got {val}");
}

#[test]
fn test_mish_reference_zero_at_zero() {
    let val = mish_reference(0.0);
    assert!(val.abs() < 1e-6, "mish(0) should be 0, got {val}");
}

#[test]
fn test_snake_reference_identity_at_zero() {
    let val = snake_reference(0.0, 1.0);
    assert!(val.abs() < 1e-6, "snake(0, 1) should be 0, got {val}");
}

#[test]
fn test_elementwise_add_reference() {
    let a = vec![1.0, 2.0, 3.0];
    let b = vec![4.0, 5.0, 6.0];
    let result = add_reference(&a, &b);
    assert_eq!(result, vec![5.0, 7.0, 9.0]);
}

#[test]
fn test_elementwise_mul_reference() {
    let a = vec![2.0, 3.0, 4.0];
    let b = vec![0.5, 2.0, 0.25];
    let result = mul_reference(&a, &b);
    assert_eq!(result, vec![1.0, 6.0, 1.0]);
}

#[test]
fn test_sum_reference_simple() {
    let input = vec![1.0, 2.0, 3.0, 4.0];
    assert!((sum_reference(&input) - 10.0).abs() < 1e-6);
}

#[test]
fn test_mean_reference_simple() {
    let input = vec![1.0, 2.0, 3.0, 4.0];
    assert!((mean_reference(&input) - 2.5).abs() < 1e-6);
}

#[test]
fn test_max_reference_simple() {
    let input = vec![1.0, 4.0, 2.0, 3.0];
    assert!((max_reference(&input) - 4.0).abs() < 1e-6);
}

#[test]
fn test_argmax_reference_simple() {
    let input = vec![1.0, 4.0, 2.0, 3.0];
    assert_eq!(argmax_reference(&input), 1);
}

#[test]
fn test_argmin_reference_simple() {
    let input = vec![3.0, 1.0, 4.0, 2.0];
    assert_eq!(argmin_reference(&input), 1);
}

#[test]
fn test_transpose_reference_2x3() {
    let input = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]; // 2x3
    let output = transpose_reference(&input, 2, 3);
    assert_eq!(output, vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]); // 3x2
}

#[test]
fn test_embedding_reference_lookup() {
    let table = vec![
        0.1, 0.2, 0.3, // token 0
        0.4, 0.5, 0.6, // token 1
        0.7, 0.8, 0.9, // token 2
        1.0, 1.1, 1.2, // token 3
    ];
    let indices = [2, 0];
    let result = embedding_reference(
        &indices.iter().map(|x| *x as u32).collect::<Vec<u32>>(),
        &table,
        3,
    );
    assert_eq!(result.len(), 6);
    assert!((result[0] - 0.7).abs() < 1e-6);
    assert!((result[3] - 0.1).abs() < 1e-6);
}

// =========================================================================
// Section 15: Structural PTX validation
// =========================================================================

#[test]
fn test_validate_ptx_structure_valid_raw_ptx() {
    let ptx = ".version 6.5\n.target sm_80\n.visible .entry kern(\n\
               .param .u64 p\n)\n{\n.reg .f32 %f<1>;\nret;\n}\n";
    let result = validate_ptx_structure(ptx, "kern");
    assert!(
        result.structural_ok,
        "failures: {:?}",
        result.structural_failures
    );
}

#[test]
fn test_validate_ptx_structure_empty_string() {
    let result = validate_ptx_structure("", "k");
    assert!(!result.structural_ok);
}

#[test]
fn test_validate_ptx_structure_missing_version() {
    let ptx = ".target sm_80\n.visible .entry kern(\n\
               .param .u64 p\n)\n{\n.reg .f32 %f<1>;\nret;\n}\n";
    let result = validate_ptx_structure(ptx, "kern");
    // Missing .version but has .entry -> still has PTX markers
    assert!(!result.structural_ok);
}

#[test]
fn test_validate_numerical_identical() {
    let a = vec![1.0, 2.0, 3.0, 4.0];
    let result = validate_numerical("test", &a, &a, 1e-6).unwrap();
    assert!(result.passed());
}

#[test]
fn test_error_stats_relative_error() {
    let actual = vec![1.01, 2.01, 3.01];
    let expected = vec![1.0, 2.0, 3.0];
    let stats = ErrorStats::compute(&actual, &expected).unwrap();
    assert!(stats.max_abs_error < 0.02);
    assert!(stats.max_rel_error < 0.02);
    assert_eq!(stats.num_nans, 0);
}

// =========================================================================
// Section 16: CudaMemcpyKind and SM target constants
// =========================================================================

#[test]
fn test_memcpy_kind_variants() {
    assert_eq!(CudaMemcpyKind::HostToHost as i32, 0);
    assert_eq!(CudaMemcpyKind::HostToDevice as i32, 1);
    assert_eq!(CudaMemcpyKind::DeviceToHost as i32, 2);
    assert_eq!(CudaMemcpyKind::DeviceToDevice as i32, 3);
}

#[test]
fn test_sm_target_values() {
    assert_eq!(sm_target::SM_70, "sm_70");
    assert_eq!(sm_target::SM_75, "sm_75");
    assert_eq!(sm_target::SM_80, "sm_80");
    assert_eq!(sm_target::SM_86, "sm_86");
    assert_eq!(sm_target::SM_89, "sm_89");
    assert_eq!(sm_target::SM_90, "sm_90");
    assert_eq!(sm_target::SM_100, "sm_100");
}

// =========================================================================
// Section 17: ptx_emit module launch configs
// =========================================================================

#[test]
fn test_ptx_emit_matmul_launch_config_symmetry() {
    let (grid, block) = matmul_launch_config(64, 128, 16);
    assert_eq!(grid, [8, 4]); // [ceil(128/16), ceil(64/16)]
    assert_eq!(block, [16, 16]);
}

#[test]
fn test_ptx_emit_reduction_small_row() {
    let (num_blocks, block_size) = reduction_launch_config(16, 4);
    assert_eq!(num_blocks, 16);
    assert_eq!(block_size, 4);
}

// =========================================================================
// Section 18: CUDA C++ kernel emission
// =========================================================================

#[test]
fn test_emit_elementwise_kernel_contains_bounds_check() {
    let src = emit_elementwise_kernel("test_k", "x * 2.0f", 512).unwrap();
    assert!(src.contains("idx >= N"), "Must have bounds check");
    assert!(src.contains("__global__"));
}

#[test]
fn test_emit_elementwise_kernel_rejects_zero_elements() {
    let result = emit_elementwise_kernel("bad", "x", 0);
    assert!(result.is_err());
}

#[test]
fn test_emit_softmax_kernel_contains_shared_memory() {
    let src = emit_softmax_kernel(128).unwrap();
    assert!(src.contains("__shared__"));
    assert!(src.contains("softmax_kernel"));
}

#[test]
fn test_emit_softmax_kernel_rejects_zero_row_size() {
    let result = emit_softmax_kernel(0);
    assert!(result.is_err());
}

#[test]
fn test_emit_matmul_kernel_contains_tiling() {
    let src = emit_matmul_kernel("test_mm", 16).unwrap();
    assert!(src.contains("__global__"));
    assert!(src.contains("__shared__"));
}

#[test]
fn test_emit_reduction_kernel_sum() {
    let src = emit_reduction_kernel("red_sum", ReductionOp::Sum, 128).unwrap();
    assert!(src.contains("__global__"));
    assert!(src.contains("red_sum"));
}

#[test]
fn test_emit_reduction_kernel_max() {
    let src = emit_reduction_kernel("red_max", ReductionOp::Max, 128).unwrap();
    assert!(src.contains("red_max"));
}

// =========================================================================
// Section 19: Validation suite
// =========================================================================

#[test]
fn test_validation_suite_multiple_entries() {
    let mut suite = CudaValidationSuite::new();
    // Structural validation requires each entry's PTX to contain
    // `.entry <kernel_name>` matching the name passed to `add`, so generate
    // per-kernel PTX rather than reusing one kernel's PTX for all three.
    let ptx_for = |name: &str| {
        format!(
            ".version 6.5\n.target sm_80\n.visible .entry {name}(\n\
             .param .u64 p\n)\n{{\n.reg .f32 %f<1>;\nret;\n}}\n"
        )
    };
    suite.add("k1", ptx_for("k1"), vec![1.0], vec![1.0], 1e-5);
    suite.add("k2", ptx_for("k2"), vec![2.0], vec![2.0], 1e-5);
    suite.add("k3", ptx_for("k3"), vec![3.0], vec![3.0], 1e-5);
    assert_eq!(suite.len(), 3);
    assert!(!suite.is_empty());
    assert!(suite.run_all_pass());
}

#[test]
fn test_validation_result_structural_failure_not_passed() {
    let mut result = ValidationResult::new("fail");
    result.structural_ok = false;
    assert!(!result.passed());
}

// =========================================================================
// Section 20: Residual launch configs
// =========================================================================

#[test]
fn test_residual_add_launch_config() {
    let cfg = residual_add_launch_config(4096);
    assert_eq!(cfg.block.x, RESIDUAL_BLOCK_SIZE);
    let expected_grid = (4096u64.div_ceil(u64::from(RESIDUAL_BLOCK_SIZE))) as u32;
    assert_eq!(cfg.grid.x, expected_grid);
}

#[test]
fn test_residual_add_relu_launch_config() {
    let cfg = residual_add_relu_launch_config(1024);
    assert_eq!(cfg.block.x, RESIDUAL_BLOCK_SIZE);
}

#[test]
fn test_residual_add_layernorm_launch_config() {
    let cfg = residual_add_layernorm_launch_config(32, 768);
    assert_eq!(cfg.grid.x, 32); // one block per row
}

// =========================================================================
// Section 21: PtxConv1dConfig construction
// =========================================================================

#[test]
fn test_conv1d_config_defaults() {
    let cfg = PtxConv1dConfig::new("c1d", 3, 64, 3);
    assert_eq!(cfg.stride, 1);
    assert_eq!(cfg.padding, 0);
    assert_eq!(cfg.dilation, 1);
    assert_eq!(cfg.groups, 1);
    assert!(!cfg.use_bias);
    assert_eq!(cfg.block_size, PTX_CONV1D_BLOCK_SIZE);
}

// =========================================================================
// Section 22: PtxConv2dConfig construction
// =========================================================================

#[test]
fn test_conv2d_config_defaults() {
    let cfg = PtxConv2dConfig::new("c2d", 3, 3);
    assert_eq!(cfg.stride_h, 1);
    assert_eq!(cfg.stride_w, 1);
    assert_eq!(cfg.pad_h, 0);
    assert_eq!(cfg.pad_w, 0);
    assert_eq!(cfg.block_h, PTX_CONV2D_BLOCK_H);
    assert_eq!(cfg.block_w, PTX_CONV2D_BLOCK_W);
}

// =========================================================================
// Section 23: PTX constant correctness
// =========================================================================

#[test]
fn test_ptx_version_is_6_5() {
    assert_eq!(PTX_VERSION, "6.5");
}

#[test]
fn test_warp_size_is_32() {
    assert_eq!(WARP_SIZE, 32);
}

#[test]
fn test_default_sm_target_is_sm_80() {
    assert_eq!(DEFAULT_SM_TARGET, "sm_80");
}

#[test]
fn test_matmul_tile_range() {
    assert_eq!(PTX_MATMUL_MIN_TILE, 4);
    assert_eq!(PTX_MATMUL_MAX_TILE, 32);
    assert!(PTX_MATMUL_TILE_SIZE >= PTX_MATMUL_MIN_TILE);
    assert!(PTX_MATMUL_TILE_SIZE <= PTX_MATMUL_MAX_TILE);
}

// =========================================================================
// Section 24: Scalar mul and other elementwise references
// =========================================================================

#[test]
fn test_scalar_mul_reference() {
    let input = vec![1.0, 2.0, 3.0];
    let result = scalar_mul_reference(&input, 2.0);
    assert_eq!(result, vec![2.0, 4.0, 6.0]);
}

#[test]
fn test_neg_reference() {
    let input = vec![1.0, -2.0, 0.0];
    let result = neg_reference(&input);
    assert_eq!(result, vec![-1.0, 2.0, 0.0]);
}

#[test]
fn test_exp_reference_known_values() {
    let input = vec![0.0, 1.0];
    let result = exp_reference(&input);
    assert!((result[0] - 1.0).abs() < 1e-6);
    assert!((result[1] - std::f32::consts::E).abs() < 1e-5);
}

#[test]
fn test_sqrt_reference_known_values() {
    let input = vec![0.0, 1.0, 4.0, 9.0];
    let result = sqrt_reference(&input);
    assert!((result[0] - 0.0).abs() < 1e-6);
    assert!((result[1] - 1.0).abs() < 1e-6);
    assert!((result[2] - 2.0).abs() < 1e-6);
    assert!((result[3] - 3.0).abs() < 1e-6);
}

// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Tests for PTX emission utilities in `ptx_emit` and `codegen_ptx`.
//!
//! Covers PTX header/prelude generation, kernel signature formatting,
//! PTX instruction formatting helpers, grid/block dimension calculations,
//! and PTX string validation for required directives.

use crate::codegen_ptx::{
    format_ptx_float, ptx_prelude, ptx_type, ptx_type_bytes, safe_ptx_uint, DEFAULT_SM_TARGET,
    PTX_BLOCK_SIZE, PTX_VERSION, REDUCE_BLOCK_SIZE, WARP_SIZE,
};
use crate::ptx_emit::{
    elementwise_launch_config, emit_activation_kernels, emit_elementwise_kernel,
    emit_matmul_kernel, emit_reduction_kernel, emit_softmax_kernel, matmul_launch_config,
    reduction_launch_config, ReductionOp,
};
use nn_dsl::ScalarType;

// =========================================================================
// 1. PTX header / prelude generation
// =========================================================================

#[test]
fn test_ptx_prelude_sm_70_contains_required_directives() {
    let header = ptx_prelude("sm_70");
    assert!(
        header.contains(".version"),
        "PTX header must contain .version directive"
    );
    assert!(
        header.contains(".target sm_70"),
        "PTX header must contain .target with specified SM"
    );
    assert!(
        header.contains(".address_size 64"),
        "PTX header must contain .address_size 64 for 64-bit pointers"
    );
}

#[test]
fn test_ptx_prelude_sm_80_default_target() {
    let header = ptx_prelude(DEFAULT_SM_TARGET);
    assert!(header.contains(".target sm_80"));
}

#[test]
fn test_ptx_prelude_sm_90_hopper() {
    let header = ptx_prelude("sm_90");
    assert!(header.contains(".target sm_90"));
    assert!(header.contains(&format!(".version {PTX_VERSION}")));
}

#[test]
fn test_ptx_prelude_version_is_6_5() {
    let header = ptx_prelude("sm_70");
    assert!(
        header.contains(".version 6.5"),
        "PTX version should be 6.5 for sm_70+ support"
    );
}

#[test]
fn test_ptx_prelude_ordering_version_before_target() {
    let header = ptx_prelude("sm_80");
    let version_pos = header.find(".version").expect(".version not found");
    let target_pos = header.find(".target").expect(".target not found");
    assert!(
        version_pos < target_pos,
        ".version directive must appear before .target directive"
    );
}

#[test]
fn test_ptx_prelude_ordering_target_before_address_size() {
    let header = ptx_prelude("sm_80");
    let target_pos = header.find(".target").expect(".target not found");
    let addr_pos = header
        .find(".address_size")
        .expect(".address_size not found");
    assert!(
        target_pos < addr_pos,
        ".target directive must appear before .address_size directive"
    );
}

// =========================================================================
// 2. Kernel signature formatting (CUDA C++ emission in ptx_emit)
// =========================================================================

#[test]
fn test_elementwise_kernel_signature_has_global() {
    let src = emit_elementwise_kernel("nn_kernel", "x * 2.0f", 512).unwrap();
    assert!(
        src.contains("__global__ void nn_kernel"),
        "Elementwise kernel must have __global__ void signature"
    );
}

#[test]
fn test_elementwise_kernel_signature_has_restrict_params() {
    let src = emit_elementwise_kernel("k", "x", 1).unwrap();
    assert!(
        src.contains("const float* __restrict__ input"),
        "Input parameter must be const float* __restrict__"
    );
    assert!(
        src.contains("float* __restrict__ output"),
        "Output parameter must be float* __restrict__"
    );
    assert!(
        src.contains("const unsigned int N"),
        "Size parameter must be const unsigned int N"
    );
}

#[test]
fn test_elementwise_kernel_contains_op_expression() {
    let expr = "x > 0.0f ? x : 0.0f";
    let src = emit_elementwise_kernel("relu", expr, 256).unwrap();
    assert!(
        src.contains(expr),
        "Generated kernel must contain the operation expression"
    );
}

#[test]
fn test_activation_kernels_each_has_global_signature() {
    let src = emit_activation_kernels();
    let expected_kernels = [
        "relu_kernel",
        "silu_kernel",
        "sigmoid_kernel",
        "tanh_kernel",
        "gelu_kernel",
    ];
    for name in &expected_kernels {
        assert!(
            src.contains(&format!("__global__ void {name}")),
            "Activation kernel {name} must have __global__ void signature"
        );
    }
}

#[test]
fn test_matmul_kernel_signature_has_three_matrix_params() {
    let src = emit_matmul_kernel("gemm", 16).unwrap();
    assert!(src.contains("const float* __restrict__ A"));
    assert!(src.contains("const float* __restrict__ B"));
    assert!(src.contains("float* __restrict__ C"));
}

#[test]
fn test_matmul_kernel_signature_has_dimension_params() {
    let src = emit_matmul_kernel("gemm", 16).unwrap();
    assert!(src.contains("const unsigned int M"));
    assert!(src.contains("const unsigned int N"));
    assert!(src.contains("const unsigned int K"));
}

#[test]
fn test_softmax_kernel_signature_has_shared_memory() {
    let src = emit_softmax_kernel(128).unwrap();
    assert!(
        src.contains("extern __shared__ float sdata[]"),
        "Softmax kernel must declare shared memory"
    );
}

#[test]
fn test_reduction_kernel_signature_has_axis_size_param() {
    let src = emit_reduction_kernel("sum_k", ReductionOp::Sum, 64).unwrap();
    assert!(src.contains("const unsigned int axis_size"));
}

// =========================================================================
// 3. PTX instruction formatting helpers
// =========================================================================

#[test]
fn test_format_ptx_float_zero() {
    let s = format_ptx_float(0.0);
    assert_eq!(s, "0f00000000");
}

#[test]
fn test_format_ptx_float_one() {
    let s = format_ptx_float(1.0);
    assert_eq!(s, "0f3F800000");
}

#[test]
fn test_format_ptx_float_negative_one() {
    let s = format_ptx_float(-1.0);
    assert_eq!(s, "0fBF800000");
}

#[test]
fn test_format_ptx_float_infinity() {
    assert_eq!(format_ptx_float(f32::INFINITY), "0x7F800000");
}

#[test]
fn test_format_ptx_float_neg_infinity() {
    assert_eq!(format_ptx_float(f32::NEG_INFINITY), "0xFF800000");
}

#[test]
fn test_format_ptx_float_nan() {
    let s = format_ptx_float(f32::NAN);
    assert_eq!(s, "0x7FC00000");
}

#[test]
fn test_format_ptx_float_half() {
    // 0.5f32 = 0x3F000000
    let s = format_ptx_float(0.5);
    assert_eq!(s, "0f3F000000");
}

#[test]
fn test_format_ptx_float_small_denormal() {
    // Smallest positive denormal: f32::MIN_POSITIVE is 1.17549435e-38 (normal)
    // A denormal is below that, like 1e-45.
    let val = f32::from_bits(1); // smallest positive denormal
    let s = format_ptx_float(val);
    assert_eq!(s, "0f00000001");
}

#[test]
fn test_ptx_type_f32() {
    assert_eq!(ptx_type(ScalarType::F32).unwrap(), ".f32");
}

#[test]
fn test_ptx_type_f16() {
    assert_eq!(ptx_type(ScalarType::F16).unwrap(), ".f16");
}

#[test]
fn test_ptx_type_bf16() {
    assert_eq!(ptx_type(ScalarType::BF16).unwrap(), ".b16");
}

#[test]
fn test_ptx_type_bytes_f32_is_4() {
    assert_eq!(ptx_type_bytes(ScalarType::F32).unwrap(), 4);
}

#[test]
fn test_ptx_type_bytes_f16_is_2() {
    assert_eq!(ptx_type_bytes(ScalarType::F16).unwrap(), 2);
}

#[test]
fn test_ptx_type_bytes_bf16_is_2() {
    assert_eq!(ptx_type_bytes(ScalarType::BF16).unwrap(), 2);
}

#[test]
fn test_safe_ptx_uint_zero() {
    assert_eq!(safe_ptx_uint(0).unwrap(), "0");
}

#[test]
fn test_safe_ptx_uint_max_u32() {
    assert_eq!(
        safe_ptx_uint(u32::MAX as usize).unwrap(),
        u32::MAX.to_string()
    );
}

#[test]
fn test_safe_ptx_uint_overflow_rejected() {
    let result = safe_ptx_uint(u32::MAX as usize + 1);
    assert!(result.is_err());
}

// =========================================================================
// 4. Grid/block dimension calculation utilities
// =========================================================================

#[test]
fn test_elementwise_launch_config_exact_multiple() {
    let (grid, block) = elementwise_launch_config(256);
    assert_eq!(block, PTX_BLOCK_SIZE);
    assert_eq!(grid, 1);
}

#[test]
fn test_elementwise_launch_config_rounds_up() {
    let (grid, block) = elementwise_launch_config(257);
    assert_eq!(block, PTX_BLOCK_SIZE);
    assert_eq!(grid, 2); // ceil(257/256) = 2
}

#[test]
fn test_elementwise_launch_config_large() {
    let (grid, block) = elementwise_launch_config(1_000_000);
    assert_eq!(block, PTX_BLOCK_SIZE);
    assert_eq!(grid, 1_000_000_usize.div_ceil(PTX_BLOCK_SIZE));
}

#[test]
fn test_elementwise_launch_config_single_element() {
    let (grid, block) = elementwise_launch_config(1);
    assert_eq!(block, PTX_BLOCK_SIZE);
    assert_eq!(grid, 1);
}

#[test]
fn test_reduction_launch_config_one_block_per_row() {
    let (num_blocks, block_size) = reduction_launch_config(10, 128);
    assert_eq!(num_blocks, 10, "One block per row");
    assert!(
        block_size <= REDUCE_BLOCK_SIZE,
        "Block size must not exceed REDUCE_BLOCK_SIZE"
    );
    assert!(
        block_size.is_power_of_two(),
        "Block size must be a power of two"
    );
}

#[test]
fn test_reduction_launch_config_small_row() {
    let (num_blocks, block_size) = reduction_launch_config(4, 16);
    assert_eq!(num_blocks, 4);
    assert_eq!(block_size, 16); // next_power_of_two(16) = 16, min(256, 16) = 16
}

#[test]
fn test_reduction_launch_config_large_row_capped() {
    let (_, block_size) = reduction_launch_config(1, 10000);
    assert_eq!(
        block_size, REDUCE_BLOCK_SIZE,
        "Block size should be capped at REDUCE_BLOCK_SIZE for large rows"
    );
}

#[test]
fn test_matmul_launch_config_square() {
    let (grid, block) = matmul_launch_config(64, 64, 16);
    assert_eq!(grid, [4, 4]); // [ceil(64/16), ceil(64/16)]
    assert_eq!(block, [16, 16]);
}

#[test]
fn test_matmul_launch_config_non_divisible() {
    let (grid, block) = matmul_launch_config(100, 50, 16);
    assert_eq!(grid, [4, 7]); // [ceil(50/16)=4, ceil(100/16)=7]
    assert_eq!(block, [16, 16]);
}

#[test]
fn test_matmul_launch_config_small() {
    let (grid, block) = matmul_launch_config(4, 4, 4);
    assert_eq!(grid, [1, 1]);
    assert_eq!(block, [4, 4]);
}

// =========================================================================
// 5. PTX string validation -- required directives
// =========================================================================

#[test]
fn test_cuda_prelude_contains_runtime_header() {
    let src = emit_elementwise_kernel("k", "x", 1).unwrap();
    assert!(
        src.contains("#include <cuda_runtime.h>"),
        "CUDA C++ kernels must include cuda_runtime.h"
    );
}

#[test]
fn test_cuda_prelude_contains_fp16_header() {
    let src = emit_elementwise_kernel("k", "x", 1).unwrap();
    assert!(
        src.contains("#include <cuda_fp16.h>"),
        "CUDA C++ kernels must include cuda_fp16.h for half precision"
    );
}

#[test]
fn test_cuda_prelude_contains_bf16_header() {
    let src = emit_elementwise_kernel("k", "x", 1).unwrap();
    assert!(
        src.contains("#include <cuda_bf16.h>"),
        "CUDA C++ kernels must include cuda_bf16.h for bfloat16"
    );
}

#[test]
fn test_softmax_kernel_contains_syncthreads() {
    let src = emit_softmax_kernel(64).unwrap();
    assert!(
        src.contains("__syncthreads"),
        "Softmax kernel with shared memory must use __syncthreads"
    );
}

#[test]
fn test_matmul_kernel_contains_tile_size_define() {
    let src = emit_matmul_kernel("mm", 8).unwrap();
    assert!(
        src.contains("#define TILE_SIZE 8"),
        "Matmul kernel must define TILE_SIZE"
    );
    assert!(
        src.contains("#undef TILE_SIZE"),
        "Matmul kernel must undef TILE_SIZE to avoid pollution"
    );
}

#[test]
fn test_matmul_kernel_contains_shared_arrays() {
    let src = emit_matmul_kernel("mm", 16).unwrap();
    assert!(src.contains("__shared__ float As[TILE_SIZE][TILE_SIZE]"));
    assert!(src.contains("__shared__ float Bs[TILE_SIZE][TILE_SIZE]"));
}

#[test]
fn test_reduction_all_ops_produce_valid_cuda() {
    for op in [
        ReductionOp::Sum,
        ReductionOp::Max,
        ReductionOp::Min,
        ReductionOp::Mean,
    ] {
        let src = emit_reduction_kernel("red", op, 256).unwrap();
        assert!(
            src.contains("__global__"),
            "{op:?} reduction must contain __global__"
        );
        assert!(
            src.contains("__shared__"),
            "{op:?} reduction must contain __shared__"
        );
        assert!(
            src.contains("__syncthreads"),
            "{op:?} reduction must contain __syncthreads"
        );
    }
}

#[test]
fn test_constants_are_reasonable() {
    assert_eq!(PTX_BLOCK_SIZE, 256);
    assert_eq!(REDUCE_BLOCK_SIZE, 256);
    assert_eq!(WARP_SIZE, 32);
    assert_eq!(PTX_VERSION, "6.5");
    assert_eq!(DEFAULT_SM_TARGET, "sm_80");
}

// =========================================================================
// Edge cases and error handling
// =========================================================================

#[test]
fn test_elementwise_kernel_zero_elements_error() {
    let result = emit_elementwise_kernel("k", "x", 0);
    assert!(result.is_err());
}

#[test]
fn test_softmax_kernel_zero_row_size_error() {
    let result = emit_softmax_kernel(0);
    assert!(result.is_err());
}

#[test]
fn test_reduction_kernel_zero_axis_size_error() {
    let result = emit_reduction_kernel("k", ReductionOp::Sum, 0);
    assert!(result.is_err());
}

#[test]
fn test_matmul_kernel_tile_zero_error() {
    assert!(emit_matmul_kernel("k", 0).is_err());
}

#[test]
fn test_matmul_kernel_tile_too_large_error() {
    assert!(emit_matmul_kernel("k", 33).is_err());
}

#[test]
fn test_matmul_kernel_tile_boundary_32_ok() {
    assert!(emit_matmul_kernel("k", 32).is_ok());
}

#[test]
fn test_matmul_kernel_tile_boundary_1_ok() {
    assert!(emit_matmul_kernel("k", 1).is_ok());
}

#[test]
fn test_softmax_kernel_non_power_of_two_row_size() {
    // Should still work -- block size internally rounds to power of two
    let src = emit_softmax_kernel(100).unwrap();
    assert!(src.contains("softmax_kernel"));
}

#[test]
fn test_reduction_kernel_min_identity_is_huge_valf() {
    let src = emit_reduction_kernel("min_k", ReductionOp::Min, 128).unwrap();
    assert!(
        src.contains("HUGE_VALF"),
        "Min reduction identity should use HUGE_VALF"
    );
}

#[test]
fn test_reduction_kernel_mean_divides_by_axis_size() {
    let src = emit_reduction_kernel("mean_k", ReductionOp::Mean, 64).unwrap();
    assert!(
        src.contains("(float)axis_size"),
        "Mean reduction must divide by axis_size"
    );
}

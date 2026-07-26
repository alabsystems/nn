// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Comprehensive CUDA/PTX backend test suite (50+ tests).
//!
//! Covers PTX code generation for all activations and reductions, PTX syntax
//! structure validation, compilation pipeline types, runtime types and
//! launch configs, FFI types, and error handling. No live CUDA GPU required.

use nn_cuda::codegen_ptx::{
    cuda_type, format_ptx_float, ptx_accumulator_type, ptx_prelude, ptx_reg_type, ptx_type,
    ptx_type_bytes, safe_ptx_uint, PtxCodegenError, DEFAULT_SM_TARGET, PTX_BLOCK_SIZE, PTX_VERSION,
    REDUCE_BLOCK_SIZE, WARP_SIZE,
};
use nn_cuda::codegen_syntax_ptx::CudaSyntax;
use nn_cuda::compile_ptx::{
    check_nvcc, check_ptxas, nvcc_command, ptxas_command, PtxCompileError, PtxModule,
};
use nn_cuda::cuda_ffi::{error_code, sm_target, CudaDim3, CudaLaunchConfig, CudaMemcpyKind};
use nn_cuda::cuda_runtime::{is_cuda_available, CudaRuntime, CudaRuntimeError};
use nn_cuda::ptx_emit::{
    elementwise_launch_config, emit_activation_kernels, emit_elementwise_kernel,
    emit_matmul_kernel, emit_reduction_kernel, emit_softmax_kernel, matmul_launch_config,
    reduction_launch_config, ReductionOp, CUDA_PRELUDE,
};
use nn_dsl::codegen_syntax::CodegenSyntax;
use nn_dsl::ScalarType;
use std::path::{Path, PathBuf};

// ===========================================================================
// A. PTX Code Generation — Elementwise Activations (20+ tests)
// ===========================================================================

#[test]
fn test_emit_relu_kernel_structure() {
    let src = emit_elementwise_kernel("relu_kernel", "x > 0.0f ? x : 0.0f", 1024).unwrap();
    assert!(src.contains("__global__"), "must contain __global__");
    assert!(src.contains("relu_kernel"), "must contain kernel name");
    assert!(
        src.contains("x > 0.0f ? x : 0.0f"),
        "must contain op expression"
    );
    assert!(src.contains("blockIdx.x"), "must use block indexing");
    assert!(src.contains("threadIdx.x"), "must use thread indexing");
}

#[test]
fn test_emit_gelu_kernel_expression() {
    let gelu_expr = "0.5f * x * (1.0f + tanhf(0.7978845608f * (x + 0.044715f * x * x * x)))";
    let src = emit_elementwise_kernel("gelu_kernel", gelu_expr, 2048).unwrap();
    assert!(src.contains("gelu_kernel"));
    assert!(src.contains("tanhf"));
    assert!(src.contains("0.044715f"));
}

#[test]
fn test_emit_silu_kernel_expression() {
    let silu_expr = "x / (1.0f + expf(-x))";
    let src = emit_elementwise_kernel("silu_kernel", silu_expr, 512).unwrap();
    assert!(src.contains("silu_kernel"));
    assert!(src.contains("expf(-x)"));
}

#[test]
fn test_emit_sigmoid_kernel_expression() {
    let sigmoid_expr = "1.0f / (1.0f + expf(-x))";
    let src = emit_elementwise_kernel("sigmoid_kernel", sigmoid_expr, 256).unwrap();
    assert!(src.contains("sigmoid_kernel"));
    assert!(src.contains("1.0f / (1.0f + expf(-x))"));
}

#[test]
fn test_emit_tanh_kernel_expression() {
    let tanh_expr = "tanhf(x)";
    let src = emit_elementwise_kernel("tanh_kernel", tanh_expr, 128).unwrap();
    assert!(src.contains("tanh_kernel"));
    assert!(src.contains("tanhf(x)"));
}

#[test]
fn test_emit_leaky_relu_kernel_expression() {
    let leaky_relu_expr = "x > 0.0f ? x : 0.01f * x";
    let src = emit_elementwise_kernel("leaky_relu_kernel", leaky_relu_expr, 1024).unwrap();
    assert!(src.contains("leaky_relu_kernel"));
    assert!(src.contains("0.01f * x"));
}

#[test]
fn test_emit_snake_kernel_expression() {
    // Snake activation: x + (1/alpha) * sin(alpha * x)^2
    let snake_expr = "x + sinf(x) * sinf(x)";
    let src = emit_elementwise_kernel("snake_kernel", snake_expr, 1024).unwrap();
    assert!(src.contains("snake_kernel"));
    assert!(src.contains("sinf(x)"));
}

#[test]
fn test_emit_elu_kernel_expression() {
    let elu_expr = "x > 0.0f ? x : 1.0f * (expf(x) - 1.0f)";
    let src = emit_elementwise_kernel("elu_kernel", elu_expr, 1024).unwrap();
    assert!(src.contains("elu_kernel"));
    assert!(src.contains("expf(x) - 1.0f"));
}

#[test]
fn test_emit_swish_kernel_expression() {
    let swish_expr = "x * (1.0f / (1.0f + expf(-x)))";
    let src = emit_elementwise_kernel("swish_kernel", swish_expr, 512).unwrap();
    assert!(src.contains("swish_kernel"));
    assert!(src.contains("expf(-x)"));
}

#[test]
fn test_emit_hardswish_kernel_expression() {
    let hs_expr = "x * fminf(fmaxf(x + 3.0f, 0.0f), 6.0f) / 6.0f";
    let src = emit_elementwise_kernel("hardswish_kernel", hs_expr, 1024).unwrap();
    assert!(src.contains("hardswish_kernel"));
    assert!(src.contains("fminf"));
    assert!(src.contains("fmaxf"));
}

#[test]
fn test_emit_mish_kernel_expression() {
    let mish_expr = "x * tanhf(logf(1.0f + expf(x)))";
    let src = emit_elementwise_kernel("mish_kernel", mish_expr, 1024).unwrap();
    assert!(src.contains("mish_kernel"));
    assert!(src.contains("logf"));
    assert!(src.contains("tanhf"));
}

#[test]
fn test_emit_elementwise_kernel_zero_elements_rejected() {
    let result = emit_elementwise_kernel("k", "x", 0);
    assert!(result.is_err(), "zero elements must be rejected");
}

#[test]
fn test_emit_elementwise_kernel_includes_cuda_prelude() {
    let src = emit_elementwise_kernel("test_k", "x", 8).unwrap();
    assert!(src.contains("#include <cuda_runtime.h>"));
    assert!(src.contains("#include <cuda_fp16.h>"));
    assert!(src.contains("#include <cuda_bf16.h>"));
}

#[test]
fn test_emit_elementwise_kernel_bounds_check_present() {
    let src = emit_elementwise_kernel("bounded_k", "x * 2.0f", 1024).unwrap();
    assert!(
        src.contains("if (idx >= N) return;"),
        "kernel must have bounds check"
    );
}

#[test]
fn test_emit_elementwise_kernel_restrict_qualifiers() {
    let src = emit_elementwise_kernel("restrict_k", "x", 1024).unwrap();
    assert!(
        src.contains("__restrict__"),
        "kernel buffers should use __restrict__"
    );
}

#[test]
fn test_emit_activation_kernels_contains_all_five() {
    let src = emit_activation_kernels();
    let expected = [
        "relu_kernel",
        "silu_kernel",
        "sigmoid_kernel",
        "tanh_kernel",
        "gelu_kernel",
    ];
    for name in &expected {
        assert!(src.contains(name), "missing activation kernel: {name}");
    }
}

#[test]
fn test_emit_activation_kernels_global_count() {
    let src = emit_activation_kernels();
    assert_eq!(
        src.matches("__global__").count(),
        5,
        "should have exactly 5 __global__ kernels"
    );
}

#[test]
fn test_emit_activation_kernels_all_have_bounds_check() {
    let src = emit_activation_kernels();
    assert_eq!(
        src.matches("if (idx >= N) return;").count(),
        5,
        "all 5 kernels should have bounds checks"
    );
}

// -- Softmax PTX emission with shared memory --

#[test]
fn test_emit_softmax_kernel_shared_memory() {
    let src = emit_softmax_kernel(512).unwrap();
    assert!(src.contains("softmax_kernel"), "kernel name present");
    assert!(src.contains("__shared__"), "must use shared memory");
    assert!(src.contains("__syncthreads"), "must synchronize threads");
    assert!(src.contains("expf"), "must use expf for exponentiation");
    assert!(
        src.contains("-HUGE_VALF"),
        "must use -HUGE_VALF as identity for max"
    );
}

#[test]
fn test_emit_softmax_kernel_three_phases() {
    let src = emit_softmax_kernel(256).unwrap();
    assert!(src.contains("Phase 1: find max"), "must document phase 1");
    assert!(
        src.contains("Phase 2: exp(x - max)"),
        "must document phase 2"
    );
    assert!(src.contains("Phase 3: normalize"), "must document phase 3");
}

#[test]
fn test_emit_softmax_kernel_zero_row_size_rejected() {
    let result = emit_softmax_kernel(0);
    assert!(result.is_err(), "row_size=0 must be rejected");
}

#[test]
fn test_emit_softmax_kernel_small_row_size() {
    // row_size=4 should use a small block size (next_power_of_two = 4)
    let src = emit_softmax_kernel(4).unwrap();
    assert!(src.contains("softmax_kernel"));
    assert!(src.contains("__shared__"));
}

// -- Reduction PTX emission --

#[test]
fn test_emit_reduction_kernel_sum() {
    let src = emit_reduction_kernel("sum_k", ReductionOp::Sum, 1024).unwrap();
    assert!(src.contains("sum_k"));
    assert!(src.contains("__shared__"));
    assert!(src.contains("sdata[tid] +="));
    assert!(src.contains("0.0f"), "sum identity is 0.0f");
}

#[test]
fn test_emit_reduction_kernel_max() {
    let src = emit_reduction_kernel("max_k", ReductionOp::Max, 256).unwrap();
    assert!(src.contains("max_k"));
    assert!(src.contains("-HUGE_VALF"), "max identity is -HUGE_VALF");
}

#[test]
fn test_emit_reduction_kernel_min() {
    let src = emit_reduction_kernel("min_k", ReductionOp::Min, 256).unwrap();
    assert!(src.contains("min_k"));
    assert!(src.contains("HUGE_VALF"), "min identity is HUGE_VALF");
    assert!(
        src.contains("if (v < val) val = v") || src.contains("sdata[tid + s] < sdata[tid]"),
        "min must compare less-than"
    );
}

#[test]
fn test_emit_reduction_kernel_mean_divides_by_axis_size() {
    let src = emit_reduction_kernel("mean_k", ReductionOp::Mean, 128).unwrap();
    assert!(
        src.contains("(float)axis_size"),
        "mean divides by axis_size cast to float"
    );
}

#[test]
fn test_emit_reduction_kernel_zero_axis_rejected() {
    let result = emit_reduction_kernel("bad_k", ReductionOp::Sum, 0);
    assert!(result.is_err(), "axis_size=0 must be rejected");
}

// -- MatMul PTX emission --

#[test]
fn test_emit_matmul_kernel_tiled() {
    let src = emit_matmul_kernel("gemm_k", 16).unwrap();
    assert!(src.contains("gemm_k"));
    assert!(src.contains("#define TILE_SIZE 16"));
    assert!(src.contains("__shared__"));
    assert!(src.contains("__syncthreads"));
    assert!(src.contains("As[threadIdx.y][threadIdx.x]"), "loads tile A");
    assert!(src.contains("Bs[threadIdx.y][threadIdx.x]"), "loads tile B");
}

#[test]
fn test_emit_matmul_kernel_accumulator() {
    let src = emit_matmul_kernel("acc_k", 8).unwrap();
    assert!(
        src.contains("float acc = 0.0f;"),
        "accumulator must be float"
    );
}

#[test]
fn test_emit_matmul_kernel_bounds_guard() {
    let src = emit_matmul_kernel("guard_k", 16).unwrap();
    assert!(
        src.contains("if (row < M && col < N)"),
        "matmul must guard output write"
    );
}

#[test]
fn test_emit_matmul_kernel_tile_size_1() {
    // Minimum valid tile size
    let src = emit_matmul_kernel("tiny_k", 1).unwrap();
    assert!(src.contains("#define TILE_SIZE 1"));
}

#[test]
fn test_emit_matmul_kernel_tile_size_32() {
    // Maximum valid tile size
    let src = emit_matmul_kernel("max_tile_k", 32).unwrap();
    assert!(src.contains("#define TILE_SIZE 32"));
}

#[test]
fn test_emit_matmul_kernel_tile_size_0_rejected() {
    assert!(emit_matmul_kernel("k", 0).is_err());
}

#[test]
fn test_emit_matmul_kernel_tile_size_64_rejected() {
    assert!(emit_matmul_kernel("k", 64).is_err());
}

#[test]
fn test_emit_matmul_kernel_undef_tile() {
    let src = emit_matmul_kernel("k", 16).unwrap();
    assert!(
        src.contains("#undef TILE_SIZE"),
        "must undef TILE_SIZE after kernel"
    );
}

// ===========================================================================
// B. PTX Syntax Structure (8+ tests)
// ===========================================================================

#[test]
fn test_ptx_prelude_version_directive() {
    let prelude = ptx_prelude("sm_80");
    assert!(
        prelude.contains(".version 6.5"),
        "must contain PTX version directive"
    );
}

#[test]
fn test_ptx_prelude_target_directive() {
    let prelude = ptx_prelude("sm_90");
    assert!(
        prelude.contains(".target sm_90"),
        "must contain .target directive"
    );
}

#[test]
fn test_ptx_prelude_address_size() {
    let prelude = ptx_prelude("sm_80");
    assert!(
        prelude.contains(".address_size 64"),
        "must declare 64-bit addressing for buffer pointers"
    );
}

#[test]
fn test_ptx_type_f32_register_type() {
    assert_eq!(ptx_type(ScalarType::F32).unwrap(), ".f32");
    assert_eq!(ptx_reg_type(ScalarType::F32).unwrap(), ".f32");
}

#[test]
fn test_ptx_type_f16_register_type() {
    assert_eq!(ptx_type(ScalarType::F16).unwrap(), ".f16");
    assert_eq!(ptx_reg_type(ScalarType::F16).unwrap(), ".f16");
}

#[test]
fn test_ptx_type_bf16_uses_b16() {
    // BF16 is stored as bitfield in PTX
    assert_eq!(ptx_type(ScalarType::BF16).unwrap(), ".b16");
    assert_eq!(ptx_reg_type(ScalarType::BF16).unwrap(), ".b16");
}

#[test]
fn test_ptx_data_type_byte_sizes() {
    assert_eq!(ptx_type_bytes(ScalarType::F32).unwrap(), 4);
    assert_eq!(ptx_type_bytes(ScalarType::F16).unwrap(), 2);
    assert_eq!(ptx_type_bytes(ScalarType::BF16).unwrap(), 2);
}

#[test]
fn test_cuda_prelude_includes_runtime_headers() {
    assert!(CUDA_PRELUDE.contains("#include <cuda_runtime.h>"));
    assert!(CUDA_PRELUDE.contains("#include <cuda_fp16.h>"));
    assert!(CUDA_PRELUDE.contains("#include <cuda_bf16.h>"));
}

// ===========================================================================
// C. Compilation Pipeline (5+ tests)
// ===========================================================================

#[test]
fn test_nvcc_command_format_sm80() {
    let cmd = nvcc_command(
        Path::new("/tmp/kernel.cu"),
        Path::new("/tmp/kernel.ptx"),
        "sm_80",
    );
    assert_eq!(cmd.len(), 7);
    assert_eq!(cmd[0], "nvcc");
    assert_eq!(cmd[1], "--ptx");
    assert_eq!(cmd[2], "--gpu-architecture=sm_80");
    assert_eq!(cmd[3], "-O3");
}

#[test]
fn test_nvcc_command_format_sm90() {
    let cmd = nvcc_command(Path::new("/src/k.cu"), Path::new("/out/k.ptx"), "sm_90");
    assert!(cmd[2].contains("sm_90"), "must target sm_90");
}

#[test]
fn test_ptxas_command_format() {
    let cmd = ptxas_command(Path::new("/tmp/k.ptx"), Path::new("/tmp/k.cubin"), "sm_80");
    assert_eq!(cmd[0], "ptxas");
    assert_eq!(cmd[1], "--gpu-name=sm_80");
    assert_eq!(cmd[2], "-O3");
    assert_eq!(cmd[3], "-o");
}

#[test]
fn test_ptxas_command_different_targets() {
    for target in ["sm_70", "sm_80", "sm_90", "sm_100"] {
        let cmd = ptxas_command(Path::new("/tmp/k.ptx"), Path::new("/tmp/k.cubin"), target);
        assert!(
            cmd[1].contains(target),
            "ptxas command must contain target {target}"
        );
    }
}

#[test]
fn test_compile_without_nvcc_returns_not_found() {
    // On macOS, nvcc is not available.
    if check_nvcc() {
        return; // Cannot test absence when nvcc exists
    }
    let result =
        nn_cuda::compile_ptx::compile_cuda_to_ptx("__global__ void k() {}", "sm_80", None);
    assert!(result.is_err());
    assert!(
        matches!(result.unwrap_err(), PtxCompileError::NvccNotFound),
        "expected NvccNotFound on macOS"
    );
}

#[test]
fn test_ptx_module_fields() {
    let module = PtxModule {
        ptx_path: PathBuf::from("/tmp/test.ptx"),
        cubin_path: Some(PathBuf::from("/tmp/test.cubin")),
        sm_target: "sm_80".to_owned(),
        cache_hit: false,
    };
    assert_eq!(module.sm_target, "sm_80");
    assert!(!module.cache_hit);
    assert!(module.cubin_path.is_some());
}

#[test]
fn test_ptx_module_no_cubin() {
    let module = PtxModule {
        ptx_path: PathBuf::from("/tmp/test.ptx"),
        cubin_path: None,
        sm_target: "sm_90".to_owned(),
        cache_hit: true,
    };
    assert!(module.cubin_path.is_none());
    assert!(module.cache_hit);
}

// ===========================================================================
// D. Runtime Types (10+ tests)
// ===========================================================================

#[test]
fn test_cuda_dim3_d1_constructor() {
    let d = CudaDim3::d1(256);
    assert_eq!(d.x, 256);
    assert_eq!(d.y, 1);
    assert_eq!(d.z, 1);
    assert_eq!(d.total(), 256);
}

#[test]
fn test_cuda_dim3_d2_constructor() {
    let d = CudaDim3::d2(16, 16);
    assert_eq!(d.x, 16);
    assert_eq!(d.y, 16);
    assert_eq!(d.z, 1);
    assert_eq!(d.total(), 256);
}

#[test]
fn test_cuda_dim3_3d_constructor() {
    let d = CudaDim3::new(4, 8, 2);
    assert_eq!(d.total(), 64);
}

#[test]
fn test_cuda_dim3_large_values_no_overflow() {
    // u32::MAX * 1 * 1 fits in u64
    let d = CudaDim3::new(u32::MAX, 1, 1);
    assert_eq!(d.total(), u64::from(u32::MAX));
}

#[test]
fn test_cuda_dim3_equality() {
    let a = CudaDim3::new(1, 2, 3);
    let b = CudaDim3::new(1, 2, 3);
    let c = CudaDim3::new(3, 2, 1);
    assert_eq!(a, b);
    assert_ne!(a, c);
}

#[test]
fn test_cuda_launch_config_elementwise() {
    let cfg = CudaLaunchConfig::for_elementwise(1024, 256);
    assert_eq!(cfg.grid.x, 4);
    assert_eq!(cfg.block.x, 256);
    assert_eq!(cfg.shared_mem_bytes, 0);
}

#[test]
fn test_cuda_launch_config_elementwise_single_element() {
    let cfg = CudaLaunchConfig::for_elementwise(1, 256);
    assert_eq!(cfg.grid.x, 1);
    assert_eq!(cfg.block.x, 256);
}

#[test]
fn test_cuda_launch_config_elementwise_not_multiple() {
    let cfg = CudaLaunchConfig::for_elementwise(1000, 256);
    assert_eq!(cfg.grid.x, 4, "ceil(1000/256) = 4");
}

#[test]
fn test_cuda_launch_config_reduction_shared_mem() {
    let cfg = CudaLaunchConfig::for_reduction(32, 256);
    assert_eq!(cfg.grid.x, 32);
    assert_eq!(cfg.shared_mem_bytes, 256 * 4, "sizeof(float) * block_size");
}

#[test]
fn test_cuda_launch_config_matmul() {
    let cfg = CudaLaunchConfig::for_matmul(128, 64, 16, 16);
    assert_eq!(cfg.grid.x, 4, "ceil(64/16)");
    assert_eq!(cfg.grid.y, 8, "ceil(128/16)");
    assert_eq!(cfg.block.x, 16);
    assert_eq!(cfg.block.y, 16);
}

#[test]
fn test_cuda_launch_config_batched() {
    let cfg = CudaLaunchConfig::for_batched(4, 8, 16, 256);
    assert_eq!(cfg.grid.x, 4);
    assert_eq!(cfg.grid.y, 8);
    assert_eq!(cfg.grid.z, 16);
    assert_eq!(cfg.block.x, 256);
}

// Launch config validation tests — test dimensions and thread counts
// via CudaDim3 and CudaLaunchConfig public API (validate_launch_config
// is pub(crate), so we test the invariants it checks indirectly).

#[test]
fn test_launch_config_valid_dimensions() {
    let config = CudaLaunchConfig {
        grid: CudaDim3::d1(4),
        block: CudaDim3::d1(256),
        shared_mem_bytes: 0,
    };
    // Valid: non-zero dims, 256 threads per block <= 1024
    assert_eq!(config.block.total(), 256);
    assert!(config.block.total() <= 1024);
    assert!(config.grid.x > 0 && config.grid.y > 0 && config.grid.z > 0);
}

#[test]
fn test_launch_config_zero_block_is_invalid() {
    let config = CudaLaunchConfig {
        grid: CudaDim3::d1(4),
        block: CudaDim3::d1(0),
        shared_mem_bytes: 0,
    };
    // Zero block dimension is invalid for CUDA launch
    assert_eq!(config.block.total(), 0);
}

#[test]
fn test_launch_config_zero_grid_is_invalid() {
    let config = CudaLaunchConfig {
        grid: CudaDim3::d1(0),
        block: CudaDim3::d1(256),
        shared_mem_bytes: 0,
    };
    assert_eq!(config.grid.total(), 0);
}

#[test]
fn test_launch_config_too_many_threads() {
    // 64 * 32 = 2048 > 1024 max threads per block
    let block = CudaDim3::d2(64, 32);
    assert_eq!(block.total(), 2048);
    assert!(block.total() > 1024, "exceeds CUDA max threads per block");
}

#[test]
fn test_launch_config_max_threads_ok() {
    // 32 * 32 = 1024 exactly at the limit
    let block = CudaDim3::d2(32, 32);
    assert_eq!(block.total(), 1024);
    assert!(block.total() <= 1024, "1024 is within CUDA max");
}

#[test]
fn test_launch_config_zero_grid_y_is_invalid() {
    let grid = CudaDim3::new(4, 0, 1);
    assert_eq!(grid.y, 0);
    assert_eq!(grid.total(), 0, "zero y dim makes total zero");
}

#[test]
fn test_launch_config_zero_block_z_is_invalid() {
    let block = CudaDim3::new(256, 1, 0);
    assert_eq!(block.z, 0);
    assert_eq!(block.total(), 0, "zero z dim makes total zero");
}

// -- Launch config helper functions from ptx_emit --

#[test]
fn test_elementwise_launch_config_exact() {
    let (grid, block) = elementwise_launch_config(1024);
    assert_eq!(block, PTX_BLOCK_SIZE);
    assert_eq!(grid, 4);
}

#[test]
fn test_elementwise_launch_config_remainder() {
    let (grid, block) = elementwise_launch_config(1000);
    assert_eq!(block, 256);
    assert_eq!(grid, 4); // ceil(1000/256) = 4
}

#[test]
fn test_reduction_launch_config_values() {
    let (num_blocks, block_size) = reduction_launch_config(32, 512);
    assert_eq!(num_blocks, 32);
    assert_eq!(block_size, 256); // min(256, 512)
}

#[test]
fn test_reduction_launch_config_small_row() {
    let (num_blocks, block_size) = reduction_launch_config(10, 3);
    assert_eq!(num_blocks, 10);
    assert_eq!(block_size, 4); // next_power_of_two(3) = 4, min(256, 4) = 4
}

#[test]
fn test_matmul_launch_config_values() {
    let (grid, block) = matmul_launch_config(128, 64, 16);
    assert_eq!(grid, [4, 8]); // [ceil(64/16), ceil(128/16)]
    assert_eq!(block, [16, 16]);
}

#[test]
fn test_matmul_launch_config_not_multiple() {
    let (grid, block) = matmul_launch_config(100, 50, 16);
    assert_eq!(grid, [4, 7]); // [ceil(50/16)=4, ceil(100/16)=7]
    assert_eq!(block, [16, 16]);
}

// -- Error type coverage --

#[test]
fn test_cuda_runtime_error_not_available_display() {
    let err = CudaRuntimeError::NotAvailable;
    let msg = err.to_string();
    assert!(msg.contains("not available") || msg.contains("NVIDIA"));
}

#[test]
fn test_cuda_runtime_error_no_devices_display() {
    let err = CudaRuntimeError::NoDevices;
    let msg = err.to_string();
    assert!(msg.contains("no NVIDIA GPU") || msg.contains("devices"));
}

#[test]
fn test_cuda_runtime_error_api_error_display() {
    let err = CudaRuntimeError::ApiError {
        function: "cudaSetDevice",
        code: 101,
    };
    let msg = err.to_string();
    assert!(msg.contains("cudaSetDevice"));
    assert!(msg.contains("101"));
}

#[test]
fn test_cuda_runtime_error_out_of_memory_display() {
    let err = CudaRuntimeError::OutOfMemory {
        requested: 1_073_741_824,
    };
    let msg = err.to_string();
    assert!(msg.contains("1073741824"));
}

#[test]
fn test_cuda_runtime_error_kernel_not_found_display() {
    let err = CudaRuntimeError::KernelNotFound {
        name: "nn_cuda_kernel".to_string(),
    };
    let msg = err.to_string();
    assert!(msg.contains("nn_cuda_kernel"));
}

#[test]
fn test_cuda_runtime_error_buffer_size_mismatch_display() {
    let err = CudaRuntimeError::BufferSizeMismatch {
        expected: 4096,
        actual: 8192,
    };
    let msg = err.to_string();
    assert!(msg.contains("4096") && msg.contains("8192"));
}

#[test]
fn test_cuda_runtime_error_invalid_launch_config_display() {
    let err = CudaRuntimeError::InvalidLaunchConfig {
        reason: "threads per block exceeds max".to_string(),
    };
    let msg = err.to_string();
    assert!(msg.contains("threads per block exceeds max"));
}

#[test]
fn test_cuda_runtime_error_module_load_failed_display() {
    let err = CudaRuntimeError::ModuleLoadFailed {
        path: "/tmp/bad.ptx".to_string(),
        reason: "file not found".to_string(),
    };
    let msg = err.to_string();
    assert!(msg.contains("/tmp/bad.ptx"));
    assert!(msg.contains("file not found"));
}

#[test]
fn test_cuda_not_available_on_macos() {
    if cfg!(target_os = "macos") {
        assert!(!is_cuda_available());
    }
}

#[test]
fn test_cuda_runtime_init_not_available_on_macos() {
    if cfg!(target_os = "macos") {
        let result = CudaRuntime::init(0);
        assert!(matches!(result, Err(CudaRuntimeError::NotAvailable)));
    }
}

// ===========================================================================
// E. FFI Types (5+ tests)
// ===========================================================================

#[test]
fn test_cuda_error_code_values() {
    assert_eq!(error_code::CUDA_SUCCESS, 0);
    assert_eq!(error_code::CUDA_ERROR_INVALID_VALUE, 1);
    assert_eq!(error_code::CUDA_ERROR_OUT_OF_MEMORY, 2);
    assert_eq!(error_code::CUDA_ERROR_NOT_INITIALIZED, 3);
    assert_eq!(error_code::CUDA_ERROR_INVALID_DEVICE, 101);
    assert_eq!(error_code::CUDA_ERROR_NO_DEVICE, 100);
    assert_eq!(error_code::CUDA_ERROR_FILE_NOT_FOUND, 301);
    assert_eq!(error_code::CUDA_ERROR_NOT_FOUND, 500);
    assert_eq!(error_code::CUDA_ERROR_LAUNCH_FAILED, 719);
}

#[test]
fn test_cuda_memcpy_kind_repr_values() {
    assert_eq!(CudaMemcpyKind::HostToHost as i32, 0);
    assert_eq!(CudaMemcpyKind::HostToDevice as i32, 1);
    assert_eq!(CudaMemcpyKind::DeviceToHost as i32, 2);
    assert_eq!(CudaMemcpyKind::DeviceToDevice as i32, 3);
}

#[test]
fn test_cuda_memcpy_kind_equality() {
    assert_eq!(CudaMemcpyKind::HostToDevice, CudaMemcpyKind::HostToDevice);
    assert_ne!(CudaMemcpyKind::HostToDevice, CudaMemcpyKind::DeviceToHost);
}

#[test]
fn test_sm_target_constants_format() {
    let targets = [
        sm_target::SM_70,
        sm_target::SM_75,
        sm_target::SM_80,
        sm_target::SM_86,
        sm_target::SM_89,
        sm_target::SM_90,
        sm_target::SM_100,
    ];
    for t in targets {
        assert!(!t.is_empty(), "SM target must not be empty");
        assert!(
            t.starts_with("sm_"),
            "NVIDIA targets must start with sm_: {t}"
        );
    }
}

#[test]
fn test_sm_target_generational_ordering() {
    // Verify targets have increasing numeric suffixes (parse the number after "sm_")
    fn sm_num(s: &str) -> u32 {
        s.strip_prefix("sm_").unwrap().parse().unwrap()
    }
    assert!(sm_num(sm_target::SM_70) < sm_num(sm_target::SM_80));
    assert!(sm_num(sm_target::SM_80) < sm_num(sm_target::SM_90));
    assert!(sm_num(sm_target::SM_90) < sm_num(sm_target::SM_100));
}

#[test]
fn test_null_handle_safety() {
    // Verify null pointers are representable for FFI handle types
    use std::ffi::c_void;
    let null_ptr: *mut c_void = std::ptr::null_mut();
    assert!(null_ptr.is_null());
}

// ===========================================================================
// F. Codegen Constants and Helpers
// ===========================================================================

#[test]
fn test_ptx_version_constant() {
    assert_eq!(PTX_VERSION, "6.5");
}

#[test]
fn test_default_sm_target_constant() {
    assert_eq!(DEFAULT_SM_TARGET, "sm_80");
}

#[test]
fn test_ptx_block_size_constant() {
    assert_eq!(PTX_BLOCK_SIZE, 256);
}

#[test]
fn test_reduce_block_size_constant() {
    assert_eq!(REDUCE_BLOCK_SIZE, 256);
}

#[test]
fn test_warp_size_constant() {
    assert_eq!(WARP_SIZE, 32);
}

#[test]
fn test_ptx_accumulator_type_always_f32() {
    for dtype in [ScalarType::F32, ScalarType::F16, ScalarType::BF16] {
        assert_eq!(
            ptx_accumulator_type(dtype),
            ".f32",
            "PTX accumulator should always be .f32 for {dtype:?}"
        );
    }
}

#[test]
fn test_cuda_type_mapping() {
    assert_eq!(cuda_type(ScalarType::F32).unwrap(), "float");
    assert_eq!(cuda_type(ScalarType::F16).unwrap(), "__half");
    assert_eq!(cuda_type(ScalarType::BF16).unwrap(), "__nv_bfloat16");
}

#[test]
fn test_format_ptx_float_special_values() {
    assert_eq!(format_ptx_float(f32::INFINITY), "0x7F800000");
    assert_eq!(format_ptx_float(f32::NEG_INFINITY), "0xFF800000");
    assert_eq!(format_ptx_float(f32::NAN), "0x7FC00000");
}

#[test]
fn test_format_ptx_float_known_values() {
    // 1.0f32 = 0x3F800000 in IEEE 754
    assert_eq!(format_ptx_float(1.0), "0f3F800000");
    // 0.0f32 = 0x00000000
    assert_eq!(format_ptx_float(0.0), "0f00000000");
    // -1.0f32 = 0xBF800000
    assert_eq!(format_ptx_float(-1.0), "0fBF800000");
}

#[test]
fn test_format_ptx_float_hex_format() {
    // All non-special floats should start with "0f" prefix
    let result = format_ptx_float(3.14);
    assert!(
        result.starts_with("0f"),
        "PTX hex float should start with 0f prefix: {result}"
    );
    assert_eq!(result.len(), 10, "0f + 8 hex digits = 10 chars");
}

#[test]
fn test_safe_ptx_uint_valid_range() {
    assert_eq!(safe_ptx_uint(0).unwrap(), "0");
    assert_eq!(safe_ptx_uint(1024).unwrap(), "1024");
    assert_eq!(safe_ptx_uint(u32::MAX as usize).unwrap(), "4294967295");
}

#[test]
fn test_safe_ptx_uint_overflow() {
    let result = safe_ptx_uint(u32::MAX as usize + 1);
    assert!(result.is_err(), "values exceeding u32::MAX must fail");
}

// ===========================================================================
// G. CudaSyntax CodegenSyntax trait (additional coverage)
// ===========================================================================

#[test]
fn test_cuda_syntax_uint_keyword() {
    let s = CudaSyntax;
    assert_eq!(s.uint_keyword(), "unsigned int");
}

#[test]
fn test_cuda_syntax_type_names() {
    let s = CudaSyntax;
    assert_eq!(s.type_name(ScalarType::F32).unwrap(), "float");
    assert_eq!(s.type_name(ScalarType::F16).unwrap(), "__half");
    assert_eq!(s.type_name(ScalarType::BF16).unwrap(), "__nv_bfloat16");
}

#[test]
fn test_cuda_syntax_accum_type_always_float() {
    let s = CudaSyntax;
    for dtype in [ScalarType::F32, ScalarType::F16, ScalarType::BF16] {
        assert_eq!(s.accum_type(dtype), "float");
    }
}

#[test]
fn test_cuda_syntax_cast_expr() {
    let s = CudaSyntax;
    assert_eq!(s.cast_expr("float", "x"), "(float)x");
    assert_eq!(s.cast_expr("__half", "val"), "(__half)val");
}

#[test]
fn test_cuda_syntax_backend_name() {
    let s = CudaSyntax;
    assert_eq!(s.backend_name(), "CUDA");
}

#[test]
fn test_cuda_syntax_const_uint_decl() {
    let s = CudaSyntax;
    let decl = s.const_uint_decl("N", "1024");
    assert!(decl.contains("const unsigned int N = 1024"));
}

#[test]
fn test_cuda_syntax_for_loop_header() {
    let s = CudaSyntax;
    let header = s.for_loop_header("i", "N");
    assert!(header.contains("unsigned int i = 0"));
    assert!(header.contains("i < N"));
    assert!(header.contains("i++"));
}

#[test]
fn test_cuda_syntax_safe_uint() {
    let s = CudaSyntax;
    assert_eq!(s.safe_uint(42).unwrap(), "42");
    assert!(s.safe_uint(u32::MAX as usize + 1).is_err());
}

// ===========================================================================
// H. PtxCompileError variant coverage
// ===========================================================================

#[test]
fn test_ptx_compile_error_nvcc_not_found_display() {
    let err = PtxCompileError::NvccNotFound;
    let msg = err.to_string();
    assert!(msg.contains("nvcc") && msg.contains("not found"));
}

#[test]
fn test_ptx_compile_error_ptxas_not_found_display() {
    let err = PtxCompileError::PtxasNotFound;
    let msg = err.to_string();
    assert!(msg.contains("ptxas") && msg.contains("not found"));
}

#[test]
fn test_ptx_compile_error_compilation_failed_display() {
    let err = PtxCompileError::CompilationFailed {
        exit_code: Some(1),
        stderr: "syntax error in kernel.cu".to_string(),
    };
    let msg = err.to_string();
    assert!(msg.contains("syntax error in kernel.cu"));
    assert!(msg.contains("1"));
}

#[test]
fn test_ptx_compile_error_assembly_failed_display() {
    let err = PtxCompileError::AssemblyFailed {
        exit_code: Some(2),
        stderr: "invalid register".to_string(),
    };
    let msg = err.to_string();
    assert!(msg.contains("invalid register"));
}

// ===========================================================================
// I. PtxCodegenError variant coverage
// ===========================================================================

#[test]
fn test_ptx_codegen_error_unsupported_type_display() {
    let err = PtxCodegenError::UnsupportedType {
        type_desc: "non-float ScalarType",
    };
    let msg = err.to_string();
    assert!(msg.contains("non-float ScalarType"));
}

#[test]
fn test_ptx_codegen_error_value_exceeds_u32_display() {
    let err = PtxCodegenError::ValueExceedsU32 {
        value: u32::MAX as usize + 1,
        max: u32::MAX,
    };
    let msg = err.to_string();
    assert!(msg.contains("u32::MAX") || msg.contains("4294967295"));
}

#[test]
fn test_ptx_codegen_error_shape_overflow_display() {
    let err = PtxCodegenError::ShapeProductOverflow {
        shape: vec![usize::MAX, 2],
    };
    let msg = err.to_string();
    assert!(msg.contains("shape product overflow"));
}

#[test]
fn test_ptx_codegen_error_unsupported_step_display() {
    let err = PtxCodegenError::UnsupportedStep {
        step_name: "WmmaMatMul",
    };
    let msg = err.to_string();
    assert!(msg.contains("WmmaMatMul"));
}

#[test]
fn test_ptx_codegen_error_invalid_parameter_display() {
    let err = PtxCodegenError::InvalidParameter("tile_size must be 1..=32".to_string());
    let msg = err.to_string();
    assert!(msg.contains("tile_size must be 1..=32"));
}

#[test]
fn test_ptx_codegen_error_axis_out_of_bounds_display() {
    let err = PtxCodegenError::AxisOutOfBounds { axis: 5, rank: 3 };
    let msg = err.to_string();
    assert!(msg.contains("axis 5") && msg.contains("rank 3"));
}

// ===========================================================================
// J. Platform detection
// ===========================================================================

#[test]
fn test_check_nvcc_returns_bool() {
    // Just verify it returns without panicking
    let _available = check_nvcc();
}

#[test]
fn test_check_ptxas_returns_bool() {
    let _available = check_ptxas();
}

#[test]
fn test_cuda_device_count_not_available_on_macos() {
    if cfg!(target_os = "macos") {
        let result = nn_cuda::cuda_runtime::cuda_device_count();
        assert!(matches!(result, Err(CudaRuntimeError::NotAvailable)));
    }
}

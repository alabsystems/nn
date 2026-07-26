// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Additional unit/integration tests for the nn-cuda crate.
//!
//! Covers: error types, codegen parameter validation, launch config edge cases,
//! HipSyntax trait, HipCache, Dim3 arithmetic, format_float, and target constants.
//!
//! Part of #3813.

use nn_cuda::codegen_hip::{format_float, hip_accumulator_type, hip_type, safe_hip_uint};
use nn_cuda::codegen_hip_moe::{
    emit_grouped_gemm_kernel, emit_moe_permute_kernel, emit_moe_unpermute_kernel,
    grouped_gemm_launch_config, moe_permute_launch_config, moe_swiglu_launch_config,
};
use nn_cuda::codegen_hip_tensor_emit_gemm::should_use_rocwmma;
use nn_cuda::compile_hip::target;
use nn_cuda::{
    check_hipcc, emit_gemm_hip, hipcc_command, HipCache, HipCodegenError, HipDispatchError,
};
use nn_cuda::{Dim3, LaunchConfig};
use nn_dsl::ScalarType;
use std::path::Path;

// ---------------------------------------------------------------------------
// 1. Error type display and variant coverage
// ---------------------------------------------------------------------------

#[test]
fn test_hip_codegen_error_shape_overflow_display() {
    let err = HipCodegenError::ShapeProductOverflow {
        shape: vec![usize::MAX, 2],
    };
    let msg = err.to_string();
    assert!(
        msg.contains("shape product overflow"),
        "error message should mention shape product overflow: {msg}"
    );
}

#[test]
fn test_hip_codegen_error_unsupported_step_display() {
    let err = HipCodegenError::UnsupportedStep {
        step_name: "FakeStep",
    };
    let msg = err.to_string();
    assert!(
        msg.contains("FakeStep"),
        "error message should contain step name: {msg}"
    );
}

#[test]
fn test_hip_codegen_error_axis_out_of_bounds_display() {
    let err = HipCodegenError::AxisOutOfBounds { axis: 5, rank: 3 };
    let msg = err.to_string();
    assert!(
        msg.contains("axis 5") && msg.contains("rank 3"),
        "error message should contain axis and rank: {msg}"
    );
}

#[test]
fn test_hip_codegen_error_empty_stack_display() {
    let err = HipCodegenError::EmptyStack;
    let msg = err.to_string();
    assert!(
        msg.contains("n_inputs=0"),
        "EmptyStack should mention n_inputs=0: {msg}"
    );
}

#[test]
fn test_hip_codegen_error_stride_exceeds_u32_display() {
    let err = HipCodegenError::StrideExceedsU32 {
        value: u32::MAX as usize + 1,
        max: u32::MAX,
    };
    let msg = err.to_string();
    assert!(
        msg.contains("u32::MAX") || msg.contains("4294967295"),
        "StrideExceedsU32 should mention max: {msg}"
    );
}

#[test]
fn test_hip_codegen_error_invalid_parameter_display() {
    let err = HipCodegenError::InvalidParameter("bad param".to_string());
    let msg = err.to_string();
    assert!(
        msg.contains("bad param"),
        "InvalidParameter should contain message: {msg}"
    );
}

// ---------------------------------------------------------------------------
// 2. HipDispatchError variant coverage
// ---------------------------------------------------------------------------

#[test]
fn test_hip_dispatch_error_from_codegen() {
    let codegen_err = HipCodegenError::EmptyStack;
    let dispatch_err: HipDispatchError = codegen_err.into();
    let msg = dispatch_err.to_string();
    assert!(
        msg.contains("codegen") || msg.contains("n_inputs=0"),
        "HipDispatchError should wrap codegen error: {msg}"
    );
}

// ---------------------------------------------------------------------------
// 3. hip_type and hip_accumulator_type edge cases
// ---------------------------------------------------------------------------

#[test]
fn test_hip_type_bf16_maps_to_hip_bfloat16() {
    assert_eq!(hip_type(ScalarType::BF16).unwrap(), "hip_bfloat16");
}

#[test]
fn test_hip_accumulator_type_always_float() {
    // All scalar types should accumulate in f32 for precision.
    for dtype in [ScalarType::F32, ScalarType::F16, ScalarType::BF16] {
        assert_eq!(
            hip_accumulator_type(dtype),
            "float",
            "accumulator type for {dtype:?} should be float"
        );
    }
}

// ---------------------------------------------------------------------------
// 4. safe_hip_uint boundary conditions
// ---------------------------------------------------------------------------

#[test]
fn test_safe_hip_uint_zero() {
    assert_eq!(safe_hip_uint(0).unwrap(), "0");
}

#[test]
fn test_safe_hip_uint_u32_max() {
    assert_eq!(
        safe_hip_uint(u32::MAX as usize).unwrap(),
        u32::MAX.to_string()
    );
}

#[test]
fn test_safe_hip_uint_one_past_u32_max_fails() {
    let result = safe_hip_uint(u32::MAX as usize + 1);
    assert!(result.is_err(), "u32::MAX+1 should fail safe_hip_uint");
}

// ---------------------------------------------------------------------------
// 5. format_float special values
// ---------------------------------------------------------------------------

#[test]
fn test_format_float_nan_produces_nanf() {
    let result = format_float(f32::NAN);
    assert_eq!(result, "nanf(\"\")");
}

#[test]
fn test_format_float_neg_infinity() {
    assert_eq!(format_float(f32::NEG_INFINITY), "(-HUGE_VALF)");
}

#[test]
fn test_format_float_pos_infinity() {
    assert_eq!(format_float(f32::INFINITY), "HUGE_VALF");
}

#[test]
fn test_format_float_zero() {
    let result = format_float(0.0);
    assert!(
        result.contains("0.00000000"),
        "zero should format with 8 decimal places: {result}"
    );
}

#[test]
fn test_format_float_negative_value() {
    let result = format_float(-3.14);
    assert!(
        result.starts_with('-'),
        "negative value should start with -: {result}"
    );
    assert!(result.contains("3.14"), "should contain value: {result}");
}

// ---------------------------------------------------------------------------
// 6. Dim3 edge cases
// ---------------------------------------------------------------------------

#[test]
fn test_dim3_total_large_values() {
    // Dim3::total() casts to u64 before multiplying, verify large but non-overflowing.
    // u32::MAX * 1 * 1 = 4294967295, fits in u64.
    let d = Dim3::new(u32::MAX, 1, 1);
    assert_eq!(d.total(), u64::from(u32::MAX));

    // Two large dims: 65536 * 65536 * 1 = 4294967296, fits in u64.
    let d2 = Dim3::new(65536, 65536, 1);
    assert_eq!(d2.total(), 65536u64 * 65536);
}

#[test]
fn test_dim3_d1_y_z_are_one() {
    let d = Dim3::d1(42);
    assert_eq!(d.x, 42);
    assert_eq!(d.y, 1);
    assert_eq!(d.z, 1);
    assert_eq!(d.total(), 42);
}

#[test]
fn test_dim3_d2_z_is_one() {
    let d = Dim3::d2(10, 20);
    assert_eq!(d.x, 10);
    assert_eq!(d.y, 20);
    assert_eq!(d.z, 1);
    assert_eq!(d.total(), 200);
}

#[test]
fn test_dim3_equality() {
    let a = Dim3::new(1, 2, 3);
    let b = Dim3::new(1, 2, 3);
    let c = Dim3::new(3, 2, 1);
    assert_eq!(a, b);
    assert_ne!(a, c);
}

// ---------------------------------------------------------------------------
// 7. LaunchConfig factory methods
// ---------------------------------------------------------------------------

#[test]
fn test_launch_config_elementwise_single_element() {
    let cfg = LaunchConfig::for_elementwise(1, 256);
    assert_eq!(cfg.grid.x, 1, "single element should need 1 block");
    assert_eq!(cfg.block.x, 256);
    assert_eq!(cfg.shared_mem_bytes, 0);
}

#[test]
fn test_launch_config_elementwise_exact_multiple() {
    let cfg = LaunchConfig::for_elementwise(512, 256);
    assert_eq!(cfg.grid.x, 2, "512/256 = 2 blocks exactly");
}

#[test]
fn test_launch_config_reduction_shared_mem() {
    let cfg = LaunchConfig::for_reduction(100, 128);
    assert_eq!(
        cfg.shared_mem_bytes,
        128 * 4,
        "shared mem should be block_size * sizeof(float)"
    );
}

#[test]
fn test_launch_config_matmul_single_tile() {
    let cfg = LaunchConfig::for_matmul(8, 8, 16, 16);
    assert_eq!(cfg.grid.x, 1, "8/16 rounds up to 1");
    assert_eq!(cfg.grid.y, 1, "8/16 rounds up to 1");
}

#[test]
fn test_launch_config_rocwmma_single_batch() {
    let cfg = LaunchConfig::for_rocwmma(32, 32, 1);
    assert_eq!(cfg.grid.x, 1, "32/32 = 1");
    assert_eq!(cfg.grid.y, 1, "32/32 = 1");
    assert_eq!(cfg.grid.z, 1, "batch_count = 1");
    assert_eq!(cfg.block.x, 256);
}

// ---------------------------------------------------------------------------
// 8. HipCache operations
// ---------------------------------------------------------------------------

#[test]
fn test_hip_cache_empty_source_hash() {
    let h1 = HipCache::content_hash("", "gfx90a");
    let h2 = HipCache::content_hash("x", "gfx90a");
    assert_ne!(
        h1, h2,
        "empty and non-empty source should produce different hashes"
    );
}

#[test]
fn test_hip_cache_new_creates_directory() {
    let tmp = tempfile::tempdir().unwrap();
    let cache_dir = tmp.path().join("hip_cache_test");
    assert!(!cache_dir.exists());
    let _cache = HipCache::new(&cache_dir).unwrap();
    assert!(cache_dir.exists(), "HipCache::new should create directory");
}

#[test]
fn test_hip_cache_lookup_returns_none_for_unknown() {
    let tmp = tempfile::tempdir().unwrap();
    let cache = HipCache::new(tmp.path()).unwrap();
    assert!(cache.lookup("nonexistent_source", "gfx90a").is_none());
}

#[test]
fn test_hip_cache_register_and_lookup_same_source_different_arch() {
    let tmp = tempfile::tempdir().unwrap();
    let cache = HipCache::new(tmp.path()).unwrap();

    let fake = tmp.path().join("fake.hsaco");
    std::fs::write(&fake, b"DATA").unwrap();
    cache.register("source_code", "gfx90a", &fake);

    assert!(cache.lookup("source_code", "gfx90a").is_some());
    assert!(
        cache.lookup("source_code", "gfx1100").is_none(),
        "different arch should not match"
    );
    assert!(
        cache.lookup("source_code", "gfx942").is_none(),
        "different arch should not match"
    );
}

// ---------------------------------------------------------------------------
// 9. Target architecture constants
// ---------------------------------------------------------------------------

#[test]
fn test_target_arch_strings_nonempty() {
    let targets = [
        target::GFX90A,
        target::GFX942,
        target::GFX950,
        target::GFX1100,
        target::GFX1102,
    ];
    for t in targets {
        assert!(!t.is_empty(), "target arch string should not be empty");
        assert!(
            t.starts_with("gfx"),
            "AMD GPU targets start with 'gfx': {t}"
        );
    }
}

// ---------------------------------------------------------------------------
// 10. hipcc_command generation
// ---------------------------------------------------------------------------

#[test]
fn test_hipcc_command_different_arches() {
    for arch in ["gfx90a", "gfx942", "gfx1100"] {
        let cmd = hipcc_command(Path::new("/tmp/k.hip.cpp"), Path::new("/tmp/k.hsaco"), arch);
        assert_eq!(cmd.len(), 7);
        assert!(
            cmd[2].contains(arch),
            "command should contain arch {arch}: {:?}",
            cmd[2]
        );
    }
}

// ---------------------------------------------------------------------------
// 11. should_use_rocwmma routing
// ---------------------------------------------------------------------------

#[test]
fn test_should_use_rocwmma_aligned_large() {
    // M, K, N all multiples of 16, M*N >= 16384, K >= 128
    assert!(
        should_use_rocwmma(128, 128, 128),
        "128x128x128 should use rocWMMA"
    );
}

#[test]
fn test_should_use_rocwmma_small_k_rejects() {
    // K=32 < 128 threshold
    assert!(
        !should_use_rocwmma(128, 32, 128),
        "K=32 should not use rocWMMA"
    );
}

#[test]
fn test_should_use_rocwmma_unaligned_rejects() {
    // M=100 is not a multiple of 16
    assert!(
        !should_use_rocwmma(100, 128, 128),
        "M=100 (not %16) should not use rocWMMA"
    );
}

// ---------------------------------------------------------------------------
// 12. MoE kernel parameter validation
// ---------------------------------------------------------------------------

#[test]
fn test_moe_grouped_gemm_unaligned_in_dim_error() {
    let result = emit_grouped_gemm_kernel("bad", 8, 100, 2048, 8192);
    assert!(result.is_err(), "in_dim=100 should fail alignment check");
}

#[test]
fn test_moe_grouped_gemm_unaligned_out_dim_error() {
    let result = emit_grouped_gemm_kernel("bad", 8, 2048, 100, 8192);
    assert!(result.is_err(), "out_dim=100 should fail alignment check");
}

#[test]
fn test_moe_permute_kernel_generation() {
    let src = emit_moe_permute_kernel("perm_test", 2048).unwrap();
    assert!(src.contains("D_HIDDEN = 2048"));
    assert!(src.contains("extern \"C\" __global__ void perm_test("));
}

#[test]
fn test_moe_unpermute_kernel_generation() {
    let src = emit_moe_unpermute_kernel("unperm_test", 4096, 8).unwrap();
    assert!(src.contains("D_HIDDEN = 4096"));
    assert!(src.contains("K = 8"));
    assert!(src.contains("atomicAdd"));
}

// ---------------------------------------------------------------------------
// 13. MoE launch config edge cases
// ---------------------------------------------------------------------------

#[test]
fn test_moe_grouped_gemm_launch_config_shared_mem() {
    let cfg = grouped_gemm_launch_config(1024, 2048);
    assert!(
        cfg.shared_mem_bytes > 0,
        "grouped GEMM launch should require shared memory"
    );
}

#[test]
fn test_moe_swiglu_launch_config_token_count() {
    let cfg = moe_swiglu_launch_config(512, 256);
    assert_eq!(cfg.grid.x, 512, "grid.x should be total_tokens");
    assert_eq!(cfg.grid.y, 1, "ceil(256/256) = 1");
}

#[test]
fn test_moe_permute_launch_config_small() {
    let cfg = moe_permute_launch_config(100);
    assert_eq!(cfg.grid.x, 1, "ceil(100/256) = 1");
    assert_eq!(cfg.block.x, 256);
}

// ---------------------------------------------------------------------------
// 14. emit_gemm_hip convenience function
// ---------------------------------------------------------------------------

#[test]
fn test_emit_gemm_hip_bf16_uses_hip_bfloat16() {
    let src = emit_gemm_hip("gemm_bf16", ScalarType::BF16, 32, 32, 32).unwrap();
    assert!(
        src.contains("hip_bfloat16"),
        "BF16 GEMM should use hip_bfloat16 type"
    );
    assert!(
        src.contains("#include <hip/hip_bfloat16.h>"),
        "BF16 GEMM should include bfloat16 header"
    );
}

#[test]
fn test_emit_gemm_hip_balanced_braces() {
    for dtype in [ScalarType::F32, ScalarType::F16, ScalarType::BF16] {
        let src = emit_gemm_hip("brace_test", dtype, 64, 64, 64).unwrap();
        let opens = src.matches('{').count();
        let closes = src.matches('}').count();
        assert_eq!(
            opens, closes,
            "unbalanced braces for {dtype:?}: opens={opens}, closes={closes}"
        );
    }
}

// ---------------------------------------------------------------------------
// 15. check_hipcc on macOS returns false (platform test)
// ---------------------------------------------------------------------------

#[test]
fn test_check_hipcc_on_macos_returns_false() {
    if cfg!(target_os = "macos") {
        assert!(!check_hipcc(), "hipcc should not be available on macOS");
    }
}

// ---------------------------------------------------------------------------
// 16. HipRuntimeError variant coverage
// ---------------------------------------------------------------------------

#[test]
fn test_hip_runtime_error_not_available_display() {
    let err = nn_cuda::HipRuntimeError::NotAvailable;
    let msg = err.to_string();
    assert!(
        msg.contains("not available") || msg.contains("ROCm"),
        "NotAvailable error should mention ROCm or availability: {msg}"
    );
}

#[test]
fn test_hip_runtime_error_no_devices_display() {
    let err = nn_cuda::HipRuntimeError::NoDevices;
    let msg = err.to_string();
    assert!(
        msg.contains("no AMD GPU") || msg.contains("devices"),
        "NoDevices error should mention devices: {msg}"
    );
}

#[test]
fn test_hip_runtime_error_out_of_memory_display() {
    let err = nn_cuda::HipRuntimeError::OutOfMemory {
        requested: 1_073_741_824,
    };
    let msg = err.to_string();
    assert!(
        msg.contains("1073741824"),
        "OutOfMemory should display requested size: {msg}"
    );
}

#[test]
fn test_hip_runtime_error_kernel_not_found_display() {
    let err = nn_cuda::HipRuntimeError::KernelNotFound {
        name: "nn_kernel".to_string(),
    };
    let msg = err.to_string();
    assert!(
        msg.contains("nn_kernel"),
        "KernelNotFound should contain kernel name: {msg}"
    );
}

#[test]
fn test_hip_runtime_error_buffer_size_mismatch_display() {
    let err = nn_cuda::HipRuntimeError::BufferSizeMismatch {
        expected: 4096,
        actual: 8192,
    };
    let msg = err.to_string();
    assert!(
        msg.contains("4096") && msg.contains("8192"),
        "BufferSizeMismatch should mention both sizes: {msg}"
    );
}

#[test]
fn test_hip_runtime_error_invalid_launch_config_display() {
    let err = nn_cuda::HipRuntimeError::InvalidLaunchConfig {
        reason: "block too large".to_string(),
    };
    let msg = err.to_string();
    assert!(
        msg.contains("block too large"),
        "InvalidLaunchConfig should contain reason: {msg}"
    );
}

// ---------------------------------------------------------------------------
// 17. HipCompileError variant coverage
// ---------------------------------------------------------------------------

#[test]
fn test_hip_compile_error_hipcc_not_found_display() {
    let err = nn_cuda::HipCompileError::HipccNotFound;
    let msg = err.to_string();
    assert!(
        msg.contains("hipcc") && msg.contains("not found"),
        "HipccNotFound should mention hipcc: {msg}"
    );
}

#[test]
fn test_hip_compile_error_compilation_failed_display() {
    let err = nn_cuda::HipCompileError::CompilationFailed {
        exit_code: Some(1),
        stderr: "undefined reference".to_string(),
    };
    let msg = err.to_string();
    assert!(
        msg.contains("undefined reference"),
        "CompilationFailed should contain stderr: {msg}"
    );
}

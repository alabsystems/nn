// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! CUDA configuration validation tests.
//!
//! Validates SM version detection, PTX type mapping, memory alignment
//! constraints, shared memory limits, and codegen prelude correctness.
//!
//! These tests verify configuration logic without requiring CUDA hardware.

use crate::codegen_ptx::{
    cuda_type, format_ptx_float, ptx_prelude, ptx_reg_type, ptx_type, ptx_type_bytes,
    safe_ptx_uint, DEFAULT_SM_TARGET, PTX_BLOCK_SIZE, PTX_VERSION, WARP_SIZE,
};
use crate::cuda_ffi::sm_target;
use nn_dsl::ScalarType;

// =========================================================================
// 1. PTX type mapping completeness
// =========================================================================

mod type_mapping {
    use super::*;

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
    fn test_ptx_reg_type_matches_ptx_type() {
        for dtype in [ScalarType::F32, ScalarType::F16, ScalarType::BF16] {
            let t = ptx_type(dtype).unwrap();
            let rt = ptx_reg_type(dtype).unwrap();
            assert_eq!(t, rt, "ptx_type and ptx_reg_type must agree for {dtype:?}");
        }
    }

    #[test]
    fn test_cuda_type_f32_is_float() {
        assert_eq!(cuda_type(ScalarType::F32).unwrap(), "float");
    }

    #[test]
    fn test_cuda_type_f16_is_half() {
        assert_eq!(cuda_type(ScalarType::F16).unwrap(), "__half");
    }

    #[test]
    fn test_cuda_type_bf16_is_nv_bfloat16() {
        assert_eq!(cuda_type(ScalarType::BF16).unwrap(), "__nv_bfloat16");
    }

    #[test]
    fn test_ptx_type_bytes_f32() {
        assert_eq!(ptx_type_bytes(ScalarType::F32).unwrap(), 4);
    }

    #[test]
    fn test_ptx_type_bytes_f16() {
        assert_eq!(ptx_type_bytes(ScalarType::F16).unwrap(), 2);
    }

    #[test]
    fn test_ptx_type_bytes_bf16() {
        assert_eq!(ptx_type_bytes(ScalarType::BF16).unwrap(), 2);
    }

    #[test]
    fn test_byte_size_consistency() {
        // F32 must be exactly 2x the byte size of F16/BF16
        let f32_bytes = ptx_type_bytes(ScalarType::F32).unwrap();
        let f16_bytes = ptx_type_bytes(ScalarType::F16).unwrap();
        let bf16_bytes = ptx_type_bytes(ScalarType::BF16).unwrap();
        assert_eq!(f32_bytes, 2 * f16_bytes);
        assert_eq!(f16_bytes, bf16_bytes);
    }
}

// =========================================================================
// 2. PTX float formatting (IEEE 754 correctness)
// =========================================================================

mod float_format {
    use super::*;

    #[test]
    fn test_positive_infinity() {
        assert_eq!(format_ptx_float(f32::INFINITY), "0x7F800000");
    }

    #[test]
    fn test_negative_infinity() {
        assert_eq!(format_ptx_float(f32::NEG_INFINITY), "0xFF800000");
    }

    #[test]
    fn test_nan() {
        let s = format_ptx_float(f32::NAN);
        assert_eq!(s, "0x7FC00000"); // quiet NaN
    }

    #[test]
    fn test_zero() {
        assert_eq!(format_ptx_float(0.0), "0f00000000");
    }

    #[test]
    fn test_one() {
        assert_eq!(format_ptx_float(1.0), "0f3F800000");
    }

    #[test]
    fn test_negative_one() {
        assert_eq!(format_ptx_float(-1.0), "0fBF800000");
    }

    #[test]
    fn test_small_epsilon() {
        // f32::EPSILON = 2^-23 = 0x34000000
        let s = format_ptx_float(f32::EPSILON);
        assert!(s.starts_with("0f"), "float constants start with 0f: {s}");
    }

    #[test]
    fn test_roundtrip_special_values() {
        // Verify that the hex representation roundtrips correctly
        let test_vals = [
            0.0f32,
            1.0,
            -1.0,
            0.5,
            3.14,
            -273.15,
            f32::MAX,
            f32::MIN_POSITIVE,
        ];
        for val in test_vals {
            let hex = format_ptx_float(val);
            assert!(
                hex.starts_with("0f"),
                "val={val}: expected 0f prefix, got {hex}"
            );
            let bits_str = &hex[2..];
            let bits = u32::from_str_radix(bits_str, 16).unwrap();
            let recovered = f32::from_bits(bits);
            assert_eq!(recovered, val, "roundtrip failed for {val}");
        }
    }
}

// =========================================================================
// 3. PTX prelude correctness
// =========================================================================

mod prelude_tests {
    use super::*;

    #[test]
    fn test_prelude_contains_version() {
        let prelude = ptx_prelude("sm_80");
        assert!(prelude.contains(&format!(".version {PTX_VERSION}")));
    }

    #[test]
    fn test_prelude_contains_target() {
        let prelude = ptx_prelude("sm_80");
        assert!(prelude.contains(".target sm_80"));
    }

    #[test]
    fn test_prelude_contains_address_size() {
        let prelude = ptx_prelude("sm_80");
        assert!(prelude.contains(".address_size 64"));
    }

    #[test]
    fn test_prelude_custom_sm_target() {
        let prelude = ptx_prelude("sm_90");
        assert!(prelude.contains(".target sm_90"));
        assert!(!prelude.contains(".target sm_80"));
    }

    #[test]
    fn test_prelude_for_all_sm_targets() {
        let targets = [
            sm_target::SM_70,
            sm_target::SM_75,
            sm_target::SM_80,
            sm_target::SM_86,
            sm_target::SM_89,
            sm_target::SM_90,
            sm_target::SM_100,
        ];
        for target in targets {
            let prelude = ptx_prelude(target);
            assert!(
                prelude.contains(&format!(".target {target}")),
                "prelude must contain .target {target}"
            );
        }
    }
}

// =========================================================================
// 4. safe_ptx_uint boundary validation
// =========================================================================

mod uint_validation {
    use super::*;

    #[test]
    fn test_zero() {
        assert_eq!(safe_ptx_uint(0).unwrap(), "0");
    }

    #[test]
    fn test_one() {
        assert_eq!(safe_ptx_uint(1).unwrap(), "1");
    }

    #[test]
    fn test_common_block_sizes() {
        assert_eq!(safe_ptx_uint(128).unwrap(), "128");
        assert_eq!(safe_ptx_uint(256).unwrap(), "256");
        assert_eq!(safe_ptx_uint(512).unwrap(), "512");
        assert_eq!(safe_ptx_uint(1024).unwrap(), "1024");
    }

    #[test]
    fn test_u32_max() {
        assert_eq!(safe_ptx_uint(u32::MAX as usize).unwrap(), "4294967295");
    }

    #[test]
    fn test_u32_max_plus_one_fails() {
        let result = safe_ptx_uint(u32::MAX as usize + 1);
        assert!(result.is_err());
    }

    #[test]
    fn test_usize_max_fails() {
        let result = safe_ptx_uint(usize::MAX);
        assert!(result.is_err());
    }

    #[test]
    fn test_large_but_valid_value() {
        // 2^31 - common in ML (buffer sizes)
        let val = 1usize << 31;
        assert!(safe_ptx_uint(val).is_ok());
    }
}

// =========================================================================
// 5. GPU hardware constants
// =========================================================================

mod hardware_constants {
    use super::*;

    #[test]
    fn test_warp_size_is_32() {
        // NVIDIA warp size has been 32 since the first CUDA GPU (G80, 2006).
        assert_eq!(WARP_SIZE, 32);
    }

    #[test]
    fn test_block_size_is_multiple_of_warp() {
        assert_eq!(PTX_BLOCK_SIZE % WARP_SIZE, 0);
    }

    #[test]
    fn test_block_size_le_1024() {
        assert!(PTX_BLOCK_SIZE <= 1024);
    }

    #[test]
    fn test_ptx_version_is_6_5() {
        assert_eq!(PTX_VERSION, "6.5");
    }

    #[test]
    fn test_default_sm_target_is_ampere() {
        // Default should target Ampere (sm_80) for wide compatibility
        assert_eq!(DEFAULT_SM_TARGET, "sm_80");
    }
}

// =========================================================================
// 6. Memory alignment for kernel dispatch
// =========================================================================

mod memory_alignment {
    use super::*;

    /// Verify that buffer sizes for common shapes are correctly aligned.
    #[test]
    fn test_f32_buffer_alignment() {
        let element_size = ptx_type_bytes(ScalarType::F32).unwrap();
        // f32 elements are naturally 4-byte aligned
        assert_eq!(element_size % 4, 0);
    }

    #[test]
    fn test_f16_buffer_alignment() {
        let element_size = ptx_type_bytes(ScalarType::F16).unwrap();
        // f16 elements are naturally 2-byte aligned
        assert_eq!(element_size % 2, 0);
    }

    #[test]
    fn test_shared_memory_for_reduction_is_aligned() {
        // Shared memory for reductions uses block_size * sizeof(float)
        let block_sizes = [128u32, 256, 512, 1024];
        for bs in block_sizes {
            let shared_bytes = bs * 4; // sizeof(float)
                                       // Must be 4-byte aligned (naturally is, since sizeof(float) = 4)
            assert_eq!(
                shared_bytes % 4,
                0,
                "shared memory for block_size={bs} must be 4-byte aligned"
            );
            // Should be <= 48KB (default shared memory limit)
            assert!(
                shared_bytes <= 48 * 1024,
                "shared memory {shared_bytes} exceeds 48KB limit for block_size={bs}"
            );
        }
    }

    #[test]
    fn test_tile_matmul_shared_memory_bounds() {
        // Tiled matmul uses 2 * tile_size^2 * sizeof(float) shared memory
        let tile_sizes = [8, 16, 32];
        for ts in tile_sizes {
            let shared_bytes: u32 = 2 * ts * ts * 4;
            assert!(
                shared_bytes <= 48 * 1024,
                "tiled matmul shared memory {shared_bytes} exceeds 48KB for tile_size={ts}"
            );
        }
    }
}

// =========================================================================
// 7. Compile pipeline command validation
// =========================================================================

mod compile_pipeline {
    use crate::compile_ptx::{nvcc_command, ptxas_command};
    use std::path::Path;

    #[test]
    fn test_nvcc_command_structure() {
        let cmd = nvcc_command(
            Path::new("/tmp/kernel.cu"),
            Path::new("/tmp/kernel.ptx"),
            "sm_80",
        );
        assert_eq!(cmd.len(), 7);
        assert_eq!(cmd[0], "nvcc");
        assert_eq!(cmd[1], "--ptx");
        assert!(cmd[2].contains("sm_80"));
        assert_eq!(cmd[3], "-O3");
    }

    #[test]
    fn test_ptxas_command_structure() {
        let cmd = ptxas_command(
            Path::new("/tmp/kernel.ptx"),
            Path::new("/tmp/kernel.cubin"),
            "sm_90",
        );
        assert_eq!(cmd.len(), 6);
        assert_eq!(cmd[0], "ptxas");
        assert!(cmd[1].contains("sm_90"));
        assert_eq!(cmd[2], "-O3");
    }

    #[test]
    fn test_nvcc_command_for_all_sm_targets() {
        use crate::cuda_ffi::sm_target;
        let targets = [
            sm_target::SM_70,
            sm_target::SM_75,
            sm_target::SM_80,
            sm_target::SM_86,
            sm_target::SM_89,
            sm_target::SM_90,
            sm_target::SM_100,
        ];
        for target in targets {
            let cmd = nvcc_command(Path::new("/tmp/k.cu"), Path::new("/tmp/k.ptx"), target);
            assert!(
                cmd[2].contains(target),
                "nvcc command must include SM target {target}: {cmd:?}"
            );
        }
    }
}

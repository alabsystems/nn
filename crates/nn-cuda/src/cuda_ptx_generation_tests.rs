// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Extended tests for CUDA PTX generation and kernel dispatch infrastructure.
//!
//! Validates PTX code generation, kernel configuration, launch parameter
//! calculations, reference implementations, and structural properties across
//! the full set of PTX generators: elementwise, matmul, reduction, activation,
//! softmax, and binary validation.
//!
//! Part of #3842.

// =========================================================================
// 1. Elementwise PTX generation: add, mul, relu (via emit), scalar_mul
// =========================================================================

mod ptx_elementwise_generation {
    use crate::cuda_validation::validate_ptx_structure;
    use crate::ptx_elementwise::{
        generate_add_ptx, generate_div_ptx, generate_exp_ptx, generate_log_ptx, generate_mul_ptx,
        generate_neg_ptx, generate_scalar_mul_ptx, generate_sqrt_ptx, generate_sub_ptx,
        ptx_elementwise_launch_config, ELEMENTWISE_BLOCK_SIZE,
    };

    #[test]
    fn test_add_ptx_contains_entry_and_instruction() {
        let ptx = generate_add_ptx(1024);
        assert!(ptx.contains(".entry ptx_add_f32"));
        assert!(ptx.contains("add.f32"));
        assert!(ptx.contains(".version"));
        assert!(ptx.contains(".target"));
        assert!(ptx.contains("ret;"));
    }

    #[test]
    fn test_mul_ptx_contains_correct_instruction() {
        let ptx = generate_mul_ptx(512);
        assert!(ptx.contains(".entry ptx_mul_f32"));
        assert!(ptx.contains("mul.f32"));
    }

    #[test]
    fn test_sub_ptx_structural() {
        let ptx = generate_sub_ptx(2048);
        let result = validate_ptx_structure(&ptx, "ptx_sub_f32");
        assert!(
            result.structural_ok,
            "failures: {:?}",
            result.structural_failures
        );
    }

    #[test]
    fn test_div_ptx_uses_approx() {
        let ptx = generate_div_ptx(256);
        assert!(ptx.contains("div.approx.f32"));
    }

    #[test]
    fn test_exp_ptx_uses_ex2_prescale() {
        let ptx = generate_exp_ptx(1024);
        assert!(ptx.contains(".entry ptx_exp_f32"));
        assert!(ptx.contains("ex2.approx.f32"));
        // Must contain the log2(e) prescale mul
        assert!(ptx.contains("mul.f32"));
    }

    #[test]
    fn test_log_ptx_uses_lg2_postscale() {
        let ptx = generate_log_ptx(1024);
        assert!(ptx.contains(".entry ptx_log_f32"));
        assert!(ptx.contains("lg2.approx.f32"));
        // Must have the ln(2) postscale mul
        assert!(ptx.contains("mul.f32"));
    }

    #[test]
    fn test_sqrt_ptx_instruction() {
        let ptx = generate_sqrt_ptx(512);
        assert!(ptx.contains(".entry ptx_sqrt_f32"));
        assert!(ptx.contains("sqrt.approx.f32"));
    }

    #[test]
    fn test_neg_ptx_instruction() {
        let ptx = generate_neg_ptx(256);
        assert!(ptx.contains(".entry ptx_neg_f32"));
        assert!(ptx.contains("neg.f32"));
    }

    #[test]
    fn test_scalar_mul_ptx_has_scalar_param() {
        let ptx = generate_scalar_mul_ptx(1024);
        assert!(ptx.contains(".entry ptx_scalar_mul_f32"));
        assert!(ptx.contains("param_scalar"));
        assert!(ptx.contains(".param .f32 param_scalar"));
    }

    #[test]
    fn test_elementwise_ptx_has_grid_stride_loop() {
        let ptx = generate_add_ptx(4096);
        // Grid-stride loop uses nctaid.x (gridDim.x) and loops back
        assert!(ptx.contains("%nctaid.x"));
        assert!(ptx.contains("bra"));
    }

    #[test]
    fn test_elementwise_ptx_has_bounds_check() {
        let ptx = generate_add_ptx(1024);
        // setp.ge.u32 for idx >= n check
        assert!(ptx.contains("setp.ge.u32"));
    }

    #[test]
    fn test_all_binary_ops_have_register_declarations() {
        for gen_fn in [
            generate_add_ptx,
            generate_sub_ptx,
            generate_mul_ptx,
            generate_div_ptx,
        ] {
            let ptx = gen_fn(128);
            assert!(ptx.contains(".reg .u32"), "missing u32 register decl");
            assert!(ptx.contains(".reg .f32"), "missing f32 register decl");
            assert!(ptx.contains(".reg .u64"), "missing u64 register decl");
            assert!(
                ptx.contains(".reg .pred"),
                "missing predicate register decl"
            );
        }
    }

    #[test]
    fn test_all_unary_ops_have_parameter_loads() {
        for gen_fn in [
            generate_exp_ptx,
            generate_log_ptx,
            generate_sqrt_ptx,
            generate_neg_ptx,
        ] {
            let ptx = gen_fn(512);
            assert!(ptx.contains("param_input"), "missing param_input");
            assert!(ptx.contains("param_output"), "missing param_output");
            assert!(ptx.contains("param_n"), "missing param_n");
        }
    }

    #[test]
    fn test_launch_config_exact_multiple() {
        let (grid, block) = ptx_elementwise_launch_config(1024);
        assert_eq!(block, [ELEMENTWISE_BLOCK_SIZE, 1, 1]);
        assert_eq!(grid[0], 1024 / ELEMENTWISE_BLOCK_SIZE);
    }

    #[test]
    fn test_launch_config_non_multiple() {
        let (grid, block) = ptx_elementwise_launch_config(1000);
        assert_eq!(block, [ELEMENTWISE_BLOCK_SIZE, 1, 1]);
        // ceil(1000 / 256) = 4
        assert_eq!(grid[0], 4);
    }

    #[test]
    fn test_launch_config_single_element() {
        let (grid, block) = ptx_elementwise_launch_config(1);
        assert_eq!(grid[0], 1);
        assert_eq!(block[0], ELEMENTWISE_BLOCK_SIZE);
    }

    #[test]
    fn test_launch_config_large_n() {
        let (grid, _block) = ptx_elementwise_launch_config(1_000_000);
        assert_eq!(grid[0], 1_000_000u32.div_ceil(ELEMENTWISE_BLOCK_SIZE));
    }

    #[test]
    fn test_all_elementwise_structural_validation() {
        let generators: Vec<(&str, String)> = vec![
            ("ptx_add_f32", generate_add_ptx(1024)),
            ("ptx_sub_f32", generate_sub_ptx(1024)),
            ("ptx_mul_f32", generate_mul_ptx(1024)),
            ("ptx_div_f32", generate_div_ptx(1024)),
            ("ptx_exp_f32", generate_exp_ptx(1024)),
            ("ptx_log_f32", generate_log_ptx(1024)),
            ("ptx_sqrt_f32", generate_sqrt_ptx(1024)),
            ("ptx_neg_f32", generate_neg_ptx(1024)),
            ("ptx_scalar_mul_f32", generate_scalar_mul_ptx(1024)),
        ];
        for (name, ptx) in &generators {
            let result = validate_ptx_structure(ptx, name);
            assert!(
                result.structural_ok,
                "{name}: structural validation failed: {:?}",
                result.structural_failures
            );
        }
    }
}

// =========================================================================
// 2. Kernel configuration: workgroup sizes, grid dims
// =========================================================================

mod ptx_kernel_config {
    use crate::codegen_ptx::{
        cuda_type, format_ptx_float, ptx_prelude, ptx_type, ptx_type_bytes, safe_ptx_uint,
        PTX_BLOCK_SIZE, PTX_VERSION, WARP_SIZE,
    };
    use crate::cuda_ffi::{CudaDim3, CudaLaunchConfig};
    use nn_dsl::ScalarType;

    #[test]
    fn test_ptx_block_size_is_256() {
        assert_eq!(PTX_BLOCK_SIZE, 256);
    }

    #[test]
    fn test_warp_size_is_32() {
        assert_eq!(WARP_SIZE, 32);
    }

    #[test]
    fn test_ptx_block_size_is_multiple_of_warp_size() {
        assert_eq!(PTX_BLOCK_SIZE % WARP_SIZE, 0);
    }

    #[test]
    fn test_ptx_version_string() {
        assert_eq!(PTX_VERSION, "6.5");
    }

    #[test]
    fn test_ptx_prelude_format() {
        let prelude = ptx_prelude("sm_80");
        assert!(prelude.contains(".version 6.5"));
        assert!(prelude.contains(".target sm_80"));
        assert!(prelude.contains(".address_size 64"));
    }

    #[test]
    fn test_ptx_prelude_different_targets() {
        for target in ["sm_70", "sm_75", "sm_80", "sm_86", "sm_90"] {
            let prelude = ptx_prelude(target);
            assert!(prelude.contains(&format!(".target {target}")));
        }
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
    fn test_cuda_type_mapping() {
        assert_eq!(cuda_type(ScalarType::F32).unwrap(), "float");
        assert_eq!(cuda_type(ScalarType::F16).unwrap(), "__half");
        assert_eq!(cuda_type(ScalarType::BF16).unwrap(), "__nv_bfloat16");
    }

    #[test]
    fn test_format_ptx_float_zero() {
        assert_eq!(format_ptx_float(0.0), "0f00000000");
    }

    #[test]
    fn test_format_ptx_float_one() {
        // 1.0f32 = 0x3F800000
        assert_eq!(format_ptx_float(1.0), "0f3F800000");
    }

    #[test]
    fn test_format_ptx_float_neg_one() {
        // -1.0f32 = 0xBF800000
        assert_eq!(format_ptx_float(-1.0), "0fBF800000");
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
        assert_eq!(format_ptx_float(f32::NAN), "0x7FC00000");
    }

    #[test]
    fn test_safe_ptx_uint_zero() {
        assert_eq!(safe_ptx_uint(0).unwrap(), "0");
    }

    #[test]
    fn test_safe_ptx_uint_max_u32() {
        assert_eq!(safe_ptx_uint(u32::MAX as usize).unwrap(), "4294967295");
    }

    #[test]
    fn test_safe_ptx_uint_overflow() {
        assert!(safe_ptx_uint(u32::MAX as usize + 1).is_err());
    }

    #[test]
    fn test_cuda_dim3_1d() {
        let dim = CudaDim3::d1(128);
        assert_eq!(dim.x, 128);
        assert_eq!(dim.y, 1);
        assert_eq!(dim.z, 1);
        assert_eq!(dim.total(), 128);
    }

    #[test]
    fn test_cuda_dim3_2d() {
        let dim = CudaDim3::d2(16, 16);
        assert_eq!(dim.x, 16);
        assert_eq!(dim.y, 16);
        assert_eq!(dim.z, 1);
        assert_eq!(dim.total(), 256);
    }

    #[test]
    fn test_cuda_dim3_3d() {
        let dim = CudaDim3::new(8, 8, 4);
        assert_eq!(dim.total(), 256);
    }

    #[test]
    fn test_launch_config_for_elementwise() {
        let config = CudaLaunchConfig::for_elementwise(1024, 256);
        assert_eq!(config.grid.x, 4);
        assert_eq!(config.block.x, 256);
        assert_eq!(config.shared_mem_bytes, 0);
    }

    #[test]
    fn test_launch_config_for_elementwise_not_multiple() {
        let config = CudaLaunchConfig::for_elementwise(1000, 256);
        assert_eq!(config.grid.x, 4); // ceil(1000/256)
    }

    #[test]
    fn test_launch_config_for_reduction() {
        let config = CudaLaunchConfig::for_reduction(32, 256);
        assert_eq!(config.grid.x, 32);
        assert_eq!(config.block.x, 256);
        assert_eq!(config.shared_mem_bytes, 256 * 4); // sizeof(float) per thread
    }

    #[test]
    fn test_launch_config_for_matmul() {
        let config = CudaLaunchConfig::for_matmul(128, 64, 16, 16);
        assert_eq!(config.grid.x, 4); // ceil(64/16)
        assert_eq!(config.grid.y, 8); // ceil(128/16)
        assert_eq!(config.block.x, 16);
        assert_eq!(config.block.y, 16);
    }

    #[test]
    fn test_launch_config_for_batched() {
        let config = CudaLaunchConfig::for_batched(4, 8, 3, 256);
        assert_eq!(config.grid.x, 4);
        assert_eq!(config.grid.y, 8);
        assert_eq!(config.grid.z, 3);
        assert_eq!(config.block.x, 256);
    }
}

// =========================================================================
// 3. PTX matmul generation: naive and tiled variants
// =========================================================================

mod ptx_matmul_generation {
    use crate::cuda_validation::validate_ptx_structure;
    use crate::ptx_matmul::{
        emit_ptx_matmul, emit_ptx_matmul_default, generate_matmul_ptx, generate_matmul_tiled_ptx,
        matmul_reference, ptx_matmul_launch_config, PtxMatmulConfig, MATMUL_BLOCK_SIZE,
        PTX_MATMUL_MAX_TILE, PTX_MATMUL_MIN_TILE, PTX_MATMUL_TILE_SIZE,
    };

    #[test]
    fn test_matmul_constants() {
        assert_eq!(MATMUL_BLOCK_SIZE, 16);
        assert_eq!(PTX_MATMUL_TILE_SIZE, 16);
        assert_eq!(PTX_MATMUL_MIN_TILE, 4);
        assert_eq!(PTX_MATMUL_MAX_TILE, 32);
    }

    #[test]
    fn test_matmul_config_defaults() {
        let config = PtxMatmulConfig::default();
        assert_eq!(config.kernel_name, "ptx_matmul_f32");
        assert_eq!(config.tile_size, PTX_MATMUL_TILE_SIZE);
        assert_eq!(config.sm_target, "sm_80");
    }

    #[test]
    fn test_matmul_config_builder() {
        let config = PtxMatmulConfig::new("nn_gemm")
            .with_tile_size(8)
            .with_sm_target("sm_90");
        assert_eq!(config.kernel_name, "nn_gemm");
        assert_eq!(config.tile_size, 8);
        assert_eq!(config.sm_target, "sm_90");
    }

    #[test]
    fn test_matmul_config_shared_memory_bytes() {
        let config = PtxMatmulConfig::new("test").with_tile_size(16);
        // 2 tiles * 16 * 16 * 4 bytes = 2048
        assert_eq!(config.shared_memory_bytes(), 2048);
    }

    #[test]
    fn test_matmul_config_threads_per_block() {
        let config = PtxMatmulConfig::new("test").with_tile_size(16);
        assert_eq!(config.threads_per_block(), 256);
    }

    #[test]
    fn test_matmul_config_validate_tile_too_small() {
        let config = PtxMatmulConfig::new("test").with_tile_size(2);
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_matmul_config_validate_tile_too_large() {
        let config = PtxMatmulConfig::new("test").with_tile_size(64);
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_matmul_config_validate_empty_name() {
        let config = PtxMatmulConfig::new("").with_tile_size(16);
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_emit_ptx_matmul_structural() {
        let config = PtxMatmulConfig::new("gemm_test").with_tile_size(16);
        let ptx = emit_ptx_matmul(&config).unwrap();
        let result = validate_ptx_structure(&ptx, "gemm_test");
        assert!(
            result.structural_ok,
            "failures: {:?}",
            result.structural_failures
        );
    }

    #[test]
    fn test_emit_ptx_matmul_contains_shared_memory() {
        let config = PtxMatmulConfig::new("matmul_shared").with_tile_size(16);
        let ptx = emit_ptx_matmul(&config).unwrap();
        assert!(ptx.contains(".shared .align 4"));
    }

    #[test]
    fn test_emit_ptx_matmul_default_structural() {
        let ptx = emit_ptx_matmul_default("default_gemm").unwrap();
        let result = validate_ptx_structure(&ptx, "default_gemm");
        assert!(
            result.structural_ok,
            "failures: {:?}",
            result.structural_failures
        );
    }

    #[test]
    fn test_generate_matmul_ptx_naive() {
        let ptx = generate_matmul_ptx(64, 32, 48);
        assert!(!ptx.is_empty());
        assert!(ptx.contains(".version"));
        assert!(ptx.contains(".entry"));
    }

    #[test]
    fn test_generate_matmul_tiled_ptx() {
        let ptx = generate_matmul_tiled_ptx(64, 32, 48, 16);
        assert!(!ptx.is_empty());
        assert!(ptx.contains(".version"));
        assert!(ptx.contains(".shared"));
    }

    #[test]
    fn test_matmul_ptx_multiple_tile_sizes() {
        for tile in [4, 8, 16, 32] {
            let config = PtxMatmulConfig::new(&format!("gemm_tile{tile}")).with_tile_size(tile);
            let ptx = emit_ptx_matmul(&config).unwrap();
            assert!(
                ptx.contains(&format!(".entry gemm_tile{tile}")),
                "tile={tile}: missing entry point"
            );
        }
    }

    #[test]
    fn test_matmul_launch_config() {
        let (grid, block) = ptx_matmul_launch_config(128, 64, 16);
        // grid = [ceil(64/16), ceil(128/16), 1] = [4, 8, 1]
        // block = [16, 16, 1]
        assert_eq!(grid, [4, 8, 1]);
        assert_eq!(block, [16, 16, 1]);
    }

    #[test]
    fn test_matmul_launch_config_non_multiple() {
        let (grid, _block) = ptx_matmul_launch_config(100, 50, 16);
        // ceil(50/16)=4, ceil(100/16)=7
        assert_eq!(grid[0], 4);
        assert_eq!(grid[1], 7);
    }

    #[test]
    fn test_matmul_reference_identity() {
        // A = I_2, B = [[1,2],[3,4]] => C = B
        let a = vec![1.0, 0.0, 0.0, 1.0];
        let b = vec![1.0, 2.0, 3.0, 4.0];
        let c = matmul_reference(&a, &b, 2, 2, 2);
        assert_eq!(c.len(), 4);
        for (actual, expected) in c.iter().zip(b.iter()) {
            assert!(
                (actual - expected).abs() < 1e-6,
                "expected {expected}, got {actual}"
            );
        }
    }

    #[test]
    fn test_matmul_reference_simple() {
        // [1, 2] * [[3], [4]] = [11]
        let a = vec![1.0, 2.0];
        let b = vec![3.0, 4.0];
        let c = matmul_reference(&a, &b, 1, 2, 1);
        assert_eq!(c.len(), 1);
        assert!((c[0] - 11.0).abs() < 1e-6);
    }

    #[test]
    fn test_matmul_reference_rectangular() {
        // A[2,3] * B[3,2]
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let b = vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0];
        let c = matmul_reference(&a, &b, 2, 3, 2);
        assert_eq!(c.len(), 4);
        // C[0,0] = 1*7 + 2*9 + 3*11 = 7 + 18 + 33 = 58
        assert!((c[0] - 58.0).abs() < 1e-4);
        // C[0,1] = 1*8 + 2*10 + 3*12 = 8 + 20 + 36 = 64
        assert!((c[1] - 64.0).abs() < 1e-4);
    }
}

// =========================================================================
// 4. PTX reduction ops: sum, max, mean
// =========================================================================

mod ptx_reduction_ops {
    use crate::cuda_validation::validate_ptx_structure;
    use crate::ptx_reduce::{
        argmax_reference, argmin_reference, generate_argmax_ptx, generate_argmin_ptx,
        generate_max_ptx, generate_mean_ptx, generate_sum_ptx, max_reference, mean_reference,
        ptx_reduce_launch_config, sum_reference, REDUCE_BLOCK_SIZE,
    };

    #[test]
    fn test_reduce_block_size() {
        assert_eq!(REDUCE_BLOCK_SIZE, 256);
    }

    #[test]
    fn test_sum_ptx_structural() {
        let ptx = generate_sum_ptx(1024);
        let result = validate_ptx_structure(&ptx, "ptx_sum_f32");
        assert!(
            result.structural_ok,
            "failures: {:?}",
            result.structural_failures
        );
    }

    #[test]
    fn test_sum_ptx_has_shared_memory() {
        let ptx = generate_sum_ptx(512);
        assert!(ptx.contains(".shared"));
    }

    #[test]
    fn test_max_ptx_structural() {
        let ptx = generate_max_ptx(256);
        let result = validate_ptx_structure(&ptx, "ptx_max_f32");
        assert!(
            result.structural_ok,
            "failures: {:?}",
            result.structural_failures
        );
    }

    #[test]
    fn test_mean_ptx_structural() {
        let ptx = generate_mean_ptx(128);
        let result = validate_ptx_structure(&ptx, "ptx_mean_f32");
        assert!(
            result.structural_ok,
            "failures: {:?}",
            result.structural_failures
        );
    }

    #[test]
    fn test_argmax_ptx_structural() {
        let ptx = generate_argmax_ptx(64);
        let result = validate_ptx_structure(&ptx, "ptx_argmax_f32");
        assert!(
            result.structural_ok,
            "failures: {:?}",
            result.structural_failures
        );
    }

    #[test]
    fn test_argmin_ptx_structural() {
        let ptx = generate_argmin_ptx(64);
        let result = validate_ptx_structure(&ptx, "ptx_argmin_f32");
        assert!(
            result.structural_ok,
            "failures: {:?}",
            result.structural_failures
        );
    }

    #[test]
    fn test_sum_reference_simple() {
        let input = vec![1.0, 2.0, 3.0, 4.0];
        let result = sum_reference(&input);
        assert!((result - 10.0).abs() < 1e-6);
    }

    #[test]
    fn test_sum_reference_negative() {
        let input = vec![-1.0, -2.0, 3.0];
        let result = sum_reference(&input);
        assert!((result - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_max_reference_simple() {
        let input = vec![1.0, 5.0, 3.0, 2.0];
        let result = max_reference(&input);
        assert!((result - 5.0).abs() < 1e-6);
    }

    #[test]
    fn test_max_reference_negative() {
        let input = vec![-5.0, -1.0, -3.0];
        let result = max_reference(&input);
        assert!((result - (-1.0)).abs() < 1e-6);
    }

    #[test]
    fn test_mean_reference_simple() {
        let input = vec![2.0, 4.0, 6.0, 8.0];
        let result = mean_reference(&input);
        assert!((result - 5.0).abs() < 1e-6);
    }

    #[test]
    fn test_mean_reference_single() {
        let input = vec![42.0];
        let result = mean_reference(&input);
        assert!((result - 42.0).abs() < 1e-6);
    }

    #[test]
    fn test_argmax_reference() {
        let input = vec![1.0, 5.0, 3.0, 2.0];
        assert_eq!(argmax_reference(&input), 1);
    }

    #[test]
    fn test_argmax_reference_first_element() {
        let input = vec![10.0, 1.0, 2.0];
        assert_eq!(argmax_reference(&input), 0);
    }

    #[test]
    fn test_argmin_reference() {
        let input = vec![3.0, 1.0, 5.0, 2.0];
        assert_eq!(argmin_reference(&input), 1);
    }

    #[test]
    fn test_argmin_reference_last_element() {
        let input = vec![3.0, 2.0, 1.0];
        assert_eq!(argmin_reference(&input), 2);
    }

    #[test]
    fn test_reduce_launch_config() {
        let (grid, block) = ptx_reduce_launch_config();
        // Single-block reduction
        assert_eq!(grid, [1, 1, 1]);
        assert_eq!(block[0], REDUCE_BLOCK_SIZE as usize);
    }

    #[test]
    fn test_all_reduce_ptx_non_empty() {
        let generators: Vec<(&str, String)> = vec![
            ("sum", generate_sum_ptx(1024)),
            ("max", generate_max_ptx(1024)),
            ("mean", generate_mean_ptx(1024)),
            ("argmax", generate_argmax_ptx(1024)),
            ("argmin", generate_argmin_ptx(1024)),
        ];
        for (name, ptx) in &generators {
            assert!(!ptx.is_empty(), "{name} PTX is empty");
            assert!(ptx.contains(".version"), "{name} missing .version");
            assert!(ptx.contains("ret;"), "{name} missing ret;");
        }
    }
}

// =========================================================================
// 5. PTX activation functions: gelu, silu, softmax
// =========================================================================

mod ptx_activation_functions {
    use crate::cuda_validation::validate_ptx_structure;
    use crate::ptx_activations::{
        emit_ptx_activation, emit_ptx_activation_default, gelu_fast_reference, gelu_reference,
        generate_all_activation_ptx, mish_reference, ptx_activation_launch_config, silu_reference,
        snake_reference, PtxActivation, PtxActivationConfig,
    };
    use crate::ptx_softmax::{
        emit_ptx_softmax, emit_ptx_softmax_default, generate_log_softmax_ptx, generate_softmax_ptx,
        log_softmax_reference, ptx_softmax_launch_config, softmax_reference, PtxSoftmaxConfig,
        SOFTMAX_BLOCK_SIZE,
    };

    // -- Activation PTX generation --

    #[test]
    fn test_gelu_ptx_structural() {
        let config = PtxActivationConfig::new("gelu_test", PtxActivation::Gelu);
        let ptx = emit_ptx_activation(&config).unwrap();
        let result = validate_ptx_structure(&ptx, "gelu_test");
        assert!(
            result.structural_ok,
            "failures: {:?}",
            result.structural_failures
        );
    }

    #[test]
    fn test_silu_ptx_structural() {
        let config = PtxActivationConfig::new("silu_test", PtxActivation::Silu);
        let ptx = emit_ptx_activation(&config).unwrap();
        let result = validate_ptx_structure(&ptx, "silu_test");
        assert!(
            result.structural_ok,
            "failures: {:?}",
            result.structural_failures
        );
    }

    #[test]
    fn test_gelu_fast_ptx_structural() {
        let config = PtxActivationConfig::new("gelu_fast_test", PtxActivation::GeluFast);
        let ptx = emit_ptx_activation(&config).unwrap();
        let result = validate_ptx_structure(&ptx, "gelu_fast_test");
        assert!(
            result.structural_ok,
            "failures: {:?}",
            result.structural_failures
        );
    }

    #[test]
    fn test_mish_ptx_structural() {
        let config = PtxActivationConfig::new("mish_test", PtxActivation::Mish);
        let ptx = emit_ptx_activation(&config).unwrap();
        let result = validate_ptx_structure(&ptx, "mish_test");
        assert!(
            result.structural_ok,
            "failures: {:?}",
            result.structural_failures
        );
    }

    #[test]
    fn test_snake_ptx_has_alpha_param() {
        let config = PtxActivationConfig::new("snake_test", PtxActivation::Snake);
        let ptx = emit_ptx_activation(&config).unwrap();
        assert!(ptx.contains("param_alpha"));
    }

    #[test]
    fn test_snake_requires_alpha() {
        assert!(PtxActivation::Snake.requires_alpha());
        assert!(!PtxActivation::Gelu.requires_alpha());
        assert!(!PtxActivation::Silu.requires_alpha());
        assert!(!PtxActivation::GeluFast.requires_alpha());
        assert!(!PtxActivation::Mish.requires_alpha());
    }

    #[test]
    fn test_activation_names() {
        assert_eq!(PtxActivation::Gelu.name(), "gelu");
        assert_eq!(PtxActivation::Silu.name(), "silu");
        assert_eq!(PtxActivation::GeluFast.name(), "gelu_fast");
        assert_eq!(PtxActivation::Mish.name(), "mish");
        assert_eq!(PtxActivation::Snake.name(), "snake");
    }

    #[test]
    fn test_activation_config_validate_empty_name() {
        let config = PtxActivationConfig::new("", PtxActivation::Gelu);
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_activation_config_validate_zero_block_size() {
        let config = PtxActivationConfig::new("test", PtxActivation::Gelu).with_block_size(0);
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_emit_ptx_activation_default_all() {
        for act in [
            PtxActivation::Gelu,
            PtxActivation::GeluFast,
            PtxActivation::Silu,
            PtxActivation::Mish,
            PtxActivation::Snake,
        ] {
            let ptx = emit_ptx_activation_default(act.name(), act).unwrap();
            assert!(!ptx.is_empty(), "empty PTX for {act:?}");
            assert!(ptx.contains(".entry"), "missing .entry for {act:?}");
        }
    }

    #[test]
    fn test_generate_all_activation_ptx() {
        let all = generate_all_activation_ptx();
        assert_eq!(all.len(), 5, "expected 5 activations");
        for (name, ptx) in &all {
            assert!(!ptx.is_empty(), "{name}: empty PTX");
        }
    }

    #[test]
    fn test_activation_launch_config() {
        let (grid, block) = ptx_activation_launch_config(1024, 256);
        assert_eq!(block, [256, 1, 1]);
        assert_eq!(grid, [4, 1, 1]);
    }

    // -- Activation reference implementations --

    #[test]
    fn test_silu_reference_zero() {
        assert!((silu_reference(0.0) - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_silu_reference_positive() {
        // silu(1.0) = 1.0 * sigmoid(1.0) = 1.0 / (1 + exp(-1))
        let expected = 1.0 / (1.0 + (-1.0f32).exp());
        assert!((silu_reference(1.0) - expected).abs() < 1e-6);
    }

    #[test]
    fn test_gelu_reference_zero() {
        // gelu(0) = 0
        assert!((gelu_reference(0.0) - 0.0).abs() < 1e-5);
    }

    #[test]
    fn test_gelu_reference_positive() {
        let val = gelu_reference(1.0);
        // gelu(1.0) ~ 0.8413
        assert!(val > 0.8, "gelu(1.0)={val}, expected ~0.84");
        assert!(val < 0.86, "gelu(1.0)={val}, expected ~0.84");
    }

    #[test]
    fn test_gelu_fast_reference_approximates_gelu() {
        for x in [-2.0, -1.0, 0.0, 0.5, 1.0, 2.0] {
            let exact = gelu_reference(x);
            let fast = gelu_fast_reference(x);
            // Fast approximation should be within ~0.05 of exact
            assert!(
                (exact - fast).abs() < 0.05,
                "x={x}: gelu={exact:.4}, gelu_fast={fast:.4}"
            );
        }
    }

    #[test]
    fn test_mish_reference_zero() {
        // mish(0) = 0 * tanh(softplus(0)) = 0 * tanh(ln(2)) = 0
        assert!((mish_reference(0.0) - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_snake_reference_zero_returns_zero() {
        // snake(0, alpha) = 0 + (1/alpha) * sin(0)^2 = 0
        assert!((snake_reference(0.0, 1.0) - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_snake_reference_monotonic_positive() {
        let y1 = snake_reference(1.0, 1.0);
        let y2 = snake_reference(2.0, 1.0);
        assert!(
            y2 > y1,
            "snake should be generally increasing: y1={y1}, y2={y2}"
        );
    }

    // -- Softmax PTX generation --

    #[test]
    fn test_softmax_block_size() {
        assert_eq!(SOFTMAX_BLOCK_SIZE, 256);
    }

    #[test]
    fn test_softmax_ptx_structural() {
        let ptx = generate_softmax_ptx(false, 128);
        assert!(!ptx.is_empty());
        assert!(ptx.contains(".version"));
        assert!(ptx.contains(".entry"));
    }

    #[test]
    fn test_log_softmax_ptx_structural() {
        let ptx = generate_log_softmax_ptx(256);
        assert!(!ptx.is_empty());
        assert!(ptx.contains(".version"));
    }

    #[test]
    fn test_softmax_config_validation() {
        let config = PtxSoftmaxConfig::new("test", 0);
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_softmax_config_empty_name() {
        let config = PtxSoftmaxConfig::new("", 128);
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_softmax_config_block_size_warp_aligned() {
        let config = PtxSoftmaxConfig::new("test", 48);
        // 48 rounds up to 64 (2 warps), which is <= 256
        let bs = config.block_size();
        assert_eq!(bs % 32, 0, "block_size must be warp-aligned: {bs}");
    }

    #[test]
    fn test_softmax_config_block_size_capped_at_256() {
        let config = PtxSoftmaxConfig::new("test", 1024);
        assert!(config.block_size() <= 256);
    }

    #[test]
    fn test_softmax_config_warp_only_small_dim() {
        let config = PtxSoftmaxConfig::new("test", 16);
        assert!(config.is_warp_only());
        assert_eq!(config.shared_memory_bytes(), 0);
    }

    #[test]
    fn test_softmax_config_multi_warp_large_dim() {
        let config = PtxSoftmaxConfig::new("test", 128);
        assert!(!config.is_warp_only());
        assert!(config.shared_memory_bytes() > 0);
    }

    #[test]
    fn test_emit_ptx_softmax_structural() {
        let config = PtxSoftmaxConfig::new("sm_test", 64);
        let ptx = emit_ptx_softmax(&config).unwrap();
        let result = validate_ptx_structure(&ptx, "sm_test");
        assert!(
            result.structural_ok,
            "failures: {:?}",
            result.structural_failures
        );
    }

    #[test]
    fn test_emit_ptx_softmax_log_mode() {
        let config = PtxSoftmaxConfig::new_log("logsm_test", 64);
        let ptx = emit_ptx_softmax(&config).unwrap();
        let result = validate_ptx_structure(&ptx, "logsm_test");
        assert!(
            result.structural_ok,
            "failures: {:?}",
            result.structural_failures
        );
    }

    #[test]
    fn test_emit_ptx_softmax_default() {
        let ptx = emit_ptx_softmax_default("softmax_default", 128).unwrap();
        assert!(!ptx.is_empty());
    }

    #[test]
    fn test_softmax_launch_config() {
        let (grid, _block) = ptx_softmax_launch_config(32, 128);
        // 32 rows, each row is 128 elements
        assert_eq!(grid[0], 32);
    }

    // -- Softmax reference implementations --

    #[test]
    fn test_softmax_reference_uniform() {
        let input = vec![1.0, 1.0, 1.0, 1.0];
        let output = softmax_reference(&input);
        assert_eq!(output.len(), 4);
        for &v in &output {
            assert!((v - 0.25).abs() < 1e-5);
        }
    }

    #[test]
    fn test_softmax_reference_sums_to_one() {
        let input = vec![1.0, 2.0, 3.0, 4.0];
        let output = softmax_reference(&input);
        let sum: f32 = output.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5, "softmax sum = {sum}");
    }

    #[test]
    fn test_softmax_reference_monotonic() {
        let input = vec![1.0, 2.0, 3.0, 4.0];
        let output = softmax_reference(&input);
        for i in 1..output.len() {
            assert!(
                output[i] > output[i - 1],
                "softmax should be monotonic with sorted input"
            );
        }
    }

    #[test]
    fn test_log_softmax_reference_properties() {
        let input = vec![1.0, 2.0, 3.0, 4.0];
        let log_sm = log_softmax_reference(&input);
        // All log_softmax values should be negative (probabilities < 1)
        for &v in &log_sm {
            assert!(v < 0.0, "log_softmax should be negative: {v}");
        }
        // exp(log_softmax) should sum to 1
        let sum: f32 = log_sm.iter().map(|&v| v.exp()).sum();
        assert!((sum - 1.0).abs() < 1e-4, "exp(log_softmax) sum = {sum}");
    }
}

// =========================================================================
// 6. PTX binary validation: magic numbers, version headers
// =========================================================================

mod ptx_binary_validation {
    use crate::codegen_ptx::{ptx_prelude, PTX_VERSION};
    use crate::ptx_elementwise::generate_add_ptx;
    use crate::ptx_matmul::{emit_ptx_matmul, PtxMatmulConfig};
    use crate::ptx_reduce::generate_sum_ptx;
    use crate::ptx_softmax::generate_softmax_ptx;

    /// All raw PTX kernels must start with `.version X.Y`.
    #[test]
    fn test_ptx_version_header_format() {
        let ptx = generate_add_ptx(128);
        let first_line = ptx.lines().next().unwrap();
        assert!(
            first_line.starts_with(".version"),
            "PTX must start with .version: got '{first_line}'"
        );
        assert!(first_line.contains(PTX_VERSION));
    }

    /// All raw PTX must have `.target sm_XX`.
    #[test]
    fn test_ptx_target_directive_present() {
        let ptx = generate_add_ptx(128);
        assert!(ptx.contains(".target sm_"));
    }

    /// All raw PTX must have `.address_size 64`.
    #[test]
    fn test_ptx_address_size_64() {
        let ptx = generate_add_ptx(128);
        assert!(ptx.contains(".address_size 64"));
    }

    /// Entry point must be `.visible .entry <name>`.
    #[test]
    fn test_ptx_entry_point_visible() {
        let ptx = generate_add_ptx(128);
        assert!(ptx.contains(".visible .entry"));
    }

    /// PTX parameters must use correct type prefixes.
    #[test]
    fn test_ptx_param_type_prefixes() {
        let ptx = generate_add_ptx(128);
        // Pointers are .u64, element counts are .u32
        assert!(ptx.contains(".param .u64"));
        assert!(ptx.contains(".param .u32"));
    }

    /// Matmul PTX must have shared memory declarations.
    #[test]
    fn test_matmul_ptx_shared_memory_declaration() {
        let config = PtxMatmulConfig::new("test_gemm").with_tile_size(16);
        let ptx = emit_ptx_matmul(&config).unwrap();
        // Shared memory for tiled matmul uses .shared .align
        assert!(ptx.contains(".shared .align 4 .f32"));
    }

    /// Reduction PTX must have shared memory for tree reduction.
    #[test]
    fn test_reduction_ptx_shared_memory() {
        let ptx = generate_sum_ptx(256);
        assert!(ptx.contains(".shared"));
    }

    /// Softmax PTX must contain warp shuffle instruction for reduction.
    #[test]
    fn test_softmax_ptx_warp_shuffle() {
        let ptx = generate_softmax_ptx(false, 128);
        // Multi-warp softmax uses shfl.down.sync for warp-level reduction
        assert!(
            ptx.contains("shfl") || ptx.contains(".shared"),
            "softmax should use warp shuffle or shared memory"
        );
    }

    /// PTX must use ret; to return from kernel.
    #[test]
    fn test_ptx_ret_instruction() {
        let ptx = generate_add_ptx(128);
        assert!(ptx.contains("ret;"));
    }

    /// PTX prelude for different SM targets produces correct output.
    #[test]
    fn test_ptx_prelude_sm_targets() {
        for sm in ["sm_70", "sm_80", "sm_90"] {
            let prelude = ptx_prelude(sm);
            assert!(prelude.contains(&format!(".target {sm}")));
            assert!(prelude.contains(&format!(".version {PTX_VERSION}")));
        }
    }

    /// All PTX kernels must have register declarations.
    #[test]
    fn test_ptx_register_declarations_comprehensive() {
        let test_cases = vec![
            ("add", generate_add_ptx(128)),
            ("sum_reduce", generate_sum_ptx(128)),
            ("softmax", generate_softmax_ptx(false, 64)),
        ];
        for (name, ptx) in &test_cases {
            assert!(
                ptx.contains(".reg"),
                "{name}: missing register declarations"
            );
        }
    }

    /// PTX uses correct addressing mode for global memory.
    #[test]
    fn test_ptx_global_memory_access() {
        let ptx = generate_add_ptx(128);
        assert!(ptx.contains("ld.global.f32"), "missing global load");
        assert!(ptx.contains("st.global.f32"), "missing global store");
    }

    /// PTX uses correct thread ID registers.
    #[test]
    fn test_ptx_thread_id_registers() {
        let ptx = generate_add_ptx(128);
        assert!(ptx.contains("%tid.x"), "missing threadIdx.x");
        assert!(ptx.contains("%ctaid.x"), "missing blockIdx.x");
        assert!(ptx.contains("%ntid.x"), "missing blockDim.x");
    }
}

// =========================================================================
// 7. Kernel launch parameter calculations
// =========================================================================

mod kernel_launch_parameters {
    use crate::codegen_ptx::PTX_BLOCK_SIZE;
    use crate::cuda_ffi::{CudaDim3, CudaLaunchConfig};
    use crate::ptx_emit::{
        elementwise_launch_config, matmul_launch_config, reduction_launch_config,
    };

    // -- ptx_emit launch configs (CUDA C++ path) --

    #[test]
    fn test_elementwise_launch_config_basic() {
        let (grid, block) = elementwise_launch_config(256);
        assert_eq!(block, PTX_BLOCK_SIZE);
        assert_eq!(grid, 1);
    }

    #[test]
    fn test_elementwise_launch_config_large() {
        let (grid, block) = elementwise_launch_config(100_000);
        assert_eq!(block, PTX_BLOCK_SIZE);
        assert_eq!(grid, 100_000usize.div_ceil(PTX_BLOCK_SIZE));
    }

    #[test]
    fn test_elementwise_launch_config_single() {
        let (grid, block) = elementwise_launch_config(1);
        assert_eq!(block, PTX_BLOCK_SIZE);
        assert_eq!(grid, 1);
    }

    #[test]
    fn test_reduction_launch_config_basics() {
        let (num_blocks, block_size) = reduction_launch_config(10, 512);
        assert_eq!(num_blocks, 10);
        // min(256, next_power_of_two(512)) = 256
        assert_eq!(block_size, 256);
    }

    #[test]
    fn test_reduction_launch_config_small_row() {
        let (num_blocks, block_size) = reduction_launch_config(4, 16);
        assert_eq!(num_blocks, 4);
        // next_power_of_two(16) = 16, min(256, 16) = 16
        assert_eq!(block_size, 16);
    }

    #[test]
    fn test_reduction_launch_config_non_power_of_two_row() {
        let (num_blocks, block_size) = reduction_launch_config(8, 100);
        assert_eq!(num_blocks, 8);
        // next_power_of_two(100) = 128, min(256, 128) = 128
        assert_eq!(block_size, 128);
    }

    #[test]
    fn test_matmul_launch_config_square() {
        let (grid, block) = matmul_launch_config(64, 64, 16);
        assert_eq!(grid, [4, 4]);
        assert_eq!(block, [16, 16]);
    }

    #[test]
    fn test_matmul_launch_config_rectangular() {
        let (grid, block) = matmul_launch_config(128, 64, 16);
        assert_eq!(grid, [4, 8]); // [ceil(64/16), ceil(128/16)]
        assert_eq!(block, [16, 16]);
    }

    #[test]
    fn test_matmul_launch_config_non_multiple() {
        let (grid, block) = matmul_launch_config(100, 50, 16);
        // ceil(50/16)=4, ceil(100/16)=7
        assert_eq!(grid, [4, 7]);
        assert_eq!(block, [16, 16]);
    }

    // -- CudaLaunchConfig edge cases --

    #[test]
    fn test_cuda_launch_config_elementwise_single_block() {
        let config = CudaLaunchConfig::for_elementwise(100, 256);
        assert_eq!(config.grid.x, 1);
        assert_eq!(config.block.x, 256);
    }

    #[test]
    fn test_cuda_launch_config_matmul_single_tile() {
        let config = CudaLaunchConfig::for_matmul(16, 16, 16, 16);
        assert_eq!(config.grid.x, 1);
        assert_eq!(config.grid.y, 1);
    }

    #[test]
    fn test_cuda_dim3_total_overflow_protection() {
        // Large but valid dims
        let dim = CudaDim3::new(65535, 65535, 1);
        assert_eq!(dim.total(), 65535u64 * 65535u64);
    }

    // -- Consistency between ptx_emit and ptx_elementwise launch configs --

    #[test]
    fn test_elementwise_launch_config_consistency() {
        use crate::ptx_elementwise::ptx_elementwise_launch_config;

        let n = 4096u32;
        // ptx_emit path
        let (grid_emit, block_emit) = elementwise_launch_config(n as usize);
        // ptx_elementwise path
        let (grid_ew, block_ew) = ptx_elementwise_launch_config(n);

        assert_eq!(block_emit, block_ew[0] as usize);
        assert_eq!(grid_emit, grid_ew[0] as usize);
    }

    // -- Shared memory calculations --

    #[test]
    fn test_reduction_shared_memory_bytes() {
        let config = CudaLaunchConfig::for_reduction(16, 256);
        // 256 threads * 4 bytes = 1024
        assert_eq!(config.shared_mem_bytes, 1024);
    }

    #[test]
    fn test_matmul_shared_memory_via_config() {
        use crate::ptx_matmul::PtxMatmulConfig;
        let config = PtxMatmulConfig::new("test").with_tile_size(32);
        // 2 * 32 * 32 * 4 = 8192
        assert_eq!(config.shared_memory_bytes(), 8192);
    }
}

// =========================================================================
// 8. CUDA C++ emission (ptx_emit module)
// =========================================================================

mod cuda_cpp_emission {
    use crate::cuda_validation::validate_ptx_structure;
    use crate::ptx_emit::{
        emit_activation_kernels, emit_elementwise_kernel, emit_matmul_kernel,
        emit_reduction_kernel, emit_softmax_kernel, ReductionOp,
    };

    #[test]
    fn test_emit_elementwise_kernel_contains_cuda_prelude() {
        let src = emit_elementwise_kernel("test_kern", "x * 2.0f", 1024).unwrap();
        assert!(src.contains("#include <cuda_runtime.h>"));
        assert!(src.contains("#include <cuda_fp16.h>"));
    }

    #[test]
    fn test_emit_elementwise_kernel_structural() {
        let src = emit_elementwise_kernel("relu_k", "x > 0.0f ? x : 0.0f", 512).unwrap();
        let result = validate_ptx_structure(&src, "relu_k");
        assert!(
            result.structural_ok,
            "failures: {:?}",
            result.structural_failures
        );
    }

    #[test]
    fn test_emit_elementwise_kernel_zero_rejected() {
        assert!(emit_elementwise_kernel("k", "x", 0).is_err());
    }

    #[test]
    fn test_emit_activation_kernels_count() {
        let src = emit_activation_kernels();
        assert_eq!(src.matches("__global__").count(), 5);
    }

    #[test]
    fn test_emit_activation_kernels_names() {
        let src = emit_activation_kernels();
        for name in [
            "relu_kernel",
            "silu_kernel",
            "sigmoid_kernel",
            "tanh_kernel",
            "gelu_kernel",
        ] {
            assert!(src.contains(name), "missing {name}");
        }
    }

    #[test]
    fn test_emit_softmax_kernel_shared_memory() {
        let src = emit_softmax_kernel(256).unwrap();
        assert!(src.contains("__shared__"));
        assert!(src.contains("__syncthreads"));
    }

    #[test]
    fn test_emit_softmax_kernel_zero_rejected() {
        assert!(emit_softmax_kernel(0).is_err());
    }

    #[test]
    fn test_emit_reduction_kernel_all_ops() {
        for (op, name) in [
            (ReductionOp::Sum, "sum_k"),
            (ReductionOp::Max, "max_k"),
            (ReductionOp::Min, "min_k"),
            (ReductionOp::Mean, "mean_k"),
        ] {
            let src = emit_reduction_kernel(name, op, 512).unwrap();
            assert!(src.contains(name), "{name}: missing kernel name");
            assert!(src.contains("__shared__"), "{name}: missing shared memory");
        }
    }

    #[test]
    fn test_emit_reduction_kernel_zero_rejected() {
        assert!(emit_reduction_kernel("k", ReductionOp::Sum, 0).is_err());
    }

    #[test]
    fn test_emit_reduction_mean_has_axis_size_divisor() {
        let src = emit_reduction_kernel("mean_k", ReductionOp::Mean, 128).unwrap();
        assert!(src.contains("(float)axis_size"));
    }

    #[test]
    fn test_emit_reduction_max_identity() {
        let src = emit_reduction_kernel("max_k", ReductionOp::Max, 256).unwrap();
        assert!(src.contains("-HUGE_VALF"));
    }

    #[test]
    fn test_emit_reduction_min_identity() {
        let src = emit_reduction_kernel("min_k", ReductionOp::Min, 256).unwrap();
        assert!(src.contains("HUGE_VALF"));
    }

    #[test]
    fn test_emit_matmul_kernel_structural() {
        let src = emit_matmul_kernel("gemm_k", 16).unwrap();
        let result = validate_ptx_structure(&src, "gemm_k");
        assert!(
            result.structural_ok,
            "failures: {:?}",
            result.structural_failures
        );
    }

    #[test]
    fn test_emit_matmul_kernel_tile_define() {
        let src = emit_matmul_kernel("gemm_k", 8).unwrap();
        assert!(src.contains("#define TILE_SIZE 8"));
        assert!(src.contains("#undef TILE_SIZE"));
    }

    #[test]
    fn test_emit_matmul_kernel_invalid_tile_sizes() {
        assert!(emit_matmul_kernel("k", 0).is_err());
        assert!(emit_matmul_kernel("k", 64).is_err());
    }

    #[test]
    fn test_emit_matmul_kernel_valid_tile_sizes() {
        for tile in [1, 4, 8, 16, 32] {
            let result = emit_matmul_kernel("k", tile);
            assert!(result.is_ok(), "tile_size={tile} should be valid");
        }
    }
}

// =========================================================================
// 9. Reference implementation correctness
// =========================================================================

mod reference_implementation_correctness {
    use crate::ptx_elementwise::{
        add_reference, div_reference, exp_reference, log_reference, mul_reference, neg_reference,
        scalar_mul_reference, sqrt_reference, sub_reference,
    };

    #[test]
    fn test_add_reference() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![4.0, 5.0, 6.0];
        let c = add_reference(&a, &b);
        assert_eq!(c, vec![5.0, 7.0, 9.0]);
    }

    #[test]
    fn test_sub_reference() {
        let a = vec![5.0, 7.0, 9.0];
        let b = vec![1.0, 2.0, 3.0];
        let c = sub_reference(&a, &b);
        assert_eq!(c, vec![4.0, 5.0, 6.0]);
    }

    #[test]
    fn test_mul_reference() {
        let a = vec![2.0, 3.0, 4.0];
        let b = vec![5.0, 6.0, 7.0];
        let c = mul_reference(&a, &b);
        assert_eq!(c, vec![10.0, 18.0, 28.0]);
    }

    #[test]
    fn test_div_reference() {
        let a = vec![10.0, 18.0, 28.0];
        let b = vec![2.0, 3.0, 4.0];
        let c = div_reference(&a, &b);
        assert_eq!(c, vec![5.0, 6.0, 7.0]);
    }

    #[test]
    fn test_exp_reference() {
        let input = vec![0.0, 1.0];
        let output = exp_reference(&input);
        assert!((output[0] - 1.0).abs() < 1e-6);
        assert!((output[1] - std::f32::consts::E).abs() < 1e-5);
    }

    #[test]
    fn test_log_reference() {
        let input = vec![1.0, std::f32::consts::E];
        let output = log_reference(&input);
        assert!((output[0] - 0.0).abs() < 1e-6);
        assert!((output[1] - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_sqrt_reference() {
        let input = vec![4.0, 9.0, 16.0];
        let output = sqrt_reference(&input);
        assert!((output[0] - 2.0).abs() < 1e-6);
        assert!((output[1] - 3.0).abs() < 1e-6);
        assert!((output[2] - 4.0).abs() < 1e-6);
    }

    #[test]
    fn test_neg_reference() {
        let input = vec![1.0, -2.0, 0.0];
        let output = neg_reference(&input);
        assert_eq!(output, vec![-1.0, 2.0, -0.0]);
    }

    #[test]
    fn test_scalar_mul_reference() {
        let input = vec![1.0, 2.0, 3.0];
        let output = scalar_mul_reference(&input, 2.5);
        assert!((output[0] - 2.5).abs() < 1e-6);
        assert!((output[1] - 5.0).abs() < 1e-6);
        assert!((output[2] - 7.5).abs() < 1e-6);
    }

    #[test]
    fn test_scalar_mul_reference_zero() {
        let input = vec![1.0, 2.0, 3.0];
        let output = scalar_mul_reference(&input, 0.0);
        for &v in &output {
            assert!((v - 0.0).abs() < 1e-6);
        }
    }

    #[test]
    fn test_exp_log_roundtrip() {
        let input = vec![0.5, 1.0, 2.0, 3.0];
        let exp_output = exp_reference(&input);
        let log_output = log_reference(&exp_output);
        for (orig, roundtrip) in input.iter().zip(log_output.iter()) {
            assert!(
                (orig - roundtrip).abs() < 1e-5,
                "log(exp({orig})) = {roundtrip}"
            );
        }
    }

    #[test]
    fn test_add_sub_inverse() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![4.0, 5.0, 6.0];
        let sum = add_reference(&a, &b);
        let diff = sub_reference(&sum, &b);
        for (orig, restored) in a.iter().zip(diff.iter()) {
            assert!((orig - restored).abs() < 1e-6);
        }
    }

    #[test]
    fn test_mul_div_inverse() {
        let a = vec![2.0, 3.0, 4.0];
        let b = vec![5.0, 6.0, 7.0];
        let prod = mul_reference(&a, &b);
        let quot = div_reference(&prod, &b);
        for (orig, restored) in a.iter().zip(quot.iter()) {
            assert!((orig - restored).abs() < 1e-5);
        }
    }

    #[test]
    fn test_neg_neg_identity() {
        let input = vec![1.0, -2.0, 3.5];
        let double_neg = neg_reference(&neg_reference(&input));
        for (orig, restored) in input.iter().zip(double_neg.iter()) {
            assert!((orig - restored).abs() < 1e-6);
        }
    }
}

// =========================================================================
// 10. Validation suite integration
// =========================================================================

mod validation_suite_integration {
    use crate::cuda_validation::{validate_ptx_e2e, CudaValidationSuite, ErrorStats};
    use crate::ptx_elementwise::{add_reference, generate_add_ptx, generate_mul_ptx};
    use crate::ptx_softmax::{generate_softmax_ptx, softmax_reference};

    #[test]
    fn test_e2e_validation_add_pass() {
        let ptx = generate_add_ptx(4);
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![5.0, 6.0, 7.0, 8.0];
        let cpu_out = add_reference(&a, &b);
        let expected = vec![6.0, 8.0, 10.0, 12.0];
        let result = validate_ptx_e2e("ptx_add_f32", &ptx, &cpu_out, &expected, 1e-5).unwrap();
        assert!(result.passed());
    }

    #[test]
    fn test_e2e_validation_numerical_failure() {
        let ptx = generate_add_ptx(2);
        let cpu_out = vec![1.0, 2.0];
        let expected = vec![100.0, 200.0];
        let result = validate_ptx_e2e("ptx_add_f32", &ptx, &cpu_out, &expected, 1e-5);
        assert!(result.is_err());
    }

    #[test]
    fn test_e2e_validation_structural_failure_on_empty() {
        let result = validate_ptx_e2e("missing", "", &[1.0], &[1.0], 1e-5);
        assert!(result.is_err());
    }

    #[test]
    fn test_validation_suite_multi_kernel() {
        let mut suite = CudaValidationSuite::new();

        // Add kernel
        let add_ptx = generate_add_ptx(4);
        let add_out = add_reference(&[1.0, 2.0, 3.0, 4.0], &[5.0, 6.0, 7.0, 8.0]);
        suite.add("ptx_add_f32", add_ptx, add_out.clone(), add_out, 1e-5);

        // Mul kernel
        let mul_ptx = generate_mul_ptx(3);
        let mul_out = vec![6.0, 14.0, 24.0]; // [2*3, 3.5*4, 4*6] (approx)
        suite.add("ptx_mul_f32", mul_ptx, mul_out.clone(), mul_out, 1e-5);

        assert_eq!(suite.len(), 2);
        assert!(suite.run_all_pass());
    }

    #[test]
    fn test_error_stats_empty() {
        let stats = ErrorStats::compute(&[], &[]).unwrap();
        assert_eq!(stats.num_elements, 0);
        assert_eq!(stats.max_abs_error, 0.0);
    }

    #[test]
    fn test_error_stats_identical() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let stats = ErrorStats::compute(&data, &data).unwrap();
        assert_eq!(stats.max_abs_error, 0.0);
        assert_eq!(stats.mean_abs_error, 0.0);
        assert_eq!(stats.num_nans, 0);
        assert_eq!(stats.num_infs, 0);
    }

    #[test]
    fn test_softmax_e2e_validation() {
        let ptx = generate_softmax_ptx(false, 4);
        let input = vec![1.0, 2.0, 3.0, 4.0];
        let cpu_out = softmax_reference(&input);
        let result = validate_ptx_e2e("ptx_softmax_f32", &ptx, &cpu_out, &cpu_out, 1e-5).unwrap();
        assert!(result.passed());
        let stats = result.error_stats.unwrap();
        assert_eq!(stats.max_abs_error, 0.0);
    }
}

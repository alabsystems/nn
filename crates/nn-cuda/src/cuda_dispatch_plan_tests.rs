// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! CUDA dispatch plan tests — validates launch configuration construction,
//! grid/block dimension calculations, and shared memory allocation for
//! common ML kernel patterns.
//!
//! These tests verify dispatch plan logic without requiring CUDA hardware.
//! They mirror the Metal dispatch plan test patterns from nn-metal.

use crate::cuda_ffi::{sm_target, CudaDim3, CudaLaunchConfig};
use crate::cuda_runtime::validate_launch_config;

// =========================================================================
// 1. CudaDim3 construction and arithmetic
// =========================================================================

mod dim3_tests {
    use super::*;

    #[test]
    fn test_d1_sets_yz_to_one() {
        let d = CudaDim3::d1(128);
        assert_eq!(d.x, 128);
        assert_eq!(d.y, 1);
        assert_eq!(d.z, 1);
    }

    #[test]
    fn test_d2_sets_z_to_one() {
        let d = CudaDim3::d2(16, 32);
        assert_eq!(d.x, 16);
        assert_eq!(d.y, 32);
        assert_eq!(d.z, 1);
    }

    #[test]
    fn test_d3_all_dimensions() {
        let d = CudaDim3::new(4, 8, 2);
        assert_eq!(d.x, 4);
        assert_eq!(d.y, 8);
        assert_eq!(d.z, 2);
    }

    #[test]
    fn test_total_d1() {
        assert_eq!(CudaDim3::d1(256).total(), 256);
    }

    #[test]
    fn test_total_d2() {
        assert_eq!(CudaDim3::d2(16, 16).total(), 256);
    }

    #[test]
    fn test_total_d3() {
        assert_eq!(CudaDim3::new(4, 8, 2).total(), 64);
    }

    #[test]
    fn test_total_overflow_safety() {
        // Large dimensions should not overflow due to u64 arithmetic
        let d = CudaDim3::new(u32::MAX, 1, 1);
        assert_eq!(d.total(), u64::from(u32::MAX));
    }

    #[test]
    fn test_total_large_product() {
        let d = CudaDim3::new(65535, 65535, 1);
        assert_eq!(d.total(), 65535u64 * 65535);
    }

    #[test]
    fn test_dim3_equality() {
        let a = CudaDim3::new(4, 8, 2);
        let b = CudaDim3::new(4, 8, 2);
        assert_eq!(a, b);
    }

    #[test]
    fn test_dim3_inequality() {
        let a = CudaDim3::d1(128);
        let b = CudaDim3::d1(256);
        assert_ne!(a, b);
    }
}

// =========================================================================
// 2. CudaLaunchConfig construction patterns
// =========================================================================

mod launch_config_tests {
    use super::*;

    #[test]
    fn test_elementwise_exact_multiple() {
        let cfg = CudaLaunchConfig::for_elementwise(1024, 256);
        assert_eq!(cfg.grid.x, 4);
        assert_eq!(cfg.block.x, 256);
        assert_eq!(cfg.shared_mem_bytes, 0);
    }

    #[test]
    fn test_elementwise_non_multiple_rounds_up() {
        let cfg = CudaLaunchConfig::for_elementwise(1000, 256);
        assert_eq!(cfg.grid.x, 4); // ceil(1000/256) = 4
        assert_eq!(cfg.block.x, 256);
    }

    #[test]
    fn test_elementwise_single_thread() {
        let cfg = CudaLaunchConfig::for_elementwise(1, 256);
        assert_eq!(cfg.grid.x, 1);
    }

    #[test]
    fn test_elementwise_exactly_block_size() {
        let cfg = CudaLaunchConfig::for_elementwise(256, 256);
        assert_eq!(cfg.grid.x, 1);
    }

    #[test]
    fn test_elementwise_large_tensor() {
        // 100M elements: common in ML (batch * seq_len * hidden)
        let cfg = CudaLaunchConfig::for_elementwise(100_000_000, 256);
        assert_eq!(cfg.grid.x, 390625); // ceil(100M / 256)
    }

    #[test]
    fn test_reduction_grid_matches_rows() {
        let cfg = CudaLaunchConfig::for_reduction(32, 256);
        assert_eq!(cfg.grid.x, 32);
        assert_eq!(cfg.block.x, 256);
    }

    #[test]
    fn test_reduction_shared_memory_sizeof_float() {
        let cfg = CudaLaunchConfig::for_reduction(16, 128);
        // shared_mem = block_size * sizeof(float)
        assert_eq!(cfg.shared_mem_bytes, 128 * 4);
    }

    #[test]
    fn test_matmul_grid_calculation() {
        // M=128, N=64, tile_m=16, tile_n=16
        let cfg = CudaLaunchConfig::for_matmul(128, 64, 16, 16);
        assert_eq!(cfg.grid.x, 4); // ceil(64/16)
        assert_eq!(cfg.grid.y, 8); // ceil(128/16)
        assert_eq!(cfg.block.x, 16);
        assert_eq!(cfg.block.y, 16);
    }

    #[test]
    fn test_matmul_non_multiple_tiles() {
        // M=100, N=50, tile_m=16, tile_n=16
        let cfg = CudaLaunchConfig::for_matmul(100, 50, 16, 16);
        assert_eq!(cfg.grid.x, 4); // ceil(50/16) = 4
        assert_eq!(cfg.grid.y, 7); // ceil(100/16) = 7
    }

    #[test]
    fn test_matmul_square() {
        let cfg = CudaLaunchConfig::for_matmul(1024, 1024, 16, 16);
        assert_eq!(cfg.grid.x, 64);
        assert_eq!(cfg.grid.y, 64);
    }

    #[test]
    fn test_batched_config() {
        let cfg = CudaLaunchConfig::for_batched(4, 8, 16, 256);
        assert_eq!(cfg.grid.x, 4);
        assert_eq!(cfg.grid.y, 8);
        assert_eq!(cfg.grid.z, 16);
        assert_eq!(cfg.block.x, 256);
    }

    #[test]
    fn test_elementwise_preserves_1d_layout() {
        let cfg = CudaLaunchConfig::for_elementwise(512, 128);
        assert_eq!(cfg.grid.y, 1);
        assert_eq!(cfg.grid.z, 1);
        assert_eq!(cfg.block.y, 1);
        assert_eq!(cfg.block.z, 1);
    }
}

// =========================================================================
// 3. Launch config validation
// =========================================================================

mod validation_tests {
    use super::*;

    #[test]
    fn test_valid_1d_config() {
        let config = CudaLaunchConfig {
            grid: CudaDim3::d1(4),
            block: CudaDim3::d1(256),
            shared_mem_bytes: 0,
        };
        assert!(validate_launch_config(&config).is_ok());
    }

    #[test]
    fn test_valid_2d_config() {
        let config = CudaLaunchConfig {
            grid: CudaDim3::d2(64, 64),
            block: CudaDim3::d2(16, 16), // 256 threads
            shared_mem_bytes: 1024,
        };
        assert!(validate_launch_config(&config).is_ok());
    }

    #[test]
    fn test_max_threads_per_block() {
        let config = CudaLaunchConfig {
            grid: CudaDim3::d1(1),
            block: CudaDim3::d1(1024), // exactly at limit
            shared_mem_bytes: 0,
        };
        assert!(validate_launch_config(&config).is_ok());
    }

    #[test]
    fn test_reject_zero_block_x() {
        let config = CudaLaunchConfig {
            grid: CudaDim3::d1(1),
            block: CudaDim3::d1(0),
            shared_mem_bytes: 0,
        };
        let err = validate_launch_config(&config).unwrap_err();
        assert!(err.to_string().contains("non-zero"));
    }

    #[test]
    fn test_reject_zero_grid_x() {
        let config = CudaLaunchConfig {
            grid: CudaDim3::d1(0),
            block: CudaDim3::d1(256),
            shared_mem_bytes: 0,
        };
        assert!(validate_launch_config(&config).is_err());
    }

    #[test]
    fn test_reject_zero_block_y() {
        let config = CudaLaunchConfig {
            grid: CudaDim3::d1(1),
            block: CudaDim3::new(256, 0, 1),
            shared_mem_bytes: 0,
        };
        assert!(validate_launch_config(&config).is_err());
    }

    #[test]
    fn test_reject_zero_block_z() {
        let config = CudaLaunchConfig {
            grid: CudaDim3::d1(1),
            block: CudaDim3::new(256, 1, 0),
            shared_mem_bytes: 0,
        };
        assert!(validate_launch_config(&config).is_err());
    }

    #[test]
    fn test_reject_exceeds_1024_threads() {
        let config = CudaLaunchConfig {
            grid: CudaDim3::d1(1),
            block: CudaDim3::d2(64, 32), // 2048 threads
            shared_mem_bytes: 0,
        };
        let err = validate_launch_config(&config).unwrap_err();
        assert!(err.to_string().contains("1024"));
    }

    #[test]
    fn test_reject_3d_exceeds_1024_threads() {
        let config = CudaLaunchConfig {
            grid: CudaDim3::d1(1),
            block: CudaDim3::new(16, 16, 8), // 2048 threads
            shared_mem_bytes: 0,
        };
        assert!(validate_launch_config(&config).is_err());
    }

    #[test]
    fn test_shared_memory_nonzero_accepted() {
        let config = CudaLaunchConfig {
            grid: CudaDim3::d1(4),
            block: CudaDim3::d1(256),
            shared_mem_bytes: 48 * 1024, // 48KB — common limit
        };
        assert!(validate_launch_config(&config).is_ok());
    }
}

// =========================================================================
// 4. SM target constants
// =========================================================================

mod sm_target_tests {
    use super::*;

    #[test]
    fn test_sm_target_strings_are_valid_format() {
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
            assert!(
                target.starts_with("sm_"),
                "SM target must start with 'sm_': {target}"
            );
            let num_part = &target[3..];
            assert!(
                num_part.parse::<u32>().is_ok(),
                "SM target numeric suffix must be a number: {target}"
            );
        }
    }

    #[test]
    fn test_sm_targets_are_ordered() {
        let targets = [
            sm_target::SM_70,
            sm_target::SM_75,
            sm_target::SM_80,
            sm_target::SM_86,
            sm_target::SM_89,
            sm_target::SM_90,
            sm_target::SM_100,
        ];
        for pair in targets.windows(2) {
            let a: u32 = pair[0][3..].parse().unwrap();
            let b: u32 = pair[1][3..].parse().unwrap();
            assert!(
                a < b,
                "SM targets must be in ascending order: {} >= {}",
                pair[0],
                pair[1]
            );
        }
    }

    #[test]
    fn test_sm_80_is_default() {
        assert_eq!(crate::codegen_ptx::DEFAULT_SM_TARGET, "sm_80");
    }
}

// =========================================================================
// 5. Dispatch plan coverage for ML operations
// =========================================================================

mod dispatch_coverage {
    use super::*;

    /// Verify that each ML operation type produces a valid launch config.
    #[test]
    fn test_elementwise_dispatch_plan_for_common_sizes() {
        // Common tensor sizes in ML models
        let sizes = [
            1,           // scalar
            128,         // small
            768,         // BERT hidden
            1024,        // GPT hidden
            4096,        // GPT-2 large hidden
            49152,       // 768 * 64 attention
            786_432,     // batch * seq * hidden
            100_663_296, // large batch
        ];
        for &n in &sizes {
            let cfg = CudaLaunchConfig::for_elementwise(n, 256);
            assert!(
                validate_launch_config(&cfg).is_ok(),
                "elementwise dispatch for {n} elements should be valid"
            );
            // Grid must cover all elements
            let total_threads = u64::from(cfg.grid.x) * u64::from(cfg.block.x);
            assert!(
                total_threads >= n as u64,
                "grid must cover all {n} elements, only covers {total_threads}"
            );
        }
    }

    #[test]
    fn test_reduction_dispatch_plan_for_common_sizes() {
        // num_rows = batch * n_heads (for softmax) or batch (for layer ops)
        let row_counts = [1, 8, 32, 128, 512, 2048];
        for &rows in &row_counts {
            let cfg = CudaLaunchConfig::for_reduction(rows, 256);
            assert!(
                validate_launch_config(&cfg).is_ok(),
                "reduction dispatch for {rows} rows should be valid"
            );
            assert_eq!(cfg.grid.x, rows as u32);
        }
    }

    #[test]
    fn test_matmul_dispatch_plan_for_common_shapes() {
        // (M, N, K) shapes common in ML
        let shapes = [
            (32, 32, 32),       // small
            (128, 768, 768),    // BERT attention
            (512, 1024, 1024),  // GPT-2
            (1024, 4096, 1024), // large FF
        ];
        for &(m, n, _k) in &shapes {
            let cfg = CudaLaunchConfig::for_matmul(m, n, 16, 16);
            assert!(
                validate_launch_config(&cfg).is_ok(),
                "matmul dispatch for ({m}, {n}) should be valid"
            );
        }
    }

    #[test]
    fn test_batched_dispatch_plan() {
        // Batched attention: batch * heads grids
        let cfg = CudaLaunchConfig::for_batched(4, 8, 32, 256);
        assert!(validate_launch_config(&cfg).is_ok());
        assert_eq!(cfg.grid.z, 32); // batch dimension
    }
}

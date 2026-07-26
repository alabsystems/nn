// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Part of #4186.
//!
//! Extended tests for workgroup utilities and compute pipeline types.
//! Focuses on cross-module integration, boundary conditions, compositional
//! correctness, and edge cases not covered by the per-module test files.

use crate::compute_pipeline::{
    compute_grid_dims, spirv_words_to_bytes, BufferBinding, CompiledShader, DispatchConfig,
    PushConstants, VulkanComputeConfig, VulkanPipelineError,
};
use crate::spirv_emit::{SPIRV_MAGIC, SPIRV_VERSION_1_5};
use crate::workgroup::{
    optimal_elementwise_workgroup, push_constants_1d, push_constants_matmul,
    push_constants_reduction, validate_dispatch, workgroup_count_1d, workgroup_count_2d,
    workgroup_count_row_reduce,
};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Build a minimal valid SPIR-V binary (20 bytes = 5 header words).
fn minimal_spirv() -> Vec<u8> {
    spirv_words_to_bytes(&[SPIRV_MAGIC, SPIRV_VERSION_1_5, 0, 0, 0])
}

/// Create a `CompiledShader` with the given binding count and push constant size.
fn shader_with(num_bindings: u32, push_constant_size: u32, wg: [u32; 3]) -> CompiledShader {
    CompiledShader::new(
        minimal_spirv(),
        "main",
        num_bindings,
        push_constant_size,
        wg,
    )
    .expect("shader construction must succeed")
}

// ===========================================================================
// 1. Cross-module: workgroup utilities feed into compute pipeline dispatch
// ===========================================================================

#[test]
fn test_workgroup_count_1d_feeds_compute_grid_dims_agreement() {
    // workgroup_count_1d and compute_grid_dims should produce identical x-values
    // for the same inputs.
    for total in [1, 7, 64, 255, 256, 257, 512, 1000, 10_000, 100_000] {
        let wg_count = workgroup_count_1d(total, 256);
        let grid = compute_grid_dims(total, [256, 1, 1]);
        assert_eq!(
            wg_count, grid[0],
            "workgroup_count_1d({total}, 256)={wg_count} != compute_grid_dims[0]={}",
            grid[0]
        );
        assert_eq!(grid[1], 1);
        assert_eq!(grid[2], 1);
    }
}

#[test]
fn test_optimal_workgroup_with_count_1d_covers_all_elements() {
    // For any total > 0, optimal_workgroup * count must cover all elements.
    for total in [
        1, 2, 3, 7, 15, 16, 17, 31, 32, 33, 63, 64, 100, 255, 256, 500, 1024, 65535,
    ] {
        let wg = optimal_elementwise_workgroup(total, 1024);
        let count = workgroup_count_1d(total, wg);
        let coverage = count * wg;
        assert!(
            coverage >= total,
            "optimal({total})={wg}, count={count}: coverage {coverage} < {total}"
        );
        // Overshoot should be less than one workgroup.
        assert!(
            coverage - total < wg,
            "optimal({total})={wg}, count={count}: overshoot {} >= {wg}",
            coverage - total
        );
    }
}

#[test]
fn test_workgroup_count_2d_validates_within_vulkan_limits() {
    // For dimensions up to 65535*tile_size, the grid should pass validation.
    for (dx, dy, tile) in [
        (16, 16, 16),
        (256, 512, 16),
        (1024, 1024, 16),
        (1023, 1023, 16),
        (8, 8, 8),
        (65535, 65535, 1),
    ] {
        let [gx, gy, gz] = workgroup_count_2d(dx, dy, tile);
        let result = validate_dispatch([gx, gy, gz], [tile, tile, 1], 65535, 1024);
        assert!(
            result.is_ok(),
            "2d({dx},{dy},{tile}) = [{gx},{gy},{gz}] failed: {result:?}"
        );
    }
}

#[test]
fn test_row_reduce_to_dispatch_config_valid() {
    // workgroup_count_row_reduce output should be directly usable in a DispatchConfig.
    let rows = 128;
    let [gx, gy, gz] = workgroup_count_row_reduce(rows);
    let shader = shader_with(2, 8, [256, 1, 1]);
    let config = DispatchConfig {
        grid: [gx, gy, gz],
        bindings: vec![
            BufferBinding {
                binding: 0,
                offset: 0,
                size: 4096,
                read_only: true,
            },
            BufferBinding {
                binding: 1,
                offset: 0,
                size: 4096,
                read_only: false,
            },
        ],
        push_constants: Some({
            let mut pc = PushConstants::new();
            pc.push_u32(512); // row_size
            pc.push_u32(rows);
            pc
        }),
    };
    assert!(shader.validate_dispatch(&config).is_ok());
}

// ===========================================================================
// 2. workgroup_count_1d edge cases
// ===========================================================================

#[test]
fn test_workgroup_count_1d_power_of_two_sizes() {
    // All power-of-two workgroup sizes should yield exact division for power-of-two totals.
    for wg_exp in 0..=8 {
        let wg = 1u32 << wg_exp;
        for total_exp in wg_exp..=20 {
            let total = 1u32 << total_exp;
            let count = workgroup_count_1d(total, wg);
            assert_eq!(
                count,
                total / wg,
                "exact: count_1d({total}, {wg}) expected {} got {count}",
                total / wg
            );
        }
    }
}

#[test]
fn test_workgroup_count_1d_off_by_one_boundary() {
    // Check n-1, n, n+1 around multiples of workgroup_size.
    let wg = 64u32;
    for multiple in 1..=10 {
        let boundary = wg * multiple;
        assert_eq!(workgroup_count_1d(boundary - 1, wg), multiple); // ceil rounds up
        assert_eq!(workgroup_count_1d(boundary, wg), multiple); // exact
        assert_eq!(workgroup_count_1d(boundary + 1, wg), multiple + 1); // one over
    }
}

#[test]
fn test_workgroup_count_1d_non_power_of_two_workgroup() {
    // Odd workgroup sizes (valid in Vulkan, just uncommon).
    assert_eq!(workgroup_count_1d(10, 3), 4); // ceil(10/3) = 4
    assert_eq!(workgroup_count_1d(9, 3), 3); // exact
    assert_eq!(workgroup_count_1d(11, 3), 4); // ceil(11/3) = 4
    assert_eq!(workgroup_count_1d(100, 7), 15); // ceil(100/7) = 15
}

// ===========================================================================
// 3. workgroup_count_2d edge cases
// ===========================================================================

#[test]
fn test_workgroup_count_2d_single_element_dimensions() {
    let [gx, gy, gz] = workgroup_count_2d(1, 1, 16);
    assert_eq!(gx, 1);
    assert_eq!(gy, 1);
    assert_eq!(gz, 1);
}

#[test]
fn test_workgroup_count_2d_non_power_of_two_tile() {
    // tile_size = 5 (not power of 2, but valid).
    let [gx, gy, _] = workgroup_count_2d(12, 7, 5);
    assert_eq!(gx, 3); // ceil(12/5) = 3
    assert_eq!(gy, 2); // ceil(7/5) = 2
}

#[test]
fn test_workgroup_count_2d_tile_size_one() {
    // tile_size = 1 means one workgroup per element.
    let [gx, gy, _] = workgroup_count_2d(100, 200, 1);
    assert_eq!(gx, 100);
    assert_eq!(gy, 200);
}

#[test]
fn test_workgroup_count_2d_tile_larger_than_dims() {
    // When tile > dim, still need 1 workgroup per dimension.
    let [gx, gy, gz] = workgroup_count_2d(3, 5, 16);
    assert_eq!(gx, 1);
    assert_eq!(gy, 1);
    assert_eq!(gz, 1);
}

// ===========================================================================
// 4. validate_dispatch: comprehensive
// ===========================================================================

#[test]
fn test_validate_dispatch_boundary_group_count() {
    // Exactly at max_group_count is OK.
    assert!(validate_dispatch([65535, 65535, 65535], [1, 1, 1], 65535, 128).is_ok());
    // One over in each dimension.
    assert!(validate_dispatch([65536, 1, 1], [1, 1, 1], 65535, 128).is_err());
    assert!(validate_dispatch([1, 65536, 1], [1, 1, 1], 65535, 128).is_err());
    assert!(validate_dispatch([1, 1, 65536], [1, 1, 1], 65535, 128).is_err());
}

#[test]
fn test_validate_dispatch_invocation_product_exact_limit() {
    // 16 * 8 * 8 = 1024 == limit.
    assert!(validate_dispatch([1, 1, 1], [16, 8, 8], 65535, 1024).is_ok());
    // 16 * 8 * 9 = 1152 > 1024.
    assert!(validate_dispatch([1, 1, 1], [16, 8, 9], 65535, 1024).is_err());
}

#[test]
fn test_validate_dispatch_all_zero_local_size() {
    let r = validate_dispatch([1, 1, 1], [0, 0, 0], 65535, 1024);
    assert!(r.is_err());
}

#[test]
fn test_validate_dispatch_mixed_zero_and_nonzero_local() {
    // [256, 0, 1] -> product is 0 -> rejected.
    let r = validate_dispatch([1, 1, 1], [256, 0, 1], 65535, 1024);
    assert!(r.is_err());
}

#[test]
fn test_validate_dispatch_custom_low_max_group_count() {
    // Device with small maxComputeWorkGroupCount.
    assert!(validate_dispatch([100, 1, 1], [64, 1, 1], 100, 1024).is_ok());
    assert!(validate_dispatch([101, 1, 1], [64, 1, 1], 100, 1024).is_err());
}

#[test]
fn test_validate_dispatch_local_size_1_1_1() {
    // Minimal local size is always valid for invocations.
    assert!(validate_dispatch([1, 1, 1], [1, 1, 1], 65535, 1).is_ok());
}

// ===========================================================================
// 5. push_constants: cross-module consistency with PushConstants type
// ===========================================================================

#[test]
fn test_push_constants_1d_matches_push_constants_builder() {
    let raw = push_constants_1d(42);
    let mut pc = PushConstants::new();
    pc.push_u32(42);
    assert_eq!(raw.as_slice(), pc.as_bytes());
}

#[test]
fn test_push_constants_reduction_matches_builder() {
    let raw = push_constants_reduction(512, 32);
    let mut pc = PushConstants::new();
    pc.push_u32(512);
    pc.push_u32(32);
    assert_eq!(raw.as_slice(), pc.as_bytes());
}

#[test]
fn test_push_constants_matmul_matches_builder() {
    let raw = push_constants_matmul(128, 256, 64);
    let mut pc = PushConstants::new();
    pc.push_u32(128);
    pc.push_u32(256);
    pc.push_u32(64);
    assert_eq!(raw.as_slice(), pc.as_bytes());
}

#[test]
fn test_push_constants_matmul_large_dimensions() {
    let m = 4096u32;
    let n = 8192u32;
    let k = 2048u32;
    let bytes = push_constants_matmul(m, n, k);
    assert_eq!(
        u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]),
        m
    );
    assert_eq!(
        u32::from_le_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]),
        n
    );
    assert_eq!(
        u32::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]),
        k
    );
}

#[test]
fn test_push_constants_reduction_max_u32() {
    let bytes = push_constants_reduction(u32::MAX, u32::MAX);
    assert_eq!(
        u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]),
        u32::MAX
    );
    assert_eq!(
        u32::from_le_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]),
        u32::MAX
    );
}

// ===========================================================================
// 6. PushConstants builder: i32 and f32 boundary values
// ===========================================================================

#[test]
fn test_push_constants_f32_special_values() {
    let mut pc = PushConstants::new();
    pc.push_f32(0.0);
    pc.push_f32(-0.0);
    pc.push_f32(f32::INFINITY);
    pc.push_f32(f32::NEG_INFINITY);
    pc.push_f32(f32::NAN);
    assert_eq!(pc.size(), 20);

    let b = pc.as_bytes();
    let v0 = f32::from_le_bytes([b[0], b[1], b[2], b[3]]);
    let v1 = f32::from_le_bytes([b[4], b[5], b[6], b[7]]);
    let v2 = f32::from_le_bytes([b[8], b[9], b[10], b[11]]);
    let v3 = f32::from_le_bytes([b[12], b[13], b[14], b[15]]);
    let v4 = f32::from_le_bytes([b[16], b[17], b[18], b[19]]);

    assert_eq!(v0, 0.0);
    assert!(v1.to_bits() == (-0.0f32).to_bits()); // negative zero
    assert!(v2.is_infinite() && v2 > 0.0);
    assert!(v3.is_infinite() && v3 < 0.0);
    assert!(v4.is_nan());
}

#[test]
fn test_push_constants_i32_boundary_values() {
    let mut pc = PushConstants::new();
    pc.push_i32(i32::MIN);
    pc.push_i32(i32::MAX);
    pc.push_i32(0);
    pc.push_i32(-1);
    assert_eq!(pc.size(), 16);

    let b = pc.as_bytes();
    assert_eq!(i32::from_le_bytes([b[0], b[1], b[2], b[3]]), i32::MIN);
    assert_eq!(i32::from_le_bytes([b[4], b[5], b[6], b[7]]), i32::MAX);
    assert_eq!(i32::from_le_bytes([b[8], b[9], b[10], b[11]]), 0);
    assert_eq!(i32::from_le_bytes([b[12], b[13], b[14], b[15]]), -1);
}

// ===========================================================================
// 7. compute_grid_dims edge cases
// ===========================================================================

#[test]
fn test_compute_grid_dims_various_workgroup_sizes() {
    for wg in [1, 2, 4, 8, 16, 32, 64, 128, 256] {
        let grid = compute_grid_dims(wg, [wg, 1, 1]);
        assert_eq!(grid, [1, 1, 1], "exact fit for wg={wg}");

        if wg > 1 {
            let grid_minus = compute_grid_dims(wg - 1, [wg, 1, 1]);
            assert_eq!(grid_minus, [1, 1, 1], "wg-1 for wg={wg}");
        }

        let grid_plus = compute_grid_dims(wg + 1, [wg, 1, 1]);
        assert_eq!(grid_plus, [2, 1, 1], "wg+1 for wg={wg}");
    }
}

#[test]
fn test_compute_grid_dims_zero_elements_all_workgroup_sizes() {
    for wg in [1, 32, 64, 128, 256] {
        let grid = compute_grid_dims(0, [wg, 1, 1]);
        assert_eq!(grid, [0, 1, 1], "zero elements with wg={wg}");
    }
}

#[test]
fn test_compute_grid_dims_ignores_yz_workgroup() {
    // compute_grid_dims only divides by workgroup_size[0].
    let g1 = compute_grid_dims(1000, [64, 1, 1]);
    let g2 = compute_grid_dims(1000, [64, 8, 4]);
    let g3 = compute_grid_dims(1000, [64, 64, 64]);
    assert_eq!(g1, g2);
    assert_eq!(g2, g3);
}

// ===========================================================================
// 8. spirv_words_to_bytes: integration with CompiledShader
// ===========================================================================

#[test]
fn test_spirv_words_to_bytes_produces_valid_shader_input() {
    let words = vec![SPIRV_MAGIC, SPIRV_VERSION_1_5, 0, 0, 0, 42, 99];
    let bytes = spirv_words_to_bytes(&words);
    assert_eq!(bytes.len(), 28);

    // These bytes should be accepted by CompiledShader::new.
    let shader = CompiledShader::new(bytes, "main", 0, 0, [64, 1, 1]);
    assert!(shader.is_ok());
}

#[test]
fn test_spirv_words_to_bytes_insufficient_for_shader() {
    // Only 4 words = 16 bytes < 20 byte minimum.
    let words = vec![SPIRV_MAGIC, SPIRV_VERSION_1_5, 0, 0];
    let bytes = spirv_words_to_bytes(&words);
    assert_eq!(bytes.len(), 16);
    let result = CompiledShader::new(bytes, "main", 0, 0, [64, 1, 1]);
    assert!(result.is_err());
}

#[test]
fn test_spirv_words_to_bytes_wrong_magic_rejected() {
    let words = vec![0xDEADBEEF, SPIRV_VERSION_1_5, 0, 0, 0];
    let bytes = spirv_words_to_bytes(&words);
    let result = CompiledShader::new(bytes, "main", 0, 0, [64, 1, 1]);
    assert!(result.is_err());
    let err = result.unwrap_err();
    assert!(matches!(err, VulkanPipelineError::SpirvValidation { .. }));
}

// ===========================================================================
// 9. VulkanComputeConfig integration with workgroup utilities
// ===========================================================================

#[test]
fn test_vulkan_config_default_invocations_with_optimal_workgroup() {
    let config = VulkanComputeConfig::default();
    let inv = config.total_workgroup_invocations();
    // For a large tensor, optimal should return DEFAULT_WORKGROUP_SIZE (256).
    let wg = optimal_elementwise_workgroup(1_000_000, inv);
    assert_eq!(wg, 256);
}

#[test]
fn test_vulkan_config_custom_workgroup_with_optimal() {
    let config = VulkanComputeConfig {
        workgroup_size_x: 64,
        workgroup_size_y: 2,
        workgroup_size_z: 1,
        ..Default::default()
    };
    let inv = config.total_workgroup_invocations();
    assert_eq!(inv, 128);
    // With max_invocations=128, optimal should clamp to 128 for large tensors.
    let wg = optimal_elementwise_workgroup(100_000, inv);
    assert_eq!(wg, 128);
}

#[test]
fn test_vulkan_config_1_1_1_workgroup() {
    let config = VulkanComputeConfig {
        workgroup_size_x: 1,
        workgroup_size_y: 1,
        workgroup_size_z: 1,
        ..Default::default()
    };
    assert_eq!(config.total_workgroup_invocations(), 1);
}

// ===========================================================================
// 10. CompiledShader + DispatchConfig: end-to-end validation
// ===========================================================================

#[test]
fn test_shader_validate_dispatch_elementwise_workflow() {
    // Simulate: elementwise kernel on 10000 elements.
    let total = 10_000u32;
    let wg = optimal_elementwise_workgroup(total, 1024);
    let count = workgroup_count_1d(total, wg);
    let grid = [count, 1, 1];

    let shader = shader_with(2, 4, [wg, 1, 1]);

    let mut pc = PushConstants::new();
    pc.push_u32(total);

    let config = DispatchConfig {
        grid,
        bindings: vec![
            BufferBinding {
                binding: 0,
                offset: 0,
                size: u64::from(total) * 4,
                read_only: true,
            },
            BufferBinding {
                binding: 1,
                offset: 0,
                size: u64::from(total) * 4,
                read_only: false,
            },
        ],
        push_constants: Some(pc),
    };

    assert!(shader.validate_dispatch(&config).is_ok());
}

#[test]
fn test_shader_validate_dispatch_matmul_workflow() {
    // Simulate: matmul dispatch M=128, N=256, K=64.
    let m = 128u32;
    let n = 256u32;
    let k = 64u32;
    let tile = 16u32;
    let [gx, gy, gz] = workgroup_count_2d(m, n, tile);

    let shader = shader_with(3, 12, [tile, tile, 1]);

    let mut pc = PushConstants::new();
    pc.push_u32(m);
    pc.push_u32(n);
    pc.push_u32(k);

    let config = DispatchConfig {
        grid: [gx, gy, gz],
        bindings: vec![
            BufferBinding {
                binding: 0,
                offset: 0,
                size: u64::from(m * k) * 4,
                read_only: true,
            },
            BufferBinding {
                binding: 1,
                offset: 0,
                size: u64::from(k * n) * 4,
                read_only: true,
            },
            BufferBinding {
                binding: 2,
                offset: 0,
                size: u64::from(m * n) * 4,
                read_only: false,
            },
        ],
        push_constants: Some(pc),
    };

    assert!(shader.validate_dispatch(&config).is_ok());
}

#[test]
fn test_shader_validate_dispatch_reduction_workflow() {
    // Simulate: row reduction, 64 rows of 512 elements.
    let rows = 64u32;
    let row_size = 512u32;
    let [gx, gy, gz] = workgroup_count_row_reduce(rows);

    let shader = shader_with(2, 8, [256, 1, 1]);

    let mut pc = PushConstants::new();
    pc.push_u32(row_size);
    pc.push_u32(rows);

    let config = DispatchConfig {
        grid: [gx, gy, gz],
        bindings: vec![
            BufferBinding {
                binding: 0,
                offset: 0,
                size: u64::from(rows * row_size) * 4,
                read_only: true,
            },
            BufferBinding {
                binding: 1,
                offset: 0,
                size: u64::from(rows) * 4,
                read_only: false,
            },
        ],
        push_constants: Some(pc),
    };

    assert!(shader.validate_dispatch(&config).is_ok());
}

// ===========================================================================
// 11. Validation error precedence and specificity
// ===========================================================================

#[test]
fn test_shader_validate_dispatch_binding_count_before_grid_check() {
    let shader = shader_with(2, 0, [64, 1, 1]);
    // Wrong binding count AND zero grid -- binding count should be reported first.
    let config = DispatchConfig {
        grid: [0, 0, 0],
        bindings: vec![],
        push_constants: None,
    };
    let err = shader.validate_dispatch(&config).unwrap_err();
    assert!(
        matches!(err, VulkanPipelineError::BindingCountMismatch { .. }),
        "expected BindingCountMismatch, got: {err:?}"
    );
}

#[test]
fn test_shader_validate_dispatch_binding_range_before_push_constant() {
    let shader = shader_with(2, 4, [64, 1, 1]);
    let mut pc = PushConstants::new();
    // 8 bytes > 4 byte limit.
    pc.push_u32(0);
    pc.push_u32(0);

    let config = DispatchConfig {
        grid: [1, 1, 1],
        bindings: vec![
            BufferBinding {
                binding: 0,
                offset: 0,
                size: 64,
                read_only: true,
            },
            BufferBinding {
                binding: 99,
                offset: 0,
                size: 64,
                read_only: false,
            },
        ],
        push_constants: Some(pc),
    };
    let err = shader.validate_dispatch(&config).unwrap_err();
    assert!(
        matches!(err, VulkanPipelineError::BindingOutOfRange { .. }),
        "expected BindingOutOfRange, got: {err:?}"
    );
}

#[test]
fn test_shader_validate_dispatch_push_constant_before_grid() {
    let shader = shader_with(0, 4, [64, 1, 1]);
    let mut pc = PushConstants::new();
    pc.push_u32(0);
    pc.push_u32(0); // 8 bytes > 4

    let config = DispatchConfig {
        grid: [0, 1, 1],
        bindings: vec![],
        push_constants: Some(pc),
    };
    let err = shader.validate_dispatch(&config).unwrap_err();
    assert!(
        matches!(err, VulkanPipelineError::PushConstantOverflow { .. }),
        "expected PushConstantOverflow, got: {err:?}"
    );
}

// ===========================================================================
// 12. optimal_elementwise_workgroup: device-constrained scenarios
// ===========================================================================

#[test]
fn test_optimal_workgroup_minimum_spec_device() {
    // Vulkan spec guarantees maxComputeWorkGroupInvocations >= 128.
    let wg = optimal_elementwise_workgroup(1_000_000, 128);
    assert_eq!(wg, 128);
    assert!(wg.is_power_of_two());
}

#[test]
fn test_optimal_workgroup_very_small_max_invocations() {
    // Hypothetical device with very low max (below spec, but we handle it).
    let wg = optimal_elementwise_workgroup(100, 4);
    assert!(wg <= 4);
    assert!(wg.is_power_of_two());
}

#[test]
fn test_optimal_workgroup_equal_to_default() {
    // When total == DEFAULT_WORKGROUP_SIZE and max >= DEFAULT_WORKGROUP_SIZE.
    let wg = optimal_elementwise_workgroup(256, 1024);
    assert_eq!(wg, 256);
}

#[test]
fn test_optimal_workgroup_just_above_default() {
    // 257 >= 256 (DEFAULT_WORKGROUP_SIZE), should return 256.
    let wg = optimal_elementwise_workgroup(257, 1024);
    assert_eq!(wg, 256);
}

#[test]
fn test_optimal_workgroup_just_below_default() {
    // 255 < 256, should round down to 128.
    let wg = optimal_elementwise_workgroup(255, 1024);
    assert_eq!(wg, 128);
}

// ===========================================================================
// 13. VulkanPipelineError: variant matching
// ===========================================================================

#[test]
fn test_pipeline_error_is_std_error() {
    let err: Box<dyn std::error::Error> = Box::new(VulkanPipelineError::NoDevice);
    assert!(!err.to_string().is_empty());
}

#[test]
fn test_pipeline_error_debug_format_all_variants() {
    let variants: Vec<VulkanPipelineError> = vec![
        VulkanPipelineError::SpirvValidation {
            reason: "test".into(),
        },
        VulkanPipelineError::BindingOutOfRange { index: 3, max: 1 },
        VulkanPipelineError::PushConstantOverflow {
            actual: 64,
            declared: 16,
        },
        VulkanPipelineError::WorkgroupSizeExceeded {
            product: 2048,
            limit: 1024,
        },
        VulkanPipelineError::BufferTooLarge {
            requested: 1 << 30,
            max: 1 << 28,
        },
        VulkanPipelineError::ZeroGridDimension { dim: "y" },
        VulkanPipelineError::NoDevice,
        VulkanPipelineError::BindingCountMismatch {
            required: 3,
            provided: 1,
        },
    ];
    for v in &variants {
        let debug = format!("{v:?}");
        let display = format!("{v}");
        assert!(!debug.is_empty());
        assert!(!display.is_empty());
        // Debug and Display should be different representations.
        assert_ne!(debug, display, "Debug and Display should differ for {v:?}");
    }
}

// ===========================================================================
// 14. DispatchConfig: large grids and many bindings
// ===========================================================================

#[test]
fn test_dispatch_config_many_bindings() {
    let n = 16u32;
    let shader = shader_with(n, 0, [64, 1, 1]);
    let bindings: Vec<BufferBinding> = (0..n)
        .map(|i| BufferBinding {
            binding: i,
            offset: 0,
            size: 1024,
            read_only: i < n / 2,
        })
        .collect();
    let config = DispatchConfig {
        grid: [1, 1, 1],
        bindings,
        push_constants: None,
    };
    assert!(shader.validate_dispatch(&config).is_ok());
}

#[test]
fn test_dispatch_config_no_bindings_no_push_constants() {
    let shader = shader_with(0, 0, [64, 1, 1]);
    let config = DispatchConfig {
        grid: [1, 1, 1],
        bindings: vec![],
        push_constants: None,
    };
    assert!(shader.validate_dispatch(&config).is_ok());
}

// ===========================================================================
// 15. CompiledShader accessors
// ===========================================================================

#[test]
fn test_compiled_shader_accessors_comprehensive() {
    let spirv = spirv_words_to_bytes(&[SPIRV_MAGIC, SPIRV_VERSION_1_5, 0, 0, 0, 1, 2, 3]);
    let shader = CompiledShader::new(spirv.clone(), "compute", 5, 32, [16, 8, 4]).unwrap();
    assert_eq!(shader.spirv(), spirv.as_slice());
    assert_eq!(shader.entry_point(), "compute");
    assert_eq!(shader.num_bindings(), 5);
    assert_eq!(shader.push_constant_size(), 32);
    assert_eq!(shader.workgroup_size(), [16, 8, 4]);
}

#[test]
fn test_compiled_shader_clone_independence() {
    let shader = shader_with(2, 8, [64, 1, 1]);
    let cloned = shader.clone();
    // Clone should have identical field values.
    assert_eq!(shader.entry_point(), cloned.entry_point());
    assert_eq!(shader.num_bindings(), cloned.num_bindings());
    assert_eq!(shader.push_constant_size(), cloned.push_constant_size());
    assert_eq!(shader.workgroup_size(), cloned.workgroup_size());
    assert_eq!(shader.spirv(), cloned.spirv());
}

// ===========================================================================
// 16. BufferBinding: field coverage
// ===========================================================================

#[test]
fn test_buffer_binding_zero_offset_and_size() {
    let bb = BufferBinding {
        binding: 0,
        offset: 0,
        size: 0,
        read_only: true,
    };
    assert_eq!(bb.binding, 0);
    assert_eq!(bb.offset, 0);
    assert_eq!(bb.size, 0);
    assert!(bb.read_only);
}

#[test]
fn test_buffer_binding_large_u64_values() {
    let bb = BufferBinding {
        binding: u32::MAX,
        offset: u64::MAX,
        size: u64::MAX,
        read_only: false,
    };
    assert_eq!(bb.binding, u32::MAX);
    assert_eq!(bb.offset, u64::MAX);
    assert_eq!(bb.size, u64::MAX);
}

// ===========================================================================
// 17. Ceiling division property: count * wg >= total for all non-zero totals
// ===========================================================================

#[test]
fn test_ceiling_division_property_sweep() {
    // Sweep a range of totals and workgroup sizes.
    for wg in [1, 2, 3, 4, 7, 8, 15, 16, 31, 32, 63, 64, 128, 256] {
        for total in 0..=wg * 4 {
            let count = workgroup_count_1d(total, wg);
            if total == 0 {
                assert_eq!(count, 0, "zero total should give zero count (wg={wg})");
            } else {
                assert!(
                    count * wg >= total,
                    "ceil property: {count}*{wg}={} < {total}",
                    count * wg
                );
                // Should not overshoot by more than wg-1.
                assert!(
                    count * wg - total < wg,
                    "overshoot: {count}*{wg}-{total}={} >= {wg}",
                    count * wg - total
                );
            }
        }
    }
}

// ===========================================================================
// 18. workgroup_count_row_reduce: zero rows and large values
// ===========================================================================

#[test]
fn test_workgroup_count_row_reduce_zero() {
    let [gx, gy, gz] = workgroup_count_row_reduce(0);
    assert_eq!(gx, 0);
    assert_eq!(gy, 1);
    assert_eq!(gz, 1);
}

#[test]
fn test_workgroup_count_row_reduce_max_u32() {
    let [gx, gy, gz] = workgroup_count_row_reduce(u32::MAX);
    assert_eq!(gx, u32::MAX);
    assert_eq!(gy, 1);
    assert_eq!(gz, 1);
}

// ===========================================================================
// 19. validate_dispatch + workgroup_count: when grid exceeds device limits
// ===========================================================================

#[test]
fn test_large_1d_dispatch_exceeds_device_limit() {
    // A tensor large enough that workgroup_count_1d exceeds 65535.
    let total = 256 * 65536; // 16,777,216 elements / 256 = 65536 groups > 65535.
    let count = workgroup_count_1d(total, 256);
    assert_eq!(count, 65536);
    let result = validate_dispatch([count, 1, 1], [256, 1, 1], 65535, 1024);
    assert!(result.is_err());
    assert!(result.unwrap_err().contains("exceeds device limit"));
}

#[test]
fn test_large_2d_dispatch_within_device_limit() {
    // 1024x1024 grid with tile=16 -> 64x64 groups, well within 65535.
    let [gx, gy, gz] = workgroup_count_2d(1024, 1024, 16);
    assert_eq!(gx, 64);
    assert_eq!(gy, 64);
    assert!(validate_dispatch([gx, gy, gz], [16, 16, 1], 65535, 1024).is_ok());
}

// ===========================================================================
// 20. VulkanComputeConfig: enable_validation follows debug_assertions
// ===========================================================================

#[test]
fn test_vulkan_config_validation_flag_in_debug() {
    let config = VulkanComputeConfig::default();
    // In debug builds (test mode), enable_validation should be true.
    assert_eq!(config.enable_validation, cfg!(debug_assertions));
}

// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Comprehensive tests for workgroup size calculation and dispatch validation.

use super::*;

// ============================================================
// workgroup_count_1d: ceiling division
// ============================================================

#[test]
fn count_1d_exact_multiples() {
    assert_eq!(workgroup_count_1d(256, 256), 1);
    assert_eq!(workgroup_count_1d(512, 256), 2);
    assert_eq!(workgroup_count_1d(768, 256), 3);
    assert_eq!(workgroup_count_1d(1024, 256), 4);
    assert_eq!(workgroup_count_1d(64, 64), 1);
}

#[test]
fn count_1d_non_multiples_round_up() {
    assert_eq!(workgroup_count_1d(1, 256), 1);
    assert_eq!(workgroup_count_1d(255, 256), 1);
    assert_eq!(workgroup_count_1d(257, 256), 2);
    assert_eq!(workgroup_count_1d(511, 256), 2);
    assert_eq!(workgroup_count_1d(513, 256), 3);
}

#[test]
fn count_1d_single_element() {
    assert_eq!(workgroup_count_1d(1, 1), 1);
    assert_eq!(workgroup_count_1d(1, 128), 1);
    assert_eq!(workgroup_count_1d(1, 256), 1);
    assert_eq!(workgroup_count_1d(1, 1024), 1);
}

#[test]
fn count_1d_zero_elements_returns_zero() {
    // 0 elements needs 0 workgroups — no work to dispatch.
    assert_eq!(workgroup_count_1d(0, 256), 0);
    assert_eq!(workgroup_count_1d(0, 1), 0);
}

#[test]
fn count_1d_very_large_elements() {
    // > 2^20 = 1_048_576
    let large = 1 << 20;
    assert_eq!(workgroup_count_1d(large, 256), large / 256);
    assert_eq!(workgroup_count_1d(large + 1, 256), large / 256 + 1);

    // > 2^24 = 16_777_216
    let very_large = 1 << 24;
    assert_eq!(workgroup_count_1d(very_large, 256), very_large / 256);
}

#[test]
fn count_1d_workgroup_size_one() {
    assert_eq!(workgroup_count_1d(0, 1), 0);
    assert_eq!(workgroup_count_1d(1, 1), 1);
    assert_eq!(workgroup_count_1d(100, 1), 100);
    assert_eq!(workgroup_count_1d(1_000_000, 1), 1_000_000);
}

#[test]
fn count_1d_never_returns_zero_for_nonzero_elements() {
    for n in 1..=1024 {
        let count = workgroup_count_1d(n, 256);
        assert!(count > 0, "workgroup_count_1d({n}, 256) returned 0");
    }
}

#[test]
#[should_panic(expected = "workgroup_size must be > 0")]
fn count_1d_panics_on_zero_workgroup_size() {
    let _ = workgroup_count_1d(100, 0);
}

// ============================================================
// workgroup_count_2d: 2D grid computation
// ============================================================

#[test]
fn count_2d_exact_tiles() {
    assert_eq!(workgroup_count_2d(16, 16, 16), [1, 1, 1]);
    assert_eq!(workgroup_count_2d(32, 32, 16), [2, 2, 1]);
    assert_eq!(workgroup_count_2d(64, 128, 16), [4, 8, 1]);
}

#[test]
fn count_2d_non_multiples_round_up() {
    assert_eq!(workgroup_count_2d(17, 33, 16), [2, 3, 1]);
    assert_eq!(workgroup_count_2d(1, 1, 16), [1, 1, 1]);
    assert_eq!(workgroup_count_2d(15, 31, 16), [1, 2, 1]);
}

#[test]
fn count_2d_asymmetric_dimensions() {
    assert_eq!(workgroup_count_2d(512, 1, 16), [32, 1, 1]);
    assert_eq!(workgroup_count_2d(1, 512, 16), [1, 32, 1]);
}

#[test]
fn count_2d_z_always_one() {
    // z dimension is always 1 for 2D tiled dispatch.
    for dim_x in [1, 16, 17, 256, 1024] {
        for dim_y in [1, 16, 17, 256, 1024] {
            let [_, _, gz] = workgroup_count_2d(dim_x, dim_y, 16);
            assert_eq!(gz, 1, "z should be 1 for count_2d({dim_x}, {dim_y}, 16)");
        }
    }
}

#[test]
fn count_2d_large_dimensions() {
    let [gx, gy, gz] = workgroup_count_2d(1 << 20, 1 << 20, 16);
    assert_eq!(gx, 1 << 16);
    assert_eq!(gy, 1 << 16);
    assert_eq!(gz, 1);
}

#[test]
#[should_panic(expected = "tile_size must be > 0")]
fn count_2d_panics_on_zero_tile() {
    let _ = workgroup_count_2d(16, 16, 0);
}

// ============================================================
// workgroup_count_row_reduce: reduction operations
// ============================================================

#[test]
fn count_row_reduce_basic() {
    assert_eq!(workgroup_count_row_reduce(1), [1, 1, 1]);
    assert_eq!(workgroup_count_row_reduce(32), [32, 1, 1]);
    assert_eq!(workgroup_count_row_reduce(1024), [1024, 1, 1]);
}

#[test]
fn count_row_reduce_zero_rows() {
    // Zero rows produces [0, 1, 1] — no workgroups dispatched.
    assert_eq!(workgroup_count_row_reduce(0), [0, 1, 1]);
}

#[test]
fn count_row_reduce_large() {
    let large = 1 << 20;
    let [gx, gy, gz] = workgroup_count_row_reduce(large);
    assert_eq!(gx, large);
    assert_eq!(gy, 1);
    assert_eq!(gz, 1);
}

#[test]
fn count_row_reduce_yz_always_one() {
    for rows in [0, 1, 7, 64, 255, 1024, 65535] {
        let [_, gy, gz] = workgroup_count_row_reduce(rows);
        assert_eq!(gy, 1);
        assert_eq!(gz, 1);
    }
}

// ============================================================
// optimal_elementwise_workgroup: valid workgroup sizes
// ============================================================

#[test]
fn optimal_returns_default_for_large_tensors() {
    // DEFAULT_WORKGROUP_SIZE is 256 and max_invocations is 1024.
    assert_eq!(optimal_elementwise_workgroup(100_000, 1024), 256);
    assert_eq!(optimal_elementwise_workgroup(1_000_000, 1024), 256);
    assert_eq!(optimal_elementwise_workgroup(1 << 24, 1024), 256);
}

#[test]
fn optimal_returns_power_of_two() {
    for n in 0..=512 {
        let wg = optimal_elementwise_workgroup(n, 1024);
        assert!(
            wg.is_power_of_two(),
            "optimal({n}, 1024) = {wg} is not power of 2"
        );
    }
}

#[test]
fn optimal_never_exceeds_max_invocations() {
    for max_inv in [64, 128, 256, 512, 1024] {
        for n in [1, 32, 64, 128, 256, 512, 1024, 100_000] {
            let wg = optimal_elementwise_workgroup(n, max_inv);
            assert!(
                wg <= max_inv,
                "optimal({n}, {max_inv}) = {wg} exceeds max_invocations"
            );
        }
    }
}

#[test]
fn optimal_zero_elements_returns_one() {
    assert_eq!(optimal_elementwise_workgroup(0, 1024), 1);
    assert_eq!(optimal_elementwise_workgroup(0, 128), 1);
}

#[test]
fn optimal_single_element_returns_one() {
    assert_eq!(optimal_elementwise_workgroup(1, 1024), 1);
}

#[test]
fn optimal_small_tensors_round_down_to_power_of_two() {
    assert_eq!(optimal_elementwise_workgroup(3, 1024), 2);
    assert_eq!(optimal_elementwise_workgroup(5, 1024), 4);
    assert_eq!(optimal_elementwise_workgroup(33, 1024), 32);
    assert_eq!(optimal_elementwise_workgroup(63, 1024), 32);
    assert_eq!(optimal_elementwise_workgroup(65, 1024), 64);
    assert_eq!(optimal_elementwise_workgroup(127, 1024), 64);
    assert_eq!(optimal_elementwise_workgroup(255, 1024), 128);
}

#[test]
fn optimal_exact_powers_of_two() {
    assert_eq!(optimal_elementwise_workgroup(2, 1024), 2);
    assert_eq!(optimal_elementwise_workgroup(4, 1024), 4);
    assert_eq!(optimal_elementwise_workgroup(8, 1024), 8);
    assert_eq!(optimal_elementwise_workgroup(16, 1024), 16);
    assert_eq!(optimal_elementwise_workgroup(32, 1024), 32);
    assert_eq!(optimal_elementwise_workgroup(64, 1024), 64);
    assert_eq!(optimal_elementwise_workgroup(128, 1024), 128);
    assert_eq!(optimal_elementwise_workgroup(256, 1024), 256);
}

#[test]
fn optimal_clamped_by_low_max_invocations() {
    // Device that only supports 128 invocations.
    assert_eq!(optimal_elementwise_workgroup(100_000, 128), 128);
    assert_eq!(optimal_elementwise_workgroup(256, 128), 128);
    assert_eq!(optimal_elementwise_workgroup(64, 128), 64);
}

#[test]
fn optimal_result_is_valid_workgroup_size() {
    // Result must be >= 1 and <= DEFAULT_WORKGROUP_SIZE.
    for n in 0..=2048 {
        let wg = optimal_elementwise_workgroup(n, 1024);
        assert!(wg >= 1, "workgroup size must be >= 1, got {wg} for n={n}");
        assert!(
            wg <= 256,
            "workgroup size must be <= 256, got {wg} for n={n}"
        );
    }
}

// ============================================================
// push_constants_1d: byte layout
// ============================================================

#[test]
fn push_constants_1d_correct_byte_layout() {
    let bytes = push_constants_1d(1024);
    assert_eq!(bytes.len(), 4);
    assert_eq!(u32::from_le_bytes(bytes), 1024);
}

#[test]
fn push_constants_1d_zero() {
    let bytes = push_constants_1d(0);
    assert_eq!(u32::from_le_bytes(bytes), 0);
}

#[test]
fn push_constants_1d_max_u32() {
    let bytes = push_constants_1d(u32::MAX);
    assert_eq!(u32::from_le_bytes(bytes), u32::MAX);
}

#[test]
fn push_constants_1d_roundtrip_various_values() {
    for val in [0, 1, 255, 256, 65535, 1 << 20, u32::MAX] {
        let bytes = push_constants_1d(val);
        assert_eq!(u32::from_le_bytes(bytes), val, "roundtrip failed for {val}");
    }
}

// ============================================================
// push_constants_matmul: M, N, K properly encoded
// ============================================================

#[test]
fn push_constants_matmul_byte_layout() {
    let bytes = push_constants_matmul(128, 256, 64);
    assert_eq!(bytes.len(), 12);
    let m = u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
    let n = u32::from_le_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]);
    let k = u32::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]);
    assert_eq!(m, 128);
    assert_eq!(n, 256);
    assert_eq!(k, 64);
}

#[test]
fn push_constants_matmul_zero_dimensions() {
    let bytes = push_constants_matmul(0, 0, 0);
    let m = u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
    let n = u32::from_le_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]);
    let k = u32::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]);
    assert_eq!(m, 0);
    assert_eq!(n, 0);
    assert_eq!(k, 0);
}

#[test]
fn push_constants_matmul_large_values() {
    let big = 1 << 20;
    let bytes = push_constants_matmul(big, big + 1, big + 2);
    let m = u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
    let n = u32::from_le_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]);
    let k = u32::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]);
    assert_eq!(m, big);
    assert_eq!(n, big + 1);
    assert_eq!(k, big + 2);
}

#[test]
fn push_constants_matmul_fields_are_independent() {
    // Ensure M, N, K don't bleed into each other.
    let bytes = push_constants_matmul(1, 0, 0);
    assert_eq!(
        u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]),
        1
    );
    assert_eq!(
        u32::from_le_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]),
        0
    );
    assert_eq!(
        u32::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]),
        0
    );

    let bytes = push_constants_matmul(0, 1, 0);
    assert_eq!(
        u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]),
        0
    );
    assert_eq!(
        u32::from_le_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]),
        1
    );
    assert_eq!(
        u32::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]),
        0
    );

    let bytes = push_constants_matmul(0, 0, 1);
    assert_eq!(
        u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]),
        0
    );
    assert_eq!(
        u32::from_le_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]),
        0
    );
    assert_eq!(
        u32::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]),
        1
    );
}

// ============================================================
// push_constants_reduction: row count and size properly encoded
// ============================================================

#[test]
fn push_constants_reduction_byte_layout() {
    let bytes = push_constants_reduction(512, 32);
    assert_eq!(bytes.len(), 8);
    let row_size = u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
    let num_rows = u32::from_le_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]);
    assert_eq!(row_size, 512);
    assert_eq!(num_rows, 32);
}

#[test]
fn push_constants_reduction_zero() {
    let bytes = push_constants_reduction(0, 0);
    assert_eq!(
        u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]),
        0
    );
    assert_eq!(
        u32::from_le_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]),
        0
    );
}

#[test]
fn push_constants_reduction_fields_are_independent() {
    let bytes = push_constants_reduction(1, 0);
    assert_eq!(
        u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]),
        1
    );
    assert_eq!(
        u32::from_le_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]),
        0
    );

    let bytes = push_constants_reduction(0, 1);
    assert_eq!(
        u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]),
        0
    );
    assert_eq!(
        u32::from_le_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]),
        1
    );
}

#[test]
fn push_constants_reduction_roundtrip_various() {
    for (row_size, num_rows) in [
        (1, 1),
        (256, 64),
        (1024, 1),
        (1, 65535),
        (u32::MAX, u32::MAX),
    ] {
        let bytes = push_constants_reduction(row_size, num_rows);
        let rs = u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
        let nr = u32::from_le_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]);
        assert_eq!(
            rs, row_size,
            "row_size roundtrip failed for ({row_size}, {num_rows})"
        );
        assert_eq!(
            nr, num_rows,
            "num_rows roundtrip failed for ({row_size}, {num_rows})"
        );
    }
}

// ============================================================
// validate_dispatch: accepts valid, rejects invalid
// ============================================================

#[test]
fn validate_dispatch_accepts_valid_configs() {
    assert!(validate_dispatch([1, 1, 1], [1, 1, 1], 65535, 128).is_ok());
    assert!(validate_dispatch([256, 1, 1], [256, 1, 1], 65535, 1024).is_ok());
    assert!(validate_dispatch([65535, 65535, 65535], [1, 1, 1], 65535, 128).is_ok());
    assert!(validate_dispatch([100, 200, 300], [4, 4, 4], 65535, 1024).is_ok());
}

#[test]
fn validate_dispatch_rejects_zero_group_count_x() {
    let r = validate_dispatch([0, 1, 1], [256, 1, 1], 65535, 1024);
    assert!(r.is_err());
    assert!(r.unwrap_err().contains("count[0]"));
}

#[test]
fn validate_dispatch_rejects_zero_group_count_y() {
    let r = validate_dispatch([1, 0, 1], [256, 1, 1], 65535, 1024);
    assert!(r.is_err());
    assert!(r.unwrap_err().contains("count[1]"));
}

#[test]
fn validate_dispatch_rejects_zero_group_count_z() {
    let r = validate_dispatch([1, 1, 0], [256, 1, 1], 65535, 1024);
    assert!(r.is_err());
    assert!(r.unwrap_err().contains("count[2]"));
}

#[test]
fn validate_dispatch_rejects_exceeding_group_count_each_dim() {
    let max = 65535;
    // x exceeds
    let r = validate_dispatch([max + 1, 1, 1], [1, 1, 1], max, 1024);
    assert!(r.is_err());
    assert!(r.unwrap_err().contains("count[0]"));

    // y exceeds
    let r = validate_dispatch([1, max + 1, 1], [1, 1, 1], max, 1024);
    assert!(r.is_err());
    assert!(r.unwrap_err().contains("count[1]"));

    // z exceeds
    let r = validate_dispatch([1, 1, max + 1], [1, 1, 1], max, 1024);
    assert!(r.is_err());
    assert!(r.unwrap_err().contains("count[2]"));
}

#[test]
fn validate_dispatch_rejects_invocations_exceeding_limit() {
    // 512 * 512 * 1 = 262144 > 1024
    let r = validate_dispatch([1, 1, 1], [512, 512, 1], 65535, 1024);
    assert!(r.is_err());
    assert!(r.unwrap_err().contains("exceeds device limit"));
}

#[test]
fn validate_dispatch_rejects_zero_local_size() {
    let r = validate_dispatch([1, 1, 1], [0, 1, 1], 65535, 1024);
    assert!(r.is_err());
    assert!(r.unwrap_err().contains("non-zero"));
}

#[test]
fn validate_dispatch_rejects_local_size_overflow() {
    // u32::MAX * u32::MAX overflows u32.
    let r = validate_dispatch([1, 1, 1], [u32::MAX, u32::MAX, 2], 65535, u32::MAX);
    assert!(r.is_err());
    assert!(r.unwrap_err().contains("overflow"));
}

#[test]
fn validate_dispatch_at_exact_group_count_limit() {
    // Exactly at max_group_count should be accepted.
    assert!(validate_dispatch([65535, 65535, 65535], [1, 1, 1], 65535, 128).is_ok());
}

#[test]
fn validate_dispatch_at_exact_invocation_limit() {
    // Exactly at max_invocations should be accepted.
    assert!(validate_dispatch([1, 1, 1], [1024, 1, 1], 65535, 1024).is_ok());
    assert!(validate_dispatch([1, 1, 1], [32, 32, 1], 65535, 1024).is_ok());
    // 32 * 32 * 1 = 1024 == max
    assert!(validate_dispatch([1, 1, 1], [8, 8, 16], 65535, 1024).is_ok());
    // 8 * 8 * 16 = 1024 == max
}

#[test]
fn validate_dispatch_one_over_invocation_limit() {
    // 1025 > 1024
    let r = validate_dispatch([1, 1, 1], [1025, 1, 1], 65535, 1024);
    assert!(r.is_err());
}

// ============================================================
// Compositional: workgroup_count + validate_dispatch integration
// ============================================================

#[test]
fn count_1d_result_passes_validation() {
    // For reasonable element counts, the 1D workgroup count should be within
    // the Vulkan spec limit of 65535.
    for n in [1, 256, 1000, 10_000, 256 * 65535] {
        let count = workgroup_count_1d(n, 256);
        let r = validate_dispatch([count, 1, 1], [256, 1, 1], 65535, 1024);
        assert!(
            r.is_ok(),
            "workgroup_count_1d({n}, 256)={count} failed validation: {r:?}"
        );
    }
}

#[test]
fn count_2d_result_passes_validation() {
    for (dx, dy) in [(16, 16), (256, 128), (1024, 1024)] {
        let [gx, gy, gz] = workgroup_count_2d(dx, dy, 16);
        let r = validate_dispatch([gx, gy, gz], [16, 16, 1], 65535, 1024);
        assert!(
            r.is_ok(),
            "workgroup_count_2d({dx}, {dy}, 16)=[{gx},{gy},{gz}] failed validation: {r:?}"
        );
    }
}

#[test]
fn row_reduce_result_passes_validation() {
    for rows in [1, 32, 1024, 65535] {
        let [gx, gy, gz] = workgroup_count_row_reduce(rows);
        let r = validate_dispatch([gx, gy, gz], [256, 1, 1], 65535, 1024);
        assert!(
            r.is_ok(),
            "workgroup_count_row_reduce({rows})=[{gx},{gy},{gz}] failed validation: {r:?}"
        );
    }
}

// ============================================================
// Edge case: count_1d with elements exactly equal to workgroup size
// ============================================================

#[test]
fn count_1d_exactly_one_workgroup() {
    for wg_size in [1, 2, 4, 8, 16, 32, 64, 128, 256] {
        assert_eq!(
            workgroup_count_1d(wg_size, wg_size),
            1,
            "exactly {wg_size} elements with wg_size={wg_size} should need 1 workgroup"
        );
    }
}

// ============================================================
// Edge case: very large total_elements (> 2^20)
// ============================================================

#[test]
fn count_1d_large_element_counts() {
    // 2^20 = 1_048_576
    assert_eq!(workgroup_count_1d(1_048_576, 256), 4096);
    // 2^20 + 1
    assert_eq!(workgroup_count_1d(1_048_577, 256), 4097);
    // 2^24 = 16_777_216
    assert_eq!(workgroup_count_1d(16_777_216, 256), 65536);
    // 2^30
    assert_eq!(workgroup_count_1d(1 << 30, 256), (1 << 30) / 256);
}

// ============================================================
// Property: workgroup_count_1d never returns 0 for non-zero elements
// ============================================================

#[test]
fn count_1d_nonzero_elements_always_positive_count() {
    // Sweep small values
    for n in 1..=512 {
        let count = workgroup_count_1d(n, 256);
        assert!(count > 0, "workgroup_count_1d({n}, 256) must be > 0");
    }
    // Spot-check large values
    for n in [1 << 16, 1 << 20, 1 << 24, 1 << 30] {
        let count = workgroup_count_1d(n, 256);
        assert!(count > 0, "workgroup_count_1d({n}, 256) must be > 0");
    }
}

// ============================================================
// Property: optimal workgroup size combined with count_1d covers all elements
// ============================================================

#[test]
fn optimal_workgroup_covers_all_elements() {
    for n in [
        0, 1, 2, 3, 31, 32, 33, 63, 64, 100, 255, 256, 257, 1000, 100_000,
    ] {
        let wg = optimal_elementwise_workgroup(n, 1024);
        if n == 0 {
            // No elements to cover; wg=1, count=0 dispatches.
            assert_eq!(workgroup_count_1d(n, wg), 0);
        } else {
            let count = workgroup_count_1d(n, wg);
            // count * wg must be >= n (covers all elements).
            assert!(
                count * wg >= n,
                "optimal({n}, 1024)={wg}, count={count}: {count}*{wg}={} < {n}",
                count * wg,
            );
        }
    }
}

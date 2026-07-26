// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Workgroup size calculation utilities for Vulkan compute dispatch.
//!
//! Provides helpers to compute optimal workgroup counts and local sizes
//! for common dispatch patterns: 1D elementwise, 2D tiled matmul, and
//! row-per-workgroup reductions.
//!
//! # Vulkan spec constraints
//!
//! - `maxComputeWorkGroupSize[0..3]`: per-dimension limit (guaranteed >= 128).
//! - `maxComputeWorkGroupInvocations`: total threads per workgroup (guaranteed >= 128).
//! - `maxComputeWorkGroupCount[0..3]`: dispatch grid limit (guaranteed >= 65535).

use crate::spirv_emit::DEFAULT_WORKGROUP_SIZE;

/// Compute the number of workgroups needed for a 1D elementwise dispatch.
///
/// Returns `ceil(total_elements / workgroup_size)`.
///
/// # Panics
///
/// Panics if `workgroup_size` is 0.
///
/// # Example
///
/// ```
/// use nn_vulkan::workgroup::workgroup_count_1d;
/// assert_eq!(workgroup_count_1d(1000, 256), 4); // ceil(1000/256) = 4
/// assert_eq!(workgroup_count_1d(256, 256), 1);  // exact multiple
/// assert_eq!(workgroup_count_1d(1, 256), 1);    // single element
/// ```
#[must_use]
pub fn workgroup_count_1d(total_elements: u32, workgroup_size: u32) -> u32 {
    assert!(workgroup_size > 0, "workgroup_size must be > 0");
    total_elements.div_ceil(workgroup_size)
}

/// Compute workgroup counts for a 2D tiled dispatch (e.g., matmul).
///
/// Returns `[ceil(dim_x / tile_size), ceil(dim_y / tile_size), 1]`.
///
/// # Panics
///
/// Panics if `tile_size` is 0.
///
/// # Example
///
/// ```
/// use nn_vulkan::workgroup::workgroup_count_2d;
/// let [gx, gy, gz] = workgroup_count_2d(512, 256, 16);
/// assert_eq!(gx, 32); // ceil(512/16)
/// assert_eq!(gy, 16); // ceil(256/16)
/// assert_eq!(gz, 1);
/// ```
#[must_use]
pub fn workgroup_count_2d(dim_x: u32, dim_y: u32, tile_size: u32) -> [u32; 3] {
    assert!(tile_size > 0, "tile_size must be > 0");
    [
        dim_x.div_ceil(tile_size),
        dim_y.div_ceil(tile_size),
        1,
    ]
}

/// Compute workgroup counts for row-per-workgroup reductions.
///
/// Each workgroup processes one row. Returns `[num_rows, 1, 1]`.
///
/// # Example
///
/// ```
/// use nn_vulkan::workgroup::workgroup_count_row_reduce;
/// let groups = workgroup_count_row_reduce(32);
/// assert_eq!(groups, [32, 1, 1]); // one workgroup per row
/// ```
#[must_use]
pub fn workgroup_count_row_reduce(num_rows: u32) -> [u32; 3] {
    [num_rows, 1, 1]
}

/// Choose the optimal workgroup size for elementwise operations.
///
/// Considers the device's `max_workgroup_invocations` limit and the total
/// number of elements. For small tensors, reduces the workgroup size to
/// avoid wasting threads.
///
/// Returns a power-of-2 workgroup size.
///
/// # Example
///
/// ```
/// use nn_vulkan::workgroup::optimal_elementwise_workgroup;
/// // Large tensor: use default 256.
/// assert_eq!(optimal_elementwise_workgroup(100_000, 1024), 256);
/// // Small tensor: reduce to avoid waste.
/// assert_eq!(optimal_elementwise_workgroup(32, 1024), 32);
/// // Very small tensor.
/// assert_eq!(optimal_elementwise_workgroup(1, 1024), 1);
/// ```
#[must_use]
pub fn optimal_elementwise_workgroup(total_elements: u32, max_invocations: u32) -> u32 {
    let target = DEFAULT_WORKGROUP_SIZE.min(max_invocations);

    if total_elements >= target {
        return target;
    }

    // Round down to nearest power of 2.
    if total_elements == 0 {
        return 1;
    }
    1 << total_elements.ilog2()
}

/// Validate that a workgroup dispatch configuration is within Vulkan spec limits.
///
/// Returns `Ok(())` if valid, or a description of the violation.
///
/// # Arguments
///
/// * `group_count` -- Dispatch workgroup counts `[x, y, z]`.
/// * `local_size` -- Workgroup local sizes `[x, y, z]`.
/// * `max_group_count` -- Device limit for `maxComputeWorkGroupCount` (usually 65535).
/// * `max_invocations` -- Device limit for `maxComputeWorkGroupInvocations`.
///
/// # Example
///
/// ```
/// use nn_vulkan::workgroup::validate_dispatch;
/// assert!(validate_dispatch([256, 1, 1], [256, 1, 1], 65535, 1024).is_ok());
/// assert!(validate_dispatch([70000, 1, 1], [256, 1, 1], 65535, 1024).is_err());
/// ```
pub fn validate_dispatch(
    group_count: [u32; 3],
    local_size: [u32; 3],
    max_group_count: u32,
    max_invocations: u32,
) -> Result<(), String> {
    // Check group count limits.
    for (i, &count) in group_count.iter().enumerate() {
        if count > max_group_count {
            return Err(format!(
                "workgroup count[{i}] = {count} exceeds device limit {max_group_count}"
            ));
        }
        if count == 0 {
            return Err(format!("workgroup count[{i}] must be > 0"));
        }
    }

    // Check total invocations per workgroup.
    let total_invocations = local_size[0]
        .checked_mul(local_size[1])
        .and_then(|v| v.checked_mul(local_size[2]));

    match total_invocations {
        Some(total) if total > max_invocations => {
            Err(format!(
                "total invocations per workgroup ({total} = {}x{}x{}) exceeds device limit {max_invocations}",
                local_size[0], local_size[1], local_size[2]
            ))
        }
        Some(0) => Err("local size must have non-zero total invocations".into()),
        None => Err("local size overflow in multiplication".into()),
        _ => Ok(()),
    }
}

/// Build push constant bytes for a 1D elementwise dispatch.
///
/// Layout: `[total_elements: u32]` (4 bytes).
///
/// # Example
///
/// ```
/// use nn_vulkan::workgroup::push_constants_1d;
/// let bytes = push_constants_1d(1024);
/// assert_eq!(bytes.len(), 4);
/// assert_eq!(u32::from_le_bytes(bytes), 1024);
/// ```
#[must_use]
pub fn push_constants_1d(total_elements: u32) -> [u8; 4] {
    total_elements.to_le_bytes()
}

/// Build push constant bytes for a reduction dispatch.
///
/// Layout: `[row_size: u32, num_rows: u32]` (8 bytes).
///
/// # Example
///
/// ```
/// use nn_vulkan::workgroup::push_constants_reduction;
/// let bytes = push_constants_reduction(512, 32);
/// assert_eq!(bytes.len(), 8);
/// ```
#[must_use]
pub fn push_constants_reduction(row_size: u32, num_rows: u32) -> [u8; 8] {
    let mut bytes = [0u8; 8];
    bytes[0..4].copy_from_slice(&row_size.to_le_bytes());
    bytes[4..8].copy_from_slice(&num_rows.to_le_bytes());
    bytes
}

/// Build push constant bytes for a matmul dispatch.
///
/// Layout: `[M: u32, N: u32, K: u32]` (12 bytes).
///
/// # Example
///
/// ```
/// use nn_vulkan::workgroup::push_constants_matmul;
/// let bytes = push_constants_matmul(128, 256, 64);
/// assert_eq!(bytes.len(), 12);
/// ```
#[must_use]
pub fn push_constants_matmul(m: u32, n: u32, k: u32) -> [u8; 12] {
    let mut bytes = [0u8; 12];
    bytes[0..4].copy_from_slice(&m.to_le_bytes());
    bytes[4..8].copy_from_slice(&n.to_le_bytes());
    bytes[8..12].copy_from_slice(&k.to_le_bytes());
    bytes
}

#[cfg(test)]
#[path = "workgroup_tests.rs"]
mod workgroup_tests;

#[cfg(test)]
mod tests {
    use super::*;

    // ---- workgroup_count_1d ----

    #[test]
    fn test_count_1d_exact_multiple() {
        assert_eq!(workgroup_count_1d(256, 256), 1);
        assert_eq!(workgroup_count_1d(512, 256), 2);
        assert_eq!(workgroup_count_1d(1024, 256), 4);
    }

    #[test]
    fn test_count_1d_non_multiple() {
        assert_eq!(workgroup_count_1d(257, 256), 2);
        assert_eq!(workgroup_count_1d(1, 256), 1);
        assert_eq!(workgroup_count_1d(255, 256), 1);
    }

    #[test]
    fn test_count_1d_large() {
        assert_eq!(workgroup_count_1d(1_000_000, 256), 3907);
    }

    #[test]
    #[should_panic(expected = "workgroup_size must be > 0")]
    fn test_count_1d_zero_workgroup() {
        let _ = workgroup_count_1d(100, 0);
    }

    // ---- workgroup_count_2d ----

    #[test]
    fn test_count_2d_exact() {
        assert_eq!(workgroup_count_2d(16, 16, 16), [1, 1, 1]);
        assert_eq!(workgroup_count_2d(32, 32, 16), [2, 2, 1]);
    }

    #[test]
    fn test_count_2d_non_multiple() {
        assert_eq!(workgroup_count_2d(17, 33, 16), [2, 3, 1]);
    }

    #[test]
    #[should_panic(expected = "tile_size must be > 0")]
    fn test_count_2d_zero_tile() {
        let _ = workgroup_count_2d(16, 16, 0);
    }

    // ---- workgroup_count_row_reduce ----

    #[test]
    fn test_count_row_reduce() {
        assert_eq!(workgroup_count_row_reduce(1), [1, 1, 1]);
        assert_eq!(workgroup_count_row_reduce(32), [32, 1, 1]);
        assert_eq!(workgroup_count_row_reduce(1024), [1024, 1, 1]);
    }

    // ---- optimal_elementwise_workgroup ----

    #[test]
    fn test_optimal_large_tensor() {
        assert_eq!(optimal_elementwise_workgroup(100_000, 1024), 256);
    }

    #[test]
    fn test_optimal_small_tensor() {
        assert_eq!(optimal_elementwise_workgroup(32, 1024), 32);
        assert_eq!(optimal_elementwise_workgroup(64, 1024), 64);
        assert_eq!(optimal_elementwise_workgroup(33, 1024), 32); // round down to power of 2
    }

    #[test]
    fn test_optimal_single_element() {
        assert_eq!(optimal_elementwise_workgroup(1, 1024), 1);
    }

    #[test]
    fn test_optimal_zero_elements() {
        assert_eq!(optimal_elementwise_workgroup(0, 1024), 1);
    }

    #[test]
    fn test_optimal_limited_device() {
        // Device with max 128 invocations.
        assert_eq!(optimal_elementwise_workgroup(100_000, 128), 128);
    }

    // ---- validate_dispatch ----

    #[test]
    fn test_validate_valid_dispatch() {
        assert!(validate_dispatch([256, 1, 1], [256, 1, 1], 65535, 1024).is_ok());
        assert!(validate_dispatch([1, 1, 1], [1, 1, 1], 65535, 128).is_ok());
    }

    #[test]
    fn test_validate_group_count_exceeds_limit() {
        let result = validate_dispatch([70000, 1, 1], [256, 1, 1], 65535, 1024);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("exceeds device limit"));
    }

    #[test]
    fn test_validate_zero_group_count() {
        assert!(validate_dispatch([0, 1, 1], [256, 1, 1], 65535, 1024).is_err());
        assert!(validate_dispatch([1, 0, 1], [256, 1, 1], 65535, 1024).is_err());
        assert!(validate_dispatch([1, 1, 0], [256, 1, 1], 65535, 1024).is_err());
    }

    #[test]
    fn test_validate_invocations_exceeds_limit() {
        // 512 * 512 * 1 = 262144 > 1024
        let result = validate_dispatch([1, 1, 1], [512, 512, 1], 65535, 1024);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("exceeds device limit"));
    }

    #[test]
    fn test_validate_zero_local_size() {
        let result = validate_dispatch([1, 1, 1], [0, 1, 1], 65535, 1024);
        assert!(result.is_err());
    }

    // ---- push constant helpers ----

    #[test]
    fn test_push_constants_1d() {
        let bytes = push_constants_1d(1024);
        assert_eq!(u32::from_le_bytes(bytes), 1024);
    }

    #[test]
    fn test_push_constants_1d_zero() {
        let bytes = push_constants_1d(0);
        assert_eq!(u32::from_le_bytes(bytes), 0);
    }

    #[test]
    fn test_push_constants_reduction() {
        let bytes = push_constants_reduction(512, 32);
        assert_eq!(
            u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]),
            512
        );
        assert_eq!(
            u32::from_le_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]),
            32
        );
    }

    #[test]
    fn test_push_constants_matmul() {
        let bytes = push_constants_matmul(128, 256, 64);
        assert_eq!(
            u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]),
            128
        );
        assert_eq!(
            u32::from_le_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]),
            256
        );
        assert_eq!(
            u32::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]),
            64
        );
    }
}

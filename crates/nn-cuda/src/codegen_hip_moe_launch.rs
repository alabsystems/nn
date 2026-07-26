// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Launch configuration helpers for MoE HIP kernels.
//!
//! Extracted from `codegen_hip_moe.rs` to keep it under the 450-line limit (#3087).

use super::{GEMM_PADDED, GEMM_TILE, MOE_BLOCK_SIZE};

/// Compute launch config for the grouped expert GEMM kernel.
///
/// - Grid: `(ceil(out_dim/32), ceil(max_total_tokens/32), 1)`
/// - Block: `(256, 1, 1)`
#[must_use]
pub fn grouped_gemm_launch_config(
    max_total_tokens: usize,
    out_dim: usize,
) -> crate::hip_ffi::LaunchConfig {
    let grid_x = out_dim.div_ceil(GEMM_TILE).min(u32::MAX as usize) as u32;
    let grid_y = max_total_tokens.div_ceil(GEMM_TILE).min(u32::MAX as usize) as u32;
    let shared = ((2 * GEMM_TILE * GEMM_PADDED) * 4) as u32; // Two TILE×PAD float arrays
    crate::hip_ffi::LaunchConfig {
        grid: crate::hip_ffi::Dim3 {
            x: grid_x,
            y: grid_y,
            z: 1,
        },
        block: crate::hip_ffi::Dim3::d1(MOE_BLOCK_SIZE as u32),
        shared_mem_bytes: shared,
    }
}

/// Compute launch config for the SwiGLU kernel.
///
/// - Grid: `(total_tokens, ceil(d_expert / 256), 1)`
/// - Block: `(256, 1, 1)`
#[must_use]
pub fn moe_swiglu_launch_config(
    max_total_tokens: usize,
    d_expert: usize,
) -> crate::hip_ffi::LaunchConfig {
    let grid_x = max_total_tokens.min(u32::MAX as usize) as u32;
    let grid_y = d_expert.div_ceil(MOE_BLOCK_SIZE).min(u32::MAX as usize) as u32;
    crate::hip_ffi::LaunchConfig {
        grid: crate::hip_ffi::Dim3 {
            x: grid_x,
            y: grid_y,
            z: 1,
        },
        block: crate::hip_ffi::Dim3::d1(MOE_BLOCK_SIZE as u32),
        shared_mem_bytes: 0,
    }
}

/// Compute launch config for permute/un-permute kernels.
///
/// - Grid: `(ceil(n_elements / 256), 1, 1)`
/// - Block: `(256, 1, 1)`
#[must_use]
pub fn moe_permute_launch_config(n_elements: usize) -> crate::hip_ffi::LaunchConfig {
    let grid_x = n_elements.div_ceil(MOE_BLOCK_SIZE).min(u32::MAX as usize) as u32;
    crate::hip_ffi::LaunchConfig {
        grid: crate::hip_ffi::Dim3::d1(grid_x),
        block: crate::hip_ffi::Dim3::d1(MOE_BLOCK_SIZE as u32),
        shared_mem_bytes: 0,
    }
}

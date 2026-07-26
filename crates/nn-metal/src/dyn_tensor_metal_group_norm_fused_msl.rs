// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! MSL source for fused GroupNorm kernel (#3294).
//!
//! Simd-accelerated two-pass Kahan-compensated mean+variance reduction.
//! 4 barriers per norm (2 per pass) instead of 16 from tree-based path.
//! Input is pre-reshaped to `[B*G, (C/G)*spatial]` by the Rust caller.
//! One threadgroup per row of the flat shape.
//!
//! After normalization, applies per-channel affine:
//! `weight[ch] * normed + bias[ch]` where `ch` is reconstructed from
//! `(group_idx, local_idx)`.
//!
//! `scalar_type` controls I/O pointer dtype: `"float"` or `"half"`.
//! Accumulators are always `float` for precision.

use super::super::welford_msl;

/// MSL source for the fused GroupNorm kernel.
///
/// Buffers:
///   - 0: `input` — `[flat_rows, flat_cols]` (read-only, reshaped)
///   - 1: `weight` — `[channels]` (read-only, per-channel scale)
///   - 2: `bias` — `[channels]` (read-only, per-channel shift)
///   - 3: `output` — `[flat_rows, flat_cols]` (write-only)
///
/// Constants (set_bytes):
///   - 4: `flat_cols` — uint (C/G * spatial)
///   - 5: `eps` — float
///   - 6: `channels_per_group` — uint (C/G, for affine index reconstruction)
///   - 7: `spatial` — uint (product of spatial dims)
///   - 8: `num_groups` — uint (for channel index: ch = group*cpg + local/spatial)
///
/// Dispatch: one threadgroup per flat_rows, 256 threads per threadgroup.
pub(super) fn fused_group_norm_msl(scalar_type: &str) -> String {
    let preamble = welford_msl::simd_reduction_helpers_msl();
    // The reduction reads from `input[base + i]` with dim_var = "flat_cols".
    let reduction = welford_msl::kahan_two_pass_simd_reduction_msl("flat_cols");

    format!(
        r#"
#include <metal_stdlib>
using namespace metal;

{preamble}

kernel void fused_group_norm_{scalar_type}(
    device const {scalar_type}* input              [[buffer(0)]],
    device const {scalar_type}* weight             [[buffer(1)]],
    device const {scalar_type}* bias               [[buffer(2)]],
    device {scalar_type}* output                   [[buffer(3)]],
    constant uint& flat_cols               [[buffer(4)]],
    constant float& eps                    [[buffer(5)]],
    constant uint& channels_per_group      [[buffer(6)]],
    constant uint& spatial                 [[buffer(7)]],
    constant uint& num_groups              [[buffer(8)]],
    uint gid [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]]
) {{
    uint base = gid * flat_cols;

    // gid = batch_idx * num_groups + group_idx
    uint group_idx_in_batch = gid % num_groups;

{reduction}

    // GroupNorm affine: weight[ch] * normed + bias[ch]
    // Reconstruct channel index: ch = group_idx * cpg + (local_idx / spatial)
    for (uint i = tid; i < flat_cols; i += tg_size) {{
        float normed = (float(input[base + i]) - mean) * inv_std;
        uint local_ch = i / spatial;   // channel within this group
        uint ch = group_idx_in_batch * channels_per_group + local_ch;
        output[base + i] = {scalar_type}(normed * float(weight[ch]) + float(bias[ch]));
    }}
}}
"#
    )
}

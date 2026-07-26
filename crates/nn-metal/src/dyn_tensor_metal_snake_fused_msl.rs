// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! MSL source for fused per-channel Snake activation kernel (#3294).
//!
//! `x + (1/alpha) * sin²(alpha * x)` in a single dispatch.
//! No reduction needed — purely elementwise with per-channel alpha broadcast.
//!
//! `scalar_type` controls I/O pointer dtype: `"float"` or `"half"`.
//! Intermediate math is always `float` for trig precision.

/// MSL source for the fused Snake activation kernel.
///
/// Buffers:
///   - 0: `input` — `[total_elems]` (read-only, flattened)
///   - 1: `alpha` — `[channels]` (read-only, per-channel)
///   - 2: `output` — `[total_elems]` (write-only)
///
/// Constants (set_bytes):
///   - 3: `total_elems` — uint
///   - 4: `channel_stride` — uint (spatial size, for channel index: ch = (i / spatial) % channels)
///   - 5: `channels` — uint
///
/// Dispatch: 1D grid, 256 threads per threadgroup.
pub(super) fn fused_snake_msl(scalar_type: &str) -> String {
    format!(
        r#"
#include <metal_stdlib>
using namespace metal;

kernel void fused_snake_{scalar_type}(
    device const {scalar_type}* input          [[buffer(0)]],
    device const {scalar_type}* alpha          [[buffer(1)]],
    device {scalar_type}* output               [[buffer(2)]],
    constant uint& total_elems         [[buffer(3)]],
    constant uint& channel_stride      [[buffer(4)]],
    constant uint& channels            [[buffer(5)]],
    uint gid [[thread_position_in_grid]]
) {{
    if (gid >= total_elems) return;

    // Reconstruct channel index for left-aligned broadcast:
    // For shape [B, C, *spatial], alpha has shape [C] or [1, C, 1].
    // channel_stride = product(spatial_dims), channels = C.
    // ch = (gid / channel_stride) % channels
    uint ch = (gid / channel_stride) % channels;

    float x = float(input[gid]);
    float a = float(alpha[ch]);
    float inv_a = 1.0f / a;
    float sin_ax = sin(a * x);
    float result = x + inv_a * sin_ax * sin_ax;

    output[gid] = {scalar_type}(result);
}}
"#
    )
}

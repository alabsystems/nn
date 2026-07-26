// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! MSL source for fused BatchNorm kernel (#4324).
//!
//! BatchNorm inference uses precomputed running statistics -- no reduction
//! needed. Per-element affine transform with per-channel parameters:
//!
//! ```text
//! y[n, c, h, w] = (x[n, c, h, w] - running_mean[c]) / sqrt(running_var[c] + eps)
//!                 * weight[c] + bias[c]
//! ```
//!
//! Input is logically `[N, C, H, W]` but the kernel operates on flat indices,
//! computing the channel index as `(flat_idx / spatial_size) % num_channels`.
//! Weight and bias are optional -- the kernel handles all 4 combinations via
//! `has_weight` and `has_bias` template parameters.

/// MSL source for the fused BatchNorm kernel.
///
/// Buffers:
///   - 0: `input`        -- `[total_elems]` (read-only)
///   - 1: `running_mean`  -- `[C]` (read-only)
///   - 2: `running_var`   -- `[C]` (read-only)
///   - 3: `weight`        -- `[C]` (read-only, may be empty if !has_weight)
///   - 4: `bias`          -- `[C]` (read-only, may be empty if !has_bias)
///   - 5: `output`        -- `[total_elems]` (write-only)
///
/// Constants (set_bytes):
///   - 6: `num_channels`  -- uint
///   - 7: `spatial_size`  -- uint
///   - 8: `eps`           -- float
///   - 9: `has_weight`    -- uint (0 or 1)
///   - 10: `has_bias`     -- uint (0 or 1)
///   - 11: `total_elems`  -- uint (for bounds guard)
///
/// Dispatch: 1D grid with `total_elems` threads (dispatch_threads).
///
/// `scalar_type` controls I/O pointer dtype: `"float"` or `"half"`.
/// Accumulators are always `float` for precision.
pub(super) fn fused_batch_norm_msl(scalar_type: &str) -> String {
    format!(
        r#"
#include <metal_stdlib>
using namespace metal;

kernel void fused_batch_norm_{scalar_type}(
    device const {scalar_type}* input        [[buffer(0)]],
    device const float* running_mean         [[buffer(1)]],
    device const float* running_var          [[buffer(2)]],
    device const float* weight               [[buffer(3)]],
    device const float* bias                 [[buffer(4)]],
    device {scalar_type}* output             [[buffer(5)]],
    constant uint& num_channels      [[buffer(6)]],
    constant uint& spatial_size      [[buffer(7)]],
    constant float& eps              [[buffer(8)]],
    constant uint& has_weight        [[buffer(9)]],
    constant uint& has_bias          [[buffer(10)]],
    constant uint& total_elems       [[buffer(11)]],
    uint gid [[thread_position_in_grid]]
) {{
    // Bounds guard: prevent out-of-bounds access if dispatch method changes
    // to dispatch_threadgroups in the future.
    if (gid >= total_elems) return;

    // Channel index: for [N, C, H, W] layout, channel = (flat_idx / spatial) % C
    uint c = (gid / spatial_size) % num_channels;

    float x_val = float(input[gid]);
    float mean = running_mean[c];
    float var = running_var[c];

    // Normalize: (x - mean) / sqrt(var + eps)
    // Clamp to 1e-12 to defend against NaN from corrupted negative variance.
    float inv_std = rsqrt(max(var + eps, 1e-12f));
    float normed = (x_val - mean) * inv_std;

    // Affine transform (optional weight and bias)
    float result = normed;
    if (has_weight) {{
        result *= weight[c];
    }}
    if (has_bias) {{
        result += bias[c];
    }}

    output[gid] = {scalar_type}(result);
}}
"#
    )
}

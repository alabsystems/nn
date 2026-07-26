// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! MSL source for fused Upsample1d + Conv1d kernel (#4310).
//!
//! Single-dispatch architecture:
//!   `fused_upsample_conv1d_{scalar_type}` — reads input `[B, C_in, T]`,
//!   performs nearest-neighbor upsample inline during the Conv1d accumulation,
//!   and writes output `[B, C_out, T_out]`.
//!
//! The key insight: nearest-neighbor upsample maps output position `t_up` to
//! input position `t_up / factor` (integer division). During Conv1d
//! accumulation over kernel positions, each upsampled input position is
//! computed on-the-fly without materializing the intermediate buffer.
//!
//! Memory traffic savings: eliminates B * C_in * T * factor f32 intermediate
//! write+read. For Kokoro f0_energy shapes (B=1, C=4, T=16, factor=4),
//! saves ~1KB per pair; across 6 pairs this adds up and reduces dispatch count.
//!
//! Part of #4310.

/// MSL kernel for fused nearest-neighbor Upsample1d + Conv1d.
///
/// Each thread computes one output element `(b, c_out, t_out)`.
/// For each input channel and kernel position, the upsampled input is
/// computed inline: `input[b, c_in, (t_out * stride + k - padding) / factor]`
/// when the upsampled position is in-bounds and maps to a valid input index.
///
/// Buffers:
///   - 0: `input`        -- `[B, C_in, T]` (read-only)
///   - 1: `weight`       -- `[C_out, C_in, K]` (read-only)
///   - 2: `bias`         -- `[C_out]` (read-only)
///   - 3: `output`       -- `[B, C_out, T_out]` (write-only)
///   - 4: `batch`        -- uint
///   - 5: `in_channels`  -- uint
///   - 6: `out_channels` -- uint
///   - 7: `in_len`       -- uint (T, pre-upsample)
///   - 8: `up_len`       -- uint (T * factor, post-upsample)
///   - 9: `out_len`      -- uint (T_out, post-conv)
///   - 10: `kernel_size` -- uint
///   - 11: `stride`      -- uint
///   - 12: `padding`     -- uint
///   - 13: `factor`      -- uint (upsample factor)
///
/// Dispatch: `[ceil(T_out / tg_x), B * C_out]` threadgroups,
///           `[tg_x, 1]` threads per threadgroup.
pub(crate) fn fused_upsample_conv1d_msl(scalar_type: &str) -> String {
    format!(
        r#"
#include <metal_stdlib>
using namespace metal;

kernel void fused_upsample_conv1d_{scalar_type}(
    device const {scalar_type}* input        [[buffer(0)]],
    device const {scalar_type}* weight       [[buffer(1)]],
    device const {scalar_type}* bias         [[buffer(2)]],
    device {scalar_type}* output             [[buffer(3)]],
    constant uint& batch             [[buffer(4)]],
    constant uint& in_channels       [[buffer(5)]],
    constant uint& out_channels      [[buffer(6)]],
    constant uint& in_len            [[buffer(7)]],
    constant uint& up_len            [[buffer(8)]],
    constant uint& out_len           [[buffer(9)]],
    constant uint& kernel_size       [[buffer(10)]],
    constant uint& stride_val        [[buffer(11)]],
    constant uint& padding_val       [[buffer(12)]],
    constant uint& factor            [[buffer(13)]],
    uint2 gid [[thread_position_in_grid]]
) {{
    uint t = gid.x;
    uint row = gid.y;

    if (t >= out_len) return;

    uint b = row / out_channels;
    uint oc = row % out_channels;
    if (b >= batch) return;

    float acc = float(bias[oc]);

    for (uint ic = 0; ic < in_channels; ic++) {{
        uint w_base = (oc * in_channels + ic) * kernel_size;
        uint in_base = (b * in_channels + ic) * in_len;

        for (uint k = 0; k < kernel_size; k++) {{
            // Position in the upsampled domain.
            int t_up = int(t * stride_val + k) - int(padding_val);

            if (t_up >= 0 && uint(t_up) < up_len) {{
                // Nearest-neighbor: map upsampled position back to input.
                uint t_in = uint(t_up) / factor;
                acc += float(input[in_base + t_in]) * float(weight[w_base + k]);
            }}
        }}
    }}

    uint out_idx = (b * out_channels + oc) * out_len + t;
    output[out_idx] = {scalar_type}(acc);
}}
"#
    )
}

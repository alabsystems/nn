// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! MSL source for fused RmsNorm kernel (#3294).
//!
//! Single-pass Kahan-compensated sum of squares with simd reduction.
//! No mean-centering (unlike LayerNorm): `x * rsqrt(mean(x²) + eps) * weight`.
//!
//! Input x is pre-reshaped to `[rows, hidden_dim]` where rows = product of
//! all dimensions except the last.
//! weight is `[hidden_dim]` (per-channel RmsNorm scale).

use super::super::welford_msl;

/// MSL source for the fused RmsNorm kernel.
///
/// Buffers:
///   - 0: `input` — `[rows, hidden_dim]` (read-only)
///   - 1: `weight` — `[hidden_dim]` (read-only, RmsNorm scale)
///   - 2: `output` — `[rows, hidden_dim]` (write-only)
///
/// Constants (set_bytes):
///   - 3: `hidden_dim` — uint
///   - 4: `eps` — float
///
/// Dispatch: one threadgroup per row, 256 threads per threadgroup.
///
/// Unlike LayerNorm, RmsNorm uses a single-pass Kahan sum of x² (no
/// mean-centering needed). This halves the memory reads compared to the
/// two-pass LayerNorm/InstanceNorm reduction.
///
/// `scalar_type` controls I/O pointer dtype: `"float"` or `"half"`.
/// Accumulators are always `float` for precision. Part of #3294.
pub(super) fn fused_rms_norm_msl(scalar_type: &str) -> String {
    let preamble = welford_msl::simd_reduction_helpers_msl();

    format!(
        r#"
#include <metal_stdlib>
using namespace metal;

{preamble}

kernel void fused_rms_norm_{scalar_type}(
    device const {scalar_type}* input      [[buffer(0)]],
    device const {scalar_type}* weight     [[buffer(1)]],
    device {scalar_type}* output           [[buffer(2)]],
    constant uint& hidden_dim      [[buffer(3)]],
    constant float& eps            [[buffer(4)]],
    uint gid [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]]
) {{
    uint base = gid * hidden_dim;

    // Simd reduction variables.
    uint simd_lane = tid & 31u;
    uint simd_group = tid >> 5u;
    uint num_simdgroups = tg_size >> 5u;
    threadgroup float shared_simd[32];

    // Single-pass Kahan-compensated sum of x² — no mean-centering needed.
    float local_sum_sq = 0.0f;
    float local_comp = 0.0f;
    for (uint i = tid; i < hidden_dim; i += tg_size) {{
        float val = float(input[base + i]);
        float sq = val * val;
        float y = sq - local_comp;
        float t = local_sum_sq + y;
        local_comp = (t - local_sum_sq) - y;
        local_sum_sq = t;
    }}
    float corrected = local_sum_sq - local_comp;
    float total_sq = simd_threadgroup_sum(corrected, shared_simd, simd_lane, simd_group, num_simdgroups);
    float mean_sq = total_sq / max(float(hidden_dim), 1.0f);
    float inv_rms = metal::precise::rsqrt(mean_sq + eps);

    // RmsNorm output: x * inv_rms * weight
    for (uint i = tid; i < hidden_dim; i += tg_size) {{
        output[base + i] = {scalar_type}(float(input[base + i]) * inv_rms * float(weight[i]));
    }}
}}
"#
    )
}

// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! MSL source for conv-with-output-stats kernel variants (#1815 Tier 2).
//!
//! These are modifications of the standard `fused_norm_conv1d_{leaky_relu,snake}`
//! kernels that add a stats epilogue: after writing the conv output, each
//! threadgroup reduces its tile to Welford partial accumulators (n, mean, m2),
//! then the last TG (via atomic counter) merges partials into final mean + inv_std.
//!
//! This eliminates the separate `compute_channel_stats` dispatch for the next
//! FusedResBlock phase, saving 1 Metal dispatch per FusedResBlock (4→3).
//!
//! The epilogue uses Kahan-compensated Welford online variance (#3309), replacing
//! the original naive E[X²]-E[X]² formula that caused amplitude regression (#3233).

use super::super::welford_msl;

/// MSL code for the Kahan-Welford stats epilogue, appended after the conv output write.
///
/// Expects these variables in scope: `b`, `oc`, `out_channels`, `batch`,
/// `out_len`, `result` (float conv output), `valid` (bool), `tid_local`,
/// `tg_pos`, and buffer bindings `next_stats`, `counter`, `partials`,
/// `grid_x_count`, `next_eps`.
///
/// Uses WelfordState and welford_merge from the preamble (welford_msl_preamble).
/// Partials buffer stores 3 floats per TG per row: (n, mean, m2).
fn stats_epilogue_msl() -> &'static str {
    r#"
    // --- Epilogue: Kahan-Welford output stats for next phase (#3309) ---
    threadgroup float shared_n_w[64];
    threadgroup float shared_mean_w[64];
    threadgroup float shared_m2_w[64];

    // Each thread contributes one sample (or nothing if out-of-bounds).
    shared_n_w[tid_local] = valid ? 1.0f : 0.0f;
    shared_mean_w[tid_local] = valid ? result : 0.0f;
    shared_m2_w[tid_local] = 0.0f;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Tree reduction via Kahan-compensated welford_merge.
    for (uint stride = 32; stride > 0; stride >>= 1) {
        if (tid_local < stride) {
            WelfordState a = {shared_n_w[tid_local], shared_mean_w[tid_local], shared_m2_w[tid_local], 0.0f};
            WelfordState b = {shared_n_w[tid_local + stride], shared_mean_w[tid_local + stride], shared_m2_w[tid_local + stride], 0.0f};
            WelfordState merged = welford_merge(a, b);
            shared_n_w[tid_local] = merged.n;
            shared_mean_w[tid_local] = merged.mean;
            shared_m2_w[tid_local] = merged.m2;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    uint total_rows = batch * out_channels;
    uint stats_row = b * out_channels + oc;

    if (grid_x_count == 1) {
        // Case 1: single TG per row — write final stats directly.
        if (tid_local == 0) {
            float final_n = max(shared_n_w[0], 1.0f);
            float var = shared_m2_w[0] / final_n;
            float inv_std = rsqrt(max(var, 0.0f) + next_eps);
            next_stats[stats_row * 2]     = shared_mean_w[0];
            next_stats[stats_row * 2 + 1] = inv_std;
        }
    } else {
        // Case 2: multiple TGs per row — write Welford partials + atomic completion.
        if (tid_local == 0) {
            uint pidx = (tg_pos.x * total_rows + stats_row) * 3;
            partials[pidx]     = shared_n_w[0];
            partials[pidx + 1] = shared_mean_w[0];
            partials[pidx + 2] = shared_m2_w[0];

            threadgroup_barrier(mem_flags::mem_device);
            uint prev = atomic_fetch_add_explicit(
                &counter[stats_row], 1u, memory_order_relaxed);
            if (prev == grid_x_count - 1) {
                // Merge all TG partials using Kahan-compensated Welford.
                WelfordState acc = {0.0f, 0.0f, 0.0f, 0.0f};
                for (uint i = 0; i < grid_x_count; i++) {
                    uint pi = (i * total_rows + stats_row) * 3;
                    WelfordState part = {partials[pi], partials[pi+1], partials[pi+2], 0.0f};
                    acc = welford_merge(acc, part);
                }
                float final_n = max(acc.n, 1.0f);
                float var = acc.m2 / final_n;
                float inv_std = rsqrt(max(var, 0.0f) + next_eps);
                next_stats[stats_row * 2]     = acc.mean;
                next_stats[stats_row * 2 + 1] = inv_std;
            }
        }
    }
"#
}

/// MSL kernel for fused NormActivConv1d + LeakyRelu + output stats epilogue.
///
/// Same as `fused_norm_conv1d_leaky_relu` but does NOT early-return for
/// out-of-bounds threads (all threads participate in the stats reduction)
/// and appends the Kahan-Welford output stats epilogue.
///
/// Extra buffers (19-23) carry the stats output, atomic counter, and partials.
pub(super) fn fused_norm_conv1d_leaky_relu_with_stats_msl(scalar_type: &str) -> String {
    let preamble = welford_msl::welford_msl_preamble();
    let epilogue = stats_epilogue_msl();
    let mut msl = format!(
        r#"
#include <metal_stdlib>
using namespace metal;

{preamble}

kernel void fused_norm_conv1d_leaky_relu_with_stats_{scalar_type}(
    device const {scalar_type}* input        [[buffer(0)]],
    device const float* stats        [[buffer(1)]],
    device const {scalar_type}* gamma        [[buffer(2)]],
    device const {scalar_type}* beta         [[buffer(3)]],
    device const {scalar_type}* weight       [[buffer(4)]],
    device const {scalar_type}* bias         [[buffer(5)]],
    device const {scalar_type}* residual     [[buffer(6)]],
    device {scalar_type}* output             [[buffer(7)]],
    constant uint& batch             [[buffer(8)]],
    constant uint& in_channels       [[buffer(9)]],
    constant uint& out_channels      [[buffer(10)]],
    constant uint& in_len            [[buffer(11)]],
    constant uint& out_len           [[buffer(12)]],
    constant uint& kernel_size       [[buffer(13)]],
    constant uint& padding           [[buffer(14)]],
    constant uint& dilation          [[buffer(15)]],
    constant float& slope            [[buffer(16)]],
    constant uint& has_residual      [[buffer(17)]],
    constant float& residual_scale   [[buffer(18)]],
    device float* next_stats         [[buffer(19)]],
    device atomic_uint* counter      [[buffer(20)]],
    device float* partials           [[buffer(21)]],
    constant uint& grid_x_count      [[buffer(22)]],
    constant float& next_eps         [[buffer(23)]],
    uint2 gid [[thread_position_in_grid]],
    uint tid_local [[thread_index_in_threadgroup]],
    uint2 tg_pos [[threadgroup_position_in_grid]]
) {{
    uint row = gid.y;
    uint t = gid.x;
    uint b = row / out_channels;
    uint oc = row % out_channels;
    bool valid = (t < out_len) && (b < batch);

    // Threadgroup-shared per-channel parameters (#4264).
    threadgroup float tg_mean;
    threadgroup float tg_inv_s;
    threadgroup float tg_g;
    threadgroup float tg_be;
    threadgroup float tg_w[16];

    float result = 0.0f;
    float acc = 0.0f;
    if (valid) {{
        acc = float(bias[oc]);
    }}

    for (uint ic = 0; ic < in_channels; ic++) {{
        if (tid_local == 0) {{
            uint stats_idx = (b * in_channels + ic) * 2;
            tg_mean = stats[stats_idx];
            tg_inv_s = stats[stats_idx + 1];
            uint affine_idx = b * in_channels + ic;
            tg_g = float(gamma[affine_idx]);
            tg_be = float(beta[affine_idx]);
            uint w_base = (oc * in_channels + ic) * kernel_size;
            for (uint k = 0; k < kernel_size && k < 16; k++) {{
                tg_w[k] = float(weight[w_base + k]);
            }}
        }}
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (valid) {{
            float mean_l = tg_mean;
            float inv_s_l = tg_inv_s;
            float g_l = tg_g;
            float be_l = tg_be;
            uint in_base = (b * in_channels + ic) * in_len;
            for (uint k = 0; k < kernel_size; k++) {{
                int t_in = int(t) + int(k) * int(dilation) - int(padding);
                if (t_in >= 0 && uint(t_in) < in_len) {{
                    float x = float(input[in_base + uint(t_in)]);
                    float normed = (x - mean_l) * inv_s_l;
                    float y = (1.0f + g_l) * normed + be_l;
                    float activated = y >= 0.0f ? y : slope * y;
                    float w_val = (k < 16) ? tg_w[k] : float(weight[(oc * in_channels + ic) * kernel_size + k]);
                    acc += activated * w_val;
                }}
            }}
        }}
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }}

    if (valid) {{
        uint out_idx = (b * out_channels + oc) * out_len + t;
        if (has_residual != 0) {{
            acc = (acc + float(residual[out_idx])) * residual_scale;
        }}
        output[out_idx] = {scalar_type}(acc);
        result = acc;
    }}
"#
    );
    msl.push_str(epilogue);
    msl.push_str("\n}\n");
    msl
}

/// MSL kernel for fused NormActivConv1d + Snake + output stats epilogue.
///
/// Same as `fused_norm_conv1d_snake` but with Kahan-Welford stats epilogue.
/// Buffer(16) is per-channel `alpha` device buffer (not scalar slope).
pub(super) fn fused_norm_conv1d_snake_with_stats_msl(scalar_type: &str) -> String {
    let preamble = welford_msl::welford_msl_preamble();
    let epilogue = stats_epilogue_msl();
    let mut msl = format!(
        r#"
#include <metal_stdlib>
using namespace metal;

{preamble}

kernel void fused_norm_conv1d_snake_with_stats_{scalar_type}(
    device const {scalar_type}* input        [[buffer(0)]],
    device const float* stats        [[buffer(1)]],
    device const {scalar_type}* gamma        [[buffer(2)]],
    device const {scalar_type}* beta         [[buffer(3)]],
    device const {scalar_type}* weight       [[buffer(4)]],
    device const {scalar_type}* bias         [[buffer(5)]],
    device const {scalar_type}* residual     [[buffer(6)]],
    device {scalar_type}* output             [[buffer(7)]],
    constant uint& batch             [[buffer(8)]],
    constant uint& in_channels       [[buffer(9)]],
    constant uint& out_channels      [[buffer(10)]],
    constant uint& in_len            [[buffer(11)]],
    constant uint& out_len           [[buffer(12)]],
    constant uint& kernel_size       [[buffer(13)]],
    constant uint& padding           [[buffer(14)]],
    constant uint& dilation          [[buffer(15)]],
    device const {scalar_type}* alpha        [[buffer(16)]],
    constant uint& has_residual      [[buffer(17)]],
    constant float& residual_scale   [[buffer(18)]],
    device float* next_stats         [[buffer(19)]],
    device atomic_uint* counter      [[buffer(20)]],
    device float* partials           [[buffer(21)]],
    constant uint& grid_x_count      [[buffer(22)]],
    constant float& next_eps         [[buffer(23)]],
    uint2 gid [[thread_position_in_grid]],
    uint tid_local [[thread_index_in_threadgroup]],
    uint2 tg_pos [[threadgroup_position_in_grid]]
) {{
    uint row = gid.y;
    uint t = gid.x;
    uint b = row / out_channels;
    uint oc = row % out_channels;
    bool valid = (t < out_len) && (b < batch);

    // Threadgroup-shared per-channel parameters (#4264).
    threadgroup float tg_mean;
    threadgroup float tg_inv_s;
    threadgroup float tg_g;
    threadgroup float tg_be;
    threadgroup float tg_a;
    threadgroup float tg_inv_a;
    threadgroup float tg_w[16];

    float result = 0.0f;
    float acc = 0.0f;
    if (valid) {{
        acc = float(bias[oc]);
    }}

    for (uint ic = 0; ic < in_channels; ic++) {{
        if (tid_local == 0) {{
            uint stats_idx = (b * in_channels + ic) * 2;
            tg_mean = stats[stats_idx];
            tg_inv_s = stats[stats_idx + 1];
            uint affine_idx = b * in_channels + ic;
            tg_g = float(gamma[affine_idx]);
            tg_be = float(beta[affine_idx]);
            tg_a = max(float(alpha[ic]), 1e-8f);
            tg_inv_a = 1.0f / tg_a;
            uint w_base = (oc * in_channels + ic) * kernel_size;
            for (uint k = 0; k < kernel_size && k < 16; k++) {{
                tg_w[k] = float(weight[w_base + k]);
            }}
        }}
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (valid) {{
            float mean_l = tg_mean;
            float inv_s_l = tg_inv_s;
            float g_l = tg_g;
            float be_l = tg_be;
            float a_l = tg_a;
            float inv_a_l = tg_inv_a;
            uint in_base = (b * in_channels + ic) * in_len;
            for (uint k = 0; k < kernel_size; k++) {{
                int t_in = int(t) + int(k) * int(dilation) - int(padding);
                if (t_in >= 0 && uint(t_in) < in_len) {{
                    float x = float(input[in_base + uint(t_in)]);
                    float normed = (x - mean_l) * inv_s_l;
                    float y = (1.0f + g_l) * normed + be_l;
                    float sin_val = sin(a_l * y);
                    float activated = y + inv_a_l * sin_val * sin_val;
                    float w_val = (k < 16) ? tg_w[k] : float(weight[(oc * in_channels + ic) * kernel_size + k]);
                    acc += activated * w_val;
                }}
            }}
        }}
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }}

    if (valid) {{
        uint out_idx = (b * out_channels + oc) * out_len + t;
        if (has_residual != 0) {{
            acc = (acc + float(residual[out_idx])) * residual_scale;
        }}
        output[out_idx] = {scalar_type}(acc);
        result = acc;
    }}
"#
    );
    msl.push_str(epilogue);
    msl.push_str("\n}\n");
    msl
}

/// Collect with-stats MSL sources for pre-compilation.
pub(super) fn collect_msl_sources() -> Vec<(&'static str, String)> {
    vec![
        (
            "fused_norm_conv1d_leaky_relu_with_stats_float",
            fused_norm_conv1d_leaky_relu_with_stats_msl("float"),
        ),
        (
            "fused_norm_conv1d_leaky_relu_with_stats_half",
            fused_norm_conv1d_leaky_relu_with_stats_msl("half"),
        ),
        (
            "fused_norm_conv1d_snake_with_stats_float",
            fused_norm_conv1d_snake_with_stats_msl("float"),
        ),
        (
            "fused_norm_conv1d_snake_with_stats_half",
            fused_norm_conv1d_snake_with_stats_msl("half"),
        ),
    ]
}

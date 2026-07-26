// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! MSL source for fused NormActivConv1d kernels (#2780).
//!
//! Two-dispatch architecture:
//!   1. `compute_channel_stats_f32` — per-channel mean + inv_std using
//!      Kahan-compensated two-pass reduction (same algorithm as AdaIN kernels).
//!   2. `fused_norm_conv1d_{leaky_relu,snake}_f32` — inline normalization +
//!      affine + activation during Conv1d accumulation. Reads precomputed
//!      stats, avoids writing a full intermediate activated tensor.
//!
//! Memory traffic savings: eliminates B×C_in×T f32 intermediate write+read
//! per NormActivConv1d phase. For typical F0 shapes (B=1, C=512, T=100),
//! this saves ~200KB per phase.
//!
//! Part of #2780: FusedAdainResBlock GPU NativeOp.

use super::super::welford_msl;

/// MSL kernel that computes per-channel mean and inv_std.
///
/// Input layout: `[rows, spatial_len]` where rows = B × C.
/// Output: `stats[rows * 2]` with `stats[2*r] = mean`, `stats[2*r+1] = inv_std`.
///
/// Dispatch: one threadgroup per row, 256 threads per threadgroup.
/// `scalar_type` controls input pointer dtype: `"float"` or `"half"`.
/// Stats output is always `float` (precision-critical mean/inv_std).
/// Part of #2981 F16 Tier 2.
pub(super) fn compute_channel_stats_msl(scalar_type: &str) -> String {
    let algo = welford_msl::DEFAULT_NORM_REDUCTION;
    let preamble = welford_msl::norm_preamble_msl(algo);
    let reduction = welford_msl::norm_reduction_msl(algo, "spatial_len", 256);

    format!(
        r#"
#include <metal_stdlib>
using namespace metal;

{preamble}

kernel void compute_channel_stats_{scalar_type}(
    device const {scalar_type}* input    [[buffer(0)]],
    device float* stats          [[buffer(1)]],
    constant uint& spatial_len   [[buffer(2)]],
    constant float& eps          [[buffer(3)]],
    uint gid [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]]
) {{
    uint base = gid * spatial_len;

{reduction}

    // Write mean and inv_std for this channel.
    if (tid == 0) {{
        stats[gid * 2]     = mean;
        stats[gid * 2 + 1] = inv_std;
    }}
}}
"#
    )
}

/// MSL kernel for fused InstanceNorm + affine + LeakyRelu + Conv1d (#2780).
///
/// Each thread computes one output element `(b, c_out, t_out)`.
/// For each input channel and kernel position, normalizes the input inline
/// using precomputed stats, applies style affine + LeakyRelu, then
/// accumulates the Conv1d dot product.
///
/// Optional residual: if `has_residual != 0`, adds `residual[out_idx]`
/// and multiplies by `residual_scale`. Used by FusedResBlock phase 2.
///
/// Buffers:
///   - 0: `input`    — `[B, C_in, T]` f32 (read-only)
///   - 1: `stats`    — `[B*C_in, 2]` f32 (mean, inv_std per channel)
///   - 2: `gamma`    — `[B*C_in]` f32 (style scale per batch×channel)
///   - 3: `beta`     — `[B*C_in]` f32 (style shift per batch×channel)
///   - 4: `weight`   — `[C_out, C_in, K]` f32 (conv weight)
///   - 5: `bias`     — `[C_out]` f32 (conv bias)
///   - 6: `residual` — `[B, C_out, T_out]` f32 (optional, for FusedResBlock)
///   - 7: `output`   — `[B, C_out, T_out]` f32 (write-only)
///
/// Function constants 0-2: FC_KERNEL_SIZE, FC_PADDING, FC_DILATION (#3449).
///
/// Constants (set_bytes):
///   - 8:  `batch`          — uint
///   - 9:  `in_channels`    — uint
///   - 10: `out_channels`   — uint
///   - 11: `in_len`         — uint (T)
///   - 12: `out_len`        — uint (T_out)
///   - 13: `slope`          — float (LeakyRelu negative slope)
///   - 14: `has_residual`   — uint (0 = no residual, 1 = add residual)
///   - 15: `residual_scale` — float (multiplied after residual add)
///
/// Dispatch: `[ceil(T_out / tg_x), B * C_out]` threadgroups,
///           `[tg_x, 1]` threads per threadgroup.
/// `scalar_type` controls I/O pointer dtype: `"float"` or `"half"`.
/// Stats buffer stays `float` (precomputed mean/inv_std). Conv accumulator
/// is always `float` for precision. Part of #2981 F16 Tier 2.
///
/// When `fast_half` is true and `scalar_type` is `"half"`, the conv
/// accumulator and all intermediate arithmetic use `half` precision for
/// ~2x throughput on Apple M-series GPUs (28 TFLOPS F16 vs 14 TFLOPS F32
/// on M4 Max). Stats buffer stays `float` (precision-critical mean/inv_std).
/// Normalization is performed in float, then the result is converted to half
/// before the activation+conv accumulation. This limits error accumulation
/// to the K*C_in multiply-adds (e.g. 1536 MADs for K=3, C_in=512).
/// Use only after NY QuantizationCertificate verification.
pub(super) fn fused_norm_conv1d_leaky_relu_msl(scalar_type: &str, fast_half: bool) -> String {
    if fast_half && scalar_type == "half" {
        return fused_norm_conv1d_leaky_relu_fast_half_msl();
    }
    format!(
        r#"
#include <metal_stdlib>
using namespace metal;

// Function constants for kernel specialization (#3449).
// The Metal compiler unrolls the conv loop and eliminates dead code
// based on these compile-time-known values.
constant uint FC_KERNEL_SIZE [[function_constant(0)]];
constant uint FC_PADDING     [[function_constant(1)]];
constant uint FC_DILATION    [[function_constant(2)]];

kernel void fused_norm_conv1d_leaky_relu_{scalar_type}(
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
    constant float& slope            [[buffer(13)]],
    constant uint& has_residual      [[buffer(14)]],
    constant float& residual_scale   [[buffer(15)]],
    uint2 gid [[thread_position_in_grid]],
    uint2 grid_size [[threads_per_grid]],
    uint tid_local [[thread_index_in_threadgroup]]
) {{
    uint row = gid.y;
    uint t = gid.x;

    uint b = row / out_channels;
    uint oc = row % out_channels;
    bool valid = (t < out_len) && (b < batch);

    // Threadgroup-shared per-channel parameters (#4264).
    // All 64 threads in the TG compute the same (b, oc) pair at different
    // timesteps. Per-channel stats/gamma/beta are identical across threads
    // — cache in shared memory to reduce device reads by ~64x.
    threadgroup float tg_mean;
    threadgroup float tg_inv_s;
    threadgroup float tg_g;
    threadgroup float tg_be;
    // Cache conv weights for each input channel — shared across all 64 threads.
    threadgroup float tg_w[16];

    float acc = 0.0f;
    if (valid) {{
        acc = float(bias[oc]);
    }}

    for (uint ic = 0; ic < in_channels; ic++) {{
        // Thread 0 loads per-channel parameters into threadgroup memory.
        if (tid_local == 0) {{
            uint stats_idx = (b * in_channels + ic) * 2;
            tg_mean = stats[stats_idx];
            tg_inv_s = stats[stats_idx + 1];

            uint affine_idx = b * in_channels + ic;
            tg_g = float(gamma[affine_idx]);
            tg_be = float(beta[affine_idx]);

            // Load conv weights for this (oc, ic) pair.
            uint w_base = (oc * in_channels + ic) * FC_KERNEL_SIZE;
            for (uint k = 0; k < FC_KERNEL_SIZE && k < 16; k++) {{
                tg_w[k] = float(weight[w_base + k]);
            }}
        }}
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (valid) {{
            // Read cached parameters from threadgroup memory.
            float mean_l = tg_mean;
            float inv_s_l = tg_inv_s;
            float g_l = tg_g;
            float be_l = tg_be;

            uint in_base = (b * in_channels + ic) * in_len;

            for (uint k = 0; k < FC_KERNEL_SIZE; k++) {{
                int t_in = int(t) + int(k) * int(FC_DILATION) - int(FC_PADDING);
                if (t_in >= 0 && uint(t_in) < in_len) {{
                    float x = float(input[in_base + uint(t_in)]);
                    float normed = (x - mean_l) * inv_s_l;
                    float y = (1.0f + g_l) * normed + be_l;
                    float activated = y >= 0.0f ? y : slope * y;
                    acc += activated * tg_w[k];
                }}
            }}
        }}
        // Barrier ensures tg_* is not overwritten until all threads are done.
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }}

    if (valid) {{
        uint out_idx = (b * out_channels + oc) * out_len + t;
        if (has_residual != 0) {{
            acc = (acc + float(residual[out_idx])) * residual_scale;
        }}
        output[out_idx] = {scalar_type}(acc);
    }}
}}
"#
    )
}

/// MSL kernel for fused InstanceNorm + affine + Snake + Conv1d (#2780).
///
/// Same architecture as [`fused_norm_conv1d_leaky_relu_msl`] but replaces
/// LeakyRelu with Snake activation: `x + (1/alpha) * sin(alpha*x)^2`.
///
/// Key difference from LeakyRelu variant: buffer(13) is a per-channel
/// `alpha` device buffer `[C_in]` (bound via `set_buffer_with_offset`),
/// not a scalar `slope` constant (bound via `set_bytes`).
///
/// Optimization (#4264): Uses threadgroup memory to cache per-channel
/// stats (mean, inv_std), gamma, beta, alpha, and conv weights. All 64
/// threads in a threadgroup compute output for the same (b, oc) pair
/// across consecutive timesteps, so they all read the same per-channel
/// parameters. Thread 0 loads these into shared memory; all threads
/// read from fast threadgroup memory instead of device memory. This
/// reduces device memory reads by ~64x for these parameters. Similarly,
/// conv weights `[K]` for each `(oc, ic)` pair are shared across all
/// 64 threads. With FC_KERNEL_SIZE <= 16 (covers all Kokoro configs),
/// the weights fit in a small shared array.
///
/// Buffers 0-12: identical to LeakyRelu variant.
/// Function constants 0-2: FC_KERNEL_SIZE, FC_PADDING, FC_DILATION (#3449).
///   - 13: `alpha`          — `[C_in]` f32 (per-channel Snake alpha, **device buffer**)
///   - 14: `has_residual`   — uint (0 = no residual, 1 = add residual)
///   - 15: `residual_scale` — float (multiplied after residual add)
///
/// Dispatch: `[ceil(T_out / tg_x), B * C_out]` threadgroups.
/// `scalar_type` controls I/O pointer dtype: `"float"` or `"half"`.
/// Stats buffer stays `float`. Part of #2981 F16 Tier 2.
pub(super) fn fused_norm_conv1d_snake_msl(scalar_type: &str, fast_half: bool) -> String {
    if fast_half && scalar_type == "half" {
        return fused_norm_conv1d_snake_fast_half_msl();
    }
    format!(
        r#"
#include <metal_stdlib>
using namespace metal;

// Function constants for kernel specialization (#3449).
constant uint FC_KERNEL_SIZE [[function_constant(0)]];
constant uint FC_PADDING     [[function_constant(1)]];
constant uint FC_DILATION    [[function_constant(2)]];

kernel void fused_norm_conv1d_snake_{scalar_type}(
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
    device const {scalar_type}* alpha        [[buffer(13)]],
    constant uint& has_residual      [[buffer(14)]],
    constant float& residual_scale   [[buffer(15)]],
    uint2 gid [[thread_position_in_grid]],
    uint2 grid_size [[threads_per_grid]],
    uint tid_local [[thread_index_in_threadgroup]]
) {{
    uint row = gid.y;
    uint t = gid.x;

    uint b = row / out_channels;
    uint oc = row % out_channels;
    bool valid = (t < out_len) && (b < batch);

    // Threadgroup-shared per-channel parameters (#4264).
    // All 64 threads in the TG compute the same (b, oc) pair at different
    // timesteps. Per-channel stats/gamma/beta/alpha are identical across
    // threads — cache in shared memory to reduce device reads by ~64x.
    threadgroup float tg_mean;
    threadgroup float tg_inv_s;
    threadgroup float tg_g;
    threadgroup float tg_be;
    threadgroup float tg_a;
    threadgroup float tg_inv_a;
    // Cache conv weights for each input channel — shared across all 64 threads.
    // FC_KERNEL_SIZE is a compile-time function constant; max 16 covers all
    // Kokoro configs (typically 3 or 7).
    threadgroup float tg_w[16];

    float acc = 0.0f;
    if (valid) {{
        acc = float(bias[oc]);
    }}

    for (uint ic = 0; ic < in_channels; ic++) {{
        // Thread 0 loads per-channel parameters into threadgroup memory.
        if (tid_local == 0) {{
            uint stats_idx = (b * in_channels + ic) * 2;
            tg_mean = stats[stats_idx];
            tg_inv_s = stats[stats_idx + 1];

            uint affine_idx = b * in_channels + ic;
            tg_g = float(gamma[affine_idx]);
            tg_be = float(beta[affine_idx]);

            tg_a = max(float(alpha[ic]), 1e-8f);
            tg_inv_a = 1.0f / tg_a;

            // Load conv weights for this (oc, ic) pair.
            uint w_base = (oc * in_channels + ic) * FC_KERNEL_SIZE;
            for (uint k = 0; k < FC_KERNEL_SIZE && k < 16; k++) {{
                tg_w[k] = float(weight[w_base + k]);
            }}
        }}
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (valid) {{
            // Read cached parameters from threadgroup memory.
            float mean_l = tg_mean;
            float inv_s_l = tg_inv_s;
            float g_l = tg_g;
            float be_l = tg_be;
            float a_l = tg_a;
            float inv_a_l = tg_inv_a;

            uint in_base = (b * in_channels + ic) * in_len;

            for (uint k = 0; k < FC_KERNEL_SIZE; k++) {{
                int t_in = int(t) + int(k) * int(FC_DILATION) - int(FC_PADDING);
                if (t_in >= 0 && uint(t_in) < in_len) {{
                    float x = float(input[in_base + uint(t_in)]);
                    float normed = (x - mean_l) * inv_s_l;
                    float y = (1.0f + g_l) * normed + be_l;
                    float sin_val = sin(a_l * y);
                    float activated = y + inv_a_l * sin_val * sin_val;
                    acc += activated * tg_w[k];
                }}
            }}
        }}
        // Barrier ensures tg_* is not overwritten until all threads are done.
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }}

    if (valid) {{
        uint out_idx = (b * out_channels + oc) * out_len + t;
        if (has_residual != 0) {{
            acc = (acc + float(residual[out_idx])) * residual_scale;
        }}
        output[out_idx] = {scalar_type}(acc);
    }}
}}
"#
    )
}

// Fast-half MSL kernel variants extracted to keep file under 500 lines.
#[path = "dyn_tensor_metal_norm_conv_fused_msl_fast_half.rs"]
mod fast_half;

use fast_half::{
    fused_norm_conv1d_leaky_relu_fast_half_msl, fused_norm_conv1d_snake_fast_half_msl,
};

// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Fast-half MSL kernels for fused NormActivConv1d (#4264).
//!
//! These variants use `half` accumulators and intermediate arithmetic
//! instead of `float`, targeting ~2x throughput on Apple M-series GPUs
//! (28 TFLOPS F16 vs 14 TFLOPS F32 on M4 Max).
//!
//! Design decisions:
//! - **Stats buffer stays `float`**: mean/inv_std are precision-critical;
//!   computed once by the stats kernel and shared across all threads.
//! - **Normalization in float, then convert**: `(x - mean) * inv_std` is
//!   done in float to avoid catastrophic cancellation, then the result
//!   is cast to `half` before activation and conv accumulation.
//! - **Activation + conv in half**: LeakyRelu/Snake + multiply-add in half.
//!   For Kokoro Stage 1 (K=3, C_in=512), this is 1536 half MADs per
//!   output element. Half has ~3 decimal digits; error can compound.
//! - **Threadgroup weights as half**: `tg_w[16]` is `half` for native
//!   half multiply in the inner loop.
//!
//! Use only after NY QuantizationCertificate verifies acceptable
//! output error bounds. The safe `_half` variants (float accumulators)
//! remain the default until verification confirms fast-half is safe.

/// Fast-half LeakyRelu variant: accumulator and intermediates are `half`.
///
/// Kernel function name: `fused_norm_conv1d_leaky_relu_fast_half`
pub(super) fn fused_norm_conv1d_leaky_relu_fast_half_msl() -> String {
    String::from(
        r#"
#include <metal_stdlib>
using namespace metal;

constant uint FC_KERNEL_SIZE [[function_constant(0)]];
constant uint FC_PADDING     [[function_constant(1)]];
constant uint FC_DILATION    [[function_constant(2)]];

kernel void fused_norm_conv1d_leaky_relu_fast_half(
    device const half* input         [[buffer(0)]],
    device const float* stats        [[buffer(1)]],
    device const half* gamma         [[buffer(2)]],
    device const half* beta          [[buffer(3)]],
    device const half* weight        [[buffer(4)]],
    device const half* bias          [[buffer(5)]],
    device const half* residual      [[buffer(6)]],
    device half* output              [[buffer(7)]],
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
) {
    uint row = gid.y;
    uint t = gid.x;

    uint b = row / out_channels;
    uint oc = row % out_channels;
    bool valid = (t < out_len) && (b < batch);

    // Stats stay float (precision-critical). Per-channel parameters
    // cached in threadgroup memory as half for fast accumulation.
    threadgroup float tg_mean;
    threadgroup float tg_inv_s;
    threadgroup half tg_g;
    threadgroup half tg_be;
    // Conv weights in half for native half MAC.
    threadgroup half tg_w[16];

    half slope_h = half(slope);
    half acc = 0.0h;
    if (valid) {
        acc = bias[oc];
    }

    for (uint ic = 0; ic < in_channels; ic++) {
        if (tid_local == 0) {
            uint stats_idx = (b * in_channels + ic) * 2;
            tg_mean = stats[stats_idx];
            tg_inv_s = stats[stats_idx + 1];

            uint affine_idx = b * in_channels + ic;
            tg_g = gamma[affine_idx];
            tg_be = beta[affine_idx];

            uint w_base = (oc * in_channels + ic) * FC_KERNEL_SIZE;
            for (uint k = 0; k < FC_KERNEL_SIZE && k < 16; k++) {
                tg_w[k] = weight[w_base + k];
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (valid) {
            float mean_l = tg_mean;
            float inv_s_l = tg_inv_s;
            half g_l = tg_g;
            half be_l = tg_be;

            uint in_base = (b * in_channels + ic) * in_len;

            for (uint k = 0; k < FC_KERNEL_SIZE; k++) {
                int t_in = int(t) + int(k) * int(FC_DILATION) - int(FC_PADDING);
                if (t_in >= 0 && uint(t_in) < in_len) {
                    // Normalize in float for precision, then convert to half.
                    float x_f = float(input[in_base + uint(t_in)]);
                    float normed_f = (x_f - mean_l) * inv_s_l;
                    half normed = half(normed_f);
                    half y = (1.0h + g_l) * normed + be_l;
                    half activated = y >= 0.0h ? y : slope_h * y;
                    acc += activated * tg_w[k];
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (valid) {
        uint out_idx = (b * out_channels + oc) * out_len + t;
        if (has_residual != 0) {
            acc = (acc + residual[out_idx]) * half(residual_scale);
        }
        output[out_idx] = acc;
    }
}
"#,
    )
}

/// Fast-half Snake variant: accumulator and intermediates are `half`.
///
/// Same architecture as [`fused_norm_conv1d_leaky_relu_fast_half_msl`]
/// but uses Snake activation: `x + (1/alpha) * sin(alpha*x)^2`.
/// Normalization stays in float; everything after the half conversion
/// uses native half precision including sin/cos intrinsics.
///
/// Kernel function name: `fused_norm_conv1d_snake_fast_half`
pub(super) fn fused_norm_conv1d_snake_fast_half_msl() -> String {
    String::from(
        r#"
#include <metal_stdlib>
using namespace metal;

constant uint FC_KERNEL_SIZE [[function_constant(0)]];
constant uint FC_PADDING     [[function_constant(1)]];
constant uint FC_DILATION    [[function_constant(2)]];

kernel void fused_norm_conv1d_snake_fast_half(
    device const half* input         [[buffer(0)]],
    device const float* stats        [[buffer(1)]],
    device const half* gamma         [[buffer(2)]],
    device const half* beta          [[buffer(3)]],
    device const half* weight        [[buffer(4)]],
    device const half* bias          [[buffer(5)]],
    device const half* residual      [[buffer(6)]],
    device half* output              [[buffer(7)]],
    constant uint& batch             [[buffer(8)]],
    constant uint& in_channels       [[buffer(9)]],
    constant uint& out_channels      [[buffer(10)]],
    constant uint& in_len            [[buffer(11)]],
    constant uint& out_len           [[buffer(12)]],
    device const half* alpha         [[buffer(13)]],
    constant uint& has_residual      [[buffer(14)]],
    constant float& residual_scale   [[buffer(15)]],
    uint2 gid [[thread_position_in_grid]],
    uint2 grid_size [[threads_per_grid]],
    uint tid_local [[thread_index_in_threadgroup]]
) {
    uint row = gid.y;
    uint t = gid.x;

    uint b = row / out_channels;
    uint oc = row % out_channels;
    bool valid = (t < out_len) && (b < batch);

    // Stats stay float. Per-channel parameters cached as half.
    threadgroup float tg_mean;
    threadgroup float tg_inv_s;
    threadgroup half tg_g;
    threadgroup half tg_be;
    threadgroup half tg_a;
    threadgroup half tg_inv_a;
    threadgroup half tg_w[16];

    half acc = 0.0h;
    if (valid) {
        acc = bias[oc];
    }

    for (uint ic = 0; ic < in_channels; ic++) {
        if (tid_local == 0) {
            uint stats_idx = (b * in_channels + ic) * 2;
            tg_mean = stats[stats_idx];
            tg_inv_s = stats[stats_idx + 1];

            uint affine_idx = b * in_channels + ic;
            tg_g = gamma[affine_idx];
            tg_be = beta[affine_idx];

            // Clamp alpha in float before converting to half to avoid
            // division by zero in half precision.
            float a_f = max(float(alpha[ic]), 1e-8f);
            tg_a = half(a_f);
            tg_inv_a = half(1.0f / a_f);

            uint w_base = (oc * in_channels + ic) * FC_KERNEL_SIZE;
            for (uint k = 0; k < FC_KERNEL_SIZE && k < 16; k++) {
                tg_w[k] = weight[w_base + k];
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (valid) {
            float mean_l = tg_mean;
            float inv_s_l = tg_inv_s;
            half g_l = tg_g;
            half be_l = tg_be;
            half a_l = tg_a;
            half inv_a_l = tg_inv_a;

            uint in_base = (b * in_channels + ic) * in_len;

            for (uint k = 0; k < FC_KERNEL_SIZE; k++) {
                int t_in = int(t) + int(k) * int(FC_DILATION) - int(FC_PADDING);
                if (t_in >= 0 && uint(t_in) < in_len) {
                    // Normalize in float, convert to half for activation+conv.
                    float x_f = float(input[in_base + uint(t_in)]);
                    float normed_f = (x_f - mean_l) * inv_s_l;
                    half normed = half(normed_f);
                    half y = (1.0h + g_l) * normed + be_l;
                    half sin_val = sin(a_l * y);
                    half activated = y + inv_a_l * sin_val * sin_val;
                    acc += activated * tg_w[k];
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (valid) {
        uint out_idx = (b * out_channels + oc) * out_len + t;
        if (has_residual != 0) {
            acc = (acc + residual[out_idx]) * half(residual_scale);
        }
        output[out_idx] = acc;
    }
}
"#,
    )
}

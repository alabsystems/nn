// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! MSL source for 2-D pooling kernels (max pool, avg pool, adaptive avg pool).
//!
//! Each kernel operates on NCHW layout: `[batch, channels, height, width]`.
//! One thread per output element. Constants are passed via `set_bytes`.

/// MSL source for the 2-D max pooling kernel.
///
/// Buffers:
///   - 0: `input` — `[B, C, H, W]` (read-only)
///   - 1: `output` — `[B, C, out_H, out_W]` (write-only)
///
/// Constants (set_bytes):
///   - 2: `params` — struct { batch, channels, in_h, in_w, out_h, out_w, kernel_size, stride, padding }
///
/// Dispatch: one thread per output element (batch * channels * out_h * out_w).
pub(super) fn max_pool2d_msl(scalar_type: &str) -> String {
    format!(
        r#"
#include <metal_stdlib>
using namespace metal;

struct Pool2dParams {{
    uint batch;
    uint channels;
    uint in_h;
    uint in_w;
    uint out_h;
    uint out_w;
    uint kernel_size;
    uint stride;
    uint padding;
    uint total_elements;
}};

kernel void max_pool2d_{scalar_type}(
    device const {scalar_type}* input   [[buffer(0)]],
    device {scalar_type}* output        [[buffer(1)]],
    constant Pool2dParams& params       [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {{
    if (gid >= params.total_elements) return;

    uint ow = gid % params.out_w;
    uint tmp = gid / params.out_w;
    uint oh = tmp % params.out_h;
    tmp = tmp / params.out_h;
    uint c = tmp % params.channels;
    uint b = tmp / params.channels;

    float max_val = -INFINITY;
    uint base = (b * params.channels + c) * params.in_h * params.in_w;

    for (uint kh = 0; kh < params.kernel_size; kh++) {{
        uint ih_padded = oh * params.stride + kh;
        if (ih_padded < params.padding || ih_padded - params.padding >= params.in_h) continue;
        uint ih = ih_padded - params.padding;
        for (uint kw = 0; kw < params.kernel_size; kw++) {{
            uint iw_padded = ow * params.stride + kw;
            if (iw_padded < params.padding || iw_padded - params.padding >= params.in_w) continue;
            uint iw = iw_padded - params.padding;
            float val = (float)input[base + ih * params.in_w + iw];
            max_val = max(max_val, val);
        }}
    }}

    output[gid] = ({scalar_type})max_val;
}}
"#
    )
}

/// MSL source for the 2-D average pooling kernel.
///
/// count_include_pad=false (matching PyTorch default): padding positions
/// are excluded from the averaging count.
///
/// Buffers:
///   - 0: `input` — `[B, C, H, W]` (read-only)
///   - 1: `output` — `[B, C, out_H, out_W]` (write-only)
///
/// Constants (set_bytes):
///   - 2: `params` — struct { batch, channels, in_h, in_w, out_h, out_w, kernel_size, stride, padding }
///
/// Dispatch: one thread per output element.
pub(super) fn avg_pool2d_msl(scalar_type: &str) -> String {
    format!(
        r#"
#include <metal_stdlib>
using namespace metal;

struct Pool2dParams {{
    uint batch;
    uint channels;
    uint in_h;
    uint in_w;
    uint out_h;
    uint out_w;
    uint kernel_size;
    uint stride;
    uint padding;
    uint total_elements;
}};

kernel void avg_pool2d_{scalar_type}(
    device const {scalar_type}* input   [[buffer(0)]],
    device {scalar_type}* output        [[buffer(1)]],
    constant Pool2dParams& params       [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {{
    if (gid >= params.total_elements) return;

    uint ow = gid % params.out_w;
    uint tmp = gid / params.out_w;
    uint oh = tmp % params.out_h;
    tmp = tmp / params.out_h;
    uint c = tmp % params.channels;
    uint b = tmp / params.channels;

    float sum = 0.0;
    uint count = 0;
    uint base = (b * params.channels + c) * params.in_h * params.in_w;

    for (uint kh = 0; kh < params.kernel_size; kh++) {{
        uint ih_padded = oh * params.stride + kh;
        if (ih_padded < params.padding || ih_padded - params.padding >= params.in_h) continue;
        uint ih = ih_padded - params.padding;
        for (uint kw = 0; kw < params.kernel_size; kw++) {{
            uint iw_padded = ow * params.stride + kw;
            if (iw_padded < params.padding || iw_padded - params.padding >= params.in_w) continue;
            uint iw = iw_padded - params.padding;
            sum += float(input[base + ih * params.in_w + iw]);
            count++;
        }}
    }}

    output[gid] = ({scalar_type})(count > 0 ? sum / float(count) : 0.0f);
}}
"#
    )
}

/// MSL source for adaptive 2-D average pooling kernel.
///
/// Computes window boundaries using the PyTorch ATen formula:
///   start_h = (oh * in_h) / out_h
///   end_h   = ((oh + 1) * in_h + out_h - 1) / out_h  (ceil division)
///
/// Buffers:
///   - 0: `input` — `[B, C, H, W]` (read-only)
///   - 1: `output` — `[B, C, out_H, out_W]` (write-only)
///
/// Constants (set_bytes):
///   - 2: `params` — struct { batch, channels, in_h, in_w, out_h, out_w }
///
/// Dispatch: one thread per output element.
pub(super) fn adaptive_avg_pool2d_msl(scalar_type: &str) -> String {
    format!(
        r#"
#include <metal_stdlib>
using namespace metal;

struct AdaptivePool2dParams {{
    uint batch;
    uint channels;
    uint in_h;
    uint in_w;
    uint out_h;
    uint out_w;
    uint total_elements;
}};

kernel void adaptive_avg_pool2d_{scalar_type}(
    device const {scalar_type}* input   [[buffer(0)]],
    device {scalar_type}* output        [[buffer(1)]],
    constant AdaptivePool2dParams& params [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {{
    if (gid >= params.total_elements) return;

    uint ow = gid % params.out_w;
    uint tmp = gid / params.out_w;
    uint oh = tmp % params.out_h;
    tmp = tmp / params.out_h;
    uint c = tmp % params.channels;
    uint b = tmp / params.channels;

    // PyTorch ATen adaptive pooling window boundaries.
    uint start_h = (oh * params.in_h) / params.out_h;
    uint end_h   = ((oh + 1) * params.in_h + params.out_h - 1) / params.out_h;
    uint start_w = (ow * params.in_w) / params.out_w;
    uint end_w   = ((ow + 1) * params.in_w + params.out_w - 1) / params.out_w;

    float sum = 0.0;
    uint count = 0;
    uint base = (b * params.channels + c) * params.in_h * params.in_w;

    for (uint ih = start_h; ih < end_h; ih++) {{
        for (uint iw = start_w; iw < end_w; iw++) {{
            sum += float(input[base + ih * params.in_w + iw]);
            count++;
        }}
    }}

    output[gid] = ({scalar_type})(count > 0 ? sum / float(count) : 0.0f);
}}
"#
    )
}

// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! MSL source for fused Add + LayerNorm kernel.
//!
//! Computes `LayerNorm(a + b, weight, bias, eps)` in a single Metal dispatch
//! without materializing the intermediate `a + b` tensor. Based on the
//! two-pass Kahan-compensated reduction from `dyn_tensor_metal_welford_msl.rs`
//! but reads from two input buffers (`a` + `b`) instead of one.
//! Part of #1815 Tier 5 D2.

/// MSL source for the fused Add + LayerNorm kernel.
///
/// Buffers:
///   - 0: `a` — `[rows, hidden_dim]` (read-only, residual)
///   - 1: `b` — `[rows, hidden_dim]` (read-only, new value)
///   - 2: `weight` — `[hidden_dim]` (read-only, LayerNorm scale)
///   - 3: `bias` — `[hidden_dim]` (read-only, LayerNorm shift)
///   - 4: `output` — `[rows, hidden_dim]` (write-only)
///
/// Constants (set_bytes):
///   - 5: `hidden_dim` — uint
///   - 6: `eps` — float
///
/// Dispatch: one threadgroup per row, 256 threads per threadgroup.
///
/// `scalar_type` controls I/O pointer dtype: `"float"` or `"half"`.
/// Accumulators are always `float` for precision.
pub(super) fn fused_add_layer_norm_msl(scalar_type: &str) -> String {
    let tg_size = 256;

    format!(
        r#"
#include <metal_stdlib>
using namespace metal;

kernel void fused_add_layer_norm_{scalar_type}(
    device const {scalar_type}* a          [[buffer(0)]],
    device const {scalar_type}* b          [[buffer(1)]],
    device const {scalar_type}* weight     [[buffer(2)]],
    device const {scalar_type}* bias       [[buffer(3)]],
    device {scalar_type}* output           [[buffer(4)]],
    constant uint& hidden_dim      [[buffer(5)]],
    constant float& eps            [[buffer(6)]],
    uint gid [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]]
) {{
    uint base = gid * hidden_dim;

    // --- Two-pass Kahan-compensated mean + variance (adapted from #2697) ---
    // Reads float(a[i]) + float(b[i]) instead of input[i].
    threadgroup float shared_val[{tg_size}];
    threadgroup float shared_comp[{tg_size}];

    // ---- Pass 1: Kahan-compensated sum for mean ----
    float local_sum = 0.0f;
    float local_comp = 0.0f;
    for (uint i = tid; i < hidden_dim; i += tg_size) {{
        float val = float(a[base + i]) + float(b[base + i]);
        float y = val - local_comp;
        float t = local_sum + y;
        local_comp = (t - local_sum) - y;
        local_sum = t;
    }}
    shared_val[tid] = local_sum;
    shared_comp[tid] = local_comp;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = {tg_size} / 2; stride > 0; stride >>= 1) {{
        if (tid < stride) {{
            float a_val = shared_val[tid];
            float a_comp = shared_comp[tid];
            float b_val = shared_val[tid + stride];
            float b_comp = shared_comp[tid + stride];
            float y = b_val - (a_comp + b_comp);
            float t = a_val + y;
            shared_comp[tid] = (t - a_val) - y;
            shared_val[tid] = t;
        }}
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }}
    float mean = shared_val[0] / max(float(hidden_dim), 1.0f);

    // ---- Pass 2: Kahan-compensated sum of (val - mean)² ----
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float local_var = 0.0f;
    float local_var_comp = 0.0f;
    for (uint i = tid; i < hidden_dim; i += tg_size) {{
        float val = float(a[base + i]) + float(b[base + i]);
        float diff = val - mean;
        float diff_sq = diff * diff;
        float y = diff_sq - local_var_comp;
        float t = local_var + y;
        local_var_comp = (t - local_var) - y;
        local_var = t;
    }}
    shared_val[tid] = local_var;
    shared_comp[tid] = local_var_comp;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = {tg_size} / 2; stride > 0; stride >>= 1) {{
        if (tid < stride) {{
            float a_val = shared_val[tid];
            float a_comp = shared_comp[tid];
            float b_val = shared_val[tid + stride];
            float b_comp = shared_comp[tid + stride];
            float y = b_val - (a_comp + b_comp);
            float t = a_val + y;
            shared_comp[tid] = (t - a_val) - y;
            shared_val[tid] = t;
        }}
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }}
    float variance = shared_val[0] / max(float(hidden_dim), 1.0f);
    float inv_std = metal::precise::rsqrt(variance + eps);
    // --- end two-pass Kahan reduction ---

    // LayerNorm affine: output = (a+b - mean) * inv_std * weight + bias
    for (uint i = tid; i < hidden_dim; i += tg_size) {{
        float val = float(a[base + i]) + float(b[base + i]);
        float normed = (val - mean) * inv_std;
        output[base + i] = {scalar_type}(normed * float(weight[i]) + float(bias[i]));
    }}
}}
"#,
    )
}

// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! MSL source for fused AdaLayerNorm kernel (#2482).
//!
//! Uses two-pass Kahan-compensated mean+variance reduction (#2697).
//! Default algorithm selected via `DEFAULT_NORM_REDUCTION`.
//!
//! Input x is pre-reshaped to `[rows, hidden_dim]` where rows = B*T.
//! gamma/beta are `[B, hidden_dim]` (from `[B, 1, C]` squeezed).
//! norm_weight/norm_bias are `[hidden_dim]` (per-channel LayerNorm params).

use super::super::welford_msl;

/// MSL source for the fused AdaLayerNorm kernel.
///
/// `scalar_type` controls I/O pointer dtype: `"float"` or `"half"`.
/// Accumulators are always `float` for precision. Part of #2981 F16 Tier 2.
pub(super) fn fused_ada_layer_norm_msl(scalar_type: &str) -> String {
    let algo = welford_msl::DEFAULT_NORM_REDUCTION;
    let preamble = welford_msl::norm_preamble_msl(algo);
    let reduction = welford_msl::norm_reduction_msl(algo, "hidden_dim", 256);

    format!(
        r#"
#include <metal_stdlib>
using namespace metal;

{preamble}

kernel void fused_ada_layer_norm_{scalar_type}(
    device const {scalar_type}* input       [[buffer(0)]],
    device const {scalar_type}* gamma       [[buffer(1)]],
    device const {scalar_type}* beta        [[buffer(2)]],
    device const {scalar_type}* norm_weight [[buffer(3)]],
    device const {scalar_type}* norm_bias   [[buffer(4)]],
    device {scalar_type}* output            [[buffer(5)]],
    constant uint& hidden_dim       [[buffer(6)]],
    constant uint& time_steps       [[buffer(7)]],
    constant float& eps             [[buffer(8)]],
    uint gid [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]]
) {{
    uint base = gid * hidden_dim;

{reduction}

    // Compute batch index: row gid = b*T + t, so b = gid / T.
    uint batch_idx = gid / time_steps;

    // LayerNorm + adaptive affine.
    // normed = (x - mean) * inv_std * norm_weight + norm_bias
    // output = (1 + gamma) * normed + beta
    for (uint i = tid; i < hidden_dim; i += tg_size) {{
        float normed = (float(input[base + i]) - mean) * inv_std;
        normed = normed * float(norm_weight[i]) + float(norm_bias[i]);
        float g = float(gamma[batch_idx * hidden_dim + i]);
        float b = float(beta[batch_idx * hidden_dim + i]);
        output[base + i] = {scalar_type}((1.0f + g) * normed + b);
    }}
}}
"#
    )
}

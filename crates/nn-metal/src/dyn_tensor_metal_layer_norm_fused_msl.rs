// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! MSL source for fused LayerNorm kernel.
//!
//! Uses simd-accelerated two-pass Kahan-compensated mean+variance reduction.
//! 4 barriers per norm (2 per pass via simd_sum + cross-simdgroup reduction)
//! instead of 16 barriers from the tree-based path.
//!
//! Input x is pre-reshaped to `[rows, hidden_dim]` where rows = product of
//! all dimensions except the last.
//! weight/bias are `[hidden_dim]` (per-channel LayerNorm params).

use super::super::welford_msl;

/// MSL source for the fused LayerNorm kernel.
///
/// Buffers:
///   - 0: `input` — `[rows, hidden_dim]` (read-only)
///   - 1: `weight` — `[hidden_dim]` (read-only, LayerNorm scale)
///   - 2: `bias` — `[hidden_dim]` (read-only, LayerNorm shift)
///   - 3: `output` — `[rows, hidden_dim]` (write-only)
///
/// Constants (set_bytes):
///   - 4: `hidden_dim` — uint
///   - 5: `eps` — float
///
/// Dispatch: one threadgroup per row, 256 threads per threadgroup.
///
/// Uses simd-accelerated reduction: `simd_sum()` within each 32-thread
/// simdgroup + shared-memory cross-simdgroup merge. 4 barriers total
/// (vs 16 for tree-based). Threadgroup memory: 128B (vs 2048B).
///
/// `scalar_type` controls I/O pointer dtype: `"float"` or `"half"`.
/// Accumulators are always `float` for precision. Part of #2981 F16 Tier 2.
pub(super) fn fused_layer_norm_msl(scalar_type: &str) -> String {
    let preamble = welford_msl::simd_reduction_helpers_msl();
    let reduction = welford_msl::kahan_two_pass_simd_reduction_msl("hidden_dim");

    format!(
        r#"
#include <metal_stdlib>
using namespace metal;

{preamble}

kernel void fused_layer_norm_{scalar_type}(
    device const {scalar_type}* input      [[buffer(0)]],
    device const {scalar_type}* weight     [[buffer(1)]],
    device const {scalar_type}* bias       [[buffer(2)]],
    device {scalar_type}* output           [[buffer(3)]],
    constant uint& hidden_dim      [[buffer(4)]],
    constant float& eps            [[buffer(5)]],
    uint gid [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]]
) {{
    uint base = gid * hidden_dim;

{reduction}

    // LayerNorm affine: normed = (x - mean) * inv_std * weight + bias
    for (uint i = tid; i < hidden_dim; i += tg_size) {{
        float normed = (float(input[base + i]) - mean) * inv_std;
        output[base + i] = {scalar_type}(normed * float(weight[i]) + float(bias[i]));
    }}
}}
"#
    )
}

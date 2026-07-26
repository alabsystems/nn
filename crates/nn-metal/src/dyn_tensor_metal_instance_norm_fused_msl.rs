// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! MSL source for the fused InstanceNorm kernel (#2472).
//!
//! Uses simd-accelerated two-pass Kahan-compensated mean+variance reduction.
//! 4 barriers per norm (2 per pass via simd_sum + cross-simdgroup reduction)
//! instead of 16 barriers from the tree-based path.
//!
//! Input is pre-reshaped to `[B*C, spatial]` — one threadgroup per (B,C) pair.

use super::super::welford_msl;

/// MSL source for the fused InstanceNorm kernel.
///
/// Kernel: `fused_instance_norm_{scalar_type}` (f32 or f16)
///
/// Buffers:
///   - 0: `input` — `[rows, spatial_len]` (read-only)
///   - 1: `output` — `[rows, spatial_len]` (write-only)
///
/// Constants (set_bytes):
///   - 2: `spatial_len` — uint
///   - 3: `eps` — float
///
/// Dispatch: one threadgroup per row, 256 threads per threadgroup.
///
/// Uses `DEFAULT_NORM_REDUCTION` algorithm (currently `PyTorchCompat`)
/// to match PyTorch MPS InstanceNorm behavior. This is critical for
/// models like Kokoro where 35+ chained FusedResBlocks amplify tiny
/// per-layer differences. Previously used simd-accelerated Kahan
/// two-pass reduction, which introduced +35.8% amplitude divergence
/// vs PyTorch reference (#4335).
///
/// `scalar_type` controls I/O pointer dtype: `"float"` or `"half"`.
/// Accumulators (mean, variance, inv_std) are always `float` for precision.
/// Part of #2981 F16 Tier 2, #4335.
pub(super) fn fused_instance_norm_msl(scalar_type: &str) -> String {
    let algo = welford_msl::DEFAULT_NORM_REDUCTION;
    let preamble = welford_msl::norm_preamble_msl(algo);
    let reduction = welford_msl::norm_reduction_msl(algo, "spatial_len", 256);

    format!(
        r#"
#include <metal_stdlib>
using namespace metal;

{preamble}

kernel void fused_instance_norm_{scalar_type}(
    device const {scalar_type}* input  [[buffer(0)]],
    device {scalar_type}* output       [[buffer(1)]],
    constant uint& spatial_len [[buffer(2)]],
    constant float& eps        [[buffer(3)]],
    uint gid [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]]
) {{
    uint base = gid * spatial_len;

{reduction}

    // Normalize output.
    for (uint i = tid; i < spatial_len; i += tg_size) {{
        output[base + i] = {scalar_type}((float(input[base + i]) - mean) * inv_std);
    }}
}}
"#
    )
}

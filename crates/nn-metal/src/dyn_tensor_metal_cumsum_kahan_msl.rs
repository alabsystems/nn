// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! MSL source for Kahan-compensated f32 cumulative sum.
//!
//! One thread per (outer, inner) slice — sequential scan along axis with
//! Kahan compensation. Error bound: O(nε) vs O(n²ε) for naive f32.
//! Intended for small axis sizes (SineGen T_frames=126).
//!
//! Reference: Kahan (1965), Higham "Accuracy and Stability of Numerical
//! Algorithms" §4.3.
//!
//! Part of #2909.

pub(super) const CUMSUM_KAHAN_F32_MSL: &str = r#"
#include <metal_stdlib>
using namespace metal;

/// Kahan-compensated cumulative sum along one axis.
///
/// One thread per slice (outer_idx, inner_idx). Sequential scan along
/// axis_size elements with Kahan compensation to achieve O(nε) error.
///
/// Buffer layout (strided): element at (outer, axis_pos, inner) is at
/// `outer * (axis_size * inner_sz) + axis_pos * inner_sz + inner`.
kernel void cumsum_kahan_f32(
    device const float* input    [[buffer(0)]],
    device float* output         [[buffer(1)]],
    constant uint& total_slices  [[buffer(2)]],
    constant uint& axis_size     [[buffer(3)]],
    constant uint& inner_sz      [[buffer(4)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= total_slices) return;

    uint outer_idx = gid / inner_sz;
    uint inner_idx = gid % inner_sz;
    uint base = outer_idx * (axis_size * inner_sz) + inner_idx;
    uint stride = inner_sz;

    float sum = 0.0f;
    float compensation = 0.0f;

    for (uint i = 0; i < axis_size; i++) {
        float y = input[base + i * stride] - compensation;
        float t = sum + y;
        compensation = (t - sum) - y;
        sum = t;
        output[base + i * stride] = sum;
    }
}
"#;

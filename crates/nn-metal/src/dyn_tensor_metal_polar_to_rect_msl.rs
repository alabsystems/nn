// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! MSL source for the fused polar-to-rectangular kernel (#2491).
//!
//! A single Metal compute kernel that converts (magnitude, phase) to
//! (real, imag) using `sincos()` intrinsic:
//!   `real = magnitude * cos(phase)`
//!   `imag = magnitude * sin(phase)`
//!
//! Replaces 4 separate dispatches (cos, sin, mul, mul) with one.
//! Used in the Kokoro iSTFT path.

/// MSL source for the fused polar-to-rectangular f32 kernel.
///
/// Kernel: `fused_polar_to_rect_f32`
///
/// Buffers:
///   - 0: `magnitude` — `[N]` f32 (read-only)
///   - 1: `phase` — `[N]` f32 (read-only)
///   - 2: `real_out` — `[N]` f32 (write-only)
///   - 3: `imag_out` — `[N]` f32 (write-only)
///
/// Constants (set_bytes):
///   - 4: `count` — uint (total element count)
///
/// Dispatch: ceil(count / 256) threadgroups, 256 threads per threadgroup.
/// No threadgroup memory required (pure elementwise).
pub(super) fn fused_polar_to_rect_msl() -> &'static str {
    r#"
#include <metal_stdlib>
using namespace metal;

kernel void fused_polar_to_rect_f32(
    device const float* magnitude [[buffer(0)]],
    device const float* phase     [[buffer(1)]],
    device float* real_out        [[buffer(2)]],
    device float* imag_out        [[buffer(3)]],
    constant uint& count          [[buffer(4)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= count) return;
    float mag = magnitude[gid];
    float c;
    float s = sincos(phase[gid], c);
    real_out[gid] = mag * c;
    imag_out[gid] = mag * s;
}
"#
}

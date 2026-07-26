// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! MSL source generation for GPU iSTFT (inverse Short-Time Fourier Transform).
//!
//! Two kernels:
//! 1. **IDFT per-frame** — each threadgroup handles one STFT frame, computing
//!    `n_fft` time-domain samples via matmul with the pre-computed DFT basis.
//!    This is the main computational bottleneck: O(n_frames × n_fft × n_bins).
//! 2. **Overlap-add + COLA** — each thread handles one output sample,
//!    accumulating windowed frame contributions and normalizing by the
//!    window sum-of-squares.
//!
//! Part of #1393.

/// MSL source for the per-frame inverse DFT kernel.
///
/// Kernel: `istft_idft_f32`
///
/// For each frame `t` and time-domain sample `k`:
///   frame[t, k] = norm * Σ_f (real[f, t] * cos_basis[f, k] - imag[f, t] * sin_basis[f, k])
///
/// Interior frequencies (f ∈ 1..n_bins-2) are doubled for conjugate symmetry.
///
/// Buffers:
///   0: real       — [n_bins, n_frames] row-major
///   1: imag       — [n_bins, n_frames] row-major
///   2: cos_basis  — [n_bins, n_fft] row-major
///   3: sin_basis  — [n_bins, n_fft] row-major
///   4: output     — [n_frames, n_fft] row-major
///   5: n_bins     — uint
///   6: n_frames   — uint
///   7: n_fft      — uint
///   8: norm       — float (1/sqrt(n_fft) or 1/n_fft)
///
/// Grid: [n_fft, n_frames, 1] — one thread per (k, t) pair.
pub(super) fn idft_msl() -> String {
    r#"
#include <metal_stdlib>
using namespace metal;

kernel void istft_idft_f32(
    device const float* real       [[buffer(0)]],
    device const float* imag       [[buffer(1)]],
    device const float* cos_basis  [[buffer(2)]],
    device const float* sin_basis  [[buffer(3)]],
    device float*       output     [[buffer(4)]],
    device const uint&  n_bins_v   [[buffer(5)]],
    device const uint&  n_frames_v [[buffer(6)]],
    device const uint&  n_fft_v    [[buffer(7)]],
    device const float& norm_v     [[buffer(8)]],
    uint2 tid [[thread_position_in_grid]]
) {
    uint k = tid.x;  // time-domain sample index within frame
    uint t = tid.y;  // frame index

    if (k >= n_fft_v || t >= n_frames_v) return;

    float sum = 0.0f;

    // DC component (f=0): no mirror.
    float r0 = real[t];  // real[0 * n_frames + t]
    float i0 = imag[t];
    sum += r0 * cos_basis[k] - i0 * sin_basis[k];  // cos_basis[0 * n_fft + k]

    // Interior frequencies (f=1..n_bins-2): doubled for conjugate symmetry.
    uint last_bin = n_bins_v - 1;
    for (uint f = 1; f < last_bin; f++) {
        float rf = real[f * n_frames_v + t];
        float imf = imag[f * n_frames_v + t];
        float cv = cos_basis[f * n_fft_v + k];
        float sv = sin_basis[f * n_fft_v + k];
        sum += 2.0f * (rf * cv - imf * sv);
    }

    // Nyquist component (f=n_bins-1): no mirror.
    float rn = real[last_bin * n_frames_v + t];
    float imn = imag[last_bin * n_frames_v + t];
    sum += rn * cos_basis[last_bin * n_fft_v + k]
         - imn * sin_basis[last_bin * n_fft_v + k];

    output[t * n_fft_v + k] = sum * norm_v;
}
"#
    .to_string()
}

/// MSL source for the fused polar→iSTFT kernel.
///
/// Kernel: `istft_fused_polar_f32`
///
/// Combines polar-to-rectangular conversion, per-frame IDFT, windowed
/// overlap-add, and COLA normalization into a single dispatch.
/// Eliminates 2 dispatches and 3 intermediate buffers vs the separate
/// polar_to_rect + IDFT + overlap-add path.
///
/// For each output sample `i`, determines contributing frames, computes
/// the IDFT inline from (magnitude, phase) using `sincos()`, and
/// accumulates the windowed overlap-add result.
///
/// Buffers:
///   0: magnitude  — [n_bins, n_frames] row-major
///   1: phase      — [n_bins, n_frames] row-major
///   2: cos_basis  — [n_bins, n_fft] row-major
///   3: sin_basis  — [n_bins, n_fft] row-major
///   4: window     — [n_fft]
///   5: output     — [full_len]
///   6: n_bins     — uint
///   7: n_frames   — uint
///   8: n_fft      — uint
///   9: hop_length — uint
///  10: full_len   — uint
///  11: norm       — float
///
/// Grid: [full_len, 1, 1] — one thread per output sample.
///
/// Part of iSTFT fusion (#3351).
pub(super) fn fused_polar_istft_msl() -> String {
    r#"
#include <metal_stdlib>
using namespace metal;

kernel void istft_fused_polar_f32(
    device const float* magnitude [[buffer(0)]],
    device const float* phase     [[buffer(1)]],
    device const float* cos_basis [[buffer(2)]],
    device const float* sin_basis [[buffer(3)]],
    device const float* window    [[buffer(4)]],
    device float*       output    [[buffer(5)]],
    device const uint&  n_bins_v   [[buffer(6)]],
    device const uint&  n_frames_v [[buffer(7)]],
    device const uint&  n_fft_v    [[buffer(8)]],
    device const uint&  hop_v      [[buffer(9)]],
    device const uint&  full_len_v [[buffer(10)]],
    device const float& norm_v     [[buffer(11)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= full_len_v) return;

    float signal_sum = 0.0f;
    float window_sq_sum = 0.0f;

    // Determine which frames contribute to this output sample.
    uint t_max = min(tid / hop_v, n_frames_v - 1);
    uint t_min = 0;
    if (tid + 1 > n_fft_v) {
        t_min = (tid + 1 - n_fft_v + hop_v - 1) / hop_v;
    }

    uint last_bin = n_bins_v - 1;

    for (uint t = t_min; t <= t_max; t++) {
        uint k = tid - t * hop_v;
        if (k >= n_fft_v) continue;

        float w = window[k];

        // Inline IDFT: compute frame[t, k] from magnitude + phase.
        float frame_val = 0.0f;

        // DC component (f=0): no conjugate mirror.
        // Guard: skip sincos when magnitude is zero to prevent NaN
        // propagation from invalid phase (0 * NaN = NaN in IEEE 754).
        {
            float mag0 = magnitude[t];
            if (mag0 != 0.0f) {
                float c0, s0;
                s0 = sincos(phase[t], c0);
                frame_val += mag0 * c0 * cos_basis[k]
                           - mag0 * s0 * sin_basis[k];
            }
        }

        // Interior frequencies (f=1..n_bins-2): doubled for conjugate symmetry.
        for (uint f = 1; f < last_bin; f++) {
            uint spec_idx = f * n_frames_v + t;
            float mag_f = magnitude[spec_idx];
            if (mag_f == 0.0f) continue;
            float cf, sf;
            sf = sincos(phase[spec_idx], cf);
            float rf = mag_f * cf;
            float imf = mag_f * sf;
            uint basis_idx = f * n_fft_v + k;
            frame_val += 2.0f * (rf * cos_basis[basis_idx]
                               - imf * sin_basis[basis_idx]);
        }

        // Nyquist component (f=n_bins-1): no conjugate mirror.
        {
            uint spec_idx = last_bin * n_frames_v + t;
            float mag_n = magnitude[spec_idx];
            if (mag_n != 0.0f) {
                float cn, sn;
                sn = sincos(phase[spec_idx], cn);
                uint basis_idx = last_bin * n_fft_v + k;
                frame_val += mag_n * cn * cos_basis[basis_idx]
                           - mag_n * sn * sin_basis[basis_idx];
            }
        }

        frame_val *= norm_v;
        signal_sum += frame_val * w;
        window_sq_sum += w * w;
    }

    // COLA normalization with epsilon guard.
    float eps = 1e-11f;
    if (window_sq_sum > eps) {
        output[tid] = signal_sum / window_sq_sum;
    } else {
        output[tid] = 0.0f;
    }
}
"#
    .to_string()
}

/// MSL source for the windowed overlap-add + COLA normalization kernel.
///
/// Kernel: `istft_overlap_add_f32`
///
/// Each thread handles one output sample `i`:
///   output[i] = Σ_t (frame[t, i - t*hop] * window[i - t*hop]) / Σ_t (window[i - t*hop]^2)
///
/// Only frames where `t*hop <= i < t*hop + n_fft` contribute.
///
/// Buffers:
///   0: frames     — [n_frames, n_fft] row-major (from IDFT kernel)
///   1: window     — [n_fft]
///   2: output     — [full_len]
///   3: n_frames   — uint
///   4: n_fft      — uint
///   5: hop_length  — uint
///   6: full_len   — uint
///
/// Grid: [full_len, 1, 1] — one thread per output sample.
pub(super) fn overlap_add_msl() -> String {
    r#"
#include <metal_stdlib>
using namespace metal;

kernel void istft_overlap_add_f32(
    device const float* frames     [[buffer(0)]],
    device const float* window     [[buffer(1)]],
    device float*       output     [[buffer(2)]],
    device const uint&  n_frames_v [[buffer(3)]],
    device const uint&  n_fft_v    [[buffer(4)]],
    device const uint&  hop_v      [[buffer(5)]],
    device const uint&  full_len_v [[buffer(6)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= full_len_v) return;

    float signal_sum = 0.0f;
    float window_sq_sum = 0.0f;

    // Determine which frames contribute to this output sample.
    // Frame t contributes if t*hop <= tid < t*hop + n_fft,
    // i.e., k = tid - t*hop is in [0, n_fft).
    //
    // t_min = max(0, ceil((tid + 1 - n_fft) / hop))
    // t_max = min(n_frames - 1, tid / hop)
    uint t_max = min(tid / hop_v, n_frames_v - 1);
    uint t_min = 0;
    if (tid + 1 > n_fft_v) {
        // ceil((tid + 1 - n_fft) / hop)
        t_min = (tid + 1 - n_fft_v + hop_v - 1) / hop_v;
    }

    for (uint t = t_min; t <= t_max; t++) {
        uint k = tid - t * hop_v;
        if (k < n_fft_v) {
            float w = window[k];
            signal_sum += frames[t * n_fft_v + k] * w;
            window_sq_sum += w * w;
        }
    }

    // COLA normalization with epsilon guard.
    float eps = 1e-11f;
    if (window_sq_sum > eps) {
        output[tid] = signal_sum / window_sq_sum;
    } else {
        output[tid] = 0.0f;
    }
}
"#
    .to_string()
}

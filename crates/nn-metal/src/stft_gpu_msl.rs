// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! MSL source generation for GPU forward STFT (Short-Time Fourier Transform).
//!
//! Single kernel: **forward DFT per-frame** — each thread computes one
//! (frequency bin, frame) pair, producing magnitude and phase output.
//!
//! The DFT basis is pre-multiplied with the Hann window on the CPU side,
//! so the kernel only needs a dot product per (f, t) pair.
//!
//! Part of #2218.

/// MSL source for the per-frame forward DFT kernel (dot-product approach).
///
/// **Known limitation:** DFT-matmul produces ~4.8% phase wrapping at ±π atan2
/// boundary vs butterfly FFT ~0.002%. Production uses [`fft_msl`] instead.
///
/// Kernel: `stft_dft_f32`
///
/// Grid: [n_bins, n_frames, 1] — one thread per (f, t) pair.
pub(super) fn dft_msl() -> String {
    r#"
#include <metal_stdlib>
using namespace metal;

kernel void stft_dft_f32(
    device const float* signal         [[buffer(0)]],
    device const float* windowed_cos   [[buffer(1)]],
    device const float* windowed_sin   [[buffer(2)]],
    device float*       magnitude      [[buffer(3)]],
    device float*       phase          [[buffer(4)]],
    device const uint&  n_bins_v       [[buffer(5)]],
    device const uint&  n_frames_v     [[buffer(6)]],
    device const uint&  n_fft_v        [[buffer(7)]],
    device const uint&  hop_v          [[buffer(8)]],
    uint2 tid [[thread_position_in_grid]]
) {
    uint f = tid.x;  // frequency bin index
    uint t = tid.y;  // frame index

    if (f >= n_bins_v || t >= n_frames_v) return;

    uint frame_start = t * hop_v;
    uint basis_row = f * n_fft_v;

    float real_sum = 0.0f;
    float imag_sum = 0.0f;

    for (uint k = 0; k < n_fft_v; k++) {
        float s = signal[frame_start + k];
        real_sum += s * windowed_cos[basis_row + k];
        imag_sum += s * windowed_sin[basis_row + k];
    }

    // Forward DFT: imag = -Σ s * sin (negative sign convention).
    float imag = -imag_sum;

    uint out_idx = f * n_frames_v + t;
    float mag = sqrt(real_sum * real_sum + imag * imag);
    magnitude[out_idx] = mag;
    phase[out_idx] = (mag == 0.0f) ? 0.0f : atan2(imag, real_sum);
}
"#
    .to_string()
}

/// MSL source for the GPU mixed-radix FFT kernel (butterfly approach).
///
/// Kernel: `stft_fft_f32`
///
/// Uses Good-Thomas PFA (Prime Factor Algorithm) for N=20 = 4×5. Two stages:
///  1. Four independent 5-point DFTs (radix-5 butterflies)
///  2. Five independent 4-point DFTs (radix-4 butterflies)
///
/// No twiddle factors needed between stages (Good-Thomas property for coprime
/// factors). This produces intermediate rounding in the same family as rustfft
/// and PyTorch pocketfft, eliminating the ±π phase wrapping that causes the
/// -21% amplitude deficit (#2928).
///
/// One thread per STFT frame. Each thread computes the full 20-point FFT in
/// registers (20 float2 = 160 bytes — well within Apple Silicon register limits).
///
/// Buffers:
///   0: signal    — `[T_padded]` flat (single batch)
///   1: window    — `[20]` Hann window (NOT pre-multiplied into basis)
///   2: magnitude — `[n_bins, n_frames]` row-major (output)
///   3: phase     — `[n_bins, n_frames]` row-major (output)
///   4: n_bins    — uint
///   5: n_frames  — uint
///   6: hop       — uint
///
/// Grid: `[n_frames, 1, 1]` — one thread per frame.
pub(super) fn fft_msl() -> String {
    r#"
#include <metal_stdlib>
using namespace metal;

// Complex multiply: (a.x + j*a.y) * (b.x + j*b.y).
static inline float2 cmul(float2 a, float2 b) {
    return float2(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}

// In-place 5-point DFT via direct butterfly.
// W5^k = exp(-2πi*k/5) twiddle factors as compile-time constants.
static void dft5(thread float2* a) {
    const float2 W1 = float2( 0.30901699f, -0.95105652f);
    const float2 W2 = float2(-0.80901699f, -0.58778525f);
    const float2 W3 = float2(-0.80901699f,  0.58778525f);
    const float2 W4 = float2( 0.30901699f,  0.95105652f);

    float2 x0 = a[0], x1 = a[1], x2 = a[2], x3 = a[3], x4 = a[4];
    a[0] = x0 + x1 + x2 + x3 + x4;
    a[1] = x0 + cmul(x1, W1) + cmul(x2, W2) + cmul(x3, W3) + cmul(x4, W4);
    a[2] = x0 + cmul(x1, W2) + cmul(x2, W4) + cmul(x3, W1) + cmul(x4, W3);
    a[3] = x0 + cmul(x1, W3) + cmul(x2, W1) + cmul(x3, W4) + cmul(x4, W2);
    a[4] = x0 + cmul(x1, W4) + cmul(x2, W3) + cmul(x3, W2) + cmul(x4, W1);
}

// In-place 4-point DFT. W4^1 = -j, W4^2 = -1, W4^3 = +j.
static void dft4(thread float2* a) {
    float2 x0 = a[0], x1 = a[1], x2 = a[2], x3 = a[3];
    a[0] = x0 + x1 + x2 + x3;
    a[1] = float2(x0.x + x1.y - x2.x - x3.y,
                   x0.y - x1.x - x2.y + x3.x);
    a[2] = x0 - x1 + x2 - x3;
    a[3] = float2(x0.x - x1.y - x2.x + x3.y,
                   x0.y + x1.x - x2.y - x3.x);
}

kernel void stft_fft_f32(
    device const float* signal      [[buffer(0)]],
    device const float* window      [[buffer(1)]],
    device float*       magnitude   [[buffer(2)]],
    device float*       phase       [[buffer(3)]],
    device const uint&  n_bins_v    [[buffer(4)]],
    device const uint&  n_frames_v  [[buffer(5)]],
    device const uint&  hop_v       [[buffer(6)]],
    uint tid [[thread_position_in_grid]]
) {
    uint frame = tid;
    if (frame >= n_frames_v) return;

    uint frame_start = frame * hop_v;

    // Load 20 windowed samples as complex (real-only, imag=0).
    float2 x[20];
    for (uint k = 0; k < 20; k++) {
        x[k] = float2(signal[frame_start + k] * window[k], 0.0f);
    }

    // Good-Thomas PFA for N=20 = 4×5.
    // Input permutation: n = (5*n1 + 16*n2) mod 20.
    const uint ip[20] = {
         0, 16, 12,  8,  4,
         5,  1, 17, 13,  9,
        10,  6,  2, 18, 14,
        15, 11,  7,  3, 19
    };
    float2 a[4][5];
    for (uint n1 = 0; n1 < 4; n1++) {
        for (uint n2 = 0; n2 < 5; n2++) {
            a[n1][n2] = x[ip[n1 * 5 + n2]];
        }
    }

    // Stage 1: Four independent 5-point DFTs (across n2 dimension).
    for (uint n1 = 0; n1 < 4; n1++) {
        dft5(a[n1]);
    }

    // Stage 2: Five independent 4-point DFTs (across n1 dimension).
    // Good-Thomas: no twiddle factors between stages (4 and 5 are coprime).
    float2 col[4];
    for (uint n2 = 0; n2 < 5; n2++) {
        for (uint n1 = 0; n1 < 4; n1++) col[n1] = a[n1][n2];
        dft4(col);
        for (uint n1 = 0; n1 < 4; n1++) a[n1][n2] = col[n1];
    }

    // Output permutation: k = (5*k1 + 4*k2) mod 20.
    const uint op[20] = {
         0,  4,  8, 12, 16,
         5,  9, 13, 17,  1,
        10, 14, 18,  2,  6,
        15, 19,  3,  7, 11
    };
    float2 X[20];
    for (uint k1 = 0; k1 < 4; k1++) {
        for (uint k2 = 0; k2 < 5; k2++) {
            X[op[k1 * 5 + k2]] = a[k1][k2];
        }
    }

    // Extract n_bins positive frequencies: magnitude and phase.
    for (uint f = 0; f < n_bins_v; f++) {
        uint out_idx = f * n_frames_v + frame;
        float mag = sqrt(X[f].x * X[f].x + X[f].y * X[f].y);
        magnitude[out_idx] = mag;
        phase[out_idx] = (mag == 0.0f) ? 0.0f : atan2(X[f].y, X[f].x);
    }
}
"#
    .to_string()
}

// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! SIMD-optimized elementwise operations with explicit scalar/NEON/AVX2 entry points.
//!
//! This module provides named entry points for each SIMD tier:
//! - `*_f32_neon` — NEON-optimized (aarch64)
//! - `*_f32_avx2` — AVX2-optimized (x86_64)
//! - `*_f32_scalar` — pure scalar fallback
//! - `*_f32` — auto-dispatch to best available
//!
//! Operations:
//! - **Binary:** `add_f32`, `mul_f32`, `fma_f32` (a*b+c)
//! - **Scalar-broadcast:** `scalar_mul_f32` (a[i] * scalar)
//! - **Activations:** `relu_f32`, `gelu_f32`, `silu_f32`
//!
//! The existing `crate::elementwise` module provides the same operations;
//! this module exposes per-tier entry points for benchmarking and
//! differential testing.

// ---------------------------------------------------------------------------
// Scalar fallback (always compiled, no cfg gate)
// ---------------------------------------------------------------------------

/// Scalar elementwise add: `out[i] = a[i] + b[i]`.
#[inline]
pub fn add_f32_scalar(a: &[f32], b: &[f32], out: &mut [f32]) {
    assert_eq!(a.len(), b.len(), "a and b must have equal length");
    assert_eq!(a.len(), out.len(), "a and out must have equal length");
    for i in 0..a.len() {
        out[i] = a[i] + b[i];
    }
}

/// Scalar elementwise multiply: `out[i] = a[i] * b[i]`.
#[inline]
pub fn mul_f32_scalar(a: &[f32], b: &[f32], out: &mut [f32]) {
    assert_eq!(a.len(), b.len(), "a and b must have equal length");
    assert_eq!(a.len(), out.len(), "a and out must have equal length");
    for i in 0..a.len() {
        out[i] = a[i] * b[i];
    }
}

/// Scalar broadcast multiply: `out[i] = a[i] * scalar`.
#[inline]
pub fn scalar_mul_f32_scalar(a: &[f32], scalar: f32, out: &mut [f32]) {
    assert_eq!(a.len(), out.len(), "a and out must have equal length");
    for i in 0..a.len() {
        out[i] = a[i] * scalar;
    }
}

/// Scalar fused multiply-add: `out[i] = a[i] * b[i] + c[i]`.
#[inline]
pub fn fma_f32_scalar(a: &[f32], b: &[f32], c: &[f32], out: &mut [f32]) {
    assert_eq!(a.len(), b.len(), "a and b must have equal length");
    assert_eq!(a.len(), c.len(), "a and c must have equal length");
    assert_eq!(a.len(), out.len(), "a and out must have equal length");
    for i in 0..a.len() {
        out[i] = a[i] * b[i] + c[i];
    }
}

/// Scalar ReLU: `out[i] = max(0, x[i])`.
#[inline]
pub fn relu_f32_scalar(x: &[f32], out: &mut [f32]) {
    assert_eq!(x.len(), out.len(), "x and out must have equal length");
    for i in 0..x.len() {
        out[i] = x[i].max(0.0);
    }
}

/// Scalar GELU (tanh approximation):
/// `out[i] = x * 0.5 * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))`.
#[inline]
pub fn gelu_f32_scalar(x: &[f32], out: &mut [f32]) {
    assert_eq!(x.len(), out.len(), "x and out must have equal length");
    let c = (2.0_f32 / std::f32::consts::PI).sqrt();
    for i in 0..x.len() {
        let v = x[i];
        out[i] = v * 0.5 * (1.0 + (c * (v + 0.044715 * v * v * v)).tanh());
    }
}

/// Scalar SiLU (swish): `out[i] = x[i] / (1 + exp(-x[i]))`.
#[inline]
pub fn silu_f32_scalar(x: &[f32], out: &mut [f32]) {
    assert_eq!(x.len(), out.len(), "x and out must have equal length");
    for i in 0..x.len() {
        let v = x[i];
        out[i] = v / (1.0 + (-v).exp());
    }
}

// ---------------------------------------------------------------------------
// NEON (aarch64) — 128-bit, 4x f32 lanes
// ---------------------------------------------------------------------------

#[cfg(target_arch = "aarch64")]
pub fn add_f32_neon(a: &[f32], b: &[f32], out: &mut [f32]) {
    use std::arch::aarch64::*;

    assert_eq!(a.len(), b.len(), "a and b must have equal length");
    assert_eq!(a.len(), out.len(), "a and out must have equal length");
    let n = a.len();
    let chunks = n / 4;
    let remainder = n % 4;

    // SAFETY: aarch64 NEON is always available. Bounded loads/stores within slice.
    unsafe {
        for i in 0..chunks {
            let offset = i * 4;
            let va = vld1q_f32(a.as_ptr().add(offset));
            let vb = vld1q_f32(b.as_ptr().add(offset));
            let r = vaddq_f32(va, vb);
            vst1q_f32(out.as_mut_ptr().add(offset), r);
        }
    }
    let tail = chunks * 4;
    for i in 0..remainder {
        out[tail + i] = a[tail + i] + b[tail + i];
    }
}

#[cfg(not(target_arch = "aarch64"))]
pub fn add_f32_neon(a: &[f32], b: &[f32], out: &mut [f32]) {
    add_f32_scalar(a, b, out);
}

#[cfg(target_arch = "aarch64")]
pub fn mul_f32_neon(a: &[f32], b: &[f32], out: &mut [f32]) {
    use std::arch::aarch64::*;

    assert_eq!(a.len(), b.len(), "a and b must have equal length");
    assert_eq!(a.len(), out.len(), "a and out must have equal length");
    let n = a.len();
    let chunks = n / 4;
    let remainder = n % 4;

    // SAFETY: aarch64 NEON is always available. Bounded loads/stores within slice.
    unsafe {
        for i in 0..chunks {
            let offset = i * 4;
            let va = vld1q_f32(a.as_ptr().add(offset));
            let vb = vld1q_f32(b.as_ptr().add(offset));
            let r = vmulq_f32(va, vb);
            vst1q_f32(out.as_mut_ptr().add(offset), r);
        }
    }
    let tail = chunks * 4;
    for i in 0..remainder {
        out[tail + i] = a[tail + i] * b[tail + i];
    }
}

#[cfg(not(target_arch = "aarch64"))]
pub fn mul_f32_neon(a: &[f32], b: &[f32], out: &mut [f32]) {
    mul_f32_scalar(a, b, out);
}

#[cfg(target_arch = "aarch64")]
pub fn scalar_mul_f32_neon(a: &[f32], scalar: f32, out: &mut [f32]) {
    use std::arch::aarch64::*;

    assert_eq!(a.len(), out.len(), "a and out must have equal length");
    let n = a.len();
    let chunks = n / 4;
    let remainder = n % 4;

    // SAFETY: aarch64 NEON is always available. Bounded loads/stores within slice.
    unsafe {
        let vs = vdupq_n_f32(scalar);
        for i in 0..chunks {
            let offset = i * 4;
            let va = vld1q_f32(a.as_ptr().add(offset));
            let r = vmulq_f32(va, vs);
            vst1q_f32(out.as_mut_ptr().add(offset), r);
        }
    }
    let tail = chunks * 4;
    for i in 0..remainder {
        out[tail + i] = a[tail + i] * scalar;
    }
}

#[cfg(not(target_arch = "aarch64"))]
pub fn scalar_mul_f32_neon(a: &[f32], scalar: f32, out: &mut [f32]) {
    scalar_mul_f32_scalar(a, scalar, out);
}

#[cfg(target_arch = "aarch64")]
pub fn fma_f32_neon(a: &[f32], b: &[f32], c: &[f32], out: &mut [f32]) {
    use std::arch::aarch64::*;

    assert_eq!(a.len(), b.len(), "a and b must have equal length");
    assert_eq!(a.len(), c.len(), "a and c must have equal length");
    assert_eq!(a.len(), out.len(), "a and out must have equal length");
    let n = a.len();
    let chunks = n / 4;
    let remainder = n % 4;

    // SAFETY: aarch64 NEON is always available. vfmaq_f32 computes c + a*b,
    // so we pass (vc, va, vb) to get a*b+c. Bounded loads/stores within slice.
    unsafe {
        for i in 0..chunks {
            let offset = i * 4;
            let va = vld1q_f32(a.as_ptr().add(offset));
            let vb = vld1q_f32(b.as_ptr().add(offset));
            let vc = vld1q_f32(c.as_ptr().add(offset));
            let r = vfmaq_f32(vc, va, vb);
            vst1q_f32(out.as_mut_ptr().add(offset), r);
        }
    }
    let tail = chunks * 4;
    for i in 0..remainder {
        out[tail + i] = a[tail + i] * b[tail + i] + c[tail + i];
    }
}

#[cfg(not(target_arch = "aarch64"))]
pub fn fma_f32_neon(a: &[f32], b: &[f32], c: &[f32], out: &mut [f32]) {
    fma_f32_scalar(a, b, c, out);
}

#[cfg(target_arch = "aarch64")]
pub fn relu_f32_neon(x: &[f32], out: &mut [f32]) {
    use std::arch::aarch64::*;

    assert_eq!(x.len(), out.len(), "x and out must have equal length");
    let n = x.len();
    let chunks = n / 4;
    let remainder = n % 4;

    // SAFETY: aarch64 NEON is always available. Bounded loads/stores within slice.
    unsafe {
        let zero = vdupq_n_f32(0.0);
        for i in 0..chunks {
            let offset = i * 4;
            let v = vld1q_f32(x.as_ptr().add(offset));
            let r = vmaxq_f32(v, zero);
            vst1q_f32(out.as_mut_ptr().add(offset), r);
        }
    }
    let tail = chunks * 4;
    for i in 0..remainder {
        out[tail + i] = x[tail + i].max(0.0);
    }
}

#[cfg(not(target_arch = "aarch64"))]
pub fn relu_f32_neon(x: &[f32], out: &mut [f32]) {
    relu_f32_scalar(x, out);
}

#[cfg(target_arch = "aarch64")]
pub fn gelu_f32_neon(x: &[f32], out: &mut [f32]) {
    use std::arch::aarch64::*;

    assert_eq!(x.len(), out.len(), "x and out must have equal length");
    let c = (2.0_f32 / std::f32::consts::PI).sqrt();
    let n = x.len();
    let chunks = n / 4;
    let remainder = n % 4;

    // SAFETY: aarch64 NEON is always available. We load 4 f32s, compute GELU
    // per-lane (tanh is transcendental — no NEON intrinsic), store 4 f32s.
    unsafe {
        for i in 0..chunks {
            let offset = i * 4;
            let v = vld1q_f32(x.as_ptr().add(offset));
            let x0 = vgetq_lane_f32::<0>(v);
            let x1 = vgetq_lane_f32::<1>(v);
            let x2 = vgetq_lane_f32::<2>(v);
            let x3 = vgetq_lane_f32::<3>(v);
            let g = |val: f32| -> f32 {
                val * 0.5 * (1.0 + (c * (val + 0.044715 * val * val * val)).tanh())
            };
            let mut r = vdupq_n_f32(0.0);
            r = vsetq_lane_f32::<0>(g(x0), r);
            r = vsetq_lane_f32::<1>(g(x1), r);
            r = vsetq_lane_f32::<2>(g(x2), r);
            r = vsetq_lane_f32::<3>(g(x3), r);
            vst1q_f32(out.as_mut_ptr().add(offset), r);
        }
    }
    let tail = chunks * 4;
    for i in 0..remainder {
        let v = x[tail + i];
        out[tail + i] = v * 0.5 * (1.0 + (c * (v + 0.044715 * v * v * v)).tanh());
    }
}

#[cfg(not(target_arch = "aarch64"))]
pub fn gelu_f32_neon(x: &[f32], out: &mut [f32]) {
    gelu_f32_scalar(x, out);
}

#[cfg(target_arch = "aarch64")]
pub fn silu_f32_neon(x: &[f32], out: &mut [f32]) {
    use std::arch::aarch64::*;

    assert_eq!(x.len(), out.len(), "x and out must have equal length");
    let n = x.len();
    let chunks = n / 4;
    let remainder = n % 4;

    // SAFETY: aarch64 NEON is always available. Per-lane exp (transcendental).
    unsafe {
        for i in 0..chunks {
            let offset = i * 4;
            let v = vld1q_f32(x.as_ptr().add(offset));
            let x0 = vgetq_lane_f32::<0>(v);
            let x1 = vgetq_lane_f32::<1>(v);
            let x2 = vgetq_lane_f32::<2>(v);
            let x3 = vgetq_lane_f32::<3>(v);
            let s = |val: f32| -> f32 { val / (1.0 + (-val).exp()) };
            let mut r = vdupq_n_f32(0.0);
            r = vsetq_lane_f32::<0>(s(x0), r);
            r = vsetq_lane_f32::<1>(s(x1), r);
            r = vsetq_lane_f32::<2>(s(x2), r);
            r = vsetq_lane_f32::<3>(s(x3), r);
            vst1q_f32(out.as_mut_ptr().add(offset), r);
        }
    }
    let tail = chunks * 4;
    for i in 0..remainder {
        let v = x[tail + i];
        out[tail + i] = v / (1.0 + (-v).exp());
    }
}

#[cfg(not(target_arch = "aarch64"))]
pub fn silu_f32_neon(x: &[f32], out: &mut [f32]) {
    silu_f32_scalar(x, out);
}

// ---------------------------------------------------------------------------
// AVX2 (x86_64) — 256-bit, 8x f32 lanes
// ---------------------------------------------------------------------------

#[cfg(target_arch = "x86_64")]
pub fn add_f32_avx2(a: &[f32], b: &[f32], out: &mut [f32]) {
    if is_x86_feature_detected!("avx2") {
        // SAFETY: AVX2 detected above.
        unsafe { add_f32_avx2_inner(a, b, out) };
    } else {
        add_f32_scalar(a, b, out);
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn add_f32_avx2_inner(a: &[f32], b: &[f32], out: &mut [f32]) {
    use std::arch::x86_64::*;

    assert_eq!(a.len(), b.len(), "a and b must have equal length");
    assert_eq!(a.len(), out.len(), "a and out must have equal length");
    let n = a.len();
    let chunks = n / 8;
    let remainder = n % 8;

    for i in 0..chunks {
        let offset = i * 8;
        // SAFETY: offset + 8 <= n from loop bound. Unaligned loads/stores.
        let va = _mm256_loadu_ps(a.as_ptr().add(offset));
        let vb = _mm256_loadu_ps(b.as_ptr().add(offset));
        let r = _mm256_add_ps(va, vb);
        _mm256_storeu_ps(out.as_mut_ptr().add(offset), r);
    }
    let tail = chunks * 8;
    for i in 0..remainder {
        out[tail + i] = a[tail + i] + b[tail + i];
    }
}

#[cfg(not(target_arch = "x86_64"))]
pub fn add_f32_avx2(a: &[f32], b: &[f32], out: &mut [f32]) {
    add_f32_scalar(a, b, out);
}

#[cfg(target_arch = "x86_64")]
pub fn mul_f32_avx2(a: &[f32], b: &[f32], out: &mut [f32]) {
    if is_x86_feature_detected!("avx2") {
        // SAFETY: AVX2 detected above.
        unsafe { mul_f32_avx2_inner(a, b, out) };
    } else {
        mul_f32_scalar(a, b, out);
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn mul_f32_avx2_inner(a: &[f32], b: &[f32], out: &mut [f32]) {
    use std::arch::x86_64::*;

    assert_eq!(a.len(), b.len(), "a and b must have equal length");
    assert_eq!(a.len(), out.len(), "a and out must have equal length");
    let n = a.len();
    let chunks = n / 8;
    let remainder = n % 8;

    for i in 0..chunks {
        let offset = i * 8;
        // SAFETY: offset + 8 <= n from loop bound. Unaligned loads/stores.
        let va = _mm256_loadu_ps(a.as_ptr().add(offset));
        let vb = _mm256_loadu_ps(b.as_ptr().add(offset));
        let r = _mm256_mul_ps(va, vb);
        _mm256_storeu_ps(out.as_mut_ptr().add(offset), r);
    }
    let tail = chunks * 8;
    for i in 0..remainder {
        out[tail + i] = a[tail + i] * b[tail + i];
    }
}

#[cfg(not(target_arch = "x86_64"))]
pub fn mul_f32_avx2(a: &[f32], b: &[f32], out: &mut [f32]) {
    mul_f32_scalar(a, b, out);
}

#[cfg(target_arch = "x86_64")]
pub fn scalar_mul_f32_avx2(a: &[f32], scalar: f32, out: &mut [f32]) {
    if is_x86_feature_detected!("avx2") {
        // SAFETY: AVX2 detected above.
        unsafe { scalar_mul_f32_avx2_inner(a, scalar, out) };
    } else {
        scalar_mul_f32_scalar(a, scalar, out);
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn scalar_mul_f32_avx2_inner(a: &[f32], scalar: f32, out: &mut [f32]) {
    use std::arch::x86_64::*;

    assert_eq!(a.len(), out.len(), "a and out must have equal length");
    let n = a.len();
    let chunks = n / 8;
    let remainder = n % 8;

    let vs = _mm256_set1_ps(scalar);
    for i in 0..chunks {
        let offset = i * 8;
        // SAFETY: offset + 8 <= n from loop bound. Unaligned loads/stores.
        let va = _mm256_loadu_ps(a.as_ptr().add(offset));
        let r = _mm256_mul_ps(va, vs);
        _mm256_storeu_ps(out.as_mut_ptr().add(offset), r);
    }
    let tail = chunks * 8;
    for i in 0..remainder {
        out[tail + i] = a[tail + i] * scalar;
    }
}

#[cfg(not(target_arch = "x86_64"))]
pub fn scalar_mul_f32_avx2(a: &[f32], scalar: f32, out: &mut [f32]) {
    scalar_mul_f32_scalar(a, scalar, out);
}

#[cfg(target_arch = "x86_64")]
pub fn fma_f32_avx2(a: &[f32], b: &[f32], c: &[f32], out: &mut [f32]) {
    if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
        // SAFETY: AVX2+FMA detected above.
        unsafe { fma_f32_avx2_inner(a, b, c, out) };
    } else {
        fma_f32_scalar(a, b, c, out);
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
unsafe fn fma_f32_avx2_inner(a: &[f32], b: &[f32], c: &[f32], out: &mut [f32]) {
    use std::arch::x86_64::*;

    assert_eq!(a.len(), b.len(), "a and b must have equal length");
    assert_eq!(a.len(), c.len(), "a and c must have equal length");
    assert_eq!(a.len(), out.len(), "a and out must have equal length");
    let n = a.len();
    let chunks = n / 8;
    let remainder = n % 8;

    for i in 0..chunks {
        let offset = i * 8;
        // SAFETY: offset + 8 <= n from loop bound. Unaligned loads/stores.
        let va = _mm256_loadu_ps(a.as_ptr().add(offset));
        let vb = _mm256_loadu_ps(b.as_ptr().add(offset));
        let vc = _mm256_loadu_ps(c.as_ptr().add(offset));
        let r = _mm256_fmadd_ps(va, vb, vc);
        _mm256_storeu_ps(out.as_mut_ptr().add(offset), r);
    }
    let tail = chunks * 8;
    for i in 0..remainder {
        out[tail + i] = a[tail + i] * b[tail + i] + c[tail + i];
    }
}

#[cfg(not(target_arch = "x86_64"))]
pub fn fma_f32_avx2(a: &[f32], b: &[f32], c: &[f32], out: &mut [f32]) {
    fma_f32_scalar(a, b, c, out);
}

#[cfg(target_arch = "x86_64")]
pub fn relu_f32_avx2(x: &[f32], out: &mut [f32]) {
    if is_x86_feature_detected!("avx2") {
        // SAFETY: AVX2 detected above.
        unsafe { relu_f32_avx2_inner(x, out) };
    } else {
        relu_f32_scalar(x, out);
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn relu_f32_avx2_inner(x: &[f32], out: &mut [f32]) {
    use std::arch::x86_64::*;

    assert_eq!(x.len(), out.len(), "x and out must have equal length");
    let n = x.len();
    let chunks = n / 8;
    let remainder = n % 8;

    let zero = _mm256_setzero_ps();
    for i in 0..chunks {
        let offset = i * 8;
        // SAFETY: offset + 8 <= n from loop bound. Unaligned loads/stores.
        let v = _mm256_loadu_ps(x.as_ptr().add(offset));
        let r = _mm256_max_ps(v, zero);
        _mm256_storeu_ps(out.as_mut_ptr().add(offset), r);
    }
    let tail = chunks * 8;
    for i in 0..remainder {
        out[tail + i] = x[tail + i].max(0.0);
    }
}

#[cfg(not(target_arch = "x86_64"))]
pub fn relu_f32_avx2(x: &[f32], out: &mut [f32]) {
    relu_f32_scalar(x, out);
}

#[cfg(target_arch = "x86_64")]
pub fn gelu_f32_avx2(x: &[f32], out: &mut [f32]) {
    if is_x86_feature_detected!("avx2") {
        // SAFETY: AVX2 detected above.
        unsafe { gelu_f32_avx2_inner(x, out) };
    } else {
        gelu_f32_scalar(x, out);
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn gelu_f32_avx2_inner(x: &[f32], out: &mut [f32]) {
    use std::arch::x86_64::*;

    assert_eq!(x.len(), out.len(), "x and out must have equal length");
    let c = (2.0_f32 / std::f32::consts::PI).sqrt();
    let n = x.len();
    let chunks = n / 8;
    let remainder = n % 8;

    for i in 0..chunks {
        let offset = i * 8;
        let mut buf = [0.0f32; 8];
        // SAFETY: offset + 8 <= n from loop bound. Unaligned loads/stores.
        let v = _mm256_loadu_ps(x.as_ptr().add(offset));
        _mm256_storeu_ps(buf.as_mut_ptr(), v);
        for b in &mut buf {
            let val = *b;
            *b = val * 0.5 * (1.0 + (c * (val + 0.044715 * val * val * val)).tanh());
        }
        let r = _mm256_loadu_ps(buf.as_ptr());
        _mm256_storeu_ps(out.as_mut_ptr().add(offset), r);
    }
    let tail = chunks * 8;
    for i in 0..remainder {
        let v = x[tail + i];
        out[tail + i] = v * 0.5 * (1.0 + (c * (v + 0.044715 * v * v * v)).tanh());
    }
}

#[cfg(not(target_arch = "x86_64"))]
pub fn gelu_f32_avx2(x: &[f32], out: &mut [f32]) {
    gelu_f32_scalar(x, out);
}

#[cfg(target_arch = "x86_64")]
pub fn silu_f32_avx2(x: &[f32], out: &mut [f32]) {
    if is_x86_feature_detected!("avx2") {
        // SAFETY: AVX2 detected above.
        unsafe { silu_f32_avx2_inner(x, out) };
    } else {
        silu_f32_scalar(x, out);
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn silu_f32_avx2_inner(x: &[f32], out: &mut [f32]) {
    use std::arch::x86_64::*;

    assert_eq!(x.len(), out.len(), "x and out must have equal length");
    let n = x.len();
    let chunks = n / 8;
    let remainder = n % 8;

    for i in 0..chunks {
        let offset = i * 8;
        let mut buf = [0.0f32; 8];
        // SAFETY: offset + 8 <= n from loop bound. Unaligned loads/stores.
        let v = _mm256_loadu_ps(x.as_ptr().add(offset));
        _mm256_storeu_ps(buf.as_mut_ptr(), v);
        for b in &mut buf {
            let val = *b;
            *b = val / (1.0 + (-val).exp());
        }
        let r = _mm256_loadu_ps(buf.as_ptr());
        _mm256_storeu_ps(out.as_mut_ptr().add(offset), r);
    }
    let tail = chunks * 8;
    for i in 0..remainder {
        let v = x[tail + i];
        out[tail + i] = v / (1.0 + (-v).exp());
    }
}

#[cfg(not(target_arch = "x86_64"))]
pub fn silu_f32_avx2(x: &[f32], out: &mut [f32]) {
    silu_f32_scalar(x, out);
}

// ---------------------------------------------------------------------------
// Public dispatch: auto-selects best SIMD path
// ---------------------------------------------------------------------------

/// Elementwise add: `out[i] = a[i] + b[i]`. Auto-dispatches to NEON/AVX2/scalar.
#[inline]
pub fn add_f32(a: &[f32], b: &[f32], out: &mut [f32]) {
    #[cfg(target_arch = "aarch64")]
    {
        add_f32_neon(a, b, out);
        return;
    }

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            add_f32_avx2(a, b, out);
            return;
        }
    }

    #[allow(unreachable_code)]
    add_f32_scalar(a, b, out);
}

/// Elementwise multiply: `out[i] = a[i] * b[i]`. Auto-dispatches to NEON/AVX2/scalar.
#[inline]
pub fn mul_f32(a: &[f32], b: &[f32], out: &mut [f32]) {
    #[cfg(target_arch = "aarch64")]
    {
        mul_f32_neon(a, b, out);
        return;
    }

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            mul_f32_avx2(a, b, out);
            return;
        }
    }

    #[allow(unreachable_code)]
    mul_f32_scalar(a, b, out);
}

/// Scalar broadcast multiply: `out[i] = a[i] * scalar`. Auto-dispatches to NEON/AVX2/scalar.
#[inline]
pub fn scalar_mul_f32(a: &[f32], scalar: f32, out: &mut [f32]) {
    #[cfg(target_arch = "aarch64")]
    {
        scalar_mul_f32_neon(a, scalar, out);
        return;
    }

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            scalar_mul_f32_avx2(a, scalar, out);
            return;
        }
    }

    #[allow(unreachable_code)]
    scalar_mul_f32_scalar(a, scalar, out);
}

/// Fused multiply-add: `out[i] = a[i] * b[i] + c[i]`. Auto-dispatches to NEON/AVX2/scalar.
///
/// Uses hardware FMA on supported platforms (NEON `vfmaq_f32`, AVX2 `_mm256_fmadd_ps`).
#[inline]
pub fn fma_f32(a: &[f32], b: &[f32], c: &[f32], out: &mut [f32]) {
    #[cfg(target_arch = "aarch64")]
    {
        fma_f32_neon(a, b, c, out);
        return;
    }

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            fma_f32_avx2(a, b, c, out);
            return;
        }
    }

    #[allow(unreachable_code)]
    fma_f32_scalar(a, b, c, out);
}

/// ReLU activation: `out[i] = max(0, x[i])`. Auto-dispatches to NEON/AVX2/scalar.
#[inline]
pub fn relu_f32(x: &[f32], out: &mut [f32]) {
    #[cfg(target_arch = "aarch64")]
    {
        relu_f32_neon(x, out);
        return;
    }

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            relu_f32_avx2(x, out);
            return;
        }
    }

    #[allow(unreachable_code)]
    relu_f32_scalar(x, out);
}

/// GELU activation (tanh approximation). Auto-dispatches to NEON/AVX2/scalar.
///
/// `out[i] = x * 0.5 * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))`
#[inline]
pub fn gelu_f32(x: &[f32], out: &mut [f32]) {
    #[cfg(target_arch = "aarch64")]
    {
        gelu_f32_neon(x, out);
        return;
    }

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            gelu_f32_avx2(x, out);
            return;
        }
    }

    #[allow(unreachable_code)]
    gelu_f32_scalar(x, out);
}

/// SiLU (swish) activation: `out[i] = x[i] * sigmoid(x[i])`. Auto-dispatches to NEON/AVX2/scalar.
#[inline]
pub fn silu_f32(x: &[f32], out: &mut [f32]) {
    #[cfg(target_arch = "aarch64")]
    {
        silu_f32_neon(x, out);
        return;
    }

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            silu_f32_avx2(x, out);
            return;
        }
    }

    #[allow(unreachable_code)]
    silu_f32_scalar(x, out);
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
#[path = "simd_elementwise_tests.rs"]
mod simd_elementwise_tests;

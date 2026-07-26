// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! SIMD-vectorized elementwise operations: binary (add, mul) and activations.
//!
//! Each function processes input slice(s) and writes to an output slice.
//! NEON (aarch64) and AVX2 (x86_64) paths with scalar fallback.

// ---------------------------------------------------------------------------
// Scalar fallback — binary ops (all architectures)
// ---------------------------------------------------------------------------

/// Scalar elementwise add: `out[i] = a[i] + b[i]`.
pub fn add_scalar(a: &[f32], b: &[f32], out: &mut [f32]) {
    assert_eq!(a.len(), b.len());
    assert_eq!(a.len(), out.len());
    for i in 0..a.len() {
        out[i] = a[i] + b[i];
    }
}

/// Scalar elementwise mul: `out[i] = a[i] * b[i]`.
pub fn mul_scalar(a: &[f32], b: &[f32], out: &mut [f32]) {
    assert_eq!(a.len(), b.len());
    assert_eq!(a.len(), out.len());
    for i in 0..a.len() {
        out[i] = a[i] * b[i];
    }
}

// ---------------------------------------------------------------------------
// Scalar fallback — unary activations (all architectures)
// ---------------------------------------------------------------------------

/// Scalar ReLU: `max(0, x)`.
pub fn relu_scalar(input: &[f32], output: &mut [f32]) {
    assert_eq!(input.len(), output.len());
    for (o, &x) in output.iter_mut().zip(input.iter()) {
        *o = x.max(0.0);
    }
}

/// Scalar SiLU (Swish): `x * sigmoid(x)`.
pub fn silu_scalar(input: &[f32], output: &mut [f32]) {
    assert_eq!(input.len(), output.len());
    for (o, &x) in output.iter_mut().zip(input.iter()) {
        *o = x / (1.0 + (-x).exp());
    }
}

/// Scalar sigmoid: `1 / (1 + exp(-x))`.
pub fn sigmoid_scalar(input: &[f32], output: &mut [f32]) {
    assert_eq!(input.len(), output.len());
    for (o, &x) in output.iter_mut().zip(input.iter()) {
        *o = 1.0 / (1.0 + (-x).exp());
    }
}

/// Scalar tanh.
pub fn tanh_scalar(input: &[f32], output: &mut [f32]) {
    assert_eq!(input.len(), output.len());
    for (o, &x) in output.iter_mut().zip(input.iter()) {
        *o = x.tanh();
    }
}

/// Scalar GELU (tanh approximation): `x * 0.5 * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))`.
pub fn gelu_scalar(input: &[f32], output: &mut [f32]) {
    assert_eq!(input.len(), output.len());
    let c = (2.0_f32 / std::f32::consts::PI).sqrt();
    for (o, &x) in output.iter_mut().zip(input.iter()) {
        *o = x * 0.5 * (1.0 + (c * (x + 0.044715 * x * x * x)).tanh());
    }
}

// ---------------------------------------------------------------------------
// NEON (aarch64) — 128-bit, 4x f32 lanes
// ---------------------------------------------------------------------------

#[cfg(target_arch = "aarch64")]
mod neon {
    use std::arch::aarch64::*;

    /// NEON elementwise add: `vaddq_f32`.
    #[inline(always)]
    pub(super) fn add_neon(a: &[f32], b: &[f32], out: &mut [f32]) {
        assert_eq!(a.len(), b.len());
        assert_eq!(a.len(), out.len());
        let n = a.len();
        let chunks = n / 4;
        let remainder = n % 4;

        // SAFETY: aarch64 NEON is always available. We read/write aligned
        // 4-element chunks within bounds, then handle the remainder scalarly.
        unsafe {
            for i in 0..chunks {
                let offset = i * 4;
                let va = vld1q_f32(a.as_ptr().add(offset));
                let vb = vld1q_f32(b.as_ptr().add(offset));
                let r = vaddq_f32(va, vb);
                vst1q_f32(out.as_mut_ptr().add(offset), r);
            }
        }
        let tail_start = chunks * 4;
        for i in 0..remainder {
            out[tail_start + i] = a[tail_start + i] + b[tail_start + i];
        }
    }

    /// NEON elementwise mul: `vmulq_f32`.
    #[inline(always)]
    pub(super) fn mul_neon(a: &[f32], b: &[f32], out: &mut [f32]) {
        assert_eq!(a.len(), b.len());
        assert_eq!(a.len(), out.len());
        let n = a.len();
        let chunks = n / 4;
        let remainder = n % 4;

        // SAFETY: aarch64 NEON is always available. Bounded loads/stores.
        unsafe {
            for i in 0..chunks {
                let offset = i * 4;
                let va = vld1q_f32(a.as_ptr().add(offset));
                let vb = vld1q_f32(b.as_ptr().add(offset));
                let r = vmulq_f32(va, vb);
                vst1q_f32(out.as_mut_ptr().add(offset), r);
            }
        }
        let tail_start = chunks * 4;
        for i in 0..remainder {
            out[tail_start + i] = a[tail_start + i] * b[tail_start + i];
        }
    }

    /// NEON ReLU: `vmax(x, 0)`.
    pub(super) fn relu_neon(input: &[f32], output: &mut [f32]) {
        assert_eq!(input.len(), output.len());
        let n = input.len();
        let chunks = n / 4;
        let remainder = n % 4;

        // SAFETY: aarch64 NEON is always available. We read/write aligned
        // 4-element chunks within bounds, then handle the remainder scalarly.
        unsafe {
            let zero = vdupq_n_f32(0.0);
            for i in 0..chunks {
                let offset = i * 4;
                let v = vld1q_f32(input.as_ptr().add(offset));
                let r = vmaxq_f32(v, zero);
                vst1q_f32(output.as_mut_ptr().add(offset), r);
            }
        }
        // Scalar tail
        let tail_start = chunks * 4;
        for i in 0..remainder {
            output[tail_start + i] = input[tail_start + i].max(0.0);
        }
    }

    /// NEON sigmoid: element-by-element (transcendental — no NEON intrinsic for exp).
    /// Processes 4 lanes with scalar exp per lane then packs back.
    pub(super) fn sigmoid_neon(input: &[f32], output: &mut [f32]) {
        assert_eq!(input.len(), output.len());
        let n = input.len();
        let chunks = n / 4;
        let remainder = n % 4;

        // SAFETY: We load 4 f32s, compute sigmoid per-lane, store 4 f32s.
        // Bounds checked by chunks/remainder split.
        unsafe {
            for i in 0..chunks {
                let offset = i * 4;
                let v = vld1q_f32(input.as_ptr().add(offset));
                // Extract lanes, compute sigmoid, repack.
                let x0 = vgetq_lane_f32::<0>(v);
                let x1 = vgetq_lane_f32::<1>(v);
                let x2 = vgetq_lane_f32::<2>(v);
                let x3 = vgetq_lane_f32::<3>(v);
                let s0 = 1.0 / (1.0 + (-x0).exp());
                let s1 = 1.0 / (1.0 + (-x1).exp());
                let s2 = 1.0 / (1.0 + (-x2).exp());
                let s3 = 1.0 / (1.0 + (-x3).exp());
                let r = {
                    let mut tmp = vdupq_n_f32(0.0);
                    tmp = vsetq_lane_f32::<0>(s0, tmp);
                    tmp = vsetq_lane_f32::<1>(s1, tmp);
                    tmp = vsetq_lane_f32::<2>(s2, tmp);
                    tmp = vsetq_lane_f32::<3>(s3, tmp);
                    tmp
                };
                vst1q_f32(output.as_mut_ptr().add(offset), r);
            }
        }
        let tail_start = chunks * 4;
        for i in 0..remainder {
            let x = input[tail_start + i];
            output[tail_start + i] = 1.0 / (1.0 + (-x).exp());
        }
    }

    /// NEON SiLU: `x * sigmoid(x)`. Uses same lane-extraction pattern.
    pub(super) fn silu_neon(input: &[f32], output: &mut [f32]) {
        assert_eq!(input.len(), output.len());
        let n = input.len();
        let chunks = n / 4;
        let remainder = n % 4;

        // SAFETY: Same pattern as sigmoid_neon — bounded loads/stores.
        unsafe {
            for i in 0..chunks {
                let offset = i * 4;
                let v = vld1q_f32(input.as_ptr().add(offset));
                let x0 = vgetq_lane_f32::<0>(v);
                let x1 = vgetq_lane_f32::<1>(v);
                let x2 = vgetq_lane_f32::<2>(v);
                let x3 = vgetq_lane_f32::<3>(v);
                let s0 = x0 / (1.0 + (-x0).exp());
                let s1 = x1 / (1.0 + (-x1).exp());
                let s2 = x2 / (1.0 + (-x2).exp());
                let s3 = x3 / (1.0 + (-x3).exp());
                let r = {
                    let mut tmp = vdupq_n_f32(0.0);
                    tmp = vsetq_lane_f32::<0>(s0, tmp);
                    tmp = vsetq_lane_f32::<1>(s1, tmp);
                    tmp = vsetq_lane_f32::<2>(s2, tmp);
                    tmp = vsetq_lane_f32::<3>(s3, tmp);
                    tmp
                };
                vst1q_f32(output.as_mut_ptr().add(offset), r);
            }
        }
        let tail_start = chunks * 4;
        for i in 0..remainder {
            let x = input[tail_start + i];
            output[tail_start + i] = x / (1.0 + (-x).exp());
        }
    }

    /// NEON tanh: lane-extraction pattern (no NEON tanh intrinsic).
    pub(super) fn tanh_neon(input: &[f32], output: &mut [f32]) {
        assert_eq!(input.len(), output.len());
        let n = input.len();
        let chunks = n / 4;
        let remainder = n % 4;

        // SAFETY: Bounded loads/stores within slice length.
        unsafe {
            for i in 0..chunks {
                let offset = i * 4;
                let v = vld1q_f32(input.as_ptr().add(offset));
                let x0 = vgetq_lane_f32::<0>(v);
                let x1 = vgetq_lane_f32::<1>(v);
                let x2 = vgetq_lane_f32::<2>(v);
                let x3 = vgetq_lane_f32::<3>(v);
                let r = {
                    let mut tmp = vdupq_n_f32(0.0);
                    tmp = vsetq_lane_f32::<0>(x0.tanh(), tmp);
                    tmp = vsetq_lane_f32::<1>(x1.tanh(), tmp);
                    tmp = vsetq_lane_f32::<2>(x2.tanh(), tmp);
                    tmp = vsetq_lane_f32::<3>(x3.tanh(), tmp);
                    tmp
                };
                vst1q_f32(output.as_mut_ptr().add(offset), r);
            }
        }
        let tail_start = chunks * 4;
        for i in 0..remainder {
            output[tail_start + i] = input[tail_start + i].tanh();
        }
    }

    /// NEON GELU (tanh approximation).
    pub(super) fn gelu_neon(input: &[f32], output: &mut [f32]) {
        assert_eq!(input.len(), output.len());
        let c = (2.0_f32 / std::f32::consts::PI).sqrt();
        let n = input.len();
        let chunks = n / 4;
        let remainder = n % 4;

        // SAFETY: Bounded loads/stores within slice length.
        unsafe {
            for i in 0..chunks {
                let offset = i * 4;
                let v = vld1q_f32(input.as_ptr().add(offset));
                let x0 = vgetq_lane_f32::<0>(v);
                let x1 = vgetq_lane_f32::<1>(v);
                let x2 = vgetq_lane_f32::<2>(v);
                let x3 = vgetq_lane_f32::<3>(v);
                let g =
                    |x: f32| -> f32 { x * 0.5 * (1.0 + (c * (x + 0.044715 * x * x * x)).tanh()) };
                let r = {
                    let mut tmp = vdupq_n_f32(0.0);
                    tmp = vsetq_lane_f32::<0>(g(x0), tmp);
                    tmp = vsetq_lane_f32::<1>(g(x1), tmp);
                    tmp = vsetq_lane_f32::<2>(g(x2), tmp);
                    tmp = vsetq_lane_f32::<3>(g(x3), tmp);
                    tmp
                };
                vst1q_f32(output.as_mut_ptr().add(offset), r);
            }
        }
        let tail_start = chunks * 4;
        for i in 0..remainder {
            let x = input[tail_start + i];
            output[tail_start + i] = x * 0.5 * (1.0 + (c * (x + 0.044715 * x * x * x)).tanh());
        }
    }
}

// ---------------------------------------------------------------------------
// AVX2 (x86_64) — 256-bit, 8x f32 lanes
// ---------------------------------------------------------------------------

#[cfg(target_arch = "x86_64")]
mod avx2 {
    use std::arch::x86_64::*;

    /// AVX2 elementwise add: `_mm256_add_ps`.
    ///
    /// # Safety
    /// Caller must verify AVX2 is available (`is_x86_feature_detected!("avx2")`).
    #[target_feature(enable = "avx2")]
    #[inline(always)]
    pub unsafe fn add_avx2(a: &[f32], b: &[f32], out: &mut [f32]) {
        assert_eq!(a.len(), b.len());
        assert_eq!(a.len(), out.len());
        let n = a.len();
        let chunks = n / 8;
        let remainder = n % 8;

        for i in 0..chunks {
            let offset = i * 8;
            let va = _mm256_loadu_ps(a.as_ptr().add(offset));
            let vb = _mm256_loadu_ps(b.as_ptr().add(offset));
            let r = _mm256_add_ps(va, vb);
            _mm256_storeu_ps(out.as_mut_ptr().add(offset), r);
        }
        // Scalar tail
        let tail_start = chunks * 8;
        for i in 0..remainder {
            out[tail_start + i] = a[tail_start + i] + b[tail_start + i];
        }
    }

    /// AVX2 elementwise mul: `_mm256_mul_ps`.
    ///
    /// # Safety
    /// Caller must verify AVX2 is available.
    #[target_feature(enable = "avx2")]
    #[inline(always)]
    pub unsafe fn mul_avx2(a: &[f32], b: &[f32], out: &mut [f32]) {
        assert_eq!(a.len(), b.len());
        assert_eq!(a.len(), out.len());
        let n = a.len();
        let chunks = n / 8;
        let remainder = n % 8;

        for i in 0..chunks {
            let offset = i * 8;
            let va = _mm256_loadu_ps(a.as_ptr().add(offset));
            let vb = _mm256_loadu_ps(b.as_ptr().add(offset));
            let r = _mm256_mul_ps(va, vb);
            _mm256_storeu_ps(out.as_mut_ptr().add(offset), r);
        }
        let tail_start = chunks * 8;
        for i in 0..remainder {
            out[tail_start + i] = a[tail_start + i] * b[tail_start + i];
        }
    }

    /// AVX2 ReLU: `vmax(x, 0)`.
    ///
    /// # Safety
    /// Caller must verify AVX2 is available (`is_x86_feature_detected!("avx2")`).
    #[target_feature(enable = "avx2")]
    pub unsafe fn relu_avx2(input: &[f32], output: &mut [f32]) {
        assert_eq!(input.len(), output.len());
        let n = input.len();
        let chunks = n / 8;
        let remainder = n % 8;

        let zero = _mm256_setzero_ps();
        for i in 0..chunks {
            let offset = i * 8;
            let v = _mm256_loadu_ps(input.as_ptr().add(offset));
            let r = _mm256_max_ps(v, zero);
            _mm256_storeu_ps(output.as_mut_ptr().add(offset), r);
        }
        // Scalar tail
        let tail_start = chunks * 8;
        for i in 0..remainder {
            output[tail_start + i] = input[tail_start + i].max(0.0);
        }
    }

    /// AVX2 sigmoid: processes 8 lanes with scalar exp per lane.
    ///
    /// # Safety
    /// Caller must verify AVX2 is available.
    #[target_feature(enable = "avx2")]
    pub unsafe fn sigmoid_avx2(input: &[f32], output: &mut [f32]) {
        assert_eq!(input.len(), output.len());
        let n = input.len();
        let chunks = n / 8;
        let remainder = n % 8;

        for i in 0..chunks {
            let offset = i * 8;
            let mut buf = [0.0f32; 8];
            let v = _mm256_loadu_ps(input.as_ptr().add(offset));
            _mm256_storeu_ps(buf.as_mut_ptr(), v);
            for b in &mut buf {
                *b = 1.0 / (1.0 + (-*b).exp());
            }
            let r = _mm256_loadu_ps(buf.as_ptr());
            _mm256_storeu_ps(output.as_mut_ptr().add(offset), r);
        }
        let tail_start = chunks * 8;
        for i in 0..remainder {
            let x = input[tail_start + i];
            output[tail_start + i] = 1.0 / (1.0 + (-x).exp());
        }
    }

    /// AVX2 SiLU.
    ///
    /// # Safety
    /// Caller must verify AVX2 is available.
    #[target_feature(enable = "avx2")]
    pub unsafe fn silu_avx2(input: &[f32], output: &mut [f32]) {
        assert_eq!(input.len(), output.len());
        let n = input.len();
        let chunks = n / 8;
        let remainder = n % 8;

        for i in 0..chunks {
            let offset = i * 8;
            let mut buf = [0.0f32; 8];
            let v = _mm256_loadu_ps(input.as_ptr().add(offset));
            _mm256_storeu_ps(buf.as_mut_ptr(), v);
            for b in &mut buf {
                *b = *b / (1.0 + (-*b).exp());
            }
            let r = _mm256_loadu_ps(buf.as_ptr());
            _mm256_storeu_ps(output.as_mut_ptr().add(offset), r);
        }
        let tail_start = chunks * 8;
        for i in 0..remainder {
            let x = input[tail_start + i];
            output[tail_start + i] = x / (1.0 + (-x).exp());
        }
    }

    /// AVX2 tanh.
    ///
    /// # Safety
    /// Caller must verify AVX2 is available.
    #[target_feature(enable = "avx2")]
    pub unsafe fn tanh_avx2(input: &[f32], output: &mut [f32]) {
        assert_eq!(input.len(), output.len());
        let n = input.len();
        let chunks = n / 8;
        let remainder = n % 8;

        for i in 0..chunks {
            let offset = i * 8;
            let mut buf = [0.0f32; 8];
            let v = _mm256_loadu_ps(input.as_ptr().add(offset));
            _mm256_storeu_ps(buf.as_mut_ptr(), v);
            for b in &mut buf {
                *b = b.tanh();
            }
            let r = _mm256_loadu_ps(buf.as_ptr());
            _mm256_storeu_ps(output.as_mut_ptr().add(offset), r);
        }
        let tail_start = chunks * 8;
        for i in 0..remainder {
            output[tail_start + i] = input[tail_start + i].tanh();
        }
    }

    /// AVX2 GELU (tanh approximation).
    ///
    /// # Safety
    /// Caller must verify AVX2 is available.
    #[target_feature(enable = "avx2")]
    pub unsafe fn gelu_avx2(input: &[f32], output: &mut [f32]) {
        assert_eq!(input.len(), output.len());
        let c = (2.0_f32 / std::f32::consts::PI).sqrt();
        let n = input.len();
        let chunks = n / 8;
        let remainder = n % 8;

        for i in 0..chunks {
            let offset = i * 8;
            let mut buf = [0.0f32; 8];
            let v = _mm256_loadu_ps(input.as_ptr().add(offset));
            _mm256_storeu_ps(buf.as_mut_ptr(), v);
            for b in &mut buf {
                let x = *b;
                *b = x * 0.5 * (1.0 + (c * (x + 0.044715 * x * x * x)).tanh());
            }
            let r = _mm256_loadu_ps(buf.as_ptr());
            _mm256_storeu_ps(output.as_mut_ptr().add(offset), r);
        }
        let tail_start = chunks * 8;
        for i in 0..remainder {
            let x = input[tail_start + i];
            output[tail_start + i] = x * 0.5 * (1.0 + (c * (x + 0.044715 * x * x * x)).tanh());
        }
    }
}

// ---------------------------------------------------------------------------
// Public dispatch: auto-selects best SIMD path
// ---------------------------------------------------------------------------

/// Elementwise add: `out[i] = a[i] + b[i]`. Auto-dispatches to NEON/AVX2/scalar.
pub fn add(a: &[f32], b: &[f32], out: &mut [f32]) {
    #[cfg(target_arch = "aarch64")]
    {
        neon::add_neon(a, b, out);
        return;
    }

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            // SAFETY: AVX2 detected above.
            unsafe { avx2::add_avx2(a, b, out) };
            return;
        }
    }

    #[allow(unreachable_code)]
    add_scalar(a, b, out);
}

/// Elementwise mul: `out[i] = a[i] * b[i]`. Auto-dispatches to NEON/AVX2/scalar.
pub fn mul(a: &[f32], b: &[f32], out: &mut [f32]) {
    #[cfg(target_arch = "aarch64")]
    {
        neon::mul_neon(a, b, out);
        return;
    }

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            // SAFETY: AVX2 detected above.
            unsafe { avx2::mul_avx2(a, b, out) };
            return;
        }
    }

    #[allow(unreachable_code)]
    mul_scalar(a, b, out);
}

/// Apply ReLU elementwise. Auto-dispatches to NEON/AVX2/scalar.
pub fn relu(input: &[f32], output: &mut [f32]) {
    #[cfg(target_arch = "aarch64")]
    {
        neon::relu_neon(input, output);
        return;
    }

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            // SAFETY: AVX2 detected above.
            unsafe { avx2::relu_avx2(input, output) };
            return;
        }
    }

    #[allow(unreachable_code)]
    relu_scalar(input, output);
}

/// Apply SiLU (Swish) elementwise. Auto-dispatches to NEON/AVX2/scalar.
pub fn silu(input: &[f32], output: &mut [f32]) {
    #[cfg(target_arch = "aarch64")]
    {
        neon::silu_neon(input, output);
        return;
    }

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            // SAFETY: AVX2 detected above.
            unsafe { avx2::silu_avx2(input, output) };
            return;
        }
    }

    #[allow(unreachable_code)]
    silu_scalar(input, output);
}

/// Apply sigmoid elementwise. Auto-dispatches to NEON/AVX2/scalar.
pub fn sigmoid(input: &[f32], output: &mut [f32]) {
    #[cfg(target_arch = "aarch64")]
    {
        neon::sigmoid_neon(input, output);
        return;
    }

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            // SAFETY: AVX2 detected above.
            unsafe { avx2::sigmoid_avx2(input, output) };
            return;
        }
    }

    #[allow(unreachable_code)]
    sigmoid_scalar(input, output);
}

/// Apply tanh elementwise. Auto-dispatches to NEON/AVX2/scalar.
pub fn tanh(input: &[f32], output: &mut [f32]) {
    #[cfg(target_arch = "aarch64")]
    {
        neon::tanh_neon(input, output);
        return;
    }

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            // SAFETY: AVX2 detected above.
            unsafe { avx2::tanh_avx2(input, output) };
            return;
        }
    }

    #[allow(unreachable_code)]
    tanh_scalar(input, output);
}

/// Apply GELU (tanh approximation) elementwise. Auto-dispatches to NEON/AVX2/scalar.
pub fn gelu(input: &[f32], output: &mut [f32]) {
    #[cfg(target_arch = "aarch64")]
    {
        neon::gelu_neon(input, output);
        return;
    }

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            // SAFETY: AVX2 detected above.
            unsafe { avx2::gelu_avx2(input, output) };
            return;
        }
    }

    #[allow(unreachable_code)]
    gelu_scalar(input, output);
}

#[cfg(test)]
#[path = "elementwise_tests.rs"]
mod elementwise_tests;

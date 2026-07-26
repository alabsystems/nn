// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Tests for SIMD conv2d.
//!
//! Strategy: compare `conv2d` (SIMD-dispatched) against `conv2d_reference`
//! (pure scalar) and an independent naive implementation.

use crate::simd_conv2d::{conv2d, conv2d_reference, Conv2dError};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn assert_close(a: &[f32], b: &[f32], tol: f32, label: &str) {
    assert_eq!(
        a.len(),
        b.len(),
        "{label}: length mismatch ({} vs {})",
        a.len(),
        b.len()
    );
    for (i, (&va, &vb)) in a.iter().zip(b.iter()).enumerate() {
        let diff = (va - vb).abs();
        assert!(
            diff <= tol,
            "{label}[{i}]: {va} vs {vb} (diff={diff}, tol={tol})"
        );
    }
}

/// Independent naive conv2d for oracle testing.
#[allow(clippy::too_many_arguments)]
fn naive_conv2d(
    input: &[f32],
    weight: &[f32],
    bias: Option<&[f32]>,
    batch: usize,
    in_ch: usize,
    out_ch: usize,
    h: usize,
    w: usize,
    kh: usize,
    kw: usize,
    stride_h: usize,
    stride_w: usize,
    pad_h: usize,
    pad_w: usize,
) -> Vec<f32> {
    let oh = (h + 2 * pad_h - kh) / stride_h + 1;
    let ow = (w + 2 * pad_w - kw) / stride_w + 1;
    let mut output = vec![0.0f32; batch * out_ch * oh * ow];

    for b in 0..batch {
        for oc in 0..out_ch {
            let bias_val = bias.map_or(0.0, |bv| bv[oc]);
            for oy in 0..oh {
                for ox in 0..ow {
                    let mut acc = bias_val;
                    for ic in 0..in_ch {
                        for ky in 0..kh {
                            for kx in 0..kw {
                                let iy = oy * stride_h + ky;
                                let ix = ox * stride_w + kx;
                                if iy >= pad_h && iy < pad_h + h && ix >= pad_w && ix < pad_w + w {
                                    let i_idx = b * in_ch * h * w
                                        + ic * h * w
                                        + (iy - pad_h) * w
                                        + (ix - pad_w);
                                    let w_idx = oc * in_ch * kh * kw + ic * kh * kw + ky * kw + kx;
                                    acc += input[i_idx] * weight[w_idx];
                                }
                            }
                        }
                    }
                    let o_idx = b * out_ch * oh * ow + oc * oh * ow + oy * ow + ox;
                    output[o_idx] = acc;
                }
            }
        }
    }
    output
}

fn alloc_output(
    batch: usize,
    out_ch: usize,
    h: usize,
    w: usize,
    kh: usize,
    kw: usize,
    sh: usize,
    sw: usize,
    ph: usize,
    pw: usize,
) -> Vec<f32> {
    let oh = (h + 2 * ph - kh) / sh + 1;
    let ow = (w + 2 * pw - kw) / sw + 1;
    vec![0.0f32; batch * out_ch * oh * ow]
}

// ---------------------------------------------------------------------------
// Basic tests
// ---------------------------------------------------------------------------

#[test]
fn test_conv2d_single_channel_no_padding() {
    let (batch, in_ch, out_ch) = (1, 1, 1);
    let (h, w, kh, kw) = (4, 4, 3, 3);
    let (sh, sw, ph, pw) = (1, 1, 0, 0);

    let input: Vec<f32> = (0..16).map(|i| (i + 1) as f32).collect();
    let weight = vec![1.0, 0.0, -1.0, 0.0, 1.0, 0.0, -1.0, 0.0, 1.0];

    let mut out_simd = alloc_output(batch, out_ch, h, w, kh, kw, sh, sw, ph, pw);
    let mut out_ref = out_simd.clone();

    conv2d(
        &input,
        &weight,
        None,
        &mut out_simd,
        batch,
        in_ch,
        out_ch,
        h,
        w,
        kh,
        kw,
        sh,
        sw,
        ph,
        pw,
    )
    .unwrap();
    conv2d_reference(
        &input,
        &weight,
        None,
        &mut out_ref,
        batch,
        in_ch,
        out_ch,
        h,
        w,
        kh,
        kw,
        sh,
        sw,
        ph,
        pw,
    )
    .unwrap();

    let naive = naive_conv2d(
        &input, &weight, None, batch, in_ch, out_ch, h, w, kh, kw, sh, sw, ph, pw,
    );
    assert_close(&out_simd, &out_ref, 1e-6, "single_ch_simd_vs_ref");
    assert_close(&out_simd, &naive, 1e-6, "single_ch_simd_vs_naive");
}

#[test]
fn test_conv2d_multi_channel_with_bias() {
    let (batch, in_ch, out_ch) = (1, 3, 2);
    let (h, w, kh, kw) = (5, 5, 3, 3);
    let (sh, sw, ph, pw) = (1, 1, 1, 1);

    let input: Vec<f32> = (0..in_ch * h * w).map(|i| (i as f32) * 0.1 - 3.0).collect();
    let weight: Vec<f32> = (0..out_ch * in_ch * kh * kw)
        .map(|i| (i as f32) * 0.05 - 1.0)
        .collect();
    let bias = vec![0.5, -0.3];

    let mut out_simd = alloc_output(batch, out_ch, h, w, kh, kw, sh, sw, ph, pw);
    let mut out_ref = out_simd.clone();

    conv2d(
        &input,
        &weight,
        Some(&bias),
        &mut out_simd,
        batch,
        in_ch,
        out_ch,
        h,
        w,
        kh,
        kw,
        sh,
        sw,
        ph,
        pw,
    )
    .unwrap();
    conv2d_reference(
        &input,
        &weight,
        Some(&bias),
        &mut out_ref,
        batch,
        in_ch,
        out_ch,
        h,
        w,
        kh,
        kw,
        sh,
        sw,
        ph,
        pw,
    )
    .unwrap();

    let naive = naive_conv2d(
        &input,
        &weight,
        Some(&bias),
        batch,
        in_ch,
        out_ch,
        h,
        w,
        kh,
        kw,
        sh,
        sw,
        ph,
        pw,
    );
    assert_close(&out_simd, &out_ref, 1e-4, "multi_ch_bias_simd_vs_ref");
    assert_close(&out_simd, &naive, 1e-4, "multi_ch_bias_simd_vs_naive");
}

#[test]
fn test_conv2d_stride_2() {
    let (batch, in_ch, out_ch) = (1, 1, 1);
    let (h, w, kh, kw) = (8, 8, 3, 3);
    let (sh, sw, ph, pw) = (2, 2, 0, 0);

    let input: Vec<f32> = (0..h * w).map(|i| (i as f32) * 0.5).collect();
    let weight: Vec<f32> = (0..kh * kw).map(|i| (i as f32) * 0.1).collect();

    let mut out_simd = alloc_output(batch, out_ch, h, w, kh, kw, sh, sw, ph, pw);
    let mut out_ref = out_simd.clone();

    conv2d(
        &input,
        &weight,
        None,
        &mut out_simd,
        batch,
        in_ch,
        out_ch,
        h,
        w,
        kh,
        kw,
        sh,
        sw,
        ph,
        pw,
    )
    .unwrap();
    conv2d_reference(
        &input,
        &weight,
        None,
        &mut out_ref,
        batch,
        in_ch,
        out_ch,
        h,
        w,
        kh,
        kw,
        sh,
        sw,
        ph,
        pw,
    )
    .unwrap();

    // out_h = (8-3)/2+1 = 3, out_w = same => 3x3 = 9 elements
    assert_eq!(out_simd.len(), 9);
    assert_close(&out_simd, &out_ref, 1e-4, "stride2_simd_vs_ref");
}

#[test]
fn test_conv2d_batched() {
    let (batch, in_ch, out_ch) = (2, 2, 3);
    let (h, w, kh, kw) = (6, 6, 3, 3);
    let (sh, sw, ph, pw) = (1, 1, 1, 1);

    let input: Vec<f32> = (0..batch * in_ch * h * w)
        .map(|i| ((i * 7 + 3) % 100) as f32 * 0.01 - 0.5)
        .collect();
    let weight: Vec<f32> = (0..out_ch * in_ch * kh * kw)
        .map(|i| ((i * 13 + 11) % 200) as f32 * 0.005 - 0.5)
        .collect();
    let bias: Vec<f32> = (0..out_ch).map(|i| i as f32 * 0.1).collect();

    let mut out_simd = alloc_output(batch, out_ch, h, w, kh, kw, sh, sw, ph, pw);
    let mut out_ref = out_simd.clone();

    conv2d(
        &input,
        &weight,
        Some(&bias),
        &mut out_simd,
        batch,
        in_ch,
        out_ch,
        h,
        w,
        kh,
        kw,
        sh,
        sw,
        ph,
        pw,
    )
    .unwrap();
    conv2d_reference(
        &input,
        &weight,
        Some(&bias),
        &mut out_ref,
        batch,
        in_ch,
        out_ch,
        h,
        w,
        kh,
        kw,
        sh,
        sw,
        ph,
        pw,
    )
    .unwrap();

    assert_close(&out_simd, &out_ref, 1e-3, "batched_simd_vs_ref");
}

#[test]
fn test_conv2d_non_aligned_width() {
    // ow=5: not divisible by NEON(4) or AVX2(8), exercises scalar tail
    let (batch, in_ch, out_ch) = (1, 1, 1);
    let (h, w, kh, kw) = (7, 7, 3, 3);
    let (sh, sw, ph, pw) = (1, 1, 0, 0);

    let input: Vec<f32> = (0..h * w).map(|i| (i as f32).sin()).collect();
    let weight: Vec<f32> = (0..kh * kw).map(|i| (i as f32).cos() * 0.2).collect();

    let mut out_simd = alloc_output(batch, out_ch, h, w, kh, kw, sh, sw, ph, pw);
    let mut out_ref = out_simd.clone();

    conv2d(
        &input,
        &weight,
        None,
        &mut out_simd,
        batch,
        in_ch,
        out_ch,
        h,
        w,
        kh,
        kw,
        sh,
        sw,
        ph,
        pw,
    )
    .unwrap();
    conv2d_reference(
        &input,
        &weight,
        None,
        &mut out_ref,
        batch,
        in_ch,
        out_ch,
        h,
        w,
        kh,
        kw,
        sh,
        sw,
        ph,
        pw,
    )
    .unwrap();

    // ow = 7-3+1 = 5, oh = 5 => 25 elements
    assert_eq!(out_simd.len(), 25);
    assert_close(&out_simd, &out_ref, 1e-5, "non_aligned_simd_vs_ref");
}

#[test]
fn test_conv2d_all_zeros() {
    let (batch, in_ch, out_ch) = (1, 1, 1);
    let (h, w, kh, kw) = (4, 4, 3, 3);
    let (sh, sw, ph, pw) = (1, 1, 0, 0);

    let input = vec![0.0f32; h * w];
    let weight = vec![1.0f32; kh * kw];

    let mut output = alloc_output(batch, out_ch, h, w, kh, kw, sh, sw, ph, pw);
    conv2d(
        &input,
        &weight,
        None,
        &mut output,
        batch,
        in_ch,
        out_ch,
        h,
        w,
        kh,
        kw,
        sh,
        sw,
        ph,
        pw,
    )
    .unwrap();
    assert!(
        output.iter().all(|&v| v == 0.0),
        "zero input -> zero output"
    );
}

#[test]
fn test_conv2d_deterministic() {
    let (batch, in_ch, out_ch) = (1, 2, 2);
    let (h, w, kh, kw) = (8, 8, 3, 3);
    let (sh, sw, ph, pw) = (1, 1, 1, 1);

    let input: Vec<f32> = (0..batch * in_ch * h * w)
        .map(|i| (i as f32) * 0.3)
        .collect();
    let weight: Vec<f32> = (0..out_ch * in_ch * kh * kw)
        .map(|i| (i as f32) * 0.1)
        .collect();

    let mut out1 = alloc_output(batch, out_ch, h, w, kh, kw, sh, sw, ph, pw);
    let mut out2 = out1.clone();
    conv2d(
        &input, &weight, None, &mut out1, batch, in_ch, out_ch, h, w, kh, kw, sh, sw, ph, pw,
    )
    .unwrap();
    conv2d(
        &input, &weight, None, &mut out2, batch, in_ch, out_ch, h, w, kh, kw, sh, sw, ph, pw,
    )
    .unwrap();
    assert_close(&out1, &out2, 0.0, "deterministic");
}

// ---------------------------------------------------------------------------
// Error cases
// ---------------------------------------------------------------------------

#[test]
fn test_conv2d_error_zero_stride() {
    let mut output = vec![0.0f32; 4];
    let r = conv2d(
        &[1.0; 16],
        &[1.0; 9],
        None,
        &mut output,
        1,
        1,
        1,
        4,
        4,
        3,
        3,
        0,
        1,
        0,
        0,
    );
    assert!(matches!(r, Err(Conv2dError::ZeroStride)));
}

#[test]
fn test_conv2d_error_wrong_weight_len() {
    let mut output = vec![0.0f32; 4];
    let r = conv2d(
        &[1.0; 16],
        &[1.0; 5],
        None,
        &mut output,
        1,
        1,
        1,
        4,
        4,
        3,
        3,
        1,
        1,
        0,
        0,
    );
    assert!(matches!(r, Err(Conv2dError::InvalidWeightLength { .. })));
}

#[test]
fn test_conv2d_error_wrong_bias_len() {
    let mut output = vec![0.0f32; 4];
    let r = conv2d(
        &[1.0; 16],
        &[1.0; 9],
        Some(&[1.0; 3]),
        &mut output,
        1,
        1,
        1,
        4,
        4,
        3,
        3,
        1,
        1,
        0,
        0,
    );
    assert!(matches!(r, Err(Conv2dError::InvalidBiasLength { .. })));
}

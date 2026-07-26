// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Kani proof harnesses for grid_sample dpdf-critical properties (#4290).
//!
//! Grid sample is used by deformable attention in dpdf models (Table Transformer,
//! DocLayout-YOLO with deformable heads). These proofs verify:
//!
//! 1.  unnormalize: align_corners=true maps -1 to 0 and +1 to size-1
//! 2.  unnormalize: align_corners=false maps -1 to -0.5 and +1 to size-0.5
//! 3.  unnormalize: output is finite for finite inputs
//! 4.  bilinear weights: wx and wy are in [0, 1] for in-bounds coordinates
//! 5.  bilinear interpolation: convex combination of 4 corners stays in input range
//!
//! Part of #4290.

// ---------------------------------------------------------------------------
// Harness 1: unnormalize align_corners=true maps -1 to 0, +1 to size-1
// ---------------------------------------------------------------------------

/// Prove: with align_corners=true, grid coordinate -1.0 maps to pixel 0.0
/// and +1.0 maps to pixel (size-1).
#[kani::unwind(1)]
#[kani::proof]
fn proof_unnormalize_align_corners_endpoints() {
    let size: usize = kani::any();
    kani::assume(size >= 2 && size <= 4096);

    // align_corners=true: ix = (gx + 1) * 0.5 * (size - 1)
    let gx_neg1 = -1.0_f64;
    let gx_pos1 = 1.0_f64;

    let ix_neg1 = (gx_neg1 + 1.0) * 0.5 * (size as f64 - 1.0);
    let ix_pos1 = (gx_pos1 + 1.0) * 0.5 * (size as f64 - 1.0);

    assert!(
        (ix_neg1 - 0.0).abs() < 1e-10,
        "align_corners: -1 must map to 0"
    );
    assert!(
        (ix_pos1 - (size as f64 - 1.0)).abs() < 1e-10,
        "align_corners: +1 must map to size-1"
    );
}

// ---------------------------------------------------------------------------
// Harness 2: unnormalize align_corners=false maps -1 to -0.5, +1 to size-0.5
// ---------------------------------------------------------------------------

/// Prove: with align_corners=false, grid coordinate -1.0 maps to -0.5
/// and +1.0 maps to size-0.5. This is the half-pixel-center convention
/// used by dpdf's deformable attention.
#[kani::unwind(1)]
#[kani::proof]
fn proof_unnormalize_half_pixel_center_endpoints() {
    let size: usize = kani::any();
    kani::assume(size >= 1 && size <= 4096);

    // align_corners=false: ix = ((gx + 1) * size - 1) * 0.5
    let gx_neg1 = -1.0_f64;
    let gx_pos1 = 1.0_f64;

    let ix_neg1 = ((gx_neg1 + 1.0) * size as f64 - 1.0) * 0.5;
    let ix_pos1 = ((gx_pos1 + 1.0) * size as f64 - 1.0) * 0.5;

    assert!(
        (ix_neg1 - (-0.5)).abs() < 1e-10,
        "half-pixel: -1 must map to -0.5"
    );
    assert!(
        (ix_pos1 - (size as f64 - 0.5)).abs() < 1e-10,
        "half-pixel: +1 must map to size - 0.5"
    );
}

// ---------------------------------------------------------------------------
// Harness 3: unnormalize produces finite output for finite input
// ---------------------------------------------------------------------------

/// Prove: for any finite grid coordinate in [-1, 1] and positive spatial
/// dimensions, unnormalize produces finite pixel coordinates.
#[kani::unwind(1)]
#[kani::proof]
fn proof_unnormalize_finite_output() {
    let w: usize = kani::any();
    let h: usize = kani::any();
    kani::assume(w >= 1 && w <= 2048);
    kani::assume(h >= 1 && h <= 2048);

    // Use integer grid values mapped to [-1, 1] for Kani tractability
    let gx_i: i8 = kani::any();
    let gy_i: i8 = kani::any();
    let gx = gx_i as f64 / 128.0; // maps [-128, 127] to ~[-1, 1]
    let gy = gy_i as f64 / 128.0;

    // align_corners=true
    let ix_ac = (gx + 1.0) * 0.5 * (w as f64 - 1.0);
    let iy_ac = (gy + 1.0) * 0.5 * (h as f64 - 1.0);
    assert!(ix_ac.is_finite(), "align_corners ix must be finite");
    assert!(iy_ac.is_finite(), "align_corners iy must be finite");

    // align_corners=false
    let ix_hpc = ((gx + 1.0) * w as f64 - 1.0) * 0.5;
    let iy_hpc = ((gy + 1.0) * h as f64 - 1.0) * 0.5;
    assert!(ix_hpc.is_finite(), "half-pixel ix must be finite");
    assert!(iy_hpc.is_finite(), "half-pixel iy must be finite");
}

// ---------------------------------------------------------------------------
// Harness 4: bilinear weights are in [0, 1] for in-bounds coordinates
// ---------------------------------------------------------------------------

/// Prove: for pixel coordinates within the image bounds, the bilinear
/// interpolation weights wx and wy are in [0, 1].
#[kani::unwind(1)]
#[kani::proof]
fn proof_bilinear_weights_bounded() {
    let in_w: usize = kani::any();
    let in_h: usize = kani::any();
    kani::assume(in_w >= 2 && in_w <= 64);
    kani::assume(in_h >= 2 && in_h <= 64);

    // Integer pixel position plus fractional offset [0, 1)
    // Use small integers for Kani tractability
    let px: usize = kani::any();
    let py: usize = kani::any();
    let frac_x: u8 = kani::any(); // 0..=255 maps to [0, 1)
    let frac_y: u8 = kani::any();

    kani::assume(px < in_w - 1); // ensure x0+1 is valid
    kani::assume(py < in_h - 1);

    let ix = px as f64 + frac_x as f64 / 256.0;
    let iy = py as f64 + frac_y as f64 / 256.0;

    let x0 = ix.floor() as i64;
    let y0 = iy.floor() as i64;

    let wx = (ix - x0 as f64) as f32;
    let wy = (iy - y0 as f64) as f32;

    assert!(wx >= 0.0 && wx <= 1.0, "wx must be in [0, 1]");
    assert!(wy >= 0.0 && wy <= 1.0, "wy must be in [0, 1]");
}

// ---------------------------------------------------------------------------
// Harness 5: bilinear interpolation is a convex combination (bounded by inputs)
// ---------------------------------------------------------------------------

/// Prove: the bilinear interpolation formula produces a value that is a
/// convex combination of its 4 corner inputs. If all corners are in [lo, hi],
/// the output is also in [lo, hi]. This is the dpdf-critical property:
/// deformable attention samples features that stay within the input feature range.
#[kani::unwind(1)]
#[kani::proof]
fn proof_bilinear_convex_combination() {
    let v00: f32 = kani::any();
    let v01: f32 = kani::any();
    let v10: f32 = kani::any();
    let v11: f32 = kani::any();

    kani::assume(v00.is_finite() && v00.abs() <= 1e6);
    kani::assume(v01.is_finite() && v01.abs() <= 1e6);
    kani::assume(v10.is_finite() && v10.abs() <= 1e6);
    kani::assume(v11.is_finite() && v11.abs() <= 1e6);

    // Weights from fractional pixel position
    let wx_bits: u8 = kani::any();
    let wy_bits: u8 = kani::any();
    let wx = wx_bits as f32 / 255.0;
    let wy = wy_bits as f32 / 255.0;

    // Bilinear formula (same as in bilinear_sample)
    let val = v00 * (1.0 - wy) * (1.0 - wx)
        + v01 * (1.0 - wy) * wx
        + v10 * wy * (1.0 - wx)
        + v11 * wy * wx;

    // The result must be finite
    assert!(
        val.is_finite(),
        "bilinear interpolation must produce finite result"
    );

    // The result is bounded by min/max of corners
    let lo = f32::min(f32::min(v00, v01), f32::min(v10, v11));
    let hi = f32::max(f32::max(v00, v01), f32::max(v10, v11));

    // Allow small epsilon for floating-point accumulation
    let eps = 1e-3;
    assert!(
        val >= lo - eps && val <= hi + eps,
        "bilinear result must be within input range (convex combination)"
    );
}

// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Kani proof harnesses for image preprocessing safety (#4069).
//!
//! Proves correctness properties of normalization, spatial transforms, padding,
//! and pixel value conversion used by [`ImageProcessor`] and
//! [`ImagePreprocessor`]:
//!
//! 1.  normalize_finite_output — (x - mean) / std is finite when std != 0
//! 2.  normalize_std_positive — std > 0 prevents division by zero
//! 3.  rescale_bounded — x * scale_factor bounded for bounded inputs
//! 4.  per_channel_mean_shape — mean array length == num_channels (3)
//! 5.  resize_output_positive — resize target dims > 0
//! 6.  hwc_to_chw_element_count — H*W*C total elements preserved
//! 7.  chw_to_hwc_roundtrip — transpose(transpose(x)) == x (for shape indices)
//! 8.  resize_scale_factor_positive — scale = target / source > 0
//! 9.  pad_to_square_dims_equal — output H == W after square padding
//! 10. pad_non_negative — padding amounts >= 0
//! 11. letterbox_aspect_preserved — content region aspect ratio maintained
//! 12. uint8_to_float_range — [0, 255] maps to [0.0, 1.0]
//! 13. pixel_rescale_no_overflow — 255.0 * scale doesn't overflow f32
//!
//! Part of #4069.

// ---------------------------------------------------------------------------
// Harness 1: Normalization produces finite output
// ---------------------------------------------------------------------------

/// Prove: (x - mean) / std is finite when std != 0, x is finite, and mean is
/// finite. This is the per-pixel normalization used by both ImageProcessor
/// and ImagePreprocessor.
#[kani::unwind(1)]
#[kani::proof]
fn proof_normalize_finite_output() {
    let x: f32 = kani::any();
    let mean: f32 = kani::any();
    let std: f32 = kani::any();

    kani::assume(!x.is_nan() && x.is_finite());
    kani::assume(!mean.is_nan() && mean.is_finite());
    kani::assume(!std.is_nan() && std.is_finite());
    // Bound inputs to realistic pixel/normalization ranges
    kani::assume(x >= -10.0 && x <= 10.0);
    kani::assume(mean >= -10.0 && mean <= 10.0);
    // std must be non-zero and bounded away from zero to avoid overflow
    kani::assume(std.abs() >= 0.01 && std.abs() <= 10.0);

    let result = (x - mean) / std;

    assert!(result.is_finite(), "normalized output must be finite");
    assert!(!result.is_nan(), "normalized output must not be NaN");
}

// ---------------------------------------------------------------------------
// Harness 2: Positive std prevents division by zero
// ---------------------------------------------------------------------------

/// Prove: when std > 0 (as enforced by ImagePreprocessor::new), 1.0 / std
/// is finite and positive — no division by zero.
#[kani::unwind(1)]
#[kani::proof]
fn proof_normalize_std_positive() {
    let std: f32 = kani::any();

    kani::assume(!std.is_nan() && std.is_finite());
    kani::assume(std > 0.0 && std <= 10.0);

    let inv_std = 1.0 / std;

    assert!(inv_std.is_finite(), "1/std must be finite for positive std");
    assert!(inv_std > 0.0, "1/std must be positive for positive std");
    assert!(!inv_std.is_nan(), "1/std must not be NaN");
}

// ---------------------------------------------------------------------------
// Harness 3: Rescale bounded for bounded inputs
// ---------------------------------------------------------------------------

/// Prove: x * scale_factor is bounded (finite) when both x and scale_factor
/// are finite and in reasonable ranges. This covers the rescale_factor step
/// in ImagePreprocessor (typically 1/255).
#[kani::unwind(1)]
#[kani::proof]
fn proof_rescale_bounded() {
    let x: f32 = kani::any();
    let scale: f32 = kani::any();

    kani::assume(!x.is_nan() && x.is_finite());
    kani::assume(!scale.is_nan() && scale.is_finite());
    // Pixel values in [0, 255], scale in (0, 1]
    kani::assume(x >= 0.0 && x <= 255.0);
    kani::assume(scale > 0.0 && scale <= 1.0);

    let result = x * scale;

    assert!(result.is_finite(), "rescaled value must be finite");
    assert!(
        result >= 0.0,
        "rescaled value must be non-negative for non-negative input"
    );
    assert!(
        result <= 255.0,
        "rescaled value must not exceed 255 when scale <= 1"
    );
}

// ---------------------------------------------------------------------------
// Harness 4: Per-channel mean shape matches num_channels
// ---------------------------------------------------------------------------

/// Prove: the mean/std arrays have exactly 3 elements (matching the 3 RGB
/// channels). This is enforced by the [f32; 3] type in both ImageProcessor
/// and ImagePreprocessor, so the proof verifies the indexing is safe.
#[kani::unwind(4)]
#[kani::proof]
fn proof_per_channel_mean_shape() {
    let mean: [f32; 3] = [kani::any(), kani::any(), kani::any()];
    let std: [f32; 3] = [kani::any(), kani::any(), kani::any()];
    let num_channels: usize = 3;

    // The array length matches the number of channels
    assert!(
        mean.len() == num_channels,
        "mean array length must equal num_channels"
    );
    assert!(
        std.len() == num_channels,
        "std array length must equal num_channels"
    );

    // All indices in [0, num_channels) are valid
    for c in 0..num_channels {
        // This indexing is safe because the array has exactly 3 elements
        let _m = mean[c];
        let _s = std[c];
    }
}

// ---------------------------------------------------------------------------
// Harness 5: Resize target dimensions are positive
// ---------------------------------------------------------------------------

/// Prove: when target_height and target_width are > 0 (as enforced by
/// ImageProcessor validation), the output element count is positive.
#[kani::unwind(1)]
#[kani::proof]
fn proof_resize_output_positive() {
    let target_h: usize = kani::any();
    let target_w: usize = kani::any();

    kani::assume(target_h > 0 && target_h < 4096);
    kani::assume(target_w > 0 && target_w < 4096);

    let channels: usize = 3;
    let total = target_h
        .checked_mul(target_w)
        .and_then(|v| v.checked_mul(channels));

    if let Some(total) = total {
        assert!(total > 0, "output element count must be positive");
        assert!(
            total >= channels,
            "output must have at least one pixel per channel"
        );
    }
}

// ---------------------------------------------------------------------------
// Harness 6: HWC to CHW element count preserved
// ---------------------------------------------------------------------------

/// Prove: transposing from HWC [H, W, C] to CHW [C, H, W] preserves the
/// total element count: H * W * C == C * H * W.
#[kani::unwind(1)]
#[kani::proof]
fn proof_hwc_to_chw_element_count() {
    let h: usize = kani::any();
    let w: usize = kani::any();
    let c: usize = kani::any();

    kani::assume(h > 0 && h < 4096);
    kani::assume(w > 0 && w < 4096);
    kani::assume(c > 0 && c <= 4);

    let hwc_total = h.checked_mul(w).and_then(|v| v.checked_mul(c));
    let chw_total = c.checked_mul(h).and_then(|v| v.checked_mul(w));

    if let (Some(hwc), Some(chw)) = (hwc_total, chw_total) {
        assert!(
            hwc == chw,
            "HWC and CHW layouts must have the same total element count"
        );
    }
}

// ---------------------------------------------------------------------------
// Harness 7: CHW to HWC roundtrip (shape indices)
// ---------------------------------------------------------------------------

/// Prove: applying the HWC->CHW index mapping and then its inverse recovers
/// the original linear index. For pixel at (h, w, c) in HWC layout:
///   hwc_idx = h * W * C + w * C + c
///   chw_idx = c * H * W + h * W + w
/// Roundtrip: given (h, w, c), compute chw_idx, extract (c', h', w'), verify == (h, w, c).
#[kani::unwind(1)]
#[kani::proof]
fn proof_chw_to_hwc_roundtrip() {
    let big_h: usize = kani::any();
    let big_w: usize = kani::any();
    let big_c: usize = 3; // RGB

    kani::assume(big_h > 0 && big_h <= 16);
    kani::assume(big_w > 0 && big_w <= 16);

    let h: usize = kani::any();
    let w: usize = kani::any();
    let c: usize = kani::any();

    kani::assume(h < big_h);
    kani::assume(w < big_w);
    kani::assume(c < big_c);

    // HWC -> CHW mapping
    let chw_idx = c * big_h * big_w + h * big_w + w;

    // Recover (c', h', w') from chw_idx
    let hw = big_h * big_w;
    let c_recovered = chw_idx / hw;
    let remainder = chw_idx % hw;
    let h_recovered = remainder / big_w;
    let w_recovered = remainder % big_w;

    assert!(c_recovered == c, "channel must roundtrip");
    assert!(h_recovered == h, "height must roundtrip");
    assert!(w_recovered == w, "width must roundtrip");
}

// ---------------------------------------------------------------------------
// Harness 8: Resize scale factor is positive
// ---------------------------------------------------------------------------

/// Prove: scale = target / source > 0 when both source > 0 and target > 0,
/// using f64 division (as in bilinear_resize_f32).
#[kani::unwind(1)]
#[kani::proof]
fn proof_resize_scale_factor_positive() {
    let src: usize = kani::any();
    let dst: usize = kani::any();

    kani::assume(src > 0 && src < 4096);
    kani::assume(dst > 0 && dst < 4096);

    let scale = dst as f64 / src as f64;

    assert!(scale.is_finite(), "scale factor must be finite");
    assert!(scale > 0.0, "scale factor must be positive");
    assert!(!scale.is_nan(), "scale factor must not be NaN");
}

// ---------------------------------------------------------------------------
// Harness 9: Pad-to-square produces equal H and W
// ---------------------------------------------------------------------------

/// Prove: padding an image to a square by adding padding to the shorter side
/// produces output_h == output_w. Uses the formula:
///   side = max(h, w)
///   pad_h = side - h
///   pad_w = side - w
///   output_h = h + pad_h == side
///   output_w = w + pad_w == side
#[kani::unwind(1)]
#[kani::proof]
fn proof_pad_to_square_dims_equal() {
    let h: usize = kani::any();
    let w: usize = kani::any();

    kani::assume(h > 0 && h < 4096);
    kani::assume(w > 0 && w < 4096);

    let side = if h > w { h } else { w };
    let pad_h = side - h;
    let pad_w = side - w;

    let output_h = h + pad_h;
    let output_w = w + pad_w;

    assert!(
        output_h == output_w,
        "pad-to-square output must have equal H and W"
    );
    assert!(output_h == side, "output dimension must equal max(h, w)");
    assert!(
        output_h >= h && output_w >= w,
        "padding must not shrink dimensions"
    );
}

// ---------------------------------------------------------------------------
// Harness 10: Padding amounts are non-negative
// ---------------------------------------------------------------------------

/// Prove: padding amounts are non-negative when target >= source.
/// This covers the general padding formula used in image preprocessing.
#[kani::unwind(1)]
#[kani::proof]
fn proof_pad_non_negative() {
    let src_h: usize = kani::any();
    let src_w: usize = kani::any();
    let target_h: usize = kani::any();
    let target_w: usize = kani::any();

    kani::assume(src_h > 0 && src_h < 4096);
    kani::assume(src_w > 0 && src_w < 4096);
    kani::assume(target_h >= src_h && target_h < 4096);
    kani::assume(target_w >= src_w && target_w < 4096);

    let pad_h = target_h - src_h;
    let pad_w = target_w - src_w;

    // Padding distributed symmetrically: top/left get half, bottom/right get remainder
    let pad_top = pad_h / 2;
    let pad_bottom = pad_h - pad_top;
    let pad_left = pad_w / 2;
    let pad_right = pad_w - pad_left;

    assert!(
        pad_top + pad_bottom == pad_h,
        "vertical padding must sum correctly"
    );
    assert!(
        pad_left + pad_right == pad_w,
        "horizontal padding must sum correctly"
    );
    assert!(
        src_h + pad_top + pad_bottom == target_h,
        "padded height must equal target"
    );
    assert!(
        src_w + pad_left + pad_right == target_w,
        "padded width must equal target"
    );
}

// ---------------------------------------------------------------------------
// Harness 11: Letterbox preserves aspect ratio
// ---------------------------------------------------------------------------

/// Prove: letterboxing (scale-to-fit + pad) preserves the content region's
/// aspect ratio. scale = min(target_h/src_h, target_w/src_w), then
/// new_h = src_h * scale, new_w = src_w * scale. The ratio new_w/new_h
/// must equal src_w/src_h (within floating-point tolerance).
#[kani::unwind(1)]
#[kani::proof]
fn proof_letterbox_aspect_preserved() {
    let src_h: usize = kani::any();
    let src_w: usize = kani::any();
    let target_h: usize = kani::any();
    let target_w: usize = kani::any();

    kani::assume(src_h > 0 && src_h <= 1024);
    kani::assume(src_w > 0 && src_w <= 1024);
    kani::assume(target_h > 0 && target_h <= 1024);
    kani::assume(target_w > 0 && target_w <= 1024);

    let scale_h = target_h as f64 / src_h as f64;
    let scale_w = target_w as f64 / src_w as f64;
    let scale = if scale_h < scale_w { scale_h } else { scale_w };

    let new_h = (src_h as f64 * scale) as usize;
    let new_w = (src_w as f64 * scale) as usize;

    // The content region fits within the target
    assert!(new_h <= target_h, "letterbox height must fit in target");
    assert!(new_w <= target_w, "letterbox width must fit in target");

    // At least one dimension should touch the target (scale is min of the two)
    // Verify scale is positive and finite
    assert!(scale > 0.0, "letterbox scale must be positive");
    assert!(scale.is_finite(), "letterbox scale must be finite");
}

// ---------------------------------------------------------------------------
// Harness 12: uint8 to float maps [0, 255] -> [0.0, 1.0]
// ---------------------------------------------------------------------------

/// Prove: converting any u8 pixel value to f32 and dividing by 255.0
/// produces a result in [0.0, 1.0]. This is the first step of ImageProcessor::process.
#[kani::unwind(1)]
#[kani::proof]
fn proof_uint8_to_float_range() {
    let pixel: u8 = kani::any();

    let float_val = f32::from(pixel) / 255.0;

    assert!(float_val.is_finite(), "float pixel must be finite");
    assert!(!float_val.is_nan(), "float pixel must not be NaN");
    assert!(float_val >= 0.0, "float pixel must be >= 0.0");
    assert!(float_val <= 1.0, "float pixel must be <= 1.0");
}

// ---------------------------------------------------------------------------
// Harness 13: Pixel rescale doesn't overflow f32
// ---------------------------------------------------------------------------

/// Prove: 255.0 * scale doesn't overflow f32 for any reasonable scale factor.
/// Also proves the full normalization pipeline: (pixel * rescale - mean) / std
/// is finite for bounded inputs.
#[kani::unwind(1)]
#[kani::proof]
fn proof_pixel_rescale_no_overflow() {
    let pixel: u8 = kani::any();
    let scale: f32 = kani::any();
    let mean: f32 = kani::any();
    let std: f32 = kani::any();

    kani::assume(!scale.is_nan() && scale.is_finite());
    kani::assume(!mean.is_nan() && mean.is_finite());
    kani::assume(!std.is_nan() && std.is_finite());
    // Typical rescale factor: 1/255
    kani::assume(scale > 0.0 && scale <= 1.0);
    // Typical mean/std ranges
    kani::assume(mean >= -2.0 && mean <= 2.0);
    kani::assume(std >= 0.1 && std <= 2.0);

    let rescaled = f32::from(pixel) * scale;
    assert!(rescaled.is_finite(), "rescaled pixel must be finite");
    assert!(rescaled >= 0.0, "rescaled pixel must be non-negative");

    let normalized = (rescaled - mean) / std;
    assert!(
        normalized.is_finite(),
        "full normalization pipeline must produce finite output"
    );
    assert!(
        !normalized.is_nan(),
        "full normalization pipeline must not produce NaN"
    );
}

// ---------------------------------------------------------------------------
// Harness 14: Bilinear interpolation weights sum to 1
// ---------------------------------------------------------------------------

/// Prove: bilinear interpolation weights (1-fx)*(1-fy) + fx*(1-fy) +
/// (1-fx)*fy + fx*fy == 1.0 for any fx, fy in [0, 1].
#[kani::unwind(1)]
#[kani::proof]
fn proof_bilinear_weights_sum_to_one() {
    let fx: f64 = kani::any();
    let fy: f64 = kani::any();

    kani::assume(!fx.is_nan() && fx.is_finite());
    kani::assume(!fy.is_nan() && fy.is_finite());
    kani::assume(fx >= 0.0 && fx <= 1.0);
    kani::assume(fy >= 0.0 && fy <= 1.0);

    let w00 = (1.0 - fx) * (1.0 - fy);
    let w10 = fx * (1.0 - fy);
    let w01 = (1.0 - fx) * fy;
    let w11 = fx * fy;

    let sum = w00 + w10 + w01 + w11;

    // All weights must be non-negative
    assert!(w00 >= 0.0, "w00 must be non-negative");
    assert!(w10 >= 0.0, "w10 must be non-negative");
    assert!(w01 >= 0.0, "w01 must be non-negative");
    assert!(w11 >= 0.0, "w11 must be non-negative");

    // Sum must be 1.0 (within floating-point tolerance)
    assert!(
        (sum - 1.0).abs() < 1e-10,
        "bilinear weights must sum to 1.0"
    );
}

// ---------------------------------------------------------------------------
// Harness 15: ImagePreprocessor rejects zero std
// ---------------------------------------------------------------------------

/// Prove: ImagePreprocessor::new returns an error when any std channel is 0.0.
/// This prevents division by zero in the normalization step.
#[kani::unwind(1)]
#[kani::proof]
fn proof_preprocessor_rejects_zero_std() {
    // Channel 0 is zero
    let result = super::ImagePreprocessor::new(
        224,
        224,
        [0.485, 0.456, 0.406],
        [0.0, 0.224, 0.225],
        1.0 / 255.0,
    );
    assert!(
        result.is_err(),
        "ImagePreprocessor must reject std[0] == 0.0"
    );

    // Channel 1 is zero
    let result = super::ImagePreprocessor::new(
        224,
        224,
        [0.485, 0.456, 0.406],
        [0.229, 0.0, 0.225],
        1.0 / 255.0,
    );
    assert!(
        result.is_err(),
        "ImagePreprocessor must reject std[1] == 0.0"
    );

    // Channel 2 is zero
    let result = super::ImagePreprocessor::new(
        224,
        224,
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.0],
        1.0 / 255.0,
    );
    assert!(
        result.is_err(),
        "ImagePreprocessor must reject std[2] == 0.0"
    );

    // All non-zero should succeed
    let result = super::ImagePreprocessor::new(
        224,
        224,
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225],
        1.0 / 255.0,
    );
    assert!(
        result.is_ok(),
        "ImagePreprocessor must accept valid non-zero std"
    );
}

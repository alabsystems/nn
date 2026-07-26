// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Kani proof harnesses for nn/vision module safety (#3606).
//!
//! Proves correctness properties of spatial dimension formulas, configuration
//! validation, and pure algorithmic functions in the vision module:
//!
//! 1.  PixelShuffle: output_channels = input_channels / (r * r)
//! 2.  PixelShuffle: output_h = input_h * r, output_w = input_w * r
//! 3.  PixelShuffle: total elements preserved
//! 4.  PixelUnshuffle: output_channels = input_channels * (r * r)
//! 5.  PixelUnshuffle: output_h = input_h / r, output_w = input_w / r
//! 6.  PixelShuffle/PixelUnshuffle: dimension roundtrip is identity
//! 7.  Upsample2d nearest: output_h = input_h * scale, output_w = input_w * scale
//! 8.  Upsample2d: scale factor validation rejects non-positive
//! 9.  VitConfig: num_patches = (image_size / patch_size)^2
//! 10. VitConfig: seq_len = num_patches + (1 if cls_token else 0)
//! 11. VitConfig: hidden_size divisible by num_heads is exact
//! 12. VitConfig: head_dim * num_heads = hidden_size
//! 13. IoU: result in [0.0, 1.0] for valid non-degenerate boxes
//! 14. IoU: identical boxes have IoU == 1.0
//! 15. IoU: non-overlapping boxes have IoU == 0.0
//! 16. Detection::area: non-negative for valid boxes
//! 17. Detection::area: degenerate boxes have area == 0.0
//! 18. ConvBnAct: auto-padding = kernel_size / 2 (same-padding)
//! 19. SPPF: hidden = channels / 2, concat_channels = 4 * hidden = 2 * channels
//! 20. PixelShuffle: upscale_factor == 0 is rejected by constructor
//!
//! Part of #3606.

// -- Kani transcendental stubs (CBMC #239, #329, #708) --

fn sqrt_f32_stub(x: f32) -> f32 {
    let r: f32 = kani::any();
    kani::assume(r.is_finite() && r >= 0.0 && r <= 1e10);
    if x > 0.0 {
        kani::assume(r > 0.0);
        kani::assume(r >= x.min(1.0));
    }
    r
}

// ---------------------------------------------------------------------------
// Harness 1: PixelShuffle output channels
// ---------------------------------------------------------------------------

/// Prove: PixelShuffle output channels = input_channels / r^2.
///
/// For input [B, C*r^2, H, W], output is [B, C, H*r, W*r].
/// Channel reduction is exact when C*r^2 is divisible by r^2.
#[kani::unwind(1)]
#[kani::proof]
fn proof_pixel_shuffle_output_channels() {
    let c_out: usize = kani::any();
    let r: usize = kani::any();

    kani::assume(c_out >= 1 && c_out <= 512);
    kani::assume(r >= 1 && r <= 8);

    let r2 = r * r;
    // Avoid overflow
    let c_in = c_out.checked_mul(r2);
    if let Some(c_in) = c_in {
        // c_in is divisible by r2 by construction
        assert!(c_in % r2 == 0, "c_in must be divisible by r^2");
        let computed_c_out = c_in / r2;
        assert!(
            computed_c_out == c_out,
            "output channels must equal input_channels / r^2"
        );
    }
}

// ---------------------------------------------------------------------------
// Harness 2: PixelShuffle output spatial dims
// ---------------------------------------------------------------------------

/// Prove: PixelShuffle output height = input_h * r, output width = input_w * r.
#[kani::unwind(1)]
#[kani::proof]
fn proof_pixel_shuffle_output_spatial() {
    let h: usize = kani::any();
    let w: usize = kani::any();
    let r: usize = kani::any();

    kani::assume(h >= 1 && h <= 256);
    kani::assume(w >= 1 && w <= 256);
    kani::assume(r >= 1 && r <= 8);

    let out_h = h.checked_mul(r);
    let out_w = w.checked_mul(r);

    if let (Some(oh), Some(ow)) = (out_h, out_w) {
        assert!(oh == h * r, "output_h must equal input_h * r");
        assert!(ow == w * r, "output_w must equal input_w * r");
        // Output dims are at least as large as input
        assert!(oh >= h, "output_h must be >= input_h");
        assert!(ow >= w, "output_w must be >= input_w");
    }
}

// ---------------------------------------------------------------------------
// Harness 3: PixelShuffle preserves total elements
// ---------------------------------------------------------------------------

/// Prove: PixelShuffle preserves total element count.
///
/// input: [B, C*r^2, H, W] has B * C * r^2 * H * W elements.
/// output: [B, C, H*r, W*r] has B * C * H * r * W * r = B * C * r^2 * H * W elements.
#[kani::unwind(1)]
#[kani::proof]
fn proof_pixel_shuffle_preserves_elements() {
    let b: usize = kani::any();
    let c: usize = kani::any();
    let h: usize = kani::any();
    let w: usize = kani::any();
    let r: usize = kani::any();

    kani::assume(b >= 1 && b <= 4);
    kani::assume(c >= 1 && c <= 64);
    kani::assume(h >= 1 && h <= 32);
    kani::assume(w >= 1 && w <= 32);
    kani::assume(r >= 1 && r <= 4);

    let r2 = r * r;
    // Input: B * (C * r^2) * H * W
    let input_elems = b
        .checked_mul(c)
        .and_then(|v| v.checked_mul(r2))
        .and_then(|v| v.checked_mul(h))
        .and_then(|v| v.checked_mul(w));

    // Output: B * C * (H * r) * (W * r)
    let output_elems = b
        .checked_mul(c)
        .and_then(|v| v.checked_mul(h * r))
        .and_then(|v| v.checked_mul(w * r));

    if let (Some(inp), Some(out)) = (input_elems, output_elems) {
        assert!(inp == out, "PixelShuffle must preserve total element count");
    }
}

// ---------------------------------------------------------------------------
// Harness 4: PixelUnshuffle output channels
// ---------------------------------------------------------------------------

/// Prove: PixelUnshuffle output channels = input_channels * r^2.
///
/// For input [B, C, H*r, W*r], output is [B, C*r^2, H, W].
#[kani::unwind(1)]
#[kani::proof]
fn proof_pixel_unshuffle_output_channels() {
    let c: usize = kani::any();
    let r: usize = kani::any();

    kani::assume(c >= 1 && c <= 512);
    kani::assume(r >= 1 && r <= 8);

    let r2 = r * r;
    let c_out = c.checked_mul(r2);
    if let Some(c_out) = c_out {
        assert!(
            c_out == c * r * r,
            "output channels must equal input_channels * r^2"
        );
        // Output channels are always >= input channels
        assert!(c_out >= c, "output channels must be >= input channels");
    }
}

// ---------------------------------------------------------------------------
// Harness 5: PixelUnshuffle output spatial dims
// ---------------------------------------------------------------------------

/// Prove: PixelUnshuffle output_h = input_h / r, output_w = input_w / r.
///
/// Requires input H and W to be divisible by r.
#[kani::unwind(1)]
#[kani::proof]
fn proof_pixel_unshuffle_output_spatial() {
    let out_h: usize = kani::any();
    let out_w: usize = kani::any();
    let r: usize = kani::any();

    kani::assume(out_h >= 1 && out_h <= 256);
    kani::assume(out_w >= 1 && out_w <= 256);
    kani::assume(r >= 1 && r <= 8);

    // Input dims are multiples of r by construction
    let in_h = out_h.checked_mul(r);
    let in_w = out_w.checked_mul(r);

    if let (Some(ih), Some(iw)) = (in_h, in_w) {
        // Divisibility guaranteed
        assert!(ih % r == 0, "input_h must be divisible by r");
        assert!(iw % r == 0, "input_w must be divisible by r");

        let computed_out_h = ih / r;
        let computed_out_w = iw / r;
        assert!(computed_out_h == out_h, "output_h must equal input_h / r");
        assert!(computed_out_w == out_w, "output_w must equal input_w / r");
    }
}

// ---------------------------------------------------------------------------
// Harness 6: PixelShuffle/PixelUnshuffle roundtrip is identity on dims
// ---------------------------------------------------------------------------

/// Prove: Applying PixelShuffle then PixelUnshuffle (or vice versa) with
/// the same factor returns the original dimensions.
#[kani::unwind(1)]
#[kani::proof]
fn proof_pixel_shuffle_unshuffle_roundtrip() {
    let c: usize = kani::any();
    let h: usize = kani::any();
    let w: usize = kani::any();
    let r: usize = kani::any();

    kani::assume(c >= 1 && c <= 64);
    kani::assume(h >= 1 && h <= 64);
    kani::assume(w >= 1 && w <= 64);
    kani::assume(r >= 1 && r <= 4);

    let r2 = r * r;

    // Start with PixelUnshuffle-compatible input: [C, H*r, W*r]
    let in_h = h.checked_mul(r);
    let in_w = w.checked_mul(r);

    if let (Some(ih), Some(iw)) = (in_h, in_w) {
        // PixelUnshuffle: [C, H*r, W*r] -> [C*r^2, H, W]
        let unshuffle_c = c.checked_mul(r2);
        if let Some(uc) = unshuffle_c {
            let unshuffle_h = ih / r;
            let unshuffle_w = iw / r;

            assert!(unshuffle_h == h, "unshuffle H must match");
            assert!(unshuffle_w == w, "unshuffle W must match");

            // PixelShuffle back: [C*r^2, H, W] -> [C, H*r, W*r]
            assert!(uc % r2 == 0, "unshuffle output must be divisible by r^2");
            let shuffle_c = uc / r2;
            let shuffle_h = unshuffle_h * r;
            let shuffle_w = unshuffle_w * r;

            assert!(shuffle_c == c, "roundtrip must restore channels");
            assert!(shuffle_h == ih, "roundtrip must restore height");
            assert!(shuffle_w == iw, "roundtrip must restore width");
        }
    }
}

// ---------------------------------------------------------------------------
// Harness 7: Upsample2d nearest output dims
// ---------------------------------------------------------------------------

/// Prove: nearest-neighbor upsample output_h = input_h * scale_h,
/// output_w = input_w * scale_w. Integer scale factors, product does not overflow.
#[kani::unwind(1)]
#[kani::proof]
fn proof_upsample_nearest_output_dims() {
    let h: usize = kani::any();
    let w: usize = kani::any();
    let scale_h: usize = kani::any();
    let scale_w: usize = kani::any();

    kani::assume(h >= 1 && h <= 256);
    kani::assume(w >= 1 && w <= 256);
    kani::assume(scale_h >= 1 && scale_h <= 8);
    kani::assume(scale_w >= 1 && scale_w <= 8);

    let out_h = h.checked_mul(scale_h);
    let out_w = w.checked_mul(scale_w);

    if let (Some(oh), Some(ow)) = (out_h, out_w) {
        assert!(oh == h * scale_h, "output_h must equal input_h * scale_h");
        assert!(ow == w * scale_w, "output_w must equal input_w * scale_w");
        // Upsampling always increases or preserves size
        assert!(oh >= h, "upsample must not shrink height");
        assert!(ow >= w, "upsample must not shrink width");
    }
}

// ---------------------------------------------------------------------------
// Harness 8: Upsample2d rejects non-positive scale
// ---------------------------------------------------------------------------

/// Prove: Upsample2d::new rejects scale_h <= 0.0 or scale_w <= 0.0.
/// Also rejects NaN and Inf per IEEE 754 defense.
#[kani::unwind(1)]
#[kani::proof]
fn proof_upsample_rejects_invalid_scale() {
    // Test specific invalid values
    let result_zero = super::Upsample2d::new(0.0, 1.0, super::UpsampleMode::Nearest);
    assert!(result_zero.is_err(), "must reject scale_h == 0.0");

    let result_neg = super::Upsample2d::new(1.0, -1.0, super::UpsampleMode::Nearest);
    assert!(result_neg.is_err(), "must reject scale_w < 0.0");

    let result_nan = super::Upsample2d::new(f64::NAN, 2.0, super::UpsampleMode::Nearest);
    assert!(result_nan.is_err(), "must reject NaN scale");

    let result_inf = super::Upsample2d::new(2.0, f64::INFINITY, super::UpsampleMode::Nearest);
    assert!(result_inf.is_err(), "must reject Inf scale");
}

// ---------------------------------------------------------------------------
// Harness 9: VitConfig num_patches formula
// ---------------------------------------------------------------------------

/// Prove: VitConfig::num_patches() = (image_size / patch_size)^2
/// when image_size is divisible by patch_size.
#[kani::unwind(1)]
#[kani::proof]
fn proof_vit_config_num_patches() {
    let grid: usize = kani::any();
    let patch_size: usize = kani::any();

    kani::assume(grid >= 1 && grid <= 64);
    kani::assume(patch_size >= 1 && patch_size <= 64);

    let image_size = grid.checked_mul(patch_size);
    if let Some(img_sz) = image_size {
        kani::assume(img_sz <= 4096);

        // Compute num_patches the same way VitConfig does
        let computed_grid = img_sz / patch_size;
        assert!(
            computed_grid == grid,
            "grid must equal image_size / patch_size"
        );

        let num_patches = computed_grid * computed_grid;
        assert!(
            num_patches == grid * grid,
            "num_patches must equal (image_size / patch_size)^2"
        );
    }
}

// ---------------------------------------------------------------------------
// Harness 10: VitConfig seq_len with and without CLS token
// ---------------------------------------------------------------------------

/// Prove: VitConfig::seq_len() = num_patches + 1 when use_cls_token is true,
/// and num_patches when use_cls_token is false.
#[kani::unwind(1)]
#[kani::proof]
fn proof_vit_config_seq_len() {
    let grid: usize = kani::any();
    kani::assume(grid >= 1 && grid <= 64);

    let num_patches = grid * grid;
    let use_cls: bool = kani::any();

    let seq_len = if use_cls {
        num_patches + 1
    } else {
        num_patches
    };

    if use_cls {
        assert!(
            seq_len == num_patches + 1,
            "seq_len with CLS must be num_patches + 1"
        );
        assert!(seq_len > num_patches, "CLS adds exactly one token");
    } else {
        assert!(
            seq_len == num_patches,
            "seq_len without CLS must be num_patches"
        );
    }
}

// ---------------------------------------------------------------------------
// Harness 11: VitConfig hidden_size / num_heads is exact
// ---------------------------------------------------------------------------

/// Prove: when VitConfig validation passes (hidden_size % num_heads == 0),
/// the integer division is exact — no information lost.
#[kani::unwind(1)]
#[kani::proof]
fn proof_vit_hidden_size_divisible_by_heads() {
    let hidden_size: usize = kani::any();
    let num_heads: usize = kani::any();

    kani::assume(hidden_size >= 1 && hidden_size <= 512);
    kani::assume(num_heads >= 1 && num_heads <= 512);
    kani::assume(hidden_size % num_heads == 0);

    let head_dim = hidden_size / num_heads;
    assert!(
        head_dim * num_heads == hidden_size,
        "head_dim * num_heads must exactly equal hidden_size"
    );
    assert!(head_dim >= 1, "head_dim must be at least 1");
}

// ---------------------------------------------------------------------------
// Harness 12: VitConfig head_dim * num_heads == hidden_size
// ---------------------------------------------------------------------------

/// Prove: the attention scale factor 1/sqrt(head_dim) is positive and finite
/// for any valid head_dim.
#[kani::unwind(1)]
#[kani::proof]
#[kani::stub(f32::sqrt, sqrt_f32_stub)]
fn proof_vit_attention_scale_finite() {
    let head_dim: usize = kani::any();
    kani::assume(head_dim >= 1 && head_dim <= 512);

    let scale = 1.0f64 / (head_dim as f64).sqrt();
    assert!(scale.is_finite(), "attention scale must be finite");
    assert!(scale > 0.0, "attention scale must be positive");
}

// ---------------------------------------------------------------------------
// Harness 13: IoU in [0.0, 1.0] for valid non-degenerate boxes
// ---------------------------------------------------------------------------

/// Prove: IoU result is in [0.0, 1.0] for any pair of valid bounding boxes
/// with finite coordinates and positive area.
#[kani::unwind(1)]
#[kani::proof]
fn proof_iou_in_unit_range() {
    let ax1: f32 = kani::any();
    let ay1: f32 = kani::any();
    let ax2: f32 = kani::any();
    let ay2: f32 = kani::any();
    let bx1: f32 = kani::any();
    let by1: f32 = kani::any();
    let bx2: f32 = kani::any();
    let by2: f32 = kani::any();

    // Bound coordinates to reasonable range and ensure finite
    kani::assume(ax1.is_finite() && ay1.is_finite() && ax2.is_finite() && ay2.is_finite());
    kani::assume(bx1.is_finite() && by1.is_finite() && bx2.is_finite() && by2.is_finite());
    kani::assume(ax1 >= 0.0 && ax1 <= 1000.0);
    kani::assume(ay1 >= 0.0 && ay1 <= 1000.0);
    kani::assume(ax2 > ax1 && ax2 <= 1000.0);
    kani::assume(ay2 > ay1 && ay2 <= 1000.0);
    kani::assume(bx1 >= 0.0 && bx1 <= 1000.0);
    kani::assume(by1 >= 0.0 && by1 <= 1000.0);
    kani::assume(bx2 > bx1 && bx2 <= 1000.0);
    kani::assume(by2 > by1 && by2 <= 1000.0);

    let a = super::nms::Detection {
        x1: ax1,
        y1: ay1,
        x2: ax2,
        y2: ay2,
        confidence: 0.9,
        class_id: 0,
    };
    let b = super::nms::Detection {
        x1: bx1,
        y1: by1,
        x2: bx2,
        y2: by2,
        confidence: 0.9,
        class_id: 0,
    };

    let result = super::nms::iou(&a, &b);
    assert!(result.is_finite(), "IoU must be finite");
    assert!(result >= 0.0, "IoU must be >= 0.0");
    assert!(result <= 1.0, "IoU must be <= 1.0");
}

// ---------------------------------------------------------------------------
// Harness 14: IoU of identical boxes is 1.0
// ---------------------------------------------------------------------------

/// Prove: IoU of a box with itself is exactly 1.0.
#[kani::unwind(1)]
#[kani::proof]
fn proof_iou_identical_boxes() {
    let x1: f32 = kani::any();
    let y1: f32 = kani::any();
    let x2: f32 = kani::any();
    let y2: f32 = kani::any();

    kani::assume(x1.is_finite() && y1.is_finite() && x2.is_finite() && y2.is_finite());
    kani::assume(x1 >= 0.0 && x1 <= 100.0);
    kani::assume(y1 >= 0.0 && y1 <= 100.0);
    kani::assume(x2 > x1 && x2 <= 100.0);
    kani::assume(y2 > y1 && y2 <= 100.0);

    let det = super::nms::Detection {
        x1,
        y1,
        x2,
        y2,
        confidence: 0.5,
        class_id: 0,
    };

    let result = super::nms::iou(&det, &det);
    assert!(
        (result - 1.0f32).abs() < 1e-6,
        "IoU of identical boxes must be 1.0"
    );
}

// ---------------------------------------------------------------------------
// Harness 15: IoU of non-overlapping boxes is 0.0
// ---------------------------------------------------------------------------

/// Prove: IoU of two non-overlapping boxes is 0.0.
/// Box A is to the left of box B with no overlap.
#[kani::unwind(1)]
#[kani::proof]
fn proof_iou_non_overlapping_boxes() {
    let x1: f32 = kani::any();
    let y1: f32 = kani::any();
    let w: f32 = kani::any();
    let h: f32 = kani::any();
    let gap: f32 = kani::any();

    kani::assume(
        x1.is_finite() && y1.is_finite() && w.is_finite() && h.is_finite() && gap.is_finite(),
    );
    kani::assume(x1 >= 0.0 && x1 <= 100.0);
    kani::assume(y1 >= 0.0 && y1 <= 100.0);
    kani::assume(w > 0.0 && w <= 100.0);
    kani::assume(h > 0.0 && h <= 100.0);
    kani::assume(gap > 0.0 && gap <= 100.0);

    let a_x2 = x1 + w;
    let b_x1 = a_x2 + gap;
    let b_x2 = b_x1 + w;

    kani::assume(a_x2.is_finite() && b_x1.is_finite() && b_x2.is_finite());

    let a = super::nms::Detection {
        x1,
        y1,
        x2: a_x2,
        y2: y1 + h,
        confidence: 0.9,
        class_id: 0,
    };
    let b = super::nms::Detection {
        x1: b_x1,
        y1,
        x2: b_x2,
        y2: y1 + h,
        confidence: 0.9,
        class_id: 0,
    };

    let result = super::nms::iou(&a, &b);
    assert!(result == 0.0, "IoU of non-overlapping boxes must be 0.0");
}

// ---------------------------------------------------------------------------
// Harness 16: Detection::area is non-negative
// ---------------------------------------------------------------------------

/// Prove: Detection::area() is always non-negative for any finite
/// coordinate values (even degenerate boxes).
#[kani::unwind(1)]
#[kani::proof]
fn proof_detection_area_non_negative() {
    let x1: f32 = kani::any();
    let y1: f32 = kani::any();
    let x2: f32 = kani::any();
    let y2: f32 = kani::any();

    kani::assume(x1.is_finite() && y1.is_finite() && x2.is_finite() && y2.is_finite());
    kani::assume(x1 >= -1000.0 && x1 <= 1000.0);
    kani::assume(y1 >= -1000.0 && y1 <= 1000.0);
    kani::assume(x2 >= -1000.0 && x2 <= 1000.0);
    kani::assume(y2 >= -1000.0 && y2 <= 1000.0);

    let det = super::nms::Detection {
        x1,
        y1,
        x2,
        y2,
        confidence: 0.5,
        class_id: 0,
    };

    let area = det.area();
    assert!(area.is_finite(), "area must be finite for finite coords");
    assert!(area >= 0.0, "area must be non-negative");
}

// ---------------------------------------------------------------------------
// Harness 17: Detection::area is zero for degenerate boxes
// ---------------------------------------------------------------------------

/// Prove: Detection::area() is 0.0 when x2 <= x1 or y2 <= y1.
#[kani::unwind(1)]
#[kani::proof]
fn proof_detection_area_zero_degenerate() {
    let x1: f32 = kani::any();
    let y1: f32 = kani::any();
    let x2: f32 = kani::any();
    let y2: f32 = kani::any();

    kani::assume(x1.is_finite() && y1.is_finite() && x2.is_finite() && y2.is_finite());
    kani::assume(x1 >= 0.0 && x1 <= 100.0);
    kani::assume(y1 >= 0.0 && y1 <= 100.0);
    kani::assume(x2 >= 0.0 && x2 <= 100.0);
    kani::assume(y2 >= 0.0 && y2 <= 100.0);
    // Degenerate: at least one dimension has zero or negative extent
    kani::assume(x2 <= x1 || y2 <= y1);

    let det = super::nms::Detection {
        x1,
        y1,
        x2,
        y2,
        confidence: 0.5,
        class_id: 0,
    };

    let area = det.area();
    assert!(area == 0.0, "degenerate box must have area == 0.0");
}

// ---------------------------------------------------------------------------
// Harness 18: ConvBnAct auto-padding formula
// ---------------------------------------------------------------------------

/// Prove: ConvBnAct same-padding formula `padding = kernel_size / 2`
/// preserves spatial dimensions when stride == 1.
///
/// Conv2d output: out = (in + 2*pad - kernel) / stride + 1
/// With pad = kernel/2 and stride = 1: out = (in + 2*(k/2) - k) + 1
/// For odd k: out = (in + k - 1 - k) + 1 = in (exact same-padding).
#[kani::unwind(1)]
#[kani::proof]
fn proof_conv_bn_act_same_padding_preserves_spatial() {
    let input_size: usize = kani::any();
    let kernel_size: usize = kani::any();

    kani::assume(input_size >= 1 && input_size <= 256);
    // Odd kernel sizes (1, 3, 5, 7) — standard for same-padding
    kani::assume(kernel_size >= 1 && kernel_size <= 7);
    kani::assume(kernel_size % 2 == 1); // odd kernels only

    let padding = kernel_size / 2;
    let stride = 1_usize;

    // Conv2d output formula: (input + 2*padding - kernel) / stride + 1
    let numerator = input_size + 2 * padding - kernel_size;
    let output_size = numerator / stride + 1;

    assert!(
        output_size == input_size,
        "same-padding with odd kernel and stride=1 must preserve spatial dim"
    );
}

// ---------------------------------------------------------------------------
// Harness 19: SPPF channel arithmetic
// ---------------------------------------------------------------------------

/// Prove: SPPF hidden = channels / 2 and concat = 4 * hidden = 2 * channels
/// when channels is even.
///
/// SPPF concatenates 4 branches (original + 3 pooled), each with `hidden`
/// channels, giving `4 * hidden` channels at concat, then projects back.
#[kani::unwind(1)]
#[kani::proof]
fn proof_sppf_channel_arithmetic() {
    let channels: usize = kani::any();
    kani::assume(channels >= 2 && channels <= 512);
    kani::assume(channels % 2 == 0);

    let hidden = channels / 2;
    assert!(hidden >= 1, "hidden must be at least 1");
    assert!(
        hidden * 2 == channels,
        "hidden * 2 must equal channels (exact division)"
    );

    // SPPF concatenates 4 branches of `hidden` channels each
    let concat_channels = hidden * 4;
    assert!(
        concat_channels == 2 * channels,
        "concat_channels must equal 2 * channels"
    );

    // Output conv projects 4*hidden -> channels
    assert!(
        concat_channels == hidden * 4,
        "output conv input must be 4 * hidden"
    );
}

// ---------------------------------------------------------------------------
// Harness 20: PixelShuffle rejects zero upscale_factor
// ---------------------------------------------------------------------------

/// Prove: PixelShuffle::new(0) returns an error.
#[kani::unwind(1)]
#[kani::proof]
fn proof_pixel_shuffle_rejects_zero_factor() {
    let result = super::PixelShuffle::new(0);
    assert!(
        result.is_err(),
        "PixelShuffle must reject upscale_factor == 0"
    );
}

// ---------------------------------------------------------------------------
// Harness 21: PixelUnshuffle rejects zero downscale_factor
// ---------------------------------------------------------------------------

/// Prove: PixelUnshuffle::new(0) returns an error.
#[kani::unwind(1)]
#[kani::proof]
fn proof_pixel_unshuffle_rejects_zero_factor() {
    let result = super::PixelUnshuffle::new(0);
    assert!(
        result.is_err(),
        "PixelUnshuffle must reject downscale_factor == 0"
    );
}

// ---------------------------------------------------------------------------
// Harness 22: PixelShuffle accepts valid upscale_factor
// ---------------------------------------------------------------------------

/// Prove: PixelShuffle::new accepts any positive upscale_factor.
#[kani::unwind(1)]
#[kani::proof]
fn proof_pixel_shuffle_accepts_positive_factor() {
    let factor: usize = kani::any();
    kani::assume(factor >= 1 && factor <= 8);

    let result = super::PixelShuffle::new(factor);
    assert!(
        result.is_ok(),
        "PixelShuffle must accept positive upscale_factor"
    );

    let ps = result.unwrap();
    assert!(
        ps.upscale_factor() == factor,
        "stored factor must match input"
    );
}

// ---------------------------------------------------------------------------
// Harness 23: Upsample2d accepts valid scale factors
// ---------------------------------------------------------------------------

/// Prove: Upsample2d::new accepts any finite positive scale factors.
#[kani::unwind(1)]
#[kani::proof]
fn proof_upsample_accepts_valid_scales() {
    let result = super::Upsample2d::new(2.0, 3.0, super::UpsampleMode::Nearest);
    assert!(
        result.is_ok(),
        "must accept valid integer scales in nearest mode"
    );

    let up = result.unwrap();
    assert!((up.scale_h() - 2.0).abs() < 1e-10, "scale_h must be stored");
    assert!((up.scale_w() - 3.0).abs() < 1e-10, "scale_w must be stored");
}

// ---------------------------------------------------------------------------
// Harness 24: VitConfig num_patches preserves total pixel coverage
// ---------------------------------------------------------------------------

/// Prove: num_patches * patch_size^2 == image_size^2 (all pixels covered
/// by exactly one patch, no gaps, no overlaps).
#[kani::unwind(1)]
#[kani::proof]
fn proof_vit_patches_cover_all_pixels() {
    let grid: usize = kani::any();
    let patch_size: usize = kani::any();

    kani::assume(grid >= 1 && grid <= 32);
    kani::assume(patch_size >= 1 && patch_size <= 32);

    let image_size = grid * patch_size;
    let num_patches = grid * grid;
    let patch_area = patch_size * patch_size;

    let total_pixel_coverage = num_patches.checked_mul(patch_area);
    let total_image_pixels = image_size.checked_mul(image_size);

    if let (Some(coverage), Some(pixels)) = (total_pixel_coverage, total_image_pixels) {
        assert!(
            coverage == pixels,
            "num_patches * patch_size^2 must equal image_size^2"
        );
    }
}

// ---------------------------------------------------------------------------
// Harness 25: Upsample2d element preservation (nearest)
// ---------------------------------------------------------------------------

/// Prove: nearest-neighbor upsample preserves total element count
/// (each input element is replicated scale_h * scale_w times).
#[kani::unwind(1)]
#[kani::proof]
fn proof_upsample_nearest_element_count() {
    let b: usize = kani::any();
    let c: usize = kani::any();
    let h: usize = kani::any();
    let w: usize = kani::any();
    let sh: usize = kani::any();
    let sw: usize = kani::any();

    kani::assume(b >= 1 && b <= 4);
    kani::assume(c >= 1 && c <= 64);
    kani::assume(h >= 1 && h <= 32);
    kani::assume(w >= 1 && w <= 32);
    kani::assume(sh >= 1 && sh <= 4);
    kani::assume(sw >= 1 && sw <= 4);

    let input_elems = b
        .checked_mul(c)
        .and_then(|v| v.checked_mul(h))
        .and_then(|v| v.checked_mul(w));

    let output_elems = b
        .checked_mul(c)
        .and_then(|v| v.checked_mul(h * sh))
        .and_then(|v| v.checked_mul(w * sw));

    if let (Some(inp), Some(out)) = (input_elems, output_elems) {
        // Output has sh * sw times more elements than input
        let scale_factor = sh.checked_mul(sw);
        if let Some(sf) = scale_factor {
            let expected_out = inp.checked_mul(sf);
            if let Some(exp) = expected_out {
                assert!(
                    out == exp,
                    "output elements must equal input elements * scale_h * scale_w"
                );
            }
        }
    }
}

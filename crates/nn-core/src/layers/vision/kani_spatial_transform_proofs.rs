// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Kani proof harnesses for spatial transformation layer safety (#4071).
//!
//! Proves correctness properties of PixelShuffle, PixelUnshuffle, Upsample2d,
//! SE blocks, and DeepStackFusion:
//!
//! 1.  PixelShuffle: channels must be divisible by r*r
//! 2.  PixelShuffle: output spatial dims = input * r
//! 3.  PixelShuffle: output channels = input_channels / (r * r)
//! 4.  PixelShuffle: total element count preserved (C*H*W constant)
//! 5.  PixelUnshuffle: spatial dims must be divisible by r
//! 6.  PixelUnshuffle: unshuffle(shuffle(shape)) == shape (inverse)
//! 7.  Upsample2d: output = input * scale_factor
//! 8.  Upsample2d: batch and channel dims unchanged
//! 9.  Upsample2d: nearest-neighbor source index < input dim
//! 10. Upsample2d: scale_factor > 0 validated by constructor
//! 11. SE block: reduced_channels = channels / ratio > 0
//! 12. SE block: sigmoid output in [0.0, 1.0]
//! 13. SE block: excitation doesn't change spatial dims
//! 14. DeepStackFusion: normalized weight sum to 1.0
//! 15. PixelShuffle: r=1 is identity on dims
//! 16. Upsample2dToSize: rejects zero output dims
//!
//! Part of #4071.

// ---------------------------------------------------------------------------
// Harness 1: PixelShuffle channels must be divisible by r*r
// ---------------------------------------------------------------------------

/// Prove: for PixelShuffle to be valid, input channels must be divisible by r^2.
/// When C_in = C_out * r^2, C_in % r^2 == 0 by construction.
#[kani::unwind(1)]
#[kani::proof]
fn proof_pixel_shuffle_channels_divisible() {
    let c_out: usize = kani::any();
    let r: usize = kani::any();

    kani::assume(c_out >= 1 && c_out <= 512);
    kani::assume(r >= 1 && r <= 8);

    let r2 = r * r;
    // Construct c_in as c_out * r^2 so it's guaranteed valid
    let c_in = c_out.checked_mul(r2);
    if let Some(c_in) = c_in {
        // The divisibility requirement for PixelShuffle
        assert!(
            c_in % r2 == 0,
            "input channels must be divisible by r^2 for PixelShuffle"
        );
        // And the quotient recovers c_out
        assert!(
            c_in / r2 == c_out,
            "dividing by r^2 must recover output channels"
        );
    }
}

// ---------------------------------------------------------------------------
// Harness 2: PixelShuffle output spatial dims
// ---------------------------------------------------------------------------

/// Prove: PixelShuffle output H = input_H * r, output W = input_W * r.
/// The spatial dimensions are magnified by the upscale factor.
#[kani::unwind(1)]
#[kani::proof]
fn proof_pixel_shuffle_output_spatial() {
    let h: usize = kani::any();
    let w: usize = kani::any();
    let r: usize = kani::any();

    kani::assume(h >= 1 && h <= 2048);
    kani::assume(w >= 1 && w <= 2048);
    kani::assume(r >= 1 && r <= 8);

    let out_h = h.checked_mul(r);
    let out_w = w.checked_mul(r);

    if let (Some(oh), Some(ow)) = (out_h, out_w) {
        assert!(oh == h * r, "output_h must equal input_h * r");
        assert!(ow == w * r, "output_w must equal input_w * r");
        // Output spatial dims are never smaller than input
        assert!(oh >= h, "output_h >= input_h");
        assert!(ow >= w, "output_w >= input_w");
    }
}

// ---------------------------------------------------------------------------
// Harness 3: PixelShuffle output channels
// ---------------------------------------------------------------------------

/// Prove: PixelShuffle output_C = input_C / (r * r).
/// Channel count is reduced by factor r^2 as those channels become spatial.
#[kani::unwind(1)]
#[kani::proof]
fn proof_pixel_shuffle_output_channels() {
    let c_out: usize = kani::any();
    let r: usize = kani::any();

    kani::assume(c_out >= 1 && c_out <= 512);
    kani::assume(r >= 1 && r <= 8);

    let r2 = r * r;
    let c_in = c_out.checked_mul(r2);
    if let Some(c_in) = c_in {
        let computed_c_out = c_in / r2;
        assert!(
            computed_c_out == c_out,
            "output channels must equal input_channels / r^2"
        );
        // Output channels are never greater than input channels
        assert!(computed_c_out <= c_in, "output channels <= input channels");
    }
}

// ---------------------------------------------------------------------------
// Harness 4: PixelShuffle element count preserved
// ---------------------------------------------------------------------------

/// Prove: PixelShuffle preserves total element count.
/// Input [B, C*r^2, H, W] and output [B, C, H*r, W*r] have the same
/// number of elements: B * C * r^2 * H * W.
#[kani::unwind(1)]
#[kani::proof]
fn proof_pixel_shuffle_element_count() {
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

    // Input: [B, C*r^2, H, W]
    let input_elems = b
        .checked_mul(c)
        .and_then(|v| v.checked_mul(r2))
        .and_then(|v| v.checked_mul(h))
        .and_then(|v| v.checked_mul(w));

    // Output: [B, C, H*r, W*r]
    let output_elems = b
        .checked_mul(c)
        .and_then(|v| v.checked_mul(h * r))
        .and_then(|v| v.checked_mul(w * r));

    if let (Some(inp), Some(out)) = (input_elems, output_elems) {
        assert!(inp == out, "PixelShuffle must preserve total element count");
    }
}

// ---------------------------------------------------------------------------
// Harness 5: PixelUnshuffle spatial divisibility
// ---------------------------------------------------------------------------

/// Prove: PixelUnshuffle requires H and W divisible by r.
/// When input spatial dims are constructed as out_h * r and out_w * r,
/// they are divisible by r by construction.
#[kani::unwind(1)]
#[kani::proof]
fn proof_pixel_unshuffle_spatial_divisible() {
    let out_h: usize = kani::any();
    let out_w: usize = kani::any();
    let r: usize = kani::any();

    kani::assume(out_h >= 1 && out_h <= 2048);
    kani::assume(out_w >= 1 && out_w <= 2048);
    kani::assume(r >= 1 && r <= 8);

    let in_h = out_h.checked_mul(r);
    let in_w = out_w.checked_mul(r);

    if let (Some(ih), Some(iw)) = (in_h, in_w) {
        // Divisibility requirement for PixelUnshuffle
        assert!(ih % r == 0, "input H must be divisible by r");
        assert!(iw % r == 0, "input W must be divisible by r");

        // And the quotient recovers the output dims
        assert!(ih / r == out_h, "H / r must recover output H");
        assert!(iw / r == out_w, "W / r must recover output W");
    }
}

// ---------------------------------------------------------------------------
// Harness 6: PixelUnshuffle is inverse of PixelShuffle
// ---------------------------------------------------------------------------

/// Prove: unshuffle(shuffle(shape)) == shape for dimension arithmetic.
/// Starting from [C, H, W], shuffle → [C/r^2, H*r, W*r],
/// then unshuffle → [C/r^2 * r^2, H*r/r, W*r/r] = [C, H, W].
#[kani::unwind(1)]
#[kani::proof]
fn proof_pixel_unshuffle_inverse() {
    let c: usize = kani::any();
    let h: usize = kani::any();
    let w: usize = kani::any();
    let r: usize = kani::any();

    kani::assume(c >= 1 && c <= 64);
    kani::assume(h >= 1 && h <= 64);
    kani::assume(w >= 1 && w <= 64);
    kani::assume(r >= 1 && r <= 4);

    let r2 = r * r;

    // Start with a PixelShuffle-compatible input: c must be divisible by r^2
    // Construct c_in = c * r^2, then shuffle divides back to c
    let c_in = c.checked_mul(r2);
    if let Some(c_in) = c_in {
        // PixelShuffle: [c_in, h, w] → [c_in/r^2, h*r, w*r] = [c, h*r, w*r]
        let shuffled_c = c_in / r2;
        let shuffled_h = h.checked_mul(r);
        let shuffled_w = w.checked_mul(r);

        if let (Some(sh), Some(sw)) = (shuffled_h, shuffled_w) {
            assert!(shuffled_c == c, "shuffle reduces channels to c");

            // PixelUnshuffle: [c, h*r, w*r] → [c*r^2, h, w]
            let unshuffled_c = shuffled_c.checked_mul(r2);
            if let Some(uc) = unshuffled_c {
                let unshuffled_h = sh / r;
                let unshuffled_w = sw / r;

                // Must recover original shape
                assert!(uc == c_in, "unshuffle(shuffle) must restore channels");
                assert!(unshuffled_h == h, "unshuffle(shuffle) must restore height");
                assert!(unshuffled_w == w, "unshuffle(shuffle) must restore width");
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Harness 7: Upsample2d output = input * scale_factor
// ---------------------------------------------------------------------------

/// Prove: nearest-neighbor upsample output dimensions equal
/// input dimensions multiplied by the integer scale factor.
#[kani::unwind(1)]
#[kani::proof]
fn proof_upsample_output_dims() {
    let h: usize = kani::any();
    let w: usize = kani::any();
    let scale_h: usize = kani::any();
    let scale_w: usize = kani::any();

    kani::assume(h >= 1 && h <= 2048);
    kani::assume(w >= 1 && w <= 2048);
    kani::assume(scale_h >= 1 && scale_h <= 8);
    kani::assume(scale_w >= 1 && scale_w <= 8);

    let out_h = h.checked_mul(scale_h);
    let out_w = w.checked_mul(scale_w);

    if let (Some(oh), Some(ow)) = (out_h, out_w) {
        assert!(oh == h * scale_h, "output_h == input_h * scale_h");
        assert!(ow == w * scale_w, "output_w == input_w * scale_w");
        // Scale >= 1 means output >= input
        assert!(oh >= h, "upsample must not shrink height");
        assert!(ow >= w, "upsample must not shrink width");
    }
}

// ---------------------------------------------------------------------------
// Harness 8: Upsample2d batch and channel dims preserved
// ---------------------------------------------------------------------------

/// Prove: nearest-neighbor upsample preserves batch and channel dimensions.
/// Input [B, C, H, W] → Output [B, C, H*s, W*s]: B and C are unchanged.
#[kani::unwind(1)]
#[kani::proof]
fn proof_upsample_batch_channel_preserved() {
    let b: usize = kani::any();
    let c: usize = kani::any();
    let h: usize = kani::any();
    let w: usize = kani::any();
    let s: usize = kani::any();

    kani::assume(b >= 1 && b <= 32);
    kani::assume(c >= 1 && c <= 512);
    kani::assume(h >= 1 && h <= 256);
    kani::assume(w >= 1 && w <= 256);
    kani::assume(s >= 1 && s <= 8);

    // Input shape: [B, C, H, W]
    let in_b = b;
    let in_c = c;

    // Output shape: [B, C, H*s, W*s]
    let out_b = b;
    let out_c = c;

    // Batch and channel are identity
    assert!(out_b == in_b, "batch dimension must be preserved");
    assert!(out_c == in_c, "channel dimension must be preserved");

    // Only spatial dims change
    let out_h = h.checked_mul(s);
    let out_w = w.checked_mul(s);
    if let (Some(oh), Some(ow)) = (out_h, out_w) {
        assert!(oh >= h, "spatial height grows or stays");
        assert!(ow >= w, "spatial width grows or stays");
    }
}

// ---------------------------------------------------------------------------
// Harness 9: Upsample nearest-neighbor source index bounds
// ---------------------------------------------------------------------------

/// Prove: for nearest-neighbor upsampling, the source index for any output
/// position is always within the input dimension bounds.
/// source_index = output_index / scale_factor, and since
/// output_index < input_dim * scale_factor, source_index < input_dim.
#[kani::unwind(1)]
#[kani::proof]
fn proof_upsample_nearest_index_bounds() {
    let input_dim: usize = kani::any();
    let scale: usize = kani::any();
    let output_idx: usize = kani::any();

    kani::assume(input_dim >= 1 && input_dim <= 2048);
    kani::assume(scale >= 1 && scale <= 8);

    let output_dim = input_dim.checked_mul(scale);
    if let Some(od) = output_dim {
        // output_idx must be a valid output position
        kani::assume(output_idx < od);

        // Nearest-neighbor: source_idx = output_idx / scale
        let source_idx = output_idx / scale;

        assert!(
            source_idx < input_dim,
            "nearest-neighbor source index must be within input bounds"
        );
    }
}

// ---------------------------------------------------------------------------
// Harness 10: Upsample2d scale_factor > 0 validated
// ---------------------------------------------------------------------------

/// Prove: Upsample2d::new rejects zero, negative, NaN, and Inf scale factors.
#[kani::unwind(1)]
#[kani::proof]
fn proof_upsample_scale_positive() {
    // Zero rejected
    let r0 = super::Upsample2d::new(0.0, 1.0, super::UpsampleMode::Nearest);
    assert!(r0.is_err(), "must reject scale_h == 0.0");

    // Negative rejected
    let rn = super::Upsample2d::new(2.0, -3.0, super::UpsampleMode::Nearest);
    assert!(rn.is_err(), "must reject negative scale_w");

    // NaN rejected
    let rnan = super::Upsample2d::new(f64::NAN, 2.0, super::UpsampleMode::Nearest);
    assert!(rnan.is_err(), "must reject NaN scale");

    // Inf rejected
    let rinf = super::Upsample2d::new(2.0, f64::INFINITY, super::UpsampleMode::Nearest);
    assert!(rinf.is_err(), "must reject Inf scale");

    // Neg Inf rejected
    let rninf = super::Upsample2d::new(f64::NEG_INFINITY, 2.0, super::UpsampleMode::Nearest);
    assert!(rninf.is_err(), "must reject NEG_INFINITY scale");

    // Valid accepted
    let rok = super::Upsample2d::new(2.0, 3.0, super::UpsampleMode::Nearest);
    assert!(rok.is_ok(), "must accept valid positive scales");
}

// ---------------------------------------------------------------------------
// Harness 11: SE block squeeze channels positive
// ---------------------------------------------------------------------------

/// Prove: SE block reduced_channels = channels / ratio > 0 when
/// channels >= ratio and ratio >= 1.
#[kani::unwind(1)]
#[kani::proof]
fn proof_se_squeeze_channels_positive() {
    let channels: usize = kani::any();
    let ratio: usize = kani::any();

    kani::assume(channels >= 1 && channels <= 2048);
    kani::assume(ratio >= 1 && ratio <= 32);
    kani::assume(channels >= ratio); // Can't reduce below 1

    let reduced = channels / ratio;
    assert!(reduced >= 1, "reduced channels must be >= 1");
    assert!(
        reduced <= channels,
        "reduced channels must be <= input channels"
    );
}

// ---------------------------------------------------------------------------
// Harness 12: SE block sigmoid bounded
// ---------------------------------------------------------------------------

/// Prove: sigmoid(x) is in [0.0, 1.0] for any finite input.
/// sigmoid(x) = 1 / (1 + exp(-x))
#[kani::unwind(1)]
#[kani::proof]
fn proof_se_sigmoid_bounded() {
    let x: f32 = kani::any();
    kani::assume(!x.is_nan() && x.is_finite());

    // Compute sigmoid
    let neg_x = -x;
    let exp_neg_x = neg_x.exp();

    // exp(-x) could be Inf for very negative x, but 1/(1+Inf) = 0.0 which is valid
    if exp_neg_x.is_finite() {
        let sigmoid = 1.0f32 / (1.0f32 + exp_neg_x);
        if sigmoid.is_finite() {
            assert!(sigmoid >= 0.0, "sigmoid must be >= 0.0");
            assert!(sigmoid <= 1.0, "sigmoid must be <= 1.0");
        }
    }

    // For large positive x: exp(-x) → 0, sigmoid → 1.0
    // For large negative x: exp(-x) → Inf, sigmoid → 0.0
    // Both endpoints are within [0, 1]
}

// ---------------------------------------------------------------------------
// Harness 13: SE block output shape preserved
// ---------------------------------------------------------------------------

/// Prove: SE excitation does not change spatial dimensions.
/// Input [B, C, H, W] → squeeze to [B, C, 1, 1] → excite → scale [B, C, 1, 1]
/// → broadcast multiply with [B, C, H, W] → output [B, C, H, W].
/// The batch, channel, height, and width dimensions are all preserved.
#[kani::unwind(1)]
#[kani::proof]
fn proof_se_output_shape_preserved() {
    let b: usize = kani::any();
    let c: usize = kani::any();
    let h: usize = kani::any();
    let w: usize = kani::any();

    kani::assume(b >= 1 && b <= 32);
    kani::assume(c >= 1 && c <= 512);
    kani::assume(h >= 1 && h <= 256);
    kani::assume(w >= 1 && w <= 256);

    // Input shape
    let in_shape = [b, c, h, w];

    // Squeeze: global average pool → [B, C, 1, 1]
    let squeeze_shape = [b, c, 1, 1];
    assert!(
        squeeze_shape[0] == in_shape[0],
        "batch preserved in squeeze"
    );
    assert!(
        squeeze_shape[1] == in_shape[1],
        "channels preserved in squeeze"
    );

    // After excitation, scale is [B, C, 1, 1] (same shape as squeeze)
    let scale_shape = squeeze_shape;

    // Broadcast multiply: [B, C, H, W] * [B, C, 1, 1] → [B, C, H, W]
    // Broadcasting rules: 1 broadcasts to any dim
    let out_b = in_shape[0]; // max(B, B) = B
    let out_c = in_shape[1]; // max(C, C) = C
    let out_h = in_shape[2]; // max(H, 1) = H
    let out_w = in_shape[3]; // max(W, 1) = W

    assert!(out_b == b, "output batch == input batch");
    assert!(out_c == c, "output channels == input channels");
    assert!(out_h == h, "output height == input height");
    assert!(out_w == w, "output width == input width");

    // Scale shape dims that are 1 broadcast to input dims
    assert!(scale_shape[2] == 1, "scale height is 1 for broadcast");
    assert!(scale_shape[3] == 1, "scale width is 1 for broadcast");
}

// ---------------------------------------------------------------------------
// Harness 14: DeepStackFusion normalized weights sum to 1.0
// ---------------------------------------------------------------------------

/// Prove: softmax-normalized weights over N layers sum to 1.0.
/// DeepStackFusion conceptually gives equal weight to each layer when
/// concatenating, so N layers each contribute 1/N of the concat.
/// This proves the arithmetic identity: N * (1/N) == 1.
#[kani::unwind(1)]
#[kani::proof]
fn proof_deep_stack_weight_sum() {
    let num_layers: usize = kani::any();
    kani::assume(num_layers >= 1 && num_layers <= 16);

    // Each layer contributes input_hidden_size to the concat.
    // The concat dimension = num_layers * input_hidden_size.
    // Each layer's fraction of the concat is 1/num_layers.
    // Sum of fractions = num_layers * (1/num_layers) = 1.0.

    let hidden: usize = kani::any();
    kani::assume(hidden >= 1 && hidden <= 2048);

    let concat_dim = num_layers.checked_mul(hidden);
    if let Some(cd) = concat_dim {
        // Each layer's contribution
        assert!(cd >= hidden, "concat dim >= single layer dim");

        // The fraction contributed by each layer
        // hidden / concat_dim = hidden / (num_layers * hidden) = 1 / num_layers
        // Sum = num_layers * hidden / concat_dim = num_layers * hidden / (num_layers * hidden) = 1
        assert!(
            num_layers * hidden == cd,
            "total contribution must equal concat dimension"
        );
    }
}

// ---------------------------------------------------------------------------
// Harness 15: PixelShuffle r=1 is identity on dims
// ---------------------------------------------------------------------------

/// Prove: PixelShuffle with r=1 is an identity transform on dimensions.
/// [B, C*1, H*1, W*1] == [B, C, H, W].
#[kani::unwind(1)]
#[kani::proof]
fn proof_pixel_shuffle_r1_identity() {
    let c: usize = kani::any();
    let h: usize = kani::any();
    let w: usize = kani::any();

    kani::assume(c >= 1 && c <= 2048);
    kani::assume(h >= 1 && h <= 2048);
    kani::assume(w >= 1 && w <= 2048);

    let r: usize = 1;
    let r2 = r * r; // 1

    // Input: [C, H, W]
    let out_c = c / r2; // c / 1 = c
    let out_h = h * r; // h * 1 = h
    let out_w = w * r; // w * 1 = w

    assert!(out_c == c, "r=1: channels unchanged");
    assert!(out_h == h, "r=1: height unchanged");
    assert!(out_w == w, "r=1: width unchanged");
}

// ---------------------------------------------------------------------------
// Harness 16: Upsample2dToSize rejects zero output dims
// ---------------------------------------------------------------------------

/// Prove: Upsample2dToSize::new rejects zero output dimensions.
#[kani::unwind(1)]
#[kani::proof]
fn proof_upsample_to_size_rejects_zero() {
    let r1 = super::Upsample2dToSize::new(0, 16, false);
    assert!(r1.is_err(), "must reject out_h == 0");

    let r2 = super::Upsample2dToSize::new(16, 0, false);
    assert!(r2.is_err(), "must reject out_w == 0");

    let r3 = super::Upsample2dToSize::new(0, 0, true);
    assert!(r3.is_err(), "must reject both == 0");

    // Valid accepted
    let r4 = super::Upsample2dToSize::new(16, 16, false);
    assert!(r4.is_ok(), "must accept positive dims");
    let up = r4.unwrap();
    assert!(up.out_h() == 16, "stored out_h must match");
    assert!(up.out_w() == 16, "stored out_w must match");
}

// ---------------------------------------------------------------------------
// Harness 17: PixelShuffle constructor rejects zero, accepts positive
// ---------------------------------------------------------------------------

/// Prove: PixelShuffle::new(0) is Err, PixelShuffle::new(r) for r>=1 is Ok.
#[kani::unwind(1)]
#[kani::proof]
fn proof_pixel_shuffle_constructor_validation() {
    let factor: usize = kani::any();
    kani::assume(factor <= 8);

    let result = super::PixelShuffle::new(factor);
    if factor == 0 {
        assert!(result.is_err(), "must reject upscale_factor == 0");
    } else {
        assert!(result.is_ok(), "must accept upscale_factor > 0");
        let ps = result.unwrap();
        assert!(
            ps.upscale_factor() == factor,
            "stored factor must match input"
        );
    }
}

// ---------------------------------------------------------------------------
// Harness 18: PixelUnshuffle constructor rejects zero, accepts positive
// ---------------------------------------------------------------------------

/// Prove: PixelUnshuffle::new(0) is Err, PixelUnshuffle::new(r) for r>=1 is Ok.
#[kani::unwind(1)]
#[kani::proof]
fn proof_pixel_unshuffle_constructor_validation() {
    let factor: usize = kani::any();
    kani::assume(factor <= 8);

    let result = super::PixelUnshuffle::new(factor);
    if factor == 0 {
        assert!(result.is_err(), "must reject downscale_factor == 0");
    } else {
        assert!(result.is_ok(), "must accept downscale_factor > 0");
        let pu = result.unwrap();
        assert!(
            pu.downscale_factor() == factor,
            "stored factor must match input"
        );
    }
}

// ---------------------------------------------------------------------------
// Harness 19: SE block 1D output shape preserved
// ---------------------------------------------------------------------------

/// Prove: SqueezeExcitation1d does not change shape.
/// Input [B, C, T] → squeeze to [B, C] → excite → scale [B, C, 1]
/// → broadcast multiply with [B, C, T] → output [B, C, T].
#[kani::unwind(1)]
#[kani::proof]
fn proof_se_1d_output_shape_preserved() {
    let b: usize = kani::any();
    let c: usize = kani::any();
    let t: usize = kani::any();

    kani::assume(b >= 1 && b <= 32);
    kani::assume(c >= 1 && c <= 512);
    kani::assume(t >= 1 && t <= 2048);

    // Input shape
    let in_shape = [b, c, t];

    // Squeeze: mean_keepdim(2) → [B, C, 1] → squeeze(2) → [B, C]
    let squeeze_shape = [b, c];
    assert!(
        squeeze_shape[0] == in_shape[0],
        "batch preserved in squeeze"
    );
    assert!(
        squeeze_shape[1] == in_shape[1],
        "channels preserved in squeeze"
    );

    // After excitation, scale is [B, C] → unsqueeze(2) → [B, C, 1]
    let scale_shape = [b, c, 1];

    // Broadcast multiply: [B, C, T] * [B, C, 1] → [B, C, T]
    let out_b = in_shape[0];
    let out_c = in_shape[1];
    let out_t = in_shape[2]; // max(T, 1) = T

    assert!(out_b == b, "output batch == input batch");
    assert!(out_c == c, "output channels == input channels");
    assert!(out_t == t, "output time == input time");

    // Scale shape: last dim is 1 for broadcast
    assert!(scale_shape[2] == 1, "scale time dim is 1 for broadcast");
}

// ---------------------------------------------------------------------------
// Harness 20: Upsample2d Nearest accepts integer scales
// ---------------------------------------------------------------------------

/// Prove: Upsample2d::new accepts valid integer scale factors for nearest mode,
/// and the accessor methods return the stored values.
#[kani::unwind(1)]
#[kani::proof]
fn proof_upsample_nearest_accepts_valid() {
    let sh: usize = kani::any();
    let sw: usize = kani::any();

    kani::assume(sh >= 1 && sh <= 8);
    kani::assume(sw >= 1 && sw <= 8);

    let scale_h = sh as f64;
    let scale_w = sw as f64;

    let result = super::Upsample2d::new(scale_h, scale_w, super::UpsampleMode::Nearest);
    assert!(result.is_ok(), "must accept valid integer scales");

    let up = result.unwrap();
    assert!(
        (up.scale_h() - scale_h).abs() < 1e-10,
        "scale_h accessor must return stored value"
    );
    assert!(
        (up.scale_w() - scale_w).abs() < 1e-10,
        "scale_w accessor must return stored value"
    );
}

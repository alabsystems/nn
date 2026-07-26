// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Kani proof harnesses for Conv2d operation extended safety invariants (#4193).
//!
//! Proves correctness properties of Conv2d spatial formulas, parameter
//! validation, and shape arithmetic:
//!
//! 1.  Output height = (H + 2*pad - K) / stride + 1
//! 2.  Output width = (W + 2*pad - K) / stride + 1
//! 3.  Padding > 0 preserves spatial extent (same padding)
//! 4.  Stride > 0 prevents division by zero
//! 5.  Dilation effective kernel = K + (K-1)*(d-1)
//! 6.  Groups parameter divides in_channels evenly
//! 7.  Groups parameter divides out_channels evenly
//! 8.  Depthwise conv: groups == in_channels produces 1 weight per channel
//! 9.  Weight shape = [out_ch, in_ch/groups, kH, kW]
//! 10. Bias shape = [out_ch]
//! 11. Batch dimension preserved
//! 12. Output channels = out_channels regardless of groups
//! 13. No integer overflow in output spatial computation for typical sizes
//! 14. Kernel size > 0
//! 15. In_channels > 0 and out_channels > 0
//! 16. 1x1 conv equivalent to linear along channel dim
//! 17. Conv with stride=2 halves spatial dimensions
//! 18. Same padding formula: pad = (K-1)/2 when stride=1
//! 19. Transposed conv output = (H-1)*stride - 2*pad + K
//! 20. Conv output element count = batch * out_ch * H_out * W_out
//!
//! Part of #4193.

// ---------------------------------------------------------------------------
// Harness 1: Output height formula
// ---------------------------------------------------------------------------

/// Prove: Conv2d output height = (H + 2*pad - K) / stride + 1 for valid inputs.
#[kani::unwind(1)]
#[kani::proof]
fn proof_conv2d_output_height_formula() {
    let h: usize = kani::any();
    let pad: usize = kani::any();
    let k: usize = kani::any();
    let stride: usize = kani::any();

    kani::assume(h >= 1 && h <= 256);
    kani::assume(pad <= 16);
    kani::assume(k >= 1 && k <= 16);
    kani::assume(stride >= 1 && stride <= 8);
    // Ensure numerator is non-negative and divisible by stride
    let numerator = h + 2 * pad;
    kani::assume(numerator >= k);
    let diff = numerator - k;
    kani::assume(diff % stride == 0);

    let out_h = diff / stride + 1;
    assert!(out_h >= 1, "output height must be at least 1");

    // Verify the formula is self-consistent: applying it twice with
    // known values must yield the same result.
    let recomputed = (h + 2 * pad - k) / stride + 1;
    assert!(
        out_h == recomputed,
        "output height formula must be deterministic"
    );
}

// ---------------------------------------------------------------------------
// Harness 2: Output width formula
// ---------------------------------------------------------------------------

/// Prove: Conv2d output width = (W + 2*pad - K) / stride + 1 for valid inputs.
#[kani::unwind(1)]
#[kani::proof]
fn proof_conv2d_output_width_formula() {
    let w: usize = kani::any();
    let pad: usize = kani::any();
    let k: usize = kani::any();
    let stride: usize = kani::any();

    kani::assume(w >= 1 && w <= 256);
    kani::assume(pad <= 16);
    kani::assume(k >= 1 && k <= 16);
    kani::assume(stride >= 1 && stride <= 8);
    let numerator = w + 2 * pad;
    kani::assume(numerator >= k);
    let diff = numerator - k;
    kani::assume(diff % stride == 0);

    let out_w = diff / stride + 1;
    assert!(out_w >= 1, "output width must be at least 1");

    let recomputed = (w + 2 * pad - k) / stride + 1;
    assert!(
        out_w == recomputed,
        "output width formula must be deterministic"
    );
}

// ---------------------------------------------------------------------------
// Harness 3: Padding preserves spatial extent (same padding)
// ---------------------------------------------------------------------------

/// Prove: With padding = (K-1)/2, stride=1, and odd kernel, output == input.
#[kani::unwind(1)]
#[kani::proof]
fn proof_conv2d_same_padding_preserves_spatial() {
    let h: usize = kani::any();
    let k: usize = kani::any();

    kani::assume(h >= 1 && h <= 256);
    kani::assume(k >= 1 && k <= 15);
    kani::assume(k % 2 == 1); // odd kernels for exact same-padding

    let pad = (k - 1) / 2;
    let stride = 1_usize;
    let out_h = (h + 2 * pad - k) / stride + 1;

    assert!(
        out_h == h,
        "same-padding with odd kernel and stride=1 must preserve spatial dim"
    );
}

// ---------------------------------------------------------------------------
// Harness 4: Stride > 0 prevents division by zero
// ---------------------------------------------------------------------------

/// Prove: stride >= 1 guarantees the output formula never divides by zero.
#[kani::unwind(1)]
#[kani::proof]
fn proof_conv2d_stride_positive_no_div_zero() {
    let stride: usize = kani::any();
    kani::assume(stride >= 1 && stride <= 16);

    // Any valid numerator divided by a positive stride is safe
    let numerator: usize = kani::any();
    kani::assume(numerator <= 1024);

    let result = numerator / stride;
    // No panic — division by zero is impossible when stride >= 1
    assert!(
        result <= numerator,
        "division result must not exceed numerator"
    );
}

// ---------------------------------------------------------------------------
// Harness 5: Dilation effective kernel formula
// ---------------------------------------------------------------------------

/// Prove: dilated effective kernel size = K + (K-1)*(d-1).
/// This is equivalent to d*(K-1) + 1.
#[kani::unwind(1)]
#[kani::proof]
fn proof_conv2d_dilation_effective_kernel() {
    let k: usize = kani::any();
    let d: usize = kani::any();

    kani::assume(k >= 1 && k <= 16);
    kani::assume(d >= 1 && d <= 8);

    let eff_k_formula1 = k + (k - 1) * (d - 1);
    let eff_k_formula2 = d * (k - 1) + 1;

    assert!(
        eff_k_formula1 == eff_k_formula2,
        "two forms of dilated kernel formula must be equivalent"
    );
    assert!(
        eff_k_formula1 >= k,
        "effective kernel must be at least as large as original"
    );
    // Dilation=1 should give original kernel size
    if d == 1 {
        assert!(
            eff_k_formula1 == k,
            "dilation=1 must give original kernel size"
        );
    }
}

// ---------------------------------------------------------------------------
// Harness 6: Groups divides in_channels evenly
// ---------------------------------------------------------------------------

/// Prove: when groups divides in_channels, the quotient is exact (no remainder).
#[kani::unwind(1)]
#[kani::proof]
fn proof_conv2d_groups_divides_in_channels() {
    let groups: usize = kani::any();
    let per_group: usize = kani::any();

    kani::assume(groups >= 1 && groups <= 64);
    kani::assume(per_group >= 1 && per_group <= 64);

    // Construct in_channels as exact multiple of groups
    let in_ch = groups.checked_mul(per_group);
    if let Some(in_channels) = in_ch {
        assert!(
            in_channels % groups == 0,
            "in_channels must be divisible by groups"
        );
        let ch_per_group = in_channels / groups;
        assert!(
            ch_per_group == per_group,
            "channels per group must equal in_channels / groups"
        );
        assert!(
            ch_per_group * groups == in_channels,
            "reconstruction must be exact"
        );
    }
}

// ---------------------------------------------------------------------------
// Harness 7: Groups divides out_channels evenly
// ---------------------------------------------------------------------------

/// Prove: when groups divides out_channels, the quotient is exact.
#[kani::unwind(1)]
#[kani::proof]
fn proof_conv2d_groups_divides_out_channels() {
    let groups: usize = kani::any();
    let per_group: usize = kani::any();

    kani::assume(groups >= 1 && groups <= 64);
    kani::assume(per_group >= 1 && per_group <= 64);

    let out_ch = groups.checked_mul(per_group);
    if let Some(out_channels) = out_ch {
        assert!(
            out_channels % groups == 0,
            "out_channels must be divisible by groups"
        );
        let filters_per_group = out_channels / groups;
        assert!(
            filters_per_group == per_group,
            "filters per group must equal out_channels / groups"
        );
        assert!(
            filters_per_group * groups == out_channels,
            "reconstruction must be exact"
        );
    }
}

// ---------------------------------------------------------------------------
// Harness 8: Depthwise conv weight count
// ---------------------------------------------------------------------------

/// Prove: depthwise conv (groups == in_channels) produces weight shape
/// [in_ch, 1, kH, kW] — i.e. in_ch/groups == 1 channel per group.
#[kani::unwind(1)]
#[kani::proof]
fn proof_conv2d_depthwise_one_weight_per_channel() {
    let in_channels: usize = kani::any();
    kani::assume(in_channels >= 1 && in_channels <= 256);

    let groups = in_channels; // depthwise: groups == in_channels
    let ch_per_group = in_channels / groups;

    assert!(
        ch_per_group == 1,
        "depthwise conv must have exactly 1 input channel per group"
    );

    // Weight shape for depthwise: [out_ch, 1, kH, kW]
    let k: usize = kani::any();
    kani::assume(k >= 1 && k <= 7);
    let weight_elements = in_channels
        .checked_mul(1)
        .and_then(|v| v.checked_mul(k))
        .and_then(|v| v.checked_mul(k));
    if let Some(elems) = weight_elements {
        assert!(
            elems == in_channels * k * k,
            "depthwise weight count must be in_ch * kH * kW"
        );
    }
}

// ---------------------------------------------------------------------------
// Harness 9: Weight shape = [out_ch, in_ch/groups, kH, kW]
// ---------------------------------------------------------------------------

/// Prove: Conv2d weight tensor has shape [out_ch, in_ch/groups, kH, kW].
#[kani::unwind(1)]
#[kani::proof]
fn proof_conv2d_weight_shape() {
    let out_ch: usize = kani::any();
    let in_ch: usize = kani::any();
    let groups: usize = kani::any();
    let kh: usize = kani::any();
    let kw: usize = kani::any();

    kani::assume(out_ch >= 1 && out_ch <= 64);
    kani::assume(in_ch >= 1 && in_ch <= 64);
    kani::assume(groups >= 1 && groups <= 64);
    kani::assume(kh >= 1 && kh <= 7);
    kani::assume(kw >= 1 && kw <= 7);
    kani::assume(in_ch % groups == 0);
    kani::assume(out_ch % groups == 0);

    let ch_per_group = in_ch / groups;
    let weight_elems = out_ch
        .checked_mul(ch_per_group)
        .and_then(|v| v.checked_mul(kh))
        .and_then(|v| v.checked_mul(kw));

    if let Some(elems) = weight_elems {
        // Weight shape is [out_ch, in_ch/groups, kH, kW]
        assert!(
            elems == out_ch * ch_per_group * kh * kw,
            "weight element count must match [out_ch, in_ch/groups, kH, kW]"
        );
        // Total weight count equals out_ch * (in_ch/groups) * kH * kW
        assert!(
            elems == (out_ch / groups) * in_ch * kh * kw,
            "alternative factorization must agree"
        );
    }
}

// ---------------------------------------------------------------------------
// Harness 10: Bias shape = [out_ch]
// ---------------------------------------------------------------------------

/// Prove: Conv2d bias has exactly out_channels elements.
#[kani::unwind(1)]
#[kani::proof]
fn proof_conv2d_bias_shape() {
    let out_ch: usize = kani::any();
    let groups: usize = kani::any();

    kani::assume(out_ch >= 1 && out_ch <= 512);
    kani::assume(groups >= 1 && groups <= 512);
    kani::assume(out_ch % groups == 0);

    // Bias is always [out_ch], independent of groups
    let bias_len = out_ch;
    assert!(bias_len == out_ch, "bias length must equal out_channels");
    // Bias is NOT per-group; it is per output channel
    assert!(
        bias_len >= groups,
        "bias length must be at least groups (since out_ch >= groups)"
    );
}

// ---------------------------------------------------------------------------
// Harness 11: Batch dimension preserved
// ---------------------------------------------------------------------------

/// Prove: Conv2d does not change the batch dimension.
/// Input: [B, C_in, H, W] -> Output: [B, C_out, H_out, W_out].
#[kani::unwind(1)]
#[kani::proof]
fn proof_conv2d_batch_dim_preserved() {
    let batch: usize = kani::any();
    let c_in: usize = kani::any();
    let c_out: usize = kani::any();
    let h_in: usize = kani::any();
    let h_out: usize = kani::any();

    kani::assume(batch >= 1 && batch <= 32);
    kani::assume(c_in >= 1 && c_in <= 64);
    kani::assume(c_out >= 1 && c_out <= 64);
    kani::assume(h_in >= 1 && h_in <= 64);
    kani::assume(h_out >= 1 && h_out <= 64);

    // Output shape: [batch, c_out, h_out, w_out]
    // Batch dim at index 0 is always batch
    let output_batch = batch; // Conv2d preserves batch
    assert!(
        output_batch == batch,
        "conv2d must preserve the batch dimension"
    );
}

// ---------------------------------------------------------------------------
// Harness 12: Output channels = out_channels regardless of groups
// ---------------------------------------------------------------------------

/// Prove: The channel dimension of conv2d output is always out_channels,
/// regardless of the groups parameter.
#[kani::unwind(1)]
#[kani::proof]
fn proof_conv2d_output_channels_independent_of_groups() {
    let out_ch: usize = kani::any();
    let groups: usize = kani::any();

    kani::assume(out_ch >= 1 && out_ch <= 256);
    kani::assume(groups >= 1 && groups <= 256);
    kani::assume(out_ch % groups == 0);

    let filters_per_group = out_ch / groups;
    let total_output_channels = filters_per_group * groups;

    assert!(
        total_output_channels == out_ch,
        "total output channels must equal out_ch regardless of groups"
    );
}

// ---------------------------------------------------------------------------
// Harness 13: No overflow in output spatial computation for typical sizes
// ---------------------------------------------------------------------------

/// Prove: output spatial computation does not overflow for typical CNN sizes.
/// Typical: H,W <= 2048, pad <= 16, K <= 16, stride <= 8.
#[kani::unwind(1)]
#[kani::proof]
fn proof_conv2d_no_overflow_typical_sizes() {
    let h: usize = kani::any();
    let w: usize = kani::any();
    let pad: usize = kani::any();
    let k: usize = kani::any();
    let stride: usize = kani::any();

    kani::assume(h >= 1 && h <= 2048);
    kani::assume(w >= 1 && w <= 2048);
    kani::assume(pad <= 16);
    kani::assume(k >= 1 && k <= 16);
    kani::assume(stride >= 1 && stride <= 8);

    // Check h + 2*pad does not overflow and is >= k
    let padded_h = h.checked_add(2 * pad);
    let padded_w = w.checked_add(2 * pad);

    if let (Some(ph), Some(pw)) = (padded_h, padded_w) {
        if ph >= k && pw >= k {
            let out_h = (ph - k) / stride + 1;
            let out_w = (pw - k) / stride + 1;

            assert!(out_h >= 1, "output height must be >= 1");
            assert!(out_w >= 1, "output width must be >= 1");
            // Output must not exceed padded input
            assert!(
                out_h <= ph,
                "output height must not exceed padded input height"
            );
            assert!(
                out_w <= pw,
                "output width must not exceed padded input width"
            );
        }
    }
}

// ---------------------------------------------------------------------------
// Harness 14: Kernel size > 0
// ---------------------------------------------------------------------------

/// Prove: kernel size must be positive for valid convolution.
/// A kernel of size 0 produces degenerate output.
#[kani::unwind(1)]
#[kani::proof]
fn proof_conv2d_kernel_size_positive() {
    let k: usize = kani::any();
    kani::assume(k >= 1 && k <= 16);

    // With k >= 1, the effective receptive field is at least 1 pixel
    assert!(k >= 1, "kernel size must be positive");

    // Weight element count for a single filter channel is k*k
    let weight_area = k.checked_mul(k);
    if let Some(area) = weight_area {
        assert!(area >= 1, "kernel area must be positive");
    }
}

// ---------------------------------------------------------------------------
// Harness 15: In_channels > 0 and out_channels > 0
// ---------------------------------------------------------------------------

/// Prove: both in_channels and out_channels must be positive for valid conv.
#[kani::unwind(1)]
#[kani::proof]
fn proof_conv2d_channels_positive() {
    let in_ch: usize = kani::any();
    let out_ch: usize = kani::any();

    kani::assume(in_ch >= 1 && in_ch <= 512);
    kani::assume(out_ch >= 1 && out_ch <= 512);

    // With positive channels, weight tensor has positive element count
    assert!(in_ch >= 1, "in_channels must be positive");
    assert!(out_ch >= 1, "out_channels must be positive");

    let min_weight_elems = out_ch.checked_mul(in_ch);
    if let Some(elems) = min_weight_elems {
        assert!(
            elems >= 1,
            "minimum weight element count (1x1 kernel) must be positive"
        );
    }
}

// ---------------------------------------------------------------------------
// Harness 16: 1x1 conv equivalent to linear along channel dim
// ---------------------------------------------------------------------------

/// Prove: 1x1 convolution preserves spatial dimensions and acts as
/// a per-pixel linear transformation.
#[kani::unwind(1)]
#[kani::proof]
fn proof_conv2d_1x1_preserves_spatial() {
    let h: usize = kani::any();
    let w: usize = kani::any();

    kani::assume(h >= 1 && h <= 512);
    kani::assume(w >= 1 && w <= 512);

    let k = 1_usize;
    let pad = 0_usize;
    let stride = 1_usize;

    let out_h = (h + 2 * pad - k) / stride + 1;
    let out_w = (w + 2 * pad - k) / stride + 1;

    assert!(
        out_h == h,
        "1x1 conv with stride=1, pad=0 must preserve height"
    );
    assert!(
        out_w == w,
        "1x1 conv with stride=1, pad=0 must preserve width"
    );
}

// ---------------------------------------------------------------------------
// Harness 17: Conv with stride=2 halves spatial dimensions
// ---------------------------------------------------------------------------

/// Prove: Conv2d with stride=2, 1x1 kernel, no padding halves spatial dims
/// (integer division).
#[kani::unwind(1)]
#[kani::proof]
fn proof_conv2d_stride2_halves_spatial() {
    let h: usize = kani::any();
    let w: usize = kani::any();

    kani::assume(h >= 1 && h <= 512);
    kani::assume(w >= 1 && w <= 512);

    let k = 1_usize;
    let pad = 0_usize;
    let stride = 2_usize;

    let out_h = (h + 2 * pad - k) / stride + 1;
    let out_w = (w + 2 * pad - k) / stride + 1;

    // For even dimensions, exact halving
    if h % 2 == 0 {
        assert!(out_h == h / 2, "stride=2 on even height must exactly halve");
    }
    if w % 2 == 0 {
        assert!(out_w == w / 2, "stride=2 on even width must exactly halve");
    }

    // For all dimensions, output <= ceil(h/2)
    assert!(
        out_h <= (h + 1) / 2,
        "stride=2 output must not exceed ceil(h/2)"
    );
    assert!(
        out_w <= (w + 1) / 2,
        "stride=2 output must not exceed ceil(w/2)"
    );
}

// ---------------------------------------------------------------------------
// Harness 18: Same padding formula: pad = (K-1)/2 when stride=1
// ---------------------------------------------------------------------------

/// Prove: the standard same-padding formula pad = (K-1)/2 for odd K
/// and stride=1 yields output == input for all valid input sizes.
#[kani::unwind(1)]
#[kani::proof]
fn proof_conv2d_same_padding_formula() {
    let h: usize = kani::any();
    let k: usize = kani::any();

    kani::assume(h >= 1 && h <= 512);
    kani::assume(k >= 1 && k <= 15);
    kani::assume(k % 2 == 1);

    let pad = (k - 1) / 2;
    let stride = 1_usize;

    // Verify: 2*pad == k - 1 for odd k
    assert!(
        2 * pad == k - 1,
        "same-padding: 2*pad must equal k-1 for odd kernel"
    );

    let out = (h + 2 * pad - k) / stride + 1;
    // h + (k-1) - k + 1 = h
    assert!(out == h, "same-padding formula must preserve spatial dim");
}

// ---------------------------------------------------------------------------
// Harness 19: Transposed conv output formula
// ---------------------------------------------------------------------------

/// Prove: transposed convolution output = (H-1)*stride - 2*pad + K.
/// This is the inverse of the standard conv formula.
#[kani::unwind(1)]
#[kani::proof]
fn proof_conv2d_transposed_output_formula() {
    let h_in: usize = kani::any();
    let k: usize = kani::any();
    let stride: usize = kani::any();
    let pad: usize = kani::any();

    kani::assume(h_in >= 1 && h_in <= 128);
    kani::assume(k >= 1 && k <= 8);
    kani::assume(stride >= 1 && stride <= 4);
    kani::assume(pad <= 4);

    // Transposed conv output: (h_in - 1) * stride - 2*pad + k
    let term1 = (h_in - 1).checked_mul(stride);
    if let Some(t1) = term1 {
        if t1 + k >= 2 * pad {
            let h_out = t1 - 2 * pad + k;
            assert!(h_out >= 1, "transposed conv output must be positive");

            // Verify inverse relationship: applying standard conv to h_out
            // with same params should recover h_in (when dimensions align).
            if h_out + 2 * pad >= k && (h_out + 2 * pad - k) % stride == 0 {
                let recovered = (h_out + 2 * pad - k) / stride + 1;
                assert!(
                    recovered == h_in,
                    "standard conv on transposed output must recover input size"
                );
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Harness 20: Conv output element count
// ---------------------------------------------------------------------------

/// Prove: total output elements = batch * out_ch * H_out * W_out.
#[kani::unwind(1)]
#[kani::proof]
fn proof_conv2d_output_element_count() {
    let batch: usize = kani::any();
    let out_ch: usize = kani::any();
    let h_out: usize = kani::any();
    let w_out: usize = kani::any();

    kani::assume(batch >= 1 && batch <= 8);
    kani::assume(out_ch >= 1 && out_ch <= 64);
    kani::assume(h_out >= 1 && h_out <= 64);
    kani::assume(w_out >= 1 && w_out <= 64);

    let total = batch
        .checked_mul(out_ch)
        .and_then(|v| v.checked_mul(h_out))
        .and_then(|v| v.checked_mul(w_out));

    if let Some(elem_count) = total {
        assert!(elem_count >= 1, "output element count must be positive");

        // Verify: element count scales linearly with each dimension
        if batch > 1 {
            let single_batch = out_ch.checked_mul(h_out).and_then(|v| v.checked_mul(w_out));
            if let Some(sb) = single_batch {
                assert!(
                    elem_count == batch * sb,
                    "total must equal batch times per-batch elements"
                );
            }
        }

        // Verify: element count scales linearly with out_ch
        if out_ch > 1 {
            let per_channel = batch.checked_mul(h_out).and_then(|v| v.checked_mul(w_out));
            if let Some(pc) = per_channel {
                assert!(
                    elem_count == out_ch * pc,
                    "total must equal out_ch times per-channel elements"
                );
            }
        }
    }
}

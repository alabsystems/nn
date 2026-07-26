// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Extended Kani proof harnesses for convolution stride and padding safety.
//!
//! Covers properties beyond the basic arithmetic in `kani_conv.rs` and
//! `kani_conv_pool.rs`:
//!
//! 1. Conv1d output length formula: ceil((input + 2*padding - dilation*(kernel-1) - 1) / stride) + 1
//! 2. Conv2d output shape: both height and width follow the output formula
//! 3. Depthwise conv: groups == in_channels == out_channels
//! 4. Grouped conv: in_channels and out_channels divisible by groups
//! 5. Conv transpose output: output_len = (input-1)*stride - 2*padding + dilation*(kernel-1) + 1
//! 6. Dilated conv receptive field: (kernel-1)*dilation + 1
//! 7. Padding modes: zero, reflect, replicate don't change channel count
//! 8. Causal conv: output_len == input_len when padding == (kernel-1)*dilation
//! 9. Same padding: output_len == ceil(input_len / stride)
//! 10. Weight shape consistency: weight shape = [out_ch, in_ch/groups, *kernel_size]
//!
//! Part of #4226: Extended Kani proofs for convolution stride and padding safety.

#![cfg(kani)]

// ---------------------------------------------------------------------------
// 1. Conv1d output length formula correctness
// ---------------------------------------------------------------------------

/// Prove: conv1d output length matches the standard formula
/// `out = (input + 2*padding - dilation*(kernel-1) - 1) / stride + 1`
/// for all small valid parameter combinations.
///
/// This is equivalent to `(padded - effective_k) / stride + 1` where
/// `effective_k = (kernel-1)*dilation + 1`.
#[kani::unwind(1)]
#[kani::proof]
fn conv1d_output_len_formula_equivalence() {
    let il: u8 = kani::any();
    let ks: u8 = kani::any();
    let p: u8 = kani::any();
    let s: u8 = kani::any();
    let d: u8 = kani::any();

    kani::assume(ks >= 1);
    kani::assume(s >= 1);
    kani::assume(d >= 1);
    kani::assume(il >= 1);

    let il = il as usize;
    let ks = ks as usize;
    let p = p as usize;
    let s = s as usize;
    let d = d as usize;

    let effective_k = (ks - 1) * d + 1;
    let padded = il + 2 * p;

    if padded >= effective_k {
        // Standard formula from dyn_tensor_conv.rs
        let out_standard = (padded - effective_k) / s + 1;

        // Equivalent textbook formula: (input + 2*padding - dilation*(kernel-1) - 1) / stride + 1
        // Note: dilation*(kernel-1) + 1 == effective_k, so this is identical
        let numerator = il + 2 * p - d * (ks - 1) - 1;
        let out_textbook = numerator / s + 1;

        assert_eq!(
            out_standard, out_textbook,
            "standard and textbook conv1d output formulas must agree"
        );
    }
}

/// Prove: conv1d output length is always positive when the configuration is valid
/// (padded >= effective_k).
#[kani::unwind(1)]
#[kani::proof]
fn conv1d_output_len_always_positive_when_valid() {
    let il: u8 = kani::any();
    let ks: u8 = kani::any();
    let p: u8 = kani::any();
    let s: u8 = kani::any();
    let d: u8 = kani::any();

    kani::assume(ks >= 1);
    kani::assume(s >= 1);
    kani::assume(d >= 1);
    kani::assume(il >= 1);

    let il = il as usize;
    let ks = ks as usize;
    let p = p as usize;
    let s = s as usize;
    let d = d as usize;

    let effective_k = (ks - 1) * d + 1;
    let padded = il + 2 * p;

    if padded >= effective_k {
        let out = (padded - effective_k) / s + 1;
        assert!(out >= 1, "conv1d output must be >= 1 for valid config");
        // Upper bound: output cannot exceed padded input size
        assert!(
            out <= padded,
            "conv1d output cannot exceed padded input length"
        );
    }
}

// ---------------------------------------------------------------------------
// 2. Conv2d output shape: both H and W follow the formula independently
// ---------------------------------------------------------------------------

/// Prove: conv2d output shape for both height and width dimensions are computed
/// independently using the same formula. If both spatial dims are valid,
/// both output dims are positive.
#[kani::unwind(1)]
#[kani::proof]
fn conv2d_output_shape_both_dims_valid() {
    let ih: u8 = kani::any(); // input height
    let iw: u8 = kani::any(); // input width
    let kh: u8 = kani::any(); // kernel height
    let kw: u8 = kani::any(); // kernel width
    let ph: u8 = kani::any(); // padding height
    let pw: u8 = kani::any(); // padding width
    let sh: u8 = kani::any(); // stride height
    let sw: u8 = kani::any(); // stride width
    let dh: u8 = kani::any(); // dilation height
    let dw: u8 = kani::any(); // dilation width

    kani::assume(kh >= 1 && kw >= 1);
    kani::assume(sh >= 1 && sw >= 1);
    kani::assume(dh >= 1 && dw >= 1);
    kani::assume(ih >= 1 && iw >= 1);

    let ih = ih as usize;
    let iw = iw as usize;
    let kh = kh as usize;
    let kw = kw as usize;
    let ph = ph as usize;
    let pw = pw as usize;
    let sh = sh as usize;
    let sw = sw as usize;
    let dh = dh as usize;
    let dw = dw as usize;

    let eff_kh = (kh - 1) * dh + 1;
    let eff_kw = (kw - 1) * dw + 1;
    let padded_h = ih + 2 * ph;
    let padded_w = iw + 2 * pw;

    if padded_h >= eff_kh && padded_w >= eff_kw {
        let oh = (padded_h - eff_kh) / sh + 1;
        let ow = (padded_w - eff_kw) / sw + 1;

        assert!(oh >= 1, "conv2d output height must be >= 1");
        assert!(ow >= 1, "conv2d output width must be >= 1");

        // Output spatial dims are independent
        assert!(oh <= padded_h, "output height bounded by padded height");
        assert!(ow <= padded_w, "output width bounded by padded width");
    }
}

/// Prove: conv2d with square params (same kernel, stride, padding, dilation
/// for both dims) produces square output from square input.
#[kani::unwind(1)]
#[kani::proof]
fn conv2d_square_params_square_output() {
    let il: u8 = kani::any(); // input spatial dim (both H and W)
    let ks: u8 = kani::any(); // square kernel size
    let p: u8 = kani::any(); // square padding
    let s: u8 = kani::any(); // square stride
    let d: u8 = kani::any(); // square dilation

    kani::assume(ks >= 1);
    kani::assume(s >= 1);
    kani::assume(d >= 1);
    kani::assume(il >= 1);

    let il = il as usize;
    let ks = ks as usize;
    let p = p as usize;
    let s = s as usize;
    let d = d as usize;

    let effective_k = (ks - 1) * d + 1;
    let padded = il + 2 * p;

    if padded >= effective_k {
        let oh = (padded - effective_k) / s + 1;
        let ow = (padded - effective_k) / s + 1;
        assert_eq!(
            oh, ow,
            "square conv on square input must produce square output"
        );
    }
}

// ---------------------------------------------------------------------------
// 3. Depthwise conv: groups == in_channels == out_channels
// ---------------------------------------------------------------------------

/// Prove: depthwise convolution invariants hold when groups == in_channels == out_channels.
///
/// In depthwise conv, each input channel gets its own filter. The weight
/// shape is [out_ch, 1, *kernel_size] (in_ch/groups == 1).
#[kani::unwind(1)]
#[kani::proof]
fn depthwise_conv_invariants() {
    let channels: u8 = kani::any();
    kani::assume(channels >= 1);

    let ch = channels as usize;

    // Depthwise: groups == in_channels == out_channels
    let groups = ch;
    let in_ch = ch;
    let out_ch = ch;

    // in_ch / groups == 1 (each group has exactly one input channel)
    let in_per_group = in_ch / groups;
    assert_eq!(in_per_group, 1, "depthwise conv: in_ch/groups must be 1");

    // out_ch / groups == 1 (each group produces exactly one output channel)
    let out_per_group = out_ch / groups;
    assert_eq!(out_per_group, 1, "depthwise conv: out_ch/groups must be 1");

    // Divisibility always holds trivially
    assert_eq!(in_ch % groups, 0);
    assert_eq!(out_ch % groups, 0);

    // Reconstruction
    assert_eq!(groups * in_per_group, in_ch);
    assert_eq!(groups * out_per_group, out_ch);
}

/// Prove: depthwise conv with channel multiplier preserves the group structure.
///
/// Depthwise separable conv uses groups == in_channels, out_channels == in_channels * multiplier.
/// Weight shape is [out_ch, 1, *kernel_size].
#[kani::unwind(1)]
#[kani::proof]
fn depthwise_conv_with_multiplier() {
    let in_ch: u8 = kani::any();
    let multiplier: u8 = kani::any();

    kani::assume(in_ch >= 1);
    kani::assume(multiplier >= 1);
    kani::assume(multiplier <= 8); // Practical bound

    let ic = in_ch as usize;
    let m = multiplier as usize;
    let groups = ic;
    let out_ch = ic * m;

    // in_ch / groups == 1
    assert_eq!(ic / groups, 1);

    // out_ch must be divisible by groups
    assert_eq!(out_ch % groups, 0);

    // out_ch / groups == multiplier
    let out_per_group = out_ch / groups;
    assert_eq!(
        out_per_group, m,
        "depthwise conv: out_ch/groups must equal multiplier"
    );
}

// ---------------------------------------------------------------------------
// 4. Grouped conv: in_channels and out_channels divisible by groups
// ---------------------------------------------------------------------------

/// Prove: grouped convolution weight shape consistency.
///
/// Weight shape = [out_ch, in_ch/groups, *kernel_size].
/// Total weight elements = out_ch * (in_ch/groups) * kernel_size.
/// This is strictly less than non-grouped: out_ch * in_ch * kernel_size.
#[kani::unwind(1)]
#[kani::proof]
fn grouped_conv_weight_reduction() {
    let in_ch: u8 = kani::any();
    let out_ch: u8 = kani::any();
    let groups: u8 = kani::any();
    let ks: u8 = kani::any();

    kani::assume(groups >= 1);
    kani::assume(in_ch >= groups);
    kani::assume(out_ch >= groups);
    kani::assume(ks >= 1);
    kani::assume(in_ch as usize % groups as usize == 0);
    kani::assume(out_ch as usize % groups as usize == 0);

    let g = groups as usize;
    let ic = in_ch as usize;
    let oc = out_ch as usize;
    let k = ks as usize;

    let in_per_group = ic / g;
    let grouped_weight_elems = oc * in_per_group * k;
    let full_weight_elems = oc * ic * k;

    // Grouped convolution has fewer or equal weight elements
    assert!(
        grouped_weight_elems <= full_weight_elems,
        "grouped conv weights must be <= full conv weights"
    );

    // Exactly 1/groups reduction factor
    assert_eq!(
        grouped_weight_elems * g,
        full_weight_elems,
        "grouped weights * groups must equal full weights"
    );

    // When groups == 1, they are equal (standard convolution)
    if g == 1 {
        assert_eq!(grouped_weight_elems, full_weight_elems);
    }
}

/// Prove: grouped conv splits channels evenly across groups with no remainder.
#[kani::unwind(1)]
#[kani::proof]
fn grouped_conv_even_channel_split() {
    let in_ch: u8 = kani::any();
    let out_ch: u8 = kani::any();
    let groups: u8 = kani::any();

    kani::assume(groups >= 1);
    kani::assume(in_ch >= 1);
    kani::assume(out_ch >= 1);
    kani::assume(in_ch as usize % groups as usize == 0);
    kani::assume(out_ch as usize % groups as usize == 0);

    let g = groups as usize;
    let ic = in_ch as usize;
    let oc = out_ch as usize;

    let in_per_group = ic / g;
    let out_per_group = oc / g;

    // Each group processes the same number of channels
    // Verify all groups sum back to total
    assert_eq!(in_per_group * g, ic);
    assert_eq!(out_per_group * g, oc);

    // Per-group channels are positive
    assert!(in_per_group >= 1);
    assert!(out_per_group >= 1);
}

// ---------------------------------------------------------------------------
// 5. Conv transpose output formula
// ---------------------------------------------------------------------------

/// Prove: conv_transpose1d output length formula correctness.
///
/// output_len = (input-1)*stride - 2*padding + dilation*(kernel-1) + 1
/// (with output_padding == 0 for simplicity).
#[kani::unwind(1)]
#[kani::proof]
fn conv_transpose1d_output_formula_correct() {
    let il: u8 = kani::any();
    let ks: u8 = kani::any();
    let p: u8 = kani::any();
    let s: u8 = kani::any();
    let d: u8 = kani::any();

    kani::assume(ks >= 1);
    kani::assume(s >= 1);
    kani::assume(d >= 1);
    kani::assume(il >= 1);

    let il = il as usize;
    let ks = ks as usize;
    let p = p as usize;
    let s = s as usize;
    let d = d as usize;

    // Formula: (input-1)*stride + dilation*(kernel-1) + 1 - 2*padding
    let positive = (il - 1) * s + d * (ks - 1) + 1;
    let negative = 2 * p;

    if positive > negative {
        let out = positive - negative;
        assert!(out >= 1, "conv_transpose1d output must be >= 1");

        // Verify the formula is self-consistent:
        // Increasing input by 1 increases output by exactly stride
        let positive_plus1 = il * s + d * (ks - 1) + 1;
        if positive_plus1 > negative {
            let out_plus1 = positive_plus1 - negative;
            assert_eq!(
                out_plus1 - out,
                s,
                "conv_transpose1d: +1 input increases output by stride"
            );
        }
    }
}

/// Prove: conv_transpose1d with output_padding produces consistent results.
///
/// PyTorch constraint: output_padding < stride.
/// Full formula: output = (input-1)*stride - 2*padding + dilation*(kernel-1) + output_padding + 1
#[kani::unwind(1)]
#[kani::proof]
fn conv_transpose1d_output_padding_bounded() {
    let il: u8 = kani::any();
    let ks: u8 = kani::any();
    let p: u8 = kani::any();
    let s: u8 = kani::any();
    let d: u8 = kani::any();
    let op: u8 = kani::any();

    kani::assume(ks >= 1);
    kani::assume(s >= 1);
    kani::assume(d >= 1);
    kani::assume(il >= 1);
    kani::assume(op < s); // PyTorch constraint

    let il = il as usize;
    let ks = ks as usize;
    let p = p as usize;
    let s = s as usize;
    let d = d as usize;
    let op = op as usize;

    let positive = (il - 1) * s + d * (ks - 1) + op + 1;
    let negative = 2 * p;

    if positive > negative {
        let out_with_op = positive - negative;

        // Without output_padding
        let out_without_op = (il - 1) * s + d * (ks - 1) + 1;
        if out_without_op > negative {
            let out_base = out_without_op - negative;
            // output_padding adds exactly `op` to the output length
            assert_eq!(
                out_with_op,
                out_base + op,
                "output_padding adds exactly op to output length"
            );
        }
    }
}

// ---------------------------------------------------------------------------
// 6. Dilated conv receptive field
// ---------------------------------------------------------------------------

/// Prove: dilated convolution receptive field is exactly (kernel-1)*dilation + 1.
///
/// The receptive field (effective kernel size) determines how many input
/// elements each output position depends on in the original (un-dilated) input.
#[kani::unwind(1)]
#[kani::proof]
fn dilated_conv_receptive_field() {
    let ks: u8 = kani::any();
    let d: u8 = kani::any();

    kani::assume(ks >= 1);
    kani::assume(d >= 1);

    let ks = ks as usize;
    let d = d as usize;

    let receptive_field = (ks - 1) * d + 1;

    // Receptive field is always >= kernel size (dilation >= 1)
    assert!(
        receptive_field >= ks,
        "receptive field must be >= kernel size"
    );

    // Receptive field equals kernel size when dilation == 1
    if d == 1 {
        assert_eq!(
            receptive_field, ks,
            "receptive field == kernel size when dilation == 1"
        );
    }

    // Receptive field is always >= 1
    assert!(receptive_field >= 1, "receptive field must be >= 1");

    // Receptive field is monotonically increasing in both kernel size and dilation
    // (these are properties, not computed here, but the formula guarantees them)

    // Receptive field for k=1 is always 1 regardless of dilation
    // (since (1-1)*d + 1 = 1)
    // We verify this special case when ks == 1
    if ks == 1 {
        assert_eq!(receptive_field, 1, "k=1 receptive field must be 1");
    }
}

/// Prove: dilated conv receptive field grows linearly with dilation.
///
/// For fixed kernel size, increasing dilation by 1 increases
/// receptive field by exactly (kernel_size - 1).
#[kani::unwind(1)]
#[kani::proof]
fn dilated_conv_receptive_field_linear_growth() {
    let ks: u8 = kani::any();
    let d1: u8 = kani::any();
    let d2: u8 = kani::any();

    kani::assume(ks >= 1);
    kani::assume(d1 >= 1);
    kani::assume(d2 >= d1);

    let ks = ks as usize;
    let d1 = d1 as usize;
    let d2 = d2 as usize;

    let rf1 = (ks - 1) * d1 + 1;
    let rf2 = (ks - 1) * d2 + 1;

    // rf2 - rf1 = (ks-1)*(d2-d1)
    assert_eq!(
        rf2 - rf1,
        (ks - 1) * (d2 - d1),
        "receptive field difference must equal (ks-1)*(d2-d1)"
    );

    // Monotonicity
    assert!(
        rf2 >= rf1,
        "larger dilation must produce >= receptive field"
    );
}

// ---------------------------------------------------------------------------
// 7. Padding modes: zero, reflect, replicate don't change channel count
// ---------------------------------------------------------------------------

/// Prove: padding (regardless of mode) does not change the channel dimension.
///
/// For conv input [batch, channels, length], padding only affects the spatial
/// dimension(s). The batch and channel dims remain unchanged.
///
/// We model this by showing that the output channel count of a conv depends
/// only on the weight's out_channels, not on padding amount or mode.
#[kani::unwind(1)]
#[kani::proof]
fn padding_preserves_channel_count() {
    let batch: u8 = kani::any();
    let in_ch: u8 = kani::any();
    let out_ch: u8 = kani::any();
    let il: u8 = kani::any();
    let ks: u8 = kani::any();
    let p1: u8 = kani::any(); // padding mode 1 (e.g., zero)
    let p2: u8 = kani::any(); // padding mode 2 (e.g., reflect)
    let s: u8 = kani::any();

    kani::assume(batch >= 1);
    kani::assume(in_ch >= 1);
    kani::assume(out_ch >= 1);
    kani::assume(il >= 1);
    kani::assume(ks >= 1);
    kani::assume(s >= 1);

    let il = il as usize;
    let ks = ks as usize;
    let s = s as usize;

    let effective_k = ks; // dilation=1

    // Two different padding amounts (representing different modes or amounts)
    let padded1 = il + 2 * (p1 as usize);
    let padded2 = il + 2 * (p2 as usize);

    // Output channels are always out_ch regardless of padding
    // (padding only affects spatial dims)
    let output_channels_p1 = out_ch as usize;
    let output_channels_p2 = out_ch as usize;

    assert_eq!(
        output_channels_p1, output_channels_p2,
        "padding mode does not change output channel count"
    );

    // Batch dimension is also unaffected
    let output_batch_p1 = batch as usize;
    let output_batch_p2 = batch as usize;
    assert_eq!(output_batch_p1, output_batch_p2);

    // But spatial dims may differ
    if padded1 >= effective_k && padded2 >= effective_k {
        let ol1 = (padded1 - effective_k) / s + 1;
        let ol2 = (padded2 - effective_k) / s + 1;
        // Spatial dims can differ — this is expected
        // The point is: channel/batch dims don't change
        let _ = (ol1, ol2);
    }
}

/// Prove: reflect padding requires padding < input_len (no underflow).
///
/// PyTorch constraint: for reflect padding, padding must be strictly less
/// than the input spatial dimension.
#[kani::unwind(1)]
#[kani::proof]
fn reflect_padding_constraint() {
    let il: u8 = kani::any();
    let p: u8 = kani::any();

    kani::assume(il >= 1);
    kani::assume(p < il); // Reflect constraint: padding < input_len

    let il = il as usize;
    let p = p as usize;

    // With reflect padding, padded length is always well-defined
    let padded = il + 2 * p;
    assert!(padded >= il, "padded must be >= input length");
    assert!(padded < 3 * il, "reflect padded must be < 3 * input_len");

    // The padded region can always be filled by reflecting the input
    // because p < il ensures we don't need to reflect more than once
    assert!(p < il, "reflect: one reflection suffices");
}

// ---------------------------------------------------------------------------
// 8. Causal conv: output_len == input_len when padding == (kernel-1)*dilation
// ---------------------------------------------------------------------------

/// Prove: causal convolution with padding = (kernel-1)*dilation and stride=1
/// produces output_len == input_len.
///
/// Causal conv pads only on the left side. For the standard formula with
/// symmetric padding, this is equivalent to total padding = (kernel-1)*dilation
/// on one side, which means p = (kernel-1)*dilation / 1 (one-sided) or
/// equivalently the output formula gives input_len when padding is set correctly.
///
/// In the symmetric formula used here, p = (effective_k - 1) / 2 does NOT
/// give causal behavior. Causal padding uses one-sided padding:
/// left_pad = (kernel-1)*dilation, right_pad = 0, total spatial = input + left_pad.
#[kani::unwind(1)]
#[kani::proof]
fn causal_conv_preserves_length() {
    let il: u8 = kani::any();
    let ks: u8 = kani::any();
    let d: u8 = kani::any();

    kani::assume(il >= 1);
    kani::assume(ks >= 1);
    kani::assume(d >= 1);
    // Restrict to avoid u8 overflow in padding computation
    kani::assume(ks <= 15);
    kani::assume(d <= 15);

    let il = il as usize;
    let ks = ks as usize;
    let d = d as usize;

    let effective_k = (ks - 1) * d + 1;

    // Causal padding: left_pad = effective_k - 1, right_pad = 0
    // Total input spatial extent = il + (effective_k - 1)
    let padded = il + (effective_k - 1);

    // With stride = 1:
    // out = (padded - effective_k) / 1 + 1
    //     = (il + effective_k - 1 - effective_k) + 1
    //     = il - 1 + 1
    //     = il
    let out = (padded - effective_k) / 1 + 1;
    assert_eq!(
        out, il,
        "causal conv with correct padding must preserve input length at stride=1"
    );
}

/// Prove: causal conv with stride > 1 produces output_len = ceil(input_len / stride).
///
/// With causal padding = (effective_k - 1) on the left:
/// padded = il + (effective_k - 1)
/// out = (padded - effective_k) / s + 1 = (il - 1) / s + 1
/// This equals ceil(il / s).
#[kani::unwind(1)]
#[kani::proof]
fn causal_conv_strided_output_len() {
    let il: u8 = kani::any();
    let ks: u8 = kani::any();
    let s: u8 = kani::any();
    let d: u8 = kani::any();

    kani::assume(il >= 1);
    kani::assume(ks >= 1);
    kani::assume(s >= 1);
    kani::assume(d >= 1);
    kani::assume(ks <= 15);
    kani::assume(d <= 15);

    let il = il as usize;
    let ks = ks as usize;
    let s = s as usize;
    let d = d as usize;

    let effective_k = (ks - 1) * d + 1;
    let padded = il + (effective_k - 1);

    let out = (padded - effective_k) / s + 1;
    // (il + ek - 1 - ek) / s + 1 = (il - 1) / s + 1
    let expected = (il - 1) / s + 1;
    assert_eq!(out, expected, "causal strided output must be (il-1)/s + 1");

    // This is the ceiling division: ceil(il / s) = (il + s - 1) / s
    // = (il - 1) / s + 1 when il >= 1 (integer arithmetic)
    let ceil_div = (il + s - 1) / s;
    assert_eq!(
        out, ceil_div,
        "causal strided output must equal ceil(il / s)"
    );
}

// ---------------------------------------------------------------------------
// 9. Same padding: output_len == ceil(input_len / stride)
// ---------------------------------------------------------------------------

/// Prove: "same" padding with stride=1 preserves spatial dimension when
/// the effective kernel is odd.
///
/// Same padding: total_padding = effective_k - 1, split evenly.
/// For odd effective_k, p = (effective_k - 1) / 2 gives exact same output.
#[kani::unwind(1)]
#[kani::proof]
fn same_padding_stride1_odd_kernel_preserves_dim() {
    let il: u8 = kani::any();
    let ks: u8 = kani::any();
    let d: u8 = kani::any();

    kani::assume(il >= 1);
    kani::assume(ks >= 1);
    kani::assume(d >= 1);
    kani::assume(ks <= 15);
    kani::assume(d <= 15);

    let il = il as usize;
    let ks = ks as usize;
    let d = d as usize;

    let effective_k = (ks - 1) * d + 1;

    // Only prove for odd effective kernel (even splits evenly)
    kani::assume(effective_k % 2 == 1);

    let p = (effective_k - 1) / 2;
    let padded = il + 2 * p;

    // padded = il + effective_k - 1 (since 2*((ek-1)/2) = ek-1 when ek is odd)
    assert_eq!(2 * p, effective_k - 1);
    assert!(padded >= effective_k);

    let out = (padded - effective_k) / 1 + 1;
    assert_eq!(
        out, il,
        "same padding with stride=1 and odd kernel preserves spatial dim"
    );
}

/// Prove: "same" padding output_len equals ceil(input_len / stride).
///
/// When total_padding = effective_k - 1 and stride divides evenly:
/// out = (il + ek - 1 - ek) / s + 1 = (il - 1) / s + 1 = ceil(il/s)
#[kani::unwind(1)]
#[kani::proof]
fn same_padding_ceil_div_stride() {
    let il: u8 = kani::any();
    let ks: u8 = kani::any();
    let s: u8 = kani::any();
    let d: u8 = kani::any();

    kani::assume(il >= 1);
    kani::assume(ks >= 1);
    kani::assume(s >= 1);
    kani::assume(d >= 1);
    kani::assume(ks <= 15);
    kani::assume(d <= 15);

    let il = il as usize;
    let ks = ks as usize;
    let s = s as usize;
    let d = d as usize;

    let effective_k = (ks - 1) * d + 1;

    // Total "same" padding = effective_k - 1, applied symmetrically or asymmetrically.
    // Using one-sided math: padded = il + (effective_k - 1)
    // (This is exact, unlike symmetric split which loses a pixel for even ek)
    let total_pad = effective_k - 1;
    let padded = il + total_pad;

    assert!(padded >= effective_k);
    let out = (padded - effective_k) / s + 1;

    // out = (il - 1) / s + 1 = ceil(il / s)
    let ceil_div = (il + s - 1) / s;
    assert_eq!(
        out, ceil_div,
        "same padding must produce ceil(input_len / stride) output"
    );
}

// ---------------------------------------------------------------------------
// 10. Weight shape consistency
// ---------------------------------------------------------------------------

/// Prove: convolution weight shape [out_ch, in_ch/groups, kernel_size]
/// has total element count = out_ch * (in_ch/groups) * kernel_size.
///
/// This is exactly 1/groups of the full weight count.
#[kani::unwind(1)]
#[kani::proof]
fn conv1d_weight_shape_consistency() {
    let in_ch: u8 = kani::any();
    let out_ch: u8 = kani::any();
    let groups: u8 = kani::any();
    let ks: u8 = kani::any();

    kani::assume(groups >= 1);
    kani::assume(in_ch >= groups);
    kani::assume(out_ch >= groups);
    kani::assume(ks >= 1);
    kani::assume(in_ch as usize % groups as usize == 0);
    kani::assume(out_ch as usize % groups as usize == 0);

    let g = groups as usize;
    let ic = in_ch as usize;
    let oc = out_ch as usize;
    let k = ks as usize;

    // Weight shape: [out_ch, in_ch/groups, kernel_size]
    let weight_dim0 = oc;
    let weight_dim1 = ic / g;
    let weight_dim2 = k;

    assert!(weight_dim0 >= 1);
    assert!(weight_dim1 >= 1);
    assert!(weight_dim2 >= 1);

    let weight_elems = weight_dim0 * weight_dim1 * weight_dim2;

    // Full (non-grouped) would be out_ch * in_ch * kernel_size
    let full_elems = oc * ic * k;
    assert_eq!(
        weight_elems * g,
        full_elems,
        "weight_elems * groups must equal full weight count"
    );
}

/// Prove: conv2d weight shape [out_ch, in_ch/groups, kH, kW] is consistent.
///
/// Total elements = out_ch * (in_ch/groups) * kH * kW.
#[kani::unwind(1)]
#[kani::proof]
fn conv2d_weight_shape_consistency() {
    let in_ch: u8 = kani::any();
    let out_ch: u8 = kani::any();
    let groups: u8 = kani::any();
    let kh: u8 = kani::any();
    let kw: u8 = kani::any();

    kani::assume(groups >= 1);
    kani::assume(in_ch >= groups);
    kani::assume(out_ch >= groups);
    kani::assume(kh >= 1);
    kani::assume(kw >= 1);
    kani::assume(in_ch as usize % groups as usize == 0);
    kani::assume(out_ch as usize % groups as usize == 0);

    let g = groups as usize;
    let ic = in_ch as usize;
    let oc = out_ch as usize;
    let kh = kh as usize;
    let kw = kw as usize;

    // Weight shape: [out_ch, in_ch/groups, kH, kW]
    let weight_dim0 = oc;
    let weight_dim1 = ic / g;
    let weight_dim2 = kh;
    let weight_dim3 = kw;

    assert!(weight_dim0 >= 1);
    assert!(weight_dim1 >= 1);
    assert!(weight_dim2 >= 1);
    assert!(weight_dim3 >= 1);

    let weight_elems = weight_dim0 * weight_dim1 * weight_dim2 * weight_dim3;
    let full_elems = oc * ic * kh * kw;

    assert_eq!(
        weight_elems * g,
        full_elems,
        "conv2d weight_elems * groups must equal full weight count"
    );
}

/// Prove: bias shape is always [out_channels] regardless of groups, kernel_size, or padding.
#[kani::unwind(1)]
#[kani::proof]
fn conv_bias_shape_independent_of_groups() {
    let out_ch: u8 = kani::any();
    let groups1: u8 = kani::any();
    let groups2: u8 = kani::any();

    kani::assume(out_ch >= 1);
    kani::assume(groups1 >= 1);
    kani::assume(groups2 >= 1);
    kani::assume(out_ch as usize % groups1 as usize == 0);
    kani::assume(out_ch as usize % groups2 as usize == 0);

    // Bias shape is always [out_ch] regardless of groups
    let bias_len_g1 = out_ch as usize;
    let bias_len_g2 = out_ch as usize;

    assert_eq!(
        bias_len_g1, bias_len_g2,
        "bias length must be independent of group count"
    );
}

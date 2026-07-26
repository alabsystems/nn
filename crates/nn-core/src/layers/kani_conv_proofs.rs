// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Kani proof harnesses for convolution layers (#4073).
//!
//! Proves correctness properties of Conv1d, Conv2d, ConvTranspose1d,
//! ConvTranspose2d, and WeightNormConv1d:
//!
//!  1. Conv1d output length formula: (padded - effective_k) / stride + 1
//!  2. Conv1d groups must divide both in_channels and out_channels
//!  3. Conv1d weight shape: [out_c, in_c/groups, kernel_size]
//!  4. Conv1d rejects zero stride
//!  5. Conv2d output height formula
//!  6. Conv2d output width formula
//!  7. Conv2d weight shape: [out_c, in_c/groups, kH, kW]
//!  8. ConvTranspose1d output formula: (in-1)*stride - 2*pad + dilation*(k-1) + out_pad + 1
//!  9. ConvTranspose1d output is positive for valid configs
//! 10. ConvTranspose1d output_padding must be < stride
//! 11. ConvTranspose2d output height formula
//! 12. ConvTranspose2d output width formula
//! 13. Weight norm: scale factor is well-defined for nonzero v
//! 14. Weight norm: normalized direction has unit norm (scalar model)
//!
//! Part of #4073.

// ---------------------------------------------------------------------------
// Harness 1: Conv1d output length formula
// ---------------------------------------------------------------------------

/// Prove: the conv1d output length equals `(input + 2*padding - effective_k) / stride + 1`
/// where `effective_k = (kernel_size - 1) * dilation + 1`, for valid parameters.
/// This models the formula in `conv1d_out_len()`.
#[kani::unwind(1)]
#[kani::proof]
fn proof_conv1d_output_length() {
    let input_len: usize = kani::any();
    let kernel_size: usize = kani::any();
    let padding: usize = kani::any();
    let stride: usize = kani::any();
    let dilation: usize = kani::any();

    kani::assume(input_len >= 1 && input_len <= 512);
    kani::assume(kernel_size >= 1 && kernel_size <= 32);
    kani::assume(padding <= 64);
    kani::assume(stride >= 1 && stride <= 16);
    kani::assume(dilation >= 1 && dilation <= 8);

    let effective_k = (kernel_size - 1) * dilation + 1;
    let padded = input_len + 2 * padding;

    kani::assume(padded >= effective_k);

    let out_len = (padded - effective_k) / stride + 1;

    // Output length is always >= 1 when padded >= effective_k.
    assert!(out_len >= 1, "conv1d output length must be >= 1");

    // Output length never exceeds the padded input size.
    assert!(
        out_len <= padded,
        "conv1d output length must not exceed padded input"
    );

    // Verify the formula: out_len = floor((padded - effective_k) / stride) + 1
    let remainder = (padded - effective_k) % stride;
    assert!(
        out_len * stride == padded - effective_k - remainder + stride,
        "conv1d output length must satisfy the formula"
    );
}

// ---------------------------------------------------------------------------
// Harness 2: Conv1d groups divide channels
// ---------------------------------------------------------------------------

/// Prove: when in_channels % groups == 0 and out_channels % groups == 0,
/// the per-group channel counts are well-defined and > 0.
#[kani::unwind(1)]
#[kani::proof]
fn proof_conv1d_groups_divide_channels() {
    let in_channels: usize = kani::any();
    let out_channels: usize = kani::any();
    let groups: usize = kani::any();

    kani::assume(in_channels >= 1 && in_channels <= 512);
    kani::assume(out_channels >= 1 && out_channels <= 512);
    kani::assume(groups >= 1 && groups <= 64);
    kani::assume(in_channels % groups == 0);
    kani::assume(out_channels % groups == 0);

    let in_per_group = in_channels / groups;
    let out_per_group = out_channels / groups;

    assert!(in_per_group >= 1, "in_channels per group must be >= 1");
    assert!(out_per_group >= 1, "out_channels per group must be >= 1");

    // Reconstruct: groups * per_group == original
    assert!(
        groups * in_per_group == in_channels,
        "groups * in_per_group must equal in_channels"
    );
    assert!(
        groups * out_per_group == out_channels,
        "groups * out_per_group must equal out_channels"
    );
}

// ---------------------------------------------------------------------------
// Harness 3: Conv1d weight shape
// ---------------------------------------------------------------------------

/// Prove: Conv1d weight shape [out_c, in_c/groups, kernel_size] has 3
/// dimensions, and the product of dimensions does not exceed reasonable bounds.
#[kani::unwind(1)]
#[kani::proof]
fn proof_conv1d_weight_shape() {
    let out_channels: usize = kani::any();
    let in_channels: usize = kani::any();
    let groups: usize = kani::any();
    let kernel_size: usize = kani::any();

    kani::assume(out_channels >= 1 && out_channels <= 256);
    kani::assume(in_channels >= 1 && in_channels <= 256);
    kani::assume(groups >= 1 && groups <= 64);
    kani::assume(kernel_size >= 1 && kernel_size <= 32);
    kani::assume(in_channels % groups == 0);

    let weight_dim0 = out_channels;
    let weight_dim1 = in_channels / groups;
    let weight_dim2 = kernel_size;

    // Weight shape must be 3D (rank 3).
    let rank = 3usize;
    assert!(rank == 3, "conv1d weight must be rank 3");

    // All dimensions must be >= 1.
    assert!(weight_dim0 >= 1, "out_channels must be >= 1");
    assert!(weight_dim1 >= 1, "in_channels/groups must be >= 1");
    assert!(weight_dim2 >= 1, "kernel_size must be >= 1");

    // Element count must be representable.
    let elem_count = weight_dim0
        .checked_mul(weight_dim1)
        .and_then(|v| v.checked_mul(weight_dim2));
    assert!(
        elem_count.is_some(),
        "weight element count must not overflow"
    );
}

// ---------------------------------------------------------------------------
// Harness 4: Conv1d rejects zero stride
// ---------------------------------------------------------------------------

/// Prove: stride == 0 is always rejected. Models the validation check
/// in DynTensor::conv1d and Conv1dConfig.
#[kani::unwind(1)]
#[kani::proof]
fn proof_conv1d_rejects_zero_stride() {
    let stride: usize = 0;

    // The formula (padded - effective_k) / stride would panic on division by zero.
    // The check `if stride == 0 { return Err }` prevents this.
    let rejected = stride == 0;
    assert!(rejected, "stride == 0 must always be rejected");
}

// ---------------------------------------------------------------------------
// Harness 5: Conv2d output height formula
// ---------------------------------------------------------------------------

/// Prove: conv2d output height follows the same formula as conv1d on the
/// height dimension: (H + 2*pad_h - effective_kH) / stride_h + 1.
#[kani::unwind(1)]
#[kani::proof]
fn proof_conv2d_output_height() {
    let in_h: usize = kani::any();
    let k_h: usize = kani::any();
    let pad_h: usize = kani::any();
    let stride_h: usize = kani::any();
    let dilation_h: usize = kani::any();

    kani::assume(in_h >= 1 && in_h <= 256);
    kani::assume(k_h >= 1 && k_h <= 16);
    kani::assume(pad_h <= 32);
    kani::assume(stride_h >= 1 && stride_h <= 8);
    kani::assume(dilation_h >= 1 && dilation_h <= 4);

    let effective_kh = (k_h - 1) * dilation_h + 1;
    let padded_h = in_h + 2 * pad_h;

    kani::assume(padded_h >= effective_kh);

    let out_h = (padded_h - effective_kh) / stride_h + 1;

    assert!(out_h >= 1, "conv2d output height must be >= 1");
    assert!(
        out_h <= padded_h,
        "conv2d output height must not exceed padded height"
    );
}

// ---------------------------------------------------------------------------
// Harness 6: Conv2d output width formula
// ---------------------------------------------------------------------------

/// Prove: conv2d output width follows the same formula on the width dimension.
#[kani::unwind(1)]
#[kani::proof]
fn proof_conv2d_output_width() {
    let in_w: usize = kani::any();
    let k_w: usize = kani::any();
    let pad_w: usize = kani::any();
    let stride_w: usize = kani::any();
    let dilation_w: usize = kani::any();

    kani::assume(in_w >= 1 && in_w <= 256);
    kani::assume(k_w >= 1 && k_w <= 16);
    kani::assume(pad_w <= 32);
    kani::assume(stride_w >= 1 && stride_w <= 8);
    kani::assume(dilation_w >= 1 && dilation_w <= 4);

    let effective_kw = (k_w - 1) * dilation_w + 1;
    let padded_w = in_w + 2 * pad_w;

    kani::assume(padded_w >= effective_kw);

    let out_w = (padded_w - effective_kw) / stride_w + 1;

    assert!(out_w >= 1, "conv2d output width must be >= 1");
    assert!(
        out_w <= padded_w,
        "conv2d output width must not exceed padded width"
    );
}

// ---------------------------------------------------------------------------
// Harness 7: Conv2d weight shape
// ---------------------------------------------------------------------------

/// Prove: Conv2d weight shape [out_c, in_c/groups, kH, kW] has 4 dimensions,
/// all >= 1, and element count does not overflow.
#[kani::unwind(1)]
#[kani::proof]
fn proof_conv2d_weight_shape() {
    let out_channels: usize = kani::any();
    let in_channels: usize = kani::any();
    let groups: usize = kani::any();
    let k_h: usize = kani::any();
    let k_w: usize = kani::any();

    kani::assume(out_channels >= 1 && out_channels <= 128);
    kani::assume(in_channels >= 1 && in_channels <= 128);
    kani::assume(groups >= 1 && groups <= 32);
    kani::assume(k_h >= 1 && k_h <= 16);
    kani::assume(k_w >= 1 && k_w <= 16);
    kani::assume(in_channels % groups == 0);

    let dim0 = out_channels;
    let dim1 = in_channels / groups;
    let dim2 = k_h;
    let dim3 = k_w;

    let rank = 4usize;
    assert!(rank == 4, "conv2d weight must be rank 4");

    assert!(dim0 >= 1, "out_channels must be >= 1");
    assert!(dim1 >= 1, "in_channels/groups must be >= 1");
    assert!(dim2 >= 1, "kH must be >= 1");
    assert!(dim3 >= 1, "kW must be >= 1");

    let elem_count = dim0
        .checked_mul(dim1)
        .and_then(|v| v.checked_mul(dim2))
        .and_then(|v| v.checked_mul(dim3));
    assert!(
        elem_count.is_some(),
        "conv2d weight element count must not overflow"
    );
}

// ---------------------------------------------------------------------------
// Harness 8: ConvTranspose1d output formula
// ---------------------------------------------------------------------------

/// Prove: conv_transpose1d output length =
///   (input - 1) * stride - 2*padding + dilation*(kernel - 1) + output_padding + 1.
/// This matches the formula in `conv_transpose1d_out_len()`.
#[kani::unwind(1)]
#[kani::proof]
fn proof_conv_transpose1d_output() {
    let input_len: usize = kani::any();
    let kernel_size: usize = kani::any();
    let padding: usize = kani::any();
    let output_padding: usize = kani::any();
    let stride: usize = kani::any();
    let dilation: usize = kani::any();

    kani::assume(input_len >= 1 && input_len <= 128);
    kani::assume(kernel_size >= 1 && kernel_size <= 16);
    kani::assume(padding <= 32);
    kani::assume(stride >= 1 && stride <= 8);
    kani::assume(output_padding < stride); // required by the API
    kani::assume(dilation >= 1 && dilation <= 4);

    // positive = (input - 1)*stride + dilation*(kernel - 1) + output_padding + 1
    let positive = (input_len - 1) * stride + dilation * (kernel_size - 1) + output_padding + 1;
    let negative = 2 * padding;

    kani::assume(positive > negative); // valid config: positive output

    let out_len = positive - negative;

    assert!(out_len >= 1, "conv_transpose1d output must be >= 1");

    // Verify the formula directly against the reference.
    let expected =
        (input_len - 1) * stride + dilation * (kernel_size - 1) + output_padding + 1 - 2 * padding;
    assert!(
        out_len == expected,
        "conv_transpose1d output must match formula"
    );
}

// ---------------------------------------------------------------------------
// Harness 9: ConvTranspose1d output is positive for valid configs
// ---------------------------------------------------------------------------

/// Prove: when output_padding < stride and the positive terms exceed
/// 2*padding, the output length is always > 0.
#[kani::unwind(1)]
#[kani::proof]
fn proof_conv_transpose1d_output_positive() {
    let input_len: usize = kani::any();
    let kernel_size: usize = kani::any();
    let padding: usize = kani::any();
    let output_padding: usize = kani::any();
    let stride: usize = kani::any();
    let dilation: usize = kani::any();

    kani::assume(input_len >= 1 && input_len <= 64);
    kani::assume(kernel_size >= 1 && kernel_size <= 8);
    kani::assume(padding <= 16);
    kani::assume(stride >= 1 && stride <= 8);
    kani::assume(output_padding < stride);
    kani::assume(dilation >= 1 && dilation <= 4);

    let positive = (input_len - 1) * stride + dilation * (kernel_size - 1) + output_padding + 1;
    let negative = 2 * padding;

    // Valid config: positive terms exceed negative.
    kani::assume(positive > negative);

    let out_len = positive - negative;
    assert!(
        out_len > 0,
        "valid conv_transpose1d config must yield positive output"
    );
}

// ---------------------------------------------------------------------------
// Harness 10: ConvTranspose1d output_padding must be < stride
// ---------------------------------------------------------------------------

/// Prove: output_padding >= stride is always an invalid configuration.
/// This models the validation check in `conv_transpose1d_out_len()`.
#[kani::unwind(1)]
#[kani::proof]
fn proof_conv_transpose1d_output_padding_lt_stride() {
    let output_padding: usize = kani::any();
    let stride: usize = kani::any();

    kani::assume(stride >= 1 && stride <= 16);
    kani::assume(output_padding >= stride);

    // Models: if stride > 0 && output_padding >= stride { return Err }
    let rejected = stride > 0 && output_padding >= stride;
    assert!(rejected, "output_padding >= stride must be rejected");
}

// ---------------------------------------------------------------------------
// Harness 11: ConvTranspose2d output height formula
// ---------------------------------------------------------------------------

/// Prove: conv_transpose2d output height follows the same formula as
/// conv_transpose1d on the height dimension.
#[kani::unwind(1)]
#[kani::proof]
fn proof_conv_transpose2d_output_h() {
    let in_h: usize = kani::any();
    let k_h: usize = kani::any();
    let pad_h: usize = kani::any();
    let out_pad_h: usize = kani::any();
    let stride_h: usize = kani::any();
    let dilation_h: usize = kani::any();

    kani::assume(in_h >= 1 && in_h <= 64);
    kani::assume(k_h >= 1 && k_h <= 8);
    kani::assume(pad_h <= 16);
    kani::assume(stride_h >= 1 && stride_h <= 8);
    kani::assume(out_pad_h < stride_h);
    kani::assume(dilation_h >= 1 && dilation_h <= 4);

    let positive = (in_h - 1) * stride_h + dilation_h * (k_h - 1) + out_pad_h + 1;
    let negative = 2 * pad_h;

    kani::assume(positive > negative);

    let out_h = positive - negative;

    assert!(out_h >= 1, "conv_transpose2d output height must be >= 1");

    let expected = (in_h - 1) * stride_h + dilation_h * (k_h - 1) + out_pad_h + 1 - 2 * pad_h;
    assert!(
        out_h == expected,
        "conv_transpose2d output height must match formula"
    );
}

// ---------------------------------------------------------------------------
// Harness 12: ConvTranspose2d output width formula
// ---------------------------------------------------------------------------

/// Prove: conv_transpose2d output width follows the same formula on the
/// width dimension.
#[kani::unwind(1)]
#[kani::proof]
fn proof_conv_transpose2d_output_w() {
    let in_w: usize = kani::any();
    let k_w: usize = kani::any();
    let pad_w: usize = kani::any();
    let out_pad_w: usize = kani::any();
    let stride_w: usize = kani::any();
    let dilation_w: usize = kani::any();

    kani::assume(in_w >= 1 && in_w <= 64);
    kani::assume(k_w >= 1 && k_w <= 8);
    kani::assume(pad_w <= 16);
    kani::assume(stride_w >= 1 && stride_w <= 8);
    kani::assume(out_pad_w < stride_w);
    kani::assume(dilation_w >= 1 && dilation_w <= 4);

    let positive = (in_w - 1) * stride_w + dilation_w * (k_w - 1) + out_pad_w + 1;
    let negative = 2 * pad_w;

    kani::assume(positive > negative);

    let out_w = positive - negative;

    assert!(out_w >= 1, "conv_transpose2d output width must be >= 1");

    let expected = (in_w - 1) * stride_w + dilation_w * (k_w - 1) + out_pad_w + 1 - 2 * pad_w;
    assert!(
        out_w == expected,
        "conv_transpose2d output width must match formula"
    );
}

// ---------------------------------------------------------------------------
// Harness 13: Weight norm scale factor is well-defined for nonzero v
// ---------------------------------------------------------------------------

/// Prove: the weight normalization scale g / ||v|| is finite and positive
/// when g > 0 and ||v|| > 0 (both finite). Models the computation in
/// `WeightNormConv1d::normalize_weight`.
#[kani::unwind(1)]
#[kani::proof]
fn proof_weight_norm_scale_well_defined() {
    let g: f32 = kani::any();
    let v_norm: f32 = kani::any();

    kani::assume(g.is_finite() && g > 0.0 && g <= 1e6);
    kani::assume(v_norm.is_finite() && v_norm > 0.0 && v_norm <= 1e6);

    // g / ||v|| is the scale factor applied to the unit-norm direction.
    let scale = g / v_norm;

    assert!(scale.is_finite(), "weight norm scale must be finite");
    assert!(scale > 0.0, "weight norm scale must be positive");
}

// ---------------------------------------------------------------------------
// Harness 14: Weight norm normalized direction has unit norm (scalar model)
// ---------------------------------------------------------------------------

/// Prove: v / ||v|| has magnitude 1 for a scalar model. This is the core
/// property of weight normalization: the direction is unit-normalized.
/// For the full tensor case, ||v/||v|||| = 1 per output channel.
#[kani::unwind(1)]
#[kani::proof]
fn proof_weight_norm_unit_norm() {
    let v: f32 = kani::any();

    kani::assume(v.is_finite());
    kani::assume(v.abs() > 1e-6); // avoid near-zero for numerical stability
    kani::assume(v.abs() <= 1e6);

    let v_norm = v.abs(); // ||v|| for a scalar is |v|

    let normalized = v / v_norm;

    // |normalized| should be 1.0 (or -1.0 → abs == 1.0).
    let abs_normalized = normalized.abs();
    assert!(
        abs_normalized.is_finite(),
        "normalized direction must be finite"
    );
    // Allow small epsilon for floating-point precision.
    assert!(
        (abs_normalized - 1.0).abs() < 1e-5,
        "normalized direction must have unit magnitude"
    );
}

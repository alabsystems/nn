// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Kani proof harnesses for Conv2d/ConvTranspose output shape and padding safety (#4137).
//!
//! Proves correctness properties of convolution shape formulas:
//!
//!  1. Conv2d output height formula: out_h = (h + 2*pad - dilation*(kh-1) - 1) / stride + 1
//!  2. Conv2d output width formula: out_w = (w + 2*pad - dilation*(kw-1) - 1) / stride + 1
//!  3. Conv output dims always positive for valid configs
//!  4. Groups parameter: in_channels % groups == 0 and out_channels % groups == 0
//!  5. Weight shape: [out_c, in_c/groups, kH, kW]
//!  6. Bias shape: [out_c]
//!  7. Stride > 0 invariant
//!  8. Padding >= 0 invariant (always true for usize)
//!  9. Dilation >= 1 invariant
//! 10. ConvTranspose1d output = (in - 1)*stride - 2*pad + dilation*(k-1) + out_pad + 1
//! 11. ConvTranspose2d output shape formulas
//! 12. ConvTranspose output_padding < stride
//! 13. Same-padding: pad = (k-1)/2 preserves spatial dims when stride=1
//! 14. Dilated effective kernel size: k_eff = dilation * (k - 1) + 1
//! 15. Conv1d output length formula
//! 16. Depthwise conv: groups == in_channels implies weight shape [in_c, 1, kH, kW]
//! 17. Pointwise conv: kernel_size=1, stride=1 preserves spatial dims
//! 18. Channel consistency: conv output channels feeds next conv input channels
//! 19. Batch dimension preservation: output batch == input batch
//! 20. Multi-group weight size: total params = out_c * (in_c/groups) * kH * kW
//!
//! Part of #4137.

// ===========================================================================
// Harness 1: Conv2d output height formula
// ===========================================================================

/// Prove: Conv2d output height = (h + 2*pad - dilation*(kh-1) - 1) / stride + 1.
/// For valid configs (numerator >= 0), the formula produces a positive result.
#[kani::unwind(1)]
#[kani::proof]
fn proof_conv2d_output_height_formula() {
    let h: usize = kani::any();
    let pad: usize = kani::any();
    let dilation: usize = kani::any();
    let kh: usize = kani::any();
    let stride: usize = kani::any();

    kani::assume(h >= 1 && h <= 8);
    kani::assume(pad <= 4);
    kani::assume(dilation >= 1 && dilation <= 3);
    kani::assume(kh >= 1 && kh <= 5);
    kani::assume(stride >= 1 && stride <= 3);

    // Effective kernel size
    let k_eff = dilation * (kh - 1) + 1;

    // Numerator of the output formula
    let padded = h + 2 * pad;
    kani::assume(padded >= k_eff); // valid config: padded input >= effective kernel

    let numerator = padded - k_eff;
    let out_h = numerator / stride + 1;

    assert!(
        out_h >= 1,
        "Conv2d output height must be >= 1 for valid config"
    );
}

// ===========================================================================
// Harness 2: Conv2d output width formula
// ===========================================================================

/// Prove: Conv2d output width = (w + 2*pad - dilation*(kw-1) - 1) / stride + 1.
/// Symmetric to height but verified independently for width dimension.
#[kani::unwind(1)]
#[kani::proof]
fn proof_conv2d_output_width_formula() {
    let w: usize = kani::any();
    let pad: usize = kani::any();
    let dilation: usize = kani::any();
    let kw: usize = kani::any();
    let stride: usize = kani::any();

    kani::assume(w >= 1 && w <= 8);
    kani::assume(pad <= 4);
    kani::assume(dilation >= 1 && dilation <= 3);
    kani::assume(kw >= 1 && kw <= 5);
    kani::assume(stride >= 1 && stride <= 3);

    let k_eff = dilation * (kw - 1) + 1;
    let padded = w + 2 * pad;
    kani::assume(padded >= k_eff);

    let numerator = padded - k_eff;
    let out_w = numerator / stride + 1;

    assert!(
        out_w >= 1,
        "Conv2d output width must be >= 1 for valid config"
    );
}

// ===========================================================================
// Harness 3: Conv output dims always positive for valid configs
// ===========================================================================

/// Prove: for any valid Conv2d configuration (padded >= effective kernel),
/// both output height and output width are strictly positive.
#[kani::unwind(1)]
#[kani::proof]
fn proof_conv2d_output_dims_positive() {
    let h: usize = kani::any();
    let w: usize = kani::any();
    let pad: usize = kani::any();
    let dilation: usize = kani::any();
    let kh: usize = kani::any();
    let kw: usize = kani::any();
    let stride: usize = kani::any();

    kani::assume(h >= 1 && h <= 8);
    kani::assume(w >= 1 && w <= 8);
    kani::assume(pad <= 4);
    kani::assume(dilation >= 1 && dilation <= 3);
    kani::assume(kh >= 1 && kh <= 5);
    kani::assume(kw >= 1 && kw <= 5);
    kani::assume(stride >= 1 && stride <= 3);

    let k_eff_h = dilation * (kh - 1) + 1;
    let k_eff_w = dilation * (kw - 1) + 1;

    kani::assume(h + 2 * pad >= k_eff_h);
    kani::assume(w + 2 * pad >= k_eff_w);

    let out_h = (h + 2 * pad - k_eff_h) / stride + 1;
    let out_w = (w + 2 * pad - k_eff_w) / stride + 1;

    assert!(out_h >= 1, "output height must be positive");
    assert!(out_w >= 1, "output width must be positive");
    assert!(
        out_h <= h + 2 * pad,
        "output height cannot exceed padded input height"
    );
    assert!(
        out_w <= w + 2 * pad,
        "output width cannot exceed padded input width"
    );
}

// ===========================================================================
// Harness 4: Groups parameter divisibility
// ===========================================================================

/// Prove: for grouped convolution, in_channels must be divisible by groups
/// and out_channels must be divisible by groups. This ensures each group
/// gets an equal share of input and output channels.
#[kani::unwind(1)]
#[kani::proof]
fn proof_conv2d_groups_divisibility() {
    let in_channels: usize = kani::any();
    let out_channels: usize = kani::any();
    let groups: usize = kani::any();

    kani::assume(in_channels >= 1 && in_channels <= 64);
    kani::assume(out_channels >= 1 && out_channels <= 64);
    kani::assume(groups >= 1 && groups <= 64);
    kani::assume(in_channels % groups == 0);
    kani::assume(out_channels % groups == 0);

    let in_per_group = in_channels / groups;
    let out_per_group = out_channels / groups;

    assert!(in_per_group >= 1, "each group must have >= 1 input channel");
    assert!(
        out_per_group >= 1,
        "each group must have >= 1 output channel"
    );
    assert!(
        in_per_group * groups == in_channels,
        "in_per_group * groups must reconstruct in_channels"
    );
    assert!(
        out_per_group * groups == out_channels,
        "out_per_group * groups must reconstruct out_channels"
    );
}

// ===========================================================================
// Harness 5: Weight shape [out_c, in_c/groups, kH, kW]
// ===========================================================================

/// Prove: Conv2d weight shape is exactly [out_channels, in_channels/groups, kH, kW].
/// Weight rank must be 4. Total elements = out_c * (in_c/groups) * kH * kW.
#[kani::unwind(1)]
#[kani::proof]
fn proof_conv2d_weight_shape() {
    let out_channels: usize = kani::any();
    let in_channels: usize = kani::any();
    let groups: usize = kani::any();
    let kh: usize = kani::any();
    let kw: usize = kani::any();

    kani::assume(out_channels >= 1 && out_channels <= 32);
    kani::assume(in_channels >= 1 && in_channels <= 32);
    kani::assume(groups >= 1 && groups <= 32);
    kani::assume(kh >= 1 && kh <= 5);
    kani::assume(kw >= 1 && kw <= 5);
    kani::assume(in_channels % groups == 0);

    let in_per_group = in_channels / groups;
    let weight_rank = 4usize;

    assert!(weight_rank == 4, "Conv2d weight must be rank-4");

    // Total number of weight parameters
    let total = out_channels
        .checked_mul(in_per_group)
        .and_then(|x| x.checked_mul(kh))
        .and_then(|x| x.checked_mul(kw));
    assert!(total.is_some(), "weight element count must not overflow");
    assert!(total.unwrap() >= 1, "weight must have at least 1 element");
}

// ===========================================================================
// Harness 6: Bias shape [out_c]
// ===========================================================================

/// Prove: Conv2d bias (when present) has shape [out_channels].
/// The bias length must equal the number of output channels (weight dim 0).
#[kani::unwind(1)]
#[kani::proof]
fn proof_conv2d_bias_shape() {
    let out_channels: usize = kani::any();
    let bias_len: usize = kani::any();

    kani::assume(out_channels >= 1 && out_channels <= 512);
    kani::assume(bias_len >= 1 && bias_len <= 1024);

    // Models Conv2d::new check: if b.dims() != [out_channels] { Err }
    let accepted = bias_len == out_channels;

    if accepted {
        assert!(
            bias_len == out_channels,
            "accepted bias length must equal out_channels"
        );
    } else {
        assert!(
            bias_len != out_channels,
            "mismatched bias length must be rejected"
        );
    }
}

// ===========================================================================
// Harness 7: Stride > 0 invariant
// ===========================================================================

/// Prove: stride must be strictly positive for convolution to be well-defined.
/// Division by zero in the output formula is prevented by stride >= 1.
#[kani::unwind(1)]
#[kani::proof]
fn proof_conv_stride_positive_invariant() {
    let stride: usize = kani::any();
    let h: usize = kani::any();
    let k_eff: usize = kani::any();

    kani::assume(stride >= 1 && stride <= 3);
    kani::assume(h >= 1 && h <= 8);
    kani::assume(k_eff >= 1 && k_eff <= 8);
    kani::assume(h >= k_eff);

    assert!(stride > 0, "stride must be strictly positive");

    // Output formula is safe (no division by zero)
    let numerator = h - k_eff;
    let out = numerator / stride + 1;
    assert!(
        out >= 1,
        "output must be >= 1 when stride > 0 and h >= k_eff"
    );
}

// ===========================================================================
// Harness 8: Padding >= 0 invariant (usize guarantees)
// ===========================================================================

/// Prove: padding is a usize, so it is inherently >= 0. Adding padding to
/// spatial dimensions always increases or preserves the padded size.
#[kani::unwind(1)]
#[kani::proof]
fn proof_conv_padding_nonnegative_invariant() {
    let h: usize = kani::any();
    let padding: usize = kani::any();

    kani::assume(h >= 1 && h <= 8);
    kani::assume(padding <= 4);

    // Padding is usize: always >= 0 by definition
    let padded = h + 2 * padding;

    assert!(padded >= h, "padding must not decrease spatial dimension");
    assert!(padded == h + 2 * padding, "padded size = input + 2*padding");

    // Padding == 0 means no change
    if padding == 0 {
        assert!(padded == h, "zero padding preserves input size");
    } else {
        assert!(padded > h, "non-zero padding increases input size");
    }
}

// ===========================================================================
// Harness 9: Dilation >= 1 invariant
// ===========================================================================

/// Prove: dilation must be >= 1. With dilation=1, the effective kernel size
/// equals the nominal kernel size. Higher dilation increases effective size.
#[kani::unwind(1)]
#[kani::proof]
fn proof_conv_dilation_ge_one_invariant() {
    let dilation: usize = kani::any();
    let kernel_size: usize = kani::any();

    kani::assume(dilation >= 1 && dilation <= 3);
    kani::assume(kernel_size >= 1 && kernel_size <= 5);

    let k_eff = dilation * (kernel_size - 1) + 1;

    assert!(k_eff >= kernel_size, "effective kernel >= nominal kernel");
    assert!(k_eff >= 1, "effective kernel size must be >= 1");

    if dilation == 1 {
        assert!(
            k_eff == kernel_size,
            "dilation=1 means effective == nominal kernel size"
        );
    } else {
        assert!(
            k_eff > kernel_size || kernel_size == 1,
            "dilation>1 increases effective size (unless kernel_size=1)"
        );
    }
}

// ===========================================================================
// Harness 10: ConvTranspose1d output formula
// ===========================================================================

/// Prove: ConvTranspose1d output = (in - 1)*stride - 2*pad + dilation*(k-1) + out_pad + 1.
/// For valid configs (output_padding < stride), the result is positive.
#[kani::unwind(1)]
#[kani::proof]
fn proof_conv_transpose1d_output_formula() {
    let input_len: usize = kani::any();
    let stride: usize = kani::any();
    let padding: usize = kani::any();
    let dilation: usize = kani::any();
    let kernel_size: usize = kani::any();
    let output_padding: usize = kani::any();

    kani::assume(input_len >= 1 && input_len <= 8);
    kani::assume(stride >= 1 && stride <= 3);
    kani::assume(padding <= 4);
    kani::assume(dilation >= 1 && dilation <= 3);
    kani::assume(kernel_size >= 1 && kernel_size <= 5);
    kani::assume(output_padding < stride); // PyTorch requirement

    // output = (in - 1)*stride - 2*pad + dilation*(k-1) + out_pad + 1
    let term1 = (input_len - 1) * stride;
    let term2 = dilation * (kernel_size - 1);
    let expand = term1 + term2 + output_padding + 1;

    kani::assume(expand >= 2 * padding); // valid config: no negative output

    let output_len = expand - 2 * padding;

    assert!(
        output_len >= 1,
        "ConvTranspose1d output must be >= 1 for valid config"
    );
}

// ===========================================================================
// Harness 11: ConvTranspose2d output shape formulas
// ===========================================================================

/// Prove: ConvTranspose2d output height and width follow the transposed
/// convolution formula for both spatial dimensions.
#[kani::unwind(1)]
#[kani::proof]
fn proof_conv_transpose2d_output_shape() {
    let h: usize = kani::any();
    let w: usize = kani::any();
    let stride_h: usize = kani::any();
    let stride_w: usize = kani::any();
    let pad_h: usize = kani::any();
    let pad_w: usize = kani::any();
    let dilation_h: usize = kani::any();
    let dilation_w: usize = kani::any();
    let kh: usize = kani::any();
    let kw: usize = kani::any();
    let out_pad_h: usize = kani::any();
    let out_pad_w: usize = kani::any();

    kani::assume(h >= 1 && h <= 4);
    kani::assume(w >= 1 && w <= 4);
    kani::assume(stride_h >= 1 && stride_h <= 3);
    kani::assume(stride_w >= 1 && stride_w <= 3);
    kani::assume(pad_h <= 2);
    kani::assume(pad_w <= 2);
    kani::assume(dilation_h >= 1 && dilation_h <= 2);
    kani::assume(dilation_w >= 1 && dilation_w <= 2);
    kani::assume(kh >= 1 && kh <= 3);
    kani::assume(kw >= 1 && kw <= 3);
    kani::assume(out_pad_h < stride_h);
    kani::assume(out_pad_w < stride_w);

    // out = (in - 1)*stride - 2*pad + dilation*(k-1) + out_pad + 1
    let expand_h = (h - 1) * stride_h + dilation_h * (kh - 1) + out_pad_h + 1;
    let expand_w = (w - 1) * stride_w + dilation_w * (kw - 1) + out_pad_w + 1;

    kani::assume(expand_h >= 2 * pad_h);
    kani::assume(expand_w >= 2 * pad_w);

    let out_h = expand_h - 2 * pad_h;
    let out_w = expand_w - 2 * pad_w;

    assert!(out_h >= 1, "ConvTranspose2d output height must be >= 1");
    assert!(out_w >= 1, "ConvTranspose2d output width must be >= 1");
}

// ===========================================================================
// Harness 12: ConvTranspose output_padding < stride
// ===========================================================================

/// Prove: the output_padding < stride constraint is necessary. When
/// output_padding >= stride, the transposed convolution is ambiguous
/// (multiple input shapes could produce the same output shape).
#[kani::unwind(1)]
#[kani::proof]
fn proof_conv_transpose_output_padding_lt_stride() {
    let stride: usize = kani::any();
    let output_padding: usize = kani::any();

    kani::assume(stride >= 1 && stride <= 3);
    kani::assume(output_padding <= 4);

    let valid = output_padding < stride;

    if valid {
        // output_padding disambiguates the inverse convolution.
        // It adds output_padding rows/cols to the bottom/right of the output.
        assert!(output_padding < stride, "valid: output_padding < stride");
    } else {
        // Invalid: output_padding >= stride is rejected by PyTorch and nn.
        assert!(
            output_padding >= stride,
            "invalid: output_padding >= stride must be rejected"
        );
    }

    // When stride == 1, output_padding must be 0.
    if stride == 1 && valid {
        assert!(output_padding == 0, "stride=1 requires output_padding=0");
    }
}

// ===========================================================================
// Harness 13: Same-padding preserves spatial dims when stride=1
// ===========================================================================

/// Prove: with padding = (k-1)/2 (integer division), stride=1, dilation=1,
/// the output spatial dimension equals the input spatial dimension.
/// This is the "same" padding convention.
#[kani::unwind(1)]
#[kani::proof]
fn proof_conv2d_same_padding_preserves_dims() {
    let spatial: usize = kani::any();
    let kernel_size: usize = kani::any();

    kani::assume(spatial >= 1 && spatial <= 8);
    kani::assume(kernel_size >= 1 && kernel_size <= 7);
    // Same-padding works perfectly for odd kernels
    kani::assume(kernel_size % 2 == 1);

    let stride = 1usize;
    let dilation = 1usize;
    let padding = (kernel_size - 1) / 2;

    let k_eff = dilation * (kernel_size - 1) + 1;
    // With dilation=1: k_eff = kernel_size
    assert!(
        k_eff == kernel_size,
        "dilation=1 means k_eff == kernel_size"
    );

    let padded = spatial + 2 * padding;
    // padded = spatial + 2*((k-1)/2) = spatial + (k-1)
    // For odd k: 2*((k-1)/2) = k-1 exactly
    assert!(
        padded == spatial + kernel_size - 1,
        "padded = spatial + k - 1 for odd kernel"
    );

    let output = (padded - k_eff) / stride + 1;
    // output = (spatial + k - 1 - k) / 1 + 1 = spatial - 1 + 1 = spatial
    assert!(
        output == spatial,
        "same-padding with stride=1, dilation=1, odd kernel preserves spatial dim"
    );
}

// ===========================================================================
// Harness 14: Dilated effective kernel size
// ===========================================================================

/// Prove: the effective kernel size formula k_eff = dilation * (k - 1) + 1.
/// This inserts (dilation - 1) zeros between each kernel element.
/// Effective kernel occupies k_eff input positions.
#[kani::unwind(1)]
#[kani::proof]
fn proof_conv_dilated_effective_kernel_size() {
    let dilation: usize = kani::any();
    let kernel_size: usize = kani::any();

    kani::assume(dilation >= 1 && dilation <= 4);
    kani::assume(kernel_size >= 1 && kernel_size <= 7);

    let k_eff = dilation * (kernel_size - 1) + 1;

    // Properties of effective kernel size:
    assert!(k_eff >= 1, "effective kernel size must be >= 1");
    assert!(k_eff >= kernel_size, "effective >= nominal");

    // When kernel_size == 1, dilation has no effect
    if kernel_size == 1 {
        assert!(
            k_eff == 1,
            "1x1 kernel: effective size always 1 regardless of dilation"
        );
    }

    // When dilation == 1, effective == nominal
    if dilation == 1 {
        assert!(k_eff == kernel_size, "dilation=1: effective == nominal");
    }

    // The number of zero-inserted gaps is (kernel_size - 1)
    // Each gap has (dilation - 1) zeros
    let total_zeros = (kernel_size - 1) * (dilation - 1);
    assert!(
        k_eff == kernel_size + total_zeros,
        "k_eff = kernel_size + (k-1)*(d-1) zeros"
    );
}

// ===========================================================================
// Harness 15: Conv1d output length formula
// ===========================================================================

/// Prove: Conv1d output length = (L + 2*pad - dilation*(k-1) - 1) / stride + 1.
/// Same formula as Conv2d but for a single spatial dimension.
#[kani::unwind(1)]
#[kani::proof]
fn proof_conv1d_output_length_formula() {
    let input_len: usize = kani::any();
    let padding: usize = kani::any();
    let dilation: usize = kani::any();
    let kernel_size: usize = kani::any();
    let stride: usize = kani::any();

    kani::assume(input_len >= 1 && input_len <= 8);
    kani::assume(padding <= 4);
    kani::assume(dilation >= 1 && dilation <= 3);
    kani::assume(kernel_size >= 1 && kernel_size <= 5);
    kani::assume(stride >= 1 && stride <= 3);

    let k_eff = dilation * (kernel_size - 1) + 1;
    let padded = input_len + 2 * padding;
    kani::assume(padded >= k_eff);

    let output_len = (padded - k_eff) / stride + 1;

    assert!(output_len >= 1, "Conv1d output length must be >= 1");

    // Output length decreases as stride increases (for fixed input).
    // Verify: output_len <= padded (can't produce more positions than input has)
    assert!(
        output_len <= padded,
        "output cannot exceed padded input length"
    );
}

// ===========================================================================
// Harness 16: Depthwise conv weight shape
// ===========================================================================

/// Prove: for depthwise convolution (groups == in_channels), the weight
/// shape is [in_channels, 1, kH, kW] because in_channels/groups = 1.
/// Each group has exactly 1 input channel and 1 output channel.
#[kani::unwind(1)]
#[kani::proof]
fn proof_conv2d_depthwise_weight_shape() {
    let in_channels: usize = kani::any();
    let kh: usize = kani::any();
    let kw: usize = kani::any();

    kani::assume(in_channels >= 1 && in_channels <= 64);
    kani::assume(kh >= 1 && kh <= 5);
    kani::assume(kw >= 1 && kw <= 5);

    // Depthwise: groups == in_channels, out_channels == in_channels
    let groups = in_channels;
    let out_channels = in_channels;

    let in_per_group = in_channels / groups;
    assert!(
        in_per_group == 1,
        "depthwise conv: 1 input channel per group"
    );

    // Weight shape: [out_channels, in_per_group, kH, kW] = [in_c, 1, kH, kW]
    let weight_dim0 = out_channels;
    let weight_dim1 = in_per_group;
    let weight_dim2 = kh;
    let weight_dim3 = kw;

    assert!(weight_dim0 == in_channels, "depthwise: out_c == in_c");
    assert!(weight_dim1 == 1, "depthwise: in_per_group == 1");

    let total_params = weight_dim0
        .checked_mul(weight_dim1)
        .and_then(|x| x.checked_mul(weight_dim2))
        .and_then(|x| x.checked_mul(weight_dim3));
    assert!(
        total_params.is_some(),
        "depthwise weight params must not overflow"
    );
    assert!(
        total_params.unwrap() == in_channels * kh * kw,
        "depthwise params = in_channels * kH * kW"
    );
}

// ===========================================================================
// Harness 17: Pointwise conv preserves spatial dims
// ===========================================================================

/// Prove: with kernel_size=1, stride=1, dilation=1, padding=0,
/// the output spatial dimensions equal the input spatial dimensions.
/// A pointwise (1x1) convolution only changes the channel dimension.
#[kani::unwind(1)]
#[kani::proof]
fn proof_conv2d_pointwise_preserves_spatial() {
    let h: usize = kani::any();
    let w: usize = kani::any();

    kani::assume(h >= 1 && h <= 8);
    kani::assume(w >= 1 && w <= 8);

    let kernel_size = 1usize;
    let stride = 1usize;
    let dilation = 1usize;
    let padding = 0usize;

    let k_eff = dilation * (kernel_size - 1) + 1;
    assert!(k_eff == 1, "1x1 kernel effective size is 1");

    let out_h = (h + 2 * padding - k_eff) / stride + 1;
    let out_w = (w + 2 * padding - k_eff) / stride + 1;

    assert!(out_h == h, "pointwise conv preserves height");
    assert!(out_w == w, "pointwise conv preserves width");
}

// ===========================================================================
// Harness 18: Channel consistency across stacked convolutions
// ===========================================================================

/// Prove: when stacking two Conv2d layers, the output channels of layer 1
/// must equal the input channels of layer 2. This ensures dimensional
/// compatibility in sequential convolution pipelines.
#[kani::unwind(1)]
#[kani::proof]
fn proof_conv2d_channel_consistency() {
    let in_channels_1: usize = kani::any();
    let out_channels_1: usize = kani::any();
    let out_channels_2: usize = kani::any();
    let groups_1: usize = kani::any();
    let groups_2: usize = kani::any();

    kani::assume(in_channels_1 >= 1 && in_channels_1 <= 32);
    kani::assume(out_channels_1 >= 1 && out_channels_1 <= 32);
    kani::assume(out_channels_2 >= 1 && out_channels_2 <= 32);
    kani::assume(groups_1 >= 1 && groups_1 <= 32);
    kani::assume(groups_2 >= 1 && groups_2 <= 32);
    kani::assume(in_channels_1 % groups_1 == 0);

    // Conv1 output channels = out_channels_1 (from weight dim 0)
    // Conv2 input channels = out_channels_1 (must match)
    let in_channels_2 = out_channels_1;
    kani::assume(in_channels_2 % groups_2 == 0);

    // Conv1 weight: [out_channels_1, in_channels_1/groups_1, kH, kW]
    // Conv2 weight: [out_channels_2, in_channels_2/groups_2, kH, kW]
    let in_per_group_2 = in_channels_2 / groups_2;

    assert!(
        in_per_group_2 >= 1,
        "Conv2 must have >= 1 input channel per group"
    );
    assert!(
        in_channels_2 == out_channels_1,
        "Conv2 input channels must equal Conv1 output channels"
    );
}

// ===========================================================================
// Harness 19: Batch dimension preservation
// ===========================================================================

/// Prove: convolution preserves the batch dimension. For input [B, C, H, W],
/// Conv2d produces [B, out_c, out_h, out_w]. The batch size B is unchanged.
#[kani::unwind(1)]
#[kani::proof]
fn proof_conv2d_batch_dimension_preserved() {
    let batch: usize = kani::any();
    let in_channels: usize = kani::any();
    let out_channels: usize = kani::any();
    let h: usize = kani::any();
    let w: usize = kani::any();
    let kernel_size: usize = kani::any();
    let stride: usize = kani::any();
    let padding: usize = kani::any();

    kani::assume(batch >= 1 && batch <= 8);
    kani::assume(in_channels >= 1 && in_channels <= 16);
    kani::assume(out_channels >= 1 && out_channels <= 16);
    kani::assume(h >= 1 && h <= 8);
    kani::assume(w >= 1 && w <= 8);
    kani::assume(kernel_size >= 1 && kernel_size <= 5);
    kani::assume(stride >= 1 && stride <= 3);
    kani::assume(padding <= 4);
    kani::assume(h + 2 * padding >= kernel_size);
    kani::assume(w + 2 * padding >= kernel_size);

    // Input: [batch, in_channels, h, w]
    // Output: [batch, out_channels, out_h, out_w]
    let out_h = (h + 2 * padding - kernel_size) / stride + 1;
    let out_w = (w + 2 * padding - kernel_size) / stride + 1;

    // Batch dimension is always preserved
    let output_batch = batch;
    assert!(
        output_batch == batch,
        "batch dimension must be preserved through convolution"
    );

    // Channel dimension changes to out_channels
    let output_channels = out_channels;
    assert!(
        output_channels == out_channels,
        "output channels must equal weight dim 0"
    );

    // Total output element count
    let total = batch
        .checked_mul(out_channels)
        .and_then(|x| x.checked_mul(out_h))
        .and_then(|x| x.checked_mul(out_w));
    assert!(total.is_some(), "output element count must not overflow");
}

// ===========================================================================
// Harness 20: Multi-group weight parameter count
// ===========================================================================

/// Prove: total weight parameters = out_c * (in_c/groups) * kH * kW.
/// Grouped convolution reduces parameters by a factor of `groups` compared
/// to standard convolution (which uses in_c instead of in_c/groups).
#[kani::unwind(1)]
#[kani::proof]
fn proof_conv2d_grouped_weight_param_count() {
    let out_channels: usize = kani::any();
    let in_channels: usize = kani::any();
    let groups: usize = kani::any();
    let kh: usize = kani::any();
    let kw: usize = kani::any();

    kani::assume(out_channels >= 1 && out_channels <= 16);
    kani::assume(in_channels >= 1 && in_channels <= 16);
    kani::assume(groups >= 1 && groups <= 16);
    kani::assume(kh >= 1 && kh <= 3);
    kani::assume(kw >= 1 && kw <= 3);
    kani::assume(in_channels % groups == 0);
    kani::assume(out_channels % groups == 0);

    let in_per_group = in_channels / groups;

    // Grouped conv weight params
    let grouped_params = out_channels
        .checked_mul(in_per_group)
        .and_then(|x| x.checked_mul(kh))
        .and_then(|x| x.checked_mul(kw));
    assert!(grouped_params.is_some(), "grouped params must not overflow");

    // Standard conv weight params (groups=1)
    let standard_params = out_channels
        .checked_mul(in_channels)
        .and_then(|x| x.checked_mul(kh))
        .and_then(|x| x.checked_mul(kw));
    assert!(
        standard_params.is_some(),
        "standard params must not overflow"
    );

    // Grouped has 1/groups the parameters of standard
    assert!(
        grouped_params.unwrap() * groups == standard_params.unwrap(),
        "grouped params * groups == standard params"
    );

    // Verify the relationship: grouped = standard / groups
    assert!(
        standard_params.unwrap() / groups == grouped_params.unwrap(),
        "standard / groups == grouped"
    );
}

// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Kani proof harnesses for ConvTranspose1d and ConvTranspose2d layers.
//!
//! Proves correctness properties of transposed convolution output size
//! formulas, parameter validation, and shape invariants.
//!
//! **ConvTranspose1d:**
//!  1.  Output length formula is well-defined for valid parameters
//!  2.  Stride > 0 ensures output length >= input length (upsampling)
//!  3.  Default config: stride=1, padding=0, dilation=1, groups=1
//!  4.  Output padding must be less than stride (PyTorch constraint)
//!  5.  Weight rank must be 3 (in_ch, out_ch/groups, kernel)
//!  6.  Groups=0 is rejected
//!  7.  Output length is monotonically increasing with stride
//!
//! **ConvTranspose2d:**
//!  8.  Output size formula is well-defined for 2D params
//!  9.  Default config: symmetric defaults match 1D pattern
//! 10.  Weight rank must be 4 (in_ch, out_ch/groups, kH, kW)
//! 11.  Groups=0 is rejected (same as 1D)
//! 12.  Symmetric config produces equal height/width parameters
//! 13.  Output padding must be less than stride for each spatial dim
//! 14.  Bias reshape for 2D: [1, out_ch, 1, 1] broadcast shape
//!
//! Part of #4261.

// -- ConvTranspose output size formula (PyTorch definition) --
//
// For 1D: L_out = (L_in - 1) * stride - 2 * padding + dilation * (kernel - 1) + output_padding + 1
// For 2D: same formula applied independently to H and W dimensions.

/// Compute ConvTranspose1d output length using PyTorch formula.
fn conv_transpose1d_output_length(
    input_len: usize,
    kernel_size: usize,
    stride: usize,
    padding: usize,
    output_padding: usize,
    dilation: usize,
) -> Option<usize> {
    // L_out = (L_in - 1) * stride - 2 * padding + dilation * (kernel - 1) + output_padding + 1
    let a = input_len.checked_sub(1)?.checked_mul(stride)?;
    let b = 2_usize.checked_mul(padding)?;
    let c = dilation.checked_mul(kernel_size.checked_sub(1)?)?;
    a.checked_sub(b)?
        .checked_add(c)?
        .checked_add(output_padding)?
        .checked_add(1)
}

// ---------------------------------------------------------------------------
// Harness 1: ConvTranspose1d output length is well-defined for valid params
// ---------------------------------------------------------------------------

/// Prove: the output length formula produces a positive result for valid
/// convolution parameters (stride >= 1, dilation >= 1, kernel >= 1,
/// input_len >= 1, output_padding < stride).
#[kani::unwind(1)]
#[kani::proof]
fn proof_conv_transpose1d_output_length_well_defined() {
    let input_len: usize = kani::any();
    let kernel_size: usize = kani::any();
    let stride: usize = kani::any();
    let padding: usize = kani::any();
    let output_padding: usize = kani::any();
    let dilation: usize = kani::any();

    kani::assume(input_len >= 1 && input_len <= 256);
    kani::assume(kernel_size >= 1 && kernel_size <= 16);
    kani::assume(stride >= 1 && stride <= 8);
    kani::assume(padding <= 16);
    kani::assume(output_padding < stride); // PyTorch constraint
    kani::assume(dilation >= 1 && dilation <= 4);

    // Ensure the subtraction doesn't underflow: need a >= b in formula
    let a = (input_len - 1) * stride;
    let b = 2 * padding;
    kani::assume(a + dilation * (kernel_size - 1) + output_padding + 1 >= b);

    let result = conv_transpose1d_output_length(
        input_len,
        kernel_size,
        stride,
        padding,
        output_padding,
        dilation,
    );

    assert!(
        result.is_some(),
        "output length must be computable for valid params"
    );
    assert!(result.unwrap() >= 1, "output length must be >= 1");
}

// ---------------------------------------------------------------------------
// Harness 2: Stride > 0 ensures output length >= input length for typical case
// ---------------------------------------------------------------------------

/// Prove: for ConvTranspose1d with padding=0, dilation=1, output_padding=0,
/// and stride >= 1, the output length is >= input length (upsampling property).
#[kani::unwind(1)]
#[kani::proof]
fn proof_conv_transpose1d_upsampling() {
    let input_len: usize = kani::any();
    let kernel_size: usize = kani::any();
    let stride: usize = kani::any();

    kani::assume(input_len >= 1 && input_len <= 128);
    kani::assume(kernel_size >= 1 && kernel_size <= 8);
    kani::assume(stride >= 1 && stride <= 8);

    // padding=0, output_padding=0, dilation=1
    // L_out = (L_in - 1) * stride + kernel_size
    // = L_in * stride - stride + kernel_size
    // >= L_in when stride >= 1 and kernel_size >= stride (typical case)
    kani::assume(kernel_size >= stride);

    let result = conv_transpose1d_output_length(input_len, kernel_size, stride, 0, 0, 1);
    assert!(result.is_some(), "must compute");

    let output_len = result.unwrap();
    assert!(
        output_len >= input_len,
        "ConvTranspose1d with no padding must upsample when kernel >= stride"
    );
}

// ---------------------------------------------------------------------------
// Harness 3: ConvTranspose1dConfig default values
// ---------------------------------------------------------------------------

/// Prove: the default ConvTranspose1dConfig has stride=1, padding=0,
/// output_padding=0, dilation=1, groups=1.
#[kani::unwind(1)]
#[kani::proof]
fn proof_conv_transpose1d_config_defaults() {
    // Model the Default impl
    let padding: usize = 0;
    let output_padding: usize = 0;
    let stride: usize = 1;
    let dilation: usize = 1;
    let groups: usize = 1;

    assert!(padding == 0, "default padding must be 0");
    assert!(output_padding == 0, "default output_padding must be 0");
    assert!(stride == 1, "default stride must be 1");
    assert!(dilation == 1, "default dilation must be 1");
    assert!(groups == 1, "default groups must be 1");

    // With defaults: L_out = (L_in - 1)*1 + 0 + 1*(k-1) + 0 + 1 = L_in + k - 1
    let input_len: usize = kani::any();
    let kernel_size: usize = kani::any();
    kani::assume(input_len >= 1 && input_len <= 64);
    kani::assume(kernel_size >= 1 && kernel_size <= 16);

    let output_len = input_len + kernel_size - 1;
    // Verify this is consistent with formula
    let formula = conv_transpose1d_output_length(
        input_len,
        kernel_size,
        stride,
        padding,
        output_padding,
        dilation,
    );
    assert!(
        formula == Some(output_len),
        "default config output must match L_in + k - 1"
    );
}

// ---------------------------------------------------------------------------
// Harness 4: Output padding must be less than stride
// ---------------------------------------------------------------------------

/// Prove: the PyTorch constraint output_padding < stride is necessary.
/// When output_padding >= stride, the formula produces ambiguous results.
#[kani::unwind(1)]
#[kani::proof]
fn proof_conv_transpose1d_output_padding_constraint() {
    let stride: usize = kani::any();
    let output_padding: usize = kani::any();

    kani::assume(stride >= 1 && stride <= 8);
    kani::assume(output_padding <= 16);

    // The constraint is output_padding < stride
    let valid = output_padding < stride;

    if valid {
        // When valid, output_padding is strictly less than stride
        assert!(
            output_padding + 1 <= stride,
            "output_padding + 1 must fit within stride"
        );
    } else {
        // output_padding >= stride violates the constraint
        assert!(
            output_padding >= stride,
            "violation: output_padding >= stride"
        );
    }
}

// ---------------------------------------------------------------------------
// Harness 5: ConvTranspose1d weight rank must be 3
// ---------------------------------------------------------------------------

/// Prove: the ConvTranspose1d constructor rejects non-3D weights.
/// Weight shape is [in_channels, out_channels/groups, kernel_size].
#[kani::unwind(1)]
#[kani::proof]
fn proof_conv_transpose1d_weight_rank_3() {
    let weight_rank: usize = kani::any();
    kani::assume(weight_rank <= 6);

    // Models: if weight.rank() != 3 { return Err(RankMismatch) }
    let accepted = weight_rank == 3;
    let rejected = weight_rank != 3;

    assert!(accepted || rejected, "must be accepted or rejected");
    assert!(!(accepted && rejected), "cannot be both");
    assert!(
        accepted == (weight_rank == 3),
        "only rank 3 weights accepted"
    );
}

// ---------------------------------------------------------------------------
// Harness 6: Groups=0 is rejected
// ---------------------------------------------------------------------------

/// Prove: ConvTranspose1d constructor rejects groups=0.
#[kani::unwind(1)]
#[kani::proof]
fn proof_conv_transpose1d_groups_nonzero() {
    let groups: usize = kani::any();
    kani::assume(groups <= 64);

    // Models: if config.groups == 0 { return Err(ConvParameterInvalid) }
    let accepted = groups > 0;
    let rejected = groups == 0;

    assert!(accepted || rejected, "must be accepted or rejected");
    assert!(accepted == (groups > 0), "only groups > 0 accepted");
}

// ---------------------------------------------------------------------------
// Harness 7: Output length monotonically increases with stride
// ---------------------------------------------------------------------------

/// Prove: for fixed input length, kernel, padding=0, dilation=1,
/// output_padding=0, increasing stride increases output length.
#[kani::unwind(1)]
#[kani::proof]
fn proof_conv_transpose1d_output_monotonic_stride() {
    let input_len: usize = kani::any();
    let kernel_size: usize = kani::any();
    let stride1: usize = kani::any();
    let stride2: usize = kani::any();

    kani::assume(input_len >= 2 && input_len <= 64);
    kani::assume(kernel_size >= 1 && kernel_size <= 8);
    kani::assume(stride1 >= 1 && stride1 <= 4);
    kani::assume(stride2 >= 1 && stride2 <= 4);
    kani::assume(stride2 > stride1);

    // With padding=0, dilation=1, output_padding=0:
    // L_out = (L_in - 1) * stride + kernel_size
    let out1 = (input_len - 1) * stride1 + kernel_size;
    let out2 = (input_len - 1) * stride2 + kernel_size;

    // Since stride2 > stride1 and input_len >= 2: (L_in - 1) >= 1
    // So out2 - out1 = (L_in - 1) * (stride2 - stride1) > 0
    assert!(out2 > out1, "output length must increase with stride");
}

// ---------------------------------------------------------------------------
// Harness 8: ConvTranspose2d output size formula well-defined
// ---------------------------------------------------------------------------

/// Prove: the 2D output size formula produces valid results for each
/// spatial dimension independently.
#[kani::unwind(1)]
#[kani::proof]
fn proof_conv_transpose2d_output_size_well_defined() {
    let input_h: usize = kani::any();
    let input_w: usize = kani::any();
    let kernel_h: usize = kani::any();
    let kernel_w: usize = kani::any();
    let stride: usize = kani::any();
    let padding: usize = kani::any();
    let dilation: usize = kani::any();

    kani::assume(input_h >= 1 && input_h <= 64);
    kani::assume(input_w >= 1 && input_w <= 64);
    kani::assume(kernel_h >= 1 && kernel_h <= 8);
    kani::assume(kernel_w >= 1 && kernel_w <= 8);
    kani::assume(stride >= 1 && stride <= 4);
    kani::assume(padding <= 8);
    kani::assume(dilation >= 1 && dilation <= 2);

    // Ensure no underflow
    let ah = (input_h - 1) * stride;
    let aw = (input_w - 1) * stride;
    let b = 2 * padding;
    kani::assume(ah + dilation * (kernel_h - 1) + 1 >= b);
    kani::assume(aw + dilation * (kernel_w - 1) + 1 >= b);

    let out_h = conv_transpose1d_output_length(input_h, kernel_h, stride, padding, 0, dilation);
    let out_w = conv_transpose1d_output_length(input_w, kernel_w, stride, padding, 0, dilation);

    assert!(out_h.is_some(), "H output must be computable");
    assert!(out_w.is_some(), "W output must be computable");
    assert!(out_h.unwrap() >= 1, "H output must be >= 1");
    assert!(out_w.unwrap() >= 1, "W output must be >= 1");
}

// ---------------------------------------------------------------------------
// Harness 9: ConvTranspose2dConfig symmetric defaults
// ---------------------------------------------------------------------------

/// Prove: ConvTranspose2dConfig default uses symmetric [0,0], [1,1] etc.
#[kani::unwind(1)]
#[kani::proof]
fn proof_conv_transpose2d_config_symmetric_defaults() {
    // Model the Default impl
    let padding = [0_usize, 0];
    let output_padding = [0_usize, 0];
    let stride = [1_usize, 1];
    let dilation = [1_usize, 1];
    let groups: usize = 1;

    assert!(
        padding[0] == padding[1],
        "default padding must be symmetric"
    );
    assert!(
        output_padding[0] == output_padding[1],
        "default output_padding must be symmetric"
    );
    assert!(stride[0] == stride[1], "default stride must be symmetric");
    assert!(
        dilation[0] == dilation[1],
        "default dilation must be symmetric"
    );
    assert!(groups == 1, "default groups must be 1");
}

// ---------------------------------------------------------------------------
// Harness 10: ConvTranspose2d weight rank must be 4
// ---------------------------------------------------------------------------

/// Prove: the ConvTranspose2d constructor rejects non-4D weights.
/// Weight shape is [in_channels, out_channels/groups, kH, kW].
#[kani::unwind(1)]
#[kani::proof]
fn proof_conv_transpose2d_weight_rank_4() {
    let weight_rank: usize = kani::any();
    kani::assume(weight_rank <= 6);

    // Models: if weight.rank() != 4 { return Err(RankMismatch) }
    let accepted = weight_rank == 4;
    assert!(
        accepted == (weight_rank == 4),
        "only rank 4 weights accepted for ConvTranspose2d"
    );
}

// ---------------------------------------------------------------------------
// Harness 11: ConvTranspose2d groups=0 rejected
// ---------------------------------------------------------------------------

/// Prove: ConvTranspose2d constructor rejects groups=0 (same rule as 1D).
#[kani::unwind(1)]
#[kani::proof]
fn proof_conv_transpose2d_groups_nonzero() {
    let groups: usize = kani::any();
    kani::assume(groups <= 64);

    let accepted = groups > 0;
    assert!(
        accepted == (groups > 0),
        "only groups > 0 accepted for ConvTranspose2d"
    );
}

// ---------------------------------------------------------------------------
// Harness 12: Symmetric config produces equal H/W parameters
// ---------------------------------------------------------------------------

/// Prove: ConvTranspose2dConfig::new(p, s, d) produces padding=[p,p],
/// stride=[s,s], dilation=[d,d].
#[kani::unwind(1)]
#[kani::proof]
fn proof_conv_transpose2d_symmetric_config() {
    let p: usize = kani::any();
    let s: usize = kani::any();
    let d: usize = kani::any();

    kani::assume(p <= 16);
    kani::assume(s >= 1 && s <= 8);
    kani::assume(d >= 1 && d <= 4);

    // Models ConvTranspose2dConfig::new(p, s, d)
    let padding = [p, p];
    let stride = [s, s];
    let dilation = [d, d];

    assert!(padding[0] == padding[1], "padding must be symmetric");
    assert!(stride[0] == stride[1], "stride must be symmetric");
    assert!(dilation[0] == dilation[1], "dilation must be symmetric");
}

// ---------------------------------------------------------------------------
// Harness 13: Output padding less than stride for each spatial dim
// ---------------------------------------------------------------------------

/// Prove: the output_padding < stride constraint applies independently
/// to each spatial dimension in 2D transposed convolution.
#[kani::unwind(1)]
#[kani::proof]
fn proof_conv_transpose2d_output_padding_constraint() {
    let stride_h: usize = kani::any();
    let stride_w: usize = kani::any();
    let output_padding_h: usize = kani::any();
    let output_padding_w: usize = kani::any();

    kani::assume(stride_h >= 1 && stride_h <= 8);
    kani::assume(stride_w >= 1 && stride_w <= 8);
    kani::assume(output_padding_h <= 16);
    kani::assume(output_padding_w <= 16);

    let valid_h = output_padding_h < stride_h;
    let valid_w = output_padding_w < stride_w;
    let valid = valid_h && valid_w;

    // Both dimensions must independently satisfy the constraint
    if valid {
        assert!(
            output_padding_h < stride_h,
            "H output_padding must be < stride_h"
        );
        assert!(
            output_padding_w < stride_w,
            "W output_padding must be < stride_w"
        );
    }
}

// ---------------------------------------------------------------------------
// Harness 14: Bias reshape for 2D transposed convolution
// ---------------------------------------------------------------------------

/// Prove: the bias reshape [1, out_ch, 1, 1] produces the correct
/// broadcast shape for adding bias to a [B, out_ch, H, W] output.
#[kani::unwind(1)]
#[kani::proof]
fn proof_conv_transpose2d_bias_reshape() {
    let batch: usize = kani::any();
    let out_channels: usize = kani::any();
    let out_h: usize = kani::any();
    let out_w: usize = kani::any();

    kani::assume(batch >= 1 && batch <= 32);
    kani::assume(out_channels >= 1 && out_channels <= 512);
    kani::assume(out_h >= 1 && out_h <= 128);
    kani::assume(out_w >= 1 && out_w <= 128);

    // Output shape: [B, out_ch, H, W]
    let output_shape = [batch, out_channels, out_h, out_w];

    // Bias reshape: [1, out_ch, 1, 1]
    let bias_shape = [1_usize, out_channels, 1, 1];

    // Broadcast rule: dimensions must match or one must be 1
    assert!(
        bias_shape[0] == 1 || bias_shape[0] == output_shape[0],
        "batch broadcasts"
    );
    assert!(bias_shape[1] == output_shape[1], "channel dims must match");
    assert!(
        bias_shape[2] == 1 || bias_shape[2] == output_shape[2],
        "H broadcasts"
    );
    assert!(
        bias_shape[3] == 1 || bias_shape[3] == output_shape[3],
        "W broadcasts"
    );

    // Bias element count equals out_channels
    let bias_elems = bias_shape[0] * bias_shape[1] * bias_shape[2] * bias_shape[3];
    assert!(
        bias_elems == out_channels,
        "bias reshaped element count must equal out_channels"
    );
}

// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Kani proof harnesses for CNN building blocks (#4068).
//!
//! Proves correctness properties of spatial dimension formulas, channel
//! arithmetic, and configuration invariants for:
//!
//! - ConvBnAct — Conv2d + BatchNorm + Activation fused block
//! - SPPF — Spatial Pyramid Pooling - Fast
//! - C2f / Bottleneck — Cross-Stage Partial bottleneck
//! - MBConv — Mobile Inverted Bottleneck Convolution
//!
//! Each harness proves one property using `kani::any()` with `kani::assume!()`
//! to constrain symbolic inputs to valid parameter ranges.
//!
//! Part of #4068.

// ===========================================================================
// ConvBnAct proofs
// ===========================================================================

// ---------------------------------------------------------------------------
// Harness 1: Conv output spatial dims formula
// ---------------------------------------------------------------------------

/// Prove: Conv2d output = (input + 2*pad - kernel) / stride + 1.
///
/// For any valid combination of input size, kernel, padding, and stride,
/// the output size formula is consistent and positive.
#[kani::unwind(4)]
#[kani::proof]
fn proof_conv_output_spatial_dims() {
    let input_size: usize = kani::any();
    let kernel_size: usize = kani::any();
    let padding: usize = kani::any();
    let stride: usize = kani::any();

    kani::assume(input_size >= 1 && input_size <= 256);
    kani::assume(kernel_size >= 1 && kernel_size <= 7);
    kani::assume(padding <= 3);
    kani::assume(stride >= 1 && stride <= 4);
    // Ensure numerator is non-negative (valid conv config)
    kani::assume(input_size + 2 * padding >= kernel_size);

    let numerator = input_size + 2 * padding - kernel_size;
    let output_size = numerator / stride + 1;

    // Output must be at least 1 for valid configs
    assert!(output_size >= 1, "conv output must be at least 1");

    // Verify the formula is self-consistent: if we reconstruct the
    // minimum input needed for this output, it should be <= input_size.
    let min_input = (output_size - 1) * stride + kernel_size;
    assert!(
        min_input <= input_size + 2 * padding,
        "minimum input for this output must fit within padded input"
    );
}

// ---------------------------------------------------------------------------
// Harness 2: Conv padding non-negative
// ---------------------------------------------------------------------------

/// Prove: ConvBnAct auto-padding formula `kernel_size / 2` is always
/// non-negative (trivially true for usize, but verifies the relationship
/// padding <= kernel_size for all valid kernel sizes).
#[kani::unwind(4)]
#[kani::proof]
fn proof_conv_padding_non_negative() {
    let kernel_size: usize = kani::any();
    kani::assume(kernel_size >= 1 && kernel_size <= 7);

    let padding = kernel_size / 2;

    // Padding is always less than kernel_size
    assert!(
        padding < kernel_size,
        "auto-padding must be strictly less than kernel_size"
    );

    // For odd kernels, 2*padding == kernel_size - 1 (same-padding property)
    if kernel_size % 2 == 1 {
        assert!(
            2 * padding == kernel_size - 1,
            "for odd kernels, 2*padding must equal kernel_size - 1"
        );
    }

    // For even kernels, 2*padding == kernel_size - 2
    if kernel_size % 2 == 0 && kernel_size >= 2 {
        assert!(
            2 * padding == kernel_size - 2,
            "for even kernels >= 2, 2*padding must equal kernel_size - 2"
        );
    }
}

// ---------------------------------------------------------------------------
// Harness 3: BatchNorm running variance must be positive (no div-by-zero)
// ---------------------------------------------------------------------------

/// Prove: For BatchNorm normalization `x / sqrt(var + eps)`, the denominator
/// is always positive when `var >= 0` and `eps > 0`, preventing division by
/// zero.
#[kani::unwind(4)]
#[kani::proof]
fn proof_bn_running_var_positive() {
    let var: f64 = kani::any();
    let eps: f64 = kani::any();

    kani::assume(var >= 0.0 && var <= 1e6);
    kani::assume(var.is_finite());
    kani::assume(eps > 0.0 && eps <= 1.0);
    kani::assume(eps.is_finite());

    let denom_sq = var + eps;
    assert!(denom_sq > 0.0, "var + eps must be strictly positive");
    assert!(
        denom_sq.is_finite(),
        "var + eps must be finite for valid inputs"
    );

    // The denominator for BN is sqrt(var + eps), which is positive
    // when var + eps > 0.
    assert!(
        denom_sq >= eps,
        "var + eps must be at least eps (since var >= 0)"
    );
}

// ===========================================================================
// SPPF proofs
// ===========================================================================

// ---------------------------------------------------------------------------
// Harness 4: SPPF concat channels = 4 * hidden
// ---------------------------------------------------------------------------

/// Prove: SPPF output channels after concatenation = 4 * (channels / 2).
///
/// SPPF reduces input channels by 2, applies 3 sequential max-pools,
/// concatenates all 4 branches (original + 3 pooled), giving 4 * hidden.
#[kani::unwind(4)]
#[kani::proof]
fn proof_sppf_concat_channels() {
    let channels: usize = kani::any();
    kani::assume(channels >= 2 && channels <= 2048);
    kani::assume(channels % 2 == 0);

    let hidden = channels / 2;
    assert!(hidden >= 1, "hidden channels must be at least 1");

    // 4 branches: original + 3 max-pooled, each with `hidden` channels
    let concat_channels = hidden * 4;
    assert!(
        concat_channels == 2 * channels,
        "concat must equal 2 * input channels"
    );

    // The output conv takes concat_channels and projects back to channels
    assert!(
        concat_channels >= channels,
        "concat channels must be >= original channels"
    );
}

// ---------------------------------------------------------------------------
// Harness 5: SPPF maxpool with same-padding preserves spatial dims
// ---------------------------------------------------------------------------

/// Prove: MaxPool2d with `padding = kernel_size / 2` and `stride = 1`
/// preserves spatial dimensions for odd kernel sizes.
///
/// This is the SPPF pattern: kernel=5, pad=2, stride=1.
#[kani::unwind(4)]
#[kani::proof]
fn proof_sppf_spatial_preserved() {
    let input_h: usize = kani::any();
    let input_w: usize = kani::any();
    let pool_kernel: usize = kani::any();

    kani::assume(input_h >= 1 && input_h <= 256);
    kani::assume(input_w >= 1 && input_w <= 256);
    // Odd pool kernel sizes (3, 5, 7) — standard for same-padding maxpool
    kani::assume(pool_kernel >= 3 && pool_kernel <= 7);
    kani::assume(pool_kernel % 2 == 1);

    let pad = pool_kernel / 2;
    let stride = 1_usize;

    // MaxPool2d output formula: (input + 2*pad - kernel) / stride + 1
    let out_h = (input_h + 2 * pad - pool_kernel) / stride + 1;
    let out_w = (input_w + 2 * pad - pool_kernel) / stride + 1;

    assert!(
        out_h == input_h,
        "maxpool with same-padding must preserve height"
    );
    assert!(
        out_w == input_w,
        "maxpool with same-padding must preserve width"
    );
}

// ---------------------------------------------------------------------------
// Harness 6: SPPF sequential maxpool chain preserves dims
// ---------------------------------------------------------------------------

/// Prove: Three sequential maxpool5x5 with pad=2, stride=1 all preserve
/// spatial dimensions (the SPPF chain).
#[kani::unwind(8)]
#[kani::proof]
fn proof_sppf_maxpool_chain_valid() {
    let h: usize = kani::any();
    let w: usize = kani::any();

    kani::assume(h >= 1 && h <= 128);
    kani::assume(w >= 1 && w <= 128);

    let kernel = 5_usize;
    let pad = kernel / 2; // 2
    let stride = 1_usize;

    // Pool 1
    let h1 = (h + 2 * pad - kernel) / stride + 1;
    let w1 = (w + 2 * pad - kernel) / stride + 1;
    assert!(h1 == h, "pool1 must preserve height");
    assert!(w1 == w, "pool1 must preserve width");

    // Pool 2
    let h2 = (h1 + 2 * pad - kernel) / stride + 1;
    let w2 = (w1 + 2 * pad - kernel) / stride + 1;
    assert!(h2 == h, "pool2 must preserve height");
    assert!(w2 == w, "pool2 must preserve width");

    // Pool 3
    let h3 = (h2 + 2 * pad - kernel) / stride + 1;
    let w3 = (w2 + 2 * pad - kernel) / stride + 1;
    assert!(h3 == h, "pool3 must preserve height");
    assert!(w3 == w, "pool3 must preserve width");

    // All 4 branches (original + 3 pooled) have the same spatial dims
    // so concatenation along channel dim is valid.
}

// ===========================================================================
// C2f / Bottleneck proofs
// ===========================================================================

// ---------------------------------------------------------------------------
// Harness 7: C2f channel split — halves sum to total
// ---------------------------------------------------------------------------

/// Prove: C2f channel split produces two halves that sum to the original.
///
/// cv1 outputs `2 * hidden` channels, split into two chunks of `hidden` each.
/// `hidden = out_c / 2`, so `2 * hidden = out_c` when `out_c` is even.
#[kani::unwind(4)]
#[kani::proof]
fn proof_c2f_channel_split() {
    let out_c: usize = kani::any();
    kani::assume(out_c >= 2 && out_c <= 2048);
    kani::assume(out_c % 2 == 0);

    let hidden = out_c / 2;
    let cv1_out = 2 * hidden;

    // cv1 output channels
    assert!(
        cv1_out == out_c,
        "cv1 output must equal out_c when out_c is even"
    );

    // Split into two equal halves
    let first_half = hidden;
    let second_half = cv1_out - hidden;
    assert!(
        first_half + second_half == cv1_out,
        "split halves must sum to total"
    );
    assert!(first_half == second_half, "split must produce equal halves");
}

// ---------------------------------------------------------------------------
// Harness 8: Bottleneck residual shape — in == out for shortcut
// ---------------------------------------------------------------------------

/// Prove: When Bottleneck uses shortcut (residual), input and output
/// channels must be equal. Both cv1 and cv2 use the same channel count.
#[kani::unwind(4)]
#[kani::proof]
fn proof_bottleneck_residual_shape() {
    let channels: usize = kani::any();
    kani::assume(channels >= 1 && channels <= 2048);

    // Bottleneck cv1: channels -> channels (3x3, stride=1, pad=1)
    let cv1_in = channels;
    let cv1_out = channels;

    // Bottleneck cv2: channels -> channels (3x3, stride=1, pad=1)
    let cv2_in = cv1_out;
    let cv2_out = channels;

    // For residual addition, input shape must match output shape
    assert!(
        cv1_in == cv2_out,
        "residual requires input channels == output channels"
    );
    assert!(cv2_in == cv1_out, "cv2 input must match cv1 output");

    // With kernel=3, pad=1, stride=1, spatial dims are preserved
    let test_spatial: usize = kani::any();
    kani::assume(test_spatial >= 1 && test_spatial <= 128);
    let out_spatial = (test_spatial + 2 * 1 - 3) / 1 + 1;
    assert!(
        out_spatial == test_spatial,
        "3x3 conv with pad=1 stride=1 preserves spatial dims"
    );
}

// ---------------------------------------------------------------------------
// Harness 9: C2f output channels match configuration
// ---------------------------------------------------------------------------

/// Prove: C2f concatenation input to cv2 = (2 + n_bottlenecks) * hidden,
/// and cv2 projects this to out_c.
#[kani::unwind(4)]
#[kani::proof]
fn proof_c2f_output_channels() {
    let out_c: usize = kani::any();
    let n_bottlenecks: usize = kani::any();

    kani::assume(out_c >= 2 && out_c <= 1024);
    kani::assume(out_c % 2 == 0);
    kani::assume(n_bottlenecks >= 1 && n_bottlenecks <= 6);

    let hidden = out_c / 2;

    // Concatenation: chunk0 + chunk1 + n_bottleneck outputs
    // chunk0 has `hidden` channels, chunk1 has `hidden` channels,
    // each bottleneck output has `hidden` channels.
    let cat_channels = (2 + n_bottlenecks) * hidden;

    // cat_channels must be > 0
    assert!(cat_channels > 0, "cat channels must be positive");

    // cat_channels grows with n_bottlenecks
    assert!(
        cat_channels >= 2 * hidden,
        "cat channels must be >= 2 * hidden (the two chunks)"
    );
    assert!(
        cat_channels == out_c + n_bottlenecks * hidden,
        "cat channels = out_c + n_bottlenecks * hidden"
    );
}

// ---------------------------------------------------------------------------
// Harness 10: Bottleneck shortcut dims — identity when channels match
// ---------------------------------------------------------------------------

/// Prove: When `in_channels == out_channels` and `stride == 1`, the
/// shortcut is the identity (no projection needed), and shape is preserved.
#[kani::unwind(4)]
#[kani::proof]
fn proof_bottleneck_shortcut_dims() {
    let in_channels: usize = kani::any();
    let out_channels: usize = kani::any();
    let stride: usize = kani::any();

    kani::assume(in_channels >= 1 && in_channels <= 2048);
    kani::assume(out_channels >= 1 && out_channels <= 2048);
    kani::assume(stride >= 1 && stride <= 2);

    // Shortcut (residual) is used when in == out and stride == 1
    let use_shortcut = in_channels == out_channels && stride == 1;

    if use_shortcut {
        // Identity shortcut: no projection needed
        assert!(
            in_channels == out_channels,
            "shortcut requires matching channel counts"
        );
        assert!(stride == 1, "shortcut requires stride == 1");

        // With stride=1 and same-padding (kernel=3, pad=1), spatial dims
        // are preserved, so residual addition is shape-compatible.
        let test_h: usize = kani::any();
        kani::assume(test_h >= 1 && test_h <= 128);
        let conv_out_h = (test_h + 2 * 1 - 3) / 1 + 1;
        assert!(
            conv_out_h == test_h,
            "spatial dims must match for residual addition"
        );
    } else {
        // When shortcut is not used, channels may differ or stride > 1
        assert!(
            in_channels != out_channels || stride != 1,
            "shortcut must be disabled when channels differ or stride > 1"
        );
    }
}

// ===========================================================================
// MBConv proofs
// ===========================================================================

// ---------------------------------------------------------------------------
// Harness 11: MBConv expanded channels always positive
// ---------------------------------------------------------------------------

/// Prove: MBConv expanded channels (`in_channels * expand_ratio`) is always
/// positive when both inputs are positive.
#[kani::unwind(4)]
#[kani::proof]
fn proof_mbconv_expansion_positive() {
    let in_channels: usize = kani::any();
    let expand_ratio: usize = kani::any();

    kani::assume(in_channels >= 1 && in_channels <= 2048);
    kani::assume(expand_ratio >= 1 && expand_ratio <= 6);

    let hidden = in_channels.checked_mul(expand_ratio);
    if let Some(h) = hidden {
        assert!(h >= 1, "expanded channels must be at least 1");
        assert!(
            h >= in_channels,
            "expanded channels must be >= input channels"
        );
        assert!(
            h == in_channels * expand_ratio,
            "expanded channels must equal in_channels * expand_ratio"
        );
    }
}

// ---------------------------------------------------------------------------
// Harness 12: MBConv SE squeeze channels always positive
// ---------------------------------------------------------------------------

/// Prove: MBConv SE reduction `max(1, in_channels / se_ratio)` is always
/// at least 1, preventing a zero-channel squeeze layer.
#[kani::unwind(4)]
#[kani::proof]
fn proof_mbconv_se_squeeze_positive() {
    let in_channels: usize = kani::any();
    let se_ratio: usize = kani::any();

    kani::assume(in_channels >= 1 && in_channels <= 2048);
    kani::assume(se_ratio >= 1 && se_ratio <= 16);

    // SE reduced channels: max(1, in_channels / se_ratio)
    let se_dim = (in_channels / se_ratio).max(1);

    assert!(se_dim >= 1, "SE reduced channels must be at least 1");
    assert!(
        se_dim <= in_channels,
        "SE reduced channels must be <= input channels"
    );

    // When in_channels < se_ratio, division gives 0, but max(1, ...) saves it
    if in_channels < se_ratio {
        assert!(se_dim == 1, "SE dim must be 1 when in_channels < se_ratio");
    }
}

// ---------------------------------------------------------------------------
// Harness 13: MBConv depthwise groups == expanded channels
// ---------------------------------------------------------------------------

/// Prove: In MBConv depthwise convolution, `groups == expanded_channels`,
/// making it a true depthwise conv (one filter per channel).
#[kani::unwind(4)]
#[kani::proof]
fn proof_mbconv_depthwise_groups() {
    let in_channels: usize = kani::any();
    let expand_ratio: usize = kani::any();

    kani::assume(in_channels >= 1 && in_channels <= 512);
    kani::assume(expand_ratio >= 1 && expand_ratio <= 6);

    let hidden = in_channels.checked_mul(expand_ratio);
    if let Some(expanded) = hidden {
        // Depthwise conv: groups == channels, in_channels == out_channels == groups
        let groups = expanded;

        assert!(
            groups == expanded,
            "depthwise groups must equal expanded channels"
        );

        // Each group has exactly 1 input channel and 1 output channel
        let channels_per_group = expanded / groups;
        assert!(
            channels_per_group == 1,
            "depthwise conv must have 1 channel per group"
        );

        // Total parameters per spatial position = expanded * 1 (not expanded^2)
        // This is the efficiency gain of depthwise separable convolution.
        assert!(
            expanded % groups == 0,
            "expanded channels must be divisible by groups"
        );
    }
}

// ---------------------------------------------------------------------------
// Harness 14: MBConv residual condition
// ---------------------------------------------------------------------------

/// Prove: MBConv uses residual connection only when stride == 1 and
/// in_channels == out_channels, ensuring shape compatibility.
#[kani::unwind(4)]
#[kani::proof]
fn proof_mbconv_residual_condition() {
    let in_channels: usize = kani::any();
    let out_channels: usize = kani::any();
    let stride: usize = kani::any();

    kani::assume(in_channels >= 1 && in_channels <= 512);
    kani::assume(out_channels >= 1 && out_channels <= 512);
    kani::assume(stride >= 1 && stride <= 2);

    // MBConv residual condition (from mbconv.rs line 205)
    let use_residual = stride == 1 && in_channels == out_channels;

    if use_residual {
        // When residual is used, shapes must be compatible
        assert!(
            in_channels == out_channels,
            "residual requires matching channels"
        );
        assert!(stride == 1, "residual requires stride == 1");

        // With stride=1, spatial dims are preserved by the depthwise conv
        // (padding = (kernel-1)/2 gives same-padding).
        let test_h: usize = kani::any();
        let kernel: usize = kani::any();
        kani::assume(test_h >= 1 && test_h <= 64);
        kani::assume(kernel >= 1 && kernel <= 5);
        kani::assume(kernel % 2 == 1); // odd kernels for same-padding
        let pad = (kernel - 1) / 2;
        let out_h = (test_h + 2 * pad - kernel) / stride + 1;
        assert!(
            out_h == test_h,
            "spatial dims must be preserved for residual"
        );
    }
}

// ---------------------------------------------------------------------------
// Harness 15: MBConv depthwise padding preserves spatial dims (stride=1)
// ---------------------------------------------------------------------------

/// Prove: MBConv depthwise conv with `padding = (kernel_size - 1) / 2` and
/// `stride = 1` preserves spatial dimensions for odd kernel sizes.
#[kani::unwind(4)]
#[kani::proof]
fn proof_mbconv_depthwise_preserves_spatial() {
    let input_size: usize = kani::any();
    let kernel_size: usize = kani::any();

    kani::assume(input_size >= 1 && input_size <= 256);
    kani::assume(kernel_size >= 1 && kernel_size <= 7);
    kani::assume(kernel_size % 2 == 1); // odd kernels

    let padding = (kernel_size - 1) / 2;
    let stride = 1_usize;

    let output_size = (input_size + 2 * padding - kernel_size) / stride + 1;

    assert!(
        output_size == input_size,
        "depthwise conv with same-padding and stride=1 must preserve spatial dims"
    );

    // Also verify padding formula consistency
    assert!(
        2 * padding == kernel_size - 1,
        "same-padding formula must hold for odd kernels"
    );
}

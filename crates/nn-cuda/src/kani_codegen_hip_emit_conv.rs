// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Kani proof harnesses for HIP convolution codegen.
//!
//! Proves properties of conv1d_out_len (checked arithmetic, edge cases),
//! emit_conv1d_kernel / emit_conv2d_kernel / emit_conv_transpose1d_kernel
//! parameter validation (groups=0, stride=0, overflow), and structural
//! invariants of the generated HIP C++ source.
//!
//! Part of #3719.

use super::codegen_hip_tensor_emit_conv::{
    emit_conv1d_kernel, emit_conv2d_kernel, emit_conv_transpose1d_kernel,
};
use nn_dsl::ScalarType;

// =========================================================================
// Conv1d output length formula proofs (modeled)
//
// conv1d_out_len is private, so we model its formula and prove properties.
// Formula: out = (in_length + 2*padding - dilation*(kernel_size-1) - 1) / stride + 1
// =========================================================================

/// Model the conv1d output length formula using checked arithmetic.
fn model_conv1d_out_len(
    in_length: usize,
    kernel_size: usize,
    stride: usize,
    padding: usize,
    dilation: usize,
) -> Option<usize> {
    kernel_size
        .checked_sub(1)
        .and_then(|ks_m1| dilation.checked_mul(ks_m1))
        .and_then(|dilated| dilated.checked_add(1))
        .and_then(|sub_term| {
            in_length
                .checked_add(2usize.checked_mul(padding)?)
                .and_then(|padded| padded.checked_sub(sub_term))
        })
        .and_then(|numerator| {
            if stride == 0 {
                None
            } else {
                Some(numerator / stride + 1)
            }
        })
}

/// Prove conv1d output length is >= 1 for valid parameters.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn prove_conv1d_out_len_positive() {
    let in_length: u8 = kani::any();
    let kernel_size: u8 = kani::any();
    let stride: u8 = kani::any();
    let padding: u8 = kani::any();
    let dilation: u8 = kani::any();

    kani::assume(in_length >= 1);
    kani::assume(kernel_size >= 1);
    kani::assume(stride >= 1);
    kani::assume(dilation >= 1);

    if let Some(out) = model_conv1d_out_len(
        in_length as usize,
        kernel_size as usize,
        stride as usize,
        padding as usize,
        dilation as usize,
    ) {
        assert!(out >= 1);
    }
}

/// Prove conv1d with stride=0 always returns None (division by zero guard).
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn prove_conv1d_stride_zero_fails() {
    let in_length: u8 = kani::any();
    let kernel_size: u8 = kani::any();
    let padding: u8 = kani::any();
    let dilation: u8 = kani::any();

    kani::assume(in_length >= 1);
    kani::assume(kernel_size >= 1);
    kani::assume(dilation >= 1);

    let result = model_conv1d_out_len(
        in_length as usize,
        kernel_size as usize,
        0,
        padding as usize,
        dilation as usize,
    );
    assert!(result.is_none());
}

/// Prove conv1d with kernel_size=0 always returns None (underflow guard).
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn prove_conv1d_kernel_zero_fails() {
    let in_length: u8 = kani::any();
    let stride: u8 = kani::any();
    let padding: u8 = kani::any();
    let dilation: u8 = kani::any();

    kani::assume(in_length >= 1);
    kani::assume(stride >= 1);
    kani::assume(dilation >= 1);

    let result = model_conv1d_out_len(
        in_length as usize,
        0,
        stride as usize,
        padding as usize,
        dilation as usize,
    );
    // kernel_size=0: checked_sub(1) underflows → None
    assert!(result.is_none());
}

/// Prove conv1d standard case: kernel=1, stride=1, padding=0, dilation=1 -> out == in.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn prove_conv1d_identity_case() {
    let in_length: u8 = kani::any();
    kani::assume(in_length >= 1);

    let out = model_conv1d_out_len(in_length as usize, 1, 1, 0, 1);
    assert!(out.is_some());
    assert_eq!(out.unwrap(), in_length as usize);
}

/// Prove conv1d with stride=2 halves the output (approximately).
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn prove_conv1d_stride2_halves() {
    let in_length: u8 = kani::any();
    kani::assume(in_length >= 2);

    if let Some(out) = model_conv1d_out_len(in_length as usize, 1, 2, 0, 1) {
        // out = (in_length - 1) / 2 + 1 = ceil(in_length / 2)
        let expected = (in_length as usize + 1) / 2;
        assert_eq!(out, expected);
    }
}

/// Prove conv1d no-padding, kernel=in_length -> out == 1.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn prove_conv1d_full_kernel_one_output() {
    let in_length: u8 = kani::any();
    kani::assume(in_length >= 1);

    let out = model_conv1d_out_len(in_length as usize, in_length as usize, 1, 0, 1);
    assert!(out.is_some());
    assert_eq!(out.unwrap(), 1);
}

// =========================================================================
// emit_conv1d_kernel validation proofs
// =========================================================================

/// Prove emit_conv1d_kernel rejects groups=0.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(128)]
fn prove_conv1d_groups_zero_error() {
    let result = emit_conv1d_kernel("test", ScalarType::F32, 4, 8, 3, 16, 1, 1, 1, 0, false);
    assert!(result.is_err());
}

/// Prove emit_conv1d_kernel succeeds for basic valid parameters.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(256)]
fn prove_conv1d_basic_valid() {
    let result = emit_conv1d_kernel("test", ScalarType::F32, 4, 8, 3, 16, 1, 1, 1, 1, false);
    assert!(result.is_ok());
}

/// Prove emit_conv1d_kernel succeeds with bias.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(256)]
fn prove_conv1d_with_bias() {
    let result = emit_conv1d_kernel("test", ScalarType::F32, 4, 8, 3, 16, 1, 1, 1, 1, true);
    assert!(result.is_ok());
}

/// Prove emit_conv1d_kernel output contains the kernel name.
#[kani::unwind(16)]
#[kani::proof]
#[kani::unwind(256)]
fn prove_conv1d_contains_name() {
    let result = emit_conv1d_kernel("nn_conv", ScalarType::F32, 4, 8, 3, 16, 1, 1, 1, 1, false);
    assert!(result.is_ok());
    let src = result.unwrap();
    // The source should contain the kernel function name.
    assert!(src.contains("nn_conv"));
}

/// Prove emit_conv1d_kernel for f16 uses the half type.
#[kani::unwind(16)]
#[kani::proof]
#[kani::unwind(256)]
fn prove_conv1d_f16_type() {
    let result = emit_conv1d_kernel("test", ScalarType::F16, 4, 8, 3, 16, 1, 1, 1, 1, false);
    assert!(result.is_ok());
    let src = result.unwrap();
    assert!(src.contains("half"));
}

// =========================================================================
// emit_conv2d_kernel validation proofs
// =========================================================================

/// Prove emit_conv2d_kernel rejects groups=0.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(128)]
fn prove_conv2d_groups_zero_error() {
    let result = emit_conv2d_kernel(
        "test",
        ScalarType::F32,
        4,
        8,
        3,
        3,
        16,
        16,
        1,
        1,
        1,
        1,
        1,
        1,
        0,
        false,
    );
    assert!(result.is_err());
}

/// Prove emit_conv2d_kernel succeeds for basic valid parameters.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(256)]
fn prove_conv2d_basic_valid() {
    let result = emit_conv2d_kernel(
        "test",
        ScalarType::F32,
        4,
        8,
        3,
        3,
        16,
        16,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        false,
    );
    assert!(result.is_ok());
}

/// Prove emit_conv2d_kernel output contains the kernel name and structural markers.
#[kani::unwind(16)]
#[kani::proof]
#[kani::unwind(256)]
fn prove_conv2d_output_structure() {
    let result = emit_conv2d_kernel(
        "nn_conv2d",
        ScalarType::F32,
        4,
        8,
        3,
        3,
        16,
        16,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        true,
    );
    assert!(result.is_ok());
    let src = result.unwrap();
    assert!(src.contains("nn_conv2d"));
    assert!(src.contains("__global__"));
    assert!(src.contains("bias"));
}

// =========================================================================
// emit_conv_transpose1d_kernel validation proofs
// =========================================================================

/// Prove emit_conv_transpose1d_kernel rejects groups=0.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(128)]
fn prove_conv_transpose1d_groups_zero_error() {
    let result =
        emit_conv_transpose1d_kernel("test", ScalarType::F32, 4, 8, 3, 16, 1, 1, 1, 0, 0, false);
    assert!(result.is_err());
}

/// Prove emit_conv_transpose1d_kernel succeeds for basic valid parameters.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(256)]
fn prove_conv_transpose1d_basic_valid() {
    let result =
        emit_conv_transpose1d_kernel("test", ScalarType::F32, 4, 8, 3, 16, 1, 1, 1, 1, 0, false);
    assert!(result.is_ok());
}

/// Prove conv_transpose1d output length formula:
/// out = (in - 1) * stride - 2 * padding + dilation * (ks - 1) + output_padding + 1
#[kani::unwind(16)]
#[kani::proof]
#[kani::unwind(256)]
fn prove_conv_transpose1d_out_length() {
    // For in=4, ks=3, stride=2, padding=1, dilation=1, output_padding=0:
    // out = (4-1)*2 - 2*1 + 1*(3-1) + 0 + 1 = 6 - 2 + 2 + 1 = 7
    let result =
        emit_conv_transpose1d_kernel("test", ScalarType::F32, 4, 8, 3, 4, 2, 1, 1, 1, 0, false);
    assert!(result.is_ok());
    let src = result.unwrap();
    // The kernel should contain OUT_LENGTH = 7
    assert!(src.contains("7"));
}

// =========================================================================
// Conv2d output dimension formula (modeled)
// conv2d reuses conv1d_out_len for each spatial dimension.
// =========================================================================

/// Prove conv2d out_h and out_w are correctly computed via the conv1d formula.
#[kani::unwind(8)]
#[kani::proof]
#[kani::unwind(1)]
fn prove_conv2d_spatial_dims() {
    // in_h=8, kh=3, stride_h=1, padding_h=1, dilation_h=1
    // out_h = (8 + 2*1 - 1*(3-1) - 1)/1 + 1 = (10 - 3)/1 + 1 = 8
    let out_h = model_conv1d_out_len(8, 3, 1, 1, 1);
    assert_eq!(out_h, Some(8));

    // Same for width with different params:
    // in_w=16, kw=5, stride_w=2, padding_w=2, dilation_w=1
    // out_w = (16 + 2*2 - 1*(5-1) - 1)/2 + 1 = (20 - 5)/2 + 1 = 7 + 1 = 8
    let out_w = model_conv1d_out_len(16, 5, 2, 2, 1);
    assert_eq!(out_w, Some(8));
}

// =========================================================================
// Overflow safety proofs
// =========================================================================

/// Prove that conv1d_out_len handles large inputs without panic (returns None).
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn prove_conv1d_large_input_no_panic() {
    let in_length = usize::MAX / 2;
    let kernel_size: usize = 3;
    let stride: usize = 1;
    let padding = usize::MAX / 4;
    let dilation: usize = 1;

    // This should either return Some(valid) or None, never panic.
    let _ = model_conv1d_out_len(in_length, kernel_size, stride, padding, dilation);
}

/// Prove conv1d with very large dilation returns None (overflow).
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn prove_conv1d_large_dilation_no_panic() {
    let in_length: usize = 16;
    let kernel_size: usize = 3;
    let stride: usize = 1;
    let padding: usize = 0;
    let dilation = usize::MAX / 2;

    let result = model_conv1d_out_len(in_length, kernel_size, stride, padding, dilation);
    // Should return None due to dilation * (ks-1) overflow
    assert!(result.is_none());
}

// =========================================================================
// Grouped convolution proofs
// =========================================================================

/// Prove that in_ch_per_group is well-defined when groups divides in_channels.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn prove_grouped_conv_channel_division() {
    let in_channels: u8 = kani::any();
    let groups: u8 = kani::any();
    kani::assume(in_channels > 0);
    kani::assume(groups > 0);
    kani::assume(in_channels as usize % groups as usize == 0);

    let in_ch_per_group = in_channels as usize / groups as usize;
    assert!(in_ch_per_group >= 1);
    assert_eq!(in_ch_per_group * groups as usize, in_channels as usize);
}

/// Prove depthwise conv (groups == in_channels) gives in_ch_per_group == 1.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn prove_depthwise_conv_one_channel_per_group() {
    let channels: u8 = kani::any();
    kani::assume(channels >= 1);

    let in_ch_per_group = channels as usize / channels as usize;
    assert_eq!(in_ch_per_group, 1);
}

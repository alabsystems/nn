// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Tests for HIP convolution op emission (Conv1d, Conv2d, ConvTranspose1d).

use crate::codegen_hip_tensor_emit_conv::*;
use nn_dsl::ScalarType;

// --- Conv1d tests ---

#[test]
fn test_conv1d_basic_no_bias() {
    let src = emit_conv1d_kernel(
        "conv1d_basic",
        ScalarType::F32,
        3,  // in_channels
        16, // out_channels
        3,  // kernel_size
        64, // in_length
        1,  // stride
        1,  // padding
        1,  // dilation
        1,  // groups
        false,
    )
    .unwrap();
    assert!(src.contains("extern \"C\" __global__ void conv1d_basic"));
    assert!(src.contains("IN_CH_PER_GROUP = 3"));
    assert!(src.contains("OUT_CHANNELS = 16"));
    assert!(src.contains("KERNEL_SIZE = 3"));
    assert!(!src.contains("bias"));
}

#[test]
fn test_conv1d_with_bias() {
    let src = emit_conv1d_kernel(
        "conv1d_bias",
        ScalarType::F32,
        8,
        32,
        5,
        128,
        2,
        2,
        1,
        1,
        true,
    )
    .unwrap();
    assert!(src.contains("bias"));
    assert!(src.contains("sum += bias[oc_local]"));
}

#[test]
fn test_conv1d_grouped() {
    let src = emit_conv1d_kernel(
        "conv1d_grp",
        ScalarType::F32,
        16, // in_channels
        32, // out_channels
        3,  // kernel_size
        64, // in_length
        1,  // stride
        1,  // padding
        1,  // dilation
        4,  // groups
        false,
    )
    .unwrap();
    assert!(src.contains("GROUPS = 4"));
    assert!(src.contains("IN_CH_PER_GROUP = 4"));
}

#[test]
fn test_conv1d_f16_accumulation() {
    let src =
        emit_conv1d_kernel("conv1d_f16", ScalarType::F16, 4, 8, 3, 32, 1, 1, 1, 1, true).unwrap();
    assert!(src.contains("float sum"));
    assert!(src.contains("(float)input"));
    assert!(src.contains("(float)weight"));
    assert!(src.contains("(half)sum"));
    assert!(src.contains("(float)bias"));
}

#[test]
fn test_conv1d_bf16_type() {
    let src = emit_conv1d_kernel(
        "conv1d_bf16",
        ScalarType::BF16,
        4,
        8,
        3,
        32,
        1,
        1,
        1,
        1,
        false,
    )
    .unwrap();
    assert!(src.contains("hip_bfloat16"));
    assert!(src.contains("(hip_bfloat16)sum"));
}

#[test]
fn test_conv1d_dilated() {
    let src = emit_conv1d_kernel(
        "conv1d_dil",
        ScalarType::F32,
        4,
        8,
        3,
        64,
        1,
        2,
        2,
        1,
        false,
    )
    .unwrap();
    assert!(src.contains("DILATION = 2"));
    assert!(src.contains("k * DILATION"));
}

#[test]
fn test_conv1d_groups_zero_error() {
    let result = emit_conv1d_kernel("bad", ScalarType::F32, 4, 8, 3, 32, 1, 0, 1, 0, false);
    assert!(result.is_err());
}

#[test]
fn test_conv1d_batched_oc_local() {
    let src = emit_conv1d_kernel(
        "conv1d_batch",
        ScalarType::F32,
        4,
        8,
        3,
        32,
        1,
        1,
        1,
        1,
        true,
    )
    .unwrap();
    // Critical invariant: oc_local for weight/bias, batch_ic_offset for input.
    assert!(src.contains("oc_local = oc % OUT_CHANNELS"));
    assert!(src.contains("batch_ic_offset = (oc / OUT_CHANNELS) * IN_CHANNELS"));
}

// --- Conv2d tests ---

#[test]
fn test_conv2d_basic() {
    let src = emit_conv2d_kernel(
        "conv2d_basic",
        ScalarType::F32,
        3,  // in_channels
        16, // out_channels
        3,
        3, // kernel_h, kernel_w
        32,
        32, // in_height, in_width
        1,
        1, // stride_h, stride_w
        1,
        1, // padding_h, padding_w
        1,
        1, // dilation_h, dilation_w
        1, // groups
        true,
    )
    .unwrap();
    assert!(src.contains("extern \"C\" __global__ void conv2d_basic"));
    assert!(src.contains("IN_HEIGHT = 32"));
    assert!(src.contains("IN_WIDTH = 32"));
    assert!(src.contains("KERNEL_H = 3"));
    assert!(src.contains("KERNEL_W = 3"));
    assert!(src.contains("bias[oc_local]"));
}

#[test]
fn test_conv2d_groups_zero_error() {
    let result = emit_conv2d_kernel(
        "bad",
        ScalarType::F32,
        4,
        8,
        3,
        3,
        16,
        16,
        1,
        1,
        0,
        0,
        1,
        1,
        0,
        false,
    );
    assert!(result.is_err());
}

// --- ConvTranspose1d tests ---

#[test]
fn test_conv_transpose1d_basic() {
    let src = emit_conv_transpose1d_kernel(
        "convt1d_basic",
        ScalarType::F32,
        8,  // in_channels
        16, // out_channels
        4,  // kernel_size
        32, // in_length
        2,  // stride
        1,  // padding
        1,  // dilation
        1,  // groups
        0,  // output_padding
        true,
    )
    .unwrap();
    assert!(src.contains("extern \"C\" __global__ void convt1d_basic"));
    assert!(src.contains("OUT_CH_PER_GROUP"));
    assert!(src.contains("oc_in_group"));
    assert!(src.contains("(ot_pad - dk) % STRIDE == 0"));
    assert!(src.contains("bias[oc_local]"));
}

#[test]
fn test_conv_transpose1d_no_bias() {
    let src = emit_conv_transpose1d_kernel(
        "convt1d_plain",
        ScalarType::F32,
        4,
        8,
        3,
        16,
        1,
        0,
        1,
        1,
        0,
        false,
    )
    .unwrap();
    assert!(!src.contains("bias"));
}

#[test]
fn test_conv_transpose1d_groups_zero_error() {
    let result =
        emit_conv_transpose1d_kernel("bad", ScalarType::F32, 4, 8, 3, 16, 1, 0, 1, 0, 0, false);
    assert!(result.is_err());
}

#[test]
fn test_conv_transpose1d_f16_casts() {
    let src = emit_conv_transpose1d_kernel(
        "convt1d_f16",
        ScalarType::F16,
        4,
        8,
        3,
        16,
        2,
        1,
        1,
        1,
        0,
        true,
    )
    .unwrap();
    assert!(src.contains("float sum"));
    assert!(src.contains("(float)input"));
    assert!(src.contains("(half)sum"));
}

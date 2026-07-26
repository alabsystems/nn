#![allow(deprecated)]
// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Tests for Conv1d, ConvTranspose1d, and Pad1d operations on DynTensor.

use crate::dyn_tensor::test_helpers::cpu;
use crate::{DynTensor, TensorError};

// -- Conv1d tests ---------------------------------------------------------

#[test]
fn test_conv1d_basic() {
    // [1, 1, 5] * [1, 1, 3] stride=1 pad=0 → [1, 1, 3]
    let x = DynTensor::new(&[1.0, 2.0, 3.0, 4.0, 5.0], &[1, 1, 5], &cpu()).unwrap();
    let k = DynTensor::new(&[1.0, 1.0, 1.0], &[1, 1, 3], &cpu()).unwrap();
    let y = x.conv1d(&k, 0, 1, 1, 1).unwrap();
    assert_eq!(y.dims(), &[1, 1, 3]);
    let v = y.to_flat_vec::<f32>().unwrap();
    // [1+2+3, 2+3+4, 3+4+5] = [6, 9, 12]
    assert_eq!(v, vec![6.0, 9.0, 12.0]);
}

#[test]
fn test_conv1d_with_padding() {
    // [1, 1, 3] * [1, 1, 3] stride=1 pad=1 → [1, 1, 3]
    let x = DynTensor::new(&[1.0, 2.0, 3.0], &[1, 1, 3], &cpu()).unwrap();
    let k = DynTensor::new(&[1.0, 1.0, 1.0], &[1, 1, 3], &cpu()).unwrap();
    let y = x.conv1d(&k, 1, 1, 1, 1).unwrap();
    assert_eq!(y.dims(), &[1, 1, 3]);
    let v = y.to_flat_vec::<f32>().unwrap();
    // [0+1+2, 1+2+3, 2+3+0] = [3, 6, 5]
    assert_eq!(v, vec![3.0, 6.0, 5.0]);
}

#[test]
fn test_conv1d_stride2() {
    // [1, 1, 6] * [1, 1, 3] stride=2 pad=0 → [1, 1, 2]
    let x = DynTensor::new(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[1, 1, 6], &cpu()).unwrap();
    let k = DynTensor::new(&[1.0, 1.0, 1.0], &[1, 1, 3], &cpu()).unwrap();
    let y = x.conv1d(&k, 0, 2, 1, 1).unwrap();
    assert_eq!(y.dims(), &[1, 1, 2]);
    let v = y.to_flat_vec::<f32>().unwrap();
    // pos 0: 1+2+3=6, pos 1: 3+4+5=12
    assert_eq!(v, vec![6.0, 12.0]);
}

#[test]
fn test_conv1d_dilation() {
    // [1, 1, 5] * [1, 1, 2] stride=1 dilation=2 pad=0 → [1, 1, 3]
    // effective kernel positions: [0, 2] for each output pos
    let x = DynTensor::new(&[1.0, 2.0, 3.0, 4.0, 5.0], &[1, 1, 5], &cpu()).unwrap();
    let k = DynTensor::new(&[1.0, 1.0], &[1, 1, 2], &cpu()).unwrap();
    let y = x.conv1d(&k, 0, 1, 2, 1).unwrap();
    assert_eq!(y.dims(), &[1, 1, 3]);
    let v = y.to_flat_vec::<f32>().unwrap();
    // pos 0: x[0]+x[2]=1+3=4, pos 1: x[1]+x[3]=2+4=6, pos 2: x[2]+x[4]=3+5=8
    assert_eq!(v, vec![4.0, 6.0, 8.0]);
}

#[test]
fn test_conv1d_multi_channel() {
    // [1, 2, 3] * [3, 2, 1] stride=1 pad=0 → [1, 3, 3]
    // 2 input channels, 3 output channels, kernel_size=1
    let x = DynTensor::new(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[1, 2, 3], &cpu()).unwrap();
    // kernel [out=3, in=2, k=1]
    let k = DynTensor::new(&[1.0, 0.0, 0.0, 1.0, 1.0, 1.0], &[3, 2, 1], &cpu()).unwrap();
    let y = x.conv1d(&k, 0, 1, 1, 1).unwrap();
    assert_eq!(y.dims(), &[1, 3, 3]);
    let v = y.to_flat_vec::<f32>().unwrap();
    // OC0: 1*ch0 + 0*ch1 = [1, 2, 3]
    // OC1: 0*ch0 + 1*ch1 = [4, 5, 6]
    // OC2: 1*ch0 + 1*ch1 = [5, 7, 9]
    assert_eq!(v, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 5.0, 7.0, 9.0]);
}

#[test]
fn test_conv1d_groups() {
    // [1, 4, 3] * [4, 2, 1] groups=2 → [1, 4, 3]
    // Group 0: in_ch [0,1] → out_ch [0,1], Group 1: in_ch [2,3] → out_ch [2,3]
    let x = DynTensor::new(
        &[
            1.0, 2.0, 3.0, // ch0
            4.0, 5.0, 6.0, // ch1
            7.0, 8.0, 9.0, // ch2
            10.0, 11.0, 12.0, // ch3
        ],
        &[1, 4, 3],
        &cpu(),
    )
    .unwrap();
    // kernel [out=4, in_per_group=2, k=1], identity-like within each group
    let k = DynTensor::new(
        &[
            1.0, 0.0, // OC0 = 1*ch0 + 0*ch1
            0.0, 1.0, // OC1 = 0*ch0 + 1*ch1
            1.0, 0.0, // OC2 = 1*ch2 + 0*ch3
            0.0, 1.0, // OC3 = 0*ch2 + 1*ch3
        ],
        &[4, 2, 1],
        &cpu(),
    )
    .unwrap();
    let y = x.conv1d(&k, 0, 1, 1, 2).unwrap();
    assert_eq!(y.dims(), &[1, 4, 3]);
    let v = y.to_flat_vec::<f32>().unwrap();
    assert_eq!(
        v,
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0]
    );
}

#[test]
fn test_conv1d_batch() {
    // [2, 1, 3] * [1, 1, 3] → [2, 1, 1]
    let x = DynTensor::new(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 1, 3], &cpu()).unwrap();
    let k = DynTensor::new(&[1.0, 1.0, 1.0], &[1, 1, 3], &cpu()).unwrap();
    let y = x.conv1d(&k, 0, 1, 1, 1).unwrap();
    assert_eq!(y.dims(), &[2, 1, 1]);
    let v = y.to_flat_vec::<f32>().unwrap();
    assert_eq!(v, vec![6.0, 15.0]);
}

#[test]
fn test_conv1d_shape_mismatch() {
    let x = DynTensor::new(&[1.0, 2.0, 3.0], &[1, 1, 3], &cpu()).unwrap();
    // Wrong in_channels: kernel expects 2 but input has 1
    let k = DynTensor::new(&[1.0, 1.0, 1.0, 1.0, 1.0, 1.0], &[1, 2, 3], &cpu()).unwrap();
    assert!(x.conv1d(&k, 0, 1, 1, 1).is_err());
}

// -- ConvTranspose1d tests ------------------------------------------------

#[test]
fn test_conv_transpose1d_basic() {
    // [1, 1, 3] * [1, 1, 3] stride=1 pad=0 → [1, 1, 5]
    let x = DynTensor::new(&[1.0, 2.0, 3.0], &[1, 1, 3], &cpu()).unwrap();
    let k = DynTensor::new(&[1.0, 1.0, 1.0], &[1, 1, 3], &cpu()).unwrap();
    let y = x.conv_transpose1d(&k, 0, 0, 1, 1, 1).unwrap();
    assert_eq!(y.dims(), &[1, 1, 5]);
    let v = y.to_flat_vec::<f32>().unwrap();
    // output[0]=1*1=1, output[1]=1*1+2*1=3, output[2]=1+2+3=6,
    // output[3]=2+3=5, output[4]=3=3
    assert_eq!(v, vec![1.0, 3.0, 6.0, 5.0, 3.0]);
}

#[test]
fn test_conv_transpose1d_stride2() {
    // [1, 1, 3] * [1, 1, 3] stride=2 pad=0 → [1, 1, 7]
    let x = DynTensor::new(&[1.0, 2.0, 3.0], &[1, 1, 3], &cpu()).unwrap();
    let k = DynTensor::new(&[1.0, 1.0, 1.0], &[1, 1, 3], &cpu()).unwrap();
    let y = x.conv_transpose1d(&k, 0, 0, 2, 1, 1).unwrap();
    assert_eq!(y.dims(), &[1, 1, 7]);
    let v = y.to_flat_vec::<f32>().unwrap();
    // stride=2: input[0] at pos 0, input[1] at pos 2, input[2] at pos 4
    // output[0]=1, output[1]=1, output[2]=1+2=3, output[3]=2, output[4]=2+3=5, output[5]=3, output[6]=3
    assert_eq!(v, vec![1.0, 1.0, 3.0, 2.0, 5.0, 3.0, 3.0]);
}

#[test]
fn test_conv_transpose1d_with_padding() {
    // [1, 1, 3] * [1, 1, 3] stride=1 pad=1 → [1, 1, 3]
    let x = DynTensor::new(&[1.0, 2.0, 3.0], &[1, 1, 3], &cpu()).unwrap();
    let k = DynTensor::new(&[1.0, 1.0, 1.0], &[1, 1, 3], &cpu()).unwrap();
    let y = x.conv_transpose1d(&k, 1, 0, 1, 1, 1).unwrap();
    assert_eq!(y.dims(), &[1, 1, 3]);
    let v = y.to_flat_vec::<f32>().unwrap();
    // Full output would be [1, 3, 6, 5, 3], padding=1 crops first and last
    assert_eq!(v, vec![3.0, 6.0, 5.0]);
}

#[test]
fn test_conv_transpose1d_multi_channel() {
    // [1, 2, 2] * [2, 1, 1] stride=1 pad=0 → [1, 1, 2]
    // kernel [in=2, out/g=1, k=1]: sum the two channels
    let x = DynTensor::new(&[1.0, 2.0, 3.0, 4.0], &[1, 2, 2], &cpu()).unwrap();
    let k = DynTensor::new(&[1.0, 1.0], &[2, 1, 1], &cpu()).unwrap();
    let y = x.conv_transpose1d(&k, 0, 0, 1, 1, 1).unwrap();
    assert_eq!(y.dims(), &[1, 1, 2]);
    let v = y.to_flat_vec::<f32>().unwrap();
    // OC0: ic0*1 + ic1*1 = [1+3, 2+4] = [4, 6]
    assert_eq!(v, vec![4.0, 6.0]);
}

// Pad1d basic tests extracted to pad_tests.rs
#[path = "pad_tests.rs"]
mod pad_tests;

// Validation guard, boundary, and regression tests extracted to conv_guard_tests.rs
#[path = "conv_guard_tests.rs"]
mod guard_tests;

// -- Direct conv_out_len helper tests (defense-in-depth) ----------------------

#[test]
fn test_conv1d_out_len_zero_stride() {
    let err = super::conv1d_out_len(10, 3, 0, 0, 1).unwrap_err();
    assert!(
        matches!(
            err,
            TensorError::ConvParameterInvalid {
                param: "stride",
                ..
            }
        ),
        "expected ConvParameterInvalid for stride, got: {err}"
    );
}

#[test]
fn test_conv1d_out_len_zero_dilation() {
    let err = super::conv1d_out_len(10, 3, 0, 1, 0).unwrap_err();
    assert!(
        matches!(
            err,
            TensorError::ConvParameterInvalid {
                param: "dilation",
                ..
            }
        ),
        "expected ConvParameterInvalid for dilation, got: {err}"
    );
}

#[test]
fn test_conv2d_out_len_zero_stride() {
    let err = super::conv2d::conv2d_out_len(10, 3, 0, 0, 1).unwrap_err();
    assert!(
        matches!(
            err,
            TensorError::ConvParameterInvalid {
                param: "stride",
                ..
            }
        ),
        "expected ConvParameterInvalid for stride, got: {err}"
    );
}

#[test]
fn test_conv2d_out_len_zero_dilation() {
    let err = super::conv2d::conv2d_out_len(10, 3, 0, 1, 0).unwrap_err();
    assert!(
        matches!(
            err,
            TensorError::ConvParameterInvalid {
                param: "dilation",
                ..
            }
        ),
        "expected ConvParameterInvalid for dilation, got: {err}"
    );
}

// -- ReflectionPad1d tests ------------------------------------------------

#[test]
fn test_reflection_pad1d_left_only() {
    // [a, b, c, d, e] with (1, 0) → [b, a, b, c, d, e]
    let x = DynTensor::new(&[1.0, 2.0, 3.0, 4.0, 5.0], &[1, 1, 5], &cpu()).unwrap();
    let y = x.reflection_pad1d(1, 0).unwrap();
    assert_eq!(y.dims(), &[1, 1, 6]);
    let v = y.to_flat_vec::<f32>().unwrap();
    assert_eq!(v, vec![2.0, 1.0, 2.0, 3.0, 4.0, 5.0]);
}

#[test]
fn test_reflection_pad1d_both_sides() {
    // [a, b, c, d, e] with (2, 1) → [c, b, a, b, c, d, e, d]
    let x = DynTensor::new(&[1.0, 2.0, 3.0, 4.0, 5.0], &[1, 1, 5], &cpu()).unwrap();
    let y = x.reflection_pad1d(2, 1).unwrap();
    assert_eq!(y.dims(), &[1, 1, 8]);
    let v = y.to_flat_vec::<f32>().unwrap();
    assert_eq!(v, vec![3.0, 2.0, 1.0, 2.0, 3.0, 4.0, 5.0, 4.0]);
}

#[test]
fn test_reflection_pad1d_noop() {
    let x = DynTensor::new(&[1.0, 2.0, 3.0], &[3], &cpu()).unwrap();
    let y = x.reflection_pad1d(0, 0).unwrap();
    assert_eq!(y.to_flat_vec::<f32>().unwrap(), vec![1.0, 2.0, 3.0]);
}

#[test]
fn test_reflection_pad1d_too_large() {
    // pad_left >= dim_len should fail
    let x = DynTensor::new(&[1.0, 2.0, 3.0], &[1, 1, 3], &cpu()).unwrap();
    let err = x.reflection_pad1d(3, 0).unwrap_err();
    assert!(
        matches!(err, TensorError::InvalidShape(..)),
        "expected InvalidShape, got: {err}"
    );
}

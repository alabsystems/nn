#![allow(deprecated)]
// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Validation guard, boundary, regression, and edge case tests for conv ops.
//! Extracted from `tests.rs` to keep files under 500 lines.

use crate::dyn_tensor::test_helpers::cpu;
use crate::{DynTensor, TensorError};

// -- Validation guard tests (added by P10 for new stride/dilation/kernel guards) --

#[test]
fn test_conv1d_zero_stride_returns_error() {
    let x = DynTensor::new(&[1.0, 2.0, 3.0], &[1, 1, 3], &cpu()).unwrap();
    let k = DynTensor::new(&[1.0], &[1, 1, 1], &cpu()).unwrap();
    let err = x.conv1d(&k, 0, 0, 1, 1).unwrap_err();
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
fn test_conv1d_zero_dilation_returns_error() {
    let x = DynTensor::new(&[1.0, 2.0, 3.0], &[1, 1, 3], &cpu()).unwrap();
    let k = DynTensor::new(&[1.0], &[1, 1, 1], &cpu()).unwrap();
    let err = x.conv1d(&k, 0, 1, 0, 1).unwrap_err();
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
fn test_conv_transpose1d_zero_stride_returns_error() {
    let x = DynTensor::new(&[1.0, 2.0, 3.0], &[1, 1, 3], &cpu()).unwrap();
    let k = DynTensor::new(&[1.0], &[1, 1, 1], &cpu()).unwrap();
    let err = x.conv_transpose1d(&k, 0, 0, 0, 1, 1).unwrap_err();
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
fn test_conv_transpose1d_zero_dilation_returns_error() {
    let x = DynTensor::new(&[1.0, 2.0, 3.0], &[1, 1, 3], &cpu()).unwrap();
    let k = DynTensor::new(&[1.0], &[1, 1, 1], &cpu()).unwrap();
    let err = x.conv_transpose1d(&k, 0, 0, 1, 0, 1).unwrap_err();
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

// -- Regression: conv_transpose1d underflow on large padding (P10 audit Finding 3) --
// Fixed: conv_transpose1d_out_len now returns Result and rejects 2*padding > positive.
#[test]
fn test_conv_transpose1d_large_padding_returns_error() {
    // in_len=1, k_size=1, stride=1, dilation=1, output_padding=0, padding=5
    // positive = (1-1)*1 + 1*(1-1) + 0 + 1 = 1
    // 2*padding = 10 > 1 → returns Err (previously usize underflow)
    let x = DynTensor::new(&[1.0], &[1, 1, 1], &cpu()).unwrap();
    let k = DynTensor::new(&[1.0], &[1, 1, 1], &cpu()).unwrap();
    let err = x.conv_transpose1d(&k, 5, 0, 1, 1, 1).unwrap_err();
    assert!(
        err.to_string().contains("2*padding"),
        "expected padding underflow error, got: {err}"
    );
}

// -- Regression: conv_transpose1d_out_len zero-length output (#1243 Kani counterexample) --
// Fixed: changed `>` to `>=` in the padding guard to reject zero-length output.
#[test]
fn test_conv_transpose1d_zero_output_len_returns_error() {
    // Kani counterexample: input_len=1, kernel_size=2, padding=1, output_padding=0,
    // stride=1, dilation=1.
    // positive = (1-1)*1 + 1*(2-1) + 0 + 1 = 2
    // 2*padding = 2*1 = 2
    // out = 2 - 2 = 0 → must return Err (zero-length output is invalid)
    let x = DynTensor::new(&[1.0], &[1, 1, 1], &cpu()).unwrap();
    let k = DynTensor::new(&[1.0, 2.0], &[1, 1, 2], &cpu()).unwrap();
    let err = x.conv_transpose1d(&k, 1, 0, 1, 1, 1).unwrap_err();
    assert!(
        err.to_string().contains("2*padding"),
        "expected zero-length output error, got: {err}"
    );
}

// -- Missing guard coverage (P10 audit Finding 4) --

#[test]
fn test_conv1d_zero_groups_returns_error() {
    let x = DynTensor::new(&[1.0, 2.0, 3.0], &[1, 1, 3], &cpu()).unwrap();
    let k = DynTensor::new(&[1.0], &[1, 1, 1], &cpu()).unwrap();
    let err = x.conv1d(&k, 0, 1, 1, 0).unwrap_err();
    assert!(
        matches!(
            err,
            TensorError::ConvParameterInvalid {
                param: "groups",
                ..
            }
        ),
        "expected ConvParameterInvalid for groups, got: {err}"
    );
}

#[test]
fn test_conv_transpose1d_zero_groups_returns_error() {
    let x = DynTensor::new(&[1.0, 2.0, 3.0], &[1, 1, 3], &cpu()).unwrap();
    let k = DynTensor::new(&[1.0], &[1, 1, 1], &cpu()).unwrap();
    let err = x.conv_transpose1d(&k, 0, 0, 1, 1, 0).unwrap_err();
    assert!(
        matches!(
            err,
            TensorError::ConvParameterInvalid {
                param: "groups",
                ..
            }
        ),
        "expected ConvParameterInvalid for groups, got: {err}"
    );
}

// -- Boundary condition regression tests (#979 audit) -------------------------

#[test]
fn test_conv1d_zero_kernel_size_returns_error() {
    // kernel_size=0 caused usize underflow in conv1d_out_len: (0 - 1) * dilation
    let x = DynTensor::new(&[1.0, 2.0, 3.0], &[1, 1, 3], &cpu()).unwrap();
    let k = DynTensor::from_vec(vec![], &[1, 1, 0], &cpu()).unwrap();
    let err = x.conv1d(&k, 0, 1, 1, 1).unwrap_err();
    assert!(
        matches!(
            err,
            TensorError::ConvParameterInvalid {
                param: "kernel_size",
                ..
            }
        ),
        "expected ConvParameterInvalid for kernel_size, got: {err}"
    );
}

#[test]
fn test_conv_transpose1d_zero_input_len_returns_error() {
    // input_len=0 caused usize underflow: (0 - 1) * stride
    let x = DynTensor::from_vec(vec![], &[1, 1, 0], &cpu()).unwrap();
    let k = DynTensor::new(&[1.0], &[1, 1, 1], &cpu()).unwrap();
    let err = x.conv_transpose1d(&k, 0, 0, 1, 1, 1).unwrap_err();
    assert!(
        matches!(
            err,
            TensorError::ConvParameterInvalid {
                param: "input_len",
                ..
            }
        ),
        "expected ConvParameterInvalid for input_len, got: {err}"
    );
}

#[test]
fn test_conv_transpose1d_zero_kernel_size_returns_error() {
    // kernel_size=0 caused usize underflow: dilation * (0 - 1)
    let x = DynTensor::new(&[1.0], &[1, 1, 1], &cpu()).unwrap();
    let k = DynTensor::from_vec(vec![], &[1, 1, 0], &cpu()).unwrap();
    let err = x.conv_transpose1d(&k, 0, 0, 1, 1, 1).unwrap_err();
    assert!(
        matches!(
            err,
            TensorError::ConvParameterInvalid {
                param: "kernel_size",
                ..
            }
        ),
        "expected ConvParameterInvalid for kernel_size, got: {err}"
    );
}

#[test]
fn test_conv1d_non_contiguous_input() {
    // Transpose produces a non-contiguous tensor. conv1d should handle it
    // by making it contiguous internally rather than returning an error.
    // Shape: [1, 4, 3] -> transpose(1,2) -> [1, 3, 4] (non-contiguous)
    // After transpose, channels are [1,4,7,10], [2,5,8,11], [3,6,9,12].
    let x = DynTensor::from_vec(
        vec![
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
        ],
        &[1, 4, 3],
        &cpu(),
    )
    .unwrap();
    let x_t = x.transpose(1, 2).unwrap(); // [1, 3, 4], non-contiguous
    assert_eq!(x_t.dims(), &[1, 3, 4]);

    let k = DynTensor::from_vec(vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0], &[1, 3, 2], &cpu()).unwrap();
    let result = x_t.conv1d(&k, 0, 1, 1, 1);
    assert!(
        result.is_ok(),
        "conv1d should handle non-contiguous input, got: {:?}",
        result.err()
    );
    let out = result.unwrap();
    assert_eq!(out.dims(), &[1, 1, 3]);
    // Verify output values, not just shape:
    // out[0] = (1+4)+(2+5)+(3+6) = 21, out[1] = (4+7)+(5+8)+(6+9) = 39,
    // out[2] = (7+10)+(8+11)+(9+12) = 57
    let vals = out.to_flat_vec::<f32>().unwrap();
    assert_eq!(vals, vec![21.0, 39.0, 57.0]);
}

#[test]
fn test_conv_transpose1d_non_contiguous_input() {
    // conv_transpose1d must also handle non-contiguous inputs.
    // Input [1, 2, 3] -> transpose(1,2) -> [1, 3, 2]: ch0=[1,4], ch1=[2,5], ch2=[3,6].
    let x = DynTensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[1, 2, 3], &cpu()).unwrap();
    let x_t = x.transpose(1, 2).unwrap(); // [1, 3, 2], non-contiguous
    assert_eq!(x_t.dims(), &[1, 3, 2]);

    // Kernel [3, 1, 2]: 3 in_ch, 1 out_ch_per_group, kernel_size=2. groups=3.
    let k = DynTensor::from_vec(vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0], &[3, 1, 2], &cpu()).unwrap();
    let result = x_t.conv_transpose1d(&k, 0, 0, 1, 1, 3);
    assert!(
        result.is_ok(),
        "conv_transpose1d should handle non-contiguous input, got: {:?}",
        result.err()
    );
    let out = result.unwrap();
    assert_eq!(out.dims(), &[1, 3, 3]);
    // Per-group all-ones kernel: ch0=[1,5,4], ch1=[2,7,5], ch2=[3,9,6].
    let vals = out.to_flat_vec::<f32>().unwrap();
    assert_eq!(vals, vec![1.0, 5.0, 4.0, 2.0, 7.0, 5.0, 3.0, 9.0, 6.0]);
}

// test_pad1d_non_contiguous_input extracted to dyn_tensor_pad_tests.rs

// -- P1 proof_coverage: groups divisibility guards ----------------------------

#[test]
fn test_conv1d_in_channels_not_divisible_by_groups() {
    let x = DynTensor::new(
        &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
        &[1, 3, 3],
        &cpu(),
    )
    .unwrap();
    let k = DynTensor::new(&[1.0, 1.0], &[2, 1, 1], &cpu()).unwrap();
    let err = x.conv1d(&k, 0, 1, 1, 2).unwrap_err();
    assert!(
        matches!(err, TensorError::ConvParameterInvalid { reason, .. } if reason.contains("divide")),
        "expected ConvParameterInvalid for in_ch divisibility, got: {err}"
    );
}

#[test]
fn test_conv1d_out_channels_not_divisible_by_groups() {
    let x = DynTensor::new(&[1.0; 12], &[1, 4, 3], &cpu()).unwrap();
    let k = DynTensor::new(&[1.0; 6], &[3, 2, 1], &cpu()).unwrap();
    let err = x.conv1d(&k, 0, 1, 1, 2).unwrap_err();
    assert!(
        matches!(err, TensorError::ConvParameterInvalid { reason, .. } if reason.contains("divide")),
        "expected ConvParameterInvalid for out_ch divisibility, got: {err}"
    );
}

#[test]
fn test_conv_transpose1d_in_channels_not_divisible_by_groups() {
    let x = DynTensor::new(
        &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
        &[1, 3, 3],
        &cpu(),
    )
    .unwrap();
    let k = DynTensor::new(&[1.0; 3], &[3, 1, 1], &cpu()).unwrap();
    let err = x.conv_transpose1d(&k, 0, 0, 1, 1, 2).unwrap_err();
    assert!(
        matches!(err, TensorError::ConvParameterInvalid { reason, .. } if reason.contains("divide")),
        "expected ConvParameterInvalid for in_ch divisibility, got: {err}"
    );
}

// test_pad1d_rank0_returns_error extracted to dyn_tensor_pad_tests.rs

// -- P1 proof_coverage: NaN/Inf propagation through conv ----------------------

#[test]
fn test_conv1d_nan_in_input_propagates() {
    let x = DynTensor::new(&[f32::NAN, 1.0, 2.0], &[1, 1, 3], &cpu()).unwrap();
    let k = DynTensor::new(&[1.0, 1.0], &[1, 1, 2], &cpu()).unwrap();
    let y = x.conv1d(&k, 0, 1, 1, 1).unwrap();
    let v = y.to_flat_vec::<f32>().unwrap();
    assert!(v[0].is_nan(), "NaN should propagate through conv1d");
    assert_eq!(v[1], 3.0);
}

#[test]
fn test_conv1d_nan_in_kernel_propagates() {
    let x = DynTensor::new(&[1.0, 2.0, 3.0], &[1, 1, 3], &cpu()).unwrap();
    let k = DynTensor::new(&[f32::NAN, 1.0], &[1, 1, 2], &cpu()).unwrap();
    let y = x.conv1d(&k, 0, 1, 1, 1).unwrap();
    let v = y.to_flat_vec::<f32>().unwrap();
    assert!(v[0].is_nan(), "NaN kernel should propagate to output[0]");
    assert!(v[1].is_nan(), "NaN kernel should propagate to output[1]");
}

#[test]
fn test_conv1d_inf_accumulation() {
    let x = DynTensor::new(&[f32::INFINITY, 1.0, 2.0], &[1, 1, 3], &cpu()).unwrap();
    let k = DynTensor::new(&[1.0, 1.0], &[1, 1, 2], &cpu()).unwrap();
    let y = x.conv1d(&k, 0, 1, 1, 1).unwrap();
    let v = y.to_flat_vec::<f32>().unwrap();
    assert!(v[0].is_infinite(), "Inf should propagate through conv1d");
}

// -- P1 proof_coverage: depthwise conv (groups == in_channels) ----------------

#[test]
fn test_conv1d_depthwise() {
    let x = DynTensor::new(
        &[
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
        ],
        &[1, 3, 4],
        &cpu(),
    )
    .unwrap();
    let k = DynTensor::new(&[1.0, 1.0, 1.0, 1.0, 1.0, 1.0], &[3, 1, 2], &cpu()).unwrap();
    let y = x.conv1d(&k, 0, 1, 1, 3).unwrap();
    assert_eq!(y.dims(), &[1, 3, 3]);
    let v = y.to_flat_vec::<f32>().unwrap();
    assert_eq!(v, vec![3.0, 5.0, 7.0, 11.0, 13.0, 15.0, 19.0, 21.0, 23.0]);
}

// -- Regression: output_padding >= stride must be rejected (PyTorch constraint) --

#[test]
fn test_conv_transpose1d_output_padding_ge_stride_rejected() {
    // PyTorch: "output_padding must be smaller than either stride or dilation"
    let x = DynTensor::new(&[1.0, 2.0, 3.0], &[1, 1, 3], &cpu()).unwrap();
    let k = DynTensor::new(&[1.0, 1.0], &[1, 1, 2], &cpu()).unwrap();
    // output_padding=2, stride=2 → ambiguous, should fail
    let result = x.conv_transpose1d(&k, 0, 2, 2, 1, 1);
    assert!(
        result.is_err(),
        "output_padding >= stride should be rejected"
    );
    let msg = result.unwrap_err().to_string();
    assert!(
        msg.contains("output_padding") && msg.contains("stride"),
        "error should mention output_padding and stride: {msg}"
    );
}

#[test]
fn test_conv_transpose1d_output_padding_less_than_stride_ok() {
    // output_padding=1, stride=2 → valid
    let x = DynTensor::new(&[1.0, 2.0, 3.0], &[1, 1, 3], &cpu()).unwrap();
    let k = DynTensor::new(&[1.0, 1.0], &[1, 1, 2], &cpu()).unwrap();
    let result = x.conv_transpose1d(&k, 0, 1, 2, 1, 1);
    assert!(result.is_ok(), "output_padding < stride should succeed");
}

// pad_with_zeros tests extracted to dyn_tensor_pad_tests.rs

// -- Checked arithmetic overflow tests (#1309) --------------------------------

#[test]
fn test_conv1d_out_len_effective_kernel_overflow() {
    // (kernel_size - 1) * dilation can overflow with large values.
    // kernel_size=usize::MAX, dilation=2 → (MAX-1)*2 overflows.
    use crate::dyn_tensor::conv::conv1d_out_len;
    let err = conv1d_out_len(10, usize::MAX, 0, 1, 2).unwrap_err();
    assert!(
        err.to_string().contains("overflow"),
        "expected overflow error, got: {err}"
    );
}

#[test]
fn test_conv1d_out_len_padding_overflow() {
    // 2 * padding can overflow with large padding.
    use crate::dyn_tensor::conv::conv1d_out_len;
    let err = conv1d_out_len(10, 3, usize::MAX, 1, 1).unwrap_err();
    assert!(
        err.to_string().contains("overflow"),
        "expected overflow error, got: {err}"
    );
}

#[test]
fn test_conv2d_out_len_effective_kernel_overflow() {
    use crate::dyn_tensor::conv::conv2d::conv2d_out_len;
    let err = conv2d_out_len(10, usize::MAX, 0, 1, 2).unwrap_err();
    assert!(
        err.to_string().contains("overflow"),
        "expected overflow error, got: {err}"
    );
}

#[test]
fn test_pool2d_padding_overflow() {
    use crate::dyn_tensor::conv::pool::pool2d_out_len;
    let err = pool2d_out_len(10, 3, usize::MAX, 1, false).unwrap_err();
    assert!(
        err.to_string().contains("overflow"),
        "expected overflow error, got: {err}"
    );
}

#[test]
fn test_conv_transpose1d_out_len_overflow() {
    // (input_len - 1) * stride overflows with large input_len and stride.
    use crate::dyn_tensor::conv::transpose::conv_transpose1d_out_len;
    let err = conv_transpose1d_out_len(usize::MAX, 1, 0, 0, 2, 1).unwrap_err();
    assert!(
        err.to_string().contains("overflow"),
        "expected overflow error, got: {err}"
    );
}

#[test]
fn test_conv_transpose2d_out_len_overflow() {
    use crate::dyn_tensor::conv::transpose2d::conv_transpose2d_out_len;
    let err = conv_transpose2d_out_len(usize::MAX, 1, 0, 0, 2, 1).unwrap_err();
    assert!(
        err.to_string().contains("overflow"),
        "expected overflow error, got: {err}"
    );
}

#[test]
fn test_conv1d_buffer_allocation_overflow() {
    // checked_buffer_len should catch overflow in batch * out_ch * out_len.
    use crate::dyn_tensor::conv::checked_buffer_len;
    let err = checked_buffer_len(&[usize::MAX, 2, 3], "test").unwrap_err();
    assert!(
        err.to_string().contains("overflow"),
        "expected buffer size overflow error, got: {err}"
    );
}

#[test]
fn test_checked_buffer_len_normal_case() {
    use crate::dyn_tensor::conv::checked_buffer_len;
    let result = checked_buffer_len(&[2, 3, 4], "test").unwrap();
    assert_eq!(result, 24);
}

#[test]
fn test_checked_buffer_len_empty_factors() {
    use crate::dyn_tensor::conv::checked_buffer_len;
    let result = checked_buffer_len(&[], "test").unwrap();
    assert_eq!(result, 1);
}

#![allow(deprecated)]
// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Tests for Conv1d, ConvTranspose1d, and WeightNormConv1d nn layers.

use super::*;
use crate::dyn_tensor::test_helpers::cpu;

#[test]
fn test_conv1d_layer_no_bias() {
    let w = DynTensor::new(&[1.0, 1.0, 1.0], &[1, 1, 3], &cpu()).unwrap();
    let layer = Conv1d::new(w, None, Conv1dConfig::default()).unwrap();
    let x = DynTensor::new(&[1.0, 2.0, 3.0, 4.0, 5.0], &[1, 1, 5], &cpu()).unwrap();
    let y = layer.forward(&x).unwrap();
    assert_eq!(y.dims(), &[1, 1, 3]);
    assert_eq!(y.to_flat_vec::<f32>().unwrap(), vec![6.0, 9.0, 12.0]);
}

#[test]
fn test_conv1d_layer_with_bias() {
    let w = DynTensor::new(&[1.0, 1.0, 1.0], &[1, 1, 3], &cpu()).unwrap();
    let b = DynTensor::from_vec(vec![10.0], &[1], &cpu()).unwrap();
    let layer = Conv1d::new(w, Some(b), Conv1dConfig::default()).unwrap();
    let x = DynTensor::new(&[1.0, 2.0, 3.0, 4.0, 5.0], &[1, 1, 5], &cpu()).unwrap();
    let y = layer.forward(&x).unwrap();
    assert_eq!(y.dims(), &[1, 1, 3]);
    assert_eq!(y.to_flat_vec::<f32>().unwrap(), vec![16.0, 19.0, 22.0]);
}

#[test]
fn test_conv1d_layer_with_stride() {
    let w = DynTensor::new(&[1.0, 1.0, 1.0], &[1, 1, 3], &cpu()).unwrap();
    let cfg = Conv1dConfig {
        stride: 2,
        ..Default::default()
    };
    let layer = Conv1d::new(w, None, cfg).unwrap();
    let x = DynTensor::new(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[1, 1, 6], &cpu()).unwrap();
    let y = layer.forward(&x).unwrap();
    assert_eq!(y.dims(), &[1, 1, 2]);
}

#[test]
fn test_conv1d_via_apply() {
    let w = DynTensor::new(&[1.0, 0.0, 0.0], &[1, 1, 3], &cpu()).unwrap();
    let layer = Conv1d::new(w, None, Conv1dConfig::default()).unwrap();
    let x = DynTensor::new(&[5.0, 6.0, 7.0, 8.0, 9.0], &[1, 1, 5], &cpu()).unwrap();
    let y = x.apply(&layer).unwrap();
    assert_eq!(y.dims(), &[1, 1, 3]);
    assert_eq!(y.to_flat_vec::<f32>().unwrap(), vec![5.0, 6.0, 7.0]);
}

#[test]
fn test_conv_transpose1d_layer_no_bias() {
    let w = DynTensor::new(&[1.0, 1.0, 1.0], &[1, 1, 3], &cpu()).unwrap();
    let layer = ConvTranspose1d::new(w, None, ConvTranspose1dConfig::default()).unwrap();
    let x = DynTensor::new(&[1.0, 2.0, 3.0], &[1, 1, 3], &cpu()).unwrap();
    let y = layer.forward(&x).unwrap();
    assert_eq!(y.dims(), &[1, 1, 5]);
    assert_eq!(
        y.to_flat_vec::<f32>().unwrap(),
        vec![1.0, 3.0, 6.0, 5.0, 3.0]
    );
}

#[test]
fn test_conv_transpose1d_layer_with_bias() {
    let w = DynTensor::new(&[1.0, 1.0, 1.0], &[1, 1, 3], &cpu()).unwrap();
    let b = DynTensor::from_vec(vec![1.0], &[1], &cpu()).unwrap();
    let layer = ConvTranspose1d::new(w, Some(b), ConvTranspose1dConfig::default()).unwrap();
    let x = DynTensor::new(&[1.0, 2.0, 3.0], &[1, 1, 3], &cpu()).unwrap();
    let y = layer.forward(&x).unwrap();
    assert_eq!(y.dims(), &[1, 1, 5]);
    assert_eq!(
        y.to_flat_vec::<f32>().unwrap(),
        vec![2.0, 4.0, 7.0, 6.0, 4.0]
    );
}

#[test]
fn test_conv_transpose1d_stride2() {
    let w = DynTensor::new(&[1.0, 1.0, 1.0], &[1, 1, 3], &cpu()).unwrap();
    let cfg = ConvTranspose1dConfig {
        stride: 2,
        ..Default::default()
    };
    let layer = ConvTranspose1d::new(w, None, cfg).unwrap();
    let x = DynTensor::new(&[1.0, 2.0, 3.0], &[1, 1, 3], &cpu()).unwrap();
    let y = layer.forward(&x).unwrap();
    assert_eq!(y.dims(), &[1, 1, 7]);
}

#[test]
fn test_weight_norm_conv1d_basic() {
    // weight_v: [1, 1, 3] = [3, 4, 0] → ||v|| = 5
    // weight_g: [1, 1, 1] = [10]
    // normalized = 10 * [3, 4, 0] / 5 = [6, 8, 0]
    let v = DynTensor::new(&[3.0, 4.0, 0.0], &[1, 1, 3], &cpu()).unwrap();
    let g = DynTensor::new(&[10.0], &[1, 1, 1], &cpu()).unwrap();
    let layer = WeightNormConv1d::new(v, g, None, Conv1dConfig::default()).unwrap();
    let x = DynTensor::new(&[1.0, 0.0, 0.0, 0.0, 0.0], &[1, 1, 5], &cpu()).unwrap();
    let y = layer.forward(&x).unwrap();
    let data = y.to_flat_vec::<f32>().unwrap();
    // y[0] = 6*1 + 8*0 + 0*0 = 6
    assert!((data[0] - 6.0).abs() < 1e-5, "got {}", data[0]);
}

#[test]
fn test_weight_norm_conv1d_preserves_gain() {
    // Unit vector v: [1, 0, 0] → ||v|| = 1 → normalized = g * v
    let v = DynTensor::new(&[1.0, 0.0, 0.0], &[1, 1, 3], &cpu()).unwrap();
    let g = DynTensor::new(&[5.0], &[1, 1, 1], &cpu()).unwrap();
    let layer = WeightNormConv1d::new(v, g, None, Conv1dConfig::default()).unwrap();
    let x = DynTensor::new(&[2.0, 0.0, 0.0, 0.0, 0.0], &[1, 1, 5], &cpu()).unwrap();
    let y = layer.forward(&x).unwrap();
    let data = y.to_flat_vec::<f32>().unwrap();
    // w = 5 * [1, 0, 0] = [5, 0, 0], y[0] = 5*2 = 10
    assert!((data[0] - 10.0).abs() < 1e-5, "got {}", data[0]);
}

/// Regression test for #1350: ConvTranspose1d with output_padding > 0 and bias
/// must use the unfused fallback path (not the GPU fused path, which drops
/// output_padding). Verifies output shape includes the extra output_padding.
#[test]
fn test_conv_transpose1d_output_padding_with_bias() {
    // stride=2, output_padding=1 disambiguates the output length.
    // output_len = (input_len - 1) * stride - 2*padding + dilation*(kernel_size - 1) + output_padding + 1
    //            = (3 - 1) * 2 - 0 + 1*(3-1) + 1 + 1 = 4 + 2 + 1 + 1 = 8
    let w = DynTensor::new(&[1.0, 1.0, 1.0], &[1, 1, 3], &cpu()).unwrap();
    let b = DynTensor::from_vec(vec![10.0], &[1], &cpu()).unwrap();
    let cfg = ConvTranspose1dConfig::default()
        .with_stride(2)
        .with_output_padding(1);
    let layer = ConvTranspose1d::new(w, Some(b), cfg).unwrap();
    let x = DynTensor::new(&[1.0, 2.0, 3.0], &[1, 1, 3], &cpu()).unwrap();
    let y = layer.forward(&x).unwrap();
    // With output_padding=1, shape should be [1, 1, 8] (not 7).
    assert_eq!(y.dims(), &[1, 1, 8]);
    // Verify bias was applied (all values should have +10).
    let vals = y.to_flat_vec::<f32>().unwrap();
    for (i, v) in vals.iter().enumerate() {
        assert!(
            *v >= 10.0,
            "bias should be applied: val[{i}] = {v}, expected >= 10.0"
        );
    }
}

// =============================================================================
// Constructor validation error-path tests
// =============================================================================

#[test]
fn test_conv1d_new_rejects_2d_weight() {
    let w = DynTensor::new(&[1.0, 2.0, 3.0, 4.0], &[2, 2], &cpu()).unwrap();
    let err = Conv1d::new(w, None, Conv1dConfig::default()).unwrap_err();
    assert!(
        matches!(err, TensorError::RankMismatch { expected: 3, .. }),
        "expected RankMismatch for 2D weight, got: {err:?}"
    );
}

#[test]
fn test_conv1d_new_rejects_1d_weight() {
    let w = DynTensor::new(&[1.0, 2.0, 3.0], &[3], &cpu()).unwrap();
    let err = Conv1d::new(w, None, Conv1dConfig::default()).unwrap_err();
    assert!(
        matches!(err, TensorError::RankMismatch { expected: 3, .. }),
        "expected RankMismatch for 1D weight, got: {err:?}"
    );
}

#[test]
fn test_conv1d_new_rejects_zero_groups() {
    let w = DynTensor::new(&[1.0, 2.0, 3.0], &[1, 1, 3], &cpu()).unwrap();
    let cfg = Conv1dConfig {
        groups: 0,
        ..Default::default()
    };
    let err = Conv1d::new(w, None, cfg).unwrap_err();
    let msg = err.to_string();
    assert!(
        msg.contains("groups"),
        "error should mention groups, got: {msg}"
    );
}

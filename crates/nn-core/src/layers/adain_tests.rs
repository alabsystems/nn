#![allow(deprecated)]
// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Tests for [`AdaIn`].

use super::*;
use crate::layers::instance_norm::{InstanceNorm, InstanceNormPrecision};
use crate::layers::{Linear, Module};
use crate::Device;

fn tensor(data: &[f32], shape: &[usize]) -> DynTensor {
    DynTensor::from_vec(data.to_vec(), shape, &Device::Cpu).expect("valid tensor")
}

#[test]
fn test_adain_forward_style() {
    // Style dim=4, channels=2 → style_linear: [4] → [4] (2*channels)
    let w = tensor(
        &[
            0.1, 0.0, 0.0, 0.0, // row 0
            0.0, 0.1, 0.0, 0.0, // row 1
            0.0, 0.0, 0.1, 0.0, // row 2
            0.0, 0.0, 0.0, 0.1, // row 3
        ],
        &[4, 4],
    );
    let linear = Linear::new(w, None).unwrap();
    let adain = AdaIn::new(linear, 1e-5).unwrap();

    let x = tensor(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[1, 2, 3]);
    let style = tensor(&[0.1, 0.2, 0.3, 0.4], &[1, 4]);
    let y = adain.forward_style(&x, &style).unwrap();
    assert_eq!(y.dims(), &[1, 2, 3]);

    // Output should be finite and non-trivial
    let vals = y.to_flat_vec::<f32>().unwrap();
    for &v in &vals {
        assert!(v.is_finite(), "output must be finite, got {v}");
    }
}

/// AdaIn: zero style projection → gamma=0, beta=0 → output = (1+0)*normed + 0 = normed.
#[test]
fn test_adain_zero_style_is_normed() {
    let w = tensor(&[0.0; 4 * 4], &[4, 4]);
    let linear = Linear::new(w, None).unwrap();
    let adain = AdaIn::new(linear, 1e-5).unwrap();

    let x = tensor(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[1, 2, 3]);
    let style = tensor(&[0.0, 0.0, 0.0, 0.0], &[1, 4]);
    let y = adain.forward_style(&x, &style).unwrap();

    // Compare with plain InstanceNorm
    let norm = InstanceNorm::new(1e-5).unwrap();
    let normed = norm.forward(&x).unwrap();
    let y_vals = y.to_flat_vec::<f32>().unwrap();
    let n_vals = normed.to_flat_vec::<f32>().unwrap();
    for (i, (&yv, &nv)) in y_vals.iter().zip(n_vals.iter()).enumerate() {
        assert!(
            (yv - nv).abs() < 1e-5,
            "zero-style AdaIn[{i}]={yv} != InstanceNorm[{i}]={nv}"
        );
    }
}

/// AdaIn with large style values should still produce finite output.
#[test]
fn test_adain_large_style_finite() {
    let mut w_data = vec![0.0f32; 4 * 4];
    for i in 0..4 {
        w_data[i * 4 + i] = 1.0;
    } // identity
    let w = tensor(&w_data, &[4, 4]);
    let linear = Linear::new(w, None).unwrap();
    let adain = AdaIn::new(linear, 1e-5).unwrap();

    let x = tensor(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[1, 2, 3]);
    let style = tensor(&[100.0, -100.0, 50.0, -50.0], &[1, 4]);
    let y = adain.forward_style(&x, &style).unwrap();
    let vals = y.to_flat_vec::<f32>().unwrap();
    for (i, &v) in vals.iter().enumerate() {
        assert!(
            v.is_finite(),
            "large-style AdaIn: element {i} not finite: {v}"
        );
    }
}

#[test]
fn test_adain_nan_input_returns_error() {
    let weight_data = vec![0.1f32; 8]; // 4 in + 4 out = 2*2 channels
    let w = DynTensor::from_vec(weight_data, &[4, 2], &Device::Cpu).unwrap();
    let style_linear = Linear::new(w, None).unwrap();
    let adain = AdaIn::new(style_linear, 1e-5).unwrap();
    let mut data = vec![1.0f32; 6];
    data[0] = f32::NAN;
    let x = DynTensor::from_vec(data, &[1, 2, 3], &Device::Cpu).unwrap();
    let style = tensor(&[0.5, 0.5], &[1, 2]);
    let result = adain.forward_style(&x, &style);
    assert!(result.is_err(), "NaN input should produce an error");
    let msg = result.unwrap_err().to_string();
    assert!(
        msg.contains("Non-finite") || msg.contains("NaN"),
        "error should mention non-finite: {msg}"
    );
}

/// AdaIn with MatchPyTorchCpu precision passes through correctly.
#[test]
fn test_adain_f32_precision() {
    let w = tensor(&[0.0; 4 * 4], &[4, 4]);
    let linear = Linear::new(w, None).unwrap();
    let adain =
        AdaIn::new_with_precision(linear, 1e-5, InstanceNormPrecision::MatchPyTorchCpu).unwrap();

    let x = tensor(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[1, 2, 3]);
    let style = tensor(&[0.0, 0.0, 0.0, 0.0], &[1, 4]);
    let y = adain.forward_style(&x, &style).unwrap();
    assert_eq!(y.dims(), &[1, 2, 3]);

    let vals = y.to_flat_vec::<f32>().unwrap();
    for &v in &vals {
        assert!(v.is_finite(), "F32 AdaIn output must be finite, got {v}");
    }
}

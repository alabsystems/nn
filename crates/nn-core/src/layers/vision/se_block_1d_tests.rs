#![allow(deprecated)]
// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

use super::*;
use crate::test_utils::cpu;

/// Create a deterministic test tensor with values in [0.1, 1.1).
fn test_tensor(dims: &[usize]) -> DynTensor {
    let numel: usize = dims.iter().product();
    let data: Vec<f32> = (0..numel)
        .map(|i| (i as f32 * 0.0073).sin().abs() + 0.1)
        .collect();
    DynTensor::from_vec(data, dims, &cpu()).expect("test tensor")
}

#[test]
fn test_se1d_shape() {
    let device = cpu();
    let vb = VarBuilder::zeros(crate::DType::F32, &device);
    let se = SqueezeExcitation1d::load(vb.pp("se"), 64, 16).expect("load");
    let x = test_tensor(&[2, 64, 100]);
    let out = se.forward(&x).expect("forward");
    assert_eq!(out.dims(), &[2, 64, 100]);
}

#[test]
fn test_se1d_channel_attention() {
    let device = cpu();
    let vb = VarBuilder::zeros(crate::DType::F32, &device);
    let se = SqueezeExcitation1d::load(vb.pp("se"), 32, 8).expect("load");
    let x = test_tensor(&[1, 32, 50]);
    let out = se.forward(&x).expect("forward");
    // With zero weights, sigmoid(0)=0.5, so output ≈ 0.5 * input
    let x_data = x.to_flat_vec::<f32>().expect("x data");
    let out_data = out.to_flat_vec::<f32>().expect("out data");
    for (xv, ov) in x_data.iter().zip(out_data.iter()) {
        assert!((ov - xv * 0.5).abs() < 1e-5, "expected ~0.5*x, got {ov}");
    }
}

#[test]
fn test_se1d_wrong_rank() {
    let device = cpu();
    let vb = VarBuilder::zeros(crate::DType::F32, &device);
    let se = SqueezeExcitation1d::load(vb.pp("se"), 16, 4).expect("load");
    let x = test_tensor(&[2, 16, 8, 8]);
    let result = se.forward(&x);
    assert!(result.is_err(), "should reject 4D input");
}

#[test]
fn test_se1d_zero_channels_rejected() {
    let fc1 = Linear::new(
        DynTensor::zeros(&[4, 0], crate::DType::F32, &cpu()).expect("w"),
        None,
    )
    .expect("fc1");
    let fc2 = Linear::new(
        DynTensor::zeros(&[0, 4], crate::DType::F32, &cpu()).expect("w"),
        None,
    )
    .expect("fc2");
    let result = SqueezeExcitation1d::new(fc1, fc2, 0);
    assert!(result.is_err(), "should reject 0 channels");
}

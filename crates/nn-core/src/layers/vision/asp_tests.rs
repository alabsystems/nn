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
fn test_asp_shape() {
    let device = cpu();
    let vb = VarBuilder::zeros(crate::DType::F32, &device);
    let asp = AttentiveStatisticsPooling::load(vb.pp("asp"), 64).expect("load");
    let x = test_tensor(&[2, 64, 100]);
    let out = asp.forward(&x).expect("forward");
    assert_eq!(out.dims(), &[2, 128]); // 2 * 64
}

#[test]
fn test_asp_variable_length() {
    let device = cpu();
    let vb = VarBuilder::zeros(crate::DType::F32, &device);
    let asp = AttentiveStatisticsPooling::load(vb.pp("asp"), 32).expect("load");

    // Different temporal lengths should produce same output shape.
    let x1 = test_tensor(&[1, 32, 50]);
    let x2 = test_tensor(&[1, 32, 200]);
    let out1 = asp.forward(&x1).expect("forward short");
    let out2 = asp.forward(&x2).expect("forward long");
    assert_eq!(out1.dims(), &[1, 64]);
    assert_eq!(out2.dims(), &[1, 64]);
}

#[test]
fn test_asp_output_range() {
    let device = cpu();
    let vb = VarBuilder::zeros(crate::DType::F32, &device);
    let asp = AttentiveStatisticsPooling::load(vb.pp("asp"), 16).expect("load");
    let x = test_tensor(&[1, 16, 30]);
    let out = asp.forward(&x).expect("forward");
    let data = out.to_flat_vec::<f32>().expect("data");
    // Std part (last 16 elements) should be non-negative.
    for &v in &data[16..] {
        assert!(v >= 0.0, "std should be non-negative, got {v}");
    }
}

#[test]
fn test_asp_wrong_rank() {
    let device = cpu();
    let vb = VarBuilder::zeros(crate::DType::F32, &device);
    let asp = AttentiveStatisticsPooling::load(vb.pp("asp"), 16).expect("load");
    let x = test_tensor(&[2, 16]);
    let result = asp.forward(&x);
    assert!(result.is_err(), "should reject 2D input");
}

#[test]
fn test_asp_zero_channels_rejected() {
    let device = cpu();
    let vb = VarBuilder::zeros(crate::DType::F32, &device);
    let attention = Linear::load(vb.pp("attn"), 0, 1).expect("linear");
    let result = AttentiveStatisticsPooling::new(attention, 0);
    assert!(result.is_err(), "should reject 0 channels");
}

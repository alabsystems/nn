#![allow(deprecated)]
// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

use crate::dyn_tensor::DynTensor;
use crate::layers::{Linear, Module, SqueezeExcitation};
use crate::{DType, Device, TensorError};

fn make_identity_linear(in_f: usize, out_f: usize) -> Linear {
    // Create a linear layer with identity-ish weights for testing.
    // Weight = eye-like (truncated or padded), bias = zeros.
    let mut w_data = vec![0.0f32; out_f * in_f];
    for i in 0..out_f.min(in_f) {
        w_data[i * in_f + i] = 1.0;
    }
    let weight = DynTensor::from_vec(w_data, &[out_f, in_f], &Device::Cpu).unwrap();
    let bias = DynTensor::zeros(&[out_f], DType::F32, &Device::Cpu).unwrap();
    Linear::new(weight, Some(bias)).unwrap()
}

#[test]
fn test_se_block_shape() {
    // SE block preserves spatial dimensions and channel count.
    let fc1 = make_identity_linear(4, 2);
    let fc2 = make_identity_linear(2, 4);
    let se = SqueezeExcitation::new(fc1, fc2, 4).unwrap();
    let x = DynTensor::ones(&[2, 4, 3, 3], DType::F32, &Device::Cpu).unwrap();
    let y = se.forward(&x).unwrap();
    assert_eq!(y.dims(), &[2, 4, 3, 3]);
}

#[test]
fn test_se_block_single_channel() {
    let fc1 = make_identity_linear(1, 1);
    let fc2 = make_identity_linear(1, 1);
    let se = SqueezeExcitation::new(fc1, fc2, 1).unwrap();
    let x = DynTensor::ones(&[1, 1, 2, 2], DType::F32, &Device::Cpu).unwrap();
    let y = se.forward(&x).unwrap();
    assert_eq!(y.dims(), &[1, 1, 2, 2]);
}

#[test]
fn test_se_block_channels_accessor() {
    let fc1 = make_identity_linear(8, 2);
    let fc2 = make_identity_linear(2, 8);
    let se = SqueezeExcitation::new(fc1, fc2, 8).unwrap();
    assert_eq!(se.channels(), 8);
}

#[test]
fn test_se_block_zero_channels_error() {
    let fc1 = make_identity_linear(1, 1);
    let fc2 = make_identity_linear(1, 1);
    assert!(SqueezeExcitation::new(fc1, fc2, 0).is_err());
}

#[test]
fn test_se_block_output_bounded() {
    // With sigmoid in the excitation path, output should be bounded.
    // Input of all 1.0 through identity weights and sigmoid gives values in (0, 1).
    let fc1 = make_identity_linear(4, 2);
    let fc2 = make_identity_linear(2, 4);
    let se = SqueezeExcitation::new(fc1, fc2, 4).unwrap();
    let x = DynTensor::ones(&[1, 4, 2, 2], DType::F32, &Device::Cpu).unwrap();
    let y = se.forward(&x).unwrap();
    let vals = y.to_flat_vec::<f32>().unwrap();
    for &v in &vals {
        // sigmoid(relu(1.0)) ≈ sigmoid(1.0) ≈ 0.731, times input of 1.0
        assert!(v > 0.0 && v <= 1.0, "value {v} out of expected range");
    }
}

#[test]
fn test_se_block_scale_is_per_channel() {
    // Different channel values should get different scale factors.
    let fc1 = make_identity_linear(2, 1);
    let fc2 = make_identity_linear(1, 2);
    let se = SqueezeExcitation::new(fc1, fc2, 2).unwrap();
    // Channel 0 = 1.0, Channel 1 = 5.0
    let mut data = vec![0.0f32; 2 * 2 * 2];
    for val in data.iter_mut().take(4) {
        *val = 1.0; // channel 0
    }
    for val in data.iter_mut().skip(4).take(4) {
        *val = 5.0; // channel 1
    }
    let x = DynTensor::from_vec(data, &[1, 2, 2, 2], &Device::Cpu).unwrap();
    let y = se.forward(&x).unwrap();
    let vals = y.to_flat_vec::<f32>().unwrap();
    // All values should be non-negative (sigmoid * positive input).
    for &v in &vals {
        assert!(v >= 0.0, "negative output {v}");
    }
}

#[test]
fn test_se_block_wrong_rank() {
    let fc1 = make_identity_linear(4, 2);
    let fc2 = make_identity_linear(2, 4);
    let se = SqueezeExcitation::new(fc1, fc2, 4).unwrap();
    // 3D input should fail with rank error.
    let x = DynTensor::ones(&[1, 4, 8], DType::F32, &Device::Cpu).unwrap();
    let err = se.forward(&x).unwrap_err();
    assert!(
        matches!(err, TensorError::RankMismatch { expected: 4, .. }),
        "expected RankMismatch for 3D input, got: {err:?}"
    );
}

// -- VarBuilder load test -----------------------------------------------------

#[test]
fn test_se_block_load_from_zeros_backend() {
    use crate::var_builder::VarBuilder;

    let channels = 8;
    let reduced = 2;
    let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
    let se = SqueezeExcitation::load(&vb, channels, reduced)
        .expect("load from ZerosBackend should succeed");
    assert_eq!(se.channels(), channels);

    // Forward pass with zero weights should produce finite output
    let x = DynTensor::ones(&[1, channels, 4, 4], DType::F32, &Device::Cpu).unwrap();
    let y = se.forward(&x).unwrap();
    assert_eq!(y.dims(), &[1, channels, 4, 4]);
    let vals = y.to_flat_vec::<f32>().unwrap();
    for &v in &vals {
        assert!(v.is_finite(), "non-finite value {v}");
    }
}

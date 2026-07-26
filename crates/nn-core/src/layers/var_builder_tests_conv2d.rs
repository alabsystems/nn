#![allow(deprecated)]
// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Tests for conv2d/conv_transpose2d VarBuilder free function constructors.
//! Split from var_builder_tests_free_fns.rs for 500-line limit.

use super::super::super::*;
use crate::dyn_tensor::test_helpers::cpu;
use crate::layers::Module;
use crate::var_builder::VarBuilder;
use crate::{DType, DynTensor};
use std::collections::HashMap;

fn zeros_vb() -> VarBuilder {
    VarBuilder::zeros(DType::F32, &cpu())
}

fn map_vb(tensors: HashMap<String, DynTensor>) -> VarBuilder {
    VarBuilder::from_tensors(tensors, DType::F32, &cpu())
}

// -- conv2d() free function ---------------------------------------------------

#[test]
fn test_conv2d_fn_loads_weight_and_bias() {
    let mut tensors = HashMap::new();
    // [out_ch=1, in_ch=1, kH=2, kW=2]
    tensors.insert(
        "weight".into(),
        DynTensor::new(&[1.0, 1.0, 1.0, 1.0], &[1, 1, 2, 2], &cpu()).unwrap(),
    );
    tensors.insert(
        "bias".into(),
        DynTensor::new(&[10.0], &[1], &cpu()).unwrap(),
    );
    let vb = map_vb(tensors);
    let c = conv2d(1, 1, 2, Conv2dConfig::default(), &vb).unwrap();
    assert!(c.bias().is_some());
    // Input [1, 1, 3, 3], kernel 2×2 sum, stride 1, no padding → output [1, 1, 2, 2]
    let x = DynTensor::new(
        &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
        &[1, 1, 3, 3],
        &cpu(),
    )
    .unwrap();
    let y = c.forward(&x).unwrap();
    assert_eq!(y.dims(), &[1, 1, 2, 2]);
    let vals = y.to_flat_vec::<f32>().unwrap();
    // Top-left 2×2 sum = 1+2+4+5 = 12 + bias 10 = 22
    assert!(
        (vals[0] - 22.0).abs() < 1e-4,
        "expected 22, got {}",
        vals[0]
    );
}

#[test]
fn test_conv2d_fn_errors_without_bias() {
    let mut tensors = HashMap::new();
    tensors.insert(
        "weight".into(),
        DynTensor::new(&[1.0, 1.0, 1.0, 1.0], &[1, 1, 2, 2], &cpu()).unwrap(),
    );
    let vb = map_vb(tensors);
    assert!(conv2d(1, 1, 2, Conv2dConfig::default(), &vb).is_err());
}

#[test]
fn test_conv2d_no_bias_fn_loads_weight_only() {
    let mut tensors = HashMap::new();
    tensors.insert(
        "weight".into(),
        DynTensor::new(&[1.0, 1.0, 1.0, 1.0], &[1, 1, 2, 2], &cpu()).unwrap(),
    );
    let vb = map_vb(tensors);
    let c = conv2d_no_bias(1, 1, 2, Conv2dConfig::default(), &vb).unwrap();
    assert!(c.bias().is_none());
    let x = DynTensor::new(
        &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
        &[1, 1, 3, 3],
        &cpu(),
    )
    .unwrap();
    let y = c.forward(&x).unwrap();
    let vals = y.to_flat_vec::<f32>().unwrap();
    // Top-left 2×2 sum = 12, no bias
    assert!(
        (vals[0] - 12.0).abs() < 1e-4,
        "expected 12, got {}",
        vals[0]
    );
}

#[test]
fn test_conv2d_fn_groups_zero_returns_error() {
    let vb = zeros_vb();
    let config = Conv2dConfig::default().with_groups(0);
    let err = conv2d(4, 8, 3, config, &vb).unwrap_err();
    assert!(err.to_string().contains("groups must be > 0"), "got: {err}");
}

#[test]
fn test_conv2d_no_bias_fn_groups_zero_returns_error() {
    let vb = zeros_vb();
    let config = Conv2dConfig::default().with_groups(0);
    let err = conv2d_no_bias(4, 8, 3, config, &vb).unwrap_err();
    assert!(err.to_string().contains("groups must be > 0"), "got: {err}");
}

// -- conv_transpose2d() free function -----------------------------------------

#[test]
fn test_conv_transpose2d_fn_loads_weight_and_bias() {
    let mut tensors = HashMap::new();
    // [in_ch=2, out_ch=4, kH=3, kW=3]
    tensors.insert(
        "weight".into(),
        DynTensor::zeros(&[2, 4, 3, 3], DType::F32, &cpu()).unwrap(),
    );
    tensors.insert(
        "bias".into(),
        DynTensor::zeros(&[4], DType::F32, &cpu()).unwrap(),
    );
    let vb = map_vb(tensors);
    let c = conv_transpose2d(2, 4, 3, ConvTranspose2dConfig::default(), &vb).unwrap();
    assert!(c.bias().is_some());
    assert_eq!(c.weight().dims(), &[2, 4, 3, 3]);
}

#[test]
fn test_conv_transpose2d_fn_errors_without_bias() {
    let mut tensors = HashMap::new();
    tensors.insert(
        "weight".into(),
        DynTensor::zeros(&[2, 4, 3, 3], DType::F32, &cpu()).unwrap(),
    );
    let vb = map_vb(tensors);
    assert!(conv_transpose2d(2, 4, 3, ConvTranspose2dConfig::default(), &vb).is_err());
}

#[test]
fn test_conv_transpose2d_no_bias_fn() {
    let mut tensors = HashMap::new();
    tensors.insert(
        "weight".into(),
        DynTensor::zeros(&[2, 4, 3, 3], DType::F32, &cpu()).unwrap(),
    );
    let vb = map_vb(tensors);
    let c = conv_transpose2d_no_bias(2, 4, 3, ConvTranspose2dConfig::default(), &vb).unwrap();
    assert!(c.bias().is_none());
}

#[test]
fn test_conv_transpose2d_fn_groups_zero_returns_error() {
    let vb = zeros_vb();
    let config = ConvTranspose2dConfig::default().with_groups(0);
    let err = conv_transpose2d(4, 8, 3, config, &vb).unwrap_err();
    assert!(err.to_string().contains("groups must be > 0"), "got: {err}");
}

#[test]
fn test_conv_transpose2d_no_bias_fn_groups_zero_returns_error() {
    let vb = zeros_vb();
    let config = ConvTranspose2dConfig::default().with_groups(0);
    let err = conv_transpose2d_no_bias(4, 8, 3, config, &vb).unwrap_err();
    assert!(err.to_string().contains("groups must be > 0"), "got: {err}");
}

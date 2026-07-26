#![allow(deprecated)]
// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Tests for candle-nn compatible free function constructors (linear(), conv1d(), etc.).
//! Split from nn_var_builder_tests.rs for 500-line limit.

use super::super::*;
use crate::dyn_tensor::test_helpers::cpu;
use crate::layers::{BatchNormConfig, Module};
use crate::var_builder::VarBuilder;
use crate::{DType, DynTensor};
use std::collections::HashMap;

fn zeros_vb() -> VarBuilder {
    VarBuilder::zeros(DType::F32, &cpu())
}

fn map_vb(tensors: HashMap<String, DynTensor>) -> VarBuilder {
    VarBuilder::from_tensors(tensors, DType::F32, &cpu())
}

// -- linear() free function ---------------------------------------------------

#[test]
fn test_linear_fn_loads_weight_and_bias() {
    let mut tensors = HashMap::new();
    tensors.insert(
        "weight".into(),
        DynTensor::new(&[1.0, 0.0, 0.0, 1.0], &[2, 2], &cpu()).unwrap(),
    );
    tensors.insert(
        "bias".into(),
        DynTensor::new(&[10.0, 20.0], &[2], &cpu()).unwrap(),
    );
    let vb = map_vb(tensors);
    let lin = linear(2, 2, &vb).unwrap();
    assert!(lin.bias().is_some());
    let x = DynTensor::new(&[1.0, 2.0], &[1, 2], &cpu()).unwrap();
    let y = lin.forward(&x).unwrap();
    let vals = y.to_flat_vec::<f32>().unwrap();
    assert!((vals[0] - 11.0).abs() < 1e-5);
    assert!((vals[1] - 22.0).abs() < 1e-5);
}

#[test]
fn test_linear_fn_errors_without_bias() {
    let mut tensors = HashMap::new();
    tensors.insert(
        "weight".into(),
        DynTensor::new(&[1.0, 0.0, 0.0, 1.0], &[2, 2], &cpu()).unwrap(),
    );
    // No "bias" key — linear() (with bias) should fail
    let vb = map_vb(tensors);
    assert!(linear(2, 2, &vb).is_err());
}

// -- linear_no_bias() free function -------------------------------------------

#[test]
fn test_linear_no_bias_fn_loads_weight_only() {
    let mut tensors = HashMap::new();
    tensors.insert(
        "weight".into(),
        DynTensor::new(&[1.0, 0.0, 0.0, 1.0], &[2, 2], &cpu()).unwrap(),
    );
    let vb = map_vb(tensors);
    let lin = linear_no_bias(2, 2, &vb).unwrap();
    assert!(lin.bias().is_none());
    let x = DynTensor::new(&[3.0, 4.0], &[1, 2], &cpu()).unwrap();
    let y = lin.forward(&x).unwrap();
    let vals = y.to_flat_vec::<f32>().unwrap();
    assert!((vals[0] - 3.0).abs() < 1e-5);
    assert!((vals[1] - 4.0).abs() < 1e-5);
}

#[test]
fn test_linear_no_bias_fn_ignores_bias_tensor() {
    let mut tensors = HashMap::new();
    tensors.insert(
        "weight".into(),
        DynTensor::new(&[1.0, 0.0, 0.0, 1.0], &[2, 2], &cpu()).unwrap(),
    );
    tensors.insert(
        "bias".into(),
        DynTensor::new(&[99.0, 99.0], &[2], &cpu()).unwrap(),
    );
    // linear_no_bias should produce None bias even if "bias" key exists
    let vb = map_vb(tensors);
    let lin = linear_no_bias(2, 2, &vb).unwrap();
    assert!(lin.bias().is_none());
}

#[test]
fn test_linear_no_bias_fn_with_prefix() {
    let mut tensors = HashMap::new();
    tensors.insert(
        "q_proj.weight".into(),
        DynTensor::new(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[3, 2], &cpu()).unwrap(),
    );
    let vb = map_vb(tensors);
    let lin = linear_no_bias(2, 3, vb.pp("q_proj")).unwrap();
    assert!(lin.bias().is_none());
    assert_eq!(lin.weight().dims(), &[3, 2]);
}

// -- conv1d() free function ---------------------------------------------------

#[test]
fn test_conv1d_fn_loads_weight_and_bias() {
    let mut tensors = HashMap::new();
    tensors.insert(
        "weight".into(),
        DynTensor::new(&[1.0, 1.0, 1.0], &[1, 1, 3], &cpu()).unwrap(),
    );
    tensors.insert("bias".into(), DynTensor::new(&[5.0], &[1], &cpu()).unwrap());
    let vb = map_vb(tensors);
    let c = conv1d(1, 1, 3, Conv1dConfig::default(), &vb).unwrap();
    assert!(c.bias().is_some());
    let x = DynTensor::new(&[1.0, 2.0, 3.0, 4.0, 5.0], &[1, 1, 5], &cpu()).unwrap();
    let y = c.forward(&x).unwrap();
    let vals = y.to_flat_vec::<f32>().unwrap();
    assert!((vals[0] - 11.0).abs() < 1e-5);
}

#[test]
fn test_conv1d_fn_errors_without_bias() {
    let mut tensors = HashMap::new();
    tensors.insert(
        "weight".into(),
        DynTensor::new(&[1.0, 1.0, 1.0], &[1, 1, 3], &cpu()).unwrap(),
    );
    let vb = map_vb(tensors);
    assert!(conv1d(1, 1, 3, Conv1dConfig::default(), &vb).is_err());
}

// -- conv1d_no_bias() free function -------------------------------------------

#[test]
fn test_conv1d_no_bias_fn_loads_weight_only() {
    let mut tensors = HashMap::new();
    tensors.insert(
        "weight".into(),
        DynTensor::new(&[1.0, 1.0, 1.0], &[1, 1, 3], &cpu()).unwrap(),
    );
    let vb = map_vb(tensors);
    let c = conv1d_no_bias(1, 1, 3, Conv1dConfig::default(), &vb).unwrap();
    assert!(c.bias().is_none());
    let x = DynTensor::new(&[1.0, 2.0, 3.0, 4.0, 5.0], &[1, 1, 5], &cpu()).unwrap();
    let y = c.forward(&x).unwrap();
    let vals = y.to_flat_vec::<f32>().unwrap();
    assert!((vals[0] - 6.0).abs() < 1e-5);
}

// -- conv_transpose1d() free function -----------------------------------------

#[test]
fn test_conv_transpose1d_fn_loads_weight_and_bias() {
    let mut tensors = HashMap::new();
    tensors.insert(
        "weight".into(),
        DynTensor::zeros(&[2, 4, 3], DType::F32, &cpu()).unwrap(),
    );
    tensors.insert(
        "bias".into(),
        DynTensor::zeros(&[4], DType::F32, &cpu()).unwrap(),
    );
    let vb = map_vb(tensors);
    let c = conv_transpose1d(2, 4, 3, ConvTranspose1dConfig::default(), &vb).unwrap();
    assert!(c.bias().is_some());
    assert_eq!(c.weight().dims(), &[2, 4, 3]);
}

#[test]
fn test_conv_transpose1d_fn_errors_without_bias() {
    let mut tensors = HashMap::new();
    tensors.insert(
        "weight".into(),
        DynTensor::zeros(&[2, 4, 3], DType::F32, &cpu()).unwrap(),
    );
    let vb = map_vb(tensors);
    assert!(conv_transpose1d(2, 4, 3, ConvTranspose1dConfig::default(), &vb).is_err());
}

// -- conv_transpose1d_no_bias() free function ---------------------------------

#[test]
fn test_conv_transpose1d_no_bias_fn() {
    let mut tensors = HashMap::new();
    tensors.insert(
        "weight".into(),
        DynTensor::zeros(&[2, 4, 3], DType::F32, &cpu()).unwrap(),
    );
    let vb = map_vb(tensors);
    let c = conv_transpose1d_no_bias(2, 4, 3, ConvTranspose1dConfig::default(), &vb).unwrap();
    assert!(c.bias().is_none());
}

// -- conv2d/conv_transpose2d tests extracted to var_builder_tests_conv2d.rs ----
#[path = "var_builder_tests_conv2d.rs"]
mod conv2d_tests;

// -- layer_norm() free function -----------------------------------------------

#[test]
fn test_layer_norm_fn() {
    let mut tensors = HashMap::new();
    tensors.insert(
        "weight".into(),
        DynTensor::ones(&[4], DType::F32, &cpu()).unwrap(),
    );
    tensors.insert(
        "bias".into(),
        DynTensor::zeros(&[4], DType::F32, &cpu()).unwrap(),
    );
    let vb = map_vb(tensors);
    let ln = layer_norm(4, LayerNormConfig::default(), &vb).unwrap();
    assert_eq!(ln.weight().dims(), &[4]);
}

#[test]
fn test_layer_norm_fn_custom_eps() {
    let vb = zeros_vb();
    let config = LayerNormConfig { eps: 1e-12 };
    let ln = layer_norm(8, config, &vb).unwrap();
    assert_eq!(ln.weight().dims(), &[8]);
}

// -- rms_norm() free function -------------------------------------------------

#[test]
fn test_rms_norm_fn() {
    let mut tensors = HashMap::new();
    tensors.insert(
        "weight".into(),
        DynTensor::ones(&[16], DType::F32, &cpu()).unwrap(),
    );
    let vb = map_vb(tensors);
    let rms = rms_norm(16, 1e-6, &vb).unwrap();
    assert_eq!(rms.weight().dims(), &[16]);
}

// -- group_norm() free function -----------------------------------------------

#[test]
fn test_group_norm_fn() {
    let vb = zeros_vb();
    let gn = group_norm(2, 8, 1e-5, &vb).unwrap();
    let x = DynTensor::ones(&[1, 8, 4], DType::F32, &cpu()).unwrap();
    let y = gn.forward(&x).unwrap();
    assert_eq!(y.dims(), &[1, 8, 4]);
}

// -- embedding() free function ------------------------------------------------

#[test]
fn test_embedding_fn() {
    let mut tensors = HashMap::new();
    tensors.insert(
        "weight".into(),
        DynTensor::new(&[10.0, 11.0, 20.0, 21.0, 30.0, 31.0], &[3, 2], &cpu()).unwrap(),
    );
    let vb = map_vb(tensors);
    let emb = embedding(3, 2, &vb).unwrap();
    let result = emb.forward_ids(&[2]).unwrap();
    let vals = result.to_flat_vec::<f32>().unwrap();
    assert!((vals[0] - 30.0).abs() < 1e-6);
    assert!((vals[1] - 31.0).abs() < 1e-6);
}

// -- validate_groups error tests (AC4 for #1144) ------------------------------

#[test]
fn test_conv1d_load_groups_zero_returns_error() {
    let vb = zeros_vb();
    let config = Conv1dConfig {
        groups: 0,
        ..Default::default()
    };
    let err = Conv1d::load(&vb, 4, 8, 3, config).unwrap_err();
    assert!(err.to_string().contains("groups must be > 0"), "got: {err}");
}

#[test]
fn test_conv1d_load_groups_not_divisible_returns_error() {
    let vb = zeros_vb();
    let config = Conv1dConfig {
        groups: 3,
        ..Default::default()
    };
    let err = Conv1d::load(&vb, 4, 8, 3, config).unwrap_err();
    assert!(
        err.to_string().contains("not divisible by groups"),
        "got: {err}"
    );
}

#[test]
fn test_conv1d_fn_groups_zero_returns_error() {
    let vb = zeros_vb();
    let config = Conv1dConfig {
        groups: 0,
        ..Default::default()
    };
    let err = conv1d(4, 8, 3, config, &vb).unwrap_err();
    assert!(err.to_string().contains("groups must be > 0"), "got: {err}");
}

#[test]
fn test_conv1d_no_bias_fn_groups_zero_returns_error() {
    let vb = zeros_vb();
    let config = Conv1dConfig {
        groups: 0,
        ..Default::default()
    };
    let err = conv1d_no_bias(4, 8, 3, config, &vb).unwrap_err();
    assert!(err.to_string().contains("groups must be > 0"), "got: {err}");
}

#[test]
fn test_conv_transpose1d_load_groups_zero_returns_error() {
    let vb = zeros_vb();
    let config = ConvTranspose1dConfig {
        groups: 0,
        ..Default::default()
    };
    let err = ConvTranspose1d::load(&vb, 4, 8, 3, config).unwrap_err();
    assert!(err.to_string().contains("groups must be > 0"), "got: {err}");
}

#[test]
fn test_conv_transpose1d_load_groups_not_divisible_returns_error() {
    let vb = zeros_vb();
    let config = ConvTranspose1dConfig {
        groups: 3,
        ..Default::default()
    };
    // ConvTranspose1d validates out_channels (8) % groups (3) != 0
    let err = ConvTranspose1d::load(&vb, 4, 8, 3, config).unwrap_err();
    assert!(
        err.to_string().contains("not divisible by groups"),
        "got: {err}"
    );
}

#[test]
fn test_conv_transpose1d_fn_groups_zero_returns_error() {
    let vb = zeros_vb();
    let config = ConvTranspose1dConfig {
        groups: 0,
        ..Default::default()
    };
    let err = conv_transpose1d(4, 8, 3, config, &vb).unwrap_err();
    assert!(err.to_string().contains("groups must be > 0"), "got: {err}");
}

#[test]
fn test_conv_transpose1d_no_bias_fn_groups_zero_returns_error() {
    let vb = zeros_vb();
    let config = ConvTranspose1dConfig {
        groups: 0,
        ..Default::default()
    };
    let err = conv_transpose1d_no_bias(4, 8, 3, config, &vb).unwrap_err();
    assert!(err.to_string().contains("groups must be > 0"), "got: {err}");
}

// -- lstm/BiLstm tests extracted to nn_var_builder_tests_lstm.rs --------------
#[path = "var_builder_tests_lstm.rs"]
mod lstm_tests;

// -- batch_norm() free function -----------------------------------------------

#[test]
fn test_batch_norm_fn() {
    let mut tensors = HashMap::new();
    tensors.insert(
        "running_mean".into(),
        DynTensor::zeros(&[4], DType::F32, &cpu()).unwrap(),
    );
    tensors.insert(
        "running_var".into(),
        DynTensor::ones(&[4], DType::F32, &cpu()).unwrap(),
    );
    tensors.insert(
        "weight".into(),
        DynTensor::ones(&[4], DType::F32, &cpu()).unwrap(),
    );
    tensors.insert(
        "bias".into(),
        DynTensor::zeros(&[4], DType::F32, &cpu()).unwrap(),
    );
    let vb = map_vb(tensors);
    let bn = batch_norm(4, BatchNormConfig::default(), &vb).unwrap();
    assert!(bn.weight().is_some());
}

#![allow(deprecated)]
// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Tests for VarBuilder-based `load()` constructors on nn layers.
//!
//! Free function tests (linear(), conv1d(), etc.) are in
//! `nn_var_builder_tests_free_fns.rs`.

use super::*;
use crate::dyn_tensor::test_helpers::cpu;
use crate::layers::{
    BatchNorm, BatchNormConfig, Embedding, GroupNorm, LayerNorm, Lstm, Module, RmsNorm,
};
use crate::var_builder::VarBuilder;
use crate::{DType, DynTensor};
use std::collections::HashMap;

#[path = "var_builder_tests_free_fns.rs"]
mod free_fns;

fn zeros_vb() -> VarBuilder {
    VarBuilder::zeros(DType::F32, &cpu())
}

fn map_vb(tensors: HashMap<String, DynTensor>) -> VarBuilder {
    VarBuilder::from_tensors(tensors, DType::F32, &cpu())
}

// -- Linear::load -------------------------------------------------------------

#[test]
fn test_linear_load_with_bias() {
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
    let linear = Linear::load(&vb, 2, 2).unwrap();
    assert!(linear.bias().is_some());
    let x = DynTensor::new(&[1.0, 2.0], &[1, 2], &cpu()).unwrap();
    let y = linear.forward(&x).unwrap();
    let vals = y.to_flat_vec::<f32>().unwrap();
    assert!((vals[0] - 11.0).abs() < 1e-5); // 1 + 10
    assert!((vals[1] - 22.0).abs() < 1e-5); // 2 + 20
}

#[test]
fn test_linear_load_no_bias() {
    let mut tensors = HashMap::new();
    tensors.insert(
        "weight".into(),
        DynTensor::new(&[1.0, 0.0, 0.0, 1.0], &[2, 2], &cpu()).unwrap(),
    );
    // No "bias" key — should load without bias
    let vb = map_vb(tensors);
    let linear = Linear::load(&vb, 2, 2).unwrap();
    assert!(linear.bias().is_none());
}

#[test]
fn test_linear_load_zeros_backend() {
    let vb = zeros_vb();
    let linear = Linear::load(&vb, 3, 4).unwrap();
    assert_eq!(linear.weight().dims(), &[4, 3]);
    // ZerosBackend contains_tensor returns true, so bias is loaded
    assert!(linear.bias().is_some());
    assert_eq!(linear.bias().unwrap().dims(), &[4]);
}

// -- Conv1d::load -------------------------------------------------------------

#[test]
fn test_conv1d_load_with_bias() {
    let mut tensors = HashMap::new();
    tensors.insert(
        "weight".into(),
        DynTensor::new(&[1.0, 1.0, 1.0], &[1, 1, 3], &cpu()).unwrap(),
    );
    tensors.insert("bias".into(), DynTensor::new(&[5.0], &[1], &cpu()).unwrap());
    let vb = map_vb(tensors);
    let conv = Conv1d::load(&vb, 1, 1, 3, Conv1dConfig::default()).unwrap();
    assert!(conv.bias().is_some());
    let x = DynTensor::new(&[1.0, 2.0, 3.0, 4.0, 5.0], &[1, 1, 5], &cpu()).unwrap();
    let y = conv.forward(&x).unwrap();
    assert_eq!(y.dims(), &[1, 1, 3]);
    let vals = y.to_flat_vec::<f32>().unwrap();
    // sum kernel [1,1,1]: [6, 9, 12] + bias 5 = [11, 14, 17]
    assert!((vals[0] - 11.0).abs() < 1e-5);
    assert!((vals[1] - 14.0).abs() < 1e-5);
    assert!((vals[2] - 17.0).abs() < 1e-5);
}

#[test]
fn test_conv1d_load_no_bias() {
    let mut tensors = HashMap::new();
    tensors.insert(
        "weight".into(),
        DynTensor::new(&[1.0, 1.0, 1.0], &[1, 1, 3], &cpu()).unwrap(),
    );
    let vb = map_vb(tensors);
    let conv = Conv1d::load(&vb, 1, 1, 3, Conv1dConfig::default()).unwrap();
    assert!(conv.bias().is_none());
}

#[test]
fn test_conv1d_load_zeros_backend() {
    let vb = zeros_vb();
    let conv = Conv1d::load(&vb, 4, 8, 3, Conv1dConfig::default()).unwrap();
    assert_eq!(conv.weight().dims(), &[8, 4, 3]);
}

// -- ConvTranspose1d::load ----------------------------------------------------

#[test]
fn test_conv_transpose1d_load_zeros_backend() {
    let vb = zeros_vb();
    let config = ConvTranspose1dConfig::default();
    let conv = ConvTranspose1d::load(&vb, 4, 8, 3, config).unwrap();
    assert_eq!(conv.weight().dims(), &[4, 8, 3]);
}

// -- LayerNorm::load ----------------------------------------------------------

#[test]
fn test_layer_norm_load() {
    let mut tensors = HashMap::new();
    tensors.insert(
        "weight".into(),
        DynTensor::full(&[4], 2.0, DType::F32, &cpu()).unwrap(),
    );
    tensors.insert(
        "bias".into(),
        DynTensor::full(&[4], 3.0, DType::F32, &cpu()).unwrap(),
    );
    let vb = map_vb(tensors);
    let ln = LayerNorm::load(&vb, 4, 1e-5).unwrap();
    assert_eq!(ln.weight().dims(), &[4]);
    assert_eq!(ln.bias().dims(), &[4]);
    // Forward: input=[1,5,3,7], mean=4, var=5, normalized * 2.0 + 3.0
    let x = DynTensor::new(&[1.0, 5.0, 3.0, 7.0], &[1, 4], &cpu()).unwrap();
    let y = ln.forward(&x).unwrap();
    let vals = y.to_flat_vec::<f32>().unwrap();
    let inv_std = 1.0 / (5.0_f32 + 1e-5).sqrt();
    let expected = [
        2.0 * (-3.0 * inv_std) + 3.0,
        2.0 * (1.0 * inv_std) + 3.0,
        2.0 * (-inv_std) + 3.0,
        2.0 * (3.0 * inv_std) + 3.0,
    ];
    for (i, (&got, &exp)) in vals.iter().zip(expected.iter()).enumerate() {
        assert!(
            (got - exp).abs() < 1e-5,
            "ln[{i}]: got {got}, expected {exp}"
        );
    }
}

#[test]
fn test_layer_norm_load_zeros() {
    let vb = zeros_vb();
    let ln = LayerNorm::load(&vb, 8, 1e-5).unwrap();
    assert_eq!(ln.weight().dims(), &[8]);
    assert_eq!(ln.bias().dims(), &[8]);
}

// -- GroupNorm::load ----------------------------------------------------------

#[test]
fn test_group_norm_load() {
    // Build GroupNorm with weight=1.0, bias=0.0 via known tensors
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
    let gn = GroupNorm::load(&vb, 2, 4, 1e-5).unwrap();
    // Input [1, 4, 1]: channels [1,5] in group 0, [3,7] in group 1
    let x = DynTensor::new(&[1.0, 5.0, 3.0, 7.0], &[1, 4, 1], &cpu()).unwrap();
    let y = gn.forward(&x).unwrap();
    assert_eq!(y.dims(), &[1, 4, 1]);
    let vals = y.to_flat_vec::<f32>().unwrap();
    // Group 0: mean=3, var=4 → inv_std=1/sqrt(4+eps) → [-1, 1]
    // Group 1: mean=5, var=4 → inv_std=1/sqrt(4+eps) → [-1, 1]
    let inv_std = 1.0 / (4.0_f32 + 1e-5).sqrt();
    assert!(
        (vals[0] - (-2.0 * inv_std)).abs() < 1e-5,
        "gn[0]={}",
        vals[0]
    );
    assert!(
        (vals[1] - (2.0 * inv_std)).abs() < 1e-5,
        "gn[1]={}",
        vals[1]
    );
    assert!(
        (vals[2] - (-2.0 * inv_std)).abs() < 1e-5,
        "gn[2]={}",
        vals[2]
    );
    assert!(
        (vals[3] - (2.0 * inv_std)).abs() < 1e-5,
        "gn[3]={}",
        vals[3]
    );
}

// -- BatchNorm::load ----------------------------------------------------------

#[test]
fn test_batch_norm_load() {
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
    let bn = BatchNorm::load(&vb, 4, BatchNormConfig::default()).unwrap();
    assert!(bn.weight().is_some());
    assert!(bn.bias().is_some());
}

#[test]
fn test_batch_norm_load_no_affine() {
    let mut tensors = HashMap::new();
    tensors.insert(
        "running_mean".into(),
        DynTensor::zeros(&[4], DType::F32, &cpu()).unwrap(),
    );
    tensors.insert(
        "running_var".into(),
        DynTensor::ones(&[4], DType::F32, &cpu()).unwrap(),
    );
    let vb = map_vb(tensors);
    let config = BatchNormConfig {
        affine: false,
        ..Default::default()
    };
    let bn = BatchNorm::load(&vb, 4, config).unwrap();
    assert!(bn.weight().is_none());
    assert!(bn.bias().is_none());
}

// -- RmsNorm::load ------------------------------------------------------------

#[test]
fn test_rms_norm_load() {
    let mut tensors = HashMap::new();
    tensors.insert(
        "weight".into(),
        DynTensor::ones(&[8], DType::F32, &cpu()).unwrap(),
    );
    let vb = map_vb(tensors);
    let rms = RmsNorm::load(&vb, 8, 1e-5).unwrap();
    assert_eq!(rms.weight().dims(), &[8]);
    // Forward: input=[3, 4], rms = sqrt((9+16)/2 + eps) = sqrt(12.5+eps)
    // With weight=1.0, output = input / rms
    let x = DynTensor::new(&[3.0, 4.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], &[1, 8], &cpu()).unwrap();
    let y = rms.forward(&x).unwrap();
    let vals = y.to_flat_vec::<f32>().unwrap();
    let rms_val = ((9.0 + 16.0) / 8.0_f32 + 1e-5).sqrt();
    assert!(
        (vals[0] - 3.0 / rms_val).abs() < 1e-5,
        "rms[0]: got {}, expected {}",
        vals[0],
        3.0 / rms_val
    );
    assert!(
        (vals[1] - 4.0 / rms_val).abs() < 1e-5,
        "rms[1]: got {}, expected {}",
        vals[1],
        4.0 / rms_val
    );
}

#[test]
fn test_rms_norm_load_zeros() {
    let vb = zeros_vb();
    let rms = RmsNorm::load(&vb, 16, 1e-6).unwrap();
    assert_eq!(rms.weight().dims(), &[16]);
}

// -- Embedding::load ----------------------------------------------------------

#[test]
fn test_embedding_load() {
    let mut tensors = HashMap::new();
    tensors.insert(
        "weight".into(),
        DynTensor::new(&[10.0, 11.0, 20.0, 21.0, 30.0, 31.0], &[3, 2], &cpu()).unwrap(),
    );
    let vb = map_vb(tensors);
    let emb = Embedding::load(&vb, 3, 2).unwrap();
    assert_eq!(emb.weight().dims(), &[3, 2]);
    let result = emb.forward_ids(&[1]).unwrap();
    let vals = result.to_flat_vec::<f32>().unwrap();
    assert!((vals[0] - 20.0).abs() < 1e-6);
    assert!((vals[1] - 21.0).abs() < 1e-6);
}

#[test]
fn test_embedding_load_zeros() {
    let vb = zeros_vb();
    let emb = Embedding::load(&vb, 1000, 512).unwrap();
    assert_eq!(emb.weight().dims(), &[1000, 512]);
}

// -- Lstm::load ---------------------------------------------------------------

#[test]
fn test_lstm_load_with_bias() {
    let hidden = 4;
    let input = 3;
    let four_h = 4 * hidden;
    let mut tensors = HashMap::new();
    tensors.insert(
        "weight_ih_l0".into(),
        DynTensor::zeros(&[four_h, input], DType::F32, &cpu()).unwrap(),
    );
    tensors.insert(
        "weight_hh_l0".into(),
        DynTensor::zeros(&[four_h, hidden], DType::F32, &cpu()).unwrap(),
    );
    tensors.insert(
        "bias_ih_l0".into(),
        DynTensor::zeros(&[four_h], DType::F32, &cpu()).unwrap(),
    );
    tensors.insert(
        "bias_hh_l0".into(),
        DynTensor::zeros(&[four_h], DType::F32, &cpu()).unwrap(),
    );
    let vb = map_vb(tensors);
    let lstm = Lstm::load(&vb, input, hidden).unwrap();
    assert_eq!(lstm.hidden_size(), hidden);
    // Forward with zero weights/bias: all gates = sigmoid(0)=0.5, g=tanh(0)=0
    // c_new = f*0 + i*g = 0.5*0 + 0.5*0 = 0, h_new = o*tanh(0) = 0
    let x = DynTensor::ones(&[1, input], DType::F32, &cpu()).unwrap();
    let (_output, state) = lstm.forward(&x, None).unwrap();
    assert_eq!(state.h.dims(), &[1, hidden]);
    assert_eq!(state.c.dims(), &[1, hidden]);
    let h_vals = state.h.to_flat_vec::<f32>().unwrap();
    let c_vals = state.c.to_flat_vec::<f32>().unwrap();
    for (i, &v) in c_vals.iter().enumerate() {
        assert!(v.abs() < 1e-6, "c[{i}] should be 0.0, got {v}");
    }
    for (i, &v) in h_vals.iter().enumerate() {
        assert!(v.abs() < 1e-6, "h[{i}] should be 0.0, got {v}");
    }
}

#[test]
fn test_lstm_load_no_bias() {
    let hidden = 4;
    let input = 3;
    let four_h = 4 * hidden;
    let mut tensors = HashMap::new();
    tensors.insert(
        "weight_ih_l0".into(),
        DynTensor::zeros(&[four_h, input], DType::F32, &cpu()).unwrap(),
    );
    tensors.insert(
        "weight_hh_l0".into(),
        DynTensor::zeros(&[four_h, hidden], DType::F32, &cpu()).unwrap(),
    );
    // No bias keys
    let vb = map_vb(tensors);
    let lstm = Lstm::load(&vb, input, hidden).unwrap();
    assert_eq!(lstm.hidden_size(), hidden);
}

// -- VarBuilder scoping with load ---------------------------------------------

#[test]
fn test_linear_load_with_prefix() {
    let mut tensors = HashMap::new();
    tensors.insert(
        "encoder.linear.weight".into(),
        DynTensor::new(&[1.0, 0.0, 0.0, 1.0], &[2, 2], &cpu()).unwrap(),
    );
    tensors.insert(
        "encoder.linear.bias".into(),
        DynTensor::new(&[5.0, 10.0], &[2], &cpu()).unwrap(),
    );
    let vb = map_vb(tensors);
    let linear = Linear::load(vb.pp("encoder").pp("linear"), 2, 2).unwrap();
    assert!(linear.bias().is_some());
    let x = DynTensor::new(&[1.0, 2.0], &[1, 2], &cpu()).unwrap();
    let y = linear.forward(&x).unwrap();
    let vals = y.to_flat_vec::<f32>().unwrap();
    assert!((vals[0] - 6.0).abs() < 1e-5);
    assert!((vals[1] - 12.0).abs() < 1e-5);
}

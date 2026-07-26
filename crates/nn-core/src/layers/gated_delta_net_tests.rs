#![allow(deprecated)]
// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Basic tests for [`GatedDeltaNet`] linear attention module.
//!
//! Tests forward pass shapes, batched forward, state initialization, validation
//! errors, single timestep, and input sensitivity. Numerical correctness,
//! NaN/Inf rejection, and VarBuilder load tests are in
//! `gated_delta_net_tests_extended.rs`.

use super::*;
use crate::dyn_tensor::DynTensor;
use crate::layers::Linear;
use crate::{DType, Device};

/// Helper: create a GatedDeltaNet with small dimensions for testing.
fn make_test_gdn(
    batch: usize,
    dim: usize,
    num_heads: usize,
    key_dim: usize,
    value_dim: usize,
) -> (GatedDeltaNet, DynTensor) {
    let device = Device::Cpu;
    let qk_total = num_heads * key_dim;
    let v_total = num_heads * value_dim;

    // Projection weights (identity-like for predictable behavior)
    let q_w = DynTensor::ones(&[qk_total, dim], DType::F32, &device).unwrap();
    let k_w = DynTensor::ones(&[qk_total, dim], DType::F32, &device).unwrap();
    let v_w = DynTensor::ones(&[v_total, dim], DType::F32, &device).unwrap();
    let gate_w = DynTensor::ones(&[num_heads, dim], DType::F32, &device).unwrap();
    let beta_w = DynTensor::ones(&[num_heads, dim], DType::F32, &device).unwrap();
    // out_proj: [dim, v_total] (maps H*V back to D)
    let out_w = DynTensor::ones(&[dim, v_total], DType::F32, &device).unwrap();

    let q_proj = Linear::new(q_w, None).unwrap();
    let k_proj = Linear::new(k_w, None).unwrap();
    let v_proj = Linear::new(v_w, None).unwrap();
    let gate_proj = Linear::new(gate_w, None).unwrap();
    let beta_proj = Linear::new(beta_w, None).unwrap();
    let out_proj = Linear::new(out_w, None).unwrap();

    let gdn = GatedDeltaNet::new(
        q_proj, k_proj, v_proj, gate_proj, beta_proj, out_proj, num_heads, key_dim, value_dim,
    )
    .unwrap();

    let seq_len = 2;
    // Small input: [batch, seq, dim]
    let input = DynTensor::from_vec(
        vec![0.1; batch * seq_len * dim],
        &[batch, seq_len, dim],
        &device,
    )
    .unwrap();

    (gdn, input)
}

#[test]
fn test_gdn_forward_basic() {
    let (gdn, input) = make_test_gdn(1, 4, 2, 2, 2);
    let (output, new_state) = gdn.forward(&input, None).unwrap();

    // Output shape: [batch=1, seq=2, dim=4]
    assert_eq!(output.dims(), &[1, 2, 4]);
    // State shape: [batch=1, heads=2, key_dim=2, value_dim=2]
    assert_eq!(new_state.state.dims(), &[1, 2, 2, 2]);
}

#[test]
fn test_gdn_forward_with_state() {
    let (gdn, input) = make_test_gdn(1, 4, 2, 2, 2);

    // First forward pass
    let (_, state1) = gdn.forward(&input, None).unwrap();

    // Second forward pass with state from first
    let (output2, state2) = gdn.forward(&input, Some(&state1)).unwrap();

    assert_eq!(output2.dims(), &[1, 2, 4]);
    assert_eq!(state2.state.dims(), &[1, 2, 2, 2]);

    // State should be different from first pass (recurrence accumulated)
    let s1_data = state1.state.to_flat_vec::<f32>().unwrap();
    let s2_data = state2.state.to_flat_vec::<f32>().unwrap();
    let state_differs = s1_data
        .iter()
        .zip(&s2_data)
        .any(|(a, b)| (a - b).abs() > 1e-6);
    assert!(
        state_differs,
        "State should change between passes with recurrence"
    );
}

#[test]
fn test_gdn_forward_batched() {
    let (gdn, input) = make_test_gdn(3, 4, 2, 2, 2);
    let (output, state) = gdn.forward(&input, None).unwrap();

    assert_eq!(output.dims(), &[3, 2, 4]);
    assert_eq!(state.state.dims(), &[3, 2, 2, 2]);
}

#[test]
fn test_gdn_zero_state_initialization() {
    let state = GatedDeltaNetState::zeros(2, 4, 3, 3, &Device::Cpu).unwrap();
    assert_eq!(state.state.dims(), &[2, 4, 3, 3]);

    let data = state.state.to_flat_vec::<f32>().unwrap();
    assert!(data.iter().all(|v| *v == 0.0));
}

#[test]
fn test_gdn_validation_zero_heads() {
    let device = Device::Cpu;
    let w = DynTensor::ones(&[4, 4], DType::F32, &device).unwrap();
    let gate_w = DynTensor::ones(&[1, 4], DType::F32, &device).unwrap();

    let result = GatedDeltaNet::new(
        Linear::new(w.clone(), None).unwrap(),
        Linear::new(w.clone(), None).unwrap(),
        Linear::new(w.clone(), None).unwrap(),
        Linear::new(gate_w.clone(), None).unwrap(),
        Linear::new(gate_w, None).unwrap(),
        Linear::new(w, None).unwrap(),
        0, // zero heads
        2,
        2,
    );
    assert!(result.is_err());
    let err = result.unwrap_err().to_string();
    assert!(
        err.contains("num_heads"),
        "Error should mention num_heads: {err}"
    );
}

#[test]
fn test_gdn_validation_zero_key_dim() {
    let device = Device::Cpu;
    let w = DynTensor::ones(&[4, 4], DType::F32, &device).unwrap();
    let gate_w = DynTensor::ones(&[2, 4], DType::F32, &device).unwrap();

    let result = GatedDeltaNet::new(
        Linear::new(w.clone(), None).unwrap(),
        Linear::new(w.clone(), None).unwrap(),
        Linear::new(w.clone(), None).unwrap(),
        Linear::new(gate_w.clone(), None).unwrap(),
        Linear::new(gate_w, None).unwrap(),
        Linear::new(w, None).unwrap(),
        2,
        0, // zero key_dim
        2,
    );
    assert!(result.is_err());
    let err = result.unwrap_err().to_string();
    assert!(
        err.contains("key_dim"),
        "Error should mention key_dim: {err}"
    );
}

#[test]
fn test_gdn_single_timestep() {
    let device = Device::Cpu;
    let dim = 4;
    let num_heads = 2;
    let key_dim = 2;
    let value_dim = 2;
    let qk_total = num_heads * key_dim;
    let v_total = num_heads * value_dim;

    let q_w = DynTensor::ones(&[qk_total, dim], DType::F32, &device).unwrap();
    let k_w = DynTensor::ones(&[qk_total, dim], DType::F32, &device).unwrap();
    let v_w = DynTensor::ones(&[v_total, dim], DType::F32, &device).unwrap();
    let gate_w = DynTensor::ones(&[num_heads, dim], DType::F32, &device).unwrap();
    let beta_w = DynTensor::ones(&[num_heads, dim], DType::F32, &device).unwrap();
    let out_w = DynTensor::ones(&[dim, v_total], DType::F32, &device).unwrap();

    let gdn = GatedDeltaNet::new(
        Linear::new(q_w, None).unwrap(),
        Linear::new(k_w, None).unwrap(),
        Linear::new(v_w, None).unwrap(),
        Linear::new(gate_w, None).unwrap(),
        Linear::new(beta_w, None).unwrap(),
        Linear::new(out_w, None).unwrap(),
        num_heads,
        key_dim,
        value_dim,
    )
    .unwrap();

    // Single timestep: [1, 1, 4]
    let input = DynTensor::from_vec(vec![0.1; dim], &[1, 1, dim], &device).unwrap();
    let (output, state) = gdn.forward(&input, None).unwrap();

    assert_eq!(output.dims(), &[1, 1, dim]);
    assert_eq!(state.state.dims(), &[1, num_heads, key_dim, value_dim]);

    // Output should be finite
    let out_data = output.to_flat_vec::<f32>().unwrap();
    assert!(out_data.iter().all(|v| v.is_finite()));
}

#[test]
fn test_gdn_output_changes_with_input() {
    let (gdn, _) = make_test_gdn(1, 4, 2, 2, 2);
    let device = Device::Cpu;

    let input_a = DynTensor::from_vec(vec![0.1; 4], &[1, 1, 4], &device).unwrap();
    let input_b = DynTensor::from_vec(vec![0.5; 4], &[1, 1, 4], &device).unwrap();

    let (out_a, _) = gdn.forward(&input_a, None).unwrap();
    let (out_b, _) = gdn.forward(&input_b, None).unwrap();

    let a_data = out_a.to_flat_vec::<f32>().unwrap();
    let b_data = out_b.to_flat_vec::<f32>().unwrap();

    let differs = a_data
        .iter()
        .zip(&b_data)
        .any(|(a, b)| (a - b).abs() > 1e-6);
    assert!(differs, "Different inputs should produce different outputs");
}

#[test]
fn test_gdn_wrong_input_rank() {
    let (gdn, _) = make_test_gdn(1, 4, 2, 2, 2);
    let device = Device::Cpu;

    // 2D input (wrong)
    let bad_input = DynTensor::from_vec(vec![0.1; 4], &[1, 4], &device).unwrap();
    let result = gdn.forward(&bad_input, None);
    assert!(result.is_err());
}

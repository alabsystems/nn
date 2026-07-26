#![allow(deprecated)]
// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Extended tests for [`GatedDeltaNet`]: numerical correctness, NaN/Inf
//! rejection, and VarBuilder load paths.
//!
//! Extracted from `gated_delta_net_tests.rs` — basic forward/shape/validation
//! tests remain there.

use super::*;
use crate::dyn_tensor::DynTensor;
use crate::layers::Linear;
use crate::var_builder::VarBuilder;
use crate::{DType, Device, TensorError};
use std::collections::HashMap;

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

    let q_w = DynTensor::ones(&[qk_total, dim], DType::F32, &device).unwrap();
    let k_w = DynTensor::ones(&[qk_total, dim], DType::F32, &device).unwrap();
    let v_w = DynTensor::ones(&[v_total, dim], DType::F32, &device).unwrap();
    let gate_w = DynTensor::ones(&[num_heads, dim], DType::F32, &device).unwrap();
    let beta_w = DynTensor::ones(&[num_heads, dim], DType::F32, &device).unwrap();
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
    let input = DynTensor::from_vec(
        vec![0.1; batch * seq_len * dim],
        &[batch, seq_len, dim],
        &device,
    )
    .unwrap();

    (gdn, input)
}

/// Build a GatedDeltaNet with non-degenerate weights for numerical testing.
fn make_nondegenerate_gdn() -> GatedDeltaNet {
    let device = Device::Cpu;
    let q_w = DynTensor::from_vec(vec![1.0, 0.5, 0.5, 1.0], &[2, 2], &device).unwrap();
    let k_w = DynTensor::from_vec(vec![0.8, -0.3, -0.3, 0.8], &[2, 2], &device).unwrap();
    let v_w = DynTensor::from_vec(vec![0.6, 0.4, 0.4, 0.6], &[2, 2], &device).unwrap();
    let gate_w = DynTensor::from_vec(vec![0.5, 0.5], &[1, 2], &device).unwrap();
    let beta_w = DynTensor::from_vec(vec![0.3, 0.7], &[1, 2], &device).unwrap();
    let out_w = DynTensor::from_vec(vec![1.0, 0.0, 0.0, 1.0], &[2, 2], &device).unwrap();
    GatedDeltaNet::new(
        Linear::new(q_w, None).unwrap(),
        Linear::new(k_w, None).unwrap(),
        Linear::new(v_w, None).unwrap(),
        Linear::new(gate_w, None).unwrap(),
        Linear::new(beta_w, None).unwrap(),
        Linear::new(out_w, None).unwrap(),
        1,
        2,
        2,
    )
    .unwrap()
}

// -- Numerical correctness tests ----------------------------------------------

/// Hand-computed reference test: input x=[0.3, 0.7], zero state.
#[test]
fn test_gdn_numerical_correctness_single_step() {
    let gdn = make_nondegenerate_gdn();
    let input = DynTensor::from_vec(vec![0.3, 0.7], &[1, 1, 2], &Device::Cpu).unwrap();

    let (output, _state) = gdn.forward(&input, None).unwrap();
    let out_data = output.to_flat_vec::<f32>().unwrap();

    assert_eq!(out_data.len(), 2);
    let tol = 1e-4;
    assert!(
        (out_data[0] - 0.087383).abs() < tol,
        "output[0]: expected ~0.087383, got {}",
        out_data[0]
    );
    assert!(
        (out_data[1] - 0.102576).abs() < tol,
        "output[1]: expected ~0.102576, got {}",
        out_data[1]
    );
}

/// Regression guard: non-degenerate weights expose sign/operand bugs.
#[test]
fn test_gdn_sign_sensitivity() {
    let gdn = make_nondegenerate_gdn();
    let input = DynTensor::from_vec(vec![0.3, 0.7, 0.5, 0.2], &[1, 2, 2], &Device::Cpu).unwrap();

    let (output, state) = gdn.forward(&input, None).unwrap();
    let out_data = output.to_flat_vec::<f32>().unwrap();
    let state_data = state.state.to_flat_vec::<f32>().unwrap();

    assert!(
        out_data.iter().all(|v| v.is_finite()),
        "non-finite output: {out_data:?}"
    );
    assert!(
        state_data.iter().all(|v| v.is_finite()),
        "non-finite state: {state_data:?}"
    );

    let (t0, t1) = (&out_data[0..2], &out_data[2..4]);
    let differs = t0.iter().zip(t1).any(|(a, b)| (a - b).abs() > 1e-6);
    assert!(
        differs,
        "timestep outputs should differ: t0={t0:?}, t1={t1:?}"
    );

    assert!(
        (t1[0] - t1[1]).abs() > 1e-6,
        "dimensions should differ: {t1:?}"
    );
}

// -- NaN/Inf input tests (#1209 finiteness coverage) -------------------------

#[test]
fn test_gdn_nan_input_returns_error() {
    let (gdn, _) = make_test_gdn(1, 4, 2, 2, 2);
    let input = DynTensor::from_vec(
        vec![f32::NAN, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7],
        &[1, 2, 4],
        &Device::Cpu,
    )
    .unwrap();
    let result = gdn.forward(&input, None);
    assert!(result.is_err(), "NaN input should produce an error");
    let err = result.map(|_| ()).unwrap_err();
    match err {
        TensorError::NonFiniteData { name, count } => {
            assert!(
                name.contains("GatedDeltaNet"),
                "expected GatedDeltaNet in name, got {name}"
            );
            assert!(count > 0);
        }
        other => panic!("expected NonFiniteData, got {other:?}"),
    }
}

#[test]
fn test_gdn_inf_input_returns_error() {
    let (gdn, _) = make_test_gdn(1, 4, 2, 2, 2);
    let input = DynTensor::from_vec(
        vec![f32::INFINITY, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7],
        &[1, 2, 4],
        &Device::Cpu,
    )
    .unwrap();
    let result = gdn.forward(&input, None);
    assert!(result.is_err(), "Inf input should produce an error");
    let err = result.map(|_| ()).unwrap_err();
    assert!(
        matches!(err, TensorError::NonFiniteData { .. }),
        "expected NonFiniteData, got {err:?}"
    );
}

// -- VarBuilder load path tests -----------------------------------------------

#[test]
fn test_gdn_load_varbuilder_no_bias() {
    let dim = 4;
    let num_heads = 2;
    let key_dim = 2;
    let value_dim = 2;
    let qk_total = num_heads * key_dim;
    let v_total = num_heads * value_dim;
    let device = Device::Cpu;

    let mut tensors: HashMap<String, DynTensor> = HashMap::new();
    for prefix in &["q_proj", "k_proj"] {
        tensors.insert(
            format!("{prefix}.weight"),
            DynTensor::ones(&[qk_total, dim], DType::F32, &device).unwrap(),
        );
    }
    tensors.insert(
        "v_proj.weight".into(),
        DynTensor::ones(&[v_total, dim], DType::F32, &device).unwrap(),
    );
    for prefix in &["gate_proj", "beta_proj"] {
        tensors.insert(
            format!("{prefix}.weight"),
            DynTensor::ones(&[num_heads, dim], DType::F32, &device).unwrap(),
        );
    }
    tensors.insert(
        "out_proj.weight".into(),
        DynTensor::ones(&[dim, v_total], DType::F32, &device).unwrap(),
    );

    let vb = VarBuilder::from_tensors(tensors, DType::F32, &device);
    let gdn = GatedDeltaNet::load(&vb, dim, num_heads, key_dim, value_dim, false).unwrap();

    let input = DynTensor::from_vec(vec![0.1; dim * 2], &[1, 2, dim], &device).unwrap();
    let (output, state) = gdn.forward(&input, None).unwrap();
    assert_eq!(output.dims(), &[1, 2, dim]);
    assert_eq!(state.state.dims(), &[1, num_heads, key_dim, value_dim]);
}

#[test]
fn test_gdn_load_varbuilder_with_bias() {
    let dim = 4;
    let num_heads = 2;
    let key_dim = 2;
    let value_dim = 2;
    let qk_total = num_heads * key_dim;
    let v_total = num_heads * value_dim;
    let device = Device::Cpu;

    let mut tensors: HashMap<String, DynTensor> = HashMap::new();
    for prefix in &["q_proj", "k_proj"] {
        tensors.insert(
            format!("{prefix}.weight"),
            DynTensor::ones(&[qk_total, dim], DType::F32, &device).unwrap(),
        );
        tensors.insert(
            format!("{prefix}.bias"),
            DynTensor::zeros(&[qk_total], DType::F32, &device).unwrap(),
        );
    }
    tensors.insert(
        "v_proj.weight".into(),
        DynTensor::ones(&[v_total, dim], DType::F32, &device).unwrap(),
    );
    tensors.insert(
        "v_proj.bias".into(),
        DynTensor::zeros(&[v_total], DType::F32, &device).unwrap(),
    );
    for prefix in &["gate_proj", "beta_proj"] {
        tensors.insert(
            format!("{prefix}.weight"),
            DynTensor::ones(&[num_heads, dim], DType::F32, &device).unwrap(),
        );
        tensors.insert(
            format!("{prefix}.bias"),
            DynTensor::zeros(&[num_heads], DType::F32, &device).unwrap(),
        );
    }
    tensors.insert(
        "out_proj.weight".into(),
        DynTensor::ones(&[dim, v_total], DType::F32, &device).unwrap(),
    );
    tensors.insert(
        "out_proj.bias".into(),
        DynTensor::zeros(&[dim], DType::F32, &device).unwrap(),
    );

    let vb = VarBuilder::from_tensors(tensors, DType::F32, &device);
    let gdn = GatedDeltaNet::load(&vb, dim, num_heads, key_dim, value_dim, true).unwrap();

    let input = DynTensor::from_vec(vec![0.1; dim * 2], &[1, 2, dim], &device).unwrap();
    let (output, _) = gdn.forward(&input, None).unwrap();
    assert_eq!(output.dims(), &[1, 2, dim]);
    let out_data = output.to_flat_vec::<f32>().unwrap();
    assert!(out_data.iter().all(|v| v.is_finite()));
}

#[test]
fn test_gdn_load_zero_heads_returns_error() {
    let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
    let result = GatedDeltaNet::load(&vb, 4, 0, 2, 2, false);
    assert!(result.is_err());
    let err = result.unwrap_err().to_string();
    assert!(
        err.contains("num_heads"),
        "expected num_heads error, got: {err}"
    );
}

// -- Issue #3550 required tests -----------------------------------------------

/// Forward shape: [2, 4, 64] input -> [2, 4, 64] output (issue #3550 gate 1).
#[test]
fn test_gdn_forward_shape_2_4_64() {
    let device = Device::Cpu;
    let dim = 64;
    let num_heads = 4;
    let key_dim = 16;
    let value_dim = 16;
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

    let input = DynTensor::from_vec(vec![0.01; 2 * 4 * dim], &[2, 4, dim], &device).unwrap();

    let (output, state) = gdn.forward(&input, None).unwrap();
    assert_eq!(
        output.dims(),
        &[2, 4, 64],
        "output shape must be [2, 4, 64]"
    );
    assert_eq!(
        state.state.dims(),
        &[2, num_heads, key_dim, value_dim],
        "state shape must be [2, H, K, V]"
    );
    let flat = output.to_flat_vec::<f32>().unwrap();
    assert!(flat.iter().all(|v| v.is_finite()), "output must be finite");
}

/// Zero input with zero-bias weights produces zero output (issue #3550 gate 3).
#[test]
fn test_gdn_zero_input_zero_output() {
    let device = Device::Cpu;
    let dim = 8;
    let num_heads = 2;
    let key_dim = 4;
    let value_dim = 4;
    let qk_total = num_heads * key_dim;
    let v_total = num_heads * value_dim;

    // Use small random-ish weights (not zero weights, since the projections
    // being zero would trivially zero everything regardless of input).
    // With zero input AND zero bias, all projections produce zero -> zero output.
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

    // Zero input, zero initial state, no bias -> output must be zero.
    let input = DynTensor::zeros(&[1, 3, dim], DType::F32, &device).unwrap();
    let (output, state) = gdn.forward(&input, None).unwrap();

    let out_data = output.to_flat_vec::<f32>().unwrap();
    assert!(
        out_data.iter().all(|v| v.abs() < 1e-7),
        "zero input with zero bias should produce zero output, got: {:?}",
        &out_data[..out_data.len().min(8)]
    );

    // State should also be zero (no information written).
    let state_data = state.state.to_flat_vec::<f32>().unwrap();
    assert!(
        state_data.iter().all(|v| v.abs() < 1e-7),
        "zero input should leave state at zero"
    );
}

/// Multi-step dependency: output at step t depends on all previous inputs
/// (issue #3550 gate 4).
#[test]
fn test_gdn_multi_step_dependency() {
    let device = Device::Cpu;
    let dim = 4;
    let num_heads = 2;
    let key_dim = 2;
    let value_dim = 2;
    let qk_total = num_heads * key_dim;
    let v_total = num_heads * value_dim;

    // Non-degenerate weights for clear signal propagation.
    let make_w = |rows: usize, cols: usize, seed: f32| -> DynTensor {
        let data: Vec<f32> = (0..rows * cols)
            .map(|i| ((i as f32 + seed) * 0.13).sin() * 0.3)
            .collect();
        DynTensor::from_vec(data, &[rows, cols], &device).unwrap()
    };

    let gdn = GatedDeltaNet::new(
        Linear::new(make_w(qk_total, dim, 0.0), None).unwrap(),
        Linear::new(make_w(qk_total, dim, 1.0), None).unwrap(),
        Linear::new(make_w(v_total, dim, 2.0), None).unwrap(),
        Linear::new(make_w(num_heads, dim, 3.0), None).unwrap(),
        Linear::new(make_w(num_heads, dim, 4.0), None).unwrap(),
        Linear::new(make_w(dim, v_total, 5.0), None).unwrap(),
        num_heads,
        key_dim,
        value_dim,
    )
    .unwrap();

    // Sequence A: [x1, x2, x3]
    let x1 = [0.5, -0.3, 0.1, 0.4];
    let x2 = [0.2, 0.7, -0.1, 0.3];
    let x3 = [-0.4, 0.1, 0.6, -0.2];

    // Sequence B: [x1, x2_alt, x3]  (differs at step 2)
    let x2_alt = [0.9, -0.5, 0.3, -0.1];

    let seq_a_data: Vec<f32> = [&x1[..], &x2[..], &x3[..]].concat();
    let seq_b_data: Vec<f32> = [&x1[..], &x2_alt[..], &x3[..]].concat();

    let seq_a = DynTensor::from_vec(seq_a_data, &[1, 3, dim], &device).unwrap();
    let seq_b = DynTensor::from_vec(seq_b_data, &[1, 3, dim], &device).unwrap();

    let (out_a, _) = gdn.forward(&seq_a, None).unwrap();
    let (out_b, _) = gdn.forward(&seq_b, None).unwrap();

    let a_data = out_a.to_flat_vec::<f32>().unwrap();
    let b_data = out_b.to_flat_vec::<f32>().unwrap();

    // Step 1 outputs (index 0..dim) should be identical: same x1, same zero state.
    let step1_same = a_data[..dim]
        .iter()
        .zip(&b_data[..dim])
        .all(|(a, b)| (a - b).abs() < 1e-6);
    assert!(
        step1_same,
        "step 1 should be identical for both sequences (same input, same initial state)"
    );

    // Step 3 outputs (index 2*dim..3*dim) should differ: recurrent state
    // carries information from step 2, which was different.
    let step3_differs = a_data[2 * dim..3 * dim]
        .iter()
        .zip(&b_data[2 * dim..3 * dim])
        .any(|(a, b)| (a - b).abs() > 1e-6);
    assert!(
        step3_differs,
        "step 3 output must differ because step 2 input was different \
         (recurrent state carries prior information). \
         seq_a step3: {:?}, seq_b step3: {:?}",
        &a_data[2 * dim..3 * dim],
        &b_data[2 * dim..3 * dim]
    );
}

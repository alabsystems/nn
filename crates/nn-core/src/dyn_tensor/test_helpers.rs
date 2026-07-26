// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Shared test helpers for DynTensor test modules.
//!
//! Consolidates `cpu()`, `t1d()`, `t2d()`, `tnd()`, `approx_eq()`, and
//! `assert_close()` that were previously duplicated across 15+ test files.

use crate::dyn_tensor::DynTensor;
use crate::layers::Linear;
use crate::Device;

/// Convenience shorthand for `Device::Cpu`.
pub fn cpu() -> Device {
    Device::Cpu
}

/// Create a 1-D f32 tensor on CPU from a slice.
pub fn t1d(data: &[f32]) -> DynTensor {
    DynTensor::from_vec(data.to_vec(), &[data.len()], &cpu())
        .expect("invariant: valid 1D test data")
}

/// Create a 2-D f32 tensor on CPU from a flat slice with given rows × cols.
pub fn t2d(data: &[f32], rows: usize, cols: usize) -> DynTensor {
    DynTensor::from_vec(data.to_vec(), &[rows, cols], &cpu())
        .expect("invariant: valid 2D test data")
}

/// Create an N-D f32 tensor on CPU from a flat slice with given dims.
pub fn tnd(data: &[f32], dims: &[usize]) -> DynTensor {
    DynTensor::from_vec(data.to_vec(), dims, &cpu()).expect("invariant: valid test data")
}

/// Create a [`Linear`] layer with deterministic sequential weights (no bias).
///
/// Weights are `[out_features, in_features]` with values `i * 0.01`.
pub fn make_linear(out_features: usize, in_features: usize) -> Linear {
    let data: Vec<f32> = (0..out_features * in_features)
        .map(|i| (i as f32) * 0.01)
        .collect();
    let weight = DynTensor::from_vec(data, &[out_features, in_features], &cpu())
        .expect("invariant: valid linear weight");
    Linear::new(weight, None).expect("invariant: valid linear layer")
}

/// Create a [`Linear`] layer with deterministic sequential weights and bias.
///
/// Weights are `[out_features, in_features]` with values `i * 0.01`.
/// Bias is `[out_features]` with values `i * 0.1`.
pub fn make_linear_with_bias(out_features: usize, in_features: usize) -> Linear {
    let data: Vec<f32> = (0..out_features * in_features)
        .map(|i| (i as f32) * 0.01)
        .collect();
    let weight = DynTensor::from_vec(data, &[out_features, in_features], &cpu())
        .expect("invariant: valid linear weight");
    let bias_data: Vec<f32> = (0..out_features).map(|i| (i as f32) * 0.1).collect();
    let bias = DynTensor::from_vec(bias_data, &[out_features], &cpu())
        .expect("invariant: valid linear bias");
    Linear::new(weight, Some(bias)).expect("invariant: valid linear layer")
}

/// Create a [`Linear`] layer with seeded deterministic weights (no bias).
///
/// Weights are `[out_features, in_features]` with values `sin((i + seed) * 0.01) * 0.1`.
/// Different seeds produce different weight matrices, useful for constructing multi-projection
/// models (Q/K/V/O in attention).
pub fn make_linear_seeded(out_features: usize, in_features: usize, seed: f32) -> Linear {
    let n = out_features * in_features;
    let data: Vec<f32> = (0..n)
        .map(|i| ((i as f32 + seed) * 0.01).sin() * 0.1)
        .collect();
    let weight = DynTensor::from_vec(data, &[out_features, in_features], &cpu())
        .expect("invariant: valid linear weight");
    Linear::new(weight, None).expect("invariant: valid linear layer")
}

/// Create a [`Linear`] layer with seeded deterministic weights and bias.
///
/// Weights use `sin((i + seed) * 0.01) * 0.1`, bias uses `cos(i * 0.1 + seed) * 0.01`.
pub fn make_linear_seeded_with_bias(out_features: usize, in_features: usize, seed: f32) -> Linear {
    let n = out_features * in_features;
    let w_data: Vec<f32> = (0..n)
        .map(|i| ((i as f32 + seed) * 0.01).sin() * 0.1)
        .collect();
    let b_data: Vec<f32> = (0..out_features)
        .map(|i| (i as f32 * 0.1 + seed).cos() * 0.01)
        .collect();
    let weight = DynTensor::from_vec(w_data, &[out_features, in_features], &cpu())
        .expect("invariant: valid linear weight");
    let bias =
        DynTensor::from_vec(b_data, &[out_features], &cpu()).expect("invariant: valid linear bias");
    Linear::new(weight, Some(bias)).expect("invariant: valid linear layer")
}

/// Scalar approximate equality check.
pub fn approx_eq(a: f32, b: f32, tol: f32) -> bool {
    (a - b).abs() <= tol
}

/// Assert two f32 slices are element-wise close within tolerance.
///
/// Panics with a detailed message showing the first mismatch position.
pub fn assert_close(a: &[f32], b: &[f32], tol: f32) {
    assert_close_with_label(a, b, tol, "");
}

/// Assert two f32 slices are element-wise close, with a diagnostic label.
///
/// Same as [`assert_close`] but prefixes assertion messages with `label`
/// for easier identification in multi-assertion tests.
pub fn assert_close_with_label(a: &[f32], b: &[f32], tol: f32, label: &str) {
    assert_eq!(
        a.len(),
        b.len(),
        "{label}: length mismatch: {} vs {}",
        a.len(),
        b.len()
    );
    for (i, (&av, &bv)) in a.iter().zip(b.iter()).enumerate() {
        assert!(
            (av - bv).abs() <= tol,
            "{label}[{i}]: {av} vs {bv} (diff={}, tol={tol})",
            (av - bv).abs()
        );
    }
}

/// Assert two f64 values are close within tolerance, with a diagnostic label.
///
/// Scalar version for bounds verification tests.
pub fn assert_close_scalar_f64(actual: f64, expected: f64, tol: f64, label: &str) {
    assert!(
        (actual - expected).abs() <= tol,
        "{label}: actual={actual}, expected={expected} (diff={}, tol={tol})",
        (actual - expected).abs()
    );
}

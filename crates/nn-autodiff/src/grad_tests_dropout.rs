#![allow(deprecated)]
// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Tests for dropout forward and backward.

use std::sync::Arc;

use nn_core::dyn_tensor::DynTensor;
use nn_core::Device;

use crate::backward;
use crate::op::Op;
use crate::tracked::TrackedTensor;
use crate::var::Var;

#[test]
fn test_dropout_zero_p_is_identity() {
    let var = Var::new(DynTensor::from_vec(vec![1.0, 2.0, 3.0], &[3], &Device::Cpu).unwrap());
    let t = Arc::new(TrackedTensor::from_var(&var).unwrap());
    let out = t.dropout(0.0).unwrap();
    // p=0 returns Arc::clone(self), no Op recorded
    assert!(out.op().is_none() || out.var_id().is_some());
    let vals = out.tensor().to_flat_vec::<f32>().unwrap();
    assert_eq!(vals, vec![1.0, 2.0, 3.0]);
}

#[test]
fn test_dropout_invalid_p_negative() {
    let var = Var::new(DynTensor::from_vec(vec![1.0], &[1], &Device::Cpu).unwrap());
    let t = Arc::new(TrackedTensor::from_var(&var).unwrap());
    assert!(t.dropout(-0.1).is_err());
}

#[test]
fn test_dropout_invalid_p_one() {
    let var = Var::new(DynTensor::from_vec(vec![1.0], &[1], &Device::Cpu).unwrap());
    let t = Arc::new(TrackedTensor::from_var(&var).unwrap());
    assert!(t.dropout(1.0).is_err());
}

#[test]
fn test_dropout_invalid_p_greater_than_one() {
    let var = Var::new(DynTensor::from_vec(vec![1.0], &[1], &Device::Cpu).unwrap());
    let t = Arc::new(TrackedTensor::from_var(&var).unwrap());
    assert!(t.dropout(1.5).is_err());
}

#[test]
fn test_dropout_invalid_p_nan() {
    let var = Var::new(DynTensor::from_vec(vec![1.0], &[1], &Device::Cpu).unwrap());
    let t = Arc::new(TrackedTensor::from_var(&var).unwrap());
    assert!(t.dropout(f64::NAN).is_err());
}

#[test]
fn test_dropout_output_shape_preserved() {
    let var = Var::new(DynTensor::from_vec(vec![1.0; 12], &[3, 4], &Device::Cpu).unwrap());
    let t = Arc::new(TrackedTensor::from_var(&var).unwrap());
    let out = t.dropout(0.5).unwrap();
    assert_eq!(out.tensor().dims(), &[3, 4]);
}

#[test]
fn test_dropout_values_scaled_or_zero() {
    // With inverted dropout, each output element is either 0 or input * scale
    let input_vals = vec![2.0; 100];
    let var = Var::new(DynTensor::from_vec(input_vals, &[100], &Device::Cpu).unwrap());
    let t = Arc::new(TrackedTensor::from_var(&var).unwrap());
    let p = 0.5;
    let scale = 1.0 / (1.0 - p);
    let out = t.dropout(p).unwrap();
    let vals = out.tensor().to_flat_vec::<f32>().unwrap();
    for &v in &vals {
        // Each value is either 0.0 (dropped) or 2.0 * scale = 4.0 (kept)
        assert!(
            (v - 0.0).abs() < 1e-6 || (v - 2.0 * scale as f32).abs() < 1e-4,
            "unexpected dropout output: {v}"
        );
    }
}

#[test]
fn test_dropout_backward_gradient_flow() {
    // Dropout backward: grad_input = upstream_grad * mask * scale
    // For elements that were kept: grad = upstream * scale
    // For elements that were dropped: grad = 0
    let var = Var::new(DynTensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[4], &Device::Cpu).unwrap());
    let t = Arc::new(TrackedTensor::from_var(&var).unwrap());
    let dropped = t.dropout(0.5).unwrap();
    // Sum to scalar for backward
    let loss = dropped.sum_keepdim(0).unwrap();
    let grads = backward(&loss).unwrap();
    let grad = grads.get(&var).unwrap().to_flat_vec::<f32>().unwrap();
    let scale = 1.0 / (1.0 - 0.5);
    // Each gradient element is either 0 (dropped) or scale (kept)
    for &g in &grad {
        assert!(
            (g - 0.0).abs() < 1e-6 || (g - scale as f32).abs() < 1e-4,
            "unexpected dropout gradient: {g}"
        );
    }
}

#[test]
fn test_dropout_backward_chain_rule() {
    // Test that dropout correctly chains with other ops:
    // y = dropout(x^2, p=0.3) -> loss = sum(y)
    // grad_y = 1 (from sum)
    // grad_dropout_input = 1 * mask * scale
    // grad_x = grad_dropout_input * 2x = mask * scale * 2x
    let var = Var::new(DynTensor::from_vec(vec![1.0, 2.0, 3.0], &[3], &Device::Cpu).unwrap());
    let t = Arc::new(TrackedTensor::from_var(&var).unwrap());
    let squared = t.sqr().unwrap();
    let dropped = squared.dropout(0.3).unwrap();
    let loss = dropped.sum_keepdim(0).unwrap();
    let grads = backward(&loss).unwrap();
    let grad = grads.get(&var).unwrap().to_flat_vec::<f32>().unwrap();
    let scale = 1.0 / (1.0 - 0.3);
    let input_vals = [1.0_f32, 2.0, 3.0];
    // Each gradient is either 0 (dropped) or 2*x*scale (kept)
    for (i, &g) in grad.iter().enumerate() {
        let expected_kept = 2.0 * input_vals[i] * scale as f32;
        assert!(
            (g - 0.0).abs() < 1e-5 || (g - expected_kept).abs() < 1e-3,
            "element {i}: unexpected grad {g}, expected 0 or {expected_kept}"
        );
    }
}

#[test]
fn test_dropout_op_debug() {
    let var = Var::new(DynTensor::from_vec(vec![1.0], &[1], &Device::Cpu).unwrap());
    let t = Arc::new(TrackedTensor::from_var(&var).unwrap());
    let out = t.dropout(0.5).unwrap();
    let debug = format!("{:?}", out.op().unwrap());
    assert!(debug.starts_with("Dropout(scale="), "got: {debug}");
}

#[test]
fn test_dropout_expected_value_preserved() {
    // Statistical test: with inverted dropout, E[output] ≈ E[input]
    // Use a large tensor to get close to expected value
    let n = 10000;
    let val = 5.0_f32;
    let input_vals = vec![val; n];
    let var = Var::new(DynTensor::from_vec(input_vals, &[n], &Device::Cpu).unwrap());
    let t = Arc::new(TrackedTensor::from_var(&var).unwrap());
    let out = t.dropout(0.3).unwrap();
    let vals = out.tensor().to_flat_vec::<f32>().unwrap();
    let mean: f32 = vals.iter().sum::<f32>() / n as f32;
    // Expected mean ≈ 5.0 (inverted dropout preserves expectation)
    // Allow generous tolerance for randomness
    assert!(
        (mean - val).abs() < 1.0,
        "expected mean ≈ {val}, got {mean}"
    );
}

/// Finite-difference gradient test for dropout.
///
/// Standard FD testing (`f(x+eps) - f(x-eps) / 2eps`) requires a fixed mask
/// because dropout generates a new random mask on each forward pass.
/// We extract the mask from the first forward pass and manually apply it
/// to the perturbed inputs to compute the numerical gradient.
#[test]
fn test_dropout_fd_gradient() {
    let data = vec![1.0_f32, -2.0, 3.0, 0.5, -1.5, 2.5, -0.3, 4.0];
    let n = data.len();
    let p = 0.4;
    let var = Var::new(DynTensor::from_vec(data.clone(), &[n], &Device::Cpu).unwrap());
    let t = Arc::new(TrackedTensor::from_var(&var).unwrap());
    let dropped = t.dropout(p).unwrap();

    // Extract mask and scale from the Op
    let (mask_vals, scale) = match dropped.op().unwrap() {
        Op::Dropout(_, mask, s) => {
            let m = mask.tensor().to_flat_vec::<f32>().unwrap();
            (m, *s)
        }
        other => panic!("expected Op::Dropout, got {other:?}"),
    };

    // Compute analytical gradients via backward
    let loss = dropped.sum_keepdim(0).unwrap();
    let grads = backward(&loss).unwrap();
    let analytical = grads.get(&var).unwrap().to_flat_vec::<f32>().unwrap();

    // Compute numerical gradients using the captured mask
    let eps = 1e-3_f32;
    for i in 0..n {
        let mut plus = data.clone();
        let mut minus = data.clone();
        plus[i] += eps;
        minus[i] -= eps;

        // Apply the same mask * scale to perturbed inputs
        let f_plus: f64 = plus
            .iter()
            .zip(&mask_vals)
            .map(|(&x, &m)| f64::from(x) * f64::from(m) * scale)
            .sum();
        let f_minus: f64 = minus
            .iter()
            .zip(&mask_vals)
            .map(|(&x, &m)| f64::from(x) * f64::from(m) * scale)
            .sum();
        let numerical = (f_plus - f_minus) / (2.0 * f64::from(eps));
        let err = (f64::from(analytical[i]) - numerical).abs();
        assert!(
            err < 1e-2,
            "grad[{i}]: analytical={}, numerical={numerical}, err={err}",
            analytical[i]
        );
    }
}

/// FD gradient test for dropout chained with a nonlinear function (x^2).
///
/// Verifies chain-rule correctness: d/dx[dropout(x^2)] = mask * scale * 2x.
#[test]
fn test_dropout_chain_sqr_fd_gradient() {
    let data = vec![1.0_f32, -1.5, 2.0, -0.5];
    let n = data.len();
    let p = 0.3;
    let var = Var::new(DynTensor::from_vec(data.clone(), &[n], &Device::Cpu).unwrap());
    let t = Arc::new(TrackedTensor::from_var(&var).unwrap());
    let squared = t.sqr().unwrap();
    let dropped = squared.dropout(p).unwrap();

    // Extract mask from the dropout op
    let (mask_vals, scale) = match dropped.op().unwrap() {
        Op::Dropout(_, mask, s) => {
            let m = mask.tensor().to_flat_vec::<f32>().unwrap();
            (m, *s)
        }
        other => panic!("expected Op::Dropout, got {other:?}"),
    };

    // Analytical gradients via backward
    let loss = dropped.sum_keepdim(0).unwrap();
    let grads = backward(&loss).unwrap();
    let analytical = grads.get(&var).unwrap().to_flat_vec::<f32>().unwrap();

    // Numerical gradients with captured mask applied to x^2
    let eps = 1e-3_f32;
    for i in 0..n {
        let mut plus = data.clone();
        let mut minus = data.clone();
        plus[i] += eps;
        minus[i] -= eps;

        let f_plus: f64 = plus
            .iter()
            .zip(&mask_vals)
            .map(|(&x, &m)| f64::from(x * x) * f64::from(m) * scale)
            .sum();
        let f_minus: f64 = minus
            .iter()
            .zip(&mask_vals)
            .map(|(&x, &m)| f64::from(x * x) * f64::from(m) * scale)
            .sum();
        let numerical = (f_plus - f_minus) / (2.0 * f64::from(eps));
        let err = (f64::from(analytical[i]) - numerical).abs();
        assert!(
            err < 1e-2,
            "grad[{i}]: analytical={}, numerical={numerical}, err={err}",
            analytical[i]
        );
    }
}

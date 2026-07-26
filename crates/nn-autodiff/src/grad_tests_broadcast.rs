#![allow(deprecated)]
// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Backward tests for broadcast gradient reduction in binary ops.
//!
//! When binary ops (Add, Sub, Mul, Div) operate on operands with different
//! shapes via broadcast, the backward pass must reduce gradients back to
//! each operand's original shape using `reduce_to_shape`.

use std::sync::Arc;

use nn_core::dyn_tensor::DynTensor;
use nn_core::test_utils::cpu;

use crate::grad::backward;
use crate::tracked::TrackedTensor;
use crate::var::Var;

#[test]
fn test_backward_add_broadcast() {
    // Bias addition pattern: x = [2, 4], bias = [1, 4]
    // y = x + bias => [2, 4] (bias broadcasts along dim 0)
    // loss = sum(y)
    // grad(x) = [2, 4] all 1s (no reduction needed)
    // grad(bias) = [1, 4] with values [2, 2, 2, 2] (sum over batch dim)
    let x_var = Var::new(
        DynTensor::from_vec(
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            &[2, 4],
            &cpu(),
        )
        .unwrap(),
    );
    let bias_var =
        Var::new(DynTensor::from_vec(vec![0.1, 0.2, 0.3, 0.4], &[1, 4], &cpu()).unwrap());
    let tx = Arc::new(TrackedTensor::from_var(&x_var).unwrap());
    let tb = Arc::new(TrackedTensor::from_var(&bias_var).unwrap());
    let y = tx.add(&tb).unwrap();
    let loss = y.sum_keepdim(0).unwrap().sum_keepdim(1).unwrap();
    let grads = backward(&loss).unwrap();

    // grad(x) should be [2, 4] all 1s
    let gx = grads.get(&x_var).unwrap();
    assert_eq!(gx.dims(), &[2, 4], "grad_x shape mismatch");
    let gx_vals = gx.to_flat_vec::<f32>().unwrap();
    for (i, &g) in gx_vals.iter().enumerate() {
        assert!((g - 1.0).abs() < 1e-5, "grad_x[{i}]: expected 1.0, got {g}");
    }

    // grad(bias) should be [1, 4] with values [2, 2, 2, 2]
    let gb = grads.get(&bias_var).unwrap();
    assert_eq!(
        gb.dims(),
        &[1, 4],
        "grad_bias shape mismatch: expected [1, 4] but got {:?} — \
         Add backward fails to reduce across broadcast dim",
        gb.dims()
    );
    let gb_vals = gb.to_flat_vec::<f32>().unwrap();
    for (i, &g) in gb_vals.iter().enumerate() {
        assert!(
            (g - 2.0).abs() < 1e-5,
            "grad_bias[{i}]: expected 2.0, got {g}"
        );
    }
}

#[test]
fn test_backward_sub_broadcast() {
    // x = [2, 3], offset = [1, 3]
    // y = x - offset => [2, 3]
    // loss = sum(y)
    // grad(offset) should be [1, 3] all -2 (negated, summed over batch dim)
    let x_var =
        Var::new(DynTensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], &cpu()).unwrap());
    let offset_var = Var::new(DynTensor::from_vec(vec![0.1, 0.2, 0.3], &[1, 3], &cpu()).unwrap());
    let tx = Arc::new(TrackedTensor::from_var(&x_var).unwrap());
    let to = Arc::new(TrackedTensor::from_var(&offset_var).unwrap());
    let y = tx.sub(&to).unwrap();
    let loss = y.sum_keepdim(0).unwrap().sum_keepdim(1).unwrap();
    let grads = backward(&loss).unwrap();

    let go = grads.get(&offset_var).unwrap();
    assert_eq!(
        go.dims(),
        &[1, 3],
        "grad_offset shape mismatch: expected [1, 3] but got {:?}",
        go.dims()
    );
    let go_vals = go.to_flat_vec::<f32>().unwrap();
    for (i, &g) in go_vals.iter().enumerate() {
        assert!(
            (g - (-2.0)).abs() < 1e-5,
            "grad_offset[{i}]: expected -2.0, got {g}"
        );
    }
}

#[test]
fn test_backward_mul_broadcast() {
    // Element-wise scale: x = [2, 3], scale = [1, 3]
    // y = x * scale => [2, 3]
    // loss = sum(y)
    // grad(scale) should be [1, 3] — sum of x along batch dim
    let x_var =
        Var::new(DynTensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], &cpu()).unwrap());
    let scale_var = Var::new(DynTensor::from_vec(vec![1.0, 1.0, 1.0], &[1, 3], &cpu()).unwrap());
    let tx = Arc::new(TrackedTensor::from_var(&x_var).unwrap());
    let ts = Arc::new(TrackedTensor::from_var(&scale_var).unwrap());
    let y = tx.mul(&ts).unwrap();
    let loss = y.sum_keepdim(0).unwrap().sum_keepdim(1).unwrap();
    let grads = backward(&loss).unwrap();

    // grad(scale): d(loss)/d(scale[j]) = sum_i(x[i,j]) = x[0,j] + x[1,j]
    //   = [1+4, 2+5, 3+6] = [5, 7, 9]
    let gs = grads.get(&scale_var).unwrap();
    assert_eq!(
        gs.dims(),
        &[1, 3],
        "grad_scale shape mismatch: expected [1, 3] but got {:?}",
        gs.dims()
    );
    let gs_vals = gs.to_flat_vec::<f32>().unwrap();
    let expected = [5.0, 7.0, 9.0];
    for (i, (&got, &exp)) in gs_vals.iter().zip(expected.iter()).enumerate() {
        assert!(
            (got - exp).abs() < 1e-5,
            "grad_scale[{i}]: expected {exp}, got {got}"
        );
    }
}

#[test]
fn test_backward_div_broadcast() {
    // x = [2, 2], divisor = [1, 2]
    // y = x / divisor => [2, 2]
    // loss = sum(y)
    //
    // x = [[2, 4], [6, 8]], divisor = [[1, 2]]
    // y = [[2, 2], [6, 4]], loss = 14
    //
    // grad(divisor) = sum over batch of -x / divisor^2
    //   batch 0: [-2/1, -4/4] = [-2, -1]
    //   batch 1: [-6/1, -8/4] = [-6, -2]
    //   sum: [-8, -3]
    let x_var = Var::new(DynTensor::from_vec(vec![2.0, 4.0, 6.0, 8.0], &[2, 2], &cpu()).unwrap());
    let div_var = Var::new(DynTensor::from_vec(vec![1.0, 2.0], &[1, 2], &cpu()).unwrap());
    let tx = Arc::new(TrackedTensor::from_var(&x_var).unwrap());
    let td = Arc::new(TrackedTensor::from_var(&div_var).unwrap());
    let y = tx.div(&td).unwrap();
    let loss = y.sum_keepdim(0).unwrap().sum_keepdim(1).unwrap();
    let grads = backward(&loss).unwrap();

    // grad(divisor) should be [1, 2] — reduced from [2, 2]
    let gd = grads.get(&div_var).unwrap();
    assert_eq!(
        gd.dims(),
        &[1, 2],
        "grad_divisor shape mismatch: expected [1, 2] but got {:?}",
        gd.dims()
    );
    let gd_vals = gd.to_flat_vec::<f32>().unwrap();
    assert!(
        (gd_vals[0] - (-8.0)).abs() < 1e-4,
        "grad_divisor[0]: expected -8.0, got {}",
        gd_vals[0]
    );
    assert!(
        (gd_vals[1] - (-3.0)).abs() < 1e-4,
        "grad_divisor[1]: expected -3.0, got {}",
        gd_vals[1]
    );
}

// Finite-difference broadcast backward tests extracted to separate file (#1453).
#[path = "grad_tests_broadcast_fd.rs"]
mod fd;

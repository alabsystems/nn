#![allow(deprecated)]
// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Conv1d backward pass tests: simple, stride=2 (remainder, even, padding), finite-difference.
//! Extracted from `grad_tests_extended.rs` for 500-line compliance.

use std::sync::Arc;

use nn_core::dyn_tensor::DynTensor;
use nn_core::test_utils::cpu;

use crate::grad::backward;
use crate::tracked::TrackedTensor;
use crate::var::Var;

// -- Conv1d backward test ----------------------------------------------------

#[test]
fn test_backward_conv1d_simple() {
    // input: [1, 1, 4] = [[[[1, 2, 3, 4]]]]
    // kernel: [1, 1, 2] = [[[[1, 1]]]]
    // padding=0, stride=1, dilation=1, groups=1
    // output: [1, 1, 3] = [[[[3, 5, 7]]]]  (1+2, 2+3, 3+4)
    // loss = sum(output) = 15
    //
    // grad_output = [[[1, 1, 1]]]
    // grad_input via conv_transpose1d: kernel flipped [1,1] * [1,1,1] => [1, 2, 2, 1]
    //   at padding=0, the result is [1, 2, 2, 1]
    // grad_kernel: correlation of input with grad_output
    //   gk[0,0,0] = sum(input * grad) at offset 0 = 1*1 + 2*1 + 3*1 = 6
    //   gk[0,0,1] = sum(input * grad) at offset 1 = 2*1 + 3*1 + 4*1 = 9
    let x_var =
        Var::new(DynTensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[1, 1, 4], &cpu()).unwrap());
    let k_var = Var::new(DynTensor::from_vec(vec![1.0, 1.0], &[1, 1, 2], &cpu()).unwrap());
    let tx = Arc::new(TrackedTensor::from_var(&x_var).unwrap());
    let tk = Arc::new(TrackedTensor::from_var(&k_var).unwrap());
    let y = tx.conv1d(&tk, 0, 1, 1, 1).unwrap();
    let loss = y
        .sum_keepdim(0)
        .unwrap()
        .sum_keepdim(1)
        .unwrap()
        .sum_keepdim(2)
        .unwrap();
    let grads = backward(&loss).unwrap();

    let gx = grads.get(&x_var).unwrap().to_flat_vec::<f32>().unwrap();
    assert!(
        (gx[0] - 1.0).abs() < 1e-5
            && (gx[1] - 2.0).abs() < 1e-5
            && (gx[2] - 2.0).abs() < 1e-5
            && (gx[3] - 1.0).abs() < 1e-5,
        "grad_input: expected [1, 2, 2, 1], got {gx:?}"
    );

    let gk = grads.get(&k_var).unwrap().to_flat_vec::<f32>().unwrap();
    assert!(
        (gk[0] - 6.0).abs() < 1e-5 && (gk[1] - 9.0).abs() < 1e-5,
        "grad_kernel: expected [6, 9], got {gk:?}"
    );
}

/// Regression test for #1024: stride=2 with odd input length (remainder=1).
#[test]
fn test_backward_conv1d_stride2_remainder() {
    let x_var =
        Var::new(DynTensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0], &[1, 1, 5], &cpu()).unwrap());
    let k_var = Var::new(DynTensor::from_vec(vec![1.0, 1.0], &[1, 1, 2], &cpu()).unwrap());
    let tx = Arc::new(TrackedTensor::from_var(&x_var).unwrap());
    let tk = Arc::new(TrackedTensor::from_var(&k_var).unwrap());
    let y = tx.conv1d(&tk, 0, 2, 1, 1).unwrap();
    let loss = y
        .sum_keepdim(0)
        .unwrap()
        .sum_keepdim(1)
        .unwrap()
        .sum_keepdim(2)
        .unwrap();
    let grads = backward(&loss).unwrap();
    let gx = grads.get(&x_var).unwrap().to_flat_vec::<f32>().unwrap();
    assert_eq!(
        gx.len(),
        5,
        "grad_input should have length 5, got {}",
        gx.len()
    );
    assert!(
        (gx[0] - 1.0).abs() < 1e-5
            && (gx[1] - 1.0).abs() < 1e-5
            && (gx[2] - 1.0).abs() < 1e-5
            && (gx[3] - 1.0).abs() < 1e-5
            && gx[4].abs() < 1e-5,
        "grad_input: expected [1, 1, 1, 1, 0], got {gx:?}"
    );
}

/// Regression test for #1024: stride=2 with even input length (no remainder).
#[test]
fn test_backward_conv1d_stride2_even() {
    let x_var =
        Var::new(DynTensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[1, 1, 4], &cpu()).unwrap());
    let k_var = Var::new(DynTensor::from_vec(vec![1.0, 1.0], &[1, 1, 2], &cpu()).unwrap());
    let tx = Arc::new(TrackedTensor::from_var(&x_var).unwrap());
    let tk = Arc::new(TrackedTensor::from_var(&k_var).unwrap());
    let y = tx.conv1d(&tk, 0, 2, 1, 1).unwrap();
    let loss = y
        .sum_keepdim(0)
        .unwrap()
        .sum_keepdim(1)
        .unwrap()
        .sum_keepdim(2)
        .unwrap();
    let grads = backward(&loss).unwrap();
    let gx = grads.get(&x_var).unwrap().to_flat_vec::<f32>().unwrap();
    assert_eq!(
        gx.len(),
        4,
        "grad_input should have length 4, got {}",
        gx.len()
    );
}

/// Regression test for #1024: stride=2 with padding=1.
#[test]
fn test_backward_conv1d_stride2_padding1() {
    let x_var = Var::new(
        DynTensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[1, 1, 6], &cpu()).unwrap(),
    );
    let k_var = Var::new(DynTensor::from_vec(vec![1.0, 1.0, 1.0], &[1, 1, 3], &cpu()).unwrap());
    let tx = Arc::new(TrackedTensor::from_var(&x_var).unwrap());
    let tk = Arc::new(TrackedTensor::from_var(&k_var).unwrap());
    let y = tx.conv1d(&tk, 1, 2, 1, 1).unwrap();
    let loss = y
        .sum_keepdim(0)
        .unwrap()
        .sum_keepdim(1)
        .unwrap()
        .sum_keepdim(2)
        .unwrap();
    let grads = backward(&loss).unwrap();
    let gx = grads.get(&x_var).unwrap().to_flat_vec::<f32>().unwrap();
    assert_eq!(
        gx.len(),
        6,
        "grad_input should have length 6, got {}",
        gx.len()
    );
}

/// Finite-difference validation for conv1d backward with stride=2.
#[test]
fn test_backward_conv1d_stride2_finite_diff() {
    let x_data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0];
    let k_data = vec![0.5f32, -0.3];
    let eps = 1e-4;
    let x_var = Var::new(DynTensor::from_vec(x_data.clone(), &[1, 1, 5], &cpu()).unwrap());
    let k_var = Var::new(DynTensor::from_vec(k_data.clone(), &[1, 1, 2], &cpu()).unwrap());
    let tx = Arc::new(TrackedTensor::from_var(&x_var).unwrap());
    let tk = Arc::new(TrackedTensor::from_var(&k_var).unwrap());
    let y = tx.conv1d(&tk, 0, 2, 1, 1).unwrap();
    let loss = y
        .sum_keepdim(0)
        .unwrap()
        .sum_keepdim(1)
        .unwrap()
        .sum_keepdim(2)
        .unwrap();
    let grads = backward(&loss).unwrap();
    let analytical_gx = grads.get(&x_var).unwrap().to_flat_vec::<f32>().unwrap();
    for i in 0..x_data.len() {
        let mut x_plus = x_data.clone();
        let mut x_minus = x_data.clone();
        x_plus[i] += eps;
        x_minus[i] -= eps;
        let forward = |data: Vec<f32>| -> f32 {
            let t = DynTensor::from_vec(data, &[1, 1, 5], &cpu()).unwrap();
            let k = DynTensor::from_vec(k_data.clone(), &[1, 1, 2], &cpu()).unwrap();
            let out = t.conv1d(&k, 0, 2, 1, 1).unwrap();
            out.to_flat_vec::<f32>().unwrap().iter().sum()
        };
        let numerical = (forward(x_plus) - forward(x_minus)) / (2.0 * eps);
        assert!(
            (analytical_gx[i] - numerical).abs() < 1e-3,
            "grad_input[{i}]: analytical={}, numerical={numerical}",
            analytical_gx[i]
        );
    }
}

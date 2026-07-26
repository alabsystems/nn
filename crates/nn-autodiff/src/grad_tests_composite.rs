#![allow(deprecated)]
// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Backward pass tests for composite operations: Softmax, LayerNorm, Embedding.
//! Conv2d tests are in `conv2d_fd` (grad_tests_conv2d_fd.rs) and
//! `conv2d_extra_fd` (grad_tests_conv2d_extra_fd.rs — groups + non-square).

use std::sync::Arc;

use nn_core::dyn_tensor::DynTensor;
use nn_core::test_utils::cpu;

use crate::grad::backward;
use crate::tracked::TrackedTensor;
use crate::var::Var;

// -- Softmax backward test ---------------------------------------------------

#[test]
fn test_backward_softmax() {
    // x = [1.0, 2.0], loss = softmax(x)[0]
    // grad_x[i] = s[i] * (grad[i] - sum(grad*s))
    //   grad_x[0] = s0*(1 - s0), grad_x[1] = -s0*s1
    let x = Var::new(DynTensor::from_vec(vec![1.0, 2.0], &[1, 2], &cpu()).unwrap());
    let t = Arc::new(TrackedTensor::from_var(&x).unwrap());
    let s = t.softmax(1).unwrap();
    let first = s.narrow(1, 0, 1).unwrap();
    let loss = first.sum_keepdim(0).unwrap().sum_keepdim(1).unwrap();
    let grads = backward(&loss).unwrap();
    let grad = grads.get(&x).unwrap().to_flat_vec::<f32>().unwrap();

    let s0 = 1.0_f32.exp() / (1.0_f32.exp() + 2.0_f32.exp());
    let s1 = 2.0_f32.exp() / (1.0_f32.exp() + 2.0_f32.exp());
    assert!((grad[0] - s0 * (1.0 - s0)).abs() < 1e-5, "softmax grad[0]");
    assert!((grad[1] - (-(s0 * s1))).abs() < 1e-5, "softmax grad[1]");
}

/// Finite-difference validation for softmax backward.
///
/// The softmax Jacobian formula is subtle (J = diag(s) - s*s^T) and
/// error-prone. This test perturbs each logit and verifies the autodiff
/// gradient matches the numerical gradient from the reference forward.
#[test]
fn test_backward_softmax_finite_diff() {
    let x_data = vec![0.5, 1.5, -0.3, 2.0, 0.1, -1.0];
    let eps = 1e-4_f32;

    // Reference forward: sum of softmax outputs (all elements)
    let forward = |d: Vec<f32>| -> f32 {
        let t = DynTensor::from_vec(d, &[2, 3], &cpu()).unwrap();
        // Use softmax then select element [0,1] to make a non-trivial loss
        let s = t.softmax(1).unwrap();
        // loss = s[0,0] + s[0,2] + s[1,1] (sparse selection)
        let flat = s.to_flat_vec::<f32>().unwrap();
        flat[0] + flat[2] + flat[4]
    };

    // Build computation graph
    let x_var = Var::new(DynTensor::from_vec(x_data.clone(), &[2, 3], &cpu()).unwrap());
    let tx = Arc::new(TrackedTensor::from_var(&x_var).unwrap());
    let sm = tx.softmax(1).unwrap();
    // Extract [0,0], [0,2], [1,1] and sum
    let row0 = sm.narrow(0, 0, 1).unwrap();
    let row1 = sm.narrow(0, 1, 1).unwrap();
    let e00 = row0.narrow(1, 0, 1).unwrap();
    let e02 = row0.narrow(1, 2, 1).unwrap();
    let e11 = row1.narrow(1, 1, 1).unwrap();
    let partial = e00.add(&e02).unwrap();
    let loss_t = partial.add(&e11).unwrap();
    let loss = loss_t.sum_keepdim(0).unwrap().sum_keepdim(1).unwrap();
    let grads = backward(&loss).unwrap();
    let analytical = grads.get(&x_var).unwrap().to_flat_vec::<f32>().unwrap();

    for i in 0..x_data.len() {
        let mut xp = x_data.clone();
        let mut xm = x_data.clone();
        xp[i] += eps;
        xm[i] -= eps;
        let numerical = (forward(xp) - forward(xm)) / (2.0 * eps);
        let err = (analytical[i] - numerical).abs();
        assert!(
            err < 1e-3,
            "softmax fd grad[{i}]: analytical={:.6}, numerical={:.6}, err={:.6}",
            analytical[i],
            numerical,
            err,
        );
    }
}

// -- LayerNorm backward test --------------------------------------------------

/// Forward layer_norm and return scalar loss (sum of all elements).
fn layer_norm_scalar_loss(x_vals: Vec<f32>) -> f32 {
    let t = DynTensor::from_vec(x_vals, &[2, 3], &cpu()).unwrap();
    let gamma = DynTensor::from_vec(vec![1.0, 1.0, 1.0], &[3], &cpu()).unwrap();
    let beta = DynTensor::from_vec(vec![0.0, 0.0, 0.0], &[3], &cpu()).unwrap();
    let mean = t.mean_keepdim(1).unwrap();
    let diff = t.sub(&mean).unwrap();
    let var = diff.sqr().unwrap().mean_keepdim(1).unwrap();
    let inv_std = var
        .add_scalar(1e-5)
        .unwrap()
        .sqrt()
        .unwrap()
        .recip()
        .unwrap();
    let normed = diff.mul(&inv_std).unwrap();
    normed
        .mul(&gamma)
        .unwrap()
        .add(&beta)
        .unwrap()
        .to_flat_vec::<f32>()
        .unwrap()
        .iter()
        .sum()
}

#[test]
fn test_backward_layer_norm() {
    // x = [[1, 2, 3], [4, 5, 6]], gamma = [1,1,1], beta = [0,0,0]
    let x_var =
        Var::new(DynTensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], &cpu()).unwrap());
    let gamma_var = Var::new(DynTensor::from_vec(vec![1.0, 1.0, 1.0], &[3], &cpu()).unwrap());
    let beta_var = Var::new(DynTensor::from_vec(vec![0.0, 0.0, 0.0], &[3], &cpu()).unwrap());

    let tx = Arc::new(TrackedTensor::from_var(&x_var).unwrap());
    let tg = Arc::new(TrackedTensor::from_var(&gamma_var).unwrap());
    let tb = Arc::new(TrackedTensor::from_var(&beta_var).unwrap());
    let y = tx.layer_norm(&tg, &tb, 1e-5).unwrap();
    let loss = y.sum_keepdim(0).unwrap().sum_keepdim(1).unwrap();
    let grads = backward(&loss).unwrap();

    // Check shapes
    assert_eq!(grads.get(&x_var).unwrap().dims(), &[2, 3]);
    assert_eq!(grads.get(&gamma_var).unwrap().dims(), &[3]);
    assert_eq!(grads.get(&beta_var).unwrap().dims(), &[3]);

    // grad(beta) = sum of grad over batch dim = [2, 2, 2]
    let gb_vals = grads.get(&beta_var).unwrap().to_flat_vec::<f32>().unwrap();
    for (i, &g) in gb_vals.iter().enumerate() {
        assert!(
            (g - 2.0).abs() < 1e-5,
            "grad_beta[{i}]: expected 2.0, got {g}"
        );
    }

    // Finite-difference check for grad_x
    let x_vals = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let gx_vals = grads.get(&x_var).unwrap().to_flat_vec::<f32>().unwrap();
    let eps = 1e-3_f32;
    for idx in 0..6 {
        let mut xp = x_vals.clone();
        xp[idx] += eps;
        let mut xm = x_vals.clone();
        xm[idx] -= eps;
        let fd = (layer_norm_scalar_loss(xp) - layer_norm_scalar_loss(xm)) / (2.0 * eps);
        assert!(
            (gx_vals[idx] - fd).abs() < 1e-3,
            "grad_x[{idx}]: analytical={:.6}, fd={:.6}",
            gx_vals[idx],
            fd
        );
    }
}

// -- Embedding backward test -------------------------------------------------

#[test]
fn test_backward_embedding() {
    // weight = [[1,2],[3,4],[5,6]] (vocab=3, embed=2)
    // indices = [0, 2, 0] => output = [[1,2],[5,6],[1,2]]
    // grad_weight[0] += [1,1] + [1,1] = [2,2], grad_weight[1] = [0,0], grad_weight[2] = [1,1]
    let w =
        Var::new(DynTensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[3, 2], &cpu()).unwrap());
    let idx_data = DynTensor::from_vec_u32(vec![0, 2, 0], &[3], &cpu()).unwrap();
    let tw = Arc::new(TrackedTensor::from_var(&w).unwrap());
    let ti = Arc::new(TrackedTensor::from_tensor(idx_data));
    let out = TrackedTensor::embedding(&tw, &ti).unwrap();
    let loss = out.sum_keepdim(0).unwrap().sum_keepdim(1).unwrap();
    let grads = backward(&loss).unwrap();
    let gw = grads.get(&w).unwrap().to_flat_vec::<f32>().unwrap();
    assert_eq!(gw, vec![2.0, 2.0, 0.0, 0.0, 1.0, 1.0]);
}

/// Finite-difference validation for embedding backward.
///
/// Perturbs each weight element and checks the autodiff gradient matches.
/// Embedding backward uses scatter-add for duplicate indices — this test
/// verifies correctness for repeated index lookups (indices [0, 2, 0]).
#[test]
fn test_backward_embedding_finite_diff() {
    let w_data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]; // [3, 2] vocab=3, embed=2
    let idx_flat = vec![0u32, 2, 0]; // 3 lookups, index 0 appears twice
    let idx_for_graph = idx_flat.clone();
    let eps = 1e-3_f32;

    // Reference forward: sum all embedding output elements
    let forward = |d: Vec<f32>| -> f32 {
        let w = DynTensor::from_vec(d, &[3, 2], &cpu()).unwrap();
        let idx = DynTensor::from_vec_u32(idx_flat.clone(), &[3], &cpu()).unwrap();
        w.index_select(&idx, 0)
            .unwrap()
            .to_flat_vec::<f32>()
            .unwrap()
            .iter()
            .sum()
    };

    let w_var = Var::new(DynTensor::from_vec(w_data.clone(), &[3, 2], &cpu()).unwrap());
    let idx_data = DynTensor::from_vec_u32(idx_for_graph, &[3], &cpu()).unwrap();
    let tw = Arc::new(TrackedTensor::from_var(&w_var).unwrap());
    let ti = Arc::new(TrackedTensor::from_tensor(idx_data));
    let out = TrackedTensor::embedding(&tw, &ti).unwrap();
    let loss = out.sum_keepdim(0).unwrap().sum_keepdim(1).unwrap();
    let grads = backward(&loss).unwrap();
    let analytical = grads.get(&w_var).unwrap().to_flat_vec::<f32>().unwrap();

    for i in 0..w_data.len() {
        let mut wp = w_data.clone();
        let mut wm = w_data.clone();
        wp[i] += eps;
        wm[i] -= eps;
        let numerical = (forward(wp) - forward(wm)) / (2.0 * eps);
        let err = (analytical[i] - numerical).abs();
        // Embedding uses index_select which introduces f32 rounding; use 1e-2
        assert!(
            err < 1e-2,
            "embedding fd grad_w[{i}]: analytical={:.6}, numerical={:.6}, err={:.6}",
            analytical[i],
            numerical,
            err,
        );
    }
}

// -- Embedding error-path tests (#1998) --------------------------------------

#[test]
fn test_embedding_1d_weight_returns_error() {
    // Weight must be at least 2D [vocab, embed_dim].
    let w_1d = DynTensor::from_vec(vec![1.0, 2.0, 3.0], &[3], &cpu()).unwrap();
    let idx = DynTensor::from_vec_u32(vec![0, 1], &[2], &cpu()).unwrap();
    let tw = Arc::new(TrackedTensor::from_tensor(w_1d));
    let ti = Arc::new(TrackedTensor::from_tensor(idx));
    let result = TrackedTensor::embedding(&tw, &ti);
    assert!(result.is_err(), "1D weight should be rejected");
    let msg = format!("{}", result.unwrap_err());
    assert!(
        msg.contains("2D"),
        "error should mention 2D requirement: {msg}"
    );
}

#[test]
fn test_embedding_zero_embed_dim_returns_error() {
    // Weight [3, 0] has embed_dim = 0, would cause div-by-zero in backward.
    let w_zero = DynTensor::zeros(&[3, 0], nn_core::DType::F32, &cpu()).unwrap();
    let idx = DynTensor::from_vec_u32(vec![0], &[1], &cpu()).unwrap();
    let tw = Arc::new(TrackedTensor::from_tensor(w_zero));
    let ti = Arc::new(TrackedTensor::from_tensor(idx));
    let result = TrackedTensor::embedding(&tw, &ti);
    assert!(result.is_err(), "zero embed_dim should be rejected");
    let msg = format!("{}", result.unwrap_err());
    assert!(
        msg.contains("embed_dim"),
        "error should mention embed_dim: {msg}"
    );
}

#[path = "grad_tests_conv2d_fd.rs"]
mod conv2d_fd;

#[path = "grad_tests_conv2d_extra_fd.rs"]
mod conv2d_extra_fd;

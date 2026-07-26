#![allow(deprecated)]
// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Finite-difference gradient tests for LSTM cell.
//!
//! The LSTM is decomposed into matmul, sigmoid, tanh, narrow, mul, add ops.
//! These FD tests verify that gradients flow correctly through the full
//! composition, catching potential errors in gate decomposition, narrow
//! backward, or state-update backward interactions.
//!
//! Re: #1098 (dvoice-integration: verified training).

use std::sync::Arc;

use nn_core::dyn_tensor::DynTensor;
use nn_core::test_utils::cpu;

use crate::grad::backward;
use crate::grad::test_helpers::{check_fd_grad, sum_f64};
use crate::tracked::TrackedTensor;
use crate::trainable::TrainableLstm;
use crate::var::Var;

/// Run a full LSTM cell forward, returning the scalar loss (sum of h_new).
///
/// This is the reference forward function used by FD perturbation.
fn lstm_cell_scalar_loss(
    x_data: &[f32],
    w_ih_data: &[f32],
    w_hh_data: &[f32],
    batch: usize,
    input_size: usize,
    hidden_size: usize,
) -> f64 {
    // Reconstruct tensors from flat data.
    let x = DynTensor::from_vec(x_data.to_vec(), &[batch, input_size], &cpu()).unwrap();
    let w_ih =
        DynTensor::from_vec(w_ih_data.to_vec(), &[4 * hidden_size, input_size], &cpu()).unwrap();
    let _w_hh =
        DynTensor::from_vec(w_hh_data.to_vec(), &[4 * hidden_size, hidden_size], &cpu()).unwrap();

    // Compute gates = x @ w_ih^T + h0 @ w_hh^T (h0 = zeros → h0 @ w_hh^T = 0).
    let w_ih_t = w_ih.transpose(0, 1).unwrap();
    let gates = x.matmul(&w_ih_t).unwrap();

    // Split gates: [batch, 4*H] → i, f, g, o each [batch, H].
    let hs = hidden_size;
    let i_gate = gates.narrow(1, 0, hs).unwrap();
    let f_gate = gates.narrow(1, hs, hs).unwrap();
    let g_gate = gates.narrow(1, 2 * hs, hs).unwrap();
    let o_gate = gates.narrow(1, 3 * hs, hs).unwrap();

    // Gate activations.
    let sigmoid = |t: DynTensor| -> DynTensor { t.sigmoid().unwrap() };
    let tanh_fn = |t: DynTensor| -> DynTensor { t.tanh().unwrap() };
    let i = sigmoid(i_gate);
    let _f = sigmoid(f_gate);
    let g = tanh_fn(g_gate);
    let o = sigmoid(o_gate);

    // State update: c_new = f * c0 + i * g (c0 = zeros → c_new = i * g).
    let c_new = i.mul(&g).unwrap();
    // Output: h_new = o * tanh(c_new).
    let h_new = o.mul(&c_new.tanh().unwrap()).unwrap();

    sum_f64(&h_new)
}

/// FD test for LSTM cell: verify grad_input (dL/dx).
///
/// Uses a single LSTM step from zero state (h0=0, c0=0) with batch=1,
/// input_size=2, hidden_size=2. Small enough for FD stability.
#[test]
fn test_backward_lstm_cell_grad_input_fd() {
    let batch = 1;
    let input_size = 2;
    let hidden_size = 2;

    // Small deterministic weights.
    let w_ih_data: Vec<f32> = vec![
        0.1, 0.2, // i gate row 0
        -0.3, 0.4, // i gate row 1
        0.5, -0.1, // f gate row 0
        0.2, 0.3, // f gate row 1
        -0.2, 0.6, // g gate row 0
        0.1, -0.4, // g gate row 1
        0.3, 0.2, // o gate row 0
        -0.1, 0.5, // o gate row 1
    ];
    let w_hh_data: Vec<f32> = vec![
        0.1, -0.1, 0.2, 0.1, -0.1, 0.3, 0.1, -0.2, 0.2, 0.1, -0.3, 0.1, 0.1, 0.2, -0.1, 0.3,
    ];
    let x_data: Vec<f32> = vec![0.5, -0.3];

    // Build TrainableLstm and run forward+backward.
    let lstm = TrainableLstm::from_tensors(
        DynTensor::from_vec(w_ih_data.clone(), &[4 * hidden_size, input_size], &cpu()).unwrap(),
        DynTensor::from_vec(w_hh_data.clone(), &[4 * hidden_size, hidden_size], &cpu()).unwrap(),
        None,
        None,
        hidden_size,
    );
    let x_var =
        Var::new(DynTensor::from_vec(x_data.clone(), &[batch, input_size], &cpu()).unwrap());
    let tx = Arc::new(TrackedTensor::from_var(&x_var).unwrap());
    let (h_new, _state) = lstm.forward_cell(&tx, None).unwrap();
    let loss = h_new.sum_keepdim(0).unwrap().sum_keepdim(1).unwrap();
    let grads = backward(&loss).unwrap();
    let analytical_x = grads.get(&x_var).unwrap().to_flat_vec::<f32>().unwrap();

    // FD verification.
    let eps = 1e-3_f32;
    check_fd_grad(&analytical_x, &x_data, eps, |d| {
        lstm_cell_scalar_loss(&d, &w_ih_data, &w_hh_data, batch, input_size, hidden_size)
    });
}

/// FD test for LSTM cell: verify grad_w_ih (dL/dW_ih).
///
/// Perturbs each element of w_ih and verifies the analytical gradient matches.
#[test]
fn test_backward_lstm_cell_grad_w_ih_fd() {
    let batch = 1;
    let input_size = 2;
    let hidden_size = 2;

    let w_ih_data: Vec<f32> = vec![
        0.1, 0.2, -0.3, 0.4, 0.5, -0.1, 0.2, 0.3, -0.2, 0.6, 0.1, -0.4, 0.3, 0.2, -0.1, 0.5,
    ];
    let w_hh_data: Vec<f32> = vec![
        0.1, -0.1, 0.2, 0.1, -0.1, 0.3, 0.1, -0.2, 0.2, 0.1, -0.3, 0.1, 0.1, 0.2, -0.1, 0.3,
    ];
    let x_data: Vec<f32> = vec![0.5, -0.3];

    let lstm = TrainableLstm::from_tensors(
        DynTensor::from_vec(w_ih_data.clone(), &[4 * hidden_size, input_size], &cpu()).unwrap(),
        DynTensor::from_vec(w_hh_data.clone(), &[4 * hidden_size, hidden_size], &cpu()).unwrap(),
        None,
        None,
        hidden_size,
    );
    let x_var =
        Var::new(DynTensor::from_vec(x_data.clone(), &[batch, input_size], &cpu()).unwrap());
    let tx = Arc::new(TrackedTensor::from_var(&x_var).unwrap());
    let (h_new, _state) = lstm.forward_cell(&tx, None).unwrap();
    let loss = h_new.sum_keepdim(0).unwrap().sum_keepdim(1).unwrap();
    let grads = backward(&loss).unwrap();
    let analytical_w_ih = grads
        .get(lstm.w_ih())
        .unwrap()
        .to_flat_vec::<f32>()
        .unwrap();

    let eps = 1e-3_f32;
    check_fd_grad(&analytical_w_ih, &w_ih_data, eps, |d| {
        lstm_cell_scalar_loss(&x_data, &d, &w_hh_data, batch, input_size, hidden_size)
    });
}

/// FD test for LSTM cell with non-zero initial state.
///
/// Verifies gradient correctness when h0 and c0 are non-zero,
/// exercising the forget gate (f * c_prev) and hidden-to-hidden (h @ w_hh^T)
/// backward paths that are trivial when state is zero.
#[test]
fn test_backward_lstm_cell_nonzero_state_fd() {
    let batch = 1;
    let input_size = 2;
    let hidden_size = 2;

    let w_ih_data: Vec<f32> = vec![
        0.1, 0.2, -0.3, 0.4, 0.5, -0.1, 0.2, 0.3, -0.2, 0.6, 0.1, -0.4, 0.3, 0.2, -0.1, 0.5,
    ];
    let w_hh_data: Vec<f32> = vec![
        0.1, -0.1, 0.2, 0.1, -0.1, 0.3, 0.1, -0.2, 0.2, 0.1, -0.3, 0.1, 0.1, 0.2, -0.1, 0.3,
    ];
    let x_data: Vec<f32> = vec![0.5, -0.3];
    let h0_data: Vec<f32> = vec![0.1, -0.2];
    let c0_data: Vec<f32> = vec![0.3, 0.4];

    // Build LSTM and run forward with non-zero state.
    let lstm = TrainableLstm::from_tensors(
        DynTensor::from_vec(w_ih_data.clone(), &[4 * hidden_size, input_size], &cpu()).unwrap(),
        DynTensor::from_vec(w_hh_data.clone(), &[4 * hidden_size, hidden_size], &cpu()).unwrap(),
        None,
        None,
        hidden_size,
    );
    let x_var =
        Var::new(DynTensor::from_vec(x_data.clone(), &[batch, input_size], &cpu()).unwrap());
    let tx = Arc::new(TrackedTensor::from_var(&x_var).unwrap());

    use crate::trainable::TrackedLstmState;
    let h0 = Arc::new(TrackedTensor::from_tensor(
        DynTensor::from_vec(h0_data.clone(), &[batch, hidden_size], &cpu()).unwrap(),
    ));
    let c0 = Arc::new(TrackedTensor::from_tensor(
        DynTensor::from_vec(c0_data.clone(), &[batch, hidden_size], &cpu()).unwrap(),
    ));
    let state = TrackedLstmState { h: h0, c: c0 };

    let (h_new, _) = lstm.forward_cell(&tx, Some(&state)).unwrap();
    let loss = h_new.sum_keepdim(0).unwrap().sum_keepdim(1).unwrap();
    let grads = backward(&loss).unwrap();
    let analytical_x = grads.get(&x_var).unwrap().to_flat_vec::<f32>().unwrap();

    // FD needs a reference forward that includes h0, c0.
    let eps = 1e-3_f32;
    check_fd_grad(&analytical_x, &x_data, eps, |d| {
        lstm_cell_with_state_loss(
            &d,
            &w_ih_data,
            &w_hh_data,
            &h0_data,
            &c0_data,
            batch,
            input_size,
            hidden_size,
        )
    });
}

/// Reference forward for LSTM cell with non-zero initial state.
fn lstm_cell_with_state_loss(
    x_data: &[f32],
    w_ih_data: &[f32],
    w_hh_data: &[f32],
    h0_data: &[f32],
    c0_data: &[f32],
    batch: usize,
    input_size: usize,
    hidden_size: usize,
) -> f64 {
    let x = DynTensor::from_vec(x_data.to_vec(), &[batch, input_size], &cpu()).unwrap();
    let w_ih =
        DynTensor::from_vec(w_ih_data.to_vec(), &[4 * hidden_size, input_size], &cpu()).unwrap();
    let w_hh =
        DynTensor::from_vec(w_hh_data.to_vec(), &[4 * hidden_size, hidden_size], &cpu()).unwrap();
    let h0 = DynTensor::from_vec(h0_data.to_vec(), &[batch, hidden_size], &cpu()).unwrap();
    let c0 = DynTensor::from_vec(c0_data.to_vec(), &[batch, hidden_size], &cpu()).unwrap();

    let w_ih_t = w_ih.transpose(0, 1).unwrap();
    let w_hh_t = w_hh.transpose(0, 1).unwrap();

    // gates = x @ w_ih^T + h0 @ w_hh^T
    let gates = x
        .matmul(&w_ih_t)
        .unwrap()
        .add(&h0.matmul(&w_hh_t).unwrap())
        .unwrap();

    let hs = hidden_size;
    let i_gate = gates.narrow(1, 0, hs).unwrap().sigmoid().unwrap();
    let f_gate = gates.narrow(1, hs, hs).unwrap().sigmoid().unwrap();
    let g_gate = gates.narrow(1, 2 * hs, hs).unwrap().tanh().unwrap();
    let o_gate = gates.narrow(1, 3 * hs, hs).unwrap().sigmoid().unwrap();

    // c_new = f * c0 + i * g
    let c_new = f_gate
        .mul(&c0)
        .unwrap()
        .add(&i_gate.mul(&g_gate).unwrap())
        .unwrap();
    // h_new = o * tanh(c_new)
    let h_new = o_gate.mul(&c_new.tanh().unwrap()).unwrap();

    sum_f64(&h_new)
}

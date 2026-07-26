#![allow(deprecated)]
// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Finite-difference gradient tests for TrainableSwiGlu and TrainableMultiHeadAttention.
//!
//! Both are composite ops decomposed into matmul+silu+mul+softmax+reshape+transpose.
//! These FD tests verify the end-to-end gradient composition is correct.
//!
//! Re: #1098 (dvoice-integration: verified training), W1-84 request.

use std::sync::Arc;

use nn_core::dyn_tensor::DynTensor;
use nn_core::test_utils::cpu;

use crate::grad::backward;
use crate::grad::test_helpers::{check_fd_grad, sum_f64};
use crate::tracked::TrackedTensor;
use crate::trainable::TrainableSwiGlu;
use crate::trainable::{TrainableLinear, TrainableModule, TrainableMultiHeadAttention};
use crate::var::Var;

// ── SwiGlu FD tests ──────────────────────────────────────────────────

/// Reference SwiGlu forward: SiLU(x @ w_gate^T) * (x @ w_up^T) @ w_down^T
fn swiglu_ref_loss(
    x_data: &[f32],
    w_gate_data: &[f32],
    w_up_data: &[f32],
    w_down_data: &[f32],
    batch: usize,
    dim: usize,
    hidden: usize,
) -> f64 {
    let x = DynTensor::from_vec(x_data.to_vec(), &[batch, dim], &cpu()).unwrap();
    let w_gate = DynTensor::from_vec(w_gate_data.to_vec(), &[hidden, dim], &cpu()).unwrap();
    let w_up = DynTensor::from_vec(w_up_data.to_vec(), &[hidden, dim], &cpu()).unwrap();
    let w_down = DynTensor::from_vec(w_down_data.to_vec(), &[dim, hidden], &cpu()).unwrap();

    // Linear forward: x @ w^T
    let gate_out = x.matmul(&w_gate.transpose(0, 1).unwrap()).unwrap();
    let gate = gate_out.silu().unwrap();
    let up = x.matmul(&w_up.transpose(0, 1).unwrap()).unwrap();
    let h = gate.mul(&up).unwrap();
    let out = h.matmul(&w_down.transpose(0, 1).unwrap()).unwrap();
    sum_f64(&out)
}

/// FD test: SwiGlu dL/dx (gradient of loss w.r.t. input).
#[test]
fn test_swiglu_grad_input_fd() {
    let batch = 1;
    let dim = 3;
    let hidden = 2;

    // Small deterministic weights (non-zero for meaningful gradients).
    let w_gate_data: Vec<f32> = vec![0.1, -0.2, 0.3, 0.4, 0.1, -0.1];
    let w_up_data: Vec<f32> = vec![-0.1, 0.3, 0.2, 0.1, -0.3, 0.4];
    let w_down_data: Vec<f32> = vec![0.2, -0.1, 0.3, 0.1, -0.2, 0.4];
    let x_data: Vec<f32> = vec![0.5, -0.3, 0.8];

    let w_gate = TrainableLinear::from_tensors(
        DynTensor::from_vec(w_gate_data.clone(), &[hidden, dim], &cpu()).unwrap(),
        None,
    );
    let w_up = TrainableLinear::from_tensors(
        DynTensor::from_vec(w_up_data.clone(), &[hidden, dim], &cpu()).unwrap(),
        None,
    );
    let w_down = TrainableLinear::from_tensors(
        DynTensor::from_vec(w_down_data.clone(), &[dim, hidden], &cpu()).unwrap(),
        None,
    );
    let swiglu = TrainableSwiGlu::new(w_gate, w_up, w_down);

    let x_var = Var::new(DynTensor::from_vec(x_data.clone(), &[batch, dim], &cpu()).unwrap());
    let tx = Arc::new(TrackedTensor::from_var(&x_var).unwrap());
    let out = swiglu.forward(&tx).unwrap();
    let loss = out.sum_keepdim(0).unwrap().sum_keepdim(1).unwrap();
    let grads = backward(&loss).unwrap();
    let analytical_x = grads.get(&x_var).unwrap().to_flat_vec::<f32>().unwrap();

    let eps = 1e-3_f32;
    check_fd_grad(&analytical_x, &x_data, eps, |d| {
        swiglu_ref_loss(
            &d,
            &w_gate_data,
            &w_up_data,
            &w_down_data,
            batch,
            dim,
            hidden,
        )
    });
}

/// FD test: SwiGlu dL/dW_gate (gradient of loss w.r.t. gate weights).
#[test]
fn test_swiglu_grad_w_gate_fd() {
    let batch = 1;
    let dim = 3;
    let hidden = 2;

    let w_gate_data: Vec<f32> = vec![0.1, -0.2, 0.3, 0.4, 0.1, -0.1];
    let w_up_data: Vec<f32> = vec![-0.1, 0.3, 0.2, 0.1, -0.3, 0.4];
    let w_down_data: Vec<f32> = vec![0.2, -0.1, 0.3, 0.1, -0.2, 0.4];
    let x_data: Vec<f32> = vec![0.5, -0.3, 0.8];

    let w_gate = TrainableLinear::from_tensors(
        DynTensor::from_vec(w_gate_data.clone(), &[hidden, dim], &cpu()).unwrap(),
        None,
    );
    let w_up = TrainableLinear::from_tensors(
        DynTensor::from_vec(w_up_data.clone(), &[hidden, dim], &cpu()).unwrap(),
        None,
    );
    let w_down = TrainableLinear::from_tensors(
        DynTensor::from_vec(w_down_data.clone(), &[dim, hidden], &cpu()).unwrap(),
        None,
    );
    let swiglu = TrainableSwiGlu::new(w_gate, w_up, w_down);

    let x_var = Var::new(DynTensor::from_vec(x_data.clone(), &[batch, dim], &cpu()).unwrap());
    let tx = Arc::new(TrackedTensor::from_var(&x_var).unwrap());
    let out = swiglu.forward(&tx).unwrap();
    let loss = out.sum_keepdim(0).unwrap().sum_keepdim(1).unwrap();
    let grads = backward(&loss).unwrap();

    // Extract w_gate weight Var from the swiglu layer.
    let gate_vars = swiglu.w_gate().vars();
    let gate_weight_var = gate_vars[0]; // weight is first var
    let analytical_wg = grads
        .get(gate_weight_var)
        .unwrap()
        .to_flat_vec::<f32>()
        .unwrap();

    let eps = 1e-3_f32;
    check_fd_grad(&analytical_wg, &w_gate_data, eps, |d| {
        swiglu_ref_loss(&x_data, &d, &w_up_data, &w_down_data, batch, dim, hidden)
    });
}

// ── MHA FD test ──────────────────────────────────────────────────────

/// FD test: MHA dL/dx (gradient of loss w.r.t. input).
///
/// Uses minimal config: model_dim=4, num_heads=2, head_dim=2, seq_len=2, batch=1.
/// No mask. Verifies gradient through Q/K/V projections + SDPA + output projection.
#[test]
fn test_mha_grad_input_fd() {
    let batch = 1;
    let seq_len = 2;
    let model_dim = 4;
    let num_heads = 2;

    // Create small deterministic weight tensors.
    let make_proj = |data: Vec<f32>| -> TrainableLinear {
        TrainableLinear::from_tensors(
            DynTensor::from_vec(data, &[model_dim, model_dim], &cpu()).unwrap(),
            None,
        )
    };

    // Q, K, V, Out projections — small varied values.
    let q_data: Vec<f32> = (0..16).map(|i| (i as f32 - 8.0) * 0.05).collect();
    let k_data: Vec<f32> = (0..16).map(|i| (i as f32 - 4.0) * 0.03).collect();
    let v_data: Vec<f32> = (0..16).map(|i| (i as f32 - 12.0) * 0.04).collect();
    let o_data: Vec<f32> = (0..16).map(|i| (i as f32 - 6.0) * 0.02).collect();

    let mha = TrainableMultiHeadAttention::new(
        make_proj(q_data),
        make_proj(k_data),
        make_proj(v_data),
        make_proj(o_data),
        num_heads,
        model_dim,
    )
    .unwrap();

    // Input: [B=1, S=2, D=4].
    let x_data: Vec<f32> = vec![0.5, -0.3, 0.8, 0.1, -0.2, 0.6, -0.4, 0.7];
    let x_var = Var::new(
        DynTensor::from_vec(x_data.clone(), &[batch, seq_len, model_dim], &cpu()).unwrap(),
    );
    let tx = Arc::new(TrackedTensor::from_var(&x_var).unwrap());
    let out = mha.forward(&tx).unwrap();
    // Sum to scalar loss.
    let loss = out
        .sum_keepdim(0)
        .unwrap()
        .sum_keepdim(1)
        .unwrap()
        .sum_keepdim(2)
        .unwrap();
    let grads = backward(&loss).unwrap();
    let analytical_x = grads.get(&x_var).unwrap().to_flat_vec::<f32>().unwrap();

    // FD uses DynTensor-only reference (no tracked ops).
    let eps = 1e-3_f32;
    let mha_vars = mha.vars();
    // Extract weight data for the reference forward.
    let q_w = mha_vars[0].data().unwrap().to_flat_vec::<f32>().unwrap();
    let k_w = mha_vars[1].data().unwrap().to_flat_vec::<f32>().unwrap();
    let v_w = mha_vars[2].data().unwrap().to_flat_vec::<f32>().unwrap();
    let o_w = mha_vars[3].data().unwrap().to_flat_vec::<f32>().unwrap();

    check_fd_grad(&analytical_x, &x_data, eps, |d| {
        mha_ref_loss(
            &d, &q_w, &k_w, &v_w, &o_w, batch, seq_len, model_dim, num_heads,
        )
    });
}

/// Reference MHA forward using only DynTensor ops (no autodiff tape).
fn mha_ref_loss(
    x_data: &[f32],
    q_w: &[f32],
    k_w: &[f32],
    v_w: &[f32],
    o_w: &[f32],
    batch: usize,
    seq_len: usize,
    model_dim: usize,
    num_heads: usize,
) -> f64 {
    let head_dim = model_dim / num_heads;
    let x = DynTensor::from_vec(x_data.to_vec(), &[batch, seq_len, model_dim], &cpu()).unwrap();

    let q_weight = DynTensor::from_vec(q_w.to_vec(), &[model_dim, model_dim], &cpu()).unwrap();
    let k_weight = DynTensor::from_vec(k_w.to_vec(), &[model_dim, model_dim], &cpu()).unwrap();
    let v_weight = DynTensor::from_vec(v_w.to_vec(), &[model_dim, model_dim], &cpu()).unwrap();
    let o_weight = DynTensor::from_vec(o_w.to_vec(), &[model_dim, model_dim], &cpu()).unwrap();

    // Q, K, V projections: [B, S, D] @ [D, D] = [B, S, D]
    let q = x.matmul(&q_weight.transpose(0, 1).unwrap()).unwrap();
    let k = x.matmul(&k_weight.transpose(0, 1).unwrap()).unwrap();
    let v = x.matmul(&v_weight.transpose(0, 1).unwrap()).unwrap();

    // Reshape to heads: [B, S, D] -> [B, S, H, head_dim] -> [B, H, S, head_dim]
    let q = q
        .reshape([batch, seq_len, num_heads, head_dim])
        .unwrap()
        .transpose(1, 2)
        .unwrap();
    let k = k
        .reshape([batch, seq_len, num_heads, head_dim])
        .unwrap()
        .transpose(1, 2)
        .unwrap();
    let v = v
        .reshape([batch, seq_len, num_heads, head_dim])
        .unwrap()
        .transpose(1, 2)
        .unwrap();

    // SDPA: scores = Q @ K^T / sqrt(head_dim)
    let scores = q.matmul(&k.transpose(2, 3).unwrap()).unwrap();
    let scale = 1.0 / (head_dim as f64).sqrt();
    let scores = scores.mul_scalar(scale).unwrap();
    let attn = scores.softmax(3).unwrap();

    // Weighted values: attn @ V -> [B, H, S, head_dim]
    let context = attn.matmul(&v).unwrap();

    // Reshape back: [B, H, S, head_dim] -> [B, S, H, head_dim] -> [B, S, D]
    let context = context
        .transpose(1, 2)
        .unwrap()
        .reshape([batch, seq_len, model_dim])
        .unwrap();

    // Output projection.
    let out = context.matmul(&o_weight.transpose(0, 1).unwrap()).unwrap();
    sum_f64(&out)
}

// ── MHA weight FD helpers ─────────────────────────────────────────────

/// Shared setup for MHA weight FD tests: creates MHA layer, computes forward + backward,
/// and returns (mha, analytical_grad_for_var, weight_data, q/k/v/o data vectors).
fn mha_weight_fd_setup() -> (
    TrainableMultiHeadAttention,
    Vec<f32>, // x_data
    Vec<f32>, // q_data
    Vec<f32>, // k_data
    Vec<f32>, // v_data
    Vec<f32>, // o_data
    usize,    // batch
    usize,    // seq_len
    usize,    // model_dim
    usize,    // num_heads
) {
    let batch = 1;
    let seq_len = 2;
    let model_dim = 4;
    let num_heads = 2;

    let make_proj = |data: Vec<f32>| -> TrainableLinear {
        TrainableLinear::from_tensors(
            DynTensor::from_vec(data, &[model_dim, model_dim], &cpu()).unwrap(),
            None,
        )
    };

    let q_data: Vec<f32> = (0..16).map(|i| (i as f32 - 8.0) * 0.05).collect();
    let k_data: Vec<f32> = (0..16).map(|i| (i as f32 - 4.0) * 0.03).collect();
    let v_data: Vec<f32> = (0..16).map(|i| (i as f32 - 12.0) * 0.04).collect();
    let o_data: Vec<f32> = (0..16).map(|i| (i as f32 - 6.0) * 0.02).collect();
    let x_data: Vec<f32> = vec![0.5, -0.3, 0.8, 0.1, -0.2, 0.6, -0.4, 0.7];

    let mha = TrainableMultiHeadAttention::new(
        make_proj(q_data.clone()),
        make_proj(k_data.clone()),
        make_proj(v_data.clone()),
        make_proj(o_data.clone()),
        num_heads,
        model_dim,
    )
    .unwrap();

    (
        mha, x_data, q_data, k_data, v_data, o_data, batch, seq_len, model_dim, num_heads,
    )
}

/// Helper: run MHA forward + backward and return analytical gradient for a given var index.
fn mha_analytical_grad(
    mha: &TrainableMultiHeadAttention,
    x_data: &[f32],
    batch: usize,
    seq_len: usize,
    model_dim: usize,
    var_idx: usize,
) -> Vec<f32> {
    let x_var = Var::new(
        DynTensor::from_vec(x_data.to_vec(), &[batch, seq_len, model_dim], &cpu()).unwrap(),
    );
    let tx = Arc::new(TrackedTensor::from_var(&x_var).unwrap());
    let out = mha.forward(&tx).unwrap();
    let loss = out
        .sum_keepdim(0)
        .unwrap()
        .sum_keepdim(1)
        .unwrap()
        .sum_keepdim(2)
        .unwrap();
    let grads = backward(&loss).unwrap();
    let target_var = &mha.vars()[var_idx];
    grads.get(target_var).unwrap().to_flat_vec::<f32>().unwrap()
}

/// FD test: MHA dL/dW_q (gradient of loss w.r.t. query projection weight).
#[test]
fn test_mha_grad_w_q_fd() {
    let (mha, x_data, q_data, k_data, v_data, o_data, batch, seq_len, model_dim, num_heads) =
        mha_weight_fd_setup();
    let analytical = mha_analytical_grad(&mha, &x_data, batch, seq_len, model_dim, 0);
    let eps = 1e-3_f32;
    check_fd_grad(&analytical, &q_data, eps, |d| {
        mha_ref_loss(
            &x_data, &d, &k_data, &v_data, &o_data, batch, seq_len, model_dim, num_heads,
        )
    });
}

/// FD test: MHA dL/dW_k (gradient of loss w.r.t. key projection weight).
#[test]
fn test_mha_grad_w_k_fd() {
    let (mha, x_data, q_data, k_data, v_data, o_data, batch, seq_len, model_dim, num_heads) =
        mha_weight_fd_setup();
    let analytical = mha_analytical_grad(&mha, &x_data, batch, seq_len, model_dim, 1);
    let eps = 1e-3_f32;
    check_fd_grad(&analytical, &k_data, eps, |d| {
        mha_ref_loss(
            &x_data, &q_data, &d, &v_data, &o_data, batch, seq_len, model_dim, num_heads,
        )
    });
}

/// FD test: MHA dL/dW_v (gradient of loss w.r.t. value projection weight).
#[test]
fn test_mha_grad_w_v_fd() {
    let (mha, x_data, q_data, k_data, v_data, o_data, batch, seq_len, model_dim, num_heads) =
        mha_weight_fd_setup();
    let analytical = mha_analytical_grad(&mha, &x_data, batch, seq_len, model_dim, 2);
    let eps = 1e-3_f32;
    check_fd_grad(&analytical, &v_data, eps, |d| {
        mha_ref_loss(
            &x_data, &q_data, &k_data, &d, &o_data, batch, seq_len, model_dim, num_heads,
        )
    });
}

/// FD test: MHA dL/dW_o (gradient of loss w.r.t. output projection weight).
#[test]
fn test_mha_grad_w_o_fd() {
    let (mha, x_data, q_data, k_data, v_data, o_data, batch, seq_len, model_dim, num_heads) =
        mha_weight_fd_setup();
    let analytical = mha_analytical_grad(&mha, &x_data, batch, seq_len, model_dim, 3);
    let eps = 1e-3_f32;
    check_fd_grad(&analytical, &o_data, eps, |d| {
        mha_ref_loss(
            &x_data, &q_data, &k_data, &v_data, &d, batch, seq_len, model_dim, num_heads,
        )
    });
}

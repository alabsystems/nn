// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Expanded test coverage for nn-autodiff.
//!
//! Covers gaps identified in the existing test suite:
//! - Log-softmax backward (FD validation)
//! - Maximum/Minimum backward (FD validation)
//! - Stack backward (FD validation)
//! - Unfold backward (FD validation)
//! - MSE/L1/Huber loss backward (gradient correctness)
//! - Multi-layer chain rule through diverse ops
//! - Zero gradient handling
//! - Gradient through broadcast+mul patterns
//! - GradStore shape mismatch rejection
//! - Memory safety: no gradient tape Arc leak

use std::sync::Arc;

use nn_autodiff::{backward, backward_for_vars, TrackedTensor, Var, VarMap};
use nn_core::dyn_tensor::DynTensor;
use nn_core::{DType, Device};

fn cpu() -> Device {
    Device::Cpu
}

fn scalar_var(val: f32) -> Var {
    Var::new(DynTensor::from_vec(vec![val], &[1], &cpu()).unwrap())
}

fn vec_var(data: Vec<f32>) -> Var {
    let n = data.len();
    Var::new(DynTensor::from_vec(data, &[n], &cpu()).unwrap())
}

fn mat_var(data: Vec<f32>, rows: usize, cols: usize) -> Var {
    Var::new(DynTensor::from_vec(data, &[rows, cols], &cpu()).unwrap())
}

/// Reduce a tracked tensor to a scalar by summing all dims from last to first.
fn reduce_to_scalar(t: &Arc<TrackedTensor>) -> Arc<TrackedTensor> {
    let mut reduced = Arc::clone(t);
    for dim in (0..reduced.tensor().rank()).rev() {
        reduced = reduced.sum_keepdim(dim).unwrap();
    }
    reduced
}

/// Finite-difference numerical gradient for a single variable.
fn finite_diff_grad(data: &[f32], eps: f64, forward: &dyn Fn(Vec<f32>) -> f64) -> Vec<f64> {
    let mut grad = Vec::with_capacity(data.len());
    for i in 0..data.len() {
        let mut plus = data.to_vec();
        let mut minus = data.to_vec();
        plus[i] += eps as f32;
        minus[i] -= eps as f32;
        let g = (forward(plus) - forward(minus)) / (2.0 * eps);
        grad.push(g);
    }
    grad
}

// ===========================================================================
// 1. Log-softmax backward (FD validation)
// ===========================================================================

#[test]
fn test_backward_log_softmax_fd() {
    let x_data = vec![1.0, 2.0, 3.0, 0.5, 1.5, -0.5];
    let x = Var::new(DynTensor::from_vec(x_data.clone(), &[2, 3], &cpu()).unwrap());
    let t = Arc::new(TrackedTensor::from_var(&x).unwrap());
    let ls = t.log_softmax(1).unwrap();
    // Use sum-of-squares as loss so gradients are input-dependent
    let y = ls.sqr().unwrap();
    let loss = reduce_to_scalar(&y);
    let grads = backward(&loss).unwrap();
    let grad = grads.get(&x).unwrap().to_flat_vec::<f32>().unwrap();

    let fd = finite_diff_grad(&x_data, 1e-4, &|xv: Vec<f32>| {
        let t = DynTensor::from_vec(xv, &[2, 3], &cpu()).unwrap();
        let ls = t.log_softmax(1).unwrap();
        let s2 = ls.sqr().unwrap();
        s2.to_flat_vec::<f32>()
            .unwrap()
            .iter()
            .map(|&v| f64::from(v))
            .sum()
    });

    for i in 0..x_data.len() {
        let err = (f64::from(grad[i]) - fd[i]).abs();
        assert!(
            err < 1e-2,
            "log_softmax sqr grad[{i}]: analytical={}, numerical={}, err={}",
            grad[i],
            fd[i],
            err
        );
    }
}

#[test]
fn test_backward_log_softmax_simple() {
    // Simpler test: log_softmax -> sum as loss
    let x_data = vec![1.0, 2.0, 3.0];
    let x = Var::new(DynTensor::from_vec(x_data.clone(), &[1, 3], &cpu()).unwrap());
    let t = Arc::new(TrackedTensor::from_var(&x).unwrap());
    let ls = t.log_softmax(1).unwrap();
    let loss = reduce_to_scalar(&ls);
    let grads = backward(&loss).unwrap();
    let grad = grads.get(&x).unwrap().to_flat_vec::<f32>().unwrap();

    let fd = finite_diff_grad(&x_data, 1e-4, &|xv: Vec<f32>| {
        let t = DynTensor::from_vec(xv, &[1, 3], &cpu()).unwrap();
        let ls = t.log_softmax(1).unwrap();
        ls.to_flat_vec::<f32>()
            .unwrap()
            .iter()
            .map(|&v| f64::from(v))
            .sum()
    });

    for i in 0..x_data.len() {
        let err = (f64::from(grad[i]) - fd[i]).abs();
        assert!(
            err < 1e-2,
            "log_softmax sum grad[{i}]: analytical={}, numerical={}, err={}",
            grad[i],
            fd[i],
            err
        );
    }
}

// ===========================================================================
// 2. Maximum / Minimum backward (FD validation)
// ===========================================================================

#[test]
fn test_backward_maximum_fd() {
    // Avoid ties in a vs b (ties cause subgradient mismatch with FD)
    let a_data = vec![1.0, 4.0, 2.5, 5.0];
    let b_data = vec![3.0, 2.0, 1.5, 6.0];

    let a = vec_var(a_data.clone());
    let b = vec_var(b_data.clone());
    let ta = Arc::new(TrackedTensor::from_var(&a).unwrap());
    let tb = Arc::new(TrackedTensor::from_var(&b).unwrap());
    let y = ta.maximum(&tb).unwrap();
    // Use sqr for non-trivial loss
    let loss = reduce_to_scalar(&y.sqr().unwrap());
    let grads = backward(&loss).unwrap();
    let grad_a = grads.get(&a).unwrap().to_flat_vec::<f32>().unwrap();
    let grad_b = grads.get(&b).unwrap().to_flat_vec::<f32>().unwrap();

    let b_ref = b_data.clone();
    let fd_a = finite_diff_grad(&a_data, 1e-4, &|av: Vec<f32>| {
        let at = DynTensor::from_vec(av, &[4], &cpu()).unwrap();
        let bt = DynTensor::from_vec(b_ref.clone(), &[4], &cpu()).unwrap();
        let m = at.maximum(&bt).unwrap();
        let s = m.sqr().unwrap();
        s.to_flat_vec::<f32>()
            .unwrap()
            .iter()
            .map(|&v| f64::from(v))
            .sum()
    });

    for i in 0..a_data.len() {
        let err = (f64::from(grad_a[i]) - fd_a[i]).abs();
        assert!(
            err < 5e-2,
            "maximum grad_a[{i}]: analytical={}, numerical={}, err={}",
            grad_a[i],
            fd_a[i],
            err
        );
    }

    let a_ref = a_data;
    let fd_b = finite_diff_grad(&b_data, 1e-4, &|bv: Vec<f32>| {
        let at = DynTensor::from_vec(a_ref.clone(), &[4], &cpu()).unwrap();
        let bt = DynTensor::from_vec(bv, &[4], &cpu()).unwrap();
        let m = at.maximum(&bt).unwrap();
        let s = m.sqr().unwrap();
        s.to_flat_vec::<f32>()
            .unwrap()
            .iter()
            .map(|&v| f64::from(v))
            .sum()
    });

    for i in 0..b_data.len() {
        let err = (f64::from(grad_b[i]) - fd_b[i]).abs();
        assert!(
            err < 5e-2,
            "maximum grad_b[{i}]: analytical={}, numerical={}, err={}",
            grad_b[i],
            fd_b[i],
            err
        );
    }
}

#[test]
fn test_backward_minimum_fd() {
    // Avoid ties in a vs b (ties cause subgradient mismatch with FD)
    let a_data = vec![1.0, 4.0, 2.5, 5.0];
    let b_data = vec![3.0, 2.0, 1.5, 6.0];

    let a = vec_var(a_data.clone());
    let b = vec_var(b_data.clone());
    let ta = Arc::new(TrackedTensor::from_var(&a).unwrap());
    let tb = Arc::new(TrackedTensor::from_var(&b).unwrap());
    let y = ta.minimum(&tb).unwrap();
    let loss = reduce_to_scalar(&y.sqr().unwrap());
    let grads = backward(&loss).unwrap();
    let grad_a = grads.get(&a).unwrap().to_flat_vec::<f32>().unwrap();

    let b_ref = b_data;
    let fd_a = finite_diff_grad(&a_data, 1e-4, &|av: Vec<f32>| {
        let at = DynTensor::from_vec(av, &[4], &cpu()).unwrap();
        let bt = DynTensor::from_vec(b_ref.clone(), &[4], &cpu()).unwrap();
        let m = at.minimum(&bt).unwrap();
        let s = m.sqr().unwrap();
        s.to_flat_vec::<f32>()
            .unwrap()
            .iter()
            .map(|&v| f64::from(v))
            .sum()
    });

    for i in 0..a_data.len() {
        let err = (f64::from(grad_a[i]) - fd_a[i]).abs();
        assert!(
            err < 5e-2,
            "minimum grad_a[{i}]: analytical={}, numerical={}, err={}",
            grad_a[i],
            fd_a[i],
            err
        );
    }
}

// ===========================================================================
// 3. Stack backward (FD validation)
// ===========================================================================

#[test]
fn test_backward_stack_fd() {
    // Stack 3 tensors of shape [2] along dim 0 -> [3, 2]
    let a_data = vec![1.0, 2.0];
    let b_data = vec![3.0, 4.0];
    let c_data = vec![5.0, 6.0];

    let a = vec_var(a_data);
    let b = vec_var(b_data);
    let c = vec_var(c_data);
    let ta = Arc::new(TrackedTensor::from_var(&a).unwrap());
    let tb = Arc::new(TrackedTensor::from_var(&b).unwrap());
    let tc = Arc::new(TrackedTensor::from_var(&c).unwrap());

    let stacked = TrackedTensor::stack(&[ta, tb, tc], 0).unwrap(); // [3, 2]
    assert_eq!(stacked.dims(), &[3, 2]);

    // Non-trivial loss: sum of squares
    let loss = reduce_to_scalar(&stacked.sqr().unwrap());
    let grads = backward(&loss).unwrap();
    let grad_a = grads.get(&a).unwrap().to_flat_vec::<f32>().unwrap();
    let grad_b = grads.get(&b).unwrap().to_flat_vec::<f32>().unwrap();
    let grad_c = grads.get(&c).unwrap().to_flat_vec::<f32>().unwrap();

    // d/dx_i sum(x_i^2) = 2*x_i
    assert!((grad_a[0] - 2.0).abs() < 1e-5, "grad_a[0] = 2*1 = 2");
    assert!((grad_a[1] - 4.0).abs() < 1e-5, "grad_a[1] = 2*2 = 4");
    assert!((grad_b[0] - 6.0).abs() < 1e-5, "grad_b[0] = 2*3 = 6");
    assert!((grad_b[1] - 8.0).abs() < 1e-5, "grad_b[1] = 2*4 = 8");
    assert!((grad_c[0] - 10.0).abs() < 1e-5, "grad_c[0] = 2*5 = 10");
    assert!((grad_c[1] - 12.0).abs() < 1e-5, "grad_c[1] = 2*6 = 12");
}

#[test]
fn test_backward_stack_dim1_fd() {
    // Stack two [3] tensors along dim 0, then use them in a multi-path graph
    // This exercises stack backward when the stacked result feeds into further ops
    let a_data = vec![1.0, 2.0, 3.0];
    let b_data = vec![4.0, 5.0, 6.0];

    let a = vec_var(a_data.clone());
    let b = vec_var(b_data.clone());
    let ta = Arc::new(TrackedTensor::from_var(&a).unwrap());
    let tb = Arc::new(TrackedTensor::from_var(&b).unwrap());

    let stacked = TrackedTensor::stack(&[ta, tb], 0).unwrap(); // [2, 3]
                                                               // Apply tanh to make it non-trivial
    let y = stacked.tanh().unwrap();
    let loss = reduce_to_scalar(&y);
    let grads = backward(&loss).unwrap();

    let grad_a = grads.get(&a).unwrap().to_flat_vec::<f32>().unwrap();
    let grad_b = grads.get(&b).unwrap().to_flat_vec::<f32>().unwrap();

    // FD check for a
    let b_ref = b_data;
    let fd_a = finite_diff_grad(&a_data, 1e-4, &|av: Vec<f32>| {
        let at = DynTensor::from_vec(av, &[3], &cpu()).unwrap();
        let bt = DynTensor::from_vec(b_ref.clone(), &[3], &cpu()).unwrap();
        let refs: Vec<&DynTensor> = vec![&at, &bt];
        let s = DynTensor::stack(&refs, 0).unwrap();
        let y = s.tanh().unwrap();
        y.to_flat_vec::<f32>()
            .unwrap()
            .iter()
            .map(|&v| f64::from(v))
            .sum()
    });

    for i in 0..a_data.len() {
        let err = (f64::from(grad_a[i]) - fd_a[i]).abs();
        assert!(
            err < 1e-2,
            "stack->tanh grad_a[{i}]: analytical={}, numerical={}, err={}",
            grad_a[i],
            fd_a[i],
            err
        );
    }

    // Verify gradient shapes are correct
    assert_eq!(grads.get(&a).unwrap().dims(), &[3]);
    assert_eq!(grads.get(&b).unwrap().dims(), &[3]);
    for &g in &grad_b {
        assert!(g.is_finite());
    }
}

// ===========================================================================
// 4. Unfold backward (FD validation)
// ===========================================================================

#[test]
fn test_backward_unfold_fd() {
    // Input: [1, 6], unfold(dim=1, size=3, step=1) -> [1, 4, 3]
    let x_data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let x = Var::new(DynTensor::from_vec(x_data.clone(), &[1, 6], &cpu()).unwrap());
    let t = Arc::new(TrackedTensor::from_var(&x).unwrap());
    let unfolded = t.unfold(1, 3, 1).unwrap(); // [1, 4, 3]
    assert_eq!(unfolded.dims(), &[1, 4, 3]);

    let loss = reduce_to_scalar(&unfolded.sqr().unwrap());
    let grads = backward(&loss).unwrap();
    let grad = grads.get(&x).unwrap().to_flat_vec::<f32>().unwrap();

    // FD check
    let fd = finite_diff_grad(&x_data, 1e-4, &|xv: Vec<f32>| {
        let t = DynTensor::from_vec(xv, &[1, 6], &cpu()).unwrap();
        let u = t.unfold(1, 3, 1).unwrap();
        let s = u.sqr().unwrap();
        s.to_flat_vec::<f32>()
            .unwrap()
            .iter()
            .map(|&v| f64::from(v))
            .sum()
    });

    for i in 0..x_data.len() {
        let err = (f64::from(grad[i]) - fd[i]).abs();
        assert!(
            err < 1e-1,
            "unfold grad[{i}]: analytical={}, numerical={}, err={}",
            grad[i],
            fd[i],
            err
        );
    }
}

#[test]
fn test_backward_unfold_step2_fd() {
    // Input: [8], unfold(dim=0, size=3, step=2) -> [3, 3]
    let x_data: Vec<f32> = (1..=8).map(|v| v as f32).collect();
    let x = Var::new(DynTensor::from_vec(x_data.clone(), &[8], &cpu()).unwrap());
    let t = Arc::new(TrackedTensor::from_var(&x).unwrap());
    let unfolded = t.unfold(0, 3, 2).unwrap();

    let loss = reduce_to_scalar(&unfolded);
    let grads = backward(&loss).unwrap();
    let grad = grads.get(&x).unwrap().to_flat_vec::<f32>().unwrap();

    let fd = finite_diff_grad(&x_data, 1e-4, &|xv: Vec<f32>| {
        let t = DynTensor::from_vec(xv, &[8], &cpu()).unwrap();
        let u = t.unfold(0, 3, 2).unwrap();
        u.to_flat_vec::<f32>()
            .unwrap()
            .iter()
            .map(|&v| f64::from(v))
            .sum()
    });

    for i in 0..x_data.len() {
        let err = (f64::from(grad[i]) - fd[i]).abs();
        assert!(
            err < 1e-2,
            "unfold step=2 grad[{i}]: analytical={}, numerical={}, err={}",
            grad[i],
            fd[i],
            err
        );
    }
}

// ===========================================================================
// 5. MSE / L1 / Huber loss backward (gradient correctness)
// ===========================================================================

#[test]
fn test_backward_mse_loss_fd() {
    let pred_data = vec![1.0, 2.0, 3.0, 4.0];
    let target_data = vec![1.5, 1.5, 3.5, 3.5];

    let pred = Var::new(DynTensor::from_vec(pred_data.clone(), &[2, 2], &cpu()).unwrap());
    let target = Arc::new(TrackedTensor::from_tensor(
        DynTensor::from_vec(target_data.clone(), &[2, 2], &cpu()).unwrap(),
    ));

    let tp = Arc::new(TrackedTensor::from_var(&pred).unwrap());
    let loss = tp.mse_loss(&target).unwrap();
    let grads = backward(&loss).unwrap();
    let grad = grads.get(&pred).unwrap().to_flat_vec::<f32>().unwrap();

    let fd = finite_diff_grad(&pred_data, 1e-4, &|pv: Vec<f32>| {
        let p = DynTensor::from_vec(pv, &[2, 2], &cpu()).unwrap();
        let t = DynTensor::from_vec(target_data.clone(), &[2, 2], &cpu()).unwrap();
        let diff = p.sub(&t).unwrap();
        let sq = diff.sqr().unwrap();
        let vals = sq.to_flat_vec::<f32>().unwrap();
        vals.iter().map(|&v| f64::from(v)).sum::<f64>() / vals.len() as f64
    });

    for i in 0..pred_data.len() {
        let err = (f64::from(grad[i]) - fd[i]).abs();
        assert!(
            err < 1e-2,
            "mse_loss grad[{i}]: analytical={}, numerical={}, err={}",
            grad[i],
            fd[i],
            err
        );
    }
}

#[test]
fn test_backward_l1_loss_fd() {
    let pred_data = vec![1.0, 2.0, 3.0, 4.0];
    let target_data = vec![1.5, 1.5, 3.5, 3.5];

    let pred = Var::new(DynTensor::from_vec(pred_data.clone(), &[2, 2], &cpu()).unwrap());
    let target = Arc::new(TrackedTensor::from_tensor(
        DynTensor::from_vec(target_data.clone(), &[2, 2], &cpu()).unwrap(),
    ));

    let tp = Arc::new(TrackedTensor::from_var(&pred).unwrap());
    let loss = tp.l1_loss(&target).unwrap();
    let grads = backward(&loss).unwrap();
    let grad = grads.get(&pred).unwrap().to_flat_vec::<f32>().unwrap();

    let fd = finite_diff_grad(&pred_data, 1e-4, &|pv: Vec<f32>| {
        let p = DynTensor::from_vec(pv, &[2, 2], &cpu()).unwrap();
        let t = DynTensor::from_vec(target_data.clone(), &[2, 2], &cpu()).unwrap();
        let diff = p.sub(&t).unwrap();
        let ab = diff.abs().unwrap();
        let vals = ab.to_flat_vec::<f32>().unwrap();
        vals.iter().map(|&v| f64::from(v)).sum::<f64>() / vals.len() as f64
    });

    for i in 0..pred_data.len() {
        let err = (f64::from(grad[i]) - fd[i]).abs();
        assert!(
            err < 1e-2,
            "l1_loss grad[{i}]: analytical={}, numerical={}, err={}",
            grad[i],
            fd[i],
            err
        );
    }
}

#[test]
fn test_backward_huber_loss_fd() {
    let pred_data = vec![1.0, 2.0, 3.0, 5.0]; // diff from target: -0.5, 0.5, -0.5, 1.5
    let target_data = vec![1.5, 1.5, 3.5, 3.5];
    let delta = 1.0;

    let pred = Var::new(DynTensor::from_vec(pred_data.clone(), &[2, 2], &cpu()).unwrap());
    let target = Arc::new(TrackedTensor::from_tensor(
        DynTensor::from_vec(target_data.clone(), &[2, 2], &cpu()).unwrap(),
    ));

    let tp = Arc::new(TrackedTensor::from_var(&pred).unwrap());
    let loss = tp.huber_loss(&target, delta).unwrap();
    let grads = backward(&loss).unwrap();
    let grad = grads.get(&pred).unwrap().to_flat_vec::<f32>().unwrap();

    let fd = finite_diff_grad(&pred_data, 1e-4, &|pv: Vec<f32>| {
        // Huber loss: 0.5*d^2 if |d| < delta, else delta*(|d| - 0.5*delta)
        let n = pv.len();
        let mut total = 0.0_f64;
        for i in 0..n {
            let d = f64::from(pv[i] - target_data[i]);
            let ad = d.abs();
            if ad < delta {
                total += 0.5 * d * d;
            } else {
                total += delta * (ad - 0.5 * delta);
            }
        }
        total / n as f64
    });

    for i in 0..pred_data.len() {
        let err = (f64::from(grad[i]) - fd[i]).abs();
        assert!(
            err < 1e-2,
            "huber_loss grad[{i}]: analytical={}, numerical={}, err={}",
            grad[i],
            fd[i],
            err
        );
    }
}

// ===========================================================================
// 6. Multi-layer chain rule through diverse ops
// ===========================================================================

#[test]
fn test_chain_rule_matmul_silu_layernorm_fd() {
    // Forward: x -> matmul(w) -> silu -> layernorm(gamma, beta) -> sum
    let w_data = vec![0.3, -0.5, 0.7, -0.2, 0.4, 0.1]; // [2, 3]
    let x_data = vec![1.0, -0.5]; // [1, 2]
    let gamma_data = vec![1.0, 1.0, 1.0]; // [3]
    let beta_data = vec![0.0, 0.0, 0.0]; // [3]

    let w = mat_var(w_data.clone(), 2, 3);
    let gamma = vec_var(gamma_data.clone());
    let beta = vec_var(beta_data.clone());
    let x = Arc::new(TrackedTensor::from_tensor(
        DynTensor::from_vec(x_data, &[1, 2], &cpu()).unwrap(),
    ));

    let tw = Arc::new(TrackedTensor::from_var(&w).unwrap());
    let tg = Arc::new(TrackedTensor::from_var(&gamma).unwrap());
    let tb = Arc::new(TrackedTensor::from_var(&beta).unwrap());

    let h = x.matmul(&tw).unwrap(); // [1, 3]
    let h = h.silu().unwrap(); // [1, 3]
    let y = h.layer_norm(&tg, &tb, 1e-5).unwrap(); // [1, 3]
    let loss = reduce_to_scalar(&y);
    let grads = backward(&loss).unwrap();

    let grad_w = grads.get(&w).unwrap().to_flat_vec::<f32>().unwrap();

    // FD check on w
    let gamma_ref = gamma_data;
    let beta_ref = beta_data;
    let fd_w = finite_diff_grad(&w_data, 1e-3, &|wv: Vec<f32>| {
        let x_t = DynTensor::from_vec(vec![1.0, -0.5], &[1, 2], &cpu()).unwrap();
        let w_t = DynTensor::from_vec(wv, &[2, 3], &cpu()).unwrap();
        let g_t = DynTensor::from_vec(gamma_ref.clone(), &[3], &cpu()).unwrap();
        let b_t = DynTensor::from_vec(beta_ref.clone(), &[3], &cpu()).unwrap();

        let h = x_t.matmul(&w_t).unwrap();
        let h = h.silu().unwrap();
        // Manual layernorm
        let mean = h.mean_keepdim(1).unwrap();
        let diff = h.sub(&mean).unwrap();
        let var = diff.sqr().unwrap().mean_keepdim(1).unwrap();
        let inv_std = var
            .add_scalar(1e-5)
            .unwrap()
            .sqrt()
            .unwrap()
            .recip()
            .unwrap();
        let normed = diff.mul(&inv_std).unwrap();
        let out = normed.mul(&g_t).unwrap().add(&b_t).unwrap();
        out.to_flat_vec::<f32>()
            .unwrap()
            .iter()
            .map(|&v| f64::from(v))
            .sum()
    });

    for i in 0..w_data.len() {
        let err = (f64::from(grad_w[i]) - fd_w[i]).abs();
        assert!(
            err < 5e-2,
            "chain matmul->silu->ln grad_w[{i}]: analytical={}, numerical={}, err={}",
            grad_w[i],
            fd_w[i],
            err
        );
    }
}

#[test]
fn test_chain_rule_four_ops_fd() {
    // Chain: x -> exp -> neg -> sqr -> sqrt -> scalar loss
    // f(x) = sqrt((-exp(x))^2) = sqrt(exp(2x)) = exp(x)
    // So df/dx = exp(x)
    let x_val = 0.5_f32;
    let x = scalar_var(x_val);
    let t = Arc::new(TrackedTensor::from_var(&x).unwrap());
    let y = t
        .exp()
        .unwrap()
        .neg()
        .unwrap()
        .sqr()
        .unwrap()
        .sqrt()
        .unwrap();
    let grads = backward(&y).unwrap();
    let grad = grads.get(&x).unwrap().to_flat_vec::<f32>().unwrap();

    let expected = x_val.exp();
    assert!(
        (grad[0] - expected).abs() < 1e-4,
        "expected {expected}, got {}",
        grad[0]
    );
}

#[test]
fn test_chain_rule_sin_cos_composition_fd() {
    // f(x) = sin(cos(x)), df/dx = cos(cos(x)) * (-sin(x))
    let x_val = 1.0_f32;
    let x = scalar_var(x_val);
    let t = Arc::new(TrackedTensor::from_var(&x).unwrap());
    let y = t.cos().unwrap().sin().unwrap();
    let grads = backward(&y).unwrap();
    let grad = grads.get(&x).unwrap().to_flat_vec::<f32>().unwrap();

    let fd = finite_diff_grad(&[x_val], 1e-4, &|xv: Vec<f32>| {
        let v = f64::from(xv[0]);
        v.cos().sin()
    });

    let err = (f64::from(grad[0]) - fd[0]).abs();
    assert!(
        err < 1e-3,
        "sin(cos(x)) grad: analytical={}, numerical={}, err={}",
        grad[0],
        fd[0],
        err
    );
}

#[test]
fn test_chain_rule_five_layer_mlp_fd() {
    // 5-layer deep chain: matmul -> tanh -> matmul -> sigmoid -> matmul -> sum
    let w1_data = vec![0.3, -0.5, 0.7, -0.2]; // [2, 2]
    let w2_data = vec![0.4, 0.1, -0.3, 0.6]; // [2, 2]
    let w3_data = vec![0.2, -0.1]; // [2, 1]
    let x_data = vec![1.0, -0.5]; // [1, 2]

    let w1 = mat_var(w1_data.clone(), 2, 2);
    let w2 = mat_var(w2_data.clone(), 2, 2);
    let w3 = Var::new(DynTensor::from_vec(w3_data.clone(), &[2, 1], &cpu()).unwrap());
    let x = Arc::new(TrackedTensor::from_tensor(
        DynTensor::from_vec(x_data, &[1, 2], &cpu()).unwrap(),
    ));

    let tw1 = Arc::new(TrackedTensor::from_var(&w1).unwrap());
    let tw2 = Arc::new(TrackedTensor::from_var(&w2).unwrap());
    let tw3 = Arc::new(TrackedTensor::from_var(&w3).unwrap());

    let h1 = x.matmul(&tw1).unwrap().tanh().unwrap();
    let h2 = h1.matmul(&tw2).unwrap().sigmoid().unwrap();
    let out = h2.matmul(&tw3).unwrap();
    let loss = reduce_to_scalar(&out);
    let grads = backward(&loss).unwrap();
    let grad_w1 = grads.get(&w1).unwrap().to_flat_vec::<f32>().unwrap();

    let w2_ref = w2_data;
    let w3_ref = w3_data;
    let fd_w1 = finite_diff_grad(&w1_data, 1e-4, &|wv: Vec<f32>| {
        let x_t = DynTensor::from_vec(vec![1.0, -0.5], &[1, 2], &cpu()).unwrap();
        let w1_t = DynTensor::from_vec(wv, &[2, 2], &cpu()).unwrap();
        let w2_t = DynTensor::from_vec(w2_ref.clone(), &[2, 2], &cpu()).unwrap();
        let w3_t = DynTensor::from_vec(w3_ref.clone(), &[2, 1], &cpu()).unwrap();
        let h1 = x_t.matmul(&w1_t).unwrap().tanh().unwrap();
        let h2 = h1.matmul(&w2_t).unwrap().sigmoid().unwrap();
        let out = h2.matmul(&w3_t).unwrap();
        out.to_flat_vec::<f32>()
            .unwrap()
            .iter()
            .map(|&v| f64::from(v))
            .sum()
    });

    for i in 0..w1_data.len() {
        let err = (f64::from(grad_w1[i]) - fd_w1[i]).abs();
        assert!(
            err < 1e-2,
            "5-layer mlp grad_w1[{i}]: analytical={}, numerical={}",
            grad_w1[i],
            fd_w1[i]
        );
    }
}

// ===========================================================================
// 7. Zero gradient handling
// ===========================================================================

#[test]
fn test_zero_gradient_relu_all_negative() {
    // All inputs negative: relu kills all gradients
    let x = vec_var(vec![-1.0, -2.0, -3.0, -4.0]);
    let t = Arc::new(TrackedTensor::from_var(&x).unwrap());
    let y = t.relu().unwrap();
    let loss = reduce_to_scalar(&y);
    let grads = backward(&loss).unwrap();
    let grad = grads.get(&x).unwrap().to_flat_vec::<f32>().unwrap();
    for (i, &g) in grad.iter().enumerate() {
        assert!(
            g.abs() < 1e-6,
            "relu with all-negative input: grad[{i}] should be 0, got {g}"
        );
    }
}

#[test]
fn test_zero_gradient_clamp_all_outside() {
    // All values outside clamp range -> zero gradient
    let x = vec_var(vec![5.0, 6.0, -5.0, -6.0]);
    let t = Arc::new(TrackedTensor::from_var(&x).unwrap());
    let y = t.clamp(-1.0, 1.0).unwrap();
    let loss = reduce_to_scalar(&y);
    let grads = backward(&loss).unwrap();
    let grad = grads.get(&x).unwrap().to_flat_vec::<f32>().unwrap();
    for (i, &g) in grad.iter().enumerate() {
        assert!(
            g.abs() < 1e-6,
            "clamp all-outside: grad[{i}] should be 0, got {g}"
        );
    }
}

#[test]
fn test_zero_gradient_mul_by_zero_constant() {
    // Multiplying by a zero constant: gradient through the variable is zero
    let x = scalar_var(5.0);
    let t = Arc::new(TrackedTensor::from_var(&x).unwrap());
    let y = t.mul_scalar(0.0).unwrap();
    let grads = backward(&y).unwrap();
    let grad = grads.get(&x).unwrap().to_flat_vec::<f32>().unwrap();
    assert!(
        grad[0].abs() < 1e-6,
        "mul by 0: gradient should be 0, got {}",
        grad[0]
    );
}

#[test]
fn test_zero_gradient_add_zero_constant() {
    // Adding zero constant: gradient is 1 (identity gradient)
    let x = scalar_var(5.0);
    let t = Arc::new(TrackedTensor::from_var(&x).unwrap());
    let y = t.add_scalar(0.0).unwrap();
    let grads = backward(&y).unwrap();
    let grad = grads.get(&x).unwrap().to_flat_vec::<f32>().unwrap();
    assert!(
        (grad[0] - 1.0).abs() < 1e-6,
        "add 0: gradient should be 1.0, got {}",
        grad[0]
    );
}

// ===========================================================================
// 8. Gradient through broadcast + mul pattern (normalization pattern)
// ===========================================================================

#[test]
fn test_gradient_broadcast_mul_pattern() {
    // Simulates weight * normalized (the broadcast pattern in normalization layers)
    // x: [2, 3], gamma: [3] -> unsqueeze(0) -> broadcast -> mul -> sum
    let x_data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let gamma_data = vec![0.5, 1.0, 2.0];

    let x = Var::new(DynTensor::from_vec(x_data, &[2, 3], &cpu()).unwrap());
    let gamma = vec_var(gamma_data);

    let tx = Arc::new(TrackedTensor::from_var(&x).unwrap());
    let tg = Arc::new(TrackedTensor::from_var(&gamma).unwrap());

    // broadcast gamma [3] -> [1, 3] -> [2, 3]
    let tg_2d = tg.unsqueeze(0).unwrap();
    let tg_broadcast = tg_2d.broadcast_as(&[2, 3]).unwrap();
    let y = tx.mul(&tg_broadcast).unwrap();
    let loss = reduce_to_scalar(&y);
    let grads = backward(&loss).unwrap();

    // grad_x[i,j] = gamma[j]
    let grad_x = grads.get(&x).unwrap().to_flat_vec::<f32>().unwrap();
    // Row 0: [0.5, 1.0, 2.0], Row 1: [0.5, 1.0, 2.0]
    assert!((grad_x[0] - 0.5).abs() < 1e-5);
    assert!((grad_x[1] - 1.0).abs() < 1e-5);
    assert!((grad_x[2] - 2.0).abs() < 1e-5);
    assert!((grad_x[3] - 0.5).abs() < 1e-5);
    assert!((grad_x[4] - 1.0).abs() < 1e-5);
    assert!((grad_x[5] - 2.0).abs() < 1e-5);

    // grad_gamma[j] = sum over batch of x[i,j]
    let grad_gamma = grads.get(&gamma).unwrap().to_flat_vec::<f32>().unwrap();
    assert!((grad_gamma[0] - 5.0).abs() < 1e-5, "sum(1,4)=5"); // x[0,0]+x[1,0] = 1+4
    assert!((grad_gamma[1] - 7.0).abs() < 1e-5, "sum(2,5)=7"); // x[0,1]+x[1,1] = 2+5
    assert!((grad_gamma[2] - 9.0).abs() < 1e-5, "sum(3,6)=9"); // x[0,2]+x[1,2] = 3+6
}

#[test]
fn test_gradient_broadcast_add_bias_pattern() {
    // Simulates output + bias pattern: x: [2, 3], bias: [3]
    let x_data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let bias_data = vec![0.1, 0.2, 0.3];

    let x = Var::new(DynTensor::from_vec(x_data.clone(), &[2, 3], &cpu()).unwrap());
    let bias = vec_var(bias_data.clone());

    let tx = Arc::new(TrackedTensor::from_var(&x).unwrap());
    let tb = Arc::new(TrackedTensor::from_var(&bias).unwrap());

    let tb_2d = tb.unsqueeze(0).unwrap();
    let tb_broadcast = tb_2d.broadcast_as(&[2, 3]).unwrap();
    let y = tx.add(&tb_broadcast).unwrap();
    // Non-trivial loss
    let loss = reduce_to_scalar(&y.sqr().unwrap());
    let grads = backward(&loss).unwrap();

    let grad_x = grads.get(&x).unwrap().to_flat_vec::<f32>().unwrap();
    let grad_bias = grads.get(&bias).unwrap().to_flat_vec::<f32>().unwrap();

    // FD check on bias
    let x_ref = x_data;
    let fd_bias = finite_diff_grad(&bias_data, 1e-4, &|bv: Vec<f32>| {
        let x_t = DynTensor::from_vec(x_ref.clone(), &[2, 3], &cpu()).unwrap();
        let b_t = DynTensor::from_vec(bv, &[3], &cpu()).unwrap();
        let b_2d = b_t.unsqueeze(0).unwrap();
        let b_bc = b_2d.expand([2, 3]).unwrap();
        let y_t = x_t.add(&b_bc).unwrap();
        let s = y_t.sqr().unwrap();
        s.to_flat_vec::<f32>()
            .unwrap()
            .iter()
            .map(|&v| f64::from(v))
            .sum()
    });

    for i in 0..bias_data.len() {
        let err = (f64::from(grad_bias[i]) - fd_bias[i]).abs();
        assert!(
            err < 5e-2,
            "broadcast add bias grad[{i}]: analytical={}, numerical={}, err={}",
            grad_bias[i],
            fd_bias[i],
            err
        );
    }

    // Also check grad_x is finite and correct shape
    assert_eq!(grads.get(&x).unwrap().dims(), &[2, 3]);
    for &g in &grad_x {
        assert!(g.is_finite());
    }
}

// ===========================================================================
// 9. GradStore accumulation correctness
// ===========================================================================

#[test]
fn test_grad_store_var_grads_iter() {
    // backward produces gradients for all variables; verify iteration.
    let w1 = scalar_var(1.0);
    let w2 = scalar_var(2.0);
    let tw1 = Arc::new(TrackedTensor::from_var(&w1).unwrap());
    let tw2 = Arc::new(TrackedTensor::from_var(&w2).unwrap());
    let y = tw1.add(&tw2).unwrap();
    let grads = backward(&y).unwrap();

    let mut count = 0;
    for (_, grad) in grads.var_grads() {
        assert_eq!(grad.dims(), &[1]);
        count += 1;
    }
    assert_eq!(count, 2, "should have gradients for both w1 and w2");
}

#[test]
fn test_grad_store_retain_only_multiple() {
    // Test retain_only with multiple targets
    let w1 = scalar_var(1.0);
    let w2 = scalar_var(2.0);
    let w3 = scalar_var(3.0);
    let tw1 = Arc::new(TrackedTensor::from_var(&w1).unwrap());
    let tw2 = Arc::new(TrackedTensor::from_var(&w2).unwrap());
    let tw3 = Arc::new(TrackedTensor::from_var(&w3).unwrap());
    let y = tw1.add(&tw2).unwrap().add(&tw3).unwrap();
    let mut grads = backward(&y).unwrap();
    assert_eq!(grads.var_count(), 3);

    grads.retain_only(&[&w1, &w3]);
    assert_eq!(grads.var_count(), 2);
    assert!(grads.get(&w1).is_some());
    assert!(grads.get(&w2).is_none());
    assert!(grads.get(&w3).is_some());
}

#[test]
fn test_backward_for_vars_with_multiple_targets() {
    let w1 = scalar_var(1.0);
    let w2 = scalar_var(2.0);
    let w3 = scalar_var(3.0);
    let tw1 = Arc::new(TrackedTensor::from_var(&w1).unwrap());
    let tw2 = Arc::new(TrackedTensor::from_var(&w2).unwrap());
    let tw3 = Arc::new(TrackedTensor::from_var(&w3).unwrap());
    let y = tw1.mul(&tw2).unwrap().add(&tw3).unwrap();
    let grads = backward_for_vars(&y, &[&w1, &w3]).unwrap();
    assert_eq!(grads.var_count(), 2);
    assert!(grads.get(&w1).is_some());
    assert!(grads.get(&w2).is_none());
    assert!(grads.get(&w3).is_some());
}

// ===========================================================================
// 10. Memory safety: no gradient tape Arc leak
// ===========================================================================

#[test]
fn test_no_arc_leak_simple_graph() {
    // After backward, dropping the loss and grads should release all Arcs.
    let x = scalar_var(3.0);
    let t = Arc::new(TrackedTensor::from_var(&x).unwrap());
    let y = t.sqr().unwrap(); // y holds Arc to t
    let z = y.exp().unwrap(); // z holds Arc to y
    let _strong_count_t = Arc::strong_count(&t);

    // Run backward
    let grads = backward(&z).unwrap();
    let _grad_val = grads.get(&x).unwrap().to_flat_vec::<f32>().unwrap();

    // After dropping grads and z, t's refcount should decrease
    drop(grads);
    drop(z);
    drop(y);
    // Only our local `t` should remain
    assert_eq!(
        Arc::strong_count(&t),
        1,
        "after dropping everything else, t should have refcount 1, got {}",
        Arc::strong_count(&t)
    );
}

#[test]
fn test_no_arc_leak_large_graph() {
    // Build a chain of 100 ops, verify no leak after cleanup
    let x = scalar_var(1.0);
    let t = Arc::new(TrackedTensor::from_var(&x).unwrap());
    let mut current = Arc::clone(&t);
    for _ in 0..100 {
        current = current.add_scalar(0.01).unwrap();
    }

    let grads = backward(&current).unwrap();
    let _g = grads.get(&x).unwrap().to_flat_vec::<f32>().unwrap();

    drop(grads);
    drop(current);
    assert_eq!(
        Arc::strong_count(&t),
        1,
        "after 100-op chain cleanup, t refcount should be 1"
    );
}

#[test]
fn test_no_arc_leak_diamond_graph() {
    // Diamond: x -> (a, b) -> c = a + b
    // After cleanup, x's tracked tensor should have refcount 1
    let x = scalar_var(2.0);
    let t = Arc::new(TrackedTensor::from_var(&x).unwrap());
    let a = t.sqr().unwrap();
    let b = t.neg().unwrap();
    let c = a.add(&b).unwrap();

    let grads = backward(&c).unwrap();
    let _g = grads.get(&x).unwrap().to_flat_vec::<f32>().unwrap();

    drop(grads);
    drop(c);
    drop(a);
    drop(b);
    assert_eq!(Arc::strong_count(&t), 1);
}

// ===========================================================================
// 11. Additional edge cases
// ===========================================================================

#[test]
fn test_backward_sqr_gradient() {
    // y = x^2, dy/dx = 2x. At x=0: grad=0.
    let x = scalar_var(0.0);
    let t = Arc::new(TrackedTensor::from_var(&x).unwrap());
    let y = t.sqr().unwrap();
    let grads = backward(&y).unwrap();
    let grad = grads.get(&x).unwrap().to_flat_vec::<f32>().unwrap();
    assert!(
        grad[0].abs() < 1e-6,
        "sqr at 0: gradient should be 0, got {}",
        grad[0]
    );
}

#[test]
fn test_backward_sqrt_gradient() {
    // y = sqrt(x), dy/dx = 1/(2*sqrt(x))
    let x = scalar_var(4.0);
    let t = Arc::new(TrackedTensor::from_var(&x).unwrap());
    let y = t.sqrt().unwrap();
    let grads = backward(&y).unwrap();
    let grad = grads.get(&x).unwrap().to_flat_vec::<f32>().unwrap();
    let expected = 1.0 / (2.0 * 4.0_f32.sqrt());
    assert!(
        (grad[0] - expected).abs() < 1e-5,
        "sqrt(4) grad: expected {expected}, got {}",
        grad[0]
    );
}

#[test]
fn test_backward_gelu_fd() {
    // GELU gradient via finite differences
    let x_data = vec![-2.0, -1.0, 0.0, 1.0, 2.0];
    let x = vec_var(x_data.clone());
    let t = Arc::new(TrackedTensor::from_var(&x).unwrap());
    let y = t.gelu().unwrap();
    let loss = reduce_to_scalar(&y);
    let grads = backward(&loss).unwrap();
    let grad = grads.get(&x).unwrap().to_flat_vec::<f32>().unwrap();

    let fd = finite_diff_grad(&x_data, 1e-4, &|xv: Vec<f32>| {
        let t = DynTensor::from_vec(xv, &[5], &cpu()).unwrap();
        let g = t.gelu().unwrap();
        g.to_flat_vec::<f32>()
            .unwrap()
            .iter()
            .map(|&v| f64::from(v))
            .sum()
    });

    for i in 0..x_data.len() {
        let err = (f64::from(grad[i]) - fd[i]).abs();
        assert!(
            err < 1e-2,
            "gelu grad[{i}]: analytical={}, numerical={}, err={}",
            grad[i],
            fd[i],
            err
        );
    }
}

#[test]
fn test_backward_silu_fd() {
    // SiLU = x * sigmoid(x), gradient via FD
    let x_data = vec![-2.0, -1.0, 0.0, 1.0, 2.0];
    let x = vec_var(x_data.clone());
    let t = Arc::new(TrackedTensor::from_var(&x).unwrap());
    let y = t.silu().unwrap();
    let loss = reduce_to_scalar(&y);
    let grads = backward(&loss).unwrap();
    let grad = grads.get(&x).unwrap().to_flat_vec::<f32>().unwrap();

    let fd = finite_diff_grad(&x_data, 1e-4, &|xv: Vec<f32>| {
        let t = DynTensor::from_vec(xv, &[5], &cpu()).unwrap();
        let s = t.silu().unwrap();
        s.to_flat_vec::<f32>()
            .unwrap()
            .iter()
            .map(|&v| f64::from(v))
            .sum()
    });

    for i in 0..x_data.len() {
        let err = (f64::from(grad[i]) - fd[i]).abs();
        assert!(
            err < 1e-2,
            "silu grad[{i}]: analytical={}, numerical={}, err={}",
            grad[i],
            fd[i],
            err
        );
    }
}

#[test]
fn test_backward_hard_sigmoid_fd() {
    let x_data = vec![-4.0, -1.0, 0.0, 1.0, 4.0];
    let x = vec_var(x_data.clone());
    let t = Arc::new(TrackedTensor::from_var(&x).unwrap());
    let y = t.hard_sigmoid().unwrap();
    let loss = reduce_to_scalar(&y);
    let grads = backward(&loss).unwrap();
    let grad = grads.get(&x).unwrap().to_flat_vec::<f32>().unwrap();

    let fd = finite_diff_grad(&x_data, 1e-4, &|xv: Vec<f32>| {
        let t = DynTensor::from_vec(xv, &[5], &cpu()).unwrap();
        let h = t.hard_sigmoid().unwrap();
        h.to_flat_vec::<f32>()
            .unwrap()
            .iter()
            .map(|&v| f64::from(v))
            .sum()
    });

    for i in 0..x_data.len() {
        let err = (f64::from(grad[i]) - fd[i]).abs();
        assert!(
            err < 1e-2,
            "hard_sigmoid grad[{i}]: analytical={}, numerical={}, err={}",
            grad[i],
            fd[i],
            err
        );
    }
}

#[test]
fn test_backward_hard_swish_fd() {
    let x_data = vec![-4.0, -1.0, 0.0, 1.0, 4.0];
    let x = vec_var(x_data.clone());
    let t = Arc::new(TrackedTensor::from_var(&x).unwrap());
    let y = t.hard_swish().unwrap();
    let loss = reduce_to_scalar(&y);
    let grads = backward(&loss).unwrap();
    let grad = grads.get(&x).unwrap().to_flat_vec::<f32>().unwrap();

    let fd = finite_diff_grad(&x_data, 1e-4, &|xv: Vec<f32>| {
        let t = DynTensor::from_vec(xv, &[5], &cpu()).unwrap();
        let h = t.hard_swish().unwrap();
        h.to_flat_vec::<f32>()
            .unwrap()
            .iter()
            .map(|&v| f64::from(v))
            .sum()
    });

    for i in 0..x_data.len() {
        let err = (f64::from(grad[i]) - fd[i]).abs();
        assert!(
            err < 1e-2,
            "hard_swish grad[{i}]: analytical={}, numerical={}, err={}",
            grad[i],
            fd[i],
            err
        );
    }
}

#[test]
fn test_backward_mish_fd() {
    let x_data = vec![-2.0, -1.0, 0.0, 1.0, 2.0];
    let x = vec_var(x_data.clone());
    let t = Arc::new(TrackedTensor::from_var(&x).unwrap());
    let y = t.mish().unwrap();
    let loss = reduce_to_scalar(&y);
    let grads = backward(&loss).unwrap();
    let grad = grads.get(&x).unwrap().to_flat_vec::<f32>().unwrap();

    let fd = finite_diff_grad(&x_data, 1e-4, &|xv: Vec<f32>| {
        let t = DynTensor::from_vec(xv, &[5], &cpu()).unwrap();
        let m = t.mish().unwrap();
        m.to_flat_vec::<f32>()
            .unwrap()
            .iter()
            .map(|&v| f64::from(v))
            .sum()
    });

    for i in 0..x_data.len() {
        let err = (f64::from(grad[i]) - fd[i]).abs();
        assert!(
            err < 1e-2,
            "mish grad[{i}]: analytical={}, numerical={}, err={}",
            grad[i],
            fd[i],
            err
        );
    }
}

#[test]
fn test_backward_selu_fd() {
    // Avoid x=0 where SELU has a discontinuous derivative
    let x_data = vec![-2.0, -1.0, 0.5, 1.0, 2.0];
    let x = vec_var(x_data.clone());
    let t = Arc::new(TrackedTensor::from_var(&x).unwrap());
    let y = t.selu().unwrap();
    let loss = reduce_to_scalar(&y);
    let grads = backward(&loss).unwrap();
    let grad = grads.get(&x).unwrap().to_flat_vec::<f32>().unwrap();

    let fd = finite_diff_grad(&x_data, 1e-4, &|xv: Vec<f32>| {
        let t = DynTensor::from_vec(xv, &[5], &cpu()).unwrap();
        let s = t.selu().unwrap();
        s.to_flat_vec::<f32>()
            .unwrap()
            .iter()
            .map(|&v| f64::from(v))
            .sum()
    });

    for i in 0..x_data.len() {
        let err = (f64::from(grad[i]) - fd[i]).abs();
        assert!(
            err < 1e-2,
            "selu grad[{i}]: analytical={}, numerical={}, err={}",
            grad[i],
            fd[i],
            err
        );
    }
}

#[test]
fn test_backward_softplus_fd() {
    let x_data = vec![-2.0, -1.0, 0.0, 1.0, 2.0];
    let x = vec_var(x_data.clone());
    let t = Arc::new(TrackedTensor::from_var(&x).unwrap());
    let y = t.softplus().unwrap();
    let loss = reduce_to_scalar(&y);
    let grads = backward(&loss).unwrap();
    let grad = grads.get(&x).unwrap().to_flat_vec::<f32>().unwrap();

    let fd = finite_diff_grad(&x_data, 1e-4, &|xv: Vec<f32>| {
        let t = DynTensor::from_vec(xv, &[5], &cpu()).unwrap();
        let s = t.softplus().unwrap();
        s.to_flat_vec::<f32>()
            .unwrap()
            .iter()
            .map(|&v| f64::from(v))
            .sum()
    });

    for i in 0..x_data.len() {
        let err = (f64::from(grad[i]) - fd[i]).abs();
        assert!(
            err < 1e-2,
            "softplus grad[{i}]: analytical={}, numerical={}, err={}",
            grad[i],
            fd[i],
            err
        );
    }
}

#[test]
fn test_backward_celu_fd() {
    let alpha = 1.5;
    let x_data = vec![-2.0, -1.0, 0.0, 1.0, 2.0];
    let x = vec_var(x_data.clone());
    let t = Arc::new(TrackedTensor::from_var(&x).unwrap());
    let y = t.celu(alpha).unwrap();
    let loss = reduce_to_scalar(&y);
    let grads = backward(&loss).unwrap();
    let grad = grads.get(&x).unwrap().to_flat_vec::<f32>().unwrap();

    let fd = finite_diff_grad(&x_data, 1e-4, &|xv: Vec<f32>| {
        let t = DynTensor::from_vec(xv, &[5], &cpu()).unwrap();
        let c = t.celu(alpha).unwrap();
        c.to_flat_vec::<f32>()
            .unwrap()
            .iter()
            .map(|&v| f64::from(v))
            .sum()
    });

    for i in 0..x_data.len() {
        let err = (f64::from(grad[i]) - fd[i]).abs();
        assert!(
            err < 1e-2,
            "celu grad[{i}]: analytical={}, numerical={}, err={}",
            grad[i],
            fd[i],
            err
        );
    }
}

// ===========================================================================
// 12. Var and VarMap edge cases
// ===========================================================================

#[test]
fn test_var_set_rejects_shape_mismatch() {
    let v = Var::new(DynTensor::from_vec(vec![1.0, 2.0], &[2], &cpu()).unwrap());
    let new_data = DynTensor::from_vec(vec![1.0, 2.0, 3.0], &[3], &cpu()).unwrap();
    let result = v.set(&new_data);
    assert!(result.is_err());
}

#[test]
fn test_var_set_rejects_dtype_mismatch() {
    let v = Var::new(DynTensor::from_vec(vec![1.0, 2.0], &[2], &cpu()).unwrap());
    let new_data = DynTensor::zeros(&[2], DType::BF16, &cpu()).unwrap();
    let result = v.set(&new_data);
    assert!(result.is_err());
}

#[test]
fn test_varmap_get_creates_zero() {
    let mut map = VarMap::new();
    let v = map.get("weight", &[3, 4], DType::F32, &cpu()).unwrap();
    assert_eq!(v.dims().unwrap(), vec![3, 4]);
    let data = v.data().unwrap().to_flat_vec::<f32>().unwrap();
    for &val in &data {
        assert_eq!(val, 0.0);
    }
}

#[test]
fn test_varmap_get_rejects_shape_mismatch() {
    let mut map = VarMap::new();
    let _v = map.get("weight", &[3, 4], DType::F32, &cpu()).unwrap();
    let result = map.get("weight", &[4, 3], DType::F32, &cpu());
    assert!(result.is_err());
}

#[test]
fn test_varmap_get_returns_same_var() {
    let mut map = VarMap::new();
    let v1 = map.get("weight", &[3, 4], DType::F32, &cpu()).unwrap();
    let v2 = map.get("weight", &[3, 4], DType::F32, &cpu()).unwrap();
    assert_eq!(v1.id(), v2.id());
}

#[test]
fn test_varmap_all_vars() {
    let mut map = VarMap::new();
    let _v1 = map.get("w1", &[2], DType::F32, &cpu()).unwrap();
    let _v2 = map.get("w2", &[3], DType::F32, &cpu()).unwrap();
    let all = map.all_vars();
    assert_eq!(all.len(), 2);
}

#[test]
fn test_varmap_to_tensors() {
    let mut map = VarMap::new();
    let v = map.get("weight", &[2, 3], DType::F32, &cpu()).unwrap();
    v.set(&DynTensor::from_vec(vec![1.0; 6], &[2, 3], &cpu()).unwrap())
        .unwrap();
    let tensors = map.to_tensors().unwrap();
    assert_eq!(tensors.len(), 1);
    assert!(tensors.contains_key("weight"));
    assert_eq!(tensors["weight"].dims(), &[2, 3]);
}

// ===========================================================================
// 13. Multi-backward independence
// ===========================================================================

#[test]
fn test_multiple_backward_different_graphs_independent() {
    // Two completely independent computation graphs on the same variable
    // Each backward should produce correct independent gradients
    let x = scalar_var(3.0);

    // Graph 1: loss = x^2, grad = 2x = 6
    let t1 = Arc::new(TrackedTensor::from_var(&x).unwrap());
    let loss1 = t1.sqr().unwrap();
    let grads1 = backward(&loss1).unwrap();

    // Graph 2: loss = x^3, grad = 3x^2 = 27
    let t2 = Arc::new(TrackedTensor::from_var(&x).unwrap());
    let loss2 = t2.powf(3.0).unwrap();
    let grads2 = backward(&loss2).unwrap();

    // Graph 3: loss = exp(x), grad = exp(3)
    let t3 = Arc::new(TrackedTensor::from_var(&x).unwrap());
    let loss3 = t3.exp().unwrap();
    let grads3 = backward(&loss3).unwrap();

    let g1 = grads1.get(&x).unwrap().to_flat_vec::<f32>().unwrap()[0];
    let g2 = grads2.get(&x).unwrap().to_flat_vec::<f32>().unwrap()[0];
    let g3 = grads3.get(&x).unwrap().to_flat_vec::<f32>().unwrap()[0];

    assert!(
        (g1 - 6.0).abs() < 1e-4,
        "graph1 grad: expected 6.0, got {g1}"
    );
    assert!(
        (g2 - 27.0).abs() < 1e-2,
        "graph2 grad: expected 27.0, got {g2}"
    );
    assert!(
        (g3 - 3.0_f32.exp()).abs() < 1e-3,
        "graph3 grad: expected {}, got {g3}",
        3.0_f32.exp()
    );
}

// ===========================================================================
// 14. Dropout backward
// ===========================================================================

#[test]
fn test_backward_dropout_zero_rate() {
    // Dropout with p=0: no elements dropped, identity pass-through
    // Gradient should pass through unchanged
    let x = vec_var(vec![1.0, 2.0, 3.0, 4.0]);
    let t = Arc::new(TrackedTensor::from_var(&x).unwrap());
    let y = t.dropout(0.0).unwrap(); // p=0 means no dropout
    let loss = reduce_to_scalar(&y);
    let grads = backward(&loss).unwrap();
    let grad = grads.get(&x).unwrap().to_flat_vec::<f32>().unwrap();
    for (i, &g) in grad.iter().enumerate() {
        assert!(
            (g - 1.0).abs() < 1e-5,
            "dropout(0.0) grad[{i}]: expected 1.0, got {g}"
        );
    }
}

#[test]
fn test_backward_dropout_inference_mode() {
    // In eval mode (training=false), dropout is identity
    let x = vec_var(vec![1.0, 2.0, 3.0, 4.0]);
    let t = Arc::new(TrackedTensor::from_var(&x).unwrap());
    let y = t.dropout_train(0.5, false).unwrap(); // training=false
    let loss = reduce_to_scalar(&y);
    let grads = backward(&loss).unwrap();
    let grad = grads.get(&x).unwrap().to_flat_vec::<f32>().unwrap();
    for (i, &g) in grad.iter().enumerate() {
        assert!(
            (g - 1.0).abs() < 1e-5,
            "dropout(eval) grad[{i}]: expected 1.0, got {g}"
        );
    }
}

// ===========================================================================
// 15. RmsNorm backward FD
// ===========================================================================

#[test]
fn test_backward_rms_norm_fd() {
    let x_data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]; // [2, 3]
    let weight_data = vec![1.0, 0.5, 2.0]; // [3]

    let x = Var::new(DynTensor::from_vec(x_data.clone(), &[2, 3], &cpu()).unwrap());
    let weight = vec_var(weight_data.clone());

    let tx = Arc::new(TrackedTensor::from_var(&x).unwrap());
    let tw = Arc::new(TrackedTensor::from_var(&weight).unwrap());

    let y = tx.rms_norm(&tw, 1e-5).unwrap();
    let loss = reduce_to_scalar(&y);
    let grads = backward(&loss).unwrap();

    let grad_x = grads.get(&x).unwrap().to_flat_vec::<f32>().unwrap();

    // FD check on x
    let w_ref = weight_data;
    let fd_x = finite_diff_grad(&x_data, 1e-3, &|xv: Vec<f32>| {
        let x_t = DynTensor::from_vec(xv, &[2, 3], &cpu()).unwrap();
        let w_t = DynTensor::from_vec(w_ref.clone(), &[3], &cpu()).unwrap();
        // RMS norm: x / rms(x) * weight
        let sq = x_t.sqr().unwrap();
        let mean_sq = sq.mean_keepdim(1).unwrap();
        let rms = mean_sq.add_scalar(1e-5).unwrap().sqrt().unwrap();
        let normed = x_t.div(&rms).unwrap();
        let out = normed.mul(&w_t).unwrap();
        out.to_flat_vec::<f32>()
            .unwrap()
            .iter()
            .map(|&v| f64::from(v))
            .sum()
    });

    for i in 0..x_data.len() {
        let err = (f64::from(grad_x[i]) - fd_x[i]).abs();
        assert!(
            err < 5e-2,
            "rms_norm grad_x[{i}]: analytical={}, numerical={}, err={}",
            grad_x[i],
            fd_x[i],
            err
        );
    }
}

// ===========================================================================
// 16. Mean backward with non-trivial loss
// ===========================================================================

#[test]
fn test_backward_mean_keepdim_with_sqr_loss_fd() {
    // x: [3, 4], mean over dim=1 -> [3, 1], then sqr -> sum
    let x_data: Vec<f32> = (1..=12).map(|v| v as f32 * 0.1).collect();
    let x = Var::new(DynTensor::from_vec(x_data.clone(), &[3, 4], &cpu()).unwrap());
    let t = Arc::new(TrackedTensor::from_var(&x).unwrap());
    let m = t.mean_keepdim(1).unwrap(); // [3, 1]
    let loss = reduce_to_scalar(&m.sqr().unwrap());
    let grads = backward(&loss).unwrap();
    let grad = grads.get(&x).unwrap().to_flat_vec::<f32>().unwrap();

    let fd = finite_diff_grad(&x_data, 1e-4, &|xv: Vec<f32>| {
        let t = DynTensor::from_vec(xv, &[3, 4], &cpu()).unwrap();
        let m = t.mean_keepdim(1).unwrap();
        let s = m.sqr().unwrap();
        s.to_flat_vec::<f32>()
            .unwrap()
            .iter()
            .map(|&v| f64::from(v))
            .sum()
    });

    for i in 0..x_data.len() {
        let err = (f64::from(grad[i]) - fd[i]).abs();
        assert!(
            err < 1e-2,
            "mean->sqr grad[{i}]: analytical={}, numerical={}, err={}",
            grad[i],
            fd[i],
            err
        );
    }
}

#![allow(deprecated)]
// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Finite-difference gradient tests for ELU, LogSoftmax, Maximum, Minimum, Stack.
//!
//! Verifies backward rules match numerical derivatives for the 5 ops
//! added in #1529. Includes edge-case coverage: equal-value subgradient
//! conservation (Minimum/Maximum) and diamond-graph gradient accumulation (Stack).

use std::sync::Arc;

use nn_core::dyn_tensor::DynTensor;
use nn_core::test_utils::cpu;

use crate::backward;
use crate::tracked::TrackedTensor;
use crate::var::Var;

use super::test_helpers::{check_fd_grad, check_fd_grad_tol, sum_f64, vec_var};

// ── ELU ─────────────────────────────────────────────────────────────

#[test]
fn test_elu_fd_positive() {
    // ELU(x, alpha) = x for x > 0; derivative = 1
    let data = vec![0.5, 1.0, 2.0];
    let x = vec_var(data.clone());
    let t = Arc::new(TrackedTensor::from_var(&x).unwrap());
    let y = t.elu(1.0).unwrap();
    let loss = y.sum_keepdim(0).unwrap();
    let grads = backward(&loss).unwrap();
    let g = grads.get(&x).unwrap().to_flat_vec::<f32>().unwrap();

    check_fd_grad(&g, &data, 1e-3, |d| {
        let t = DynTensor::from_vec(d, &[3], &cpu()).unwrap();
        sum_f64(&t.elu(1.0).unwrap())
    });
}

#[test]
fn test_elu_fd_negative() {
    // ELU(x, alpha) = alpha * (exp(x) - 1) for x <= 0
    let data = vec![-0.5, -1.0, -2.0];
    let x = vec_var(data.clone());
    let t = Arc::new(TrackedTensor::from_var(&x).unwrap());
    let y = t.elu(1.0).unwrap();
    let loss = y.sum_keepdim(0).unwrap();
    let grads = backward(&loss).unwrap();
    let g = grads.get(&x).unwrap().to_flat_vec::<f32>().unwrap();

    check_fd_grad(&g, &data, 1e-3, |d| {
        let t = DynTensor::from_vec(d, &[3], &cpu()).unwrap();
        sum_f64(&t.elu(1.0).unwrap())
    });
}

#[test]
fn test_elu_fd_mixed_alpha2() {
    // Mixed positive/negative with alpha=2.0
    let data = vec![-1.0, 0.5, -0.3, 1.5];
    let n = data.len();
    let x = vec_var(data.clone());
    let t = Arc::new(TrackedTensor::from_var(&x).unwrap());
    let y = t.elu(2.0).unwrap();
    let loss = y.sum_keepdim(0).unwrap();
    let grads = backward(&loss).unwrap();
    let g = grads.get(&x).unwrap().to_flat_vec::<f32>().unwrap();

    check_fd_grad(&g, &data, 1e-3, |d| {
        let t = DynTensor::from_vec(d, &[n], &cpu()).unwrap();
        sum_f64(&t.elu(2.0).unwrap())
    });
}

// ── LogSoftmax ──────────────────────────────────────────────────────

#[test]
fn test_log_softmax_fd() {
    let data = vec![1.0, 2.0, 3.0];
    let x = vec_var(data.clone());
    let t = Arc::new(TrackedTensor::from_var(&x).unwrap());
    let y = t.log_softmax(0).unwrap();
    let loss = y.sum_keepdim(0).unwrap();
    let grads = backward(&loss).unwrap();
    let g = grads.get(&x).unwrap().to_flat_vec::<f32>().unwrap();

    check_fd_grad(&g, &data, 1e-3, |d| {
        let t = DynTensor::from_vec(d, &[3], &cpu()).unwrap();
        sum_f64(&t.log_softmax(0).unwrap())
    });
}

#[test]
fn test_log_softmax_fd_2d() {
    // 2D: [2, 3] log_softmax along dim=1
    let data = vec![1.0, 2.0, 3.0, 0.5, 1.5, 2.5];
    let x = Var::new(DynTensor::from_vec(data.clone(), &[2, 3], &cpu()).unwrap());
    let t = Arc::new(TrackedTensor::from_var(&x).unwrap());
    let y = t.log_softmax(1).unwrap();
    // sum all → scalar
    let loss = y.sum_keepdim(0).unwrap().sum_keepdim(1).unwrap();
    let grads = backward(&loss).unwrap();
    let g = grads.get(&x).unwrap().to_flat_vec::<f32>().unwrap();

    check_fd_grad(&g, &data, 1e-3, |d| {
        let t = DynTensor::from_vec(d, &[2, 3], &cpu()).unwrap();
        sum_f64(&t.log_softmax(1).unwrap())
    });
}

// ── Maximum ─────────────────────────────────────────────────────────

#[test]
fn test_maximum_fd() {
    // max(a, b): grad flows to whichever is larger
    let data_a = vec![1.0, 3.0, 2.0];
    let data_b = vec![2.0, 1.0, 4.0];
    let a = vec_var(data_a.clone());
    let b = vec_var(data_b.clone());
    let ta = Arc::new(TrackedTensor::from_var(&a).unwrap());
    let tb = Arc::new(TrackedTensor::from_var(&b).unwrap());
    let y = ta.maximum(&tb).unwrap();
    let loss = y.sum_keepdim(0).unwrap();
    let grads = backward(&loss).unwrap();
    let ga = grads.get(&a).unwrap().to_flat_vec::<f32>().unwrap();
    let gb = grads.get(&b).unwrap().to_flat_vec::<f32>().unwrap();

    // FD for a
    check_fd_grad(&ga, &data_a, 1e-3, |d| {
        let ta = DynTensor::from_vec(d, &[3], &cpu()).unwrap();
        let tb = DynTensor::from_vec(data_b.clone(), &[3], &cpu()).unwrap();
        sum_f64(&ta.maximum(&tb).unwrap())
    });

    // FD for b
    check_fd_grad(&gb, &data_b, 1e-3, |d| {
        let ta = DynTensor::from_vec(data_a.clone(), &[3], &cpu()).unwrap();
        let tb = DynTensor::from_vec(d, &[3], &cpu()).unwrap();
        sum_f64(&ta.maximum(&tb).unwrap())
    });
}

#[test]
fn test_maximum_fd_equal_values() {
    // When a == b, subgradient assigns grad to a (ge mask)
    // FD may see 0.5/0.5 split; we use relaxed tolerance
    let a = vec_var(vec![2.0, 2.0]);
    let b = vec_var(vec![2.0, 2.0]);
    let ta = Arc::new(TrackedTensor::from_var(&a).unwrap());
    let tb = Arc::new(TrackedTensor::from_var(&b).unwrap());
    let y = ta.maximum(&tb).unwrap();
    let loss = y.sum_keepdim(0).unwrap();
    let grads = backward(&loss).unwrap();
    let ga = grads.get(&a).unwrap().to_flat_vec::<f32>().unwrap();
    let gb = grads.get(&b).unwrap().to_flat_vec::<f32>().unwrap();

    // At tie, our impl gives grad to a (ge=true for diff==0), b gets 0
    // FD sees derivative ~0.5 for both since either perturbation shifts max
    // Just check analytical grads sum to 1.0 per element (conservation)
    for i in 0..ga.len() {
        let total = ga[i] + gb[i];
        assert!(
            (total - 1.0).abs() < 1e-4,
            "max grad sum[{i}]={total}, expected 1.0"
        );
    }
}

// ── Minimum ─────────────────────────────────────────────────────────

#[test]
fn test_minimum_fd_equal_values() {
    // When a == b, subgradient assigns grad to a (le mask)
    // FD may see 0.5/0.5 split; we use relaxed tolerance
    let a = vec_var(vec![2.0, 2.0]);
    let b = vec_var(vec![2.0, 2.0]);
    let ta = Arc::new(TrackedTensor::from_var(&a).unwrap());
    let tb = Arc::new(TrackedTensor::from_var(&b).unwrap());
    let y = ta.minimum(&tb).unwrap();
    let loss = y.sum_keepdim(0).unwrap();
    let grads = backward(&loss).unwrap();
    let ga = grads.get(&a).unwrap().to_flat_vec::<f32>().unwrap();
    let gb = grads.get(&b).unwrap().to_flat_vec::<f32>().unwrap();

    // At tie, our impl gives grad to a (le=true for diff==0), b gets 0
    // FD sees derivative ~0.5 for both since either perturbation shifts min
    // Just check analytical grads sum to 1.0 per element (conservation)
    for i in 0..ga.len() {
        let total = ga[i] + gb[i];
        assert!(
            (total - 1.0).abs() < 1e-4,
            "min grad sum[{i}]={total}, expected 1.0"
        );
    }
}

#[test]
fn test_minimum_fd() {
    let data_a = vec![1.0, 3.0, 2.0];
    let data_b = vec![2.0, 1.0, 4.0];
    let a = vec_var(data_a.clone());
    let b = vec_var(data_b.clone());
    let ta = Arc::new(TrackedTensor::from_var(&a).unwrap());
    let tb = Arc::new(TrackedTensor::from_var(&b).unwrap());
    let y = ta.minimum(&tb).unwrap();
    let loss = y.sum_keepdim(0).unwrap();
    let grads = backward(&loss).unwrap();
    let ga = grads.get(&a).unwrap().to_flat_vec::<f32>().unwrap();
    let gb = grads.get(&b).unwrap().to_flat_vec::<f32>().unwrap();

    // FD for a
    check_fd_grad(&ga, &data_a, 1e-3, |d| {
        let ta = DynTensor::from_vec(d, &[3], &cpu()).unwrap();
        let tb = DynTensor::from_vec(data_b.clone(), &[3], &cpu()).unwrap();
        sum_f64(&ta.minimum(&tb).unwrap())
    });

    // FD for b
    check_fd_grad(&gb, &data_b, 1e-3, |d| {
        let ta = DynTensor::from_vec(data_a.clone(), &[3], &cpu()).unwrap();
        let tb = DynTensor::from_vec(d, &[3], &cpu()).unwrap();
        sum_f64(&ta.minimum(&tb).unwrap())
    });
}

// ── Stack ───────────────────────────────────────────────────────────

#[test]
fn test_stack_fd() {
    // Stack two [3]-vectors along dim=0 → [2, 3], then sqr + sum
    // sqr makes gradient input-dependent: d(sum(x^2))/dx_i = 2*x_i
    let data_a = vec![1.0, 2.0, 3.0];
    let data_b = vec![4.0, 5.0, 6.0];
    let a = vec_var(data_a.clone());
    let b = vec_var(data_b.clone());
    let ta = Arc::new(TrackedTensor::from_var(&a).unwrap());
    let tb = Arc::new(TrackedTensor::from_var(&b).unwrap());
    let y = TrackedTensor::stack(&[ta, tb], 0).unwrap();
    assert_eq!(y.dims(), &[2, 3]);
    let y2 = y.sqr().unwrap();
    let loss = y2.sum_keepdim(0).unwrap().sum_keepdim(1).unwrap();
    let grads = backward(&loss).unwrap();
    let ga = grads.get(&a).unwrap().to_flat_vec::<f32>().unwrap();
    let gb = grads.get(&b).unwrap().to_flat_vec::<f32>().unwrap();

    // FD for a: d(sum(stack(a,b)^2))/da_i = 2*a_i
    check_fd_grad_tol(&ga, &data_a, 1e-3, 1e-2, &|d| {
        let ta = DynTensor::from_vec(d, &[3], &cpu()).unwrap();
        let tb = DynTensor::from_vec(data_b.clone(), &[3], &cpu()).unwrap();
        let s = DynTensor::stack(&[&ta, &tb], 0).unwrap();
        sum_f64(&s.sqr().unwrap())
    });

    // FD for b: d(sum(stack(a,b)^2))/db_i = 2*b_i
    check_fd_grad_tol(&gb, &data_b, 1e-3, 1e-2, &|d| {
        let ta = DynTensor::from_vec(data_a.clone(), &[3], &cpu()).unwrap();
        let tb = DynTensor::from_vec(d, &[3], &cpu()).unwrap();
        let s = DynTensor::stack(&[&ta, &tb], 0).unwrap();
        sum_f64(&s.sqr().unwrap())
    });
}

#[test]
fn test_stack_fd_weighted() {
    // Stack + sqr + mul_scalar: gradient is 4*x_i (input-dependent)
    let data_a = vec![1.0, 2.0];
    let data_b = vec![3.0, 4.0];
    let a = vec_var(data_a.clone());
    let b = vec_var(data_b.clone());
    let ta = Arc::new(TrackedTensor::from_var(&a).unwrap());
    let tb = Arc::new(TrackedTensor::from_var(&b).unwrap());
    let y = TrackedTensor::stack(&[ta, tb], 0).unwrap();
    let y2 = y.sqr().unwrap().mul_scalar(2.0).unwrap();
    let loss = y2.sum_keepdim(0).unwrap().sum_keepdim(1).unwrap();
    let grads = backward(&loss).unwrap();
    let ga = grads.get(&a).unwrap().to_flat_vec::<f32>().unwrap();
    let gb = grads.get(&b).unwrap().to_flat_vec::<f32>().unwrap();

    // FD for a: d(sum(2*stack(a,b)^2))/da_i = 4*a_i
    check_fd_grad_tol(&ga, &data_a, 1e-3, 1e-2, &|d| {
        let ta = DynTensor::from_vec(d, &[2], &cpu()).unwrap();
        let tb = DynTensor::from_vec(data_b.clone(), &[2], &cpu()).unwrap();
        let s = DynTensor::stack(&[&ta, &tb], 0).unwrap();
        sum_f64(&s.sqr().unwrap().mul_scalar(2.0).unwrap())
    });

    // FD for b: d(sum(2*stack(a,b)^2))/db_i = 4*b_i
    check_fd_grad_tol(&gb, &data_b, 1e-3, 1e-2, &|d| {
        let ta = DynTensor::from_vec(data_a.clone(), &[2], &cpu()).unwrap();
        let tb = DynTensor::from_vec(d, &[2], &cpu()).unwrap();
        let s = DynTensor::stack(&[&ta, &tb], 0).unwrap();
        sum_f64(&s.sqr().unwrap().mul_scalar(2.0).unwrap())
    });
}

#[test]
fn test_stack_fd_dim1() {
    // Stack along dim=1: two [2]-vectors → [2, 2]
    // Then weighted sum via sqr to test gradient correctness
    let data_a = vec![1.0, 2.0];
    let data_b = vec![3.0, 4.0];
    let a = vec_var(data_a.clone());
    let b = vec_var(data_b.clone());
    let ta = Arc::new(TrackedTensor::from_var(&a).unwrap());
    let tb = Arc::new(TrackedTensor::from_var(&b).unwrap());
    let y = TrackedTensor::stack(&[ta, tb], 1).unwrap();
    assert_eq!(y.dims(), &[2, 2]);
    let y2 = y.sqr().unwrap();
    let loss = y2.sum_keepdim(0).unwrap().sum_keepdim(1).unwrap();
    let grads = backward(&loss).unwrap();
    let ga = grads.get(&a).unwrap().to_flat_vec::<f32>().unwrap();
    let gb = grads.get(&b).unwrap().to_flat_vec::<f32>().unwrap();

    // FD for a: d(sum(stack(a,b)^2))/da_i = 2*a_i
    check_fd_grad_tol(&ga, &data_a, 1e-3, 1e-2, &|d| {
        let ta = DynTensor::from_vec(d, &[2], &cpu()).unwrap();
        let tb = DynTensor::from_vec(data_b.clone(), &[2], &cpu()).unwrap();
        let s = DynTensor::stack(&[&ta, &tb], 1).unwrap();
        sum_f64(&s.sqr().unwrap())
    });

    // FD for b
    check_fd_grad_tol(&gb, &data_b, 1e-3, 1e-2, &|d| {
        let ta = DynTensor::from_vec(data_a.clone(), &[2], &cpu()).unwrap();
        let tb = DynTensor::from_vec(d, &[2], &cpu()).unwrap();
        let s = DynTensor::stack(&[&ta, &tb], 1).unwrap();
        sum_f64(&s.sqr().unwrap())
    });
}

#[test]
fn test_stack_fd_diamond() {
    // Diamond graph: one variable used as both inputs to stack.
    // stack([x, x], 0) → [2, 3], then sqr + sum.
    // Gradient should be 2*x from first copy + 2*x from second = 4*x total,
    // because `accumulate` is called twice for the same Var.
    let data = vec![1.0, 2.0, 3.0];
    let x = vec_var(data.clone());
    let tx = Arc::new(TrackedTensor::from_var(&x).unwrap());
    let y = TrackedTensor::stack(&[Arc::clone(&tx), Arc::clone(&tx)], 0).unwrap();
    assert_eq!(y.dims(), &[2, 3]);
    let y2 = y.sqr().unwrap();
    let loss = y2.sum_keepdim(0).unwrap().sum_keepdim(1).unwrap();
    let grads = backward(&loss).unwrap();
    let gx = grads.get(&x).unwrap().to_flat_vec::<f32>().unwrap();

    // Analytical: each copy contributes 2*x_i, and accumulate sums → 4*x_i
    for (i, &val) in data.iter().enumerate() {
        let expected = 4.0 * val;
        let err = (gx[i] - expected).abs();
        assert!(
            err < 1e-4,
            "diamond grad[{i}]: got={}, expected={expected}, err={err}",
            gx[i]
        );
    }

    // FD verification: perturb x_i, both copies shift
    check_fd_grad_tol(&gx, &data, 1e-3, 1e-2, &|d| {
        let tx = DynTensor::from_vec(d, &[3], &cpu()).unwrap();
        let s = DynTensor::stack(&[&tx, &tx], 0).unwrap();
        sum_f64(&s.sqr().unwrap())
    });
}

// ── Maximum/Minimum NaN defense (#1999) ─────────────────────────────

#[test]
fn test_maximum_backward_nan_input_returns_error() {
    // When one operand is NaN, backward must return NonFiniteBackwardInput
    // rather than silently dropping the gradient (both IEEE 754 comparison
    // masks become zero).
    let a = vec_var(vec![1.0, f32::NAN, 3.0]);
    let b = vec_var(vec![2.0, 2.0, 1.0]);
    let ta = Arc::new(TrackedTensor::from_var(&a).unwrap());
    let tb = Arc::new(TrackedTensor::from_var(&b).unwrap());
    let y = ta.maximum(&tb).unwrap();
    let loss = y.sum_keepdim(0).unwrap();
    let result = backward(&loss);
    assert!(
        result.is_err(),
        "backward through Maximum with NaN must error"
    );
    let err = result.unwrap_err();
    assert!(
        matches!(
            err,
            crate::AutodiffError::NonFiniteBackwardInput { op: "Maximum" }
        ),
        "expected NonFiniteBackwardInput for Maximum, got: {err:?}"
    );
}

#[test]
fn test_minimum_backward_nan_input_returns_error() {
    let a = vec_var(vec![1.0, 2.0, 3.0]);
    let b = vec_var(vec![2.0, f32::NAN, 1.0]);
    let ta = Arc::new(TrackedTensor::from_var(&a).unwrap());
    let tb = Arc::new(TrackedTensor::from_var(&b).unwrap());
    let y = ta.minimum(&tb).unwrap();
    let loss = y.sum_keepdim(0).unwrap();
    let result = backward(&loss);
    assert!(
        result.is_err(),
        "backward through Minimum with NaN must error"
    );
    let err = result.unwrap_err();
    assert!(
        matches!(
            err,
            crate::AutodiffError::NonFiniteBackwardInput { op: "Minimum" }
        ),
        "expected NonFiniteBackwardInput for Minimum, got: {err:?}"
    );
}

#[test]
fn test_maximum_backward_inf_input_returns_error() {
    // Inf - finite = Inf (non-finite diff), must also be caught.
    let a = vec_var(vec![f32::INFINITY, 2.0]);
    let b = vec_var(vec![1.0, 3.0]);
    let ta = Arc::new(TrackedTensor::from_var(&a).unwrap());
    let tb = Arc::new(TrackedTensor::from_var(&b).unwrap());
    let y = ta.maximum(&tb).unwrap();
    let loss = y.sum_keepdim(0).unwrap();
    let result = backward(&loss);
    assert!(
        result.is_err(),
        "backward through Maximum with Inf must error"
    );
}

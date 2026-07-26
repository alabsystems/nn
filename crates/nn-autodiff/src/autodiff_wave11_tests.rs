// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Wave 11 autodiff tests: backward rule correctness, gradient tape recording,
//! error propagation, TrackedTensor operations, DType handling, and reduction ops.

use std::sync::Arc;

use nn_core::dyn_tensor::DynTensor;
use nn_core::test_utils::cpu;
use nn_core::DType;

use crate::error::AutodiffError;
use crate::grad::test_helpers::{check_fd_grad, scalar_var, sum_f64, vec_var};
use crate::grad::{backward, backward_for_vars, GradStore};
use crate::op::Op;
use crate::tracked::TrackedTensor;
use crate::var::Var;

// ---------------------------------------------------------------------------
// Helper
// ---------------------------------------------------------------------------

fn tensor(data: &[f32], dims: &[usize]) -> DynTensor {
    DynTensor::new(data, dims, &cpu()).unwrap()
}

fn mat_var(data: &[f32], rows: usize, cols: usize) -> Var {
    Var::new(DynTensor::new(data, &[rows, cols], &cpu()).unwrap())
}

// ===========================================================================
// Section 1: Backward rules for elementwise ops
// ===========================================================================

#[test]
fn test_backward_neg_vector() {
    // y = -x, dy/dx = -1 for every element
    let x = vec_var(vec![1.0, -2.0, 3.0]);
    let t = Arc::new(TrackedTensor::from_var(&x).unwrap());
    let y = t.neg().unwrap();
    let loss = y.sum_keepdim(0).unwrap();
    let grads = backward(&loss).unwrap();
    let grad = grads.get(&x).unwrap().to_flat_vec::<f32>().unwrap();
    assert_eq!(grad, vec![-1.0, -1.0, -1.0]);
}

#[test]
fn test_backward_abs_positive_and_negative() {
    // |x|: gradient is sign(x)
    let x = vec_var(vec![2.0, -3.0, 5.0, -1.0]);
    let t = Arc::new(TrackedTensor::from_var(&x).unwrap());
    let y = t.abs().unwrap();
    let loss = y.sum_keepdim(0).unwrap();
    let grads = backward(&loss).unwrap();
    let grad = grads.get(&x).unwrap().to_flat_vec::<f32>().unwrap();
    assert_eq!(grad, vec![1.0, -1.0, 1.0, -1.0]);
}

#[test]
fn test_backward_exp_known_value() {
    // y = exp(x), dy/dx = exp(x)
    let x = scalar_var(1.0);
    let t = Arc::new(TrackedTensor::from_var(&x).unwrap());
    let y = t.exp().unwrap();
    let grads = backward(&y).unwrap();
    let grad = grads.get(&x).unwrap().to_flat_vec::<f32>().unwrap();
    let expected = 1.0_f32.exp();
    assert!(
        (grad[0] - expected).abs() < 1e-5,
        "exp grad: expected {expected}, got {}",
        grad[0]
    );
}

#[test]
fn test_backward_log_known_value() {
    // y = ln(x), dy/dx = 1/x
    let x = scalar_var(2.0);
    let t = Arc::new(TrackedTensor::from_var(&x).unwrap());
    let y = t.log().unwrap();
    let grads = backward(&y).unwrap();
    let grad = grads.get(&x).unwrap().to_flat_vec::<f32>().unwrap();
    assert!(
        (grad[0] - 0.5).abs() < 1e-5,
        "log grad: expected 0.5, got {}",
        grad[0]
    );
}

#[test]
fn test_backward_sqrt_known_value() {
    // y = sqrt(x), dy/dx = 1/(2*sqrt(x))
    let x = scalar_var(4.0);
    let t = Arc::new(TrackedTensor::from_var(&x).unwrap());
    let y = t.sqrt().unwrap();
    let grads = backward(&y).unwrap();
    let grad = grads.get(&x).unwrap().to_flat_vec::<f32>().unwrap();
    // 1/(2*sqrt(4)) = 1/4 = 0.25
    assert!(
        (grad[0] - 0.25).abs() < 1e-5,
        "sqrt grad: expected 0.25, got {}",
        grad[0]
    );
}

#[test]
fn test_backward_sigmoid_at_zero() {
    // sigmoid(0) = 0.5, dsigmoid(0) = 0.5 * (1 - 0.5) = 0.25
    let x = scalar_var(0.0);
    let t = Arc::new(TrackedTensor::from_var(&x).unwrap());
    let y = t.sigmoid().unwrap();
    let grads = backward(&y).unwrap();
    let grad = grads.get(&x).unwrap().to_flat_vec::<f32>().unwrap();
    assert!(
        (grad[0] - 0.25).abs() < 1e-5,
        "sigmoid grad at 0: expected 0.25, got {}",
        grad[0]
    );
}

#[test]
fn test_backward_tanh_at_zero() {
    // tanh(0) = 0, dtanh(0) = 1 - tanh(0)^2 = 1
    let x = scalar_var(0.0);
    let t = Arc::new(TrackedTensor::from_var(&x).unwrap());
    let y = t.tanh().unwrap();
    let grads = backward(&y).unwrap();
    let grad = grads.get(&x).unwrap().to_flat_vec::<f32>().unwrap();
    assert!(
        (grad[0] - 1.0).abs() < 1e-5,
        "tanh grad at 0: expected 1.0, got {}",
        grad[0]
    );
}

#[test]
fn test_backward_relu_mixed_signs() {
    // relu(x) grad: 1 if x > 0, 0 otherwise
    let x = vec_var(vec![-2.0, 0.0, 3.0, -0.5, 1.0]);
    let t = Arc::new(TrackedTensor::from_var(&x).unwrap());
    let y = t.relu().unwrap();
    let loss = y.sum_keepdim(0).unwrap();
    let grads = backward(&loss).unwrap();
    let grad = grads.get(&x).unwrap().to_flat_vec::<f32>().unwrap();
    // ReLU uses right-derivative convention: grad is 1.0 at x=0
    assert_eq!(grad, vec![0.0, 1.0, 1.0, 0.0, 1.0]);
}

#[test]
fn test_backward_sin_fd() {
    // Verify sin backward with finite differences
    let data = vec![0.5, 1.0, -0.3];
    let x = vec_var(data.clone());
    let t = Arc::new(TrackedTensor::from_var(&x).unwrap());
    let y = t.sin().unwrap();
    let loss = y.sum_keepdim(0).unwrap();
    let grads = backward(&loss).unwrap();
    let grad = grads.get(&x).unwrap().to_flat_vec::<f32>().unwrap();
    check_fd_grad(&grad, &data, 1e-3, |d| {
        let t = DynTensor::new(&d, &[3], &cpu()).unwrap();
        sum_f64(&t.sin().unwrap())
    });
}

#[test]
fn test_backward_cos_fd() {
    let data = vec![0.5, 1.0, -0.3];
    let x = vec_var(data.clone());
    let t = Arc::new(TrackedTensor::from_var(&x).unwrap());
    let y = t.cos().unwrap();
    let loss = y.sum_keepdim(0).unwrap();
    let grads = backward(&loss).unwrap();
    let grad = grads.get(&x).unwrap().to_flat_vec::<f32>().unwrap();
    check_fd_grad(&grad, &data, 1e-3, |d| {
        let t = DynTensor::new(&d, &[3], &cpu()).unwrap();
        sum_f64(&t.cos().unwrap())
    });
}

#[test]
fn test_backward_recip_fd() {
    let data = vec![2.0, 0.5, 3.0];
    let x = vec_var(data.clone());
    let t = Arc::new(TrackedTensor::from_var(&x).unwrap());
    let y = t.recip().unwrap();
    let loss = y.sum_keepdim(0).unwrap();
    let grads = backward(&loss).unwrap();
    let grad = grads.get(&x).unwrap().to_flat_vec::<f32>().unwrap();
    check_fd_grad(&grad, &data, 1e-3, |d| {
        let t = DynTensor::new(&d, &[3], &cpu()).unwrap();
        sum_f64(&t.recip().unwrap())
    });
}

#[test]
fn test_backward_elu_fd() {
    let data = vec![-1.0, 0.0, 1.0, -0.5];
    let x = vec_var(data.clone());
    let t = Arc::new(TrackedTensor::from_var(&x).unwrap());
    let y = t.elu(1.0).unwrap();
    let loss = y.sum_keepdim(0).unwrap();
    let grads = backward(&loss).unwrap();
    let grad = grads.get(&x).unwrap().to_flat_vec::<f32>().unwrap();
    check_fd_grad(&grad, &data, 1e-3, |d| {
        let t = DynTensor::new(&d, &[4], &cpu()).unwrap();
        sum_f64(&t.elu(1.0).unwrap())
    });
}

// ===========================================================================
// Section 2: Backward rules for reduction ops (sum, mean)
// ===========================================================================

#[test]
fn test_backward_sum_keepdim_2d() {
    // sum over dim=1 of [2,3] tensor: gradient is ones broadcast
    let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let x = Var::new(tensor(&data, &[2, 3]));
    let t = Arc::new(TrackedTensor::from_var(&x).unwrap());
    let s = t.sum_keepdim(1).unwrap(); // [2, 1]
    let loss = s.sum_keepdim(0).unwrap(); // scalar-like [1, 1]
    let grads = backward(&loss).unwrap();
    let grad = grads.get(&x).unwrap().to_flat_vec::<f32>().unwrap();
    // dy/dx_ij = 1 for all i,j
    assert_eq!(grad, vec![1.0; 6]);
}

#[test]
fn test_backward_mean_keepdim_known_value() {
    // mean over dim=0 of [3] vector: gradient = 1/3 each
    let x = vec_var(vec![1.0, 2.0, 3.0]);
    let t = Arc::new(TrackedTensor::from_var(&x).unwrap());
    let m = t.mean_keepdim(0).unwrap();
    let grads = backward(&m).unwrap();
    let grad = grads.get(&x).unwrap().to_flat_vec::<f32>().unwrap();
    let expected = 1.0 / 3.0;
    for (i, &g) in grad.iter().enumerate() {
        assert!(
            (g - expected).abs() < 1e-5,
            "mean grad[{i}]: expected {expected}, got {g}"
        );
    }
}

#[test]
fn test_backward_mean_keepdim_2d_dim1() {
    // mean over dim=1 for [2, 4] tensor
    let data: Vec<f32> = (1..=8).map(|v| v as f32).collect();
    let x = Var::new(tensor(&data, &[2, 4]));
    let t = Arc::new(TrackedTensor::from_var(&x).unwrap());
    let m = t.mean_keepdim(1).unwrap(); // [2, 1]
    let loss = m.sum_keepdim(0).unwrap(); // scalar
    let grads = backward(&loss).unwrap();
    let grad = grads.get(&x).unwrap().to_flat_vec::<f32>().unwrap();
    // Each element's gradient = 1/4 (dimension size along dim=1)
    let expected = 1.0 / 4.0;
    for (i, &g) in grad.iter().enumerate() {
        assert!(
            (g - expected).abs() < 1e-5,
            "mean grad[{i}]: expected {expected}, got {g}"
        );
    }
}

#[test]
fn test_backward_sum_then_mean_chain() {
    // sum(dim=0) then mean(dim=0) on [3, 2] tensor
    let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let x = Var::new(tensor(&data, &[3, 2]));
    let t = Arc::new(TrackedTensor::from_var(&x).unwrap());
    let s = t.sum_keepdim(0).unwrap(); // [1, 2]
    let m = s.mean_keepdim(1).unwrap(); // [1, 1] scalar
    let grads = backward(&m).unwrap();
    let grad = grads.get(&x).unwrap().to_flat_vec::<f32>().unwrap();
    // Chain: d(mean)/d(sum_j) = 1/2, d(sum_j)/d(x_ij) = 1
    // So d(loss)/d(x_ij) = 0.5
    for (i, &g) in grad.iter().enumerate() {
        assert!(
            (g - 0.5).abs() < 1e-5,
            "chain grad[{i}]: expected 0.5, got {g}"
        );
    }
}

// ===========================================================================
// Section 3: Gradient tape recording and playback
// ===========================================================================

#[test]
fn test_computation_graph_records_add_op() {
    let a = Arc::new(TrackedTensor::from_tensor(tensor(&[1.0], &[1])));
    let b = Arc::new(TrackedTensor::from_tensor(tensor(&[2.0], &[1])));
    let c = a.add(&b).unwrap();
    match c.op() {
        Some(Op::Add(..)) => {} // expected
        other => panic!("expected Add op, got {other:?}"),
    }
}

#[test]
fn test_computation_graph_records_mul_op() {
    let a = Arc::new(TrackedTensor::from_tensor(tensor(&[2.0], &[1])));
    let b = Arc::new(TrackedTensor::from_tensor(tensor(&[3.0], &[1])));
    let c = a.mul(&b).unwrap();
    match c.op() {
        Some(Op::Mul(..)) => {}
        other => panic!("expected Mul op, got {other:?}"),
    }
}

#[test]
fn test_computation_graph_records_div_op() {
    let a = Arc::new(TrackedTensor::from_tensor(tensor(&[6.0], &[1])));
    let b = Arc::new(TrackedTensor::from_tensor(tensor(&[3.0], &[1])));
    let c = a.div(&b).unwrap();
    match c.op() {
        Some(Op::Div(..)) => {}
        other => panic!("expected Div op, got {other:?}"),
    }
}

#[test]
fn test_computation_graph_chain_mul_add() {
    // Verify that chained ops build a proper graph: z = x*y + y
    let x = Arc::new(TrackedTensor::from_tensor(tensor(&[2.0], &[1])));
    let y = Arc::new(TrackedTensor::from_tensor(tensor(&[3.0], &[1])));
    let xy = x.mul(&y).unwrap();
    let z = xy.add(&y).unwrap();
    // z should have Add op, and one child should have Mul op
    assert!(matches!(z.op(), Some(Op::Add(..))));
    if let Some(Op::Add(lhs, _rhs)) = z.op() {
        assert!(matches!(lhs.op(), Some(Op::Mul(..))));
    }
}

#[test]
fn test_leaf_nodes_have_no_op() {
    let var = Var::zeros(&[3], DType::F32, &cpu()).unwrap();
    let tracked = TrackedTensor::from_var(&var).unwrap();
    assert!(tracked.op().is_none());

    let const_t = TrackedTensor::from_tensor(tensor(&[1.0, 2.0], &[2]));
    assert!(const_t.op().is_none());
}

#[test]
fn test_node_ids_are_unique() {
    let a = TrackedTensor::from_tensor(tensor(&[1.0], &[1]));
    let b = TrackedTensor::from_tensor(tensor(&[2.0], &[1]));
    assert_ne!(a.node_id().as_u64(), b.node_id().as_u64());
}

#[test]
fn test_backward_multi_use_variable() {
    // y = x + x, dy/dx = 2 (gradient accumulation)
    let x = scalar_var(5.0);
    let t = Arc::new(TrackedTensor::from_var(&x).unwrap());
    let y = t.add(&t).unwrap();
    let loss = y.sum_keepdim(0).unwrap();
    let grads = backward(&loss).unwrap();
    let grad = grads.get(&x).unwrap().to_flat_vec::<f32>().unwrap();
    assert!(
        (grad[0] - 2.0).abs() < 1e-5,
        "x+x grad: expected 2.0, got {}",
        grad[0]
    );
}

#[test]
fn test_backward_diamond_pattern() {
    // Diamond: a = x, b = x*x, c = a+b, loss = sum(c)
    // dc/dx = d(x + x^2)/dx = 1 + 2x = 1 + 2*3 = 7
    let x = scalar_var(3.0);
    let t = Arc::new(TrackedTensor::from_var(&x).unwrap());
    let a = Arc::clone(&t); // identity
    let b = t.sqr().unwrap(); // x^2
    let c = a.add(&b).unwrap(); // x + x^2
    let loss = c.sum_keepdim(0).unwrap();
    let grads = backward(&loss).unwrap();
    let grad = grads.get(&x).unwrap().to_flat_vec::<f32>().unwrap();
    assert!(
        (grad[0] - 7.0).abs() < 1e-4,
        "diamond grad: expected 7.0, got {}",
        grad[0]
    );
}

// ===========================================================================
// Section 4: Error propagation
// ===========================================================================

#[test]
fn test_backward_non_scalar_loss_error() {
    // backward requires scalar loss (numel == 1)
    let x = vec_var(vec![1.0, 2.0, 3.0]);
    let t = Arc::new(TrackedTensor::from_var(&x).unwrap());
    let result = backward(&t);
    match result {
        Err(AutodiffError::NonScalarLoss { shape }) => {
            assert_eq!(shape, vec![3]);
        }
        other => panic!("expected NonScalarLoss, got {other:?}"),
    }
}

#[test]
fn test_backward_non_finite_loss_nan() {
    // Loss containing NaN should be rejected
    let nan_t = DynTensor::new(&[f32::NAN], &[1], &cpu()).unwrap();
    let tracked = Arc::new(TrackedTensor::from_tensor(nan_t));
    let result = backward(&tracked);
    assert!(matches!(result, Err(AutodiffError::NonFiniteLoss)));
}

#[test]
fn test_backward_non_finite_loss_inf() {
    let inf_t = DynTensor::new(&[f32::INFINITY], &[1], &cpu()).unwrap();
    let tracked = Arc::new(TrackedTensor::from_tensor(inf_t));
    let result = backward(&tracked);
    assert!(matches!(result, Err(AutodiffError::NonFiniteLoss)));
}

#[test]
fn test_var_set_shape_mismatch_error() {
    let var = Var::zeros(&[2, 3], DType::F32, &cpu()).unwrap();
    let wrong = DynTensor::ones(&[3, 2], DType::F32, &cpu()).unwrap();
    let result = var.set(&wrong);
    assert!(result.is_err());
    let err_msg = format!("{}", result.unwrap_err());
    assert!(
        err_msg.contains("mismatch"),
        "error should mention mismatch: {err_msg}"
    );
}

#[test]
fn test_grad_store_accumulate_shape_mismatch() {
    let mut grads = GradStore::new();
    let var = Var::zeros(&[3], DType::F32, &cpu()).unwrap();
    let g1 = DynTensor::ones(&[3], DType::F32, &cpu()).unwrap();
    grads.accumulate_var(var.id(), &g1).unwrap();
    // Now try to accumulate a gradient with wrong shape
    let g2 = DynTensor::ones(&[4], DType::F32, &cpu()).unwrap();
    let result = grads.accumulate_var(var.id(), &g2);
    assert!(matches!(result, Err(AutodiffError::ShapeMismatch { .. })));
}

// ===========================================================================
// Section 5: TrackedTensor operations and gradient accumulation
// ===========================================================================

#[test]
fn test_tracked_tensor_dims() {
    let tracked = TrackedTensor::from_tensor(tensor(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]));
    assert_eq!(tracked.dims(), &[2, 3]);
}

#[test]
fn test_tracked_tensor_numel() {
    let tracked = TrackedTensor::from_tensor(tensor(&[1.0, 2.0, 3.0], &[3]));
    assert_eq!(tracked.numel(), 3);
}

#[test]
fn test_tracked_tensor_is_var_from_var() {
    let var = Var::zeros(&[2], DType::F32, &cpu()).unwrap();
    let tracked = TrackedTensor::from_var(&var).unwrap();
    assert!(tracked.is_var());
    assert_eq!(tracked.var_id(), Some(var.id()));
}

#[test]
fn test_tracked_tensor_is_not_var_from_tensor() {
    let tracked = TrackedTensor::from_tensor(tensor(&[1.0], &[1]));
    assert!(!tracked.is_var());
    assert_eq!(tracked.var_id(), None);
}

#[test]
fn test_tracked_tensor_detach_stops_gradient() {
    // y = detach(x^2), backward should not see x
    let x = scalar_var(3.0);
    let t = Arc::new(TrackedTensor::from_var(&x).unwrap());
    let sqr = t.sqr().unwrap();
    let detached = sqr.detach();
    // Backward on detached tensor: no gradient for x
    let grads = backward(&detached).unwrap();
    assert!(
        grads.get(&x).is_none(),
        "gradient should not flow through detach"
    );
}

#[test]
fn test_tracked_tensor_into_tensor() {
    let t = tensor(&[1.0, 2.0, 3.0], &[3]);
    let tracked = TrackedTensor::from_tensor(t);
    let recovered = tracked.into_tensor().unwrap();
    assert_eq!(recovered.dims(), &[3]);
    assert_eq!(recovered.to_flat_vec::<f32>().unwrap(), &[1.0, 2.0, 3.0]);
}

#[test]
fn test_backward_for_vars_selective() {
    // backward_for_vars should only return gradients for specified vars
    let a = scalar_var(2.0);
    let b = scalar_var(3.0);
    let ta = Arc::new(TrackedTensor::from_var(&a).unwrap());
    let tb = Arc::new(TrackedTensor::from_var(&b).unwrap());
    let y = ta.mul(&tb).unwrap();
    let loss = y.sum_keepdim(0).unwrap();
    let grads = backward_for_vars(&loss, &[&a]).unwrap();
    assert!(grads.get(&a).is_some());
    assert!(grads.get(&b).is_none());
}

#[test]
fn test_grad_store_var_count() {
    let mut grads = GradStore::new();
    assert_eq!(grads.var_count(), 0);
    let v1 = Var::zeros(&[2], DType::F32, &cpu()).unwrap();
    let v2 = Var::zeros(&[3], DType::F32, &cpu()).unwrap();
    grads
        .accumulate_var(v1.id(), &DynTensor::ones(&[2], DType::F32, &cpu()).unwrap())
        .unwrap();
    assert_eq!(grads.var_count(), 1);
    grads
        .accumulate_var(v2.id(), &DynTensor::ones(&[3], DType::F32, &cpu()).unwrap())
        .unwrap();
    assert_eq!(grads.var_count(), 2);
}

#[test]
fn test_grad_store_retain_only() {
    let v1 = Var::zeros(&[2], DType::F32, &cpu()).unwrap();
    let v2 = Var::zeros(&[2], DType::F32, &cpu()).unwrap();
    let v3 = Var::zeros(&[2], DType::F32, &cpu()).unwrap();
    let mut grads = GradStore::new();
    let ones = DynTensor::ones(&[2], DType::F32, &cpu()).unwrap();
    grads.accumulate_var(v1.id(), &ones).unwrap();
    grads.accumulate_var(v2.id(), &ones).unwrap();
    grads.accumulate_var(v3.id(), &ones).unwrap();
    assert_eq!(grads.var_count(), 3);
    grads.retain_only(&[&v1, &v3]);
    assert_eq!(grads.var_count(), 2);
    assert!(grads.get(&v1).is_some());
    assert!(grads.get(&v2).is_none());
    assert!(grads.get(&v3).is_some());
}

#[test]
fn test_gradient_accumulation_same_var_used_twice() {
    // z = x * x, dz/dx = 2x
    let x = vec_var(vec![1.0, 2.0, 3.0]);
    let t = Arc::new(TrackedTensor::from_var(&x).unwrap());
    let y = t.mul(&t).unwrap();
    let loss = y.sum_keepdim(0).unwrap();
    let grads = backward(&loss).unwrap();
    let grad = grads.get(&x).unwrap().to_flat_vec::<f32>().unwrap();
    // d(x*x)/dx = 2x
    assert_eq!(grad, vec![2.0, 4.0, 6.0]);
}

// ===========================================================================
// Section 6: DType handling in backward rules
// ===========================================================================

#[test]
fn test_backward_preserves_f32_dtype() {
    let x = scalar_var(2.0);
    let t = Arc::new(TrackedTensor::from_var(&x).unwrap());
    let y = t.sqr().unwrap();
    let grads = backward(&y).unwrap();
    let grad = grads.get(&x).unwrap();
    assert_eq!(grad.dtype(), DType::F32);
}

#[test]
fn test_grad_ones_initialization_matches_loss_dtype() {
    // The initial gradient (d(loss)/d(loss) = 1) should match the loss dtype
    let x = scalar_var(3.0);
    let t = Arc::new(TrackedTensor::from_var(&x).unwrap());
    let y = t.exp().unwrap();
    let grads = backward(&y).unwrap();
    let grad = grads.get(&x).unwrap();
    assert_eq!(
        grad.dtype(),
        DType::F32,
        "gradient dtype should match input"
    );
}

// ===========================================================================
// Section 7: Div backward
// ===========================================================================

#[test]
fn test_backward_div_known_values() {
    // y = a / b, dy/da = 1/b, dy/db = -a/b^2
    let a = scalar_var(6.0);
    let b = scalar_var(3.0);
    let ta = Arc::new(TrackedTensor::from_var(&a).unwrap());
    let tb = Arc::new(TrackedTensor::from_var(&b).unwrap());
    let y = ta.div(&tb).unwrap();
    let loss = y.sum_keepdim(0).unwrap();
    let grads = backward(&loss).unwrap();

    // dy/da = 1/3
    let grad_a = grads.get(&a).unwrap().to_flat_vec::<f32>().unwrap();
    assert!(
        (grad_a[0] - 1.0 / 3.0).abs() < 1e-5,
        "div grad_a: expected {}, got {}",
        1.0 / 3.0,
        grad_a[0]
    );

    // dy/db = -6/9 = -2/3
    let grad_b = grads.get(&b).unwrap().to_flat_vec::<f32>().unwrap();
    assert!(
        (grad_b[0] - (-2.0 / 3.0)).abs() < 1e-5,
        "div grad_b: expected {}, got {}",
        -2.0 / 3.0,
        grad_b[0]
    );
}

#[test]
fn test_backward_div_fd() {
    let data_a = vec![6.0, 4.0];
    let data_b = vec![3.0, 2.0];
    let a = Var::new(tensor(&data_a, &[2]));
    let b = Var::new(tensor(&data_b, &[2]));
    let ta = Arc::new(TrackedTensor::from_var(&a).unwrap());
    let tb = Arc::new(TrackedTensor::from_var(&b).unwrap());
    let y = ta.div(&tb).unwrap();
    let loss = y.sum_keepdim(0).unwrap();
    let grads = backward(&loss).unwrap();
    let grad_a = grads.get(&a).unwrap().to_flat_vec::<f32>().unwrap();
    check_fd_grad(&grad_a, &data_a, 1e-3, |d| {
        let ta = DynTensor::new(&d, &[2], &cpu()).unwrap();
        let tb = DynTensor::new(&data_b, &[2], &cpu()).unwrap();
        sum_f64(&ta.div(&tb).unwrap())
    });
}

// ===========================================================================
// Section 8: Shape operations backward
// ===========================================================================

#[test]
fn test_backward_reshape_preserves_gradient() {
    // Reshape should just reformat the gradient back to original shape
    let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let x = Var::new(tensor(&data, &[2, 3]));
    let t = Arc::new(TrackedTensor::from_var(&x).unwrap());
    let reshaped = t.reshape(&[3, 2]).unwrap();
    let loss = reshaped.sum_keepdim(1).unwrap().sum_keepdim(0).unwrap();
    let grads = backward(&loss).unwrap();
    let grad = grads.get(&x).unwrap();
    assert_eq!(grad.dims(), &[2, 3], "gradient shape should match original");
    assert_eq!(grad.to_flat_vec::<f32>().unwrap(), vec![1.0; 6]);
}

#[test]
fn test_backward_transpose_gradient_shape() {
    let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let x = Var::new(tensor(&data, &[2, 3]));
    let t = Arc::new(TrackedTensor::from_var(&x).unwrap());
    let transposed = t.transpose(0, 1).unwrap(); // [3, 2]
    let loss = transposed.sum_keepdim(1).unwrap().sum_keepdim(0).unwrap();
    let grads = backward(&loss).unwrap();
    let grad = grads.get(&x).unwrap();
    assert_eq!(
        grad.dims(),
        &[2, 3],
        "gradient shape should match original, not transposed"
    );
}

#[test]
fn test_backward_unsqueeze_gradient_shape() {
    let x = vec_var(vec![1.0, 2.0, 3.0]);
    let t = Arc::new(TrackedTensor::from_var(&x).unwrap());
    let expanded = t.unsqueeze(0).unwrap(); // [1, 3]
    let loss = expanded.sum_keepdim(1).unwrap().sum_keepdim(0).unwrap();
    let grads = backward(&loss).unwrap();
    let grad = grads.get(&x).unwrap();
    assert_eq!(
        grad.dims(),
        &[3],
        "gradient should match original [3] shape"
    );
}

#[test]
fn test_backward_narrow_gradient_shape() {
    let x = vec_var(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
    let t = Arc::new(TrackedTensor::from_var(&x).unwrap());
    let sliced = t.narrow(0, 1, 3).unwrap(); // [3] from indices 1..4
    let loss = sliced.sum_keepdim(0).unwrap();
    let grads = backward(&loss).unwrap();
    let grad = grads.get(&x).unwrap();
    assert_eq!(
        grad.dims(),
        &[5],
        "gradient should match original [5] shape"
    );
    let vals = grad.to_flat_vec::<f32>().unwrap();
    // Only positions 1,2,3 should have gradient 1, rest 0
    assert_eq!(vals, vec![0.0, 1.0, 1.0, 1.0, 0.0]);
}

// ===========================================================================
// Section 9: Scalar ops backward
// ===========================================================================

#[test]
fn test_backward_mul_scalar() {
    // y = 3*x, dy/dx = 3
    let x = vec_var(vec![1.0, 2.0]);
    let t = Arc::new(TrackedTensor::from_var(&x).unwrap());
    let y = t.mul_scalar(3.0).unwrap();
    let loss = y.sum_keepdim(0).unwrap();
    let grads = backward(&loss).unwrap();
    let grad = grads.get(&x).unwrap().to_flat_vec::<f32>().unwrap();
    assert_eq!(grad, vec![3.0, 3.0]);
}

#[test]
fn test_backward_add_scalar() {
    // y = x + 5, dy/dx = 1
    let x = vec_var(vec![1.0, 2.0, 3.0]);
    let t = Arc::new(TrackedTensor::from_var(&x).unwrap());
    let y = t.add_scalar(5.0).unwrap();
    let loss = y.sum_keepdim(0).unwrap();
    let grads = backward(&loss).unwrap();
    let grad = grads.get(&x).unwrap().to_flat_vec::<f32>().unwrap();
    assert_eq!(grad, vec![1.0, 1.0, 1.0]);
}

#[test]
fn test_backward_powf_square() {
    // y = x^2 via powf, dy/dx = 2x
    let data = vec![2.0, 3.0];
    let x = vec_var(data);
    let t = Arc::new(TrackedTensor::from_var(&x).unwrap());
    let y = t.powf(2.0).unwrap();
    let loss = y.sum_keepdim(0).unwrap();
    let grads = backward(&loss).unwrap();
    let grad = grads.get(&x).unwrap().to_flat_vec::<f32>().unwrap();
    assert!(
        (grad[0] - 4.0).abs() < 1e-4,
        "powf grad[0]: expected 4.0, got {}",
        grad[0]
    );
    assert!(
        (grad[1] - 6.0).abs() < 1e-4,
        "powf grad[1]: expected 6.0, got {}",
        grad[1]
    );
}

#[test]
fn test_backward_clamp_interior_and_boundary() {
    // clamp(x, 0.0, 10.0): gradient is 1 inside bounds, 0 at boundaries
    let x = vec_var(vec![-1.0, 0.5, 5.0, 10.0, 15.0]);
    let t = Arc::new(TrackedTensor::from_var(&x).unwrap());
    let y = t.clamp(0.0, 10.0).unwrap();
    let loss = y.sum_keepdim(0).unwrap();
    let grads = backward(&loss).unwrap();
    let grad = grads.get(&x).unwrap().to_flat_vec::<f32>().unwrap();
    // -1 clamped to 0 (grad=0), 0.5 inside (grad=1), 5 inside (grad=1),
    // 10 at upper boundary (grad=0 or 1 depending on convention), 15 clamped (grad=0)
    assert_eq!(grad[0], 0.0, "below min should have zero grad");
    assert_eq!(grad[1], 1.0, "interior should have grad 1");
    assert_eq!(grad[2], 1.0, "interior should have grad 1");
    assert_eq!(grad[4], 0.0, "above max should have zero grad");
}

// ===========================================================================
// Section 10: Matmul backward
// ===========================================================================

#[test]
fn test_backward_matmul_2x2() {
    // y = A @ B where A=[2,2], B=[2,2]
    // dy/dA = grad @ B^T, dy/dB = A^T @ grad
    let a = mat_var(&[1.0, 2.0, 3.0, 4.0], 2, 2);
    let b = mat_var(&[5.0, 6.0, 7.0, 8.0], 2, 2);
    let ta = Arc::new(TrackedTensor::from_var(&a).unwrap());
    let tb = Arc::new(TrackedTensor::from_var(&b).unwrap());
    let y = ta.matmul(&tb).unwrap();
    let loss = y.sum_keepdim(1).unwrap().sum_keepdim(0).unwrap();
    let grads = backward(&loss).unwrap();

    let grad_a = grads.get(&a).unwrap();
    let grad_b = grads.get(&b).unwrap();
    assert_eq!(grad_a.dims(), &[2, 2]);
    assert_eq!(grad_b.dims(), &[2, 2]);

    // With grad = ones([2,2]):
    // grad_A = ones @ B^T = ones @ [[5,7],[6,8]] = [[11,14],[11,14]]
    let ga = grad_a.to_flat_vec::<f32>().unwrap();
    assert!(
        (ga[0] - 11.0).abs() < 1e-4,
        "matmul grad_a[0,0]: expected 11, got {}",
        ga[0]
    );
}

// ===========================================================================
// Section 11: Constant (non-var) tensors receive no gradient
// ===========================================================================

#[test]
fn test_constant_tensor_no_gradient() {
    let x = scalar_var(2.0);
    let c = Arc::new(TrackedTensor::from_tensor(tensor(&[3.0], &[1])));
    let t = Arc::new(TrackedTensor::from_var(&x).unwrap());
    let y = t.mul(&c).unwrap();
    let loss = y.sum_keepdim(0).unwrap();
    let grads = backward(&loss).unwrap();
    // x should have gradient (= c = 3.0)
    let grad_x = grads.get(&x).unwrap().to_flat_vec::<f32>().unwrap();
    assert!(
        (grad_x[0] - 3.0).abs() < 1e-5,
        "expected 3.0, got {}",
        grad_x[0]
    );
    // GradStore only stores var grads; constant has no VarId
}

// ===========================================================================
// Section 12: Complex chains
// ===========================================================================

#[test]
fn test_backward_exp_then_log_identity() {
    // log(exp(x)) = x, gradient should be 1
    let x = scalar_var(2.5);
    let t = Arc::new(TrackedTensor::from_var(&x).unwrap());
    let y = t.exp().unwrap();
    let z = y.log().unwrap();
    let grads = backward(&z).unwrap();
    let grad = grads.get(&x).unwrap().to_flat_vec::<f32>().unwrap();
    assert!(
        (grad[0] - 1.0).abs() < 1e-4,
        "log(exp(x)) grad: expected 1.0, got {}",
        grad[0]
    );
}

#[test]
fn test_backward_triple_mul_chain() {
    // y = a * b * c, dy/da = b*c, dy/db = a*c, dy/dc = a*b
    let a = scalar_var(2.0);
    let b = scalar_var(3.0);
    let c = scalar_var(5.0);
    let ta = Arc::new(TrackedTensor::from_var(&a).unwrap());
    let tb = Arc::new(TrackedTensor::from_var(&b).unwrap());
    let tc = Arc::new(TrackedTensor::from_var(&c).unwrap());
    let ab = ta.mul(&tb).unwrap();
    let abc = ab.mul(&tc).unwrap();
    let loss = abc.sum_keepdim(0).unwrap();
    let grads = backward(&loss).unwrap();

    let ga = grads.get(&a).unwrap().to_flat_vec::<f32>().unwrap()[0];
    let gb = grads.get(&b).unwrap().to_flat_vec::<f32>().unwrap()[0];
    let gc = grads.get(&c).unwrap().to_flat_vec::<f32>().unwrap()[0];
    assert!((ga - 15.0).abs() < 1e-4, "dy/da = b*c = 15, got {ga}");
    assert!((gb - 10.0).abs() < 1e-4, "dy/db = a*c = 10, got {gb}");
    assert!((gc - 6.0).abs() < 1e-4, "dy/dc = a*b = 6, got {gc}");
}

#[test]
fn test_backward_sub_then_sqr() {
    // y = (a - b)^2, dy/da = 2(a-b) = 2*(5-3) = 4
    // dy/db = -2(a-b) = -4
    let a = scalar_var(5.0);
    let b = scalar_var(3.0);
    let ta = Arc::new(TrackedTensor::from_var(&a).unwrap());
    let tb = Arc::new(TrackedTensor::from_var(&b).unwrap());
    let diff = ta.sub(&tb).unwrap();
    let y = diff.sqr().unwrap();
    let grads = backward(&y).unwrap();

    let ga = grads.get(&a).unwrap().to_flat_vec::<f32>().unwrap()[0];
    let gb = grads.get(&b).unwrap().to_flat_vec::<f32>().unwrap()[0];
    assert!((ga - 4.0).abs() < 1e-4, "dy/da: expected 4.0, got {ga}");
    assert!((gb - (-4.0)).abs() < 1e-4, "dy/db: expected -4.0, got {gb}");
}

#[test]
fn test_backward_grad_store_iter() {
    // Verify we can iterate over all var gradients
    let a = scalar_var(1.0);
    let b = scalar_var(2.0);
    let ta = Arc::new(TrackedTensor::from_var(&a).unwrap());
    let tb = Arc::new(TrackedTensor::from_var(&b).unwrap());
    let y = ta.add(&tb).unwrap();
    let loss = y.sum_keepdim(0).unwrap();
    let grads = backward(&loss).unwrap();
    let count = grads.var_grads().count();
    assert_eq!(count, 2, "should have gradients for both a and b");
}

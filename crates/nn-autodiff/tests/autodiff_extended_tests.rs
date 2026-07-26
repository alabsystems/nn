// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Extended tests for nn-autodiff: backward rules, gradient tape, error handling.
//! Targets gaps not covered by existing in-crate unit tests.

use std::sync::Arc;

use nn_autodiff::{backward, backward_for_vars, GradStore, TrackedTensor, Var, VarMap};
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

// ---------------------------------------------------------------------------
// Error handling tests
// ---------------------------------------------------------------------------

#[test]
fn test_backward_rejects_non_scalar_loss() {
    // backward() requires numel == 1
    let t = Arc::new(TrackedTensor::from_tensor(
        DynTensor::from_vec(vec![1.0, 2.0, 3.0], &[3], &cpu()).unwrap(),
    ));
    let result = backward(&t);
    assert!(result.is_err());
    let err_msg = format!("{}", result.unwrap_err());
    assert!(
        err_msg.contains("scalar"),
        "expected scalar error, got: {err_msg}"
    );
}

#[test]
fn test_backward_rejects_non_finite_loss_nan() {
    // Create a NaN loss via log(0) = -inf, then exp(-inf) - 1 produces NaN-like issues.
    // Simpler: directly create a NaN tensor as a tracked constant.
    let nan_tensor = DynTensor::from_vec(vec![f32::NAN], &[1], &cpu()).unwrap();
    let t = Arc::new(TrackedTensor::from_tensor(nan_tensor));
    let result = backward(&t);
    assert!(result.is_err());
    let err_msg = format!("{}", result.unwrap_err());
    assert!(
        err_msg.contains("finite"),
        "expected non-finite error, got: {err_msg}"
    );
}

#[test]
fn test_backward_rejects_non_finite_loss_inf() {
    let inf_tensor = DynTensor::from_vec(vec![f32::INFINITY], &[1], &cpu()).unwrap();
    let t = Arc::new(TrackedTensor::from_tensor(inf_tensor));
    let result = backward(&t);
    assert!(result.is_err());
    let err_msg = format!("{}", result.unwrap_err());
    assert!(
        err_msg.contains("finite"),
        "expected non-finite error, got: {err_msg}"
    );
}

#[test]
fn test_backward_rejects_non_finite_loss_neg_inf() {
    let neg_inf_tensor = DynTensor::from_vec(vec![f32::NEG_INFINITY], &[1], &cpu()).unwrap();
    let t = Arc::new(TrackedTensor::from_tensor(neg_inf_tensor));
    let result = backward(&t);
    assert!(result.is_err());
}

// ---------------------------------------------------------------------------
// GradStore tests
// ---------------------------------------------------------------------------

#[test]
fn test_grad_store_new_is_empty() {
    let store = GradStore::new();
    assert_eq!(store.var_count(), 0);
}

#[test]
fn test_grad_store_default_is_empty() {
    let store = GradStore::default();
    assert_eq!(store.var_count(), 0);
}

#[test]
fn test_grad_store_get_missing_var_returns_none() {
    let store = GradStore::new();
    let v = Var::zeros(&[2], DType::F32, &cpu()).unwrap();
    assert!(store.get(&v).is_none());
}

#[test]
fn test_grad_store_get_id_missing_returns_none() {
    let store = GradStore::new();
    let v = Var::zeros(&[2], DType::F32, &cpu()).unwrap();
    assert!(store.get_id(&v.id()).is_none());
}

#[test]
fn test_backward_for_vars_filters_gradients() {
    // Build: loss = sum(w1 * x) + sum(w2 * x)
    // Then backward_for_vars with only w1 should not include w2.
    let w1 = scalar_var(2.0);
    let w2 = scalar_var(3.0);
    let x = Arc::new(TrackedTensor::from_tensor(
        DynTensor::from_vec(vec![1.0], &[1], &cpu()).unwrap(),
    ));

    let tw1 = Arc::new(TrackedTensor::from_var(&w1).unwrap());
    let tw2 = Arc::new(TrackedTensor::from_var(&w2).unwrap());
    let prod1 = tw1.mul(&x).unwrap();
    let prod2 = tw2.mul(&x).unwrap();
    let loss = prod1.add(&prod2).unwrap();

    let grads = backward_for_vars(&loss, &[&w1]).unwrap();
    assert!(grads.get(&w1).is_some());
    assert!(grads.get(&w2).is_none());
    assert_eq!(grads.var_count(), 1);
}

#[test]
fn test_backward_for_vars_empty_targets() {
    let x = scalar_var(5.0);
    let t = Arc::new(TrackedTensor::from_var(&x).unwrap());
    let grads = backward_for_vars(&t, &[]).unwrap();
    assert_eq!(grads.var_count(), 0);
}

// ---------------------------------------------------------------------------
// TrackedTensor structural tests
// ---------------------------------------------------------------------------

#[test]
fn test_tracked_tensor_detach_breaks_graph() {
    let x = scalar_var(3.0);
    let t = Arc::new(TrackedTensor::from_var(&x).unwrap());
    let y = t.sqr().unwrap(); // y = x^2
    let detached = y.detach(); // breaks gradient flow
                               // detached has no op, no var_id
    assert!(detached.op().is_none());
    assert!(!detached.is_var());
    // backward from detached gives no gradient for x
    let grads = backward(&detached).unwrap();
    assert!(grads.get(&x).is_none());
}

#[test]
fn test_tracked_tensor_into_tensor_extracts_data() {
    let data = DynTensor::from_vec(vec![1.0, 2.0, 3.0], &[3], &cpu()).unwrap();
    let tracked = TrackedTensor::from_tensor(data);
    let extracted = tracked.into_tensor().unwrap();
    assert_eq!(extracted.dims(), &[3]);
    assert_eq!(extracted.to_flat_vec::<f32>().unwrap(), vec![1.0, 2.0, 3.0]);
}

#[test]
fn test_tracked_tensor_node_ids_are_unique() {
    let a = TrackedTensor::from_tensor(DynTensor::from_vec(vec![1.0], &[1], &cpu()).unwrap());
    let b = TrackedTensor::from_tensor(DynTensor::from_vec(vec![2.0], &[1], &cpu()).unwrap());
    assert_ne!(a.node_id(), b.node_id());
}

#[test]
fn test_tracked_tensor_numel_and_dims() {
    let t = TrackedTensor::from_tensor(
        DynTensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], &cpu()).unwrap(),
    );
    assert_eq!(t.numel(), 6);
    assert_eq!(t.dims(), &[2, 3]);
}

#[test]
fn test_tracked_tensor_debug_format() {
    let t = TrackedTensor::from_tensor(DynTensor::from_vec(vec![1.0], &[1], &cpu()).unwrap());
    let debug = format!("{t:?}");
    assert!(debug.contains("TrackedTensor"));
    assert!(debug.contains("dims"));
}

// ---------------------------------------------------------------------------
// Backward rule correctness tests (specific ops)
// ---------------------------------------------------------------------------

#[test]
fn test_backward_div_gradient() {
    // y = a / b, dy/da = 1/b, dy/db = -a/b^2
    let a = scalar_var(6.0);
    let b = scalar_var(3.0);
    let ta = Arc::new(TrackedTensor::from_var(&a).unwrap());
    let tb = Arc::new(TrackedTensor::from_var(&b).unwrap());
    let y = ta.div(&tb).unwrap();
    let grads = backward(&y).unwrap();

    let grad_a = grads.get(&a).unwrap().to_flat_vec::<f32>().unwrap();
    let grad_b = grads.get(&b).unwrap().to_flat_vec::<f32>().unwrap();
    // dy/da = 1/3 = 0.333...
    assert!(
        (grad_a[0] - 1.0 / 3.0).abs() < 1e-5,
        "expected ~0.333, got {}",
        grad_a[0]
    );
    // dy/db = -6/9 = -0.666...
    assert!(
        (grad_b[0] - (-6.0 / 9.0)).abs() < 1e-5,
        "expected ~-0.667, got {}",
        grad_b[0]
    );
}

#[test]
fn test_backward_neg_gradient() {
    // y = -x, dy/dx = -1
    let x = scalar_var(7.0);
    let t = Arc::new(TrackedTensor::from_var(&x).unwrap());
    let y = t.neg().unwrap();
    let grads = backward(&y).unwrap();
    let grad = grads.get(&x).unwrap().to_flat_vec::<f32>().unwrap();
    assert!((grad[0] - (-1.0)).abs() < 1e-6);
}

#[test]
fn test_backward_exp_gradient() {
    // y = exp(x), dy/dx = exp(x)
    let x = scalar_var(1.0);
    let t = Arc::new(TrackedTensor::from_var(&x).unwrap());
    let y = t.exp().unwrap();
    let grads = backward(&y).unwrap();
    let grad = grads.get(&x).unwrap().to_flat_vec::<f32>().unwrap();
    let expected = 1.0_f32.exp();
    assert!(
        (grad[0] - expected).abs() < 1e-5,
        "expected {expected}, got {}",
        grad[0]
    );
}

#[test]
fn test_backward_log_gradient() {
    // y = ln(x), dy/dx = 1/x
    let x = scalar_var(2.0);
    let t = Arc::new(TrackedTensor::from_var(&x).unwrap());
    let y = t.log().unwrap();
    let grads = backward(&y).unwrap();
    let grad = grads.get(&x).unwrap().to_flat_vec::<f32>().unwrap();
    assert!(
        (grad[0] - 0.5).abs() < 1e-5,
        "expected 0.5, got {}",
        grad[0]
    );
}

#[test]
fn test_backward_tanh_gradient() {
    // y = tanh(x), dy/dx = 1 - tanh(x)^2
    let x = scalar_var(0.5);
    let t = Arc::new(TrackedTensor::from_var(&x).unwrap());
    let y = t.tanh().unwrap();
    let grads = backward(&y).unwrap();
    let grad = grads.get(&x).unwrap().to_flat_vec::<f32>().unwrap();
    let tanh_val = 0.5_f32.tanh();
    let expected = 1.0 - tanh_val * tanh_val;
    assert!(
        (grad[0] - expected).abs() < 1e-5,
        "expected {expected}, got {}",
        grad[0]
    );
}

#[test]
fn test_backward_sigmoid_gradient() {
    // y = sigmoid(x), dy/dx = sigmoid(x) * (1 - sigmoid(x))
    let x = scalar_var(1.0);
    let t = Arc::new(TrackedTensor::from_var(&x).unwrap());
    let y = t.sigmoid().unwrap();
    let grads = backward(&y).unwrap();
    let grad = grads.get(&x).unwrap().to_flat_vec::<f32>().unwrap();
    let sig = 1.0 / (1.0 + (-1.0_f32).exp());
    let expected = sig * (1.0 - sig);
    assert!(
        (grad[0] - expected).abs() < 1e-5,
        "expected {expected}, got {}",
        grad[0]
    );
}

#[test]
fn test_backward_relu_gradient_positive() {
    // y = relu(x) where x > 0, dy/dx = 1
    let x = scalar_var(3.0);
    let t = Arc::new(TrackedTensor::from_var(&x).unwrap());
    let y = t.relu().unwrap();
    let grads = backward(&y).unwrap();
    let grad = grads.get(&x).unwrap().to_flat_vec::<f32>().unwrap();
    assert!((grad[0] - 1.0).abs() < 1e-6);
}

#[test]
fn test_backward_relu_gradient_negative() {
    // y = relu(x) where x < 0, dy/dx = 0
    let x = scalar_var(-3.0);
    let t = Arc::new(TrackedTensor::from_var(&x).unwrap());
    let y = t.relu().unwrap();
    let grads = backward(&y).unwrap();
    let grad = grads.get(&x).unwrap().to_flat_vec::<f32>().unwrap();
    assert!((grad[0]).abs() < 1e-6);
}

#[test]
fn test_backward_abs_gradient() {
    // y = |x|, dy/dx = sign(x) with sign(0) = 0
    let x = vec_var(vec![-2.0, 0.0, 3.0]);
    let t = Arc::new(TrackedTensor::from_var(&x).unwrap());
    let y = t.abs().unwrap();
    let loss = y.sum_keepdim(0).unwrap();
    let grads = backward(&loss).unwrap();
    let grad = grads.get(&x).unwrap().to_flat_vec::<f32>().unwrap();
    assert!((grad[0] - (-1.0)).abs() < 1e-6, "grad[-2.0] should be -1");
    assert!((grad[1]).abs() < 1e-6, "grad[0.0] should be 0");
    assert!((grad[2] - 1.0).abs() < 1e-6, "grad[3.0] should be 1");
}

#[test]
fn test_backward_mul_scalar() {
    // y = x * 3.0, dy/dx = 3.0
    let x = scalar_var(5.0);
    let t = Arc::new(TrackedTensor::from_var(&x).unwrap());
    let y = t.mul_scalar(3.0).unwrap();
    let grads = backward(&y).unwrap();
    let grad = grads.get(&x).unwrap().to_flat_vec::<f32>().unwrap();
    assert!((grad[0] - 3.0).abs() < 1e-6);
}

#[test]
fn test_backward_add_scalar() {
    // y = x + 7.0, dy/dx = 1
    let x = scalar_var(2.0);
    let t = Arc::new(TrackedTensor::from_var(&x).unwrap());
    let y = t.add_scalar(7.0).unwrap();
    let grads = backward(&y).unwrap();
    let grad = grads.get(&x).unwrap().to_flat_vec::<f32>().unwrap();
    assert!((grad[0] - 1.0).abs() < 1e-6);
}

#[test]
fn test_backward_sin_gradient() {
    // y = sin(x), dy/dx = cos(x)
    let x = scalar_var(1.0);
    let t = Arc::new(TrackedTensor::from_var(&x).unwrap());
    let y = t.sin().unwrap();
    let grads = backward(&y).unwrap();
    let grad = grads.get(&x).unwrap().to_flat_vec::<f32>().unwrap();
    let expected = 1.0_f32.cos();
    assert!(
        (grad[0] - expected).abs() < 1e-5,
        "expected {expected}, got {}",
        grad[0]
    );
}

#[test]
fn test_backward_cos_gradient() {
    // y = cos(x), dy/dx = -sin(x)
    let x = scalar_var(1.0);
    let t = Arc::new(TrackedTensor::from_var(&x).unwrap());
    let y = t.cos().unwrap();
    let grads = backward(&y).unwrap();
    let grad = grads.get(&x).unwrap().to_flat_vec::<f32>().unwrap();
    let expected = -(1.0_f32.sin());
    assert!(
        (grad[0] - expected).abs() < 1e-5,
        "expected {expected}, got {}",
        grad[0]
    );
}

#[test]
fn test_backward_recip_gradient() {
    // y = 1/x, dy/dx = -1/x^2
    let x = scalar_var(4.0);
    let t = Arc::new(TrackedTensor::from_var(&x).unwrap());
    let y = t.recip().unwrap();
    let grads = backward(&y).unwrap();
    let grad = grads.get(&x).unwrap().to_flat_vec::<f32>().unwrap();
    let expected = -1.0 / 16.0;
    assert!(
        (grad[0] - expected).abs() < 1e-5,
        "expected {expected}, got {}",
        grad[0]
    );
}

#[test]
fn test_backward_powf_gradient() {
    // y = x^3, dy/dx = 3*x^2 = 3*4 = 12
    let x = scalar_var(2.0);
    let t = Arc::new(TrackedTensor::from_var(&x).unwrap());
    let y = t.powf(3.0).unwrap();
    let grads = backward(&y).unwrap();
    let grad = grads.get(&x).unwrap().to_flat_vec::<f32>().unwrap();
    assert!(
        (grad[0] - 12.0).abs() < 1e-4,
        "expected 12.0, got {}",
        grad[0]
    );
}

#[test]
fn test_backward_powf_zero_exponent() {
    // y = x^0 = 1 (constant), dy/dx = 0
    let x = scalar_var(5.0);
    let t = Arc::new(TrackedTensor::from_var(&x).unwrap());
    let y = t.powf(0.0).unwrap();
    let grads = backward(&y).unwrap();
    let grad = grads.get(&x).unwrap().to_flat_vec::<f32>().unwrap();
    assert!(grad[0].abs() < 1e-6, "expected 0.0, got {}", grad[0]);
}

#[test]
fn test_backward_clamp_gradient_inside() {
    // y = clamp(x, -1, 1) where -1 <= x <= 1, dy/dx = 1
    let x = scalar_var(0.5);
    let t = Arc::new(TrackedTensor::from_var(&x).unwrap());
    let y = t.clamp(-1.0, 1.0).unwrap();
    let grads = backward(&y).unwrap();
    let grad = grads.get(&x).unwrap().to_flat_vec::<f32>().unwrap();
    assert!((grad[0] - 1.0).abs() < 1e-6);
}

#[test]
fn test_backward_clamp_gradient_outside() {
    // y = clamp(x, -1, 1) where x > 1, dy/dx = 0
    let x = scalar_var(5.0);
    let t = Arc::new(TrackedTensor::from_var(&x).unwrap());
    let y = t.clamp(-1.0, 1.0).unwrap();
    let grads = backward(&y).unwrap();
    let grad = grads.get(&x).unwrap().to_flat_vec::<f32>().unwrap();
    assert!(grad[0].abs() < 1e-6);
}

#[test]
fn test_backward_elu_gradient_positive() {
    // y = elu(x, 1.0) where x > 0, dy/dx = 1
    let x = scalar_var(2.0);
    let t = Arc::new(TrackedTensor::from_var(&x).unwrap());
    let y = t.elu(1.0).unwrap();
    let grads = backward(&y).unwrap();
    let grad = grads.get(&x).unwrap().to_flat_vec::<f32>().unwrap();
    assert!(
        (grad[0] - 1.0).abs() < 1e-5,
        "expected 1.0, got {}",
        grad[0]
    );
}

#[test]
fn test_backward_elu_gradient_negative() {
    // y = elu(x, alpha) where x <= 0, dy/dx = alpha * exp(x)
    let alpha = 1.5;
    let x_val = -1.0_f32;
    let x = scalar_var(x_val);
    let t = Arc::new(TrackedTensor::from_var(&x).unwrap());
    let y = t.elu(alpha).unwrap();
    let grads = backward(&y).unwrap();
    let grad = grads.get(&x).unwrap().to_flat_vec::<f32>().unwrap();
    let expected = alpha as f32 * x_val.exp();
    assert!(
        (grad[0] - expected).abs() < 1e-5,
        "expected {expected}, got {}",
        grad[0]
    );
}

// ---------------------------------------------------------------------------
// Shape backward tests
// ---------------------------------------------------------------------------

#[test]
fn test_backward_reshape() {
    // Reshape should restore original shape in backward.
    let x = vec_var(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let t = Arc::new(TrackedTensor::from_var(&x).unwrap());
    let reshaped = t.reshape(&[2, 3]).unwrap();
    let loss = reshaped.sum_keepdim(1).unwrap().sum_keepdim(0).unwrap();
    let grads = backward(&loss).unwrap();
    let grad = grads.get(&x).unwrap();
    // Gradient of sum over all elements is all 1s, shape should match original [6]
    assert_eq!(grad.dims(), &[6]);
    assert_eq!(grad.to_flat_vec::<f32>().unwrap(), vec![1.0; 6]);
}

#[test]
fn test_backward_transpose() {
    // Transpose backward should undo the transpose.
    let x = mat_var(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 2, 3);
    let t = Arc::new(TrackedTensor::from_var(&x).unwrap());
    let transposed = t.transpose(0, 1).unwrap(); // [3, 2]
    let loss = transposed.sum_keepdim(1).unwrap().sum_keepdim(0).unwrap();
    let grads = backward(&loss).unwrap();
    let grad = grads.get(&x).unwrap();
    assert_eq!(grad.dims(), &[2, 3]);
}

#[test]
fn test_backward_unsqueeze_squeeze() {
    // unsqueeze then squeeze should be identity in backward
    let x = vec_var(vec![1.0, 2.0, 3.0]);
    let t = Arc::new(TrackedTensor::from_var(&x).unwrap());
    let unsqueezed = t.unsqueeze(0).unwrap(); // [1, 3]
    let loss = unsqueezed.sum_keepdim(1).unwrap().sum_keepdim(0).unwrap();
    let grads = backward(&loss).unwrap();
    let grad = grads.get(&x).unwrap();
    assert_eq!(grad.dims(), &[3]);
    assert_eq!(grad.to_flat_vec::<f32>().unwrap(), vec![1.0; 3]);
}

#[test]
fn test_backward_narrow() {
    // Narrow slices a dimension; backward zero-pads the rest.
    let x = vec_var(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
    let t = Arc::new(TrackedTensor::from_var(&x).unwrap());
    let sliced = t.narrow(0, 1, 3).unwrap(); // elements [2, 3, 4]
    let loss = sliced.sum_keepdim(0).unwrap();
    let grads = backward(&loss).unwrap();
    let grad = grads.get(&x).unwrap().to_flat_vec::<f32>().unwrap();
    // Gradient should be [0, 1, 1, 1, 0]
    assert_eq!(grad, vec![0.0, 1.0, 1.0, 1.0, 0.0]);
}

#[test]
fn test_backward_permute() {
    // Permute [0,1,2] -> [2,0,1], backward should apply inverse [1,2,0]
    let data: Vec<f32> = (1..=24).map(|v| v as f32).collect();
    let x = Var::new(DynTensor::from_vec(data, &[2, 3, 4], &cpu()).unwrap());
    let t = Arc::new(TrackedTensor::from_var(&x).unwrap());
    let permuted = t.permute(&[2, 0, 1]).unwrap(); // [4, 2, 3]
    assert_eq!(permuted.dims(), &[4, 2, 3]);
    let loss = permuted
        .sum_keepdim(2)
        .unwrap()
        .sum_keepdim(1)
        .unwrap()
        .sum_keepdim(0)
        .unwrap();
    let grads = backward(&loss).unwrap();
    let grad = grads.get(&x).unwrap();
    assert_eq!(grad.dims(), &[2, 3, 4]);
}

// ---------------------------------------------------------------------------
// Composite / chain rule tests
// ---------------------------------------------------------------------------

#[test]
fn test_backward_chain_rule_sqr_then_exp() {
    // y = exp(x^2), dy/dx = 2x * exp(x^2)
    let x = scalar_var(1.0);
    let t = Arc::new(TrackedTensor::from_var(&x).unwrap());
    let sqr = t.sqr().unwrap();
    let y = sqr.exp().unwrap();
    let grads = backward(&y).unwrap();
    let grad = grads.get(&x).unwrap().to_flat_vec::<f32>().unwrap();
    let expected = 2.0 * 1.0_f32.exp(); // 2 * e ≈ 5.4366
    assert!(
        (grad[0] - expected).abs() < 1e-4,
        "expected {expected}, got {}",
        grad[0]
    );
}

#[test]
fn test_backward_sum_keepdim_gradient() {
    // y = sum(x), dy/dx_i = 1 for all i
    let x = vec_var(vec![1.0, 2.0, 3.0, 4.0]);
    let t = Arc::new(TrackedTensor::from_var(&x).unwrap());
    let y = t.sum_keepdim(0).unwrap();
    let grads = backward(&y).unwrap();
    let grad = grads.get(&x).unwrap().to_flat_vec::<f32>().unwrap();
    assert_eq!(grad, vec![1.0, 1.0, 1.0, 1.0]);
}

#[test]
fn test_backward_mean_keepdim_gradient() {
    // y = mean(x), dy/dx_i = 1/n for all i
    let x = vec_var(vec![1.0, 2.0, 3.0, 4.0]);
    let t = Arc::new(TrackedTensor::from_var(&x).unwrap());
    let y = t.mean_keepdim(0).unwrap();
    let grads = backward(&y).unwrap();
    let grad = grads.get(&x).unwrap().to_flat_vec::<f32>().unwrap();
    for &g in &grad {
        assert!((g - 0.25).abs() < 1e-6, "expected 0.25, got {g}");
    }
}

// ---------------------------------------------------------------------------
// MatMul backward
// ---------------------------------------------------------------------------

#[test]
fn test_backward_matmul_gradient() {
    // y = a @ b, loss = sum(y)
    // dy/da = ones @ b^T, dy/db = a^T @ ones
    let a = mat_var(vec![1.0, 2.0, 3.0, 4.0], 2, 2);
    let b = mat_var(vec![5.0, 6.0, 7.0, 8.0], 2, 2);
    let ta = Arc::new(TrackedTensor::from_var(&a).unwrap());
    let tb = Arc::new(TrackedTensor::from_var(&b).unwrap());
    let y = ta.matmul(&tb).unwrap();
    let loss = y.sum_keepdim(1).unwrap().sum_keepdim(0).unwrap();
    let grads = backward(&loss).unwrap();

    let grad_a = grads.get(&a).unwrap();
    let grad_b = grads.get(&b).unwrap();
    assert_eq!(grad_a.dims(), &[2, 2]);
    assert_eq!(grad_b.dims(), &[2, 2]);

    // grad_a = ones @ b^T
    // ones = [[1,1],[1,1]], b^T = [[5,7],[6,8]]
    // grad_a = [[1*5+1*6, 1*7+1*8],[1*5+1*6, 1*7+1*8]] = [[11,15],[11,15]]
    let ga = grad_a.to_flat_vec::<f32>().unwrap();
    assert!((ga[0] - 11.0).abs() < 1e-4);
    assert!((ga[1] - 15.0).abs() < 1e-4);
}

// ---------------------------------------------------------------------------
// Var / VarMap edge cases
// ---------------------------------------------------------------------------

#[test]
fn test_var_clone_shares_id() {
    let v = Var::zeros(&[3], DType::F32, &cpu()).unwrap();
    let v2 = v.clone();
    assert_eq!(v.id(), v2.id());
}

#[test]
fn test_varmap_default_is_empty() {
    let map = VarMap::new();
    assert!(map.is_empty());
    assert_eq!(map.len(), 0);
}

// ---------------------------------------------------------------------------
// Op Debug tests for variants not covered by in-crate tests
// ---------------------------------------------------------------------------

#[test]
fn test_op_debug_additional_variants() {
    use nn_autodiff::Op;
    let t = Arc::new(TrackedTensor::from_tensor(
        DynTensor::from_vec(vec![1.0], &[1], &cpu()).unwrap(),
    ));

    // Test Debug output for various Op variants
    let debug_str = format!("{:?}", Op::Sigmoid(Arc::clone(&t)));
    assert_eq!(debug_str, "Sigmoid");

    let debug_str = format!("{:?}", Op::Exp(Arc::clone(&t)));
    assert_eq!(debug_str, "Exp");

    let debug_str = format!("{:?}", Op::Log(Arc::clone(&t)));
    assert_eq!(debug_str, "Log");

    let debug_str = format!("{:?}", Op::Sqrt(Arc::clone(&t)));
    assert_eq!(debug_str, "Sqrt");

    let debug_str = format!("{:?}", Op::Neg(Arc::clone(&t)));
    assert_eq!(debug_str, "Neg");

    let debug_str = format!("{:?}", Op::Abs(Arc::clone(&t)));
    assert_eq!(debug_str, "Abs");

    let debug_str = format!("{:?}", Op::Sin(Arc::clone(&t)));
    assert_eq!(debug_str, "Sin");

    let debug_str = format!("{:?}", Op::Cos(Arc::clone(&t)));
    assert_eq!(debug_str, "Cos");

    let debug_str = format!("{:?}", Op::Recip(Arc::clone(&t)));
    assert_eq!(debug_str, "Recip");

    let debug_str = format!("{:?}", Op::Powf(Arc::clone(&t), 2.5));
    assert_eq!(debug_str, "Powf(2.5)");

    let debug_str = format!("{:?}", Op::Clamp(Arc::clone(&t), -1.0, 1.0));
    assert_eq!(debug_str, "Clamp(-1, 1)");

    let debug_str = format!("{:?}", Op::Elu(Arc::clone(&t), 1.0));
    assert_eq!(debug_str, "Elu(alpha=1)");

    let debug_str = format!("{:?}", Op::LogSoftmax(Arc::clone(&t), 0));
    assert_eq!(debug_str, "LogSoftmax(dim=0)");

    let t2 = Arc::new(TrackedTensor::from_tensor(
        DynTensor::from_vec(vec![2.0], &[1], &cpu()).unwrap(),
    ));
    let debug_str = format!("{:?}", Op::Maximum(Arc::clone(&t), Arc::clone(&t2)));
    assert_eq!(debug_str, "Maximum");

    let debug_str = format!("{:?}", Op::Minimum(Arc::clone(&t), Arc::clone(&t2)));
    assert_eq!(debug_str, "Minimum");

    let debug_str = format!("{:?}", Op::Stack(vec![Arc::clone(&t)], 0));
    assert_eq!(debug_str, "Stack(dim=0)");

    let debug_str = format!("{:?}", Op::AddScalar(Arc::clone(&t), 5.0));
    assert_eq!(debug_str, "AddScalar(5)");

    let debug_str = format!("{:?}", Op::Permute(Arc::clone(&t), vec![1, 0]));
    assert_eq!(debug_str, "Permute([1, 0])");

    let debug_str = format!("{:?}", Op::Unsqueeze(Arc::clone(&t), 0));
    assert_eq!(debug_str, "Unsqueeze(0)");

    let debug_str = format!("{:?}", Op::Squeeze(Arc::clone(&t), 0));
    assert_eq!(debug_str, "Squeeze(0)");

    let debug_str = format!("{:?}", Op::Unfold(Arc::clone(&t), 0, 3, 1));
    assert_eq!(debug_str, "Unfold(dim=0, size=3, step=1)");
}

// ---------------------------------------------------------------------------
// Gradient accumulation (fan-in)
// ---------------------------------------------------------------------------

#[test]
fn test_backward_gradient_accumulation_fan_in() {
    // y = x + x = 2*x, dy/dx = 2
    // Tests that gradients accumulate when a variable is used multiple times.
    let x = scalar_var(3.0);
    let t = Arc::new(TrackedTensor::from_var(&x).unwrap());
    let y = t.add(&t).unwrap();
    let grads = backward(&y).unwrap();
    let grad = grads.get(&x).unwrap().to_flat_vec::<f32>().unwrap();
    assert!(
        (grad[0] - 2.0).abs() < 1e-6,
        "expected 2.0, got {}",
        grad[0]
    );
}

#[test]
fn test_backward_gradient_accumulation_triple() {
    // y = x + x + x = 3*x, dy/dx = 3
    let x = scalar_var(1.0);
    let t = Arc::new(TrackedTensor::from_var(&x).unwrap());
    let y1 = t.add(&t).unwrap(); // 2x
    let y2 = y1.add(&t).unwrap(); // 3x
    let grads = backward(&y2).unwrap();
    let grad = grads.get(&x).unwrap().to_flat_vec::<f32>().unwrap();
    assert!(
        (grad[0] - 3.0).abs() < 1e-5,
        "expected 3.0, got {}",
        grad[0]
    );
}

// ---------------------------------------------------------------------------
// Constant leaves should not receive gradients
// ---------------------------------------------------------------------------

#[test]
fn test_backward_constant_no_gradient() {
    // Only variables (from Var) receive gradients, not constants (from_tensor).
    let x = scalar_var(2.0);
    let c = Arc::new(TrackedTensor::from_tensor(
        DynTensor::from_vec(vec![3.0], &[1], &cpu()).unwrap(),
    ));
    let tx = Arc::new(TrackedTensor::from_var(&x).unwrap());
    let y = tx.mul(&c).unwrap();
    let grads = backward(&y).unwrap();
    // x gets gradient 3.0 (value of c)
    let grad_x = grads.get(&x).unwrap().to_flat_vec::<f32>().unwrap();
    assert!((grad_x[0] - 3.0).abs() < 1e-6);
    // var_count should be 1 (only x)
    assert_eq!(grads.var_count(), 1);
}

// ---------------------------------------------------------------------------
// AutodiffError Display tests
// ---------------------------------------------------------------------------

#[test]
fn test_error_display_non_scalar_loss() {
    let err = nn_autodiff::AutodiffError::NonScalarLoss { shape: vec![2, 3] };
    let msg = format!("{err}");
    assert!(msg.contains("scalar"));
    assert!(msg.contains("[2, 3]"));
}

#[test]
fn test_error_display_non_finite_loss() {
    let err = nn_autodiff::AutodiffError::NonFiniteLoss;
    let msg = format!("{err}");
    assert!(msg.contains("finite"));
}

#[test]
fn test_error_display_unsupported_backward() {
    let err = nn_autodiff::AutodiffError::UnsupportedBackward("FooOp".to_string());
    let msg = format!("{err}");
    assert!(msg.contains("FooOp"));
}

#[test]
fn test_error_display_shape_mismatch() {
    let err = nn_autodiff::AutodiffError::ShapeMismatch {
        expected: vec![2, 3],
        got: vec![4, 5],
    };
    let msg = format!("{err}");
    assert!(msg.contains("shape mismatch"));
}

#[test]
fn test_error_display_dropout() {
    let err = nn_autodiff::AutodiffError::Dropout { p: 1.5 };
    let msg = format!("{err}");
    assert!(msg.contains("1.5"));
}

#[test]
fn test_error_display_matmul_rank_too_low() {
    let err = nn_autodiff::AutodiffError::MatMulRankTooLow {
        rank_a: 1,
        rank_b: 1,
    };
    let msg = format!("{err}");
    assert!(msg.contains("rank"));
}

#[test]
fn test_error_display_not_contiguous() {
    let err = nn_autodiff::AutodiffError::NotContiguous { op: "MaxPool1d" };
    let msg = format!("{err}");
    assert!(msg.contains("MaxPool1d"));
    assert!(msg.contains("contiguous"));
}

#[test]
fn test_error_display_wrong_input_rank() {
    let err = nn_autodiff::AutodiffError::WrongInputRank {
        op: "Conv1d",
        expected: 3,
        actual: 2,
    };
    let msg = format!("{err}");
    assert!(msg.contains("Conv1d"));
    assert!(msg.contains("3"));
}

#[test]
fn test_error_display_empty_sequence() {
    let err = nn_autodiff::AutodiffError::EmptySequence { op: "LSTM" };
    let msg = format!("{err}");
    assert!(msg.contains("LSTM"));
    assert!(msg.contains("empty"));
}

#[test]
fn test_error_display_lock_poisoned() {
    let err = nn_autodiff::AutodiffError::LockPoisoned {
        context: "Var::data() read",
    };
    let msg = format!("{err}");
    assert!(msg.contains("poisoned"));
}

#[test]
fn test_error_display_non_finite_checkpoint() {
    let err = nn_autodiff::AutodiffError::NonFiniteCheckpoint {
        name: "layer.weight".to_string(),
        count: 5,
    };
    let msg = format!("{err}");
    assert!(msg.contains("layer.weight"));
    assert!(msg.contains("5"));
}

#[test]
fn test_error_display_non_finite_backward_input() {
    let err = nn_autodiff::AutodiffError::NonFiniteBackwardInput { op: "Maximum" };
    let msg = format!("{err}");
    assert!(msg.contains("Maximum"));
    assert!(msg.contains("non-finite"));
}

#[test]
fn test_error_display_dtype_mismatch() {
    let err = nn_autodiff::AutodiffError::DTypeMismatch {
        name: "weight".to_string(),
        expected: DType::F32,
        got: DType::BF16,
    };
    let msg = format!("{err}");
    assert!(msg.contains("weight"));
    assert!(msg.contains("dtype mismatch"));
}

#[test]
fn test_error_display_invalid_config() {
    let err = nn_autodiff::AutodiffError::InvalidConfig {
        op: "GroupNorm",
        reason: "num_groups must divide channels".to_string(),
    };
    let msg = format!("{err}");
    assert!(msg.contains("GroupNorm"));
    assert!(msg.contains("num_groups"));
}

#[test]
fn test_error_display_index_overflow() {
    let err = nn_autodiff::AutodiffError::IndexOverflow {
        op: "MaxPool2d",
        index: 5_000_000_000,
        max: u32::MAX,
    };
    let msg = format!("{err}");
    assert!(msg.contains("MaxPool2d"));
    assert!(msg.contains("u32"));
}

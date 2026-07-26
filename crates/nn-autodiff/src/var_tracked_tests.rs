// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Comprehensive tests for Var, TrackedTensor, Op recording, GradStore,
//! backward traversal, multi-path gradients, and memory safety.

use crate::grad::{backward, backward_for_vars, GradStore};
use crate::op::Op;
use crate::tracked::TrackedTensor;
use crate::var::Var;
use crate::AutodiffError;
use nn_core::dyn_tensor::DynTensor;
use nn_core::test_utils::cpu;
use nn_core::DType;
use std::sync::Arc;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn t(data: &[f32], dims: &[usize]) -> DynTensor {
    DynTensor::new(data, dims, &cpu()).unwrap()
}

fn scalar(val: f32) -> DynTensor {
    DynTensor::from_vec(vec![val], &[1], &cpu()).unwrap()
}

fn flat(tensor: &DynTensor) -> Vec<f32> {
    tensor.to_flat_vec::<f32>().unwrap()
}

// ---------------------------------------------------------------------------
// Var: creation
// ---------------------------------------------------------------------------

#[test]
fn test_var_new_preserves_data_and_shape() {
    let data = t(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
    let var = Var::new(data);
    assert_eq!(var.dims().unwrap(), &[2, 3]);
    assert_eq!(var.dtype().unwrap(), DType::F32);
    assert_eq!(flat(&var.data().unwrap()), &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
}

#[test]
fn test_var_zeros_all_zeros() {
    let var = Var::zeros(&[4, 3], DType::F32, &cpu()).unwrap();
    let data = flat(&var.data().unwrap());
    assert_eq!(data.len(), 12);
    assert!(data.iter().all(|&v| v == 0.0));
}

#[test]
fn test_var_from_tensor_clones_data() {
    let original = t(&[10.0, 20.0], &[2]);
    let var = Var::from_tensor(&original);
    assert_eq!(flat(&var.data().unwrap()), &[10.0, 20.0]);
}

#[test]
fn test_var_each_has_unique_id() {
    let vars: Vec<Var> = (0..10)
        .map(|_| Var::zeros(&[1], DType::F32, &cpu()).unwrap())
        .collect();
    let ids: Vec<_> = vars.iter().map(Var::id).collect();
    for i in 0..ids.len() {
        for j in (i + 1)..ids.len() {
            assert_ne!(ids[i], ids[j], "Var IDs must be globally unique");
        }
    }
}

// ---------------------------------------------------------------------------
// Var: set_data
// ---------------------------------------------------------------------------

#[test]
fn test_var_set_data_replaces_contents() {
    let var = Var::zeros(&[3], DType::F32, &cpu()).unwrap();
    let new = t(&[7.0, 8.0, 9.0], &[3]);
    var.set(&new).unwrap();
    assert_eq!(flat(&var.data().unwrap()), &[7.0, 8.0, 9.0]);
}

#[test]
fn test_var_set_rejects_wrong_shape() {
    let var = Var::zeros(&[2, 3], DType::F32, &cpu()).unwrap();
    let wrong = t(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[3, 2]);
    let err = var.set(&wrong).unwrap_err();
    let msg = format!("{err}");
    assert!(
        msg.contains("shape") || msg.contains("mismatch"),
        "error: {msg}"
    );
}

#[test]
fn test_var_set_rejects_wrong_dtype() {
    let var = Var::zeros(&[2], DType::F32, &cpu()).unwrap();
    let wrong_dtype = DynTensor::zeros(&[2], DType::U32, &cpu()).unwrap();
    let err = var.set(&wrong_dtype).unwrap_err();
    let msg = format!("{err}");
    assert!(msg.contains("mismatch"), "error: {msg}");
}

// ---------------------------------------------------------------------------
// Var: clone shares underlying storage via Arc<RwLock>
// ---------------------------------------------------------------------------

#[test]
fn test_var_clone_shares_state() {
    let var = Var::new(t(&[1.0, 2.0], &[2]));
    let cloned = var.clone();
    // Same ID
    assert_eq!(var.id(), cloned.id());
    // Mutation through clone visible from original
    let updated = t(&[99.0, 100.0], &[2]);
    cloned.set(&updated).unwrap();
    assert_eq!(flat(&var.data().unwrap()), &[99.0, 100.0]);
}

// ---------------------------------------------------------------------------
// TrackedTensor: from_var and from_tensor
// ---------------------------------------------------------------------------

#[test]
fn test_tracked_from_var_records_var_id() {
    let var = Var::new(t(&[5.0], &[1]));
    let tracked = TrackedTensor::from_var(&var).unwrap();
    assert!(tracked.is_var());
    assert_eq!(tracked.var_id(), Some(var.id()));
    assert!(tracked.op().is_none(), "leaf has no op");
}

#[test]
fn test_tracked_from_tensor_is_constant_leaf() {
    let tracked = TrackedTensor::from_tensor(t(&[1.0, 2.0, 3.0], &[3]));
    assert!(!tracked.is_var());
    assert_eq!(tracked.var_id(), None);
    assert!(tracked.op().is_none());
    assert_eq!(tracked.dims(), &[3]);
    assert_eq!(tracked.numel(), 3);
}

// ---------------------------------------------------------------------------
// TrackedTensor: operations record correct Op variants
// ---------------------------------------------------------------------------

#[test]
fn test_op_add_variant() {
    let a = Arc::new(TrackedTensor::from_tensor(t(&[1.0], &[1])));
    let b = Arc::new(TrackedTensor::from_tensor(t(&[2.0], &[1])));
    let c = a.add(&b).unwrap();
    assert!(matches!(c.op(), Some(Op::Add(..))));
    assert_eq!(flat(c.tensor()), &[3.0]);
}

#[test]
fn test_op_sub_variant() {
    let a = Arc::new(TrackedTensor::from_tensor(t(&[5.0], &[1])));
    let b = Arc::new(TrackedTensor::from_tensor(t(&[3.0], &[1])));
    let c = a.sub(&b).unwrap();
    assert!(matches!(c.op(), Some(Op::Sub(..))));
    assert_eq!(flat(c.tensor()), &[2.0]);
}

#[test]
fn test_op_mul_variant() {
    let a = Arc::new(TrackedTensor::from_tensor(t(&[3.0], &[1])));
    let b = Arc::new(TrackedTensor::from_tensor(t(&[4.0], &[1])));
    let c = a.mul(&b).unwrap();
    assert!(matches!(c.op(), Some(Op::Mul(..))));
    assert_eq!(flat(c.tensor()), &[12.0]);
}

#[test]
fn test_op_div_variant() {
    let a = Arc::new(TrackedTensor::from_tensor(t(&[10.0], &[1])));
    let b = Arc::new(TrackedTensor::from_tensor(t(&[2.0], &[1])));
    let c = a.div(&b).unwrap();
    assert!(matches!(c.op(), Some(Op::Div(..))));
    assert_eq!(flat(c.tensor()), &[5.0]);
}

#[test]
fn test_op_matmul_variant() {
    let a = Arc::new(TrackedTensor::from_tensor(t(
        &[1.0, 2.0, 3.0, 4.0],
        &[2, 2],
    )));
    let b = Arc::new(TrackedTensor::from_tensor(t(
        &[1.0, 0.0, 0.0, 1.0],
        &[2, 2],
    )));
    let c = a.matmul(&b).unwrap();
    assert!(matches!(c.op(), Some(Op::MatMul(..))));
    // Identity matmul
    assert_eq!(flat(c.tensor()), &[1.0, 2.0, 3.0, 4.0]);
}

#[test]
fn test_op_relu_variant() {
    let x = Arc::new(TrackedTensor::from_tensor(t(&[-1.0, 0.0, 1.0], &[3])));
    let y = x.relu().unwrap();
    assert!(matches!(y.op(), Some(Op::Relu(..))));
}

#[test]
fn test_op_sqr_variant() {
    let x = Arc::new(TrackedTensor::from_tensor(t(&[3.0], &[1])));
    let y = x.sqr().unwrap();
    assert!(matches!(y.op(), Some(Op::Sqr(..))));
    assert_eq!(flat(y.tensor()), &[9.0]);
}

#[test]
fn test_op_neg_variant() {
    let x = Arc::new(TrackedTensor::from_tensor(t(&[5.0], &[1])));
    let y = x.neg().unwrap();
    assert!(matches!(y.op(), Some(Op::Neg(..))));
    assert_eq!(flat(y.tensor()), &[-5.0]);
}

#[test]
fn test_op_exp_variant() {
    let x = Arc::new(TrackedTensor::from_tensor(t(&[0.0], &[1])));
    let y = x.exp().unwrap();
    assert!(matches!(y.op(), Some(Op::Exp(..))));
    let vals = flat(y.tensor());
    assert!((vals[0] - 1.0).abs() < 1e-6);
}

#[test]
fn test_op_log_variant() {
    let x = Arc::new(TrackedTensor::from_tensor(t(&[1.0], &[1])));
    let y = x.log().unwrap();
    assert!(matches!(y.op(), Some(Op::Log(..))));
    let vals = flat(y.tensor());
    assert!((vals[0]).abs() < 1e-6); // ln(1) = 0
}

#[test]
fn test_op_tanh_variant() {
    let x = Arc::new(TrackedTensor::from_tensor(t(&[0.0], &[1])));
    let y = x.tanh().unwrap();
    assert!(matches!(y.op(), Some(Op::Tanh(..))));
    assert!((flat(y.tensor())[0]).abs() < 1e-6); // tanh(0) = 0
}

#[test]
fn test_op_sigmoid_variant() {
    let x = Arc::new(TrackedTensor::from_tensor(t(&[0.0], &[1])));
    let y = x.sigmoid().unwrap();
    assert!(matches!(y.op(), Some(Op::Sigmoid(..))));
    assert!((flat(y.tensor())[0] - 0.5).abs() < 1e-6); // sigmoid(0) = 0.5
}

#[test]
fn test_op_sum_keepdim_variant() {
    let x = Arc::new(TrackedTensor::from_tensor(t(
        &[1.0, 2.0, 3.0, 4.0],
        &[2, 2],
    )));
    let y = x.sum_keepdim(1).unwrap();
    assert!(matches!(y.op(), Some(Op::SumKeepDim(_, 1))));
    assert_eq!(y.dims(), &[2, 1]);
    assert_eq!(flat(y.tensor()), &[3.0, 7.0]);
}

#[test]
fn test_op_mean_keepdim_variant() {
    let x = Arc::new(TrackedTensor::from_tensor(t(
        &[2.0, 4.0, 6.0, 8.0],
        &[2, 2],
    )));
    let y = x.mean_keepdim(1).unwrap();
    assert!(matches!(y.op(), Some(Op::MeanKeepDim(_, 1))));
    assert_eq!(y.dims(), &[2, 1]);
    assert_eq!(flat(y.tensor()), &[3.0, 7.0]);
}

#[test]
fn test_op_reshape_stores_original_shape() {
    let x = Arc::new(TrackedTensor::from_tensor(t(
        &[1.0, 2.0, 3.0, 4.0],
        &[2, 2],
    )));
    let y = x.reshape(&[4]).unwrap();
    match y.op() {
        Some(Op::Reshape(_, orig)) => assert_eq!(orig, &[2, 2]),
        other => panic!("expected Reshape, got {other:?}"),
    }
    assert_eq!(y.dims(), &[4]);
}

#[test]
fn test_op_transpose_variant() {
    let x = Arc::new(TrackedTensor::from_tensor(t(
        &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        &[2, 3],
    )));
    let y = x.transpose(0, 1).unwrap();
    assert!(matches!(y.op(), Some(Op::Transpose(_, 0, 1))));
    assert_eq!(y.dims(), &[3, 2]);
}

#[test]
fn test_op_unsqueeze_variant() {
    let x = Arc::new(TrackedTensor::from_tensor(t(&[1.0, 2.0], &[2])));
    let y = x.unsqueeze(0).unwrap();
    assert!(matches!(y.op(), Some(Op::Unsqueeze(_, 0))));
    assert_eq!(y.dims(), &[1, 2]);
}

#[test]
fn test_op_squeeze_variant() {
    let x = Arc::new(TrackedTensor::from_tensor(t(&[1.0, 2.0], &[1, 2])));
    let y = x.squeeze(0).unwrap();
    assert!(matches!(y.op(), Some(Op::Squeeze(_, 0))));
    assert_eq!(y.dims(), &[2]);
}

#[test]
fn test_op_mul_scalar_variant() {
    let x = Arc::new(TrackedTensor::from_tensor(t(&[2.0], &[1])));
    let y = x.mul_scalar(3.0).unwrap();
    match y.op() {
        Some(Op::MulScalar(_, v)) => assert!((v - 3.0).abs() < 1e-12),
        other => panic!("expected MulScalar, got {other:?}"),
    }
    assert_eq!(flat(y.tensor()), &[6.0]);
}

#[test]
fn test_op_add_scalar_variant() {
    let x = Arc::new(TrackedTensor::from_tensor(t(&[10.0], &[1])));
    let y = x.add_scalar(5.0).unwrap();
    match y.op() {
        Some(Op::AddScalar(_, v)) => assert!((v - 5.0).abs() < 1e-12),
        other => panic!("expected AddScalar, got {other:?}"),
    }
    assert_eq!(flat(y.tensor()), &[15.0]);
}

#[test]
fn test_op_powf_variant() {
    let x = Arc::new(TrackedTensor::from_tensor(t(&[2.0], &[1])));
    let y = x.powf(3.0).unwrap();
    match y.op() {
        Some(Op::Powf(_, exp)) => assert!((exp - 3.0).abs() < 1e-12),
        other => panic!("expected Powf, got {other:?}"),
    }
    assert!((flat(y.tensor())[0] - 8.0).abs() < 1e-4);
}

#[test]
fn test_op_clamp_variant() {
    let x = Arc::new(TrackedTensor::from_tensor(t(&[-5.0, 0.0, 5.0], &[3])));
    let y = x.clamp(-1.0, 1.0).unwrap();
    assert!(matches!(y.op(), Some(Op::Clamp(_, _, _))));
    assert_eq!(flat(y.tensor()), &[-1.0, 0.0, 1.0]);
}

#[test]
fn test_op_abs_variant() {
    let x = Arc::new(TrackedTensor::from_tensor(t(&[-3.0, 4.0], &[2])));
    let y = x.abs().unwrap();
    assert!(matches!(y.op(), Some(Op::Abs(..))));
    assert_eq!(flat(y.tensor()), &[3.0, 4.0]);
}

// ---------------------------------------------------------------------------
// GradStore: basic backward traversal and gradient shapes
// ---------------------------------------------------------------------------

#[test]
fn test_backward_scalar_loss_x_squared() {
    // f(x) = x^2, df/dx = 2x. At x=3, grad=6.
    let x = Var::new(scalar(3.0));
    let tx = Arc::new(TrackedTensor::from_var(&x).unwrap());
    let loss = tx.sqr().unwrap();
    let grads = backward(&loss).unwrap();
    let g = grads.get(&x).unwrap();
    assert_eq!(g.dims(), &[1]);
    assert!(
        (flat(g)[0] - 6.0).abs() < 1e-5,
        "df/dx = 2*3 = 6, got {}",
        flat(g)[0]
    );
}

#[test]
fn test_backward_add_both_grads_are_one() {
    // f(a, b) = a + b, df/da = df/db = 1
    let a = Var::new(scalar(2.0));
    let b = Var::new(scalar(5.0));
    let ta = Arc::new(TrackedTensor::from_var(&a).unwrap());
    let tb = Arc::new(TrackedTensor::from_var(&b).unwrap());
    let loss = ta.add(&tb).unwrap();
    let grads = backward(&loss).unwrap();
    assert!((flat(grads.get(&a).unwrap())[0] - 1.0).abs() < 1e-6);
    assert!((flat(grads.get(&b).unwrap())[0] - 1.0).abs() < 1e-6);
}

#[test]
fn test_backward_mul_product_rule() {
    // f(a, b) = a * b, df/da = b, df/db = a
    let a = Var::new(scalar(3.0));
    let b = Var::new(scalar(4.0));
    let ta = Arc::new(TrackedTensor::from_var(&a).unwrap());
    let tb = Arc::new(TrackedTensor::from_var(&b).unwrap());
    let loss = ta.mul(&tb).unwrap();
    let grads = backward(&loss).unwrap();
    assert!((flat(grads.get(&a).unwrap())[0] - 4.0).abs() < 1e-5);
    assert!((flat(grads.get(&b).unwrap())[0] - 3.0).abs() < 1e-5);
}

#[test]
fn test_backward_gradient_shape_matches_variable() {
    // Var shape [2, 3], loss = sum(x^2)
    let x = Var::new(t(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]));
    let tx = Arc::new(TrackedTensor::from_var(&x).unwrap());
    let sq = tx.sqr().unwrap();
    let loss = sq.sum_keepdim(1).unwrap().sum_keepdim(0).unwrap();
    let grads = backward(&loss).unwrap();
    let g = grads.get(&x).unwrap();
    assert_eq!(
        g.dims(),
        &[2, 3],
        "gradient shape must match variable shape"
    );
    // d(sum(x^2))/dx_i = 2*x_i
    let expected = [2.0, 4.0, 6.0, 8.0, 10.0, 12.0];
    let actual = flat(g);
    for (a, e) in actual.iter().zip(expected.iter()) {
        assert!((a - e).abs() < 1e-4, "expected {e}, got {a}");
    }
}

// ---------------------------------------------------------------------------
// GradStore: backward_for_vars (selective)
// ---------------------------------------------------------------------------

#[test]
fn test_backward_for_vars_filters_correctly() {
    let w1 = Var::new(scalar(2.0));
    let w2 = Var::new(scalar(3.0));
    let tw1 = Arc::new(TrackedTensor::from_var(&w1).unwrap());
    let tw2 = Arc::new(TrackedTensor::from_var(&w2).unwrap());
    let loss = tw1.mul(&tw2).unwrap(); // loss = w1 * w2
    let grads = backward_for_vars(&loss, &[&w1]).unwrap();
    assert!(grads.get(&w1).is_some(), "w1 gradient should be present");
    assert!(
        grads.get(&w2).is_none(),
        "w2 gradient should be filtered out"
    );
    assert_eq!(grads.var_count(), 1);
}

// ---------------------------------------------------------------------------
// GradStore: retain_only and var_count
// ---------------------------------------------------------------------------

#[test]
fn test_gradstore_retain_only() {
    let a = Var::new(scalar(1.0));
    let b = Var::new(scalar(2.0));
    let c = Var::new(scalar(3.0));
    let ta = Arc::new(TrackedTensor::from_var(&a).unwrap());
    let tb = Arc::new(TrackedTensor::from_var(&b).unwrap());
    let tc = Arc::new(TrackedTensor::from_var(&c).unwrap());

    // loss = a + b + c
    let ab = ta.add(&tb).unwrap();
    let loss = ab.add(&tc).unwrap();
    let mut grads = backward(&loss).unwrap();
    assert_eq!(grads.var_count(), 3);

    grads.retain_only(&[&a, &c]);
    assert_eq!(grads.var_count(), 2);
    assert!(grads.get(&a).is_some());
    assert!(grads.get(&b).is_none());
    assert!(grads.get(&c).is_some());
}

#[test]
fn test_gradstore_var_grads_iter() {
    let a = Var::new(scalar(2.0));
    let b = Var::new(scalar(3.0));
    let ta = Arc::new(TrackedTensor::from_var(&a).unwrap());
    let tb = Arc::new(TrackedTensor::from_var(&b).unwrap());
    let loss = ta.add(&tb).unwrap();
    let grads = backward(&loss).unwrap();
    let count = grads.var_grads().count();
    assert_eq!(count, 2);
}

// ---------------------------------------------------------------------------
// Multi-path gradients: diamond pattern
// ---------------------------------------------------------------------------

#[test]
fn test_diamond_gradient_accumulation() {
    // Diamond: x -> y = x*2, x -> z = x*3, loss = y + z = 5x
    // d(loss)/dx = 5
    let x = Var::new(scalar(1.0));
    let tx = Arc::new(TrackedTensor::from_var(&x).unwrap());
    let y = tx.mul_scalar(2.0).unwrap(); // y = 2x
    let z = tx.mul_scalar(3.0).unwrap(); // z = 3x
    let loss = y.add(&z).unwrap(); // loss = 2x + 3x = 5x
    let grads = backward(&loss).unwrap();
    let g = flat(grads.get(&x).unwrap());
    assert!((g[0] - 5.0).abs() < 1e-5, "d(5x)/dx = 5, got {}", g[0]);
}

#[test]
fn test_diamond_with_multiply() {
    // x -> y = x + 1, x -> z = x * 2, loss = y * z
    // loss = (x+1)(2x) = 2x^2 + 2x. d(loss)/dx = 4x + 2
    // At x=3: d(loss)/dx = 14
    let x = Var::new(scalar(3.0));
    let tx = Arc::new(TrackedTensor::from_var(&x).unwrap());
    let y = tx.add_scalar(1.0).unwrap(); // y = x + 1 = 4
    let z = tx.mul_scalar(2.0).unwrap(); // z = 2x = 6
    let loss = y.mul(&z).unwrap(); // loss = 24
    assert!((flat(loss.tensor())[0] - 24.0).abs() < 1e-4);
    let grads = backward(&loss).unwrap();
    let g = flat(grads.get(&x).unwrap());
    assert!(
        (g[0] - 14.0).abs() < 1e-4,
        "d(2x^2+2x)/dx at x=3 = 14, got {}",
        g[0]
    );
}

#[test]
fn test_triple_fan_out() {
    // x used three times: loss = x + x + x = 3x, d(loss)/dx = 3
    let x = Var::new(scalar(7.0));
    let tx = Arc::new(TrackedTensor::from_var(&x).unwrap());
    let a = tx.add(&tx).unwrap(); // 2x
    let loss = a.add(&tx).unwrap(); // 3x
    let grads = backward(&loss).unwrap();
    let g = flat(grads.get(&x).unwrap());
    assert!((g[0] - 3.0).abs() < 1e-5, "d(3x)/dx = 3, got {}", g[0]);
}

// ---------------------------------------------------------------------------
// Edge cases: scalar loss validation
// ---------------------------------------------------------------------------

#[test]
fn test_backward_rejects_non_scalar_loss() {
    let x = Var::new(t(&[1.0, 2.0], &[2]));
    let tx = Arc::new(TrackedTensor::from_var(&x).unwrap());
    let err = backward(&tx).unwrap_err();
    assert!(
        matches!(err, AutodiffError::NonScalarLoss { .. }),
        "expected NonScalarLoss, got {err:?}"
    );
}

#[test]
fn test_backward_rejects_nan_loss() {
    let x = Var::new(scalar(f32::NAN));
    let tx = Arc::new(TrackedTensor::from_var(&x).unwrap());
    let err = backward(&tx).unwrap_err();
    assert!(
        matches!(err, AutodiffError::NonFiniteLoss),
        "expected NonFiniteLoss, got {err:?}"
    );
}

#[test]
fn test_backward_rejects_inf_loss() {
    let x = Var::new(scalar(f32::INFINITY));
    let tx = Arc::new(TrackedTensor::from_var(&x).unwrap());
    let err = backward(&tx).unwrap_err();
    assert!(
        matches!(err, AutodiffError::NonFiniteLoss),
        "expected NonFiniteLoss, got {err:?}"
    );
}

// ---------------------------------------------------------------------------
// Edge case: constant tensors get no gradient
// ---------------------------------------------------------------------------

#[test]
fn test_constant_gets_no_gradient() {
    let x = Var::new(scalar(2.0));
    let c = t(&[10.0], &[1]); // constant, not a Var
    let tx = Arc::new(TrackedTensor::from_var(&x).unwrap());
    let tc = Arc::new(TrackedTensor::from_tensor(c));
    let loss = tx.add(&tc).unwrap(); // loss = x + 10
    let grads = backward(&loss).unwrap();
    // Only x should have gradient
    assert_eq!(grads.var_count(), 1);
    assert!((flat(grads.get(&x).unwrap())[0] - 1.0).abs() < 1e-6);
}

// ---------------------------------------------------------------------------
// Edge case: zero gradients when variable is unused
// ---------------------------------------------------------------------------

#[test]
fn test_unused_variable_gets_no_gradient() {
    let x = Var::new(scalar(5.0));
    let y = Var::new(scalar(3.0));
    let ty = Arc::new(TrackedTensor::from_var(&y).unwrap());
    // loss only depends on y, not x
    let loss = ty.mul_scalar(2.0).unwrap();
    let grads = backward(&loss).unwrap();
    assert!(
        grads.get(&x).is_none(),
        "unused var should have no gradient"
    );
    assert!(grads.get(&y).is_some());
}

// ---------------------------------------------------------------------------
// Memory: Arc reference counting
// ---------------------------------------------------------------------------

#[test]
fn test_arc_refcount_after_op() {
    let a = Arc::new(TrackedTensor::from_tensor(t(&[1.0], &[1])));
    assert_eq!(Arc::strong_count(&a), 1);
    let b = Arc::new(TrackedTensor::from_tensor(t(&[2.0], &[1])));
    let c = a.add(&b).unwrap();
    // a and b are referenced by c's Op::Add
    assert_eq!(Arc::strong_count(&a), 2, "a referenced by c's Op");
    assert_eq!(Arc::strong_count(&b), 2, "b referenced by c's Op");
    drop(c);
    assert_eq!(
        Arc::strong_count(&a),
        1,
        "after drop(c), a refcount back to 1"
    );
    assert_eq!(
        Arc::strong_count(&b),
        1,
        "after drop(c), b refcount back to 1"
    );
}

#[test]
fn test_diamond_refcounts() {
    let x = Arc::new(TrackedTensor::from_tensor(t(&[1.0], &[1])));
    assert_eq!(Arc::strong_count(&x), 1);
    let y = x.mul_scalar(2.0).unwrap(); // holds Arc to x
    assert_eq!(Arc::strong_count(&x), 2);
    let z = x.mul_scalar(3.0).unwrap(); // holds Arc to x
    assert_eq!(Arc::strong_count(&x), 3);
    let _sum = y.add(&z).unwrap(); // holds Arcs to y and z
    assert_eq!(
        Arc::strong_count(&x),
        3,
        "x still referenced by y and z ops"
    );
    // y and z each have 2 refs: local + sum's Op
    assert_eq!(Arc::strong_count(&y), 2);
    assert_eq!(Arc::strong_count(&z), 2);
}

#[test]
fn test_no_leak_on_drop() {
    let x = Arc::new(TrackedTensor::from_tensor(t(&[1.0], &[1])));
    let weak = Arc::downgrade(&x);
    let y = x.sqr().unwrap();
    let z = y.neg().unwrap();
    // Drop everything
    drop(z);
    drop(y);
    drop(x);
    assert!(
        weak.upgrade().is_none(),
        "all references should be freed after drop"
    );
}

// ---------------------------------------------------------------------------
// TrackedTensor: detach severs graph
// ---------------------------------------------------------------------------

#[test]
fn test_detach_stops_gradient_flow() {
    let x = Var::new(scalar(4.0));
    let tx = Arc::new(TrackedTensor::from_var(&x).unwrap());
    let sq = tx.sqr().unwrap(); // sq = x^2 = 16
    let detached = sq.detach(); // severs graph
    let loss = detached.add_scalar(1.0).unwrap(); // loss = 17
    let grads = backward(&loss).unwrap();
    assert!(
        grads.get(&x).is_none(),
        "gradient should not flow past detach"
    );
}

// ---------------------------------------------------------------------------
// TrackedTensor: into_tensor
// ---------------------------------------------------------------------------

#[test]
fn test_into_tensor_returns_data() {
    let data = t(&[10.0, 20.0, 30.0], &[3]);
    let tracked = TrackedTensor::from_tensor(data);
    let recovered = tracked.into_tensor().unwrap();
    assert_eq!(flat(&recovered), &[10.0, 20.0, 30.0]);
}

// ---------------------------------------------------------------------------
// GradStore: default construction
// ---------------------------------------------------------------------------

#[test]
fn test_gradstore_default_is_empty() {
    let gs = GradStore::default();
    assert_eq!(gs.var_count(), 0);
}

// ---------------------------------------------------------------------------
// Chain rule through multiple ops
// ---------------------------------------------------------------------------

#[test]
fn test_chain_rule_exp_of_mul() {
    // f(x) = exp(2*x). df/dx = 2*exp(2*x). At x=1, df/dx = 2*e^2 ~ 14.778
    let x = Var::new(scalar(1.0));
    let tx = Arc::new(TrackedTensor::from_var(&x).unwrap());
    let doubled = tx.mul_scalar(2.0).unwrap();
    let loss = doubled.exp().unwrap();
    let grads = backward(&loss).unwrap();
    let g = flat(grads.get(&x).unwrap())[0];
    let expected = 2.0 * (2.0_f32).exp();
    assert!((g - expected).abs() < 1e-3, "expected {expected}, got {g}");
}

#[test]
fn test_chain_rule_neg_of_sqr() {
    // f(x) = -(x^2). df/dx = -2x. At x=5, df/dx = -10.
    let x = Var::new(scalar(5.0));
    let tx = Arc::new(TrackedTensor::from_var(&x).unwrap());
    let sq = tx.sqr().unwrap();
    let loss = sq.neg().unwrap();
    let grads = backward(&loss).unwrap();
    let g = flat(grads.get(&x).unwrap())[0];
    assert!((g - (-10.0)).abs() < 1e-4, "expected -10, got {g}");
}

#[test]
fn test_chain_three_ops() {
    // f(x) = (x + 1)^2 = x^2 + 2x + 1. df/dx = 2x + 2. At x=3, grad = 8.
    let x = Var::new(scalar(3.0));
    let tx = Arc::new(TrackedTensor::from_var(&x).unwrap());
    let plus_one = tx.add_scalar(1.0).unwrap();
    let loss = plus_one.sqr().unwrap();
    let grads = backward(&loss).unwrap();
    let g = flat(grads.get(&x).unwrap())[0];
    assert!((g - 8.0).abs() < 1e-4, "expected 8, got {g}");
}

// ---------------------------------------------------------------------------
// GradStore: get_id
// ---------------------------------------------------------------------------

#[test]
fn test_gradstore_get_id() {
    let x = Var::new(scalar(2.0));
    let var_id = x.id();
    let tx = Arc::new(TrackedTensor::from_var(&x).unwrap());
    let loss = tx.sqr().unwrap();
    let grads = backward(&loss).unwrap();
    let by_var = grads.get(&x).unwrap();
    let by_id = grads.get_id(&var_id).unwrap();
    assert_eq!(flat(by_var), flat(by_id));
}

// ---------------------------------------------------------------------------
// Multi-variable gradient: linear combination
// ---------------------------------------------------------------------------

#[test]
fn test_multi_variable_linear_combination() {
    // loss = 2*a + 3*b. d(loss)/da = 2, d(loss)/db = 3.
    let a = Var::new(scalar(10.0));
    let b = Var::new(scalar(20.0));
    let ta = Arc::new(TrackedTensor::from_var(&a).unwrap());
    let tb = Arc::new(TrackedTensor::from_var(&b).unwrap());
    let two_a = ta.mul_scalar(2.0).unwrap();
    let three_b = tb.mul_scalar(3.0).unwrap();
    let loss = two_a.add(&three_b).unwrap();
    let grads = backward(&loss).unwrap();
    assert!((flat(grads.get(&a).unwrap())[0] - 2.0).abs() < 1e-5);
    assert!((flat(grads.get(&b).unwrap())[0] - 3.0).abs() < 1e-5);
}

// ---------------------------------------------------------------------------
// Reduction ops through backward
// ---------------------------------------------------------------------------

#[test]
fn test_sum_keepdim_gradient_is_ones() {
    // loss = sum(x). d(loss)/dx = [1, 1, 1, 1].
    let x = Var::new(t(&[1.0, 2.0, 3.0, 4.0], &[1, 4]));
    let tx = Arc::new(TrackedTensor::from_var(&x).unwrap());
    let loss = tx.sum_keepdim(1).unwrap();
    let grads = backward(&loss).unwrap();
    let g = flat(grads.get(&x).unwrap());
    assert_eq!(g, &[1.0, 1.0, 1.0, 1.0]);
}

#[test]
fn test_mean_keepdim_gradient() {
    // loss = mean(x) over dim 1 with 4 elements. d(loss)/dx_i = 1/4.
    let x = Var::new(t(&[4.0, 8.0, 12.0, 16.0], &[1, 4]));
    let tx = Arc::new(TrackedTensor::from_var(&x).unwrap());
    let loss = tx.mean_keepdim(1).unwrap();
    let grads = backward(&loss).unwrap();
    let g = flat(grads.get(&x).unwrap());
    for val in &g {
        assert!((val - 0.25).abs() < 1e-5, "mean grad = 1/4, got {val}");
    }
}

// ---------------------------------------------------------------------------
// Verify Op Debug format
// ---------------------------------------------------------------------------

#[test]
fn test_op_debug_format_coverage() {
    let x = Arc::new(TrackedTensor::from_tensor(t(&[1.0], &[1])));
    let y = Arc::new(TrackedTensor::from_tensor(t(&[2.0], &[1])));

    let ops_and_names: Vec<(Op, &str)> = vec![
        (Op::Add(Arc::clone(&x), Arc::clone(&y)), "Add"),
        (Op::Sub(Arc::clone(&x), Arc::clone(&y)), "Sub"),
        (Op::Mul(Arc::clone(&x), Arc::clone(&y)), "Mul"),
        (Op::Div(Arc::clone(&x), Arc::clone(&y)), "Div"),
        (Op::Relu(Arc::clone(&x)), "Relu"),
        (Op::Sqr(Arc::clone(&x)), "Sqr"),
        (Op::Neg(Arc::clone(&x)), "Neg"),
        (Op::Exp(Arc::clone(&x)), "Exp"),
        (Op::Log(Arc::clone(&x)), "Log"),
        (Op::Tanh(Arc::clone(&x)), "Tanh"),
        (Op::Sigmoid(Arc::clone(&x)), "Sigmoid"),
        (Op::Abs(Arc::clone(&x)), "Abs"),
        (Op::Sin(Arc::clone(&x)), "Sin"),
        (Op::Cos(Arc::clone(&x)), "Cos"),
        (Op::Recip(Arc::clone(&x)), "Recip"),
        (Op::Gelu(Arc::clone(&x)), "Gelu"),
        (Op::Silu(Arc::clone(&x)), "Silu"),
        (Op::Sqrt(Arc::clone(&x)), "Sqrt"),
        (Op::SumKeepDim(Arc::clone(&x), 0), "SumKeepDim"),
        (Op::MeanKeepDim(Arc::clone(&x), 1), "MeanKeepDim"),
        (Op::Reshape(Arc::clone(&x), vec![1]), "Reshape"),
        (Op::Transpose(Arc::clone(&x), 0, 1), "Transpose"),
        (Op::Unsqueeze(Arc::clone(&x), 0), "Unsqueeze"),
        (Op::Squeeze(Arc::clone(&x), 0), "Squeeze"),
        (Op::MulScalar(Arc::clone(&x), 2.0), "MulScalar"),
        (Op::AddScalar(Arc::clone(&x), 1.0), "AddScalar"),
        (Op::Powf(Arc::clone(&x), 2.0), "Powf"),
        (Op::Maximum(Arc::clone(&x), Arc::clone(&y)), "Maximum"),
        (Op::Minimum(Arc::clone(&x), Arc::clone(&y)), "Minimum"),
    ];
    for (op, expected_name) in ops_and_names {
        let debug = format!("{op:?}");
        assert!(
            debug.contains(expected_name),
            "Op::Debug for {expected_name} = '{debug}'"
        );
    }
}

// ---------------------------------------------------------------------------
// NodeId uniqueness across TrackedTensors
// ---------------------------------------------------------------------------

#[test]
fn test_node_ids_globally_unique() {
    let tensors: Vec<TrackedTensor> = (0..20)
        .map(|i| TrackedTensor::from_tensor(t(&[i as f32], &[1])))
        .collect();
    let ids: Vec<u64> = tensors.iter().map(|t| t.node_id().as_u64()).collect();
    for i in 0..ids.len() {
        for j in (i + 1)..ids.len() {
            assert_ne!(ids[i], ids[j], "node IDs must be unique");
        }
    }
}

// ---------------------------------------------------------------------------
// Var: device accessor
// ---------------------------------------------------------------------------

#[test]
fn test_var_device_is_cpu() {
    let var = Var::zeros(&[2], DType::F32, &cpu()).unwrap();
    assert_eq!(var.device().unwrap(), nn_core::Device::Cpu);
}

// ---------------------------------------------------------------------------
// GradStore: var_grads_mut
// ---------------------------------------------------------------------------

#[test]
fn test_gradstore_var_grads_mut_modifies_grads() {
    let x = Var::new(scalar(3.0));
    let tx = Arc::new(TrackedTensor::from_var(&x).unwrap());
    let loss = tx.sqr().unwrap(); // loss = 9, grad = 6
    let mut grads = backward(&loss).unwrap();
    // Scale all gradients by 0.5
    for (_, grad) in grads.var_grads_mut() {
        let scaled = grad.mul_scalar(0.5).unwrap();
        *grad = scaled;
    }
    let g = flat(grads.get(&x).unwrap())[0];
    assert!((g - 3.0).abs() < 1e-5, "6.0 * 0.5 = 3.0, got {g}");
}

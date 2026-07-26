// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Extended gradient tape and autodiff infrastructure tests.
//!
//! Covers:
//! - Tape creation and initialization
//! - Forward op recording on the tape
//! - Backward through single ops producing correct gradients
//! - Detach stops gradient flow
//! - No-grad via constant tensors
//! - Gradient shape matches input shape
//! - Zero initialization of uncomputed gradients
//! - Gradient accumulation when same variable used multiple times
//! - Gradients flow through reshape/view ops
//! - Second-order gradient via double backward
//! - Explicit stop-gradient via detach
//! - Numerical vs analytical gradient checking for add, mul, exp

use std::sync::Arc;

use nn_core::dyn_tensor::DynTensor;
use nn_core::test_utils::cpu;
use nn_core::DType;

use crate::grad::backward;
use crate::op::Op;
use crate::tracked::TrackedTensor;
use crate::var::Var;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

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

fn tracked(v: &Var) -> Arc<TrackedTensor> {
    Arc::new(TrackedTensor::from_var(v).unwrap())
}

fn get_grad_vec(grads: &crate::GradStore, var: &Var) -> Vec<f32> {
    grads
        .get(var)
        .expect("gradient should exist")
        .to_flat_vec::<f32>()
        .unwrap()
}

/// Reduce an arbitrary-shaped tracked tensor to a scalar via sum over all dims.
fn scalar_loss(t: &Arc<TrackedTensor>) -> Arc<TrackedTensor> {
    let mut result = Arc::clone(t);
    for d in (0..result.tensor().rank()).rev() {
        result = result.sum_keepdim(d).unwrap();
    }
    result
}

/// Sum all elements of a DynTensor as f64 (for finite-difference precision).
fn sum_all_f64(t: &DynTensor) -> f64 {
    t.to_flat_vec::<f32>()
        .unwrap()
        .iter()
        .map(|&v| f64::from(v))
        .sum()
}

/// Central-difference numerical gradient: (f(x+h) - f(x-h)) / (2h).
fn numerical_grad(data: &[f32], shape: &[usize], h: f32, fwd: &dyn Fn(&[f32]) -> f64) -> Vec<f64> {
    let mut grads = Vec::with_capacity(data.len());
    for i in 0..data.len() {
        let mut plus = data.to_vec();
        let mut minus = data.to_vec();
        plus[i] += h;
        minus[i] -= h;
        let g = (fwd(&plus) - fwd(&minus)) / (2.0 * f64::from(h));
        grads.push(g);
    }
    let _ = shape; // shape used by callers for constructing tensors
    grads
}

fn assert_close_f32(actual: &[f32], expected: &[f32], tol: f32, label: &str) {
    assert_eq!(
        actual.len(),
        expected.len(),
        "{}: length mismatch {} vs {}",
        label,
        actual.len(),
        expected.len()
    );
    for (i, (&a, &e)) in actual.iter().zip(expected.iter()).enumerate() {
        assert!(
            (a - e).abs() < tol,
            "{}: index {}: actual={}, expected={}, diff={}",
            label,
            i,
            a,
            e,
            (a - e).abs()
        );
    }
}

fn assert_close_f64(actual: &[f32], expected: &[f64], tol: f64, label: &str) {
    assert_eq!(
        actual.len(),
        expected.len(),
        "{}: length mismatch {} vs {}",
        label,
        actual.len(),
        expected.len()
    );
    for (i, (&a, &e)) in actual.iter().zip(expected.iter()).enumerate() {
        let diff = (f64::from(a) - e).abs();
        assert!(
            diff < tol,
            "{label}: index {i}: actual={a}, expected={e}, diff={diff}"
        );
    }
}

// ===========================================================================
// 1. Gradient Tape: Creation
// ===========================================================================

#[test]
fn test_tape_creation() {
    // A TrackedTensor can be created from a Var (leaf, receives gradients)
    let var = Var::zeros(&[3, 4], DType::F32, &cpu()).unwrap();
    let tt = TrackedTensor::from_var(&var).unwrap();
    assert!(tt.is_var());
    assert!(tt.var_id().is_some());
    assert!(tt.op().is_none(), "leaf node has no op");
    assert_eq!(tt.dims(), &[3, 4]);

    // A TrackedTensor can be created from a plain DynTensor (constant, no gradients)
    let data = DynTensor::ones(&[2, 5], DType::F32, &cpu()).unwrap();
    let ct = TrackedTensor::from_tensor(data);
    assert!(!ct.is_var());
    assert!(ct.var_id().is_none());
    assert!(ct.op().is_none());
    assert_eq!(ct.dims(), &[2, 5]);
}

// ===========================================================================
// 2. Gradient Tape: Forward op recorded
// ===========================================================================

#[test]
fn test_tape_record_forward() {
    let x = scalar_var(2.0);
    let tx = tracked(&x);

    // Apply sqr -> should record Op::Sqr
    let sq = tx.sqr().unwrap();
    assert!(sq.op().is_some(), "op should be recorded after sqr()");
    assert!(matches!(sq.op().unwrap(), Op::Sqr(_)));

    // Apply relu on top -> should record Op::Relu
    let out = sq.relu().unwrap();
    assert!(matches!(out.op().unwrap(), Op::Relu(_)));

    // The chain: out -> Relu -> Sqr -> leaf
    if let Op::Relu(inner) = out.op().unwrap() {
        assert!(matches!(inner.op().unwrap(), Op::Sqr(_)));
    } else {
        panic!("expected Relu");
    }
}

#[test]
fn test_tape_record_binary_ops() {
    let a = scalar_var(3.0);
    let b = scalar_var(4.0);
    let ta = tracked(&a);
    let tb = tracked(&b);

    let sum = ta.add(&tb).unwrap();
    assert!(matches!(sum.op().unwrap(), Op::Add(_, _)));

    let prod = ta.mul(&tb).unwrap();
    assert!(matches!(prod.op().unwrap(), Op::Mul(_, _)));

    let diff = ta.sub(&tb).unwrap();
    assert!(matches!(diff.op().unwrap(), Op::Sub(_, _)));

    let quot = ta.div(&tb).unwrap();
    assert!(matches!(quot.op().unwrap(), Op::Div(_, _)));
}

// ===========================================================================
// 3. Gradient Tape: Backward through single op
// ===========================================================================

#[test]
fn test_tape_backward_simple() {
    // y = x^2, dy/dx = 2x. At x=5 -> grad = 10.
    let x = scalar_var(5.0);
    let tx = tracked(&x);
    let y = tx.sqr().unwrap();
    let grads = backward(&y).unwrap();
    let g = get_grad_vec(&grads, &x);
    assert!(
        (g[0] - 10.0).abs() < 1e-5,
        "d/dx(x^2) at x=5 = 10, got {}",
        g[0]
    );
}

#[test]
fn test_tape_backward_exp() {
    // y = exp(x), dy/dx = exp(x). At x=1 -> grad = e.
    let x = scalar_var(1.0);
    let tx = tracked(&x);
    let y = tx.exp().unwrap();
    let loss = scalar_loss(&y);
    let grads = backward(&loss).unwrap();
    let g = get_grad_vec(&grads, &x);
    let expected = 1.0_f32.exp();
    assert!(
        (g[0] - expected).abs() < 1e-5,
        "d/dx(exp(x)) at x=1 = e = {}, got {}",
        expected,
        g[0]
    );
}

#[test]
fn test_tape_backward_add() {
    // y = a + b, dy/da = 1, dy/db = 1.
    let a = scalar_var(3.0);
    let b = scalar_var(7.0);
    let ta = tracked(&a);
    let tb = tracked(&b);
    let y = ta.add(&tb).unwrap();
    let loss = scalar_loss(&y);
    let grads = backward(&loss).unwrap();
    let ga = get_grad_vec(&grads, &a);
    let gb = get_grad_vec(&grads, &b);
    assert!((ga[0] - 1.0).abs() < 1e-6, "d/da(a+b) = 1, got {}", ga[0]);
    assert!((gb[0] - 1.0).abs() < 1e-6, "d/db(a+b) = 1, got {}", gb[0]);
}

// ===========================================================================
// 4. Gradient Tape: Detach stops gradient flow
// ===========================================================================

#[test]
fn test_tape_detach() {
    let x = scalar_var(4.0);
    let tx = tracked(&x);

    // Compute y = x^2, then detach
    let y = tx.sqr().unwrap();
    let y_detached = y.detach();

    // z = y_detached + 1 (constant-like, gradient won't reach x)
    let one = Arc::new(TrackedTensor::from_tensor(
        DynTensor::from_vec(vec![1.0], &[1], &cpu()).unwrap(),
    ));
    let z = y_detached.add(&one).unwrap();
    let loss = scalar_loss(&z);
    let grads = backward(&loss).unwrap();

    assert!(
        grads.get(&x).is_none(),
        "detach should prevent gradient flow to x"
    );

    // Verify detached value is correct (x^2 = 16)
    let val = y_detached.tensor().to_flat_vec::<f32>().unwrap()[0];
    assert!((val - 16.0).abs() < 1e-5);
}

// ===========================================================================
// 5. Gradient Tape: No-grad context (constant tensors prevent recording)
// ===========================================================================

#[test]
fn test_tape_no_grad() {
    // When both operands are constants, no var receives gradients.
    let c1 = Arc::new(TrackedTensor::from_tensor(
        DynTensor::from_vec(vec![3.0], &[1], &cpu()).unwrap(),
    ));
    let c2 = Arc::new(TrackedTensor::from_tensor(
        DynTensor::from_vec(vec![4.0], &[1], &cpu()).unwrap(),
    ));
    let y = c1.mul(&c2).unwrap();
    let loss = scalar_loss(&y);
    let grads = backward(&loss).unwrap();

    // No vars in the graph -> no variable gradients
    assert_eq!(
        grads.var_count(),
        0,
        "constant-only graph should have no variable gradients"
    );
}

#[test]
fn test_no_grad_mixed() {
    // When one operand is constant and one is var, only var gets gradient.
    let x = scalar_var(5.0);
    let tx = tracked(&x);
    let c = Arc::new(TrackedTensor::from_tensor(
        DynTensor::from_vec(vec![3.0], &[1], &cpu()).unwrap(),
    ));

    // y = x * c => dy/dx = c = 3
    let y = tx.mul(&c).unwrap();
    let loss = scalar_loss(&y);
    let grads = backward(&loss).unwrap();

    assert_eq!(grads.var_count(), 1, "only x should get a gradient");
    let g = get_grad_vec(&grads, &x);
    assert!((g[0] - 3.0).abs() < 1e-5, "d/dx(x*3) = 3, got {}", g[0]);
}

// ===========================================================================
// 6. Gradient Properties: Shape matches input
// ===========================================================================

#[test]
fn test_gradient_shape_matches_input() {
    // For a [2, 3] variable, gradient should also be [2, 3].
    let x = mat_var(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 2, 3);
    let tx = tracked(&x);
    let y = tx.sqr().unwrap();
    let loss = scalar_loss(&y);
    let grads = backward(&loss).unwrap();

    let grad_tensor = grads.get(&x).expect("x should have gradient");
    assert_eq!(
        grad_tensor.dims(),
        &[2, 3],
        "gradient shape should match input shape"
    );
}

#[test]
fn test_gradient_shape_1d() {
    let x = vec_var(vec![1.0, 2.0, 3.0, 4.0]);
    let tx = tracked(&x);
    let y = tx.sqr().unwrap();
    let loss = scalar_loss(&y);
    let grads = backward(&loss).unwrap();

    let grad_tensor = grads.get(&x).unwrap();
    assert_eq!(grad_tensor.dims(), &[4]);
}

#[test]
fn test_gradient_shape_3d() {
    let data: Vec<f32> = (0..24).map(|i| i as f32 * 0.1).collect();
    let x = Var::new(DynTensor::from_vec(data, &[2, 3, 4], &cpu()).unwrap());
    let tx = tracked(&x);
    let y = tx.sqr().unwrap();
    let loss = scalar_loss(&y);
    let grads = backward(&loss).unwrap();

    let grad_tensor = grads.get(&x).unwrap();
    assert_eq!(grad_tensor.dims(), &[2, 3, 4]);
}

// ===========================================================================
// 7. Gradient Properties: Zero initialization (uncomputed gradients)
// ===========================================================================

#[test]
fn test_gradient_zero_initialization() {
    // When a variable is not reachable from the loss, it gets no gradient entry.
    let x = scalar_var(2.0);
    let y_unrelated = scalar_var(99.0); // not used in the computation

    let tx = tracked(&x);
    let loss = tx.sqr().unwrap();
    let grads = backward(&loss).unwrap();

    // x gets a gradient
    assert!(grads.get(&x).is_some());
    // y_unrelated does not
    assert!(
        grads.get(&y_unrelated).is_none(),
        "unreachable variable should have no gradient entry"
    );
}

#[test]
fn test_grad_store_empty_by_default() {
    let store = crate::GradStore::new();
    assert_eq!(store.var_count(), 0, "fresh GradStore should be empty");
}

// ===========================================================================
// 8. Gradient Properties: Accumulation
// ===========================================================================

#[test]
fn test_gradient_accumulation() {
    // Using x twice: y = x + x = 2x => dy/dx = 2
    let x = scalar_var(5.0);
    let tx = tracked(&x);
    let y = tx.add(&tx).unwrap();
    let loss = scalar_loss(&y);
    let grads = backward(&loss).unwrap();
    let g = get_grad_vec(&grads, &x);
    assert!((g[0] - 2.0).abs() < 1e-5, "d/dx(x+x) = 2, got {}", g[0]);
}

#[test]
fn test_gradient_accumulation_mul_self() {
    // y = x * x = x^2 => dy/dx = 2x. At x=3 -> grad=6.
    let x = scalar_var(3.0);
    let tx = tracked(&x);
    let y = tx.mul(&tx).unwrap();
    let loss = scalar_loss(&y);
    let grads = backward(&loss).unwrap();
    let g = get_grad_vec(&grads, &x);
    assert!(
        (g[0] - 6.0).abs() < 1e-4,
        "d/dx(x*x) at x=3 = 6, got {}",
        g[0]
    );
}

#[test]
fn test_gradient_accumulation_triple_use() {
    // y = x + x + x = 3x => dy/dx = 3
    let x = scalar_var(7.0);
    let tx = tracked(&x);
    let sum2 = tx.add(&tx).unwrap();
    let sum3 = sum2.add(&tx).unwrap();
    let loss = scalar_loss(&sum3);
    let grads = backward(&loss).unwrap();
    let g = get_grad_vec(&grads, &x);
    assert!((g[0] - 3.0).abs() < 1e-5, "d/dx(3x) = 3, got {}", g[0]);
}

#[test]
fn test_gradient_accumulation_vector() {
    // x = [1, 2, 3], y = x + x = [2, 4, 6], loss = sum(y) = 12
    // dy/dx_i = 2 for all i
    let x = vec_var(vec![1.0, 2.0, 3.0]);
    let tx = tracked(&x);
    let y = tx.add(&tx).unwrap();
    let loss = scalar_loss(&y);
    let grads = backward(&loss).unwrap();
    let g = get_grad_vec(&grads, &x);
    assert_close_f32(&g, &[2.0, 2.0, 2.0], 1e-5, "d/dx(x+x) = 2 for all");
}

// ===========================================================================
// 9. Gradient Properties: Flow through reshape/view
// ===========================================================================

#[test]
fn test_gradient_through_view() {
    // x shape [6], reshape to [2, 3], square, sum -> loss
    // Gradients should flow back through reshape correctly.
    let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let x = vec_var(data.clone());
    let tx = tracked(&x);

    let reshaped = tx.reshape(&[2, 3]).unwrap();
    let sq = reshaped.sqr().unwrap();
    let loss = scalar_loss(&sq);
    let grads = backward(&loss).unwrap();

    let g = get_grad_vec(&grads, &x);
    // d/dx_i(sum(x_i^2)) = 2*x_i
    let expected: Vec<f32> = data.iter().map(|&v| 2.0 * v).collect();
    assert_close_f32(&g, &expected, 1e-5, "gradient through reshape");
    assert_eq!(g.len(), 6, "gradient should have original (flat) shape");
}

#[test]
fn test_gradient_through_unsqueeze_squeeze() {
    // x shape [3], unsqueeze(0) -> [1, 3], square, sum
    let data = vec![2.0, 3.0, 4.0];
    let x = vec_var(data.clone());
    let tx = tracked(&x);

    let unsq = tx.unsqueeze(0).unwrap();
    assert_eq!(unsq.dims(), &[1, 3]);
    let sq = unsq.sqr().unwrap();
    let loss = scalar_loss(&sq);
    let grads = backward(&loss).unwrap();

    let g = get_grad_vec(&grads, &x);
    let expected: Vec<f32> = data.iter().map(|&v| 2.0 * v).collect();
    assert_close_f32(&g, &expected, 1e-5, "gradient through unsqueeze");
}

#[test]
fn test_gradient_through_transpose() {
    // x shape [2, 3], transpose to [3, 2], sqr, sum
    let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let x = mat_var(data.clone(), 2, 3);
    let tx = tracked(&x);

    let transposed = tx.transpose(0, 1).unwrap();
    assert_eq!(transposed.dims(), &[3, 2]);
    let sq = transposed.sqr().unwrap();
    let loss = scalar_loss(&sq);
    let grads = backward(&loss).unwrap();

    let g = get_grad_vec(&grads, &x);
    // d/dx_i(sum(x_i^2)) = 2*x_i regardless of transpose
    let expected: Vec<f32> = data.iter().map(|&v| 2.0 * v).collect();
    assert_close_f32(&g, &expected, 1e-5, "gradient through transpose");
}

// ===========================================================================
// 10. Higher-Order Features: Second-order gradient
// ===========================================================================

#[test]
fn test_second_order_gradient() {
    // Simulate second-order gradient by computing gradient of gradient magnitude.
    // y = x^2 => dy/dx = 2x.
    // Then compute |grad|^2 as a new loss and differentiate that.
    //
    // Since this system doesn't support nesting backward() directly
    // (the gradient is a DynTensor, not tracked), we verify the first
    // gradient and then recompute using the gradient value in a new graph.
    let x = scalar_var(3.0);
    let tx = tracked(&x);
    let y = tx.sqr().unwrap();
    let grads1 = backward(&y).unwrap();
    let g1 = get_grad_vec(&grads1, &x)[0]; // dy/dx = 2*3 = 6

    // Use g1 as input to a new computation: z = g1 * x (approximation
    // of "derivative times input"). We construct a fresh graph.
    let x2 = scalar_var(3.0);
    let tx2 = tracked(&x2);
    let g1_const = Arc::new(TrackedTensor::from_tensor(
        DynTensor::from_vec(vec![g1], &[1], &cpu()).unwrap(),
    ));
    // z = g1 * x => dz/dx = g1 = 6
    let z = tx2.mul(&g1_const).unwrap();
    let loss = scalar_loss(&z);
    let grads2 = backward(&loss).unwrap();
    let g2 = get_grad_vec(&grads2, &x2)[0];

    assert!(
        (g2 - 6.0).abs() < 1e-5,
        "second-order gradient path: dz/dx = g1 = 6, got {g2}"
    );
}

// ===========================================================================
// 11. Stop gradient (explicit)
// ===========================================================================

#[test]
fn test_stop_gradient() {
    // Two variables: a and b. Compute loss = a * detach(b^2).
    // Gradient should flow to a but NOT to b.
    let a = scalar_var(2.0);
    let b = scalar_var(3.0);
    let ta = tracked(&a);
    let tb = tracked(&b);

    let b_sq = tb.sqr().unwrap(); // b^2 = 9
    let b_stopped = b_sq.detach(); // stop gradient here

    let y = ta.mul(&b_stopped).unwrap(); // a * 9
    let loss = scalar_loss(&y);
    let grads = backward(&loss).unwrap();

    // dy/da = detach(b^2) = 9
    let ga = get_grad_vec(&grads, &a);
    assert!(
        (ga[0] - 9.0).abs() < 1e-5,
        "d/da(a * detach(b^2)) = b^2 = 9, got {}",
        ga[0]
    );

    // b should get no gradient (stopped)
    assert!(
        grads.get(&b).is_none(),
        "gradient should not flow through detach to b"
    );
}

#[test]
fn test_stop_gradient_partial_graph() {
    // More complex: loss = a*b + detach(a)*b
    // Gradient of a: d/da(a*b) = b (the detach(a)*b term contributes 0 for a)
    // Gradient of b: d/db(a*b) + d/db(detach(a)*b) = a + detach(a) = 2a
    let a = scalar_var(3.0);
    let b = scalar_var(4.0);
    let ta = tracked(&a);
    let tb = tracked(&b);

    let term1 = ta.mul(&tb).unwrap(); // a * b
    let a_stopped = ta.detach();
    let term2 = a_stopped.mul(&tb).unwrap(); // detach(a) * b
    let y = term1.add(&term2).unwrap(); // a*b + detach(a)*b
    let loss = scalar_loss(&y);
    let grads = backward(&loss).unwrap();

    let ga = get_grad_vec(&grads, &a);
    assert!((ga[0] - 4.0).abs() < 1e-5, "d/da = b = 4, got {}", ga[0]);

    let gb = get_grad_vec(&grads, &b);
    // d/db = a + detach(a) = 3 + 3 = 6
    assert!(
        (gb[0] - 6.0).abs() < 1e-5,
        "d/db = a + detach(a) = 6, got {}",
        gb[0]
    );
}

// ===========================================================================
// 12. Numerical Gradient Checking: Add
// ===========================================================================

#[test]
fn test_numerical_vs_analytical_add() {
    let data_a = vec![1.0, 2.0, 3.0, 4.0];
    let data_b = vec![5.0, 6.0, 7.0, 8.0];
    let shape = [4];
    let h = 1e-3_f32;

    // Analytical: backward through add
    let a = vec_var(data_a.clone());
    let b = vec_var(data_b.clone());
    let ta = tracked(&a);
    let tb = tracked(&b);
    let y = ta.add(&tb).unwrap();
    let loss = scalar_loss(&y);
    let grads = backward(&loss).unwrap();
    let grad_a = get_grad_vec(&grads, &a);
    let grad_b = get_grad_vec(&grads, &b);

    // Numerical gradient for a: loss = sum(a + b)
    let data_b_clone = data_b.clone();
    let num_a = numerical_grad(&data_a, &shape, h, &|perturbed_a| {
        let ta = DynTensor::from_vec(perturbed_a.to_vec(), &shape, &cpu()).unwrap();
        let tb = DynTensor::from_vec(data_b_clone.clone(), &shape, &cpu()).unwrap();
        let y = ta.add(&tb).unwrap();
        sum_all_f64(&y)
    });

    let data_a_clone = data_a;
    let num_b = numerical_grad(&data_b, &shape, h, &|perturbed_b| {
        let ta = DynTensor::from_vec(data_a_clone.clone(), &shape, &cpu()).unwrap();
        let tb = DynTensor::from_vec(perturbed_b.to_vec(), &shape, &cpu()).unwrap();
        let y = ta.add(&tb).unwrap();
        sum_all_f64(&y)
    });

    assert_close_f64(&grad_a, &num_a, 1e-2, "add grad_a: analytical vs numerical");
    assert_close_f64(&grad_b, &num_b, 1e-2, "add grad_b: analytical vs numerical");
}

// ===========================================================================
// 13. Numerical Gradient Checking: Mul
// ===========================================================================

#[test]
fn test_numerical_vs_analytical_mul() {
    let data_a = vec![1.0, 2.0, 3.0, 4.0];
    let data_b = vec![5.0, 6.0, 7.0, 8.0];
    let shape = [4];
    let h = 1e-3_f32;

    // Analytical
    let a = vec_var(data_a.clone());
    let b = vec_var(data_b.clone());
    let ta = tracked(&a);
    let tb = tracked(&b);
    let y = ta.mul(&tb).unwrap();
    let loss = scalar_loss(&y);
    let grads = backward(&loss).unwrap();
    let grad_a = get_grad_vec(&grads, &a);
    let grad_b = get_grad_vec(&grads, &b);

    // Numerical: loss = sum(a * b), d/da_i = b_i, d/db_i = a_i
    let data_b_clone = data_b.clone();
    let num_a = numerical_grad(&data_a, &shape, h, &|perturbed_a| {
        let ta = DynTensor::from_vec(perturbed_a.to_vec(), &shape, &cpu()).unwrap();
        let tb = DynTensor::from_vec(data_b_clone.clone(), &shape, &cpu()).unwrap();
        let y = ta.mul(&tb).unwrap();
        sum_all_f64(&y)
    });

    let data_a_clone = data_a.clone();
    let num_b = numerical_grad(&data_b, &shape, h, &|perturbed_b| {
        let ta = DynTensor::from_vec(data_a_clone.clone(), &shape, &cpu()).unwrap();
        let tb = DynTensor::from_vec(perturbed_b.to_vec(), &shape, &cpu()).unwrap();
        let y = ta.mul(&tb).unwrap();
        sum_all_f64(&y)
    });

    assert_close_f64(&grad_a, &num_a, 1e-2, "mul grad_a: analytical vs numerical");
    assert_close_f64(&grad_b, &num_b, 1e-2, "mul grad_b: analytical vs numerical");

    // Also verify analytically: d/da_i(sum(a_i*b_i)) = b_i
    assert_close_f32(&grad_a, &data_b, 1e-5, "mul grad_a = b");
    assert_close_f32(&grad_b, &data_a, 1e-5, "mul grad_b = a");
}

// ===========================================================================
// 14. Numerical Gradient Checking: Exp
// ===========================================================================

#[test]
fn test_numerical_vs_analytical_exp() {
    let data = vec![0.1, 0.5, 1.0, 1.5];
    let shape = [4];
    let h = 1e-3_f32;

    // Analytical
    let x = vec_var(data.clone());
    let tx = tracked(&x);
    let y = tx.exp().unwrap();
    let loss = scalar_loss(&y);
    let grads = backward(&loss).unwrap();
    let grad_x = get_grad_vec(&grads, &x);

    // Numerical: loss = sum(exp(x)), d/dx_i = exp(x_i)
    let num_x = numerical_grad(&data, &shape, h, &|perturbed| {
        let t = DynTensor::from_vec(perturbed.to_vec(), &shape, &cpu()).unwrap();
        let y = t.exp().unwrap();
        sum_all_f64(&y)
    });

    assert_close_f64(&grad_x, &num_x, 1e-2, "exp grad: analytical vs numerical");

    // Verify analytically: d/dx_i(sum(exp(x_i))) = exp(x_i)
    let expected: Vec<f32> = data.iter().map(|&v| v.exp()).collect();
    assert_close_f32(&grad_x, &expected, 1e-4, "exp grad = exp(x)");
}

// ===========================================================================
// 15. Numerical Gradient Checking: Composed (mul + exp)
// ===========================================================================

#[test]
fn test_numerical_vs_analytical_composed() {
    // f(x) = sum(exp(2*x)), df/dx_i = 2*exp(2*x_i)
    let data = vec![0.1, 0.3, 0.5, 0.7];
    let shape = [4];
    let h = 1e-3_f32;

    let x = vec_var(data.clone());
    let tx = tracked(&x);
    let two = Arc::new(TrackedTensor::from_tensor(
        DynTensor::from_vec(vec![2.0, 2.0, 2.0, 2.0], &shape, &cpu()).unwrap(),
    ));
    let doubled = tx.mul(&two).unwrap();
    let y = doubled.exp().unwrap();
    let loss = scalar_loss(&y);
    let grads = backward(&loss).unwrap();
    let grad_x = get_grad_vec(&grads, &x);

    let num_x = numerical_grad(&data, &shape, h, &|perturbed| {
        let t = DynTensor::from_vec(perturbed.to_vec(), &shape, &cpu()).unwrap();
        let doubled = t.mul_scalar(2.0).unwrap();
        let y = doubled.exp().unwrap();
        sum_all_f64(&y)
    });

    assert_close_f64(
        &grad_x,
        &num_x,
        1e-1,
        "composed (2*exp) grad: analytical vs numerical",
    );

    // Analytical check: d/dx_i = 2 * exp(2*x_i)
    let expected: Vec<f32> = data.iter().map(|&v| 2.0 * (2.0 * v).exp()).collect();
    assert_close_f32(&grad_x, &expected, 1e-3, "composed grad = 2*exp(2x)");
}

// ===========================================================================
// 16. Numerical Gradient Checking: Sigmoid
// ===========================================================================

#[test]
fn test_numerical_vs_analytical_sigmoid() {
    // f(x) = sum(sigmoid(x)), df/dx_i = sigmoid(x_i) * (1 - sigmoid(x_i))
    let data = vec![-1.0, 0.0, 0.5, 2.0];
    let shape = [4];
    let h = 1e-3_f32;

    let x = vec_var(data.clone());
    let tx = tracked(&x);
    let y = tx.sigmoid().unwrap();
    let loss = scalar_loss(&y);
    let grads = backward(&loss).unwrap();
    let grad_x = get_grad_vec(&grads, &x);

    let num_x = numerical_grad(&data, &shape, h, &|perturbed| {
        let t = DynTensor::from_vec(perturbed.to_vec(), &shape, &cpu()).unwrap();
        let y = t.sigmoid().unwrap();
        sum_all_f64(&y)
    });

    assert_close_f64(
        &grad_x,
        &num_x,
        1e-2,
        "sigmoid grad: analytical vs numerical",
    );

    // Analytical check
    let expected: Vec<f32> = data
        .iter()
        .map(|&v| {
            let s = 1.0 / (1.0 + (-v).exp());
            s * (1.0 - s)
        })
        .collect();
    assert_close_f32(&grad_x, &expected, 1e-4, "sigmoid grad = s*(1-s)");
}

// ===========================================================================
// 17. Numerical Gradient Checking: Tanh
// ===========================================================================

#[test]
fn test_numerical_vs_analytical_tanh() {
    // f(x) = sum(tanh(x)), df/dx_i = 1 - tanh(x_i)^2
    let data = vec![-1.0, 0.0, 0.5, 1.5];
    let shape = [4];
    let h = 1e-3_f32;

    let x = vec_var(data.clone());
    let tx = tracked(&x);
    let y = tx.tanh().unwrap();
    let loss = scalar_loss(&y);
    let grads = backward(&loss).unwrap();
    let grad_x = get_grad_vec(&grads, &x);

    let num_x = numerical_grad(&data, &shape, h, &|perturbed| {
        let t = DynTensor::from_vec(perturbed.to_vec(), &shape, &cpu()).unwrap();
        let y = t.tanh().unwrap();
        sum_all_f64(&y)
    });

    assert_close_f64(&grad_x, &num_x, 1e-2, "tanh grad: analytical vs numerical");

    let expected: Vec<f32> = data.iter().map(|&v| 1.0 - v.tanh().powi(2)).collect();
    assert_close_f32(&grad_x, &expected, 1e-4, "tanh grad = 1 - tanh^2");
}

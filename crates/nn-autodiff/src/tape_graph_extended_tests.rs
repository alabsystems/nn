// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Extended tests for gradient tape recording, backward pass execution,
//! graph topology, memory management, mixed operations, gradient accumulation,
//! no-grad scope, and DOT graph visualization.
//!
//! Covers:
//! 1. Tape recording -- verify forward ops are recorded correctly
//! 2. Backward pass execution -- verify gradients flow correctly
//! 3. Graph topology -- diamond patterns, chain rule through multiple ops
//! 4. Memory management -- tape clears after backward, no gradient leaks
//! 5. Mixed operations -- add+mul+relu+matmul+softmax chains
//! 6. Gradient accumulation -- multiple backward passes
//! 7. No-grad scope -- constant tensors prevent gradient recording
//! 8. Graph visualization DOT export -- valid DOT format

use std::sync::Arc;

use nn_core::dyn_tensor::DynTensor;
use nn_core::test_utils::cpu;
use nn_core::Device;

use crate::grad::{backward, backward_for_vars};
use crate::graph_viz;
use crate::op::Op;
use crate::tracked::TrackedTensor;
use crate::var::Var;
use crate::GradStore;

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

fn get_grad_vec(grads: &GradStore, var: &Var) -> Vec<f32> {
    grads
        .get(var)
        .expect("gradient should exist")
        .to_flat_vec::<f32>()
        .unwrap()
}

fn assert_approx(actual: &[f32], expected: &[f32], tol: f32) {
    assert_eq!(
        actual.len(),
        expected.len(),
        "length mismatch: {} vs {}",
        actual.len(),
        expected.len()
    );
    for (i, (a, e)) in actual.iter().zip(expected).enumerate() {
        assert!(
            (a - e).abs() < tol,
            "index {i}: actual={a}, expected={e}, diff={}",
            (a - e).abs()
        );
    }
}

/// Reduce an arbitrary-shaped tracked tensor to a scalar via sum over all dims.
fn scalar_loss(t: &Arc<TrackedTensor>) -> Arc<TrackedTensor> {
    let mut result = Arc::clone(t);
    for d in (0..result.tensor().rank()).rev() {
        result = result.sum_keepdim(d).unwrap();
    }
    result
}

// ===========================================================================
// 1. Tape Recording -- verify forward ops are recorded correctly
// ===========================================================================

#[test]
fn test_tape_records_unary_chain() {
    // x -> neg -> abs -> exp -> output
    let x = scalar_var(3.0);
    let tx = tracked(&x);

    let neg = tx.neg().unwrap();
    assert!(matches!(neg.op().unwrap(), Op::Neg(_)));

    let absd = neg.abs().unwrap();
    assert!(matches!(absd.op().unwrap(), Op::Abs(_)));

    let out = absd.exp().unwrap();
    assert!(matches!(out.op().unwrap(), Op::Exp(_)));

    // Verify the chain is intact by walking backward through ops
    if let Op::Exp(inner) = out.op().unwrap() {
        assert!(matches!(inner.op().unwrap(), Op::Abs(_)));
        if let Op::Abs(deeper) = inner.op().unwrap() {
            assert!(matches!(deeper.op().unwrap(), Op::Neg(_)));
        } else {
            panic!("expected Abs op in chain");
        }
    } else {
        panic!("expected Exp op at top of chain");
    }
}

#[test]
fn test_tape_records_scalar_ops() {
    // x -> mul_scalar(3.0) -> add_scalar(1.0)
    let x = scalar_var(2.0);
    let tx = tracked(&x);

    let scaled = tx.mul_scalar(3.0).unwrap();
    assert!(matches!(scaled.op().unwrap(), Op::MulScalar(_, _)));

    let shifted = scaled.add_scalar(1.0).unwrap();
    assert!(matches!(shifted.op().unwrap(), Op::AddScalar(_, _)));
}

#[test]
fn test_tape_records_reduction_ops() {
    let x = vec_var(vec![1.0, 2.0, 3.0]);
    let tx = tracked(&x);

    let summed = tx.sum_keepdim(0).unwrap();
    assert!(matches!(summed.op().unwrap(), Op::SumKeepDim(_, 0)));

    // Build a fresh graph for mean
    let tx2 = tracked(&x);
    let meaned = tx2.mean_keepdim(0).unwrap();
    assert!(matches!(meaned.op().unwrap(), Op::MeanKeepDim(_, 0)));
}

#[test]
fn test_tape_records_shape_ops() {
    let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let x = vec_var(data);
    let tx = tracked(&x);

    let reshaped = tx.reshape(&[2, 3]).unwrap();
    assert!(matches!(reshaped.op().unwrap(), Op::Reshape(_, _)));
    assert_eq!(reshaped.dims(), &[2, 3]);

    let transposed = reshaped.transpose(0, 1).unwrap();
    assert!(matches!(transposed.op().unwrap(), Op::Transpose(_, 0, 1)));
    assert_eq!(transposed.dims(), &[3, 2]);
}

// ===========================================================================
// 2. Backward Pass -- verify gradients flow correctly
// ===========================================================================

#[test]
fn test_backward_through_long_chain() {
    // Chain: x -> sqr -> add_scalar(1) -> sqrt -> neg -> sum
    // f(x) = -sqrt(x^2 + 1) at x=3
    // f'(x) = -x / sqrt(x^2 + 1) = -3 / sqrt(10)
    let x = scalar_var(3.0);
    let tx = tracked(&x);
    let sq = tx.sqr().unwrap();
    let shifted = sq.add_scalar(1.0).unwrap();
    let rooted = shifted.sqrt().unwrap();
    let out = rooted.neg().unwrap();
    let grads = backward(&out).unwrap();

    let expected = -3.0_f32 / (10.0_f32).sqrt();
    assert_approx(&get_grad_vec(&grads, &x), &[expected], 1e-4);
}

#[test]
fn test_backward_sigmoid_chain() {
    // f(x) = sigmoid(2x) at x=0
    // f'(x) = 2 * sigmoid(2x) * (1 - sigmoid(2x)) = 2 * 0.5 * 0.5 = 0.5
    let x = scalar_var(0.0);
    let tx = tracked(&x);
    let doubled = tx.mul_scalar(2.0).unwrap();
    let sig = doubled.sigmoid().unwrap();
    let grads = backward(&sig).unwrap();

    assert_approx(&get_grad_vec(&grads, &x), &[0.5], 1e-4);
}

#[test]
fn test_backward_relu_positive() {
    // f(x) = relu(x)^2 at x=3
    // f'(x) = 2*relu(x) * d_relu = 2*3*1 = 6
    let x = scalar_var(3.0);
    let tx = tracked(&x);
    let r = tx.relu().unwrap();
    let sq = r.sqr().unwrap();
    let grads = backward(&sq).unwrap();

    assert_approx(&get_grad_vec(&grads, &x), &[6.0], 1e-5);
}

#[test]
fn test_backward_relu_negative() {
    // f(x) = relu(x)^2 at x=-3
    // relu(-3) = 0, so f(-3)=0, f'(-3)=0
    let x = scalar_var(-3.0);
    let tx = tracked(&x);
    let r = tx.relu().unwrap();
    let sq = r.sqr().unwrap();
    let grads = backward(&sq).unwrap();

    assert_approx(&get_grad_vec(&grads, &x), &[0.0], 1e-5);
}

#[test]
fn test_backward_vector_element_wise() {
    // x = [1, 2, 3], f(x) = sum(x^2) = 14
    // df/dx_i = 2*x_i = [2, 4, 6]
    let x = vec_var(vec![1.0, 2.0, 3.0]);
    let tx = tracked(&x);
    let sq = tx.sqr().unwrap();
    let loss = scalar_loss(&sq);
    let grads = backward(&loss).unwrap();

    assert_approx(&get_grad_vec(&grads, &x), &[2.0, 4.0, 6.0], 1e-5);
}

// ===========================================================================
// 3. Graph Topology -- diamond patterns, chain rule through multiple ops
// ===========================================================================

#[test]
fn test_diamond_shared_intermediate() {
    // Diamond pattern with a shared intermediate:
    //     x
    //    / \
    //  x^2  x^3
    //    \ /
    //   sum
    // d/dx(x^2 + x^3) = 2x + 3x^2, at x=2 => 4 + 12 = 16
    let x = scalar_var(2.0);
    let tx = tracked(&x);
    let sq = tx.sqr().unwrap(); // x^2
    let cube = sq.mul(&tx).unwrap(); // x^2 * x = x^3
    let sum = sq.add(&cube).unwrap(); // x^2 + x^3
    let grads = backward(&sum).unwrap();

    assert_approx(&get_grad_vec(&grads, &x), &[16.0], 1e-4);
}

#[test]
fn test_diamond_two_variables() {
    // Diamond with two variables meeting:
    //   a     b
    //   |     |
    //  a^2   b^2
    //    \  /
    //    a^2 * b^2
    // d/da(a^2 * b^2) = 2a * b^2, at a=3, b=2 => 2*3*4 = 24
    // d/db(a^2 * b^2) = a^2 * 2b, at a=3, b=2 => 9*4 = 36
    let a = scalar_var(3.0);
    let b = scalar_var(2.0);
    let ta = tracked(&a);
    let tb = tracked(&b);
    let a_sq = ta.sqr().unwrap();
    let b_sq = tb.sqr().unwrap();
    let prod = a_sq.mul(&b_sq).unwrap();
    let grads = backward(&prod).unwrap();

    assert_approx(&get_grad_vec(&grads, &a), &[24.0], 1e-3);
    assert_approx(&get_grad_vec(&grads, &b), &[36.0], 1e-3);
}

#[test]
fn test_wide_diamond_four_paths() {
    // x branches into 4 paths, all merged:
    // z = x + x^2 + 2*x + 3*x
    // z = 6x + x^2
    // dz/dx = 6 + 2x, at x=5 => 16
    let x = scalar_var(5.0);
    let tx = tracked(&x);
    let sq = tx.sqr().unwrap();
    let double = tx.mul_scalar(2.0).unwrap();
    let triple = tx.mul_scalar(3.0).unwrap();

    let s1 = tx.add(&sq).unwrap(); // x + x^2
    let s2 = s1.add(&double).unwrap(); // x + x^2 + 2x
    let z = s2.add(&triple).unwrap(); // x + x^2 + 2x + 3x = x^2 + 6x

    let grads = backward(&z).unwrap();
    assert_approx(&get_grad_vec(&grads, &x), &[16.0], 1e-4);
}

#[test]
fn test_chain_rule_five_ops() {
    // f(x) = exp(sin(x^2))
    // f'(x) = exp(sin(x^2)) * cos(x^2) * 2x
    // at x = 0.5:
    //   x^2 = 0.25
    //   sin(0.25) ~= 0.2474
    //   cos(0.25) ~= 0.9689
    //   exp(sin(0.25)) ~= 1.2808
    //   f'(0.5) = 1.2808 * 0.9689 * 1.0 ~= 1.2409
    let x = scalar_var(0.5);
    let tx = tracked(&x);
    let sq = tx.sqr().unwrap();
    let s = sq.sin().unwrap();
    let out = s.exp().unwrap();
    let grads = backward(&out).unwrap();

    let x_val = 0.5_f32;
    let expected = (x_val * x_val).sin().exp() * (x_val * x_val).cos() * 2.0 * x_val;
    assert_approx(&get_grad_vec(&grads, &x), &[expected], 1e-3);
}

// ===========================================================================
// 4. Memory Management -- tape clears after backward, no gradient leaks
// ===========================================================================

#[test]
fn test_graph_drops_after_backward() {
    let x = scalar_var(2.0);
    let tx = tracked(&x);
    let sq = tx.sqr().unwrap();
    let loss = scalar_loss(&sq);

    let grads = backward(&loss).unwrap();
    let g = get_grad_vec(&grads, &x);
    assert!((g[0] - 4.0).abs() < 1e-5, "d/dx(x^2) at x=2 = 4");

    // Drop the entire computation graph
    drop(loss);
    drop(sq);

    // tx should be the only remaining reference
    assert_eq!(
        Arc::strong_count(&tx),
        1,
        "after dropping graph, only our reference should remain"
    );
}

#[test]
fn test_into_tensor_drops_graph() {
    let x = scalar_var(3.0);
    let tx = tracked(&x);
    let sq = tx.sqr().unwrap();

    // Extract data via into_tensor, which drops the op graph
    let data = Arc::try_unwrap(sq).unwrap().into_tensor().unwrap();
    let vals = data.to_flat_vec::<f32>().unwrap();
    assert_approx(&vals, &[9.0], 1e-5);

    // tx should have only one reference now
    assert_eq!(Arc::strong_count(&tx), 1);
}

#[test]
fn test_no_gradient_accumulation_leak_across_iterations() {
    // Multiple training iterations on same variable -- each backward creates
    // independent GradStores, so no leaks accumulate.
    let w = scalar_var(5.0);

    for _ in 0..3 {
        let tw = tracked(&w);
        let loss = tw.sqr().unwrap();
        let grads = backward(&loss).unwrap();
        let g = get_grad_vec(&grads, &w);
        // Each iteration should give gradient = 2*w (current value)
        let w_val = w.data().unwrap().to_flat_vec::<f32>().unwrap()[0];
        assert!(
            (g[0] - 2.0 * w_val).abs() < 1e-4,
            "gradient should be 2*w = {}, got {}",
            2.0 * w_val,
            g[0]
        );
        // Update w: SGD step
        let new_w = w
            .data()
            .unwrap()
            .sub(&grads.get(&w).unwrap().mul_scalar(0.1).unwrap())
            .unwrap();
        w.set(&new_w).unwrap();
    }

    // After 3 iterations, w should have decreased from 5.0
    let w_final = w.data().unwrap().to_flat_vec::<f32>().unwrap()[0];
    assert!(
        w_final < 5.0,
        "w should decrease after SGD steps, got {w_final}"
    );
}

#[test]
fn test_gradstore_var_count_correct() {
    let a = scalar_var(1.0);
    let b = scalar_var(2.0);
    let c = scalar_var(3.0);
    let ta = tracked(&a);
    let tb = tracked(&b);
    let tc = tracked(&c);

    let sum = ta.add(&tb).unwrap();
    let loss = sum.mul(&tc).unwrap();
    let grads = backward(&loss).unwrap();

    assert_eq!(grads.var_count(), 3, "three vars should have gradients");
    assert!(grads.get(&a).is_some());
    assert!(grads.get(&b).is_some());
    assert!(grads.get(&c).is_some());
}

// ===========================================================================
// 5. Mixed Operations -- add+mul+relu+matmul+softmax chains
// ===========================================================================

#[test]
fn test_mixed_add_mul_relu() {
    // f(x, y) = relu(x * y + x), loss = sum(f)
    // At x=[1, -1], y=[2, 3]:
    //   x*y = [2, -3], x*y + x = [3, -4], relu = [3, 0]
    //   loss = 3
    let x = vec_var(vec![1.0, -1.0]);
    let y = vec_var(vec![2.0, 3.0]);
    let tx = tracked(&x);
    let ty = tracked(&y);

    let prod = tx.mul(&ty).unwrap();
    let sum = prod.add(&tx).unwrap();
    let out = sum.relu().unwrap();
    let loss = scalar_loss(&out);
    let grads = backward(&loss).unwrap();

    // Verify loss value
    let loss_val = loss.tensor().to_flat_vec::<f32>().unwrap()[0];
    assert!(
        (loss_val - 3.0).abs() < 1e-5,
        "loss should be 3, got {loss_val}"
    );

    // f = relu(x*y + x) = relu(x*(y+1))
    // df/dx_i = (y_i + 1) if x*(y+1) > 0, else 0
    // x=1, y=2: 1*(2+1)=3>0, df/dx=3
    // x=-1, y=3: -1*(3+1)=-4<0, df/dx=0
    let gx = get_grad_vec(&grads, &x);
    assert_approx(&gx, &[3.0, 0.0], 1e-4);
}

#[test]
fn test_mixed_matmul_add() {
    // z = x @ w + b, loss = sum(z)
    // x = [[1, 2]], w = [[3, 4], [5, 6]], b = [10, 20]
    // z = [[1*3+2*5, 1*4+2*6]] + [10, 20] = [[13, 16]] + [10, 20] = [[23, 36]]
    let w = mat_var(vec![3.0, 4.0, 5.0, 6.0], 2, 2);
    let b = vec_var(vec![10.0, 20.0]);

    let x_data = DynTensor::from_vec(vec![1.0, 2.0], &[1, 2], &cpu()).unwrap();
    let x = Arc::new(TrackedTensor::from_tensor(x_data));

    let tw = tracked(&w);
    let tb = Arc::new(TrackedTensor::from_var(&b).unwrap());

    let z = x.matmul(&tw).unwrap(); // [1, 2]
    let b_expanded = tb.reshape(&[1, 2]).unwrap();
    let out = z.add(&b_expanded).unwrap();
    let loss = scalar_loss(&out);
    let grads = backward(&loss).unwrap();

    // dL/db = [1, 1] (gradient of sum through add)
    let gb = get_grad_vec(&grads, &b);
    assert_approx(&gb, &[1.0, 1.0], 1e-4);

    // w should have a gradient
    assert!(grads.get(&w).is_some(), "weight should have gradient");
}

#[test]
fn test_mixed_softmax_backward() {
    // Softmax on a vector, then sum (which should be 1.0 for softmax output).
    // More meaningful: softmax then multiply by weights, then sum.
    let x = vec_var(vec![1.0, 2.0, 3.0]);
    let tx = tracked(&x);
    let sm = tx.softmax(0).unwrap();

    // Verify softmax output sums to 1
    let sm_vals = sm.tensor().to_flat_vec::<f32>().unwrap();
    let sm_sum: f32 = sm_vals.iter().sum();
    assert!(
        (sm_sum - 1.0).abs() < 1e-5,
        "softmax should sum to 1, got {sm_sum}"
    );

    // loss = sum(softmax(x) * [0, 0, 1]) = softmax(x)[2] = e^3 / (e^1 + e^2 + e^3)
    let target = Arc::new(TrackedTensor::from_tensor(
        DynTensor::from_vec(vec![0.0, 0.0, 1.0], &[3], &cpu()).unwrap(),
    ));
    let weighted = sm.mul(&target).unwrap();
    let loss = scalar_loss(&weighted);
    let grads = backward(&loss).unwrap();

    let gx = get_grad_vec(&grads, &x);
    // The gradient should exist and have 3 elements
    assert_eq!(gx.len(), 3);
    // Here the loss is L = softmax(x)[2] (NOT cross-entropy / -log).
    // The softmax Jacobian-vector product gives:
    //   dL/dx_j = softmax[2] * (delta_{2j} - softmax[j])
    // so the target element grad is POSITIVE (softmax[2]*(1-softmax[2]))
    // and the non-target grads are NEGATIVE (-softmax[2]*softmax[j]).
    assert!(
        gx[2] > 0.0,
        "target element grad should be positive (softmax[2]*(1-softmax[2])), got {}",
        gx[2]
    );
    assert!(
        gx[0] < 0.0 && gx[1] < 0.0,
        "non-target grads should be negative, got [{}, {}]",
        gx[0],
        gx[1]
    );
}

#[test]
fn test_mixed_mul_add_sigmoid_chain() {
    // f(a, b) = sigmoid(a * b + a)
    // At a=1, b=1: f = sigmoid(1*1+1) = sigmoid(2) ~= 0.8808
    // df/da = sigmoid'(a*b+a) * (b+1) = s*(1-s)*(b+1) = 0.8808*0.1192*2 ~= 0.2100
    let a = scalar_var(1.0);
    let b = scalar_var(1.0);
    let ta = tracked(&a);
    let tb = tracked(&b);
    let prod = ta.mul(&tb).unwrap();
    let sum = prod.add(&ta).unwrap();
    let out = sum.sigmoid().unwrap();
    let grads = backward(&out).unwrap();

    let s = 1.0 / (1.0 + (-2.0_f32).exp());
    let expected_da = s * (1.0 - s) * 2.0;
    assert_approx(&get_grad_vec(&grads, &a), &[expected_da], 1e-3);

    let expected_db = s * (1.0 - s) * 1.0;
    assert_approx(&get_grad_vec(&grads, &b), &[expected_db], 1e-3);
}

// ===========================================================================
// 6. Gradient Accumulation -- multiple backward passes
// ===========================================================================

#[test]
fn test_multiple_backward_same_graph_identical_grads() {
    let x = scalar_var(4.0);
    let tx = tracked(&x);
    let loss = tx.sqr().unwrap(); // d/dx(x^2) = 8

    let grads1 = backward(&loss).unwrap();
    let grads2 = backward(&loss).unwrap();

    let g1 = get_grad_vec(&grads1, &x)[0];
    let g2 = get_grad_vec(&grads2, &x)[0];

    assert!(
        (g1 - g2).abs() < 1e-7,
        "repeated backward should give identical gradients: {g1} vs {g2}"
    );
    assert!((g1 - 8.0).abs() < 1e-5);
}

#[test]
fn test_manual_gradient_accumulation_pattern() {
    // Simulate gradient accumulation over a mini-batch by running
    // backward multiple times and summing gradients manually.
    let w = scalar_var(2.0);
    let inputs = vec![1.0_f32, 3.0, 5.0];
    let mut accumulated = 0.0_f64;

    for &input_val in &inputs {
        let tw = tracked(&w);
        let input = Arc::new(TrackedTensor::from_tensor(
            DynTensor::from_vec(vec![input_val], &[1], &cpu()).unwrap(),
        ));
        let pred = tw.mul(&input).unwrap();
        let loss = pred.sqr().unwrap(); // (w*x)^2
        let grads = backward(&loss).unwrap();
        let g = get_grad_vec(&grads, &w)[0];
        accumulated += f64::from(g);
    }

    // d/dw (w*x)^2 = 2*w*x^2
    // sum for x in [1,3,5]: 2*2*1 + 2*2*9 + 2*2*25 = 4+36+100 = 140
    assert!(
        (accumulated - 140.0).abs() < 0.5,
        "accumulated gradient should be 140, got {accumulated}"
    );
}

#[test]
fn test_backward_for_vars_filters_correctly() {
    let a = scalar_var(2.0);
    let b = scalar_var(3.0);
    let c = scalar_var(4.0);
    let ta = tracked(&a);
    let tb = tracked(&b);
    let tc = tracked(&c);

    // loss = a*b + b*c
    let ab = ta.mul(&tb).unwrap();
    let bc = tb.mul(&tc).unwrap();
    let loss = ab.add(&bc).unwrap();

    // Only get gradients for a and c
    let grads = backward_for_vars(&loss, &[&a, &c]).unwrap();

    assert!(grads.get(&a).is_some(), "a should have gradient");
    assert!(grads.get(&c).is_some(), "c should have gradient");
    assert!(grads.get(&b).is_none(), "b should be filtered out");

    // dL/da = b = 3
    assert_approx(&get_grad_vec(&grads, &a), &[3.0], 1e-5);
    // dL/dc = b = 3
    assert_approx(&get_grad_vec(&grads, &c), &[3.0], 1e-5);
}

// ===========================================================================
// 7. No-Grad Scope -- constant tensors prevent gradient recording
// ===========================================================================

#[test]
fn test_no_grad_constant_only_graph() {
    // When all inputs are from_tensor (not from_var), no variable gradients.
    let c1 = Arc::new(TrackedTensor::from_tensor(
        DynTensor::from_vec(vec![2.0], &[1], &cpu()).unwrap(),
    ));
    let c2 = Arc::new(TrackedTensor::from_tensor(
        DynTensor::from_vec(vec![3.0], &[1], &cpu()).unwrap(),
    ));
    let y = c1.mul(&c2).unwrap();
    let loss = scalar_loss(&y);
    let grads = backward(&loss).unwrap();

    assert_eq!(
        grads.var_count(),
        0,
        "constant-only graph should have 0 var grads"
    );
}

#[test]
fn test_no_grad_from_tensor_does_not_record() {
    let c = Arc::new(TrackedTensor::from_tensor(
        DynTensor::from_vec(vec![5.0], &[1], &cpu()).unwrap(),
    ));

    // from_tensor should have no op and no var_id
    assert!(c.op().is_none(), "constant should have no op");
    assert!(c.var_id().is_none(), "constant should have no var_id");
    assert!(!c.is_var(), "constant should not be a var");
}

#[test]
fn test_no_grad_mixed_var_and_constant() {
    // loss = var * const + const
    // Only var should receive gradient
    let x = scalar_var(3.0);
    let tx = tracked(&x);
    let c1 = Arc::new(TrackedTensor::from_tensor(
        DynTensor::from_vec(vec![4.0], &[1], &cpu()).unwrap(),
    ));
    let c2 = Arc::new(TrackedTensor::from_tensor(
        DynTensor::from_vec(vec![10.0], &[1], &cpu()).unwrap(),
    ));

    let prod = tx.mul(&c1).unwrap();
    let out = prod.add(&c2).unwrap();
    let loss = scalar_loss(&out);
    let grads = backward(&loss).unwrap();

    assert_eq!(grads.var_count(), 1);
    // d/dx(x*4 + 10) = 4
    assert_approx(&get_grad_vec(&grads, &x), &[4.0], 1e-5);
}

#[test]
fn test_no_grad_detach_creates_constant_like() {
    // After detach, the tensor acts like a constant for subsequent ops.
    let x = scalar_var(2.0);
    let tx = tracked(&x);
    let sq = tx.sqr().unwrap();
    let detached = sq.detach();

    // Verify detached tensor acts like a constant
    assert!(detached.op().is_none());
    assert!(!detached.is_var());

    // Use detached in a new graph -- gradients stop at detach
    let y = scalar_var(3.0);
    let ty = tracked(&y);
    let loss = ty.mul(&detached).unwrap(); // y * detach(x^2) = y * 4
    let loss_scalar = scalar_loss(&loss);
    let grads = backward(&loss_scalar).unwrap();

    // y gets gradient = detach(x^2) = 4
    assert_approx(&get_grad_vec(&grads, &y), &[4.0], 1e-5);
    // x gets nothing (detached)
    assert!(grads.get(&x).is_none());
}

// ===========================================================================
// 8. Graph Visualization DOT Export -- valid DOT format
// ===========================================================================

#[test]
fn test_dot_export_basic_structure() {
    let x = scalar_var(2.0);
    let tx = tracked(&x);
    let sq = tx.sqr().unwrap();
    let out = sq.relu().unwrap();

    let dot = graph_viz::graph_to_dot(&out);

    // Must contain digraph keyword
    assert!(dot.contains("digraph"), "DOT must contain 'digraph'");
    // Must have opening and closing braces
    assert!(dot.contains('{'), "DOT must have opening brace");
    assert!(dot.contains('}'), "DOT must have closing brace");
    // Must contain node definitions
    assert!(dot.contains("label="), "DOT must contain node labels");
    // Must contain edges
    assert!(dot.contains(" -> "), "DOT must contain edges");
}

#[test]
fn test_dot_export_node_count() {
    // x -> sqr -> relu = 3 nodes
    let x = scalar_var(2.0);
    let tx = tracked(&x);
    let sq = tx.sqr().unwrap();
    let out = sq.relu().unwrap();

    let count = graph_viz::node_count(&out);
    assert_eq!(count, 3, "x -> sqr -> relu should have 3 nodes");
}

#[test]
fn test_dot_export_edge_count() {
    // x -> sqr -> relu = 2 edges
    let x = scalar_var(2.0);
    let tx = tracked(&x);
    let sq = tx.sqr().unwrap();
    let out = sq.relu().unwrap();

    let edges = graph_viz::edge_count(&out);
    assert_eq!(edges, 2, "x -> sqr -> relu should have 2 edges");
}

#[test]
fn test_dot_export_diamond_no_duplicate_nodes() {
    // Diamond: x -> sqr, x -> neg, sqr + neg -> add
    // Should have 4 unique nodes: x, sqr, neg, add
    let x = Var::new(DynTensor::from_vec(vec![3.0], &[1], &Device::Cpu).unwrap());
    let tx = Arc::new(TrackedTensor::from_var(&x).unwrap());
    let sq = tx.sqr().unwrap();
    let neg = tx.neg().unwrap();
    let sum = sq.add(&neg).unwrap();

    let count = graph_viz::node_count(&sum);
    assert_eq!(count, 4, "diamond should have 4 unique nodes");

    let edges = graph_viz::edge_count(&sum);
    assert_eq!(
        edges, 4,
        "diamond: x->sqr, x->neg, sqr->add, neg->add = 4 edges"
    );
}

#[test]
fn test_dot_export_color_coding_present() {
    let x = Var::new(DynTensor::from_vec(vec![1.0], &[1], &Device::Cpu).unwrap());
    let tx = Arc::new(TrackedTensor::from_var(&x).unwrap());
    let out = tx.sqr().unwrap();

    let dot = graph_viz::graph_to_dot(&out);

    // Green for variable input
    assert!(dot.contains("#90EE90"), "should have green for var input");
    // Pink for output node
    assert!(dot.contains("#FFB6C1"), "should have pink for output node");
}

#[test]
fn test_dot_export_minimal_mode() {
    let x = Var::new(DynTensor::from_vec(vec![1.0, 2.0], &[2], &Device::Cpu).unwrap());
    let tx = Arc::new(TrackedTensor::from_var(&x).unwrap());
    let out = tx.sqr().unwrap();

    let full = graph_viz::graph_to_dot(&out);
    let minimal = graph_viz::graph_to_dot_minimal(&out);

    // Minimal should not have grad= annotations
    assert!(
        !minimal.contains("grad="),
        "minimal should omit grad annotations"
    );
    // Full should have them
    assert!(
        full.contains("grad="),
        "full should include grad annotations"
    );
    // Both should be valid DOT
    assert!(minimal.contains("digraph"));
    assert!(full.contains("digraph"));
}

#[test]
fn test_dot_export_constant_node_labeled() {
    let x = Var::new(DynTensor::from_vec(vec![1.0], &[1], &Device::Cpu).unwrap());
    let tx = Arc::new(TrackedTensor::from_var(&x).unwrap());
    let c = Arc::new(TrackedTensor::from_tensor(
        DynTensor::from_vec(vec![5.0], &[1], &Device::Cpu).unwrap(),
    ));
    let sum = tx.add(&c).unwrap();

    let dot = graph_viz::graph_to_dot(&sum);

    // Should contain gray for constant
    assert!(dot.contains("#D3D3D3"), "should have gray for constant");
    // Should label constant as Const
    assert!(
        dot.contains("Const"),
        "should label constant nodes as Const"
    );
}

#[test]
fn test_dot_export_backward_edges() {
    let x = Var::new(DynTensor::from_vec(vec![2.0], &[1], &Device::Cpu).unwrap());
    let tx = Arc::new(TrackedTensor::from_var(&x).unwrap());
    let out = tx.sqr().unwrap();

    let dot = graph_viz::graph_to_dot(&out);

    // Should contain backward gradient flow section
    assert!(
        dot.contains("Backward gradient flow"),
        "DOT should have backward flow section"
    );
    // Backward edges are dashed
    assert!(
        dot.contains("style=dashed"),
        "backward edges should be dashed"
    );
}

#[test]
fn test_dot_write_file_roundtrip() {
    let x = Var::new(DynTensor::from_vec(vec![1.0], &[1], &Device::Cpu).unwrap());
    let tx = Arc::new(TrackedTensor::from_var(&x).unwrap());
    let out = tx.sqr().unwrap();

    let tmp = std::env::temp_dir().join("nn_tape_graph_extended_test.dot");
    graph_viz::write_dot_file(&out, &tmp).expect("write_dot_file should succeed");

    let content = std::fs::read_to_string(&tmp).expect("should read file");
    assert!(content.contains("digraph"), "file should contain valid DOT");
    assert!(content.contains("Sqr"), "file should contain Sqr op label");

    let _ = std::fs::remove_file(&tmp);
}

#[test]
fn test_dot_export_op_names() {
    let x = Var::new(DynTensor::from_vec(vec![2.0], &[1], &Device::Cpu).unwrap());
    let tx = Arc::new(TrackedTensor::from_var(&x).unwrap());
    let sq = tx.sqr().unwrap();
    let out = sq.relu().unwrap();

    let dot = graph_viz::graph_to_dot(&out);

    assert!(dot.contains("Var"), "should have Var label");
    assert!(dot.contains("Sqr"), "should have Sqr label");
    assert!(dot.contains("Relu"), "should have Relu label");
}

#[test]
fn test_dot_export_single_leaf() {
    // Single leaf node with no ops
    let x = Var::new(DynTensor::from_vec(vec![1.0, 2.0], &[2], &Device::Cpu).unwrap());
    let tx = Arc::new(TrackedTensor::from_var(&x).unwrap());

    let dot = graph_viz::graph_to_dot(&tx);
    assert!(dot.contains("digraph"));

    let count = graph_viz::node_count(&tx);
    assert_eq!(count, 1, "single leaf should have 1 node");

    let edges = graph_viz::edge_count(&tx);
    assert_eq!(edges, 0, "single leaf should have 0 edges");
}

// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Tests for tape mechanics: computation graph recording, memory behavior,
//! detach semantics, constant gradient behavior, and graph structure.
//!
//! Covers:
//! - Tape recording captures all ops (op chain verification)
//! - Tape clear / drop frees memory (Arc refcount)
//! - Nested scopes don't leak (scope isolation)
//! - Detach stops gradient flow
//! - Gradient of constant is zero (no gradient)
//! - Multiple backward calls on same graph
//! - Fan-out (one tensor feeds two consumers)
//! - Node ID uniqueness

use std::sync::Arc;

use nn_core::dyn_tensor::DynTensor;
use nn_core::test_utils::cpu;

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

fn scalar_loss(t: &Arc<TrackedTensor>) -> Arc<TrackedTensor> {
    let mut result = Arc::clone(t);
    for d in (0..result.tensor().rank()).rev() {
        result = result.sum_keepdim(d).unwrap();
    }
    result
}

// ===========================================================================
// 1. Tape recording captures all ops
// ===========================================================================

#[test]
fn test_tape_records_op_chain() {
    let x = scalar_var(2.0);
    let tx = tracked(&x);

    // x -> sqr -> relu -> result
    let sq = tx.sqr().unwrap();
    let out = sq.relu().unwrap();

    // The output should have an op (Relu), and its input should have Sqr
    assert!(out.op().is_some(), "output should have a recorded op");
    match out.op().unwrap() {
        Op::Relu(inner) => {
            assert!(inner.op().is_some(), "inner should have Sqr op");
            assert!(
                matches!(inner.op().unwrap(), Op::Sqr(_)),
                "inner op should be Sqr"
            );
        }
        other => panic!("expected Relu op, got {other:?}"),
    }
}

// ===========================================================================
// 2. Tape drop frees computation graph (Arc refcount)
// ===========================================================================

#[test]
fn test_tape_drop_frees_graph() {
    let x = scalar_var(3.0);
    let tx = tracked(&x);

    let sq = tx.sqr().unwrap();
    let loss = scalar_loss(&sq);

    // After backward, we get grads. Then drop the loss and computation graph.
    let grads = backward(&loss).unwrap();
    let grad_val = get_grad_vec(&grads, &x);
    assert!((grad_val[0] - 6.0).abs() < 1e-5, "d/dx(x^2) at x=3 = 6");

    // Drop the computation graph references
    drop(loss);
    drop(sq);

    // tx should be the only strong reference (from our scope)
    // Arc::strong_count == 1 means no leaked references in the graph
    assert_eq!(
        Arc::strong_count(&tx),
        1,
        "after dropping graph, only our reference to tx should remain"
    );
}

// ===========================================================================
// 3. Nested scopes don't leak
// ===========================================================================

#[test]
fn test_nested_scopes_no_leak() {
    let x = scalar_var(2.0);

    let grads = {
        let tx = tracked(&x);
        let inner_result = {
            let sq = tx.sqr().unwrap();
            let cube = sq.mul(&tx).unwrap(); // x^3
            scalar_loss(&cube)
        };
        // inner_result holds the graph alive
        backward(&inner_result).unwrap()
        // inner_result dropped here
    };
    // tx dropped here

    // Gradients should still be valid even though the graph is dropped
    let grad_val = get_grad_vec(&grads, &x);
    // d/dx(x^3) = 3x^2 = 3*4 = 12
    assert!(
        (grad_val[0] - 12.0).abs() < 1e-4,
        "d/dx(x^3) at x=2 should be 12, got {}",
        grad_val[0]
    );
}

// ===========================================================================
// 4. Detach stops gradient flow
// ===========================================================================

#[test]
fn test_detach_stops_gradient_flow() {
    let x = scalar_var(3.0);
    let tx = tracked(&x);

    // y = x^2, but detach y before further computation
    let y = tx.sqr().unwrap();
    let y_detached = y.detach();

    // z = y_detached * 2 (gradient won't flow back to x)
    let two = Arc::new(TrackedTensor::from_tensor(
        DynTensor::from_vec(vec![2.0], &[1], &cpu()).unwrap(),
    ));
    let z = y_detached.mul(&two).unwrap();
    let loss = scalar_loss(&z);
    let grads = backward(&loss).unwrap();

    // x should have no gradient because detach broke the chain
    assert!(
        grads.get(&x).is_none(),
        "detach should prevent gradient flow to x"
    );
}

#[test]
fn test_detach_preserves_value() {
    let x = scalar_var(4.0);
    let tx = tracked(&x);
    let sq = tx.sqr().unwrap();
    let detached = sq.detach();

    // Value should be preserved
    let val = detached.tensor().to_flat_vec::<f32>().unwrap()[0];
    assert!(
        (val - 16.0).abs() < 1e-5,
        "detached tensor should have same value: expected 16.0, got {val}"
    );

    // But it should have no op
    assert!(detached.op().is_none(), "detached tensor should have no op");
    assert!(!detached.is_var(), "detached tensor should not be a var");
}

// ===========================================================================
// 5. Gradient of constant is zero (no gradient stored)
// ===========================================================================

#[test]
fn test_constant_has_no_gradient() {
    let x = scalar_var(2.0);
    let tx = tracked(&x);

    let c = Arc::new(TrackedTensor::from_tensor(
        DynTensor::from_vec(vec![5.0], &[1], &cpu()).unwrap(),
    ));

    // loss = x * c = 2*5 = 10
    let prod = tx.mul(&c).unwrap();
    let loss = scalar_loss(&prod);
    let grads = backward(&loss).unwrap();

    // x gets gradient = c = 5
    let grad_x = get_grad_vec(&grads, &x);
    assert!(
        (grad_x[0] - 5.0).abs() < 1e-5,
        "d/dx(x*c) = c = 5, got {}",
        grad_x[0]
    );

    // Constant has no VarId, so no gradient in var_grads
    assert_eq!(
        grads.var_count(),
        1,
        "only one variable (x) should have a gradient"
    );
}

// ===========================================================================
// 6. Multiple backward calls on same graph
// ===========================================================================

#[test]
fn test_multiple_backward_calls() {
    let x = scalar_var(3.0);
    let tx = tracked(&x);
    let sq = tx.sqr().unwrap();
    let loss = scalar_loss(&sq);

    // First backward
    let grads1 = backward(&loss).unwrap();
    let g1 = get_grad_vec(&grads1, &x)[0];

    // Second backward on the same graph should give the same result
    let grads2 = backward(&loss).unwrap();
    let g2 = get_grad_vec(&grads2, &x)[0];

    assert!(
        (g1 - g2).abs() < 1e-7,
        "repeated backward should give same gradients: g1={g1}, g2={g2}"
    );
    assert!((g1 - 6.0).abs() < 1e-5, "d/dx(x^2) at x=3 = 6");
}

// ===========================================================================
// 7. Fan-out: one tensor feeds two consumers
// ===========================================================================

#[test]
fn test_fan_out_gradient_accumulation() {
    let x = scalar_var(2.0);
    let tx = tracked(&x);

    // y1 = x^2, y2 = x^3 (both use tx)
    let sq = tx.sqr().unwrap(); // x^2
    let cube = sq.mul(&tx).unwrap(); // x^2 * x = x^3

    // loss = x^2 + x^3
    let sum = sq.add(&cube).unwrap();
    let loss = scalar_loss(&sum);
    let grads = backward(&loss).unwrap();

    // d/dx(x^2 + x^3) = 2x + 3x^2 = 4 + 12 = 16
    let grad = get_grad_vec(&grads, &x)[0];
    assert!(
        (grad - 16.0).abs() < 1e-4,
        "d/dx(x^2 + x^3) at x=2 = 16, got {grad}"
    );
}

// ===========================================================================
// 8. Node ID uniqueness
// ===========================================================================

#[test]
fn test_node_ids_are_unique() {
    let a = scalar_var(1.0);
    let b = scalar_var(2.0);
    let ta = tracked(&a);
    let tb = tracked(&b);
    let sum = ta.add(&tb).unwrap();
    let sq = sum.sqr().unwrap();

    // Collect all node IDs
    let ids = [
        ta.node_id().as_u64(),
        tb.node_id().as_u64(),
        sum.node_id().as_u64(),
        sq.node_id().as_u64(),
    ];

    // All should be distinct
    let mut sorted = ids.to_vec();
    sorted.sort_unstable();
    sorted.dedup();
    assert_eq!(
        sorted.len(),
        ids.len(),
        "all node IDs should be unique: {ids:?}"
    );
}

// ===========================================================================
// 9. Leaf variable has no op
// ===========================================================================

#[test]
fn test_leaf_variable_has_no_op() {
    let x = scalar_var(1.0);
    let tx = tracked(&x);

    assert!(tx.op().is_none(), "leaf variable should have no op");
    assert!(tx.is_var(), "leaf from_var should be marked as var");
    assert!(tx.var_id().is_some(), "leaf from_var should have a VarId");
}

// ===========================================================================
// 10. from_tensor leaf is not a var
// ===========================================================================

#[test]
fn test_from_tensor_is_not_var() {
    let t = Arc::new(TrackedTensor::from_tensor(
        DynTensor::from_vec(vec![1.0, 2.0], &[2], &cpu()).unwrap(),
    ));

    assert!(!t.is_var(), "from_tensor should not be a var");
    assert!(t.var_id().is_none(), "from_tensor should have no VarId");
    assert!(t.op().is_none(), "from_tensor should have no op");
}

// ===========================================================================
// 11. Backward on non-scalar fails
// ===========================================================================

#[test]
fn test_backward_non_scalar_fails() {
    let x = vec_var(vec![1.0, 2.0, 3.0]);
    let tx = tracked(&x);
    let sq = tx.sqr().unwrap();

    // sq has shape [3], not scalar
    let result = backward(&sq);
    assert!(result.is_err(), "backward on non-scalar should fail");
}

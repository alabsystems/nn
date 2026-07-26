#![allow(deprecated)]
// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

use super::*;
use crate::var::Var;
use nn_core::dyn_tensor::DynTensor;
use nn_core::test_utils::cpu;
use nn_core::DType;

fn vec_tensor(data: &[f32], dims: &[usize]) -> DynTensor {
    DynTensor::new(data, dims, &cpu()).unwrap()
}

#[test]
fn test_from_var_is_leaf() {
    let var = Var::zeros(&[3], DType::F32, &cpu()).unwrap();
    let tracked = TrackedTensor::from_var(&var).unwrap();
    assert!(tracked.is_var());
    assert_eq!(tracked.var_id(), Some(var.id()));
    assert!(tracked.op().is_none());
}

#[test]
fn test_from_tensor_is_constant() {
    let t = vec_tensor(&[1.0, 2.0], &[2]);
    let tracked = TrackedTensor::from_tensor(t);
    assert!(!tracked.is_var());
    assert_eq!(tracked.var_id(), None);
    assert!(tracked.op().is_none());
}

#[test]
fn test_from_op_has_op() {
    let t = vec_tensor(&[1.0], &[1]);
    let inner = Arc::new(TrackedTensor::from_tensor(t.clone()));
    let tracked = TrackedTensor::from_op(t, Op::Neg(inner));
    assert!(!tracked.is_var());
    assert!(tracked.op().is_some());
}

#[test]
fn test_add_records_op() {
    let a = Arc::new(TrackedTensor::from_tensor(vec_tensor(&[1.0, 2.0], &[2])));
    let b = Arc::new(TrackedTensor::from_tensor(vec_tensor(&[3.0, 4.0], &[2])));
    let c = a.add(&b).unwrap();
    assert!(c.op().is_some());
    assert_eq!(c.tensor().to_flat_vec::<f32>().unwrap(), &[4.0, 6.0]);
}

#[test]
fn test_sub_records_op() {
    let a = Arc::new(TrackedTensor::from_tensor(vec_tensor(&[5.0, 3.0], &[2])));
    let b = Arc::new(TrackedTensor::from_tensor(vec_tensor(&[1.0, 1.0], &[2])));
    let c = a.sub(&b).unwrap();
    assert_eq!(c.tensor().to_flat_vec::<f32>().unwrap(), &[4.0, 2.0]);
}

#[test]
fn test_mul_records_op() {
    let a = Arc::new(TrackedTensor::from_tensor(vec_tensor(&[2.0, 3.0], &[2])));
    let b = Arc::new(TrackedTensor::from_tensor(vec_tensor(&[4.0, 5.0], &[2])));
    let c = a.mul(&b).unwrap();
    assert_eq!(c.tensor().to_flat_vec::<f32>().unwrap(), &[8.0, 15.0]);
}

#[test]
fn test_matmul_records_op() {
    let a = Arc::new(TrackedTensor::from_tensor(vec_tensor(
        &[1.0, 2.0, 3.0, 4.0],
        &[2, 2],
    )));
    let b = Arc::new(TrackedTensor::from_tensor(vec_tensor(
        &[5.0, 6.0, 7.0, 8.0],
        &[2, 2],
    )));
    let c = a.matmul(&b).unwrap();
    assert_eq!(c.dims(), &[2, 2]);
    // [1*5+2*7, 1*6+2*8, 3*5+4*7, 3*6+4*8] = [19, 22, 43, 50]
    assert_eq!(
        c.tensor().to_flat_vec::<f32>().unwrap(),
        &[19.0, 22.0, 43.0, 50.0]
    );
}

#[test]
fn test_relu_records_op() {
    let x = Arc::new(TrackedTensor::from_tensor(vec_tensor(
        &[-1.0, 0.0, 1.0],
        &[3],
    )));
    let y = x.relu().unwrap();
    assert_eq!(y.tensor().to_flat_vec::<f32>().unwrap(), &[0.0, 0.0, 1.0]);
}

#[test]
fn test_neg_records_op() {
    let x = Arc::new(TrackedTensor::from_tensor(vec_tensor(&[1.0, -2.0], &[2])));
    let y = x.neg().unwrap();
    assert_eq!(y.tensor().to_flat_vec::<f32>().unwrap(), &[-1.0, 2.0]);
}

#[test]
fn test_sqr_records_op() {
    let x = Arc::new(TrackedTensor::from_tensor(vec_tensor(&[3.0, -2.0], &[2])));
    let y = x.sqr().unwrap();
    assert_eq!(y.tensor().to_flat_vec::<f32>().unwrap(), &[9.0, 4.0]);
}

#[test]
fn test_mul_scalar_records_op() {
    let x = Arc::new(TrackedTensor::from_tensor(vec_tensor(&[2.0, 3.0], &[2])));
    let y = x.mul_scalar(10.0).unwrap();
    assert_eq!(y.tensor().to_flat_vec::<f32>().unwrap(), &[20.0, 30.0]);
}

#[test]
fn test_add_scalar_records_op() {
    let x = Arc::new(TrackedTensor::from_tensor(vec_tensor(&[1.0, 2.0], &[2])));
    let y = x.add_scalar(5.0).unwrap();
    assert_eq!(y.tensor().to_flat_vec::<f32>().unwrap(), &[6.0, 7.0]);
}

#[test]
fn test_reshape_records_original_shape() {
    let x = Arc::new(TrackedTensor::from_tensor(vec_tensor(
        &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        &[2, 3],
    )));
    let y = x.reshape(&[3, 2]).unwrap();
    assert_eq!(y.dims(), &[3, 2]);
    // Op should store original shape [2, 3] for backward reshape.
    match y.op() {
        Some(Op::Reshape(_, orig)) => assert_eq!(orig, &[2, 3]),
        other => panic!("expected Reshape op, got {other:?}"),
    }
}

#[test]
fn test_transpose_records_dims() {
    let x = Arc::new(TrackedTensor::from_tensor(vec_tensor(
        &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        &[2, 3],
    )));
    let y = x.transpose(0, 1).unwrap();
    assert_eq!(y.dims(), &[3, 2]);
}

#[test]
fn test_node_id_unique() {
    let a = TrackedTensor::from_tensor(vec_tensor(&[1.0], &[1]));
    let b = TrackedTensor::from_tensor(vec_tensor(&[2.0], &[1]));
    assert_ne!(a.node_id, b.node_id);
}

#[test]
fn test_numel_and_dims() {
    let t = vec_tensor(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
    let tracked = TrackedTensor::from_tensor(t);
    assert_eq!(tracked.numel(), 6);
    assert_eq!(tracked.dims(), &[2, 3]);
}

#[test]
fn test_into_tensor() {
    let t = vec_tensor(&[1.0, 2.0], &[2]);
    let tracked = TrackedTensor::from_tensor(t);
    let recovered = tracked.into_tensor().unwrap();
    assert_eq!(recovered.to_flat_vec::<f32>().unwrap(), &[1.0, 2.0]);
}

#[test]
fn test_chain_ops() {
    // (x + y) * z
    let x = Arc::new(TrackedTensor::from_tensor(vec_tensor(&[1.0, 2.0], &[2])));
    let y = Arc::new(TrackedTensor::from_tensor(vec_tensor(&[3.0, 4.0], &[2])));
    let z = Arc::new(TrackedTensor::from_tensor(vec_tensor(&[2.0, 2.0], &[2])));

    let sum = x.add(&y).unwrap();
    let product = sum.mul(&z).unwrap();
    assert_eq!(product.tensor().to_flat_vec::<f32>().unwrap(), &[8.0, 12.0]);
    assert!(product.op().is_some());
}

#[test]
fn test_debug_format() {
    let tracked = TrackedTensor::from_tensor(vec_tensor(&[1.0], &[1]));
    let debug = format!("{tracked:?}");
    assert!(debug.contains("TrackedTensor"));
    assert!(debug.contains("[1]"));
}

#[test]
fn test_detach_severs_graph() {
    let a = Arc::new(TrackedTensor::from_tensor(vec_tensor(&[1.0, 2.0], &[2])));
    let b = Arc::new(TrackedTensor::from_tensor(vec_tensor(&[3.0, 4.0], &[2])));
    let c = a.add(&b).unwrap();
    assert!(c.op().is_some(), "c has an op before detach");

    let d = c.detach();
    assert!(d.op().is_none(), "detached tensor has no op");
    assert!(!d.is_var(), "detached tensor is not a variable");
    assert_eq!(d.var_id(), None, "detached tensor has no var_id");
    assert_eq!(d.tensor().to_flat_vec::<f32>().unwrap(), &[4.0, 6.0]);
    assert_ne!(
        d.node_id(),
        c.node_id(),
        "detached tensor gets fresh node_id"
    );
}

#[test]
fn test_detach_blocks_gradient_flow() {
    use crate::grad::backward;

    // x -> (x * 2) -> detach -> (result + y) -> loss
    // Gradient should flow to y but NOT to x (detach blocks it).
    let x = Var::from_tensor(&vec_tensor(&[3.0], &[1]));
    let y = Var::from_tensor(&vec_tensor(&[1.0], &[1]));

    let tx = Arc::new(TrackedTensor::from_var(&x).unwrap());
    let ty = Arc::new(TrackedTensor::from_var(&y).unwrap());

    let doubled = tx.mul_scalar(2.0).unwrap(); // 6.0
    let detached = doubled.detach(); // 6.0, but no graph connection to x
    let loss = detached.add(&ty).unwrap(); // 7.0

    // Verify the forward computation is correct
    assert_eq!(
        loss.tensor().to_flat_vec::<f32>().unwrap(),
        &[7.0],
        "loss = 6.0 + 1.0 = 7.0"
    );

    let grads = backward(&loss).unwrap();

    // d(loss)/d(y) = 1.0 — gradient flows through the add to y
    let y_grad = grads.get(&y);
    assert!(y_grad.is_some(), "y should have gradient");
    assert_eq!(
        y_grad.unwrap().to_flat_vec::<f32>().unwrap(),
        &[1.0],
        "d(loss)/d(y) = 1.0"
    );

    // x should NOT have a gradient — detach severed the graph
    let x_grad = grads.get(&x);
    assert!(
        x_grad.is_none(),
        "x should NOT have gradient (detach severed the graph)"
    );
}

/// Verify that dropping a deep TrackedTensor chain does not stack-overflow.
///
/// Without the iterative drop impl in tracked_drop.rs, a chain of 50,000
/// nodes would recurse through `TrackedTensor → Op → Arc<TrackedTensor> → ...`
/// consuming one stack frame per node. The default thread stack is 8 MiB on
/// most platforms, so ~50k frames of ~128 bytes each would overflow.
#[test]
fn test_deep_chain_drop_no_stack_overflow() {
    use super::NodeId;
    let depth = 50_000;
    let initial = vec_tensor(&[1.0], &[1]);
    let mut current = Arc::new(TrackedTensor::from_tensor(initial));

    for _ in 0..depth {
        let prev = current;
        let data = prev.tensor().clone();
        let next = TrackedTensor {
            data,
            op: Some(Op::Relu(prev)),
            is_var: false,
            var_id: None,
            node_id: NodeId::next(),
        };
        current = Arc::new(next);
    }

    // Dropping `current` should NOT stack-overflow thanks to the iterative
    // drop implementation in tracked_drop.rs. If it did, this test would
    // crash with SIGSEGV rather than returning.
    drop(current);
}

/// Verify that into_tensor returns the underlying data and that the
/// Drop impl runs cleanly (op is None after take).
#[test]
fn test_into_tensor_drop_safety() {
    use super::NodeId;
    let a_var = Var::zeros(&[4], DType::F32, &cpu()).unwrap();
    let a = TrackedTensor::from_var(&a_var).unwrap();
    let b = TrackedTensor::from_tensor(vec_tensor(&[1.0, 2.0, 3.0, 4.0], &[4]));

    // Build a graph: c = a + b
    let c = Arc::new(TrackedTensor {
        data: a.tensor().clone(),
        op: Some(Op::Add(Arc::new(a), Arc::new(b))),
        is_var: false,
        var_id: None,
        node_id: NodeId::next(),
    });

    // Unwrap via into_tensor. The Arc::try_unwrap should succeed (sole owner).
    let c_owned = Arc::try_unwrap(c).expect("sole owner");
    let tensor = c_owned.into_tensor().unwrap();
    assert_eq!(tensor.dims(), &[4]);
    // Verify the returned data is the original (a's zeros), not garbage or
    // the dummy replacement tensor.  Without this assertion, a bug in
    // into_tensor that returned the dummy instead of the real data would
    // go undetected.
    assert_eq!(
        tensor.to_flat_vec::<f32>().unwrap(),
        &[0.0, 0.0, 0.0, 0.0],
        "into_tensor should return the original data (a's zeros)"
    );
    // tensor drops here — should be clean since op was taken
}

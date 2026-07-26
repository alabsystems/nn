#![allow(deprecated)]
// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Matmul backward tests: batched matmul (3D, 4D), broadcast reduction, rank guard.

use std::sync::Arc;

use nn_core::dyn_tensor::DynTensor;
use nn_core::test_utils::cpu;

use crate::grad::backward;
use crate::tracked::TrackedTensor;
use crate::var::Var;

// -- Batched matmul backward test -------------------------------------------

#[test]
fn test_backward_matmul_3d_batch() {
    // a = [2, 1, 3], b = [2, 3, 1] => y = [2, 1, 1] (batched matmul)
    // Batch 0: [1,2,3] @ [[1],[2],[3]] = [[14]]
    // Batch 1: [4,5,6] @ [[4],[5],[6]] = [[77]]
    // loss = sum(y) = 14 + 77 = 91
    //
    // dy/da: grad @ b^T
    //   batch 0: [[1]] @ [[1,2,3]] = [[1,2,3]]
    //   batch 1: [[1]] @ [[4,5,6]] = [[4,5,6]]
    let a_var = Var::new(
        DynTensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 1, 3], &cpu()).unwrap(),
    );
    let b_var = Var::new(
        DynTensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3, 1], &cpu()).unwrap(),
    );
    let ta = Arc::new(TrackedTensor::from_var(&a_var).unwrap());
    let tb = Arc::new(TrackedTensor::from_var(&b_var).unwrap());
    let y = ta.matmul(&tb).unwrap(); // [2, 1, 1]
    let loss = y
        .sum_keepdim(0)
        .unwrap()
        .sum_keepdim(1)
        .unwrap()
        .sum_keepdim(2)
        .unwrap();
    let grads = backward(&loss).unwrap();

    let ga = grads.get(&a_var).unwrap().to_flat_vec::<f32>().unwrap();
    // batch 0: [1, 2, 3], batch 1: [4, 5, 6]
    assert!(
        (ga[0] - 1.0).abs() < 1e-5 && (ga[1] - 2.0).abs() < 1e-5 && (ga[2] - 3.0).abs() < 1e-5,
        "batch 0 grad_a: expected [1,2,3], got {:?}",
        &ga[0..3]
    );
    assert!(
        (ga[3] - 4.0).abs() < 1e-5 && (ga[4] - 5.0).abs() < 1e-5 && (ga[5] - 6.0).abs() < 1e-5,
        "batch 1 grad_a: expected [4,5,6], got {:?}",
        &ga[3..6]
    );

    // dy/db: a^T @ grad
    //   batch 0: [[1],[2],[3]] @ [[1]] = [[1],[2],[3]]
    //   batch 1: [[4],[5],[6]] @ [[1]] = [[4],[5],[6]]
    let gb = grads.get(&b_var).unwrap().to_flat_vec::<f32>().unwrap();
    assert!(
        (gb[0] - 1.0).abs() < 1e-5 && (gb[1] - 2.0).abs() < 1e-5 && (gb[2] - 3.0).abs() < 1e-5,
        "batch 0 grad_b: expected [1,2,3], got {:?}",
        &gb[0..3]
    );
    assert!(
        (gb[3] - 4.0).abs() < 1e-5 && (gb[4] - 5.0).abs() < 1e-5 && (gb[5] - 6.0).abs() < 1e-5,
        "batch 1 grad_b: expected [4,5,6], got {:?}",
        &gb[3..6]
    );
}

// -- Linear on batched input: [B, S, D] × [D, H] backward test --------

#[test]
fn test_backward_matmul_3d_times_2d() {
    // This is the common Linear::forward() on batched input:
    // a = [2, 1, 3] (batch=2, seq=1, dim=3)
    // b = [3, 2] (weight matrix: dim_in=3, dim_out=2)
    // y = a @ b => [2, 1, 2]
    //
    // For loss = sum(y), the gradient for b should be [3, 2] (same shape
    // as b), computed as: sum over batch of a^T @ grad
    let a_var = Var::new(
        DynTensor::from_vec(vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0], &[2, 1, 3], &cpu()).unwrap(),
    );
    let b_var =
        Var::new(DynTensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[3, 2], &cpu()).unwrap());
    let ta = Arc::new(TrackedTensor::from_var(&a_var).unwrap());
    let tb = Arc::new(TrackedTensor::from_var(&b_var).unwrap());
    let y = ta.matmul(&tb).unwrap(); // [2, 1, 2]
    assert_eq!(y.tensor().dims(), &[2, 1, 2]);

    let loss = y
        .sum_keepdim(0)
        .unwrap()
        .sum_keepdim(1)
        .unwrap()
        .sum_keepdim(2)
        .unwrap();
    let grads = backward(&loss).unwrap();

    // grad_a should have shape [2, 1, 3] (same as a)
    let ga = grads.get(&a_var).unwrap();
    assert_eq!(ga.dims(), &[2, 1, 3], "grad_a shape mismatch");

    // Verify grad_a values: grad @ b^T
    // batch 0: [[1,1]] @ [[1,3,5],[2,4,6]] = [[3, 7, 11]]
    // batch 1: [[1,1]] @ [[1,3,5],[2,4,6]] = [[3, 7, 11]]
    let ga_vals = ga.to_flat_vec::<f32>().unwrap();
    let expected_ga = [3.0, 7.0, 11.0, 3.0, 7.0, 11.0];
    for (i, (&got, &exp)) in ga_vals.iter().zip(expected_ga.iter()).enumerate() {
        assert!(
            (got - exp).abs() < 1e-5,
            "grad_a[{i}]: expected {exp}, got {got}"
        );
    }

    // grad_b should have shape [3, 2] (same as b) — this is the critical check.
    // If the backward rule doesn't reduce across the batch dimension, grad_b
    // will incorrectly have shape [2, 3, 2].
    let gb = grads.get(&b_var).unwrap();
    assert_eq!(
        gb.dims(),
        &[3, 2],
        "grad_b shape mismatch: expected [3, 2] but got {:?} — \
         backward rule fails to reduce across batch dim for 3D×2D matmul",
        gb.dims()
    );

    // Verify grad_b values: sum over batch of a^T @ grad
    // batch 0: [[1],[0],[0]] @ [[1,1]] = [[1,1],[0,0],[0,0]]
    // batch 1: [[0],[1],[0]] @ [[1,1]] = [[0,0],[1,1],[0,0]]
    // sum:     [[1,1],[1,1],[0,0]]
    let gb_vals = gb.to_flat_vec::<f32>().unwrap();
    let expected_gb = [1.0, 1.0, 1.0, 1.0, 0.0, 0.0];
    for (i, (&got, &exp)) in gb_vals.iter().zip(expected_gb.iter()).enumerate() {
        assert!(
            (got - exp).abs() < 1e-5,
            "grad_b[{i}]: expected {exp}, got {got}"
        );
    }
}

#[test]
fn test_backward_matmul_4d_times_2d() {
    // 4D×2D: a = [2, 1, 1, 3] (batch=2, heads=1, seq=1, dim=3)
    // b = [3, 2] (weight matrix: dim_in=3, dim_out=2)
    // y = a @ b => [2, 1, 1, 2]
    //
    // grad_b should be [3, 2] (same shape as b) — reduced from [2, 1, 3, 2]
    let a_var = Var::new(
        DynTensor::from_vec(vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0], &[2, 1, 1, 3], &cpu()).unwrap(),
    );
    let b_var =
        Var::new(DynTensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[3, 2], &cpu()).unwrap());
    let ta = Arc::new(TrackedTensor::from_var(&a_var).unwrap());
    let tb = Arc::new(TrackedTensor::from_var(&b_var).unwrap());
    let y = ta.matmul(&tb).unwrap(); // [2, 1, 1, 2]
    assert_eq!(y.tensor().dims(), &[2, 1, 1, 2]);

    let loss = y
        .sum_keepdim(0)
        .unwrap()
        .sum_keepdim(1)
        .unwrap()
        .sum_keepdim(2)
        .unwrap()
        .sum_keepdim(3)
        .unwrap();
    let grads = backward(&loss).unwrap();

    // grad_a should have shape [2, 1, 1, 3] (same as a)
    let ga = grads.get(&a_var).unwrap();
    assert_eq!(ga.dims(), &[2, 1, 1, 3], "grad_a shape mismatch");

    // Verify grad_a values: same as 3D case but with extra head dim
    let ga_vals = ga.to_flat_vec::<f32>().unwrap();
    let expected_ga = [3.0, 7.0, 11.0, 3.0, 7.0, 11.0];
    for (i, (&got, &exp)) in ga_vals.iter().zip(expected_ga.iter()).enumerate() {
        assert!(
            (got - exp).abs() < 1e-5,
            "4D grad_a[{i}]: expected {exp}, got {got}"
        );
    }

    // grad_b must be reduced back to [3, 2] — not [2, 1, 3, 2]
    let gb = grads.get(&b_var).unwrap();
    assert_eq!(
        gb.dims(),
        &[3, 2],
        "grad_b shape mismatch: expected [3, 2] but got {:?} — \
         backward rule fails to reduce across batch dims for 4D×2D matmul",
        gb.dims()
    );

    // Verify grad_b values: same as 3D case — [[1,1],[1,1],[0,0]]
    let gb_vals = gb.to_flat_vec::<f32>().unwrap();
    let expected_gb = [1.0, 1.0, 1.0, 1.0, 0.0, 0.0];
    for (i, (&got, &exp)) in gb_vals.iter().zip(expected_gb.iter()).enumerate() {
        assert!(
            (got - exp).abs() < 1e-5,
            "4D grad_b[{i}]: expected {exp}, got {got}"
        );
    }
}

// -- MatMul backward rank guard (#1515 AC1) ----------------------------------

/// MatMul backward must return an error for rank-1 tensors (vectors),
/// not panic from usize underflow on `r - 2`.
#[test]
fn test_backward_matmul_rejects_rank1_inputs() {
    let a = Var::new(DynTensor::from_vec(vec![1.0, 2.0, 3.0], &[3], &cpu()).unwrap());
    let b = Var::new(DynTensor::from_vec(vec![4.0, 5.0, 6.0], &[3], &cpu()).unwrap());
    let ta = Arc::new(TrackedTensor::from_var(&a).unwrap());
    let tb = Arc::new(TrackedTensor::from_var(&b).unwrap());

    // Forward matmul of two 1D tensors produces a scalar (dot product).
    // But backward requires rank >= 2 for transpose. This should error, not panic.
    let result = ta.matmul(&tb);
    match result {
        Err(e) => {
            // Forward matmul rejects rank-1 × rank-1 — that's fine too
            let msg = format!("{e}");
            assert!(
                msg.contains("rank") || msg.contains("matmul") || msg.contains("shape"),
                "expected shape/rank error, got: {msg}"
            );
        }
        Ok(output) => {
            // If forward succeeds, backward must not panic
            let err = backward(&output).unwrap_err();
            let msg = format!("{err}");
            assert!(
                msg.contains("rank") || msg.contains("MatMul"),
                "expected rank error in backward, got: {msg}"
            );
        }
    }
}

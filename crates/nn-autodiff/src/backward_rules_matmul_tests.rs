// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Backward rule tests for matmul, including batched matmul, transpose matmul,
//! numerical gradient checking, and shape validation.

use std::sync::Arc;

use nn_core::dyn_tensor::DynTensor;
use nn_core::test_utils::cpu;

use crate::grad::backward;
use crate::tracked::TrackedTensor;
use crate::var::Var;

/// Helper: create a Var from a flat vec with given shape.
fn var_from(data: Vec<f32>, shape: &[usize]) -> Var {
    Var::new(DynTensor::from_vec(data, shape, &cpu()).unwrap())
}

/// Helper: extract flat gradient for a var from GradStore.
fn grad_vec(grads: &crate::grad::GradStore, var: &Var) -> Vec<f32> {
    grads.get(var).unwrap().to_flat_vec::<f32>().unwrap()
}

/// Central-difference numerical gradient: (f(x+eps) - f(x-eps)) / (2*eps).
fn numerical_grad(data: &[f32], eps: f32, fwd: impl Fn(Vec<f32>) -> f64) -> Vec<f64> {
    let mut result = Vec::with_capacity(data.len());
    for i in 0..data.len() {
        let mut plus = data.to_vec();
        let mut minus = data.to_vec();
        plus[i] += eps;
        minus[i] -= eps;
        result.push((fwd(plus) - fwd(minus)) / (2.0 * f64::from(eps)));
    }
    result
}

/// Assert analytical and numerical gradients match within tolerance.
fn assert_grad_close(analytical: &[f32], numerical: &[f64], tol: f64, label: &str) {
    assert_eq!(
        analytical.len(),
        numerical.len(),
        "{label}: length mismatch"
    );
    for (i, (&a, &n)) in analytical.iter().zip(numerical.iter()).enumerate() {
        let err = (f64::from(a) - n).abs();
        assert!(
            err < tol,
            "{label}[{i}]: analytical={a}, numerical={n}, err={err}, tol={tol}"
        );
    }
}

// ── MatMul gradient shapes match input shapes ──────────────────────────

#[test]
fn test_matmul_backward_grad_a_shape_2d() {
    // [M, K] x [K, N] -> [M, N]; grad_a should be [M, K]
    let a = var_from(vec![1.0; 6], &[2, 3]);
    let b = var_from(vec![1.0; 12], &[3, 4]);
    let ta = Arc::new(TrackedTensor::from_var(&a).unwrap());
    let tb = Arc::new(TrackedTensor::from_var(&b).unwrap());
    let y = ta.matmul(&tb).unwrap();
    let loss = y.sum_keepdim(0).unwrap().sum_keepdim(1).unwrap();
    let grads = backward(&loss).unwrap();

    let ga = grads.get(&a).unwrap();
    let gb = grads.get(&b).unwrap();
    assert_eq!(ga.dims(), &[2, 3], "grad_a shape must match input A");
    assert_eq!(gb.dims(), &[3, 4], "grad_b shape must match input B");
}

#[test]
fn test_matmul_backward_grad_b_shape_2d() {
    // Verify grad_b shape for non-square matmul
    let a = var_from(vec![0.5; 15], &[5, 3]);
    let b = var_from(vec![0.5; 6], &[3, 2]);
    let ta = Arc::new(TrackedTensor::from_var(&a).unwrap());
    let tb = Arc::new(TrackedTensor::from_var(&b).unwrap());
    let y = ta.matmul(&tb).unwrap();
    let loss = y.sum_keepdim(0).unwrap().sum_keepdim(1).unwrap();
    let grads = backward(&loss).unwrap();

    let gb = grads.get(&b).unwrap();
    assert_eq!(gb.dims(), &[3, 2], "grad_b shape must match input B [3, 2]");
}

// ── MatMul gradient values via finite differences ──────────────────────

#[test]
fn test_matmul_backward_grad_a_fd() {
    let a_data = vec![0.5_f32, -0.3, 1.2, 0.8, -0.6, 0.4];
    let b_data = vec![1.0_f32, 0.3, -0.5, 0.7, 0.2, -0.8];

    let a = var_from(a_data.clone(), &[2, 3]);
    let b = var_from(b_data.clone(), &[3, 2]);
    let ta = Arc::new(TrackedTensor::from_var(&a).unwrap());
    let tb = Arc::new(TrackedTensor::from_var(&b).unwrap());
    let y = ta.matmul(&tb).unwrap();
    let loss = y
        .sqr()
        .unwrap()
        .sum_keepdim(0)
        .unwrap()
        .sum_keepdim(1)
        .unwrap();
    let grads = backward(&loss).unwrap();

    let ga = grad_vec(&grads, &a);
    let num = numerical_grad(&a_data, 1e-3, |d| {
        let at = DynTensor::from_vec(d, &[2, 3], &cpu()).unwrap();
        let bt = DynTensor::from_vec(b_data.clone(), &[3, 2], &cpu()).unwrap();
        let y = at.matmul(&bt).unwrap();
        y.to_flat_vec::<f32>()
            .unwrap()
            .iter()
            .map(|&v| f64::from(v) * f64::from(v))
            .sum()
    });
    assert_grad_close(&ga, &num, 1e-2, "matmul_fd_a");
}

#[test]
fn test_matmul_backward_grad_b_fd() {
    let a_data = vec![0.5_f32, -0.3, 1.2, 0.8, -0.6, 0.4];
    let b_data = vec![1.0_f32, 0.3, -0.5, 0.7, 0.2, -0.8];

    let a = var_from(a_data.clone(), &[2, 3]);
    let b = var_from(b_data.clone(), &[3, 2]);
    let ta = Arc::new(TrackedTensor::from_var(&a).unwrap());
    let tb = Arc::new(TrackedTensor::from_var(&b).unwrap());
    let y = ta.matmul(&tb).unwrap();
    let loss = y
        .sqr()
        .unwrap()
        .sum_keepdim(0)
        .unwrap()
        .sum_keepdim(1)
        .unwrap();
    let grads = backward(&loss).unwrap();

    let gb = grad_vec(&grads, &b);
    let num = numerical_grad(&b_data, 1e-3, |d| {
        let at = DynTensor::from_vec(a_data.clone(), &[2, 3], &cpu()).unwrap();
        let bt = DynTensor::from_vec(d, &[3, 2], &cpu()).unwrap();
        let y = at.matmul(&bt).unwrap();
        y.to_flat_vec::<f32>()
            .unwrap()
            .iter()
            .map(|&v| f64::from(v) * f64::from(v))
            .sum()
    });
    assert_grad_close(&gb, &num, 1e-2, "matmul_fd_b");
}

// ── Batched matmul backward (3D) ──────────────────────────────────────

#[test]
fn test_batched_matmul_backward_shapes_3d() {
    // [B, M, K] x [B, K, N] -> [B, M, N]
    let a = var_from(vec![0.1; 24], &[2, 3, 4]); // B=2, M=3, K=4
    let b = var_from(vec![0.1; 40], &[2, 4, 5]); // B=2, K=4, N=5
    let ta = Arc::new(TrackedTensor::from_var(&a).unwrap());
    let tb = Arc::new(TrackedTensor::from_var(&b).unwrap());
    let y = ta.matmul(&tb).unwrap();
    let loss = y
        .sum_keepdim(0)
        .unwrap()
        .sum_keepdim(1)
        .unwrap()
        .sum_keepdim(2)
        .unwrap();
    let grads = backward(&loss).unwrap();

    let ga = grads.get(&a).unwrap();
    let gb = grads.get(&b).unwrap();
    assert_eq!(ga.dims(), &[2, 3, 4], "batched grad_a shape");
    assert_eq!(gb.dims(), &[2, 4, 5], "batched grad_b shape");
}

#[test]
fn test_batched_matmul_backward_fd_grad_a() {
    let a_data: Vec<f32> = (0..12).map(|i| i as f32 * 0.1 - 0.3).collect();
    let b_data: Vec<f32> = (0..12).map(|i| i as f32 * 0.08 + 0.1).collect();

    let a = var_from(a_data.clone(), &[2, 2, 3]); // B=2, M=2, K=3
    let b = var_from(b_data.clone(), &[2, 3, 2]); // B=2, K=3, N=2
    let ta = Arc::new(TrackedTensor::from_var(&a).unwrap());
    let tb = Arc::new(TrackedTensor::from_var(&b).unwrap());
    let y = ta.matmul(&tb).unwrap();
    // Use sum-of-squares loss for non-trivial gradients
    let loss = y
        .sqr()
        .unwrap()
        .sum_keepdim(0)
        .unwrap()
        .sum_keepdim(1)
        .unwrap()
        .sum_keepdim(2)
        .unwrap();
    let grads = backward(&loss).unwrap();

    let ga = grad_vec(&grads, &a);
    let num = numerical_grad(&a_data, 1e-3, |d| {
        let at = DynTensor::from_vec(d, &[2, 2, 3], &cpu()).unwrap();
        let bt = DynTensor::from_vec(b_data.clone(), &[2, 3, 2], &cpu()).unwrap();
        let y = at.matmul(&bt).unwrap();
        y.to_flat_vec::<f32>()
            .unwrap()
            .iter()
            .map(|&v| f64::from(v) * f64::from(v))
            .sum()
    });
    assert_grad_close(&ga, &num, 1e-2, "batched_matmul_fd_a");
}

#[test]
fn test_batched_matmul_backward_fd_grad_b() {
    let a_data: Vec<f32> = (0..12).map(|i| i as f32 * 0.1 - 0.3).collect();
    let b_data: Vec<f32> = (0..12).map(|i| i as f32 * 0.08 + 0.1).collect();

    let a = var_from(a_data.clone(), &[2, 2, 3]);
    let b = var_from(b_data.clone(), &[2, 3, 2]);
    let ta = Arc::new(TrackedTensor::from_var(&a).unwrap());
    let tb = Arc::new(TrackedTensor::from_var(&b).unwrap());
    let y = ta.matmul(&tb).unwrap();
    let loss = y
        .sqr()
        .unwrap()
        .sum_keepdim(0)
        .unwrap()
        .sum_keepdim(1)
        .unwrap()
        .sum_keepdim(2)
        .unwrap();
    let grads = backward(&loss).unwrap();

    let gb = grad_vec(&grads, &b);
    let num = numerical_grad(&b_data, 1e-3, |d| {
        let at = DynTensor::from_vec(a_data.clone(), &[2, 2, 3], &cpu()).unwrap();
        let bt = DynTensor::from_vec(d, &[2, 3, 2], &cpu()).unwrap();
        let y = at.matmul(&bt).unwrap();
        y.to_flat_vec::<f32>()
            .unwrap()
            .iter()
            .map(|&v| f64::from(v) * f64::from(v))
            .sum()
    });
    assert_grad_close(&gb, &num, 1e-2, "batched_matmul_fd_b");
}

// ── Transpose + matmul gradient ────────────────────────────────────────

#[test]
fn test_transpose_matmul_backward_fd() {
    // Compute A^T @ B where A is [3, 2] transposed to [2, 3], B is [3, 4]
    // This tests the interaction of transpose and matmul backward rules.
    let a_data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let b_data: Vec<f32> = (0..12).map(|i| i as f32 * 0.1 - 0.3).collect();

    let a = var_from(a_data.clone(), &[3, 2]);
    let b = var_from(b_data.clone(), &[3, 4]);
    let ta = Arc::new(TrackedTensor::from_var(&a).unwrap());
    let tb = Arc::new(TrackedTensor::from_var(&b).unwrap());
    // y = A^T @ B: [2, 3] @ [3, 4] = [2, 4]
    let at = ta.transpose(0, 1).unwrap();
    let y = at.matmul(&tb).unwrap();
    let loss = y
        .sqr()
        .unwrap()
        .sum_keepdim(0)
        .unwrap()
        .sum_keepdim(1)
        .unwrap();
    let grads = backward(&loss).unwrap();

    let ga = grads.get(&a).unwrap();
    assert_eq!(ga.dims(), &[3, 2], "grad_a shape after transpose+matmul");

    let g_a_vec = grad_vec(&grads, &a);
    let num = numerical_grad(&a_data, 1e-3, |d| {
        let at = DynTensor::from_vec(d, &[3, 2], &cpu()).unwrap();
        let bt = DynTensor::from_vec(b_data.clone(), &[3, 4], &cpu()).unwrap();
        let y = at.transpose(0, 1).unwrap().matmul(&bt).unwrap();
        y.to_flat_vec::<f32>()
            .unwrap()
            .iter()
            .map(|&v| f64::from(v) * f64::from(v))
            .sum()
    });
    assert_grad_close(&g_a_vec, &num, 1e-2, "transpose_matmul_fd_a");
}

// ── Matmul with identity matrix ────────────────────────────────────────

#[test]
fn test_matmul_identity_gradient_passthrough() {
    // A @ I = A, so d(sum(A @ I)) / dA = I (all ones for sum loss)
    let a_data = vec![1.0, 2.0, 3.0, 4.0];
    let identity = vec![1.0, 0.0, 0.0, 1.0];

    let a = var_from(a_data, &[2, 2]);
    let i_mat = var_from(identity, &[2, 2]);
    let ta = Arc::new(TrackedTensor::from_var(&a).unwrap());
    let ti = Arc::new(TrackedTensor::from_var(&i_mat).unwrap());
    let y = ta.matmul(&ti).unwrap();
    let loss = y.sum_keepdim(0).unwrap().sum_keepdim(1).unwrap();
    let grads = backward(&loss).unwrap();

    let ga = grad_vec(&grads, &a);
    // d/dA sum(A @ I) = sum over output, grad propagates through I^T = I
    // Each element of A contributes exactly 1.0 to the sum
    assert_eq!(ga, vec![1.0, 1.0, 1.0, 1.0]);
}

// ── Square matmul analytical gradient check ────────────────────────────

#[test]
fn test_matmul_backward_analytical_values() {
    // For loss = sum(A @ B), grad_A = ones @ B^T, grad_B = A^T @ ones
    let a_data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]; // [2, 3]
    let b_data = vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6]; // [3, 2]

    let a = var_from(a_data, &[2, 3]);
    let b = var_from(b_data, &[3, 2]);
    let ta = Arc::new(TrackedTensor::from_var(&a).unwrap());
    let tb = Arc::new(TrackedTensor::from_var(&b).unwrap());
    let y = ta.matmul(&tb).unwrap();
    let loss = y.sum_keepdim(0).unwrap().sum_keepdim(1).unwrap();
    let grads = backward(&loss).unwrap();

    let ga = grad_vec(&grads, &a);
    let gb = grad_vec(&grads, &b);

    // grad_A = ones([2,2]) @ B^T([2,3])
    // B^T = [[0.1, 0.3, 0.5], [0.2, 0.4, 0.6]]
    // ones @ B^T = [[0.3, 0.7, 1.1], [0.3, 0.7, 1.1]]
    let expected_ga = [0.3, 0.7, 1.1, 0.3, 0.7, 1.1];
    for (i, (&a_val, &e_val)) in ga.iter().zip(expected_ga.iter()).enumerate() {
        assert!(
            (a_val - e_val).abs() < 1e-5,
            "grad_a[{i}]: expected={e_val}, got={a_val}"
        );
    }

    // grad_B = A^T([3,2]) @ ones([2,2])
    // A^T = [[1,4],[2,5],[3,6]]
    // A^T @ ones = [[5,5],[7,7],[9,9]]
    let expected_gb = [5.0, 5.0, 7.0, 7.0, 9.0, 9.0];
    for (i, (&b_val, &e_val)) in gb.iter().zip(expected_gb.iter()).enumerate() {
        assert!(
            (b_val - e_val).abs() < 1e-5,
            "grad_b[{i}]: expected={e_val}, got={b_val}"
        );
    }
}

// ── Matmul chain rule: d/dA loss(f(A @ B)) ────────────────────────────

#[test]
fn test_matmul_through_activation_fd() {
    // f(A, B) = sum(relu(A @ B)), check gradient of A
    let a_data: Vec<f32> = vec![0.5, -0.3, 1.2, 0.8, -0.6, 0.4];
    let b_data: Vec<f32> = vec![1.0, -0.5, 0.3, 0.7, -0.2, 0.8];

    let a = var_from(a_data.clone(), &[2, 3]);
    let b = var_from(b_data.clone(), &[3, 2]);
    let ta = Arc::new(TrackedTensor::from_var(&a).unwrap());
    let tb = Arc::new(TrackedTensor::from_var(&b).unwrap());
    let y = ta.matmul(&tb).unwrap().relu().unwrap();
    let loss = y.sum_keepdim(0).unwrap().sum_keepdim(1).unwrap();
    let grads = backward(&loss).unwrap();

    let ga = grad_vec(&grads, &a);
    let num = numerical_grad(&a_data, 1e-3, |d| {
        let at = DynTensor::from_vec(d, &[2, 3], &cpu()).unwrap();
        let bt = DynTensor::from_vec(b_data.clone(), &[3, 2], &cpu()).unwrap();
        let y = at.matmul(&bt).unwrap().relu().unwrap();
        y.to_flat_vec::<f32>()
            .unwrap()
            .iter()
            .map(|&v| f64::from(v))
            .sum()
    });
    assert_grad_close(&ga, &num, 1e-2, "matmul_relu_fd_a");
}

// ── Matmul self-multiply (A @ A^T) gradient ───────────────────────────

#[test]
fn test_matmul_self_transpose_fd() {
    // f(A) = sum((A @ A^T)^2), tests fan-out + transpose interaction
    let a_data: Vec<f32> = vec![1.0, 0.5, -0.3, 0.8, 0.2, -0.7];

    let a = var_from(a_data.clone(), &[2, 3]);
    let ta = Arc::new(TrackedTensor::from_var(&a).unwrap());
    let at = ta.transpose(0, 1).unwrap();
    let y = ta.matmul(&at).unwrap(); // [2, 2]
    let loss = y
        .sqr()
        .unwrap()
        .sum_keepdim(0)
        .unwrap()
        .sum_keepdim(1)
        .unwrap();
    let grads = backward(&loss).unwrap();

    let ga = grad_vec(&grads, &a);
    let num = numerical_grad(&a_data, 1e-3, |d| {
        let t = DynTensor::from_vec(d, &[2, 3], &cpu()).unwrap();
        let y = t.matmul(&t.transpose(0, 1).unwrap()).unwrap();
        y.to_flat_vec::<f32>()
            .unwrap()
            .iter()
            .map(|&v| f64::from(v) * f64::from(v))
            .sum()
    });
    assert_grad_close(&ga, &num, 5e-2, "matmul_self_transpose_fd");
}

// ── 4D batched matmul shapes (attention-style) ─────────────────────────

#[test]
fn test_matmul_4d_backward_shapes() {
    // [B, H, S_q, d_k] x [B, H, d_k, S_kv] -> [B, H, S_q, S_kv]
    let a = var_from(vec![0.1; 48], &[1, 2, 3, 8]); // B=1, H=2, S_q=3, d_k=8
    let b = var_from(vec![0.1; 80], &[1, 2, 8, 5]); // B=1, H=2, d_k=8, S_kv=5
    let ta = Arc::new(TrackedTensor::from_var(&a).unwrap());
    let tb = Arc::new(TrackedTensor::from_var(&b).unwrap());
    let y = ta.matmul(&tb).unwrap();
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

    let ga = grads.get(&a).unwrap();
    let gb = grads.get(&b).unwrap();
    assert_eq!(ga.dims(), &[1, 2, 3, 8], "4D grad_a shape");
    assert_eq!(gb.dims(), &[1, 2, 8, 5], "4D grad_b shape");
}

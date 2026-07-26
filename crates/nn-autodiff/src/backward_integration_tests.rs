// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Integration tests for backward rules covering elementwise, reduction,
//! matmul, shape, softmax, layer_norm, and chain-rule compositions.
//!
//! Each test creates small tensors, runs forward + backward, and verifies:
//! 1. Gradient shape matches input shape.
//! 2. Gradient values match hand-computed expectations (where tractable).
//! 3. Finite-difference cross-check for non-trivial ops.

use std::sync::Arc;

use nn_core::dyn_tensor::DynTensor;
use nn_core::test_utils::cpu;

use crate::grad::backward;
use crate::tracked::TrackedTensor;
use crate::var::Var;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn var_from(data: Vec<f32>, shape: &[usize]) -> Var {
    Var::new(DynTensor::from_vec(data, shape, &cpu()).unwrap())
}

fn const_from(data: Vec<f32>, shape: &[usize]) -> Arc<TrackedTensor> {
    Arc::new(TrackedTensor::from_tensor(
        DynTensor::from_vec(data, shape, &cpu()).unwrap(),
    ))
}

fn tracked(v: &Var) -> Arc<TrackedTensor> {
    Arc::new(TrackedTensor::from_var(v).unwrap())
}

fn grad_vec(grads: &crate::grad::GradStore, v: &Var) -> Vec<f32> {
    grads.get(v).unwrap().to_flat_vec::<f32>().unwrap()
}

/// Make a scalar loss from an arbitrary-shaped tensor by sum-of-all-elements.
fn scalar_loss(t: &Arc<TrackedTensor>) -> Arc<TrackedTensor> {
    let mut result = Arc::clone(t);
    // Sum each dimension from last to first, keepdim, so we end up with shape [1, 1, ..., 1].
    for d in (0..result.tensor().rank()).rev() {
        result = result.sum_keepdim(d).unwrap();
    }
    result
}

fn assert_close(actual: &[f32], expected: &[f32], tol: f32, msg: &str) {
    assert_eq!(
        actual.len(),
        expected.len(),
        "{msg}: length mismatch: got {}, expected {}",
        actual.len(),
        expected.len()
    );
    for (i, (&a, &e)) in actual.iter().zip(expected.iter()).enumerate() {
        assert!(
            (a - e).abs() < tol,
            "{msg}[{i}]: got {a}, expected {e}, diff={}",
            (a - e).abs()
        );
    }
}

// ===========================================================================
// Elementwise ops
// ===========================================================================

#[test]
fn test_grad_add() {
    // loss = sum(a + b), d/da = 1, d/db = 1
    let a = var_from(vec![1.0, 2.0, 3.0], &[3]);
    let b = var_from(vec![4.0, 5.0, 6.0], &[3]);
    let ta = tracked(&a);
    let tb = tracked(&b);
    let y = ta.add(&tb).unwrap();
    let loss = scalar_loss(&y);
    let grads = backward(&loss).unwrap();

    assert_eq!(grads.get(&a).unwrap().dims(), &[3]);
    assert_close(&grad_vec(&grads, &a), &[1.0, 1.0, 1.0], 1e-6, "d/da");
    assert_close(&grad_vec(&grads, &b), &[1.0, 1.0, 1.0], 1e-6, "d/db");
}

#[test]
fn test_grad_sub() {
    // loss = sum(a - b), d/da = 1, d/db = -1
    let a = var_from(vec![5.0, 3.0], &[2]);
    let b = var_from(vec![1.0, 2.0], &[2]);
    let ta = tracked(&a);
    let tb = tracked(&b);
    let y = ta.sub(&tb).unwrap();
    let loss = scalar_loss(&y);
    let grads = backward(&loss).unwrap();

    assert_close(&grad_vec(&grads, &a), &[1.0, 1.0], 1e-6, "d/da");
    assert_close(&grad_vec(&grads, &b), &[-1.0, -1.0], 1e-6, "d/db");
}

#[test]
fn test_grad_mul() {
    // loss = sum(a * b), d/da_i = b_i, d/db_i = a_i
    let a = var_from(vec![2.0, 3.0], &[2]);
    let b = var_from(vec![4.0, 5.0], &[2]);
    let ta = tracked(&a);
    let tb = tracked(&b);
    let y = ta.mul(&tb).unwrap();
    let loss = scalar_loss(&y);
    let grads = backward(&loss).unwrap();

    assert_close(&grad_vec(&grads, &a), &[4.0, 5.0], 1e-6, "d/da");
    assert_close(&grad_vec(&grads, &b), &[2.0, 3.0], 1e-6, "d/db");
}

#[test]
fn test_grad_div() {
    // loss = sum(a / b)
    // d(loss)/d(a_i) = 1/b_i, d(loss)/d(b_i) = -a_i / b_i^2
    let a = var_from(vec![6.0, 8.0], &[2]);
    let b = var_from(vec![2.0, 4.0], &[2]);
    let ta = tracked(&a);
    let tb = tracked(&b);
    let y = ta.div(&tb).unwrap();
    let loss = scalar_loss(&y);
    let grads = backward(&loss).unwrap();

    // d/da = [1/2, 1/4] = [0.5, 0.25]
    assert_close(&grad_vec(&grads, &a), &[0.5, 0.25], 1e-5, "d/da");
    // d/db = [-6/4, -8/16] = [-1.5, -0.5]
    assert_close(&grad_vec(&grads, &b), &[-1.5, -0.5], 1e-5, "d/db");
}

#[test]
fn test_grad_neg() {
    // loss = sum(-x), d/dx = -1
    let x = var_from(vec![1.0, -2.0, 3.0], &[3]);
    let tx = tracked(&x);
    let y = tx.neg().unwrap();
    let loss = scalar_loss(&y);
    let grads = backward(&loss).unwrap();

    assert_close(&grad_vec(&grads, &x), &[-1.0, -1.0, -1.0], 1e-6, "d/dx");
}

#[test]
fn test_grad_exp() {
    // loss = sum(exp(x)), d/dx_i = exp(x_i)
    let x = var_from(vec![0.0, 1.0, -1.0], &[3]);
    let tx = tracked(&x);
    let y = tx.exp().unwrap();
    let loss = scalar_loss(&y);
    let grads = backward(&loss).unwrap();

    let expected = [0.0_f32.exp(), 1.0_f32.exp(), (-1.0_f32).exp()];
    assert_close(&grad_vec(&grads, &x), &expected, 1e-5, "d/dx");
}

#[test]
fn test_grad_log() {
    // loss = sum(log(x)), d/dx_i = 1/x_i
    let x = var_from(vec![1.0, 2.0, 0.5], &[3]);
    let tx = tracked(&x);
    let y = tx.log().unwrap();
    let loss = scalar_loss(&y);
    let grads = backward(&loss).unwrap();

    assert_close(&grad_vec(&grads, &x), &[1.0, 0.5, 2.0], 1e-5, "d/dx");
}

#[test]
fn test_grad_sqrt() {
    // loss = sum(sqrt(x)), d/dx_i = 1/(2*sqrt(x_i))
    let x = var_from(vec![4.0, 9.0, 1.0], &[3]);
    let tx = tracked(&x);
    let y = tx.sqrt().unwrap();
    let loss = scalar_loss(&y);
    let grads = backward(&loss).unwrap();

    // 1/(2*2)=0.25, 1/(2*3)~=0.1667, 1/(2*1)=0.5
    let expected = [0.25, 1.0 / 6.0, 0.5];
    assert_close(&grad_vec(&grads, &x), &expected, 1e-4, "d/dx");
}

#[test]
fn test_grad_sin() {
    // loss = sum(sin(x)), d/dx_i = cos(x_i)
    let x = var_from(
        vec![0.0, std::f32::consts::FRAC_PI_2, std::f32::consts::PI],
        &[3],
    );
    let tx = tracked(&x);
    let y = tx.sin().unwrap();
    let loss = scalar_loss(&y);
    let grads = backward(&loss).unwrap();

    let expected = [
        0.0_f32.cos(),
        std::f32::consts::FRAC_PI_2.cos(),
        std::f32::consts::PI.cos(),
    ];
    assert_close(&grad_vec(&grads, &x), &expected, 1e-5, "d/dx");
}

#[test]
fn test_grad_cos() {
    // loss = sum(cos(x)), d/dx_i = -sin(x_i)
    let x = var_from(
        vec![0.0, std::f32::consts::FRAC_PI_2, std::f32::consts::PI],
        &[3],
    );
    let tx = tracked(&x);
    let y = tx.cos().unwrap();
    let loss = scalar_loss(&y);
    let grads = backward(&loss).unwrap();

    let expected = [
        -(0.0_f32.sin()),
        -(std::f32::consts::FRAC_PI_2.sin()),
        -(std::f32::consts::PI.sin()),
    ];
    assert_close(&grad_vec(&grads, &x), &expected, 1e-5, "d/dx");
}

#[test]
fn test_grad_tanh() {
    // loss = sum(tanh(x)), d/dx_i = 1 - tanh(x_i)^2
    let x = var_from(vec![0.0, 1.0, -0.5], &[3]);
    let tx = tracked(&x);
    let y = tx.tanh().unwrap();
    let loss = scalar_loss(&y);
    let grads = backward(&loss).unwrap();

    let expected: Vec<f32> = [0.0_f32, 1.0, -0.5]
        .iter()
        .map(|&v| 1.0 - v.tanh().powi(2))
        .collect();
    assert_close(&grad_vec(&grads, &x), &expected, 1e-5, "d/dx");
}

#[test]
fn test_grad_sigmoid() {
    // loss = sum(sigmoid(x)), d/dx_i = sigmoid(x_i) * (1 - sigmoid(x_i))
    let x = var_from(vec![0.0, 2.0, -2.0], &[3]);
    let tx = tracked(&x);
    let y = tx.sigmoid().unwrap();
    let loss = scalar_loss(&y);
    let grads = backward(&loss).unwrap();

    fn sig(x: f32) -> f32 {
        1.0 / (1.0 + (-x).exp())
    }
    let expected: Vec<f32> = [0.0_f32, 2.0, -2.0]
        .iter()
        .map(|&v| sig(v) * (1.0 - sig(v)))
        .collect();
    assert_close(&grad_vec(&grads, &x), &expected, 1e-5, "d/dx");
}

#[test]
fn test_grad_relu() {
    // loss = sum(relu(x)), d/dx_i = 1 if x_i > 0, 0 otherwise
    let x = var_from(vec![-2.0, 0.0, 3.0, -1.0], &[4]);
    let tx = tracked(&x);
    let y = tx.relu().unwrap();
    let loss = scalar_loss(&y);
    let grads = backward(&loss).unwrap();

    // x >= 0: [0,1,1,0] (ge includes 0)
    assert_close(&grad_vec(&grads, &x), &[0.0, 1.0, 1.0, 0.0], 1e-6, "d/dx");
}

#[test]
fn test_grad_gelu_fd() {
    // Finite-difference check for GELU
    let vals = vec![0.5, -0.3, 1.2, -1.5];
    let x = var_from(vals.clone(), &[4]);
    let tx = tracked(&x);
    let y = tx.gelu().unwrap();
    let loss = scalar_loss(&y);
    let grads = backward(&loss).unwrap();
    let analytical = grad_vec(&grads, &x);

    let eps = 1e-4_f32;
    for i in 0..vals.len() {
        let mut plus = vals.clone();
        let mut minus = vals.clone();
        plus[i] += eps;
        minus[i] -= eps;

        let fwd = |d: Vec<f32>| -> f64 {
            let t = DynTensor::from_vec(d, &[4], &cpu()).unwrap();
            let g = t.gelu().unwrap();
            g.to_flat_vec::<f32>()
                .unwrap()
                .iter()
                .map(|&v| f64::from(v))
                .sum::<f64>()
        };
        let numerical = ((fwd(plus) - fwd(minus)) / (2.0 * f64::from(eps))) as f32;
        assert!(
            (analytical[i] - numerical).abs() < 0.01,
            "gelu grad[{i}]: analytical={}, numerical={}, diff={}",
            analytical[i],
            numerical,
            (analytical[i] - numerical).abs()
        );
    }
}

#[test]
fn test_grad_silu_fd() {
    // Finite-difference check for SiLU: silu(x) = x * sigmoid(x)
    let vals = vec![0.5, -0.3, 1.2, -1.5];
    let x = var_from(vals.clone(), &[4]);
    let tx = tracked(&x);
    let y = tx.silu().unwrap();
    let loss = scalar_loss(&y);
    let grads = backward(&loss).unwrap();
    let analytical = grad_vec(&grads, &x);

    let eps = 1e-4_f32;
    for i in 0..vals.len() {
        let mut plus = vals.clone();
        let mut minus = vals.clone();
        plus[i] += eps;
        minus[i] -= eps;

        let fwd = |d: Vec<f32>| -> f64 {
            let t = DynTensor::from_vec(d, &[4], &cpu()).unwrap();
            let s = t.silu().unwrap();
            s.to_flat_vec::<f32>()
                .unwrap()
                .iter()
                .map(|&v| f64::from(v))
                .sum::<f64>()
        };
        let numerical = ((fwd(plus) - fwd(minus)) / (2.0 * f64::from(eps))) as f32;
        assert!(
            (analytical[i] - numerical).abs() < 0.01,
            "silu grad[{i}]: analytical={}, numerical={}, diff={}",
            analytical[i],
            numerical,
            (analytical[i] - numerical).abs()
        );
    }
}

// ===========================================================================
// Reduction ops
// ===========================================================================

#[test]
fn test_grad_sum_keepdim_axis0() {
    // x = [[1,2],[3,4]], loss = sum_keepdim(x, dim=0) => [[4,6]]
    // Then sum all to scalar. d(sum)/dx = all 1s.
    let x = var_from(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]);
    let tx = tracked(&x);
    let s = tx.sum_keepdim(0).unwrap(); // shape [1, 2]
    let loss = scalar_loss(&s);
    let grads = backward(&loss).unwrap();

    assert_eq!(grads.get(&x).unwrap().dims(), &[2, 2]);
    assert_close(&grad_vec(&grads, &x), &[1.0, 1.0, 1.0, 1.0], 1e-6, "d/dx");
}

#[test]
fn test_grad_sum_keepdim_axis1() {
    // x = [[1,2,3],[4,5,6]], sum_keepdim(dim=1) => [[6],[15]]
    // Then scalar_loss => sum = 21. d/dx = all 1s.
    let x = var_from(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
    let tx = tracked(&x);
    let s = tx.sum_keepdim(1).unwrap(); // shape [2, 1]
    let loss = scalar_loss(&s);
    let grads = backward(&loss).unwrap();

    assert_eq!(grads.get(&x).unwrap().dims(), &[2, 3]);
    assert_close(
        &grad_vec(&grads, &x),
        &[1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        1e-6,
        "d/dx",
    );
}

#[test]
fn test_grad_mean_keepdim() {
    // x = [2, 4, 6], mean_keepdim(0) = [4]
    // loss = mean. d/dx_i = 1/3
    let x = var_from(vec![2.0, 4.0, 6.0], &[3]);
    let tx = tracked(&x);
    let m = tx.mean_keepdim(0).unwrap();
    let loss = scalar_loss(&m);
    let grads = backward(&loss).unwrap();

    assert_eq!(grads.get(&x).unwrap().dims(), &[3]);
    let expected = [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0];
    assert_close(&grad_vec(&grads, &x), &expected, 1e-5, "d/dx");
}

#[test]
fn test_grad_mean_keepdim_2d() {
    // x shape [2, 3], mean_keepdim(dim=1) => [2, 1], then scalar_loss
    // d/dx = 1/3 for each element (since mean is over dim=1 which has size 3)
    let x = var_from(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
    let tx = tracked(&x);
    let m = tx.mean_keepdim(1).unwrap();
    let loss = scalar_loss(&m);
    let grads = backward(&loss).unwrap();

    assert_eq!(grads.get(&x).unwrap().dims(), &[2, 3]);
    let third = 1.0 / 3.0_f32;
    assert_close(
        &grad_vec(&grads, &x),
        &[third, third, third, third, third, third],
        1e-5,
        "d/dx",
    );
}

// ===========================================================================
// MatMul
// ===========================================================================

#[test]
fn test_grad_matmul_square() {
    // A [2,2] @ B [2,2] => C [2,2], loss = sum(C)
    // d(loss)/dA = ones[2,2] @ B^T, d(loss)/dB = A^T @ ones[2,2]
    let a = var_from(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]);
    let b = var_from(vec![5.0, 6.0, 7.0, 8.0], &[2, 2]);
    let ta = tracked(&a);
    let tb = tracked(&b);
    let c = ta.matmul(&tb).unwrap();
    let loss = scalar_loss(&c);
    let grads = backward(&loss).unwrap();

    assert_eq!(grads.get(&a).unwrap().dims(), &[2, 2]);
    assert_eq!(grads.get(&b).unwrap().dims(), &[2, 2]);

    // d/dA = ones @ B^T = [[1,1],[1,1]] @ [[5,7],[6,8]] = [[11,15],[11,15]]
    // Wait: ones @ B^T means grad @ B^T. grad = ones [2,2].
    // B^T = [[5,7],[6,8]], ones @ B^T = [[5+6, 7+8],[5+6, 7+8]] = [[11,15],[11,15]]
    assert_close(
        &grad_vec(&grads, &a),
        &[11.0, 15.0, 11.0, 15.0],
        1e-4,
        "d/dA",
    );

    // d/dB = A^T @ ones = [[1,3],[2,4]] @ [[1,1],[1,1]] = [[4,4],[6,6]]
    assert_close(&grad_vec(&grads, &b), &[4.0, 4.0, 6.0, 6.0], 1e-4, "d/dB");
}

#[test]
fn test_grad_matmul_rectangular() {
    // A [1,3] @ B [3,2] => C [1,2], loss = sum(C)
    let a = var_from(vec![1.0, 2.0, 3.0], &[1, 3]);
    let b = var_from(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[3, 2]);
    let ta = tracked(&a);
    let tb = tracked(&b);
    let c = ta.matmul(&tb).unwrap();
    let loss = scalar_loss(&c);
    let grads = backward(&loss).unwrap();

    assert_eq!(grads.get(&a).unwrap().dims(), &[1, 3]);
    assert_eq!(grads.get(&b).unwrap().dims(), &[3, 2]);

    // d/dA = ones[1,2] @ B^T[2,3]
    // B^T = [[1,3,5],[2,4,6]]
    // [1,1] @ B^T = [1+2, 3+4, 5+6] = [3, 7, 11]
    assert_close(&grad_vec(&grads, &a), &[3.0, 7.0, 11.0], 1e-4, "d/dA");

    // d/dB = A^T[3,1] @ ones[1,2]
    // A^T = [[1],[2],[3]], ones = [[1,1]]
    // A^T @ ones = [[1,1],[2,2],[3,3]]
    assert_close(
        &grad_vec(&grads, &b),
        &[1.0, 1.0, 2.0, 2.0, 3.0, 3.0],
        1e-4,
        "d/dB",
    );
}

#[test]
fn test_grad_matmul_batched() {
    // A [2,1,2] @ B [2,2,1] => C [2,1,1], loss = sum(C)
    let a = var_from(vec![1.0, 2.0, 3.0, 4.0], &[2, 1, 2]);
    let b = var_from(vec![5.0, 6.0, 7.0, 8.0], &[2, 2, 1]);
    let ta = tracked(&a);
    let tb = tracked(&b);
    let c = ta.matmul(&tb).unwrap(); // [2, 1, 1]
    let loss = scalar_loss(&c);
    let grads = backward(&loss).unwrap();

    assert_eq!(grads.get(&a).unwrap().dims(), &[2, 1, 2]);
    assert_eq!(grads.get(&b).unwrap().dims(), &[2, 2, 1]);

    // batch 0: grad[0] = ones[1,1] @ B[0]^T = [[5, 6]]
    // batch 1: grad[1] = ones[1,1] @ B[1]^T = [[7, 8]]
    assert_close(
        &grad_vec(&grads, &a),
        &[5.0, 6.0, 7.0, 8.0],
        1e-4,
        "d/dA batched",
    );

    // batch 0: A[0]^T @ ones = [[1],[2]] @ [[1]] = [[1],[2]]
    // batch 1: A[1]^T @ ones = [[3],[4]] @ [[1]] = [[3],[4]]
    assert_close(
        &grad_vec(&grads, &b),
        &[1.0, 2.0, 3.0, 4.0],
        1e-4,
        "d/dB batched",
    );
}

// ===========================================================================
// Transpose
// ===========================================================================

#[test]
fn test_grad_transpose_shape() {
    // x [2,3], transpose(0,1) => [3,2], loss = sum
    // d/dx should be [2,3] all ones
    let x = var_from(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
    let tx = tracked(&x);
    let t = tx.transpose(0, 1).unwrap(); // [3, 2]
    let loss = scalar_loss(&t);
    let grads = backward(&loss).unwrap();

    assert_eq!(grads.get(&x).unwrap().dims(), &[2, 3]);
    assert_close(
        &grad_vec(&grads, &x),
        &[1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        1e-6,
        "d/dx",
    );
}

#[test]
fn test_grad_transpose_matmul() {
    // y = x^T @ x where x is [3,2], x^T is [2,3], result is [2,2]
    // This verifies gradients flow correctly through transpose into matmul.
    let x = var_from(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[3, 2]);
    let tx = tracked(&x);
    let xt = tx.transpose(0, 1).unwrap(); // [2, 3]
    let y = xt.matmul(&tx).unwrap(); // [2, 2]
    let loss = scalar_loss(&y);
    let grads = backward(&loss).unwrap();

    // Gradient shape must match input shape
    assert_eq!(grads.get(&x).unwrap().dims(), &[3, 2]);
}

// ===========================================================================
// Reshape / view
// ===========================================================================

#[test]
fn test_grad_reshape() {
    // x [2,3] reshape to [6], loss = sum
    // gradient flows back through reshape to original shape
    let x = var_from(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
    let tx = tracked(&x);
    let flat = tx.reshape(&[6]).unwrap();
    let loss = scalar_loss(&flat);
    let grads = backward(&loss).unwrap();

    assert_eq!(grads.get(&x).unwrap().dims(), &[2, 3]);
    assert_close(
        &grad_vec(&grads, &x),
        &[1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        1e-6,
        "d/dx",
    );
}

#[test]
fn test_grad_reshape_nonlinear() {
    // x [6] -> reshape [2,3] -> sqr -> sum
    // d/dx_i = 2 * x_i (through reshape)
    let vals = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let x = var_from(vals.clone(), &[6]);
    let tx = tracked(&x);
    let reshaped = tx.reshape(&[2, 3]).unwrap();
    let sq = reshaped.sqr().unwrap();
    let loss = scalar_loss(&sq);
    let grads = backward(&loss).unwrap();

    assert_eq!(grads.get(&x).unwrap().dims(), &[6]);
    let expected: Vec<f32> = vals.iter().map(|&v| 2.0 * v).collect();
    assert_close(&grad_vec(&grads, &x), &expected, 1e-5, "d/dx");
}

// ===========================================================================
// Softmax
// ===========================================================================

#[test]
fn test_grad_softmax_shape() {
    // Softmax gradient should preserve input shape
    let x = var_from(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
    let tx = tracked(&x);
    let s = tx.softmax(1).unwrap(); // softmax over last dim
    let loss = scalar_loss(&s);
    let grads = backward(&loss).unwrap();

    assert_eq!(grads.get(&x).unwrap().dims(), &[2, 3]);
}

#[test]
fn test_grad_softmax_analytical() {
    // For loss = sum(softmax(x)), d(loss)/dx = 0 because
    // sum(softmax(x)) = 1 for any x, so the gradient is zero.
    let x = var_from(vec![1.0, 2.0, 3.0], &[1, 3]);
    let tx = tracked(&x);
    let s = tx.softmax(1).unwrap();
    let loss = scalar_loss(&s);
    let grads = backward(&loss).unwrap();

    // sum(softmax) is always 1.0, so gradients should be ~0
    let g = grad_vec(&grads, &x);
    for (i, &v) in g.iter().enumerate() {
        assert!(v.abs() < 1e-5, "softmax grad[{i}] should be ~0, got {v}");
    }
}

#[test]
fn test_grad_softmax_weighted_fd() {
    // Use a weighted loss: loss = sum(w * softmax(x)) to get non-zero gradients.
    // Verify against finite differences.
    let vals = vec![1.0_f32, 2.0, 3.0];
    let x = var_from(vals.clone(), &[1, 3]);
    let tx = tracked(&x);
    let s = tx.softmax(1).unwrap();
    // Multiply by weights to get non-trivial gradient
    let w = const_from(vec![1.0, 0.0, 0.0], &[1, 3]);
    let ws = s.mul(&w).unwrap();
    let loss = scalar_loss(&ws);
    let grads = backward(&loss).unwrap();
    let analytical = grad_vec(&grads, &x);

    let eps = 1e-4_f32;
    for i in 0..3 {
        let mut plus = vals.clone();
        let mut minus = vals.clone();
        plus[i] += eps;
        minus[i] -= eps;

        let fwd = |d: Vec<f32>| -> f64 {
            let t = DynTensor::from_vec(d, &[1, 3], &cpu()).unwrap();
            let sm = t.softmax(1).unwrap();
            // Extract first element (weight=[1,0,0])
            let flat = sm.to_flat_vec::<f32>().unwrap();
            f64::from(flat[0])
        };
        let numerical = ((fwd(plus) - fwd(minus)) / (2.0 * f64::from(eps))) as f32;
        assert!(
            (analytical[i] - numerical).abs() < 1e-3,
            "softmax fd grad[{i}]: analytical={}, numerical={}, diff={}",
            analytical[i],
            numerical,
            (analytical[i] - numerical).abs()
        );
    }
}

// ===========================================================================
// LayerNorm
// ===========================================================================

#[test]
fn test_grad_layer_norm_shape() {
    // LayerNorm gradient should preserve all input shapes
    let x = var_from(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
    let w = var_from(vec![1.0, 1.0, 1.0], &[3]);
    let b = var_from(vec![0.0, 0.0, 0.0], &[3]);
    let tx = tracked(&x);
    let tw = tracked(&w);
    let tb = tracked(&b);
    let y = tx.layer_norm(&tw, &tb, 1e-5).unwrap();
    let loss = scalar_loss(&y);
    let grads = backward(&loss).unwrap();

    // Input gradient shape matches input
    assert_eq!(grads.get(&x).unwrap().dims(), &[2, 3]);
    // Weight gradient shape matches weight
    assert_eq!(grads.get(&w).unwrap().dims(), &[3]);
    // Bias gradient shape matches bias
    assert_eq!(grads.get(&b).unwrap().dims(), &[3]);
}

#[test]
fn test_grad_layer_norm_bias_gradient() {
    // For loss = sum(layer_norm(x)), the bias gradient = sum over batch dim of
    // the upstream gradient. With sum loss, upstream grad of the normalized
    // output is all ones projected back; bias grad = number-of-batch-samples.
    let x = var_from(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
    let w = var_from(vec![1.0, 1.0, 1.0], &[3]);
    let b = var_from(vec![0.0, 0.0, 0.0], &[3]);
    let tx = tracked(&x);
    let tw = tracked(&w);
    let tb = tracked(&b);
    let y = tx.layer_norm(&tw, &tb, 1e-5).unwrap();
    let loss = scalar_loss(&y);
    let grads = backward(&loss).unwrap();

    // Bias gradient = d(loss)/d(bias) = sum of upstream gradient over batch dim.
    // Since loss = sum of all elements, bias grad for each feature = batch_size = 2.
    let bias_grad = grad_vec(&grads, &b);
    assert_close(&bias_grad, &[2.0, 2.0, 2.0], 1e-4, "d/dbias");
}

// ===========================================================================
// Chain rule: multi-op compositions
// ===========================================================================

#[test]
fn test_chain_relu_matmul_bias() {
    // loss = sum(relu(x @ w + b))
    // x [1,2], w [2,3], b [1,3]
    let x = var_from(vec![1.0, -1.0], &[1, 2]);
    let w = var_from(vec![1.0, 0.0, -1.0, 0.0, 1.0, 1.0], &[2, 3]);
    let b = var_from(vec![0.0, 0.5, -0.5], &[1, 3]);
    let tx = tracked(&x);
    let tw = tracked(&w);
    let tb = tracked(&b);

    // Forward: xw = x @ w, then + b, then relu
    let xw = tx.matmul(&tw).unwrap(); // [1, 3]
    let xwb = xw.add(&tb).unwrap(); // [1, 3]
    let out = xwb.relu().unwrap(); // [1, 3]
    let loss = scalar_loss(&out);
    let grads = backward(&loss).unwrap();

    // All gradient shapes must match their variable shapes
    assert_eq!(grads.get(&x).unwrap().dims(), &[1, 2]);
    assert_eq!(grads.get(&w).unwrap().dims(), &[2, 3]);
    assert_eq!(grads.get(&b).unwrap().dims(), &[1, 3]);

    // Forward values:
    // x @ w = [1*1+(-1)*0, 1*0+(-1)*1, 1*(-1)+(-1)*1] = [1, -1, -2]
    // + b = [1, -0.5, -2.5]
    // relu = [1, 0, 0]
    // loss = 1
    //
    // relu grad: mask = [1, 0, 0] (only first element positive)
    // d(loss)/d(xwb) = [1, 0, 0]
    // d(loss)/db = [1, 0, 0]
    assert_close(&grad_vec(&grads, &b), &[1.0, 0.0, 0.0], 1e-5, "d/db");
}

#[test]
fn test_chain_sqr_mul_sum() {
    // loss = sum(x^2 * y) for x=[1,2,3], y=[2,3,4]
    // d/dx_i = 2 * x_i * y_i
    // d/dy_i = x_i^2
    let x = var_from(vec![1.0, 2.0, 3.0], &[3]);
    let y = var_from(vec![2.0, 3.0, 4.0], &[3]);
    let tx = tracked(&x);
    let ty = tracked(&y);
    let sq = tx.sqr().unwrap();
    let prod = sq.mul(&ty).unwrap();
    let loss = scalar_loss(&prod);
    let grads = backward(&loss).unwrap();

    // d/dx = [2*1*2, 2*2*3, 2*3*4] = [4, 12, 24]
    assert_close(&grad_vec(&grads, &x), &[4.0, 12.0, 24.0], 1e-4, "d/dx");
    // d/dy = [1, 4, 9]
    assert_close(&grad_vec(&grads, &y), &[1.0, 4.0, 9.0], 1e-4, "d/dy");
}

#[test]
fn test_chain_exp_log_identity() {
    // loss = sum(log(exp(x))) = sum(x), so d/dx = 1
    let x = var_from(vec![0.5, 1.0, 1.5, 2.0], &[4]);
    let tx = tracked(&x);
    let e = tx.exp().unwrap();
    let le = e.log().unwrap();
    let loss = scalar_loss(&le);
    let grads = backward(&loss).unwrap();

    assert_close(&grad_vec(&grads, &x), &[1.0, 1.0, 1.0, 1.0], 1e-4, "d/dx");
}

#[test]
fn test_chain_sigmoid_sum_mean() {
    // loss = mean(sigmoid(x)) where x = [0, 1, -1]
    // d/dx_i = sigmoid(x_i) * (1 - sigmoid(x_i)) / n
    let x = var_from(vec![0.0, 1.0, -1.0], &[3]);
    let tx = tracked(&x);
    let s = tx.sigmoid().unwrap();
    let m = s.mean_keepdim(0).unwrap();
    let loss = scalar_loss(&m);
    let grads = backward(&loss).unwrap();

    fn sig(x: f32) -> f32 {
        1.0 / (1.0 + (-x).exp())
    }
    let expected: Vec<f32> = [0.0_f32, 1.0, -1.0]
        .iter()
        .map(|&v| sig(v) * (1.0 - sig(v)) / 3.0)
        .collect();
    assert_close(&grad_vec(&grads, &x), &expected, 1e-5, "d/dx");
}

#[test]
fn test_chain_multi_layer_mlp() {
    // Simple 2-layer MLP: loss = sum(relu(relu(x @ w1) @ w2))
    // x [1,2], w1 [2,3], w2 [3,1]
    let x_var = var_from(vec![1.0, 0.5], &[1, 2]);
    let w1_var = var_from(vec![0.5, 0.3, -0.2, 0.1, -0.4, 0.6], &[2, 3]);
    let w2_var = var_from(vec![0.7, -0.3, 0.5], &[3, 1]);
    let tx = tracked(&x_var);
    let tw1 = tracked(&w1_var);
    let tw2 = tracked(&w2_var);

    let h = tx.matmul(&tw1).unwrap(); // [1, 3]
    let h_relu = h.relu().unwrap();
    let out = h_relu.matmul(&tw2).unwrap(); // [1, 1]
    let out_relu = out.relu().unwrap();
    let loss = scalar_loss(&out_relu);
    let grads = backward(&loss).unwrap();

    // All shapes must be correct
    assert_eq!(grads.get(&x_var).unwrap().dims(), &[1, 2]);
    assert_eq!(grads.get(&w1_var).unwrap().dims(), &[2, 3]);
    assert_eq!(grads.get(&w2_var).unwrap().dims(), &[3, 1]);

    // Verify gradients are finite (no NaN/Inf from chain of ops)
    for &v in grad_vec(&grads, &x_var).iter() {
        assert!(v.is_finite(), "x grad is not finite: {v}");
    }
    for &v in grad_vec(&grads, &w1_var).iter() {
        assert!(v.is_finite(), "w1 grad is not finite: {v}");
    }
    for &v in grad_vec(&grads, &w2_var).iter() {
        assert!(v.is_finite(), "w2 grad is not finite: {v}");
    }
}

#[test]
fn test_chain_mul_scalar_add_scalar() {
    // loss = sum((x * 3.0) + 2.0) = sum(3x + 2) = 3*sum(x) + 2*n
    // d/dx_i = 3
    let x = var_from(vec![1.0, 2.0, 3.0, 4.0], &[4]);
    let tx = tracked(&x);
    let scaled = tx.mul_scalar(3.0).unwrap();
    let shifted = scaled.add_scalar(2.0).unwrap();
    let loss = scalar_loss(&shifted);
    let grads = backward(&loss).unwrap();

    assert_close(&grad_vec(&grads, &x), &[3.0, 3.0, 3.0, 3.0], 1e-6, "d/dx");
}

#[test]
fn test_chain_div_sqrt_composition() {
    // loss = sum(x / sqrt(x)) = sum(sqrt(x)) for x > 0
    // d/dx_i = 1 / (2 * sqrt(x_i))
    let x = var_from(vec![4.0, 9.0, 16.0], &[3]);
    let tx = tracked(&x);
    let s = tx.sqrt().unwrap();
    let y = tx.div(&s).unwrap();
    let loss = scalar_loss(&y);
    let grads = backward(&loss).unwrap();

    // Numerically: this equals sqrt(x), so gradient is 1/(2*sqrt(x))
    let expected = [1.0 / (2.0 * 2.0), 1.0 / (2.0 * 3.0), 1.0 / (2.0 * 4.0)];
    assert_close(&grad_vec(&grads, &x), &expected, 1e-3, "d/dx");
}

#[test]
fn test_grad_unsqueeze_squeeze() {
    // x [3] -> unsqueeze(0) -> [1,3] -> sqr -> sum => loss
    // Gradient must flow through unsqueeze back to [3].
    let x = var_from(vec![2.0, 3.0, 4.0], &[3]);
    let tx = tracked(&x);
    let u = tx.unsqueeze(0).unwrap(); // [1, 3]
    let sq = u.sqr().unwrap();
    let loss = scalar_loss(&sq);
    let grads = backward(&loss).unwrap();

    assert_eq!(grads.get(&x).unwrap().dims(), &[3]);
    assert_close(&grad_vec(&grads, &x), &[4.0, 6.0, 8.0], 1e-5, "d/dx");
}

#[test]
fn test_grad_narrow() {
    // x [5] -> narrow(dim=0, start=1, len=3) => [3] subset
    // loss = sum(narrow(x))
    // Gradient into x should be [0, 1, 1, 1, 0]
    let x = var_from(vec![10.0, 20.0, 30.0, 40.0, 50.0], &[5]);
    let tx = tracked(&x);
    let sliced = tx.narrow(0, 1, 3).unwrap(); // [20, 30, 40]
    let loss = scalar_loss(&sliced);
    let grads = backward(&loss).unwrap();

    assert_eq!(grads.get(&x).unwrap().dims(), &[5]);
    assert_close(
        &grad_vec(&grads, &x),
        &[0.0, 1.0, 1.0, 1.0, 0.0],
        1e-6,
        "d/dx",
    );
}

#[test]
fn test_grad_cat() {
    // a [2], b [3], cat(dim=0) => [5], loss = sum
    // d/da = [1, 1], d/db = [1, 1, 1]
    let a = var_from(vec![1.0, 2.0], &[2]);
    let b = var_from(vec![3.0, 4.0, 5.0], &[3]);
    let ta = tracked(&a);
    let tb = tracked(&b);
    let c = TrackedTensor::cat(&[&ta, &tb], 0).unwrap();
    let loss = scalar_loss(&c);
    let grads = backward(&loss).unwrap();

    assert_eq!(grads.get(&a).unwrap().dims(), &[2]);
    assert_eq!(grads.get(&b).unwrap().dims(), &[3]);
    assert_close(&grad_vec(&grads, &a), &[1.0, 1.0], 1e-6, "d/da");
    assert_close(&grad_vec(&grads, &b), &[1.0, 1.0, 1.0], 1e-6, "d/db");
}

#[test]
fn test_grad_permute() {
    // x [2,3] -> permute [1,0] -> [3,2] -> sqr -> sum
    // Gradient must flow back through permute to [2,3]
    let vals = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let x = var_from(vals.clone(), &[2, 3]);
    let tx = tracked(&x);
    let p = tx.permute(&[1, 0]).unwrap(); // [3, 2]
    let sq = p.sqr().unwrap();
    let loss = scalar_loss(&sq);
    let grads = backward(&loss).unwrap();

    assert_eq!(grads.get(&x).unwrap().dims(), &[2, 3]);
    // d/dx_i = 2 * x_i
    let expected: Vec<f32> = vals.iter().map(|&v| 2.0 * v).collect();
    assert_close(&grad_vec(&grads, &x), &expected, 1e-5, "d/dx");
}

#[test]
fn test_grad_abs() {
    // loss = sum(|x|), d/dx_i = sign(x_i), with sign(0) = 0
    let x = var_from(vec![-3.0, 0.0, 2.0, -1.0], &[4]);
    let tx = tracked(&x);
    let a = tx.abs().unwrap();
    let loss = scalar_loss(&a);
    let grads = backward(&loss).unwrap();

    assert_close(&grad_vec(&grads, &x), &[-1.0, 0.0, 1.0, -1.0], 1e-6, "d/dx");
}

#[test]
fn test_grad_recip() {
    // loss = sum(1/x), d/dx_i = -1/x_i^2
    let x = var_from(vec![2.0, 4.0, 0.5], &[3]);
    let tx = tracked(&x);
    let r = tx.recip().unwrap();
    let loss = scalar_loss(&r);
    let grads = backward(&loss).unwrap();

    // -1/4, -1/16, -1/0.25 = -4
    assert_close(&grad_vec(&grads, &x), &[-0.25, -0.0625, -4.0], 1e-4, "d/dx");
}

#[test]
fn test_grad_powf() {
    // loss = sum(x^3), d/dx_i = 3 * x_i^2
    let x = var_from(vec![1.0, 2.0, 3.0], &[3]);
    let tx = tracked(&x);
    let p = tx.powf(3.0).unwrap();
    let loss = scalar_loss(&p);
    let grads = backward(&loss).unwrap();

    assert_close(&grad_vec(&grads, &x), &[3.0, 12.0, 27.0], 1e-3, "d/dx");
}

#[test]
fn test_grad_clamp() {
    // loss = sum(clamp(x, 0.0, 2.0))
    // d/dx_i = 1 if 0 < x_i < 2, else 0 (at boundaries, implementation dependent)
    let x = var_from(vec![-1.0, 0.5, 1.5, 3.0], &[4]);
    let tx = tracked(&x);
    let c = tx.clamp(0.0, 2.0).unwrap();
    let loss = scalar_loss(&c);
    let grads = backward(&loss).unwrap();

    // -1 => clamped to 0 (grad=0), 0.5 => pass (grad=1), 1.5 => pass (grad=1), 3.0 => clamped to 2 (grad=0)
    assert_close(&grad_vec(&grads, &x), &[0.0, 1.0, 1.0, 0.0], 1e-6, "d/dx");
}

#[test]
fn test_grad_sqr() {
    // loss = sum(x^2), d/dx_i = 2 * x_i
    let x = var_from(vec![1.0, -2.0, 3.0, 0.0], &[4]);
    let tx = tracked(&x);
    let sq = tx.sqr().unwrap();
    let loss = scalar_loss(&sq);
    let grads = backward(&loss).unwrap();

    assert_close(&grad_vec(&grads, &x), &[2.0, -4.0, 6.0, 0.0], 1e-5, "d/dx");
}

// ===========================================================================
// Detach (stop-gradient)
// ===========================================================================

#[test]
fn test_detach_stops_gradient() {
    // loss = sum(detach(x) * y). Gradient should flow to y but not to x.
    let x = var_from(vec![2.0, 3.0], &[2]);
    let y = var_from(vec![4.0, 5.0], &[2]);
    let tx = tracked(&x);
    let ty = tracked(&y);
    let detached = tx.detach();
    let prod = detached.mul(&ty).unwrap();
    let loss = scalar_loss(&prod);
    let grads = backward(&loss).unwrap();

    // x is detached, so no gradient
    assert!(
        grads.get(&x).is_none(),
        "x should have no gradient after detach"
    );
    // y gets gradient = detached x values
    assert_close(&grad_vec(&grads, &y), &[2.0, 3.0], 1e-6, "d/dy");
}

// ===========================================================================
// Non-scalar loss rejection
// ===========================================================================

#[test]
fn test_backward_rejects_non_scalar() {
    let x = var_from(vec![1.0, 2.0, 3.0], &[3]);
    let tx = tracked(&x);
    let result = backward(&tx);
    assert!(result.is_err(), "backward should reject non-scalar loss");
}

// ===========================================================================
// Stack
// ===========================================================================

#[test]
fn test_grad_stack() {
    // Stack [2] and [2] along dim=0 => [2, 2], loss = sum
    let a = var_from(vec![1.0, 2.0], &[2]);
    let b = var_from(vec![3.0, 4.0], &[2]);
    let ta = tracked(&a);
    let tb = tracked(&b);
    let s = TrackedTensor::stack(&[ta, tb], 0).unwrap();
    let loss = scalar_loss(&s);
    let grads = backward(&loss).unwrap();

    assert_eq!(grads.get(&a).unwrap().dims(), &[2]);
    assert_eq!(grads.get(&b).unwrap().dims(), &[2]);
    assert_close(&grad_vec(&grads, &a), &[1.0, 1.0], 1e-6, "d/da");
    assert_close(&grad_vec(&grads, &b), &[1.0, 1.0], 1e-6, "d/db");
}

// ===========================================================================
// Log softmax
// ===========================================================================

#[test]
fn test_grad_log_softmax_fd() {
    // Finite-difference check for log_softmax
    let vals = vec![1.0_f32, 2.0, 3.0];
    let x = var_from(vals.clone(), &[1, 3]);
    let tx = tracked(&x);
    let ls = tx.log_softmax(1).unwrap();
    let loss = scalar_loss(&ls);
    let grads = backward(&loss).unwrap();
    let analytical = grad_vec(&grads, &x);

    let eps = 1e-4_f32;
    for i in 0..3 {
        let mut plus = vals.clone();
        let mut minus = vals.clone();
        plus[i] += eps;
        minus[i] -= eps;

        let fwd = |d: Vec<f32>| -> f64 {
            let t = DynTensor::from_vec(d, &[1, 3], &cpu()).unwrap();
            let ls = t.log_softmax(1).unwrap();
            ls.to_flat_vec::<f32>()
                .unwrap()
                .iter()
                .map(|&v| f64::from(v))
                .sum::<f64>()
        };
        let numerical = ((fwd(plus) - fwd(minus)) / (2.0 * f64::from(eps))) as f32;
        assert!(
            (analytical[i] - numerical).abs() < 5e-3,
            "log_softmax fd grad[{i}]: analytical={}, numerical={}, diff={}",
            analytical[i],
            numerical,
            (analytical[i] - numerical).abs()
        );
    }
}

// ===========================================================================
// Broadcast backward
// ===========================================================================

#[test]
fn test_grad_broadcast_add() {
    // a [2, 3] + b [1, 3] with broadcast
    // d/db should sum over the broadcast dimension (dim 0)
    let a = var_from(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
    let b = var_from(vec![0.1, 0.2, 0.3], &[1, 3]);
    let ta = tracked(&a);
    let tb = tracked(&b);
    let y = ta.add(&tb).unwrap();
    let loss = scalar_loss(&y);
    let grads = backward(&loss).unwrap();

    // d/da = all ones [2,3]
    assert_eq!(grads.get(&a).unwrap().dims(), &[2, 3]);
    assert_close(
        &grad_vec(&grads, &a),
        &[1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        1e-6,
        "d/da",
    );
    // d/db = sum over dim 0 of ones = [2, 2, 2] (squeezed to [1,3])
    assert_eq!(grads.get(&b).unwrap().dims(), &[1, 3]);
    assert_close(&grad_vec(&grads, &b), &[2.0, 2.0, 2.0], 1e-6, "d/db");
}

// ===========================================================================
// maximum / minimum
// ===========================================================================

#[test]
fn test_grad_maximum() {
    // loss = sum(max(a, b))
    // Gradient goes to a where a >= b, to b where b > a.
    let a = var_from(vec![1.0, 5.0, 3.0], &[3]);
    let b = var_from(vec![2.0, 4.0, 3.0], &[3]);
    let ta = tracked(&a);
    let tb = tracked(&b);
    let m = ta.maximum(&tb).unwrap();
    let loss = scalar_loss(&m);
    let grads = backward(&loss).unwrap();

    // max(1,2)=2 (b wins), max(5,4)=5 (a wins), max(3,3)=3 (tie: a wins)
    assert_close(&grad_vec(&grads, &a), &[0.0, 1.0, 1.0], 1e-6, "d/da");
    assert_close(&grad_vec(&grads, &b), &[1.0, 0.0, 0.0], 1e-6, "d/db");
}

#[test]
fn test_grad_minimum() {
    // loss = sum(min(a, b))
    // Gradient goes to a where a <= b, to b where b < a.
    let a = var_from(vec![1.0, 5.0, 3.0], &[3]);
    let b = var_from(vec![2.0, 4.0, 3.0], &[3]);
    let ta = tracked(&a);
    let tb = tracked(&b);
    let m = ta.minimum(&tb).unwrap();
    let loss = scalar_loss(&m);
    let grads = backward(&loss).unwrap();

    // min(1,2)=1 (a wins), min(5,4)=4 (b wins), min(3,3)=3 (tie: a wins)
    assert_close(&grad_vec(&grads, &a), &[1.0, 0.0, 1.0], 1e-6, "d/da");
    assert_close(&grad_vec(&grads, &b), &[0.0, 1.0, 0.0], 1e-6, "d/db");
}

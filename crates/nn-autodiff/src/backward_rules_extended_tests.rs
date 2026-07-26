// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Extended backward rules tests covering 13 categories:
//! 1. Linear backward (matmul + bias)
//! 2. Conv1d backward (stride, padding)
//! 3. MatMul backward (grad@B^T, A^T@grad)
//! 4. Softmax backward (Jacobian-vector product)
//! 5. LayerNorm backward
//! 6. ReLU backward (0 for negative, 1 for positive)
//! 7. Sigmoid backward (grad * sigmoid(x) * (1 - sigmoid(x)))
//! 8. Tanh backward (grad * (1 - tanh(x)^2))
//! 9. Add backward (gradients pass through unchanged)
//! 10. Mul backward (grad_a = grad * b, grad_b = grad * a)
//! 11. Chain rule (composite function gradients)
//! 12. Broadcast backward (reduction to original shape)
//! 13. Reshape backward (gradient has same shape as input)
//!
//! Each test uses finite-difference (central difference) cross-checks
//! to verify analytical gradients match numerical gradients.

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

fn tracked(v: &Var) -> Arc<TrackedTensor> {
    Arc::new(TrackedTensor::from_var(v).unwrap())
}

fn grad_vec(grads: &crate::grad::GradStore, v: &Var) -> Vec<f32> {
    grads.get(v).unwrap().to_flat_vec::<f32>().unwrap()
}

/// Make a scalar loss from an arbitrary-shaped tensor by sum-of-all-elements.
fn scalar_loss(t: &Arc<TrackedTensor>) -> Arc<TrackedTensor> {
    let mut result = Arc::clone(t);
    for d in (0..result.tensor().rank()).rev() {
        result = result.sum_keepdim(d).unwrap();
    }
    result
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

/// Sum all elements of a DynTensor into a scalar f64 value.
fn sum_all(t: &DynTensor) -> f64 {
    t.to_flat_vec::<f32>()
        .unwrap()
        .iter()
        .map(|&v| f64::from(v))
        .sum()
}

// ===========================================================================
// 1. Linear backward: grad_weight = input^T @ grad_output,
//    grad_bias = sum(grad_output)
// ===========================================================================

/// Linear layer: y = x @ W^T + b. Verify weight and bias gradients.
#[test]
fn test_ext_linear_backward_weight_grad() {
    // x: [2, 3], W: [4, 3] => y = x @ W^T => [2, 4]
    // With loss = sum(y), grad_output = ones([2, 4])
    // grad_W = grad_output^T @ x = [4, 2] @ [2, 3] = [4, 3]
    let x_data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let w_data = vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2];

    let x_var = var_from(x_data, &[2, 3]);
    let w_var = var_from(w_data, &[4, 3]);

    let tx = tracked(&x_var);
    let tw = tracked(&w_var);

    // y = x @ W^T
    let wt = tw.transpose(0, 1).unwrap();
    let y = tx.matmul(&wt).unwrap();
    let loss = scalar_loss(&y);
    let grads = backward(&loss).unwrap();

    let gw = grads.get(&w_var).unwrap();
    assert_eq!(gw.dims(), &[4, 3], "grad_weight shape should match weight");

    // grad_W = ones([2,4])^T @ x = [[1,1],[1,1],[1,1],[1,1]] @ [[1,2,3],[4,5,6]]
    // Each row of grad_W = sum of rows of x = [5, 7, 9]
    let gw_flat = gw.to_flat_vec::<f32>().unwrap();
    for row in 0..4 {
        assert_close(
            &gw_flat[row * 3..(row + 1) * 3],
            &[5.0, 7.0, 9.0],
            1e-4,
            &format!("grad_W row {row}"),
        );
    }
}

#[test]
fn test_ext_linear_backward_bias_grad() {
    // Linear: y = x @ W^T + b. bias grad = sum(grad_output, dim=0)
    let x_data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let w_data = vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6];
    let b_data = vec![0.01, 0.02];

    let x_var = var_from(x_data, &[2, 3]);
    let w_var = var_from(w_data, &[2, 3]);
    let b_var = var_from(b_data, &[1, 2]);

    let tx = tracked(&x_var);
    let tw = tracked(&w_var);
    let tb = tracked(&b_var);

    let wt = tw.transpose(0, 1).unwrap();
    let y = tx.matmul(&wt).unwrap(); // [2, 2]
    let y_biased = y.add(&tb).unwrap(); // broadcast [1,2] to [2,2]
    let loss = scalar_loss(&y_biased);
    let grads = backward(&loss).unwrap();

    // grad_bias = sum(ones([2,2]), dim=0) = [2, 2]
    let gb = grad_vec(&grads, &b_var);
    assert_close(&gb, &[2.0, 2.0], 1e-4, "grad_bias");
}

#[test]
fn test_ext_linear_backward_fd() {
    let x_data = vec![0.5_f32, -0.3, 1.2, 0.8, -0.6, 0.4];
    let w_data = vec![0.1, 0.2, 0.3, -0.1, 0.5, -0.2];

    let x_var = var_from(x_data.clone(), &[2, 3]);
    let w_var = var_from(w_data.clone(), &[2, 3]);
    let tx = tracked(&x_var);
    let tw = tracked(&w_var);
    let wt = tw.transpose(0, 1).unwrap();
    let y = tx.matmul(&wt).unwrap();
    let loss = scalar_loss(&y);
    let grads = backward(&loss).unwrap();

    let gw = grad_vec(&grads, &w_var);
    let num = numerical_grad(&w_data, 1e-3, |d| {
        let x_t = DynTensor::from_vec(x_data.clone(), &[2, 3], &cpu()).unwrap();
        let w_t = DynTensor::from_vec(d, &[2, 3], &cpu()).unwrap();
        let wt_t = w_t.transpose(0, 1).unwrap();
        sum_all(&x_t.matmul(&wt_t).unwrap())
    });
    assert_grad_close(&gw, &num, 1e-2, "linear_weight_fd");
}

// ===========================================================================
// 2. Conv1d backward: gradient flows correctly through stride and padding
// ===========================================================================

#[test]
fn test_ext_conv1d_backward_no_padding() {
    // input: [1, 1, 5], kernel: [1, 1, 3], stride=1, padding=0
    let x_data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let k_data = vec![1.0, 0.0, -1.0];

    let x_var = var_from(x_data.clone(), &[1, 1, 5]);
    let k_var = var_from(k_data.clone(), &[1, 1, 3]);
    let tx = tracked(&x_var);
    let tk = tracked(&k_var);

    let y = tx.conv1d(&tk, 0, 1, 1, 1).unwrap();
    let loss = scalar_loss(&y);
    let grads = backward(&loss).unwrap();

    let gx = grads.get(&x_var).unwrap();
    let gk = grads.get(&k_var).unwrap();
    assert_eq!(gx.dims(), &[1, 1, 5], "conv1d grad_input shape");
    assert_eq!(gk.dims(), &[1, 1, 3], "conv1d grad_kernel shape");

    // FD check on input
    let gx_flat = grad_vec(&grads, &x_var);
    let num = numerical_grad(&x_data, 1e-3, |d| {
        let xt = DynTensor::from_vec(d, &[1, 1, 5], &cpu()).unwrap();
        let kt = DynTensor::from_vec(k_data.clone(), &[1, 1, 3], &cpu()).unwrap();
        sum_all(&xt.conv1d(&kt, 0, 1, 1, 1).unwrap())
    });
    assert_grad_close(&gx_flat, &num, 1e-2, "conv1d_no_pad_input_fd");
}

#[test]
fn test_ext_conv1d_backward_with_padding() {
    let x_data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let k_data = vec![0.5, 1.0, 0.5];

    let x_var = var_from(x_data.clone(), &[1, 1, 5]);
    let k_var = var_from(k_data.clone(), &[1, 1, 3]);
    let tx = tracked(&x_var);
    let tk = tracked(&k_var);

    let y = tx.conv1d(&tk, 1, 1, 1, 1).unwrap(); // padding=1
    let loss = scalar_loss(&y);
    let grads = backward(&loss).unwrap();

    let gx_flat = grad_vec(&grads, &x_var);
    let num = numerical_grad(&x_data, 1e-3, |d| {
        let xt = DynTensor::from_vec(d, &[1, 1, 5], &cpu()).unwrap();
        let kt = DynTensor::from_vec(k_data.clone(), &[1, 1, 3], &cpu()).unwrap();
        sum_all(&xt.conv1d(&kt, 1, 1, 1, 1).unwrap())
    });
    assert_grad_close(&gx_flat, &num, 1e-2, "conv1d_padded_input_fd");
}

#[test]
fn test_ext_conv1d_backward_stride2() {
    let x_data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let k_data = vec![0.5, 1.0, 0.5];

    let x_var = var_from(x_data.clone(), &[1, 1, 8]);
    let k_var = var_from(k_data.clone(), &[1, 1, 3]);
    let tx = tracked(&x_var);
    let tk = tracked(&k_var);

    let y = tx.conv1d(&tk, 0, 2, 1, 1).unwrap(); // stride=2
    let loss = scalar_loss(&y);
    let grads = backward(&loss).unwrap();

    let gx_flat = grad_vec(&grads, &x_var);
    let num = numerical_grad(&x_data, 1e-3, |d| {
        let xt = DynTensor::from_vec(d, &[1, 1, 8], &cpu()).unwrap();
        let kt = DynTensor::from_vec(k_data.clone(), &[1, 1, 3], &cpu()).unwrap();
        sum_all(&xt.conv1d(&kt, 0, 2, 1, 1).unwrap())
    });
    assert_grad_close(&gx_flat, &num, 1e-2, "conv1d_stride2_input_fd");
}

#[test]
fn test_ext_conv1d_backward_kernel_grad_fd() {
    let x_data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let k_data = vec![0.5, 1.0, 0.5];

    let x_var = var_from(x_data.clone(), &[1, 1, 5]);
    let k_var = var_from(k_data.clone(), &[1, 1, 3]);
    let tx = tracked(&x_var);
    let tk = tracked(&k_var);

    let y = tx.conv1d(&tk, 1, 1, 1, 1).unwrap();
    let loss = scalar_loss(&y);
    let grads = backward(&loss).unwrap();

    let gk_flat = grad_vec(&grads, &k_var);
    let num = numerical_grad(&k_data, 1e-3, |d| {
        let xt = DynTensor::from_vec(x_data.clone(), &[1, 1, 5], &cpu()).unwrap();
        let kt = DynTensor::from_vec(d, &[1, 1, 3], &cpu()).unwrap();
        sum_all(&xt.conv1d(&kt, 1, 1, 1, 1).unwrap())
    });
    assert_grad_close(&gk_flat, &num, 1e-2, "conv1d_kernel_fd");
}

// ===========================================================================
// 3. MatMul backward: grad_A = grad @ B^T, grad_B = A^T @ grad
// ===========================================================================

#[test]
fn test_ext_matmul_backward_grad_a_is_grad_times_bt() {
    // A: [2, 3], B: [3, 4] => C = A @ B => [2, 4]
    // loss = sum(C), grad_output = ones([2, 4])
    // grad_A = ones([2,4]) @ B^T => [2, 3]
    let a_data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let b_data = vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2];

    let a_var = var_from(a_data, &[2, 3]);
    let b_var = var_from(b_data.clone(), &[3, 4]);
    let ta = tracked(&a_var);
    let tb = tracked(&b_var);

    let c = ta.matmul(&tb).unwrap();
    let loss = scalar_loss(&c);
    let grads = backward(&loss).unwrap();

    let ga = grads.get(&a_var).unwrap();
    assert_eq!(ga.dims(), &[2, 3], "grad_A shape matches A");

    // grad_A = ones([2,4]) @ B^T
    // B^T column sums: each row of grad_A[i, j] = sum of B[j, :] = sum of row j of B
    let b_t = DynTensor::from_vec(b_data, &[3, 4], &cpu()).unwrap();
    let expected_ga = DynTensor::ones(&[2, 4], nn_core::DType::F32, &cpu())
        .unwrap()
        .matmul(&b_t.transpose(0, 1).unwrap())
        .unwrap();
    let ga_flat = ga.to_flat_vec::<f32>().unwrap();
    let expected_flat = expected_ga.to_flat_vec::<f32>().unwrap();
    assert_close(&ga_flat, &expected_flat, 1e-4, "matmul_grad_a_analytical");
}

#[test]
fn test_ext_matmul_backward_grad_b_is_at_times_grad() {
    let a_data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let b_data = vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2];

    let a_var = var_from(a_data.clone(), &[2, 3]);
    let b_var = var_from(b_data, &[3, 4]);
    let ta = tracked(&a_var);
    let tb = tracked(&b_var);

    let c = ta.matmul(&tb).unwrap();
    let loss = scalar_loss(&c);
    let grads = backward(&loss).unwrap();

    let gb = grads.get(&b_var).unwrap();
    assert_eq!(gb.dims(), &[3, 4], "grad_B shape matches B");

    // grad_B = A^T @ ones([2,4])
    let a_t = DynTensor::from_vec(a_data, &[2, 3], &cpu())
        .unwrap()
        .transpose(0, 1)
        .unwrap();
    let expected_gb = a_t
        .matmul(&DynTensor::ones(&[2, 4], nn_core::DType::F32, &cpu()).unwrap())
        .unwrap();
    let gb_flat = gb.to_flat_vec::<f32>().unwrap();
    let expected_flat = expected_gb.to_flat_vec::<f32>().unwrap();
    assert_close(&gb_flat, &expected_flat, 1e-4, "matmul_grad_b_analytical");
}

#[test]
fn test_ext_matmul_backward_fd() {
    let a_data = vec![0.5_f32, -0.3, 1.2, 0.8, -0.6, 0.4];
    let b_data = vec![1.0_f32, 0.3, -0.5, 0.7, 0.2, -0.8];

    let a_var = var_from(a_data.clone(), &[2, 3]);
    let b_var = var_from(b_data.clone(), &[3, 2]);
    let ta = tracked(&a_var);
    let tb = tracked(&b_var);

    let c = ta.matmul(&tb).unwrap();
    let loss = scalar_loss(&c);
    let grads = backward(&loss).unwrap();

    // FD for A
    let ga = grad_vec(&grads, &a_var);
    let num_a = numerical_grad(&a_data, 1e-3, |d| {
        let at = DynTensor::from_vec(d, &[2, 3], &cpu()).unwrap();
        let bt = DynTensor::from_vec(b_data.clone(), &[3, 2], &cpu()).unwrap();
        sum_all(&at.matmul(&bt).unwrap())
    });
    assert_grad_close(&ga, &num_a, 1e-2, "matmul_a_fd");

    // FD for B
    let gb = grad_vec(&grads, &b_var);
    let num_b = numerical_grad(&b_data, 1e-3, |d| {
        let at = DynTensor::from_vec(a_data.clone(), &[2, 3], &cpu()).unwrap();
        let bt = DynTensor::from_vec(d, &[3, 2], &cpu()).unwrap();
        sum_all(&at.matmul(&bt).unwrap())
    });
    assert_grad_close(&gb, &num_b, 1e-2, "matmul_b_fd");
}

// ===========================================================================
// 4. Softmax backward: Jacobian-vector product property
// ===========================================================================

#[test]
fn test_ext_softmax_backward_jvp_property() {
    // Softmax backward: grad_input = softmax * (grad - sum(grad * softmax, dim))
    // Key property: sum(grad_input, dim) == 0 (softmax outputs sum to 1,
    // so the tangent must have zero sum).
    let x_data = vec![1.0_f32, 2.0, 3.0, 0.5, 1.5, 2.5];
    let x_var = var_from(x_data, &[2, 3]);
    let tx = tracked(&x_var);

    // Use a non-trivial loss: sum(softmax(x)^2)
    let s = tx.softmax(1).unwrap();
    let loss = s.sqr().unwrap();
    let loss = scalar_loss(&loss);
    let grads = backward(&loss).unwrap();

    let gx = grads.get(&x_var).unwrap();
    assert_eq!(gx.dims(), &[2, 3], "softmax grad shape");

    // Verify gradient sums to approximately 0 along softmax dim for each row
    let g_flat = gx.to_flat_vec::<f32>().unwrap();
    for row in 0..2 {
        let row_sum: f32 = g_flat[row * 3..(row + 1) * 3].iter().sum();
        assert!(
            row_sum.abs() < 1e-5,
            "softmax grad row {row} sum={row_sum}, expected ~0"
        );
    }
}

#[test]
fn test_ext_softmax_backward_fd() {
    let x_data = vec![1.0_f32, 2.0, 3.0, 0.5, 1.5, 2.5];
    let x_var = var_from(x_data.clone(), &[2, 3]);
    let tx = tracked(&x_var);

    let s = tx.softmax(1).unwrap();
    let loss = s.sqr().unwrap();
    let loss = scalar_loss(&loss);
    let grads = backward(&loss).unwrap();

    let gx = grad_vec(&grads, &x_var);
    let num = numerical_grad(&x_data, 1e-3, |d| {
        let t = DynTensor::from_vec(d, &[2, 3], &cpu()).unwrap();
        let s = t.softmax(1).unwrap();
        let sq = s.sqr().unwrap();
        sum_all(&sq)
    });
    assert_grad_close(&gx, &num, 1e-2, "softmax_fd");
}

#[test]
fn test_ext_softmax_backward_uniform_input() {
    // When all inputs equal, softmax = uniform, gradients should be equal per row
    let x_data = vec![1.0_f32; 6];
    let x_var = var_from(x_data, &[2, 3]);
    let tx = tracked(&x_var);

    let s = tx.softmax(1).unwrap();
    let loss = scalar_loss(&s);
    let grads = backward(&loss).unwrap();

    let g = grad_vec(&grads, &x_var);
    // For uniform input, softmax = 1/3 for each, and grad should be 0
    // since sum of softmax = 1 is constant w.r.t. input
    for (i, &v) in g.iter().enumerate() {
        assert!(v.abs() < 1e-5, "softmax uniform grad[{i}]={v}, expected ~0");
    }
}

// ===========================================================================
// 5. LayerNorm backward: gradient through normalization
// ===========================================================================

#[test]
fn test_ext_layer_norm_backward_shapes() {
    // LayerNorm: input [2, 4], weight [4], bias [4]
    let x_data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let w_data = vec![1.0, 1.0, 1.0, 1.0];
    let b_data = vec![0.0, 0.0, 0.0, 0.0];

    let x_var = var_from(x_data, &[2, 4]);
    let w_var = var_from(w_data, &[4]);
    let b_var = var_from(b_data, &[4]);

    let tx = tracked(&x_var);
    let tw = tracked(&w_var);
    let tb = tracked(&b_var);

    let y = tx.layer_norm(&tw, &tb, 1e-5).unwrap();
    let loss = scalar_loss(&y);
    let grads = backward(&loss).unwrap();

    let gx = grads.get(&x_var).unwrap();
    let gw = grads.get(&w_var).unwrap();
    let gb = grads.get(&b_var).unwrap();
    assert_eq!(gx.dims(), &[2, 4], "layer_norm grad_input shape");
    assert_eq!(gw.dims(), &[4], "layer_norm grad_weight shape");
    assert_eq!(gb.dims(), &[4], "layer_norm grad_bias shape");
}

#[test]
fn test_ext_layer_norm_backward_bias_grad() {
    // grad_bias = sum(grad_output) over batch dim
    let x_data = vec![1.0, 2.0, 3.0, 5.0, 6.0, 7.0];
    let w_data = vec![1.0, 1.0, 1.0];
    let b_data = vec![0.0, 0.0, 0.0];

    let x_var = var_from(x_data, &[2, 3]);
    let w_var = var_from(w_data, &[3]);
    let b_var = var_from(b_data, &[3]);

    let tx = tracked(&x_var);
    let tw = tracked(&w_var);
    let tb = tracked(&b_var);

    let y = tx.layer_norm(&tw, &tb, 1e-5).unwrap();
    let loss = scalar_loss(&y);
    let grads = backward(&loss).unwrap();

    // Since loss = sum(y), grad_output = ones. grad_bias = sum(ones, dim=0) = [2, 2, 2]
    let gb = grad_vec(&grads, &b_var);
    assert_close(&gb, &[2.0, 2.0, 2.0], 1e-4, "layer_norm_grad_bias");
}

#[test]
fn test_ext_layer_norm_backward_input_fd() {
    let x_data = vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    let w_data = vec![0.5_f32, 1.0, 1.5];
    let b_data = vec![0.1_f32, 0.2, 0.3];

    let x_var = var_from(x_data.clone(), &[2, 3]);
    let w_var = var_from(w_data.clone(), &[3]);
    let b_var = var_from(b_data.clone(), &[3]);

    let tx = tracked(&x_var);
    let tw = tracked(&w_var);
    let tb = tracked(&b_var);

    let y = tx.layer_norm(&tw, &tb, 1e-5).unwrap();
    let loss = scalar_loss(&y);
    let grads = backward(&loss).unwrap();

    let gx = grad_vec(&grads, &x_var);
    let num = numerical_grad(&x_data, 1e-3, |d| {
        let xt = DynTensor::from_vec(d, &[2, 3], &cpu()).unwrap();
        let wt = DynTensor::from_vec(w_data.clone(), &[3], &cpu()).unwrap();
        let bt = DynTensor::from_vec(b_data.clone(), &[3], &cpu()).unwrap();
        // Recompute layer_norm forward manually
        let mean = xt.mean_keepdim(1).unwrap();
        let diff = xt.sub(&mean).unwrap();
        let var = diff.sqr().unwrap().mean_keepdim(1).unwrap();
        let inv_std = var
            .add_scalar(1e-5)
            .unwrap()
            .sqrt()
            .unwrap()
            .recip()
            .unwrap();
        let normed = diff.mul(&inv_std).unwrap();
        let out = normed.mul(&wt).unwrap().add(&bt).unwrap();
        sum_all(&out)
    });
    assert_grad_close(&gx, &num, 5e-2, "layer_norm_input_fd");
}

// ===========================================================================
// 6. ReLU backward: gradient is 0 for negative inputs, 1 for positive
// ===========================================================================

#[test]
fn test_ext_relu_backward_zero_for_negative() {
    let x_var = var_from(vec![-3.0, -2.0, -1.0, -0.5, -0.01], &[5]);
    let tx = tracked(&x_var);
    let y = tx.relu().unwrap();
    let loss = scalar_loss(&y);
    let grads = backward(&loss).unwrap();

    let g = grad_vec(&grads, &x_var);
    assert_close(&g, &[0.0, 0.0, 0.0, 0.0, 0.0], 1e-6, "relu_negative");
}

#[test]
fn test_ext_relu_backward_one_for_positive() {
    let x_var = var_from(vec![0.01, 0.5, 1.0, 2.0, 10.0], &[5]);
    let tx = tracked(&x_var);
    let y = tx.relu().unwrap();
    let loss = scalar_loss(&y);
    let grads = backward(&loss).unwrap();

    let g = grad_vec(&grads, &x_var);
    assert_close(&g, &[1.0, 1.0, 1.0, 1.0, 1.0], 1e-6, "relu_positive");
}

#[test]
fn test_ext_relu_backward_mixed_fd() {
    let x_data = vec![-2.0_f32, -0.5, 0.5, 1.5, 3.0];
    let x_var = var_from(x_data.clone(), &[5]);
    let tx = tracked(&x_var);
    let y = tx.relu().unwrap();
    let loss = y.sqr().unwrap();
    let loss = scalar_loss(&loss);
    let grads = backward(&loss).unwrap();

    let gx = grad_vec(&grads, &x_var);
    let num = numerical_grad(&x_data, 1e-3, |d| {
        let t = DynTensor::from_vec(d, &[5], &cpu()).unwrap();
        let y = t.relu().unwrap().sqr().unwrap();
        sum_all(&y)
    });
    assert_grad_close(&gx, &num, 1e-2, "relu_mixed_fd");
}

#[test]
fn test_ext_relu_backward_2d() {
    let x_var = var_from(vec![-1.0, 2.0, -3.0, 4.0, 0.0, -5.0], &[2, 3]);
    let tx = tracked(&x_var);
    let y = tx.relu().unwrap();
    let loss = scalar_loss(&y);
    let grads = backward(&loss).unwrap();

    let g = grad_vec(&grads, &x_var);
    // ReLU grad: 0 where x<=0, 1 where x>0
    assert_close(&g, &[0.0, 1.0, 0.0, 1.0, 1.0, 0.0], 1e-6, "relu_2d");
}

// ===========================================================================
// 7. Sigmoid backward: grad * sigmoid(x) * (1 - sigmoid(x))
// ===========================================================================

#[test]
fn test_ext_sigmoid_backward_analytical() {
    let vals = vec![-2.0_f32, -1.0, 0.0, 1.0, 2.0];
    let x_var = var_from(vals.clone(), &[5]);
    let tx = tracked(&x_var);
    let y = tx.sigmoid().unwrap();
    let loss = scalar_loss(&y);
    let grads = backward(&loss).unwrap();

    let g = grad_vec(&grads, &x_var);
    for (i, &v) in vals.iter().enumerate() {
        let s = 1.0 / (1.0 + (-v).exp());
        let expected = s * (1.0 - s);
        assert!(
            (g[i] - expected).abs() < 1e-5,
            "sigmoid grad[{i}]: expected={expected}, got={}",
            g[i]
        );
    }
}

#[test]
fn test_ext_sigmoid_backward_at_zero() {
    // sigmoid(0) = 0.5, sigmoid'(0) = 0.25
    let x_var = var_from(vec![0.0], &[1]);
    let tx = tracked(&x_var);
    let y = tx.sigmoid().unwrap();
    let loss = scalar_loss(&y);
    let grads = backward(&loss).unwrap();

    let g = grad_vec(&grads, &x_var);
    assert!(
        (g[0] - 0.25).abs() < 1e-5,
        "sigmoid'(0) = 0.25, got {}",
        g[0]
    );
}

#[test]
fn test_ext_sigmoid_backward_fd() {
    let x_data = vec![-3.0_f32, -1.0, 0.0, 1.0, 3.0];
    let x_var = var_from(x_data.clone(), &[5]);
    let tx = tracked(&x_var);
    let y = tx.sigmoid().unwrap();
    let loss = scalar_loss(&y);
    let grads = backward(&loss).unwrap();

    let gx = grad_vec(&grads, &x_var);
    let num = numerical_grad(&x_data, 1e-4, |d| {
        let t = DynTensor::from_vec(d, &[5], &cpu()).unwrap();
        sum_all(&t.sigmoid().unwrap())
    });
    assert_grad_close(&gx, &num, 1e-2, "sigmoid_fd");
}

// ===========================================================================
// 8. Tanh backward: grad * (1 - tanh(x)^2)
// ===========================================================================

#[test]
fn test_ext_tanh_backward_analytical() {
    let vals = vec![0.0_f32, 0.5, -0.5, 1.0, -1.0, 2.0];
    let x_var = var_from(vals.clone(), &[6]);
    let tx = tracked(&x_var);
    let y = tx.tanh().unwrap();
    let loss = scalar_loss(&y);
    let grads = backward(&loss).unwrap();

    let g = grad_vec(&grads, &x_var);
    for (i, &v) in vals.iter().enumerate() {
        let expected = 1.0 - v.tanh().powi(2);
        assert!(
            (g[i] - expected).abs() < 1e-5,
            "tanh grad[{i}]: expected={expected}, got={}",
            g[i]
        );
    }
}

#[test]
fn test_ext_tanh_backward_at_zero() {
    // tanh(0) = 0, tanh'(0) = 1
    let x_var = var_from(vec![0.0], &[1]);
    let tx = tracked(&x_var);
    let y = tx.tanh().unwrap();
    let loss = scalar_loss(&y);
    let grads = backward(&loss).unwrap();

    let g = grad_vec(&grads, &x_var);
    assert!((g[0] - 1.0).abs() < 1e-5, "tanh'(0) = 1, got {}", g[0]);
}

#[test]
fn test_ext_tanh_backward_saturation() {
    // For large |x|, tanh(x) ~= +/-1, tanh'(x) ~= 0
    let x_var = var_from(vec![-10.0, 10.0], &[2]);
    let tx = tracked(&x_var);
    let y = tx.tanh().unwrap();
    let loss = scalar_loss(&y);
    let grads = backward(&loss).unwrap();

    let g = grad_vec(&grads, &x_var);
    assert!(g[0].abs() < 1e-4, "tanh'(-10) ~= 0, got {}", g[0]);
    assert!(g[1].abs() < 1e-4, "tanh'(10) ~= 0, got {}", g[1]);
}

#[test]
fn test_ext_tanh_backward_fd() {
    let x_data = vec![-1.5_f32, -0.5, 0.0, 0.5, 1.5];
    let x_var = var_from(x_data.clone(), &[5]);
    let tx = tracked(&x_var);
    let y = tx.tanh().unwrap();
    let loss = scalar_loss(&y);
    let grads = backward(&loss).unwrap();

    let gx = grad_vec(&grads, &x_var);
    let num = numerical_grad(&x_data, 1e-4, |d| {
        let t = DynTensor::from_vec(d, &[5], &cpu()).unwrap();
        sum_all(&t.tanh().unwrap())
    });
    assert_grad_close(&gx, &num, 1e-2, "tanh_fd");
}

// ===========================================================================
// 9. Add backward: gradients pass through unchanged
// ===========================================================================

#[test]
fn test_ext_add_backward_passthrough() {
    let x_var = var_from(vec![1.0, 2.0, 3.0], &[3]);
    let y_var = var_from(vec![4.0, 5.0, 6.0], &[3]);
    let tx = tracked(&x_var);
    let ty = tracked(&y_var);

    let z = tx.add(&ty).unwrap();
    let loss = scalar_loss(&z);
    let grads = backward(&loss).unwrap();

    // d(x+y)/dx = 1 for all elements, d(x+y)/dy = 1 for all elements
    assert_close(
        &grad_vec(&grads, &x_var),
        &[1.0, 1.0, 1.0],
        1e-6,
        "add_grad_x",
    );
    assert_close(
        &grad_vec(&grads, &y_var),
        &[1.0, 1.0, 1.0],
        1e-6,
        "add_grad_y",
    );
}

#[test]
fn test_ext_add_backward_with_upstream_grad() {
    // If loss = sum((x + y)^2), grad_output = 2*(x+y)
    // grad_x = 2*(x+y), grad_y = 2*(x+y)
    let x_data = vec![1.0, 2.0, 3.0];
    let y_data = vec![0.5, 1.0, 1.5];

    let x_var = var_from(x_data.clone(), &[3]);
    let y_var = var_from(y_data.clone(), &[3]);
    let tx = tracked(&x_var);
    let ty = tracked(&y_var);

    let z = tx.add(&ty).unwrap();
    let loss = z.sqr().unwrap();
    let loss = scalar_loss(&loss);
    let grads = backward(&loss).unwrap();

    let gx = grad_vec(&grads, &x_var);
    let gy = grad_vec(&grads, &y_var);
    for i in 0..3 {
        let expected = 2.0 * (x_data[i] + y_data[i]);
        assert!(
            (gx[i] - expected).abs() < 1e-4,
            "add_sqr gx[{i}]={}, expected={expected}",
            gx[i]
        );
        assert!(
            (gy[i] - expected).abs() < 1e-4,
            "add_sqr gy[{i}]={}, expected={expected}",
            gy[i]
        );
    }
}

#[test]
fn test_ext_add_backward_2d_fd() {
    let x_data = vec![0.1_f32, -0.2, 0.3, -0.4, 0.5, -0.6];
    let y_data = vec![0.6_f32, 0.5, 0.4, 0.3, 0.2, 0.1];

    let x_var = var_from(x_data.clone(), &[2, 3]);
    let y_var = var_from(y_data.clone(), &[2, 3]);
    let tx = tracked(&x_var);
    let ty = tracked(&y_var);

    let z = tx.add(&ty).unwrap().sqr().unwrap();
    let loss = scalar_loss(&z);
    let grads = backward(&loss).unwrap();

    let gx = grad_vec(&grads, &x_var);
    let num = numerical_grad(&x_data, 1e-3, |d| {
        let xt = DynTensor::from_vec(d, &[2, 3], &cpu()).unwrap();
        let yt = DynTensor::from_vec(y_data.clone(), &[2, 3], &cpu()).unwrap();
        sum_all(&xt.add(&yt).unwrap().sqr().unwrap())
    });
    assert_grad_close(&gx, &num, 1e-2, "add_2d_fd");
}

// ===========================================================================
// 10. Mul backward: grad_a = grad * b, grad_b = grad * a
// ===========================================================================

#[test]
fn test_ext_mul_backward_analytical() {
    let x_data = vec![2.0, 3.0, 4.0];
    let y_data = vec![5.0, 6.0, 7.0];

    let x_var = var_from(x_data, &[3]);
    let y_var = var_from(y_data, &[3]);
    let tx = tracked(&x_var);
    let ty = tracked(&y_var);

    let z = tx.mul(&ty).unwrap();
    let loss = scalar_loss(&z);
    let grads = backward(&loss).unwrap();

    // d/dx (x*y) = y, d/dy (x*y) = x (with grad_output = 1)
    assert_close(
        &grad_vec(&grads, &x_var),
        &[5.0, 6.0, 7.0],
        1e-5,
        "mul_grad_x",
    );
    assert_close(
        &grad_vec(&grads, &y_var),
        &[2.0, 3.0, 4.0],
        1e-5,
        "mul_grad_y",
    );
}

#[test]
fn test_ext_mul_backward_with_upstream_grad() {
    // loss = sum((x*y)^2), grad = 2*x*y
    // grad_x = 2*x*y * y = 2*x*y^2
    // grad_y = 2*x*y * x = 2*x^2*y
    let x_data = vec![1.0, 2.0];
    let y_data = vec![3.0, 4.0];

    let x_var = var_from(x_data.clone(), &[2]);
    let y_var = var_from(y_data.clone(), &[2]);
    let tx = tracked(&x_var);
    let ty = tracked(&y_var);

    let z = tx.mul(&ty).unwrap();
    let loss = z.sqr().unwrap();
    let loss = scalar_loss(&loss);
    let grads = backward(&loss).unwrap();

    let gx = grad_vec(&grads, &x_var);
    let gy = grad_vec(&grads, &y_var);
    for i in 0..2 {
        let ex_gx = 2.0 * x_data[i] * y_data[i] * y_data[i];
        let ex_gy = 2.0 * x_data[i] * x_data[i] * y_data[i];
        assert!(
            (gx[i] - ex_gx).abs() < 1e-3,
            "mul_sqr gx[{i}]={}, expected={ex_gx}",
            gx[i]
        );
        assert!(
            (gy[i] - ex_gy).abs() < 1e-3,
            "mul_sqr gy[{i}]={}, expected={ex_gy}",
            gy[i]
        );
    }
}

#[test]
fn test_ext_mul_self_backward() {
    // x * x = x^2, gradient should be 2*x
    let x_data = vec![1.0, -2.0, 3.0, 0.5];
    let x_var = var_from(x_data.clone(), &[4]);
    let tx = tracked(&x_var);

    let z = tx.mul(&tx).unwrap();
    let loss = scalar_loss(&z);
    let grads = backward(&loss).unwrap();

    let g = grad_vec(&grads, &x_var);
    for (i, &v) in x_data.iter().enumerate() {
        let expected = 2.0 * v;
        assert!(
            (g[i] - expected).abs() < 1e-5,
            "mul_self grad[{i}]={}, expected={expected}",
            g[i]
        );
    }
}

// ===========================================================================
// 11. Chain rule: composite function gradient = product of individual gradients
// ===========================================================================

#[test]
fn test_ext_chain_rule_sigmoid_of_linear() {
    // f(x) = sigmoid(2x + 1), f'(x) = 2 * sigmoid(2x+1) * (1 - sigmoid(2x+1))
    let x_data = vec![-1.0_f32, 0.0, 0.5, 1.0];
    let x_var = var_from(x_data.clone(), &[4]);
    let tx = tracked(&x_var);

    let y = tx
        .mul_scalar(2.0)
        .unwrap()
        .add_scalar(1.0)
        .unwrap()
        .sigmoid()
        .unwrap();
    let loss = scalar_loss(&y);
    let grads = backward(&loss).unwrap();

    let g = grad_vec(&grads, &x_var);
    for (i, &v) in x_data.iter().enumerate() {
        let inner = 2.0 * v + 1.0;
        let s = 1.0 / (1.0 + (-inner).exp());
        let expected = 2.0 * s * (1.0 - s);
        assert!(
            (g[i] - expected).abs() < 1e-4,
            "chain_sigmoid_linear grad[{i}]={}, expected={expected}",
            g[i]
        );
    }
}

#[test]
fn test_ext_chain_rule_exp_of_sqr() {
    // f(x) = exp(x^2), f'(x) = 2x * exp(x^2)
    let x_data = vec![-0.5_f32, 0.0, 0.3, 0.7];
    let x_var = var_from(x_data.clone(), &[4]);
    let tx = tracked(&x_var);

    let y = tx.sqr().unwrap().exp().unwrap();
    let loss = scalar_loss(&y);
    let grads = backward(&loss).unwrap();

    let g = grad_vec(&grads, &x_var);
    for (i, &v) in x_data.iter().enumerate() {
        let expected = 2.0 * v * (v * v).exp();
        assert!(
            (g[i] - expected).abs() < 1e-3,
            "chain_exp_sqr grad[{i}]={}, expected={expected}",
            g[i]
        );
    }
}

#[test]
fn test_ext_chain_rule_triple_compose_fd() {
    // f(x) = tanh(sigmoid(exp(x)))
    let x_data = vec![-0.5_f32, 0.0, 0.3, 0.5];
    let x_var = var_from(x_data.clone(), &[4]);
    let tx = tracked(&x_var);

    let y = tx.exp().unwrap().sigmoid().unwrap().tanh().unwrap();
    let loss = scalar_loss(&y);
    let grads = backward(&loss).unwrap();

    let gx = grad_vec(&grads, &x_var);
    let num = numerical_grad(&x_data, 1e-4, |d| {
        let t = DynTensor::from_vec(d, &[4], &cpu()).unwrap();
        sum_all(&t.exp().unwrap().sigmoid().unwrap().tanh().unwrap())
    });
    assert_grad_close(&gx, &num, 1e-2, "chain_triple_fd");
}

#[test]
fn test_ext_chain_rule_product_fd() {
    // f(x) = x^2 * sin(x), f'(x) = 2x*sin(x) + x^2*cos(x)
    let x_data = vec![0.5_f32, 1.0, -0.7, 2.0];
    let x_var = var_from(x_data.clone(), &[4]);
    let tx = tracked(&x_var);

    let x_sq = tx.sqr().unwrap();
    let sin_x = tx.sin().unwrap();
    let prod = x_sq.mul(&sin_x).unwrap();
    let loss = scalar_loss(&prod);
    let grads = backward(&loss).unwrap();

    let gx = grad_vec(&grads, &x_var);
    for (i, &v) in x_data.iter().enumerate() {
        let expected = 2.0 * v * v.sin() + v * v * v.cos();
        assert!(
            (gx[i] - expected).abs() < 1e-3,
            "chain_product grad[{i}]={}, expected={expected}",
            gx[i]
        );
    }
}

// ===========================================================================
// 12. Broadcast backward: reduction to original shape
// ===========================================================================

#[test]
fn test_ext_broadcast_backward_scalar_to_vector() {
    // x: [1] broadcast to [3], then sum. Gradient should accumulate.
    let x_var = var_from(vec![2.0], &[1]);
    let tx = tracked(&x_var);

    let expanded = tx.broadcast_as(&[3]).unwrap();
    let loss = scalar_loss(&expanded);
    let grads = backward(&loss).unwrap();

    // Gradient reduces from [3] to [1]: sum of [1, 1, 1] = 3
    let g = grad_vec(&grads, &x_var);
    assert_close(&g, &[3.0], 1e-5, "broadcast_scalar_to_vec");
}

#[test]
fn test_ext_broadcast_backward_row_to_matrix() {
    // x: [1, 3] broadcast to [2, 3]
    let x_var = var_from(vec![1.0, 2.0, 3.0], &[1, 3]);
    let tx = tracked(&x_var);

    let expanded = tx.broadcast_as(&[2, 3]).unwrap();
    let loss = scalar_loss(&expanded);
    let grads = backward(&loss).unwrap();

    let gx = grads.get(&x_var).unwrap();
    assert_eq!(gx.dims(), &[1, 3], "broadcast grad shape matches original");

    // Gradient reduces from [2, 3] to [1, 3]: sum over dim 0 => [2, 2, 2]
    let g = grad_vec(&grads, &x_var);
    assert_close(&g, &[2.0, 2.0, 2.0], 1e-5, "broadcast_row_to_matrix");
}

#[test]
fn test_ext_broadcast_backward_col_to_matrix() {
    // x: [2, 1] broadcast to [2, 3]
    let x_var = var_from(vec![1.0, 2.0], &[2, 1]);
    let tx = tracked(&x_var);

    let expanded = tx.broadcast_as(&[2, 3]).unwrap();
    let loss = scalar_loss(&expanded);
    let grads = backward(&loss).unwrap();

    let gx = grads.get(&x_var).unwrap();
    assert_eq!(gx.dims(), &[2, 1], "broadcast grad shape matches original");

    // Gradient reduces from [2, 3] to [2, 1]: sum over dim 1 => [3, 3]
    let g = grad_vec(&grads, &x_var);
    assert_close(&g, &[3.0, 3.0], 1e-5, "broadcast_col_to_matrix");
}

#[test]
fn test_ext_broadcast_backward_with_computation_fd() {
    // x: [1, 3], broadcast to [2, 3], then sqr, then sum
    let x_data = vec![1.0_f32, 2.0, 3.0];
    let x_var = var_from(x_data.clone(), &[1, 3]);
    let tx = tracked(&x_var);

    let expanded = tx.broadcast_as(&[2, 3]).unwrap();
    let y = expanded.sqr().unwrap();
    let loss = scalar_loss(&y);
    let grads = backward(&loss).unwrap();

    let gx = grad_vec(&grads, &x_var);
    let num = numerical_grad(&x_data, 1e-3, |d| {
        let t = DynTensor::from_vec(d, &[1, 3], &cpu()).unwrap();
        let e = t.broadcast_as([2, 3]).unwrap();
        sum_all(&e.sqr().unwrap())
    });
    assert_grad_close(&gx, &num, 1e-2, "broadcast_sqr_fd");
}

// ===========================================================================
// 13. Reshape backward: gradient has same shape as input
// ===========================================================================

#[test]
fn test_ext_reshape_backward_preserves_shape() {
    let x_var = var_from(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
    let tx = tracked(&x_var);

    let flat = tx.reshape(&[6]).unwrap();
    let y = flat.sqr().unwrap();
    let loss = scalar_loss(&y);
    let grads = backward(&loss).unwrap();

    let gx = grads.get(&x_var).unwrap();
    assert_eq!(
        gx.dims(),
        &[2, 3],
        "reshape grad shape must match input [2,3]"
    );
}

#[test]
fn test_ext_reshape_backward_values() {
    // loss = sum((reshape(x))^2) = sum(x^2), grad = 2*x regardless of reshape
    let x_data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let x_var = var_from(x_data.clone(), &[2, 3]);
    let tx = tracked(&x_var);

    let flat = tx.reshape(&[3, 2]).unwrap();
    let y = flat.sqr().unwrap();
    let loss = scalar_loss(&y);
    let grads = backward(&loss).unwrap();

    let g = grad_vec(&grads, &x_var);
    for (i, &v) in x_data.iter().enumerate() {
        let expected = 2.0 * v;
        assert!(
            (g[i] - expected).abs() < 1e-5,
            "reshape grad[{i}]={}, expected={expected}",
            g[i]
        );
    }
}

#[test]
fn test_ext_reshape_backward_3d_to_1d() {
    let x_data: Vec<f32> = (1..=24).map(|i| i as f32).collect();
    let x_var = var_from(x_data, &[2, 3, 4]);
    let tx = tracked(&x_var);

    let flat = tx.reshape(&[24]).unwrap();
    let loss = scalar_loss(&flat);
    let grads = backward(&loss).unwrap();

    let gx = grads.get(&x_var).unwrap();
    assert_eq!(gx.dims(), &[2, 3, 4], "reshape 3d->1d grad shape");

    // loss = sum(x), grad = all ones
    let g = grad_vec(&grads, &x_var);
    assert_close(&g, &[1.0; 24], 1e-6, "reshape_3d_to_1d");
}

#[test]
fn test_ext_reshape_backward_fd() {
    let x_data = vec![0.5_f32, -0.3, 1.2, 0.8, -0.6, 0.4];
    let x_var = var_from(x_data.clone(), &[2, 3]);
    let tx = tracked(&x_var);

    let reshaped = tx.reshape(&[3, 2]).unwrap();
    let y = reshaped.exp().unwrap();
    let loss = scalar_loss(&y);
    let grads = backward(&loss).unwrap();

    let gx = grad_vec(&grads, &x_var);
    let num = numerical_grad(&x_data, 1e-4, |d| {
        let t = DynTensor::from_vec(d, &[2, 3], &cpu()).unwrap();
        sum_all(&t.reshape([3, 2]).unwrap().exp().unwrap())
    });
    assert_grad_close(&gx, &num, 1e-2, "reshape_exp_fd");
}

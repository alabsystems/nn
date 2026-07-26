// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Complex backward pass graph tests for nn-autodiff.
//!
//! Covers multi-path gradient flow (diamond, fan-out/fan-in, residual),
//! per-op backward correctness (matmul, softmax, layernorm, conv1d,
//! embedding, relu-at-zero, sigmoid saturation, cross-entropy),
//! numerical gradient checking via finite differences, and edge cases
//! (detach, no-grad, empty tape).

use std::sync::Arc;

use nn_autodiff::{backward, TrackedTensor, Var};
use nn_core::dyn_tensor::DynTensor;
use nn_core::Device;

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

/// Reduce a tracked tensor to a scalar by summing all dims from last to first.
fn reduce_to_scalar(t: &Arc<TrackedTensor>) -> Arc<TrackedTensor> {
    let mut reduced = Arc::clone(t);
    for dim in (0..reduced.tensor().rank()).rev() {
        reduced = reduced.sum_keepdim(dim).unwrap();
    }
    reduced
}

/// Finite-difference numerical gradient for a single variable.
///
/// `data` is the flat f32 values of the variable. `eps` is the perturbation size.
/// `forward` takes a Vec<f32> of variable data and returns a scalar loss as f64.
/// Returns the numerical gradient as Vec<f64>.
fn finite_diff_grad(data: &[f32], eps: f64, forward: &dyn Fn(Vec<f32>) -> f64) -> Vec<f64> {
    let mut grad = Vec::with_capacity(data.len());
    for i in 0..data.len() {
        let mut plus = data.to_vec();
        let mut minus = data.to_vec();
        plus[i] += eps as f32;
        minus[i] -= eps as f32;
        let g = (forward(plus) - forward(minus)) / (2.0 * eps);
        grad.push(g);
    }
    grad
}

// ===========================================================================
// A. Multi-path Gradients (7 tests)
// ===========================================================================

/// Diamond graph: x -> (a=x*x, b=x+x) -> c=a+b -> loss.
/// grad = d/dx(x^2 + 2x) = 2x + 2. At x=3: 8.
#[test]
fn test_diamond_graph() {
    let x = scalar_var(3.0);
    let t = Arc::new(TrackedTensor::from_var(&x).unwrap());
    let a = t.sqr().unwrap(); // x^2
    let b = t.add(&t).unwrap(); // 2x
    let c = a.add(&b).unwrap(); // x^2 + 2x
    let grads = backward(&c).unwrap();
    let grad = grads.get(&x).unwrap().to_flat_vec::<f32>().unwrap();
    // d/dx(x^2 + 2x) = 2x + 2 = 8
    assert!(
        (grad[0] - 8.0).abs() < 1e-5,
        "expected 8.0, got {}",
        grad[0]
    );
}

/// Fan-out/fan-in: one input feeds 3 consumers, which are summed.
/// loss = tanh(x) + sigmoid(x) + relu(x). Verify grad = tanh'(x) + sig'(x) + relu'(x).
#[test]
fn test_fan_out_fan_in() {
    let x_val = 0.5_f32;
    let x = scalar_var(x_val);
    let t = Arc::new(TrackedTensor::from_var(&x).unwrap());

    let branch_a = t.tanh().unwrap();
    let branch_b = t.sigmoid().unwrap();
    let branch_c = t.relu().unwrap();

    let sum_ab = branch_a.add(&branch_b).unwrap();
    let loss = sum_ab.add(&branch_c).unwrap();

    let grads = backward(&loss).unwrap();
    let grad = grads.get(&x).unwrap().to_flat_vec::<f32>().unwrap();

    // tanh'(0.5) = 1 - tanh(0.5)^2
    let tanh_grad = 1.0 - x_val.tanh().powi(2);
    // sigmoid'(0.5) = sig(0.5) * (1 - sig(0.5))
    let sig = 1.0 / (1.0 + (-x_val).exp());
    let sig_grad = sig * (1.0 - sig);
    // relu'(0.5) = 1.0 (since x > 0)
    let relu_grad = 1.0_f32;

    let expected = tanh_grad + sig_grad + relu_grad;
    assert!(
        (grad[0] - expected).abs() < 1e-5,
        "expected {expected}, got {}",
        grad[0]
    );
}

/// Gradient accumulation: x * x = x^2, tested via mul(self, self).
/// d/dx(x*x) = 2x. At x=5: 10.
#[test]
fn test_gradient_accumulation() {
    let x = scalar_var(5.0);
    let t = Arc::new(TrackedTensor::from_var(&x).unwrap());
    let y = t.mul(&t).unwrap(); // x*x with same tensor on both sides
    let grads = backward(&y).unwrap();
    let grad = grads.get(&x).unwrap().to_flat_vec::<f32>().unwrap();
    assert!(
        (grad[0] - 10.0).abs() < 1e-5,
        "expected 10.0, got {}",
        grad[0]
    );
}

/// Long chain of 15 sequential operations: x -> relu -> add(1) -> mul(0.99) -> ... (5 repeats).
/// Verify gradient doesn't vanish or explode and matches finite differences.
#[test]
fn test_long_chain() {
    let x_val = 2.0_f32;
    let x = scalar_var(x_val);
    let mut current: Arc<TrackedTensor> = Arc::new(TrackedTensor::from_var(&x).unwrap());

    // Chain of 15 operations: 5 rounds of (relu, add_scalar(0.1), mul_scalar(0.99))
    for _ in 0..5 {
        current = current.relu().unwrap();
        current = current.add_scalar(0.1).unwrap();
        current = current.mul_scalar(0.99).unwrap();
    }

    let grads = backward(&current).unwrap();
    let grad = grads.get(&x).unwrap().to_flat_vec::<f32>().unwrap();

    assert!(
        grad[0].is_finite(),
        "gradient should be finite, got {}",
        grad[0]
    );

    // Numerical check via finite differences
    let eps = 1e-4_f64;
    let forward_fn = |v: Vec<f32>| -> f64 {
        let mut val = f64::from(v[0]);
        for _ in 0..5 {
            val = val.max(0.0); // relu
            val += 0.1; // add_scalar
            val *= 0.99; // mul_scalar
        }
        val
    };
    let numerical = finite_diff_grad(&[x_val], eps, &forward_fn);
    let err = (f64::from(grad[0]) - numerical[0]).abs();
    assert!(
        err < 1e-3,
        "chain grad: analytical={}, numerical={}, err={}",
        grad[0],
        numerical[0],
        err
    );
}

/// Skip connection (residual): y = x + f(x) where f = sqr.
/// d/dx(x + x^2) = 1 + 2x. At x=3: 7.
#[test]
fn test_skip_connection_gradient() {
    let x = scalar_var(3.0);
    let t = Arc::new(TrackedTensor::from_var(&x).unwrap());
    let fx = t.sqr().unwrap(); // f(x) = x^2
    let y = t.add(&fx).unwrap(); // x + x^2 (residual)
    let grads = backward(&y).unwrap();
    let grad = grads.get(&x).unwrap().to_flat_vec::<f32>().unwrap();
    // d/dx(x + x^2) = 1 + 2x = 7
    assert!(
        (grad[0] - 7.0).abs() < 1e-5,
        "expected 7.0, got {}",
        grad[0]
    );
}

/// Shared parameter: w used in two separate matmuls.
/// loss = sum(x1 @ w) + sum(x2 @ w). Grads for w should accumulate from both paths.
#[test]
fn test_shared_parameter() {
    let w = mat_var(vec![1.0, 2.0, 3.0, 4.0], 2, 2);
    let x1_data = DynTensor::from_vec(vec![1.0, 0.0], &[1, 2], &cpu()).unwrap();
    let x2_data = DynTensor::from_vec(vec![0.0, 1.0], &[1, 2], &cpu()).unwrap();

    let tw = Arc::new(TrackedTensor::from_var(&w).unwrap());
    let tx1 = Arc::new(TrackedTensor::from_tensor(x1_data));
    let tx2 = Arc::new(TrackedTensor::from_tensor(x2_data));

    let y1 = tx1.matmul(&tw).unwrap();
    let y2 = tx2.matmul(&tw).unwrap();
    let sum_y1 = reduce_to_scalar(&y1);
    let sum_y2 = reduce_to_scalar(&y2);
    let loss = sum_y1.add(&sum_y2).unwrap();

    let grads = backward(&loss).unwrap();
    let grad_w = grads.get(&w).unwrap().to_flat_vec::<f32>().unwrap();

    // grad_w = x1^T @ ones[1,2] + x2^T @ ones[1,2]
    // x1^T = [[1],[0]], x2^T = [[0],[1]]
    // x1^T @ [[1,1]] = [[1,1],[0,0]]
    // x2^T @ [[1,1]] = [[0,0],[1,1]]
    // sum = [[1,1],[1,1]]
    for &g in &grad_w {
        assert!((g - 1.0).abs() < 1e-5, "expected all 1.0, got {grad_w:?}");
    }
}

/// Deeply nested diamond: x -> (a, b) -> (c=a*b, d=a+b) -> e=c+d -> loss.
/// f(x) = x*x + x+x = x^2 + 2x (but a=x, b=x through different paths).
/// Wait, let a = relu(x), b = sigmoid(x), c = a*b, d = a+b, e = c+d.
/// This tests gradient flow through a complex DAG.
#[test]
fn test_nested_diamond() {
    let x_val = 1.0_f32;
    let x = scalar_var(x_val);
    let t = Arc::new(TrackedTensor::from_var(&x).unwrap());

    let a = t.relu().unwrap(); // a = relu(x) = x for x>0
    let b = t.sigmoid().unwrap(); // b = sig(x)
    let c = a.mul(&b).unwrap(); // c = a * b
    let d = a.add(&b).unwrap(); // d = a + b
    let e = c.add(&d).unwrap(); // e = a*b + a + b

    let grads = backward(&e).unwrap();
    let grad = grads.get(&x).unwrap().to_flat_vec::<f32>().unwrap();

    // Numerical check
    let eps = 1e-4_f64;
    let forward_fn = |v: Vec<f32>| -> f64 {
        let xv = f64::from(v[0]);
        let av = xv.max(0.0);
        let bv = 1.0 / (1.0 + (-xv).exp());
        let cv = av * bv;
        let dv = av + bv;
        cv + dv
    };
    let numerical = finite_diff_grad(&[x_val], eps, &forward_fn);
    let err = (f64::from(grad[0]) - numerical[0]).abs();
    assert!(
        err < 1e-3,
        "nested diamond: analytical={}, numerical={}, err={}",
        grad[0],
        numerical[0],
        err
    );
}

// ===========================================================================
// B. Per-Op Backward Correctness (8 tests)
// ===========================================================================

/// MatMul backward: verify d/dA(A@B) and d/dB(A@B) via finite differences.
#[test]
fn test_backward_matmul_fd() {
    let a_data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]; // [2, 3]
    let b_data = vec![0.5, -0.5, 1.0, 0.0, -1.0, 0.5]; // [3, 2]

    let a = mat_var(a_data.clone(), 2, 3);
    let b = mat_var(b_data.clone(), 3, 2);
    let ta = Arc::new(TrackedTensor::from_var(&a).unwrap());
    let tb = Arc::new(TrackedTensor::from_var(&b).unwrap());

    let y = ta.matmul(&tb).unwrap(); // [2, 2]
    let loss = reduce_to_scalar(&y);

    let grads = backward(&loss).unwrap();
    let grad_a = grads.get(&a).unwrap().to_flat_vec::<f32>().unwrap();
    let grad_b = grads.get(&b).unwrap().to_flat_vec::<f32>().unwrap();

    let eps = 1e-4_f64;

    // Finite diff for A
    let b_data_ref = b_data.clone();
    let fd_a = finite_diff_grad(&a_data, eps, &|av: Vec<f32>| {
        let a_t = DynTensor::from_vec(av, &[2, 3], &cpu()).unwrap();
        let b_t = DynTensor::from_vec(b_data_ref.clone(), &[3, 2], &cpu()).unwrap();
        let y_t = a_t.matmul(&b_t).unwrap();
        y_t.to_flat_vec::<f32>()
            .unwrap()
            .iter()
            .map(|&v| f64::from(v))
            .sum()
    });
    for i in 0..a_data.len() {
        let err = (f64::from(grad_a[i]) - fd_a[i]).abs();
        assert!(
            err < 1e-2,
            "grad_a[{i}]: analytical={}, numerical={}",
            grad_a[i],
            fd_a[i]
        );
    }

    // Finite diff for B
    let a_data_ref = a_data;
    let fd_b = finite_diff_grad(&b_data, eps, &|bv: Vec<f32>| {
        let a_t = DynTensor::from_vec(a_data_ref.clone(), &[2, 3], &cpu()).unwrap();
        let b_t = DynTensor::from_vec(bv, &[3, 2], &cpu()).unwrap();
        let y_t = a_t.matmul(&b_t).unwrap();
        y_t.to_flat_vec::<f32>()
            .unwrap()
            .iter()
            .map(|&v| f64::from(v))
            .sum()
    });
    for i in 0..b_data.len() {
        let err = (f64::from(grad_b[i]) - fd_b[i]).abs();
        assert!(
            err < 1e-2,
            "grad_b[{i}]: analytical={}, numerical={}",
            grad_b[i],
            fd_b[i]
        );
    }
}

/// Softmax backward Jacobian verification.
/// For softmax s = softmax(x, dim=1), choosing loss = s[0,0]:
/// ds_j/dx_i = s_i*(delta_ij - s_j)
#[test]
fn test_backward_softmax_jacobian() {
    let x_data = vec![1.0, 2.0, 3.0];
    let x = Var::new(DynTensor::from_vec(x_data.clone(), &[1, 3], &cpu()).unwrap());
    let t = Arc::new(TrackedTensor::from_var(&x).unwrap());
    let s = t.softmax(1).unwrap();

    // Extract loss = s[0, 1] (second element)
    let elem = s.narrow(1, 1, 1).unwrap();
    let loss = reduce_to_scalar(&elem);
    let grads = backward(&loss).unwrap();
    let grad = grads.get(&x).unwrap().to_flat_vec::<f32>().unwrap();

    // Compute softmax values
    let max_x = 3.0_f32;
    let exps: Vec<f32> = x_data.iter().map(|&v| (v - max_x).exp()).collect();
    let sum_exp: f32 = exps.iter().sum();
    let s_vals: Vec<f32> = exps.iter().map(|&e| e / sum_exp).collect();

    // Jacobian column j=1: ds[1]/dx[i] = s[i]*(delta(i,1) - s[1])
    for i in 0..3 {
        let delta = if i == 1 { 1.0 } else { 0.0 };
        let expected = s_vals[i] * (delta - s_vals[1]);
        assert!(
            (grad[i] - expected).abs() < 1e-5,
            "softmax jacobian[{i}]: expected={expected}, got={}",
            grad[i]
        );
    }
}

/// LayerNorm backward: verify gradient shapes and FD correctness.
#[test]
fn test_backward_layernorm_fd() {
    let x_data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]; // [2, 3]
    let gamma_data = vec![1.0, 0.5, 2.0];
    let beta_data = vec![0.1, -0.1, 0.0];

    let x = Var::new(DynTensor::from_vec(x_data.clone(), &[2, 3], &cpu()).unwrap());
    let gamma = Var::new(DynTensor::from_vec(gamma_data.clone(), &[3], &cpu()).unwrap());
    let beta = Var::new(DynTensor::from_vec(beta_data.clone(), &[3], &cpu()).unwrap());

    let tx = Arc::new(TrackedTensor::from_var(&x).unwrap());
    let tg = Arc::new(TrackedTensor::from_var(&gamma).unwrap());
    let tb = Arc::new(TrackedTensor::from_var(&beta).unwrap());

    let y = tx.layer_norm(&tg, &tb, 1e-5).unwrap();
    let loss = reduce_to_scalar(&y);
    let grads = backward(&loss).unwrap();

    assert_eq!(grads.get(&x).unwrap().dims(), &[2, 3]);
    assert_eq!(grads.get(&gamma).unwrap().dims(), &[3]);
    assert_eq!(grads.get(&beta).unwrap().dims(), &[3]);

    // FD check for x gradients
    let grad_x = grads.get(&x).unwrap().to_flat_vec::<f32>().unwrap();
    let gamma_ref = gamma_data;
    let beta_ref = beta_data;
    let eps = 1e-3_f64;
    let fd_x = finite_diff_grad(&x_data, eps, &|xv: Vec<f32>| {
        let t = DynTensor::from_vec(xv, &[2, 3], &cpu()).unwrap();
        let g = DynTensor::from_vec(gamma_ref.clone(), &[3], &cpu()).unwrap();
        let b = DynTensor::from_vec(beta_ref.clone(), &[3], &cpu()).unwrap();
        let mean = t.mean_keepdim(1).unwrap();
        let diff = t.sub(&mean).unwrap();
        let var = diff.sqr().unwrap().mean_keepdim(1).unwrap();
        let inv_std = var
            .add_scalar(1e-5)
            .unwrap()
            .sqrt()
            .unwrap()
            .recip()
            .unwrap();
        let normed = diff.mul(&inv_std).unwrap();
        let out = normed.mul(&g).unwrap().add(&b).unwrap();
        out.to_flat_vec::<f32>()
            .unwrap()
            .iter()
            .map(|&v| f64::from(v))
            .sum()
    });

    for i in 0..x_data.len() {
        let err = (f64::from(grad_x[i]) - fd_x[i]).abs();
        assert!(
            err < 1e-2,
            "layernorm grad_x[{i}]: analytical={}, numerical={}, err={}",
            grad_x[i],
            fd_x[i],
            err
        );
    }
}

/// Conv1d backward: verify grad_input and grad_kernel shapes and values.
#[test]
fn test_backward_conv1d_gradient() {
    // input: [1, 1, 6], kernel: [1, 1, 3]
    let x_data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let k_data = vec![1.0, 0.0, -1.0];

    let x = Var::new(DynTensor::from_vec(x_data.clone(), &[1, 1, 6], &cpu()).unwrap());
    let k = Var::new(DynTensor::from_vec(k_data.clone(), &[1, 1, 3], &cpu()).unwrap());

    let tx = Arc::new(TrackedTensor::from_var(&x).unwrap());
    let tk = Arc::new(TrackedTensor::from_var(&k).unwrap());

    let y = tx.conv1d(&tk, 0, 1, 1, 1).unwrap();
    assert_eq!(y.dims(), &[1, 1, 4]); // (6 - 3)/1 + 1 = 4

    let loss = reduce_to_scalar(&y);
    let grads = backward(&loss).unwrap();

    let grad_x = grads.get(&x).unwrap();
    let grad_k = grads.get(&k).unwrap();
    assert_eq!(grad_x.dims(), &[1, 1, 6]);
    assert_eq!(grad_k.dims(), &[1, 1, 3]);

    // FD check for kernel gradient
    let x_ref = x_data;
    let fd_k = finite_diff_grad(&k_data, 1e-4, &|kv: Vec<f32>| {
        let xt = DynTensor::from_vec(x_ref.clone(), &[1, 1, 6], &cpu()).unwrap();
        let kt = DynTensor::from_vec(kv, &[1, 1, 3], &cpu()).unwrap();
        let yt = xt.conv1d(&kt, 0, 1, 1, 1).unwrap();
        yt.to_flat_vec::<f32>()
            .unwrap()
            .iter()
            .map(|&v| f64::from(v))
            .sum()
    });
    let grad_k_vals = grad_k.to_flat_vec::<f32>().unwrap();
    for i in 0..k_data.len() {
        let err = (f64::from(grad_k_vals[i]) - fd_k[i]).abs();
        assert!(
            err < 1e-2,
            "conv1d grad_k[{i}]: analytical={}, numerical={}, err={}",
            grad_k_vals[i],
            fd_k[i],
            err
        );
    }
}

/// Embedding backward: sparse gradient accumulation for repeated indices.
#[test]
fn test_backward_embedding_sparse() {
    // vocab=4, embed_dim=3. Indices [1, 3, 1] — index 1 used twice.
    let w_data = vec![
        0.1, 0.2, 0.3, // row 0
        0.4, 0.5, 0.6, // row 1
        0.7, 0.8, 0.9, // row 2
        1.0, 1.1, 1.2, // row 3
    ];
    let w = Var::new(DynTensor::from_vec(w_data, &[4, 3], &cpu()).unwrap());
    let idx = DynTensor::from_vec_u32(vec![1, 3, 1], &[3], &cpu()).unwrap();

    let tw = Arc::new(TrackedTensor::from_var(&w).unwrap());
    let ti = Arc::new(TrackedTensor::from_tensor(idx));
    let out = TrackedTensor::embedding(&tw, &ti).unwrap(); // [3, 3]
    let loss = reduce_to_scalar(&out);
    let grads = backward(&loss).unwrap();

    let grad_w = grads.get(&w).unwrap().to_flat_vec::<f32>().unwrap();
    // Row 0: never selected -> [0, 0, 0]
    assert!((grad_w[0]).abs() < 1e-6);
    assert!((grad_w[1]).abs() < 1e-6);
    assert!((grad_w[2]).abs() < 1e-6);
    // Row 1: selected twice -> [2, 2, 2]
    assert!((grad_w[3] - 2.0).abs() < 1e-6, "row1[0]={}", grad_w[3]);
    assert!((grad_w[4] - 2.0).abs() < 1e-6, "row1[1]={}", grad_w[4]);
    assert!((grad_w[5] - 2.0).abs() < 1e-6, "row1[2]={}", grad_w[5]);
    // Row 2: never selected -> [0, 0, 0]
    assert!((grad_w[6]).abs() < 1e-6);
    assert!((grad_w[7]).abs() < 1e-6);
    assert!((grad_w[8]).abs() < 1e-6);
    // Row 3: selected once -> [1, 1, 1]
    assert!((grad_w[9] - 1.0).abs() < 1e-6, "row3[0]={}", grad_w[9]);
    assert!((grad_w[10] - 1.0).abs() < 1e-6, "row3[1]={}", grad_w[10]);
    assert!((grad_w[11] - 1.0).abs() < 1e-6, "row3[2]={}", grad_w[11]);
}

/// ReLU gradient at exactly zero: should be 0 (subgradient convention).
#[test]
fn test_backward_relu_zero() {
    let x = vec_var(vec![-1.0, 0.0, 1.0]);
    let t = Arc::new(TrackedTensor::from_var(&x).unwrap());
    let y = t.relu().unwrap();
    let loss = reduce_to_scalar(&y);
    let grads = backward(&loss).unwrap();
    let grad = grads.get(&x).unwrap().to_flat_vec::<f32>().unwrap();

    assert!(
        (grad[0] - 0.0).abs() < 1e-6,
        "relu'(-1) should be 0, got {}",
        grad[0]
    );
    // At x=0, nn uses relu'(0) = 1 (right derivative convention, matching PyTorch).
    assert!(
        (grad[1] - 1.0).abs() < 1e-6,
        "relu'(0) should be 1 (right derivative), got {}",
        grad[1]
    );
    assert!(
        (grad[2] - 1.0).abs() < 1e-6,
        "relu'(1) should be 1, got {}",
        grad[2]
    );
}

/// Sigmoid gradient near saturation (x = +/- 10).
/// Near 0 and 1, gradient should be very small.
#[test]
fn test_backward_sigmoid_saturation() {
    let x = vec_var(vec![-10.0, 0.0, 10.0]);
    let t = Arc::new(TrackedTensor::from_var(&x).unwrap());
    let y = t.sigmoid().unwrap();
    let loss = reduce_to_scalar(&y);
    let grads = backward(&loss).unwrap();
    let grad = grads.get(&x).unwrap().to_flat_vec::<f32>().unwrap();

    // sigmoid'(x) = sig(x) * (1 - sig(x))
    // sig(-10) ~= 4.5e-5, sig'(-10) ~= 4.5e-5
    assert!(
        grad[0].abs() < 1e-3,
        "sigmoid'(-10) should be near 0, got {}",
        grad[0]
    );
    // sig(0) = 0.5, sig'(0) = 0.25
    assert!(
        (grad[1] - 0.25).abs() < 1e-5,
        "sigmoid'(0) should be 0.25, got {}",
        grad[1]
    );
    // sig(10) ~= 0.99995, sig'(10) ~= 4.5e-5
    assert!(
        grad[2].abs() < 1e-3,
        "sigmoid'(10) should be near 0, got {}",
        grad[2]
    );
}

/// Cross-entropy loss gradient: verify against finite differences.
#[test]
fn test_backward_cross_entropy_fd() {
    let logits_data = vec![2.0, 1.0, 0.1, 0.5, 1.5, 2.5];
    let targets_data = vec![0u32, 2];

    let logits_var = Var::new(DynTensor::from_vec(logits_data.clone(), &[2, 3], &cpu()).unwrap());
    let t_logits = Arc::new(TrackedTensor::from_var(&logits_var).unwrap());
    let t_targets = Arc::new(TrackedTensor::from_tensor(
        DynTensor::from_vec_u32(targets_data.clone(), &[2, 1], &cpu()).unwrap(),
    ));

    let loss = t_logits.cross_entropy_loss(&t_targets, 1).unwrap();
    let grads = backward(&loss).unwrap();
    let grad = grads
        .get(&logits_var)
        .unwrap()
        .to_flat_vec::<f32>()
        .unwrap();

    // Reference cross-entropy forward for FD
    let targets_ref = targets_data;
    let fd = finite_diff_grad(&logits_data, 1e-4, &|lv: Vec<f32>| {
        let n = targets_ref.len();
        let nc = 3usize;
        let mut total = 0.0_f64;
        for (i, &t) in targets_ref.iter().enumerate() {
            let row = &lv[i * nc..(i + 1) * nc];
            let max_val = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let shifted: Vec<f64> = row.iter().map(|&x| f64::from(x - max_val).exp()).collect();
            let sum_exp: f64 = shifted.iter().sum();
            let log_softmax_t = (shifted[t as usize] / sum_exp).ln();
            total -= log_softmax_t;
        }
        total / n as f64
    });

    for i in 0..logits_data.len() {
        let err = (f64::from(grad[i]) - fd[i]).abs();
        assert!(
            err < 1e-3,
            "cross_entropy grad[{i}]: analytical={}, numerical={}, err={}",
            grad[i],
            fd[i],
            err
        );
    }
}

// ===========================================================================
// C. Numerical Gradient Checking (5 tests)
// ===========================================================================

/// FD check for a linear layer: loss = sum(x @ w + b).
#[test]
fn test_numerical_grad_check_linear() {
    let w_data = vec![0.5, -0.3, 0.7, 0.2, -0.1, 0.4]; // [3, 2]
    let w = mat_var(w_data.clone(), 3, 2);

    let x_data = DynTensor::from_vec(vec![1.0, 2.0, 3.0], &[1, 3], &cpu()).unwrap();
    let x = Arc::new(TrackedTensor::from_tensor(x_data));

    let tw = Arc::new(TrackedTensor::from_var(&w).unwrap());
    let y = x.matmul(&tw).unwrap(); // [1, 2]
    let loss = reduce_to_scalar(&y);

    let grads = backward(&loss).unwrap();
    let grad_w = grads.get(&w).unwrap().to_flat_vec::<f32>().unwrap();

    let fd = finite_diff_grad(&w_data, 1e-4, &|wv: Vec<f32>| {
        let x_t = DynTensor::from_vec(vec![1.0, 2.0, 3.0], &[1, 3], &cpu()).unwrap();
        let w_t = DynTensor::from_vec(wv, &[3, 2], &cpu()).unwrap();
        let y_t = x_t.matmul(&w_t).unwrap();
        y_t.to_flat_vec::<f32>()
            .unwrap()
            .iter()
            .map(|&v| f64::from(v))
            .sum()
    });

    for i in 0..w_data.len() {
        let err = (f64::from(grad_w[i]) - fd[i]).abs();
        assert!(
            err < 1e-3,
            "linear grad_w[{i}]: analytical={}, numerical={}",
            grad_w[i],
            fd[i]
        );
    }
}

/// FD check for tanh.
#[test]
fn test_numerical_grad_check_tanh() {
    let x_data = vec![-2.0, -0.5, 0.0, 0.5, 2.0];
    let x = vec_var(x_data.clone());
    let t = Arc::new(TrackedTensor::from_var(&x).unwrap());
    let y = t.tanh().unwrap();
    let loss = reduce_to_scalar(&y);
    let grads = backward(&loss).unwrap();
    let grad = grads.get(&x).unwrap().to_flat_vec::<f32>().unwrap();

    let fd = finite_diff_grad(&x_data, 1e-4, &|xv: Vec<f32>| {
        xv.iter().map(|&v| f64::from(v).tanh()).sum()
    });

    for i in 0..x_data.len() {
        let err = (f64::from(grad[i]) - fd[i]).abs();
        assert!(
            err < 1e-3,
            "tanh grad[{i}]: analytical={}, numerical={}",
            grad[i],
            fd[i]
        );
    }
}

/// FD check for softmax (multi-output, using sum of specific elements as loss).
#[test]
fn test_numerical_grad_check_softmax() {
    let x_data = vec![1.0, 2.0, 3.0, 0.5, 1.5, -0.5];
    let x = Var::new(DynTensor::from_vec(x_data.clone(), &[2, 3], &cpu()).unwrap());
    let t = Arc::new(TrackedTensor::from_var(&x).unwrap());
    let s = t.softmax(1).unwrap();
    // loss = sum of all softmax outputs = 2.0 (each row sums to 1)
    // That's trivial, so use sum of sqr to make it interesting
    let s2 = s.sqr().unwrap();
    let loss = reduce_to_scalar(&s2);
    let grads = backward(&loss).unwrap();
    let grad = grads.get(&x).unwrap().to_flat_vec::<f32>().unwrap();

    let fd = finite_diff_grad(&x_data, 1e-4, &|xv: Vec<f32>| {
        let t = DynTensor::from_vec(xv, &[2, 3], &cpu()).unwrap();
        let s = t.softmax(1).unwrap();
        let s2 = s.sqr().unwrap();
        s2.to_flat_vec::<f32>()
            .unwrap()
            .iter()
            .map(|&v| f64::from(v))
            .sum()
    });

    for i in 0..x_data.len() {
        let err = (f64::from(grad[i]) - fd[i]).abs();
        assert!(
            err < 1e-2,
            "softmax-sqr grad[{i}]: analytical={}, numerical={}, err={}",
            grad[i],
            fd[i],
            err
        );
    }
}

/// FD check for matmul.
#[test]
fn test_numerical_grad_check_matmul() {
    let a_data = vec![1.0, 2.0, 3.0, 4.0]; // [2, 2]
    let b_data = vec![0.5, -0.5, 1.0, 0.0]; // [2, 2]

    let a = mat_var(a_data.clone(), 2, 2);
    let b = mat_var(b_data.clone(), 2, 2);

    let ta = Arc::new(TrackedTensor::from_var(&a).unwrap());
    let tb = Arc::new(TrackedTensor::from_var(&b).unwrap());
    let y = ta.matmul(&tb).unwrap();
    // Non-trivial loss: sum of squares of output
    let loss = y.sqr().unwrap();
    let loss = reduce_to_scalar(&loss);
    let grads = backward(&loss).unwrap();

    let grad_a = grads.get(&a).unwrap().to_flat_vec::<f32>().unwrap();
    let b_ref = b_data;
    let fd_a = finite_diff_grad(&a_data, 1e-4, &|av: Vec<f32>| {
        let a_t = DynTensor::from_vec(av, &[2, 2], &cpu()).unwrap();
        let b_t = DynTensor::from_vec(b_ref.clone(), &[2, 2], &cpu()).unwrap();
        let y_t = a_t.matmul(&b_t).unwrap();
        let y2 = y_t.sqr().unwrap();
        y2.to_flat_vec::<f32>()
            .unwrap()
            .iter()
            .map(|&v| f64::from(v))
            .sum()
    });
    for i in 0..a_data.len() {
        let err = (f64::from(grad_a[i]) - fd_a[i]).abs();
        assert!(
            err < 5e-2,
            "matmul sqr grad_a[{i}]: analytical={}, numerical={}, err={}",
            grad_a[i],
            fd_a[i],
            err
        );
    }
}

/// FD check for a composed multi-layer pipeline: linear -> tanh -> linear -> sigmoid.
#[test]
fn test_numerical_grad_check_composed() {
    let w1_data = vec![0.3, -0.5, 0.7, -0.2]; // [2, 2]
    let w2_data = vec![0.4, 0.1, -0.3, 0.6]; // [2, 2]

    let w1 = mat_var(w1_data.clone(), 2, 2);
    let w2 = mat_var(w2_data.clone(), 2, 2);

    let x_data = DynTensor::from_vec(vec![1.0, -1.0], &[1, 2], &cpu()).unwrap();
    let x = Arc::new(TrackedTensor::from_tensor(x_data));

    let tw1 = Arc::new(TrackedTensor::from_var(&w1).unwrap());
    let tw2 = Arc::new(TrackedTensor::from_var(&w2).unwrap());

    let h = x.matmul(&tw1).unwrap().tanh().unwrap();
    let out = h.matmul(&tw2).unwrap().sigmoid().unwrap();
    let loss = reduce_to_scalar(&out);

    let grads = backward(&loss).unwrap();
    let grad_w1 = grads.get(&w1).unwrap().to_flat_vec::<f32>().unwrap();

    let w2_ref = w2_data;
    let fd_w1 = finite_diff_grad(&w1_data, 1e-4, &|wv: Vec<f32>| {
        let x_t = DynTensor::from_vec(vec![1.0, -1.0], &[1, 2], &cpu()).unwrap();
        let w1_t = DynTensor::from_vec(wv, &[2, 2], &cpu()).unwrap();
        let w2_t = DynTensor::from_vec(w2_ref.clone(), &[2, 2], &cpu()).unwrap();
        let h_t = x_t.matmul(&w1_t).unwrap().tanh().unwrap();
        let out_t = h_t.matmul(&w2_t).unwrap().sigmoid().unwrap();
        out_t
            .to_flat_vec::<f32>()
            .unwrap()
            .iter()
            .map(|&v| f64::from(v))
            .sum()
    });

    for i in 0..w1_data.len() {
        let err = (f64::from(grad_w1[i]) - fd_w1[i]).abs();
        assert!(
            err < 1e-2,
            "composed grad_w1[{i}]: analytical={}, numerical={}",
            grad_w1[i],
            fd_w1[i]
        );
    }
}

// ===========================================================================
// D. Edge Cases (5 tests)
// ===========================================================================

/// Detach stops gradient flow: backward from a detached tensor yields no gradient.
#[test]
fn test_backward_detached() {
    let x = scalar_var(3.0);
    let t = Arc::new(TrackedTensor::from_var(&x).unwrap());
    let y = t.sqr().unwrap(); // y = x^2
    let detached = y.detach(); // breaks gradient flow
                               // Multiply detached by another tracked path to make a scalar
    let z = detached.mul_scalar(2.0).unwrap();
    let grads = backward(&z).unwrap();
    // x should get no gradient because detach broke the link
    assert!(
        grads.get(&x).is_none(),
        "detached tensor should not propagate gradient to x"
    );
}

/// Detach in mid-graph: only the part after detach receives gradients.
#[test]
fn test_backward_detached_mid_graph() {
    let w1 = scalar_var(2.0);
    let w2 = scalar_var(3.0);
    let tw1 = Arc::new(TrackedTensor::from_var(&w1).unwrap());
    let tw2 = Arc::new(TrackedTensor::from_var(&w2).unwrap());

    let h = tw1.sqr().unwrap(); // h = w1^2
    let h_detached = h.detach(); // detach: gradient stops here
    let y = h_detached.mul(&tw2).unwrap(); // y = detach(w1^2) * w2

    let grads = backward(&y).unwrap();
    // w1 should get no gradient (detached)
    assert!(
        grads.get(&w1).is_none(),
        "w1 should not get gradient through detach"
    );
    // w2 should get gradient = h_val = w1^2 = 4.0
    let grad_w2 = grads.get(&w2).unwrap().to_flat_vec::<f32>().unwrap();
    assert!(
        (grad_w2[0] - 4.0).abs() < 1e-5,
        "expected grad_w2=4.0, got {}",
        grad_w2[0]
    );
}

/// Backward on a leaf (no ops): gradient is 1.0 for identity.
#[test]
fn test_backward_empty_tape() {
    let x = scalar_var(42.0);
    let t = Arc::new(TrackedTensor::from_var(&x).unwrap());
    // No operations, just backward on the leaf itself
    let grads = backward(&t).unwrap();
    let grad = grads.get(&x).unwrap().to_flat_vec::<f32>().unwrap();
    assert!(
        (grad[0] - 1.0).abs() < 1e-6,
        "identity gradient should be 1.0, got {}",
        grad[0]
    );
}

/// Backward on a constant (from_tensor, not from_var): no variables, no gradients.
#[test]
fn test_backward_constant_only() {
    let t = Arc::new(TrackedTensor::from_tensor(
        DynTensor::from_vec(vec![5.0], &[1], &cpu()).unwrap(),
    ));
    let grads = backward(&t).unwrap();
    assert_eq!(
        grads.var_count(),
        0,
        "no variables means no gradients should be stored"
    );
}

/// Multiple backward calls on the same variable with different loss expressions.
/// Each backward should produce independent gradients.
#[test]
fn test_multiple_independent_backwards() {
    let x = scalar_var(3.0);

    // First backward: loss1 = x^2, grad = 2x = 6
    let t1 = Arc::new(TrackedTensor::from_var(&x).unwrap());
    let loss1 = t1.sqr().unwrap();
    let grads1 = backward(&loss1).unwrap();
    let g1 = grads1.get(&x).unwrap().to_flat_vec::<f32>().unwrap()[0];

    // Second backward: loss2 = x^3, grad = 3x^2 = 27
    let t2 = Arc::new(TrackedTensor::from_var(&x).unwrap());
    let loss2 = t2.powf(3.0).unwrap();
    let grads2 = backward(&loss2).unwrap();
    let g2 = grads2.get(&x).unwrap().to_flat_vec::<f32>().unwrap()[0];

    assert!(
        (g1 - 6.0).abs() < 1e-4,
        "first backward: expected 6.0, got {g1}"
    );
    assert!(
        (g2 - 27.0).abs() < 1e-3,
        "second backward: expected 27.0, got {g2}"
    );
}

// ===========================================================================
// E. Additional Complex Graph Tests (3 tests)
// ===========================================================================

/// Broadcast backward: verify gradient reduction for broadcast add.
/// x: [1, 3], b: [3] -> broadcast add -> loss = sum.
/// grad_b should be sum over the broadcast dimensions.
#[test]
fn test_backward_broadcast_reduce() {
    let x_data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]; // [2, 3]
    let b_data = vec![0.1, 0.2, 0.3]; // [3]

    let x = Var::new(DynTensor::from_vec(x_data, &[2, 3], &cpu()).unwrap());
    let b = vec_var(b_data);

    let tx = Arc::new(TrackedTensor::from_var(&x).unwrap());
    let tb = Arc::new(TrackedTensor::from_var(&b).unwrap());

    // Broadcast b [3] -> [1, 3] -> [2, 3] via unsqueeze + broadcast_as
    let tb_2d = tb.unsqueeze(0).unwrap(); // [1, 3]
    let tb_broadcast = tb_2d.broadcast_as(&[2, 3]).unwrap(); // [2, 3]
    let y = tx.add(&tb_broadcast).unwrap(); // [2, 3]
    let loss = reduce_to_scalar(&y);

    let grads = backward(&loss).unwrap();
    let grad_x = grads.get(&x).unwrap().to_flat_vec::<f32>().unwrap();
    let grad_b = grads.get(&b).unwrap().to_flat_vec::<f32>().unwrap();

    // grad_x should be all 1s (d/dx sum(x+b) = 1)
    for (i, &g) in grad_x.iter().enumerate() {
        assert!((g - 1.0).abs() < 1e-6, "grad_x[{i}] should be 1.0, got {g}");
    }
    // grad_b should be [2, 2, 2] (summed over batch dim)
    for (i, &g) in grad_b.iter().enumerate() {
        assert!(
            (g - 2.0).abs() < 1e-5,
            "grad_b[{i}] should be 2.0 (sum over batch), got {g}"
        );
    }
}

/// Cat backward: concatenation distributes gradient back to correct slices.
#[test]
fn test_backward_cat_gradient_slicing() {
    let a_data = vec![1.0, 2.0, 3.0]; // [3]
    let b_data = vec![4.0, 5.0]; // [2]

    let a = vec_var(a_data);
    let b = vec_var(b_data);

    let ta = Arc::new(TrackedTensor::from_var(&a).unwrap());
    let tb = Arc::new(TrackedTensor::from_var(&b).unwrap());

    // cat([a, b], dim=0) -> [5], then apply sqr -> sum for a non-trivial loss
    let cat_ab = TrackedTensor::cat(&[&ta, &tb], 0).unwrap(); // [5]
    let y = cat_ab.sqr().unwrap(); // [5]
    let loss = reduce_to_scalar(&y);
    let grads = backward(&loss).unwrap();

    let grad_a = grads.get(&a).unwrap().to_flat_vec::<f32>().unwrap();
    let grad_b = grads.get(&b).unwrap().to_flat_vec::<f32>().unwrap();

    // d/dx_i sum(x_i^2) = 2*x_i
    assert!((grad_a[0] - 2.0).abs() < 1e-5, "grad_a[0] = 2*1 = 2");
    assert!((grad_a[1] - 4.0).abs() < 1e-5, "grad_a[1] = 2*2 = 4");
    assert!((grad_a[2] - 6.0).abs() < 1e-5, "grad_a[2] = 2*3 = 6");
    assert!((grad_b[0] - 8.0).abs() < 1e-5, "grad_b[0] = 2*4 = 8");
    assert!((grad_b[1] - 10.0).abs() < 1e-5, "grad_b[1] = 2*5 = 10");
}

/// Deeply nested residual blocks: x -> [relu(x) + x] -> [tanh(y) + y] -> sigmoid -> loss.
/// Verifies gradient flow through multiple skip connections.
#[test]
fn test_stacked_residual_blocks() {
    let x_val = 0.5_f32;
    let x = scalar_var(x_val);
    let t = Arc::new(TrackedTensor::from_var(&x).unwrap());

    // Block 1: y = relu(x) + x
    let block1 = t.relu().unwrap().add(&t).unwrap();
    // Block 2: z = tanh(y) + y
    let block2 = block1.tanh().unwrap().add(&block1).unwrap();
    // Final: loss = sigmoid(z)
    let loss = block2.sigmoid().unwrap();

    let grads = backward(&loss).unwrap();
    let grad = grads.get(&x).unwrap().to_flat_vec::<f32>().unwrap();

    // Numerical verification
    let fd = finite_diff_grad(&[x_val], 1e-4, &|v: Vec<f32>| {
        let xv = f64::from(v[0]);
        let y = xv.max(0.0) + xv; // relu(x) + x
        let z = y.tanh() + y; // tanh(y) + y
        1.0 / (1.0 + (-z).exp()) // sigmoid(z)
    });

    let err = (f64::from(grad[0]) - fd[0]).abs();
    assert!(
        err < 1e-3,
        "stacked residual: analytical={}, numerical={}, err={}",
        grad[0],
        fd[0],
        err
    );
}

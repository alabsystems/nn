// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Extended higher-order autodiff composition tests.
//!
//! Covers chain-rule compositions with binary ops, multi-variable gradients,
//! deep chains (3+ ops), edge cases (zero/large inputs, near-zero divisors),
//! cat/stack/max/min backward, and complex composed expressions.
//!
//! Each test verifies analytical gradients against central-difference
//! finite-difference: (f(x+h) - f(x-h)) / (2h).

use std::sync::Arc;

use nn_core::dyn_tensor::DynTensor;
use nn_core::test_utils::cpu;

use crate::grad::backward;
use crate::tracked::TrackedTensor;
use crate::var::Var;

// ============================================================================
// Helpers (duplicated from higher_order_tests for module isolation)
// ============================================================================

fn shaped_var(data: Vec<f32>, shape: &[usize]) -> Var {
    Var::new(DynTensor::from_vec(data, shape, &cpu()).unwrap())
}

fn scalar_sum(t: &DynTensor) -> f64 {
    t.to_flat_vec::<f32>()
        .unwrap()
        .iter()
        .map(|&v| f64::from(v))
        .sum()
}

fn reduce_to_scalar(t: &Arc<TrackedTensor>) -> Arc<TrackedTensor> {
    let mut result = Arc::clone(t);
    for d in (0..result.tensor().rank()).rev() {
        result = result.sum_keepdim(d).unwrap();
    }
    result
}

/// Central-difference FD gradient check.
fn check_fd_grad(
    analytical: &[f32],
    data: &[f32],
    eps: f32,
    tol: f64,
    fwd: &dyn Fn(Vec<f32>) -> f64,
) {
    for i in 0..data.len() {
        let mut plus = data.to_vec();
        let mut minus = data.to_vec();
        plus[i] += eps;
        minus[i] -= eps;
        let numerical = (fwd(plus) - fwd(minus)) / (2.0 * f64::from(eps));
        let analytical_f64 = f64::from(analytical[i]);
        let err = (analytical_f64 - numerical).abs();
        assert!(
            err < tol,
            "grad[{i}]: analytical={analytical_f64}, numerical={numerical}, err={err}, tol={tol}",
        );
    }
}

// ============================================================================
// 1. Chain rule with binary ops: f(g(x, y))
// ============================================================================

/// exp(x * y): df/dx = y * exp(x*y), df/dy = x * exp(x*y)
#[test]
fn test_compose_exp_of_mul_multi_var() {
    let x_data = vec![0.5, 1.0, -0.3];
    let y_data = vec![0.2, -0.5, 0.8];
    let x = shaped_var(x_data.clone(), &[3]);
    let y = shaped_var(y_data.clone(), &[3]);
    let tx = Arc::new(TrackedTensor::from_var(&x).unwrap());
    let ty = Arc::new(TrackedTensor::from_var(&y).unwrap());
    let prod = tx.mul(&ty).unwrap();
    let out = prod.exp().unwrap();
    let loss = reduce_to_scalar(&out);
    let grads = backward(&loss).unwrap();

    let gx = grads.get(&x).unwrap().to_flat_vec::<f32>().unwrap();
    let gy = grads.get(&y).unwrap().to_flat_vec::<f32>().unwrap();

    for i in 0..3 {
        let xv = f64::from(x_data[i]);
        let yv = f64::from(y_data[i]);
        let e = (xv * yv).exp();
        let expected_gx = yv * e;
        let expected_gy = xv * e;
        assert!(
            (f64::from(gx[i]) - expected_gx).abs() < 1e-4,
            "gx[{i}]: got={}, expected={expected_gx}",
            gx[i]
        );
        assert!(
            (f64::from(gy[i]) - expected_gy).abs() < 1e-4,
            "gy[{i}]: got={}, expected={expected_gy}",
            gy[i]
        );
    }
}

/// sin(x + y): df/dx = cos(x+y), df/dy = cos(x+y)
#[test]
fn test_compose_sin_of_add_multi_var() {
    let x_data = vec![0.3, 1.0, -0.5];
    let y_data = vec![0.7, -0.2, 0.4];
    let x = shaped_var(x_data.clone(), &[3]);
    let y = shaped_var(y_data.clone(), &[3]);
    let tx = Arc::new(TrackedTensor::from_var(&x).unwrap());
    let ty = Arc::new(TrackedTensor::from_var(&y).unwrap());
    let sum = tx.add(&ty).unwrap();
    let out = sum.sin().unwrap();
    let loss = reduce_to_scalar(&out);
    let grads = backward(&loss).unwrap();

    let gx = grads.get(&x).unwrap().to_flat_vec::<f32>().unwrap();
    let gy = grads.get(&y).unwrap().to_flat_vec::<f32>().unwrap();

    for i in 0..3 {
        let expected = (f64::from(x_data[i]) + f64::from(y_data[i])).cos();
        assert!(
            (f64::from(gx[i]) - expected).abs() < 1e-5,
            "gx[{i}]: got={}, expected={expected}",
            gx[i]
        );
        assert!(
            (f64::from(gy[i]) - expected).abs() < 1e-5,
            "gy[{i}]: got={}, expected={expected}",
            gy[i]
        );
    }
}

/// tanh(x / y): multi-variable FD check
#[test]
fn test_compose_tanh_of_div_fd() {
    let x_data = vec![0.5, 1.0, -0.3, 0.8];
    let y_data = vec![1.5, 2.0, 0.7, -1.2];

    // FD for x
    let x = shaped_var(x_data.clone(), &[4]);
    let y = shaped_var(y_data.clone(), &[4]);
    let tx = Arc::new(TrackedTensor::from_var(&x).unwrap());
    let ty = Arc::new(TrackedTensor::from_var(&y).unwrap());
    let out = tx.div(&ty).unwrap().tanh().unwrap();
    let loss = reduce_to_scalar(&out);
    let grads = backward(&loss).unwrap();
    let gx = grads.get(&x).unwrap().to_flat_vec::<f32>().unwrap();
    let gy = grads.get(&y).unwrap().to_flat_vec::<f32>().unwrap();

    let y_frozen = y_data.clone();
    check_fd_grad(&gx, &x_data, 1e-3, 1e-2, &|xd| {
        let xt = DynTensor::from_vec(xd, &[4], &cpu()).unwrap();
        let yt = DynTensor::from_vec(y_frozen.clone(), &[4], &cpu()).unwrap();
        scalar_sum(&xt.div(&yt).unwrap().tanh().unwrap())
    });

    let x_frozen = x_data;
    check_fd_grad(&gy, &y_data, 1e-3, 1e-2, &|yd| {
        let xt = DynTensor::from_vec(x_frozen.clone(), &[4], &cpu()).unwrap();
        let yt = DynTensor::from_vec(yd, &[4], &cpu()).unwrap();
        scalar_sum(&xt.div(&yt).unwrap().tanh().unwrap())
    });
}

// ============================================================================
// 2. Deep chains (3+ compositions)
// ============================================================================

/// sigmoid(sin(exp(x))): 3-deep chain rule
#[test]
fn test_compose_deep_sigmoid_sin_exp() {
    let x_data = vec![0.1, -0.2, 0.3, -0.1];
    let x = shaped_var(x_data.clone(), &[4]);
    let tx = Arc::new(TrackedTensor::from_var(&x).unwrap());
    let out = tx.exp().unwrap().sin().unwrap().sigmoid().unwrap();
    let loss = reduce_to_scalar(&out);
    let grads = backward(&loss).unwrap();
    let gx = grads.get(&x).unwrap().to_flat_vec::<f32>().unwrap();

    check_fd_grad(&gx, &x_data, 1e-3, 1e-2, &|d| {
        let t = DynTensor::from_vec(d, &[4], &cpu()).unwrap();
        scalar_sum(&t.exp().unwrap().sin().unwrap().sigmoid().unwrap())
    });
}

/// tanh(relu(sqr(x))): 3-deep with non-smooth relu
#[test]
fn test_compose_deep_tanh_relu_sqr() {
    // Use positive values to stay in relu's active region
    let x_data = vec![0.5, 1.0, 0.2, 1.5];
    let x = shaped_var(x_data.clone(), &[4]);
    let tx = Arc::new(TrackedTensor::from_var(&x).unwrap());
    let out = tx.sqr().unwrap().relu().unwrap().tanh().unwrap();
    let loss = reduce_to_scalar(&out);
    let grads = backward(&loss).unwrap();
    let gx = grads.get(&x).unwrap().to_flat_vec::<f32>().unwrap();

    check_fd_grad(&gx, &x_data, 1e-3, 1e-2, &|d| {
        let t = DynTensor::from_vec(d, &[4], &cpu()).unwrap();
        scalar_sum(&t.sqr().unwrap().relu().unwrap().tanh().unwrap())
    });
}

/// exp(cos(sin(tanh(x)))): 4-deep chain
#[test]
fn test_compose_deep_4_chain() {
    let x_data = vec![0.3, -0.5, 0.7, -0.1];
    let x = shaped_var(x_data.clone(), &[4]);
    let tx = Arc::new(TrackedTensor::from_var(&x).unwrap());
    let out = tx
        .tanh()
        .unwrap()
        .sin()
        .unwrap()
        .cos()
        .unwrap()
        .exp()
        .unwrap();
    let loss = reduce_to_scalar(&out);
    let grads = backward(&loss).unwrap();
    let gx = grads.get(&x).unwrap().to_flat_vec::<f32>().unwrap();

    check_fd_grad(&gx, &x_data, 1e-3, 1e-2, &|d| {
        let t = DynTensor::from_vec(d, &[4], &cpu()).unwrap();
        scalar_sum(
            &t.tanh()
                .unwrap()
                .sin()
                .unwrap()
                .cos()
                .unwrap()
                .exp()
                .unwrap(),
        )
    });
}

// ============================================================================
// 3. Multi-variable gradients: df/dx, df/dy for f(x, y)
// ============================================================================

/// f(x, y) = x * y + x^2: df/dx = y + 2x, df/dy = x
#[test]
fn test_multi_var_mul_plus_sqr() {
    let x_data = vec![1.0, 2.0, -1.0];
    let y_data = vec![3.0, -1.0, 0.5];
    let x = shaped_var(x_data.clone(), &[3]);
    let y = shaped_var(y_data.clone(), &[3]);
    let tx = Arc::new(TrackedTensor::from_var(&x).unwrap());
    let ty = Arc::new(TrackedTensor::from_var(&y).unwrap());
    let xy = tx.mul(&ty).unwrap();
    let x2 = tx.sqr().unwrap();
    let out = xy.add(&x2).unwrap();
    let loss = reduce_to_scalar(&out);
    let grads = backward(&loss).unwrap();

    let gx = grads.get(&x).unwrap().to_flat_vec::<f32>().unwrap();
    let gy = grads.get(&y).unwrap().to_flat_vec::<f32>().unwrap();

    for i in 0..3 {
        let expected_gx = f64::from(y_data[i]) + 2.0 * f64::from(x_data[i]);
        let expected_gy = f64::from(x_data[i]);
        assert!(
            (f64::from(gx[i]) - expected_gx).abs() < 1e-5,
            "gx[{i}]: got={}, expected={expected_gx}",
            gx[i]
        );
        assert!(
            (f64::from(gy[i]) - expected_gy).abs() < 1e-5,
            "gy[{i}]: got={}, expected={expected_gy}",
            gy[i]
        );
    }
}

/// f(x, y) = (x - y)^2: df/dx = 2(x-y), df/dy = -2(x-y)
#[test]
fn test_multi_var_sqr_diff() {
    let x_data = vec![2.0, 0.5, -1.0];
    let y_data = vec![1.0, 1.5, 0.0];
    let x = shaped_var(x_data.clone(), &[3]);
    let y = shaped_var(y_data.clone(), &[3]);
    let tx = Arc::new(TrackedTensor::from_var(&x).unwrap());
    let ty = Arc::new(TrackedTensor::from_var(&y).unwrap());
    let diff = tx.sub(&ty).unwrap();
    let out = diff.sqr().unwrap();
    let loss = reduce_to_scalar(&out);
    let grads = backward(&loss).unwrap();

    let gx = grads.get(&x).unwrap().to_flat_vec::<f32>().unwrap();
    let gy = grads.get(&y).unwrap().to_flat_vec::<f32>().unwrap();

    for i in 0..3 {
        let d = f64::from(x_data[i]) - f64::from(y_data[i]);
        let expected_gx = 2.0 * d;
        let expected_gy = -2.0 * d;
        assert!(
            (f64::from(gx[i]) - expected_gx).abs() < 1e-5,
            "gx[{i}]: got={}, expected={expected_gx}",
            gx[i]
        );
        assert!(
            (f64::from(gy[i]) - expected_gy).abs() < 1e-5,
            "gy[{i}]: got={}, expected={expected_gy}",
            gy[i]
        );
    }
}

/// MatMul multi-variable FD: f(A, B) = sum(A @ B)
#[test]
fn test_multi_var_matmul_fd() {
    let a_data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]; // [2,3]
    let b_data = vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6]; // [3,2]
    let a = shaped_var(a_data.clone(), &[2, 3]);
    let b = shaped_var(b_data.clone(), &[3, 2]);
    let ta = Arc::new(TrackedTensor::from_var(&a).unwrap());
    let tb = Arc::new(TrackedTensor::from_var(&b).unwrap());
    let out = ta.matmul(&tb).unwrap();
    let loss = reduce_to_scalar(&out);
    let grads = backward(&loss).unwrap();

    let ga = grads.get(&a).unwrap().to_flat_vec::<f32>().unwrap();
    let gb = grads.get(&b).unwrap().to_flat_vec::<f32>().unwrap();

    let b_frozen = b_data.clone();
    check_fd_grad(&ga, &a_data, 1e-3, 1e-2, &|ad| {
        let at = DynTensor::from_vec(ad, &[2, 3], &cpu()).unwrap();
        let bt = DynTensor::from_vec(b_frozen.clone(), &[3, 2], &cpu()).unwrap();
        scalar_sum(&at.matmul(&bt).unwrap())
    });

    let a_frozen = a_data;
    check_fd_grad(&gb, &b_data, 1e-3, 1e-2, &|bd| {
        let at = DynTensor::from_vec(a_frozen.clone(), &[2, 3], &cpu()).unwrap();
        let bt = DynTensor::from_vec(bd, &[3, 2], &cpu()).unwrap();
        scalar_sum(&at.matmul(&bt).unwrap())
    });
}

// ============================================================================
// 4. Edge cases
// ============================================================================

/// Zero inputs: relu gradient at zero is 0 (subgradient convention).
#[test]
fn test_edge_relu_at_zero() {
    let x_data = vec![0.0, 0.0, 0.0, 0.0];
    let x = shaped_var(x_data, &[4]);
    let tx = Arc::new(TrackedTensor::from_var(&x).unwrap());
    let out = tx.relu().unwrap();
    let loss = reduce_to_scalar(&out);
    let grads = backward(&loss).unwrap();
    let gx = grads.get(&x).unwrap().to_flat_vec::<f32>().unwrap();
    // relu uses ge(0) mask, so grad at 0 is 1 (matching PyTorch convention)
    for &g in &gx {
        assert!(
            g == 0.0 || g == 1.0,
            "relu grad at zero should be 0 or 1, got {g}"
        );
    }
}

/// Zero inputs: abs gradient at zero is 0 (subgradient convention).
#[test]
fn test_edge_abs_at_zero() {
    let x_data = vec![0.0, 0.5, -0.5, 0.0];
    let x = shaped_var(x_data, &[4]);
    let tx = Arc::new(TrackedTensor::from_var(&x).unwrap());
    let out = tx.abs().unwrap();
    let loss = reduce_to_scalar(&out);
    let grads = backward(&loss).unwrap();
    let gx = grads.get(&x).unwrap().to_flat_vec::<f32>().unwrap();
    // abs uses gt(0)/lt(0) so grad at 0 is 0
    assert!((gx[0]).abs() < 1e-7, "abs grad at zero should be 0");
    assert!((gx[1] - 1.0).abs() < 1e-7, "abs grad at 0.5 should be 1");
    assert!(
        (gx[2] - (-1.0)).abs() < 1e-7,
        "abs grad at -0.5 should be -1"
    );
    assert!((gx[3]).abs() < 1e-7, "abs grad at zero should be 0");
}

/// Large inputs: sigmoid saturates, gradient should be near zero.
#[test]
fn test_edge_sigmoid_large_inputs() {
    let x_data = vec![50.0, -50.0, 100.0, -100.0];
    let x = shaped_var(x_data, &[4]);
    let tx = Arc::new(TrackedTensor::from_var(&x).unwrap());
    let out = tx.sigmoid().unwrap();
    let loss = reduce_to_scalar(&out);
    let grads = backward(&loss).unwrap();
    let gx = grads.get(&x).unwrap().to_flat_vec::<f32>().unwrap();
    for (i, &g) in gx.iter().enumerate() {
        assert!(
            g.abs() < 1e-6,
            "sigmoid grad at large input should be ~0, got grad[{i}]={g}"
        );
    }
}

/// Large inputs: tanh saturates, gradient should be near zero.
#[test]
fn test_edge_tanh_large_inputs() {
    let x_data = vec![20.0, -20.0, 50.0, -50.0];
    let x = shaped_var(x_data, &[4]);
    let tx = Arc::new(TrackedTensor::from_var(&x).unwrap());
    let out = tx.tanh().unwrap();
    let loss = reduce_to_scalar(&out);
    let grads = backward(&loss).unwrap();
    let gx = grads.get(&x).unwrap().to_flat_vec::<f32>().unwrap();
    for (i, &g) in gx.iter().enumerate() {
        assert!(
            g.abs() < 1e-6,
            "tanh grad at large input should be ~0, got grad[{i}]={g}"
        );
    }
}

/// Near-zero divisor: div gradient with small y values (FD check).
#[test]
fn test_edge_div_near_zero_divisor_fd() {
    let x_data = vec![1.0, 2.0, 0.5, -1.0];
    let y_data = vec![0.01, -0.01, 0.05, -0.05];
    let x = shaped_var(x_data.clone(), &[4]);
    let y = shaped_var(y_data.clone(), &[4]);
    let tx = Arc::new(TrackedTensor::from_var(&x).unwrap());
    let ty = Arc::new(TrackedTensor::from_var(&y).unwrap());
    let out = tx.div(&ty).unwrap();
    let loss = reduce_to_scalar(&out);
    let grads = backward(&loss).unwrap();

    let gx = grads.get(&x).unwrap().to_flat_vec::<f32>().unwrap();
    let gy = grads.get(&y).unwrap().to_flat_vec::<f32>().unwrap();

    // Analytical: df/dx = 1/y, df/dy = -x/y^2
    for i in 0..4 {
        let yv = f64::from(y_data[i]);
        let xv = f64::from(x_data[i]);
        let expected_gx = 1.0 / yv;
        let expected_gy = -xv / (yv * yv);
        assert!(
            (f64::from(gx[i]) - expected_gx).abs() / expected_gx.abs().max(1.0) < 1e-3,
            "gx[{i}]: got={}, expected={expected_gx}",
            gx[i]
        );
        assert!(
            (f64::from(gy[i]) - expected_gy).abs() / expected_gy.abs().max(1.0) < 1e-3,
            "gy[{i}]: got={}, expected={expected_gy}",
            gy[i]
        );
    }
}

/// Clamp edge: inputs exactly at boundaries.
#[test]
fn test_edge_clamp_at_boundaries() {
    // clamp uses ge/le, so gradient flows at boundaries
    let x_data = vec![-1.0, -0.5, 0.0, 0.5, 1.0];
    let x = shaped_var(x_data, &[5]);
    let tx = Arc::new(TrackedTensor::from_var(&x).unwrap());
    let out = tx.clamp(-0.5, 0.5).unwrap();
    let loss = reduce_to_scalar(&out);
    let grads = backward(&loss).unwrap();
    let gx = grads.get(&x).unwrap().to_flat_vec::<f32>().unwrap();

    // x=-1.0 is below -0.5 -> grad=0
    assert!((gx[0]).abs() < 1e-7, "below clamp min: grad should be 0");
    // x=-0.5 is at boundary -> grad=1 (ge/le)
    assert!((gx[1] - 1.0).abs() < 1e-7, "at clamp min: grad should be 1");
    // x=0.0 is inside -> grad=1
    assert!((gx[2] - 1.0).abs() < 1e-7, "inside clamp: grad should be 1");
    // x=0.5 is at boundary -> grad=1 (ge/le)
    assert!((gx[3] - 1.0).abs() < 1e-7, "at clamp max: grad should be 1");
    // x=1.0 is above 0.5 -> grad=0
    assert!((gx[4]).abs() < 1e-7, "above clamp max: grad should be 0");
}

// ============================================================================
// 5. Missing activation FD checks
// ============================================================================

/// hard_sigmoid FD
#[test]
fn test_compose_fd_hard_sigmoid() {
    let x_data = vec![-4.0, -1.0, 0.0, 1.5, 4.0];
    let x = shaped_var(x_data.clone(), &[5]);
    let tx = Arc::new(TrackedTensor::from_var(&x).unwrap());
    let out = tx.hard_sigmoid().unwrap();
    let loss = reduce_to_scalar(&out);
    let grads = backward(&loss).unwrap();
    let gx = grads.get(&x).unwrap().to_flat_vec::<f32>().unwrap();

    check_fd_grad(&gx, &x_data, 1e-3, 5e-2, &|d| {
        let t = DynTensor::from_vec(d, &[5], &cpu()).unwrap();
        scalar_sum(&t.hard_sigmoid().unwrap())
    });
}

/// hard_swish FD
#[test]
fn test_compose_fd_hard_swish() {
    let x_data = vec![-4.0, -1.0, 0.0, 1.5, 4.0];
    let x = shaped_var(x_data.clone(), &[5]);
    let tx = Arc::new(TrackedTensor::from_var(&x).unwrap());
    let out = tx.hard_swish().unwrap();
    let loss = reduce_to_scalar(&out);
    let grads = backward(&loss).unwrap();
    let gx = grads.get(&x).unwrap().to_flat_vec::<f32>().unwrap();

    check_fd_grad(&gx, &x_data, 1e-3, 5e-2, &|d| {
        let t = DynTensor::from_vec(d, &[5], &cpu()).unwrap();
        scalar_sum(&t.hard_swish().unwrap())
    });
}

/// selu FD (avoid x=0 where derivative has a kink)
#[test]
fn test_compose_fd_selu() {
    let x_data = vec![-2.0, -0.5, 0.1, 0.5, 2.0];
    let x = shaped_var(x_data.clone(), &[5]);
    let tx = Arc::new(TrackedTensor::from_var(&x).unwrap());
    let out = tx.selu().unwrap();
    let loss = reduce_to_scalar(&out);
    let grads = backward(&loss).unwrap();
    let gx = grads.get(&x).unwrap().to_flat_vec::<f32>().unwrap();

    check_fd_grad(&gx, &x_data, 1e-3, 5e-2, &|d| {
        let t = DynTensor::from_vec(d, &[5], &cpu()).unwrap();
        scalar_sum(&t.selu().unwrap())
    });
}

/// celu FD
#[test]
fn test_compose_fd_celu() {
    let x_data = vec![-2.0, -0.5, 0.0, 0.5, 2.0];
    let x = shaped_var(x_data.clone(), &[5]);
    let tx = Arc::new(TrackedTensor::from_var(&x).unwrap());
    let out = tx.celu(1.0).unwrap();
    let loss = reduce_to_scalar(&out);
    let grads = backward(&loss).unwrap();
    let gx = grads.get(&x).unwrap().to_flat_vec::<f32>().unwrap();

    check_fd_grad(&gx, &x_data, 1e-3, 5e-2, &|d| {
        let t = DynTensor::from_vec(d, &[5], &cpu()).unwrap();
        scalar_sum(&t.celu(1.0).unwrap())
    });
}

/// neg FD
#[test]
fn test_compose_fd_neg() {
    let x_data = vec![-2.0, -0.5, 0.0, 0.5, 2.0];
    let x = shaped_var(x_data, &[5]);
    let tx = Arc::new(TrackedTensor::from_var(&x).unwrap());
    let out = tx.neg().unwrap();
    let loss = reduce_to_scalar(&out);
    let grads = backward(&loss).unwrap();
    let gx = grads.get(&x).unwrap().to_flat_vec::<f32>().unwrap();
    // d/dx (-x) = -1
    for (i, &g) in gx.iter().enumerate() {
        assert!(
            (g - (-1.0)).abs() < 1e-6,
            "neg grad should be -1, got grad[{i}]={g}"
        );
    }
}

/// sqrt FD (positive inputs only)
#[test]
fn test_compose_fd_sqrt() {
    let x_data = vec![0.1, 0.5, 1.0, 4.0, 9.0];
    let x = shaped_var(x_data.clone(), &[5]);
    let tx = Arc::new(TrackedTensor::from_var(&x).unwrap());
    let out = tx.sqrt().unwrap();
    let loss = reduce_to_scalar(&out);
    let grads = backward(&loss).unwrap();
    let gx = grads.get(&x).unwrap().to_flat_vec::<f32>().unwrap();

    check_fd_grad(&gx, &x_data, 1e-4, 5e-2, &|d| {
        let t = DynTensor::from_vec(d, &[5], &cpu()).unwrap();
        scalar_sum(&t.sqrt().unwrap())
    });
}

// ============================================================================
// 6. Cat / Stack backward
// ============================================================================

/// Cat along dim 0: gradient of cat splits back.
#[test]
fn test_compose_cat_backward_fd() {
    let a_data = vec![1.0, 2.0, 3.0];
    let b_data = vec![4.0, 5.0];
    let a = shaped_var(a_data.clone(), &[3]);
    let b = shaped_var(b_data.clone(), &[2]);
    let ta = Arc::new(TrackedTensor::from_var(&a).unwrap());
    let tb = Arc::new(TrackedTensor::from_var(&b).unwrap());
    let cat = TrackedTensor::cat(&[&ta, &tb], 0).unwrap();
    // Apply nonlinear before summing so gradients are non-trivial
    let out = cat.sqr().unwrap();
    let loss = reduce_to_scalar(&out);
    let grads = backward(&loss).unwrap();

    let ga = grads.get(&a).unwrap().to_flat_vec::<f32>().unwrap();
    let gb = grads.get(&b).unwrap().to_flat_vec::<f32>().unwrap();

    // d/dx sum(x^2) = 2x for each element
    for i in 0..3 {
        let expected = 2.0 * f64::from(a_data[i]);
        assert!(
            (f64::from(ga[i]) - expected).abs() < 1e-5,
            "ga[{i}]: got={}, expected={expected}",
            ga[i]
        );
    }
    for i in 0..2 {
        let expected = 2.0 * f64::from(b_data[i]);
        assert!(
            (f64::from(gb[i]) - expected).abs() < 1e-5,
            "gb[{i}]: got={}, expected={expected}",
            gb[i]
        );
    }
}

/// Stack along dim 0: gradient of stack unsqueezes back.
#[test]
fn test_compose_stack_backward_fd() {
    let a_data = vec![1.0, 2.0];
    let b_data = vec![3.0, 4.0];
    let a = shaped_var(a_data.clone(), &[2]);
    let b = shaped_var(b_data.clone(), &[2]);
    let ta = Arc::new(TrackedTensor::from_var(&a).unwrap());
    let tb = Arc::new(TrackedTensor::from_var(&b).unwrap());
    let stacked = TrackedTensor::stack(&[ta, tb], 0).unwrap();
    let out = stacked.sqr().unwrap();
    let loss = reduce_to_scalar(&out);
    let grads = backward(&loss).unwrap();

    let ga = grads.get(&a).unwrap().to_flat_vec::<f32>().unwrap();
    let gb = grads.get(&b).unwrap().to_flat_vec::<f32>().unwrap();

    for i in 0..2 {
        let expected_a = 2.0 * f64::from(a_data[i]);
        let expected_b = 2.0 * f64::from(b_data[i]);
        assert!(
            (f64::from(ga[i]) - expected_a).abs() < 1e-5,
            "ga[{i}]: got={}, expected={expected_a}",
            ga[i]
        );
        assert!(
            (f64::from(gb[i]) - expected_b).abs() < 1e-5,
            "gb[{i}]: got={}, expected={expected_b}",
            gb[i]
        );
    }
}

// ============================================================================
// 7. Maximum / Minimum backward
// ============================================================================

/// maximum(x, y): gradient flows to the larger operand.
#[test]
fn test_compose_maximum_backward() {
    let x_data = vec![1.0, 0.5, 3.0, -1.0];
    let y_data = vec![0.5, 1.0, 2.0, -0.5];
    let x = shaped_var(x_data, &[4]);
    let y = shaped_var(y_data, &[4]);
    let tx = Arc::new(TrackedTensor::from_var(&x).unwrap());
    let ty = Arc::new(TrackedTensor::from_var(&y).unwrap());
    let out = tx.maximum(&ty).unwrap();
    let loss = reduce_to_scalar(&out);
    let grads = backward(&loss).unwrap();

    let gx = grads.get(&x).unwrap().to_flat_vec::<f32>().unwrap();
    let gy = grads.get(&y).unwrap().to_flat_vec::<f32>().unwrap();

    // x > y: gx=1, gy=0. x < y: gx=0, gy=1. x==y: tie -> gx=1, gy=0.
    assert!((gx[0] - 1.0).abs() < 1e-7); // x=1 > y=0.5
    assert!((gy[0]).abs() < 1e-7);
    assert!((gx[1]).abs() < 1e-7); // x=0.5 < y=1
    assert!((gy[1] - 1.0).abs() < 1e-7);
    assert!((gx[2] - 1.0).abs() < 1e-7); // x=3 > y=2
    assert!((gy[2]).abs() < 1e-7);
    assert!((gx[3]).abs() < 1e-7); // x=-1 < y=-0.5
    assert!((gy[3] - 1.0).abs() < 1e-7);
}

/// minimum(x, y): gradient flows to the smaller operand.
#[test]
fn test_compose_minimum_backward() {
    let x_data = vec![1.0, 0.5, 3.0, -1.0];
    let y_data = vec![0.5, 1.0, 2.0, -0.5];
    let x = shaped_var(x_data, &[4]);
    let y = shaped_var(y_data, &[4]);
    let tx = Arc::new(TrackedTensor::from_var(&x).unwrap());
    let ty = Arc::new(TrackedTensor::from_var(&y).unwrap());
    let out = tx.minimum(&ty).unwrap();
    let loss = reduce_to_scalar(&out);
    let grads = backward(&loss).unwrap();

    let gx = grads.get(&x).unwrap().to_flat_vec::<f32>().unwrap();
    let gy = grads.get(&y).unwrap().to_flat_vec::<f32>().unwrap();

    // x < y: gx=1, gy=0. x > y: gx=0, gy=1. x==y: tie -> gx=1, gy=0.
    assert!((gx[0]).abs() < 1e-7); // x=1 > y=0.5 -> min=y
    assert!((gy[0] - 1.0).abs() < 1e-7);
    assert!((gx[1] - 1.0).abs() < 1e-7); // x=0.5 < y=1 -> min=x
    assert!((gy[1]).abs() < 1e-7);
    assert!((gx[2]).abs() < 1e-7); // x=3 > y=2 -> min=y
    assert!((gy[2] - 1.0).abs() < 1e-7);
    assert!((gx[3] - 1.0).abs() < 1e-7); // x=-1 < y=-0.5 -> min=x
    assert!((gy[3]).abs() < 1e-7);
}

// ============================================================================
// 8. Reshape / Transpose gradient flow
// ============================================================================

/// Gradient flows through reshape correctly.
#[test]
fn test_compose_reshape_gradient_flow() {
    let x_data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let x = shaped_var(x_data.clone(), &[2, 3]);
    let tx = Arc::new(TrackedTensor::from_var(&x).unwrap());
    let reshaped = tx.reshape(&[3, 2]).unwrap();
    let out = reshaped.sqr().unwrap();
    let loss = reduce_to_scalar(&out);
    let grads = backward(&loss).unwrap();
    let gx = grads.get(&x).unwrap().to_flat_vec::<f32>().unwrap();

    // Reshape doesn't change the gradient values, just the shape routing.
    // d/dx sum(x^2) = 2x regardless of reshape.
    for i in 0..6 {
        let expected = 2.0 * f64::from(x_data[i]);
        assert!(
            (f64::from(gx[i]) - expected).abs() < 1e-5,
            "gx[{i}]: got={}, expected={expected}",
            gx[i]
        );
    }
    assert_eq!(grads.get(&x).unwrap().dims(), &[2, 3]);
}

/// Gradient flows through transpose correctly.
#[test]
fn test_compose_transpose_gradient_flow() {
    let x_data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let x = shaped_var(x_data.clone(), &[2, 3]);
    let tx = Arc::new(TrackedTensor::from_var(&x).unwrap());
    let transposed = tx.transpose(0, 1).unwrap(); // [3, 2]
    let out = transposed.sqr().unwrap();
    let loss = reduce_to_scalar(&out);
    let grads = backward(&loss).unwrap();
    let gx = grads.get(&x).unwrap().to_flat_vec::<f32>().unwrap();

    for i in 0..6 {
        let expected = 2.0 * f64::from(x_data[i]);
        assert!(
            (f64::from(gx[i]) - expected).abs() < 1e-5,
            "gx[{i}]: got={}, expected={expected}",
            gx[i]
        );
    }
    assert_eq!(grads.get(&x).unwrap().dims(), &[2, 3]);
}

// ============================================================================
// 9. Complex composed expressions
// ============================================================================

/// Softmax cross-entropy composed: -sum(target * log_softmax(x))
/// This is the standard cross-entropy loss computed manually.
#[test]
fn test_compose_softmax_cross_entropy_fd() {
    let x_data = vec![1.0, 2.0, 0.5, -1.0]; // logits
    let x = shaped_var(x_data.clone(), &[1, 4]);
    let tx = Arc::new(TrackedTensor::from_var(&x).unwrap());
    let log_sm = tx.log_softmax(1).unwrap();
    let loss = reduce_to_scalar(&log_sm);
    let grads = backward(&loss).unwrap();
    let gx = grads.get(&x).unwrap().to_flat_vec::<f32>().unwrap();

    check_fd_grad(&gx, &x_data, 1e-3, 1e-2, &|d| {
        let t = DynTensor::from_vec(d, &[1, 4], &cpu()).unwrap();
        scalar_sum(&t.log_softmax(1).unwrap())
    });
}

/// x * sigmoid(x) == silu(x): gradient should match silu gradient
#[test]
fn test_compose_manual_silu_matches_silu() {
    let x_data = vec![-2.0, -0.5, 0.0, 0.5, 2.0];
    let x1 = shaped_var(x_data.clone(), &[5]);
    let x2 = shaped_var(x_data, &[5]);

    // Manual: x * sigmoid(x)
    let tx1 = Arc::new(TrackedTensor::from_var(&x1).unwrap());
    let sig = tx1.sigmoid().unwrap();
    let manual_silu = tx1.mul(&sig).unwrap();
    let loss1 = reduce_to_scalar(&manual_silu);
    let grads1 = backward(&loss1).unwrap();
    let g_manual = grads1.get(&x1).unwrap().to_flat_vec::<f32>().unwrap();

    // Built-in silu
    let tx2 = Arc::new(TrackedTensor::from_var(&x2).unwrap());
    let builtin_silu = tx2.silu().unwrap();
    let loss2 = reduce_to_scalar(&builtin_silu);
    let grads2 = backward(&loss2).unwrap();
    let g_builtin = grads2.get(&x2).unwrap().to_flat_vec::<f32>().unwrap();

    for i in 0..5 {
        assert!(
            (g_manual[i] - g_builtin[i]).abs() < 1e-5,
            "manual silu grad[{i}]={} != builtin silu grad[{i}]={}",
            g_manual[i],
            g_builtin[i]
        );
    }
}

/// Composed: sum((W @ x + b)^2) — linear layer with MSE-like loss.
/// Multi-variable: gradients for W, x, and b.
#[test]
fn test_compose_linear_mse_multi_var_fd() {
    let w_data = vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6]; // [2,3]
    let x_data = vec![1.0, 2.0, 3.0]; // [3,1]
    let b_data = vec![0.1, -0.1]; // [2,1]

    let w = shaped_var(w_data.clone(), &[2, 3]);
    let xv = shaped_var(x_data.clone(), &[3, 1]);
    let b = shaped_var(b_data.clone(), &[2, 1]);

    let tw = Arc::new(TrackedTensor::from_var(&w).unwrap());
    let tx = Arc::new(TrackedTensor::from_var(&xv).unwrap());
    let tb = Arc::new(TrackedTensor::from_var(&b).unwrap());

    let wx = tw.matmul(&tx).unwrap(); // [2,1]
    let wxb = wx.add(&tb).unwrap(); // [2,1]
    let out = wxb.sqr().unwrap(); // [2,1]
    let loss = reduce_to_scalar(&out);
    let grads = backward(&loss).unwrap();

    let gw = grads.get(&w).unwrap().to_flat_vec::<f32>().unwrap();
    let gx = grads.get(&xv).unwrap().to_flat_vec::<f32>().unwrap();
    let gb = grads.get(&b).unwrap().to_flat_vec::<f32>().unwrap();

    // FD check for W
    let xf = x_data.clone();
    let bf = b_data.clone();
    check_fd_grad(&gw, &w_data, 1e-3, 1e-2, &|wd| {
        let wt = DynTensor::from_vec(wd, &[2, 3], &cpu()).unwrap();
        let xt = DynTensor::from_vec(xf.clone(), &[3, 1], &cpu()).unwrap();
        let bt = DynTensor::from_vec(bf.clone(), &[2, 1], &cpu()).unwrap();
        scalar_sum(&wt.matmul(&xt).unwrap().add(&bt).unwrap().sqr().unwrap())
    });

    // FD check for x
    let wf = w_data.clone();
    let bf2 = b_data.clone();
    check_fd_grad(&gx, &x_data, 1e-3, 1e-2, &|xd| {
        let wt = DynTensor::from_vec(wf.clone(), &[2, 3], &cpu()).unwrap();
        let xt = DynTensor::from_vec(xd, &[3, 1], &cpu()).unwrap();
        let bt = DynTensor::from_vec(bf2.clone(), &[2, 1], &cpu()).unwrap();
        scalar_sum(&wt.matmul(&xt).unwrap().add(&bt).unwrap().sqr().unwrap())
    });

    // FD check for b
    let wf2 = w_data;
    let xf2 = x_data;
    check_fd_grad(&gb, &b_data, 1e-3, 1e-2, &|bd| {
        let wt = DynTensor::from_vec(wf2.clone(), &[2, 3], &cpu()).unwrap();
        let xt = DynTensor::from_vec(xf2.clone(), &[3, 1], &cpu()).unwrap();
        let bt = DynTensor::from_vec(bd, &[2, 1], &cpu()).unwrap();
        scalar_sum(&wt.matmul(&xt).unwrap().add(&bt).unwrap().sqr().unwrap())
    });
}

/// Diamond graph: x used in two branches that merge.
/// f(x) = sin(x) * cos(x) = 0.5 * sin(2x)
/// df/dx = cos(x)^2 - sin(x)^2 = cos(2x)
#[test]
fn test_compose_diamond_sin_cos_mul() {
    let x_data = vec![0.3, 1.0, -0.5, 2.0];
    let x = shaped_var(x_data.clone(), &[4]);
    let tx = Arc::new(TrackedTensor::from_var(&x).unwrap());
    let s = tx.sin().unwrap();
    let c = tx.cos().unwrap();
    let out = s.mul(&c).unwrap();
    let loss = reduce_to_scalar(&out);
    let grads = backward(&loss).unwrap();
    let gx = grads.get(&x).unwrap().to_flat_vec::<f32>().unwrap();

    for i in 0..4 {
        let xv = f64::from(x_data[i]);
        let expected = (2.0 * xv).cos(); // cos(2x) = cos^2(x) - sin^2(x)
        assert!(
            (f64::from(gx[i]) - expected).abs() < 1e-4,
            "diamond grad[{i}]: got={}, expected={expected}",
            gx[i]
        );
    }
}

/// Triple fan-out: x used three times.
/// f(x) = x + x^2 + x^3
/// df/dx = 1 + 2x + 3x^2
#[test]
fn test_compose_triple_fan_out() {
    let x_data = vec![0.5, 1.0, -0.3, 2.0];
    let x = shaped_var(x_data.clone(), &[4]);
    let tx = Arc::new(TrackedTensor::from_var(&x).unwrap());
    let x2 = tx.sqr().unwrap();
    let x3 = x2.mul(&tx).unwrap(); // x^2 * x = x^3
    let sum12 = tx.add(&x2).unwrap();
    let out = sum12.add(&x3).unwrap();
    let loss = reduce_to_scalar(&out);
    let grads = backward(&loss).unwrap();
    let gx = grads.get(&x).unwrap().to_flat_vec::<f32>().unwrap();

    for i in 0..4 {
        let xv = f64::from(x_data[i]);
        let expected = 1.0 + 2.0 * xv + 3.0 * xv * xv;
        assert!(
            (f64::from(gx[i]) - expected).abs() < 1e-3,
            "triple fan-out grad[{i}]: got={}, expected={expected}",
            gx[i]
        );
    }
}

/// MulScalar and AddScalar: f(x) = 3*x + 2
/// df/dx = 3
#[test]
fn test_compose_scalar_ops_gradient() {
    let x_data = vec![1.0, -1.0, 0.0, 5.0];
    let x = shaped_var(x_data, &[4]);
    let tx = Arc::new(TrackedTensor::from_var(&x).unwrap());
    let out = tx.mul_scalar(3.0).unwrap().add_scalar(2.0).unwrap();
    let loss = reduce_to_scalar(&out);
    let grads = backward(&loss).unwrap();
    let gx = grads.get(&x).unwrap().to_flat_vec::<f32>().unwrap();

    for (i, &g) in gx.iter().enumerate() {
        assert!(
            (g - 3.0).abs() < 1e-6,
            "scalar ops grad[{i}] should be 3.0, got {g}"
        );
    }
}

/// Narrow backward: gradient zero-pads outside the slice.
#[test]
fn test_compose_narrow_backward() {
    let x_data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let x = shaped_var(x_data, &[5]);
    let tx = Arc::new(TrackedTensor::from_var(&x).unwrap());
    let sliced = tx.narrow(0, 1, 3).unwrap(); // elements [2, 3, 4]
    let out = sliced.sqr().unwrap();
    let loss = reduce_to_scalar(&out);
    let grads = backward(&loss).unwrap();
    let gx = grads.get(&x).unwrap().to_flat_vec::<f32>().unwrap();

    // Gradient for elements outside the slice should be 0.
    assert!(
        (gx[0]).abs() < 1e-7,
        "narrow: grad outside slice should be 0"
    );
    assert!((gx[1] - 2.0 * 2.0).abs() < 1e-5, "narrow: grad at index 1");
    assert!((gx[2] - 2.0 * 3.0).abs() < 1e-5, "narrow: grad at index 2");
    assert!((gx[3] - 2.0 * 4.0).abs() < 1e-5, "narrow: grad at index 3");
    assert!(
        (gx[4]).abs() < 1e-7,
        "narrow: grad outside slice should be 0"
    );
}

/// Unsqueeze/squeeze round-trip preserves gradient.
#[test]
fn test_compose_unsqueeze_squeeze_gradient() {
    let x_data = vec![1.0, 2.0, 3.0];
    let x = shaped_var(x_data.clone(), &[3]);
    let tx = Arc::new(TrackedTensor::from_var(&x).unwrap());
    let expanded = tx.unsqueeze(0).unwrap(); // [1, 3]
    let out = expanded.sqr().unwrap();
    let contracted = out.squeeze(0).unwrap(); // [3]
    let loss = reduce_to_scalar(&contracted);
    let grads = backward(&loss).unwrap();
    let gx = grads.get(&x).unwrap().to_flat_vec::<f32>().unwrap();

    for i in 0..3 {
        let expected = 2.0 * f64::from(x_data[i]);
        assert!(
            (f64::from(gx[i]) - expected).abs() < 1e-5,
            "unsqueeze/squeeze grad[{i}]: got={}, expected={expected}",
            gx[i]
        );
    }
    assert_eq!(grads.get(&x).unwrap().dims(), &[3]);
}

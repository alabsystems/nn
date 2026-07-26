// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Higher-order autodiff composition and numerical gradient tests.
//!
//! Tests gradient correctness for composed operations, reductions, broadcasts,
//! matmul, normalization, and loss functions. Each test verifies:
//! - Forward pass produces correct values
//! - Backward pass produces gradients with correct shapes
//! - Analytical gradients match finite-difference numerical gradients
//!
//! Finite-difference method: central difference (f(x+h) - f(x-h)) / (2h).

use std::sync::Arc;

use nn_core::dyn_tensor::DynTensor;
use nn_core::test_utils::cpu;

use crate::grad::backward;
use crate::tracked::TrackedTensor;
use crate::var::Var;

// ============================================================================
// Helpers
// ============================================================================

/// Create a Var from data with a given shape.
fn shaped_var(data: Vec<f32>, shape: &[usize]) -> Var {
    Var::new(DynTensor::from_vec(data, shape, &cpu()).unwrap())
}

/// Scalar loss from a DynTensor: sum of all elements.
fn scalar_sum(t: &DynTensor) -> f64 {
    t.to_flat_vec::<f32>()
        .unwrap()
        .iter()
        .map(|&v| f64::from(v))
        .sum()
}

/// Scalar loss from a DynTensor: sum of squares.
fn scalar_sum_sqr(t: &DynTensor) -> f64 {
    t.to_flat_vec::<f32>()
        .unwrap()
        .iter()
        .map(|&v| f64::from(v) * f64::from(v))
        .sum()
}

/// Reduce a tracked tensor to a scalar loss via sum over all dims.
fn reduce_to_scalar(t: &Arc<TrackedTensor>) -> Arc<TrackedTensor> {
    let mut result = Arc::clone(t);
    for d in (0..result.tensor().rank()).rev() {
        result = result.sum_keepdim(d).unwrap();
    }
    result
}

/// Central-difference finite-difference gradient check.
///
/// Perturbs each element of `data` by +/-eps, computes the scalar loss,
/// and compares the numerical gradient (f(x+h)-f(x-h))/(2h) against the
/// analytical gradient from backward(). Uses f64 arithmetic for comparison.
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
// 1. Composition gradients
// ============================================================================

/// sin(x^2): d/dx = 2x * cos(x^2)
#[test]
fn test_higher_order_sin_of_sqr() {
    let x_data = vec![0.5, 1.0, -0.3, 2.0];
    let x = shaped_var(x_data.clone(), &[4]);
    let t = Arc::new(TrackedTensor::from_var(&x).unwrap());
    let sq = t.sqr().unwrap();
    let y = sq.sin().unwrap();
    let loss = reduce_to_scalar(&y);
    let grads = backward(&loss).unwrap();

    let grad = grads.get(&x).unwrap();
    assert_eq!(grad.dims(), &[4], "gradient shape must match input");

    let analytical = grad.to_flat_vec::<f32>().unwrap();
    check_fd_grad(&analytical, &x_data, 1e-3, 1e-2, &|d| {
        let t = DynTensor::from_vec(d, &[4], &cpu()).unwrap();
        scalar_sum(&t.sqr().unwrap().sin().unwrap())
    });
}

/// exp(log(x)) = x: gradient should be 1.0 for positive inputs.
#[test]
fn test_higher_order_exp_of_log() {
    let x_data = vec![0.5, 1.0, 2.0, 3.0];
    let x = shaped_var(x_data, &[4]);
    let t = Arc::new(TrackedTensor::from_var(&x).unwrap());
    let lg = t.log().unwrap();
    let y = lg.exp().unwrap();
    let loss = reduce_to_scalar(&y);
    let grads = backward(&loss).unwrap();

    let grad = grads.get(&x).unwrap();
    let grad_vals = grad.to_flat_vec::<f32>().unwrap();
    // d/dx exp(log(x)) = 1.0
    for (i, &g) in grad_vals.iter().enumerate() {
        assert!((g - 1.0).abs() < 1e-4, "grad[{i}] = {g}, expected ~1.0");
    }
}

/// tanh(sigmoid(x)): composed activations, FD verified.
#[test]
fn test_higher_order_tanh_of_sigmoid() {
    let x_data = vec![-1.0, 0.0, 0.5, 2.0];
    let x = shaped_var(x_data.clone(), &[4]);
    let t = Arc::new(TrackedTensor::from_var(&x).unwrap());
    let sig = t.sigmoid().unwrap();
    let y = sig.tanh().unwrap();
    let loss = reduce_to_scalar(&y);
    let grads = backward(&loss).unwrap();

    let grad = grads.get(&x).unwrap();
    assert_eq!(grad.dims(), &[4]);

    let analytical = grad.to_flat_vec::<f32>().unwrap();
    check_fd_grad(&analytical, &x_data, 1e-3, 1e-2, &|d| {
        let t = DynTensor::from_vec(d, &[4], &cpu()).unwrap();
        scalar_sum(&t.sigmoid().unwrap().tanh().unwrap())
    });
}

/// sigmoid(sin(x)): chain rule through sin then sigmoid.
#[test]
fn test_higher_order_sigmoid_of_sin() {
    let x_data = vec![0.3, 1.0, -0.7, 1.5];
    let x = shaped_var(x_data.clone(), &[4]);
    let t = Arc::new(TrackedTensor::from_var(&x).unwrap());
    let s = t.sin().unwrap();
    let y = s.sigmoid().unwrap();
    let loss = reduce_to_scalar(&y);
    let grads = backward(&loss).unwrap();

    let grad = grads.get(&x).unwrap();
    assert_eq!(grad.dims(), &[4]);

    let analytical = grad.to_flat_vec::<f32>().unwrap();
    check_fd_grad(&analytical, &x_data, 1e-3, 1e-2, &|d| {
        let t = DynTensor::from_vec(d, &[4], &cpu()).unwrap();
        scalar_sum(&t.sin().unwrap().sigmoid().unwrap())
    });
}

/// neg(cos(sqr(x))): three-deep composition.
#[test]
fn test_higher_order_neg_cos_sqr() {
    let x_data = vec![0.3, -0.8, 1.2, 0.7];
    let x = shaped_var(x_data.clone(), &[4]);
    let t = Arc::new(TrackedTensor::from_var(&x).unwrap());
    let y = t.sqr().unwrap().cos().unwrap().neg().unwrap();
    let loss = reduce_to_scalar(&y);
    let grads = backward(&loss).unwrap();

    let grad = grads.get(&x).unwrap();
    assert_eq!(grad.dims(), &[4]);

    let analytical = grad.to_flat_vec::<f32>().unwrap();
    check_fd_grad(&analytical, &x_data, 1e-3, 1e-2, &|d| {
        let t = DynTensor::from_vec(d, &[4], &cpu()).unwrap();
        scalar_sum(&t.sqr().unwrap().cos().unwrap().neg().unwrap())
    });
}

/// sqr(tanh(x)): activation then elementwise squaring.
#[test]
fn test_higher_order_sqr_of_tanh() {
    let x_data = vec![-0.5, 0.0, 0.5, 1.0];
    let x = shaped_var(x_data.clone(), &[4]);
    let t = Arc::new(TrackedTensor::from_var(&x).unwrap());
    let th = t.tanh().unwrap();
    let y = th.sqr().unwrap();
    let loss = reduce_to_scalar(&y);
    let grads = backward(&loss).unwrap();

    let grad = grads.get(&x).unwrap();
    assert_eq!(grad.dims(), &[4]);

    let analytical = grad.to_flat_vec::<f32>().unwrap();
    check_fd_grad(&analytical, &x_data, 1e-3, 1e-2, &|d| {
        let t = DynTensor::from_vec(d, &[4], &cpu()).unwrap();
        scalar_sum(&t.tanh().unwrap().sqr().unwrap())
    });
}

// ============================================================================
// 2. Reduction gradients
// ============================================================================

/// Sum reduction: gradient is all ones.
#[test]
fn test_higher_order_sum_keepdim_gradient() {
    let x_data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let x = shaped_var(x_data, &[2, 3]);
    let t = Arc::new(TrackedTensor::from_var(&x).unwrap());
    let y = t.sum_keepdim(1).unwrap(); // [2, 1]
    let loss = reduce_to_scalar(&y);
    let grads = backward(&loss).unwrap();

    let grad = grads.get(&x).unwrap();
    assert_eq!(grad.dims(), &[2, 3], "grad shape must match input [2,3]");
    let vals = grad.to_flat_vec::<f32>().unwrap();
    for &v in &vals {
        assert!((v - 1.0).abs() < 1e-6, "sum grad should be 1.0, got {v}");
    }
}

/// Mean reduction: gradient is 1/N for the reduced dimension.
#[test]
fn test_higher_order_mean_keepdim_gradient() {
    let x_data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let x = shaped_var(x_data, &[2, 3]);
    let t = Arc::new(TrackedTensor::from_var(&x).unwrap());
    let y = t.mean_keepdim(1).unwrap(); // [2, 1]
    let loss = reduce_to_scalar(&y);
    let grads = backward(&loss).unwrap();

    let grad = grads.get(&x).unwrap();
    assert_eq!(grad.dims(), &[2, 3]);
    let vals = grad.to_flat_vec::<f32>().unwrap();
    let expected = 1.0 / 3.0;
    for &v in &vals {
        assert!(
            (v - expected).abs() < 1e-5,
            "mean grad should be {expected}, got {v}"
        );
    }
}

/// Sum then sqr: tests reduction + nonlinear composition FD.
#[test]
fn test_higher_order_sum_then_sqr_fd() {
    let x_data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let x = shaped_var(x_data.clone(), &[2, 3]);
    let t = Arc::new(TrackedTensor::from_var(&x).unwrap());
    let s = t.sum_keepdim(1).unwrap(); // [2, 1]
    let y = s.sqr().unwrap();
    let loss = reduce_to_scalar(&y);
    let grads = backward(&loss).unwrap();

    let grad = grads.get(&x).unwrap();
    assert_eq!(grad.dims(), &[2, 3]);

    let analytical = grad.to_flat_vec::<f32>().unwrap();
    // Larger tolerance: gradients are ~30, f32 FD accumulates rounding error
    check_fd_grad(&analytical, &x_data, 1e-3, 5e-2, &|d| {
        let t = DynTensor::from_vec(d, &[2, 3], &cpu()).unwrap();
        scalar_sum(&t.sum_keepdim(1).unwrap().sqr().unwrap())
    });
}

/// Mean over dim 0 of a 2D tensor.
#[test]
fn test_higher_order_mean_dim0_fd() {
    let x_data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let x = shaped_var(x_data.clone(), &[3, 2]);
    let t = Arc::new(TrackedTensor::from_var(&x).unwrap());
    let y = t.mean_keepdim(0).unwrap(); // [1, 2]
    let loss = y.sqr().unwrap();
    let scalar = reduce_to_scalar(&loss);
    let grads = backward(&scalar).unwrap();

    let grad = grads.get(&x).unwrap();
    assert_eq!(grad.dims(), &[3, 2]);

    let analytical = grad.to_flat_vec::<f32>().unwrap();
    check_fd_grad(&analytical, &x_data, 1e-3, 1e-2, &|d| {
        let t = DynTensor::from_vec(d, &[3, 2], &cpu()).unwrap();
        scalar_sum_sqr(&t.mean_keepdim(0).unwrap())
    });
}

// ============================================================================
// 3. Broadcast gradients
// ============================================================================

/// Scalar + vector broadcast: scalar grad should be sum of upstream.
#[test]
fn test_higher_order_broadcast_scalar_plus_vector() {
    let s_data = vec![2.0];
    let v_data = vec![1.0, 2.0, 3.0];
    let s_var = shaped_var(s_data, &[1]);
    let v_var = shaped_var(v_data, &[3]);

    let ts = Arc::new(TrackedTensor::from_var(&s_var).unwrap());
    let tv = Arc::new(TrackedTensor::from_var(&v_var).unwrap());
    let y = ts.add(&tv).unwrap(); // broadcast [1] + [3] = [3]
    let loss = reduce_to_scalar(&y);
    let grads = backward(&loss).unwrap();

    let gs = grads.get(&s_var).unwrap();
    let gv = grads.get(&v_var).unwrap();
    assert_eq!(gs.dims(), &[1], "scalar grad shape");
    assert_eq!(gv.dims(), &[3], "vector grad shape");
    // d(sum(s+v))/ds = 3 (add grad=1 for each of 3 elements, reduced to [1])
    let gs_val = gs.to_flat_vec::<f32>().unwrap();
    assert!((gs_val[0] - 3.0).abs() < 1e-6);
}

/// Scalar * vector broadcast: d(sum(s*v))/ds = sum(v).
#[test]
fn test_higher_order_broadcast_scalar_mul_vector() {
    let s_data = vec![3.0];
    let v_data = vec![1.0, 2.0, 4.0];
    let s_var = shaped_var(s_data, &[1]);
    let v_var = shaped_var(v_data, &[3]);

    let ts = Arc::new(TrackedTensor::from_var(&s_var).unwrap());
    let tv = Arc::new(TrackedTensor::from_var(&v_var).unwrap());
    let y = ts.mul(&tv).unwrap();
    let loss = reduce_to_scalar(&y);
    let grads = backward(&loss).unwrap();

    let gs = grads.get(&s_var).unwrap().to_flat_vec::<f32>().unwrap();
    let gv = grads.get(&v_var).unwrap().to_flat_vec::<f32>().unwrap();
    // d(sum(s*v))/ds = sum(v) = 7.0
    assert!((gs[0] - 7.0).abs() < 1e-5, "scalar grad = sum(v)");
    // d(sum(s*v))/dv_i = s = 3.0
    for &g in &gv {
        assert!((g - 3.0).abs() < 1e-5, "vector grad = s");
    }
}

/// Vector + matrix broadcast: [1, 3] + [2, 3], grad for vector reduces over dim 0.
#[test]
fn test_higher_order_broadcast_vector_plus_matrix_fd() {
    let v_data = vec![1.0, 2.0, 3.0];
    let m_data = vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6];
    let v_var = shaped_var(v_data.clone(), &[1, 3]);
    let m_var = shaped_var(m_data.clone(), &[2, 3]);

    let tv = Arc::new(TrackedTensor::from_var(&v_var).unwrap());
    let tm = Arc::new(TrackedTensor::from_var(&m_var).unwrap());
    let y = tv.add(&tm).unwrap().sqr().unwrap();
    let loss = reduce_to_scalar(&y);
    let grads = backward(&loss).unwrap();

    let gv = grads.get(&v_var).unwrap();
    let gm = grads.get(&m_var).unwrap();
    assert_eq!(gv.dims(), &[1, 3], "vector grad shape reduced");
    assert_eq!(gm.dims(), &[2, 3], "matrix grad shape preserved");

    let analytical_v = gv.to_flat_vec::<f32>().unwrap();
    check_fd_grad(&analytical_v, &v_data, 1e-3, 1e-2, &|d| {
        let v = DynTensor::from_vec(d, &[1, 3], &cpu()).unwrap();
        let m = DynTensor::from_vec(m_data.clone(), &[2, 3], &cpu()).unwrap();
        scalar_sum(&v.add(&m).unwrap().sqr().unwrap())
    });
}

/// Broadcast mul with [2, 1] * [2, 3]: column vector broadcast.
#[test]
fn test_higher_order_broadcast_column_mul_matrix() {
    let col_data = vec![2.0, 3.0];
    let m_data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let col_var = shaped_var(col_data.clone(), &[2, 1]);
    let m_var = shaped_var(m_data.clone(), &[2, 3]);

    let tc = Arc::new(TrackedTensor::from_var(&col_var).unwrap());
    let tm = Arc::new(TrackedTensor::from_var(&m_var).unwrap());
    let y = tc.mul(&tm).unwrap();
    let loss = reduce_to_scalar(&y);
    let grads = backward(&loss).unwrap();

    let gc = grads.get(&col_var).unwrap();
    let gm = grads.get(&m_var).unwrap();
    assert_eq!(gc.dims(), &[2, 1], "column grad shape");
    assert_eq!(gm.dims(), &[2, 3], "matrix grad shape");

    // FD check for column
    let analytical_c = gc.to_flat_vec::<f32>().unwrap();
    check_fd_grad(&analytical_c, &col_data, 1e-3, 1e-2, &|d| {
        let c = DynTensor::from_vec(d, &[2, 1], &cpu()).unwrap();
        let m = DynTensor::from_vec(m_data.clone(), &[2, 3], &cpu()).unwrap();
        scalar_sum(&c.mul(&m).unwrap())
    });
}

// ============================================================================
// 4. MatMul gradients
// ============================================================================

/// 2D matmul: [2, 3] @ [3, 4] gradient shapes and FD for both operands.
#[test]
fn test_higher_order_matmul_2d_fd() {
    let a_data = vec![0.5, -0.3, 1.2, 0.8, -0.6, 0.4];
    let b_data = vec![
        1.0, -0.5, 0.3, 0.2, 0.7, 0.1, -0.2, 0.8, -0.4, 1.1, 0.6, -0.1,
    ];
    let a_var = shaped_var(a_data.clone(), &[2, 3]);
    let b_var = shaped_var(b_data.clone(), &[3, 4]);

    let ta = Arc::new(TrackedTensor::from_var(&a_var).unwrap());
    let tb = Arc::new(TrackedTensor::from_var(&b_var).unwrap());
    let y = ta.matmul(&tb).unwrap();
    assert_eq!(y.dims(), &[2, 4]);
    let loss = y.sqr().unwrap();
    let scalar = reduce_to_scalar(&loss);
    let grads = backward(&scalar).unwrap();

    let ga = grads.get(&a_var).unwrap();
    let gb = grads.get(&b_var).unwrap();
    assert_eq!(ga.dims(), &[2, 3], "grad_a shape");
    assert_eq!(gb.dims(), &[3, 4], "grad_b shape");

    let analytical_a = ga.to_flat_vec::<f32>().unwrap();
    check_fd_grad(&analytical_a, &a_data, 1e-3, 1e-2, &|d| {
        let a = DynTensor::from_vec(d, &[2, 3], &cpu()).unwrap();
        let b = DynTensor::from_vec(b_data.clone(), &[3, 4], &cpu()).unwrap();
        scalar_sum_sqr(&a.matmul(&b).unwrap())
    });

    let analytical_b = gb.to_flat_vec::<f32>().unwrap();
    check_fd_grad(&analytical_b, &b_data, 1e-3, 1e-2, &|d| {
        let a = DynTensor::from_vec(a_data.clone(), &[2, 3], &cpu()).unwrap();
        let b = DynTensor::from_vec(d, &[3, 4], &cpu()).unwrap();
        scalar_sum_sqr(&a.matmul(&b).unwrap())
    });
}

/// Square matrix matmul: [3, 3] @ [3, 3].
#[test]
fn test_higher_order_matmul_square() {
    let a_data = vec![1.0, 2.0, 0.5, -1.0, 0.3, 1.5, 0.7, -0.2, 0.8];
    let b_data = vec![0.3, -0.5, 1.0, 0.6, 0.1, -0.3, -0.2, 0.8, 0.4];
    let a_var = shaped_var(a_data.clone(), &[3, 3]);
    let b_var = shaped_var(b_data.clone(), &[3, 3]);

    let ta = Arc::new(TrackedTensor::from_var(&a_var).unwrap());
    let tb = Arc::new(TrackedTensor::from_var(&b_var).unwrap());
    let y = ta.matmul(&tb).unwrap();
    let loss = y.sqr().unwrap();
    let scalar = reduce_to_scalar(&loss);
    let grads = backward(&scalar).unwrap();

    let ga = grads.get(&a_var).unwrap();
    assert_eq!(ga.dims(), &[3, 3]);

    let analytical = ga.to_flat_vec::<f32>().unwrap();
    check_fd_grad(&analytical, &a_data, 1e-3, 1e-2, &|d| {
        let a = DynTensor::from_vec(d, &[3, 3], &cpu()).unwrap();
        let b = DynTensor::from_vec(b_data.clone(), &[3, 3], &cpu()).unwrap();
        scalar_sum_sqr(&a.matmul(&b).unwrap())
    });
}

/// Batched 3D matmul: [2, 2, 3] @ [2, 3, 2].
#[test]
fn test_higher_order_matmul_batched_3d() {
    let a_data: Vec<f32> = (0..12).map(|i| (i as f32) * 0.1 - 0.5).collect();
    let b_data: Vec<f32> = (0..12).map(|i| (i as f32) * 0.15 - 0.3).collect();
    let a_var = shaped_var(a_data.clone(), &[2, 2, 3]);
    let b_var = shaped_var(b_data.clone(), &[2, 3, 2]);

    let ta = Arc::new(TrackedTensor::from_var(&a_var).unwrap());
    let tb = Arc::new(TrackedTensor::from_var(&b_var).unwrap());
    let y = ta.matmul(&tb).unwrap();
    assert_eq!(y.dims(), &[2, 2, 2]);
    let loss = y.sqr().unwrap();
    let scalar = reduce_to_scalar(&loss);
    let grads = backward(&scalar).unwrap();

    let ga = grads.get(&a_var).unwrap();
    let gb = grads.get(&b_var).unwrap();
    assert_eq!(ga.dims(), &[2, 2, 3]);
    assert_eq!(gb.dims(), &[2, 3, 2]);

    let analytical_a = ga.to_flat_vec::<f32>().unwrap();
    check_fd_grad(&analytical_a, &a_data, 1e-3, 1e-2, &|d| {
        let a = DynTensor::from_vec(d, &[2, 2, 3], &cpu()).unwrap();
        let b = DynTensor::from_vec(b_data.clone(), &[2, 3, 2], &cpu()).unwrap();
        scalar_sum_sqr(&a.matmul(&b).unwrap())
    });
}

// ============================================================================
// 5. Normalization gradients
// ============================================================================

/// LayerNorm gradient: input, weight, bias all receive correct shapes + FD check.
#[test]
fn test_higher_order_layer_norm_gradient() {
    let x_data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let w_data = vec![1.0, 1.0, 1.0];
    let b_data = vec![0.0, 0.0, 0.0];
    let x_var = shaped_var(x_data.clone(), &[2, 3]);
    let w_var = shaped_var(w_data.clone(), &[3]);
    let b_var = shaped_var(b_data.clone(), &[3]);

    let tx = Arc::new(TrackedTensor::from_var(&x_var).unwrap());
    let tw = Arc::new(TrackedTensor::from_var(&w_var).unwrap());
    let tb = Arc::new(TrackedTensor::from_var(&b_var).unwrap());
    let y = tx.layer_norm(&tw, &tb, 1e-5).unwrap();
    let loss = y.sqr().unwrap();
    let scalar = reduce_to_scalar(&loss);
    let grads = backward(&scalar).unwrap();

    let gx = grads.get(&x_var).unwrap();
    let gw = grads.get(&w_var).unwrap();
    let gb = grads.get(&b_var).unwrap();
    assert_eq!(gx.dims(), &[2, 3], "input grad shape");
    assert_eq!(gw.dims(), &[3], "weight grad shape");
    assert_eq!(gb.dims(), &[3], "bias grad shape");

    // FD check for input
    let analytical_x = gx.to_flat_vec::<f32>().unwrap();
    check_fd_grad(&analytical_x, &x_data, 1e-3, 5e-2, &|d| {
        let x = DynTensor::from_vec(d, &[2, 3], &cpu()).unwrap();
        let w = DynTensor::from_vec(w_data.clone(), &[3], &cpu()).unwrap();
        let b = DynTensor::from_vec(b_data.clone(), &[3], &cpu()).unwrap();
        let mean = x.mean_keepdim(1).unwrap();
        let diff = x.sub(&mean).unwrap();
        let var = diff.sqr().unwrap().mean_keepdim(1).unwrap();
        let inv_std = var
            .add_scalar(1e-5)
            .unwrap()
            .sqrt()
            .unwrap()
            .recip()
            .unwrap();
        let normed = diff.mul(&inv_std).unwrap();
        let y = normed.mul(&w).unwrap().add(&b).unwrap();
        scalar_sum_sqr(&y)
    });
}

/// RmsNorm gradient: input and weight shapes + FD check.
#[test]
fn test_higher_order_rms_norm_gradient() {
    let x_data = vec![1.0, -0.5, 0.3, 0.8, -1.2, 0.6];
    let w_data = vec![1.0, 1.0, 1.0];
    let x_var = shaped_var(x_data.clone(), &[2, 3]);
    let w_var = shaped_var(w_data.clone(), &[3]);

    let tx = Arc::new(TrackedTensor::from_var(&x_var).unwrap());
    let tw = Arc::new(TrackedTensor::from_var(&w_var).unwrap());
    let y = tx.rms_norm(&tw, 1e-5).unwrap();
    let loss = y.sqr().unwrap();
    let scalar = reduce_to_scalar(&loss);
    let grads = backward(&scalar).unwrap();

    let gx = grads.get(&x_var).unwrap();
    let gw = grads.get(&w_var).unwrap();
    assert_eq!(gx.dims(), &[2, 3], "input grad shape");
    assert_eq!(gw.dims(), &[3], "weight grad shape");

    // FD check for input via manual rms_norm forward
    let analytical_x = gx.to_flat_vec::<f32>().unwrap();
    check_fd_grad(&analytical_x, &x_data, 1e-3, 5e-2, &|d| {
        let x = DynTensor::from_vec(d, &[2, 3], &cpu()).unwrap();
        let w = DynTensor::from_vec(w_data.clone(), &[3], &cpu()).unwrap();
        let rms_sq = x.sqr().unwrap().mean_keepdim(1).unwrap();
        let inv_rms = rms_sq
            .add_scalar(1e-5)
            .unwrap()
            .sqrt()
            .unwrap()
            .recip()
            .unwrap();
        let normed = x.mul(&inv_rms).unwrap();
        let y = normed.mul(&w).unwrap();
        scalar_sum_sqr(&y)
    });
}

/// LayerNorm weight gradient FD check.
#[test]
fn test_higher_order_layer_norm_weight_fd() {
    let x_data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let w_data = vec![0.5, 1.5, 2.0];
    let b_data = vec![0.1, -0.1, 0.0];
    let x_var = shaped_var(x_data.clone(), &[2, 3]);
    let w_var = shaped_var(w_data.clone(), &[3]);
    let b_var = shaped_var(b_data.clone(), &[3]);

    let tx = Arc::new(TrackedTensor::from_var(&x_var).unwrap());
    let tw = Arc::new(TrackedTensor::from_var(&w_var).unwrap());
    let tb = Arc::new(TrackedTensor::from_var(&b_var).unwrap());
    let y = tx.layer_norm(&tw, &tb, 1e-5).unwrap();
    let loss = y.sqr().unwrap();
    let scalar = reduce_to_scalar(&loss);
    let grads = backward(&scalar).unwrap();

    let gw = grads.get(&w_var).unwrap();
    let analytical_w = gw.to_flat_vec::<f32>().unwrap();
    check_fd_grad(&analytical_w, &w_data, 1e-3, 5e-2, &|d| {
        let x = DynTensor::from_vec(x_data.clone(), &[2, 3], &cpu()).unwrap();
        let w = DynTensor::from_vec(d, &[3], &cpu()).unwrap();
        let b = DynTensor::from_vec(b_data.clone(), &[3], &cpu()).unwrap();
        let mean = x.mean_keepdim(1).unwrap();
        let diff = x.sub(&mean).unwrap();
        let var = diff.sqr().unwrap().mean_keepdim(1).unwrap();
        let inv_std = var
            .add_scalar(1e-5)
            .unwrap()
            .sqrt()
            .unwrap()
            .recip()
            .unwrap();
        let normed = diff.mul(&inv_std).unwrap();
        let y = normed.mul(&w).unwrap().add(&b).unwrap();
        scalar_sum_sqr(&y)
    });
}

// ============================================================================
// 6. Loss function gradients
// ============================================================================

/// MSE loss gradient: d/dx = 2(x - target) / N, verified analytically.
#[test]
fn test_higher_order_mse_loss_analytical() {
    let x_data = vec![1.0, 2.5, -0.3, 0.7];
    let target_data = vec![1.5, 2.0, 0.0, 1.0];
    let x_var = shaped_var(x_data.clone(), &[2, 2]);
    let t_var = shaped_var(target_data.clone(), &[2, 2]);

    let tx = Arc::new(TrackedTensor::from_var(&x_var).unwrap());
    let tt = Arc::new(TrackedTensor::from_var(&t_var).unwrap());
    let loss = tx.mse_loss(&tt).unwrap();
    let grads = backward(&loss).unwrap();

    let gx = grads.get(&x_var).unwrap();
    assert_eq!(gx.dims(), &[2, 2], "MSE grad shape matches input");

    let n = 4.0;
    let expected: Vec<f32> = x_data
        .iter()
        .zip(target_data.iter())
        .map(|(x, t)| 2.0 * (x - t) / n)
        .collect();
    let actual = gx.to_flat_vec::<f32>().unwrap();
    for i in 0..4 {
        assert!(
            (actual[i] - expected[i]).abs() < 1e-5,
            "MSE grad[{i}]: got {}, expected {}",
            actual[i],
            expected[i]
        );
    }
}

/// MSE loss FD check.
#[test]
fn test_higher_order_mse_loss_fd() {
    let x_data = vec![1.0, 2.5, -0.3, 0.7];
    let target_data = vec![1.5, 2.0, 0.0, 1.0];
    let x_var = shaped_var(x_data.clone(), &[2, 2]);
    let t_var = shaped_var(target_data.clone(), &[2, 2]);

    let tx = Arc::new(TrackedTensor::from_var(&x_var).unwrap());
    let tt = Arc::new(TrackedTensor::from_var(&t_var).unwrap());
    let loss = tx.mse_loss(&tt).unwrap();
    let grads = backward(&loss).unwrap();

    let gx = grads.get(&x_var).unwrap();
    let analytical = gx.to_flat_vec::<f32>().unwrap();
    check_fd_grad(&analytical, &x_data, 1e-3, 1e-2, &|d| {
        let x = DynTensor::from_vec(d, &[2, 2], &cpu()).unwrap();
        let t = DynTensor::from_vec(target_data.clone(), &[2, 2], &cpu()).unwrap();
        let diff = x.sub(&t).unwrap();
        let sq = diff.sqr().unwrap();
        let vals = sq.to_flat_vec::<f32>().unwrap();
        vals.iter().map(|&v| f64::from(v)).sum::<f64>() / vals.len() as f64
    });
}

/// L1 loss gradient shape check + sign verification.
#[test]
fn test_higher_order_l1_loss_gradient() {
    let x_data = vec![1.0, 2.5, -0.3, 0.7];
    let target_data = vec![1.5, 2.0, 0.5, 1.0];
    let x_var = shaped_var(x_data, &[2, 2]);
    let t_var = shaped_var(target_data, &[2, 2]);

    let tx = Arc::new(TrackedTensor::from_var(&x_var).unwrap());
    let tt = Arc::new(TrackedTensor::from_var(&t_var).unwrap());
    let loss = tx.l1_loss(&tt).unwrap();
    let grads = backward(&loss).unwrap();

    let gx = grads.get(&x_var).unwrap();
    let gt = grads.get(&t_var).unwrap();
    assert_eq!(gx.dims(), &[2, 2], "L1 input grad shape");
    assert_eq!(gt.dims(), &[2, 2], "L1 target grad shape");
    // L1 grad for input and target should be negatives of each other
    let gx_vals = gx.to_flat_vec::<f32>().unwrap();
    let gt_vals = gt.to_flat_vec::<f32>().unwrap();
    for i in 0..4 {
        assert!(
            (gx_vals[i] + gt_vals[i]).abs() < 1e-5,
            "L1 grad[{i}]: input={}, target={}, should be negatives",
            gx_vals[i],
            gt_vals[i]
        );
    }
}

/// Huber loss gradient FD check.
#[test]
fn test_higher_order_huber_loss_fd() {
    let x_data = vec![1.0, 2.5, -0.3, 0.7];
    let target_data = vec![1.5, 2.0, 0.0, 1.0];
    let delta = 1.0;
    let x_var = shaped_var(x_data.clone(), &[2, 2]);
    let t_var = shaped_var(target_data.clone(), &[2, 2]);

    let tx = Arc::new(TrackedTensor::from_var(&x_var).unwrap());
    let tt = Arc::new(TrackedTensor::from_var(&t_var).unwrap());
    let loss = tx.huber_loss(&tt, delta).unwrap();
    let grads = backward(&loss).unwrap();

    let gx = grads.get(&x_var).unwrap();
    assert_eq!(gx.dims(), &[2, 2]);

    let analytical = gx.to_flat_vec::<f32>().unwrap();
    check_fd_grad(&analytical, &x_data, 1e-3, 1e-2, &|d| {
        let x = DynTensor::from_vec(d, &[2, 2], &cpu()).unwrap();
        let t = DynTensor::from_vec(target_data.clone(), &[2, 2], &cpu()).unwrap();
        let diff = x.sub(&t).unwrap();
        let abs_diff = diff.abs().unwrap();
        let quad = diff.sqr().unwrap().mul_scalar(0.5 / delta).unwrap();
        let lin = abs_diff.add_scalar(-0.5 * delta).unwrap();
        let mask = abs_diff.lt(delta).unwrap();
        let result = mask.where_cond(&quad, &lin).unwrap();
        let vals = result.to_flat_vec::<f32>().unwrap();
        vals.iter().map(|&v| f64::from(v)).sum::<f64>() / vals.len() as f64
    });
}

/// Cross-entropy loss gradient shape.
#[test]
fn test_higher_order_cross_entropy_gradient_shape() {
    let logits_data = vec![0.5, 1.2, -0.3, 0.7, -0.1, 0.9, 0.2, 1.5];
    let x_var = shaped_var(logits_data, &[2, 4]);
    // Cross-entropy targets must be U32 (gather indices)
    let targets_u32 = DynTensor::from_vec_u32(vec![1u32, 3u32], &[2, 1], &cpu()).unwrap();
    let targets = Var::new(targets_u32);

    let tx = Arc::new(TrackedTensor::from_var(&x_var).unwrap());
    let tt = Arc::new(TrackedTensor::from_var(&targets).unwrap());
    let loss = tx.cross_entropy_loss(&tt, 1).unwrap();
    let grads = backward(&loss).unwrap();

    let gx = grads.get(&x_var).unwrap();
    assert_eq!(gx.dims(), &[2, 4], "CE grad shape matches logits");
}

// ============================================================================
// 7. Numerical gradient checking for key operations
// ============================================================================

/// FD check: sin.
#[test]
fn test_higher_order_fd_sin() {
    let x_data = vec![0.3, 1.0, -0.7, 2.0];
    let x = shaped_var(x_data.clone(), &[4]);
    let t = Arc::new(TrackedTensor::from_var(&x).unwrap());
    let loss = reduce_to_scalar(&t.sin().unwrap());
    let grads = backward(&loss).unwrap();
    let analytical = grads.get(&x).unwrap().to_flat_vec::<f32>().unwrap();
    check_fd_grad(&analytical, &x_data, 1e-4, 1e-2, &|d| {
        scalar_sum(&DynTensor::from_vec(d, &[4], &cpu()).unwrap().sin().unwrap())
    });
}

/// FD check: cos.
#[test]
fn test_higher_order_fd_cos() {
    let x_data = vec![0.3, 1.0, -0.7, 2.0];
    let x = shaped_var(x_data.clone(), &[4]);
    let t = Arc::new(TrackedTensor::from_var(&x).unwrap());
    let loss = reduce_to_scalar(&t.cos().unwrap());
    let grads = backward(&loss).unwrap();
    let analytical = grads.get(&x).unwrap().to_flat_vec::<f32>().unwrap();
    check_fd_grad(&analytical, &x_data, 1e-4, 1e-2, &|d| {
        scalar_sum(&DynTensor::from_vec(d, &[4], &cpu()).unwrap().cos().unwrap())
    });
}

/// FD check: sigmoid.
#[test]
fn test_higher_order_fd_sigmoid() {
    let x_data = vec![-1.0, 0.0, 0.5, 2.0];
    let x = shaped_var(x_data.clone(), &[4]);
    let t = Arc::new(TrackedTensor::from_var(&x).unwrap());
    let loss = reduce_to_scalar(&t.sigmoid().unwrap());
    let grads = backward(&loss).unwrap();
    let analytical = grads.get(&x).unwrap().to_flat_vec::<f32>().unwrap();
    check_fd_grad(&analytical, &x_data, 1e-4, 1e-2, &|d| {
        scalar_sum(
            &DynTensor::from_vec(d, &[4], &cpu())
                .unwrap()
                .sigmoid()
                .unwrap(),
        )
    });
}

/// FD check: tanh.
#[test]
fn test_higher_order_fd_tanh() {
    let x_data = vec![-1.0, 0.0, 0.5, 1.5];
    let x = shaped_var(x_data.clone(), &[4]);
    let t = Arc::new(TrackedTensor::from_var(&x).unwrap());
    let loss = reduce_to_scalar(&t.tanh().unwrap());
    let grads = backward(&loss).unwrap();
    let analytical = grads.get(&x).unwrap().to_flat_vec::<f32>().unwrap();
    check_fd_grad(&analytical, &x_data, 1e-4, 1e-2, &|d| {
        scalar_sum(
            &DynTensor::from_vec(d, &[4], &cpu())
                .unwrap()
                .tanh()
                .unwrap(),
        )
    });
}

/// FD check: silu.
#[test]
fn test_higher_order_fd_silu() {
    let x_data = vec![-1.0, 0.0, 0.5, 2.0];
    let x = shaped_var(x_data.clone(), &[4]);
    let t = Arc::new(TrackedTensor::from_var(&x).unwrap());
    let loss = reduce_to_scalar(&t.silu().unwrap());
    let grads = backward(&loss).unwrap();
    let analytical = grads.get(&x).unwrap().to_flat_vec::<f32>().unwrap();
    check_fd_grad(&analytical, &x_data, 1e-4, 1e-2, &|d| {
        scalar_sum(
            &DynTensor::from_vec(d, &[4], &cpu())
                .unwrap()
                .silu()
                .unwrap(),
        )
    });
}

/// FD check: gelu.
#[test]
fn test_higher_order_fd_gelu() {
    let x_data = vec![-1.0, 0.0, 0.5, 2.0];
    let x = shaped_var(x_data.clone(), &[4]);
    let t = Arc::new(TrackedTensor::from_var(&x).unwrap());
    let loss = reduce_to_scalar(&t.gelu().unwrap());
    let grads = backward(&loss).unwrap();
    let analytical = grads.get(&x).unwrap().to_flat_vec::<f32>().unwrap();
    check_fd_grad(&analytical, &x_data, 1e-4, 1e-2, &|d| {
        scalar_sum(
            &DynTensor::from_vec(d, &[4], &cpu())
                .unwrap()
                .gelu()
                .unwrap(),
        )
    });
}

/// FD check: recip (1/x), avoiding zero.
#[test]
fn test_higher_order_fd_recip() {
    let x_data = vec![0.5, 1.0, 2.0, -1.5];
    let x = shaped_var(x_data.clone(), &[4]);
    let t = Arc::new(TrackedTensor::from_var(&x).unwrap());
    let loss = reduce_to_scalar(&t.recip().unwrap());
    let grads = backward(&loss).unwrap();
    let analytical = grads.get(&x).unwrap().to_flat_vec::<f32>().unwrap();
    check_fd_grad(&analytical, &x_data, 1e-4, 1e-2, &|d| {
        scalar_sum(
            &DynTensor::from_vec(d, &[4], &cpu())
                .unwrap()
                .recip()
                .unwrap(),
        )
    });
}

/// FD check: powf (x^2.5), positive inputs.
#[test]
fn test_higher_order_fd_powf() {
    let x_data = vec![0.5, 1.0, 2.0, 3.0];
    let x = shaped_var(x_data.clone(), &[4]);
    let t = Arc::new(TrackedTensor::from_var(&x).unwrap());
    let loss = reduce_to_scalar(&t.powf(2.5).unwrap());
    let grads = backward(&loss).unwrap();
    let analytical = grads.get(&x).unwrap().to_flat_vec::<f32>().unwrap();
    check_fd_grad(&analytical, &x_data, 1e-3, 1e-2, &|d| {
        scalar_sum(
            &DynTensor::from_vec(d, &[4], &cpu())
                .unwrap()
                .powf(2.5)
                .unwrap(),
        )
    });
}

/// FD check: softmax (nonlinear loss to expose Jacobian).
#[test]
fn test_higher_order_fd_softmax() {
    let x_data = vec![1.0, 2.0, 3.0, 0.5, 1.5, 2.5];
    let x = shaped_var(x_data.clone(), &[2, 3]);
    let t = Arc::new(TrackedTensor::from_var(&x).unwrap());
    let y = t.softmax(1).unwrap().sqr().unwrap();
    let scalar = reduce_to_scalar(&y);
    let grads = backward(&scalar).unwrap();
    let gx = grads.get(&x).unwrap();
    assert_eq!(gx.dims(), &[2, 3]);
    let analytical = gx.to_flat_vec::<f32>().unwrap();
    check_fd_grad(&analytical, &x_data, 1e-3, 1e-2, &|d| {
        let t = DynTensor::from_vec(d, &[2, 3], &cpu()).unwrap();
        scalar_sum_sqr(&t.softmax(1).unwrap())
    });
}

/// FD check: log_softmax.
#[test]
fn test_higher_order_fd_log_softmax() {
    let x_data = vec![1.0, 2.0, 3.0, 0.5, 1.5, 2.5];
    let x = shaped_var(x_data.clone(), &[2, 3]);
    let t = Arc::new(TrackedTensor::from_var(&x).unwrap());
    let y = t.log_softmax(1).unwrap().sqr().unwrap();
    let scalar = reduce_to_scalar(&y);
    let grads = backward(&scalar).unwrap();
    let gx = grads.get(&x).unwrap();
    assert_eq!(gx.dims(), &[2, 3]);
    let analytical = gx.to_flat_vec::<f32>().unwrap();
    check_fd_grad(&analytical, &x_data, 1e-3, 1e-2, &|d| {
        let t = DynTensor::from_vec(d, &[2, 3], &cpu()).unwrap();
        scalar_sum_sqr(&t.log_softmax(1).unwrap())
    });
}

/// FD check: softplus.
#[test]
fn test_higher_order_fd_softplus() {
    let x_data = vec![-2.0, -0.5, 0.0, 1.0, 3.0];
    let x = shaped_var(x_data.clone(), &[5]);
    let t = Arc::new(TrackedTensor::from_var(&x).unwrap());
    let loss = reduce_to_scalar(&t.softplus().unwrap());
    let grads = backward(&loss).unwrap();
    let analytical = grads.get(&x).unwrap().to_flat_vec::<f32>().unwrap();
    check_fd_grad(&analytical, &x_data, 1e-4, 1e-2, &|d| {
        scalar_sum(
            &DynTensor::from_vec(d, &[5], &cpu())
                .unwrap()
                .softplus()
                .unwrap(),
        )
    });
}

/// FD check: elu(alpha=1.0).
#[test]
fn test_higher_order_fd_elu() {
    let x_data = vec![-2.0, -0.5, 0.0, 1.0, 3.0];
    let x = shaped_var(x_data.clone(), &[5]);
    let t = Arc::new(TrackedTensor::from_var(&x).unwrap());
    let loss = reduce_to_scalar(&t.elu(1.0).unwrap());
    let grads = backward(&loss).unwrap();
    let analytical = grads.get(&x).unwrap().to_flat_vec::<f32>().unwrap();
    check_fd_grad(&analytical, &x_data, 1e-4, 1e-2, &|d| {
        scalar_sum(
            &DynTensor::from_vec(d, &[5], &cpu())
                .unwrap()
                .elu(1.0)
                .unwrap(),
        )
    });
}

/// FD check: mish.
#[test]
fn test_higher_order_fd_mish() {
    let x_data = vec![-2.0, -0.5, 0.0, 1.0, 3.0];
    let x = shaped_var(x_data.clone(), &[5]);
    let t = Arc::new(TrackedTensor::from_var(&x).unwrap());
    let loss = reduce_to_scalar(&t.mish().unwrap());
    let grads = backward(&loss).unwrap();
    let analytical = grads.get(&x).unwrap().to_flat_vec::<f32>().unwrap();
    check_fd_grad(&analytical, &x_data, 1e-4, 1e-2, &|d| {
        scalar_sum(
            &DynTensor::from_vec(d, &[5], &cpu())
                .unwrap()
                .mish()
                .unwrap(),
        )
    });
}

/// FD check: clamp spanning below/within/above range.
#[test]
fn test_higher_order_fd_clamp() {
    let x_data = vec![-2.0, 0.0, 0.5, 3.0];
    let x = shaped_var(x_data.clone(), &[4]);
    let t = Arc::new(TrackedTensor::from_var(&x).unwrap());
    let y = t.clamp(-1.0, 2.0).unwrap().sqr().unwrap();
    let scalar = reduce_to_scalar(&y);
    let grads = backward(&scalar).unwrap();
    let analytical = grads.get(&x).unwrap().to_flat_vec::<f32>().unwrap();
    check_fd_grad(&analytical, &x_data, 1e-3, 1e-2, &|d| {
        scalar_sum_sqr(
            &DynTensor::from_vec(d, &[4], &cpu())
                .unwrap()
                .clamp(-1.0, 2.0)
                .unwrap(),
        )
    });
}

// ============================================================================
// 8. Multi-step composition: realistic network patterns
// ============================================================================

/// Linear + relu pattern: y = relu(x @ w + b).
#[test]
fn test_higher_order_linear_relu_pattern() {
    let x_data = vec![0.5, -0.3, 1.2, 0.8];
    let w_data = vec![0.1, 0.3, -0.2, 0.5, 0.4, -0.1];
    let b_data = vec![0.1, 0.2, 0.3];
    let x_var = shaped_var(x_data.clone(), &[2, 2]);
    let w_var = shaped_var(w_data.clone(), &[2, 3]);
    let b_var = shaped_var(b_data.clone(), &[1, 3]);

    let tx = Arc::new(TrackedTensor::from_var(&x_var).unwrap());
    let tw = Arc::new(TrackedTensor::from_var(&w_var).unwrap());
    let tb = Arc::new(TrackedTensor::from_var(&b_var).unwrap());

    let h = tx.matmul(&tw).unwrap();
    let h_bias = h.add(&tb).unwrap();
    let y = h_bias.relu().unwrap();
    let loss = y.sqr().unwrap();
    let scalar = reduce_to_scalar(&loss);
    let grads = backward(&scalar).unwrap();

    let gx = grads.get(&x_var).unwrap();
    let gw = grads.get(&w_var).unwrap();
    let gb = grads.get(&b_var).unwrap();
    assert_eq!(gx.dims(), &[2, 2], "input grad shape");
    assert_eq!(gw.dims(), &[2, 3], "weight grad shape");
    assert_eq!(gb.dims(), &[1, 3], "bias grad shape");

    let analytical_w = gw.to_flat_vec::<f32>().unwrap();
    check_fd_grad(&analytical_w, &w_data, 1e-3, 1e-2, &|d| {
        let x = DynTensor::from_vec(x_data.clone(), &[2, 2], &cpu()).unwrap();
        let w = DynTensor::from_vec(d, &[2, 3], &cpu()).unwrap();
        let b = DynTensor::from_vec(b_data.clone(), &[1, 3], &cpu()).unwrap();
        scalar_sum_sqr(&x.matmul(&w).unwrap().add(&b).unwrap().relu().unwrap())
    });
}

/// Two-layer MLP: sigmoid(relu(x @ w1) @ w2).
#[test]
fn test_higher_order_two_layer_mlp() {
    let x_data = vec![0.5, -0.3, 1.0, 0.2];
    let w1_data = vec![0.1, 0.4, -0.2, 0.3, 0.5, -0.1];
    let w2_data = vec![0.2, -0.1, 0.3, 0.4, -0.3, 0.1];
    let x_var = shaped_var(x_data.clone(), &[2, 2]);
    let w1_var = shaped_var(w1_data.clone(), &[2, 3]);
    let w2_var = shaped_var(w2_data.clone(), &[3, 2]);

    let tx = Arc::new(TrackedTensor::from_var(&x_var).unwrap());
    let tw1 = Arc::new(TrackedTensor::from_var(&w1_var).unwrap());
    let tw2 = Arc::new(TrackedTensor::from_var(&w2_var).unwrap());

    let h1 = tx.matmul(&tw1).unwrap().relu().unwrap();
    let h2 = h1.matmul(&tw2).unwrap().sigmoid().unwrap();
    let loss = h2.sqr().unwrap();
    let scalar = reduce_to_scalar(&loss);
    let grads = backward(&scalar).unwrap();

    assert_eq!(grads.get(&w1_var).unwrap().dims(), &[2, 3]);
    assert_eq!(grads.get(&w2_var).unwrap().dims(), &[3, 2]);

    let analytical_w2 = grads.get(&w2_var).unwrap().to_flat_vec::<f32>().unwrap();
    check_fd_grad(&analytical_w2, &w2_data, 1e-3, 1e-2, &|d| {
        let x = DynTensor::from_vec(x_data.clone(), &[2, 2], &cpu()).unwrap();
        let w1 = DynTensor::from_vec(w1_data.clone(), &[2, 3], &cpu()).unwrap();
        let w2 = DynTensor::from_vec(d, &[3, 2], &cpu()).unwrap();
        let h1 = x.matmul(&w1).unwrap().relu().unwrap();
        scalar_sum_sqr(&h1.matmul(&w2).unwrap().sigmoid().unwrap())
    });
}

/// Residual connection: y = x + relu(x @ w), gradient accumulates from both paths.
#[test]
fn test_higher_order_residual_connection() {
    let x_data = vec![0.5, -0.3, 1.0, 0.2];
    let w_data = vec![0.1, 0.4, -0.2, 0.3];
    let x_var = shaped_var(x_data.clone(), &[2, 2]);
    let w_var = shaped_var(w_data.clone(), &[2, 2]);

    let tx = Arc::new(TrackedTensor::from_var(&x_var).unwrap());
    let tw = Arc::new(TrackedTensor::from_var(&w_var).unwrap());

    let branch = tx.matmul(&tw).unwrap().relu().unwrap();
    let y = tx.add(&branch).unwrap();
    let loss = y.sqr().unwrap();
    let scalar = reduce_to_scalar(&loss);
    let grads = backward(&scalar).unwrap();

    let gx = grads.get(&x_var).unwrap();
    let gw = grads.get(&w_var).unwrap();
    assert_eq!(gx.dims(), &[2, 2]);
    assert_eq!(gw.dims(), &[2, 2]);

    // FD for x (gradient from both identity path and branch)
    let analytical_x = gx.to_flat_vec::<f32>().unwrap();
    check_fd_grad(&analytical_x, &x_data, 1e-3, 1e-2, &|d| {
        let x = DynTensor::from_vec(d, &[2, 2], &cpu()).unwrap();
        let w = DynTensor::from_vec(w_data.clone(), &[2, 2], &cpu()).unwrap();
        let branch = x.matmul(&w).unwrap().relu().unwrap();
        scalar_sum_sqr(&x.add(&branch).unwrap())
    });

    // FD for w
    let analytical_w = gw.to_flat_vec::<f32>().unwrap();
    check_fd_grad(&analytical_w, &w_data, 1e-3, 1e-2, &|d| {
        let x = DynTensor::from_vec(x_data.clone(), &[2, 2], &cpu()).unwrap();
        let w = DynTensor::from_vec(d, &[2, 2], &cpu()).unwrap();
        let branch = x.matmul(&w).unwrap().relu().unwrap();
        scalar_sum_sqr(&x.add(&branch).unwrap())
    });
}

/// Attention-like: softmax(Q @ K^T / sqrt(d)) @ V.
#[test]
fn test_higher_order_attention_pattern_fd() {
    let q_data: Vec<f32> = vec![0.1, 0.2, 0.3, -0.1, 0.5, 0.4];
    let k_data: Vec<f32> = vec![0.3, -0.2, 0.1, 0.4, -0.3, 0.5];
    let v_data: Vec<f32> = vec![1.0, 0.5, -0.3, 0.8, 0.7, -0.1];
    let scale = (2.0_f64).sqrt();

    let q_var = shaped_var(q_data.clone(), &[1, 3, 2]);
    let k_var = shaped_var(k_data.clone(), &[1, 3, 2]);
    let v_var = shaped_var(v_data.clone(), &[1, 3, 2]);

    let tq = Arc::new(TrackedTensor::from_var(&q_var).unwrap());
    let tk = Arc::new(TrackedTensor::from_var(&k_var).unwrap());
    let tv = Arc::new(TrackedTensor::from_var(&v_var).unwrap());

    let k_t = tk.transpose(1, 2).unwrap();
    let scores = tq.matmul(&k_t).unwrap().mul_scalar(1.0 / scale).unwrap();
    let attn = scores.softmax(2).unwrap();
    let y = attn.matmul(&tv).unwrap();
    let loss = y.sqr().unwrap();
    let scalar = reduce_to_scalar(&loss);
    let grads = backward(&scalar).unwrap();

    assert_eq!(grads.get(&q_var).unwrap().dims(), &[1, 3, 2], "Q grad");
    assert_eq!(grads.get(&k_var).unwrap().dims(), &[1, 3, 2], "K grad");
    assert_eq!(grads.get(&v_var).unwrap().dims(), &[1, 3, 2], "V grad");

    // FD check for V
    let analytical_v = grads.get(&v_var).unwrap().to_flat_vec::<f32>().unwrap();
    check_fd_grad(&analytical_v, &v_data, 1e-3, 5e-2, &|d| {
        let q = DynTensor::from_vec(q_data.clone(), &[1, 3, 2], &cpu()).unwrap();
        let k = DynTensor::from_vec(k_data.clone(), &[1, 3, 2], &cpu()).unwrap();
        let v = DynTensor::from_vec(d, &[1, 3, 2], &cpu()).unwrap();
        let k_t = k.transpose(1, 2).unwrap();
        let scores = q.matmul(&k_t).unwrap().mul_scalar(1.0 / scale).unwrap();
        let attn = scores.softmax(2).unwrap();
        scalar_sum_sqr(&attn.matmul(&v).unwrap())
    });
}

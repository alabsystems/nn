// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Higher-order autodiff tests: second derivatives (grad-of-grad via FD),
//! Jacobian computation, mixed partial derivatives, chain rule verification,
//! and gradient accumulation across multiple backward passes.
//!
//! Since nn's autodiff produces `DynTensor` gradients (not `TrackedTensor`),
//! true second-order differentiation through the backward graph is not
//! supported. Instead, these tests verify second derivatives by applying
//! finite-difference perturbation to the *first derivative function*
//! (i.e., perturb x, recompute backward, observe how the gradient changes).

use std::sync::Arc;

use nn_core::dyn_tensor::DynTensor;
use nn_core::test_utils::cpu;

use crate::grad::backward;
use crate::grad::test_helpers::{check_fd_grad_tol, scalar_var};
use crate::tracked::TrackedTensor;
use crate::var::Var;

// ---------------------------------------------------------------------------
// Helper: compute the gradient of a scalar function at a given point.
//
// `build_loss` takes a `Var` and returns the loss `Arc<TrackedTensor>`.
// Returns the gradient vector w.r.t. the variable.
// ---------------------------------------------------------------------------

fn grad_at(
    data: &[f32],
    shape: &[usize],
    build_loss: &dyn Fn(&Var) -> Arc<TrackedTensor>,
) -> Vec<f32> {
    let var = Var::new(DynTensor::from_vec(data.to_vec(), shape, &cpu()).unwrap());
    let loss = build_loss(&var);
    let grads = backward(&loss).unwrap();
    grads.get(&var).unwrap().to_flat_vec::<f32>().unwrap()
}

/// Compute second derivative via finite differences on the first derivative.
///
/// For a scalar function f(x), the second derivative at x0 is approximated by:
///   f''(x0) ≈ (f'(x0 + eps) - f'(x0 - eps)) / (2 * eps)
fn second_deriv_fd(x0: f32, eps: f32, build_loss: &dyn Fn(&Var) -> Arc<TrackedTensor>) -> f32 {
    let grad_plus = grad_at(&[x0 + eps], &[1], build_loss)[0];
    let grad_minus = grad_at(&[x0 - eps], &[1], build_loss)[0];
    (grad_plus - grad_minus) / (2.0 * eps)
}

// ===========================================================================
// Second derivative tests (grad of grad) for simple functions
// ===========================================================================

/// f(x) = x^3, f'(x) = 3x^2, f''(x) = 6x.
///
/// At x=2: f''(2) = 12.
#[test]
fn test_second_deriv_x_cubed() {
    let build = |var: &Var| -> Arc<TrackedTensor> {
        let t = Arc::new(TrackedTensor::from_var(var).unwrap());
        // x^3 = x * x^2
        let x_sqr = t.sqr().unwrap();
        let x_cubed = x_sqr.mul(&t).unwrap();
        // Sum to scalar (already scalar [1], but sum_keepdim is required)
        x_cubed.sum_keepdim(0).unwrap()
    };

    let x0 = 2.0_f32;
    let eps = 1e-3_f32;
    let d2 = second_deriv_fd(x0, eps, &build);
    let expected = 6.0 * x0; // 12.0
    assert!(
        (d2 - expected).abs() < 0.1,
        "x^3 second deriv at x={x0}: expected {expected}, got {d2}"
    );
}

/// f(x) = sin(x), f'(x) = cos(x), f''(x) = -sin(x).
///
/// At x=1: f''(1) = -sin(1) ≈ -0.8415.
#[test]
fn test_second_deriv_sin() {
    let build = |var: &Var| -> Arc<TrackedTensor> {
        let t = Arc::new(TrackedTensor::from_var(var).unwrap());
        t.sin().unwrap()
    };

    let x0 = 1.0_f32;
    let eps = 1e-3_f32;
    let d2 = second_deriv_fd(x0, eps, &build);
    let expected = -x0.sin();
    assert!(
        (d2 - expected).abs() < 1e-2,
        "sin(x) second deriv at x={x0}: expected {expected}, got {d2}"
    );
}

/// f(x) = exp(x), f'(x) = exp(x), f''(x) = exp(x).
///
/// At x=1: f''(1) = e ≈ 2.7183.
#[test]
fn test_second_deriv_exp() {
    let build = |var: &Var| -> Arc<TrackedTensor> {
        let t = Arc::new(TrackedTensor::from_var(var).unwrap());
        t.exp().unwrap()
    };

    let x0 = 1.0_f32;
    let eps = 1e-3_f32;
    let d2 = second_deriv_fd(x0, eps, &build);
    let expected = x0.exp();
    assert!(
        (d2 - expected).abs() < 1e-2,
        "exp(x) second deriv at x={x0}: expected {expected}, got {d2}"
    );
}

/// f(x) = log(x), f'(x) = 1/x, f''(x) = -1/x^2.
///
/// At x=2: f''(2) = -0.25.
#[test]
fn test_second_deriv_log() {
    let build = |var: &Var| -> Arc<TrackedTensor> {
        let t = Arc::new(TrackedTensor::from_var(var).unwrap());
        t.log().unwrap()
    };

    let x0 = 2.0_f32;
    let eps = 1e-3_f32;
    let d2 = second_deriv_fd(x0, eps, &build);
    let expected = -1.0 / (x0 * x0);
    assert!(
        (d2 - expected).abs() < 1e-2,
        "log(x) second deriv at x={x0}: expected {expected}, got {d2}"
    );
}

/// f(x) = tanh(x), f'(x) = 1 - tanh(x)^2, f''(x) = -2*tanh(x)*(1 - tanh(x)^2).
///
/// At x=0.5: f''(0.5) = -2*tanh(0.5)*(1 - tanh(0.5)^2).
#[test]
fn test_second_deriv_tanh() {
    let build = |var: &Var| -> Arc<TrackedTensor> {
        let t = Arc::new(TrackedTensor::from_var(var).unwrap());
        t.tanh().unwrap()
    };

    let x0 = 0.5_f32;
    let eps = 1e-3_f32;
    let d2 = second_deriv_fd(x0, eps, &build);
    let th = x0.tanh();
    let expected = -2.0 * th * (1.0 - th * th);
    assert!(
        (d2 - expected).abs() < 1e-2,
        "tanh(x) second deriv at x={x0}: expected {expected}, got {d2}"
    );
}

/// f(x) = x^4, f'(x) = 4x^3, f''(x) = 12x^2.
///
/// At x=1.5: f''(1.5) = 27.0.
#[test]
fn test_second_deriv_x_fourth() {
    let build = |var: &Var| -> Arc<TrackedTensor> {
        let t = Arc::new(TrackedTensor::from_var(var).unwrap());
        // x^4 = (x^2)^2
        let x_sqr = t.sqr().unwrap();
        x_sqr.sqr().unwrap()
    };

    let x0 = 1.5_f32;
    let eps = 1e-3_f32;
    let d2 = second_deriv_fd(x0, eps, &build);
    let expected = 12.0 * x0 * x0; // 27.0
    assert!(
        (d2 - expected).abs() < 0.2,
        "x^4 second deriv at x={x0}: expected {expected}, got {d2}"
    );
}

// ===========================================================================
// Jacobian computation for vector-valued functions
// ===========================================================================

/// Compute the Jacobian matrix J[i][j] = d(output_i)/d(input_j)
/// by running backward once per output element.
///
/// f: R^n -> R^m, Jacobian is m x n.
fn compute_jacobian(
    input_data: &[f32],
    input_shape: &[usize],
    build_outputs: &dyn Fn(&Var) -> Arc<TrackedTensor>,
    num_outputs: usize,
) -> Vec<Vec<f32>> {
    let mut jacobian = Vec::with_capacity(num_outputs);

    for i in 0..num_outputs {
        let var = Var::new(DynTensor::from_vec(input_data.to_vec(), input_shape, &cpu()).unwrap());
        let outputs = build_outputs(&var);
        // Select the i-th output element by narrowing, then sum to scalar
        let selected = outputs.narrow(0, i, 1).unwrap();
        let loss = selected.sum_keepdim(0).unwrap();
        let grads = backward(&loss).unwrap();
        let grad_row = grads.get(&var).unwrap().to_flat_vec::<f32>().unwrap();
        jacobian.push(grad_row);
    }

    jacobian
}

/// Jacobian of f(x) = [x1^2, x1*x2, x2^2] where x = [x1, x2].
///
/// J = [[2*x1, 0   ],
///      [x2,   x1  ],
///      [0,    2*x2]]
#[test]
fn test_jacobian_quadratic_map() {
    let x_data = [3.0_f32, 2.0];

    let build = |var: &Var| -> Arc<TrackedTensor> {
        let t = Arc::new(TrackedTensor::from_var(var).unwrap());
        // x1 = t[0], x2 = t[1]
        let x1 = t.narrow(0, 0, 1).unwrap(); // [1]
        let x2 = t.narrow(0, 1, 1).unwrap(); // [1]

        // f1 = x1^2
        let f1 = x1.sqr().unwrap();
        // f2 = x1 * x2
        let f2 = x1.mul(&x2).unwrap();
        // f3 = x2^2
        let f3 = x2.sqr().unwrap();

        // Stack: [f1, f2, f3] reshaped to [3]
        // Use add after reshaping to build the output vector
        let f1_r = f1.reshape(&[1]).unwrap();
        let f2_r = f2.reshape(&[1]).unwrap();
        let f3_r = f3.reshape(&[1]).unwrap();

        // Concatenate by building a [3]-tensor through TrackedTensor::stack
        TrackedTensor::stack(&[f1_r, f2_r, f3_r], 0)
            .unwrap()
            .squeeze(1)
            .unwrap()
    };

    let jac = compute_jacobian(&x_data, &[2], &build, 3);

    let (x1, x2) = (x_data[0], x_data[1]);
    // Row 0: [2*x1, 0]
    assert!((jac[0][0] - 2.0 * x1).abs() < 1e-4, "J[0][0]");
    assert!((jac[0][1] - 0.0).abs() < 1e-4, "J[0][1]");
    // Row 1: [x2, x1]
    assert!((jac[1][0] - x2).abs() < 1e-4, "J[1][0]");
    assert!((jac[1][1] - x1).abs() < 1e-4, "J[1][1]");
    // Row 2: [0, 2*x2]
    assert!((jac[2][0] - 0.0).abs() < 1e-4, "J[2][0]");
    assert!((jac[2][1] - 2.0 * x2).abs() < 1e-4, "J[2][1]");
}

/// Jacobian of f(x) = [sin(x1), cos(x2), x1*x2] where x = [x1, x2].
///
/// J = [[cos(x1),  0      ],
///      [0,       -sin(x2)],
///      [x2,       x1     ]]
#[test]
fn test_jacobian_trig_map() {
    let x_data = [1.0_f32, 0.5];

    let build = |var: &Var| -> Arc<TrackedTensor> {
        let t = Arc::new(TrackedTensor::from_var(var).unwrap());
        let x1 = t.narrow(0, 0, 1).unwrap();
        let x2 = t.narrow(0, 1, 1).unwrap();

        let f1 = x1.sin().unwrap();
        let f2 = x2.cos().unwrap();
        let f3 = x1.mul(&x2).unwrap();

        TrackedTensor::stack(&[f1, f2, f3], 0)
            .unwrap()
            .squeeze(1)
            .unwrap()
    };

    let jac = compute_jacobian(&x_data, &[2], &build, 3);

    let (x1, x2) = (x_data[0], x_data[1]);
    // Row 0: [cos(x1), 0]
    assert!((jac[0][0] - x1.cos()).abs() < 1e-4, "J[0][0]: cos(x1)");
    assert!((jac[0][1]).abs() < 1e-4, "J[0][1]: 0");
    // Row 1: [0, -sin(x2)]
    assert!((jac[1][0]).abs() < 1e-4, "J[1][0]: 0");
    assert!((jac[1][1] - (-x2.sin())).abs() < 1e-4, "J[1][1]: -sin(x2)");
    // Row 2: [x2, x1]
    assert!((jac[2][0] - x2).abs() < 1e-4, "J[2][0]: x2");
    assert!((jac[2][1] - x1).abs() < 1e-4, "J[2][1]: x1");
}

/// Jacobian of linear map f(x) = A*x where A = [[1,2],[3,4],[5,6]], x = [x1,x2].
///
/// Jacobian = A (constant, regardless of x).
#[test]
fn test_jacobian_linear_map() {
    let x_data = [1.0_f32, 1.0];

    let build = |var: &Var| -> Arc<TrackedTensor> {
        let t = Arc::new(TrackedTensor::from_var(var).unwrap());
        // A is a constant [3, 2] matrix
        let a_data =
            DynTensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[3, 2], &cpu()).unwrap();
        let a = Arc::new(TrackedTensor::from_tensor(a_data));

        // f = A @ x, need x as [2, 1] then squeeze
        let x_col = t.reshape(&[2, 1]).unwrap();
        let result = a.matmul(&x_col).unwrap(); // [3, 1]
        result.squeeze(1).unwrap() // [3]
    };

    let jac = compute_jacobian(&x_data, &[2], &build, 3);

    // Jacobian should be A
    let a = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];
    for i in 0..3 {
        for j in 0..2 {
            assert!(
                (jac[i][j] - a[i][j]).abs() < 1e-4,
                "J[{i}][{j}]: expected {}, got {}",
                a[i][j],
                jac[i][j]
            );
        }
    }
}

// ===========================================================================
// Mixed partial derivatives
// ===========================================================================

/// f(x, y) = x^2 * y + x * y^3
///
/// df/dx = 2*x*y + y^3
/// df/dy = x^2 + 3*x*y^2
/// d^2f/dxdy = 2*x + 3*y^2
///
/// Verify mixed partial via FD on df/dx w.r.t. y.
#[test]
fn test_mixed_partial_polynomial() {
    let x0 = 2.0_f32;
    let y0 = 1.5_f32;
    let eps = 1e-3_f32;

    // Compute df/dx at (x0, y0+eps) and (x0, y0-eps)
    let grad_x_at = |y_val: f32| -> f32 {
        let x_var = Var::new(DynTensor::from_vec(vec![x0], &[1], &cpu()).unwrap());
        let y_const = Arc::new(TrackedTensor::from_tensor(
            DynTensor::from_vec(vec![y_val], &[1], &cpu()).unwrap(),
        ));
        let tx = Arc::new(TrackedTensor::from_var(&x_var).unwrap());

        // f = x^2 * y + x * y^3
        let x_sqr = tx.sqr().unwrap();
        let term1 = x_sqr.mul(&y_const).unwrap();
        let y_sqr = y_const.sqr().unwrap();
        let y_cubed = y_sqr.mul(&y_const).unwrap();
        let term2 = tx.mul(&y_cubed).unwrap();
        let loss = term1.add(&term2).unwrap();
        let loss_scalar = loss.sum_keepdim(0).unwrap();

        let grads = backward(&loss_scalar).unwrap();
        grads.get(&x_var).unwrap().to_flat_vec::<f32>().unwrap()[0]
    };

    let d2f_dxdy = (grad_x_at(y0 + eps) - grad_x_at(y0 - eps)) / (2.0 * eps);
    let expected = 2.0 * x0 + 3.0 * y0 * y0; // 2*2 + 3*2.25 = 10.75

    assert!(
        (d2f_dxdy - expected).abs() < 0.1,
        "d^2f/dxdy at ({x0},{y0}): expected {expected}, got {d2f_dxdy}"
    );
}

/// f(x, y) = sin(x) * exp(y)
///
/// df/dx = cos(x) * exp(y)
/// d^2f/dxdy = cos(x) * exp(y)  (same as df/dx!)
/// d^2f/dydx = cos(x) * exp(y)  (equality of mixed partials)
///
/// Verify symmetry of mixed partials.
#[test]
fn test_mixed_partial_symmetry() {
    let x0 = 1.0_f32;
    let y0 = 0.5_f32;
    let eps = 1e-3_f32;

    // d^2f/dxdy: perturb y, observe change in df/dx
    let grad_x_at_y = |y_val: f32| -> f32 {
        let x_var = Var::new(DynTensor::from_vec(vec![x0], &[1], &cpu()).unwrap());
        let y_const = Arc::new(TrackedTensor::from_tensor(
            DynTensor::from_vec(vec![y_val], &[1], &cpu()).unwrap(),
        ));
        let tx = Arc::new(TrackedTensor::from_var(&x_var).unwrap());
        let f = tx.sin().unwrap().mul(&y_const.exp().unwrap()).unwrap();
        let loss = f.sum_keepdim(0).unwrap();
        let grads = backward(&loss).unwrap();
        grads.get(&x_var).unwrap().to_flat_vec::<f32>().unwrap()[0]
    };

    // d^2f/dydx: perturb x, observe change in df/dy
    let grad_y_at_x = |x_val: f32| -> f32 {
        let y_var = Var::new(DynTensor::from_vec(vec![y0], &[1], &cpu()).unwrap());
        let x_const = Arc::new(TrackedTensor::from_tensor(
            DynTensor::from_vec(vec![x_val], &[1], &cpu()).unwrap(),
        ));
        let ty = Arc::new(TrackedTensor::from_var(&y_var).unwrap());
        let f = x_const.sin().unwrap().mul(&ty.exp().unwrap()).unwrap();
        let loss = f.sum_keepdim(0).unwrap();
        let grads = backward(&loss).unwrap();
        grads.get(&y_var).unwrap().to_flat_vec::<f32>().unwrap()[0]
    };

    let d2_dxdy = (grad_x_at_y(y0 + eps) - grad_x_at_y(y0 - eps)) / (2.0 * eps);
    let d2_dydx = (grad_y_at_x(x0 + eps) - grad_y_at_x(x0 - eps)) / (2.0 * eps);
    let expected = x0.cos() * y0.exp();

    assert!(
        (d2_dxdy - expected).abs() < 1e-2,
        "d^2f/dxdy: expected {expected}, got {d2_dxdy}"
    );
    assert!(
        (d2_dydx - expected).abs() < 1e-2,
        "d^2f/dydx: expected {expected}, got {d2_dydx}"
    );
    assert!(
        (d2_dxdy - d2_dydx).abs() < 1e-2,
        "mixed partials should be equal: d2_dxdy={d2_dxdy}, d2_dydx={d2_dydx}"
    );
}

// ===========================================================================
// Chain rule verification for composed functions
// ===========================================================================

/// f(x) = exp(sin(x)), f'(x) = cos(x) * exp(sin(x)).
///
/// Verify analytical gradient matches FD.
#[test]
fn test_chain_rule_exp_sin() {
    let x_data = vec![0.5_f32, 1.0, 1.5, 2.0];
    let eps = 1e-4_f32;

    let var = Var::new(DynTensor::from_vec(x_data.clone(), &[4], &cpu()).unwrap());
    let t = Arc::new(TrackedTensor::from_var(&var).unwrap());
    let y = t.sin().unwrap().exp().unwrap();
    let loss = y.sum_keepdim(0).unwrap();
    let grads = backward(&loss).unwrap();
    let analytical = grads.get(&var).unwrap().to_flat_vec::<f32>().unwrap();

    // FD reference
    let fwd = |d: Vec<f32>| -> f64 { d.iter().map(|&x| f64::from((x.sin()).exp())).sum() };

    check_fd_grad_tol(&analytical, &x_data, eps, 1e-2, &fwd);

    // Also check against closed-form: f'(x_i) = cos(x_i) * exp(sin(x_i))
    for (i, &x) in x_data.iter().enumerate() {
        let expected = x.cos() * x.sin().exp();
        assert!(
            (analytical[i] - expected).abs() < 1e-4,
            "chain rule exp(sin(x)) at x={x}: expected {expected}, got {}",
            analytical[i]
        );
    }
}

/// f(x) = log(x^2 + 1), f'(x) = 2x / (x^2 + 1).
///
/// Tests chain rule through log and add_scalar composition.
#[test]
fn test_chain_rule_log_sqr_plus_one() {
    let x_data = vec![0.5_f32, 1.0, 2.0, -1.0];
    let eps = 1e-4_f32;

    let var = Var::new(DynTensor::from_vec(x_data.clone(), &[4], &cpu()).unwrap());
    let t = Arc::new(TrackedTensor::from_var(&var).unwrap());
    // f(x) = log(x^2 + 1)
    let y = t.sqr().unwrap().add_scalar(1.0).unwrap().log().unwrap();
    let loss = y.sum_keepdim(0).unwrap();
    let grads = backward(&loss).unwrap();
    let analytical = grads.get(&var).unwrap().to_flat_vec::<f32>().unwrap();

    // Check against closed-form
    for (i, &x) in x_data.iter().enumerate() {
        let expected = 2.0 * x / (x * x + 1.0);
        assert!(
            (analytical[i] - expected).abs() < 1e-4,
            "chain rule log(x^2+1) at x={x}: expected {expected}, got {}",
            analytical[i]
        );
    }

    // FD cross-check
    let fwd = |d: Vec<f32>| -> f64 { d.iter().map(|&x| f64::from(x * x + 1.0).ln()).sum() };
    check_fd_grad_tol(&analytical, &x_data, eps, 1e-2, &fwd);
}

/// f(x) = tanh(exp(x)), f'(x) = exp(x) * (1 - tanh(exp(x))^2).
///
/// Tests chain rule through tanh and exp composition.
#[test]
fn test_chain_rule_tanh_exp() {
    // Use small x to avoid tanh saturation
    let x_data = vec![-0.5_f32, 0.0, 0.3, 0.5];
    let eps = 1e-4_f32;

    let var = Var::new(DynTensor::from_vec(x_data.clone(), &[4], &cpu()).unwrap());
    let t = Arc::new(TrackedTensor::from_var(&var).unwrap());
    let y = t.exp().unwrap().tanh().unwrap();
    let loss = y.sum_keepdim(0).unwrap();
    let grads = backward(&loss).unwrap();
    let analytical = grads.get(&var).unwrap().to_flat_vec::<f32>().unwrap();

    for (i, &x) in x_data.iter().enumerate() {
        let ex = x.exp();
        let th = ex.tanh();
        let expected = ex * (1.0 - th * th);
        assert!(
            (analytical[i] - expected).abs() < 1e-4,
            "chain rule tanh(exp(x)) at x={x}: expected {expected}, got {}",
            analytical[i]
        );
    }

    let fwd = |d: Vec<f32>| -> f64 { d.iter().map(|&x| f64::from(x.exp().tanh())).sum() };
    check_fd_grad_tol(&analytical, &x_data, eps, 1e-2, &fwd);
}

/// Triple composition: f(x) = sqrt(exp(sin(x))).
///
/// f'(x) = cos(x) * exp(sin(x)) / (2 * sqrt(exp(sin(x))))
///       = cos(x) * sqrt(exp(sin(x))) / 2
#[test]
fn test_chain_rule_triple_composition() {
    let x_data = vec![0.3_f32, 0.7, 1.2];
    let eps = 1e-4_f32;

    let var = Var::new(DynTensor::from_vec(x_data.clone(), &[3], &cpu()).unwrap());
    let t = Arc::new(TrackedTensor::from_var(&var).unwrap());
    let y = t.sin().unwrap().exp().unwrap().sqrt().unwrap();
    let loss = y.sum_keepdim(0).unwrap();
    let grads = backward(&loss).unwrap();
    let analytical = grads.get(&var).unwrap().to_flat_vec::<f32>().unwrap();

    for (i, &x) in x_data.iter().enumerate() {
        let val = x.sin().exp().sqrt();
        let expected = x.cos() * val / 2.0;
        assert!(
            (analytical[i] - expected).abs() < 1e-4,
            "sqrt(exp(sin(x))) at x={x}: expected {expected}, got {}",
            analytical[i]
        );
    }

    let fwd = |d: Vec<f32>| -> f64 { d.iter().map(|&x| f64::from(x.sin().exp().sqrt())).sum() };
    check_fd_grad_tol(&analytical, &x_data, eps, 1e-2, &fwd);
}

// ===========================================================================
// Gradient accumulation across multiple backward passes
// ===========================================================================

/// Run backward twice on independent computation graphs sharing the same Var.
///
/// Gradients should be independent (each backward returns its own GradStore).
/// This verifies that backward does not mutate the Var or share state between
/// independent calls.
#[test]
fn test_gradient_accumulation_independent_backward() {
    let x = scalar_var(3.0);

    // First backward: y1 = x^2, dy1/dx = 6.0
    let t1 = Arc::new(TrackedTensor::from_var(&x).unwrap());
    let y1 = t1.sqr().unwrap();
    let grads1 = backward(&y1).unwrap();
    let g1 = grads1.get(&x).unwrap().to_flat_vec::<f32>().unwrap()[0];

    // Second backward: y2 = x^3, dy2/dx = 27.0
    let t2 = Arc::new(TrackedTensor::from_var(&x).unwrap());
    let x_sqr = t2.sqr().unwrap();
    let y2 = x_sqr.mul(&t2).unwrap();
    let y2_loss = y2.sum_keepdim(0).unwrap();
    let grads2 = backward(&y2_loss).unwrap();
    let g2 = grads2.get(&x).unwrap().to_flat_vec::<f32>().unwrap()[0];

    assert!(
        (g1 - 6.0).abs() < 1e-5,
        "first backward: expected 6.0, got {g1}"
    );
    assert!(
        (g2 - 27.0).abs() < 1e-4,
        "second backward: expected 27.0, got {g2}"
    );
}

/// Simulate manual gradient accumulation (as done in mini-batch training):
/// sum gradients from multiple backward passes, then verify the accumulated
/// gradient equals the gradient of the summed loss.
#[test]
fn test_gradient_accumulation_manual_sum() {
    // Two "mini-batches": loss1 = (x-1)^2, loss2 = (x-3)^2
    // Accumulated: d/dx[(x-1)^2 + (x-3)^2] = 2(x-1) + 2(x-3) = 4x - 8
    // At x=2: grad = 0.0

    let x_val = 2.0_f32;

    // Compute gradients separately and accumulate
    let grad_for_target = |target: f32| -> f32 {
        let x = Var::new(DynTensor::from_vec(vec![x_val], &[1], &cpu()).unwrap());
        let t = Arc::new(TrackedTensor::from_var(&x).unwrap());
        let c = Arc::new(TrackedTensor::from_tensor(
            DynTensor::from_vec(vec![target], &[1], &cpu()).unwrap(),
        ));
        let diff = t.sub(&c).unwrap();
        let loss = diff.sqr().unwrap();
        let grads = backward(&loss).unwrap();
        grads.get(&x).unwrap().to_flat_vec::<f32>().unwrap()[0]
    };

    let g1 = grad_for_target(1.0); // 2*(2-1) = 2
    let g2 = grad_for_target(3.0); // 2*(2-3) = -2
    let accumulated = g1 + g2; // 0

    assert!(
        (g1 - 2.0).abs() < 1e-5,
        "grad for target=1: expected 2.0, got {g1}"
    );
    assert!(
        (g2 - (-2.0)).abs() < 1e-5,
        "grad for target=3: expected -2.0, got {g2}"
    );
    assert!(
        accumulated.abs() < 1e-5,
        "accumulated grad at optimum: expected 0.0, got {accumulated}"
    );
}

/// Verify that the gradient of a sum of losses equals the sum of individual
/// gradients (linearity of differentiation).
#[test]
fn test_gradient_linearity() {
    let x_data = vec![1.0_f32, 2.0, 3.0];

    // Loss 1: sum(sin(x))
    let var1 = Var::new(DynTensor::from_vec(x_data.clone(), &[3], &cpu()).unwrap());
    let t1 = Arc::new(TrackedTensor::from_var(&var1).unwrap());
    let loss1 = t1.sin().unwrap().sum_keepdim(0).unwrap();
    let grads1 = backward(&loss1).unwrap();
    let g1 = grads1.get(&var1).unwrap().to_flat_vec::<f32>().unwrap();

    // Loss 2: sum(x^2)
    let var2 = Var::new(DynTensor::from_vec(x_data.clone(), &[3], &cpu()).unwrap());
    let t2 = Arc::new(TrackedTensor::from_var(&var2).unwrap());
    let loss2 = t2.sqr().unwrap().sum_keepdim(0).unwrap();
    let grads2 = backward(&loss2).unwrap();
    let g2 = grads2.get(&var2).unwrap().to_flat_vec::<f32>().unwrap();

    // Combined: sum(sin(x) + x^2)
    let var_combined = Var::new(DynTensor::from_vec(x_data, &[3], &cpu()).unwrap());
    let tc = Arc::new(TrackedTensor::from_var(&var_combined).unwrap());
    let combined_loss = tc
        .sin()
        .unwrap()
        .add(&tc.sqr().unwrap())
        .unwrap()
        .sum_keepdim(0)
        .unwrap();
    let grads_combined = backward(&combined_loss).unwrap();
    let gc = grads_combined
        .get(&var_combined)
        .unwrap()
        .to_flat_vec::<f32>()
        .unwrap();

    for i in 0..3 {
        let sum_of_grads = g1[i] + g2[i];
        assert!(
            (gc[i] - sum_of_grads).abs() < 1e-5,
            "linearity at [{i}]: combined={}, sum={sum_of_grads}",
            gc[i]
        );
    }
}

// ===========================================================================
// Second derivative of multi-element functions (Hessian diagonal)
// ===========================================================================

/// Compute the diagonal of the Hessian via FD on the gradient.
///
/// For f: R^n -> R, H[i][i] = d^2f/dx_i^2 ≈ (df/dx_i(x+eps*e_i) - df/dx_i(x-eps*e_i)) / (2*eps)
#[test]
fn test_hessian_diagonal_sum_of_functions() {
    // f(x) = sum_i [ sin(x_i) + x_i^3 ]
    // f'(x_i) = cos(x_i) + 3*x_i^2
    // f''(x_i) = -sin(x_i) + 6*x_i  (Hessian diagonal)
    let x_data = [1.0_f32, 2.0, 0.5];
    let eps = 1e-3_f32;

    let build = |var: &Var| -> Arc<TrackedTensor> {
        let t = Arc::new(TrackedTensor::from_var(var).unwrap());
        // sin(x)
        let sinx = t.sin().unwrap();
        // x^3 = x * x^2
        let x_sqr = t.sqr().unwrap();
        let x_cubed = x_sqr.mul(&t).unwrap();
        let total = sinx.add(&x_cubed).unwrap();
        total.sum_keepdim(0).unwrap()
    };

    for i in 0..3 {
        // Perturb element i and get gradient at element i
        let mut x_plus = x_data.to_vec();
        let mut x_minus = x_data.to_vec();
        x_plus[i] += eps;
        x_minus[i] -= eps;

        let grad_plus = grad_at(&x_plus, &[3], &build);
        let grad_minus = grad_at(&x_minus, &[3], &build);

        let hessian_ii = (grad_plus[i] - grad_minus[i]) / (2.0 * eps);
        let expected = -x_data[i].sin() + 6.0 * x_data[i];

        assert!(
            (hessian_ii - expected).abs() < 0.1,
            "H[{i}][{i}] at x={}: expected {expected}, got {hessian_ii}",
            x_data[i]
        );
    }
}

/// Hessian off-diagonal element for a coupled function.
///
/// f(x, y) = sin(x * y), d^2f/dxdy = cos(x*y) - x*y*sin(x*y).
#[test]
fn test_hessian_off_diagonal() {
    let x0 = 1.0_f32;
    let y0 = 0.5_f32;
    let eps = 1e-3_f32;

    // Compute df/dx at (x0, y0+eps) and (x0, y0-eps)
    let grad_x_at_y = |y_val: f32| -> f32 {
        let data = vec![x0, y_val];
        let var = Var::new(DynTensor::from_vec(data, &[2], &cpu()).unwrap());
        let t = Arc::new(TrackedTensor::from_var(&var).unwrap());
        let x_elem = t.narrow(0, 0, 1).unwrap();
        let y_elem = t.narrow(0, 1, 1).unwrap();
        let product = x_elem.mul(&y_elem).unwrap();
        let f = product.sin().unwrap();
        let loss = f.sum_keepdim(0).unwrap();
        let grads = backward(&loss).unwrap();
        grads.get(&var).unwrap().to_flat_vec::<f32>().unwrap()[0] // df/dx
    };

    let d2f_dxdy = (grad_x_at_y(y0 + eps) - grad_x_at_y(y0 - eps)) / (2.0 * eps);
    let xy = x0 * y0;
    // d^2f/dxdy = d/dy[y * cos(xy)] = cos(xy) + y * (-sin(xy) * x)
    //           = cos(xy) - xy * sin(xy)
    let expected = xy.cos() - xy * xy.sin();

    assert!(
        (d2f_dxdy - expected).abs() < 0.05,
        "d^2f/dxdy for sin(xy) at ({x0},{y0}): expected {expected}, got {d2f_dxdy}"
    );
}

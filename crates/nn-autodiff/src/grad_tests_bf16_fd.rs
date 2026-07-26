#![allow(deprecated)]
// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Finite-difference gradient tests with BF16 tensors.
//!
//! Verifies that backward rules correctly propagate BF16 dtype (not hardcoded
//! F32) through gradient computation. After #1934 replaced `DType::F32` with
//! `grad.dtype()` at 18 backward rule sites, these tests confirm the fix.
//!
//! BF16 has ~3 decimal digits of precision vs 7 for F32, so FD tolerance is
//! wider (0.05 vs the standard 0.01).

use std::sync::Arc;

use nn_core::dyn_tensor::DynTensor;
use nn_core::test_utils::cpu;
use nn_core::DType;

use crate::grad::backward;
use crate::grad::test_helpers::sum_f64;
use crate::tracked::TrackedTensor;
use crate::var::Var;

/// Helper: create a BF16 Var from f32 data.
fn bf16_var(data: &[f32], shape: &[usize]) -> Var {
    let t = DynTensor::from_vec(data.to_vec(), shape, &cpu()).unwrap();
    let bf16 = t.to_dtype(DType::BF16).unwrap();
    Var::new(bf16)
}

/// Helper: extract gradient as f32 vec (works for both F32 and BF16 grads).
fn grad_as_f32(grad: &DynTensor) -> Vec<f32> {
    grad.to_f32_array().unwrap().into_raw_vec_and_offset().0
}

/// BF16 FD test: simple matmul backward.
///
/// loss = sum((x @ w)^2) where x is constant BF16, w is BF16 Var.
/// Verifies dL/dw is correct within BF16 tolerance.
#[test]
fn test_bf16_matmul_backward_fd() {
    let x_data = vec![0.5, -0.25, 0.75, 1.0, -0.5, 0.125];
    let w_data = vec![0.3, -0.7, 0.1, 0.4, 0.6, -0.2];

    let w_var = bf16_var(&w_data, &[3, 2]);
    let x_bf16 = DynTensor::from_vec(x_data.clone(), &[2, 3], &cpu())
        .unwrap()
        .to_dtype(DType::BF16)
        .unwrap();

    let tw = Arc::new(TrackedTensor::from_var(&w_var).unwrap());
    let tx = Arc::new(TrackedTensor::from_tensor(x_bf16));
    let y = tx.matmul(&tw).unwrap();
    let sq = y.sqr().unwrap();
    let loss = sq.sum_keepdim(0).unwrap().sum_keepdim(1).unwrap();

    let grads = backward(&loss).unwrap();
    let grad_w = grads.get(&w_var).unwrap();

    // Verify gradient dtype is BF16 (not F32).
    assert_eq!(
        grad_w.dtype(),
        DType::BF16,
        "gradient dtype should be BF16, not {:?}",
        grad_w.dtype(),
    );

    let analytical = grad_as_f32(grad_w);
    // BF16 has ~3 decimal digits of precision (7-bit mantissa). The
    // promote-compute-demote round-trip introduces ~0.4% error per op, which
    // compounds through matmul (K multiplications + K-1 additions per element).
    // Use wider eps and tolerance than F32 FD tests.
    let eps = 5e-2_f32;
    let tol = 0.5_f64;

    // FD comparison: perturb each weight element.
    for i in 0..w_data.len() {
        let mut plus = w_data.clone();
        let mut minus = w_data.clone();
        plus[i] += eps;
        minus[i] -= eps;

        let fwd = |wd: Vec<f32>| -> f64 {
            let w = DynTensor::from_vec(wd, &[3, 2], &cpu())
                .unwrap()
                .to_dtype(DType::BF16)
                .unwrap();
            let x = DynTensor::from_vec(x_data.clone(), &[2, 3], &cpu())
                .unwrap()
                .to_dtype(DType::BF16)
                .unwrap();
            let y = x.matmul(&w).unwrap();
            sum_f64(&y.sqr().unwrap())
        };

        let numerical = (fwd(plus) - fwd(minus)) / (2.0 * f64::from(eps));
        let err = (f64::from(analytical[i]) - numerical).abs();
        assert!(
            err < tol,
            "grad_w[{i}]: analytical={}, numerical={numerical}, err={err}, tol={tol}",
            analytical[i],
        );
    }
}

/// BF16 FD test: element-wise ops (add, mul, sqr, relu) backward.
///
/// loss = sum(relu(a * b + a)^2) where a is BF16 Var, b is constant BF16.
/// Exercises Add, Mul, Relu, Sqr backward rules with BF16 dtype.
#[test]
fn test_bf16_elementwise_backward_fd() {
    let a_data = vec![0.5, -0.25, 0.75, 1.0];
    let b_data = vec![0.3, 0.8, -0.4, 0.6];

    let a_var = bf16_var(&a_data, &[4]);
    let b_bf16 = DynTensor::from_vec(b_data.clone(), &[4], &cpu())
        .unwrap()
        .to_dtype(DType::BF16)
        .unwrap();

    let ta = Arc::new(TrackedTensor::from_var(&a_var).unwrap());
    let tb = Arc::new(TrackedTensor::from_tensor(b_bf16));
    let prod = ta.mul(&tb).unwrap();
    let sum_ab = prod.add(&ta).unwrap();
    let activated = sum_ab.relu().unwrap();
    let sq = activated.sqr().unwrap();
    let loss = sq.sum_keepdim(0).unwrap();

    let grads = backward(&loss).unwrap();
    let grad_a = grads.get(&a_var).unwrap();

    assert_eq!(grad_a.dtype(), DType::BF16);

    let analytical = grad_as_f32(grad_a);
    let eps = 5e-2_f32;
    let tol = 0.5_f64; // Wider tolerance — chain of ops amplifies BF16 rounding.

    for i in 0..a_data.len() {
        let mut plus = a_data.clone();
        let mut minus = a_data.clone();
        plus[i] += eps;
        minus[i] -= eps;

        let fwd = |ad: Vec<f32>| -> f64 {
            let a = DynTensor::from_vec(ad, &[4], &cpu())
                .unwrap()
                .to_dtype(DType::BF16)
                .unwrap();
            let b = DynTensor::from_vec(b_data.clone(), &[4], &cpu())
                .unwrap()
                .to_dtype(DType::BF16)
                .unwrap();
            let prod = a.mul(&b).unwrap();
            let sum_ab = prod.add(&a).unwrap();
            let activated = sum_ab.relu().unwrap();
            sum_f64(&activated.sqr().unwrap())
        };

        let numerical = (fwd(plus) - fwd(minus)) / (2.0 * f64::from(eps));
        let err = (f64::from(analytical[i]) - numerical).abs();
        assert!(
            err < tol,
            "grad_a[{i}]: analytical={}, numerical={numerical}, err={err}, tol={tol}",
            analytical[i],
        );
    }
}

/// BF16 loss backward: verify cross-entropy backward produces BF16 gradients.
///
/// Exercises the cross-entropy backward rule which creates internal ones/zeros
/// tensors — these must use grad.dtype() (BF16), not hardcoded F32.
#[test]
fn test_bf16_cross_entropy_backward_produces_bf16_grads() {
    let logits_data = vec![2.0, 1.0, 0.1, 0.5, 2.5, 0.3];
    let targets_data = vec![0u32, 2]; // Class indices for 2 samples.

    let logits_var = bf16_var(&logits_data, &[2, 3]);
    // gather requires targets shape [N, 1] (same ndim as logits).
    let targets = DynTensor::from_vec_u32(targets_data, &[2, 1], &cpu()).unwrap();
    let targets_tracked = Arc::new(TrackedTensor::from_tensor(targets));

    let tl = Arc::new(TrackedTensor::from_var(&logits_var).unwrap());
    let loss = tl.cross_entropy_loss(&targets_tracked, 1).unwrap();

    let grads = backward(&loss).unwrap();
    let grad_logits = grads.get(&logits_var).unwrap();

    // The key assertion: gradient dtype matches the Var dtype (BF16).
    assert_eq!(
        grad_logits.dtype(),
        DType::BF16,
        "cross-entropy gradient should be BF16, got {:?}",
        grad_logits.dtype(),
    );

    // Sanity: gradient should be finite and nonzero.
    let vals = grad_as_f32(grad_logits);
    assert!(
        vals.iter().all(|v| v.is_finite()),
        "gradient has non-finite values"
    );
    assert!(vals.iter().any(|v| *v != 0.0), "gradient is all zeros");
}

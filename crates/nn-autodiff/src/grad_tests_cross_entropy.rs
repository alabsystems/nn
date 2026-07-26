#![allow(deprecated)]
// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Cross-entropy loss backward tests.
//!
//! AC3: Gradient correctness via finite-difference verification.
//! AC4: Multi-class classification (≥3 classes).
//! AC5: Numerical stability with large logit magnitudes.

use std::sync::Arc;

use nn_core::dyn_tensor::DynTensor;
use nn_core::test_utils::cpu;

use crate::grad::backward;
use crate::tracked::TrackedTensor;
use crate::var::Var;

/// Compute cross-entropy loss value directly (for reference).
fn reference_cross_entropy(logits: &[f32], targets: &[u32], num_classes: usize) -> f32 {
    let n = targets.len();
    let mut total = 0.0f32;
    for (i, &t) in targets.iter().enumerate() {
        let row = &logits[i * num_classes..(i + 1) * num_classes];
        let max_val = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let shifted: Vec<f32> = row.iter().map(|&x| (x - max_val).exp()).collect();
        let sum_exp: f32 = shifted.iter().sum();
        let log_softmax_t = (shifted[t as usize] / sum_exp).ln();
        total -= log_softmax_t;
    }
    total / n as f32
}

// -- AC3: Finite-difference gradient verification --

#[test]
fn test_cross_entropy_backward_finite_diff_binary() {
    // Binary classification: 2 classes, 3 samples
    let logits_data = vec![1.0, 2.0, 0.5, 0.5, -1.0, 3.0];
    let targets_data = vec![1u32, 0, 1];

    let logits_var = Var::new(DynTensor::from_vec(logits_data.clone(), &[3, 2], &cpu()).unwrap());
    let t_logits = Arc::new(TrackedTensor::from_var(&logits_var).unwrap());
    let t_targets = Arc::new(TrackedTensor::from_tensor(
        DynTensor::from_vec_u32(targets_data.clone(), &[3, 1], &cpu()).unwrap(),
    ));

    let loss = t_logits.cross_entropy_loss(&t_targets, 1).unwrap();
    let grads = backward(&loss).unwrap();
    let grad = grads
        .get(&logits_var)
        .unwrap()
        .to_flat_vec::<f32>()
        .unwrap();

    // Verify gradients via finite differences
    let eps = 1e-3;
    for idx in 0..logits_data.len() {
        let mut plus = logits_data.clone();
        plus[idx] += eps;
        let loss_plus = reference_cross_entropy(&plus, &targets_data, 2);

        let mut minus = logits_data.clone();
        minus[idx] -= eps;
        let loss_minus = reference_cross_entropy(&minus, &targets_data, 2);

        let fd_grad = (loss_plus - loss_minus) / (2.0 * eps);
        let err = (grad[idx] - fd_grad).abs();
        assert!(
            err < 1e-3,
            "grad[{idx}]: autodiff={:.6}, fd={:.6}, err={:.6}",
            grad[idx],
            fd_grad,
            err,
        );
    }
}

// -- AC4: Multi-class classification (≥3 classes) --

#[test]
fn test_cross_entropy_backward_multi_class() {
    // 4 classes, 2 samples
    let logits_data = vec![1.0, 2.0, 3.0, 4.0, 0.1, 0.2, 0.3, 0.4];
    let targets_data = vec![2u32, 0];

    let logits_var = Var::new(DynTensor::from_vec(logits_data.clone(), &[2, 4], &cpu()).unwrap());
    let t_logits = Arc::new(TrackedTensor::from_var(&logits_var).unwrap());
    let t_targets = Arc::new(TrackedTensor::from_tensor(
        DynTensor::from_vec_u32(targets_data.clone(), &[2, 1], &cpu()).unwrap(),
    ));

    let loss = t_logits.cross_entropy_loss(&t_targets, 1).unwrap();

    // Verify loss value matches reference
    let loss_val = loss.tensor().to_flat_vec::<f32>().unwrap()[0];
    let expected = reference_cross_entropy(&logits_data, &targets_data, 4);
    assert!(
        (loss_val - expected).abs() < 1e-5,
        "loss: got {loss_val}, expected {expected}",
    );

    // Verify gradients via finite differences
    let grads = backward(&loss).unwrap();
    let grad = grads
        .get(&logits_var)
        .unwrap()
        .to_flat_vec::<f32>()
        .unwrap();

    let eps = 1e-3;
    for idx in 0..logits_data.len() {
        let mut plus = logits_data.clone();
        plus[idx] += eps;
        let loss_plus = reference_cross_entropy(&plus, &targets_data, 4);

        let mut minus = logits_data.clone();
        minus[idx] -= eps;
        let loss_minus = reference_cross_entropy(&minus, &targets_data, 4);

        let fd_grad = (loss_plus - loss_minus) / (2.0 * eps);
        let err = (grad[idx] - fd_grad).abs();
        assert!(
            err < 1e-3,
            "grad[{idx}]: autodiff={:.6}, fd={:.6}, err={:.6}",
            grad[idx],
            fd_grad,
            err,
        );
    }
}

// -- AC5: Numerical stability with large logit magnitudes --

#[test]
fn test_cross_entropy_backward_large_logits() {
    // Large logit values that would cause overflow without log-sum-exp trick
    let logits_data = vec![1000.0, 1001.0, 999.0, -1000.0, -999.0, -998.0];
    let targets_data = vec![1u32, 2];

    let logits_var = Var::new(DynTensor::from_vec(logits_data, &[2, 3], &cpu()).unwrap());
    let t_logits = Arc::new(TrackedTensor::from_var(&logits_var).unwrap());
    let t_targets = Arc::new(TrackedTensor::from_tensor(
        DynTensor::from_vec_u32(targets_data, &[2, 1], &cpu()).unwrap(),
    ));

    let loss = t_logits.cross_entropy_loss(&t_targets, 1).unwrap();
    let loss_val = loss.tensor().to_flat_vec::<f32>().unwrap()[0];

    // Loss should be finite (no NaN or Inf)
    assert!(
        loss_val.is_finite(),
        "loss should be finite, got {loss_val}",
    );

    let grads = backward(&loss).unwrap();
    let grad = grads
        .get(&logits_var)
        .unwrap()
        .to_flat_vec::<f32>()
        .unwrap();

    // All gradients should be finite
    for (i, &g) in grad.iter().enumerate() {
        assert!(g.is_finite(), "grad[{i}] should be finite, got {g}");
    }

    // Gradient for correct class should be negative
    assert!(
        grad[1] < 0.0,
        "gradient for correct class should be negative, got {}",
        grad[1],
    );
}

#[test]
fn test_cross_entropy_forward_value_simple() {
    // Simple case: logits = [[0, 0]], target = [0]
    // log_softmax = [[-ln(2), -ln(2)]]
    // loss = -(-ln(2)) = ln(2) ≈ 0.6931
    let logits_var = Var::new(DynTensor::from_vec(vec![0.0, 0.0], &[1, 2], &cpu()).unwrap());
    let t_logits = Arc::new(TrackedTensor::from_var(&logits_var).unwrap());
    let t_targets = Arc::new(TrackedTensor::from_tensor(
        DynTensor::from_vec_u32(vec![0u32], &[1, 1], &cpu()).unwrap(),
    ));

    let loss = t_logits.cross_entropy_loss(&t_targets, 1).unwrap();
    let loss_val = loss.tensor().to_flat_vec::<f32>().unwrap()[0];
    let expected = 2.0f32.ln();
    assert!(
        (loss_val - expected).abs() < 1e-5,
        "expected {expected}, got {loss_val}",
    );
}

#[test]
fn test_cross_entropy_gradient_sums_to_zero() {
    // For cross-entropy with softmax, the gradient of each sample's logits
    // sums to zero (softmax probabilities sum to 1, one_hot sums to 1).
    let logits_data = vec![0.5, 1.5, 2.5, 3.5, -0.5, 0.5, 1.5, 2.5, 0.0, 0.0, 0.0, 0.0];
    let targets_data = vec![2u32, 1, 3];

    let logits_var = Var::new(DynTensor::from_vec(logits_data, &[3, 4], &cpu()).unwrap());
    let t_logits = Arc::new(TrackedTensor::from_var(&logits_var).unwrap());
    let t_targets = Arc::new(TrackedTensor::from_tensor(
        DynTensor::from_vec_u32(targets_data, &[3, 1], &cpu()).unwrap(),
    ));

    let loss = t_logits.cross_entropy_loss(&t_targets, 1).unwrap();
    let grads = backward(&loss).unwrap();
    let grad = grads
        .get(&logits_var)
        .unwrap()
        .to_flat_vec::<f32>()
        .unwrap();

    // Each sample's gradient should sum to ~0
    for sample in 0..3 {
        let row_sum: f32 = grad[sample * 4..(sample + 1) * 4].iter().sum();
        assert!(
            row_sum.abs() < 1e-6,
            "sample {sample} gradient sum should be ~0, got {row_sum}",
        );
    }
}

// -- AC2 (#1460): dim=0 cross-entropy with classes on first axis --

/// Reference cross-entropy with classes on axis 0 (transposed layout).
///
/// logits shape: [num_classes, num_samples], dim=0
/// targets: [num_samples] class indices
fn reference_cross_entropy_dim0(logits: &[f32], targets: &[u32], num_classes: usize) -> f32 {
    let num_samples = targets.len();
    let mut total = 0.0f32;
    for (j, &t) in targets.iter().enumerate() {
        // Gather the logits for sample j across all classes (column j)
        let col: Vec<f32> = (0..num_classes)
            .map(|c| logits[c * num_samples + j])
            .collect();
        let max_val = col.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let shifted: Vec<f32> = col.iter().map(|&x| (x - max_val).exp()).collect();
        let sum_exp: f32 = shifted.iter().sum();
        let log_softmax_t = (shifted[t as usize] / sum_exp).ln();
        total -= log_softmax_t;
    }
    total / num_samples as f32
}

#[test]
fn test_cross_entropy_backward_dim0_finite_diff() {
    // Classes on first axis: logits shape [3, 4] (3 classes, 4 samples), dim=0
    let logits_data = vec![
        1.0, 0.5, -0.5, 2.0, // class 0: [s0, s1, s2, s3]
        2.0, 1.0, 0.5, -1.0, // class 1
        0.0, 0.5, 1.0, 0.0, // class 2
    ];
    let targets_data = vec![1u32, 0, 2, 0]; // 4 samples, target classes

    let logits_var = Var::new(DynTensor::from_vec(logits_data.clone(), &[3, 4], &cpu()).unwrap());
    let t_logits = Arc::new(TrackedTensor::from_var(&logits_var).unwrap());
    let t_targets = Arc::new(TrackedTensor::from_tensor(
        DynTensor::from_vec_u32(targets_data.clone(), &[1, 4], &cpu()).unwrap(),
    ));

    let loss = t_logits.cross_entropy_loss(&t_targets, 0).unwrap();
    let loss_val = loss.tensor().to_flat_vec::<f32>().unwrap()[0];

    // Verify loss value matches reference
    let expected = reference_cross_entropy_dim0(&logits_data, &targets_data, 3);
    assert!(
        (loss_val - expected).abs() < 1e-4,
        "loss: got {loss_val}, expected {expected}",
    );

    // Verify gradients via finite differences
    let grads = backward(&loss).unwrap();
    let grad = grads
        .get(&logits_var)
        .unwrap()
        .to_flat_vec::<f32>()
        .unwrap();

    let eps = 1e-3;
    for idx in 0..logits_data.len() {
        let mut plus = logits_data.clone();
        plus[idx] += eps;
        let loss_plus = reference_cross_entropy_dim0(&plus, &targets_data, 3);

        let mut minus = logits_data.clone();
        minus[idx] -= eps;
        let loss_minus = reference_cross_entropy_dim0(&minus, &targets_data, 3);

        let fd_grad = (loss_plus - loss_minus) / (2.0 * eps);
        let err = (grad[idx] - fd_grad).abs();
        assert!(
            err < 1e-3,
            "grad[{idx}]: autodiff={:.6}, fd={:.6}, err={:.6}",
            grad[idx],
            fd_grad,
            err,
        );
    }
}

#[test]
fn test_cross_entropy_dim0_gradient_sums_to_zero() {
    // For dim=0, each column's gradients should sum to zero
    let logits_data = vec![
        0.5, 1.5, // class 0
        2.5, 0.5, // class 1
        1.0, 1.0, // class 2
    ];
    let targets_data = vec![2u32, 0];

    let logits_var = Var::new(DynTensor::from_vec(logits_data, &[3, 2], &cpu()).unwrap());
    let t_logits = Arc::new(TrackedTensor::from_var(&logits_var).unwrap());
    let t_targets = Arc::new(TrackedTensor::from_tensor(
        DynTensor::from_vec_u32(targets_data, &[1, 2], &cpu()).unwrap(),
    ));

    let loss = t_logits.cross_entropy_loss(&t_targets, 0).unwrap();
    let grads = backward(&loss).unwrap();
    let grad = grads
        .get(&logits_var)
        .unwrap()
        .to_flat_vec::<f32>()
        .unwrap();

    // Each column's (sample's) gradient should sum to ~0
    // Layout is [3, 2] row-major: [c0s0, c0s1, c1s0, c1s1, c2s0, c2s1]
    for sample in 0..2 {
        let col_sum: f32 = (0..3).map(|c| grad[c * 2 + sample]).sum();
        assert!(
            col_sum.abs() < 1e-6,
            "sample {sample} gradient column sum should be ~0, got {col_sum}",
        );
    }
}

// -- Empty batch guard (#1515 AC2) -------------------------------------------

/// Cross-entropy backward with zero-element batch must not produce Inf
/// from `1.0 / 0.0` division. Before the fix, `n = numel / num_classes = 0`
/// caused Inf propagation through the gradient.
#[test]
fn test_cross_entropy_empty_batch_no_inf_gradient() {
    // [0, 3] logits — zero batch, 3 classes
    let logits_var = Var::new(DynTensor::from_vec(Vec::<f32>::new(), &[0, 3], &cpu()).unwrap());
    let targets_var = Var::new(DynTensor::from_vec(Vec::<f32>::new(), &[0], &cpu()).unwrap());
    let logits = Arc::new(TrackedTensor::from_var(&logits_var).unwrap());
    let targets = Arc::new(TrackedTensor::from_var(&targets_var).unwrap());

    // cross_entropy_loss may reject empty tensors at the forward level
    // (e.g., shape validation). Either outcome is acceptable:
    // - Err: forward rejects the degenerate input
    // - Ok + backward: no Inf in gradients (the n==0 guard in backward)
    match logits.cross_entropy_loss(&targets, 1) {
        Err(_) => {} // Forward rejection of empty batch is acceptable
        Ok(loss) => {
            let grads = backward(&loss).unwrap();
            let g = grads.get(&logits_var);
            assert!(
                g.is_none() || g.unwrap().numel() == 0,
                "empty batch should produce no gradient"
            );
        }
    }
}

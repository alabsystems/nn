// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Extended tests for loss functions and gradient computation in nn-autodiff.
//!
//! Covers:
//! - MSE loss: gradient correctness, zero-loss, large tensors, symmetry, scaling
//! - Cross-entropy loss: uniform logits, one-hot, batch gradient, numerical stability
//! - Huber loss: delta sweep, negative inputs, symmetry, gradient continuity
//! - L1 loss: large differences, all-equal, sign consistency
//! - Gradient accumulation: multi-pass, weighted combination, independent variables
//! - Gradient clipping: norm-based clipping via manual implementation
//! - Composite losses: weighted sum, multi-term, chain through activations
//! - Finite-difference verification for all loss types

use crate::{backward, TrackedTensor, Var};
use nn_core::dyn_tensor::DynTensor;
use nn_core::Device;
use std::sync::Arc;

fn cpu() -> Device {
    Device::Cpu
}

fn var_from(data: Vec<f32>, shape: &[usize]) -> Var {
    Var::new(DynTensor::from_vec(data, shape, &cpu()).unwrap())
}

fn tracked_from_var(v: &Var) -> Arc<TrackedTensor> {
    Arc::new(TrackedTensor::from_var(v).unwrap())
}

fn const_tensor(data: Vec<f32>, shape: &[usize]) -> Arc<TrackedTensor> {
    Arc::new(TrackedTensor::from_tensor(
        DynTensor::from_vec(data, shape, &cpu()).unwrap(),
    ))
}

/// Cross-entropy targets must be class indices in a U32 tensor (they index
/// into log-softmax via `gather`).
fn u32_target(data: Vec<u32>, shape: &[usize]) -> Arc<TrackedTensor> {
    Arc::new(TrackedTensor::from_tensor(
        DynTensor::from_vec_u32(data, shape, &cpu()).unwrap(),
    ))
}

/// Scalar loss from an arbitrary-shaped tracked tensor by sum over all dims.
fn to_scalar_loss(t: &Arc<TrackedTensor>) -> Arc<TrackedTensor> {
    let mut result = Arc::clone(t);
    for d in (0..result.tensor().rank()).rev() {
        result = result.sum_keepdim(d).unwrap();
    }
    result
}

// ===========================================================================
// 1. MSE loss gradient correctness
// ===========================================================================

#[test]
fn test_mse_grad_single_element() {
    let var = var_from(vec![7.0], &[1]);
    let tgt = const_tensor(vec![3.0], &[1]);
    let x = tracked_from_var(&var);
    let loss = x.mse_loss(&tgt).unwrap();
    let grads = backward(&loss).unwrap();
    let g = grads.get(&var).unwrap().to_flat_vec::<f32>().unwrap();
    // MSE = (7-3)^2 = 16, grad = 2*(7-3)/1 = 8
    assert!((g[0] - 8.0).abs() < 1e-5, "got {}", g[0]);
}

#[test]
fn test_mse_grad_zero_loss_has_zero_gradient() {
    let var = var_from(vec![1.0, 2.0, 3.0], &[3]);
    let tgt = const_tensor(vec![1.0, 2.0, 3.0], &[3]);
    let x = tracked_from_var(&var);
    let loss = x.mse_loss(&tgt).unwrap();
    let val = loss.tensor().to_flat_vec::<f32>().unwrap()[0];
    assert!(val.abs() < 1e-7, "loss should be 0, got {val}");
    let grads = backward(&loss).unwrap();
    let g = grads.get(&var).unwrap().to_flat_vec::<f32>().unwrap();
    for (i, gi) in g.iter().enumerate() {
        assert!(gi.abs() < 1e-6, "grad[{i}] should be 0, got {gi}");
    }
}

#[test]
fn test_mse_grad_negative_values() {
    let pred = vec![-2.0, -1.0, 0.0, 1.0];
    let tgt_vals = vec![2.0, 1.0, 0.0, -1.0];
    let n = pred.len() as f32;
    let var = var_from(pred.clone(), &[4]);
    let tgt = const_tensor(tgt_vals.clone(), &[4]);
    let x = tracked_from_var(&var);
    let loss = x.mse_loss(&tgt).unwrap();
    let grads = backward(&loss).unwrap();
    let g = grads.get(&var).unwrap().to_flat_vec::<f32>().unwrap();
    for i in 0..4 {
        let expected = 2.0 * (pred[i] - tgt_vals[i]) / n;
        assert!(
            (g[i] - expected).abs() < 1e-5,
            "grad[{i}]: expected {expected}, got {}",
            g[i]
        );
    }
}

#[test]
fn test_mse_grad_large_tensor_100_elements() {
    let n = 100;
    let pred: Vec<f32> = (0..n).map(|i| i as f32 * 0.1).collect();
    let tgt_vals: Vec<f32> = (0..n).map(|i| (n - i) as f32 * 0.1).collect();
    let var = var_from(pred.clone(), &[n]);
    let tgt = const_tensor(tgt_vals.clone(), &[n]);
    let x = tracked_from_var(&var);
    let loss = x.mse_loss(&tgt).unwrap();
    let grads = backward(&loss).unwrap();
    let g = grads.get(&var).unwrap().to_flat_vec::<f32>().unwrap();
    assert_eq!(g.len(), n);
    let nf = n as f32;
    for i in 0..n {
        let expected = 2.0 * (pred[i] - tgt_vals[i]) / nf;
        assert!(
            (g[i] - expected).abs() < 1e-4,
            "grad[{i}]: expected {expected}, got {}",
            g[i]
        );
    }
}

#[test]
fn test_mse_loss_symmetry_pred_target_swap() {
    // MSE(a, b) == MSE(b, a)
    let a_data = vec![1.0, 3.0, 5.0];
    let b_data = vec![2.0, 4.0, 6.0];
    let a = const_tensor(a_data, &[3]);
    let b = const_tensor(b_data, &[3]);
    let loss_ab = a.mse_loss(&b).unwrap();
    let loss_ba = b.mse_loss(&a).unwrap();
    let v_ab = loss_ab.tensor().to_flat_vec::<f32>().unwrap()[0];
    let v_ba = loss_ba.tensor().to_flat_vec::<f32>().unwrap()[0];
    assert!(
        (v_ab - v_ba).abs() < 1e-6,
        "MSE should be symmetric: {v_ab} vs {v_ba}"
    );
}

#[test]
fn test_mse_grad_scaling_with_constant_offset() {
    // If target = pred + c for all elements, grad[i] = 2*(-c)/n = -2c/n
    let c = 3.0_f32;
    let pred = vec![0.0, 1.0, 2.0, 3.0, 4.0];
    let tgt_vals: Vec<f32> = pred.iter().map(|p| p + c).collect();
    let n = pred.len() as f32;
    let var = var_from(pred, &[5]);
    let tgt = const_tensor(tgt_vals, &[5]);
    let x = tracked_from_var(&var);
    let loss = x.mse_loss(&tgt).unwrap();
    let grads = backward(&loss).unwrap();
    let g = grads.get(&var).unwrap().to_flat_vec::<f32>().unwrap();
    let expected_grad = -2.0 * c / n;
    for (i, gi) in g.iter().enumerate() {
        assert!(
            (gi - expected_grad).abs() < 1e-5,
            "grad[{i}]: expected {expected_grad}, got {gi}"
        );
    }
}

#[test]
fn test_mse_3d_tensor_gradient() {
    let pred = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let tgt_vals = vec![0.0; 8];
    let n = 8.0_f32;
    let var = var_from(pred.clone(), &[2, 2, 2]);
    let tgt = const_tensor(tgt_vals, &[2, 2, 2]);
    let x = tracked_from_var(&var);
    let loss = x.mse_loss(&tgt).unwrap();
    let grads = backward(&loss).unwrap();
    let g = grads.get(&var).unwrap().to_flat_vec::<f32>().unwrap();
    for i in 0..8 {
        let expected = 2.0 * pred[i] / n;
        assert!(
            (g[i] - expected).abs() < 1e-5,
            "3D MSE grad[{i}]: expected {expected}, got {}",
            g[i]
        );
    }
}

// ===========================================================================
// 2. Cross-entropy loss
// ===========================================================================

#[test]
fn test_ce_uniform_logits_maximum_entropy() {
    // Uniform logits -> max entropy -> loss = log(num_classes)
    let num_classes = 5;
    let logits = vec![0.0; num_classes];
    let input = const_tensor(logits, &[1, num_classes]);
    let target = u32_target(vec![0], &[1, 1]);
    let loss = input.cross_entropy_loss(&target, 1).unwrap();
    let val = loss.tensor().to_flat_vec::<f32>().unwrap()[0];
    let expected = (num_classes as f32).ln();
    assert!(
        (val - expected).abs() < 1e-4,
        "uniform CE = {val}, expected {expected}"
    );
}

#[test]
fn test_ce_confident_prediction_low_loss() {
    // Very confident correct prediction -> loss near 0
    let logits = vec![100.0, 0.0, 0.0];
    let input = const_tensor(logits, &[1, 3]);
    let target = u32_target(vec![0], &[1, 1]);
    let loss = input.cross_entropy_loss(&target, 1).unwrap();
    let val = loss.tensor().to_flat_vec::<f32>().unwrap()[0];
    assert!(
        val < 1e-3,
        "confident prediction should have near-zero loss, got {val}"
    );
}

#[test]
fn test_ce_confident_wrong_prediction_high_loss() {
    // Very confident wrong prediction -> high loss
    let logits = vec![0.0, 0.0, 100.0];
    let input = const_tensor(logits, &[1, 3]);
    let target = u32_target(vec![0], &[1, 1]); // correct class is 0
    let loss = input.cross_entropy_loss(&target, 1).unwrap();
    let val = loss.tensor().to_flat_vec::<f32>().unwrap()[0];
    assert!(
        val > 10.0,
        "wrong confident prediction should have high loss, got {val}"
    );
}

#[test]
fn test_ce_gradient_pushes_toward_correct_class() {
    // Gradient for correct class should be negative (increase logit),
    // gradient for wrong classes should be positive (decrease logit)
    let logits = vec![1.0, 1.0, 1.0];
    let var = var_from(logits, &[1, 3]);
    let target = u32_target(vec![1], &[1, 1]); // correct class = 1
    let x = tracked_from_var(&var);
    let loss = x.cross_entropy_loss(&target, 1).unwrap();
    let grads = backward(&loss).unwrap();
    let g = grads.get(&var).unwrap().to_flat_vec::<f32>().unwrap();
    // Correct class gradient should be negative (push logit up)
    assert!(
        g[1] < 0.0,
        "correct class grad should be negative, got {}",
        g[1]
    );
    // Wrong class gradients should be positive (push logits down)
    assert!(
        g[0] > 0.0,
        "wrong class grad[0] should be positive, got {}",
        g[0]
    );
    assert!(
        g[2] > 0.0,
        "wrong class grad[2] should be positive, got {}",
        g[2]
    );
}

#[test]
fn test_ce_gradient_sum_is_zero() {
    // For softmax-based CE, gradients over classes sum to ~0 for each sample
    let logits = vec![2.0, 1.0, 0.5, -1.0];
    let var = var_from(logits, &[1, 4]);
    let target = u32_target(vec![2], &[1, 1]);
    let x = tracked_from_var(&var);
    let loss = x.cross_entropy_loss(&target, 1).unwrap();
    let grads = backward(&loss).unwrap();
    let g = grads.get(&var).unwrap().to_flat_vec::<f32>().unwrap();
    let grad_sum: f32 = g.iter().sum();
    assert!(
        grad_sum.abs() < 1e-4,
        "CE gradients should sum to ~0, got {grad_sum}"
    );
}

#[test]
fn test_ce_batch_of_3_samples() {
    let logits = vec![
        2.0, 0.0, 0.0, // sample 0: class 0 correct
        0.0, 2.0, 0.0, // sample 1: class 1 correct
        0.0, 0.0, 2.0, // sample 2: class 2 correct
    ];
    let input = const_tensor(logits, &[3, 3]);
    let targets = u32_target(vec![0, 1, 2], &[3, 1]);
    let loss = input.cross_entropy_loss(&targets, 1).unwrap();
    let val = loss.tensor().to_flat_vec::<f32>().unwrap()[0];
    // All samples have correct class with highest logit -> moderate loss
    assert!(
        val > 0.0 && val < 2.0,
        "batch CE should be moderate, got {val}"
    );
}

// ===========================================================================
// 3. L1 loss edge cases
// ===========================================================================

#[test]
fn test_l1_large_differences() {
    let pred = vec![1000.0, -1000.0];
    let tgt_vals = vec![-1000.0, 1000.0];
    let input = const_tensor(pred, &[2]);
    let target = const_tensor(tgt_vals, &[2]);
    let loss = input.l1_loss(&target).unwrap();
    let val = loss.tensor().to_flat_vec::<f32>().unwrap()[0];
    // mean(|2000| + |2000|) / 2 = 2000
    assert!(
        (val - 2000.0).abs() < 1.0,
        "L1 large diff = {val}, expected 2000"
    );
}

#[test]
fn test_l1_all_equal_zero_loss() {
    let vals = vec![42.0, 42.0, 42.0, 42.0];
    let input = const_tensor(vals.clone(), &[4]);
    let target = const_tensor(vals, &[4]);
    let loss = input.l1_loss(&target).unwrap();
    let val = loss.tensor().to_flat_vec::<f32>().unwrap()[0];
    assert!(val.abs() < 1e-7, "L1 equal should be 0, got {val}");
}

#[test]
fn test_l1_gradient_magnitude_uniform() {
    // L1 gradient magnitude should be 1/n for all elements (when diff != 0)
    let pred = vec![10.0, 20.0, 30.0, 40.0, 50.0];
    let tgt_vals = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let n = pred.len() as f32;
    let var = var_from(pred, &[5]);
    let tgt = const_tensor(tgt_vals, &[5]);
    let x = tracked_from_var(&var);
    let loss = x.l1_loss(&tgt).unwrap();
    let grads = backward(&loss).unwrap();
    let g = grads.get(&var).unwrap().to_flat_vec::<f32>().unwrap();
    for (i, gi) in g.iter().enumerate() {
        assert!(
            (gi.abs() - 1.0 / n).abs() < 1e-5,
            "L1 |grad[{i}]| = {}, expected {}",
            gi.abs(),
            1.0 / n
        );
    }
}

#[test]
fn test_l1_2d_tensor() {
    let pred = vec![1.0, 3.0, 5.0, 7.0];
    let tgt_vals = vec![2.0, 2.0, 2.0, 2.0];
    let input = const_tensor(pred, &[2, 2]);
    let target = const_tensor(tgt_vals, &[2, 2]);
    let loss = input.l1_loss(&target).unwrap();
    let val = loss.tensor().to_flat_vec::<f32>().unwrap()[0];
    // mean(|1-2|+|3-2|+|5-2|+|7-2|) = mean(1+1+3+5) = 10/4 = 2.5
    assert!((val - 2.5).abs() < 1e-5, "L1 2D = {val}, expected 2.5");
}

// ===========================================================================
// 4. Huber loss boundary cases
// ===========================================================================

#[test]
fn test_huber_delta_sweep_monotonicity() {
    // As delta increases, more values fall in quadratic region, loss decreases
    let pred = vec![5.0, 10.0, -3.0];
    let tgt_vals = vec![1.0, 2.0, 1.0];
    let mut prev_loss = f32::MAX;
    for delta_int in &[1, 2, 5, 10, 50, 100] {
        let delta = f64::from(*delta_int);
        let input = const_tensor(pred.clone(), &[3]);
        let target = const_tensor(tgt_vals.clone(), &[3]);
        let loss = input.huber_loss(&target, delta).unwrap();
        let val = loss.tensor().to_flat_vec::<f32>().unwrap()[0];
        assert!(
            val <= prev_loss + 1e-5,
            "Huber(delta={delta}) = {val} should be <= prev {prev_loss}"
        );
        prev_loss = val;
    }
}

#[test]
fn test_huber_negative_differences() {
    let pred = vec![-5.0, -10.0];
    let tgt_vals = vec![0.0, 0.0];
    let delta = 2.0;
    let input = const_tensor(pred, &[2]);
    let target = const_tensor(tgt_vals, &[2]);
    let loss = input.huber_loss(&target, delta).unwrap();
    let val = loss.tensor().to_flat_vec::<f32>().unwrap()[0];
    // diffs = [-5, -10], |diff| = [5, 10] > delta=2
    // linear: (5 - 1) + (10 - 1) = 4 + 9 = 13, mean = 6.5
    let expected = f32::midpoint(4.0, 9.0);
    assert!(
        (val - expected).abs() < 1e-4,
        "Huber negative = {val}, expected {expected}"
    );
}

#[test]
fn test_huber_symmetry_pred_target_swap() {
    let a = vec![3.0, 7.0, -1.0];
    let b = vec![1.0, 2.0, 4.0];
    let delta = 2.0;
    let loss_ab = const_tensor(a.clone(), &[3])
        .huber_loss(&const_tensor(b.clone(), &[3]), delta)
        .unwrap();
    let loss_ba = const_tensor(b, &[3])
        .huber_loss(&const_tensor(a, &[3]), delta)
        .unwrap();
    let v_ab = loss_ab.tensor().to_flat_vec::<f32>().unwrap()[0];
    let v_ba = loss_ba.tensor().to_flat_vec::<f32>().unwrap()[0];
    assert!(
        (v_ab - v_ba).abs() < 1e-5,
        "Huber should be symmetric: {v_ab} vs {v_ba}"
    );
}

#[test]
fn test_huber_large_delta_approximates_mse() {
    // When delta >> |diff|, Huber ~ 0.5 * diff^2 / delta
    let pred = vec![1.0, 2.0, 3.0];
    let tgt_vals = vec![1.5, 2.5, 3.5];
    let delta = 1000.0;
    let input_h = const_tensor(pred, &[3]);
    let target_h = const_tensor(tgt_vals, &[3]);
    let huber = input_h.huber_loss(&target_h, delta).unwrap();
    let huber_val = huber.tensor().to_flat_vec::<f32>().unwrap()[0];
    // Quadratic: 0.5 * diff^2 / delta = 0.5 * 0.25 / 1000 = 0.000125 per elem
    let expected = 0.5 * 0.25 / delta as f32;
    assert!(
        (huber_val - expected).abs() < 1e-5,
        "Huber(large delta) = {huber_val}, expected ~{expected}"
    );
}

#[test]
fn test_huber_gradient_quadratic_region_analytical() {
    // In quadratic region: grad = diff / (n * delta)
    let delta = 5.0;
    let pred = vec![1.0, 2.0]; // diffs = [0.5, 0.5], |diff| < delta
    let tgt_vals = vec![0.5, 1.5];
    let n = 2.0_f32;
    let var = var_from(pred.clone(), &[2]);
    let tgt = const_tensor(tgt_vals.clone(), &[2]);
    let x = tracked_from_var(&var);
    let loss = x.huber_loss(&tgt, delta).unwrap();
    let grads = backward(&loss).unwrap();
    let g = grads.get(&var).unwrap().to_flat_vec::<f32>().unwrap();
    for i in 0..2 {
        let diff = pred[i] - tgt_vals[i];
        let expected = diff / (delta as f32 * n);
        assert!(
            (g[i] - expected).abs() < 1e-4,
            "Huber quad grad[{i}]: expected {expected}, got {}",
            g[i]
        );
    }
}

// ===========================================================================
// 5. Gradient accumulation patterns
// ===========================================================================

#[test]
fn test_grad_accumulation_5_passes() {
    let var = var_from(vec![4.0], &[1]);
    let mut total_grad = 0.0_f32;
    for _ in 0..5 {
        let x = tracked_from_var(&var);
        let loss = x.sqr().unwrap(); // grad = 2*4 = 8
        let grads = backward(&loss).unwrap();
        total_grad += grads.get(&var).unwrap().to_flat_vec::<f32>().unwrap()[0];
    }
    assert!(
        (total_grad - 40.0).abs() < 1e-4,
        "accumulated = {total_grad}, expected 40.0"
    );
}

#[test]
fn test_grad_accumulation_different_loss_functions_three_types() {
    // Accumulate MSE + L1 + Huber grads
    let var = var_from(vec![5.0], &[1]);
    let target = const_tensor(vec![2.0], &[1]);

    // MSE grad: 2*(5-2)/1 = 6
    let x1 = tracked_from_var(&var);
    let mse = x1.mse_loss(&target).unwrap();
    let g1 = backward(&mse)
        .unwrap()
        .get(&var)
        .unwrap()
        .to_flat_vec::<f32>()
        .unwrap()[0];

    // L1 grad: sign(5-2)/1 = 1
    let x2 = tracked_from_var(&var);
    let l1 = x2.l1_loss(&target).unwrap();
    let g2 = backward(&l1)
        .unwrap()
        .get(&var)
        .unwrap()
        .to_flat_vec::<f32>()
        .unwrap()[0];

    // Huber grad (delta=1.0, diff=3>delta): sign(3)/1 = 1
    let x3 = tracked_from_var(&var);
    let huber = x3.huber_loss(&target, 1.0).unwrap();
    let g3 = backward(&huber)
        .unwrap()
        .get(&var)
        .unwrap()
        .to_flat_vec::<f32>()
        .unwrap()[0];

    let total = g1 + g2 + g3;
    // 6 + 1 + 1 = 8
    assert!(
        (total - 8.0).abs() < 1e-4,
        "total grad = {total}, expected 8.0 (MSE={g1}, L1={g2}, Huber={g3})"
    );
}

#[test]
fn test_grad_accumulation_independent_vars() {
    // Two independent variables, each getting their own gradient
    let a = var_from(vec![3.0], &[1]);
    let b = var_from(vec![7.0], &[1]);

    let xa = tracked_from_var(&a);
    let xb = tracked_from_var(&b);
    let loss_a = xa.sqr().unwrap(); // loss = 9, grad_a = 6
    let loss_b = xb.sqr().unwrap(); // loss = 49, grad_b = 14
    let total = loss_a.add(&loss_b).unwrap();
    let grads = backward(&total).unwrap();

    let ga = grads.get(&a).unwrap().to_flat_vec::<f32>().unwrap()[0];
    let gb = grads.get(&b).unwrap().to_flat_vec::<f32>().unwrap()[0];
    assert!((ga - 6.0).abs() < 1e-4, "grad_a = {ga}, expected 6.0");
    assert!((gb - 14.0).abs() < 1e-4, "grad_b = {gb}, expected 14.0");
}

#[test]
fn test_grad_accumulation_shared_variable_two_paths() {
    // Same variable used in two different computations -> gradients accumulate
    let var = var_from(vec![2.0], &[1]);

    let x1 = tracked_from_var(&var);
    let x2 = tracked_from_var(&var);
    let sq = x1.sqr().unwrap(); // grad1 = 4
    let cube_approx = x2.mul_scalar(3.0).unwrap(); // grad2 = 3
    let total = sq.add(&cube_approx).unwrap();
    let grads = backward(&total).unwrap();
    let g = grads.get(&var).unwrap().to_flat_vec::<f32>().unwrap()[0];
    // Note: x1 and x2 are independent tracked tensors from same var,
    // but backward() only traverses the computation graph from `total`.
    // x1's path contributes grad=4, x2's path contributes grad=3
    assert!(
        (g - 7.0).abs() < 1e-4,
        "shared var grad = {g}, expected 7.0"
    );
}

// ===========================================================================
// 6. Gradient clipping (manual norm-based)
// ===========================================================================

#[test]
fn test_gradient_clipping_by_norm() {
    // Simulate gradient clipping: if ||grad|| > max_norm, scale down
    let var = var_from(vec![100.0, 200.0], &[2]);
    let target = const_tensor(vec![0.0, 0.0], &[2]);
    let x = tracked_from_var(&var);
    let loss = x.mse_loss(&target).unwrap();
    let grads = backward(&loss).unwrap();
    let g = grads.get(&var).unwrap().to_flat_vec::<f32>().unwrap();

    let grad_norm: f32 = g.iter().map(|gi| gi * gi).sum::<f32>().sqrt();
    let max_norm = 1.0_f32;
    let clipped: Vec<f32> = if grad_norm > max_norm {
        let scale = max_norm / grad_norm;
        g.iter().map(|gi| gi * scale).collect()
    } else {
        g.clone()
    };

    let clipped_norm: f32 = clipped.iter().map(|c| c * c).sum::<f32>().sqrt();
    assert!(
        (clipped_norm - max_norm).abs() < 1e-4,
        "clipped norm = {clipped_norm}, expected {max_norm}"
    );
    // Direction should be preserved
    assert!(clipped[0] > 0.0 && clipped[1] > 0.0, "direction preserved");
    assert!(
        (clipped[1] / clipped[0] - g[1] / g[0]).abs() < 1e-4,
        "ratio should be preserved"
    );
}

#[test]
fn test_gradient_clipping_no_clip_needed() {
    // Small gradient -> no clipping applied
    let var = var_from(vec![0.001], &[1]);
    let target = const_tensor(vec![0.0], &[1]);
    let x = tracked_from_var(&var);
    let loss = x.mse_loss(&target).unwrap();
    let grads = backward(&loss).unwrap();
    let g = grads.get(&var).unwrap().to_flat_vec::<f32>().unwrap();

    let max_norm = 10.0_f32;
    let grad_norm = g[0].abs();
    assert!(
        grad_norm < max_norm,
        "gradient norm {grad_norm} should be < max_norm {max_norm}"
    );
}

#[test]
fn test_gradient_clipping_by_value() {
    // Clip each gradient element to [-clip_val, clip_val]
    let var = var_from(vec![50.0, -50.0, 0.5], &[3]);
    let target = const_tensor(vec![0.0, 0.0, 0.0], &[3]);
    let x = tracked_from_var(&var);
    let loss = x.mse_loss(&target).unwrap();
    let grads = backward(&loss).unwrap();
    let g = grads.get(&var).unwrap().to_flat_vec::<f32>().unwrap();

    let clip_val = 5.0_f32;
    let clipped: Vec<f32> = g.iter().map(|gi| gi.clamp(-clip_val, clip_val)).collect();

    for c in &clipped {
        assert!(
            *c >= -clip_val && *c <= clip_val,
            "clipped value {c} out of bounds"
        );
    }
}

// ===========================================================================
// 7. Composite losses
// ===========================================================================

#[test]
fn test_weighted_mse_plus_l1_loss() {
    // Composite: 0.7 * MSE + 0.3 * L1
    let var = var_from(vec![4.0, 6.0], &[2]);
    let target = const_tensor(vec![2.0, 2.0], &[2]);

    let x = tracked_from_var(&var);
    let mse = x.mse_loss(&target).unwrap();
    let mse_weighted = mse.mul_scalar(0.7).unwrap();

    // Need a new tracked tensor for L1 (same var)
    let x2 = tracked_from_var(&var);
    let l1 = x2.l1_loss(&target).unwrap();
    let l1_weighted = l1.mul_scalar(0.3).unwrap();

    let total = mse_weighted.add(&l1_weighted).unwrap();
    let grads = backward(&total).unwrap();
    let g = grads.get(&var).unwrap().to_flat_vec::<f32>().unwrap();

    // Should have non-zero gradient combining both loss signals
    assert!(g[0].abs() > 1e-6, "composite grad[0] should be non-zero");
    assert!(g[1].abs() > 1e-6, "composite grad[1] should be non-zero");
    // Both elements are above target -> both grads positive
    assert!(g[0] > 0.0, "grad[0] should be positive");
    assert!(g[1] > 0.0, "grad[1] should be positive");
}

#[test]
fn test_loss_through_activation_relu() {
    // MSE after ReLU: loss = (relu(x) - target)^2
    let var = var_from(vec![-1.0, 0.5, 2.0], &[3]);
    let target = const_tensor(vec![0.0, 0.0, 0.0], &[3]);
    let x = tracked_from_var(&var);
    let activated = x.relu().unwrap();
    let loss = activated.mse_loss(&target).unwrap();
    let grads = backward(&loss).unwrap();
    let g = grads.get(&var).unwrap().to_flat_vec::<f32>().unwrap();
    // x=-1 -> relu=0 -> grad=0 (killed by relu)
    assert!(g[0].abs() < 1e-6, "relu kills negative grad, got {}", g[0]);
    // x=0.5 -> relu=0.5 -> grad through
    assert!(g[1] > 0.0, "positive x should have positive grad");
    // x=2.0 -> relu=2.0 -> grad through
    assert!(g[2] > 0.0, "positive x should have positive grad");
}

#[test]
fn test_loss_through_activation_sigmoid() {
    // MSE after sigmoid: loss = (sigmoid(x) - target)^2
    let var = var_from(vec![0.0, 5.0, -5.0], &[3]);
    let target = const_tensor(vec![0.5, 0.5, 0.5], &[3]);
    let x = tracked_from_var(&var);
    let activated = x.sigmoid().unwrap();
    let loss = activated.mse_loss(&target).unwrap();
    let grads = backward(&loss).unwrap();
    let g = grads.get(&var).unwrap().to_flat_vec::<f32>().unwrap();
    // x=0 -> sigmoid(0)=0.5 -> loss=0 -> grad~0
    assert!(
        g[0].abs() < 1e-4,
        "sigmoid(0)=target -> grad~0, got {}",
        g[0]
    );
}

#[test]
fn test_loss_through_multiple_ops_chain() {
    // loss = MSE(tanh(relu(x * 2 + 1)), target)
    let var = var_from(vec![0.5, 1.0, -0.5], &[3]);
    let target = const_tensor(vec![0.0, 0.0, 0.0], &[3]);
    let x = tracked_from_var(&var);
    let scaled = x.mul_scalar(2.0).unwrap();
    let shifted = scaled.add_scalar(1.0).unwrap();
    let activated = shifted.relu().unwrap();
    let squashed = activated.tanh().unwrap();
    let loss = squashed.mse_loss(&target).unwrap();
    let grads = backward(&loss).unwrap();
    let g = grads.get(&var).unwrap().to_flat_vec::<f32>().unwrap();
    // All gradients should be finite
    for (i, gi) in g.iter().enumerate() {
        assert!(gi.is_finite(), "grad[{i}] should be finite, got {gi}");
    }
}

// ===========================================================================
// 8. Numerical stability and edge cases
// ===========================================================================

#[test]
fn test_mse_very_small_values() {
    let pred = vec![1e-7, 2e-7, 3e-7];
    let tgt_vals = vec![0.0, 0.0, 0.0];
    let input = const_tensor(pred, &[3]);
    let target = const_tensor(tgt_vals, &[3]);
    let loss = input.mse_loss(&target).unwrap();
    let val = loss.tensor().to_flat_vec::<f32>().unwrap()[0];
    assert!(val >= 0.0, "MSE should be non-negative, got {val}");
    assert!(val.is_finite(), "MSE should be finite, got {val}");
    assert!(val < 1e-10, "MSE of tiny values should be tiny, got {val}");
}

#[test]
fn test_mse_loss_always_non_negative() {
    // Verify MSE >= 0 for various inputs
    let test_cases: Vec<(Vec<f32>, Vec<f32>)> = vec![
        (vec![1.0, 2.0], vec![3.0, 4.0]),
        (vec![-1.0, -2.0], vec![-3.0, -4.0]),
        (vec![0.0, 0.0], vec![0.0, 0.0]),
        (vec![100.0], vec![-100.0]),
        (vec![0.001, -0.001], vec![-0.001, 0.001]),
    ];
    for (pred, tgt) in test_cases {
        let n = pred.len();
        let input = const_tensor(pred, &[n]);
        let target = const_tensor(tgt, &[n]);
        let loss = input.mse_loss(&target).unwrap();
        let val = loss.tensor().to_flat_vec::<f32>().unwrap()[0];
        assert!(val >= 0.0, "MSE should be >= 0, got {val}");
    }
}

#[test]
fn test_l1_loss_always_non_negative() {
    let test_cases: Vec<(Vec<f32>, Vec<f32>)> = vec![
        (vec![1.0, 2.0], vec![3.0, 4.0]),
        (vec![-5.0], vec![5.0]),
        (vec![0.0], vec![0.0]),
    ];
    for (pred, tgt) in test_cases {
        let n = pred.len();
        let input = const_tensor(pred, &[n]);
        let target = const_tensor(tgt, &[n]);
        let loss = input.l1_loss(&target).unwrap();
        let val = loss.tensor().to_flat_vec::<f32>().unwrap()[0];
        assert!(val >= 0.0, "L1 should be >= 0, got {val}");
    }
}

#[test]
fn test_huber_loss_always_non_negative() {
    let test_cases: Vec<(Vec<f32>, Vec<f32>)> = vec![
        (vec![1.0], vec![5.0]),
        (vec![-3.0], vec![3.0]),
        (vec![0.0], vec![0.0]),
    ];
    for (pred, tgt) in test_cases {
        let n = pred.len();
        let input = const_tensor(pred, &[n]);
        let target = const_tensor(tgt, &[n]);
        let loss = input.huber_loss(&target, 1.0).unwrap();
        let val = loss.tensor().to_flat_vec::<f32>().unwrap()[0];
        assert!(val >= -1e-7, "Huber should be >= 0, got {val}");
    }
}

// ===========================================================================
// 9. Finite-difference gradient verification
// ===========================================================================

/// Central-difference gradient check for a loss function (default step).
fn fd_check<F>(vals: &[f32], targets: &[f32], loss_fn: F, tol: f32)
where
    F: Fn(&Arc<TrackedTensor>, &Arc<TrackedTensor>) -> crate::error::Result<Arc<TrackedTensor>>,
{
    fd_check_eps(vals, targets, loss_fn, tol, 1e-3_f32);
}

/// Central-difference gradient check with an explicit step size.
///
/// The step `eps` must be chosen for the input magnitude: for large-magnitude
/// f32 inputs, a tiny step causes catastrophic cancellation when differencing
/// squared values, so a larger step is required. Central differences are exact
/// for quadratics (e.g. MSE) up to round-off, so a larger step adds no
/// truncation error there.
fn fd_check_eps<F>(vals: &[f32], targets: &[f32], loss_fn: F, tol: f32, eps: f32)
where
    F: Fn(&Arc<TrackedTensor>, &Arc<TrackedTensor>) -> crate::error::Result<Arc<TrackedTensor>>,
{
    let var = var_from(vals.to_vec(), &[vals.len()]);
    let target = const_tensor(targets.to_vec(), &[targets.len()]);

    let x = tracked_from_var(&var);
    let loss = loss_fn(&x, &target).unwrap();
    let grads = backward(&loss).unwrap();
    let grad = grads.get(&var).unwrap().to_flat_vec::<f32>().unwrap();

    for i in 0..vals.len() {
        let mut v_plus = vals.to_vec();
        v_plus[i] += eps;
        let t_plus = const_tensor(v_plus, &[vals.len()]);
        let l_plus = loss_fn(&t_plus, &target)
            .unwrap()
            .tensor()
            .to_flat_vec::<f32>()
            .unwrap()[0];

        let mut v_minus = vals.to_vec();
        v_minus[i] -= eps;
        let t_minus = const_tensor(v_minus, &[vals.len()]);
        let l_minus = loss_fn(&t_minus, &target)
            .unwrap()
            .tensor()
            .to_flat_vec::<f32>()
            .unwrap()[0];

        let fd = (l_plus - l_minus) / (2.0 * eps);
        let err = (grad[i] - fd).abs();
        assert!(
            err < tol,
            "FD[{i}]: analytical={:.6}, numerical={:.6}, err={:.6}",
            grad[i],
            fd,
            err
        );
    }
}

#[test]
fn test_fd_mse_loss_various_inputs() {
    fd_check(
        &[3.0, 1.0, -2.0, 7.0],
        &[1.0, 2.0, 3.0, 4.0],
        TrackedTensor::mse_loss,
        1e-3,
    );
}

#[test]
fn test_fd_mse_loss_close_values() {
    fd_check(
        &[1.01, 2.01, 3.01],
        &[1.0, 2.0, 3.0],
        TrackedTensor::mse_loss,
        1e-3,
    );
}

#[test]
fn test_fd_l1_loss_well_separated() {
    // Avoid values where pred==target (non-differentiable point)
    fd_check(
        &[5.0, -3.0, 10.0, -7.0],
        &[1.0, 1.0, 1.0, 1.0],
        TrackedTensor::l1_loss,
        1e-3,
    );
}

#[test]
fn test_fd_huber_loss_quadratic_region() {
    fd_check(
        &[1.1, 2.1, 3.1],
        &[1.0, 2.0, 3.0],
        |x, t| x.huber_loss(t, 5.0),
        1e-3,
    );
}

#[test]
fn test_fd_huber_loss_linear_region() {
    fd_check(
        &[10.0, -10.0, 20.0],
        &[1.0, 1.0, 1.0],
        |x, t| x.huber_loss(t, 0.5),
        1e-3,
    );
}

#[test]
fn test_fd_mse_large_range() {
    // Large-magnitude f32 inputs: a 1e-3 step causes catastrophic cancellation
    // when differencing squared values (~12100 ± 0.2 in f32). MSE is quadratic
    // so central differences are exact up to round-off; a unit step keeps the
    // finite-difference error at the f32 round-off floor (~1e-4).
    fd_check_eps(
        &[-100.0, -50.0, 0.0, 50.0, 100.0],
        &[10.0, 20.0, 30.0, 40.0, 50.0],
        TrackedTensor::mse_loss,
        1e-2,
        1.0,
    );
}

// ===========================================================================
// 10. Training convergence with different losses
// ===========================================================================

#[test]
fn test_sgd_mse_converges_2d_variable() {
    // Minimize (a-1)^2 + (b-2)^2 using MSE on [a,b] vs [1,2]
    let var = var_from(vec![10.0, -5.0], &[2]);
    let lr = 0.1_f32;
    for _ in 0..100 {
        let x = tracked_from_var(&var);
        let target = const_tensor(vec![1.0, 2.0], &[2]);
        let loss = x.mse_loss(&target).unwrap();
        let grads = backward(&loss).unwrap();
        let g = grads.get(&var).unwrap().to_flat_vec::<f32>().unwrap();
        let data = var.data().unwrap().to_flat_vec::<f32>().unwrap();
        let new: Vec<f32> = data.iter().zip(&g).map(|(d, gi)| d - lr * gi).collect();
        var.set(&DynTensor::from_vec(new, &[2], &cpu()).unwrap())
            .unwrap();
    }
    let final_vals = var.data().unwrap().to_flat_vec::<f32>().unwrap();
    assert!(
        (final_vals[0] - 1.0).abs() < 0.01,
        "a should -> 1, got {}",
        final_vals[0]
    );
    assert!(
        (final_vals[1] - 2.0).abs() < 0.01,
        "b should -> 2, got {}",
        final_vals[1]
    );
}

#[test]
fn test_sgd_l1_converges() {
    let var = var_from(vec![10.0], &[1]);
    let lr = 0.5_f32;
    for _ in 0..30 {
        let x = tracked_from_var(&var);
        let target = const_tensor(vec![3.0], &[1]);
        let loss = x.l1_loss(&target).unwrap();
        let grads = backward(&loss).unwrap();
        let g = grads.get(&var).unwrap().to_flat_vec::<f32>().unwrap();
        let data = var.data().unwrap().to_flat_vec::<f32>().unwrap();
        let new: Vec<f32> = data.iter().zip(&g).map(|(d, gi)| d - lr * gi).collect();
        var.set(&DynTensor::from_vec(new, &[1], &cpu()).unwrap())
            .unwrap();
    }
    let final_val = var.data().unwrap().to_flat_vec::<f32>().unwrap()[0];
    assert!(
        (final_val - 3.0).abs() < 1.0,
        "should approach 3.0, got {final_val}"
    );
}

#[test]
fn test_sgd_huber_converges() {
    let var = var_from(vec![20.0], &[1]);
    let lr = 0.3_f32;
    let delta = 1.0;
    for _ in 0..80 {
        let x = tracked_from_var(&var);
        let target = const_tensor(vec![5.0], &[1]);
        let loss = x.huber_loss(&target, delta).unwrap();
        let grads = backward(&loss).unwrap();
        let g = grads.get(&var).unwrap().to_flat_vec::<f32>().unwrap();
        let data = var.data().unwrap().to_flat_vec::<f32>().unwrap();
        let new: Vec<f32> = data.iter().zip(&g).map(|(d, gi)| d - lr * gi).collect();
        var.set(&DynTensor::from_vec(new, &[1], &cpu()).unwrap())
            .unwrap();
    }
    let final_val = var.data().unwrap().to_flat_vec::<f32>().unwrap()[0];
    assert!(
        (final_val - 5.0).abs() < 0.5,
        "should approach 5.0, got {final_val}"
    );
}

// ===========================================================================
// 11. Loss value correctness
// ===========================================================================

#[test]
fn test_mse_known_values_batch() {
    // MSE([1,2,3,4], [5,6,7,8]) = mean(16+16+16+16) = 16
    let input = const_tensor(vec![1.0, 2.0, 3.0, 4.0], &[4]);
    let target = const_tensor(vec![5.0, 6.0, 7.0, 8.0], &[4]);
    let loss = input.mse_loss(&target).unwrap();
    let val = loss.tensor().to_flat_vec::<f32>().unwrap()[0];
    assert!((val - 16.0).abs() < 1e-5, "MSE = {val}, expected 16.0");
}

#[test]
fn test_l1_known_values_batch() {
    // L1([1,2,3,4], [5,6,7,8]) = mean(4+4+4+4) = 4
    let input = const_tensor(vec![1.0, 2.0, 3.0, 4.0], &[4]);
    let target = const_tensor(vec![5.0, 6.0, 7.0, 8.0], &[4]);
    let loss = input.l1_loss(&target).unwrap();
    let val = loss.tensor().to_flat_vec::<f32>().unwrap()[0];
    assert!((val - 4.0).abs() < 1e-5, "L1 = {val}, expected 4.0");
}

#[test]
fn test_huber_all_quadratic_region() {
    // All diffs = 0.1, delta = 1.0 -> all in quadratic region
    // huber = 0.5 * 0.01 / 1.0 = 0.005 per elem
    let input = const_tensor(vec![1.1, 2.1, 3.1], &[3]);
    let target = const_tensor(vec![1.0, 2.0, 3.0], &[3]);
    let loss = input.huber_loss(&target, 1.0).unwrap();
    let val = loss.tensor().to_flat_vec::<f32>().unwrap()[0];
    let expected = 0.5 * 0.01 / 1.0;
    assert!(
        (val - expected).abs() < 1e-5,
        "Huber quad = {val}, expected {expected}"
    );
}

#[test]
fn test_huber_all_linear_region() {
    // All diffs = 10, delta = 1.0 -> all in linear region
    // huber = 10 - 0.5 = 9.5 per elem
    let input = const_tensor(vec![10.0, 10.0], &[2]);
    let target = const_tensor(vec![0.0, 0.0], &[2]);
    let loss = input.huber_loss(&target, 1.0).unwrap();
    let val = loss.tensor().to_flat_vec::<f32>().unwrap()[0];
    let expected = 9.5;
    assert!(
        (val - expected).abs() < 1e-4,
        "Huber linear = {val}, expected {expected}"
    );
}

// ===========================================================================
// 12. Detach and stop-gradient patterns
// ===========================================================================

#[test]
fn test_detach_stops_gradient_flow() {
    let var = var_from(vec![3.0], &[1]);
    let x = tracked_from_var(&var);
    let detached = x.detach();
    let loss = detached.sqr().unwrap();
    let loss_scalar = to_scalar_loss(&loss);
    let grads = backward(&loss_scalar).unwrap();
    // After detach, no gradient should reach var
    assert!(
        grads.get(&var).is_none(),
        "detach should prevent gradient flow"
    );
}

#[test]
fn test_partial_detach_only_some_gradients() {
    // x contributes gradient, y (detached) does not
    let var_x = var_from(vec![2.0], &[1]);
    let var_y = var_from(vec![3.0], &[1]);

    let x = tracked_from_var(&var_x);
    let y = tracked_from_var(&var_y);
    let y_detached = y.detach();
    // loss = (x + detach(y))^2 = (2+3)^2 = 25
    // grad_x = 2*(x+y) = 10, grad_y = None (detached)
    let sum = x.add(&y_detached).unwrap();
    let loss = sum.sqr().unwrap();
    let loss_scalar = to_scalar_loss(&loss);
    let grads = backward(&loss_scalar).unwrap();

    let gx = grads.get(&var_x).unwrap().to_flat_vec::<f32>().unwrap()[0];
    assert!((gx - 10.0).abs() < 1e-4, "grad_x = {gx}, expected 10.0");
    assert!(
        grads.get(&var_y).is_none(),
        "detached y should have no gradient"
    );
}

// ===========================================================================
// 13. Loss scaling for mixed precision
// ===========================================================================

#[test]
fn test_loss_scaling_preserves_gradient_direction() {
    use crate::loss_scaling::{DynamicLossScaler, MixedPrecisionConfig};

    let config = MixedPrecisionConfig::bf16_training();
    let scaler = DynamicLossScaler::new(config).unwrap();

    let loss = DynTensor::from_vec(vec![2.0], &[1], &cpu()).unwrap();
    let scaled = scaler.scale_loss(&loss).unwrap();
    let scaled_val = scaled.to_flat_vec::<f32>().unwrap()[0];

    assert!(scaled_val > 0.0, "scaled loss should be positive");
    assert!(
        scaled_val > 2.0,
        "scaled loss should be larger than original"
    );
}

#[test]
fn test_loss_scaler_growth_after_good_steps() {
    use crate::loss_scaling::{DynamicLossScaler, MixedPrecisionConfig};

    let mut config = MixedPrecisionConfig::bf16_training();
    config.growth_interval = 3; // grow after 3 good steps
    let mut scaler = DynamicLossScaler::new(config).unwrap();
    let initial_scale = scaler.scale_factor();

    // 3 good steps
    scaler.update(false);
    scaler.update(false);
    scaler.update(false);

    let new_scale = scaler.scale_factor();
    assert!(
        new_scale > initial_scale,
        "scale should grow after good steps: {initial_scale} -> {new_scale}"
    );
}

#[test]
fn test_loss_scaler_backoff_on_inf() {
    use crate::loss_scaling::{DynamicLossScaler, MixedPrecisionConfig};

    let config = MixedPrecisionConfig::bf16_training();
    let mut scaler = DynamicLossScaler::new(config).unwrap();
    let initial_scale = scaler.scale_factor();

    scaler.update(true); // inf found

    let new_scale = scaler.scale_factor();
    assert!(
        new_scale < initial_scale,
        "scale should decrease on inf: {initial_scale} -> {new_scale}"
    );
    assert_eq!(scaler.consecutive_good_steps(), 0);
}

// ===========================================================================
// 14. Cross-entropy numerical edge cases
// ===========================================================================

#[test]
fn test_ce_single_class_trivial() {
    // With only 1 class, CE should be ~0 (log(softmax) = log(1) = 0)
    let logits = vec![5.0];
    let input = const_tensor(logits, &[1, 1]);
    let target = u32_target(vec![0], &[1, 1]);
    let loss = input.cross_entropy_loss(&target, 1).unwrap();
    let val = loss.tensor().to_flat_vec::<f32>().unwrap()[0];
    assert!(val.abs() < 1e-4, "single class CE should be ~0, got {val}");
}

#[test]
fn test_ce_two_class_balanced() {
    // Equal logits for 2 classes -> loss = log(2)
    let logits = vec![0.0, 0.0];
    let input = const_tensor(logits, &[1, 2]);
    let target = u32_target(vec![0], &[1, 1]);
    let loss = input.cross_entropy_loss(&target, 1).unwrap();
    let val = loss.tensor().to_flat_vec::<f32>().unwrap()[0];
    let expected = 2.0_f32.ln();
    assert!(
        (val - expected).abs() < 1e-4,
        "balanced 2-class CE = {val}, expected {expected}"
    );
}

#[test]
fn test_ce_10_classes_uniform() {
    let logits = vec![0.0; 10];
    let input = const_tensor(logits, &[1, 10]);
    let target = u32_target(vec![5], &[1, 1]);
    let loss = input.cross_entropy_loss(&target, 1).unwrap();
    let val = loss.tensor().to_flat_vec::<f32>().unwrap()[0];
    let expected = 10.0_f32.ln();
    assert!(
        (val - expected).abs() < 1e-3,
        "10-class uniform CE = {val}, expected {expected}"
    );
}

// ===========================================================================
// 15. Gradient shape consistency
// ===========================================================================

#[test]
fn test_mse_gradient_shape_matches_input_1d() {
    let var = var_from(vec![1.0, 2.0, 3.0, 4.0, 5.0], &[5]);
    let tgt = const_tensor(vec![0.0; 5], &[5]);
    let x = tracked_from_var(&var);
    let loss = x.mse_loss(&tgt).unwrap();
    let grads = backward(&loss).unwrap();
    let g = grads.get(&var).unwrap();
    assert_eq!(g.dims(), &[5], "grad shape should match input shape");
}

#[test]
fn test_mse_gradient_shape_matches_input_2d() {
    let var = var_from(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
    let tgt = const_tensor(vec![0.0; 6], &[2, 3]);
    let x = tracked_from_var(&var);
    let loss = x.mse_loss(&tgt).unwrap();
    let grads = backward(&loss).unwrap();
    let g = grads.get(&var).unwrap();
    assert_eq!(g.dims(), &[2, 3], "grad shape should match input shape");
}

#[test]
fn test_l1_gradient_shape_matches_input() {
    let var = var_from(vec![1.0; 12], &[3, 4]);
    let tgt = const_tensor(vec![0.0; 12], &[3, 4]);
    let x = tracked_from_var(&var);
    let loss = x.l1_loss(&tgt).unwrap();
    let grads = backward(&loss).unwrap();
    let g = grads.get(&var).unwrap();
    assert_eq!(g.dims(), &[3, 4], "L1 grad shape should match input");
}

#[test]
fn test_huber_gradient_shape_matches_input() {
    let var = var_from(vec![1.0; 6], &[2, 3]);
    let tgt = const_tensor(vec![0.0; 6], &[2, 3]);
    let x = tracked_from_var(&var);
    let loss = x.huber_loss(&tgt, 1.0).unwrap();
    let grads = backward(&loss).unwrap();
    let g = grads.get(&var).unwrap();
    assert_eq!(g.dims(), &[2, 3], "Huber grad shape should match input");
}

#[test]
fn test_ce_gradient_shape_matches_logits() {
    let var = var_from(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
    let target = u32_target(vec![0, 2], &[2, 1]);
    let x = tracked_from_var(&var);
    let loss = x.cross_entropy_loss(&target, 1).unwrap();
    let grads = backward(&loss).unwrap();
    let g = grads.get(&var).unwrap();
    assert_eq!(g.dims(), &[2, 3], "CE grad shape should match logits shape");
}

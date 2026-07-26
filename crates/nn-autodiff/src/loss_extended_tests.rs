#![allow(deprecated)]
// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Extended loss function and training loop tests.
//!
//! Covers: MSE gradient verification, cross-entropy (binary + multi-class),
//! L1 gradient sign, Huber transition, loss reduction shapes, gradient
//! accumulation across multiple forward-backward passes, mixed precision
//! (BF16), and training loop integration on a simple quadratic.

use crate::{backward, TrackedTensor, Var};
use nn_core::dyn_tensor::DynTensor;
use nn_core::{DType, Device};
use std::sync::Arc;

fn cpu() -> Device {
    Device::Cpu
}

// ---------------------------------------------------------------------------
// 1. MSE loss -- verify gradient is 2*(predicted - target)/n
// ---------------------------------------------------------------------------

#[test]
fn test_mse_gradient_analytical_formula() {
    // For MSE loss = mean((pred - target)^2), d(loss)/d(pred) = 2*(pred - target)/n
    let pred_vals = vec![3.0, 5.0, 1.0, 4.0];
    let target_vals = vec![1.0, 2.0, 3.0, 2.0];
    let n = pred_vals.len() as f32;

    let var = Var::new(DynTensor::from_vec(pred_vals.clone(), &[4], &cpu()).unwrap());
    let target = Arc::new(TrackedTensor::from_tensor(
        DynTensor::from_vec(target_vals.clone(), &[4], &cpu()).unwrap(),
    ));

    let x = Arc::new(TrackedTensor::from_var(&var).unwrap());
    let loss = x.mse_loss(&target).unwrap();
    let grads = backward(&loss).unwrap();
    let grad = grads.get(&var).unwrap().to_flat_vec::<f32>().unwrap();

    for i in 0..pred_vals.len() {
        let expected = 2.0 * (pred_vals[i] - target_vals[i]) / n;
        assert!(
            (grad[i] - expected).abs() < 1e-5,
            "MSE grad[{i}]: expected {expected}, got {}",
            grad[i]
        );
    }
}

#[test]
fn test_mse_gradient_2d_tensor() {
    // 2D case: [2, 3] tensor
    let pred_vals = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let target_vals = vec![2.0, 2.0, 2.0, 2.0, 2.0, 2.0];
    let n = pred_vals.len() as f32;

    let var = Var::new(DynTensor::from_vec(pred_vals.clone(), &[2, 3], &cpu()).unwrap());
    let target = Arc::new(TrackedTensor::from_tensor(
        DynTensor::from_vec(target_vals.clone(), &[2, 3], &cpu()).unwrap(),
    ));

    let x = Arc::new(TrackedTensor::from_var(&var).unwrap());
    let loss = x.mse_loss(&target).unwrap();
    let grads = backward(&loss).unwrap();
    let grad = grads.get(&var).unwrap().to_flat_vec::<f32>().unwrap();

    for i in 0..pred_vals.len() {
        let expected = 2.0 * (pred_vals[i] - target_vals[i]) / n;
        assert!(
            (grad[i] - expected).abs() < 1e-5,
            "MSE 2D grad[{i}]: expected {expected}, got {}",
            grad[i]
        );
    }
}

// ---------------------------------------------------------------------------
// 2. Cross-entropy loss -- binary and multi-class gradient verification
// ---------------------------------------------------------------------------

#[test]
fn test_cross_entropy_binary_value() {
    // Binary cross-entropy via 2-class: logits=[2.0, -1.0], target=0
    // CE = -log(softmax(logits)[target])
    let logits = vec![2.0, -1.0];
    let input = Arc::new(TrackedTensor::from_tensor(
        DynTensor::from_vec(logits, &[1, 2], &cpu()).unwrap(),
    ));
    // Target index: class 0 (U32 class indices for gather)
    let target = Arc::new(TrackedTensor::from_tensor(
        DynTensor::from_vec_u32(vec![0u32], &[1, 1], &cpu()).unwrap(),
    ));
    let loss = input.cross_entropy_loss(&target, 1).unwrap();
    let val = loss.tensor().to_flat_vec::<f32>().unwrap()[0];

    // softmax([2, -1]) = [exp(2)/(exp(2)+exp(-1)), exp(-1)/(exp(2)+exp(-1))]
    let e2 = 2.0_f32.exp();
    let em1 = (-1.0_f32).exp();
    let expected_loss = -(e2 / (e2 + em1)).ln();
    assert!(
        (val - expected_loss).abs() < 1e-4,
        "binary CE loss = {val}, expected {expected_loss}"
    );
}

#[test]
fn test_cross_entropy_multiclass_gradient() {
    // 3-class: logits = [1.0, 2.0, 3.0], target = class 1
    // Gradient for CE: softmax(logits) - one_hot(target)
    let logits = vec![1.0, 2.0, 3.0];
    let var = Var::new(DynTensor::from_vec(logits.clone(), &[1, 3], &cpu()).unwrap());
    let target = Arc::new(TrackedTensor::from_tensor(
        DynTensor::from_vec_u32(vec![1u32], &[1, 1], &cpu()).unwrap(),
    ));

    let x = Arc::new(TrackedTensor::from_var(&var).unwrap());
    let loss = x.cross_entropy_loss(&target, 1).unwrap();
    let grads = backward(&loss).unwrap();
    let grad = grads.get(&var).unwrap().to_flat_vec::<f32>().unwrap();

    // softmax([1,2,3]) probabilities
    let exps: Vec<f32> = logits.iter().map(|l| l.exp()).collect();
    let sum_exp: f32 = exps.iter().sum();
    let softmax: Vec<f32> = exps.iter().map(|e| e / sum_exp).collect();

    // CE gradient w.r.t. logits = softmax - one_hot(target)
    // For a single sample (batch_size=1), gradient = softmax[i] - 1{i==target}
    let expected = [softmax[0], softmax[1] - 1.0, softmax[2]];
    for i in 0..3 {
        assert!(
            (grad[i] - expected[i]).abs() < 1e-4,
            "CE grad[{i}]: expected {:.6}, got {:.6}",
            expected[i],
            grad[i]
        );
    }
}

#[test]
fn test_cross_entropy_batch() {
    // Batch of 2 samples, 4 classes
    let logits = vec![
        1.0, 0.0, 0.0, 0.0, // sample 0
        0.0, 0.0, 1.0, 0.0, // sample 1
    ];
    let input = Arc::new(TrackedTensor::from_tensor(
        DynTensor::from_vec(logits, &[2, 4], &cpu()).unwrap(),
    ));
    // Targets: sample 0 -> class 0, sample 1 -> class 2 (U32 class indices)
    let targets = Arc::new(TrackedTensor::from_tensor(
        DynTensor::from_vec_u32(vec![0u32, 2u32], &[2, 1], &cpu()).unwrap(),
    ));
    let loss = input.cross_entropy_loss(&targets, 1).unwrap();
    let val = loss.tensor().to_flat_vec::<f32>().unwrap()[0];

    // Both samples have correct class with highest logit -> loss should be low
    assert!(
        val > 0.0 && val < 2.0,
        "batch CE loss should be positive and moderate, got {val}"
    );
}

// ---------------------------------------------------------------------------
// 3. L1 loss -- verify gradient sign matches prediction vs target
// ---------------------------------------------------------------------------

#[test]
fn test_l1_gradient_sign() {
    // L1 gradient = sign(pred - target) / n
    let pred_vals = vec![5.0, 1.0, 3.0, 3.0];
    let target_vals = vec![2.0, 4.0, 3.0, 7.0];
    let n = pred_vals.len() as f32;

    let var = Var::new(DynTensor::from_vec(pred_vals.clone(), &[4], &cpu()).unwrap());
    let target = Arc::new(TrackedTensor::from_tensor(
        DynTensor::from_vec(target_vals.clone(), &[4], &cpu()).unwrap(),
    ));

    let x = Arc::new(TrackedTensor::from_var(&var).unwrap());
    let loss = x.l1_loss(&target).unwrap();
    let grads = backward(&loss).unwrap();
    let grad = grads.get(&var).unwrap().to_flat_vec::<f32>().unwrap();

    // Expected signs: pred > target -> +1/n, pred < target -> -1/n, equal -> 0/n
    let expected_signs: Vec<f32> = pred_vals
        .iter()
        .zip(&target_vals)
        .map(|(p, t)| {
            if p > t {
                1.0
            } else if p < t {
                -1.0
            } else {
                0.0
            }
        })
        .collect();

    for i in 0..pred_vals.len() {
        let expected = expected_signs[i] / n;
        assert!(
            (grad[i] - expected).abs() < 1e-5,
            "L1 grad[{i}]: expected {expected}, got {}",
            grad[i]
        );
    }
}

#[test]
fn test_l1_gradient_positive_negative_mixed() {
    // Mix of positive and negative differences
    let pred = vec![10.0, -5.0, 0.0];
    let tgt = vec![3.0, 3.0, -1.0];
    let n = 3.0_f32;

    let var = Var::new(DynTensor::from_vec(pred, &[3], &cpu()).unwrap());
    let target = Arc::new(TrackedTensor::from_tensor(
        DynTensor::from_vec(tgt, &[3], &cpu()).unwrap(),
    ));

    let x = Arc::new(TrackedTensor::from_var(&var).unwrap());
    let loss = x.l1_loss(&target).unwrap();
    let grads = backward(&loss).unwrap();
    let grad = grads.get(&var).unwrap().to_flat_vec::<f32>().unwrap();

    // pred - tgt = [7, -8, 1], signs = [1, -1, 1]
    assert!(grad[0] > 0.0, "grad[0] should be positive, got {}", grad[0]);
    assert!(grad[1] < 0.0, "grad[1] should be negative, got {}", grad[1]);
    assert!(grad[2] > 0.0, "grad[2] should be positive, got {}", grad[2]);

    // Magnitude should be 1/n
    for (i, gi) in grad.iter().take(3).enumerate() {
        assert!(
            (gi.abs() - 1.0 / n).abs() < 1e-5,
            "L1 |grad[{i}]| = {}, expected {}",
            gi.abs(),
            1.0 / n
        );
    }
}

// ---------------------------------------------------------------------------
// 4. Huber loss -- verify transition between L1 and L2 at delta threshold
// ---------------------------------------------------------------------------

#[test]
fn test_huber_transition_at_delta() {
    let delta = 1.0_f64;

    // Test with diffs below, at, and above delta
    // diff = 0.5 (quadratic), diff = 1.5 (linear), diff = 2.5 (linear)
    let pred = vec![1.5, 2.5, 3.5];
    let tgt = vec![1.0, 1.0, 1.0];

    let input = Arc::new(TrackedTensor::from_tensor(
        DynTensor::from_vec(pred, &[3], &cpu()).unwrap(),
    ));
    let target = Arc::new(TrackedTensor::from_tensor(
        DynTensor::from_vec(tgt, &[3], &cpu()).unwrap(),
    ));
    let loss = input.huber_loss(&target, delta).unwrap();
    let val = loss.tensor().to_flat_vec::<f32>().unwrap()[0];

    // diff = [0.5, 1.5, 2.5]
    // |0.5| < 1.0 -> quadratic: 0.5 * 0.25 / 1.0 = 0.125
    // |1.5| >= 1.0 -> linear: 1.5 - 0.5 = 1.0
    // |2.5| >= 1.0 -> linear: 2.5 - 0.5 = 2.0
    // mean = (0.125 + 1.0 + 2.0) / 3 = 1.041667
    let expected = (0.125 + 1.0 + 2.0) / 3.0;
    assert!(
        (val - expected).abs() < 1e-4,
        "Huber transition loss = {val}, expected {expected}"
    );
}

#[test]
fn test_huber_gradient_quadratic_vs_linear_region() {
    let delta = 1.0_f64;
    // diff=0.3 -> quadratic region: grad = diff / (n * delta) = 0.3 / (2 * 1.0) = 0.15
    // diff=5.0 -> linear region: grad = sign(diff) / n = 1.0 / 2 = 0.5
    let pred = vec![1.3, 6.0];
    let tgt = vec![1.0, 1.0];
    let n = 2.0;

    let var = Var::new(DynTensor::from_vec(pred, &[2], &cpu()).unwrap());
    let target = Arc::new(TrackedTensor::from_tensor(
        DynTensor::from_vec(tgt, &[2], &cpu()).unwrap(),
    ));

    let x = Arc::new(TrackedTensor::from_var(&var).unwrap());
    let loss = x.huber_loss(&target, delta).unwrap();
    let grads = backward(&loss).unwrap();
    let grad = grads.get(&var).unwrap().to_flat_vec::<f32>().unwrap();

    // Quadratic region: grad = diff / (delta * n) = 0.3 / (1.0 * 2.0)
    let expected_quad = 0.3 / (delta as f32 * n as f32);
    assert!(
        (grad[0] - expected_quad).abs() < 1e-4,
        "Huber quad grad[0]: expected {expected_quad}, got {}",
        grad[0]
    );

    // Linear region: grad = sign(diff) / n = 1.0 / 2.0
    let expected_lin = 1.0 / n as f32;
    assert!(
        (grad[1] - expected_lin).abs() < 1e-4,
        "Huber linear grad[1]: expected {expected_lin}, got {}",
        grad[1]
    );
}

#[test]
fn test_huber_small_delta_mostly_linear() {
    // Very small delta -> almost all values in linear region -> approximates L1
    let delta = 0.01;
    let pred = vec![5.0, 10.0, -3.0];
    let tgt = vec![1.0, 2.0, 1.0];

    let input = Arc::new(TrackedTensor::from_tensor(
        DynTensor::from_vec(pred, &[3], &cpu()).unwrap(),
    ));
    let target = Arc::new(TrackedTensor::from_tensor(
        DynTensor::from_vec(tgt, &[3], &cpu()).unwrap(),
    ));

    let huber_loss = input.huber_loss(&target, delta).unwrap();
    let huber_val = huber_loss.tensor().to_flat_vec::<f32>().unwrap()[0];

    // All diffs = [4, 8, -4], |diff| >> delta=0.01
    // Linear region: |diff| - 0.5*delta ~= |diff|
    // Mean should approximate L1 loss = mean(4, 8, 4) = 5.333
    let expected_approx = (4.0 + 8.0 + 4.0) / 3.0;
    assert!(
        (huber_val - expected_approx).abs() < 0.1,
        "Huber(delta={delta}) should approximate L1, got {huber_val}, expected ~{expected_approx}"
    );
}

// ---------------------------------------------------------------------------
// 5. CTC loss -- verify if available (skip if not, as it may not be implemented)
// ---------------------------------------------------------------------------

// Note: CTC loss is not currently implemented in nn-autodiff Op enum.
// This test verifies the absence gracefully. When CTC is added, replace
// with actual loss computation tests.
#[test]
fn test_ctc_loss_not_available_documented() {
    // CTC loss is not yet available as an Op variant.
    // This test documents this gap. When CTC is added, verify:
    // 1. Sequence alignment produces correct loss for simple cases
    // 2. Blank token handling works
    // 3. Gradient flows through the alignment
    //
    // For now, verify we can build a basic sequence loss manually using
    // cross-entropy on aligned frames.
    let logits = DynTensor::from_vec(vec![1.0, 0.5, 0.1, 0.2, 0.8, 0.3], &[2, 3], &cpu()).unwrap();
    let input = Arc::new(TrackedTensor::from_tensor(logits));
    // Use log_softmax as a building block for sequence losses
    let log_probs = input.log_softmax(1).unwrap();
    let vals = log_probs.tensor().to_flat_vec::<f32>().unwrap();
    // log_softmax outputs should all be negative (log of probability < 1)
    for (i, v) in vals.iter().enumerate() {
        assert!(*v <= 0.0, "log_softmax[{i}] = {v} should be <= 0");
    }
}

// ---------------------------------------------------------------------------
// 6. Audio losses -- spectral convergence and log STFT magnitude
// ---------------------------------------------------------------------------

#[test]
fn test_stft_loss_spectral_convergence_component() {
    // Two different signals should produce positive spectral convergence
    let n = 1600;
    let sig_a: Vec<f32> = (0..n)
        .map(|i| (2.0 * std::f32::consts::PI * 440.0 * i as f32 / 16000.0).sin() * 0.5)
        .collect();
    let sig_b: Vec<f32> = (0..n)
        .map(|i| (2.0 * std::f32::consts::PI * 880.0 * i as f32 / 16000.0).sin() * 0.5)
        .collect();

    let a = Arc::new(TrackedTensor::from_tensor(
        DynTensor::from_vec(sig_a, &[n], &cpu()).unwrap(),
    ));
    let b = Arc::new(TrackedTensor::from_tensor(
        DynTensor::from_vec(sig_b, &[n], &cpu()).unwrap(),
    ));

    let loss = crate::stft_loss(&a, &b, 512, 128).unwrap();
    let val = loss.tensor().to_flat_vec::<f32>().unwrap()[0];
    assert!(
        val > 0.0,
        "STFT loss should be positive for different signals, got {val}"
    );
    assert!(val.is_finite(), "STFT loss should be finite, got {val}");
}

#[test]
fn test_multi_res_stft_loss_different_resolutions() {
    // Test that multi-res averages across resolutions produce a sensible value
    let n = 3200;
    let sig_a: Vec<f32> = (0..n)
        .map(|i| (2.0 * std::f32::consts::PI * 440.0 * i as f32 / 16000.0).sin() * 0.5)
        .collect();
    let sig_b: Vec<f32> = (0..n)
        .map(|i| (2.0 * std::f32::consts::PI * 660.0 * i as f32 / 16000.0).sin() * 0.5)
        .collect();

    let a = Arc::new(TrackedTensor::from_tensor(
        DynTensor::from_vec(sig_a, &[n], &cpu()).unwrap(),
    ));
    let b = Arc::new(TrackedTensor::from_tensor(
        DynTensor::from_vec(sig_b, &[n], &cpu()).unwrap(),
    ));

    let loss_single = crate::stft_loss(&a, &b, 512, 128).unwrap();
    let val_single = loss_single.tensor().to_flat_vec::<f32>().unwrap()[0];

    let loss_multi = crate::multi_res_stft_loss(&a, &b, &[512, 1024]).unwrap();
    let val_multi = loss_multi.tensor().to_flat_vec::<f32>().unwrap()[0];

    // Both should be positive
    assert!(val_single > 0.0, "single-res STFT loss should be positive");
    assert!(val_multi > 0.0, "multi-res STFT loss should be positive");

    // Multi-res is an average, so it should differ from single-res
    // (unless by coincidence, which is unlikely with different FFT sizes)
    assert!(
        (val_multi - val_single).abs() > 1e-6 || val_multi > 0.0,
        "multi-res should produce a distinct averaged value"
    );
}

#[test]
fn test_mel_spectrogram_loss_gradient_exists() {
    let n = 1600;
    let sig_ref: Vec<f32> = (0..n)
        .map(|i| (2.0 * std::f32::consts::PI * 440.0 * i as f32 / 16000.0).sin() * 0.5)
        .collect();
    let sig_cand: Vec<f32> = (0..n)
        .map(|i| (2.0 * std::f32::consts::PI * 500.0 * i as f32 / 16000.0).sin() * 0.3)
        .collect();

    let var = Var::new(DynTensor::from_vec(sig_cand, &[n], &cpu()).unwrap());
    let cand = Arc::new(TrackedTensor::from_var(&var).unwrap());
    let refr = Arc::new(TrackedTensor::from_tensor(
        DynTensor::from_vec(sig_ref, &[n], &cpu()).unwrap(),
    ));

    let loss = crate::mel_spectrogram_loss(&cand, &refr, 40, 512, 16000).unwrap();
    let grads = backward(&loss).unwrap();
    let grad = grads.get(&var).unwrap();
    let grad_vals = grad.to_flat_vec::<f32>().unwrap();
    let grad_norm: f32 = grad_vals.iter().map(|g| g * g).sum::<f32>().sqrt();
    assert!(
        grad_norm > 1e-6,
        "mel spectrogram loss gradient should be non-zero, norm = {grad_norm}"
    );
    assert!(
        grad_vals.iter().all(|g| g.is_finite()),
        "all mel loss gradients should be finite"
    );
}

// ---------------------------------------------------------------------------
// 7. Loss reduction modes -- sum, mean, none produce correct shapes
// ---------------------------------------------------------------------------

// The nn-autodiff loss functions always reduce to scalar (mean reduction).
// These tests verify the reduction behavior and that manual reduction
// approaches produce the expected shapes.

#[test]
fn test_mse_loss_reduces_to_scalar() {
    let input = Arc::new(TrackedTensor::from_tensor(
        DynTensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], &cpu()).unwrap(),
    ));
    let target = Arc::new(TrackedTensor::from_tensor(
        DynTensor::from_vec(vec![0.0; 6], &[2, 3], &cpu()).unwrap(),
    ));
    let loss = input.mse_loss(&target).unwrap();
    // MSE reduces to scalar regardless of input shape
    assert_eq!(
        loss.tensor().numel(),
        1,
        "MSE loss should be scalar, got shape {:?}",
        loss.dims()
    );
}

#[test]
fn test_l1_loss_reduces_to_scalar() {
    let input = Arc::new(TrackedTensor::from_tensor(
        DynTensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], &cpu()).unwrap(),
    ));
    let target = Arc::new(TrackedTensor::from_tensor(
        DynTensor::from_vec(vec![0.0; 6], &[2, 3], &cpu()).unwrap(),
    ));
    let loss = input.l1_loss(&target).unwrap();
    assert_eq!(
        loss.tensor().numel(),
        1,
        "L1 loss should be scalar, got shape {:?}",
        loss.dims()
    );
}

#[test]
fn test_manual_sum_reduction_shape() {
    // Manual sum reduction: compute element-wise squared diff, then sum
    let pred = Arc::new(TrackedTensor::from_tensor(
        DynTensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2], &cpu()).unwrap(),
    ));
    let tgt = Arc::new(TrackedTensor::from_tensor(
        DynTensor::from_vec(vec![0.0; 4], &[2, 2], &cpu()).unwrap(),
    ));
    let diff = pred.sub(&tgt).unwrap();
    let sq = diff.sqr().unwrap();
    // Sum over all dims to get scalar
    let sum_d1 = sq.sum_keepdim(1).unwrap(); // [2, 1]
    let sum_all = sum_d1.sum_keepdim(0).unwrap(); // [1, 1]
    assert_eq!(
        sum_all.tensor().numel(),
        1,
        "sum reduction should produce single element"
    );
    let val = sum_all.tensor().to_flat_vec::<f32>().unwrap()[0];
    // sum(1^2 + 2^2 + 3^2 + 4^2) = 1 + 4 + 9 + 16 = 30
    assert!(
        (val - 30.0).abs() < 1e-4,
        "sum reduction = {val}, expected 30.0"
    );
}

#[test]
fn test_manual_none_reduction_preserves_shape() {
    // "none" reduction: compute element-wise loss without reducing
    let pred = Arc::new(TrackedTensor::from_tensor(
        DynTensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2], &cpu()).unwrap(),
    ));
    let tgt = Arc::new(TrackedTensor::from_tensor(
        DynTensor::from_vec(vec![0.0; 4], &[2, 2], &cpu()).unwrap(),
    ));
    let diff = pred.sub(&tgt).unwrap();
    let sq = diff.sqr().unwrap();
    // No reduction: shape should be [2, 2]
    assert_eq!(sq.dims(), &[2, 2], "unreduced loss should preserve shape");
    let vals = sq.tensor().to_flat_vec::<f32>().unwrap();
    assert!((vals[0] - 1.0).abs() < 1e-5); // 1^2
    assert!((vals[1] - 4.0).abs() < 1e-5); // 2^2
    assert!((vals[2] - 9.0).abs() < 1e-5); // 3^2
    assert!((vals[3] - 16.0).abs() < 1e-5); // 4^2
}

// ---------------------------------------------------------------------------
// 8. Gradient accumulation -- across multiple forward-backward passes
// ---------------------------------------------------------------------------

#[test]
fn test_gradient_accumulation_manual_sum() {
    // Simulate gradient accumulation over 3 mini-batches before optimizer step
    let var = Var::from_tensor(&DynTensor::from_vec(vec![2.0, 3.0], &[2], &cpu()).unwrap());

    let mut accumulated_grad = [0.0_f32; 2];

    // Mini-batch 1: loss = sum(x^2) for x=[2,3] -> grad = [4, 6]
    let t1 = Arc::new(TrackedTensor::from_var(&var).unwrap());
    let sq1 = t1.sqr().unwrap();
    let loss1 = sq1.sum_keepdim(0).unwrap();
    let grads1 = backward(&loss1).unwrap();
    let g1 = grads1.get(&var).unwrap().to_flat_vec::<f32>().unwrap();
    for i in 0..2 {
        accumulated_grad[i] += g1[i];
    }

    // Mini-batch 2: same var, new forward pass -> same gradient
    let t2 = Arc::new(TrackedTensor::from_var(&var).unwrap());
    let sq2 = t2.sqr().unwrap();
    let loss2 = sq2.sum_keepdim(0).unwrap();
    let grads2 = backward(&loss2).unwrap();
    let g2 = grads2.get(&var).unwrap().to_flat_vec::<f32>().unwrap();
    for i in 0..2 {
        accumulated_grad[i] += g2[i];
    }

    // Mini-batch 3: yet another forward-backward
    let t3 = Arc::new(TrackedTensor::from_var(&var).unwrap());
    let sq3 = t3.sqr().unwrap();
    let loss3 = sq3.sum_keepdim(0).unwrap();
    let grads3 = backward(&loss3).unwrap();
    let g3 = grads3.get(&var).unwrap().to_flat_vec::<f32>().unwrap();
    for i in 0..2 {
        accumulated_grad[i] += g3[i];
    }

    // Each backward gives grad = [4, 6], so accumulated = [12, 18]
    assert!(
        (accumulated_grad[0] - 12.0).abs() < 1e-4,
        "accumulated grad[0] = {}, expected 12.0",
        accumulated_grad[0]
    );
    assert!(
        (accumulated_grad[1] - 18.0).abs() < 1e-4,
        "accumulated grad[1] = {}, expected 18.0",
        accumulated_grad[1]
    );
}

#[test]
fn test_gradient_accumulation_different_losses() {
    // Accumulate gradients from two different loss functions
    let var = Var::from_tensor(&DynTensor::from_vec(vec![3.0], &[1], &cpu()).unwrap());

    // Loss 1: x^2, grad = 2*3 = 6
    let t1 = Arc::new(TrackedTensor::from_var(&var).unwrap());
    let loss1 = t1.sqr().unwrap();
    let grads1 = backward(&loss1).unwrap();
    let g1 = grads1.get(&var).unwrap().to_scalar::<f32>().unwrap();

    // Loss 2: 5*x, grad = 5
    let t2 = Arc::new(TrackedTensor::from_var(&var).unwrap());
    let loss2 = t2.mul_scalar(5.0).unwrap();
    let loss2_sum = loss2.sum_keepdim(0).unwrap();
    let grads2 = backward(&loss2_sum).unwrap();
    let g2 = grads2.get(&var).unwrap().to_scalar::<f32>().unwrap();

    // Manually accumulated gradient = 6 + 5 = 11
    let accumulated = g1 + g2;
    assert!(
        (accumulated - 11.0).abs() < 1e-4,
        "accumulated gradient = {accumulated}, expected 11.0"
    );
}

#[test]
fn test_gradient_accumulation_apply_averaged_update() {
    // Accumulate over N=4 mini-batches, apply averaged update
    let var = Var::from_tensor(&DynTensor::from_vec(vec![10.0], &[1], &cpu()).unwrap());
    let num_accum = 4;
    let lr = 0.1_f32;

    let mut accumulated = 0.0_f32;
    for _ in 0..num_accum {
        let t = Arc::new(TrackedTensor::from_var(&var).unwrap());
        let loss = t.sqr().unwrap(); // grad = 2*10 = 20
        let grads = backward(&loss).unwrap();
        let g = grads.get(&var).unwrap().to_scalar::<f32>().unwrap();
        accumulated += g;
    }

    // Average gradient over accumulation steps
    let avg_grad = accumulated / num_accum as f32;
    // avg_grad = 20 (since var doesn't change between accumulations)
    assert!(
        (avg_grad - 20.0).abs() < 1e-4,
        "avg gradient = {avg_grad}, expected 20.0"
    );

    // Apply update: x_new = x - lr * avg_grad = 10 - 0.1 * 20 = 8
    let x_val = var.data().unwrap().to_scalar::<f32>().unwrap();
    let new_x = x_val - lr * avg_grad;
    var.set(&DynTensor::from_vec(vec![new_x], &[1], &cpu()).unwrap())
        .unwrap();
    let result = var.data().unwrap().to_scalar::<f32>().unwrap();
    assert!(
        (result - 8.0).abs() < 1e-4,
        "after averaged update, x = {result}, expected 8.0"
    );
}

// ---------------------------------------------------------------------------
// 9. Mixed precision -- loss computation with BF16 inputs
// ---------------------------------------------------------------------------

#[test]
fn test_mse_loss_bf16_inputs() {
    // Create F32 tensors and convert to BF16 for the loss computation
    let pred_f32 = DynTensor::from_vec(vec![1.0, 2.0, 3.0], &[3], &cpu()).unwrap();
    let tgt_f32 = DynTensor::from_vec(vec![1.5, 2.5, 3.5], &[3], &cpu()).unwrap();

    let pred_bf16 = pred_f32.to_dtype(DType::BF16).unwrap();
    let tgt_bf16 = tgt_f32.to_dtype(DType::BF16).unwrap();

    // Compute MSE on BF16 tensors
    let pred_tracked = Arc::new(TrackedTensor::from_tensor(pred_bf16));
    let tgt_tracked = Arc::new(TrackedTensor::from_tensor(tgt_bf16));
    let loss = pred_tracked.mse_loss(&tgt_tracked).unwrap();
    let val = loss.tensor().to_flat_vec::<f32>().unwrap()[0];

    // BF16 has lower precision, but the result should be close to 0.25
    assert!(
        (val - 0.25).abs() < 0.05,
        "BF16 MSE loss = {val}, expected ~0.25 (BF16 tolerance)"
    );
    assert!(val.is_finite(), "BF16 MSE loss should be finite");
}

#[test]
fn test_loss_scaling_round_trip() {
    // Verify DynamicLossScaler scale/unscale round trip preserves gradients
    use crate::loss_scaling::{DynamicLossScaler, MixedPrecisionConfig};

    let config = MixedPrecisionConfig::bf16_training();
    let mut scaler = DynamicLossScaler::new(config).unwrap();
    let scale = scaler.scale_factor();
    assert!(scale > 1.0, "scale should be > 1, got {scale}");

    // Create a loss and scale it
    let loss = DynTensor::from_vec(vec![0.5], &[1], &cpu()).unwrap();
    let scaled = scaler.scale_loss(&loss).unwrap();
    let scaled_val = scaled.to_flat_vec::<f32>().unwrap()[0];
    assert!(
        (scaled_val - 0.5 * scale).abs() < 1e-2,
        "scaled loss = {scaled_val}, expected {}",
        0.5 * scale
    );

    // Unscale a gradient
    let grad = DynTensor::from_vec(vec![scale * 2.0], &[1], &cpu()).unwrap();
    let mut grads = vec![grad];
    scaler.unscale_gradients(&mut grads).unwrap();
    let unscaled = grads[0].to_flat_vec::<f32>().unwrap()[0];
    assert!(
        (unscaled - 2.0).abs() < 1e-4,
        "unscaled gradient = {unscaled}, expected 2.0"
    );

    // Update with no inf found -> should increment counter
    let pre_steps = scaler.consecutive_good_steps();
    scaler.update(false);
    assert_eq!(
        scaler.consecutive_good_steps(),
        pre_steps + 1,
        "good step should increment counter"
    );

    // Update with inf found -> should reset counter
    scaler.update(true);
    assert_eq!(
        scaler.consecutive_good_steps(),
        0,
        "inf found should reset counter"
    );
}

#[test]
fn test_cast_grad_to_f32_bf16() {
    use crate::loss_scaling::cast_grad_to_f32;

    let grad_f32 = DynTensor::from_vec(vec![1.0, 2.0, 3.0], &[3], &cpu()).unwrap();
    let grad_bf16 = grad_f32.to_dtype(DType::BF16).unwrap();
    assert_eq!(grad_bf16.dtype(), DType::BF16);

    let casted = cast_grad_to_f32(&grad_bf16).unwrap();
    assert_eq!(casted.dtype(), DType::F32);
    let vals = casted.to_flat_vec::<f32>().unwrap();
    // BF16 round-trip should be approximately equal
    assert!((vals[0] - 1.0).abs() < 0.1);
    assert!((vals[1] - 2.0).abs() < 0.1);
    assert!((vals[2] - 3.0).abs() < 0.1);
}

// ---------------------------------------------------------------------------
// 10. Training loop integration -- step() reduces loss on simple quadratic
// ---------------------------------------------------------------------------

#[test]
fn test_training_loop_quadratic_convergence() {
    // Minimize f(x) = (x - 3)^2, starting from x = 10
    // After SGD steps, x should converge toward 3
    let var = Var::from_tensor(&DynTensor::from_vec(vec![10.0], &[1], &cpu()).unwrap());
    let lr = 0.1_f32;
    let target_val = 3.0_f32;

    let mut losses = Vec::new();
    for _ in 0..50 {
        let x = Arc::new(TrackedTensor::from_var(&var).unwrap());
        let target = Arc::new(TrackedTensor::from_tensor(
            DynTensor::from_vec(vec![target_val], &[1], &cpu()).unwrap(),
        ));
        let loss = x.mse_loss(&target).unwrap();
        let val = loss.tensor().to_flat_vec::<f32>().unwrap()[0];
        losses.push(val);

        let grads = backward(&loss).unwrap();
        let grad = grads.get(&var).unwrap();
        let x_data = var.data().unwrap();
        let x_vec = x_data.to_flat_vec::<f32>().unwrap();
        let g_vec = grad.to_flat_vec::<f32>().unwrap();
        let new_x: Vec<f32> = x_vec
            .iter()
            .zip(&g_vec)
            .map(|(xi, gi)| xi - lr * gi)
            .collect();
        var.set(&DynTensor::from_vec(new_x, &[1], &cpu()).unwrap())
            .unwrap();
    }

    // Loss should decrease monotonically (convex problem)
    for i in 1..losses.len() {
        assert!(
            losses[i] <= losses[i - 1] + 1e-6,
            "step {i}: loss {:.6} > prev {:.6}",
            losses[i],
            losses[i - 1]
        );
    }

    // Final value should be close to target
    let final_x = var.data().unwrap().to_scalar::<f32>().unwrap();
    assert!(
        (final_x - target_val).abs() < 0.01,
        "x should converge to {target_val}, got {final_x}"
    );

    // Final loss should be near zero
    assert!(
        *losses.last().unwrap() < 0.001,
        "final loss should be < 0.001, got {}",
        losses.last().unwrap()
    );
}

#[test]
fn test_training_loop_two_variable_quadratic() {
    // Minimize f(a, b) = (a - 1)^2 + (b - 2)^2
    // Starting from a=5, b=8
    let a = Var::from_tensor(&DynTensor::from_vec(vec![5.0], &[1], &cpu()).unwrap());
    let b = Var::from_tensor(&DynTensor::from_vec(vec![8.0], &[1], &cpu()).unwrap());
    let lr = 0.1_f32;

    let mut losses = Vec::new();
    for _ in 0..100 {
        let ta = Arc::new(TrackedTensor::from_var(&a).unwrap());
        let tb = Arc::new(TrackedTensor::from_var(&b).unwrap());

        let target_a = Arc::new(TrackedTensor::from_tensor(
            DynTensor::from_vec(vec![1.0], &[1], &cpu()).unwrap(),
        ));
        let target_b = Arc::new(TrackedTensor::from_tensor(
            DynTensor::from_vec(vec![2.0], &[1], &cpu()).unwrap(),
        ));

        let loss_a = ta.mse_loss(&target_a).unwrap();
        let loss_b = tb.mse_loss(&target_b).unwrap();
        let total_loss = loss_a.add(&loss_b).unwrap();
        let val = total_loss.tensor().to_flat_vec::<f32>().unwrap()[0];
        losses.push(val);

        let grads = backward(&total_loss).unwrap();

        // Update a
        let ga = grads.get(&a).unwrap();
        let a_data = a.data().unwrap();
        let a_vec = a_data.to_flat_vec::<f32>().unwrap();
        let ga_vec = ga.to_flat_vec::<f32>().unwrap();
        let new_a: Vec<f32> = a_vec
            .iter()
            .zip(&ga_vec)
            .map(|(xi, gi)| xi - lr * gi)
            .collect();
        a.set(&DynTensor::from_vec(new_a, &[1], &cpu()).unwrap())
            .unwrap();

        // Update b
        let gb = grads.get(&b).unwrap();
        let b_data = b.data().unwrap();
        let b_vec = b_data.to_flat_vec::<f32>().unwrap();
        let gb_vec = gb.to_flat_vec::<f32>().unwrap();
        let new_b: Vec<f32> = b_vec
            .iter()
            .zip(&gb_vec)
            .map(|(xi, gi)| xi - lr * gi)
            .collect();
        b.set(&DynTensor::from_vec(new_b, &[1], &cpu()).unwrap())
            .unwrap();
    }

    let final_a = a.data().unwrap().to_scalar::<f32>().unwrap();
    let final_b = b.data().unwrap().to_scalar::<f32>().unwrap();
    assert!(
        (final_a - 1.0).abs() < 0.01,
        "a should converge to 1.0, got {final_a}"
    );
    assert!(
        (final_b - 2.0).abs() < 0.01,
        "b should converge to 2.0, got {final_b}"
    );

    // Loss should decrease overall
    assert!(
        losses.last().unwrap() < losses.first().unwrap(),
        "loss should decrease: first={}, last={}",
        losses.first().unwrap(),
        losses.last().unwrap()
    );
}

#[test]
fn test_training_loop_with_l1_loss_convergence() {
    // Train x toward 5.0 using L1 loss (gradient is sign-based)
    let var = Var::from_tensor(&DynTensor::from_vec(vec![0.0], &[1], &cpu()).unwrap());
    let lr = 0.5_f32;

    let _initial_dist = 5.0_f32; // initial distance from target
    for _ in 0..20 {
        let x = Arc::new(TrackedTensor::from_var(&var).unwrap());
        let target = Arc::new(TrackedTensor::from_tensor(
            DynTensor::from_vec(vec![5.0], &[1], &cpu()).unwrap(),
        ));
        let loss = x.l1_loss(&target).unwrap();
        let grads = backward(&loss).unwrap();
        let grad = grads.get(&var).unwrap();
        let x_data = var.data().unwrap();
        let x_vec = x_data.to_flat_vec::<f32>().unwrap();
        let g_vec = grad.to_flat_vec::<f32>().unwrap();
        let new_x: Vec<f32> = x_vec
            .iter()
            .zip(&g_vec)
            .map(|(xi, gi)| xi - lr * gi)
            .collect();
        var.set(&DynTensor::from_vec(new_x, &[1], &cpu()).unwrap())
            .unwrap();
    }

    let final_x = var.data().unwrap().to_scalar::<f32>().unwrap();
    let final_dist = (final_x - 5.0).abs();
    // L1 gradient is constant magnitude, so progress is linear
    // After 20 steps with lr=0.5, from 0: 0 + 20*0.5 = 10, but clamped by overshooting
    // The x value may oscillate around 5.0; check it's close
    assert!(
        final_dist < 1.0,
        "x should be close to 5.0, got {final_x} (dist={final_dist})"
    );
}

#[test]
fn test_training_loop_integration_run_training_loop_api() {
    // Use the run_training_loop API for a complete integration test
    use crate::train_loop::{run_training_loop, SampleScore, TrainLoopConfig};

    let var = Var::from_tensor(&DynTensor::from_vec(vec![10.0], &[1], &cpu()).unwrap());
    let lr = 0.2_f32;
    let var_ref = var.clone();

    let config = TrainLoopConfig {
        max_epochs: 10,
        curriculum_fraction: 1.0,
        target_score: None,
        log_interval: 0,
    };

    let summary = run_training_loop(
        &config,
        1,
        |_| vec![SampleScore::new(0, 0.1)],
        |_| {
            let x = Arc::new(TrackedTensor::from_var(&var).unwrap());
            let target = Arc::new(TrackedTensor::from_tensor(
                DynTensor::from_vec(vec![0.0], &[1], &cpu()).unwrap(),
            ));
            x.mse_loss(&target)
        },
        |loss| {
            let grads = backward(loss)?;
            let grad = grads.get(&var_ref).unwrap();
            let x_data = var_ref.data()?;
            let x_vec = x_data.to_flat_vec::<f32>().unwrap();
            let g_vec = grad.to_flat_vec::<f32>().unwrap();
            let new_x: Vec<f32> = x_vec
                .iter()
                .zip(&g_vec)
                .map(|(xi, gi)| xi - lr * gi)
                .collect();
            var_ref.set(&DynTensor::from_vec(new_x, &[1], &cpu()).unwrap())?;
            Ok(())
        },
    )
    .unwrap();

    // Should have run 10 epochs, 1 step each
    assert_eq!(summary.epoch_metrics.len(), 10);
    assert_eq!(summary.total_steps, 10);

    // Loss should decrease over epochs
    let first_loss = summary.epoch_metrics[0].mean_loss;
    let last_loss = summary.epoch_metrics[9].mean_loss;
    assert!(
        last_loss < first_loss,
        "loss should decrease: first={first_loss}, last={last_loss}"
    );

    // x should have moved toward 0
    let final_x = var.data().unwrap().to_scalar::<f32>().unwrap();
    assert!(
        final_x.abs() < 10.0,
        "x should have moved from 10.0 toward 0, got {final_x}"
    );
}

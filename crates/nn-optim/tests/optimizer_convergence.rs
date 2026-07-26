// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Cross-crate integration tests for optimizer convergence behavior.
//!
//! Tests verify that Adam, SGD, and AdaFactor minimize known objective
//! functions, and that hyperparameter changes have expected effects.

use std::sync::Arc;

use nn_autodiff::{TrackedTensor, Var};
use nn_core::dyn_tensor::DynTensor;
use nn_core::Device;
use nn_optim::{AdaFactor, AdaFactorConfig, AdamConfig, AdamW, Optimizer, Sgd, SgdConfig};

fn cpu() -> Device {
    Device::Cpu
}

/// Create a scalar (shape [1]) Var with the given value.
fn scalar_var(val: f32) -> Var {
    Var::new(DynTensor::from_vec(vec![val], &[1], &cpu()).unwrap())
}

/// Read the single f32 element from a scalar Var.
fn read_scalar(var: &Var) -> f32 {
    var.data().unwrap().to_flat_vec::<f32>().unwrap()[0]
}

/// Build scalar loss = x^2 from a Var and run backward_step.
fn step_quadratic(opt: &mut dyn Optimizer, var: &Var) {
    let t = Arc::new(TrackedTensor::from_var(var).unwrap());
    let loss = t.sqr().unwrap();
    opt.backward_step(&loss).unwrap();
}

fn adam_config(lr: f64, weight_decay: f64) -> AdamConfig {
    let mut c = AdamConfig::default();
    c.lr = lr;
    c.weight_decay = weight_decay;
    c
}

fn adam_config_betas(lr: f64, beta1: f64, weight_decay: f64) -> AdamConfig {
    let mut c = AdamConfig::default();
    c.lr = lr;
    c.beta1 = beta1;
    c.weight_decay = weight_decay;
    c
}

fn sgd_config(lr: f64, momentum: f64, weight_decay: f64) -> SgdConfig {
    let mut c = SgdConfig::default();
    c.lr = lr;
    c.momentum = momentum;
    c.weight_decay = weight_decay;
    c
}

fn adafactor_config(lr: f64, weight_decay: f64) -> AdaFactorConfig {
    let mut c = AdaFactorConfig::default();
    c.lr = lr;
    c.relative_step = false;
    c.weight_decay = weight_decay;
    c
}

// ============================================================================
// Adam convergence tests
// ============================================================================

#[test]
fn test_adam_converges_quadratic() {
    // Minimize f(x) = x^2. Global minimum at x = 0.
    let x = scalar_var(5.0);
    let mut adam = AdamW::new(vec![x.clone()], adam_config(0.1, 0.0)).unwrap();

    for _ in 0..200 {
        step_quadratic(&mut adam, &x);
    }

    let val = read_scalar(&x);
    assert!(
        val.abs() < 0.01,
        "Adam should converge x near 0 on quadratic, got {val}"
    );
}

#[test]
fn test_adam_converges_rosenbrock_2d() {
    // f(a,b) = (1-a)^2 + 100*(b - a^2)^2
    // Minimum at (1, 1). We only need approximate convergence.
    let a = scalar_var(0.0);
    let b = scalar_var(0.0);
    let mut adam = AdamW::new(vec![a.clone(), b.clone()], adam_config(0.005, 0.0)).unwrap();

    for _ in 0..3000 {
        let ta = Arc::new(TrackedTensor::from_var(&a).unwrap());
        let tb = Arc::new(TrackedTensor::from_var(&b).unwrap());

        // (1 - a)^2
        let one = Arc::new(TrackedTensor::from_tensor(
            DynTensor::from_vec(vec![1.0f32], &[1], &cpu()).unwrap(),
        ));
        let diff_a = one.sub(&ta).unwrap();
        let term1 = diff_a.sqr().unwrap();

        // 100 * (b - a^2)^2
        let a_sq = ta.sqr().unwrap();
        let diff_b = tb.sub(&a_sq).unwrap();
        let term2 = diff_b.sqr().unwrap().mul_scalar(100.0).unwrap();

        let loss = term1.add(&term2).unwrap();
        adam.backward_step(&loss).unwrap();
    }

    let va = read_scalar(&a);
    let vb = read_scalar(&b);
    assert!(
        (va - 1.0).abs() < 0.3 && (vb - 1.0).abs() < 0.3,
        "Adam should approach (1,1) on Rosenbrock, got ({va}, {vb})"
    );
}

#[test]
fn test_adam_beta_sensitivity() {
    // Different beta1 values produce different optimization trajectories.
    // Both should converge, but to different intermediate values at step N.
    let x_low = scalar_var(5.0);
    let x_high = scalar_var(5.0);

    let mut adam_low = AdamW::new(vec![x_low.clone()], adam_config_betas(0.1, 0.5, 0.0)).unwrap();

    let mut adam_high =
        AdamW::new(vec![x_high.clone()], adam_config_betas(0.1, 0.99, 0.0)).unwrap();

    for _ in 0..20 {
        step_quadratic(&mut adam_low, &x_low);
        step_quadratic(&mut adam_high, &x_high);
    }

    let val_low = read_scalar(&x_low);
    let val_high = read_scalar(&x_high);
    // Different betas should produce different trajectories
    assert!(
        (val_low - val_high).abs() > 0.01,
        "Different beta1 values should produce different trajectories: low_beta={val_low}, high_beta={val_high}"
    );
    // Both should still converge (be closer to 0 than the start of 5.0)
    assert!(
        val_low.abs() < 4.0 && val_high.abs() < 4.0,
        "Both should make progress: low_beta={val_low}, high_beta={val_high}"
    );
}

// ============================================================================
// SGD convergence tests
// ============================================================================

#[test]
fn test_sgd_converges_quadratic() {
    let x = scalar_var(5.0);
    let mut sgd = Sgd::new(vec![x.clone()], sgd_config(0.05, 0.0, 0.0)).unwrap();

    for _ in 0..200 {
        step_quadratic(&mut sgd, &x);
    }

    let val = read_scalar(&x);
    assert!(val.abs() < 0.01, "SGD should converge x near 0, got {val}");
}

#[test]
fn test_sgd_momentum_accelerates_convergence() {
    // With momentum, SGD should reach lower values in same number of steps.
    let x_no = scalar_var(5.0);
    let x_mom = scalar_var(5.0);

    let mut sgd_no = Sgd::new(vec![x_no.clone()], sgd_config(0.01, 0.0, 0.0)).unwrap();
    let mut sgd_mom = Sgd::new(vec![x_mom.clone()], sgd_config(0.01, 0.9, 0.0)).unwrap();

    for _ in 0..50 {
        step_quadratic(&mut sgd_no, &x_no);
        step_quadratic(&mut sgd_mom, &x_mom);
    }

    let val_no = read_scalar(&x_no).abs();
    let val_mom = read_scalar(&x_mom).abs();
    assert!(
        val_mom < val_no,
        "Momentum should accelerate convergence: no_momentum={val_no}, momentum={val_mom}"
    );
}

// ============================================================================
// AdaFactor convergence tests
// ============================================================================

#[test]
fn test_adafactor_converges_quadratic_scalar() {
    // AdaFactor with rank < 2 uses full second moments (like Adam).
    let x = scalar_var(5.0);
    let mut af = AdaFactor::new(vec![x.clone()], adafactor_config(0.1, 0.0)).unwrap();

    for _ in 0..200 {
        step_quadratic(&mut af, &x);
    }

    let val = read_scalar(&x);
    assert!(
        val.abs() < 0.05,
        "AdaFactor should converge x near 0, got {val}"
    );
}

#[test]
fn test_adafactor_converges_matrix() {
    // AdaFactor with rank >= 2 uses factored second moments.
    // Minimize f(W) = sum(W^2) for a 4x4 matrix.
    let data = DynTensor::from_vec(vec![1.0f32; 16], &[4, 4], &cpu()).unwrap();
    let w = Var::new(data);
    let mut af = AdaFactor::new(vec![w.clone()], adafactor_config(0.05, 0.0)).unwrap();

    for _ in 0..200 {
        let t = Arc::new(TrackedTensor::from_var(&w).unwrap());
        // loss = sum(W^2) via sum_keepdim chains
        let sq = t.sqr().unwrap();
        let loss = sq.sum_keepdim(0).unwrap().sum_keepdim(1).unwrap();
        // Reshape to scalar [1]
        let loss = loss.reshape(&[1]).unwrap();
        af.backward_step(&loss).unwrap();
    }

    let vals = w.data().unwrap().to_flat_vec::<f32>().unwrap();
    let max_abs = vals.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
    assert!(
        max_abs < 0.1,
        "AdaFactor should minimize matrix elements near 0, max_abs={max_abs}"
    );
}

// ============================================================================
// Learning rate effect
// ============================================================================

#[test]
fn test_learning_rate_effect_adam() {
    // Higher LR should converge faster (before overshooting).
    let x_low = scalar_var(5.0);
    let x_high = scalar_var(5.0);

    let mut adam_low = AdamW::new(vec![x_low.clone()], adam_config(0.01, 0.0)).unwrap();
    let mut adam_high = AdamW::new(vec![x_high.clone()], adam_config(0.1, 0.0)).unwrap();

    for _ in 0..30 {
        step_quadratic(&mut adam_low, &x_low);
        step_quadratic(&mut adam_high, &x_high);
    }

    let val_low = read_scalar(&x_low).abs();
    let val_high = read_scalar(&x_high).abs();
    assert!(
        val_high < val_low,
        "Higher LR should converge faster: low_lr={val_low}, high_lr={val_high}"
    );
}

#[test]
fn test_learning_rate_effect_sgd() {
    let x_low = scalar_var(5.0);
    let x_high = scalar_var(5.0);

    let mut sgd_low = Sgd::new(vec![x_low.clone()], sgd_config(0.001, 0.0, 0.0)).unwrap();
    let mut sgd_high = Sgd::new(vec![x_high.clone()], sgd_config(0.05, 0.0, 0.0)).unwrap();

    for _ in 0..30 {
        step_quadratic(&mut sgd_low, &x_low);
        step_quadratic(&mut sgd_high, &x_high);
    }

    let val_low = read_scalar(&x_low).abs();
    let val_high = read_scalar(&x_high).abs();
    assert!(
        val_high < val_low,
        "Higher LR SGD should converge faster: low={val_low}, high={val_high}"
    );
}

// ============================================================================
// Weight decay
// ============================================================================

#[test]
fn test_weight_decay_shrinks_weights_adam() {
    // With weight decay and zero gradient, AdamW should shrink weights.
    let x = scalar_var(5.0);
    let mut adam = AdamW::new(vec![x.clone()], adam_config(0.01, 0.1)).unwrap();

    // Use loss = 0 * x (gradient is 0, but weight decay still applies)
    for _ in 0..50 {
        let t = Arc::new(TrackedTensor::from_var(&x).unwrap());
        let loss = t.mul_scalar(0.0).unwrap();
        adam.backward_step(&loss).unwrap();
    }

    let val = read_scalar(&x);
    assert!(
        val < 5.0,
        "Weight decay should shrink weights from 5.0, got {val}"
    );
}

#[test]
fn test_weight_decay_shrinks_weights_sgd() {
    // Weight decay adds L2 regularization pushing weights toward zero.
    // Compare convergence with and without weight decay.
    let x_wd = scalar_var(5.0);
    let x_no = scalar_var(5.0);
    let mut sgd_wd = Sgd::new(vec![x_wd.clone()], sgd_config(0.01, 0.0, 0.1)).unwrap();
    let mut sgd_no = Sgd::new(vec![x_no.clone()], sgd_config(0.01, 0.0, 0.0)).unwrap();

    for _ in 0..50 {
        step_quadratic(&mut sgd_wd, &x_wd);
        step_quadratic(&mut sgd_no, &x_no);
    }

    let val_wd = read_scalar(&x_wd).abs();
    let val_no = read_scalar(&x_no).abs();
    // Weight decay should push weights closer to zero
    assert!(
        val_wd < val_no,
        "SGD with weight decay should converge faster: wd={val_wd}, no_wd={val_no}"
    );
}

// ============================================================================
// Gradient clipping
// ============================================================================

#[test]
fn test_gradient_clipping_norm() {
    use nn_autodiff::backward;
    use nn_optim::clip_grad_norm;

    // Create a variable with a large value to produce large gradients.
    let x = scalar_var(100.0);
    let t = Arc::new(TrackedTensor::from_var(&x).unwrap());
    let loss = t.sqr().unwrap(); // grad = 200

    let mut grads = backward(&loss).unwrap();
    let original_norm = clip_grad_norm(&mut grads, 1.0).unwrap();

    // Original gradient norm should be 200 (from d/dx x^2 at x=100)
    assert!(
        original_norm > 100.0,
        "Original norm should be large, got {original_norm}"
    );

    // After clipping, the gradient should have norm ~1.0
    let clipped_grad = grads.get(&x).unwrap();
    let clipped_val = clipped_grad.to_flat_vec::<f32>().unwrap()[0];
    assert!(
        clipped_val.abs() < 1.1,
        "Clipped gradient should have magnitude ~1.0, got {clipped_val}"
    );
}

#[test]
fn test_gradient_clipping_value() {
    use nn_autodiff::backward;
    use nn_optim::clip_grad_value;

    let x = scalar_var(100.0);
    let t = Arc::new(TrackedTensor::from_var(&x).unwrap());
    let loss = t.sqr().unwrap(); // grad = 200

    let mut grads = backward(&loss).unwrap();
    clip_grad_value(&mut grads, 5.0).unwrap();

    let grad = grads.get(&x).unwrap();
    let val = grad.to_flat_vec::<f32>().unwrap()[0];
    assert!(
        (val - 5.0).abs() < 1e-5,
        "Gradient should be clamped to 5.0, got {val}"
    );
}

// ============================================================================
// Zero gradient: no update
// ============================================================================

#[test]
fn test_zero_gradient_no_update_adam() {
    let x = scalar_var(3.0);
    let mut adam = AdamW::new(vec![x.clone()], adam_config(0.1, 0.0)).unwrap();

    // loss = 0 (constant, gradient w.r.t. x is 0)
    let t = Arc::new(TrackedTensor::from_var(&x).unwrap());
    let loss = t.mul_scalar(0.0).unwrap();
    adam.backward_step(&loss).unwrap();

    let val = read_scalar(&x);
    assert!(
        (val - 3.0).abs() < 1e-5,
        "Zero gradient should produce no update, got {val}"
    );
}

#[test]
fn test_zero_gradient_no_update_sgd() {
    let x = scalar_var(3.0);
    let mut sgd = Sgd::new(vec![x.clone()], sgd_config(0.1, 0.0, 0.0)).unwrap();

    let t = Arc::new(TrackedTensor::from_var(&x).unwrap());
    let loss = t.mul_scalar(0.0).unwrap();
    sgd.backward_step(&loss).unwrap();

    let val = read_scalar(&x);
    assert!(
        (val - 3.0).abs() < 1e-5,
        "Zero gradient should produce no update, got {val}"
    );
}

// ============================================================================
// set_learning_rate
// ============================================================================

#[test]
fn test_set_learning_rate_adam() {
    let x = scalar_var(1.0);
    let mut adam = AdamW::new(vec![x], adam_config(0.001, 0.0)).unwrap();

    assert!((adam.learning_rate() - 0.001).abs() < f64::EPSILON);

    adam.set_learning_rate(0.05).unwrap();
    assert!((adam.learning_rate() - 0.05).abs() < f64::EPSILON);

    // Negative LR should error
    assert!(adam.set_learning_rate(-1.0).is_err());
}

#[test]
fn test_set_learning_rate_sgd() {
    let x = scalar_var(1.0);
    let mut sgd = Sgd::new(vec![x], sgd_config(0.01, 0.0, 0.0)).unwrap();

    sgd.set_learning_rate(0.1).unwrap();
    assert!((sgd.learning_rate() - 0.1).abs() < f64::EPSILON);

    // NaN LR should error
    assert!(sgd.set_learning_rate(f64::NAN).is_err());
}

// ============================================================================
// LR Schedule integration
// ============================================================================

#[test]
fn test_warmup_schedule_with_adam() {
    use nn_optim::{step_with_schedule, WarmupSchedule};

    let x = scalar_var(5.0);
    let mut adam = AdamW::new(vec![x.clone()], adam_config(0.1, 0.0)).unwrap();

    let schedule = WarmupSchedule::new(0.1, 10).unwrap();

    for step in 0..200 {
        let t = Arc::new(TrackedTensor::from_var(&x).unwrap());
        let loss = t.sqr().unwrap();
        let grads = nn_autodiff::backward(&loss).unwrap();
        step_with_schedule(&mut adam, &grads, &schedule, step).unwrap();
    }

    let val = read_scalar(&x);
    assert!(
        val.abs() < 2.0,
        "Adam with warmup schedule should converge, got {val}"
    );
}

#[test]
fn test_cosine_schedule_with_sgd() {
    use nn_optim::{step_with_schedule, CosineSchedule};

    let x = scalar_var(5.0);
    let mut sgd = Sgd::new(vec![x.clone()], sgd_config(0.1, 0.9, 0.0)).unwrap();

    let schedule = CosineSchedule::new(0.1, 0.001, 5, 100).unwrap();

    for step in 0..100 {
        let t = Arc::new(TrackedTensor::from_var(&x).unwrap());
        let loss = t.sqr().unwrap();
        let grads = nn_autodiff::backward(&loss).unwrap();
        step_with_schedule(&mut sgd, &grads, &schedule, step).unwrap();
    }

    let val = read_scalar(&x);
    assert!(
        val.abs() < 1.0,
        "SGD with cosine schedule should converge, got {val}"
    );
}

// ============================================================================
// Invalid config rejection
// ============================================================================

#[test]
fn test_adam_rejects_invalid_beta() {
    let x = scalar_var(1.0);
    let mut config = AdamConfig::default();
    config.beta1 = 1.0; // invalid: must be < 1.0
    let result = AdamW::new(vec![x], config);
    assert!(result.is_err(), "Adam should reject beta1=1.0");
}

#[test]
fn test_sgd_rejects_negative_momentum() {
    let x = scalar_var(1.0);
    let mut config = SgdConfig::default();
    config.momentum = -0.1;
    let result = Sgd::new(vec![x], config);
    assert!(result.is_err(), "SGD should reject negative momentum");
}

#[test]
fn test_adafactor_rejects_positive_decay_rate() {
    let x = scalar_var(1.0);
    let mut config = AdaFactorConfig::default();
    config.decay_rate = 0.8; // must be negative
    let result = AdaFactor::new(vec![x], config);
    assert!(
        result.is_err(),
        "AdaFactor should reject positive decay_rate"
    );
}

// ============================================================================
// Multi-variable optimization
// ============================================================================

#[test]
fn test_adam_multi_variable() {
    // Minimize f(x,y) = x^2 + y^2
    let x = scalar_var(3.0);
    let y = scalar_var(-4.0);
    let mut adam = AdamW::new(vec![x.clone(), y.clone()], adam_config(0.1, 0.0)).unwrap();

    for _ in 0..200 {
        let tx = Arc::new(TrackedTensor::from_var(&x).unwrap());
        let ty = Arc::new(TrackedTensor::from_var(&y).unwrap());
        let loss = tx.sqr().unwrap().add(&ty.sqr().unwrap()).unwrap();
        adam.backward_step(&loss).unwrap();
    }

    let vx = read_scalar(&x);
    let vy = read_scalar(&y);
    assert!(
        vx.abs() < 0.01 && vy.abs() < 0.01,
        "Both variables should converge near 0, got ({vx}, {vy})"
    );
}

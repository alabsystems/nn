// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Extended cross-crate tests for nn-optim optimizers.
//!
//! Covers: Adam momentum tracking + bias correction + quadratic convergence,
//! SGD momentum accumulation + dampening via weight decay,
//! AdaFactor scale-invariant updates + factored second moments,
//! learning rate scheduling, weight decay application,
//! gradient clipping, zero-gradient state initialization,
//! and multi-step convergence toward optimum.

use std::sync::Arc;

use nn_autodiff::{backward, TrackedTensor, Var};
use nn_core::dyn_tensor::DynTensor;
use nn_core::Device;
use nn_optim::{
    clip_grad_norm, clip_grad_value, step_with_schedule, AdaFactor, AdaFactorConfig, AdamConfig,
    AdamW, CosineSchedule, GradScaler, GradScalerConfig, LrSchedule, Optimizer, Sgd, SgdConfig,
    WarmupSchedule,
};

// -- Helpers -----------------------------------------------------------------

fn cpu() -> Device {
    Device::Cpu
}

fn scalar_var(val: f32) -> Var {
    Var::new(DynTensor::from_vec(vec![val], &[1], &cpu()).unwrap())
}

fn vec_var(vals: &[f32]) -> Var {
    let n = vals.len();
    Var::new(DynTensor::from_vec(vals.to_vec(), &[n], &cpu()).unwrap())
}

fn mat_var(vals: &[f32], rows: usize, cols: usize) -> Var {
    Var::new(DynTensor::from_vec(vals.to_vec(), &[rows, cols], &cpu()).unwrap())
}

fn read_scalar(var: &Var) -> f32 {
    var.data().unwrap().to_flat_vec::<f32>().unwrap()[0]
}

fn read_vec(var: &Var) -> Vec<f32> {
    var.data().unwrap().to_flat_vec::<f32>().unwrap()
}

// Config builder helpers: use Default + field mutation for #[non_exhaustive].

fn adam_cfg(lr: f64, wd: f64) -> AdamConfig {
    let mut c = AdamConfig::default();
    c.lr = lr;
    c.weight_decay = wd;
    c
}

fn adam_cfg_betas(lr: f64, beta1: f64, beta2: f64, wd: f64) -> AdamConfig {
    let mut c = AdamConfig::default();
    c.lr = lr;
    c.beta1 = beta1;
    c.beta2 = beta2;
    c.weight_decay = wd;
    c
}

fn sgd_cfg(lr: f64, momentum: f64, wd: f64) -> SgdConfig {
    let mut c = SgdConfig::default();
    c.lr = lr;
    c.momentum = momentum;
    c.weight_decay = wd;
    c
}

fn adafactor_cfg(lr: f64, wd: f64) -> AdaFactorConfig {
    let mut c = AdaFactorConfig::default();
    c.lr = lr;
    c.relative_step = false;
    c.weight_decay = wd;
    c
}

fn adafactor_cfg_beta1(lr: f64, beta1: Option<f64>, wd: f64) -> AdaFactorConfig {
    let mut c = AdaFactorConfig::default();
    c.lr = lr;
    c.relative_step = false;
    c.beta1 = beta1;
    c.weight_decay = wd;
    c
}

fn adafactor_cfg_relative(wd: f64) -> AdaFactorConfig {
    let mut c = AdaFactorConfig::default();
    c.relative_step = true;
    c.weight_decay = wd;
    c
}

fn grad_scaler_cfg(init_scale: f64, growth_interval: usize) -> GradScalerConfig {
    let mut c = GradScalerConfig::default();
    c.init_scale = init_scale;
    c.growth_interval = growth_interval;
    c
}

/// Build loss = x^2 (scalar) and run backward_step.
fn step_quadratic(opt: &mut dyn Optimizer, var: &Var) {
    let t = Arc::new(TrackedTensor::from_var(var).unwrap());
    let loss = t.sqr().unwrap();
    opt.backward_step(&loss).unwrap();
}

/// Build loss = sum(x_i^2) for a rank-1 var and run backward_step.
fn step_quadratic_vec(opt: &mut dyn Optimizer, var: &Var) {
    let t = Arc::new(TrackedTensor::from_var(var).unwrap());
    let sq = t.sqr().unwrap();
    // Reduce rank-1 [N] to scalar [1] via sum_keepdim(0)
    let loss = sq.sum_keepdim(0).unwrap();
    opt.backward_step(&loss).unwrap();
}

/// Build loss = sum(X_{ij}^2) for a rank-2 var and run backward_step.
fn step_quadratic_mat(opt: &mut dyn Optimizer, var: &Var) {
    let t = Arc::new(TrackedTensor::from_var(var).unwrap());
    let sq = t.sqr().unwrap();
    // Reduce [R, C] -> [1, C] -> [1, 1] = scalar
    let loss = sq.sum_keepdim(0).unwrap().sum_keepdim(1).unwrap();
    opt.backward_step(&loss).unwrap();
}

// ============================================================================
// Adam: momentum tracking and bias correction
// ============================================================================

#[test]
fn test_adam_ext_momentum_consistent_gradient_direction() {
    // With consistent gradient direction, Adam's first moment should build up,
    // accelerating convergence. Starting from x=10, after many steps the
    // parameter should be much closer to 0 than a single step.
    let x = scalar_var(10.0);
    let mut adam = AdamW::new(vec![x.clone()], adam_cfg(0.1, 0.0)).unwrap();

    let mut prev = 10.0f32;
    // Adam moves ~lr (0.1) per step on this quadratic, so reaching |x| < 0.1
    // from x=10 needs ~250 steps; 50 steps only reaches ~5.37. The optimizer is
    // correct — the step budget was too small. |x| stays monotonically
    // decreasing throughout for this 1D problem at this lr.
    let total_steps = 300;
    for i in 0..total_steps {
        step_quadratic(&mut adam, &x);
        let val = read_scalar(&x);
        if i > 0 {
            assert!(
                val.abs() <= prev.abs() + 1e-6,
                "step {i}: |x| should decrease: prev={prev}, now={val}"
            );
        }
        prev = val;
    }
    assert!(
        prev.abs() < 0.1,
        "after {total_steps} Adam steps, |x| should be < 0.1, got {prev}"
    );
}

#[test]
fn test_adam_ext_bias_correction_first_step() {
    // Bias correction should amplify the first few steps (when moments are
    // biased toward zero). Both high and low beta1 should make meaningful steps.
    let x_high = scalar_var(5.0);
    let x_low = scalar_var(5.0);

    let mut adam_high =
        AdamW::new(vec![x_high.clone()], adam_cfg_betas(0.01, 0.99, 0.999, 0.0)).unwrap();
    let mut adam_low =
        AdamW::new(vec![x_low.clone()], adam_cfg_betas(0.01, 0.1, 0.999, 0.0)).unwrap();

    step_quadratic(&mut adam_high, &x_high);
    step_quadratic(&mut adam_low, &x_low);

    let val_high = read_scalar(&x_high);
    let val_low = read_scalar(&x_low);

    assert!(val_high < 5.0);
    assert!(val_low < 5.0);
    // With bias correction, both should make meaningful steps
    assert!(
        (val_high - val_low).abs() < 1.0,
        "bias correction prevents massive step size difference: high={val_high}, low={val_low}"
    );
}

#[test]
fn test_adam_ext_step_count_increments() {
    let x = scalar_var(1.0);
    let mut adam = AdamW::new(vec![x.clone()], adam_cfg(0.01, 0.0)).unwrap();
    assert_eq!(adam.step_count(), 0);

    for expected in 1..=10 {
        step_quadratic(&mut adam, &x);
        assert_eq!(adam.step_count(), expected);
    }
}

#[test]
fn test_adam_ext_convergence_quadratic_multi_dim() {
    let x = vec_var(&[3.0, -2.0, 5.0, -1.0]);
    let mut adam = AdamW::new(vec![x.clone()], adam_cfg(0.1, 0.0)).unwrap();

    for _ in 0..200 {
        step_quadratic_vec(&mut adam, &x);
    }

    let vals = read_vec(&x);
    for (i, v) in vals.iter().enumerate() {
        assert!(v.abs() < 0.05, "dim {i}: expected near 0, got {v}");
    }
}

#[test]
fn test_adam_ext_config_accessors() {
    let x = scalar_var(1.0);
    let config = adam_cfg_betas(2e-4, 0.95, 0.98, 0.05);
    let adam = AdamW::new(vec![x], config).unwrap();
    let c = adam.config();
    assert!((c.lr - 2e-4).abs() < f64::EPSILON);
    assert!((c.beta1 - 0.95).abs() < f64::EPSILON);
    assert!((c.beta2 - 0.98).abs() < f64::EPSILON);
    assert!((c.weight_decay - 0.05).abs() < f64::EPSILON);
}

// ============================================================================
// SGD: gradient accumulation and momentum
// ============================================================================

#[test]
fn test_sgd_ext_momentum_accumulation() {
    // With momentum, velocity should build up, leading to larger effective
    // steps than without momentum.
    let x_mom = scalar_var(10.0);
    let x_plain = scalar_var(10.0);

    let mut sgd_mom = Sgd::new(vec![x_mom.clone()], sgd_cfg(0.01, 0.9, 0.0)).unwrap();
    let mut sgd_plain = Sgd::new(vec![x_plain.clone()], sgd_cfg(0.01, 0.0, 0.0)).unwrap();

    for _ in 0..20 {
        step_quadratic(&mut sgd_mom, &x_mom);
        step_quadratic(&mut sgd_plain, &x_plain);
    }

    let val_mom = read_scalar(&x_mom);
    let val_plain = read_scalar(&x_plain);

    assert!(
        val_mom.abs() < val_plain.abs(),
        "momentum SGD should converge faster: mom={val_mom}, plain={val_plain}"
    );
}

#[test]
fn test_sgd_ext_momentum_velocity_buildup() {
    // Consecutive steps with same gradient direction should cause velocity
    // to accumulate, making each step progressively larger.
    let x = scalar_var(5.0);
    let mut sgd = Sgd::new(vec![x.clone()], sgd_cfg(0.01, 0.9, 0.0)).unwrap();

    let mut step_sizes = Vec::new();
    let mut prev_val = 5.0f32;

    for _ in 0..10 {
        step_quadratic(&mut sgd, &x);
        let val = read_scalar(&x);
        step_sizes.push((prev_val - val).abs());
        prev_val = val;
    }

    // Early steps should be smaller than later steps (velocity building up)
    assert!(
        step_sizes[4] > step_sizes[0],
        "step size should grow with momentum: step[0]={}, step[4]={}",
        step_sizes[0],
        step_sizes[4]
    );
}

#[test]
fn test_sgd_ext_weight_decay_via_gradient() {
    // SGD applies weight decay as L2 regularization: grad += wd * theta.
    // loss = x^2, gradient = 2x. With wd=0.1: effective grad = 2*1 + 0.1*1 = 2.1
    // new = 1.0 - 0.1*2.1 = 0.79
    let x = scalar_var(1.0);
    let mut sgd = Sgd::new(vec![x.clone()], sgd_cfg(0.1, 0.0, 0.1)).unwrap();

    step_quadratic(&mut sgd, &x);
    let val = read_scalar(&x);
    assert!(
        (val - 0.79).abs() < 1e-5,
        "expected ~0.79 with weight decay, got {val}"
    );
}

#[test]
fn test_sgd_ext_convergence_multi_dim() {
    let x = vec_var(&[4.0, -3.0, 2.0]);
    let mut sgd = Sgd::new(vec![x.clone()], sgd_cfg(0.01, 0.9, 0.0)).unwrap();

    for _ in 0..500 {
        step_quadratic_vec(&mut sgd, &x);
    }

    let vals = read_vec(&x);
    for (i, v) in vals.iter().enumerate() {
        assert!(v.abs() < 0.05, "SGD dim {i}: expected near 0, got {v}");
    }
}

#[test]
fn test_sgd_ext_no_momentum_exact_update() {
    // Without momentum: theta_new = theta - lr * grad
    // loss = x^2, grad = 2x. At x=3: grad=6, theta_new = 3 - 0.05*6 = 2.7
    let x = scalar_var(3.0);
    let mut sgd = Sgd::new(vec![x.clone()], sgd_cfg(0.05, 0.0, 0.0)).unwrap();

    step_quadratic(&mut sgd, &x);
    let val = read_scalar(&x);
    assert!((val - 2.7).abs() < 1e-5, "expected 2.7, got {val}");
}

// ============================================================================
// AdaFactor: scale-invariant updates and factored second moments
// ============================================================================

#[test]
fn test_adafactor_ext_convergence_scalar() {
    let x = scalar_var(5.0);
    let mut opt = AdaFactor::new(vec![x.clone()], adafactor_cfg(0.1, 0.0)).unwrap();

    for _ in 0..200 {
        step_quadratic(&mut opt, &x);
    }

    let val = read_scalar(&x);
    assert!(
        val.abs() < 0.5,
        "AdaFactor should converge toward 0, got {val}"
    );
}

#[test]
fn test_adafactor_ext_factored_moments_matrix() {
    // For rank-2 tensors, AdaFactor should use factored second moments.
    let x = mat_var(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 2, 3);
    let mut opt = AdaFactor::new(vec![x.clone()], adafactor_cfg(0.05, 0.0)).unwrap();

    for _ in 0..300 {
        step_quadratic_mat(&mut opt, &x);
    }

    let vals = read_vec(&x);
    let max_abs = vals.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
    assert!(
        max_abs < 1.0,
        "AdaFactor should reduce matrix elements toward 0, max_abs={max_abs}"
    );
}

#[test]
fn test_adafactor_ext_relative_step_mode() {
    // In relative_step mode, large parameters should get proportionally
    // larger updates.
    let x_big = scalar_var(100.0);
    let x_small = scalar_var(0.1);

    let mut opt_big = AdaFactor::new(vec![x_big.clone()], adafactor_cfg_relative(0.0)).unwrap();
    let mut opt_small = AdaFactor::new(vec![x_small.clone()], adafactor_cfg_relative(0.0)).unwrap();

    step_quadratic(&mut opt_big, &x_big);
    step_quadratic(&mut opt_small, &x_small);

    let change_big = (100.0 - read_scalar(&x_big)).abs();
    let change_small = (0.1 - read_scalar(&x_small)).abs();

    assert!(
        change_big > change_small,
        "relative step should scale updates: big={change_big}, small={change_small}"
    );
}

#[test]
fn test_adafactor_ext_with_momentum() {
    let x = scalar_var(5.0);
    let mut opt =
        AdaFactor::new(vec![x.clone()], adafactor_cfg_beta1(0.1, Some(0.9), 0.0)).unwrap();

    for _ in 0..100 {
        step_quadratic(&mut opt, &x);
    }

    let val = read_scalar(&x);
    assert!(
        val.abs() < 1.0,
        "AdaFactor with momentum should converge, got {val}"
    );
}

#[test]
fn test_adafactor_ext_weight_decay() {
    let x_wd = scalar_var(5.0);
    let x_no_wd = scalar_var(5.0);

    let mut opt_wd = AdaFactor::new(vec![x_wd.clone()], adafactor_cfg(0.05, 0.1)).unwrap();
    let mut opt_no_wd = AdaFactor::new(vec![x_no_wd.clone()], adafactor_cfg(0.05, 0.0)).unwrap();

    for _ in 0..50 {
        step_quadratic(&mut opt_wd, &x_wd);
        step_quadratic(&mut opt_no_wd, &x_no_wd);
    }

    let val_wd = read_scalar(&x_wd);
    let val_no_wd = read_scalar(&x_no_wd);

    assert!(
        val_wd.abs() <= val_no_wd.abs() + 0.1,
        "weight decay should help convergence: wd={val_wd}, no_wd={val_no_wd}"
    );
}

#[test]
fn test_adafactor_ext_step_count() {
    let x = scalar_var(1.0);
    let mut opt = AdaFactor::new(vec![x.clone()], adafactor_cfg(0.01, 0.0)).unwrap();
    assert_eq!(opt.step_count(), 0);

    step_quadratic(&mut opt, &x);
    assert_eq!(opt.step_count(), 1);

    step_quadratic(&mut opt, &x);
    assert_eq!(opt.step_count(), 2);
}

// ============================================================================
// Learning rate scheduling
// ============================================================================

#[test]
fn test_ext_warmup_schedule_linear_ramp() {
    let schedule = WarmupSchedule::new(1e-3, 100).unwrap();

    assert!((schedule.lr_at_step(0) - 0.0).abs() < 1e-10);
    assert!((schedule.lr_at_step(50) - 5e-4).abs() < 1e-10);
    assert!((schedule.lr_at_step(100) - 1e-3).abs() < 1e-10);
    assert!((schedule.lr_at_step(200) - 1e-3).abs() < 1e-10);
}

#[test]
fn test_ext_warmup_schedule_zero_warmup_steps() {
    let schedule = WarmupSchedule::new(1e-3, 0).unwrap();
    assert!((schedule.lr_at_step(0) - 1e-3).abs() < 1e-10);
    assert!((schedule.lr_at_step(100) - 1e-3).abs() < 1e-10);
}

#[test]
fn test_ext_warmup_schedule_accessors() {
    let schedule = WarmupSchedule::new(2e-3, 50).unwrap();
    assert!((schedule.base_lr() - 2e-3).abs() < f64::EPSILON);
    assert_eq!(schedule.warmup_steps(), 50);
}

#[test]
fn test_ext_cosine_schedule_shape() {
    let schedule = CosineSchedule::new(1e-3, 1e-5, 10, 100).unwrap();

    // At warmup boundary
    let lr_at_warmup = schedule.lr_at_step(10);
    assert!(
        (lr_at_warmup - 1e-3).abs() < 1e-8,
        "at warmup end: expected 1e-3, got {lr_at_warmup}"
    );

    // At total_steps
    let lr_at_end = schedule.lr_at_step(100);
    assert!(
        (lr_at_end - 1e-5).abs() < 1e-8,
        "at end: expected 1e-5, got {lr_at_end}"
    );

    // At midpoint of cosine decay: (base + min) / 2
    let midpoint = 10 + (100 - 10) / 2;
    let lr_mid = schedule.lr_at_step(midpoint);
    let expected_mid = f64::midpoint(1e-3, 1e-5);
    assert!(
        (lr_mid - expected_mid).abs() < 1e-6,
        "at midpoint: expected ~{expected_mid}, got {lr_mid}"
    );

    // Past total_steps: clamps to min_lr
    assert!((schedule.lr_at_step(200) - 1e-5).abs() < 1e-8);
}

#[test]
fn test_ext_cosine_schedule_warmup_phase() {
    let schedule = CosineSchedule::new(1e-3, 0.0, 10, 100).unwrap();

    let lr_0 = schedule.lr_at_step(0);
    assert!(lr_0.abs() < 1e-10, "step 0 should be ~0, got {lr_0}");

    let lr_5 = schedule.lr_at_step(5);
    assert!(
        (lr_5 - 5e-4).abs() < 1e-8,
        "step 5 should be 5e-4, got {lr_5}"
    );
}

#[test]
fn test_ext_cosine_schedule_monotone_decreasing() {
    let schedule = CosineSchedule::new(0.01, 0.001, 0, 100).unwrap();

    let mut prev_lr = schedule.lr_at_step(0);
    for step in 1..=100 {
        let lr = schedule.lr_at_step(step);
        assert!(
            lr <= prev_lr + 1e-10,
            "step {step}: lr should decrease: prev={prev_lr}, now={lr}"
        );
        prev_lr = lr;
    }
}

#[test]
fn test_ext_cosine_schedule_accessors() {
    let schedule = CosineSchedule::new(1e-3, 1e-5, 10, 1000).unwrap();
    assert!((schedule.base_lr() - 1e-3).abs() < f64::EPSILON);
    assert!((schedule.min_lr() - 1e-5).abs() < f64::EPSILON);
    assert_eq!(schedule.total_steps(), 1000);
}

#[test]
fn test_ext_step_with_schedule_integration() {
    let x = scalar_var(5.0);
    let mut adam = AdamW::new(vec![x.clone()], adam_cfg(0.0, 0.0)).unwrap();
    let schedule = WarmupSchedule::new(0.1, 10).unwrap();

    let t = Arc::new(TrackedTensor::from_var(&x).unwrap());
    let loss = t.sqr().unwrap();
    let grads = backward(&loss).unwrap();

    step_with_schedule(&mut adam, &grads, &schedule, 5).unwrap();

    let expected_lr = schedule.lr_at_step(5);
    assert!(
        (adam.learning_rate() - expected_lr).abs() < 1e-10,
        "lr should match schedule: expected {expected_lr}, got {}",
        adam.learning_rate()
    );
    let val = read_scalar(&x);
    assert!(val < 5.0, "parameter should have decreased, got {val}");
}

// ============================================================================
// Weight decay application
// ============================================================================

#[test]
fn test_ext_adam_weight_decay_decoupled() {
    // AdamW uses decoupled weight decay: theta *= (1 - lr * wd) BEFORE update.
    let x_wd = scalar_var(5.0);
    let x_no_wd = scalar_var(5.0);

    let mut adam_wd = AdamW::new(vec![x_wd.clone()], adam_cfg(0.01, 0.1)).unwrap();
    let mut adam_no_wd = AdamW::new(vec![x_no_wd.clone()], adam_cfg(0.01, 0.0)).unwrap();

    for _ in 0..10 {
        step_quadratic(&mut adam_wd, &x_wd);
        step_quadratic(&mut adam_no_wd, &x_no_wd);
    }

    let val_wd = read_scalar(&x_wd);
    let val_no_wd = read_scalar(&x_no_wd);

    assert!(
        val_wd.abs() < val_no_wd.abs(),
        "weight decay should shrink params: wd={val_wd}, no_wd={val_no_wd}"
    );
}

#[test]
fn test_ext_sgd_weight_decay_l2_regularization() {
    let x_wd = scalar_var(3.0);
    let x_no_wd = scalar_var(3.0);

    let mut sgd_wd = Sgd::new(vec![x_wd.clone()], sgd_cfg(0.01, 0.0, 0.1)).unwrap();
    let mut sgd_no_wd = Sgd::new(vec![x_no_wd.clone()], sgd_cfg(0.01, 0.0, 0.0)).unwrap();

    for _ in 0..100 {
        step_quadratic(&mut sgd_wd, &x_wd);
        step_quadratic(&mut sgd_no_wd, &x_no_wd);
    }

    let val_wd = read_scalar(&x_wd);
    let val_no_wd = read_scalar(&x_no_wd);

    assert!(
        val_wd.abs() <= val_no_wd.abs() + 1e-5,
        "weight decay should help converge: wd={val_wd}, no_wd={val_no_wd}"
    );
}

// ============================================================================
// Gradient clipping
// ============================================================================

#[test]
fn test_ext_clip_grad_norm_scales_down() {
    let x = scalar_var(50.0);
    let t = Arc::new(TrackedTensor::from_var(&x).unwrap());
    let loss = t.sqr().unwrap(); // grad = 2*50 = 100
    let mut grads = backward(&loss).unwrap();

    let total_norm = clip_grad_norm(&mut grads, 1.0).unwrap();
    assert!(
        total_norm > 1.0,
        "original norm should be > 1.0, got {total_norm}"
    );

    let clipped_grad = grads.get(&x).unwrap();
    let clipped_norm_sq: f32 = clipped_grad
        .sqr()
        .unwrap()
        .sum_all()
        .unwrap()
        .to_scalar()
        .unwrap();
    let clipped_norm = f64::from(clipped_norm_sq).sqrt();
    assert!(
        (clipped_norm - 1.0).abs() < 1e-4,
        "clipped norm should be 1.0, got {clipped_norm}"
    );
}

#[test]
fn test_ext_clip_grad_norm_no_change_when_below() {
    let x = scalar_var(0.1);
    let t = Arc::new(TrackedTensor::from_var(&x).unwrap());
    let loss = t.sqr().unwrap(); // grad = 0.2
    let mut grads = backward(&loss).unwrap();

    let grad_before: f32 = grads.get(&x).unwrap().to_flat_vec::<f32>().unwrap()[0];

    let total_norm = clip_grad_norm(&mut grads, 100.0).unwrap();
    assert!(total_norm < 100.0);

    let grad_after: f32 = grads.get(&x).unwrap().to_flat_vec::<f32>().unwrap()[0];

    assert!(
        (grad_before - grad_after).abs() < 1e-7,
        "gradient should not change: before={grad_before}, after={grad_after}"
    );
}

#[test]
fn test_ext_clip_grad_value_clamps_elements() {
    let x = vec_var(&[10.0, -10.0, 0.5]);
    let t = Arc::new(TrackedTensor::from_var(&x).unwrap());
    // loss = sum(x_i^2), grads = [20, -20, 1]
    let sq = t.sqr().unwrap();
    let loss = sq.sum_keepdim(0).unwrap();
    let mut grads = backward(&loss).unwrap();

    clip_grad_value(&mut grads, 5.0).unwrap();

    let clipped: Vec<f32> = grads.get(&x).unwrap().to_flat_vec::<f32>().unwrap();

    assert!(clipped[0] <= 5.0 + 1e-6, "positive clamped: {}", clipped[0]);
    assert!(
        clipped[1] >= -5.0 - 1e-6,
        "negative clamped: {}",
        clipped[1]
    );
    assert!(
        (clipped[2] - 1.0).abs() < 1e-5,
        "small unchanged: {}",
        clipped[2]
    );
}

#[test]
fn test_ext_clip_grad_norm_invalid_max_norm() {
    let x = scalar_var(1.0);
    let t = Arc::new(TrackedTensor::from_var(&x).unwrap());
    let loss = t.sqr().unwrap();
    let mut grads = backward(&loss).unwrap();

    assert!(clip_grad_norm(&mut grads, 0.0).is_err());
    assert!(clip_grad_norm(&mut grads, -1.0).is_err());
    assert!(clip_grad_norm(&mut grads, f64::NAN).is_err());
    assert!(clip_grad_norm(&mut grads, f64::INFINITY).is_err());
}

#[test]
fn test_ext_clip_grad_value_invalid_clip_value() {
    let x = scalar_var(1.0);
    let t = Arc::new(TrackedTensor::from_var(&x).unwrap());
    let loss = t.sqr().unwrap();
    let mut grads = backward(&loss).unwrap();

    assert!(clip_grad_value(&mut grads, 0.0).is_err());
    assert!(clip_grad_value(&mut grads, -1.0).is_err());
    assert!(clip_grad_value(&mut grads, f64::NAN).is_err());
}

// ============================================================================
// Zero gradient / initial state
// ============================================================================

#[test]
fn test_ext_adam_zero_gradient_initial_state() {
    // When a variable has no gradient flowing to it, it should be unchanged.
    let x = scalar_var(5.0);
    let y = scalar_var(3.0);
    let mut adam = AdamW::new(vec![x.clone(), y.clone()], adam_cfg(0.1, 0.0)).unwrap();

    // Loss only depends on x
    let t = Arc::new(TrackedTensor::from_var(&x).unwrap());
    let loss = t.sqr().unwrap();
    adam.backward_step(&loss).unwrap();

    let x_val = read_scalar(&x);
    let y_val = read_scalar(&y);
    assert!(x_val < 5.0, "x should decrease");
    assert!(
        (y_val - 3.0).abs() < 1e-7,
        "y should be unchanged, got {y_val}"
    );
}

#[test]
fn test_ext_sgd_zero_gradient_parameter_unchanged() {
    let x = scalar_var(5.0);
    let y = scalar_var(3.0);
    let mut sgd = Sgd::new(vec![x.clone(), y.clone()], sgd_cfg(0.1, 0.9, 0.0)).unwrap();

    let t = Arc::new(TrackedTensor::from_var(&x).unwrap());
    let loss = t.sqr().unwrap();
    sgd.backward_step(&loss).unwrap();

    let y_val = read_scalar(&y);
    assert!(
        (y_val - 3.0).abs() < 1e-7,
        "y without gradient should be unchanged: {y_val}"
    );
}

#[test]
fn test_ext_adafactor_zero_gradient_parameter_unchanged() {
    let x = scalar_var(5.0);
    let y = scalar_var(3.0);
    let mut opt = AdaFactor::new(vec![x.clone(), y.clone()], adafactor_cfg(0.1, 0.0)).unwrap();

    let t = Arc::new(TrackedTensor::from_var(&x).unwrap());
    let loss = t.sqr().unwrap();
    opt.backward_step(&loss).unwrap();

    let y_val = read_scalar(&y);
    assert!(
        (y_val - 3.0).abs() < 1e-7,
        "AdaFactor: y should be unchanged: {y_val}"
    );
}

// ============================================================================
// Multi-step convergence toward optimum
// ============================================================================

#[test]
fn test_ext_adam_multi_var_convergence() {
    // Two variables minimizing f(x,y) = x^2 + y^2.
    let x = scalar_var(5.0);
    let y = scalar_var(-3.0);
    let mut adam = AdamW::new(vec![x.clone(), y.clone()], adam_cfg(0.1, 0.0)).unwrap();

    for _ in 0..200 {
        let tx = Arc::new(TrackedTensor::from_var(&x).unwrap());
        let ty = Arc::new(TrackedTensor::from_var(&y).unwrap());
        let loss = tx.sqr().unwrap().add(&ty.sqr().unwrap()).unwrap();
        adam.backward_step(&loss).unwrap();
    }

    let xv = read_scalar(&x);
    let yv = read_scalar(&y);
    assert!(xv.abs() < 0.05, "x should converge to 0, got {xv}");
    assert!(yv.abs() < 0.05, "y should converge to 0, got {yv}");
}

#[test]
fn test_ext_sgd_multi_step_monotone_decrease() {
    let x = scalar_var(10.0);
    let mut sgd = Sgd::new(vec![x.clone()], sgd_cfg(0.01, 0.0, 0.0)).unwrap();

    let mut prev_loss = 100.0f32;
    for i in 0..50 {
        step_quadratic(&mut sgd, &x);
        let val = read_scalar(&x);
        let loss = val * val;
        assert!(
            loss <= prev_loss + 1e-6,
            "step {i}: loss should decrease: prev={prev_loss}, now={loss}"
        );
        prev_loss = loss;
    }
}

// ============================================================================
// Parameter validation
// ============================================================================

#[test]
fn test_ext_adam_invalid_params() {
    let x = scalar_var(1.0);

    // Negative lr
    let mut c = AdamConfig::default();
    c.lr = -0.01;
    assert!(AdamW::new(vec![x.clone()], c).is_err());

    // beta1 >= 1.0
    let mut c = AdamConfig::default();
    c.beta1 = 1.0;
    assert!(AdamW::new(vec![x.clone()], c).is_err());

    // beta2 >= 1.0
    let mut c = AdamConfig::default();
    c.beta2 = 1.0;
    assert!(AdamW::new(vec![x.clone()], c).is_err());

    // eps <= 0
    let mut c = AdamConfig::default();
    c.eps = 0.0;
    assert!(AdamW::new(vec![x.clone()], c).is_err());

    // Negative weight decay
    let mut c = AdamConfig::default();
    c.weight_decay = -0.01;
    assert!(AdamW::new(vec![x.clone()], c).is_err());

    // NaN lr
    let mut c = AdamConfig::default();
    c.lr = f64::NAN;
    assert!(AdamW::new(vec![x], c).is_err());
}

#[test]
fn test_ext_sgd_invalid_params() {
    let x = scalar_var(1.0);

    // Negative momentum
    let mut c = SgdConfig::default();
    c.momentum = -0.1;
    assert!(Sgd::new(vec![x.clone()], c).is_err());

    // Negative lr
    let mut c = SgdConfig::default();
    c.lr = -0.01;
    assert!(Sgd::new(vec![x.clone()], c).is_err());

    // Negative weight decay
    let mut c = SgdConfig::default();
    c.weight_decay = -0.01;
    assert!(Sgd::new(vec![x.clone()], c).is_err());

    // NaN momentum
    let mut c = SgdConfig::default();
    c.momentum = f64::NAN;
    assert!(Sgd::new(vec![x], c).is_err());
}

#[test]
fn test_ext_adafactor_invalid_params() {
    let x = scalar_var(1.0);

    // Positive decay_rate
    let mut c = AdaFactorConfig::default();
    c.decay_rate = 0.8;
    assert!(AdaFactor::new(vec![x.clone()], c).is_err());

    // beta1 >= 1.0
    let mut c = AdaFactorConfig::default();
    c.beta1 = Some(1.0);
    assert!(AdaFactor::new(vec![x.clone()], c).is_err());

    // eps_denom <= 0
    let mut c = AdaFactorConfig::default();
    c.eps_denom = 0.0;
    assert!(AdaFactor::new(vec![x.clone()], c).is_err());

    // eps_rms <= 0
    let mut c = AdaFactorConfig::default();
    c.eps_rms = -1e-3;
    assert!(AdaFactor::new(vec![x], c).is_err());
}

// ============================================================================
// set_learning_rate
// ============================================================================

#[test]
fn test_ext_adam_set_learning_rate() {
    let x = scalar_var(1.0);
    let mut adam = AdamW::new(vec![x], AdamConfig::default()).unwrap();

    adam.set_learning_rate(0.05).unwrap();
    assert!((adam.learning_rate() - 0.05).abs() < f64::EPSILON);

    assert!(adam.set_learning_rate(-0.01).is_err());
    assert!(adam.set_learning_rate(f64::NAN).is_err());
    assert!(adam.set_learning_rate(f64::INFINITY).is_err());
}

#[test]
fn test_ext_sgd_set_learning_rate() {
    let x = scalar_var(1.0);
    let mut sgd = Sgd::new(vec![x], SgdConfig::default()).unwrap();

    sgd.set_learning_rate(0.05).unwrap();
    assert!((sgd.learning_rate() - 0.05).abs() < f64::EPSILON);

    assert!(sgd.set_learning_rate(-0.01).is_err());
}

#[test]
fn test_ext_adafactor_set_learning_rate() {
    let x = scalar_var(1.0);
    let mut opt = AdaFactor::new(vec![x], AdaFactorConfig::default()).unwrap();

    opt.set_learning_rate(0.05).unwrap();
    assert!((opt.learning_rate() - 0.05).abs() < f64::EPSILON);

    assert!(opt.set_learning_rate(-0.01).is_err());
}

// ============================================================================
// LR schedule validation
// ============================================================================

#[test]
fn test_ext_warmup_schedule_invalid_base_lr() {
    assert!(WarmupSchedule::new(-1e-3, 100).is_err());
    assert!(WarmupSchedule::new(f64::NAN, 100).is_err());
    assert!(WarmupSchedule::new(f64::INFINITY, 100).is_err());
}

#[test]
fn test_ext_cosine_schedule_invalid_params() {
    assert!(CosineSchedule::new(1e-3, 0.0, 0, 0).is_err()); // total = 0
    assert!(CosineSchedule::new(1e-3, 0.0, 100, 100).is_err()); // warmup >= total
    assert!(CosineSchedule::new(1e-3, 1e-2, 0, 100).is_err()); // min > base
    assert!(CosineSchedule::new(-1e-3, 0.0, 0, 100).is_err()); // negative base
    assert!(CosineSchedule::new(1e-3, -1.0, 0, 100).is_err()); // negative min
}

// ============================================================================
// GradScaler basic behavior
// ============================================================================

#[test]
fn test_ext_grad_scaler_default_config() {
    let scaler = GradScaler::new(GradScalerConfig::default()).unwrap();
    assert!((scaler.scale_factor() - 65536.0).abs() < f64::EPSILON);
    assert!(!scaler.found_inf());
}

#[test]
fn test_ext_grad_scaler_scale_and_unscale() {
    let mut scaler = GradScaler::new(GradScalerConfig::default()).unwrap();
    let x = scalar_var(2.0);

    let t = Arc::new(TrackedTensor::from_var(&x).unwrap());
    let loss = t.sqr().unwrap(); // loss = 4
    let scaled_loss = scaler.scale_loss(&loss).unwrap();

    let mut grads = backward(&scaled_loss).unwrap();

    let ok = scaler.unscale_and_check(&mut grads).unwrap();
    assert!(ok, "unscale should succeed for finite gradients");
    assert!(!scaler.found_inf());

    let grad_val: f32 = grads.get(&x).unwrap().to_flat_vec::<f32>().unwrap()[0];
    assert!(
        (grad_val - 4.0).abs() < 0.01,
        "unscaled gradient should be ~4.0, got {grad_val}"
    );
}

#[test]
fn test_ext_grad_scaler_growth_on_clean_steps() {
    let mut scaler = GradScaler::new(grad_scaler_cfg(1.0, 3)).unwrap();

    for _ in 0..3 {
        scaler.update();
    }

    assert!(
        (scaler.scale_factor() - 2.0).abs() < f64::EPSILON,
        "scale should grow to 2.0 after 3 clean steps, got {}",
        scaler.scale_factor()
    );
}

#[test]
fn test_ext_grad_scaler_invalid_config() {
    // init_scale <= 0
    let mut c = GradScalerConfig::default();
    c.init_scale = 0.0;
    assert!(GradScaler::new(c).is_err());

    // growth_factor <= 1
    let mut c = GradScalerConfig::default();
    c.growth_factor = 1.0;
    assert!(GradScaler::new(c).is_err());

    // backoff_factor >= 1
    let mut c = GradScalerConfig::default();
    c.backoff_factor = 1.0;
    assert!(GradScaler::new(c).is_err());

    // growth_interval = 0
    let mut c = GradScalerConfig::default();
    c.growth_interval = 0;
    assert!(GradScaler::new(c).is_err());
}

// ============================================================================
// Adam: multi-step convergence on asymmetric quadratic
// ============================================================================

#[test]
fn test_ext_adam_convergence_asymmetric_quadratic() {
    // Minimize f(x, y) = 100*x^2 + y^2 (poorly conditioned).
    let x = scalar_var(1.0);
    let y = scalar_var(1.0);
    let mut adam = AdamW::new(vec![x.clone(), y.clone()], adam_cfg(0.01, 0.0)).unwrap();

    for _ in 0..500 {
        let tx = Arc::new(TrackedTensor::from_var(&x).unwrap());
        let ty = Arc::new(TrackedTensor::from_var(&y).unwrap());
        let loss_x = tx.sqr().unwrap().mul_scalar(100.0).unwrap();
        let loss_y = ty.sqr().unwrap();
        let loss = loss_x.add(&loss_y).unwrap();
        adam.backward_step(&loss).unwrap();
    }

    let xv = read_scalar(&x);
    let yv = read_scalar(&y);
    assert!(xv.abs() < 0.1, "x should converge, got {xv}");
    assert!(yv.abs() < 0.1, "y should converge, got {yv}");
}

// ============================================================================
// SGD + momentum + weight decay combined
// ============================================================================

#[test]
fn test_ext_sgd_momentum_and_weight_decay_combined() {
    let x = scalar_var(5.0);
    let mut sgd = Sgd::new(vec![x.clone()], sgd_cfg(0.01, 0.9, 0.01)).unwrap();

    for _ in 0..300 {
        step_quadratic(&mut sgd, &x);
    }

    let val = read_scalar(&x);
    assert!(
        val.abs() < 0.1,
        "SGD with momentum+wd should converge, got {val}"
    );
}

// ============================================================================
// Multi-variable different shapes
// ============================================================================

#[test]
fn test_ext_adam_different_shaped_vars() {
    let scalar = scalar_var(3.0);
    let vector = vec_var(&[1.0, -2.0, 3.0]);
    let matrix = mat_var(&[1.0, 2.0, 3.0, 4.0], 2, 2);

    // Adam moves ~lr per step, so at lr=0.01 the largest element (matrix 4.0)
    // only reaches ~3.05 after 100 steps. lr=0.05 drives every element below
    // 1.0 within the 100-step budget; the optimizer itself is correct.
    let mut adam = AdamW::new(
        vec![scalar.clone(), vector.clone(), matrix.clone()],
        adam_cfg(0.05, 0.0),
    )
    .unwrap();

    for _ in 0..100 {
        let ts = Arc::new(TrackedTensor::from_var(&scalar).unwrap());
        let tv = Arc::new(TrackedTensor::from_var(&vector).unwrap());
        let tm = Arc::new(TrackedTensor::from_var(&matrix).unwrap());
        // loss = scalar^2 + sum(vector^2) + sum(matrix^2)
        let v_loss = tv.sqr().unwrap().sum_keepdim(0).unwrap();
        let m_loss = tm
            .sqr()
            .unwrap()
            .sum_keepdim(0)
            .unwrap()
            .sum_keepdim(1)
            .unwrap();
        let loss = ts
            .sqr()
            .unwrap()
            .add(&v_loss)
            .unwrap()
            .add(&m_loss)
            .unwrap();
        adam.backward_step(&loss).unwrap();
    }

    let sv = read_scalar(&scalar);
    let vv = read_vec(&vector);
    let mv = read_vec(&matrix);

    assert!(sv.abs() < 1.0, "scalar should decrease, got {sv}");
    assert!(
        vv.iter().all(|v| v.abs() < 1.0),
        "vector elements should decrease: {vv:?}"
    );
    assert!(
        mv.iter().all(|v| v.abs() < 1.0),
        "matrix elements should decrease: {mv:?}"
    );
}

// ============================================================================
// Empty optimizer (no vars)
// ============================================================================

#[test]
fn test_ext_adam_empty_vars() {
    let mut adam = AdamW::new(vec![], AdamConfig::default()).unwrap();
    assert_eq!(adam.step_count(), 0);

    let grads = nn_autodiff::GradStore::new();
    adam.step(&grads).unwrap();
    assert_eq!(adam.step_count(), 1);
}

#[test]
fn test_ext_sgd_empty_vars() {
    let mut sgd = Sgd::new(vec![], SgdConfig::default()).unwrap();
    let grads = nn_autodiff::GradStore::new();
    sgd.step(&grads).unwrap();
}

#[test]
fn test_ext_adafactor_empty_vars() {
    let mut c = AdaFactorConfig::default();
    c.relative_step = false;
    let mut opt = AdaFactor::new(vec![], c).unwrap();
    let grads = nn_autodiff::GradStore::new();
    opt.step(&grads).unwrap();
    assert_eq!(opt.step_count(), 1);
}

// ============================================================================
// Gradient clipping + optimizer combined workflow
// ============================================================================

#[test]
fn test_ext_clip_then_optimize_workflow() {
    let x = scalar_var(100.0);
    let mut adam = AdamW::new(vec![x.clone()], adam_cfg(0.01, 0.0)).unwrap();

    for _ in 0..20 {
        let t = Arc::new(TrackedTensor::from_var(&x).unwrap());
        let loss = t.sqr().unwrap();
        let mut grads = backward(&loss).unwrap();

        let _total_norm = clip_grad_norm(&mut grads, 10.0).unwrap();
        adam.step(&grads).unwrap();
    }

    let val = read_scalar(&x);
    assert!(val < 100.0, "should decrease with clipped grads, got {val}");
}

// ============================================================================
// Schedule-driven training loop
// ============================================================================

#[test]
fn test_ext_cosine_schedule_driven_adam() {
    let x = scalar_var(5.0);
    let mut adam = AdamW::new(vec![x.clone()], adam_cfg(0.0, 0.0)).unwrap();
    let schedule = CosineSchedule::new(0.1, 0.001, 0, 100).unwrap();

    for step in 0..100 {
        let t = Arc::new(TrackedTensor::from_var(&x).unwrap());
        let loss = t.sqr().unwrap();
        let grads = backward(&loss).unwrap();
        step_with_schedule(&mut adam, &grads, &schedule, step).unwrap();
    }

    let val = read_scalar(&x);
    assert!(
        val.abs() < 1.0,
        "cosine-scheduled Adam should converge, got {val}"
    );
}

#[test]
fn test_ext_warmup_schedule_driven_sgd() {
    let x = scalar_var(5.0);
    let mut sgd = Sgd::new(vec![x.clone()], sgd_cfg(0.0, 0.9, 0.0)).unwrap();
    let schedule = WarmupSchedule::new(0.05, 10).unwrap();

    for step in 0..200 {
        let t = Arc::new(TrackedTensor::from_var(&x).unwrap());
        let loss = t.sqr().unwrap();
        let grads = backward(&loss).unwrap();
        step_with_schedule(&mut sgd, &grads, &schedule, step).unwrap();
    }

    let val = read_scalar(&x);
    assert!(
        val.abs() < 1.0,
        "warmup-scheduled SGD should converge, got {val}"
    );
}

// ============================================================================
// GradScaler save/load state
// ============================================================================

#[test]
fn test_ext_grad_scaler_save_load_roundtrip() {
    let mut scaler = GradScaler::new(grad_scaler_cfg(1024.0, 5)).unwrap();

    for _ in 0..3 {
        scaler.update();
    }

    let state = scaler.save_state();

    let mut scaler2 = GradScaler::new(grad_scaler_cfg(1024.0, 5)).unwrap();
    scaler2.load_state(&state).unwrap();

    assert!(
        (scaler2.scale_factor() - scaler.scale_factor()).abs() < f64::EPSILON,
        "restored scale should match"
    );
}

// ============================================================================
// Convergence speed comparison: Adam vs SGD
// ============================================================================

#[test]
fn test_ext_adam_converges_faster_than_plain_sgd() {
    let x_adam = scalar_var(5.0);
    let x_sgd = scalar_var(5.0);

    let mut adam = AdamW::new(vec![x_adam.clone()], adam_cfg(0.01, 0.0)).unwrap();
    let mut sgd = Sgd::new(vec![x_sgd.clone()], sgd_cfg(0.01, 0.0, 0.0)).unwrap();

    for _ in 0..100 {
        step_quadratic(&mut adam, &x_adam);
        step_quadratic(&mut sgd, &x_sgd);
    }

    let adam_val = read_scalar(&x_adam);
    let sgd_val = read_scalar(&x_sgd);

    assert!(
        adam_val.abs() < sgd_val.abs(),
        "Adam should converge faster: adam={adam_val}, sgd={sgd_val}"
    );
}

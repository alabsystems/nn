// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Part of #4557.
//!
//! Extended tests for learning rate schedulers and optimizer configuration.
//! Covers: warmup schedule edge cases, cosine annealing properties,
//! optimizer parameter groups, weight decay configuration, momentum tracking,
//! AdamW config validation, AdaFactor config validation, set_learning_rate
//! behavior, and schedule+optimizer integration scenarios.

use std::sync::Arc;

use crate::adafactor::{AdaFactor, AdaFactorConfig};
use crate::adam::{AdamConfig, AdamW};
use crate::error::OptimError;
use crate::lr_schedule::{step_with_schedule, CosineSchedule, LrSchedule, WarmupSchedule};
use crate::optimizer::Optimizer;
use crate::sgd::{Sgd, SgdConfig};
use nn_autodiff::{backward, TrackedTensor, Var};
use nn_core::dyn_tensor::DynTensor;
use nn_core::test_utils::cpu;

// ============================================================================
// Helpers
// ============================================================================

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

fn get_val(var: &Var) -> f32 {
    var.data().unwrap().to_flat_vec::<f32>().unwrap()[0]
}

fn get_vals(var: &Var) -> Vec<f32> {
    var.data().unwrap().to_flat_vec::<f32>().unwrap()
}

/// Build a simple gradient for a scalar variable via autodiff.
fn make_grad(var: &Var) -> nn_autodiff::GradStore {
    let t = Arc::new(TrackedTensor::from_var(var).unwrap());
    let loss = t.sqr().unwrap();
    backward(&loss).unwrap()
}

// ============================================================================
// WarmupSchedule — linearity and interpolation
// ============================================================================

#[test]
fn test_warmup_linearity_first_step() {
    let sched = WarmupSchedule::new(0.5, 200).unwrap();
    let lr = sched.lr_at_step(1);
    let expected = 0.5 * (1.0 / 200.0);
    assert!(
        (lr - expected).abs() < 1e-12,
        "first step: expected {expected}, got {lr}"
    );
}

#[test]
fn test_warmup_linearity_arbitrary_fractions() {
    let sched = WarmupSchedule::new(0.4, 400).unwrap();
    for &(step, frac) in &[(100, 0.25), (200, 0.5), (300, 0.75)] {
        let lr = sched.lr_at_step(step);
        let expected = 0.4 * frac;
        assert!(
            (lr - expected).abs() < 1e-12,
            "step {step}: expected {expected}, got {lr}"
        );
    }
}

#[test]
fn test_warmup_very_large_base_lr() {
    let sched = WarmupSchedule::new(1e10, 100).unwrap();
    let lr_mid = sched.lr_at_step(50);
    assert!(
        (lr_mid - 5e9).abs() / 5e9 < 1e-10,
        "large base_lr midpoint: expected 5e9, got {lr_mid}"
    );
    let lr_post = sched.lr_at_step(200);
    assert!(
        (lr_post - 1e10).abs() / 1e10 < 1e-10,
        "post-warmup: expected 1e10, got {lr_post}"
    );
}

#[test]
fn test_warmup_very_small_base_lr() {
    let sched = WarmupSchedule::new(1e-15, 10).unwrap();
    let lr = sched.lr_at_step(5);
    let expected = 1e-15 * 0.5;
    assert!(
        (lr - expected).abs() < 1e-25,
        "very small base_lr: expected {expected}, got {lr}"
    );
}

#[test]
fn test_warmup_single_warmup_step_boundary() {
    let sched = WarmupSchedule::new(0.2, 1).unwrap();
    assert!(
        sched.lr_at_step(0).abs() < f64::EPSILON,
        "step 0 before 1-step warmup"
    );
    let lr1 = sched.lr_at_step(1);
    assert!(
        (lr1 - 0.2).abs() < f64::EPSILON,
        "step 1 after 1-step warmup: expected 0.2, got {lr1}"
    );
}

#[test]
fn test_warmup_consecutive_steps_constant_increment() {
    let sched = WarmupSchedule::new(1.0, 1000).unwrap();
    let increment = 1.0 / 1000.0;
    for step in 1..10 {
        let lr = sched.lr_at_step(step);
        let expected = increment * step as f64;
        assert!(
            (lr - expected).abs() < 1e-12,
            "step {step}: expected {expected}, got {lr}"
        );
    }
}

#[test]
fn test_warmup_debug_format() {
    let sched = WarmupSchedule::new(0.01, 100).unwrap();
    let debug = format!("{sched:?}");
    assert!(
        debug.contains("WarmupSchedule"),
        "debug should contain type name: {debug}"
    );
}

// ============================================================================
// CosineSchedule — cosine curve properties
// ============================================================================

#[test]
fn test_cosine_lr_always_between_min_and_base() {
    let sched = CosineSchedule::new(0.1, 0.01, 0, 500).unwrap();
    for step in 0..600 {
        let lr = sched.lr_at_step(step);
        assert!(lr >= 0.01 - 1e-12, "step {step}: lr {lr} < min_lr 0.01");
        assert!(lr <= 0.1 + 1e-12, "step {step}: lr {lr} > base_lr 0.1");
    }
}

#[test]
fn test_cosine_with_warmup_lr_always_bounded() {
    let sched = CosineSchedule::new(0.05, 0.005, 100, 1000).unwrap();
    for step in 0..1200 {
        let lr = sched.lr_at_step(step);
        // During warmup, lr goes from 0 to base_lr, so minimum could be 0.
        assert!(lr >= -1e-12, "step {step}: lr {lr} should be non-negative");
        assert!(lr <= 0.05 + 1e-12, "step {step}: lr {lr} > base_lr 0.05");
    }
}

#[test]
fn test_cosine_peak_at_warmup_boundary() {
    let sched = CosineSchedule::new(0.1, 0.0, 200, 1000).unwrap();
    let lr_at_warmup_end = sched.lr_at_step(200);
    // At step == warmup_steps, cosine decay starts, progress = 0, cos(0) = 1
    // lr = min_lr + 0.5 * (base_lr - min_lr) * (1 + 1) = base_lr
    assert!(
        (lr_at_warmup_end - 0.1).abs() < 1e-10,
        "peak at warmup boundary: expected 0.1, got {lr_at_warmup_end}"
    );
}

#[test]
fn test_cosine_derivative_sign_during_decay() {
    // Cosine should be strictly decreasing in the decay phase
    let sched = CosineSchedule::new(0.1, 0.0, 0, 500).unwrap();
    for step in 1..500 {
        let lr_prev = sched.lr_at_step(step - 1);
        let lr_cur = sched.lr_at_step(step);
        assert!(
            lr_cur <= lr_prev + 1e-14,
            "cosine not decreasing at step {step}: prev={lr_prev}, cur={lr_cur}"
        );
    }
}

#[test]
fn test_cosine_smoothness_no_jumps() {
    let sched = CosineSchedule::new(0.1, 0.0, 50, 500).unwrap();
    let max_delta = 0.1 / 50.0 + 1e-6; // max possible step during warmup
    for step in 1..550 {
        let lr_prev = sched.lr_at_step(step - 1);
        let lr_cur = sched.lr_at_step(step);
        let delta = (lr_cur - lr_prev).abs();
        assert!(
            delta < max_delta,
            "step {step}: jump too large: delta={delta}, max={max_delta}"
        );
    }
}

#[test]
fn test_cosine_warmup_one_step_before_total() {
    let sched = CosineSchedule::new(0.1, 0.0, 0, 100).unwrap();
    let lr = sched.lr_at_step(99);
    let progress = 99.0 / 100.0;
    let expected = 0.5 * 0.1 * (1.0 + (progress * std::f64::consts::PI).cos());
    assert!(
        (lr - expected).abs() < 1e-10,
        "one step before end: expected {expected}, got {lr}"
    );
}

#[test]
fn test_cosine_with_nonzero_min_lr_midpoint() {
    let sched = CosineSchedule::new(0.1, 0.02, 0, 100).unwrap();
    let lr_mid = sched.lr_at_step(50);
    // cos(pi * 0.5) = 0, so lr = 0.02 + 0.5 * (0.1 - 0.02) * (1 + 0) = 0.02 + 0.04 = 0.06
    assert!(
        (lr_mid - 0.06).abs() < 1e-10,
        "midpoint with min_lr=0.02: expected 0.06, got {lr_mid}"
    );
}

#[test]
fn test_cosine_schedule_debug_format() {
    let sched = CosineSchedule::new(0.01, 0.001, 50, 1000).unwrap();
    let debug = format!("{sched:?}");
    assert!(
        debug.contains("CosineSchedule"),
        "debug should contain type name: {debug}"
    );
}

#[test]
fn test_cosine_schedule_accessor_methods() {
    let sched = CosineSchedule::new(0.05, 0.001, 100, 2000).unwrap();
    assert!((sched.base_lr() - 0.05).abs() < f64::EPSILON);
    assert!((sched.min_lr() - 0.001).abs() < f64::EPSILON);
    assert_eq!(sched.total_steps(), 2000);
}

// ============================================================================
// CosineSchedule — construction edge cases
// ============================================================================

#[test]
fn test_cosine_inf_base_lr_rejected() {
    assert!(CosineSchedule::new(f64::INFINITY, 0.0, 0, 100).is_err());
}

#[test]
fn test_cosine_inf_min_lr_rejected() {
    assert!(CosineSchedule::new(0.01, f64::INFINITY, 0, 100).is_err());
}

#[test]
fn test_cosine_warmup_one_less_than_total_succeeds() {
    // warmup = total - 1 is valid (one cosine decay step)
    let sched = CosineSchedule::new(1.0, 0.0, 99, 100).unwrap();
    // warmup phase: step 0..99
    let lr_0 = sched.lr_at_step(0);
    assert!(lr_0.abs() < f64::EPSILON);
    // decay phase has one step (step 99): progress = 0/1 = 0
    let lr_99 = sched.lr_at_step(99);
    assert!(
        (lr_99 - 1.0).abs() < 1e-10,
        "single decay step: expected 1.0, got {lr_99}"
    );
}

#[test]
fn test_cosine_zero_base_lr_and_zero_min_lr() {
    let sched = CosineSchedule::new(0.0, 0.0, 0, 100).unwrap();
    for step in [0, 25, 50, 75, 100, 200] {
        let lr = sched.lr_at_step(step);
        assert!(
            lr.abs() < 1e-15,
            "zero/zero schedule: step {step} should be ~0, got {lr}"
        );
    }
}

// ============================================================================
// SGD — weight decay configuration
// ============================================================================

#[test]
fn test_sgd_weight_decay_config_default_is_zero() {
    let cfg = SgdConfig::default();
    assert!(
        cfg.weight_decay.abs() < f64::EPSILON,
        "default weight_decay should be 0"
    );
}

#[test]
fn test_sgd_weight_decay_affects_update() {
    let x_wd = scalar_var(5.0);
    let x_no_wd = scalar_var(5.0);

    let mut sgd_wd = Sgd::new(
        vec![x_wd.clone()],
        SgdConfig {
            lr: 0.1,
            weight_decay: 0.1,
            ..SgdConfig::default()
        },
    )
    .unwrap();
    let mut sgd_no_wd = Sgd::new(
        vec![x_no_wd.clone()],
        SgdConfig {
            lr: 0.1,
            weight_decay: 0.0,
            ..SgdConfig::default()
        },
    )
    .unwrap();

    let grads_wd = make_grad(&x_wd);
    let grads_no_wd = make_grad(&x_no_wd);

    sgd_wd.step(&grads_wd).unwrap();
    sgd_no_wd.step(&grads_no_wd).unwrap();

    let val_wd = get_val(&x_wd);
    let val_no_wd = get_val(&x_no_wd);

    // With weight decay, the update is larger (grad + wd*theta)
    assert!(
        val_wd < val_no_wd,
        "weight decay should produce a smaller parameter: wd={val_wd}, no_wd={val_no_wd}"
    );
}

#[test]
fn test_sgd_weight_decay_negative_rejected() {
    let x = scalar_var(1.0);
    let result = Sgd::new(
        vec![x],
        SgdConfig {
            lr: 0.01,
            weight_decay: -0.1,
            ..SgdConfig::default()
        },
    );
    assert!(result.is_err());
    let msg = format!("{}", result.unwrap_err());
    assert!(
        msg.contains("weight_decay"),
        "error should cite weight_decay: {msg}"
    );
}

#[test]
fn test_sgd_weight_decay_nan_rejected() {
    let x = scalar_var(1.0);
    assert!(Sgd::new(
        vec![x],
        SgdConfig {
            lr: 0.01,
            weight_decay: f64::NAN,
            ..SgdConfig::default()
        },
    )
    .is_err());
}

#[test]
fn test_sgd_weight_decay_inf_rejected() {
    let x = scalar_var(1.0);
    assert!(Sgd::new(
        vec![x],
        SgdConfig {
            lr: 0.01,
            weight_decay: f64::INFINITY,
            ..SgdConfig::default()
        },
    )
    .is_err());
}

#[test]
fn test_sgd_weight_decay_accessor() {
    let x = scalar_var(1.0);
    let sgd = Sgd::new(
        vec![x],
        SgdConfig {
            lr: 0.01,
            weight_decay: 0.05,
            ..SgdConfig::default()
        },
    )
    .unwrap();
    assert!((sgd.weight_decay() - 0.05).abs() < f64::EPSILON);
}

// ============================================================================
// SGD — momentum tracking
// ============================================================================

#[test]
fn test_sgd_momentum_zero_no_velocity() {
    let x = scalar_var(2.0);
    let mut sgd = Sgd::new(
        vec![x.clone()],
        SgdConfig {
            lr: 0.1,
            momentum: 0.0,
            ..SgdConfig::default()
        },
    )
    .unwrap();
    // Two identical steps with same gradient should produce same-size updates
    let grads = make_grad(&x);
    let v0 = get_val(&x);
    sgd.step(&grads).unwrap();
    let v1 = get_val(&x);
    let delta1 = v0 - v1;

    let grads2 = make_grad(&x);
    sgd.step(&grads2).unwrap();
    let v2 = get_val(&x);
    let delta2 = v1 - v2;

    // Without momentum, delta2 should be smaller because gradient is smaller (x is closer to 0)
    // (grad = 2*x, smaller x => smaller grad => smaller delta)
    assert!(
        delta2 < delta1 + 1e-6,
        "without momentum, second step should not be larger: delta1={delta1}, delta2={delta2}"
    );
}

#[test]
fn test_sgd_momentum_accumulates_velocity() {
    let x = scalar_var(10.0);
    let mut sgd = Sgd::new(
        vec![x.clone()],
        SgdConfig {
            lr: 0.01,
            momentum: 0.9,
            ..SgdConfig::default()
        },
    )
    .unwrap();

    let mut deltas = Vec::new();
    for _ in 0..5 {
        let val_before = get_val(&x);
        let grads = make_grad(&x);
        sgd.step(&grads).unwrap();
        let val_after = get_val(&x);
        deltas.push(val_before - val_after);
    }
    // With momentum and consistent gradient direction, updates should grow
    // (velocity accumulates)
    assert!(
        deltas[2] > deltas[0],
        "momentum should accumulate: step3_delta={}, step1_delta={}",
        deltas[2],
        deltas[0]
    );
}

#[test]
fn test_sgd_momentum_accessor() {
    let x = scalar_var(1.0);
    let sgd = Sgd::new(
        vec![x],
        SgdConfig {
            lr: 0.01,
            momentum: 0.95,
            ..SgdConfig::default()
        },
    )
    .unwrap();
    assert!((sgd.momentum() - 0.95).abs() < f64::EPSILON);
}

#[test]
fn test_sgd_momentum_negative_rejected() {
    let x = scalar_var(1.0);
    assert!(Sgd::new(
        vec![x],
        SgdConfig {
            lr: 0.01,
            momentum: -0.1,
            ..SgdConfig::default()
        },
    )
    .is_err());
}

#[test]
fn test_sgd_momentum_nan_rejected() {
    let x = scalar_var(1.0);
    assert!(Sgd::new(
        vec![x],
        SgdConfig {
            lr: 0.01,
            momentum: f64::NAN,
            ..SgdConfig::default()
        },
    )
    .is_err());
}

#[test]
fn test_sgd_momentum_inf_rejected() {
    let x = scalar_var(1.0);
    assert!(Sgd::new(
        vec![x],
        SgdConfig {
            lr: 0.01,
            momentum: f64::INFINITY,
            ..SgdConfig::default()
        },
    )
    .is_err());
}

#[test]
fn test_sgd_config_accessor() {
    let x = scalar_var(1.0);
    let cfg = SgdConfig {
        lr: 0.05,
        momentum: 0.8,
        weight_decay: 0.001,
    };
    let sgd = Sgd::new(vec![x], cfg).unwrap();
    assert!((sgd.config().lr - 0.05).abs() < f64::EPSILON);
    assert!((sgd.config().momentum - 0.8).abs() < f64::EPSILON);
    assert!((sgd.config().weight_decay - 0.001).abs() < f64::EPSILON);
}

// ============================================================================
// AdamW — config validation
// ============================================================================

#[test]
fn test_adam_config_default_values() {
    let cfg = AdamConfig::default();
    assert!((cfg.lr - 1e-3).abs() < f64::EPSILON);
    assert!((cfg.beta1 - 0.9).abs() < f64::EPSILON);
    assert!((cfg.beta2 - 0.999).abs() < f64::EPSILON);
    assert!((cfg.eps - 1e-8).abs() < f64::EPSILON);
    assert!((cfg.weight_decay - 0.01).abs() < f64::EPSILON);
}

#[test]
fn test_adam_negative_lr_rejected() {
    let x = scalar_var(1.0);
    assert!(AdamW::new(
        vec![x],
        AdamConfig {
            lr: -0.001,
            ..AdamConfig::default()
        },
    )
    .is_err());
}

#[test]
fn test_adam_nan_lr_rejected() {
    let x = scalar_var(1.0);
    assert!(AdamW::new(
        vec![x],
        AdamConfig {
            lr: f64::NAN,
            ..AdamConfig::default()
        },
    )
    .is_err());
}

#[test]
fn test_adam_beta1_out_of_range_rejected() {
    let x = scalar_var(1.0);
    // beta1 = 1.0 is invalid (causes division by zero in bias correction)
    assert!(AdamW::new(
        vec![x.clone()],
        AdamConfig {
            beta1: 1.0,
            ..AdamConfig::default()
        },
    )
    .is_err());
    // beta1 = -0.1 is invalid
    assert!(AdamW::new(
        vec![x],
        AdamConfig {
            beta1: -0.1,
            ..AdamConfig::default()
        },
    )
    .is_err());
}

#[test]
fn test_adam_beta2_out_of_range_rejected() {
    let x = scalar_var(1.0);
    assert!(AdamW::new(
        vec![x.clone()],
        AdamConfig {
            beta2: 1.0,
            ..AdamConfig::default()
        },
    )
    .is_err());
    assert!(AdamW::new(
        vec![x],
        AdamConfig {
            beta2: -0.5,
            ..AdamConfig::default()
        },
    )
    .is_err());
}

#[test]
fn test_adam_eps_zero_rejected() {
    let x = scalar_var(1.0);
    assert!(AdamW::new(
        vec![x],
        AdamConfig {
            eps: 0.0,
            ..AdamConfig::default()
        },
    )
    .is_err());
}

#[test]
fn test_adam_eps_negative_rejected() {
    let x = scalar_var(1.0);
    assert!(AdamW::new(
        vec![x],
        AdamConfig {
            eps: -1e-8,
            ..AdamConfig::default()
        },
    )
    .is_err());
}

#[test]
fn test_adam_weight_decay_negative_rejected() {
    let x = scalar_var(1.0);
    assert!(AdamW::new(
        vec![x],
        AdamConfig {
            weight_decay: -0.01,
            ..AdamConfig::default()
        },
    )
    .is_err());
}

#[test]
fn test_adam_step_count_increments() {
    let x = scalar_var(3.0);
    let mut adam = AdamW::new(vec![x.clone()], AdamConfig::default()).unwrap();
    assert_eq!(adam.step_count(), 0);
    let grads = make_grad(&x);
    adam.step(&grads).unwrap();
    assert_eq!(adam.step_count(), 1);
    let grads2 = make_grad(&x);
    adam.step(&grads2).unwrap();
    assert_eq!(adam.step_count(), 2);
}

#[test]
fn test_adam_config_accessor() {
    let x = scalar_var(1.0);
    let cfg = AdamConfig {
        lr: 0.002,
        beta1: 0.85,
        beta2: 0.99,
        eps: 1e-6,
        weight_decay: 0.05,
    };
    let adam = AdamW::new(vec![x], cfg).unwrap();
    assert!((adam.config().lr - 0.002).abs() < f64::EPSILON);
    assert!((adam.config().beta1 - 0.85).abs() < f64::EPSILON);
    assert!((adam.config().beta2 - 0.99).abs() < f64::EPSILON);
    assert!((adam.config().eps - 1e-6).abs() < f64::EPSILON);
    assert!((adam.config().weight_decay - 0.05).abs() < f64::EPSILON);
}

#[test]
fn test_adam_zero_weight_decay_no_shrinkage() {
    let x = scalar_var(5.0);
    let mut adam = AdamW::new(
        vec![x.clone()],
        AdamConfig {
            lr: 0.01,
            weight_decay: 0.0,
            ..AdamConfig::default()
        },
    )
    .unwrap();
    // Build a zero-gradient scenario: the parameter should not change
    // (no gradient + no weight decay = no update)
    let var2 = scalar_var(99.0);
    let grads = make_grad(&var2); // grads for var2, not x
                                  // x has no gradient in this GradStore, so step skips it
    adam.step(&grads).unwrap();
    let val = get_val(&x);
    assert!(
        (val - 5.0).abs() < 1e-6,
        "no gradient + no wd: param should be unchanged, got {val}"
    );
}

// ============================================================================
// AdaFactor — config validation
// ============================================================================

#[test]
fn test_adafactor_config_default_values() {
    let cfg = AdaFactorConfig::default();
    assert!((cfg.lr - 1e-3).abs() < f64::EPSILON);
    assert!(!cfg.relative_step);
    assert!((cfg.eps_rms - 1e-3).abs() < f64::EPSILON);
    assert!((cfg.eps_denom - 1e-30).abs() < f64::EPSILON);
    assert!((cfg.decay_rate - (-0.8)).abs() < f64::EPSILON);
    assert!(cfg.beta1.is_none());
    assert!(cfg.weight_decay.abs() < f64::EPSILON);
}

#[test]
fn test_adafactor_positive_decay_rate_rejected() {
    let x = scalar_var(1.0);
    assert!(AdaFactor::new(
        vec![x],
        AdaFactorConfig {
            decay_rate: 0.8,
            ..AdaFactorConfig::default()
        },
    )
    .is_err());
}

#[test]
fn test_adafactor_nan_decay_rate_rejected() {
    let x = scalar_var(1.0);
    assert!(AdaFactor::new(
        vec![x],
        AdaFactorConfig {
            decay_rate: f64::NAN,
            ..AdaFactorConfig::default()
        },
    )
    .is_err());
}

#[test]
fn test_adafactor_beta1_out_of_range_rejected() {
    let x = scalar_var(1.0);
    assert!(AdaFactor::new(
        vec![x],
        AdaFactorConfig {
            beta1: Some(1.0),
            ..AdaFactorConfig::default()
        },
    )
    .is_err());
}

#[test]
fn test_adafactor_eps_denom_zero_rejected() {
    let x = scalar_var(1.0);
    assert!(AdaFactor::new(
        vec![x],
        AdaFactorConfig {
            eps_denom: 0.0,
            ..AdaFactorConfig::default()
        },
    )
    .is_err());
}

#[test]
fn test_adafactor_eps_rms_negative_rejected() {
    let x = scalar_var(1.0);
    assert!(AdaFactor::new(
        vec![x],
        AdaFactorConfig {
            eps_rms: -1e-3,
            ..AdaFactorConfig::default()
        },
    )
    .is_err());
}

#[test]
fn test_adafactor_step_count_increments() {
    let x = scalar_var(3.0);
    let mut af = AdaFactor::new(vec![x.clone()], AdaFactorConfig::default()).unwrap();
    assert_eq!(af.step_count(), 0);
    let grads = make_grad(&x);
    af.step(&grads).unwrap();
    assert_eq!(af.step_count(), 1);
}

#[test]
fn test_adafactor_config_accessor() {
    let x = scalar_var(1.0);
    let cfg = AdaFactorConfig {
        lr: 0.005,
        relative_step: true,
        eps_rms: 1e-4,
        eps_denom: 1e-20,
        decay_rate: -0.5,
        beta1: Some(0.9),
        weight_decay: 0.01,
    };
    let af = AdaFactor::new(vec![x], cfg).unwrap();
    assert!((af.config().lr - 0.005).abs() < f64::EPSILON);
    assert!(af.config().relative_step);
    assert!((af.config().decay_rate - (-0.5)).abs() < f64::EPSILON);
    assert_eq!(af.config().beta1, Some(0.9));
}

// ============================================================================
// Optimizer::set_learning_rate
// ============================================================================

#[test]
fn test_sgd_set_learning_rate_valid() {
    let x = scalar_var(1.0);
    let mut sgd = Sgd::new(
        vec![x],
        SgdConfig {
            lr: 0.01,
            ..SgdConfig::default()
        },
    )
    .unwrap();
    sgd.set_learning_rate(0.05).unwrap();
    assert!((sgd.learning_rate() - 0.05).abs() < f64::EPSILON);
}

#[test]
fn test_sgd_set_learning_rate_zero() {
    let x = scalar_var(1.0);
    let mut sgd = Sgd::new(
        vec![x],
        SgdConfig {
            lr: 0.01,
            ..SgdConfig::default()
        },
    )
    .unwrap();
    sgd.set_learning_rate(0.0).unwrap();
    assert!(sgd.learning_rate().abs() < f64::EPSILON);
}

#[test]
fn test_sgd_set_learning_rate_negative_rejected() {
    let x = scalar_var(1.0);
    let mut sgd = Sgd::new(
        vec![x],
        SgdConfig {
            lr: 0.01,
            ..SgdConfig::default()
        },
    )
    .unwrap();
    assert!(sgd.set_learning_rate(-0.01).is_err());
    // Verify lr unchanged after rejection
    assert!((sgd.learning_rate() - 0.01).abs() < f64::EPSILON);
}

#[test]
fn test_sgd_set_learning_rate_nan_rejected() {
    let x = scalar_var(1.0);
    let mut sgd = Sgd::new(vec![x], SgdConfig::default()).unwrap();
    assert!(sgd.set_learning_rate(f64::NAN).is_err());
}

#[test]
fn test_sgd_set_learning_rate_inf_rejected() {
    let x = scalar_var(1.0);
    let mut sgd = Sgd::new(vec![x], SgdConfig::default()).unwrap();
    assert!(sgd.set_learning_rate(f64::INFINITY).is_err());
}

#[test]
fn test_adam_set_learning_rate_valid() {
    let x = scalar_var(1.0);
    let mut adam = AdamW::new(vec![x], AdamConfig::default()).unwrap();
    adam.set_learning_rate(0.005).unwrap();
    assert!((adam.learning_rate() - 0.005).abs() < f64::EPSILON);
}

#[test]
fn test_adam_set_learning_rate_negative_rejected() {
    let x = scalar_var(1.0);
    let mut adam = AdamW::new(vec![x], AdamConfig::default()).unwrap();
    assert!(adam.set_learning_rate(-0.001).is_err());
}

#[test]
fn test_adafactor_set_learning_rate_valid() {
    let x = scalar_var(1.0);
    let mut af = AdaFactor::new(vec![x], AdaFactorConfig::default()).unwrap();
    af.set_learning_rate(0.01).unwrap();
    assert!((af.learning_rate() - 0.01).abs() < f64::EPSILON);
}

#[test]
fn test_adafactor_set_learning_rate_negative_rejected() {
    let x = scalar_var(1.0);
    let mut af = AdaFactor::new(vec![x], AdaFactorConfig::default()).unwrap();
    assert!(af.set_learning_rate(-0.01).is_err());
}

// ============================================================================
// Optimizer — empty var list
// ============================================================================

#[test]
fn test_sgd_empty_vars_step_succeeds() {
    let mut sgd = Sgd::new(vec![], SgdConfig::default()).unwrap();
    let grads = nn_autodiff::GradStore::new();
    sgd.step(&grads).unwrap();
}

#[test]
fn test_adam_empty_vars_step_succeeds() {
    let mut adam = AdamW::new(vec![], AdamConfig::default()).unwrap();
    let grads = nn_autodiff::GradStore::new();
    adam.step(&grads).unwrap();
}

#[test]
fn test_adafactor_empty_vars_step_succeeds() {
    let mut af = AdaFactor::new(vec![], AdaFactorConfig::default()).unwrap();
    let grads = nn_autodiff::GradStore::new();
    af.step(&grads).unwrap();
}

// ============================================================================
// Schedule + Optimizer integration — warmup with SGD
// ============================================================================

#[test]
fn test_warmup_schedule_with_sgd_lr_progression() {
    let x = scalar_var(1.0);
    let mut sgd = Sgd::new(
        vec![x.clone()],
        SgdConfig {
            lr: 0.0,
            ..SgdConfig::default()
        },
    )
    .unwrap();
    let schedule = WarmupSchedule::new(0.1, 10).unwrap();

    let mut lrs = Vec::new();
    for step in 0..15 {
        let grads = make_grad(&x);
        step_with_schedule(&mut sgd, &grads, &schedule, step).unwrap();
        lrs.push(sgd.learning_rate());
    }

    // During warmup, lr should increase linearly
    for i in 1..10 {
        assert!(
            lrs[i] > lrs[i - 1] - 1e-12,
            "warmup: lr[{i}]={} should be >= lr[{}]={}",
            lrs[i],
            i - 1,
            lrs[i - 1]
        );
    }
    // After warmup, lr should be constant at base_lr
    for i in 10..15 {
        assert!(
            (lrs[i] - 0.1).abs() < f64::EPSILON,
            "post-warmup: lr[{i}]={}, expected 0.1",
            lrs[i]
        );
    }
}

#[test]
fn test_cosine_schedule_with_adam_lr_progression() {
    let x = scalar_var(2.0);
    let mut adam = AdamW::new(
        vec![x.clone()],
        AdamConfig {
            lr: 0.0,
            ..AdamConfig::default()
        },
    )
    .unwrap();
    let schedule = CosineSchedule::new(0.01, 0.001, 0, 100).unwrap();

    let mut lrs = Vec::new();
    for step in 0..50 {
        let grads = make_grad(&x);
        step_with_schedule(&mut adam, &grads, &schedule, step).unwrap();
        lrs.push(adam.learning_rate());
    }

    // lr should be decreasing (cosine decay)
    for i in 1..50 {
        assert!(
            lrs[i] <= lrs[i - 1] + 1e-12,
            "cosine decay: lr[{i}]={} should be <= lr[{}]={}",
            lrs[i],
            i - 1,
            lrs[i - 1]
        );
    }
}

// ============================================================================
// Schedule + Optimizer — convergence
// ============================================================================

#[test]
fn test_warmup_cosine_full_training_loop_convergence() {
    // Minimize x^2 using SGD with cosine schedule including warmup
    let x = scalar_var(10.0);
    let mut sgd = Sgd::new(
        vec![x.clone()],
        SgdConfig {
            lr: 0.0,
            ..SgdConfig::default()
        },
    )
    .unwrap();
    let schedule = CosineSchedule::new(0.05, 0.001, 10, 200).unwrap();

    let initial_loss = get_val(&x).powi(2);
    for step in 0..100 {
        let grads = make_grad(&x);
        step_with_schedule(&mut sgd, &grads, &schedule, step).unwrap();
    }
    let final_loss = get_val(&x).powi(2);

    assert!(
        final_loss < initial_loss * 0.1,
        "loss should decrease by 10x: initial={initial_loss}, final={final_loss}"
    );
}

#[test]
fn test_adam_with_warmup_schedule_convergence() {
    let x = scalar_var(5.0);
    let mut adam = AdamW::new(
        vec![x.clone()],
        AdamConfig {
            lr: 0.0,
            weight_decay: 0.0,
            ..AdamConfig::default()
        },
    )
    .unwrap();
    let schedule = WarmupSchedule::new(0.1, 20).unwrap();

    // 20 warmup steps plus Adam's ~lr-per-step movement (base_lr=0.1) means
    // reaching |x| < 1.0 from x=5 needs ~70 steps; 50 steps only reaches
    // ~1.63. The optimizer is correct; the step budget was too small.
    for step in 0..80 {
        let grads = make_grad(&x);
        step_with_schedule(&mut adam, &grads, &schedule, step).unwrap();
    }
    let val = get_val(&x);
    assert!(
        val.abs() < 1.0,
        "Adam with warmup should converge toward 0: got {val}"
    );
}

// ============================================================================
// Multiple parameters — optimizer handles all vars
// ============================================================================

#[test]
fn test_sgd_multiple_vars_all_updated() {
    let x1 = scalar_var(3.0);
    let x2 = scalar_var(-4.0);
    let mut sgd = Sgd::new(
        vec![x1.clone(), x2.clone()],
        SgdConfig {
            lr: 0.01,
            ..SgdConfig::default()
        },
    )
    .unwrap();

    // Build grads for both vars
    let t1 = Arc::new(TrackedTensor::from_var(&x1).unwrap());
    let t2 = Arc::new(TrackedTensor::from_var(&x2).unwrap());
    let loss = t1.sqr().unwrap().add(&t2.sqr().unwrap()).unwrap();
    let grads = backward(&loss).unwrap();

    let v1_before = get_val(&x1);
    let v2_before = get_val(&x2);
    sgd.step(&grads).unwrap();
    let v1_after = get_val(&x1);
    let v2_after = get_val(&x2);

    // Both should move toward zero
    assert!(
        v1_after.abs() < v1_before.abs(),
        "x1 should decrease in magnitude"
    );
    assert!(
        v2_after.abs() < v2_before.abs(),
        "x2 should decrease in magnitude"
    );
}

#[test]
fn test_adam_multiple_vars_all_updated() {
    let x1 = scalar_var(5.0);
    let x2 = scalar_var(-3.0);
    let mut adam = AdamW::new(
        vec![x1.clone(), x2.clone()],
        AdamConfig {
            lr: 0.1,
            weight_decay: 0.0,
            ..AdamConfig::default()
        },
    )
    .unwrap();

    let t1 = Arc::new(TrackedTensor::from_var(&x1).unwrap());
    let t2 = Arc::new(TrackedTensor::from_var(&x2).unwrap());
    let loss = t1.sqr().unwrap().add(&t2.sqr().unwrap()).unwrap();
    let grads = backward(&loss).unwrap();

    adam.step(&grads).unwrap();

    let v1 = get_val(&x1);
    let v2 = get_val(&x2);
    assert!(v1 < 5.0, "x1 should decrease from 5.0, got {v1}");
    assert!(v2 > -3.0, "x2 should increase toward 0 from -3.0, got {v2}");
}

// ============================================================================
// Vector and matrix variables
// ============================================================================

#[test]
fn test_sgd_vector_var_update() {
    let x = vec_var(&[1.0, 2.0, 3.0]);
    let mut sgd = Sgd::new(
        vec![x.clone()],
        SgdConfig {
            lr: 0.01,
            ..SgdConfig::default()
        },
    )
    .unwrap();

    let t = Arc::new(TrackedTensor::from_var(&x).unwrap());
    let loss = t.sqr().unwrap().sum_keepdim(0).unwrap();
    let grads = backward(&loss).unwrap();
    sgd.step(&grads).unwrap();

    let vals = get_vals(&x);
    // Each element should decrease: x_i = x_i - lr * 2 * x_i
    for (i, &v) in vals.iter().enumerate() {
        let original = (i + 1) as f32;
        assert!(
            v < original,
            "element {i} should decrease: {v} >= {original}"
        );
    }
}

#[test]
fn test_adam_matrix_var_update() {
    let x = mat_var(&[1.0, 2.0, 3.0, 4.0], 2, 2);
    let mut adam = AdamW::new(
        vec![x.clone()],
        AdamConfig {
            lr: 0.1,
            weight_decay: 0.0,
            ..AdamConfig::default()
        },
    )
    .unwrap();

    let t = Arc::new(TrackedTensor::from_var(&x).unwrap());
    // Reduce [2, 2] -> [1, 2] -> [1, 1] (scalar). Summing dim 0 twice would
    // leave shape [1, 2], which backward() rejects as a non-scalar loss.
    let loss = t
        .sqr()
        .unwrap()
        .sum_keepdim(0)
        .unwrap()
        .sum_keepdim(1)
        .unwrap();
    let grads = backward(&loss).unwrap();
    adam.step(&grads).unwrap();

    let vals = get_vals(&x);
    for (i, &v) in vals.iter().enumerate() {
        let original = (i + 1) as f32;
        assert!(
            v < original,
            "matrix element {i} should decrease: {v} >= {original}"
        );
    }
}

// ============================================================================
// Schedule — LrSchedule trait object usage
// ============================================================================

#[test]
fn test_lr_schedule_trait_object_warmup() {
    let sched: Box<dyn LrSchedule> = Box::new(WarmupSchedule::new(0.5, 100).unwrap());
    let lr = sched.lr_at_step(50);
    assert!((lr - 0.25).abs() < 1e-12);
}

#[test]
fn test_lr_schedule_trait_object_cosine() {
    let sched: Box<dyn LrSchedule> = Box::new(CosineSchedule::new(0.1, 0.0, 0, 100).unwrap());
    let lr = sched.lr_at_step(50);
    assert!((lr - 0.05).abs() < 1e-10);
}

#[test]
fn test_lr_schedule_trait_ref_step_with_schedule() {
    let x = scalar_var(3.0);
    let mut sgd = Sgd::new(
        vec![x.clone()],
        SgdConfig {
            lr: 0.0,
            ..SgdConfig::default()
        },
    )
    .unwrap();
    let warmup = WarmupSchedule::new(0.1, 10).unwrap();
    let cosine = CosineSchedule::new(0.1, 0.01, 0, 100).unwrap();

    // Use warmup for first 10 steps
    for step in 0..10 {
        let grads = make_grad(&x);
        step_with_schedule(&mut sgd, &grads, &warmup, step).unwrap();
    }
    // Then switch to cosine
    for step in 0..10 {
        let grads = make_grad(&x);
        step_with_schedule(&mut sgd, &grads, &cosine, step).unwrap();
    }
    // Should have converged somewhat
    assert!(
        get_val(&x).abs() < 3.0,
        "after 20 steps, should move from 3.0 toward 0"
    );
}

// ============================================================================
// SGD — lr=0 produces no update
// ============================================================================

#[test]
fn test_sgd_zero_lr_no_update() {
    let x = scalar_var(7.0);
    let mut sgd = Sgd::new(
        vec![x.clone()],
        SgdConfig {
            lr: 0.0,
            ..SgdConfig::default()
        },
    )
    .unwrap();
    let grads = make_grad(&x);
    sgd.step(&grads).unwrap();
    let val = get_val(&x);
    assert!(
        (val - 7.0).abs() < 1e-6,
        "lr=0 should produce no update, got {val}"
    );
}

// ============================================================================
// AdamW weight decay vs no weight decay
// ============================================================================

#[test]
fn test_adam_weight_decay_shrinks_params() {
    let x_wd = scalar_var(5.0);
    let x_no_wd = scalar_var(5.0);

    let mut adam_wd = AdamW::new(
        vec![x_wd.clone()],
        AdamConfig {
            lr: 0.01,
            weight_decay: 0.1,
            ..AdamConfig::default()
        },
    )
    .unwrap();
    let mut adam_no_wd = AdamW::new(
        vec![x_no_wd.clone()],
        AdamConfig {
            lr: 0.01,
            weight_decay: 0.0,
            ..AdamConfig::default()
        },
    )
    .unwrap();

    let grads_wd = make_grad(&x_wd);
    let grads_no_wd = make_grad(&x_no_wd);

    adam_wd.step(&grads_wd).unwrap();
    adam_no_wd.step(&grads_no_wd).unwrap();

    let val_wd = get_val(&x_wd);
    let val_no_wd = get_val(&x_no_wd);

    // Decoupled weight decay shrinks theta directly: theta *= (1 - lr * wd)
    // So with weight decay, the parameter should be smaller
    assert!(
        val_wd < val_no_wd,
        "AdamW weight decay should shrink param more: wd={val_wd}, no_wd={val_no_wd}"
    );
}

// ============================================================================
// AdaFactor — factored vs full second moment
// ============================================================================

#[test]
fn test_adafactor_vector_uses_full_moment() {
    // rank < 2 uses full second moment
    let x = vec_var(&[1.0, 2.0, 3.0]);
    let mut af = AdaFactor::new(vec![x.clone()], AdaFactorConfig::default()).unwrap();
    let t = Arc::new(TrackedTensor::from_var(&x).unwrap());
    let loss = t.sqr().unwrap().sum_keepdim(0).unwrap();
    let grads = backward(&loss).unwrap();
    af.step(&grads).unwrap();

    let vals = get_vals(&x);
    for (i, &v) in vals.iter().enumerate() {
        let original = (i + 1) as f32;
        assert!(
            v < original,
            "AdaFactor vector element {i}: {v} should be < {original}"
        );
    }
}

#[test]
fn test_adafactor_matrix_uses_factored_moment() {
    // rank >= 2 uses factored second moment
    let x = mat_var(&[1.0, 2.0, 3.0, 4.0], 2, 2);
    let mut af = AdaFactor::new(vec![x.clone()], AdaFactorConfig::default()).unwrap();
    let t = Arc::new(TrackedTensor::from_var(&x).unwrap());
    // Reduce [2, 2] -> [1, 2] -> [1, 1] (scalar). Summing dim 0 twice would
    // leave shape [1, 2], which backward() rejects as a non-scalar loss.
    let loss = t
        .sqr()
        .unwrap()
        .sum_keepdim(0)
        .unwrap()
        .sum_keepdim(1)
        .unwrap();
    let grads = backward(&loss).unwrap();
    af.step(&grads).unwrap();

    let vals = get_vals(&x);
    for (i, &v) in vals.iter().enumerate() {
        let original = (i + 1) as f32;
        assert!(
            v < original,
            "AdaFactor matrix element {i}: {v} should be < {original}"
        );
    }
}

// ============================================================================
// SgdConfig / AdamConfig Clone and PartialEq
// ============================================================================

#[test]
fn test_sgd_config_clone_eq() {
    let cfg = SgdConfig {
        lr: 0.05,
        momentum: 0.9,
        weight_decay: 0.01,
    };
    let cloned = cfg.clone();
    assert_eq!(cfg, cloned);
}

#[test]
fn test_adam_config_clone_eq() {
    let cfg = AdamConfig {
        lr: 0.002,
        beta1: 0.85,
        beta2: 0.98,
        eps: 1e-7,
        weight_decay: 0.05,
    };
    let cloned = cfg.clone();
    assert_eq!(cfg, cloned);
}

#[test]
fn test_adafactor_config_clone_eq() {
    let cfg = AdaFactorConfig {
        lr: 0.005,
        relative_step: true,
        eps_rms: 1e-4,
        eps_denom: 1e-20,
        decay_rate: -0.5,
        beta1: Some(0.9),
        weight_decay: 0.01,
    };
    let cloned = cfg.clone();
    assert_eq!(cfg, cloned);
}

// ============================================================================
// Warmup + weight decay combined
// ============================================================================

#[test]
fn test_warmup_schedule_with_weight_decay_sgd() {
    let x = scalar_var(10.0);
    let mut sgd = Sgd::new(
        vec![x.clone()],
        SgdConfig {
            lr: 0.0,
            momentum: 0.0,
            weight_decay: 0.01,
        },
    )
    .unwrap();
    let schedule = WarmupSchedule::new(0.1, 10).unwrap();

    let initial = get_val(&x);
    for step in 0..20 {
        let grads = make_grad(&x);
        step_with_schedule(&mut sgd, &grads, &schedule, step).unwrap();
    }
    let final_val = get_val(&x);

    assert!(
        final_val.abs() < initial.abs(),
        "warmup + weight_decay should converge: initial={initial}, final={final_val}"
    );
}

// ============================================================================
// Momentum + cosine schedule combined
// ============================================================================

#[test]
fn test_cosine_schedule_with_momentum_sgd() {
    let x = scalar_var(8.0);
    let mut sgd = Sgd::new(
        vec![x.clone()],
        SgdConfig {
            lr: 0.0,
            momentum: 0.9,
            weight_decay: 0.0,
        },
    )
    .unwrap();
    let schedule = CosineSchedule::new(0.02, 0.001, 5, 100).unwrap();

    let initial = get_val(&x);
    for step in 0..50 {
        let grads = make_grad(&x);
        step_with_schedule(&mut sgd, &grads, &schedule, step).unwrap();
    }
    let final_val = get_val(&x);

    assert!(
        final_val.abs() < initial.abs() * 0.5,
        "momentum + cosine should converge: initial={initial}, final={final_val}"
    );
}

// ============================================================================
// Schedule switching — warmup then cosine
// ============================================================================

#[test]
fn test_schedule_switching_warmup_then_cosine() {
    let x = scalar_var(5.0);
    let mut sgd = Sgd::new(
        vec![x.clone()],
        SgdConfig {
            lr: 0.0,
            ..SgdConfig::default()
        },
    )
    .unwrap();

    let warmup = WarmupSchedule::new(0.1, 10).unwrap();
    let cosine = CosineSchedule::new(0.1, 0.01, 0, 100).unwrap();

    // Phase 1: warmup
    for step in 0..10 {
        let grads = make_grad(&x);
        step_with_schedule(&mut sgd, &grads, &warmup, step).unwrap();
    }
    let after_warmup = get_val(&x);

    // Phase 2: cosine
    for step in 0..20 {
        let grads = make_grad(&x);
        step_with_schedule(&mut sgd, &grads, &cosine, step).unwrap();
    }
    let after_cosine = get_val(&x);

    assert!(
        after_cosine.abs() < after_warmup.abs(),
        "cosine phase should continue convergence: warmup={after_warmup}, cosine={after_cosine}"
    );
}

// ============================================================================
// Error type matching
// ============================================================================

#[test]
fn test_error_invalid_param_variant() {
    let x = scalar_var(1.0);
    let result = Sgd::new(
        vec![x],
        SgdConfig {
            lr: -1.0,
            ..SgdConfig::default()
        },
    );
    match result {
        Err(OptimError::InvalidParam { param, .. }) => {
            assert_eq!(param, "lr");
        }
        other => panic!("expected InvalidParam, got {other:?}"),
    }
}

#[test]
fn test_error_display_contains_param_name() {
    let err = WarmupSchedule::new(-0.01, 10).unwrap_err();
    let msg = format!("{err}");
    assert!(msg.contains("base_lr"), "error display: {msg}");
}

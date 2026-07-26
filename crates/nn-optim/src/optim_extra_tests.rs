// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Additional tests for nn-optim: config validation, scheduler bounds,
//! error types, and edge cases across all optimizers.

use std::sync::Arc;

use nn_autodiff::{TrackedTensor, Var};
use nn_core::dyn_tensor::DynTensor;
use nn_core::test_utils::cpu;

use crate::adafactor::{AdaFactor, AdaFactorConfig};
use crate::adam::{AdamConfig, AdamW};
use crate::error::OptimError;
use crate::grad_scaler::{GradScaler, GradScalerConfig};
use crate::lr_schedule::{CosineSchedule, LrSchedule, WarmupSchedule};
use crate::optimizer::Optimizer;
use crate::sgd::{Sgd, SgdConfig};

fn scalar_var(val: f32) -> Var {
    Var::new(DynTensor::from_vec(vec![val], &[1], &cpu()).unwrap())
}

// ============================================================================
// OptimError Display and variant tests
// ============================================================================

#[test]
fn test_optim_error_invalid_param_display() {
    let err = OptimError::InvalidParam {
        param: "lr",
        reason: "must be non-negative".to_string(),
    };
    let msg = format!("{err}");
    assert!(msg.contains("lr"), "should contain param name: {msg}");
    assert!(
        msg.contains("must be non-negative"),
        "should contain reason: {msg}"
    );
}

#[test]
fn test_optim_error_missing_state_display() {
    let err = OptimError::MissingState {
        optimizer: "AdaFactor",
        state: "row_factor",
    };
    let msg = format!("{err}");
    assert!(msg.contains("AdaFactor"), "should mention optimizer: {msg}");
    assert!(msg.contains("row_factor"), "should mention state: {msg}");
}

#[test]
fn test_optim_error_non_finite_gradient_display() {
    let err = OptimError::NonFiniteGradient { count: 42 };
    let msg = format!("{err}");
    assert!(msg.contains("42"), "should contain count: {msg}");
    assert!(
        msg.contains("non-finite"),
        "should mention non-finite: {msg}"
    );
}

#[test]
fn test_optim_error_non_finite_update_display() {
    let err = OptimError::NonFiniteUpdate { count: 7 };
    let msg = format!("{err}");
    assert!(msg.contains("7"), "should contain count: {msg}");
}

#[test]
fn test_optim_error_checkpoint_shape_mismatch_display() {
    let err = OptimError::CheckpointShapeMismatch {
        key: "adam_0_m".to_string(),
        expected: vec![3, 4],
        got: vec![2, 4],
    };
    let msg = format!("{err}");
    assert!(msg.contains("adam_0_m"), "should contain key: {msg}");
    assert!(
        msg.contains("[3, 4]"),
        "should contain expected shape: {msg}"
    );
    assert!(msg.contains("[2, 4]"), "should contain got shape: {msg}");
}

#[test]
fn test_optim_error_checkpoint_step_overflow_display() {
    let err = OptimError::CheckpointStepOverflow { step: i64::MAX };
    let msg = format!("{err}");
    assert!(msg.contains("step"), "should mention step: {msg}");
}

#[test]
fn test_optim_error_non_finite_checkpoint_display() {
    let checkpoint_slot = "sgd_0_velocity".to_string();
    let err = OptimError::NonFiniteCheckpoint {
        key: checkpoint_slot,
        count: 3,
    };
    let msg = format!("{err}");
    assert!(msg.contains("sgd_0_velocity"), "should contain key: {msg}");
    assert!(msg.contains("3"), "should contain count: {msg}");
}

#[test]
fn test_optim_error_corrupted_state_display() {
    let err = OptimError::CorruptedState {
        optimizer: "AdaFactor",
        reason: "neither factored nor full second moment exists",
    };
    let msg = format!("{err}");
    assert!(msg.contains("AdaFactor"), "should mention optimizer: {msg}");
    assert!(
        msg.contains("neither factored"),
        "should contain reason: {msg}"
    );
}

#[test]
fn test_optim_error_checkpoint_serde_display() {
    let err = OptimError::CheckpointSerde {
        reason: "invalid JSON".to_string(),
    };
    let msg = format!("{err}");
    assert!(msg.contains("invalid JSON"), "should contain reason: {msg}");
}

// ============================================================================
// SgdConfig validation edge cases
// ============================================================================

#[test]
fn test_sgd_rejects_nan_lr() {
    let x = scalar_var(1.0);
    let result = Sgd::new(
        vec![x],
        SgdConfig {
            lr: f64::NAN,
            ..Default::default()
        },
    );
    assert!(result.is_err(), "NaN lr should be rejected");
    let msg = result.unwrap_err().to_string();
    assert!(msg.contains("lr"), "error should mention lr: {msg}");
}

#[test]
fn test_sgd_rejects_negative_lr() {
    let x = scalar_var(1.0);
    let result = Sgd::new(
        vec![x],
        SgdConfig {
            lr: -0.01,
            ..Default::default()
        },
    );
    assert!(result.is_err(), "negative lr should be rejected");
}

#[test]
fn test_sgd_rejects_inf_lr() {
    let x = scalar_var(1.0);
    let result = Sgd::new(
        vec![x],
        SgdConfig {
            lr: f64::INFINITY,
            ..Default::default()
        },
    );
    assert!(result.is_err(), "infinite lr should be rejected");
}

#[test]
fn test_sgd_rejects_nan_momentum() {
    let x = scalar_var(1.0);
    let result = Sgd::new(
        vec![x],
        SgdConfig {
            momentum: f64::NAN,
            ..Default::default()
        },
    );
    assert!(result.is_err(), "NaN momentum should be rejected");
}

#[test]
fn test_sgd_rejects_negative_momentum() {
    let x = scalar_var(1.0);
    let result = Sgd::new(
        vec![x],
        SgdConfig {
            momentum: -0.1,
            ..Default::default()
        },
    );
    assert!(result.is_err(), "negative momentum should be rejected");
}

#[test]
fn test_sgd_rejects_nan_weight_decay() {
    let x = scalar_var(1.0);
    let result = Sgd::new(
        vec![x],
        SgdConfig {
            weight_decay: f64::NAN,
            ..Default::default()
        },
    );
    assert!(result.is_err(), "NaN weight_decay should be rejected");
}

#[test]
fn test_sgd_rejects_negative_weight_decay() {
    let x = scalar_var(1.0);
    let result = Sgd::new(
        vec![x],
        SgdConfig {
            weight_decay: -0.01,
            ..Default::default()
        },
    );
    assert!(result.is_err(), "negative weight_decay should be rejected");
}

#[test]
fn test_sgd_config_default_values() {
    let config = SgdConfig::default();
    assert!((config.lr - 1e-2).abs() < f64::EPSILON, "default lr");
    assert!(
        (config.momentum - 0.0).abs() < f64::EPSILON,
        "default momentum"
    );
    assert!(
        (config.weight_decay - 0.0).abs() < f64::EPSILON,
        "default weight_decay"
    );
}

// ============================================================================
// Adam set_learning_rate validation
// ============================================================================

#[test]
fn test_adam_set_learning_rate_rejects_nan() {
    let x = scalar_var(1.0);
    let mut adam = AdamW::new(vec![x], AdamConfig::default()).unwrap();
    let result = adam.set_learning_rate(f64::NAN);
    assert!(
        result.is_err(),
        "NaN lr should be rejected by set_learning_rate"
    );
}

#[test]
fn test_adam_set_learning_rate_rejects_negative() {
    let x = scalar_var(1.0);
    let mut adam = AdamW::new(vec![x], AdamConfig::default()).unwrap();
    let result = adam.set_learning_rate(-0.01);
    assert!(
        result.is_err(),
        "negative lr should be rejected by set_learning_rate"
    );
}

#[test]
fn test_adam_set_learning_rate_rejects_inf() {
    let x = scalar_var(1.0);
    let mut adam = AdamW::new(vec![x], AdamConfig::default()).unwrap();
    let result = adam.set_learning_rate(f64::INFINITY);
    assert!(
        result.is_err(),
        "infinite lr should be rejected by set_learning_rate"
    );
}

#[test]
fn test_adam_set_learning_rate_accepts_zero() {
    let x = scalar_var(1.0);
    let mut adam = AdamW::new(vec![x], AdamConfig::default()).unwrap();
    adam.set_learning_rate(0.0).unwrap();
    assert!((adam.learning_rate() - 0.0).abs() < f64::EPSILON);
}

// ============================================================================
// SGD set_learning_rate validation
// ============================================================================

#[test]
fn test_sgd_set_learning_rate_rejects_nan() {
    let x = scalar_var(1.0);
    let mut sgd = Sgd::new(vec![x], SgdConfig::default()).unwrap();
    let result = sgd.set_learning_rate(f64::NAN);
    assert!(result.is_err(), "NaN lr should be rejected");
}

#[test]
fn test_sgd_set_learning_rate_rejects_negative() {
    let x = scalar_var(1.0);
    let mut sgd = Sgd::new(vec![x], SgdConfig::default()).unwrap();
    let result = sgd.set_learning_rate(-1.0);
    assert!(result.is_err(), "negative lr should be rejected");
}

// ============================================================================
// AdaFactor set_learning_rate validation and config accessor
// ============================================================================

#[test]
fn test_adafactor_set_learning_rate_rejects_nan() {
    let x = scalar_var(1.0);
    let mut opt = AdaFactor::new(vec![x], AdaFactorConfig::default()).unwrap();
    let result = opt.set_learning_rate(f64::NAN);
    assert!(result.is_err(), "NaN lr should be rejected");
}

#[test]
fn test_adafactor_set_learning_rate_rejects_negative() {
    let x = scalar_var(1.0);
    let mut opt = AdaFactor::new(vec![x], AdaFactorConfig::default()).unwrap();
    let result = opt.set_learning_rate(-0.01);
    assert!(result.is_err(), "negative lr should be rejected");
}

#[test]
fn test_adafactor_set_learning_rate_accepts_zero() {
    let x = scalar_var(1.0);
    let mut opt = AdaFactor::new(vec![x], AdaFactorConfig::default()).unwrap();
    opt.set_learning_rate(0.0).unwrap();
    assert!((opt.learning_rate() - 0.0).abs() < f64::EPSILON);
}

#[test]
fn test_adafactor_config_accessor() {
    let x = scalar_var(1.0);
    let config = AdaFactorConfig {
        lr: 0.05,
        relative_step: true,
        beta1: Some(0.85),
        ..Default::default()
    };
    let opt = AdaFactor::new(vec![x], config).unwrap();
    assert!((opt.config().lr - 0.05).abs() < f64::EPSILON);
    assert!(opt.config().relative_step);
    assert_eq!(opt.config().beta1, Some(0.85));
}

#[test]
fn test_adafactor_config_default_values() {
    let config = AdaFactorConfig::default();
    assert!((config.lr - 1e-3).abs() < f64::EPSILON);
    assert!(!config.relative_step);
    assert!((config.eps_rms - 1e-3).abs() < f64::EPSILON);
    assert!((config.eps_denom - 1e-30).abs() < f64::EPSILON);
    assert!((config.decay_rate - (-0.8)).abs() < f64::EPSILON);
    assert_eq!(config.beta1, None);
    assert!((config.weight_decay - 0.0).abs() < f64::EPSILON);
}

// ============================================================================
// AdaFactor config validation edge cases
// ============================================================================

#[test]
fn test_adafactor_rejects_nan_decay_rate() {
    let x = scalar_var(1.0);
    let config = AdaFactorConfig {
        decay_rate: f64::NAN,
        ..Default::default()
    };
    let result = AdaFactor::new(vec![x], config);
    assert!(result.is_err(), "NaN decay_rate should be rejected");
    let msg = result.unwrap_err().to_string();
    assert!(
        msg.contains("decay_rate"),
        "error should mention decay_rate: {msg}"
    );
}

#[test]
fn test_adafactor_rejects_inf_decay_rate() {
    let x = scalar_var(1.0);
    let config = AdaFactorConfig {
        decay_rate: f64::INFINITY,
        ..Default::default()
    };
    let result = AdaFactor::new(vec![x], config);
    assert!(result.is_err(), "infinite decay_rate should be rejected");
}

#[test]
fn test_adafactor_accepts_zero_decay_rate() {
    // decay_rate=0.0 is not positive, so it passes the > 0 check.
    // With decay_rate=0, rho_t = 1 - t^0 = 0 for all t > 0, meaning
    // second moments rely entirely on the current gradient (no history).
    let x = scalar_var(1.0);
    let config = AdaFactorConfig {
        decay_rate: 0.0,
        ..Default::default()
    };
    let result = AdaFactor::new(vec![x], config);
    assert!(
        result.is_ok(),
        "zero decay_rate should be accepted (only positive is rejected)"
    );
}

#[test]
fn test_adafactor_rejects_negative_weight_decay() {
    let x = scalar_var(1.0);
    let config = AdaFactorConfig {
        weight_decay: -0.1,
        ..Default::default()
    };
    let result = AdaFactor::new(vec![x], config);
    assert!(result.is_err(), "negative weight_decay should be rejected");
}

// ============================================================================
// WarmupSchedule validation edge cases
// ============================================================================

#[test]
fn test_warmup_rejects_nan_base_lr() {
    let result = WarmupSchedule::new(f64::NAN, 100);
    assert!(result.is_err(), "NaN base_lr should be rejected");
    let msg = result.unwrap_err().to_string();
    assert!(
        msg.contains("base_lr"),
        "error should mention base_lr: {msg}"
    );
}

#[test]
fn test_warmup_rejects_negative_base_lr() {
    let result = WarmupSchedule::new(-0.01, 100);
    assert!(result.is_err(), "negative base_lr should be rejected");
}

#[test]
fn test_warmup_rejects_inf_base_lr() {
    let result = WarmupSchedule::new(f64::INFINITY, 100);
    assert!(result.is_err(), "infinite base_lr should be rejected");
}

#[test]
fn test_warmup_accepts_zero_base_lr() {
    let sched = WarmupSchedule::new(0.0, 100).unwrap();
    // With base_lr=0, lr is always 0 regardless of step
    assert!((sched.lr_at_step(0) - 0.0).abs() < f64::EPSILON);
    assert!((sched.lr_at_step(50) - 0.0).abs() < f64::EPSILON);
    assert!((sched.lr_at_step(200) - 0.0).abs() < f64::EPSILON);
}

// ============================================================================
// CosineSchedule validation edge cases
// ============================================================================

#[test]
fn test_cosine_rejects_nan_base_lr() {
    let result = CosineSchedule::new(f64::NAN, 0.0, 0, 1000);
    assert!(result.is_err(), "NaN base_lr should be rejected");
}

#[test]
fn test_cosine_rejects_negative_base_lr() {
    let result = CosineSchedule::new(-0.01, 0.0, 0, 1000);
    assert!(result.is_err(), "negative base_lr should be rejected");
}

#[test]
fn test_cosine_rejects_nan_min_lr() {
    let result = CosineSchedule::new(0.01, f64::NAN, 0, 1000);
    assert!(result.is_err(), "NaN min_lr should be rejected");
}

#[test]
fn test_cosine_rejects_negative_min_lr() {
    let result = CosineSchedule::new(0.01, -0.001, 0, 1000);
    assert!(result.is_err(), "negative min_lr should be rejected");
}

#[test]
fn test_cosine_equal_base_and_min_lr() {
    // When base_lr == min_lr, the schedule should always return that value
    let sched = CosineSchedule::new(0.01, 0.01, 0, 1000).unwrap();
    assert!(
        (sched.lr_at_step(0) - 0.01).abs() < 1e-10,
        "should be 0.01 at step 0"
    );
    assert!(
        (sched.lr_at_step(500) - 0.01).abs() < 1e-10,
        "should be 0.01 at midpoint"
    );
    assert!(
        (sched.lr_at_step(999) - 0.01).abs() < 1e-10,
        "should be 0.01 near end"
    );
}

#[test]
fn test_cosine_monotonically_decreasing_in_decay_phase() {
    let sched = CosineSchedule::new(0.01, 0.0, 0, 100).unwrap();
    let mut prev_lr = sched.lr_at_step(0);
    for step in 1..100 {
        let lr = sched.lr_at_step(step);
        assert!(
            lr <= prev_lr + 1e-15,
            "lr should be monotonically decreasing: step={step}, prev={prev_lr}, cur={lr}"
        );
        prev_lr = lr;
    }
}

#[test]
fn test_cosine_warmup_then_monotonic_decay() {
    let sched = CosineSchedule::new(0.1, 0.0, 10, 100).unwrap();
    // Warmup phase: monotonically increasing
    let mut prev_lr = 0.0;
    for step in 0..10 {
        let lr = sched.lr_at_step(step);
        assert!(
            lr >= prev_lr - 1e-15,
            "warmup should be increasing: step={step}, prev={prev_lr}, cur={lr}"
        );
        prev_lr = lr;
    }
    // Decay phase: monotonically decreasing
    prev_lr = sched.lr_at_step(10);
    for step in 11..100 {
        let lr = sched.lr_at_step(step);
        assert!(
            lr <= prev_lr + 1e-15,
            "decay should be decreasing: step={step}, prev={prev_lr}, cur={lr}"
        );
        prev_lr = lr;
    }
}

// ============================================================================
// GradScaler config validation edge cases
// ============================================================================

#[test]
fn test_grad_scaler_rejects_nan_init_scale() {
    let result = GradScaler::new(GradScalerConfig {
        init_scale: f64::NAN,
        ..Default::default()
    });
    assert!(result.is_err(), "NaN init_scale should be rejected");
}

#[test]
fn test_grad_scaler_rejects_growth_interval_zero() {
    let result = GradScaler::new(GradScalerConfig {
        growth_interval: 0,
        ..Default::default()
    });
    assert!(result.is_err(), "growth_interval=0 should be rejected");
    let msg = result.unwrap_err().to_string();
    assert!(
        msg.contains("growth_interval"),
        "error should mention growth_interval: {msg}"
    );
}

#[test]
fn test_grad_scaler_rejects_backoff_factor_zero() {
    let result = GradScaler::new(GradScalerConfig {
        backoff_factor: 0.0,
        ..Default::default()
    });
    assert!(result.is_err(), "backoff_factor=0 should be rejected");
}

#[test]
fn test_grad_scaler_rejects_backoff_factor_negative() {
    let result = GradScaler::new(GradScalerConfig {
        backoff_factor: -0.5,
        ..Default::default()
    });
    assert!(
        result.is_err(),
        "negative backoff_factor should be rejected"
    );
}

#[test]
fn test_grad_scaler_rejects_nan_growth_factor() {
    let result = GradScaler::new(GradScalerConfig {
        growth_factor: f64::NAN,
        ..Default::default()
    });
    assert!(result.is_err(), "NaN growth_factor should be rejected");
}

#[test]
fn test_grad_scaler_rejects_inf_growth_factor() {
    let result = GradScaler::new(GradScalerConfig {
        growth_factor: f64::INFINITY,
        ..Default::default()
    });
    assert!(result.is_err(), "infinite growth_factor should be rejected");
}

#[test]
fn test_grad_scaler_rejects_init_below_min() {
    let result = GradScaler::new(GradScalerConfig {
        init_scale: 0.5,
        min_scale: 1.0,
        ..Default::default()
    });
    assert!(
        result.is_err(),
        "init_scale below min_scale should be rejected"
    );
}

#[test]
fn test_grad_scaler_rejects_init_above_max() {
    let result = GradScaler::new(GradScalerConfig {
        init_scale: 100.0,
        max_scale: 50.0,
        ..Default::default()
    });
    assert!(
        result.is_err(),
        "init_scale above max_scale should be rejected"
    );
}

#[test]
fn test_grad_scaler_default_config_values() {
    let config = GradScalerConfig::default();
    assert!((config.init_scale - 65536.0).abs() < f64::EPSILON);
    assert!((config.growth_factor - 2.0).abs() < f64::EPSILON);
    assert!((config.backoff_factor - 0.5).abs() < f64::EPSILON);
    assert_eq!(config.growth_interval, 2000);
    assert!((config.min_scale - 1.0).abs() < f64::EPSILON);
    assert!((config.max_scale - 16_777_216.0).abs() < f64::EPSILON);
}

// ============================================================================
// Adam config validation: NaN/Inf beta values
// ============================================================================

#[test]
fn test_adam_rejects_nan_beta1() {
    let x = scalar_var(1.0);
    let config = AdamConfig {
        beta1: f64::NAN,
        ..AdamConfig::default()
    };
    let result = AdamW::new(vec![x], config);
    assert!(result.is_err(), "NaN beta1 should be rejected");
}

#[test]
fn test_adam_rejects_negative_beta1() {
    let x = scalar_var(1.0);
    let config = AdamConfig {
        beta1: -0.1,
        ..AdamConfig::default()
    };
    let result = AdamW::new(vec![x], config);
    assert!(result.is_err(), "negative beta1 should be rejected");
}

#[test]
fn test_adam_rejects_negative_lr() {
    let x = scalar_var(1.0);
    let config = AdamConfig {
        lr: -0.01,
        ..AdamConfig::default()
    };
    let result = AdamW::new(vec![x], config);
    assert!(result.is_err(), "negative lr should be rejected");
}

#[test]
fn test_adam_rejects_negative_weight_decay() {
    let x = scalar_var(1.0);
    let config = AdamConfig {
        weight_decay: -0.01,
        ..AdamConfig::default()
    };
    let result = AdamW::new(vec![x], config);
    assert!(result.is_err(), "negative weight_decay should be rejected");
}

#[test]
fn test_adam_rejects_eps_negative() {
    let x = scalar_var(1.0);
    let config = AdamConfig {
        eps: -1e-8,
        ..AdamConfig::default()
    };
    let result = AdamW::new(vec![x], config);
    assert!(result.is_err(), "negative eps should be rejected");
}

// ============================================================================
// Multi-step behavior: step_count increments correctly
// ============================================================================

#[test]
fn test_adam_step_count_increments() {
    let x = scalar_var(1.0);
    let mut adam = AdamW::new(
        vec![x.clone()],
        AdamConfig {
            lr: 0.01,
            weight_decay: 0.0,
            ..AdamConfig::default()
        },
    )
    .unwrap();
    assert_eq!(adam.step_count(), 0);

    for i in 1..=5 {
        let t = Arc::new(TrackedTensor::from_var(&x).unwrap());
        let loss = t.sqr().unwrap();
        adam.backward_step(&loss).unwrap();
        assert_eq!(adam.step_count(), i);
    }
}

#[test]
fn test_sgd_config_accessor() {
    let x = scalar_var(1.0);
    let config = SgdConfig {
        lr: 0.05,
        momentum: 0.95,
        weight_decay: 1e-5,
    };
    let sgd = Sgd::new(vec![x], config).unwrap();
    let cfg = sgd.config();
    assert!((cfg.lr - 0.05).abs() < f64::EPSILON);
    assert!((cfg.momentum - 0.95).abs() < f64::EPSILON);
    assert!((cfg.weight_decay - 1e-5).abs() < f64::EPSILON);
}

// ============================================================================
// WarmupSchedule: single-step warmup
// ============================================================================

#[test]
fn test_warmup_single_step() {
    let sched = WarmupSchedule::new(0.1, 1).unwrap();
    // step 0: lr = 0.1 * 0/1 = 0.0
    assert!((sched.lr_at_step(0) - 0.0).abs() < f64::EPSILON);
    // step 1: lr = 0.1 (past warmup)
    assert!((sched.lr_at_step(1) - 0.1).abs() < f64::EPSILON);
}

// ============================================================================
// CosineSchedule: single decay step
// ============================================================================

#[test]
fn test_cosine_single_decay_step() {
    // total_steps=2, warmup=1 -> 1 decay step
    let sched = CosineSchedule::new(0.01, 0.0, 1, 2).unwrap();
    // step 0: warmup -> 0.0
    assert!((sched.lr_at_step(0) - 0.0).abs() < f64::EPSILON);
    // step 1: end of warmup / start of decay -> base_lr
    let lr1 = sched.lr_at_step(1);
    assert!(
        (lr1 - 0.01).abs() < 1e-10,
        "at warmup end should be base_lr, got {lr1}"
    );
    // step 2: past total_steps -> min_lr
    assert!((sched.lr_at_step(2) - 0.0).abs() < 1e-10);
}

// ============================================================================
// GradScaler: save_state and load_state round-trip
// ============================================================================

#[test]
fn test_grad_scaler_state_roundtrip() {
    let scaler = GradScaler::new(GradScalerConfig {
        init_scale: 512.0,
        growth_interval: 10,
        ..Default::default()
    })
    .unwrap();

    // Save initial state (scale=512, tracker=0)
    let state = scaler.save_state();
    assert!((state.scale - 512.0).abs() < f64::EPSILON);
    assert_eq!(state.growth_tracker, 0);

    // Load into a fresh scaler with different init_scale
    let mut scaler2 = GradScaler::new(GradScalerConfig {
        init_scale: 1024.0,
        growth_interval: 10,
        ..Default::default()
    })
    .unwrap();
    assert!((scaler2.scale_factor() - 1024.0).abs() < f64::EPSILON);

    scaler2.load_state(&state).unwrap();
    // Should have loaded the scale from the state
    assert!((scaler2.scale_factor() - 512.0).abs() < f64::EPSILON);
}

#[test]
fn test_grad_scaler_load_state_rejects_non_finite() {
    let mut scaler = GradScaler::new(GradScalerConfig::default()).unwrap();
    let bad_state = crate::checkpoint::GradScalerState {
        scale: f64::NAN,
        growth_tracker: 0,
    };
    let result = scaler.load_state(&bad_state);
    assert!(result.is_err(), "NaN scale in state should be rejected");
}

#[test]
fn test_grad_scaler_load_state_rejects_zero_scale() {
    let mut scaler = GradScaler::new(GradScalerConfig::default()).unwrap();
    let bad_state = crate::checkpoint::GradScalerState {
        scale: 0.0,
        growth_tracker: 0,
    };
    let result = scaler.load_state(&bad_state);
    assert!(result.is_err(), "zero scale in state should be rejected");
}

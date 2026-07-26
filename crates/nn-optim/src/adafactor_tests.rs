#![allow(deprecated)]
// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Tests for AdaFactor optimizer.

use super::*;
use nn_autodiff::{backward, TrackedTensor, Var};
use nn_core::dyn_tensor::DynTensor;
use nn_core::test_utils::cpu;
use std::sync::Arc;

use crate::error::OptimError;
use crate::lr_schedule::LrSchedule;
use crate::optimizer::Optimizer;

#[test]
fn test_adafactor_basic_step_vector() {
    // 1D parameter — uses full second moment (not factored)
    let var = Var::new(DynTensor::from_vec(vec![1.0, 2.0, 3.0], &[3], &cpu()).unwrap());
    let config = AdaFactorConfig {
        lr: 0.1,
        relative_step: false,
        ..Default::default()
    };
    let mut opt = AdaFactor::new(vec![var.clone()], config).unwrap();

    // Simple loss = sum(x^2) => grad = 2*x
    let t = Arc::new(TrackedTensor::from_var(&var).unwrap());
    let sq = t.mul(&t).unwrap();
    let loss = sq.sum_keepdim(0).unwrap();
    let grads = backward(&loss).unwrap();
    opt.step(&grads).unwrap();

    // Parameters should decrease (gradient descent)
    let vals = var.data().unwrap().to_flat_vec::<f32>().unwrap();
    assert!(vals[0] < 1.0, "param[0] should decrease, got {}", vals[0]);
    assert!(vals[1] < 2.0, "param[1] should decrease, got {}", vals[1]);
    assert!(vals[2] < 3.0, "param[2] should decrease, got {}", vals[2]);
}

#[test]
fn test_adafactor_basic_step_matrix() {
    // 2D parameter — uses factored second moments
    let var =
        Var::new(DynTensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], &cpu()).unwrap());
    let config = AdaFactorConfig {
        lr: 0.1,
        relative_step: false,
        ..Default::default()
    };
    let mut opt = AdaFactor::new(vec![var.clone()], config).unwrap();

    let t = Arc::new(TrackedTensor::from_var(&var).unwrap());
    let sq = t.mul(&t).unwrap();
    let loss = sq.sum_keepdim(0).unwrap().sum_keepdim(1).unwrap();
    let grads = backward(&loss).unwrap();
    opt.step(&grads).unwrap();

    let vals = var.data().unwrap().to_flat_vec::<f32>().unwrap();
    // All params should decrease with positive initial values
    for (i, &v) in vals.iter().enumerate() {
        let init = (i + 1) as f32;
        assert!(v < init, "param[{i}] should decrease from {init}, got {v}");
    }
}

#[test]
fn test_adafactor_multiple_steps_converge() {
    // Multiple steps should keep reducing the loss
    let var = Var::new(DynTensor::from_vec(vec![5.0, -3.0], &[2], &cpu()).unwrap());
    let config = AdaFactorConfig {
        lr: 0.1,
        relative_step: false,
        ..Default::default()
    };
    let mut opt = AdaFactor::new(vec![var.clone()], config).unwrap();

    let mut prev_loss_val = f64::MAX;
    for _ in 0..20 {
        let t = Arc::new(TrackedTensor::from_var(&var).unwrap());
        let sq = t.mul(&t).unwrap();
        let loss = sq.sum_keepdim(0).unwrap();
        let loss_val: f64 = loss
            .tensor()
            .to_flat_vec::<f32>()
            .unwrap()
            .iter()
            .map(|&v| f64::from(v))
            .sum();
        assert!(
            loss_val < prev_loss_val,
            "loss should decrease: {loss_val} >= {prev_loss_val}"
        );
        prev_loss_val = loss_val;
        let grads = backward(&loss).unwrap();
        opt.step(&grads).unwrap();
    }
    // Starting loss ~ 34 (5^2 + 3^2). After 20 steps it should decrease significantly.
    // AdaFactor normalizes updates by gradient RMS, so convergence is slower than Adam.
    assert!(
        prev_loss_val < 20.0,
        "loss should decrease after 20 steps, got {prev_loss_val}"
    );
}

#[test]
fn test_adafactor_with_momentum() {
    let var = Var::new(DynTensor::from_vec(vec![3.0, -2.0], &[2], &cpu()).unwrap());
    let config = AdaFactorConfig {
        lr: 0.1,
        relative_step: false,
        beta1: Some(0.9),
        ..Default::default()
    };
    let mut opt = AdaFactor::new(vec![var.clone()], config).unwrap();

    let t = Arc::new(TrackedTensor::from_var(&var).unwrap());
    let sq = t.mul(&t).unwrap();
    let loss = sq.sum_keepdim(0).unwrap();
    let grads = backward(&loss).unwrap();
    opt.step(&grads).unwrap();

    let vals = var.data().unwrap().to_flat_vec::<f32>().unwrap();
    assert!(vals[0] < 3.0, "param should decrease with momentum");
    assert!(vals[1] > -2.0, "negative param should increase toward zero");
}

#[test]
fn test_adafactor_relative_step() {
    let var = Var::new(DynTensor::from_vec(vec![10.0, 20.0], &[2], &cpu()).unwrap());
    let config = AdaFactorConfig {
        relative_step: true,
        ..Default::default()
    };
    let mut opt = AdaFactor::new(vec![var.clone()], config).unwrap();

    let t = Arc::new(TrackedTensor::from_var(&var).unwrap());
    let sq = t.mul(&t).unwrap();
    let loss = sq.sum_keepdim(0).unwrap();
    let grads = backward(&loss).unwrap();
    opt.step(&grads).unwrap();

    // Relative step should still move parameters
    let vals = var.data().unwrap().to_flat_vec::<f32>().unwrap();
    assert!(vals[0] < 10.0, "param should decrease with relative step");
}

#[test]
fn test_adafactor_weight_decay() {
    let var = Var::new(DynTensor::from_vec(vec![10.0, 10.0], &[2], &cpu()).unwrap());
    let config = AdaFactorConfig {
        lr: 0.01,
        relative_step: false,
        weight_decay: 0.1,
        ..Default::default()
    };
    let mut opt_wd = AdaFactor::new(vec![var.clone()], config).unwrap();

    let t = Arc::new(TrackedTensor::from_var(&var).unwrap());
    let sq = t.mul(&t).unwrap();
    let loss = sq.sum_keepdim(0).unwrap();
    let grads = backward(&loss).unwrap();
    opt_wd.step(&grads).unwrap();

    let vals = var.data().unwrap().to_flat_vec::<f32>().unwrap();
    // Weight decay should shrink parameters extra
    assert!(vals[0] < 10.0, "weight decay should shrink params");
}

#[test]
fn test_adafactor_config_validation() {
    let var = Var::new(DynTensor::from_vec(vec![1.0], &[1], &cpu()).unwrap());

    // Invalid beta1
    let config = AdaFactorConfig {
        beta1: Some(1.5),
        ..Default::default()
    };
    assert!(AdaFactor::new(vec![var.clone()], config).is_err());

    // Invalid eps_denom
    let config = AdaFactorConfig {
        eps_denom: 0.0,
        ..Default::default()
    };
    assert!(AdaFactor::new(vec![var.clone()], config).is_err());

    // NaN eps_rms
    let config = AdaFactorConfig {
        eps_rms: f64::NAN,
        ..Default::default()
    };
    assert!(AdaFactor::new(vec![var], config).is_err());
}

#[test]
fn test_adafactor_step_count() {
    let var = Var::new(DynTensor::from_vec(vec![1.0], &[1], &cpu()).unwrap());
    let config = AdaFactorConfig {
        lr: 0.1,
        relative_step: false,
        ..Default::default()
    };
    let mut opt = AdaFactor::new(vec![var.clone()], config).unwrap();
    assert_eq!(opt.step_count(), 0);

    let t = Arc::new(TrackedTensor::from_var(&var).unwrap());
    let loss = t.mul(&t).unwrap().sum_keepdim(0).unwrap();
    let grads = backward(&loss).unwrap();
    opt.step(&grads).unwrap();
    assert_eq!(opt.step_count(), 1);
}

#[test]
fn test_adafactor_backward_step() {
    let var = Var::new(DynTensor::from_vec(vec![3.0, -1.0], &[2], &cpu()).unwrap());
    let config = AdaFactorConfig {
        lr: 0.1,
        relative_step: false,
        ..Default::default()
    };
    let mut opt = AdaFactor::new(vec![var.clone()], config).unwrap();

    let t = Arc::new(TrackedTensor::from_var(&var).unwrap());
    let sq = t.mul(&t).unwrap();
    let loss = sq.sum_keepdim(0).unwrap();
    opt.backward_step(&loss).unwrap();

    let vals = var.data().unwrap().to_flat_vec::<f32>().unwrap();
    assert!(vals[0] < 3.0, "backward_step should update params");
    assert_eq!(opt.step_count(), 1);
}

#[test]
fn test_adafactor_no_grad_skips_var() {
    let var1 = Var::new(DynTensor::from_vec(vec![1.0, 2.0], &[2], &cpu()).unwrap());
    let var2 = Var::new(DynTensor::from_vec(vec![5.0, 6.0], &[2], &cpu()).unwrap());
    let config = AdaFactorConfig {
        lr: 0.1,
        relative_step: false,
        ..Default::default()
    };
    let mut opt = AdaFactor::new(vec![var1.clone(), var2.clone()], config).unwrap();

    // Only compute gradients for var1
    let t1 = Arc::new(TrackedTensor::from_var(&var1).unwrap());
    let loss = t1.mul(&t1).unwrap().sum_keepdim(0).unwrap();
    let grads = backward(&loss).unwrap();
    opt.step(&grads).unwrap();

    // var2 should be unchanged (no gradient)
    let vals2 = var2.data().unwrap().to_flat_vec::<f32>().unwrap();
    assert_eq!(vals2, vec![5.0, 6.0], "var2 should be unchanged");
}

// -- decay_rate validation tests --------------------------------------------

/// AdaFactor must reject positive decay_rate at construction time.
/// Positive decay_rate makes `rho_t = 1 - t^decay_rate` decrease toward
/// negative values, zeroing out second-moment history and producing
/// non-functional optimization. The paper default is -0.8.
#[test]
fn test_adafactor_positive_decay_rate_rejected() {
    let var = Var::new(DynTensor::from_vec(vec![1.0, 2.0, 3.0], &[3], &cpu()).unwrap());
    let config = AdaFactorConfig {
        lr: 0.01,
        relative_step: false,
        decay_rate: 0.5,
        ..Default::default()
    };
    let result = AdaFactor::new(vec![var], config);
    assert!(result.is_err(), "positive decay_rate should be rejected");
    let err = result.unwrap_err();
    let msg = format!("{err}");
    assert!(
        msg.contains("decay_rate"),
        "error should mention decay_rate: {msg}"
    );
}

// -- NaN gradient rejection ------------------------------------------------

/// AdaFactor must reject NaN gradients to prevent moment corruption.
#[test]
fn test_adafactor_rejects_nan_gradient() {
    let x = Var::new(DynTensor::from_vec(vec![1.0, 2.0, 3.0], &[3], &cpu()).unwrap());
    let config = AdaFactorConfig::default();
    let mut af = AdaFactor::new(vec![x.clone()], config).unwrap();

    // Compute real grads then inject NaN
    let t = Arc::new(TrackedTensor::from_var(&x).unwrap());
    let loss = t.sqr().unwrap().sum_keepdim(0).unwrap();
    let mut grads = backward(&loss).unwrap();

    for (_, grad) in grads.var_grads_mut() {
        *grad = DynTensor::from_vec(vec![f32::NAN, 1.0, 2.0], &[3], &cpu()).unwrap();
    }

    let result = af.step(&grads);
    assert!(
        matches!(result, Err(OptimError::NonFiniteGradient { .. })),
        "AdaFactor should reject non-finite gradients, got: {result:?}"
    );
}

/// AdaFactor must reject Inf gradients (matching SGD's test_sgd_rejects_inf_gradient).
#[test]
fn test_adafactor_rejects_inf_gradient() {
    let x = Var::new(DynTensor::from_vec(vec![1.0, 2.0, 3.0], &[3], &cpu()).unwrap());
    let config = AdaFactorConfig::default();
    let mut af = AdaFactor::new(vec![x.clone()], config).unwrap();

    let t = Arc::new(TrackedTensor::from_var(&x).unwrap());
    let loss = t.sqr().unwrap().sum_keepdim(0).unwrap();
    let mut grads = backward(&loss).unwrap();

    for (_, grad) in grads.var_grads_mut() {
        *grad = DynTensor::from_vec(vec![f32::INFINITY, 1.0, 2.0], &[3], &cpu()).unwrap();
    }

    let result = af.step(&grads);
    assert!(
        matches!(result, Err(OptimError::NonFiniteGradient { .. })),
        "AdaFactor should reject Inf gradients, got: {result:?}"
    );
}

// -- Zero learning rate ---------------------------------------------------

/// With lr=0 and relative_step=false, AdaFactor should produce zero-magnitude updates.
#[test]
fn test_adafactor_zero_lr_no_update() {
    let var = Var::new(DynTensor::from_vec(vec![5.0, -3.0], &[2], &cpu()).unwrap());
    let config = AdaFactorConfig {
        lr: 0.0,
        relative_step: false,
        weight_decay: 0.0,
        ..Default::default()
    };
    let mut opt = AdaFactor::new(vec![var.clone()], config).unwrap();

    let t = Arc::new(TrackedTensor::from_var(&var).unwrap());
    let loss = t.sqr().unwrap().sum_keepdim(0).unwrap();
    let grads = backward(&loss).unwrap();
    opt.step(&grads).unwrap();

    let vals = var.data().unwrap().to_flat_vec::<f32>().unwrap();
    assert!(
        (vals[0] - 5.0).abs() < 1e-7,
        "lr=0 should produce no update, got {}",
        vals[0]
    );
    assert!(
        (vals[1] - (-3.0)).abs() < 1e-7,
        "lr=0 should produce no update, got {}",
        vals[1]
    );
}

// -- rho_t numerical value regression (#1993) --------------------------------

/// Verify rho_t matches the AdaFactor paper (Shazeer & Stern, 2018) formula:
///   rho_t = 1 - t^decay_rate
///
/// With default decay_rate = -0.8:
///   step 1 → t=1, rho = 1 - 1^(-0.8) = 0.0
///   step 2 → t=2, rho = 1 - 2^(-0.8) ≈ 0.4257
///   step 10 → t=10, rho = 1 - 10^(-0.8) ≈ 0.8415
///
/// The old code had `saturating_add(1)` which made step 1 use t=2,
/// returning ~0.4257 instead of 0.0. This test catches that off-by-one.
#[test]
fn test_rho_t_matches_paper_formula() {
    let var = Var::new(DynTensor::from_vec(vec![1.0], &[1], &cpu()).unwrap());
    let config = AdaFactorConfig::default();
    assert!(
        (config.decay_rate - (-0.8)).abs() < 1e-15,
        "test assumes default decay_rate=-0.8"
    );

    let mut opt = AdaFactor::new(vec![var.clone()], config).unwrap();

    // Before any step: step_t = 0, rho_t would be 1 - 0^(-0.8) = NaN → clamp to 0.0
    assert_eq!(opt.step_count(), 0);
    // rho_t() is private but accessible from the test module via super::*
    let rho_before = opt.rho_t();
    // 0.0^(-0.8) = Inf, so 1 - Inf = -Inf, clamped to 0.0
    assert!(
        (rho_before - 0.0).abs() < 1e-15,
        "rho_t at step_t=0 should be clamped to 0.0, got {rho_before}"
    );

    // After first step: step_t = 1, rho_t = 1 - 1^(-0.8) = 0.0
    let t = Arc::new(TrackedTensor::from_var(&var).unwrap());
    let loss = t.sqr().unwrap().sum_keepdim(0).unwrap();
    let grads = backward(&loss).unwrap();
    opt.step(&grads).unwrap();

    assert_eq!(opt.step_count(), 1);
    let rho_step1 = opt.rho_t();
    assert!(
        (rho_step1 - 0.0).abs() < 1e-15,
        "rho_t at step 1 should be 0.0 (1 - 1^(-0.8) = 0), got {rho_step1}"
    );

    // After second step: step_t = 2, rho_t = 1 - 2^(-0.8) ≈ 0.42565
    let t2 = Arc::new(TrackedTensor::from_var(&var).unwrap());
    let loss2 = t2.sqr().unwrap().sum_keepdim(0).unwrap();
    let grads2 = backward(&loss2).unwrap();
    opt.step(&grads2).unwrap();

    assert_eq!(opt.step_count(), 2);
    let rho_step2 = opt.rho_t();
    let expected_step2 = 1.0 - 2.0_f64.powf(-0.8);
    assert!(
        (rho_step2 - expected_step2).abs() < 1e-10,
        "rho_t at step 2 should be {expected_step2:.6}, got {rho_step2:.6}"
    );

    // After tenth step: step_t = 10, rho_t = 1 - 10^(-0.8) ≈ 0.8415
    for _ in 3..=10 {
        let ti = Arc::new(TrackedTensor::from_var(&var).unwrap());
        let li = ti.sqr().unwrap().sum_keepdim(0).unwrap();
        let gi = backward(&li).unwrap();
        opt.step(&gi).unwrap();
    }
    assert_eq!(opt.step_count(), 10);
    let rho_step10 = opt.rho_t();
    let expected_step10 = 1.0 - 10.0_f64.powf(-0.8);
    assert!(
        (rho_step10 - expected_step10).abs() < 1e-10,
        "rho_t at step 10 should be {expected_step10:.6}, got {rho_step10:.6}"
    );
}

// -- Empty parameter list -------------------------------------------------

/// AdaFactor with zero parameters should accept step() without error.
#[test]
fn test_adafactor_empty_params() {
    let x = Var::new(DynTensor::from_vec(vec![1.0], &[1], &cpu()).unwrap());
    let mut opt = AdaFactor::new(vec![], AdaFactorConfig::default()).unwrap();

    let t = Arc::new(TrackedTensor::from_var(&x).unwrap());
    let loss = t.sqr().unwrap().sum_keepdim(0).unwrap();
    let grads = backward(&loss).unwrap();

    opt.step(&grads).unwrap();
    assert_eq!(opt.step_count(), 1);
}

// -- Default config verification ------------------------------------------

/// Verify all AdaFactorConfig default values match the paper (Shazeer & Stern, 2018).
#[test]
fn test_adafactor_config_defaults() {
    let config = AdaFactorConfig::default();
    assert!((config.lr - 1e-3).abs() < 1e-15, "default lr");
    assert!(
        !config.relative_step,
        "default relative_step should be false"
    );
    assert!((config.eps_rms - 1e-3).abs() < 1e-15, "default eps_rms");
    assert!(
        (config.eps_denom - 1e-30).abs() < 1e-45,
        "default eps_denom"
    );
    assert!(
        (config.decay_rate - (-0.8)).abs() < 1e-15,
        "default decay_rate"
    );
    assert!(config.beta1.is_none(), "default beta1 should be None");
    assert!(
        (config.weight_decay - 0.0).abs() < 1e-15,
        "default weight_decay"
    );
}

// -- Learning rate schedule integration -----------------------------------

/// AdaFactor works with WarmupSchedule via step_with_schedule.
#[test]
fn test_adafactor_warmup_schedule_integration() {
    let var = Var::new(DynTensor::from_vec(vec![5.0, -3.0, 2.0], &[3], &cpu()).unwrap());
    let config = AdaFactorConfig {
        lr: 0.01,
        relative_step: false,
        ..Default::default()
    };
    let mut opt = AdaFactor::new(vec![var.clone()], config).unwrap();
    let sched = crate::lr_schedule::WarmupSchedule::new(0.01, 10).unwrap();

    // Step through warmup phase.
    for step in 0..15 {
        let t = Arc::new(TrackedTensor::from_var(&var).unwrap());
        let loss = t.sqr().unwrap().sum_keepdim(0).unwrap();
        let grads = backward(&loss).unwrap();
        crate::lr_schedule::step_with_schedule(&mut opt, &grads, &sched, step).unwrap();
    }

    // After warmup, LR should be at base_lr.
    assert!(
        (opt.learning_rate() - 0.01).abs() < 1e-10,
        "after warmup LR should be base_lr, got {}",
        opt.learning_rate()
    );

    // Parameters should have moved from initial values.
    let vals = var.data().unwrap().to_flat_vec::<f32>().unwrap();
    assert!(vals[0] < 5.0, "param[0] should decrease from 5.0");
}

/// AdaFactor works with CosineSchedule.
#[test]
fn test_adafactor_cosine_schedule_integration() {
    let var = Var::new(DynTensor::from_vec(vec![3.0, 3.0], &[2], &cpu()).unwrap());
    let config = AdaFactorConfig {
        lr: 0.0, // will be overwritten by schedule
        relative_step: false,
        ..Default::default()
    };
    let mut opt = AdaFactor::new(vec![var.clone()], config).unwrap();
    let sched = crate::lr_schedule::CosineSchedule::new(0.05, 0.001, 5, 100).unwrap();

    for step in 0..50 {
        let t = Arc::new(TrackedTensor::from_var(&var).unwrap());
        let loss = t.sqr().unwrap().sum_keepdim(0).unwrap();
        let grads = backward(&loss).unwrap();
        crate::lr_schedule::step_with_schedule(&mut opt, &grads, &sched, step).unwrap();
    }

    // Last step_with_schedule call used step=49, so LR should match that.
    let expected_lr = sched.lr_at_step(49);
    assert!(
        (opt.learning_rate() - expected_lr).abs() < 1e-10,
        "LR should match schedule at step 49: expected {expected_lr}, got {}",
        opt.learning_rate()
    );
    assert!(expected_lr < 0.05, "LR should be decayed below base_lr");
    assert!(expected_lr > 0.001, "LR should be above min_lr");
}

// -- Weight decay comparison (with vs. without) ----------------------------

/// Weight decay causes stronger parameter shrinkage than without it.
#[test]
fn test_adafactor_weight_decay_comparison() {
    let init_vals = vec![10.0, -10.0, 5.0, -5.0];
    let var_wd = Var::new(DynTensor::from_vec(init_vals.clone(), &[4], &cpu()).unwrap());
    let var_no_wd = Var::new(DynTensor::from_vec(init_vals, &[4], &cpu()).unwrap());

    let config_wd = AdaFactorConfig {
        lr: 0.01,
        relative_step: false,
        weight_decay: 0.1,
        ..Default::default()
    };
    let config_no_wd = AdaFactorConfig {
        lr: 0.01,
        relative_step: false,
        weight_decay: 0.0,
        ..Default::default()
    };
    let mut opt_wd = AdaFactor::new(vec![var_wd.clone()], config_wd).unwrap();
    let mut opt_no_wd = AdaFactor::new(vec![var_no_wd.clone()], config_no_wd).unwrap();

    for _ in 0..10 {
        let t_wd = Arc::new(TrackedTensor::from_var(&var_wd).unwrap());
        let loss_wd = t_wd.sqr().unwrap().sum_keepdim(0).unwrap();
        let grads_wd = backward(&loss_wd).unwrap();
        opt_wd.step(&grads_wd).unwrap();

        let t_no = Arc::new(TrackedTensor::from_var(&var_no_wd).unwrap());
        let loss_no = t_no.sqr().unwrap().sum_keepdim(0).unwrap();
        let grads_no = backward(&loss_no).unwrap();
        opt_no_wd.step(&grads_no).unwrap();
    }

    let vals_wd = var_wd.data().unwrap().to_flat_vec::<f32>().unwrap();
    let vals_no_wd = var_no_wd.data().unwrap().to_flat_vec::<f32>().unwrap();

    // Weight decay should push params closer to zero: |param_wd| < |param_no_wd|
    for (i, (&wd, &no_wd)) in vals_wd.iter().zip(vals_no_wd.iter()).enumerate() {
        assert!(
            wd.abs() < no_wd.abs() + 1e-6,
            "weight decay should shrink param[{i}] more: |wd|={}, |no_wd|={}",
            wd.abs(),
            no_wd.abs()
        );
    }
}

// -- 2D factored convergence -----------------------------------------------

/// AdaFactor converges on 2D matrix with factored second moments over many steps.
#[test]
fn test_adafactor_2d_factored_convergence() {
    let var = Var::new(
        DynTensor::from_vec(vec![5.0, -3.0, 2.0, -4.0, 1.0, 6.0], &[2, 3], &cpu()).unwrap(),
    );
    let config = AdaFactorConfig {
        lr: 0.1,
        relative_step: false,
        ..Default::default()
    };
    let mut opt = AdaFactor::new(vec![var.clone()], config).unwrap();

    let initial_loss: f64 = {
        let vals = var.data().unwrap().to_flat_vec::<f32>().unwrap();
        vals.iter().map(|&v| f64::from(v) * f64::from(v)).sum()
    };

    for _ in 0..30 {
        let t = Arc::new(TrackedTensor::from_var(&var).unwrap());
        let sq = t.sqr().unwrap();
        let loss = sq.sum_keepdim(0).unwrap().sum_keepdim(1).unwrap();
        let grads = backward(&loss).unwrap();
        opt.step(&grads).unwrap();
    }

    let final_loss: f64 = {
        let vals = var.data().unwrap().to_flat_vec::<f32>().unwrap();
        vals.iter().map(|&v| f64::from(v) * f64::from(v)).sum()
    };

    assert!(
        final_loss < initial_loss * 0.5,
        "factored 2D should converge: initial={initial_loss}, final={final_loss}"
    );
}

// -- Config validation: additional edge cases ------------------------------

/// Negative learning rate is rejected.
#[test]
fn test_adafactor_negative_lr_rejected() {
    let var = Var::new(DynTensor::from_vec(vec![1.0], &[1], &cpu()).unwrap());
    let config = AdaFactorConfig {
        lr: -0.01,
        ..Default::default()
    };
    let result = AdaFactor::new(vec![var], config);
    assert!(result.is_err(), "negative lr should be rejected");
    let msg = format!("{}", result.unwrap_err());
    assert!(msg.contains("lr"), "error should mention lr: {msg}");
}

/// Infinite learning rate is rejected.
#[test]
fn test_adafactor_inf_lr_rejected() {
    let var = Var::new(DynTensor::from_vec(vec![1.0], &[1], &cpu()).unwrap());
    let config = AdaFactorConfig {
        lr: f64::INFINITY,
        ..Default::default()
    };
    assert!(AdaFactor::new(vec![var], config).is_err());
}

/// Negative weight decay is rejected.
#[test]
fn test_adafactor_negative_weight_decay_rejected() {
    let var = Var::new(DynTensor::from_vec(vec![1.0], &[1], &cpu()).unwrap());
    let config = AdaFactorConfig {
        weight_decay: -0.1,
        ..Default::default()
    };
    let result = AdaFactor::new(vec![var], config);
    assert!(result.is_err(), "negative weight_decay should be rejected");
    let msg = format!("{}", result.unwrap_err());
    assert!(
        msg.contains("weight_decay"),
        "error should mention weight_decay: {msg}"
    );
}

/// beta1 exactly at 1.0 is rejected (must be < 1.0).
#[test]
fn test_adafactor_beta1_at_boundary_rejected() {
    let var = Var::new(DynTensor::from_vec(vec![1.0], &[1], &cpu()).unwrap());
    let config = AdaFactorConfig {
        beta1: Some(1.0),
        ..Default::default()
    };
    assert!(
        AdaFactor::new(vec![var], config).is_err(),
        "beta1=1.0 should be rejected"
    );
}

/// Negative beta1 is rejected.
#[test]
fn test_adafactor_negative_beta1_rejected() {
    let var = Var::new(DynTensor::from_vec(vec![1.0], &[1], &cpu()).unwrap());
    let config = AdaFactorConfig {
        beta1: Some(-0.1),
        ..Default::default()
    };
    assert!(
        AdaFactor::new(vec![var], config).is_err(),
        "negative beta1 should be rejected"
    );
}

/// NaN decay_rate is rejected.
#[test]
fn test_adafactor_nan_decay_rate_rejected() {
    let var = Var::new(DynTensor::from_vec(vec![1.0], &[1], &cpu()).unwrap());
    let config = AdaFactorConfig {
        decay_rate: f64::NAN,
        ..Default::default()
    };
    assert!(
        AdaFactor::new(vec![var], config).is_err(),
        "NaN decay_rate should be rejected"
    );
}

/// Negative eps_denom is rejected.
#[test]
fn test_adafactor_negative_eps_denom_rejected() {
    let var = Var::new(DynTensor::from_vec(vec![1.0], &[1], &cpu()).unwrap());
    let config = AdaFactorConfig {
        eps_denom: -1e-30,
        ..Default::default()
    };
    assert!(
        AdaFactor::new(vec![var], config).is_err(),
        "negative eps_denom should be rejected"
    );
}

// -- set_learning_rate ----------------------------------------------------

/// set_learning_rate updates the LR and rejects invalid values.
#[test]
fn test_adafactor_set_learning_rate() {
    let var = Var::new(DynTensor::from_vec(vec![1.0], &[1], &cpu()).unwrap());
    let mut opt = AdaFactor::new(vec![var], AdaFactorConfig::default()).unwrap();

    assert!((opt.learning_rate() - 1e-3).abs() < 1e-15);

    opt.set_learning_rate(0.05).unwrap();
    assert!((opt.learning_rate() - 0.05).abs() < 1e-15);

    // Invalid LR should be rejected.
    assert!(opt.set_learning_rate(-1.0).is_err());
    assert!(opt.set_learning_rate(f64::NAN).is_err());
    assert!(opt.set_learning_rate(f64::INFINITY).is_err());

    // LR should remain unchanged after rejected set.
    assert!((opt.learning_rate() - 0.05).abs() < 1e-15);
}

// -- Multiple steps with momentum on matrix --------------------------------

/// Momentum (beta1) accelerates convergence on 2D parameters with factored moments.
#[test]
fn test_adafactor_momentum_accelerates_2d() {
    let init = vec![4.0, -2.0, 3.0, -1.0, 2.0, -3.0];
    let var_mom = Var::new(DynTensor::from_vec(init.clone(), &[2, 3], &cpu()).unwrap());
    let var_no_mom = Var::new(DynTensor::from_vec(init, &[2, 3], &cpu()).unwrap());

    let config_mom = AdaFactorConfig {
        lr: 0.1,
        relative_step: false,
        beta1: Some(0.9),
        ..Default::default()
    };
    let config_no_mom = AdaFactorConfig {
        lr: 0.1,
        relative_step: false,
        beta1: None,
        ..Default::default()
    };
    let mut opt_mom = AdaFactor::new(vec![var_mom.clone()], config_mom).unwrap();
    let mut opt_no = AdaFactor::new(vec![var_no_mom.clone()], config_no_mom).unwrap();

    for _ in 0..20 {
        let t_m = Arc::new(TrackedTensor::from_var(&var_mom).unwrap());
        let loss_m = t_m
            .sqr()
            .unwrap()
            .sum_keepdim(0)
            .unwrap()
            .sum_keepdim(1)
            .unwrap();
        let grads_m = backward(&loss_m).unwrap();
        opt_mom.step(&grads_m).unwrap();

        let t_n = Arc::new(TrackedTensor::from_var(&var_no_mom).unwrap());
        let loss_n = t_n
            .sqr()
            .unwrap()
            .sum_keepdim(0)
            .unwrap()
            .sum_keepdim(1)
            .unwrap();
        let grads_n = backward(&loss_n).unwrap();
        opt_no.step(&grads_n).unwrap();
    }

    let loss_mom: f64 = var_mom
        .data()
        .unwrap()
        .to_flat_vec::<f32>()
        .unwrap()
        .iter()
        .map(|&v| f64::from(v).powi(2))
        .sum();
    let loss_no: f64 = var_no_mom
        .data()
        .unwrap()
        .to_flat_vec::<f32>()
        .unwrap()
        .iter()
        .map(|&v| f64::from(v).powi(2))
        .sum();

    // Initial loss is ~43. Both should decrease significantly after 20 steps.
    assert!(
        loss_mom < 30.0,
        "momentum variant should converge, got {loss_mom}"
    );
    assert!(
        loss_no < 30.0,
        "no-momentum variant should converge, got {loss_no}"
    );
}

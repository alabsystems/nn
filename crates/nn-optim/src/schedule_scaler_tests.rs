#![allow(deprecated)]
// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Comprehensive tests for learning rate schedules, gradient scaler, and
//! gradient clipping. Supplements per-module tests with cross-cutting
//! integration scenarios and edge cases.

use std::sync::Arc;

use nn_autodiff::{backward, GradStore, TrackedTensor, Var};
use nn_core::dyn_tensor::DynTensor;
use nn_core::Device;

use crate::grad_clip::{clip_grad_norm, clip_grad_value};
use crate::grad_scaler::{GradScaler, GradScalerConfig};
use crate::lr_schedule::{step_with_schedule, CosineSchedule, LrSchedule, WarmupSchedule};
use crate::optimizer::Optimizer;
use crate::sgd::{Sgd, SgdConfig};

// -- helpers ------------------------------------------------------------------

fn cpu() -> Device {
    Device::Cpu
}

fn scalar_var(val: f32) -> Var {
    Var::new(DynTensor::from_vec(vec![val], &[1], &cpu()).unwrap())
}

/// Build a GradStore where the single var has gradient equal to `grad_values`.
fn grads_for(grad_values: &[f32]) -> (Var, GradStore) {
    let n = grad_values.len();
    let var = Var::new(DynTensor::from_vec(vec![1.0; n], &[n], &cpu()).unwrap());
    let t = Arc::new(TrackedTensor::from_var(&var).unwrap());
    let target = Arc::new(TrackedTensor::from_tensor(
        DynTensor::from_vec(grad_values.to_vec(), &[n], &cpu()).unwrap(),
    ));
    let product = t.mul(&target).unwrap();
    let loss = product.sum_keepdim(0).unwrap();
    let grads = backward(&loss).unwrap();
    (var, grads)
}

/// Build a finite GradStore for a scaler, run unscale_and_check (found_inf=false).
fn scaler_clean_step(scaler: &mut GradScaler) {
    let var = Var::new(DynTensor::from_vec(vec![1.0f32], &[1], &cpu()).unwrap());
    let t = Arc::new(TrackedTensor::from_var(&var).unwrap());
    let loss = t.mul_scalar(1.0).unwrap();
    let scaled = scaler.scale_loss(&loss).unwrap();
    let mut grads = backward(&scaled).unwrap();
    let ok = scaler.unscale_and_check(&mut grads).unwrap();
    assert!(ok, "clean step should return ok=true");
}

/// Build an inf GradStore for a scaler, run unscale_and_check (found_inf=true).
fn scaler_inf_step(scaler: &mut GradScaler) {
    let var = Var::new(DynTensor::from_vec(vec![1.0f32], &[1], &cpu()).unwrap());
    let t = Arc::new(TrackedTensor::from_var(&var).unwrap());
    let loss = t.mul_scalar(1.0).unwrap();
    let scaled = scaler.scale_loss(&loss).unwrap();
    let mut grads = backward(&scaled).unwrap();
    // Inject inf into gradients
    let inf_grad = DynTensor::from_vec(vec![f32::INFINITY], &[1], &cpu()).unwrap();
    for (_, grad) in grads.var_grads_mut() {
        *grad = inf_grad.clone();
    }
    let ok = scaler.unscale_and_check(&mut grads).unwrap();
    assert!(!ok, "inf step should return ok=false");
}

// == WarmupSchedule ===========================================================

#[test]
fn test_warmup_negative_base_lr_rejected() {
    let err = WarmupSchedule::new(-0.01, 100);
    assert!(err.is_err(), "negative base_lr should be rejected");
    let msg = format!("{}", err.unwrap_err());
    assert!(msg.contains("base_lr"), "error should cite base_lr: {msg}");
}

#[test]
fn test_warmup_nan_base_lr_rejected() {
    let err = WarmupSchedule::new(f64::NAN, 100);
    assert!(err.is_err(), "NaN base_lr should be rejected");
}

#[test]
fn test_warmup_inf_base_lr_rejected() {
    let err = WarmupSchedule::new(f64::INFINITY, 100);
    assert!(err.is_err(), "Inf base_lr should be rejected");
}

#[test]
fn test_warmup_zero_base_lr_is_valid() {
    let sched = WarmupSchedule::new(0.0, 100).unwrap();
    assert!(sched.lr_at_step(0).abs() < f64::EPSILON);
    assert!(sched.lr_at_step(50).abs() < f64::EPSILON);
    assert!(sched.lr_at_step(200).abs() < f64::EPSILON);
}

#[test]
fn test_warmup_single_step() {
    let sched = WarmupSchedule::new(0.1, 1).unwrap();
    assert!(sched.lr_at_step(0).abs() < f64::EPSILON, "step 0 -> 0");
    assert!(
        (sched.lr_at_step(1) - 0.1).abs() < f64::EPSILON,
        "step 1 -> base_lr"
    );
    assert!(
        (sched.lr_at_step(100) - 0.1).abs() < f64::EPSILON,
        "step 100 -> base_lr"
    );
}

#[test]
fn test_warmup_boundary_minus_one() {
    let sched = WarmupSchedule::new(1.0, 100).unwrap();
    let lr = sched.lr_at_step(99);
    assert!((lr - 0.99).abs() < 1e-10, "step 99/100 -> 0.99, got {lr}");
}

#[test]
fn test_warmup_monotonic_increase() {
    let sched = WarmupSchedule::new(0.01, 50).unwrap();
    let mut prev = 0.0;
    for step in 0..=60 {
        let lr = sched.lr_at_step(step);
        assert!(lr >= prev, "lr should be non-decreasing: step {step}");
        prev = lr;
    }
}

// == CosineSchedule ===========================================================

#[test]
fn test_cosine_negative_base_lr_rejected() {
    assert!(CosineSchedule::new(-0.01, 0.0, 0, 100).is_err());
}

#[test]
fn test_cosine_nan_base_lr_rejected() {
    assert!(CosineSchedule::new(f64::NAN, 0.0, 0, 100).is_err());
}

#[test]
fn test_cosine_negative_min_lr_rejected() {
    assert!(CosineSchedule::new(0.01, -0.001, 0, 100).is_err());
}

#[test]
fn test_cosine_nan_min_lr_rejected() {
    assert!(CosineSchedule::new(0.01, f64::NAN, 0, 100).is_err());
}

#[test]
fn test_cosine_equal_base_and_min_lr() {
    let sched = CosineSchedule::new(0.01, 0.01, 0, 100).unwrap();
    for step in [0, 25, 50, 75, 100, 200] {
        let lr = sched.lr_at_step(step);
        assert!(
            (lr - 0.01).abs() < 1e-12,
            "flat schedule: step {step} -> {lr}"
        );
    }
}

#[test]
fn test_cosine_quarter_point() {
    let sched = CosineSchedule::new(0.01, 0.0, 0, 1000).unwrap();
    let lr = sched.lr_at_step(250);
    let expected = 0.5 * 0.01 * (1.0 + (0.25 * std::f64::consts::PI).cos());
    assert!(
        (lr - expected).abs() < 1e-10,
        "quarter-point: expected {expected}, got {lr}"
    );
}

#[test]
fn test_cosine_three_quarter_point() {
    let sched = CosineSchedule::new(0.01, 0.0, 0, 1000).unwrap();
    let lr = sched.lr_at_step(750);
    let expected = 0.5 * 0.01 * (1.0 + (0.75 * std::f64::consts::PI).cos());
    assert!(
        (lr - expected).abs() < 1e-10,
        "three-quarter-point: expected {expected}, got {lr}"
    );
}

#[test]
fn test_cosine_monotonic_decay_no_warmup() {
    let sched = CosineSchedule::new(0.01, 0.0, 0, 1000).unwrap();
    let mut prev = sched.lr_at_step(0);
    for step in 1..=1000 {
        let lr = sched.lr_at_step(step);
        assert!(
            lr <= prev + 1e-15,
            "cosine should be non-increasing: step {step}, prev {prev}, cur {lr}"
        );
        prev = lr;
    }
}

#[test]
fn test_cosine_monotonic_with_warmup() {
    let sched = CosineSchedule::new(0.01, 0.0, 100, 1000).unwrap();
    let mut prev = 0.0;
    for step in 0..100 {
        let lr = sched.lr_at_step(step);
        assert!(lr >= prev - 1e-15, "warmup should increase: step {step}");
        prev = lr;
    }
    prev = sched.lr_at_step(100);
    for step in 101..=1000 {
        let lr = sched.lr_at_step(step);
        assert!(lr <= prev + 1e-15, "decay should decrease: step {step}");
        prev = lr;
    }
}

#[test]
fn test_cosine_min_total_steps() {
    let sched = CosineSchedule::new(1.0, 0.0, 1, 2).unwrap();
    assert!(
        sched.lr_at_step(0).abs() < f64::EPSILON,
        "warmup step 0 -> 0"
    );
    assert!(
        (sched.lr_at_step(1) - 1.0).abs() < 1e-10,
        "decay step 0 -> base_lr (cos(0)=1)"
    );
    assert!(sched.lr_at_step(2).abs() < 1e-10, "past total -> min_lr");
}

#[test]
fn test_cosine_large_step_count() {
    let sched = CosineSchedule::new(0.01, 1e-6, 0, 1_000_000).unwrap();
    assert!((sched.lr_at_step(0) - 0.01).abs() < 1e-10);
    assert!((sched.lr_at_step(1_000_000) - 1e-6).abs() < 1e-10);
    let mid = sched.lr_at_step(500_000);
    let expected = 1e-6 + 0.5 * (0.01 - 1e-6) * (1.0 + (0.5 * std::f64::consts::PI).cos());
    assert!((mid - expected).abs() < 1e-10);
}

// == step_with_schedule integration ===========================================

#[test]
fn test_step_with_schedule_at_end_of_cosine() {
    let x = scalar_var(5.0);
    let mut sgd = Sgd::new(
        vec![x.clone()],
        SgdConfig {
            lr: 0.0,
            ..SgdConfig::default()
        },
    )
    .unwrap();
    let schedule = CosineSchedule::new(0.1, 0.0, 0, 100).unwrap();

    let t = Arc::new(TrackedTensor::from_var(&x).unwrap());
    let loss = t.sqr().unwrap();
    let grads = backward(&loss).unwrap();
    step_with_schedule(&mut sgd, &grads, &schedule, 100).unwrap();

    let val = x.data().unwrap().to_flat_vec::<f32>().unwrap()[0];
    assert!(
        (val - 5.0).abs() < 1e-6,
        "at min_lr=0, param should not change, got {val}"
    );
    assert!(
        sgd.learning_rate().abs() < f64::EPSILON,
        "lr should be 0 past total_steps"
    );
}

#[test]
fn test_step_with_schedule_warmup_after_warmup() {
    let x = scalar_var(10.0);
    let mut sgd = Sgd::new(
        vec![x.clone()],
        SgdConfig {
            lr: 0.0,
            ..SgdConfig::default()
        },
    )
    .unwrap();
    let schedule = WarmupSchedule::new(0.1, 10).unwrap();

    let t = Arc::new(TrackedTensor::from_var(&x).unwrap());
    let loss = t.sqr().unwrap(); // grad = 2x = 20
    let grads = backward(&loss).unwrap();
    step_with_schedule(&mut sgd, &grads, &schedule, 20).unwrap();

    let val = x.data().unwrap().to_flat_vec::<f32>().unwrap()[0];
    assert!(
        (val - 8.0).abs() < 1e-4,
        "expected ~8.0 with full lr, got {val}"
    );
    assert!(
        (sgd.learning_rate() - 0.1).abs() < f64::EPSILON,
        "lr should be base_lr after warmup"
    );
}

#[test]
fn test_step_with_schedule_multiple_steps_decreasing_loss() {
    let x = scalar_var(5.0);
    let mut sgd = Sgd::new(
        vec![x.clone()],
        SgdConfig {
            lr: 0.0,
            ..SgdConfig::default()
        },
    )
    .unwrap();
    let schedule = CosineSchedule::new(0.01, 0.001, 0, 100).unwrap();

    let mut losses = Vec::new();
    for step in 0..20 {
        let t = Arc::new(TrackedTensor::from_var(&x).unwrap());
        let loss = t.sqr().unwrap();
        let loss_val = loss.tensor().to_flat_vec::<f32>().unwrap()[0];
        losses.push(loss_val);
        let grads = backward(&loss).unwrap();
        step_with_schedule(&mut sgd, &grads, &schedule, step).unwrap();
    }

    assert!(
        losses.last().unwrap() < losses.first().unwrap(),
        "loss should decrease: first={}, last={}",
        losses.first().unwrap(),
        losses.last().unwrap()
    );
}

// == GradScaler (public API only) =============================================

#[test]
fn test_grad_scaler_consecutive_backoffs() {
    let mut scaler = GradScaler::new(GradScalerConfig {
        init_scale: 1024.0,
        backoff_factor: 0.5,
        min_scale: 1.0,
        ..Default::default()
    })
    .unwrap();

    for i in 0..10 {
        scaler_inf_step(&mut scaler);
        scaler.update();
        let expected = (1024.0 * 0.5f64.powi(i + 1)).max(1.0);
        assert!(
            (scaler.scale_factor() - expected).abs() < f64::EPSILON,
            "after {} backoffs: expected {expected}, got {}",
            i + 1,
            scaler.scale_factor()
        );
    }
    assert!(
        (scaler.scale_factor() - 1.0).abs() < f64::EPSILON,
        "should be at min_scale"
    );
}

#[test]
fn test_grad_scaler_growth_interval_one() {
    let mut scaler = GradScaler::new(GradScalerConfig {
        init_scale: 1.0,
        growth_factor: 2.0,
        growth_interval: 1,
        min_scale: 1.0,
        max_scale: 1024.0,
        ..Default::default()
    })
    .unwrap();

    scaler_clean_step(&mut scaler);
    scaler.update();
    assert!(
        (scaler.scale_factor() - 2.0).abs() < f64::EPSILON,
        "growth_interval=1: should grow every step"
    );
    scaler_clean_step(&mut scaler);
    scaler.update();
    assert!((scaler.scale_factor() - 4.0).abs() < f64::EPSILON);
}

#[test]
fn test_grad_scaler_very_large_scale() {
    // Start at max_scale so that one growth (x2) would exceed it and must be
    // clamped back to max_scale. (With init=1e15 a single growth only reaches
    // 2e15, which is below max_scale=1e16 and so is *not* clamped — the old
    // expectation of 1e16 was simply wrong.)
    let config = GradScalerConfig {
        init_scale: 1e16,
        max_scale: 1e16,
        min_scale: 1.0,
        growth_factor: 2.0,
        growth_interval: 1,
        ..Default::default()
    };
    let mut scaler = GradScaler::new(config).unwrap();
    assert!((scaler.scale_factor() - 1e16).abs() < 1.0);

    scaler_clean_step(&mut scaler);
    scaler.update();
    assert!(
        (scaler.scale_factor() - 1e16).abs() < 1.0,
        "should clamp to max_scale, got {}",
        scaler.scale_factor()
    );
}

#[test]
fn test_grad_scaler_alternating_inf_clean() {
    let mut scaler = GradScaler::new(GradScalerConfig {
        init_scale: 1024.0,
        backoff_factor: 0.5,
        growth_factor: 2.0,
        growth_interval: 2,
        min_scale: 1.0,
        max_scale: 1e6,
    })
    .unwrap();

    let initial = scaler.scale_factor();
    for _ in 0..10 {
        scaler_inf_step(&mut scaler);
        scaler.update();
        scaler_clean_step(&mut scaler);
        scaler.update();
    }
    assert!(
        scaler.scale_factor() < initial,
        "alternating inf/clean should decrease scale: {} >= {}",
        scaler.scale_factor(),
        initial
    );
}

// == GradScalerConfig defaults ================================================

#[test]
fn test_grad_scaler_config_default_values() {
    let cfg = GradScalerConfig::default();
    assert!((cfg.init_scale - 65536.0).abs() < f64::EPSILON);
    assert!((cfg.growth_factor - 2.0).abs() < f64::EPSILON);
    assert!((cfg.backoff_factor - 0.5).abs() < f64::EPSILON);
    assert_eq!(cfg.growth_interval, 2000);
    assert!((cfg.min_scale - 1.0).abs() < f64::EPSILON);
    assert!((cfg.max_scale - 16_777_216.0).abs() < f64::EPSILON);
}

#[test]
fn test_grad_scaler_config_backoff_factor_zero_rejected() {
    let err = GradScaler::new(GradScalerConfig {
        backoff_factor: 0.0,
        ..Default::default()
    });
    assert!(err.is_err(), "backoff_factor=0 should be rejected");
}

#[test]
fn test_grad_scaler_config_backoff_factor_negative_rejected() {
    let err = GradScaler::new(GradScalerConfig {
        backoff_factor: -0.5,
        ..Default::default()
    });
    assert!(err.is_err(), "negative backoff_factor should be rejected");
}

#[test]
fn test_grad_scaler_config_growth_factor_exactly_one_rejected() {
    let err = GradScaler::new(GradScalerConfig {
        growth_factor: 1.0,
        ..Default::default()
    });
    assert!(err.is_err(), "growth_factor=1.0 should be rejected");
}

#[test]
fn test_grad_scaler_config_min_scale_zero_rejected() {
    let err = GradScaler::new(GradScalerConfig {
        min_scale: 0.0,
        ..Default::default()
    });
    assert!(err.is_err(), "min_scale=0 should be rejected");
}

// == clip_grad_norm additional tests ==========================================

#[test]
fn test_clip_grad_norm_single_element() {
    let (_var, mut grads) = grads_for(&[7.0]);
    let total_norm = clip_grad_norm(&mut grads, 3.0).unwrap();
    assert!(
        (total_norm - 7.0).abs() < 1e-5,
        "single element norm = abs(val) = 7.0, got {total_norm}"
    );
    for (_id, g) in grads.var_grads() {
        let vals = g.to_flat_vec::<f32>().unwrap();
        assert!(
            (vals[0] - 3.0).abs() < 1e-5,
            "clipped to max_norm=3.0, got {}",
            vals[0]
        );
    }
}

#[test]
fn test_clip_grad_norm_very_large_max_norm() {
    let (_var, mut grads) = grads_for(&[3.0, 4.0]);
    let total_norm = clip_grad_norm(&mut grads, 1e10).unwrap();
    assert!((total_norm - 5.0).abs() < 1e-5);
    for (_id, g) in grads.var_grads() {
        let vals = g.to_flat_vec::<f32>().unwrap();
        assert!((vals[0] - 3.0).abs() < 1e-5);
        assert!((vals[1] - 4.0).abs() < 1e-5);
    }
}

#[test]
fn test_clip_grad_norm_tiny_max_norm() {
    let (_var, mut grads) = grads_for(&[100.0, 200.0]);
    let total_norm = clip_grad_norm(&mut grads, 0.001).unwrap();
    let expected_norm = 100.0f64.hypot(200.0f64);
    assert!((total_norm - expected_norm).abs() < 1.0, "original norm");

    let mut clipped_norm_sq = 0.0f64;
    for (_id, g) in grads.var_grads() {
        let vals = g.to_flat_vec::<f32>().unwrap();
        for &v in &vals {
            clipped_norm_sq += f64::from(v) * f64::from(v);
        }
    }
    let clipped_norm = clipped_norm_sq.sqrt();
    assert!(
        (clipped_norm - 0.001).abs() < 1e-6,
        "clipped norm should be ~0.001, got {clipped_norm}"
    );
}

#[test]
fn test_clip_grad_norm_negative_gradients() {
    let (_var, mut grads) = grads_for(&[-3.0, -4.0]);
    let total_norm = clip_grad_norm(&mut grads, 2.5).unwrap();
    assert!(
        (total_norm - 5.0).abs() < 1e-5,
        "norm should be 5.0 for [-3,-4]"
    );
    for (_id, g) in grads.var_grads() {
        let vals = g.to_flat_vec::<f32>().unwrap();
        assert!((vals[0] - (-1.5)).abs() < 1e-5);
        assert!((vals[1] - (-2.0)).abs() < 1e-5);
    }
}

// == clip_grad_value additional tests =========================================

#[test]
fn test_clip_grad_value_all_zeros() {
    let (_var, mut grads) = grads_for(&[0.0, 0.0, 0.0]);
    clip_grad_value(&mut grads, 1.0).unwrap();
    for (_id, g) in grads.var_grads() {
        let vals = g.to_flat_vec::<f32>().unwrap();
        for &v in &vals {
            assert!(v.abs() < 1e-10, "zero gradient unchanged, got {v}");
        }
    }
}

#[test]
fn test_clip_grad_value_exact_boundary_values() {
    let (_var, mut grads) = grads_for(&[1.0, -1.0]);
    clip_grad_value(&mut grads, 1.0).unwrap();
    for (_id, g) in grads.var_grads() {
        let vals = g.to_flat_vec::<f32>().unwrap();
        assert!((vals[0] - 1.0).abs() < 1e-6, "at boundary: {}", vals[0]);
        assert!((vals[1] - (-1.0)).abs() < 1e-6, "at boundary: {}", vals[1]);
    }
}

#[test]
fn test_clip_grad_value_very_small_clip() {
    let (_var, mut grads) = grads_for(&[1.0, -1.0, 0.5]);
    clip_grad_value(&mut grads, 1e-6).unwrap();
    for (_id, g) in grads.var_grads() {
        let vals = g.to_flat_vec::<f32>().unwrap();
        for &v in &vals {
            assert!(
                v.abs() <= 1e-6 + 1e-10,
                "all values should be within [-1e-6, 1e-6], got {v}"
            );
        }
    }
}

// == Integration: schedule + scaler + clipping ================================

#[test]
fn test_full_training_loop_schedule_scaler_clip() {
    let w = Var::new(DynTensor::from_vec(vec![3.0f32, -4.0], &[2], &cpu()).unwrap());
    let mut sgd = Sgd::new(
        vec![w.clone()],
        SgdConfig {
            lr: 0.0,
            ..SgdConfig::default()
        },
    )
    .unwrap();
    let schedule = CosineSchedule::new(0.01, 0.001, 5, 50).unwrap();
    let mut scaler = GradScaler::new(GradScalerConfig {
        init_scale: 256.0,
        growth_interval: 100,
        ..Default::default()
    })
    .unwrap();

    let mut losses = Vec::new();
    for step in 0..30 {
        let tw = Arc::new(TrackedTensor::from_var(&w).unwrap());
        let loss = tw.sqr().unwrap().sum_keepdim(0).unwrap();
        losses.push(loss.tensor().to_flat_vec::<f32>().unwrap()[0]);

        let scaled = scaler.scale_loss(&loss).unwrap();
        let mut grads = backward(&scaled).unwrap();
        if scaler.unscale_and_check(&mut grads).unwrap() {
            clip_grad_norm(&mut grads, 1.0).unwrap();
            sgd.set_learning_rate(schedule.lr_at_step(step)).unwrap();
            sgd.step(&grads).unwrap();
        }
        scaler.update();
    }

    assert!(
        losses.last().unwrap() < losses.first().unwrap(),
        "loss should decrease with full pipeline: first={}, last={}",
        losses.first().unwrap(),
        losses.last().unwrap()
    );
}

#[test]
fn test_grad_scaler_with_value_clipping() {
    let w = Var::new(DynTensor::from_vec(vec![100.0f32], &[1], &cpu()).unwrap());
    let mut scaler = GradScaler::new(GradScalerConfig {
        init_scale: 1024.0,
        growth_interval: 100,
        ..Default::default()
    })
    .unwrap();

    let tw = Arc::new(TrackedTensor::from_var(&w).unwrap());
    let loss = tw.sqr().unwrap();
    let scaled = scaler.scale_loss(&loss).unwrap();
    let mut grads = backward(&scaled).unwrap();

    let ok = scaler.unscale_and_check(&mut grads).unwrap();
    assert!(ok, "gradients should be finite");

    clip_grad_value(&mut grads, 1.0).unwrap();

    let grad = grads.get(&w).unwrap().to_flat_vec::<f32>().unwrap();
    assert!(
        (grad[0] - 1.0).abs() < 1e-6,
        "gradient should be clipped to 1.0, got {}",
        grad[0]
    );
}

// == Edge cases ===============================================================

#[test]
fn test_warmup_schedule_very_large_step() {
    let sched = WarmupSchedule::new(0.01, 100).unwrap();
    let lr = sched.lr_at_step(usize::MAX);
    assert!(
        (lr - 0.01).abs() < f64::EPSILON,
        "very large step should return base_lr, got {lr}"
    );
}

#[test]
fn test_cosine_schedule_very_large_step() {
    let sched = CosineSchedule::new(0.01, 0.001, 0, 1000).unwrap();
    let lr = sched.lr_at_step(usize::MAX);
    assert!(
        (lr - 0.001).abs() < f64::EPSILON,
        "very large step should clamp to min_lr, got {lr}"
    );
}

#[test]
fn test_grad_scaler_no_vars_in_gradstore() {
    let mut scaler = GradScaler::new(GradScalerConfig::default()).unwrap();

    let t = Arc::new(TrackedTensor::from_tensor(
        DynTensor::from_vec(vec![1.0f32], &[1], &cpu()).unwrap(),
    ));
    let loss = t.mul_scalar(1.0).unwrap();
    let mut grads = backward(&loss).unwrap();

    let ok = scaler.unscale_and_check(&mut grads).unwrap();
    assert!(ok, "empty grads -> no inf -> ok=true");
    assert!(!scaler.found_inf());
}

#[test]
fn test_clip_grad_norm_preserves_zero_gradient() {
    let (_var, mut grads) = grads_for(&[0.0, 0.0]);
    let total_norm = clip_grad_norm(&mut grads, 1.0).unwrap();
    assert!(total_norm.abs() < 1e-10);
    for (_id, g) in grads.var_grads() {
        let vals = g.to_flat_vec::<f32>().unwrap();
        assert!(vals[0].abs() < 1e-10);
        assert!(vals[1].abs() < 1e-10);
    }
}

#[test]
fn test_warmup_schedule_clone_and_eq() {
    let sched = WarmupSchedule::new(0.01, 100).unwrap();
    let cloned = sched.clone();
    assert_eq!(sched, cloned);
    assert_eq!(sched.base_lr(), cloned.base_lr());
    assert_eq!(sched.warmup_steps(), cloned.warmup_steps());
}

#[test]
fn test_cosine_schedule_clone_and_eq() {
    let sched = CosineSchedule::new(0.01, 0.001, 50, 1000).unwrap();
    let cloned = sched.clone();
    assert_eq!(sched, cloned);
}

#[test]
fn test_grad_scaler_config_clone_and_eq() {
    let cfg1 = GradScalerConfig::default();
    let cfg2 = cfg1.clone();
    assert_eq!(cfg1, cfg2);
}

#[test]
fn test_grad_scaler_debug_format() {
    let scaler = GradScaler::new(GradScalerConfig::default()).unwrap();
    let debug = format!("{scaler:?}");
    assert!(debug.contains("GradScaler"), "debug format: {debug}");
    assert!(debug.contains("65536"), "debug should show scale: {debug}");
}

#[test]
fn test_grad_scaler_scale_factor_after_clean_update() {
    let mut scaler = GradScaler::new(GradScalerConfig {
        init_scale: 100.0,
        growth_factor: 3.0,
        growth_interval: 5,
        min_scale: 1.0,
        max_scale: 10000.0,
        ..Default::default()
    })
    .unwrap();

    for _ in 0..4 {
        scaler_clean_step(&mut scaler);
        scaler.update();
    }
    assert!(
        (scaler.scale_factor() - 100.0).abs() < f64::EPSILON,
        "4 steps < interval of 5: no growth yet, got {}",
        scaler.scale_factor()
    );

    scaler_clean_step(&mut scaler);
    scaler.update();
    assert!(
        (scaler.scale_factor() - 300.0).abs() < f64::EPSILON,
        "5th step = interval: should grow to 300, got {}",
        scaler.scale_factor()
    );
}

#[test]
fn test_grad_scaler_inf_then_recovery() {
    let mut scaler = GradScaler::new(GradScalerConfig {
        init_scale: 64.0,
        backoff_factor: 0.5,
        growth_factor: 2.0,
        growth_interval: 2,
        min_scale: 1.0,
        max_scale: 1000.0,
    })
    .unwrap();

    scaler_inf_step(&mut scaler);
    scaler.update();
    assert!(
        (scaler.scale_factor() - 32.0).abs() < f64::EPSILON,
        "backoff: 64 * 0.5 = 32"
    );

    scaler_clean_step(&mut scaler);
    scaler.update();
    scaler_clean_step(&mut scaler);
    scaler.update();
    assert!(
        (scaler.scale_factor() - 64.0).abs() < f64::EPSILON,
        "recovery: 32 * 2 = 64, got {}",
        scaler.scale_factor()
    );
}

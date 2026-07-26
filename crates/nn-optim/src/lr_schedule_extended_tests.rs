// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Part of #4186.
//!
//! Extended tests for learning rate schedules, gradient clipping, and
//! gradient scaler. Covers construction validation, boundary behavior,
//! curve shape verification, edge cases, and integration scenarios.

use std::sync::Arc;

use nn_autodiff::{backward, GradStore, TrackedTensor, Var};
use nn_core::dyn_tensor::DynTensor;
use nn_core::Device;

use crate::grad_clip::{clip_grad_norm, clip_grad_value};
use crate::grad_scaler::{GradScaler, GradScalerConfig};
use crate::lr_schedule::{step_with_schedule, CosineSchedule, LrSchedule, WarmupSchedule};
use crate::optimizer::Optimizer;
use crate::sgd::{Sgd, SgdConfig};
use crate::GradScalerState;

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
    let inf_grad = DynTensor::from_vec(vec![f32::INFINITY], &[1], &cpu()).unwrap();
    for (_, grad) in grads.var_grads_mut() {
        *grad = inf_grad.clone();
    }
    let ok = scaler.unscale_and_check(&mut grads).unwrap();
    assert!(!ok, "inf step should return ok=false");
}

// =============================================================================
// WarmupSchedule — construction
// =============================================================================

#[test]
fn test_warmup_construction_valid_params_succeeds() {
    let sched = WarmupSchedule::new(0.01, 100);
    assert!(sched.is_ok());
    let sched = sched.unwrap();
    assert!((sched.base_lr() - 0.01).abs() < f64::EPSILON);
    assert_eq!(sched.warmup_steps(), 100);
}

#[test]
fn test_warmup_construction_zero_lr_succeeds() {
    let sched = WarmupSchedule::new(0.0, 50).unwrap();
    assert!(sched.base_lr().abs() < f64::EPSILON);
}

#[test]
fn test_warmup_construction_negative_lr_rejected() {
    let err = WarmupSchedule::new(-1e-5, 100);
    assert!(err.is_err());
    let msg = format!("{}", err.unwrap_err());
    assert!(msg.contains("base_lr"), "error should cite base_lr: {msg}");
}

#[test]
fn test_warmup_construction_nan_lr_rejected() {
    assert!(WarmupSchedule::new(f64::NAN, 10).is_err());
}

#[test]
fn test_warmup_construction_inf_lr_rejected() {
    assert!(WarmupSchedule::new(f64::INFINITY, 10).is_err());
}

#[test]
fn test_warmup_construction_neg_inf_lr_rejected() {
    assert!(WarmupSchedule::new(f64::NEG_INFINITY, 10).is_err());
}

// =============================================================================
// WarmupSchedule — lr_at_step
// =============================================================================

#[test]
fn test_warmup_lr_at_step_zero_returns_zero() {
    let sched = WarmupSchedule::new(0.1, 100).unwrap();
    assert!(sched.lr_at_step(0).abs() < f64::EPSILON);
}

#[test]
fn test_warmup_lr_midway_is_half_base() {
    let sched = WarmupSchedule::new(0.1, 100).unwrap();
    let lr = sched.lr_at_step(50);
    assert!((lr - 0.05).abs() < 1e-10, "expected 0.05, got {lr}");
}

#[test]
fn test_warmup_lr_at_boundary_equals_base_lr() {
    let sched = WarmupSchedule::new(0.1, 100).unwrap();
    let lr = sched.lr_at_step(100);
    assert!((lr - 0.1).abs() < f64::EPSILON, "expected 0.1, got {lr}");
}

#[test]
fn test_warmup_lr_post_warmup_stays_constant() {
    let sched = WarmupSchedule::new(0.1, 100).unwrap();
    for step in [101, 200, 500, 10000] {
        let lr = sched.lr_at_step(step);
        assert!(
            (lr - 0.1).abs() < f64::EPSILON,
            "step {step}: expected 0.1, got {lr}"
        );
    }
}

#[test]
fn test_warmup_lr_one_step_before_boundary() {
    let sched = WarmupSchedule::new(1.0, 1000).unwrap();
    let lr = sched.lr_at_step(999);
    assert!(
        (lr - 0.999).abs() < 1e-10,
        "step 999/1000: expected 0.999, got {lr}"
    );
}

#[test]
fn test_warmup_zero_warmup_steps_returns_base_lr_always() {
    let sched = WarmupSchedule::new(0.05, 0).unwrap();
    for step in [0, 1, 100, usize::MAX] {
        let lr = sched.lr_at_step(step);
        assert!(
            (lr - 0.05).abs() < f64::EPSILON,
            "zero warmup: step {step}, expected 0.05, got {lr}"
        );
    }
}

#[test]
fn test_warmup_monotonic_non_decreasing() {
    let sched = WarmupSchedule::new(0.01, 200).unwrap();
    let mut prev = 0.0;
    for step in 0..300 {
        let lr = sched.lr_at_step(step);
        assert!(
            lr >= prev - 1e-15,
            "non-decreasing violated at step {step}: prev={prev}, cur={lr}"
        );
        prev = lr;
    }
}

#[test]
fn test_warmup_large_step_number_returns_base_lr() {
    let sched = WarmupSchedule::new(0.001, 10).unwrap();
    let lr = sched.lr_at_step(usize::MAX);
    assert!(
        (lr - 0.001).abs() < f64::EPSILON,
        "very large step: expected 0.001, got {lr}"
    );
}

// =============================================================================
// CosineSchedule — construction
// =============================================================================

#[test]
fn test_cosine_construction_valid_params_succeeds() {
    let sched = CosineSchedule::new(0.01, 0.001, 100, 1000);
    assert!(sched.is_ok());
    let sched = sched.unwrap();
    assert!((sched.base_lr() - 0.01).abs() < f64::EPSILON);
    assert!((sched.min_lr() - 0.001).abs() < f64::EPSILON);
    assert_eq!(sched.total_steps(), 1000);
}

#[test]
fn test_cosine_construction_no_warmup_succeeds() {
    assert!(CosineSchedule::new(0.01, 0.0, 0, 500).is_ok());
}

#[test]
fn test_cosine_construction_negative_base_lr_rejected() {
    assert!(CosineSchedule::new(-0.01, 0.0, 0, 100).is_err());
}

#[test]
fn test_cosine_construction_negative_min_lr_rejected() {
    assert!(CosineSchedule::new(0.01, -0.001, 0, 100).is_err());
}

#[test]
fn test_cosine_construction_nan_min_lr_rejected() {
    assert!(CosineSchedule::new(0.01, f64::NAN, 0, 100).is_err());
}

#[test]
fn test_cosine_construction_zero_total_steps_rejected() {
    assert!(CosineSchedule::new(0.01, 0.0, 0, 0).is_err());
}

#[test]
fn test_cosine_construction_warmup_ge_total_rejected() {
    assert!(CosineSchedule::new(0.01, 0.0, 100, 100).is_err());
    assert!(CosineSchedule::new(0.01, 0.0, 200, 100).is_err());
}

#[test]
fn test_cosine_construction_min_lr_gt_base_lr_rejected() {
    let err = CosineSchedule::new(0.001, 0.01, 0, 100);
    assert!(err.is_err());
    let msg = format!("{}", err.unwrap_err());
    assert!(msg.contains("min_lr"), "error should cite min_lr: {msg}");
}

#[test]
fn test_cosine_construction_equal_base_and_min_lr_succeeds() {
    assert!(CosineSchedule::new(0.01, 0.01, 0, 100).is_ok());
}

// =============================================================================
// CosineSchedule — lr_at_step curve shape
// =============================================================================

#[test]
fn test_cosine_lr_at_start_equals_base_lr() {
    let sched = CosineSchedule::new(0.01, 0.0, 0, 1000).unwrap();
    let lr = sched.lr_at_step(0);
    assert!((lr - 0.01).abs() < 1e-10, "start: expected 0.01, got {lr}");
}

#[test]
fn test_cosine_lr_at_midpoint_is_average() {
    let sched = CosineSchedule::new(0.01, 0.0, 0, 1000).unwrap();
    let lr = sched.lr_at_step(500);
    // cos(pi * 0.5) = 0, so lr = 0 + 0.5 * 0.01 * (1 + 0) = 0.005
    assert!((lr - 0.005).abs() < 1e-10, "mid: expected 0.005, got {lr}");
}

#[test]
fn test_cosine_lr_at_end_equals_min_lr() {
    let sched = CosineSchedule::new(0.01, 0.001, 0, 1000).unwrap();
    let lr = sched.lr_at_step(1000);
    assert!((lr - 0.001).abs() < 1e-10, "end: expected 0.001, got {lr}");
}

#[test]
fn test_cosine_lr_past_total_steps_clamps_to_min_lr() {
    let sched = CosineSchedule::new(0.01, 0.001, 0, 100).unwrap();
    for step in [101, 200, 1000, usize::MAX] {
        let lr = sched.lr_at_step(step);
        assert!(
            (lr - 0.001).abs() < f64::EPSILON,
            "step {step}: expected min_lr 0.001, got {lr}"
        );
    }
}

#[test]
fn test_cosine_lr_quarter_point_matches_formula() {
    let sched = CosineSchedule::new(0.1, 0.0, 0, 400).unwrap();
    let lr = sched.lr_at_step(100); // progress = 0.25
    let expected = 0.5 * 0.1 * (1.0 + (0.25 * std::f64::consts::PI).cos());
    assert!(
        (lr - expected).abs() < 1e-10,
        "quarter: expected {expected}, got {lr}"
    );
}

#[test]
fn test_cosine_lr_three_quarter_point_matches_formula() {
    let sched = CosineSchedule::new(0.1, 0.0, 0, 400).unwrap();
    let lr = sched.lr_at_step(300); // progress = 0.75
    let expected = 0.5 * 0.1 * (1.0 + (0.75 * std::f64::consts::PI).cos());
    assert!(
        (lr - expected).abs() < 1e-10,
        "three-quarter: expected {expected}, got {lr}"
    );
}

#[test]
fn test_cosine_monotonic_decay_no_warmup() {
    let sched = CosineSchedule::new(0.1, 0.0, 0, 500).unwrap();
    let mut prev = sched.lr_at_step(0);
    for step in 1..=500 {
        let lr = sched.lr_at_step(step);
        assert!(
            lr <= prev + 1e-15,
            "non-increasing violated at step {step}: prev={prev}, cur={lr}"
        );
        prev = lr;
    }
}

#[test]
fn test_cosine_monotonic_warmup_then_decay() {
    let sched = CosineSchedule::new(0.1, 0.0, 50, 500).unwrap();
    // Warmup phase: non-decreasing
    let mut prev = 0.0;
    for step in 0..50 {
        let lr = sched.lr_at_step(step);
        assert!(lr >= prev - 1e-15, "warmup increase at step {step}");
        prev = lr;
    }
    // Decay phase: non-increasing
    prev = sched.lr_at_step(50);
    for step in 51..=500 {
        let lr = sched.lr_at_step(step);
        assert!(lr <= prev + 1e-15, "decay decrease at step {step}");
        prev = lr;
    }
}

#[test]
fn test_cosine_equal_base_and_min_lr_flat_schedule() {
    let sched = CosineSchedule::new(0.05, 0.05, 0, 100).unwrap();
    for step in [0, 25, 50, 75, 100, 200] {
        let lr = sched.lr_at_step(step);
        assert!(
            (lr - 0.05).abs() < 1e-12,
            "flat schedule: step {step}, got {lr}"
        );
    }
}

#[test]
fn test_cosine_with_warmup_warmup_phase_is_linear() {
    let sched = CosineSchedule::new(0.1, 0.0, 100, 1000).unwrap();
    // Warmup phase should be linear from 0 to base_lr
    for step in [0, 10, 25, 50, 75, 99] {
        let lr = sched.lr_at_step(step);
        let expected = 0.1 * (step as f64 / 100.0);
        assert!(
            (lr - expected).abs() < 1e-10,
            "warmup linear at step {step}: expected {expected}, got {lr}"
        );
    }
}

#[test]
fn test_cosine_minimal_two_step_schedule() {
    // warmup=0, total=1 => single decay step
    let sched = CosineSchedule::new(1.0, 0.0, 0, 1).unwrap();
    let lr0 = sched.lr_at_step(0);
    // progress = 0/1 = 0, lr = 0 + 0.5 * 1.0 * (1 + cos(0)) = 1.0
    assert!((lr0 - 1.0).abs() < 1e-10, "step 0: expected 1.0, got {lr0}");
    let lr1 = sched.lr_at_step(1);
    assert!((lr1 - 0.0).abs() < 1e-10, "step 1: clamp to min_lr");
}

// =============================================================================
// LrSchedule trait — both implementations satisfy the trait
// =============================================================================

fn schedule_at_step(sched: &dyn LrSchedule, step: usize) -> f64 {
    sched.lr_at_step(step)
}

#[test]
fn test_lr_schedule_trait_warmup_dynamic_dispatch() {
    let sched = WarmupSchedule::new(0.1, 10).unwrap();
    let lr = schedule_at_step(&sched, 5);
    assert!((lr - 0.05).abs() < 1e-10);
}

#[test]
fn test_lr_schedule_trait_cosine_dynamic_dispatch() {
    let sched = CosineSchedule::new(0.1, 0.0, 0, 100).unwrap();
    let lr = schedule_at_step(&sched, 50);
    assert!((lr - 0.05).abs() < 1e-10);
}

// =============================================================================
// step_with_schedule — integration with optimizer
// =============================================================================

#[test]
fn test_step_with_schedule_warmup_updates_lr_and_param() {
    let x = scalar_var(5.0);
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
    let loss = t.sqr().unwrap(); // grad = 2x = 10
    let grads = backward(&loss).unwrap();
    // step 5: lr = 0.1 * 5/10 = 0.05
    step_with_schedule(&mut sgd, &grads, &schedule, 5).unwrap();

    let val = x.data().unwrap().to_flat_vec::<f32>().unwrap()[0];
    // x = 5.0 - 0.05 * 10 = 4.5
    assert!((val - 4.5).abs() < 1e-4, "expected ~4.5, got {val}");
    assert!(
        (sgd.learning_rate() - 0.05).abs() < f64::EPSILON,
        "lr should be 0.05"
    );
}

#[test]
fn test_step_with_schedule_cosine_at_midpoint() {
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
    let loss = t.sqr().unwrap(); // grad = 2x = 10
    let grads = backward(&loss).unwrap();
    // step 50: lr = 0.05
    step_with_schedule(&mut sgd, &grads, &schedule, 50).unwrap();

    let val = x.data().unwrap().to_flat_vec::<f32>().unwrap()[0];
    assert!((val - 4.5).abs() < 1e-4, "expected ~4.5, got {val}");
}

#[test]
fn test_step_with_schedule_at_zero_lr_no_param_change() {
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
    // step 100: at total_steps, lr = min_lr = 0
    step_with_schedule(&mut sgd, &grads, &schedule, 100).unwrap();

    let val = x.data().unwrap().to_flat_vec::<f32>().unwrap()[0];
    assert!(
        (val - 5.0).abs() < 1e-6,
        "lr=0 should not change param, got {val}"
    );
}

#[test]
fn test_step_with_schedule_multi_step_loss_decreases() {
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
    for step in 0..15 {
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

// =============================================================================
// clip_grad_norm
// =============================================================================

#[test]
fn test_clip_grad_norm_below_threshold_unchanged() {
    let (_var, mut grads) = grads_for(&[3.0, 4.0]); // norm = 5.0
    let total_norm = clip_grad_norm(&mut grads, 10.0).unwrap();
    assert!((total_norm - 5.0).abs() < 1e-5, "original norm = 5.0");
    for (_id, g) in grads.var_grads() {
        let vals = g.to_flat_vec::<f32>().unwrap();
        assert!((vals[0] - 3.0).abs() < 1e-5, "unchanged: {}", vals[0]);
        assert!((vals[1] - 4.0).abs() < 1e-5, "unchanged: {}", vals[1]);
    }
}

#[test]
fn test_clip_grad_norm_above_threshold_scaled_down() {
    let (_var, mut grads) = grads_for(&[3.0, 4.0]); // norm = 5.0
    let total_norm = clip_grad_norm(&mut grads, 2.5).unwrap();
    assert!((total_norm - 5.0).abs() < 1e-5);
    // Scaled by 2.5/5.0 = 0.5
    for (_id, g) in grads.var_grads() {
        let vals = g.to_flat_vec::<f32>().unwrap();
        assert!((vals[0] - 1.5).abs() < 1e-5, "scaled: {}", vals[0]);
        assert!((vals[1] - 2.0).abs() < 1e-5, "scaled: {}", vals[1]);
    }
}

#[test]
fn test_clip_grad_norm_exact_threshold_unchanged() {
    let (_var, mut grads) = grads_for(&[3.0, 4.0]); // norm = 5.0
    let total_norm = clip_grad_norm(&mut grads, 5.0).unwrap();
    assert!((total_norm - 5.0).abs() < 1e-5);
    for (_id, g) in grads.var_grads() {
        let vals = g.to_flat_vec::<f32>().unwrap();
        assert!((vals[0] - 3.0).abs() < 1e-5);
        assert!((vals[1] - 4.0).abs() < 1e-5);
    }
}

#[test]
fn test_clip_grad_norm_preserves_direction() {
    let (_var, mut grads) = grads_for(&[6.0, -8.0]); // norm = 10.0
    clip_grad_norm(&mut grads, 5.0).unwrap();
    // Scaled by 5/10 = 0.5
    for (_id, g) in grads.var_grads() {
        let vals = g.to_flat_vec::<f32>().unwrap();
        assert!(
            (vals[0] - 3.0).abs() < 1e-5,
            "direction preserved: {}",
            vals[0]
        );
        assert!(
            (vals[1] - (-4.0)).abs() < 1e-5,
            "direction preserved: {}",
            vals[1]
        );
    }
}

#[test]
fn test_clip_grad_norm_zero_gradient_unchanged() {
    let (_var, mut grads) = grads_for(&[0.0, 0.0]);
    let total_norm = clip_grad_norm(&mut grads, 1.0).unwrap();
    assert!(total_norm.abs() < 1e-10, "zero gradient norm = 0");
    for (_id, g) in grads.var_grads() {
        let vals = g.to_flat_vec::<f32>().unwrap();
        assert!(vals[0].abs() < 1e-10);
        assert!(vals[1].abs() < 1e-10);
    }
}

#[test]
fn test_clip_grad_norm_very_small_max_norm() {
    let (_var, mut grads) = grads_for(&[10.0, 10.0]); // norm ~= 14.14
    clip_grad_norm(&mut grads, 1e-4).unwrap();
    let mut clipped_norm_sq = 0.0f64;
    for (_id, g) in grads.var_grads() {
        for &v in &g.to_flat_vec::<f32>().unwrap() {
            clipped_norm_sq += f64::from(v) * f64::from(v);
        }
    }
    let clipped_norm = clipped_norm_sq.sqrt();
    assert!(
        (clipped_norm - 1e-4).abs() < 1e-8,
        "clipped norm should be ~1e-4, got {clipped_norm}"
    );
}

#[test]
fn test_clip_grad_norm_very_large_max_norm() {
    let (_var, mut grads) = grads_for(&[1.0, 2.0]); // norm ~= 2.24
    let total_norm = clip_grad_norm(&mut grads, 1e10).unwrap();
    assert!((total_norm - (1.0f64 + 4.0).sqrt()).abs() < 1e-4);
    for (_id, g) in grads.var_grads() {
        let vals = g.to_flat_vec::<f32>().unwrap();
        assert!((vals[0] - 1.0).abs() < 1e-5, "unchanged");
        assert!((vals[1] - 2.0).abs() < 1e-5, "unchanged");
    }
}

#[test]
fn test_clip_grad_norm_invalid_max_norm_rejected() {
    let (_var, mut grads) = grads_for(&[1.0]);
    assert!(clip_grad_norm(&mut grads, 0.0).is_err());
    assert!(clip_grad_norm(&mut grads, -1.0).is_err());
    assert!(clip_grad_norm(&mut grads, f64::NAN).is_err());
    assert!(clip_grad_norm(&mut grads, f64::INFINITY).is_err());
}

// =============================================================================
// clip_grad_value
// =============================================================================

#[test]
fn test_clip_grad_value_clamps_to_range() {
    let (_var, mut grads) = grads_for(&[5.0, -5.0, 0.5]);
    clip_grad_value(&mut grads, 1.0).unwrap();
    for (_id, g) in grads.var_grads() {
        let vals = g.to_flat_vec::<f32>().unwrap();
        assert!((vals[0] - 1.0).abs() < 1e-6, "clamped high: {}", vals[0]);
        assert!((vals[1] - (-1.0)).abs() < 1e-6, "clamped low: {}", vals[1]);
        assert!((vals[2] - 0.5).abs() < 1e-6, "within range: {}", vals[2]);
    }
}

#[test]
fn test_clip_grad_value_within_range_unchanged() {
    let (_var, mut grads) = grads_for(&[0.1, -0.1, 0.0]);
    clip_grad_value(&mut grads, 1.0).unwrap();
    for (_id, g) in grads.var_grads() {
        let vals = g.to_flat_vec::<f32>().unwrap();
        assert!((vals[0] - 0.1).abs() < 1e-6);
        assert!((vals[1] - (-0.1)).abs() < 1e-6);
        assert!(vals[2].abs() < 1e-6);
    }
}

#[test]
fn test_clip_grad_value_exact_boundary_not_clipped() {
    let (_var, mut grads) = grads_for(&[2.0, -2.0]);
    clip_grad_value(&mut grads, 2.0).unwrap();
    for (_id, g) in grads.var_grads() {
        let vals = g.to_flat_vec::<f32>().unwrap();
        assert!((vals[0] - 2.0).abs() < 1e-6, "at boundary: {}", vals[0]);
        assert!((vals[1] - (-2.0)).abs() < 1e-6, "at boundary: {}", vals[1]);
    }
}

#[test]
fn test_clip_grad_value_very_small_clip() {
    let (_var, mut grads) = grads_for(&[100.0, -100.0]);
    clip_grad_value(&mut grads, 1e-6).unwrap();
    for (_id, g) in grads.var_grads() {
        let vals = g.to_flat_vec::<f32>().unwrap();
        for &v in &vals {
            assert!(v.abs() <= 1e-6 + 1e-10, "all within [-1e-6, 1e-6], got {v}");
        }
    }
}

#[test]
fn test_clip_grad_value_very_large_clip() {
    let (_var, mut grads) = grads_for(&[5.0, -5.0]);
    clip_grad_value(&mut grads, 1e10).unwrap();
    for (_id, g) in grads.var_grads() {
        let vals = g.to_flat_vec::<f32>().unwrap();
        assert!((vals[0] - 5.0).abs() < 1e-5, "unchanged: {}", vals[0]);
        assert!((vals[1] - (-5.0)).abs() < 1e-5, "unchanged: {}", vals[1]);
    }
}

#[test]
fn test_clip_grad_value_all_zeros_unchanged() {
    let (_var, mut grads) = grads_for(&[0.0, 0.0, 0.0]);
    clip_grad_value(&mut grads, 0.5).unwrap();
    for (_id, g) in grads.var_grads() {
        for &v in &g.to_flat_vec::<f32>().unwrap() {
            assert!(v.abs() < 1e-10, "zero unchanged, got {v}");
        }
    }
}

#[test]
fn test_clip_grad_value_invalid_clip_value_rejected() {
    let (_var, mut grads) = grads_for(&[1.0]);
    assert!(clip_grad_value(&mut grads, 0.0).is_err());
    assert!(clip_grad_value(&mut grads, -0.5).is_err());
    assert!(clip_grad_value(&mut grads, f64::NAN).is_err());
    assert!(clip_grad_value(&mut grads, f64::INFINITY).is_err());
}

// =============================================================================
// GradScaler — construction
// =============================================================================

#[test]
fn test_grad_scaler_default_config_construction() {
    let scaler = GradScaler::new(GradScalerConfig::default()).unwrap();
    assert!((scaler.scale_factor() - 65536.0).abs() < f64::EPSILON);
    assert!(!scaler.found_inf());
}

#[test]
fn test_grad_scaler_custom_config_construction() {
    let config = GradScalerConfig {
        init_scale: 1024.0,
        growth_factor: 3.0,
        backoff_factor: 0.25,
        growth_interval: 500,
        min_scale: 0.5,
        max_scale: 1e10,
    };
    let scaler = GradScaler::new(config).unwrap();
    assert!((scaler.scale_factor() - 1024.0).abs() < f64::EPSILON);
}

#[test]
fn test_grad_scaler_config_default_field_values() {
    let cfg = GradScalerConfig::default();
    assert!((cfg.init_scale - 65536.0).abs() < f64::EPSILON);
    assert!((cfg.growth_factor - 2.0).abs() < f64::EPSILON);
    assert!((cfg.backoff_factor - 0.5).abs() < f64::EPSILON);
    assert_eq!(cfg.growth_interval, 2000);
    assert!((cfg.min_scale - 1.0).abs() < f64::EPSILON);
    assert!((cfg.max_scale - 16_777_216.0).abs() < f64::EPSILON);
}

// =============================================================================
// GradScaler — config validation
// =============================================================================

#[test]
fn test_grad_scaler_zero_init_scale_rejected() {
    assert!(GradScaler::new(GradScalerConfig {
        init_scale: 0.0,
        ..Default::default()
    })
    .is_err());
}

#[test]
fn test_grad_scaler_negative_init_scale_rejected() {
    assert!(GradScaler::new(GradScalerConfig {
        init_scale: -1.0,
        ..Default::default()
    })
    .is_err());
}

#[test]
fn test_grad_scaler_growth_factor_one_rejected() {
    assert!(GradScaler::new(GradScalerConfig {
        growth_factor: 1.0,
        ..Default::default()
    })
    .is_err());
}

#[test]
fn test_grad_scaler_growth_factor_below_one_rejected() {
    assert!(GradScaler::new(GradScalerConfig {
        growth_factor: 0.5,
        ..Default::default()
    })
    .is_err());
}

#[test]
fn test_grad_scaler_backoff_factor_zero_rejected() {
    assert!(GradScaler::new(GradScalerConfig {
        backoff_factor: 0.0,
        ..Default::default()
    })
    .is_err());
}

#[test]
fn test_grad_scaler_backoff_factor_one_rejected() {
    assert!(GradScaler::new(GradScalerConfig {
        backoff_factor: 1.0,
        ..Default::default()
    })
    .is_err());
}

#[test]
fn test_grad_scaler_backoff_factor_negative_rejected() {
    assert!(GradScaler::new(GradScalerConfig {
        backoff_factor: -0.5,
        ..Default::default()
    })
    .is_err());
}

#[test]
fn test_grad_scaler_min_scale_zero_rejected() {
    assert!(GradScaler::new(GradScalerConfig {
        min_scale: 0.0,
        ..Default::default()
    })
    .is_err());
}

#[test]
fn test_grad_scaler_max_scale_below_min_rejected() {
    assert!(GradScaler::new(GradScalerConfig {
        min_scale: 100.0,
        max_scale: 50.0,
        init_scale: 75.0,
        ..Default::default()
    })
    .is_err());
}

#[test]
fn test_grad_scaler_growth_interval_zero_rejected() {
    assert!(GradScaler::new(GradScalerConfig {
        growth_interval: 0,
        ..Default::default()
    })
    .is_err());
}

#[test]
fn test_grad_scaler_init_scale_outside_bounds_rejected() {
    assert!(GradScaler::new(GradScalerConfig {
        init_scale: 0.5,
        min_scale: 1.0,
        max_scale: 1000.0,
        ..Default::default()
    })
    .is_err());
    assert!(GradScaler::new(GradScalerConfig {
        init_scale: 2000.0,
        min_scale: 1.0,
        max_scale: 1000.0,
        ..Default::default()
    })
    .is_err());
}

// =============================================================================
// GradScaler — step behavior
// =============================================================================

#[test]
fn test_grad_scaler_scale_loss_multiplies_by_factor() {
    let scaler = GradScaler::new(GradScalerConfig {
        init_scale: 256.0,
        ..Default::default()
    })
    .unwrap();
    let var = Var::new(DynTensor::from_vec(vec![2.0f32], &[1], &cpu()).unwrap());
    let t = Arc::new(TrackedTensor::from_var(&var).unwrap());
    let loss = t.mul_scalar(1.0).unwrap();
    let scaled = scaler.scale_loss(&loss).unwrap();
    let scaled_val = scaled.tensor().to_flat_vec::<f32>().unwrap()[0];
    assert!(
        (scaled_val - 512.0).abs() < 1e-3,
        "2.0 * 256.0 = 512.0, got {scaled_val}"
    );
}

#[test]
fn test_grad_scaler_clean_step_does_not_set_found_inf() {
    let mut scaler = GradScaler::new(GradScalerConfig::default()).unwrap();
    scaler_clean_step(&mut scaler);
    assert!(!scaler.found_inf());
}

#[test]
fn test_grad_scaler_inf_step_sets_found_inf() {
    let mut scaler = GradScaler::new(GradScalerConfig::default()).unwrap();
    scaler_inf_step(&mut scaler);
    assert!(scaler.found_inf());
}

#[test]
fn test_grad_scaler_backoff_on_inf() {
    let mut scaler = GradScaler::new(GradScalerConfig {
        init_scale: 1024.0,
        backoff_factor: 0.5,
        ..Default::default()
    })
    .unwrap();
    scaler_inf_step(&mut scaler);
    scaler.update();
    assert!(
        (scaler.scale_factor() - 512.0).abs() < f64::EPSILON,
        "1024 * 0.5 = 512, got {}",
        scaler.scale_factor()
    );
}

#[test]
fn test_grad_scaler_growth_after_interval() {
    let mut scaler = GradScaler::new(GradScalerConfig {
        init_scale: 100.0,
        growth_factor: 2.0,
        growth_interval: 3,
        min_scale: 1.0,
        max_scale: 10000.0,
        ..Default::default()
    })
    .unwrap();

    // Two clean steps: no growth yet
    for _ in 0..2 {
        scaler_clean_step(&mut scaler);
        scaler.update();
    }
    assert!(
        (scaler.scale_factor() - 100.0).abs() < f64::EPSILON,
        "before interval: no growth"
    );

    // Third clean step: growth triggered
    scaler_clean_step(&mut scaler);
    scaler.update();
    assert!(
        (scaler.scale_factor() - 200.0).abs() < f64::EPSILON,
        "at interval: growth to 200, got {}",
        scaler.scale_factor()
    );
}

#[test]
fn test_grad_scaler_growth_clamped_at_max() {
    let mut scaler = GradScaler::new(GradScalerConfig {
        init_scale: 500.0,
        growth_factor: 3.0,
        growth_interval: 1,
        min_scale: 1.0,
        max_scale: 1000.0,
        ..Default::default()
    })
    .unwrap();
    scaler_clean_step(&mut scaler);
    scaler.update();
    assert!(
        (scaler.scale_factor() - 1000.0).abs() < f64::EPSILON,
        "clamped to max: 500*3=1500 > 1000, got {}",
        scaler.scale_factor()
    );
}

#[test]
fn test_grad_scaler_backoff_clamped_at_min() {
    let mut scaler = GradScaler::new(GradScalerConfig {
        init_scale: 2.0,
        backoff_factor: 0.5,
        min_scale: 1.0,
        max_scale: 1000.0,
        ..Default::default()
    })
    .unwrap();
    scaler_inf_step(&mut scaler);
    scaler.update();
    assert!(
        (scaler.scale_factor() - 1.0).abs() < f64::EPSILON,
        "2*0.5=1.0=min, got {}",
        scaler.scale_factor()
    );
    // Another backoff should stay at min
    scaler_inf_step(&mut scaler);
    scaler.update();
    assert!(
        (scaler.scale_factor() - 1.0).abs() < f64::EPSILON,
        "clamped to min, got {}",
        scaler.scale_factor()
    );
}

#[test]
fn test_grad_scaler_inf_resets_growth_tracker() {
    let mut scaler = GradScaler::new(GradScalerConfig {
        init_scale: 100.0,
        growth_factor: 2.0,
        growth_interval: 3,
        min_scale: 1.0,
        max_scale: 10000.0,
        ..Default::default()
    })
    .unwrap();

    // 2 clean steps
    scaler_clean_step(&mut scaler);
    scaler.update();
    scaler_clean_step(&mut scaler);
    scaler.update();
    // 1 inf step resets tracker
    scaler_inf_step(&mut scaler);
    scaler.update();
    let scale_after_inf = scaler.scale_factor();
    // 2 more clean steps: should NOT trigger growth (tracker was reset)
    scaler_clean_step(&mut scaler);
    scaler.update();
    scaler_clean_step(&mut scaler);
    scaler.update();
    assert!(
        (scaler.scale_factor() - scale_after_inf).abs() < f64::EPSILON,
        "growth tracker reset: 2 clean steps after inf not enough for interval=3"
    );
}

// =============================================================================
// GradScaler — save/load state
// =============================================================================

#[test]
fn test_grad_scaler_save_load_state_roundtrip() {
    let mut scaler = GradScaler::new(GradScalerConfig {
        init_scale: 256.0,
        growth_interval: 5,
        min_scale: 1.0,
        max_scale: 10000.0,
        ..Default::default()
    })
    .unwrap();

    // Advance state
    scaler_clean_step(&mut scaler);
    scaler.update();
    scaler_clean_step(&mut scaler);
    scaler.update();

    let state = scaler.save_state();
    assert!((state.scale - 256.0).abs() < f64::EPSILON);
    assert_eq!(state.growth_tracker, 2);

    // Create a fresh scaler and load state
    let mut scaler2 = GradScaler::new(GradScalerConfig {
        init_scale: 100.0,
        growth_interval: 5,
        min_scale: 1.0,
        max_scale: 10000.0,
        ..Default::default()
    })
    .unwrap();
    scaler2.load_state(&state).unwrap();
    assert!((scaler2.scale_factor() - 256.0).abs() < f64::EPSILON);
}

#[test]
fn test_grad_scaler_load_state_invalid_scale_rejected() {
    let mut scaler = GradScaler::new(GradScalerConfig::default()).unwrap();
    let bad_state = GradScalerState {
        scale: f64::NAN,
        growth_tracker: 0,
    };
    assert!(scaler.load_state(&bad_state).is_err());

    let bad_state2 = GradScalerState {
        scale: -1.0,
        growth_tracker: 0,
    };
    assert!(scaler.load_state(&bad_state2).is_err());
}

// =============================================================================
// GradScaler — Debug
// =============================================================================

#[test]
fn test_grad_scaler_debug_contains_type_name() {
    let scaler = GradScaler::new(GradScalerConfig::default()).unwrap();
    let debug = format!("{scaler:?}");
    assert!(debug.contains("GradScaler"), "debug: {debug}");
}

// =============================================================================
// Edge cases: zero lr, zero warmup, extreme values
// =============================================================================

#[test]
fn test_warmup_zero_base_lr_all_steps_zero() {
    let sched = WarmupSchedule::new(0.0, 100).unwrap();
    for step in [0, 50, 100, 200] {
        assert!(
            sched.lr_at_step(step).abs() < f64::EPSILON,
            "zero base_lr: step {step} should be 0"
        );
    }
}

#[test]
fn test_cosine_zero_min_lr_reaches_zero() {
    let sched = CosineSchedule::new(0.01, 0.0, 0, 100).unwrap();
    let lr = sched.lr_at_step(100);
    assert!(
        lr.abs() < 1e-10,
        "at total_steps with min=0: should be ~0, got {lr}"
    );
}

#[test]
fn test_cosine_symmetry_around_midpoint() {
    let sched = CosineSchedule::new(1.0, 0.0, 0, 1000).unwrap();
    // Cosine is symmetric: lr(x) + lr(1000-x) = 1.0 for all x in [0, 1000]
    for step in [100, 200, 300, 400] {
        let lr_lo = sched.lr_at_step(step);
        let lr_hi = sched.lr_at_step(1000 - step);
        assert!(
            (lr_lo + lr_hi - 1.0).abs() < 1e-10,
            "symmetry: lr({step}) + lr({}) = {} != 1.0",
            1000 - step,
            lr_lo + lr_hi
        );
    }
}

#[test]
fn test_clip_grad_norm_single_element_exact() {
    let (_var, mut grads) = grads_for(&[7.0]);
    let total_norm = clip_grad_norm(&mut grads, 3.5).unwrap();
    assert!((total_norm - 7.0).abs() < 1e-5);
    for (_id, g) in grads.var_grads() {
        let vals = g.to_flat_vec::<f32>().unwrap();
        assert!(
            (vals[0] - 3.5).abs() < 1e-5,
            "clipped single element: expected 3.5, got {}",
            vals[0]
        );
    }
}

#[test]
fn test_clip_grad_value_mixed_positive_negative() {
    let (_var, mut grads) = grads_for(&[10.0, -10.0, 0.3, -0.3]);
    clip_grad_value(&mut grads, 0.5).unwrap();
    for (_id, g) in grads.var_grads() {
        let vals = g.to_flat_vec::<f32>().unwrap();
        assert!((vals[0] - 0.5).abs() < 1e-6);
        assert!((vals[1] - (-0.5)).abs() < 1e-6);
        assert!((vals[2] - 0.3).abs() < 1e-6);
        assert!((vals[3] - (-0.3)).abs() < 1e-6);
    }
}

// =============================================================================
// Integration: schedule + clipping + scaler
// =============================================================================

#[test]
fn test_integration_cosine_schedule_with_grad_clipping() {
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

    let mut losses = Vec::new();
    for step in 0..25 {
        let tw = Arc::new(TrackedTensor::from_var(&w).unwrap());
        let loss = tw.sqr().unwrap().sum_keepdim(0).unwrap();
        losses.push(loss.tensor().to_flat_vec::<f32>().unwrap()[0]);

        let mut grads = backward(&loss).unwrap();
        clip_grad_norm(&mut grads, 1.0).unwrap();
        sgd.set_learning_rate(schedule.lr_at_step(step)).unwrap();
        sgd.step(&grads).unwrap();
    }

    assert!(
        losses.last().unwrap() < losses.first().unwrap(),
        "loss should decrease: first={}, last={}",
        losses.first().unwrap(),
        losses.last().unwrap()
    );
}

#[test]
fn test_integration_scaler_with_value_clipping() {
    let w = Var::new(DynTensor::from_vec(vec![50.0f32], &[1], &cpu()).unwrap());
    let mut scaler = GradScaler::new(GradScalerConfig {
        init_scale: 512.0,
        growth_interval: 100,
        ..Default::default()
    })
    .unwrap();

    let tw = Arc::new(TrackedTensor::from_var(&w).unwrap());
    let loss = tw.sqr().unwrap();
    let scaled = scaler.scale_loss(&loss).unwrap();
    let mut grads = backward(&scaled).unwrap();

    let ok = scaler.unscale_and_check(&mut grads).unwrap();
    assert!(ok);
    clip_grad_value(&mut grads, 2.0).unwrap();

    let grad = grads.get(&w).unwrap().to_flat_vec::<f32>().unwrap();
    assert!(
        (grad[0] - 2.0).abs() < 1e-5,
        "grad clamped to 2.0, got {}",
        grad[0]
    );
}

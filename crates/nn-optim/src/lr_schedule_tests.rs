#![allow(deprecated)]
// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Tests for learning rate schedulers.

use std::sync::Arc;

use nn_autodiff::{TrackedTensor, Var};
use nn_core::dyn_tensor::DynTensor;
use nn_core::test_utils::cpu;

use crate::lr_schedule::{step_with_schedule, CosineSchedule, LrSchedule, WarmupSchedule};
use crate::optimizer::Optimizer;
use crate::sgd::{Sgd, SgdConfig};

fn scalar_var(val: f32) -> Var {
    Var::new(DynTensor::from_vec(vec![val], &[1], &cpu()).unwrap())
}

// -- WarmupSchedule -----------------------------------------------------------

#[test]
fn test_warmup_starts_at_zero() {
    let sched = WarmupSchedule::new(0.01, 100).unwrap();
    assert!((sched.lr_at_step(0) - 0.0).abs() < f64::EPSILON);
}

#[test]
fn test_warmup_linear_ramp() {
    let sched = WarmupSchedule::new(0.01, 100).unwrap();
    // At step 50: lr = 0.01 * 50/100 = 0.005
    let lr = sched.lr_at_step(50);
    assert!((lr - 0.005).abs() < 1e-10, "expected 0.005, got {lr}");
}

#[test]
fn test_warmup_reaches_base_lr() {
    let sched = WarmupSchedule::new(0.01, 100).unwrap();
    let lr = sched.lr_at_step(100);
    assert!((lr - 0.01).abs() < f64::EPSILON, "expected 0.01, got {lr}");
}

#[test]
fn test_warmup_stays_constant_after() {
    let sched = WarmupSchedule::new(0.01, 100).unwrap();
    let lr = sched.lr_at_step(500);
    assert!(
        (lr - 0.01).abs() < f64::EPSILON,
        "expected 0.01 after warmup, got {lr}"
    );
}

#[test]
fn test_warmup_zero_steps() {
    // warmup_steps=0 means constant lr from start
    let sched = WarmupSchedule::new(0.01, 0).unwrap();
    assert!((sched.lr_at_step(0) - 0.01).abs() < f64::EPSILON);
    assert!((sched.lr_at_step(100) - 0.01).abs() < f64::EPSILON);
}

#[test]
fn test_warmup_accessors() {
    let sched = WarmupSchedule::new(0.01, 100).unwrap();
    assert!((sched.base_lr() - 0.01).abs() < f64::EPSILON);
    assert_eq!(sched.warmup_steps(), 100);
}

// -- CosineSchedule -----------------------------------------------------------

#[test]
fn test_cosine_starts_at_base_lr() {
    // No warmup: starts at base_lr
    let sched = CosineSchedule::new(0.01, 0.0, 0, 1000).unwrap();
    let lr = sched.lr_at_step(0);
    assert!((lr - 0.01).abs() < 1e-10, "expected 0.01, got {lr}");
}

#[test]
fn test_cosine_ends_at_min_lr() {
    let sched = CosineSchedule::new(0.01, 1e-5, 0, 1000).unwrap();
    let lr = sched.lr_at_step(1000);
    // At total_steps, lr should clamp to min_lr
    assert!((lr - 1e-5).abs() < 1e-10, "expected 1e-5, got {lr}");
}

#[test]
fn test_cosine_midpoint() {
    // At midpoint of cosine, lr should be halfway between base and min
    let sched = CosineSchedule::new(0.01, 0.0, 0, 1000).unwrap();
    let lr = sched.lr_at_step(500);
    // cos(pi * 0.5) = 0, so lr = 0 + 0.5 * (0.01 - 0) * (1 + 0) = 0.005
    assert!(
        (lr - 0.005).abs() < 1e-10,
        "expected 0.005 at midpoint, got {lr}"
    );
}

#[test]
fn test_cosine_with_warmup() {
    let sched = CosineSchedule::new(0.01, 0.0, 100, 1000).unwrap();

    // During warmup
    let lr0 = sched.lr_at_step(0);
    assert!((lr0 - 0.0).abs() < f64::EPSILON, "warmup starts at 0");

    let lr50 = sched.lr_at_step(50);
    assert!((lr50 - 0.005).abs() < 1e-10, "linear warmup midpoint");

    // End of warmup
    let lr100 = sched.lr_at_step(100);
    assert!((lr100 - 0.01).abs() < 1e-10, "warmup peak");

    // During cosine decay
    let lr550 = sched.lr_at_step(550);
    // progress = (550-100)/(1000-100) = 450/900 = 0.5
    // lr = 0 + 0.5 * 0.01 * (1 + cos(0.5 * pi)) = 0.005
    assert!(
        (lr550 - 0.005).abs() < 1e-10,
        "cosine midpoint with warmup: expected 0.005, got {lr550}"
    );
}

#[test]
fn test_cosine_past_total_clamps() {
    let sched = CosineSchedule::new(0.01, 1e-5, 0, 1000).unwrap();
    let lr = sched.lr_at_step(2000);
    assert!(
        (lr - 1e-5).abs() < 1e-10,
        "past total_steps should clamp to min_lr, got {lr}"
    );
}

#[test]
fn test_cosine_warmup_exceeds_total_rejected() {
    // Misconfiguration: warmup_steps >= total_steps must be rejected at construction
    let result = CosineSchedule::new(0.01, 0.0, 200, 100);
    assert!(
        result.is_err(),
        "warmup_steps > total_steps should be rejected"
    );
    let err = result.unwrap_err();
    assert!(
        format!("{err}").contains("warmup_steps"),
        "error should mention warmup_steps, got: {err}"
    );
}

#[test]
fn test_cosine_warmup_equals_total_rejected() {
    let result = CosineSchedule::new(0.01, 0.0, 100, 100);
    assert!(
        result.is_err(),
        "warmup_steps == total_steps should be rejected"
    );
}

#[test]
fn test_cosine_total_steps_zero_rejected() {
    let result = CosineSchedule::new(0.01, 0.0, 0, 0);
    assert!(result.is_err(), "total_steps == 0 should be rejected");
}

#[test]
fn test_cosine_min_lr_exceeds_base_lr_rejected() {
    let result = CosineSchedule::new(0.01, 0.1, 0, 1000);
    assert!(result.is_err(), "min_lr > base_lr should be rejected");
    let err = result.unwrap_err();
    assert!(
        format!("{err}").contains("min_lr"),
        "error should mention min_lr, got: {err}"
    );
}

#[test]
fn test_cosine_accessors() {
    let sched = CosineSchedule::new(0.01, 1e-5, 100, 1000).unwrap();
    assert!((sched.base_lr() - 0.01).abs() < f64::EPSILON);
    assert!((sched.min_lr() - 1e-5).abs() < f64::EPSILON);
    assert_eq!(sched.total_steps(), 1000);
}

// -- step_with_schedule integration ------------------------------------------

#[test]
fn test_step_with_schedule_warmup() {
    let x = scalar_var(5.0);
    let mut sgd = Sgd::new(
        vec![x.clone()],
        SgdConfig {
            lr: 0.0,
            ..SgdConfig::default()
        },
    )
    .unwrap(); // lr=0 initially
    let schedule = WarmupSchedule::new(0.1, 10).unwrap();

    // Step 5: lr should be 0.1 * 5/10 = 0.05
    let t = Arc::new(TrackedTensor::from_var(&x).unwrap());
    let loss = t.sqr().unwrap(); // d/dx = 2x = 10
    let grads = nn_autodiff::backward(&loss).unwrap();
    step_with_schedule(&mut sgd, &grads, &schedule, 5).unwrap();

    // After step: x = 5.0 - 0.05 * 10 = 4.5
    let val = x.data().unwrap().to_flat_vec::<f32>().unwrap()[0];
    assert!(
        (val - 4.5).abs() < 1e-4,
        "expected ~4.5 with warmup lr=0.05, got {val}"
    );
    assert!(
        (sgd.learning_rate() - 0.05).abs() < f64::EPSILON,
        "optimizer lr should be updated to 0.05"
    );
}

#[test]
fn test_step_with_schedule_cosine() {
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

    // At step 50: lr = 0.05 (midpoint of cosine)
    let t = Arc::new(TrackedTensor::from_var(&x).unwrap());
    let loss = t.sqr().unwrap();
    let grads = nn_autodiff::backward(&loss).unwrap();
    step_with_schedule(&mut sgd, &grads, &schedule, 50).unwrap();

    let val = x.data().unwrap().to_flat_vec::<f32>().unwrap()[0];
    // x = 5.0 - 0.05 * 10 = 4.5
    assert!(
        (val - 4.5).abs() < 1e-4,
        "expected ~4.5 with cosine midpoint lr, got {val}"
    );
}

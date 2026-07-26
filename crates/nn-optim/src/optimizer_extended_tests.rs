// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Extended tests for nn-optim: Adam, AdaFactor, SGD, LR scheduling,
//! gradient clipping, weight decay, checkpoint roundtrip, numerical stability,
//! convergence, and parameter bounds checking.

#![allow(deprecated)]

use std::collections::HashMap;
use std::sync::Arc;

use crate::adafactor::{AdaFactor, AdaFactorConfig};
use crate::adam::{AdamConfig, AdamW};
use crate::checkpoint::{OptimizerCheckpoint, OptimizerSnapshot};
use crate::error::OptimError;
use crate::grad_clip::{clip_grad_norm, clip_grad_value};
use crate::grad_scaler::{GradScaler, GradScalerConfig};
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

/// Build a GradStore with known gradient for a single variable.
/// Uses loss = sum(x * target), so d(loss)/dx_i = target_i.
fn make_grads(var: &Var, grad_values: &[f32]) -> nn_autodiff::GradStore {
    // Target must share the variable's shape so the element-wise multiply
    // type-checks for rank >= 2 variables (e.g. matrices), not just rank-1.
    let dims = var.data().unwrap().dims().to_vec();
    let t = Arc::new(TrackedTensor::from_var(var).unwrap());
    let target = Arc::new(TrackedTensor::from_tensor(
        DynTensor::from_vec(grad_values.to_vec(), &dims, &cpu()).unwrap(),
    ));
    let product = t.mul(&target).unwrap();
    // Reduce over every dimension to obtain a scalar loss.
    let mut loss = product;
    for d in 0..dims.len() {
        loss = loss.sum_keepdim(d).unwrap();
    }
    backward(&loss).unwrap()
}

/// Build a GradStore for multiple variables with known gradients.
fn make_multi_grads(vars: &[&Var], grad_sets: &[&[f32]]) -> nn_autodiff::GradStore {
    assert_eq!(vars.len(), grad_sets.len());
    let mut tracked = Vec::new();
    for (var, grads) in vars.iter().zip(grad_sets.iter()) {
        let n = grads.len();
        let t = Arc::new(TrackedTensor::from_var(var).unwrap());
        let target = Arc::new(TrackedTensor::from_tensor(
            DynTensor::from_vec(grads.to_vec(), &[n], &cpu()).unwrap(),
        ));
        let product = t.mul(&target).unwrap();
        tracked.push(product);
    }
    let mut total = tracked[0].sum_keepdim(0).unwrap();
    for t in &tracked[1..] {
        let s = t.sum_keepdim(0).unwrap();
        total = total.add(&s).unwrap();
    }
    backward(&total).unwrap()
}

// ============================================================================
// 1. Adam: step updates, momentum, bias correction (10 tests)
// ============================================================================

#[test]
fn test_adam_step_decreases_loss_on_quadratic() {
    let x = scalar_var(3.0);
    let config = AdamConfig {
        lr: 0.01,
        weight_decay: 0.0,
        ..AdamConfig::default()
    };
    let mut adam = AdamW::new(vec![x.clone()], config).unwrap();

    let before = get_val(&x);
    let t = Arc::new(TrackedTensor::from_var(&x).unwrap());
    let loss = t.sqr().unwrap();
    adam.backward_step(&loss).unwrap();
    let after = get_val(&x);

    // x should move toward 0 (gradient of x^2 is 2x > 0 for x=3)
    assert!(
        after < before,
        "Adam step should decrease x: before={before}, after={after}"
    );
}

#[test]
fn test_adam_momentum_accumulates_across_steps() {
    // Run two identical steps; the second should have accumulated momentum.
    let x = scalar_var(5.0);
    let config = AdamConfig {
        lr: 0.01,
        weight_decay: 0.0,
        ..AdamConfig::default()
    };
    let mut adam = AdamW::new(vec![x.clone()], config).unwrap();

    // Step 1
    let t = Arc::new(TrackedTensor::from_var(&x).unwrap());
    let loss = t.sqr().unwrap();
    adam.backward_step(&loss).unwrap();
    let after_step1 = get_val(&x);

    // Step 2
    let t = Arc::new(TrackedTensor::from_var(&x).unwrap());
    let loss = t.sqr().unwrap();
    adam.backward_step(&loss).unwrap();
    let after_step2 = get_val(&x);

    // Both steps should move toward 0
    assert!(after_step2 < after_step1, "Adam should continue decreasing");
    assert_eq!(adam.step_count(), 2);
}

#[test]
fn test_adam_bias_correction_amplifies_first_step() {
    // With beta1=0.9, bias correction on step 1 multiplies first moment by 10x.
    // Compare step magnitude with bias correction vs. what raw EMA would give.
    let x1 = scalar_var(1.0);
    let config = AdamConfig {
        lr: 0.1,
        weight_decay: 0.0,
        ..AdamConfig::default()
    };
    let mut adam = AdamW::new(vec![x1.clone()], config).unwrap();

    let t = Arc::new(TrackedTensor::from_var(&x1).unwrap());
    let loss = t.sqr().unwrap();
    adam.backward_step(&loss).unwrap();
    let val = get_val(&x1);

    // With bias correction at step 1, the effective update is lr * sign(grad) ~ 0.1
    // Without bias correction, m_hat would be much smaller (0.1 * grad * lr).
    // The key check: the update is non-trivial (not just lr * (1-beta1) * grad).
    assert!(
        (val - 1.0).abs() > 0.05,
        "bias correction should produce meaningful update: got {val}"
    );
}

#[test]
fn test_adam_second_moment_dampens_large_gradients() {
    // A large gradient should be dampened by the adaptive second moment.
    let x = scalar_var(100.0);
    let config = AdamConfig {
        lr: 0.01,
        weight_decay: 0.0,
        ..AdamConfig::default()
    };
    let mut adam = AdamW::new(vec![x.clone()], config).unwrap();

    let t = Arc::new(TrackedTensor::from_var(&x).unwrap());
    let loss = t.sqr().unwrap(); // grad = 200.0
    adam.backward_step(&loss).unwrap();

    let val = get_val(&x);
    // Despite grad=200, Adam normalizes by sqrt(v_hat) so the effective step is ~ lr
    // The update should be bounded, not proportional to 200
    let delta = (100.0 - val).abs();
    assert!(
        delta < 1.0,
        "Adam should dampen large gradient; delta={delta}, expected < 1.0"
    );
}

#[test]
fn test_adam_step_count_increments() {
    let x = scalar_var(1.0);
    let mut adam = AdamW::new(vec![x.clone()], AdamConfig::default()).unwrap();
    assert_eq!(adam.step_count(), 0);

    for expected in 1..=5 {
        let t = Arc::new(TrackedTensor::from_var(&x).unwrap());
        let loss = t.sqr().unwrap();
        adam.backward_step(&loss).unwrap();
        assert_eq!(adam.step_count(), expected);
    }
}

#[test]
fn test_adam_multiple_params_independent_moments() {
    let a = scalar_var(10.0);
    let b = scalar_var(0.1);
    let config = AdamConfig {
        lr: 0.01,
        weight_decay: 0.0,
        ..AdamConfig::default()
    };
    let mut adam = AdamW::new(vec![a.clone(), b.clone()], config).unwrap();

    // loss = a^2 + b^2 => grad_a = 20, grad_b = 0.2
    let ta = Arc::new(TrackedTensor::from_var(&a).unwrap());
    let tb = Arc::new(TrackedTensor::from_var(&b).unwrap());
    let loss = ta.sqr().unwrap().add(&tb.sqr().unwrap()).unwrap();
    adam.backward_step(&loss).unwrap();

    let a_val = get_val(&a);
    let b_val = get_val(&b);
    // Both should decrease
    assert!(a_val < 10.0, "a should decrease from 10.0");
    assert!(b_val < 0.1, "b should decrease from 0.1");
}

#[test]
fn test_adam_custom_betas() {
    let x = scalar_var(5.0);
    let config = AdamConfig {
        lr: 0.01,
        beta1: 0.5,
        beta2: 0.9,
        weight_decay: 0.0,
        ..AdamConfig::default()
    };
    let mut adam = AdamW::new(vec![x.clone()], config).unwrap();

    let t = Arc::new(TrackedTensor::from_var(&x).unwrap());
    let loss = t.sqr().unwrap();
    adam.backward_step(&loss).unwrap();

    let val = get_val(&x);
    assert!(val < 5.0, "Adam with custom betas should update: got {val}");
}

#[test]
fn test_adam_config_accessor_returns_correct_values() {
    let x = scalar_var(1.0);
    let config = AdamConfig {
        lr: 3e-4,
        beta1: 0.85,
        beta2: 0.98,
        eps: 1e-6,
        weight_decay: 0.05,
    };
    let adam = AdamW::new(vec![x], config).unwrap();
    let c = adam.config();
    assert!((c.lr - 3e-4).abs() < f64::EPSILON);
    assert!((c.beta1 - 0.85).abs() < f64::EPSILON);
    assert!((c.beta2 - 0.98).abs() < f64::EPSILON);
    assert!((c.eps - 1e-6).abs() < f64::EPSILON);
    assert!((c.weight_decay - 0.05).abs() < f64::EPSILON);
}

#[test]
fn test_adam_rejects_negative_lr() {
    let x = scalar_var(1.0);
    let config = AdamConfig {
        lr: -0.01,
        ..AdamConfig::default()
    };
    assert!(AdamW::new(vec![x], config).is_err());
}

#[test]
fn test_adam_rejects_negative_weight_decay() {
    let x = scalar_var(1.0);
    let config = AdamConfig {
        weight_decay: -0.1,
        ..AdamConfig::default()
    };
    assert!(AdamW::new(vec![x], config).is_err());
}

// ============================================================================
// 2. AdaFactor: relative step size, factored second moments (7 tests)
// ============================================================================

#[test]
fn test_adafactor_basic_step_vector() {
    // Vector params (rank < 2) use full second moments
    let x = vec_var(&[3.0, 4.0, 5.0]);
    let config = AdaFactorConfig {
        lr: 0.1,
        ..AdaFactorConfig::default()
    };
    let mut af = AdaFactor::new(vec![x.clone()], config).unwrap();

    let grads = make_grads(&x, &[1.0, 1.0, 1.0]);
    af.step(&grads).unwrap();

    let vals = get_vals(&x);
    for (i, &v) in vals.iter().enumerate() {
        let original = [3.0, 4.0, 5.0][i];
        assert!(
            v < original,
            "AdaFactor should decrease param[{i}]: {v} < {original}"
        );
    }
}

#[test]
fn test_adafactor_factored_matrix() {
    // Matrix params (rank >= 2) use factored row/col second moments
    let x = mat_var(&[1.0, 2.0, 3.0, 4.0], 2, 2);
    let config = AdaFactorConfig {
        lr: 0.1,
        ..AdaFactorConfig::default()
    };
    let mut af = AdaFactor::new(vec![x.clone()], config).unwrap();

    let grads = make_grads(&x, &[1.0, 1.0, 1.0, 1.0]);
    af.step(&grads).unwrap();

    let vals = get_vals(&x);
    // All values should have decreased from constant positive gradient
    for (i, &v) in vals.iter().enumerate() {
        let original = [1.0, 2.0, 3.0, 4.0][i];
        assert!(v < original, "param[{i}] should decrease: {v} < {original}");
    }
}

#[test]
fn test_adafactor_relative_step_size() {
    // With relative_step=true, lr is derived from parameter RMS
    let x = vec_var(&[10.0, 10.0, 10.0]);
    let config = AdaFactorConfig {
        relative_step: true,
        ..AdaFactorConfig::default()
    };
    let mut af = AdaFactor::new(vec![x.clone()], config).unwrap();

    let grads = make_grads(&x, &[1.0, 1.0, 1.0]);
    af.step(&grads).unwrap();

    let vals = get_vals(&x);
    for &v in &vals {
        assert!(v < 10.0, "relative step should produce update: {v}");
    }
}

#[test]
fn test_adafactor_with_momentum() {
    let x = vec_var(&[5.0, 5.0]);
    let config = AdaFactorConfig {
        lr: 0.1,
        beta1: Some(0.9),
        ..AdaFactorConfig::default()
    };
    let mut af = AdaFactor::new(vec![x.clone()], config).unwrap();

    // Two steps to exercise momentum
    for _ in 0..2 {
        let grads = make_grads(&x, &[1.0, 1.0]);
        af.step(&grads).unwrap();
    }

    let vals = get_vals(&x);
    for &v in &vals {
        assert!(v < 5.0, "AdaFactor with momentum should decrease: {v}");
    }
    assert_eq!(af.step_count(), 2);
}

#[test]
fn test_adafactor_step_count() {
    let x = scalar_var(1.0);
    let mut af = AdaFactor::new(vec![x.clone()], AdaFactorConfig::default()).unwrap();
    assert_eq!(af.step_count(), 0);

    let grads = make_grads(&x, &[1.0]);
    af.step(&grads).unwrap();
    assert_eq!(af.step_count(), 1);
}

#[test]
fn test_adafactor_rejects_positive_decay_rate() {
    let x = scalar_var(1.0);
    let config = AdaFactorConfig {
        decay_rate: 0.5,
        ..AdaFactorConfig::default()
    };
    assert!(AdaFactor::new(vec![x], config).is_err());
}

#[test]
fn test_adafactor_rejects_invalid_beta1() {
    let x = scalar_var(1.0);
    let config = AdaFactorConfig {
        beta1: Some(1.0),
        ..AdaFactorConfig::default()
    };
    assert!(AdaFactor::new(vec![x], config).is_err());
}

// ============================================================================
// 3. SGD with momentum: update rule, Nesterov-like behavior (6 tests)
// ============================================================================

#[test]
fn test_sgd_exact_update_no_momentum() {
    // theta_new = theta - lr * grad
    // x=5.0, grad=2.0, lr=0.1 => x_new = 5.0 - 0.1*2.0 = 4.8
    let x = scalar_var(5.0);
    let mut sgd = Sgd::new(
        vec![x.clone()],
        SgdConfig {
            lr: 0.1,
            ..Default::default()
        },
    )
    .unwrap();

    let grads = make_grads(&x, &[2.0]);
    sgd.step(&grads).unwrap();

    let val = get_val(&x);
    assert!((val - 4.8).abs() < 1e-5, "expected 4.8, got {val}");
}

#[test]
fn test_sgd_momentum_velocity_accumulation() {
    // Step 1: v = 0 * mom + grad = grad, x -= lr * v
    // Step 2: v = mom * v_prev + grad, x -= lr * v (larger step due to momentum)
    let x = scalar_var(10.0);
    let config = SgdConfig {
        lr: 0.1,
        momentum: 0.9,
        ..Default::default()
    };
    let mut sgd = Sgd::new(vec![x.clone()], config).unwrap();

    // Step 1: grad = 2.0
    let grads = make_grads(&x, &[2.0]);
    sgd.step(&grads).unwrap();
    let after1 = get_val(&x);
    let delta1 = 10.0 - after1;

    // Step 2: same gradient
    let grads = make_grads(&x, &[2.0]);
    sgd.step(&grads).unwrap();
    let after2 = get_val(&x);
    let delta2 = after1 - after2;

    // Second step should be larger due to accumulated momentum
    assert!(
        delta2 > delta1,
        "momentum should increase step size: delta1={delta1}, delta2={delta2}"
    );
}

#[test]
fn test_sgd_momentum_convergence_faster() {
    let x_no = scalar_var(10.0);
    let x_mom = scalar_var(10.0);

    let mut sgd_no = Sgd::new(
        vec![x_no.clone()],
        SgdConfig {
            lr: 0.01,
            ..Default::default()
        },
    )
    .unwrap();
    let mut sgd_mom = Sgd::new(
        vec![x_mom.clone()],
        SgdConfig {
            lr: 0.01,
            momentum: 0.9,
            ..Default::default()
        },
    )
    .unwrap();

    for _ in 0..50 {
        let t = Arc::new(TrackedTensor::from_var(&x_no).unwrap());
        let loss = t.sqr().unwrap();
        sgd_no.backward_step(&loss).unwrap();

        let t = Arc::new(TrackedTensor::from_var(&x_mom).unwrap());
        let loss = t.sqr().unwrap();
        sgd_mom.backward_step(&loss).unwrap();
    }

    let val_no = get_val(&x_no).abs();
    let val_mom = get_val(&x_mom).abs();
    assert!(
        val_mom < val_no,
        "momentum should converge faster: no_mom={val_no}, mom={val_mom}"
    );
}

#[test]
fn test_sgd_weight_decay_l2_applied_to_gradient() {
    // SGD weight decay: grad += wd * theta, then theta -= lr * grad
    // x=4.0, grad=0, wd=0.1, lr=0.5
    // effective_grad = 0 + 0.1 * 4.0 = 0.4
    // x_new = 4.0 - 0.5 * 0.4 = 3.8
    let x = scalar_var(4.0);
    let config = SgdConfig {
        lr: 0.5,
        weight_decay: 0.1,
        ..Default::default()
    };
    let mut sgd = Sgd::new(vec![x.clone()], config).unwrap();

    // Zero gradient from loss computation
    let t = Arc::new(TrackedTensor::from_var(&x).unwrap());
    let zero = Arc::new(TrackedTensor::from_tensor(
        DynTensor::from_vec(vec![0.0], &[1], &cpu()).unwrap(),
    ));
    let y = t.mul(&zero).unwrap();
    let grads = backward(&y).unwrap();
    sgd.step(&grads).unwrap();

    let val = get_val(&x);
    assert!(
        (val - 3.8).abs() < 1e-4,
        "L2 weight decay: expected ~3.8, got {val}"
    );
}

#[test]
fn test_sgd_multiple_params_update() {
    let a = scalar_var(3.0);
    let b = scalar_var(4.0);
    let mut sgd = Sgd::new(
        vec![a.clone(), b.clone()],
        SgdConfig {
            lr: 0.1,
            ..Default::default()
        },
    )
    .unwrap();

    let ta = Arc::new(TrackedTensor::from_var(&a).unwrap());
    let tb = Arc::new(TrackedTensor::from_var(&b).unwrap());
    let loss = ta.sqr().unwrap().add(&tb.sqr().unwrap()).unwrap();
    sgd.backward_step(&loss).unwrap();

    let a_val = get_val(&a);
    let b_val = get_val(&b);
    // a_new = 3 - 0.1 * 2*3 = 2.4
    // b_new = 4 - 0.1 * 2*4 = 3.2
    assert!((a_val - 2.4).abs() < 1e-5, "expected ~2.4, got {a_val}");
    assert!((b_val - 3.2).abs() < 1e-5, "expected ~3.2, got {b_val}");
}

#[test]
fn test_sgd_rejects_negative_momentum() {
    let x = scalar_var(1.0);
    let config = SgdConfig {
        momentum: -0.5,
        ..Default::default()
    };
    assert!(Sgd::new(vec![x], config).is_err());
}

// ============================================================================
// 4. Learning rate scheduling (8 tests)
// ============================================================================

#[test]
fn test_warmup_schedule_linear_ramp() {
    let sched = WarmupSchedule::new(0.1, 100).unwrap();
    assert!((sched.lr_at_step(0) - 0.0).abs() < 1e-10);
    assert!((sched.lr_at_step(50) - 0.05).abs() < 1e-10);
    assert!((sched.lr_at_step(100) - 0.1).abs() < 1e-10);
    assert!((sched.lr_at_step(200) - 0.1).abs() < 1e-10);
}

#[test]
fn test_warmup_schedule_zero_warmup_steps() {
    let sched = WarmupSchedule::new(0.05, 0).unwrap();
    assert!((sched.lr_at_step(0) - 0.05).abs() < 1e-10);
    assert!((sched.lr_at_step(100) - 0.05).abs() < 1e-10);
}

#[test]
fn test_cosine_schedule_endpoints() {
    let sched = CosineSchedule::new(0.1, 0.01, 0, 1000).unwrap();
    // Step 0: start of cosine = base_lr
    assert!((sched.lr_at_step(0) - 0.1).abs() < 1e-6);
    // Step 1000: end = min_lr
    assert!((sched.lr_at_step(1000) - 0.01).abs() < 1e-6);
}

#[test]
fn test_cosine_schedule_midpoint() {
    // At midpoint, cosine = 0 so lr = min + 0.5*(base - min)*(1+0) = (base+min)/2
    let sched = CosineSchedule::new(0.1, 0.0, 0, 1000).unwrap();
    let mid_lr = sched.lr_at_step(500);
    let expected = 0.05; // (0.1 + 0.0) / 2
    assert!(
        (mid_lr - expected).abs() < 1e-6,
        "cosine midpoint: expected {expected}, got {mid_lr}"
    );
}

#[test]
fn test_cosine_schedule_with_warmup() {
    let sched = CosineSchedule::new(0.1, 0.0, 100, 1000).unwrap();
    // During warmup: linear ramp
    let lr_50 = sched.lr_at_step(50);
    assert!((lr_50 - 0.05).abs() < 1e-6, "warmup lr at step 50: {lr_50}");
    // At warmup boundary
    let lr_100 = sched.lr_at_step(100);
    assert!((lr_100 - 0.1).abs() < 1e-6, "lr at warmup end: {lr_100}");
    // Past total
    let lr_2000 = sched.lr_at_step(2000);
    assert!((lr_2000 - 0.0).abs() < 1e-6, "lr past total: {lr_2000}");
}

#[test]
fn test_cosine_schedule_monotone_decreasing_after_warmup() {
    let sched = CosineSchedule::new(0.1, 0.001, 100, 1000).unwrap();
    let mut prev = sched.lr_at_step(100);
    for step in (101..1000).step_by(10) {
        let lr = sched.lr_at_step(step);
        assert!(
            lr <= prev + 1e-10,
            "cosine should decrease: step={step}, prev={prev}, lr={lr}"
        );
        prev = lr;
    }
}

#[test]
fn test_step_with_schedule_applies_lr() {
    let x = scalar_var(5.0);
    let mut sgd = Sgd::new(
        vec![x.clone()],
        SgdConfig {
            lr: 0.0,
            ..Default::default()
        },
    )
    .unwrap();

    let sched = WarmupSchedule::new(0.1, 10).unwrap();

    // At step 5, lr = 0.1 * 5/10 = 0.05
    let grads = make_grads(&x, &[2.0]);
    step_with_schedule(&mut sgd, &grads, &sched, 5).unwrap();

    assert!((sgd.learning_rate() - 0.05).abs() < 1e-10);
    let val = get_val(&x);
    // x_new = 5.0 - 0.05 * 2.0 = 4.9
    assert!((val - 4.9).abs() < 1e-5, "expected ~4.9, got {val}");
}

#[test]
fn test_cosine_schedule_rejects_invalid_params() {
    // min_lr > base_lr
    assert!(CosineSchedule::new(0.01, 0.1, 0, 100).is_err());
    // warmup >= total
    assert!(CosineSchedule::new(0.1, 0.01, 100, 100).is_err());
    // total = 0
    assert!(CosineSchedule::new(0.1, 0.01, 0, 0).is_err());
    // negative base_lr
    assert!(CosineSchedule::new(-0.1, 0.01, 0, 100).is_err());
}

// ============================================================================
// 5. Gradient clipping (5 tests)
// ============================================================================

#[test]
fn test_clip_grad_norm_scales_proportionally() {
    let x = vec_var(&[1.0, 1.0]);
    // grad = [6, 8], norm = 10
    let mut grads = make_grads(&x, &[6.0, 8.0]);
    let norm = clip_grad_norm(&mut grads, 5.0).unwrap();
    assert!((norm - 10.0).abs() < 1e-4, "original norm: {norm}");

    let clipped = grads.get(&x).unwrap().to_flat_vec::<f32>().unwrap();
    // scale = 5/10 = 0.5
    assert!((clipped[0] - 3.0).abs() < 1e-4);
    assert!((clipped[1] - 4.0).abs() < 1e-4);
}

#[test]
fn test_clip_grad_norm_no_op_when_within_limit() {
    let x = vec_var(&[1.0, 1.0]);
    let mut grads = make_grads(&x, &[0.3, 0.4]); // norm = 0.5
    let norm = clip_grad_norm(&mut grads, 10.0).unwrap();
    assert!((norm - 0.5).abs() < 1e-4);

    let vals = grads.get(&x).unwrap().to_flat_vec::<f32>().unwrap();
    assert!((vals[0] - 0.3).abs() < 1e-5);
    assert!((vals[1] - 0.4).abs() < 1e-5);
}

#[test]
fn test_clip_grad_value_clamps_symmetrically() {
    let x = vec_var(&[1.0, 1.0, 1.0]);
    let mut grads = make_grads(&x, &[10.0, -10.0, 0.5]);
    clip_grad_value(&mut grads, 1.0).unwrap();

    let vals = grads.get(&x).unwrap().to_flat_vec::<f32>().unwrap();
    assert!((vals[0] - 1.0).abs() < 1e-6);
    assert!((vals[1] - (-1.0)).abs() < 1e-6);
    assert!((vals[2] - 0.5).abs() < 1e-6);
}

#[test]
fn test_clip_grad_norm_multiple_vars() {
    let a = vec_var(&[1.0, 1.0]);
    let b = vec_var(&[1.0]);
    // grad_a = [3,0], grad_b = [4] => total norm = sqrt(9+16) = 5
    let mut grads = make_multi_grads(&[&a, &b], &[&[3.0, 0.0], &[4.0]]);
    let norm = clip_grad_norm(&mut grads, 2.5).unwrap();
    assert!((norm - 5.0).abs() < 0.1, "total norm: {norm}");
}

#[test]
fn test_clip_grad_norm_zero_gradient_returns_zero_norm() {
    let x = vec_var(&[1.0, 1.0]);
    let mut grads = make_grads(&x, &[0.0, 0.0]);
    let norm = clip_grad_norm(&mut grads, 1.0).unwrap();
    assert!(norm < 1e-6, "zero gradient norm: {norm}");
}

// ============================================================================
// 6. Weight decay: decoupled (Adam) vs. L2 (SGD) (4 tests)
// ============================================================================

#[test]
fn test_adam_decoupled_weight_decay_shrinks_params() {
    // Decoupled weight decay: theta *= (1 - lr * wd) before update.
    // Use small lr with a trivial gradient to observe decay effect.
    let x = scalar_var(10.0);
    let config = AdamConfig {
        lr: 0.001,
        weight_decay: 0.5,
        ..AdamConfig::default()
    };
    let mut adam = AdamW::new(vec![x.clone()], config).unwrap();

    let t = Arc::new(TrackedTensor::from_var(&x).unwrap());
    let tiny = Arc::new(TrackedTensor::from_tensor(
        DynTensor::from_vec(vec![0.001], &[1], &cpu()).unwrap(),
    ));
    let loss = t.mul(&tiny).unwrap();
    adam.backward_step(&loss).unwrap();

    let val = get_val(&x);
    assert!(val < 10.0, "decoupled weight decay should shrink: {val}");
}

#[test]
fn test_sgd_l2_weight_decay_effect() {
    // SGD applies weight decay to gradient: grad += wd * theta
    let x = scalar_var(8.0);
    let config = SgdConfig {
        lr: 0.1,
        weight_decay: 0.5,
        ..Default::default()
    };
    let mut sgd = Sgd::new(vec![x.clone()], config).unwrap();

    // Zero external gradient => effective grad = wd * theta = 0.5 * 8.0 = 4.0
    // x_new = 8.0 - 0.1 * 4.0 = 7.6
    let t = Arc::new(TrackedTensor::from_var(&x).unwrap());
    let zero = Arc::new(TrackedTensor::from_tensor(
        DynTensor::from_vec(vec![0.0], &[1], &cpu()).unwrap(),
    ));
    let loss = t.mul(&zero).unwrap();
    sgd.backward_step(&loss).unwrap();

    let val = get_val(&x);
    assert!(
        (val - 7.6).abs() < 1e-4,
        "L2 decay: expected ~7.6, got {val}"
    );
}

#[test]
fn test_weight_decay_zero_has_no_effect_adam() {
    let x = scalar_var(5.0);
    let config = AdamConfig {
        lr: 0.01,
        weight_decay: 0.0,
        ..AdamConfig::default()
    };
    let mut adam = AdamW::new(vec![x.clone()], config).unwrap();

    let t = Arc::new(TrackedTensor::from_var(&x).unwrap());
    let loss = t.sqr().unwrap();
    adam.backward_step(&loss).unwrap();
    let val_no_wd = get_val(&x);

    // Compare with the same setup (fresh optimizer)
    let y = scalar_var(5.0);
    let config2 = AdamConfig {
        lr: 0.01,
        weight_decay: 0.0,
        ..AdamConfig::default()
    };
    let mut adam2 = AdamW::new(vec![y.clone()], config2).unwrap();
    let t2 = Arc::new(TrackedTensor::from_var(&y).unwrap());
    let loss2 = t2.sqr().unwrap();
    adam2.backward_step(&loss2).unwrap();
    let val_no_wd2 = get_val(&y);

    assert!(
        (val_no_wd - val_no_wd2).abs() < 1e-7,
        "zero weight decay should produce identical results"
    );
}

#[test]
fn test_weight_decay_zero_has_no_effect_sgd() {
    let x = scalar_var(5.0);
    let config = SgdConfig {
        lr: 0.1,
        weight_decay: 0.0,
        ..Default::default()
    };
    let mut sgd = Sgd::new(vec![x.clone()], config).unwrap();

    let grads = make_grads(&x, &[2.0]);
    sgd.step(&grads).unwrap();

    let val = get_val(&x);
    // Without WD: x = 5 - 0.1*2 = 4.8
    assert!(
        (val - 4.8).abs() < 1e-5,
        "no WD SGD: expected 4.8, got {val}"
    );
}

// ============================================================================
// 7. Zero gradient initialization (2 tests)
// ============================================================================

#[test]
fn test_adam_zero_moments_initial_state() {
    // After creation, first/second moments are zero. First step should
    // produce a well-defined update (not NaN from 0/0).
    let x = scalar_var(3.0);
    let mut adam = AdamW::new(
        vec![x.clone()],
        AdamConfig {
            lr: 0.01,
            weight_decay: 0.0,
            ..AdamConfig::default()
        },
    )
    .unwrap();

    let t = Arc::new(TrackedTensor::from_var(&x).unwrap());
    let loss = t.sqr().unwrap();
    adam.backward_step(&loss).unwrap();

    let val = get_val(&x);
    assert!(
        val.is_finite(),
        "first step from zero moments should be finite: {val}"
    );
    assert!(val < 3.0, "should decrease from 3.0");
}

#[test]
fn test_sgd_zero_velocity_initial_state() {
    let x = scalar_var(5.0);
    let mut sgd = Sgd::new(
        vec![x.clone()],
        SgdConfig {
            lr: 0.1,
            momentum: 0.9,
            ..Default::default()
        },
    )
    .unwrap();

    let grads = make_grads(&x, &[2.0]);
    sgd.step(&grads).unwrap();

    let val = get_val(&x);
    assert!(
        val.is_finite(),
        "first step from zero velocity should be finite"
    );
    // First step with momentum: v = 0*0.9 + grad = grad, x -= lr*v
    assert!((val - 4.8).abs() < 1e-5, "expected 4.8, got {val}");
}

// ============================================================================
// 8. State dict save/load roundtrip (5 tests)
// ============================================================================

#[test]
fn test_adam_checkpoint_roundtrip_preserves_step_count() {
    let x = scalar_var(5.0);
    let mut adam = AdamW::new(vec![x.clone()], AdamConfig::default()).unwrap();

    for _ in 0..5 {
        let t = Arc::new(TrackedTensor::from_var(&x).unwrap());
        let loss = t.sqr().unwrap();
        adam.backward_step(&loss).unwrap();
    }
    assert_eq!(adam.step_count(), 5);

    let snapshot = adam.save_checkpoint().unwrap();

    // Create a fresh optimizer and load
    let y = scalar_var(5.0);
    let mut adam2 = AdamW::new(vec![y], AdamConfig::default()).unwrap();
    adam2.load_checkpoint(&snapshot).unwrap();

    assert_eq!(adam2.step_count(), 5);
}

#[test]
fn test_adam_checkpoint_roundtrip_preserves_config() {
    let x = scalar_var(1.0);
    let config = AdamConfig {
        lr: 5e-4,
        beta1: 0.85,
        beta2: 0.98,
        eps: 1e-6,
        weight_decay: 0.05,
    };
    let adam = AdamW::new(vec![x], config).unwrap();
    let snapshot = adam.save_checkpoint().unwrap();

    let y = scalar_var(1.0);
    let mut adam2 = AdamW::new(vec![y], AdamConfig::default()).unwrap();
    adam2.load_checkpoint(&snapshot).unwrap();

    let c = adam2.config();
    assert!((c.lr - 5e-4).abs() < 1e-10);
    assert!((c.beta1 - 0.85).abs() < 1e-10);
    assert!((c.beta2 - 0.98).abs() < 1e-10);
}

#[test]
fn test_sgd_checkpoint_roundtrip_preserves_velocity() {
    let x = scalar_var(10.0);
    let config = SgdConfig {
        lr: 0.1,
        momentum: 0.9,
        ..Default::default()
    };
    let mut sgd = Sgd::new(vec![x.clone()], config.clone()).unwrap();

    // Run a few steps to build velocity
    for _ in 0..3 {
        let grads = make_grads(&x, &[1.0]);
        sgd.step(&grads).unwrap();
    }
    let val_before_save = get_val(&x);

    let snapshot = sgd.save_checkpoint().unwrap();

    // Restore into a fresh optimizer with same var
    let y = scalar_var(val_before_save);
    let mut sgd2 = Sgd::new(vec![y.clone()], config).unwrap();
    sgd2.load_checkpoint(&snapshot).unwrap();

    // Take one more step on each; results should be very similar
    let grads1 = make_grads(&x, &[1.0]);
    sgd.step(&grads1).unwrap();
    let val1 = get_val(&x);

    let grads2 = make_grads(&y, &[1.0]);
    sgd2.step(&grads2).unwrap();
    let val2 = get_val(&y);

    assert!(
        (val1 - val2).abs() < 1e-4,
        "checkpoint roundtrip should preserve velocity: {val1} vs {val2}"
    );
}

#[test]
fn test_adam_checkpoint_rejects_shape_mismatch() {
    let x = vec_var(&[1.0, 2.0, 3.0]);
    let _adam = AdamW::new(vec![x], AdamConfig::default()).unwrap();

    // Create a snapshot with wrong-shaped moment tensors
    let mut tensors = HashMap::new();
    tensors.insert(
        "adam_0_m".to_string(),
        DynTensor::from_vec(vec![1.0, 2.0], &[2], &cpu()).unwrap(),
    );
    let snapshot = OptimizerSnapshot {
        tensors,
        metadata: serde_json::json!({}),
    };

    let y = vec_var(&[1.0, 2.0, 3.0]);
    let mut adam2 = AdamW::new(vec![y], AdamConfig::default()).unwrap();
    let result = adam2.load_checkpoint(&snapshot);
    assert!(
        matches!(result, Err(OptimError::CheckpointShapeMismatch { .. })),
        "should reject shape mismatch: {result:?}"
    );
}

#[test]
fn test_adam_checkpoint_rejects_nan_tensors() {
    let x = scalar_var(1.0);
    let mut adam = AdamW::new(vec![x], AdamConfig::default()).unwrap();

    let mut tensors = HashMap::new();
    tensors.insert(
        "adam_0_m".to_string(),
        DynTensor::from_vec(vec![f32::NAN], &[1], &cpu()).unwrap(),
    );
    let snapshot = OptimizerSnapshot {
        tensors,
        metadata: serde_json::json!({}),
    };

    let result = adam.load_checkpoint(&snapshot);
    assert!(
        matches!(result, Err(OptimError::NonFiniteCheckpoint { .. })),
        "should reject NaN checkpoint tensor: {result:?}"
    );
}

// ============================================================================
// 9. GradScaler mixed precision (3 tests)
// ============================================================================

#[test]
fn test_grad_scaler_scale_and_unscale_roundtrip() {
    let config = GradScalerConfig {
        init_scale: 256.0,
        ..Default::default()
    };
    let mut scaler = GradScaler::new(config).unwrap();

    let x = scalar_var(2.0);
    let t = Arc::new(TrackedTensor::from_var(&x).unwrap());
    let loss = t.sqr().unwrap();
    let scaled = scaler.scale_loss(&loss).unwrap();
    let mut grads = backward(&scaled).unwrap();

    // Unscale should divide by 256
    let ok = scaler.unscale_and_check(&mut grads).unwrap();
    assert!(ok, "finite gradients should pass check");

    let g = grads.get(&x).unwrap().to_flat_vec::<f32>().unwrap()[0];
    // Original grad of x^2 at x=2 is 4.0; after scale+unscale should be ~4.0
    assert!(
        (g - 4.0).abs() < 1e-3,
        "unscaled grad: expected ~4.0, got {g}"
    );
}

#[test]
fn test_grad_scaler_update_grows_on_clean_steps() {
    let config = GradScalerConfig {
        init_scale: 1.0,
        growth_interval: 2,
        growth_factor: 2.0,
        ..Default::default()
    };
    let mut scaler = GradScaler::new(config).unwrap();

    // Two clean steps should trigger growth
    let x = scalar_var(1.0);
    for _ in 0..2 {
        let t = Arc::new(TrackedTensor::from_var(&x).unwrap());
        let loss = t.sqr().unwrap();
        let scaled = scaler.scale_loss(&loss).unwrap();
        let mut grads = backward(&scaled).unwrap();
        scaler.unscale_and_check(&mut grads).unwrap();
        scaler.update();
    }

    assert!(
        (scaler.scale_factor() - 2.0).abs() < 1e-6,
        "scale should grow to 2.0, got {}",
        scaler.scale_factor()
    );
}

#[test]
fn test_grad_scaler_backoff_on_inf() {
    let config = GradScalerConfig {
        init_scale: 100.0,
        backoff_factor: 0.5,
        min_scale: 1.0,
        ..Default::default()
    };
    let mut scaler = GradScaler::new(config).unwrap();

    // Inject inf gradient
    let x = scalar_var(1.0);
    let t = Arc::new(TrackedTensor::from_var(&x).unwrap());
    let loss = t.sqr().unwrap();
    let scaled = scaler.scale_loss(&loss).unwrap();
    let mut grads = backward(&scaled).unwrap();

    for (_, grad) in grads.var_grads_mut() {
        *grad = DynTensor::from_vec(vec![f32::INFINITY], &[1], &cpu()).unwrap();
    }

    let ok = scaler.unscale_and_check(&mut grads).unwrap();
    assert!(!ok, "inf should be detected");
    scaler.update();

    assert!(
        (scaler.scale_factor() - 50.0).abs() < 1e-6,
        "scale should back off to 50.0, got {}",
        scaler.scale_factor()
    );
}

// ============================================================================
// 10. Gradient accumulation over multiple steps (2 tests)
// ============================================================================

#[test]
fn test_gradient_accumulation_adam() {
    // Simulate gradient accumulation: collect grads, average, then step
    let x = scalar_var(5.0);
    let config = AdamConfig {
        lr: 0.01,
        weight_decay: 0.0,
        ..AdamConfig::default()
    };
    let mut adam = AdamW::new(vec![x.clone()], config).unwrap();

    // Accumulate over 4 micro-batches: grad = 2x for each
    let mut accumulated_grad: Option<DynTensor> = None;
    let accumulation_steps = 4;
    for _ in 0..accumulation_steps {
        let t = Arc::new(TrackedTensor::from_var(&x).unwrap());
        let loss = t.sqr().unwrap();
        let grads = backward(&loss).unwrap();
        let g = grads.get(&x).unwrap().clone();
        accumulated_grad = Some(match accumulated_grad {
            Some(acc) => acc.add(&g).unwrap(),
            None => g,
        });
    }

    // Average the accumulated gradients
    let avg_grad = accumulated_grad
        .unwrap()
        .mul_scalar(1.0 / f64::from(accumulation_steps))
        .unwrap();

    // Create a GradStore with the averaged gradient
    // Use backward to get a properly structured GradStore, then replace the gradient
    let t = Arc::new(TrackedTensor::from_var(&x).unwrap());
    let loss = t.sqr().unwrap();
    let mut grads = backward(&loss).unwrap();
    for (_, grad) in grads.var_grads_mut() {
        *grad = avg_grad.clone();
    }

    adam.step(&grads).unwrap();
    let val = get_val(&x);
    assert!(val < 5.0, "accumulated gradient step should update: {val}");
    assert!(val.is_finite(), "result should be finite: {val}");
}

#[test]
fn test_gradient_accumulation_sgd_equivalent_to_single() {
    // For SGD without momentum: N steps with grad g/N == 1 step with grad g
    let x_accum = scalar_var(5.0);
    let x_single = scalar_var(5.0);

    let mut sgd_accum = Sgd::new(
        vec![x_accum.clone()],
        SgdConfig {
            lr: 0.1,
            ..Default::default()
        },
    )
    .unwrap();
    let mut sgd_single = Sgd::new(
        vec![x_single.clone()],
        SgdConfig {
            lr: 0.1,
            ..Default::default()
        },
    )
    .unwrap();

    // Single step with grad = 4.0
    let grads_single = make_grads(&x_single, &[4.0]);
    sgd_single.step(&grads_single).unwrap();

    // Two steps with grad = 2.0 each (accumulated = 4.0, averaged = 2.0 per step)
    // Actually: 2 steps with grad 2.0 each = x - 0.1*2 - 0.1*2 = x - 0.4
    // vs 1 step with grad 4.0 = x - 0.1*4 = x - 0.4
    let grads_half = make_grads(&x_accum, &[2.0]);
    sgd_accum.step(&grads_half).unwrap();
    let grads_half = make_grads(&x_accum, &[2.0]);
    sgd_accum.step(&grads_half).unwrap();

    let val_single = get_val(&x_single);
    let val_accum = get_val(&x_accum);

    assert!(
        (val_single - val_accum).abs() < 1e-5,
        "should be equivalent: single={val_single}, accum={val_accum}"
    );
}

// ============================================================================
// 11. Numerical stability (4 tests)
// ============================================================================

#[test]
fn test_adam_small_gradient_stability() {
    // Very small gradients should not produce NaN via underflow in v_hat
    let x = scalar_var(1.0);
    let config = AdamConfig {
        lr: 0.01,
        weight_decay: 0.0,
        ..AdamConfig::default()
    };
    let mut adam = AdamW::new(vec![x.clone()], config).unwrap();

    let grads = make_grads(&x, &[1e-7]);
    adam.step(&grads).unwrap();

    let val = get_val(&x);
    assert!(
        val.is_finite(),
        "small gradient should produce finite result: {val}"
    );
}

#[test]
fn test_adam_moderate_gradient_no_overflow() {
    // Gradients near the bounds of normal operation
    let x = scalar_var(1.0);
    let config = AdamConfig {
        lr: 1e-4,
        weight_decay: 0.0,
        ..AdamConfig::default()
    };
    let mut adam = AdamW::new(vec![x.clone()], config).unwrap();

    let grads = make_grads(&x, &[1000.0]);
    adam.step(&grads).unwrap();

    let val = get_val(&x);
    assert!(
        val.is_finite(),
        "large gradient should produce finite result: {val}"
    );
}

#[test]
fn test_sgd_very_small_lr_preserves_value() {
    let x = scalar_var(5.0);
    let mut sgd = Sgd::new(
        vec![x.clone()],
        SgdConfig {
            lr: 1e-10,
            ..Default::default()
        },
    )
    .unwrap();

    let grads = make_grads(&x, &[1.0]);
    sgd.step(&grads).unwrap();

    let val = get_val(&x);
    assert!(
        (val - 5.0).abs() < 1e-7,
        "tiny lr should barely change: {val}"
    );
}

#[test]
fn test_adafactor_eps_denom_prevents_division_by_zero() {
    // With zero gradient, second moments are zero but eps_denom prevents NaN
    let x = vec_var(&[5.0, 5.0]);
    let config = AdaFactorConfig {
        lr: 0.1,
        ..AdaFactorConfig::default()
    };
    let mut af = AdaFactor::new(vec![x.clone()], config).unwrap();

    let grads = make_grads(&x, &[0.0, 0.0]);
    // This should not produce NaN thanks to eps_denom
    af.step(&grads).unwrap();

    let vals = get_vals(&x);
    for &v in &vals {
        assert!(v.is_finite(), "zero grad should produce finite result: {v}");
    }
}

// ============================================================================
// 12. Convergence on simple quadratic (3 tests)
// ============================================================================

#[test]
fn test_sgd_converges_2d_quadratic() {
    // Minimize f(a,b) = (a-2)^2 + (b-3)^2
    let a = scalar_var(10.0);
    let b = scalar_var(-5.0);
    let mut sgd = Sgd::new(
        vec![a.clone(), b.clone()],
        SgdConfig {
            lr: 0.05,
            ..Default::default()
        },
    )
    .unwrap();

    for _ in 0..200 {
        let ta = Arc::new(TrackedTensor::from_var(&a).unwrap());
        let tb = Arc::new(TrackedTensor::from_var(&b).unwrap());
        let target_a = Arc::new(TrackedTensor::from_tensor(
            DynTensor::from_vec(vec![2.0], &[1], &cpu()).unwrap(),
        ));
        let target_b = Arc::new(TrackedTensor::from_tensor(
            DynTensor::from_vec(vec![3.0], &[1], &cpu()).unwrap(),
        ));
        let loss = ta
            .sub(&target_a)
            .unwrap()
            .sqr()
            .unwrap()
            .add(&tb.sub(&target_b).unwrap().sqr().unwrap())
            .unwrap();
        sgd.backward_step(&loss).unwrap();
    }

    let a_val = get_val(&a);
    let b_val = get_val(&b);
    assert!((a_val - 2.0).abs() < 0.5, "a should converge to 2: {a_val}");
    assert!((b_val - 3.0).abs() < 0.5, "b should converge to 3: {b_val}");
}

#[test]
fn test_adam_converges_quadratic_100_steps() {
    let x = scalar_var(10.0);
    let config = AdamConfig {
        // Adam's effective step is ~lr per iteration, so from x=10 reaching
        // |x| < 0.1 within 100 steps needs lr >= ~0.2. lr=0.1 only reaches
        // |x| ~= 2.24 in 100 steps (the optimizer is correct; the old lr was
        // simply too small for the 100-step budget).
        lr: 0.25,
        weight_decay: 0.0,
        ..AdamConfig::default()
    };
    let mut adam = AdamW::new(vec![x.clone()], config).unwrap();

    for _ in 0..100 {
        let t = Arc::new(TrackedTensor::from_var(&x).unwrap());
        let loss = t.sqr().unwrap();
        adam.backward_step(&loss).unwrap();
    }

    let val = get_val(&x).abs();
    assert!(val < 0.1, "Adam should converge to ~0: {val}");
}

#[test]
fn test_adafactor_converges_quadratic() {
    let x = vec_var(&[5.0, -3.0]);
    let config = AdaFactorConfig {
        lr: 0.5,
        ..AdaFactorConfig::default()
    };
    let mut af = AdaFactor::new(vec![x.clone()], config).unwrap();

    for _ in 0..100 {
        let t = Arc::new(TrackedTensor::from_var(&x).unwrap());
        // loss = sum(x^2)
        let sq = t.sqr().unwrap();
        let loss = sq.sum_keepdim(0).unwrap();
        af.backward_step(&loss).unwrap();
    }

    let vals = get_vals(&x);
    for &v in &vals {
        assert!(v.abs() < 1.0, "AdaFactor should converge toward 0: {v}");
    }
}

// ============================================================================
// 13. Parameter bounds checking after updates (3 tests)
// ============================================================================

#[test]
fn test_adam_rejects_nan_gradient_preserves_param() {
    let x = scalar_var(7.0);
    let mut adam = AdamW::new(vec![x.clone()], AdamConfig::default()).unwrap();

    let t = Arc::new(TrackedTensor::from_var(&x).unwrap());
    let loss = t.sqr().unwrap();
    let mut grads = backward(&loss).unwrap();

    for (_, grad) in grads.var_grads_mut() {
        *grad = DynTensor::from_vec(vec![f32::NAN], &[1], &cpu()).unwrap();
    }

    let result = adam.step(&grads);
    assert!(result.is_err());
    let val = get_val(&x);
    assert!(
        (val - 7.0).abs() < 1e-7,
        "param preserved after NaN rejection: {val}"
    );
}

#[test]
fn test_sgd_rejects_inf_gradient_preserves_param() {
    let x = scalar_var(3.0);
    let mut sgd = Sgd::new(
        vec![x.clone()],
        SgdConfig {
            lr: 0.1,
            ..Default::default()
        },
    )
    .unwrap();

    let t = Arc::new(TrackedTensor::from_var(&x).unwrap());
    let loss = t.sqr().unwrap();
    let mut grads = backward(&loss).unwrap();

    for (_, grad) in grads.var_grads_mut() {
        *grad = DynTensor::from_vec(vec![f32::INFINITY], &[1], &cpu()).unwrap();
    }

    let result = sgd.step(&grads);
    assert!(result.is_err());
    let val = get_val(&x);
    assert!(
        (val - 3.0).abs() < 1e-7,
        "param preserved after inf rejection: {val}"
    );
}

#[test]
fn test_all_optimizers_finite_after_many_steps() {
    // Run 50 steps on each optimizer and verify all results are finite
    let names = ["SGD", "SGD+mom", "AdamW"];

    for name in names {
        let x = scalar_var(5.0);
        let mut opt: Box<dyn Optimizer> = match name {
            "SGD" => Box::new(
                Sgd::new(
                    vec![x.clone()],
                    SgdConfig {
                        lr: 0.01,
                        ..Default::default()
                    },
                )
                .unwrap(),
            ),
            "SGD+mom" => Box::new(
                Sgd::new(
                    vec![x.clone()],
                    SgdConfig {
                        lr: 0.01,
                        momentum: 0.9,
                        ..Default::default()
                    },
                )
                .unwrap(),
            ),
            "AdamW" => Box::new(
                AdamW::new(
                    vec![x.clone()],
                    AdamConfig {
                        lr: 0.01,
                        weight_decay: 0.0,
                        ..AdamConfig::default()
                    },
                )
                .unwrap(),
            ),
            _ => unreachable!(),
        };

        for step in 0..50 {
            let t = Arc::new(TrackedTensor::from_var(&x).unwrap());
            let loss = t.sqr().unwrap();
            let result = opt.backward_step(&loss);
            assert!(result.is_ok(), "{name} failed at step {step}: {result:?}");
        }

        let val = get_val(&x);
        assert!(
            val.is_finite(),
            "{name} produced non-finite value after 50 steps: {val}"
        );
    }
}

// ============================================================================
// 14. Parameter group and set_learning_rate (3 tests)
// ============================================================================

#[test]
fn test_set_learning_rate_adam() {
    let x = scalar_var(1.0);
    let mut adam = AdamW::new(vec![x], AdamConfig::default()).unwrap();

    adam.set_learning_rate(5e-4).unwrap();
    assert!((adam.learning_rate() - 5e-4).abs() < 1e-15);

    adam.set_learning_rate(0.0).unwrap();
    assert!((adam.learning_rate() - 0.0).abs() < 1e-15);
}

#[test]
fn test_set_learning_rate_rejects_invalid() {
    let x = scalar_var(1.0);
    let mut sgd = Sgd::new(vec![x], SgdConfig::default()).unwrap();

    assert!(sgd.set_learning_rate(-1.0).is_err());
    assert!(sgd.set_learning_rate(f64::NAN).is_err());
    assert!(sgd.set_learning_rate(f64::INFINITY).is_err());
}

#[test]
fn test_set_learning_rate_affects_next_step() {
    let x1 = scalar_var(5.0);
    let x2 = scalar_var(5.0);

    let mut sgd1 = Sgd::new(
        vec![x1.clone()],
        SgdConfig {
            lr: 0.1,
            ..Default::default()
        },
    )
    .unwrap();
    let mut sgd2 = Sgd::new(
        vec![x2.clone()],
        SgdConfig {
            lr: 0.01,
            ..Default::default()
        },
    )
    .unwrap();

    // Change sgd2's LR to match sgd1
    sgd2.set_learning_rate(0.1).unwrap();

    let grads1 = make_grads(&x1, &[2.0]);
    sgd1.step(&grads1).unwrap();
    let grads2 = make_grads(&x2, &[2.0]);
    sgd2.step(&grads2).unwrap();

    let v1 = get_val(&x1);
    let v2 = get_val(&x2);
    assert!(
        (v1 - v2).abs() < 1e-5,
        "same LR should give same result: {v1} vs {v2}"
    );
}

// ============================================================================
// 15. Skipping vars without gradients (2 tests)
// ============================================================================

#[test]
fn test_adam_skips_var_without_gradient() {
    let x = scalar_var(5.0);
    let y = scalar_var(10.0);
    let config = AdamConfig {
        lr: 0.1,
        weight_decay: 0.0,
        ..AdamConfig::default()
    };
    let mut adam = AdamW::new(vec![x.clone(), y.clone()], config).unwrap();

    // Only x in the computation graph
    let t = Arc::new(TrackedTensor::from_var(&x).unwrap());
    let loss = t.sqr().unwrap();
    adam.backward_step(&loss).unwrap();

    let y_val = get_val(&y);
    assert!(
        (y_val - 10.0).abs() < 1e-7,
        "y should be unchanged: {y_val}"
    );
}

#[test]
fn test_adafactor_skips_var_without_gradient() {
    let x = scalar_var(3.0);
    let y = scalar_var(7.0);
    let config = AdaFactorConfig {
        lr: 0.1,
        ..AdaFactorConfig::default()
    };
    let mut af = AdaFactor::new(vec![x.clone(), y.clone()], config).unwrap();

    let t = Arc::new(TrackedTensor::from_var(&x).unwrap());
    let loss = t.sqr().unwrap();
    af.backward_step(&loss).unwrap();

    let y_val = get_val(&y);
    assert!((y_val - 7.0).abs() < 1e-7, "y should be unchanged: {y_val}");
}

// ============================================================================
// 16. Empty optimizer (2 tests)
// ============================================================================

#[test]
fn test_adam_empty_params_step_succeeds() {
    let mut adam = AdamW::new(vec![], AdamConfig::default()).unwrap();
    let x = scalar_var(1.0);
    let t = Arc::new(TrackedTensor::from_var(&x).unwrap());
    let loss = t.sqr().unwrap();
    let grads = backward(&loss).unwrap();
    adam.step(&grads).unwrap();
    assert_eq!(adam.step_count(), 1);
}

#[test]
fn test_sgd_empty_params_step_succeeds() {
    let mut sgd = Sgd::new(vec![], SgdConfig::default()).unwrap();
    let x = scalar_var(1.0);
    let t = Arc::new(TrackedTensor::from_var(&x).unwrap());
    let loss = t.sqr().unwrap();
    let grads = backward(&loss).unwrap();
    sgd.step(&grads).unwrap();
}

// ============================================================================
// 17. GradScaler configuration validation (3 tests)
// ============================================================================

#[test]
fn test_grad_scaler_rejects_zero_init_scale() {
    let config = GradScalerConfig {
        init_scale: 0.0,
        ..Default::default()
    };
    assert!(GradScaler::new(config).is_err());
}

#[test]
fn test_grad_scaler_rejects_growth_factor_leq_one() {
    let config = GradScalerConfig {
        growth_factor: 1.0,
        ..Default::default()
    };
    assert!(GradScaler::new(config).is_err());
    let config = GradScalerConfig {
        growth_factor: 0.5,
        ..Default::default()
    };
    assert!(GradScaler::new(config).is_err());
}

#[test]
fn test_grad_scaler_rejects_backoff_out_of_range() {
    let config = GradScalerConfig {
        backoff_factor: 0.0,
        ..Default::default()
    };
    assert!(GradScaler::new(config).is_err());
    let config = GradScalerConfig {
        backoff_factor: 1.0,
        ..Default::default()
    };
    assert!(GradScaler::new(config).is_err());
    let config = GradScalerConfig {
        backoff_factor: 1.5,
        ..Default::default()
    };
    assert!(GradScaler::new(config).is_err());
}

// ============================================================================
// 18. LoRA basic (2 tests — integration-level, light touch)
// ============================================================================

#[test]
fn test_lora_config_defaults() {
    let config = crate::lora::LoraConfig::default();
    assert_eq!(config.rank, 8);
    assert!((config.alpha - 8.0).abs() < f64::EPSILON);
    assert_eq!(config.targets, vec!["q_proj", "v_proj"]);
}

#[test]
fn test_lora_reject_zero_rank() {
    let weight = DynTensor::from_vec(vec![1.0, 0.0, 0.0, 1.0], &[2, 2], &cpu()).unwrap();
    let linear = nn_core::layers::Linear::new(weight, None).unwrap();
    let result = crate::lora::LoraLinear::from_linear(&linear, 0, 8.0);
    assert!(result.is_err());
}

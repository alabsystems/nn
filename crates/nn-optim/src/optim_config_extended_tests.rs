// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Part of #4560.
//!
//! Extended optimizer configuration and behavior tests.
//! Covers: Adam config validation boundary cases, moment estimation math,
//! bias correction at specific timesteps, decoupled vs L2 weight decay
//! comparison, SGD Nesterov-like momentum patterns, AdaFactor config modes,
//! LR scheduling composition, gradient clipping integration, parameter groups
//! with different LR, zero-gradient no-op, and NaN gradient detection.

use std::sync::Arc;

use nn_autodiff::{backward, GradStore, TrackedTensor, Var};
use nn_core::dyn_tensor::DynTensor;
use nn_core::test_utils::cpu;

use crate::adafactor::{AdaFactor, AdaFactorConfig};
use crate::adam::{AdamConfig, AdamW};
use crate::error::OptimError;
use crate::grad_clip::{clip_grad_norm, clip_grad_value};
use crate::lr_schedule::{step_with_schedule, CosineSchedule, LrSchedule, WarmupSchedule};
use crate::optimizer::Optimizer;
use crate::sgd::{Sgd, SgdConfig};

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

fn get_val(var: &Var) -> f32 {
    var.data().unwrap().to_flat_vec::<f32>().unwrap()[0]
}

fn get_vals(var: &Var) -> Vec<f32> {
    var.data().unwrap().to_flat_vec::<f32>().unwrap()
}

/// Build a GradStore with a known gradient for a single variable.
fn make_grads(var: &Var, grad_values: &[f32]) -> GradStore {
    let n = grad_values.len();
    let t = Arc::new(TrackedTensor::from_var(var).unwrap());
    let target = Arc::new(TrackedTensor::from_tensor(
        DynTensor::from_vec(grad_values.to_vec(), &[n], &cpu()).unwrap(),
    ));
    let product = t.mul(&target).unwrap();
    let loss = product.sum_keepdim(0).unwrap();
    backward(&loss).unwrap()
}

/// Compute the simple backward gradient for x^2 loss.
fn make_sqr_grads(var: &Var) -> GradStore {
    let t = Arc::new(TrackedTensor::from_var(var).unwrap());
    let loss = t.sqr().unwrap();
    backward(&loss).unwrap()
}

// ============================================================================
// 1. Adam config validation: boundary cases
// ============================================================================

#[test]
fn test_adam_beta1_zero_accepted() {
    // beta1=0 means no first moment accumulation (pure gradient)
    let x = scalar_var(1.0);
    let config = AdamConfig {
        beta1: 0.0,
        ..AdamConfig::default()
    };
    let result = AdamW::new(vec![x], config);
    assert!(result.is_ok(), "beta1=0 should be valid");
}

#[test]
fn test_adam_beta2_zero_accepted() {
    // beta2=0 means no second moment accumulation
    let x = scalar_var(1.0);
    let config = AdamConfig {
        beta2: 0.0,
        ..AdamConfig::default()
    };
    let result = AdamW::new(vec![x], config);
    assert!(result.is_ok(), "beta2=0 should be valid");
}

#[test]
fn test_adam_lr_zero_accepted_no_update() {
    // lr=0 should be valid but produce no parameter change
    let x = scalar_var(5.0);
    let config = AdamConfig {
        lr: 0.0,
        weight_decay: 0.0,
        ..AdamConfig::default()
    };
    let mut adam = AdamW::new(vec![x.clone()], config).unwrap();
    let grads = make_grads(&x, &[2.0]);
    adam.step(&grads).unwrap();
    let val = get_val(&x);
    assert!(
        (val - 5.0).abs() < 1e-7,
        "lr=0 should not update params: got {val}"
    );
}

#[test]
fn test_adam_eps_inf_rejected() {
    let x = scalar_var(1.0);
    let config = AdamConfig {
        eps: f64::INFINITY,
        ..AdamConfig::default()
    };
    assert!(AdamW::new(vec![x], config).is_err());
}

#[test]
fn test_adam_eps_nan_rejected() {
    let x = scalar_var(1.0);
    let config = AdamConfig {
        eps: f64::NAN,
        ..AdamConfig::default()
    };
    assert!(AdamW::new(vec![x], config).is_err());
}

#[test]
fn test_adam_weight_decay_inf_rejected() {
    let x = scalar_var(1.0);
    let config = AdamConfig {
        weight_decay: f64::INFINITY,
        ..AdamConfig::default()
    };
    assert!(AdamW::new(vec![x], config).is_err());
}

#[test]
fn test_adam_weight_decay_nan_rejected() {
    let x = scalar_var(1.0);
    let config = AdamConfig {
        weight_decay: f64::NAN,
        ..AdamConfig::default()
    };
    assert!(AdamW::new(vec![x], config).is_err());
}

#[test]
fn test_adam_beta1_near_one_rejected() {
    // beta1 = 1.0 - 1e-15 is technically < 1.0, should be accepted
    let x = scalar_var(1.0);
    let config = AdamConfig {
        beta1: 1.0 - 1e-15,
        ..AdamConfig::default()
    };
    assert!(
        AdamW::new(vec![x], config).is_ok(),
        "beta1 just below 1.0 should be accepted"
    );
}

// ============================================================================
// 2. Adam moment estimation: first moment tracks mean, second tracks variance
// ============================================================================

#[test]
fn test_adam_first_moment_tracks_gradient_direction() {
    // With constant positive gradient, the parameter should consistently decrease.
    // First moment (mean estimator) should align with gradient direction.
    let x = scalar_var(10.0);
    let config = AdamConfig {
        lr: 0.01,
        weight_decay: 0.0,
        ..AdamConfig::default()
    };
    let mut adam = AdamW::new(vec![x.clone()], config).unwrap();

    let mut prev = get_val(&x);
    for _ in 0..10 {
        let grads = make_grads(&x, &[1.0]);
        adam.step(&grads).unwrap();
        let cur = get_val(&x);
        assert!(
            cur < prev,
            "constant positive gradient should always decrease x: prev={prev}, cur={cur}"
        );
        prev = cur;
    }
}

#[test]
fn test_adam_second_moment_normalizes_gradient_scale() {
    // Two variables with very different gradient magnitudes should have
    // similar-scale updates due to the adaptive second moment normalization.
    let x_small = scalar_var(1.0);
    let x_large = scalar_var(1.0);
    let config = AdamConfig {
        lr: 0.01,
        weight_decay: 0.0,
        ..AdamConfig::default()
    };
    let mut adam_small = AdamW::new(vec![x_small.clone()], config.clone()).unwrap();
    let mut adam_large = AdamW::new(vec![x_large.clone()], config).unwrap();

    // Small gradient
    let grads_s = make_grads(&x_small, &[0.001]);
    adam_small.step(&grads_s).unwrap();
    let delta_small = (1.0 - get_val(&x_small)).abs();

    // Large gradient
    let grads_l = make_grads(&x_large, &[1000.0]);
    adam_large.step(&grads_l).unwrap();
    let delta_large = (1.0 - get_val(&x_large)).abs();

    // The ratio of updates should be much less than the ratio of gradients (1e6)
    // because Adam normalizes by sqrt(v). At step 1 with bias correction,
    // the effective update is approximately lr * sign(grad).
    let gradient_ratio = 1_000_000.0_f64;
    let update_ratio = f64::from(delta_large) / f64::from(delta_small).max(1e-15);
    assert!(
        update_ratio < gradient_ratio * 0.01,
        "Adam should normalize gradient scale: update_ratio={update_ratio}, grad_ratio={gradient_ratio}"
    );
}

#[test]
fn test_adam_alternating_gradients_small_net_update() {
    // Alternating +/- gradients should result in near-zero first moment,
    // producing small net updates.
    let x = scalar_var(5.0);
    let config = AdamConfig {
        lr: 0.01,
        weight_decay: 0.0,
        ..AdamConfig::default()
    };
    let mut adam = AdamW::new(vec![x.clone()], config).unwrap();

    for i in 0..20 {
        let grad_val = if i % 2 == 0 { 1.0 } else { -1.0 };
        let grads = make_grads(&x, &[grad_val]);
        adam.step(&grads).unwrap();
    }

    let val = get_val(&x);
    // With alternating gradients, the first moment oscillates near zero,
    // so the net displacement should be small.
    assert!(
        (val - 5.0).abs() < 1.0,
        "alternating gradients should produce small net update: val={val}"
    );
}

// ============================================================================
// 3. Adam bias correction: behavior at different timesteps
// ============================================================================

#[test]
fn test_adam_bias_correction_step1_vs_step100() {
    // At step 1, bias correction for beta1=0.9 divides by (1 - 0.9^1) = 0.1
    // At step 100, divides by (1 - 0.9^100) ~ 1.0
    // So step 1 has a 10x amplification of the raw EMA vs. step 100's ~1x.
    // We verify that the first step update is relatively large.
    let x1 = scalar_var(0.0);
    let config = AdamConfig {
        lr: 0.1,
        beta1: 0.9,
        beta2: 0.999,
        weight_decay: 0.0,
        ..AdamConfig::default()
    };
    let mut adam = AdamW::new(vec![x1.clone()], config).unwrap();

    // Apply a unit gradient at step 1
    let grads = make_grads(&x1, &[1.0]);
    adam.step(&grads).unwrap();
    let delta_step1 = get_val(&x1).abs();

    // The effective update at step 1 with bias correction should be close to lr
    // because m_hat = (0.1 * 1.0) / 0.1 = 1.0 and v_hat correction similarly
    // amplifies, making the step ~ lr * m_hat / sqrt(v_hat + eps) ~ lr.
    assert!(
        delta_step1 > 0.05,
        "bias correction at step 1 should amplify: delta={delta_step1}"
    );
}

#[test]
fn test_adam_bias_correction_converges_to_raw_ema() {
    // After many steps, 1 - beta^t -> 1.0, so bias correction becomes identity.
    // The corrected moment should equal the raw EMA for large t.
    let beta1 = 0.9_f64;
    let beta2 = 0.999_f64;
    // At t=1000: beta1^1000 ~ 0, beta2^1000 ~ 0.368
    let bc1_t1000 = 1.0 / (1.0 - beta1.powi(1000));
    let bc2_t1000 = 1.0 / (1.0 - beta2.powi(1000));
    assert!(
        (bc1_t1000 - 1.0).abs() < 1e-10,
        "beta1 correction at t=1000 should be ~1.0: {bc1_t1000}"
    );
    assert!(
        (bc2_t1000 - 1.0).abs() < 0.6,
        "beta2 correction at t=1000 should approach 1.0: {bc2_t1000}"
    );

    // At t=10000: both corrections are effectively 1.0
    let bc1_t10000 = 1.0 / (1.0 - beta1.powi(10000));
    let bc2_t10000 = 1.0 / (1.0 - beta2.powi(10000));
    assert!(
        (bc1_t10000 - 1.0).abs() < 1e-10,
        "beta1 correction at t=10000: {bc1_t10000}"
    );
    assert!(
        (bc2_t10000 - 1.0).abs() < 1e-4,
        "beta2 correction at t=10000: {bc2_t10000}"
    );
}

// ============================================================================
// 4. Adam weight decay: decoupled (AdamW) vs L2 (SGD) comparison
// ============================================================================

#[test]
fn test_adamw_decoupled_decay_independent_of_gradient() {
    // In AdamW, weight decay is applied directly to the parameter:
    // theta *= (1 - lr * wd), independently of gradient.
    // So even with zero gradient, the parameter shrinks.
    let x = scalar_var(10.0);
    let config = AdamConfig {
        lr: 0.1,
        weight_decay: 0.5,
        ..AdamConfig::default()
    };
    let mut adam = AdamW::new(vec![x.clone()], config).unwrap();

    // Give x a zero gradient (not a gradient from an unrelated variable): the
    // optimizer only updates variables present in the GradStore, so a var with
    // no entry is skipped entirely — including its decoupled weight decay.
    // With g=0, the adaptive step is 0 and only the decay factor applies.
    let grads = make_grads(&x, &[0.0]);
    adam.step(&grads).unwrap();

    let val = get_val(&x);
    // x should be shrunk by decay_factor = 1 - lr * wd = 1 - 0.05 = 0.95
    // Since x has no gradient, only decay applies: x_new = 10.0 * 0.95 = 9.5
    assert!(
        (val - 9.5).abs() < 1e-4,
        "decoupled weight decay with no gradient: expected ~9.5, got {val}"
    );
}

#[test]
fn test_sgd_l2_decay_modifies_effective_gradient() {
    // SGD L2 regularization adds wd * theta to the gradient.
    // With grad=0 and wd>0: effective_grad = wd * theta.
    let x = scalar_var(10.0);
    let config = SgdConfig {
        lr: 0.1,
        weight_decay: 0.5,
        ..Default::default()
    };
    let mut sgd = Sgd::new(vec![x.clone()], config).unwrap();

    let grads = make_grads(&x, &[0.0]);
    sgd.step(&grads).unwrap();

    let val = get_val(&x);
    // effective_grad = 0 + 0.5 * 10 = 5.0
    // x_new = 10 - 0.1 * 5 = 9.5
    assert!(
        (val - 9.5).abs() < 1e-4,
        "SGD L2 weight decay: expected ~9.5, got {val}"
    );
}

#[test]
fn test_adamw_vs_sgd_decay_with_gradient() {
    // Both start at x=10 with grad=2 and wd=0.1, lr=0.1.
    // AdamW: theta = theta * (1 - lr*wd) - lr * m_hat/sqrt(v_hat + eps)
    // SGD:   effective_grad = grad + wd*theta; theta -= lr * effective_grad
    let x_adam = scalar_var(10.0);
    let x_sgd = scalar_var(10.0);

    let mut adam = AdamW::new(
        vec![x_adam.clone()],
        AdamConfig {
            lr: 0.1,
            weight_decay: 0.1,
            ..AdamConfig::default()
        },
    )
    .unwrap();
    let mut sgd = Sgd::new(
        vec![x_sgd.clone()],
        SgdConfig {
            lr: 0.1,
            weight_decay: 0.1,
            ..Default::default()
        },
    )
    .unwrap();

    let grads_adam = make_grads(&x_adam, &[2.0]);
    let grads_sgd = make_grads(&x_sgd, &[2.0]);
    adam.step(&grads_adam).unwrap();
    sgd.step(&grads_sgd).unwrap();

    let val_adam = get_val(&x_adam);
    let val_sgd = get_val(&x_sgd);

    // They should produce different results because the decay mechanisms differ.
    // SGD: x = 10 - 0.1*(2 + 0.1*10) = 10 - 0.1*3 = 9.7
    assert!(
        (val_sgd - 9.7).abs() < 1e-4,
        "SGD L2 result: expected ~9.7, got {val_sgd}"
    );
    // AdamW: different due to adaptive moment normalization
    assert!(
        (val_adam - val_sgd).abs() > 1e-4,
        "AdamW and SGD should differ: adam={val_adam}, sgd={val_sgd}"
    );
}

// ============================================================================
// 5. SGD with momentum: accumulation and velocity growth
// ============================================================================

#[test]
fn test_sgd_momentum_first_step_equals_no_momentum() {
    // On the first step, velocity is zero, so v = 0*mom + grad = grad.
    // The update should be identical to SGD without momentum.
    let x_mom = scalar_var(5.0);
    let x_no = scalar_var(5.0);

    let mut sgd_mom = Sgd::new(
        vec![x_mom.clone()],
        SgdConfig {
            lr: 0.1,
            momentum: 0.9,
            ..Default::default()
        },
    )
    .unwrap();
    let mut sgd_no = Sgd::new(
        vec![x_no.clone()],
        SgdConfig {
            lr: 0.1,
            momentum: 0.0,
            ..Default::default()
        },
    )
    .unwrap();

    let grads_mom = make_grads(&x_mom, &[3.0]);
    let grads_no = make_grads(&x_no, &[3.0]);
    sgd_mom.step(&grads_mom).unwrap();
    sgd_no.step(&grads_no).unwrap();

    let val_mom = get_val(&x_mom);
    let val_no = get_val(&x_no);
    assert!(
        (val_mom - val_no).abs() < 1e-6,
        "first step should be identical: mom={val_mom}, no_mom={val_no}"
    );
}

#[test]
fn test_sgd_momentum_velocity_grows_with_consistent_gradient() {
    // With constant gradient, the velocity grows: v_t = mom * v_{t-1} + grad
    // v_0 = grad, v_1 = mom*grad + grad = grad*(1+mom), v_2 = grad*(1+mom+mom^2), ...
    // The update at each step should grow.
    let x = scalar_var(100.0);
    let config = SgdConfig {
        lr: 0.01,
        momentum: 0.9,
        ..Default::default()
    };
    let mut sgd = Sgd::new(vec![x.clone()], config).unwrap();

    let mut deltas = Vec::new();
    for _ in 0..5 {
        let prev = get_val(&x);
        let grads = make_grads(&x, &[1.0]);
        sgd.step(&grads).unwrap();
        let cur = get_val(&x);
        deltas.push((prev - cur).abs());
    }

    // Each successive delta should be larger (velocity accumulation)
    for i in 1..deltas.len() {
        assert!(
            deltas[i] > deltas[i - 1] - 1e-6,
            "momentum should grow updates: delta[{i}]={}, delta[{}]={}",
            deltas[i],
            i - 1,
            deltas[i - 1]
        );
    }
}

#[test]
fn test_sgd_high_momentum_overshoots() {
    // Very high momentum can cause the optimizer to overshoot the minimum.
    // Minimizing x^2 starting at x=5 with high momentum: x can go negative.
    let x = scalar_var(5.0);
    let mut sgd = Sgd::new(
        vec![x.clone()],
        SgdConfig {
            lr: 0.1,
            momentum: 0.99,
            ..Default::default()
        },
    )
    .unwrap();

    for _ in 0..50 {
        let grads = make_sqr_grads(&x);
        sgd.step(&grads).unwrap();
    }

    // With such high momentum, x may oscillate around 0
    let val = get_val(&x);
    assert!(
        val.is_finite(),
        "high momentum should still produce finite results: {val}"
    );
    // The value should have moved significantly from 5.0
    assert!(
        (val - 5.0).abs() > 1.0,
        "high momentum should produce significant movement: val={val}"
    );
}

// ============================================================================
// 6. AdaFactor config: scale_parameter, relative_step, warmup interaction
// ============================================================================

#[test]
fn test_adafactor_default_config_no_relative_step() {
    let config = AdaFactorConfig::default();
    assert!(
        !config.relative_step,
        "default should not use relative_step"
    );
    assert!(
        config.beta1.is_none(),
        "default should have no beta1 (no momentum)"
    );
}

#[test]
fn test_adafactor_relative_step_lr_scales_with_param_magnitude() {
    // With relative_step=true, lr is proportional to RMS of parameters.
    // Larger params -> larger effective lr.
    let x_small = vec_var(&[0.01, 0.01, 0.01]);
    let x_large = vec_var(&[100.0, 100.0, 100.0]);

    let config = AdaFactorConfig {
        relative_step: true,
        ..AdaFactorConfig::default()
    };
    let mut af_small = AdaFactor::new(vec![x_small.clone()], config.clone()).unwrap();
    let mut af_large = AdaFactor::new(vec![x_large.clone()], config).unwrap();

    let grads_small = make_grads(&x_small, &[1.0, 1.0, 1.0]);
    let grads_large = make_grads(&x_large, &[1.0, 1.0, 1.0]);
    af_small.step(&grads_small).unwrap();
    af_large.step(&grads_large).unwrap();

    let delta_small = (0.01 - get_vals(&x_small)[0]).abs();
    let delta_large = (100.0 - get_vals(&x_large)[0]).abs();

    // Relative step makes lr proportional to param RMS, so larger params get
    // larger updates (for the same gradient).
    assert!(
        delta_large > delta_small,
        "relative_step: larger params should get larger updates: small={delta_small}, large={delta_large}"
    );
}

#[test]
fn test_adafactor_with_warmup_schedule() {
    // AdaFactor should work with external warmup schedule via step_with_schedule
    let x = vec_var(&[5.0, 5.0]);
    let config = AdaFactorConfig {
        lr: 0.0, // will be set by schedule
        ..AdaFactorConfig::default()
    };
    let mut af = AdaFactor::new(vec![x.clone()], config).unwrap();
    let schedule = WarmupSchedule::new(0.5, 10).unwrap();

    for step in 0..15 {
        let grads = make_grads(&x, &[1.0, 1.0]);
        step_with_schedule(&mut af, &grads, &schedule, step).unwrap();
    }

    let vals = get_vals(&x);
    for &v in &vals {
        assert!(v < 5.0, "AdaFactor with warmup should update: {v}");
    }
}

#[test]
fn test_adafactor_weight_decay_shrinks_params() {
    let x = vec_var(&[10.0, 10.0]);
    let config = AdaFactorConfig {
        lr: 0.1,
        weight_decay: 0.5,
        ..AdaFactorConfig::default()
    };
    let mut af = AdaFactor::new(vec![x.clone()], config).unwrap();

    let grads = make_grads(&x, &[0.01, 0.01]);
    af.step(&grads).unwrap();

    let vals = get_vals(&x);
    for &v in &vals {
        assert!(v < 10.0, "AdaFactor weight decay should shrink: {v}");
    }
}

// ============================================================================
// 7. Learning rate scheduling: constant, linear warmup, cosine decay
// ============================================================================

#[test]
fn test_constant_lr_via_warmup_zero_steps() {
    // WarmupSchedule with warmup_steps=0 acts as a constant schedule
    let sched = WarmupSchedule::new(0.03, 0).unwrap();
    for step in [0, 1, 10, 100, 10000] {
        let lr = sched.lr_at_step(step);
        assert!(
            (lr - 0.03).abs() < 1e-15,
            "constant schedule at step {step}: expected 0.03, got {lr}"
        );
    }
}

#[test]
fn test_linear_warmup_exact_quarter_points() {
    let sched = WarmupSchedule::new(0.4, 100).unwrap();
    assert!((sched.lr_at_step(0) - 0.0).abs() < 1e-15);
    assert!((sched.lr_at_step(25) - 0.1).abs() < 1e-12);
    assert!((sched.lr_at_step(50) - 0.2).abs() < 1e-12);
    assert!((sched.lr_at_step(75) - 0.3).abs() < 1e-12);
    assert!((sched.lr_at_step(100) - 0.4).abs() < 1e-12);
}

#[test]
fn test_cosine_decay_start_equals_base_lr() {
    let sched = CosineSchedule::new(0.05, 0.001, 0, 200).unwrap();
    let lr = sched.lr_at_step(0);
    assert!(
        (lr - 0.05).abs() < 1e-12,
        "cosine start should equal base_lr: {lr}"
    );
}

#[test]
fn test_cosine_decay_end_equals_min_lr() {
    let sched = CosineSchedule::new(0.05, 0.001, 0, 200).unwrap();
    let lr = sched.lr_at_step(200);
    assert!(
        (lr - 0.001).abs() < 1e-12,
        "cosine end should equal min_lr: {lr}"
    );
}

#[test]
fn test_cosine_past_total_clamps_to_min() {
    let sched = CosineSchedule::new(0.1, 0.01, 0, 100).unwrap();
    for step in [101, 200, 1000, 100_000] {
        let lr = sched.lr_at_step(step);
        assert!(
            (lr - 0.01).abs() < 1e-12,
            "step {step} past total should clamp to min_lr: {lr}"
        );
    }
}

#[test]
fn test_warmup_then_cosine_seamless_transition() {
    let sched = CosineSchedule::new(0.1, 0.0, 50, 200).unwrap();
    // At step 49 (last warmup step): lr = 0.1 * 49/50 = 0.098
    let lr_49 = sched.lr_at_step(49);
    // At step 50 (first cosine step): progress=0, lr = 0 + 0.5*0.1*(1+cos(0)) = 0.1
    let lr_50 = sched.lr_at_step(50);
    // The transition should be smooth (no discontinuity)
    assert!(
        (lr_50 - lr_49).abs() < 0.01,
        "warmup->cosine transition should be smooth: lr_49={lr_49}, lr_50={lr_50}"
    );
}

// ============================================================================
// 8. Gradient clipping: max_norm and value clipping
// ============================================================================

#[test]
fn test_clip_grad_norm_exactly_at_threshold_no_op() {
    let x = vec_var(&[1.0, 1.0]);
    // grad = [3, 4], norm = 5
    let mut grads = make_grads(&x, &[3.0, 4.0]);
    let norm = clip_grad_norm(&mut grads, 5.0).unwrap();
    assert!((norm - 5.0).abs() < 0.1, "norm should be ~5.0: {norm}");

    let vals = grads.get(&x).unwrap().to_flat_vec::<f32>().unwrap();
    // When norm == max_norm, no scaling should occur
    assert!((vals[0] - 3.0).abs() < 0.1, "should not scale: {}", vals[0]);
    assert!((vals[1] - 4.0).abs() < 0.1, "should not scale: {}", vals[1]);
}

#[test]
fn test_clip_grad_norm_preserves_direction() {
    let x = vec_var(&[1.0, 1.0, 1.0]);
    // grad = [6, 8, 0], norm = 10, max_norm = 2 => scale = 0.2
    let mut grads = make_grads(&x, &[6.0, 8.0, 0.0]);
    clip_grad_norm(&mut grads, 2.0).unwrap();

    let vals = grads.get(&x).unwrap().to_flat_vec::<f32>().unwrap();
    // Direction should be preserved: ratio of components unchanged
    let ratio = vals[0] / vals[1];
    assert!(
        (ratio - 0.75).abs() < 1e-4,
        "clipping should preserve direction: ratio={ratio}"
    );
}

#[test]
fn test_clip_grad_value_leaves_small_values_unchanged() {
    let x = vec_var(&[1.0, 1.0, 1.0, 1.0]);
    let mut grads = make_grads(&x, &[0.1, -0.2, 0.5, -0.5]);
    clip_grad_value(&mut grads, 1.0).unwrap();

    let vals = grads.get(&x).unwrap().to_flat_vec::<f32>().unwrap();
    assert!((vals[0] - 0.1).abs() < 1e-5);
    assert!((vals[1] - (-0.2)).abs() < 1e-5);
    assert!((vals[2] - 0.5).abs() < 1e-5);
    assert!((vals[3] - (-0.5)).abs() < 1e-5);
}

#[test]
fn test_clip_grad_norm_rejects_zero_max_norm() {
    let x = vec_var(&[1.0]);
    let mut grads = make_grads(&x, &[1.0]);
    assert!(clip_grad_norm(&mut grads, 0.0).is_err());
}

#[test]
fn test_clip_grad_norm_rejects_negative_max_norm() {
    let x = vec_var(&[1.0]);
    let mut grads = make_grads(&x, &[1.0]);
    assert!(clip_grad_norm(&mut grads, -1.0).is_err());
}

#[test]
fn test_clip_grad_value_rejects_zero_clip_value() {
    let x = vec_var(&[1.0]);
    let mut grads = make_grads(&x, &[1.0]);
    assert!(clip_grad_value(&mut grads, 0.0).is_err());
}

// ============================================================================
// 9. Parameter groups: different LR for different parameter sets
// ============================================================================

#[test]
fn test_separate_optimizers_different_lr() {
    // Simulate parameter groups by using separate optimizers with different lr.
    let backbone = vec_var(&[5.0, 5.0]);
    let head = vec_var(&[5.0, 5.0]);

    let mut opt_backbone = Sgd::new(
        vec![backbone.clone()],
        SgdConfig {
            lr: 0.001, // Small lr for backbone
            ..Default::default()
        },
    )
    .unwrap();
    let mut opt_head = Sgd::new(
        vec![head.clone()],
        SgdConfig {
            lr: 0.1, // Large lr for head
            ..Default::default()
        },
    )
    .unwrap();

    let grads_bb = make_grads(&backbone, &[1.0, 1.0]);
    let grads_hd = make_grads(&head, &[1.0, 1.0]);
    opt_backbone.step(&grads_bb).unwrap();
    opt_head.step(&grads_hd).unwrap();

    let bb_vals = get_vals(&backbone);
    let hd_vals = get_vals(&head);

    let bb_delta = (5.0 - bb_vals[0]).abs();
    let hd_delta = (5.0 - hd_vals[0]).abs();

    assert!(
        hd_delta > bb_delta * 10.0,
        "head should update 100x more than backbone: hd_delta={hd_delta}, bb_delta={bb_delta}"
    );
}

#[test]
fn test_differential_lr_warmup_for_groups() {
    // Different warmup schedules for different parameter groups
    let fast_param = scalar_var(10.0);
    let slow_param = scalar_var(10.0);

    let mut opt_fast = Sgd::new(
        vec![fast_param.clone()],
        SgdConfig {
            lr: 0.0,
            ..Default::default()
        },
    )
    .unwrap();
    let mut opt_slow = Sgd::new(
        vec![slow_param.clone()],
        SgdConfig {
            lr: 0.0,
            ..Default::default()
        },
    )
    .unwrap();

    let fast_sched = WarmupSchedule::new(0.1, 5).unwrap();
    let slow_sched = WarmupSchedule::new(0.01, 20).unwrap();

    for step in 0..25 {
        let grads_f = make_sqr_grads(&fast_param);
        let grads_s = make_sqr_grads(&slow_param);
        step_with_schedule(&mut opt_fast, &grads_f, &fast_sched, step).unwrap();
        step_with_schedule(&mut opt_slow, &grads_s, &slow_sched, step).unwrap();
    }

    let fast_val = get_val(&fast_param);
    let slow_val = get_val(&slow_param);

    assert!(
        fast_val.abs() < slow_val.abs(),
        "fast group should converge more: fast={fast_val}, slow={slow_val}"
    );
}

// ============================================================================
// 10. Zero gradient: optimizer step with zero gradient doesn't change params
// ============================================================================

#[test]
fn test_sgd_zero_gradient_no_change() {
    let x = scalar_var(7.5);
    let mut sgd = Sgd::new(
        vec![x.clone()],
        SgdConfig {
            lr: 0.1,
            ..Default::default()
        },
    )
    .unwrap();

    let grads = make_grads(&x, &[0.0]);
    sgd.step(&grads).unwrap();

    let val = get_val(&x);
    assert!(
        (val - 7.5).abs() < 1e-7,
        "zero gradient should produce no update: got {val}"
    );
}

#[test]
fn test_adam_zero_gradient_minimal_change() {
    // Adam with zero gradient at step 1:
    // m = 0, v = 0 => m_hat = 0, update = 0
    // With weight_decay=0, no decay either.
    let x = scalar_var(3.0);
    let config = AdamConfig {
        lr: 0.1,
        weight_decay: 0.0,
        ..AdamConfig::default()
    };
    let mut adam = AdamW::new(vec![x.clone()], config).unwrap();

    let grads = make_grads(&x, &[0.0]);
    adam.step(&grads).unwrap();

    let val = get_val(&x);
    assert!(
        (val - 3.0).abs() < 1e-5,
        "zero gradient with zero weight decay should not change: got {val}"
    );
}

#[test]
fn test_sgd_zero_gradient_with_momentum_no_change_first_step() {
    // Even with momentum, first step with zero gradient should not move.
    let x = scalar_var(4.0);
    let mut sgd = Sgd::new(
        vec![x.clone()],
        SgdConfig {
            lr: 0.1,
            momentum: 0.9,
            ..Default::default()
        },
    )
    .unwrap();

    let grads = make_grads(&x, &[0.0]);
    sgd.step(&grads).unwrap();

    let val = get_val(&x);
    assert!(
        (val - 4.0).abs() < 1e-7,
        "zero gradient first step with momentum should not change: got {val}"
    );
}

#[test]
fn test_sgd_zero_gradient_with_weight_decay_still_decays() {
    // With weight_decay > 0, zero gradient still causes decay in SGD.
    let x = scalar_var(10.0);
    let mut sgd = Sgd::new(
        vec![x.clone()],
        SgdConfig {
            lr: 0.1,
            weight_decay: 0.1,
            ..Default::default()
        },
    )
    .unwrap();

    let grads = make_grads(&x, &[0.0]);
    sgd.step(&grads).unwrap();

    let val = get_val(&x);
    // effective_grad = 0 + 0.1*10 = 1.0, x = 10 - 0.1*1 = 9.9
    assert!(
        (val - 9.9).abs() < 1e-4,
        "zero gradient with weight decay should still decay: got {val}"
    );
}

// ============================================================================
// 11. NaN gradient handling: detection and reporting
// ============================================================================

#[test]
fn test_adam_nan_gradient_returns_error() {
    let x = scalar_var(5.0);
    let mut adam = AdamW::new(vec![x.clone()], AdamConfig::default()).unwrap();

    let t = Arc::new(TrackedTensor::from_var(&x).unwrap());
    let loss = t.sqr().unwrap();
    let mut grads = backward(&loss).unwrap();

    // Inject NaN gradient
    for (_, grad) in grads.var_grads_mut() {
        *grad = DynTensor::from_vec(vec![f32::NAN], &[1], &cpu()).unwrap();
    }

    let result = adam.step(&grads);
    assert!(result.is_err(), "NaN gradient should be rejected");
    match result.unwrap_err() {
        OptimError::NonFiniteGradient { count } => {
            assert!(count > 0, "should report non-zero count");
        }
        other => panic!("expected NonFiniteGradient, got {other:?}"),
    }
}

#[test]
fn test_sgd_nan_gradient_returns_error() {
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
        *grad = DynTensor::from_vec(vec![f32::NAN], &[1], &cpu()).unwrap();
    }

    let result = sgd.step(&grads);
    assert!(result.is_err(), "SGD should reject NaN gradient");
}

#[test]
fn test_adam_inf_gradient_returns_error() {
    let x = scalar_var(2.0);
    let mut adam = AdamW::new(vec![x.clone()], AdamConfig::default()).unwrap();

    let t = Arc::new(TrackedTensor::from_var(&x).unwrap());
    let loss = t.sqr().unwrap();
    let mut grads = backward(&loss).unwrap();

    for (_, grad) in grads.var_grads_mut() {
        *grad = DynTensor::from_vec(vec![f32::INFINITY], &[1], &cpu()).unwrap();
    }

    let result = adam.step(&grads);
    assert!(result.is_err(), "Inf gradient should be rejected");
}

#[test]
fn test_adam_neg_inf_gradient_returns_error() {
    let x = scalar_var(2.0);
    let mut adam = AdamW::new(vec![x.clone()], AdamConfig::default()).unwrap();

    let t = Arc::new(TrackedTensor::from_var(&x).unwrap());
    let loss = t.sqr().unwrap();
    let mut grads = backward(&loss).unwrap();

    for (_, grad) in grads.var_grads_mut() {
        *grad = DynTensor::from_vec(vec![f32::NEG_INFINITY], &[1], &cpu()).unwrap();
    }

    let result = adam.step(&grads);
    assert!(result.is_err(), "NEG_INFINITY gradient should be rejected");
}

#[test]
fn test_nan_gradient_preserves_parameter_adam() {
    // When NaN is rejected, the parameter should remain unchanged.
    let x = scalar_var(7.0);
    let mut adam = AdamW::new(vec![x.clone()], AdamConfig::default()).unwrap();

    let t = Arc::new(TrackedTensor::from_var(&x).unwrap());
    let loss = t.sqr().unwrap();
    let mut grads = backward(&loss).unwrap();

    for (_, grad) in grads.var_grads_mut() {
        *grad = DynTensor::from_vec(vec![f32::NAN], &[1], &cpu()).unwrap();
    }

    let _ = adam.step(&grads); // Expected to fail
    let val = get_val(&x);
    assert!(
        (val - 7.0).abs() < 1e-7,
        "NaN rejection should preserve parameter: got {val}"
    );
}

#[test]
fn test_nan_in_multi_element_gradient_detected() {
    let x = vec_var(&[1.0, 2.0, 3.0]);
    let mut adam = AdamW::new(vec![x.clone()], AdamConfig::default()).unwrap();

    let t = Arc::new(TrackedTensor::from_var(&x).unwrap());
    let loss = t.sqr().unwrap().sum_keepdim(0).unwrap();
    let mut grads = backward(&loss).unwrap();

    // Only one element is NaN
    for (_, grad) in grads.var_grads_mut() {
        *grad = DynTensor::from_vec(vec![1.0, f32::NAN, 3.0], &[3], &cpu()).unwrap();
    }

    let result = adam.step(&grads);
    assert!(result.is_err(), "single NaN element should be detected");
}

// ============================================================================
// 12. Gradient clipping + optimizer integration
// ============================================================================

#[test]
fn test_clip_then_adam_step_produces_bounded_update() {
    let x = scalar_var(5.0);
    let mut adam = AdamW::new(
        vec![x.clone()],
        AdamConfig {
            lr: 0.1,
            weight_decay: 0.0,
            ..AdamConfig::default()
        },
    )
    .unwrap();

    // Create large gradient
    let t = Arc::new(TrackedTensor::from_var(&x).unwrap());
    let loss = t.sqr().unwrap();
    let mut grads = backward(&loss).unwrap();

    // Clip to small norm
    clip_grad_norm(&mut grads, 0.1).unwrap();
    adam.step(&grads).unwrap();

    let val = get_val(&x);
    // The clipped gradient is small, so the update should be bounded
    assert!(
        (val - 5.0).abs() < 1.0,
        "clipped gradient should produce bounded update: {val}"
    );
}

#[test]
fn test_clip_value_then_sgd_step() {
    let x = vec_var(&[1.0, 1.0]);
    let mut sgd = Sgd::new(
        vec![x.clone()],
        SgdConfig {
            lr: 0.1,
            ..Default::default()
        },
    )
    .unwrap();

    let mut grads = make_grads(&x, &[100.0, -100.0]);
    clip_grad_value(&mut grads, 1.0).unwrap();
    sgd.step(&grads).unwrap();

    let vals = get_vals(&x);
    // After clipping: grads = [1.0, -1.0], then x -= 0.1 * grad
    assert!(
        (vals[0] - 0.9).abs() < 1e-4,
        "clipped value update: expected ~0.9, got {}",
        vals[0]
    );
    assert!(
        (vals[1] - 1.1).abs() < 1e-4,
        "clipped value update: expected ~1.1, got {}",
        vals[1]
    );
}

// ============================================================================
// 13. AdaFactor eps validation
// ============================================================================

#[test]
fn test_adafactor_eps_denom_inf_rejected() {
    let x = scalar_var(1.0);
    assert!(AdaFactor::new(
        vec![x],
        AdaFactorConfig {
            eps_denom: f64::INFINITY,
            ..AdaFactorConfig::default()
        },
    )
    .is_err());
}

#[test]
fn test_adafactor_eps_rms_inf_rejected() {
    let x = scalar_var(1.0);
    assert!(AdaFactor::new(
        vec![x],
        AdaFactorConfig {
            eps_rms: f64::INFINITY,
            ..AdaFactorConfig::default()
        },
    )
    .is_err());
}

#[test]
fn test_adafactor_lr_inf_rejected() {
    let x = scalar_var(1.0);
    assert!(AdaFactor::new(
        vec![x],
        AdaFactorConfig {
            lr: f64::INFINITY,
            ..AdaFactorConfig::default()
        },
    )
    .is_err());
}

#[test]
fn test_adafactor_lr_nan_rejected() {
    let x = scalar_var(1.0);
    assert!(AdaFactor::new(
        vec![x],
        AdaFactorConfig {
            lr: f64::NAN,
            ..AdaFactorConfig::default()
        },
    )
    .is_err());
}

// ============================================================================
// 14. Combined: schedule + clipping + optimizer
// ============================================================================

#[test]
fn test_warmup_clip_adam_training_loop() {
    let x = scalar_var(10.0);
    let mut adam = AdamW::new(
        vec![x.clone()],
        AdamConfig {
            lr: 0.0,
            weight_decay: 0.0,
            ..AdamConfig::default()
        },
    )
    .unwrap();
    let schedule = WarmupSchedule::new(0.1, 5).unwrap();

    // Adam moves ~lr (0.1) per step after warmup, so reducing |x| from 10 to
    // below 5 needs ~60 steps; 20 steps only reaches ~8.3. The optimizer and
    // clipping are correct — the step budget was simply too small.
    for step in 0..80 {
        // Set LR from schedule
        adam.set_learning_rate(schedule.lr_at_step(step)).unwrap();

        let t = Arc::new(TrackedTensor::from_var(&x).unwrap());
        let loss = t.sqr().unwrap();
        let mut grads = backward(&loss).unwrap();

        // Clip gradients
        clip_grad_norm(&mut grads, 5.0).unwrap();

        adam.step(&grads).unwrap();
    }

    let val = get_val(&x);
    assert!(
        val.abs() < 5.0,
        "warmup + clip + adam should converge: {val}"
    );
}

#[test]
fn test_cosine_clip_sgd_training_loop() {
    let x = vec_var(&[8.0, -6.0]);
    let mut sgd = Sgd::new(
        vec![x.clone()],
        SgdConfig {
            lr: 0.0,
            momentum: 0.9,
            ..Default::default()
        },
    )
    .unwrap();
    let schedule = CosineSchedule::new(0.05, 0.001, 5, 50).unwrap();

    for step in 0..40 {
        sgd.set_learning_rate(schedule.lr_at_step(step)).unwrap();

        let t = Arc::new(TrackedTensor::from_var(&x).unwrap());
        let loss = t.sqr().unwrap().sum_keepdim(0).unwrap();
        let mut grads = backward(&loss).unwrap();

        clip_grad_value(&mut grads, 10.0).unwrap();
        sgd.step(&grads).unwrap();
    }

    let vals = get_vals(&x);
    for &v in &vals {
        assert!(
            v.abs() < 5.0,
            "cosine + clip + momentum SGD should converge: {v}"
        );
    }
}

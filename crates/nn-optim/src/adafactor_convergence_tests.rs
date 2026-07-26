// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! AdaFactor convergence and behavioral tests.
//!
//! Complements `adafactor_tests.rs` (in adafactor.rs) with convergence analysis,
//! multi-variable optimization, 3D tensor support, relative_step vs fixed lr
//! comparison, and gradient accumulation patterns.

use std::sync::Arc;

use nn_autodiff::{backward, GradStore, TrackedTensor, Var};
use nn_core::dyn_tensor::DynTensor;
use nn_core::test_utils::cpu;

use crate::adafactor::{AdaFactor, AdaFactorConfig};
use crate::lr_schedule::{step_with_schedule, CosineSchedule, LrSchedule, WarmupSchedule};
use crate::optimizer::Optimizer;

/// Helper: create a Var from a flat vec with given shape.
fn make_var(vals: &[f32], shape: &[usize]) -> Var {
    Var::new(DynTensor::from_vec(vals.to_vec(), shape, &cpu()).unwrap())
}

/// Compute loss = sum(x^2) across all dims and return (loss_tracked, loss_scalar, grads).
fn quadratic_loss(var: &Var) -> (Arc<TrackedTensor>, f64, GradStore) {
    let t = Arc::new(TrackedTensor::from_var(var).unwrap());
    let sq = t.sqr().unwrap();
    let mut loss = sq;
    for d in 0..var.data().unwrap().dims().len() {
        loss = loss.sum_keepdim(d).unwrap();
    }
    let loss_val: f64 = loss
        .tensor()
        .to_flat_vec::<f32>()
        .unwrap()
        .iter()
        .map(|&v| f64::from(v))
        .sum();
    let grads = backward(&loss).unwrap();
    (loss, loss_val, grads)
}

/// Compute scalar loss value for a Var (sum of squares).
fn loss_of(var: &Var) -> f64 {
    var.data()
        .unwrap()
        .to_flat_vec::<f32>()
        .unwrap()
        .iter()
        .map(|&v| f64::from(v) * f64::from(v))
        .sum()
}

// ============================================================================
// 1. Config construction with defaults
// ============================================================================

#[test]
fn test_adafactor_config_default_construction() {
    let config = AdaFactorConfig::default();
    assert!((config.lr - 1e-3).abs() < 1e-15);
    assert!(!config.relative_step);
    assert!((config.eps_rms - 1e-3).abs() < 1e-15);
    assert!((config.eps_denom - 1e-30).abs() < 1e-45);
    assert!((config.decay_rate - (-0.8)).abs() < 1e-15);
    assert!(config.beta1.is_none());
    assert!((config.weight_decay - 0.0).abs() < 1e-15);

    // Verify config can be cloned and compared
    let config2 = config.clone();
    assert_eq!(config, config2);
}

#[test]
fn test_adafactor_config_custom_construction() {
    let config = AdaFactorConfig {
        lr: 0.05,
        relative_step: true,
        eps_rms: 1e-2,
        eps_denom: 1e-20,
        decay_rate: -0.5,
        beta1: Some(0.9),
        weight_decay: 0.01,
    };
    assert!((config.lr - 0.05).abs() < 1e-15);
    assert!(config.relative_step);
    assert!((config.eps_rms - 1e-2).abs() < 1e-15);
    assert!((config.eps_denom - 1e-20).abs() < 1e-35);
    assert!((config.decay_rate - (-0.5)).abs() < 1e-15);
    assert_eq!(config.beta1, Some(0.9));
    assert!((config.weight_decay - 0.01).abs() < 1e-15);
}

// ============================================================================
// 2. Single step on 1D parameter
// ============================================================================

#[test]
fn test_adafactor_single_step_1d_gradient_direction() {
    // 1D parameter uses full (non-factored) second moments.
    // After one step of gradient descent on f(x)=sum(x^2), each param should
    // move toward zero (the minimum).
    let var = make_var(&[3.0, -5.0, 0.0, 7.0], &[4]);
    let initial = var.data().unwrap().to_flat_vec::<f32>().unwrap();
    let config = AdaFactorConfig {
        lr: 0.1,
        relative_step: false,
        ..Default::default()
    };
    let mut opt = AdaFactor::new(vec![var.clone()], config).unwrap();

    let (_, _, grads) = quadratic_loss(&var);
    opt.step(&grads).unwrap();

    let after = var.data().unwrap().to_flat_vec::<f32>().unwrap();
    // Positive params should decrease, negative should increase (toward 0).
    // Zero param should stay near zero.
    assert!(after[0] < initial[0], "positive param should decrease");
    assert!(
        after[1] > initial[1],
        "negative param should increase toward 0"
    );
    assert!(
        after[2].abs() < 1e-6,
        "zero param should stay near zero, got {}",
        after[2]
    );
    assert!(after[3] < initial[3], "positive param should decrease");
}

// ============================================================================
// 3. Single step on 2D parameter (factored second moments)
// ============================================================================

#[test]
fn test_adafactor_single_step_2d_factored_moments() {
    // 2D parameter triggers factored second moments (row + col factors).
    // Verify the optimizer initializes factored state correctly and takes a valid step.
    let var = make_var(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
    let config = AdaFactorConfig {
        lr: 0.05,
        relative_step: false,
        ..Default::default()
    };
    let mut opt = AdaFactor::new(vec![var.clone()], config).unwrap();

    let initial_loss = loss_of(&var);
    let (_, _, grads) = quadratic_loss(&var);
    opt.step(&grads).unwrap();
    let after_loss = loss_of(&var);

    assert!(
        after_loss < initial_loss,
        "loss should decrease after one step: {after_loss} >= {initial_loss}"
    );
    assert_eq!(opt.step_count(), 1);
}

// ============================================================================
// 4. Multiple steps show loss decrease on simple quadratic
// ============================================================================

#[test]
fn test_adafactor_quadratic_convergence_overall() {
    // On f(x) = sum(x^2), AdaFactor should decrease loss over training.
    // Note: AdaFactor may not be strictly monotonic at every step due to its
    // adaptive second-moment estimation, but the overall trend should be downward.
    let var = make_var(&[10.0, -8.0, 5.0], &[3]);
    let config = AdaFactorConfig {
        lr: 0.1,
        relative_step: false,
        ..Default::default()
    };
    let mut opt = AdaFactor::new(vec![var.clone()], config).unwrap();

    let mut losses = Vec::new();
    for _ in 0..30 {
        let (_, loss_val, grads) = quadratic_loss(&var);
        losses.push(loss_val);
        opt.step(&grads).unwrap();
    }

    // First loss should be near 189 (10^2 + 8^2 + 5^2).
    assert!(
        (losses[0] - 189.0).abs() < 1.0,
        "initial loss should be ~189, got {}",
        losses[0]
    );

    // Loss at step 10 should be less than initial loss.
    assert!(
        losses[10] < losses[0],
        "loss at step 10 ({}) should be less than initial ({})",
        losses[10],
        losses[0]
    );

    // Final loss should be significantly less than initial.
    let final_loss = loss_of(&var);
    assert!(
        final_loss < 150.0,
        "loss should decrease from 189, got {final_loss}"
    );
}

#[test]
fn test_adafactor_quadratic_convergence_with_momentum() {
    // Momentum (beta1) should also converge on quadratic loss.
    let var = make_var(&[8.0, -6.0, 4.0, -2.0], &[4]);
    let config = AdaFactorConfig {
        lr: 0.1,
        relative_step: false,
        beta1: Some(0.9),
        ..Default::default()
    };
    let mut opt = AdaFactor::new(vec![var.clone()], config).unwrap();

    let initial_loss = loss_of(&var);
    for _ in 0..40 {
        let (_, _, grads) = quadratic_loss(&var);
        opt.step(&grads).unwrap();
    }
    let final_loss = loss_of(&var);
    assert!(
        final_loss < initial_loss * 0.5,
        "momentum variant should converge: initial={initial_loss}, final={final_loss}"
    );
}

// ============================================================================
// 5. Weight decay application (parameter shrinks)
// ============================================================================

#[test]
fn test_adafactor_weight_decay_shrinks_params() {
    // With weight decay, parameters should shrink faster than without.
    // Run identical initial conditions with and without weight decay.
    let init = vec![20.0, -15.0, 10.0];

    let var_wd = make_var(&init, &[3]);
    let var_no = make_var(&init, &[3]);

    let config_wd = AdaFactorConfig {
        lr: 0.01,
        relative_step: false,
        weight_decay: 0.5, // strong decay
        ..Default::default()
    };
    let config_no = AdaFactorConfig {
        lr: 0.01,
        relative_step: false,
        weight_decay: 0.0,
        ..Default::default()
    };

    let mut opt_wd = AdaFactor::new(vec![var_wd.clone()], config_wd).unwrap();
    let mut opt_no = AdaFactor::new(vec![var_no.clone()], config_no).unwrap();

    for _ in 0..15 {
        let (_, _, g_wd) = quadratic_loss(&var_wd);
        opt_wd.step(&g_wd).unwrap();

        let (_, _, g_no) = quadratic_loss(&var_no);
        opt_no.step(&g_no).unwrap();
    }

    let vals_wd = var_wd.data().unwrap().to_flat_vec::<f32>().unwrap();
    let vals_no = var_no.data().unwrap().to_flat_vec::<f32>().unwrap();

    // Weight decay should push params closer to zero.
    for (i, (&wd, &no)) in vals_wd.iter().zip(vals_no.iter()).enumerate() {
        assert!(
            wd.abs() <= no.abs() + 1e-5,
            "weight decay should shrink param[{i}] more: |wd|={}, |no|={}",
            wd.abs(),
            no.abs()
        );
    }
}

#[test]
fn test_adafactor_weight_decay_relative_step() {
    // Weight decay should also work with relative_step mode.
    let var = make_var(&[10.0, -10.0], &[2]);
    let config = AdaFactorConfig {
        relative_step: true,
        weight_decay: 0.1,
        ..Default::default()
    };
    let mut opt = AdaFactor::new(vec![var.clone()], config).unwrap();

    let initial_loss = loss_of(&var);
    for _ in 0..20 {
        let (_, _, grads) = quadratic_loss(&var);
        opt.step(&grads).unwrap();
    }
    let final_loss = loss_of(&var);
    assert!(
        final_loss < initial_loss,
        "relative_step + weight_decay should still converge: init={initial_loss}, final={final_loss}"
    );
}

// ============================================================================
// 6. Learning rate schedule integration
// ============================================================================

#[test]
fn test_adafactor_warmup_schedule_lr_ramp() {
    // During warmup, the LR should ramp from 0 to base_lr.
    let var = make_var(&[5.0, -3.0], &[2]);
    let config = AdaFactorConfig {
        lr: 0.0, // overwritten by schedule
        relative_step: false,
        ..Default::default()
    };
    let mut opt = AdaFactor::new(vec![var.clone()], config).unwrap();
    let sched = WarmupSchedule::new(0.1, 10).unwrap();

    // At step 0, LR should be 0.
    assert!((sched.lr_at_step(0) - 0.0).abs() < 1e-15);

    // At step 5 (halfway), LR should be 0.05.
    assert!((sched.lr_at_step(5) - 0.05).abs() < 1e-10);

    // At step 10 (end of warmup), LR should be 0.1.
    assert!((sched.lr_at_step(10) - 0.1).abs() < 1e-10);

    // Run a few steps and verify opt's LR is updated.
    for step in 0..15 {
        let (_, _, grads) = quadratic_loss(&var);
        step_with_schedule(&mut opt, &grads, &sched, step).unwrap();
    }
    assert!(
        (opt.learning_rate() - 0.1).abs() < 1e-10,
        "LR should be at base_lr after warmup, got {}",
        opt.learning_rate()
    );

    // Parameters should have moved.
    let vals = var.data().unwrap().to_flat_vec::<f32>().unwrap();
    assert!(vals[0] < 5.0, "param should decrease from 5.0");
}

#[test]
fn test_adafactor_cosine_schedule_decay() {
    // CosineSchedule decays from base_lr to min_lr after warmup.
    let var = make_var(&[4.0, -4.0], &[2]);
    let config = AdaFactorConfig {
        lr: 0.0,
        relative_step: false,
        ..Default::default()
    };
    let mut opt = AdaFactor::new(vec![var.clone()], config).unwrap();
    let sched = CosineSchedule::new(0.1, 0.001, 5, 100).unwrap();

    // Mid-schedule LR should be between min and max.
    let mid_lr = sched.lr_at_step(50);
    assert!(
        mid_lr > 0.001 && mid_lr < 0.1,
        "mid-schedule LR should be between bounds, got {mid_lr}"
    );

    // At end of schedule, LR should be near min_lr.
    let end_lr = sched.lr_at_step(99);
    assert!(
        end_lr < 0.01,
        "end-schedule LR should be near min_lr, got {end_lr}"
    );

    // Run training steps.
    let initial_loss = loss_of(&var);
    for step in 0..80 {
        let (_, _, grads) = quadratic_loss(&var);
        step_with_schedule(&mut opt, &grads, &sched, step).unwrap();
    }
    let final_loss = loss_of(&var);
    assert!(
        final_loss < initial_loss,
        "cosine schedule should allow convergence: init={initial_loss}, final={final_loss}"
    );
}

// ============================================================================
// 7. Config with relative_step vs fixed lr
// ============================================================================

#[test]
fn test_adafactor_relative_step_vs_fixed_both_converge() {
    // Both relative_step and fixed lr should converge on quadratic loss.
    let init = vec![8.0, -6.0, 4.0];

    let var_rel = make_var(&init, &[3]);
    let var_fix = make_var(&init, &[3]);

    let config_rel = AdaFactorConfig {
        relative_step: true,
        ..Default::default()
    };
    let config_fix = AdaFactorConfig {
        lr: 0.1,
        relative_step: false,
        ..Default::default()
    };

    let mut opt_rel = AdaFactor::new(vec![var_rel.clone()], config_rel).unwrap();
    let mut opt_fix = AdaFactor::new(vec![var_fix.clone()], config_fix).unwrap();

    let initial_loss = loss_of(&var_rel);

    for _ in 0..30 {
        let (_, _, g_rel) = quadratic_loss(&var_rel);
        opt_rel.step(&g_rel).unwrap();

        let (_, _, g_fix) = quadratic_loss(&var_fix);
        opt_fix.step(&g_fix).unwrap();
    }

    let loss_rel = loss_of(&var_rel);
    let loss_fix = loss_of(&var_fix);

    assert!(
        loss_rel < initial_loss,
        "relative_step should converge: initial={initial_loss}, final={loss_rel}"
    );
    assert!(
        loss_fix < initial_loss,
        "fixed lr should converge: initial={initial_loss}, final={loss_fix}"
    );
}

#[test]
fn test_adafactor_relative_step_adapts_to_param_scale() {
    // Relative step size scales with parameter RMS.
    // Large params should get larger updates than small params (in absolute terms).
    let var_large = make_var(&[100.0, -100.0], &[2]);
    let var_small = make_var(&[0.1, -0.1], &[2]);

    let config = AdaFactorConfig {
        relative_step: true,
        ..Default::default()
    };

    let mut opt_large = AdaFactor::new(vec![var_large.clone()], config.clone()).unwrap();
    let mut opt_small = AdaFactor::new(vec![var_small.clone()], config).unwrap();

    let before_large = var_large.data().unwrap().to_flat_vec::<f32>().unwrap();
    let before_small = var_small.data().unwrap().to_flat_vec::<f32>().unwrap();

    let (_, _, g_large) = quadratic_loss(&var_large);
    opt_large.step(&g_large).unwrap();
    let (_, _, g_small) = quadratic_loss(&var_small);
    opt_small.step(&g_small).unwrap();

    let after_large = var_large.data().unwrap().to_flat_vec::<f32>().unwrap();
    let after_small = var_small.data().unwrap().to_flat_vec::<f32>().unwrap();

    let delta_large = (after_large[0] - before_large[0]).abs();
    let delta_small = (after_small[0] - before_small[0]).abs();

    // Absolute step size should scale with parameter magnitude.
    assert!(
        delta_large > delta_small,
        "relative step should produce larger absolute update for larger params: \
         delta_large={delta_large}, delta_small={delta_small}"
    );
}

// ============================================================================
// 8. 3D tensor parameter (rank >= 2, uses factored)
// ============================================================================

#[test]
fn test_adafactor_3d_tensor_factored() {
    // 3D tensor (rank=3 >= 2) should use factored second moments.
    // Shape [2, 3, 4] => row_factor=[2,3,1], col_factor=[2,1,4].
    let vals: Vec<f32> = (0..24).map(|i| (i as f32 - 12.0) * 0.5).collect();
    let var = make_var(&vals, &[2, 3, 4]);
    let config = AdaFactorConfig {
        lr: 0.1,
        relative_step: false,
        ..Default::default()
    };
    let mut opt = AdaFactor::new(vec![var.clone()], config).unwrap();

    let initial_loss = loss_of(&var);
    for _ in 0..20 {
        let t = Arc::new(TrackedTensor::from_var(&var).unwrap());
        let sq = t.sqr().unwrap();
        // Sum across all dims to get scalar loss.
        let loss = sq
            .sum_keepdim(0)
            .unwrap()
            .sum_keepdim(1)
            .unwrap()
            .sum_keepdim(2)
            .unwrap();
        let grads = backward(&loss).unwrap();
        opt.step(&grads).unwrap();
    }
    let final_loss = loss_of(&var);
    assert!(
        final_loss < initial_loss * 0.5,
        "3D factored should converge: initial={initial_loss}, final={final_loss}"
    );
}

// ============================================================================
// 9. Multi-variable optimization
// ============================================================================

#[test]
fn test_adafactor_multi_variable_simultaneous() {
    // Optimize two variables simultaneously: a 1D and a 2D parameter.
    let var1 = make_var(&[5.0, -3.0], &[2]);
    let var2 = make_var(&[2.0, -1.0, 3.0, -2.0], &[2, 2]);
    let config = AdaFactorConfig {
        lr: 0.1,
        relative_step: false,
        ..Default::default()
    };
    let mut opt = AdaFactor::new(vec![var1.clone(), var2.clone()], config).unwrap();

    let initial_loss1 = loss_of(&var1);
    let initial_loss2 = loss_of(&var2);

    for _ in 0..20 {
        // Build combined loss: sum(var1^2) + sum(var2^2)
        let t1 = Arc::new(TrackedTensor::from_var(&var1).unwrap());
        let t2 = Arc::new(TrackedTensor::from_var(&var2).unwrap());

        let l1 = t1.sqr().unwrap().sum_keepdim(0).unwrap();
        let l2 = t2
            .sqr()
            .unwrap()
            .sum_keepdim(0)
            .unwrap()
            .sum_keepdim(1)
            .unwrap();
        let loss = l1.add(&l2).unwrap();
        let grads = backward(&loss).unwrap();
        opt.step(&grads).unwrap();
    }

    let final_loss1 = loss_of(&var1);
    let final_loss2 = loss_of(&var2);

    assert!(
        final_loss1 < initial_loss1,
        "var1 should converge: {final_loss1} >= {initial_loss1}"
    );
    assert!(
        final_loss2 < initial_loss2,
        "var2 should converge: {final_loss2} >= {initial_loss2}"
    );
}

// ============================================================================
// 10. Step count increments correctly across multiple steps
// ============================================================================

#[test]
fn test_adafactor_step_count_increments() {
    let var = make_var(&[1.0, 2.0], &[2]);
    let config = AdaFactorConfig {
        lr: 0.01,
        relative_step: false,
        ..Default::default()
    };
    let mut opt = AdaFactor::new(vec![var.clone()], config).unwrap();
    assert_eq!(opt.step_count(), 0);

    for expected in 1..=10 {
        let (_, _, grads) = quadratic_loss(&var);
        opt.step(&grads).unwrap();
        assert_eq!(opt.step_count(), expected);
    }
}

// ============================================================================
// 11. Config accessor
// ============================================================================

#[test]
fn test_adafactor_config_accessor() {
    let config = AdaFactorConfig {
        lr: 0.05,
        relative_step: true,
        eps_rms: 1e-2,
        eps_denom: 1e-20,
        decay_rate: -0.5,
        beta1: Some(0.95),
        weight_decay: 0.01,
    };
    let var = make_var(&[1.0], &[1]);
    let opt = AdaFactor::new(vec![var], config.clone()).unwrap();
    let retrieved = opt.config();
    assert_eq!(retrieved, &config);
}

// ============================================================================
// 12. Zero-gradient does not update parameters
// ============================================================================

#[test]
fn test_adafactor_zero_gradient_no_movement() {
    // If all gradients are zero, the parameter should not change (ignoring weight decay).
    let var = make_var(&[5.0, -3.0], &[2]);
    let config = AdaFactorConfig {
        lr: 0.1,
        relative_step: false,
        weight_decay: 0.0,
        ..Default::default()
    };
    let mut opt = AdaFactor::new(vec![var.clone()], config).unwrap();

    // Manually construct zero gradients.
    let t = Arc::new(TrackedTensor::from_var(&var).unwrap());
    let loss = t.sqr().unwrap().sum_keepdim(0).unwrap();
    let mut grads = backward(&loss).unwrap();
    let zero_grad = DynTensor::zeros(&[2], nn_core::DType::F32, &cpu()).unwrap();
    for (_, g) in grads.var_grads_mut() {
        *g = zero_grad.clone();
    }

    opt.step(&grads).unwrap();

    let vals = var.data().unwrap().to_flat_vec::<f32>().unwrap();
    assert!(
        (vals[0] - 5.0).abs() < 1e-7,
        "zero gradient should not move param, got {}",
        vals[0]
    );
    assert!(
        (vals[1] - (-3.0)).abs() < 1e-7,
        "zero gradient should not move param, got {}",
        vals[1]
    );
}

// ============================================================================
// 13. Convergence to known target on shifted quadratic
// ============================================================================

#[test]
fn test_adafactor_converges_toward_minimum() {
    // Minimize f(x) = sum((x - target)^2) where target = [1, -1, 0.5].
    // Gradient: 2*(x - target).
    let var = make_var(&[10.0, -10.0, 5.0], &[3]);
    let target = DynTensor::from_vec(vec![1.0, -1.0, 0.5], &[3], &cpu()).unwrap();
    let config = AdaFactorConfig {
        lr: 0.1,
        relative_step: false,
        ..Default::default()
    };
    let mut opt = AdaFactor::new(vec![var.clone()], config).unwrap();

    for _ in 0..50 {
        // loss = sum((x - target)^2)
        let t = Arc::new(TrackedTensor::from_var(&var).unwrap());
        let target_tracked = Arc::new(TrackedTensor::from_tensor(target.clone()));
        let diff = t.sub(&target_tracked).unwrap();
        let loss = diff.sqr().unwrap().sum_keepdim(0).unwrap();
        let grads = backward(&loss).unwrap();
        opt.step(&grads).unwrap();
    }

    let vals = var.data().unwrap().to_flat_vec::<f32>().unwrap();
    // After 50 steps, params should be closer to target than the initial values.
    let dist_sq: f64 = vals
        .iter()
        .zip([1.0f32, -1.0, 0.5].iter())
        .map(|(&v, &t)| f64::from(v - t).powi(2))
        .sum();
    // Initial distance: (10-1)^2 + (-10+1)^2 + (5-0.5)^2 = 81 + 81 + 20.25 = 182.25
    assert!(
        dist_sq < 50.0,
        "should converge toward target, distance^2={dist_sq}"
    );
}

// ============================================================================
// 14. decay_rate controls beta2 schedule shape
// ============================================================================

#[test]
fn test_adafactor_different_decay_rates() {
    // More negative decay_rate => beta2 increases faster (more memory of old gradients).
    // Both should converge, but the behavior is different.
    let init = vec![5.0, -5.0];

    let var_fast = make_var(&init, &[2]);
    let var_slow = make_var(&init, &[2]);

    let config_fast = AdaFactorConfig {
        lr: 0.1,
        relative_step: false,
        decay_rate: -0.8,
        ..Default::default()
    };
    let config_slow = AdaFactorConfig {
        lr: 0.1,
        relative_step: false,
        decay_rate: -0.3,
        ..Default::default()
    };

    let mut opt_fast = AdaFactor::new(vec![var_fast.clone()], config_fast).unwrap();
    let mut opt_slow = AdaFactor::new(vec![var_slow.clone()], config_slow).unwrap();

    let initial_loss = loss_of(&var_fast);

    for _ in 0..25 {
        let (_, _, g_fast) = quadratic_loss(&var_fast);
        opt_fast.step(&g_fast).unwrap();
        let (_, _, g_slow) = quadratic_loss(&var_slow);
        opt_slow.step(&g_slow).unwrap();
    }

    let loss_fast = loss_of(&var_fast);
    let loss_slow = loss_of(&var_slow);

    // Both should converge from initial ~50.
    assert!(
        loss_fast < initial_loss,
        "fast decay should converge: {loss_fast}"
    );
    assert!(
        loss_slow < initial_loss,
        "slow decay should converge: {loss_slow}"
    );
}

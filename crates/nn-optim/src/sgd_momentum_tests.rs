// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Comprehensive tests for SGD momentum, weight decay, Nesterov comparison,
//! learning rate scheduling, and multi-parameter-group behavior.
//!
//! Covers scenarios not exercised by sgd_tests.rs and optim_expanded_tests.rs:
//! - Manual computation verification for multi-element tensors
//! - Three-step momentum accumulation with exact velocity tracking
//! - Classical vs. Nesterov-like update divergence (proves classical is used)
//! - Weight decay magnitude reduction over multiple steps
//! - LR scheduling integration (warmup + cosine) with SGD
//! - Zero gradient with prior momentum (velocity drains)
//! - Multiple parameter groups at different learning rates
//! - Convergence on f(x) = x^2 from negative initial value

use std::sync::Arc;

use nn_autodiff::{backward, TrackedTensor, Var};
use nn_core::dyn_tensor::DynTensor;
use nn_core::test_utils::cpu;

use crate::lr_schedule::{step_with_schedule, CosineSchedule, LrSchedule, WarmupSchedule};
use crate::optimizer::Optimizer;
use crate::sgd::{Sgd, SgdConfig};

fn scalar_var(val: f32) -> Var {
    Var::new(DynTensor::from_vec(vec![val], &[1], &cpu()).unwrap())
}

fn vec_var(vals: &[f32]) -> Var {
    let n = vals.len();
    Var::new(DynTensor::from_vec(vals.to_vec(), &[n], &cpu()).unwrap())
}

/// Helper: compute gradient of sum(x^2) for a variable.
fn grad_sum_sq(var: &Var) -> nn_autodiff::GradStore {
    let t = Arc::new(TrackedTensor::from_var(var).unwrap());
    let sq = t.sqr().unwrap();
    let mut loss = sq;
    for d in 0..var.data().unwrap().dims().len() {
        loss = loss.sum_keepdim(d).unwrap();
    }
    backward(&loss).unwrap()
}

// ============================================================================
// 1. Basic SGD step matches manual computation (multi-element)
// ============================================================================

/// Verify SGD step on a 4-element vector matches hand-computed values.
/// loss = sum(x^2), grad = 2*x, x_new = x - lr * grad.
#[test]
fn test_sgd_step_manual_computation_vector() {
    let x = vec_var(&[1.0, -2.0, 3.0, -4.0]);
    let mut sgd = Sgd::new(
        vec![x.clone()],
        SgdConfig {
            lr: 0.05,
            momentum: 0.0,
            weight_decay: 0.0,
        },
    )
    .unwrap();

    // grad = 2 * [1, -2, 3, -4] = [2, -4, 6, -8]
    // x_new = [1, -2, 3, -4] - 0.05 * [2, -4, 6, -8]
    //       = [1 - 0.1, -2 + 0.2, 3 - 0.3, -4 + 0.4]
    //       = [0.9, -1.8, 2.7, -3.6]
    let grads = grad_sum_sq(&x);
    sgd.step(&grads).unwrap();

    let vals = x.data().unwrap().to_flat_vec::<f32>().unwrap();
    let expected = [0.9, -1.8, 2.7, -3.6];
    for (i, (&got, &exp)) in vals.iter().zip(expected.iter()).enumerate() {
        assert!(
            (got - exp).abs() < 1e-5,
            "x[{i}]: expected {exp}, got {got}"
        );
    }
}

/// Verify SGD step with weight decay matches manual computation.
/// grad_eff = grad + wd * theta, x_new = x - lr * grad_eff.
#[test]
fn test_sgd_step_manual_computation_with_weight_decay() {
    let x = scalar_var(4.0);
    let mut sgd = Sgd::new(
        vec![x.clone()],
        SgdConfig {
            lr: 0.1,
            momentum: 0.0,
            weight_decay: 0.05,
        },
    )
    .unwrap();

    // loss = x^2, grad = 2*x = 8.0
    // grad_eff = 8.0 + 0.05 * 4.0 = 8.2
    // x_new = 4.0 - 0.1 * 8.2 = 3.18
    let t = Arc::new(TrackedTensor::from_var(&x).unwrap());
    let loss = t.sqr().unwrap();
    sgd.backward_step(&loss).unwrap();

    let val = x.data().unwrap().to_flat_vec::<f32>().unwrap()[0];
    assert!((val - 3.18).abs() < 1e-4, "expected 3.18, got {val}");
}

// ============================================================================
// 2. Momentum accumulation across three steps with exact values
// ============================================================================

/// Track velocity accumulation across 3 steps with exact numerical verification.
/// v_t = momentum * v_{t-1} + grad_t
/// x_t = x_{t-1} - lr * v_t
#[test]
fn test_sgd_momentum_accumulation_three_steps() {
    let x = scalar_var(3.0);
    let lr = 0.1;
    let mom = 0.9;
    let mut sgd = Sgd::new(
        vec![x.clone()],
        SgdConfig {
            lr,
            momentum: mom,
            weight_decay: 0.0,
        },
    )
    .unwrap();

    // Step 1: x=3.0, grad=2*3=6.0
    //   v1 = 0.9 * 0 + 6.0 = 6.0 (first step, no prior velocity)
    //   x1 = 3.0 - 0.1 * 6.0 = 2.4
    let t = Arc::new(TrackedTensor::from_var(&x).unwrap());
    let loss = t.sqr().unwrap();
    sgd.backward_step(&loss).unwrap();
    let x1 = x.data().unwrap().to_flat_vec::<f32>().unwrap()[0];
    assert!((x1 - 2.4).abs() < 1e-5, "step 1: expected 2.4, got {x1}");

    // Step 2: x=2.4, grad=2*2.4=4.8
    //   v2 = 0.9 * 6.0 + 4.8 = 10.2
    //   x2 = 2.4 - 0.1 * 10.2 = 1.38
    let t = Arc::new(TrackedTensor::from_var(&x).unwrap());
    let loss = t.sqr().unwrap();
    sgd.backward_step(&loss).unwrap();
    let x2 = x.data().unwrap().to_flat_vec::<f32>().unwrap()[0];
    assert!((x2 - 1.38).abs() < 1e-4, "step 2: expected 1.38, got {x2}");

    // Step 3: x=1.38, grad=2*1.38=2.76
    //   v3 = 0.9 * 10.2 + 2.76 = 11.94
    //   x3 = 1.38 - 0.1 * 11.94 = 0.186
    let t = Arc::new(TrackedTensor::from_var(&x).unwrap());
    let loss = t.sqr().unwrap();
    sgd.backward_step(&loss).unwrap();
    let x3 = x.data().unwrap().to_flat_vec::<f32>().unwrap()[0];
    assert!(
        (x3 - 0.186).abs() < 0.05,
        "step 3: expected ~0.186, got {x3}"
    );
}

/// Momentum on a vector: verify all elements accumulate velocity independently.
#[test]
fn test_sgd_momentum_accumulation_vector() {
    let x = vec_var(&[2.0, -3.0]);
    let lr = 0.1;
    let mom = 0.5;
    let mut sgd = Sgd::new(
        vec![x.clone()],
        SgdConfig {
            lr,
            momentum: mom,
            weight_decay: 0.0,
        },
    )
    .unwrap();

    // Step 1: grad = [4.0, -6.0]
    //   v1 = [4.0, -6.0] (no prior)
    //   x1 = [2.0 - 0.4, -3.0 + 0.6] = [1.6, -2.4]
    let grads = grad_sum_sq(&x);
    sgd.step(&grads).unwrap();
    let vals1 = x.data().unwrap().to_flat_vec::<f32>().unwrap();
    assert!(
        (vals1[0] - 1.6).abs() < 1e-5,
        "step1 x[0]: expected 1.6, got {}",
        vals1[0]
    );
    assert!(
        (vals1[1] - (-2.4)).abs() < 1e-5,
        "step1 x[1]: expected -2.4, got {}",
        vals1[1]
    );

    // Step 2: grad = [2*1.6, 2*(-2.4)] = [3.2, -4.8]
    //   v2 = [0.5*4.0 + 3.2, 0.5*(-6.0) + (-4.8)] = [5.2, -7.8]
    //   x2 = [1.6 - 0.52, -2.4 + 0.78] = [1.08, -1.62]
    let grads = grad_sum_sq(&x);
    sgd.step(&grads).unwrap();
    let vals2 = x.data().unwrap().to_flat_vec::<f32>().unwrap();
    assert!(
        (vals2[0] - 1.08).abs() < 1e-4,
        "step2 x[0]: expected 1.08, got {}",
        vals2[0]
    );
    assert!(
        (vals2[1] - (-1.62)).abs() < 1e-4,
        "step2 x[1]: expected -1.62, got {}",
        vals2[1]
    );
}

// ============================================================================
// 3. Classical momentum is NOT Nesterov
// ============================================================================

/// Prove the SGD implementation uses classical momentum (v = m*v + g, x -= lr*v)
/// and NOT Nesterov (x -= lr*(m*v + g) with look-ahead). We do this by computing
/// both formulas manually and checking the implementation matches classical.
#[test]
fn test_sgd_uses_classical_not_nesterov_momentum() {
    let x_impl = scalar_var(5.0);
    let mut sgd = Sgd::new(
        vec![x_impl.clone()],
        SgdConfig {
            lr: 0.1,
            momentum: 0.9,
            weight_decay: 0.0,
        },
    )
    .unwrap();

    // Step 1: grad = 2*5 = 10.0
    let t = Arc::new(TrackedTensor::from_var(&x_impl).unwrap());
    let loss = t.sqr().unwrap();
    sgd.backward_step(&loss).unwrap();
    let _after_step1 = x_impl.data().unwrap().to_flat_vec::<f32>().unwrap()[0];

    // Step 2: grad = 2 * after_step1
    let t = Arc::new(TrackedTensor::from_var(&x_impl).unwrap());
    let loss = t.sqr().unwrap();
    sgd.backward_step(&loss).unwrap();
    let impl_result = x_impl.data().unwrap().to_flat_vec::<f32>().unwrap()[0];

    // Manual classical momentum computation:
    // Step 1: v1 = 10.0, x1 = 5.0 - 0.1 * 10.0 = 4.0
    let x1_classical = 4.0_f64;
    let v1 = 10.0_f64;
    // Step 2: grad2 = 2 * 4.0 = 8.0
    //   v2 = 0.9 * 10.0 + 8.0 = 17.0
    //   x2 = 4.0 - 0.1 * 17.0 = 2.3
    let grad2 = 2.0 * x1_classical;
    let v2_classical = 0.9 * v1 + grad2;
    let x2_classical = x1_classical - 0.1 * v2_classical;

    // Manual Nesterov momentum computation:
    // Step 2 Nesterov: v2 = 0.9 * 10.0 + 8.0 = 17.0 (same velocity)
    //   BUT update: x2 = 4.0 - 0.1 * (0.9 * 17.0 + 8.0) = 4.0 - 0.1 * 23.3 = 1.67
    let v2_nesterov = 0.9 * v1 + grad2;
    let x2_nesterov = x1_classical - 0.1 * (0.9 * v2_nesterov + grad2);

    // Verify implementation matches classical
    assert!(
        (f64::from(impl_result) - x2_classical).abs() < 1e-4,
        "implementation ({impl_result}) should match classical momentum ({x2_classical})"
    );

    // Verify classical and Nesterov give DIFFERENT results
    assert!(
        (x2_classical - x2_nesterov).abs() > 0.1,
        "classical ({x2_classical}) and Nesterov ({x2_nesterov}) should differ"
    );

    // Verify implementation does NOT match Nesterov
    assert!(
        (f64::from(impl_result) - x2_nesterov).abs() > 0.1,
        "implementation ({impl_result}) should NOT match Nesterov ({x2_nesterov})"
    );
}

// ============================================================================
// 4. Weight decay reduces parameter magnitude over multiple steps
// ============================================================================

/// Weight decay monotonically reduces magnitude on every step (with zero gradient).
#[test]
fn test_sgd_weight_decay_monotonic_reduction() {
    let x = scalar_var(10.0);
    let mut sgd = Sgd::new(
        vec![x.clone()],
        SgdConfig {
            lr: 0.1,
            weight_decay: 0.1,
            momentum: 0.0,
        },
    )
    .unwrap();

    let mut prev_mag = 10.0_f32;
    for step in 0..10 {
        // loss = x * 0 => grad = 0, but weight_decay adds wd * theta
        let t = Arc::new(TrackedTensor::from_var(&x).unwrap());
        let zero = Arc::new(TrackedTensor::from_tensor(
            DynTensor::from_vec(vec![0.0], &[1], &cpu()).unwrap(),
        ));
        let loss = t.mul(&zero).unwrap();
        sgd.backward_step(&loss).unwrap();

        let cur_mag = x.data().unwrap().to_flat_vec::<f32>().unwrap()[0].abs();
        assert!(
            cur_mag < prev_mag,
            "step {step}: magnitude should decrease ({cur_mag} >= {prev_mag})"
        );
        prev_mag = cur_mag;
    }

    // After 10 steps: x * (1 - lr * wd)^10 = 10 * (1 - 0.01)^10 = 10 * 0.904 ~ 9.04
    let final_val = x.data().unwrap().to_flat_vec::<f32>().unwrap()[0];
    let expected = 10.0 * (1.0 - 0.1 * 0.1_f64).powi(10);
    assert!(
        (f64::from(final_val) - expected).abs() < 0.1,
        "expected ~{expected:.3}, got {final_val}"
    );
}

/// Weight decay on a vector: verify all elements shrink independently.
#[test]
fn test_sgd_weight_decay_vector_shrinkage() {
    let x = vec_var(&[5.0, -8.0, 3.0]);
    let initial_mags: Vec<f32> = vec![5.0, 8.0, 3.0];
    let mut sgd = Sgd::new(
        vec![x.clone()],
        SgdConfig {
            lr: 0.2,
            weight_decay: 0.5,
            momentum: 0.0,
        },
    )
    .unwrap();

    // With zero gradients, weight decay alone: x_new = x - lr * (wd * x) = x * (1 - lr*wd)
    // decay_factor = 1 - 0.2 * 0.5 = 0.9
    let t = Arc::new(TrackedTensor::from_var(&x).unwrap());
    let zero = Arc::new(TrackedTensor::from_tensor(
        DynTensor::from_vec(vec![0.0, 0.0, 0.0], &[3], &cpu()).unwrap(),
    ));
    let loss = t.mul(&zero).unwrap().sum_keepdim(0).unwrap();
    sgd.backward_step(&loss).unwrap();

    let vals = x.data().unwrap().to_flat_vec::<f32>().unwrap();
    let decay_factor = 0.9_f32;
    let expected = [5.0 * decay_factor, -8.0 * decay_factor, 3.0 * decay_factor];
    for (i, (&got, &exp)) in vals.iter().zip(expected.iter()).enumerate() {
        assert!(
            got.abs() < initial_mags[i],
            "x[{i}]: magnitude should decrease"
        );
        assert!(
            (got - exp).abs() < 1e-4,
            "x[{i}]: expected {exp}, got {got}"
        );
    }
}

// ============================================================================
// 5. Learning rate scheduling with SGD
// ============================================================================

/// Warmup schedule with SGD: verify LR ramps up linearly.
#[test]
fn test_sgd_warmup_schedule_integration() {
    let x = scalar_var(5.0);
    let mut sgd = Sgd::new(
        vec![x.clone()],
        SgdConfig {
            lr: 0.0,
            momentum: 0.0,
            weight_decay: 0.0,
        },
    )
    .unwrap();
    let schedule = WarmupSchedule::new(0.1, 10).unwrap();

    // Step 0: lr = 0.0, no update
    let grads = grad_sum_sq(&x);
    step_with_schedule(&mut sgd, &grads, &schedule, 0).unwrap();
    assert!(
        (sgd.learning_rate() - 0.0).abs() < f64::EPSILON,
        "step 0: lr should be 0"
    );
    let val0 = x.data().unwrap().to_flat_vec::<f32>().unwrap()[0];
    assert!((val0 - 5.0).abs() < 1e-6, "step 0: no update with lr=0");

    // Step 5: lr = 0.1 * 5/10 = 0.05
    let grads = grad_sum_sq(&x);
    step_with_schedule(&mut sgd, &grads, &schedule, 5).unwrap();
    assert!(
        (sgd.learning_rate() - 0.05).abs() < f64::EPSILON,
        "step 5: lr should be 0.05"
    );
    let val5 = x.data().unwrap().to_flat_vec::<f32>().unwrap()[0];
    assert!(
        val5 < 5.0,
        "step 5: x should decrease with lr=0.05, got {val5}"
    );

    // Step 10: lr = 0.1 (full)
    let grads = grad_sum_sq(&x);
    step_with_schedule(&mut sgd, &grads, &schedule, 10).unwrap();
    assert!(
        (sgd.learning_rate() - 0.1).abs() < f64::EPSILON,
        "step 10: lr should be 0.1"
    );
}

/// Cosine schedule with SGD: verify LR decays and training converges.
#[test]
fn test_sgd_cosine_schedule_convergence() {
    let x = scalar_var(5.0);
    let mut sgd = Sgd::new(
        vec![x.clone()],
        SgdConfig {
            lr: 0.0,
            momentum: 0.9,
            weight_decay: 0.0,
        },
    )
    .unwrap();
    let schedule = CosineSchedule::new(0.1, 0.001, 0, 100).unwrap();

    for step in 0..100 {
        let t = Arc::new(TrackedTensor::from_var(&x).unwrap());
        let loss = t.sqr().unwrap();
        let grads = backward(&loss).unwrap();
        step_with_schedule(&mut sgd, &grads, &schedule, step).unwrap();
    }

    let final_val = x.data().unwrap().to_flat_vec::<f32>().unwrap()[0];
    assert!(
        final_val.abs() < 1.0,
        "cosine+SGD should converge to near 0, got {final_val}"
    );

    // At step 99, lr should be near min_lr
    let final_lr = schedule.lr_at_step(99);
    assert!(
        final_lr < 0.01,
        "cosine lr at step 99 should be small, got {final_lr}"
    );
}

/// Manual set_learning_rate mid-training simulates a step schedule.
#[test]
fn test_sgd_manual_lr_step_schedule() {
    let x = scalar_var(5.0);
    let mut sgd = Sgd::new(
        vec![x.clone()],
        SgdConfig {
            lr: 0.1,
            momentum: 0.0,
            weight_decay: 0.0,
        },
    )
    .unwrap();

    // Phase 1: 5 steps at lr=0.1
    for _ in 0..5 {
        let t = Arc::new(TrackedTensor::from_var(&x).unwrap());
        let loss = t.sqr().unwrap();
        sgd.backward_step(&loss).unwrap();
    }
    let val_phase1 = x.data().unwrap().to_flat_vec::<f32>().unwrap()[0];

    // Drop LR by 10x
    sgd.set_learning_rate(0.01).unwrap();
    assert!((sgd.learning_rate() - 0.01).abs() < f64::EPSILON);

    // Phase 2: 5 more steps at lr=0.01
    let val_before_phase2 = val_phase1;
    for _ in 0..5 {
        let t = Arc::new(TrackedTensor::from_var(&x).unwrap());
        let loss = t.sqr().unwrap();
        sgd.backward_step(&loss).unwrap();
    }
    let val_phase2 = x.data().unwrap().to_flat_vec::<f32>().unwrap()[0];

    // Phase 2 updates should be smaller (lower LR)
    let delta_phase1 = (5.0 - val_phase1).abs();
    let delta_phase2 = (val_before_phase2 - val_phase2).abs();
    assert!(
        delta_phase2 < delta_phase1,
        "phase2 (lr=0.01) update {delta_phase2} should be smaller than phase1 (lr=0.1) {delta_phase1}"
    );
}

// ============================================================================
// 6. Zero gradient with and without momentum
// ============================================================================

/// Zero gradient with no momentum: parameter unchanged.
#[test]
fn test_sgd_zero_gradient_no_momentum_no_update() {
    let x = scalar_var(7.0);
    let mut sgd = Sgd::new(
        vec![x.clone()],
        SgdConfig {
            lr: 0.5,
            momentum: 0.0,
            weight_decay: 0.0,
        },
    )
    .unwrap();

    let t = Arc::new(TrackedTensor::from_var(&x).unwrap());
    let zero = Arc::new(TrackedTensor::from_tensor(
        DynTensor::from_vec(vec![0.0], &[1], &cpu()).unwrap(),
    ));
    let loss = t.mul(&zero).unwrap();
    sgd.backward_step(&loss).unwrap();

    let val = x.data().unwrap().to_flat_vec::<f32>().unwrap()[0];
    assert!(
        (val - 7.0).abs() < 1e-7,
        "zero grad, no momentum: x should be unchanged, got {val}"
    );
}

/// Zero gradient WITH prior momentum: velocity should drain via momentum decay.
/// After building velocity with a real gradient, then applying zero gradient,
/// the momentum-weighted velocity still causes an update.
#[test]
fn test_sgd_zero_gradient_with_prior_momentum_drains() {
    let x = scalar_var(5.0);
    let mut sgd = Sgd::new(
        vec![x.clone()],
        SgdConfig {
            lr: 0.1,
            momentum: 0.9,
            weight_decay: 0.0,
        },
    )
    .unwrap();

    // Step 1: build up velocity with real gradient
    // grad = 2*5 = 10, v1 = 10, x1 = 5 - 1.0 = 4.0
    let t = Arc::new(TrackedTensor::from_var(&x).unwrap());
    let loss = t.sqr().unwrap();
    sgd.backward_step(&loss).unwrap();
    let x_after_real = x.data().unwrap().to_flat_vec::<f32>().unwrap()[0];

    // Step 2: zero gradient - but velocity v2 = 0.9*10 + 0 = 9.0
    // x2 = x1 - 0.1 * 9.0 = x1 - 0.9
    let t = Arc::new(TrackedTensor::from_var(&x).unwrap());
    let zero = Arc::new(TrackedTensor::from_tensor(
        DynTensor::from_vec(vec![0.0], &[1], &cpu()).unwrap(),
    ));
    let loss = t.mul(&zero).unwrap();
    sgd.backward_step(&loss).unwrap();
    let x_after_zero = x.data().unwrap().to_flat_vec::<f32>().unwrap()[0];

    // Momentum should cause update even with zero gradient
    assert!(
        x_after_zero < x_after_real,
        "momentum should cause update even with zero grad: before={x_after_real}, after={x_after_zero}"
    );

    // Verify the exact value: v2 = 0.9 * 10.0 + 0.0 = 9.0
    // x2 = x1 - 0.1 * 9.0
    let expected = x_after_real - 0.1 * 9.0;
    assert!(
        (x_after_zero - expected).abs() < 1e-4,
        "expected {expected}, got {x_after_zero}"
    );
}

/// After many zero-gradient steps, momentum velocity drains to near zero.
#[test]
fn test_sgd_momentum_velocity_drains_to_zero() {
    let x = scalar_var(5.0);
    let mut sgd = Sgd::new(
        vec![x.clone()],
        SgdConfig {
            lr: 0.1,
            momentum: 0.9,
            weight_decay: 0.0,
        },
    )
    .unwrap();

    // Build velocity with a real gradient step
    let t = Arc::new(TrackedTensor::from_var(&x).unwrap());
    let loss = t.sqr().unwrap();
    sgd.backward_step(&loss).unwrap();

    // Now run 50 zero-gradient steps to drain velocity
    for _ in 0..50 {
        let t = Arc::new(TrackedTensor::from_var(&x).unwrap());
        let zero = Arc::new(TrackedTensor::from_tensor(
            DynTensor::from_vec(vec![0.0], &[1], &cpu()).unwrap(),
        ));
        let loss = t.mul(&zero).unwrap();
        sgd.backward_step(&loss).unwrap();
    }

    // After 50 steps: v = 10.0 * 0.9^50 ≈ 0.005
    // The parameter change from one more step should be negligible
    let val_before = x.data().unwrap().to_flat_vec::<f32>().unwrap()[0];
    let t = Arc::new(TrackedTensor::from_var(&x).unwrap());
    let zero = Arc::new(TrackedTensor::from_tensor(
        DynTensor::from_vec(vec![0.0], &[1], &cpu()).unwrap(),
    ));
    let loss = t.mul(&zero).unwrap();
    sgd.backward_step(&loss).unwrap();
    let val_after = x.data().unwrap().to_flat_vec::<f32>().unwrap()[0];

    assert!(
        (val_after - val_before).abs() < 0.01,
        "velocity should drain to near zero: delta={}",
        (val_after - val_before).abs()
    );
}

// ============================================================================
// 7. Multiple parameter groups with different learning rates
// ============================================================================

/// Simulate two parameter groups at different learning rates using two SGD
/// instances. Verify the high-LR group updates faster.
#[test]
fn test_sgd_two_param_groups_different_lr() {
    let x_fast = scalar_var(5.0);
    let x_slow = scalar_var(5.0);

    let mut sgd_fast = Sgd::new(
        vec![x_fast.clone()],
        SgdConfig {
            lr: 0.1,
            momentum: 0.0,
            weight_decay: 0.0,
        },
    )
    .unwrap();

    let mut sgd_slow = Sgd::new(
        vec![x_slow.clone()],
        SgdConfig {
            lr: 0.001,
            momentum: 0.0,
            weight_decay: 0.0,
        },
    )
    .unwrap();

    for _ in 0..20 {
        // Fast group
        let t = Arc::new(TrackedTensor::from_var(&x_fast).unwrap());
        let loss = t.sqr().unwrap();
        sgd_fast.backward_step(&loss).unwrap();

        // Slow group
        let t = Arc::new(TrackedTensor::from_var(&x_slow).unwrap());
        let loss = t.sqr().unwrap();
        sgd_slow.backward_step(&loss).unwrap();
    }

    let val_fast = x_fast.data().unwrap().to_flat_vec::<f32>().unwrap()[0].abs();
    let val_slow = x_slow.data().unwrap().to_flat_vec::<f32>().unwrap()[0].abs();

    assert!(
        val_fast < val_slow,
        "fast group (lr=0.1) should converge more: fast={val_fast}, slow={val_slow}"
    );
    assert!(
        val_fast < 1.0,
        "fast group should be near 0, got {val_fast}"
    );
    assert!(
        val_slow > 2.0,
        "slow group should still be far from 0, got {val_slow}"
    );
}

/// Multiple parameter groups where one uses momentum and the other does not.
#[test]
fn test_sgd_two_param_groups_momentum_vs_no_momentum() {
    let x_mom = scalar_var(5.0);
    let x_plain = scalar_var(5.0);

    let mut sgd_mom = Sgd::new(
        vec![x_mom.clone()],
        SgdConfig {
            lr: 0.01,
            momentum: 0.9,
            weight_decay: 0.0,
        },
    )
    .unwrap();

    let mut sgd_plain = Sgd::new(
        vec![x_plain.clone()],
        SgdConfig {
            lr: 0.01,
            momentum: 0.0,
            weight_decay: 0.0,
        },
    )
    .unwrap();

    for _ in 0..50 {
        let t = Arc::new(TrackedTensor::from_var(&x_mom).unwrap());
        let loss = t.sqr().unwrap();
        sgd_mom.backward_step(&loss).unwrap();

        let t = Arc::new(TrackedTensor::from_var(&x_plain).unwrap());
        let loss = t.sqr().unwrap();
        sgd_plain.backward_step(&loss).unwrap();
    }

    let val_mom = x_mom.data().unwrap().to_flat_vec::<f32>().unwrap()[0].abs();
    let val_plain = x_plain.data().unwrap().to_flat_vec::<f32>().unwrap()[0].abs();

    assert!(
        val_mom < val_plain,
        "momentum group should converge faster: mom={val_mom}, plain={val_plain}"
    );
}

// ============================================================================
// 8. Convergence on f(x) = x^2, minimum at x=0
// ============================================================================

/// Convergence from negative initial value: f(x) = x^2, x0 = -7.
#[test]
fn test_sgd_convergence_quadratic_negative_initial() {
    let x = scalar_var(-7.0);
    let mut sgd = Sgd::new(
        vec![x.clone()],
        SgdConfig {
            lr: 0.05,
            momentum: 0.0,
            weight_decay: 0.0,
        },
    )
    .unwrap();

    for _ in 0..100 {
        let t = Arc::new(TrackedTensor::from_var(&x).unwrap());
        let loss = t.sqr().unwrap();
        sgd.backward_step(&loss).unwrap();
    }

    let val = x.data().unwrap().to_flat_vec::<f32>().unwrap()[0];
    assert!(
        val.abs() < 0.01,
        "SGD should converge to 0 from negative start, got {val}"
    );
}

/// Convergence with momentum from negative initial value.
#[test]
fn test_sgd_convergence_quadratic_with_momentum() {
    let x = scalar_var(-7.0);
    let mut sgd = Sgd::new(
        vec![x.clone()],
        SgdConfig {
            lr: 0.05,
            momentum: 0.9,
            weight_decay: 0.0,
        },
    )
    .unwrap();

    for _ in 0..100 {
        let t = Arc::new(TrackedTensor::from_var(&x).unwrap());
        let loss = t.sqr().unwrap();
        sgd.backward_step(&loss).unwrap();
    }

    let val = x.data().unwrap().to_flat_vec::<f32>().unwrap()[0];
    assert!(
        val.abs() < 0.05,
        "SGD+momentum should converge to 0 from -7, got {val}"
    );
}

/// Convergence on multi-dimensional quadratic: f(x) = sum(x_i^2), minimum at origin.
#[test]
fn test_sgd_convergence_quadratic_multidim_with_momentum() {
    let x = vec_var(&[3.0, -5.0, 7.0, -2.0, 4.0]);
    let mut sgd = Sgd::new(
        vec![x.clone()],
        SgdConfig {
            lr: 0.05,
            momentum: 0.9,
            weight_decay: 0.0,
        },
    )
    .unwrap();

    for _ in 0..200 {
        let grads = grad_sum_sq(&x);
        sgd.step(&grads).unwrap();
    }

    let vals = x.data().unwrap().to_flat_vec::<f32>().unwrap();
    for (i, &v) in vals.iter().enumerate() {
        assert!(
            v.abs() < 0.01,
            "SGD 5D convergence: x[{i}] = {v}, expected near 0"
        );
    }
}

/// Convergence with all three features: momentum + weight decay + schedule.
#[test]
fn test_sgd_convergence_full_featured() {
    let x = scalar_var(10.0);
    let mut sgd = Sgd::new(
        vec![x.clone()],
        SgdConfig {
            lr: 0.0,
            momentum: 0.9,
            weight_decay: 0.001,
        },
    )
    .unwrap();
    let schedule = WarmupSchedule::new(0.1, 10).unwrap();

    for step in 0..200 {
        let t = Arc::new(TrackedTensor::from_var(&x).unwrap());
        let loss = t.sqr().unwrap();
        let grads = backward(&loss).unwrap();
        step_with_schedule(&mut sgd, &grads, &schedule, step).unwrap();
    }

    let val = x.data().unwrap().to_flat_vec::<f32>().unwrap()[0];
    assert!(
        val.abs() < 0.5,
        "full-featured SGD should converge, got {val}"
    );
}

// ============================================================================
// Edge cases
// ============================================================================

/// High momentum (0.99) with small LR: should still converge but take longer.
#[test]
fn test_sgd_high_momentum_convergence() {
    let x = scalar_var(5.0);
    let mut sgd = Sgd::new(
        vec![x.clone()],
        SgdConfig {
            lr: 0.001,
            momentum: 0.99,
            weight_decay: 0.0,
        },
    )
    .unwrap();

    for _ in 0..500 {
        let t = Arc::new(TrackedTensor::from_var(&x).unwrap());
        let loss = t.sqr().unwrap();
        sgd.backward_step(&loss).unwrap();
    }

    let val = x.data().unwrap().to_flat_vec::<f32>().unwrap()[0];
    assert!(
        val.abs() < 1.0,
        "high momentum SGD should converge (maybe slowly), got {val}"
    );
    assert!(val.is_finite(), "should not overflow");
}

/// Momentum=0 should behave identically to vanilla gradient descent.
/// Compare momentum=0 SGD to manual gradient descent.
#[test]
fn test_sgd_momentum_zero_equals_vanilla_gd() {
    let x_sgd = scalar_var(4.0);
    let x_manual = scalar_var(4.0);
    let lr = 0.1_f64;

    let mut sgd = Sgd::new(
        vec![x_sgd.clone()],
        SgdConfig {
            lr,
            momentum: 0.0,
            weight_decay: 0.0,
        },
    )
    .unwrap();

    for _ in 0..10 {
        // SGD step
        let t = Arc::new(TrackedTensor::from_var(&x_sgd).unwrap());
        let loss = t.sqr().unwrap();
        sgd.backward_step(&loss).unwrap();

        // Manual GD step: x = x - lr * 2x = x * (1 - 2*lr)
        let cur = x_manual.data().unwrap().to_flat_vec::<f32>().unwrap()[0];
        let new_val = cur - (lr as f32) * 2.0 * cur;
        x_manual
            .set(&DynTensor::from_vec(vec![new_val], &[1], &cpu()).unwrap())
            .unwrap();
    }

    let val_sgd = x_sgd.data().unwrap().to_flat_vec::<f32>().unwrap()[0];
    let val_manual = x_manual.data().unwrap().to_flat_vec::<f32>().unwrap()[0];
    assert!(
        (val_sgd - val_manual).abs() < 1e-4,
        "SGD(momentum=0) should match manual GD: sgd={val_sgd}, manual={val_manual}"
    );
}

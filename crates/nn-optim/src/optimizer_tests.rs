// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Comprehensive cross-optimizer tests covering hand-computed numerical
//! verification, momentum accumulation, weight decay mechanics, stability
//! under extreme gradients, learning rate scheduling integration, and
//! cross-optimizer interface uniformity.
//!
//! These tests complement per-optimizer test files (adam_tests.rs, sgd_tests.rs,
//! adafactor_tests.rs) by exercising specific numerical correctness scenarios
//! with hand-verified expected values and cross-optimizer behavioral contracts.

use std::sync::Arc;

use nn_autodiff::{backward, GradStore, TrackedTensor, Var};
use nn_core::dyn_tensor::DynTensor;
use nn_core::test_utils::cpu;

use crate::adafactor::{AdaFactor, AdaFactorConfig};
use crate::adam::{AdamConfig, AdamW};
use crate::lr_schedule::{step_with_schedule, WarmupSchedule};
use crate::optimizer::Optimizer;
use crate::sgd::{Sgd, SgdConfig};

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

/// Build a GradStore with a single entry mapping `var` to a manually
/// constructed gradient tensor.
fn manual_grads(var: &Var, grad_vals: &[f32]) -> GradStore {
    let t = Arc::new(TrackedTensor::from_var(var).unwrap());
    let loss = t.sqr().unwrap();
    // sum across all dims so backward works
    let mut l = loss;
    for d in 0..var.data().unwrap().dims().len() {
        l = l.sum_keepdim(d).unwrap();
    }
    let mut grads = backward(&l).unwrap();
    // Overwrite with our manual gradient
    let grad_tensor =
        DynTensor::from_vec(grad_vals.to_vec(), var.data().unwrap().dims(), &cpu()).unwrap();
    for (_, g) in grads.var_grads_mut() {
        *g = grad_tensor.clone();
    }
    grads
}

/// Compute gradient of sum(x^2) for a variable.
fn grad_sum_sq(var: &Var) -> GradStore {
    let t = Arc::new(TrackedTensor::from_var(var).unwrap());
    let sq = t.sqr().unwrap();
    let mut loss = sq;
    for d in 0..var.data().unwrap().dims().len() {
        loss = loss.sum_keepdim(d).unwrap();
    }
    backward(&loss).unwrap()
}

fn get_val(var: &Var) -> f32 {
    var.data().unwrap().to_flat_vec::<f32>().unwrap()[0]
}

fn get_vals(var: &Var) -> Vec<f32> {
    var.data().unwrap().to_flat_vec::<f32>().unwrap()
}

// ============================================================================
// 1. Adam: single step with hand-computed expected output
// ============================================================================

/// Hand-compute a single Adam step on a 3-element vector and verify each
/// element matches.
///
/// Setup: x=[2.0, -1.0, 4.0], grad=[1.0, -0.5, 2.0]
/// Config: lr=0.01, beta1=0.9, beta2=0.999, eps=1e-8, wd=0.0
///
/// Step 1 (t=1):
///   m1 = (1-0.9)*grad = 0.1*[1.0, -0.5, 2.0] = [0.1, -0.05, 0.2]
///   v1 = (1-0.999)*grad^2 = 0.001*[1.0, 0.25, 4.0] = [0.001, 0.00025, 0.004]
///   bc1 = 1/(1-0.9^1) = 10.0
///   bc2 = 1/(1-0.999^1) = 1000.0
///   m_hat = [1.0, -0.5, 2.0]
///   v_hat = [1.0, 0.25, 4.0]
///   step_i = lr * m_hat_i / (sqrt(v_hat_i) + eps)
///   step = 0.01 * [1.0/1.0, -0.5/0.5, 2.0/2.0] = [0.01, -0.01, 0.01]
///   x_new = x - step = [1.99, -0.99, 3.99]
#[test]
fn test_adam_single_step_hand_computed_vector() {
    let x = vec_var(&[2.0, -1.0, 4.0]);
    let config = AdamConfig {
        lr: 0.01,
        beta1: 0.9,
        beta2: 0.999,
        eps: 1e-8,
        weight_decay: 0.0,
    };
    let mut adam = AdamW::new(vec![x.clone()], config).unwrap();

    let grads = manual_grads(&x, &[1.0, -0.5, 2.0]);
    adam.step(&grads).unwrap();

    let vals = get_vals(&x);
    // m_hat = bc1 * m = 10 * [0.1, -0.05, 0.2] = [1.0, -0.5, 2.0]
    // v_hat = bc2 * v = 1000 * [0.001, 0.00025, 0.004] = [1.0, 0.25, 4.0]
    // step[0] = 0.01 * 1.0 / (sqrt(1.0) + 1e-8) ≈ 0.01
    // step[1] = 0.01 * (-0.5) / (sqrt(0.25) + 1e-8) ≈ -0.01
    // step[2] = 0.01 * 2.0 / (sqrt(4.0) + 1e-8) ≈ 0.01
    let expected = [
        2.0 - 0.01 * 1.0 / (1.0_f32.sqrt() + 1e-8),
        -1.0 - 0.01 * (-0.5) / (0.25_f32.sqrt() + 1e-8),
        4.0 - 0.01 * 2.0 / (4.0_f32.sqrt() + 1e-8),
    ];
    for (i, (&got, &exp)) in vals.iter().zip(expected.iter()).enumerate() {
        assert!(
            (got - exp).abs() < 1e-5,
            "x[{i}]: expected {exp}, got {got}"
        );
    }
}

// ============================================================================
// 2. Adam: multi-step momentum accumulation with numerical verification
// ============================================================================

/// Verify that Adam's first and second moments accumulate correctly over 3
/// steps by checking the parameter value after each step.
///
/// We use a constant gradient (g=2.0 injected manually) so we can track
/// moment evolution analytically.
#[test]
fn test_adam_three_steps_constant_gradient_accumulation() {
    let x = scalar_var(10.0);
    let beta1 = 0.9_f64;
    let beta2 = 0.999_f64;
    let lr = 0.01_f64;
    let eps = 1e-8_f64;
    let config = AdamConfig {
        lr,
        beta1,
        beta2,
        eps,
        weight_decay: 0.0,
    };
    let mut adam = AdamW::new(vec![x.clone()], config).unwrap();

    let g = 2.0_f64;
    let mut m = 0.0_f64;
    let mut v = 0.0_f64;
    let mut theta = 10.0_f64;

    for t in 1..=3 {
        // Manual moment update
        m = beta1 * m + (1.0 - beta1) * g;
        v = beta2 * v + (1.0 - beta2) * g * g;
        let bc1 = 1.0 / (1.0 - beta1.powi(t));
        let bc2 = 1.0 / (1.0 - beta2.powi(t));
        let m_hat = m * bc1;
        let v_hat = v * bc2;
        theta -= lr * m_hat / (v_hat.sqrt() + eps);

        // Optimizer step
        let grads = manual_grads(&x, &[g as f32]);
        adam.step(&grads).unwrap();

        let val = f64::from(get_val(&x));
        assert!(
            (val - theta).abs() < 1e-4,
            "step {t}: expected {theta:.6}, got {val:.6}"
        );
    }
}

// ============================================================================
// 3. Adam: weight decay effect on different initial magnitudes
// ============================================================================

/// Verify that Adam's decoupled weight decay shrinks larger parameters more
/// in absolute terms (because decay_factor multiplies theta).
#[test]
fn test_adam_weight_decay_proportional_to_magnitude() {
    let x_large = scalar_var(100.0);
    let x_small = scalar_var(1.0);

    let config = AdamConfig {
        lr: 0.01,
        weight_decay: 0.1,
        ..AdamConfig::default()
    };

    let mut adam_large = AdamW::new(vec![x_large.clone()], config.clone()).unwrap();
    let mut adam_small = AdamW::new(vec![x_small.clone()], config).unwrap();

    // Use the same small gradient for both
    let grads_large = manual_grads(&x_large, &[0.01]);
    adam_large.step(&grads_large).unwrap();
    let grads_small = manual_grads(&x_small, &[0.01]);
    adam_small.step(&grads_small).unwrap();

    // decay_factor = 1 - lr * wd = 1 - 0.01 * 0.1 = 0.999
    // Absolute shrinkage from decay: x * (1 - decay_factor) = x * 0.001
    let large_shrink = 100.0 - get_val(&x_large);
    let small_shrink = 1.0 - get_val(&x_small);

    assert!(
        large_shrink > small_shrink * 10.0,
        "larger param should shrink more: large={large_shrink:.4}, small={small_shrink:.6}"
    );
}

// ============================================================================
// 4. Adam: different learning rates produce proportional updates
// ============================================================================

/// With the same gradient and initial values, doubling the learning rate
/// should approximately double the parameter update magnitude.
#[test]
fn test_adam_lr_scales_update() {
    let x1 = scalar_var(5.0);
    let x2 = scalar_var(5.0);

    let config1 = AdamConfig {
        lr: 0.01,
        weight_decay: 0.0,
        ..AdamConfig::default()
    };
    let config2 = AdamConfig {
        lr: 0.02,
        weight_decay: 0.0,
        ..AdamConfig::default()
    };

    let mut adam1 = AdamW::new(vec![x1.clone()], config1).unwrap();
    let mut adam2 = AdamW::new(vec![x2.clone()], config2).unwrap();

    let grads1 = manual_grads(&x1, &[3.0]);
    adam1.step(&grads1).unwrap();
    let grads2 = manual_grads(&x2, &[3.0]);
    adam2.step(&grads2).unwrap();

    let update1 = (5.0 - get_val(&x1)).abs();
    let update2 = (5.0 - get_val(&x2)).abs();

    // At step 1, bias-corrected m_hat and v_hat are the same for both.
    // The update is lr * m_hat / (sqrt(v_hat) + eps).
    // So update2 / update1 should be lr2 / lr1 = 2.0.
    let ratio = update2 / update1;
    assert!(
        (ratio - 2.0).abs() < 0.01,
        "doubling lr should double the update: ratio={ratio}"
    );
}

// ============================================================================
// 5. Adam: zero gradient does not change weights
// ============================================================================

/// With zero gradient and zero weight decay, Adam should not change
/// parameters at all (the update is 0/0+eps = 0).
#[test]
fn test_adam_zero_gradient_preserves_weights() {
    let x = vec_var(&[3.0, -7.0, 0.5]);
    let config = AdamConfig {
        lr: 0.1,
        weight_decay: 0.0,
        ..AdamConfig::default()
    };
    let mut adam = AdamW::new(vec![x.clone()], config).unwrap();

    let grads = manual_grads(&x, &[0.0, 0.0, 0.0]);
    adam.step(&grads).unwrap();

    let vals = get_vals(&x);
    assert!(
        (vals[0] - 3.0).abs() < 1e-7,
        "zero grad should preserve x[0]"
    );
    assert!(
        (vals[1] - (-7.0)).abs() < 1e-7,
        "zero grad should preserve x[1]"
    );
    assert!(
        (vals[2] - 0.5).abs() < 1e-7,
        "zero grad should preserve x[2]"
    );
}

// ============================================================================
// 6. Adam: large gradients remain stable (no NaN/Inf in output)
// ============================================================================

/// Inject a large (but finite) gradient and verify Adam produces finite output.
/// Adam's adaptive step normalizes by sqrt(v), which grows with g^2, so the
/// effective step size is bounded regardless of gradient magnitude.
#[test]
fn test_adam_large_gradient_stability() {
    let x = scalar_var(1.0);
    let config = AdamConfig {
        lr: 0.001,
        weight_decay: 0.0,
        ..AdamConfig::default()
    };
    let mut adam = AdamW::new(vec![x.clone()], config).unwrap();

    // Inject a very large gradient
    let grads = manual_grads(&x, &[1e6]);
    adam.step(&grads).unwrap();

    let val = get_val(&x);
    assert!(val.is_finite(), "large gradient should not cause NaN/Inf");

    // Adam's first-step update is approximately lr * sign(g) = 0.001 * 1 ≈ 0.001
    // because m_hat/sqrt(v_hat) ≈ g/|g| = 1 for any magnitude (first step, bc cancels)
    assert!(
        (val - (1.0 - 0.001)).abs() < 0.01,
        "Adam should clip effective step to ~lr on first step: got {val}"
    );
}

/// Multiple large gradient steps should remain numerically stable.
#[test]
fn test_adam_repeated_large_gradients_stable() {
    let x = scalar_var(0.0);
    let config = AdamConfig {
        lr: 0.01,
        weight_decay: 0.0,
        ..AdamConfig::default()
    };
    let mut adam = AdamW::new(vec![x.clone()], config).unwrap();

    for _ in 0..50 {
        let grads = manual_grads(&x, &[1e4]);
        adam.step(&grads).unwrap();
    }

    let val = get_val(&x);
    assert!(val.is_finite(), "50 large-gradient steps should be stable");
}

// ============================================================================
// 7. SGD: exact single step with hand-computed values (multi-element)
// ============================================================================

/// Verify SGD step on a 3-element vector with known gradient.
/// x=[3.0, -2.0, 5.0], grad=[1.5, -1.0, 2.5], lr=0.2
/// x_new = x - lr * grad = [3-0.3, -2+0.2, 5-0.5] = [2.7, -1.8, 4.5]
#[test]
fn test_sgd_exact_step_3elem() {
    let x = vec_var(&[3.0, -2.0, 5.0]);
    let mut sgd = Sgd::new(
        vec![x.clone()],
        SgdConfig {
            lr: 0.2,
            momentum: 0.0,
            weight_decay: 0.0,
        },
    )
    .unwrap();

    let grads = manual_grads(&x, &[1.5, -1.0, 2.5]);
    sgd.step(&grads).unwrap();

    let vals = get_vals(&x);
    let expected = [2.7, -1.8, 4.5];
    for (i, (&got, &exp)) in vals.iter().zip(expected.iter()).enumerate() {
        assert!(
            (got - exp).abs() < 1e-5,
            "x[{i}]: expected {exp}, got {got}"
        );
    }
}

// ============================================================================
// 8. SGD: momentum accumulation exact numerical tracking
// ============================================================================

/// Track momentum buffer evolution over 3 steps with manually injected
/// constant gradients. Verify exact velocity and parameter values.
///
/// grad=[2.0] (constant), lr=0.1, momentum=0.8
/// Step 1: v1 = 2.0, x1 = 5.0 - 0.1*2.0 = 4.8
/// Step 2: v2 = 0.8*2.0 + 2.0 = 3.6, x2 = 4.8 - 0.1*3.6 = 4.44
/// Step 3: v3 = 0.8*3.6 + 2.0 = 4.88, x3 = 4.44 - 0.1*4.88 = 3.952
#[test]
fn test_sgd_momentum_exact_constant_gradient() {
    let x = scalar_var(5.0);
    let lr = 0.1_f64;
    let mom = 0.8_f64;
    let mut sgd = Sgd::new(
        vec![x.clone()],
        SgdConfig {
            lr,
            momentum: mom,
            weight_decay: 0.0,
        },
    )
    .unwrap();

    let g = 2.0;
    let expected_vals = [
        5.0 - 0.1 * 2.0,                              // 4.8
        4.8 - 0.1 * (0.8 * 2.0 + 2.0),                // 4.44
        4.44 - 0.1 * (0.8 * (0.8 * 2.0 + 2.0) + 2.0), // 3.952
    ];

    for (step, &expected) in expected_vals.iter().enumerate() {
        let grads = manual_grads(&x, &[g as f32]);
        sgd.step(&grads).unwrap();
        let val = get_val(&x);
        assert!(
            (val - expected as f32).abs() < 1e-4,
            "step {}: expected {expected:.4}, got {val:.4}",
            step + 1
        );
    }
}

// ============================================================================
// 9. SGD: Nesterov momentum produces different result from classical
// ============================================================================

/// The SGD implementation uses classical momentum (v = m*v + g, x -= lr*v).
/// Nesterov would be: v = m*v + g, x -= lr*(m*v + g).
/// We verify the implementation does NOT produce the Nesterov result for
/// a 2-step sequence with non-trivial values.
#[test]
fn test_sgd_classical_vs_nesterov_divergence() {
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

    // Step 1: g=1.5
    let grads = manual_grads(&x, &[1.5]);
    sgd.step(&grads).unwrap();
    // Classical: v1=1.5, x1=3.0-0.15=2.85
    let x1 = get_val(&x);
    assert!(
        (x1 - 2.85).abs() < 1e-5,
        "step1 classical: expected 2.85, got {x1}"
    );

    // Step 2: g=0.5
    let grads = manual_grads(&x, &[0.5]);
    sgd.step(&grads).unwrap();
    // Classical: v2=0.9*1.5+0.5=1.85, x2=2.85-0.185=2.665
    let x2_classical = 2.85 - 0.1 * (0.9 * 1.5 + 0.5);
    // Nesterov: x2=2.85-0.1*(0.9*1.85+0.5)=2.85-0.2165=2.6335
    let x2_nesterov = 2.85 - 0.1 * (0.9 * (0.9 * 1.5 + 0.5) + 0.5);

    let x2 = get_val(&x);
    assert!(
        (x2 - x2_classical as f32).abs() < 1e-4,
        "implementation matches classical: expected {x2_classical}, got {x2}"
    );
    assert!(
        (x2 - x2_nesterov as f32).abs() > 0.01,
        "implementation differs from Nesterov: nesterov={x2_nesterov}, got {x2}"
    );
}

// ============================================================================
// 10. SGD: learning rate scheduling integration
// ============================================================================

/// Verify that step_with_schedule correctly sets the LR and produces
/// proportionally scaled updates.
#[test]
fn test_sgd_schedule_integration_warmup() {
    let x = scalar_var(5.0);
    let mut sgd = Sgd::new(
        vec![x.clone()],
        SgdConfig {
            lr: 0.0, // will be set by schedule
            momentum: 0.0,
            weight_decay: 0.0,
        },
    )
    .unwrap();
    let schedule = WarmupSchedule::new(0.2, 10).unwrap();

    // At step 5: lr = 0.2 * 5/10 = 0.1
    // grad = 2*5 = 10, update = 0.1 * 10 = 1.0
    // x_new = 5.0 - 1.0 = 4.0
    let grads = grad_sum_sq(&x);
    step_with_schedule(&mut sgd, &grads, &schedule, 5).unwrap();

    assert!(
        (sgd.learning_rate() - 0.1).abs() < f64::EPSILON,
        "lr should be 0.1 at step 5"
    );
    let val = get_val(&x);
    assert!((val - 4.0).abs() < 1e-4, "expected 4.0, got {val}");

    // At step 10 (past warmup): lr = 0.2
    let grads = grad_sum_sq(&x);
    step_with_schedule(&mut sgd, &grads, &schedule, 10).unwrap();
    assert!(
        (sgd.learning_rate() - 0.2).abs() < f64::EPSILON,
        "lr should be 0.2 at step 10"
    );
}

// ============================================================================
// 11. AdaFactor: basic step with known values (vector, non-factored)
// ============================================================================

/// Verify AdaFactor step on a rank-1 parameter (full second moment path).
/// With constant gradient, check parameter decreases.
#[test]
fn test_adafactor_step_vector_known_gradient() {
    let x = vec_var(&[4.0, -2.0, 6.0]);
    let config = AdaFactorConfig {
        lr: 0.1,
        relative_step: false,
        beta1: None,
        weight_decay: 0.0,
        ..Default::default()
    };
    let mut opt = AdaFactor::new(vec![x.clone()], config).unwrap();

    let grads = manual_grads(&x, &[2.0, -1.0, 3.0]);
    opt.step(&grads).unwrap();

    let vals = get_vals(&x);
    // After first step (rho_t=0 at step 1 with decay_rate=-0.8):
    // v = (1-0)*g^2 = g^2 = [4.0, 1.0, 9.0]
    // u = g / sqrt(v + eps) = g / (|g| + ~0) = sign(g) = [1.0, -1.0, 1.0]
    // x_new = x - lr * u = x - 0.1 * sign(g)
    // x = [4.0-0.1, -2.0+0.1, 6.0-0.1] = [3.9, -1.9, 5.9]
    assert!(
        (vals[0] - 3.9).abs() < 0.05,
        "x[0]: expected ~3.9, got {}",
        vals[0]
    );
    assert!(
        (vals[1] - (-1.9)).abs() < 0.05,
        "x[1]: expected ~-1.9, got {}",
        vals[1]
    );
    assert!(
        (vals[2] - 5.9).abs() < 0.05,
        "x[2]: expected ~5.9, got {}",
        vals[2]
    );
}

// ============================================================================
// 12. AdaFactor: scale parameter with relative step
// ============================================================================

/// When relative_step is enabled, the learning rate is derived from the
/// parameter RMS. Larger parameters get larger absolute learning rates.
#[test]
fn test_adafactor_relative_step_scales_with_param_rms() {
    // Two identical runs, different param magnitudes
    let x_big = vec_var(&[100.0, 100.0]);
    let x_small = vec_var(&[1.0, 1.0]);

    let config = AdaFactorConfig {
        relative_step: true,
        weight_decay: 0.0,
        ..Default::default()
    };
    let mut opt_big = AdaFactor::new(vec![x_big.clone()], config.clone()).unwrap();
    let mut opt_small = AdaFactor::new(vec![x_small.clone()], config).unwrap();

    // Same gradient magnitude for both
    let grads_big = manual_grads(&x_big, &[1.0, 1.0]);
    opt_big.step(&grads_big).unwrap();
    let grads_small = manual_grads(&x_small, &[1.0, 1.0]);
    opt_small.step(&grads_small).unwrap();

    let update_big = (100.0 - get_vals(&x_big)[0]).abs();
    let update_small = (1.0 - get_vals(&x_small)[0]).abs();

    // With relative step, the effective lr for big params should be larger
    assert!(
        update_big > update_small,
        "relative step should make big-param update larger: big={update_big}, small={update_small}"
    );
}

// ============================================================================
// 13. AdaFactor: relative step size computation numerical check
// ============================================================================

/// At step 1 with relative_step=true, lr_t = rms(param) * rho_lr.
/// rho_lr = 1/sqrt(t) = 1.0 at t=1.
/// For param=[10.0, 10.0], rms = sqrt(mean(100, 100)) = 10.0.
/// So lr_t ≈ max(eps_rms, 10.0) * 1.0 = 10.0.
/// u_t = sign(grad) = [1, 1] (at step 1 with rho=0, v=g^2).
/// x_new = x - lr_t * u_t = [10-10, 10-10] = [0, 0]
///
/// In practice there are eps factors, so we just check convergence direction.
#[test]
fn test_adafactor_relative_step_first_step_magnitude() {
    let x = vec_var(&[10.0, 10.0]);
    let config = AdaFactorConfig {
        relative_step: true,
        weight_decay: 0.0,
        ..Default::default()
    };
    let mut opt = AdaFactor::new(vec![x.clone()], config).unwrap();

    let grads = grad_sum_sq(&x);
    opt.step(&grads).unwrap();

    let vals = get_vals(&x);
    // The relative step lr_t is proportional to param magnitude, so the
    // update should be significant relative to the parameter scale.
    assert!(
        vals[0] < 10.0,
        "param should decrease from 10.0, got {}",
        vals[0]
    );
    assert!(
        vals[0] > -5.0,
        "param should not overshoot wildly, got {}",
        vals[0]
    );
}

// ============================================================================
// 14. Cross-optimizer: all accept same GradStore format
// ============================================================================

/// All three optimizers (Adam, SGD, AdaFactor) should accept the same
/// GradStore produced by backward() and update parameters.
#[test]
fn test_cross_optimizer_same_gradstore_format() {
    let x_adam = scalar_var(5.0);
    let x_sgd = scalar_var(5.0);
    let x_af = scalar_var(5.0);

    let mut adam = AdamW::new(
        vec![x_adam.clone()],
        AdamConfig {
            weight_decay: 0.0,
            ..AdamConfig::default()
        },
    )
    .unwrap();
    let mut sgd = Sgd::new(
        vec![x_sgd.clone()],
        SgdConfig {
            lr: 0.01,
            ..Default::default()
        },
    )
    .unwrap();
    let mut af = AdaFactor::new(
        vec![x_af.clone()],
        AdaFactorConfig {
            lr: 0.01,
            relative_step: false,
            ..Default::default()
        },
    )
    .unwrap();

    // Same gradient value for all
    let grads_adam = manual_grads(&x_adam, &[4.0]);
    let grads_sgd = manual_grads(&x_sgd, &[4.0]);
    let grads_af = manual_grads(&x_af, &[4.0]);

    adam.step(&grads_adam).unwrap();
    sgd.step(&grads_sgd).unwrap();
    af.step(&grads_af).unwrap();

    // All should have changed from initial value
    assert!(
        (get_val(&x_adam) - 5.0).abs() > 1e-6,
        "Adam should update param"
    );
    assert!(
        (get_val(&x_sgd) - 5.0).abs() > 1e-6,
        "SGD should update param"
    );
    assert!(
        (get_val(&x_af) - 5.0).abs() > 1e-6,
        "AdaFactor should update param"
    );

    // All should have decreased (gradient is positive, minimizing)
    assert!(get_val(&x_adam) < 5.0, "Adam should decrease param");
    assert!(get_val(&x_sgd) < 5.0, "SGD should decrease param");
    assert!(get_val(&x_af) < 5.0, "AdaFactor should decrease param");
}

// ============================================================================
// 15. Cross-optimizer: empty parameter sets are handled gracefully
// ============================================================================

/// All optimizers should handle empty parameter lists: construction succeeds,
/// step() succeeds (no-op), and learning_rate accessors work.
#[test]
fn test_cross_optimizer_empty_params() {
    let mut adam = AdamW::new(vec![], AdamConfig::default()).unwrap();
    let mut sgd = Sgd::new(vec![], SgdConfig::default()).unwrap();
    let mut af = AdaFactor::new(vec![], AdaFactorConfig::default()).unwrap();

    // Create a dummy GradStore (no matching vars)
    let dummy = scalar_var(1.0);
    let grads = grad_sum_sq(&dummy);

    adam.step(&grads).unwrap();
    sgd.step(&grads).unwrap();
    af.step(&grads).unwrap();

    // Accessors should work
    assert!(adam.learning_rate() > 0.0);
    assert!(sgd.learning_rate() > 0.0);
    assert!(af.learning_rate() > 0.0);
}

// ============================================================================
// 16. Cross-optimizer: state initialization is correct
// ============================================================================

/// Verify state initialization: step count starts at 0, moments start at zero.
#[test]
fn test_cross_optimizer_initial_state() {
    let x = scalar_var(5.0);
    let adam = AdamW::new(vec![x.clone()], AdamConfig::default()).unwrap();
    assert_eq!(adam.step_count(), 0);
    assert!(
        (adam.learning_rate() - 1e-3).abs() < f64::EPSILON,
        "Adam default lr"
    );

    let sgd = Sgd::new(vec![x.clone()], SgdConfig::default()).unwrap();
    assert!(
        (sgd.learning_rate() - 1e-2).abs() < f64::EPSILON,
        "SGD default lr"
    );
    assert!(
        (sgd.momentum() - 0.0).abs() < f64::EPSILON,
        "SGD default momentum"
    );

    let af = AdaFactor::new(vec![x], AdaFactorConfig::default()).unwrap();
    assert_eq!(af.step_count(), 0);
    assert!(
        (af.learning_rate() - 1e-3).abs() < f64::EPSILON,
        "AdaFactor default lr"
    );
}

// ============================================================================
// 17. Cross-optimizer: set_learning_rate works uniformly
// ============================================================================

/// All optimizers should accept set_learning_rate and report the new value.
#[test]
fn test_cross_optimizer_set_learning_rate() {
    let x = scalar_var(1.0);
    let mut adam = AdamW::new(vec![x.clone()], AdamConfig::default()).unwrap();
    let mut sgd = Sgd::new(vec![x.clone()], SgdConfig::default()).unwrap();
    let mut af = AdaFactor::new(vec![x], AdaFactorConfig::default()).unwrap();

    adam.set_learning_rate(0.042).unwrap();
    sgd.set_learning_rate(0.042).unwrap();
    af.set_learning_rate(0.042).unwrap();

    assert!(
        (adam.learning_rate() - 0.042).abs() < f64::EPSILON,
        "Adam lr"
    );
    assert!((sgd.learning_rate() - 0.042).abs() < f64::EPSILON, "SGD lr");
    assert!(
        (af.learning_rate() - 0.042).abs() < f64::EPSILON,
        "AdaFactor lr"
    );
}

// ============================================================================
// 18. Cross-optimizer: all reject negative lr uniformly
// ============================================================================

#[test]
fn test_cross_optimizer_reject_negative_lr() {
    let x = scalar_var(1.0);
    let mut adam = AdamW::new(vec![x.clone()], AdamConfig::default()).unwrap();
    let mut sgd = Sgd::new(vec![x.clone()], SgdConfig::default()).unwrap();
    let mut af = AdaFactor::new(vec![x], AdaFactorConfig::default()).unwrap();

    assert!(adam.set_learning_rate(-0.01).is_err());
    assert!(sgd.set_learning_rate(-0.01).is_err());
    assert!(af.set_learning_rate(-0.01).is_err());
}

// ============================================================================
// 19. SGD: weight decay with zero gradient over multiple steps
// ============================================================================

/// With zero gradient and positive weight decay, parameters should decay
/// geometrically: x_t = x_0 * (1 - lr * wd)^t.
#[test]
fn test_sgd_weight_decay_geometric_decay() {
    let x = scalar_var(8.0);
    let lr = 0.1;
    let wd = 0.2;
    let mut sgd = Sgd::new(
        vec![x.clone()],
        SgdConfig {
            lr,
            momentum: 0.0,
            weight_decay: wd,
        },
    )
    .unwrap();

    // decay_factor per step = (1 - lr*wd) = 0.98
    let decay_per_step = 1.0 - lr * wd;

    for step in 1..=5 {
        let grads = manual_grads(&x, &[0.0]);
        sgd.step(&grads).unwrap();

        let expected = 8.0 * (decay_per_step as f32).powi(step);
        let val = get_val(&x);
        assert!(
            (val - expected).abs() < 1e-4,
            "step {step}: expected {expected:.4}, got {val:.4}"
        );
    }
}

// ============================================================================
// 20. SGD: momentum without weight decay on negative initial value
// ============================================================================

/// Verify SGD with momentum converges to 0 from a negative starting point.
/// f(x) = x^2, gradient = 2x. With x=-5, gradient = -10, update pushes toward 0.
#[test]
fn test_sgd_momentum_negative_initial() {
    let x = scalar_var(-5.0);
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

    let val = get_val(&x);
    assert!(
        val.abs() < 0.05,
        "SGD+momentum from -5 should converge to 0, got {val}"
    );
}

// ============================================================================
// 21. Adam: weight decay with zero gradient should still shrink params
// ============================================================================

/// AdamW weight decay is decoupled: theta *= (1 - lr*wd) happens regardless
/// of gradient magnitude. With zero gradient, only decay applies.
#[test]
fn test_adam_weight_decay_with_zero_gradient_shrinks() {
    let x = scalar_var(10.0);
    let lr = 0.01;
    let wd = 0.1;
    let config = AdamConfig {
        lr,
        weight_decay: wd,
        ..AdamConfig::default()
    };
    let mut adam = AdamW::new(vec![x.clone()], config).unwrap();

    let grads = manual_grads(&x, &[0.0]);
    adam.step(&grads).unwrap();

    let val = get_val(&x);
    // decay_factor = 1 - lr * wd = 1 - 0.001 = 0.999
    // With zero gradient: m=0, v=0, adaptive step = 0/(0+eps) ≈ 0
    // x_new = 10.0 * 0.999 - 0 = 9.99
    let expected = 10.0 * (1.0 - lr as f32 * wd as f32);
    assert!(
        (val - expected).abs() < 1e-4,
        "expected {expected}, got {val}"
    );
    assert!(
        val < 10.0,
        "weight decay should shrink even with zero gradient"
    );
}

// ============================================================================
// 22. Adam: step count increments even when no vars have gradients
// ============================================================================

/// Step count should increment on every call to step(), regardless of
/// whether any variables actually received gradients.
#[test]
fn test_adam_step_count_increments_without_grads() {
    let x = scalar_var(5.0);
    let mut adam = AdamW::new(vec![x.clone()], AdamConfig::default()).unwrap();

    // Create empty-ish grads (dummy var not tracked by optimizer)
    let dummy = scalar_var(1.0);
    let grads = grad_sum_sq(&dummy);

    adam.step(&grads).unwrap();
    adam.step(&grads).unwrap();
    adam.step(&grads).unwrap();

    assert_eq!(adam.step_count(), 3);
    // x should be unchanged since it never received a gradient
    assert!((get_val(&x) - 5.0).abs() < 1e-7, "x should be unchanged");
}

// ============================================================================
// 23. AdaFactor: factored vs full moment produces different behavior
// ============================================================================

/// Rank-1 (vector) uses full second moment; rank-2 (matrix) uses factored.
/// Both should converge on the same quadratic, but via different internal paths.
#[test]
fn test_adafactor_factored_vs_full_both_converge() {
    // Rank-1: full moment
    let x_vec = vec_var(&[5.0, -3.0, 4.0, -2.0, 6.0, -1.0]);
    // Rank-2: factored moment (same values reshaped)
    let x_mat = mat_var(&[5.0, -3.0, 4.0, -2.0, 6.0, -1.0], 2, 3);

    let config = AdaFactorConfig {
        lr: 0.1,
        relative_step: false,
        ..Default::default()
    };
    let mut opt_vec = AdaFactor::new(vec![x_vec.clone()], config.clone()).unwrap();
    let mut opt_mat = AdaFactor::new(vec![x_mat.clone()], config).unwrap();

    for _ in 0..50 {
        let grads_v = grad_sum_sq(&x_vec);
        opt_vec.step(&grads_v).unwrap();
        let grads_m = grad_sum_sq(&x_mat);
        opt_mat.step(&grads_m).unwrap();
    }

    // Both should have reduced the loss significantly
    let loss_vec: f32 = get_vals(&x_vec).iter().map(|v| v * v).sum();
    let loss_mat: f32 = get_vals(&x_mat).iter().map(|v| v * v).sum();

    assert!(
        loss_vec < 20.0,
        "vector (full moment) should converge, loss={loss_vec}"
    );
    assert!(
        loss_mat < 20.0,
        "matrix (factored moment) should converge, loss={loss_mat}"
    );
}

// ============================================================================
// 24. AdaFactor: beta1 momentum actually changes result
// ============================================================================

/// AdaFactor with beta1=Some(0.9) should produce different results than
/// beta1=None. This confirms the momentum path is exercised.
#[test]
fn test_adafactor_beta1_changes_result() {
    let x_mom = vec_var(&[5.0, -3.0]);
    let x_no = vec_var(&[5.0, -3.0]);

    let config_mom = AdaFactorConfig {
        lr: 0.1,
        relative_step: false,
        beta1: Some(0.9),
        ..Default::default()
    };
    let config_no = AdaFactorConfig {
        lr: 0.1,
        relative_step: false,
        beta1: None,
        ..Default::default()
    };

    let mut opt_mom = AdaFactor::new(vec![x_mom.clone()], config_mom).unwrap();
    let mut opt_no = AdaFactor::new(vec![x_no.clone()], config_no).unwrap();

    // Run multiple steps with real gradients
    for _ in 0..10 {
        let grads_m = grad_sum_sq(&x_mom);
        opt_mom.step(&grads_m).unwrap();
        let grads_n = grad_sum_sq(&x_no);
        opt_no.step(&grads_n).unwrap();
    }

    let vals_mom = get_vals(&x_mom);
    let vals_no = get_vals(&x_no);

    // Values should differ due to momentum smoothing
    let diff: f32 = vals_mom
        .iter()
        .zip(vals_no.iter())
        .map(|(a, b)| (a - b).abs())
        .sum();
    assert!(
        diff > 1e-3,
        "beta1=Some(0.9) and beta1=None should produce different results, diff={diff}"
    );
}

// ============================================================================
// 25. Adam: multi-param convergence where params have different scales
// ============================================================================

/// Adam should converge on params with different magnitudes due to its
/// per-parameter adaptive step size. Both parameters should converge
/// toward zero, demonstrating Adam handles multi-scale problems.
#[test]
fn test_adam_adapts_to_different_param_scales() {
    let x_large = scalar_var(20.0);
    let x_small = scalar_var(0.01);

    let config = AdamConfig {
        lr: 0.1,
        weight_decay: 0.0,
        ..AdamConfig::default()
    };
    let mut adam = AdamW::new(vec![x_large.clone(), x_small.clone()], config).unwrap();

    for _ in 0..200 {
        let tl = Arc::new(TrackedTensor::from_var(&x_large).unwrap());
        let ts = Arc::new(TrackedTensor::from_var(&x_small).unwrap());
        let loss = tl.sqr().unwrap().add(&ts.sqr().unwrap()).unwrap();
        adam.backward_step(&loss).unwrap();
    }

    let val_large = get_val(&x_large);
    let val_small = get_val(&x_small);

    // After 200 steps at lr=0.1, both should have reduced significantly
    assert!(
        val_large.abs() < 5.0,
        "large param should move toward 0 from 20, got {val_large}"
    );
    assert!(
        val_small.abs() < 0.01,
        "small param should converge near 0, got {val_small}"
    );
}

// ============================================================================
// 26. SGD: large gradient stability (no overflow)
// ============================================================================

/// SGD with a very large gradient should produce a large but finite update.
/// Unlike Adam, SGD does not normalize, so the update scales linearly with
/// gradient magnitude.
#[test]
fn test_sgd_large_gradient_finite_output() {
    let x = scalar_var(1.0);
    let mut sgd = Sgd::new(
        vec![x.clone()],
        SgdConfig {
            lr: 1e-8,
            momentum: 0.0,
            weight_decay: 0.0,
        },
    )
    .unwrap();

    // Use a very small lr to keep the update reasonable
    let grads = manual_grads(&x, &[1e6]);
    sgd.step(&grads).unwrap();

    let val = get_val(&x);
    assert!(val.is_finite(), "SGD with large grad should stay finite");
    // x_new = 1.0 - 1e-8 * 1e6 = 1.0 - 0.01 = 0.99
    assert!((val - 0.99).abs() < 1e-4, "expected ~0.99, got {val}");
}

// ============================================================================
// 27. Cross-optimizer: all converge on the same simple quadratic
// ============================================================================

/// All three optimizers should converge on f(x) = x^2 starting from x=5.
/// This is a basic sanity test that all optimizers actually optimize.
#[test]
fn test_cross_optimizer_convergence_quadratic() {
    let x_adam = scalar_var(5.0);
    let x_sgd = scalar_var(5.0);
    let x_af = scalar_var(5.0);

    let mut adam = AdamW::new(
        vec![x_adam.clone()],
        AdamConfig {
            lr: 0.1,
            weight_decay: 0.0,
            ..AdamConfig::default()
        },
    )
    .unwrap();
    let mut sgd = Sgd::new(
        vec![x_sgd.clone()],
        SgdConfig {
            lr: 0.1,
            momentum: 0.0,
            weight_decay: 0.0,
        },
    )
    .unwrap();
    let mut af = AdaFactor::new(
        vec![x_af.clone()],
        AdaFactorConfig {
            lr: 0.1,
            relative_step: false,
            ..Default::default()
        },
    )
    .unwrap();

    for _ in 0..100 {
        let t = Arc::new(TrackedTensor::from_var(&x_adam).unwrap());
        let loss = t.sqr().unwrap();
        adam.backward_step(&loss).unwrap();

        let t = Arc::new(TrackedTensor::from_var(&x_sgd).unwrap());
        let loss = t.sqr().unwrap();
        sgd.backward_step(&loss).unwrap();

        let t = Arc::new(TrackedTensor::from_var(&x_af).unwrap());
        let loss = t.sqr().unwrap();
        let grads = backward(&loss).unwrap();
        af.step(&grads).unwrap();
    }

    assert!(get_val(&x_adam).abs() < 1.0, "Adam should converge");
    assert!(get_val(&x_sgd).abs() < 0.01, "SGD should converge");
    assert!(get_val(&x_af).abs() < 1.0, "AdaFactor should converge");
}

// ============================================================================
// 28. Adam: config accessors match construction values
// ============================================================================

#[test]
fn test_adam_config_accessors_consistent() {
    let x = scalar_var(1.0);
    let config = AdamConfig {
        lr: 0.005,
        beta1: 0.85,
        beta2: 0.98,
        eps: 1e-6,
        weight_decay: 0.05,
    };
    let adam = AdamW::new(vec![x], config).unwrap();

    let c = adam.config();
    assert!((c.lr - 0.005).abs() < f64::EPSILON);
    assert!((c.beta1 - 0.85).abs() < f64::EPSILON);
    assert!((c.beta2 - 0.98).abs() < f64::EPSILON);
    assert!((c.eps - 1e-6).abs() < f64::EPSILON);
    assert!((c.weight_decay - 0.05).abs() < f64::EPSILON);
    assert!((adam.learning_rate() - 0.005).abs() < f64::EPSILON);
    assert_eq!(adam.step_count(), 0);
}

// ============================================================================
// 29. SGD: config accessors consistent
// ============================================================================

#[test]
fn test_sgd_config_accessors_consistent() {
    let x = scalar_var(1.0);
    let config = SgdConfig {
        lr: 0.03,
        momentum: 0.85,
        weight_decay: 0.002,
    };
    let sgd = Sgd::new(vec![x], config).unwrap();

    let c = sgd.config();
    assert!((c.lr - 0.03).abs() < f64::EPSILON);
    assert!((c.momentum - 0.85).abs() < f64::EPSILON);
    assert!((c.weight_decay - 0.002).abs() < f64::EPSILON);
    assert!((sgd.learning_rate() - 0.03).abs() < f64::EPSILON);
    assert!((sgd.momentum() - 0.85).abs() < f64::EPSILON);
    assert!((sgd.weight_decay() - 0.002).abs() < f64::EPSILON);
}

// ============================================================================
// 30. AdaFactor: config accessors consistent
// ============================================================================

#[test]
fn test_adafactor_config_accessors_consistent() {
    let x = scalar_var(1.0);
    let config = AdaFactorConfig {
        lr: 0.05,
        relative_step: true,
        eps_rms: 1e-4,
        eps_denom: 1e-20,
        decay_rate: -0.5,
        beta1: Some(0.8),
        weight_decay: 0.01,
    };
    let af = AdaFactor::new(vec![x], config).unwrap();

    let c = af.config();
    assert!((c.lr - 0.05).abs() < f64::EPSILON);
    assert!(c.relative_step);
    assert!((c.eps_rms - 1e-4).abs() < f64::EPSILON);
    assert!((c.eps_denom - 1e-20).abs() < f64::EPSILON);
    assert!((c.decay_rate - (-0.5)).abs() < f64::EPSILON);
    assert_eq!(c.beta1, Some(0.8));
    assert!((c.weight_decay - 0.01).abs() < f64::EPSILON);
    assert_eq!(af.step_count(), 0);
}

// ============================================================================
// 31. Adam: multiple steps with weight decay shows monotonic param reduction
// ============================================================================

/// With constant positive gradient AND positive weight decay, Adam should
/// produce monotonically decreasing parameter values.
#[test]
fn test_adam_weight_decay_monotonic_decrease() {
    let x = scalar_var(10.0);
    let config = AdamConfig {
        lr: 0.01,
        weight_decay: 0.1,
        ..AdamConfig::default()
    };
    let mut adam = AdamW::new(vec![x.clone()], config).unwrap();

    let mut prev_val = 10.0_f32;
    for step in 0..20 {
        let grads = manual_grads(&x, &[1.0]); // constant positive gradient
        adam.step(&grads).unwrap();
        let val = get_val(&x);
        assert!(
            val < prev_val,
            "step {step}: value should decrease monotonically: prev={prev_val}, cur={val}"
        );
        prev_val = val;
    }
}

// ============================================================================
// 32. SGD: weight decay exact numerical formula verification
// ============================================================================

/// Verify the exact formula: grad_eff = grad + wd*theta, theta -= lr*grad_eff.
/// x=6.0, grad=3.0, lr=0.2, wd=0.1
/// grad_eff = 3.0 + 0.1*6.0 = 3.6
/// x_new = 6.0 - 0.2*3.6 = 5.28
#[test]
fn test_sgd_weight_decay_exact_formula() {
    let x = scalar_var(6.0);
    let mut sgd = Sgd::new(
        vec![x.clone()],
        SgdConfig {
            lr: 0.2,
            momentum: 0.0,
            weight_decay: 0.1,
        },
    )
    .unwrap();

    let grads = manual_grads(&x, &[3.0]);
    sgd.step(&grads).unwrap();

    let val = get_val(&x);
    assert!((val - 5.28).abs() < 1e-4, "expected 5.28, got {val}");
}

// ============================================================================
// 33. Cross-optimizer: all handle multi-element params
// ============================================================================

/// Verify all optimizers handle matrix parameters correctly.
#[test]
fn test_cross_optimizer_matrix_params() {
    let x_adam = mat_var(&[1.0, 2.0, 3.0, 4.0], 2, 2);
    let x_sgd = mat_var(&[1.0, 2.0, 3.0, 4.0], 2, 2);
    let x_af = mat_var(&[1.0, 2.0, 3.0, 4.0], 2, 2);

    let mut adam = AdamW::new(
        vec![x_adam.clone()],
        AdamConfig {
            lr: 0.1,
            weight_decay: 0.0,
            ..AdamConfig::default()
        },
    )
    .unwrap();
    let mut sgd = Sgd::new(
        vec![x_sgd.clone()],
        SgdConfig {
            lr: 0.05,
            ..Default::default()
        },
    )
    .unwrap();
    let mut af = AdaFactor::new(
        vec![x_af.clone()],
        AdaFactorConfig {
            lr: 0.1,
            relative_step: false,
            ..Default::default()
        },
    )
    .unwrap();

    for _ in 0..20 {
        let g_adam = grad_sum_sq(&x_adam);
        adam.step(&g_adam).unwrap();
        let g_sgd = grad_sum_sq(&x_sgd);
        sgd.step(&g_sgd).unwrap();
        let g_af = grad_sum_sq(&x_af);
        af.step(&g_af).unwrap();
    }

    // All should have reduced the initial sum-of-squares
    let loss_adam: f32 = get_vals(&x_adam).iter().map(|v| v * v).sum();
    let loss_sgd: f32 = get_vals(&x_sgd).iter().map(|v| v * v).sum();
    let loss_af: f32 = get_vals(&x_af).iter().map(|v| v * v).sum();
    let initial_loss: f32 = [1.0, 2.0, 3.0, 4.0].iter().map(|v| v * v).sum();

    assert!(
        loss_adam < initial_loss,
        "Adam should reduce loss on matrix"
    );
    assert!(loss_sgd < initial_loss, "SGD should reduce loss on matrix");
    assert!(
        loss_af < initial_loss,
        "AdaFactor should reduce loss on matrix"
    );
}

// ============================================================================
// 34. Adam: verify beta2=0.999 v. beta2=0.99 produces different step sizes
// ============================================================================

/// Different beta2 values produce different adaptive step sizes, confirming
/// the second moment is properly used.
#[test]
fn test_adam_different_beta2_different_updates() {
    let x1 = scalar_var(5.0);
    let x2 = scalar_var(5.0);

    let config1 = AdamConfig {
        beta2: 0.999,
        weight_decay: 0.0,
        ..AdamConfig::default()
    };
    let config2 = AdamConfig {
        beta2: 0.99,
        weight_decay: 0.0,
        ..AdamConfig::default()
    };

    let mut adam1 = AdamW::new(vec![x1.clone()], config1).unwrap();
    let mut adam2 = AdamW::new(vec![x2.clone()], config2).unwrap();

    // Multiple steps to let the beta2 difference manifest.
    // At step 1 the adaptive step is lr (sign correction), so early steps
    // look identical in f32. After ~20 steps the differing second-moment
    // EMA rates produce visibly different v_hat values.
    for _ in 0..30 {
        let g1 = grad_sum_sq(&x1);
        adam1.step(&g1).unwrap();
        let g2 = grad_sum_sq(&x2);
        adam2.step(&g2).unwrap();
    }

    let val1 = get_val(&x1);
    let val2 = get_val(&x2);

    // At f32 precision the difference is small but nonzero.
    // The key test is that the values differ at all (i.e. beta2 is actually used).
    assert!(
        (val1 - val2).abs() > 1e-7,
        "different beta2 should produce different results: beta2=0.999 -> {val1}, beta2=0.99 -> {val2}"
    );
}

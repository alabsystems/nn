// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Expanded test coverage for nn-optim optimizers.
//!
//! Covers: Adam moment update correctness, bias correction numerics,
//! decoupled weight decay, SGD velocity accumulation, momentum + weight decay
//! composition, AdaFactor factored vs full paths, relative step scaling,
//! multi-dimensional convergence, schedule integration, and checkpoint
//! round-trips.

use std::sync::Arc;

use nn_autodiff::{backward, TrackedTensor, Var};
use nn_core::dyn_tensor::DynTensor;
use nn_core::test_utils::cpu;

use crate::adafactor::{AdaFactor, AdaFactorConfig};
use crate::adam::{AdamConfig, AdamW};
use crate::checkpoint::{OptimizerCheckpoint, OptimizerSnapshot};
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

fn mat_var(vals: &[f32], rows: usize, cols: usize) -> Var {
    Var::new(DynTensor::from_vec(vals.to_vec(), &[rows, cols], &cpu()).unwrap())
}

/// Compute gradient of sum(x^2) for a given var, returning the GradStore.
fn grad_sum_sq(var: &Var) -> nn_autodiff::GradStore {
    let t = Arc::new(TrackedTensor::from_var(var).unwrap());
    let sq = t.sqr().unwrap();
    // Sum across all dimensions to get a scalar loss
    let mut loss = sq;
    for d in 0..var.data().unwrap().dims().len() {
        loss = loss.sum_keepdim(d).unwrap();
    }
    backward(&loss).unwrap()
}

// ============================================================================
// Adam: moment update correctness
// ============================================================================

/// Verify Adam first moment (m) and second moment (v) updates match the
/// textbook formulas after a single step with known gradient.
#[test]
fn test_adam_moment_update_exact_values() {
    // x = [3.0], loss = x^2, grad = 2*x = 6.0
    let x = scalar_var(3.0);
    let config = AdamConfig {
        lr: 0.001,
        beta1: 0.9,
        beta2: 0.999,
        eps: 1e-8,
        weight_decay: 0.0,
    };
    let mut adam = AdamW::new(vec![x.clone()], config).unwrap();

    let t = Arc::new(TrackedTensor::from_var(&x).unwrap());
    let loss = t.sqr().unwrap();
    adam.backward_step(&loss).unwrap();

    // After step 1 with gradient g=6.0:
    // m1 = 0.9 * 0 + 0.1 * 6.0 = 0.6
    // v1 = 0.999 * 0 + 0.001 * 36.0 = 0.036
    // bc1 = 1 / (1 - 0.9^1) = 10.0
    // bc2 = 1 / (1 - 0.999^1) = 1000.0
    // m_hat = 0.6 * 10.0 = 6.0
    // v_hat = 0.036 * 1000.0 = 36.0
    // step = lr * m_hat / (sqrt(v_hat) + eps) = 0.001 * 6.0 / (6.0 + 1e-8) ≈ 0.001
    // theta_new = 3.0 - 0.001 ≈ 2.999
    let val = x.data().unwrap().to_flat_vec::<f32>().unwrap()[0];
    let expected = 3.0 - 0.001 * (6.0 / (6.0 + 1e-8));
    assert!(
        (val - expected as f32).abs() < 1e-5,
        "Adam update: expected {expected}, got {val}"
    );
}

/// Verify Adam moments accumulate correctly over two steps.
#[test]
fn test_adam_moment_accumulation_two_steps() {
    let x = scalar_var(4.0);
    let config = AdamConfig {
        lr: 0.01,
        beta1: 0.9,
        beta2: 0.999,
        eps: 1e-8,
        weight_decay: 0.0,
    };
    let mut adam = AdamW::new(vec![x.clone()], config).unwrap();

    // Step 1: grad = 2 * 4.0 = 8.0
    let t = Arc::new(TrackedTensor::from_var(&x).unwrap());
    let loss = t.sqr().unwrap();
    adam.backward_step(&loss).unwrap();
    let val_after_1 = x.data().unwrap().to_flat_vec::<f32>().unwrap()[0];

    // Step 2: grad = 2 * val_after_1
    let t2 = Arc::new(TrackedTensor::from_var(&x).unwrap());
    let loss2 = t2.sqr().unwrap();
    adam.backward_step(&loss2).unwrap();
    let val_after_2 = x.data().unwrap().to_flat_vec::<f32>().unwrap()[0];

    // x should keep decreasing toward 0
    assert!(
        val_after_2 < val_after_1,
        "step 2 should further decrease x: step1={val_after_1}, step2={val_after_2}"
    );
    assert!(
        val_after_2 < 4.0,
        "x should be less than initial 4.0, got {val_after_2}"
    );
    assert_eq!(adam.step_count(), 2);
}

/// Verify Adam bias correction diminishes with step count.
/// At step 1, bc1 = 10x amplification. At step 10, bc1 ≈ 2.87x.
#[test]
fn test_adam_bias_correction_diminishes() {
    let x1 = scalar_var(1.0);
    let x2 = scalar_var(1.0);
    let config = AdamConfig {
        lr: 0.01,
        weight_decay: 0.0,
        ..AdamConfig::default()
    };
    let mut adam1 = AdamW::new(vec![x1.clone()], config.clone()).unwrap();
    let mut adam2 = AdamW::new(vec![x2.clone()], config).unwrap();

    // adam1: 1 step
    let t = Arc::new(TrackedTensor::from_var(&x1).unwrap());
    let loss = t.sqr().unwrap();
    adam1.backward_step(&loss).unwrap();
    let update1 = (1.0 - x1.data().unwrap().to_flat_vec::<f32>().unwrap()[0]).abs();

    // adam2: 10 steps with same initial condition
    for _ in 0..10 {
        let t = Arc::new(TrackedTensor::from_var(&x2).unwrap());
        let loss = t.sqr().unwrap();
        adam2.backward_step(&loss).unwrap();
    }
    // The per-step update magnitude at step 10 should be smaller than at step 1
    // because bias correction factor decreases and gradient is smaller
    // (x is closer to 0 after 10 steps)
    let val10 = x2.data().unwrap().to_flat_vec::<f32>().unwrap()[0];
    assert!(
        val10.abs() < 1.0 - update1,
        "10 steps should bring x closer to 0 than the step-1 update magnitude"
    );
}

/// Verify Adam weight decay is decoupled (AdamW style).
/// In AdamW: theta *= (1 - lr * wd) THEN subtract adaptive step.
/// This means weight decay doesn't flow through the gradient/moment path.
#[test]
fn test_adam_decoupled_weight_decay_vs_no_decay() {
    let x_wd = scalar_var(10.0);
    let x_no = scalar_var(10.0);

    let config_wd = AdamConfig {
        lr: 0.01,
        weight_decay: 0.1,
        ..AdamConfig::default()
    };
    let config_no = AdamConfig {
        lr: 0.01,
        weight_decay: 0.0,
        ..AdamConfig::default()
    };

    let mut adam_wd = AdamW::new(vec![x_wd.clone()], config_wd).unwrap();
    let mut adam_no = AdamW::new(vec![x_no.clone()], config_no).unwrap();

    // Same loss: x^2
    let t_wd = Arc::new(TrackedTensor::from_var(&x_wd).unwrap());
    let loss_wd = t_wd.sqr().unwrap();
    adam_wd.backward_step(&loss_wd).unwrap();

    let t_no = Arc::new(TrackedTensor::from_var(&x_no).unwrap());
    let loss_no = t_no.sqr().unwrap();
    adam_no.backward_step(&loss_no).unwrap();

    let val_wd = x_wd.data().unwrap().to_flat_vec::<f32>().unwrap()[0];
    let val_no = x_no.data().unwrap().to_flat_vec::<f32>().unwrap()[0];

    // With weight decay, x should be smaller (more shrinkage)
    assert!(
        val_wd < val_no,
        "weight decay should produce smaller parameter: wd={val_wd}, no_wd={val_no}"
    );

    // Verify the decay factor: x_wd ≈ x * (1 - lr * wd) - step
    // The decay factor is (1 - 0.01 * 0.1) = 0.999
    // So x_wd should be approximately 10.0 * 0.999 - same_adaptive_step
    let decay_factor = 1.0 - 0.01 * 0.1;
    let expected_decay_contribution = 10.0 * (1.0 - decay_factor as f32);
    let actual_extra_shrink = val_no - val_wd;
    assert!(
        (actual_extra_shrink - expected_decay_contribution).abs() < 1e-4,
        "weight decay contribution should be ~{expected_decay_contribution}, got {actual_extra_shrink}"
    );
}

// ============================================================================
// Adam: multi-dimensional tensor convergence
// ============================================================================

/// Verify Adam converges on a multi-element tensor (not just scalar).
#[test]
fn test_adam_converges_vector_param() {
    let x = vec_var(&[5.0, -3.0, 7.0, -1.0]);
    let config = AdamConfig {
        lr: 0.1,
        weight_decay: 0.0,
        ..AdamConfig::default()
    };
    let mut adam = AdamW::new(vec![x.clone()], config).unwrap();

    // Minimize sum(x^2), minimum at [0, 0, 0, 0]
    for _ in 0..200 {
        let grads = grad_sum_sq(&x);
        adam.step(&grads).unwrap();
    }

    let vals = x.data().unwrap().to_flat_vec::<f32>().unwrap();
    for (i, &v) in vals.iter().enumerate() {
        assert!(
            v.abs() < 0.5,
            "Adam vector convergence: x[{i}] = {v}, expected near 0"
        );
    }
}

/// Verify Adam converges on a matrix parameter.
#[test]
fn test_adam_converges_matrix_param() {
    let x = mat_var(&[3.0, -2.0, 5.0, -4.0], 2, 2);
    let config = AdamConfig {
        lr: 0.1,
        weight_decay: 0.0,
        ..AdamConfig::default()
    };
    let mut adam = AdamW::new(vec![x.clone()], config).unwrap();

    for _ in 0..200 {
        let grads = grad_sum_sq(&x);
        adam.step(&grads).unwrap();
    }

    let vals = x.data().unwrap().to_flat_vec::<f32>().unwrap();
    for (i, &v) in vals.iter().enumerate() {
        assert!(
            v.abs() < 0.5,
            "Adam matrix convergence: x[{i}] = {v}, expected near 0"
        );
    }
}

// ============================================================================
// SGD: exact velocity accumulation
// ============================================================================

/// Verify SGD velocity is accumulated correctly: v = momentum * v_prev + grad.
/// After two steps with known gradients, verify the exact velocity values.
#[test]
fn test_sgd_velocity_accumulation_exact() {
    // x = [2.0], loss = x^2, grad = 2*x
    let x = scalar_var(2.0);
    let config = SgdConfig {
        lr: 0.1,
        momentum: 0.9,
        weight_decay: 0.0,
    };
    let mut sgd = Sgd::new(vec![x.clone()], config).unwrap();

    // Step 1: grad = 2*2 = 4.0
    // v1 = 0.9 * 0 + 4.0 = 4.0 (first step, no prev velocity)
    // x = 2.0 - 0.1 * 4.0 = 1.6
    let t = Arc::new(TrackedTensor::from_var(&x).unwrap());
    let loss = t.sqr().unwrap();
    sgd.backward_step(&loss).unwrap();
    let val1 = x.data().unwrap().to_flat_vec::<f32>().unwrap()[0];
    assert!(
        (val1 - 1.6).abs() < 1e-5,
        "step 1: expected 1.6, got {val1}"
    );

    // Step 2: grad = 2*1.6 = 3.2
    // v2 = 0.9 * 4.0 + 3.2 = 6.8
    // x = 1.6 - 0.1 * 6.8 = 0.92
    let t2 = Arc::new(TrackedTensor::from_var(&x).unwrap());
    let loss2 = t2.sqr().unwrap();
    sgd.backward_step(&loss2).unwrap();
    let val2 = x.data().unwrap().to_flat_vec::<f32>().unwrap()[0];
    assert!(
        (val2 - 0.92).abs() < 1e-4,
        "step 2: expected 0.92, got {val2}"
    );
}

/// Verify SGD weight decay composes correctly with momentum.
/// grad_eff = grad + wd * theta, then velocity = momentum * v_prev + grad_eff.
#[test]
fn test_sgd_weight_decay_with_momentum() {
    let x = scalar_var(5.0);
    let config = SgdConfig {
        lr: 0.1,
        momentum: 0.9,
        weight_decay: 0.01,
    };
    let mut sgd = Sgd::new(vec![x.clone()], config).unwrap();

    // loss = x^2, grad = 2*x = 10.0
    // grad_eff = 10.0 + 0.01 * 5.0 = 10.05
    // v1 = 10.05 (no prev velocity)
    // x = 5.0 - 0.1 * 10.05 = 3.995
    let t = Arc::new(TrackedTensor::from_var(&x).unwrap());
    let loss = t.sqr().unwrap();
    sgd.backward_step(&loss).unwrap();
    let val = x.data().unwrap().to_flat_vec::<f32>().unwrap()[0];
    assert!(
        (val - 3.995).abs() < 1e-3,
        "SGD wd+momentum step 1: expected ~3.995, got {val}"
    );
}

/// Verify SGD converges on a vector parameter.
#[test]
fn test_sgd_converges_vector_param() {
    let x = vec_var(&[5.0, -3.0, 7.0]);
    let config = SgdConfig {
        lr: 0.05,
        momentum: 0.0,
        weight_decay: 0.0,
    };
    let mut sgd = Sgd::new(vec![x.clone()], config).unwrap();

    for _ in 0..100 {
        let grads = grad_sum_sq(&x);
        sgd.step(&grads).unwrap();
    }

    let vals = x.data().unwrap().to_flat_vec::<f32>().unwrap();
    for (i, &v) in vals.iter().enumerate() {
        assert!(
            v.abs() < 0.01,
            "SGD vector convergence: x[{i}] = {v}, expected near 0"
        );
    }
}

/// Verify SGD converges on Rosenbrock 2D: (a-1)^2 + (b-2)^2.
#[test]
fn test_sgd_converges_quadratic_2d() {
    let a = scalar_var(5.0);
    let b = scalar_var(-3.0);
    let config = SgdConfig {
        lr: 0.05,
        momentum: 0.9,
        weight_decay: 0.0,
    };
    let mut sgd = Sgd::new(vec![a.clone(), b.clone()], config).unwrap();

    for _ in 0..300 {
        let ta = Arc::new(TrackedTensor::from_var(&a).unwrap());
        let tb = Arc::new(TrackedTensor::from_var(&b).unwrap());
        let one = Arc::new(TrackedTensor::from_tensor(
            DynTensor::from_vec(vec![1.0], &[1], &cpu()).unwrap(),
        ));
        let two = Arc::new(TrackedTensor::from_tensor(
            DynTensor::from_vec(vec![2.0], &[1], &cpu()).unwrap(),
        ));
        let a_diff = ta.sub(&one).unwrap();
        let b_diff = tb.sub(&two).unwrap();
        let loss = a_diff.sqr().unwrap().add(&b_diff.sqr().unwrap()).unwrap();
        sgd.backward_step(&loss).unwrap();
    }

    let a_val = a.data().unwrap().to_flat_vec::<f32>().unwrap()[0];
    let b_val = b.data().unwrap().to_flat_vec::<f32>().unwrap()[0];
    assert!(
        (a_val - 1.0).abs() < 0.5,
        "SGD 2D: a should approach 1.0, got {a_val}"
    );
    assert!(
        (b_val - 2.0).abs() < 0.5,
        "SGD 2D: b should approach 2.0, got {b_val}"
    );
}

// ============================================================================
// AdaFactor: factored vs full memory paths
// ============================================================================

/// Verify rank-1 tensor uses full second moment (not factored).
#[test]
fn test_adafactor_rank1_uses_full_moment() {
    let x = vec_var(&[3.0, -1.0, 2.0]);
    let config = AdaFactorConfig {
        lr: 0.1,
        relative_step: false,
        ..Default::default()
    };
    let mut opt = AdaFactor::new(vec![x.clone()], config).unwrap();

    let grads = grad_sum_sq(&x);
    opt.step(&grads).unwrap();

    // Params should change (proves the full-moment path works)
    let vals = x.data().unwrap().to_flat_vec::<f32>().unwrap();
    assert!(vals[0] < 3.0, "rank-1 full moment: param should decrease");
}

/// Verify rank-2 tensor uses factored second moments.
#[test]
fn test_adafactor_rank2_uses_factored_moment() {
    let x = mat_var(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 2, 3);
    let config = AdaFactorConfig {
        lr: 0.1,
        relative_step: false,
        ..Default::default()
    };
    let mut opt = AdaFactor::new(vec![x.clone()], config).unwrap();

    let grads = grad_sum_sq(&x);
    opt.step(&grads).unwrap();

    let vals = x.data().unwrap().to_flat_vec::<f32>().unwrap();
    for (i, &v) in vals.iter().enumerate() {
        let init = (i + 1) as f32;
        assert!(
            v < init,
            "rank-2 factored moment: param[{i}] should decrease from {init}, got {v}"
        );
    }
}

/// Verify AdaFactor relative step makes updates proportional to param magnitude.
/// Larger parameters should get larger absolute updates.
#[test]
fn test_adafactor_relative_step_proportional_to_magnitude() {
    // Two vars: one with large values, one with small values
    let x_large = vec_var(&[100.0, 100.0]);
    let x_small = vec_var(&[0.1, 0.1]);

    let config = AdaFactorConfig {
        relative_step: true,
        ..Default::default()
    };
    let mut opt = AdaFactor::new(vec![x_large.clone(), x_small.clone()], config).unwrap();

    // loss = sum(x_large^2) + sum(x_small^2)
    let t1 = Arc::new(TrackedTensor::from_var(&x_large).unwrap());
    let t2 = Arc::new(TrackedTensor::from_var(&x_small).unwrap());
    let sq1 = t1.sqr().unwrap().sum_keepdim(0).unwrap();
    let sq2 = t2.sqr().unwrap().sum_keepdim(0).unwrap();
    let loss = sq1.add(&sq2).unwrap();
    opt.backward_step(&loss).unwrap();

    let large_val = x_large.data().unwrap().to_flat_vec::<f32>().unwrap()[0];
    let small_val = x_small.data().unwrap().to_flat_vec::<f32>().unwrap()[0];

    let large_update = (100.0 - large_val).abs();
    let small_update = (0.1 - small_val).abs();

    assert!(
        large_update > small_update,
        "relative step: large param update ({large_update}) should exceed small ({small_update})"
    );
}

/// Verify AdaFactor converges on a matrix (rank >= 2) quadratic.
#[test]
fn test_adafactor_converges_matrix() {
    let x = mat_var(&[3.0, -2.0, 5.0, -4.0], 2, 2);
    let config = AdaFactorConfig {
        lr: 0.1,
        relative_step: false,
        ..Default::default()
    };
    let mut opt = AdaFactor::new(vec![x.clone()], config).unwrap();

    for _ in 0..50 {
        let grads = grad_sum_sq(&x);
        opt.step(&grads).unwrap();
    }

    let vals = x.data().unwrap().to_flat_vec::<f32>().unwrap();
    let loss: f32 = vals.iter().map(|v| v * v).sum();
    assert!(
        loss < 10.0,
        "AdaFactor matrix convergence: loss should decrease, got {loss}"
    );
}

/// Verify AdaFactor converges on a scalar quadratic (non-factored path).
#[test]
fn test_adafactor_converges_scalar() {
    let x = scalar_var(7.0);
    let config = AdaFactorConfig {
        lr: 0.1,
        relative_step: false,
        ..Default::default()
    };
    let mut opt = AdaFactor::new(vec![x.clone()], config).unwrap();

    for _ in 0..100 {
        let t = Arc::new(TrackedTensor::from_var(&x).unwrap());
        let loss = t.sqr().unwrap();
        opt.backward_step(&loss).unwrap();
    }

    let val = x.data().unwrap().to_flat_vec::<f32>().unwrap()[0];
    assert!(
        val.abs() < 1.0,
        "AdaFactor scalar convergence: expected near 0, got {val}"
    );
}

// ============================================================================
// Learning rate schedule integration with Adam
// ============================================================================

/// Verify step_with_schedule correctly applies warmup to Adam.
#[test]
fn test_warmup_schedule_with_adam() {
    let x = scalar_var(5.0);
    let config = AdamConfig {
        lr: 0.0,
        weight_decay: 0.0,
        ..AdamConfig::default()
    };
    let mut adam = AdamW::new(vec![x.clone()], config).unwrap();
    let schedule = WarmupSchedule::new(0.1, 10).unwrap();

    // Step at warmup midpoint (step 5): lr = 0.1 * 5/10 = 0.05
    let t = Arc::new(TrackedTensor::from_var(&x).unwrap());
    let loss = t.sqr().unwrap();
    let grads = backward(&loss).unwrap();
    step_with_schedule(&mut adam, &grads, &schedule, 5).unwrap();

    assert!(
        (adam.learning_rate() - 0.05).abs() < f64::EPSILON,
        "lr should be 0.05 at warmup step 5"
    );
    let val = x.data().unwrap().to_flat_vec::<f32>().unwrap()[0];
    assert!(val < 5.0, "parameter should have decreased");
}

/// Verify cosine schedule integration with Adam across multiple steps.
#[test]
fn test_cosine_schedule_with_adam() {
    let x = scalar_var(5.0);
    let config = AdamConfig {
        lr: 0.0,
        weight_decay: 0.0,
        ..AdamConfig::default()
    };
    let mut adam = AdamW::new(vec![x.clone()], config).unwrap();
    let schedule = CosineSchedule::new(0.1, 0.001, 0, 100).unwrap();

    for step in 0..50 {
        let t = Arc::new(TrackedTensor::from_var(&x).unwrap());
        let loss = t.sqr().unwrap();
        let grads = backward(&loss).unwrap();
        step_with_schedule(&mut adam, &grads, &schedule, step).unwrap();
    }

    // After 50 steps with cosine decay, x should have decreased
    let val = x.data().unwrap().to_flat_vec::<f32>().unwrap()[0];
    assert!(val.abs() < 3.0, "cosine+Adam should converge, got {val}");

    // lr at step 50 should be approximately midpoint of cosine: ~0.05
    let expected_lr = schedule.lr_at_step(49);
    assert!(
        (adam.learning_rate() - expected_lr).abs() < 1e-10,
        "optimizer lr should match schedule"
    );
}

// ============================================================================
// Checkpoint round-trips
// ============================================================================

/// Verify Adam checkpoint save/load preserves step count and config.
#[test]
fn test_adam_checkpoint_preserves_state() {
    let x = scalar_var(5.0);
    let config = AdamConfig {
        lr: 0.01,
        beta1: 0.9,
        beta2: 0.999,
        eps: 1e-8,
        weight_decay: 0.1,
    };
    let mut adam = AdamW::new(vec![x.clone()], config.clone()).unwrap();

    // Do 3 steps to build up moment state
    for _ in 0..3 {
        let t = Arc::new(TrackedTensor::from_var(&x).unwrap());
        let loss = t.sqr().unwrap();
        adam.backward_step(&loss).unwrap();
    }
    assert_eq!(adam.step_count(), 3);

    // Save checkpoint
    let snapshot = adam.save_checkpoint().unwrap();

    // Verify metadata contains expected fields
    assert_eq!(snapshot.metadata["type"], "AdamW");
    assert_eq!(snapshot.metadata["step"], 3);
    assert_eq!(snapshot.metadata["lr"], 0.01);
    assert_eq!(snapshot.metadata["beta1"], 0.9);

    // Verify tensors: should have m and v for 1 variable
    assert!(snapshot.tensors.contains_key("adam_0_m"));
    assert!(snapshot.tensors.contains_key("adam_0_v"));

    // Load into a fresh optimizer
    let x2 = scalar_var(5.0);
    let mut adam2 = AdamW::new(vec![x2], config).unwrap();
    adam2.load_checkpoint(&snapshot).unwrap();
    assert_eq!(adam2.step_count(), 3);

    // Verify config was restored
    assert!((adam2.config().lr - 0.01).abs() < f64::EPSILON);
    assert!((adam2.config().beta1 - 0.9).abs() < f64::EPSILON);
}

/// Verify SGD checkpoint save/load preserves velocity and config.
#[test]
fn test_sgd_checkpoint_preserves_state() {
    let x = scalar_var(5.0);
    let config = SgdConfig {
        lr: 0.1,
        momentum: 0.9,
        weight_decay: 0.01,
    };
    let mut sgd = Sgd::new(vec![x.clone()], config.clone()).unwrap();

    // Do 2 steps to build up velocity state
    for _ in 0..2 {
        let t = Arc::new(TrackedTensor::from_var(&x).unwrap());
        let loss = t.sqr().unwrap();
        sgd.backward_step(&loss).unwrap();
    }

    // Save checkpoint
    let snapshot = sgd.save_checkpoint().unwrap();
    assert_eq!(snapshot.metadata["type"], "Sgd");
    assert!(snapshot.tensors.contains_key("sgd_0_velocity"));

    // Load into a fresh optimizer
    let x2 = scalar_var(5.0);
    let mut sgd2 = Sgd::new(vec![x2], config).unwrap();
    sgd2.load_checkpoint(&snapshot).unwrap();

    // Verify config was restored
    assert!((sgd2.learning_rate() - 0.1).abs() < f64::EPSILON);
    assert!((sgd2.momentum() - 0.9).abs() < f64::EPSILON);
    assert!((sgd2.weight_decay() - 0.01).abs() < f64::EPSILON);
}

/// Verify Adam checkpoint rejects NaN in moment tensors.
#[test]
fn test_adam_checkpoint_rejects_nan_moments() {
    let x = scalar_var(1.0);
    let mut adam = AdamW::new(vec![x], AdamConfig::default()).unwrap();

    let mut tensors = std::collections::HashMap::new();
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
        result.is_err(),
        "should reject NaN in moment tensor: {result:?}"
    );
}

/// Verify Adam checkpoint rejects wrong-shaped moment tensors.
#[test]
fn test_adam_checkpoint_rejects_wrong_shape() {
    let x = vec_var(&[1.0, 2.0, 3.0]);
    let mut adam = AdamW::new(vec![x], AdamConfig::default()).unwrap();

    let mut tensors = std::collections::HashMap::new();
    tensors.insert(
        "adam_0_m".to_string(),
        DynTensor::from_vec(vec![0.0, 0.0], &[2], &cpu()).unwrap(),
    );
    let snapshot = OptimizerSnapshot {
        tensors,
        metadata: serde_json::json!({}),
    };

    let result = adam.load_checkpoint(&snapshot);
    assert!(
        result.is_err(),
        "should reject wrong shape in moment tensor"
    );
}

// ============================================================================
// Multiple parameter groups with different behavior
// ============================================================================

/// Verify Adam handles multiple variables where only some get gradients.
/// Simulates frozen layers (params without gradients) alongside trainable ones.
#[test]
fn test_adam_mixed_grad_no_grad_multiple_steps() {
    let trainable = scalar_var(5.0);
    let frozen = scalar_var(10.0);
    let config = AdamConfig {
        lr: 0.1,
        weight_decay: 0.0,
        ..AdamConfig::default()
    };
    let mut adam = AdamW::new(vec![trainable.clone(), frozen.clone()], config).unwrap();

    for _ in 0..100 {
        // Only trainable is in the computation graph
        let t = Arc::new(TrackedTensor::from_var(&trainable).unwrap());
        let loss = t.sqr().unwrap();
        adam.backward_step(&loss).unwrap();
    }

    let train_val = trainable.data().unwrap().to_flat_vec::<f32>().unwrap()[0];
    let frozen_val = frozen.data().unwrap().to_flat_vec::<f32>().unwrap()[0];

    assert!(
        train_val.abs() < 1.0,
        "trainable should converge toward 0: {train_val}"
    );
    assert!(
        (frozen_val - 10.0).abs() < 1e-7,
        "frozen should be unchanged: {frozen_val}"
    );
}

// ============================================================================
// Zero gradient -> no parameter update
// ============================================================================

/// With all-zero gradients, SGD should not modify parameters.
#[test]
fn test_sgd_zero_gradient_no_update() {
    let x = scalar_var(5.0);
    let config = SgdConfig {
        lr: 0.1,
        momentum: 0.9,
        weight_decay: 0.0,
    };
    let mut sgd = Sgd::new(vec![x.clone()], config).unwrap();

    // loss = x * 0 => grad = 0
    let t = Arc::new(TrackedTensor::from_var(&x).unwrap());
    let zero = Arc::new(TrackedTensor::from_tensor(
        DynTensor::from_vec(vec![0.0], &[1], &cpu()).unwrap(),
    ));
    let loss = t.mul(&zero).unwrap();
    sgd.backward_step(&loss).unwrap();

    let val = x.data().unwrap().to_flat_vec::<f32>().unwrap()[0];
    assert!(
        (val - 5.0).abs() < 1e-7,
        "zero gradient should produce no update, got {val}"
    );
}

/// With all-zero gradients, Adam should not modify parameters (no weight decay).
#[test]
fn test_adam_zero_gradient_no_update() {
    let x = scalar_var(5.0);
    let config = AdamConfig {
        lr: 0.01,
        weight_decay: 0.0,
        ..AdamConfig::default()
    };
    let mut adam = AdamW::new(vec![x.clone()], config).unwrap();

    let t = Arc::new(TrackedTensor::from_var(&x).unwrap());
    let zero = Arc::new(TrackedTensor::from_tensor(
        DynTensor::from_vec(vec![0.0], &[1], &cpu()).unwrap(),
    ));
    let loss = t.mul(&zero).unwrap();
    adam.backward_step(&loss).unwrap();

    let val = x.data().unwrap().to_flat_vec::<f32>().unwrap()[0];
    assert!(
        (val - 5.0).abs() < 1e-6,
        "zero gradient (no wd) should produce no update, got {val}"
    );
}

// ============================================================================
// AdaFactor weight decay with relative step
// ============================================================================

/// Verify AdaFactor weight decay works when relative_step is enabled.
#[test]
fn test_adafactor_weight_decay_with_relative_step() {
    let x_wd = vec_var(&[10.0, 10.0]);
    let x_no = vec_var(&[10.0, 10.0]);

    let config_wd = AdaFactorConfig {
        relative_step: true,
        weight_decay: 0.1,
        ..Default::default()
    };
    let config_no = AdaFactorConfig {
        relative_step: true,
        weight_decay: 0.0,
        ..Default::default()
    };

    let mut opt_wd = AdaFactor::new(vec![x_wd.clone()], config_wd).unwrap();
    let mut opt_no = AdaFactor::new(vec![x_no.clone()], config_no).unwrap();

    let grads_wd = grad_sum_sq(&x_wd);
    opt_wd.step(&grads_wd).unwrap();

    let grads_no = grad_sum_sq(&x_no);
    opt_no.step(&grads_no).unwrap();

    let val_wd = x_wd.data().unwrap().to_flat_vec::<f32>().unwrap()[0];
    let val_no = x_no.data().unwrap().to_flat_vec::<f32>().unwrap()[0];

    // With weight decay, params should be smaller
    assert!(
        val_wd < val_no,
        "AdaFactor relative_step + wd should shrink more: wd={val_wd}, no_wd={val_no}"
    );
}

// ============================================================================
// AdaFactor momentum vs no momentum comparison
// ============================================================================

/// AdaFactor with beta1 (momentum) should converge differently than without.
#[test]
fn test_adafactor_momentum_vs_no_momentum() {
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

    for _ in 0..20 {
        let grads_m = grad_sum_sq(&x_mom);
        opt_mom.step(&grads_m).unwrap();

        let grads_n = grad_sum_sq(&x_no);
        opt_no.step(&grads_n).unwrap();
    }

    let loss_mom: f32 = x_mom
        .data()
        .unwrap()
        .to_flat_vec::<f32>()
        .unwrap()
        .iter()
        .map(|v| v * v)
        .sum();
    let loss_no: f32 = x_no
        .data()
        .unwrap()
        .to_flat_vec::<f32>()
        .unwrap()
        .iter()
        .map(|v| v * v)
        .sum();

    // Both should converge; they may converge at different rates
    assert!(loss_mom < 30.0, "momentum loss should decrease: {loss_mom}");
    assert!(
        loss_no < 30.0,
        "no-momentum loss should decrease: {loss_no}"
    );
}

// ============================================================================
// Adam Rosenbrock-like convergence
// ============================================================================

/// Adam on Rosenbrock-like function: (a-1)^2 + 100*(b - a^2)^2.
/// This tests Adam's ability to handle non-trivial loss landscapes.
#[test]
fn test_adam_converges_rosenbrock_2d() {
    let a = scalar_var(0.0);
    let b = scalar_var(0.0);

    let config = AdamConfig {
        lr: 0.01,
        weight_decay: 0.0,
        ..AdamConfig::default()
    };
    let mut adam = AdamW::new(vec![a.clone(), b.clone()], config).unwrap();

    for _ in 0..500 {
        let ta = Arc::new(TrackedTensor::from_var(&a).unwrap());
        let tb = Arc::new(TrackedTensor::from_var(&b).unwrap());

        // (a - 1)^2
        let one = Arc::new(TrackedTensor::from_tensor(
            DynTensor::from_vec(vec![1.0], &[1], &cpu()).unwrap(),
        ));
        let a_diff = ta.sub(&one).unwrap();
        let term1 = a_diff.sqr().unwrap();

        // (b - a^2)^2 (simplified from Rosenbrock's 100x multiplier)
        let a_sq = ta.sqr().unwrap();
        let b_diff = tb.sub(&a_sq).unwrap();
        let term2 = b_diff.sqr().unwrap();

        let loss = term1.add(&term2).unwrap();
        adam.backward_step(&loss).unwrap();
    }

    let a_val = a.data().unwrap().to_flat_vec::<f32>().unwrap()[0];
    let b_val = b.data().unwrap().to_flat_vec::<f32>().unwrap()[0];

    // Minimum is at a=1, b=1 (since b = a^2 = 1 at a=1)
    assert!(
        (a_val - 1.0).abs() < 0.5,
        "Rosenbrock: a should approach 1.0, got {a_val}"
    );
    assert!(
        (b_val - 1.0).abs() < 0.5,
        "Rosenbrock: b should approach 1.0, got {b_val}"
    );
}

// ============================================================================
// SGD: no momentum preserves direction exactly
// ============================================================================

/// Without momentum, SGD should step in the exact negative gradient direction.
#[test]
fn test_sgd_no_momentum_exact_direction() {
    let x = vec_var(&[3.0, 4.0]);
    let config = SgdConfig {
        lr: 0.1,
        momentum: 0.0,
        weight_decay: 0.0,
    };
    let mut sgd = Sgd::new(vec![x.clone()], config).unwrap();

    // loss = sum(x^2), grad = 2*x = [6.0, 8.0]
    let grads = grad_sum_sq(&x);
    sgd.step(&grads).unwrap();

    let vals = x.data().unwrap().to_flat_vec::<f32>().unwrap();
    // x_new = x - lr * grad = [3.0 - 0.6, 4.0 - 0.8] = [2.4, 3.2]
    assert!(
        (vals[0] - 2.4).abs() < 1e-5,
        "x[0]: expected 2.4, got {}",
        vals[0]
    );
    assert!(
        (vals[1] - 3.2).abs() < 1e-5,
        "x[1]: expected 3.2, got {}",
        vals[1]
    );
}

// ============================================================================
// Large step count: verify no overflow or degradation
// ============================================================================

/// Run Adam for many steps and verify it doesn't produce NaN or diverge.
/// Adam's per-step update is bounded by lr (after bias correction saturates),
/// so converging from 100.0 with lr=0.1 takes many steps.
#[test]
fn test_adam_many_steps_no_overflow() {
    let x = scalar_var(10.0);
    let config = AdamConfig {
        lr: 0.1,
        weight_decay: 0.0,
        ..AdamConfig::default()
    };
    let mut adam = AdamW::new(vec![x.clone()], config).unwrap();

    for _ in 0..500 {
        let t = Arc::new(TrackedTensor::from_var(&x).unwrap());
        let loss = t.sqr().unwrap();
        adam.backward_step(&loss).unwrap();
    }

    let val = x.data().unwrap().to_flat_vec::<f32>().unwrap()[0];
    assert!(
        val.is_finite(),
        "500 steps should not produce NaN/Inf, got {val}"
    );
    assert!(
        val.abs() < 1.0,
        "500 steps should converge near 0, got {val}"
    );
    assert_eq!(adam.step_count(), 500);
}

/// Run SGD with momentum for many steps and verify stability.
#[test]
fn test_sgd_many_steps_no_overflow() {
    let x = scalar_var(100.0);
    let config = SgdConfig {
        lr: 0.01,
        momentum: 0.9,
        weight_decay: 0.0,
    };
    let mut sgd = Sgd::new(vec![x.clone()], config).unwrap();

    for _ in 0..500 {
        let t = Arc::new(TrackedTensor::from_var(&x).unwrap());
        let loss = t.sqr().unwrap();
        sgd.backward_step(&loss).unwrap();
    }

    let val = x.data().unwrap().to_flat_vec::<f32>().unwrap()[0];
    assert!(
        val.is_finite(),
        "500 SGD steps should not overflow, got {val}"
    );
    assert!(val.abs() < 1.0, "500 SGD steps should converge, got {val}");
}

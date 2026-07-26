#![allow(deprecated)]
// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Tests for AdamW optimizer.

use std::sync::Arc;

use nn_autodiff::{TrackedTensor, Var};
use nn_core::dyn_tensor::DynTensor;
use nn_core::test_utils::cpu;

use crate::adam::{AdamConfig, AdamW};
use crate::error::OptimError;
use crate::optimizer::Optimizer;

fn scalar_var(val: f32) -> Var {
    Var::new(DynTensor::from_vec(vec![val], &[1], &cpu()).unwrap())
}

// -- AdamConfig defaults (AC4) -----------------------------------------------

#[test]
fn test_adam_config_defaults() {
    let config = AdamConfig::default();
    assert!((config.lr - 1e-3).abs() < f64::EPSILON);
    assert!((config.beta1 - 0.9).abs() < f64::EPSILON);
    assert!((config.beta2 - 0.999).abs() < f64::EPSILON);
    assert!((config.eps - 1e-8).abs() < f64::EPSILON);
    assert!((config.weight_decay - 0.01).abs() < f64::EPSILON);
}

#[test]
fn test_adam_config_custom() {
    let config = AdamConfig {
        lr: 5e-4,
        beta1: 0.95,
        beta2: 0.98,
        eps: 1e-6,
        weight_decay: 0.0,
    };
    assert!((config.lr - 5e-4).abs() < f64::EPSILON);
    assert!((config.beta1 - 0.95).abs() < f64::EPSILON);
}

// -- Basic Adam step --------------------------------------------------------

#[test]
fn test_adam_basic_step() {
    let x = scalar_var(5.0);
    let config = AdamConfig {
        lr: 0.01,
        weight_decay: 0.0,
        ..AdamConfig::default()
    };
    let mut adam = AdamW::new(vec![x.clone()], config).unwrap();

    // loss = x^2, d/dx = 2x = 10
    let t = Arc::new(TrackedTensor::from_var(&x).unwrap());
    let loss = t.sqr().unwrap();
    adam.backward_step(&loss).unwrap();

    let val = x.data().unwrap().to_flat_vec::<f32>().unwrap()[0];
    // After 1 step, x should decrease from 5.0
    assert!(val < 5.0, "Adam should decrease x from 5.0, got {val}");
    assert_eq!(adam.step_count(), 1);
}

// -- Adam convergence on quadratic -------------------------------------------

#[test]
fn test_adam_convergence_quadratic() {
    // Minimize f(x) = x^2. Minimum at x=0.
    let x = scalar_var(5.0);
    let config = AdamConfig {
        lr: 0.1,
        weight_decay: 0.0,
        ..AdamConfig::default()
    };
    let mut adam = AdamW::new(vec![x.clone()], config).unwrap();

    for _ in 0..100 {
        let t = Arc::new(TrackedTensor::from_var(&x).unwrap());
        let loss = t.sqr().unwrap();
        adam.backward_step(&loss).unwrap();
    }

    let val = x.data().unwrap().to_flat_vec::<f32>().unwrap()[0];
    assert!(
        val.abs() < 0.1,
        "Adam should converge to ~0 on x^2, got {val}"
    );
}

// -- Adam convergence on 2D Rosenbrock-like -----------------------------------

#[test]
fn test_adam_convergence_2d() {
    // Minimize f(a,b) = (a-1)^2 + (b-2)^2. Minimum at (1, 2).
    let a = scalar_var(5.0);
    let b = scalar_var(-3.0);

    let config = AdamConfig {
        lr: 0.05,
        weight_decay: 0.0,
        ..AdamConfig::default()
    };
    let mut adam = AdamW::new(vec![a.clone(), b.clone()], config).unwrap();

    for _ in 0..150 {
        let ta = Arc::new(TrackedTensor::from_var(&a).unwrap());
        let tb = Arc::new(TrackedTensor::from_var(&b).unwrap());

        // (a - 1)^2
        let one = Arc::new(TrackedTensor::from_tensor(
            DynTensor::from_vec(vec![1.0], &[1], &cpu()).unwrap(),
        ));
        let two = Arc::new(TrackedTensor::from_tensor(
            DynTensor::from_vec(vec![2.0], &[1], &cpu()).unwrap(),
        ));
        let a_diff = ta.sub(&one).unwrap();
        let b_diff = tb.sub(&two).unwrap();
        let loss = a_diff.sqr().unwrap().add(&b_diff.sqr().unwrap()).unwrap();

        adam.backward_step(&loss).unwrap();
    }

    let a_val = a.data().unwrap().to_flat_vec::<f32>().unwrap()[0];
    let b_val = b.data().unwrap().to_flat_vec::<f32>().unwrap()[0];
    assert!(
        (a_val - 1.0).abs() < 0.5,
        "Adam should approach a=1, got {a_val}"
    );
    assert!(
        (b_val - 2.0).abs() < 0.5,
        "Adam should approach b=2, got {b_val}"
    );
}

// -- Weight decay shrinks weights -------------------------------------------

#[test]
fn test_adam_weight_decay() {
    // With large weight decay and near-zero gradient, weights should shrink.
    let x = scalar_var(10.0);
    let config = AdamConfig {
        lr: 0.01,
        weight_decay: 0.5,
        ..AdamConfig::default()
    };
    let mut adam = AdamW::new(vec![x.clone()], config).unwrap();

    // Small gradient (via small coefficient)
    let t = Arc::new(TrackedTensor::from_var(&x).unwrap());
    let tiny = Arc::new(TrackedTensor::from_tensor(
        DynTensor::from_vec(vec![0.001], &[1], &cpu()).unwrap(),
    ));
    let loss = t.mul(&tiny).unwrap();
    adam.backward_step(&loss).unwrap();

    let val = x.data().unwrap().to_flat_vec::<f32>().unwrap()[0];
    // Weight decay: x *= (1 - lr * wd) = (1 - 0.01 * 0.5) = 0.995
    // So x should be close to 10.0 * 0.995 - update ~ 9.95
    assert!(val < 10.0, "weight decay should reduce x, got {val}");
}

// -- Learning rate accessors ------------------------------------------------

#[test]
fn test_adam_learning_rate() {
    let x = scalar_var(1.0);
    let mut adam = AdamW::new(vec![x], AdamConfig::default()).unwrap();
    assert!((adam.learning_rate() - 1e-3).abs() < f64::EPSILON);

    adam.set_learning_rate(5e-4).unwrap();
    assert!((adam.learning_rate() - 5e-4).abs() < f64::EPSILON);
}

// -- Config accessor --------------------------------------------------------

#[test]
fn test_adam_config_accessor() {
    let x = scalar_var(1.0);
    let config = AdamConfig {
        lr: 2e-3,
        beta1: 0.85,
        ..AdamConfig::default()
    };
    let adam = AdamW::new(vec![x], config).unwrap();
    assert!((adam.config().lr - 2e-3).abs() < f64::EPSILON);
    assert!((adam.config().beta1 - 0.85).abs() < f64::EPSILON);
}

// -- Bias correction effect --------------------------------------------------

#[test]
fn test_adam_bias_correction() {
    // The first step has the largest bias correction effect.
    // With beta1=0.9, bias_correction_1 = 1/(1-0.9) = 10x amplification.
    let x = scalar_var(1.0);
    let config = AdamConfig {
        lr: 0.01,
        weight_decay: 0.0,
        ..AdamConfig::default()
    };
    let mut adam = AdamW::new(vec![x.clone()], config).unwrap();

    let t = Arc::new(TrackedTensor::from_var(&x).unwrap());
    let loss = t.sqr().unwrap();
    adam.backward_step(&loss).unwrap();

    let val = x.data().unwrap().to_flat_vec::<f32>().unwrap()[0];
    // With lr=0.01 and bias correction, the effective step is larger than
    // naive lr=0.01 without correction
    assert!(
        val < 1.0,
        "bias correction should produce a meaningful update, got {val}"
    );
}

// -- Skips vars without gradients -------------------------------------------

#[test]
fn test_adam_skips_no_grad_var() {
    let x = scalar_var(5.0);
    let y = scalar_var(10.0);
    let config = AdamConfig {
        lr: 0.1,
        weight_decay: 0.0,
        ..AdamConfig::default()
    };
    let mut adam = AdamW::new(vec![x.clone(), y.clone()], config).unwrap();

    // Only x in graph
    let t = Arc::new(TrackedTensor::from_var(&x).unwrap());
    let loss = t.sqr().unwrap();
    adam.backward_step(&loss).unwrap();

    let y_val = y.data().unwrap().to_flat_vec::<f32>().unwrap()[0];
    assert!(
        (y_val - 10.0).abs() < 1e-7,
        "y should be unchanged, got {y_val}"
    );
}

// -- Config validation -------------------------------------------------------

#[test]
fn test_adam_rejects_beta1_one() {
    let x = scalar_var(1.0);
    let config = AdamConfig {
        beta1: 1.0,
        ..AdamConfig::default()
    };
    let result = AdamW::new(vec![x], config);
    let err = match result {
        Err(e) => e.to_string(),
        Ok(_) => panic!("beta1=1.0 should be rejected"),
    };
    assert!(err.contains("beta1"), "error should mention beta1: {err}");
}

#[test]
fn test_adam_rejects_beta2_negative() {
    let x = scalar_var(1.0);
    let config = AdamConfig {
        beta2: -0.1,
        ..AdamConfig::default()
    };
    let result = AdamW::new(vec![x], config);
    assert!(result.is_err(), "beta2=-0.1 should be rejected");
}

#[test]
fn test_adam_rejects_eps_zero() {
    let x = scalar_var(1.0);
    let config = AdamConfig {
        eps: 0.0,
        ..AdamConfig::default()
    };
    let result = AdamW::new(vec![x], config);
    let err = match result {
        Err(e) => e.to_string(),
        Ok(_) => panic!("eps=0.0 should be rejected"),
    };
    assert!(err.contains("eps"), "error should mention eps: {err}");
}

#[test]
fn test_adam_rejects_eps_nan() {
    let x = scalar_var(1.0);
    let config = AdamConfig {
        eps: f64::NAN,
        ..AdamConfig::default()
    };
    let result = AdamW::new(vec![x], config);
    let err = match result {
        Err(e) => e.to_string(),
        Ok(_) => panic!("eps=NaN should be rejected"),
    };
    assert!(err.contains("eps"), "error should mention eps: {err}");
}

#[test]
fn test_adam_rejects_eps_inf() {
    let x = scalar_var(1.0);
    let config = AdamConfig {
        eps: f64::INFINITY,
        ..AdamConfig::default()
    };
    let result = AdamW::new(vec![x], config);
    let err = match result {
        Err(e) => e.to_string(),
        Ok(_) => panic!("eps=Inf should be rejected"),
    };
    assert!(err.contains("eps"), "error should mention eps: {err}");
}

// -- NaN gradient rejection ------------------------------------------------

/// AdamW must reject NaN gradients to prevent permanent moment corruption.
#[test]
fn test_adam_rejects_nan_gradient() {
    use nn_autodiff::backward;

    let x = scalar_var(5.0);
    let mut adam = AdamW::new(vec![x.clone()], AdamConfig::default()).unwrap();

    // Compute real grads then inject NaN
    let t = Arc::new(TrackedTensor::from_var(&x).unwrap());
    let loss = t.sqr().unwrap();
    let mut grads = backward(&loss).unwrap();

    for (_, grad) in grads.var_grads_mut() {
        *grad = DynTensor::from_vec(vec![f32::NAN], &[1], &cpu()).unwrap();
    }

    let result = adam.step(&grads);
    assert!(
        matches!(result, Err(OptimError::NonFiniteGradient { .. })),
        "AdamW should reject non-finite gradients, got: {result:?}"
    );
    // Verify weights preserved
    let val = x.data().unwrap().to_flat_vec::<f32>().unwrap()[0];
    assert!(
        (val - 5.0).abs() < 1e-7,
        "weights unchanged after NaN rejection"
    );
}

/// AdamW must reject Inf gradients (matching SGD's test_sgd_rejects_inf_gradient).
#[test]
fn test_adam_rejects_inf_gradient() {
    use nn_autodiff::backward;

    let x = scalar_var(3.0);
    let mut adam = AdamW::new(vec![x.clone()], AdamConfig::default()).unwrap();

    let t = Arc::new(TrackedTensor::from_var(&x).unwrap());
    let loss = t.sqr().unwrap();
    let mut grads = backward(&loss).unwrap();

    for (_, grad) in grads.var_grads_mut() {
        *grad = DynTensor::from_vec(vec![f32::INFINITY], &[1], &cpu()).unwrap();
    }

    let result = adam.step(&grads);
    assert!(
        matches!(result, Err(OptimError::NonFiniteGradient { .. })),
        "AdamW should reject Inf gradients, got: {result:?}"
    );
    // Verify weights preserved
    let val = x.data().unwrap().to_flat_vec::<f32>().unwrap()[0];
    assert!(
        (val - 3.0).abs() < 1e-7,
        "weights unchanged after Inf rejection"
    );
}

// -- Zero learning rate ---------------------------------------------------

/// With lr=0, Adam should produce zero-magnitude parameter updates
/// (weight decay aside — we disable it too).
#[test]
fn test_adam_zero_lr_no_update() {
    let x = scalar_var(5.0);
    let config = AdamConfig {
        lr: 0.0,
        weight_decay: 0.0,
        ..AdamConfig::default()
    };
    let mut adam = AdamW::new(vec![x.clone()], config).unwrap();

    let t = Arc::new(TrackedTensor::from_var(&x).unwrap());
    let loss = t.sqr().unwrap();
    adam.backward_step(&loss).unwrap();

    let val = x.data().unwrap().to_flat_vec::<f32>().unwrap()[0];
    assert!(
        (val - 5.0).abs() < 1e-7,
        "lr=0 should produce no update, got {val}"
    );
}

// -- Empty parameter list -------------------------------------------------

/// AdamW with zero parameters should accept step() without error.
#[test]
fn test_adam_empty_params() {
    use nn_autodiff::backward;

    let x = scalar_var(1.0);
    let mut adam = AdamW::new(vec![], AdamConfig::default()).unwrap();

    // Compute grads that don't reference any optimizer vars
    let t = Arc::new(TrackedTensor::from_var(&x).unwrap());
    let loss = t.sqr().unwrap();
    let grads = backward(&loss).unwrap();

    // step() should succeed — nothing to update
    adam.step(&grads).unwrap();
    assert_eq!(adam.step_count(), 1);
}

#![allow(deprecated)]
// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Tests for SGD optimizer.

use std::sync::Arc;

use nn_autodiff::{backward, TrackedTensor, Var};
use nn_core::dyn_tensor::DynTensor;
use nn_core::test_utils::cpu;

use crate::error::OptimError;
use crate::optimizer::Optimizer;
use crate::sgd::{Sgd, SgdConfig};

fn scalar_var(val: f32) -> Var {
    Var::new(DynTensor::from_vec(vec![val], &[1], &cpu()).unwrap())
}

// -- Basic SGD step -------------------------------------------------------

#[test]
fn test_sgd_basic_step() {
    // theta=5.0, grad=2.0, lr=0.1 → theta_new = 5.0 - 0.1*2.0 = 4.8
    let x = scalar_var(5.0);
    let mut sgd = Sgd::new(
        vec![x.clone()],
        SgdConfig {
            lr: 0.1,
            ..Default::default()
        },
    )
    .unwrap();

    // Construct a gradient manually: y = 2*x, dy/dx = 2
    let t = Arc::new(TrackedTensor::from_var(&x).unwrap());
    let two = Arc::new(TrackedTensor::from_tensor(
        DynTensor::from_vec(vec![2.0], &[1], &cpu()).unwrap(),
    ));
    let y = t.mul(&two).unwrap();
    let grads = backward(&y).unwrap();

    sgd.step(&grads).unwrap();

    let new_val = x.data().unwrap().to_flat_vec::<f32>().unwrap()[0];
    assert!((new_val - 4.8).abs() < 1e-6, "expected 4.8, got {new_val}");
}

#[test]
fn test_sgd_learning_rate_accessor() {
    let x = scalar_var(1.0);
    let sgd = Sgd::new(
        vec![x],
        SgdConfig {
            lr: 0.01,
            ..Default::default()
        },
    )
    .unwrap();
    assert!((sgd.learning_rate() - 0.01).abs() < f64::EPSILON);
}

#[test]
fn test_sgd_set_learning_rate() {
    let x = scalar_var(1.0);
    let mut sgd = Sgd::new(
        vec![x],
        SgdConfig {
            lr: 0.01,
            ..Default::default()
        },
    )
    .unwrap();
    sgd.set_learning_rate(0.05).unwrap();
    assert!((sgd.learning_rate() - 0.05).abs() < f64::EPSILON);
}

// -- SGD convergence: minimize y = x^2 (minimum at x=0) --------------------

#[test]
fn test_sgd_convergence_quadratic() {
    // f(x) = x^2, f'(x) = 2x. Minimum at x=0.
    let x = scalar_var(3.0);
    let mut sgd = Sgd::new(
        vec![x.clone()],
        SgdConfig {
            lr: 0.1,
            ..Default::default()
        },
    )
    .unwrap();

    for _ in 0..50 {
        let t = Arc::new(TrackedTensor::from_var(&x).unwrap());
        let loss = t.sqr().unwrap();
        sgd.backward_step(&loss).unwrap();
    }

    let final_val = x.data().unwrap().to_flat_vec::<f32>().unwrap()[0];
    assert!(
        final_val.abs() < 0.01,
        "SGD should converge to ~0 on x^2, got {final_val}"
    );
}

// -- Momentum accelerates convergence ---------------------------------------

#[test]
fn test_sgd_momentum() {
    // With momentum, SGD should converge faster than without.
    let x_no_mom = scalar_var(5.0);
    let x_mom = scalar_var(5.0);

    let mut sgd_no = Sgd::new(
        vec![x_no_mom.clone()],
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

    let steps = 30;
    for _ in 0..steps {
        // No momentum
        let t = Arc::new(TrackedTensor::from_var(&x_no_mom).unwrap());
        let loss = t.sqr().unwrap();
        sgd_no.backward_step(&loss).unwrap();

        // With momentum
        let t = Arc::new(TrackedTensor::from_var(&x_mom).unwrap());
        let loss = t.sqr().unwrap();
        sgd_mom.backward_step(&loss).unwrap();
    }

    let val_no = x_no_mom.data().unwrap().to_flat_vec::<f32>().unwrap()[0].abs();
    let val_mom = x_mom.data().unwrap().to_flat_vec::<f32>().unwrap()[0].abs();

    assert!(
        val_mom < val_no,
        "momentum should converge faster: no_mom={val_no}, mom={val_mom}"
    );
}

// -- Weight decay shrinks weights -------------------------------------------

#[test]
fn test_sgd_weight_decay() {
    // With weight decay and zero gradient, weights should shrink toward zero.
    let x = scalar_var(2.0);
    let mut sgd = Sgd::new(
        vec![x.clone()],
        SgdConfig {
            lr: 0.1,
            weight_decay: 0.1,
            ..Default::default()
        },
    )
    .unwrap();

    // Create a zero gradient: y = 0 * x (grad = 0, but weight_decay adds wd * theta)
    let t = Arc::new(TrackedTensor::from_var(&x).unwrap());
    let zero = Arc::new(TrackedTensor::from_tensor(
        DynTensor::from_vec(vec![0.0], &[1], &cpu()).unwrap(),
    ));
    let y = t.mul(&zero).unwrap();
    let grads = backward(&y).unwrap();

    let before = x.data().unwrap().to_flat_vec::<f32>().unwrap()[0];
    sgd.step(&grads).unwrap();
    let after = x.data().unwrap().to_flat_vec::<f32>().unwrap()[0];

    assert!(
        after.abs() < before.abs(),
        "weight decay should shrink weights: before={before}, after={after}"
    );
}

// -- Multiple variables -----------------------------------------------------

#[test]
fn test_sgd_multiple_vars() {
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

    // loss = a^2 + b^2, d/da = 2a, d/db = 2b
    let ta = Arc::new(TrackedTensor::from_var(&a).unwrap());
    let tb = Arc::new(TrackedTensor::from_var(&b).unwrap());
    let a_sq = ta.sqr().unwrap();
    let b_sq = tb.sqr().unwrap();
    let loss = a_sq.add(&b_sq).unwrap();

    sgd.backward_step(&loss).unwrap();

    let new_a = a.data().unwrap().to_flat_vec::<f32>().unwrap()[0];
    let new_b = b.data().unwrap().to_flat_vec::<f32>().unwrap()[0];

    // a_new = 3.0 - 0.1 * 2 * 3.0 = 3.0 - 0.6 = 2.4
    assert!((new_a - 2.4).abs() < 1e-5, "expected ~2.4, got {new_a}");
    // b_new = 4.0 - 0.1 * 2 * 4.0 = 4.0 - 0.8 = 3.2
    assert!((new_b - 3.2).abs() < 1e-5, "expected ~3.2, got {new_b}");
}

// -- Config struct ---------------------------------------------------------

#[test]
fn test_sgd_config() {
    let x = scalar_var(1.0);
    let sgd = Sgd::new(
        vec![x],
        SgdConfig {
            lr: 0.01,
            momentum: 0.9,
            weight_decay: 1e-4,
        },
    )
    .unwrap();
    assert!((sgd.momentum() - 0.9).abs() < f64::EPSILON);
    assert!((sgd.weight_decay() - 1e-4).abs() < f64::EPSILON);
    assert!((sgd.learning_rate() - 0.01).abs() < f64::EPSILON);
}

// -- Skips vars without gradients -------------------------------------------

#[test]
fn test_sgd_skips_no_grad_var() {
    let x = scalar_var(5.0);
    let y = scalar_var(10.0);
    let mut sgd = Sgd::new(
        vec![x.clone(), y.clone()],
        SgdConfig {
            lr: 0.1,
            ..Default::default()
        },
    )
    .unwrap();

    // Only compute gradient for x: loss = x^2 (y not in graph)
    let t = Arc::new(TrackedTensor::from_var(&x).unwrap());
    let loss = t.sqr().unwrap();
    sgd.backward_step(&loss).unwrap();

    let x_val = x.data().unwrap().to_flat_vec::<f32>().unwrap()[0];
    let y_val = y.data().unwrap().to_flat_vec::<f32>().unwrap()[0];

    // x should change: 5.0 - 0.1 * 2 * 5.0 = 4.0
    assert!((x_val - 4.0).abs() < 1e-5, "x should be updated");
    // y should remain unchanged
    assert!((y_val - 10.0).abs() < 1e-7, "y should be unchanged");
}

// -- NaN gradient rejection ------------------------------------------------

/// SGD must reject NaN gradients instead of silently corrupting weights.
#[test]
fn test_sgd_rejects_nan_gradient() {
    let x = scalar_var(5.0);
    let mut sgd = Sgd::new(
        vec![x.clone()],
        SgdConfig {
            lr: 0.1,
            ..Default::default()
        },
    )
    .unwrap();

    // Compute real grads then inject NaN
    let t = Arc::new(TrackedTensor::from_var(&x).unwrap());
    let loss = t.sqr().unwrap();
    let mut grads = backward(&loss).unwrap();

    for (_, grad) in grads.var_grads_mut() {
        *grad = DynTensor::from_vec(vec![f32::NAN], &[1], &cpu()).unwrap();
    }

    let result = sgd.step(&grads);
    assert!(
        matches!(result, Err(OptimError::NonFiniteGradient { .. })),
        "SGD should reject non-finite gradients, got: {result:?}"
    );
    // Verify weights were NOT corrupted
    let val = x.data().unwrap().to_flat_vec::<f32>().unwrap()[0];
    assert!(
        (val - 5.0).abs() < 1e-7,
        "weights should be unchanged after NaN rejection"
    );
}

/// SGD must reject Inf gradients.
#[test]
fn test_sgd_rejects_inf_gradient() {
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
    assert!(matches!(result, Err(OptimError::NonFiniteGradient { .. })));
}

// -- Checkpoint shape validation with None velocity --------------------------

/// Loading a wrong-shaped velocity checkpoint before the first step must be
/// rejected. Before the fix, the `None` velocity path skipped shape validation
/// entirely, allowing shape-mismatched tensors to be loaded.
#[test]
fn test_sgd_checkpoint_wrong_shape_before_first_step() {
    use crate::checkpoint::{OptimizerCheckpoint, OptimizerSnapshot};
    use std::collections::HashMap;

    let var = Var::new(DynTensor::from_vec(vec![1.0, 2.0, 3.0], &[3], &cpu()).unwrap());
    let config = SgdConfig {
        lr: 0.01,
        momentum: 0.9,
        ..Default::default()
    };
    let mut sgd = Sgd::new(vec![var], config).unwrap();

    // No step taken — velocity is None.
    // Construct a snapshot with wrong-shaped velocity [2] instead of [3].
    let mut tensors = HashMap::new();
    tensors.insert(
        "sgd_0_velocity".to_string(),
        DynTensor::from_vec(vec![1.0, 2.0], &[2], &cpu()).unwrap(),
    );
    let snapshot = OptimizerSnapshot {
        tensors,
        metadata: serde_json::json!({}),
    };

    let result = sgd.load_checkpoint(&snapshot);
    assert!(
        matches!(result, Err(OptimError::CheckpointShapeMismatch { .. })),
        "should reject wrong-shaped velocity before first step, got: {result:?}"
    );
}

// -- Zero learning rate ---------------------------------------------------

/// With lr=0 and no weight_decay, SGD should produce zero-magnitude updates.
#[test]
fn test_sgd_zero_lr_no_update() {
    let x = scalar_var(5.0);
    let mut sgd = Sgd::new(
        vec![x.clone()],
        SgdConfig {
            lr: 0.0,
            ..Default::default()
        },
    )
    .unwrap();

    let t = Arc::new(TrackedTensor::from_var(&x).unwrap());
    let loss = t.sqr().unwrap();
    sgd.backward_step(&loss).unwrap();

    let val = x.data().unwrap().to_flat_vec::<f32>().unwrap()[0];
    assert!(
        (val - 5.0).abs() < 1e-7,
        "lr=0 should produce no update, got {val}"
    );
}

// -- Empty parameter list -------------------------------------------------

/// SGD with zero parameters should accept step() without error.
#[test]
fn test_sgd_empty_params() {
    let x = scalar_var(1.0);
    let mut sgd = Sgd::new(vec![], SgdConfig::default()).unwrap();

    let t = Arc::new(TrackedTensor::from_var(&x).unwrap());
    let loss = t.sqr().unwrap();
    let grads = backward(&loss).unwrap();

    // step() should succeed — nothing to update
    sgd.step(&grads).unwrap();
}

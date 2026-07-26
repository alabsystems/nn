#![allow(deprecated)]
// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Config restoration and behavioral equivalence tests for checkpoint save/load.

use std::sync::Arc;

use nn_autodiff::{TrackedTensor, Var};
use nn_core::dyn_tensor::DynTensor;
use nn_core::test_utils::cpu;

use crate::checkpoint::OptimizerCheckpoint;
use crate::{AdaFactor, AdaFactorConfig, AdamConfig, AdamW, Optimizer, Sgd, SgdConfig};

fn scalar_loss(v: &Var) -> Arc<TrackedTensor> {
    let t = Arc::new(TrackedTensor::from_var(v).unwrap());
    t.sqr().unwrap().sum_keepdim(0).unwrap()
}

// ---------------------------------------------------------------------------
// Config restoration: AdamW
// Verify load_checkpoint restores config fields from saved metadata.
// ---------------------------------------------------------------------------

#[test]
fn test_adamw_checkpoint_restores_config() {
    let v = Var::new(DynTensor::from_vec(vec![1.0, 2.0], &[2], &cpu()).unwrap());
    let config = AdamConfig {
        lr: 0.005,
        beta1: 0.85,
        beta2: 0.995,
        eps: 1e-6,
        weight_decay: 0.05,
    };
    let mut adam = AdamW::new(vec![v.clone()], config).unwrap();
    let loss = scalar_loss(&v);
    adam.backward_step(&loss).unwrap();

    let snapshot = adam.save_checkpoint().unwrap();

    // Load into optimizer with DEFAULT config — config should be overwritten.
    let mut adam2 = AdamW::new(vec![v], AdamConfig::default()).unwrap();
    adam2.load_checkpoint(&snapshot).unwrap();

    let c = adam2.config();
    assert!((c.lr - 0.005).abs() < 1e-10, "lr not restored: {}", c.lr);
    assert!(
        (c.beta1 - 0.85).abs() < 1e-10,
        "beta1 not restored: {}",
        c.beta1
    );
    assert!(
        (c.beta2 - 0.995).abs() < 1e-10,
        "beta2 not restored: {}",
        c.beta2
    );
    assert!((c.eps - 1e-6).abs() < 1e-15, "eps not restored: {}", c.eps);
    assert!(
        (c.weight_decay - 0.05).abs() < 1e-10,
        "weight_decay not restored: {}",
        c.weight_decay
    );
}

// ---------------------------------------------------------------------------
// Config restoration: SGD
// ---------------------------------------------------------------------------

#[test]
fn test_sgd_checkpoint_restores_config() {
    let v = Var::new(DynTensor::from_vec(vec![1.0, 2.0], &[2], &cpu()).unwrap());
    let config = SgdConfig {
        lr: 0.05,
        momentum: 0.95,
        weight_decay: 0.001,
    };
    let mut sgd = Sgd::new(vec![v.clone()], config).unwrap();

    let loss = scalar_loss(&v);
    sgd.backward_step(&loss).unwrap();
    let snapshot = sgd.save_checkpoint().unwrap();

    // Load into SGD with different config — saved config should overwrite.
    let config2 = SgdConfig {
        lr: 0.01,
        momentum: 0.9,
        ..SgdConfig::default()
    };
    let mut sgd2 = Sgd::new(vec![v], config2).unwrap();
    sgd2.load_checkpoint(&snapshot).unwrap();

    assert!(
        (sgd2.learning_rate() - 0.05).abs() < 1e-10,
        "lr not restored: {}",
        sgd2.learning_rate()
    );
    assert!(
        (sgd2.momentum() - 0.95).abs() < 1e-10,
        "momentum not restored: {}",
        sgd2.momentum()
    );
    assert!(
        (sgd2.weight_decay() - 0.001).abs() < 1e-10,
        "weight_decay not restored: {}",
        sgd2.weight_decay()
    );
}

// ---------------------------------------------------------------------------
// Config restoration: AdaFactor
// ---------------------------------------------------------------------------

#[test]
fn test_adafactor_checkpoint_restores_config() {
    let v = Var::new(DynTensor::from_vec(vec![1.0, 2.0, 3.0], &[3], &cpu()).unwrap());
    let config = AdaFactorConfig {
        lr: 0.005,
        relative_step: false,
        eps_rms: 1e-25,
        eps_denom: 1e-28,
        decay_rate: -0.7,
        beta1: Some(0.85),
        weight_decay: 0.02,
    };
    let mut ada = AdaFactor::new(vec![v.clone()], config).unwrap();
    let loss = scalar_loss(&v);
    ada.backward_step(&loss).unwrap();
    let snapshot = ada.save_checkpoint().unwrap();

    // Load into optimizer with default config — all fields should be overwritten.
    let mut ada2 = AdaFactor::new(vec![v], AdaFactorConfig::default()).unwrap();
    ada2.load_checkpoint(&snapshot).unwrap();

    let c = ada2.config();
    assert!((c.lr - 0.005).abs() < 1e-10, "lr not restored: {}", c.lr);
    assert!(!c.relative_step, "relative_step not restored");
    assert!(
        (c.eps_rms - 1e-25).abs() < 1e-30,
        "eps_rms not restored: {}",
        c.eps_rms
    );
    assert!(
        (c.eps_denom - 1e-28).abs() < 1e-33,
        "eps_denom not restored: {}",
        c.eps_denom
    );
    assert!(
        (c.decay_rate - (-0.7)).abs() < 1e-10,
        "decay_rate not restored: {}",
        c.decay_rate
    );
    assert_eq!(c.beta1, Some(0.85), "beta1 not restored: {:?}", c.beta1);
    assert!(
        (c.weight_decay - 0.02).abs() < 1e-10,
        "weight_decay not restored: {}",
        c.weight_decay
    );
}

// ---------------------------------------------------------------------------
// Config restoration: AdaFactor with beta1=None preserved
// ---------------------------------------------------------------------------

#[test]
fn test_adafactor_checkpoint_restores_beta1_none() {
    let v = Var::new(DynTensor::from_vec(vec![1.0, 2.0], &[2], &cpu()).unwrap());
    let config = AdaFactorConfig {
        beta1: None,
        ..AdaFactorConfig::default()
    };
    let mut ada = AdaFactor::new(vec![v.clone()], config).unwrap();
    let loss = scalar_loss(&v);
    ada.backward_step(&loss).unwrap();
    let snapshot = ada.save_checkpoint().unwrap();

    // Load into optimizer initialized with Some(0.9).
    // After load, beta1 should become None (matching saved state).
    let config2 = AdaFactorConfig {
        beta1: Some(0.9),
        ..AdaFactorConfig::default()
    };
    let mut ada2 = AdaFactor::new(vec![v], config2).unwrap();
    ada2.load_checkpoint(&snapshot).unwrap();
    assert_eq!(ada2.config().beta1, None, "beta1 should be None after load");
}

// ---------------------------------------------------------------------------
// Behavioral equivalence: restored optimizer produces identical weights
// ---------------------------------------------------------------------------

/// After checkpoint restore, one more training step must produce the same
/// weight update as the original optimizer. This catches moment tensor value
/// corruption that structural checks (step count, tensor count) miss.
///
/// Critical: the restored optimizer must NOT re-train before loading, since
/// identical re-training would produce the same moments anyway, making the
/// test vacuously true (load_checkpoint could be a no-op and still pass).
/// Instead, we set the weight values to match via Var::set, load the
/// checkpoint into a fresh (zero-moment) optimizer, and verify that the
/// loaded moments produce the same weight update as the original.
#[test]
fn test_adamw_checkpoint_behavioral_equivalence() {
    // Train original optimizer for 5 steps.
    let v_orig = Var::new(DynTensor::from_vec(vec![1.0, 2.0, 3.0], &[3], &cpu()).unwrap());
    let mut adam_orig = AdamW::new(vec![v_orig.clone()], AdamConfig::default()).unwrap();
    for _ in 0..5 {
        let loss = scalar_loss(&v_orig);
        adam_orig.backward_step(&loss).unwrap();
    }

    // Save checkpoint and record current weight values.
    let snapshot = adam_orig.save_checkpoint().unwrap();
    let weights_at_step5 = v_orig.data().unwrap().to_flat_vec::<f32>().unwrap();

    // Take one more step on original.
    let loss = scalar_loss(&v_orig);
    adam_orig.backward_step(&loss).unwrap();
    let weights_orig = v_orig.data().unwrap().to_flat_vec::<f32>().unwrap();

    // Create fresh optimizer with zero moments (step 0).
    // Set weight values to match step-5 state WITHOUT re-training,
    // so the moments are genuinely different (zeros vs accumulated).
    let v_restored = Var::new(DynTensor::from_vec(vec![1.0, 2.0, 3.0], &[3], &cpu()).unwrap());
    v_restored
        .set(&DynTensor::from_vec(weights_at_step5, &[3], &cpu()).unwrap())
        .unwrap();
    let mut adam_restored = AdamW::new(vec![v_restored.clone()], AdamConfig::default()).unwrap();

    // Load checkpoint — this MUST overwrite zero moments with saved values.
    adam_restored.load_checkpoint(&snapshot).unwrap();

    // Take one more step on restored — should match original.
    let loss = scalar_loss(&v_restored);
    adam_restored.backward_step(&loss).unwrap();
    let weights_restored = v_restored.data().unwrap().to_flat_vec::<f32>().unwrap();

    for (i, (&orig, &restored)) in weights_orig.iter().zip(weights_restored.iter()).enumerate() {
        assert!(
            (orig - restored).abs() < 1e-6,
            "weight[{i}]: orig={orig}, restored={restored} — moment values \
             were not correctly restored by checkpoint load"
        );
    }
}

// -- AC5: set_learning_rate(NaN) returns error --------------------------------

#[test]
fn test_set_learning_rate_nan_rejected() {
    let v = Var::new(DynTensor::from_vec(vec![1.0], &[1], &cpu()).unwrap());
    let mut adam = AdamW::new(vec![v.clone()], AdamConfig::default()).unwrap();
    assert!(adam.set_learning_rate(f64::NAN).is_err());

    let mut sgd = Sgd::new(
        vec![v.clone()],
        SgdConfig {
            lr: 0.01,
            ..SgdConfig::default()
        },
    )
    .unwrap();
    assert!(sgd.set_learning_rate(f64::NAN).is_err());

    let mut adafactor = AdaFactor::new(vec![v], AdaFactorConfig::default()).unwrap();
    assert!(adafactor.set_learning_rate(f64::NAN).is_err());
}

#[test]
fn test_set_learning_rate_neg_inf_rejected() {
    let v = Var::new(DynTensor::from_vec(vec![1.0], &[1], &cpu()).unwrap());
    let mut adam = AdamW::new(vec![v], AdamConfig::default()).unwrap();
    assert!(adam.set_learning_rate(f64::NEG_INFINITY).is_err());
}

#[test]
fn test_set_learning_rate_negative_rejected() {
    let v = Var::new(DynTensor::from_vec(vec![1.0], &[1], &cpu()).unwrap());
    let mut sgd = Sgd::new(
        vec![v],
        SgdConfig {
            lr: 0.01,
            ..SgdConfig::default()
        },
    )
    .unwrap();
    assert!(sgd.set_learning_rate(-0.1).is_err());
}

// -- AC6: checkpoint with beta1=2.0 returns error -----------------------------

#[test]
fn test_adam_checkpoint_invalid_beta1_rejected() {
    let v = Var::new(DynTensor::from_vec(vec![1.0], &[1], &cpu()).unwrap());
    let mut adam = AdamW::new(vec![v], AdamConfig::default()).unwrap();
    let mut snapshot = adam.save_checkpoint().unwrap();
    // Corrupt beta1 to 2.0 (outside [0, 1))
    snapshot.metadata["beta1"] = serde_json::json!(2.0);
    let result = adam.load_checkpoint(&snapshot);
    assert!(
        result.is_err(),
        "beta1=2.0 should be rejected by load_checkpoint"
    );
}

#[test]
fn test_adam_checkpoint_negative_lr_rejected() {
    let v = Var::new(DynTensor::from_vec(vec![1.0], &[1], &cpu()).unwrap());
    let mut adam = AdamW::new(vec![v], AdamConfig::default()).unwrap();
    let mut snapshot = adam.save_checkpoint().unwrap();
    // JSON cannot represent NaN/Inf (they serialize as null), so test with
    // a negative lr which JSON round-trips correctly.
    snapshot.metadata["lr"] = serde_json::json!(-0.01);
    let result = adam.load_checkpoint(&snapshot);
    assert!(
        result.is_err(),
        "negative lr should be rejected by load_checkpoint"
    );
}

#[test]
fn test_sgd_checkpoint_invalid_momentum_rejected() {
    let v = Var::new(DynTensor::from_vec(vec![1.0], &[1], &cpu()).unwrap());
    let config = SgdConfig {
        lr: 0.01,
        momentum: 0.9,
        ..SgdConfig::default()
    };
    let mut sgd = Sgd::new(vec![v], config).unwrap();
    let mut snapshot = sgd.save_checkpoint().unwrap();
    // Corrupt momentum to negative
    snapshot.metadata["momentum"] = serde_json::json!(-1.0);
    let result = sgd.load_checkpoint(&snapshot);
    assert!(result.is_err(), "negative momentum should be rejected");
}

#[test]
fn test_adafactor_checkpoint_invalid_eps_rejected() {
    let v = Var::new(DynTensor::from_vec(vec![1.0], &[1], &cpu()).unwrap());
    let mut af = AdaFactor::new(vec![v], AdaFactorConfig::default()).unwrap();
    let mut snapshot = af.save_checkpoint().unwrap();
    // Corrupt eps_denom to 0.0 (must be positive)
    snapshot.metadata["eps_denom"] = serde_json::json!(0.0);
    let result = af.load_checkpoint(&snapshot);
    assert!(result.is_err(), "eps_denom=0.0 should be rejected");
}

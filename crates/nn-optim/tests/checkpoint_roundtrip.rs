// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Cross-crate integration tests for optimizer checkpoint save/load.
//!
//! Tests verify that optimizer state (moments, config, step counter) survives
//! a save/load cycle, and that corrupted or mismatched checkpoints are rejected.

use std::sync::Arc;

use nn_autodiff::{TrackedTensor, Var};
use nn_core::dyn_tensor::DynTensor;
use nn_core::Device;
use nn_optim::checkpoint::OptimizerCheckpoint;
use nn_optim::{AdaFactor, AdaFactorConfig, AdamConfig, AdamW, Optimizer, Sgd, SgdConfig};

fn cpu() -> Device {
    Device::Cpu
}

fn scalar_var(val: f32) -> Var {
    Var::new(DynTensor::from_vec(vec![val], &[1], &cpu()).unwrap())
}

fn read_scalar(var: &Var) -> f32 {
    var.data().unwrap().to_flat_vec::<f32>().unwrap()[0]
}

fn step_quadratic(opt: &mut dyn Optimizer, var: &Var) {
    let t = Arc::new(TrackedTensor::from_var(var).unwrap());
    let loss = t.sqr().unwrap();
    opt.backward_step(&loss).unwrap();
}

fn adam_config(lr: f64, weight_decay: f64) -> AdamConfig {
    let mut c = AdamConfig::default();
    c.lr = lr;
    c.weight_decay = weight_decay;
    c
}

fn sgd_config_mom(lr: f64, momentum: f64) -> SgdConfig {
    let mut c = SgdConfig::default();
    c.lr = lr;
    c.momentum = momentum;
    c.weight_decay = 0.0;
    c
}

fn adafactor_config_beta1(lr: f64) -> AdaFactorConfig {
    let mut c = AdaFactorConfig::default();
    c.lr = lr;
    c.relative_step = false;
    c.weight_decay = 0.0;
    c.beta1 = Some(0.9);
    c
}

fn adafactor_config_simple(lr: f64) -> AdaFactorConfig {
    let mut c = AdaFactorConfig::default();
    c.lr = lr;
    c.relative_step = false;
    c.weight_decay = 0.0;
    c
}

// ============================================================================
// Adam checkpoint roundtrip
// ============================================================================

#[test]
fn test_adam_checkpoint_roundtrip() {
    let x = scalar_var(5.0);
    let config = adam_config(0.05, 0.0);
    let mut adam = AdamW::new(vec![x.clone()], config).unwrap();

    // Train for 10 steps to build up moment state
    for _ in 0..10 {
        step_quadratic(&mut adam, &x);
    }

    let val_after_10 = read_scalar(&x);

    // Save checkpoint
    let snapshot = adam.save_checkpoint().unwrap();

    // Verify metadata roundtrips correctly
    assert_eq!(
        snapshot.metadata.get("type").and_then(|v| v.as_str()),
        Some("AdamW")
    );
    assert_eq!(
        snapshot.metadata.get("step").and_then(serde_json::value::Value::as_u64),
        Some(10)
    );

    // Create a fresh optimizer and load the snapshot
    let x2 = scalar_var(val_after_10);
    let mut adam2 = AdamW::new(vec![x2], adam_config(0.05, 0.0)).unwrap();
    adam2.load_checkpoint(&snapshot).unwrap();

    assert_eq!(adam2.step_count(), 10);

    // Verify moment tensors were restored
    assert!(
        !snapshot.tensors.is_empty(),
        "Snapshot should contain moment tensors"
    );
    assert!(
        snapshot.tensors.contains_key("adam_0_m"),
        "Snapshot should have first moment"
    );
    assert!(
        snapshot.tensors.contains_key("adam_0_v"),
        "Snapshot should have second moment"
    );
}

// ============================================================================
// SGD checkpoint roundtrip
// ============================================================================

#[test]
fn test_sgd_checkpoint_roundtrip() {
    let x = scalar_var(5.0);
    let config = sgd_config_mom(0.05, 0.9);
    let mut sgd = Sgd::new(vec![x.clone()], config).unwrap();

    for _ in 0..10 {
        step_quadratic(&mut sgd, &x);
    }

    let snapshot = sgd.save_checkpoint().unwrap();

    // Verify metadata
    assert_eq!(
        snapshot.metadata.get("type").and_then(|v| v.as_str()),
        Some("Sgd")
    );

    // Create fresh optimizer and load
    let val = read_scalar(&x);
    let x2 = scalar_var(val);
    let mut sgd2 = Sgd::new(vec![x2], sgd_config_mom(0.05, 0.9)).unwrap();
    sgd2.load_checkpoint(&snapshot).unwrap();

    // Velocity tensor should have been restored
    assert!(
        snapshot.tensors.contains_key("sgd_0_velocity"),
        "Snapshot should have velocity tensor"
    );
}

// ============================================================================
// AdaFactor checkpoint roundtrip
// ============================================================================

#[test]
fn test_adafactor_checkpoint_roundtrip() {
    let x = scalar_var(5.0);
    let config = adafactor_config_beta1(0.05);
    let mut af = AdaFactor::new(vec![x.clone()], config).unwrap();

    for _ in 0..10 {
        step_quadratic(&mut af, &x);
    }

    let snapshot = af.save_checkpoint().unwrap();

    assert_eq!(
        snapshot.metadata.get("type").and_then(|v| v.as_str()),
        Some("AdaFactor")
    );
    assert_eq!(
        snapshot.metadata.get("step").and_then(serde_json::value::Value::as_u64),
        Some(10)
    );

    // Scalar var has rank < 2, so uses full second moment (not factored)
    assert!(
        snapshot.tensors.contains_key("adafactor_0_v_full"),
        "Scalar var should have full second moment"
    );

    // Load into fresh optimizer
    let val = read_scalar(&x);
    let x2 = scalar_var(val);
    let mut af2 = AdaFactor::new(vec![x2], adafactor_config_beta1(0.05)).unwrap();
    af2.load_checkpoint(&snapshot).unwrap();
    assert_eq!(af2.step_count(), 10);
}

#[test]
fn test_adafactor_checkpoint_roundtrip_matrix() {
    // Matrix var uses factored second moments
    let data = DynTensor::from_vec(vec![1.0f32; 16], &[4, 4], &cpu()).unwrap();
    let w = Var::new(data);
    let config = adafactor_config_simple(0.05);
    let mut af = AdaFactor::new(vec![w.clone()], config).unwrap();

    for _ in 0..5 {
        let t = Arc::new(TrackedTensor::from_var(&w).unwrap());
        let sq = t.sqr().unwrap();
        let loss = sq.sum_keepdim(0).unwrap().sum_keepdim(1).unwrap();
        let loss = loss.reshape(&[1]).unwrap();
        af.backward_step(&loss).unwrap();
    }

    let snapshot = af.save_checkpoint().unwrap();

    // Matrix should have row/col factors, not full second moment
    assert!(
        snapshot.tensors.contains_key("adafactor_0_row"),
        "Matrix var should have row factor"
    );
    assert!(
        snapshot.tensors.contains_key("adafactor_0_col"),
        "Matrix var should have col factor"
    );

    // Load into fresh optimizer
    let data2 = w.data().unwrap();
    let w2 = Var::new(data2);
    let mut af2 = AdaFactor::new(vec![w2], adafactor_config_simple(0.05)).unwrap();
    af2.load_checkpoint(&snapshot).unwrap();
    assert_eq!(af2.step_count(), 5);
}

// ============================================================================
// Checkpoint continue training: save -> load -> continue == uninterrupted
// ============================================================================

#[test]
fn test_checkpoint_continue_training_adam() {
    let config = adam_config(0.05, 0.0);

    // Train 20 steps uninterrupted.
    let x_ref = scalar_var(5.0);
    let mut adam_ref = AdamW::new(vec![x_ref.clone()], config.clone()).unwrap();
    for _ in 0..20 {
        step_quadratic(&mut adam_ref, &x_ref);
    }
    let ref_val = read_scalar(&x_ref);

    // Train 10, save, load into new optimizer, train 10 more.
    let x_split = scalar_var(5.0);
    let mut adam_split = AdamW::new(vec![x_split.clone()], config.clone()).unwrap();
    for _ in 0..10 {
        step_quadratic(&mut adam_split, &x_split);
    }

    let snapshot = adam_split.save_checkpoint().unwrap();
    let split_val_at_10 = read_scalar(&x_split);

    // Load into fresh optimizer with same variable state
    let x_cont = scalar_var(split_val_at_10);
    let mut adam_cont = AdamW::new(vec![x_cont.clone()], config).unwrap();
    adam_cont.load_checkpoint(&snapshot).unwrap();

    for _ in 0..10 {
        step_quadratic(&mut adam_cont, &x_cont);
    }
    let cont_val = read_scalar(&x_cont);

    // Should be very close to uninterrupted training
    assert!(
        (cont_val - ref_val).abs() < 0.05,
        "Continued training should match uninterrupted: ref={ref_val}, continued={cont_val}"
    );
}

// ============================================================================
// Checkpoint validation: shape mismatch
// ============================================================================

#[test]
fn test_checkpoint_shape_mismatch_adam() {
    // Create a snapshot from a [1]-shaped var
    let x1 = scalar_var(5.0);
    let mut adam1 = AdamW::new(vec![x1.clone()], AdamConfig::default()).unwrap();
    step_quadratic(&mut adam1, &x1);
    let snapshot = adam1.save_checkpoint().unwrap();

    // Try to load into optimizer with [2]-shaped var
    let x2 = Var::new(DynTensor::from_vec(vec![1.0, 2.0], &[2], &cpu()).unwrap());
    let mut adam2 = AdamW::new(vec![x2], AdamConfig::default()).unwrap();
    let result = adam2.load_checkpoint(&snapshot);

    assert!(
        result.is_err(),
        "Loading checkpoint with mismatched shapes should fail"
    );
}

#[test]
fn test_checkpoint_shape_mismatch_sgd() {
    let x1 = scalar_var(5.0);
    let mut sgd1 = Sgd::new(vec![x1.clone()], sgd_config_mom(0.01, 0.9)).unwrap();
    step_quadratic(&mut sgd1, &x1);
    let snapshot = sgd1.save_checkpoint().unwrap();

    let x2 = Var::new(DynTensor::from_vec(vec![1.0, 2.0], &[2], &cpu()).unwrap());
    let mut sgd2 = Sgd::new(vec![x2], sgd_config_mom(0.01, 0.9)).unwrap();
    let result = sgd2.load_checkpoint(&snapshot);

    assert!(
        result.is_err(),
        "SGD checkpoint with mismatched shapes should fail"
    );
}

// ============================================================================
// Checkpoint validation: non-finite tensors
// ============================================================================

#[test]
fn test_checkpoint_rejects_nan_moment() {
    // Save a valid checkpoint, then corrupt a moment tensor with NaN
    let x1 = scalar_var(5.0);
    let mut adam1 = AdamW::new(vec![x1.clone()], AdamConfig::default()).unwrap();
    step_quadratic(&mut adam1, &x1);
    let mut snapshot = adam1.save_checkpoint().unwrap();

    // Replace the first moment tensor with a NaN tensor
    let nan_tensor = DynTensor::from_vec(vec![f32::NAN], &[1], &cpu()).unwrap();
    snapshot.tensors.insert("adam_0_m".to_string(), nan_tensor);

    // Loading the corrupted snapshot should fail
    let x2 = scalar_var(1.0);
    let mut adam2 = AdamW::new(vec![x2], AdamConfig::default()).unwrap();
    let result = adam2.load_checkpoint(&snapshot);

    assert!(
        result.is_err(),
        "Loading checkpoint with NaN moments should fail"
    );
}

// ============================================================================
// Checkpoint metadata validation
// ============================================================================

#[test]
fn test_checkpoint_preserves_config_adam() {
    let x = scalar_var(5.0);
    let mut config = AdamConfig::default();
    config.lr = 0.005;
    config.beta1 = 0.85;
    config.beta2 = 0.98;
    config.eps = 1e-6;
    config.weight_decay = 0.05;
    let mut adam = AdamW::new(vec![x.clone()], config).unwrap();
    step_quadratic(&mut adam, &x);

    let snapshot = adam.save_checkpoint().unwrap();

    // Verify config values are in metadata
    let meta = &snapshot.metadata;
    assert_eq!(meta.get("lr").and_then(serde_json::value::Value::as_f64), Some(0.005));
    assert_eq!(meta.get("beta1").and_then(serde_json::value::Value::as_f64), Some(0.85));
    assert_eq!(meta.get("beta2").and_then(serde_json::value::Value::as_f64), Some(0.98));
    assert_eq!(meta.get("eps").and_then(serde_json::value::Value::as_f64), Some(1e-6));
    assert_eq!(
        meta.get("weight_decay").and_then(serde_json::value::Value::as_f64),
        Some(0.05)
    );
}

#[test]
fn test_checkpoint_preserves_config_sgd() {
    let x = scalar_var(5.0);
    let mut config = SgdConfig::default();
    config.lr = 0.02;
    config.momentum = 0.95;
    config.weight_decay = 0.03;
    let mut sgd = Sgd::new(vec![x.clone()], config).unwrap();
    step_quadratic(&mut sgd, &x);

    let snapshot = sgd.save_checkpoint().unwrap();
    let meta = &snapshot.metadata;
    assert_eq!(meta.get("lr").and_then(serde_json::value::Value::as_f64), Some(0.02));
    assert_eq!(meta.get("momentum").and_then(serde_json::value::Value::as_f64), Some(0.95));
    assert_eq!(
        meta.get("weight_decay").and_then(serde_json::value::Value::as_f64),
        Some(0.03)
    );
}

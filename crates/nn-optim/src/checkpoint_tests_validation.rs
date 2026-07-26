// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Checkpoint finiteness and shape validation tests.
//!
//! Extracted from `checkpoint_tests.rs` for 500-line compliance.
//! Tests that optimizer checkpoints correctly reject NaN/Inf moment tensors
//! and shape-mismatched state on load.

use std::sync::Arc;

use nn_autodiff::{TrackedTensor, Var};
use nn_core::dyn_tensor::DynTensor;
use nn_core::test_utils::cpu;

use crate::checkpoint::OptimizerCheckpoint;
use crate::{AdaFactor, AdaFactorConfig, AdamConfig, AdamW, Optimizer, Sgd, SgdConfig};

// ---------------------------------------------------------------------------
// Checkpoint finiteness validation — NaN/Inf moment tensors rejected on load
// ---------------------------------------------------------------------------

/// Adam rejects checkpoint with NaN in first moment.
#[test]
fn test_adamw_checkpoint_rejects_nan_moment() {
    let v = Var::new(DynTensor::from_vec(vec![1.0, 2.0], &[2], &cpu()).unwrap());
    let mut adam = AdamW::new(vec![v.clone()], AdamConfig::default()).unwrap();

    let t = Arc::new(TrackedTensor::from_var(&v).unwrap());
    let loss = t.sqr().unwrap().sum_keepdim(0).unwrap();
    adam.backward_step(&loss).unwrap();

    let mut snapshot = adam.save_checkpoint().unwrap();
    // Inject NaN into first moment
    snapshot.tensors.insert(
        "adam_0_m".to_string(),
        DynTensor::from_vec(vec![f32::NAN, 1.0], &[2], &cpu()).unwrap(),
    );

    let mut adam2 = AdamW::new(vec![v], AdamConfig::default()).unwrap();
    let err = adam2.load_checkpoint(&snapshot).unwrap_err();
    let msg = format!("{err}");
    assert!(
        msg.contains("non-finite") && msg.contains("adam_0_m"),
        "expected non-finite error for adam_0_m, got: {msg}"
    );
}

/// Adam rejects checkpoint with Inf in second moment.
#[test]
fn test_adamw_checkpoint_rejects_inf_moment() {
    let v = Var::new(DynTensor::from_vec(vec![1.0, 2.0], &[2], &cpu()).unwrap());
    let mut adam = AdamW::new(vec![v.clone()], AdamConfig::default()).unwrap();

    let t = Arc::new(TrackedTensor::from_var(&v).unwrap());
    let loss = t.sqr().unwrap().sum_keepdim(0).unwrap();
    adam.backward_step(&loss).unwrap();

    let mut snapshot = adam.save_checkpoint().unwrap();
    snapshot.tensors.insert(
        "adam_0_v".to_string(),
        DynTensor::from_vec(vec![f32::INFINITY, 1.0], &[2], &cpu()).unwrap(),
    );

    let mut adam2 = AdamW::new(vec![v], AdamConfig::default()).unwrap();
    let err = adam2.load_checkpoint(&snapshot).unwrap_err();
    let msg = format!("{err}");
    assert!(
        msg.contains("non-finite") && msg.contains("adam_0_v"),
        "expected non-finite error for adam_0_v, got: {msg}"
    );
}

/// SGD rejects checkpoint with NaN in velocity tensor.
#[test]
fn test_sgd_checkpoint_rejects_nan_velocity() {
    let v = Var::new(DynTensor::from_vec(vec![1.0, 2.0], &[2], &cpu()).unwrap());
    let config = SgdConfig {
        lr: 0.01,
        momentum: 0.9,
        ..Default::default()
    };
    let mut sgd = Sgd::new(vec![v.clone()], config.clone()).unwrap();

    let t = Arc::new(TrackedTensor::from_var(&v).unwrap());
    let loss = t.sqr().unwrap().sum_keepdim(0).unwrap();
    sgd.backward_step(&loss).unwrap();

    let mut snapshot = sgd.save_checkpoint().unwrap();
    snapshot.tensors.insert(
        "sgd_0_velocity".to_string(),
        DynTensor::from_vec(vec![f32::NAN, f32::NEG_INFINITY], &[2], &cpu()).unwrap(),
    );

    let mut sgd2 = Sgd::new(vec![v], config).unwrap();
    let err = sgd2.load_checkpoint(&snapshot).unwrap_err();
    let msg = format!("{err}");
    assert!(
        msg.contains("non-finite") && msg.contains("sgd_0_velocity"),
        "expected non-finite error for sgd_0_velocity, got: {msg}"
    );
}

// ---------------------------------------------------------------------------
// AdaFactor None-slot matrix shape mismatch (row/col factor validation)
// ---------------------------------------------------------------------------

/// Verify that loading a checkpoint into a fresh (None-slot) optimizer with a
/// different-shaped matrix variable rejects mismatched row_factor and col_factor.
///
/// This specifically tests the `expected_dims` path in `restore_tensor()` added
/// by W2-98 (0b4e39e9), which validates shape even when the optimizer state
/// hasn't been initialized (slots are None).
#[test]
fn test_adafactor_checkpoint_matrix_none_slot_shape_mismatch() {
    // Train with a [2, 3] matrix to produce row_factor [2, 1] and col_factor [1, 3].
    let v =
        Var::new(DynTensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], &cpu()).unwrap());
    let config = AdaFactorConfig {
        beta1: Some(0.9),
        ..AdaFactorConfig::default()
    };
    let mut opt = AdaFactor::new(vec![v.clone()], config.clone()).unwrap();

    let t = Arc::new(TrackedTensor::from_var(&v).unwrap());
    let sq = t.sqr().unwrap();
    let loss = sq.sum_keepdim(0).unwrap().sum_keepdim(1).unwrap();
    opt.backward_step(&loss).unwrap();

    let snapshot = opt.save_checkpoint().unwrap();
    // Verify the snapshot has row and col factors.
    assert!(snapshot.tensors.contains_key("adafactor_0_row"));
    assert!(snapshot.tensors.contains_key("adafactor_0_col"));

    // Load into fresh optimizer with a [3, 4] matrix (different shape).
    // Fresh optimizer has None slots — the expected_dims validation must catch
    // the mismatch: saved row_factor [2, 1] vs expected [3, 1].
    let v2 = Var::new(
        DynTensor::from_vec((0..12).map(|i| i as f32).collect(), &[3, 4], &cpu()).unwrap(),
    );
    let mut opt2 = AdaFactor::new(vec![v2], config).unwrap();
    let err = opt2.load_checkpoint(&snapshot).unwrap_err();
    let msg = format!("{err}");
    assert!(
        msg.contains("shape mismatch"),
        "expected shape mismatch on matrix None-slot load, got: {msg}"
    );
}

// ---------------------------------------------------------------------------
// AdaFactor checkpoint — NaN/Inf rejection
// ---------------------------------------------------------------------------

/// AdaFactor rejects checkpoint with NaN in row factor.
#[test]
fn test_adafactor_checkpoint_rejects_nan_factor() {
    let v =
        Var::new(DynTensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], &cpu()).unwrap());
    let config = AdaFactorConfig::default();
    let mut opt = AdaFactor::new(vec![v.clone()], config.clone()).unwrap();

    let t = Arc::new(TrackedTensor::from_var(&v).unwrap());
    let sq = t.sqr().unwrap();
    let loss = sq.sum_keepdim(0).unwrap().sum_keepdim(1).unwrap();
    opt.backward_step(&loss).unwrap();

    let mut snapshot = opt.save_checkpoint().unwrap();
    // row_factor for [2,3] has shape [2,1]
    snapshot.tensors.insert(
        "adafactor_0_row".to_string(),
        DynTensor::from_vec(vec![f32::NAN, 1.0], &[2, 1], &cpu()).unwrap(),
    );

    let mut opt2 = AdaFactor::new(vec![v], config).unwrap();
    let err = opt2.load_checkpoint(&snapshot).unwrap_err();
    let msg = format!("{err}");
    assert!(
        msg.contains("non-finite") && msg.contains("adafactor_0_row"),
        "expected non-finite error for adafactor_0_row, got: {msg}"
    );
}

#![allow(deprecated)]
// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

use std::sync::Arc;

use nn_autodiff::{TrackedTensor, Var, VarMap};
use nn_core::dyn_tensor::DynTensor;
use nn_core::test_utils::cpu;
use nn_core::DType;

use crate::checkpoint::{OptimizerCheckpoint, TrainingCheckpoint, TrainingMetadata};
use crate::{AdaFactor, AdaFactorConfig, AdamConfig, AdamW, Optimizer, Sgd, SgdConfig};

#[path = "checkpoint_tests_config.rs"]
mod config_tests;

#[path = "checkpoint_exact_tests.rs"]
mod exact_tests;

#[path = "checkpoint_tests_validation.rs"]
mod validation_tests;

#[path = "checkpoint_state_tests.rs"]
mod state_tests;

#[path = "checkpoint_roundtrip_tests.rs"]
mod roundtrip_tests;

// ---------------------------------------------------------------------------
// AdamW checkpoint round-trip
// ---------------------------------------------------------------------------

#[test]
fn test_adamw_checkpoint_roundtrip() {
    let v1 = Var::new(DynTensor::from_vec(vec![1.0, 2.0], &[2], &cpu()).unwrap());
    let v2 = Var::new(DynTensor::from_vec(vec![3.0], &[1], &cpu()).unwrap());

    let mut adam = AdamW::new(vec![v1.clone(), v2.clone()], AdamConfig::default()).unwrap();

    // Train 5 steps to accumulate moment state.
    for _ in 0..5 {
        let t = Arc::new(TrackedTensor::from_var(&v1).unwrap());
        let loss = t.sqr().unwrap().sum_keepdim(0).unwrap();
        adam.backward_step(&loss).unwrap();
    }

    let snapshot = adam.save_checkpoint().unwrap();
    assert_eq!(snapshot.metadata["step"], 5);
    assert_eq!(snapshot.tensors.len(), 4); // 2 vars × (m + v)

    // Create new optimizer and load state.
    let mut adam2 = AdamW::new(vec![v1, v2], AdamConfig::default()).unwrap();
    assert_eq!(adam2.step_count(), 0);
    adam2.load_checkpoint(&snapshot).unwrap();
    assert_eq!(adam2.step_count(), 5);
}

// ---------------------------------------------------------------------------
// SGD checkpoint round-trip
// ---------------------------------------------------------------------------

#[test]
fn test_sgd_checkpoint_roundtrip() {
    let v = Var::new(DynTensor::from_vec(vec![1.0, 2.0, 3.0], &[3], &cpu()).unwrap());
    let config = SgdConfig {
        lr: 0.01,
        momentum: 0.9,
        ..Default::default()
    };
    let mut sgd = Sgd::new(vec![v.clone()], config.clone()).unwrap();

    // Train a step to populate velocity.
    let t = Arc::new(TrackedTensor::from_var(&v).unwrap());
    let loss = t.sqr().unwrap().sum_keepdim(0).unwrap();
    sgd.backward_step(&loss).unwrap();

    let snapshot = sgd.save_checkpoint().unwrap();
    assert_eq!(snapshot.metadata["type"], "Sgd");
    assert_eq!(snapshot.tensors.len(), 1); // 1 var with velocity

    // Restore into fresh SGD.
    let mut sgd2 = Sgd::new(vec![v], config).unwrap();
    sgd2.load_checkpoint(&snapshot).unwrap();
}

// ---------------------------------------------------------------------------
// Full TrainingCheckpoint save/load
// ---------------------------------------------------------------------------

#[test]
fn test_training_checkpoint_full_roundtrip() {
    let mut map = VarMap::new();
    let w = map.get("weight", &[2, 3], DType::F32, &cpu()).unwrap();
    let b = map.get("bias", &[3], DType::F32, &cpu()).unwrap();

    // Set some non-zero weights.
    w.set(&DynTensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], &cpu()).unwrap())
        .unwrap();
    b.set(&DynTensor::from_vec(vec![0.1, 0.2, 0.3], &[3], &cpu()).unwrap())
        .unwrap();

    let mut adam = AdamW::new(vec![w.clone(), b.clone()], AdamConfig::default()).unwrap();

    // Train a few steps.
    for _ in 0..3 {
        let t = Arc::new(TrackedTensor::from_var(&w).unwrap());
        let loss = t
            .sqr()
            .unwrap()
            .sum_keepdim(0)
            .unwrap()
            .sum_keepdim(1)
            .unwrap();
        adam.backward_step(&loss).unwrap();
    }

    let dir = std::env::temp_dir().join(format!("nn_ckpt_test_{}", std::process::id()));
    let metadata = TrainingMetadata {
        step: 3,
        lr: adam.learning_rate(),
        grad_scaler: None,
        extra: None,
    };
    TrainingCheckpoint::save(&dir, &map, &adam, &metadata).unwrap();

    // Verify files were created.
    assert!(dir.join("model.safetensors").exists());
    assert!(dir.join("optimizer.safetensors").exists());
    assert!(dir.join("training_state.json").exists());

    // Load into fresh state.
    let mut map2 = VarMap::new();
    map2.get("weight", &[2, 3], DType::F32, &cpu()).unwrap();
    map2.get("bias", &[3], DType::F32, &cpu()).unwrap();

    let mut adam2 = AdamW::new(vec![w, b], AdamConfig::default()).unwrap();

    let loaded_meta = TrainingCheckpoint::load(&dir, &mut map2, &mut adam2).unwrap();
    assert_eq!(loaded_meta.step, 3);
    assert_eq!(adam2.step_count(), 3);

    // Verify weights were restored.
    let tensors = map2.to_tensors().unwrap();
    assert_eq!(tensors["weight"].to_flat_vec::<f32>().unwrap().len(), 6);

    std::fs::remove_dir_all(&dir).ok();
}

// ---------------------------------------------------------------------------
// AdamW shape mismatch on load
// ---------------------------------------------------------------------------

#[test]
fn test_adamw_checkpoint_shape_mismatch() {
    let v = Var::new(DynTensor::from_vec(vec![1.0, 2.0], &[2], &cpu()).unwrap());
    let mut adam = AdamW::new(vec![v.clone()], AdamConfig::default()).unwrap();

    // Train 1 step.
    let t = Arc::new(TrackedTensor::from_var(&v).unwrap());
    let loss = t.sqr().unwrap().sum_keepdim(0).unwrap();
    adam.backward_step(&loss).unwrap();

    let snapshot = adam.save_checkpoint().unwrap();

    // Try loading into optimizer with different shape.
    let v3 = Var::new(DynTensor::from_vec(vec![1.0, 2.0, 3.0], &[3], &cpu()).unwrap());
    let mut adam2 = AdamW::new(vec![v3], AdamConfig::default()).unwrap();
    let err = adam2.load_checkpoint(&snapshot).unwrap_err();
    let msg = format!("{err}");
    assert!(
        msg.contains("shape mismatch"),
        "expected shape mismatch, got: {msg}"
    );
}

// ---------------------------------------------------------------------------
// AdaFactor checkpoint round-trip — vector (rank < 2, full second moment)
// ---------------------------------------------------------------------------

#[test]
fn test_adafactor_checkpoint_roundtrip_vector() {
    let v = Var::new(DynTensor::from_vec(vec![1.0, 2.0, 3.0], &[3], &cpu()).unwrap());
    let config = AdaFactorConfig {
        beta1: Some(0.9),
        ..AdaFactorConfig::default()
    };
    let mut opt = AdaFactor::new(vec![v.clone()], config.clone()).unwrap();

    // Train 3 steps to build moment state.
    for _ in 0..3 {
        let t = Arc::new(TrackedTensor::from_var(&v).unwrap());
        let loss = t.sqr().unwrap().sum_keepdim(0).unwrap();
        opt.backward_step(&loss).unwrap();
    }

    let snapshot = opt.save_checkpoint().unwrap();
    assert_eq!(snapshot.metadata["type"], "AdaFactor");
    assert_eq!(snapshot.metadata["step"], 3);
    // vector: 1 first_moment + 1 second_moment_full = 2 tensors
    assert_eq!(snapshot.tensors.len(), 2);
    assert!(snapshot.tensors.contains_key("adafactor_0_m"));
    assert!(snapshot.tensors.contains_key("adafactor_0_v_full"));

    // Load into fresh optimizer.
    let mut opt2 = AdaFactor::new(vec![v], config).unwrap();
    assert_eq!(opt2.step_count(), 0);
    opt2.load_checkpoint(&snapshot).unwrap();
    assert_eq!(opt2.step_count(), 3);
}

// ---------------------------------------------------------------------------
// AdaFactor checkpoint round-trip — matrix (rank >= 2, factored row/col)
// ---------------------------------------------------------------------------

#[test]
fn test_adafactor_checkpoint_roundtrip_matrix() {
    let v =
        Var::new(DynTensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], &cpu()).unwrap());
    let config = AdaFactorConfig {
        beta1: Some(0.9),
        ..AdaFactorConfig::default()
    };
    let mut opt = AdaFactor::new(vec![v.clone()], config.clone()).unwrap();

    // Train 3 steps.
    for _ in 0..3 {
        let t = Arc::new(TrackedTensor::from_var(&v).unwrap());
        let sq = t.sqr().unwrap();
        let loss = sq.sum_keepdim(0).unwrap().sum_keepdim(1).unwrap();
        opt.backward_step(&loss).unwrap();
    }

    let snapshot = opt.save_checkpoint().unwrap();
    assert_eq!(snapshot.metadata["step"], 3);
    // matrix: 1 first_moment + 1 row_factor + 1 col_factor = 3 tensors
    assert_eq!(snapshot.tensors.len(), 3);
    assert!(snapshot.tensors.contains_key("adafactor_0_m"));
    assert!(snapshot.tensors.contains_key("adafactor_0_row"));
    assert!(snapshot.tensors.contains_key("adafactor_0_col"));
    assert!(!snapshot.tensors.contains_key("adafactor_0_v_full"));

    // Load into fresh optimizer.
    let mut opt2 = AdaFactor::new(vec![v], config).unwrap();
    assert_eq!(opt2.step_count(), 0);
    opt2.load_checkpoint(&snapshot).unwrap();
    assert_eq!(opt2.step_count(), 3);
}

// ---------------------------------------------------------------------------
// AdaFactor checkpoint — no beta1 (no first moment saved)
// ---------------------------------------------------------------------------

#[test]
fn test_adafactor_checkpoint_no_beta1() {
    let v = Var::new(DynTensor::from_vec(vec![1.0, 2.0], &[2], &cpu()).unwrap());
    let config = AdaFactorConfig {
        beta1: None,
        ..AdaFactorConfig::default()
    };
    let mut opt = AdaFactor::new(vec![v.clone()], config.clone()).unwrap();

    let t = Arc::new(TrackedTensor::from_var(&v).unwrap());
    let loss = t.sqr().unwrap().sum_keepdim(0).unwrap();
    opt.backward_step(&loss).unwrap();

    let snapshot = opt.save_checkpoint().unwrap();
    // No first moment: only second_moment_full = 1 tensor
    assert_eq!(snapshot.tensors.len(), 1);
    assert!(snapshot.tensors.contains_key("adafactor_0_v_full"));
    assert!(!snapshot.tensors.contains_key("adafactor_0_m"));

    let mut opt2 = AdaFactor::new(vec![v], config).unwrap();
    opt2.load_checkpoint(&snapshot).unwrap();
    assert_eq!(opt2.step_count(), 1);
}

// ---------------------------------------------------------------------------
// AdaFactor checkpoint shape mismatch
// ---------------------------------------------------------------------------

#[test]
fn test_adafactor_checkpoint_shape_mismatch() {
    let v = Var::new(DynTensor::from_vec(vec![1.0, 2.0], &[2], &cpu()).unwrap());
    let config = AdaFactorConfig::default();
    let mut opt = AdaFactor::new(vec![v.clone()], config.clone()).unwrap();

    let t = Arc::new(TrackedTensor::from_var(&v).unwrap());
    let loss = t.sqr().unwrap().sum_keepdim(0).unwrap();
    opt.backward_step(&loss).unwrap();

    let snapshot = opt.save_checkpoint().unwrap();

    // Try loading into optimizer with different shape.
    let v3 = Var::new(DynTensor::from_vec(vec![1.0, 2.0, 3.0], &[3], &cpu()).unwrap());
    let mut opt2 = AdaFactor::new(vec![v3], config).unwrap();
    let err = opt2.load_checkpoint(&snapshot).unwrap_err();
    let msg = format!("{err}");
    assert!(
        msg.contains("shape mismatch"),
        "expected shape mismatch, got: {msg}"
    );
}

// ---------------------------------------------------------------------------
// SGD checkpoint — no momentum (no velocity tensors)
// ---------------------------------------------------------------------------

#[test]
fn test_sgd_checkpoint_no_momentum() {
    let v = Var::new(DynTensor::from_vec(vec![1.0, 2.0], &[2], &cpu()).unwrap());
    let config = SgdConfig {
        lr: 0.01,
        ..Default::default()
    };
    let mut sgd = Sgd::new(vec![v.clone()], config.clone()).unwrap();

    let t = Arc::new(TrackedTensor::from_var(&v).unwrap());
    let loss = t.sqr().unwrap().sum_keepdim(0).unwrap();
    sgd.backward_step(&loss).unwrap();

    let snapshot = sgd.save_checkpoint().unwrap();
    // No momentum → no velocity tensors saved.
    assert_eq!(snapshot.tensors.len(), 0);
    assert_eq!(snapshot.metadata["momentum"], 0.0);

    let mut sgd2 = Sgd::new(vec![v], config).unwrap();
    sgd2.load_checkpoint(&snapshot).unwrap();
}

// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Training pipeline checkpoint/resume and TrainableModule integration tests.
//!
//! Split from `training_pipeline.rs` for 500-line compliance (#1544 D10).
//!
//! Run: `cargo test -p nn --test training_pipeline_checkpoint --features training`

#![cfg(feature = "training")]

use std::sync::Arc;

use nn::training::{
    backward, AdaFactor, AdaFactorConfig, AdamW, Optimizer, Sgd, SgdConfig, TrackedTensor,
    TrainableLoraLinear, TrainableModule, TrainingCheckpoint, TrainingMetadata, Var, VarMap,
};
use nn::{DType, Device, Linear};

use super::common::{adam_config_no_wd, make_data, make_weight, train_step};

// ── Checkpoint save/load/resume ──────────────────────────────────────

/// Create a VarMap with 4 named vars (w1, b1, w2, b2).
fn make_varmap_vars(
    in_dim: usize,
    hidden: usize,
    num_classes: usize,
    init_weights: bool,
) -> (VarMap, Var, Var, Var, Var) {
    let mut var_map = VarMap::new();
    let w1 = var_map
        .get("w1", &[hidden, in_dim], DType::F32, &Device::Cpu)
        .unwrap();
    let b1 = var_map
        .get("b1", &[1, hidden], DType::F32, &Device::Cpu)
        .unwrap();
    let w2 = var_map
        .get("w2", &[num_classes, hidden], DType::F32, &Device::Cpu)
        .unwrap();
    let b2 = var_map
        .get("b2", &[1, num_classes], DType::F32, &Device::Cpu)
        .unwrap();
    if init_weights {
        w1.set(&make_weight(hidden, in_dim, 17)).unwrap();
        w2.set(&make_weight(num_classes, hidden, 23)).unwrap();
    }
    (var_map, w1, b1, w2, b2)
}

/// Collect cloned vars as a Vec for optimizer construction.
fn var_vec(w1: &Var, b1: &Var, w2: &Var, b2: &Var) -> Vec<Var> {
    vec![w1.clone(), b1.clone(), w2.clone(), b2.clone()]
}

/// Create a VarMap with 4 named vars (w1, b1, w2, b2) and an AdamW optimizer.
fn make_varmap_mlp(
    in_dim: usize,
    hidden: usize,
    num_classes: usize,
    init_weights: bool,
) -> (VarMap, Var, Var, Var, Var, AdamW) {
    let (var_map, w1, b1, w2, b2) = make_varmap_vars(in_dim, hidden, num_classes, init_weights);
    let adam = AdamW::new(var_vec(&w1, &b1, &w2, &b2), adam_config_no_wd()).unwrap();
    (var_map, w1, b1, w2, b2, adam)
}

/// Train 5 steps -> save checkpoint -> load into fresh optimizer -> train 5 more.
/// Verifies complete checkpoint round-trip preserves optimizer state.
#[test]
fn test_checkpoint_save_load_resume() {
    let (in_dim, hidden, num_classes) = (4, 8, 3);
    let (x_data, t_data) = make_data(12, in_dim, num_classes);

    // Phase 1: train 5 steps with VarMap-managed variables
    let (var_map, w1, b1, w2, b2, mut adam) = make_varmap_mlp(in_dim, hidden, num_classes, true);
    let mut losses = Vec::new();
    for _ in 0..5 {
        let (loss, loss_val) = train_step(&x_data, &t_data, &w1, &b1, &w2, &b2);
        losses.push(loss_val);
        adam.backward_step(&loss).unwrap();
    }
    let loss_at_save = *losses.last().unwrap();

    // Save checkpoint
    let dir = std::env::temp_dir().join(format!("nn_ckpt_test_{}", std::process::id()));
    let metadata: TrainingMetadata = serde_json::from_value(serde_json::json!({
        "step": 5, "lr": adam.learning_rate(),
    }))
    .unwrap();
    TrainingCheckpoint::save(&dir, &var_map, &adam, &metadata).unwrap();

    // Phase 2: fresh VarMap + optimizer, load checkpoint, continue training
    let (mut var_map2, w1b, b1b, w2b, b2b, mut adam2) =
        make_varmap_mlp(in_dim, hidden, num_classes, false);
    let loaded_meta = TrainingCheckpoint::load(&dir, &mut var_map2, &mut adam2).unwrap();
    assert_eq!(loaded_meta.step, 5);

    for _ in 0..5 {
        let (loss, loss_val) = train_step(&x_data, &t_data, &w1b, &b1b, &w2b, &b2b);
        losses.push(loss_val);
        adam2.backward_step(&loss).unwrap();
    }

    // Loss after loading should be close to loss at save (tolerance for optimizer state)
    let loss_after_load = losses[5];
    assert!(
        (loss_after_load - loss_at_save).abs() < 0.5,
        "loss after load vs save: {loss_after_load} vs {loss_at_save}",
    );
    assert!(
        *losses.last().unwrap() < losses[0],
        "checkpoint resume: loss should decrease: {} -> {}",
        losses[0],
        losses.last().unwrap(),
    );
    let _ = std::fs::remove_dir_all(&dir);
}

// ── SGD checkpoint round-trip ────────────────────────────────────────

/// SGD checkpoint round-trip: train → save → load into fresh SGD → resume training.
#[test]
fn test_sgd_checkpoint_save_load_resume() {
    let (in_dim, hidden, num_classes) = (4, 8, 3);
    let (x_data, t_data) = make_data(12, in_dim, num_classes);

    let mut sgd_config = SgdConfig::default();
    sgd_config.lr = 0.05;
    sgd_config.momentum = 0.9;

    // Phase 1: train 5 steps
    let (var_map, w1, b1, w2, b2) = make_varmap_vars(in_dim, hidden, num_classes, true);
    let mut sgd = Sgd::new(var_vec(&w1, &b1, &w2, &b2), sgd_config.clone()).unwrap();
    let mut losses = Vec::new();
    for _ in 0..5 {
        let (loss, loss_val) = train_step(&x_data, &t_data, &w1, &b1, &w2, &b2);
        losses.push(loss_val);
        sgd.backward_step(&loss).unwrap();
    }
    // Save checkpoint
    let dir = std::env::temp_dir().join(format!("nn_sgd_ckpt_{}", std::process::id()));
    let metadata: TrainingMetadata = serde_json::from_value(serde_json::json!({
        "step": 5, "lr": sgd.learning_rate(),
    }))
    .unwrap();
    TrainingCheckpoint::save(&dir, &var_map, &sgd, &metadata).unwrap();

    // Phase 2: fresh VarMap + SGD, load checkpoint, resume
    let (mut var_map2, w1b, b1b, w2b, b2b) = make_varmap_vars(in_dim, hidden, num_classes, false);
    let mut sgd2 = Sgd::new(var_vec(&w1b, &b1b, &w2b, &b2b), sgd_config).unwrap();
    let loaded_meta = TrainingCheckpoint::load(&dir, &mut var_map2, &mut sgd2).unwrap();
    assert_eq!(loaded_meta.step, 5);

    for _ in 0..5 {
        let (loss, loss_val) = train_step(&x_data, &t_data, &w1b, &b1b, &w2b, &b2b);
        losses.push(loss_val);
        sgd2.backward_step(&loss).unwrap();
    }

    // Verify checkpoint restored state: loss after loading should be lower
    // than initial loss (training progress preserved across save/load).
    let loss_after_load = losses[5];
    assert!(
        loss_after_load < losses[0],
        "SGD: loss after load should be below initial: {loss_after_load} vs {}",
        losses[0],
    );
    assert!(
        *losses.last().unwrap() < losses[0],
        "SGD checkpoint resume: loss should decrease: {} -> {}",
        losses[0],
        losses.last().unwrap(),
    );
    let _ = std::fs::remove_dir_all(&dir);
}

// ── AdaFactor checkpoint round-trip ─────────────────────────────────

/// AdaFactor checkpoint round-trip: train → save → load into fresh AdaFactor → resume.
#[test]
fn test_adafactor_checkpoint_save_load_resume() {
    let (in_dim, hidden, num_classes) = (4, 8, 3);
    let (x_data, t_data) = make_data(12, in_dim, num_classes);

    let mut config = AdaFactorConfig::default();
    config.lr = 0.01;

    // Phase 1: train 5 steps
    let (var_map, w1, b1, w2, b2) = make_varmap_vars(in_dim, hidden, num_classes, true);
    let mut adafactor = AdaFactor::new(var_vec(&w1, &b1, &w2, &b2), config.clone()).unwrap();
    let mut losses = Vec::new();
    for _ in 0..5 {
        let (loss, loss_val) = train_step(&x_data, &t_data, &w1, &b1, &w2, &b2);
        losses.push(loss_val);
        adafactor.backward_step(&loss).unwrap();
    }
    // Save checkpoint
    let dir = std::env::temp_dir().join(format!("nn_adafactor_ckpt_{}", std::process::id()));
    let metadata: TrainingMetadata = serde_json::from_value(serde_json::json!({
        "step": 5, "lr": adafactor.learning_rate(),
    }))
    .unwrap();
    TrainingCheckpoint::save(&dir, &var_map, &adafactor, &metadata).unwrap();

    // Phase 2: fresh VarMap + AdaFactor, load checkpoint, resume
    let (mut var_map2, w1b, b1b, w2b, b2b) = make_varmap_vars(in_dim, hidden, num_classes, false);
    let mut adafactor2 = AdaFactor::new(var_vec(&w1b, &b1b, &w2b, &b2b), config).unwrap();
    let loaded_meta = TrainingCheckpoint::load(&dir, &mut var_map2, &mut adafactor2).unwrap();
    assert_eq!(loaded_meta.step, 5);

    for _ in 0..5 {
        let (loss, loss_val) = train_step(&x_data, &t_data, &w1b, &b1b, &w2b, &b2b);
        losses.push(loss_val);
        adafactor2.backward_step(&loss).unwrap();
    }

    // Verify checkpoint restored state: loss after loading should be lower
    // than initial loss (training progress preserved across save/load).
    let loss_after_load = losses[5];
    assert!(
        loss_after_load < losses[0],
        "AdaFactor: loss after load should be below initial: {loss_after_load} vs {}",
        losses[0],
    );
    assert!(
        *losses.last().unwrap() < losses[0],
        "AdaFactor checkpoint resume: loss should decrease: {} -> {}",
        losses[0],
        losses.last().unwrap(),
    );
    let _ = std::fs::remove_dir_all(&dir);
}

// ── TrainableLoraLinear via TrainableModule trait ─────────────────────

/// TrainableLoraLinear can be used through the TrainableModule trait bound.
/// This verifies generic training loops accept it alongside other Trainable* types.
#[test]
fn test_trainable_lora_linear_via_trait() {
    let (in_dim, out_dim) = (8, 4);
    let rank = 4;
    let alpha = 4.0;
    let batch = 6;

    let frozen_weight = make_weight(out_dim, in_dim, 31);
    let linear = Linear::new(frozen_weight, None).unwrap();
    let lora = TrainableLoraLinear::from_linear(&linear, rank, alpha).unwrap();

    // Exercise through trait object — proves TrainableModule is implemented.
    let module: &dyn TrainableModule = &lora;
    let vars = module.vars();
    assert_eq!(vars.len(), 2, "LoRA has A and B trainable vars");

    // Forward through trait
    let x_data = make_weight(batch, in_dim, 7);
    let tx = Arc::new(TrackedTensor::from_tensor(x_data));
    let y = module.forward(&tx).unwrap();
    let y_shape = y.tensor().dims();
    assert_eq!(y_shape, &[batch, out_dim]);

    // Backward produces gradients for A and B
    let grads = backward(&y.mean_keepdim(1).unwrap().mean_keepdim(0).unwrap()).unwrap();
    for v in module.vars() {
        assert!(grads.get(v).is_some(), "gradient should exist for LoRA var");
    }
}

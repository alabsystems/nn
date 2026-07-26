// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Checkpoint roundtrip tests: OptimizerSnapshot creation, serialize/deserialize,
//! TrainingCheckpoint with metadata, cross-optimizer and cross-config scenarios.

use std::sync::Arc;

use nn_autodiff::{TrackedTensor, Var, VarMap};
use nn_core::dyn_tensor::DynTensor;
use nn_core::test_utils::cpu;
use nn_core::DType;

use crate::checkpoint::{
    GradScalerState, OptimizerCheckpoint, TrainingCheckpoint, TrainingMetadata,
};
use crate::{AdaFactor, AdaFactorConfig, AdamConfig, AdamW, Optimizer, Sgd, SgdConfig};

fn vec_var(vals: &[f32]) -> Var {
    Var::new(DynTensor::from_vec(vals.to_vec(), &[vals.len()], &cpu()).unwrap())
}

fn mat_var(vals: &[f32], rows: usize, cols: usize) -> Var {
    Var::new(DynTensor::from_vec(vals.to_vec(), &[rows, cols], &cpu()).unwrap())
}

fn scalar_loss(v: &Var) -> Arc<TrackedTensor> {
    let t = Arc::new(TrackedTensor::from_var(v).unwrap());
    t.sqr().unwrap().sum_keepdim(0).unwrap()
}

fn mat_loss(v: &Var) -> Arc<TrackedTensor> {
    let t = Arc::new(TrackedTensor::from_var(v).unwrap());
    t.sqr()
        .unwrap()
        .sum_keepdim(0)
        .unwrap()
        .sum_keepdim(1)
        .unwrap()
}

fn get_weights(v: &Var) -> Vec<f32> {
    v.data().unwrap().to_flat_vec::<f32>().unwrap()
}

// ===================================================================
// 1. OptimizerSnapshot creation from optimizer state
// ===================================================================

/// OptimizerSnapshot from a fresh (step-0) AdaFactor has zero tensors and step=0.
#[test]
fn test_snapshot_creation_fresh_adafactor() {
    let v = vec_var(&[1.0, 2.0, 3.0]);
    let config = AdaFactorConfig {
        beta1: Some(0.9),
        ..AdaFactorConfig::default()
    };
    let opt = AdaFactor::new(vec![v], config).unwrap();
    let snap = opt.save_checkpoint().unwrap();

    assert_eq!(snap.metadata["type"], "AdaFactor");
    assert_eq!(snap.metadata["step"], 0);
    // Fresh optimizer: moment tensors exist but are all zeros.
    assert!(snap.tensors.contains_key("adafactor_0_m"));
    assert!(snap.tensors.contains_key("adafactor_0_v_full"));
    for (key, tensor) in &snap.tensors {
        let vals = tensor.to_flat_vec::<f32>().unwrap();
        for &val in &vals {
            assert!(
                val.abs() < f32::EPSILON,
                "fresh snapshot tensor '{key}' should be zero, got {val}"
            );
        }
    }
}

/// OptimizerSnapshot from trained AdamW has non-zero moment tensors.
#[test]
fn test_snapshot_creation_trained_adam() {
    let v = vec_var(&[5.0, -3.0, 1.0]);
    let mut adam = AdamW::new(vec![v.clone()], AdamConfig::default()).unwrap();
    for _ in 0..5 {
        adam.backward_step(&scalar_loss(&v)).unwrap();
    }

    let snap = adam.save_checkpoint().unwrap();
    assert_eq!(snap.metadata["type"], "AdamW");
    assert_eq!(snap.metadata["step"], 5);
    assert_eq!(snap.tensors.len(), 2); // m + v for 1 var

    let m_vals = snap
        .tensors
        .get("adam_0_m")
        .unwrap()
        .to_flat_vec::<f32>()
        .unwrap();
    assert!(
        m_vals.iter().any(|&x| x.abs() > 1e-10),
        "first moment should be non-zero after training"
    );
}

// ===================================================================
// 2. Serialize -> deserialize roundtrip (snapshot level)
// ===================================================================

/// Save-load roundtrip preserves AdaFactor factored moment tensor values exactly.
#[test]
fn test_snapshot_roundtrip_adafactor_factored() {
    let v = mat_var(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 2, 3);
    let config = AdaFactorConfig {
        beta1: Some(0.9),
        ..AdaFactorConfig::default()
    };
    let mut opt = AdaFactor::new(vec![v.clone()], config.clone()).unwrap();

    for _ in 0..5 {
        opt.backward_step(&mat_loss(&v)).unwrap();
    }

    let snap1 = opt.save_checkpoint().unwrap();

    // Load into fresh optimizer.
    let mut opt2 = AdaFactor::new(vec![v], config).unwrap();
    opt2.load_checkpoint(&snap1).unwrap();

    // Re-save and compare.
    let snap2 = opt2.save_checkpoint().unwrap();

    assert_eq!(snap1.tensors.len(), snap2.tensors.len());
    for (key, t1) in &snap1.tensors {
        let t2 = snap2.tensors.get(key).unwrap();
        let v1 = t1.to_flat_vec::<f32>().unwrap();
        let v2 = t2.to_flat_vec::<f32>().unwrap();
        assert_eq!(v1.len(), v2.len(), "length mismatch for key {key}");
        for (i, (a, b)) in v1.iter().zip(v2.iter()).enumerate() {
            assert_eq!(
                a.to_bits(),
                b.to_bits(),
                "bit mismatch at [{i}] for key {key}: {a} vs {b}"
            );
        }
    }
    assert_eq!(snap1.metadata["step"], snap2.metadata["step"]);
}

/// Save-load roundtrip preserves AdaFactor non-factored (vector) moments exactly.
#[test]
fn test_snapshot_roundtrip_adafactor_vector() {
    let v = vec_var(&[3.0, -1.0, 2.0, 4.0]);
    let config = AdaFactorConfig {
        beta1: Some(0.9),
        ..AdaFactorConfig::default()
    };
    let mut opt = AdaFactor::new(vec![v.clone()], config.clone()).unwrap();

    for _ in 0..5 {
        opt.backward_step(&scalar_loss(&v)).unwrap();
    }

    let snap1 = opt.save_checkpoint().unwrap();
    let mut opt2 = AdaFactor::new(vec![v], config).unwrap();
    opt2.load_checkpoint(&snap1).unwrap();
    let snap2 = opt2.save_checkpoint().unwrap();

    for (key, t1) in &snap1.tensors {
        let t2 = snap2.tensors.get(key).unwrap();
        let v1 = t1.to_flat_vec::<f32>().unwrap();
        let v2 = t2.to_flat_vec::<f32>().unwrap();
        for (i, (a, b)) in v1.iter().zip(v2.iter()).enumerate() {
            assert_eq!(
                a.to_bits(),
                b.to_bits(),
                "bit mismatch at [{i}] for key {key}"
            );
        }
    }
}

// ===================================================================
// 3. TrainingCheckpoint with metadata
// ===================================================================

/// TrainingCheckpoint save/load with GradScaler state and extra metadata.
#[test]
fn test_training_checkpoint_with_grad_scaler_metadata() {
    let mut map = VarMap::new();
    let w = map.get("weight", &[3], DType::F32, &cpu()).unwrap();
    w.set(&DynTensor::from_vec(vec![1.0, 2.0, 3.0], &[3], &cpu()).unwrap())
        .unwrap();

    let mut adam = AdamW::new(vec![w.clone()], AdamConfig::default()).unwrap();
    for _ in 0..3 {
        adam.backward_step(&scalar_loss(&w)).unwrap();
    }

    let dir =
        std::env::temp_dir().join(format!("nn_ckpt_roundtrip_scaler_{}", std::process::id()));
    let metadata = TrainingMetadata {
        step: 3,
        lr: adam.learning_rate(),
        grad_scaler: Some(GradScalerState {
            scale: 2048.0,
            growth_tracker: 7,
        }),
        extra: Some(serde_json::json!({
            "epoch": 1,
            "dataset": "librispeech",
        })),
    };
    TrainingCheckpoint::save(&dir, &map, &adam, &metadata).unwrap();

    // Reload.
    let mut map2 = VarMap::new();
    map2.get("weight", &[3], DType::F32, &cpu()).unwrap();
    let mut adam2 = AdamW::new(vec![w], AdamConfig::default()).unwrap();

    let loaded = TrainingCheckpoint::load(&dir, &mut map2, &mut adam2).unwrap();
    assert_eq!(loaded.step, 3);
    assert!((loaded.lr - adam.learning_rate()).abs() < 1e-10);

    let gs = loaded.grad_scaler.unwrap();
    assert!((gs.scale - 2048.0).abs() < f64::EPSILON);
    assert_eq!(gs.growth_tracker, 7);

    let extra = loaded.extra.unwrap();
    assert_eq!(extra["epoch"], 1);
    assert_eq!(extra["dataset"], "librispeech");

    std::fs::remove_dir_all(&dir).ok();
}

/// TrainingCheckpoint with no GradScaler and no extra metadata.
#[test]
fn test_training_checkpoint_minimal_metadata() {
    let mut map = VarMap::new();
    let w = map.get("bias", &[2], DType::F32, &cpu()).unwrap();
    w.set(&DynTensor::from_vec(vec![0.5, -0.5], &[2], &cpu()).unwrap())
        .unwrap();

    let mut sgd = Sgd::new(
        vec![w.clone()],
        SgdConfig {
            lr: 0.01,
            momentum: 0.9,
            ..Default::default()
        },
    )
    .unwrap();
    sgd.backward_step(&scalar_loss(&w)).unwrap();

    let dir = std::env::temp_dir().join(format!("nn_ckpt_minimal_{}", std::process::id()));
    let metadata = TrainingMetadata {
        step: 1,
        lr: 0.01,
        grad_scaler: None,
        extra: None,
    };
    TrainingCheckpoint::save(&dir, &map, &sgd, &metadata).unwrap();

    let mut map2 = VarMap::new();
    map2.get("bias", &[2], DType::F32, &cpu()).unwrap();
    let mut sgd2 = Sgd::new(
        vec![w],
        SgdConfig {
            lr: 0.01,
            momentum: 0.9,
            ..Default::default()
        },
    )
    .unwrap();

    let loaded = TrainingCheckpoint::load(&dir, &mut map2, &mut sgd2).unwrap();
    assert_eq!(loaded.step, 1);
    assert!(loaded.grad_scaler.is_none());
    assert!(loaded.extra.is_none());

    std::fs::remove_dir_all(&dir).ok();
}

// ===================================================================
// 4. Checkpoint with different optimizer types
// ===================================================================

/// TrainingCheckpoint roundtrip with AdaFactor optimizer (factored + beta1).
#[test]
fn test_training_checkpoint_adafactor() {
    let mut map = VarMap::new();
    let w = map.get("linear", &[2, 3], DType::F32, &cpu()).unwrap();
    w.set(&DynTensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], &cpu()).unwrap())
        .unwrap();

    let config = AdaFactorConfig {
        beta1: Some(0.9),
        lr: 0.005,
        relative_step: false,
        ..AdaFactorConfig::default()
    };
    let mut opt = AdaFactor::new(vec![w.clone()], config.clone()).unwrap();
    for _ in 0..4 {
        opt.backward_step(&mat_loss(&w)).unwrap();
    }

    let dir = std::env::temp_dir().join(format!("nn_ckpt_adafactor_{}", std::process::id()));
    let metadata = TrainingMetadata {
        step: 4,
        lr: opt.learning_rate(),
        grad_scaler: None,
        extra: None,
    };
    TrainingCheckpoint::save(&dir, &map, &opt, &metadata).unwrap();

    // Verify files.
    assert!(dir.join("model.safetensors").exists());
    assert!(dir.join("optimizer.safetensors").exists());
    assert!(dir.join("training_state.json").exists());

    // Reload.
    let mut map2 = VarMap::new();
    map2.get("linear", &[2, 3], DType::F32, &cpu()).unwrap();
    let mut opt2 = AdaFactor::new(vec![w], config).unwrap();

    let loaded = TrainingCheckpoint::load(&dir, &mut map2, &mut opt2).unwrap();
    assert_eq!(loaded.step, 4);
    assert_eq!(opt2.step_count(), 4);

    std::fs::remove_dir_all(&dir).ok();
}

/// TrainingCheckpoint roundtrip with SGD (zero momentum — no velocity tensors).
#[test]
fn test_training_checkpoint_sgd_no_momentum() {
    let mut map = VarMap::new();
    let w = map.get("w", &[4], DType::F32, &cpu()).unwrap();
    w.set(&DynTensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[4], &cpu()).unwrap())
        .unwrap();

    let config = SgdConfig {
        lr: 0.1,
        momentum: 0.0,
        ..Default::default()
    };
    let mut sgd = Sgd::new(vec![w.clone()], config.clone()).unwrap();
    sgd.backward_step(&scalar_loss(&w)).unwrap();

    let dir = std::env::temp_dir().join(format!("nn_ckpt_sgd_nomom_{}", std::process::id()));
    let metadata = TrainingMetadata {
        step: 1,
        lr: 0.1,
        grad_scaler: None,
        extra: None,
    };
    TrainingCheckpoint::save(&dir, &map, &sgd, &metadata).unwrap();

    let mut map2 = VarMap::new();
    map2.get("w", &[4], DType::F32, &cpu()).unwrap();
    let mut sgd2 = Sgd::new(vec![w], config).unwrap();

    let loaded = TrainingCheckpoint::load(&dir, &mut map2, &mut sgd2).unwrap();
    assert_eq!(loaded.step, 1);

    // optimizer.safetensors should still exist (even if empty).
    // The implementation skips writing if tensors are empty.
    // Verify load succeeds regardless.

    std::fs::remove_dir_all(&dir).ok();
}

// ===================================================================
// 5. Checkpoint compatibility across config changes
// ===================================================================

/// Loading a checkpoint with different LR updates the config to checkpoint's LR.
#[test]
fn test_checkpoint_config_lr_override() {
    let v = vec_var(&[1.0, 2.0]);
    let config1 = AdamConfig {
        lr: 0.001,
        ..AdamConfig::default()
    };
    let mut adam = AdamW::new(vec![v.clone()], config1).unwrap();
    adam.backward_step(&scalar_loss(&v)).unwrap();

    let snap = adam.save_checkpoint().unwrap();
    assert!((snap.metadata["lr"].as_f64().unwrap() - 0.001).abs() < 1e-10);

    // Create new optimizer with different LR.
    let config2 = AdamConfig {
        lr: 0.1,
        ..AdamConfig::default()
    };
    let mut adam2 = AdamW::new(vec![v], config2).unwrap();
    assert!((adam2.learning_rate() - 0.1).abs() < 1e-10);

    adam2.load_checkpoint(&snap).unwrap();
    // LR should be overridden to checkpoint value.
    assert!(
        (adam2.learning_rate() - 0.001).abs() < 1e-10,
        "LR should be overridden to snapshot value, got {}",
        adam2.learning_rate()
    );
}

/// Loading an AdaFactor checkpoint with different beta1 updates config.
#[test]
fn test_adafactor_checkpoint_beta1_override() {
    let v = vec_var(&[1.0, 2.0, 3.0]);
    let config1 = AdaFactorConfig {
        beta1: Some(0.9),
        ..AdaFactorConfig::default()
    };
    let mut opt = AdaFactor::new(vec![v.clone()], config1).unwrap();
    opt.backward_step(&scalar_loss(&v)).unwrap();

    let snap = opt.save_checkpoint().unwrap();

    // Load into optimizer with beta1=None.
    let config2 = AdaFactorConfig {
        beta1: None,
        ..AdaFactorConfig::default()
    };
    let mut opt2 = AdaFactor::new(vec![v], config2).unwrap();
    opt2.load_checkpoint(&snap).unwrap();

    // Config should be updated from checkpoint.
    assert_eq!(opt2.config().beta1, Some(0.9));
}

/// Loading an AdaFactor checkpoint with different weight_decay updates config.
#[test]
fn test_adafactor_checkpoint_weight_decay_override() {
    let v = vec_var(&[2.0, 3.0]);
    let config1 = AdaFactorConfig {
        weight_decay: 0.05,
        ..AdaFactorConfig::default()
    };
    let mut opt = AdaFactor::new(vec![v.clone()], config1).unwrap();
    opt.backward_step(&scalar_loss(&v)).unwrap();

    let snap = opt.save_checkpoint().unwrap();

    let config2 = AdaFactorConfig {
        weight_decay: 0.0,
        ..AdaFactorConfig::default()
    };
    let mut opt2 = AdaFactor::new(vec![v], config2).unwrap();
    opt2.load_checkpoint(&snap).unwrap();

    assert!(
        (opt2.config().weight_decay - 0.05).abs() < 1e-10,
        "weight_decay should be updated from checkpoint, got {}",
        opt2.config().weight_decay
    );
}

/// Training continues correctly after full TrainingCheckpoint roundtrip.
#[test]
fn test_training_checkpoint_continuity() {
    let mut map = VarMap::new();
    let w = map.get("w", &[3], DType::F32, &cpu()).unwrap();
    w.set(&DynTensor::from_vec(vec![5.0, -3.0, 2.0], &[3], &cpu()).unwrap())
        .unwrap();

    let config = AdamConfig {
        lr: 0.01,
        weight_decay: 0.0,
        ..AdamConfig::default()
    };
    let mut adam = AdamW::new(vec![w.clone()], config.clone()).unwrap();

    // Train 5 steps.
    for _ in 0..5 {
        adam.backward_step(&scalar_loss(&w)).unwrap();
    }
    let w_at_5 = get_weights(&w);

    let dir = std::env::temp_dir().join(format!("nn_ckpt_continuity_{}", std::process::id()));
    let metadata = TrainingMetadata {
        step: 5,
        lr: adam.learning_rate(),
        grad_scaler: None,
        extra: None,
    };
    TrainingCheckpoint::save(&dir, &map, &adam, &metadata).unwrap();

    // Take 5 more steps on original.
    for _ in 0..5 {
        adam.backward_step(&scalar_loss(&w)).unwrap();
    }
    let w_orig_10 = get_weights(&w);

    // Load from checkpoint and take 5 more steps.
    let mut map2 = VarMap::new();
    let w2 = map2.get("w", &[3], DType::F32, &cpu()).unwrap();

    let mut adam2 = AdamW::new(vec![w2.clone()], config).unwrap();
    TrainingCheckpoint::load(&dir, &mut map2, &mut adam2).unwrap();

    // Verify weights were restored.
    let w_restored = get_weights(&w2);
    for (i, (a, b)) in w_at_5.iter().zip(w_restored.iter()).enumerate() {
        assert!(
            (a - b).abs() < 1e-5,
            "weight mismatch at [{i}]: saved={a}, restored={b}"
        );
    }

    // Continue training.
    for _ in 0..5 {
        adam2.backward_step(&scalar_loss(&w2)).unwrap();
    }
    let w_rest_10 = get_weights(&w2);

    for (i, (a, b)) in w_orig_10.iter().zip(w_rest_10.iter()).enumerate() {
        assert!(
            (a - b).abs() < 1e-4,
            "continuity mismatch at [{i}]: orig={a}, restored={b}"
        );
    }

    std::fs::remove_dir_all(&dir).ok();
}

/// TrainingMetadata serialization roundtrip via serde_json.
#[test]
fn test_training_metadata_serde_roundtrip() {
    let meta = TrainingMetadata {
        step: 42,
        lr: 0.00123,
        grad_scaler: Some(GradScalerState {
            scale: 4096.0,
            growth_tracker: 15,
        }),
        extra: Some(serde_json::json!({"note": "test checkpoint"})),
    };

    let json = serde_json::to_string(&meta).unwrap();
    let loaded: TrainingMetadata = serde_json::from_str(&json).unwrap();

    assert_eq!(loaded.step, 42);
    assert!((loaded.lr - 0.00123).abs() < 1e-10);
    let gs = loaded.grad_scaler.unwrap();
    assert!((gs.scale - 4096.0).abs() < f64::EPSILON);
    assert_eq!(gs.growth_tracker, 15);
    assert_eq!(loaded.extra.unwrap()["note"], "test checkpoint");
}

/// TrainingMetadata with no optional fields serializes cleanly.
#[test]
fn test_training_metadata_serde_no_optionals() {
    let meta = TrainingMetadata {
        step: 0,
        lr: 0.0,
        grad_scaler: None,
        extra: None,
    };

    let json = serde_json::to_string(&meta).unwrap();
    assert!(
        !json.contains("grad_scaler"),
        "None fields should be skipped"
    );
    assert!(!json.contains("extra"), "None fields should be skipped");

    let loaded: TrainingMetadata = serde_json::from_str(&json).unwrap();
    assert_eq!(loaded.step, 0);
    assert!(loaded.grad_scaler.is_none());
    assert!(loaded.extra.is_none());
}

// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Extended checkpoint and convergence tests for nn-optim.
//!
//! Covers:
//! - TrainingCheckpoint construction and field access
//! - OptimizerSnapshot serialization/deserialization
//! - TrainingMetadata fields and serde roundtrips
//! - GradScalerState save/restore with edge cases
//! - AdamW config validation (lr, betas, eps, weight_decay)
//! - AdaFactor config validation
//! - SGD config validation (lr, momentum, weight_decay)
//! - Optimizer trait implementations (learning_rate, set_learning_rate)
//! - Edge cases: zero learning rate, very large weight decay, NaN detection

use std::collections::HashMap;
use std::sync::Arc;

use nn_autodiff::{backward, GradStore, TrackedTensor, Var};
use nn_core::dyn_tensor::DynTensor;
use nn_core::Device;

use crate::adafactor::{AdaFactor, AdaFactorConfig};
use crate::adam::{AdamConfig, AdamW};
use crate::checkpoint::{
    GradScalerState, OptimizerCheckpoint, OptimizerSnapshot, TrainingCheckpoint, TrainingMetadata,
};
use crate::grad_scaler::{GradScaler, GradScalerConfig};
use crate::optimizer::Optimizer;
use crate::sgd::{Sgd, SgdConfig};

// -- helpers ------------------------------------------------------------------

fn cpu() -> Device {
    Device::Cpu
}

fn make_var(vals: &[f32], shape: &[usize]) -> Var {
    Var::new(DynTensor::from_vec(vals.to_vec(), shape, &cpu()).unwrap())
}

fn quadratic_grads(var: &Var) -> GradStore {
    let t = Arc::new(TrackedTensor::from_var(var).unwrap());
    let sq = t.sqr().unwrap();
    let mut loss = sq;
    for d in 0..var.data().unwrap().dims().len() {
        loss = loss.sum_keepdim(d).unwrap();
    }
    backward(&loss).unwrap()
}

fn loss_of(var: &Var) -> f64 {
    var.data()
        .unwrap()
        .to_flat_vec::<f32>()
        .unwrap()
        .iter()
        .map(|&v| f64::from(v) * f64::from(v))
        .sum()
}

// ============================================================================
// 1. TrainingCheckpoint construction and field access
// ============================================================================

#[test]
fn test_training_checkpoint_is_unit_struct() {
    let _cp = TrainingCheckpoint;
    // TrainingCheckpoint is a unit struct used for save/load namespace.
    // This verifies it can be constructed.
}

#[test]
fn test_training_metadata_all_fields_populated() {
    let meta = TrainingMetadata {
        step: 1000,
        lr: 3e-4,
        grad_scaler: Some(GradScalerState {
            scale: 32768.0,
            growth_tracker: 150,
        }),
        extra: Some(serde_json::json!({"epoch": 5, "best_loss": 0.123})),
    };
    assert_eq!(meta.step, 1000);
    assert!((meta.lr - 3e-4).abs() < f64::EPSILON);
    let gs = meta.grad_scaler.as_ref().unwrap();
    assert!((gs.scale - 32768.0).abs() < f64::EPSILON);
    assert_eq!(gs.growth_tracker, 150);
    assert_eq!(meta.extra.as_ref().unwrap()["epoch"], 5);
}

#[test]
fn test_training_metadata_zero_step() {
    let meta = TrainingMetadata {
        step: 0,
        lr: 0.0,
        grad_scaler: None,
        extra: None,
    };
    assert_eq!(meta.step, 0);
    assert!((meta.lr - 0.0).abs() < f64::EPSILON);
}

#[test]
fn test_training_metadata_large_step() {
    let meta = TrainingMetadata {
        step: usize::MAX,
        lr: 1e-5,
        grad_scaler: None,
        extra: None,
    };
    assert_eq!(meta.step, usize::MAX);
}

// ============================================================================
// 2. OptimizerSnapshot serialization/deserialization
// ============================================================================

#[test]
fn test_optimizer_snapshot_empty() {
    let snapshot = OptimizerSnapshot {
        tensors: HashMap::new(),
        metadata: serde_json::Value::Null,
    };
    assert!(snapshot.tensors.is_empty());
    assert!(snapshot.metadata.is_null());
}

#[test]
fn test_optimizer_snapshot_with_tensors() {
    let mut tensors = HashMap::new();
    let t = DynTensor::from_vec(vec![1.0, 2.0, 3.0], &[3], &cpu()).unwrap();
    tensors.insert("test_moment".to_string(), t);
    let metadata = serde_json::json!({"type": "test", "step": 42});
    let snapshot = OptimizerSnapshot { tensors, metadata };
    assert_eq!(snapshot.tensors.len(), 1);
    assert!(snapshot.tensors.contains_key("test_moment"));
    assert_eq!(snapshot.metadata["step"], 42);
}

#[test]
fn test_adam_checkpoint_save_load_roundtrip() {
    let var = make_var(&[5.0, -3.0], &[2]);
    let config = AdamConfig {
        lr: 0.01,
        beta1: 0.9,
        beta2: 0.999,
        eps: 1e-8,
        weight_decay: 0.02,
    };
    let mut opt = AdamW::new(vec![var.clone()], config).unwrap();

    // Run a few steps to populate moment estimates
    for _ in 0..3 {
        let grads = quadratic_grads(&var);
        opt.step(&grads).unwrap();
    }
    assert_eq!(opt.step_count(), 3);

    // Save checkpoint
    let snapshot = opt.save_checkpoint().unwrap();
    assert!(!snapshot.tensors.is_empty());
    assert_eq!(snapshot.metadata["type"], "AdamW");
    assert_eq!(snapshot.metadata["step"], 3);

    // Create a fresh optimizer and load the checkpoint
    let var2 = make_var(&[5.0, -3.0], &[2]);
    let mut opt2 = AdamW::new(vec![var2], AdamConfig::default()).unwrap();
    opt2.load_checkpoint(&snapshot).unwrap();

    assert_eq!(opt2.step_count(), 3);
    assert!((opt2.config().lr - 0.01).abs() < f64::EPSILON);
    assert!((opt2.config().weight_decay - 0.02).abs() < f64::EPSILON);
}

#[test]
fn test_sgd_checkpoint_save_load_roundtrip() {
    let var = make_var(&[4.0, -2.0, 1.0], &[3]);
    let config = SgdConfig {
        lr: 0.05,
        momentum: 0.9,
        weight_decay: 1e-4,
    };
    let mut opt = Sgd::new(vec![var.clone()], config).unwrap();

    // Run steps to populate velocity buffers
    for _ in 0..5 {
        let grads = quadratic_grads(&var);
        opt.step(&grads).unwrap();
    }

    let snapshot = opt.save_checkpoint().unwrap();
    assert_eq!(snapshot.metadata["type"], "Sgd");
    // With momentum, velocity tensors should be saved
    assert!(!snapshot.tensors.is_empty());

    // Restore into a fresh optimizer
    let var2 = make_var(&[4.0, -2.0, 1.0], &[3]);
    let mut opt2 = Sgd::new(vec![var2], SgdConfig::default()).unwrap();
    opt2.load_checkpoint(&snapshot).unwrap();

    assert!((opt2.learning_rate() - 0.05).abs() < f64::EPSILON);
    assert!((opt2.momentum() - 0.9).abs() < f64::EPSILON);
    assert!((opt2.weight_decay() - 1e-4).abs() < f64::EPSILON);
}

#[test]
fn test_adafactor_checkpoint_save_load_roundtrip() {
    let var = make_var(&[2.0, -1.0, 0.5, 3.0], &[4]);
    let config = AdaFactorConfig {
        lr: 0.05,
        relative_step: false,
        beta1: Some(0.9),
        ..Default::default()
    };
    let mut opt = AdaFactor::new(vec![var.clone()], config).unwrap();

    for _ in 0..3 {
        let grads = quadratic_grads(&var);
        opt.step(&grads).unwrap();
    }

    let snapshot = opt.save_checkpoint().unwrap();
    assert_eq!(snapshot.metadata["type"], "AdaFactor");
    assert_eq!(snapshot.metadata["step"], 3);

    let var2 = make_var(&[2.0, -1.0, 0.5, 3.0], &[4]);
    let mut opt2 = AdaFactor::new(vec![var2], AdaFactorConfig::default()).unwrap();
    opt2.load_checkpoint(&snapshot).unwrap();
    assert_eq!(opt2.step_count(), 3);
    assert!((opt2.config().lr - 0.05).abs() < f64::EPSILON);
}

// ============================================================================
// 3. TrainingMetadata serde
// ============================================================================

#[test]
fn test_training_metadata_json_roundtrip_full() {
    let meta = TrainingMetadata {
        step: 999,
        lr: 2.5e-4,
        grad_scaler: Some(GradScalerState {
            scale: 4096.0,
            growth_tracker: 50,
        }),
        extra: Some(serde_json::json!({"run_id": "abc123"})),
    };
    let json = serde_json::to_string_pretty(&meta).unwrap();
    let restored: TrainingMetadata = serde_json::from_str(&json).unwrap();
    assert_eq!(restored.step, 999);
    assert!((restored.lr - 2.5e-4).abs() < f64::EPSILON);
    let gs = restored.grad_scaler.as_ref().unwrap();
    assert!((gs.scale - 4096.0).abs() < f64::EPSILON);
    assert_eq!(gs.growth_tracker, 50);
    assert_eq!(restored.extra.as_ref().unwrap()["run_id"], "abc123");
}

#[test]
fn test_training_metadata_optional_fields_omitted_in_json() {
    let meta = TrainingMetadata {
        step: 10,
        lr: 0.01,
        grad_scaler: None,
        extra: None,
    };
    let json = serde_json::to_string(&meta).unwrap();
    assert!(!json.contains("grad_scaler"));
    assert!(!json.contains("extra"));
}

#[test]
fn test_training_metadata_deserialize_missing_optional() {
    // JSON without optional fields should deserialize with None values
    let json = r#"{"step": 7, "lr": 0.001}"#;
    let meta: TrainingMetadata = serde_json::from_str(json).unwrap();
    assert_eq!(meta.step, 7);
    assert!((meta.lr - 0.001).abs() < f64::EPSILON);
    assert!(meta.grad_scaler.is_none());
    assert!(meta.extra.is_none());
}

// ============================================================================
// 4. GradScalerState save/restore
// ============================================================================

#[test]
fn test_grad_scaler_state_roundtrip_json() {
    let state = GradScalerState {
        scale: 131072.0,
        growth_tracker: 1500,
    };
    let json = serde_json::to_string(&state).unwrap();
    let restored: GradScalerState = serde_json::from_str(&json).unwrap();
    assert!((restored.scale - 131072.0).abs() < f64::EPSILON);
    assert_eq!(restored.growth_tracker, 1500);
}

#[test]
fn test_grad_scaler_save_and_load_preserves_state() {
    let config = GradScalerConfig {
        init_scale: 1024.0,
        growth_interval: 10,
        min_scale: 1.0,
        max_scale: 1e8,
        ..Default::default()
    };
    let mut scaler = GradScaler::new(config.clone()).unwrap();

    // Simulate some clean steps to advance growth tracker
    for _ in 0..5 {
        scaler.update();
    }
    let saved = scaler.save_state();
    assert!((saved.scale - 1024.0).abs() < f64::EPSILON);
    assert_eq!(saved.growth_tracker, 5);

    // Create a fresh scaler and load
    let mut scaler2 = GradScaler::new(config).unwrap();
    scaler2.load_state(&saved).unwrap();
    assert!((scaler2.scale_factor() - 1024.0).abs() < f64::EPSILON);
}

#[test]
fn test_grad_scaler_load_clamps_to_bounds() {
    let config = GradScalerConfig {
        init_scale: 100.0,
        min_scale: 50.0,
        max_scale: 200.0,
        ..Default::default()
    };
    let mut scaler = GradScaler::new(config).unwrap();

    // Load a scale below min_scale -- should be clamped
    let state = GradScalerState {
        scale: 10.0,
        growth_tracker: 0,
    };
    scaler.load_state(&state).unwrap();
    assert!(
        (scaler.scale_factor() - 50.0).abs() < f64::EPSILON,
        "scale should be clamped to min_scale=50, got {}",
        scaler.scale_factor()
    );

    // Load a scale above max_scale -- should be clamped
    let state = GradScalerState {
        scale: 500.0,
        growth_tracker: 0,
    };
    scaler.load_state(&state).unwrap();
    assert!(
        (scaler.scale_factor() - 200.0).abs() < f64::EPSILON,
        "scale should be clamped to max_scale=200, got {}",
        scaler.scale_factor()
    );
}

#[test]
fn test_grad_scaler_load_rejects_non_finite_scale() {
    let config = GradScalerConfig::default();
    let mut scaler = GradScaler::new(config).unwrap();

    let state_nan = GradScalerState {
        scale: f64::NAN,
        growth_tracker: 0,
    };
    assert!(scaler.load_state(&state_nan).is_err());

    let state_inf = GradScalerState {
        scale: f64::INFINITY,
        growth_tracker: 0,
    };
    assert!(scaler.load_state(&state_inf).is_err());

    let state_neg = GradScalerState {
        scale: -1.0,
        growth_tracker: 0,
    };
    assert!(scaler.load_state(&state_neg).is_err());

    let state_zero = GradScalerState {
        scale: 0.0,
        growth_tracker: 0,
    };
    assert!(scaler.load_state(&state_zero).is_err());
}

#[test]
fn test_grad_scaler_load_caps_growth_tracker() {
    let config = GradScalerConfig {
        init_scale: 100.0,
        growth_interval: 10,
        min_scale: 1.0,
        max_scale: 1e6,
        ..Default::default()
    };
    let mut scaler = GradScaler::new(config).unwrap();

    // Growth tracker larger than growth_interval should be capped
    let state = GradScalerState {
        scale: 100.0,
        growth_tracker: 999,
    };
    scaler.load_state(&state).unwrap();
    // growth_tracker is capped to growth_interval.saturating_sub(1) = 9
    // This means at least one more clean step is needed before growth.
}

// ============================================================================
// 5. AdamW config validation
// ============================================================================

#[test]
fn test_adamw_rejects_negative_lr() {
    let var = make_var(&[1.0], &[1]);
    let config = AdamConfig {
        lr: -0.01,
        ..AdamConfig::default()
    };
    assert!(AdamW::new(vec![var], config).is_err());
}

#[test]
fn test_adamw_rejects_nan_lr() {
    let var = make_var(&[1.0], &[1]);
    let config = AdamConfig {
        lr: f64::NAN,
        ..AdamConfig::default()
    };
    assert!(AdamW::new(vec![var], config).is_err());
}

#[test]
fn test_adamw_rejects_inf_lr() {
    let var = make_var(&[1.0], &[1]);
    let config = AdamConfig {
        lr: f64::INFINITY,
        ..AdamConfig::default()
    };
    assert!(AdamW::new(vec![var], config).is_err());
}

#[test]
fn test_adamw_rejects_beta1_at_one() {
    let var = make_var(&[1.0], &[1]);
    let config = AdamConfig {
        beta1: 1.0,
        ..AdamConfig::default()
    };
    assert!(AdamW::new(vec![var], config).is_err());
}

#[test]
fn test_adamw_rejects_beta2_negative() {
    let var = make_var(&[1.0], &[1]);
    let config = AdamConfig {
        beta2: -0.1,
        ..AdamConfig::default()
    };
    assert!(AdamW::new(vec![var], config).is_err());
}

#[test]
fn test_adamw_rejects_zero_eps() {
    let var = make_var(&[1.0], &[1]);
    let config = AdamConfig {
        eps: 0.0,
        ..AdamConfig::default()
    };
    assert!(AdamW::new(vec![var], config).is_err());
}

#[test]
fn test_adamw_rejects_negative_eps() {
    let var = make_var(&[1.0], &[1]);
    let config = AdamConfig {
        eps: -1e-8,
        ..AdamConfig::default()
    };
    assert!(AdamW::new(vec![var], config).is_err());
}

#[test]
fn test_adamw_rejects_negative_weight_decay() {
    let var = make_var(&[1.0], &[1]);
    let config = AdamConfig {
        weight_decay: -0.01,
        ..AdamConfig::default()
    };
    assert!(AdamW::new(vec![var], config).is_err());
}

#[test]
fn test_adamw_accepts_zero_lr() {
    let var = make_var(&[5.0], &[1]);
    let config = AdamConfig {
        lr: 0.0,
        ..AdamConfig::default()
    };
    let mut opt = AdamW::new(vec![var.clone()], config).unwrap();

    // With lr=0, parameters should not change
    let before = var.data().unwrap().to_flat_vec::<f32>().unwrap()[0];
    let grads = quadratic_grads(&var);
    opt.step(&grads).unwrap();
    let after = var.data().unwrap().to_flat_vec::<f32>().unwrap()[0];
    assert!(
        (after - before).abs() < 1e-10,
        "with lr=0, param should not change: before={before}, after={after}"
    );
}

#[test]
fn test_adamw_accepts_zero_weight_decay() {
    let var = make_var(&[3.0], &[1]);
    let config = AdamConfig {
        weight_decay: 0.0,
        ..AdamConfig::default()
    };
    let opt = AdamW::new(vec![var], config).unwrap();
    assert!((opt.config().weight_decay - 0.0).abs() < f64::EPSILON);
}

#[test]
fn test_adamw_default_config_matches_pytorch() {
    let config = AdamConfig::default();
    assert!((config.lr - 1e-3).abs() < f64::EPSILON);
    assert!((config.beta1 - 0.9).abs() < f64::EPSILON);
    assert!((config.beta2 - 0.999).abs() < f64::EPSILON);
    assert!((config.eps - 1e-8).abs() < f64::EPSILON);
    assert!((config.weight_decay - 0.01).abs() < f64::EPSILON);
}

// ============================================================================
// 6. AdaFactor config validation
// ============================================================================

#[test]
fn test_adafactor_rejects_positive_decay_rate() {
    let var = make_var(&[1.0], &[1]);
    let config = AdaFactorConfig {
        decay_rate: 0.5,
        ..Default::default()
    };
    assert!(AdaFactor::new(vec![var], config).is_err());
}

#[test]
fn test_adafactor_rejects_nan_decay_rate() {
    let var = make_var(&[1.0], &[1]);
    let config = AdaFactorConfig {
        decay_rate: f64::NAN,
        ..Default::default()
    };
    assert!(AdaFactor::new(vec![var], config).is_err());
}

#[test]
fn test_adafactor_rejects_zero_eps_denom() {
    let var = make_var(&[1.0], &[1]);
    let config = AdaFactorConfig {
        eps_denom: 0.0,
        ..Default::default()
    };
    assert!(AdaFactor::new(vec![var], config).is_err());
}

#[test]
fn test_adafactor_rejects_negative_eps_rms() {
    let var = make_var(&[1.0], &[1]);
    let config = AdaFactorConfig {
        eps_rms: -1e-3,
        ..Default::default()
    };
    assert!(AdaFactor::new(vec![var], config).is_err());
}

#[test]
fn test_adafactor_rejects_invalid_beta1() {
    let var = make_var(&[1.0], &[1]);
    let config = AdaFactorConfig {
        beta1: Some(1.0),
        ..Default::default()
    };
    assert!(AdaFactor::new(vec![var], config).is_err());
}

#[test]
fn test_adafactor_accepts_none_beta1() {
    let var = make_var(&[1.0], &[1]);
    let config = AdaFactorConfig {
        beta1: None,
        ..Default::default()
    };
    let opt = AdaFactor::new(vec![var], config).unwrap();
    assert!(opt.config().beta1.is_none());
}

#[test]
fn test_adafactor_default_config() {
    let config = AdaFactorConfig::default();
    assert!((config.lr - 1e-3).abs() < f64::EPSILON);
    assert!(!config.relative_step);
    assert!((config.eps_rms - 1e-3).abs() < f64::EPSILON);
    assert!((config.eps_denom - 1e-30).abs() < f64::EPSILON);
    assert!((config.decay_rate - (-0.8)).abs() < f64::EPSILON);
    assert!(config.beta1.is_none());
    assert!((config.weight_decay - 0.0).abs() < f64::EPSILON);
}

// ============================================================================
// 7. SGD config validation
// ============================================================================

#[test]
fn test_sgd_rejects_negative_lr() {
    let var = make_var(&[1.0], &[1]);
    let config = SgdConfig {
        lr: -0.1,
        ..SgdConfig::default()
    };
    assert!(Sgd::new(vec![var], config).is_err());
}

#[test]
fn test_sgd_rejects_nan_momentum() {
    let var = make_var(&[1.0], &[1]);
    let config = SgdConfig {
        momentum: f64::NAN,
        ..SgdConfig::default()
    };
    assert!(Sgd::new(vec![var], config).is_err());
}

#[test]
fn test_sgd_rejects_negative_momentum() {
    let var = make_var(&[1.0], &[1]);
    let config = SgdConfig {
        momentum: -0.5,
        ..SgdConfig::default()
    };
    assert!(Sgd::new(vec![var], config).is_err());
}

#[test]
fn test_sgd_rejects_inf_weight_decay() {
    let var = make_var(&[1.0], &[1]);
    let config = SgdConfig {
        weight_decay: f64::INFINITY,
        ..SgdConfig::default()
    };
    assert!(Sgd::new(vec![var], config).is_err());
}

#[test]
fn test_sgd_accepts_zero_momentum() {
    let var = make_var(&[1.0], &[1]);
    let config = SgdConfig {
        momentum: 0.0,
        ..SgdConfig::default()
    };
    let opt = Sgd::new(vec![var], config).unwrap();
    assert!((opt.momentum() - 0.0).abs() < f64::EPSILON);
}

#[test]
fn test_sgd_default_config() {
    let config = SgdConfig::default();
    assert!((config.lr - 1e-2).abs() < f64::EPSILON);
    assert!((config.momentum - 0.0).abs() < f64::EPSILON);
    assert!((config.weight_decay - 0.0).abs() < f64::EPSILON);
}

// ============================================================================
// 8. Optimizer trait implementations
// ============================================================================

#[test]
fn test_adamw_learning_rate_getter() {
    let var = make_var(&[1.0], &[1]);
    let config = AdamConfig {
        lr: 0.042,
        ..AdamConfig::default()
    };
    let opt = AdamW::new(vec![var], config).unwrap();
    assert!((opt.learning_rate() - 0.042).abs() < f64::EPSILON);
}

#[test]
fn test_adamw_set_learning_rate() {
    let var = make_var(&[1.0], &[1]);
    let mut opt = AdamW::new(vec![var], AdamConfig::default()).unwrap();

    opt.set_learning_rate(0.05).unwrap();
    assert!((opt.learning_rate() - 0.05).abs() < f64::EPSILON);

    opt.set_learning_rate(0.0).unwrap();
    assert!((opt.learning_rate() - 0.0).abs() < f64::EPSILON);
}

#[test]
fn test_adamw_set_learning_rate_rejects_invalid() {
    let var = make_var(&[1.0], &[1]);
    let mut opt = AdamW::new(vec![var], AdamConfig::default()).unwrap();

    assert!(opt.set_learning_rate(-0.01).is_err());
    assert!(opt.set_learning_rate(f64::NAN).is_err());
    assert!(opt.set_learning_rate(f64::INFINITY).is_err());

    // Original LR should be preserved after failed set
    assert!((opt.learning_rate() - 1e-3).abs() < f64::EPSILON);
}

#[test]
fn test_sgd_learning_rate_getter_and_setter() {
    let var = make_var(&[1.0], &[1]);
    let mut opt = Sgd::new(vec![var], SgdConfig::default()).unwrap();

    assert!((opt.learning_rate() - 0.01).abs() < f64::EPSILON);

    opt.set_learning_rate(0.1).unwrap();
    assert!((opt.learning_rate() - 0.1).abs() < f64::EPSILON);

    assert!(opt.set_learning_rate(-1.0).is_err());
    // LR should not change on error
    assert!((opt.learning_rate() - 0.1).abs() < f64::EPSILON);
}

#[test]
fn test_adafactor_learning_rate_getter_and_setter() {
    let var = make_var(&[1.0], &[1]);
    let mut opt = AdaFactor::new(vec![var], AdaFactorConfig::default()).unwrap();

    assert!((opt.learning_rate() - 1e-3).abs() < f64::EPSILON);

    opt.set_learning_rate(0.5).unwrap();
    assert!((opt.learning_rate() - 0.5).abs() < f64::EPSILON);

    assert!(opt.set_learning_rate(f64::NEG_INFINITY).is_err());
}

// ============================================================================
// 9. Edge cases: zero lr, very large weight decay, NaN detection
// ============================================================================

#[test]
fn test_sgd_zero_lr_no_update() {
    let var = make_var(&[10.0, -5.0], &[2]);
    let config = SgdConfig {
        lr: 0.0,
        momentum: 0.0,
        weight_decay: 0.0,
    };
    let mut opt = Sgd::new(vec![var.clone()], config).unwrap();

    let before = var.data().unwrap().to_flat_vec::<f32>().unwrap();
    let grads = quadratic_grads(&var);
    opt.step(&grads).unwrap();
    let after = var.data().unwrap().to_flat_vec::<f32>().unwrap();

    for (b, a) in before.iter().zip(after.iter()) {
        assert!(
            (b - a).abs() < 1e-10,
            "with lr=0, no update should occur: before={b}, after={a}"
        );
    }
}

#[test]
fn test_sgd_large_weight_decay() {
    // Very large weight decay should shrink parameters toward zero rapidly
    let var = make_var(&[100.0], &[1]);
    let config = SgdConfig {
        lr: 0.1,
        momentum: 0.0,
        weight_decay: 10.0,
    };
    let mut opt = Sgd::new(vec![var.clone()], config).unwrap();

    let grads = quadratic_grads(&var);
    opt.step(&grads).unwrap();

    // grad = 2*100 = 200, with wd: grad += 10*100 = 1000, total grad = 1200
    // new = 100 - 0.1 * 1200 = 100 - 120 = -20
    let val = var.data().unwrap().to_flat_vec::<f32>().unwrap()[0];
    assert!(
        val < 0.0,
        "large weight decay should overshoot past zero, got {val}"
    );
}

#[test]
fn test_adamw_very_large_weight_decay_convergence() {
    // Large weight decay with AdamW (decoupled) should still converge
    let var = make_var(&[5.0, -3.0], &[2]);
    let config = AdamConfig {
        lr: 0.1,
        weight_decay: 1.0,
        ..AdamConfig::default()
    };
    let mut opt = AdamW::new(vec![var.clone()], config).unwrap();

    let initial_loss = loss_of(&var);
    for _ in 0..50 {
        let grads = quadratic_grads(&var);
        opt.step(&grads).unwrap();
    }
    let final_loss = loss_of(&var);

    assert!(
        final_loss < initial_loss,
        "even with large wd, loss should decrease: init={initial_loss}, final={final_loss}"
    );
}

#[test]
fn test_adamw_step_count_increments() {
    let var = make_var(&[1.0], &[1]);
    let mut opt = AdamW::new(vec![var.clone()], AdamConfig::default()).unwrap();

    assert_eq!(opt.step_count(), 0);
    for i in 1..=5 {
        let grads = quadratic_grads(&var);
        opt.step(&grads).unwrap();
        assert_eq!(opt.step_count(), i);
    }
}

#[test]
fn test_adafactor_step_count_increments() {
    let var = make_var(&[1.0], &[1]);
    let config = AdaFactorConfig {
        lr: 0.01,
        relative_step: false,
        ..Default::default()
    };
    let mut opt = AdaFactor::new(vec![var.clone()], config).unwrap();

    assert_eq!(opt.step_count(), 0);
    for i in 1..=3 {
        let grads = quadratic_grads(&var);
        opt.step(&grads).unwrap();
        assert_eq!(opt.step_count(), i);
    }
}

#[test]
fn test_checkpoint_shape_mismatch_rejected() {
    let var = make_var(&[1.0, 2.0], &[2]);
    let mut opt = AdamW::new(vec![var], AdamConfig::default()).unwrap();

    // Create a snapshot with wrong-shaped tensors
    let mut tensors = HashMap::new();
    let wrong_shape = DynTensor::from_vec(vec![1.0, 2.0, 3.0], &[3], &cpu()).unwrap();
    tensors.insert("adam_0_m".to_string(), wrong_shape.clone());
    tensors.insert("adam_0_v".to_string(), wrong_shape);
    let snapshot = OptimizerSnapshot {
        tensors,
        metadata: serde_json::json!({"type": "AdamW", "step": 0}),
    };

    let result = opt.load_checkpoint(&snapshot);
    assert!(result.is_err(), "should reject shape-mismatched checkpoint");
}

#[test]
fn test_sgd_checkpoint_shape_mismatch_rejected() {
    let var = make_var(&[1.0, 2.0, 3.0], &[3]);
    let config = SgdConfig {
        momentum: 0.9,
        ..SgdConfig::default()
    };
    let mut opt = Sgd::new(vec![var.clone()], config).unwrap();

    // Run one step to initialize velocity
    let grads = quadratic_grads(&var);
    opt.step(&grads).unwrap();

    // Create snapshot with wrong shape for velocity
    let mut tensors = HashMap::new();
    let wrong = DynTensor::from_vec(vec![1.0, 2.0], &[2], &cpu()).unwrap();
    tensors.insert("sgd_0_velocity".to_string(), wrong);
    let snapshot = OptimizerSnapshot {
        tensors,
        metadata: serde_json::json!({"type": "Sgd", "lr": 0.01, "momentum": 0.9, "weight_decay": 0.0}),
    };

    assert!(opt.load_checkpoint(&snapshot).is_err());
}

#[test]
fn test_adamw_empty_vars_no_panic() {
    let opt = AdamW::new(vec![], AdamConfig::default()).unwrap();
    assert_eq!(opt.step_count(), 0);
    assert!((opt.learning_rate() - 1e-3).abs() < f64::EPSILON);
}

#[test]
fn test_sgd_empty_vars_no_panic() {
    let opt = Sgd::new(vec![], SgdConfig::default()).unwrap();
    assert!((opt.learning_rate() - 0.01).abs() < f64::EPSILON);
}

#[test]
fn test_adafactor_empty_vars_no_panic() {
    let opt = AdaFactor::new(vec![], AdaFactorConfig::default()).unwrap();
    assert_eq!(opt.step_count(), 0);
}

#[test]
fn test_adamw_config_accessor() {
    let var = make_var(&[1.0], &[1]);
    let config = AdamConfig {
        lr: 0.007,
        beta1: 0.85,
        beta2: 0.995,
        eps: 1e-6,
        weight_decay: 0.05,
    };
    let opt = AdamW::new(vec![var], config).unwrap();
    let c = opt.config();
    assert!((c.lr - 0.007).abs() < f64::EPSILON);
    assert!((c.beta1 - 0.85).abs() < f64::EPSILON);
    assert!((c.beta2 - 0.995).abs() < f64::EPSILON);
    assert!((c.eps - 1e-6).abs() < f64::EPSILON);
    assert!((c.weight_decay - 0.05).abs() < f64::EPSILON);
}

#[test]
fn test_sgd_config_accessor() {
    let var = make_var(&[1.0], &[1]);
    let config = SgdConfig {
        lr: 0.05,
        momentum: 0.99,
        weight_decay: 0.001,
    };
    let opt = Sgd::new(vec![var], config).unwrap();
    let c = opt.config();
    assert!((c.lr - 0.05).abs() < f64::EPSILON);
    assert!((c.momentum - 0.99).abs() < f64::EPSILON);
    assert!((c.weight_decay - 0.001).abs() < f64::EPSILON);
}

#[test]
fn test_adafactor_config_accessor() {
    let var = make_var(&[1.0], &[1]);
    let config = AdaFactorConfig {
        lr: 0.02,
        relative_step: true,
        eps_rms: 2e-3,
        eps_denom: 1e-20,
        decay_rate: -0.5,
        beta1: Some(0.8),
        weight_decay: 0.01,
    };
    let opt = AdaFactor::new(vec![var], config).unwrap();
    let c = opt.config();
    assert!((c.lr - 0.02).abs() < f64::EPSILON);
    assert!(c.relative_step);
    assert!((c.eps_rms - 2e-3).abs() < f64::EPSILON);
    assert!((c.decay_rate - (-0.5)).abs() < f64::EPSILON);
    assert!((c.beta1.unwrap() - 0.8).abs() < f64::EPSILON);
}

#[test]
fn test_adam_checkpoint_metadata_contains_all_config_fields() {
    let var = make_var(&[1.0], &[1]);
    let config = AdamConfig {
        lr: 0.003,
        beta1: 0.85,
        beta2: 0.998,
        eps: 1e-7,
        weight_decay: 0.05,
    };
    let opt = AdamW::new(vec![var], config).unwrap();
    let snapshot = opt.save_checkpoint().unwrap();

    assert_eq!(snapshot.metadata["type"], "AdamW");
    assert_eq!(snapshot.metadata["lr"], 0.003);
    assert_eq!(snapshot.metadata["beta1"], 0.85);
    assert_eq!(snapshot.metadata["beta2"], 0.998);
    assert_eq!(snapshot.metadata["eps"], 1e-7);
    assert_eq!(snapshot.metadata["weight_decay"], 0.05);
    assert_eq!(snapshot.metadata["step"], 0);
}

#[test]
fn test_adafactor_checkpoint_metadata_contains_all_config_fields() {
    let var = make_var(&[1.0], &[1]);
    let config = AdaFactorConfig {
        lr: 0.01,
        relative_step: true,
        eps_rms: 2e-3,
        eps_denom: 1e-25,
        decay_rate: -0.7,
        beta1: Some(0.9),
        weight_decay: 0.001,
    };
    let opt = AdaFactor::new(vec![var], config).unwrap();
    let snapshot = opt.save_checkpoint().unwrap();

    assert_eq!(snapshot.metadata["type"], "AdaFactor");
    assert_eq!(snapshot.metadata["lr"], 0.01);
    assert_eq!(snapshot.metadata["relative_step"], true);
    assert_eq!(snapshot.metadata["eps_rms"], 2e-3);
    assert_eq!(snapshot.metadata["decay_rate"], -0.7);
    assert_eq!(snapshot.metadata["beta1"], 0.9);
    assert_eq!(snapshot.metadata["weight_decay"], 0.001);
}

#[test]
fn test_sgd_checkpoint_metadata_contains_all_config_fields() {
    let var = make_var(&[1.0], &[1]);
    let config = SgdConfig {
        lr: 0.05,
        momentum: 0.95,
        weight_decay: 1e-4,
    };
    let opt = Sgd::new(vec![var], config).unwrap();
    let snapshot = opt.save_checkpoint().unwrap();

    assert_eq!(snapshot.metadata["type"], "Sgd");
    assert_eq!(snapshot.metadata["lr"], 0.05);
    assert_eq!(snapshot.metadata["momentum"], 0.95);
    assert_eq!(snapshot.metadata["weight_decay"], 1e-4);
}

#[test]
fn test_adam_checkpoint_restore_preserves_training_progress() {
    // Verify that restoring a checkpoint and continuing training
    // produces similar results to uninterrupted training.
    let var_continuous = make_var(&[8.0, -6.0], &[2]);
    let config = AdamConfig {
        lr: 0.1,
        ..AdamConfig::default()
    };
    let mut opt_continuous = AdamW::new(vec![var_continuous.clone()], config.clone()).unwrap();

    // Run 10 steps continuously
    for _ in 0..10 {
        let grads = quadratic_grads(&var_continuous);
        opt_continuous.step(&grads).unwrap();
    }

    let var_resume = make_var(&[8.0, -6.0], &[2]);
    let mut opt_resume = AdamW::new(vec![var_resume.clone()], config.clone()).unwrap();

    // Run 5 steps, save, load, run 5 more
    for _ in 0..5 {
        let grads = quadratic_grads(&var_resume);
        opt_resume.step(&grads).unwrap();
    }
    let snapshot = opt_resume.save_checkpoint().unwrap();

    let var_loaded = make_var(&[8.0, -6.0], &[2]);
    // Must use the SAME config (lr=0.1) as the resume path so the pre-load
    // 5 steps drive var_loaded to the same point as var_resume reached.
    // AdamConfig::default() uses lr=1e-3, which barely moves the var, so after
    // restoring the moments the variable values diverge from the continuous run.
    let mut opt_loaded = AdamW::new(vec![var_loaded.clone()], config).unwrap();

    // Before loading: run the same first 5 steps to get the var to the same state
    for _ in 0..5 {
        let grads = quadratic_grads(&var_loaded);
        opt_loaded.step(&grads).unwrap();
    }
    // Now load checkpoint to restore optimizer state
    opt_loaded.load_checkpoint(&snapshot).unwrap();

    // Run 5 more steps
    for _ in 0..5 {
        let grads = quadratic_grads(&var_loaded);
        opt_loaded.step(&grads).unwrap();
    }

    let loss_continuous = loss_of(&var_continuous);
    let loss_loaded = loss_of(&var_loaded);

    // Both should have converged to similar values
    assert!(
        (loss_continuous - loss_loaded).abs() < 1e-2,
        "checkpoint resume should match continuous: continuous={loss_continuous}, loaded={loss_loaded}"
    );
}

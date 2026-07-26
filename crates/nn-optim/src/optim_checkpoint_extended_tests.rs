// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Extended tests for optimizer checkpoint, gradient clipping, learning rate
//! schedules, and gradient scaler utilities.

use std::sync::Arc;

use nn_autodiff::{backward, GradStore, TrackedTensor, Var};
use nn_core::dyn_tensor::DynTensor;
use nn_core::Device;

use crate::checkpoint::{GradScalerState, TrainingCheckpoint, TrainingMetadata};
use crate::grad_clip::{clip_grad_norm, clip_grad_value};
use crate::grad_scaler::{GradScaler, GradScalerConfig};
use crate::lr_schedule::{step_with_schedule, CosineSchedule, LrSchedule, WarmupSchedule};
use crate::optimizer::Optimizer;
use crate::sgd::{Sgd, SgdConfig};

// -- helpers ------------------------------------------------------------------

fn cpu() -> Device {
    Device::Cpu
}

fn scalar_var(val: f32) -> Var {
    Var::new(DynTensor::from_vec(vec![val], &[1], &cpu()).unwrap())
}

/// Build a GradStore where the single var has gradient equal to `grad_values`.
fn grads_for(grad_values: &[f32]) -> (Var, GradStore) {
    let n = grad_values.len();
    let var = Var::new(DynTensor::from_vec(vec![1.0; n], &[n], &cpu()).unwrap());
    let t = Arc::new(TrackedTensor::from_var(&var).unwrap());
    let target = Arc::new(TrackedTensor::from_tensor(
        DynTensor::from_vec(grad_values.to_vec(), &[n], &cpu()).unwrap(),
    ));
    let product = t.mul(&target).unwrap();
    let loss = product.sum_keepdim(0).unwrap();
    let grads = backward(&loss).unwrap();
    (var, grads)
}

// == TrainingCheckpoint creation ==============================================

#[test]
fn test_training_checkpoint_creation() {
    // TrainingCheckpoint is a unit struct; verify it can be referenced and that
    // save/load methods exist (we don't call them because they require VarMap
    // + filesystem, but construction proves the type is usable).
    let _checkpoint = TrainingCheckpoint;
}

// == TrainingMetadata fields ==================================================

#[test]
fn test_training_metadata_fields() {
    let meta = TrainingMetadata {
        step: 42,
        lr: 0.001,
        grad_scaler: None,
        extra: None,
    };
    assert_eq!(meta.step, 42);
    assert!((meta.lr - 0.001).abs() < f64::EPSILON);
    assert!(meta.grad_scaler.is_none());
    assert!(meta.extra.is_none());
}

#[test]
fn test_training_metadata_with_grad_scaler() {
    let meta = TrainingMetadata {
        step: 100,
        lr: 0.01,
        grad_scaler: Some(GradScalerState {
            scale: 65536.0,
            growth_tracker: 5,
        }),
        extra: None,
    };
    let scaler_state = meta.grad_scaler.as_ref().unwrap();
    assert!((scaler_state.scale - 65536.0).abs() < f64::EPSILON);
    assert_eq!(scaler_state.growth_tracker, 5);
}

#[test]
fn test_training_metadata_with_extra() {
    let extra = serde_json::json!({"experiment": "test_run_1", "batch_size": 32});
    let meta = TrainingMetadata {
        step: 0,
        lr: 1e-4,
        grad_scaler: None,
        extra: Some(extra),
    };
    assert_eq!(meta.extra.as_ref().unwrap()["experiment"], "test_run_1");
    assert_eq!(meta.extra.as_ref().unwrap()["batch_size"], 32);
}

// == TrainingMetadata serialization ===========================================

#[test]
fn test_training_metadata_serialization_roundtrip() {
    let meta = TrainingMetadata {
        step: 500,
        lr: 0.005,
        grad_scaler: Some(GradScalerState {
            scale: 1024.0,
            growth_tracker: 10,
        }),
        extra: Some(serde_json::json!({"note": "test"})),
    };

    let json = serde_json::to_string(&meta).unwrap();
    let deserialized: TrainingMetadata = serde_json::from_str(&json).unwrap();

    assert_eq!(deserialized.step, 500);
    assert!((deserialized.lr - 0.005).abs() < f64::EPSILON);
    let gs = deserialized.grad_scaler.as_ref().unwrap();
    assert!((gs.scale - 1024.0).abs() < f64::EPSILON);
    assert_eq!(gs.growth_tracker, 10);
    assert_eq!(deserialized.extra.as_ref().unwrap()["note"], "test");
}

#[test]
fn test_training_metadata_serialization_without_optional_fields() {
    let meta = TrainingMetadata {
        step: 0,
        lr: 0.0,
        grad_scaler: None,
        extra: None,
    };

    let json = serde_json::to_string(&meta).unwrap();
    // skip_serializing_if = "Option::is_none" should omit these fields
    assert!(
        !json.contains("grad_scaler"),
        "None fields should be omitted: {json}"
    );
    assert!(
        !json.contains("extra"),
        "None fields should be omitted: {json}"
    );

    let deserialized: TrainingMetadata = serde_json::from_str(&json).unwrap();
    assert_eq!(deserialized.step, 0);
    assert!(deserialized.grad_scaler.is_none());
    assert!(deserialized.extra.is_none());
}

// == GradScalerState ==========================================================

#[test]
fn test_grad_scaler_state_creation_and_fields() {
    let state = GradScalerState {
        scale: 512.0,
        growth_tracker: 42,
    };
    assert!((state.scale - 512.0).abs() < f64::EPSILON);
    assert_eq!(state.growth_tracker, 42);
}

#[test]
fn test_grad_scaler_state_serialization_roundtrip() {
    let state = GradScalerState {
        scale: 65536.0,
        growth_tracker: 1999,
    };
    let json = serde_json::to_string(&state).unwrap();
    let deserialized: GradScalerState = serde_json::from_str(&json).unwrap();
    assert!((deserialized.scale - 65536.0).abs() < f64::EPSILON);
    assert_eq!(deserialized.growth_tracker, 1999);
}

// == Gradient Clipping ========================================================

#[test]
fn test_clip_grad_norm_basic() {
    // Gradient = [3.0, 4.0], L2 norm = 5.0, max_norm = 1.0 -> scale by 1/5
    let (_var, mut grads) = grads_for(&[3.0, 4.0]);
    let total_norm = clip_grad_norm(&mut grads, 1.0).unwrap();

    // Original norm should be ~5.0
    assert!(
        (total_norm - 5.0).abs() < 1e-5,
        "original norm should be 5.0, got {total_norm}"
    );

    // After clipping, new norm should be ~1.0
    let mut clipped_norm_sq = 0.0f64;
    for (_id, g) in grads.var_grads() {
        let vals = g.to_flat_vec::<f32>().unwrap();
        for &v in &vals {
            clipped_norm_sq += f64::from(v) * f64::from(v);
        }
    }
    let clipped_norm = clipped_norm_sq.sqrt();
    assert!(
        (clipped_norm - 1.0).abs() < 1e-5,
        "clipped norm should be 1.0, got {clipped_norm}"
    );
}

#[test]
fn test_clip_grad_norm_no_clip_needed() {
    // Gradient = [0.1, 0.2], L2 norm = sqrt(0.05) ~ 0.2236, max_norm = 1.0
    let (_var, mut grads) = grads_for(&[0.1, 0.2]);
    let total_norm = clip_grad_norm(&mut grads, 1.0).unwrap();

    let expected_norm = (0.01f64 + 0.04).sqrt();
    assert!(
        (total_norm - expected_norm).abs() < 1e-5,
        "norm should be ~{expected_norm}, got {total_norm}"
    );

    // Gradients should be unchanged since norm < max_norm
    for (_id, g) in grads.var_grads() {
        let vals = g.to_flat_vec::<f32>().unwrap();
        assert!(
            (vals[0] - 0.1).abs() < 1e-6,
            "gradient should be unchanged, got {}",
            vals[0]
        );
        assert!(
            (vals[1] - 0.2).abs() < 1e-6,
            "gradient should be unchanged, got {}",
            vals[1]
        );
    }
}

#[test]
fn test_clip_grad_value_basic() {
    // Gradient = [5.0, -5.0, 0.3], clip to [-1.0, 1.0]
    let (_var, mut grads) = grads_for(&[5.0, -5.0, 0.3]);
    clip_grad_value(&mut grads, 1.0).unwrap();

    for (_id, g) in grads.var_grads() {
        let vals = g.to_flat_vec::<f32>().unwrap();
        assert!(
            (vals[0] - 1.0).abs() < 1e-6,
            "positive should clamp to 1.0, got {}",
            vals[0]
        );
        assert!(
            (vals[1] - (-1.0)).abs() < 1e-6,
            "negative should clamp to -1.0, got {}",
            vals[1]
        );
        assert!(
            (vals[2] - 0.3).abs() < 1e-6,
            "within range should be unchanged, got {}",
            vals[2]
        );
    }
}

#[test]
fn test_clip_grad_value_symmetric() {
    // Verify clamping is symmetric: equal positive and negative values
    // should clamp to equal magnitude
    let (_var, mut grads) = grads_for(&[10.0, -10.0]);
    clip_grad_value(&mut grads, 2.5).unwrap();

    for (_id, g) in grads.var_grads() {
        let vals = g.to_flat_vec::<f32>().unwrap();
        assert!(
            (vals[0] - 2.5).abs() < 1e-6,
            "positive clamp to 2.5, got {}",
            vals[0]
        );
        assert!(
            (vals[1] - (-2.5)).abs() < 1e-6,
            "negative clamp to -2.5, got {}",
            vals[1]
        );
        // Verify symmetry: |clipped_positive| == |clipped_negative|
        assert!(
            (vals[0].abs() - vals[1].abs()).abs() < 1e-6,
            "clipping should be symmetric: {} vs {}",
            vals[0],
            vals[1]
        );
    }
}

// == Learning Rate Schedules ==================================================

#[test]
fn test_warmup_schedule_starts_low() {
    let sched = WarmupSchedule::new(0.1, 100).unwrap();
    let lr_at_0 = sched.lr_at_step(0);
    assert!(
        lr_at_0.abs() < f64::EPSILON,
        "warmup should start at 0.0, got {lr_at_0}"
    );
    let lr_at_1 = sched.lr_at_step(1);
    assert!(
        lr_at_1 > 0.0 && lr_at_1 < 0.1,
        "early warmup should be between 0 and base_lr, got {lr_at_1}"
    );
}

#[test]
fn test_warmup_schedule_reaches_target() {
    let sched = WarmupSchedule::new(0.05, 200).unwrap();
    let lr = sched.lr_at_step(200);
    assert!(
        (lr - 0.05).abs() < f64::EPSILON,
        "at warmup_steps, lr should be base_lr=0.05, got {lr}"
    );
    // Also verify it stays at base_lr after warmup
    let lr_after = sched.lr_at_step(500);
    assert!(
        (lr_after - 0.05).abs() < f64::EPSILON,
        "after warmup, lr should remain at base_lr=0.05, got {lr_after}"
    );
}

#[test]
fn test_cosine_schedule_starts_at_max() {
    let sched = CosineSchedule::new(0.01, 0.001, 0, 1000).unwrap();
    let lr = sched.lr_at_step(0);
    assert!(
        (lr - 0.01).abs() < 1e-10,
        "cosine should start at base_lr=0.01, got {lr}"
    );
}

#[test]
fn test_cosine_schedule_ends_at_min() {
    let sched = CosineSchedule::new(0.01, 0.001, 0, 1000).unwrap();
    let lr = sched.lr_at_step(1000);
    assert!(
        (lr - 0.001).abs() < 1e-10,
        "cosine should end at min_lr=0.001, got {lr}"
    );
}

#[test]
fn test_cosine_schedule_is_monotonic() {
    let sched = CosineSchedule::new(0.01, 0.0, 0, 500).unwrap();
    let mut prev = sched.lr_at_step(0);
    for step in 1..=500 {
        let lr = sched.lr_at_step(step);
        assert!(
            lr <= prev + 1e-15,
            "cosine should be monotonically non-increasing: step {step}, prev {prev}, cur {lr}"
        );
        prev = lr;
    }
}

#[test]
fn test_step_with_schedule_applies_correct_lr() {
    let x = scalar_var(10.0);
    let mut sgd = Sgd::new(
        vec![x.clone()],
        SgdConfig {
            lr: 0.0,
            ..SgdConfig::default()
        },
    )
    .unwrap();

    let schedule = WarmupSchedule::new(0.1, 10).unwrap();

    // At step 5, lr should be 0.1 * 5/10 = 0.05
    let expected_lr = schedule.lr_at_step(5);
    assert!(
        (expected_lr - 0.05).abs() < f64::EPSILON,
        "expected lr=0.05 at step 5, got {expected_lr}"
    );

    let t = Arc::new(TrackedTensor::from_var(&x).unwrap());
    let loss = t.sqr().unwrap(); // grad = 2*10 = 20
    let grads = backward(&loss).unwrap();

    step_with_schedule(&mut sgd, &grads, &schedule, 5).unwrap();

    // After step: x = 10 - 0.05 * 20 = 10 - 1.0 = 9.0
    let val = x.data().unwrap().to_flat_vec::<f32>().unwrap()[0];
    assert!((val - 9.0).abs() < 1e-4, "expected x ~ 9.0, got {val}");
    assert!(
        (sgd.learning_rate() - 0.05).abs() < f64::EPSILON,
        "optimizer lr should be set to schedule lr"
    );
}

// == GradScaler ===============================================================

#[test]
fn test_grad_scaler_config_defaults() {
    let cfg = GradScalerConfig::default();
    assert!(
        (cfg.init_scale - 65536.0).abs() < f64::EPSILON,
        "init_scale default"
    );
    assert!(
        (cfg.growth_factor - 2.0).abs() < f64::EPSILON,
        "growth_factor default"
    );
    assert!(
        (cfg.backoff_factor - 0.5).abs() < f64::EPSILON,
        "backoff_factor default"
    );
    assert_eq!(cfg.growth_interval, 2000, "growth_interval default");
    assert!(
        (cfg.min_scale - 1.0).abs() < f64::EPSILON,
        "min_scale default"
    );
    assert!(
        (cfg.max_scale - 16_777_216.0).abs() < f64::EPSILON,
        "max_scale default"
    );
}

#[test]
fn test_grad_scaler_creation() {
    let scaler = GradScaler::new(GradScalerConfig::default()).unwrap();
    assert!(
        (scaler.scale_factor() - 65536.0).abs() < f64::EPSILON,
        "initial scale should match config"
    );
    assert!(
        !scaler.found_inf(),
        "fresh scaler should not have found inf"
    );
}

#[test]
fn test_grad_scaler_creation_custom_config() {
    let config = GradScalerConfig {
        init_scale: 1024.0,
        growth_factor: 3.0,
        backoff_factor: 0.25,
        growth_interval: 500,
        min_scale: 0.5,
        max_scale: 1e10,
    };
    let scaler = GradScaler::new(config).unwrap();
    assert!(
        (scaler.scale_factor() - 1024.0).abs() < f64::EPSILON,
        "custom init_scale"
    );
}

#[test]
fn test_grad_scaler_scale_factor() {
    let scaler = GradScaler::new(GradScalerConfig {
        init_scale: 256.0,
        min_scale: 1.0,
        max_scale: 1e6,
        ..Default::default()
    })
    .unwrap();
    assert!(
        (scaler.scale_factor() - 256.0).abs() < f64::EPSILON,
        "initial scale factor should be 256.0, got {}",
        scaler.scale_factor()
    );
}

#[test]
fn test_grad_scaler_save_and_load_state() {
    let mut scaler = GradScaler::new(GradScalerConfig {
        init_scale: 512.0,
        min_scale: 1.0,
        max_scale: 1e6,
        ..Default::default()
    })
    .unwrap();

    let state = scaler.save_state();
    assert!((state.scale - 512.0).abs() < f64::EPSILON);
    assert_eq!(state.growth_tracker, 0);

    // Modify state and load it back
    let modified_state = GradScalerState {
        scale: 2048.0,
        growth_tracker: 100,
    };
    scaler.load_state(&modified_state).unwrap();
    assert!(
        (scaler.scale_factor() - 2048.0).abs() < f64::EPSILON,
        "scale should be restored to 2048.0, got {}",
        scaler.scale_factor()
    );
}

#[test]
fn test_grad_scaler_invalid_configs_rejected() {
    // Zero init_scale
    assert!(GradScaler::new(GradScalerConfig {
        init_scale: 0.0,
        ..Default::default()
    })
    .is_err());

    // Negative init_scale
    assert!(GradScaler::new(GradScalerConfig {
        init_scale: -1.0,
        ..Default::default()
    })
    .is_err());

    // NaN init_scale
    assert!(GradScaler::new(GradScalerConfig {
        init_scale: f64::NAN,
        ..Default::default()
    })
    .is_err());

    // growth_interval = 0
    assert!(GradScaler::new(GradScalerConfig {
        growth_interval: 0,
        ..Default::default()
    })
    .is_err());
}

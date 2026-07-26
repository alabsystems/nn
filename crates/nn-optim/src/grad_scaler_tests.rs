#![allow(deprecated)]
// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

use super::*;
use nn_autodiff::{backward, TrackedTensor, Var};
use nn_core::dyn_tensor::DynTensor;
use nn_core::test_utils::cpu;
use std::sync::Arc;

#[path = "grad_scaler_edge_tests.rs"]
mod edge;

#[path = "grad_scaler_state_tests.rs"]
mod state;

#[test]
fn test_grad_scaler_default_config() {
    let scaler = GradScaler::new(GradScalerConfig::default()).unwrap();
    assert!((scaler.scale_factor() - 65536.0).abs() < f64::EPSILON);
    assert!(!scaler.found_inf());
}

#[test]
fn test_scale_loss_multiplies_by_factor() {
    let scaler = GradScaler::new(GradScalerConfig {
        init_scale: 1024.0,
        ..Default::default()
    })
    .unwrap();
    let x = Var::new(DynTensor::from_vec(vec![3.0f32], &[1], &cpu()).unwrap());
    let t = Arc::new(TrackedTensor::from_var(&x).unwrap());
    let loss = Arc::new(t.sqr().unwrap()); // loss = 9.0

    let scaled = scaler.scale_loss(&loss).unwrap();
    let grads = backward(&scaled).unwrap();
    // dy/dx = 2*x*scale = 2*3*1024 = 6144
    let grad = grads.get(&x).unwrap();
    let val = grad.to_flat_vec::<f32>().unwrap();
    assert!(
        (val[0] - 6144.0).abs() < 1.0,
        "expected ~6144, got {}",
        val[0]
    );
}

#[test]
fn test_unscale_restores_original_gradient() {
    let mut scaler = GradScaler::new(GradScalerConfig {
        init_scale: 1024.0,
        ..Default::default()
    })
    .unwrap();
    let x = Var::new(DynTensor::from_vec(vec![3.0f32], &[1], &cpu()).unwrap());
    let t = Arc::new(TrackedTensor::from_var(&x).unwrap());
    let loss = Arc::new(t.sqr().unwrap());

    let scaled = scaler.scale_loss(&loss).unwrap();
    let mut grads = backward(&scaled).unwrap();

    let ok = scaler.unscale_and_check(&mut grads).unwrap();
    assert!(ok, "gradients should be finite");
    assert!(!scaler.found_inf());

    // After unscaling: grad = 6144 / 1024 = 6.0
    let grad = grads.get(&x).unwrap();
    let val = grad.to_flat_vec::<f32>().unwrap();
    assert!((val[0] - 6.0).abs() < 0.01, "expected ~6.0, got {}", val[0]);
}

#[test]
fn test_unscale_detects_inf() {
    // backward() rejects non-finite loss values, so we create a finite loss,
    // run backward to get valid gradients, then manually inject inf to test
    // that unscale_and_check detects it.
    let x = Var::new(DynTensor::from_vec(vec![1.0f32], &[1], &cpu()).unwrap());
    let t = Arc::new(TrackedTensor::from_var(&x).unwrap());
    let mut scaler = GradScaler::new(GradScalerConfig {
        init_scale: 1e20,
        max_scale: 1e20,
        ..Default::default()
    })
    .unwrap();

    // Finite loss: loss = x * 1.0 = 1.0
    let loss_finite = t.mul_scalar(1.0).unwrap();
    let scaled = scaler.scale_loss(&loss_finite).unwrap();
    let mut grads = backward(&scaled).unwrap();

    // Manually replace the gradient with inf to test unscale_and_check's detection.
    let inf_grad = DynTensor::from_vec(vec![f32::INFINITY], &[1], &cpu()).unwrap();
    for (_, grad) in grads.var_grads_mut() {
        *grad = inf_grad.clone();
    }

    let ok = scaler.unscale_and_check(&mut grads).unwrap();
    assert!(!ok, "should detect inf in gradients");
    assert!(scaler.found_inf());
}

#[test]
fn test_update_reduces_scale_on_inf() {
    let mut scaler = GradScaler::new(GradScalerConfig {
        init_scale: 1024.0,
        backoff_factor: 0.5,
        ..Default::default()
    })
    .unwrap();
    // Simulate finding inf
    scaler.found_inf = true;
    scaler.update();
    assert!((scaler.scale_factor() - 512.0).abs() < f64::EPSILON);
}

#[test]
fn test_update_grows_scale_after_interval() {
    let mut scaler = GradScaler::new(GradScalerConfig {
        init_scale: 1024.0,
        growth_factor: 2.0,
        growth_interval: 3,
        ..Default::default()
    })
    .unwrap();

    // 3 clean steps should trigger growth
    for _ in 0..3 {
        scaler.found_inf = false;
        scaler.update();
    }
    assert!((scaler.scale_factor() - 2048.0).abs() < f64::EPSILON);
}

#[test]
fn test_scale_respects_min_max_bounds() {
    let mut scaler = GradScaler::new(GradScalerConfig {
        init_scale: 2.0,
        backoff_factor: 0.5,
        min_scale: 1.0,
        max_scale: 4.0,
        growth_factor: 2.0,
        growth_interval: 1,
    })
    .unwrap();

    // Back off below min → clamped to min
    scaler.found_inf = true;
    scaler.update(); // 2.0 * 0.5 = 1.0
    assert!((scaler.scale_factor() - 1.0).abs() < f64::EPSILON);

    scaler.found_inf = true;
    scaler.update(); // 1.0 * 0.5 = 0.5 → clamped to 1.0
    assert!((scaler.scale_factor() - 1.0).abs() < f64::EPSILON);

    // Grow beyond max → clamped to max
    scaler.found_inf = false;
    scaler.update(); // 1.0 * 2.0 = 2.0
    assert!((scaler.scale_factor() - 2.0).abs() < f64::EPSILON);

    scaler.found_inf = false;
    scaler.update(); // 2.0 * 2.0 = 4.0
    assert!((scaler.scale_factor() - 4.0).abs() < f64::EPSILON);

    scaler.found_inf = false;
    scaler.update(); // 4.0 * 2.0 = 8.0 → clamped to 4.0
    assert!((scaler.scale_factor() - 4.0).abs() < f64::EPSILON);
}

#[test]
fn test_inf_resets_growth_tracker() {
    let mut scaler = GradScaler::new(GradScalerConfig {
        init_scale: 1024.0,
        growth_interval: 5,
        ..Default::default()
    })
    .unwrap();

    // 4 clean steps (not enough to trigger growth)
    for _ in 0..4 {
        scaler.found_inf = false;
        scaler.update();
    }
    assert!((scaler.scale_factor() - 1024.0).abs() < f64::EPSILON);

    // One inf resets the counter
    scaler.found_inf = true;
    scaler.update();
    assert!((scaler.scale_factor() - 512.0).abs() < f64::EPSILON);

    // Now need 5 more clean steps (not 1) to grow
    for _ in 0..4 {
        scaler.found_inf = false;
        scaler.update();
    }
    assert!((scaler.scale_factor() - 512.0).abs() < f64::EPSILON);

    scaler.found_inf = false;
    scaler.update(); // 5th clean step → growth
    assert!((scaler.scale_factor() - 1024.0).abs() < f64::EPSILON);
}

#[test]
fn test_e2e_mixed_precision_training() {
    // Train a simple model with GradScaler to verify the full pipeline works
    let w = Var::new(DynTensor::from_vec(vec![0.5f32, -0.3], &[2], &cpu()).unwrap());
    let mut scaler = GradScaler::new(GradScalerConfig {
        init_scale: 256.0,
        growth_interval: 100,
        ..Default::default()
    })
    .unwrap();

    let lr = 0.01f32;
    let mut losses = Vec::new();

    for _ in 0..10 {
        let tw = Arc::new(TrackedTensor::from_var(&w).unwrap());
        let loss = tw.sqr().unwrap().sum_keepdim(0).unwrap(); // loss = w0^2 + w1^2

        // Save loss value
        losses.push(loss.tensor().to_flat_vec::<f32>().unwrap()[0]);

        // Scale, backward, unscale, step
        let scaled = scaler.scale_loss(&loss).unwrap();
        let mut grads = backward(&scaled).unwrap();
        if scaler.unscale_and_check(&mut grads).unwrap() {
            // Manual SGD step: w = w - lr * grad
            let g = grads.get(&w).unwrap();
            let new_w = w
                .data()
                .unwrap()
                .sub(&g.mul_scalar(f64::from(lr)).unwrap())
                .unwrap();
            w.set(&new_w).unwrap();
        }
        scaler.update();
    }

    // Verify loss decreased
    assert!(
        losses.last().unwrap() < losses.first().unwrap(),
        "loss should decrease: first={}, last={}",
        losses.first().unwrap(),
        losses.last().unwrap()
    );
}

#[test]
fn test_invalid_config_zero_scale() {
    let err = GradScaler::new(GradScalerConfig {
        init_scale: 0.0,
        ..Default::default()
    })
    .unwrap_err();
    let msg = format!("{err}");
    assert!(
        msg.contains("init_scale"),
        "error should mention init_scale: {msg}"
    );
}

#[test]
fn test_invalid_config_negative_scale() {
    let err = GradScaler::new(GradScalerConfig {
        init_scale: -1.0,
        ..Default::default()
    })
    .unwrap_err();
    let msg = format!("{err}");
    assert!(
        msg.contains("init_scale"),
        "error should mention init_scale: {msg}"
    );
}

#[test]
fn test_invalid_config_growth_factor_le_one() {
    let err = GradScaler::new(GradScalerConfig {
        growth_factor: 1.0,
        ..Default::default()
    })
    .unwrap_err();
    let msg = format!("{err}");
    assert!(
        msg.contains("growth_factor"),
        "error should mention growth_factor: {msg}"
    );
}

#[test]
fn test_invalid_config_backoff_factor_ge_one() {
    let err = GradScaler::new(GradScalerConfig {
        backoff_factor: 1.0,
        ..Default::default()
    })
    .unwrap_err();
    let msg = format!("{err}");
    assert!(
        msg.contains("backoff_factor"),
        "error should mention backoff_factor: {msg}"
    );
}

#[test]
fn test_invalid_config_min_gt_max() {
    let err = GradScaler::new(GradScalerConfig {
        min_scale: 100.0,
        max_scale: 10.0,
        ..Default::default()
    })
    .unwrap_err();
    let msg = format!("{err}");
    assert!(
        msg.contains("max_scale"),
        "error should mention max_scale: {msg}"
    );
}

// State persistence and checkpoint tests are in `grad_scaler_state_tests.rs`.
// Config validation edge cases are in `grad_scaler_state_tests.rs`.

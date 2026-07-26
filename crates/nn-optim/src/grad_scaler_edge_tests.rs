#![allow(deprecated)]
// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Edge-case tests for GradScaler: bf16 training, NaN/Inf config validation.

use super::*;
use nn_autodiff::{backward, TrackedTensor, Var};
use nn_core::dyn_tensor::DynTensor;
use nn_core::test_utils::cpu;
use nn_core::DType;
use std::sync::Arc;

/// AC2: Train a 2-layer MLP with bf16-labeled inputs through GradScaler.
/// Verifies the full mixed-precision pipeline: bf16 input → f32 forward →
/// scaled backward → unscale → SGD step → loss decreases.
#[test]
fn test_bf16_input_mlp_training() {
    // Create bf16-labeled input (internally f32 with precision loss simulation)
    let input_f32 = DynTensor::from_vec(vec![1.0f32, 0.5, -0.3, 0.8], &[2, 2], &cpu()).unwrap();
    let input_bf16 = input_f32.to_dtype(DType::BF16).unwrap();
    // Extract the precision-lossy f32 values for use with TrackedTensor
    let bf16_data = input_bf16.to_flat_vec::<f32>().unwrap();

    // 2-layer MLP weights (f32)
    let w1 = Var::new(DynTensor::from_vec(vec![0.1, 0.2, 0.3, 0.4], &[2, 2], &cpu()).unwrap());
    let w2 = Var::new(DynTensor::from_vec(vec![0.5, -0.5], &[2, 1], &cpu()).unwrap());

    let mut scaler = GradScaler::new(GradScalerConfig {
        init_scale: 256.0,
        growth_interval: 100,
        ..Default::default()
    })
    .unwrap();

    let lr = 0.01f32;
    let mut losses = Vec::new();

    for _ in 0..20 {
        // Forward: bf16_input → matmul(W1) → relu → matmul(W2) → sqr → sum (loss)
        let tx = Arc::new(TrackedTensor::from_tensor(
            DynTensor::from_vec(bf16_data.clone(), &[2, 2], &cpu()).unwrap(),
        ));
        let tw1 = Arc::new(TrackedTensor::from_var(&w1).unwrap());
        let tw2 = Arc::new(TrackedTensor::from_var(&w2).unwrap());

        let h = tx.matmul(&tw1).unwrap(); // [2, 2]
        let h = h.relu().unwrap(); // [2, 2]
        let out = h.matmul(&tw2).unwrap(); // [2, 1]
        let loss = out.sqr().unwrap().sum_keepdim(0).unwrap();

        losses.push(loss.tensor().to_flat_vec::<f32>().unwrap()[0]);

        // GradScaler pipeline
        let scaled = scaler.scale_loss(&loss).unwrap();
        let mut grads = backward(&scaled).unwrap();
        if scaler.unscale_and_check(&mut grads).unwrap() {
            for var in [&w1, &w2] {
                if let Some(g) = grads.get(var) {
                    let new_w = var
                        .data()
                        .unwrap()
                        .sub(&g.mul_scalar(f64::from(lr)).unwrap())
                        .unwrap();
                    var.set(&new_w).unwrap();
                }
            }
        }
        scaler.update();
    }

    // Loss should decrease over training
    assert!(
        losses.last().unwrap() < losses.first().unwrap(),
        "bf16 MLP training loss should decrease: first={}, last={}",
        losses.first().unwrap(),
        losses.last().unwrap()
    );
}

/// AC3: Verify that gradients from bf16-labeled inputs match f32-only
/// gradients within tolerance. Since DynTensor stores bf16 as f32 internally
/// (with precision-loss simulation via bf16 round-trip), gradients should be
/// nearly identical for representable values.
#[test]
fn test_bf16_vs_f32_gradient_parity() {
    let input_data = vec![1.0f32, 0.5, -0.3, 0.8];
    let w_data = vec![0.1f32, 0.2, 0.3, 0.4];

    // --- f32 path ---
    let w_f32 = Var::new(DynTensor::from_vec(w_data.clone(), &[2, 2], &cpu()).unwrap());
    let x_f32 = Arc::new(TrackedTensor::from_tensor(
        DynTensor::from_vec(input_data.clone(), &[2, 2], &cpu()).unwrap(),
    ));
    let tw_f32 = Arc::new(TrackedTensor::from_var(&w_f32).unwrap());
    let loss_f32 = x_f32
        .matmul(&tw_f32)
        .unwrap()
        .sqr()
        .unwrap()
        .sum_keepdim(1)
        .unwrap()
        .sum_keepdim(0)
        .unwrap();
    let grads_f32 = backward(&loss_f32).unwrap();
    let grad_f32 = grads_f32.get(&w_f32).unwrap().to_flat_vec::<f32>().unwrap();

    // --- bf16 path (simulated precision loss on input) ---
    let input_bf16 = DynTensor::from_vec(input_data, &[2, 2], &cpu())
        .unwrap()
        .to_dtype(DType::BF16)
        .unwrap();
    let bf16_data = input_bf16.to_flat_vec::<f32>().unwrap();

    let w_bf16 = Var::new(DynTensor::from_vec(w_data, &[2, 2], &cpu()).unwrap());
    let x_bf16 = Arc::new(TrackedTensor::from_tensor(
        DynTensor::from_vec(bf16_data, &[2, 2], &cpu()).unwrap(),
    ));
    let tw_bf16 = Arc::new(TrackedTensor::from_var(&w_bf16).unwrap());
    let loss_bf16 = x_bf16
        .matmul(&tw_bf16)
        .unwrap()
        .sqr()
        .unwrap()
        .sum_keepdim(1)
        .unwrap()
        .sum_keepdim(0)
        .unwrap();
    let grads_bf16 = backward(&loss_bf16).unwrap();
    let grad_bf16 = grads_bf16
        .get(&w_bf16)
        .unwrap()
        .to_flat_vec::<f32>()
        .unwrap();

    // Gradients should be very close — bf16 precision loss is small for these values
    assert_eq!(grad_f32.len(), grad_bf16.len());
    for (i, (f32_val, bf16_val)) in grad_f32.iter().zip(&grad_bf16).enumerate() {
        let abs_diff = (f32_val - bf16_val).abs();
        assert!(
            abs_diff < 0.01,
            "gradient[{i}] diverged: f32={f32_val}, bf16={bf16_val}, diff={abs_diff}"
        );
    }
}

// -- NaN/Inf edge case tests for all config fields ---

#[test]
fn test_invalid_config_nan_init_scale() {
    let err = GradScaler::new(GradScalerConfig {
        init_scale: f64::NAN,
        ..Default::default()
    })
    .unwrap_err();
    let msg = format!("{err}");
    assert!(msg.contains("init_scale"), "NaN init_scale rejected: {msg}");
}

#[test]
fn test_invalid_config_inf_init_scale() {
    let err = GradScaler::new(GradScalerConfig {
        init_scale: f64::INFINITY,
        ..Default::default()
    })
    .unwrap_err();
    let msg = format!("{err}");
    assert!(msg.contains("init_scale"), "Inf init_scale rejected: {msg}");
}

#[test]
fn test_invalid_config_nan_growth_factor() {
    let err = GradScaler::new(GradScalerConfig {
        growth_factor: f64::NAN,
        ..Default::default()
    })
    .unwrap_err();
    let msg = format!("{err}");
    assert!(
        msg.contains("growth_factor"),
        "NaN growth_factor rejected: {msg}"
    );
}

#[test]
fn test_invalid_config_inf_growth_factor() {
    let err = GradScaler::new(GradScalerConfig {
        growth_factor: f64::INFINITY,
        ..Default::default()
    })
    .unwrap_err();
    let msg = format!("{err}");
    assert!(
        msg.contains("growth_factor"),
        "Inf growth_factor rejected: {msg}"
    );
}

#[test]
fn test_invalid_config_nan_backoff_factor() {
    let err = GradScaler::new(GradScalerConfig {
        backoff_factor: f64::NAN,
        ..Default::default()
    })
    .unwrap_err();
    let msg = format!("{err}");
    assert!(
        msg.contains("backoff_factor"),
        "NaN backoff_factor rejected: {msg}"
    );
}

#[test]
fn test_invalid_config_inf_backoff_factor() {
    let err = GradScaler::new(GradScalerConfig {
        backoff_factor: f64::INFINITY,
        ..Default::default()
    })
    .unwrap_err();
    let msg = format!("{err}");
    assert!(
        msg.contains("backoff_factor"),
        "Inf backoff_factor rejected: {msg}"
    );
}

#[test]
fn test_invalid_config_nan_min_scale() {
    let err = GradScaler::new(GradScalerConfig {
        min_scale: f64::NAN,
        ..Default::default()
    })
    .unwrap_err();
    let msg = format!("{err}");
    assert!(msg.contains("min_scale"), "NaN min_scale rejected: {msg}");
}

#[test]
fn test_invalid_config_nan_max_scale() {
    let err = GradScaler::new(GradScalerConfig {
        max_scale: f64::NAN,
        ..Default::default()
    })
    .unwrap_err();
    let msg = format!("{err}");
    assert!(msg.contains("max_scale"), "NaN max_scale rejected: {msg}");
}

#[test]
fn test_invalid_config_neg_inf_init_scale() {
    let err = GradScaler::new(GradScalerConfig {
        init_scale: f64::NEG_INFINITY,
        ..Default::default()
    })
    .unwrap_err();
    let msg = format!("{err}");
    assert!(
        msg.contains("init_scale"),
        "NEG_INFINITY init_scale rejected: {msg}"
    );
}

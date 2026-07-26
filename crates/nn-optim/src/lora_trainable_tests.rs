#![allow(deprecated)]
// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Tests for TrainableLoraLinear — LoRA with gradient tracking for training.

use super::*;
use nn_autodiff::grad::backward;
use nn_autodiff::tracked::TrackedTensor;
use nn_core::dyn_tensor::DynTensor;
use nn_core::layers::Module;
use nn_core::test_utils::{cpu, make_linear, make_linear_with_bias};
use std::sync::Arc;

// ---- Construction tests ----

#[test]
fn test_trainable_lora_from_linear() {
    let linear = make_linear(4, 8);
    let lora = TrainableLoraLinear::from_linear(&linear, 4, 4.0).unwrap();
    assert_eq!(lora.vars().len(), 2);
    assert_eq!(lora.lora_a().dims().unwrap(), &[4, 8]);
    assert_eq!(lora.lora_b().dims().unwrap(), &[4, 4]);
    assert!((lora.scaling() - 1.0).abs() < 1e-10);
}

#[test]
fn test_trainable_lora_from_lora_linear() {
    let linear = make_linear(4, 8);
    let inference_lora = LoraLinear::from_linear(&linear, 4, 4.0).unwrap();
    let trainable_lora = TrainableLoraLinear::from_lora_linear(&inference_lora).unwrap();
    assert_eq!(trainable_lora.vars().len(), 2);
    assert!((trainable_lora.scaling() - inference_lora.scaling()).abs() < 1e-10);
}

// ---- Forward pass tests ----

#[test]
fn test_trainable_lora_forward_shape() {
    let linear = make_linear(4, 8);
    let lora = TrainableLoraLinear::from_linear(&linear, 4, 4.0).unwrap();

    let x_data = vec![1.0f32; 2 * 8]; // [2, 8]
    let x = DynTensor::from_vec(x_data, &[2, 8], &cpu()).unwrap();
    let x_tracked = Arc::new(TrackedTensor::from_tensor(x));

    let y = lora.forward(&x_tracked).unwrap();
    assert_eq!(y.tensor().dims(), &[2, 4]);
}

#[test]
fn test_trainable_lora_forward_with_bias_shape() {
    let linear = make_linear_with_bias(4, 8);
    let lora = TrainableLoraLinear::from_linear(&linear, 4, 4.0).unwrap();

    let x_data = vec![1.0f32; 2 * 8];
    let x = DynTensor::from_vec(x_data, &[2, 8], &cpu()).unwrap();
    let x_tracked = Arc::new(TrackedTensor::from_tensor(x));

    let y = lora.forward(&x_tracked).unwrap();
    assert_eq!(y.tensor().dims(), &[2, 4]);
}

#[test]
fn test_trainable_lora_zero_init_matches_base() {
    // B is zero-initialized, so initial LoRA output == base linear output
    let linear = make_linear(4, 8);
    let lora = TrainableLoraLinear::from_linear(&linear, 4, 4.0).unwrap();

    let x_data: Vec<f32> = (0..16).map(|i| i as f32 * 0.1).collect();
    let x = DynTensor::from_vec(x_data, &[2, 8], &cpu()).unwrap();

    // Inference path
    let y_base = linear.forward(&x).unwrap();

    // Training path
    let x_tracked = Arc::new(TrackedTensor::from_tensor(x));
    let y_lora = lora.forward(&x_tracked).unwrap();

    let base_vals = y_base.to_flat_vec::<f32>().unwrap();
    let lora_vals = y_lora.tensor().to_flat_vec::<f32>().unwrap();
    for (i, (b, l)) in base_vals.iter().zip(lora_vals.iter()).enumerate() {
        assert!(
            (b - l).abs() < 1e-5,
            "mismatch at [{i}]: base={b}, lora={l}"
        );
    }
}

// ---- Backward pass tests ----

#[test]
fn test_trainable_lora_backward_produces_gradients() {
    let linear = make_linear(3, 4);
    let lora = TrainableLoraLinear::from_linear(&linear, 2, 2.0).unwrap();

    let x_data: Vec<f32> = (0..8).map(|i| i as f32 * 0.1).collect();
    let x = DynTensor::from_vec(x_data, &[2, 4], &cpu()).unwrap();
    let x_tracked = Arc::new(TrackedTensor::from_tensor(x));

    let y = lora.forward(&x_tracked).unwrap();
    // Sum all outputs to get scalar loss
    let loss = y.sum_keepdim(0).unwrap().sum_keepdim(1).unwrap();
    let grads = backward(&loss).unwrap();

    // Both A and B should have gradients
    let grad_a = grads.get(lora.lora_a()).expect("grad_a missing");
    let grad_b = grads.get(lora.lora_b()).expect("grad_b missing");
    assert_eq!(grad_a.dims(), &[2, 4]); // [rank, in_features]
    assert_eq!(grad_b.dims(), &[3, 2]); // [out_features, rank]
}

#[test]
fn test_trainable_lora_frozen_weight_no_gradient() {
    // The frozen weight should NOT appear in gradients (it's from_tensor, not from_var)
    let linear = make_linear(3, 4);
    let lora = TrainableLoraLinear::from_linear(&linear, 2, 2.0).unwrap();

    let x_var = Var::new(DynTensor::from_vec(vec![1.0f32; 8], &[2, 4], &cpu()).unwrap());
    let x_tracked = Arc::new(TrackedTensor::from_var(&x_var).unwrap());

    let y = lora.forward(&x_tracked).unwrap();
    let loss = y.sum_keepdim(0).unwrap().sum_keepdim(1).unwrap();
    let grads = backward(&loss).unwrap();

    // x_var gets gradients (it's a Var)
    assert!(grads.get(&x_var).is_some());
    // lora_a and lora_b get gradients
    assert!(grads.get(lora.lora_a()).is_some());
    assert!(grads.get(lora.lora_b()).is_some());
    // Total number of Var gradients must be exactly 3 (x, A, B).
    // Frozen weight is from_tensor, so no gradient entry.
    // If someone changed from_tensor to from_var in forward(), this count would be 4.
    let grad_count = grads.var_grads().count();
    assert_eq!(
        grad_count, 3,
        "expected exactly 3 var gradients (x, A, B), got {grad_count} — frozen weight may be leaking gradients"
    );
}

#[test]
fn test_trainable_lora_gradient_finite() {
    let linear = make_linear(3, 4);
    let lora = TrainableLoraLinear::from_linear(&linear, 2, 2.0).unwrap();

    let x_data: Vec<f32> = (0..8).map(|i| (i as f32 + 1.0) * 0.1).collect();
    let x = DynTensor::from_vec(x_data, &[2, 4], &cpu()).unwrap();
    let x_tracked = Arc::new(TrackedTensor::from_tensor(x));

    let y = lora.forward(&x_tracked).unwrap();
    let loss = y.sum_keepdim(0).unwrap().sum_keepdim(1).unwrap();
    let grads = backward(&loss).unwrap();

    for var in lora.vars() {
        let g = grads.get(var).unwrap();
        let g_data = g.to_flat_vec::<f32>().unwrap();
        for (i, &v) in g_data.iter().enumerate() {
            assert!(v.is_finite(), "non-finite gradient at [{i}]: {v}");
        }
    }
}

// ---- Merge tests ----

#[test]
fn test_trainable_lora_merge() {
    let linear = make_linear(3, 4);
    let lora = TrainableLoraLinear::from_linear(&linear, 2, 2.0).unwrap();
    let merged = lora.merge().unwrap();
    assert_eq!(merged.dims(), &[3, 4]);
}

// ---- Training loop integration test ----

#[test]
fn test_trainable_lora_training_step() {
    use crate::adam::{AdamConfig, AdamW};
    use crate::optimizer::Optimizer;

    let linear = make_linear(3, 4);
    let lora = TrainableLoraLinear::from_linear(&linear, 2, 2.0).unwrap();

    // Create optimizer over LoRA vars
    let config = AdamConfig {
        lr: 0.01,
        ..AdamConfig::default()
    };
    let vars: Vec<_> = lora.vars().into_iter().cloned().collect();
    let mut optim = AdamW::new(vars, config).unwrap();

    // Training step
    let x_data: Vec<f32> = (0..8).map(|i| (i as f32 + 1.0) * 0.1).collect();
    let x = DynTensor::from_vec(x_data, &[2, 4], &cpu()).unwrap();
    let x_tracked = Arc::new(TrackedTensor::from_tensor(x));

    let y = lora.forward(&x_tracked).unwrap();
    let loss = y.sum_keepdim(0).unwrap().sum_keepdim(1).unwrap();

    // Backward + optimizer step
    optim.backward_step(&loss).unwrap();

    // Verify B is no longer all zeros (optimizer moved it)
    let b_data = lora.lora_b().data().unwrap().to_flat_vec::<f32>().unwrap();
    let all_zero = b_data.iter().all(|v| *v == 0.0);
    assert!(!all_zero, "lora_b should change after optimizer step");
}

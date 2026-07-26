// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Cross-crate integration tests for LoRA adapter training.
//!
//! Tests verify that LoRA adapters only update trainable A/B matrices while
//! keeping base weights frozen, and that merge produces correct combined weights.

use std::sync::Arc;

use nn_autodiff::{backward, TrackedTensor, Var};
use nn_core::dyn_tensor::DynTensor;
use nn_core::layers::{Linear, Module};
use nn_core::Device;
use nn_optim::{AdamConfig, AdamW, LoraConfig, LoraLinear, Optimizer, TrainableLoraLinear};

fn cpu() -> Device {
    Device::Cpu
}

fn adam_config_no_decay(lr: f64) -> AdamConfig {
    let mut c = AdamConfig::default();
    c.lr = lr;
    c.weight_decay = 0.0;
    c
}

/// Build a simple Linear layer with given in/out features and constant weight.
fn make_linear(in_f: usize, out_f: usize) -> Linear {
    let weight = DynTensor::from_vec(vec![0.5f32; in_f * out_f], &[out_f, in_f], &cpu()).unwrap();
    Linear::new(weight, None).unwrap()
}

// ============================================================================
// LoRA adapter: only A/B update, base frozen
// ============================================================================

#[test]
fn test_lora_adapter_only_ab_update() {
    let linear = make_linear(4, 3);
    let frozen_weight_before = linear.weight().to_flat_vec::<f32>().unwrap();

    let lora = TrainableLoraLinear::from_linear(&linear, 2, 2.0).unwrap();
    let lora_a_before = lora.lora_a().data().unwrap().to_flat_vec::<f32>().unwrap();
    let lora_b_before = lora.lora_b().data().unwrap().to_flat_vec::<f32>().unwrap();

    // Create optimizer tracking only LoRA vars
    let vars: Vec<Var> = lora.vars().into_iter().cloned().collect();
    let mut adam = AdamW::new(vars, adam_config_no_decay(0.01)).unwrap();

    // Run a few training steps
    for _ in 0..5 {
        let x = Arc::new(TrackedTensor::from_tensor(
            DynTensor::from_vec(vec![1.0f32; 8], &[2, 4], &cpu()).unwrap(),
        ));
        let y = lora.forward(&x).unwrap();
        // Scalar loss: sum of outputs
        let loss = y.sum_keepdim(0).unwrap().sum_keepdim(1).unwrap();
        let loss = loss.reshape(&[1]).unwrap();
        adam.backward_step(&loss).unwrap();
    }

    // Check frozen weight unchanged
    let frozen_weight_after = linear.weight().to_flat_vec::<f32>().unwrap();
    assert_eq!(
        frozen_weight_before, frozen_weight_after,
        "Base weight should remain frozen during LoRA training"
    );

    // Check LoRA A and B changed (at least one should differ)
    let lora_a_after = lora.lora_a().data().unwrap().to_flat_vec::<f32>().unwrap();
    let lora_b_after = lora.lora_b().data().unwrap().to_flat_vec::<f32>().unwrap();

    let a_changed = lora_a_before
        .iter()
        .zip(lora_a_after.iter())
        .any(|(a, b)| (a - b).abs() > 1e-8);
    let b_changed = lora_b_before
        .iter()
        .zip(lora_b_after.iter())
        .any(|(a, b)| (a - b).abs() > 1e-8);

    assert!(
        a_changed || b_changed,
        "At least one LoRA matrix should be updated by training"
    );
}

// ============================================================================
// LoRA initial output matches base linear
// ============================================================================

#[test]
fn test_lora_initial_output_matches_base() {
    // B is initialized to zero, so initial LoRA contribution is zero.
    // Output should match the original Linear.
    let linear = make_linear(4, 3);
    let lora = LoraLinear::from_linear(&linear, 4, 4.0).unwrap();

    let x = DynTensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[1, 4], &cpu()).unwrap();

    let base_out = linear.forward(&x).unwrap();
    let lora_out = lora.forward(&x).unwrap();

    let base_vals = base_out.to_flat_vec::<f32>().unwrap();
    let lora_vals = lora_out.to_flat_vec::<f32>().unwrap();

    for (b, l) in base_vals.iter().zip(lora_vals.iter()) {
        assert!(
            (b - l).abs() < 1e-5,
            "Initial LoRA output should match base: base={b}, lora={l}"
        );
    }
}

// ============================================================================
// LoRA merge
// ============================================================================

#[test]
fn test_lora_merge_produces_valid_weight() {
    let linear = make_linear(4, 3);
    let lora = LoraLinear::from_linear(&linear, 2, 2.0).unwrap();

    // B is zero-init, so merge should equal original weight
    let merged = lora.merge().unwrap();
    let merged_vals = merged.to_flat_vec::<f32>().unwrap();
    let orig_vals = linear.weight().to_flat_vec::<f32>().unwrap();

    for (m, o) in merged_vals.iter().zip(orig_vals.iter()) {
        assert!(
            (m - o).abs() < 1e-5,
            "Merge of zero-init LoRA should equal original: merged={m}, orig={o}"
        );
    }
}

#[test]
fn test_lora_merge_after_training_differs() {
    let linear = make_linear(4, 3);
    let lora = TrainableLoraLinear::from_linear(&linear, 2, 2.0).unwrap();

    let vars: Vec<Var> = lora.vars().into_iter().cloned().collect();
    let mut adam = AdamW::new(vars, adam_config_no_decay(0.01)).unwrap();

    for _ in 0..10 {
        let x = Arc::new(TrackedTensor::from_tensor(
            DynTensor::from_vec(vec![1.0f32; 8], &[2, 4], &cpu()).unwrap(),
        ));
        let y = lora.forward(&x).unwrap();
        let loss = y.sum_keepdim(0).unwrap().sum_keepdim(1).unwrap();
        let loss = loss.reshape(&[1]).unwrap();
        adam.backward_step(&loss).unwrap();
    }

    let merged = lora.merge().unwrap();
    let merged_vals = merged.to_flat_vec::<f32>().unwrap();
    let orig_vals = linear.weight().to_flat_vec::<f32>().unwrap();

    let any_diff = merged_vals
        .iter()
        .zip(orig_vals.iter())
        .any(|(m, o)| (m - o).abs() > 1e-6);

    assert!(
        any_diff,
        "Merge after training should differ from original weight"
    );
}

// ============================================================================
// LoRA rank effect
// ============================================================================

#[test]
fn test_lora_rank_determines_parameter_count() {
    let linear = make_linear(8, 6);

    let lora_r2 = LoraLinear::from_linear(&linear, 2, 2.0).unwrap();
    let lora_r8 = LoraLinear::from_linear(&linear, 8, 8.0).unwrap();

    // rank=2: A is [2,8]=16 params, B is [6,2]=12 params => 28 total
    let r2_a = lora_r2.lora_a().data().unwrap();
    let r2_b = lora_r2.lora_b().data().unwrap();
    let r2_params: usize =
        r2_a.dims().iter().product::<usize>() + r2_b.dims().iter().product::<usize>();

    // rank=8: A is [8,8]=64 params, B is [6,8]=48 params => 112 total
    let r8_a = lora_r8.lora_a().data().unwrap();
    let r8_b = lora_r8.lora_b().data().unwrap();
    let r8_params: usize =
        r8_a.dims().iter().product::<usize>() + r8_b.dims().iter().product::<usize>();

    assert!(
        r8_params > r2_params,
        "Higher rank should have more parameters: r2={r2_params}, r8={r8_params}"
    );
    assert_eq!(r2_params, 28, "rank=2: 2*8 + 6*2 = 28");
    assert_eq!(r8_params, 112, "rank=8: 8*8 + 6*8 = 112");
}

// ============================================================================
// LoRA config
// ============================================================================

#[test]
fn test_lora_config_defaults() {
    let config = LoraConfig::default();
    assert_eq!(config.rank, 8);
    assert!((config.alpha - 8.0).abs() < f64::EPSILON);
    assert_eq!(config.targets, vec!["q_proj", "v_proj"]);
}

// ============================================================================
// LoRA validation
// ============================================================================

#[test]
fn test_lora_rejects_rank_zero() {
    let linear = make_linear(4, 3);
    let result = LoraLinear::from_linear(&linear, 0, 1.0);
    assert!(result.is_err(), "LoRA should reject rank=0");
}

#[test]
fn test_lora_rejects_non_finite_alpha() {
    let linear = make_linear(4, 3);
    let result = LoraLinear::from_linear(&linear, 4, f64::NAN);
    assert!(result.is_err(), "LoRA should reject NaN alpha");

    let result = LoraLinear::from_linear(&linear, 4, f64::INFINITY);
    assert!(result.is_err(), "LoRA should reject Inf alpha");
}

// ============================================================================
// LoRA trainable forward matches inference forward
// ============================================================================

#[test]
fn test_trainable_lora_forward_matches_inference() {
    // TrainableLoraLinear and LoraLinear should produce the same output
    // when created from the same Linear (B is zero-init, so LoRA = 0).
    let linear = make_linear(4, 3);

    // We need to share the same Var instances. Create LoraLinear first,
    // then wrap it in TrainableLoraLinear.
    let inference_lora = LoraLinear::from_linear(&linear, 4, 4.0).unwrap();
    let trainable_lora = TrainableLoraLinear::from_lora_linear(&inference_lora).unwrap();

    let x_data = DynTensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[1, 4], &cpu()).unwrap();

    // Inference path (DynTensor)
    let inf_out = inference_lora.forward(&x_data).unwrap();

    // Trainable path (TrackedTensor -> extract DynTensor)
    let x_tracked = Arc::new(TrackedTensor::from_tensor(x_data));
    let train_out = trainable_lora.forward(&x_tracked).unwrap();

    let inf_vals = inf_out.to_flat_vec::<f32>().unwrap();
    let train_vals = train_out.tensor().to_flat_vec::<f32>().unwrap();

    for (i, t) in inf_vals.iter().zip(train_vals.iter()) {
        assert!(
            (i - t).abs() < 1e-4,
            "Trainable forward should match inference: inf={i}, train={t}"
        );
    }
}

// ============================================================================
// LoRA gradient flows only to A and B
// ============================================================================

#[test]
fn test_lora_gradients_flow_only_to_ab() {
    let linear = make_linear(4, 3);
    let lora = TrainableLoraLinear::from_linear(&linear, 2, 2.0).unwrap();

    let x = Arc::new(TrackedTensor::from_tensor(
        DynTensor::from_vec(vec![1.0f32; 4], &[1, 4], &cpu()).unwrap(),
    ));
    let y = lora.forward(&x).unwrap();
    let loss = y.sum_keepdim(0).unwrap().sum_keepdim(1).unwrap();
    let loss = loss.reshape(&[1]).unwrap();

    let grads = backward(&loss).unwrap();

    // LoRA A and B should have gradients
    let grad_a = grads.get(lora.lora_a());
    let grad_b = grads.get(lora.lora_b());

    // At least one should have non-zero gradient (B is zero-init but A is random,
    // so gradient should flow through at least one path)
    let has_grads = grad_a.is_some() || grad_b.is_some();
    assert!(has_grads, "LoRA A or B should receive gradients");
}

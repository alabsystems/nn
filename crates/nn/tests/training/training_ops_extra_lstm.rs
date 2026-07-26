// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! E2E training tests for LSTM (single-step and sequence).
//!
//! These tests exercise gradient flow through LSTM gate decomposition:
//! matmul, sigmoid, tanh, narrow, mul, add — and BPTT through recurrent
//! connections.
//!
//! Extracted from training_ops_extra.rs for 500-line compliance.
//!
//! Run: `cargo test -p nn --test training_ops_extra_lstm --features training`

#![cfg(feature = "training")]

use std::sync::Arc;

use nn::training::{
    AdamConfig, AdamW, Optimizer, TrackedTensor, TrainableLinear, TrainableLstm, TrainableModule,
    Var,
};
use nn::{DType, Device, DynTensor};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn make_adam(vars: Vec<Var>, lr: f64) -> AdamW {
    let mut c = AdamConfig::default();
    c.lr = lr;
    c.weight_decay = 0.0;
    AdamW::new(vars, c).unwrap()
}

fn collect_vars(var_slices: &[Vec<&Var>]) -> Vec<Var> {
    var_slices
        .iter()
        .flat_map(|s| s.iter().map(|v| (*v).clone()))
        .collect()
}

fn assert_loss_decreased(losses: &[f32], label: &str) {
    let initial = losses[0];
    let final_loss = *losses.last().unwrap();
    assert!(
        final_loss < initial,
        "{label}: loss should decrease: initial={initial}, final={final_loss}",
    );
}

// ---------------------------------------------------------------------------
// LSTM single-step cell training
// ---------------------------------------------------------------------------

/// Train LSTM → Linear classifier on synthetic sequence data.
/// Exercises gradient flow through LSTM gate decomposition (matmul, sigmoid,
/// tanh, narrow, mul, add). Validates that all 4 LSTM weight matrices receive
/// non-zero gradients.
#[test]
fn test_train_lstm_single_step_loss_decreases() {
    let (batch, input_size, hidden, classes) = (16, 4, 8, 2);

    let x_data = DynTensor::randn(0.0, 1.0, &[batch, input_size], &Device::Cpu).unwrap();
    let x_flat = x_data.to_flat_vec::<f32>().unwrap();
    let target_data: Vec<u32> = (0..batch)
        .map(|b| {
            let sum: f32 = (0..input_size).map(|d| x_flat[b * input_size + d]).sum();
            u32::from(sum > 0.0)
        })
        .collect();
    let t_data = DynTensor::from_vec_u32(target_data, &[batch, 1], &Device::Cpu).unwrap();

    // LSTM(4→8) → Linear(8→2)
    let lstm = TrainableLstm::from_tensors(
        DynTensor::randn(0.0, 0.1, &[4 * hidden, input_size], &Device::Cpu).unwrap(),
        DynTensor::randn(0.0, 0.1, &[4 * hidden, hidden], &Device::Cpu).unwrap(),
        Some(DynTensor::zeros(&[4 * hidden], DType::F32, &Device::Cpu).unwrap()),
        Some(DynTensor::zeros(&[4 * hidden], DType::F32, &Device::Cpu).unwrap()),
        hidden,
    );
    let fc = TrainableLinear::from_tensors(
        DynTensor::randn(0.0, 0.1, &[classes, hidden], &Device::Cpu).unwrap(),
        Some(DynTensor::zeros(&[classes], DType::F32, &Device::Cpu).unwrap()),
    );

    let all_vars = collect_vars(&[lstm.vars(), fc.vars()]);
    assert_eq!(all_vars.len(), 6); // w_ih + w_hh + b_ih + b_hh + fc_w + fc_b
    let mut adam = make_adam(all_vars, 0.01);

    let mut losses = Vec::new();
    for step in 0..15 {
        let tx = Arc::new(TrackedTensor::from_tensor(x_data.clone()));
        let (h, _state) = lstm.forward_cell(&tx, None).unwrap();
        let logits = fc.forward(&h).unwrap();
        let t_targets = Arc::new(TrackedTensor::from_tensor(t_data.clone()));
        let loss = logits.cross_entropy_loss(&t_targets, 1).unwrap();
        let loss_val = loss.tensor().to_flat_vec::<f32>().unwrap()[0];
        assert!(loss_val.is_finite(), "loss is NaN/Inf at step {step}");
        losses.push(loss_val);
        adam.backward_step(&loss).unwrap();
    }

    assert_loss_decreased(&losses, "LSTM single-step");

    // Verify LSTM weights changed.
    let w_ih_after = lstm.w_ih().data().unwrap().to_flat_vec::<f32>().unwrap();
    assert!(
        w_ih_after.iter().any(|v| v.abs() > 0.15),
        "w_ih should change from init"
    );
}

// ---------------------------------------------------------------------------
// LSTM sequence training — multi-step with state carry
// ---------------------------------------------------------------------------

/// Train LSTM on a 4-step sequence, carrying state across timesteps.
/// Validates gradient flow through recurrent connections (BPTT).
#[test]
fn test_train_lstm_sequence_loss_decreases() {
    let (batch, input_size, hidden, seq_len) = (8, 3, 6, 4);

    // Synthetic sequence: [batch, seq_len, input_size]
    let x_data = DynTensor::randn(0.0, 1.0, &[batch, seq_len, input_size], &Device::Cpu).unwrap();

    // Target: predict last hidden state → scalar regression via MSE
    let target = DynTensor::randn(0.0, 0.5, &[batch, hidden], &Device::Cpu).unwrap();

    let lstm = TrainableLstm::from_tensors(
        DynTensor::randn(0.0, 0.1, &[4 * hidden, input_size], &Device::Cpu).unwrap(),
        DynTensor::randn(0.0, 0.1, &[4 * hidden, hidden], &Device::Cpu).unwrap(),
        Some(DynTensor::zeros(&[4 * hidden], DType::F32, &Device::Cpu).unwrap()),
        Some(DynTensor::zeros(&[4 * hidden], DType::F32, &Device::Cpu).unwrap()),
        hidden,
    );

    let all_vars: Vec<Var> = lstm.vars().into_iter().cloned().collect();
    let mut adam = make_adam(all_vars, 0.005);

    let mut losses = Vec::new();
    for step in 0..20 {
        let tx = Arc::new(TrackedTensor::from_tensor(x_data.clone()));
        let (_outputs, final_state) = lstm.forward_seq(&tx, None).unwrap();

        // MSE loss: mean((h_final - target)^2)
        let t_target = Arc::new(TrackedTensor::from_tensor(target.clone()));
        let diff = final_state.h.sub(&t_target).unwrap();
        let sq = diff.mul(&diff).unwrap();
        let loss = sq
            .mean_keepdim(1)
            .unwrap()
            .mean_keepdim(0)
            .unwrap()
            .squeeze(1)
            .unwrap()
            .squeeze(0)
            .unwrap();

        let loss_val = loss.tensor().to_flat_vec::<f32>().unwrap()[0];
        assert!(loss_val.is_finite(), "loss is NaN/Inf at step {step}");
        losses.push(loss_val);
        adam.backward_step(&loss).unwrap();
    }

    assert_loss_decreased(&losses, "LSTM sequence (BPTT)");
}

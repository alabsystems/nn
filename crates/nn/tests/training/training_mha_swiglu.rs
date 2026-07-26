// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! E2E training tests for TrainableMultiHeadAttention and TrainableSwiGlu.
//!
//! Run: `cargo test -p nn --test training_mha_swiglu --features training`

#![cfg(feature = "training")]

use std::sync::Arc;

use nn::training::{
    AdamConfig, AdamW, Optimizer, TrackedTensor, TrainableLinear, TrainableModule,
    TrainableMultiHeadAttention, TrainableSwiGlu, Var,
};
use nn::{Device, DynTensor};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn make_adam(vars: Vec<Var>, lr: f64) -> AdamW {
    let mut c = AdamConfig::default();
    c.lr = lr;
    AdamW::new(vars, c).unwrap()
}

fn collect_vars(groups: &[Vec<&Var>]) -> Vec<Var> {
    groups
        .iter()
        .flat_map(|g| g.iter().map(|v| (*v).clone()))
        .collect()
}

fn assert_loss_decreased(losses: &[f32], label: &str) {
    let first = losses[0];
    let last = *losses.last().unwrap();
    assert!(
        last < first,
        "{label}: loss should decrease, first={first:.6} last={last:.6}"
    );
}

// ---------------------------------------------------------------------------
// TrainableMultiHeadAttention
// ---------------------------------------------------------------------------

/// Train MHA self-attention on a simple sequence-to-sequence regression task.
/// Input: [batch, seq_len, model_dim] -> MHA -> Linear -> [batch, seq_len, 1]
/// Validates gradient flow through Q/K/V/out projections, softmax, and matmul.
#[test]
fn test_train_mha_loss_decreases() {
    let (batch, seq_len, model_dim, num_heads) = (4, 6, 8, 2);

    let x_data = DynTensor::randn(0.0, 0.5, &[batch, seq_len, model_dim], &Device::Cpu).unwrap();
    let target = DynTensor::randn(0.0, 0.5, &[batch, seq_len, 1], &Device::Cpu).unwrap();

    let mha = TrainableMultiHeadAttention::zeros(model_dim, num_heads, true).unwrap();
    let fc = TrainableLinear::new(model_dim, 1, true).unwrap();

    let all_vars = collect_vars(&[mha.vars(), fc.vars()]);
    // MHA: 4 projections * (weight + bias) = 8 vars, FC: weight + bias = 2 vars
    assert_eq!(all_vars.len(), 10);
    let mut adam = make_adam(all_vars, 0.001);

    let mut losses = Vec::new();
    for step in 0..20 {
        let tx = Arc::new(TrackedTensor::from_tensor(x_data.clone()));
        let h = mha.forward(&tx).unwrap();
        let logits = fc.forward(&h).unwrap();

        // MSE loss
        let t_target = Arc::new(TrackedTensor::from_tensor(target.clone()));
        let diff = logits.sub(&t_target).unwrap();
        let sq = diff.mul(&diff).unwrap();
        let loss = sq
            .mean_keepdim(2)
            .unwrap()
            .mean_keepdim(1)
            .unwrap()
            .mean_keepdim(0)
            .unwrap()
            .squeeze(2)
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

    assert_loss_decreased(&losses, "MHA self-attention");
}

/// Verify MHA rejects invalid num_heads.
#[test]
fn test_mha_rejects_zero_heads() {
    let err = TrainableMultiHeadAttention::zeros(8, 0, true);
    assert!(err.is_err());
}

/// Verify MHA rejects non-divisible model_dim.
#[test]
fn test_mha_rejects_non_divisible_dim() {
    let err = TrainableMultiHeadAttention::zeros(7, 2, true);
    assert!(err.is_err());
}

// ---------------------------------------------------------------------------
// TrainableSwiGlu
// ---------------------------------------------------------------------------

/// Train SwiGlu FFN on a simple regression task.
/// Input: [batch, dim] -> SwiGlu -> Linear -> [batch, 1]
/// Validates gradient flow through gate (SiLU), up, and down projections.
#[test]
fn test_train_swiglu_loss_decreases() {
    let (batch, dim, hidden_dim) = (16, 8, 16);

    let x_data = DynTensor::randn(0.0, 0.5, &[batch, dim], &Device::Cpu).unwrap();
    let target = DynTensor::randn(0.0, 0.5, &[batch, 1], &Device::Cpu).unwrap();

    let swiglu = TrainableSwiGlu::zeros(dim, hidden_dim, false).unwrap();
    let fc = TrainableLinear::new(dim, 1, true).unwrap();

    let all_vars = collect_vars(&[swiglu.vars(), fc.vars()]);
    // SwiGlu: 3 linear (no bias) = 3 weight vars, FC: weight + bias = 2 vars
    assert_eq!(all_vars.len(), 5);
    let mut adam = make_adam(all_vars, 0.001);

    let mut losses = Vec::new();
    for step in 0..20 {
        let tx = Arc::new(TrackedTensor::from_tensor(x_data.clone()));
        let h = swiglu.forward(&tx).unwrap();
        let pred = fc.forward(&h).unwrap();

        // MSE loss
        let t_target = Arc::new(TrackedTensor::from_tensor(target.clone()));
        let diff = pred.sub(&t_target).unwrap();
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

    assert_loss_decreased(&losses, "SwiGlu FFN");
}

/// Train SwiGlu with bias enabled.
#[test]
fn test_train_swiglu_with_bias_loss_decreases() {
    let (batch, dim, hidden_dim) = (8, 4, 8);

    let x_data = DynTensor::randn(0.0, 0.5, &[batch, dim], &Device::Cpu).unwrap();
    let target = DynTensor::randn(0.0, 0.5, &[batch, 1], &Device::Cpu).unwrap();

    let swiglu = TrainableSwiGlu::zeros(dim, hidden_dim, true).unwrap();
    let fc = TrainableLinear::new(dim, 1, true).unwrap();

    let all_vars = collect_vars(&[swiglu.vars(), fc.vars()]);
    // SwiGlu with bias: 3 * (weight + bias) = 6, FC: 2
    assert_eq!(all_vars.len(), 8);
    let mut adam = make_adam(all_vars, 0.001);

    let mut losses = Vec::new();
    for step in 0..15 {
        let tx = Arc::new(TrackedTensor::from_tensor(x_data.clone()));
        let h = swiglu.forward(&tx).unwrap();
        let pred = fc.forward(&h).unwrap();

        let t_target = Arc::new(TrackedTensor::from_tensor(target.clone()));
        let diff = pred.sub(&t_target).unwrap();
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

    assert_loss_decreased(&losses, "SwiGlu FFN with bias");
}

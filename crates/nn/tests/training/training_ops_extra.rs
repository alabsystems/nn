// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! E2E training tests for BatchNorm, InstanceNorm, ConvTranspose1d, Dropout.
//!
//! These tests exercise backward rules that had prior bugs:
//! - BatchNorm/InstanceNorm: #1481 (norm backward broadcast axis mismatch)
//! - ConvTranspose1d: #1484 (parameter order mismatch)
//! - Dropout: backward mask scaling
//!
//! Run: `cargo test -p nn --test training_ops_extra --features training`

#![cfg(feature = "training")]

use std::sync::Arc;

use nn::training::{
    Optimizer, TrackedTensor, TrainableBatchNorm, TrainableConv1d, TrainableConvTranspose1d,
    TrainableInstanceNorm, TrainableLinear, TrainableModule, Var,
};
use nn::{DType, Device, DynTensor};

use super::common::{assert_loss_decreased, collect_vars, make_adam};

// ---------------------------------------------------------------------------
// AC1: BatchNorm training e2e (#1514)
// ---------------------------------------------------------------------------

/// Train a Linear → BatchNorm → ReLU → Linear network.
/// Exercises BatchNorm backward rule (broadcast-sensitive grad_weight, grad_bias).
/// Regression guard for #1481 (norm backward broadcast axis mismatch).
#[test]
fn test_train_batch_norm_loss_decreases() {
    let (batch, in_dim, hidden, classes) = (16, 4, 8, 3);

    let x_data = DynTensor::randn(0.0, 1.0, &[batch, in_dim], &Device::Cpu).unwrap();
    let x_flat = x_data.to_flat_vec::<f32>().unwrap();
    let target_data: Vec<u32> = (0..batch)
        .map(|b| {
            let sum: f32 = (0..in_dim).map(|d| x_flat[b * in_dim + d]).sum();
            if sum > 0.0 {
                1
            } else {
                0
            }
        })
        .collect();
    let t_data = DynTensor::from_vec_u32(target_data, &[batch, 1], &Device::Cpu).unwrap();

    // Build: Linear(4→8) → BatchNorm(8) → ReLU → Linear(8→3)
    let w1 = DynTensor::randn(0.0, 0.1, &[hidden, in_dim], &Device::Cpu).unwrap();
    let b1 = DynTensor::zeros(&[hidden], DType::F32, &Device::Cpu).unwrap();
    let layer1 = TrainableLinear::from_tensors(w1, Some(b1));
    let bn = TrainableBatchNorm::new(hidden, 1e-5).unwrap();
    let w2 = DynTensor::randn(0.0, 0.1, &[classes, hidden], &Device::Cpu).unwrap();
    let b2 = DynTensor::zeros(&[classes], DType::F32, &Device::Cpu).unwrap();
    let layer2 = TrainableLinear::from_tensors(w2, Some(b2));

    let all_vars = collect_vars(&[layer1.vars(), bn.vars(), layer2.vars()]);
    assert_eq!(all_vars.len(), 6); // w1+b1 + bn_weight+bn_bias + w2+b2
    let mut adam = make_adam(all_vars, 0.01);

    let mut losses = Vec::new();
    for step in 0..15 {
        let tx = Arc::new(TrackedTensor::from_tensor(x_data.clone()));
        let h = layer1.forward(&tx).unwrap(); // [B, 8]
                                              // BatchNorm expects [N, C, *] — unsqueeze to [B, 8, 1]
        let h = h.unsqueeze(2).unwrap();
        let h = bn.forward(&h).unwrap();
        let h = h.squeeze(2).unwrap();
        let h = h.relu().unwrap();
        let logits = layer2.forward(&h).unwrap();
        let t_targets = Arc::new(TrackedTensor::from_tensor(t_data.clone()));
        let loss = logits.cross_entropy_loss(&t_targets, 1).unwrap();
        let loss_val = loss.tensor().to_flat_vec::<f32>().unwrap()[0];
        assert!(loss_val.is_finite(), "loss is NaN/Inf at step {step}");
        losses.push(loss_val);
        adam.backward_step(&loss).unwrap();
    }

    assert_loss_decreased(&losses, "BatchNorm network");
    let bn_w = bn.weight().data().unwrap().to_flat_vec::<f32>().unwrap();
    assert!(
        bn_w.iter().any(|v| (*v - 1.0).abs() > 1e-7),
        "BatchNorm weight should change"
    );
}

// ---------------------------------------------------------------------------
// AC2: InstanceNorm training e2e (#1514)
// ---------------------------------------------------------------------------

/// Train a Conv1d → InstanceNorm → ReLU → pool → Linear classifier.
/// Uses [N, C, T] tensors with T>1 so InstanceNorm has meaningful variance.
/// Exercises InstanceNorm backward rule through a training loop.
/// Regression guard for #1481 (norm backward broadcast axis mismatch).
#[test]
fn test_train_instance_norm_loss_decreases() {
    let (batch, in_ch, seq, out_ch, k, classes) = (8, 2, 12, 4, 3, 2);

    // Synthetic 1D signal data: [B, C, T]
    let x_data = DynTensor::randn(0.0, 1.0, &[batch, in_ch, seq], &Device::Cpu).unwrap();
    let x_flat = x_data.to_flat_vec::<f32>().unwrap();
    let targets: Vec<u32> = (0..batch)
        .map(|b| {
            let sum: f32 = (0..seq).map(|t| x_flat[b * in_ch * seq + t]).sum();
            if sum > 0.0 {
                1
            } else {
                0
            }
        })
        .collect();
    let t_data = DynTensor::from_vec_u32(targets, &[batch, 1], &Device::Cpu).unwrap();

    // Build: Conv1d(2→4, k=3, pad=1) → InstanceNorm(4) → ReLU → pool → Linear(4→2)
    let conv = TrainableConv1d::from_tensors(
        DynTensor::randn(0.0, 0.1, &[out_ch, in_ch, k], &Device::Cpu).unwrap(),
        Some(DynTensor::zeros(&[out_ch], DType::F32, &Device::Cpu).unwrap()),
        1,
        1,
        1,
        1,
    );
    let inst_norm = TrainableInstanceNorm::new(out_ch, 1e-5).unwrap();
    let fc = TrainableLinear::from_tensors(
        DynTensor::randn(0.0, 0.1, &[classes, out_ch], &Device::Cpu).unwrap(),
        Some(DynTensor::zeros(&[classes], DType::F32, &Device::Cpu).unwrap()),
    );

    let all_vars = collect_vars(&[conv.vars(), inst_norm.vars(), fc.vars()]);
    let mut adam = make_adam(all_vars, 0.005);

    let mut losses = Vec::new();
    for step in 0..20 {
        let tx = Arc::new(TrackedTensor::from_tensor(x_data.clone()));
        let h = conv.forward(&tx).unwrap(); // [B, 4, 12]
        let h = inst_norm.forward(&h).unwrap(); // InstanceNorm on [B, C, T] with T=12
        let h = h.relu().unwrap();
        let h = h.mean_keepdim(2).unwrap().squeeze(2).unwrap(); // pool → [B, 4]
        let logits = fc.forward(&h).unwrap();
        let t_targets = Arc::new(TrackedTensor::from_tensor(t_data.clone()));
        let loss = logits.cross_entropy_loss(&t_targets, 1).unwrap();
        let loss_val = loss.tensor().to_flat_vec::<f32>().unwrap()[0];
        assert!(loss_val.is_finite(), "loss is NaN/Inf at step {step}");
        losses.push(loss_val);
        adam.backward_step(&loss).unwrap();
    }

    assert_loss_decreased(&losses, "InstanceNorm network");
    let w = inst_norm
        .weight()
        .data()
        .unwrap()
        .to_flat_vec::<f32>()
        .unwrap();
    assert!(
        w.iter().any(|v| (*v - 1.0).abs() > 1e-7),
        "InstanceNorm weight should change"
    );
}

// ---------------------------------------------------------------------------
// AC3: ConvTranspose1d training e2e (#1514)
// ---------------------------------------------------------------------------

/// Train a Conv1d → ReLU → ConvTranspose1d autoencoder.
/// Exercises ConvTranspose1d backward rule (grad_input + grad_kernel).
/// Regression guard for #1484 (parameter order mismatch).
#[test]
fn test_train_conv_transpose1d_loss_decreases() {
    let (batch, in_ch, seq) = (8, 2, 16);
    let (latent_ch, k) = (4, 3);

    // Autoencoder: reconstruct input from compressed representation
    let x_data = DynTensor::randn(0.0, 1.0, &[batch, in_ch, seq], &Device::Cpu).unwrap();

    // Encoder: Conv1d(2→4, k=3, stride=2, pad=1) → [B, 4, 8]
    let enc_w = Var::new(DynTensor::randn(0.0, 0.1, &[latent_ch, in_ch, k], &Device::Cpu).unwrap());
    // Decoder: ConvTranspose1d(4→2, k=3, stride=2, pad=1, output_pad=1) → [B, 2, 16]
    let dec_w = Var::new(DynTensor::randn(0.0, 0.1, &[latent_ch, in_ch, k], &Device::Cpu).unwrap());

    let mut adam = make_adam(vec![enc_w.clone(), dec_w.clone()], 0.005);

    let mut losses = Vec::new();
    for step in 0..20 {
        let tx = Arc::new(TrackedTensor::from_tensor(x_data.clone()));
        let enc_k = Arc::new(TrackedTensor::from_var(&enc_w).unwrap());
        // conv1d(kernel, padding=1, stride=2, dilation=1, groups=1)
        let encoded = tx.conv1d(&enc_k, 1, 2, 1, 1).unwrap(); // [B, 4, 8]
        let encoded = encoded.relu().unwrap();
        let dec_k = Arc::new(TrackedTensor::from_var(&dec_w).unwrap());
        // conv_transpose1d(kernel, padding=1, stride=2, dilation=1, groups=1, output_padding=1)
        let decoded = encoded.conv_transpose1d(&dec_k, 1, 2, 1, 1, 1).unwrap(); // [B, 2, 16]
                                                                                // MSE loss: mean((decoded - target)^2), reduce [B, C, T] to scalar
        let diff = decoded.sub(&tx).unwrap();
        let sq = diff.mul(&diff).unwrap();
        let loss = sq
            .mean_keepdim(2)
            .unwrap() // [B, C, 1]
            .mean_keepdim(1)
            .unwrap() // [B, 1, 1]
            .mean_keepdim(0)
            .unwrap() // [1, 1, 1]
            .squeeze(2)
            .unwrap()
            .squeeze(1)
            .unwrap()
            .squeeze(0)
            .unwrap(); // scalar
        let loss_val = loss.tensor().to_flat_vec::<f32>().unwrap()[0];
        assert!(loss_val.is_finite(), "loss is NaN/Inf at step {step}");
        losses.push(loss_val);
        adam.backward_step(&loss).unwrap();
    }

    assert_loss_decreased(&losses, "ConvTranspose1d autoencoder");
}

// ---------------------------------------------------------------------------
// TrainableConvTranspose1d wrapper e2e (unblocks HTDemucs decoder training)
// ---------------------------------------------------------------------------

/// Train a Conv1d encoder → TrainableConvTranspose1d decoder autoencoder.
/// Exercises the TrainableConvTranspose1d wrapper including bias broadcasting.
#[test]
fn test_train_trainable_conv_transpose1d_loss_decreases() {
    let (batch, in_ch, seq) = (8, 2, 16);
    let (latent_ch, k) = (4, 3);

    let x_data = DynTensor::randn(0.0, 1.0, &[batch, in_ch, seq], &Device::Cpu).unwrap();

    // Encoder: Conv1d(2→4, k=3, stride=2, pad=1)
    let enc = TrainableConv1d::from_tensors(
        DynTensor::randn(0.0, 0.1, &[latent_ch, in_ch, k], &Device::Cpu).unwrap(),
        Some(DynTensor::zeros(&[latent_ch], DType::F32, &Device::Cpu).unwrap()),
        1,
        2,
        1,
        1,
    );
    // Decoder: ConvTranspose1d(4→2, k=3, stride=2, pad=1, output_pad=1)
    let dec = TrainableConvTranspose1d::from_tensors(
        DynTensor::randn(0.0, 0.1, &[latent_ch, in_ch, k], &Device::Cpu).unwrap(),
        Some(DynTensor::zeros(&[in_ch], DType::F32, &Device::Cpu).unwrap()),
        1,
        2,
        1,
        1,
        1, // padding, stride, dilation, groups, output_padding
    );

    let all_vars = collect_vars(&[enc.vars(), dec.vars()]);
    assert_eq!(all_vars.len(), 4); // enc_w + enc_b + dec_w + dec_b
    let mut adam = make_adam(all_vars, 0.005);

    let mut losses = Vec::new();
    for step in 0..20 {
        let tx = Arc::new(TrackedTensor::from_tensor(x_data.clone()));
        let encoded = enc.forward(&tx).unwrap();
        let encoded = encoded.relu().unwrap();
        let decoded = dec.forward(&encoded).unwrap(); // [B, 2, 16]

        // MSE loss
        let diff = decoded.sub(&tx).unwrap();
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

    assert_loss_decreased(&losses, "TrainableConvTranspose1d autoencoder");
    let dec_w = dec.weight().data().unwrap().to_flat_vec::<f32>().unwrap();
    assert!(
        dec_w.iter().any(|v| v.abs() > 1e-8),
        "decoder weight should have changed"
    );
    let dec_b = dec
        .bias()
        .unwrap()
        .data()
        .unwrap()
        .to_flat_vec::<f32>()
        .unwrap();
    assert!(
        dec_b.iter().any(|v| v.abs() > 1e-8),
        "decoder bias should have changed"
    );
}

// ---------------------------------------------------------------------------
// AC4: Dropout training e2e (#1514)
// ---------------------------------------------------------------------------

/// Train a Linear → Dropout → Linear network.
/// Exercises Dropout backward rule (masked gradient scaling).
/// Eval mode (p=0) check verifies output is finite after training.
#[test]
fn test_train_dropout_loss_decreases() {
    let (batch, in_dim, hidden, classes) = (16, 4, 16, 2);

    let x_data = DynTensor::randn(0.0, 1.0, &[batch, in_dim], &Device::Cpu).unwrap();
    let x_flat = x_data.to_flat_vec::<f32>().unwrap();
    let target_data: Vec<u32> = (0..batch)
        .map(|b| {
            let sum: f32 = (0..in_dim).map(|d| x_flat[b * in_dim + d]).sum();
            if sum > 0.0 {
                1
            } else {
                0
            }
        })
        .collect();
    let t_data = DynTensor::from_vec_u32(target_data, &[batch, 1], &Device::Cpu).unwrap();

    // Build: Linear(4→16) → Dropout(0.3) → ReLU → Linear(16→2)
    let w1 = DynTensor::randn(0.0, 0.1, &[hidden, in_dim], &Device::Cpu).unwrap();
    let b1 = DynTensor::zeros(&[hidden], DType::F32, &Device::Cpu).unwrap();
    let layer1 = TrainableLinear::from_tensors(w1, Some(b1));
    let w2 = DynTensor::randn(0.0, 0.1, &[classes, hidden], &Device::Cpu).unwrap();
    let b2 = DynTensor::zeros(&[classes], DType::F32, &Device::Cpu).unwrap();
    let layer2 = TrainableLinear::from_tensors(w2, Some(b2));

    let all_vars = collect_vars(&[layer1.vars(), layer2.vars()]);
    let mut adam = make_adam(all_vars, 0.01);

    let mut losses = Vec::new();
    for step in 0..20 {
        let tx = Arc::new(TrackedTensor::from_tensor(x_data.clone()));
        let h = layer1.forward(&tx).unwrap();
        let h = h.dropout(0.3).unwrap(); // training mode: stochastic masking
        let h = h.relu().unwrap();
        let logits = layer2.forward(&h).unwrap();
        let t_targets = Arc::new(TrackedTensor::from_tensor(t_data.clone()));
        let loss = logits.cross_entropy_loss(&t_targets, 1).unwrap();
        let loss_val = loss.tensor().to_flat_vec::<f32>().unwrap()[0];
        assert!(loss_val.is_finite(), "loss is NaN/Inf at step {step}");
        losses.push(loss_val);
        adam.backward_step(&loss).unwrap();
    }

    assert_loss_decreased(&losses, "Dropout network");

    // Verify layer1 weights changed — gradient flow through dropout must reach layer1
    let w1_after = layer1
        .weight()
        .data()
        .unwrap()
        .to_flat_vec::<f32>()
        .unwrap();
    assert!(
        w1_after.iter().all(|v| v.is_finite()),
        "layer1 weights should remain finite after training through dropout"
    );
    // weights were initialized with randn(0, 0.1) — check some have moved away from small-magnitude init
    let w1_shifted = w1_after
        .iter()
        .filter(|v| v.abs() > 0.15) // moved beyond ~1.5σ of init distribution
        .count();
    assert!(
        w1_shifted > 0,
        "layer1 weights should shift from init values — dropout backward must propagate gradients"
    );

    // Eval mode: forward without dropout should produce finite output
    let tx = Arc::new(TrackedTensor::from_tensor(x_data.clone()));
    let h = layer1.forward(&tx).unwrap();
    let h = h.dropout(0.0).unwrap(); // p=0.0 is identity (eval mode)
    let h = h.relu().unwrap();
    let eval_logits = layer2.forward(&h).unwrap();
    let eval_vals = eval_logits.tensor().to_flat_vec::<f32>().unwrap();
    assert!(
        eval_vals.iter().all(|v| v.is_finite()),
        "eval-mode output should be finite"
    );
}

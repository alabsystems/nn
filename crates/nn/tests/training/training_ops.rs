// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! E2E training tests for Conv1d, Conv2d, Embedding, RmsNorm, GroupNorm.
//!
//! Pool backward tests (MaxPool2d, AvgPool2d, AdaptiveAvgPool2d) are in
//! `training_ops_pool.rs`.
//! BatchNorm, InstanceNorm, ConvTranspose1d, and Dropout tests are in
//! `training_ops_extra.rs`.
//!
//! Run: `cargo test -p nn --test training_ops --features training`

#![cfg(feature = "training")]

use std::sync::Arc;

use nn::training::{
    Optimizer, TrackedTensor, TrainableConv1d, TrainableConv2d, TrainableEmbedding,
    TrainableGroupNorm, TrainableLinear, TrainableModule, TrainableRmsNorm,
};
use nn::{DType, Device, DynTensor};

use super::common::{assert_loss_decreased, collect_vars, make_adam, make_binary_targets};

// ---------------------------------------------------------------------------
// Conv1d training e2e
// ---------------------------------------------------------------------------

/// Train a Conv1d → ReLU → Linear classifier on synthetic 1D sequences.
/// Exercises Conv1d backward rule (grad_input + grad_weight + grad_bias).
#[test]
fn test_train_conv1d_loss_decreases() {
    let (batch, in_ch, seq, out_ch, k, classes) = (8, 3, 16, 4, 3, 2);

    // Synthetic data: random sequences, binary labels
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

    // Build: Conv1d(3→4, k=3, pad=1) → ReLU → mean_pool → Linear(4→2)
    let conv_w = DynTensor::randn(0.0, 0.1, &[out_ch, in_ch, k], &Device::Cpu).unwrap();
    let conv_b = DynTensor::zeros(&[out_ch], DType::F32, &Device::Cpu).unwrap();
    let conv = TrainableConv1d::from_tensors(conv_w, Some(conv_b), 1, 1, 1, 1);
    let fc_w = DynTensor::randn(0.0, 0.1, &[classes, out_ch], &Device::Cpu).unwrap();
    let fc_b = DynTensor::zeros(&[classes], DType::F32, &Device::Cpu).unwrap();
    let fc = TrainableLinear::from_tensors(fc_w, Some(fc_b));

    let mut adam = make_adam(collect_vars(&[conv.vars(), fc.vars()]), 0.005);

    let mut losses = Vec::new();
    for step in 0..20 {
        let tx = Arc::new(TrackedTensor::from_tensor(x_data.clone()));
        let h = conv.forward(&tx).unwrap();
        let h = h.relu().unwrap();
        let h = h.mean_keepdim(2).unwrap().squeeze(2).unwrap(); // pool over time
        let logits = fc.forward(&h).unwrap();
        let t_targets = Arc::new(TrackedTensor::from_tensor(t_data.clone()));
        let loss = logits.cross_entropy_loss(&t_targets, 1).unwrap();
        let loss_val = loss.tensor().to_flat_vec::<f32>().unwrap()[0];
        assert!(loss_val.is_finite(), "loss is NaN/Inf at step {step}");
        losses.push(loss_val);
        adam.backward_step(&loss).unwrap();
    }

    assert_loss_decreased(&losses, "Conv1d network");
    let w_after = conv.weight().data().unwrap().to_flat_vec::<f32>().unwrap();
    assert!(
        w_after.iter().any(|v| v.abs() > 1e-8),
        "conv weight should have changed"
    );
}

// ---------------------------------------------------------------------------
// Embedding training e2e
// ---------------------------------------------------------------------------

/// Train an Embedding → Linear classifier on synthetic token classification.
/// Exercises Embedding backward rule (scatter_add-based gradient accumulation).
#[test]
fn test_train_embedding_loss_decreases() {
    let (vocab, dim, seq, batch, classes) = (10, 8, 4, 6, 3);

    // Synthetic token IDs in [0, vocab) and targets
    let token_ids: Vec<u32> = (0..batch * seq)
        .map(|i| ((i / seq * 7 + i % seq * 3) % vocab) as u32)
        .collect();
    let tokens = DynTensor::from_vec_u32(token_ids.clone(), &[batch, seq], &Device::Cpu).unwrap();
    let target_data: Vec<u32> = (0..batch)
        .map(|b| (token_ids[b * seq] as usize % classes) as u32)
        .collect();
    let t_data = DynTensor::from_vec_u32(target_data, &[batch, 1], &Device::Cpu).unwrap();

    // Build: Embedding(10, 8) → mean_pool → Linear(8→3)
    let emb_w = DynTensor::randn(0.0, 0.1, &[vocab, dim], &Device::Cpu).unwrap();
    let emb = TrainableEmbedding::from_tensor(emb_w);
    let fc_w = DynTensor::randn(0.0, 0.1, &[classes, dim], &Device::Cpu).unwrap();
    let fc_b = DynTensor::zeros(&[classes], DType::F32, &Device::Cpu).unwrap();
    let fc = TrainableLinear::from_tensors(fc_w, Some(fc_b));

    let mut adam = make_adam(collect_vars(&[emb.vars(), fc.vars()]), 0.01);
    let emb_before = emb.weight().data().unwrap().to_flat_vec::<f32>().unwrap();

    let mut losses = Vec::new();
    for step in 0..15 {
        let t_tokens = Arc::new(TrackedTensor::from_tensor(tokens.clone()));
        let h = emb.forward(&t_tokens).unwrap(); // [B, seq, dim]
        let h = h.mean_keepdim(1).unwrap().squeeze(1).unwrap(); // pool over seq
        let logits = fc.forward(&h).unwrap();
        let t_targets = Arc::new(TrackedTensor::from_tensor(t_data.clone()));
        let loss = logits.cross_entropy_loss(&t_targets, 1).unwrap();
        let loss_val = loss.tensor().to_flat_vec::<f32>().unwrap()[0];
        assert!(loss_val.is_finite(), "loss is NaN/Inf at step {step}");
        losses.push(loss_val);
        adam.backward_step(&loss).unwrap();
    }

    assert_loss_decreased(&losses, "Embedding network");
    let emb_after = emb.weight().data().unwrap().to_flat_vec::<f32>().unwrap();
    let changed = emb_before
        .iter()
        .zip(emb_after.iter())
        .any(|(a, b)| (a - b).abs() > 1e-8);
    assert!(changed, "embedding weights should have changed");
}

// ---------------------------------------------------------------------------
// RmsNorm training e2e
// ---------------------------------------------------------------------------

/// Train a Linear → RmsNorm → Linear network.
/// Exercises RmsNorm backward rule through a training loop.
#[test]
fn test_train_rms_norm_network_loss_decreases() {
    let (batch, in_dim, hidden, classes) = (12, 4, 8, 3);

    let x_data = DynTensor::randn(0.0, 1.0, &[batch, in_dim], &Device::Cpu).unwrap();
    let x_flat = x_data.to_flat_vec::<f32>().unwrap();
    let t_data = make_binary_targets(&x_flat, batch, in_dim);

    // Build: Linear(4→8) → RmsNorm(8) → ReLU → Linear(8→3)
    let w1 = DynTensor::randn(0.0, 0.1, &[hidden, in_dim], &Device::Cpu).unwrap();
    let b1 = DynTensor::zeros(&[hidden], DType::F32, &Device::Cpu).unwrap();
    let layer1 = TrainableLinear::from_tensors(w1, Some(b1));
    let rms_norm = TrainableRmsNorm::new(hidden, 1e-5).unwrap();
    let w2 = DynTensor::randn(0.0, 0.1, &[classes, hidden], &Device::Cpu).unwrap();
    let b2 = DynTensor::zeros(&[classes], DType::F32, &Device::Cpu).unwrap();
    let layer2 = TrainableLinear::from_tensors(w2, Some(b2));

    let all_vars = collect_vars(&[layer1.vars(), rms_norm.vars(), layer2.vars()]);
    assert_eq!(all_vars.len(), 5); // w1+b1 + rms_weight + w2+b2
    let mut adam = make_adam(all_vars, 0.01);

    let mut losses = Vec::new();
    for step in 0..15 {
        let tx = Arc::new(TrackedTensor::from_tensor(x_data.clone()));
        let h = layer1.forward(&tx).unwrap();
        let h = rms_norm.forward(&h).unwrap();
        let h = h.relu().unwrap();
        let logits = layer2.forward(&h).unwrap();
        let t_targets = Arc::new(TrackedTensor::from_tensor(t_data.clone()));
        let loss = logits.cross_entropy_loss(&t_targets, 1).unwrap();
        let loss_val = loss.tensor().to_flat_vec::<f32>().unwrap()[0];
        assert!(loss_val.is_finite(), "loss is NaN/Inf at step {step}");
        losses.push(loss_val);
        adam.backward_step(&loss).unwrap();
    }

    assert_loss_decreased(&losses, "RmsNorm network");
    let rms_w = rms_norm
        .weight()
        .data()
        .unwrap()
        .to_flat_vec::<f32>()
        .unwrap();
    assert!(
        rms_w.iter().any(|v| (*v - 1.0).abs() > 1e-7),
        "RmsNorm weight should change"
    );
}

// ---------------------------------------------------------------------------
// Conv2d training e2e
// ---------------------------------------------------------------------------

/// Train a Conv2d → ReLU → AvgPool → Linear classifier on synthetic images.
/// Exercises Conv2d backward rule (grad_input + grad_kernel + grad_bias).
#[test]
fn test_train_conv2d_loss_decreases() {
    let (batch, in_ch, img_h, img_w) = (8, 1, 8, 8);
    let (out_ch, k, classes) = (4, 3, 2);

    let x_data = DynTensor::randn(0.0, 1.0, &[batch, in_ch, img_h, img_w], &Device::Cpu).unwrap();
    let x_flat = x_data.to_flat_vec::<f32>().unwrap();
    let targets: Vec<u32> = (0..batch)
        .map(|b| {
            let sum: f32 = (0..img_h * img_w)
                .map(|i| x_flat[b * in_ch * img_h * img_w + i])
                .sum();
            if sum > 0.0 {
                1
            } else {
                0
            }
        })
        .collect();
    let t_data = DynTensor::from_vec_u32(targets, &[batch, 1], &Device::Cpu).unwrap();

    let conv_w = DynTensor::randn(0.0, 0.1, &[out_ch, in_ch, k, k], &Device::Cpu).unwrap();
    let conv_b = DynTensor::zeros(&[out_ch], DType::F32, &Device::Cpu).unwrap();
    let conv = TrainableConv2d::from_tensors(conv_w, Some(conv_b), 1, 1, 1, 1);
    let fc_w = DynTensor::randn(0.0, 0.1, &[classes, out_ch], &Device::Cpu).unwrap();
    let fc_b = DynTensor::zeros(&[classes], DType::F32, &Device::Cpu).unwrap();
    let fc = TrainableLinear::from_tensors(fc_w, Some(fc_b));

    let mut adam = make_adam(collect_vars(&[conv.vars(), fc.vars()]), 0.005);

    let mut losses = Vec::new();
    for step in 0..15 {
        let tx = Arc::new(TrackedTensor::from_tensor(x_data.clone()));
        let h_out = conv.forward(&tx).unwrap();
        let h_out = h_out.relu().unwrap();
        let h_out = h_out.mean_keepdim(2).unwrap().mean_keepdim(3).unwrap();
        let h_out = h_out.squeeze(3).unwrap().squeeze(2).unwrap();
        let logits = fc.forward(&h_out).unwrap();
        let t_targets = Arc::new(TrackedTensor::from_tensor(t_data.clone()));
        let loss = logits.cross_entropy_loss(&t_targets, 1).unwrap();
        let loss_val = loss.tensor().to_flat_vec::<f32>().unwrap()[0];
        assert!(loss_val.is_finite(), "loss is NaN/Inf at step {step}");
        losses.push(loss_val);
        adam.backward_step(&loss).unwrap();
    }

    assert_loss_decreased(&losses, "Conv2d network");
    let w_after = conv.weight().data().unwrap().to_flat_vec::<f32>().unwrap();
    assert!(
        w_after.iter().any(|v| v.abs() > 1e-8),
        "conv2d weight should have changed"
    );
}

// ---------------------------------------------------------------------------
// GroupNorm training e2e
// ---------------------------------------------------------------------------

/// Train a Linear → GroupNorm → ReLU → Linear network.
/// Exercises GroupNorm backward rule through a training loop.
#[test]
fn test_train_group_norm_loss_decreases() {
    let (batch, in_dim, hidden, classes, num_groups) = (12, 4, 8, 3, 2);

    let x_data = DynTensor::randn(0.0, 1.0, &[batch, in_dim], &Device::Cpu).unwrap();
    let x_flat = x_data.to_flat_vec::<f32>().unwrap();
    let t_data = make_binary_targets(&x_flat, batch, in_dim);

    let w1 = DynTensor::randn(0.0, 0.1, &[hidden, in_dim], &Device::Cpu).unwrap();
    let b1 = DynTensor::zeros(&[hidden], DType::F32, &Device::Cpu).unwrap();
    let layer1 = TrainableLinear::from_tensors(w1, Some(b1));
    let gn = TrainableGroupNorm::new(hidden, num_groups, 1e-5).unwrap();
    let w2 = DynTensor::randn(0.0, 0.1, &[classes, hidden], &Device::Cpu).unwrap();
    let b2 = DynTensor::zeros(&[classes], DType::F32, &Device::Cpu).unwrap();
    let layer2 = TrainableLinear::from_tensors(w2, Some(b2));

    let all_vars = collect_vars(&[layer1.vars(), gn.vars(), layer2.vars()]);
    assert_eq!(all_vars.len(), 6);
    let mut adam = make_adam(all_vars, 0.01);

    let mut losses = Vec::new();
    for step in 0..15 {
        let tx = Arc::new(TrackedTensor::from_tensor(x_data.clone()));
        let h_out = layer1.forward(&tx).unwrap();
        let h_out = h_out.unsqueeze(2).unwrap(); // GroupNorm expects [N, C, *]
        let h_out = gn.forward(&h_out).unwrap();
        let h_out = h_out.squeeze(2).unwrap();
        let h_out = h_out.relu().unwrap();
        let logits = layer2.forward(&h_out).unwrap();
        let t_targets = Arc::new(TrackedTensor::from_tensor(t_data.clone()));
        let loss = logits.cross_entropy_loss(&t_targets, 1).unwrap();
        let loss_val = loss.tensor().to_flat_vec::<f32>().unwrap()[0];
        assert!(loss_val.is_finite(), "loss is NaN/Inf at step {step}");
        losses.push(loss_val);
        adam.backward_step(&loss).unwrap();
    }

    assert_loss_decreased(&losses, "GroupNorm network");
    let gn_w = gn.weight().data().unwrap().to_flat_vec::<f32>().unwrap();
    assert!(
        gn_w.iter().any(|v| (*v - 1.0).abs() > 1e-7),
        "GroupNorm weight should change"
    );
}

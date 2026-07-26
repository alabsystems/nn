// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! E2E training tests for pool backward rules: MaxPool2d, AvgPool2d,
//! AdaptiveAvgPool2d.
//!
//! Extracted from `training_ops.rs` for 500-line compliance.
//!
//! Run: `cargo test -p nn --test training_ops_pool --features training`

#![cfg(feature = "training")]

use std::sync::Arc;

use nn::training::{Optimizer, TrackedTensor, TrainableConv2d, TrainableLinear, TrainableModule};
use nn::{DType, Device, DynTensor};

use super::common::{assert_loss_decreased, collect_vars, make_adam, make_binary_targets};

// ---------------------------------------------------------------------------
// MaxPool2d training e2e
// ---------------------------------------------------------------------------

/// Train a Conv2d → MaxPool2d → Linear classifier on synthetic 2D data.
/// Exercises MaxPool2d backward rule (gradient routes to argmax position).
#[test]
fn test_train_maxpool2d_loss_decreases() {
    let (batch, in_ch, h, w, out_ch, classes) = (8, 1, 8, 8, 4, 2);

    let x_data = DynTensor::randn(0.0, 1.0, &[batch, in_ch, h, w], &Device::Cpu).unwrap();
    let x_flat = x_data.to_flat_vec::<f32>().unwrap();
    let t_data = make_binary_targets(&x_flat, batch, in_ch * h * w);

    // Conv2d(1→4, k=3, pad=1) → MaxPool2d(k=2, s=2) → flatten → Linear(4*4*4→2)
    let conv_w = DynTensor::randn(0.0, 0.1, &[out_ch, in_ch, 3, 3], &Device::Cpu).unwrap();
    let conv_b = DynTensor::zeros(&[out_ch], DType::F32, &Device::Cpu).unwrap();
    let conv = TrainableConv2d::from_tensors(conv_w, Some(conv_b), 1, 1, 1, 1);
    let pooled_dim = out_ch * 4 * 4; // 8/2=4 spatial after pool
    let fc_w = DynTensor::randn(0.0, 0.1, &[classes, pooled_dim], &Device::Cpu).unwrap();
    let fc_b = DynTensor::zeros(&[classes], DType::F32, &Device::Cpu).unwrap();
    let fc = TrainableLinear::from_tensors(fc_w, Some(fc_b));

    let all_vars = collect_vars(&[conv.vars(), fc.vars()]);
    let mut adam = make_adam(all_vars, 0.01);

    let mut losses = Vec::new();
    for step in 0..15 {
        let tx = Arc::new(TrackedTensor::from_tensor(x_data.clone()));
        let h_out = conv.forward(&tx).unwrap();
        let h_out = h_out.relu().unwrap();
        let h_out = h_out.max_pool2d(2, 2, 0).unwrap();
        // Flatten: [batch, out_ch, 4, 4] → [batch, out_ch*4*4]
        let h_out = h_out.reshape(&[batch, pooled_dim]).unwrap();
        let logits = fc.forward(&h_out).unwrap();
        let t_targets = Arc::new(TrackedTensor::from_tensor(t_data.clone()));
        let loss = logits.cross_entropy_loss(&t_targets, 1).unwrap();
        let loss_val = loss.tensor().to_flat_vec::<f32>().unwrap()[0];
        assert!(
            loss_val.is_finite(),
            "MaxPool loss is NaN/Inf at step {step}"
        );
        losses.push(loss_val);
        adam.backward_step(&loss).unwrap();
    }

    assert_loss_decreased(&losses, "MaxPool2d training");
}

// ---------------------------------------------------------------------------
// AvgPool2d training e2e
// ---------------------------------------------------------------------------

/// Train a Conv2d → AvgPool2d → Linear classifier on synthetic 2D data.
/// Exercises AvgPool2d backward rule (gradient distributed uniformly over window).
#[test]
fn test_train_avgpool2d_loss_decreases() {
    let (batch, in_ch, h, w, out_ch, classes) = (8, 1, 8, 8, 4, 2);

    let x_data = DynTensor::randn(0.0, 1.0, &[batch, in_ch, h, w], &Device::Cpu).unwrap();
    let x_flat = x_data.to_flat_vec::<f32>().unwrap();
    let t_data = make_binary_targets(&x_flat, batch, in_ch * h * w);

    let conv_w = DynTensor::randn(0.0, 0.1, &[out_ch, in_ch, 3, 3], &Device::Cpu).unwrap();
    let conv_b = DynTensor::zeros(&[out_ch], DType::F32, &Device::Cpu).unwrap();
    let conv = TrainableConv2d::from_tensors(conv_w, Some(conv_b), 1, 1, 1, 1);
    let pooled_dim = out_ch * 4 * 4;
    let fc_w = DynTensor::randn(0.0, 0.1, &[classes, pooled_dim], &Device::Cpu).unwrap();
    let fc_b = DynTensor::zeros(&[classes], DType::F32, &Device::Cpu).unwrap();
    let fc = TrainableLinear::from_tensors(fc_w, Some(fc_b));

    let all_vars = collect_vars(&[conv.vars(), fc.vars()]);
    let mut adam = make_adam(all_vars, 0.01);

    let mut losses = Vec::new();
    for step in 0..15 {
        let tx = Arc::new(TrackedTensor::from_tensor(x_data.clone()));
        let h_out = conv.forward(&tx).unwrap();
        let h_out = h_out.relu().unwrap();
        let h_out = h_out.avg_pool2d(2, 2, 0).unwrap();
        let h_out = h_out.reshape(&[batch, pooled_dim]).unwrap();
        let logits = fc.forward(&h_out).unwrap();
        let t_targets = Arc::new(TrackedTensor::from_tensor(t_data.clone()));
        let loss = logits.cross_entropy_loss(&t_targets, 1).unwrap();
        let loss_val = loss.tensor().to_flat_vec::<f32>().unwrap()[0];
        assert!(
            loss_val.is_finite(),
            "AvgPool loss is NaN/Inf at step {step}"
        );
        losses.push(loss_val);
        adam.backward_step(&loss).unwrap();
    }

    assert_loss_decreased(&losses, "AvgPool2d training");
}

// ---------------------------------------------------------------------------
// AdaptiveAvgPool2d training e2e
// ---------------------------------------------------------------------------

/// Train with AdaptiveAvgPool2d → Linear classifier.
/// Exercises AdaptiveAvgPool2d backward rule (variable window sizes).
#[test]
fn test_train_adaptive_avgpool2d_loss_decreases() {
    let (batch, in_ch, h, w, out_ch, classes) = (8, 1, 8, 8, 4, 2);

    let x_data = DynTensor::randn(0.0, 1.0, &[batch, in_ch, h, w], &Device::Cpu).unwrap();
    let x_flat = x_data.to_flat_vec::<f32>().unwrap();
    let t_data = make_binary_targets(&x_flat, batch, in_ch * h * w);

    let conv_w = DynTensor::randn(0.0, 0.1, &[out_ch, in_ch, 3, 3], &Device::Cpu).unwrap();
    let conv_b = DynTensor::zeros(&[out_ch], DType::F32, &Device::Cpu).unwrap();
    let conv = TrainableConv2d::from_tensors(conv_w, Some(conv_b), 1, 1, 1, 1);
    // AdaptiveAvgPool2d to 2×2 → flatten to out_ch*2*2
    let pooled_dim = out_ch * 2 * 2;
    let fc_w = DynTensor::randn(0.0, 0.1, &[classes, pooled_dim], &Device::Cpu).unwrap();
    let fc_b = DynTensor::zeros(&[classes], DType::F32, &Device::Cpu).unwrap();
    let fc = TrainableLinear::from_tensors(fc_w, Some(fc_b));

    let all_vars = collect_vars(&[conv.vars(), fc.vars()]);
    let mut adam = make_adam(all_vars, 0.01);

    let mut losses = Vec::new();
    for step in 0..15 {
        let tx = Arc::new(TrackedTensor::from_tensor(x_data.clone()));
        let h_out = conv.forward(&tx).unwrap();
        let h_out = h_out.relu().unwrap();
        let h_out = h_out.adaptive_avg_pool2d(2, 2).unwrap();
        let h_out = h_out.reshape(&[batch, pooled_dim]).unwrap();
        let logits = fc.forward(&h_out).unwrap();
        let t_targets = Arc::new(TrackedTensor::from_tensor(t_data.clone()));
        let loss = logits.cross_entropy_loss(&t_targets, 1).unwrap();
        let loss_val = loss.tensor().to_flat_vec::<f32>().unwrap()[0];
        assert!(
            loss_val.is_finite(),
            "AdaptiveAvgPool loss NaN/Inf at step {step}"
        );
        losses.push(loss_val);
        adam.backward_step(&loss).unwrap();
    }

    assert_loss_decreased(&losses, "AdaptiveAvgPool2d training");
}

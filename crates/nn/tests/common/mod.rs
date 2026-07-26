// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Shared test helpers for training integration tests.
//!
//! Used by `training_pipeline.rs`, `training_pipeline_checkpoint.rs`,
//! `training_e2e.rs`, `training_ops.rs`, `training_ops_extra.rs`, and
//! `training_ops_pool.rs` to avoid duplicating MLP forward logic,
//! weight initialization, optimizer construction, and training step helpers.

#![allow(dead_code, unreachable_pub)]

use std::sync::Arc;

use nn::training::{AdamConfig, AdamW, TrackedTensor, Var};
use nn::{DType, Device, DynTensor};

/// AdamConfig with lr=0.01 and no weight decay.
pub fn adam_config_no_wd() -> AdamConfig {
    let mut c = AdamConfig::default();
    c.lr = 0.01;
    c.weight_decay = 0.0;
    c
}

/// Create a 2-layer MLP forward pass through TrackedTensors.
/// logits = relu(x @ w1^T + b1) @ w2^T + b2
pub fn forward_mlp(
    x: &Arc<TrackedTensor>,
    w1: &Arc<TrackedTensor>,
    b1: &Arc<TrackedTensor>,
    w2: &Arc<TrackedTensor>,
    b2: &Arc<TrackedTensor>,
) -> Arc<TrackedTensor> {
    let w1t = w1.transpose(0, 1).unwrap();
    let h = x.matmul(&w1t).unwrap();
    let h = h.add(b1).unwrap();
    let h = h.relu().unwrap();
    let w2t = w2.transpose(0, 1).unwrap();
    let logits = h.matmul(&w2t).unwrap();
    logits.add(b2).unwrap()
}

/// Deterministic weight initialization.
pub fn make_weight(rows: usize, cols: usize, seed: usize) -> DynTensor {
    DynTensor::from_vec(
        (0..rows * cols)
            .map(|i| ((i * seed + 3) % 100) as f32 * 0.02 - 1.0)
            .collect(),
        &[rows, cols],
        &Device::Cpu,
    )
    .unwrap()
}

/// Synthetic classification data.
/// Returns (inputs [N, D], target_classes [N, 1]).
pub fn make_data(n: usize, dim: usize, num_classes: usize) -> (DynTensor, DynTensor) {
    let mut data = Vec::with_capacity(n * dim);
    let mut targets = Vec::with_capacity(n);
    for i in 0..n {
        let class = (i % num_classes) as u32;
        targets.push(class);
        for d in 0..dim {
            let centroid = (class as f32 + 1.0) * (d as f32 + 1.0) * 0.5;
            let noise = ((i * 7 + d * 13) % 100) as f32 * 0.01 - 0.5;
            data.push(centroid + noise);
        }
    }
    let x = DynTensor::from_vec(data, &[n, dim], &Device::Cpu).unwrap();
    let t = DynTensor::from_vec_u32(targets, &[n, 1], &Device::Cpu).unwrap();
    (x, t)
}

/// Run one training step: forward MLP → cross-entropy loss → return (loss, loss_val).
pub fn train_step(
    x_data: &DynTensor,
    t_data: &DynTensor,
    w1: &Var,
    b1: &Var,
    w2: &Var,
    b2: &Var,
) -> (Arc<TrackedTensor>, f32) {
    let tw1 = Arc::new(TrackedTensor::from_var(w1).unwrap());
    let tb1 = Arc::new(TrackedTensor::from_var(b1).unwrap());
    let tw2 = Arc::new(TrackedTensor::from_var(w2).unwrap());
    let tb2 = Arc::new(TrackedTensor::from_var(b2).unwrap());
    let tx = Arc::new(TrackedTensor::from_tensor(x_data.clone()));
    let logits = forward_mlp(&tx, &tw1, &tb1, &tw2, &tb2);
    let t_targets = Arc::new(TrackedTensor::from_tensor(t_data.clone()));
    let loss = logits.cross_entropy_loss(&t_targets, 1).unwrap();
    #[allow(deprecated)]
    let loss_val = loss.tensor().to_flat_vec::<f32>().unwrap()[0];
    assert!(loss_val.is_finite(), "loss is NaN/Inf");
    (loss, loss_val)
}

/// Create Vars for a 2-layer MLP (w1, b1, w2, b2).
pub fn make_vars(in_dim: usize, hidden: usize, num_classes: usize) -> (Var, Var, Var, Var) {
    let w1 = Var::new(make_weight(hidden, in_dim, 17));
    let b1 = Var::new(DynTensor::zeros(&[1, hidden], DType::F32, &Device::Cpu).unwrap());
    let w2 = Var::new(make_weight(num_classes, hidden, 23));
    let b2 = Var::new(DynTensor::zeros(&[1, num_classes], DType::F32, &Device::Cpu).unwrap());
    (w1, b1, w2, b2)
}

/// Create an AdamW optimizer with a given learning rate and zero weight decay.
pub fn make_adam(vars: Vec<Var>, lr: f64) -> AdamW {
    let mut c = AdamConfig::default();
    c.lr = lr;
    c.weight_decay = 0.0;
    AdamW::new(vars, c).unwrap()
}

/// Collect vars from multiple trainable modules.
pub fn collect_vars(var_slices: &[Vec<&Var>]) -> Vec<Var> {
    var_slices
        .iter()
        .flat_map(|s| s.iter().map(|v| (*v).clone()))
        .collect()
}

/// Assert loss decreased over training and all losses were finite.
pub fn assert_loss_decreased(losses: &[f32], label: &str) {
    let initial = losses[0];
    let final_loss = *losses.last().unwrap();
    assert!(
        final_loss < initial,
        "{label}: loss should decrease: initial={initial}, final={final_loss}",
    );
}

/// Generate synthetic binary classification targets from input data.
pub fn make_binary_targets(x_flat: &[f32], batch: usize, in_dim: usize) -> DynTensor {
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
    DynTensor::from_vec_u32(target_data, &[batch, 1], &Device::Cpu).unwrap()
}

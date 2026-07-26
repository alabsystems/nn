// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Training pipeline integration tests.
//!
//! Covers critical integration gaps not tested by training_e2e.rs:
//! - LoRA fine-tuning with optimizer
//! - GradScaler + real Optimizer.step()
//! - Gradient clipping in backward→clip→step pipeline
//! - SGD and AdaFactor on multi-layer models
//! - LR schedule integration
//!
//! Checkpoint/resume and TrainableModule tests are in `training_pipeline_checkpoint.rs`.
//!
//! Run: `cargo test -p nn --test training_pipeline --features training`

#![cfg(feature = "training")]

use std::sync::Arc;

use nn::training::{
    backward, clip_grad_norm, step_with_schedule, AdaFactor, AdaFactorConfig, AdamW,
    CosineSchedule, GradScaler, GradScalerConfig, LoraLinear, Optimizer, Sgd, SgdConfig,
    TrackedTensor,
};
use nn::{Device, DynTensor, Linear};

use super::common::{adam_config_no_wd, make_data, make_vars, make_weight, train_step};

// ── LoRA fine-tuning tests ───────────────────────────────────────────

/// LoRA fine-tuning: train only A and B matrices, loss decreases.
#[test]
fn test_lora_finetune_loss_decreases() {
    let (in_dim, out_dim) = (8, 4);
    let rank = 4;
    let alpha = 4.0;
    let batch = 6;

    // Create a frozen linear layer.
    let frozen_weight = make_weight(out_dim, in_dim, 31);
    let linear = Linear::new(frozen_weight, None).unwrap();
    let lora = LoraLinear::from_linear(&linear, rank, alpha).unwrap();
    let vars: Vec<_> = lora.trainable_vars().into_iter().cloned().collect();

    let mut adam = AdamW::new(vars, adam_config_no_wd()).unwrap();

    // Synthetic regression: minimize ||lora(x) - target||^2
    let x_data = make_weight(batch, in_dim, 7);
    let target = make_weight(batch, out_dim, 41);

    let mut losses = Vec::new();
    for _ in 0..15 {
        // Build tracked computation graph manually (LoRA forward is DynTensor-based).
        let tx = Arc::new(TrackedTensor::from_tensor(x_data.clone()));
        let t_frozen_w = Arc::new(TrackedTensor::from_tensor(lora.frozen_weight().clone()));
        let t_a = Arc::new(TrackedTensor::from_var(lora.lora_a()).unwrap());
        let t_b = Arc::new(TrackedTensor::from_var(lora.lora_b()).unwrap());

        // y = x @ W^T + (x @ A^T) @ B^T * scaling
        let wt = t_frozen_w.transpose(0, 1).unwrap();
        let base = tx.matmul(&wt).unwrap();
        let at = t_a.transpose(0, 1).unwrap();
        let bt = t_b.transpose(0, 1).unwrap();
        let lora_out = tx.matmul(&at).unwrap().matmul(&bt).unwrap();
        let scaling_t = Arc::new(TrackedTensor::from_tensor(
            DynTensor::new(&[lora.scaling() as f32], &[1, 1], &Device::Cpu).unwrap(),
        ));
        let scaled = lora_out.mul(&scaling_t).unwrap();
        let y = base.add(&scaled).unwrap();

        // MSE loss: mean((y - target)^2)
        let t_target = Arc::new(TrackedTensor::from_tensor(target.clone()));
        let diff = y.sub(&t_target).unwrap();
        let sq = diff.sqr().unwrap();
        let loss = sq.mean_keepdim(1).unwrap().mean_keepdim(0).unwrap();

        let loss_val = loss.tensor().to_flat_vec::<f32>().unwrap()[0];
        assert!(loss_val.is_finite(), "LoRA loss is NaN/Inf");
        losses.push(loss_val);
        adam.backward_step(&loss).unwrap();
    }

    let initial = losses[0];
    let final_loss = *losses.last().unwrap();
    assert!(
        final_loss < initial,
        "LoRA loss should decrease: initial={initial}, final={final_loss}",
    );
}

// ── GradScaler + real Optimizer tests ────────────────────────────────

/// GradScaler + AdamW.step() — the documented usage pattern.
#[test]
fn test_grad_scaler_with_adam_optimizer() {
    let (in_dim, hidden, num_classes) = (4, 8, 3);
    let (x_data, t_data) = make_data(12, in_dim, num_classes);
    let (w1, b1, w2, b2) = make_vars(in_dim, hidden, num_classes);

    let mut scaler = GradScaler::new(GradScalerConfig::default()).unwrap();
    let mut adam = AdamW::new(
        vec![w1.clone(), b1.clone(), w2.clone(), b2.clone()],
        adam_config_no_wd(),
    )
    .unwrap();

    let mut losses = Vec::new();
    for _ in 0..10 {
        let (loss, loss_val) = train_step(&x_data, &t_data, &w1, &b1, &w2, &b2);
        losses.push(loss_val);

        // The documented GradScaler pattern:
        let scaled_loss = scaler.scale_loss(&loss).unwrap();
        let mut grads = backward(&scaled_loss).unwrap();
        if scaler.unscale_and_check(&mut grads).unwrap() {
            adam.step(&grads).unwrap();
        }
        scaler.update();
    }

    let initial = losses[0];
    let final_loss = *losses.last().unwrap();
    assert!(
        final_loss < initial,
        "GradScaler+Adam: loss should decrease: initial={initial}, final={final_loss}",
    );
}

// ── Gradient clipping in pipeline tests ──────────────────────────────

/// backward → clip_grad_norm → optimizer.step() pipeline.
#[test]
fn test_grad_clip_in_training_pipeline() {
    let (in_dim, hidden, num_classes) = (4, 8, 3);
    let (x_data, t_data) = make_data(12, in_dim, num_classes);
    let (w1, b1, w2, b2) = make_vars(in_dim, hidden, num_classes);

    let mut adam = AdamW::new(
        vec![w1.clone(), b1.clone(), w2.clone(), b2.clone()],
        adam_config_no_wd(),
    )
    .unwrap();

    let max_norm = 1.0;
    let mut losses = Vec::new();
    let mut norms = Vec::new();

    for _ in 0..10 {
        let (loss, loss_val) = train_step(&x_data, &t_data, &w1, &b1, &w2, &b2);
        losses.push(loss_val);

        let mut grads = backward(&loss).unwrap();
        let original_norm = clip_grad_norm(&mut grads, max_norm).unwrap();
        norms.push(original_norm);
        adam.step(&grads).unwrap();
    }

    // Loss should decrease.
    let initial = losses[0];
    let final_loss = *losses.last().unwrap();
    assert!(
        final_loss < initial,
        "Clipped training: loss should decrease: initial={initial}, final={final_loss}",
    );

    // Some norms should have been above max_norm (clipping happened).
    let any_clipped = norms.iter().any(|n| *n > max_norm + 1e-6);
    assert!(
        any_clipped,
        "expected some gradient norms to exceed max_norm={max_norm}, got {:?}",
        norms
    );
}

// ── SGD on multi-layer model ─────────────────────────────────────────

/// SGD with momentum trains a 2-layer MLP — loss decreases.
#[test]
fn test_sgd_mlp_training_loss_decreases() {
    let (in_dim, hidden, num_classes) = (4, 8, 3);
    let (x_data, t_data) = make_data(12, in_dim, num_classes);
    let (w1, b1, w2, b2) = make_vars(in_dim, hidden, num_classes);

    let mut sgd_config = SgdConfig::default();
    sgd_config.lr = 0.05;
    sgd_config.momentum = 0.9;
    sgd_config.weight_decay = 1e-4;
    let mut sgd = Sgd::new(
        vec![w1.clone(), b1.clone(), w2.clone(), b2.clone()],
        sgd_config,
    )
    .unwrap();

    let mut losses = Vec::new();
    for _ in 0..20 {
        let (loss, loss_val) = train_step(&x_data, &t_data, &w1, &b1, &w2, &b2);
        losses.push(loss_val);
        sgd.backward_step(&loss).unwrap();
    }

    let initial = losses[0];
    let final_loss = *losses.last().unwrap();
    assert!(
        final_loss < initial,
        "SGD+momentum: loss should decrease: initial={initial}, final={final_loss}",
    );
}

// ── AdaFactor on multi-layer model ───────────────────────────────────

/// AdaFactor trains a 2-layer MLP — loss decreases.
#[test]
fn test_adafactor_mlp_training_loss_decreases() {
    let (in_dim, hidden, num_classes) = (4, 8, 3);
    let (x_data, t_data) = make_data(12, in_dim, num_classes);
    let (w1, b1, w2, b2) = make_vars(in_dim, hidden, num_classes);

    let mut config = AdaFactorConfig::default();
    config.lr = 0.01;
    let mut adafactor =
        AdaFactor::new(vec![w1.clone(), b1.clone(), w2.clone(), b2.clone()], config).unwrap();

    let mut losses = Vec::new();
    for _ in 0..15 {
        let (loss, loss_val) = train_step(&x_data, &t_data, &w1, &b1, &w2, &b2);
        losses.push(loss_val);
        adafactor.backward_step(&loss).unwrap();
    }

    let initial = losses[0];
    let final_loss = *losses.last().unwrap();
    assert!(
        final_loss < initial,
        "AdaFactor MLP: loss should decrease: initial={initial}, final={final_loss}",
    );
}

// ── LR schedule in training loop ─────────────────────────────────────

/// CosineSchedule with warmup applied via step_with_schedule in a training loop.
/// Verifies LR transitions through warmup → cosine decay → min_lr.
#[test]
fn test_lr_schedule_in_training_loop() {
    let (in_dim, hidden, num_classes) = (4, 8, 3);
    let (x_data, t_data) = make_data(12, in_dim, num_classes);
    let (w1, b1, w2, b2) = make_vars(in_dim, hidden, num_classes);

    let base_lr = 0.01;
    let min_lr = 0.001;
    let warmup_steps = 5;
    let total_steps = 20;

    let schedule = CosineSchedule::new(base_lr, min_lr, warmup_steps, total_steps).unwrap();
    let mut adam = AdamW::new(
        vec![w1.clone(), b1.clone(), w2.clone(), b2.clone()],
        adam_config_no_wd(),
    )
    .unwrap();

    let mut losses = Vec::new();
    let mut lrs = Vec::new();
    for step in 0..total_steps {
        let (loss, loss_val) = train_step(&x_data, &t_data, &w1, &b1, &w2, &b2);
        losses.push(loss_val);
        let grads = backward(&loss).unwrap();
        step_with_schedule(&mut adam, &grads, &schedule, step).unwrap();
        lrs.push(adam.learning_rate());
    }

    // Verify warmup: LR should increase during warmup phase
    assert!(
        lrs[0] < base_lr * 0.5,
        "step 0 lr should be in warmup: {}",
        lrs[0]
    );
    // At end of warmup, LR should be near base_lr
    let lr_at_warmup_end = lrs[warmup_steps - 1];
    assert!(
        (lr_at_warmup_end - base_lr).abs() < base_lr * 0.3,
        "lr at warmup end should be near base_lr: {} vs {}",
        lr_at_warmup_end,
        base_lr,
    );
    // At end of training, LR should be near min_lr
    let final_lr = *lrs.last().unwrap();
    assert!(
        (final_lr - min_lr).abs() < base_lr * 0.15,
        "final lr should be near min_lr: {} vs {}",
        final_lr,
        min_lr,
    );
    // LR should generally decrease after warmup
    assert!(
        lrs[warmup_steps] > lrs[total_steps - 1],
        "lr should decrease from warmup end to final: {} -> {}",
        lrs[warmup_steps],
        lrs[total_steps - 1],
    );
    // Loss should decrease
    assert!(
        *losses.last().unwrap() < losses[0],
        "cosine schedule training: loss should decrease: {} -> {}",
        losses[0],
        losses.last().unwrap(),
    );
}

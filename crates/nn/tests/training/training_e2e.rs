// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! End-to-end training integration test.
//!
//! Verifies the full training chain works:
//! nn layers → forward → loss → backward → optimizer.step() → weight update → repeat.
//!
//! This is the first test that actually trains a neural network in nn.
//! All prior training tests optimize scalar functions (minimize x²) without
//! going through matmul→activation→loss→backward on real weight matrices.
//!
//! Run: `cargo test -p nn --test training_e2e --features training`

#![cfg(feature = "training")]

use std::sync::Arc;

use nn::training::{
    AdamConfig, AdamW, Optimizer, TrackedTensor, TrainableLinear, TrainableModule, Var,
};
use nn::{DType, Device, DynTensor};

use super::common::{forward_mlp, make_data};

// -- AC1: Integration test trains a 2-layer MLP on synthetic data for 10 steps --

#[test]
fn test_train_mlp_loss_decreases() {
    let batch = 12;
    let in_dim = 4;
    let hidden = 8;
    let num_classes = 3;

    let (x_data, t_data) = make_data(batch, in_dim, num_classes);

    // Create trainable weight Vars with small random-like initialization.
    let w1 = Var::new(
        DynTensor::from_vec(
            (0..hidden * in_dim)
                .map(|i| ((i * 17 + 3) % 100) as f32 * 0.02 - 1.0)
                .collect(),
            &[hidden, in_dim],
            &Device::Cpu,
        )
        .unwrap(),
    );
    let b1 = Var::new(DynTensor::zeros(&[1, hidden], DType::F32, &Device::Cpu).unwrap());
    let w2 = Var::new(
        DynTensor::from_vec(
            (0..num_classes * hidden)
                .map(|i| ((i * 23 + 7) % 100) as f32 * 0.02 - 1.0)
                .collect(),
            &[num_classes, hidden],
            &Device::Cpu,
        )
        .unwrap(),
    );
    let b2 = Var::new(DynTensor::zeros(&[1, num_classes], DType::F32, &Device::Cpu).unwrap());

    let mut adam_config = AdamConfig::default();
    adam_config.lr = 0.01;
    adam_config.weight_decay = 0.0;
    let mut adam = AdamW::new(
        vec![w1.clone(), b1.clone(), w2.clone(), b2.clone()],
        adam_config,
    )
    .unwrap();

    let mut losses = Vec::new();
    let num_steps = 10;

    for _ in 0..num_steps {
        let tw1 = Arc::new(TrackedTensor::from_var(&w1).unwrap());
        let tb1 = Arc::new(TrackedTensor::from_var(&b1).unwrap());
        let tw2 = Arc::new(TrackedTensor::from_var(&w2).unwrap());
        let tb2 = Arc::new(TrackedTensor::from_var(&b2).unwrap());
        let tx = Arc::new(TrackedTensor::from_tensor(x_data.clone()));

        let logits = forward_mlp(&tx, &tw1, &tb1, &tw2, &tb2);

        let t_targets = Arc::new(TrackedTensor::from_tensor(t_data.clone()));
        let loss = logits.cross_entropy_loss(&t_targets, 1).unwrap();

        let loss_val = loss.tensor().to_flat_vec::<f32>().unwrap()[0];
        assert!(
            loss_val.is_finite(),
            "loss is NaN/Inf at step {}",
            losses.len()
        );
        losses.push(loss_val);

        adam.backward_step(&loss).unwrap();
    }

    // AC2: Verify loss decreases (final < initial).
    let initial = losses[0];
    let final_loss = *losses.last().unwrap();
    assert!(
        final_loss < initial,
        "loss should decrease: initial={initial}, final={final_loss}",
    );
}

// -- AC3: All trainable parameters received non-zero gradients --

#[test]
fn test_train_mlp_all_gradients_nonzero() {
    let batch = 8;
    let in_dim = 4;
    let hidden = 6;
    let num_classes = 3;

    let (x_data, t_data) = make_data(batch, in_dim, num_classes);

    let w1 = Var::new(
        DynTensor::from_vec(
            (0..hidden * in_dim)
                .map(|i| ((i * 17 + 3) % 100) as f32 * 0.02 - 1.0)
                .collect(),
            &[hidden, in_dim],
            &Device::Cpu,
        )
        .unwrap(),
    );
    let b1 = Var::new(DynTensor::zeros(&[1, hidden], DType::F32, &Device::Cpu).unwrap());
    let w2 = Var::new(
        DynTensor::from_vec(
            (0..num_classes * hidden)
                .map(|i| ((i * 23 + 7) % 100) as f32 * 0.02 - 1.0)
                .collect(),
            &[num_classes, hidden],
            &Device::Cpu,
        )
        .unwrap(),
    );
    let b2 = Var::new(DynTensor::zeros(&[1, num_classes], DType::F32, &Device::Cpu).unwrap());

    // Single forward+backward to get gradients.
    let tw1 = Arc::new(TrackedTensor::from_var(&w1).unwrap());
    let tb1 = Arc::new(TrackedTensor::from_var(&b1).unwrap());
    let tw2 = Arc::new(TrackedTensor::from_var(&w2).unwrap());
    let tb2 = Arc::new(TrackedTensor::from_var(&b2).unwrap());
    let tx = Arc::new(TrackedTensor::from_tensor(x_data));

    let logits = forward_mlp(&tx, &tw1, &tb1, &tw2, &tb2);
    let t_targets = Arc::new(TrackedTensor::from_tensor(t_data));
    let loss = logits.cross_entropy_loss(&t_targets, 1).unwrap();

    let grads = nn::training::backward(&loss).unwrap();

    // Every trainable parameter should have a non-zero gradient.
    for (name, var) in [("w1", &w1), ("b1", &b1), ("w2", &w2), ("b2", &b2)] {
        let g = grads
            .get(var)
            .unwrap_or_else(|| panic!("no gradient for {name}"));
        let max_abs = g
            .to_flat_vec::<f32>()
            .unwrap()
            .iter()
            .map(|v| v.abs())
            .fold(0.0f32, f32::max);
        assert!(
            max_abs > 0.0,
            "gradient for {name} is all-zero (max_abs={max_abs})",
        );
    }
}

// -- AC4: Full pipeline uses nn-style linear + cross_entropy + AdamW --

#[test]
fn test_train_mlp_weights_actually_update() {
    let batch = 8;
    let in_dim = 4;
    let hidden = 6;
    let num_classes = 3;

    let (x_data, t_data) = make_data(batch, in_dim, num_classes);

    let w1 = Var::new(
        DynTensor::from_vec(
            (0..hidden * in_dim)
                .map(|i| ((i * 17 + 3) % 100) as f32 * 0.02 - 1.0)
                .collect(),
            &[hidden, in_dim],
            &Device::Cpu,
        )
        .unwrap(),
    );
    let b1 = Var::new(DynTensor::zeros(&[1, hidden], DType::F32, &Device::Cpu).unwrap());
    let w2 = Var::new(
        DynTensor::from_vec(
            (0..num_classes * hidden)
                .map(|i| ((i * 23 + 7) % 100) as f32 * 0.02 - 1.0)
                .collect(),
            &[num_classes, hidden],
            &Device::Cpu,
        )
        .unwrap(),
    );
    let b2 = Var::new(DynTensor::zeros(&[1, num_classes], DType::F32, &Device::Cpu).unwrap());

    // Snapshot initial weights.
    let w1_before = w1.data().unwrap().to_flat_vec::<f32>().unwrap();
    let w2_before = w2.data().unwrap().to_flat_vec::<f32>().unwrap();
    let b1_before = b1.data().unwrap().to_flat_vec::<f32>().unwrap();
    let b2_before = b2.data().unwrap().to_flat_vec::<f32>().unwrap();

    let mut adam_config = AdamConfig::default();
    adam_config.lr = 0.01;
    adam_config.weight_decay = 0.0;
    let mut adam = AdamW::new(
        vec![w1.clone(), b1.clone(), w2.clone(), b2.clone()],
        adam_config,
    )
    .unwrap();

    // Single training step.
    let tw1 = Arc::new(TrackedTensor::from_var(&w1).unwrap());
    let tb1 = Arc::new(TrackedTensor::from_var(&b1).unwrap());
    let tw2 = Arc::new(TrackedTensor::from_var(&w2).unwrap());
    let tb2 = Arc::new(TrackedTensor::from_var(&b2).unwrap());
    let tx = Arc::new(TrackedTensor::from_tensor(x_data));
    let logits = forward_mlp(&tx, &tw1, &tb1, &tw2, &tb2);
    let t_targets = Arc::new(TrackedTensor::from_tensor(t_data));
    let loss = logits.cross_entropy_loss(&t_targets, 1).unwrap();
    adam.backward_step(&loss).unwrap();

    // Verify weights changed.
    let w1_after = w1.data().unwrap().to_flat_vec::<f32>().unwrap();
    let w2_after = w2.data().unwrap().to_flat_vec::<f32>().unwrap();
    let b1_after = b1.data().unwrap().to_flat_vec::<f32>().unwrap();
    let b2_after = b2.data().unwrap().to_flat_vec::<f32>().unwrap();

    assert_ne!(w1_before, w1_after, "w1 should change after optimizer step");
    assert_ne!(w2_before, w2_after, "w2 should change after optimizer step");
    assert_ne!(b1_before, b1_after, "b1 should change after optimizer step");
    assert_ne!(b2_before, b2_after, "b2 should change after optimizer step");
}

// -- Helpers for TrainableLinear tests --

/// Create a TrainableLinear layer with deterministic pseudo-random weights.
fn make_trainable_layer(out: usize, inp: usize, seed: usize) -> TrainableLinear {
    TrainableLinear::from_tensors(
        DynTensor::from_vec(
            (0..out * inp)
                .map(|i| ((i * seed + 3) % 100) as f32 * 0.02 - 1.0)
                .collect(),
            &[out, inp],
            &Device::Cpu,
        )
        .unwrap(),
        Some(DynTensor::zeros(&[1, out], DType::F32, &Device::Cpu).unwrap()),
    )
}

// -- AC5: layers::Linear equivalent via TrainableLinear (#1456 AC1+AC2) --

#[test]
fn test_train_trainable_linear_loss_decreases() {
    let (in_dim, hidden, num_classes) = (4, 8, 3);
    let (x_data, t_data) = make_data(12, in_dim, num_classes);

    let layer1 = make_trainable_layer(hidden, in_dim, 17);
    let layer2 = make_trainable_layer(num_classes, hidden, 23);

    let all_vars: Vec<Var> = [layer1.vars(), layer2.vars()]
        .concat()
        .into_iter()
        .cloned()
        .collect();

    let mut adam_config = AdamConfig::default();
    adam_config.lr = 0.01;
    adam_config.weight_decay = 0.0;
    let mut adam = AdamW::new(all_vars, adam_config).unwrap();

    let mut losses = Vec::new();
    for _ in 0..10 {
        let tx = Arc::new(TrackedTensor::from_tensor(x_data.clone()));
        let h = layer1.forward(&tx).unwrap();
        let h = h.relu().unwrap();
        let logits = layer2.forward(&h).unwrap();

        let t_targets = Arc::new(TrackedTensor::from_tensor(t_data.clone()));
        let loss = logits.cross_entropy_loss(&t_targets, 1).unwrap();

        let loss_val = loss.tensor().to_flat_vec::<f32>().unwrap()[0];
        assert!(
            loss_val.is_finite(),
            "loss is NaN/Inf at step {}",
            losses.len()
        );
        losses.push(loss_val);
        adam.backward_step(&loss).unwrap();
    }

    let initial = losses[0];
    let final_loss = *losses.last().unwrap();
    assert!(
        final_loss < initial,
        "TrainableLinear: loss should decrease: initial={initial}, final={final_loss}",
    );
}

// -- AC6: TrainableModule trait enables generic training loop (#1456 AC1) --

#[test]
fn test_trainable_module_generic_loop() {
    let batch = 8;
    let in_dim = 4;
    let out_dim = 3;

    let x_data = DynTensor::from_vec(
        (0..batch * in_dim)
            .map(|i| ((i * 11 + 5) % 100) as f32 * 0.02 - 1.0)
            .collect(),
        &[batch, in_dim],
        &Device::Cpu,
    )
    .unwrap();

    let layer = TrainableLinear::new(in_dim, out_dim, true).unwrap();

    // Use trait object to demonstrate generic interface.
    let module: &dyn TrainableModule = &layer;
    let vars: Vec<Var> = module.vars().into_iter().cloned().collect();
    assert_eq!(vars.len(), 2, "weight + bias");

    let mut adam_config = AdamConfig::default();
    adam_config.lr = 0.01;
    let mut adam = AdamW::new(vars, adam_config).unwrap();

    // Single step: forward → scalar loss → backward → step.
    let tx = Arc::new(TrackedTensor::from_tensor(x_data));
    let y = module.forward(&tx).unwrap();
    let loss = y.sum_keepdim(1).unwrap().sum_keepdim(0).unwrap();
    adam.backward_step(&loss).unwrap();

    // Weight should have changed.
    let w_after = layer.weight().data().unwrap().to_flat_vec::<f32>().unwrap();
    let all_zero = w_after.iter().all(|v| *v == 0.0);
    assert!(!all_zero, "weights should be non-zero after optimizer step");
}

// -- AC7: E2E training test with non-Linear trainable layers (#1479 AC5) --

/// Train a Linear→LayerNorm→Linear network using TrainableModule wrappers.
/// Exercises TrainableLayerNorm backward through a real training loop.
#[test]
fn test_train_layer_norm_network_loss_decreases() {
    use nn::training::TrainableLayerNorm;

    let batch = 12;
    let in_dim = 4;
    let hidden = 8;
    let num_classes = 3;

    let (x_data, t_data) = make_data(batch, in_dim, num_classes);

    // Build: Linear(4→8) → LayerNorm(8) → ReLU → Linear(8→3)
    let layer1 = make_trainable_layer(hidden, in_dim, 17);
    let ln = TrainableLayerNorm::new(hidden, 1e-5).unwrap();
    let layer2 = make_trainable_layer(num_classes, hidden, 23);

    // Collect all Vars from all layers.
    let all_vars: Vec<Var> = [layer1.vars(), ln.vars(), layer2.vars()]
        .concat()
        .into_iter()
        .cloned()
        .collect();
    assert_eq!(all_vars.len(), 6, "w1+b1 + ln_weight+ln_bias + w2+b2");

    let mut adam_config = AdamConfig::default();
    adam_config.lr = 0.01;
    adam_config.weight_decay = 0.0;
    let mut adam = AdamW::new(all_vars, adam_config).unwrap();

    let mut losses = Vec::new();
    for step in 0..15 {
        let tx = Arc::new(TrackedTensor::from_tensor(x_data.clone()));
        let h = layer1.forward(&tx).unwrap();
        let h = ln.forward(&h).unwrap();
        let h = h.relu().unwrap();
        let logits = layer2.forward(&h).unwrap();

        let t_targets = Arc::new(TrackedTensor::from_tensor(t_data.clone()));
        let loss = logits.cross_entropy_loss(&t_targets, 1).unwrap();

        let loss_val = loss.tensor().to_flat_vec::<f32>().unwrap()[0];
        assert!(loss_val.is_finite(), "loss is NaN/Inf at step {step}",);
        losses.push(loss_val);
        adam.backward_step(&loss).unwrap();
    }

    let initial = losses[0];
    let final_loss = *losses.last().unwrap();
    assert!(
        final_loss < initial,
        "LayerNorm network: loss should decrease: initial={initial}, final={final_loss}",
    );

    // Verify LayerNorm weight and bias actually updated.
    let ln_w = ln.weight().data().unwrap().to_flat_vec::<f32>().unwrap();
    let ln_b = ln.bias().data().unwrap().to_flat_vec::<f32>().unwrap();
    let any_w_changed = ln_w.iter().any(|v| (*v - 1.0).abs() > 1e-7);
    let any_b_changed = ln_b.iter().any(|v| v.abs() > 1e-7);
    assert!(
        any_w_changed,
        "LayerNorm weight should differ from init=1.0"
    );
    assert!(any_b_changed, "LayerNorm bias should differ from init=0.0");
}

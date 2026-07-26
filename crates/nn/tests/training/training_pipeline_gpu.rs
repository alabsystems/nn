// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! GPU training integration tests.
//!
//! Verifies that the full training pipeline (forward → backward → optimizer step)
//! works end-to-end on Metal GPU tensors. This is the first test file that
//! exercises `nn-autodiff` and `nn-optim` with `Device::Metal`.
//!
//! All existing training tests are CPU-only. These tests verify that:
//! - Backward pass produces correct gradients on GPU (device-agnostic ops)
//! - Adam/SGD optimizer steps work on GPU gradients
//! - GradScaler + mixed-precision pipeline works on GPU
//! - Gradient clipping operates on GPU tensors
//! - GPU training produces the same results as CPU training (parity)
//!
//! Run: `cargo test -p nn --test training_pipeline_gpu --features training`

#![cfg(feature = "training")]

use std::sync::Arc;

use nn::training::{
    backward, clip_grad_norm, AdamConfig, AdamW, GradScaler, GradScalerConfig, Optimizer, Sgd,
    SgdConfig, TrackedTensor, Var,
};
use nn::{DType, Device, DynTensor};

/// Initialize Metal GPU backend for DynTensor ops. Returns true if GPU is available.
fn init_gpu() -> bool {
    match nn_metal::MetalBackend::init() {
        Ok(_) => {
            nn_metal::register_metal_dyn_backend();
            true
        }
        Err(_) => false,
    }
}

/// Create a deterministic weight tensor on the specified device.
fn make_weight(rows: usize, cols: usize, seed: usize, device: &Device) -> DynTensor {
    let cpu = DynTensor::from_vec(
        (0..rows * cols)
            .map(|i| ((i * seed + 3) % 100) as f32 * 0.02 - 1.0)
            .collect(),
        &[rows, cols],
        &Device::Cpu,
    )
    .unwrap();
    if device.is_gpu() {
        cpu.to_device(device).unwrap()
    } else {
        cpu
    }
}

/// Create synthetic classification data on the specified device.
fn make_data(n: usize, dim: usize, num_classes: usize, device: &Device) -> (DynTensor, DynTensor) {
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
    if device.is_gpu() {
        (x.to_device(device).unwrap(), t.to_device(device).unwrap())
    } else {
        (x, t)
    }
}

/// 2-layer MLP forward pass: logits = relu(x @ w1^T + b1) @ w2^T + b2
fn forward_mlp(
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

/// Run a training step and return (loss_tracked, loss_val).
fn train_step(
    x: &DynTensor,
    t: &DynTensor,
    w1: &Var,
    b1: &Var,
    w2: &Var,
    b2: &Var,
) -> (Arc<TrackedTensor>, f32) {
    let tw1 = Arc::new(TrackedTensor::from_var(w1).unwrap());
    let tb1 = Arc::new(TrackedTensor::from_var(b1).unwrap());
    let tw2 = Arc::new(TrackedTensor::from_var(w2).unwrap());
    let tb2 = Arc::new(TrackedTensor::from_var(b2).unwrap());
    let tx = Arc::new(TrackedTensor::from_tensor(x.clone()));
    let logits = forward_mlp(&tx, &tw1, &tb1, &tw2, &tb2);
    let t_targets = Arc::new(TrackedTensor::from_tensor(t.clone()));
    let loss = logits.cross_entropy_loss(&t_targets, 1).unwrap();
    let loss_val = loss.tensor().to_scalar::<f32>().unwrap();
    assert!(loss_val.is_finite(), "loss is NaN/Inf: {loss_val}");
    (loss, loss_val)
}

// ── GPU training: Adam ──────────────────────────────────────────────

/// Train a 2-layer MLP on GPU with Adam. Verify loss decreases.
#[test]
fn test_gpu_adam_loss_decreases() {
    if !init_gpu() {
        eprintln!("Metal GPU not available, skipping test");
        return;
    }
    let device = Device::metal();
    let (batch, in_dim, hidden, classes) = (12, 4, 8, 3);

    let (x_data, t_data) = make_data(batch, in_dim, classes, &device);
    let w1 = Var::new(make_weight(hidden, in_dim, 17, &device));
    let b1 = Var::new(DynTensor::zeros(&[1, hidden], DType::F32, &device).unwrap());
    let w2 = Var::new(make_weight(classes, hidden, 23, &device));
    let b2 = Var::new(DynTensor::zeros(&[1, classes], DType::F32, &device).unwrap());

    let mut config = AdamConfig::default();
    config.lr = 0.01;
    config.weight_decay = 0.0;
    let mut adam =
        AdamW::new(vec![w1.clone(), b1.clone(), w2.clone(), b2.clone()], config).unwrap();

    let mut losses = Vec::new();
    for _ in 0..10 {
        let (loss, loss_val) = train_step(&x_data, &t_data, &w1, &b1, &w2, &b2);
        losses.push(loss_val);
        adam.backward_step(&loss).unwrap();
    }

    let first = losses[0];
    let last = *losses.last().unwrap();
    assert!(
        last < first,
        "GPU Adam: loss should decrease: first={first}, last={last}"
    );
}

// ── GPU training: SGD ───────────────────────────────────────────────

/// Train on GPU with SGD + momentum. Verify loss decreases.
#[test]
fn test_gpu_sgd_loss_decreases() {
    if !init_gpu() {
        eprintln!("Metal GPU not available, skipping test");
        return;
    }
    let device = Device::metal();
    let (batch, in_dim, hidden, classes) = (12, 4, 8, 3);

    let (x_data, t_data) = make_data(batch, in_dim, classes, &device);
    let w1 = Var::new(make_weight(hidden, in_dim, 17, &device));
    let b1 = Var::new(DynTensor::zeros(&[1, hidden], DType::F32, &device).unwrap());
    let w2 = Var::new(make_weight(classes, hidden, 23, &device));
    let b2 = Var::new(DynTensor::zeros(&[1, classes], DType::F32, &device).unwrap());

    let mut config = SgdConfig::default();
    config.lr = 0.1;
    config.momentum = 0.9;
    let mut sgd = Sgd::new(vec![w1.clone(), b1.clone(), w2.clone(), b2.clone()], config).unwrap();

    let mut losses = Vec::new();
    for _ in 0..10 {
        let (loss, loss_val) = train_step(&x_data, &t_data, &w1, &b1, &w2, &b2);
        losses.push(loss_val);
        sgd.backward_step(&loss).unwrap();
    }

    let first = losses[0];
    let last = *losses.last().unwrap();
    assert!(
        last < first,
        "GPU SGD: loss should decrease: first={first}, last={last}"
    );
}

// ── GPU/CPU gradient parity ─────────────────────────────────────────

/// Verify GPU and CPU produce the same gradients for a single backward pass.
#[test]
fn test_gpu_cpu_gradient_parity() {
    if !init_gpu() {
        eprintln!("Metal GPU not available, skipping test");
        return;
    }

    let (batch, in_dim, hidden, classes) = (8, 4, 6, 3);

    // Create identical data on CPU and GPU
    let (x_cpu, t_cpu) = make_data(batch, in_dim, classes, &Device::Cpu);
    let x_gpu = x_cpu.to_device(&Device::metal()).unwrap();
    let t_gpu = t_cpu.to_device(&Device::metal()).unwrap();

    // Create identical weights on CPU
    let w1_data = make_weight(hidden, in_dim, 17, &Device::Cpu);
    let b1_data = DynTensor::zeros(&[1, hidden], DType::F32, &Device::Cpu).unwrap();
    let w2_data = make_weight(classes, hidden, 23, &Device::Cpu);
    let b2_data = DynTensor::zeros(&[1, classes], DType::F32, &Device::Cpu).unwrap();

    // CPU backward
    let w1_cpu = Var::new(w1_data.clone());
    let b1_cpu = Var::new(b1_data.clone());
    let w2_cpu = Var::new(w2_data.clone());
    let b2_cpu = Var::new(b2_data.clone());

    let (cpu_loss, _) = train_step(&x_cpu, &t_cpu, &w1_cpu, &b1_cpu, &w2_cpu, &b2_cpu);
    let cpu_grads = backward(&cpu_loss).unwrap();

    // GPU backward
    let w1_gpu = Var::new(w1_data.to_device(&Device::metal()).unwrap());
    let b1_gpu = Var::new(b1_data.to_device(&Device::metal()).unwrap());
    let w2_gpu = Var::new(w2_data.to_device(&Device::metal()).unwrap());
    let b2_gpu = Var::new(b2_data.to_device(&Device::metal()).unwrap());

    let (gpu_loss, _) = train_step(&x_gpu, &t_gpu, &w1_gpu, &b1_gpu, &w2_gpu, &b2_gpu);
    let gpu_grads = backward(&gpu_loss).unwrap();

    // Compare gradients for each variable
    let tol = 1e-4;
    for (cpu_var, gpu_var, name) in [
        (&w1_cpu, &w1_gpu, "w1"),
        (&b1_cpu, &b1_gpu, "b1"),
        (&w2_cpu, &w2_gpu, "w2"),
        (&b2_cpu, &b2_gpu, "b2"),
    ] {
        let cpu_grad = cpu_grads.get(cpu_var).unwrap();
        let gpu_grad = gpu_grads.get(gpu_var).unwrap();

        let cpu_vals = cpu_grad.to_flat_vec::<f32>().unwrap();
        let gpu_vals = gpu_grad
            .to_device(&Device::Cpu)
            .unwrap()
            .to_flat_vec::<f32>()
            .unwrap();

        assert_eq!(
            cpu_vals.len(),
            gpu_vals.len(),
            "{name}: gradient length mismatch"
        );
        for (i, (c, g)) in cpu_vals.iter().zip(gpu_vals.iter()).enumerate() {
            assert!(
                (c - g).abs() < tol,
                "{name}[{i}]: CPU grad={c}, GPU grad={g}, diff={}",
                (c - g).abs()
            );
        }
    }
}

// ── GPU GradScaler ──────────────────────────────────────────────────

/// GradScaler + Adam on GPU: scale → backward → unscale → step → update.
#[test]
fn test_gpu_grad_scaler_pipeline() {
    if !init_gpu() {
        eprintln!("Metal GPU not available, skipping test");
        return;
    }
    let device = Device::metal();
    let (batch, in_dim, hidden, classes) = (8, 4, 6, 3);

    let (x_data, t_data) = make_data(batch, in_dim, classes, &device);
    let w1 = Var::new(make_weight(hidden, in_dim, 17, &device));
    let b1 = Var::new(DynTensor::zeros(&[1, hidden], DType::F32, &device).unwrap());
    let w2 = Var::new(make_weight(classes, hidden, 23, &device));
    let b2 = Var::new(DynTensor::zeros(&[1, classes], DType::F32, &device).unwrap());

    let mut config = AdamConfig::default();
    config.lr = 0.01;
    config.weight_decay = 0.0;
    let mut adam =
        AdamW::new(vec![w1.clone(), b1.clone(), w2.clone(), b2.clone()], config).unwrap();
    let mut scaler_config = GradScalerConfig::default();
    scaler_config.init_scale = 256.0;
    scaler_config.growth_interval = 100;
    let mut scaler = GradScaler::new(scaler_config).unwrap();

    let mut losses = Vec::new();
    for _ in 0..10 {
        let (loss, loss_val) = train_step(&x_data, &t_data, &w1, &b1, &w2, &b2);
        losses.push(loss_val);

        let scaled = scaler.scale_loss(&loss).unwrap();
        let mut grads = backward(&scaled).unwrap();
        if scaler.unscale_and_check(&mut grads).unwrap() {
            adam.step(&grads).unwrap();
        }
        scaler.update();
    }

    let first = losses[0];
    let last = *losses.last().unwrap();
    assert!(
        last < first,
        "GPU GradScaler+Adam: loss should decrease: first={first}, last={last}"
    );
    assert!(
        !scaler.found_inf(),
        "no inf should be found with normal training"
    );
}

// ── GPU gradient clipping ───────────────────────────────────────────

/// Gradient clipping on GPU: backward → clip → step.
#[test]
fn test_gpu_gradient_clipping() {
    if !init_gpu() {
        eprintln!("Metal GPU not available, skipping test");
        return;
    }
    let device = Device::metal();
    let (batch, in_dim, hidden, classes) = (8, 4, 6, 3);

    let (x_data, t_data) = make_data(batch, in_dim, classes, &device);
    let w1 = Var::new(make_weight(hidden, in_dim, 17, &device));
    let b1 = Var::new(DynTensor::zeros(&[1, hidden], DType::F32, &device).unwrap());
    let w2 = Var::new(make_weight(classes, hidden, 23, &device));
    let b2 = Var::new(DynTensor::zeros(&[1, classes], DType::F32, &device).unwrap());

    let mut config = AdamConfig::default();
    config.lr = 0.01;
    config.weight_decay = 0.0;
    let mut adam =
        AdamW::new(vec![w1.clone(), b1.clone(), w2.clone(), b2.clone()], config).unwrap();

    let mut losses = Vec::new();
    for _ in 0..10 {
        let (loss, loss_val) = train_step(&x_data, &t_data, &w1, &b1, &w2, &b2);
        losses.push(loss_val);

        let mut grads = backward(&loss).unwrap();
        let total_norm = clip_grad_norm(&mut grads, 1.0).unwrap();
        assert!(total_norm.is_finite(), "total_norm should be finite");
        adam.step(&grads).unwrap();
    }

    let first = losses[0];
    let last = *losses.last().unwrap();
    assert!(
        last < first,
        "GPU clipped Adam: loss should decrease: first={first}, last={last}"
    );
}

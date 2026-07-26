// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! GPU conv backward integration tests.
//!
//! Verifies that the GEMM-based (im2col + matmul) conv backward rules produce
//! correct gradients when tensors live on Metal GPU. These tests compare
//! GPU backward gradients against CPU backward gradients element-by-element.
//!
//! Run: `cargo test -p nn --test training_conv_gpu --features training`

#![cfg(feature = "training")]

use std::sync::Arc;

use nn::training::{backward, AdamConfig, AdamW, Optimizer, TrackedTensor, Var};
use nn::{DType, Device, DynTensor};

/// Initialize Metal GPU backend. Returns true if available.
fn init_gpu() -> bool {
    match nn_metal::MetalBackend::init() {
        Ok(_) => {
            nn_metal::register_metal_dyn_backend();
            true
        }
        Err(_) => false,
    }
}

/// Create a deterministic tensor from a seed. Values in [-1, 1].
fn seeded_tensor(dims: &[usize], seed: usize, device: &Device) -> DynTensor {
    let n: usize = dims.iter().product();
    let data: Vec<f32> = (0..n)
        .map(|i| ((i * seed + 7) % 200) as f32 * 0.01 - 1.0)
        .collect();
    let cpu = DynTensor::from_vec(data, dims, &Device::Cpu).unwrap();
    if device.is_gpu() {
        cpu.to_device(device).unwrap()
    } else {
        cpu
    }
}

/// Reduce a tracked tensor to a scalar by summing over all dimensions.
fn sum_all(t: &Arc<TrackedTensor>) -> Arc<TrackedTensor> {
    let ndim = t.tensor().dims().len();
    let mut result = Arc::clone(t);
    for d in (0..ndim).rev() {
        result = result.sum_keepdim(d).unwrap();
    }
    result
}

/// Assert two gradient tensors match within tolerance.
fn assert_grads_close(cpu_grad: &DynTensor, gpu_grad: &DynTensor, tol: f32, label: &str) {
    let cpu_vals = cpu_grad.to_flat_vec::<f32>().unwrap();
    let gpu_vals = gpu_grad
        .to_device(&Device::Cpu)
        .unwrap()
        .to_flat_vec::<f32>()
        .unwrap();
    assert_eq!(
        cpu_vals.len(),
        gpu_vals.len(),
        "{label}: gradient length mismatch"
    );
    for (i, (c, g)) in cpu_vals.iter().zip(gpu_vals.iter()).enumerate() {
        let diff = (c - g).abs();
        assert!(
            diff < tol,
            "{label}[{i}]: CPU grad={c}, GPU grad={g}, diff={diff}"
        );
    }
}

// ── Conv1d GPU/CPU backward parity ──────────────────────────────────

/// Conv1d backward: verify GPU gradients match CPU for both input and kernel.
#[test]
fn test_gpu_conv1d_backward_parity() {
    if !init_gpu() {
        eprintln!("Metal GPU not available, skipping test");
        return;
    }
    let gpu = Device::metal();
    let (batch, in_ch, in_len, out_ch, k_size) = (2, 3, 8, 4, 3);
    let (padding, stride, dilation, groups) = (1, 1, 1, 1);

    // CPU backward
    let x_data = seeded_tensor(&[batch, in_ch, in_len], 11, &Device::Cpu);
    let k_data = seeded_tensor(&[out_ch, in_ch, k_size], 13, &Device::Cpu);

    let vx_cpu = Var::new(x_data.clone());
    let vk_cpu = Var::new(k_data.clone());
    let tx_cpu = Arc::new(TrackedTensor::from_var(&vx_cpu).unwrap());
    let tk_cpu = Arc::new(TrackedTensor::from_var(&vk_cpu).unwrap());
    let out_cpu = tx_cpu
        .conv1d(&tk_cpu, padding, stride, dilation, groups)
        .unwrap();
    let loss_cpu = sum_all(&out_cpu.sqr().unwrap());
    let grads_cpu = backward(&loss_cpu).unwrap();

    // GPU backward
    let vx_gpu = Var::new(x_data.to_device(&gpu).unwrap());
    let vk_gpu = Var::new(k_data.to_device(&gpu).unwrap());
    let tx_gpu = Arc::new(TrackedTensor::from_var(&vx_gpu).unwrap());
    let tk_gpu = Arc::new(TrackedTensor::from_var(&vk_gpu).unwrap());
    let out_gpu = tx_gpu
        .conv1d(&tk_gpu, padding, stride, dilation, groups)
        .unwrap();
    let loss_gpu = sum_all(&out_gpu.sqr().unwrap());
    let grads_gpu = backward(&loss_gpu).unwrap();

    let tol = 1e-4;
    assert_grads_close(
        grads_cpu.get(&vx_cpu).unwrap(),
        grads_gpu.get(&vx_gpu).unwrap(),
        tol,
        "conv1d_input_grad",
    );
    assert_grads_close(
        grads_cpu.get(&vk_cpu).unwrap(),
        grads_gpu.get(&vk_gpu).unwrap(),
        tol,
        "conv1d_kernel_grad",
    );
}

/// Conv1d backward with groups=2 and dilation=2: GPU/CPU parity.
#[test]
fn test_gpu_conv1d_groups_dilation_backward_parity() {
    if !init_gpu() {
        eprintln!("Metal GPU not available, skipping test");
        return;
    }
    let gpu = Device::metal();
    let (batch, in_ch, in_len, out_ch, k_size) = (2, 4, 10, 6, 3);
    let (padding, stride, dilation, groups) = (1, 1, 2, 2);

    let x_data = seeded_tensor(&[batch, in_ch, in_len], 17, &Device::Cpu);
    let k_data = seeded_tensor(&[out_ch, in_ch / groups, k_size], 19, &Device::Cpu);

    let vx_cpu = Var::new(x_data.clone());
    let vk_cpu = Var::new(k_data.clone());
    let tx_cpu = Arc::new(TrackedTensor::from_var(&vx_cpu).unwrap());
    let tk_cpu = Arc::new(TrackedTensor::from_var(&vk_cpu).unwrap());
    let out_cpu = tx_cpu
        .conv1d(&tk_cpu, padding, stride, dilation, groups)
        .unwrap();
    let loss_cpu = sum_all(&out_cpu.sqr().unwrap());
    let grads_cpu = backward(&loss_cpu).unwrap();

    let vx_gpu = Var::new(x_data.to_device(&gpu).unwrap());
    let vk_gpu = Var::new(k_data.to_device(&gpu).unwrap());
    let tx_gpu = Arc::new(TrackedTensor::from_var(&vx_gpu).unwrap());
    let tk_gpu = Arc::new(TrackedTensor::from_var(&vk_gpu).unwrap());
    let out_gpu = tx_gpu
        .conv1d(&tk_gpu, padding, stride, dilation, groups)
        .unwrap();
    let loss_gpu = sum_all(&out_gpu.sqr().unwrap());
    let grads_gpu = backward(&loss_gpu).unwrap();

    let tol = 1e-4;
    assert_grads_close(
        grads_cpu.get(&vx_cpu).unwrap(),
        grads_gpu.get(&vx_gpu).unwrap(),
        tol,
        "conv1d_groups2_dilation2_input_grad",
    );
    assert_grads_close(
        grads_cpu.get(&vk_cpu).unwrap(),
        grads_gpu.get(&vk_gpu).unwrap(),
        tol,
        "conv1d_groups2_dilation2_kernel_grad",
    );
}

// ── ConvTranspose1d GPU/CPU backward parity ─────────────────────────

/// ConvTranspose1d backward: GPU/CPU parity for both input and kernel gradients.
#[test]
fn test_gpu_conv_transpose1d_backward_parity() {
    if !init_gpu() {
        eprintln!("Metal GPU not available, skipping test");
        return;
    }
    let gpu = Device::metal();
    let (batch, in_ch, in_len, out_ch, k_size) = (2, 3, 6, 4, 3);
    let (padding, stride, dilation, groups, output_padding) = (1, 1, 1, 1, 0);

    let x_data = seeded_tensor(&[batch, in_ch, in_len], 23, &Device::Cpu);
    let k_data = seeded_tensor(&[in_ch, out_ch, k_size], 29, &Device::Cpu);

    let vx_cpu = Var::new(x_data.clone());
    let vk_cpu = Var::new(k_data.clone());
    let tx_cpu = Arc::new(TrackedTensor::from_var(&vx_cpu).unwrap());
    let tk_cpu = Arc::new(TrackedTensor::from_var(&vk_cpu).unwrap());
    let out_cpu = tx_cpu
        .conv_transpose1d(&tk_cpu, padding, stride, dilation, groups, output_padding)
        .unwrap();
    let loss_cpu = sum_all(&out_cpu.sqr().unwrap());
    let grads_cpu = backward(&loss_cpu).unwrap();

    let vx_gpu = Var::new(x_data.to_device(&gpu).unwrap());
    let vk_gpu = Var::new(k_data.to_device(&gpu).unwrap());
    let tx_gpu = Arc::new(TrackedTensor::from_var(&vx_gpu).unwrap());
    let tk_gpu = Arc::new(TrackedTensor::from_var(&vk_gpu).unwrap());
    let out_gpu = tx_gpu
        .conv_transpose1d(&tk_gpu, padding, stride, dilation, groups, output_padding)
        .unwrap();
    let loss_gpu = sum_all(&out_gpu.sqr().unwrap());
    let grads_gpu = backward(&loss_gpu).unwrap();

    let tol = 1e-4;
    assert_grads_close(
        grads_cpu.get(&vx_cpu).unwrap(),
        grads_gpu.get(&vx_gpu).unwrap(),
        tol,
        "conv_transpose1d_input_grad",
    );
    assert_grads_close(
        grads_cpu.get(&vk_cpu).unwrap(),
        grads_gpu.get(&vk_gpu).unwrap(),
        tol,
        "conv_transpose1d_kernel_grad",
    );
}

/// ConvTranspose1d backward with stride=2: GPU/CPU parity.
#[test]
fn test_gpu_conv_transpose1d_stride2_backward_parity() {
    if !init_gpu() {
        eprintln!("Metal GPU not available, skipping test");
        return;
    }
    let gpu = Device::metal();
    let (batch, in_ch, in_len, out_ch, k_size) = (2, 3, 6, 4, 3);
    let (padding, stride, dilation, groups, output_padding) = (1, 2, 1, 1, 1);

    let x_data = seeded_tensor(&[batch, in_ch, in_len], 31, &Device::Cpu);
    let k_data = seeded_tensor(&[in_ch, out_ch, k_size], 37, &Device::Cpu);

    let vx_cpu = Var::new(x_data.clone());
    let vk_cpu = Var::new(k_data.clone());
    let tx_cpu = Arc::new(TrackedTensor::from_var(&vx_cpu).unwrap());
    let tk_cpu = Arc::new(TrackedTensor::from_var(&vk_cpu).unwrap());
    let out_cpu = tx_cpu
        .conv_transpose1d(&tk_cpu, padding, stride, dilation, groups, output_padding)
        .unwrap();
    let loss_cpu = sum_all(&out_cpu.sqr().unwrap());
    let grads_cpu = backward(&loss_cpu).unwrap();

    let vx_gpu = Var::new(x_data.to_device(&gpu).unwrap());
    let vk_gpu = Var::new(k_data.to_device(&gpu).unwrap());
    let tx_gpu = Arc::new(TrackedTensor::from_var(&vx_gpu).unwrap());
    let tk_gpu = Arc::new(TrackedTensor::from_var(&vk_gpu).unwrap());
    let out_gpu = tx_gpu
        .conv_transpose1d(&tk_gpu, padding, stride, dilation, groups, output_padding)
        .unwrap();
    let loss_gpu = sum_all(&out_gpu.sqr().unwrap());
    let grads_gpu = backward(&loss_gpu).unwrap();

    let tol = 1e-4;
    assert_grads_close(
        grads_cpu.get(&vx_cpu).unwrap(),
        grads_gpu.get(&vx_gpu).unwrap(),
        tol,
        "conv_transpose1d_stride2_input_grad",
    );
    assert_grads_close(
        grads_cpu.get(&vk_cpu).unwrap(),
        grads_gpu.get(&vk_gpu).unwrap(),
        tol,
        "conv_transpose1d_stride2_kernel_grad",
    );
}

// ── Conv1d training loop on GPU ─────────────────────────────────────

/// Full Conv1d + Adam training loop on GPU: verify loss decreases.
#[test]
fn test_gpu_conv1d_training_loss_decreases() {
    if !init_gpu() {
        eprintln!("Metal GPU not available, skipping test");
        return;
    }
    let gpu = Device::metal();
    let (batch, in_ch, in_len, out_ch, k_size) = (4, 2, 8, 3, 3);

    let x_data = seeded_tensor(&[batch, in_ch, in_len], 41, &gpu);
    let target_data = seeded_tensor(&[batch, out_ch], 43, &gpu);

    let kernel = Var::new(seeded_tensor(&[out_ch, in_ch, k_size], 47, &gpu));
    let bias = Var::new(DynTensor::zeros(&[1, out_ch, 1], DType::F32, &gpu).unwrap());

    let mut config = AdamConfig::default();
    config.lr = 0.01;
    config.weight_decay = 0.0;
    let mut adam = AdamW::new(vec![kernel.clone(), bias.clone()], config).unwrap();

    let mut losses = Vec::new();
    for _ in 0..10 {
        let tk = Arc::new(TrackedTensor::from_var(&kernel).unwrap());
        let tb = Arc::new(TrackedTensor::from_var(&bias).unwrap());
        let tx = Arc::new(TrackedTensor::from_tensor(x_data.clone()));

        let conv_out = tx.conv1d(&tk, 1, 1, 1, 1).unwrap();
        let conv_out = conv_out.add(&tb).unwrap();
        // Mean over spatial dim, then flatten to [batch, out_ch]
        let pred = conv_out.mean_keepdim(2).unwrap();
        let pred = pred.reshape(&[batch, out_ch]).unwrap();

        let tt = Arc::new(TrackedTensor::from_tensor(target_data.clone()));
        let diff = pred.sub(&tt).unwrap();
        let loss = sum_all(&diff.sqr().unwrap());
        let loss_val = loss.tensor().to_scalar::<f32>().unwrap();
        assert!(loss_val.is_finite(), "loss is NaN/Inf: {loss_val}");
        losses.push(loss_val);
        adam.backward_step(&loss).unwrap();
    }

    let first = losses[0];
    let last = *losses.last().unwrap();
    assert!(
        last < first,
        "GPU Conv1d training: loss should decrease: first={first}, last={last}"
    );
}

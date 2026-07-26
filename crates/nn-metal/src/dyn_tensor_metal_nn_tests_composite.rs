#![allow(deprecated)]
// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Composite GPU nn layer tests — LSTM, InstanceNorm, Embedding, multi-layer
//! model forward passes. Split from dyn_tensor_metal_nn_tests.rs (#1115).

use nn_core::dyn_tensor::DynTensor;
use nn_core::layers::{Embedding, InstanceNorm, Linear, Lstm};
use nn_core::{DType, Device};

use crate::test_common::{assert_close, assert_gpu_matches_cpu, init};

// -- #1077 AC5-AC7: LSTM, InstanceNorm, Embedding GPU forward tests -----------
//
// These verify that the Device::Cpu fix (eb326337) actually works — nn layers
// create intermediates on the input's device, not hardcoded Device::Cpu.

#[test]
fn test_lstm_forward_gpu_no_initial_state() {
    // AC5: LSTM zero-state created on input device (was Device::Cpu before fix).
    // LSTM forward uses matmul + sigmoid + tanh + mul + add — all have GPU paths.
    init();

    let input_size = 4;
    let hidden_size = 3;
    let batch = 2;

    // Build LSTM weights on GPU — small sizes for fast test
    let w_ih = DynTensor::new(
        &vec![0.1f32; 4 * hidden_size * input_size],
        &[4 * hidden_size, input_size],
        &Device::metal(),
    )
    .unwrap();
    let w_hh = DynTensor::new(
        &vec![0.01f32; 4 * hidden_size * hidden_size],
        &[4 * hidden_size, hidden_size],
        &Device::metal(),
    )
    .unwrap();
    let lstm = Lstm::new(w_ih, w_hh, None, None, hidden_size).unwrap();

    let input = DynTensor::new(
        &vec![1.0f32; batch * input_size],
        &[batch, input_size],
        &Device::metal(),
    )
    .unwrap();
    assert_eq!(input.device(), Device::metal());

    // Forward without initial state — the bug was that zero-state was on CPU
    let (output, state) = lstm.forward(&input, None).unwrap();
    assert_eq!(
        output.device(),
        Device::metal(),
        "LSTM output should be on GPU"
    );
    assert_eq!(
        state.h.device(),
        Device::metal(),
        "LSTM h output should be on GPU"
    );
    assert_eq!(
        state.c.device(),
        Device::metal(),
        "LSTM c output should be on GPU"
    );
    assert_eq!(state.h.dims(), &[batch, hidden_size]);
    assert_eq!(state.c.dims(), &[batch, hidden_size]);

    // Verify values are finite (not NaN from device mismatch)
    let h_vals = state
        .h
        .to_device(&Device::Cpu)
        .unwrap()
        .to_flat_vec::<f32>()
        .unwrap();
    for (i, v) in h_vals.iter().enumerate() {
        assert!(v.is_finite(), "h[{i}] is not finite: {v}");
    }
}

#[test]
fn test_instance_norm_forward_gpu() {
    // AC6: InstanceNorm eps tensor created on input device (was Device::Cpu before fix).
    // Uses mean_keepdim + broadcast_sub + sqr + broadcast_add + sqrt + recip + broadcast_mul.
    assert_gpu_matches_cpu(
        |dev| {
            let layer = InstanceNorm::new(1e-5).unwrap();
            // InstanceNorm requires rank >= 3: [B, C, T]
            let x = DynTensor::new(
                &[1.0, 2.0, 3.0, 4.0, 10.0, 20.0, 30.0, 40.0],
                &[1, 2, 4],
                dev,
            )
            .unwrap();
            (Box::new(layer), x)
        },
        1e-4,
        "instance_norm_gpu",
    );
}

#[test]
fn test_embedding_forward_ids_gpu() {
    // AC7: Embedding output on weight device (was always CPU before fix).
    // Embedding::forward_ids reads weights via to_flat_vec_f32 (CPU roundtrip)
    // but now creates the output tensor on weight.device().
    init();

    let vocab_size = 5;
    let embed_dim = 3;
    // Row i has values [i*10+1, i*10+2, i*10+3]
    let weight_data: Vec<f32> = (0..vocab_size)
        .flat_map(|i| {
            let base = (i * 10) as f32;
            vec![base + 1.0, base + 2.0, base + 3.0]
        })
        .collect();

    let weight = DynTensor::new(&weight_data, &[vocab_size, embed_dim], &Device::metal()).unwrap();
    assert_eq!(weight.device(), Device::metal());

    let emb = Embedding::new(weight).unwrap();
    let out = emb.forward_ids(&[0, 2, 4]).unwrap();

    // Output should be on GPU (same as weight)
    assert_eq!(
        out.device(),
        Device::metal(),
        "Embedding output should be on weight device (GPU)"
    );
    assert_eq!(out.dims(), &[3, embed_dim]);

    // Verify correct values: rows 0, 2, 4
    let vals = out
        .to_device(&Device::Cpu)
        .unwrap()
        .to_flat_vec::<f32>()
        .unwrap();
    assert_close(
        &vals,
        &[1.0, 2.0, 3.0, 21.0, 22.0, 23.0, 41.0, 42.0, 43.0],
        1e-6,
        "embedding_ids_gpu",
    );
}

// -- #1022 AC6: Multi-layer DynTensor model forward on GPU --------------------
//
// Verifies that composing multiple nn layers into a model-like forward pass
// works entirely on GPU via the DynTensor GpuBackend. This is the integration
// test that proves DynTensor Metal Phase 2 enables model-level GPU inference.

/// Build a 3-layer MLP: Linear(4->8) -> ReLU -> LayerNorm(8) -> Linear(8->3).
fn build_mlp(dev: &Device) -> (nn_core::layers::Sequential, DynTensor) {
    use nn_core::layers::{LayerNorm, Sequential};

    #[rustfmt::skip]
    let w1_data = [
        0.1, 0.2, -0.1, 0.3,  0.4, -0.2, 0.1, -0.3,
       -0.1, 0.5,  0.2, 0.1,  0.3, -0.4, 0.2,  0.1,
        0.2, 0.1,  0.3,-0.1, -0.2,  0.4,-0.1,  0.2,
       -0.3, 0.1,  0.2, 0.4,  0.1, -0.2, 0.3, -0.1,
    ];
    let w1 = DynTensor::new(&w1_data, &[8, 4], dev).unwrap();
    let b1 = DynTensor::new(&[0.1, -0.1, 0.2, 0.0, -0.2, 0.1, 0.0, 0.3], &[8], dev).unwrap();

    let ln_w = DynTensor::ones(&[8], DType::F32, dev).unwrap();
    let ln_b = DynTensor::zeros(&[8], DType::F32, dev).unwrap();

    #[rustfmt::skip]
    let w2_data = [
        0.1, 0.2,-0.1, 0.3, 0.1,-0.2, 0.4, 0.1,
       -0.3, 0.1, 0.2, 0.2,-0.1, 0.3, 0.1,-0.2,
        0.2, 0.1, 0.3,-0.1,-0.2, 0.4,-0.1, 0.2,
    ];
    let w2 = DynTensor::new(&w2_data, &[3, 8], dev).unwrap();

    let mut model = Sequential::new();
    model.add(Linear::new(w1, Some(b1)).unwrap());
    model.add_fn(DynTensor::relu);
    model.add(LayerNorm::new(ln_w, ln_b, 1e-5).unwrap());
    model.add(Linear::new(w2, None).unwrap());

    let x = DynTensor::new(&[1.0, 2.0, 3.0, 4.0, -1.0, 0.5, 2.0, -0.5], &[2, 4], dev).unwrap();
    (model, x)
}

#[test]
fn test_mlp_forward_gpu() {
    // Linear(4->8) -> ReLU -> LayerNorm(8) -> Linear(8->3) on GPU.
    use nn_core::layers::Module;
    init();

    let (cpu_model, cpu_x) = build_mlp(&Device::Cpu);
    let cpu_vals = cpu_model
        .forward(&cpu_x)
        .unwrap()
        .to_flat_vec::<f32>()
        .unwrap();

    let (gpu_model, gpu_x) = build_mlp(&Device::metal());
    assert_eq!(gpu_x.device(), Device::metal());
    let gpu_out = gpu_model.forward(&gpu_x).unwrap();

    assert_eq!(
        gpu_out.device(),
        Device::metal(),
        "MLP output should stay on GPU"
    );
    assert_eq!(gpu_out.dims(), &[2, 3], "MLP output shape");

    let gpu_vals = gpu_out
        .to_device(&Device::Cpu)
        .unwrap()
        .to_flat_vec::<f32>()
        .unwrap();
    assert_close(&gpu_vals, &cpu_vals, 1e-3, "mlp_forward_gpu");
}

/// Build LSTM(4->6) + Linear(6->2) on given device.
fn build_lstm_mlp(dev: &Device) -> (Lstm, Linear, DynTensor) {
    let (input_size, hidden_size, output_size) = (4, 6, 2);
    let w_ih = DynTensor::new(
        &vec![0.05f32; 4 * hidden_size * input_size],
        &[4 * hidden_size, input_size],
        dev,
    )
    .unwrap();
    let w_hh = DynTensor::new(
        &vec![0.02f32; 4 * hidden_size * hidden_size],
        &[4 * hidden_size, hidden_size],
        dev,
    )
    .unwrap();
    let lstm = Lstm::new(w_ih, w_hh, None, None, hidden_size).unwrap();

    let w_out = DynTensor::new(
        &vec![0.1f32; output_size * hidden_size],
        &[output_size, hidden_size],
        dev,
    )
    .unwrap();
    let b_out = DynTensor::new(&[0.5, -0.5], &[output_size], dev).unwrap();
    let linear_out = Linear::new(w_out, Some(b_out)).unwrap();

    let x = DynTensor::new(
        &[1.0, 0.5, -0.5, 2.0, -1.0, 1.5, 0.0, 0.5],
        &[2, input_size],
        dev,
    )
    .unwrap();
    (lstm, linear_out, x)
}

#[test]
fn test_lstm_mlp_forward_gpu() {
    // LSTM(4->6) -> Linear(6->2): sequence model forward pass on GPU.
    use nn_core::layers::Module;
    init();

    let (cpu_lstm, cpu_linear, cpu_x) = build_lstm_mlp(&Device::Cpu);
    let (cpu_h, _cpu_state) = cpu_lstm.forward(&cpu_x, None).unwrap();
    let cpu_vals = cpu_linear
        .forward(&cpu_h)
        .unwrap()
        .to_flat_vec::<f32>()
        .unwrap();

    let (gpu_lstm, gpu_linear, gpu_x) = build_lstm_mlp(&Device::metal());
    assert_eq!(gpu_x.device(), Device::metal());
    let (gpu_h, _gpu_state) = gpu_lstm.forward(&gpu_x, None).unwrap();
    assert_eq!(gpu_h.device(), Device::metal());
    let gpu_out = gpu_linear.forward(&gpu_h).unwrap();

    assert_eq!(
        gpu_out.device(),
        Device::metal(),
        "LSTM+Linear output should stay on GPU"
    );
    assert_eq!(gpu_out.dims(), &[2, 2]);

    let gpu_vals = gpu_out
        .to_device(&Device::Cpu)
        .unwrap()
        .to_flat_vec::<f32>()
        .unwrap();
    assert_close(&gpu_vals, &cpu_vals, 1e-4, "lstm_mlp_forward_gpu");
}

// -- #2040: Fused GPU InstanceNorm kernel parity tests -------------------------

#[test]
fn test_instance_norm_fused_gpu_batched() {
    // Multi-batch, multi-channel: [B=2, C=3, T=4] — exercises the fused kernel
    // (reshape [B,C,*spatial] → [B*C, spatial_flat], normalize, reshape back).
    assert_gpu_matches_cpu(
        |dev| {
            let layer = InstanceNorm::new(1e-5).unwrap();
            let data: Vec<f32> = (0..24).map(|i| (i as f32) * 0.1 + 0.5).collect();
            let x = DynTensor::new(&data, &[2, 3, 4], dev).unwrap();
            (Box::new(layer), x)
        },
        1e-4,
        "instance_norm_fused_batched",
    );
}

#[test]
fn test_instance_norm_fused_gpu_large_spatial() {
    // Larger spatial dim: [B=1, C=2, T=64] — ensures fused kernel handles
    // non-trivial reductions correctly.
    assert_gpu_matches_cpu(
        |dev| {
            let layer = InstanceNorm::new(1e-5).unwrap();
            let data: Vec<f32> = (0..128).map(|i| ((i as f32) * 0.37).sin()).collect();
            let x = DynTensor::new(&data, &[1, 2, 64], dev).unwrap();
            (Box::new(layer), x)
        },
        1e-4,
        "instance_norm_fused_large_spatial",
    );
}

#[test]
fn test_instance_norm_fused_gpu_near_zero_variance() {
    // Near-constant input per channel: variance ≈ 0. The fused kernel uses
    // rsqrt(var + eps) which handles this cleanly; the decomposed path
    // uses recip() which can produce Inf on near-zero denominators.
    assert_gpu_matches_cpu(
        |dev| {
            let layer = InstanceNorm::new(1e-5).unwrap();
            // Channel 0: nearly constant 5.0, Channel 1: nearly constant -3.0
            let x = DynTensor::new(
                &[5.0, 5.0, 5.0001, 5.0, -3.0, -3.0, -3.0001, -3.0],
                &[1, 2, 4],
                dev,
            )
            .unwrap();
            (Box::new(layer), x)
        },
        1e-3,
        "instance_norm_fused_near_zero_var",
    );
}

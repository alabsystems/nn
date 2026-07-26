// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Full model parity tests: CPU vs Metal.
//!
//! Tests composed model blocks (MLP, attention) on both backends to ensure
//! that multi-layer compositions produce identical results.

use super::test_utils::{assert_gpu_cpu_close, gpu_init};
use nn_core::dyn_tensor::DynTensor;
use nn_core::layers::{LayerNorm, Linear, Module};
use nn_core::test_prng::rand_f32_vec;
use nn_core::{DType, Device};

const TOL: f32 = 5e-4;

fn init() {
    gpu_init();
}

/// Build a simple 2-layer MLP: Linear(in, hidden) -> relu -> Linear(hidden, out)
/// Returns the output tensor.
fn run_mlp(
    x_data: &[f32],
    w1_data: &[f32],
    b1_data: &[f32],
    w2_data: &[f32],
    b2_data: &[f32],
    batch: usize,
    in_feat: usize,
    hidden: usize,
    out_feat: usize,
    device: &Device,
) -> DynTensor {
    let x = DynTensor::new(x_data, &[batch, in_feat], device).unwrap();
    let w1 = DynTensor::new(w1_data, &[hidden, in_feat], device).unwrap();
    let b1 = DynTensor::new(b1_data, &[hidden], device).unwrap();
    let w2 = DynTensor::new(w2_data, &[out_feat, hidden], device).unwrap();
    let b2 = DynTensor::new(b2_data, &[out_feat], device).unwrap();

    let linear1 = Linear::new(w1, Some(b1)).unwrap();
    let linear2 = Linear::new(w2, Some(b2)).unwrap();

    let h = linear1.forward(&x).unwrap();
    let h = h.relu().unwrap();
    
    linear2.forward(&h).unwrap()
}

#[test]
fn test_parity_simple_mlp() {
    init();
    let batch = 4;
    let in_feat = 32;
    let hidden = 64;
    let out_feat = 16;

    let x_data = rand_f32_vec(200, batch * in_feat, -1.0, 1.0);
    let w1_data = rand_f32_vec(201, hidden * in_feat, -0.3, 0.3);
    let b1_data = rand_f32_vec(202, hidden, -0.1, 0.1);
    let w2_data = rand_f32_vec(203, out_feat * hidden, -0.3, 0.3);
    let b2_data = rand_f32_vec(204, out_feat, -0.1, 0.1);

    let cpu_out = run_mlp(
        &x_data,
        &w1_data,
        &b1_data,
        &w2_data,
        &b2_data,
        batch,
        in_feat,
        hidden,
        out_feat,
        &Device::Cpu,
    );
    let gpu_out = run_mlp(
        &x_data,
        &w1_data,
        &b1_data,
        &w2_data,
        &b2_data,
        batch,
        in_feat,
        hidden,
        out_feat,
        &Device::metal(),
    );

    assert_eq!(gpu_out.device(), Device::metal());
    assert_eq!(gpu_out.dims(), &[batch, out_feat]);
    assert_eq!(gpu_out.dims(), cpu_out.dims());
    assert_gpu_cpu_close(&gpu_out, &cpu_out, TOL, "simple_mlp");
}

// -- 3-layer MLP with silu activation and LayerNorm -----------------------

#[test]
fn test_parity_mlp_with_layernorm() {
    init();
    let batch = 2;
    let in_feat = 16;
    let hidden = 32;
    let out_feat = 8;

    let x_data = rand_f32_vec(210, batch * in_feat, -1.0, 1.0);
    let w1_data = rand_f32_vec(211, hidden * in_feat, -0.3, 0.3);
    let b1_data = rand_f32_vec(212, hidden, -0.1, 0.1);
    let ln_w_data = rand_f32_vec(213, hidden, 0.8, 1.2);
    let ln_b_data = rand_f32_vec(214, hidden, -0.05, 0.05);
    let w2_data = rand_f32_vec(215, out_feat * hidden, -0.3, 0.3);
    let b2_data = rand_f32_vec(216, out_feat, -0.1, 0.1);

    let run = |device: &Device| -> DynTensor {
        let x = DynTensor::new(&x_data, &[batch, in_feat], device).unwrap();
        let w1 = DynTensor::new(&w1_data, &[hidden, in_feat], device).unwrap();
        let b1 = DynTensor::new(&b1_data, &[hidden], device).unwrap();
        let ln_w = DynTensor::new(&ln_w_data, &[hidden], device).unwrap();
        let ln_b = DynTensor::new(&ln_b_data, &[hidden], device).unwrap();
        let w2 = DynTensor::new(&w2_data, &[out_feat, hidden], device).unwrap();
        let b2 = DynTensor::new(&b2_data, &[out_feat], device).unwrap();

        let linear1 = Linear::new(w1, Some(b1)).unwrap();
        let ln = LayerNorm::new(ln_w, ln_b, 1e-5).unwrap();
        let linear2 = Linear::new(w2, Some(b2)).unwrap();

        let h = linear1.forward(&x).unwrap();
        let h = h.silu().unwrap();
        let h = ln.forward(&h).unwrap();
        
        linear2.forward(&h).unwrap()
    };

    let cpu_out = run(&Device::Cpu);
    let gpu_out = run(&Device::metal());

    assert_eq!(gpu_out.device(), Device::metal());
    assert_eq!(gpu_out.dims(), &[batch, out_feat]);
    assert_gpu_cpu_close(&gpu_out, &cpu_out, TOL, "mlp_with_layernorm");
}

// -- Attention block: Q*K^T/sqrt(d) -> softmax -> V -----------------------

#[test]
fn test_parity_attention_block() {
    init();
    let batch = 2;
    let seq = 8;
    let d_model = 16;

    let q_data = rand_f32_vec(220, batch * seq * d_model, -1.0, 1.0);
    let k_data = rand_f32_vec(221, batch * seq * d_model, -1.0, 1.0);
    let v_data = rand_f32_vec(222, batch * seq * d_model, -1.0, 1.0);

    let run = |device: &Device| -> DynTensor {
        let q = DynTensor::new(&q_data, &[batch, seq, d_model], device).unwrap();
        let k = DynTensor::new(&k_data, &[batch, seq, d_model], device).unwrap();
        let v = DynTensor::new(&v_data, &[batch, seq, d_model], device).unwrap();

        // Attention: softmax(Q @ K^T / sqrt(d)) @ V
        let kt = k.transpose(1, 2).unwrap(); // [batch, d_model, seq]
        let scores = q.matmul(&kt).unwrap(); // [batch, seq, seq]

        let scale = (d_model as f32).sqrt();
        let scale_tensor = DynTensor::full(&[1], f64::from(scale), DType::F32, device).unwrap();
        let scores = scores.broadcast_div(&scale_tensor).unwrap();

        let attn = scores.softmax(2).unwrap(); // softmax over last dim
        let out = attn.matmul(&v).unwrap(); // [batch, seq, d_model]
        out
    };

    let cpu_out = run(&Device::Cpu);
    let gpu_out = run(&Device::metal());

    assert_eq!(gpu_out.device(), Device::metal());
    assert_eq!(gpu_out.dims(), &[batch, seq, d_model]);
    assert_eq!(gpu_out.dims(), cpu_out.dims());
    // Attention involves multiple matmuls + softmax, so tolerance is slightly wider
    assert_gpu_cpu_close(&gpu_out, &cpu_out, 1e-3, "attention_block");
}

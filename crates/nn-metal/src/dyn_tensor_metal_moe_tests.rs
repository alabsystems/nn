// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Tests for fused MoE GPU scatter-gather dispatch (#3547).

use nn_core::dyn_tensor::DynTensor;
use nn_core::{DType, Device, Result};

fn init_metal() {
    crate::MetalBackend::init().ok();
    crate::register_metal_dyn_backend();
}

/// Create a simple expert weight set (gate, up, down) with small dimensions.
fn make_expert_weights(
    model_dim: usize,
    intermediate: usize,
    device: &Device,
    seed: f32,
) -> Result<(DynTensor, DynTensor, DynTensor)> {
    // gate_proj: [intermediate, model_dim]
    let gate_data: Vec<f32> = (0..intermediate * model_dim)
        .map(|i| ((i as f32 + seed) * 0.01).sin() * 0.1)
        .collect();
    let gate = DynTensor::from_vec(gate_data, &[intermediate, model_dim], device)?;

    // up_proj: [intermediate, model_dim]
    let up_data: Vec<f32> = (0..intermediate * model_dim)
        .map(|i| ((i as f32 + seed + 1.0) * 0.01).cos() * 0.1)
        .collect();
    let up = DynTensor::from_vec(up_data, &[intermediate, model_dim], device)?;

    // down_proj: [model_dim, intermediate]
    let down_data: Vec<f32> = (0..model_dim * intermediate)
        .map(|i| ((i as f32 + seed + 2.0) * 0.01).sin() * 0.1)
        .collect();
    let down = DynTensor::from_vec(down_data, &[model_dim, intermediate], device)?;

    Ok((gate, up, down))
}

/// Helper: run MoE scatter-gather on CPU using the per-expert loop.
fn cpu_moe_scatter_gather(
    hidden: &DynTensor,
    indices: &DynTensor,
    weights: &DynTensor,
    expert_gate_weights: &[DynTensor],
    expert_up_weights: &[DynTensor],
    expert_down_weights: &[DynTensor],
    num_experts: usize,
) -> Result<DynTensor> {
    let dims = hidden.dims();
    let n_tokens = dims[0];
    let model_dim = dims[1];
    let k = indices.dims()[1];
    let device = hidden.device();

    let idx_arr = indices.as_cpu_u32()?;
    let wt_arr = weights.to_f32_array()?;

    let mut output = DynTensor::zeros(&[n_tokens, model_dim], DType::F32, &device)?;

    for expert_idx in 0..num_experts {
        let mut token_ids = Vec::new();
        let mut expert_wts = Vec::new();

        for t in 0..n_tokens {
            for s in 0..k {
                let coord = &[t, s];
                let e = idx_arr[ndarray::IxDyn(coord)] as usize;
                if e == expert_idx {
                    token_ids.push(t as u32);
                    expert_wts.push(wt_arr.view()[ndarray::IxDyn(coord)]);
                }
            }
        }

        if token_ids.is_empty() {
            continue;
        }

        let num_routed = token_ids.len();
        let ids = DynTensor::from_vec_u32(token_ids, &[num_routed], &device)?;
        let w_tensor = DynTensor::from_vec(expert_wts, &[num_routed, 1], &device)?;

        let gathered = hidden.index_select(&ids, 0)?;

        // SwiGLU: gate=silu(x@gate^T), up=x@up^T, out=(gate*up)@down^T
        let gate_w_t = expert_gate_weights[expert_idx].transpose(0, 1)?;
        let up_w_t = expert_up_weights[expert_idx].transpose(0, 1)?;
        let down_w_t = expert_down_weights[expert_idx].transpose(0, 1)?;

        let gate = gathered.matmul(&gate_w_t)?.silu()?;
        let up = gathered.matmul(&up_w_t)?;
        let h = gate.broadcast_mul(&up)?;
        let expert_out = h.matmul(&down_w_t)?;

        let weighted = expert_out.broadcast_mul(&w_tensor)?;
        output = output.index_add(0, &ids, &weighted)?;
    }

    Ok(output)
}

#[test]
fn test_moe_gpu_scatter_gather_basic() {
    init_metal();

    let n_tokens = 4;
    let model_dim = 8;
    let intermediate = 16;
    let num_experts = 4;
    let k = 2;

    let cpu = Device::Cpu;
    let gpu = Device::metal();

    // Create input hidden states.
    let hidden_data: Vec<f32> = (0..n_tokens * model_dim)
        .map(|i| (i as f32 * 0.1).sin())
        .collect();
    let hidden_cpu = DynTensor::from_vec(hidden_data, &[n_tokens, model_dim], &cpu).unwrap();
    let hidden_gpu = hidden_cpu.to_device(&gpu).unwrap();

    // Create routing: each token assigned to 2 experts.
    let indices_data: Vec<u32> = vec![0, 1, 1, 2, 2, 3, 3, 0];
    let indices_cpu = DynTensor::from_vec_u32(indices_data, &[n_tokens, k], &cpu).unwrap();
    let indices_gpu = indices_cpu.to_device(&gpu).unwrap();

    // Routing weights (normalized per-token).
    let weights_data: Vec<f32> = vec![0.6, 0.4, 0.5, 0.5, 0.7, 0.3, 0.55, 0.45];
    let weights_cpu = DynTensor::from_vec(weights_data, &[n_tokens, k], &cpu).unwrap();
    let weights_gpu = weights_cpu.to_device(&gpu).unwrap();

    // Create expert weights.
    let mut gate_ws_cpu = Vec::new();
    let mut up_ws_cpu = Vec::new();
    let mut down_ws_cpu = Vec::new();
    let mut gate_ws_gpu = Vec::new();
    let mut up_ws_gpu = Vec::new();
    let mut down_ws_gpu = Vec::new();

    for e in 0..num_experts {
        let (g, u, d) = make_expert_weights(model_dim, intermediate, &cpu, e as f32 * 10.0).unwrap();
        gate_ws_gpu.push(g.to_device(&gpu).unwrap());
        up_ws_gpu.push(u.to_device(&gpu).unwrap());
        down_ws_gpu.push(d.to_device(&gpu).unwrap());
        gate_ws_cpu.push(g);
        up_ws_cpu.push(u);
        down_ws_cpu.push(d);
    }

    // Run on CPU.
    let cpu_result = cpu_moe_scatter_gather(
        &hidden_cpu, &indices_cpu, &weights_cpu,
        &gate_ws_cpu, &up_ws_cpu, &down_ws_cpu,
        num_experts,
    ).unwrap();

    // Run on GPU (should use fused dispatch).
    crate::gpu_scope::flush().unwrap();
    let gpu_result = super::MetalDynBackend::gpu_moe_scatter_gather(
        &hidden_gpu, &indices_gpu, &weights_gpu,
        &gate_ws_gpu, &up_ws_gpu, &down_ws_gpu,
        num_experts,
    );

    assert!(gpu_result.is_some(), "GPU MoE dispatch should not fall back");
    let gpu_result = gpu_result.unwrap().unwrap();

    // Transfer GPU result to CPU for comparison.
    crate::gpu_scope::flush().unwrap();
    let gpu_on_cpu = gpu_result.to_device(&cpu).unwrap();

    let cpu_flat = cpu_result.to_flat_vec::<f32>().unwrap();
    let gpu_flat = gpu_on_cpu.to_flat_vec::<f32>().unwrap();

    assert_eq!(cpu_flat.len(), gpu_flat.len(), "output length mismatch");

    for (i, (c, g)) in cpu_flat.iter().zip(gpu_flat.iter()).enumerate() {
        let diff = (c - g).abs();
        assert!(
            diff < 1e-4,
            "mismatch at index {i}: cpu={c}, gpu={g}, diff={diff}"
        );
    }
}

#[test]
fn test_moe_gpu_scatter_gather_empty_expert() {
    init_metal();

    let n_tokens = 2;
    let model_dim = 4;
    let intermediate = 8;
    let num_experts = 4;
    let k = 1;

    let cpu = Device::Cpu;
    let gpu = Device::metal();

    let hidden_data: Vec<f32> = (0..n_tokens * model_dim)
        .map(|i| (i as f32 * 0.2).cos())
        .collect();
    let hidden_gpu = DynTensor::from_vec(hidden_data.clone(), &[n_tokens, model_dim], &gpu).unwrap();
    let hidden_cpu = DynTensor::from_vec(hidden_data, &[n_tokens, model_dim], &cpu).unwrap();

    // Both tokens go to expert 0 — experts 1, 2, 3 get no tokens.
    let indices_data: Vec<u32> = vec![0, 0];
    let indices_gpu = DynTensor::from_vec_u32(indices_data.clone(), &[n_tokens, k], &gpu).unwrap();
    let indices_cpu = DynTensor::from_vec_u32(indices_data, &[n_tokens, k], &cpu).unwrap();

    let weights_data: Vec<f32> = vec![1.0, 1.0];
    let weights_gpu = DynTensor::from_vec(weights_data.clone(), &[n_tokens, k], &gpu).unwrap();
    let weights_cpu = DynTensor::from_vec(weights_data, &[n_tokens, k], &cpu).unwrap();

    let mut gate_ws_cpu = Vec::new();
    let mut up_ws_cpu = Vec::new();
    let mut down_ws_cpu = Vec::new();
    let mut gate_ws_gpu = Vec::new();
    let mut up_ws_gpu = Vec::new();
    let mut down_ws_gpu = Vec::new();

    for e in 0..num_experts {
        let (g, u, d) = make_expert_weights(model_dim, intermediate, &cpu, e as f32 * 5.0).unwrap();
        gate_ws_gpu.push(g.to_device(&gpu).unwrap());
        up_ws_gpu.push(u.to_device(&gpu).unwrap());
        down_ws_gpu.push(d.to_device(&gpu).unwrap());
        gate_ws_cpu.push(g);
        up_ws_cpu.push(u);
        down_ws_cpu.push(d);
    }

    let cpu_result = cpu_moe_scatter_gather(
        &hidden_cpu, &indices_cpu, &weights_cpu,
        &gate_ws_cpu, &up_ws_cpu, &down_ws_cpu,
        num_experts,
    ).unwrap();

    crate::gpu_scope::flush().unwrap();
    let gpu_result = super::MetalDynBackend::gpu_moe_scatter_gather(
        &hidden_gpu, &indices_gpu, &weights_gpu,
        &gate_ws_gpu, &up_ws_gpu, &down_ws_gpu,
        num_experts,
    ).unwrap().unwrap();

    crate::gpu_scope::flush().unwrap();
    let gpu_on_cpu = gpu_result.to_device(&cpu).unwrap();

    let cpu_flat = cpu_result.to_flat_vec::<f32>().unwrap();
    let gpu_flat = gpu_on_cpu.to_flat_vec::<f32>().unwrap();

    for (i, (c, g)) in cpu_flat.iter().zip(gpu_flat.iter()).enumerate() {
        let diff = (c - g).abs();
        assert!(
            diff < 1e-4,
            "mismatch at index {i}: cpu={c}, gpu={g}, diff={diff}"
        );
    }
}

#[test]
fn test_moe_gpu_non_f32_returns_none() {
    init_metal();

    let gpu = Device::metal();
    let hidden = DynTensor::zeros(&[2, 4], DType::BF16, &gpu).unwrap();
    let indices = DynTensor::from_vec_u32(vec![0, 1], &[2, 1], &gpu).unwrap();
    let weights = DynTensor::from_vec(vec![1.0, 1.0], &[2, 1], &gpu).unwrap();

    let (g, u, d) = make_expert_weights(4, 8, &gpu, 0.0).unwrap();

    let result = super::MetalDynBackend::gpu_moe_scatter_gather(
        &hidden, &indices, &weights,
        std::slice::from_ref(&g),
        std::slice::from_ref(&u),
        std::slice::from_ref(&d),
        1,
    );

    assert!(result.is_none(), "BF16 should return None for CPU fallback");
}

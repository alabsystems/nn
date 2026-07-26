// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

use crate::dyn_tensor::DynTensor;
use crate::layers::{Linear, Module};
use crate::var_builder::VarBuilder;
use crate::{DType, Device, TensorError};
use std::collections::HashMap;

use super::{MoeLayer, MoeRouter, SwiGluExpert};

// -- MoeRouter tests ----------------------------------------------------------

fn make_router(num_experts: usize, top_k: usize, model_dim: usize) -> MoeRouter {
    // Identity-ish gate: weight = eye-like so expert i gets highest logit
    // when input feature i is largest.
    let mut gate_data = vec![0.0f32; num_experts * model_dim];
    for e in 0..num_experts.min(model_dim) {
        gate_data[e * model_dim + e] = 1.0;
    }
    let gate_weight =
        DynTensor::from_vec(gate_data, &[num_experts, model_dim], &Device::Cpu).unwrap();
    let gate = Linear::new(gate_weight, None).unwrap();
    MoeRouter::new(gate, num_experts, top_k).unwrap()
}

#[test]
fn test_router_basic_routing() {
    let router = make_router(4, 2, 4);
    // Input: 1 token with feature 0 largest → should route to expert 0 first
    let x = DynTensor::from_vec(vec![10.0, 1.0, 0.5, 0.1], &[1, 4], &Device::Cpu).unwrap();
    let out = router.forward(&x).unwrap();
    assert_eq!(out.weights.dims(), &[1, 2]);
    assert_eq!(out.indices.dims(), &[1, 2]);
    assert_eq!(out.indices.dtype(), DType::U32);
    // Top expert should be 0 (highest logit)
    let idx = out.indices.as_cpu_u32().unwrap();
    assert_eq!(idx[ndarray::IxDyn(&[0, 0])], 0);
}

#[test]
fn test_router_weight_normalization() {
    let router = make_router(4, 2, 4);
    let x = DynTensor::from_vec(vec![3.0, 2.0, 1.0, 0.0], &[1, 4], &Device::Cpu).unwrap();
    let out = router.forward(&x).unwrap();
    let w = out.weights.as_cpu_f32().unwrap();
    // Sum of top-k weights should be ~1.0 (renormalized)
    let sum: f32 = w.iter().sum();
    assert!(
        (sum - 1.0).abs() < 1e-5,
        "weights sum = {sum}, expected ~1.0"
    );
}

#[test]
fn test_router_batch() {
    let router = make_router(4, 2, 4);
    let x = DynTensor::from_vec(
        vec![
            10.0, 1.0, 0.5, 0.1, // token 0 → expert 0
            0.1, 10.0, 0.5, 0.1, // token 1 → expert 1
        ],
        &[2, 4],
        &Device::Cpu,
    )
    .unwrap();
    let out = router.forward(&x).unwrap();
    assert_eq!(out.indices.dims(), &[2, 2]);
    let idx = out.indices.as_cpu_u32().unwrap();
    assert_eq!(idx[ndarray::IxDyn(&[0, 0])], 0);
    assert_eq!(idx[ndarray::IxDyn(&[1, 0])], 1);
}

#[test]
fn test_router_invalid_topk_zero() {
    let gate_weight = DynTensor::from_vec(vec![1.0; 8], &[4, 2], &Device::Cpu).unwrap();
    let gate = Linear::new(gate_weight, None).unwrap();
    assert!(MoeRouter::new(gate, 4, 0).is_err());
}

#[test]
fn test_router_invalid_topk_exceeds_experts() {
    let gate_weight = DynTensor::from_vec(vec![1.0; 8], &[4, 2], &Device::Cpu).unwrap();
    let gate = Linear::new(gate_weight, None).unwrap();
    // top_k=5 > num_experts=4 should fail
    let result = MoeRouter::new(gate, 4, 5);
    assert!(result.is_err());
    let err_msg = result.unwrap_err().to_string();
    assert!(
        err_msg.contains("top_k"),
        "Expected error mentioning top_k, got: {err_msg}"
    );
}

#[test]
fn test_router_topk_equals_experts_valid() {
    // top_k == num_experts should be valid (all experts selected)
    let gate_weight = DynTensor::from_vec(vec![1.0; 8], &[4, 2], &Device::Cpu).unwrap();
    let gate = Linear::new(gate_weight, None).unwrap();
    assert!(MoeRouter::new(gate, 4, 4).is_ok());
}

// -- SwiGluExpert tests -------------------------------------------------------

fn make_tiny_expert(dim: usize, ff_dim: usize) -> SwiGluExpert {
    // Small random-ish weights for testing
    let gate_w =
        DynTensor::from_vec(vec![0.1; ff_dim * dim], &[ff_dim, dim], &Device::Cpu).unwrap();
    let up_w = DynTensor::from_vec(vec![0.1; ff_dim * dim], &[ff_dim, dim], &Device::Cpu).unwrap();
    let down_w =
        DynTensor::from_vec(vec![0.1; dim * ff_dim], &[dim, ff_dim], &Device::Cpu).unwrap();
    SwiGluExpert::new(
        Linear::new(gate_w, None).unwrap(),
        Linear::new(up_w, None).unwrap(),
        Linear::new(down_w, None).unwrap(),
    )
    .unwrap()
}

#[test]
fn test_expert_forward_shape() {
    let expert = make_tiny_expert(4, 8);
    let x = DynTensor::from_vec(vec![1.0; 4], &[1, 4], &Device::Cpu).unwrap();
    let out = expert.forward(&x).unwrap();
    assert_eq!(out.dims(), &[1, 4]);
}

// -- MoeLayer tests -----------------------------------------------------------

fn make_moe_layer(
    model_dim: usize,
    ff_dim: usize,
    num_experts: usize,
    top_k: usize,
    with_shared: bool,
) -> MoeLayer {
    let router = make_router(num_experts, top_k, model_dim);
    let experts: Vec<SwiGluExpert> = (0..num_experts)
        .map(|_| make_tiny_expert(model_dim, ff_dim))
        .collect();
    let shared = if with_shared {
        Some(make_tiny_expert(model_dim, ff_dim))
    } else {
        None
    };
    MoeLayer::new(router, experts, shared).unwrap()
}

#[test]
fn test_moe_forward_shape() {
    let moe = make_moe_layer(4, 8, 4, 2, false);
    let x = DynTensor::from_vec(vec![1.0; 2 * 4], &[2, 4], &Device::Cpu).unwrap();
    let out = moe.forward(&x).unwrap();
    assert_eq!(out.dims(), &[2, 4]);
}

#[test]
fn test_moe_with_shared_expert() {
    let moe = make_moe_layer(4, 8, 4, 2, true);
    let x = DynTensor::from_vec(vec![1.0; 2 * 4], &[2, 4], &Device::Cpu).unwrap();
    let out = moe.forward(&x).unwrap();
    assert_eq!(out.dims(), &[2, 4]);
    // With shared expert, output should be non-zero
    let arr = out.as_cpu_f32().unwrap();
    let any_nonzero = arr.iter().any(|&v| v.abs() > 1e-10);
    assert!(
        any_nonzero,
        "MoE output with shared expert should be non-zero"
    );
}

#[test]
fn test_moe_3d_input() {
    // [B, T, D] input
    let moe = make_moe_layer(4, 8, 4, 2, false);
    let x = DynTensor::from_vec(vec![1.0; 2 * 3 * 4], &[2, 3, 4], &Device::Cpu).unwrap();
    let out = moe.forward(&x).unwrap();
    assert_eq!(out.dims(), &[2, 3, 4]);
}

#[test]
fn test_moe_expert_count_mismatch() {
    let router = make_router(4, 2, 4);
    let experts: Vec<SwiGluExpert> = (0..3) // wrong: 3 instead of 4
        .map(|_| make_tiny_expert(4, 8))
        .collect();
    assert!(MoeLayer::new(router, experts, None).is_err());
}

#[test]
fn test_moe_output_finiteness() {
    let moe = make_moe_layer(4, 8, 4, 2, true);
    let x = DynTensor::from_vec(
        vec![1.0, 2.0, 3.0, 4.0, 0.5, 1.5, 2.5, 3.5],
        &[2, 4],
        &Device::Cpu,
    )
    .unwrap();
    let out = moe.forward(&x).unwrap();
    let arr = out.as_cpu_f32().unwrap();
    assert!(
        arr.iter().all(|v| v.is_finite()),
        "MoE output contains NaN/Inf"
    );
}

#[test]
fn test_router_nan_input_returns_error() {
    // NaN input → softmax produces NaN → check_output_finite catches it.
    let router = make_router(4, 2, 4);
    let x = DynTensor::from_vec(vec![f32::NAN, 1.0, 0.5, 0.1], &[1, 4], &Device::Cpu).unwrap();
    let result = router.forward(&x);
    match result {
        Err(TensorError::NonFiniteData { name, count }) => {
            assert!(
                name.contains("MoeRouter") || name.contains("softmax"),
                "expected MoeRouter or softmax in name, got {name}"
            );
            assert!(count > 0, "expected non-zero count");
        }
        Err(other) => panic!("expected NonFiniteData, got {other:?}"),
        Ok(_) => panic!("expected error from NaN input"),
    }
}

#[test]
fn test_moe_layer_nan_input_returns_error() {
    // NaN input propagates through routing + expert dispatch → NonFiniteData.
    let moe = make_moe_layer(4, 8, 4, 2, false);
    let x = DynTensor::from_vec(vec![f32::NAN, 1.0, 0.5, 0.1], &[1, 4], &Device::Cpu).unwrap();
    let result = moe.forward(&x);
    // Should fail at either MoeRouter or MoeLayer check
    assert!(result.is_err());
}

// -- VarBuilder load path tests -----------------------------------------------

/// Build tensor entries for a single SwiGluExpert at a given prefix.
fn expert_tensors(
    prefix: &str,
    model_dim: usize,
    ff_dim: usize,
    device: &Device,
) -> Vec<(String, DynTensor)> {
    vec![
        (
            format!("{prefix}.gate_proj.weight"),
            DynTensor::ones(&[ff_dim, model_dim], DType::F32, device).unwrap(),
        ),
        (
            format!("{prefix}.up_proj.weight"),
            DynTensor::ones(&[ff_dim, model_dim], DType::F32, device).unwrap(),
        ),
        (
            format!("{prefix}.down_proj.weight"),
            DynTensor::ones(&[model_dim, ff_dim], DType::F32, device).unwrap(),
        ),
    ]
}

#[test]
fn test_moe_load_varbuilder_no_shared() {
    let model_dim = 4;
    let ff_dim = 8;
    let num_experts = 2;
    let top_k = 1;
    let device = Device::Cpu;

    let mut tensors: HashMap<String, DynTensor> = HashMap::new();
    tensors.insert(
        "gate.weight".into(),
        DynTensor::ones(&[num_experts, model_dim], DType::F32, &device).unwrap(),
    );
    for e in 0..num_experts {
        for (k, v) in expert_tensors(&format!("experts.{e}"), model_dim, ff_dim, &device) {
            tensors.insert(k, v);
        }
    }

    let vb = VarBuilder::from_tensors(tensors, DType::F32, &device);
    let moe = MoeLayer::load(&vb, model_dim, ff_dim, num_experts, top_k, false).unwrap();

    let x = DynTensor::from_vec(vec![1.0; 2 * model_dim], &[2, model_dim], &device).unwrap();
    let out = moe.forward(&x).unwrap();
    assert_eq!(out.dims(), &[2, model_dim]);
    let arr = out.as_cpu_f32().unwrap();
    assert!(arr.iter().all(|v| v.is_finite()), "output should be finite");
}

#[test]
fn test_moe_load_varbuilder_with_shared_expert() {
    let model_dim = 4;
    let ff_dim = 8;
    let num_experts = 2;
    let top_k = 1;
    let device = Device::Cpu;

    let mut tensors: HashMap<String, DynTensor> = HashMap::new();
    tensors.insert(
        "gate.weight".into(),
        DynTensor::ones(&[num_experts, model_dim], DType::F32, &device).unwrap(),
    );
    for e in 0..num_experts {
        for (k, v) in expert_tensors(&format!("experts.{e}"), model_dim, ff_dim, &device) {
            tensors.insert(k, v);
        }
    }
    for (k, v) in expert_tensors("shared_expert", model_dim, ff_dim, &device) {
        tensors.insert(k, v);
    }

    let vb = VarBuilder::from_tensors(tensors, DType::F32, &device);
    let moe = MoeLayer::load(&vb, model_dim, ff_dim, num_experts, top_k, true).unwrap();

    let x = DynTensor::from_vec(vec![1.0; model_dim], &[1, model_dim], &device).unwrap();
    let out = moe.forward(&x).unwrap();
    assert_eq!(out.dims(), &[1, model_dim]);
}

#[test]
fn test_moe_transposed_input() {
    // Create a [D, N] tensor, transpose to [N, D] → non-contiguous view.
    // Exercises reshape-to-contiguous in forward path.
    let moe = make_moe_layer(4, 8, 4, 2, false);
    let data: Vec<f32> = (0..12).map(|i| (i as f32) * 0.1 + 0.5).collect();
    let x_td = DynTensor::from_vec(data, &[4, 3], &Device::Cpu).unwrap();
    let x_nt = x_td.transpose(0, 1).unwrap(); // [3, 4] — non-contiguous
    assert_eq!(x_nt.dims(), &[3, 4]);
    let out = moe.forward(&x_nt).unwrap();
    assert_eq!(out.dims(), &[3, 4]);
    let arr = out.as_cpu_f32().unwrap();
    assert!(
        arr.iter().all(|v| v.is_finite()),
        "MoE output from transposed input contains NaN/Inf"
    );
}

#[test]
fn test_moe_load_varbuilder_zeros_backend() {
    let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
    let moe = MoeLayer::load(&vb, 4, 8, 3, 2, false).unwrap();
    assert_eq!(moe.router().num_experts(), 3);
}

// -- 8 experts, top-2 routing (#3542) -----------------------------------------

#[test]
fn test_moe_8_experts_top2_routing() {
    let model_dim = 16;
    let ff_dim = 32;
    let num_experts = 8;
    let top_k = 2;
    let moe = make_moe_layer(model_dim, ff_dim, num_experts, top_k, false);
    // 4 tokens, each routed to top-2 of 8 experts.
    let x = DynTensor::from_vec(
        (0..4 * model_dim)
            .map(|i| (i as f32) * 0.01 + 0.1)
            .collect::<Vec<_>>(),
        &[4, model_dim],
        &Device::Cpu,
    )
    .unwrap();
    let out = moe.forward(&x).unwrap();
    assert_eq!(out.dims(), &[4, model_dim], "output shape must match input");
    let arr = out.as_cpu_f32().unwrap();
    assert!(
        arr.iter().all(|v| v.is_finite()),
        "MoE 8-expert top-2 output contains NaN/Inf"
    );
    // With 8 experts and top-2, every token should produce non-zero output
    // (experts have non-zero weights, input is non-zero).
    let any_nonzero = arr.iter().any(|&v| v.abs() > 1e-12);
    assert!(
        any_nonzero,
        "expected non-zero output from 8-expert top-2 MoE"
    );
}

#[test]
fn test_moe_8_experts_top2_3d_input() {
    let model_dim = 16;
    let ff_dim = 32;
    let num_experts = 8;
    let top_k = 2;
    let moe = make_moe_layer(model_dim, ff_dim, num_experts, top_k, false);
    // [B=2, T=3, D=16] — batched 3D input.
    let x = DynTensor::from_vec(
        (0..2 * 3 * model_dim)
            .map(|i| (i as f32) * 0.01 + 0.5)
            .collect::<Vec<_>>(),
        &[2, 3, model_dim],
        &Device::Cpu,
    )
    .unwrap();
    let out = moe.forward(&x).unwrap();
    assert_eq!(out.dims(), &[2, 3, model_dim]);
    let arr = out.as_cpu_f32().unwrap();
    assert!(
        arr.iter().all(|v| v.is_finite()),
        "MoE 8-expert top-2 3D output contains NaN/Inf"
    );
}

// Value-correctness tests extracted to moe_tests_value.rs
#[path = "moe_tests_value.rs"]
mod value;

// Routing edge-case tests (AC1-AC5) extracted to moe_tests_routing.rs
#[path = "moe_tests_routing.rs"]
mod routing;

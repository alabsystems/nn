// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! MoE routing edge-case tests (AC1-AC5 from #1448).
//! Extracted from moe_tests.rs for 500-line limit compliance.

use crate::dyn_tensor::DynTensor;
use crate::layers::{Linear, Module};
use crate::{Device, TensorError};

use super::super::{MoeLayer, MoeRouter, SwiGluExpert};

fn make_router(num_experts: usize, top_k: usize, model_dim: usize) -> MoeRouter {
    let mut gate_data = vec![0.0f32; num_experts * model_dim];
    for e in 0..num_experts.min(model_dim) {
        gate_data[e * model_dim + e] = 1.0;
    }
    let gate_weight =
        DynTensor::from_vec(gate_data, &[num_experts, model_dim], &Device::Cpu).unwrap();
    let gate = Linear::new(gate_weight, None).unwrap();
    MoeRouter::new(gate, num_experts, top_k).unwrap()
}

fn make_tiny_expert(dim: usize, ff_dim: usize) -> SwiGluExpert {
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

// -- AC1: Uniform routing weights (all expert logits identical) ----------------

#[test]
fn test_moe_uniform_routing_weights() {
    let moe = make_moe_layer(4, 8, 4, 2, false);
    let x = DynTensor::from_vec(vec![1.0; 3 * 4], &[3, 4], &Device::Cpu).unwrap();
    let out = moe.forward(&x).unwrap();
    assert_eq!(out.dims(), &[3, 4]);
    let arr = out.as_cpu_f32().unwrap();
    assert!(
        arr.iter().all(|v| v.is_finite()),
        "uniform routing produced NaN/Inf"
    );
    let router = moe.router();
    let routing = router.forward(&x).unwrap();
    let w = routing.weights.as_cpu_f32().unwrap();
    for t in 0..3 {
        let sum: f32 = (0..2).map(|k| w[ndarray::IxDyn(&[t, k])]).sum();
        assert!((sum - 1.0).abs() < 1e-5, "token {t}: weight sum = {sum}");
    }
}

// -- AC2: Single expert (num_experts=1, top_k=1) degenerate case ---------------

#[test]
fn test_moe_single_expert() {
    let moe = make_moe_layer(4, 8, 1, 1, false);
    let x = DynTensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[1, 4], &Device::Cpu).unwrap();
    let out = moe.forward(&x).unwrap();
    assert_eq!(out.dims(), &[1, 4]);
    let arr = out.as_cpu_f32().unwrap();
    assert!(
        arr.iter().all(|v| v.is_finite()),
        "single expert produced NaN/Inf"
    );
    let routing = moe.router().forward(&x).unwrap();
    let w = routing.weights.as_cpu_f32().unwrap();
    assert!((w[ndarray::IxDyn(&[0, 0])] - 1.0).abs() < 1e-5);
}

// -- AC3: All tokens routed to same expert (identical input features) ----------

#[test]
fn test_moe_all_tokens_same_expert() {
    let moe = make_moe_layer(4, 8, 4, 1, false);
    let x = DynTensor::from_vec(
        vec![
            10.0, 0.0, 0.0, 0.0, 10.0, 0.0, 0.0, 0.0, 10.0, 0.0, 0.0, 0.0,
        ],
        &[3, 4],
        &Device::Cpu,
    )
    .unwrap();
    let out = moe.forward(&x).unwrap();
    assert_eq!(out.dims(), &[3, 4]);
    let arr = out.as_cpu_f32().unwrap();
    assert!(
        arr.iter().all(|v| v.is_finite()),
        "same-expert routing NaN/Inf"
    );
    let row0: Vec<f32> = (0..4).map(|d| arr[ndarray::IxDyn(&[0, d])]).collect();
    let row1: Vec<f32> = (0..4).map(|d| arr[ndarray::IxDyn(&[1, d])]).collect();
    let row2: Vec<f32> = (0..4).map(|d| arr[ndarray::IxDyn(&[2, d])]).collect();
    assert_eq!(
        row0, row1,
        "identical tokens should produce identical output"
    );
    assert_eq!(
        row1, row2,
        "identical tokens should produce identical output"
    );
}

// -- AC4: Expert receiving zero tokens (assignments.is_empty() branch) ---------

#[test]
fn test_moe_expert_receives_zero_tokens() {
    let moe = make_moe_layer(4, 8, 8, 1, false);
    let x = DynTensor::from_vec(
        vec![10.0, 0.0, 0.0, 0.0, 10.0, 0.0, 0.0, 0.0],
        &[2, 4],
        &Device::Cpu,
    )
    .unwrap();
    let out = moe.forward(&x).unwrap();
    assert_eq!(out.dims(), &[2, 4]);
    let arr = out.as_cpu_f32().unwrap();
    assert!(
        arr.iter().all(|v| v.is_finite()),
        "zero-token expert path NaN/Inf"
    );
}

// -- AC5: top_k == num_experts forward pass (all experts active) ---------------

#[test]
fn test_moe_topk_equals_num_experts_forward() {
    let moe = make_moe_layer(4, 8, 4, 4, false);
    let x = DynTensor::from_vec(
        vec![1.0, 2.0, 3.0, 4.0, 0.5, 1.5, 2.5, 3.5],
        &[2, 4],
        &Device::Cpu,
    )
    .unwrap();
    let out = moe.forward(&x).unwrap();
    assert_eq!(out.dims(), &[2, 4]);
    let arr = out.as_cpu_f32().unwrap();
    assert!(
        arr.iter().all(|v| v.is_finite()),
        "all-experts-active NaN/Inf"
    );
    let routing = moe.router().forward(&x).unwrap();
    assert_eq!(routing.indices.dims(), &[2, 4]);
    let w = routing.weights.as_cpu_f32().unwrap();
    for t in 0..2 {
        let sum: f32 = (0..4).map(|k| w[ndarray::IxDyn(&[t, k])]).sum();
        assert!((sum - 1.0).abs() < 1e-5, "token {t}: weight sum = {sum}");
    }
}

// -- group_tokens_by_expert boundary tests ------------------------------------

#[test]
fn test_group_tokens_by_expert_oob_index_returns_error() {
    let idx_data = vec![0u32, 5];
    let wt_data = vec![0.6f32, 0.4];
    let idx = ndarray::ArrayD::from_shape_vec(ndarray::IxDyn(&[1, 2]), idx_data).unwrap();
    let wt = ndarray::ArrayD::from_shape_vec(ndarray::IxDyn(&[1, 2]), wt_data).unwrap();
    let result = super::super::group_tokens_by_expert(&idx.view(), &wt.view(), 1, 2, 4);
    let err = result.unwrap_err();
    assert!(
        matches!(err, TensorError::DimensionOutOfRange { .. }),
        "expected DimensionOutOfRange for OOB routing index, got: {err:?}"
    );
}

#[test]
fn test_group_tokens_by_expert_valid_indices() {
    let idx_data = vec![0u32, 2, 1, 3];
    let wt_data = vec![0.6f32, 0.4, 0.7, 0.3];
    let idx = ndarray::ArrayD::from_shape_vec(ndarray::IxDyn(&[2, 2]), idx_data).unwrap();
    let wt = ndarray::ArrayD::from_shape_vec(ndarray::IxDyn(&[2, 2]), wt_data).unwrap();
    let groups = super::super::group_tokens_by_expert(&idx.view(), &wt.view(), 2, 2, 4).unwrap();
    assert_eq!(groups.len(), 4);
    assert_eq!(groups[0].len(), 1);
    assert_eq!(groups[0][0].0, 0);
    assert!((groups[0][0].1 - 0.6).abs() < 1e-6);
    assert_eq!(groups[1].len(), 1);
    assert_eq!(groups[1][0].0, 1);
    assert_eq!(groups[2].len(), 1);
    assert_eq!(groups[2][0].0, 0);
    assert_eq!(groups[3].len(), 1);
    assert_eq!(groups[3][0].0, 1);
}

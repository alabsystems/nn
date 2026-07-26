// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Tests for MoE GPU dispatch (scatter/gather routing).

use crate::dyn_tensor::DynTensor;
use crate::layers::{Linear, Module, SwiGluExpert};
use crate::var_builder::VarBuilder;
use crate::{DType, Device};

use super::{MoeDispatch, MoeDispatchConfig};

// -- Helpers -----------------------------------------------------------------

fn make_config(
    num_experts: usize,
    top_k: usize,
    hidden_size: usize,
    ff_dim: usize,
    norm: bool,
) -> MoeDispatchConfig {
    MoeDispatchConfig::new(num_experts, top_k, hidden_size, ff_dim, norm).unwrap()
}

fn make_router_linear(num_experts: usize, model_dim: usize) -> Linear {
    // Identity-ish gate: expert i gets highest logit when input feature i is largest.
    let mut gate_data = vec![0.0f32; num_experts * model_dim];
    for e in 0..num_experts.min(model_dim) {
        gate_data[e * model_dim + e] = 1.0;
    }
    let gate_weight =
        DynTensor::from_vec(gate_data, &[num_experts, model_dim], &Device::Cpu).unwrap();
    Linear::new(gate_weight, None).unwrap()
}

fn make_expert(dim: usize, ff_dim: usize, scale: f32) -> SwiGluExpert {
    let gate_w =
        DynTensor::from_vec(vec![scale; ff_dim * dim], &[ff_dim, dim], &Device::Cpu).unwrap();
    let up_w =
        DynTensor::from_vec(vec![scale; ff_dim * dim], &[ff_dim, dim], &Device::Cpu).unwrap();
    let down_w =
        DynTensor::from_vec(vec![scale; dim * ff_dim], &[dim, ff_dim], &Device::Cpu).unwrap();
    SwiGluExpert::new(
        Linear::new(gate_w, None).unwrap(),
        Linear::new(up_w, None).unwrap(),
        Linear::new(down_w, None).unwrap(),
    )
    .unwrap()
}

fn make_dispatch(
    model_dim: usize,
    ff_dim: usize,
    num_experts: usize,
    top_k: usize,
    norm: bool,
) -> MoeDispatch {
    let cfg = make_config(num_experts, top_k, model_dim, ff_dim, norm);
    let router = make_router_linear(num_experts, model_dim);
    let experts: Vec<SwiGluExpert> = (0..num_experts)
        .map(|_| make_expert(model_dim, ff_dim, 0.1))
        .collect();
    MoeDispatch::new(router, experts, cfg).unwrap()
}

// -- Config validation -------------------------------------------------------

#[test]
fn test_config_valid() {
    let cfg = MoeDispatchConfig::new(8, 2, 512, 2048, true);
    assert!(cfg.is_ok());
    let cfg = cfg.unwrap();
    assert_eq!(cfg.num_experts, 8);
    assert_eq!(cfg.top_k, 2);
    assert_eq!(cfg.hidden_size, 512);
    assert_eq!(cfg.expert_intermediate_size, 2048);
    assert!(cfg.norm_topk_prob);
}

#[test]
fn test_config_topk_zero_rejected() {
    let result = MoeDispatchConfig::new(4, 0, 64, 128, true);
    assert!(result.is_err());
}

#[test]
fn test_config_topk_exceeds_experts_rejected() {
    let result = MoeDispatchConfig::new(4, 5, 64, 128, true);
    assert!(result.is_err());
}

#[test]
fn test_config_hidden_size_zero_rejected() {
    let result = MoeDispatchConfig::new(4, 2, 0, 128, true);
    assert!(result.is_err());
}

#[test]
fn test_config_ff_dim_zero_rejected() {
    let result = MoeDispatchConfig::new(4, 2, 64, 0, true);
    assert!(result.is_err());
}

#[test]
fn test_config_topk_equals_experts_valid() {
    let result = MoeDispatchConfig::new(4, 4, 64, 128, false);
    assert!(result.is_ok());
}

// -- Routing correctness -----------------------------------------------------

#[test]
fn test_compute_routing_shapes() {
    let dispatch = make_dispatch(4, 8, 4, 2, true);
    let x = DynTensor::from_vec(vec![1.0; 3 * 4], &[3, 4], &Device::Cpu).unwrap();
    let (indices, weights) = dispatch.compute_routing(&x).unwrap();
    assert_eq!(indices.dims(), &[3, 2]);
    assert_eq!(weights.dims(), &[3, 2]);
    assert_eq!(indices.dtype(), DType::U32);
}

#[test]
fn test_compute_routing_normalized_weights_sum_to_one() {
    let dispatch = make_dispatch(4, 8, 4, 2, true);
    let x = DynTensor::from_vec(vec![3.0, 2.0, 1.0, 0.0], &[1, 4], &Device::Cpu).unwrap();
    let (_, weights) = dispatch.compute_routing(&x).unwrap();
    let w = weights.as_cpu_f32().unwrap();
    let sum: f32 = w.iter().sum();
    assert!(
        (sum - 1.0).abs() < 1e-5,
        "normalized routing weights sum = {sum}, expected ~1.0"
    );
}

#[test]
fn test_compute_routing_unnormalized_weights() {
    let dispatch = make_dispatch(4, 8, 4, 2, false);
    let x = DynTensor::from_vec(vec![10.0, 1.0, 0.5, 0.1], &[1, 4], &Device::Cpu).unwrap();
    let (_, weights) = dispatch.compute_routing(&x).unwrap();
    let w = weights.as_cpu_f32().unwrap();
    // Without normalization, weights are raw softmax top-k values.
    // Sum should be less than 1.0 (only 2 of 4 experts selected).
    let sum: f32 = w.iter().sum();
    assert!(
        sum < 1.0 + 1e-5,
        "unnormalized top-2 weights sum should be <= 1.0, got {sum}"
    );
}

#[test]
fn test_compute_routing_selects_correct_expert() {
    let dispatch = make_dispatch(4, 8, 4, 1, true);
    // Feature 0 is dominant -> expert 0 should be selected.
    let x = DynTensor::from_vec(vec![10.0, 0.1, 0.1, 0.1], &[1, 4], &Device::Cpu).unwrap();
    let (indices, _) = dispatch.compute_routing(&x).unwrap();
    let idx = indices.as_cpu_u32().unwrap();
    assert_eq!(idx[ndarray::IxDyn(&[0, 0])], 0);
}

// -- Forward shape tests -----------------------------------------------------

#[test]
fn test_forward_2d_shape() {
    let dispatch = make_dispatch(4, 8, 4, 2, true);
    let x = DynTensor::from_vec(vec![1.0; 3 * 4], &[3, 4], &Device::Cpu).unwrap();
    let out = dispatch.forward(&x).unwrap();
    assert_eq!(out.dims(), &[3, 4]);
}

#[test]
fn test_forward_3d_shape() {
    let dispatch = make_dispatch(4, 8, 4, 2, true);
    let x = DynTensor::from_vec(vec![1.0; 2 * 3 * 4], &[2, 3, 4], &Device::Cpu).unwrap();
    let out = dispatch.forward(&x).unwrap();
    assert_eq!(out.dims(), &[2, 3, 4]);
}

#[test]
fn test_forward_output_finite() {
    let dispatch = make_dispatch(4, 8, 4, 2, true);
    let x = DynTensor::from_vec(
        vec![1.0, 2.0, 3.0, 4.0, 0.5, 1.5, 2.5, 3.5],
        &[2, 4],
        &Device::Cpu,
    )
    .unwrap();
    let out = dispatch.forward(&x).unwrap();
    let arr = out.as_cpu_f32().unwrap();
    assert!(arr.iter().all(|v| v.is_finite()), "output contains NaN/Inf");
}

// -- Top-1 and top-2 mode tests ----------------------------------------------

#[test]
fn test_forward_top1_mode() {
    let cfg = make_config(4, 1, 4, 8, true);
    let router = make_router_linear(4, 4);
    let experts: Vec<SwiGluExpert> = (0..4)
        .map(|i| make_expert(4, 8, 0.1 * (i as f32 + 1.0)))
        .collect();
    let dispatch = MoeDispatch::new(router, experts, cfg).unwrap();

    let x = DynTensor::from_vec(vec![10.0, 0.0, 0.0, 0.0], &[1, 4], &Device::Cpu).unwrap();
    let out = dispatch.forward(&x).unwrap();
    assert_eq!(out.dims(), &[1, 4]);
    let arr = out.as_cpu_f32().unwrap();
    assert!(arr.iter().all(|v| v.is_finite()));
}

#[test]
fn test_forward_top2_mode() {
    let cfg = make_config(4, 2, 4, 8, true);
    let router = make_router_linear(4, 4);
    let experts: Vec<SwiGluExpert> = (0..4)
        .map(|i| make_expert(4, 8, 0.1 * (i as f32 + 1.0)))
        .collect();
    let dispatch = MoeDispatch::new(router, experts, cfg).unwrap();

    let x = DynTensor::from_vec(vec![5.0, 3.0, 1.0, 0.0], &[1, 4], &Device::Cpu).unwrap();
    let out = dispatch.forward(&x).unwrap();
    assert_eq!(out.dims(), &[1, 4]);
    let arr = out.as_cpu_f32().unwrap();
    assert!(arr.iter().all(|v| v.is_finite()));
}

// -- Value correctness -------------------------------------------------------

#[test]
fn test_forward_value_correctness_top1() {
    let model_dim = 2;
    let ff_dim = 4;
    let num_experts = 2;

    let expert_0 = make_expert(model_dim, ff_dim, 0.1);
    let expert_1 = make_expert(model_dim, ff_dim, 0.5);

    let x = DynTensor::from_vec(vec![1.0, 0.0], &[1, model_dim], &Device::Cpu).unwrap();

    // Expert 0 should be selected (feature 0 dominant with identity gate).
    let expected = expert_0.forward(&x).unwrap();
    let expected_vals = expected.to_flat_vec::<f32>().unwrap();

    let cfg = make_config(num_experts, 1, model_dim, ff_dim, true);
    let router = make_router_linear(num_experts, model_dim);
    let dispatch = MoeDispatch::new(router, vec![expert_0, expert_1], cfg).unwrap();
    let out = dispatch.forward(&x).unwrap();
    let out_vals = out.to_flat_vec::<f32>().unwrap();

    for i in 0..out_vals.len() {
        let err = (out_vals[i] - expected_vals[i]).abs();
        assert!(
            err < 1e-6,
            "top-1 value mismatch at [{i}]: got={:.8}, expected={:.8}",
            out_vals[i],
            expected_vals[i],
        );
    }
}

// -- Auxiliary loss -----------------------------------------------------------

#[test]
fn test_forward_with_aux_loss_shapes() {
    let dispatch = make_dispatch(4, 8, 4, 2, true);
    let x = DynTensor::from_vec(vec![1.0; 3 * 4], &[3, 4], &Device::Cpu).unwrap();
    let result = dispatch.forward_with_aux_loss(&x).unwrap();
    assert_eq!(result.hidden_states.dims(), &[3, 4]);
    // aux_loss should be scalar (rank 0).
    assert_eq!(result.aux_loss.rank(), 0);
}

#[test]
fn test_aux_loss_is_non_negative() {
    let dispatch = make_dispatch(4, 8, 4, 2, true);
    let x = DynTensor::from_vec(
        vec![1.0, 2.0, 3.0, 4.0, 4.0, 3.0, 2.0, 1.0],
        &[2, 4],
        &Device::Cpu,
    )
    .unwrap();
    let result = dispatch.forward_with_aux_loss(&x).unwrap();
    let loss_val = result.aux_loss.to_flat_vec::<f32>().unwrap()[0];
    assert!(
        loss_val >= 0.0,
        "aux loss should be non-negative, got {loss_val}"
    );
    assert!(
        loss_val.is_finite(),
        "aux loss should be finite, got {loss_val}"
    );
}

#[test]
fn test_forward_with_aux_loss_matches_forward() {
    let dispatch = make_dispatch(4, 8, 4, 2, true);
    let x = DynTensor::from_vec(
        vec![1.0, 2.0, 3.0, 4.0, 0.5, 1.5, 2.5, 3.5],
        &[2, 4],
        &Device::Cpu,
    )
    .unwrap();

    let out_forward = dispatch.forward(&x).unwrap();
    let out_with_loss = dispatch.forward_with_aux_loss(&x).unwrap();

    let vals_fwd = out_forward.to_flat_vec::<f32>().unwrap();
    let vals_aux = out_with_loss.hidden_states.to_flat_vec::<f32>().unwrap();

    assert_eq!(vals_fwd.len(), vals_aux.len());
    for i in 0..vals_fwd.len() {
        let err = (vals_fwd[i] - vals_aux[i]).abs();
        assert!(
            err < 1e-6,
            "forward vs forward_with_aux_loss mismatch at [{i}]: {:.8} vs {:.8}",
            vals_fwd[i],
            vals_aux[i],
        );
    }
}

// -- Expert count mismatch ---------------------------------------------------

#[test]
fn test_expert_count_mismatch() {
    let cfg = make_config(4, 2, 4, 8, true);
    let router = make_router_linear(4, 4);
    let experts: Vec<SwiGluExpert> = (0..3).map(|_| make_expert(4, 8, 0.1)).collect();
    let result = MoeDispatch::new(router, experts, cfg);
    assert!(result.is_err());
}

// -- VarBuilder load ---------------------------------------------------------

#[test]
fn test_load_varbuilder_zeros() {
    let cfg = make_config(4, 2, 4, 8, true);
    let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
    let dispatch = MoeDispatch::load(&vb, cfg).unwrap();
    assert_eq!(dispatch.config().num_experts, 4);

    let x = DynTensor::from_vec(vec![1.0; 2 * 4], &[2, 4], &Device::Cpu).unwrap();
    let out = dispatch.forward(&x).unwrap();
    assert_eq!(out.dims(), &[2, 4]);
}

// -- Edge cases --------------------------------------------------------------

#[test]
fn test_single_expert_top1() {
    let dispatch = make_dispatch(4, 8, 1, 1, true);
    let x = DynTensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[1, 4], &Device::Cpu).unwrap();
    let out = dispatch.forward(&x).unwrap();
    assert_eq!(out.dims(), &[1, 4]);
    let arr = out.as_cpu_f32().unwrap();
    assert!(arr.iter().all(|v| v.is_finite()));
}

#[test]
fn test_all_experts_selected() {
    // top_k == num_experts: all experts active for every token.
    let dispatch = make_dispatch(4, 8, 4, 4, true);
    let x = DynTensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[1, 4], &Device::Cpu).unwrap();
    let out = dispatch.forward(&x).unwrap();
    assert_eq!(out.dims(), &[1, 4]);
}

#[test]
fn test_uniform_input_all_tokens_same_expert() {
    // All tokens identical -> all routed to same expert(s).
    let dispatch = make_dispatch(4, 8, 4, 1, true);
    let x = DynTensor::from_vec(vec![1.0; 3 * 4], &[3, 4], &Device::Cpu).unwrap();
    let out = dispatch.forward(&x).unwrap();
    assert_eq!(out.dims(), &[3, 4]);
    let arr = out.as_cpu_f32().unwrap();
    // All rows should be identical since input is identical.
    let row0: Vec<f32> = (0..4).map(|d| arr[ndarray::IxDyn(&[0, d])]).collect();
    let row1: Vec<f32> = (0..4).map(|d| arr[ndarray::IxDyn(&[1, d])]).collect();
    let row2: Vec<f32> = (0..4).map(|d| arr[ndarray::IxDyn(&[2, d])]).collect();
    assert_eq!(row0, row1);
    assert_eq!(row1, row2);
}

// -- 8-expert configurations (#3547) -----------------------------------------

#[test]
fn test_8_experts_top2_forward() {
    let model_dim = 16;
    let ff_dim = 32;
    let dispatch = make_dispatch(model_dim, ff_dim, 8, 2, true);
    let x = DynTensor::from_vec(
        (0..4 * model_dim)
            .map(|i| (i as f32) * 0.01 + 0.1)
            .collect::<Vec<_>>(),
        &[4, model_dim],
        &Device::Cpu,
    )
    .unwrap();
    let out = dispatch.forward(&x).unwrap();
    assert_eq!(out.dims(), &[4, model_dim]);
    let arr = out.as_cpu_f32().unwrap();
    assert!(arr.iter().all(|v| v.is_finite()), "8-expert output NaN/Inf");
    assert!(
        arr.iter().any(|&v| v.abs() > 1e-12),
        "8-expert output should be non-zero"
    );
}

#[test]
fn test_8_experts_top2_3d_batched() {
    let model_dim = 16;
    let ff_dim = 32;
    let dispatch = make_dispatch(model_dim, ff_dim, 8, 2, true);
    let x = DynTensor::from_vec(
        (0..2 * 3 * model_dim)
            .map(|i| (i as f32) * 0.01 + 0.5)
            .collect::<Vec<_>>(),
        &[2, 3, model_dim],
        &Device::Cpu,
    )
    .unwrap();
    let out = dispatch.forward(&x).unwrap();
    assert_eq!(out.dims(), &[2, 3, model_dim]);
    let arr = out.as_cpu_f32().unwrap();
    assert!(arr.iter().all(|v| v.is_finite()));
}

#[test]
fn test_8_experts_top4_forward() {
    let model_dim = 16;
    let ff_dim = 32;
    let dispatch = make_dispatch(model_dim, ff_dim, 8, 4, true);
    let x = DynTensor::from_vec(
        (0..2 * model_dim)
            .map(|i| (i as f32) * 0.02 + 0.3)
            .collect::<Vec<_>>(),
        &[2, model_dim],
        &Device::Cpu,
    )
    .unwrap();
    let out = dispatch.forward(&x).unwrap();
    assert_eq!(out.dims(), &[2, model_dim]);
    let arr = out.as_cpu_f32().unwrap();
    assert!(arr.iter().all(|v| v.is_finite()));
}

#[test]
fn test_8_experts_aux_loss() {
    let model_dim = 16;
    let ff_dim = 32;
    let dispatch = make_dispatch(model_dim, ff_dim, 8, 2, true);
    let x = DynTensor::from_vec(
        (0..4 * model_dim)
            .map(|i| (i as f32) * 0.01 + 0.1)
            .collect::<Vec<_>>(),
        &[4, model_dim],
        &Device::Cpu,
    )
    .unwrap();
    let result = dispatch.forward_with_aux_loss(&x).unwrap();
    assert_eq!(result.hidden_states.dims(), &[4, model_dim]);
    assert_eq!(result.aux_loss.rank(), 0);
    let loss_val = result.aux_loss.to_flat_vec::<f32>().unwrap()[0];
    assert!(
        loss_val >= 0.0,
        "8-expert aux loss must be non-negative, got {loss_val}"
    );
    assert!(
        loss_val.is_finite(),
        "8-expert aux loss must be finite, got {loss_val}"
    );
}

// -- Aux loss computation tests (#3547) ---------------------------------------

#[test]
fn test_aux_loss_uniform_routing_near_one() {
    // Uniform routing: f_e = 1/E, P_e = 1/E for all experts.
    // aux_loss = E * sum(1/E * 1/E) = E * E * 1/E^2 = 1.0
    let model_dim = 4;
    let ff_dim = 8;
    let num_experts = 4;
    let dispatch = make_dispatch(model_dim, ff_dim, num_experts, num_experts, true);
    let x = DynTensor::from_vec(vec![1.0; 4 * model_dim], &[4, model_dim], &Device::Cpu).unwrap();
    let result = dispatch.forward_with_aux_loss(&x).unwrap();
    let loss_val = result.aux_loss.to_flat_vec::<f32>().unwrap()[0];
    assert!(
        (loss_val - 1.0).abs() < 0.1,
        "uniform routing aux_loss should be near 1.0, got {loss_val}"
    );
}

#[test]
fn test_aux_loss_skewed_routing_exceeds_one() {
    // Skewed: all tokens routed to expert 0. f_0=1, P_0 dominant.
    // aux_loss = E * 1 * P_0 > 1.0 since E > 1.
    let model_dim = 4;
    let ff_dim = 8;
    let num_experts = 4;
    let dispatch = make_dispatch(model_dim, ff_dim, num_experts, 1, true);
    let x = DynTensor::from_vec(
        vec![
            10.0, 0.0, 0.0, 0.0, 10.0, 0.0, 0.0, 0.0, 10.0, 0.0, 0.0, 0.0, 10.0, 0.0, 0.0, 0.0,
        ],
        &[4, model_dim],
        &Device::Cpu,
    )
    .unwrap();
    let result = dispatch.forward_with_aux_loss(&x).unwrap();
    let loss_val = result.aux_loss.to_flat_vec::<f32>().unwrap()[0];
    assert!(
        loss_val > 1.0,
        "skewed routing aux_loss should exceed 1.0, got {loss_val}"
    );
}

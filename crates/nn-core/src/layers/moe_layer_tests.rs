// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Tests for config-driven MoE layer (ExpertFFN + MoeLayer + MoeLayerConfig).

use crate::dyn_tensor::DynTensor;
use crate::layers::{Linear, Module};
use crate::var_builder::VarBuilder;
use crate::{DType, Device};
use std::collections::HashMap;

use super::{ExpertFFN, MoeLayer, MoeLayerConfig};

// -- Helpers -----------------------------------------------------------------

fn make_config(
    num_experts: usize,
    top_k: usize,
    hidden_size: usize,
    ff_dim: usize,
    norm: bool,
    shared: bool,
) -> MoeLayerConfig {
    MoeLayerConfig::new(num_experts, top_k, hidden_size, ff_dim, norm, shared).unwrap()
}

fn make_router_linear(num_experts: usize, model_dim: usize) -> Linear {
    let mut gate_data = vec![0.0f32; num_experts * model_dim];
    for e in 0..num_experts.min(model_dim) {
        gate_data[e * model_dim + e] = 1.0;
    }
    let gate_weight =
        DynTensor::from_vec(gate_data, &[num_experts, model_dim], &Device::Cpu).unwrap();
    Linear::new(gate_weight, None).unwrap()
}

fn make_expert(dim: usize, ff_dim: usize, scale: f32) -> ExpertFFN {
    let gate_w =
        DynTensor::from_vec(vec![scale; ff_dim * dim], &[ff_dim, dim], &Device::Cpu).unwrap();
    let up_w =
        DynTensor::from_vec(vec![scale; ff_dim * dim], &[ff_dim, dim], &Device::Cpu).unwrap();
    let down_w =
        DynTensor::from_vec(vec![scale; dim * ff_dim], &[dim, ff_dim], &Device::Cpu).unwrap();
    ExpertFFN::new(
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
    norm: bool,
    with_shared: bool,
) -> MoeLayer {
    let cfg = make_config(num_experts, top_k, model_dim, ff_dim, norm, with_shared);
    let router = make_router_linear(num_experts, model_dim);
    let experts: Vec<ExpertFFN> = (0..num_experts)
        .map(|_| make_expert(model_dim, ff_dim, 0.1))
        .collect();
    let shared = if with_shared {
        Some(make_expert(model_dim, ff_dim, 0.1))
    } else {
        None
    };
    MoeLayer::new(router, experts, shared, cfg).unwrap()
}

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

// -- Config validation -------------------------------------------------------

#[test]
fn test_config_valid() {
    let cfg = MoeLayerConfig::new(8, 2, 512, 2048, true, false);
    assert!(cfg.is_ok());
    let cfg = cfg.unwrap();
    assert_eq!(cfg.num_experts, 8);
    assert_eq!(cfg.top_k, 2);
    assert_eq!(cfg.hidden_size, 512);
    assert_eq!(cfg.expert_intermediate_size, 2048);
    assert!(cfg.norm_topk_prob);
    assert!(!cfg.shared_expert);
}

#[test]
fn test_config_topk_zero_rejected() {
    assert!(MoeLayerConfig::new(4, 0, 64, 128, true, false).is_err());
}

#[test]
fn test_config_topk_exceeds_experts_rejected() {
    assert!(MoeLayerConfig::new(4, 5, 64, 128, true, false).is_err());
}

#[test]
fn test_config_hidden_size_zero_rejected() {
    assert!(MoeLayerConfig::new(4, 2, 0, 128, true, false).is_err());
}

#[test]
fn test_config_ff_dim_zero_rejected() {
    assert!(MoeLayerConfig::new(4, 2, 64, 0, true, false).is_err());
}

#[test]
fn test_config_num_experts_zero_rejected() {
    assert!(MoeLayerConfig::new(0, 0, 64, 128, true, false).is_err());
}

#[test]
fn test_config_topk_equals_experts_valid() {
    assert!(MoeLayerConfig::new(4, 4, 64, 128, false, false).is_ok());
}

#[test]
fn test_config_shared_expert_intermediate_size() {
    let cfg = MoeLayerConfig::new(8, 2, 512, 2048, true, true)
        .unwrap()
        .with_shared_intermediate_size(4096)
        .unwrap();
    assert_eq!(cfg.shared_ff_dim(), 4096);
}

#[test]
fn test_config_shared_ff_dim_defaults_to_expert() {
    let cfg = MoeLayerConfig::new(8, 2, 512, 2048, true, true).unwrap();
    assert_eq!(cfg.shared_ff_dim(), 2048);
}

#[test]
fn test_config_shared_intermediate_size_zero_rejected() {
    let cfg = MoeLayerConfig::new(8, 2, 512, 2048, true, true).unwrap();
    assert!(cfg.with_shared_intermediate_size(0).is_err());
}

// -- ExpertFFN ---------------------------------------------------------------

#[test]
fn test_expert_ffn_forward_shape() {
    let expert = make_expert(4, 8, 0.1);
    let x = DynTensor::from_vec(vec![1.0; 4], &[1, 4], &Device::Cpu).unwrap();
    let out = expert.forward(&x).unwrap();
    assert_eq!(out.dims(), &[1, 4]);
}

#[test]
fn test_expert_ffn_batch_forward() {
    let expert = make_expert(4, 8, 0.1);
    let x = DynTensor::from_vec(vec![1.0; 3 * 4], &[3, 4], &Device::Cpu).unwrap();
    let out = expert.forward(&x).unwrap();
    assert_eq!(out.dims(), &[3, 4]);
}

#[test]
fn test_expert_ffn_dimension_mismatch_rejected() {
    let gate_w = DynTensor::from_vec(vec![0.1; 8 * 4], &[8, 4], &Device::Cpu).unwrap();
    let up_w = DynTensor::from_vec(vec![0.1; 6 * 4], &[6, 4], &Device::Cpu).unwrap(); // mismatch
    let down_w = DynTensor::from_vec(vec![0.1; 4 * 8], &[4, 8], &Device::Cpu).unwrap();
    let result = ExpertFFN::new(
        Linear::new(gate_w, None).unwrap(),
        Linear::new(up_w, None).unwrap(),
        Linear::new(down_w, None).unwrap(),
    );
    assert!(result.is_err());
}

// -- MoeLayer forward --------------------------------------------------------

#[test]
fn test_forward_2d_shape() {
    let moe = make_moe_layer(4, 8, 4, 2, true, false);
    let x = DynTensor::from_vec(vec![1.0; 3 * 4], &[3, 4], &Device::Cpu).unwrap();
    let out = moe.forward(&x).unwrap();
    assert_eq!(out.dims(), &[3, 4]);
}

#[test]
fn test_forward_3d_shape() {
    let moe = make_moe_layer(4, 8, 4, 2, true, false);
    let x = DynTensor::from_vec(vec![1.0; 2 * 3 * 4], &[2, 3, 4], &Device::Cpu).unwrap();
    let out = moe.forward(&x).unwrap();
    assert_eq!(out.dims(), &[2, 3, 4]);
}

#[test]
fn test_forward_output_finite() {
    let moe = make_moe_layer(4, 8, 4, 2, true, false);
    let x = DynTensor::from_vec(
        vec![1.0, 2.0, 3.0, 4.0, 0.5, 1.5, 2.5, 3.5],
        &[2, 4],
        &Device::Cpu,
    )
    .unwrap();
    let out = moe.forward(&x).unwrap();
    let arr = out.as_cpu_f32().unwrap();
    assert!(arr.iter().all(|v| v.is_finite()), "output contains NaN/Inf");
}

// -- Shared expert -----------------------------------------------------------

#[test]
fn test_shared_expert_adds_to_output() {
    let moe_no_shared = make_moe_layer(4, 8, 4, 2, true, false);
    let moe_shared = make_moe_layer(4, 8, 4, 2, true, true);
    let x = DynTensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[1, 4], &Device::Cpu).unwrap();
    let out_no_shared = moe_no_shared.forward(&x).unwrap();
    let out_shared = moe_shared.forward(&x).unwrap();

    // Both should be finite.
    let arr_ns = out_no_shared.as_cpu_f32().unwrap();
    let arr_s = out_shared.as_cpu_f32().unwrap();
    assert!(arr_ns.iter().all(|v| v.is_finite()));
    assert!(arr_s.iter().all(|v| v.is_finite()));

    // Shared expert output should differ from non-shared (shared adds extra values).
    let vals_ns: Vec<f32> = arr_ns.iter().copied().collect();
    let vals_s: Vec<f32> = arr_s.iter().copied().collect();
    let any_diff = vals_ns
        .iter()
        .zip(&vals_s)
        .any(|(a, b)| (a - b).abs() > 1e-10);
    assert!(any_diff, "shared expert should change the output");
}

#[test]
fn test_shared_expert_config_mismatch_rejected() {
    // Config says shared_expert=true but no shared expert provided.
    let cfg = make_config(4, 2, 4, 8, true, true);
    let router = make_router_linear(4, 4);
    let experts: Vec<ExpertFFN> = (0..4).map(|_| make_expert(4, 8, 0.1)).collect();
    let result = MoeLayer::new(router, experts, None, cfg);
    assert!(result.is_err());
}

// -- Auxiliary loss -----------------------------------------------------------

#[test]
fn test_forward_with_aux_shapes() {
    let moe = make_moe_layer(4, 8, 4, 2, true, false);
    let x = DynTensor::from_vec(vec![1.0; 3 * 4], &[3, 4], &Device::Cpu).unwrap();
    let result = moe.forward_with_aux(&x).unwrap();
    assert_eq!(result.hidden_states.dims(), &[3, 4]);
    assert_eq!(result.aux_loss.rank(), 0);
}

#[test]
fn test_aux_loss_non_negative_and_finite() {
    let moe = make_moe_layer(4, 8, 4, 2, true, false);
    let x = DynTensor::from_vec(
        vec![1.0, 2.0, 3.0, 4.0, 4.0, 3.0, 2.0, 1.0],
        &[2, 4],
        &Device::Cpu,
    )
    .unwrap();
    let result = moe.forward_with_aux(&x).unwrap();
    let loss_val = result.aux_loss.to_flat_vec::<f32>().unwrap()[0];
    assert!(loss_val >= 0.0, "aux loss should be >= 0, got {loss_val}");
    assert!(
        loss_val.is_finite(),
        "aux loss should be finite, got {loss_val}"
    );
}

#[test]
fn test_forward_with_aux_matches_forward() {
    let moe = make_moe_layer(4, 8, 4, 2, true, false);
    let x = DynTensor::from_vec(
        vec![1.0, 2.0, 3.0, 4.0, 0.5, 1.5, 2.5, 3.5],
        &[2, 4],
        &Device::Cpu,
    )
    .unwrap();
    let out_fwd = moe.forward(&x).unwrap();
    let out_aux = moe.forward_with_aux(&x).unwrap();

    let vals_fwd = out_fwd.to_flat_vec::<f32>().unwrap();
    let vals_aux = out_aux.hidden_states.to_flat_vec::<f32>().unwrap();
    assert_eq!(vals_fwd.len(), vals_aux.len());
    for i in 0..vals_fwd.len() {
        let err = (vals_fwd[i] - vals_aux[i]).abs();
        assert!(
            err < 1e-6,
            "forward vs forward_with_aux mismatch at [{i}]: {:.8} vs {:.8}",
            vals_fwd[i],
            vals_aux[i],
        );
    }
}

// -- Expert count mismatch ---------------------------------------------------

#[test]
fn test_expert_count_mismatch() {
    let cfg = make_config(4, 2, 4, 8, true, false);
    let router = make_router_linear(4, 4);
    let experts: Vec<ExpertFFN> = (0..3).map(|_| make_expert(4, 8, 0.1)).collect();
    let result = MoeLayer::new(router, experts, None, cfg);
    assert!(result.is_err());
}

// -- VarBuilder load ---------------------------------------------------------

#[test]
fn test_load_varbuilder_zeros() {
    let cfg = make_config(4, 2, 4, 8, true, false);
    let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
    let moe = MoeLayer::load(&vb, cfg).unwrap();
    assert_eq!(moe.config().num_experts, 4);

    let x = DynTensor::from_vec(vec![1.0; 2 * 4], &[2, 4], &Device::Cpu).unwrap();
    let out = moe.forward(&x).unwrap();
    assert_eq!(out.dims(), &[2, 4]);
}

#[test]
fn test_load_varbuilder_with_shared() {
    let model_dim = 4;
    let ff_dim = 8;
    let num_experts = 2;
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
    let cfg = MoeLayerConfig::new(num_experts, 1, model_dim, ff_dim, true, true).unwrap();
    let moe = MoeLayer::load(&vb, cfg).unwrap();

    let x = DynTensor::from_vec(vec![1.0; model_dim], &[1, model_dim], &device).unwrap();
    let out = moe.forward(&x).unwrap();
    assert_eq!(out.dims(), &[1, model_dim]);
    let arr = out.as_cpu_f32().unwrap();
    assert!(arr.iter().all(|v| v.is_finite()));
}

// -- Edge cases --------------------------------------------------------------

#[test]
fn test_single_expert_top1() {
    let moe = make_moe_layer(4, 8, 1, 1, true, false);
    let x = DynTensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[1, 4], &Device::Cpu).unwrap();
    let out = moe.forward(&x).unwrap();
    assert_eq!(out.dims(), &[1, 4]);
    let arr = out.as_cpu_f32().unwrap();
    assert!(arr.iter().all(|v| v.is_finite()));
}

#[test]
fn test_all_experts_selected() {
    let moe = make_moe_layer(4, 8, 4, 4, true, false);
    let x = DynTensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[1, 4], &Device::Cpu).unwrap();
    let out = moe.forward(&x).unwrap();
    assert_eq!(out.dims(), &[1, 4]);
}

#[test]
fn test_uniform_input_all_rows_identical() {
    let moe = make_moe_layer(4, 8, 4, 1, true, false);
    let x = DynTensor::from_vec(vec![1.0; 3 * 4], &[3, 4], &Device::Cpu).unwrap();
    let out = moe.forward(&x).unwrap();
    let arr = out.as_cpu_f32().unwrap();
    let row0: Vec<f32> = (0..4).map(|d| arr[ndarray::IxDyn(&[0, d])]).collect();
    let row1: Vec<f32> = (0..4).map(|d| arr[ndarray::IxDyn(&[1, d])]).collect();
    let row2: Vec<f32> = (0..4).map(|d| arr[ndarray::IxDyn(&[2, d])]).collect();
    assert_eq!(row0, row1);
    assert_eq!(row1, row2);
}

#[test]
fn test_nan_input_returns_error() {
    let moe = make_moe_layer(4, 8, 4, 2, true, false);
    let x = DynTensor::from_vec(vec![f32::NAN, 1.0, 0.5, 0.1], &[1, 4], &Device::Cpu).unwrap();
    let result = moe.forward(&x);
    assert!(result.is_err());
}

// -- 8-expert configurations (#3547) -----------------------------------------

#[test]
fn test_8_experts_top2_forward() {
    let model_dim = 16;
    let ff_dim = 32;
    let moe = make_moe_layer(model_dim, ff_dim, 8, 2, true, false);
    let x = DynTensor::from_vec(
        (0..4 * model_dim)
            .map(|i| (i as f32) * 0.01 + 0.1)
            .collect::<Vec<_>>(),
        &[4, model_dim],
        &Device::Cpu,
    )
    .unwrap();
    let out = moe.forward(&x).unwrap();
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
    let moe = make_moe_layer(model_dim, ff_dim, 8, 2, true, false);
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
    assert!(arr.iter().all(|v| v.is_finite()));
}

#[test]
fn test_8_experts_top4_forward() {
    let model_dim = 16;
    let ff_dim = 32;
    let moe = make_moe_layer(model_dim, ff_dim, 8, 4, true, false);
    let x = DynTensor::from_vec(
        (0..2 * model_dim)
            .map(|i| (i as f32) * 0.02 + 0.3)
            .collect::<Vec<_>>(),
        &[2, model_dim],
        &Device::Cpu,
    )
    .unwrap();
    let out = moe.forward(&x).unwrap();
    assert_eq!(out.dims(), &[2, model_dim]);
    let arr = out.as_cpu_f32().unwrap();
    assert!(arr.iter().all(|v| v.is_finite()));
}

#[test]
fn test_8_experts_with_shared_expert() {
    let model_dim = 16;
    let ff_dim = 32;
    let moe = make_moe_layer(model_dim, ff_dim, 8, 2, true, true);
    let x = DynTensor::from_vec(
        (0..3 * model_dim)
            .map(|i| (i as f32) * 0.01 + 0.2)
            .collect::<Vec<_>>(),
        &[3, model_dim],
        &Device::Cpu,
    )
    .unwrap();
    let out = moe.forward(&x).unwrap();
    assert_eq!(out.dims(), &[3, model_dim]);
    let arr = out.as_cpu_f32().unwrap();
    assert!(arr.iter().all(|v| v.is_finite()));
}

#[test]
fn test_8_experts_aux_loss() {
    let model_dim = 16;
    let ff_dim = 32;
    let moe = make_moe_layer(model_dim, ff_dim, 8, 2, true, false);
    let x = DynTensor::from_vec(
        (0..4 * model_dim)
            .map(|i| (i as f32) * 0.01 + 0.1)
            .collect::<Vec<_>>(),
        &[4, model_dim],
        &Device::Cpu,
    )
    .unwrap();
    let result = moe.forward_with_aux(&x).unwrap();
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

// -- Aux loss load-balancing computation tests (#3547) ------------------------

#[test]
fn test_aux_loss_uniform_routing_near_one() {
    // With uniform routing (all tokens routed equally to all experts),
    // aux_loss = num_experts * sum_e(f_e * P_e).
    // When routing is uniform: f_e = 1/E for all e, P_e = 1/E for all e.
    // aux_loss = E * E * (1/E * 1/E) = E * E * 1/E^2 = 1.0.
    // With softmax and identity gate, routing is not perfectly uniform
    // but should be close to 1.0.
    let model_dim = 4;
    let ff_dim = 8;
    let num_experts = 4;
    let moe = make_moe_layer(model_dim, ff_dim, num_experts, num_experts, true, false);
    // All-ones input: uniform routing because identity gate gives equal logits.
    let x = DynTensor::from_vec(vec![1.0; 4 * model_dim], &[4, model_dim], &Device::Cpu).unwrap();
    let result = moe.forward_with_aux(&x).unwrap();
    let loss_val = result.aux_loss.to_flat_vec::<f32>().unwrap()[0];
    // With uniform routing, aux_loss should be approximately 1.0.
    assert!(
        (loss_val - 1.0).abs() < 0.1,
        "uniform routing aux_loss should be near 1.0, got {loss_val}"
    );
}

#[test]
fn test_aux_loss_skewed_routing_exceeds_one() {
    // When routing is skewed (all tokens go to one expert),
    // f_e=1 for that expert, f_e=0 for others.
    // P_e is also skewed. aux_loss = E * f_0 * P_0 > 1.0 for E > 1.
    let model_dim = 4;
    let ff_dim = 8;
    let num_experts = 4;
    let top_k = 1;
    let moe = make_moe_layer(model_dim, ff_dim, num_experts, top_k, true, false);
    // All tokens have feature 0 dominant: all route to expert 0.
    let x = DynTensor::from_vec(
        vec![
            10.0, 0.0, 0.0, 0.0, 10.0, 0.0, 0.0, 0.0, 10.0, 0.0, 0.0, 0.0, 10.0, 0.0, 0.0, 0.0,
        ],
        &[4, model_dim],
        &Device::Cpu,
    )
    .unwrap();
    let result = moe.forward_with_aux(&x).unwrap();
    let loss_val = result.aux_loss.to_flat_vec::<f32>().unwrap()[0];
    // Skewed routing: aux_loss should be > 1.0 (penalizing imbalance).
    assert!(
        loss_val > 1.0,
        "skewed routing aux_loss should exceed 1.0, got {loss_val}"
    );
}

// -- ExpertMlp tests (#3547) --------------------------------------------------

use super::ExpertMlp;
use crate::layers::Activation;

fn make_mlp_expert(dim: usize, ff_dim: usize, scale: f32, act: Activation) -> ExpertMlp {
    let up_w =
        DynTensor::from_vec(vec![scale; ff_dim * dim], &[ff_dim, dim], &Device::Cpu).unwrap();
    let down_w =
        DynTensor::from_vec(vec![scale; dim * ff_dim], &[dim, ff_dim], &Device::Cpu).unwrap();
    ExpertMlp::new(
        Linear::new(up_w, None).unwrap(),
        Linear::new(down_w, None).unwrap(),
        act,
    )
    .unwrap()
}

#[test]
fn test_expert_mlp_forward_shape() {
    let expert = make_mlp_expert(4, 8, 0.1, Activation::Gelu);
    let x = DynTensor::from_vec(vec![1.0; 4], &[1, 4], &Device::Cpu).unwrap();
    let out = expert.forward(&x).unwrap();
    assert_eq!(out.dims(), &[1, 4]);
}

#[test]
fn test_expert_mlp_batch_forward() {
    let expert = make_mlp_expert(4, 8, 0.1, Activation::Relu);
    let x = DynTensor::from_vec(vec![1.0; 3 * 4], &[3, 4], &Device::Cpu).unwrap();
    let out = expert.forward(&x).unwrap();
    assert_eq!(out.dims(), &[3, 4]);
    let arr = out.as_cpu_f32().unwrap();
    assert!(arr.iter().all(|v| v.is_finite()));
}

#[test]
fn test_expert_mlp_relu_zeros_negative() {
    // ReLU activation: negative intermediate values should be zeroed.
    let expert = make_mlp_expert(2, 4, 0.1, Activation::Relu);
    let x = DynTensor::from_vec(vec![-1.0, -1.0], &[1, 2], &Device::Cpu).unwrap();
    let out = expert.forward(&x).unwrap();
    let arr = out.as_cpu_f32().unwrap();
    // All-negative input through uniform positive up_proj -> negative intermediate
    // -> ReLU zeros it -> down_proj of zeros = zeros.
    for v in arr.iter() {
        assert!(
            v.abs() < 1e-6,
            "ReLU expert with all-negative input should produce zero, got {v}"
        );
    }
}

#[test]
fn test_expert_mlp_silu_nonzero() {
    let expert = make_mlp_expert(4, 8, 0.1, Activation::Silu);
    let x = DynTensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[1, 4], &Device::Cpu).unwrap();
    let out = expert.forward(&x).unwrap();
    let arr = out.as_cpu_f32().unwrap();
    assert!(arr.iter().all(|v| v.is_finite()));
    assert!(
        arr.iter().any(|&v| v.abs() > 1e-10),
        "SiLU expert with positive input should produce non-zero output"
    );
}

#[test]
fn test_expert_mlp_different_activations_differ() {
    // Different activations should produce different outputs.
    let expert_relu = make_mlp_expert(4, 8, 0.1, Activation::Relu);
    let expert_gelu = make_mlp_expert(4, 8, 0.1, Activation::Gelu);
    let x = DynTensor::from_vec(vec![1.0, 2.0, -1.0, 0.5], &[1, 4], &Device::Cpu).unwrap();
    let out_relu = expert_relu
        .forward(&x)
        .unwrap()
        .to_flat_vec::<f32>()
        .unwrap();
    let out_gelu = expert_gelu
        .forward(&x)
        .unwrap()
        .to_flat_vec::<f32>()
        .unwrap();
    let diff: f32 = out_relu
        .iter()
        .zip(&out_gelu)
        .map(|(a, b)| (a - b).abs())
        .sum();
    assert!(
        diff > 1e-6,
        "ReLU and GELU experts should produce different outputs, diff={diff}"
    );
}

#[test]
fn test_expert_mlp_dimension_mismatch_rejected() {
    let up_w = DynTensor::from_vec(vec![0.1; 8 * 4], &[8, 4], &Device::Cpu).unwrap();
    let down_w = DynTensor::from_vec(vec![0.1; 4 * 6], &[4, 6], &Device::Cpu).unwrap(); // mismatch: 6 != 8
    let result = ExpertMlp::new(
        Linear::new(up_w, None).unwrap(),
        Linear::new(down_w, None).unwrap(),
        Activation::Gelu,
    );
    assert!(result.is_err());
}

#[test]
fn test_expert_mlp_load_varbuilder() {
    let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
    let expert = ExpertMlp::load(&vb, 4, 8, Activation::Silu).unwrap();
    let x = DynTensor::from_vec(vec![1.0; 2 * 4], &[2, 4], &Device::Cpu).unwrap();
    let out = expert.forward(&x).unwrap();
    assert_eq!(out.dims(), &[2, 4]);
}

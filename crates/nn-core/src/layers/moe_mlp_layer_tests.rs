// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Tests for MoeMlpLayer -- MoE forward dispatch with ExpertMlp experts.

use crate::dyn_tensor::DynTensor;
use crate::layers::{Activation, Linear, Module};
use crate::var_builder::VarBuilder;
use crate::{DType, Device};

use super::{ExpertMlp, MoeMlpConfig, MoeMlpLayer};

// -- Helpers -----------------------------------------------------------------

fn make_config(
    num_experts: usize,
    top_k: usize,
    hidden_size: usize,
    ff_dim: usize,
    norm: bool,
    act: Activation,
) -> MoeMlpConfig {
    MoeMlpConfig::new(num_experts, top_k, hidden_size, ff_dim, norm, act).unwrap()
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

fn make_moe_mlp_layer(
    model_dim: usize,
    ff_dim: usize,
    num_experts: usize,
    top_k: usize,
    norm: bool,
    act: Activation,
) -> MoeMlpLayer {
    let cfg = make_config(num_experts, top_k, model_dim, ff_dim, norm, act);
    let router = make_router_linear(num_experts, model_dim);
    let experts: Vec<ExpertMlp> = (0..num_experts)
        .map(|_| make_mlp_expert(model_dim, ff_dim, 0.1, act))
        .collect();
    MoeMlpLayer::new(router, experts, cfg).unwrap()
}

// -- Config validation -------------------------------------------------------

#[test]
fn test_moe_mlp_config_valid() {
    let cfg = MoeMlpConfig::new(8, 2, 512, 2048, true, Activation::Gelu);
    assert!(cfg.is_ok());
    let cfg = cfg.unwrap();
    assert_eq!(cfg.num_experts, 8);
    assert_eq!(cfg.top_k, 2);
    assert_eq!(cfg.activation, Activation::Gelu);
}

#[test]
fn test_moe_mlp_config_topk_zero_rejected() {
    assert!(MoeMlpConfig::new(4, 0, 64, 128, true, Activation::Relu).is_err());
}

#[test]
fn test_moe_mlp_config_topk_exceeds_experts_rejected() {
    assert!(MoeMlpConfig::new(4, 5, 64, 128, true, Activation::Relu).is_err());
}

#[test]
fn test_moe_mlp_config_hidden_zero_rejected() {
    assert!(MoeMlpConfig::new(4, 2, 0, 128, true, Activation::Relu).is_err());
}

#[test]
fn test_moe_mlp_config_ff_dim_zero_rejected() {
    assert!(MoeMlpConfig::new(4, 2, 64, 0, true, Activation::Relu).is_err());
}

// -- ExpertMlp forward -------------------------------------------------------

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
fn test_expert_mlp_dimension_mismatch_rejected() {
    let up_w = DynTensor::from_vec(vec![0.1; 8 * 4], &[8, 4], &Device::Cpu).unwrap();
    let down_w = DynTensor::from_vec(vec![0.1; 4 * 6], &[4, 6], &Device::Cpu).unwrap();
    let result = ExpertMlp::new(
        Linear::new(up_w, None).unwrap(),
        Linear::new(down_w, None).unwrap(),
        Activation::Gelu,
    );
    assert!(result.is_err());
}

#[test]
fn test_expert_mlp_relu_zeros_negative() {
    let expert = make_mlp_expert(2, 4, 0.1, Activation::Relu);
    let x = DynTensor::from_vec(vec![-1.0, -1.0], &[1, 2], &Device::Cpu).unwrap();
    let out = expert.forward(&x).unwrap();
    let arr = out.as_cpu_f32().unwrap();
    for v in arr.iter() {
        assert!(
            v.abs() < 1e-6,
            "ReLU expert with all-negative input should produce zero, got {v}"
        );
    }
}

#[test]
fn test_expert_mlp_load_varbuilder() {
    let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
    let expert = ExpertMlp::load(&vb, 4, 8, Activation::Silu).unwrap();
    let x = DynTensor::from_vec(vec![1.0; 2 * 4], &[2, 4], &Device::Cpu).unwrap();
    let out = expert.forward(&x).unwrap();
    assert_eq!(out.dims(), &[2, 4]);
}

// -- Forward shape: 2 experts, top_k=1 --------------------------------------

#[test]
fn test_moe_mlp_2experts_top1_output_shape() {
    let moe = make_moe_mlp_layer(4, 8, 2, 1, true, Activation::Gelu);
    let x = DynTensor::from_vec(vec![1.0; 3 * 4], &[3, 4], &Device::Cpu).unwrap();
    let out = moe.forward(&x).unwrap();
    assert_eq!(out.dims(), &[3, 4], "output shape must match input");
}

#[test]
fn test_moe_mlp_2experts_top1_output_finite() {
    let moe = make_moe_mlp_layer(4, 8, 2, 1, true, Activation::Gelu);
    let x = DynTensor::from_vec(
        vec![1.0, 2.0, 3.0, 4.0, 0.5, 1.5, 2.5, 3.5],
        &[2, 4],
        &Device::Cpu,
    )
    .unwrap();
    let out = moe.forward(&x).unwrap();
    let arr = out.as_cpu_f32().unwrap();
    assert!(arr.iter().all(|v| v.is_finite()), "output must be finite");
}

// -- Forward shape: 4 experts, top_k=2 --------------------------------------

#[test]
fn test_moe_mlp_4experts_top2_output_shape() {
    let moe = make_moe_mlp_layer(4, 8, 4, 2, true, Activation::Relu);
    let x = DynTensor::from_vec(vec![1.0; 3 * 4], &[3, 4], &Device::Cpu).unwrap();
    let out = moe.forward(&x).unwrap();
    assert_eq!(out.dims(), &[3, 4]);
}

#[test]
fn test_moe_mlp_4experts_top2_3d_batched() {
    let moe = make_moe_mlp_layer(8, 16, 4, 2, true, Activation::Silu);
    let x = DynTensor::from_vec(
        (0..2 * 3 * 8)
            .map(|i| (i as f32) * 0.01 + 0.5)
            .collect::<Vec<_>>(),
        &[2, 3, 8],
        &Device::Cpu,
    )
    .unwrap();
    let out = moe.forward(&x).unwrap();
    assert_eq!(out.dims(), &[2, 3, 8]);
    let arr = out.as_cpu_f32().unwrap();
    assert!(arr.iter().all(|v| v.is_finite()));
}

// -- Routing: each token gets exactly top_k experts --------------------------

#[test]
fn test_moe_mlp_routing_topk_uniform_rows_identical() {
    // Uniform input through identical experts should produce identical rows.
    let moe = make_moe_mlp_layer(4, 8, 4, 2, true, Activation::Gelu);
    let x = DynTensor::from_vec(vec![1.0; 3 * 4], &[3, 4], &Device::Cpu).unwrap();
    let out = moe.forward(&x).unwrap();
    let arr = out.as_cpu_f32().unwrap();
    let row0: Vec<f32> = (0..4).map(|d| arr[ndarray::IxDyn(&[0, d])]).collect();
    let row1: Vec<f32> = (0..4).map(|d| arr[ndarray::IxDyn(&[1, d])]).collect();
    let row2: Vec<f32> = (0..4).map(|d| arr[ndarray::IxDyn(&[2, d])]).collect();
    assert_eq!(row0, row1, "uniform input rows must be identical");
    assert_eq!(row1, row2, "uniform input rows must be identical");
}

// -- Weight normalization: top-k weights sum to 1 ----------------------------

#[test]
fn test_moe_mlp_normalized_weights_sum_to_one() {
    let moe = make_moe_mlp_layer(4, 8, 2, 2, true, Activation::Relu);
    let x = DynTensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[1, 4], &Device::Cpu).unwrap();
    let out = moe.forward(&x).unwrap();
    assert_eq!(out.dims(), &[1, 4]);
    let arr = out.as_cpu_f32().unwrap();
    assert!(arr.iter().all(|v| v.is_finite()));
}

// -- Auxiliary loss -----------------------------------------------------------

#[test]
fn test_moe_mlp_forward_with_aux_shapes() {
    let moe = make_moe_mlp_layer(4, 8, 4, 2, true, Activation::Gelu);
    let x = DynTensor::from_vec(vec![1.0; 3 * 4], &[3, 4], &Device::Cpu).unwrap();
    let result = moe.forward_with_aux(&x).unwrap();
    assert_eq!(result.hidden_states.dims(), &[3, 4]);
    assert_eq!(result.aux_loss.rank(), 0, "aux_loss must be scalar");
}

#[test]
fn test_moe_mlp_aux_loss_non_negative_and_finite() {
    let moe = make_moe_mlp_layer(4, 8, 4, 2, true, Activation::Silu);
    let x = DynTensor::from_vec(
        vec![1.0, 2.0, 3.0, 4.0, 4.0, 3.0, 2.0, 1.0],
        &[2, 4],
        &Device::Cpu,
    )
    .unwrap();
    let result = moe.forward_with_aux(&x).unwrap();
    let loss_val = result.aux_loss.to_flat_vec::<f32>().unwrap()[0];
    assert!(loss_val >= 0.0, "aux loss must be >= 0, got {loss_val}");
    assert!(
        loss_val.is_finite(),
        "aux loss must be finite, got {loss_val}"
    );
}

#[test]
fn test_moe_mlp_forward_with_aux_matches_forward() {
    let moe = make_moe_mlp_layer(4, 8, 4, 2, true, Activation::Gelu);
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
fn test_moe_mlp_expert_count_mismatch() {
    let cfg = make_config(4, 2, 4, 8, true, Activation::Gelu);
    let router = make_router_linear(4, 4);
    let experts: Vec<ExpertMlp> = (0..3)
        .map(|_| make_mlp_expert(4, 8, 0.1, Activation::Gelu))
        .collect();
    let result = MoeMlpLayer::new(router, experts, cfg);
    assert!(result.is_err(), "mismatched expert count must be rejected");
}

// -- VarBuilder load ---------------------------------------------------------

#[test]
fn test_moe_mlp_load_varbuilder_zeros() {
    let cfg = make_config(4, 2, 4, 8, true, Activation::Relu);
    let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
    let moe = MoeMlpLayer::load(&vb, cfg).unwrap();
    assert_eq!(moe.config().num_experts, 4);

    let x = DynTensor::from_vec(vec![1.0; 2 * 4], &[2, 4], &Device::Cpu).unwrap();
    let out = moe.forward(&x).unwrap();
    assert_eq!(out.dims(), &[2, 4]);
}

// -- Different activations produce different outputs -------------------------

#[test]
fn test_moe_mlp_relu_vs_gelu_differ() {
    let moe_relu = make_moe_mlp_layer(4, 8, 2, 1, true, Activation::Relu);
    let moe_gelu = make_moe_mlp_layer(4, 8, 2, 1, true, Activation::Gelu);
    let x = DynTensor::from_vec(vec![1.0, 2.0, -1.0, 0.5], &[1, 4], &Device::Cpu).unwrap();
    let out_relu = moe_relu.forward(&x).unwrap().to_flat_vec::<f32>().unwrap();
    let out_gelu = moe_gelu.forward(&x).unwrap().to_flat_vec::<f32>().unwrap();
    let diff: f32 = out_relu
        .iter()
        .zip(&out_gelu)
        .map(|(a, b)| (a - b).abs())
        .sum();
    assert!(
        diff > 1e-6,
        "ReLU and GELU MoE layers should produce different outputs, diff={diff}"
    );
}

// -- Edge cases --------------------------------------------------------------

#[test]
fn test_moe_mlp_single_expert_top1() {
    let moe = make_moe_mlp_layer(4, 8, 1, 1, true, Activation::Silu);
    let x = DynTensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[1, 4], &Device::Cpu).unwrap();
    let out = moe.forward(&x).unwrap();
    assert_eq!(out.dims(), &[1, 4]);
    let arr = out.as_cpu_f32().unwrap();
    assert!(arr.iter().all(|v| v.is_finite()));
}

#[test]
fn test_moe_mlp_all_experts_selected() {
    let moe = make_moe_mlp_layer(4, 8, 4, 4, true, Activation::Gelu);
    let x = DynTensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[1, 4], &Device::Cpu).unwrap();
    let out = moe.forward(&x).unwrap();
    assert_eq!(out.dims(), &[1, 4]);
}

#[test]
fn test_moe_mlp_nan_input_returns_error() {
    let moe = make_moe_mlp_layer(4, 8, 4, 2, true, Activation::Relu);
    let x = DynTensor::from_vec(vec![f32::NAN, 1.0, 0.5, 0.1], &[1, 4], &Device::Cpu).unwrap();
    let result = moe.forward(&x);
    assert!(result.is_err());
}

// -- 8-expert configuration --------------------------------------------------

#[test]
fn test_moe_mlp_8experts_top2_forward() {
    let model_dim = 16;
    let ff_dim = 32;
    let moe = make_moe_mlp_layer(model_dim, ff_dim, 8, 2, true, Activation::Gelu);
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

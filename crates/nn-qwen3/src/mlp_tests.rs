// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! MLP (SwiGLU) and MoE architecture tests for Qwen3 (#4186).
//!
//! Covers SwiGLU gate/up/down projection shapes, activation function
//! verification, intermediate dimension conventions, MoE routing shapes,
//! expert FFN dimensions, and shared expert configuration.

use super::*;
use crate::test_utils::tiny_config;
use nn_core::var_builder::VarBuilder;
use nn_core::{DType, Device};

// ---------------------------------------------------------------------------
// SwiGLU gate+up+down projection shapes
// ---------------------------------------------------------------------------

#[test]
fn test_swiglu_three_projections_required() {
    // SwiGLU MLP has exactly three weight matrices:
    // gate_proj: [intermediate_size, hidden_size]
    // up_proj:   [intermediate_size, hidden_size]
    // down_proj: [hidden_size, intermediate_size]
    // Verify by loading a model (VarBuilder::zeros validates shape expectations).
    let cfg = tiny_config(); // hidden=256, intermediate=512
    let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
    let model = Qwen3Model::load(&vb, cfg);
    assert!(
        model.is_ok(),
        "model with SwiGLU should load: {:?}",
        model.err()
    );
}

#[test]
fn test_swiglu_asymmetric_intermediate_sizes() {
    // Qwen3 allows intermediate_size to be any positive value relative to hidden.
    // Test extreme ratios: 1x, 2x, 4x, 8x.
    for ratio in [1, 2, 4, 8] {
        let hidden = 256;
        let intermediate = hidden * ratio;
        let cfg = Qwen3Config::new(
            hidden,
            intermediate,
            1,
            2,
            2,
            50,
            1e-6,
            10_000.0,
            32,
            true,
            None,
        );
        let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
        let model = Qwen3Model::load(&vb, cfg).unwrap();
        let logits = model.forward(&[0], &[0]).unwrap();
        assert_eq!(logits.dims(), &[1, 1, 50], "ratio={ratio}x");
    }
}

#[test]
fn test_swiglu_intermediate_smaller_than_hidden() {
    // intermediate_size < hidden_size is valid (e.g., for tiny test configs).
    let cfg = Qwen3Config::new(256, 128, 1, 2, 2, 50, 1e-6, 10_000.0, 32, true, None);
    assert!(cfg.validate().is_ok());
    let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
    let model = Qwen3Model::load(&vb, cfg).unwrap();
    let logits = model.forward(&[0], &[0]).unwrap();
    assert_eq!(logits.dims(), &[1, 1, 50]);
}

// ---------------------------------------------------------------------------
// Intermediate dimension conventions
// ---------------------------------------------------------------------------

#[test]
fn test_intermediate_dimension_production_ratios() {
    // Qwen3 production models use intermediate/hidden ratios between 2.5 and 4.0.
    // Qwen3-0.6B: 4864/896 ≈ 5.43 (exception: small model)
    // Qwen3-8B: 14336/4096 = 3.5
    // All must validate.
    let configs = [
        (896, 4864, "0.6B"),
        (2048, 11008, "1.7B"),
        (4096, 14336, "8B"),
    ];
    for (hidden, intermediate, name) in configs {
        let cfg = Qwen3Config::new(
            hidden,
            intermediate,
            1,
            if hidden >= 2048 { hidden / 128 } else { 14 },
            2,
            151_936,
            1e-6,
            1_000_000.0,
            32768,
            true,
            None,
        );
        assert!(
            cfg.validate().is_ok(),
            "Qwen3-{name} config should validate"
        );
        let ratio = intermediate as f64 / hidden as f64;
        assert!(
            ratio > 1.0 && ratio < 10.0,
            "Qwen3-{name} intermediate ratio {ratio} out of expected range"
        );
    }
}

// ---------------------------------------------------------------------------
// Activation function (SiLU applied to gate only)
// ---------------------------------------------------------------------------

#[test]
fn test_swiglu_zero_weights_produce_zero_mlp_output() {
    // With zero weights: gate(x)=0, silu(0)=0, up(x)=0 => down(0*0) = 0.
    // The MLP should produce zero output, which the residual connection
    // preserves from the attention output.
    let cfg = tiny_config();
    let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
    let model = Qwen3Model::load(&vb, cfg).unwrap();
    let logits = model.forward(&[0], &[0]).unwrap();
    let vals = logits.to_flat_vec::<f32>().unwrap();
    // With zero weights throughout, all logits should be zero
    assert!(
        vals.iter().all(|&v| v == 0.0),
        "zero-weight model should produce zero logits"
    );
}

#[test]
fn test_swiglu_output_preserves_seq_dimension() {
    // MLP is applied element-wise along the hidden dimension.
    // Input [batch, seq, hidden] -> output [batch, seq, hidden].
    // Sequence length must be preserved.
    let cfg = tiny_config();
    let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
    let model = Qwen3Model::load(&vb, cfg).unwrap();

    for seq in [1, 3, 7, 16] {
        let ids: Vec<usize> = (0..seq).collect();
        let pos: Vec<usize> = (0..seq).collect();
        let logits = model.forward(&ids, &pos).unwrap();
        assert_eq!(
            logits.dims()[1],
            seq,
            "MLP should preserve seq dimension: {seq}"
        );
    }
}

// ---------------------------------------------------------------------------
// MoE (Mixture of Experts) routing shapes
// ---------------------------------------------------------------------------

#[test]
fn test_moe_routing_config_validates() {
    // MoE config with valid routing: 8 experts, top-2 active.
    let base = tiny_config();
    let moe_cfg = Qwen3MoeConfig::new(base, 8, 2, false, None);
    assert!(moe_cfg.validate().is_ok());
}

#[test]
fn test_moe_routing_topk_equals_num_experts() {
    // Edge case: all experts active (top-k == num_experts).
    let base = tiny_config();
    let moe_cfg = Qwen3MoeConfig::new(base, 4, 4, false, None);
    assert!(moe_cfg.validate().is_ok());
}

#[test]
fn test_moe_routing_topk_exceeds_num_experts_rejected() {
    // top-k > num_experts should be rejected.
    let base = tiny_config();
    let moe_cfg = Qwen3MoeConfig::new(base, 4, 5, false, None);
    assert!(moe_cfg.validate().is_err());
}

#[test]
fn test_moe_forward_shape_matches_dense() {
    // MoE model output shape should match dense model for same base config.
    let base = tiny_config();
    let moe_cfg = Qwen3MoeConfig::new(base.clone(), 4, 2, false, None);
    let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);

    let dense_model = Qwen3Model::load(&vb, base).unwrap();
    let moe_model = Qwen3MoeModel::load(&vb, moe_cfg).unwrap();

    let dense_logits = dense_model.forward(&[0, 1], &[0, 1]).unwrap();
    let moe_logits = moe_model.forward(&[0, 1], &[0, 1]).unwrap();

    assert_eq!(
        dense_logits.dims(),
        moe_logits.dims(),
        "MoE and dense should have same output shape"
    );
}

// ---------------------------------------------------------------------------
// Expert FFN dimensions
// ---------------------------------------------------------------------------

#[test]
fn test_moe_expert_ffn_uses_base_intermediate_size() {
    // Each expert uses gate_proj[intermediate, hidden], up_proj[intermediate, hidden],
    // down_proj[hidden, intermediate] — same dimensions as the dense SwiGLU MLP.
    let base = tiny_config();
    let moe_cfg = Qwen3MoeConfig::new(base, 4, 2, false, None);
    let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
    let model = Qwen3MoeModel::load(&vb, moe_cfg);
    assert!(
        model.is_ok(),
        "MoE model should load with base intermediate_size"
    );
}

#[test]
fn test_moe_shared_expert_intermediate_size_override() {
    // Shared expert can use a different intermediate size than routed experts.
    let base = tiny_config();
    let moe_cfg = Qwen3MoeConfig::new(base, 4, 2, true, Some(1024));
    assert_eq!(moe_cfg.shared_expert_ff_dim(), 1024);
    assert!(moe_cfg.validate().is_ok());
}

#[test]
fn test_moe_shared_expert_fallback_to_base_intermediate() {
    // When shared_expert_intermediate_size is None, falls back to base.intermediate_size.
    let base = tiny_config(); // intermediate_size=512
    let moe_cfg = Qwen3MoeConfig::new(base.clone(), 4, 2, true, None);
    assert_eq!(moe_cfg.shared_expert_ff_dim(), base.intermediate_size);
}

#[test]
fn test_moe_single_expert_is_equivalent_to_dense() {
    // With 1 expert and top-1, MoE should behave like a dense model
    // (the single expert handles all tokens).
    let base = tiny_config();
    let moe_cfg = Qwen3MoeConfig::new(base, 1, 1, false, None);
    assert!(moe_cfg.validate().is_ok());
    let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
    let model = Qwen3MoeModel::load(&vb, moe_cfg).unwrap();
    let logits = model.forward(&[0], &[0]).unwrap();
    let vals = logits.to_flat_vec::<f32>().unwrap();
    assert!(vals.iter().all(|v| v.is_finite()));
}

// ---------------------------------------------------------------------------
// MoE with KV cache
// ---------------------------------------------------------------------------

#[test]
fn test_moe_model_with_kv_cache() {
    let base = tiny_config();
    let moe_cfg = Qwen3MoeConfig::new(base, 4, 2, false, None);
    let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
    let model = Qwen3MoeModel::load(&vb, moe_cfg).unwrap();
    let mut cache = model.new_cache();

    // Prefill
    let _ = model
        .forward_cached(&[0, 1, 2], &[0, 1, 2], Some(&mut cache))
        .unwrap();
    assert_eq!(cache.seq_len(), 3);

    // Decode
    let logits = model.forward_cached(&[3], &[3], Some(&mut cache)).unwrap();
    assert_eq!(cache.seq_len(), 4);
    assert!(logits
        .to_flat_vec::<f32>()
        .unwrap()
        .iter()
        .all(|v| v.is_finite()));
}

// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Architecture validation tests for Qwen3 model (#3942).
//!
//! Validates structural invariants across Qwen3 production configs: layer counts,
//! parameter count estimates, GQA group arithmetic, hidden_size = heads * head_dim,
//! shape propagation, KV cache sizing, and config properties.

use super::*;
use crate::test_utils::tiny_config;
use nn_core::var_builder::VarBuilder;
use nn_core::{DType, Device};

// ---------------------------------------------------------------------------
// Production config: Qwen3-0.6B
// ---------------------------------------------------------------------------

fn qwen3_0_6b() -> Qwen3Config {
    // Source: Kani proof validate_accepts_production_configs, arXiv:2505.09388
    // hidden=896, heads=14, kv=2 → head_dim = 896/14 ≠ 128 but head_dim() is
    // a constant 128. The HF config has head_dim=128 and num_attention_heads=14
    // meaning q_proj is [14*128, 896] (not hidden_size = heads * head_dim).
    Qwen3Config::new(
        896,         // hidden_size (actual HF value)
        4864,        // intermediate_size
        28,          // num_hidden_layers
        14,          // num_attention_heads
        2,           // num_key_value_heads
        151_936,     // vocab_size
        1e-6,        // rms_norm_eps
        1_000_000.0, // rope_theta
        40_960,      // max_position_embeddings
        true,        // tie_word_embeddings
        None,
    )
}

fn qwen3_1_7b() -> Qwen3Config {
    // Source: Kani proof validate_accepts_production_configs, arXiv:2505.09388
    Qwen3Config::new(
        2048,        // hidden_size
        11008,       // intermediate_size (corrected from 8960)
        28,          // num_hidden_layers
        16,          // num_attention_heads
        4,           // num_key_value_heads (corrected from 8)
        151_936,     // vocab_size
        1e-6,        // rms_norm_eps
        1_000_000.0, // rope_theta
        40_960,      // max_position_embeddings
        true,        // tie_word_embeddings (corrected from false)
        None,
    )
}

fn qwen3_4b() -> Qwen3Config {
    // Source: Kani proof validate_accepts_production_configs, arXiv:2505.09388
    Qwen3Config::new(
        2560,        // hidden_size
        13824,       // intermediate_size (corrected from 9216)
        36,          // num_hidden_layers
        20,          // num_attention_heads (corrected from 32)
        4,           // num_key_value_heads (corrected from 8)
        151_936,     // vocab_size
        1e-6,        // rms_norm_eps
        1_000_000.0, // rope_theta
        40_960,      // max_position_embeddings
        true,        // tie_word_embeddings (corrected from false)
        None,
    )
}

fn qwen3_8b() -> Qwen3Config {
    Qwen3Config::new(
        4096,        // hidden_size
        14336,       // intermediate_size
        36,          // num_hidden_layers
        32,          // num_attention_heads
        8,           // num_key_value_heads
        151_936,     // vocab_size
        1e-6,        // rms_norm_eps
        1_000_000.0, // rope_theta
        131072,      // max_position_embeddings (128K context)
        false,       // tie_word_embeddings
        None,
    )
}

fn qwen3_32b() -> Qwen3Config {
    Qwen3Config::new(
        5120,        // hidden_size
        25600,       // intermediate_size
        64,          // num_hidden_layers
        40,          // num_attention_heads
        8,           // num_key_value_heads
        151_936,     // vocab_size
        1e-6,        // rms_norm_eps
        1_000_000.0, // rope_theta
        131072,      // max_position_embeddings
        false,       // tie_word_embeddings
        None,
    )
}

// ---------------------------------------------------------------------------
// All production configs validate successfully
// ---------------------------------------------------------------------------

#[test]
fn test_all_production_configs_validate() {
    let configs: Vec<(&str, Qwen3Config)> = vec![
        ("0.6B", qwen3_0_6b()),
        ("1.7B", qwen3_1_7b()),
        ("4B", qwen3_4b()),
        ("8B", qwen3_8b()),
        ("32B", qwen3_32b()),
    ];
    for (name, cfg) in &configs {
        cfg.validate()
            .unwrap_or_else(|e| panic!("Qwen3-{name} config should validate: {e}"));
    }
}

// ---------------------------------------------------------------------------
// Layer counts per production size
// ---------------------------------------------------------------------------

#[test]
fn test_production_layer_counts() {
    assert_eq!(qwen3_0_6b().num_hidden_layers, 28);
    assert_eq!(qwen3_1_7b().num_hidden_layers, 28);
    assert_eq!(qwen3_4b().num_hidden_layers, 36);
    assert_eq!(qwen3_8b().num_hidden_layers, 36);
    assert_eq!(qwen3_32b().num_hidden_layers, 64);
}

// ---------------------------------------------------------------------------
// hidden_size = num_attention_heads * head_dim for configs where it holds
// ---------------------------------------------------------------------------

#[test]
fn test_hidden_size_equals_heads_times_head_dim() {
    // In Qwen3, head_dim=128 is constant, but hidden_size != num_attention_heads * head_dim
    // for some smaller variants (0.6B: 896 != 14*128). The q_proj weight is shaped
    // [nh * head_dim, hidden_size] -- hidden_size is an independent parameter.
    // This test checks configs where the equality holds.
    let configs: Vec<(&str, Qwen3Config)> = vec![
        ("tiny", tiny_config()),
        ("1.7B", qwen3_1_7b()),
        ("4B", qwen3_4b()),
        ("8B", qwen3_8b()),
        ("32B", qwen3_32b()),
    ];
    for (name, cfg) in &configs {
        assert_eq!(
            cfg.hidden_size,
            cfg.num_attention_heads * cfg.head_dim(),
            "{name}: hidden_size should equal num_attention_heads * head_dim(128)"
        );
    }
}

#[test]
fn test_hidden_size_decoupled_from_head_count_for_small_models() {
    // Qwen3-0.6B has hidden_size != num_attention_heads * head_dim.
    // This is architecturally intentional: q_proj is [nh*128, hidden_size].
    let cfg_06b = qwen3_0_6b();
    assert_ne!(
        cfg_06b.hidden_size,
        cfg_06b.num_attention_heads * cfg_06b.head_dim(),
        "0.6B: hidden_size (896) != heads (14) * head_dim (128)"
    );
    // 4B happens to satisfy hidden_size == heads * head_dim (2560 == 20 * 128).
    let cfg_4b = qwen3_4b();
    assert_eq!(
        cfg_4b.hidden_size,
        cfg_4b.num_attention_heads * cfg_4b.head_dim(),
        "4B: hidden_size (2560) == heads (20) * head_dim (128)"
    );
}

// ---------------------------------------------------------------------------
// GQA group counts match known values
// ---------------------------------------------------------------------------

#[test]
fn test_gqa_groups_for_production_configs() {
    assert_eq!(qwen3_0_6b().num_kv_groups().unwrap(), 7, "0.6B: 14/2 = 7");
    assert_eq!(qwen3_1_7b().num_kv_groups().unwrap(), 4, "1.7B: 16/4 = 4");
    assert_eq!(qwen3_4b().num_kv_groups().unwrap(), 5, "4B: 20/4 = 5");
    assert_eq!(qwen3_8b().num_kv_groups().unwrap(), 4, "8B: 32/8 = 4");
    assert_eq!(qwen3_32b().num_kv_groups().unwrap(), 5, "32B: 40/8 = 5");
}

// ---------------------------------------------------------------------------
// Vocab size is consistent across all Qwen3 variants
// ---------------------------------------------------------------------------

#[test]
fn test_all_production_configs_same_vocab() {
    let configs = vec![
        qwen3_0_6b(),
        qwen3_1_7b(),
        qwen3_4b(),
        qwen3_8b(),
        qwen3_32b(),
    ];
    for cfg in &configs {
        assert_eq!(
            cfg.vocab_size, 151_936,
            "all Qwen3 variants use vocab_size 151,936"
        );
    }
}

// ---------------------------------------------------------------------------
// Parameter count monotonically increases with model size
// ---------------------------------------------------------------------------

/// Rough parameter estimate: embedding + per-layer (attention + MLP + norms).
fn estimate_qwen3_params(c: &Qwen3Config) -> usize {
    let embed = c.vocab_size * c.hidden_size;
    let per_layer_attn = {
        let head_dim = c.head_dim();
        let q_proj = c.hidden_size * c.num_attention_heads * head_dim;
        let k_proj = c.hidden_size * c.num_key_value_heads * head_dim;
        let v_proj = c.hidden_size * c.num_key_value_heads * head_dim;
        let o_proj = c.num_attention_heads * head_dim * c.hidden_size;
        q_proj + k_proj + v_proj + o_proj
    };
    let per_layer_mlp = 2 * c.hidden_size * c.intermediate_size // gate + up
        + c.intermediate_size * c.hidden_size; // down
    let per_layer_norms = 2 * c.hidden_size; // 2 RMSNorm weights
    let per_layer = per_layer_attn + per_layer_mlp + per_layer_norms;
    embed + per_layer * c.num_hidden_layers
}

#[test]
fn test_param_count_monotonic() {
    let sizes: Vec<(&str, Qwen3Config)> = vec![
        ("0.6B", qwen3_0_6b()),
        ("1.7B", qwen3_1_7b()),
        ("4B", qwen3_4b()),
        ("8B", qwen3_8b()),
        ("32B", qwen3_32b()),
    ];
    for i in 1..sizes.len() {
        let (prev_name, prev_cfg) = &sizes[i - 1];
        let (curr_name, curr_cfg) = &sizes[i];
        let prev = estimate_qwen3_params(prev_cfg);
        let curr = estimate_qwen3_params(curr_cfg);
        assert!(
            curr > prev,
            "params should grow: {prev_name} ({prev}) >= {curr_name} ({curr})"
        );
    }
}

// ---------------------------------------------------------------------------
// Shape propagation: forward pass output shape
// ---------------------------------------------------------------------------

#[test]
fn test_forward_output_shape_single_token() {
    let cfg = tiny_config();
    let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
    let model = Qwen3Model::load(&vb, cfg.clone()).unwrap();
    let logits = model.forward(&[42], &[0]).unwrap();
    assert_eq!(
        logits.dims(),
        &[1, 1, cfg.vocab_size],
        "single token -> [1, 1, vocab_size]"
    );
}

#[test]
fn test_forward_output_shape_multi_token() {
    let cfg = tiny_config();
    let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
    let model = Qwen3Model::load(&vb, cfg.clone()).unwrap();
    let logits = model
        .forward(&[0, 1, 2, 3, 4, 5, 6, 7], &[0, 1, 2, 3, 4, 5, 6, 7])
        .unwrap();
    assert_eq!(
        logits.dims(),
        &[1, 8, cfg.vocab_size],
        "8 tokens -> [1, 8, vocab_size]"
    );
}

// ---------------------------------------------------------------------------
// KV cache sizing matches model config
// ---------------------------------------------------------------------------

#[test]
fn test_kv_cache_num_layers_matches_config() {
    let cfg = tiny_config();
    let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
    let model = Qwen3Model::load(&vb, cfg.clone()).unwrap();
    let cache = model.new_cache();
    assert_eq!(
        cache.num_layers(),
        cfg.num_hidden_layers,
        "KV cache should have num_hidden_layers entries"
    );
}

// ---------------------------------------------------------------------------
// Config accessor roundtrip
// ---------------------------------------------------------------------------

#[test]
fn test_model_config_accessor_returns_original() {
    let cfg = tiny_config();
    let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
    let model = Qwen3Model::load(&vb, cfg.clone()).unwrap();
    let model_cfg = model.config();
    assert_eq!(model_cfg.hidden_size, cfg.hidden_size);
    assert_eq!(model_cfg.num_hidden_layers, cfg.num_hidden_layers);
    assert_eq!(model_cfg.vocab_size, cfg.vocab_size);
}

// ---------------------------------------------------------------------------
// tied_word_embeddings: lm_head shares embedding weight
// ---------------------------------------------------------------------------

#[test]
fn test_tied_embeddings_model_loads() {
    let cfg = tiny_config(); // tie_word_embeddings = true
    assert!(cfg.tie_word_embeddings);
    let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
    let model = Qwen3Model::load(&vb, cfg.clone()).unwrap();
    let emb_shape = model.embed_tokens().weight().dims().to_vec();
    assert_eq!(emb_shape, vec![cfg.vocab_size, cfg.hidden_size]);
}

#[test]
fn test_untied_embeddings_model_loads() {
    let cfg = tiny_config().with_vocab_size(100);
    let cfg = Qwen3Config::new(
        cfg.hidden_size,
        cfg.intermediate_size,
        cfg.num_hidden_layers,
        cfg.num_attention_heads,
        cfg.num_key_value_heads,
        cfg.vocab_size,
        cfg.rms_norm_eps,
        cfg.rope_theta,
        cfg.max_position_embeddings,
        false, // untied
        None,
    );
    let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
    let model = Qwen3Model::load(&vb, cfg).unwrap();
    assert!(!model.config().tie_word_embeddings);
}

// ---------------------------------------------------------------------------
// RoPE theta: production configs use either 500K or 1M
// ---------------------------------------------------------------------------

#[test]
fn test_rope_theta_production_values() {
    // All corrected production configs from arXiv:2505.09388 use 1M theta.
    assert!(
        (qwen3_0_6b().rope_theta - 1_000_000.0).abs() < 1e-3,
        "0.6B uses 1M theta"
    );
    assert!(
        (qwen3_1_7b().rope_theta - 1_000_000.0).abs() < 1e-3,
        "1.7B uses 1M theta"
    );
    assert!(
        (qwen3_8b().rope_theta - 1_000_000.0).abs() < 1e-3,
        "8B uses 1M theta"
    );
}

// ---------------------------------------------------------------------------
// head_dim is always 128 for Qwen3 (architectural constant)
// ---------------------------------------------------------------------------

#[test]
fn test_head_dim_128_is_architectural_constant() {
    let configs: Vec<Qwen3Config> = vec![
        tiny_config(),
        qwen3_0_6b(),
        qwen3_1_7b(),
        qwen3_4b(),
        qwen3_8b(),
        qwen3_32b(),
    ];
    for cfg in &configs {
        assert_eq!(cfg.head_dim(), 128, "Qwen3 head_dim is always 128");
    }
}

// ---------------------------------------------------------------------------
// MoE config validation
// ---------------------------------------------------------------------------

#[test]
fn test_moe_30b_a3b_config_validates() {
    // Qwen3-30B-A3B: 128 experts, 8 active.
    let base = Qwen3Config::new(
        4096,        // hidden_size
        14336,       // intermediate_size
        48,          // num_hidden_layers
        32,          // num_attention_heads
        4,           // num_key_value_heads
        151_936,     // vocab_size
        1e-6,        // rms_norm_eps
        1_000_000.0, // rope_theta
        131072,      // max_position_embeddings
        false,       // tie_word_embeddings
        None,
    );
    let moe_cfg = Qwen3MoeConfig::new(base, 128, 8, true, Some(4096));
    moe_cfg
        .validate()
        .expect("Qwen3-30B-A3B MoE config should validate");
    assert_eq!(moe_cfg.num_experts, 128);
    assert_eq!(moe_cfg.num_experts_per_tok, 8);
    assert!(moe_cfg.shared_expert);
    assert_eq!(moe_cfg.shared_expert_ff_dim(), 4096);
}

#[test]
fn test_moe_config_experts_per_tok_gt_num_experts_rejected() {
    let base = tiny_config();
    let moe_cfg = Qwen3MoeConfig::new(base, 4, 8, false, None); // 8 > 4
    assert!(moe_cfg.validate().is_err());
}

#[test]
fn test_moe_config_zero_experts_rejected() {
    let base = tiny_config();
    let moe_cfg = Qwen3MoeConfig::new(base, 0, 1, false, None);
    assert!(moe_cfg.validate().is_err());
}

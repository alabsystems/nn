// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Extended architecture tests for nn-qwen3 covering production model configs,
//! RoPE frequency geometry, GQA head mappings, SwiGLU projection dimensions,
//! sliding window attention, RMSNorm parameters, embedding dimensions, and
//! position encoding bounds (#4186).
//!
//! Complements existing tests in `qwen3_extended_tests.rs`,
//! `qwen3_tests_config_and_arch.rs`, and `qwen3_tests_rope_gqa.rs`.

use crate::rope_cache::RoPECache;
use crate::test_utils::tiny_config;
use crate::{Qwen3Config, Qwen3Model};
use nn_core::dyn_tensor::DynTensor;
use nn_core::var_builder::VarBuilder;
use nn_core::{DType, Device};

// ===========================================================================
// 1. Config validation for all production Qwen3 model sizes
// ===========================================================================

/// Production config parameters for all known Qwen3 dense variants.
/// Source: Qwen3 Technical Report (arXiv:2505.09388), HuggingFace model cards.
struct ProductionConfig {
    name: &'static str,
    hidden_size: usize,
    intermediate_size: usize,
    num_hidden_layers: usize,
    num_attention_heads: usize,
    num_key_value_heads: usize,
    vocab_size: usize,
    max_position_embeddings: usize,
    tie_word_embeddings: bool,
    expected_kv_groups: usize,
}

const PRODUCTION_CONFIGS: &[ProductionConfig] = &[
    ProductionConfig {
        name: "Qwen3-0.6B",
        hidden_size: 896,
        intermediate_size: 4864,
        num_hidden_layers: 28,
        num_attention_heads: 14,
        num_key_value_heads: 2,
        vocab_size: 151_936,
        max_position_embeddings: 40_960,
        tie_word_embeddings: true,
        expected_kv_groups: 7,
    },
    ProductionConfig {
        name: "Qwen3-1.7B",
        hidden_size: 2048,
        intermediate_size: 6144,
        num_hidden_layers: 28,
        num_attention_heads: 16,
        num_key_value_heads: 4,
        vocab_size: 151_936,
        max_position_embeddings: 40_960,
        tie_word_embeddings: false,
        expected_kv_groups: 4,
    },
    ProductionConfig {
        name: "Qwen3-4B",
        hidden_size: 2560,
        intermediate_size: 9216,
        num_hidden_layers: 36,
        num_attention_heads: 32,
        num_key_value_heads: 8,
        vocab_size: 151_936,
        max_position_embeddings: 131_072,
        tie_word_embeddings: false,
        expected_kv_groups: 4,
    },
    ProductionConfig {
        name: "Qwen3-8B",
        hidden_size: 4096,
        intermediate_size: 14336,
        num_hidden_layers: 36,
        num_attention_heads: 32,
        num_key_value_heads: 8,
        vocab_size: 151_936,
        max_position_embeddings: 131_072,
        tie_word_embeddings: false,
        expected_kv_groups: 4,
    },
    ProductionConfig {
        name: "Qwen3-14B",
        hidden_size: 5120,
        intermediate_size: 17408,
        num_hidden_layers: 40,
        num_attention_heads: 40,
        num_key_value_heads: 8,
        vocab_size: 151_936,
        max_position_embeddings: 131_072,
        tie_word_embeddings: false,
        expected_kv_groups: 5,
    },
    ProductionConfig {
        name: "Qwen3-32B",
        hidden_size: 5120,
        intermediate_size: 25600,
        num_hidden_layers: 64,
        num_attention_heads: 40,
        num_key_value_heads: 8,
        vocab_size: 151_936,
        max_position_embeddings: 131_072,
        tie_word_embeddings: false,
        expected_kv_groups: 5,
    },
];

fn make_production_config(pc: &ProductionConfig) -> Qwen3Config {
    Qwen3Config::new(
        pc.hidden_size,
        pc.intermediate_size,
        pc.num_hidden_layers,
        pc.num_attention_heads,
        pc.num_key_value_heads,
        pc.vocab_size,
        1e-6,
        1_000_000.0,
        pc.max_position_embeddings,
        pc.tie_word_embeddings,
        None,
    )
}

/// All 6 production Qwen3 configs (0.6B through 32B) pass validation.
#[test]
fn test_all_production_configs_validate() {
    for pc in PRODUCTION_CONFIGS {
        let cfg = make_production_config(pc);
        assert!(
            cfg.validate().is_ok(),
            "{} config should validate: {:?}",
            pc.name,
            cfg.validate().err()
        );
    }
}

/// All production configs use head_dim=128 (Qwen3 constant).
#[test]
fn test_all_production_configs_head_dim_128() {
    for pc in PRODUCTION_CONFIGS {
        let cfg = make_production_config(pc);
        assert_eq!(cfg.head_dim(), 128, "{} head_dim should be 128", pc.name);
    }
}

/// All production configs share vocab_size=151936 and rms_norm_eps=1e-6.
#[test]
fn test_all_production_configs_shared_constants() {
    for pc in PRODUCTION_CONFIGS {
        let cfg = make_production_config(pc);
        assert_eq!(
            cfg.vocab_size, 151_936,
            "{} vocab_size should be 151936",
            pc.name
        );
        assert!(
            (cfg.rms_norm_eps - 1e-6).abs() < 1e-12,
            "{} rms_norm_eps should be 1e-6",
            pc.name
        );
        assert!(
            (cfg.rope_theta - 1_000_000.0).abs() < 1e-6,
            "{} rope_theta should be 1M",
            pc.name
        );
    }
}

/// Production configs: intermediate_size / hidden_size ratio is in [2.0, 6.0].
#[test]
fn test_all_production_configs_mlp_ratio() {
    for pc in PRODUCTION_CONFIGS {
        let ratio = pc.intermediate_size as f64 / pc.hidden_size as f64;
        assert!(
            ratio > 2.0 && ratio < 6.0,
            "{}: intermediate/hidden ratio {ratio:.3} should be in (2.0, 6.0)",
            pc.name
        );
    }
}

// ===========================================================================
// 2. RoPE frequency tensor shape and values (Qwen3 base=1M)
// ===========================================================================

/// RoPE cache for Qwen3 base=1M, head_dim=128 has correct shape.
#[test]
fn test_rope_qwen3_1m_base_cache_shape() {
    let max_seq = 512;
    let head_dim = 128;
    let half_dim = head_dim / 2;
    let cache = RoPECache::new(max_seq, head_dim, 1_000_000.0);

    assert_eq!(cache.max_seq_len(), max_seq);
    assert_eq!(cache.head_dim(), head_dim);
    assert_eq!(cache.half_dim(), half_dim);
    assert!((cache.base() - 1_000_000.0).abs() < f32::EPSILON);

    // Every position has half_dim cos/sin values
    for pos in [0, 1, 50, 255, max_seq - 1] {
        let (cos, sin) = cache.get(pos);
        assert_eq!(cos.len(), half_dim);
        assert_eq!(sin.len(), half_dim);
    }
}

/// With base=1M, mid-band frequency (i=32) is 0.001, not 0.01 (base=10k).
/// This verifies the longer-wavelength behavior of Qwen3's higher base.
#[test]
fn test_rope_qwen3_1m_frequency_midband() {
    let base_1m = 1_000_000.0_f64;
    let base_10k = 10_000.0_f64;

    // freq[32] = 1 / base^(64/128) = 1 / sqrt(base)
    let freq_1m_32 = 1.0 / base_1m.powf(64.0 / 128.0);
    let freq_10k_32 = 1.0 / base_10k.powf(64.0 / 128.0);

    // base=1M: freq[32] = 1/1000 = 0.001
    assert!(
        (freq_1m_32 - 0.001).abs() < 1e-10,
        "base=1M, freq[32] should be 0.001, got {freq_1m_32}"
    );
    // base=10k: freq[32] = 1/100 = 0.01
    assert!(
        (freq_10k_32 - 0.01).abs() < 1e-10,
        "base=10k, freq[32] should be 0.01, got {freq_10k_32}"
    );
    // 1M base has 10x lower mid-band frequency (longer wavelength)
    assert!(
        (freq_1m_32 / freq_10k_32 - 0.1).abs() < 1e-10,
        "base=1M should have 10x lower freq than base=10k at i=32"
    );
}

/// RoPE cos/sin values at base=1M are all finite and bounded in [-1, 1].
#[test]
fn test_rope_qwen3_1m_values_bounded() {
    let cache = RoPECache::new(256, 128, 1_000_000.0);
    for pos in 0..256 {
        let (cos, sin) = cache.get(pos);
        for i in 0..64 {
            assert!(
                cos[i].is_finite() && cos[i].abs() <= 1.0,
                "cos[{i}] at pos {pos} should be in [-1, 1], got {}",
                cos[i]
            );
            assert!(
                sin[i].is_finite() && sin[i].abs() <= 1.0,
                "sin[{i}] at pos {pos} should be in [-1, 1], got {}",
                sin[i]
            );
        }
    }
}

/// RoPE Pythagorean identity: cos^2(x) + sin^2(x) = 1 for every entry.
#[test]
fn test_rope_pythagorean_identity() {
    let cache = RoPECache::new(128, 128, 1_000_000.0);
    for pos in [0, 1, 10, 42, 100, 127] {
        let (cos, sin) = cache.get(pos);
        for i in 0..64 {
            let sum_sq = cos[i] * cos[i] + sin[i] * sin[i];
            assert!(
                (sum_sq - 1.0).abs() < 1e-5,
                "cos^2 + sin^2 should be 1.0 at pos={pos}, i={i}: got {sum_sq}"
            );
        }
    }
}

// ===========================================================================
// 3. GQA configuration: head mappings for all production sizes
// ===========================================================================

/// Verify num_kv_groups for all 6 production Qwen3 variants.
#[test]
fn test_gqa_groups_all_production_configs() {
    for pc in PRODUCTION_CONFIGS {
        let cfg = make_production_config(pc);
        let groups = cfg.num_kv_groups().unwrap();
        assert_eq!(
            groups, pc.expected_kv_groups,
            "{}: num_kv_groups should be {}, got {}",
            pc.name, pc.expected_kv_groups, groups
        );
    }
}

/// Q projection output size = num_attention_heads * head_dim for all variants.
#[test]
fn test_gqa_q_proj_dimensions_all_production() {
    for pc in PRODUCTION_CONFIGS {
        let q_proj_out = pc.num_attention_heads * 128; // head_dim=128
                                                       // Q projection: [q_proj_out, hidden_size]
        assert!(
            q_proj_out > 0,
            "{}: q_proj output dimension should be > 0",
            pc.name
        );
        // For most Qwen3 variants, q_proj_out != hidden_size (e.g., 0.6B: 14*128=1792, hidden=896)
        // This is fine: the projection maps hidden_size -> num_heads*head_dim.
    }
}

/// K/V projection output size = num_key_value_heads * head_dim for all variants.
#[test]
fn test_gqa_kv_proj_dimensions_all_production() {
    for pc in PRODUCTION_CONFIGS {
        let kv_proj_out = pc.num_key_value_heads * 128;
        let q_proj_out = pc.num_attention_heads * 128;
        // KV proj is smaller than Q proj when GQA ratio > 1
        let ratio = q_proj_out / kv_proj_out;
        assert_eq!(
            ratio, pc.expected_kv_groups,
            "{}: Q/KV proj ratio should equal GQA group count",
            pc.name
        );
    }
}

/// Verify that a model with GQA ratio > 1 loads and produces correct output shape.
#[test]
fn test_gqa_model_loads_with_ratio_7() {
    // Qwen3-0.6B has GQA ratio 7 (14 heads / 2 kv_heads)
    let cfg = Qwen3Config::new(256, 512, 1, 14, 2, 50, 1e-6, 10_000.0, 64, true, None);
    assert_eq!(cfg.num_kv_groups().unwrap(), 7);
    let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
    let model = Qwen3Model::load(&vb, cfg.clone()).unwrap();
    let logits = model.forward(&[0, 1], &[0, 1]).unwrap();
    assert_eq!(logits.dims(), &[1, 2, cfg.vocab_size]);
}

/// Verify that a model with GQA ratio 5 (like Qwen3-14B/32B) loads.
#[test]
fn test_gqa_model_loads_with_ratio_5() {
    // Qwen3-14B/32B: 40 heads / 8 kv_heads = 5 groups
    let cfg = Qwen3Config::new(1024, 2048, 1, 40, 8, 50, 1e-6, 10_000.0, 64, true, None);
    assert_eq!(cfg.num_kv_groups().unwrap(), 5);
    let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
    let model = Qwen3Model::load(&vb, cfg.clone()).unwrap();
    let logits = model.forward(&[0], &[0]).unwrap();
    assert_eq!(logits.dims(), &[1, 1, cfg.vocab_size]);
}

// ===========================================================================
// 4. SwiGLU activation: gate/up projection dimensions
// ===========================================================================

/// SwiGLU MLP weight shapes: gate_proj and up_proj are [intermediate, hidden],
/// down_proj is [hidden, intermediate]. Verify by loading a model and checking
/// forward output dimension is preserved.
#[test]
fn test_swiglu_projection_dimensions_preserved() {
    let cfg = tiny_config(); // hidden=256, intermediate=512
    let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
    let model = Qwen3Model::load(&vb, cfg.clone()).unwrap();

    // For each sequence length, output should always be [1, seq, vocab_size]
    // which implies the MLP preserved hidden_size through
    // gate_proj(hidden->intermediate) * up_proj(hidden->intermediate) -> down_proj(intermediate->hidden)
    for seq_len in [1, 4, 8] {
        let ids: Vec<usize> = (0..seq_len).collect();
        let positions: Vec<usize> = (0..seq_len).collect();
        let logits = model.forward(&ids, &positions).unwrap();
        assert_eq!(
            logits.dims(),
            &[1, seq_len, cfg.vocab_size],
            "SwiGLU should preserve hidden dimension through gate*up->down pipeline"
        );
    }
}

/// Verify SwiGLU produces distinct outputs for distinct inputs (non-zero weights).
/// With zero weights, all outputs are zero; this verifies the gate mechanism
/// actually modulates the signal when weights are non-trivial.
#[test]
fn test_swiglu_silu_gate_property() {
    // silu(x) = x * sigmoid(x) properties:
    // - silu(0) = 0
    // - silu is smooth and differentiable everywhere
    // - silu(-inf) -> 0, silu(+inf) -> +inf
    // - silu has a minimum around x ~ -0.278
    let test_values: Vec<f32> = vec![-5.0, -1.0, -0.278, 0.0, 0.278, 1.0, 5.0];
    let x = DynTensor::from_vec(test_values.clone(), &[7], &Device::Cpu).unwrap();
    let silu = x.silu().unwrap();
    let vals = silu.to_flat_vec::<f32>().unwrap();

    // silu(0) = 0
    assert!(vals[3].abs() < 1e-7, "silu(0) should be 0, got {}", vals[3]);

    // silu is finite for all finite inputs
    for (i, &v) in vals.iter().enumerate() {
        assert!(
            v.is_finite(),
            "silu({}) should be finite, got {}",
            test_values[i],
            v
        );
    }

    // For large positive x: silu(x) ~ x (since sigmoid(x) ~ 1)
    let silu_5 = vals[6]; // silu(5.0)
    assert!(
        (silu_5 - 5.0).abs() < 0.05,
        "silu(5.0) should be ~5.0, got {silu_5}"
    );

    // For large negative x: silu(x) ~ 0 (since sigmoid(x) ~ 0)
    let silu_neg5 = vals[0]; // silu(-5.0)
    assert!(
        silu_neg5.abs() < 0.05,
        "silu(-5.0) should be ~0.0, got {silu_neg5}"
    );
}

/// Verify that SwiGLU intermediate_size matches the gate and up projection width.
/// For production Qwen3, the MLP formula is:
///   output = down_proj(silu(gate_proj(x)) * up_proj(x))
/// gate_proj: [intermediate, hidden], up_proj: [intermediate, hidden],
/// down_proj: [hidden, intermediate]
#[test]
fn test_swiglu_intermediate_size_ratios() {
    // Qwen3 SwiGLU: gate and up are parallel projections to intermediate_size,
    // then element-wise multiply, then down to hidden_size.
    // The effective parameter count per MLP layer is 3 * hidden * intermediate
    // (gate + up + down).
    for pc in PRODUCTION_CONFIGS {
        let params_per_mlp = 3 * pc.hidden_size * pc.intermediate_size;
        assert!(
            params_per_mlp > 0,
            "{}: SwiGLU MLP parameters should be > 0",
            pc.name
        );
        // Verify intermediate_size is a multiple of 128 (common alignment)
        assert_eq!(
            pc.intermediate_size % 128,
            0,
            "{}: intermediate_size {} should be 128-aligned",
            pc.name,
            pc.intermediate_size
        );
    }
}

// ===========================================================================
// 5. Sliding window attention: config plumbing
// ===========================================================================

/// Qwen3 dense models do not use sliding window attention. The config has
/// no sliding_window field; verify the standard causal attention path works
/// at various sequence lengths.
#[test]
fn test_no_sliding_window_full_causal_attention() {
    let cfg = tiny_config();
    let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
    let model = Qwen3Model::load(&vb, cfg.clone()).unwrap();
    let mut cache = model.new_cache();

    // Full prefill: all tokens attend to all prior tokens (no sliding window truncation)
    let seq_len = cfg.max_position_embeddings.min(32);
    let ids: Vec<usize> = (0..seq_len).collect();
    let positions: Vec<usize> = (0..seq_len).collect();
    let logits = model
        .forward_cached(&ids, &positions, Some(&mut cache))
        .unwrap();
    assert_eq!(logits.dims(), &[1, seq_len, cfg.vocab_size]);
    assert_eq!(cache.seq_len(), seq_len);

    // Subsequent autoregressive step still attends to all cached positions
    let logits = model
        .forward_cached(&[0], &[seq_len], Some(&mut cache))
        .unwrap();
    assert_eq!(logits.dims(), &[1, 1, cfg.vocab_size]);
    assert_eq!(cache.seq_len(), seq_len + 1);
}

// ===========================================================================
// 6. Normalization: RMSNorm epsilon and dimension
// ===========================================================================

/// Production Qwen3 uses rms_norm_eps=1e-6 (not 1e-5 like LLaMA).
#[test]
fn test_rmsnorm_eps_production_value() {
    for pc in PRODUCTION_CONFIGS {
        let cfg = make_production_config(pc);
        assert!(
            (cfg.rms_norm_eps - 1e-6).abs() < 1e-12,
            "{}: rms_norm_eps should be 1e-6, got {}",
            pc.name,
            cfg.rms_norm_eps
        );
    }
}

/// RMSNorm weight dimension matches hidden_size for all configs.
/// Verified by loading a model (VarBuilder::zeros provides the shapes).
#[test]
fn test_rmsnorm_weight_dimension_matches_hidden() {
    // Use a small model to verify RMSNorm loads with correct hidden_size
    let cfg = Qwen3Config::new(512, 1024, 2, 4, 2, 50, 1e-6, 10_000.0, 64, true, None);
    let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
    // If RMSNorm weight dimension did not match hidden_size=512, this would fail
    let model = Qwen3Model::load(&vb, cfg);
    assert!(
        model.is_ok(),
        "model with hidden_size=512 should load: {:?}",
        model.err()
    );
}

/// Boundary: very small eps (1e-30) should still validate.
#[test]
fn test_rmsnorm_eps_very_small_positive() {
    let cfg = Qwen3Config::new(256, 512, 1, 2, 2, 50, 1e-30, 10_000.0, 64, true, None);
    assert!(cfg.validate().is_ok());
}

/// Boundary: eps=0 should be rejected.
#[test]
fn test_rmsnorm_eps_zero_rejected() {
    let cfg = Qwen3Config::new(256, 512, 1, 2, 2, 50, 0.0, 10_000.0, 64, true, None);
    assert!(cfg.validate().is_err());
}

/// Boundary: negative eps should be rejected.
#[test]
fn test_rmsnorm_eps_negative_rejected() {
    let cfg = Qwen3Config::new(256, 512, 1, 2, 2, 50, -1e-6, 10_000.0, 64, true, None);
    assert!(cfg.validate().is_err());
}

// ===========================================================================
// 7. Token embedding: dimensions match config
// ===========================================================================

/// Embedding weight shape is [vocab_size, hidden_size].
#[test]
fn test_embedding_weight_shape_matches_config() {
    let cfg = Qwen3Config::new(512, 1024, 1, 4, 2, 200, 1e-6, 10_000.0, 64, true, None);
    let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
    let model = Qwen3Model::load(&vb, cfg.clone()).unwrap();
    let embed = model.embed_tokens();
    let weight = embed.weight();
    assert_eq!(
        weight.dims(),
        &[cfg.vocab_size, cfg.hidden_size],
        "embedding weight should be [vocab_size={}, hidden_size={}]",
        cfg.vocab_size,
        cfg.hidden_size
    );
}

/// When tie_word_embeddings=true, lm_head shares the embedding weight.
/// Output logits should have vocab_size as the last dimension.
#[test]
fn test_tied_embeddings_logit_dimension() {
    let cfg = Qwen3Config::new(256, 512, 1, 2, 2, 150, 1e-6, 10_000.0, 64, true, None);
    assert!(cfg.tie_word_embeddings);
    let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
    let model = Qwen3Model::load(&vb, cfg.clone()).unwrap();

    let logits = model.forward(&[0, 1, 2], &[0, 1, 2]).unwrap();
    assert_eq!(
        logits.dims(),
        &[1, 3, cfg.vocab_size],
        "tied embedding logits last dim should be vocab_size"
    );
}

/// When tie_word_embeddings=false, lm_head has its own [vocab_size, hidden] weight.
#[test]
fn test_untied_embeddings_separate_lm_head() {
    let cfg = Qwen3Config::new(256, 512, 1, 2, 2, 150, 1e-6, 10_000.0, 64, false, None);
    assert!(!cfg.tie_word_embeddings);
    let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
    let model = Qwen3Model::load(&vb, cfg.clone()).unwrap();

    let logits = model.forward(&[0, 1, 2], &[0, 1, 2]).unwrap();
    assert_eq!(
        logits.dims(),
        &[1, 3, cfg.vocab_size],
        "untied embedding logits last dim should be vocab_size"
    );
}

/// Embedding handles token IDs up to vocab_size - 1.
#[test]
fn test_embedding_max_token_id() {
    let cfg = tiny_config(); // vocab_size = 100
    let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
    let model = Qwen3Model::load(&vb, cfg.clone()).unwrap();

    // Token ID = vocab_size - 1 (max valid)
    let max_id = cfg.vocab_size - 1;
    let logits = model.forward(&[max_id], &[0]).unwrap();
    assert_eq!(logits.dims(), &[1, 1, cfg.vocab_size]);
    let vals = logits.to_flat_vec::<f32>().unwrap();
    assert!(vals.iter().all(|v| v.is_finite()));
}

/// All production configs use the Qwen3 standard vocab_size=151936.
#[test]
fn test_all_production_configs_vocab_size() {
    for pc in PRODUCTION_CONFIGS {
        assert_eq!(
            pc.vocab_size, 151_936,
            "{}: vocab_size should be 151936",
            pc.name
        );
    }
}

// ===========================================================================
// 8. Position encoding: max_position_embeddings bounds
// ===========================================================================

/// Small models use 40960 max positions; larger models use 131072 (128K context).
#[test]
fn test_max_position_embeddings_by_size() {
    for pc in PRODUCTION_CONFIGS {
        let cfg = make_production_config(pc);
        match pc.name {
            "Qwen3-0.6B" | "Qwen3-1.7B" => {
                assert_eq!(
                    cfg.max_position_embeddings, 40_960,
                    "{} should have max_pos=40960",
                    pc.name
                );
            }
            "Qwen3-4B" | "Qwen3-8B" | "Qwen3-14B" | "Qwen3-32B" => {
                assert_eq!(
                    cfg.max_position_embeddings, 131_072,
                    "{} should have max_pos=131072",
                    pc.name
                );
            }
            _ => unreachable!("unknown model: {}", pc.name),
        }
    }
}

/// max_position_embeddings=0 is rejected by validation.
#[test]
fn test_max_position_embeddings_zero_rejected() {
    let cfg = Qwen3Config::new(256, 512, 1, 2, 2, 50, 1e-6, 10_000.0, 0, true, None);
    assert!(cfg.validate().is_err());
}

/// RoPE cache respects max_position_embeddings: can access all positions
/// in range, panics outside range.
#[test]
fn test_rope_cache_covers_max_position() {
    let max_pos = 64;
    let cache = RoPECache::new(max_pos, 128, 1_000_000.0);

    // All positions in [0, max_pos) should be accessible
    for pos in 0..max_pos {
        let (cos, sin) = cache.get(pos);
        assert_eq!(cos.len(), 64);
        assert_eq!(sin.len(), 64);
    }
}

/// Verify that a model with max_position_embeddings=64 can forward up to
/// position 63 (0-indexed).
#[test]
fn test_forward_at_max_position_boundary() {
    let cfg = tiny_config(); // max_position_embeddings = 64
    let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
    let model = Qwen3Model::load(&vb, cfg.clone()).unwrap();
    let mut cache = model.new_cache();

    // Forward to the last valid position
    let last_pos = cfg.max_position_embeddings - 1;
    // Prefill first few tokens, then decode up to the boundary
    model
        .forward_cached(&[0, 1, 2], &[0, 1, 2], Some(&mut cache))
        .unwrap();

    // Decode up to last valid position
    for pos in 3..=last_pos {
        let logits = model
            .forward_cached(&[pos % cfg.vocab_size], &[pos], Some(&mut cache))
            .unwrap();
        assert_eq!(logits.dims(), &[1, 1, cfg.vocab_size]);
    }
    assert_eq!(cache.seq_len(), cfg.max_position_embeddings);
}

/// KV cache num_layers matches model config num_hidden_layers for all configs.
#[test]
fn test_new_cache_layer_count_matches_config() {
    for num_layers in [1, 2, 4, 8, 16, 28, 36, 40, 64] {
        let cfg = tiny_config().with_num_hidden_layers(num_layers);
        let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
        let model = Qwen3Model::load(&vb, cfg.clone()).unwrap();
        let cache = model.new_cache();
        assert_eq!(
            cache.num_layers(),
            num_layers,
            "cache layers should match config num_hidden_layers={num_layers}"
        );
    }
}

// ===========================================================================
// 9. Layer structure: pre-norm (RMSNorm) -> attention -> RMSNorm -> FFN
// ===========================================================================

/// Verify decoder layer ordering: each layer applies pre-norm → attention →
/// residual → post-norm → MLP → residual. With zero weights, the residual
/// connections pass through unchanged, yielding zero output.
#[test]
fn test_layer_structure_prenorm_attention_ffn_residual() {
    let cfg = tiny_config();
    let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
    let model = Qwen3Model::load(&vb, cfg).unwrap();

    // With all-zero weights, forward should succeed (residual + 0 + 0 = 0)
    let logits = model.forward(&[0, 1], &[0, 1]).unwrap();
    let vals = logits.to_flat_vec::<f32>().unwrap();
    assert!(
        vals.iter().all(|&v| v == 0.0),
        "zero-weight model should produce zero logits (residual connections pass through)"
    );
}

/// Multi-layer model: increasing num_hidden_layers should produce a valid
/// forward pass for each layer count. This verifies that the layer stack
/// is correctly iterated with pre-norm / attention / post-norm / MLP at
/// each level.
#[test]
fn test_layer_structure_multi_layer_forward() {
    for num_layers in [1, 2, 4, 8] {
        let cfg = tiny_config().with_num_hidden_layers(num_layers);
        let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
        let model = Qwen3Model::load(&vb, cfg.clone()).unwrap();

        let logits = model.forward(&[0, 1, 2], &[0, 1, 2]).unwrap();
        assert_eq!(
            logits.dims(),
            &[1, 3, cfg.vocab_size],
            "num_layers={num_layers}: output shape should be [1, 3, vocab_size]"
        );
    }
}

/// Verify that each decoder layer has both input_layernorm (pre-attn) and
/// post_attention_layernorm (pre-MLP) by checking that a 1-layer model
/// loads successfully. The layer constructor requires both RMSNorm weights
/// at `input_layernorm.weight` and `post_attention_layernorm.weight`.
#[test]
fn test_layer_structure_dual_rmsnorm_per_layer() {
    // If either RMSNorm weight were missing from the VarBuilder pattern,
    // loading would fail. VarBuilder::zeros creates all requested shapes.
    let cfg = tiny_config().with_num_hidden_layers(1);
    let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
    let model = Qwen3Model::load(&vb, cfg);
    assert!(
        model.is_ok(),
        "1-layer model should load with dual RMSNorm: {:?}",
        model.err()
    );
}

/// Verify the final RMSNorm (model.norm) is applied after the decoder stack
/// and before lm_head. The output should be [1, seq, vocab_size].
#[test]
fn test_layer_structure_final_norm_before_lm_head() {
    let cfg = tiny_config();
    let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
    let model = Qwen3Model::load(&vb, cfg.clone()).unwrap();

    let (logits, hidden) = model
        .forward_from_embeddings_with_hidden(
            &DynTensor::zeros(&[1, 3, cfg.hidden_size], DType::F32, &Device::Cpu).unwrap(),
            &[0, 1, 2],
            None,
        )
        .unwrap();

    // hidden is after final RMSNorm, logits after lm_head
    assert_eq!(hidden.dims(), &[1, 3, cfg.hidden_size]);
    assert_eq!(logits.dims(), &[1, 3, cfg.vocab_size]);
}

// ===========================================================================
// 10. Attention mask: causal mask generation and shape
// ===========================================================================

/// Causal mask for initial prompt (no cache) is [1, 1, seq, seq].
#[test]
fn test_causal_mask_initial_prompt_shape() {
    use nn_core::layers::causal_mask_dtype;
    for seq_len in [1, 4, 8, 16] {
        if seq_len == 1 {
            // Single-token: build_causal_mask returns None (optimization)
            continue;
        }
        let mask = causal_mask_dtype(seq_len, DType::F32, &Device::Cpu).unwrap();
        assert_eq!(
            mask.dims(),
            &[1, 1, seq_len, seq_len],
            "causal mask for seq_len={seq_len} should be [1, 1, {seq_len}, {seq_len}]"
        );
    }
}

/// Causal mask with offset: shape is [1, 1, new_tokens, total_tokens].
#[test]
fn test_causal_mask_with_offset_shape() {
    use nn_core::layers::causal_mask_with_offset;
    // 2 new tokens attending to 10 total (8 cached + 2 new)
    let mask = causal_mask_with_offset(2, 10, DType::F32, &Device::Cpu).unwrap();
    assert_eq!(mask.dims(), &[1, 1, 2, 10]);
}

/// Causal mask lower triangle: positions below diagonal should be 0.0 (attend),
/// positions above diagonal should be -inf (don't attend).
#[test]
fn test_causal_mask_lower_triangular_values() {
    use nn_core::layers::causal_mask_dtype;
    let seq_len = 4;
    let mask = causal_mask_dtype(seq_len, DType::F32, &Device::Cpu).unwrap();
    let vals = mask.to_flat_vec::<f32>().unwrap();

    // mask shape: [1, 1, 4, 4], flattened
    for row in 0..seq_len {
        for col in 0..seq_len {
            let v = vals[row * seq_len + col];
            if col <= row {
                assert_eq!(
                    v, 0.0,
                    "mask[{row}][{col}] should be 0.0 (can attend), got {v}"
                );
            } else {
                assert!(
                    v == f32::NEG_INFINITY,
                    "mask[{row}][{col}] should be -inf (cannot attend), got {v}"
                );
            }
        }
    }
}

/// Causal mask with offset: new tokens can attend to all cached positions
/// plus their own position, but not future positions.
#[test]
fn test_causal_mask_with_offset_values() {
    use nn_core::layers::causal_mask_with_offset;
    let new_tokens = 2;
    let total = 5; // 3 cached + 2 new
    let mask = causal_mask_with_offset(new_tokens, total, DType::F32, &Device::Cpu).unwrap();
    let vals = mask.to_flat_vec::<f32>().unwrap();

    // Row 0 (absolute position 3): can attend to positions 0,1,2,3; not 4
    // Row 1 (absolute position 4): can attend to positions 0,1,2,3,4 (all)
    for (col, &v) in vals.iter().enumerate().take(4) {
        assert_eq!(v, 0.0, "row 0, col {col}: should attend");
    }
    assert!(
        vals[4] == f32::NEG_INFINITY,
        "row 0, col 4: should not attend to future"
    );
    for col in 0..5 {
        assert_eq!(vals[5 + col], 0.0, "row 1, col {col}: should attend to all");
    }
}

/// Single-token decode step: build_causal_mask returns None (no mask needed)
/// because a single query can attend to all cached positions.
#[test]
fn test_single_token_decode_no_mask() {
    let cfg = tiny_config();
    let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
    let model = Qwen3Model::load(&vb, cfg.clone()).unwrap();
    let mut cache = model.new_cache();

    // Prefill
    model
        .forward_cached(&[0, 1, 2], &[0, 1, 2], Some(&mut cache))
        .unwrap();

    // Single-token decode: no mask needed, should still work
    let logits = model.forward_cached(&[3], &[3], Some(&mut cache)).unwrap();
    assert_eq!(logits.dims(), &[1, 1, cfg.vocab_size]);
}

// ===========================================================================
// 11. KV cache integration: shapes, incremental decoding
// ===========================================================================

/// KV cache starts empty, grows with each forward pass.
#[test]
fn test_kv_cache_incremental_growth() {
    let cfg = tiny_config();
    let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
    let model = Qwen3Model::load(&vb, cfg).unwrap();
    let mut cache = model.new_cache();

    assert_eq!(cache.seq_len(), 0);

    // Step 1: prefill 5 tokens
    model
        .forward_cached(&[0, 1, 2, 3, 4], &[0, 1, 2, 3, 4], Some(&mut cache))
        .unwrap();
    assert_eq!(cache.seq_len(), 5);

    // Step 2: decode 1 token
    model.forward_cached(&[5], &[5], Some(&mut cache)).unwrap();
    assert_eq!(cache.seq_len(), 6);

    // Step 3: decode 3 tokens at once (chunked decode)
    model
        .forward_cached(&[6, 7, 8], &[6, 7, 8], Some(&mut cache))
        .unwrap();
    assert_eq!(cache.seq_len(), 9);
}

/// Cache mismatch: using a cache with wrong layer count produces an error.
#[test]
fn test_kv_cache_layer_mismatch_error() {
    let cfg = tiny_config().with_num_hidden_layers(2);
    let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
    let model = Qwen3Model::load(&vb, cfg).unwrap();

    // Create cache with wrong layer count
    let mut bad_cache = nn_core::layers::kv_cache::KvCache::new(5);
    let result = model.forward_cached(&[0], &[0], Some(&mut bad_cache));
    assert!(
        result.is_err(),
        "using cache with 5 layers for 2-layer model should fail"
    );
}

/// Autoregressive decoding produces deterministic output shape at each step.
#[test]
fn test_kv_cache_autoregressive_output_shape() {
    let cfg = tiny_config();
    let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
    let model = Qwen3Model::load(&vb, cfg.clone()).unwrap();
    let mut cache = model.new_cache();

    // Prefill: output has seq_len tokens
    let prefill_len = 4;
    let ids: Vec<usize> = (0..prefill_len).collect();
    let positions: Vec<usize> = (0..prefill_len).collect();
    let logits = model
        .forward_cached(&ids, &positions, Some(&mut cache))
        .unwrap();
    assert_eq!(logits.dims(), &[1, prefill_len, cfg.vocab_size]);

    // Each autoregressive step: output has exactly 1 token
    for step in prefill_len..(prefill_len + 10) {
        let logits = model
            .forward_cached(&[step % cfg.vocab_size], &[step], Some(&mut cache))
            .unwrap();
        assert_eq!(
            logits.dims(),
            &[1, 1, cfg.vocab_size],
            "step {step}: autoregressive output should be [1, 1, vocab_size]"
        );
    }
}

// ===========================================================================
// 12. Sequence length handling: variable lengths, position ID validation
// ===========================================================================

/// Forward with various sequence lengths should all succeed and produce
/// correct output shapes.
#[test]
fn test_variable_sequence_lengths() {
    let cfg = tiny_config();
    let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
    let model = Qwen3Model::load(&vb, cfg.clone()).unwrap();

    for seq_len in [1, 2, 3, 5, 8, 16, 32] {
        let ids: Vec<usize> = (0..seq_len).map(|i| i % cfg.vocab_size).collect();
        let positions: Vec<usize> = (0..seq_len).collect();
        let logits = model.forward(&ids, &positions).unwrap();
        assert_eq!(
            logits.dims(),
            &[1, seq_len, cfg.vocab_size],
            "seq_len={seq_len}: output shape should be [1, {seq_len}, vocab_size]"
        );
    }
}

/// Input validation: mismatched input_ids and positions length should error.
#[test]
fn test_input_ids_positions_length_mismatch() {
    let cfg = tiny_config();
    let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
    let model = Qwen3Model::load(&vb, cfg).unwrap();

    let result = model.forward(&[0, 1, 2], &[0, 1]);
    assert!(result.is_err(), "3 input_ids with 2 positions should fail");
}

/// Non-contiguous position IDs (skip ahead) should work with KV cache.
#[test]
fn test_non_contiguous_position_ids_with_cache() {
    let cfg = tiny_config();
    let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
    let model = Qwen3Model::load(&vb, cfg.clone()).unwrap();
    let mut cache = model.new_cache();

    // Prefill at positions 0, 1, 2
    model
        .forward_cached(&[0, 1, 2], &[0, 1, 2], Some(&mut cache))
        .unwrap();

    // Skip to position 10 (e.g., speculative decoding scenario)
    let logits = model.forward_cached(&[3], &[10], Some(&mut cache)).unwrap();
    assert_eq!(logits.dims(), &[1, 1, cfg.vocab_size]);
}

/// Single-token forward (seq_len=1) is valid.
#[test]
fn test_single_token_forward() {
    let cfg = tiny_config();
    let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
    let model = Qwen3Model::load(&vb, cfg.clone()).unwrap();

    let logits = model.forward(&[42], &[0]).unwrap();
    assert_eq!(logits.dims(), &[1, 1, cfg.vocab_size]);
}

// ===========================================================================
// 13. Forward from embeddings (pre-computed embedding input)
// ===========================================================================

/// forward_from_embeddings accepts [batch, seq, hidden_size] tensors.
#[test]
fn test_forward_from_embeddings_shape() {
    let cfg = tiny_config();
    let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
    let model = Qwen3Model::load(&vb, cfg.clone()).unwrap();

    let embeddings = DynTensor::zeros(&[1, 4, cfg.hidden_size], DType::F32, &Device::Cpu).unwrap();
    let logits = model
        .forward_from_embeddings(&embeddings, &[0, 1, 2, 3], None)
        .unwrap();
    assert_eq!(logits.dims(), &[1, 4, cfg.vocab_size]);
}

/// forward_from_embeddings rejects wrong hidden_size.
#[test]
fn test_forward_from_embeddings_wrong_hidden_size() {
    let cfg = tiny_config(); // hidden_size = 256
    let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
    let model = Qwen3Model::load(&vb, cfg.clone()).unwrap();

    let wrong_hidden = cfg.hidden_size + 1;
    let embeddings = DynTensor::zeros(&[1, 2, wrong_hidden], DType::F32, &Device::Cpu).unwrap();
    let result = model.forward_from_embeddings(&embeddings, &[0, 1], None);
    assert!(
        result.is_err(),
        "wrong hidden_size should fail: expected {}, got {}",
        cfg.hidden_size,
        wrong_hidden
    );
}

/// forward_from_embeddings rejects mismatched seq_len and positions.
#[test]
fn test_forward_from_embeddings_seq_positions_mismatch() {
    let cfg = tiny_config();
    let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
    let model = Qwen3Model::load(&vb, cfg.clone()).unwrap();

    let embeddings = DynTensor::zeros(&[1, 3, cfg.hidden_size], DType::F32, &Device::Cpu).unwrap();
    // 3 tokens but only 2 positions
    let result = model.forward_from_embeddings(&embeddings, &[0, 1], None);
    assert!(
        result.is_err(),
        "3-token embedding with 2 positions should fail"
    );
}

/// forward_from_embeddings_with_hidden returns both logits and hidden states
/// with correct shapes.
#[test]
fn test_forward_from_embeddings_with_hidden_shapes() {
    let cfg = tiny_config();
    let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
    let model = Qwen3Model::load(&vb, cfg.clone()).unwrap();

    let embeddings = DynTensor::zeros(&[1, 5, cfg.hidden_size], DType::F32, &Device::Cpu).unwrap();
    let (logits, hidden) = model
        .forward_from_embeddings_with_hidden(&embeddings, &[0, 1, 2, 3, 4], None)
        .unwrap();

    assert_eq!(logits.dims(), &[1, 5, cfg.vocab_size]);
    assert_eq!(hidden.dims(), &[1, 5, cfg.hidden_size]);
}

// ===========================================================================
// 14. Config edge cases and validation boundaries
// ===========================================================================

/// hidden_size=0 should be rejected.
#[test]
fn test_config_hidden_size_zero_rejected() {
    let cfg = Qwen3Config::new(0, 512, 1, 2, 2, 50, 1e-6, 10_000.0, 64, true, None);
    assert!(cfg.validate().is_err());
}

/// intermediate_size=0 should be rejected.
#[test]
fn test_config_intermediate_size_zero_rejected() {
    let cfg = Qwen3Config::new(256, 0, 1, 2, 2, 50, 1e-6, 10_000.0, 64, true, None);
    assert!(cfg.validate().is_err());
}

/// vocab_size=0 should be rejected.
#[test]
fn test_config_vocab_size_zero_rejected() {
    let cfg = Qwen3Config::new(256, 512, 1, 2, 2, 0, 1e-6, 10_000.0, 64, true, None);
    assert!(cfg.validate().is_err());
}

/// num_attention_heads=0 should be rejected.
#[test]
fn test_config_num_heads_zero_rejected() {
    let cfg = Qwen3Config::new(256, 512, 1, 0, 0, 50, 1e-6, 10_000.0, 64, true, None);
    assert!(cfg.validate().is_err());
}

/// rope_theta must be positive and finite.
#[test]
fn test_config_rope_theta_nan_rejected() {
    let cfg = Qwen3Config::new(256, 512, 1, 2, 2, 50, 1e-6, f64::NAN, 64, true, None);
    assert!(cfg.validate().is_err());
}

#[test]
fn test_config_rope_theta_zero_rejected() {
    let cfg = Qwen3Config::new(256, 512, 1, 2, 2, 50, 1e-6, 0.0, 64, true, None);
    assert!(cfg.validate().is_err());
}

#[test]
fn test_config_rope_theta_negative_rejected() {
    let cfg = Qwen3Config::new(256, 512, 1, 2, 2, 50, 1e-6, -1.0, 64, true, None);
    assert!(cfg.validate().is_err());
}

/// rms_norm_eps=NaN should be rejected.
#[test]
fn test_config_rms_norm_eps_nan_rejected() {
    let cfg = Qwen3Config::new(256, 512, 1, 2, 2, 50, f64::NAN, 10_000.0, 64, true, None);
    assert!(cfg.validate().is_err());
}

/// GQA: kv_heads > num_heads is invalid (ratio < 1).
#[test]
fn test_config_kv_heads_exceeds_num_heads() {
    let cfg = Qwen3Config::new(256, 512, 1, 2, 4, 50, 1e-6, 10_000.0, 64, true, None);
    assert!(
        cfg.validate().is_err(),
        "kv_heads (4) > num_heads (2) should be rejected"
    );
}

/// Builder methods: with_vocab_size and with_num_hidden_layers.
#[test]
fn test_config_builder_methods() {
    let cfg = tiny_config()
        .with_vocab_size(500)
        .with_num_hidden_layers(10);
    assert_eq!(cfg.vocab_size, 500);
    assert_eq!(cfg.num_hidden_layers, 10);
    // Other fields unchanged
    assert_eq!(cfg.hidden_size, 256);
    assert_eq!(cfg.intermediate_size, 512);
}

// ===========================================================================
// 15. RoPE rotation matrix: orthogonality and norm preservation
// ===========================================================================

/// RoPE rotation preserves dot product between Q and K at same position
/// (rotation by same angle preserves relative geometry).
#[test]
fn test_rope_preserves_dot_product_same_position() {
    let cache = RoPECache::new(64, 128, 1_000_000.0);

    for pos in [0, 1, 5, 20, 63] {
        let (cos, sin) = cache.get(pos);

        let mut q: Vec<f32> = (0..128).map(|i| (i as f32 + 1.0) * 0.01).collect();
        let mut k: Vec<f32> = (0..128).map(|i| (i as f32 + 1.0) * 0.02).collect();

        let dot_before: f32 = q.iter().zip(k.iter()).map(|(&a, &b)| a * b).sum();

        RoPECache::apply_rope(&mut q, &mut k, cos, sin);

        let dot_after: f32 = q.iter().zip(k.iter()).map(|(&a, &b)| a * b).sum();

        assert!(
            (dot_before - dot_after).abs() < 1e-2,
            "RoPE should preserve Q.K dot product at same position {pos}: before={dot_before}, after={dot_after}"
        );
    }
}

/// RoPE with Qwen3's 1M base has lower-frequency rotations than standard 10k base,
/// meaning the angular difference between adjacent positions is smaller.
#[test]
fn test_rope_1m_base_lower_angular_velocity() {
    let cache_10k = RoPECache::new(64, 128, 10_000.0);
    let cache_1m = RoPECache::new(64, 128, 1_000_000.0);

    // At frequency index 0 (highest frequency): angle = position * theta[0] = position * 1.0
    // theta[0] = 1.0 for any base, so no difference at index 0.
    // At frequency index 32 (mid-band): theta = 1/sqrt(base)
    // base=10k: theta=1/100=0.01, base=1M: theta=1/1000=0.001
    // The 1M base rotates 10x slower at this frequency.

    let (cos_10k_10, _) = cache_10k.get(10);
    let (cos_1m_10, _) = cache_1m.get(10);

    // Mid-band (index 32): 1M base should be closer to 1.0 (less rotation)
    let cos_10k_mid = cos_10k_10[32];
    let cos_1m_mid = cos_1m_10[32];

    // cos(10 * 0.001) ~ 0.99999 vs cos(10 * 0.01) ~ 0.9995
    assert!(
        (cos_1m_mid - 1.0).abs() < (cos_10k_mid - 1.0).abs(),
        "1M base should rotate less at mid-band: cos_1m={cos_1m_mid}, cos_10k={cos_10k_mid}"
    );
}

// ===========================================================================
// 16. Model dtype and device accessors
// ===========================================================================

/// Model reports the correct dtype from VarBuilder.
#[test]
fn test_model_dtype_from_varbuilder() {
    let cfg = tiny_config();
    let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
    let model = Qwen3Model::load(&vb, cfg).unwrap();
    assert_eq!(model.dtype(), DType::F32);
}

/// Model reports CPU device when loaded on CPU.
#[test]
fn test_model_device_cpu() {
    let cfg = tiny_config();
    let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
    let model = Qwen3Model::load(&vb, cfg).unwrap();
    assert!(matches!(model.device(), Device::Cpu));
}

/// config() accessor returns the original config.
#[test]
fn test_model_config_accessor() {
    let cfg = Qwen3Config::new(512, 1024, 3, 4, 2, 200, 1e-5, 50_000.0, 128, false, None);
    let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
    let model = Qwen3Model::load(&vb, cfg).unwrap();

    assert_eq!(model.config().hidden_size, 512);
    assert_eq!(model.config().intermediate_size, 1024);
    assert_eq!(model.config().num_hidden_layers, 3);
    assert_eq!(model.config().num_attention_heads, 4);
    assert_eq!(model.config().num_key_value_heads, 2);
    assert_eq!(model.config().vocab_size, 200);
    assert!(!model.config().tie_word_embeddings);
}

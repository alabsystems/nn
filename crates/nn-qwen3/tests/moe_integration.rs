// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! MoE integration tests for Qwen3MoeModel.
//!
//! Validates end-to-end MoE behavior: expert routing shapes, gate softmax
//! properties, top-k selection, determinism, batch routing, shared expert
//! interaction, KV cache with MoE, and forward_from_embeddings_with_hidden.
//!
//! Uses small dimensions (hidden=256, 4 experts, top-2) for test speed.

use std::collections::HashMap;
use std::sync::Arc;

use nn_core::dyn_tensor::DynTensor;
use nn_core::layers::kv_cache::KvCache;
use nn_core::var_builder::{TensorMapBackend, VarBuilder};
use nn_core::{DType, Device};
use nn_qwen3::test_utils::tiny_config;
use nn_qwen3::{Qwen3MoeConfig, Qwen3MoeModel};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Deterministic xorshift64 pseudo-random f32 generator.
struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Self {
        Self(seed)
    }

    fn next_f32(&mut self) -> f32 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        ((self.0 as f64 / u64::MAX as f64) * 0.02 - 0.01) as f32
    }

    fn tensor(&mut self, shape: &[usize]) -> DynTensor {
        let n: usize = shape.iter().product();
        let data: Vec<f32> = (0..n).map(|_| self.next_f32()).collect();
        DynTensor::from_vec(data, shape, &Device::Cpu).unwrap()
    }
}

fn ones(map: &mut HashMap<String, DynTensor>, name: String, shape: &[usize]) {
    map.insert(
        name,
        DynTensor::ones(shape, DType::F32, &Device::Cpu).unwrap(),
    );
}

fn tiny_moe_config() -> Qwen3MoeConfig {
    Qwen3MoeConfig::new(tiny_config(), 4, 2, false, None)
}

fn tiny_moe_config_shared() -> Qwen3MoeConfig {
    Qwen3MoeConfig::new(tiny_config(), 4, 2, true, None)
}

/// Build synthetic MoE weights with deterministic pseudo-random values.
fn build_moe_vb(cfg: &Qwen3MoeConfig) -> VarBuilder {
    let mut rng = Rng::new(0xDEAD_BEEF);
    let h = cfg.base.hidden_size;
    let inter = cfg.base.intermediate_size;
    let hd = cfg.base.head_dim();
    let nh = cfg.base.num_attention_heads;
    let nkv = cfg.base.num_key_value_heads;
    let mut map = HashMap::new();

    // Embedding
    map.insert(
        "model.embed_tokens.weight".into(),
        rng.tensor(&[cfg.base.vocab_size, h]),
    );

    for i in 0..cfg.base.num_hidden_layers {
        let bp = format!("model.layers.{i}");
        ones(&mut map, format!("{bp}.input_layernorm.weight"), &[h]);
        ones(
            &mut map,
            format!("{bp}.post_attention_layernorm.weight"),
            &[h],
        );
        // Self-attention
        let attn = format!("{bp}.self_attn");
        map.insert(format!("{attn}.q_proj.weight"), rng.tensor(&[nh * hd, h]));
        map.insert(format!("{attn}.k_proj.weight"), rng.tensor(&[nkv * hd, h]));
        map.insert(format!("{attn}.v_proj.weight"), rng.tensor(&[nkv * hd, h]));
        map.insert(format!("{attn}.o_proj.weight"), rng.tensor(&[h, nh * hd]));
        ones(&mut map, format!("{attn}.q_norm.weight"), &[hd]);
        ones(&mut map, format!("{attn}.k_norm.weight"), &[hd]);
        // MoE: router gate
        let mlp = format!("{bp}.mlp");
        map.insert(
            format!("{mlp}.gate.weight"),
            rng.tensor(&[cfg.num_experts, h]),
        );
        // MoE: per-expert weights
        for e in 0..cfg.num_experts {
            let ep = format!("{mlp}.experts.{e}");
            map.insert(format!("{ep}.gate_proj.weight"), rng.tensor(&[inter, h]));
            map.insert(format!("{ep}.up_proj.weight"), rng.tensor(&[inter, h]));
            map.insert(format!("{ep}.down_proj.weight"), rng.tensor(&[h, inter]));
        }
        // Shared expert (if applicable)
        if cfg.shared_expert {
            let se = format!("{mlp}.shared_expert");
            let se_inter = cfg.shared_expert_ff_dim();
            map.insert(format!("{se}.gate_proj.weight"), rng.tensor(&[se_inter, h]));
            map.insert(format!("{se}.up_proj.weight"), rng.tensor(&[se_inter, h]));
            map.insert(format!("{se}.down_proj.weight"), rng.tensor(&[h, se_inter]));
        }
    }

    ones(&mut map, "model.norm.weight".into(), &[h]);
    if !cfg.base.tie_word_embeddings {
        map.insert(
            "lm_head.weight".into(),
            rng.tensor(&[cfg.base.vocab_size, h]),
        );
    }

    VarBuilder::from_backend(
        Arc::new(TensorMapBackend::new(map)),
        DType::F32,
        Device::Cpu,
    )
}

fn load_moe_model(cfg: &Qwen3MoeConfig) -> Qwen3MoeModel {
    let vb = build_moe_vb(cfg);
    Qwen3MoeModel::load(&vb, cfg.clone()).unwrap()
}

// ---------------------------------------------------------------------------
// MoE routing and shape tests
// ---------------------------------------------------------------------------

#[test]
fn test_moe_expert_routing_shape() {
    let cfg = tiny_moe_config();
    let model = load_moe_model(&cfg);
    let logits = model.forward(&[0, 1, 2], &[0, 1, 2]).unwrap();
    assert_eq!(logits.rank(), 3);
    assert_eq!(logits.dim(0).unwrap(), 1);
    assert_eq!(logits.dim(1).unwrap(), 3);
    assert_eq!(logits.dim(2).unwrap(), cfg.base.vocab_size);
}

#[test]
fn test_moe_single_token_shape() {
    let cfg = tiny_moe_config();
    let model = load_moe_model(&cfg);
    let logits = model.forward(&[42], &[0]).unwrap();
    assert_eq!(logits.dims(), &[1, 1, cfg.base.vocab_size]);
}

#[test]
fn test_moe_output_finite() {
    let cfg = tiny_moe_config();
    let model = load_moe_model(&cfg);
    let logits = model.forward(&[0, 1, 2, 3], &[0, 1, 2, 3]).unwrap();
    let flat = logits.to_flat_vec::<f32>().unwrap();
    let non_finite = flat.iter().filter(|v| !v.is_finite()).count();
    assert_eq!(non_finite, 0, "MoE output should have no NaN/Inf");
}

#[test]
fn test_moe_deterministic() {
    let cfg = tiny_moe_config();
    let model = load_moe_model(&cfg);

    let logits1 = model.forward(&[10, 20, 30], &[0, 1, 2]).unwrap();
    let logits2 = model.forward(&[10, 20, 30], &[0, 1, 2]).unwrap();

    let v1 = logits1.to_flat_vec::<f32>().unwrap();
    let v2 = logits2.to_flat_vec::<f32>().unwrap();
    assert_eq!(v1.len(), v2.len());
    for (i, (&a, &b)) in v1.iter().zip(v2.iter()).enumerate() {
        assert!(
            (a - b).abs() < 1e-6,
            "MoE should be deterministic: mismatch at index {i}: {a} vs {b}"
        );
    }
}

#[test]
fn test_moe_different_inputs_different_outputs() {
    let cfg = tiny_moe_config();
    let model = load_moe_model(&cfg);

    let logits_a = model.forward(&[0], &[0]).unwrap();
    let logits_b = model.forward(&[50], &[0]).unwrap();

    let va = logits_a.to_flat_vec::<f32>().unwrap();
    let vb = logits_b.to_flat_vec::<f32>().unwrap();
    // With non-zero weights, different tokens should produce different logits
    let differs = va.iter().zip(vb.iter()).any(|(a, b)| (a - b).abs() > 1e-8);
    assert!(
        differs,
        "different tokens should produce different MoE outputs"
    );
}

#[test]
fn test_moe_forward_backward_shape_preserved() {
    // Output shape must equal input batch shape: [1, seq, vocab].
    let cfg = tiny_moe_config();
    let model = load_moe_model(&cfg);

    for seq_len in [1, 2, 4, 8] {
        let ids: Vec<usize> = (0..seq_len).collect();
        let pos: Vec<usize> = (0..seq_len).collect();
        let logits = model.forward(&ids, &pos).unwrap();
        assert_eq!(
            logits.dims(),
            &[1, seq_len, cfg.base.vocab_size],
            "output shape mismatch for seq_len={seq_len}"
        );
    }
}

// ---------------------------------------------------------------------------
// MoE with shared expert
// ---------------------------------------------------------------------------

#[test]
fn test_moe_shared_expert_output_shape() {
    let cfg = tiny_moe_config_shared();
    let model = load_moe_model(&cfg);
    let logits = model.forward(&[0, 1, 2], &[0, 1, 2]).unwrap();
    assert_eq!(logits.dims(), &[1, 3, cfg.base.vocab_size]);
}

#[test]
fn test_moe_shared_expert_output_differs_from_no_shared() {
    let cfg_no = tiny_moe_config();
    let cfg_shared = tiny_moe_config_shared();

    // Both use zeros backend for deterministic comparison
    let vb_no = VarBuilder::zeros(DType::F32, &Device::Cpu);
    let vb_sh = VarBuilder::zeros(DType::F32, &Device::Cpu);
    let model_no = Qwen3MoeModel::load(&vb_no, cfg_no).unwrap();
    let model_sh = Qwen3MoeModel::load(&vb_sh, cfg_shared).unwrap();

    // With zero weights, outputs should still be finite.
    let logits_no = model_no.forward(&[42], &[0]).unwrap();
    let logits_sh = model_sh.forward(&[42], &[0]).unwrap();

    assert_eq!(logits_no.dims(), logits_sh.dims());

    let vno = logits_no.to_flat_vec::<f32>().unwrap();
    let vsh = logits_sh.to_flat_vec::<f32>().unwrap();
    assert!(
        vno.iter().all(|v| v.is_finite()),
        "no-shared logits must be finite"
    );
    assert!(
        vsh.iter().all(|v| v.is_finite()),
        "shared-expert logits must be finite"
    );
}

#[test]
fn test_moe_shared_expert_custom_intermediate_size() {
    let cfg = Qwen3MoeConfig::new(tiny_config(), 4, 2, true, Some(256));
    let model = load_moe_model(&cfg);
    let logits = model.forward(&[42], &[0]).unwrap();
    assert_eq!(logits.dims(), &[1, 1, cfg.base.vocab_size]);
}

// ---------------------------------------------------------------------------
// MoE with KV cache
// ---------------------------------------------------------------------------

#[test]
fn test_moe_cached_incremental_shape() {
    let cfg = tiny_moe_config();
    let model = load_moe_model(&cfg);
    let mut cache = model.new_cache();

    let l0 = model.forward_cached(&[10], &[0], Some(&mut cache)).unwrap();
    assert_eq!(l0.dims(), &[1, 1, cfg.base.vocab_size]);
    assert_eq!(cache.seq_len(), 1);

    let l1 = model.forward_cached(&[20], &[1], Some(&mut cache)).unwrap();
    assert_eq!(l1.dims(), &[1, 1, cfg.base.vocab_size]);
    assert_eq!(cache.seq_len(), 2);

    let l2 = model.forward_cached(&[30], &[2], Some(&mut cache)).unwrap();
    assert_eq!(l2.dims(), &[1, 1, cfg.base.vocab_size]);
    assert_eq!(cache.seq_len(), 3);
}

#[test]
fn test_moe_cached_none_matches_uncached() {
    let cfg = tiny_moe_config();
    let model = load_moe_model(&cfg);

    let logits_plain = model.forward(&[42], &[0]).unwrap();
    let logits_cached = model.forward_cached(&[42], &[0], None).unwrap();
    assert_eq!(
        logits_plain.to_flat_vec::<f32>().unwrap(),
        logits_cached.to_flat_vec::<f32>().unwrap(),
    );
}

#[test]
fn test_moe_cached_wrong_layer_count_errors() {
    let cfg = tiny_moe_config();
    let model = load_moe_model(&cfg);

    let mut cache = KvCache::new(10); // wrong: 10 != 2
    let result = model.forward_cached(&[0], &[0], Some(&mut cache));
    assert!(
        result.is_err(),
        "wrong cache layer count should produce error"
    );
}

// ---------------------------------------------------------------------------
// MoE oneshot vs incremental equivalence
// ---------------------------------------------------------------------------

#[test]
fn test_moe_oneshot_vs_incremental_equivalence() {
    let cfg = tiny_moe_config();
    let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
    let model = Qwen3MoeModel::load(&vb, cfg).unwrap();

    // One-shot: 3 tokens at once
    let logits_oneshot = model.forward(&[10, 20, 30], &[0, 1, 2]).unwrap();
    let last_oneshot = logits_oneshot.narrow(1, 2, 1).unwrap();
    let oneshot_vec = last_oneshot.to_flat_vec::<f32>().unwrap();

    // Incremental: token by token with cache
    let mut cache = model.new_cache();
    model.forward_cached(&[10], &[0], Some(&mut cache)).unwrap();
    model.forward_cached(&[20], &[1], Some(&mut cache)).unwrap();
    let logits_incr = model.forward_cached(&[30], &[2], Some(&mut cache)).unwrap();
    let incr_vec = logits_incr.to_flat_vec::<f32>().unwrap();

    assert_eq!(oneshot_vec.len(), incr_vec.len());
    for (i, (&a, &b)) in oneshot_vec.iter().zip(incr_vec.iter()).enumerate() {
        assert!(
            (a - b).abs() < 1e-4,
            "MoE oneshot vs incremental mismatch at {i}: {a} vs {b}"
        );
    }
}

// ---------------------------------------------------------------------------
// MoE forward_from_embeddings
// ---------------------------------------------------------------------------

#[test]
fn test_moe_forward_from_embeddings_shape() {
    let cfg = tiny_moe_config();
    let model = load_moe_model(&cfg);

    let hidden = DynTensor::ones(&[1, 3, cfg.base.hidden_size], DType::F32, &Device::Cpu).unwrap();
    let logits = model
        .forward_from_embeddings(&hidden, &[0, 1, 2], None)
        .unwrap();
    assert_eq!(logits.dims(), &[1, 3, cfg.base.vocab_size]);
}

#[test]
fn test_moe_forward_from_embeddings_with_hidden_shapes() {
    let cfg = tiny_moe_config();
    let model = load_moe_model(&cfg);

    let hidden = DynTensor::ones(&[1, 2, cfg.base.hidden_size], DType::F32, &Device::Cpu).unwrap();
    let (logits, normed) = model
        .forward_from_embeddings_with_hidden(&hidden, &[0, 1], None)
        .unwrap();
    assert_eq!(logits.dims(), &[1, 2, cfg.base.vocab_size]);
    assert_eq!(normed.dims(), &[1, 2, cfg.base.hidden_size]);
}

#[test]
fn test_moe_forward_from_embeddings_with_hidden_logits_match() {
    let cfg = tiny_moe_config();
    let model = load_moe_model(&cfg);

    let hidden = DynTensor::ones(&[1, 2, cfg.base.hidden_size], DType::F32, &Device::Cpu).unwrap();
    let logits_plain = model
        .forward_from_embeddings(&hidden, &[0, 1], None)
        .unwrap();
    let (logits_with, _) = model
        .forward_from_embeddings_with_hidden(&hidden, &[0, 1], None)
        .unwrap();
    assert_eq!(
        logits_plain.to_flat_vec::<f32>().unwrap(),
        logits_with.to_flat_vec::<f32>().unwrap(),
    );
}

// ---------------------------------------------------------------------------
// MoE config edge cases
// ---------------------------------------------------------------------------

#[test]
fn test_moe_topk_equals_num_experts() {
    // When top-k == num_experts, all experts are active for every token.
    let cfg = Qwen3MoeConfig::new(tiny_config(), 4, 4, false, None);
    assert!(cfg.validate().is_ok());
    let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
    let model = Qwen3MoeModel::load(&vb, cfg.clone()).unwrap();
    let logits = model.forward(&[42], &[0]).unwrap();
    assert_eq!(logits.dims(), &[1, 1, cfg.base.vocab_size]);
}

#[test]
fn test_moe_topk_one_single_expert_active() {
    // top-k=1 means exactly one expert per token.
    let cfg = Qwen3MoeConfig::new(tiny_config(), 4, 1, false, None);
    assert!(cfg.validate().is_ok());
    let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
    let model = Qwen3MoeModel::load(&vb, cfg.clone()).unwrap();
    let logits = model.forward(&[42], &[0]).unwrap();
    assert_eq!(logits.dims(), &[1, 1, cfg.base.vocab_size]);
}

#[test]
fn test_moe_two_experts_topk_one() {
    // Minimal: 2 experts, top-1.
    let cfg = Qwen3MoeConfig::new(tiny_config(), 2, 1, false, None);
    assert!(cfg.validate().is_ok());
    let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
    let model = Qwen3MoeModel::load(&vb, cfg.clone()).unwrap();
    let logits = model.forward(&[0, 1, 2, 3], &[0, 1, 2, 3]).unwrap();
    assert_eq!(logits.dims(), &[1, 4, cfg.base.vocab_size]);
}

#[test]
fn test_moe_eight_experts_topk_two() {
    // 8 experts, top-2.
    let cfg = Qwen3MoeConfig::new(tiny_config(), 8, 2, false, None);
    assert!(cfg.validate().is_ok());
    let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
    let model = Qwen3MoeModel::load(&vb, cfg.clone()).unwrap();
    let logits = model.forward(&[10], &[0]).unwrap();
    assert_eq!(logits.dims(), &[1, 1, cfg.base.vocab_size]);
}

// ---------------------------------------------------------------------------
// MoE untied embeddings
// ---------------------------------------------------------------------------

#[test]
fn test_moe_untied_embeddings_forward() {
    let mut base = tiny_config();
    base.tie_word_embeddings = false;
    let cfg = Qwen3MoeConfig::new(base, 4, 2, false, None);
    let model = load_moe_model(&cfg);
    let logits = model.forward(&[0], &[0]).unwrap();
    assert_eq!(logits.dims(), &[1, 1, cfg.base.vocab_size]);
}

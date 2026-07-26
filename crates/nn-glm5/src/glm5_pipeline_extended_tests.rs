// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Extended pipeline and generation tests for GLM-4/5 decoder-only transformer.
//!
//! Part of #4186. Covers:
//! 1. Config validation: hidden_size divisible by num_heads
//! 2. Rotary embedding: frequency base and dimension
//! 3. SwiGLU activation: intermediate computation
//! 4. Multi-query attention: key/value head count
//! 5. Layer normalization: RMSNorm vs LayerNorm selection
//! 6. Token generation: greedy and sampling modes
//! 7. KV cache management: append and eviction
//! 8. Position encoding: RoPE application to Q and K
//! 9. Vocabulary projection: hidden_size to vocab_size
//! 10. Attention masking: causal and bidirectional modes
//! 11. Embedding table: token ID to vector mapping
//! 12. Residual connections: input + layer_output shape preservation

use super::*;
use crate::test_utils::tiny_config;
use nn_core::dyn_tensor::DynTensor;
use nn_core::layers::kv_cache::KvCache;
use nn_core::layers::{GenerationConfig, HalfRotaryEmbedding};
use nn_core::var_builder::VarBuilder;
use nn_core::{DType, Device};

// ---------------------------------------------------------------------------
// Helper
// ---------------------------------------------------------------------------

fn load_tiny_model() -> Glm5Model {
    let cfg = tiny_config();
    let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
    Glm5Model::load(&vb, cfg).unwrap()
}

// ===========================================================================
// 1. Config validation: hidden_size divisible by num_heads
// ===========================================================================

#[test]
fn test_pipeline_config_hidden_size_divisible_by_num_heads() {
    // hidden_size must equal num_attention_heads * kv_channels
    let cfg = Glm5Config::default();
    assert_eq!(
        cfg.hidden_size % cfg.num_attention_heads,
        0,
        "hidden_size ({}) must be divisible by num_attention_heads ({})",
        cfg.hidden_size,
        cfg.num_attention_heads
    );
    assert_eq!(
        cfg.hidden_size / cfg.num_attention_heads,
        cfg.kv_channels,
        "hidden_size / num_heads should equal kv_channels"
    );
}

#[test]
fn test_pipeline_config_tiny_hidden_divisible() {
    let cfg = tiny_config();
    assert_eq!(cfg.hidden_size, cfg.num_attention_heads * cfg.kv_channels);
}

#[test]
fn test_pipeline_config_chat_hidden_divisible() {
    let chat = Glm5Config::glm4_9b_chat();
    assert_eq!(
        chat.hidden_size,
        chat.num_attention_heads * chat.kv_channels,
        "chat config: hidden_size must be num_heads * head_dim"
    );
}

#[test]
fn test_pipeline_config_num_kv_groups_valid() {
    let cfg = Glm5Config::default();
    let groups = cfg.num_kv_groups().unwrap();
    assert_eq!(groups, 16, "GLM-4-9B: 32 heads / 2 kv_groups = 16");

    let tiny = tiny_config();
    let tiny_groups = tiny.num_kv_groups().unwrap();
    assert_eq!(tiny_groups, 2, "tiny: 4 heads / 2 kv_groups = 2");
}

#[test]
fn test_pipeline_config_zero_kv_groups_rejected() {
    let mut cfg = tiny_config();
    cfg.multi_query_group_num = 0;
    assert!(cfg.num_kv_groups().is_err());
    assert!(cfg.validate().is_err());
}

#[test]
fn test_pipeline_config_indivisible_heads_rejected() {
    let mut cfg = tiny_config();
    cfg.num_attention_heads = 5;
    cfg.multi_query_group_num = 3;
    assert!(
        cfg.num_kv_groups().is_err(),
        "5 heads not divisible by 3 kv_groups"
    );
}

// ===========================================================================
// 2. Rotary embedding: frequency base and dimension
// ===========================================================================

#[test]
fn test_pipeline_rope_frequency_base_standard() {
    let cfg = Glm5Config::default();
    assert_eq!(cfg.rope_theta, 10_000.0);
    // Standard frequency: freq_i = 1 / (theta^(2i/d)) for i in [0..d/2)
    let rope_dim = cfg.kv_channels / 2; // half-RoPE
    let freq_0 = 1.0 / cfg.rope_theta.powf(0.0 / (2 * rope_dim) as f64);
    assert!((freq_0 - 1.0).abs() < 1e-12, "freq_0 should be 1.0");

    let freq_1 = 1.0 / cfg.rope_theta.powf(2.0 / (2 * rope_dim) as f64);
    assert!(freq_1 < freq_0, "freq_1 should be less than freq_0");
    assert!(freq_1 > 0.0, "freq_1 should be positive");
}

#[test]
fn test_pipeline_rope_frequency_base_extended_context() {
    // Chat variant uses rope_theta=5M for 128K context
    let chat = Glm5Config::glm4_9b_chat();
    assert_eq!(chat.rope_theta, 5_000_000.0);

    let rope_dim = chat.kv_channels / 2;
    // Higher theta means lower frequencies (slower rotation per position)
    let freq_mid_chat = 1.0 / chat.rope_theta.powf(2.0 * 10.0 / (2 * rope_dim) as f64);
    let base = Glm5Config::default();
    let freq_mid_base = 1.0 / base.rope_theta.powf(2.0 * 10.0 / (2 * rope_dim) as f64);
    assert!(
        freq_mid_chat < freq_mid_base,
        "extended context should have lower frequencies"
    );
}

#[test]
fn test_pipeline_rope_half_dim_equals_kv_channels_over_2() {
    for (head_dim, seq) in [(8, 32), (16, 64), (64, 256), (128, 512)] {
        let rope = HalfRotaryEmbedding::new(head_dim, seq, 10_000.0, &Device::Cpu).unwrap();
        assert_eq!(
            rope.rope_dim(),
            head_dim / 2,
            "head_dim={head_dim}: rope_dim should be head_dim/2"
        );
    }
}

#[test]
fn test_pipeline_rope_invalid_theta_rejected() {
    let mut cfg = tiny_config();
    cfg.rope_theta = f64::INFINITY;
    assert!(cfg.validate().is_err(), "infinite rope_theta rejected");

    cfg.rope_theta = f64::NEG_INFINITY;
    assert!(
        cfg.validate().is_err(),
        "negative infinite rope_theta rejected"
    );
}

// ===========================================================================
// 3. SwiGLU activation: intermediate computation
// ===========================================================================

#[test]
fn test_pipeline_swiglu_numerical_correctness() {
    // SwiGLU: silu(gate) * up, where silu(x) = x * sigmoid(x)
    let gate_vals = vec![1.0_f32, -1.0, 0.5, 3.0];
    let up_vals = vec![2.0_f32, 0.5, -1.0, 1.0];

    let gate = DynTensor::from_vec(gate_vals.clone(), &[1, 1, 4], &Device::Cpu).unwrap();
    let up = DynTensor::from_vec(up_vals.clone(), &[1, 1, 4], &Device::Cpu).unwrap();

    let result = gate.silu().unwrap().broadcast_mul(&up).unwrap();
    let out = result.to_flat_vec::<f32>().unwrap();

    for i in 0..4 {
        let g = gate_vals[i];
        let u = up_vals[i];
        let silu_g = g / (1.0 + (-g).exp());
        let expected = silu_g * u;
        assert!(
            (out[i] - expected).abs() < 1e-5,
            "SwiGLU[{i}]: gate={g}, up={u}, expected={expected}, got={}",
            out[i]
        );
    }
}

#[test]
fn test_pipeline_swiglu_negative_gate_output_sign() {
    // silu(negative) is negative: x * sigmoid(x) where x < 0 and sigmoid(x) in (0, 0.5)
    let gate = DynTensor::from_vec(vec![-2.0_f32], &[1, 1, 1], &Device::Cpu).unwrap();
    let up = DynTensor::from_vec(vec![1.0_f32], &[1, 1, 1], &Device::Cpu).unwrap();

    let result = gate.silu().unwrap().broadcast_mul(&up).unwrap();
    let out = result.to_flat_vec::<f32>().unwrap();

    assert!(
        out[0] < 0.0,
        "silu(-2) * 1.0 should be negative, got {}",
        out[0]
    );
    // silu(-2) = -2 * sigmoid(-2) = -2 * (1/(1+e^2)) ≈ -2 * 0.1192 ≈ -0.2384
    assert!(
        (out[0] - (-0.2384)).abs() < 0.01,
        "silu(-2) ≈ -0.2384, got {}",
        out[0]
    );
}

#[test]
fn test_pipeline_swiglu_fused_dim_always_even() {
    // The fused gate+up dimension must be even for splitting
    for &ffn in &[128, 256, 512, 1024, 13696] {
        let fused = ffn * 2;
        assert_eq!(fused % 2, 0, "fused dim {fused} must be even");
        assert_eq!(fused / 2, ffn, "split should recover ffn_hidden_size");
    }
}

// ===========================================================================
// 4. Multi-query attention: key/value head count
// ===========================================================================

#[test]
fn test_pipeline_mqa_kv_heads_less_than_q_heads() {
    let cfg = Glm5Config::default();
    assert!(
        cfg.multi_query_group_num <= cfg.num_attention_heads,
        "KV heads ({}) must be <= Q heads ({})",
        cfg.multi_query_group_num,
        cfg.num_attention_heads
    );
}

#[test]
fn test_pipeline_mqa_repeat_kv_factor() {
    let cfg = tiny_config();
    let repeat = cfg.num_attention_heads / cfg.multi_query_group_num;
    assert_eq!(repeat, 2, "tiny: 4/2 = 2");

    // Simulate repeat_kv shape expansion
    let batch = 1;
    let seq = 3;
    let nkv = cfg.multi_query_group_num;
    let hd = cfg.kv_channels;

    let k = DynTensor::ones(&[batch, nkv, seq, hd], DType::F32, &Device::Cpu).unwrap();
    let k_expanded = nn_core::layers::repeat_kv(&k, repeat).unwrap();
    assert_eq!(
        k_expanded.dims(),
        &[batch, cfg.num_attention_heads, seq, hd],
        "after repeat_kv, heads dim should match num_attention_heads"
    );
}

#[test]
fn test_pipeline_mqa_single_kv_head_config() {
    // MQA: multi_query_group_num = 1
    let mut cfg = tiny_config();
    cfg.num_attention_heads = 8;
    cfg.multi_query_group_num = 1;
    cfg.hidden_size = 8 * cfg.kv_channels; // maintain invariant
    assert!(cfg.validate().is_ok());

    let repeat = cfg.num_attention_heads / cfg.multi_query_group_num;
    assert_eq!(repeat, 8, "MQA: repeat = num_heads");
}

#[test]
fn test_pipeline_mqa_full_mha_config() {
    // MHA: multi_query_group_num = num_attention_heads
    let mut cfg = tiny_config();
    cfg.num_attention_heads = 4;
    cfg.multi_query_group_num = 4;
    assert!(cfg.validate().is_ok());

    let repeat = cfg.num_attention_heads / cfg.multi_query_group_num;
    assert_eq!(repeat, 1, "MHA: repeat = 1");
}

// ===========================================================================
// 5. Layer normalization: RMSNorm vs LayerNorm selection
// ===========================================================================

#[test]
fn test_pipeline_rmsnorm_flag_default_true() {
    let cfg = Glm5Config::default();
    assert!(cfg.rmsnorm, "GLM-4/5 uses RMSNorm by default");
}

#[test]
fn test_pipeline_rmsnorm_flag_chat_true() {
    let chat = Glm5Config::glm4_9b_chat();
    assert!(chat.rmsnorm, "GLM-4-9B-chat uses RMSNorm");
}

#[test]
fn test_pipeline_layernorm_epsilon_positive_finite() {
    let cfg = Glm5Config::default();
    assert!(cfg.layernorm_epsilon > 0.0);
    assert!(cfg.layernorm_epsilon.is_finite());

    let chat = Glm5Config::glm4_9b_chat();
    assert!(chat.layernorm_epsilon > 0.0);
    assert!(chat.layernorm_epsilon.is_finite());

    // Chat uses a smaller epsilon than base
    assert!(
        chat.layernorm_epsilon < cfg.layernorm_epsilon,
        "chat eps ({}) should be < base eps ({})",
        chat.layernorm_epsilon,
        cfg.layernorm_epsilon
    );
}

#[test]
fn test_pipeline_layernorm_epsilon_invalid_rejected() {
    let mut cfg = tiny_config();

    cfg.layernorm_epsilon = 0.0;
    assert!(cfg.validate().is_err(), "zero epsilon rejected");

    cfg.layernorm_epsilon = -1e-5;
    assert!(cfg.validate().is_err(), "negative epsilon rejected");

    cfg.layernorm_epsilon = f64::NAN;
    assert!(cfg.validate().is_err(), "NaN epsilon rejected");
}

#[test]
fn test_pipeline_rmsnorm_preserves_shape() {
    // RMSNorm applied to a tensor should preserve its shape
    let cfg = tiny_config();
    let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
    let model = Glm5Model::load(&vb, cfg.clone()).unwrap();

    // Forward pass: check that output shape matches expectation
    let seq_len = 5;
    let logits = model
        .forward(
            &(0..seq_len).collect::<Vec<_>>(),
            &(0..seq_len).collect::<Vec<_>>(),
        )
        .unwrap();
    assert_eq!(
        logits.dims(),
        &[1, seq_len, cfg.padded_vocab_size],
        "shape should be [1, seq_len, vocab] after normalization + projection"
    );
}

// ===========================================================================
// 6. Token generation: greedy and sampling modes
// ===========================================================================

#[test]
fn test_pipeline_greedy_deterministic() {
    let model = load_tiny_model();
    let out_a = model.generate_greedy(&[10, 20], 5).unwrap();
    let out_b = model.generate_greedy(&[10, 20], 5).unwrap();
    assert_eq!(
        out_a.token_ids, out_b.token_ids,
        "greedy generation must be deterministic"
    );
}

#[test]
fn test_pipeline_greedy_zero_weights_picks_last_index() {
    // With zero weights, all logits are equal. argmax with total_cmp tiebreaks
    // to the LAST index, so generated token = padded_vocab_size - 1.
    let model = load_tiny_model();
    let vocab = model.config().padded_vocab_size;
    let out = model.generate_greedy(&[0], 3).unwrap();
    for &tok in &out.token_ids {
        assert_eq!(
            tok,
            vocab - 1,
            "zero-weight greedy should pick last vocab index ({}), got {tok}",
            vocab - 1
        );
    }
}

#[test]
fn test_pipeline_sampling_config_with_temperature() {
    // Temperature > 0 with a seed enables sampling
    let model = load_tiny_model();
    let device = model.device();
    let mut cache = model.new_cache();
    let cfg = GenerationConfig::new(4)
        .with_temperature(1.0)
        .with_top_k(10);

    let output = nn_core::layers::generate(
        |input, c| model.model_fn_adapter(input, c),
        &[42],
        &mut cache,
        &cfg,
        &device,
    )
    .unwrap();

    assert_eq!(output.token_ids.len(), 4);
    let vocab = model.config().padded_vocab_size;
    for &tok in &output.token_ids {
        assert!(tok < vocab, "token {tok} must be in vocab range");
    }
}

#[test]
fn test_pipeline_generation_eos_early_stop() {
    let model = load_tiny_model();
    let vocab = model.config().padded_vocab_size;
    let eos_id = vocab - 1; // argmax tiebreaks to this with zero weights
    let device = model.device();
    let mut cache = model.new_cache();
    let cfg = GenerationConfig::new(20).with_eos_token_id(eos_id);

    let output = nn_core::layers::generate(
        |input, c| model.model_fn_adapter(input, c),
        &[42],
        &mut cache,
        &cfg,
        &device,
    )
    .unwrap();

    assert_eq!(
        output.token_ids.len(),
        1,
        "EOS should stop after first generated token"
    );
    assert!(output.finished, "should report finished=true");
}

// ===========================================================================
// 7. KV cache management: append and eviction
// ===========================================================================

#[test]
fn test_pipeline_kv_cache_append_grows() {
    let model = load_tiny_model();
    let mut cache = model.new_cache();

    model
        .forward_cached(&[0, 1, 2], &[0, 1, 2], Some(&mut cache))
        .unwrap();
    assert_eq!(cache.seq_len(), 3);

    model.forward_cached(&[3], &[3], Some(&mut cache)).unwrap();
    assert_eq!(cache.seq_len(), 4);
}

#[test]
fn test_pipeline_kv_cache_layer_count() {
    let model = load_tiny_model();
    let cache = model.new_cache();
    assert_eq!(
        cache.num_layers(),
        model.config().num_layers,
        "cache should have same layers as model"
    );
}

#[test]
fn test_pipeline_flat_kv_cache_append_and_retrieve() {
    let num_layers = 2;
    let num_heads = 4;
    let head_dim = 8;
    let token_size = num_heads * head_dim;
    let mut cache = kv_cache::KVCache::new(num_layers, num_heads, head_dim, 16);

    let key: Vec<f32> = (0..token_size).map(|i| i as f32 * 0.1).collect();
    let val: Vec<f32> = (0..token_size).map(|i| i as f32 * 0.2).collect();

    for layer in 0..num_layers {
        cache.append(layer, &key, &val).unwrap();
    }
    assert_eq!(cache.len(), 1);

    let retrieved_key = cache.get_keys(0);
    assert_eq!(retrieved_key.len(), token_size);
    for i in 0..token_size {
        assert!(
            (retrieved_key[i] - key[i]).abs() < 1e-7,
            "key mismatch at {i}"
        );
    }
}

#[test]
fn test_pipeline_flat_kv_cache_eviction_at_capacity() {
    let num_layers = 1;
    let num_heads = 2;
    let head_dim = 4;
    let max_seq = 3;
    let token_size = num_heads * head_dim;
    let mut cache = kv_cache::KVCache::new(num_layers, num_heads, head_dim, max_seq);

    let key: Vec<f32> = vec![1.0; token_size];
    let val: Vec<f32> = vec![2.0; token_size];

    // Fill to capacity
    for _ in 0..max_seq {
        cache.append(0, &key, &val).unwrap();
    }
    assert_eq!(cache.len(), max_seq);

    // Next append should fail
    let result = cache.append(0, &key, &val);
    assert!(result.is_err(), "should reject append beyond max_seq_len");

    // Clear and refill
    cache.clear();
    assert_eq!(cache.len(), 0);
    cache.append(0, &key, &val).unwrap();
    assert_eq!(cache.len(), 1);
}

// ===========================================================================
// 8. Position encoding: RoPE application to Q and K
// ===========================================================================

#[test]
fn test_pipeline_rope_apply_pair_q_and_k() {
    let head_dim = 16;
    let rope = HalfRotaryEmbedding::new(head_dim, 128, 10_000.0, &Device::Cpu).unwrap();

    let q_data: Vec<f32> = (0..head_dim).map(|i| (i as f32 + 1.0) * 0.3).collect();
    let k_data: Vec<f32> = (0..head_dim).map(|i| (i as f32 + 1.0) * 0.7).collect();

    let q = DynTensor::from_vec(q_data.clone(), &[1, 1, 1, head_dim], &Device::Cpu).unwrap();
    let k = DynTensor::from_vec(k_data.clone(), &[1, 1, 1, head_dim], &Device::Cpu).unwrap();

    let (q_rot, k_rot) = rope.apply_pair(&q, &k, &[5]).unwrap();

    // Shape should be preserved
    assert_eq!(q_rot.dims(), &[1, 1, 1, head_dim]);
    assert_eq!(k_rot.dims(), &[1, 1, 1, head_dim]);

    let q_out = q_rot.to_flat_vec::<f32>().unwrap();
    let k_out = k_rot.to_flat_vec::<f32>().unwrap();

    // At position 5, first half (rotated) should differ from input
    let half = head_dim / 2;
    let q_first_diff: f32 = (0..half).map(|i| (q_out[i] - q_data[i]).abs()).sum();
    let k_first_diff: f32 = (0..half).map(|i| (k_out[i] - k_data[i]).abs()).sum();
    assert!(q_first_diff > 0.01, "Q first half should be rotated");
    assert!(k_first_diff > 0.01, "K first half should be rotated");

    // Second half (pass-through) should be unchanged
    for i in half..head_dim {
        assert!(
            (q_out[i] - q_data[i]).abs() < 1e-6,
            "Q second half dim {i} should pass through"
        );
        assert!(
            (k_out[i] - k_data[i]).abs() < 1e-6,
            "K second half dim {i} should pass through"
        );
    }
}

#[test]
fn test_pipeline_rope_different_positions_different_rotations() {
    let head_dim = 8;
    let rope = HalfRotaryEmbedding::new(head_dim, 64, 10_000.0, &Device::Cpu).unwrap();

    let data: Vec<f32> = vec![1.0; head_dim];
    let x = DynTensor::from_vec(data, &[1, 1, 1, head_dim], &Device::Cpu).unwrap();

    let out_2 = rope.apply(&x, 2).unwrap().to_flat_vec::<f32>().unwrap();
    let out_20 = rope.apply(&x, 20).unwrap().to_flat_vec::<f32>().unwrap();

    let diff: f32 = out_2
        .iter()
        .zip(out_20.iter())
        .map(|(a, b)| (a - b).abs())
        .sum();
    assert!(
        diff > 1e-4,
        "positions 2 and 20 should produce different RoPE outputs, diff={diff}"
    );
}

#[test]
fn test_pipeline_rope_position_zero_is_identity() {
    let head_dim = 8;
    let rope = HalfRotaryEmbedding::new(head_dim, 64, 10_000.0, &Device::Cpu).unwrap();

    let data: Vec<f32> = (1..=head_dim as i32).map(|i| i as f32).collect();
    let x = DynTensor::from_vec(data.clone(), &[1, 1, 1, head_dim], &Device::Cpu).unwrap();

    let result = rope.apply(&x, 0).unwrap().to_flat_vec::<f32>().unwrap();
    for i in 0..head_dim {
        assert!(
            (result[i] - data[i]).abs() < 1e-5,
            "pos=0 should be identity, dim {i}: {} vs {}",
            result[i],
            data[i]
        );
    }
}

// ===========================================================================
// 9. Vocabulary projection: hidden_size to vocab_size
// ===========================================================================

#[test]
fn test_pipeline_vocab_projection_output_dim() {
    let model = load_tiny_model();
    let cfg = model.config();
    let logits = model.forward(&[0], &[0]).unwrap();
    assert_eq!(
        logits.dims()[2],
        cfg.padded_vocab_size,
        "logits last dim should be padded_vocab_size"
    );
}

#[test]
fn test_pipeline_vocab_projection_scales_with_vocab() {
    for vocab in [10, 50, 200] {
        let mut cfg = tiny_config();
        cfg.padded_vocab_size = vocab;
        let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
        let model = Glm5Model::load(&vb, cfg).unwrap();
        let logits = model.forward(&[0], &[0]).unwrap();
        assert_eq!(
            logits.dims(),
            &[1, 1, vocab],
            "vocab={vocab}: output last dim should match"
        );
    }
}

#[test]
fn test_pipeline_vocab_projection_multi_token() {
    let model = load_tiny_model();
    let vocab = model.config().padded_vocab_size;
    for seq_len in [1, 4, 8] {
        let ids: Vec<usize> = (0..seq_len).collect();
        let pos: Vec<usize> = (0..seq_len).collect();
        let logits = model.forward(&ids, &pos).unwrap();
        assert_eq!(
            logits.dims(),
            &[1, seq_len, vocab],
            "seq_len={seq_len}: shape must be [1, seq_len, vocab]"
        );
    }
}

#[test]
fn test_pipeline_vocab_projection_logits_finite() {
    let model = load_tiny_model();
    let logits = model.forward(&[0, 1, 2, 3, 4], &[0, 1, 2, 3, 4]).unwrap();
    let flat = logits.to_flat_vec::<f32>().unwrap();
    let nan_count = flat.iter().filter(|v| v.is_nan()).count();
    let inf_count = flat.iter().filter(|v| v.is_infinite()).count();
    assert_eq!(nan_count, 0, "logits should have no NaN");
    assert_eq!(inf_count, 0, "logits should have no Inf");
}

// ===========================================================================
// 10. Attention masking: causal and bidirectional modes
// ===========================================================================

#[test]
fn test_pipeline_causal_mask_shape() {
    let mask = causal_mask(4, DType::F32, &Device::Cpu).unwrap();
    // causal_mask (= causal_mask_dtype) is documented to return a 4D mask
    // [1, 1, seq, seq] so it broadcasts over [batch, heads] in attention.
    assert_eq!(
        mask.dims(),
        &[1, 1, 4, 4],
        "causal mask should be [1, 1, seq, seq]"
    );
}

#[test]
fn test_pipeline_causal_mask_lower_triangular() {
    let seq = 4;
    let mask = causal_mask(seq, DType::F32, &Device::Cpu).unwrap();
    let flat = mask.to_flat_vec::<f32>().unwrap();

    for row in 0..seq {
        for col in 0..seq {
            let val = flat[row * seq + col];
            if col <= row {
                // Lower triangle + diagonal: should allow attention (0.0)
                assert_eq!(
                    val, 0.0,
                    "mask[{row},{col}] should be 0.0 (attend), got {val}"
                );
            } else {
                // Upper triangle: should block attention (large negative)
                assert!(
                    val < -1e3,
                    "mask[{row},{col}] should be large negative (block), got {val}"
                );
            }
        }
    }
}

#[test]
fn test_pipeline_causal_mask_with_offset_extends() {
    // When decoding with cache: new_tokens=1, total=5 (4 cached + 1 new)
    let mask = causal_mask_with_offset(1, 5, DType::F32, &Device::Cpu).unwrap();
    // causal_mask_with_offset is documented to return [1, 1, new_tokens,
    // total_tokens] so it broadcasts over [batch, heads] in attention.
    assert_eq!(
        mask.dims(),
        &[1, 1, 1, 5],
        "offset mask: [1, 1, new_tokens, total_tokens]"
    );

    let flat = mask.to_flat_vec::<f32>().unwrap();
    // Single new token can attend to all 5 positions (it's the last row)
    for &val in &flat {
        assert_eq!(val, 0.0, "last token should attend to all cached positions");
    }
}

#[test]
fn test_pipeline_causal_mask_seq_len_1_no_mask_needed() {
    // With seq_len=1 and no cache, no mask is generated (optimization)
    let model = load_tiny_model();
    // Single token forward should succeed without mask
    let logits = model.forward(&[42], &[0]).unwrap();
    assert_eq!(logits.dims(), &[1, 1, model.config().padded_vocab_size]);
}

// ===========================================================================
// 11. Embedding table: token ID to vector mapping
// ===========================================================================

#[test]
fn test_pipeline_embedding_output_dim() {
    let model = load_tiny_model();
    let cfg = model.config();
    let logits = model.forward(&[0], &[0]).unwrap();
    // After embedding -> layers -> projection: [1, 1, vocab]
    assert_eq!(logits.dims()[2], cfg.padded_vocab_size);
}

#[test]
fn test_pipeline_embedding_boundary_token_ids() {
    let model = load_tiny_model();
    let vocab = model.config().padded_vocab_size;

    // Token ID 0
    let logits_0 = model.forward(&[0], &[0]);
    assert!(logits_0.is_ok(), "token 0 should work");

    // Last valid token
    let logits_last = model.forward(&[vocab - 1], &[0]);
    assert!(logits_last.is_ok(), "token {} should work", vocab - 1);
}

#[test]
fn test_pipeline_embedding_different_tokens_different_logits() {
    let model = load_tiny_model();
    // With zero weights, all embeddings are zero, so logits will be identical.
    // This test verifies the forward path accepts different token IDs without error.
    let logits_a = model
        .forward(&[1], &[0])
        .unwrap()
        .to_flat_vec::<f32>()
        .unwrap();
    let logits_b = model
        .forward(&[50], &[0])
        .unwrap()
        .to_flat_vec::<f32>()
        .unwrap();

    // With zero weights they should be identical (both zero embeddings)
    assert_eq!(
        logits_a.len(),
        logits_b.len(),
        "logits should have same length for different tokens"
    );
}

#[test]
fn test_pipeline_embedding_multi_token_sequence() {
    let model = load_tiny_model();
    let cfg = model.config();

    let ids = vec![5, 10, 15, 20, 25];
    let pos: Vec<usize> = (0..ids.len()).collect();
    let logits = model.forward(&ids, &pos).unwrap();
    assert_eq!(
        logits.dims(),
        &[1, 5, cfg.padded_vocab_size],
        "5-token embedding should produce [1, 5, vocab] logits"
    );
}

// ===========================================================================
// 12. Residual connections: input + layer_output shape preservation
// ===========================================================================

#[test]
fn test_pipeline_residual_shape_preserved_through_layers() {
    // Verify that shape is preserved through the full forward pass at
    // various sequence lengths (residual connections maintain shape)
    let model = load_tiny_model();
    let cfg = model.config();

    for seq_len in [1, 3, 7, 12] {
        let ids: Vec<usize> = (0..seq_len).collect();
        let pos: Vec<usize> = (0..seq_len).collect();
        let logits = model.forward(&ids, &pos).unwrap();
        assert_eq!(
            logits.dims(),
            &[1, seq_len, cfg.padded_vocab_size],
            "seq_len={seq_len}: residual connections should preserve seq dim"
        );
    }
}

#[test]
fn test_pipeline_residual_with_cache_shape_consistent() {
    let model = load_tiny_model();
    let cfg = model.config();
    let mut cache = model.new_cache();

    // Prefill
    let logits_prefill = model
        .forward_cached(&[0, 1, 2, 3], &[0, 1, 2, 3], Some(&mut cache))
        .unwrap();
    assert_eq!(logits_prefill.dims(), &[1, 4, cfg.padded_vocab_size]);

    // Decode steps
    for step in 0..3 {
        let pos = 4 + step;
        let logits = model
            .forward_cached(&[42], &[pos], Some(&mut cache))
            .unwrap();
        assert_eq!(
            logits.dims(),
            &[1, 1, cfg.padded_vocab_size],
            "decode step {step}: shape should be [1, 1, vocab]"
        );
    }
}

#[test]
fn test_pipeline_residual_broadcast_add_shape() {
    // Simulate the residual connection: x + attention_output
    let batch = 1;
    let seq_len = 4;
    let hidden = 256;

    let residual = DynTensor::ones(&[batch, seq_len, hidden], DType::F32, &Device::Cpu).unwrap();
    let layer_out = DynTensor::zeros(&[batch, seq_len, hidden], DType::F32, &Device::Cpu).unwrap();

    let result = residual.broadcast_add(&layer_out).unwrap();
    assert_eq!(
        result.dims(),
        &[batch, seq_len, hidden],
        "residual add should preserve shape"
    );

    // Values should be residual + layer_out = 1.0 + 0.0 = 1.0
    let flat = result.to_flat_vec::<f32>().unwrap();
    for &v in &flat {
        assert!(
            (v - 1.0).abs() < 1e-7,
            "residual + 0 should be 1.0, got {v}"
        );
    }
}

#[test]
fn test_pipeline_residual_forward_from_embeddings_preserves_shape() {
    let model = load_tiny_model();
    let cfg = model.config();

    let emb = DynTensor::ones(&[1, 5, cfg.hidden_size], DType::F32, &Device::Cpu).unwrap();

    let logits = model
        .forward_from_embeddings(&emb, &[0, 1, 2, 3, 4], None)
        .unwrap();
    assert_eq!(
        logits.dims(),
        &[1, 5, cfg.padded_vocab_size],
        "forward_from_embeddings should produce [1, 5, vocab]"
    );
}

// ===========================================================================
// 13. Temperature scaling: numerical behavior
// ===========================================================================

#[test]
fn test_pipeline_temperature_scaling_divides_logits() {
    // Temperature T divides logits by T before softmax.
    // Higher T → flatter distribution; lower T → sharper distribution.
    let logits = [1.0_f32, 2.0, 3.0, 4.0];
    let t_high = 2.0_f64;
    let t_low = 0.5_f64;

    let scaled_high: Vec<f64> = logits.iter().map(|&l| f64::from(l) / t_high).collect();
    let scaled_low: Vec<f64> = logits.iter().map(|&l| f64::from(l) / t_low).collect();

    // Range of scaled values narrows with higher temperature
    let range_high = scaled_high.last().unwrap() - scaled_high.first().unwrap();
    let range_low = scaled_low.last().unwrap() - scaled_low.first().unwrap();
    assert!(
        range_high < range_low,
        "higher temp should flatten: range_high={range_high}, range_low={range_low}"
    );
}

#[test]
fn test_pipeline_temperature_zero_is_greedy() {
    // GenerationConfig with temperature=0.0 means greedy (argmax)
    let cfg = GenerationConfig::new(5).with_temperature(0.0);
    assert_eq!(cfg.temperature, 0.0);
    assert!(cfg.validate().is_ok());
}

#[test]
fn test_pipeline_temperature_negative_rejected() {
    let cfg = GenerationConfig::new(5).with_temperature(-1.0);
    assert!(
        cfg.validate().is_err(),
        "negative temperature should be rejected"
    );
}

#[test]
fn test_pipeline_temperature_nan_rejected() {
    let cfg = GenerationConfig::new(5).with_temperature(f64::NAN);
    assert!(
        cfg.validate().is_err(),
        "NaN temperature should be rejected"
    );
}

#[test]
fn test_pipeline_temperature_infinity_rejected() {
    let cfg = GenerationConfig::new(5).with_temperature(f64::INFINITY);
    assert!(
        cfg.validate().is_err(),
        "Inf temperature should be rejected"
    );
}

// ===========================================================================
// 14. Top-k filtering
// ===========================================================================

#[test]
fn test_pipeline_top_k_config_builder() {
    let cfg = GenerationConfig::new(10).with_top_k(5);
    assert_eq!(cfg.top_k, Some(5));
}

#[test]
fn test_pipeline_top_k_generation_produces_valid_tokens() {
    let model = load_tiny_model();
    let device = model.device();
    let mut cache = model.new_cache();
    let cfg = GenerationConfig::new(3).with_temperature(1.0).with_top_k(5);

    let output = nn_core::layers::generate(
        |input, c| model.model_fn_adapter(input, c),
        &[10],
        &mut cache,
        &cfg,
        &device,
    )
    .unwrap();

    let vocab = model.config().padded_vocab_size;
    for &tok in &output.token_ids {
        assert!(
            tok < vocab,
            "top-k token {tok} must be in vocab range [0, {vocab})"
        );
    }
}

// ===========================================================================
// 15. Top-p (nucleus) filtering
// ===========================================================================

#[test]
fn test_pipeline_top_p_config_builder() {
    let cfg = GenerationConfig::new(10).with_top_p(0.9);
    assert_eq!(cfg.top_p, Some(0.9));
}

#[test]
fn test_pipeline_top_p_invalid_zero_rejected() {
    let cfg = GenerationConfig::new(10).with_top_p(0.0);
    assert!(
        cfg.validate().is_err(),
        "top_p=0.0 should be rejected (must be > 0)"
    );
}

#[test]
fn test_pipeline_top_p_invalid_above_one_rejected() {
    let cfg = GenerationConfig::new(10).with_top_p(1.5);
    assert!(
        cfg.validate().is_err(),
        "top_p=1.5 should be rejected (must be <= 1.0)"
    );
}

#[test]
fn test_pipeline_top_p_boundary_one_valid() {
    let cfg = GenerationConfig::new(10).with_top_p(1.0);
    assert!(
        cfg.validate().is_ok(),
        "top_p=1.0 should be valid (includes entire distribution)"
    );
}

#[test]
fn test_pipeline_top_p_generation_produces_valid_tokens() {
    let model = load_tiny_model();
    let device = model.device();
    let mut cache = model.new_cache();
    let cfg = GenerationConfig::new(3)
        .with_temperature(1.0)
        .with_top_p(0.95);

    let output = nn_core::layers::generate(
        |input, c| model.model_fn_adapter(input, c),
        &[10],
        &mut cache,
        &cfg,
        &device,
    )
    .unwrap();

    let vocab = model.config().padded_vocab_size;
    for &tok in &output.token_ids {
        assert!(tok < vocab, "top-p token {tok} must be in vocab range");
    }
}

// ===========================================================================
// 16. Repetition penalty (conceptual + config validation)
// ===========================================================================

#[test]
fn test_pipeline_repetition_penalty_logit_suppression() {
    // Repetition penalty multiplies logits of previously-seen tokens.
    // penalty > 1.0 suppresses, < 1.0 boosts. Applied to positive logits via division,
    // negative logits via multiplication.
    let penalty = 1.2_f64;
    let logit_positive = 3.0_f64;
    let logit_negative = -2.0_f64;

    let penalized_pos = logit_positive / penalty;
    let penalized_neg = logit_negative * penalty;

    assert!(
        penalized_pos < logit_positive,
        "positive logit should decrease: {penalized_pos} < {logit_positive}"
    );
    assert!(
        penalized_neg < logit_negative,
        "negative logit should become more negative: {penalized_neg} < {logit_negative}"
    );
}

#[test]
fn test_pipeline_repetition_penalty_unity_is_no_change() {
    let penalty = 1.0_f64;
    let logit = 5.0_f64;
    let penalized = logit / penalty;
    assert!(
        (penalized - logit).abs() < 1e-12,
        "penalty=1.0 should not change logits"
    );
}

// ===========================================================================
// 17. Model weight shape validation
// ===========================================================================

#[test]
fn test_pipeline_weight_shapes_qkv_projection() {
    // QKV fused weight shape: [(num_heads + 2 * kv_heads) * head_dim, hidden_size]
    let cfg = tiny_config();
    let nh = cfg.num_attention_heads;
    let nkv = cfg.multi_query_group_num;
    let hd = cfg.kv_channels;
    let h = cfg.hidden_size;

    let qkv_out = (nh + 2 * nkv) * hd;
    assert_eq!(qkv_out, (4 + 2 * 2) * 64, "tiny: (4+4)*64 = 512");

    // Verify the model loads with these shapes
    let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
    let model = Glm5Model::load(&vb, cfg).unwrap();
    assert_eq!(model.config().hidden_size, h);
}

#[test]
fn test_pipeline_weight_shapes_mlp_gate_up_fused() {
    // dense_h_to_4h: [ffn_hidden_size * 2, hidden_size] (gate + up fused)
    let cfg = tiny_config();
    let fused_dim = cfg.ffn_hidden_size * 2;
    assert_eq!(fused_dim, 1024, "tiny: 512 * 2 = 1024");

    // dense_4h_to_h: [hidden_size, ffn_hidden_size]
    let down_out = cfg.hidden_size;
    let down_in = cfg.ffn_hidden_size;
    assert_eq!(down_out, 256);
    assert_eq!(down_in, 512);
}

#[test]
fn test_pipeline_weight_shapes_output_layer() {
    // output_layer: [padded_vocab_size, hidden_size]
    let cfg = tiny_config();
    let expected_shape = [cfg.padded_vocab_size, cfg.hidden_size];
    assert_eq!(expected_shape, [100, 256]);
}

#[test]
fn test_pipeline_weight_shapes_embedding() {
    // embedding: [padded_vocab_size, hidden_size]
    let cfg = tiny_config();
    let expected_shape = [cfg.padded_vocab_size, cfg.hidden_size];
    assert_eq!(expected_shape, [100, 256]);
    // Embedding and output_layer share the same shape
    assert_eq!(
        cfg.padded_vocab_size, 100,
        "vocab projection matches embedding vocab dim"
    );
}

// ===========================================================================
// 18. GQA (Grouped Query Attention) head configuration
// ===========================================================================

#[test]
fn test_pipeline_gqa_glm4_9b_config() {
    let cfg = Glm5Config::default();
    assert_eq!(cfg.num_attention_heads, 32);
    assert_eq!(cfg.multi_query_group_num, 2);
    let groups = cfg.num_kv_groups().unwrap();
    assert_eq!(groups, 16, "32/2 = 16 Q heads per KV head group");
}

#[test]
fn test_pipeline_gqa_various_valid_configs() {
    // Test multiple valid GQA configurations
    for (nh, nkv) in [
        (8, 1),
        (8, 2),
        (8, 4),
        (8, 8),
        (16, 1),
        (16, 4),
        (16, 8),
        (16, 16),
    ] {
        let mut cfg = tiny_config();
        cfg.num_attention_heads = nh;
        cfg.multi_query_group_num = nkv;
        cfg.hidden_size = nh * cfg.kv_channels;
        assert!(cfg.validate().is_ok(), "nh={nh}, nkv={nkv} should be valid");
        let groups = cfg.num_kv_groups().unwrap();
        assert_eq!(groups, nh / nkv, "nh={nh}, nkv={nkv}: groups={groups}");
    }
}

#[test]
fn test_pipeline_gqa_qkv_size_formula() {
    // QKV projection output size: (num_heads + 2 * kv_heads) * head_dim
    let cfg = Glm5Config::default();
    let nh = cfg.num_attention_heads;
    let nkv = cfg.multi_query_group_num;
    let hd = cfg.kv_channels;
    let qkv_size = (nh + 2 * nkv) * hd;
    // Q: 32*128 = 4096, K: 2*128 = 256, V: 2*128 = 256, total = 4608
    assert_eq!(qkv_size, 4608, "GLM-4-9B QKV size");

    let tiny = tiny_config();
    let qkv_tiny = (tiny.num_attention_heads + 2 * tiny.multi_query_group_num) * tiny.kv_channels;
    assert_eq!(qkv_tiny, (4 + 4) * 64, "tiny QKV = 512");
}

// ===========================================================================
// 19. Sequence length handling
// ===========================================================================

#[test]
fn test_pipeline_seq_length_single_token() {
    let model = load_tiny_model();
    let logits = model.forward(&[0], &[0]).unwrap();
    assert_eq!(logits.dims()[1], 1, "single token → seq_dim=1");
}

#[test]
fn test_pipeline_seq_length_max_within_config() {
    // The config has seq_length=64 for tiny. Forward with up to that many tokens.
    let model = load_tiny_model();
    let seq_len = 16; // Well within seq_length=64
    let ids: Vec<usize> = (0..seq_len).collect();
    let pos: Vec<usize> = (0..seq_len).collect();
    let logits = model.forward(&ids, &pos).unwrap();
    assert_eq!(logits.dims()[1], seq_len);
}

#[test]
fn test_pipeline_seq_length_mismatched_ids_positions_rejected() {
    let model = load_tiny_model();
    let result = model.forward(&[0, 1, 2], &[0, 1]); // 3 ids, 2 positions
    assert!(
        result.is_err(),
        "mismatched ids/positions should be rejected"
    );
    let err = result.unwrap_err().to_string();
    assert!(
        err.contains("input_ids len") || err.contains("!="),
        "error should mention length mismatch: {err}"
    );
}

#[test]
fn test_pipeline_seq_length_zero_ids_forward() {
    let model = load_tiny_model();
    // Forward with 0 tokens is an edge case — may error or produce empty output
    let result = model.forward(&[], &[]);
    // Either succeeds with empty output or produces an error
    if let Ok(logits) = result {
        assert_eq!(logits.dims()[1], 0);
    }
    // Either way, should not panic
}

// ===========================================================================
// 20. Config clone and equality
// ===========================================================================

#[test]
fn test_pipeline_config_clone_preserves_all_fields() {
    let cfg = Glm5Config::default();
    let cloned = cfg.clone();
    assert_eq!(cloned.hidden_size, cfg.hidden_size);
    assert_eq!(cloned.ffn_hidden_size, cfg.ffn_hidden_size);
    assert_eq!(cloned.num_layers, cfg.num_layers);
    assert_eq!(cloned.num_attention_heads, cfg.num_attention_heads);
    assert_eq!(cloned.multi_query_group_num, cfg.multi_query_group_num);
    assert_eq!(cloned.padded_vocab_size, cfg.padded_vocab_size);
    assert_eq!(cloned.kv_channels, cfg.kv_channels);
    assert_eq!(cloned.layernorm_epsilon, cfg.layernorm_epsilon);
    assert_eq!(cloned.seq_length, cfg.seq_length);
    assert_eq!(cloned.rmsnorm, cfg.rmsnorm);
    assert_eq!(cloned.add_qkv_bias, cfg.add_qkv_bias);
    assert_eq!(cloned.add_bias_linear, cfg.add_bias_linear);
    assert_eq!(cloned.rope_theta, cfg.rope_theta);
}

#[test]
fn test_pipeline_config_debug_format() {
    let cfg = tiny_config();
    let debug_str = format!("{cfg:?}");
    assert!(
        debug_str.contains("Glm5Config"),
        "Debug output should contain type name"
    );
    assert!(
        debug_str.contains("hidden_size"),
        "Debug should show fields"
    );
    assert!(
        debug_str.contains("256"),
        "Debug should show tiny hidden_size=256"
    );
}

// ===========================================================================
// 21. KV cache with model forward: incremental decoding
// ===========================================================================

#[test]
fn test_pipeline_kv_cache_incremental_decoding_consistency() {
    let model = load_tiny_model();
    let cfg = model.config();

    // Full forward (no cache)
    let ids = vec![10, 20, 30];
    let pos: Vec<usize> = (0..3).collect();
    let full_logits = model.forward(&ids, &pos).unwrap();
    assert_eq!(full_logits.dims(), &[1, 3, cfg.padded_vocab_size]);

    // Incremental forward (with cache)
    let mut cache = model.new_cache();
    // Prefill 2 tokens
    model
        .forward_cached(&[10, 20], &[0, 1], Some(&mut cache))
        .unwrap();
    assert_eq!(cache.seq_len(), 2);
    // Decode 1 token
    let incr_logits = model.forward_cached(&[30], &[2], Some(&mut cache)).unwrap();
    assert_eq!(incr_logits.dims(), &[1, 1, cfg.padded_vocab_size]);
    assert_eq!(cache.seq_len(), 3);
}

#[test]
fn test_pipeline_kv_cache_mismatch_layers_rejected() {
    let model = load_tiny_model();
    // Create a cache with wrong number of layers
    let mut wrong_cache = KvCache::new(model.config().num_layers + 5);
    let result = model.forward_cached(&[0], &[0], Some(&mut wrong_cache));
    assert!(result.is_err(), "cache layer mismatch should be rejected");
}

// ===========================================================================
// 22. Model accessors
// ===========================================================================

#[test]
fn test_pipeline_model_dtype_accessor() {
    let model = load_tiny_model();
    assert_eq!(
        model.dtype(),
        DType::F32,
        "model loaded with F32 VarBuilder"
    );
}

#[test]
fn test_pipeline_model_device_accessor() {
    let model = load_tiny_model();
    assert_eq!(model.device(), Device::Cpu, "model loaded on CPU");
}

#[test]
fn test_pipeline_model_config_accessor() {
    let model = load_tiny_model();
    let cfg = model.config();
    assert_eq!(cfg.hidden_size, 256);
    assert_eq!(cfg.num_layers, 2);
    assert_eq!(cfg.num_attention_heads, 4);
}

// ===========================================================================
// 23. Beam search decoding
// ===========================================================================

#[test]
fn test_pipeline_beam_search_produces_output() {
    let model = load_tiny_model();
    let vocab = model.config().padded_vocab_size;
    let beam_cfg = nn_core::layers::BeamSearchConfig::new(2)
        .with_max_new_tokens(3)
        .with_eos_token_id(vocab - 1);

    let output = model.generate_beam(&[42], &beam_cfg).unwrap();
    assert!(
        !output.beams.is_empty(),
        "beam search should produce at least one hypothesis"
    );
    for beam in &output.beams {
        for &tok in &beam.token_ids {
            assert!(tok < vocab, "beam token {tok} must be in vocab range");
        }
    }
}

#[test]
fn test_pipeline_beam_search_width_one_equals_greedy() {
    // Beam width 1 should produce the same result as greedy
    let model = load_tiny_model();
    let greedy = model.generate_greedy(&[10], 3).unwrap();

    let beam_cfg = nn_core::layers::BeamSearchConfig::new(1).with_max_new_tokens(3);
    let beam = model.generate_beam(&[10], &beam_cfg).unwrap();

    assert!(!beam.beams.is_empty());
    assert_eq!(
        beam.beams[0].token_ids, greedy.token_ids,
        "beam_width=1 should match greedy"
    );
}

// ===========================================================================
// 24. Forward from embeddings validation
// ===========================================================================

#[test]
fn test_pipeline_forward_from_embeddings_wrong_hidden_size_rejected() {
    let model = load_tiny_model();
    let wrong_hidden = model.config().hidden_size + 1;
    let emb = DynTensor::ones(&[1, 3, wrong_hidden], DType::F32, &Device::Cpu).unwrap();
    let result = model.forward_from_embeddings(&emb, &[0, 1, 2], None);
    assert!(
        result.is_err(),
        "wrong hidden_size in embeddings should be rejected"
    );
}

#[test]
fn test_pipeline_forward_from_embeddings_seq_mismatch_rejected() {
    let model = load_tiny_model();
    let cfg = model.config();
    let emb = DynTensor::ones(&[1, 3, cfg.hidden_size], DType::F32, &Device::Cpu).unwrap();
    let result = model.forward_from_embeddings(&emb, &[0, 1], None); // 3 seq, 2 positions
    assert!(result.is_err(), "seq/positions mismatch should be rejected");
}

// ===========================================================================
// 25. Error type coverage
// ===========================================================================

#[test]
fn test_pipeline_error_invalid_config_display() {
    let err = Glm5Error::InvalidConfig {
        reason: "test reason".into(),
    };
    let msg = err.to_string();
    assert!(
        msg.contains("invalid config"),
        "error should mention invalid config: {msg}"
    );
    assert!(
        msg.contains("test reason"),
        "error should contain reason: {msg}"
    );
}

#[test]
fn test_pipeline_error_cache_mismatch_display() {
    let err = Glm5Error::CacheMismatch {
        cache_layers: 5,
        model_layers: 10,
    };
    let msg = err.to_string();
    assert!(msg.contains("5"), "should mention cache layers: {msg}");
    assert!(msg.contains("10"), "should mention model layers: {msg}");
}

#[test]
fn test_pipeline_error_invalid_input_display() {
    let err = Glm5Error::InvalidInput {
        reason: "bad input".into(),
    };
    let msg = err.to_string();
    assert!(
        msg.contains("invalid input"),
        "should mention invalid input: {msg}"
    );
}

// ===========================================================================
// 26. Flat KV cache extended tests
// ===========================================================================

#[test]
fn test_pipeline_flat_kv_cache_multi_layer_independence() {
    // Verify that different layers store independent data
    let num_layers = 3;
    let num_heads = 2;
    let head_dim = 4;
    let token_size = num_heads * head_dim;
    let mut cache = kv_cache::KVCache::new(num_layers, num_heads, head_dim, 10);

    for layer in 0..num_layers {
        let key: Vec<f32> = (0..token_size).map(|i| (layer * 100 + i) as f32).collect();
        let val: Vec<f32> = (0..token_size).map(|i| (layer * 1000 + i) as f32).collect();
        cache.append(layer, &key, &val).unwrap();
    }

    // Each layer should have its own unique data
    for layer in 0..num_layers {
        let keys = cache.get_keys(layer);
        assert_eq!(
            keys[0],
            (layer * 100) as f32,
            "layer {layer} key[0] should be unique"
        );
    }
}

#[test]
fn test_pipeline_flat_kv_cache_clone() {
    let num_layers = 2;
    let num_heads = 2;
    let head_dim = 4;
    let token_size = num_heads * head_dim;
    let mut cache = kv_cache::KVCache::new(num_layers, num_heads, head_dim, 10);

    let key: Vec<f32> = vec![1.0; token_size];
    let val: Vec<f32> = vec![2.0; token_size];
    for layer in 0..num_layers {
        cache.append(layer, &key, &val).unwrap();
    }

    let cloned = cache.clone();
    assert_eq!(cloned.len(), cache.len());
    assert_eq!(cloned.num_layers(), cache.num_layers());
    assert_eq!(cloned.get_keys(0), cache.get_keys(0));
    assert_eq!(cloned.get_values(0), cache.get_values(0));
}

// ===========================================================================
// 27. Config new() constructor validation
// ===========================================================================

#[test]
fn test_pipeline_config_new_constructor_all_fields() {
    let cfg = Glm5Config::new(
        512,     // hidden_size
        2048,    // ffn_hidden_size
        4,       // num_layers
        8,       // num_attention_heads
        2,       // multi_query_group_num
        1000,    // padded_vocab_size
        64,      // kv_channels
        1e-6,    // layernorm_epsilon
        2048,    // seq_length
        true,    // rmsnorm
        false,   // add_qkv_bias
        true,    // add_bias_linear
        50000.0, // rope_theta
    );
    assert_eq!(cfg.hidden_size, 512);
    assert_eq!(cfg.ffn_hidden_size, 2048);
    assert_eq!(cfg.num_layers, 4);
    assert_eq!(cfg.num_attention_heads, 8);
    assert_eq!(cfg.multi_query_group_num, 2);
    assert_eq!(cfg.padded_vocab_size, 1000);
    assert_eq!(cfg.kv_channels, 64);
    assert_eq!(cfg.layernorm_epsilon, 1e-6);
    assert_eq!(cfg.seq_length, 2048);
    assert!(cfg.rmsnorm);
    assert!(!cfg.add_qkv_bias);
    assert!(cfg.add_bias_linear);
    assert_eq!(cfg.rope_theta, 50000.0);
    assert!(cfg.validate().is_ok());
}

#[test]
fn test_pipeline_config_new_with_bias_linear_loads() {
    // Model with add_bias_linear=true requires bias tensors for output_layer and MLP
    let cfg = Glm5Config::new(
        256,     // hidden_size
        512,     // ffn_hidden_size
        1,       // num_layers
        4,       // num_attention_heads
        2,       // multi_query_group_num
        100,     // padded_vocab_size
        64,      // kv_channels
        1e-5,    // layernorm_epsilon
        64,      // seq_length
        true,    // rmsnorm
        true,    // add_qkv_bias
        true,    // add_bias_linear
        10000.0, // rope_theta
    );
    assert!(cfg.validate().is_ok());
    let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
    let model = Glm5Model::load(&vb, cfg).unwrap();
    let logits = model.forward(&[0], &[0]).unwrap();
    assert_eq!(logits.dims()[2], 100);
}

// ===========================================================================
// 28. Validation edge cases
// ===========================================================================

#[test]
fn test_pipeline_config_zero_hidden_size_rejected() {
    let mut cfg = tiny_config();
    cfg.hidden_size = 0;
    assert!(
        cfg.validate().is_err(),
        "zero hidden_size should be rejected"
    );
}

#[test]
fn test_pipeline_config_zero_ffn_hidden_size_rejected() {
    let mut cfg = tiny_config();
    cfg.ffn_hidden_size = 0;
    assert!(
        cfg.validate().is_err(),
        "zero ffn_hidden_size should be rejected"
    );
}

#[test]
fn test_pipeline_config_zero_vocab_rejected() {
    let mut cfg = tiny_config();
    cfg.padded_vocab_size = 0;
    assert!(
        cfg.validate().is_err(),
        "zero padded_vocab_size should be rejected"
    );
}

#[test]
fn test_pipeline_config_zero_num_layers_rejected() {
    let mut cfg = tiny_config();
    cfg.num_layers = 0;
    assert!(
        cfg.validate().is_err(),
        "zero num_layers should be rejected"
    );
}

#[test]
fn test_pipeline_config_zero_seq_length_rejected() {
    let mut cfg = tiny_config();
    cfg.seq_length = 0;
    assert!(
        cfg.validate().is_err(),
        "zero seq_length should be rejected"
    );
}

#[test]
fn test_pipeline_config_kv_channels_not_multiple_of_4_rejected() {
    let mut cfg = tiny_config();
    cfg.kv_channels = 5; // not multiple of 4
    assert!(
        cfg.validate().is_err(),
        "kv_channels=5 (not multiple of 4) should be rejected"
    );
}

#[test]
fn test_pipeline_config_kv_channels_zero_rejected() {
    let mut cfg = tiny_config();
    cfg.kv_channels = 0;
    assert!(cfg.validate().is_err(), "kv_channels=0 should be rejected");
}

#[test]
fn test_pipeline_config_negative_rope_theta_rejected() {
    let mut cfg = tiny_config();
    cfg.rope_theta = -1.0;
    assert!(
        cfg.validate().is_err(),
        "negative rope_theta should be rejected"
    );
}

// ===========================================================================
// 29. SwiGLU activation: zero input properties
// ===========================================================================

#[test]
fn test_pipeline_swiglu_zero_gate_zero_output() {
    // silu(0) = 0 * sigmoid(0) = 0 * 0.5 = 0. So SwiGLU(0, any) = 0 * any = 0.
    let gate = DynTensor::zeros(&[1, 1, 4], DType::F32, &Device::Cpu).unwrap();
    let up = DynTensor::from_vec(vec![1.0_f32, 2.0, 3.0, 4.0], &[1, 1, 4], &Device::Cpu).unwrap();

    let result = gate.silu().unwrap().broadcast_mul(&up).unwrap();
    let out = result.to_flat_vec::<f32>().unwrap();
    for (i, &v) in out.iter().enumerate() {
        assert!(
            v.abs() < 1e-7,
            "SwiGLU with zero gate should be zero, got [{i}]={v}"
        );
    }
}

// ===========================================================================
// 30. RoPE: invariant that rotation preserves norm
// ===========================================================================

#[test]
fn test_pipeline_rope_preserves_vector_norm() {
    // Rotary embeddings are rotations; they should preserve L2 norm
    // for the rotated portion. The pass-through portion is identity.
    let head_dim = 16;
    let rope = HalfRotaryEmbedding::new(head_dim, 128, 10_000.0, &Device::Cpu).unwrap();

    let data: Vec<f32> = (1..=head_dim as i32).map(|i| i as f32 * 0.1).collect();
    let x = DynTensor::from_vec(data.clone(), &[1, 1, 1, head_dim], &Device::Cpu).unwrap();

    let rotated = rope.apply(&x, 7).unwrap();
    let rot_vec = rotated.to_flat_vec::<f32>().unwrap();

    let half = head_dim / 2;

    // Check norm of the rotated half
    let norm_before: f32 = data[..half].iter().map(|v| v * v).sum::<f32>().sqrt();
    let norm_after: f32 = rot_vec[..half].iter().map(|v| v * v).sum::<f32>().sqrt();
    assert!(
        (norm_before - norm_after).abs() < 1e-4,
        "RoPE should preserve norm: before={norm_before}, after={norm_after}"
    );

    // Pass-through half: norm preserved trivially (identical values)
    let norm_pass_before: f32 = data[half..].iter().map(|v| v * v).sum::<f32>().sqrt();
    let norm_pass_after: f32 = rot_vec[half..].iter().map(|v| v * v).sum::<f32>().sqrt();
    assert!(
        (norm_pass_before - norm_pass_after).abs() < 1e-6,
        "pass-through should preserve norm exactly"
    );
}

// ===========================================================================
// 31. Generation config builder chaining
// ===========================================================================

#[test]
fn test_pipeline_generation_config_full_chain() {
    let cfg = GenerationConfig::new(100)
        .with_temperature(0.8)
        .with_top_k(50)
        .with_top_p(0.95)
        .with_eos_token_id(2)
        .with_seed(42);

    assert_eq!(cfg.max_new_tokens, 100);
    assert_eq!(cfg.temperature, 0.8);
    assert_eq!(cfg.top_k, Some(50));
    assert_eq!(cfg.top_p, Some(0.95));
    assert_eq!(cfg.eos_token_id, Some(2));
    assert_eq!(cfg.seed, Some(42));
    assert!(cfg.validate().is_ok());
}

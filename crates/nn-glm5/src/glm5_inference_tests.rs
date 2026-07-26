// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Generation and inference tests for GLM-4/5 decoder-only transformer.
//!
//! Covers:
//! - Rotary embedding cos/sin numerical values at known positions
//! - GQA Q/K/V projection shape verification post-split
//! - SwiGLU FFN gate * silu(up) computation pattern
//! - KV cache growth and truncation under capacity limits
//! - Top-k / top-p sampling via GenerationConfig
//! - Temperature effects on generation diversity
//! - EOS token early stopping
//! - Forward pass input validation (mismatched lengths, cache mismatch)
//! - Incremental decode numerical consistency
//!
//! Part of #4186

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

// ---------------------------------------------------------------------------
// Rotary embedding: cos/sin values at known positions
// ---------------------------------------------------------------------------

#[test]
fn test_rope_position_zero_cos_one_sin_zero() {
    // At position 0: angle = 0 for all frequencies, so cos(0)=1, sin(0)=0.
    // Half-RoPE means rotation is identity at position 0.
    let head_dim = 8;
    let rope = HalfRotaryEmbedding::new(head_dim, 64, 10_000.0, &Device::Cpu).unwrap();

    // Input with distinct values per dim
    let data: Vec<f32> = (1..=head_dim as i32).map(|i| i as f32).collect();
    let x = DynTensor::from_vec(data.clone(), &[1, 1, 1, head_dim], &Device::Cpu).unwrap();
    let result = rope.apply(&x, 0).unwrap();
    let out = result.to_flat_vec::<f32>().unwrap();

    // At position 0, all dims should be unchanged (cos=1, sin=0 => identity rotation)
    for i in 0..head_dim {
        assert!(
            (out[i] - data[i]).abs() < 1e-5,
            "pos=0: dim {i} should be identity, expected {} got {}",
            data[i],
            out[i]
        );
    }
}

#[test]
fn test_rope_nonzero_position_rotates_first_half_only() {
    // Half-RoPE: first head_dim/2 dims are rotated, rest unchanged.
    // At position > 0, first half should differ from input.
    let head_dim = 16;
    let rope = HalfRotaryEmbedding::new(head_dim, 128, 10_000.0, &Device::Cpu).unwrap();

    let data: Vec<f32> = (1..=head_dim as i32).map(|i| i as f32).collect();
    let x = DynTensor::from_vec(data.clone(), &[1, 1, 1, head_dim], &Device::Cpu).unwrap();
    let result = rope.apply(&x, 7).unwrap(); // position 7
    let out = result.to_flat_vec::<f32>().unwrap();

    let half = head_dim / 2;

    // First half should be different (rotated)
    let first_half_diff: f32 = (0..half).map(|i| (out[i] - data[i]).abs()).sum();
    assert!(
        first_half_diff > 0.01,
        "first half should be rotated at pos=7, total diff={first_half_diff}"
    );

    // Second half should be unchanged (pass-through)
    for i in half..head_dim {
        assert!(
            (out[i] - data[i]).abs() < 1e-6,
            "second half dim {i} should pass through, expected {} got {}",
            data[i],
            out[i]
        );
    }
}

#[test]
fn test_rope_pair_consistency() {
    // apply_pair(q, k) should match apply(q) and apply(k) independently
    // for the same positions.
    let head_dim = 8;
    let rope = HalfRotaryEmbedding::new(head_dim, 64, 10_000.0, &Device::Cpu).unwrap();

    let q_data: Vec<f32> = (0..head_dim).map(|i| (i as f32 + 1.0) * 0.1).collect();
    let k_data: Vec<f32> = (0..head_dim).map(|i| (i as f32 + 1.0) * 0.2).collect();
    let q = DynTensor::from_vec(q_data, &[1, 1, 1, head_dim], &Device::Cpu).unwrap();
    let k = DynTensor::from_vec(k_data, &[1, 1, 1, head_dim], &Device::Cpu).unwrap();

    let (q_pair, k_pair) = rope.apply_pair(&q, &k, &[3]).unwrap();
    let q_solo = rope.apply(&q, 3).unwrap();
    let k_solo = rope.apply(&k, 3).unwrap();

    let q_pair_flat = q_pair.to_flat_vec::<f32>().unwrap();
    let q_solo_flat = q_solo.to_flat_vec::<f32>().unwrap();
    let k_pair_flat = k_pair.to_flat_vec::<f32>().unwrap();
    let k_solo_flat = k_solo.to_flat_vec::<f32>().unwrap();

    for i in 0..head_dim {
        assert!(
            (q_pair_flat[i] - q_solo_flat[i]).abs() < 1e-5,
            "Q dim {i}: pair {} vs solo {}",
            q_pair_flat[i],
            q_solo_flat[i]
        );
        assert!(
            (k_pair_flat[i] - k_solo_flat[i]).abs() < 1e-5,
            "K dim {i}: pair {} vs solo {}",
            k_pair_flat[i],
            k_solo_flat[i]
        );
    }
}

#[test]
fn test_rope_different_positions_yield_different_outputs() {
    let head_dim = 8;
    let rope = HalfRotaryEmbedding::new(head_dim, 64, 10_000.0, &Device::Cpu).unwrap();

    let data: Vec<f32> = vec![1.0; head_dim];
    let x = DynTensor::from_vec(data, &[1, 1, 1, head_dim], &Device::Cpu).unwrap();

    let out_pos0 = rope.apply(&x, 0).unwrap().to_flat_vec::<f32>().unwrap();
    let out_pos1 = rope.apply(&x, 1).unwrap().to_flat_vec::<f32>().unwrap();
    let out_pos10 = rope.apply(&x, 10).unwrap().to_flat_vec::<f32>().unwrap();

    // Position 0 is identity; position 1 and 10 should differ
    let diff_01: f32 = out_pos0
        .iter()
        .zip(out_pos1.iter())
        .map(|(a, b)| (a - b).abs())
        .sum();
    let diff_010: f32 = out_pos0
        .iter()
        .zip(out_pos10.iter())
        .map(|(a, b)| (a - b).abs())
        .sum();

    assert!(
        diff_01 > 1e-4,
        "positions 0 and 1 should produce different outputs, diff={diff_01}"
    );
    assert!(
        diff_010 > 1e-4,
        "positions 0 and 10 should produce different outputs, diff={diff_010}"
    );
    // Larger position difference should generally yield larger rotation difference
    assert!(
        diff_010 > diff_01,
        "pos 10 should differ more than pos 1 from pos 0: {diff_010} vs {diff_01}"
    );
}

// ---------------------------------------------------------------------------
// GQA attention: Q, K, V projection shape verification
// ---------------------------------------------------------------------------

#[test]
fn test_gqa_qkv_split_dimensions() {
    // Verify the math: fused QKV output splits into Q, K, V correctly.
    let cfg = tiny_config();
    // num_heads=4, multi_query_group_num=2, kv_channels=64
    let nh = cfg.num_attention_heads;
    let nkv = cfg.multi_query_group_num;
    let hd = cfg.kv_channels;

    let q_size = nh * hd; // 4 * 64 = 256
    let k_size = nkv * hd; // 2 * 64 = 128
    let v_size = nkv * hd; // 2 * 64 = 128
    let total_qkv = q_size + k_size + v_size; // 512

    assert_eq!(q_size, 256);
    assert_eq!(k_size, 128);
    assert_eq!(v_size, 128);
    assert_eq!(total_qkv, 512);
    assert_eq!(total_qkv, (nh + 2 * nkv) * hd);
}

#[test]
fn test_gqa_q_reshaped_to_batch_heads_seq_dim() {
    // After QKV split, Q is reshaped from [B, S, nh*hd] to [B, nh, S, hd]
    let cfg = tiny_config();
    let batch = 1;
    let seq_len = 3;
    let nh = cfg.num_attention_heads;
    let hd = cfg.kv_channels;
    let q_flat_dim = nh * hd; // 256

    // Simulate reshape + transpose
    let q_flat = DynTensor::ones(&[batch, seq_len, q_flat_dim], DType::F32, &Device::Cpu).unwrap();
    let q_reshaped = q_flat.reshape([batch, seq_len, nh, hd]).unwrap();
    let q_transposed = q_reshaped.transpose(1, 2).unwrap();

    assert_eq!(
        q_transposed.dims(),
        &[batch, nh, seq_len, hd],
        "Q should reshape to [B, nh, S, hd]"
    );
}

#[test]
fn test_gqa_kv_reshaped_to_batch_kvheads_seq_dim() {
    // K/V are reshaped from [B, S, nkv*hd] to [B, nkv, S, hd]
    let cfg = tiny_config();
    let batch = 1;
    let seq_len = 5;
    let nkv = cfg.multi_query_group_num;
    let hd = cfg.kv_channels;
    let kv_flat_dim = nkv * hd; // 128

    let k_flat = DynTensor::ones(&[batch, seq_len, kv_flat_dim], DType::F32, &Device::Cpu).unwrap();
    let k_reshaped = k_flat.reshape([batch, seq_len, nkv, hd]).unwrap();
    let k_transposed = k_reshaped.transpose(1, 2).unwrap();

    assert_eq!(
        k_transposed.dims(),
        &[batch, nkv, seq_len, hd],
        "K should reshape to [B, nkv, S, hd]"
    );
}

#[test]
fn test_gqa_repeat_kv_expands_to_query_heads() {
    // After GQA repeat, K/V should have num_heads dim matching Q
    let cfg = tiny_config();
    let batch = 1;
    let seq_len = 3;
    let nkv = cfg.multi_query_group_num; // 2
    let nh = cfg.num_attention_heads; // 4
    let hd = cfg.kv_channels; // 64
    let repeat_factor = nh / nkv; // 2

    let k = DynTensor::ones(&[batch, nkv, seq_len, hd], DType::F32, &Device::Cpu).unwrap();
    let k_expanded = nn_core::layers::repeat_kv(&k, repeat_factor).unwrap();

    assert_eq!(
        k_expanded.dims(),
        &[batch, nh, seq_len, hd],
        "repeated K should match Q head count"
    );
}

// ---------------------------------------------------------------------------
// SwiGLU FFN computation pattern
// ---------------------------------------------------------------------------

#[test]
fn test_swiglu_split_and_activation_shape() {
    // SwiGLU: dense_h_to_4h outputs [B, S, ffn*2], split into gate and up of [B, S, ffn]
    let cfg = tiny_config();
    let batch = 1;
    let seq_len = 3;
    let ffn = cfg.ffn_hidden_size; // 512
    let fused_dim = ffn * 2; // 1024

    // Simulate the intermediate tensor after dense_h_to_4h
    let intermediate =
        DynTensor::ones(&[batch, seq_len, fused_dim], DType::F32, &Device::Cpu).unwrap();

    let gate = intermediate.narrow(2, 0, ffn).unwrap();
    let up = intermediate.narrow(2, ffn, ffn).unwrap();

    assert_eq!(gate.dims(), &[batch, seq_len, ffn]);
    assert_eq!(up.dims(), &[batch, seq_len, ffn]);
}

#[test]
fn test_swiglu_silu_gate_times_up() {
    // SwiGLU computes silu(gate) * up. Verify numerically.
    // silu(x) = x * sigmoid(x) = x / (1 + exp(-x))
    let gate_vals = vec![0.0_f32, 1.0, -1.0, 2.0];
    let up_vals = vec![1.0_f32, 2.0, 3.0, 0.5];

    let gate = DynTensor::from_vec(gate_vals.clone(), &[1, 1, 4], &Device::Cpu).unwrap();
    let up = DynTensor::from_vec(up_vals.clone(), &[1, 1, 4], &Device::Cpu).unwrap();

    let activated = gate.silu().unwrap().broadcast_mul(&up).unwrap();
    let out = activated.to_flat_vec::<f32>().unwrap();

    for i in 0..4 {
        let g = gate_vals[i];
        let u = up_vals[i];
        let silu_g = g / (1.0 + (-g).exp());
        let expected = silu_g * u;
        assert!(
            (out[i] - expected).abs() < 1e-5,
            "SwiGLU[{i}]: expected {expected}, got {}",
            out[i]
        );
    }
}

#[test]
fn test_swiglu_zero_gate_produces_zero() {
    // silu(0) = 0 * sigmoid(0) = 0 * 0.5 = 0, so output is always 0
    let gate = DynTensor::zeros(&[1, 1, 8], DType::F32, &Device::Cpu).unwrap();
    let up = DynTensor::ones(&[1, 1, 8], DType::F32, &Device::Cpu).unwrap();

    let result = gate.silu().unwrap().broadcast_mul(&up).unwrap();
    let flat = result.to_flat_vec::<f32>().unwrap();

    for (i, &v) in flat.iter().enumerate() {
        assert!(
            v.abs() < 1e-7,
            "SwiGLU with zero gate should be zero, dim {i} got {v}"
        );
    }
}

// ---------------------------------------------------------------------------
// KV cache: flat KVCache growth and truncation
// ---------------------------------------------------------------------------

#[test]
fn test_flat_kv_cache_capacity_enforcement() {
    // KVCache with max_seq_len=3 should reject the 4th token
    let num_layers = 2;
    let num_heads = 2;
    let head_dim = 4;
    let token_size = num_heads * head_dim;
    let mut cache = kv_cache::KVCache::new(num_layers, num_heads, head_dim, 3);

    let key: Vec<f32> = vec![1.0; token_size];
    let val: Vec<f32> = vec![2.0; token_size];

    // Fill 3 tokens
    for _tok in 0..3 {
        for layer in 0..num_layers {
            cache.append(layer, &key, &val).unwrap();
        }
    }
    assert_eq!(cache.len(), 3);

    // 4th token should fail on layer 0
    let result = cache.append(0, &key, &val);
    assert!(result.is_err(), "should reject token beyond max_seq_len");
}

#[test]
fn test_flat_kv_cache_clear_allows_reuse_to_capacity() {
    let num_layers = 1;
    let num_heads = 2;
    let head_dim = 4;
    let max_seq = 2;
    let token_size = num_heads * head_dim;
    let mut cache = kv_cache::KVCache::new(num_layers, num_heads, head_dim, max_seq);

    let key: Vec<f32> = vec![1.0; token_size];
    let val: Vec<f32> = vec![2.0; token_size];

    // Fill to capacity
    for _tok in 0..max_seq {
        cache.append(0, &key, &val).unwrap();
    }
    assert_eq!(cache.len(), max_seq);

    // Clear and refill
    cache.clear();
    assert_eq!(cache.len(), 0);

    for _tok in 0..max_seq {
        cache.append(0, &key, &val).unwrap();
    }
    assert_eq!(cache.len(), max_seq);
}

#[test]
fn test_dyntensor_kv_cache_layer_count_matches_model() {
    let model = load_tiny_model();
    let cache = model.new_cache();
    assert_eq!(
        cache.num_layers(),
        model.config().num_layers,
        "cache layers should match model layers"
    );
}

// ---------------------------------------------------------------------------
// GenerationConfig: top-k, top-p, temperature, EOS
// ---------------------------------------------------------------------------

#[test]
fn test_generation_config_builder_pattern() {
    let cfg = GenerationConfig::new(50)
        .with_temperature(0.8)
        .with_top_k(40)
        .with_top_p(0.95)
        .with_eos_token_id(2);

    assert_eq!(cfg.max_new_tokens, 50);
    assert!((cfg.temperature - 0.8).abs() < 1e-12);
    assert_eq!(cfg.top_k, Some(40));
    assert_eq!(cfg.top_p, Some(0.95));
    assert_eq!(cfg.eos_token_id, Some(2));
}

#[test]
fn test_generation_config_default_is_greedy() {
    let cfg = GenerationConfig::default();
    assert!(
        (cfg.temperature - 0.0).abs() < 1e-12,
        "default is greedy (temp=0)"
    );
    assert!(cfg.top_k.is_none());
    assert!(cfg.top_p.is_none());
    assert!(cfg.eos_token_id.is_none());
}

#[test]
fn test_greedy_with_custom_config_produces_tokens() {
    // Use the generate infrastructure with a custom GenerationConfig
    let model = load_tiny_model();
    let device = model.device();
    let mut cache = model.new_cache();
    let cfg = GenerationConfig::new(5).with_temperature(0.0);

    let output = nn_core::layers::generate(
        |input, c| model.model_fn_adapter(input, c),
        &[42],
        &mut cache,
        &cfg,
        &device,
    )
    .unwrap();

    assert_eq!(output.token_ids.len(), 5);
}

#[test]
fn test_eos_token_stops_generation_early() {
    // With zero weights, all logits are equal (0.0). argmax with total_cmp
    // tiebreaks to the LAST index, so generated token = padded_vocab_size - 1.
    // Set eos_token_id to that value to trigger early stopping.
    let model = load_tiny_model();
    let vocab = model.config().padded_vocab_size; // 100
    let eos_id = vocab - 1; // 99 -- the token argmax picks with zero weights
    let device = model.device();
    let mut cache = model.new_cache();
    let cfg = GenerationConfig::new(10).with_eos_token_id(eos_id);

    let output = nn_core::layers::generate(
        |input, c| model.model_fn_adapter(input, c),
        &[42],
        &mut cache,
        &cfg,
        &device,
    )
    .unwrap();

    // Should stop after 1 token: first generated token (99) matches EOS
    assert_eq!(
        output.token_ids.len(),
        1,
        "EOS={eos_id} should stop generation after first token, got {} tokens: {:?}",
        output.token_ids.len(),
        output.token_ids
    );
    assert!(
        output.finished,
        "generation should report finished=true when EOS is hit"
    );
    assert_eq!(
        output.token_ids[0], eos_id,
        "first generated token should be {eos_id} (argmax tiebreaks to last index)"
    );
}

#[test]
fn test_generation_without_eos_runs_to_max() {
    // With zero weights, argmax tiebreaks to last vocab index every step.
    // No EOS set, so generation runs to max_new_tokens.
    let model = load_tiny_model();
    let vocab = model.config().padded_vocab_size;
    let expected_token = vocab - 1; // 99
    let output = model.generate_greedy(&[42], 5).unwrap();

    assert_eq!(
        output.token_ids.len(),
        5,
        "without EOS, should produce exactly max_new_tokens=5 tokens"
    );
    // All tokens should be vocab-1 (argmax tiebreaks to last index with zero logits)
    for (i, &tok) in output.token_ids.iter().enumerate() {
        assert_eq!(
            tok, expected_token,
            "token {i} should be {expected_token} with zero weights"
        );
    }
}

#[test]
fn test_top_k_config_accepted() {
    // Verify top_k config does not cause an error (with temp=0, argmax still used)
    let model = load_tiny_model();
    let device = model.device();
    let mut cache = model.new_cache();
    let cfg = GenerationConfig::new(3).with_top_k(10);

    let output = nn_core::layers::generate(
        |input, c| model.model_fn_adapter(input, c),
        &[42],
        &mut cache,
        &cfg,
        &device,
    )
    .unwrap();

    assert_eq!(output.token_ids.len(), 3);
}

#[test]
fn test_top_p_config_accepted() {
    // Verify top_p config does not cause an error
    let model = load_tiny_model();
    let device = model.device();
    let mut cache = model.new_cache();
    let cfg = GenerationConfig::new(3).with_top_p(0.9);

    let output = nn_core::layers::generate(
        |input, c| model.model_fn_adapter(input, c),
        &[42],
        &mut cache,
        &cfg,
        &device,
    )
    .unwrap();

    assert_eq!(output.token_ids.len(), 3);
}

// ---------------------------------------------------------------------------
// Forward pass input validation
// ---------------------------------------------------------------------------

#[test]
fn test_forward_mismatched_ids_positions_errors() {
    let model = load_tiny_model();
    // 3 input IDs but only 2 positions
    let result = model.forward(&[0, 1, 2], &[0, 1]);
    assert!(
        result.is_err(),
        "mismatched input_ids and positions should error"
    );
    let err_msg = result.unwrap_err().to_string();
    assert!(
        err_msg.contains("len"),
        "error should mention length mismatch: {err_msg}"
    );
}

#[test]
fn test_forward_cached_mismatched_cache_layers_errors() {
    let model = load_tiny_model();
    // Create a cache with wrong number of layers
    let wrong_layers = model.config().num_layers + 1;
    let mut bad_cache = KvCache::new(wrong_layers);

    let result = model.forward_cached(&[0], &[0], Some(&mut bad_cache));
    assert!(result.is_err(), "cache with wrong layer count should error");
    let err_msg = result.unwrap_err().to_string();
    assert!(
        err_msg.contains("cache") || err_msg.contains("mismatch"),
        "error should mention cache mismatch: {err_msg}"
    );
}

#[test]
fn test_forward_from_embeddings_wrong_hidden_size_errors() {
    let model = load_tiny_model();
    let cfg = model.config();
    // Wrong hidden size: use hidden_size + 1
    let bad_emb = DynTensor::ones(&[1, 3, cfg.hidden_size + 1], DType::F32, &Device::Cpu).unwrap();

    let result = model.forward_from_embeddings(&bad_emb, &[0, 1, 2], None);
    assert!(
        result.is_err(),
        "wrong hidden_size in embeddings should error"
    );
}

#[test]
fn test_forward_from_embeddings_wrong_seq_len_errors() {
    let model = load_tiny_model();
    let cfg = model.config();
    // Embedding with seq_len=3 but positions only have 2 entries
    let emb = DynTensor::ones(&[1, 3, cfg.hidden_size], DType::F32, &Device::Cpu).unwrap();

    let result = model.forward_from_embeddings(&emb, &[0, 1], None);
    assert!(
        result.is_err(),
        "mismatched embedding seq_len and positions should error"
    );
}

// ---------------------------------------------------------------------------
// Incremental decode: single-token steps match batch decode
// ---------------------------------------------------------------------------

#[test]
fn test_incremental_single_token_matches_batch() {
    // Forward [0,1,2] in one call vs one-by-one with cache should produce
    // the same last-position logits.
    let model = load_tiny_model();
    let vocab = model.config().padded_vocab_size;

    // Batch forward (no cache)
    let batch_logits = model.forward(&[0, 1, 2], &[0, 1, 2]).unwrap();
    let batch_flat = batch_logits.to_flat_vec::<f32>().unwrap();
    // Last position logits
    let batch_last = &batch_flat[2 * vocab..3 * vocab];

    // Incremental forward (with cache)
    let mut cache = model.new_cache();
    let _ = model.forward_cached(&[0], &[0], Some(&mut cache)).unwrap();
    let _ = model.forward_cached(&[1], &[1], Some(&mut cache)).unwrap();
    let incr_logits = model.forward_cached(&[2], &[2], Some(&mut cache)).unwrap();
    let incr_flat = incr_logits.to_flat_vec::<f32>().unwrap();

    assert_eq!(incr_flat.len(), vocab);
    for (i, (a, b)) in batch_last.iter().zip(incr_flat.iter()).enumerate() {
        assert!(
            (a - b).abs() < 1e-5,
            "logit mismatch at vocab {i}: batch={a}, incremental={b}"
        );
    }
}

#[test]
fn test_prefill_then_decode_logits_finite() {
    let model = load_tiny_model();
    let mut cache = model.new_cache();

    // Prefill
    let prefill_logits = model
        .forward_cached(&[0, 1, 2, 3, 4], &[0, 1, 2, 3, 4], Some(&mut cache))
        .unwrap();
    let prefill_flat = prefill_logits.to_flat_vec::<f32>().unwrap();
    assert!(
        prefill_flat.iter().all(|v| v.is_finite()),
        "prefill logits should all be finite"
    );

    // Decode steps
    for step in 0..5 {
        let pos = 5 + step;
        let decode_logits = model
            .forward_cached(&[42], &[pos], Some(&mut cache))
            .unwrap();
        let decode_flat = decode_logits.to_flat_vec::<f32>().unwrap();
        assert!(
            decode_flat.iter().all(|v| v.is_finite()),
            "decode step {step} logits should all be finite"
        );
    }
}

// ---------------------------------------------------------------------------
// Model dtype and device consistency
// ---------------------------------------------------------------------------

#[test]
fn test_model_dtype_matches_varbuilder() {
    let cfg = tiny_config();
    let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
    let model = Glm5Model::load(&vb, cfg).unwrap();
    assert_eq!(model.dtype(), DType::F32);
}

#[test]
fn test_generation_output_all_tokens_in_vocab() {
    let model = load_tiny_model();
    let vocab = model.config().padded_vocab_size;

    let output = model.generate_greedy(&[1, 2, 3], 10).unwrap();
    for &tok in &output.token_ids {
        assert!(
            tok < vocab,
            "generated token {tok} exceeds vocab size {vocab}"
        );
    }
}

// ---------------------------------------------------------------------------
// Multiple generation calls reuse model without interference
// ---------------------------------------------------------------------------

#[test]
fn test_multiple_independent_generations() {
    let model = load_tiny_model();

    // Two independent generation calls should not interfere
    let out_a = model.generate_greedy(&[1], 3).unwrap();
    let out_b = model.generate_greedy(&[2], 3).unwrap();

    // Both should succeed and produce 3 tokens
    assert_eq!(out_a.token_ids.len(), 3);
    assert_eq!(out_b.token_ids.len(), 3);

    // With zero weights, both should produce the same tokens
    assert_eq!(out_a.token_ids, out_b.token_ids);
}

#[test]
fn test_cache_grows_during_generation() {
    // Verify cache grows monotonically during generation steps.
    let model = load_tiny_model();
    let mut cache = model.new_cache();

    // Prefill 4 tokens
    let _ = model
        .forward_cached(&[0, 1, 2, 3], &[0, 1, 2, 3], Some(&mut cache))
        .unwrap();
    let after_prefill = cache.seq_len();
    assert_eq!(after_prefill, 4, "cache should have 4 tokens after prefill");

    // Decode 3 tokens one at a time
    for step in 0..3 {
        let pos = after_prefill + step;
        let _ = model
            .forward_cached(&[0], &[pos], Some(&mut cache))
            .unwrap();
        assert_eq!(
            cache.seq_len(),
            after_prefill + step + 1,
            "cache should grow by 1 per decode step"
        );
    }
    assert_eq!(
        cache.seq_len(),
        7,
        "total should be prefill(4) + decode(3) = 7"
    );
}

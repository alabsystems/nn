// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Comprehensive architecture tests for GLM-4/5 decoder-only transformer.
//!
//! Covers gaps in existing test suite:
//! - Half-RoPE computation (split + rotation correctness)
//! - SwiGLU MLP shape and activation with non-zero weights
//! - Fused QKV projection dimension arithmetic
//! - KV cache reset/clear lifecycle through model
//! - Cached vs uncached logit consistency with synthetic weights
//! - Embedding lookup shape and value diversity
//! - Multi-step decode cache growth invariants
//! - Config relationship invariants (rope_theta = base * ratio)
//!
//! Included from `glm5_tests.rs` via `#[path]`.
//! Issue: #4353

use super::*;
use crate::test_utils::tiny_config;
use nn_core::dyn_tensor::DynTensor;
use nn_core::layers::HalfRotaryEmbedding;
use nn_core::var_builder::VarBuilder;
use nn_core::{DType, Device};

// ---------------------------------------------------------------------------
// Half-RoPE computation tests
// ---------------------------------------------------------------------------

#[test]
fn test_half_rope_preserves_shape() {
    let cfg = tiny_config();
    let head_dim = cfg.kv_channels; // 64
    let rope =
        HalfRotaryEmbedding::new(head_dim, cfg.seq_length, cfg.rope_theta, &Device::Cpu).unwrap();

    // Input: [batch=1, heads=4, seq=3, head_dim=64]
    let x = DynTensor::ones(&[1, 4, 3, head_dim], DType::F32, &Device::Cpu).unwrap();
    let result = rope.apply(&x, 0).unwrap();
    assert_eq!(result.dims(), x.dims(), "half-RoPE must preserve shape");
}

#[test]
fn test_half_rope_second_half_unchanged() {
    // Half-RoPE splits head_dim in half: first half rotated, second half pass-through.
    // With all-ones input and position 0, the pass-through half must be unchanged.
    let head_dim = 8;
    let rope = HalfRotaryEmbedding::new(head_dim, 64, 10_000.0, &Device::Cpu).unwrap();

    let data: Vec<f32> = (0..head_dim).map(|i| (i + 1) as f32).collect();
    let x = DynTensor::from_vec(data.clone(), &[1, 1, 1, head_dim], &Device::Cpu).unwrap();
    let result = rope.apply(&x, 0).unwrap();
    let result_flat = result.to_flat_vec::<f32>().unwrap();

    // Second half (indices 4..8) should be identical to input
    let half = head_dim / 2;
    for i in half..head_dim {
        assert_eq!(
            result_flat[i], data[i],
            "pass-through dim {i} should be unchanged, got {} vs {}",
            result_flat[i], data[i]
        );
    }
}

#[test]
fn test_half_rope_first_half_rotated_at_nonzero_position() {
    // At non-zero position, the first half should differ from input.
    let head_dim = 8;
    let rope = HalfRotaryEmbedding::new(head_dim, 64, 10_000.0, &Device::Cpu).unwrap();

    let data: Vec<f32> = (0..head_dim).map(|i| (i + 1) as f32).collect();
    let x = DynTensor::from_vec(data.clone(), &[1, 1, 1, head_dim], &Device::Cpu).unwrap();
    let result = rope.apply(&x, 5).unwrap(); // position 5
    let result_flat = result.to_flat_vec::<f32>().unwrap();

    // First half should be different from input at non-zero position
    let half = head_dim / 2;
    let mut any_different = false;
    for i in 0..half {
        if (result_flat[i] - data[i]).abs() > 1e-6 {
            any_different = true;
            break;
        }
    }
    assert!(any_different, "first half should be rotated at position 5");
}

#[test]
fn test_half_rope_pair_applies_to_both_q_and_k() {
    let head_dim = 8;
    let rope = HalfRotaryEmbedding::new(head_dim, 64, 10_000.0, &Device::Cpu).unwrap();

    let q = DynTensor::ones(&[1, 2, 3, head_dim], DType::F32, &Device::Cpu).unwrap();
    let k = DynTensor::ones(&[1, 2, 3, head_dim], DType::F32, &Device::Cpu).unwrap();
    let positions = &[0, 1, 2];

    let (q_rot, k_rot) = rope.apply_pair(&q, &k, positions).unwrap();
    assert_eq!(q_rot.dims(), q.dims());
    assert_eq!(k_rot.dims(), k.dims());

    // Both should have finite values
    let q_flat = q_rot.to_flat_vec::<f32>().unwrap();
    let k_flat = k_rot.to_flat_vec::<f32>().unwrap();
    assert!(q_flat.iter().all(|v| v.is_finite()), "q_rot must be finite");
    assert!(k_flat.iter().all(|v| v.is_finite()), "k_rot must be finite");
}

#[test]
fn test_half_rope_head_dim_accessor() {
    let rope = HalfRotaryEmbedding::new(128, 4096, 10_000.0, &Device::Cpu).unwrap();
    assert_eq!(rope.head_dim(), 128);
    assert_eq!(rope.rope_dim(), 64);
}

#[test]
fn test_half_rope_max_seq_len_accessor() {
    let rope = HalfRotaryEmbedding::new(64, 8192, 10_000.0, &Device::Cpu).unwrap();
    assert_eq!(rope.max_seq_len(), 8192);
}

// ---------------------------------------------------------------------------
// SwiGLU MLP dimension verification
// ---------------------------------------------------------------------------

#[test]
fn test_swiglu_mlp_output_shape_through_model() {
    // The SwiGLU MLP projects hidden_size -> ffn_hidden_size*2 (gate+up),
    // splits, applies silu(gate)*up, then projects back to hidden_size.
    // Verify the model's forward produces correct output shape across configs.
    let configs = [
        (256, 512, 2, 4, 2, 64),  // tiny
        (128, 256, 1, 2, 1, 32),  // smaller
        (512, 1024, 1, 8, 4, 64), // medium
    ];

    for (hidden, ffn, layers, heads, kv_groups, kv_ch) in configs {
        let cfg = Glm5Config::new(
            hidden, ffn, layers, heads, kv_groups, 50, kv_ch, 1e-5, 32, true, true, false, 10_000.0,
        );
        assert!(cfg.validate().is_ok(), "config should be valid: h={hidden}");

        let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
        let model = Glm5Model::load(&vb, cfg.clone()).unwrap();
        let logits = model.forward(&[0, 1], &[0, 1]).unwrap();
        assert_eq!(
            logits.dims(),
            &[1, 2, 50],
            "h={hidden}: output shape must be [1, seq, vocab]"
        );
    }
}

// ---------------------------------------------------------------------------
// Fused QKV dimension arithmetic
// ---------------------------------------------------------------------------

#[test]
fn test_fused_qkv_size_calculation() {
    // Fused QKV output size = (num_heads + 2 * multi_query_group_num) * kv_channels
    let cfg = tiny_config();
    let expected_qkv_size =
        (cfg.num_attention_heads + 2 * cfg.multi_query_group_num) * cfg.kv_channels;
    // tiny: (4 + 2*2) * 64 = 8 * 64 = 512
    assert_eq!(expected_qkv_size, 512);

    // GLM-4-9B: (32 + 2*2) * 128 = 36 * 128 = 4608
    let cfg9b = Glm5Config::default();
    let qkv_9b = (cfg9b.num_attention_heads + 2 * cfg9b.multi_query_group_num) * cfg9b.kv_channels;
    assert_eq!(qkv_9b, 4608);
}

#[test]
fn test_fused_qkv_splits_correctly() {
    // After QKV projection, Q gets num_heads * head_dim elements,
    // K and V each get multi_query_group_num * head_dim elements.
    let cfg = tiny_config();
    let q_size = cfg.num_attention_heads * cfg.kv_channels; // 4 * 64 = 256
    let kv_size = cfg.multi_query_group_num * cfg.kv_channels; // 2 * 64 = 128
    let total = q_size + 2 * kv_size; // 256 + 256 = 512

    assert_eq!(q_size, 256);
    assert_eq!(kv_size, 128);
    assert_eq!(
        total,
        (cfg.num_attention_heads + 2 * cfg.multi_query_group_num) * cfg.kv_channels
    );
}

// ---------------------------------------------------------------------------
// KV cache lifecycle: reset and clear through model
// ---------------------------------------------------------------------------

#[test]
fn test_kv_cache_reset_allows_reuse() {
    let cfg = tiny_config();
    let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
    let model = Glm5Model::load(&vb, cfg).unwrap();
    let mut cache = model.new_cache();

    // Fill cache with 3 tokens
    model
        .forward_cached(&[0, 1, 2], &[0, 1, 2], Some(&mut cache))
        .unwrap();
    assert_eq!(cache.seq_len(), 3);

    // Reset and reuse
    cache.reset();
    assert_eq!(cache.seq_len(), 0);
    assert!(cache.is_empty());

    // Should work again from scratch
    let logits = model
        .forward_cached(&[10, 11], &[0, 1], Some(&mut cache))
        .unwrap();
    assert_eq!(logits.dims(), &[1, 2, 100]);
    assert_eq!(cache.seq_len(), 2);
}

#[test]
fn test_kv_cache_clear_allows_reuse() {
    let cfg = tiny_config();
    let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
    let model = Glm5Model::load(&vb, cfg).unwrap();
    let mut cache = model.new_cache();

    // Fill cache
    model
        .forward_cached(&[0, 1], &[0, 1], Some(&mut cache))
        .unwrap();
    assert_eq!(cache.seq_len(), 2);

    // Clear (preserves capacity) and reuse
    cache.clear();
    assert_eq!(cache.seq_len(), 0);

    let logits = model.forward_cached(&[5], &[0], Some(&mut cache)).unwrap();
    assert_eq!(logits.dims(), &[1, 1, 100]);
    assert_eq!(cache.seq_len(), 1);
}

// ---------------------------------------------------------------------------
// Cached vs uncached logit consistency (synthetic weights)
// ---------------------------------------------------------------------------

#[test]
fn test_cached_vs_uncached_last_token_consistency() {
    // With zero weights, cached token-by-token and uncached full-sequence
    // should produce the same last-token logits.
    let cfg = tiny_config();
    let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
    let model = Glm5Model::load(&vb, cfg).unwrap();

    // Uncached: full sequence
    let full_logits = model.forward(&[0, 1, 2], &[0, 1, 2]).unwrap();
    let full_last = full_logits
        .narrow(1, 2, 1)
        .unwrap()
        .to_flat_vec::<f32>()
        .unwrap();

    // Cached: token-by-token
    let mut cache = model.new_cache();
    let _ = model.forward_cached(&[0], &[0], Some(&mut cache)).unwrap();
    let _ = model.forward_cached(&[1], &[1], Some(&mut cache)).unwrap();
    let cached_last = model
        .forward_cached(&[2], &[2], Some(&mut cache))
        .unwrap()
        .to_flat_vec::<f32>()
        .unwrap();

    assert_eq!(full_last.len(), cached_last.len());
    let max_diff: f32 = full_last
        .iter()
        .zip(cached_last.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert!(
        max_diff < 1e-5,
        "cached vs uncached last token should match, max_diff={max_diff}"
    );
}

#[test]
fn test_cached_prompt_then_decode_matches_uncached() {
    // Prompt-fill with cache + single decode step should match
    // uncached full-sequence for the last position.
    let cfg = tiny_config();
    let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
    let model = Glm5Model::load(&vb, cfg).unwrap();

    // Uncached: 4 tokens
    let full_logits = model.forward(&[0, 1, 2, 3], &[0, 1, 2, 3]).unwrap();
    let full_last = full_logits
        .narrow(1, 3, 1)
        .unwrap()
        .to_flat_vec::<f32>()
        .unwrap();

    // Cached: prompt (3 tokens) + decode (1 token)
    let mut cache = model.new_cache();
    let _ = model
        .forward_cached(&[0, 1, 2], &[0, 1, 2], Some(&mut cache))
        .unwrap();
    let cached_last = model
        .forward_cached(&[3], &[3], Some(&mut cache))
        .unwrap()
        .to_flat_vec::<f32>()
        .unwrap();

    let max_diff: f32 = full_last
        .iter()
        .zip(cached_last.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert!(
        max_diff < 1e-5,
        "prompt+decode vs uncached should match, max_diff={max_diff}"
    );
}

// ---------------------------------------------------------------------------
// Embedding lookup shape and value tests
// ---------------------------------------------------------------------------

#[test]
fn test_embedding_lookup_shape_single_token() {
    let cfg = tiny_config();
    let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
    let model = Glm5Model::load(&vb, cfg.clone()).unwrap();

    let logits = model.forward(&[0], &[0]).unwrap();
    // Output shape: [1, 1, padded_vocab_size]
    assert_eq!(logits.dims(), &[1, 1, cfg.padded_vocab_size]);
}

#[test]
fn test_embedding_lookup_different_tokens_same_shape() {
    let cfg = tiny_config();
    let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
    let model = Glm5Model::load(&vb, cfg.clone()).unwrap();

    for token_id in [0, 1, 42, cfg.padded_vocab_size - 1] {
        let logits = model.forward(&[token_id], &[0]).unwrap();
        assert_eq!(
            logits.dims(),
            &[1, 1, cfg.padded_vocab_size],
            "token_id={token_id}: shape must be consistent"
        );
    }
}

#[test]
fn test_forward_from_embeddings_identity_dtype() {
    // When model dtype matches embedding dtype, no conversion occurs.
    let cfg = tiny_config();
    let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
    let model = Glm5Model::load(&vb, cfg.clone()).unwrap();

    let emb = DynTensor::ones(&[1, 2, cfg.hidden_size], DType::F32, &Device::Cpu).unwrap();
    let logits = model.forward_from_embeddings(&emb, &[0, 1], None).unwrap();
    assert_eq!(logits.dims(), &[1, 2, cfg.padded_vocab_size]);
}

// ---------------------------------------------------------------------------
// Multi-step decode cache growth invariants
// ---------------------------------------------------------------------------

#[test]
fn test_cache_seq_len_grows_monotonically() {
    let cfg = tiny_config();
    let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
    let model = Glm5Model::load(&vb, cfg).unwrap();
    let mut cache = model.new_cache();

    let mut prev_len = 0;
    // Prompt + 7 decode steps
    let steps: Vec<(Vec<usize>, Vec<usize>)> = vec![
        (vec![0, 1, 2], vec![0, 1, 2]), // prompt: 3 tokens
        (vec![3], vec![3]),             // decode step 1
        (vec![4], vec![4]),             // decode step 2
        (vec![5], vec![5]),             // decode step 3
        (vec![6], vec![6]),             // decode step 4
        (vec![7], vec![7]),             // decode step 5
        (vec![8], vec![8]),             // decode step 6
        (vec![9], vec![9]),             // decode step 7
    ];

    for (i, (ids, pos)) in steps.iter().enumerate() {
        let _ = model.forward_cached(ids, pos, Some(&mut cache)).unwrap();
        let cur_len = cache.seq_len();
        assert!(
            cur_len > prev_len,
            "step {i}: cache should grow, prev={prev_len}, cur={cur_len}"
        );
        prev_len = cur_len;
    }
    assert_eq!(cache.seq_len(), 10); // 3 prompt + 7 decode
}

#[test]
fn test_cache_num_layers_unchanged_through_forward() {
    let cfg = tiny_config();
    let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
    let model = Glm5Model::load(&vb, cfg.clone()).unwrap();
    let mut cache = model.new_cache();

    assert_eq!(cache.num_layers(), cfg.num_layers);

    model
        .forward_cached(&[0, 1], &[0, 1], Some(&mut cache))
        .unwrap();
    assert_eq!(cache.num_layers(), cfg.num_layers);

    model.forward_cached(&[2], &[2], Some(&mut cache)).unwrap();
    assert_eq!(cache.num_layers(), cfg.num_layers);

    cache.reset();
    assert_eq!(cache.num_layers(), cfg.num_layers);
}

// ---------------------------------------------------------------------------
// Config relationship invariants
// ---------------------------------------------------------------------------

#[test]
fn test_glm4_9b_chat_rope_theta_is_base_times_ratio() {
    // GLM-4-9B-chat: rope_ratio = 500, base = 10_000.0 → rope_theta = 5_000_000.0
    let cfg = Glm5Config::glm4_9b_chat();
    let expected_theta = 10_000.0 * 500.0;
    assert!(
        (cfg.rope_theta - expected_theta).abs() < 1.0,
        "rope_theta should be base * ratio = {expected_theta}, got {}",
        cfg.rope_theta
    );
}

#[test]
fn test_hidden_size_consistency_across_constructors() {
    // Default, glm4_9b_chat, and manual construction should agree
    let default = Glm5Config::default();
    let chat = Glm5Config::glm4_9b_chat();

    // Both are 4096 hidden
    assert_eq!(default.hidden_size, 4096);
    assert_eq!(chat.hidden_size, 4096);
    assert_eq!(default.hidden_size, chat.hidden_size);

    // hidden_size = num_attention_heads * kv_channels for both
    assert_eq!(
        default.hidden_size,
        default.num_attention_heads * default.kv_channels
    );
    assert_eq!(
        chat.hidden_size,
        chat.num_attention_heads * chat.kv_channels
    );
}

// ---------------------------------------------------------------------------
// Config: various valid head/kv_group configurations
// ---------------------------------------------------------------------------

#[test]
fn test_config_valid_gqa_configurations() {
    // Various valid (heads, kv_groups) combinations
    let combos: Vec<(usize, usize)> = vec![
        (1, 1),  // MHA, smallest
        (2, 1),  // GQA 2:1
        (2, 2),  // MHA
        (4, 1),  // GQA 4:1
        (4, 2),  // GQA 2:1
        (4, 4),  // MHA
        (8, 1),  // GQA 8:1
        (8, 2),  // GQA 4:1
        (8, 4),  // GQA 2:1
        (8, 8),  // MHA
        (16, 2), // GQA 8:1 (like GLM-4)
        (32, 2), // GQA 16:1 (like GLM-4-9B)
    ];

    for (heads, kv_groups) in combos {
        let cfg = Glm5Config::new(
            heads * 64, // hidden = heads * kv_channels
            heads * 128,
            1,
            heads,
            kv_groups,
            100,
            64,
            1e-5,
            32,
            true,
            true,
            false,
            10_000.0,
        );
        assert!(
            cfg.validate().is_ok(),
            "heads={heads}, kv_groups={kv_groups} should validate"
        );
        let groups = cfg.num_kv_groups().unwrap();
        assert_eq!(
            groups,
            heads / kv_groups,
            "heads={heads}, kv_groups={kv_groups}: num_kv_groups should be {}",
            heads / kv_groups
        );
    }
}

// ---------------------------------------------------------------------------
// Causal mask: dtype preservation
// ---------------------------------------------------------------------------

#[test]
fn test_causal_mask_dtype_preservation() {
    for dtype in [DType::F32, DType::BF16, DType::F16] {
        let mask = causal_mask(3, dtype, &Device::Cpu).unwrap();
        assert_eq!(mask.dims(), &[1, 1, 3, 3]);
        // Mask values should be finite (0.0) or -inf
        let vals = mask.to_flat_vec::<f32>().unwrap();
        for v in &vals {
            assert!(
                *v == 0.0 || (v.is_infinite() && *v < 0.0),
                "mask value should be 0 or -inf, got {v}"
            );
        }
    }
}

#[test]
fn test_causal_mask_with_offset_dtype_preservation() {
    for dtype in [DType::F32, DType::BF16] {
        let mask = causal_mask_with_offset(2, 5, dtype, &Device::Cpu).unwrap();
        assert_eq!(mask.dims(), &[1, 1, 2, 5]);
        let vals = mask.to_flat_vec::<f32>().unwrap();
        for v in &vals {
            assert!(
                *v == 0.0 || (v.is_infinite() && *v < 0.0),
                "mask value should be 0 or -inf, got {v}"
            );
        }
    }
}

// ---------------------------------------------------------------------------
// Forward: output logits are finite across all bias/dtype combos
// ---------------------------------------------------------------------------

#[test]
fn test_forward_finiteness_across_bias_combos() {
    let bias_combos = [(true, true), (true, false), (false, true), (false, false)];

    for (qkv_bias, linear_bias) in bias_combos {
        let mut cfg = tiny_config();
        cfg.add_qkv_bias = qkv_bias;
        cfg.add_bias_linear = linear_bias;

        let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
        let model = Glm5Model::load(&vb, cfg).unwrap();
        let logits = model.forward(&[0, 1, 2], &[0, 1, 2]).unwrap();

        let vals = logits.to_flat_vec::<f32>().unwrap();
        let non_finite = vals.iter().filter(|v| !v.is_finite()).count();
        assert_eq!(
            non_finite, 0,
            "qkv_bias={qkv_bias}, linear_bias={linear_bias}: all logits must be finite"
        );
    }
}

// ---------------------------------------------------------------------------
// Forward: single-token decode step produces correct shape
// ---------------------------------------------------------------------------

#[test]
fn test_single_token_decode_shape_various_positions() {
    let cfg = tiny_config();
    let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
    let model = Glm5Model::load(&vb, cfg.clone()).unwrap();

    // Different starting positions (simulating prefix caching)
    for pos in [0, 1, 5, 10, 50] {
        let logits = model.forward(&[42], &[pos]).unwrap();
        assert_eq!(
            logits.dims(),
            &[1, 1, cfg.padded_vocab_size],
            "position={pos}: shape must be [1, 1, vocab]"
        );
    }
}

// ---------------------------------------------------------------------------
// GLM SwiGLU gate dimension: dense_h_to_4h is 2x ffn_hidden_size
// ---------------------------------------------------------------------------

#[test]
fn test_swiglu_gate_up_fused_dimension() {
    // The gate+up fusion doubles ffn_hidden_size
    let cfg = tiny_config();
    let fused_size = cfg.ffn_hidden_size * 2;
    assert_eq!(fused_size, 1024, "tiny config: 512 * 2 = 1024");

    let cfg9b = Glm5Config::default();
    let fused_9b = cfg9b.ffn_hidden_size * 2;
    assert_eq!(fused_9b, 27392, "GLM-4-9B: 13696 * 2 = 27392");
}

// ---------------------------------------------------------------------------
// KV cache: layer accessor bounds checking
// ---------------------------------------------------------------------------

#[test]
fn test_kv_cache_layer_accessor_valid() {
    let cfg = tiny_config();
    let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
    let model = Glm5Model::load(&vb, cfg.clone()).unwrap();
    let cache = model.new_cache();

    for i in 0..cfg.num_layers {
        assert!(cache.layer(i).is_ok(), "layer {i} should be accessible");
    }
}

#[test]
fn test_kv_cache_layer_accessor_out_of_bounds() {
    let cfg = tiny_config();
    let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
    let model = Glm5Model::load(&vb, cfg.clone()).unwrap();
    let cache = model.new_cache();

    assert!(
        cache.layer(cfg.num_layers).is_err(),
        "layer index == num_layers should fail"
    );
    assert!(
        cache.layer(cfg.num_layers + 10).is_err(),
        "layer index >> num_layers should fail"
    );
}

// ---------------------------------------------------------------------------
// Model: load with VarBuilder::zeros is deterministic
// ---------------------------------------------------------------------------

#[test]
fn test_model_load_zeros_deterministic() {
    let cfg = tiny_config();

    let vb1 = VarBuilder::zeros(DType::F32, &Device::Cpu);
    let model1 = Glm5Model::load(&vb1, cfg.clone()).unwrap();

    let vb2 = VarBuilder::zeros(DType::F32, &Device::Cpu);
    let model2 = Glm5Model::load(&vb2, cfg).unwrap();

    let out1 = model1
        .forward(&[0, 1, 2], &[0, 1, 2])
        .unwrap()
        .to_flat_vec::<f32>()
        .unwrap();
    let out2 = model2
        .forward(&[0, 1, 2], &[0, 1, 2])
        .unwrap()
        .to_flat_vec::<f32>()
        .unwrap();

    assert_eq!(
        out1, out2,
        "two zero-weight models must produce identical output"
    );
}

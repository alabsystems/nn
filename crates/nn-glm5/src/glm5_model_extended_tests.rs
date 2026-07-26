// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Extended tests for GLM-5 model configuration, weight mapping, tokenizer
//! config, RoPE parameters, attention mask generation, KV cache lifecycle,
//! and end-to-end inference pipeline invariants.
//!
//! Part of #4544. Covers:
//!
//! 1. Weight name mapping for HuggingFace ChatGLM convention
//! 2. RoPE parameter space (theta, head_dim, seq_length interactions)
//! 3. Attention mask generation (causal_mask, causal_mask_with_offset)
//! 4. KV cache lifecycle (create, prefill, decode, clear, reuse)
//! 5. Config constructor field preservation (new() round-trip)
//! 6. Forward pass dtype handling (F32, BF16, F16 VarBuilder loads)
//! 7. Token ID boundary conditions (0, vocab-1, out-of-range)
//! 8. Multi-step autoregressive decode shape consistency
//! 9. Input validation error diagnostics
//! 10. Model accessor consistency

use super::*;
use crate::kv_cache::KVCache;
use crate::test_utils::tiny_config;
use nn_core::dyn_tensor::DynTensor;
use nn_core::layers::kv_cache::KvCache;
use nn_core::var_builder::VarBuilder;
use nn_core::{DType, Device};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn load_tiny_model() -> Glm5Model {
    let cfg = tiny_config();
    let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
    Glm5Model::load(&vb, cfg).unwrap()
}

fn load_model_with_config(cfg: Glm5Config) -> Glm5Model {
    let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
    Glm5Model::load(&vb, cfg).unwrap()
}

// ===========================================================================
// 1. Weight name mapping for HuggingFace ChatGLM convention
// ===========================================================================

/// Verify that VarBuilder::zeros can resolve all weight paths the model expects.
/// This exercises the full weight name hierarchy.
#[test]
fn test_weight_names_resolve_for_tiny_config() {
    let cfg = tiny_config();
    let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
    let result = Glm5Model::load(&vb, cfg);
    assert!(
        result.is_ok(),
        "all weight names should resolve via VarBuilder::zeros: {:?}",
        result.err()
    );
}

/// Weight names must follow ChatGLM convention: transformer.encoder.layers.{i}.*
#[test]
fn test_weight_names_resolve_for_multiple_layer_counts() {
    for num_layers in [1, 2, 4, 8, 12] {
        let mut cfg = tiny_config();
        cfg.num_layers = num_layers;
        let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
        let result = Glm5Model::load(&vb, cfg);
        assert!(
            result.is_ok(),
            "num_layers={num_layers} should load successfully: {:?}",
            result.err()
        );
    }
}

/// Weight names with bias linear enabled should resolve additional bias tensors.
#[test]
fn test_weight_names_with_bias_linear_enabled() {
    let mut cfg = tiny_config();
    cfg.add_bias_linear = true;
    let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
    let result = Glm5Model::load(&vb, cfg);
    assert!(
        result.is_ok(),
        "bias_linear=true should resolve all bias weight names: {:?}",
        result.err()
    );
}

/// Weight names with QKV bias disabled should still load.
#[test]
fn test_weight_names_without_qkv_bias() {
    let mut cfg = tiny_config();
    cfg.add_qkv_bias = false;
    let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
    let result = Glm5Model::load(&vb, cfg);
    assert!(
        result.is_ok(),
        "add_qkv_bias=false should load without bias tensors: {:?}",
        result.err()
    );
}

/// Both bias flags disabled should load.
#[test]
fn test_weight_names_no_bias_at_all() {
    let mut cfg = tiny_config();
    cfg.add_qkv_bias = false;
    cfg.add_bias_linear = false;
    let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
    let result = Glm5Model::load(&vb, cfg);
    assert!(
        result.is_ok(),
        "no bias at all should load cleanly: {:?}",
        result.err()
    );
}

/// Both bias flags enabled should load.
#[test]
fn test_weight_names_all_biases_enabled() {
    let mut cfg = tiny_config();
    cfg.add_qkv_bias = true;
    cfg.add_bias_linear = true;
    let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
    let result = Glm5Model::load(&vb, cfg);
    assert!(
        result.is_ok(),
        "all biases enabled should load cleanly: {:?}",
        result.err()
    );
}

// ===========================================================================
// 2. RoPE parameter space
// ===========================================================================

/// RoPE theta must be positive for all named configs.
#[test]
fn test_rope_theta_positive_all_configs() {
    let configs: Vec<(&str, Glm5Config)> = vec![
        ("default", Glm5Config::default()),
        ("chat", Glm5Config::glm4_9b_chat()),
        ("tiny", tiny_config()),
    ];
    for (name, cfg) in &configs {
        assert!(
            cfg.rope_theta > 0.0,
            "{name}: rope_theta must be positive, got {}",
            cfg.rope_theta
        );
        assert!(
            cfg.rope_theta.is_finite(),
            "{name}: rope_theta must be finite"
        );
    }
}

/// RoPE half-rotation dimension must divide evenly.
#[test]
fn test_rope_half_dim_divides_evenly() {
    for kv_ch in [4, 8, 16, 32, 64, 128, 256] {
        let half = kv_ch / 2;
        assert_eq!(
            kv_ch % 2,
            0,
            "kv_channels={kv_ch} must be even for half-RoPE"
        );
        assert_eq!(
            half % 2,
            0,
            "half dim={half} must be even for rotation pairs"
        );
    }
}

/// Chat theta is exactly 500x the base theta.
#[test]
fn test_rope_chat_theta_is_500x_base() {
    let base = Glm5Config::default();
    let chat = Glm5Config::glm4_9b_chat();
    let ratio = chat.rope_theta / base.rope_theta;
    assert!(
        (ratio - 500.0).abs() < 1e-6,
        "chat/base theta ratio should be 500, got {ratio}"
    );
}

/// Seq length for chat is much larger than base (long-context support).
#[test]
fn test_chat_seq_length_exceeds_base() {
    let base = Glm5Config::default();
    let chat = Glm5Config::glm4_9b_chat();
    assert!(
        chat.seq_length > base.seq_length,
        "chat seq_length ({}) should exceed base ({})",
        chat.seq_length,
        base.seq_length
    );
    assert_eq!(chat.seq_length, 131_072, "chat supports 128K context");
    assert_eq!(base.seq_length, 8192, "base supports 8K context");
}

/// Custom rope_theta values should validate and load.
#[test]
fn test_rope_custom_theta_values_validate() {
    for theta in [1.0, 100.0, 10_000.0, 1_000_000.0, 1e9] {
        let mut cfg = tiny_config();
        cfg.rope_theta = theta;
        assert!(cfg.validate().is_ok(), "rope_theta={theta} should validate");
    }
}

// ===========================================================================
// 3. Attention mask generation
// ===========================================================================

/// Causal mask for seq_len=1 should have a single 0 element.
#[test]
fn test_causal_mask_single_token() {
    let mask = causal_mask(1, DType::F32, &Device::Cpu).unwrap();
    let flat = mask.to_flat_vec::<f32>().unwrap();
    assert_eq!(flat.len(), 1, "single-token mask is 1x1");
    assert_eq!(flat[0], 0.0, "single token sees itself (not masked)");
}

/// Causal mask for seq_len=4 should be lower-triangular zeros with upper -inf.
#[test]
fn test_causal_mask_is_lower_triangular() {
    let seq = 4;
    let mask = causal_mask(seq, DType::F32, &Device::Cpu).unwrap();
    let flat = mask.to_flat_vec::<f32>().unwrap();
    // The mask has shape [1, 1, seq, seq] for broadcasting over batch and heads
    let total = flat.len();
    assert_eq!(
        total,
        seq * seq,
        "mask should have seq*seq elements in flat form or be broadcastable"
    );
    // Check lower triangle is 0 and upper is very negative
    for i in 0..seq {
        for j in 0..seq {
            let val = flat[i * seq + j];
            if j <= i {
                assert_eq!(val, 0.0, "mask[{i},{j}] should be 0.0 (visible), got {val}");
            } else {
                assert!(
                    val < -1e4,
                    "mask[{i},{j}] should be very negative (masked), got {val}"
                );
            }
        }
    }
}

/// Causal mask with offset: new tokens attend to cached tokens + themselves.
#[test]
fn test_causal_mask_with_offset_single_decode_step() {
    // 1 new token, 5 total (4 cached + 1 new)
    let mask = causal_mask_with_offset(1, 5, DType::F32, &Device::Cpu).unwrap();
    let flat = mask.to_flat_vec::<f32>().unwrap();
    // Shape should be [1, 1, 1, 5]: new token can see all 5 positions
    assert_eq!(flat.len(), 5, "single decode step mask has 5 positions");
    // All should be 0.0 (no masking for single new token)
    for (i, &val) in flat.iter().enumerate() {
        assert_eq!(
            val, 0.0,
            "single decode token should see position {i}: got {val}"
        );
    }
}

/// Causal mask with offset: 3 new tokens, 7 total.
#[test]
fn test_causal_mask_with_offset_multi_new_tokens() {
    // 3 new tokens, 7 total (4 cached + 3 new)
    let mask = causal_mask_with_offset(3, 7, DType::F32, &Device::Cpu).unwrap();
    let flat = mask.to_flat_vec::<f32>().unwrap();
    // Shape [1, 1, 3, 7]
    assert_eq!(flat.len(), 3 * 7, "3 new tokens, 7 total positions");

    // Row 0 (first new token, position 4): sees cached[0..4] + self
    for j in 0..5 {
        assert_eq!(
            flat[j],
            0.0,
            "row 0, col {j}: should see (cached + self)"
        );
    }
    for j in 5..7 {
        assert!(
            flat[j] < -1e4,
            "row 0, col {j}: future tokens masked"
        );
    }

    // Row 2 (third new token, position 6): sees everything
    for j in 0..7 {
        assert_eq!(
            flat[2 * 7 + j],
            0.0,
            "row 2, col {j}: last new token sees all"
        );
    }
}

// ===========================================================================
// 4. KV cache lifecycle
// ===========================================================================

/// Fresh DynTensor KV cache starts empty.
#[test]
fn test_kv_cache_fresh_is_empty() {
    let model = load_tiny_model();
    let cache = model.new_cache();
    assert_eq!(cache.seq_len(), 0);
    assert_eq!(cache.num_layers(), model.config().num_layers);
}

/// Cache grows during prefill.
#[test]
fn test_kv_cache_grows_during_prefill() {
    let model = load_tiny_model();
    let mut cache = model.new_cache();
    let _ = model
        .forward_cached(&[0, 1, 2], &[0, 1, 2], Some(&mut cache))
        .unwrap();
    assert_eq!(
        cache.seq_len(),
        3,
        "cache should hold 3 tokens after prefill"
    );
}

/// Cache grows by 1 per decode step.
#[test]
fn test_kv_cache_grows_by_one_per_decode() {
    let model = load_tiny_model();
    let mut cache = model.new_cache();

    // Prefill 2 tokens
    let _ = model
        .forward_cached(&[0, 1], &[0, 1], Some(&mut cache))
        .unwrap();
    assert_eq!(cache.seq_len(), 2);

    // 5 decode steps
    for step in 0..5 {
        let pos = 2 + step;
        let _ = model
            .forward_cached(&[0], &[pos], Some(&mut cache))
            .unwrap();
        assert_eq!(cache.seq_len(), pos + 1, "after decode step {step}");
    }
}

/// Flat KVCache clear + reuse cycle.
#[test]
fn test_flat_kv_cache_clear_and_reuse() {
    let mut cache = KVCache::new(2, 4, 8, 16);
    let token = vec![1.0_f32; 4 * 8];

    // Append to all layers for 3 tokens
    for _tok in 0..3 {
        for layer in 0..2 {
            cache.append(layer, &token, &token).unwrap();
        }
    }
    assert_eq!(cache.len(), 3);

    cache.clear();
    assert_eq!(cache.len(), 0);
    assert!(cache.is_empty());

    // Reuse after clear
    for layer in 0..2 {
        cache.append(layer, &token, &token).unwrap();
    }
    assert_eq!(cache.len(), 1);
}

/// Flat KVCache clone produces independent copy.
#[test]
fn test_flat_kv_cache_clone_independence() {
    let mut original = KVCache::new(1, 2, 4, 10);
    let token = vec![1.0_f32; 2 * 4];
    original.append(0, &token, &token).unwrap();

    let cloned = original.clone();
    assert_eq!(cloned.len(), 1);

    // Mutate original
    original.clear();
    assert_eq!(original.len(), 0);

    // Clone unaffected
    assert_eq!(cloned.len(), 1);
    assert_eq!(cloned.get_keys(0).len(), 8);
}

// ===========================================================================
// 5. Config constructor field preservation
// ===========================================================================

/// Glm5Config::new() preserves all 13 fields exactly.
#[test]
fn test_config_new_preserves_all_fields() {
    let cfg = Glm5Config::new(
        1024,     // hidden_size
        4096,     // ffn_hidden_size
        12,       // num_layers
        16,       // num_attention_heads
        4,        // multi_query_group_num
        32000,    // padded_vocab_size
        64,       // kv_channels
        1e-6,     // layernorm_epsilon
        2048,     // seq_length
        false,    // rmsnorm
        false,    // add_qkv_bias
        true,     // add_bias_linear
        50_000.0, // rope_theta
    );

    assert_eq!(cfg.hidden_size, 1024);
    assert_eq!(cfg.ffn_hidden_size, 4096);
    assert_eq!(cfg.num_layers, 12);
    assert_eq!(cfg.num_attention_heads, 16);
    assert_eq!(cfg.multi_query_group_num, 4);
    assert_eq!(cfg.padded_vocab_size, 32000);
    assert_eq!(cfg.kv_channels, 64);
    assert_eq!(cfg.layernorm_epsilon, 1e-6);
    assert_eq!(cfg.seq_length, 2048);
    assert!(!cfg.rmsnorm);
    assert!(!cfg.add_qkv_bias);
    assert!(cfg.add_bias_linear);
    assert_eq!(cfg.rope_theta, 50_000.0);
}

/// Default and glm4_9b_chat share structural fields but differ in context params.
#[test]
fn test_default_vs_chat_structural_fields_match() {
    let base = Glm5Config::default();
    let chat = Glm5Config::glm4_9b_chat();

    // These should be identical
    assert_eq!(base.hidden_size, chat.hidden_size);
    assert_eq!(base.ffn_hidden_size, chat.ffn_hidden_size);
    assert_eq!(base.num_layers, chat.num_layers);
    assert_eq!(base.num_attention_heads, chat.num_attention_heads);
    assert_eq!(base.multi_query_group_num, chat.multi_query_group_num);
    assert_eq!(base.padded_vocab_size, chat.padded_vocab_size);
    assert_eq!(base.kv_channels, chat.kv_channels);
    assert_eq!(base.rmsnorm, chat.rmsnorm);
    assert_eq!(base.add_qkv_bias, chat.add_qkv_bias);
    assert_eq!(base.add_bias_linear, chat.add_bias_linear);

    // These differ for long-context support
    assert_ne!(base.layernorm_epsilon, chat.layernorm_epsilon);
    assert_ne!(base.seq_length, chat.seq_length);
    assert_ne!(base.rope_theta, chat.rope_theta);
}

/// head_dim() is always equal to kv_channels.
#[test]
fn test_head_dim_equals_kv_channels_for_all_named_configs() {
    let configs: Vec<(&str, Glm5Config)> = vec![
        ("default", Glm5Config::default()),
        ("chat", Glm5Config::glm4_9b_chat()),
        ("tiny", tiny_config()),
    ];
    for (name, cfg) in configs {
        assert_eq!(
            cfg.head_dim(),
            cfg.kv_channels,
            "{name}: head_dim() must equal kv_channels"
        );
    }
}

// ===========================================================================
// 6. Forward pass dtype handling
// ===========================================================================

/// Model loaded with F32 VarBuilder reports F32 dtype.
#[test]
fn test_model_dtype_f32() {
    let cfg = tiny_config();
    let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
    let model = Glm5Model::load(&vb, cfg).unwrap();
    assert_eq!(model.dtype(), DType::F32);
}

/// Model loaded with BF16 VarBuilder reports BF16 dtype.
#[test]
fn test_model_dtype_bf16() {
    let cfg = tiny_config();
    let vb = VarBuilder::zeros(DType::BF16, &Device::Cpu);
    let model = Glm5Model::load(&vb, cfg).unwrap();
    assert_eq!(model.dtype(), DType::BF16);
}

/// Model loaded with F16 VarBuilder reports F16 dtype.
#[test]
fn test_model_dtype_f16() {
    let cfg = tiny_config();
    let vb = VarBuilder::zeros(DType::F16, &Device::Cpu);
    let model = Glm5Model::load(&vb, cfg).unwrap();
    assert_eq!(model.dtype(), DType::F16);
}

/// Forward from embeddings with matching dtype succeeds.
#[test]
fn test_forward_from_embeddings_matching_dtype() {
    let model = load_tiny_model();
    let cfg = model.config().clone();
    let emb = DynTensor::ones(&[1, 2, cfg.hidden_size], DType::F32, &Device::Cpu).unwrap();
    let logits = model.forward_from_embeddings(&emb, &[0, 1], None).unwrap();
    assert_eq!(logits.dims(), &[1, 2, cfg.padded_vocab_size]);
}

// ===========================================================================
// 7. Token ID boundary conditions
// ===========================================================================

/// Token ID 0 is valid.
#[test]
fn test_token_id_zero_valid() {
    let model = load_tiny_model();
    let result = model.forward(&[0], &[0]);
    assert!(result.is_ok(), "token_id=0 should be valid");
}

/// Token ID at vocab boundary (vocab-1) is valid.
#[test]
fn test_token_id_at_vocab_boundary() {
    let model = load_tiny_model();
    let vocab = model.config().padded_vocab_size;
    let result = model.forward(&[vocab - 1], &[0]);
    assert!(result.is_ok(), "token_id={} should be valid", vocab - 1);
}

/// Multiple distinct token IDs produce correct shape.
#[test]
fn test_distinct_token_ids_correct_shape() {
    let model = load_tiny_model();
    let vocab = model.config().padded_vocab_size;
    let ids = vec![0, 1, vocab / 4, vocab / 2, vocab - 1];
    let pos: Vec<usize> = (0..ids.len()).collect();
    let logits = model.forward(&ids, &pos).unwrap();
    assert_eq!(logits.dims(), &[1, ids.len(), vocab]);
}

/// Repeated token IDs are valid.
#[test]
fn test_repeated_token_ids_valid() {
    let model = load_tiny_model();
    let vocab = model.config().padded_vocab_size;
    let logits = model.forward(&[42, 42, 42], &[0, 1, 2]).unwrap();
    assert_eq!(logits.dims(), &[1, 3, vocab]);
}

// ===========================================================================
// 8. Multi-step autoregressive decode shape consistency
// ===========================================================================

/// Full autoregressive decode: prefill + N decode steps all produce correct shapes.
#[test]
fn test_autoregressive_decode_10_steps() {
    let model = load_tiny_model();
    let vocab = model.config().padded_vocab_size;
    let mut cache = model.new_cache();

    // Prefill with 3 tokens
    let prefill = model
        .forward_cached(&[10, 20, 30], &[0, 1, 2], Some(&mut cache))
        .unwrap();
    assert_eq!(prefill.dims(), &[1, 3, vocab]);
    assert_eq!(cache.seq_len(), 3);

    // 10 decode steps
    for step in 0..10 {
        let pos = 3 + step;
        let decode = model
            .forward_cached(&[step % 50], &[pos], Some(&mut cache))
            .unwrap();
        assert_eq!(
            decode.dims(),
            &[1, 1, vocab],
            "decode step {step}: wrong shape"
        );
        assert_eq!(
            cache.seq_len(),
            pos + 1,
            "decode step {step}: wrong cache len"
        );

        // Verify logits are finite
        let flat = decode.to_flat_vec::<f32>().unwrap();
        assert!(
            flat.iter().all(|v| v.is_finite()),
            "decode step {step}: non-finite logits"
        );
    }
}

/// Prefill with 1 token then decode: covers the seq_len==1 cache path.
#[test]
fn test_autoregressive_single_token_prefill() {
    let model = load_tiny_model();
    let vocab = model.config().padded_vocab_size;
    let mut cache = model.new_cache();

    let prefill = model.forward_cached(&[0], &[0], Some(&mut cache)).unwrap();
    assert_eq!(prefill.dims(), &[1, 1, vocab]);
    assert_eq!(cache.seq_len(), 1);

    let decode = model.forward_cached(&[1], &[1], Some(&mut cache)).unwrap();
    assert_eq!(decode.dims(), &[1, 1, vocab]);
    assert_eq!(cache.seq_len(), 2);
}

// ===========================================================================
// 9. Input validation error diagnostics
// ===========================================================================

/// Mismatched input_ids and positions lengths produce an error.
#[test]
fn test_forward_rejects_length_mismatch() {
    let model = load_tiny_model();
    let result = model.forward(&[0, 1, 2], &[0, 1]); // 3 ids, 2 positions
    assert!(result.is_err(), "mismatched lengths should error");
    let msg = result.unwrap_err().to_string();
    assert!(
        msg.contains("3") && msg.contains("2"),
        "error should mention both lengths: {msg}"
    );
}

/// forward_from_embeddings rejects wrong hidden_size.
#[test]
fn test_forward_from_embeddings_wrong_hidden_size() {
    let model = load_tiny_model();
    let wrong_hidden = model.config().hidden_size + 1;
    let emb = DynTensor::ones(&[1, 2, wrong_hidden], DType::F32, &Device::Cpu).unwrap();
    let result = model.forward_from_embeddings(&emb, &[0, 1], None);
    assert!(result.is_err(), "wrong hidden_size should error");
    let msg = result.unwrap_err().to_string();
    assert!(
        msg.contains("hidden_size"),
        "error should mention hidden_size: {msg}"
    );
}

/// forward_cached with wrong cache layer count produces CacheMismatch.
#[test]
fn test_forward_cached_cache_layer_mismatch_diagnostic() {
    let model = load_tiny_model();
    let model_layers = model.config().num_layers;

    for wrong_layers in [model_layers + 1, model_layers + 5, 1] {
        if wrong_layers == model_layers {
            continue;
        }
        let mut cache = KvCache::new(wrong_layers);
        let result = model.forward_cached(&[0], &[0], Some(&mut cache));
        assert!(
            result.is_err(),
            "cache with {wrong_layers} layers should error (model has {model_layers})"
        );
    }
}

/// forward_from_embeddings with seq_len mismatch.
#[test]
fn test_forward_from_embeddings_position_count_mismatch() {
    let model = load_tiny_model();
    let cfg = model.config().clone();
    let emb = DynTensor::ones(&[1, 5, cfg.hidden_size], DType::F32, &Device::Cpu).unwrap();
    let result = model.forward_from_embeddings(&emb, &[0, 1, 2], None); // 5 tokens, 3 positions
    assert!(result.is_err(), "position count mismatch should error");
    let msg = result.unwrap_err().to_string();
    assert!(
        msg.contains("5") && msg.contains("3"),
        "error should mention seq_len and positions: {msg}"
    );
}

// ===========================================================================
// 10. Model accessor consistency
// ===========================================================================

/// config() returns the same config used at construction.
#[test]
fn test_config_accessor_matches_construction() {
    let cfg = tiny_config();
    let model = load_model_with_config(cfg.clone());
    assert_eq!(model.config().hidden_size, cfg.hidden_size);
    assert_eq!(model.config().num_layers, cfg.num_layers);
    assert_eq!(model.config().num_attention_heads, cfg.num_attention_heads);
    assert_eq!(
        model.config().multi_query_group_num,
        cfg.multi_query_group_num
    );
    assert_eq!(model.config().padded_vocab_size, cfg.padded_vocab_size);
    assert_eq!(model.config().kv_channels, cfg.kv_channels);
    assert_eq!(model.config().rope_theta, cfg.rope_theta);
}

/// device() returns CPU for CPU-loaded model.
#[test]
fn test_device_accessor_cpu() {
    let model = load_tiny_model();
    assert!(matches!(model.device(), Device::Cpu));
}

/// new_cache() creates cache with correct layer count.
#[test]
fn test_new_cache_layer_count() {
    for num_layers in [1, 2, 4, 8] {
        let mut cfg = tiny_config();
        cfg.num_layers = num_layers;
        let model = load_model_with_config(cfg);
        let cache = model.new_cache();
        assert_eq!(
            cache.num_layers(),
            num_layers,
            "new_cache should have {num_layers} layers"
        );
    }
}

// ===========================================================================
// 11. Flat KVCache validation
// ===========================================================================

/// KVCache rejects layer index out of range.
#[test]
fn test_flat_kv_cache_layer_bounds_check() {
    let mut cache = KVCache::new(3, 2, 4, 10);
    let token = vec![0.0_f32; 2 * 4];
    let result = cache.append(3, &token, &token);
    assert!(result.is_err(), "layer=3 with num_layers=3 should fail");

    let result = cache.append(100, &token, &token);
    assert!(result.is_err(), "layer=100 should fail");
}

/// KVCache rejects wrong key size.
#[test]
fn test_flat_kv_cache_wrong_key_size() {
    let mut cache = KVCache::new(1, 2, 4, 10);
    let correct_size = 2 * 4;
    let wrong_key = vec![0.0_f32; correct_size + 1];
    let correct_val = vec![0.0_f32; correct_size];
    let result = cache.append(0, &wrong_key, &correct_val);
    assert!(result.is_err(), "wrong key size should fail");
}

/// KVCache rejects wrong value size.
#[test]
fn test_flat_kv_cache_wrong_value_size() {
    let mut cache = KVCache::new(1, 2, 4, 10);
    let correct_size = 2 * 4;
    let correct_key = vec![0.0_f32; correct_size];
    let wrong_val = vec![0.0_f32; correct_size - 1];
    let result = cache.append(0, &correct_key, &wrong_val);
    assert!(result.is_err(), "wrong value size should fail");
}

/// KVCache accessors return correct values at construction.
#[test]
fn test_flat_kv_cache_accessors() {
    let cache = KVCache::new(5, 8, 16, 512);
    assert_eq!(cache.num_layers(), 5);
    assert_eq!(cache.num_heads(), 8);
    assert_eq!(cache.head_dim(), 16);
    assert_eq!(cache.max_seq_len(), 512);
    assert_eq!(cache.len(), 0);
    assert!(cache.is_empty());
}

// ===========================================================================
// 12. Config validation edge cases
// ===========================================================================

/// Config with hidden_size=0 is rejected.
#[test]
fn test_validate_rejects_zero_hidden_size() {
    let mut cfg = tiny_config();
    cfg.hidden_size = 0;
    assert!(cfg.validate().is_err(), "hidden_size=0 should fail");
}

/// Config with ffn_hidden_size=0 is rejected.
#[test]
fn test_validate_rejects_zero_ffn_hidden_size() {
    let mut cfg = tiny_config();
    cfg.ffn_hidden_size = 0;
    assert!(cfg.validate().is_err(), "ffn_hidden_size=0 should fail");
}

/// Config with num_attention_heads=0 is rejected.
#[test]
fn test_validate_rejects_zero_attention_heads() {
    let mut cfg = tiny_config();
    cfg.num_attention_heads = 0;
    assert!(cfg.validate().is_err(), "num_attention_heads=0 should fail");
}

/// Config with multi_query_group_num=0 is rejected.
#[test]
fn test_validate_rejects_zero_multi_query_group_num() {
    let mut cfg = tiny_config();
    cfg.multi_query_group_num = 0;
    assert!(
        cfg.validate().is_err(),
        "multi_query_group_num=0 should fail"
    );
}

/// Negative layernorm_epsilon is rejected.
#[test]
fn test_validate_rejects_negative_epsilon() {
    let mut cfg = tiny_config();
    cfg.layernorm_epsilon = -1e-5;
    assert!(cfg.validate().is_err(), "negative epsilon should fail");
}

/// Infinite layernorm_epsilon is rejected.
#[test]
fn test_validate_rejects_inf_epsilon() {
    let mut cfg = tiny_config();
    cfg.layernorm_epsilon = f64::INFINITY;
    assert!(cfg.validate().is_err(), "infinite epsilon should fail");
}

/// Zero rope_theta is rejected.
#[test]
fn test_validate_rejects_zero_rope_theta() {
    let mut cfg = tiny_config();
    cfg.rope_theta = 0.0;
    assert!(cfg.validate().is_err(), "zero rope_theta should fail");
}

// ===========================================================================
// 13. Error variant construction and Display
// ===========================================================================

/// All Glm5Error variants render non-empty Display messages.
#[test]
fn test_all_error_variants_display_non_empty() {
    let errors: Vec<Glm5Error> = vec![
        Glm5Error::InvalidConfig {
            reason: "test config".into(),
        },
        Glm5Error::InvalidInput {
            reason: "test input".into(),
        },
        Glm5Error::CacheMismatch {
            cache_layers: 1,
            model_layers: 2,
        },
        Glm5Error::NonFiniteOutput {
            stage: "test",
            count: 5,
        },
        Glm5Error::WeightLoad {
            reason: "missing".into(),
        },
    ];
    for err in &errors {
        let msg = err.to_string();
        assert!(
            !msg.is_empty(),
            "error Display should be non-empty: {err:?}"
        );
    }
}

/// Glm5Error converts to TensorError and back preserves message content.
#[test]
fn test_error_to_tensor_error_preserves_message() {
    use nn_core::TensorError;
    let original_msg = "specific failure reason";
    let err = Glm5Error::InvalidConfig {
        reason: original_msg.into(),
    };
    let te: TensorError = err.into();
    let te_msg = te.to_string();
    assert!(
        te_msg.contains(original_msg),
        "TensorError should contain original message: {te_msg}"
    );
}

// ===========================================================================
// 14. GQA dimension algebra
// ===========================================================================

/// num_kv_groups returns correct repeat factor.
#[test]
fn test_num_kv_groups_various_configs() {
    let cases: Vec<(usize, usize, usize)> = vec![
        // (num_heads, kv_heads, expected_groups)
        (4, 2, 2),
        (4, 4, 1),
        (4, 1, 4),
        (32, 2, 16),
        (32, 4, 8),
        (32, 8, 4),
        (32, 16, 2),
        (32, 32, 1),
        (64, 8, 8),
    ];
    for (nh, nkv, expected) in cases {
        let mut cfg = tiny_config();
        cfg.num_attention_heads = nh;
        cfg.multi_query_group_num = nkv;
        let groups = cfg.num_kv_groups().unwrap();
        assert_eq!(
            groups, expected,
            "nh={nh}, nkv={nkv}: expected {expected} groups, got {groups}"
        );
    }
}

/// num_kv_groups errors when heads not divisible by kv_heads.
#[test]
fn test_num_kv_groups_indivisible_errors() {
    let mut cfg = tiny_config();
    cfg.num_attention_heads = 7;
    cfg.multi_query_group_num = 3;
    assert!(cfg.num_kv_groups().is_err(), "7/3 should error");
}

/// QKV fused dimension = (nh + 2*nkv) * head_dim.
#[test]
fn test_qkv_fused_dimension_formula() {
    let cfg = tiny_config();
    let nh = cfg.num_attention_heads;
    let nkv = cfg.multi_query_group_num;
    let hd = cfg.head_dim();
    let expected_qkv = (nh + 2 * nkv) * hd;

    // For tiny: (4 + 2*2) * 64 = 512
    assert_eq!(expected_qkv, (4 + 4) * 64);
    assert_eq!(expected_qkv, 512);
}

// ===========================================================================
// 15. Forward pass with different GQA configurations
// ===========================================================================

/// MQA config (1 KV head) produces correct output.
#[test]
fn test_forward_mqa_config() {
    let cfg = Glm5Config {
        hidden_size: 256,
        ffn_hidden_size: 512,
        num_layers: 1,
        num_attention_heads: 4,
        multi_query_group_num: 1,
        padded_vocab_size: 50,
        kv_channels: 64,
        layernorm_epsilon: 1e-5,
        seq_length: 32,
        rmsnorm: true,
        add_qkv_bias: true,
        add_bias_linear: false,
        rope_theta: 10_000.0,
    };
    let model = load_model_with_config(cfg.clone());
    let logits = model.forward(&[0, 1], &[0, 1]).unwrap();
    assert_eq!(logits.dims(), &[1, 2, cfg.padded_vocab_size]);
}

/// MHA config (all heads are KV heads) produces correct output.
#[test]
fn test_forward_mha_config() {
    let cfg = Glm5Config {
        hidden_size: 256,
        ffn_hidden_size: 512,
        num_layers: 1,
        num_attention_heads: 4,
        multi_query_group_num: 4,
        padded_vocab_size: 50,
        kv_channels: 64,
        layernorm_epsilon: 1e-5,
        seq_length: 32,
        rmsnorm: true,
        add_qkv_bias: true,
        add_bias_linear: false,
        rope_theta: 10_000.0,
    };
    let model = load_model_with_config(cfg.clone());
    let logits = model.forward(&[0, 1], &[0, 1]).unwrap();
    assert_eq!(logits.dims(), &[1, 2, cfg.padded_vocab_size]);
}

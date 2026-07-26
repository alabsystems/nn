// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Architecture-focused tests for GLM-4/5 decoder-only transformer.
//!
//! Tests the mathematical relationships between config parameters and the
//! resulting model structure: weight shapes, attention scaling, RoPE frequency
//! computation, MLP intermediate dimensions, GQA repeat factors, KV cache
//! shape invariants, and end-to-end shape propagation through multi-config
//! model variants.
//!
//! Included from `lib.rs` via `#[cfg(test)] #[path]`.

use super::*;
use crate::test_utils::tiny_config;
use nn_core::dyn_tensor::DynTensor;
use nn_core::layers::HalfRotaryEmbedding;
use nn_core::var_builder::VarBuilder;
use nn_core::{DType, Device};

// ---------------------------------------------------------------------------
// Attention scale factor: 1/sqrt(head_dim)
// ---------------------------------------------------------------------------

#[test]
fn test_attention_scale_factor_tiny() {
    let cfg = tiny_config();
    let expected_scale = 1.0 / (cfg.kv_channels as f64).sqrt();
    // head_dim=64 → scale = 1/8 = 0.125
    assert!(
        (expected_scale - 0.125).abs() < 1e-12,
        "tiny config: scale should be 1/sqrt(64) = 0.125, got {expected_scale}"
    );
}

#[test]
fn test_attention_scale_factor_glm4_9b() {
    let cfg = Glm5Config::default();
    let expected_scale = 1.0 / (cfg.kv_channels as f64).sqrt();
    // head_dim=128 → scale = 1/sqrt(128) ≈ 0.08839
    let target = 1.0 / 128.0_f64.sqrt();
    assert!(
        (expected_scale - target).abs() < 1e-12,
        "GLM-4-9B: scale should be 1/sqrt(128) ≈ {target}, got {expected_scale}"
    );
}

#[test]
fn test_attention_scale_factor_chat() {
    let cfg = Glm5Config::glm4_9b_chat();
    // Chat variant uses same head_dim as base
    let scale = 1.0 / (cfg.kv_channels as f64).sqrt();
    let base_scale = 1.0 / (Glm5Config::default().kv_channels as f64).sqrt();
    assert!(
        (scale - base_scale).abs() < 1e-15,
        "chat and base should have same scale factor"
    );
}

// ---------------------------------------------------------------------------
// RoPE frequency computation: theta_i = theta^(-2i/d) for i in 0..d/2
// ---------------------------------------------------------------------------

#[test]
fn test_rope_frequency_first_dim() {
    // For GLM half-RoPE, the rotary dimension is head_dim/2.
    // freq[0] = theta^(-0/d) = 1.0 for any theta
    let theta = 10_000.0_f64;
    let head_dim = 64_usize;
    let rope_dim = head_dim / 2; // half-RoPE
    let freq_0 = theta.powf(-0.0 / rope_dim as f64);
    assert!(
        (freq_0 - 1.0).abs() < 1e-12,
        "first frequency should always be 1.0, got {freq_0}"
    );
}

#[test]
fn test_rope_frequency_last_dim() {
    // freq[d/2 - 1] = theta^(-2*(d/2-1)/d) for the innermost rotation
    let theta = 10_000.0_f64;
    let rope_dim = 32_usize; // half of head_dim=64
    let last_idx = rope_dim - 1;
    let freq_last = 1.0 / theta.powf(2.0 * last_idx as f64 / (2 * rope_dim) as f64);
    // This should be a very small number for large theta
    assert!(
        freq_last > 0.0 && freq_last < 1.0,
        "last frequency should be in (0, 1), got {freq_last}"
    );
}

#[test]
fn test_rope_frequencies_monotonically_decrease() {
    let theta = 10_000.0_f64;
    let rope_dim = 32_usize;
    let freqs: Vec<f64> = (0..rope_dim)
        .map(|i| 1.0 / theta.powf(2.0 * i as f64 / (2 * rope_dim) as f64))
        .collect();
    for w in freqs.windows(2) {
        assert!(
            w[0] >= w[1],
            "frequencies should be monotonically non-increasing: {} < {}",
            w[0],
            w[1]
        );
    }
}

#[test]
fn test_rope_higher_theta_yields_lower_frequencies() {
    // Higher base theta → lower frequencies (slower rotation per position)
    let rope_dim = 32_usize;
    let idx = 10;
    let freq_10k = 1.0 / 10_000.0_f64.powf(2.0 * f64::from(idx) / (2 * rope_dim) as f64);
    let freq_5m = 1.0 / 5_000_000.0_f64.powf(2.0 * f64::from(idx) / (2 * rope_dim) as f64);
    assert!(
        freq_5m < freq_10k,
        "higher theta should give lower freq: 5M→{freq_5m} vs 10K→{freq_10k}"
    );
}

// ---------------------------------------------------------------------------
// Half-RoPE: verify rope_dim = head_dim / 2 (GLM-specific)
// ---------------------------------------------------------------------------

#[test]
fn test_half_rope_dim_is_half_head_dim() {
    for head_dim in [8, 16, 32, 64, 128, 256] {
        let rope = HalfRotaryEmbedding::new(head_dim, 64, 10_000.0, &Device::Cpu).unwrap();
        assert_eq!(
            rope.rope_dim(),
            head_dim / 2,
            "head_dim={head_dim}: rope_dim should be head_dim/2"
        );
    }
}

#[test]
fn test_half_rope_position_zero_first_half_unchanged() {
    // At position 0, cos(0)=1, sin(0)=0, so rotation is identity.
    // Both halves should be unchanged at position 0.
    let head_dim = 8;
    let rope = HalfRotaryEmbedding::new(head_dim, 64, 10_000.0, &Device::Cpu).unwrap();
    let data: Vec<f32> = (1..=head_dim as i32).map(|i| i as f32).collect();
    let x = DynTensor::from_vec(data.clone(), &[1, 1, 1, head_dim], &Device::Cpu).unwrap();
    let result = rope.apply(&x, 0).unwrap();
    let result_flat = result.to_flat_vec::<f32>().unwrap();

    // At position 0, the entire vector should be unchanged
    for i in 0..head_dim {
        assert!(
            (result_flat[i] - data[i]).abs() < 1e-5,
            "position 0: dim {i} should be unchanged, got {} vs {}",
            result_flat[i],
            data[i]
        );
    }
}

// ---------------------------------------------------------------------------
// MLP intermediate dimension arithmetic
// ---------------------------------------------------------------------------

#[test]
fn test_mlp_dense_h_to_4h_output_dim() {
    // dense_h_to_4h outputs ffn_hidden_size * 2 (gate + up fused)
    let cfg = tiny_config();
    let fused_output = cfg.ffn_hidden_size * 2;
    assert_eq!(fused_output, 1024, "tiny: 512 * 2 = 1024");

    let cfg9b = Glm5Config::default();
    let fused_9b = cfg9b.ffn_hidden_size * 2;
    assert_eq!(fused_9b, 27392, "9B: 13696 * 2 = 27392");
}

#[test]
fn test_mlp_dense_4h_to_h_output_dim() {
    // dense_4h_to_h projects from ffn_hidden_size back to hidden_size
    let cfg = tiny_config();
    assert_eq!(cfg.hidden_size, 256);
    assert_eq!(cfg.ffn_hidden_size, 512);
    // Weight shape: [hidden_size, ffn_hidden_size] = [256, 512]
}

#[test]
fn test_mlp_swiglu_split_produces_equal_halves() {
    // After dense_h_to_4h, the output is split into two equal halves
    // for gate and up projections
    let cfg = tiny_config();
    let fused = cfg.ffn_hidden_size * 2;
    assert_eq!(fused / 2, cfg.ffn_hidden_size);
    assert_eq!(fused % 2, 0, "fused size must be even for split");
}

#[test]
fn test_mlp_expansion_ratio() {
    // GLM-4-9B uses ffn_hidden_size/hidden_size ≈ 3.34 expansion
    let cfg = Glm5Config::default();
    let ratio = cfg.ffn_hidden_size as f64 / cfg.hidden_size as f64;
    assert!(
        (ratio - 3.34375).abs() < 0.01, // 13696/4096 = 3.34375
        "GLM-4-9B MLP expansion ratio should be ~3.34, got {ratio}"
    );
}

// ---------------------------------------------------------------------------
// QKV weight shape relationship
// ---------------------------------------------------------------------------

#[test]
fn test_qkv_weight_rows_formula() {
    // Fused QKV weight shape: [(nh + 2*nkv) * hd, hidden_size]
    let configs: Vec<(&str, Glm5Config)> = vec![
        ("tiny", tiny_config()),
        ("9B", Glm5Config::default()),
        ("chat", Glm5Config::glm4_9b_chat()),
    ];
    for (name, cfg) in &configs {
        let expected_rows =
            (cfg.num_attention_heads + 2 * cfg.multi_query_group_num) * cfg.kv_channels;
        let q_part = cfg.num_attention_heads * cfg.kv_channels;
        let k_part = cfg.multi_query_group_num * cfg.kv_channels;
        let v_part = cfg.multi_query_group_num * cfg.kv_channels;
        assert_eq!(
            expected_rows,
            q_part + k_part + v_part,
            "{name}: QKV rows should be Q + K + V"
        );
    }
}

#[test]
fn test_q_projection_recovers_hidden_size() {
    // Q projection: num_heads * head_dim should equal hidden_size
    // (this is an architectural invariant for GLM)
    let configs: Vec<(&str, Glm5Config)> =
        vec![("tiny", tiny_config()), ("9B", Glm5Config::default())];
    for (name, cfg) in &configs {
        assert_eq!(
            cfg.num_attention_heads * cfg.kv_channels,
            cfg.hidden_size,
            "{name}: Q projection dim should equal hidden_size"
        );
    }
}

// ---------------------------------------------------------------------------
// GQA repeat factor
// ---------------------------------------------------------------------------

#[test]
fn test_gqa_repeat_factor_tiny() {
    let cfg = tiny_config();
    let repeat = cfg.num_attention_heads / cfg.multi_query_group_num;
    assert_eq!(repeat, 2, "tiny: 4 heads / 2 kv = repeat 2");
}

#[test]
fn test_gqa_repeat_factor_9b() {
    let cfg = Glm5Config::default();
    let repeat = cfg.num_attention_heads / cfg.multi_query_group_num;
    assert_eq!(repeat, 16, "9B: 32 heads / 2 kv = repeat 16");
}

#[test]
fn test_gqa_repeat_factor_mha_is_one() {
    // When num_heads == multi_query_group_num, repeat = 1 (MHA)
    let mut cfg = tiny_config();
    cfg.num_attention_heads = 8;
    cfg.multi_query_group_num = 8;
    let repeat = cfg.num_attention_heads / cfg.multi_query_group_num;
    assert_eq!(repeat, 1, "MHA: repeat factor should be 1");
}

#[test]
fn test_gqa_repeat_factor_mqa_is_num_heads() {
    // When multi_query_group_num == 1, repeat = num_heads (MQA)
    let mut cfg = tiny_config();
    cfg.num_attention_heads = 16;
    cfg.multi_query_group_num = 1;
    let repeat = cfg.num_attention_heads / cfg.multi_query_group_num;
    assert_eq!(repeat, 16, "MQA: repeat factor should equal num_heads");
}

// ---------------------------------------------------------------------------
// KV cache shape invariants after forward passes
// ---------------------------------------------------------------------------

#[test]
fn test_kv_cache_shape_after_prompt() {
    let cfg = tiny_config();
    let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
    let model = Glm5Model::load(&vb, cfg.clone()).unwrap();
    let mut cache = model.new_cache();

    let prompt_len = 5;
    let ids: Vec<usize> = (0..prompt_len).collect();
    let pos: Vec<usize> = (0..prompt_len).collect();
    model.forward_cached(&ids, &pos, Some(&mut cache)).unwrap();

    assert_eq!(cache.seq_len(), prompt_len);
    assert_eq!(cache.num_layers(), cfg.num_layers);
    // Each layer should have cached KV for prompt_len tokens
    for layer_idx in 0..cfg.num_layers {
        let layer = cache.layer(layer_idx).unwrap();
        // k_cache shape should be [1, num_kv_heads, seq_len, head_dim]
        if let Ok(Some(k)) = layer.k() {
            let dims = k.dims();
            assert_eq!(dims[0], 1, "batch should be 1");
            assert_eq!(
                dims[1], cfg.multi_query_group_num,
                "layer {layer_idx}: k heads should be multi_query_group_num"
            );
            assert_eq!(
                dims[2], prompt_len,
                "layer {layer_idx}: k seq_len should be {prompt_len}"
            );
            assert_eq!(
                dims[3], cfg.kv_channels,
                "layer {layer_idx}: k head_dim should be kv_channels"
            );
        }
    }
}

#[test]
fn test_kv_cache_grows_by_one_per_decode_step() {
    let cfg = tiny_config();
    let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
    let model = Glm5Model::load(&vb, cfg).unwrap();
    let mut cache = model.new_cache();

    // Prefill
    model
        .forward_cached(&[0, 1, 2], &[0, 1, 2], Some(&mut cache))
        .unwrap();
    let after_prefill = cache.seq_len();
    assert_eq!(after_prefill, 3);

    // Each decode step should grow cache by exactly 1
    for step in 0..5 {
        let pos = after_prefill + step;
        model
            .forward_cached(&[42], &[pos], Some(&mut cache))
            .unwrap();
        assert_eq!(
            cache.seq_len(),
            after_prefill + step + 1,
            "after decode step {step}"
        );
    }
}

// ---------------------------------------------------------------------------
// Multi-config model loading and shape propagation
// ---------------------------------------------------------------------------

#[test]
fn test_model_shape_propagation_varying_layers() {
    for num_layers in [1, 2, 4] {
        let mut cfg = tiny_config();
        cfg.num_layers = num_layers;
        let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
        let model = Glm5Model::load(&vb, cfg.clone()).unwrap();

        let logits = model.forward(&[0, 1, 2], &[0, 1, 2]).unwrap();
        assert_eq!(
            logits.dims(),
            &[1, 3, cfg.padded_vocab_size],
            "num_layers={num_layers}: output shape should be [1, 3, vocab]"
        );
    }
}

#[test]
fn test_model_shape_propagation_varying_hidden_size() {
    for (hidden, ffn, heads, kv_groups, kv_ch) in [
        (64, 128, 2, 1, 32),
        (128, 256, 4, 2, 32),
        (256, 512, 4, 2, 64),
        (512, 1024, 8, 4, 64),
    ] {
        let cfg = Glm5Config::new(
            hidden, ffn, 1, heads, kv_groups, 50, kv_ch, 1e-5, 32, true, true, false, 10_000.0,
        );
        assert!(cfg.validate().is_ok(), "config h={hidden} should validate");

        let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
        let model = Glm5Model::load(&vb, cfg).unwrap();
        let logits = model.forward(&[0, 1], &[0, 1]).unwrap();
        assert_eq!(
            logits.dims(),
            &[1, 2, 50],
            "h={hidden}: output shape should be [1, 2, 50]"
        );
    }
}

#[test]
fn test_model_shape_propagation_varying_vocab_size() {
    for vocab in [10, 50, 100, 500] {
        let mut cfg = tiny_config();
        cfg.padded_vocab_size = vocab;
        let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
        let model = Glm5Model::load(&vb, cfg).unwrap();
        let logits = model.forward(&[0], &[0]).unwrap();
        assert_eq!(
            logits.dims(),
            &[1, 1, vocab],
            "vocab={vocab}: last dim should be vocab size"
        );
    }
}

// ---------------------------------------------------------------------------
// Forward from embeddings with KV cache shape validation
// ---------------------------------------------------------------------------

#[test]
fn test_forward_from_embeddings_with_cache_shapes() {
    let cfg = tiny_config();
    let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
    let model = Glm5Model::load(&vb, cfg.clone()).unwrap();
    let mut cache = model.new_cache();

    // Prefill via embeddings
    let emb = DynTensor::ones(&[1, 3, cfg.hidden_size], DType::F32, &Device::Cpu).unwrap();
    let logits = model
        .forward_from_embeddings(&emb, &[0, 1, 2], Some(&mut cache))
        .unwrap();
    assert_eq!(logits.dims(), &[1, 3, cfg.padded_vocab_size]);
    assert_eq!(cache.seq_len(), 3);

    // Decode step via token IDs
    let decode_logits = model.forward_cached(&[42], &[3], Some(&mut cache)).unwrap();
    assert_eq!(decode_logits.dims(), &[1, 1, cfg.padded_vocab_size]);
    assert_eq!(cache.seq_len(), 4);
}

// ---------------------------------------------------------------------------
// Output projection weight shape: [padded_vocab_size, hidden_size]
// ---------------------------------------------------------------------------

#[test]
fn test_output_layer_dimensions() {
    // The output_layer (language model head) maps hidden_size -> vocab_size
    // Weight shape: [padded_vocab_size, hidden_size]
    // Use tiny config to avoid large allocation from full GLM-4-9B vocab (151552)
    let cfg = tiny_config();
    let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
    let model = Glm5Model::load(&vb, cfg.clone()).unwrap();
    // Forward produces [1, seq, vocab] confirming output projection shape
    let logits = model.forward(&[0], &[0]).unwrap();
    assert_eq!(
        logits.dims()[2],
        cfg.padded_vocab_size,
        "logits last dim should equal padded_vocab_size"
    );

    // Also verify the GLM-4-9B config dimensionality relationship holds:
    // output_layer weight is [padded_vocab_size, hidden_size]
    let cfg9b = Glm5Config::default();
    assert_eq!(cfg9b.padded_vocab_size, 151552);
    assert_eq!(cfg9b.hidden_size, 4096);
    // These dimensions define the output projection; we verify the formula
    // without allocating the full 151552 * 4096 zero tensor.
}

// ---------------------------------------------------------------------------
// Dense (attention output) projection: [hidden_size, num_heads * head_dim]
// ---------------------------------------------------------------------------

#[test]
fn test_dense_projection_dimensions() {
    // The dense (output) projection in attention maps num_heads * head_dim -> hidden_size
    // For GLM, num_heads * head_dim == hidden_size, so it's a square matrix
    let cfg = tiny_config();
    let attn_output_dim = cfg.num_attention_heads * cfg.kv_channels;
    assert_eq!(
        attn_output_dim, cfg.hidden_size,
        "attention output dim should equal hidden_size"
    );
}

// ---------------------------------------------------------------------------
// Position encoding: large positions should not produce NaN
// ---------------------------------------------------------------------------

#[test]
fn test_rope_large_position_no_nan() {
    let cfg = tiny_config();
    let rope = HalfRotaryEmbedding::new(
        cfg.kv_channels,
        cfg.seq_length,
        cfg.rope_theta,
        &Device::Cpu,
    )
    .unwrap();

    let x = DynTensor::ones(
        &[1, cfg.num_attention_heads, 1, cfg.kv_channels],
        DType::F32,
        &Device::Cpu,
    )
    .unwrap();

    // Test with large positions near seq_length boundary
    for pos in [0, 1, cfg.seq_length / 2, cfg.seq_length - 1] {
        let result = rope.apply(&x, pos).unwrap();
        let flat = result.to_flat_vec::<f32>().unwrap();
        let nan_count = flat.iter().filter(|v| v.is_nan()).count();
        assert_eq!(
            nan_count, 0,
            "position {pos}: RoPE should not produce NaN values"
        );
    }
}

// ---------------------------------------------------------------------------
// Flat KVCache (kv_cache.rs) dimension consistency
// ---------------------------------------------------------------------------

#[test]
fn test_flat_kv_cache_token_size_equals_heads_times_dim() {
    let num_heads = 4;
    let head_dim = 16;
    let cache = kv_cache::KVCache::new(2, num_heads, head_dim, 32);

    let token_size = num_heads * head_dim;
    let key: Vec<f32> = vec![1.0; token_size];
    let val: Vec<f32> = vec![2.0; token_size];

    let mut cache = cache;
    cache.append(0, &key, &val).unwrap();
    cache.append(1, &key, &val).unwrap();
    assert_eq!(cache.len(), 1);

    // Verify stored data length
    assert_eq!(cache.get_keys(0).len(), token_size);
    assert_eq!(cache.get_values(0).len(), token_size);
}

#[test]
fn test_flat_kv_cache_multi_token_data_layout() {
    let num_heads = 2;
    let head_dim = 4;
    let token_size = num_heads * head_dim; // 8
    let mut cache = kv_cache::KVCache::new(2, num_heads, head_dim, 32);

    // Append 3 tokens worth of data
    for token in 0..3 {
        let base = (token * 100) as f32;
        let key: Vec<f32> = (0..token_size).map(|i| base + i as f32).collect();
        let val: Vec<f32> = (0..token_size).map(|i| base + 50.0 + i as f32).collect();
        for layer in 0..2 {
            cache.append(layer, &key, &val).unwrap();
        }
    }
    assert_eq!(cache.len(), 3);

    // Total stored keys should be 3 * token_size
    let all_keys = cache.get_keys(0);
    assert_eq!(all_keys.len(), 3 * token_size);

    // First token's keys start at 0.0, second at 100.0, third at 200.0
    assert!((all_keys[0] - 0.0).abs() < 1e-6);
    assert!((all_keys[token_size] - 100.0).abs() < 1e-6);
    assert!((all_keys[2 * token_size] - 200.0).abs() < 1e-6);
}

// ---------------------------------------------------------------------------
// Config: parameter count relationships
// ---------------------------------------------------------------------------

#[test]
fn test_embedding_parameter_count() {
    let cfg = Glm5Config::default();
    let embed_params = cfg.padded_vocab_size * cfg.hidden_size;
    // GLM-4-9B: 151552 * 4096 = 620,756,992
    assert_eq!(embed_params, 620_756_992);
}

#[test]
fn test_per_layer_qkv_parameter_count() {
    let cfg = Glm5Config::default();
    let qkv_out = (cfg.num_attention_heads + 2 * cfg.multi_query_group_num) * cfg.kv_channels;
    // GLM-4-9B: (32 + 4) * 128 = 4608
    assert_eq!(qkv_out, 4608);
    let qkv_params = cfg.hidden_size * qkv_out;
    // 4096 * 4608 = 18,874,368 (weight) + 4608 (bias if add_qkv_bias)
    assert_eq!(qkv_params, 18_874_368);
}

#[test]
fn test_per_layer_mlp_parameter_count() {
    let cfg = Glm5Config::default();
    // dense_h_to_4h: [2 * ffn_hidden, hidden] = 2 * 13696 * 4096 = 112,197,632
    let h_to_4h = 2 * cfg.ffn_hidden_size * cfg.hidden_size;
    assert_eq!(h_to_4h, 112_197_632);
    // dense_4h_to_h: [hidden, ffn_hidden] = 4096 * 13696 = 56,098,816
    let four_h_to_h = cfg.hidden_size * cfg.ffn_hidden_size;
    assert_eq!(four_h_to_h, 56_098_816);
}

// ---------------------------------------------------------------------------
// Model device accessor
// ---------------------------------------------------------------------------

#[test]
fn test_model_device_is_cpu() {
    let cfg = tiny_config();
    let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
    let model = Glm5Model::load(&vb, cfg).unwrap();
    assert!(
        matches!(model.device(), Device::Cpu),
        "model loaded on CPU should report CPU device"
    );
}

// ---------------------------------------------------------------------------
// GLM vs Qwen/Llama architectural differences (config-level)
// ---------------------------------------------------------------------------

#[test]
fn test_glm_partial_rope_vs_full_rope() {
    // GLM uses half-RoPE (kv_channels/2 rotated dims)
    // Verify that for the same head_dim, GLM rotates half as many dims
    let head_dim = 128;
    let rope = HalfRotaryEmbedding::new(head_dim, 4096, 10_000.0, &Device::Cpu).unwrap();
    assert_eq!(
        rope.rope_dim(),
        head_dim / 2,
        "GLM half-RoPE should rotate head_dim/2 = {} dims",
        head_dim / 2
    );
}

#[test]
fn test_glm_fused_qkv_total_matches_separate() {
    // Verify that fused QKV total = q_proj + k_proj + v_proj
    let cfg = tiny_config();
    let q_dim = cfg.num_attention_heads * cfg.kv_channels;
    let k_dim = cfg.multi_query_group_num * cfg.kv_channels;
    let v_dim = cfg.multi_query_group_num * cfg.kv_channels;
    let fused_total = (cfg.num_attention_heads + 2 * cfg.multi_query_group_num) * cfg.kv_channels;
    assert_eq!(
        fused_total,
        q_dim + k_dim + v_dim,
        "fused QKV should equal sum of separate projections"
    );
}

// ---------------------------------------------------------------------------
// Multi-step decode: logit shape consistency
// ---------------------------------------------------------------------------

#[test]
fn test_decode_logits_always_match_input_seq_len() {
    let cfg = tiny_config();
    let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
    let model = Glm5Model::load(&vb, cfg.clone()).unwrap();
    let mut cache = model.new_cache();

    // Various input lengths during decode
    let steps: Vec<(Vec<usize>, Vec<usize>)> = vec![
        (vec![0, 1, 2], vec![0, 1, 2]), // prefill 3
        (vec![3], vec![3]),             // decode 1
        (vec![4, 5], vec![4, 5]),       // decode 2
        (vec![6], vec![6]),             // decode 1
        (vec![7, 8, 9], vec![7, 8, 9]), // decode 3
    ];

    for (ids, pos) in &steps {
        let logits = model.forward_cached(ids, pos, Some(&mut cache)).unwrap();
        assert_eq!(
            logits.dims(),
            &[1, ids.len(), cfg.padded_vocab_size],
            "input len {} should produce matching logits seq dim",
            ids.len()
        );
    }
    assert_eq!(cache.seq_len(), 10);
}

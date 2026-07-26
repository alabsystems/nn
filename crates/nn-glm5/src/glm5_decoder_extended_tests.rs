// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Extended architecture validation tests for the GLM-5 decoder-only LLM.
//!
//! Part of #4186. Covers:
//!
//! 1. Config validation across model sizes (GLM-4-9B, chat, tiny, custom)
//! 2. Rotary embedding dimension calculations for various head dims
//! 3. GQA key/value head counts vs query head counts
//! 4. SwiGLU FFN intermediate dimension correctness
//! 5. Causal attention mask shape and values
//! 6. Token embedding (vocab_size x hidden_dim) table lookup
//! 7. Output projection produces [batch, seq, vocab_size] logits
//! 8. KV cache shape updates across decoding steps

use super::*;
use crate::kv_cache::KVCache;
use crate::test_utils::tiny_config;
use nn_core::dyn_tensor::DynTensor;
use nn_core::layers::HalfRotaryEmbedding;
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

fn load_model_with_config(cfg: Glm5Config) -> Glm5Model {
    let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
    Glm5Model::load(&vb, cfg).unwrap()
}

// ===========================================================================
// 1. Config validation: verify GLM5 configs for different model sizes
// ===========================================================================

/// GLM-4-9B default config satisfies the fundamental architectural invariant
/// hidden_size == num_attention_heads * kv_channels.
#[test]
fn test_config_9b_architectural_invariant() {
    let cfg = Glm5Config::default();
    assert_eq!(
        cfg.hidden_size,
        cfg.num_attention_heads * cfg.kv_channels,
        "9B: hidden_size must equal num_heads * kv_channels"
    );
}

/// GLM-4-9B-chat config shares architectural dimensions with base but
/// differs in context-extension parameters.
#[test]
fn test_config_chat_layer_head_dim_match_base() {
    let base = Glm5Config::default();
    let chat = Glm5Config::glm4_9b_chat();

    assert_eq!(chat.num_layers, base.num_layers);
    assert_eq!(chat.num_attention_heads, base.num_attention_heads);
    assert_eq!(chat.kv_channels, base.kv_channels);
    assert_eq!(chat.hidden_size, base.hidden_size);
    assert_eq!(chat.ffn_hidden_size, base.ffn_hidden_size);

    // Context extension only
    assert!(chat.seq_length > base.seq_length);
    assert!(chat.rope_theta > base.rope_theta);
}

/// A hypothetical smaller GLM model (e.g., 1.5B-class) with 24 layers,
/// 16 heads, 2 KV groups validates correctly.
#[test]
fn test_config_small_model_validates() {
    let cfg = Glm5Config::new(
        2048,     // hidden_size
        5504,     // ffn_hidden_size (approx 2/3 * 4 * 2048 rounded)
        24,       // num_layers
        16,       // num_attention_heads
        2,        // multi_query_group_num
        151552,   // padded_vocab_size
        128,      // kv_channels: 2048 / 16 = 128
        1e-5,     // layernorm_epsilon
        8192,     // seq_length
        true,     // rmsnorm
        true,     // add_qkv_bias
        false,    // add_bias_linear
        10_000.0, // rope_theta
    );
    assert!(cfg.validate().is_ok());
    assert_eq!(cfg.num_layers, 24);
    assert_eq!(cfg.num_attention_heads, 16);
    assert_eq!(cfg.head_dim(), 128);
    assert_eq!(
        cfg.hidden_size,
        cfg.num_attention_heads * cfg.kv_channels,
        "small model: hidden = heads * head_dim"
    );
}

/// Validation rejects configs where kv_channels is not a multiple of 4
/// (HalfRotaryEmbedding requirement for the inner rope_dim = head_dim/2
/// to be even).
#[test]
fn test_config_rejects_kv_channels_not_multiple_of_4() {
    let mut cfg = tiny_config();
    cfg.kv_channels = 6; // not divisible by 4
    assert!(cfg.validate().is_err());

    cfg.kv_channels = 12; // divisible by 4
    assert!(cfg.validate().is_ok());
}

/// Validation rejects NaN rope_theta.
#[test]
fn test_config_rejects_nan_rope_theta() {
    let mut cfg = tiny_config();
    cfg.rope_theta = f64::NAN;
    assert!(cfg.validate().is_err());
}

/// Validation rejects zero padded_vocab_size.
#[test]
fn test_config_rejects_zero_vocab_size() {
    let mut cfg = tiny_config();
    cfg.padded_vocab_size = 0;
    assert!(cfg.validate().is_err());
}

// ===========================================================================
// 2. Rotary embedding dimensions for different head dims
// ===========================================================================

/// For GLM's half-RoPE, the rotary dimension is always head_dim / 2.
/// Verify this across a range of valid head dimensions.
#[test]
fn test_rope_dim_is_half_head_dim_for_all_valid_sizes() {
    for head_dim in [4, 8, 16, 32, 64, 128, 256] {
        let rope = HalfRotaryEmbedding::new(head_dim, 128, 10_000.0, &Device::Cpu).unwrap();
        assert_eq!(
            rope.rope_dim(),
            head_dim / 2,
            "head_dim={head_dim}: rope_dim should be head_dim/2"
        );
        assert_eq!(rope.head_dim(), head_dim);
    }
}

/// RoPE frequency at index i follows theta_i = 1 / base^(2i / rope_dim*2).
/// The first frequency (i=0) should be 1.0 regardless of base.
/// Each subsequent frequency should be smaller.
#[test]
fn test_rope_frequency_computation_for_different_head_dims() {
    for (head_dim, base) in [
        (64, 10_000.0_f64),
        (128, 10_000.0_f64),
        (64, 5_000_000.0_f64),
    ] {
        let rope_dim = head_dim / 2; // half-RoPE
        let freqs: Vec<f64> = (0..rope_dim)
            .map(|i| 1.0 / base.powf(2.0 * i as f64 / (2 * rope_dim) as f64))
            .collect();

        assert!(
            (freqs[0] - 1.0).abs() < 1e-12,
            "head_dim={head_dim}, base={base}: freq[0] should be 1.0"
        );

        // Strictly decreasing
        for w in freqs.windows(2) {
            assert!(
                w[0] > w[1],
                "head_dim={head_dim}, base={base}: freqs should decrease: {} vs {}",
                w[0],
                w[1]
            );
        }

        // Last frequency should be well below 1.0 for large dim
        if rope_dim > 1 {
            assert!(
                freqs[rope_dim - 1] < 0.5,
                "head_dim={head_dim}, base={base}: last freq should be small, got {}",
                freqs[rope_dim - 1]
            );
        }
    }
}

/// Higher rope_theta produces *smaller* inverse-frequencies at any index i>0
/// (slower rotation per position), which is what gives long-context models their
/// extended reach. From freq_i = 1 / theta^(2i/(2*rope_dim)), the exponent is
/// positive for i>0, so a larger theta yields a larger denominator and thus a
/// smaller freq. This is used for long-context models (e.g., chat variant with
/// theta=5M).
#[test]
fn test_rope_higher_theta_means_slower_decay() {
    let rope_dim = 32;
    let mid_idx = rope_dim / 2;

    let freq_10k = 1.0 / 10_000.0_f64.powf(2.0 * f64::from(mid_idx) / f64::from(2 * rope_dim));
    let freq_5m = 1.0 / 5_000_000.0_f64.powf(2.0 * f64::from(mid_idx) / f64::from(2 * rope_dim));

    // For a fixed positive index, higher theta => smaller inverse-frequency
    // (slower rotation). 5M=0.000447 vs 10K=0.01, so freq_5m < freq_10k.
    assert!(
        freq_5m < freq_10k,
        "higher theta should yield smaller (slower) mid-frequency: 5M={freq_5m} vs 10K={freq_10k}"
    );
}

// ===========================================================================
// 3. GQA (grouped query attention): KV heads vs Q heads
// ===========================================================================

/// In GQA, each KV head group is shared by (num_heads / multi_query_group_num)
/// query heads. Verify the repeat factor for multiple configs.
#[test]
fn test_gqa_repeat_factor_across_configs() {
    let test_cases: Vec<(&str, usize, usize, usize)> = vec![
        ("9B", 32, 2, 16), // GLM-4-9B: 32Q / 2KV = 16x repeat
        ("tiny", 4, 2, 2), // tiny: 4Q / 2KV = 2x repeat
        ("MQA", 8, 1, 8),  // MQA: 8Q / 1KV = 8x repeat
        ("MHA", 4, 4, 1),  // MHA: 4Q / 4KV = 1x repeat (no repeat)
    ];

    for (name, nh, nkv, expected_repeat) in test_cases {
        let repeat = nh / nkv;
        assert_eq!(
            repeat, expected_repeat,
            "{name}: {nh}Q / {nkv}KV should give {expected_repeat}x repeat"
        );
    }
}

/// The KV projection size is much smaller than Q projection in GQA.
/// This is a key memory saving. Verify the ratio.
#[test]
fn test_gqa_kv_projection_smaller_than_q() {
    let cfg = Glm5Config::default();
    let q_proj_size = cfg.num_attention_heads * cfg.kv_channels;
    let kv_proj_size = cfg.multi_query_group_num * cfg.kv_channels;

    assert!(
        kv_proj_size < q_proj_size,
        "KV projection ({kv_proj_size}) should be smaller than Q projection ({q_proj_size})"
    );

    let ratio = q_proj_size as f64 / kv_proj_size as f64;
    assert!(
        (ratio - 16.0).abs() < 0.01,
        "GLM-4-9B: Q/KV ratio should be 16 (32 heads / 2 KV groups), got {ratio}"
    );
}

/// In MQA (multi_query_group_num=1), there is exactly one set of K and V
/// shared across all Q heads. The memory savings is maximal.
#[test]
fn test_gqa_mqa_single_kv_head_forward() {
    let cfg = Glm5Config::new(
        256,      // hidden_size (8 heads * 32 dim)
        512,      // ffn_hidden_size
        1,        // num_layers
        8,        // num_attention_heads
        1,        // multi_query_group_num (MQA)
        64,       // padded_vocab_size
        32,       // kv_channels
        1e-5,     // layernorm_epsilon
        32,       // seq_length
        true,     // rmsnorm
        true,     // add_qkv_bias
        false,    // add_bias_linear
        10_000.0, // rope_theta
    );
    assert_eq!(cfg.num_kv_groups().unwrap(), 8);

    let model = load_model_with_config(cfg.clone());
    let logits = model.forward(&[0, 1, 2], &[0, 1, 2]).unwrap();
    assert_eq!(logits.dims(), &[1, 3, cfg.padded_vocab_size]);
}

/// In MHA (multi_query_group_num == num_attention_heads), each Q head has its
/// own dedicated K/V head. No sharing, no repeat.
#[test]
fn test_gqa_mha_no_sharing_forward() {
    let cfg = Glm5Config::new(
        256,      // hidden_size (4 heads * 64 dim)
        512,      // ffn_hidden_size
        1,        // num_layers
        4,        // num_attention_heads
        4,        // multi_query_group_num (MHA, same as num_heads)
        64,       // padded_vocab_size
        64,       // kv_channels
        1e-5,     // layernorm_epsilon
        32,       // seq_length
        true,     // rmsnorm
        true,     // add_qkv_bias
        false,    // add_bias_linear
        10_000.0, // rope_theta
    );
    assert_eq!(cfg.num_kv_groups().unwrap(), 1, "MHA: repeat factor is 1");

    // QKV fused size in MHA: (nh + 2*nh) * hd = 3 * nh * hd
    let qkv_size = (cfg.num_attention_heads + 2 * cfg.multi_query_group_num) * cfg.kv_channels;
    let expected = 3 * cfg.num_attention_heads * cfg.kv_channels;
    assert_eq!(qkv_size, expected, "MHA: fused QKV = 3 * hidden_size");

    let model = load_model_with_config(cfg.clone());
    let logits = model.forward(&[0], &[0]).unwrap();
    assert_eq!(logits.dims(), &[1, 1, cfg.padded_vocab_size]);
}

// ===========================================================================
// 4. SwiGLU FFN intermediate dimension
// ===========================================================================

/// GLM-4-9B uses ffn_hidden_size = 13696 which is approximately
/// 2/3 * 4 * hidden_size = 2/3 * 4 * 4096 = 10922.67 rounded up.
/// The actual value 13696 is a different formula: it includes SwiGLU overhead.
#[test]
fn test_swiglu_ffn_dim_relationship_to_hidden() {
    let cfg = Glm5Config::default();
    // The SwiGLU MLP in GLM uses a fused gate+up projection of size ffn_hidden_size * 2
    // which is then split in half. The ratio ffn/hidden varies by model family.
    let ratio = cfg.ffn_hidden_size as f64 / cfg.hidden_size as f64;
    // GLM-4-9B uses 13696/4096 = 3.34375
    assert!(
        (ratio - 3.34375).abs() < 1e-6,
        "GLM-4-9B ffn expansion ratio: expected 3.34375, got {ratio}"
    );
}

/// The fused dense_h_to_4h weight has 2x ffn_hidden_size output because
/// it contains both gate and up projections concatenated.
#[test]
fn test_swiglu_fused_gate_up_dimension() {
    for (name, ffn, hidden) in [("tiny", 512_usize, 256_usize), ("9B", 13696, 4096)] {
        let fused_out = ffn * 2;
        assert_eq!(
            fused_out % 2,
            0,
            "{name}: fused output must be even for gate/up split"
        );
        assert_eq!(
            fused_out / 2,
            ffn,
            "{name}: each half of fused output equals ffn_hidden_size"
        );
        // dense_h_to_4h weight shape: [ffn*2, hidden]
        // dense_4h_to_h weight shape: [hidden, ffn]
        let h_to_4h_params = fused_out * hidden;
        let four_h_to_h_params = hidden * ffn;
        assert!(
            h_to_4h_params > four_h_to_h_params,
            "{name}: dense_h_to_4h has more params than dense_4h_to_h"
        );
    }
}

/// SwiGLU computation: silu(gate) * up correctly zeros when gate is zero.
/// This is important for sparsity patterns in the FFN.
#[test]
fn test_swiglu_silu_values_at_key_points() {
    // silu(x) = x * sigmoid(x)
    // silu(0) = 0 * 0.5 = 0
    // silu(1) = 1 * sigmoid(1) = 1 / (1 + e^-1) ≈ 0.7311
    // silu(-1) = -1 * sigmoid(-1) = -1 / (1 + e^1) ≈ -0.2689
    let inputs = vec![0.0_f32, 1.0, -1.0, 5.0, -5.0];
    let x = DynTensor::from_vec(inputs, &[1, 1, 5], &Device::Cpu).unwrap();
    let result = x.silu().unwrap();
    let out = result.to_flat_vec::<f32>().unwrap();

    // silu(0) = 0
    assert!(out[0].abs() < 1e-6, "silu(0) should be 0, got {}", out[0]);
    // silu(1) ≈ 0.7311
    assert!(
        (out[1] - 0.7311).abs() < 0.001,
        "silu(1) should be ~0.7311, got {}",
        out[1]
    );
    // silu(-1) ≈ -0.2689
    assert!(
        (out[2] - (-0.2689)).abs() < 0.001,
        "silu(-1) should be ~-0.2689, got {}",
        out[2]
    );
    // silu is monotonically increasing for x >= 0 (silu(5) > silu(1)), but it is
    // NOT monotonic on the negatives: it has a global minimum near x ~= -1.278,
    // then rises back toward 0 as x -> -inf. So silu(-5) (~ -0.033) is *greater*
    // than silu(-1) (~ -0.269).
    assert!(out[3] > out[1], "silu(5) > silu(1)");
    assert!(out[4] > out[2], "silu(-5) > silu(-1)");
}

// ===========================================================================
// 5. Causal attention mask shape and values
// ===========================================================================

/// Causal mask for a prompt of length S has shape [1, 1, S, S].
/// Lower triangle (including diagonal) = 0.0, upper triangle = -inf.
#[test]
fn test_causal_mask_lower_triangle_zero_upper_neginf() {
    let seq_len = 4;
    let mask = causal_mask(seq_len, DType::F32, &Device::Cpu).unwrap();
    assert_eq!(mask.dims(), &[1, 1, seq_len, seq_len]);

    let data = mask.to_flat_vec::<f32>().unwrap();
    for row in 0..seq_len {
        for col in 0..seq_len {
            let val = data[row * seq_len + col];
            if col <= row {
                assert_eq!(
                    val, 0.0,
                    "mask[{row},{col}] should be 0.0 (visible), got {val}"
                );
            } else {
                assert!(
                    val.is_infinite() && val < 0.0,
                    "mask[{row},{col}] should be -inf (masked), got {val}"
                );
            }
        }
    }
}

/// Causal mask with KV cache offset: new tokens see all previous cached tokens.
/// Shape is [1, 1, new_seq, total_seq] where total_seq = cached + new.
#[test]
fn test_causal_mask_with_offset_shape_and_values() {
    let new_tokens = 2;
    let cached_tokens = 5;
    let total = cached_tokens + new_tokens;
    let mask = causal_mask_with_offset(new_tokens, total, DType::F32, &Device::Cpu).unwrap();
    assert_eq!(
        mask.dims(),
        &[1, 1, new_tokens, total],
        "offset mask shape: [1, 1, new, total]"
    );

    let data = mask.to_flat_vec::<f32>().unwrap();
    // Row 0 corresponds to the first new token (position = cached_tokens).
    // It can see all cached tokens (0..cached_tokens) and itself.
    // Row 0, col range: [0..cached_tokens] = visible (0.0), [cached_tokens+1..total] = masked (-inf)
    for col in 0..total {
        let val = data[col]; // row 0
        if col <= cached_tokens {
            assert_eq!(
                val, 0.0,
                "row 0, col {col}: first new token should see cached + itself"
            );
        } else {
            assert!(
                val.is_infinite() && val < 0.0,
                "row 0, col {col}: first new token should not see future"
            );
        }
    }
    // Row 1 = second new token, sees all cached + both new tokens
    for col in 0..total {
        let val = data[total + col]; // row 1
        if col <= cached_tokens + 1 {
            assert_eq!(
                val, 0.0,
                "row 1, col {col}: second new token should see cached + both new"
            );
        }
        // (no future tokens remain for the last new token in this case)
    }
}

/// Single-token decode step: mask is None (model skips mask for seq_len=1).
/// Verify the model handles this case correctly.
#[test]
fn test_causal_mask_skipped_for_single_token_decode() {
    let model = load_tiny_model();
    let mut cache = model.new_cache();

    // Prefill with mask (seq_len > 1)
    let _ = model
        .forward_cached(&[0, 1, 2], &[0, 1, 2], Some(&mut cache))
        .unwrap();

    // Decode single token: no explicit mask needed, attention is over
    // entire KV cache (all positions are visible from a single query).
    let logits = model.forward_cached(&[3], &[3], Some(&mut cache)).unwrap();
    assert_eq!(
        logits.dims(),
        &[1, 1, model.config().padded_vocab_size],
        "single-token decode should produce [1, 1, vocab]"
    );
    let flat = logits.to_flat_vec::<f32>().unwrap();
    assert!(
        flat.iter().all(|v| v.is_finite()),
        "single-token decode logits should be finite"
    );
}

// ===========================================================================
// 6. Token embedding: vocab_size x hidden_dim table lookup
// ===========================================================================

/// The embedding table maps each token ID in [0, padded_vocab_size) to a
/// vector of hidden_size dimensions. Verify boundary token IDs work.
#[test]
fn test_embedding_boundary_token_ids() {
    let model = load_tiny_model();
    let cfg = model.config().clone();
    let vocab = cfg.padded_vocab_size;

    // Token ID 0 (first entry)
    let logits_first = model.forward(&[0], &[0]).unwrap();
    assert_eq!(logits_first.dims(), &[1, 1, vocab]);

    // Token ID vocab-1 (last valid entry)
    let logits_last = model.forward(&[vocab - 1], &[0]).unwrap();
    assert_eq!(logits_last.dims(), &[1, 1, vocab]);

    // Both should produce finite outputs
    let first_flat = logits_first.to_flat_vec::<f32>().unwrap();
    let last_flat = logits_last.to_flat_vec::<f32>().unwrap();
    assert!(first_flat.iter().all(|v| v.is_finite()));
    assert!(last_flat.iter().all(|v| v.is_finite()));
}

/// Multiple different token IDs in the same sequence produce the same
/// output shape regardless of which tokens are used.
#[test]
fn test_embedding_different_tokens_produce_same_shape() {
    let model = load_tiny_model();
    let cfg = model.config().clone();

    let logits_a = model.forward(&[0, 1, 2], &[0, 1, 2]).unwrap();
    let logits_b = model.forward(&[50, 60, 70], &[0, 1, 2]).unwrap();
    assert_eq!(logits_a.dims(), logits_b.dims());
    assert_eq!(logits_a.dims(), &[1, 3, cfg.padded_vocab_size]);
}

/// The embedding weight has shape [padded_vocab_size, hidden_size].
/// Verify indirectly by checking the forward pass produces the correct
/// output dimension chain: token_id -> [hidden_size] -> ... -> [vocab_size].
#[test]
fn test_embedding_hidden_dim_chain() {
    // Use a custom config to verify the dimension chain is correct
    // with a different hidden_size / vocab_size combination.
    let cfg = Glm5Config::new(
        128,      // hidden_size
        256,      // ffn_hidden_size
        1,        // num_layers
        2,        // num_attention_heads
        1,        // multi_query_group_num
        42,       // padded_vocab_size (non-standard)
        64,       // kv_channels
        1e-5,     // layernorm_epsilon
        32,       // seq_length
        true,     // rmsnorm
        false,    // add_qkv_bias
        false,    // add_bias_linear
        10_000.0, // rope_theta
    );
    let model = load_model_with_config(cfg);
    let logits = model.forward(&[5], &[0]).unwrap();
    // Embedding: token 5 -> [1, 1, 128] hidden
    // ... layers ...
    // Output projection: [1, 1, 128] -> [1, 1, 42]
    assert_eq!(logits.dims(), &[1, 1, 42]);
}

// ===========================================================================
// 7. Output projection: logits shape is [batch, seq, vocab_size]
// ===========================================================================

/// Output projection consistently produces [1, seq_len, vocab] for
/// various sequence lengths.
#[test]
fn test_output_projection_shape_varies_with_seq_len() {
    let model = load_tiny_model();
    let vocab = model.config().padded_vocab_size;

    for seq_len in [1, 2, 5, 10, 20] {
        let ids: Vec<usize> = (0..seq_len).collect();
        let pos: Vec<usize> = (0..seq_len).collect();
        let logits = model.forward(&ids, &pos).unwrap();
        assert_eq!(
            logits.dims(),
            &[1, seq_len, vocab],
            "seq_len={seq_len}: expected [1, {seq_len}, {vocab}]"
        );
    }
}

/// Output logits from forward_from_embeddings have the same shape as from
/// forward with token IDs.
#[test]
fn test_output_projection_from_embeddings_matches_from_ids() {
    let model = load_tiny_model();
    let cfg = model.config().clone();

    let logits_ids = model.forward(&[0, 1, 2], &[0, 1, 2]).unwrap();

    let emb = DynTensor::zeros(&[1, 3, cfg.hidden_size], DType::F32, &Device::Cpu).unwrap();
    let logits_emb = model
        .forward_from_embeddings(&emb, &[0, 1, 2], None)
        .unwrap();

    assert_eq!(
        logits_ids.dims(),
        logits_emb.dims(),
        "logits from IDs and embeddings should have same shape"
    );
    assert_eq!(logits_ids.dims(), &[1, 3, cfg.padded_vocab_size]);
}

/// Output projection shape is correct for models of varying vocab sizes.
#[test]
fn test_output_projection_varying_vocab_sizes() {
    for vocab in [16, 32, 100, 256, 1000] {
        let mut cfg = tiny_config();
        cfg.padded_vocab_size = vocab;
        let model = load_model_with_config(cfg);
        let logits = model.forward(&[0, 1], &[0, 1]).unwrap();
        assert_eq!(
            logits.dims()[2],
            vocab,
            "vocab={vocab}: last dim should equal padded_vocab_size"
        );
    }
}

/// Logits are finite for a multi-step cached decode sequence.
#[test]
fn test_output_projection_logits_finite_after_cached_decode() {
    let model = load_tiny_model();
    let vocab = model.config().padded_vocab_size;
    let mut cache = model.new_cache();

    // Prefill
    let prefill = model
        .forward_cached(&[0, 1, 2, 3, 4], &[0, 1, 2, 3, 4], Some(&mut cache))
        .unwrap();
    assert_eq!(prefill.dims(), &[1, 5, vocab]);

    // 5 decode steps
    for step in 0..5 {
        let logits = model
            .forward_cached(&[42], &[5 + step], Some(&mut cache))
            .unwrap();
        assert_eq!(logits.dims(), &[1, 1, vocab]);
        let flat = logits.to_flat_vec::<f32>().unwrap();
        let non_finite = flat.iter().filter(|v| !v.is_finite()).count();
        assert_eq!(
            non_finite, 0,
            "decode step {step}: all logits should be finite"
        );
    }
}

// ===========================================================================
// 8. KV cache shape updates per decoding step
// ===========================================================================

/// DynTensor KV cache seq_len grows by exactly 1 per single-token decode step.
#[test]
fn test_kv_cache_seq_len_grows_per_step() {
    let model = load_tiny_model();
    let mut cache = model.new_cache();
    assert_eq!(cache.seq_len(), 0);

    // Prefill 4 tokens
    let _ = model
        .forward_cached(&[0, 1, 2, 3], &[0, 1, 2, 3], Some(&mut cache))
        .unwrap();
    assert_eq!(cache.seq_len(), 4);

    // 6 single-token decode steps
    for step in 0..6 {
        let _ = model
            .forward_cached(&[0], &[4 + step], Some(&mut cache))
            .unwrap();
        assert_eq!(
            cache.seq_len(),
            5 + step,
            "after step {step}: cache seq_len should be {}",
            5 + step
        );
    }
    assert_eq!(cache.seq_len(), 10, "total after prefill + 6 decode = 10");
}

/// Each layer in the KV cache stores [1, num_kv_heads, seq_len, head_dim]
/// tensors. Verify the per-layer shapes after prefill and decode.
#[test]
fn test_kv_cache_per_layer_shape_after_prefill_and_decode() {
    let cfg = tiny_config();
    let model = load_model_with_config(cfg.clone());
    let mut cache = model.new_cache();

    // Prefill 3 tokens
    let _ = model
        .forward_cached(&[0, 1, 2], &[0, 1, 2], Some(&mut cache))
        .unwrap();

    for layer_idx in 0..cfg.num_layers {
        let layer = cache.layer(layer_idx).unwrap();
        if let Ok(Some(k)) = layer.k() {
            assert_eq!(k.dims()[0], 1, "batch=1");
            assert_eq!(k.dims()[1], cfg.multi_query_group_num, "kv heads");
            assert_eq!(k.dims()[2], 3, "seq_len=3 after prefill");
            assert_eq!(k.dims()[3], cfg.kv_channels, "head_dim");
        }
    }

    // Decode 2 more tokens
    let _ = model.forward_cached(&[3], &[3], Some(&mut cache)).unwrap();
    let _ = model.forward_cached(&[4], &[4], Some(&mut cache)).unwrap();

    for layer_idx in 0..cfg.num_layers {
        let layer = cache.layer(layer_idx).unwrap();
        if let Ok(Some(k)) = layer.k() {
            assert_eq!(
                k.dims()[2],
                5,
                "layer {layer_idx}: seq_len should be 5 after 3 prefill + 2 decode"
            );
        }
    }
}

/// Flat KV cache (kv_cache::KVCache) stores num_heads * head_dim floats per
/// token per layer. Verify data layout after multiple appends.
#[test]
fn test_flat_kv_cache_data_size_per_token() {
    let num_layers = 2;
    let num_heads = 4;
    let head_dim = 8;
    let token_size = num_heads * head_dim; // 32 floats per token
    let mut cache = KVCache::new(num_layers, num_heads, head_dim, 64);

    let key_t1: Vec<f32> = (0..token_size).map(|i| i as f32).collect();
    let val_t1: Vec<f32> = (0..token_size).map(|i| (i + 100) as f32).collect();

    // Append token 1 to all layers
    for layer in 0..num_layers {
        cache.append(layer, &key_t1, &val_t1).unwrap();
    }
    assert_eq!(cache.len(), 1);
    assert_eq!(cache.get_keys(0).len(), token_size);
    assert_eq!(cache.get_values(0).len(), token_size);

    let key_t2: Vec<f32> = (0..token_size).map(|i| (i + 200) as f32).collect();
    let val_t2: Vec<f32> = (0..token_size).map(|i| (i + 300) as f32).collect();

    // Append token 2 to all layers
    for layer in 0..num_layers {
        cache.append(layer, &key_t2, &val_t2).unwrap();
    }
    assert_eq!(cache.len(), 2);
    assert_eq!(cache.get_keys(0).len(), 2 * token_size);
    assert_eq!(cache.get_values(0).len(), 2 * token_size);

    // Verify data order: token 1 keys followed by token 2 keys
    let all_keys = cache.get_keys(0);
    assert!((all_keys[0] - 0.0).abs() < 1e-6, "first token key[0]");
    assert!(
        (all_keys[token_size] - 200.0).abs() < 1e-6,
        "second token key[0]"
    );
}

/// After clearing the flat cache and reusing, the new data layout is clean.
#[test]
fn test_flat_kv_cache_clear_resets_completely() {
    let num_layers = 1;
    let num_heads = 2;
    let head_dim = 4;
    let token_size = num_heads * head_dim;
    let mut cache = KVCache::new(num_layers, num_heads, head_dim, 32);

    let old_key: Vec<f32> = vec![99.0; token_size];
    let old_val: Vec<f32> = vec![88.0; token_size];
    cache.append(0, &old_key, &old_val).unwrap();
    assert_eq!(cache.len(), 1);

    cache.clear();
    assert_eq!(cache.len(), 0);
    assert!(cache.is_empty());
    assert!(cache.get_keys(0).is_empty());

    // Append new data after clear
    let new_key: Vec<f32> = vec![1.0; token_size];
    let new_val: Vec<f32> = vec![2.0; token_size];
    cache.append(0, &new_key, &new_val).unwrap();
    assert_eq!(cache.len(), 1);
    assert_eq!(cache.get_keys(0), new_key.as_slice());
    assert_eq!(cache.get_values(0), new_val.as_slice());
}

/// Verify that the model's new_cache() produces a cache with the correct
/// number of layers and starts empty.
#[test]
fn test_model_new_cache_initial_state() {
    let model = load_tiny_model();
    let cache = model.new_cache();
    assert_eq!(cache.num_layers(), model.config().num_layers);
    assert_eq!(cache.seq_len(), 0, "new cache should be empty");
}

// ===========================================================================
// 9. Config trait and serialization behavior
// ===========================================================================

/// Glm5Config implements Clone: cloned config produces identical field values.
#[test]
fn test_config_clone_produces_identical_copy() {
    let original = Glm5Config::glm4_9b_chat();
    let cloned = original.clone();
    assert_eq!(original.hidden_size, cloned.hidden_size);
    assert_eq!(original.ffn_hidden_size, cloned.ffn_hidden_size);
    assert_eq!(original.num_layers, cloned.num_layers);
    assert_eq!(original.num_attention_heads, cloned.num_attention_heads);
    assert_eq!(original.multi_query_group_num, cloned.multi_query_group_num);
    assert_eq!(original.padded_vocab_size, cloned.padded_vocab_size);
    assert_eq!(original.kv_channels, cloned.kv_channels);
    assert_eq!(original.layernorm_epsilon, cloned.layernorm_epsilon);
    assert_eq!(original.seq_length, cloned.seq_length);
    assert_eq!(original.rmsnorm, cloned.rmsnorm);
    assert_eq!(original.add_qkv_bias, cloned.add_qkv_bias);
    assert_eq!(original.add_bias_linear, cloned.add_bias_linear);
    assert_eq!(original.rope_theta, cloned.rope_theta);
}

/// Glm5Config implements Debug: format string is non-empty and contains field names.
#[test]
fn test_config_debug_contains_field_names() {
    let cfg = tiny_config();
    let debug_str = format!("{cfg:?}");
    assert!(
        debug_str.contains("hidden_size"),
        "Debug output should contain 'hidden_size': {debug_str}"
    );
    assert!(
        debug_str.contains("num_layers"),
        "Debug output should contain 'num_layers': {debug_str}"
    );
    assert!(
        debug_str.contains("rope_theta"),
        "Debug output should contain 'rope_theta': {debug_str}"
    );
}

/// Modifying a cloned config does not affect the original.
#[test]
fn test_config_clone_independence() {
    let original = tiny_config();
    let mut modified = original.clone();
    modified.hidden_size = 9999;
    modified.num_layers = 128;
    modified.rope_theta = 1e12;

    assert_eq!(original.hidden_size, 256, "original should be unchanged");
    assert_eq!(original.num_layers, 2, "original should be unchanged");
    assert_eq!(
        original.rope_theta, 10_000.0,
        "original should be unchanged"
    );
    assert_eq!(modified.hidden_size, 9999);
    assert_eq!(modified.num_layers, 128);
}

// ===========================================================================
// 10. Config validation edge cases
// ===========================================================================

/// Validation rejects Inf rope_theta.
#[test]
fn test_config_rejects_inf_rope_theta() {
    let mut cfg = tiny_config();
    cfg.rope_theta = f64::INFINITY;
    let err = cfg.validate().unwrap_err();
    let msg = err.to_string();
    assert!(
        msg.contains("rope_theta"),
        "error should mention rope_theta: {msg}"
    );
}

/// Validation rejects negative rope_theta.
#[test]
fn test_config_rejects_negative_rope_theta() {
    let mut cfg = tiny_config();
    cfg.rope_theta = -1.0;
    assert!(cfg.validate().is_err());
}

/// Validation rejects zero rope_theta.
#[test]
fn test_config_rejects_zero_rope_theta() {
    let mut cfg = tiny_config();
    cfg.rope_theta = 0.0;
    assert!(cfg.validate().is_err());
}

/// Validation rejects zero kv_channels.
#[test]
fn test_config_rejects_zero_kv_channels() {
    let mut cfg = tiny_config();
    cfg.kv_channels = 0;
    let err = cfg.validate().unwrap_err();
    let msg = err.to_string();
    assert!(
        msg.contains("kv_channels"),
        "error should mention kv_channels: {msg}"
    );
}

/// Validation rejects zero num_attention_heads.
#[test]
fn test_config_rejects_zero_attention_heads() {
    let mut cfg = tiny_config();
    cfg.num_attention_heads = 0;
    assert!(cfg.validate().is_err());
}

/// Validation rejects zero multi_query_group_num.
#[test]
fn test_config_rejects_zero_multi_query_group_num() {
    let mut cfg = tiny_config();
    cfg.multi_query_group_num = 0;
    assert!(cfg.validate().is_err());
}

/// Validation rejects heads not divisible by kv groups.
#[test]
fn test_config_rejects_indivisible_heads_and_kv_groups() {
    let mut cfg = tiny_config();
    cfg.num_attention_heads = 5;
    cfg.multi_query_group_num = 3;
    let err = cfg.validate().unwrap_err();
    let msg = err.to_string();
    assert!(
        msg.contains("divisible"),
        "error should mention divisibility: {msg}"
    );
}

/// Validation rejects zero seq_length.
#[test]
fn test_config_rejects_zero_seq_length() {
    let mut cfg = tiny_config();
    cfg.seq_length = 0;
    assert!(cfg.validate().is_err());
}

/// Validation rejects zero num_layers.
#[test]
fn test_config_rejects_zero_num_layers() {
    let mut cfg = tiny_config();
    cfg.num_layers = 0;
    assert!(cfg.validate().is_err());
}

/// Validation rejects NaN layernorm_epsilon.
#[test]
fn test_config_rejects_nan_layernorm_epsilon() {
    let mut cfg = tiny_config();
    cfg.layernorm_epsilon = f64::NAN;
    assert!(cfg.validate().is_err());
}

/// Validation rejects zero layernorm_epsilon.
#[test]
fn test_config_rejects_zero_layernorm_epsilon() {
    let mut cfg = tiny_config();
    cfg.layernorm_epsilon = 0.0;
    assert!(cfg.validate().is_err());
}

// ===========================================================================
// 11. Flat KV cache extended tests
// ===========================================================================

/// Flat KVCache Clone preserves all data and metadata.
#[test]
fn test_flat_kv_cache_clone_preserves_data() {
    let num_layers = 2;
    let num_heads = 2;
    let head_dim = 4;
    let token_size = num_heads * head_dim;
    let mut original = KVCache::new(num_layers, num_heads, head_dim, 16);

    let key: Vec<f32> = (0..token_size).map(|i| (i + 1) as f32).collect();
    let val: Vec<f32> = (0..token_size).map(|i| (i + 50) as f32).collect();
    for layer in 0..num_layers {
        original.append(layer, &key, &val).unwrap();
    }

    let cloned = original.clone();
    assert_eq!(cloned.len(), original.len());
    assert_eq!(cloned.num_layers(), original.num_layers());
    assert_eq!(cloned.num_heads(), original.num_heads());
    assert_eq!(cloned.head_dim(), original.head_dim());
    assert_eq!(cloned.max_seq_len(), original.max_seq_len());
    assert_eq!(cloned.get_keys(0), original.get_keys(0));
    assert_eq!(cloned.get_values(0), original.get_values(0));
    assert_eq!(cloned.get_keys(1), original.get_keys(1));
    assert_eq!(cloned.get_values(1), original.get_values(1));
}

/// Flat KVCache max_seq_len accessor returns the configured maximum.
#[test]
fn test_flat_kv_cache_max_seq_len_accessor() {
    for max_len in [1, 16, 256, 8192] {
        let cache = KVCache::new(1, 2, 4, max_len);
        assert_eq!(
            cache.max_seq_len(),
            max_len,
            "max_seq_len accessor should return {max_len}"
        );
    }
}

/// Flat KVCache per-layer isolation: appending to layer 0 does not
/// affect layer 1 data.
#[test]
fn test_flat_kv_cache_layer_isolation() {
    let num_heads = 2;
    let head_dim = 4;
    let token_size = num_heads * head_dim;
    let mut cache = KVCache::new(2, num_heads, head_dim, 16);

    let key_a: Vec<f32> = vec![1.0; token_size];
    let val_a: Vec<f32> = vec![2.0; token_size];
    let key_b: Vec<f32> = vec![10.0; token_size];
    let val_b: Vec<f32> = vec![20.0; token_size];

    // Append different data to each layer
    cache.append(0, &key_a, &val_a).unwrap();
    cache.append(1, &key_b, &val_b).unwrap();

    // Layer 0 should have key_a data, layer 1 should have key_b data
    assert_eq!(cache.get_keys(0), key_a.as_slice());
    assert_eq!(cache.get_keys(1), key_b.as_slice());
    assert_eq!(cache.get_values(0), val_a.as_slice());
    assert_eq!(cache.get_values(1), val_b.as_slice());
}

// ===========================================================================
// 12. RoPE apply_pair with multiple positions
// ===========================================================================

/// apply_pair with multiple positions returns tensors with matching seq dim.
#[test]
fn test_rope_apply_pair_multi_position() {
    let head_dim = 8;
    let rope = HalfRotaryEmbedding::new(head_dim, 64, 10_000.0, &Device::Cpu).unwrap();

    // Multi-position sequence: [batch=1, heads=1, seq=3, head_dim=8]
    let data: Vec<f32> = (0..3 * head_dim).map(|i| (i + 1) as f32 * 0.1).collect();
    let q = DynTensor::from_vec(data.clone(), &[1, 1, 3, head_dim], &Device::Cpu).unwrap();
    let k = DynTensor::from_vec(data, &[1, 1, 3, head_dim], &Device::Cpu).unwrap();

    let (q_rot, k_rot) = rope.apply_pair(&q, &k, &[0, 1, 2]).unwrap();
    assert_eq!(q_rot.dims(), &[1, 1, 3, head_dim]);
    assert_eq!(k_rot.dims(), &[1, 1, 3, head_dim]);

    let q_flat = q_rot.to_flat_vec::<f32>().unwrap();
    let k_flat = k_rot.to_flat_vec::<f32>().unwrap();
    assert!(
        q_flat.iter().all(|v| v.is_finite()),
        "Q should be all finite"
    );
    assert!(
        k_flat.iter().all(|v| v.is_finite()),
        "K should be all finite"
    );
}

// ===========================================================================
// 13. Model forward with bias linear enabled
// ===========================================================================

/// Model with add_bias_linear=true loads and produces correct output shape.
#[test]
fn test_model_forward_with_all_biases_enabled() {
    let cfg = Glm5Config::new(
        128,      // hidden_size
        256,      // ffn_hidden_size
        1,        // num_layers
        2,        // num_attention_heads
        1,        // multi_query_group_num
        32,       // padded_vocab_size
        64,       // kv_channels
        1e-5,     // layernorm_epsilon
        32,       // seq_length
        true,     // rmsnorm
        true,     // add_qkv_bias
        true,     // add_bias_linear (both MLP and output have bias)
        10_000.0, // rope_theta
    );
    assert!(cfg.validate().is_ok());
    let model = load_model_with_config(cfg.clone());
    let logits = model.forward(&[0, 1, 2], &[0, 1, 2]).unwrap();
    assert_eq!(logits.dims(), &[1, 3, cfg.padded_vocab_size]);
    let flat = logits.to_flat_vec::<f32>().unwrap();
    assert!(
        flat.iter().all(|v| v.is_finite()),
        "all logits should be finite"
    );
}

/// Model with no biases at all (add_qkv_bias=false, add_bias_linear=false).
#[test]
fn test_model_forward_with_no_biases() {
    let cfg = Glm5Config::new(
        128,      // hidden_size
        256,      // ffn_hidden_size
        1,        // num_layers
        2,        // num_attention_heads
        2,        // multi_query_group_num (MHA)
        32,       // padded_vocab_size
        64,       // kv_channels
        1e-5,     // layernorm_epsilon
        32,       // seq_length
        true,     // rmsnorm
        false,    // add_qkv_bias
        false,    // add_bias_linear
        10_000.0, // rope_theta
    );
    assert!(cfg.validate().is_ok());
    let model = load_model_with_config(cfg.clone());
    let logits = model.forward(&[0, 1], &[0, 1]).unwrap();
    assert_eq!(logits.dims(), &[1, 2, cfg.padded_vocab_size]);
    let flat = logits.to_flat_vec::<f32>().unwrap();
    assert!(
        flat.iter().all(|v| v.is_finite()),
        "all logits should be finite"
    );
}

// ===========================================================================
// 14. Forward from embeddings error paths
// ===========================================================================

/// forward_from_embeddings rejects mismatched seq_len in hidden_states
/// vs positions length.
#[test]
fn test_forward_from_embeddings_rejects_seq_len_mismatch() {
    let model = load_tiny_model();
    let cfg = model.config().clone();

    // hidden_states has seq_len=3, but positions has 2 entries
    let emb = DynTensor::zeros(&[1, 3, cfg.hidden_size], DType::F32, &Device::Cpu).unwrap();
    let err = model
        .forward_from_embeddings(&emb, &[0, 1], None)
        .unwrap_err();
    let msg = err.to_string();
    assert!(
        msg.contains("seq_len") || msg.contains("positions"),
        "error should mention seq_len/positions mismatch: {msg}"
    );
}

/// forward_from_embeddings with dtype mismatch auto-converts (BF16 model, F32 input).
#[test]
fn test_forward_from_embeddings_dtype_autocast() {
    let cfg = tiny_config();
    let vb = VarBuilder::zeros(DType::BF16, &Device::Cpu);
    let model = Glm5Model::load(&vb, cfg.clone()).unwrap();
    assert_eq!(model.dtype(), DType::BF16);

    // Pass F32 embeddings to BF16 model -- should auto-convert
    let emb = DynTensor::zeros(&[1, 2, cfg.hidden_size], DType::F32, &Device::Cpu).unwrap();
    let logits = model.forward_from_embeddings(&emb, &[0, 1], None).unwrap();
    assert_eq!(logits.dims(), &[1, 2, cfg.padded_vocab_size]);
}

// ===========================================================================
// 15. GLM-4 vs GLM-5 config compatibility
// ===========================================================================

/// Both base and chat configs produce models with the same parameter count
/// relationships (same weight shapes, different RoPE/context parameters).
#[test]
fn test_base_and_chat_same_parameter_counts() {
    let base = Glm5Config::default();
    let chat = Glm5Config::glm4_9b_chat();

    // Embedding params: vocab * hidden
    let base_emb = base.padded_vocab_size * base.hidden_size;
    let chat_emb = chat.padded_vocab_size * chat.hidden_size;
    assert_eq!(
        base_emb, chat_emb,
        "base and chat should have same embedding parameter count"
    );

    // QKV params per layer
    let base_qkv = (base.num_attention_heads + 2 * base.multi_query_group_num)
        * base.kv_channels
        * base.hidden_size;
    let chat_qkv = (chat.num_attention_heads + 2 * chat.multi_query_group_num)
        * chat.kv_channels
        * chat.hidden_size;
    assert_eq!(
        base_qkv, chat_qkv,
        "base and chat should have same QKV parameter count per layer"
    );

    // MLP params per layer
    let base_mlp =
        base.ffn_hidden_size * 2 * base.hidden_size + base.hidden_size * base.ffn_hidden_size;
    let chat_mlp =
        chat.ffn_hidden_size * 2 * chat.hidden_size + chat.hidden_size * chat.ffn_hidden_size;
    assert_eq!(
        base_mlp, chat_mlp,
        "base and chat should have same MLP parameter count per layer"
    );
}

/// GLM-4-9B-chat's rope_theta is exactly 500x the base theta (rope_ratio=500).
#[test]
fn test_chat_rope_theta_is_500x_base() {
    let base = Glm5Config::default();
    let chat = Glm5Config::glm4_9b_chat();
    let ratio = chat.rope_theta / base.rope_theta;
    assert!(
        (ratio - 500.0).abs() < 1e-6,
        "chat theta / base theta should be 500 (rope_ratio), got {ratio}"
    );
}

/// GLM-4-9B-chat seq_length is exactly 16x the base (8192 * 16 = 131072).
#[test]
fn test_chat_seq_length_is_16x_base() {
    let base = Glm5Config::default();
    let chat = Glm5Config::glm4_9b_chat();
    assert_eq!(
        chat.seq_length,
        base.seq_length * 16,
        "chat seq_length should be 16x base: {} vs {} * 16",
        chat.seq_length,
        base.seq_length
    );
}

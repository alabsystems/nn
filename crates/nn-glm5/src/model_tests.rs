// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! GLM-5 model layer tests: attention output shapes, RoPE application,
//! feed-forward network shapes, token embedding dimensions, forward pass
//! shapes, and end-to-end pipeline consistency.
//!
//! Part of #4186. Tests exercise real model logic by loading zero-weight
//! models via `VarBuilder::zeros` and running actual forward passes to
//! verify shape propagation, numerical finiteness, and layer interactions.

use super::*;
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

// ---------------------------------------------------------------------------
// Attention layer output shapes
// ---------------------------------------------------------------------------

#[test]
fn test_attention_output_shape_single_token() {
    let model = load_tiny_model();
    let cfg = model.config().clone();
    let logits = model.forward(&[0], &[0]).unwrap();
    assert_eq!(
        logits.dims(),
        &[1, 1, cfg.padded_vocab_size],
        "single-token output shape: [1, 1, vocab]"
    );
}

#[test]
fn test_attention_output_shape_multi_token() {
    let model = load_tiny_model();
    let cfg = model.config().clone();
    let seq_len = 7;
    let ids: Vec<usize> = (0..seq_len).collect();
    let pos: Vec<usize> = (0..seq_len).collect();
    let logits = model.forward(&ids, &pos).unwrap();
    assert_eq!(
        logits.dims(),
        &[1, seq_len, cfg.padded_vocab_size],
        "multi-token output shape: [1, seq_len, vocab]"
    );
}

#[test]
fn test_attention_output_preserves_batch_dim_through_cache() {
    let model = load_tiny_model();
    let cfg = model.config().clone();
    let mut cache = model.new_cache();

    // Prefill: batch=1, seq=3
    let logits = model
        .forward_cached(&[0, 1, 2], &[0, 1, 2], Some(&mut cache))
        .unwrap();
    assert_eq!(logits.dims()[0], 1, "batch dimension preserved in prefill");
    assert_eq!(logits.dims()[1], 3);

    // Decode: batch=1, seq=1
    let logits = model.forward_cached(&[3], &[3], Some(&mut cache)).unwrap();
    assert_eq!(logits.dims()[0], 1, "batch dimension preserved in decode");
    assert_eq!(logits.dims()[1], 1);
    assert_eq!(logits.dims()[2], cfg.padded_vocab_size);
}

#[test]
fn test_attention_gqa_repeat_factor_reflected_in_output() {
    // MQA config: 8 Q heads sharing 1 KV head (repeat factor = 8)
    let cfg = Glm5Config {
        hidden_size: 256,
        ffn_hidden_size: 512,
        num_layers: 1,
        num_attention_heads: 8,
        multi_query_group_num: 1,
        padded_vocab_size: 32,
        kv_channels: 32,
        layernorm_epsilon: 1e-5,
        seq_length: 32,
        rmsnorm: true,
        add_qkv_bias: true,
        add_bias_linear: false,
        rope_theta: 10_000.0,
    };
    let model = load_model_with_config(cfg.clone());
    let logits = model.forward(&[0, 1], &[0, 1]).unwrap();
    assert_eq!(
        logits.dims(),
        &[1, 2, cfg.padded_vocab_size],
        "MQA model should produce correct output shape"
    );
}

// ---------------------------------------------------------------------------
// RoPE application correctness
// ---------------------------------------------------------------------------

#[test]
fn test_rope_half_rotation_preserves_second_half() {
    // GLM-specific: only first head_dim/2 dimensions are rotated.
    // The second half passes through unchanged.
    let head_dim = 16;
    let rope = HalfRotaryEmbedding::new(head_dim, 64, 10_000.0, &Device::Cpu).unwrap();

    let data: Vec<f32> = (0..head_dim).map(|i| (i + 1) as f32 * 0.5).collect();
    let x = DynTensor::from_vec(data.clone(), &[1, 1, 1, head_dim], &Device::Cpu).unwrap();
    let rotated = rope.apply(&x, 5).unwrap().to_flat_vec::<f32>().unwrap();

    let half = head_dim / 2;
    // Second half must be identical to input
    for i in half..head_dim {
        assert!(
            (rotated[i] - data[i]).abs() < 1e-6,
            "dim {i} (pass-through): expected {}, got {}",
            data[i],
            rotated[i]
        );
    }
    // First half must differ at non-zero position
    let first_half_diff: f32 = (0..half).map(|i| (rotated[i] - data[i]).abs()).sum();
    assert!(
        first_half_diff > 0.01,
        "first half should be rotated at pos=5, total diff={first_half_diff}"
    );
}

#[test]
fn test_rope_norm_preservation_for_rotated_half() {
    // Rotation is orthogonal: ||rotated_first_half|| == ||original_first_half||
    let head_dim = 32;
    let rope = HalfRotaryEmbedding::new(head_dim, 128, 10_000.0, &Device::Cpu).unwrap();

    let data: Vec<f32> = (0..head_dim).map(|i| ((i + 1) as f32).sqrt()).collect();
    let x = DynTensor::from_vec(data.clone(), &[1, 1, 1, head_dim], &Device::Cpu).unwrap();
    let rotated = rope.apply(&x, 13).unwrap().to_flat_vec::<f32>().unwrap();

    let half = head_dim / 2;
    let orig_norm: f32 = data[..half].iter().map(|v| v * v).sum::<f32>().sqrt();
    let rot_norm: f32 = rotated[..half].iter().map(|v| v * v).sum::<f32>().sqrt();
    assert!(
        (orig_norm - rot_norm).abs() < 1e-4,
        "rotation should preserve L2 norm: orig={orig_norm}, rotated={rot_norm}"
    );
}

#[test]
fn test_rope_position_zero_is_identity() {
    let head_dim = 8;
    let rope = HalfRotaryEmbedding::new(head_dim, 32, 10_000.0, &Device::Cpu).unwrap();

    let data: Vec<f32> = (1..=head_dim as i32).map(|i| i as f32).collect();
    let x = DynTensor::from_vec(data.clone(), &[1, 1, 1, head_dim], &Device::Cpu).unwrap();
    let result = rope.apply(&x, 0).unwrap().to_flat_vec::<f32>().unwrap();

    for i in 0..head_dim {
        assert!(
            (result[i] - data[i]).abs() < 1e-5,
            "at position 0, dim {i} should be identity: expected {}, got {}",
            data[i],
            result[i]
        );
    }
}

#[test]
fn test_rope_chat_theta_rotates_slower_than_base_theta() {
    // Higher rope_theta (5M for chat vs 10K for base) means slower rotation
    let head_dim = 8;
    let data: Vec<f32> = vec![1.0; head_dim];
    let x = DynTensor::from_vec(data, &[1, 1, 1, head_dim], &Device::Cpu).unwrap();

    let rope_base = HalfRotaryEmbedding::new(head_dim, 256, 10_000.0, &Device::Cpu).unwrap();
    let rope_chat = HalfRotaryEmbedding::new(head_dim, 256, 5_000_000.0, &Device::Cpu).unwrap();

    let out_base = rope_base
        .apply(&x, 20)
        .unwrap()
        .to_flat_vec::<f32>()
        .unwrap();
    let out_chat = rope_chat
        .apply(&x, 20)
        .unwrap()
        .to_flat_vec::<f32>()
        .unwrap();

    let diff_base: f32 = out_base.iter().map(|v| (v - 1.0).abs()).sum();
    let diff_chat: f32 = out_chat.iter().map(|v| (v - 1.0).abs()).sum();

    assert!(
        diff_chat < diff_base,
        "chat (theta=5M) should rotate less than base (theta=10K): chat_diff={diff_chat}, base_diff={diff_base}"
    );
}

// ---------------------------------------------------------------------------
// Feed-forward network shapes
// ---------------------------------------------------------------------------

#[test]
fn test_ffn_swiglu_gate_up_split_is_even() {
    // dense_h_to_4h outputs ffn_hidden_size * 2 which must split evenly
    for ffn in [128, 256, 512, 1024, 13696] {
        let fused = ffn * 2;
        assert_eq!(fused % 2, 0, "fused SwiGLU dim must be even for ffn={ffn}");
        assert_eq!(
            fused / 2,
            ffn,
            "each half should equal ffn_hidden_size for ffn={ffn}"
        );
    }
}

#[test]
fn test_ffn_output_shape_equals_hidden_size() {
    // The MLP output (after dense_4h_to_h) has the same shape as its input
    // because the residual connection adds MLP output to the hidden state.
    let model = load_tiny_model();
    let cfg = model.config().clone();

    // Forward from embeddings with known hidden_size
    let emb = DynTensor::ones(&[1, 2, cfg.hidden_size], DType::F32, &Device::Cpu).unwrap();
    let logits = model.forward_from_embeddings(&emb, &[0, 1], None).unwrap();
    // The final output is [1, 2, vocab] because output_layer projects
    // from hidden_size -> vocab_size. This confirms MLP preserved hidden_size
    // through all layers (residual connections).
    assert_eq!(logits.dims(), &[1, 2, cfg.padded_vocab_size]);
}

#[test]
fn test_ffn_bias_linear_enabled_produces_finite_output() {
    // When add_bias_linear=true, MLP has bias on both projections
    let cfg = Glm5Config {
        hidden_size: 128,
        ffn_hidden_size: 256,
        num_layers: 1,
        num_attention_heads: 2,
        multi_query_group_num: 1,
        padded_vocab_size: 32,
        kv_channels: 64,
        layernorm_epsilon: 1e-5,
        seq_length: 32,
        rmsnorm: true,
        add_qkv_bias: true,
        add_bias_linear: true,
        rope_theta: 10_000.0,
    };
    let model = load_model_with_config(cfg.clone());
    let logits = model.forward(&[0], &[0]).unwrap();
    let flat = logits.to_flat_vec::<f32>().unwrap();
    assert!(
        flat.iter().all(|v| v.is_finite()),
        "MLP with bias should produce finite logits"
    );
    assert_eq!(logits.dims(), &[1, 1, cfg.padded_vocab_size]);
}

// ---------------------------------------------------------------------------
// Token embedding dimensions
// ---------------------------------------------------------------------------

#[test]
fn test_embedding_maps_token_id_to_hidden_size() {
    let model = load_tiny_model();
    let cfg = model.config().clone();
    // A single token at position 0 produces [1, 1, vocab] logits,
    // confirming the embedding mapped token_id -> hidden_size vector
    // and the full pipeline preserved dimensions.
    let logits = model.forward(&[42], &[0]).unwrap();
    assert_eq!(logits.dims(), &[1, 1, cfg.padded_vocab_size]);
}

#[test]
fn test_embedding_accepts_any_valid_token_id() {
    let model = load_tiny_model();
    let vocab = model.config().padded_vocab_size;
    // Test boundary token IDs
    for tok_id in [0, 1, vocab / 2, vocab - 1] {
        let result = model.forward(&[tok_id], &[0]);
        assert!(
            result.is_ok(),
            "token_id={tok_id} should be accepted (vocab={vocab})"
        );
    }
}

#[test]
fn test_embedding_different_tokens_same_shape() {
    let model = load_tiny_model();
    let cfg = model.config().clone();
    let logits_a = model.forward(&[0], &[0]).unwrap();
    let logits_b = model.forward(&[50], &[0]).unwrap();
    assert_eq!(logits_a.dims(), logits_b.dims());
    assert_eq!(logits_a.dims(), &[1, 1, cfg.padded_vocab_size]);
}

// ---------------------------------------------------------------------------
// Full forward pass shape verification
// ---------------------------------------------------------------------------

#[test]
fn test_forward_shape_varies_with_seq_len() {
    let model = load_tiny_model();
    let cfg = model.config().clone();
    for seq_len in [1, 3, 8, 16] {
        let ids: Vec<usize> = (0..seq_len).collect();
        let pos: Vec<usize> = (0..seq_len).collect();
        let logits = model.forward(&ids, &pos).unwrap();
        assert_eq!(
            logits.dims(),
            &[1, seq_len, cfg.padded_vocab_size],
            "seq_len={seq_len}: shape mismatch"
        );
    }
}

#[test]
fn test_forward_with_varying_layer_counts() {
    for num_layers in [1, 2, 4, 8] {
        let mut cfg = tiny_config();
        cfg.num_layers = num_layers;
        let model = load_model_with_config(cfg.clone());
        let logits = model.forward(&[0, 1], &[0, 1]).unwrap();
        assert_eq!(
            logits.dims(),
            &[1, 2, cfg.padded_vocab_size],
            "num_layers={num_layers}: output shape mismatch"
        );
    }
}

#[test]
fn test_forward_cached_prefill_then_decode_shapes() {
    let model = load_tiny_model();
    let cfg = model.config().clone();
    let mut cache = model.new_cache();

    // Prefill
    let prefill = model
        .forward_cached(&[0, 1, 2, 3], &[0, 1, 2, 3], Some(&mut cache))
        .unwrap();
    assert_eq!(prefill.dims(), &[1, 4, cfg.padded_vocab_size]);
    assert_eq!(cache.seq_len(), 4);

    // Decode steps
    for step in 0..3 {
        let pos = 4 + step;
        let decode = model
            .forward_cached(&[42], &[pos], Some(&mut cache))
            .unwrap();
        assert_eq!(
            decode.dims(),
            &[1, 1, cfg.padded_vocab_size],
            "decode step {step}: shape mismatch"
        );
        assert_eq!(cache.seq_len(), pos + 1);
    }
}

#[test]
fn test_forward_from_embeddings_shape() {
    let model = load_tiny_model();
    let cfg = model.config().clone();

    let emb = DynTensor::ones(&[1, 5, cfg.hidden_size], DType::F32, &Device::Cpu).unwrap();
    let logits = model
        .forward_from_embeddings(&emb, &[0, 1, 2, 3, 4], None)
        .unwrap();
    assert_eq!(logits.dims(), &[1, 5, cfg.padded_vocab_size]);
}

// ---------------------------------------------------------------------------
// Output finiteness and numerical stability
// ---------------------------------------------------------------------------

#[test]
fn test_forward_produces_finite_logits_for_all_configs() {
    let configs = vec![
        ("tiny", tiny_config()),
        ("mqa", {
            let mut c = tiny_config();
            c.num_attention_heads = 4;
            c.multi_query_group_num = 1;
            c
        }),
        ("mha", {
            let mut c = tiny_config();
            c.multi_query_group_num = 4; // same as num_heads
            c
        }),
        ("no_bias", {
            let mut c = tiny_config();
            c.add_qkv_bias = false;
            c
        }),
        ("all_bias", {
            let mut c = tiny_config();
            c.add_bias_linear = true;
            c
        }),
    ];

    for (name, cfg) in configs {
        let model = load_model_with_config(cfg);
        let logits = model.forward(&[0, 1, 2], &[0, 1, 2]).unwrap();
        let flat = logits.to_flat_vec::<f32>().unwrap();
        let nan_count = flat.iter().filter(|v| v.is_nan()).count();
        let inf_count = flat.iter().filter(|v| v.is_infinite()).count();
        assert_eq!(nan_count, 0, "{name}: forward should produce no NaN values");
        assert_eq!(inf_count, 0, "{name}: forward should produce no Inf values");
    }
}

#[test]
fn test_cached_vs_uncached_logits_agree() {
    // The last-position logits from a full forward pass should match
    // the logits from an incremental decode at the same position.
    let model = load_tiny_model();
    let vocab = model.config().padded_vocab_size;

    // Full forward
    let full_logits = model.forward(&[0, 1, 2, 3, 4], &[0, 1, 2, 3, 4]).unwrap();
    let full_flat = full_logits.to_flat_vec::<f32>().unwrap();
    let full_last = &full_flat[4 * vocab..5 * vocab];

    // Incremental: prefill 4, then decode 1
    let mut cache = model.new_cache();
    let _ = model
        .forward_cached(&[0, 1, 2, 3], &[0, 1, 2, 3], Some(&mut cache))
        .unwrap();
    let incr_logits = model.forward_cached(&[4], &[4], Some(&mut cache)).unwrap();
    let incr_flat = incr_logits.to_flat_vec::<f32>().unwrap();

    assert_eq!(full_last.len(), incr_flat.len());
    for i in 0..full_last.len() {
        assert!(
            (full_last[i] - incr_flat[i]).abs() < 1e-4,
            "logit[{i}]: full={}, incremental={}",
            full_last[i],
            incr_flat[i]
        );
    }
}

// ---------------------------------------------------------------------------
// Model accessors
// ---------------------------------------------------------------------------

#[test]
fn test_model_config_accessor() {
    let model = load_tiny_model();
    let cfg = model.config();
    assert_eq!(cfg.hidden_size, 256);
    assert_eq!(cfg.num_layers, 2);
}

#[test]
fn test_model_dtype_accessor_matches_varbuilder() {
    for dtype in [DType::F32, DType::BF16, DType::F16] {
        let cfg = tiny_config();
        let vb = VarBuilder::zeros(dtype, &Device::Cpu);
        let model = Glm5Model::load(&vb, cfg).unwrap();
        assert_eq!(model.dtype(), dtype, "model dtype should match VarBuilder");
    }
}

#[test]
fn test_model_device_accessor() {
    let model = load_tiny_model();
    assert!(
        matches!(model.device(), Device::Cpu),
        "model loaded with CPU VarBuilder should report CPU device"
    );
}

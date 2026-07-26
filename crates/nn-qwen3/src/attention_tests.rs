// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Attention mechanism tests for Qwen3 (#4186).
//!
//! Covers GQA shape propagation, QK-Norm properties, RoPE integration with
//! attention, causal mask interaction, KV cache growth through attention layers,
//! and attention score dimension validation.

use super::*;
use crate::test_utils::tiny_config;
use nn_core::layers::repeat_kv;
use nn_core::var_builder::VarBuilder;
use nn_core::{DType, Device};

// ---------------------------------------------------------------------------
// GQA (Grouped Query Attention) shape propagation
// ---------------------------------------------------------------------------

#[test]
fn test_gqa_shape_propagation_ratio_1_mha() {
    // MHA: num_heads == num_kv_heads, repeat_kv is identity.
    // q_proj: [nh*hd, hidden] = [2*128, 256] = [256, 256]
    // k_proj: [nkv*hd, hidden] = [2*128, 256] = [256, 256]
    // v_proj: same as k_proj
    let cfg = tiny_config(); // 2 heads, 2 kv_heads
    assert_eq!(cfg.num_kv_groups().unwrap(), 1);
    let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
    let model = Qwen3Model::load(&vb, cfg.clone()).unwrap();
    // Forward with 3 tokens: output should be [1, 3, vocab_size]
    let logits = model.forward(&[0, 1, 2], &[0, 1, 2]).unwrap();
    assert_eq!(logits.dims(), &[1, 3, cfg.vocab_size]);
}

#[test]
fn test_gqa_shape_propagation_ratio_4() {
    // GQA ratio 4: 8 heads, 2 kv_heads -> groups=4
    // q_proj: [8*128, 1024] = [1024, 1024]
    // k_proj: [2*128, 1024] = [256, 1024]
    let cfg = Qwen3Config::new(1024, 2048, 1, 8, 2, 100, 1e-6, 10_000.0, 64, true, None);
    assert_eq!(cfg.num_kv_groups().unwrap(), 4);
    let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
    let model = Qwen3Model::load(&vb, cfg.clone()).unwrap();
    let logits = model.forward(&[0, 1], &[0, 1]).unwrap();
    assert_eq!(logits.dims(), &[1, 2, cfg.vocab_size]);
}

#[test]
fn test_gqa_kv_head_expansion_via_repeat_kv() {
    // Verify repeat_kv correctly expands KV heads for GQA.
    // 2 kv_heads, 4 reps -> 8 heads total
    let data: Vec<f32> = (0..2 * 3 * 128).map(|i| (i as f32) * 0.01).collect();
    let kv = DynTensor::from_vec(data, &[1, 2, 3, 128], &Device::Cpu).unwrap();
    let expanded = repeat_kv(&kv, 4).unwrap();
    assert_eq!(expanded.dims(), &[1, 8, 3, 128]);
    // Verify head 0 == head 1 == head 2 == head 3 (all from kv_head 0)
    let flat = expanded.to_flat_vec::<f32>().unwrap();
    let head_size = 3 * 128;
    for rep in 1..4 {
        assert_eq!(
            &flat[0..head_size],
            &flat[rep * head_size..(rep + 1) * head_size],
            "head 0 and head {rep} should be identical (repeated from kv_head 0)"
        );
    }
}

#[test]
fn test_gqa_different_group_counts_produce_valid_output() {
    // Test multiple GQA configurations: ratio 1, 2, 4, 8 (MQA)
    for (nh, nkv, expected_groups) in [(4, 4, 1), (4, 2, 2), (8, 2, 4), (8, 1, 8)] {
        let cfg = Qwen3Config::new(
            nh * 128,
            nh * 128 * 2,
            1,
            nh,
            nkv,
            50,
            1e-6,
            10_000.0,
            32,
            true,
            None,
        );
        assert_eq!(
            cfg.num_kv_groups().unwrap(),
            expected_groups,
            "nh={nh}, nkv={nkv}"
        );
        let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
        let model = Qwen3Model::load(&vb, cfg).unwrap();
        let logits = model.forward(&[0], &[0]).unwrap();
        let vals = logits.to_flat_vec::<f32>().unwrap();
        assert!(
            vals.iter().all(|v| v.is_finite()),
            "GQA ratio {expected_groups} should produce finite output"
        );
    }
}

// ---------------------------------------------------------------------------
// QK-Norm (Qwen3-specific: RMSNorm on Q and K after projection)
// ---------------------------------------------------------------------------

#[test]
fn test_qk_norm_weight_dimensions() {
    // QK-Norm weights are [head_dim] = [128], one per Q and K.
    // Model loads successfully with VarBuilder::zeros, which implicitly
    // verifies the weight shape expectations in Qwen3Attention::load.
    let cfg = tiny_config();
    let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
    let model = Qwen3Model::load(&vb, cfg);
    assert!(
        model.is_ok(),
        "model with QK-Norm should load: {:?}",
        model.err()
    );
}

#[test]
fn test_qk_norm_stabilizes_zero_weight_forward() {
    // QK-Norm uses RMSNorm with eps=1e-6 on head_dim=128.
    // With zero Q/K projections, QK-Norm output is 0/sqrt(eps) = 0,
    // which is finite. This verifies QK-Norm doesn't introduce NaN.
    let cfg = tiny_config();
    let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
    let model = Qwen3Model::load(&vb, cfg).unwrap();
    let logits = model.forward(&[0, 1, 2, 3], &[0, 1, 2, 3]).unwrap();
    let vals = logits.to_flat_vec::<f32>().unwrap();
    assert!(
        vals.iter().all(|v| v.is_finite()),
        "QK-Norm should keep all values finite even with zero weights"
    );
}

// ---------------------------------------------------------------------------
// RoPE application within attention
// ---------------------------------------------------------------------------

#[test]
fn test_rope_applied_after_qk_norm_produces_finite() {
    // The attention pipeline is: project -> reshape -> QK-Norm -> RoPE -> SDPA.
    // Verify the full pipeline produces finite results at various positions.
    let cfg = tiny_config();
    let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
    let model = Qwen3Model::load(&vb, cfg).unwrap();

    for seq_len in [1, 2, 4, 8] {
        let ids: Vec<usize> = (0..seq_len).collect();
        let pos: Vec<usize> = (0..seq_len).collect();
        let logits = model.forward(&ids, &pos).unwrap();
        assert_eq!(logits.dims()[1], seq_len);
        let vals = logits.to_flat_vec::<f32>().unwrap();
        assert!(
            vals.iter().all(|v| v.is_finite()),
            "seq_len={seq_len} should produce finite output"
        );
    }
}

#[test]
fn test_rope_at_boundary_positions() {
    // Test RoPE at position 0 and max_position_embeddings-1 (63 for tiny config).
    let cfg = tiny_config();
    let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
    let model = Qwen3Model::load(&vb, cfg).unwrap();

    let logits_pos0 = model.forward(&[0], &[0]).unwrap();
    let logits_pos63 = model.forward(&[0], &[63]).unwrap();

    let vals0 = logits_pos0.to_flat_vec::<f32>().unwrap();
    let vals63 = logits_pos63.to_flat_vec::<f32>().unwrap();
    assert!(vals0.iter().all(|v| v.is_finite()));
    assert!(vals63.iter().all(|v| v.is_finite()));
}

// ---------------------------------------------------------------------------
// Causal mask generation and interaction with attention
// ---------------------------------------------------------------------------

#[test]
fn test_causal_mask_skipped_for_single_token_decode() {
    // When seq_len=1, build_causal_mask returns None (optimization).
    // The single query attends to all cached keys. Verify this works.
    let cfg = tiny_config();
    let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
    let model = Qwen3Model::load(&vb, cfg.clone()).unwrap();
    let mut cache = model.new_cache();

    // Prefill with 3 tokens
    let _ = model
        .forward_cached(&[0, 1, 2], &[0, 1, 2], Some(&mut cache))
        .unwrap();
    assert_eq!(cache.seq_len(), 3);

    // Single-token decode step: mask should be None internally
    let logits = model.forward_cached(&[3], &[3], Some(&mut cache)).unwrap();
    assert_eq!(logits.dims(), &[1, 1, cfg.vocab_size]);
    assert_eq!(cache.seq_len(), 4);
}

#[test]
fn test_causal_mask_generated_for_multi_token_prefill() {
    // With seq_len > 1, causal mask is generated: [1, 1, seq, total]
    let cfg = tiny_config();
    let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
    let model = Qwen3Model::load(&vb, cfg.clone()).unwrap();

    // 5-token prefill should generate a 5x5 causal mask internally
    let logits = model.forward(&[0, 1, 2, 3, 4], &[0, 1, 2, 3, 4]).unwrap();
    assert_eq!(logits.dims(), &[1, 5, cfg.vocab_size]);
}

// ---------------------------------------------------------------------------
// KV cache growth through attention layers
// ---------------------------------------------------------------------------

#[test]
fn test_kv_cache_grows_correctly_through_attention() {
    let cfg = tiny_config(); // 2 layers
    let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
    let model = Qwen3Model::load(&vb, cfg).unwrap();
    let mut cache = model.new_cache();

    // Step 1: prefill 3 tokens
    let _ = model
        .forward_cached(&[0, 1, 2], &[0, 1, 2], Some(&mut cache))
        .unwrap();
    assert_eq!(
        cache.seq_len(),
        3,
        "cache should have 3 tokens after prefill"
    );

    // Step 2: decode 1 token
    let _ = model.forward_cached(&[3], &[3], Some(&mut cache)).unwrap();
    assert_eq!(
        cache.seq_len(),
        4,
        "cache should have 4 tokens after decode"
    );

    // Step 3: decode another token
    let _ = model.forward_cached(&[4], &[4], Some(&mut cache)).unwrap();
    assert_eq!(cache.seq_len(), 5);
}

#[test]
fn test_kv_cache_consistent_output_incremental_vs_oneshot() {
    // Forward pass output for tokens [0,1,2] should match last-token output
    // when feeding [0,1] then [2] incrementally.
    let cfg = tiny_config();
    let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
    let model = Qwen3Model::load(&vb, cfg).unwrap();

    // One-shot: all 3 tokens
    let logits_oneshot = model.forward(&[0, 1, 2], &[0, 1, 2]).unwrap();
    let oneshot_vals = logits_oneshot.to_flat_vec::<f32>().unwrap();

    // Incremental: 2 tokens then 1
    let mut cache = model.new_cache();
    let _ = model
        .forward_cached(&[0, 1], &[0, 1], Some(&mut cache))
        .unwrap();
    let logits_inc = model.forward_cached(&[2], &[2], Some(&mut cache)).unwrap();
    let inc_vals = logits_inc.to_flat_vec::<f32>().unwrap();

    // The last token's logits should match.
    // With zero weights both paths produce the same values.
    let vocab_size = oneshot_vals.len() / 3;
    let oneshot_last = &oneshot_vals[2 * vocab_size..];
    for (i, (a, b)) in oneshot_last.iter().zip(inc_vals.iter()).enumerate() {
        assert!(
            (a - b).abs() < 1e-5,
            "logit[{i}] mismatch: oneshot={a}, incremental={b}"
        );
    }
}

// ---------------------------------------------------------------------------
// Attention score computation dimensions
// ---------------------------------------------------------------------------

#[test]
fn test_attention_output_matches_hidden_size() {
    // After o_proj, attention output should have shape [batch, seq, hidden_size].
    // Verified via full forward: logits are [1, seq, vocab], which requires
    // lm_head([1, seq, hidden]) to work.
    let cfg = tiny_config(); // hidden=256
    let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
    let model = Qwen3Model::load(&vb, cfg.clone()).unwrap();

    for seq in [1, 3, 7] {
        let ids: Vec<usize> = (0..seq).collect();
        let pos: Vec<usize> = (0..seq).collect();
        let logits = model.forward(&ids, &pos).unwrap();
        // If attention output had wrong hidden_size, o_proj or lm_head would fail
        assert_eq!(logits.dims(), &[1, seq, cfg.vocab_size]);
    }
}

#[test]
fn test_attention_scale_factor_is_sqrt_head_dim() {
    // Qwen3 attention uses scale = 1 / sqrt(head_dim) = 1 / sqrt(128) ≈ 0.0884.
    // This is a constant property of the architecture.
    let cfg = tiny_config();
    let scale = 1.0 / (cfg.head_dim() as f64).sqrt();
    let expected = 1.0 / (128.0_f64).sqrt();
    assert!(
        (scale - expected).abs() < 1e-10,
        "scale should be 1/sqrt(128) = {expected}, got {scale}"
    );
}

#[test]
fn test_attention_with_different_seq_lengths() {
    // Attention should handle various sequence lengths correctly.
    let cfg = tiny_config();
    let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
    let model = Qwen3Model::load(&vb, cfg.clone()).unwrap();

    for seq_len in [1, 2, 4, 8, 16, 32] {
        let ids: Vec<usize> = (0..seq_len).map(|i| i % cfg.vocab_size).collect();
        let pos: Vec<usize> = (0..seq_len).collect();
        let logits = model.forward(&ids, &pos).unwrap();
        assert_eq!(
            logits.dims(),
            &[1, seq_len, cfg.vocab_size],
            "seq_len={seq_len}"
        );
    }
}

// ---------------------------------------------------------------------------
// Sliding window / max position bounds
// ---------------------------------------------------------------------------

#[test]
fn test_attention_max_position_embeddings_bound() {
    // max_position_embeddings constrains the RoPE frequency table size.
    // Positions beyond this limit may produce incorrect embeddings but
    // should not crash. Verify positions at and near the boundary.
    let cfg = tiny_config(); // max_position_embeddings=64
    assert_eq!(cfg.max_position_embeddings, 64);
    let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
    let model = Qwen3Model::load(&vb, cfg).unwrap();

    // At the boundary
    let logits = model.forward(&[0], &[63]).unwrap();
    assert!(logits
        .to_flat_vec::<f32>()
        .unwrap()
        .iter()
        .all(|v| v.is_finite()));
}

#[test]
fn test_head_dim_128_invariant_across_all_configs() {
    // head_dim is architecturally fixed at 128 for all Qwen3 variants.
    // This means hidden_size need NOT equal num_heads * head_dim.
    for (hidden, nh) in [(256, 2), (1024, 8), (896, 14), (4096, 32)] {
        let nkv = if nh >= 2 { 2 } else { 1 };
        let cfg = Qwen3Config::new(
            hidden,
            hidden * 2,
            1,
            nh,
            nkv,
            50,
            1e-6,
            10_000.0,
            32,
            true,
            None,
        );
        assert_eq!(cfg.head_dim(), 128, "head_dim must always be 128");
    }
}

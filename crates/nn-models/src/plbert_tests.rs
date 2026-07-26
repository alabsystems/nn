#![allow(deprecated)]
// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Tests for [`PlBert`] (ALBERT text encoder for Kokoro TTS).

use super::*;
use nn_core::{DType, Device, DynTensor, TensorError};
use std::collections::HashMap;

/// Helper: insert a uniform-value tensor into the weight map.
fn insert(tensors: &mut HashMap<String, DynTensor>, name: &str, shape: &[usize], val: f64) {
    let t = DynTensor::full(shape, val, DType::F32, &Device::Cpu).unwrap();
    tensors.insert(name.to_string(), t);
}

/// Helper: create a tensor with a deterministic varied pattern (not all same value).
fn varied_tensor(shape: &[usize], base: f32) -> DynTensor {
    let numel: usize = shape.iter().product();
    let data: Vec<f32> = (0..numel)
        .map(|i| base + 0.01 * (i as f32) - 0.005 * ((i % 3) as f32))
        .collect();
    DynTensor::from_vec(data, shape, &Device::Cpu).unwrap()
}

/// Test dimensions for PlBert.
const TEST_EMB_DIM: usize = 4;
const TEST_HIDDEN: usize = 8;
const TEST_INTERMEDIATE: usize = 16;
const TEST_VOCAB: usize = 10;
const TEST_MAX_POS: usize = 16;

/// Insert embedding weights (word, position, token_type, LayerNorm).
fn insert_embeddings(m: &mut HashMap<String, DynTensor>, uniform: bool) {
    if uniform {
        insert(
            m,
            "embeddings.word_embeddings.weight",
            &[TEST_VOCAB, TEST_EMB_DIM],
            0.1,
        );
        insert(
            m,
            "embeddings.position_embeddings.weight",
            &[TEST_MAX_POS, TEST_EMB_DIM],
            0.05,
        );
        insert(
            m,
            "embeddings.token_type_embeddings.weight",
            &[2, TEST_EMB_DIM],
            0.01,
        );
        insert(m, "embeddings.LayerNorm.weight", &[TEST_EMB_DIM], 1.0);
        insert(m, "embeddings.LayerNorm.bias", &[TEST_EMB_DIM], 0.0);
    } else {
        m.insert(
            "embeddings.word_embeddings.weight".into(),
            varied_tensor(&[TEST_VOCAB, TEST_EMB_DIM], 0.1),
        );
        m.insert(
            "embeddings.position_embeddings.weight".into(),
            varied_tensor(&[TEST_MAX_POS, TEST_EMB_DIM], 0.05),
        );
        m.insert(
            "embeddings.token_type_embeddings.weight".into(),
            varied_tensor(&[2, TEST_EMB_DIM], 0.01),
        );
        m.insert(
            "embeddings.LayerNorm.weight".into(),
            varied_tensor(&[TEST_EMB_DIM], 1.0),
        );
        m.insert(
            "embeddings.LayerNorm.bias".into(),
            varied_tensor(&[TEST_EMB_DIM], 0.0),
        );
    }
}

/// Insert factorized projection weights.
fn insert_projection(m: &mut HashMap<String, DynTensor>, uniform: bool) {
    if uniform {
        insert(
            m,
            "encoder.embedding_hidden_mapping_in.weight",
            &[TEST_HIDDEN, TEST_EMB_DIM],
            0.1,
        );
        insert(
            m,
            "encoder.embedding_hidden_mapping_in.bias",
            &[TEST_HIDDEN],
            0.0,
        );
    } else {
        m.insert(
            "encoder.embedding_hidden_mapping_in.weight".into(),
            varied_tensor(&[TEST_HIDDEN, TEST_EMB_DIM], 0.08),
        );
        m.insert(
            "encoder.embedding_hidden_mapping_in.bias".into(),
            varied_tensor(&[TEST_HIDDEN], 0.0),
        );
    }
}

const LAYER_PREFIX: &str = "encoder.albert_layer_groups.0.albert_layers.0";

/// Insert attention weights (Q, K, V, dense + LayerNorm) for the shared ALBERT layer.
fn insert_attention_weights(m: &mut HashMap<String, DynTensor>, uniform: bool) {
    let prefix = LAYER_PREFIX;
    let attn_names = [
        "attention.query",
        "attention.key",
        "attention.value",
        "attention.dense",
    ];
    let attn_bases = [0.05f32, 0.07, 0.09, 0.11];

    for (i, name) in attn_names.iter().enumerate() {
        if uniform {
            insert(
                m,
                &format!("{prefix}.{name}.weight"),
                &[TEST_HIDDEN, TEST_HIDDEN],
                0.1,
            );
            insert(m, &format!("{prefix}.{name}.bias"), &[TEST_HIDDEN], 0.0);
        } else {
            m.insert(
                format!("{prefix}.{name}.weight"),
                varied_tensor(&[TEST_HIDDEN, TEST_HIDDEN], attn_bases[i]),
            );
            m.insert(
                format!("{prefix}.{name}.bias"),
                varied_tensor(&[TEST_HIDDEN], 0.0),
            );
        }
    }

    if uniform {
        insert(
            m,
            &format!("{prefix}.attention.LayerNorm.weight"),
            &[TEST_HIDDEN],
            1.0,
        );
        insert(
            m,
            &format!("{prefix}.attention.LayerNorm.bias"),
            &[TEST_HIDDEN],
            0.0,
        );
    } else {
        m.insert(
            format!("{prefix}.attention.LayerNorm.weight"),
            varied_tensor(&[TEST_HIDDEN], 1.0),
        );
        m.insert(
            format!("{prefix}.attention.LayerNorm.bias"),
            varied_tensor(&[TEST_HIDDEN], 0.0),
        );
    }
}

/// Insert FFN weights and post-FFN LayerNorm for the shared ALBERT layer.
fn insert_ffn_weights(m: &mut HashMap<String, DynTensor>, uniform: bool) {
    let prefix = LAYER_PREFIX;
    if uniform {
        insert(
            m,
            &format!("{prefix}.ffn.weight"),
            &[TEST_INTERMEDIATE, TEST_HIDDEN],
            0.1,
        );
        insert(m, &format!("{prefix}.ffn.bias"), &[TEST_INTERMEDIATE], 0.0);
        insert(
            m,
            &format!("{prefix}.ffn_output.weight"),
            &[TEST_HIDDEN, TEST_INTERMEDIATE],
            0.1,
        );
        insert(m, &format!("{prefix}.ffn_output.bias"), &[TEST_HIDDEN], 0.0);
        insert(
            m,
            &format!("{prefix}.full_layer_layer_norm.weight"),
            &[TEST_HIDDEN],
            1.0,
        );
        insert(
            m,
            &format!("{prefix}.full_layer_layer_norm.bias"),
            &[TEST_HIDDEN],
            0.0,
        );
    } else {
        m.insert(
            format!("{prefix}.ffn.weight"),
            varied_tensor(&[TEST_INTERMEDIATE, TEST_HIDDEN], 0.06),
        );
        m.insert(
            format!("{prefix}.ffn.bias"),
            varied_tensor(&[TEST_INTERMEDIATE], 0.0),
        );
        m.insert(
            format!("{prefix}.ffn_output.weight"),
            varied_tensor(&[TEST_HIDDEN, TEST_INTERMEDIATE], 0.04),
        );
        m.insert(
            format!("{prefix}.ffn_output.bias"),
            varied_tensor(&[TEST_HIDDEN], 0.0),
        );
        m.insert(
            format!("{prefix}.full_layer_layer_norm.weight"),
            varied_tensor(&[TEST_HIDDEN], 1.0),
        );
        m.insert(
            format!("{prefix}.full_layer_layer_norm.bias"),
            varied_tensor(&[TEST_HIDDEN], 0.0),
        );
    }
}

/// Insert all shared ALBERT layer weights (attention + FFN + layer norms).
fn insert_albert_layer(m: &mut HashMap<String, DynTensor>, uniform: bool) {
    insert_attention_weights(m, uniform);
    insert_ffn_weights(m, uniform);
}

/// Build a minimal PlBert weight set for testing (uniform values).
fn make_plbert_weights() -> HashMap<String, DynTensor> {
    let mut m = HashMap::new();
    insert_embeddings(&mut m, true);
    insert_projection(&mut m, true);
    insert_albert_layer(&mut m, true);
    m
}

/// Build PlBert weights with varied (non-uniform) values.
///
/// Uniform weights combined with LayerNorm produce a fixed point after one iteration,
/// making the weight-sharing test meaningless. Use distinct values per weight tensor
/// to break the symmetry.
fn make_plbert_varied_weights() -> HashMap<String, DynTensor> {
    let mut m = HashMap::new();
    insert_embeddings(&mut m, false);
    insert_projection(&mut m, false);
    insert_albert_layer(&mut m, false);
    m
}

fn test_config() -> PlbertConfig {
    PlbertConfig {
        vocab_size: 10,
        embedding_dim: 4,
        hidden_size: 8,
        num_attention_heads: 2,
        intermediate_size: 16,
        max_position_embeddings: 16,
        num_hidden_layers: 2, // Use 2 layers for faster tests
        layer_norm_eps: 1e-12,
    }
}

#[test]
fn test_plbert_load() {
    let weights = make_plbert_weights();
    let vb = VarBuilder::from_tensors(weights, DType::F32, &Device::Cpu);
    let config = test_config();
    let plbert = PlBert::load(&vb, &config);
    assert!(plbert.is_ok(), "PlBert should load: {:?}", plbert.err());
}

#[test]
fn test_plbert_forward_shape() {
    let weights = make_plbert_weights();
    let vb = VarBuilder::from_tensors(weights, DType::F32, &Device::Cpu);
    let config = test_config();
    let plbert = PlBert::load(&vb, &config).unwrap();

    // Input: [B=1, T=5] token IDs as f32
    let input = DynTensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0], &[1, 5], &Device::Cpu).unwrap();
    let output = plbert.forward(&input).unwrap();

    // Output should be [1, 5, hidden_size=8]
    assert_eq!(output.dims(), &[1, 5, 8]);
}

#[test]
fn test_plbert_forward_batch() {
    let weights = make_plbert_weights();
    let vb = VarBuilder::from_tensors(weights, DType::F32, &Device::Cpu);
    let config = test_config();
    let plbert = PlBert::load(&vb, &config).unwrap();

    // Input: [B=2, T=3]
    let input =
        DynTensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], &Device::Cpu).unwrap();
    let output = plbert.forward(&input).unwrap();

    assert_eq!(output.dims(), &[2, 3, 8]);
}

#[test]
fn test_plbert_output_finiteness() {
    let weights = make_plbert_weights();
    let vb = VarBuilder::from_tensors(weights, DType::F32, &Device::Cpu);
    let config = test_config();
    let plbert = PlBert::load(&vb, &config).unwrap();

    let input = DynTensor::from_vec(vec![0.0, 1.0, 2.0, 3.0], &[1, 4], &Device::Cpu).unwrap();
    let output = plbert.forward(&input).unwrap();

    let vals = output.to_flat_vec::<f32>().unwrap();
    assert!(
        vals.iter().all(|v| v.is_finite()),
        "all PlBert outputs must be finite"
    );
}

#[test]
fn test_plbert_hidden_size() {
    let weights = make_plbert_weights();
    let vb = VarBuilder::from_tensors(weights, DType::F32, &Device::Cpu);
    let config = test_config();
    let plbert = PlBert::load(&vb, &config).unwrap();
    assert_eq!(plbert.hidden_size(), 8);
}

#[test]
fn test_plbert_invalid_input_rank() {
    let weights = make_plbert_weights();
    let vb = VarBuilder::from_tensors(weights, DType::F32, &Device::Cpu);
    let config = test_config();
    let plbert = PlBert::load(&vb, &config).unwrap();

    // 1D input should fail (needs 2D [B, T])
    let input = DynTensor::from_vec(vec![1.0, 2.0, 3.0], &[3], &Device::Cpu).unwrap();
    let result = plbert.forward(&input);
    assert!(result.is_err());
}

#[test]
fn test_plbert_rejects_seq_len_above_max_position_embeddings() {
    let weights = make_plbert_weights();
    let vb = VarBuilder::from_tensors(weights, DType::F32, &Device::Cpu);
    let config = test_config();
    let plbert = PlBert::load(&vb, &config).unwrap();

    let oversized: Vec<f32> = (0..=TEST_MAX_POS)
        .map(|i| (i % TEST_VOCAB) as f32)
        .collect();
    let input = DynTensor::from_vec(oversized, &[1, TEST_MAX_POS + 1], &Device::Cpu).unwrap();
    let err = plbert.forward(&input).unwrap_err();

    assert!(
        matches!(
            &err,
            TensorError::Unsupported(msg)
                if msg.contains("seq_len 17 exceeds max_position_embeddings 16")
        ),
        "expected explicit seq_len guard, got: {err:?}"
    );
}

#[test]
fn test_plbert_default_config() {
    let config = PlbertConfig::default();
    assert_eq!(config.vocab_size, 178);
    assert_eq!(config.embedding_dim, 128);
    assert_eq!(config.hidden_size, 768);
    assert_eq!(config.num_attention_heads, 12);
    assert_eq!(config.intermediate_size, 2048);
    assert_eq!(config.max_position_embeddings, 512);
    assert_eq!(config.num_hidden_layers, 12);
}

#[test]
fn test_plbert_weight_sharing() {
    // Verify that the shared layer is reused by running with different num_hidden_layers.
    // Same weights, different iteration counts → different outputs.
    // Use varied (non-uniform) weights so LayerNorm doesn't collapse to a fixed point.
    let weights = make_plbert_varied_weights();
    let vb1 = VarBuilder::from_tensors(weights.clone(), DType::F32, &Device::Cpu);
    let vb2 = VarBuilder::from_tensors(weights, DType::F32, &Device::Cpu);

    let config1 = PlbertConfig {
        num_hidden_layers: 1,
        ..test_config()
    };
    let config2 = PlbertConfig {
        num_hidden_layers: 4,
        ..test_config()
    };

    let plbert1 = PlBert::load(&vb1, &config1).unwrap();
    let plbert2 = PlBert::load(&vb2, &config2).unwrap();

    let input = DynTensor::from_vec(vec![1.0, 2.0, 3.0], &[1, 3], &Device::Cpu).unwrap();
    let out1 = plbert1.forward(&input).unwrap();
    let out2 = plbert2.forward(&input).unwrap();

    // Both should have same shape but different values
    assert_eq!(out1.dims(), out2.dims());
    let v1 = out1.to_flat_vec::<f32>().unwrap();
    let v2 = out2.to_flat_vec::<f32>().unwrap();
    let max_diff = v1
        .iter()
        .zip(v2.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert!(
        max_diff > 1e-6,
        "different num_hidden_layers should produce different outputs, max_diff={max_diff}"
    );
}

// -- PlBert::expand_vocab tests (#3460) -------------------------------------

#[test]
fn test_plbert_expand_vocab_noop() {
    let weights = make_plbert_weights();
    let vb = VarBuilder::from_tensors(weights, DType::F32, &Device::Cpu);
    let config = test_config();
    let mut plbert = PlBert::load(&vb, &config).unwrap();
    assert_eq!(plbert.vocab_size(), TEST_VOCAB);
    // Expanding to same size is a no-op
    plbert.expand_vocab(TEST_VOCAB).unwrap();
    assert_eq!(plbert.vocab_size(), TEST_VOCAB);
    // Expanding to smaller is also a no-op
    plbert.expand_vocab(5).unwrap();
    assert_eq!(plbert.vocab_size(), TEST_VOCAB);
}

#[test]
fn test_plbert_expand_vocab_grows() {
    let weights = make_plbert_weights();
    let vb = VarBuilder::from_tensors(weights, DType::F32, &Device::Cpu);
    let config = test_config();
    let mut plbert = PlBert::load(&vb, &config).unwrap();
    assert_eq!(plbert.vocab_size(), TEST_VOCAB);
    plbert.expand_vocab(TEST_VOCAB + 5).unwrap();
    assert_eq!(plbert.vocab_size(), TEST_VOCAB + 5);
    // Embedding weight shape should be [new_vocab, emb_dim]
    assert_eq!(
        plbert.word_embeddings().weight().dims(),
        &[TEST_VOCAB + 5, TEST_EMB_DIM]
    );
}

#[test]
fn test_plbert_expand_vocab_new_rows_are_mean_initialized() {
    let weights = make_plbert_varied_weights();
    let vb = VarBuilder::from_tensors(weights, DType::F32, &Device::Cpu);
    let config = test_config();
    let mut plbert = PlBert::load(&vb, &config).unwrap();

    // Compute expected mean of original embedding rows
    let orig_weight = plbert.word_embeddings().weight().clone();
    let expected_mean = orig_weight.mean(0).unwrap();
    let expected_vals = expected_mean.to_flat_vec::<f32>().unwrap();

    plbert.expand_vocab(TEST_VOCAB + 2).unwrap();
    let new_weight = plbert.word_embeddings().weight();

    // New rows (indices TEST_VOCAB and TEST_VOCAB+1) should be mean of original
    for row_idx in TEST_VOCAB..TEST_VOCAB + 2 {
        let row = new_weight.narrow(0, row_idx, 1).unwrap();
        let row_vals = row.to_flat_vec::<f32>().unwrap();
        for (j, (&actual, &expected)) in row_vals.iter().zip(expected_vals.iter()).enumerate() {
            assert!(
                (actual - expected).abs() < 1e-5,
                "row {row_idx} col {j}: {actual} != {expected}"
            );
        }
    }
}

#[test]
fn test_plbert_expand_vocab_preserves_original_rows() {
    let weights = make_plbert_varied_weights();
    let vb = VarBuilder::from_tensors(weights, DType::F32, &Device::Cpu);
    let config = test_config();
    let mut plbert = PlBert::load(&vb, &config).unwrap();

    let orig_vals = plbert
        .word_embeddings()
        .weight()
        .to_flat_vec::<f32>()
        .unwrap();
    plbert.expand_vocab(TEST_VOCAB + 3).unwrap();
    let new_vals = plbert
        .word_embeddings()
        .weight()
        .to_flat_vec::<f32>()
        .unwrap();

    // First TEST_VOCAB * TEST_EMB_DIM values should be unchanged
    let orig_count = TEST_VOCAB * TEST_EMB_DIM;
    for i in 0..orig_count {
        assert!(
            (new_vals[i] - orig_vals[i]).abs() < 1e-7,
            "original row data changed at index {i}"
        );
    }
}

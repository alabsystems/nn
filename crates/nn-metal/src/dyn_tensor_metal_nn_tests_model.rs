#![allow(deprecated)]
// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Model-level GPU forward tests — runs real DynTensor models on Metal GPU
//! and compares against CPU reference. Split from composite tests (#839).
//!
//! This validates the M3 critical path: DynTensor models can run end-to-end
//! on Metal GPU, which is required for dvoice candle→nn migration.

use nn_core::dyn_tensor::DynTensor;
use nn_core::var_builder::VarBuilder;
use nn_core::{DType, Device};
use nn_models::plbert::{PlBert, PlbertConfig};
use std::collections::HashMap;

use crate::test_common::{assert_close, init};

// -- PlBert weight helpers (split for function-size limit) --------------------

const TEST_EMB_DIM: usize = 4;
const TEST_HIDDEN: usize = 8;
const TEST_INTERMEDIATE: usize = 16;
const TEST_VOCAB: usize = 10;
const TEST_MAX_POS: usize = 16;
const LAYER_PREFIX: &str = "encoder.albert_layer_groups.0.albert_layers.0";

fn test_config() -> PlbertConfig {
    let mut cfg = PlbertConfig::default();
    cfg.vocab_size = TEST_VOCAB;
    cfg.embedding_dim = TEST_EMB_DIM;
    cfg.hidden_size = TEST_HIDDEN;
    cfg.num_attention_heads = 2;
    cfg.intermediate_size = TEST_INTERMEDIATE;
    cfg.max_position_embeddings = TEST_MAX_POS;
    cfg.num_hidden_layers = 2;
    cfg.layer_norm_eps = 1e-12;
    cfg
}

/// Deterministic varied-value tensor (not all same value — breaks LayerNorm symmetry).
fn v(shape: &[usize], base: f32) -> DynTensor {
    let n: usize = shape.iter().product();
    let data: Vec<f32> = (0..n)
        .map(|i| base + 0.01 * (i as f32) - 0.005 * ((i % 3) as f32))
        .collect();
    DynTensor::from_vec(data, shape, &Device::Cpu).unwrap()
}

/// Insert embedding + projection weights.
fn insert_embeddings(m: &mut HashMap<String, DynTensor>) {
    m.insert(
        "embeddings.word_embeddings.weight".into(),
        v(&[TEST_VOCAB, TEST_EMB_DIM], 0.1),
    );
    m.insert(
        "embeddings.position_embeddings.weight".into(),
        v(&[TEST_MAX_POS, TEST_EMB_DIM], 0.05),
    );
    m.insert(
        "embeddings.token_type_embeddings.weight".into(),
        v(&[2, TEST_EMB_DIM], 0.01),
    );
    m.insert(
        "embeddings.LayerNorm.weight".into(),
        v(&[TEST_EMB_DIM], 1.0),
    );
    m.insert("embeddings.LayerNorm.bias".into(), v(&[TEST_EMB_DIM], 0.0));
    m.insert(
        "encoder.embedding_hidden_mapping_in.weight".into(),
        v(&[TEST_HIDDEN, TEST_EMB_DIM], 0.08),
    );
    m.insert(
        "encoder.embedding_hidden_mapping_in.bias".into(),
        v(&[TEST_HIDDEN], 0.0),
    );
}

/// Insert attention (Q/K/V/dense + LayerNorm) weights for shared ALBERT layer.
fn insert_attention(m: &mut HashMap<String, DynTensor>) {
    for (name, base) in [
        ("query", 0.05f32),
        ("key", 0.07),
        ("value", 0.09),
        ("dense", 0.11),
    ] {
        m.insert(
            format!("{LAYER_PREFIX}.attention.{name}.weight"),
            v(&[TEST_HIDDEN, TEST_HIDDEN], base),
        );
        m.insert(
            format!("{LAYER_PREFIX}.attention.{name}.bias"),
            v(&[TEST_HIDDEN], 0.0),
        );
    }
    m.insert(
        format!("{LAYER_PREFIX}.attention.LayerNorm.weight"),
        v(&[TEST_HIDDEN], 1.0),
    );
    m.insert(
        format!("{LAYER_PREFIX}.attention.LayerNorm.bias"),
        v(&[TEST_HIDDEN], 0.0),
    );
}

/// Insert FFN (up/down + LayerNorm) weights for shared ALBERT layer.
fn insert_ffn(m: &mut HashMap<String, DynTensor>) {
    m.insert(
        format!("{LAYER_PREFIX}.ffn.weight"),
        v(&[TEST_INTERMEDIATE, TEST_HIDDEN], 0.06),
    );
    m.insert(
        format!("{LAYER_PREFIX}.ffn.bias"),
        v(&[TEST_INTERMEDIATE], 0.0),
    );
    m.insert(
        format!("{LAYER_PREFIX}.ffn_output.weight"),
        v(&[TEST_HIDDEN, TEST_INTERMEDIATE], 0.04),
    );
    m.insert(
        format!("{LAYER_PREFIX}.ffn_output.bias"),
        v(&[TEST_HIDDEN], 0.0),
    );
    m.insert(
        format!("{LAYER_PREFIX}.full_layer_layer_norm.weight"),
        v(&[TEST_HIDDEN], 1.0),
    );
    m.insert(
        format!("{LAYER_PREFIX}.full_layer_layer_norm.bias"),
        v(&[TEST_HIDDEN], 0.0),
    );
}

/// Build complete PlBert weight map with varied (non-uniform) values.
fn make_weights() -> HashMap<String, DynTensor> {
    let mut m = HashMap::new();
    insert_embeddings(&mut m);
    insert_attention(&mut m);
    insert_ffn(&mut m);
    m
}

// -- Tests --------------------------------------------------------------------

#[test]
fn test_plbert_forward_gpu_matches_cpu() {
    // Full PlBert (ALBERT) forward: Embedding → projection → 2x shared layer
    // (attention + GELU FFN + LayerNorm + residual). B=1, T=5.
    // GPU ops: matmul, broadcast_add/sub/mul/div, gelu, softmax, transpose,
    // reshape, mean_keepdim, sqr, sqrt, recip, mul_scalar, exp.
    init();
    let weights = make_weights();
    let cpu_vb = VarBuilder::from_tensors(weights.clone(), DType::F32, &Device::Cpu);
    let cpu_model = PlBert::load(&cpu_vb, &test_config()).unwrap();
    let cpu_input =
        DynTensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0], &[1, 5], &Device::Cpu).unwrap();
    let cpu_vals = cpu_model
        .forward(&cpu_input)
        .unwrap()
        .to_flat_vec::<f32>()
        .unwrap();

    let gpu_vb = VarBuilder::from_tensors(weights, DType::F32, &Device::metal());
    let gpu_model = PlBert::load(&gpu_vb, &test_config()).unwrap();
    let gpu_input =
        DynTensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0], &[1, 5], &Device::metal()).unwrap();
    let gpu_out = gpu_model.forward(&gpu_input).unwrap();

    assert_eq!(gpu_out.dims(), &[1, 5, TEST_HIDDEN]);
    let gpu_vals = gpu_out
        .to_device(&Device::Cpu)
        .unwrap()
        .to_flat_vec::<f32>()
        .unwrap();
    assert_close(&gpu_vals, &cpu_vals, 5e-3, "plbert_forward_gpu");
    assert!(
        gpu_vals.iter().all(|x| x.is_finite()),
        "GPU output must be finite"
    );
}

#[test]
fn test_plbert_forward_gpu_batch() {
    // Batched PlBert on GPU: [B=2, T=3].
    // #1134 fix: MSL matmul now broadcasts right tensor for unbatched weights.
    // Both batch items must match CPU within tolerance.
    init();
    let weights = make_weights();
    let cpu_vb = VarBuilder::from_tensors(weights.clone(), DType::F32, &Device::Cpu);
    let cpu_model = PlBert::load(&cpu_vb, &test_config()).unwrap();
    let cpu_in =
        DynTensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], &Device::Cpu).unwrap();
    let cpu_vals = cpu_model
        .forward(&cpu_in)
        .unwrap()
        .to_flat_vec::<f32>()
        .unwrap();

    let gpu_vb = VarBuilder::from_tensors(weights, DType::F32, &Device::metal());
    let gpu_model = PlBert::load(&gpu_vb, &test_config()).unwrap();
    let gpu_in = DynTensor::from_vec(
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        &[2, 3],
        &Device::metal(),
    )
    .unwrap();
    let gpu_out = gpu_model.forward(&gpu_in).unwrap();

    assert_eq!(gpu_out.dims(), &[2, 3, TEST_HIDDEN]);
    let gpu_vals = gpu_out
        .to_device(&Device::Cpu)
        .unwrap()
        .to_flat_vec::<f32>()
        .unwrap();
    assert!(
        gpu_vals.iter().all(|x| x.is_finite()),
        "batched output must be finite"
    );

    // Full output comparison — both batch items must match CPU
    assert_close(&gpu_vals, &cpu_vals, 5e-3, "plbert_batch_all_gpu");
}

#[test]
fn test_plbert_gpu_output_device() {
    // PlBert output must remain on Metal device.
    init();
    let gpu_vb = VarBuilder::from_tensors(make_weights(), DType::F32, &Device::metal());
    let gpu_model = PlBert::load(&gpu_vb, &test_config()).unwrap();
    let input = DynTensor::from_vec(vec![1.0, 2.0, 3.0], &[1, 3], &Device::metal()).unwrap();
    assert_eq!(gpu_model.forward(&input).unwrap().device(), Device::metal());
}

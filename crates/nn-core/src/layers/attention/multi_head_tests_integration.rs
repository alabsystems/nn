#![allow(deprecated)]
// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Integration tests for [`MultiHeadAttention`]: KV cache, RoPE, VarBuilder loading, bias.

use super::MultiHeadAttention;
use crate::dyn_tensor::test_helpers::{
    make_linear_seeded as make_linear, make_linear_seeded_with_bias as make_linear_with_bias,
};
use crate::dyn_tensor::DynTensor;
use crate::layers::{KvCacheLayer, RotaryEmbedding};
use crate::{Device, Result};

/// Helper: create a standard MHA (no GQA) with dim=64, 4 heads.
fn make_mha_4h_64d() -> MultiHeadAttention {
    let dim = 64;
    let num_heads = 4;
    MultiHeadAttention::new(
        make_linear(dim, dim, 1.0),
        make_linear(dim, dim, 2.0),
        make_linear(dim, dim, 3.0),
        make_linear(dim, dim, 4.0),
        num_heads,
        num_heads,
    )
    .expect("valid MHA")
}

/// Helper: create a random input tensor.
fn make_input(batch: usize, seq: usize, dim: usize, seed: f32) -> DynTensor {
    let n = batch * seq * dim;
    let data: Vec<f32> = (0..n)
        .map(|i| ((i as f32 + seed) * 0.017).sin() * 0.5)
        .collect();
    DynTensor::from_vec(data, &[batch, seq, dim], &Device::Cpu).unwrap()
}

// -- KV cache integration -----------------------------------------------------

#[test]
fn test_kv_cached_single_step() {
    let mha = make_mha_4h_64d();
    let x = make_input(1, 1, 64, 0.0);
    let mut cache = KvCacheLayer::empty();
    let out = mha
        .forward_kv_cached(&x, None, &mut cache, None, None, 0)
        .unwrap();
    assert_eq!(out.dims(), &[1, 1, 64]);
    assert_eq!(cache.seq_len(), 1);
}

#[test]
fn test_kv_cached_multi_step() {
    let mha = make_mha_4h_64d();
    let mut cache = KvCacheLayer::empty();

    // Step 1: prefill with 3 tokens
    let x1 = make_input(1, 3, 64, 0.0);
    let out1 = mha
        .forward_kv_cached(&x1, None, &mut cache, None, None, 0)
        .unwrap();
    assert_eq!(out1.dims(), &[1, 3, 64]);
    assert_eq!(cache.seq_len(), 3);

    // Step 2: decode 1 token
    let x2 = make_input(1, 1, 64, 1.0);
    let out2 = mha
        .forward_kv_cached(&x2, None, &mut cache, None, None, 0)
        .unwrap();
    assert_eq!(out2.dims(), &[1, 1, 64]);
    assert_eq!(cache.seq_len(), 4);

    // Step 3: decode another token
    let x3 = make_input(1, 1, 64, 2.0);
    let out3 = mha
        .forward_kv_cached(&x3, None, &mut cache, None, None, 0)
        .unwrap();
    assert_eq!(out3.dims(), &[1, 1, 64]);
    assert_eq!(cache.seq_len(), 5);

    // Outputs should be finite
    for out in [&out1, &out2, &out3] {
        let flat = out.to_flat_vec::<f32>().unwrap();
        assert!(flat.iter().all(|v| v.is_finite()));
    }
}

#[test]
fn test_kv_cached_gqa() {
    let dim = 64;
    let mha = MultiHeadAttention::new(
        make_linear(dim, dim, 1.0),
        make_linear(16, dim, 2.0), // 2 kv heads
        make_linear(16, dim, 3.0),
        make_linear(dim, dim, 4.0),
        8,
        2,
    )
    .unwrap();

    let mut cache = KvCacheLayer::empty();
    let x = make_input(1, 3, 64, 0.0);
    let out = mha
        .forward_kv_cached(&x, None, &mut cache, None, None, 0)
        .unwrap();
    assert_eq!(out.dims(), &[1, 3, 64]);
    assert_eq!(cache.seq_len(), 3);
}

// -- RoPE integration ---------------------------------------------------------

#[test]
fn test_self_attention_with_rope() {
    let head_dim = 16; // 64 / 4 heads
    let rope = RotaryEmbedding::new(head_dim, 128, 10000.0, &Device::Cpu).unwrap();
    let mha = make_mha_4h_64d();
    let x = make_input(1, 5, 64, 0.0);
    let out = mha.forward(&x, None, None, Some(&rope), 0).unwrap();
    assert_eq!(out.dims(), &[1, 5, 64]);
    let flat = out.to_flat_vec::<f32>().unwrap();
    assert!(flat.iter().all(|v| v.is_finite()));
}

#[test]
fn test_kv_cached_with_rope() {
    let head_dim = 16;
    let rope = RotaryEmbedding::new(head_dim, 128, 10000.0, &Device::Cpu).unwrap();
    let mha = make_mha_4h_64d();
    let mut cache = KvCacheLayer::empty();

    // Prefill
    let x1 = make_input(1, 3, 64, 0.0);
    let _out1 = mha
        .forward_kv_cached(&x1, None, &mut cache, None, Some(&rope), 0)
        .unwrap();
    assert_eq!(cache.seq_len(), 3);

    // Decode with offset
    let x2 = make_input(1, 1, 64, 1.0);
    let out2 = mha
        .forward_kv_cached(&x2, None, &mut cache, None, Some(&rope), 3)
        .unwrap();
    assert_eq!(out2.dims(), &[1, 1, 64]);
    assert_eq!(cache.seq_len(), 4);
}

// -- VarBuilder loading -------------------------------------------------------

#[test]
fn test_load_from_var_builder() -> Result<()> {
    use crate::var_builder::VarBuilder;
    use std::collections::HashMap;

    // Build a tensor map with the expected weight names
    let dim = 32usize;
    let num_heads = 4usize;
    let head_dim = dim / num_heads;
    let kv_dim = dim; // standard MHA (num_kv_heads == num_heads)

    let mut tensors: HashMap<String, DynTensor> = HashMap::new();
    let make_w = |rows: usize, cols: usize, seed: f32| -> DynTensor {
        let n = rows * cols;
        let data: Vec<f32> = (0..n)
            .map(|i| ((i as f32 + seed) * 0.01).sin() * 0.1)
            .collect();
        DynTensor::from_vec(data, &[rows, cols], &Device::Cpu).unwrap()
    };
    let make_b = |size: usize, seed: f32| -> DynTensor {
        let data: Vec<f32> = (0..size).map(|i| (i as f32 + seed) * 0.001).collect();
        DynTensor::from_vec(data, &[size], &Device::Cpu).unwrap()
    };

    tensors.insert("q_proj.weight".into(), make_w(dim, dim, 1.0));
    tensors.insert("q_proj.bias".into(), make_b(dim, 10.0));
    tensors.insert("k_proj.weight".into(), make_w(kv_dim, dim, 2.0));
    tensors.insert("k_proj.bias".into(), make_b(kv_dim, 20.0));
    tensors.insert("v_proj.weight".into(), make_w(kv_dim, dim, 3.0));
    tensors.insert("v_proj.bias".into(), make_b(kv_dim, 30.0));
    tensors.insert("out_proj.weight".into(), make_w(dim, dim, 4.0));
    tensors.insert("out_proj.bias".into(), make_b(dim, 40.0));

    let vb = VarBuilder::from_tensors(tensors, crate::DType::F32, &Device::Cpu);
    let mha = MultiHeadAttention::load(&vb, dim, num_heads, num_heads, true)?;
    assert_eq!(mha.num_heads(), num_heads);
    assert_eq!(mha.head_dim(), head_dim);

    // Verify forward works
    let x = make_input(1, 3, dim, 0.0);
    let out = mha.forward(&x, None, None, None, 0)?;
    assert_eq!(out.dims(), &[1, 3, dim]);
    Ok(())
}

#[test]
fn test_load_gqa_from_var_builder() -> Result<()> {
    use crate::var_builder::VarBuilder;
    use std::collections::HashMap;

    let dim = 64usize;
    let num_heads = 8usize;
    let num_kv_heads = 2usize;
    let head_dim = dim / num_heads; // 8
    let kv_dim = num_kv_heads * head_dim; // 16

    let mut tensors: HashMap<String, DynTensor> = HashMap::new();
    let make_w = |rows: usize, cols: usize, seed: f32| -> DynTensor {
        let n = rows * cols;
        let data: Vec<f32> = (0..n)
            .map(|i| ((i as f32 + seed) * 0.01).sin() * 0.1)
            .collect();
        DynTensor::from_vec(data, &[rows, cols], &Device::Cpu).unwrap()
    };

    tensors.insert("q_proj.weight".into(), make_w(dim, dim, 1.0));
    tensors.insert("k_proj.weight".into(), make_w(kv_dim, dim, 2.0));
    tensors.insert("v_proj.weight".into(), make_w(kv_dim, dim, 3.0));
    tensors.insert("out_proj.weight".into(), make_w(dim, dim, 4.0));

    let vb = VarBuilder::from_tensors(tensors, crate::DType::F32, &Device::Cpu);
    let mha = MultiHeadAttention::load(&vb, dim, num_heads, num_kv_heads, false)?;
    assert_eq!(mha.num_heads(), 8);
    assert_eq!(mha.num_kv_heads(), 2);
    assert_eq!(mha.head_dim(), 8);

    let x = make_input(1, 5, dim, 0.0);
    let out = mha.forward(&x, None, None, None, 0)?;
    assert_eq!(out.dims(), &[1, 5, dim]);
    Ok(())
}

// -- Bias vs no-bias ----------------------------------------------------------

#[test]
fn test_with_bias_output_differs() {
    let dim = 64;
    let num_heads = 4;
    let mha_no_bias = MultiHeadAttention::new(
        make_linear(dim, dim, 1.0),
        make_linear(dim, dim, 2.0),
        make_linear(dim, dim, 3.0),
        make_linear(dim, dim, 4.0),
        num_heads,
        num_heads,
    )
    .unwrap();
    let mha_bias = MultiHeadAttention::new(
        make_linear_with_bias(dim, dim, 1.0),
        make_linear_with_bias(dim, dim, 2.0),
        make_linear_with_bias(dim, dim, 3.0),
        make_linear_with_bias(dim, dim, 4.0),
        num_heads,
        num_heads,
    )
    .unwrap();

    let x = make_input(1, 3, 64, 0.0);
    let out_no = mha_no_bias.forward(&x, None, None, None, 0).unwrap();
    let out_yes = mha_bias.forward(&x, None, None, None, 0).unwrap();
    let v1 = out_no.to_flat_vec::<f32>().unwrap();
    let v2 = out_yes.to_flat_vec::<f32>().unwrap();
    // Outputs should differ due to bias
    let diff: f32 = v1.iter().zip(v2.iter()).map(|(a, b)| (a - b).abs()).sum();
    assert!(diff > 1e-6, "bias should change output, diff={diff}");
}

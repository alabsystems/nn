#![allow(deprecated)]
// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Qwen3 model GPU forward tests — runs a tiny Qwen3 decoder on Metal GPU
//! and compares against CPU reference output.
//!
//! Exercises the full Qwen3 stack on GPU: Embedding, RmsNorm (5 instances),
//! QK-Norm (per-head RmsNorm), RotaryEmbedding, GQA attention with causal
//! masking, SwiGLU MLP, and residual connections.
//!
//! Issue: #1287 AC2

use nn_core::dyn_tensor::DynTensor;
use nn_core::var_builder::VarBuilder;
use nn_core::{DType, Device};
use nn_qwen3::{Qwen3Config, Qwen3Model};
use std::collections::HashMap;

use crate::test_common::{assert_close, init};

// -- Test config --------------------------------------------------------------

/// Tiny Qwen3 config for fast GPU tests.
///
/// Based on nn-qwen3 test_utils::tiny_config() with 1 layer and small vocab.
fn tiny_config() -> Qwen3Config {
    nn_qwen3::test_utils::tiny_config()
        .with_num_hidden_layers(1)
        .with_vocab_size(32)
}

/// Deterministic varied-value tensor (non-uniform to break symmetry).
fn v(shape: &[usize], base: f32) -> DynTensor {
    let n: usize = shape.iter().product();
    let data: Vec<f32> = (0..n)
        .map(|i| base + 0.01 * (i as f32) - 0.005 * ((i % 3) as f32))
        .collect();
    DynTensor::from_vec(data, shape, &Device::Cpu).unwrap()
}

// -- Weight construction ------------------------------------------------------

/// Build full weight map for a 1-layer Qwen3 with tied embeddings.
///
/// Weight names follow HuggingFace convention:
///   model.embed_tokens.weight           [vocab, hidden]
///   model.layers.0.input_layernorm.weight      [hidden]
///   model.layers.0.self_attn.q_proj.weight     [nh*hd, hidden]
///   model.layers.0.self_attn.k_proj.weight     [nkv*hd, hidden]
///   model.layers.0.self_attn.v_proj.weight     [nkv*hd, hidden]
///   model.layers.0.self_attn.o_proj.weight     [hidden, nh*hd]
///   model.layers.0.self_attn.q_norm.weight     [hd]
///   model.layers.0.self_attn.k_norm.weight     [hd]
///   model.layers.0.post_attention_layernorm.weight [hidden]
///   model.layers.0.mlp.gate_proj.weight        [intermediate, hidden]
///   model.layers.0.mlp.up_proj.weight          [intermediate, hidden]
///   model.layers.0.mlp.down_proj.weight        [hidden, intermediate]
///   model.norm.weight                          [hidden]
fn make_weights(cfg: &Qwen3Config) -> HashMap<String, DynTensor> {
    let h = cfg.hidden_size;
    let i = cfg.intermediate_size;
    let hd = cfg.head_dim();
    let nh = cfg.num_attention_heads;
    let nkv = cfg.num_key_value_heads;
    let vocab = cfg.vocab_size;

    let mut m = HashMap::new();

    // Embedding (tied to lm_head)
    m.insert("model.embed_tokens.weight".into(), v(&[vocab, h], 0.1));

    // Layer 0 attention
    let prefix = "model.layers.0";
    m.insert(format!("{prefix}.input_layernorm.weight"), v(&[h], 1.0));
    m.insert(
        format!("{prefix}.self_attn.q_proj.weight"),
        v(&[nh * hd, h], 0.02),
    );
    m.insert(
        format!("{prefix}.self_attn.k_proj.weight"),
        v(&[nkv * hd, h], 0.03),
    );
    m.insert(
        format!("{prefix}.self_attn.v_proj.weight"),
        v(&[nkv * hd, h], 0.04),
    );
    m.insert(
        format!("{prefix}.self_attn.o_proj.weight"),
        v(&[h, nh * hd], 0.05),
    );
    m.insert(format!("{prefix}.self_attn.q_norm.weight"), v(&[hd], 1.0));
    m.insert(format!("{prefix}.self_attn.k_norm.weight"), v(&[hd], 1.0));

    // Layer 0 MLP
    m.insert(
        format!("{prefix}.post_attention_layernorm.weight"),
        v(&[h], 1.0),
    );
    m.insert(format!("{prefix}.mlp.gate_proj.weight"), v(&[i, h], 0.03));
    m.insert(format!("{prefix}.mlp.up_proj.weight"), v(&[i, h], 0.04));
    m.insert(format!("{prefix}.mlp.down_proj.weight"), v(&[h, i], 0.02));

    // Final norm
    m.insert("model.norm.weight".into(), v(&[h], 1.0));

    m
}

// -- Tests --------------------------------------------------------------------

#[test]
fn test_qwen3_forward_gpu_matches_cpu() {
    // Full Qwen3 forward: Embedding → 1 decoder layer (RmsNorm → GQA attention
    // with QK-Norm + RoPE + causal mask → residual → RmsNorm → SwiGLU → residual)
    // → final RmsNorm → lm_head projection.
    // B=1, T=4, vocab=32, hidden=256, heads=2, head_dim=128, ff=512.
    //
    // GPU ops exercised: matmul (8: Q/K/V/O proj, gate/up/down proj, lm_head),
    // broadcast_add/mul/div, silu, softmax, transpose, reshape, contiguous,
    // mean_keepdim, sqr, sqrt, recip, mul_scalar, exp, sin, cos (RoPE),
    // index_select (embedding), expand (GQA repeat_kv=1 identity).
    init();
    let cfg = tiny_config();
    let weights = make_weights(&cfg);

    let cpu_vb = VarBuilder::from_tensors(weights.clone(), DType::F32, &Device::Cpu);
    let cpu_model = Qwen3Model::load(&cpu_vb, cfg.clone()).unwrap();
    let input_ids = &[1, 5, 10, 20];
    let positions = &[0, 1, 2, 3];
    let cpu_out = cpu_model.forward(input_ids, positions).unwrap();
    let cpu_vals = cpu_out.to_flat_vec::<f32>().unwrap();

    let gpu_vb = VarBuilder::from_tensors(weights, DType::F32, &Device::metal());
    let gpu_model = Qwen3Model::load(&gpu_vb, cfg).unwrap();
    let gpu_out = gpu_model.forward(input_ids, positions).unwrap();

    assert_eq!(gpu_out.dims(), &[1, 4, 32]); // [batch, seq_len, vocab_size]
    assert_eq!(gpu_out.device(), Device::metal(), "output must stay on GPU");

    let gpu_vals = gpu_out
        .to_device(&Device::Cpu)
        .unwrap()
        .to_flat_vec::<f32>()
        .unwrap();
    assert!(
        gpu_vals.iter().all(|x| x.is_finite()),
        "GPU output must be finite"
    );
    // Wider tolerance: simdgroup_matrix GEMM accumulates in different order
    // than CPU sequential reduction, causing O(eps * K) drift on large
    // accumulated values (e.g., logit 47078 with diff 0.012). Per P1-165.
    assert_close(&gpu_vals, &cpu_vals, 5e-2, "qwen3_forward_gpu");
}

#[test]
fn test_qwen3_forward_gpu_single_token() {
    // Single token forward — exercises the seq_len=1 path (no causal mask needed).
    init();
    let cfg = tiny_config();
    let weights = make_weights(&cfg);

    let cpu_vb = VarBuilder::from_tensors(weights.clone(), DType::F32, &Device::Cpu);
    let cpu_model = Qwen3Model::load(&cpu_vb, cfg.clone()).unwrap();
    let cpu_out = cpu_model.forward(&[7], &[0]).unwrap();
    let cpu_vals = cpu_out.to_flat_vec::<f32>().unwrap();

    let gpu_vb = VarBuilder::from_tensors(weights, DType::F32, &Device::metal());
    let gpu_model = Qwen3Model::load(&gpu_vb, cfg).unwrap();
    let gpu_out = gpu_model.forward(&[7], &[0]).unwrap();

    assert_eq!(gpu_out.dims(), &[1, 1, 32]);
    assert_eq!(gpu_out.device(), Device::metal());

    let gpu_vals = gpu_out
        .to_device(&Device::Cpu)
        .unwrap()
        .to_flat_vec::<f32>()
        .unwrap();
    assert!(
        gpu_vals.iter().all(|x| x.is_finite()),
        "single-token GPU output must be finite"
    );
    assert_close(&gpu_vals, &cpu_vals, 1e-2, "qwen3_single_token_gpu");
}

#[test]
fn test_qwen3_forward_gpu_longer_sequence() {
    // Longer sequence (8 tokens) — exercises more RoPE positions and a larger
    // causal mask. Validates GPU numerics stay stable as sequence grows.
    init();
    let cfg = tiny_config();
    let weights = make_weights(&cfg);

    let input_ids: Vec<usize> = (0..8).collect();
    let positions: Vec<usize> = (0..8).collect();

    let cpu_vb = VarBuilder::from_tensors(weights.clone(), DType::F32, &Device::Cpu);
    let cpu_model = Qwen3Model::load(&cpu_vb, cfg.clone()).unwrap();
    let cpu_out = cpu_model.forward(&input_ids, &positions).unwrap();
    let cpu_vals = cpu_out.to_flat_vec::<f32>().unwrap();

    let gpu_vb = VarBuilder::from_tensors(weights, DType::F32, &Device::metal());
    let gpu_model = Qwen3Model::load(&gpu_vb, cfg).unwrap();
    let gpu_out = gpu_model.forward(&input_ids, &positions).unwrap();

    assert_eq!(gpu_out.dims(), &[1, 8, 32]);
    assert_eq!(gpu_out.device(), Device::metal());

    let gpu_vals = gpu_out
        .to_device(&Device::Cpu)
        .unwrap()
        .to_flat_vec::<f32>()
        .unwrap();
    assert!(
        gpu_vals.iter().all(|x| x.is_finite()),
        "longer-seq GPU output must be finite"
    );
    // Wider tolerance for longer sequences: simdgroup_matrix GEMM accumulates
    // in different order than CPU sequential reduction, causing O(eps * K) drift
    // on large accumulated values (e.g., 47078 with diff 0.012). Per P1-165.
    assert_close(&gpu_vals, &cpu_vals, 5e-2, "qwen3_longer_seq_gpu");
}

#[test]
fn test_qwen3_gpu_output_device() {
    // Output must remain on Metal device, not silently transfer to CPU.
    init();
    let cfg = tiny_config();
    let weights = make_weights(&cfg);
    let gpu_vb = VarBuilder::from_tensors(weights, DType::F32, &Device::metal());
    let gpu_model = Qwen3Model::load(&gpu_vb, cfg).unwrap();
    let out = gpu_model.forward(&[1, 2], &[0, 1]).unwrap();
    assert_eq!(out.device(), Device::metal());
}

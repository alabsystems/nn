// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! GLM-4/5 model GPU forward pass tests.
//!
//! Verifies the full GLM-4/5 decoder-only transformer runs on Metal GPU tensors.
//! Uses VarBuilder::zeros for deterministic weights and a tiny config
//! (1 layer, 4 heads, hidden=256) to keep test fast.
//!
//! CPU vs GPU comparison validates the DynTensor→GpuBackend→Metal path
//! for all GLM-4/5 components: Embedding, RmsNorm, Linear, HalfRoPE,
//! SwiGLU MLP, fused QKV, GQA attention, causal mask, and KV cache.

use super::test_utils::{assert_gpu_cpu_close, gpu_init};
use nn_core::dyn_tensor::DynTensor;
use nn_core::layers::kv_cache::KvCache;
use nn_core::{DType, Device, VarBuilder};
use nn_glm5::{Glm5Config, Glm5Model};

const TOL: f32 = 1e-3;

fn init() {
    gpu_init();
}

/// Minimal config for GPU tests: 1 layer keeps memory small,
/// 4 heads with 2 KV groups for GQA coverage.
fn tiny_gpu_config() -> Glm5Config {
    Glm5Config::new(
        256,      // hidden_size
        512,      // ffn_hidden_size
        1,        // num_layers (1 for fast GPU test)
        4,        // num_attention_heads
        2,        // multi_query_group_num
        100,      // padded_vocab_size
        64,       // kv_channels
        1e-5,     // layernorm_epsilon
        64,       // seq_length
        true,     // rmsnorm
        true,     // add_qkv_bias
        false,    // add_bias_linear
        10_000.0, // rope_theta
    )
}

fn assert_close(gpu_result: &DynTensor, cpu_result: &DynTensor, label: &str) {
    assert_gpu_cpu_close(gpu_result, cpu_result, TOL, label);
}

// -- GLM5 forward pass completes on GPU ----------------------------------------

#[test]
fn test_glm5_forward_gpu() {
    init();
    let cfg = tiny_gpu_config();
    let vb_gpu = VarBuilder::zeros(DType::F32, &Device::metal());
    let model_gpu = Glm5Model::load(&vb_gpu, cfg.clone()).expect("GPU model load");

    let input_ids = &[0, 1, 2];
    let positions = &[0, 1, 2];

    let result = model_gpu.forward(input_ids, positions);
    assert!(
        result.is_ok(),
        "GLM5 forward on GPU should succeed: {result:?}"
    );

    let logits = result.unwrap();
    assert_eq!(logits.rank(), 3, "logits should be [batch, seq, vocab]");
    assert_eq!(logits.dim(0).unwrap(), 1, "batch dim");
    assert_eq!(logits.dim(1).unwrap(), 3, "seq_len dim");
    assert_eq!(logits.dim(2).unwrap(), cfg.padded_vocab_size, "vocab dim");
}

// -- CPU vs GPU correctness comparison -----------------------------------------

#[test]
fn test_glm5_cpu_gpu_match() {
    init();
    let cfg = tiny_gpu_config();

    // CPU reference
    let vb_cpu = VarBuilder::zeros(DType::F32, &Device::Cpu);
    let model_cpu = Glm5Model::load(&vb_cpu, cfg.clone()).expect("CPU model load");
    let logits_cpu = model_cpu.forward(&[5, 10, 15], &[0, 1, 2]).unwrap();

    // GPU
    let vb_gpu = VarBuilder::zeros(DType::F32, &Device::metal());
    let model_gpu = Glm5Model::load(&vb_gpu, cfg).expect("GPU model load");
    let logits_gpu = model_gpu.forward(&[5, 10, 15], &[0, 1, 2]).unwrap();

    assert_close(&logits_gpu, &logits_cpu, "glm5_cpu_gpu");
}

// -- Single token forward on GPU -----------------------------------------------

#[test]
fn test_glm5_single_token_gpu() {
    init();
    let cfg = tiny_gpu_config();
    let vb = VarBuilder::zeros(DType::F32, &Device::metal());
    let model = Glm5Model::load(&vb, cfg.clone()).expect("GPU model load");

    let logits = model.forward(&[42], &[0]).unwrap();
    assert_eq!(logits.dims(), &[1, 1, cfg.padded_vocab_size]);
}

// -- Different sequence lengths ------------------------------------------------

#[test]
fn test_glm5_varying_seq_len_gpu() {
    init();
    let cfg = tiny_gpu_config();
    let vb = VarBuilder::zeros(DType::F32, &Device::metal());
    let model = Glm5Model::load(&vb, cfg.clone()).expect("GPU model load");

    for seq_len in [1, 2, 4, 8, 16] {
        let input_ids: Vec<usize> = (0..seq_len).collect();
        let positions: Vec<usize> = (0..seq_len).collect();

        let logits = model
            .forward(&input_ids, &positions)
            .unwrap_or_else(|e| panic!("GLM5 forward failed for seq_len={seq_len}: {e}"));
        assert_eq!(
            logits.dims(),
            &[1, seq_len, cfg.padded_vocab_size],
            "shape mismatch for seq_len={seq_len}"
        );
    }
}

// -- GLM5 with KV cache on GPU ------------------------------------------------

#[test]
fn test_glm5_kv_cache_gpu() {
    init();
    let cfg = tiny_gpu_config();
    let vb = VarBuilder::zeros(DType::F32, &Device::metal());
    let model = Glm5Model::load(&vb, cfg.clone()).expect("GPU model load");

    let mut cache = KvCache::new(cfg.num_layers);

    // Prefill: multi-token with cache.
    let logits1 = model
        .forward_cached(&[0, 1, 2], &[0, 1, 2], Some(&mut cache))
        .expect("prefill with KV cache on GPU should succeed");
    assert_eq!(logits1.rank(), 3);
    assert_eq!(logits1.dim(1).unwrap(), 3, "seq_len dim");
    assert_eq!(logits1.dim(2).unwrap(), cfg.padded_vocab_size, "vocab dim");

    // Decode step: single token, reuse cached KV.
    let logits2 = model
        .forward_cached(&[3], &[3], Some(&mut cache))
        .expect("decode step with cached KV on GPU should succeed");
    assert_eq!(logits2.dim(1).unwrap(), 1, "single token decode");
    assert_eq!(logits2.dim(2).unwrap(), cfg.padded_vocab_size, "vocab dim");
}

// -- GLM5 forward_from_embeddings on GPU ---------------------------------------

#[test]
fn test_glm5_from_embeddings_gpu() {
    init();
    let cfg = tiny_gpu_config();
    let vb = VarBuilder::zeros(DType::F32, &Device::metal());
    let model = Glm5Model::load(&vb, cfg.clone()).expect("GPU model load");

    // Pre-computed embeddings: [1, 2, hidden_size] on GPU
    let hidden = DynTensor::zeros(&[1, 2, cfg.hidden_size], DType::F32, &Device::metal())
        .expect("hidden states");

    let result = model.forward_from_embeddings(&hidden, &[0, 1], None);
    assert!(
        result.is_ok(),
        "forward_from_embeddings on GPU should succeed: {result:?}"
    );
    let logits = result.unwrap();
    assert_eq!(logits.dim(1).unwrap(), 2);
    assert_eq!(logits.dim(2).unwrap(), cfg.padded_vocab_size);
}

// -- CPU vs GPU parity with KV cache (autoregressive) --------------------------

#[test]
fn test_glm5_kv_cache_cpu_gpu_parity() {
    init();
    let cfg = tiny_gpu_config();

    // CPU path
    let vb_cpu = VarBuilder::zeros(DType::F32, &Device::Cpu);
    let model_cpu = Glm5Model::load(&vb_cpu, cfg.clone()).expect("CPU model load");
    let mut cache_cpu = KvCache::new(cfg.num_layers);
    let _cpu_0 = model_cpu
        .forward_cached(&[0], &[0], Some(&mut cache_cpu))
        .unwrap();
    let cpu_1 = model_cpu
        .forward_cached(&[1], &[1], Some(&mut cache_cpu))
        .unwrap();

    // GPU path
    let vb_gpu = VarBuilder::zeros(DType::F32, &Device::metal());
    let model_gpu = Glm5Model::load(&vb_gpu, cfg.clone()).expect("GPU model load");
    let mut cache_gpu = KvCache::new(cfg.num_layers);
    let _gpu_0 = model_gpu
        .forward_cached(&[0], &[0], Some(&mut cache_gpu))
        .unwrap();
    let gpu_1 = model_gpu
        .forward_cached(&[1], &[1], Some(&mut cache_gpu))
        .unwrap();

    assert_close(&gpu_1, &cpu_1, "glm5_kv_cache_step1_cpu_gpu");
}

// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Qwen3 model GPU forward pass tests (#1287).
//!
//! Verifies the full Qwen3 decoder-only transformer runs on Metal GPU tensors.
//! Uses VarBuilder::zeros for deterministic weights and a tiny config
//! (1 layer, 2 heads, hidden=256) to keep test fast.
//!
//! CPU vs GPU comparison validates the DynTensor→GpuBackend→Metal path
//! for all Qwen3 components: Embedding, RmsNorm, Linear, RoPE, SwiGLU MLP,
//! GQA attention, causal mask, and KV cache.

use super::test_utils::{assert_gpu_cpu_close, gpu_init};
use nn_core::dyn_tensor::DynTensor;
use nn_core::layers::kv_cache::KvCache;
use nn_core::{DType, Device, VarBuilder};
use nn_qwen3::{Qwen3Config, Qwen3Model};

const TOL: f32 = 1e-3;

fn init() {
    gpu_init();
}

/// Minimal config for GPU tests: 1 layer keeps memory small, 2 heads for GQA.
fn tiny_gpu_config() -> Qwen3Config {
    nn_qwen3::test_utils::tiny_config().with_num_hidden_layers(1)
}

fn assert_close(gpu_result: &DynTensor, cpu_result: &DynTensor, label: &str) {
    assert_gpu_cpu_close(gpu_result, cpu_result, TOL, label);
}

// -- AC2: Qwen3 forward pass completes on GPU ----------------------------------

#[test]
fn test_qwen3_forward_gpu() {
    init();
    let cfg = tiny_gpu_config();
    let vb_gpu = VarBuilder::zeros(DType::F32, &Device::metal());
    let model_gpu = Qwen3Model::load(&vb_gpu, cfg.clone()).expect("GPU model load");

    let input_ids = &[0, 1, 2];
    let positions = &[0, 1, 2];

    let result = model_gpu.forward(input_ids, positions);
    assert!(
        result.is_ok(),
        "Qwen3 forward on GPU should succeed: {result:?}"
    );

    let logits = result.unwrap();
    assert_eq!(logits.rank(), 3, "logits should be [batch, seq, vocab]");
    assert_eq!(logits.dim(0).unwrap(), 1, "batch dim");
    assert_eq!(logits.dim(1).unwrap(), 3, "seq_len dim");
    assert_eq!(logits.dim(2).unwrap(), cfg.vocab_size, "vocab dim");
}

// -- AC4: CPU vs GPU correctness comparison ------------------------------------

#[test]
fn test_qwen3_cpu_gpu_match() {
    init();
    let cfg = tiny_gpu_config();

    // CPU reference
    let vb_cpu = VarBuilder::zeros(DType::F32, &Device::Cpu);
    let model_cpu = Qwen3Model::load(&vb_cpu, cfg.clone()).expect("CPU model load");
    let logits_cpu = model_cpu.forward(&[5, 10, 15], &[0, 1, 2]).unwrap();

    // GPU
    let vb_gpu = VarBuilder::zeros(DType::F32, &Device::metal());
    let model_gpu = Qwen3Model::load(&vb_gpu, cfg).expect("GPU model load");
    let logits_gpu = model_gpu.forward(&[5, 10, 15], &[0, 1, 2]).unwrap();

    assert_close(&logits_gpu, &logits_cpu, "qwen3_cpu_gpu");
}

// -- Qwen3 with KV cache on GPU -----------------------------------------------

#[test]
fn test_qwen3_kv_cache_gpu() {
    init();
    let cfg = tiny_gpu_config();
    let vb = VarBuilder::zeros(DType::F32, &Device::metal());
    let model = Qwen3Model::load(&vb, cfg.clone()).expect("GPU model load");

    let mut cache = KvCache::new(cfg.num_hidden_layers);

    // Prefill: slice_set now supports GPU tensors via CPU round-trip (#1292).
    let logits1 = model
        .forward_cached(&[0, 1, 2], &[0, 1, 2], Some(&mut cache))
        .expect("prefill with KV cache on GPU should succeed");
    assert_eq!(logits1.rank(), 3);
    assert_eq!(logits1.dim(1).unwrap(), 3, "seq_len dim");
    assert_eq!(logits1.dim(2).unwrap(), cfg.vocab_size, "vocab dim");

    // Decode step: reuse cached KV.
    let logits2 = model
        .forward_cached(&[3], &[3], Some(&mut cache))
        .expect("decode step with cached KV on GPU should succeed");
    assert_eq!(logits2.dim(1).unwrap(), 1, "single token decode");
    assert_eq!(logits2.dim(2).unwrap(), cfg.vocab_size, "vocab dim");
}

// -- Qwen3 forward_from_embeddings on GPU --------------------------------------

#[test]
fn test_qwen3_from_embeddings_gpu() {
    init();
    let cfg = tiny_gpu_config();
    let vb = VarBuilder::zeros(DType::F32, &Device::metal());
    let model = Qwen3Model::load(&vb, cfg.clone()).expect("GPU model load");

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
    assert_eq!(logits.dim(2).unwrap(), cfg.vocab_size);
}

// -- Qwen3 forward_from_embeddings_with_hidden on GPU --------------------------

#[test]
fn test_qwen3_from_embeddings_with_hidden_gpu() {
    init();
    let cfg = tiny_gpu_config();
    let vb = VarBuilder::zeros(DType::F32, &Device::metal());
    let model = Qwen3Model::load(&vb, cfg.clone()).expect("GPU model load");

    let hidden = DynTensor::zeros(&[1, 3, cfg.hidden_size], DType::F32, &Device::metal())
        .expect("hidden states");

    let result = model.forward_from_embeddings_with_hidden(&hidden, &[0, 1, 2], None);
    assert!(
        result.is_ok(),
        "forward_from_embeddings_with_hidden on GPU should succeed: {result:?}"
    );
    let (logits, normed) = result.unwrap();
    assert_eq!(logits.dim(1).unwrap(), 3);
    assert_eq!(logits.dim(2).unwrap(), cfg.vocab_size);
    assert_eq!(normed.dim(2).unwrap(), cfg.hidden_size);
}

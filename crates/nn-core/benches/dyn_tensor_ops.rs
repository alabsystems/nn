// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Criterion benchmarks for core DynTensor operations at real-model tensor sizes.
//!
//! Run with: `cargo bench -p nn-core --bench dyn_tensor_ops`
//!
//! These benchmarks measure CPU-only performance of the fundamental tensor
//! operations used throughout nn model inference, using tensor shapes drawn
//! from real production models (Whisper, Qwen3, standard transformer blocks).
//!
//! Benchmark groups:
//! - **matmul_real_models**: Whisper encoder, Qwen3 attention, Kokoro FFN shapes
//! - **softmax_attention**: Attention-score softmax at multi-head sizes
//! - **layer_norm_transformer**: LayerNorm at standard transformer hidden sizes
//! - **gelu_ffn**: GELU activation at FFN intermediate sizes
//! - **embedding_lookup**: Embedding table lookups at Whisper/GPT vocab sizes
//! - **permute_attention**: Head-reshape permutations for multi-head attention
//! - **cat_kv**: KV-cache concatenation patterns
//! - **elementwise_activations**: relu/silu/sigmoid at transformer sizes

use std::hint::black_box;
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use nn_core::layers::{Embedding, LayerNorm, Module};
use nn_core::{DType, Device, DynTensor};
use rand::RngExt;

/// Generate a random f32 tensor on CPU with uniform [lo, hi) values.
fn rand_tensor(dims: &[usize], lo: f32, hi: f32) -> DynTensor {
    let mut rng = rand::rng();
    let numel: usize = dims.iter().product();
    let data: Vec<f32> = (0..numel)
        .map(|_| rng.random::<f32>() * (hi - lo) + lo)
        .collect();
    DynTensor::from_vec(data, dims, &Device::Cpu).unwrap()
}

// ---------------------------------------------------------------------------
// MatMul — real model sizes
// ---------------------------------------------------------------------------

fn bench_matmul_real_models(c: &mut Criterion) {
    let mut group = c.benchmark_group("matmul_real_models");

    // Whisper encoder: [1, 384, 512] x [512, 512]
    // Batched 3D matmul used in Whisper's multi-head self-attention projection.
    // FLOPs = 2 * 1 * 384 * 512 * 512 = 201,326,592
    let a_whisper = rand_tensor(&[1, 384, 512], -1.0, 1.0);
    let b_whisper = rand_tensor(&[512, 512], -1.0, 1.0);
    group.bench_with_input(
        BenchmarkId::new("whisper_encoder", "1x384x512_x_512x512"),
        &(),
        |bencher, ()| {
            bencher.iter(|| black_box(a_whisper.matmul(&b_whisper).unwrap()));
        },
    );

    // Qwen3 attention: [1, 1, 2048] x [2048, 2048]
    // Single-token decode step in Qwen3's Q/K/V projection.
    // FLOPs = 2 * 1 * 1 * 2048 * 2048 = 8,388,608
    let a_qwen3 = rand_tensor(&[1, 1, 2048], -1.0, 1.0);
    let b_qwen3 = rand_tensor(&[2048, 2048], -1.0, 1.0);
    group.bench_with_input(
        BenchmarkId::new("qwen3_attention", "1x1x2048_x_2048x2048"),
        &(),
        |bencher, ()| {
            bencher.iter(|| black_box(a_qwen3.matmul(&b_qwen3).unwrap()));
        },
    );

    // Kokoro FFN up-project: [1, 256, 768] x [768, 3072]
    // FLOPs = 2 * 1 * 256 * 768 * 3072 = 1,207,959,552
    let a_kokoro = rand_tensor(&[1, 256, 768], -1.0, 1.0);
    let b_kokoro = rand_tensor(&[768, 3072], -1.0, 1.0);
    group.bench_with_input(
        BenchmarkId::new("kokoro_ffn_up", "1x256x768_x_768x3072"),
        &(),
        |bencher, ()| {
            bencher.iter(|| black_box(a_kokoro.matmul(&b_kokoro).unwrap()));
        },
    );

    // Qwen3 prefill: [1, 128, 2048] x [2048, 2048]
    // 128-token prefill in Qwen3 attention.
    // FLOPs = 2 * 1 * 128 * 2048 * 2048 = 1,073,741,824
    let a_prefill = rand_tensor(&[1, 128, 2048], -1.0, 1.0);
    let b_prefill = rand_tensor(&[2048, 2048], -1.0, 1.0);
    group.bench_with_input(
        BenchmarkId::new("qwen3_prefill", "1x128x2048_x_2048x2048"),
        &(),
        |bencher, ()| {
            bencher.iter(|| black_box(a_prefill.matmul(&b_prefill).unwrap()));
        },
    );

    group.finish();
}

// ---------------------------------------------------------------------------
// Softmax — attention score sizes
// ---------------------------------------------------------------------------

fn bench_softmax_attention(c: &mut Criterion) {
    let mut group = c.benchmark_group("softmax_attention");

    // Standard multi-head attention scores: [1, 8, 128, 128]
    // 8 heads, sequence length 128. Softmax over last dim.
    let attn_8h = rand_tensor(&[1, 8, 128, 128], -5.0, 5.0);
    group.bench_with_input(
        BenchmarkId::new("8_heads_seq128", "1x8x128x128"),
        &(),
        |bencher, ()| {
            bencher.iter(|| black_box(attn_8h.softmax(nn_core::D::Minus1).unwrap()));
        },
    );

    // Whisper attention: [1, 6, 384, 384]
    // 6 heads, 384 encoder positions.
    let attn_whisper = rand_tensor(&[1, 6, 384, 384], -5.0, 5.0);
    group.bench_with_input(
        BenchmarkId::new("whisper_6h_seq384", "1x6x384x384"),
        &(),
        |bencher, ()| {
            bencher.iter(|| {
                black_box(attn_whisper.softmax(nn_core::D::Minus1).unwrap());
            });
        },
    );

    // Qwen3 GQA: [1, 32, 1, 128]
    // 32 heads, single decode token attends to 128 cached positions.
    let attn_qwen3 = rand_tensor(&[1, 32, 1, 128], -5.0, 5.0);
    group.bench_with_input(
        BenchmarkId::new("qwen3_gqa_32h_kv128", "1x32x1x128"),
        &(),
        |bencher, ()| {
            bencher.iter(|| {
                black_box(attn_qwen3.softmax(nn_core::D::Minus1).unwrap());
            });
        },
    );

    group.finish();
}

// ---------------------------------------------------------------------------
// LayerNorm — standard transformer hidden sizes
// ---------------------------------------------------------------------------

fn bench_layer_norm_transformer(c: &mut Criterion) {
    let mut group = c.benchmark_group("layer_norm_transformer");
    let dev = Device::Cpu;

    // Standard transformer: [1, 128, 512]
    // hidden_size=512, typical for smaller encoder models.
    let x_512 = rand_tensor(&[1, 128, 512], -1.0, 1.0);
    let w_512 = DynTensor::ones(&[512], DType::F32, &dev).unwrap();
    let b_512 = DynTensor::zeros(&[512], DType::F32, &dev).unwrap();
    let ln_512 = LayerNorm::new(w_512, b_512, 1e-5).unwrap();
    group.bench_with_input(
        BenchmarkId::new("hidden512_seq128", "1x128x512"),
        &(),
        |bencher, ()| {
            bencher.iter(|| black_box(ln_512.forward(&x_512).unwrap()));
        },
    );

    // Whisper: [1, 384, 512]
    // Whisper encoder hidden size 512, 384 positions (30s audio).
    let x_whisper = rand_tensor(&[1, 384, 512], -1.0, 1.0);
    group.bench_with_input(
        BenchmarkId::new("whisper_seq384", "1x384x512"),
        &(),
        |bencher, ()| {
            bencher.iter(|| black_box(ln_512.forward(&x_whisper).unwrap()));
        },
    );

    // Qwen3: [1, 128, 2048]
    // Qwen3 hidden_size=2048.
    let x_qwen3 = rand_tensor(&[1, 128, 2048], -1.0, 1.0);
    let w_2048 = DynTensor::ones(&[2048], DType::F32, &dev).unwrap();
    let b_2048 = DynTensor::zeros(&[2048], DType::F32, &dev).unwrap();
    let ln_2048 = LayerNorm::new(w_2048, b_2048, 1e-5).unwrap();
    group.bench_with_input(
        BenchmarkId::new("qwen3_hidden2048", "1x128x2048"),
        &(),
        |bencher, ()| {
            bencher.iter(|| black_box(ln_2048.forward(&x_qwen3).unwrap()));
        },
    );

    group.finish();
}

// ---------------------------------------------------------------------------
// GELU — FFN intermediate sizes
// ---------------------------------------------------------------------------

fn bench_gelu_ffn(c: &mut Criterion) {
    let mut group = c.benchmark_group("gelu_ffn");

    // Standard transformer FFN: [1, 128, 2048]
    // After up-projection to 4x hidden_size (512 -> 2048).
    let x_std = rand_tensor(&[1, 128, 2048], -3.0, 3.0);
    group.bench_with_input(
        BenchmarkId::new("ffn_hidden2048", "1x128x2048"),
        &(),
        |bencher, ()| {
            bencher.iter(|| black_box(x_std.gelu().unwrap()));
        },
    );

    // Whisper FFN: [1, 384, 2048]
    // Whisper intermediate_size=2048 (4x 512).
    let x_whisper = rand_tensor(&[1, 384, 2048], -3.0, 3.0);
    group.bench_with_input(
        BenchmarkId::new("whisper_ffn", "1x384x2048"),
        &(),
        |bencher, ()| {
            bencher.iter(|| black_box(x_whisper.gelu().unwrap()));
        },
    );

    // Qwen3 FFN: [1, 128, 5504]
    // Qwen3 intermediate_size=5504 (≈2.67x of 2048).
    let x_qwen3 = rand_tensor(&[1, 128, 5504], -3.0, 3.0);
    group.bench_with_input(
        BenchmarkId::new("qwen3_ffn", "1x128x5504"),
        &(),
        |bencher, ()| {
            bencher.iter(|| black_box(x_qwen3.gelu().unwrap()));
        },
    );

    group.finish();
}

// ---------------------------------------------------------------------------
// Embedding lookup — real vocab sizes
// ---------------------------------------------------------------------------

fn bench_embedding_lookup(c: &mut Criterion) {
    let mut group = c.benchmark_group("embedding_lookup");

    // Whisper: vocab_size=51865, embed_dim=512
    let w_whisper = rand_tensor(&[51865, 512], -0.02, 0.02);
    let emb_whisper = Embedding::new(w_whisper).unwrap();
    // Simulate 384 encoder positions (30s audio mel frames)
    let ids_whisper: Vec<u32> = (0..384).map(|i| (i * 137) % 51865).collect();
    let ids_whisper_t = DynTensor::from_vec_u32(ids_whisper, &[1, 384], &Device::Cpu).unwrap();
    group.bench_with_input(
        BenchmarkId::new("whisper_51865x512", "batch1_seq384"),
        &(),
        |bencher, ()| {
            bencher.iter(|| black_box(emb_whisper.forward(&ids_whisper_t).unwrap()));
        },
    );

    // Qwen3: vocab_size=151936, embed_dim=2048
    let w_qwen3 = rand_tensor(&[151936, 2048], -0.02, 0.02);
    let emb_qwen3 = Embedding::new(w_qwen3).unwrap();
    // 128-token prompt
    let ids_qwen3: Vec<u32> = (0..128).map(|i| (i * 1171) % 151936).collect();
    let ids_qwen3_t = DynTensor::from_vec_u32(ids_qwen3, &[1, 128], &Device::Cpu).unwrap();
    group.bench_with_input(
        BenchmarkId::new("qwen3_151936x2048", "batch1_seq128"),
        &(),
        |bencher, ()| {
            bencher.iter(|| black_box(emb_qwen3.forward(&ids_qwen3_t).unwrap()));
        },
    );

    // GPT-2 style: vocab_size=50257, embed_dim=768
    let w_gpt2 = rand_tensor(&[50257, 768], -0.02, 0.02);
    let emb_gpt2 = Embedding::new(w_gpt2).unwrap();
    let ids_gpt2: Vec<u32> = (0..32).map(|i| (i * 1571) % 50257).collect();
    let ids_gpt2_t = DynTensor::from_vec_u32(ids_gpt2, &[1, 32], &Device::Cpu).unwrap();
    group.bench_with_input(
        BenchmarkId::new("gpt2_50257x768", "batch1_seq32"),
        &(),
        |bencher, ()| {
            bencher.iter(|| black_box(emb_gpt2.forward(&ids_gpt2_t).unwrap()));
        },
    );

    group.finish();
}

// ---------------------------------------------------------------------------
// Permute — attention head reshaping
// ---------------------------------------------------------------------------

fn bench_permute_attention(c: &mut Criterion) {
    let mut group = c.benchmark_group("permute_attention");

    // Whisper: [1, 384, 6, 64] -> [1, 6, 384, 64] (head reshape for 6-head attn)
    let x_whisper = rand_tensor(&[1, 384, 6, 64], -1.0, 1.0);
    group.bench_with_input(
        BenchmarkId::new("whisper_6head", "1x384x6x64"),
        &(),
        |bencher, ()| {
            bencher.iter(|| black_box(x_whisper.permute([0, 2, 1, 3]).unwrap()));
        },
    );

    // Qwen3 GQA: [1, 128, 32, 64] -> [1, 32, 128, 64]
    let x_qwen3 = rand_tensor(&[1, 128, 32, 64], -1.0, 1.0);
    group.bench_with_input(
        BenchmarkId::new("qwen3_32head", "1x128x32x64"),
        &(),
        |bencher, ()| {
            bencher.iter(|| black_box(x_qwen3.permute([0, 2, 1, 3]).unwrap()));
        },
    );

    // Transpose for matmul: [512, 2048] -> [2048, 512]
    let w = rand_tensor(&[512, 2048], -1.0, 1.0);
    group.bench_with_input(
        BenchmarkId::new("weight_transpose", "512x2048"),
        &(),
        |bencher, ()| {
            bencher.iter(|| black_box(w.t().unwrap()));
        },
    );

    group.finish();
}

// ---------------------------------------------------------------------------
// Cat — KV cache concatenation
// ---------------------------------------------------------------------------

fn bench_cat_kv(c: &mut Criterion) {
    let mut group = c.benchmark_group("cat_kv_cache");

    // KV cache append: cat([1, 8, 127, 64], [1, 8, 1, 64]) along dim=2
    // Appending one new token to existing cache of 127 tokens.
    let cache = rand_tensor(&[1, 8, 127, 64], -1.0, 1.0);
    let new_kv = rand_tensor(&[1, 8, 1, 64], -1.0, 1.0);
    group.bench_with_input(
        BenchmarkId::new("append_token_8h", "127+1_d64"),
        &(),
        |bencher, ()| {
            bencher.iter(|| {
                black_box(DynTensor::cat(&[cache.clone(), new_kv.clone()], 2).unwrap());
            });
        },
    );

    // Concatenating 4 attention head outputs: [1, 128, 64] x 4 along dim=2
    let heads: Vec<DynTensor> = (0..4)
        .map(|_| rand_tensor(&[1, 128, 64], -1.0, 1.0))
        .collect();
    group.bench_with_input(
        BenchmarkId::new("concat_4_heads", "4x_1x128x64_dim2"),
        &(),
        |bencher, ()| {
            bencher.iter(|| {
                black_box(DynTensor::cat(&heads, 2).unwrap());
            });
        },
    );

    group.finish();
}

// ---------------------------------------------------------------------------
// Element-wise activations at real model sizes
// ---------------------------------------------------------------------------

fn bench_elementwise_activations(c: &mut Criterion) {
    let mut group = c.benchmark_group("activations");

    // Transformer hidden state: [1, 128, 2048]
    let x = rand_tensor(&[1, 128, 2048], -3.0, 3.0);

    group.bench_with_input(
        BenchmarkId::new("relu", "1x128x2048"),
        &(),
        |bencher, ()| {
            bencher.iter(|| black_box(x.relu().unwrap()));
        },
    );

    group.bench_with_input(
        BenchmarkId::new("silu", "1x128x2048"),
        &(),
        |bencher, ()| {
            bencher.iter(|| black_box(x.silu().unwrap()));
        },
    );

    group.bench_with_input(
        BenchmarkId::new("gelu", "1x128x2048"),
        &(),
        |bencher, ()| {
            bencher.iter(|| black_box(x.gelu().unwrap()));
        },
    );

    group.bench_with_input(
        BenchmarkId::new("sigmoid", "1x128x2048"),
        &(),
        |bencher, ()| {
            bencher.iter(|| black_box(x.sigmoid().unwrap()));
        },
    );

    // Larger FFN intermediate: [1, 384, 2048] (Whisper FFN size)
    let x_large = rand_tensor(&[1, 384, 2048], -3.0, 3.0);
    group.bench_with_input(
        BenchmarkId::new("gelu_whisper_ffn", "1x384x2048"),
        &(),
        |bencher, ()| {
            bencher.iter(|| black_box(x_large.gelu().unwrap()));
        },
    );

    group.finish();
}

// ---------------------------------------------------------------------------
// Criterion harness
// ---------------------------------------------------------------------------

criterion_group!(
    benches,
    bench_matmul_real_models,
    bench_softmax_attention,
    bench_layer_norm_transformer,
    bench_gelu_ffn,
    bench_embedding_lookup,
    bench_permute_attention,
    bench_cat_kv,
    bench_elementwise_activations,
);
criterion_main!(benches);

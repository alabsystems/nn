// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Extended Criterion benchmarks for DynTensor operations targeting RTF and
//! throughput measurement.
//!
//! Complements `dyn_tensor_ops.rs` with:
//! - MatMul at multiple sizes (256x256, 512x512, 1024x1024, batched)
//! - Softmax over batch of sequences
//! - LayerNorm at typical hidden dims (512, 768, 1024)
//! - Conv1d at audio-sized inputs (Kokoro/Demucs shapes)
//! - Embedding lookup at vocabulary-sized tables (50k, 178)
//!
//! No model weights required -- all synthetic data.
//!
//! Run: `cargo bench -p nn-core --bench dyn_tensor_ops_extended`

use std::hint::black_box;
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use nn_core::layers::{Conv1d, Conv1dConfig, Embedding, LayerNorm, Module};
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
// MatMul -- various sizes for throughput measurement
// ---------------------------------------------------------------------------

fn bench_matmul_sizes(c: &mut Criterion) {
    let mut group = c.benchmark_group("matmul_square");

    for &size in &[256usize, 512, 1024] {
        let a = rand_tensor(&[size, size], -1.0, 1.0);
        let b = rand_tensor(&[size, size], -1.0, 1.0);

        group.bench_with_input(
            BenchmarkId::new("cpu", format!("{size}x{size}")),
            &size,
            |bencher, _| {
                bencher.iter(|| black_box(a.matmul(&b).unwrap()));
            },
        );
    }

    group.finish();
}

fn bench_matmul_rectangular(c: &mut Criterion) {
    let mut group = c.benchmark_group("matmul_rect");

    // Transformer FFN shapes: up-project and down-project
    let shapes: &[(&str, usize, usize, usize)] = &[
        ("ffn_up_256x768x3072", 256, 768, 3072),
        ("ffn_down_256x3072x768", 256, 3072, 768),
        ("attn_qkv_128x512x512", 128, 512, 512),
        ("small_32x64x32", 32, 64, 32),
    ];

    for &(label, m, k, n) in shapes {
        let a = rand_tensor(&[m, k], -1.0, 1.0);
        let b = rand_tensor(&[k, n], -1.0, 1.0);

        group.bench_with_input(BenchmarkId::new("cpu", label), &(), |bencher, ()| {
            bencher.iter(|| black_box(a.matmul(&b).unwrap()));
        });
    }

    group.finish();
}

fn bench_matmul_batched(c: &mut Criterion) {
    let mut group = c.benchmark_group("matmul_batched");

    // Batched attention: [B, H, T, D] x [B, H, D, T]
    let a = rand_tensor(&[4, 8, 128, 64], -1.0, 1.0);
    let b = rand_tensor(&[4, 8, 64, 128], -1.0, 1.0);

    group.bench_function("4x8x128x64_x_4x8x64x128", |bencher| {
        bencher.iter(|| black_box(a.matmul(&b).unwrap()));
    });

    group.finish();
}

// ---------------------------------------------------------------------------
// Softmax -- batch of sequences
// ---------------------------------------------------------------------------

fn bench_softmax_batch(c: &mut Criterion) {
    let mut group = c.benchmark_group("softmax");

    // Attention scores: [B, H, T, T]
    let shapes: &[(&str, &[usize])] = &[
        ("attn_4x8x128x128", &[4, 8, 128, 128]),
        ("attn_1x12x256x256", &[1, 12, 256, 256]),
        ("logits_32x50257", &[32, 50257]),
    ];

    for &(label, shape) in shapes {
        let x = rand_tensor(shape, -5.0, 5.0);
        group.bench_with_input(BenchmarkId::new("dim_neg1", label), &(), |bencher, ()| {
            bencher.iter(|| black_box(x.softmax(nn_core::D::Minus1).unwrap()));
        });
    }

    group.finish();
}

// ---------------------------------------------------------------------------
// LayerNorm -- typical hidden dims
// ---------------------------------------------------------------------------

fn bench_layer_norm_sizes(c: &mut Criterion) {
    let mut group = c.benchmark_group("layer_norm");
    let dev = Device::Cpu;

    let configs: &[(&str, usize, usize, usize)] = &[
        ("kokoro_1x64x512", 1, 64, 512),
        ("whisper_1x1500x384", 1, 1500, 384),
        ("bert_32x128x768", 32, 128, 768),
        ("large_8x256x1024", 8, 256, 1024),
    ];

    for &(label, batch, seq, hidden) in configs {
        let x = rand_tensor(&[batch, seq, hidden], -1.0, 1.0);
        let weight = DynTensor::ones(&[hidden], DType::F32, &dev).unwrap();
        let bias = DynTensor::zeros(&[hidden], DType::F32, &dev).unwrap();
        let ln = LayerNorm::new(weight, bias, 1e-5).unwrap();

        group.bench_with_input(BenchmarkId::new("cpu", label), &(), |bencher, ()| {
            bencher.iter(|| black_box(ln.forward(&x).unwrap()));
        });
    }

    group.finish();
}

// ---------------------------------------------------------------------------
// Conv1d -- audio-sized inputs
// ---------------------------------------------------------------------------

fn bench_conv1d_audio(c: &mut Criterion) {
    let mut group = c.benchmark_group("conv1d");
    let dev = Device::Cpu;

    // Kokoro text encoder: Conv1d(512, 512, 5) with padding=2
    {
        let input = rand_tensor(&[1, 512, 64], -1.0, 1.0);
        let kernel = rand_tensor(&[512, 512, 5], -0.01, 0.01);
        let bias = DynTensor::zeros(&[512], DType::F32, &dev).unwrap();
        let config = Conv1dConfig::new(2, 1, 1);
        let conv = Conv1d::new(kernel, Some(bias), config).unwrap();

        group.bench_function("kokoro_text_enc_512x512x5", |bencher| {
            bencher.iter(|| black_box(conv.forward(&input).unwrap()));
        });
    }

    // Demucs encoder stage: Conv1d(1, 48, 8) with stride=4
    {
        let input = rand_tensor(&[1, 1, 24000], -1.0, 1.0); // 1 second at 24kHz
        let kernel = rand_tensor(&[48, 1, 8], -0.01, 0.01);
        let bias = DynTensor::zeros(&[48], DType::F32, &dev).unwrap();
        let config = Conv1dConfig::new(2, 4, 1);
        let conv = Conv1d::new(kernel, Some(bias), config).unwrap();

        group.bench_function("demucs_enc_1x48x8_s4_24k", |bencher| {
            bencher.iter(|| black_box(conv.forward(&input).unwrap()));
        });
    }

    // Silero VAD encoder stage: Conv1d(129, 128, 3)
    {
        let input = rand_tensor(&[1, 129, 5], -1.0, 1.0); // STFT frames
        let kernel = rand_tensor(&[128, 129, 3], -0.01, 0.01);
        let bias = DynTensor::zeros(&[128], DType::F32, &dev).unwrap();
        let config = Conv1dConfig::new(1, 1, 1);
        let conv = Conv1d::new(kernel, Some(bias), config).unwrap();

        group.bench_function("silero_vad_enc_129x128x3", |bencher| {
            bencher.iter(|| black_box(conv.forward(&input).unwrap()));
        });
    }

    group.finish();
}

// ---------------------------------------------------------------------------
// Embedding lookup -- vocabulary-sized tables
// ---------------------------------------------------------------------------

fn bench_embedding_sizes(c: &mut Criterion) {
    let mut group = c.benchmark_group("embedding");

    // GPT-2/Whisper vocab: 50257 x 768
    {
        let weight = rand_tensor(&[50257, 768], -0.02, 0.02);
        let emb = Embedding::new(weight).unwrap();
        let ids_data: Vec<u32> = (0..128).map(|i| (i * 1571) % 50257).collect();
        let ids = DynTensor::from_vec_u32(ids_data, &[1, 128], &Device::Cpu).unwrap();

        group.bench_function("gpt2_50257x768_seq128", |bencher| {
            bencher.iter(|| black_box(emb.forward(&ids).unwrap()));
        });
    }

    // Kokoro vocab: 178 x 128
    {
        let weight = rand_tensor(&[178, 128], -0.02, 0.02);
        let emb = Embedding::new(weight).unwrap();
        let ids_data: Vec<u32> = (0..32).map(|i| (i * 7) % 178).collect();
        let ids = DynTensor::from_vec_u32(ids_data, &[1, 32], &Device::Cpu).unwrap();

        group.bench_function("kokoro_178x128_seq32", |bencher| {
            bencher.iter(|| black_box(emb.forward(&ids).unwrap()));
        });
    }

    // Large LLM vocab: 32000 x 512
    {
        let weight = rand_tensor(&[32000, 512], -0.02, 0.02);
        let emb = Embedding::new(weight).unwrap();
        let ids_data: Vec<u32> = (0..256).map(|i| (i * 1571) % 32000).collect();
        let ids = DynTensor::from_vec_u32(ids_data, &[1, 256], &Device::Cpu).unwrap();

        group.bench_function("llm_32000x512_seq256", |bencher| {
            bencher.iter(|| black_box(emb.forward(&ids).unwrap()));
        });
    }

    group.finish();
}

// ---------------------------------------------------------------------------
// Criterion harness
// ---------------------------------------------------------------------------

criterion_group!(
    benches,
    bench_matmul_sizes,
    bench_matmul_rectangular,
    bench_matmul_batched,
    bench_softmax_batch,
    bench_layer_norm_sizes,
    bench_conv1d_audio,
    bench_embedding_sizes,
);
criterion_main!(benches);

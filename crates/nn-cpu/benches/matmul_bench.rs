// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Criterion benchmarks for nn-cpu matmul.
//!
//! Measures SIMD-tiled matmul at representative sizes:
//! - Small (16x16): micro-kernel overhead dominates
//! - Medium (128x128): cache-friendly tiling
//! - Large (512x512): memory-bandwidth bound
//! - Rectangular (256x64 * 64x256): non-square workloads
//! - Transposed-B path comparison

use std::hint::black_box;
use criterion::{criterion_group, criterion_main, Criterion};
use nn_cpu::matmul;

/// Generate deterministic matrix data for benchmarks.
fn make_matrix(rows: usize, cols: usize) -> Vec<f32> {
    (0..rows * cols)
        .map(|i| ((i % 97) as f32) * 0.01 - 0.48)
        .collect()
}

fn bench_matmul_16x16(c: &mut Criterion) {
    let m = 16;
    let k = 16;
    let n = 16;
    let a = make_matrix(m, k);
    let b = make_matrix(k, n);

    c.bench_function("matmul_16x16", |bench| {
        bench.iter(|| {
            black_box(matmul::matmul(black_box(&a), black_box(&b), m, k, n));
        });
    });
}

fn bench_matmul_64x64(c: &mut Criterion) {
    let m = 64;
    let k = 64;
    let n = 64;
    let a = make_matrix(m, k);
    let b = make_matrix(k, n);

    c.bench_function("matmul_64x64", |bench| {
        bench.iter(|| {
            black_box(matmul::matmul(black_box(&a), black_box(&b), m, k, n));
        });
    });
}

fn bench_matmul_128x128(c: &mut Criterion) {
    let m = 128;
    let k = 128;
    let n = 128;
    let a = make_matrix(m, k);
    let b = make_matrix(k, n);

    c.bench_function("matmul_128x128", |bench| {
        bench.iter(|| {
            black_box(matmul::matmul(black_box(&a), black_box(&b), m, k, n));
        });
    });
}

fn bench_matmul_256x256(c: &mut Criterion) {
    let m = 256;
    let k = 256;
    let n = 256;
    let a = make_matrix(m, k);
    let b = make_matrix(k, n);

    c.bench_function("matmul_256x256", |bench| {
        bench.iter(|| {
            black_box(matmul::matmul(black_box(&a), black_box(&b), m, k, n));
        });
    });
}

fn bench_matmul_512x512(c: &mut Criterion) {
    let m = 512;
    let k = 512;
    let n = 512;
    let a = make_matrix(m, k);
    let b = make_matrix(k, n);

    c.bench_function("matmul_512x512", |bench| {
        bench.iter(|| {
            black_box(matmul::matmul(black_box(&a), black_box(&b), m, k, n));
        });
    });
}

fn bench_matmul_rectangular(c: &mut Criterion) {
    let m = 256;
    let k = 64;
    let n = 256;
    let a = make_matrix(m, k);
    let b = make_matrix(k, n);

    c.bench_function("matmul_256x64x256", |bench| {
        bench.iter(|| {
            black_box(matmul::matmul(black_box(&a), black_box(&b), m, k, n));
        });
    });
}

fn bench_matmul_transposed_b_128(c: &mut Criterion) {
    let m = 128;
    let k = 128;
    let n = 128;
    let a = make_matrix(m, k);
    let b_t = make_matrix(n, k);

    c.bench_function("matmul_transposed_b_128x128", |bench| {
        bench.iter(|| {
            let mut c_out = vec![0.0f32; m * n];
            matmul::matmul_with_transposed_b(
                black_box(&a),
                black_box(&b_t),
                black_box(&mut c_out),
                m,
                k,
                n,
            );
            black_box(c_out);
        });
    });
}

fn bench_matmul_non_aligned(c: &mut Criterion) {
    let m = 65;
    let k = 70;
    let n = 33;
    let a = make_matrix(m, k);
    let b = make_matrix(k, n);

    c.bench_function("matmul_65x70x33_non_aligned", |bench| {
        bench.iter(|| {
            black_box(matmul::matmul(black_box(&a), black_box(&b), m, k, n));
        });
    });
}

criterion_group!(
    benches,
    bench_matmul_16x16,
    bench_matmul_64x64,
    bench_matmul_128x128,
    bench_matmul_256x256,
    bench_matmul_512x512,
    bench_matmul_rectangular,
    bench_matmul_transposed_b_128,
    bench_matmul_non_aligned,
);
criterion_main!(benches);

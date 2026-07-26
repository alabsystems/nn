// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! D2 benchmark: fused vs decomposed GPU kernels at Kokoro shapes (#3348).
//!
//! Compares per-operation GPU wall-clock time for fused (single TensorBlockBuilder
//! dispatch) vs decomposed (multi-dispatch DynTensor op chain) normalization
//! kernels at Kokoro's specific tensor shapes.
//!
//! Run: `cargo test -p nn-metal --test gpu_ops_all --release
//!       -- fused_norm_shape_benchmark --nocapture`

use nn_core::dyn_tensor::DynTensor;
use nn_core::{DType, Device};

use std::time::Instant;

// ---------------------------------------------------------------------------
// Decomposed (multi-dispatch) reference implementations
// ---------------------------------------------------------------------------

/// GroupNorm decomposed: ~12 separate GPU dispatches.
/// Replicates the group_norm_cpu() chain but on GPU tensors so each op is a
/// separate Metal dispatch.
fn group_norm_decomposed(
    x: &DynTensor,
    num_groups: usize,
    weight: &DynTensor,
    bias: &DynTensor,
    eps: f64,
) -> DynTensor {
    let dims = x.dims();
    let batch = dims[0];
    let channels = dims[1];
    let channels_per_group = channels / num_groups;
    let spatial: usize = dims[2..].iter().product();
    let cpg_spatial = channels_per_group * spatial;

    let x_reshaped = x.reshape([batch, num_groups, cpg_spatial]).unwrap();
    let mean = x_reshaped.mean_keepdim(2).unwrap();
    let centered = x_reshaped.broadcast_sub(&mean).unwrap();
    let var = centered.sqr().unwrap().mean_keepdim(2).unwrap();
    let eps_t = DynTensor::full(&[1, 1, 1], eps, DType::F32, &x.device()).unwrap();
    let std_inv = var
        .broadcast_add(&eps_t)
        .unwrap()
        .sqrt()
        .unwrap()
        .recip()
        .unwrap();
    let normed = centered.broadcast_mul(&std_inv).unwrap();
    let normed = normed.reshape(dims).unwrap();

    // Per-channel affine: weight [C] -> [1, C, 1, ...], bias same.
    let mut wb_shape = vec![1usize; dims.len()];
    wb_shape[1] = channels;
    let w = weight.reshape(&wb_shape).unwrap();
    let b = bias.reshape(&wb_shape).unwrap();
    normed.broadcast_mul(&w).unwrap().broadcast_add(&b).unwrap()
}

/// RmsNorm decomposed: ~6 separate GPU dispatches.
fn rms_norm_decomposed(x: &DynTensor, weight: &DynTensor, eps: f64) -> DynTensor {
    let rank = x.rank();
    let last_dim = rank - 1;
    let x_sq = x.sqr().unwrap();
    let mean_sq = x_sq.mean_keepdim(last_dim).unwrap();
    let eps_t = DynTensor::full(vec![1; rank], eps, DType::F32, &x.device()).unwrap();
    let rms = mean_sq.broadcast_add(&eps_t).unwrap().sqrt().unwrap();
    let normed = x.broadcast_div(&rms).unwrap();
    normed.broadcast_mul(weight).unwrap()
}

/// Snake decomposed: ~7 separate GPU dispatches.
/// x + (1/alpha) * sin^2(alpha * x)
fn snake_decomposed(x: &DynTensor, alpha: &DynTensor) -> DynTensor {
    let alpha_safe = alpha.clamp(1e-8, 1e6).unwrap();
    let scaled = x.broadcast_mul(&alpha_safe).unwrap();
    let sin_sq = scaled.sin().unwrap().sqr().unwrap();
    let inv_alpha = alpha_safe.recip().unwrap();
    x.add(&sin_sq.broadcast_mul(&inv_alpha).unwrap()).unwrap()
}

// ---------------------------------------------------------------------------
// Fused (single TensorBlockBuilder dispatch) paths
// ---------------------------------------------------------------------------

/// GroupNorm fused: uses GpuNnOps::group_norm → single TensorBlockBuilder dispatch.
fn group_norm_fused(
    x: &DynTensor,
    num_groups: usize,
    weight: &DynTensor,
    bias: &DynTensor,
    eps: f64,
) -> DynTensor {
    use nn_core::layers::{GroupNorm, Module};
    let num_channels = x.dims()[1];
    let gn = GroupNorm::new(num_groups, num_channels, weight.clone(), bias.clone(), eps)
        .expect("valid GroupNorm");
    gn.forward(x).unwrap()
}

/// RmsNorm fused: uses GpuNnOps::rms_norm → single TensorBlockBuilder dispatch.
fn rms_norm_fused(x: &DynTensor, weight: &DynTensor, eps: f64) -> DynTensor {
    use nn_core::layers::{Module, RmsNorm};
    let rn = RmsNorm::new(weight.clone(), eps).expect("valid RmsNorm");
    rn.forward(x).unwrap()
}

/// Snake fused: uses GpuNnOps::snake_tensor → single TensorBlockBuilder dispatch.
fn snake_fused(x: &DynTensor, alpha: &DynTensor) -> DynTensor {
    x.snake_tensor(alpha).unwrap()
}

// ---------------------------------------------------------------------------
// Benchmark harness
// ---------------------------------------------------------------------------

struct BenchResult {
    fused_us: f64,
    decomposed_us: f64,
    label: String,
    numel: usize,
}

impl BenchResult {
    fn ratio(&self) -> f64 {
        self.decomposed_us / self.fused_us
    }
}

fn bench_pair(
    label: &str,
    numel: usize,
    warmup: usize,
    iters: usize,
    fused_fn: impl Fn(),
    decomposed_fn: impl Fn(),
) -> BenchResult {
    // Warmup both paths.
    for _ in 0..warmup {
        fused_fn();
    }
    nn_metal::flush().unwrap();

    // Time fused.
    let start = Instant::now();
    for _ in 0..iters {
        fused_fn();
    }
    nn_metal::flush().unwrap();
    let fused_elapsed = start.elapsed();

    // Warmup decomposed.
    for _ in 0..warmup {
        decomposed_fn();
    }
    nn_metal::flush().unwrap();

    // Time decomposed.
    let start = Instant::now();
    for _ in 0..iters {
        decomposed_fn();
    }
    nn_metal::flush().unwrap();
    let decomposed_elapsed = start.elapsed();

    BenchResult {
        fused_us: fused_elapsed.as_micros() as f64 / iters as f64,
        decomposed_us: decomposed_elapsed.as_micros() as f64 / iters as f64,
        label: label.to_string(),
        numel,
    }
}

fn make_gpu_tensor(shape: &[usize], seed: u64) -> DynTensor {
    let numel: usize = shape.iter().product();
    let data = super::test_utils::rand_f32_vec(seed, numel, -1.0, 1.0);
    DynTensor::from_vec(data, shape, &Device::Cpu)
        .unwrap()
        .to_device(&Device::metal())
        .unwrap()
}

fn make_gpu_ones(shape: &[usize]) -> DynTensor {
    DynTensor::ones(shape, DType::F32, &Device::Cpu)
        .unwrap()
        .to_device(&Device::metal())
        .unwrap()
}

// ---------------------------------------------------------------------------
// Benchmark tests
// ---------------------------------------------------------------------------

/// GroupNorm fused vs decomposed at Kokoro channel counts.
///
/// Kokoro uses GroupNorm in ResBlocks with channels ∈ {48, 96, 192, 384, 512}.
/// The fused kernel uses a single TensorBlockBuilder dispatch; the decomposed
/// path chains ~12 separate GPU dispatches.
#[test]
fn fused_vs_decomposed_group_norm() {
    super::test_utils::gpu_init();

    let warmup = 3;
    let iters = 10;
    let eps = 1e-5;
    // Kokoro ResBlocks: num_groups typically = channels (InstanceNorm-like via GroupNorm)
    // or a fixed divisor. Testing both small and standard group counts.
    let configs: Vec<(&str, usize, usize, usize)> = vec![
        // (label, batch, channels, temporal)
        ("C=48_T=256", 1, 48, 256),
        ("C=96_T=128", 1, 96, 128),
        ("C=192_T=64", 1, 192, 64),
        ("C=384_T=32", 1, 384, 32),
        ("C=512_T=32", 1, 512, 32),
        // Larger temporal for small channels (Kokoro early encoder stages).
        ("C=48_T=1024", 1, 48, 1024),
        ("C=96_T=512", 1, 96, 512),
    ];

    eprintln!("\n=== GroupNorm: fused vs decomposed (Kokoro shapes) ===");
    eprintln!(
        "{:<20} {:>10} {:>10} {:>10} {:>8}",
        "Config", "Fused(us)", "Decomp(us)", "Ratio", "Numel"
    );

    let mut results = Vec::new();
    for (label, batch, channels, temporal) in &configs {
        let shape = [*batch, *channels, *temporal];
        let numel: usize = shape.iter().product();
        let x = make_gpu_tensor(&shape, 0xBE_0001);
        let weight = make_gpu_ones(&[*channels]);
        let bias = make_gpu_tensor(&[*channels], 0xBE_0002);
        let num_groups = *channels; // GroupNorm with groups=channels (InstanceNorm-like)

        let x_ref = x.clone();
        let w_ref = weight.clone();
        let b_ref = bias.clone();
        let x_dec = x.clone();
        let w_dec = weight.clone();
        let b_dec = bias.clone();

        let result = bench_pair(
            &format!("GroupNorm {label}"),
            numel,
            warmup,
            iters,
            move || {
                let _ = group_norm_fused(&x_ref, num_groups, &w_ref, &b_ref, eps);
            },
            move || {
                let _ = group_norm_decomposed(&x_dec, num_groups, &w_dec, &b_dec, eps);
            },
        );

        eprintln!(
            "{:<20} {:>10.0} {:>10.0} {:>10.2}x {:>8}",
            label,
            result.fused_us,
            result.decomposed_us,
            result.ratio(),
            result.numel,
        );
        results.push(result);
    }

    // Report summary.
    let avg_ratio: f64 = results.iter().map(BenchResult::ratio).sum::<f64>() / results.len() as f64;
    eprintln!("\nAverage decomposed/fused ratio: {avg_ratio:.2}x");

    // If fused is slower (ratio < 1.0), that's the regression signal.
    let any_fused_slower = results.iter().any(|r| r.ratio() < 0.8);
    if any_fused_slower {
        eprintln!("WARNING: fused kernel is SLOWER than decomposed for some shapes!");
    }
}

/// RmsNorm fused vs decomposed at Kokoro/LLM shapes.
///
/// RmsNorm is used in Qwen3/GLM5 transformer layers with dim ∈ {256, 512, 1024}.
/// Also used in Kokoro's prosody predictor (D=256).
#[test]
fn fused_vs_decomposed_rms_norm() {
    super::test_utils::gpu_init();

    let warmup = 3;
    let iters = 10;
    let eps = 1e-6;
    let configs: Vec<(&str, usize, usize, usize)> = vec![
        // (label, batch, seq_len, dim)
        ("S=8_D=256", 1, 8, 256),
        ("S=8_D=512", 1, 8, 512),
        ("S=32_D=256", 1, 32, 256),
        ("S=32_D=512", 1, 32, 512),
        ("S=128_D=512", 1, 128, 512),
        // Miniaturized Qwen3 shape.
        ("S=8_D=1024", 1, 8, 1024),
    ];

    eprintln!("\n=== RmsNorm: fused vs decomposed ===");
    eprintln!(
        "{:<20} {:>10} {:>10} {:>10} {:>8}",
        "Config", "Fused(us)", "Decomp(us)", "Ratio", "Numel"
    );

    let mut results = Vec::new();
    for (label, batch, seq_len, dim) in &configs {
        let shape = [*batch, *seq_len, *dim];
        let numel: usize = shape.iter().product();
        let x = make_gpu_tensor(&shape, 0xBE_2001);
        let weight = make_gpu_ones(&[*dim]);

        let x_ref = x.clone();
        let w_ref = weight.clone();
        let x_dec = x.clone();
        let w_dec = weight.clone();

        let result = bench_pair(
            &format!("RmsNorm {label}"),
            numel,
            warmup,
            iters,
            move || {
                let _ = rms_norm_fused(&x_ref, &w_ref, eps);
            },
            move || {
                let _ = rms_norm_decomposed(&x_dec, &w_dec, eps);
            },
        );

        eprintln!(
            "{:<20} {:>10.0} {:>10.0} {:>10.2}x {:>8}",
            label,
            result.fused_us,
            result.decomposed_us,
            result.ratio(),
            result.numel,
        );
        results.push(result);
    }

    let avg_ratio: f64 = results.iter().map(BenchResult::ratio).sum::<f64>() / results.len() as f64;
    eprintln!("\nAverage decomposed/fused ratio: {avg_ratio:.2}x");

    let any_fused_slower = results.iter().any(|r| r.ratio() < 0.8);
    if any_fused_slower {
        eprintln!("WARNING: fused kernel is SLOWER than decomposed for some shapes!");
    }
}

/// Snake fused vs decomposed at Kokoro shapes.
///
/// Snake activation is used in every Kokoro ResBlock (after GroupNorm).
/// shapes: [1, channels, T] with channels ∈ {48, 96, 192, 384, 512}.
#[test]
fn fused_vs_decomposed_snake() {
    super::test_utils::gpu_init();

    let warmup = 3;
    let iters = 10;
    let configs: Vec<(&str, usize, usize, usize)> = vec![
        ("C=48_T=256", 1, 48, 256),
        ("C=96_T=128", 1, 96, 128),
        ("C=192_T=64", 1, 192, 64),
        ("C=384_T=32", 1, 384, 32),
        ("C=512_T=32", 1, 512, 32),
        ("C=48_T=1024", 1, 48, 1024),
        ("C=96_T=512", 1, 96, 512),
    ];

    eprintln!("\n=== Snake: fused vs decomposed (Kokoro shapes) ===");
    eprintln!(
        "{:<20} {:>10} {:>10} {:>10} {:>8}",
        "Config", "Fused(us)", "Decomp(us)", "Ratio", "Numel"
    );

    let mut results = Vec::new();
    for (label, batch, channels, temporal) in &configs {
        let shape = [*batch, *channels, *temporal];
        let numel: usize = shape.iter().product();
        let x = make_gpu_tensor(&shape, 0xBE_3001);
        // Alpha is per-channel: [1, C, 1] to match broadcast_left alignment.
        let alpha = make_gpu_tensor(&[1, *channels, 1], 0xBE_3002);

        let x_ref = x.clone();
        let a_ref = alpha.clone();
        let x_dec = x.clone();
        let a_dec = alpha.clone();

        let result = bench_pair(
            &format!("Snake {label}"),
            numel,
            warmup,
            iters,
            move || {
                let _ = snake_fused(&x_ref, &a_ref);
            },
            move || {
                let _ = snake_decomposed(&x_dec, &a_dec);
            },
        );

        eprintln!(
            "{:<20} {:>10.0} {:>10.0} {:>10.2}x {:>8}",
            label,
            result.fused_us,
            result.decomposed_us,
            result.ratio(),
            result.numel,
        );
        results.push(result);
    }

    let avg_ratio: f64 = results.iter().map(BenchResult::ratio).sum::<f64>() / results.len() as f64;
    eprintln!("\nAverage decomposed/fused ratio: {avg_ratio:.2}x");

    let any_fused_slower = results.iter().any(|r| r.ratio() < 0.8);
    if any_fused_slower {
        eprintln!("WARNING: fused kernel is SLOWER than decomposed for some shapes!");
    }
}

/// Combined summary: all three ops at the most representative Kokoro shape.
///
/// Tests GroupNorm+Snake+RmsNorm at the shape that appears most frequently
/// in Kokoro's ResBlock stack to estimate total per-block impact.
#[test]
fn fused_norm_combined_kokoro_resblock() {
    super::test_utils::gpu_init();

    let warmup = 3;
    let iters = 15;
    let eps = 1e-5;

    // Representative Kokoro ResBlock shape: [1, 512, 32] at D=512 (generator).
    let batch = 1;
    let channels = 512;
    let temporal = 32;
    let shape = [batch, channels, temporal];

    let x = make_gpu_tensor(&shape, 0xBE_4001);
    let weight = make_gpu_ones(&[channels]);
    let bias = make_gpu_tensor(&[channels], 0xBE_4002);
    let alpha = make_gpu_tensor(&[1, channels, 1], 0xBE_4003);

    eprintln!("\n=== Combined Kokoro ResBlock benchmark [1, {channels}, {temporal}] ===");
    eprintln!("Operations: GroupNorm(groups={channels}) + Snake + residual");
    eprintln!("Warmup: {warmup}, Iterations: {iters}\n");

    // Fused combined: GroupNorm → Snake (via nn module, each gets fused dispatch).
    let x_f = x.clone();
    let w_f = weight.clone();
    let b_f = bias.clone();
    let a_f = alpha.clone();
    let fused_resblock = move || {
        let normed = group_norm_fused(&x_f, channels, &w_f, &b_f, eps);
        let _activated = snake_fused(&normed, &a_f);
    };

    // Decomposed combined: manual op chains.
    let x_d = x;
    let w_d = weight;
    let b_d = bias;
    let a_d = alpha;
    let decomposed_resblock = move || {
        let normed = group_norm_decomposed(&x_d, channels, &w_d, &b_d, eps);
        let _activated = snake_decomposed(&normed, &a_d);
    };

    // Warmup.
    for _ in 0..warmup {
        fused_resblock();
    }
    nn_metal::flush().unwrap();
    for _ in 0..warmup {
        decomposed_resblock();
    }
    nn_metal::flush().unwrap();

    // Measure fused.
    let start = Instant::now();
    for _ in 0..iters {
        fused_resblock();
    }
    nn_metal::flush().unwrap();
    let fused_us = start.elapsed().as_micros() as f64 / f64::from(iters);

    // Measure decomposed.
    let start = Instant::now();
    for _ in 0..iters {
        decomposed_resblock();
    }
    nn_metal::flush().unwrap();
    let decomposed_us = start.elapsed().as_micros() as f64 / f64::from(iters);

    let ratio = decomposed_us / fused_us;
    eprintln!("Fused (GroupNorm+Snake):      {fused_us:>8.0} us/iter");
    eprintln!("Decomposed (GroupNorm+Snake):  {decomposed_us:>8.0} us/iter");
    eprintln!("Ratio (decomposed/fused):      {ratio:>8.2}x");
    eprintln!();

    if ratio < 1.0 {
        eprintln!(
            "REGRESSION: fused path is {:.0}% SLOWER than decomposed! \
             This likely contributes to the #3348 per-step slowdown.",
            (1.0 / ratio - 1.0) * 100.0
        );
    } else {
        eprintln!(
            "Fused path is {:.0}% FASTER than decomposed. \
             Fused kernels are NOT the #3348 regression cause for this shape.",
            (ratio - 1.0) * 100.0
        );
    }
}

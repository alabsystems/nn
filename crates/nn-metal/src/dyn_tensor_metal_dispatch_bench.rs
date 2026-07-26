#![allow(deprecated)]
// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Dispatch overhead benchmark: DynTensor (per-op) vs TensorDispatch (fused).
//!
//! Compares the two GPU dispatch systems for the same computation:
//! 1. **Fused (TensorDispatch):** `TensorBlockBuilder` builds a multi-op
//!    `TensorKernelDef` dispatched in one Metal command buffer commit.
//! 2. **DynTensor (per-op):** Each op is a separate GPU kernel launch with
//!    its own command buffer commit_and_wait.
//!
//! Benchmark patterns:
//! - Conv1d + GELU (2-op pipeline)
//! - Conv1d + GELU + Conv1d (3-op pipeline, simulates deeper nets)
//!
//! Issue: #1185 AC3

use std::collections::HashMap;
use std::time::Instant;

use nn_core::dyn_tensor::DynTensor;
use nn_core::Device;

use nn_dsl::ir::ScalarType;
use nn_dsl::TensorBlockBuilder;

use crate::cache::PipelineCache;
use crate::metal_backend::MetalBackend;
use crate::tensor_dispatch::execute_tensor_dispatch;

fn init() -> PipelineCache {
    let _ = MetalBackend::init();
    super::register_metal_dyn_backend();
    PipelineCache::new_global().expect("Metal required for benchmark")
}

/// Deterministic varied-value data vector.
fn data_vec(n: usize, base: f32) -> Vec<f32> {
    (0..n)
        .map(|i| base + 0.01 * (i as f32) - 0.005 * ((i % 3) as f32))
        .collect()
}

// -- Benchmark dimensions -----------------------------------------------------
// Realistic dimensions for measuring dispatch overhead. Two scales: small
// (representative of per-layer overhead), and medium (typical production size).

const WARMUP_RUNS: u32 = 1;
const TIMED_RUNS: u32 = 3;

// -- Conv1d + GELU fused benchmark -------------------------------------------

/// Build a fused Conv1d+GELU TensorKernelDef.
fn build_fused_conv_gelu(
    in_ch: usize,
    out_ch: usize,
    in_len: usize,
    k_size: usize,
    stride: usize,
    padding: usize,
) -> (nn_dsl::TensorKernelDef, usize) {
    let out_len = (in_len + 2 * padding - k_size) / stride + 1;
    let mut b = TensorBlockBuilder::new("fused_conv_gelu");
    let data = b.add_input("data", &[in_ch, in_len]);
    let weight = b.add_input("weight", &[out_ch, in_ch, k_size]);
    let conv = b.add_conv1d(data, weight, None, stride, padding, &[out_ch, out_len]);
    let gelu = b.add_gelu(conv, &[out_ch, out_len]);
    let def = b.build(gelu).expect("valid fused graph");
    (def, out_len)
}

/// Benchmark: Conv1d + GELU at given scale.
///
/// Prints timing comparison to stderr. Verifies correctness (fused vs DynTensor
/// produce the same output) then measures dispatch overhead.
#[test]
fn bench_conv_gelu_dispatch_overhead() {
    let cache = init();

    // Two scales: small (per-layer overhead dominates) and medium (compute grows).
    let scales: &[(usize, usize, usize, usize, &str)] = &[
        // (in_ch, out_ch, in_len, k_size, label)
        (16, 32, 128, 3, "small"),
        (48, 96, 512, 8, "medium"),
        (96, 192, 1024, 8, "large"),
    ];

    for &(in_ch, out_ch, in_len, k_size, label) in scales {
        let stride = 1;
        let padding = k_size / 2;
        let out_len = (in_len + 2 * padding - k_size) / stride + 1;

        let data = data_vec(in_ch * in_len, 0.1);
        let weight = data_vec(out_ch * in_ch * k_size, 0.02);

        // -- Fused path: TensorDispatch ------------------------------------
        let (def, _) = build_fused_conv_gelu(in_ch, out_ch, in_len, k_size, stride, padding);
        let inputs: HashMap<&str, &[f32]> =
            HashMap::from([("data", data.as_slice()), ("weight", weight.as_slice())]);

        // Warmup
        for _ in 0..WARMUP_RUNS {
            let _ = execute_tensor_dispatch(&cache, &def, ScalarType::F32, &inputs);
        }
        let fused_start = Instant::now();
        for _ in 0..TIMED_RUNS {
            let _ = execute_tensor_dispatch(&cache, &def, ScalarType::F32, &inputs);
        }
        let fused_elapsed = fused_start.elapsed();
        let fused_out = execute_tensor_dispatch(&cache, &def, ScalarType::F32, &inputs)
            .expect("fused dispatch");

        // -- DynTensor path: per-op dispatch -------------------------------
        let gpu_dev = Device::metal();
        let dt_data = DynTensor::from_vec(data.clone(), &[1, in_ch, in_len], &gpu_dev).unwrap();
        let dt_weight =
            DynTensor::from_vec(weight.clone(), &[out_ch, in_ch, k_size], &gpu_dev).unwrap();

        // Warmup
        for _ in 0..WARMUP_RUNS {
            let conv_out = dt_data.conv1d(&dt_weight, padding, stride, 1, 1).unwrap();
            let _ = conv_out.gelu().unwrap();
        }
        let dyn_start = Instant::now();
        for _ in 0..TIMED_RUNS {
            let conv_out = dt_data.conv1d(&dt_weight, padding, stride, 1, 1).unwrap();
            let _ = conv_out.gelu().unwrap();
        }
        let dyn_elapsed = dyn_start.elapsed();
        let dyn_gelu = {
            let conv_out = dt_data.conv1d(&dt_weight, padding, stride, 1, 1).unwrap();
            conv_out.gelu().unwrap()
        };

        // -- Correctness check: both produce same output -------------------
        let dyn_vals = dyn_gelu
            .to_device(&Device::Cpu)
            .unwrap()
            .to_flat_vec::<f32>()
            .unwrap();
        assert_eq!(
            fused_out.len(),
            out_ch * out_len,
            "{label}: fused output length"
        );
        // DynTensor has batch dim [1, out_ch, out_len]
        assert_eq!(
            dyn_vals.len(),
            out_ch * out_len,
            "{label}: dyn output length"
        );
        let mut max_diff = 0.0f32;
        for (i, (&f, &d)) in fused_out.iter().zip(dyn_vals.iter()).enumerate() {
            let diff = (f - d).abs();
            if diff > max_diff {
                max_diff = diff;
            }
            assert!(
                diff < 0.01,
                "{label}: fused vs dyn mismatch at [{i}]: fused={f}, dyn={d}, diff={diff}"
            );
        }

        // -- Report timing -------------------------------------------------
        let fused_avg_us = fused_elapsed.as_micros() as f64 / f64::from(TIMED_RUNS);
        let dyn_avg_us = dyn_elapsed.as_micros() as f64 / f64::from(TIMED_RUNS);
        let overhead_pct = if fused_avg_us > 0.0 {
            (dyn_avg_us / fused_avg_us - 1.0) * 100.0
        } else {
            0.0
        };

        eprintln!(
            "[{label}] Conv1d({in_ch}->{out_ch}, k={k_size}, T={in_len}) + GELU:\n\
             \x20 fused={fused_avg_us:.0}us, dyn={dyn_avg_us:.0}us, \
             overhead={overhead_pct:.0}%, max_diff={max_diff:.2e}\n\
             \x20 (ops=2, {TIMED_RUNS} runs, {WARMUP_RUNS} warmup)"
        );
    }
}

// -- Conv1d + GELU + Conv1d(1x1) fused benchmark ----------------------------

/// Build a fused Conv1d+GELU+Conv1d(1x1) TensorKernelDef (simulates deeper
/// pipeline: conv + activation + pointwise projection).
fn build_fused_conv_gelu_proj(
    in_ch: usize,
    mid_ch: usize,
    out_ch: usize,
    in_len: usize,
    k_size: usize,
    stride: usize,
    padding: usize,
) -> (nn_dsl::TensorKernelDef, usize) {
    let mid_len = (in_len + 2 * padding - k_size) / stride + 1;
    let mut b = TensorBlockBuilder::new("fused_conv_gelu_proj");
    let data = b.add_input("data", &[in_ch, in_len]);
    let w1 = b.add_input("w1", &[mid_ch, in_ch, k_size]);
    let w2 = b.add_input("w2", &[out_ch, mid_ch, 1]);
    let conv1 = b.add_conv1d(data, w1, None, stride, padding, &[mid_ch, mid_len]);
    let gelu = b.add_gelu(conv1, &[mid_ch, mid_len]);
    let conv2 = b.add_conv1d(gelu, w2, None, 1, 0, &[out_ch, mid_len]);
    let def = b.build(conv2).expect("valid fused graph");
    (def, mid_len)
}

/// Benchmark: Conv1d + GELU + Conv1d(1x1) at multiple scales.
///
/// 3-op pipeline: measures the amplification of per-op overhead as the
/// op count grows.
#[test]
fn bench_conv_gelu_proj_dispatch_overhead() {
    let cache = init();

    let scales: &[(usize, usize, usize, usize, usize, &str)] = &[
        // (in_ch, mid_ch, out_ch, in_len, k_size, label)
        (16, 32, 16, 128, 3, "small"),
        (48, 96, 48, 512, 8, "medium"),
        (96, 192, 96, 1024, 8, "large"),
    ];

    for &(in_ch, mid_ch, out_ch, in_len, k_size, label) in scales {
        let stride = 1;
        let padding = k_size / 2;
        let mid_len = (in_len + 2 * padding - k_size) / stride + 1;

        let data = data_vec(in_ch * in_len, 0.1);
        let w1 = data_vec(mid_ch * in_ch * k_size, 0.02);
        let w2 = data_vec(out_ch * mid_ch, 0.03);

        // -- Fused path ---
        let (def, _) =
            build_fused_conv_gelu_proj(in_ch, mid_ch, out_ch, in_len, k_size, stride, padding);
        let inputs: HashMap<&str, &[f32]> = HashMap::from([
            ("data", data.as_slice()),
            ("w1", w1.as_slice()),
            ("w2", w2.as_slice()),
        ]);

        for _ in 0..WARMUP_RUNS {
            let _ = execute_tensor_dispatch(&cache, &def, ScalarType::F32, &inputs);
        }
        let fused_start = Instant::now();
        for _ in 0..TIMED_RUNS {
            let _ = execute_tensor_dispatch(&cache, &def, ScalarType::F32, &inputs);
        }
        let fused_elapsed = fused_start.elapsed();
        let fused_out = execute_tensor_dispatch(&cache, &def, ScalarType::F32, &inputs)
            .expect("fused dispatch");

        // -- DynTensor path ---
        let gpu_dev = Device::metal();
        let dt_data = DynTensor::from_vec(data.clone(), &[1, in_ch, in_len], &gpu_dev).unwrap();
        let dt_w1 = DynTensor::from_vec(w1.clone(), &[mid_ch, in_ch, k_size], &gpu_dev).unwrap();
        let dt_w2 = DynTensor::from_vec(w2.clone(), &[out_ch, mid_ch, 1], &gpu_dev).unwrap();

        for _ in 0..WARMUP_RUNS {
            let c1 = dt_data.conv1d(&dt_w1, padding, stride, 1, 1).unwrap();
            let g = c1.gelu().unwrap();
            let _ = g.conv1d(&dt_w2, 0, 1, 1, 1).unwrap();
        }
        let dyn_start = Instant::now();
        for _ in 0..TIMED_RUNS {
            let c1 = dt_data.conv1d(&dt_w1, padding, stride, 1, 1).unwrap();
            let g = c1.gelu().unwrap();
            let _ = g.conv1d(&dt_w2, 0, 1, 1, 1).unwrap();
        }
        let dyn_elapsed = dyn_start.elapsed();
        let dyn_out = {
            let c1 = dt_data.conv1d(&dt_w1, padding, stride, 1, 1).unwrap();
            let g = c1.gelu().unwrap();
            g.conv1d(&dt_w2, 0, 1, 1, 1).unwrap()
        };

        // -- Correctness check ---
        let dyn_vals = dyn_out
            .to_device(&Device::Cpu)
            .unwrap()
            .to_flat_vec::<f32>()
            .unwrap();
        assert_eq!(fused_out.len(), out_ch * mid_len, "{label}: fused len");
        assert_eq!(dyn_vals.len(), out_ch * mid_len, "{label}: dyn len");
        let mut max_diff = 0.0f32;
        for (&f, &d) in fused_out.iter().zip(dyn_vals.iter()) {
            let diff = (f - d).abs();
            if diff > max_diff {
                max_diff = diff;
            }
        }
        assert!(max_diff < 0.05, "{label}: fused vs dyn max_diff={max_diff}");

        // -- Report ---
        let fused_avg_us = fused_elapsed.as_micros() as f64 / f64::from(TIMED_RUNS);
        let dyn_avg_us = dyn_elapsed.as_micros() as f64 / f64::from(TIMED_RUNS);
        let overhead_pct = if fused_avg_us > 0.0 {
            (dyn_avg_us / fused_avg_us - 1.0) * 100.0
        } else {
            0.0
        };

        eprintln!(
            "[{label}] Conv1d({in_ch}->{mid_ch}, k={k_size}) + GELU + Conv1d({mid_ch}->{out_ch}, 1x1):\n\
             \x20 fused={fused_avg_us:.0}us, dyn={dyn_avg_us:.0}us, \
             overhead={overhead_pct:.0}%, max_diff={max_diff:.2e}\n\
             \x20 (ops=3, {TIMED_RUNS} runs, {WARMUP_RUNS} warmup)"
        );
    }
}

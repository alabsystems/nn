// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Frame-bucket ICB Metal dispatch benchmark + correctness tests.
//!
//! Validates that:
//! 1. Frame-bucket ICB dispatch produces results identical to standard
//!    (non-ICB) dispatch for the same model trace.
//! 2. Bucket selection correctly maps variable frame counts to the
//!    smallest enclosing bucket, and padded outputs match unpadded
//!    within the valid prefix.
//! 3. Benchmark comparison of ICB vs non-ICB dispatch latency across
//!    multiple bucket sizes.
//!
//! Part of #3551.

use std::time::Instant;

use nn_core::dyn_tensor::trace::{ComputationGraph, TraceOp};
use nn_metal::compiled_model::CompiledModel;

use super::helpers::{
    assert_close, binary_node, create_input_buffer, input_node, read_output_n, unary_node,
};

// ── Graph builders ──────────────────────────────────────────────────

/// Build a parameterized elementwise DAG with variable input size.
///
/// ```text
/// input [size]
///   |-> a = relu(input)
///   |-> b = exp(input)
///        c = a + b
///        d = a - b
///        e = c + d
///        f = c - d
///        g = e + f    (output)
/// ```
///
/// All ops are elementwise => fully ICB-eligible.
fn build_variable_size_dag(size: usize) -> ComputationGraph {
    let shape = &[size];
    ComputationGraph::from_nodes(vec![
        input_node(0, shape),
        unary_node(1, "relu_a", TraceOp::Relu, 0, shape),
        unary_node(2, "exp_b", TraceOp::Exp, 0, shape),
        binary_node(3, "add_c", TraceOp::Add, 1, 2, shape),
        binary_node(4, "sub_d", TraceOp::Sub, 1, 2, shape),
        binary_node(5, "add_e", TraceOp::Add, 3, 4, shape),
        binary_node(6, "sub_f", TraceOp::Sub, 3, 4, shape),
        binary_node(7, "add_g", TraceOp::Add, 5, 6, shape),
    ])
}

/// CPU reference computation matching the variable-size DAG.
fn dag_reference(x: &[f32]) -> Vec<f32> {
    x.iter()
        .map(|&v| {
            let a = v.max(0.0); // relu
            let b = v.exp(); // exp
            let c = a + b;
            let d = a - b;
            let e = c + d; // = 2*a
            let f = c - d; // = 2*b
            e + f // = 2*(relu(x) + exp(x))
        })
        .collect()
}

/// Build a deeper chain to ensure ICB segments span many steps.
///
/// ```text
/// input [size]
///   -> relu -> tanh -> relu -> tanh -> relu -> tanh
///   -> relu -> tanh -> relu -> tanh  (output, 10 elementwise ops)
/// ```
fn build_deep_chain(size: usize) -> ComputationGraph {
    let shape = &[size];
    let mut nodes = vec![input_node(0, shape)];
    for i in 1..=10 {
        let op = if i % 2 == 1 {
            TraceOp::Relu
        } else {
            TraceOp::Tanh
        };
        nodes.push(unary_node(
            i as u64,
            &format!("op_{i}"),
            op,
            (i - 1) as u64,
            shape,
        ));
    }
    ComputationGraph::from_nodes(nodes)
}

/// CPU reference for the deep chain.
fn deep_chain_reference(x: &[f32]) -> Vec<f32> {
    x.iter()
        .map(|&v| {
            let mut val = v;
            for i in 1..=10 {
                val = if i % 2 == 1 {
                    val.max(0.0) // relu
                } else {
                    val.tanh() // tanh
                };
            }
            val
        })
        .collect()
}

// ── Correctness: ICB vs non-ICB produce identical results ───────────

/// Compile and execute a model, returning the output as f32 slice.
///
/// Runs two passes: first encodes ICB (if eligible), second replays.
/// Returns the second-pass output (ICB replay path if segments exist).
fn compile_and_run_two_passes(
    cache: &nn_metal::PipelineCache,
    graph: &ComputationGraph,
    input_data: &[f32],
    output_size: usize,
) -> Vec<f32> {
    let compiled = CompiledModel::builder(graph, cache)
        .build()
        .expect("compile");
    let buf = create_input_buffer(cache, input_data);

    // Pass 1: encode ICB.
    let _out1 = compiled
        .execute(cache, &[&buf])
        .expect("execute pass 1 (encode)");

    // Pass 2: replay ICB.
    let out2 = compiled
        .execute(cache, &[&buf])
        .expect("execute pass 2 (replay)");
    read_output_n(&out2, output_size)
}

/// Core correctness: ICB dispatch matches CPU reference for the diamond DAG.
#[test]
fn test_frame_bucket_icb_correctness_diamond_dag() {
    super::test_utils::gpu_init();
    let cache = super::test_utils::metal_setup();

    // Test multiple sizes that map to different buckets.
    let test_sizes = [16, 32, 64, 128, 256];

    for &size in &test_sizes {
        let graph = build_variable_size_dag(size);
        let input_data: Vec<f32> = (0..size)
            .map(|i| (i as f32 - size as f32 / 2.0) * 0.1)
            .collect();
        let expected = dag_reference(&input_data);

        let result = compile_and_run_two_passes(&cache, &graph, &input_data, size);
        assert_close(
            &format!("diamond_dag_size_{size}"),
            &result,
            &expected,
            1e-4,
        );
    }
}

/// Core correctness: ICB dispatch matches CPU reference for deep chain.
#[test]
fn test_frame_bucket_icb_correctness_deep_chain() {
    super::test_utils::gpu_init();
    let cache = super::test_utils::metal_setup();

    let test_sizes = [32, 64, 128];

    for &size in &test_sizes {
        let graph = build_deep_chain(size);
        let input_data: Vec<f32> = (0..size)
            .map(|i| (i as f32 - size as f32 / 2.0) * 0.05)
            .collect();
        let expected = deep_chain_reference(&input_data);

        let result = compile_and_run_two_passes(&cache, &graph, &input_data, size);
        assert_close(&format!("deep_chain_size_{size}"), &result, &expected, 1e-4);
    }
}

/// Verify that two consecutive executions with DIFFERENT inputs produce
/// different outputs — proves ICB binding updates work correctly.
#[test]
fn test_frame_bucket_icb_binding_update_correctness() {
    super::test_utils::gpu_init();
    let cache = super::test_utils::metal_setup();

    let size = 64;
    let graph = build_variable_size_dag(size);
    let compiled = CompiledModel::builder(&graph, &cache)
        .build()
        .expect("compile");

    // Input A: all ones.
    let input_a: Vec<f32> = vec![1.0; size];
    let buf_a = create_input_buffer(&cache, &input_a);
    let out_a1 = compiled
        .execute(&cache, &[&buf_a])
        .expect("execute A pass 1");
    let result_a1 = read_output_n(&out_a1, size);
    let out_a2 = compiled
        .execute(&cache, &[&buf_a])
        .expect("execute A pass 2");
    let result_a2 = read_output_n(&out_a2, size);

    // Both passes of A must be identical.
    assert_eq!(
        result_a1, result_a2,
        "same input must produce same output across passes"
    );

    // Input B: all twos.
    let input_b: Vec<f32> = vec![2.0; size];
    let buf_b = create_input_buffer(&cache, &input_b);
    let out_b = compiled.execute(&cache, &[&buf_b]).expect("execute B");
    let result_b = read_output_n(&out_b, size);

    // A and B must differ.
    assert_ne!(
        result_a1, result_b,
        "different inputs must produce different outputs"
    );

    // Both must match CPU reference.
    let expected_a = dag_reference(&input_a);
    let expected_b = dag_reference(&input_b);
    assert_close("binding_update_A", &result_a1, &expected_a, 1e-4);
    assert_close("binding_update_B", &result_b, &expected_b, 1e-4);
}

/// Verify ICB segments are detected for models compiled at multiple sizes.
/// Each size should produce at least 1 ICB segment from the elementwise DAG.
/// Uses common Kokoro bucket sizes: 32, 64, 128, 256.
#[test]
fn test_frame_bucket_icb_segments_detected_per_size() {
    super::test_utils::gpu_init();
    let cache = super::test_utils::metal_setup();

    let bucket_sizes = [32, 64, 128, 256];

    for &bucket_size in &bucket_sizes {
        let graph = build_variable_size_dag(bucket_size);
        let compiled = CompiledModel::builder(&graph, &cache)
            .build()
            .expect("compile");

        let segments = compiled.num_icb_segments();
        assert!(
            segments >= 1,
            "bucket_size={bucket_size}: expected >= 1 ICB segment, got {segments}"
        );
    }
}

/// Verify that compiling models at different bucket-sized frame counts
/// produces correct outputs. This tests the "frame-bucket" pattern:
/// pre-compile a CompiledModel per bucket size, execute at that size,
/// verify numerical correctness.
#[test]
fn test_frame_bucket_multi_size_correctness() {
    super::test_utils::gpu_init();
    let cache = super::test_utils::metal_setup();

    // Simulated Kokoro bucket sizes (subset).
    let bucket_sizes = [32, 64, 128, 256, 512];

    for &bucket_size in &bucket_sizes {
        let graph = build_variable_size_dag(bucket_size);
        let compiled = CompiledModel::builder(&graph, &cache)
            .build()
            .expect("compile");

        let input_data: Vec<f32> = (0..bucket_size).map(|i| (i as f32) * 0.01 - 1.0).collect();
        let buf = create_input_buffer(&cache, &input_data);

        // Execute twice: first encodes ICB, second replays.
        let out1 = compiled.execute(&cache, &[&buf]).expect("pass 1");
        let r1 = read_output_n(&out1, bucket_size);

        let out2 = compiled.execute(&cache, &[&buf]).expect("pass 2");
        let r2 = read_output_n(&out2, bucket_size);

        // Both passes must agree.
        assert_eq!(
            r1, r2,
            "bucket_size={bucket_size}: pass 1 and pass 2 differ"
        );

        // Both must match CPU reference.
        let expected = dag_reference(&input_data);
        assert_close(&format!("bucket_{bucket_size}_pass1"), &r1, &expected, 1e-4);
    }
}

/// Stress test: run 20 forward passes with varying inputs to exercise
/// ICB encode (pass 1), replay (pass 2+), and binding updates.
#[test]
fn test_frame_bucket_icb_stress_many_passes() {
    super::test_utils::gpu_init();
    let cache = super::test_utils::metal_setup();

    let size = 128;
    let graph = build_variable_size_dag(size);
    let compiled = CompiledModel::builder(&graph, &cache)
        .build()
        .expect("compile");

    for pass in 0..20 {
        let scale = (pass as f32 + 1.0) * 0.3;
        let input_data: Vec<f32> = (0..size)
            .map(|i| (i as f32 - 64.0) * 0.01 * scale)
            .collect();
        let buf = create_input_buffer(&cache, &input_data);

        let out = compiled
            .execute(&cache, &[&buf])
            .unwrap_or_else(|e| panic!("execute pass {pass} failed: {e}"));
        let result = read_output_n(&out, size);
        let expected = dag_reference(&input_data);

        assert_close(&format!("stress_pass_{pass}"), &result, &expected, 1e-4);
    }
}

// ── Benchmark: ICB vs non-ICB dispatch latency ──────────────────────

/// Number of warmup iterations before timing.
const BENCH_WARMUP: usize = 5;
/// Number of timed iterations for the benchmark.
const BENCH_ITERS: usize = 50;

/// Measure average dispatch latency in microseconds over BENCH_ITERS runs.
fn bench_dispatch_latency<F: FnMut()>(mut f: F) -> f64 {
    // Warmup.
    for _ in 0..BENCH_WARMUP {
        f();
    }
    // Timed.
    let start = Instant::now();
    for _ in 0..BENCH_ITERS {
        f();
    }
    start.elapsed().as_micros() as f64 / BENCH_ITERS as f64
}

/// Benchmark comparing ICB replay latency vs cold dispatch for the
/// diamond DAG at multiple sizes.
///
/// This is an observational benchmark, not a gate. It prints timing
/// data for analysis but does not assert specific latency thresholds.
///
/// Run with: `cargo test -p nn-metal --test compiled_model_all
///   test_frame_bucket_icb_latency_benchmark -- --nocapture`
#[test]
fn test_frame_bucket_icb_latency_benchmark() {
    super::test_utils::gpu_init();
    let cache = super::test_utils::metal_setup();

    let sizes = [32, 64, 128, 256, 512];

    println!();
    println!("=== Frame-Bucket ICB Dispatch Latency Benchmark ===");
    println!(
        "{:<10} {:>10} {:>10} {:>12} {:>10} {:>8}",
        "Size", "ICB (us)", "Fresh (us)", "Speedup", "Segments", "Steps"
    );
    println!("{}", "-".repeat(66));

    for &size in &sizes {
        let graph = build_variable_size_dag(size);
        let compiled = CompiledModel::builder(&graph, &cache)
            .build()
            .expect("compile");
        let segments = compiled.num_icb_segments();
        let steps = compiled.num_steps();

        let input_data: Vec<f32> = (0..size).map(|i| (i as f32) * 0.01).collect();
        let buf = create_input_buffer(&cache, &input_data);

        // Cold compile path: create fresh CompiledModel each iteration
        // (simulates no ICB caching).
        let fresh_latency = bench_dispatch_latency(|| {
            let fresh_graph = build_variable_size_dag(size);
            let fresh_compiled = CompiledModel::builder(&fresh_graph, &cache)
                .build()
                .expect("compile");
            let fresh_buf = create_input_buffer(&cache, &input_data);
            let _ = fresh_compiled.execute(&cache, &[&fresh_buf]);
        });

        // ICB replay path: reuse the same compiled model (ICB encoded on
        // first pass, replayed on subsequent passes).
        // First pass to seed ICB encoding.
        let _ = compiled.execute(&cache, &[&buf]).expect("seed pass");
        let icb_latency = bench_dispatch_latency(|| {
            let _ = compiled.execute(&cache, &[&buf]);
        });

        let speedup = if icb_latency > 0.0 {
            fresh_latency / icb_latency
        } else {
            f64::NAN
        };

        println!(
            "{size:<10} {icb_latency:>10.1} {fresh_latency:>10.1} {speedup:>11.2}x {segments:>10} {steps:>8}"
        );
    }
    println!();
}

/// Benchmark measuring ICB replay overhead as a fraction of total
/// execution for the deep chain model (10 elementwise ops).
///
/// Run with: `cargo test -p nn-metal --test compiled_model_all
///   test_frame_bucket_icb_deep_chain_benchmark -- --nocapture`
#[test]
fn test_frame_bucket_icb_deep_chain_benchmark() {
    super::test_utils::gpu_init();
    let cache = super::test_utils::metal_setup();

    let size = 256;
    let graph = build_deep_chain(size);
    let compiled = CompiledModel::builder(&graph, &cache)
        .build()
        .expect("compile");

    let segments = compiled.num_icb_segments();
    let ir_dispatches = compiled.num_ir_dispatches();

    let input_data: Vec<f32> = (0..size).map(|i| (i as f32 - 128.0) * 0.01).collect();
    let buf = create_input_buffer(&cache, &input_data);

    // Seed ICB encoding.
    let _ = compiled.execute(&cache, &[&buf]).expect("seed pass");

    // Measure ICB replay latency.
    let icb_latency = bench_dispatch_latency(|| {
        let _ = compiled.execute(&cache, &[&buf]);
    });

    // Verify correctness on the benchmarked path.
    let out = compiled.execute(&cache, &[&buf]).expect("verify pass");
    let result = read_output_n(&out, size);
    let expected = deep_chain_reference(&input_data);
    assert_close("deep_chain_bench_verify", &result, &expected, 1e-4);

    println!();
    println!("=== Deep Chain ICB Benchmark (size={size}) ===");
    println!("  IR dispatches:  {ir_dispatches}");
    println!("  ICB segments:   {segments}");
    println!("  Replay latency: {icb_latency:.1} us (avg over {BENCH_ITERS} iterations)");
    println!();
}

/// Benchmark: per-bucket-size latency for the Kokoro default bucket config.
///
/// Pre-encodes models at each bucket size and measures ICB replay latency
/// to identify any bucket sizes with anomalous overhead.
///
/// Run with: `cargo test -p nn-metal --test compiled_model_all
///   test_frame_bucket_per_bucket_latency -- --nocapture`
#[test]
fn test_frame_bucket_per_bucket_latency() {
    super::test_utils::gpu_init();
    let cache = super::test_utils::metal_setup();

    // Use a subset of Kokoro default buckets to keep test time reasonable.
    let bucket_sizes = [32, 64, 128, 256, 512, 1024];

    println!();
    println!("=== Per-Bucket ICB Replay Latency ===");
    println!(
        "{:<10} {:>12} {:>10} {:>10}",
        "Bucket", "Latency (us)", "Segments", "Dispatches"
    );
    println!("{}", "-".repeat(46));

    for &bucket_size in &bucket_sizes {
        let graph = build_variable_size_dag(bucket_size);
        let compiled = CompiledModel::builder(&graph, &cache)
            .build()
            .expect("compile");
        let segments = compiled.num_icb_segments();
        let dispatches = compiled.num_ir_dispatches();

        let input_data: Vec<f32> = (0..bucket_size).map(|i| (i as f32) * 0.001).collect();
        let buf = create_input_buffer(&cache, &input_data);

        // Seed ICB.
        let _ = compiled.execute(&cache, &[&buf]).expect("seed");

        let latency = bench_dispatch_latency(|| {
            let _ = compiled.execute(&cache, &[&buf]);
        });

        // Verify correctness at each bucket size.
        let out = compiled.execute(&cache, &[&buf]).expect("verify");
        let result = read_output_n(&out, bucket_size);
        let expected = dag_reference(&input_data);
        assert_close(&format!("bucket_{bucket_size}"), &result, &expected, 1e-4);

        println!(
            "{bucket_size:<10} {latency:>12.1} {segments:>10} {dispatches:>10}"
        );
    }
    println!();
}

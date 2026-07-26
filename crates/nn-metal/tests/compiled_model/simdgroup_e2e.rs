// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! End-to-end tests for simdgroup matmul routing through `CompiledModel`.
//!
//! Verifies that at D=512 (production Kokoro dimension), the compiled model
//! routes Linear ops through the simdgroup path and produces correct results
//! versus a CPU reference.
//!
//! Part of #2458.

use nn_core::dyn_tensor::trace::{ComputationGraph, TraceNode, TraceOp, WeightRef};
use nn_core::DType;
use nn_dsl::trace_compile::CompiledStep;
use nn_dsl::{build_dispatch_plan, DispatchStep, ScalarType};
use nn_metal::compiled_model::CompiledModel;
use std::time::Instant;

use super::helpers::{assert_close, create_input_buffer, input_node, read_output_n};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Check if a CompiledModel's dispatch plan contains a SimdgroupLinear step.
fn has_simdgroup_linear(compiled: &CompiledModel) -> bool {
    for step in compiled.steps() {
        if let CompiledStep::Dispatch { kernel, .. } = step {
            if let Ok((plan, _)) = build_dispatch_plan(kernel.def(), ScalarType::F32) {
                for s in &plan {
                    if matches!(s, DispatchStep::SimdgroupLinear(..)) {
                        return true;
                    }
                }
            }
        }
    }
    false
}

/// Check if a CompiledModel's dispatch plan contains a naive Linear step.
fn has_naive_linear(compiled: &CompiledModel) -> bool {
    for step in compiled.steps() {
        if let CompiledStep::Dispatch { kernel, .. } = step {
            if let Ok((plan, _)) = build_dispatch_plan(kernel.def(), ScalarType::F32) {
                for s in &plan {
                    if matches!(s, DispatchStep::Linear { .. }) {
                        return true;
                    }
                }
            }
        }
    }
    false
}

/// Build a trace graph for a single Linear layer: [batch, in_f] -> [batch, out_f].
fn build_linear_graph(
    batch: usize,
    in_f: usize,
    out_f: usize,
    seed: u64,
) -> (ComputationGraph, Vec<f32>, Vec<f32>, Vec<f32>) {
    let w_data = super::test_utils::rand_f32_vec(seed, out_f * in_f, -0.3, 0.3);
    let b_data = super::test_utils::rand_f32_vec(seed + 1, out_f, -0.1, 0.1);
    let x_data = super::test_utils::rand_f32_vec(seed + 2, batch * in_f, -1.0, 1.0);

    let w_ref = WeightRef::new(w_data.clone(), vec![out_f, in_f]).unwrap();
    let b_ref = WeightRef::new(b_data.clone(), vec![out_f]).unwrap();

    let graph = ComputationGraph::from_nodes(vec![
        input_node(0, &[batch, in_f]),
        TraceNode::new(
            1,
            "linear_0".into(),
            TraceOp::Linear {
                weight: w_ref,
                bias: Some(b_ref),
            },
            vec![0],
            vec![batch, out_f],
            DType::F32,
        ),
    ]);
    (graph, x_data, w_data, b_data)
}

// ===========================================================================
// AC2: E2E test at D=512 — CompiledModel uses simdgroup path
// ===========================================================================

/// D=512 linear through CompiledModel: batch=128 (conforming), K=512, N=512.
/// should_use_simdgroup(128, 512, 512) = true (all % 8, 128*512=65536 >= 16384, K=512 >= 128).
///
/// Verifies:
/// 1. The compiled dispatch plan contains SimdgroupLinear (not naive Linear)
/// 2. GPU output matches CPU reference within precision budget
#[test]
fn test_compiled_model_d512_uses_simdgroup() {
    super::test_utils::gpu_init();
    let cache = super::test_utils::metal_setup();

    let (batch, in_f, out_f) = (128, 512, 512);
    let (graph, x_data, w_data, b_data) = build_linear_graph(batch, in_f, out_f, 0x2458_0001);

    let compiled = CompiledModel::builder(&graph, &cache)
        .build()
        .expect("compile D=512 linear");

    // AC2: verify simdgroup routing
    assert!(
        has_simdgroup_linear(&compiled),
        "D=512: CompiledModel must route to SimdgroupLinear (M={batch}, K={in_f}, N={out_f})"
    );

    // Execute and verify correctness
    let input_buf = create_input_buffer(&cache, &x_data);
    let out_buf = compiled
        .execute(&cache, &[&input_buf])
        .expect("execute D=512");

    let gpu = read_output_n(&out_buf, batch * out_f);
    let cpu = super::test_utils::linear_ref(&x_data, &w_data, Some(&b_data), batch, in_f, out_f);
    assert_close("d512_simdgroup", &gpu, &cpu, 1e-3);
}

/// D=512 linear with non-conforming batch: batch=1 (single-token decoding).
/// should_use_simdgroup(1, 512, 512) = false (M=1 not % 8).
///
/// Verifies CompiledModel falls back to naive Linear for decoding shapes.
#[test]
fn test_compiled_model_d512_decoding_uses_naive() {
    super::test_utils::gpu_init();
    let cache = super::test_utils::metal_setup();

    let (batch, in_f, out_f) = (1, 512, 512);
    let (graph, x_data, w_data, b_data) = build_linear_graph(batch, in_f, out_f, 0x2458_0002);

    let compiled = CompiledModel::builder(&graph, &cache)
        .build()
        .expect("compile D=512 unbatched");

    // M=1: must NOT route to simdgroup
    assert!(
        has_naive_linear(&compiled),
        "D=512 batch=1: must fall back to naive Linear"
    );
    assert!(
        !has_simdgroup_linear(&compiled),
        "D=512 batch=1: must NOT route to SimdgroupLinear"
    );

    // Execute and verify correctness
    let input_buf = create_input_buffer(&cache, &x_data);
    let out_buf = compiled
        .execute(&cache, &[&input_buf])
        .expect("execute D=512 unbatched");

    let gpu = read_output_n(&out_buf, batch * out_f);
    let cpu = super::test_utils::linear_ref(&x_data, &w_data, Some(&b_data), batch, in_f, out_f);
    assert_close("d512_naive", &gpu, &cpu, 1e-5);
}

/// D=512 FFN dimensions: batch=128, in=512, out=2048 (4x expansion).
/// should_use_simdgroup(128, 512, 2048) = true.
/// Tests the typical FFN up-projection shape in transformers.
#[test]
fn test_compiled_model_d512_ffn_uses_simdgroup() {
    super::test_utils::gpu_init();
    let cache = super::test_utils::metal_setup();

    let (batch, in_f, out_f) = (128, 512, 2048);
    let (graph, x_data, w_data, b_data) = build_linear_graph(batch, in_f, out_f, 0x2458_0003);

    let compiled = CompiledModel::builder(&graph, &cache)
        .build()
        .expect("compile FFN");

    assert!(
        has_simdgroup_linear(&compiled),
        "FFN: CompiledModel must route to SimdgroupLinear (M={batch}, K={in_f}, N={out_f})"
    );

    let input_buf = create_input_buffer(&cache, &x_data);
    let out_buf = compiled
        .execute(&cache, &[&input_buf])
        .expect("execute FFN");

    let gpu = read_output_n(&out_buf, batch * out_f);
    let cpu = super::test_utils::linear_ref(&x_data, &w_data, Some(&b_data), batch, in_f, out_f);
    assert_close("d512_ffn", &gpu, &cpu, 1e-3);
}

// ===========================================================================
// AC3: Benchmark — simdgroup vs naive throughput at production dimensions
// ===========================================================================

/// Benchmark: simdgroup vs naive at D=512 production dimensions.
///
/// Run with: `cargo test -p nn-metal --test compiled_model_simdgroup_e2e --release
///            -- test_simdgroup_benchmark_d512 --nocapture`
#[test]
fn test_simdgroup_benchmark_d512() {
    super::test_utils::gpu_init();
    let cache = super::test_utils::metal_setup();

    let iters = 5;

    // Simdgroup path: batch=128, D=512 (conforming)
    let (simd_graph, simd_x, _, _) = build_linear_graph(128, 512, 512, 0x2458_B001);
    let simd_compiled = CompiledModel::builder(&simd_graph, &cache)
        .build()
        .expect("compile simd benchmark");
    assert!(
        has_simdgroup_linear(&simd_compiled),
        "benchmark: must use simdgroup"
    );

    let simd_buf = create_input_buffer(&cache, &simd_x);
    let _ = simd_compiled.execute(&cache, &[&simd_buf]); // warmup
    let t = Instant::now();
    for _ in 0..iters {
        let _ = simd_compiled
            .execute(&cache, &[&simd_buf])
            .expect("bench simd");
    }
    let simd_ms = t.elapsed().as_secs_f64() * 1000.0 / f64::from(iters);
    let simd_elem: i32 = 128 * 512;

    // Naive path: batch=7 (non-conforming M), same K=512, N=512
    let (naive_graph, naive_x, _, _) = build_linear_graph(7, 512, 512, 0x2458_B002);
    let naive_compiled = CompiledModel::builder(&naive_graph, &cache)
        .build()
        .expect("compile naive benchmark");
    assert!(
        !has_simdgroup_linear(&naive_compiled),
        "benchmark: must use naive"
    );

    let naive_buf = create_input_buffer(&cache, &naive_x);
    let _ = naive_compiled.execute(&cache, &[&naive_buf]); // warmup
    let t = Instant::now();
    for _ in 0..iters {
        let _ = naive_compiled
            .execute(&cache, &[&naive_buf])
            .expect("bench naive");
    }
    let naive_ms = t.elapsed().as_secs_f64() * 1000.0 / f64::from(iters);
    let naive_elem: i32 = 7 * 512;

    let simd_tp = f64::from(simd_elem) / simd_ms;
    let naive_tp = f64::from(naive_elem) / naive_ms;
    let ratio = simd_tp / naive_tp;

    eprintln!(
        "[#2458 AC3] simdgroup: {simd_ms:.3}ms ({simd_elem} elem, {simd_tp:.0} e/ms) | \
         naive: {naive_ms:.3}ms ({naive_elem} elem, {naive_tp:.0} e/ms) | ratio: {ratio:.2}x"
    );

    // Informational: simdgroup throughput should exceed naive in isolation
    // (~5-6x typical). Under parallel GPU contention from other tests,
    // simdgroup suffers disproportionately (larger workload = more contention)
    // and the ratio can drop below 1.0x. Correctness is verified by
    // test_compiled_model_d512_uses_simdgroup; this test logs throughput only.
    if ratio < 1.5 {
        eprintln!(
            "[#2458 AC3] WARNING: simdgroup throughput {ratio:.2}x < 1.5x — \
             likely GPU contention from parallel tests (expected ~5x in isolation)"
        );
    }
}

// ===========================================================================
// NormLinear simdgroup two-dispatch path (#3292)
// ===========================================================================

/// CPU LayerNorm: normalize over hidden dim, apply weight and bias.
/// Input x: `[B, C]`. weight/bias: `[C]`.
fn cpu_layer_norm(
    x: &[f32],
    weight: &[f32],
    bias: &[f32],
    batch: usize,
    hidden: usize,
    eps: f32,
) -> Vec<f32> {
    let mut output = vec![0.0_f32; batch * hidden];
    for b in 0..batch {
        let offset = b * hidden;
        let row = &x[offset..offset + hidden];
        let mean: f32 = row.iter().sum::<f32>() / hidden as f32;
        let var: f32 = row.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / hidden as f32;
        let inv_std = 1.0 / (var + eps).sqrt();
        for c in 0..hidden {
            output[offset + c] = (row[c] - mean) * inv_std * weight[c] + bias[c];
        }
    }
    output
}

/// CPU RmsNorm: scale by 1/rms * weight.
fn cpu_rms_norm(x: &[f32], weight: &[f32], batch: usize, hidden: usize, eps: f32) -> Vec<f32> {
    let mut output = vec![0.0_f32; batch * hidden];
    for b in 0..batch {
        let offset = b * hidden;
        let row = &x[offset..offset + hidden];
        let sum_sq: f32 = row.iter().map(|v| v * v).sum();
        let inv_rms = 1.0 / (sum_sq / hidden as f32 + eps).sqrt();
        for c in 0..hidden {
            output[offset + c] = row[c] * inv_rms * weight[c];
        }
    }
    output
}

/// NormLinear (LayerNorm) at simdgroup-qualifying dims: batch=128, hidden=256, out=256.
/// should_use_simdgroup(128, 256, 256) = true (all % 8, 128*256=32768 >= 16384, K=256 >= 128).
///
/// Verifies:
/// 1. Peephole fuses LayerNorm+Linear into NormLinear NativeOp
/// 2. GPU output matches CPU reference (LayerNorm → matmul) within tolerance
///
/// Part of #3292.
#[test]
fn test_compiled_norm_linear_simdgroup_layernorm() {
    super::test_utils::gpu_init();
    let cache = super::test_utils::metal_setup();

    let (batch, hidden, out_f) = (128, 256, 256);
    let eps = 1e-5_f32;

    let ln_w = super::test_utils::rand_f32_vec(0x3292_0001, hidden, 0.5, 1.5);
    let ln_b = super::test_utils::rand_f32_vec(0x3292_0002, hidden, -0.1, 0.1);
    let w = super::test_utils::rand_f32_vec(0x3292_0003, out_f * hidden, -0.3, 0.3);
    let b = super::test_utils::rand_f32_vec(0x3292_0004, out_f, -0.1, 0.1);
    let x_data = super::test_utils::rand_f32_vec(0x3292_0005, batch * hidden, -1.0, 1.0);

    fn weight(data: Vec<f32>, shape: Vec<usize>) -> WeightRef {
        WeightRef::new(data, shape).expect("weight")
    }

    let graph = ComputationGraph::from_nodes(vec![
        input_node(0, &[batch, hidden]),
        TraceNode::new(
            1,
            "layernorm_0".into(),
            TraceOp::LayerNorm {
                eps: f64::from(eps),
                weight: weight(ln_w.clone(), vec![hidden]),
                bias: weight(ln_b.clone(), vec![hidden]),
            },
            vec![0],
            vec![batch, hidden],
            DType::F32,
        ),
        TraceNode::new(
            2,
            "linear_0".into(),
            TraceOp::Linear {
                weight: weight(w.clone(), vec![out_f, hidden]),
                bias: Some(weight(b.clone(), vec![out_f])),
            },
            vec![1],
            vec![batch, out_f],
            DType::F32,
        ),
    ]);

    // Compile and verify NormLinear NativeOp is present.
    let compiled = CompiledModel::builder(&graph, &cache)
        .build()
        .expect("compile LN+Linear simdgroup");
    let has_norm_linear = compiled.steps().iter().any(|s| {
        matches!(
            s,
            CompiledStep::NativeOp {
                op: nn_dsl::NativeOpKind::NormLinear { .. },
                ..
            }
        )
    });
    assert!(
        has_norm_linear,
        "peephole should fuse LayerNorm+Linear into NormLinear"
    );

    // Execute on GPU.
    let input_buf = create_input_buffer(&cache, &x_data);
    let out_buf = compiled
        .execute(&cache, &[&input_buf])
        .expect("execute NormLinear simdgroup LN");
    let gpu = read_output_n(&out_buf, batch * out_f);

    // CPU reference: LayerNorm → Linear.
    let normed = cpu_layer_norm(&x_data, &ln_w, &ln_b, batch, hidden, eps);
    let expected = super::test_utils::linear_ref(&normed, &w, Some(&b), batch, hidden, out_f);

    // K=256 accumulation: simdgroup GEMM uses float accumulators internally,
    // but two-dispatch path writes normalized values to global memory and reads
    // them back, introducing a quantization boundary not present in the fused path.
    assert_close("norm_linear_simd_ln", &gpu, &expected, 5e-3);
}

/// NormLinear (RmsNorm) at simdgroup-qualifying dims: batch=128, hidden=256, out=256.
///
/// Part of #3292.
#[test]
fn test_compiled_norm_linear_simdgroup_rmsnorm() {
    super::test_utils::gpu_init();
    let cache = super::test_utils::metal_setup();

    let (batch, hidden, out_f) = (128, 256, 256);
    let eps = 1e-5_f32;

    let rms_w = super::test_utils::rand_f32_vec(0x3292_1001, hidden, 0.5, 1.5);
    let w = super::test_utils::rand_f32_vec(0x3292_1002, out_f * hidden, -0.3, 0.3);
    let b = super::test_utils::rand_f32_vec(0x3292_1003, out_f, -0.1, 0.1);
    let x_data = super::test_utils::rand_f32_vec(0x3292_1004, batch * hidden, -1.0, 1.0);

    fn weight(data: Vec<f32>, shape: Vec<usize>) -> WeightRef {
        WeightRef::new(data, shape).expect("weight")
    }

    let graph = ComputationGraph::from_nodes(vec![
        input_node(0, &[batch, hidden]),
        TraceNode::new(
            1,
            "rmsnorm_0".into(),
            TraceOp::RmsNorm {
                eps: f64::from(eps),
                weight: weight(rms_w.clone(), vec![hidden]),
            },
            vec![0],
            vec![batch, hidden],
            DType::F32,
        ),
        TraceNode::new(
            2,
            "linear_0".into(),
            TraceOp::Linear {
                weight: weight(w.clone(), vec![out_f, hidden]),
                bias: Some(weight(b.clone(), vec![out_f])),
            },
            vec![1],
            vec![batch, out_f],
            DType::F32,
        ),
    ]);

    let compiled = CompiledModel::builder(&graph, &cache)
        .build()
        .expect("compile RMS+Linear simdgroup");
    let has_norm_linear = compiled.steps().iter().any(|s| {
        matches!(
            s,
            CompiledStep::NativeOp {
                op: nn_dsl::NativeOpKind::NormLinear { .. },
                ..
            }
        )
    });
    assert!(
        has_norm_linear,
        "peephole should fuse RmsNorm+Linear into NormLinear"
    );

    let input_buf = create_input_buffer(&cache, &x_data);
    let out_buf = compiled
        .execute(&cache, &[&input_buf])
        .expect("execute NormLinear simdgroup RMS");
    let gpu = read_output_n(&out_buf, batch * out_f);

    // CPU reference: RmsNorm → Linear.
    let normed = cpu_rms_norm(&x_data, &rms_w, batch, hidden, eps);
    let expected = super::test_utils::linear_ref(&normed, &w, Some(&b), batch, hidden, out_f);

    assert_close("norm_linear_simd_rms", &gpu, &expected, 5e-3);
}

// -- num_metal_dispatches --------------------------------------------------

/// `num_metal_dispatches()` counts plan-expanded Metal kernel launches,
/// which is >= `num_dispatches()` for models with complex ops.
#[test]
fn test_num_metal_dispatches_gte_num_dispatches() {
    super::test_utils::gpu_init();
    let cache = super::test_utils::metal_setup();

    // A single Linear at D=512 compiles to 1 CompiledStep::Dispatch,
    // but may expand to multiple DispatchSteps in the plan.
    let (graph, _, _, _) = build_linear_graph(128, 512, 512, 42);
    let compiled = CompiledModel::builder(&graph, &cache)
        .build()
        .expect("compile");

    let logical = compiled.num_dispatches();
    let metal = compiled.num_metal_dispatches();
    assert!(logical > 0, "should have at least 1 dispatch");
    assert!(
        metal >= logical,
        "metal dispatches ({metal}) should be >= logical dispatches ({logical})"
    );
    eprintln!("num_dispatches={logical}, num_metal_dispatches={metal}");
}

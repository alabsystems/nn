// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Simdgroup GEMM integration tests for the compiled tensor dispatch pipeline.
//!
//! Verifies that the compiled pipeline:
//! - Routes conforming shapes to simdgroup kernels (AC1, covered in dispatch_coverage.rs)
//! - Falls back to naive kernels for non-conforming shapes with correct results (AC3)
//! - Produces faster execution via simdgroup vs naive for large matmuls (AC2)
//!
//! Part of #2275.

use super::test_utils::{assert_within_budget, linear_ref, matmul_ref, metal_setup, rand_f32_vec};
use nn_dsl::linear::{build_linear, build_linear_batched};
use nn_dsl::{build_dispatch_plan, ScalarType};
use nn_metal::{execute_tensor_dispatch, PipelineCache};
use std::collections::HashMap;
use std::time::Instant;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn step_tag(step: &nn_dsl::DispatchStep) -> &'static str {
    match step {
        nn_dsl::DispatchStep::Linear { .. } => "Linear",
        nn_dsl::DispatchStep::MatMul { .. } => "MatMul",
        nn_dsl::DispatchStep::SimdgroupLinear(..) => "SimdgroupLinear",
        nn_dsl::DispatchStep::SimdgroupMatMul(..) => "SimdgroupMatMul",
        nn_dsl::DispatchStep::TiledLinear(..) => "TiledLinear",
        nn_dsl::DispatchStep::TiledMatMul(..) => "TiledMatMul",
        _ => "Other",
    }
}

fn plan_has_tag(kernel: &nn_dsl::TensorKernelDef, expected: &str) -> bool {
    let (plan, _) = build_dispatch_plan(kernel, ScalarType::F32).expect("plan");
    plan.iter().any(|s| step_tag(s) == expected)
}

// ===========================================================================
// AC3: Non-conforming shapes fall back to naive and produce correct results
// ===========================================================================

/// Non-conforming linear (M=7): M not divisible by 8 → naive Linear dispatch.
/// Verifies routing AND correctness against CPU reference.
#[test]
fn test_simdgroup_fallback_linear_odd_batch() {
    let cache = metal_setup();
    let (batch, in_f, out_f) = (7, 128, 256);
    let def =
        build_linear_batched("fallback_odd_batch", batch, in_f, out_f, true).expect("build linear");

    // Verify routing: must NOT produce SimdgroupLinear (M=7 not % 8)
    assert!(
        plan_has_tag(&def, "Linear"),
        "M=7: must fall back to naive Linear"
    );
    assert!(
        !plan_has_tag(&def, "SimdgroupLinear"),
        "M=7: must NOT route to SimdgroupLinear"
    );

    // Execute and verify correctness
    let x = rand_f32_vec(0xAC3_0001, batch * in_f, -1.0, 1.0);
    let w = rand_f32_vec(0xAC3_0002, out_f * in_f, -0.3, 0.3);
    let b = rand_f32_vec(0xAC3_0003, out_f, -0.1, 0.1);
    let cpu = linear_ref(&x, &w, Some(&b), batch, in_f, out_f);

    let mut inputs = HashMap::new();
    inputs.insert("data", x);
    inputs.insert("weight", w);
    inputs.insert("bias", b);
    let gpu = execute_tensor_dispatch(&cache, &def, ScalarType::F32, &inputs)
        .expect("naive linear dispatch");
    assert_eq!(gpu.len(), batch * out_f);
    assert_within_budget("fallback_odd_batch", &gpu, &cpu);
}

/// Non-conforming linear (K=64 < 128): K too small for simdgroup → tiled dispatch.
/// M=128, K=64, N=256 meets tiled criteria (M≥16, N≥16, K≥8) but not simdgroup (K<128).
#[test]
fn test_simdgroup_fallback_linear_small_k() {
    let cache = metal_setup();
    let (batch, in_f, out_f) = (128, 64, 256);
    let def =
        build_linear_batched("fallback_small_k", batch, in_f, out_f, false).expect("build linear");

    assert!(
        plan_has_tag(&def, "TiledLinear"),
        "K=64: must route to TiledLinear (not simdgroup, meets tiled criteria)"
    );
    assert!(
        !plan_has_tag(&def, "SimdgroupLinear"),
        "K=64: must NOT route to SimdgroupLinear"
    );

    let x = rand_f32_vec(0xAC3_1001, batch * in_f, -1.0, 1.0);
    let w = rand_f32_vec(0xAC3_1002, out_f * in_f, -0.3, 0.3);
    let cpu = linear_ref(&x, &w, None, batch, in_f, out_f);

    let mut inputs = HashMap::new();
    inputs.insert("data", x);
    inputs.insert("weight", w);
    let gpu =
        execute_tensor_dispatch(&cache, &def, ScalarType::F32, &inputs).expect("tiled linear K=64");
    assert_eq!(gpu.len(), batch * out_f);
    assert_within_budget("fallback_small_k", &gpu, &cpu);
}

/// Non-conforming linear (M*N < 16384): area too small → naive Linear dispatch.
#[test]
fn test_simdgroup_fallback_linear_small_area() {
    let cache = metal_setup();
    // M=8, N=8 → M*N = 64 < 16384. K=128 is fine.
    let (batch, in_f, out_f) = (8, 128, 8);
    let def = build_linear_batched("fallback_small_area", batch, in_f, out_f, true)
        .expect("build linear");

    assert!(
        plan_has_tag(&def, "Linear"),
        "M*N=64: must fall back to naive Linear"
    );

    let x = rand_f32_vec(0xAC3_2001, batch * in_f, -1.0, 1.0);
    let w = rand_f32_vec(0xAC3_2002, out_f * in_f, -0.3, 0.3);
    let b = rand_f32_vec(0xAC3_2003, out_f, -0.1, 0.1);
    let cpu = linear_ref(&x, &w, Some(&b), batch, in_f, out_f);

    let mut inputs = HashMap::new();
    inputs.insert("data", x);
    inputs.insert("weight", w);
    inputs.insert("bias", b);
    let gpu = execute_tensor_dispatch(&cache, &def, ScalarType::F32, &inputs)
        .expect("naive linear small area");
    assert_eq!(gpu.len(), batch * out_f);
    assert_within_budget("fallback_small_area", &gpu, &cpu);
}

/// Non-conforming unbatched linear (build_linear): M=1 (single sample) → naive.
#[test]
fn test_simdgroup_fallback_unbatched_linear() {
    let cache = metal_setup();
    // build_linear creates rank-1 input [in_features], so batch_size=1.
    // M=1 is not divisible by 8: always naive.
    let (in_f, out_f) = (256, 512);
    let def = build_linear("fallback_unbatched", in_f, out_f, true).expect("build");

    assert!(
        plan_has_tag(&def, "Linear"),
        "unbatched M=1: must fall back to naive"
    );

    let x = rand_f32_vec(0xAC3_3001, in_f, -1.0, 1.0);
    let w = rand_f32_vec(0xAC3_3002, out_f * in_f, -0.3, 0.3);
    let b = rand_f32_vec(0xAC3_3003, out_f, -0.1, 0.1);
    let cpu = linear_ref(&x, &w, Some(&b), 1, in_f, out_f);

    let mut inputs = HashMap::new();
    inputs.insert("data", x);
    inputs.insert("weight", w);
    inputs.insert("bias", b);
    let gpu =
        execute_tensor_dispatch(&cache, &def, ScalarType::F32, &inputs).expect("unbatched linear");
    assert_eq!(gpu.len(), out_f);
    assert_within_budget("fallback_unbatched", &gpu, &cpu);
}

// ===========================================================================
// AC3 bonus: conforming shapes route to simdgroup and produce correct results
// ===========================================================================

/// Conforming linear: M=128, K=128, N=128 → SimdgroupLinear.
/// Verifies routing and correctness.
#[test]
fn test_simdgroup_linear_conforming_correctness() {
    let cache = metal_setup();
    let (batch, in_f, out_f) = (128, 128, 128);
    let def =
        build_linear_batched("simd_lin_correct", batch, in_f, out_f, true).expect("build linear");

    assert!(
        plan_has_tag(&def, "SimdgroupLinear"),
        "conforming 128×128×128: must route to SimdgroupLinear"
    );

    let x = rand_f32_vec(0xAC3_4001, batch * in_f, -1.0, 1.0);
    let w = rand_f32_vec(0xAC3_4002, out_f * in_f, -0.3, 0.3);
    let b = rand_f32_vec(0xAC3_4003, out_f, -0.1, 0.1);
    let cpu = linear_ref(&x, &w, Some(&b), batch, in_f, out_f);

    let mut inputs = HashMap::new();
    inputs.insert("data", x);
    inputs.insert("weight", w);
    inputs.insert("bias", b);
    let gpu =
        execute_tensor_dispatch(&cache, &def, ScalarType::F32, &inputs).expect("simdgroup linear");
    assert_eq!(gpu.len(), batch * out_f);
    assert_within_budget("simd_lin_correct", &gpu, &cpu);
}

/// Build a matmul kernel def: left [m, k] @ right [k, n] → [m, n].
fn build_matmul_def(name: &str, m: usize, k: usize, n: usize) -> nn_dsl::TensorKernelDef {
    use nn_dsl::tensor_ir::{TensorKernelDef, TensorNode, TensorNodeId, TensorOpKind};
    TensorKernelDef::new(
        name,
        vec![
            TensorNode::new(
                TensorNodeId::new(0),
                TensorOpKind::Input {
                    name: "left".into(),
                    shape: vec![m, k],
                },
                vec![m, k],
            ),
            TensorNode::new(
                TensorNodeId::new(1),
                TensorOpKind::Input {
                    name: "right".into(),
                    shape: vec![k, n],
                },
                vec![k, n],
            ),
            TensorNode::new(
                TensorNodeId::new(2),
                TensorOpKind::MatMul {
                    left: TensorNodeId::new(0),
                    right: TensorNodeId::new(1),
                    transpose_right: false,
                    scale: None,
                },
                vec![m, n],
            ),
        ],
        TensorNodeId::new(2),
    )
}

/// Conforming matmul: M=128, K=256, N=512 → SimdgroupMatMul.
/// Verifies routing and correctness.
#[test]
fn test_simdgroup_matmul_conforming_correctness() {
    let cache = metal_setup();
    let (m, k, n) = (128, 256, 512);
    let def = build_matmul_def("simd_mm_correct", m, k, n);

    assert!(
        plan_has_tag(&def, "SimdgroupMatMul"),
        "128×256×512: must route to SimdgroupMatMul"
    );

    let left = rand_f32_vec(0xAC3_5001, m * k, -1.0, 1.0);
    let right = rand_f32_vec(0xAC3_5002, k * n, -1.0, 1.0);
    let cpu = matmul_ref(&left, &right, m, k, n, false, None);

    let mut inputs = HashMap::new();
    inputs.insert("left", left);
    inputs.insert("right", right);
    let gpu =
        execute_tensor_dispatch(&cache, &def, ScalarType::F32, &inputs).expect("simdgroup matmul");
    assert_eq!(gpu.len(), m * n);
    assert_within_budget("simd_mm_correct", &gpu, &cpu);
}

// ===========================================================================
// AC2: Benchmark — simdgroup vs naive for FFN-scale compiled pipeline
// ===========================================================================

/// Time `iters` dispatches of a linear kernel, returning (ms_per_iter, output_elements).
fn bench_linear(
    cache: &PipelineCache,
    name: &str,
    batch: usize,
    in_f: usize,
    out_f: usize,
    seed_base: u64,
    iters: i32,
) -> (f64, usize) {
    let def = build_linear_batched(name, batch, in_f, out_f, false).expect("build linear");
    let x = rand_f32_vec(seed_base, batch * in_f, -1.0, 1.0);
    let w = rand_f32_vec(seed_base + 1, out_f * in_f, -0.3, 0.3);
    let mut inputs = HashMap::new();
    inputs.insert("data", x);
    inputs.insert("weight", w);

    let _ = execute_tensor_dispatch(cache, &def, ScalarType::F32, &inputs); // warmup
    let t = Instant::now();
    for _ in 0..iters {
        let _ = execute_tensor_dispatch(cache, &def, ScalarType::F32, &inputs).expect("bench");
    }
    let ms = t.elapsed().as_secs_f64() * 1000.0 / f64::from(iters);
    (ms, batch * out_f)
}

/// Benchmark: compiled pipeline simdgroup vs naive Linear at FFN scale.
///
/// Run with: `cargo test -p nn-metal --test tensor_dispatch_simdgroup --release
///            -- test_simdgroup_vs_naive_benchmark --nocapture`
#[test]
fn test_simdgroup_vs_naive_benchmark() {
    let cache = metal_setup();

    // Simdgroup path: M=128, K=256, N=512 (conforming)
    let (simd_ms, simd_elem) = bench_linear(&cache, "bench_simd", 128, 256, 512, 0xBE_0001, 3);
    // Naive path: M=7, K=256, N=512 (non-conforming M not % 8)
    let (naive_ms, naive_elem) = bench_linear(&cache, "bench_naive", 7, 256, 512, 0xBE_1001, 3);

    let simd_tp = simd_elem as f64 / simd_ms;
    let naive_tp = naive_elem as f64 / naive_ms;
    let ratio = simd_tp / naive_tp;

    eprintln!(
        "[#2275 AC2] simdgroup: {simd_ms:.3}ms ({simd_elem} elem, {simd_tp:.0} e/ms) | \
         naive: {naive_ms:.3}ms ({naive_elem} elem, {naive_tp:.0} e/ms) | ratio: {ratio:.2}x"
    );

    assert!(
        ratio > 1.5,
        "simdgroup throughput {ratio:.2}x should be > 1.5x vs naive"
    );
}

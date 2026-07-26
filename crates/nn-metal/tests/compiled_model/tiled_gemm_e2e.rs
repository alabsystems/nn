// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! End-to-end tests for tiled shared-memory GEMM: routing assertions and
//! batched matmul (z-grid dispatch).
//!
//! Complements `ops_e2e_8.rs` which covers aligned/non-aligned/no-bias shapes.
//! This file adds:
//! - Routing assertions (`has_tiled_linear`, `has_tiled_matmul`)
//! - Batched 3D matmul exercising z-grid dispatch
//! - F16 autocast vs F32 compiled baseline comparison
//!
//! Part of #3230 (Gap 1, D6).

use nn_core::dyn_tensor::trace::{ComputationGraph, TraceNode, TraceOp, WeightRef};
use nn_core::DType;
use nn_dsl::trace_compile::CompiledStep;
use nn_dsl::{build_dispatch_plan, DispatchStep, ScalarType};
use nn_metal::compiled_model::CompiledModel;

use super::helpers::{assert_close, create_input_buffer, input_node, read_output_n};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Check if a CompiledModel's dispatch plan contains a TiledLinear step.
fn has_tiled_linear(compiled: &CompiledModel) -> bool {
    for step in compiled.steps() {
        if let CompiledStep::Dispatch { kernel, .. } = step {
            if let Ok((plan, _)) = build_dispatch_plan(kernel.def(), ScalarType::F32) {
                for s in &plan {
                    if matches!(s, DispatchStep::TiledLinear(..)) {
                        return true;
                    }
                }
            }
        }
    }
    false
}

/// Check if a CompiledModel's dispatch plan contains a TiledMatMul step.
fn has_tiled_matmul(compiled: &CompiledModel) -> bool {
    for step in compiled.steps() {
        if let CompiledStep::Dispatch { kernel, .. } = step {
            if let Ok((plan, _)) = build_dispatch_plan(kernel.def(), ScalarType::F32) {
                for s in &plan {
                    if matches!(s, DispatchStep::TiledMatMul(..)) {
                        return true;
                    }
                }
            }
        }
    }
    false
}

// ---------------------------------------------------------------------------
// Test 1: Routing assertion — tiled linear dispatch
// ---------------------------------------------------------------------------

/// Linear [32, 100] -> [32, 64]: K=100 not %8 fails simdgroup, routes to tiled.
/// Asserts the dispatch plan contains `TiledLinear`, not just correctness.
#[test]
fn test_tiled_linear_routing() {
    super::test_utils::gpu_init();
    let cache = super::test_utils::metal_setup();

    let (batch, in_f, out_f) = (32, 100, 64);
    let w_data = super::test_utils::rand_f32_vec(0x7110_0010, out_f * in_f, -0.3, 0.3);
    let b_data = super::test_utils::rand_f32_vec(0x7110_0011, out_f, -0.1, 0.1);
    let x_data = super::test_utils::rand_f32_vec(0x7110_0012, batch * in_f, -1.0, 1.0);

    let graph = ComputationGraph::from_nodes(vec![
        input_node(0, &[batch, in_f]),
        TraceNode::new(
            1,
            "linear_tiled".into(),
            TraceOp::Linear {
                weight: WeightRef::new(w_data.clone(), vec![out_f, in_f]).unwrap(),
                bias: Some(WeightRef::new(b_data.clone(), vec![out_f]).unwrap()),
            },
            vec![0],
            vec![batch, out_f],
            DType::F32,
        ),
    ]);

    let compiled = CompiledModel::builder(&graph, &cache)
        .build()
        .expect("compile");

    assert!(
        has_tiled_linear(&compiled),
        "linear [32,100]->[32,64] must route to TiledLinear (K=100 not % 8)"
    );

    let input_buf = create_input_buffer(&cache, &x_data);
    let out_buf = compiled.execute(&cache, &[&input_buf]).expect("execute");
    let gpu = read_output_n(&out_buf, batch * out_f);

    let cpu = super::test_utils::linear_ref(&x_data, &w_data, Some(&b_data), batch, in_f, out_f);
    assert_close("tiled_linear_routing", &gpu, &cpu, 1e-3);
}

// ---------------------------------------------------------------------------
// Test 2: Batched 3D matmul — z-grid dispatch
// ---------------------------------------------------------------------------

/// Attention QK^T: [8, 64, 64] x [8, 64, 64] — 8 batches via z-grid.
/// Tests that `tg_pos.z` batch indexing works correctly in the tiled kernel.
/// ops_e2e_8 only tests unbatched 2D matmul; this exercises the batch path.
#[test]
fn test_tiled_matmul_batched_attention() {
    super::test_utils::gpu_init();
    let cache = super::test_utils::metal_setup();

    let (batch, m, k, n) = (8, 64, 64, 64);
    let left_data = super::test_utils::rand_f32_vec(0x7110_0001, batch * m * k, -1.0, 1.0);
    let right_data = super::test_utils::rand_f32_vec(0x7110_0002, batch * k * n, -1.0, 1.0);

    let graph = ComputationGraph::from_nodes(vec![
        input_node(0, &[batch, m, k]),
        input_node(1, &[batch, k, n]),
        TraceNode::new(
            2,
            "matmul_attn".into(),
            TraceOp::MatMul,
            vec![0, 1],
            vec![batch, m, n],
            DType::F32,
        ),
    ]);

    let compiled = CompiledModel::builder(&graph, &cache)
        .build()
        .expect("compile");

    assert!(
        has_tiled_matmul(&compiled),
        "attention shape [8,64,64]x[8,64,64] must route to TiledMatMul"
    );

    let left_buf = create_input_buffer(&cache, &left_data);
    let right_buf = create_input_buffer(&cache, &right_data);
    let out_buf = compiled
        .execute(&cache, &[&left_buf, &right_buf])
        .expect("execute");
    let gpu = read_output_n(&out_buf, batch * m * n);

    // CPU reference: batch matmul.
    let mut expected = vec![0.0_f32; batch * m * n];
    for b in 0..batch {
        let l_off = b * m * k;
        let r_off = b * k * n;
        let o_off = b * m * n;
        let batch_ref = super::test_utils::matmul_ref(
            &left_data[l_off..l_off + m * k],
            &right_data[r_off..r_off + k * n],
            m,
            k,
            n,
            false,
            None,
        );
        expected[o_off..o_off + m * n].copy_from_slice(&batch_ref);
    }

    assert_close("tiled_matmul_batched_attention", &gpu, &expected, 1e-3);
}

// ---------------------------------------------------------------------------
// Test 3: F16 autocast vs F32 compiled baseline
// ---------------------------------------------------------------------------

/// Batched [8, 64, 64] matmul: F16 autocast vs F32 compiled output.
/// ops_e2e_8 compares F16 against CPU reference; this compares two compiled
/// paths (F32 compiled vs F16 compiled) to validate autocast tiled parity.
#[test]
fn test_tiled_matmul_f16_vs_f32() {
    use nn_core::mixed_precision::MixedPrecisionPolicy;

    super::test_utils::gpu_init();
    let cache = super::test_utils::metal_setup();

    let (batch, m, k, n) = (8, 64, 64, 64);
    let left_data = super::test_utils::rand_f32_vec(0x7110_F001, batch * m * k, -1.0, 1.0);
    let right_data = super::test_utils::rand_f32_vec(0x7110_F002, batch * k * n, -1.0, 1.0);

    let graph = ComputationGraph::from_nodes(vec![
        input_node(0, &[batch, m, k]),
        input_node(1, &[batch, k, n]),
        TraceNode::new(
            2,
            "matmul_attn_f16".into(),
            TraceOp::MatMul,
            vec![0, 1],
            vec![batch, m, n],
            DType::F32,
        ),
    ]);

    // F32 baseline (compiled, not CPU ref).
    let f32_compiled = CompiledModel::builder(&graph, &cache)
        .build()
        .expect("compile f32");
    let left_buf = create_input_buffer(&cache, &left_data);
    let right_buf = create_input_buffer(&cache, &right_data);
    let f32_buf = f32_compiled
        .execute(&cache, &[&left_buf, &right_buf])
        .expect("f32 exec");
    let f32_result = read_output_n(&f32_buf, batch * m * n);

    // Autocast F16.
    let policy = MixedPrecisionPolicy::apple_silicon_default();
    let f16_compiled = CompiledModel::builder(&graph, &cache)
        .autocast(policy)
        .build()
        .expect("compile autocast");
    assert!(f16_compiled.is_autocast(), "model should be autocast");

    let f16_buf = f16_compiled
        .execute(&cache, &[&left_buf, &right_buf])
        .expect("f16 exec");
    let f16_result = read_output_n(&f16_buf, batch * m * n);

    // F16 matmul with K=64 accumulation — expect ~1e-2 tolerance.
    assert_close("tiled_matmul_f16_vs_f32", &f16_result, &f32_result, 5e-2);
}

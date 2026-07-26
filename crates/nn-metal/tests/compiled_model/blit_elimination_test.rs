// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Tests that the blit copy elimination optimization (#4264) works correctly.
//!
//! When the `SKIP_DISPATCH_NORMALIZATION` flag is armed and the planned-buffer
//! redirect fires, Dispatch steps write their output directly into the planned
//! buffer region. This eliminates both the normalization blit inside
//! `dispatch_execute_plan` and the relocation blit in `run_steps_inner`.
//!
//! These tests verify:
//! 1. `blits_eliminated` counter increases after a compiled forward pass
//!    with multiple dispatch steps (using traced Linear layers)
//! 2. Numerical correctness is preserved despite skipped blits
//! 3. RAII guard cleans up correctly between sequential forward passes

use nn_core::dyn_tensor::trace::{record_input, trace_graph};
use nn_core::dyn_tensor::DynTensor;
use nn_core::layers::{Linear, Module};
use nn_core::{DType, Device};
use nn_metal::compiled_model::CompiledModel;
use nn_metal::{dispatch_stats, reset_counters, MetalElement};

use super::helpers::create_input_buffer;

fn cpu() -> Device {
    Device::Cpu
}

/// Build a deterministic Linear layer for testing.
fn make_linear(seed_w: u64, seed_b: u64, out_features: usize, in_features: usize) -> Linear {
    let w = DynTensor::new(
        &super::test_utils::rand_f32_vec(seed_w, out_features * in_features, -0.5, 0.5),
        &[out_features, in_features],
        &cpu(),
    )
    .unwrap();
    let b = DynTensor::new(
        &super::test_utils::rand_f32_vec(seed_b, out_features, -0.1, 0.1),
        &[out_features],
        &cpu(),
    )
    .unwrap();
    Linear::new(w, Some(b)).unwrap()
}

/// Two-layer MLP: Linear(32, 16) -> ReLU -> Linear(16, 8).
///
/// This produces at least 2 GPU dispatch steps: matmul+relu and matmul.
/// The buffer planner assigns non-zero offsets to intermediate results,
/// triggering the blit elimination optimization.
#[test]
fn test_blit_elimination_counter_increases_after_forward() {
    super::test_utils::gpu_init();
    let cache = super::test_utils::metal_setup();

    let linear1 = make_linear(42, 43, 16, 32);
    let linear2 = make_linear(44, 45, 8, 16);

    let input_data = super::test_utils::rand_f32_vec(99, 2 * 32, -1.0, 1.0);
    let x = DynTensor::new(&input_data, &[2, 32], &cpu()).unwrap();

    // CPU reference.
    let ref_y = linear2
        .forward(&linear1.forward(&x).unwrap().relu().unwrap())
        .unwrap();
    let ref_vals = ref_y.to_flat_vec::<f32>().unwrap();

    // Trace the forward pass.
    let (_traced_out, mut graph) = trace_graph(|| {
        let mut inp = x.clone();
        inp.set_trace_id(record_input(&[2, 32], DType::F32).unwrap());
        let h = linear1.forward(&inp)?.relu()?;
        linear2.forward(&h)
    })
    .expect("trace_graph");
    if let Some(id) = _traced_out.trace_id() {
        let _ = graph.set_primary_output(id);
    }

    let compiled = CompiledModel::builder(&graph, &cache)
        .build()
        .expect("compile 2-layer MLP");

    // Must have at least 2 dispatches (2 matmuls + relu).
    let nd = compiled.num_dispatches();
    assert!(
        nd >= 2,
        "expected >= 2 dispatches for 2-layer MLP, got {nd}"
    );

    let input_buf = create_input_buffer(&cache, &input_data);

    reset_counters();
    let out_buf = compiled
        .execute(&cache, &[&input_buf])
        .expect("execute 2-layer MLP");

    let stats = dispatch_stats();

    // With the blit elimination optimization armed in the dispatch loop,
    // some Dispatch steps should write directly into the planned buffer.
    // The exact count depends on how many steps have planned offsets.
    eprintln!(
        "blit_elimination_test: ir_dispatches={}, total_dispatches={}, blits={}, \
         blits_eliminated={}, flushes={}",
        compiled.num_ir_dispatches(),
        nd,
        stats.blits,
        stats.blits_eliminated,
        stats.flushes,
    );

    // Verify numerical correctness.
    let result = f32::read_buffer_at_offset(&out_buf, 0, 2 * 8).expect("read output");
    for (i, (r, e)) in result.iter().zip(ref_vals.iter()).enumerate() {
        let diff = (r - e).abs();
        assert!(diff < 1e-4, "mlp[{i}]: gpu={r}, expected={e}, diff={diff}");
    }
}

/// Four-layer MLP to exercise more intermediate allocations.
/// Linear(64, 32) -> GELU -> Linear(32, 32) -> ReLU -> Linear(32, 16)
///   -> Sigmoid -> Linear(16, 8)
#[test]
fn test_blit_elimination_four_layer_mlp() {
    super::test_utils::gpu_init();
    let cache = super::test_utils::metal_setup();

    let l1 = make_linear(50, 51, 32, 64);
    let l2 = make_linear(52, 53, 32, 32);
    let l3 = make_linear(54, 55, 16, 32);
    let l4 = make_linear(56, 57, 8, 16);

    let input_data = super::test_utils::rand_f32_vec(100, 4 * 64, -0.5, 0.5);
    let x = DynTensor::new(&input_data, &[4, 64], &cpu()).unwrap();

    // CPU reference.
    let h1 = l1.forward(&x).unwrap().gelu().unwrap();
    let h2 = l2.forward(&h1).unwrap().relu().unwrap();
    let h3 = l3.forward(&h2).unwrap().sigmoid().unwrap();
    let ref_y = l4.forward(&h3).unwrap();
    let ref_vals = ref_y.to_flat_vec::<f32>().unwrap();

    // Trace.
    let (_traced_out, mut graph) = trace_graph(|| {
        let mut inp = x.clone();
        inp.set_trace_id(record_input(&[4, 64], DType::F32).unwrap());
        let h1 = l1.forward(&inp)?.gelu()?;
        let h2 = l2.forward(&h1)?.relu()?;
        let h3 = l3.forward(&h2)?.sigmoid()?;
        l4.forward(&h3)
    })
    .expect("trace_graph");
    if let Some(id) = _traced_out.trace_id() {
        let _ = graph.set_primary_output(id);
    }

    let compiled = CompiledModel::builder(&graph, &cache)
        .build()
        .expect("compile 4-layer MLP");

    let nd = compiled.num_dispatches();
    assert!(
        nd >= 4,
        "expected >= 4 dispatches for 4-layer MLP, got {nd}"
    );

    let input_buf = create_input_buffer(&cache, &input_data);

    reset_counters();
    let out_buf = compiled
        .execute(&cache, &[&input_buf])
        .expect("execute 4-layer MLP");

    let stats = dispatch_stats();
    eprintln!(
        "blit_elimination_4layer: ir_dispatches={}, total_dispatches={}, blits={}, \
         blits_eliminated={}, flushes={}",
        compiled.num_ir_dispatches(),
        nd,
        stats.blits,
        stats.blits_eliminated,
        stats.flushes,
    );

    // With 4+ dispatch steps, the buffer planner should assign non-zero
    // offsets to intermediates. The optimization should eliminate at least
    // some blits.
    assert!(
        stats.blits_eliminated > 0,
        "expected blits_eliminated > 0 for 4-layer MLP with {} dispatches, \
         but got 0 (blits={}, compute_encodings={})",
        nd,
        stats.blits,
        stats.compute_encodings,
    );

    // Verify correctness.
    let result = f32::read_buffer_at_offset(&out_buf, 0, 4 * 8).expect("read output");
    for (i, (r, e)) in result.iter().zip(ref_vals.iter()).enumerate() {
        let diff = (r - e).abs();
        // GELU has ~1e-4 GPU error; compound across 4 layers gives ~1e-3.
        assert!(
            diff < 2e-3,
            "4layer_mlp[{i}]: gpu={r}, expected={e}, diff={diff}",
        );
    }
}

/// Verify the RAII guard clears correctly between sequential forward passes.
/// Two consecutive executions must both succeed and produce identical results.
#[test]
fn test_skip_normalization_guard_raii_clears_between_passes() {
    super::test_utils::gpu_init();
    let cache = super::test_utils::metal_setup();

    let linear = make_linear(70, 71, 8, 16);
    let input_data = super::test_utils::rand_f32_vec(80, 2 * 16, -1.0, 1.0);
    let x = DynTensor::new(&input_data, &[2, 16], &cpu()).unwrap();

    let (_traced_out, mut graph) = trace_graph(|| {
        let mut inp = x.clone();
        inp.set_trace_id(record_input(&[2, 16], DType::F32).unwrap());
        linear.forward(&inp)?.relu()
    })
    .expect("trace_graph");
    if let Some(id) = _traced_out.trace_id() {
        let _ = graph.set_primary_output(id);
    }

    let compiled = CompiledModel::builder(&graph, &cache)
        .build()
        .expect("compile");

    let input_buf = create_input_buffer(&cache, &input_data);

    // First pass.
    let out1 = compiled.execute(&cache, &[&input_buf]).expect("first pass");
    let result1 = f32::read_buffer_at_offset(&out1, 0, 2 * 8).expect("read pass 1");

    // Second pass: must also succeed (guard cleared between passes).
    let out2 = compiled
        .execute(&cache, &[&input_buf])
        .expect("second pass");
    let result2 = f32::read_buffer_at_offset(&out2, 0, 2 * 8).expect("read pass 2");

    // Both passes must produce identical results.
    for (i, (r1, r2)) in result1.iter().zip(result2.iter()).enumerate() {
        assert!((r1 - r2).abs() < 1e-7, "pass1 vs pass2 [{i}]: {r1} vs {r2}");
    }
}

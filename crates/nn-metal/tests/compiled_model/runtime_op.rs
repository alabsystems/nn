// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! GPU tests for `CompiledStep::RuntimeOp` execution.
//!
//! RuntimeOp handles data-dependent operations whose output shapes cannot
//! be determined at compile time. The buffer planner allocates 0 bytes for
//! these steps — the executor allocates dynamically at inference time.
//!
//! Currently covers `RuntimeOpKind::RepeatInterleave` (variable-length
//! repeat counts, used by Kokoro `length_regulate`).
//!
//! Part of #2234 (RuntimeOp for data-dependent ops).
//! Part of #2218 (Kokoro epic).

use std::collections::HashMap;

use nn_core::dyn_tensor::trace::{record_input, trace_graph, ComputationGraph};
use nn_core::dyn_tensor::DynTensor;
use nn_core::layers::{Linear, Module};
use nn_core::{DType, Device, VarBuilder};
use nn_dsl::trace_compile::{compile_trace_to_plan_with_fusion, CompiledPlan, CompiledStep};
use nn_dsl::RuntimeOpKind;
use nn_metal::compiled_model::CompiledModel;
use nn_models::kokoro_tts::length_regulate;

fn cpu() -> Device {
    Device::Cpu
}

fn gpu() -> Device {
    Device::Metal { device_id: 0 }
}

/// Check if a compiled plan contains a RuntimeOp::RepeatInterleave step.
fn plan_has_runtime_op(plan: &CompiledPlan) -> bool {
    plan.steps.iter().any(|s| {
        matches!(
            s,
            CompiledStep::RuntimeOp {
                op: RuntimeOpKind::RepeatInterleave { .. }
            }
        )
    })
}

/// Trace repeat_interleave into a computation graph.
fn trace_repeat_interleave(
    input: &DynTensor,
    counts: &DynTensor,
    dim: usize,
) -> (DynTensor, ComputationGraph) {
    let (traced_out, mut graph) = trace_graph(|| {
        let mut inp = input.clone();
        inp.set_trace_id(record_input(inp.dims(), DType::F32).unwrap());
        let mut cts = counts.clone();
        cts.set_trace_id(record_input(cts.dims(), DType::F32).unwrap());
        inp.repeat_interleave(dim, &cts)
    })
    .expect("trace_graph");

    if let Some(id) = traced_out.trace_id() {
        assert!(graph.set_primary_output(id), "output not in graph");
    }
    (traced_out, graph)
}

/// Execute a compiled plan on GPU and return output values.
fn execute_and_readback(
    plan: &CompiledPlan,
    graph: &ComputationGraph,
    cache: &nn_metal::PipelineCache,
    gpu_inputs: &[&DynTensor],
) -> Vec<f32> {
    let compiled = CompiledModel::from_plan(plan, graph, cache).expect("from_plan");
    let result = compiled
        .execute_dyn(cache, gpu_inputs)
        .expect("execute_dyn");
    result
        .to_device(&cpu())
        .unwrap()
        .to_flat_vec::<f32>()
        .unwrap()
}

/// Assert GPU values match reference within tolerance.
fn assert_close(gpu_vals: &[f32], ref_vals: &[f32], tol: f32, label: &str) {
    assert_eq!(
        gpu_vals.len(),
        ref_vals.len(),
        "{label}: length mismatch gpu={} ref={}",
        gpu_vals.len(),
        ref_vals.len()
    );
    let mut max_diff: f32 = 0.0;
    for (i, (g, r)) in gpu_vals.iter().zip(ref_vals.iter()).enumerate() {
        let diff = (g - r).abs();
        max_diff = max_diff.max(diff);
        assert!(diff <= tol, "{label}[{i}]: gpu={g}, ref={r}, diff={diff}");
    }
    eprintln!("{label} max diff: {max_diff:.2e}");
}

/// Helper: trace repeat_interleave, compile, execute on GPU, verify vs CPU.
fn trace_compile_execute_repeat_interleave(
    cache: &nn_metal::PipelineCache,
    input: &DynTensor,
    counts: &DynTensor,
    dim: usize,
    tol: f32,
    label: &str,
) {
    let ref_output = input.repeat_interleave(dim, counts).unwrap();
    let ref_vals = ref_output.to_flat_vec::<f32>().unwrap();

    let (_traced_out, graph) = trace_repeat_interleave(input, counts, dim);
    let plan = compile_trace_to_plan_with_fusion(&graph).expect("compile");
    assert!(
        plan_has_runtime_op(&plan),
        "{label}: should compile to RuntimeOp"
    );

    let input_gpu = input.to_device(&gpu()).unwrap();
    let counts_gpu = counts.to_device(&gpu()).unwrap();
    let gpu_vals = execute_and_readback(&plan, &graph, cache, &[&input_gpu, &counts_gpu]);
    assert_close(&gpu_vals, &ref_vals, tol, label);
}

/// RuntimeOp with counts that include zeros (some rows dropped).
///
/// Input: `[4, 2]`, repeats `[0, 2, 0, 3]` along dim 0.
/// Expected output: `[5, 2]` (0+2+0+3 = 5, 5 % 4 != 0 → RuntimeOp).
#[test]
fn test_runtime_op_repeat_interleave_with_zeros() {
    super::test_utils::gpu_init();
    let cache = super::test_utils::metal_setup();

    let input = DynTensor::new(
        &(0..8).map(|i| (i as f32) * 0.5 + 1.0).collect::<Vec<_>>(),
        &[4, 2],
        &cpu(),
    )
    .unwrap();
    let counts = DynTensor::new(&[0.0_f32, 2.0, 0.0, 3.0], &[4], &cpu()).unwrap();
    trace_compile_execute_repeat_interleave(&cache, &input, &counts, 0, 1e-6, "zeros");
}

/// RuntimeOp with non-divisible total (3+1 = 4, 4 % 3 != 0 → RuntimeOp).
#[test]
fn test_runtime_op_repeat_interleave_nondivisible() {
    super::test_utils::gpu_init();
    let cache = super::test_utils::metal_setup();

    let input = DynTensor::new(&[1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0], &[3, 2], &cpu()).unwrap();
    let counts = DynTensor::new(&[3.0_f32, 1.0, 0.0], &[3], &cpu()).unwrap();
    trace_compile_execute_repeat_interleave(&cache, &input, &counts, 0, 1e-6, "nondivisible");
}

/// Regression test for #2452: non-uniform counts where total % input_dim == 0.
///
/// Counts [1, 2, 0]: total = 3 = input_dim. Previously miscompiled as uniform
/// repeat=1 (identity). Now correctly emits RuntimeOp. Fixes #2452.
#[test]
fn test_repeat_interleave_divisible_but_nonuniform_fixed() {
    super::test_utils::gpu_init();
    let cache = super::test_utils::metal_setup();

    let input = DynTensor::new(&[10.0_f32, 20.0, 30.0, 40.0, 50.0, 60.0], &[2, 3], &cpu()).unwrap();
    let counts = DynTensor::new(&[1.0_f32, 2.0, 0.0], &[3], &cpu()).unwrap();
    trace_compile_execute_repeat_interleave(&cache, &input, &counts, 1, 1e-6, "divisible_fix");
}

/// Kokoro `length_regulate` traces and compiles with RuntimeOp boundary.
/// Part of #2234 AC2.
#[test]
fn test_kokoro_length_regulate_compiles_with_runtime_op() {
    super::test_utils::gpu_init();
    let cache = super::test_utils::metal_setup();

    let features = DynTensor::new(
        &(0..12).map(|i| (i as f32) * 0.1 + 0.1).collect::<Vec<_>>(),
        &[1, 4, 3],
        &cpu(),
    )
    .unwrap();
    // Total = 5, 5 % 3 != 0 → RuntimeOp (avoids #2452 divisible miscompilation).
    let durations = DynTensor::new(&[1.8_f32, 0.6, 1.7], &[1, 3], &cpu()).unwrap();

    let ref_output = length_regulate(&features, &durations).unwrap();
    let ref_vals = ref_output.to_flat_vec::<f32>().unwrap();
    assert_eq!(ref_output.dims(), &[1, 4, 5], "reference output shape");

    let (traced_out, mut graph) = trace_graph(|| {
        let mut feat = features.clone();
        feat.set_trace_id(record_input(feat.dims(), DType::F32).unwrap());
        let mut dur = durations.clone();
        dur.set_trace_id(record_input(dur.dims(), DType::F32).unwrap());
        length_regulate(&feat, &dur).map_err(Into::into)
    })
    .expect("trace_graph");
    if let Some(id) = traced_out.trace_id() {
        assert!(graph.set_primary_output(id), "output not in graph");
    }

    let plan = compile_trace_to_plan_with_fusion(&graph).expect("compile");
    assert!(plan_has_runtime_op(&plan), "should compile to RuntimeOp");

    let feat_gpu = features.to_device(&gpu()).unwrap();
    let dur_gpu = durations.to_device(&gpu()).unwrap();
    let gpu_vals = execute_and_readback(&plan, &graph, &cache, &[&feat_gpu, &dur_gpu]);
    assert_close(&gpu_vals, &ref_vals, 1e-5, "length_regulate");
}

/// Build a Linear layer from constant weight/bias values.
fn build_linear(rows: usize, cols: usize, seed: f64) -> Linear {
    let mut m = HashMap::new();
    m.insert(
        "weight".to_string(),
        DynTensor::full(&[rows, cols], seed, DType::F32, &cpu()).unwrap(),
    );
    m.insert(
        "bias".to_string(),
        DynTensor::full(&[rows], seed * 0.1, DType::F32, &cpu()).unwrap(),
    );
    let vb = VarBuilder::from_tensors(m, DType::F32, &cpu());
    let w = vb.get(&[rows, cols], "weight").unwrap();
    let b = vb.get(&[rows], "bias").unwrap();
    Linear::new(w, Some(b)).unwrap()
}

/// Static → RuntimeOp → Static round-trip. Part of #2234 AC4.
#[test]
fn test_static_runtime_static_round_trip() {
    super::test_utils::gpu_init();
    let cache = super::test_utils::metal_setup();

    let (d_in, d_mid, d_out, t) = (6, 4, 3, 3);
    let pre_linear = build_linear(d_mid, d_in, 0.02);
    let post_linear = build_linear(d_out, d_mid, 0.03);

    let input = DynTensor::new(
        &(0..(d_in * t))
            .map(|i| (i as f32) * 0.1 + 0.05)
            .collect::<Vec<_>>(),
        &[1, d_in, t],
        &cpu(),
    )
    .unwrap();
    // Total = 5, 5 % 3 != 0 → RuntimeOp (avoids #2452).
    let durations = DynTensor::new(&[1.8_f32, 0.6, 1.7], &[1, 3], &cpu()).unwrap();

    // Eager reference: pre_linear → length_regulate → post_linear.
    let pre_out = pre_linear
        .forward(&input.transpose(1, 2).unwrap())
        .unwrap()
        .transpose(1, 2)
        .unwrap();
    let aligned = length_regulate(&pre_out, &durations).unwrap();
    let ref_output = post_linear
        .forward(&aligned.transpose(1, 2).unwrap())
        .unwrap()
        .transpose(1, 2)
        .unwrap();
    let ref_vals = ref_output.to_flat_vec::<f32>().unwrap();

    let (traced_out, mut graph) = trace_graph(|| {
        let mut inp = input.clone();
        inp.set_trace_id(record_input(inp.dims(), DType::F32).unwrap());
        let mut dur = durations.clone();
        dur.set_trace_id(record_input(dur.dims(), DType::F32).unwrap());
        let pre = pre_linear.forward(&inp.transpose(1, 2)?)?.transpose(1, 2)?;
        let aligned = length_regulate(&pre, &dur)?;
        post_linear
            .forward(&aligned.transpose(1, 2)?)?
            .transpose(1, 2)
    })
    .expect("trace_graph");
    if let Some(id) = traced_out.trace_id() {
        assert!(graph.set_primary_output(id), "output not in graph");
    }

    let plan = compile_trace_to_plan_with_fusion(&graph).expect("compile");
    assert!(plan_has_runtime_op(&plan), "should contain RuntimeOp");
    let num_dispatches = plan
        .steps
        .iter()
        .filter(|s| matches!(s, CompiledStep::Dispatch { .. }))
        .count();
    assert!(
        num_dispatches >= 2,
        "should have ≥2 Dispatch steps; got {num_dispatches}"
    );

    let input_gpu = input.to_device(&gpu()).unwrap();
    let dur_gpu = durations.to_device(&gpu()).unwrap();
    let gpu_vals = execute_and_readback(&plan, &graph, &cache, &[&input_gpu, &dur_gpu]);
    assert_close(&gpu_vals, &ref_vals, 1e-3, "static-runtime-static");
}

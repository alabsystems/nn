// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Integration tests for `CompiledModel::execute_dyn_profiled()` (#2257).
//!
//! Exercises the per-step profiling API: verifies that `ExecutionProfile`
//! returns correct step counts, step names, timing data, and GPU dispatch
//! classification. Also verifies that profiled execution produces the same
//! numerical results as non-profiled execution.

use std::sync::Arc;

use nn_core::dyn_tensor::trace::{ComputationGraph, TraceNode, TraceOp, WeightRef};
use nn_core::dyn_tensor::DynTensor;
use nn_core::mixed_precision::MixedPrecisionPolicy;
use nn_core::{DType, Device};
use nn_metal::compiled_model::CompiledModel;

use super::helpers::{assert_close, binary_node, create_input_buffer, input_node, unary_node};

// -- Helpers -----------------------------------------------------------------

/// Create a GPU-backed DynTensor from f32 data.
fn gpu_dyn_tensor(cache: &nn_metal::PipelineCache, data: &[f32], shape: &[usize]) -> DynTensor {
    let buf = create_input_buffer(cache, data);
    let storage = nn_metal::MetalTensorData::new(buf);
    DynTensor::from_gpu_storage(
        shape.to_vec(),
        DType::F32,
        Arc::new(storage),
        Device::metal(),
    )
    .expect("create GPU DynTensor")
}

// -- Tests: profile metadata -------------------------------------------------

/// Single relu: 1 input (non-dispatch) + 1 dispatch step.
#[test]
fn test_profiled_single_relu_step_count() {
    super::test_utils::gpu_init();
    let cache = super::test_utils::metal_setup();

    let graph = ComputationGraph::from_nodes(vec![
        input_node(0, &[4]),
        unary_node(1, "relu_0", TraceOp::Relu, 0, &[4]),
    ]);
    let compiled = CompiledModel::builder(&graph, &cache)
        .build()
        .expect("compile relu");

    let input = gpu_dyn_tensor(&cache, &[-2.0, -1.0, 1.0, 2.0], &[4]);
    let (output, profile) = compiled
        .execute_dyn_profiled(&cache, &[&input])
        .expect("profiled execute");

    // Verify profile metadata.
    assert_eq!(profile.steps.len(), 2, "relu graph: 1 input + 1 dispatch");
    assert_eq!(profile.num_gpu_dispatches(), 1);
    assert!(
        profile.total_wall_time_us > 0.0,
        "total time should be positive"
    );

    // Verify step names.
    assert_eq!(profile.steps[0].step_name, "input");
    assert!(!profile.steps[0].is_gpu_dispatch);
    assert!(profile.steps[1].is_gpu_dispatch);

    // Verify numerical correctness — same as non-profiled path.
    let result = output.to_flat_vec::<f32>().expect("read output");
    assert_eq!(result, vec![0.0, 0.0, 1.0, 2.0]);
}

/// Fused chain: 3 elementwise ops should fuse into 1 dispatch.
/// Profile should still report individual step timings.
#[test]
fn test_profiled_fused_chain_step_names() {
    super::test_utils::gpu_init();
    let cache = super::test_utils::metal_setup();

    let graph = ComputationGraph::from_nodes(vec![
        input_node(0, &[4]),
        unary_node(1, "exp_0", TraceOp::Exp, 0, &[4]),
        unary_node(2, "relu_0", TraceOp::Relu, 1, &[4]),
        unary_node(3, "sigmoid_0", TraceOp::Sigmoid, 2, &[4]),
    ]);
    let compiled = CompiledModel::builder(&graph, &cache)
        .build()
        .expect("compile fused");

    let input = gpu_dyn_tensor(&cache, &[0.5, -1.0, 0.0, 1.5], &[4]);
    let (output, profile) = compiled
        .execute_dyn_profiled(&cache, &[&input])
        .expect("profiled fused");

    // Steps: 1 input + 3 (identity/fused dispatch placeholders + actual dispatch).
    assert_eq!(profile.steps.len(), 4);
    // With fusion, only 1 dispatch.
    assert_eq!(
        profile.num_gpu_dispatches(),
        compiled.num_dispatches(),
        "profiled dispatch count should match compiled"
    );

    // Verify numerical correctness: sigmoid(relu(exp(x))).
    let result = output.to_flat_vec::<f32>().expect("read output");
    let expected: Vec<f32> = [0.5_f32, -1.0, 0.0, 1.5]
        .iter()
        .map(|x| {
            let e = x.exp();
            let r = e.max(0.0);
            1.0 / (1.0 + (-r).exp())
        })
        .collect();
    for (i, (r, e)) in result.iter().zip(expected.iter()).enumerate() {
        assert!(
            (r - e).abs() < 1e-5,
            "fused_profiled[{i}]: got {r}, expected {e}"
        );
    }
}

// -- Tests: profiled matches non-profiled output -----------------------------

/// Diamond topology: verify profiled and non-profiled outputs are identical.
#[test]
fn test_profiled_matches_non_profiled() {
    super::test_utils::gpu_init();
    let cache = super::test_utils::metal_setup();

    let graph = ComputationGraph::from_nodes(vec![
        input_node(0, &[4]),
        unary_node(1, "relu_0", TraceOp::Relu, 0, &[4]),
        unary_node(2, "sigmoid_0", TraceOp::Sigmoid, 0, &[4]),
        binary_node(3, "add_0", TraceOp::Add, 1, 2, &[4]),
    ]);
    let compiled = CompiledModel::builder(&graph, &cache)
        .build()
        .expect("compile diamond");

    let data = [-1.0_f32, 0.5, -0.5, 2.0];
    let input = gpu_dyn_tensor(&cache, &data, &[4]);

    // Non-profiled execution.
    let output_normal = compiled
        .execute_dyn(&cache, &[&input])
        .expect("non-profiled");
    let result_normal = output_normal.to_flat_vec::<f32>().expect("read normal");

    // Profiled execution.
    let (output_profiled, profile) = compiled
        .execute_dyn_profiled(&cache, &[&input])
        .expect("profiled");
    let result_profiled = output_profiled.to_flat_vec::<f32>().expect("read profiled");

    // Outputs must match exactly.
    assert_eq!(
        result_normal, result_profiled,
        "profiled and non-profiled must produce identical output"
    );

    // Profile should have steps.
    assert_eq!(profile.steps.len(), compiled.num_steps());
}

// -- Tests: slowest_steps and gpu_time_fraction ------------------------------

#[test]
fn test_profiled_slowest_steps_ordering() {
    super::test_utils::gpu_init();
    let cache = super::test_utils::metal_setup();

    // Multi-dispatch graph: relu and sigmoid are separate dispatches
    // (input has fan-out 2, so no fusion possible).
    let graph = ComputationGraph::from_nodes(vec![
        input_node(0, &[4]),
        unary_node(1, "relu_0", TraceOp::Relu, 0, &[4]),
        unary_node(2, "sigmoid_0", TraceOp::Sigmoid, 0, &[4]),
        binary_node(3, "add_0", TraceOp::Add, 1, 2, &[4]),
    ]);
    let compiled = CompiledModel::builder(&graph, &cache)
        .build()
        .expect("compile");

    let input = gpu_dyn_tensor(&cache, &[1.0, -1.0, 0.0, 2.0], &[4]);
    let (_output, profile) = compiled
        .execute_dyn_profiled(&cache, &[&input])
        .expect("profiled");

    // slowest_steps(2) should return 2 entries sorted by wall time descending.
    let slowest = profile.slowest_steps(2);
    assert_eq!(slowest.len(), 2);
    assert!(
        slowest[0].wall_time_us >= slowest[1].wall_time_us,
        "slowest_steps should be sorted descending"
    );

    // gpu_time_fraction should be between 0 and 1.
    let frac = profile.gpu_time_fraction();
    assert!(
        (0.0..=1.0).contains(&frac),
        "gpu_time_fraction should be in [0, 1], got {frac}"
    );
}

// -- Tests: Display formatting -----------------------------------------------

#[test]
fn test_profiled_display_contains_summary() {
    super::test_utils::gpu_init();
    let cache = super::test_utils::metal_setup();

    let graph = ComputationGraph::from_nodes(vec![
        input_node(0, &[4]),
        unary_node(1, "relu_0", TraceOp::Relu, 0, &[4]),
    ]);
    let compiled = CompiledModel::builder(&graph, &cache)
        .build()
        .expect("compile");

    let input = gpu_dyn_tensor(&cache, &[1.0, -1.0, 2.0, -2.0], &[4]);
    let (_output, profile) = compiled
        .execute_dyn_profiled(&cache, &[&input])
        .expect("profiled");

    let display = format!("{profile}");
    assert!(
        display.contains("ExecutionProfile"),
        "Display should contain header"
    );
    assert!(
        display.contains("GPU"),
        "Display should mention GPU dispatches"
    );
    assert!(
        display.contains("Top 10 slowest"),
        "Display should list slowest steps"
    );
}

// -- Tests: empty profile edge case ------------------------------------------

#[test]
fn test_profiled_empty_plan_returns_error() {
    let cache = super::test_utils::metal_setup();
    let graph = ComputationGraph::from_nodes(vec![]);
    let compiled = CompiledModel::builder(&graph, &cache)
        .build()
        .expect("compile empty");

    // No DynTensor inputs for empty plan — should error, not return empty profile.
    let err_msg = compiled
        .execute_dyn_profiled(&cache, &[])
        .unwrap_err()
        .to_string();
    assert!(
        err_msg.contains("compiled plan is empty"),
        "expected EmptyPlan error, got: {err_msg}"
    );
}

// -- Tests: ExecutionProfile::new and helpers --------------------------------

#[test]
fn test_execution_profile_new_computes_total() {
    use nn_metal::compiled_model::profile::{ExecutionProfile, StepProfile};

    let steps = vec![
        StepProfile::new(0, "input".into(), 5.0, false, 0),
        StepProfile::new(1, "relu".into(), 100.0, true, 0),
        StepProfile::new(2, "sigmoid".into(), 150.0, true, 0),
    ];
    let profile = ExecutionProfile::new(steps);
    assert!((profile.total_wall_time_us - 255.0).abs() < 1e-9);
    assert_eq!(profile.num_gpu_dispatches(), 2);

    let frac = profile.gpu_time_fraction();
    let expected_frac = 250.0 / 255.0;
    assert!(
        (frac - expected_frac).abs() < 1e-6,
        "gpu_time_fraction: got {frac}, expected {expected_frac}"
    );

    let slowest = profile.slowest_steps(1);
    assert_eq!(slowest.len(), 1);
    assert_eq!(slowest[0].step_name, "sigmoid");
}

// -- Tests: autocast profiled vs non-profiled equivalence --------------------

/// Autocast (mixed GEMM) profiled path must produce identical output to
/// the non-profiled path. The run_steps / run_steps_profiled code is
/// duplicated (~270 lines each); this test catches divergence.
///
/// Part of #3020 (proof certificate pipeline — buffer aliasing safety).
#[test]
fn test_profiled_autocast_mixed_gemm_matches_non_profiled() {
    super::test_utils::gpu_init();
    let cache = super::test_utils::metal_setup();

    // TGs = ceil(384/32)*ceil(1024/32) = 12*32 = 384 >= MIN_TGS_FOR_MIXED_GEMM.
    let (m, k, n) = (384, 128, 1024);
    let weight_data = super::test_utils::rand_f32_vec(0xBE_EF01, k * n, -0.1, 0.1);
    let weight = WeightRef::new(weight_data, vec![n, k]).expect("weight");

    let graph = ComputationGraph::from_nodes(vec![
        input_node(0, &[m, k]),
        TraceNode::new(
            1,
            "linear_0".into(),
            TraceOp::Linear { weight, bias: None },
            vec![0],
            vec![m, n],
            DType::F32,
        ),
    ]);

    let policy = MixedPrecisionPolicy::apple_silicon_default();
    let compiled = CompiledModel::builder(&graph, &cache)
        .autocast(policy)
        .build()
        .expect("compile autocast");
    assert!(compiled.is_autocast(), "should be in autocast mode");
    assert_eq!(compiled.num_mixed_gemm_steps(), 1, "should use mixed GEMM");

    let input_data = super::test_utils::rand_f32_vec(0xBE_EF02, m * k, -1.0, 1.0);
    let input_tensor = gpu_dyn_tensor(&cache, &input_data, &[m, k]);

    // Non-profiled execution.
    let output_normal = compiled
        .execute_dyn(&cache, &[&input_tensor])
        .expect("non-profiled autocast");
    let result_normal = output_normal.to_flat_vec::<f32>().expect("read normal");

    // Profiled execution.
    let (output_profiled, profile) = compiled
        .execute_dyn_profiled(&cache, &[&input_tensor])
        .expect("profiled autocast");
    let result_profiled = output_profiled.to_flat_vec::<f32>().expect("read profiled");

    // Outputs must match within FP tolerance (both paths use same GPU commands
    // but flush timing may cause minor reordering in non-deterministic GPU paths).
    assert_eq!(
        result_normal.len(),
        result_profiled.len(),
        "output lengths must match"
    );
    assert_close(
        "profiled_autocast_mixed_gemm",
        &result_profiled,
        &result_normal,
        1e-6, // near-exact — both paths use identical computation
    );

    // Profile should have meaningful step data.
    assert_eq!(profile.steps.len(), compiled.num_steps());
    assert!(
        profile.num_gpu_dispatches() >= 1,
        "autocast model should have at least 1 GPU dispatch"
    );
}

/// Mixed GEMM steps produce F32 output (float accumulators), even though
/// step_scalar_types marks them F16 (for weight upload). The profiled
/// path's output_bytes must reflect the actual output dtype (F32 = 4
/// bytes/elem), not the step_scalar_types dtype (F16 = 2 bytes/elem).
///
/// Part of #3020 (proof certificate pipeline — profiled path accuracy).
#[test]
fn test_profiled_mixed_gemm_output_bytes_f32() {
    super::test_utils::gpu_init();
    let cache = super::test_utils::metal_setup();

    // TGs = 12*32 = 384 >= threshold.
    let (m, k, n) = (384, 128, 1024);
    let weight_data = super::test_utils::rand_f32_vec(0xCA_FE01, k * n, -0.1, 0.1);
    let weight = WeightRef::new(weight_data, vec![n, k]).expect("weight");

    let graph = ComputationGraph::from_nodes(vec![
        input_node(0, &[m, k]),
        TraceNode::new(
            1,
            "linear_0".into(),
            TraceOp::Linear { weight, bias: None },
            vec![0],
            vec![m, n],
            DType::F32,
        ),
    ]);

    let policy = MixedPrecisionPolicy::apple_silicon_default();
    let compiled = CompiledModel::builder(&graph, &cache)
        .autocast(policy)
        .build()
        .expect("compile autocast");
    assert_eq!(compiled.num_mixed_gemm_steps(), 1);

    let input_data = super::test_utils::rand_f32_vec(0xCA_FE02, m * k, -1.0, 1.0);
    let input_tensor = gpu_dyn_tensor(&cache, &input_data, &[m, k]);

    let (_output, profile) = compiled
        .execute_dyn_profiled(&cache, &[&input_tensor])
        .expect("profiled execute");

    // Mixed GEMM output is F32: m*n elements × 4 bytes.
    let linear_step = &profile.steps[1];
    let expected_output_bytes = m * n * 4;
    assert_eq!(
        linear_step.output_bytes, expected_output_bytes,
        "mixed GEMM output_bytes should use F32 size (4 bytes/elem), \
         not F16 (2 bytes/elem). Got {} bytes, expected {} bytes. \
         step_name={:?}",
        linear_step.output_bytes, expected_output_bytes, linear_step.step_name,
    );
}

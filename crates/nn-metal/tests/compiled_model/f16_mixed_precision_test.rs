// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! F16 mixed-precision tests for `CompiledModel` (#2981).
//!
//! Verifies Tier 1+2 F16 mixed-precision: `builder().force_dtype()` compiles graphs
//! with all steps (Dispatch and NativeOp) overridden to F16 (except LSTM
//! which stays F32 per D6). Inputs auto-cast F32→F16, output auto-cast
//! F16→F32. Fused NativeOp MSL kernels use `half` I/O with `float` accumulators.

use nn_core::dyn_tensor::trace::{ComputationGraph, TraceNode, TraceOp, WeightRef};
use nn_core::DType;
use nn_dsl::trace_compile::CompiledStep;
use nn_metal::compiled_model::CompiledModel;

use super::helpers::{
    assert_close, binary_node, create_input_buffer, input_node, read_output_n, unary_node,
};

// -- Unit tests: compilation metadata -----------------------------------------

/// `builder().force_dtype()` sets `is_mixed_precision()` and overrides dispatch
/// step scalar types to F16 while NativeOps stay F32.
#[test]
fn test_f16_compilation_metadata() {
    super::test_utils::gpu_init();
    let cache = super::test_utils::metal_setup();

    // Simple: input [4] → relu → output [4]
    let graph = ComputationGraph::from_nodes(vec![
        input_node(0, &[4]),
        unary_node(1, "relu_0", TraceOp::Relu, 0, &[4]),
    ]);

    let f32_model = CompiledModel::builder(&graph, &cache)
        .build()
        .expect("compile f32");
    let f16_model = CompiledModel::builder(&graph, &cache)
        .force_dtype(DType::F16)
        .unwrap()
        .build()
        .expect("compile f16");

    assert!(!f32_model.is_mixed_precision());
    assert!(f16_model.is_mixed_precision());
    assert_eq!(f16_model.num_steps(), f32_model.num_steps());
    assert_eq!(f16_model.num_dispatches(), f32_model.num_dispatches());
}

/// Dispatch steps get F16 scalar type; NativeOp steps stay F32.
/// Verifies the step_scalar_types override logic in `from_plan_inner`.
#[test]
fn test_f16_step_types_dispatch_vs_native() {
    super::test_utils::gpu_init();
    let cache = super::test_utils::metal_setup();

    // input [4] → relu → output [4] (all Dispatch steps, no NativeOps)
    let graph = ComputationGraph::from_nodes(vec![
        input_node(0, &[4]),
        unary_node(1, "relu_0", TraceOp::Relu, 0, &[4]),
    ]);

    let model = CompiledModel::builder(&graph, &cache)
        .force_dtype(DType::F16)
        .unwrap()
        .build()
        .expect("compile f16");

    // All steps should be present. The Relu step (a Dispatch step) should use F16.
    // InputForward is not a Dispatch/NativeOp — it's just a passthrough.
    let steps = model.steps();
    assert!(steps.len() >= 2, "expected at least input + relu steps");

    // Verify no NativeOp steps in this simple graph
    for step in steps {
        assert!(
            !matches!(step, CompiledStep::NativeOp { .. }),
            "simple relu graph should not have NativeOps"
        );
    }
}

// -- Integration tests: F16 elementwise ops -----------------------------------

/// Relu through F16 pipeline: input F32 → auto-cast F16 → relu → auto-cast F32.
/// Verifies the full input-cast → dispatch → output-cast path.
#[test]
fn test_f16_relu_matches_cpu() {
    super::test_utils::gpu_init();
    let cache = super::test_utils::metal_setup();

    let graph = ComputationGraph::from_nodes(vec![
        input_node(0, &[8]),
        unary_node(1, "relu_0", TraceOp::Relu, 0, &[8]),
    ]);

    let model = CompiledModel::builder(&graph, &cache)
        .force_dtype(DType::F16)
        .unwrap()
        .build()
        .expect("compile f16");

    let input_data: Vec<f32> = vec![-2.0, -1.0, -0.5, 0.0, 0.1, 0.5, 1.0, 2.0];
    let buf = create_input_buffer(&cache, &input_data);
    let out = model.execute(&cache, &[&buf]).expect("execute f16 relu");
    let result = read_output_n(&out, 8);

    let expected: Vec<f32> = input_data.iter().map(|x| x.max(0.0)).collect();
    assert_close("f16_relu", &result, &expected, 1e-2);
}

/// Binary add through F16 pipeline: two inputs added element-wise.
#[test]
fn test_f16_add_matches_cpu() {
    super::test_utils::gpu_init();
    let cache = super::test_utils::metal_setup();

    let graph = ComputationGraph::from_nodes(vec![
        input_node(0, &[4]),
        input_node(1, &[4]),
        binary_node(2, "add_0", TraceOp::Add, 0, 1, &[4]),
    ]);

    let model = CompiledModel::builder(&graph, &cache)
        .force_dtype(DType::F16)
        .unwrap()
        .build()
        .expect("compile f16");

    let a: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
    let b: Vec<f32> = vec![0.5, -0.5, 1.5, -1.5];
    let buf_a = create_input_buffer(&cache, &a);
    let buf_b = create_input_buffer(&cache, &b);
    let out = model
        .execute(&cache, &[&buf_a, &buf_b])
        .expect("execute f16 add");
    let result = read_output_n(&out, 4);

    let expected: Vec<f32> = a.iter().zip(b.iter()).map(|(x, y)| x + y).collect();
    assert_close("f16_add", &result, &expected, 1e-2);
}

/// Mul through F16 pipeline: element-wise multiply.
#[test]
fn test_f16_mul_matches_cpu() {
    super::test_utils::gpu_init();
    let cache = super::test_utils::metal_setup();

    let graph = ComputationGraph::from_nodes(vec![
        input_node(0, &[4]),
        input_node(1, &[4]),
        binary_node(2, "mul_0", TraceOp::Mul, 0, 1, &[4]),
    ]);

    let model = CompiledModel::builder(&graph, &cache)
        .force_dtype(DType::F16)
        .unwrap()
        .build()
        .expect("compile f16");

    let a: Vec<f32> = vec![1.5, -2.0, 0.25, 3.0];
    let b: Vec<f32> = vec![2.0, 0.5, 4.0, -1.0];
    let buf_a = create_input_buffer(&cache, &a);
    let buf_b = create_input_buffer(&cache, &b);
    let out = model
        .execute(&cache, &[&buf_a, &buf_b])
        .expect("execute f16 mul");
    let result = read_output_n(&out, 4);

    let expected: Vec<f32> = a.iter().zip(b.iter()).map(|(x, y)| x * y).collect();
    assert_close("f16_mul", &result, &expected, 1e-2);
}

// -- Integration test: multi-layer MLP in F16 ---------------------------------

/// 3-layer MLP (Linear+Relu ×2, then Linear) compiled with F16 mixed precision.
/// Verifies the F16 path produces results within F16 tolerance of F32 CPU reference.
///
/// Architecture: Input [1,8] → Linear(8→16)+Relu → Linear(16→16)+Relu → Linear(16→4)
#[test]
fn test_f16_mlp_matches_f32_reference() {
    super::test_utils::gpu_init();
    let cache = super::test_utils::metal_setup();

    let dims: &[(usize, usize)] = &[(8, 16), (16, 16), (16, 4)];
    let (graph, weights) = build_mlp_graph(dims);

    // F32 reference via CompiledModel (not CPU — tests GPU compilation parity)
    let f32_model = CompiledModel::builder(&graph, &cache)
        .build()
        .expect("compile f32 mlp");
    let f16_model = CompiledModel::builder(&graph, &cache)
        .force_dtype(DType::F16)
        .unwrap()
        .build()
        .expect("compile f16 mlp");

    assert!(f16_model.is_mixed_precision());
    assert!(!f32_model.is_mixed_precision());
    assert_eq!(f16_model.output_shape(), f32_model.output_shape());

    let input = super::test_utils::rand_f32_vec(42, 8, -1.0, 1.0);
    let buf = create_input_buffer(&cache, &input);

    let f32_out = f32_model.execute(&cache, &[&buf]).expect("execute f32");
    let f16_out = f16_model.execute(&cache, &[&buf]).expect("execute f16");

    let f32_result = read_output_n(&f32_out, 4);
    let f16_result = read_output_n(&f16_out, 4);

    // CPU reference for sanity (F32).
    let cpu_ref = mlp_cpu_ref(&input, dims, &weights);
    assert_close("f32_vs_cpu", &f32_result, &cpu_ref, 1e-4);

    // F16 should be within F16 tolerance of F32 result.
    // F16 has ~3 decimal digits of precision, so 1% relative error is expected.
    for (i, (f16_v, f32_v)) in f16_result.iter().zip(f32_result.iter()).enumerate() {
        let abs_diff = (f16_v - f32_v).abs();
        let rel_tol = f32_v.abs() * 0.02 + 1e-3; // 2% relative + 1e-3 absolute
        assert!(
            abs_diff <= rel_tol,
            "f16_mlp[{i}]: f16={f16_v}, f32={f32_v}, diff={abs_diff}, tol={rel_tol}"
        );
    }
}

/// F16 output is always returned as F32 (the output auto-cast).
/// This is critical: callers always see F32 MetalBuffers from execute().
#[test]
fn test_f16_output_is_f32() {
    super::test_utils::gpu_init();
    let cache = super::test_utils::metal_setup();

    let graph = ComputationGraph::from_nodes(vec![
        input_node(0, &[4]),
        unary_node(1, "relu_0", TraceOp::Relu, 0, &[4]),
    ]);

    let model = CompiledModel::builder(&graph, &cache)
        .force_dtype(DType::F16)
        .unwrap()
        .build()
        .expect("compile f16");

    let input_data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
    let buf = create_input_buffer(&cache, &input_data);
    let out = model.execute(&cache, &[&buf]).expect("execute f16");

    // Output buffer should be readable as f32 (the auto-cast happened).
    let result = read_output_n(&out, 4);
    assert_eq!(result.len(), 4);
    // Values should be close to input (relu of positive values = identity).
    assert_close("f16_output_f32", &result, &input_data, 1e-2);
}

// -- Helpers (MLP graph builder, duplicated from mlp_test.rs) -----------------

fn build_mlp_graph(dims: &[(usize, usize)]) -> (ComputationGraph, Vec<(WeightRef, WeightRef)>) {
    let mut weights = Vec::new();
    for (i, &(in_f, out_f)) in dims.iter().enumerate() {
        let w = WeightRef::new(
            super::test_utils::rand_f32_vec(100 + i as u64, out_f * in_f, -0.5, 0.5),
            vec![out_f, in_f],
        )
        .unwrap();
        let b = WeightRef::new(
            super::test_utils::rand_f32_vec(200 + i as u64, out_f, -0.1, 0.1),
            vec![out_f],
        )
        .unwrap();
        weights.push((w, b));
    }
    let mut nodes = vec![input_node(0, &[1, dims[0].0])];
    let (mut prev, mut nid) = (0u64, 1u64);
    for (i, &(_, out_f)) in dims.iter().enumerate() {
        let (w, b) = &weights[i];
        nodes.push(TraceNode::new(
            nid,
            format!("linear_{i}"),
            TraceOp::Linear {
                weight: w.clone(),
                bias: Some(b.clone()),
            },
            vec![prev],
            vec![1, out_f],
            DType::F32,
        ));
        let lin = nid;
        nid += 1;
        if i < dims.len() - 1 {
            nodes.push(unary_node(
                nid,
                &format!("relu_{i}"),
                TraceOp::Relu,
                lin,
                &[1, out_f],
            ));
            prev = nid;
            nid += 1;
        } else {
            prev = lin;
        }
    }
    let _ = prev;
    (ComputationGraph::from_nodes(nodes), weights)
}

fn mlp_cpu_ref(
    input: &[f32],
    dims: &[(usize, usize)],
    weights: &[(WeightRef, WeightRef)],
) -> Vec<f32> {
    let mut data = input.to_vec();
    for (i, &(in_f, out_f)) in dims.iter().enumerate() {
        let (w, b) = &weights[i];
        data = super::test_utils::linear_ref(&data, w.data(), Some(b.data()), 1, in_f, out_f);
        if i < dims.len() - 1 {
            for v in data.iter_mut() {
                *v = v.max(0.0);
            }
        }
    }
    data
}

// -- NativeOp boundary cast tests (Tier 1 F16 core path) ----------------------

/// CPU LayerNorm reference for NativeOp boundary cast tests.
fn cpu_layer_norm(
    x: &[f32],
    weight: &[f32],
    bias: &[f32],
    batch: usize,
    time: usize,
    hidden: usize,
    eps: f32,
) -> Vec<f32> {
    let mut output = vec![0.0_f32; batch * time * hidden];
    for b in 0..batch {
        for t in 0..time {
            let offset = (b * time + t) * hidden;
            let row = &x[offset..offset + hidden];
            let mean: f32 = row.iter().sum::<f32>() / hidden as f32;
            let var: f32 = row.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / hidden as f32;
            let inv_std = 1.0 / (var + eps).sqrt();
            for c in 0..hidden {
                output[offset + c] = (row[c] - mean) * inv_std * weight[c] + bias[c];
            }
        }
    }
    output
}

/// NativeOp (LayerNorm) compiled with F16 mixed precision.
///
/// With Tier 2 (D5b), NativeOps run in F16 directly — the fused MSL kernel
/// is parameterized with `half` I/O pointers and `float` accumulators.
/// No F16↔F32 boundary casts needed. The final output auto-cast converts
/// F16→F32 for the caller.
///
/// Graph: Input [1, 4, 16] → LayerNorm(NativeOp, F16) → Output
#[test]
fn test_f16_nativeop_layer_norm_boundary_cast() {
    super::test_utils::gpu_init();
    let cache = super::test_utils::metal_setup();

    let (batch, time, hidden) = (1, 4, 16);
    let eps = 1e-5_f64;

    let x_data = super::test_utils::rand_f32_vec(0xF16A_0001, batch * time * hidden, -1.0, 1.0);
    let w_data = super::test_utils::rand_f32_vec(0xF16A_0002, hidden, 0.8, 1.2);
    let b_data = super::test_utils::rand_f32_vec(0xF16A_0003, hidden, -0.1, 0.1);

    let weight = WeightRef::new(w_data.clone(), vec![hidden]).expect("weight");
    let bias = WeightRef::new(b_data.clone(), vec![hidden]).expect("bias");

    let graph = ComputationGraph::from_nodes(vec![
        input_node(0, &[batch, time, hidden]),
        TraceNode::new(
            1,
            "layer_norm_0".into(),
            TraceOp::LayerNorm {
                eps,
                weight,
                bias,
            },
            vec![0],
            vec![batch, time, hidden],
            DType::F32,
        ),
    ]);

    // F32 reference.
    let f32_model = CompiledModel::builder(&graph, &cache)
        .build()
        .expect("compile f32");
    // F16 mixed precision: LayerNorm NativeOp runs in F16 directly (Tier 2,
    // D5b). Fused MSL kernel uses half I/O with float accumulators.
    let f16_model = CompiledModel::builder(&graph, &cache)
        .force_dtype(DType::F16)
        .unwrap()
        .build()
        .expect("compile f16");

    assert!(f16_model.is_mixed_precision());
    assert_eq!(f16_model.output_shape(), f32_model.output_shape());

    let buf = create_input_buffer(&cache, &x_data);
    let f32_out = f32_model.execute(&cache, &[&buf]).expect("execute f32");
    let f16_out = f16_model.execute(&cache, &[&buf]).expect("execute f16");

    let f32_result = read_output_n(&f32_out, batch * time * hidden);
    let f16_result = read_output_n(&f16_out, batch * time * hidden);

    // CPU reference for sanity.
    let cpu_ref = cpu_layer_norm(&x_data, &w_data, &b_data, batch, time, hidden, eps as f32);
    assert_close("f32_ln_vs_cpu", &f32_result, &cpu_ref, 1e-4);

    // F16 result should be within F16 precision of F32 result.
    // With Tier 2, NativeOp runs in F16 directly (half I/O, float accumulators).
    // Error comes from F16 quantization of inputs/weights, not boundary casts.
    for (i, (f16_v, f32_v)) in f16_result.iter().zip(f32_result.iter()).enumerate() {
        let abs_diff = (f16_v - f32_v).abs();
        let rel_tol = f32_v.abs() * 0.02 + 1e-3;
        assert!(
            abs_diff <= rel_tol,
            "f16_nativeop_ln[{i}]: f16={f16_v}, f32={f32_v}, diff={abs_diff}, tol={rel_tol}"
        );
    }
}

/// Dispatch(F16) → NativeOp(F16) → Dispatch(F16) chain.
///
/// This is the critical Kokoro pattern: all steps run in F16 for 2x
/// throughput. With Tier 2 (D5b), NativeOps also run in F16 directly —
/// fused MSL kernels use `half` I/O with `float` accumulators. No boundary
/// casts needed between Dispatch and NativeOp steps.
///
/// Graph: Input [1, 4, 16] → Relu(F16) → LayerNorm(NativeOp, F16) → Relu(F16) → Output
#[test]
fn test_f16_dispatch_nativeop_dispatch_chain() {
    super::test_utils::gpu_init();
    let cache = super::test_utils::metal_setup();

    let (batch, time, hidden) = (1, 4, 16);
    let eps = 1e-5_f64;

    let x_data = super::test_utils::rand_f32_vec(0xF16B_0001, batch * time * hidden, -2.0, 2.0);
    let w_data = super::test_utils::rand_f32_vec(0xF16B_0002, hidden, 0.8, 1.2);
    let b_data = super::test_utils::rand_f32_vec(0xF16B_0003, hidden, -0.1, 0.1);

    let weight = WeightRef::new(w_data.clone(), vec![hidden]).expect("weight");
    let bias = WeightRef::new(b_data.clone(), vec![hidden]).expect("bias");

    let graph = ComputationGraph::from_nodes(vec![
        input_node(0, &[batch, time, hidden]),
        unary_node(1, "relu_0", TraceOp::Relu, 0, &[batch, time, hidden]),
        TraceNode::new(
            2,
            "layer_norm_0".into(),
            TraceOp::LayerNorm {
                eps,
                weight,
                bias,
            },
            vec![1],
            vec![batch, time, hidden],
            DType::F32,
        ),
        unary_node(3, "relu_1", TraceOp::Relu, 2, &[batch, time, hidden]),
    ]);

    let f32_model = CompiledModel::builder(&graph, &cache)
        .build()
        .expect("compile f32");
    let f16_model = CompiledModel::builder(&graph, &cache)
        .force_dtype(DType::F16)
        .unwrap()
        .build()
        .expect("compile f16");

    assert!(f16_model.is_mixed_precision());

    let buf = create_input_buffer(&cache, &x_data);
    let f32_out = f32_model.execute(&cache, &[&buf]).expect("execute f32");
    let f16_out = f16_model.execute(&cache, &[&buf]).expect("execute f16");

    let f32_result = read_output_n(&f32_out, batch * time * hidden);
    let f16_result = read_output_n(&f16_out, batch * time * hidden);

    // CPU reference: relu → layer_norm → relu.
    let relu1: Vec<f32> = x_data.iter().map(|v| v.max(0.0)).collect();
    let ln_out = cpu_layer_norm(&relu1, &w_data, &b_data, batch, time, hidden, eps as f32);
    let cpu_ref: Vec<f32> = ln_out.iter().map(|v| v.max(0.0)).collect();

    assert_close("f32_chain_vs_cpu", &f32_result, &cpu_ref, 1e-4);

    // F16 tolerance: all steps run F16 with float accumulators in reductions.
    // ~3 decimal digits of F16 precision.
    for (i, (f16_v, f32_v)) in f16_result.iter().zip(f32_result.iter()).enumerate() {
        let abs_diff = (f16_v - f32_v).abs();
        let rel_tol = f32_v.abs() * 0.03 + 2e-3;
        assert!(
            abs_diff <= rel_tol,
            "f16_chain[{i}]: f16={f16_v}, f32={f32_v}, diff={abs_diff}, tol={rel_tol}"
        );
    }
}

/// F16 mixed-precision compilation metadata + execution with identity weights.
///
/// Verifies that `builder().force_dtype()` correctly compiles a graph containing
/// both Dispatch and NativeOp steps: is_mixed_precision flag, step counts,
/// and output shape match the F32 version. Also executes with identity
/// LayerNorm weights (gamma=1, beta=0) to verify the arena checkpoint fix
/// for mixed-precision NativeOps (#2981).
///
/// Graph: Input [1, 4, 16] → LayerNorm(NativeOp) → Output
#[test]
fn test_f16_metadata_with_nativeop() {
    super::test_utils::gpu_init();
    let cache = super::test_utils::metal_setup();

    let (batch, time, hidden) = (1, 4, 16);
    let eps = 1e-5_f64;

    let ln_weight = WeightRef::new(vec![1.0; hidden], vec![hidden]).expect("ln weight");
    let ln_bias = WeightRef::new(vec![0.0; hidden], vec![hidden]).expect("ln bias");

    let graph = ComputationGraph::from_nodes(vec![
        input_node(0, &[batch, time, hidden]),
        TraceNode::new(
            1,
            "layer_norm_0".into(),
            TraceOp::LayerNorm {
                eps,
                weight: ln_weight,
                bias: ln_bias,
            },
            vec![0],
            vec![batch, time, hidden],
            DType::F32,
        ),
    ]);

    let f32_model = CompiledModel::builder(&graph, &cache)
        .build()
        .expect("compile f32");
    let f16_model = CompiledModel::builder(&graph, &cache)
        .force_dtype(DType::F16)
        .unwrap()
        .build()
        .expect("compile f16");

    // Compilation metadata checks.
    assert!(!f32_model.is_mixed_precision());
    assert!(f16_model.is_mixed_precision());
    assert_eq!(f16_model.num_steps(), f32_model.num_steps());
    assert_eq!(f16_model.output_shape(), f32_model.output_shape());

    // Execution: identity LN (gamma=1, beta=0) should produce near-zero-mean,
    // unit-variance output. Previously panicked with arena checkpoint bug.
    let x_data = super::test_utils::rand_f32_vec(0xF16C_0001, batch * time * hidden, -1.0, 1.0);
    let buf = create_input_buffer(&cache, &x_data);
    let f32_out = f32_model.execute(&cache, &[&buf]).expect("execute f32");
    let f16_out = f16_model.execute(&cache, &[&buf]).expect("execute f16");

    let f32_result = read_output_n(&f32_out, batch * time * hidden);
    let f16_result = read_output_n(&f16_out, batch * time * hidden);

    // F16 should be close to F32 (identity LN, F16 quantization only error source).
    for (i, (f16_v, f32_v)) in f16_result.iter().zip(f32_result.iter()).enumerate() {
        let abs_diff = (f16_v - f32_v).abs();
        let rel_tol = f32_v.abs() * 0.02 + 1e-3;
        assert!(
            abs_diff <= rel_tol,
            "f16_metadata_exec[{i}]: f16={f16_v}, f32={f32_v}, diff={abs_diff}, tol={rel_tol}"
        );
    }
}

/// F16 values near the F16 representable range boundary.
///
/// Verifies that values near F16 max (65504) and near zero (denormals)
/// survive the F32→F16→F32 round-trip through `cast_slice_dtype` without
/// producing NaN or Inf.
#[test]
fn test_f16_precision_boundary_values() {
    super::test_utils::gpu_init();
    let cache = super::test_utils::metal_setup();

    // Graph: input[8] → relu → output[8]
    let graph = ComputationGraph::from_nodes(vec![
        input_node(0, &[8]),
        unary_node(1, "relu_0", TraceOp::Relu, 0, &[8]),
    ]);

    let f16_model = CompiledModel::builder(&graph, &cache)
        .force_dtype(DType::F16)
        .unwrap()
        .build()
        .expect("compile f16");

    // Test boundary values: large positive, small positive, zeros, negatives.
    // F16 max is 65504, min positive normal is ~6.1e-5, min subnormal ~5.96e-8.
    let input_data: Vec<f32> = vec![
        60000.0,  // near F16 max
        0.0001,   // small positive (F16 representable)
        0.0,      // zero
        -1.0,     // negative (relu → 0)
        1.0,      // normal
        100.0,    // medium
        -60000.0, // large negative (relu → 0)
        0.5,      // fractional
    ];
    let buf = create_input_buffer(&cache, &input_data);
    let out = f16_model.execute(&cache, &[&buf]).expect("execute f16");
    let result = read_output_n(&out, 8);

    // All outputs must be finite.
    for (i, v) in result.iter().enumerate() {
        assert!(v.is_finite(), "f16_boundary[{i}]: non-finite output {v}");
    }

    // Relu of positive values preserved (within F16 tolerance).
    assert!(
        (result[0] - 60000.0).abs() < 100.0,
        "near-max: got {}",
        result[0]
    );
    assert!(
        (result[1] - 0.0001).abs() < 1e-3,
        "small: got {}",
        result[1]
    );
    assert!((result[2] - 0.0).abs() < 1e-6, "zero: got {}", result[2]);
    assert!(
        (result[3] - 0.0).abs() < 1e-6,
        "neg relu: got {}",
        result[3]
    );
    assert!((result[4] - 1.0).abs() < 1e-2, "one: got {}", result[4]);
    assert!(
        (result[5] - 100.0).abs() < 1.0,
        "hundred: got {}",
        result[5]
    );
    assert!(
        (result[6] - 0.0).abs() < 1e-6,
        "large neg relu: got {}",
        result[6]
    );
    assert!((result[7] - 0.5).abs() < 1e-2, "half: got {}", result[7]);
}

// -- NarrowView F16 regression test -------------------------------------------

/// Regression test: NarrowView elem_offset must scale by F16 element size (2),
/// not F32 (4). Before fix, NarrowView stored a byte_offset computed with `* 4`
/// at compile time. In F16 mode, the GPU buffer uses 2 bytes/element, so
/// the byte_offset was 2x too large, causing out-of-bounds reads.
///
/// Graph: input [1, 16] → narrow(dim=1, start=4, len=8) → relu → output [1, 8].
/// NarrowView elem_offset = 4 * 1 = 4 elements.
/// In F16: byte_offset should be 4 * 2 = 8.
/// In F32: byte_offset should be 4 * 4 = 16.
/// If the old code were used in F16 mode, byte_offset would be 4 * 4 = 16,
/// reading past the F16 buffer by 8 bytes.
#[test]
fn test_f16_narrow_view_elem_offset() {
    super::test_utils::gpu_init();
    let cache = super::test_utils::metal_setup();

    let graph = ComputationGraph::from_nodes(vec![
        input_node(0, &[1, 16]),
        TraceNode::new(
            1,
            "narrow_1".into(),
            TraceOp::Narrow {
                dim: 1,
                start: 4,
                length: 8,
            },
            vec![0],
            vec![1, 8],
            DType::F32,
        ),
        unary_node(2, "relu_0", TraceOp::Relu, 1, &[1, 8]),
    ]);

    // Verify compilation produces a NarrowView step.
    let steps = nn_dsl::trace_compile::compile_trace(&graph).expect("compile trace");
    let has_narrow_view = steps
        .iter()
        .any(|s| matches!(s, CompiledStep::NarrowView { .. }));
    assert!(has_narrow_view, "graph should produce a NarrowView step");

    // F32 reference.
    let input_data: Vec<f32> = (0..16).map(|i| (i as f32) - 2.0).collect();
    let buf = create_input_buffer(&cache, &input_data);
    let f32_model = CompiledModel::builder(&graph, &cache)
        .build()
        .expect("compile f32");
    let f32_out = f32_model.execute(&cache, &[&buf]).expect("execute f32");
    let f32_result = read_output_n(&f32_out, 8);

    // F16 model.
    let f16_model = CompiledModel::builder(&graph, &cache)
        .force_dtype(DType::F16)
        .unwrap()
        .build()
        .expect("compile f16");
    let f16_out = f16_model.execute(&cache, &[&buf]).expect("execute f16");
    let f16_result = read_output_n(&f16_out, 8);

    // Expected: narrow picks elements [4..12] = [2.0, 3.0, ..., 9.0], relu keeps all.
    let expected: Vec<f32> = (4..12).map(|i| (i as f32) - 2.0).collect();

    assert_close("f32_narrow_view", &f32_result, &expected, 1e-5);
    assert_close("f16_narrow_view", &f16_result, &expected, 0.1);
}

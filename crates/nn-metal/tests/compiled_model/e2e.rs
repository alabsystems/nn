// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! End-to-end integration tests for the compiled model pipeline.
//!
//! These tests exercise the full path:
//!   1. Construct a real nn model using `DynTensor` + `Module`
//!   2. Trace the forward pass via `trace_graph()`
//!   3. Compile the trace into a `CompiledModel`
//!   4. Execute the compiled model on GPU
//!   5. Verify numerical correctness against the DynTensor output
//!
//! This validates that the trace recording, trace compilation, and
//! CompiledModel executor are compatible end-to-end -- the D4 integration
//! test from the compile-time graph execution design (#2080).

use nn_core::dyn_tensor::trace::{
    record_input, trace_graph, ComputationGraph, TraceNode, TraceOp,
};
use nn_core::dyn_tensor::DynTensor;
use nn_core::layers::{Activation, Linear, Module};
use nn_core::{DType, Device};
use nn_metal::compiled_model::CompiledModel;
use nn_metal::MetalElement;

use super::helpers::{create_input_buffer, read_output};

fn cpu() -> Device {
    Device::Cpu
}

fn rand_linear(seed_w: u64, seed_b: u64, out: usize, inp: usize) -> Linear {
    let w = DynTensor::new(
        &super::test_utils::rand_f32_vec(seed_w, out * inp, -0.5, 0.5),
        &[out, inp],
        &cpu(),
    )
    .unwrap();
    let b = DynTensor::new(
        &super::test_utils::rand_f32_vec(seed_b, out, -0.1, 0.1),
        &[out],
        &cpu(),
    )
    .unwrap();
    Linear::new(w, Some(b)).unwrap()
}

// -- Test: Single Linear layer ------------------------------------------------

/// Single Linear layer: trace -> compile -> execute -> verify against DynTensor.
#[test]
fn test_e2e_compiled_linear() {
    super::test_utils::gpu_init();
    let cache = super::test_utils::metal_setup();

    // Build a Linear(4, 3) with known weights.
    let weight = DynTensor::new(
        &[1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
        &[3, 4],
        &cpu(),
    )
    .unwrap();
    let bias = DynTensor::new(&[0.1, 0.2, 0.3], &[3], &cpu()).unwrap();
    let linear = Linear::new(weight, Some(bias)).unwrap();

    // Input: [2, 4] -- batch of 2.
    let input_data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let x = DynTensor::new(&input_data, &[2, 4], &cpu()).unwrap();

    // Run DynTensor reference.
    let reference = linear.forward(&x).unwrap();
    let ref_vals = reference.to_flat_vec::<f32>().unwrap();

    // Trace the forward pass.
    let (_traced_output, graph) = trace_graph(|| {
        let mut input = x.clone();
        let id = record_input(&[2, 4], DType::F32).expect("tracing active");
        input.set_trace_id(id);
        let y = linear.forward(&input)?;
        Ok(y)
    })
    .expect("trace_graph should succeed");

    // Compile and execute.
    let compiled = CompiledModel::builder(&graph, &cache)
        .build()
        .expect("compile from trace");
    assert!(compiled.num_inputs() >= 1, "should have at least 1 input");

    let input_buf = create_input_buffer(&cache, &input_data);
    let out_buf = compiled
        .execute(&cache, &[&input_buf])
        .expect("execute compiled linear");
    let gpu_vals = read_output(&out_buf);
    assert_eq!(
        gpu_vals.len(),
        ref_vals.len(),
        "GPU/ref output length mismatch in e2e linear"
    );
    for (i, (g, r)) in gpu_vals.iter().zip(ref_vals.iter()).enumerate() {
        assert!(
            (g - r).abs() < 1e-4,
            "e2e linear[{i}]: gpu={g}, ref={r}, diff={}",
            (g - r).abs()
        );
    }
}

// -- Test: Linear + ReLU chain ------------------------------------------------

/// Linear(4,3) -> ReLU: two-op model traced end-to-end.
#[test]
fn test_e2e_compiled_linear_relu() {
    super::test_utils::gpu_init();
    let cache = super::test_utils::metal_setup();

    // Linear(4, 3) with negative bias to produce some negative outputs.
    let weight = DynTensor::new(
        &[1.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
        &[3, 4],
        &cpu(),
    )
    .unwrap();
    let bias = DynTensor::new(&[-0.5, 0.0, -2.0], &[3], &cpu()).unwrap();
    let linear = Linear::new(weight, Some(bias)).unwrap();
    let relu = Activation::Relu;

    let input_data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let x = DynTensor::new(&input_data, &[2, 4], &cpu()).unwrap();

    // DynTensor reference.
    let lin_out = linear.forward(&x).unwrap();
    let reference = relu.forward(&lin_out).unwrap();
    let ref_vals = reference.to_flat_vec::<f32>().unwrap();

    // Trace.
    let (_traced_output, graph) = trace_graph(|| {
        let mut input = x.clone();
        let id = record_input(&[2, 4], DType::F32).expect("tracing active");
        input.set_trace_id(id);
        let h = linear.forward(&input)?;
        let y = relu.forward(&h)?;
        Ok(y)
    })
    .expect("trace_graph should succeed");

    // Compile and execute.
    let compiled = CompiledModel::builder(&graph, &cache)
        .build()
        .expect("compile linear+relu");
    let input_buf = create_input_buffer(&cache, &input_data);
    let out_buf = compiled
        .execute(&cache, &[&input_buf])
        .expect("execute linear+relu");
    let gpu_vals = read_output(&out_buf);
    assert_eq!(
        gpu_vals.len(),
        ref_vals.len(),
        "GPU/ref output length mismatch in e2e linear_relu"
    );
    for (i, (g, r)) in gpu_vals.iter().zip(ref_vals.iter()).enumerate() {
        assert!(
            (g - r).abs() < 1e-4,
            "e2e linear_relu[{i}]: gpu={g}, ref={r}",
        );
    }
}

// -- Test: 5-layer MLP (5 Linear + 4 GELU activations) -----------------------

/// MLP with 5 Linear layers and 4 GELU activations, matching the D4 spec
/// from the compile-time graph execution design doc (#2080, #2124 AC3).
///
/// Architecture: Linear(4,8) -> GELU -> Linear(8,16) -> GELU -> Linear(16,16) ->
///               GELU -> Linear(16,8) -> GELU -> Linear(8,3)
#[test]
fn test_e2e_compiled_mlp_5layer() {
    super::test_utils::gpu_init();
    let cache = super::test_utils::metal_setup();

    let linear1 = rand_linear(42, 43, 8, 4);
    let linear2 = rand_linear(44, 45, 16, 8);
    let linear3 = rand_linear(46, 47, 16, 16);
    let linear4 = rand_linear(48, 49, 8, 16);
    let linear5 = rand_linear(50, 51, 3, 8);

    let gelu = Activation::Gelu;

    let input_data = super::test_utils::rand_f32_vec(100, 2 * 4, -1.0, 1.0);
    let x = DynTensor::new(&input_data, &[2, 4], &cpu()).unwrap();

    // DynTensor reference: 5 Linear layers interleaved with 4 GELU activations.
    let h1 = gelu.forward(&linear1.forward(&x).unwrap()).unwrap();
    let h2 = gelu.forward(&linear2.forward(&h1).unwrap()).unwrap();
    let h3 = gelu.forward(&linear3.forward(&h2).unwrap()).unwrap();
    let h4 = gelu.forward(&linear4.forward(&h3).unwrap()).unwrap();
    let reference = linear5.forward(&h4).unwrap();
    let ref_vals = reference.to_flat_vec::<f32>().unwrap();

    // Trace.
    let (_traced_output, graph) = trace_graph(|| {
        let mut input = x.clone();
        let id = record_input(&[2, 4], DType::F32).expect("tracing active");
        input.set_trace_id(id);
        let h1 = gelu.forward(&linear1.forward(&input)?)?;
        let h2 = gelu.forward(&linear2.forward(&h1)?)?;
        let h3 = gelu.forward(&linear3.forward(&h2)?)?;
        let h4 = gelu.forward(&linear4.forward(&h3)?)?;
        let y = linear5.forward(&h4)?;
        Ok(y)
    })
    .expect("trace_graph should succeed");

    // Compile and execute.
    let compiled = CompiledModel::builder(&graph, &cache)
        .build()
        .expect("compile 5-layer MLP");
    assert_eq!(compiled.num_inputs(), 1);
    assert_eq!(compiled.output_shape(), &[2, 3]);

    // Dispatch count reduction: 5 Linear + 4 GELU = 9 total ops.
    let num_dispatches = compiled.num_dispatches();
    assert!(
        num_dispatches <= 9,
        "5-layer MLP should have at most 9 dispatches (5 Linear + 4 GELU), got {num_dispatches}"
    );

    let input_buf = create_input_buffer(&cache, &input_data);
    let out_buf = compiled
        .execute(&cache, &[&input_buf])
        .expect("execute 5-layer MLP");
    let gpu_vals = read_output(&out_buf);
    assert_eq!(
        gpu_vals.len(),
        ref_vals.len(),
        "GPU/ref output length mismatch in e2e mlp"
    );
    for (i, (g, r)) in gpu_vals.iter().zip(ref_vals.iter()).enumerate() {
        assert!(
            (g - r).abs() < 1e-5,
            "e2e mlp[{i}]: gpu={g}, ref={r}, diff={} exceeds 1e-5",
            (g - r).abs()
        );
    }
}

// -- Test: Model reuse (multiple forward passes) ------------------------------

/// Verify that CompiledModel can be reused with different input data.
#[test]
fn test_e2e_compiled_model_reuse() {
    super::test_utils::gpu_init();
    let cache = super::test_utils::metal_setup();

    let weight = DynTensor::new(&[2.0, 0.0, 0.0, 3.0], &[2, 2], &cpu()).unwrap();
    let bias = DynTensor::new(&[1.0, -1.0], &[2], &cpu()).unwrap();
    let linear = Linear::new(weight, Some(bias)).unwrap();
    let relu = Activation::Relu;

    let x1_data = vec![1.0, 1.0];
    let x2_data = vec![-1.0, -1.0];
    let x1 = DynTensor::new(&x1_data, &[1, 2], &cpu()).unwrap();
    let x2 = DynTensor::new(&x2_data, &[1, 2], &cpu()).unwrap();

    // Reference outputs.
    let ref1 = relu
        .forward(&linear.forward(&x1).unwrap())
        .unwrap()
        .to_flat_vec::<f32>()
        .unwrap();
    let ref2 = relu
        .forward(&linear.forward(&x2).unwrap())
        .unwrap()
        .to_flat_vec::<f32>()
        .unwrap();

    // Trace once.
    let (_, graph) = trace_graph(|| {
        let mut input = x1.clone();
        let id = record_input(&[1, 2], DType::F32).expect("tracing active");
        input.set_trace_id(id);
        let y = relu.forward(&linear.forward(&input)?)?;
        Ok(y)
    })
    .unwrap();

    let compiled = CompiledModel::builder(&graph, &cache)
        .build()
        .expect("compile");

    // Execute twice with different data.
    let buf1 = create_input_buffer(&cache, &x1_data);
    let out1_buf = compiled.execute(&cache, &[&buf1]).expect("exec 1");
    let out1 = read_output(&out1_buf);

    let buf2 = create_input_buffer(&cache, &x2_data);
    let out2_buf = compiled.execute(&cache, &[&buf2]).expect("exec 2");
    let out2 = read_output(&out2_buf);

    assert_eq!(
        out1.len(),
        ref1.len(),
        "GPU/ref output length mismatch in reuse_x1"
    );
    for (i, (g, r)) in out1.iter().zip(ref1.iter()).enumerate() {
        assert!((g - r).abs() < 1e-5, "reuse_x1[{i}]: gpu={g}, ref={r}");
    }
    assert_eq!(
        out2.len(),
        ref2.len(),
        "GPU/ref output length mismatch in reuse_x2"
    );
    for (i, (g, r)) in out2.iter().zip(ref2.iter()).enumerate() {
        assert!((g - r).abs() < 1e-5, "reuse_x2[{i}]: gpu={g}, ref={r}");
    }
}

// -- Test: compile_forward convenience API ------------------------------------

/// Verify that `compile_forward` produces the same results as manual
/// `trace_graph` + `from_trace`, with zero input-registration boilerplate.
#[test]
#[allow(deprecated)]
fn test_e2e_compile_forward_linear_relu() {
    super::test_utils::gpu_init();
    let cache = super::test_utils::metal_setup();

    let weight = DynTensor::new(
        &[1.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
        &[3, 4],
        &cpu(),
    )
    .unwrap();
    let bias = DynTensor::new(&[-0.5, 0.0, -2.0], &[3], &cpu()).unwrap();
    let linear = Linear::new(weight, Some(bias)).unwrap();
    let relu = Activation::Relu;

    let input_data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let x = DynTensor::new(&input_data, &[2, 4], &cpu()).unwrap();

    // DynTensor reference.
    let reference = relu.forward(&linear.forward(&x).unwrap()).unwrap();
    let ref_vals = reference.to_flat_vec::<f32>().unwrap();

    // compile_forward: one call replaces trace_graph + record_input +
    // set_trace_id + from_trace.
    let compiled = CompiledModel::compile_forward(
        &[&x],
        |traced_inputs| {
            let h = linear.forward(&traced_inputs[0])?;
            relu.forward(&h)
        },
        &cache,
    )
    .expect("compile_forward should succeed");

    assert_eq!(compiled.num_inputs(), 1);
    assert_eq!(compiled.output_shape(), &[2, 3]);

    let input_buf = create_input_buffer(&cache, &input_data);
    let out_buf = compiled
        .execute(&cache, &[&input_buf])
        .expect("execute compiled linear+relu");
    let gpu_vals = read_output(&out_buf);

    assert_eq!(
        gpu_vals.len(),
        ref_vals.len(),
        "GPU/ref output length mismatch in compile_forward linear_relu"
    );
    for (i, (g, r)) in gpu_vals.iter().zip(ref_vals.iter()).enumerate() {
        assert!(
            (g - r).abs() < 1e-4,
            "compile_forward linear_relu[{i}]: gpu={g}, ref={r}",
        );
    }
}

/// Verify `compile_forward` on a 3-layer MLP with fusion.
#[test]
#[allow(deprecated)]
fn test_e2e_compile_forward_mlp() {
    super::test_utils::gpu_init();
    let cache = super::test_utils::metal_setup();

    let linear1 = rand_linear(42, 43, 8, 4);
    let linear2 = rand_linear(44, 45, 8, 8);
    let linear3 = rand_linear(46, 47, 3, 8);
    let gelu = Activation::Gelu;

    let input_data = super::test_utils::rand_f32_vec(100, 2 * 4, -1.0, 1.0);
    let x = DynTensor::new(&input_data, &[2, 4], &cpu()).unwrap();

    // DynTensor reference.
    let h1 = gelu.forward(&linear1.forward(&x).unwrap()).unwrap();
    let h2 = gelu.forward(&linear2.forward(&h1).unwrap()).unwrap();
    let reference = linear3.forward(&h2).unwrap();
    let ref_vals = reference.to_flat_vec::<f32>().unwrap();

    // compile_forward: single call for the full pipeline.
    let compiled = CompiledModel::compile_forward(
        &[&x],
        |traced_inputs| {
            let h1 = gelu.forward(&linear1.forward(&traced_inputs[0])?)?;
            let h2 = gelu.forward(&linear2.forward(&h1)?)?;
            linear3.forward(&h2)
        },
        &cache,
    )
    .expect("compile_forward MLP");

    assert_eq!(compiled.num_inputs(), 1);
    assert_eq!(compiled.output_shape(), &[2, 3]);

    let input_buf = create_input_buffer(&cache, &input_data);
    let out_buf = compiled
        .execute(&cache, &[&input_buf])
        .expect("execute compiled MLP");
    let gpu_vals = read_output(&out_buf);
    assert_eq!(
        gpu_vals.len(),
        ref_vals.len(),
        "GPU/ref output length mismatch in compile_forward mlp"
    );
    for (i, (g, r)) in gpu_vals.iter().zip(ref_vals.iter()).enumerate() {
        assert!(
            (g - r).abs() < 1e-3,
            "compile_forward mlp[{i}]: gpu={g}, ref={r}, diff={}",
            (g - r).abs()
        );
    }
}

// -- Test: Permute dispatch (#2168) -------------------------------------------

/// Verify that `TraceOp::Permute { axes: [0, 2, 1] }` on a [2, 3, 4] tensor
/// compiles to a GPU dispatch (not passthrough) and produces correct output.
#[test]
fn test_e2e_compiled_permute() {
    super::test_utils::gpu_init();
    let cache = super::test_utils::metal_setup();

    // Input: [2, 3, 4] contiguous. Permute axes [0, 2, 1] -> output [2, 4, 3].
    let shape = [2, 3, 4];
    let numel: usize = shape.iter().product();
    let input_data: Vec<f32> = (0..numel).map(|i| i as f32).collect();

    // CPU reference: permute [B, H, W] -> [B, W, H] (axes [0, 2, 1]).
    let mut expected = vec![0.0_f32; numel];
    for b in 0..2 {
        for h in 0..3 {
            for w in 0..4 {
                let src_idx = b * 12 + h * 4 + w;
                let dst_idx = b * 12 + w * 3 + h;
                expected[dst_idx] = input_data[src_idx];
            }
        }
    }

    let graph = ComputationGraph::from_nodes(vec![
        TraceNode::new(
            0,
            "input_0".into(),
            TraceOp::Input,
            vec![],
            shape.to_vec(),
            DType::F32,
        ),
        TraceNode::new(
            1,
            "permute_0".into(),
            TraceOp::Permute {
                axes: vec![0, 2, 1],
            },
            vec![0],
            vec![2, 4, 3],
            DType::F32,
        ),
    ]);

    let compiled = CompiledModel::builder(&graph, &cache)
        .build()
        .expect("compile permute");
    assert_eq!(compiled.num_inputs(), 1);
    assert_eq!(compiled.output_shape(), &[2, 4, 3]);
    assert!(
        compiled.num_dispatches() >= 1,
        "Permute must compile to at least 1 dispatch, got {}",
        compiled.num_dispatches()
    );

    let input_buf = create_input_buffer(&cache, &input_data);
    let out_buf = compiled
        .execute(&cache, &[&input_buf])
        .expect("execute permute");
    let result = read_output(&out_buf);

    assert_eq!(
        result.len(),
        expected.len(),
        "GPU/ref output length mismatch in permute"
    );
    for (i, (g, e)) in result.iter().zip(expected.iter()).enumerate() {
        assert!((g - e).abs() < 1e-6, "permute[{i}]: gpu={g}, expected={e}");
    }
}

// -- Test: Linear + LeakyRelu (compiled vs eager parity, #2223) ---------------

/// Linear(4,3) -> LeakyRelu: verifies compiled LeakyRelu output matches
/// eager DynTensor execution within epsilon.
///
/// The compiled path decomposes LeakyRelu as `relu(x) - 0.01 * relu(-x)`,
/// matching the hardcoded negative_slope=0.01 default.
#[test]
fn test_e2e_compiled_linear_leaky_relu() {
    super::test_utils::gpu_init();
    let cache = super::test_utils::metal_setup();

    // Linear(4, 3) with negative bias to produce some negative outputs for
    // LeakyRelu to exercise both the positive and negative branches.
    let weight = DynTensor::new(
        &[1.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
        &[3, 4],
        &cpu(),
    )
    .unwrap();
    let bias = DynTensor::new(&[-0.5, 0.0, -2.0], &[3], &cpu()).unwrap();
    let linear = Linear::new(weight, Some(bias)).unwrap();
    // negative_slope=0.01 must match the compiled path's hardcoded default.
    let leaky_relu = Activation::LeakyRelu(0.01);

    let input_data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let x = DynTensor::new(&input_data, &[2, 4], &cpu()).unwrap();

    // DynTensor eager reference.
    let reference = leaky_relu.forward(&linear.forward(&x).unwrap()).unwrap();
    let ref_vals = reference.to_flat_vec::<f32>().unwrap();

    // Trace the forward pass.
    let (_traced_output, graph) = trace_graph(|| {
        let mut input = x.clone();
        let id = record_input(&[2, 4], DType::F32).expect("tracing active");
        input.set_trace_id(id);
        let h = linear.forward(&input)?;
        let y = leaky_relu.forward(&h)?;
        Ok(y)
    })
    .expect("trace_graph should succeed");

    // Compile and execute on GPU.
    let compiled = CompiledModel::builder(&graph, &cache)
        .build()
        .expect("compile linear+leaky_relu");
    let input_buf = create_input_buffer(&cache, &input_data);
    let out_buf = compiled
        .execute(&cache, &[&input_buf])
        .expect("execute linear+leaky_relu");
    let gpu_vals = read_output(&out_buf);
    assert_eq!(
        gpu_vals.len(),
        ref_vals.len(),
        "GPU/ref output length mismatch in e2e linear_leaky_relu"
    );
    for (i, (g, r)) in gpu_vals.iter().zip(ref_vals.iter()).enumerate() {
        assert!(
            (g - r).abs() < 1e-4,
            "e2e linear_leaky_relu[{i}]: gpu={g}, ref={r}, diff={}",
            (g - r).abs()
        );
    }
}

// -- Test: Arena offset propagation (regression for #2167) --------------------

/// Regression test for #2167: multi-step dispatch with arena-allocated
/// intermediate buffers must produce correct results.
#[test]
fn test_e2e_arena_offset_regression_2167() {
    super::test_utils::gpu_init();
    let cache = super::test_utils::metal_setup();

    // 3-layer MLP: Linear(4,8) -> ReLU -> Linear(8,8) -> ReLU -> Linear(8,2)
    let linear1 = rand_linear(200, 201, 8, 4);
    let linear2 = rand_linear(202, 203, 8, 8);
    let linear3 = rand_linear(204, 205, 2, 8);
    let relu = Activation::Relu;

    let input_data = super::test_utils::rand_f32_vec(300, 4, -1.0, 1.0);
    let x = DynTensor::new(&input_data, &[1, 4], &cpu()).unwrap();

    // CPU reference.
    let h1 = relu.forward(&linear1.forward(&x).unwrap()).unwrap();
    let h2 = relu.forward(&linear2.forward(&h1).unwrap()).unwrap();
    let reference = linear3.forward(&h2).unwrap();
    let ref_vals = reference.to_flat_vec::<f32>().unwrap();

    // Trace -> compile -> execute.
    let (_, graph) = trace_graph(|| {
        let mut input = x.clone();
        let id = record_input(&[1, 4], DType::F32).expect("tracing active");
        input.set_trace_id(id);
        let h1 = relu.forward(&linear1.forward(&input)?)?;
        let h2 = relu.forward(&linear2.forward(&h1)?)?;
        linear3.forward(&h2)
    })
    .expect("trace_graph");

    let compiled = CompiledModel::builder(&graph, &cache)
        .build()
        .expect("compile");

    // Execute multiple times to exercise arena reuse across forward passes.
    for pass in 0..3 {
        let input_buf = create_input_buffer(&cache, &input_data);
        let out_buf = compiled
            .execute(&cache, &[&input_buf])
            .expect("execute pass");
        let gpu_vals = read_output(&out_buf);

        assert_eq!(gpu_vals.len(), ref_vals.len());
        for (i, (g, r)) in gpu_vals.iter().zip(ref_vals.iter()).enumerate() {
            assert!(
                (g - r).abs() < 1e-3,
                "arena_offset pass {pass} [{i}]: gpu={g}, ref={r}, diff={}",
                (g - r).abs()
            );
        }
    }
}

// -- Test: Fusion dispatch count reduction (e2e with real nn modules) ----------

/// Verifies that elementwise chain fusion reduces GPU dispatch count
/// through the full trace -> compile -> execute pipeline with real nn modules.
///
/// Linear(4,3) -> Relu -> Sigmoid:
/// - Without fusion: 1 matmul + 1 relu + 1 sigmoid = 3 dispatches
/// - With fusion: 1 matmul + 1 fused(relu+sigmoid) = 2 dispatches
///
/// This is the acceptance criteria integration test for #2126 Task 11.
#[test]
fn test_e2e_fusion_reduces_dispatch_count() {
    super::test_utils::gpu_init();
    let cache = super::test_utils::metal_setup();

    let weight = DynTensor::new(
        &[1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
        &[3, 4],
        &cpu(),
    )
    .unwrap();
    let bias = DynTensor::new(&[0.1, -0.2, 0.3], &[3], &cpu()).unwrap();
    let linear = Linear::new(weight, Some(bias)).unwrap();
    let relu = Activation::Relu;

    let input_data = vec![1.0_f32, -2.0, 0.5, 3.0, -1.0, 0.0, 2.0, -0.5];
    let x = DynTensor::new(&input_data, &[2, 4], &cpu()).unwrap();

    // DynTensor eager reference: linear -> relu -> sigmoid.
    let lin_out = linear.forward(&x).unwrap();
    let relu_out = relu.forward(&lin_out).unwrap();
    let reference = relu_out.sigmoid().unwrap();
    let ref_vals = reference.to_flat_vec::<f32>().unwrap();

    // Trace the forward pass.
    let (_traced, graph) = trace_graph(|| {
        let mut input = x.clone();
        let id = record_input(&[2, 4], DType::F32).expect("tracing active");
        input.set_trace_id(id);
        let h = linear.forward(&input)?;
        let h = relu.forward(&h)?;
        let y = h.sigmoid()?;
        Ok(y)
    })
    .expect("trace_graph");

    let compiled = CompiledModel::builder(&graph, &cache)
        .build()
        .expect("compile");

    // Key assertion: fusion should reduce dispatch count.
    // Without fusion: matmul(1) + relu(1) + sigmoid(1) = 3 dispatches.
    // With fusion: relu+sigmoid fuse -> matmul(1) + fused(1) = 2 dispatches.
    let dispatches = compiled.num_dispatches();
    assert!(
        dispatches < 3,
        "fusion should reduce dispatches from 3 to 2, got {dispatches}"
    );
    eprintln!(
        "fusion dispatch count: {dispatches} (steps: {}, inputs: {})",
        compiled.num_steps(),
        compiled.num_inputs(),
    );

    // Verify numerical correctness.
    let input_buf = create_input_buffer(&cache, &input_data);
    let out_buf = compiled
        .execute(&cache, &[&input_buf])
        .expect("execute fused");
    let output_numel = ref_vals.len();
    let gpu_vals = f32::read_buffer_at_offset(&out_buf, 0, output_numel).expect("read GPU output");

    assert_eq!(
        gpu_vals.len(),
        ref_vals.len(),
        "GPU/ref output length mismatch in fusion_dispatch"
    );
    for (i, (g, r)) in gpu_vals.iter().zip(ref_vals.iter()).enumerate() {
        assert!(
            (g - r).abs() < 1e-5,
            "fusion_dispatch[{i}]: gpu={g}, ref={r}, diff={}",
            (g - r).abs()
        );
    }
}

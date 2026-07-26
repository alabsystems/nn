// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! End-to-end tests: real nn module trace → compile_forward → GPU execute.
//!
//! Unlike `compiled_model_ops_e2e.rs` (which constructs ComputationGraphs
//! manually), these tests exercise the full pipeline including weight capture
//! during tracing. This verifies that `traced_forward()` correctly records
//! weight data into `WeightRef` structures that the compiler and executor
//! can consume.
//!
//! Part of #2270.

use nn_core::dyn_tensor::trace;
use nn_core::dyn_tensor::DynTensor;
use nn_core::layers::{Activation, Conv1d, Conv1dConfig, LayerNorm, Linear, Module};
use nn_core::{DType, Device};
use nn_metal::compiled_model::CompiledModel;

fn cpu() -> Device {
    Device::Cpu
}

fn gpu() -> Device {
    Device::Metal { device_id: 0 }
}

use super::helpers::assert_close;

// -- Test 1: Conv1d via compile_forward ---------------------------------------

/// Conv1d(2, 4, 3, pad=1) + ReLU via compile_forward.
///
/// Verifies that tracing a real `Conv1d` module captures weight data
/// correctly in the computation graph, and that compile_forward produces
/// the same output as eager DynTensor execution.
#[test]
#[allow(deprecated)]
fn test_trace_compile_conv1d_relu() {
    super::test_utils::gpu_init();
    let cache = super::test_utils::metal_setup();

    let (in_ch, out_ch, ks) = (2, 4, 3);
    let config = Conv1dConfig::default().with_padding(1);
    let w = DynTensor::new(
        &super::test_utils::rand_f32_vec(0xAC01, out_ch * in_ch * ks, -0.5, 0.5),
        &[out_ch, in_ch, ks],
        &cpu(),
    )
    .unwrap();
    let b = DynTensor::new(
        &super::test_utils::rand_f32_vec(0xAC02, out_ch, -0.1, 0.1),
        &[out_ch],
        &cpu(),
    )
    .unwrap();
    let conv = Conv1d::new(w, Some(b), config).unwrap();
    let relu = Activation::Relu;

    // Input: [1, 2, 16] — batch=1, in_channels=2, length=16
    let in_len = 16;
    let input_data = super::test_utils::rand_f32_vec(0xAC03, in_ch * in_len, -1.0, 1.0);
    let x = DynTensor::new(&input_data, &[1, in_ch, in_len], &cpu()).unwrap();

    // DynTensor eager reference.
    let ref_out = relu.forward(&conv.forward(&x).unwrap()).unwrap();
    let ref_vals = ref_out.to_flat_vec::<f32>().unwrap();

    // compile_forward: trace real Conv1d + ReLU modules.
    let compiled = CompiledModel::compile_forward(
        &[&x],
        |traced| {
            let h = conv.forward(&traced[0])?;
            relu.forward(&h)
        },
        &cache,
    )
    .expect("compile_forward conv1d+relu");

    assert_eq!(compiled.num_inputs(), 1);
    assert!(
        compiled.num_dispatches() >= 1,
        "should have at least 1 dispatch"
    );

    // Move input to GPU, execute, and compare.
    let x_gpu = x.to_device(&gpu()).unwrap();
    let result = compiled.execute_dyn(&cache, &[&x_gpu]).unwrap();
    let gpu_vals = result
        .to_device(&cpu())
        .unwrap()
        .to_flat_vec::<f32>()
        .unwrap();
    assert_close("conv1d_relu_trace", &gpu_vals, &ref_vals, 1e-4);
}

/// Create a random Linear(in_f, out_f) with seeded weights and bias.
fn rand_linear(w_seed: u64, b_seed: u64, in_f: usize, out_f: usize) -> Linear {
    let w = DynTensor::new(
        &super::test_utils::rand_f32_vec(w_seed, out_f * in_f, -0.5, 0.5),
        &[out_f, in_f],
        &cpu(),
    )
    .unwrap();
    let b = DynTensor::new(
        &super::test_utils::rand_f32_vec(b_seed, out_f, -0.1, 0.1),
        &[out_f],
        &cpu(),
    )
    .unwrap();
    Linear::new(w, Some(b)).unwrap()
}

// -- Test 2: Linear + LayerNorm + Linear via compile_forward ------------------

/// Linear(8, 16) → LayerNorm(16) → Linear(16, 4) via compile_forward.
///
/// Tests that LayerNorm's weight (gamma) and bias (beta) are correctly
/// captured during tracing and uploaded to GPU for execution.
#[test]
#[allow(deprecated)]
fn test_trace_compile_linear_layernorm_linear() {
    super::test_utils::gpu_init();
    let cache = super::test_utils::metal_setup();

    let (in_f, hidden, out_f) = (8, 16, 4);
    let linear1 = rand_linear(0xA101, 0xA102, in_f, hidden);

    let ln_w = DynTensor::new(
        &super::test_utils::rand_f32_vec(0xA103, hidden, 0.5, 1.5),
        &[hidden],
        &cpu(),
    )
    .unwrap();
    let ln_b = DynTensor::new(
        &super::test_utils::rand_f32_vec(0xA104, hidden, -0.1, 0.1),
        &[hidden],
        &cpu(),
    )
    .unwrap();
    let layer_norm = LayerNorm::new(ln_w, ln_b, 1e-5).unwrap();

    let linear2 = rand_linear(0xA105, 0xA106, hidden, out_f);

    // Input: [2, 8] — batch=2, features=8
    let batch = 2;
    let input_data = super::test_utils::rand_f32_vec(0xA107, batch * in_f, -1.0, 1.0);
    let x = DynTensor::new(&input_data, &[batch, in_f], &cpu()).unwrap();

    // DynTensor eager reference.
    let h1 = linear1.forward(&x).unwrap();
    let h2 = layer_norm.forward(&h1).unwrap();
    let ref_out = linear2.forward(&h2).unwrap();
    let ref_vals = ref_out.to_flat_vec::<f32>().unwrap();

    // compile_forward: trace real Linear + LayerNorm + Linear.
    let compiled = CompiledModel::compile_forward(
        &[&x],
        |traced| {
            let h1 = linear1.forward(&traced[0])?;
            let h2 = layer_norm.forward(&h1)?;
            linear2.forward(&h2)
        },
        &cache,
    )
    .expect("compile_forward linear+ln+linear");

    assert_eq!(compiled.num_inputs(), 1);
    assert_eq!(compiled.output_shape(), &[batch, out_f]);

    let x_gpu = x.to_device(&gpu()).unwrap();
    let result = compiled.execute_dyn(&cache, &[&x_gpu]).unwrap();
    let gpu_vals = result
        .to_device(&cpu())
        .unwrap()
        .to_flat_vec::<f32>()
        .unwrap();
    assert_close("ln_chain_trace", &gpu_vals, &ref_vals, 1e-3);
}

// -- Test 3: Conv1d + LayerNorm → mini encoder block --------------------------

/// Conv1d(2, 8, 3, pad=1) → ReLU → Mean(dim=2) → LayerNorm(8)
///
/// Tests a mini encoder-like pattern: conv feature extraction → temporal
/// pooling → normalization. Exercises Conv1d, activation, reduction, and
/// LayerNorm all captured through trace_graph.
#[test]
#[allow(deprecated)]
fn test_trace_compile_mini_encoder() {
    super::test_utils::gpu_init();
    let cache = super::test_utils::metal_setup();

    let (in_ch, out_ch, ks, in_len) = (2, 8, 3, 16);
    let config = Conv1dConfig::default().with_padding(1);
    let w = DynTensor::new(
        &super::test_utils::rand_f32_vec(0xBE01, out_ch * in_ch * ks, -0.3, 0.3),
        &[out_ch, in_ch, ks],
        &cpu(),
    )
    .unwrap();
    let b = DynTensor::new(
        &super::test_utils::rand_f32_vec(0xBE02, out_ch, -0.1, 0.1),
        &[out_ch],
        &cpu(),
    )
    .unwrap();
    let conv = Conv1d::new(w, Some(b), config).unwrap();
    let relu = Activation::Relu;

    let ln_w = DynTensor::new(
        &super::test_utils::rand_f32_vec(0xBE03, out_ch, 0.5, 1.5),
        &[out_ch],
        &cpu(),
    )
    .unwrap();
    let ln_b = DynTensor::new(
        &super::test_utils::rand_f32_vec(0xBE04, out_ch, -0.1, 0.1),
        &[out_ch],
        &cpu(),
    )
    .unwrap();
    let layer_norm = LayerNorm::new(ln_w, ln_b, 1e-5).unwrap();

    let batch = 1;
    let input_data = super::test_utils::rand_f32_vec(0xBE05, batch * in_ch * in_len, -1.0, 1.0);
    let x = DynTensor::new(&input_data, &[batch, in_ch, in_len], &cpu()).unwrap();

    // DynTensor eager reference:
    //   Conv1d [1,2,16] -> [1,8,16], ReLU, Mean(dim=2) -> [1,8], LayerNorm
    let h = relu.forward(&conv.forward(&x).unwrap()).unwrap();
    let pooled = h.mean(2).unwrap(); // [1, 8]
    let ref_out = layer_norm.forward(&pooled).unwrap();
    let ref_vals = ref_out.to_flat_vec::<f32>().unwrap();

    let compiled = CompiledModel::compile_forward(
        &[&x],
        |traced| {
            let h = relu.forward(&conv.forward(&traced[0])?)?;
            let pooled = h.mean(2)?;
            layer_norm.forward(&pooled)
        },
        &cache,
    )
    .expect("compile_forward mini encoder");

    assert_eq!(compiled.num_inputs(), 1);
    assert_eq!(compiled.output_shape(), &[batch, out_ch]);

    let x_gpu = x.to_device(&gpu()).unwrap();
    let result = compiled.execute_dyn(&cache, &[&x_gpu]).unwrap();
    let gpu_vals = result
        .to_device(&cpu())
        .unwrap()
        .to_flat_vec::<f32>()
        .unwrap();
    assert_close("mini_encoder_trace", &gpu_vals, &ref_vals, 1e-3);
}

// -- Test 4: Softmax via compile_forward --------------------------------------

/// Linear(4, 6) → Softmax(dim=-1) via compile_forward.
///
/// Verifies softmax dispatch through the traced model path (not manual graph).
#[test]
#[allow(deprecated)]
fn test_trace_compile_softmax() {
    super::test_utils::gpu_init();
    let cache = super::test_utils::metal_setup();

    let (in_f, out_f, batch) = (4, 6, 3);
    let w = DynTensor::new(
        &super::test_utils::rand_f32_vec(0xDA01, out_f * in_f, -0.5, 0.5),
        &[out_f, in_f],
        &cpu(),
    )
    .unwrap();
    let b = DynTensor::new(
        &super::test_utils::rand_f32_vec(0xDA02, out_f, -0.1, 0.1),
        &[out_f],
        &cpu(),
    )
    .unwrap();
    let linear = Linear::new(w, Some(b)).unwrap();

    let input_data = super::test_utils::rand_f32_vec(0xDA03, batch * in_f, -1.0, 1.0);
    let x = DynTensor::new(&input_data, &[batch, in_f], &cpu()).unwrap();

    // DynTensor eager reference.
    let logits = linear.forward(&x).unwrap();
    let ref_out = logits.softmax(1).unwrap();
    let ref_vals = ref_out.to_flat_vec::<f32>().unwrap();

    let compiled = CompiledModel::compile_forward(
        &[&x],
        |traced| {
            let logits = linear.forward(&traced[0])?;
            logits.softmax(1)
        },
        &cache,
    )
    .expect("compile_forward softmax");

    assert_eq!(compiled.num_inputs(), 1);

    let x_gpu = x.to_device(&gpu()).unwrap();
    let result = compiled.execute_dyn(&cache, &[&x_gpu]).unwrap();
    let gpu_vals = result
        .to_device(&cpu())
        .unwrap()
        .to_flat_vec::<f32>()
        .unwrap();
    assert_close("softmax_trace", &gpu_vals, &ref_vals, 1e-5);

    // Verify softmax rows sum to 1.
    for row in 0..batch {
        let sum: f32 = gpu_vals[row * out_f..(row + 1) * out_f].iter().sum();
        assert!((sum - 1.0).abs() < 1e-5, "softmax row {row} sum={sum}");
    }
}

// -- Test 5: QKV parallel projections (pass 10 batching) ----------------------

/// Three parallel Linear(768, 256) projections from the same input (Q/K/V pattern).
///
/// Verifies that peephole pass 12 (BatchedLinearProjection) triggers and
/// produces correct output compared to eager DynTensor execution.
/// The batched path does: single matmul [768, 768] → narrow Q, K, V.
///
/// Part of #3269.
#[test]
#[allow(deprecated)]
fn test_trace_compile_qkv_batched_projection() {
    super::test_utils::gpu_init();
    let cache = super::test_utils::metal_setup();

    let (batch, seq, hidden) = (2, 4, 64);
    let proj_dim = 32;

    // Three Q/K/V projections sharing the same in_features.
    let q_proj = rand_linear(0xE501, 0xE502, hidden, proj_dim);
    let k_proj = rand_linear(0xE503, 0xE504, hidden, proj_dim);
    let v_proj = rand_linear(0xE505, 0xE506, hidden, proj_dim);

    // Input: [batch, seq, hidden].
    let input_data = super::test_utils::rand_f32_vec(0xE507, batch * seq * hidden, -1.0, 1.0);
    let x = DynTensor::new(&input_data, &[batch, seq, hidden], &cpu()).unwrap();

    // DynTensor eager reference: Q + K + V element-wise sum.
    let q_ref = q_proj.forward(&x).unwrap();
    let k_ref = k_proj.forward(&x).unwrap();
    let v_ref = v_proj.forward(&x).unwrap();
    let ref_out = q_ref.add(&k_ref).unwrap().add(&v_ref).unwrap();
    let ref_vals = ref_out.to_flat_vec::<f32>().unwrap();

    // compile_forward: trace 3 Linears from the same input, then sum.
    let compiled = CompiledModel::compile_forward(
        &[&x],
        |traced| {
            let q = q_proj.forward(&traced[0])?;
            let k = k_proj.forward(&traced[0])?;
            let v = v_proj.forward(&traced[0])?;
            q.add(&k)?.add(&v)
        },
        &cache,
    )
    .expect("compile_forward qkv");

    assert_eq!(compiled.num_inputs(), 1);
    assert_eq!(compiled.output_shape(), &[batch, seq, proj_dim]);

    // Verify pass 10 triggered: should have BatchedLinearProjection in dispatch breakdown.
    let (_, native_ops) = compiled.dispatch_breakdown();
    let batched_count = native_ops
        .iter()
        .find(|(name, _)| name == "BatchedLinearProjection")
        .map(|(_, c)| *c)
        .unwrap_or(0);
    let proj_slice_count = native_ops
        .iter()
        .find(|(name, _)| name == "ProjectionSlice")
        .map(|(_, c)| *c)
        .unwrap_or(0);
    assert!(
        batched_count > 0,
        "Pass 10 should create BatchedLinearProjection, got native_ops: {native_ops:?}"
    );
    assert_eq!(
        proj_slice_count, 2,
        "3 Linears → 1 batched + 2 ProjectionSlice, got {proj_slice_count}"
    );

    // Execute on GPU and compare.
    let x_gpu = x.to_device(&gpu()).unwrap();
    let result = compiled.execute_dyn(&cache, &[&x_gpu]).unwrap();
    let gpu_vals = result
        .to_device(&cpu())
        .unwrap()
        .to_flat_vec::<f32>()
        .unwrap();
    assert_close("qkv_batched_trace", &gpu_vals, &ref_vals, 1e-4);
}

// -- Test 6: QKV batching with autocast F16 -----------------------------------

/// QKV batched projection with autocast: F16 compute for matmul, F32 for add.
///
/// Verifies that `BatchedLinearProjection` and `ProjectionSlice` steps
/// execute correctly under autocast mode (F16 weights, F32 accumulation).
/// The edge_map correctly resolves source_step NarrowView dependencies
/// through the autocast boundary cast insertion.
///
/// Part of #3272.
#[test]
fn test_trace_compile_qkv_batched_projection_autocast() {
    use nn_core::mixed_precision::MixedPrecisionPolicy;

    super::test_utils::gpu_init();
    let cache = super::test_utils::metal_setup();

    let (batch, seq, hidden) = (2, 4, 64);
    let proj_dim = 32;

    let q_proj = rand_linear(0xF601, 0xF602, hidden, proj_dim);
    let k_proj = rand_linear(0xF603, 0xF604, hidden, proj_dim);
    let v_proj = rand_linear(0xF605, 0xF606, hidden, proj_dim);

    let input_data = super::test_utils::rand_f32_vec(0xF607, batch * seq * hidden, -1.0, 1.0);
    let x = DynTensor::new(&input_data, &[batch, seq, hidden], &cpu()).unwrap();

    // Trace the graph: 3 Linears from same input → Q + K + V sum.
    let shape = vec![batch, seq, hidden];
    let (_output, graph) = trace::trace_graph(|| {
        let mut traced_x = x.clone();
        if let Some(id) = trace::record_input(&shape, DType::F32) {
            traced_x.set_trace_id(id);
        }
        let q = q_proj.forward(&traced_x)?;
        let k = k_proj.forward(&traced_x)?;
        let v = v_proj.forward(&traced_x)?;
        q.add(&k)?.add(&v)
    })
    .expect("trace qkv");

    // F32 baseline.
    let f32_model = CompiledModel::builder(&graph, &cache)
        .build()
        .expect("compile f32 qkv");
    let x_gpu = x.to_device(&gpu()).unwrap();
    let f32_out = f32_model.execute_dyn(&cache, &[&x_gpu]).unwrap();
    let f32_vals = f32_out
        .to_device(&cpu())
        .unwrap()
        .to_flat_vec::<f32>()
        .unwrap();

    // Autocast: matmul runs F16, add runs F32.
    let policy = MixedPrecisionPolicy::apple_silicon_default();
    let ac_model = CompiledModel::builder(&graph, &cache)
        .autocast(policy)
        .build()
        .expect("compile autocast qkv");

    assert!(ac_model.is_autocast(), "model should be autocast");
    assert!(
        ac_model.num_autocast_f16_steps() > 0,
        "BatchedLinearProjection should have F16 steps"
    );

    // Verify pass 10 triggered under autocast.
    let (_, native_ops) = ac_model.dispatch_breakdown();
    let batched_count = native_ops
        .iter()
        .find(|(name, _)| name == "BatchedLinearProjection")
        .map(|(_, c)| *c)
        .unwrap_or(0);
    assert!(
        batched_count > 0,
        "Pass 10 should trigger under autocast, got native_ops: {native_ops:?}"
    );

    // Guard: BatchedLinearProjection must NOT appear in mixed_gemm_infos
    // (the classification-level mixed GEMM path). It uses is_compute_native_op
    // instead. DynTensor matmul at F16 routes to simd_gemm_f16 (F32
    // accumulators) internally. #3277, #3281.
    assert_eq!(
        ac_model.num_mixed_gemm_steps(),
        0,
        "BatchedLinearProjection must not be in mixed_gemm_infos (see #3277)"
    );

    let ac_out = ac_model.execute_dyn(&cache, &[&x_gpu]).unwrap();
    let ac_vals = ac_out
        .to_device(&cpu())
        .unwrap()
        .to_flat_vec::<f32>()
        .unwrap();

    // F16 matmul tolerance: wider than F32 due to half-precision rounding.
    assert_close("qkv_batched_autocast", &ac_vals, &f32_vals, 5e-2);
}

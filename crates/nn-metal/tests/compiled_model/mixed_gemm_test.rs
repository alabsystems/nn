// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! End-to-end tests for mixed-precision simdgroup GEMM dispatch.
//!
//! When autocast is active and a Linear/MatMul step has simdgroup-eligible
//! dimensions (all dims % 8, M*N >= 16384, K >= 128), the step bypasses
//! the IR dispatch and uses `simd_gemm_mixed` directly: F32 activations
//! × F16 weights → F32 output.
//!
//! Part of #3085 (per-op autocast Phase 2), #2981 (F16 pipeline).

use nn_core::dyn_tensor::trace::{TraceNode, TraceOp, WeightRef};
use nn_core::mixed_precision::MixedPrecisionPolicy;
use nn_core::DType;
use nn_metal::compiled_model::CompiledModel;

use crate::helpers::{assert_close, create_input_buffer, input_node, read_output_n};
use crate::test_utils;

// -- Build-time extraction tests ----------------------------------------------

/// Verify that simdgroup-eligible Linear with sufficient TGs produces a MixedGemmInfo.
#[test]
fn test_mixed_gemm_eligible_linear_detected() {
    let cache = test_utils::metal_setup();

    // TGs = ceil(384/32)*ceil(1024/32) = 12*32 = 384 >= MIN_TGS_FOR_MIXED_GEMM.
    let (m, k, n) = (384, 128, 1024);
    let weight_data = test_utils::rand_f32_vec(0xAC_D401, k * n, -0.1, 0.1);
    let weight = WeightRef::new(weight_data, vec![n, k]).expect("weight");

    let graph = nn_core::dyn_tensor::trace::ComputationGraph::from_nodes(vec![
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
        .expect("compile");
    assert!(compiled.is_autocast());
    assert_eq!(
        compiled.num_mixed_gemm_steps(),
        1,
        "simdgroup-eligible Linear should produce a mixed GEMM entry"
    );
}

/// Verify that sub-threshold Linear does NOT produce a MixedGemmInfo entry.
#[test]
fn test_mixed_gemm_ineligible_linear_skipped() {
    let cache = test_utils::metal_setup();

    // M=2, K=4, N=3: too small for simdgroup. No mixed GEMM.
    let (m, k, n) = (2, 4, 3);
    let weight_data = test_utils::rand_f32_vec(0xAC_D402, k * n, -0.1, 0.1);
    let weight = WeightRef::new(weight_data, vec![n, k]).expect("weight");

    let graph = nn_core::dyn_tensor::trace::ComputationGraph::from_nodes(vec![
        input_node(0, &[m, k]),
        TraceNode::new(
            1,
            "linear_small".into(),
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
        .expect("compile");
    assert_eq!(
        compiled.num_mixed_gemm_steps(),
        0,
        "sub-threshold Linear should NOT use mixed GEMM"
    );
}

/// Without autocast, no mixed GEMM entries regardless of shape.
#[test]
fn test_mixed_gemm_requires_autocast() {
    let cache = test_utils::metal_setup();

    let (m, k, n) = (128, 128, 128);
    let weight_data = test_utils::rand_f32_vec(0xAC_D403, k * n, -0.1, 0.1);
    let weight = WeightRef::new(weight_data, vec![n, k]).expect("weight");

    let graph = nn_core::dyn_tensor::trace::ComputationGraph::from_nodes(vec![
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

    // Plain F32 model — no autocast.
    let compiled = CompiledModel::builder(&graph, &cache)
        .build()
        .expect("compile");
    assert_eq!(
        compiled.num_mixed_gemm_steps(),
        0,
        "F32 model should have no mixed GEMM entries"
    );
}

// -- Diagnostic test ----------------------------------------------------------

/// Identity weight at below-threshold size: routes through F32 dispatch.
///
/// 128×128 identity gives TGs = 4*4 = 16, below MIN_TGS_FOR_MIXED_GEMM (384).
/// Output equals input exactly since F32 dispatch is used.
#[test]
fn test_mixed_gemm_identity_weight() {
    test_utils::gpu_init();
    let cache = test_utils::metal_setup();

    let dim = 128;
    let mut weight_data = vec![0.0_f32; dim * dim];
    for i in 0..dim {
        weight_data[i * dim + i] = 1.0;
    }
    let weight = WeightRef::new(weight_data, vec![dim, dim]).expect("weight");

    let graph = nn_core::dyn_tensor::trace::ComputationGraph::from_nodes(vec![
        input_node(0, &[dim, dim]),
        TraceNode::new(
            1,
            "linear_id".into(),
            TraceOp::Linear { weight, bias: None },
            vec![0],
            vec![dim, dim],
            DType::F32,
        ),
    ]);

    let input_data: Vec<f32> = (0..dim * dim)
        .map(|i| (i as f32 / (dim * dim) as f32) * 2.0 - 1.0)
        .collect();
    let buf = create_input_buffer(&cache, &input_data);

    let policy = MixedPrecisionPolicy::apple_silicon_default();
    let ac_model = CompiledModel::builder(&graph, &cache)
        .autocast(policy)
        .build()
        .expect("compile autocast");
    assert_eq!(ac_model.num_mixed_gemm_steps(), 0, "below TG threshold");

    let ac_out = ac_model.execute(&cache, &[&buf]).expect("autocast exec");
    let ac_result = read_output_n(&ac_out, dim * dim);

    assert_close("mixed_gemm_identity", &ac_result, &input_data, 1e-3);
}

// -- Execution correctness tests ----------------------------------------------

/// E2E: mixed GEMM Linear matches F32 baseline (TGs = 384, above threshold).
#[test]
fn test_mixed_gemm_linear_matches_f32() {
    test_utils::gpu_init();
    let cache = test_utils::metal_setup();

    let (m, k, n) = (384, 128, 1024);
    let weight_data = test_utils::rand_f32_vec(0xAC_D410, k * n, -0.1, 0.1);
    let weight = WeightRef::new(weight_data, vec![n, k]).expect("weight");

    let graph = nn_core::dyn_tensor::trace::ComputationGraph::from_nodes(vec![
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

    let input_data = test_utils::rand_f32_vec(0xAC_D411, m * k, -1.0, 1.0);
    let buf = create_input_buffer(&cache, &input_data);

    // F32 baseline.
    let f32_model = CompiledModel::builder(&graph, &cache)
        .build()
        .expect("compile f32");
    let f32_out = f32_model.execute(&cache, &[&buf]).expect("f32 exec");
    let f32_result = read_output_n(&f32_out, m * n);

    // Autocast with mixed GEMM.
    let policy = MixedPrecisionPolicy::apple_silicon_default();
    let ac_model = CompiledModel::builder(&graph, &cache)
        .autocast(policy)
        .build()
        .expect("compile autocast");
    assert_eq!(ac_model.num_mixed_gemm_steps(), 1, "should use mixed GEMM");

    let ac_out = ac_model.execute(&cache, &[&buf]).expect("autocast exec");
    let ac_result = read_output_n(&ac_out, m * n);

    // F16 weight quantization on values ~[-0.1, 0.1] has very small error.
    // K=128 accumulation can amplify slightly; 1e-2 is conservative.
    assert_close("mixed_gemm_linear", &ac_result, &f32_result, 1e-2);
}

/// E2E: mixed GEMM with bias term.
#[test]
fn test_mixed_gemm_linear_with_bias_matches_f32() {
    test_utils::gpu_init();
    let cache = test_utils::metal_setup();

    let (m, k, n) = (384, 128, 1024);
    let weight_data = test_utils::rand_f32_vec(0xAC_D420, k * n, -0.1, 0.1);
    let bias_data = test_utils::rand_f32_vec(0xAC_D421, n, -0.5, 0.5);
    let weight = WeightRef::new(weight_data, vec![n, k]).expect("weight");
    let bias = WeightRef::new(bias_data, vec![n]).expect("bias");

    let graph = nn_core::dyn_tensor::trace::ComputationGraph::from_nodes(vec![
        input_node(0, &[m, k]),
        TraceNode::new(
            1,
            "linear_bias".into(),
            TraceOp::Linear {
                weight,
                bias: Some(bias),
            },
            vec![0],
            vec![m, n],
            DType::F32,
        ),
    ]);

    let input_data = test_utils::rand_f32_vec(0xAC_D422, m * k, -1.0, 1.0);
    let buf = create_input_buffer(&cache, &input_data);

    let f32_model = CompiledModel::builder(&graph, &cache)
        .build()
        .expect("compile f32");
    let f32_out = f32_model.execute(&cache, &[&buf]).expect("f32 exec");
    let f32_result = read_output_n(&f32_out, m * n);

    let policy = MixedPrecisionPolicy::apple_silicon_default();
    let ac_model = CompiledModel::builder(&graph, &cache)
        .autocast(policy)
        .build()
        .expect("compile autocast");
    assert_eq!(ac_model.num_mixed_gemm_steps(), 1);

    let ac_out = ac_model.execute(&cache, &[&buf]).expect("autocast exec");
    let ac_result = read_output_n(&ac_out, m * n);

    assert_close("mixed_gemm_linear_bias", &ac_result, &f32_result, 1e-2);
}

/// E2E: chain of Linear → ReLU → Linear with autocast.
///
/// The first Linear+ReLU is fused into a LinearActivation NativeOp by the
/// peephole optimizer, so only the second standalone Linear remains as a
/// Dispatch step eligible for mixed GEMM. Tests that mixed GEMM output is
/// correctly consumed by the fused NativeOp and vice versa.
#[test]
fn test_mixed_gemm_chain_linear_relu_linear() {
    test_utils::gpu_init();
    let cache = test_utils::metal_setup();

    // Both linears need TGs >= 384: L1 384x128→1024 (12*32=384), L2 384x1024→1024 (12*32=384).
    let (m, k1, k2, n) = (384, 128, 1024, 1024);
    let w1_data = test_utils::rand_f32_vec(0xAC_D430, k1 * k2, -0.05, 0.05);
    let w2_data = test_utils::rand_f32_vec(0xAC_D431, k2 * n, -0.05, 0.05);
    let w1 = WeightRef::new(w1_data, vec![k2, k1]).expect("w1");
    let w2 = WeightRef::new(w2_data, vec![n, k2]).expect("w2");

    let graph = nn_core::dyn_tensor::trace::ComputationGraph::from_nodes(vec![
        input_node(0, &[m, k1]),
        TraceNode::new(
            1,
            "linear_0".into(),
            TraceOp::Linear {
                weight: w1,
                bias: None,
            },
            vec![0],
            vec![m, k2],
            DType::F32,
        ),
        TraceNode::new(
            2,
            "relu_0".into(),
            TraceOp::Relu,
            vec![1],
            vec![m, k2],
            DType::F32,
        ),
        TraceNode::new(
            3,
            "linear_1".into(),
            TraceOp::Linear {
                weight: w2,
                bias: None,
            },
            vec![2],
            vec![m, n],
            DType::F32,
        ),
    ]);

    let input_data = test_utils::rand_f32_vec(0xAC_D432, m * k1, -1.0, 1.0);
    let buf = create_input_buffer(&cache, &input_data);

    let f32_model = CompiledModel::builder(&graph, &cache)
        .build()
        .expect("compile f32");
    let f32_out = f32_model.execute(&cache, &[&buf]).expect("f32 exec");
    let f32_result = read_output_n(&f32_out, m * n);

    let policy = MixedPrecisionPolicy::apple_silicon_default();
    let ac_model = CompiledModel::builder(&graph, &cache)
        .autocast(policy)
        .build()
        .expect("compile autocast");
    // First Linear+ReLU fused into LinearActivation NativeOp (mixed GEMM
    // with activation epilogue). Second standalone Linear is a Dispatch step
    // with mixed GEMM. Both qualify for simdgroup mixed GEMM. (#2981)
    assert_eq!(
        ac_model.num_mixed_gemm_steps(),
        2,
        "both LinearActivation NativeOp and standalone Dispatch should use mixed GEMM"
    );

    let ac_out = ac_model.execute(&cache, &[&buf]).expect("autocast exec");
    let ac_result = read_output_n(&ac_out, m * n);

    // F16 weight quantization + NativeOp chain → moderate error accumulation.
    assert_close("mixed_gemm_chain", &ac_result, &f32_result, 0.02);
}

/// E2E: NativeOp LinearActivation (fused Linear+ReLU) uses mixed GEMM.
///
/// Verifies that when autocast is active, a simdgroup-eligible
/// LinearActivation NativeOp (created by peephole fusion of Linear+ReLU)
/// dispatches via the mixed GEMM kernel with F32 activations × F16 weights,
/// with the ReLU activation applied in the kernel epilogue.
///
/// Part of #2981.
#[test]
fn test_mixed_gemm_native_op_linear_activation() {
    test_utils::gpu_init();
    let cache = test_utils::metal_setup();

    // TGs = ceil(384/32)*ceil(1024/32) = 12*32 = 384 >= threshold.
    let (m, k, n) = (384, 128, 1024);
    let weight_data = test_utils::rand_f32_vec(0xAC_D440, k * n, -0.1, 0.1);
    let weight = WeightRef::new(weight_data, vec![n, k]).expect("weight");

    let graph = nn_core::dyn_tensor::trace::ComputationGraph::from_nodes(vec![
        input_node(0, &[m, k]),
        TraceNode::new(
            1,
            "linear_0".into(),
            TraceOp::Linear { weight, bias: None },
            vec![0],
            vec![m, n],
            DType::F32,
        ),
        TraceNode::new(
            2,
            "relu_0".into(),
            TraceOp::Relu,
            vec![1],
            vec![m, n],
            DType::F32,
        ),
    ]);

    let input_data = test_utils::rand_f32_vec(0xAC_D441, m * k, -1.0, 1.0);
    let buf = create_input_buffer(&cache, &input_data);

    // F32 baseline (peephole fuses Linear+ReLU into LinearActivation NativeOp).
    let f32_model = CompiledModel::builder(&graph, &cache)
        .build()
        .expect("compile f32");
    let f32_out = f32_model.execute(&cache, &[&buf]).expect("f32 exec");
    let f32_result = read_output_n(&f32_out, m * n);

    // Autocast: LinearActivation should use mixed GEMM.
    let policy = MixedPrecisionPolicy::apple_silicon_default();
    let ac_model = CompiledModel::builder(&graph, &cache)
        .autocast(policy)
        .build()
        .expect("compile autocast");
    assert_eq!(
        ac_model.num_mixed_gemm_steps(),
        1,
        "LinearActivation NativeOp should use mixed GEMM"
    );

    let ac_out = ac_model.execute(&cache, &[&buf]).expect("autocast exec");
    let ac_result = read_output_n(&ac_out, m * n);

    // ReLU clamps negatives to 0; remaining values have F16 weight tolerance.
    assert_close(
        "mixed_gemm_native_linear_act",
        &ac_result,
        &f32_result,
        1e-2,
    );
}

/// E2E: NativeOp LinearActivation with bias uses mixed GEMM.
///
/// Part of #2981.
#[test]
fn test_mixed_gemm_native_op_linear_activation_with_bias() {
    test_utils::gpu_init();
    let cache = test_utils::metal_setup();

    let (m, k, n) = (384, 128, 1024);
    let weight_data = test_utils::rand_f32_vec(0xAC_D450, k * n, -0.1, 0.1);
    let bias_data = test_utils::rand_f32_vec(0xAC_D451, n, -0.5, 0.5);
    let weight = WeightRef::new(weight_data, vec![n, k]).expect("weight");
    let bias = WeightRef::new(bias_data, vec![n]).expect("bias");

    let graph = nn_core::dyn_tensor::trace::ComputationGraph::from_nodes(vec![
        input_node(0, &[m, k]),
        TraceNode::new(
            1,
            "linear_0".into(),
            TraceOp::Linear {
                weight,
                bias: Some(bias),
            },
            vec![0],
            vec![m, n],
            DType::F32,
        ),
        TraceNode::new(
            2,
            "relu_0".into(),
            TraceOp::Relu,
            vec![1],
            vec![m, n],
            DType::F32,
        ),
    ]);

    let input_data = test_utils::rand_f32_vec(0xAC_D452, m * k, -1.0, 1.0);
    let buf = create_input_buffer(&cache, &input_data);

    let f32_model = CompiledModel::builder(&graph, &cache)
        .build()
        .expect("compile f32");
    let f32_out = f32_model.execute(&cache, &[&buf]).expect("f32 exec");
    let f32_result = read_output_n(&f32_out, m * n);

    let policy = MixedPrecisionPolicy::apple_silicon_default();
    let ac_model = CompiledModel::builder(&graph, &cache)
        .autocast(policy)
        .build()
        .expect("compile autocast");
    assert_eq!(ac_model.num_mixed_gemm_steps(), 1);

    let ac_out = ac_model.execute(&cache, &[&buf]).expect("autocast exec");
    let ac_result = read_output_n(&ac_out, m * n);

    assert_close(
        "mixed_gemm_native_linear_act_bias",
        &ac_result,
        &f32_result,
        1e-2,
    );
}

/// E2E: LinearActivation with GeluErf activation (PlBert FFN pattern).
///
/// PlBert uses Linear + GELU (erf variant) in its feed-forward layers.
/// Verifies the GELU epilogue in the mixed GEMM kernel produces correct
/// output matching F32 baseline.
///
/// Part of #2981.
#[test]
fn test_mixed_gemm_native_op_linear_gelu_erf() {
    test_utils::gpu_init();
    let cache = test_utils::metal_setup();

    let (m, k, n) = (384, 128, 1024);
    let weight_data = test_utils::rand_f32_vec(0xAC_D460, k * n, -0.1, 0.1);
    let bias_data = test_utils::rand_f32_vec(0xAC_D461, n, -0.3, 0.3);
    let weight = WeightRef::new(weight_data, vec![n, k]).expect("weight");
    let bias = WeightRef::new(bias_data, vec![n]).expect("bias");

    let graph = nn_core::dyn_tensor::trace::ComputationGraph::from_nodes(vec![
        input_node(0, &[m, k]),
        TraceNode::new(
            1,
            "linear_0".into(),
            TraceOp::Linear {
                weight,
                bias: Some(bias),
            },
            vec![0],
            vec![m, n],
            DType::F32,
        ),
        TraceNode::new(
            2,
            "gelu_0".into(),
            TraceOp::GeluErf,
            vec![1],
            vec![m, n],
            DType::F32,
        ),
    ]);

    let input_data = test_utils::rand_f32_vec(0xAC_D462, m * k, -1.0, 1.0);
    let buf = create_input_buffer(&cache, &input_data);

    let f32_model = CompiledModel::builder(&graph, &cache)
        .build()
        .expect("compile f32");
    let f32_out = f32_model.execute(&cache, &[&buf]).expect("f32 exec");
    let f32_result = read_output_n(&f32_out, m * n);

    let policy = MixedPrecisionPolicy::apple_silicon_default();
    let ac_model = CompiledModel::builder(&graph, &cache)
        .autocast(policy)
        .build()
        .expect("compile autocast");
    assert_eq!(
        ac_model.num_mixed_gemm_steps(),
        1,
        "Linear+GeluErf should fuse to LinearActivation with mixed GEMM"
    );

    let ac_out = ac_model.execute(&cache, &[&buf]).expect("autocast exec");
    let ac_result = read_output_n(&ac_out, m * n);

    // GELU is smooth, so F16 weight quantization error is modest.
    assert_close("mixed_gemm_linear_gelu_erf", &ac_result, &f32_result, 1e-2);
}

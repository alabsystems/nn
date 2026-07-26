// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! End-to-end compiled model tests for ops beyond Linear+elementwise.
//!
//! Exercises the full pipeline: build graph → compile → GPU execute → verify
//! against CPU reference for Conv1d, LayerNorm, Softmax, MatMul, Cat,
//! Embedding, ReduceSum, ReduceMean, Transpose, Reshape, Narrow, and RmsNorm.
//!
//! Part of #2214.

use nn_core::dyn_tensor::trace::{ComputationGraph, TraceNode, TraceOp, WeightRef};
use nn_core::DType;

use super::helpers::{assert_close, compile_and_run, create_input_buffer, input_node};

fn weight(data: Vec<f32>, shape: Vec<usize>) -> WeightRef {
    WeightRef::new(data, shape).expect("weight")
}

/// CPU reference for LayerNorm: normalize over last dim, then scale+shift.
fn cpu_layernorm(input: &[f32], gamma: &[f32], beta: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    let eps = 1e-5_f32;
    let mut output = vec![0.0_f32; rows * cols];
    for b in 0..rows {
        let row = &input[b * cols..(b + 1) * cols];
        let mean = row.iter().copied().sum::<f32>() / cols as f32;
        let var = row.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / cols as f32;
        let inv_std = 1.0 / (var + eps).sqrt();
        for h in 0..cols {
            output[b * cols + h] = (row[h] - mean) * inv_std * gamma[h] + beta[h];
        }
    }
    output
}

/// CPU reference for softmax over last dimension.
fn cpu_softmax(input: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    let mut output = vec![0.0_f32; rows * cols];
    for b in 0..rows {
        let row = &input[b * cols..(b + 1) * cols];
        let max_val = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let exp_sum: f32 = row.iter().map(|v| (v - max_val).exp()).sum();
        for d in 0..cols {
            output[b * cols + d] = (row[d] - max_val).exp() / exp_sum;
        }
    }
    output
}

// -- Test 1: Conv1d + ReLU ----------------------------------------------------

/// Conv1d(1, 16, 3) + ReLU: convolution compiled end-to-end on GPU.
#[test]
fn test_compiled_conv1d_relu() {
    super::test_utils::gpu_init();
    let cache = super::test_utils::metal_setup();

    let (in_ch, out_ch, ks, in_len, pad) = (1, 16, 3, 8, 1);
    let out_len = (in_len + 2 * pad - ks) + 1;

    let w_data = super::test_utils::rand_f32_vec(0xC01D_0001, out_ch * in_ch * ks, -0.5, 0.5);
    let b_data = super::test_utils::rand_f32_vec(0xC01D_0002, out_ch, -0.1, 0.1);
    let input_data = super::test_utils::rand_f32_vec(0xC01D_0003, in_ch * in_len, -1.0, 1.0);

    let graph = ComputationGraph::from_nodes(vec![
        input_node(0, &[in_ch, in_len]),
        TraceNode::new(
            1,
            "conv1d_0".into(),
            TraceOp::Conv1d {
                weight: weight(w_data.clone(), vec![out_ch, in_ch, ks]),
                bias: Some(weight(b_data.clone(), vec![out_ch])),
                padding: pad,
                stride: 1,
                dilation: 1,
                groups: 1,
            },
            vec![0],
            vec![out_ch, out_len],
            DType::F32,
        ),
        TraceNode::new(
            2,
            "relu_0".into(),
            TraceOp::Relu,
            vec![1],
            vec![out_ch, out_len],
            DType::F32,
        ),
    ]);

    let result = compile_and_run(
        &cache,
        graph,
        &[&create_input_buffer(&cache, &input_data)],
        out_ch * out_len,
    );

    let conv_out = super::test_utils::conv1d_ref(
        &input_data,
        &w_data,
        Some(&b_data),
        in_ch,
        out_ch,
        ks,
        in_len,
        1,
        pad,
    );
    let expected: Vec<f32> = conv_out.iter().map(|v| v.max(0.0)).collect();
    assert_close("conv1d_relu", &result, &expected, 1e-4);
}

// -- Test 1b: Conv1d autocast parity (F16 vs F32) ----------------------------

/// Conv1d autocast: F16 result matches F32 baseline within tolerance.
/// Validates that `is_non_gemm_compute_dispatch()` correctly classifies
/// Conv1d for F16, and the MSL kernel runs with half* buffers + F32
/// accumulators. Part of #2981.
#[test]
fn test_autocast_conv1d_parity() {
    use nn_core::mixed_precision::MixedPrecisionPolicy;
    use nn_metal::compiled_model::CompiledModel;

    super::test_utils::gpu_init();
    let cache = super::test_utils::metal_setup();

    let (in_ch, out_ch, ks, in_len, pad) = (4, 8, 3, 16, 1);
    let out_len = (in_len + 2 * pad - ks) + 1;

    let w_data = super::test_utils::rand_f32_vec(0xF16C_0001, out_ch * in_ch * ks, -0.5, 0.5);
    let b_data = super::test_utils::rand_f32_vec(0xF16C_0002, out_ch, -0.1, 0.1);
    let input_data = super::test_utils::rand_f32_vec(0xF16C_0003, in_ch * in_len, -1.0, 1.0);

    let graph = ComputationGraph::from_nodes(vec![
        input_node(0, &[in_ch, in_len]),
        TraceNode::new(
            1,
            "conv1d_0".into(),
            TraceOp::Conv1d {
                weight: weight(w_data, vec![out_ch, in_ch, ks]),
                bias: Some(weight(b_data, vec![out_ch])),
                padding: pad,
                stride: 1,
                dilation: 1,
                groups: 1,
            },
            vec![0],
            vec![out_ch, out_len],
            DType::F32,
        ),
    ]);

    let input_buf = create_input_buffer(&cache, &input_data);

    // F32 baseline.
    let f32_model = CompiledModel::builder(&graph, &cache)
        .build()
        .expect("compile f32");
    let f32_buf = f32_model.execute(&cache, &[&input_buf]).expect("f32 exec");
    let f32_result = super::helpers::read_output_n(&f32_buf, out_ch * out_len);

    // Autocast (F16 for Conv1d).
    let policy = MixedPrecisionPolicy::apple_silicon_default();
    let ac_model = CompiledModel::builder(&graph, &cache)
        .autocast(policy)
        .build()
        .expect("compile autocast");
    assert!(ac_model.is_autocast(), "model should be autocast");
    assert!(
        ac_model.num_autocast_f16_steps() > 0,
        "Conv1d should be classified as F16 step"
    );
    let ac_buf = ac_model
        .execute(&cache, &[&input_buf])
        .expect("autocast exec");
    let ac_result = super::helpers::read_output_n(&ac_buf, out_ch * out_len);

    // F16 tolerance: Conv1d MSL uses float accumulators, so error is small.
    assert_close("autocast_conv1d", &ac_result, &f32_result, 1e-2);
}

// -- Test 2: Linear + LayerNorm + Linear --------------------------------------

/// Linear(4, 8) -> LayerNorm(8) -> Linear(8, 3): normalization in compiled pipeline.
#[test]
fn test_compiled_linear_layernorm_linear() {
    super::test_utils::gpu_init();
    let cache = super::test_utils::metal_setup();

    let (in_f, h, out_f, batch) = (4, 8, 3, 2);
    let w1 = super::test_utils::rand_f32_vec(0x1A_0001, h * in_f, -0.5, 0.5);
    let b1 = super::test_utils::rand_f32_vec(0x1A_0002, h, -0.1, 0.1);
    let gamma = super::test_utils::rand_f32_vec(0x1A_0003, h, 0.5, 1.5);
    let beta = super::test_utils::rand_f32_vec(0x1A_0004, h, -0.1, 0.1);
    let w2 = super::test_utils::rand_f32_vec(0x1A_0005, out_f * h, -0.5, 0.5);
    let b2 = super::test_utils::rand_f32_vec(0x1A_0006, out_f, -0.1, 0.1);
    let input_data = super::test_utils::rand_f32_vec(0x1A_0007, batch * in_f, -1.0, 1.0);

    let graph = ComputationGraph::from_nodes(vec![
        input_node(0, &[batch, in_f]),
        TraceNode::new(
            1,
            "linear_0".into(),
            TraceOp::Linear {
                weight: weight(w1.clone(), vec![h, in_f]),
                bias: Some(weight(b1.clone(), vec![h])),
            },
            vec![0],
            vec![batch, h],
            DType::F32,
        ),
        TraceNode::new(
            2,
            "layernorm_0".into(),
            TraceOp::LayerNorm {
                eps: 1e-5,
                weight: weight(gamma.clone(), vec![h]),
                bias: weight(beta.clone(), vec![h]),
            },
            vec![1],
            vec![batch, h],
            DType::F32,
        ),
        TraceNode::new(
            3,
            "linear_1".into(),
            TraceOp::Linear {
                weight: weight(w2.clone(), vec![out_f, h]),
                bias: Some(weight(b2.clone(), vec![out_f])),
            },
            vec![2],
            vec![batch, out_f],
            DType::F32,
        ),
    ]);

    let result = compile_and_run(
        &cache,
        graph,
        &[&create_input_buffer(&cache, &input_data)],
        batch * out_f,
    );

    let h1 = super::test_utils::linear_ref(&input_data, &w1, Some(&b1), batch, in_f, h);
    let h2 = cpu_layernorm(&h1, &gamma, &beta, batch, h);
    let expected = super::test_utils::linear_ref(&h2, &w2, Some(&b2), batch, h, out_f);
    assert_close("ln_chain", &result, &expected, 1e-3);
}

// -- Test 3: Linear + Softmax ------------------------------------------------

/// Linear(4, 6) -> Softmax(dim=1): softmax in compiled pipeline.
#[test]
fn test_compiled_linear_softmax() {
    super::test_utils::gpu_init();
    let cache = super::test_utils::metal_setup();

    let (in_f, out_f, batch) = (4, 6, 3);
    let w = super::test_utils::rand_f32_vec(0x50F7_0001, out_f * in_f, -0.5, 0.5);
    let b = super::test_utils::rand_f32_vec(0x50F7_0002, out_f, -0.1, 0.1);
    let input_data = super::test_utils::rand_f32_vec(0x50F7_0003, batch * in_f, -1.0, 1.0);

    let graph = ComputationGraph::from_nodes(vec![
        input_node(0, &[batch, in_f]),
        TraceNode::new(
            1,
            "linear_0".into(),
            TraceOp::Linear {
                weight: weight(w.clone(), vec![out_f, in_f]),
                bias: Some(weight(b.clone(), vec![out_f])),
            },
            vec![0],
            vec![batch, out_f],
            DType::F32,
        ),
        TraceNode::new(
            2,
            "softmax_0".into(),
            TraceOp::Softmax { dim: 1 },
            vec![1],
            vec![batch, out_f],
            DType::F32,
        ),
    ]);

    let result = compile_and_run(
        &cache,
        graph,
        &[&create_input_buffer(&cache, &input_data)],
        batch * out_f,
    );

    let logits = super::test_utils::linear_ref(&input_data, &w, Some(&b), batch, in_f, out_f);
    let expected = cpu_softmax(&logits, batch, out_f);
    assert_close("softmax", &result, &expected, 1e-5);

    // Softmax outputs must sum to 1 per row.
    for row in 0..batch {
        let row_sum: f32 = result[row * out_f..(row + 1) * out_f].iter().sum();
        assert!(
            (row_sum - 1.0).abs() < 1e-5,
            "softmax row {row} sum={row_sum}"
        );
    }
}

// -- Test 4: MatMul (two variable inputs) ------------------------------------

/// MatMul: [2, 4] x [4, 3] -> [2, 3]. Two variable inputs, no weights.
#[test]
fn test_compiled_matmul_two_inputs() {
    super::test_utils::gpu_init();
    let cache = super::test_utils::metal_setup();

    let (m, k, n) = (2, 4, 3);
    let left = super::test_utils::rand_f32_vec(0xA7A7_0001, m * k, -1.0, 1.0);
    let right = super::test_utils::rand_f32_vec(0xA7A7_0002, k * n, -1.0, 1.0);

    let graph = ComputationGraph::from_nodes(vec![
        input_node(0, &[m, k]),
        input_node(1, &[k, n]),
        TraceNode::new(
            2,
            "matmul_0".into(),
            TraceOp::MatMul,
            vec![0, 1],
            vec![m, n],
            DType::F32,
        ),
    ]);

    let left_buf = create_input_buffer(&cache, &left);
    let right_buf = create_input_buffer(&cache, &right);
    let result = compile_and_run(&cache, graph, &[&left_buf, &right_buf], m * n);

    let expected = super::test_utils::matmul_ref(&left, &right, m, k, n, false, None);
    assert_close("matmul", &result, &expected, 1e-4);
}

// -- Test 5: Cat + Linear (structural) ----------------------------------------

/// Cat([1,4],[1,4], dim=1) -> [1,8] -> Linear(8, 3): two inputs concatenated.
#[test]
fn test_compiled_cat_linear() {
    super::test_utils::gpu_init();
    let cache = super::test_utils::metal_setup();

    let (seg, out_f) = (4, 3);
    let w = super::test_utils::rand_f32_vec(0xCA7_0001, out_f * (seg * 2), -0.5, 0.5);
    let b = super::test_utils::rand_f32_vec(0xCA7_0002, out_f, -0.1, 0.1);
    let a_data = super::test_utils::rand_f32_vec(0xCA7_0003, seg, -1.0, 1.0);
    let b_data = super::test_utils::rand_f32_vec(0xCA7_0004, seg, -1.0, 1.0);

    let graph = ComputationGraph::from_nodes(vec![
        input_node(0, &[1, seg]),
        input_node(1, &[1, seg]),
        TraceNode::new(
            2,
            "cat_0".into(),
            TraceOp::Cat {
                dim: 1,
                num_inputs: 2,
            },
            vec![0, 1],
            vec![1, seg * 2],
            DType::F32,
        ),
        TraceNode::new(
            3,
            "linear_0".into(),
            TraceOp::Linear {
                weight: weight(w.clone(), vec![out_f, seg * 2]),
                bias: Some(weight(b.clone(), vec![out_f])),
            },
            vec![2],
            vec![1, out_f],
            DType::F32,
        ),
    ]);

    let buf_a = create_input_buffer(&cache, &a_data);
    let buf_b = create_input_buffer(&cache, &b_data);
    let result = compile_and_run(&cache, graph, &[&buf_a, &buf_b], out_f);

    let catted: Vec<f32> = a_data.iter().chain(b_data.iter()).copied().collect();
    let expected = super::test_utils::linear_ref(&catted, &w, Some(&b), 1, seg * 2, out_f);
    assert_close("cat_linear", &result, &expected, 1e-4);
}

// -- Test 6: ReduceSum + ReduceMean -------------------------------------------

/// Linear(4, 8) -> ReduceSum(dim=1, keepdim=false) -> [2]: reduction in pipeline.
#[test]
fn test_compiled_reduce_sum() {
    super::test_utils::gpu_init();
    let cache = super::test_utils::metal_setup();

    let (batch, features) = (2, 8);
    let w = super::test_utils::rand_f32_vec(0xBED0_0001, features * 4, -0.5, 0.5);
    let b = super::test_utils::rand_f32_vec(0xBED0_0002, features, -0.1, 0.1);
    let input_data = super::test_utils::rand_f32_vec(0xBED0_0003, batch * 4, -1.0, 1.0);

    let graph = ComputationGraph::from_nodes(vec![
        input_node(0, &[batch, 4]),
        TraceNode::new(
            1,
            "linear_0".into(),
            TraceOp::Linear {
                weight: weight(w.clone(), vec![features, 4]),
                bias: Some(weight(b.clone(), vec![features])),
            },
            vec![0],
            vec![batch, features],
            DType::F32,
        ),
        TraceNode::new(
            2,
            "reduce_sum_0".into(),
            TraceOp::ReduceSum {
                dim: 1,
                keepdim: false,
            },
            vec![1],
            vec![batch],
            DType::F32,
        ),
    ]);

    let result = compile_and_run(
        &cache,
        graph,
        &[&create_input_buffer(&cache, &input_data)],
        batch,
    );

    let linear_out = super::test_utils::linear_ref(&input_data, &w, Some(&b), batch, 4, features);
    let mut expected = vec![0.0_f32; batch];
    for r in 0..batch {
        expected[r] = linear_out[r * features..(r + 1) * features].iter().sum();
    }
    assert_close("reduce_sum", &result, &expected, 1e-3);
}

/// Linear(4, 8) -> ReduceMean(dim=1, keepdim=true) -> [2, 1]: mean reduction.
#[test]
fn test_compiled_reduce_mean() {
    super::test_utils::gpu_init();
    let cache = super::test_utils::metal_setup();

    let (batch, features) = (3, 6);
    let w = super::test_utils::rand_f32_vec(0xBED1_0001, features * 4, -0.5, 0.5);
    let b = super::test_utils::rand_f32_vec(0xBED1_0002, features, -0.1, 0.1);
    let input_data = super::test_utils::rand_f32_vec(0xBED1_0003, batch * 4, -1.0, 1.0);

    let graph = ComputationGraph::from_nodes(vec![
        input_node(0, &[batch, 4]),
        TraceNode::new(
            1,
            "linear_0".into(),
            TraceOp::Linear {
                weight: weight(w.clone(), vec![features, 4]),
                bias: Some(weight(b.clone(), vec![features])),
            },
            vec![0],
            vec![batch, features],
            DType::F32,
        ),
        TraceNode::new(
            2,
            "reduce_mean_0".into(),
            TraceOp::ReduceMean {
                dim: 1,
                keepdim: true,
            },
            vec![1],
            vec![batch, 1],
            DType::F32,
        ),
    ]);

    let result = compile_and_run(
        &cache,
        graph,
        &[&create_input_buffer(&cache, &input_data)],
        batch,
    );

    let linear_out = super::test_utils::linear_ref(&input_data, &w, Some(&b), batch, 4, features);
    let mut expected = vec![0.0_f32; batch];
    for r in 0..batch {
        expected[r] = linear_out[r * features..(r + 1) * features]
            .iter()
            .sum::<f32>()
            / features as f32;
    }
    assert_close("reduce_mean", &result, &expected, 1e-4);
}

// -- Test 7: Transpose + Reshape ----------------------------------------------

/// [2, 3] -> Transpose(0, 1) -> [3, 2] -> Reshape([6]) -> [6]: shape ops.
#[test]
fn test_compiled_transpose_reshape() {
    super::test_utils::gpu_init();
    let cache = super::test_utils::metal_setup();

    let input_data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];

    let graph = ComputationGraph::from_nodes(vec![
        input_node(0, &[2, 3]),
        TraceNode::new(
            1,
            "transpose_0".into(),
            TraceOp::Transpose { dim0: 0, dim1: 1 },
            vec![0],
            vec![3, 2],
            DType::F32,
        ),
        TraceNode::new(
            2,
            "reshape_0".into(),
            TraceOp::Reshape {
                target_shape: vec![6],
            },
            vec![1],
            vec![6],
            DType::F32,
        ),
    ]);

    let result = compile_and_run(
        &cache,
        graph,
        &[&create_input_buffer(&cache, &input_data)],
        6,
    );

    // [[1,2,3],[4,5,6]] transposed = [[1,4],[2,5],[3,6]], flattened = [1,4,2,5,3,6]
    let expected: Vec<f32> = vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0];
    assert_close("transpose_reshape", &result, &expected, 1e-6);
}

// -- Test 8: Narrow -----------------------------------------------------------

/// [1, 8] -> Linear(8, 6) -> Narrow(dim=1, start=1, len=3) -> [1, 3]: slicing.
#[test]
fn test_compiled_narrow() {
    super::test_utils::gpu_init();
    let cache = super::test_utils::metal_setup();

    let (in_f, out_f) = (8, 6);
    let w = super::test_utils::rand_f32_vec(0x4A80_0001, out_f * in_f, -0.5, 0.5);
    let b = super::test_utils::rand_f32_vec(0x4A80_0002, out_f, -0.1, 0.1);
    let input_data = super::test_utils::rand_f32_vec(0x4A80_0003, in_f, -1.0, 1.0);

    let graph = ComputationGraph::from_nodes(vec![
        input_node(0, &[1, in_f]),
        TraceNode::new(
            1,
            "linear_0".into(),
            TraceOp::Linear {
                weight: weight(w.clone(), vec![out_f, in_f]),
                bias: Some(weight(b.clone(), vec![out_f])),
            },
            vec![0],
            vec![1, out_f],
            DType::F32,
        ),
        TraceNode::new(
            2,
            "narrow_0".into(),
            TraceOp::Narrow {
                dim: 1,
                start: 1,
                length: 3,
            },
            vec![1],
            vec![1, 3],
            DType::F32,
        ),
    ]);

    let result = compile_and_run(
        &cache,
        graph,
        &[&create_input_buffer(&cache, &input_data)],
        3,
    );

    let linear_out = super::test_utils::linear_ref(&input_data, &w, Some(&b), 1, in_f, out_f);
    let expected = linear_out[1..4].to_vec();
    assert_close("narrow", &result, &expected, 1e-4);
}

// -- Test 9: RmsNorm ----------------------------------------------------------

/// Linear(4, 8) -> RmsNorm(8) -> Linear(8, 3): RMS normalization in pipeline.
#[test]
fn test_compiled_rmsnorm() {
    super::test_utils::gpu_init();
    let cache = super::test_utils::metal_setup();

    let (in_f, h, out_f, batch) = (4, 8, 3, 2);
    let w1 = super::test_utils::rand_f32_vec(0xB450_0001, h * in_f, -0.5, 0.5);
    let b1 = super::test_utils::rand_f32_vec(0xB450_0002, h, -0.1, 0.1);
    let rms_w = super::test_utils::rand_f32_vec(0xB450_0003, h, 0.5, 1.5);
    let w2 = super::test_utils::rand_f32_vec(0xB450_0004, out_f * h, -0.5, 0.5);
    let b2 = super::test_utils::rand_f32_vec(0xB450_0005, out_f, -0.1, 0.1);
    let input_data = super::test_utils::rand_f32_vec(0xB450_0006, batch * in_f, -1.0, 1.0);

    let graph = ComputationGraph::from_nodes(vec![
        input_node(0, &[batch, in_f]),
        TraceNode::new(
            1,
            "linear_0".into(),
            TraceOp::Linear {
                weight: weight(w1.clone(), vec![h, in_f]),
                bias: Some(weight(b1.clone(), vec![h])),
            },
            vec![0],
            vec![batch, h],
            DType::F32,
        ),
        TraceNode::new(
            2,
            "rmsnorm_0".into(),
            TraceOp::RmsNorm {
                eps: 1e-5,
                weight: weight(rms_w.clone(), vec![h]),
            },
            vec![1],
            vec![batch, h],
            DType::F32,
        ),
        TraceNode::new(
            3,
            "linear_1".into(),
            TraceOp::Linear {
                weight: weight(w2.clone(), vec![out_f, h]),
                bias: Some(weight(b2.clone(), vec![out_f])),
            },
            vec![2],
            vec![batch, out_f],
            DType::F32,
        ),
    ]);

    let result = compile_and_run(
        &cache,
        graph,
        &[&create_input_buffer(&cache, &input_data)],
        batch * out_f,
    );

    // CPU reference: RmsNorm = x * weight / sqrt(mean(x^2) + eps)
    let h1 = super::test_utils::linear_ref(&input_data, &w1, Some(&b1), batch, in_f, h);
    let mut h2 = vec![0.0_f32; batch * h];
    let eps = 1e-5_f32;
    for row in 0..batch {
        let row_data = &h1[row * h..(row + 1) * h];
        let ms: f32 = row_data.iter().map(|v| v * v).sum::<f32>() / h as f32;
        let rms = (ms + eps).sqrt();
        for col in 0..h {
            h2[row * h + col] = row_data[col] / rms * rms_w[col];
        }
    }
    let expected = super::test_utils::linear_ref(&h2, &w2, Some(&b2), batch, h, out_f);
    assert_close("rmsnorm_chain", &result, &expected, 1e-3);
}

// NOTE: LSTM compiled model test removed — LSTM MSL codegen is explicitly deferred.
// The verification path uses decomposed Linear+Sigmoid+Tanh+BinaryMul+BinaryAdd.
// Tracked for future: LSTM compiled pipeline requires trace-level decomposition.

// -- Test 10: Embedding + Linear -----------------------------------------------

/// Embedding(10, 8) -> Linear(8, 3): embedding lookup with f32 indices.
#[test]
fn test_compiled_embedding_linear() {
    super::test_utils::gpu_init();
    let cache = super::test_utils::metal_setup();

    let (vocab, edim, out_f, n_idx) = (10, 8, 3, 4);
    let emb_data = super::test_utils::rand_f32_vec(0xE8BD_1001, vocab * edim, -1.0, 1.0);
    let w = super::test_utils::rand_f32_vec(0xE8BD_1002, out_f * edim, -0.5, 0.5);
    let b = super::test_utils::rand_f32_vec(0xE8BD_1003, out_f, -0.1, 0.1);
    let indices: Vec<f32> = vec![0.0, 3.0, 7.0, 1.0];

    let graph = ComputationGraph::from_nodes(vec![
        input_node(0, &[n_idx]),
        TraceNode::new(
            1,
            "embedding_0".into(),
            TraceOp::Embedding {
                weight: weight(emb_data.clone(), vec![vocab, edim]),
            },
            vec![0],
            vec![n_idx, edim],
            DType::F32,
        ),
        TraceNode::new(
            2,
            "linear_0".into(),
            TraceOp::Linear {
                weight: weight(w.clone(), vec![out_f, edim]),
                bias: Some(weight(b.clone(), vec![out_f])),
            },
            vec![1],
            vec![n_idx, out_f],
            DType::F32,
        ),
    ]);

    let result = compile_and_run(
        &cache,
        graph,
        &[&create_input_buffer(&cache, &indices)],
        n_idx * out_f,
    );

    // CPU reference: embedding lookup then linear.
    let mut emb_out = vec![0.0_f32; n_idx * edim];
    for (i, &idx) in indices.iter().enumerate() {
        let row = idx as usize;
        emb_out[i * edim..(i + 1) * edim].copy_from_slice(&emb_data[row * edim..(row + 1) * edim]);
    }
    let expected = super::test_utils::linear_ref(&emb_out, &w, Some(&b), n_idx, edim, out_f);
    assert_close("embedding_linear", &result, &expected, 1e-4);
}

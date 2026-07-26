// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! End-to-end compiled model tests: BinaryAdd, BinaryMul, ConvTranspose1d,
//! IndexSelect, Gather, ReduceMax, Exp, LeakyRelu.
//!
//! Continuation of `compiled_model_ops_e2e_2.rs` (tests 24–33).
//! Exercises the full pipeline: build graph → compile → GPU execute → verify
//! against CPU reference.
//!
//! Part of #2270, #3230.

use nn_core::dyn_tensor::trace::{ComputationGraph, TraceNode, TraceOp, WeightRef};
use nn_core::DType;

use super::helpers::{assert_close, compile_and_run, create_input_buffer, input_node};

fn weight(data: Vec<f32>, shape: Vec<usize>) -> WeightRef {
    WeightRef::new(data, shape).expect("weight")
}

// -- Test 24: BinaryAdd (standalone, two variable inputs) ---------------------

/// Add: [2, 4] + [2, 4] -> [2, 4]. Two variable inputs, element-wise add.
#[test]
fn test_compiled_binary_add() {
    super::test_utils::gpu_init();
    let cache = super::test_utils::metal_setup();

    let (rows, cols) = (2, 4);
    let a_data = super::test_utils::rand_f32_vec(0xADD0_0001, rows * cols, -1.0, 1.0);
    let b_data = super::test_utils::rand_f32_vec(0xADD0_0002, rows * cols, -1.0, 1.0);

    let graph = ComputationGraph::from_nodes(vec![
        input_node(0, &[rows, cols]),
        input_node(1, &[rows, cols]),
        TraceNode::new(
            2,
            "add_0".into(),
            TraceOp::Add,
            vec![0, 1],
            vec![rows, cols],
            DType::F32,
        ),
    ]);

    let buf_a = create_input_buffer(&cache, &a_data);
    let buf_b = create_input_buffer(&cache, &b_data);
    let result = compile_and_run(&cache, graph, &[&buf_a, &buf_b], rows * cols);

    let expected: Vec<f32> = a_data
        .iter()
        .zip(b_data.iter())
        .map(|(a, b)| a + b)
        .collect();
    assert_close("binary_add", &result, &expected, 1e-6);
}

// -- Test 25: BinaryMul (standalone, two variable inputs) ---------------------

/// Mul: [3, 5] * [3, 5] -> [3, 5]. Two variable inputs, element-wise multiply.
#[test]
fn test_compiled_binary_mul() {
    super::test_utils::gpu_init();
    let cache = super::test_utils::metal_setup();

    let (rows, cols) = (3, 5);
    let a_data = super::test_utils::rand_f32_vec(0xA010_0001, rows * cols, -1.0, 1.0);
    let b_data = super::test_utils::rand_f32_vec(0xA010_0002, rows * cols, -1.0, 1.0);

    let graph = ComputationGraph::from_nodes(vec![
        input_node(0, &[rows, cols]),
        input_node(1, &[rows, cols]),
        TraceNode::new(
            2,
            "mul_0".into(),
            TraceOp::Mul,
            vec![0, 1],
            vec![rows, cols],
            DType::F32,
        ),
    ]);

    let buf_a = create_input_buffer(&cache, &a_data);
    let buf_b = create_input_buffer(&cache, &b_data);
    let result = compile_and_run(&cache, graph, &[&buf_a, &buf_b], rows * cols);

    let expected: Vec<f32> = a_data
        .iter()
        .zip(b_data.iter())
        .map(|(a, b)| a * b)
        .collect();
    assert_close("binary_mul", &result, &expected, 1e-6);
}

// -- Test 26: ConvTranspose1d -------------------------------------------------

/// ConvTranspose1d(4, 1, kernel=4, stride=4): transposed convolution for upsampling.
///
/// This op is critical for HTDemucs decoder. Tests the full compiled pipeline
/// with known weights.
#[test]
fn test_compiled_conv_transpose1d() {
    super::test_utils::gpu_init();
    let cache = super::test_utils::metal_setup();

    let (in_ch, out_ch, ks, stride, in_len) = (4, 1, 4, 4, 3);
    let out_len = (in_len - 1) * stride + ks; // no padding, no output_padding

    let w_data = super::test_utils::rand_f32_vec(0xC7D1_0001, in_ch * out_ch * ks, -0.5, 0.5);
    let b_data = super::test_utils::rand_f32_vec(0xC7D1_0002, out_ch, -0.1, 0.1);
    let input_data = super::test_utils::rand_f32_vec(0xC7D1_0003, in_ch * in_len, -1.0, 1.0);

    let graph = ComputationGraph::from_nodes(vec![
        input_node(0, &[in_ch, in_len]),
        TraceNode::new(
            1,
            "conv_transpose1d_0".into(),
            TraceOp::ConvTranspose1d {
                weight: weight(w_data.clone(), vec![in_ch, out_ch, ks]),
                bias: Some(weight(b_data.clone(), vec![out_ch])),
                padding: 0,
                output_padding: 0,
                stride,
                dilation: 1,
                groups: 1,
            },
            vec![0],
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

    // CPU reference: transposed convolution.
    // For each input channel and position, scatter-add weight*input to output.
    let mut expected = vec![0.0_f32; out_ch * out_len];
    for oc in 0..out_ch {
        for ic in 0..in_ch {
            for t in 0..in_len {
                let x_val = input_data[ic * in_len + t];
                for k in 0..ks {
                    let out_t = t * stride + k;
                    let w_idx = ic * out_ch * ks + oc * ks + k;
                    expected[oc * out_len + out_t] += x_val * w_data[w_idx];
                }
            }
        }
        // Add bias.
        for t in 0..out_len {
            expected[oc * out_len + t] += b_data[oc];
        }
    }
    assert_close("conv_transpose1d", &result, &expected, 1e-3);
}

// -- Test 27: IndexSelect -----------------------------------------------------

/// IndexSelect(dim=0): select rows from a [4, 3] tensor using indices [1, 3, 0].
#[test]
fn test_compiled_index_select() {
    super::test_utils::gpu_init();
    let cache = super::test_utils::metal_setup();

    let (rows, cols) = (4, 3);
    let input_data: Vec<f32> = (0..rows * cols).map(|i| i as f32).collect();
    // Indices as f32 (compiled model uses f32 index buffers).
    let indices: Vec<f32> = vec![1.0, 3.0, 0.0];
    let n_select = indices.len();

    let graph = ComputationGraph::from_nodes(vec![
        input_node(0, &[rows, cols]),
        input_node(1, &[n_select]),
        TraceNode::new(
            2,
            "index_select_0".into(),
            TraceOp::IndexSelect { dim: 0 },
            vec![0, 1],
            vec![n_select, cols],
            DType::F32,
        ),
    ]);

    let buf_input = create_input_buffer(&cache, &input_data);
    let buf_indices = create_input_buffer(&cache, &indices);
    let result = compile_and_run(&cache, graph, &[&buf_input, &buf_indices], n_select * cols);

    // CPU reference: gather rows by index.
    let mut expected = vec![0.0_f32; n_select * cols];
    for (i, &idx) in indices.iter().enumerate() {
        let row = idx as usize;
        expected[i * cols..(i + 1) * cols]
            .copy_from_slice(&input_data[row * cols..(row + 1) * cols]);
    }
    assert_close("index_select", &result, &expected, 1e-6);
}

// -- Test 28: Gather ----------------------------------------------------------

/// Gather(dim=1): gather elements from [2, 4] using index tensor [2, 3].
#[test]
fn test_compiled_gather() {
    super::test_utils::gpu_init();
    let cache = super::test_utils::metal_setup();

    let (batch, in_cols) = (2, 4);
    let out_cols = 3;
    let input_data: Vec<f32> = vec![10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0];
    // Gather indices (as f32): row 0 picks cols [2, 0, 3], row 1 picks cols [1, 3, 0].
    let index_data: Vec<f32> = vec![2.0, 0.0, 3.0, 1.0, 3.0, 0.0];

    let graph = ComputationGraph::from_nodes(vec![
        input_node(0, &[batch, in_cols]),
        input_node(1, &[batch, out_cols]),
        TraceNode::new(
            2,
            "gather_0".into(),
            TraceOp::Gather { dim: 1 },
            vec![0, 1],
            vec![batch, out_cols],
            DType::F32,
        ),
    ]);

    let buf_input = create_input_buffer(&cache, &input_data);
    let buf_index = create_input_buffer(&cache, &index_data);
    let result = compile_and_run(&cache, graph, &[&buf_input, &buf_index], batch * out_cols);

    // CPU reference: gather along dim 1.
    let mut expected = vec![0.0_f32; batch * out_cols];
    for b in 0..batch {
        for c in 0..out_cols {
            let idx = index_data[b * out_cols + c] as usize;
            expected[b * out_cols + c] = input_data[b * in_cols + idx];
        }
    }
    assert_close("gather", &result, &expected, 1e-6);
}

// -- Test 29: ReduceMax -------------------------------------------------------

/// Linear(4, 8) -> ReduceMax(dim=1, keepdim=false) -> [2]: max reduction.
#[test]
fn test_compiled_reduce_max() {
    super::test_utils::gpu_init();
    let cache = super::test_utils::metal_setup();

    let (batch, features) = (2, 8);
    let w = super::test_utils::rand_f32_vec(0xBED2_0001, features * 4, -0.5, 0.5);
    let b = super::test_utils::rand_f32_vec(0xBED2_0002, features, -0.1, 0.1);
    let input_data = super::test_utils::rand_f32_vec(0xBED2_0003, batch * 4, -1.0, 1.0);

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
            "reduce_max_0".into(),
            TraceOp::ReduceMax {
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
    let mut expected = vec![f32::NEG_INFINITY; batch];
    for r in 0..batch {
        for c in 0..features {
            expected[r] = expected[r].max(linear_out[r * features + c]);
        }
    }
    assert_close("reduce_max", &result, &expected, 1e-4);
}

// -- Test 30: BinaryAdd + BinaryMul chain (residual pattern) ------------------

/// Linear(4, 6) -> add(x, linear(x)) -> mul(result, result): residual + squaring.
///
/// Tests that standalone BinaryAdd and BinaryMul work in a multi-step pipeline.
#[test]
fn test_compiled_binary_residual_chain() {
    super::test_utils::gpu_init();
    let cache = super::test_utils::metal_setup();

    let (batch, in_f, out_f) = (2, 4, 4);
    let w = super::test_utils::rand_f32_vec(0xBE51_0001, out_f * in_f, -0.3, 0.3);
    let b = super::test_utils::rand_f32_vec(0xBE51_0002, out_f, -0.1, 0.1);
    let input_data = super::test_utils::rand_f32_vec(0xBE51_0003, batch * in_f, -1.0, 1.0);

    // Graph: input -> linear -> add(input, linear) -> mul(sum, sum)
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
            "add_residual".into(),
            TraceOp::Add,
            vec![0, 1], // residual: input + linear(input)
            vec![batch, out_f],
            DType::F32,
        ),
        TraceNode::new(
            3,
            "mul_square".into(),
            TraceOp::Mul,
            vec![2, 2], // square: result * result
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

    // CPU reference.
    let linear_out = super::test_utils::linear_ref(&input_data, &w, Some(&b), batch, in_f, out_f);
    let expected: Vec<f32> = input_data
        .iter()
        .zip(linear_out.iter())
        .map(|(x, l)| {
            let sum = x + l;
            sum * sum
        })
        .collect();
    assert_close("binary_residual_chain", &result, &expected, 1e-4);
}

// -- Test 31: Exp (standalone) ------------------------------------------------

/// Exp: [2, 4] -> exp -> [2, 4]. Element-wise exponential.
#[test]
fn test_compiled_exp() {
    super::test_utils::gpu_init();
    let cache = super::test_utils::metal_setup();

    let (rows, cols) = (2, 4);
    let input_data = super::test_utils::rand_f32_vec(0xE4F0_0001, rows * cols, -2.0, 2.0);

    let graph = ComputationGraph::from_nodes(vec![
        input_node(0, &[rows, cols]),
        TraceNode::new(
            1,
            "exp_0".into(),
            TraceOp::Exp,
            vec![0],
            vec![rows, cols],
            DType::F32,
        ),
    ]);

    let result = compile_and_run(
        &cache,
        graph,
        &[&create_input_buffer(&cache, &input_data)],
        rows * cols,
    );

    let expected: Vec<f32> = input_data.iter().map(|x| x.exp()).collect();
    assert_close("exp", &result, &expected, 1e-5);
}

// -- Test 32: LeakyRelu (standalone) ------------------------------------------

/// LeakyRelu(slope=0.01): [2, 6] -> leaky_relu -> [2, 6].
#[test]
fn test_compiled_leaky_relu() {
    super::test_utils::gpu_init();
    let cache = super::test_utils::metal_setup();

    let (rows, cols) = (2, 6);
    let input_data = super::test_utils::rand_f32_vec(0x1EA0_0001, rows * cols, -3.0, 3.0);
    let slope = 0.01_f32;

    let graph = ComputationGraph::from_nodes(vec![
        input_node(0, &[rows, cols]),
        TraceNode::new(
            1,
            "leaky_relu_0".into(),
            TraceOp::LeakyRelu {
                slope: f64::from(slope),
            },
            vec![0],
            vec![rows, cols],
            DType::F32,
        ),
    ]);

    let result = compile_and_run(
        &cache,
        graph,
        &[&create_input_buffer(&cache, &input_data)],
        rows * cols,
    );

    let expected: Vec<f32> = input_data
        .iter()
        .map(|&x| if x >= 0.0 { x } else { slope * x })
        .collect();
    assert_close("leaky_relu", &result, &expected, 1e-6);
}

// -- Test 33: Softplus (standalone) -------------------------------------------

/// Softplus: [2, 6] -> softplus -> [2, 6]. log(1 + exp(x)).
#[test]
fn test_compiled_softplus() {
    super::test_utils::gpu_init();
    let cache = super::test_utils::metal_setup();

    let (rows, cols) = (2, 6);
    let input_data = super::test_utils::rand_f32_vec(0x50F7_0001, rows * cols, -4.0, 4.0);

    let graph = ComputationGraph::from_nodes(vec![
        input_node(0, &[rows, cols]),
        TraceNode::new(
            1,
            "softplus_0".into(),
            TraceOp::Softplus,
            vec![0],
            vec![rows, cols],
            DType::F32,
        ),
    ]);

    let result = compile_and_run(
        &cache,
        graph,
        &[&create_input_buffer(&cache, &input_data)],
        rows * cols,
    );

    let expected: Vec<f32> = input_data.iter().map(|&x| x.exp().ln_1p()).collect();
    assert_close("softplus", &result, &expected, 1e-5);
}

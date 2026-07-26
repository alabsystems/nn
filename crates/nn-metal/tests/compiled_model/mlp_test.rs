// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! 5-layer MLP end-to-end integration test for `CompiledModel` (#2124 AC3).
//!
//! Builds a 5-layer MLP (Linear + ReLU × 4, then Linear), compiles via
//! `CompiledModel::builder().build()`, executes on GPU, and verifies outputs
//! match a CPU reference within 1e-5.

use nn_core::dyn_tensor::trace::{ComputationGraph, TraceNode, TraceOp, WeightRef};
use nn_core::DType;
use nn_metal::compiled_model::CompiledModel;

use super::helpers::{create_input_buffer, input_node, unary_node};

fn read_output_at(
    buf: &nn_metal::MetalBuffer,
    byte_offset: usize,
    num_elements: usize,
) -> Vec<f32> {
    debug_assert_eq!(
        byte_offset % size_of::<f32>(),
        0,
        "byte_offset must be f32-aligned"
    );
    let all = buf.contents::<f32>().expect("read GPU output");
    let start = byte_offset / size_of::<f32>();
    all[start..start + num_elements].to_vec()
}

/// Build an MLP computation graph: Linear nodes with ReLU after each except
/// the last. Returns `(graph, weights)` for CPU reference computation.
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

/// CPU reference for an MLP: Linear + ReLU per layer (no ReLU on last).
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

// -- Tests --------------------------------------------------------------------

/// 5-layer MLP: trace → compile → execute on GPU, verify outputs match
/// CPU reference within 1e-5. Documents dispatch count (#2124 AC3/AC4).
///
/// Architecture: Input [1,8] → Linear+ReLU ×4 → Linear → Output [1,4]
/// Layer dims: 8→16→16→16→16→4
///
/// Without fusion: 5 linear + 4 relu = 9 GPU dispatches.
/// Linear is a composite op (matmul + bias_add), so relu does not fuse
/// with it — the fusion optimizer targets pure elementwise chains.
#[test]
fn test_5_layer_mlp_gpu_matches_cpu() {
    super::test_utils::gpu_init();
    let cache = super::test_utils::metal_setup();

    let dims: &[(usize, usize)] = &[(8, 16), (16, 16), (16, 16), (16, 16), (16, 4)];
    let (graph, weights) = build_mlp_graph(dims);
    let compiled = CompiledModel::builder(&graph, &cache)
        .build()
        .expect("compile MLP");

    assert_eq!(compiled.num_inputs(), 1);
    assert_eq!(compiled.output_shape(), &[1, 4]);
    let nd = compiled.num_dispatches();
    // 5 linear + 4 relu = 9 unfused dispatches. Log actual count.
    assert!((1..=9).contains(&nd), "dispatches={nd}, expected 1..=9");
    eprintln!(
        "[MLP] steps={}, dispatches={nd} (unfused=9)",
        compiled.num_steps()
    );

    let input = super::test_utils::rand_f32_vec(42, 8, -1.0, 1.0);
    let buf = create_input_buffer(&cache, &input);
    let out = compiled.execute(&cache, &[&buf]).expect("execute MLP");

    let expected = mlp_cpu_ref(&input, dims, &weights);
    let result = read_output_at(&out, 0, expected.len());
    for (i, (r, e)) in result.iter().zip(expected.iter()).enumerate() {
        assert!(
            (r - e).abs() < 1e-5,
            "mlp[{i}]: gpu={r}, cpu={e}, diff={}",
            (r - e).abs()
        );
    }
}

/// BF16 relu: trace with BF16 dtype nodes, compile, execute on GPU,
/// verify output matches CPU reference. Tests the multi-dtype dispatch
/// path added for #2169.
#[test]
fn test_bf16_relu_compiled() {
    super::test_utils::gpu_init();
    let cache = super::test_utils::metal_setup();

    // Build a simple BF16 relu graph: input [4] -> relu -> output [4].
    let graph = ComputationGraph::from_nodes(vec![
        TraceNode::new(
            0,
            "input_0".into(),
            TraceOp::Input,
            vec![],
            vec![4],
            DType::BF16,
        ),
        TraceNode::new(
            1,
            "relu_0".into(),
            TraceOp::Relu,
            vec![0],
            vec![4],
            DType::BF16,
        ),
    ]);
    let compiled = CompiledModel::builder(&graph, &cache)
        .build()
        .expect("compile bf16 relu");

    assert_eq!(compiled.output_dtype(), DType::BF16);

    // Create BF16 input buffer: convert f32 data to f16 bits (Metal uses f16 for bf16).
    let data_f32: [f32; 4] = [-1.0, 0.5, -0.25, 2.0];
    let encoded: Vec<u16> = data_f32
        .iter()
        .map(|&v| half::f16::from_f32(v).to_bits())
        .collect();
    let buf = cache
        .context()
        .create_buffer(&encoded)
        .expect("create bf16 input");

    let out = compiled
        .execute(&cache, &[&buf])
        .expect("execute bf16 relu");

    // Read output as u16 (f16 bits) and convert back to f32 for comparison.
    let all_u16 = out.contents::<u16>().expect("read bf16 output");
    let start = 0;
    let result: Vec<f32> = all_u16[start..start + 4]
        .iter()
        .map(|&bits| half::f16::from_bits(bits).to_f32())
        .collect();

    // Expected: relu(-1.0)=0, relu(0.5)=0.5, relu(-0.25)=0, relu(2.0)=2.0
    let expected: [f32; 4] = [0.0, 0.5, 0.0, 2.0];
    for (i, (r, e)) in result.iter().zip(expected.iter()).enumerate() {
        assert!(
            (r - e).abs() < 1e-2,
            "bf16_relu[{i}]: gpu={r}, cpu={e}, diff={}",
            (r - e).abs()
        );
    }
}

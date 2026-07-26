// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Integration tests for packed elementwise dispatch (>28 inputs).
//!
//! Extracted from `tensor_dispatch_packed.rs` (#1669) to keep files under
//! 500 lines. Tests verify that scalar kernel dispatch with more than
//! `MAX_DIRECT_BINDING_INPUTS` (28) parameters correctly assembles packed
//! buffers and dispatches the packed kernel variant.
//!
//! Part of #1649.

use super::test_utils::{metal_setup, rand_f32_vec};
use nn_dsl::ir::{IRNode, IRNodeKind, KernelDef, NodeId, Param};
use nn_dsl::tensor_block_builder::TensorBlockBuilder;
use nn_dsl::ScalarType;
use nn_metal::execute_tensor_dispatch;
use std::collections::HashMap;

/// Build a scalar KernelDef with `n` parameters that sums them all.
fn sum_n_kernel(n: usize) -> KernelDef {
    let params: Vec<Param> = (0..n)
        .map(|i| Param::new(format!("p{i}"), ScalarType::F32))
        .collect();
    let mut nodes: Vec<IRNode> = (0..n)
        .map(|i| IRNode::new(NodeId::new(i), IRNodeKind::Param(i)))
        .collect();
    let sum_id = NodeId::new(n);
    let input_ids: Vec<NodeId> = (0..n).map(NodeId::new).collect();
    nodes.push(IRNode::new(
        sum_id,
        IRNodeKind::SumReduce { inputs: input_ids },
    ));
    KernelDef::new("sum_n", params, ScalarType::F32, nodes, sum_id)
}

// ===========================================================================
// Packed Elementwise — 35 inputs (general scalar kernel, exceeds 28)
// ===========================================================================

#[test]
fn test_packed_elementwise_35_inputs() {
    let cache = metal_setup();
    let n_inputs = 35;
    let elem_count = 16;

    // Build a scalar kernel that sums 35 parameters.
    let kernel = sum_n_kernel(n_inputs);

    // Each input is [1, elem_count]. Elementwise output is same shape.
    let mut b = TensorBlockBuilder::new("packed_elem_35");
    let mut input_ids = Vec::with_capacity(n_inputs);
    for i in 0..n_inputs {
        let name = format!("x{i}");
        let id = b.add_input(&name, &[1, elem_count]);
        input_ids.push(id);
    }
    let out_shape = [1, elem_count];
    let output = b.add_elementwise(kernel, &input_ids, &out_shape);
    let def = b.build(output).expect("valid elementwise graph");

    // Generate deterministic input data.
    let mut inputs = HashMap::new();
    let mut all_data = Vec::with_capacity(n_inputs);
    for i in 0..n_inputs {
        let name = format!("x{i}");
        let data = rand_f32_vec(0xDEAD_0000 + i as u64, elem_count, -1.0, 1.0);
        all_data.push(data.clone());
        inputs.insert(name, data);
    }

    let inputs_ref: HashMap<&str, &Vec<f32>> =
        inputs.iter().map(|(k, v)| (k.as_str(), v)).collect();

    let gpu_out = execute_tensor_dispatch(&cache, &def, ScalarType::F32, &inputs_ref)
        .expect("packed elementwise dispatch");

    // CPU reference: for each element position, sum across all 35 inputs.
    assert_eq!(gpu_out.len(), elem_count, "output length mismatch");

    let mut cpu_out = vec![0.0_f32; elem_count];
    for data in &all_data {
        for (j, &val) in data.iter().enumerate() {
            cpu_out[j] += val;
        }
    }

    for (i, (&g, &c)) in gpu_out.iter().zip(cpu_out.iter()).enumerate() {
        assert!(
            (g - c).abs() < 1e-4,
            "packed_elem_35[{i}]: gpu={g} cpu={c} diff={}",
            (g - c).abs()
        );
    }
}

// ===========================================================================
// Packed Elementwise — 50 inputs (stress test, well above threshold)
// ===========================================================================

#[test]
fn test_packed_elementwise_50_inputs() {
    let cache = metal_setup();
    let n_inputs = 50;
    let elem_count = 32;

    let kernel = sum_n_kernel(n_inputs);

    let mut b = TensorBlockBuilder::new("packed_elem_50");
    let mut input_ids = Vec::with_capacity(n_inputs);
    for i in 0..n_inputs {
        let name = format!("x{i}");
        let id = b.add_input(&name, &[1, elem_count]);
        input_ids.push(id);
    }
    let out_shape = [1, elem_count];
    let output = b.add_elementwise(kernel, &input_ids, &out_shape);
    let def = b.build(output).expect("valid elementwise graph");

    let mut inputs = HashMap::new();
    let mut all_data = Vec::with_capacity(n_inputs);
    for i in 0..n_inputs {
        let name = format!("x{i}");
        let data = rand_f32_vec(0xF00D_0000 + i as u64, elem_count, -0.5, 0.5);
        all_data.push(data.clone());
        inputs.insert(name, data);
    }

    let inputs_ref: HashMap<&str, &Vec<f32>> =
        inputs.iter().map(|(k, v)| (k.as_str(), v)).collect();

    let gpu_out = execute_tensor_dispatch(&cache, &def, ScalarType::F32, &inputs_ref)
        .expect("packed elementwise 50 dispatch");

    assert_eq!(gpu_out.len(), elem_count, "output length mismatch");

    let mut cpu_out = vec![0.0_f32; elem_count];
    for data in &all_data {
        for (j, &val) in data.iter().enumerate() {
            cpu_out[j] += val;
        }
    }

    for (i, (&g, &c)) in gpu_out.iter().zip(cpu_out.iter()).enumerate() {
        assert!(
            (g - c).abs() < 1e-3,
            "packed_elem_50[{i}]: gpu={g} cpu={c} diff={}",
            (g - c).abs()
        );
    }
}

// ===========================================================================
// Direct Elementwise — 4 inputs (below threshold, exercises existing path)
// ===========================================================================

#[test]
fn test_direct_elementwise_4_inputs() {
    let cache = metal_setup();
    let n_inputs = 4;
    let elem_count = 16;

    let kernel = sum_n_kernel(n_inputs);

    let mut b = TensorBlockBuilder::new("direct_elem_4");
    let mut input_ids = Vec::with_capacity(n_inputs);
    for i in 0..n_inputs {
        let name = format!("x{i}");
        let id = b.add_input(&name, &[1, elem_count]);
        input_ids.push(id);
    }
    let out_shape = [1, elem_count];
    let output = b.add_elementwise(kernel, &input_ids, &out_shape);
    let def = b.build(output).expect("valid elementwise graph");

    let mut inputs = HashMap::new();
    let mut all_data = Vec::with_capacity(n_inputs);
    for i in 0..n_inputs {
        let name = format!("x{i}");
        let data = rand_f32_vec(0x3333_0000 + i as u64, elem_count, -2.0, 2.0);
        all_data.push(data.clone());
        inputs.insert(name, data);
    }

    let inputs_ref: HashMap<&str, &Vec<f32>> =
        inputs.iter().map(|(k, v)| (k.as_str(), v)).collect();

    let gpu_out = execute_tensor_dispatch(&cache, &def, ScalarType::F32, &inputs_ref)
        .expect("direct elementwise dispatch");

    assert_eq!(gpu_out.len(), elem_count, "output length mismatch");

    let mut cpu_out = vec![0.0_f32; elem_count];
    for data in &all_data {
        for (j, &val) in data.iter().enumerate() {
            cpu_out[j] += val;
        }
    }

    for (i, (&g, &c)) in gpu_out.iter().zip(cpu_out.iter()).enumerate() {
        assert!(
            (g - c).abs() < 1e-5,
            "direct_elem_4[{i}]: gpu={g} cpu={c} diff={}",
            (g - c).abs()
        );
    }
}

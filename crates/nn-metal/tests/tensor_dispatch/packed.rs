// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Integration tests for packed buffer dispatch — Stack and Concat (>28 inputs).
//!
//! When a Stack or Concat operation has more than `MAX_DIRECT_BINDING_INPUTS`
//! (28) inputs, the MSL codegen switches to a packed kernel variant that packs
//! all input buffers into one contiguous buffer with an offsets array. These
//! tests verify that the dispatch layer correctly assembles packed buffers on
//! GPU and dispatches the packed kernel.
//!
//! Elementwise packed dispatch tests are in `tensor_dispatch_packed_elementwise.rs`.
//!
//! Axis 0 is reserved for the batch dimension in tensor IR, so structural
//! tests (Stack/Concat) use axis >= 1 with a leading batch dimension of 1.
//!
//! Part of #1649.

use super::test_utils::{metal_setup, rand_f32_vec};
use nn_dsl::tensor_block_builder::TensorBlockBuilder;
use nn_dsl::ScalarType;
use nn_metal::execute_tensor_dispatch;
use std::collections::HashMap;

// ===========================================================================
// Packed Stack — 30 inputs (exceeds MAX_DIRECT_BINDING_INPUTS = 28)
// ===========================================================================

#[test]
fn test_packed_stack_30_inputs() {
    let cache = metal_setup();
    let n_inputs = 30;
    let input_len = 8;

    // Each input is [1, 8], stack along axis 1 → output [1, 30, 8].
    let mut b = TensorBlockBuilder::new("packed_stack_30");
    let mut input_ids = Vec::with_capacity(n_inputs);
    for i in 0..n_inputs {
        let name = format!("x{i}");
        let id = b.add_input(&name, &[1, input_len]);
        input_ids.push(id);
    }
    let out_shape = [1, n_inputs, input_len];
    let stacked = b.add_stack(&input_ids, 1, &out_shape);
    let def = b.build(stacked).expect("valid stack graph");

    // Generate deterministic input data.
    let mut inputs = HashMap::new();
    let mut all_data = Vec::with_capacity(n_inputs);
    for i in 0..n_inputs {
        let name = format!("x{i}");
        let data = rand_f32_vec(0xABCD_0000 + i as u64, input_len, -1.0, 1.0);
        all_data.push(data.clone());
        inputs.insert(name, data);
    }

    let inputs_ref: HashMap<&str, &Vec<f32>> =
        inputs.iter().map(|(k, v)| (k.as_str(), v)).collect();

    let gpu_out = execute_tensor_dispatch(&cache, &def, ScalarType::F32, &inputs_ref)
        .expect("packed stack dispatch");

    // CPU reference: with batch=1 and stack along axis 1, flat layout =
    // concatenation of all input data in order.
    let expected_len = n_inputs * input_len;
    assert_eq!(gpu_out.len(), expected_len, "output length mismatch");

    let mut cpu_out = Vec::with_capacity(expected_len);
    for data in &all_data {
        cpu_out.extend_from_slice(data);
    }

    for (i, (&g, &c)) in gpu_out.iter().zip(cpu_out.iter()).enumerate() {
        assert!(
            (g - c).abs() < 1e-6,
            "packed_stack_30[{i}]: gpu={g} cpu={c} diff={}",
            (g - c).abs()
        );
    }
}

// ===========================================================================
// Packed Stack — 40 inputs with 3D shape (batch, channels, time)
// ===========================================================================

#[test]
fn test_packed_stack_40_inputs_3d() {
    let cache = metal_setup();
    let n_inputs = 40;
    let channels = 4;
    let time = 16;

    // Each input is [1, channels, time], stack along axis 1 →
    // output [1, 40, channels, time].
    let mut b = TensorBlockBuilder::new("packed_stack_40_3d");
    let mut input_ids = Vec::with_capacity(n_inputs);
    for i in 0..n_inputs {
        let name = format!("x{i}");
        let id = b.add_input(&name, &[1, channels, time]);
        input_ids.push(id);
    }
    let out_shape = [1, n_inputs, channels, time];
    let stacked = b.add_stack(&input_ids, 1, &out_shape);
    let def = b.build(stacked).expect("valid stack graph");

    let mut inputs = HashMap::new();
    let mut all_data = Vec::with_capacity(n_inputs);
    let elems_per_input = channels * time;
    for i in 0..n_inputs {
        let name = format!("x{i}");
        let data = rand_f32_vec(0xFEED_0000 + i as u64, elems_per_input, -2.0, 2.0);
        all_data.push(data.clone());
        inputs.insert(name, data);
    }

    let inputs_ref: HashMap<&str, &Vec<f32>> =
        inputs.iter().map(|(k, v)| (k.as_str(), v)).collect();

    let gpu_out = execute_tensor_dispatch(&cache, &def, ScalarType::F32, &inputs_ref)
        .expect("packed stack 40 3D dispatch");

    // With batch=1, flat layout = concatenation of all input data.
    let expected_len = n_inputs * elems_per_input;
    assert_eq!(gpu_out.len(), expected_len, "output length mismatch");

    let mut cpu_out = Vec::with_capacity(expected_len);
    for data in &all_data {
        cpu_out.extend_from_slice(data);
    }

    for (i, (&g, &c)) in gpu_out.iter().zip(cpu_out.iter()).enumerate() {
        assert!(
            (g - c).abs() < 1e-6,
            "packed_stack_40_3d[{i}]: gpu={g} cpu={c} diff={}",
            (g - c).abs()
        );
    }
}

// ===========================================================================
// Packed Concat — 30 inputs along axis 1
// ===========================================================================

#[test]
fn test_packed_concat_30_inputs() {
    let cache = metal_setup();
    let n_inputs = 30;
    let axis = 1;
    let per_input_axis = 4;
    let trailing = 8;

    // Each input is [1, 4, 8], concat along axis 1 → output [1, 120, 8].
    let mut b = TensorBlockBuilder::new("packed_concat_30");
    let mut input_ids = Vec::with_capacity(n_inputs);
    for i in 0..n_inputs {
        let name = format!("x{i}");
        let id = b.add_input(&name, &[1, per_input_axis, trailing]);
        input_ids.push(id);
    }
    let out_shape = [1, n_inputs * per_input_axis, trailing];
    let concatenated = b.add_concat(&input_ids, axis, &out_shape);
    let def = b.build(concatenated).expect("valid concat graph");

    let mut inputs = HashMap::new();
    let mut all_data = Vec::with_capacity(n_inputs);
    let elems_per_input = per_input_axis * trailing;
    for i in 0..n_inputs {
        let name = format!("x{i}");
        let data = rand_f32_vec(0xCAFE_0000 + i as u64, elems_per_input, -1.0, 1.0);
        all_data.push(data.clone());
        inputs.insert(name, data);
    }

    let inputs_ref: HashMap<&str, &Vec<f32>> =
        inputs.iter().map(|(k, v)| (k.as_str(), v)).collect();

    let gpu_out = execute_tensor_dispatch(&cache, &def, ScalarType::F32, &inputs_ref)
        .expect("packed concat dispatch");

    // CPU reference: with batch=1, concat along axis 1 with uniform axis
    // sizes = simple concatenation in row-major layout.
    let expected_len = n_inputs * elems_per_input;
    assert_eq!(gpu_out.len(), expected_len, "output length mismatch");

    let mut cpu_out = Vec::with_capacity(expected_len);
    for data in &all_data {
        cpu_out.extend_from_slice(data);
    }

    for (i, (&g, &c)) in gpu_out.iter().zip(cpu_out.iter()).enumerate() {
        assert!(
            (g - c).abs() < 1e-6,
            "packed_concat_30[{i}]: gpu={g} cpu={c} diff={}",
            (g - c).abs()
        );
    }
}

// ===========================================================================
// Direct Stack — 4 inputs (below threshold, exercises existing path)
// ===========================================================================

#[test]
fn test_direct_stack_4_inputs() {
    let cache = metal_setup();
    let n_inputs = 4;
    let input_len = 16;

    // Each input is [1, 16], stack along axis 1 → output [1, 4, 16].
    let mut b = TensorBlockBuilder::new("direct_stack_4");
    let mut input_ids = Vec::with_capacity(n_inputs);
    for i in 0..n_inputs {
        let name = format!("x{i}");
        let id = b.add_input(&name, &[1, input_len]);
        input_ids.push(id);
    }
    let out_shape = [1, n_inputs, input_len];
    let stacked = b.add_stack(&input_ids, 1, &out_shape);
    let def = b.build(stacked).expect("valid stack graph");

    let mut inputs = HashMap::new();
    let mut all_data = Vec::with_capacity(n_inputs);
    for i in 0..n_inputs {
        let name = format!("x{i}");
        let data = rand_f32_vec(0x1111_0000 + i as u64, input_len, -3.0, 3.0);
        all_data.push(data.clone());
        inputs.insert(name, data);
    }

    let inputs_ref: HashMap<&str, &Vec<f32>> =
        inputs.iter().map(|(k, v)| (k.as_str(), v)).collect();

    let gpu_out = execute_tensor_dispatch(&cache, &def, ScalarType::F32, &inputs_ref)
        .expect("direct stack dispatch");

    let expected_len = n_inputs * input_len;
    assert_eq!(gpu_out.len(), expected_len, "output length mismatch");

    let mut cpu_out = Vec::with_capacity(expected_len);
    for data in &all_data {
        cpu_out.extend_from_slice(data);
    }

    for (i, (&g, &c)) in gpu_out.iter().zip(cpu_out.iter()).enumerate() {
        assert!(
            (g - c).abs() < 1e-6,
            "direct_stack_4[{i}]: gpu={g} cpu={c} diff={}",
            (g - c).abs()
        );
    }
}

// ===========================================================================
// Direct Concat — 4 inputs (below threshold, exercises existing path)
// ===========================================================================

#[test]
fn test_direct_concat_4_inputs() {
    let cache = metal_setup();
    let n_inputs = 4;
    let axis = 1;
    let per_input_axis = 8;
    let trailing = 4;

    // Each input is [1, 8, 4], concat along axis 1 → output [1, 32, 4].
    let mut b = TensorBlockBuilder::new("direct_concat_4");
    let mut input_ids = Vec::with_capacity(n_inputs);
    for i in 0..n_inputs {
        let name = format!("x{i}");
        let id = b.add_input(&name, &[1, per_input_axis, trailing]);
        input_ids.push(id);
    }
    let out_shape = [1, n_inputs * per_input_axis, trailing];
    let concatenated = b.add_concat(&input_ids, axis, &out_shape);
    let def = b.build(concatenated).expect("valid concat graph");

    let mut inputs = HashMap::new();
    let mut all_data = Vec::with_capacity(n_inputs);
    let elems_per_input = per_input_axis * trailing;
    for i in 0..n_inputs {
        let name = format!("x{i}");
        let data = rand_f32_vec(0x2222_0000 + i as u64, elems_per_input, -1.5, 1.5);
        all_data.push(data.clone());
        inputs.insert(name, data);
    }

    let inputs_ref: HashMap<&str, &Vec<f32>> =
        inputs.iter().map(|(k, v)| (k.as_str(), v)).collect();

    let gpu_out = execute_tensor_dispatch(&cache, &def, ScalarType::F32, &inputs_ref)
        .expect("direct concat dispatch");

    let expected_len = n_inputs * elems_per_input;
    assert_eq!(gpu_out.len(), expected_len, "output length mismatch");

    let mut cpu_out = Vec::with_capacity(expected_len);
    for data in &all_data {
        cpu_out.extend_from_slice(data);
    }

    for (i, (&g, &c)) in gpu_out.iter().zip(cpu_out.iter()).enumerate() {
        assert!(
            (g - c).abs() < 1e-6,
            "direct_concat_4[{i}]: gpu={g} cpu={c} diff={}",
            (g - c).abs()
        );
    }
}

// ===========================================================================
// Packed Stack — boundary (exactly 29 = MAX_DIRECT_BINDING_INPUTS + 1)
// ===========================================================================

#[test]
fn test_packed_stack_boundary_29_inputs() {
    let cache = metal_setup();
    let n_inputs = 29;
    let input_len = 4;

    // Each input is [1, 4], stack along axis 1 → output [1, 29, 4].
    let mut b = TensorBlockBuilder::new("packed_stack_29");
    let mut input_ids = Vec::with_capacity(n_inputs);
    for i in 0..n_inputs {
        let name = format!("x{i}");
        let id = b.add_input(&name, &[1, input_len]);
        input_ids.push(id);
    }
    let out_shape = [1, n_inputs, input_len];
    let stacked = b.add_stack(&input_ids, 1, &out_shape);
    let def = b.build(stacked).expect("valid stack graph");

    let mut inputs = HashMap::new();
    let mut all_data = Vec::with_capacity(n_inputs);
    for i in 0..n_inputs {
        let name = format!("x{i}");
        let data = rand_f32_vec(0xBEEF_0000 + i as u64, input_len, -1.0, 1.0);
        all_data.push(data.clone());
        inputs.insert(name, data);
    }

    let inputs_ref: HashMap<&str, &Vec<f32>> =
        inputs.iter().map(|(k, v)| (k.as_str(), v)).collect();

    let gpu_out = execute_tensor_dispatch(&cache, &def, ScalarType::F32, &inputs_ref)
        .expect("packed stack 29 dispatch");

    let expected_len = n_inputs * input_len;
    assert_eq!(gpu_out.len(), expected_len, "output length mismatch");

    let mut cpu_out = Vec::with_capacity(expected_len);
    for data in &all_data {
        cpu_out.extend_from_slice(data);
    }

    for (i, (&g, &c)) in gpu_out.iter().zip(cpu_out.iter()).enumerate() {
        assert!(
            (g - c).abs() < 1e-6,
            "packed_stack_29[{i}]: gpu={g} cpu={c} diff={}",
            (g - c).abs()
        );
    }
}

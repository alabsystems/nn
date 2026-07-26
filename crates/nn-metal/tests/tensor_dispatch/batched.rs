// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Direct unit tests for `execute_tensor_dispatch_batched`.
//!
//! The batched dispatch API was previously tested only indirectly through the
//! spectral encoder/decoder modules. These tests exercise the three code paths
//! directly: empty batch, single-element fast path, and multi-element batch.
//!
//! Uses `TensorBlockBuilder` to construct a simple elementwise kernel (sigmoid)
//! that is stable across API rename cycles.
//!
//! Part of #868 — batched spectral dispatch.

use super::test_utils::{metal_setup, rand_f32_vec};
use nn_dsl::tensor_block_builder::TensorBlockBuilder;
use nn_dsl::ScalarType;
use nn_metal::{execute_tensor_dispatch, execute_tensor_dispatch_batched};
use std::collections::HashMap;

/// CPU sigmoid reference: 1 / (1 + exp(-x)).
fn sigmoid_ref(data: &[f32]) -> Vec<f32> {
    data.iter().map(|&x| 1.0 / (1.0 + (-x).exp())).collect()
}

/// Build a sigmoid kernel via TensorBlockBuilder.
fn build_sigmoid_kernel(len: usize) -> nn_dsl::TensorKernelDef {
    let mut b = TensorBlockBuilder::new("batched_sigmoid");
    let x = b.add_input("x", &[len]);
    let out = b.add_sigmoid(x, &[len]);
    b.build(out).expect("valid sigmoid graph")
}

// ===========================================================================
// Empty batch: returns Ok(Vec::new())
// ===========================================================================

/// Empty batch_inputs should return an empty result vector without touching GPU.
#[test]
fn test_batched_dispatch_empty_batch() {
    let cache = metal_setup();
    let kernel = build_sigmoid_kernel(16);

    let batch_inputs: Vec<HashMap<&str, Vec<f32>>> = vec![];
    let results = execute_tensor_dispatch_batched(&cache, &kernel, ScalarType::F32, &batch_inputs)
        .expect("empty batch should succeed");

    assert!(
        results.is_empty(),
        "empty batch should return empty results"
    );
}

// ===========================================================================
// Single-element batch: delegates to non-batched fast path
// ===========================================================================

/// Single-element batch should produce identical results to non-batched dispatch.
#[test]
fn test_batched_dispatch_single_element_matches_non_batched() {
    let cache = metal_setup();
    let len = 64;
    let kernel = build_sigmoid_kernel(len);

    let x_data = rand_f32_vec(0xBA7C_0001, len, -3.0, 3.0);
    let cpu_out = sigmoid_ref(&x_data);

    let mut inputs = HashMap::new();
    inputs.insert("x", x_data);

    // Non-batched dispatch
    let single_out = execute_tensor_dispatch(&cache, &kernel, ScalarType::F32, &inputs)
        .expect("non-batched dispatch");

    // Batched dispatch with 1 element
    let batch_results =
        execute_tensor_dispatch_batched(&cache, &kernel, ScalarType::F32, &[inputs])
            .expect("single-element batch");

    assert_eq!(
        batch_results.len(),
        1,
        "single-element batch should return 1 result"
    );
    assert_eq!(
        single_out.len(),
        batch_results[0].len(),
        "single-element batch output length must match non-batched"
    );

    // Both must match CPU reference within tolerance
    for (i, ((&gpu, &cpu), &batched)) in single_out
        .iter()
        .zip(cpu_out.iter())
        .zip(batch_results[0].iter())
        .enumerate()
    {
        let delta_single = (gpu - cpu).abs();
        assert!(
            delta_single < 1e-5,
            "non-batched[{i}]: gpu={gpu}, cpu={cpu}, delta={delta_single:.6e}"
        );
        let delta_batch = (batched - cpu).abs();
        assert!(
            delta_batch < 1e-5,
            "batched[{i}]: gpu={batched}, cpu={cpu}, delta={delta_batch:.6e}"
        );
    }

    // Batched and non-batched must be bit-identical (same code path via delegation)
    assert_eq!(
        single_out, batch_results[0],
        "single-element batch must be bit-identical to non-batched dispatch"
    );
}

// ===========================================================================
// Multi-element batch: distinct inputs, single commit_and_wait
// ===========================================================================

/// Multi-element batch: each element uses different data, results must match
/// independent non-batched dispatches.
#[test]
fn test_batched_dispatch_multi_element() {
    let cache = metal_setup();
    let len = 32;
    let n_batch = 4;
    let kernel = build_sigmoid_kernel(len);

    // Build 4 distinct input sets with different seeds
    let mut batch_inputs = Vec::with_capacity(n_batch);
    let mut cpu_refs = Vec::with_capacity(n_batch);
    for i in 0..n_batch {
        let seed = 0xBA7C_0010 + i as u64;
        let x_data = rand_f32_vec(seed, len, -3.0, 3.0);
        let cpu_out = sigmoid_ref(&x_data);

        let mut inputs = HashMap::new();
        inputs.insert("x", x_data);
        batch_inputs.push(inputs);
        cpu_refs.push(cpu_out);
    }

    // Batched dispatch
    let batch_results =
        execute_tensor_dispatch_batched(&cache, &kernel, ScalarType::F32, &batch_inputs)
            .expect("multi-element batch");

    assert_eq!(
        batch_results.len(),
        n_batch,
        "batch results count must match input count"
    );

    // Each result must match its own CPU reference
    for (i, (gpu, cpu)) in batch_results.iter().zip(cpu_refs.iter()).enumerate() {
        assert_eq!(
            gpu.len(),
            cpu.len(),
            "batch element {i}: output length mismatch"
        );
        for (j, (&g, &c)) in gpu.iter().zip(cpu.iter()).enumerate() {
            let delta = (g - c).abs();
            assert!(
                delta < 1e-5,
                "batch[{i}][{j}]: gpu={g}, cpu={c}, delta={delta:.6e}"
            );
        }
    }
}

/// Multi-element batch results must be order-preserving: result[i] matches
/// input[i], not some permuted order.
#[test]
fn test_batched_dispatch_preserves_order() {
    let cache = metal_setup();
    let len = 32;
    let kernel = build_sigmoid_kernel(len);

    // Build 3 input sets with very different seeds to ensure distinct outputs
    let seeds = [0xAAAA_0001u64, 0xBBBB_0002, 0xCCCC_0003];
    let mut batch_inputs = Vec::with_capacity(seeds.len());
    let mut individual_results = Vec::with_capacity(seeds.len());

    for &seed in &seeds {
        let x_data = rand_f32_vec(seed, len, -5.0, 5.0);
        let mut inputs = HashMap::new();
        inputs.insert("x", x_data);
        // Run each individually to get reference results
        let single = execute_tensor_dispatch(&cache, &kernel, ScalarType::F32, &inputs)
            .expect("individual dispatch");
        individual_results.push(single);
        batch_inputs.push(inputs);
    }

    // Run all as batch
    let batch_results =
        execute_tensor_dispatch_batched(&cache, &kernel, ScalarType::F32, &batch_inputs)
            .expect("batched dispatch");

    assert_eq!(batch_results.len(), seeds.len());

    for (i, (batched, individual)) in batch_results
        .iter()
        .zip(individual_results.iter())
        .enumerate()
    {
        assert_eq!(
            batched, individual,
            "batch element {i} must match individual dispatch (order preservation)"
        );
    }
}

// ===========================================================================
// Error path: dtype mismatch (W4-315 fix — was debug_assert, now Result)
// ===========================================================================

/// Non-batched dispatch with dtype mismatch should return DtypeMismatch error.
///
/// Passing `ScalarType::F16` while the generic type is `f32` would cause the
/// GPU to run f16 kernels while the CPU reads the output as f32 — silent
/// data corruption in release builds before the fix in W4-315.
#[test]
fn test_dispatch_dtype_mismatch_error() {
    let cache = metal_setup();
    let kernel = build_sigmoid_kernel(16);

    let x_data = rand_f32_vec(0xD7E0_0001, 16, -2.0, 2.0);
    let mut inputs = HashMap::new();
    inputs.insert("x", x_data);

    // f32 element type but F16 scalar type → DtypeMismatch
    let result =
        execute_tensor_dispatch::<f32, Vec<f32>>(&cache, &kernel, ScalarType::F16, &inputs);

    assert!(
        result.is_err(),
        "dtype mismatch should produce an error, got: {result:?}"
    );
    let err = result.unwrap_err();
    let err_str = format!("{err:?}");
    assert!(
        err_str.contains("DtypeMismatch"),
        "error should be DtypeMismatch, got: {err_str}"
    );
}

/// Batched dispatch (multi-element) with dtype mismatch should return DtypeMismatch.
///
/// The batched path's DtypeMismatch check is only reachable with 2+ elements
/// (single-element batches delegate to the non-batched function).
#[test]
fn test_batched_dispatch_dtype_mismatch_error() {
    let cache = metal_setup();
    let len = 16;
    let kernel = build_sigmoid_kernel(len);

    let x1 = rand_f32_vec(0xD7E0_0002, len, -2.0, 2.0);
    let x2 = rand_f32_vec(0xD7E0_0003, len, -2.0, 2.0);
    let mut inputs1 = HashMap::new();
    inputs1.insert("x", x1);
    let mut inputs2 = HashMap::new();
    inputs2.insert("x", x2);

    let batch_inputs = vec![inputs1, inputs2];

    // f32 element type but F16 scalar type → DtypeMismatch
    let result = execute_tensor_dispatch_batched::<f32, Vec<f32>>(
        &cache,
        &kernel,
        ScalarType::F16,
        &batch_inputs,
    );

    assert!(
        result.is_err(),
        "batched dtype mismatch should produce an error, got: {result:?}"
    );
    let err = result.unwrap_err();
    let err_str = format!("{err:?}");
    assert!(
        err_str.contains("DtypeMismatch"),
        "error should be DtypeMismatch, got: {err_str}"
    );
}

// ===========================================================================
// Error path: missing input name
// ===========================================================================

/// Batched dispatch with a missing input name should return MissingInput error.
#[test]
fn test_batched_dispatch_missing_input_error() {
    let cache = metal_setup();
    let len = 16;
    let kernel = build_sigmoid_kernel(len);

    // First element valid, second element missing "x"
    let x_data = rand_f32_vec(0xEEE0_0001, len, -2.0, 2.0);
    let mut valid_inputs = HashMap::new();
    valid_inputs.insert("x", x_data);

    let bad_inputs: HashMap<&str, Vec<f32>> = HashMap::new();

    let batch_inputs = vec![valid_inputs, bad_inputs];
    let result = execute_tensor_dispatch_batched::<f32, Vec<f32>>(
        &cache,
        &kernel,
        ScalarType::F32,
        &batch_inputs,
    );

    assert!(
        result.is_err(),
        "missing input should produce an error, got: {result:?}"
    );
    let err = result.unwrap_err();
    let err_str = format!("{err:?}");
    assert!(
        err_str.contains("MissingInput"),
        "error should be MissingInput, got: {err_str}"
    );
}

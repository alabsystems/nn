// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Cross-backend contract tests for Embedding lookup tensor kernel:
//! GPU output within NY verified bounds.
//!
//! Tests the full pipeline: IR → dispatch plan → MSL codegen → Metal execution,
//! verified against NY IBP bounds from `tensor_kernel_to_graph`.
//!
//! Part of #743 (Direction 5).

use super::test_utils::{metal_setup, rand_f32_vec};

use nn_dsl::tensor_block_builder::TensorBlockBuilder;
use nn_dsl::ScalarType;
use nn_metal::execute_tensor_dispatch;
use nn_verify::{tensor_kernel_to_graph, BoundedTensor, TensorParamBinding};
use ndarray::{ArrayD, IxDyn};
use std::collections::HashMap;

fn constant_tensor(shape: &[usize], data: Vec<f32>) -> TensorParamBinding {
    TensorParamBinding::ConstantTensor(
        ArrayD::from_shape_vec(IxDyn(shape), data).expect("shape/data length mismatch"),
    )
}

/// Prove IBP bounds for an Embedding tensor kernel.
///
/// The NY graph models embedding as `AddConstant(midpoint)` where
/// midpoint is `(min + max) / 2` per dimension. To get proper IBP bounds
/// that cover all possible row selections, we provide input bounds of
/// `[-half_width, +half_width]` so that `midpoint ± half_width = [min, max]`.
///
/// This also verifies the graph builds correctly.
fn prove_embedding_bounds(
    def: &nn_dsl::TensorKernelDef,
    bindings: &[TensorParamBinding],
    weight_data: &[f32],
    num_embeddings: usize,
    embedding_dim: usize,
) -> (ArrayD<f32>, ArrayD<f32>) {
    let graph = tensor_kernel_to_graph(def, bindings).expect("embedding graph");

    // Compute per-dimension half-width from the weight table.
    // The AddConstant(midpoint) layer adds midpoint to input.
    // With input bounds [-hw, +hw], output = [midpoint - hw, midpoint + hw] = [min, max].
    let mut lower_in = Vec::with_capacity(embedding_dim);
    let mut upper_in = Vec::with_capacity(embedding_dim);
    for d in 0..embedding_dim {
        let mut lo = f32::INFINITY;
        let mut hi = f32::NEG_INFINITY;
        for row in 0..num_embeddings {
            let val = weight_data[row * embedding_dim + d];
            if val < lo {
                lo = val;
            }
            if val > hi {
                hi = val;
            }
        }
        let half_width = (hi - lo) / 2.0;
        lower_in.push(-half_width);
        upper_in.push(half_width);
    }

    let lower = ArrayD::from_shape_vec(IxDyn(&[embedding_dim]), lower_in).expect("lower shape");
    let upper = ArrayD::from_shape_vec(IxDyn(&[embedding_dim]), upper_in).expect("upper shape");
    let input_bounds = BoundedTensor::new(lower, upper).expect("input bounds");
    let output_bounds = graph.propagate_ibp(&input_bounds).expect("IBP propagation");
    let (lo, hi) = output_bounds.lower_upper();
    assert!(
        lo.iter().all(|v| v.is_finite()),
        "proved lower must be finite"
    );
    assert!(
        hi.iter().all(|v| v.is_finite()),
        "proved upper must be finite"
    );
    (lo.clone(), hi.clone())
}

/// Build an embedding tensor kernel using TensorBlockBuilder.
fn build_embedding_kernel(
    name: &str,
    num_indices: usize,
    num_embeddings: usize,
    embedding_dim: usize,
) -> nn_dsl::TensorKernelDef {
    let mut b = TensorBlockBuilder::new(name);
    let idx = b.add_input("indices", &[num_indices]);
    let w = b.add_input("weight", &[num_embeddings, embedding_dim]);
    let emb = b.add_embedding(idx, w, &[num_indices, embedding_dim]);
    b.build(emb).expect("valid graph")
}

// ===========================================================================
// Embedding contract tests
// ===========================================================================

/// Embedding contract: 4 indices into a 10×8 weight table.
/// GPU output must match exact row lookups within proved bounds.
/// Part of #743.
#[test]
fn test_embedding_gpu_output_within_verified_bounds() {
    let (num_indices, num_embeddings, embedding_dim) = (4, 10, 8);

    let def = build_embedding_kernel("embed_contract", num_indices, num_embeddings, embedding_dim);

    let weight_data = rand_f32_vec(0xE8BD_0001, num_embeddings * embedding_dim, -1.0, 1.0);
    let bindings = vec![
        TensorParamBinding::Variable,
        constant_tensor(&[num_embeddings, embedding_dim], weight_data.clone()),
    ];

    let (proved_lo, proved_hi) =
        prove_embedding_bounds(&def, &bindings, &weight_data, num_embeddings, embedding_dim);

    let cache = metal_setup();

    // Indices: select rows 0, 3, 7, 1 (passed as f32).
    let indices: Vec<f32> = vec![0.0, 3.0, 7.0, 1.0];
    let mut inputs = HashMap::new();
    inputs.insert("indices", indices.clone());
    inputs.insert("weight", weight_data.clone());

    let gpu_out = execute_tensor_dispatch(&cache, &def, ScalarType::F32, &inputs)
        .expect("embedding GPU dispatch");
    assert_eq!(gpu_out.len(), num_indices * embedding_dim, "output length");

    // Verify GPU output matches exact row lookups (CPU reference).
    for (i, &idx_f) in indices.iter().enumerate() {
        let row = idx_f as usize;
        for d in 0..embedding_dim {
            let expected = weight_data[row * embedding_dim + d];
            let actual = gpu_out[i * embedding_dim + d];
            assert!(
                (actual - expected).abs() < 1e-6,
                "embedding[{i}][{d}]: expected {expected}, got {actual}",
            );
        }
    }

    // Also verify within proved IBP bounds. The bounds are per-dimension
    // (shape [embedding_dim]) and apply to every output position.
    let lo_slice = proved_lo.as_slice().expect("contiguous lower");
    let hi_slice = proved_hi.as_slice().expect("contiguous upper");
    for (i, &val) in gpu_out.iter().enumerate() {
        let d = i % embedding_dim;
        let lo = lo_slice[d];
        let hi = hi_slice[d];
        let ulp_margin = (hi - lo).abs() * f32::EPSILON;
        assert!(
            val >= lo - ulp_margin && val <= hi + ulp_margin,
            "embedding output[{i}] violates proved bounds: gpu={val}, proved=[{lo}, {hi}]",
        );
    }
}

/// Embedding contract: sequential indices covering all rows.
/// Part of #743.
#[test]
fn test_embedding_gpu_all_rows() {
    let (num_embeddings, embedding_dim) = (6, 4);

    let def = build_embedding_kernel("embed_all", num_embeddings, num_embeddings, embedding_dim);

    let weight_data: Vec<f32> = (0..num_embeddings * embedding_dim)
        .map(|i| i as f32 * 0.1)
        .collect();
    let bindings = vec![
        TensorParamBinding::Variable,
        constant_tensor(&[num_embeddings, embedding_dim], weight_data.clone()),
    ];

    let (proved_lo, proved_hi) =
        prove_embedding_bounds(&def, &bindings, &weight_data, num_embeddings, embedding_dim);

    let cache = metal_setup();

    // Select all rows in order: 0, 1, 2, 3, 4, 5.
    let indices: Vec<f32> = (0..num_embeddings).map(|i| i as f32).collect();
    let mut inputs = HashMap::new();
    inputs.insert("indices", indices);
    inputs.insert("weight", weight_data.clone());

    let gpu_out = execute_tensor_dispatch(&cache, &def, ScalarType::F32, &inputs)
        .expect("embedding all-rows GPU dispatch");
    assert_eq!(gpu_out.len(), num_embeddings * embedding_dim);

    // GPU output should exactly match the flattened weight table.
    for (i, (&actual, &expected)) in gpu_out.iter().zip(weight_data.iter()).enumerate() {
        assert!(
            (actual - expected).abs() < 1e-6,
            "embedding all-rows[{i}]: expected {expected}, got {actual}",
        );
    }

    // Verify within proved bounds (per-dimension).
    let lo_slice = proved_lo.as_slice().expect("contiguous lower");
    let hi_slice = proved_hi.as_slice().expect("contiguous upper");
    for (i, &val) in gpu_out.iter().enumerate() {
        let d = i % embedding_dim;
        let lo = lo_slice[d];
        let hi = hi_slice[d];
        let ulp_margin = (hi - lo).abs() * f32::EPSILON;
        assert!(
            val >= lo - ulp_margin && val <= hi + ulp_margin,
            "embedding all-rows output[{i}] violates bounds: gpu={val}, proved=[{lo}, {hi}]",
        );
    }
}

/// Embedding contract: dvoice-representative dimensions (256 phonemes, 64 dims).
/// Part of #743.
#[test]
fn test_embedding_gpu_dvoice_dims() {
    let (num_indices, num_embeddings, embedding_dim) = (32, 256, 64);

    let def = build_embedding_kernel("embed_dv", num_indices, num_embeddings, embedding_dim);

    let weight_data = rand_f32_vec(0xDA5E_0001, num_embeddings * embedding_dim, -0.5, 0.5);
    let bindings = vec![
        TensorParamBinding::Variable,
        constant_tensor(&[num_embeddings, embedding_dim], weight_data.clone()),
    ];

    let (proved_lo, proved_hi) =
        prove_embedding_bounds(&def, &bindings, &weight_data, num_embeddings, embedding_dim);

    let cache = metal_setup();

    // Random indices in [0, 255].
    let indices: Vec<f32> = rand_f32_vec(0xDA5E_0002, num_indices, 0.0, 255.0)
        .iter()
        .map(|v| v.floor())
        .collect();
    let mut inputs = HashMap::new();
    inputs.insert("indices", indices.clone());
    inputs.insert("weight", weight_data.clone());

    let gpu_out = execute_tensor_dispatch(&cache, &def, ScalarType::F32, &inputs)
        .expect("dvoice embedding GPU dispatch");
    assert_eq!(gpu_out.len(), num_indices * embedding_dim);

    // Verify each element matches the expected row lookup.
    for (i, &idx_f) in indices.iter().enumerate() {
        let row = idx_f as usize;
        for d in 0..embedding_dim {
            let expected = weight_data[row * embedding_dim + d];
            let actual = gpu_out[i * embedding_dim + d];
            assert!(
                (actual - expected).abs() < 1e-6,
                "dvoice embedding[{i}][{d}]: expected {expected}, got {actual}",
            );
        }
    }

    // Verify within proved IBP bounds.
    let lo_slice = proved_lo.as_slice().expect("contiguous lower");
    let hi_slice = proved_hi.as_slice().expect("contiguous upper");
    for (i, &val) in gpu_out.iter().enumerate() {
        let d = i % embedding_dim;
        let lo = lo_slice[d];
        let hi = hi_slice[d];
        let ulp_margin = (hi - lo).abs() * f32::EPSILON;
        assert!(
            val >= lo - ulp_margin && val <= hi + ulp_margin,
            "dvoice embedding output[{i}] violates bounds: gpu={val}, proved=[{lo}, {hi}]",
        );
    }
}

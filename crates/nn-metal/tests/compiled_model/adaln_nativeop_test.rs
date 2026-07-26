// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! End-to-end compiled model test for fused NativeOp::AdaLayerNorm.
//!
//! Exercises the full pipeline: build trace graph -> compile (NativeOpKind) ->
//! GPU execute via fused MSL kernel -> verify against CPU reference.
//!
//! Part of #2482 (Fused AdaLayerNorm MSL kernel).

use nn_core::dyn_tensor::trace::{ComputationGraph, KokoroFusedOp, TraceNode, TraceOp, WeightRef};
use nn_core::DType;

use super::helpers::{assert_close, compile_and_run, create_input_buffer, input_node};

// -- CPU reference ------------------------------------------------------------

/// CPU AdaLayerNorm: LayerNorm(x, w, b, eps) then adaptive affine.
///
/// Input x: `[B, T, C]` row-major.
/// gamma/beta: `[B, C]` (per batch, applied uniformly across T).
/// norm_weight/norm_bias: `[C]` (per-channel LayerNorm parameters).
///
/// For each (b, t):
///   normed[c] = (x[b,t,c] - mean) / sqrt(var + eps) * w[c] + b[c]
///   output[b,t,c] = (1 + gamma[b,c]) * normed[c] + beta[b,c]
fn cpu_ada_layer_norm(
    x: &[f32],
    gamma: &[f32],
    beta: &[f32],
    norm_weight: &[f32],
    norm_bias: &[f32],
    batch: usize,
    time: usize,
    hidden: usize,
    eps: f32,
) -> Vec<f32> {
    let mut output = vec![0.0_f32; batch * time * hidden];
    for b in 0..batch {
        for t in 0..time {
            let row_offset = (b * time + t) * hidden;
            let row = &x[row_offset..row_offset + hidden];

            // Mean over hidden dim
            let mean: f32 = row.iter().sum::<f32>() / hidden as f32;
            // Variance over hidden dim
            let var: f32 = row.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / hidden as f32;
            let inv_std = 1.0 / (var + eps).sqrt();

            for c in 0..hidden {
                // LayerNorm
                let normed = (row[c] - mean) * inv_std * norm_weight[c] + norm_bias[c];
                // Adaptive affine: (1 + gamma) * normed + beta
                let g = gamma[b * hidden + c];
                let be = beta[b * hidden + c];
                output[row_offset + c] = (1.0 + g) * normed + be;
            }
        }
    }
    output
}

// -- Test: AdaLayerNorm NativeOp through CompiledModel ------------------------

/// [1, 4, 16] -> AdaLayerNorm(eps=1e-5): fused GPU kernel.
///
/// Verifies NativeOpKind::AdaLayerNorm with 3 tensor inputs (x, gamma, beta)
/// and 2 weights (norm_weight, norm_bias) executes correctly through the
/// compiled model pipeline.
#[test]
fn test_compiled_ada_layer_norm_nativeop() {
    super::test_utils::gpu_init();
    let cache = super::test_utils::metal_setup();

    let (batch, time, hidden) = (1, 4, 16);
    let eps = 1e-5_f64;

    let x_data = super::test_utils::rand_f32_vec(0xAD1A_0001, batch * time * hidden, -1.0, 1.0);
    let gamma_data = super::test_utils::rand_f32_vec(0xAD1A_0002, batch * hidden, -0.3, 0.3);
    let beta_data = super::test_utils::rand_f32_vec(0xAD1A_0003, batch * hidden, -0.2, 0.2);
    let norm_w_data = super::test_utils::rand_f32_vec(0xAD1A_0004, hidden, 0.8, 1.2);
    let norm_b_data = super::test_utils::rand_f32_vec(0xAD1A_0005, hidden, -0.1, 0.1);

    let norm_weight = WeightRef::new(norm_w_data.clone(), vec![hidden]).expect("norm_weight");
    let norm_bias = WeightRef::new(norm_b_data.clone(), vec![hidden]).expect("norm_bias");

    let graph = ComputationGraph::from_nodes(vec![
        input_node(0, &[batch, time, hidden]),
        input_node(1, &[batch, 1, hidden]),
        input_node(2, &[batch, 1, hidden]),
        TraceNode::new(
            3,
            "ada_layer_norm_0".into(),
            TraceOp::KokoroFused(KokoroFusedOp::AdaLayerNorm {
                norm_weight,
                norm_bias,
                eps,
            }),
            vec![0, 1, 2],
            vec![batch, time, hidden],
            DType::F32,
        ),
    ]);

    let x_buf = create_input_buffer(&cache, &x_data);
    let gamma_buf = create_input_buffer(&cache, &gamma_data);
    let beta_buf = create_input_buffer(&cache, &beta_data);

    let result = compile_and_run(
        &cache,
        graph,
        &[&x_buf, &gamma_buf, &beta_buf],
        batch * time * hidden,
    );

    let expected = cpu_ada_layer_norm(
        &x_data,
        &gamma_data,
        &beta_data,
        &norm_w_data,
        &norm_b_data,
        batch,
        time,
        hidden,
        eps as f32,
    );
    assert_close("ada_layer_norm_nativeop", &result, &expected, 1e-4);
}

/// [2, 8, 32] -> AdaLayerNorm: batched case with larger dimensions.
///
/// Verifies correct batch indexing (batch_idx = gid / time_steps) and
/// that gamma/beta are applied per-batch across all time steps.
#[test]
fn test_compiled_ada_layer_norm_nativeop_batched() {
    super::test_utils::gpu_init();
    let cache = super::test_utils::metal_setup();

    let (batch, time, hidden) = (2, 8, 32);
    let eps = 1e-5_f64;

    let x_data = super::test_utils::rand_f32_vec(0xAD1A_0010, batch * time * hidden, -2.0, 2.0);
    let gamma_data = super::test_utils::rand_f32_vec(0xAD1A_0011, batch * hidden, -0.5, 0.5);
    let beta_data = super::test_utils::rand_f32_vec(0xAD1A_0012, batch * hidden, -0.3, 0.3);
    let norm_w_data = super::test_utils::rand_f32_vec(0xAD1A_0013, hidden, 0.5, 1.5);
    let norm_b_data = super::test_utils::rand_f32_vec(0xAD1A_0014, hidden, -0.2, 0.2);

    let norm_weight = WeightRef::new(norm_w_data.clone(), vec![hidden]).expect("norm_weight");
    let norm_bias = WeightRef::new(norm_b_data.clone(), vec![hidden]).expect("norm_bias");

    let graph = ComputationGraph::from_nodes(vec![
        input_node(0, &[batch, time, hidden]),
        input_node(1, &[batch, 1, hidden]),
        input_node(2, &[batch, 1, hidden]),
        TraceNode::new(
            3,
            "ada_layer_norm_0".into(),
            TraceOp::KokoroFused(KokoroFusedOp::AdaLayerNorm {
                norm_weight,
                norm_bias,
                eps,
            }),
            vec![0, 1, 2],
            vec![batch, time, hidden],
            DType::F32,
        ),
    ]);

    let x_buf = create_input_buffer(&cache, &x_data);
    let gamma_buf = create_input_buffer(&cache, &gamma_data);
    let beta_buf = create_input_buffer(&cache, &beta_data);

    let result = compile_and_run(
        &cache,
        graph,
        &[&x_buf, &gamma_buf, &beta_buf],
        batch * time * hidden,
    );

    let expected = cpu_ada_layer_norm(
        &x_data,
        &gamma_data,
        &beta_data,
        &norm_w_data,
        &norm_b_data,
        batch,
        time,
        hidden,
        eps as f32,
    );
    assert_close("ada_layer_norm_nativeop_batched", &result, &expected, 1e-4);
}

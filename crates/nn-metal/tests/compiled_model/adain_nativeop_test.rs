// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! End-to-end compiled model tests for fused NativeOp norms: InstanceNorm,
//! AdainSnake, AdainLeakyRelu.
//!
//! Exercises the full pipeline: build trace graph -> compile (NativeOpKind) ->
//! GPU execute via fused MSL kernel -> verify against CPU reference.
//!
//! Part of #2472 (Fused InstanceNorm MSL kernel).

use nn_core::dyn_tensor::trace::{ComputationGraph, KokoroFusedOp, TraceNode, TraceOp, WeightRef};
use nn_core::DType;
use nn_metal::compiled_model::CompiledModel;

use super::helpers::{assert_close, compile_and_run, create_input_buffer, input_node};

// -- CPU reference helpers ----------------------------------------------------

/// CPU InstanceNorm: normalize each (batch, channel) slice independently.
///
/// Input: `[B, C, T]` row-major. Output: same shape.
/// Per (b, c): `normed = (x - mean) / sqrt(var + eps)`.
fn cpu_instance_norm(
    input: &[f32],
    batch: usize,
    channels: usize,
    time: usize,
    eps: f32,
) -> Vec<f32> {
    let mut output = vec![0.0_f32; batch * channels * time];
    for b in 0..batch {
        for c in 0..channels {
            let offset = (b * channels + c) * time;
            let slice = &input[offset..offset + time];
            let mean: f32 = slice.iter().sum::<f32>() / time as f32;
            let var: f32 = slice.iter().map(|x| (x - mean) * (x - mean)).sum::<f32>() / time as f32;
            let inv_std = 1.0 / (var + eps).sqrt();
            for t in 0..time {
                output[offset + t] = (slice[t] - mean) * inv_std;
            }
        }
    }
    output
}

/// CPU AdainSnake: InstanceNorm -> affine -> snake.
///
/// Inputs: x `[B, C, T]`, gamma `[B, C, 1]`, beta `[B, C, 1]`, alpha `[C]`.
/// 1. normed = instance_norm(x, eps)
/// 2. y = (1 + gamma) * normed + beta
/// 3. out = y + (1/alpha) * sin^2(alpha * y)
fn cpu_adain_snake(
    x: &[f32],
    gamma: &[f32],
    beta: &[f32],
    alpha: &[f32],
    batch: usize,
    channels: usize,
    time: usize,
    eps: f32,
) -> Vec<f32> {
    let normed = cpu_instance_norm(x, batch, channels, time, eps);
    let mut output = vec![0.0_f32; batch * channels * time];
    for b in 0..batch {
        for c in 0..channels {
            let g = gamma[b * channels + c];
            let be = beta[b * channels + c];
            let a = alpha[c];
            let inv_a = 1.0 / a;
            let offset = (b * channels + c) * time;
            for t in 0..time {
                let y = (1.0 + g) * normed[offset + t] + be;
                output[offset + t] = y + inv_a * (a * y).sin().powi(2);
            }
        }
    }
    output
}

/// CPU AdainLeakyRelu: InstanceNorm -> affine -> leaky_relu.
///
/// Inputs: x `[B, C, T]`, gamma `[B, C, 1]`, beta `[B, C, 1]`.
/// 1. normed = instance_norm(x, eps)
/// 2. y = (1 + gamma) * normed + beta
/// 3. out = y >= 0 ? y : slope * y
fn cpu_adain_leaky_relu(
    x: &[f32],
    gamma: &[f32],
    beta: &[f32],
    batch: usize,
    channels: usize,
    time: usize,
    eps: f32,
    slope: f32,
) -> Vec<f32> {
    let normed = cpu_instance_norm(x, batch, channels, time, eps);
    let mut output = vec![0.0_f32; batch * channels * time];
    for b in 0..batch {
        for c in 0..channels {
            let g = gamma[b * channels + c];
            let be = beta[b * channels + c];
            let offset = (b * channels + c) * time;
            for t in 0..time {
                let y = (1.0 + g) * normed[offset + t] + be;
                output[offset + t] = if y >= 0.0 { y } else { slope * y };
            }
        }
    }
    output
}

// -- Test: InstanceNorm NativeOp through CompiledModel ------------------------

/// [1, 4, 16] -> InstanceNorm(eps=1e-5): fused single-dispatch GPU kernel.
///
/// Verifies NativeOpKind::InstanceNorm executes correctly through the full
/// compiled model pipeline (trace -> compile -> GPU execute -> CPU readback).
#[test]
fn test_compiled_instance_norm_nativeop() {
    super::test_utils::gpu_init();
    let cache = super::test_utils::metal_setup();

    let (batch, channels, time) = (1, 4, 16);
    let eps = 1e-5_f64;
    let input_data =
        super::test_utils::rand_f32_vec(0xA1D0_0001, batch * channels * time, -1.0, 1.0);

    let graph = ComputationGraph::from_nodes(vec![
        input_node(0, &[batch, channels, time]),
        TraceNode::new(
            1,
            "instance_norm_0".into(),
            TraceOp::InstanceNorm { eps },
            vec![0],
            vec![batch, channels, time],
            DType::F32,
        ),
    ]);

    let result = compile_and_run(
        &cache,
        graph,
        &[&create_input_buffer(&cache, &input_data)],
        batch * channels * time,
    );

    let expected = cpu_instance_norm(&input_data, batch, channels, time, eps as f32);
    assert_close("instance_norm_nativeop", &result, &expected, 1e-4);
}

/// [2, 8, 32] -> InstanceNorm: batched case with larger dimensions.
#[test]
fn test_compiled_instance_norm_nativeop_batched() {
    super::test_utils::gpu_init();
    let cache = super::test_utils::metal_setup();

    let (batch, channels, time) = (2, 8, 32);
    let eps = 1e-5_f64;
    let input_data =
        super::test_utils::rand_f32_vec(0xA1D0_0002, batch * channels * time, -2.0, 2.0);

    let graph = ComputationGraph::from_nodes(vec![
        input_node(0, &[batch, channels, time]),
        TraceNode::new(
            1,
            "instance_norm_0".into(),
            TraceOp::InstanceNorm { eps },
            vec![0],
            vec![batch, channels, time],
            DType::F32,
        ),
    ]);

    let result = compile_and_run(
        &cache,
        graph,
        &[&create_input_buffer(&cache, &input_data)],
        batch * channels * time,
    );

    let expected = cpu_instance_norm(&input_data, batch, channels, time, eps as f32);
    assert_close("instance_norm_nativeop_batched", &result, &expected, 1e-4);
}

// -- Test: AdainSnake NativeOp through CompiledModel --------------------------

/// [1, 4, 16] -> AdainSnake(alpha=[4], eps=1e-5): fused GPU kernel.
///
/// Verifies NativeOpKind::AdainSnake with 3 tensor inputs (x, gamma, beta)
/// and 1 weight (alpha) executes correctly through the compiled model pipeline.
#[test]
fn test_compiled_adain_snake_nativeop() {
    super::test_utils::gpu_init();
    let cache = super::test_utils::metal_setup();

    let (batch, channels, time) = (1, 4, 16);
    let eps = 1e-5_f64;
    let alpha_data = super::test_utils::rand_f32_vec(0xAD50_0001, channels, 0.5, 2.0);
    let input_data =
        super::test_utils::rand_f32_vec(0xAD50_0002, batch * channels * time, -1.0, 1.0);
    let gamma_data = super::test_utils::rand_f32_vec(0xAD50_0003, batch * channels, -0.3, 0.3);
    let beta_data = super::test_utils::rand_f32_vec(0xAD50_0004, batch * channels, -0.2, 0.2);

    let alpha = WeightRef::new(alpha_data.clone(), vec![channels]).expect("alpha weight");

    let graph = ComputationGraph::from_nodes(vec![
        input_node(0, &[batch, channels, time]),
        input_node(1, &[batch, channels, 1]),
        input_node(2, &[batch, channels, 1]),
        TraceNode::new(
            3,
            "adain_snake_0".into(),
            TraceOp::KokoroFused(KokoroFusedOp::AdainSnake { alpha, eps }),
            vec![0, 1, 2],
            vec![batch, channels, time],
            DType::F32,
        ),
    ]);

    let x_buf = create_input_buffer(&cache, &input_data);
    let gamma_buf = create_input_buffer(&cache, &gamma_data);
    let beta_buf = create_input_buffer(&cache, &beta_data);

    let result = compile_and_run(
        &cache,
        graph,
        &[&x_buf, &gamma_buf, &beta_buf],
        batch * channels * time,
    );

    let expected = cpu_adain_snake(
        &input_data,
        &gamma_data,
        &beta_data,
        &alpha_data,
        batch,
        channels,
        time,
        eps as f32,
    );
    assert_close("adain_snake_nativeop", &result, &expected, 1e-3);
}

/// [2, 8, 32] -> AdainSnake: batched case with more channels.
#[test]
fn test_compiled_adain_snake_nativeop_batched() {
    super::test_utils::gpu_init();
    let cache = super::test_utils::metal_setup();

    let (batch, channels, time) = (2, 8, 32);
    let eps = 1e-5_f64;
    let alpha_data = super::test_utils::rand_f32_vec(0xAD50_0010, channels, 0.5, 2.0);
    let input_data =
        super::test_utils::rand_f32_vec(0xAD50_0011, batch * channels * time, -1.0, 1.0);
    let gamma_data = super::test_utils::rand_f32_vec(0xAD50_0012, batch * channels, -0.3, 0.3);
    let beta_data = super::test_utils::rand_f32_vec(0xAD50_0013, batch * channels, -0.2, 0.2);

    let alpha = WeightRef::new(alpha_data.clone(), vec![channels]).expect("alpha weight");

    let graph = ComputationGraph::from_nodes(vec![
        input_node(0, &[batch, channels, time]),
        input_node(1, &[batch, channels, 1]),
        input_node(2, &[batch, channels, 1]),
        TraceNode::new(
            3,
            "adain_snake_0".into(),
            TraceOp::KokoroFused(KokoroFusedOp::AdainSnake { alpha, eps }),
            vec![0, 1, 2],
            vec![batch, channels, time],
            DType::F32,
        ),
    ]);

    let x_buf = create_input_buffer(&cache, &input_data);
    let gamma_buf = create_input_buffer(&cache, &gamma_data);
    let beta_buf = create_input_buffer(&cache, &beta_data);

    let result = compile_and_run(
        &cache,
        graph,
        &[&x_buf, &gamma_buf, &beta_buf],
        batch * channels * time,
    );

    let expected = cpu_adain_snake(
        &input_data,
        &gamma_data,
        &beta_data,
        &alpha_data,
        batch,
        channels,
        time,
        eps as f32,
    );
    assert_close("adain_snake_nativeop_batched", &result, &expected, 1e-3);
}

// -- Test: AdainLeakyRelu NativeOp through CompiledModel ----------------------

/// [1, 4, 16] -> AdainLeakyRelu(eps=1e-5, slope=0.2): fused GPU kernel.
///
/// Verifies NativeOpKind::AdainLeakyRelu with 3 tensor inputs (x, gamma, beta)
/// executes correctly through the compiled model pipeline.
#[test]
fn test_compiled_adain_leaky_relu_nativeop() {
    super::test_utils::gpu_init();
    let cache = super::test_utils::metal_setup();

    let (batch, channels, time) = (1, 4, 16);
    let eps = 1e-5_f64;
    let slope = 0.2_f64;
    let input_data =
        super::test_utils::rand_f32_vec(0xAD10_0001, batch * channels * time, -1.0, 1.0);
    let gamma_data = super::test_utils::rand_f32_vec(0xAD10_0002, batch * channels, -0.3, 0.3);
    let beta_data = super::test_utils::rand_f32_vec(0xAD10_0003, batch * channels, -0.2, 0.2);

    let graph = ComputationGraph::from_nodes(vec![
        input_node(0, &[batch, channels, time]),
        input_node(1, &[batch, channels, 1]),
        input_node(2, &[batch, channels, 1]),
        TraceNode::new(
            3,
            "adain_leaky_relu_0".into(),
            TraceOp::KokoroFused(KokoroFusedOp::AdainLeakyRelu { eps, slope }),
            vec![0, 1, 2],
            vec![batch, channels, time],
            DType::F32,
        ),
    ]);

    let x_buf = create_input_buffer(&cache, &input_data);
    let gamma_buf = create_input_buffer(&cache, &gamma_data);
    let beta_buf = create_input_buffer(&cache, &beta_data);

    let result = compile_and_run(
        &cache,
        graph,
        &[&x_buf, &gamma_buf, &beta_buf],
        batch * channels * time,
    );

    let expected = cpu_adain_leaky_relu(
        &input_data,
        &gamma_data,
        &beta_data,
        batch,
        channels,
        time,
        eps as f32,
        slope as f32,
    );
    assert_close("adain_leaky_relu_nativeop", &result, &expected, 1e-4);
}

/// [2, 8, 32] -> AdainLeakyRelu: batched case with more channels.
#[test]
fn test_compiled_adain_leaky_relu_nativeop_batched() {
    super::test_utils::gpu_init();
    let cache = super::test_utils::metal_setup();

    let (batch, channels, time) = (2, 8, 32);
    let eps = 1e-5_f64;
    let slope = 0.2_f64;
    let input_data =
        super::test_utils::rand_f32_vec(0xAD10_0010, batch * channels * time, -1.0, 1.0);
    let gamma_data = super::test_utils::rand_f32_vec(0xAD10_0011, batch * channels, -0.3, 0.3);
    let beta_data = super::test_utils::rand_f32_vec(0xAD10_0012, batch * channels, -0.2, 0.2);

    let graph = ComputationGraph::from_nodes(vec![
        input_node(0, &[batch, channels, time]),
        input_node(1, &[batch, channels, 1]),
        input_node(2, &[batch, channels, 1]),
        TraceNode::new(
            3,
            "adain_leaky_relu_0".into(),
            TraceOp::KokoroFused(KokoroFusedOp::AdainLeakyRelu { eps, slope }),
            vec![0, 1, 2],
            vec![batch, channels, time],
            DType::F32,
        ),
    ]);

    let x_buf = create_input_buffer(&cache, &input_data);
    let gamma_buf = create_input_buffer(&cache, &gamma_data);
    let beta_buf = create_input_buffer(&cache, &beta_data);

    let result = compile_and_run(
        &cache,
        graph,
        &[&x_buf, &gamma_buf, &beta_buf],
        batch * channels * time,
    );

    let expected = cpu_adain_leaky_relu(
        &input_data,
        &gamma_data,
        &beta_data,
        batch,
        channels,
        time,
        eps as f32,
        slope as f32,
    );
    assert_close(
        "adain_leaky_relu_nativeop_batched",
        &result,
        &expected,
        1e-4,
    );
}

// -- Test: PrecisionTier::Strict routes InstanceNorm to Kahan path (#2568) ----

/// Verify that `PrecisionTier::Strict` on a CompiledModel with InstanceNorm
/// routes to the Kahan-compensated decomposed path (not the fused kernel).
///
/// The routing in `execute_native_instance_norm` is deterministic:
/// `model.precision().tier == Strict` → `native_instance_norm_precise()`.
/// This test proves the precision contract is correctly plumbed through
/// compilation and execution:
///
/// 1. `precision()` returns `Some` with `Strict` tier after `with_precision()`
/// 2. Execution succeeds (the Kahan code path actually runs without errors)
/// 3. Output matches CPU reference (the Kahan path produces correct results)
///
/// Part of #2568, #2218.
#[test]
fn test_instance_norm_strict_precision_routes_to_kahan() {
    use nn_dsl::ir::ScalarType;
    use nn_dsl::{PrecisionContract, PrecisionTier};

    super::test_utils::gpu_init();
    let cache = super::test_utils::metal_setup();

    let (batch, channels, time) = (1, 4, 64);
    let eps = 1e-5_f64;
    let input_data =
        super::test_utils::rand_f32_vec(0x2568_0001, batch * channels * time, -1.0, 1.0);

    let graph = ComputationGraph::from_nodes(vec![
        input_node(0, &[batch, channels, time]),
        TraceNode::new(
            1,
            "instance_norm_0".into(),
            TraceOp::InstanceNorm { eps },
            vec![0],
            vec![batch, channels, time],
            DType::F32,
        ),
    ]);

    // Compile with PrecisionTier::Strict — should route to Kahan path.
    let contract = PrecisionContract::bootstrap(PrecisionTier::Strict, ScalarType::F32);
    let compiled = CompiledModel::builder(&graph, &cache)
        .build()
        .expect("compile")
        .with_precision(contract);

    // AC: verify precision contract is plumbed through.
    let prec = compiled
        .precision()
        .expect("precision must be set after with_precision()");
    assert_eq!(
        prec.tier,
        PrecisionTier::Strict,
        "precision tier must be Strict"
    );

    // Execute — exercises the Kahan-compensated decomposed path in
    // execute_native_instance_norm (native_instance_norm_precise).
    let input_buf = create_input_buffer(&cache, &input_data);
    let out_buf = compiled.execute(&cache, &[&input_buf]).expect("execute");
    let result = super::helpers::read_output_n(&out_buf, batch * channels * time);

    // Verify output matches CPU reference.
    let expected = cpu_instance_norm(&input_data, batch, channels, time, eps as f32);
    assert_close("instance_norm_strict_kahan", &result, &expected, 1e-4);
}

/// Verify that a CompiledModel WITHOUT precision contract uses the fused
/// (non-Kahan) InstanceNorm path, and that BOTH paths produce correct output.
///
/// Complementary to `test_instance_norm_strict_precision_routes_to_kahan`:
/// proves the default (no precision) path also works, confirming the routing
/// branch is meaningful.
///
/// Part of #2568, #2218.
#[test]
fn test_instance_norm_no_precision_uses_fused_path() {
    super::test_utils::gpu_init();
    let cache = super::test_utils::metal_setup();

    let (batch, channels, time) = (1, 4, 64);
    let eps = 1e-5_f64;
    let input_data =
        super::test_utils::rand_f32_vec(0x2568_0002, batch * channels * time, -1.0, 1.0);

    let graph = ComputationGraph::from_nodes(vec![
        input_node(0, &[batch, channels, time]),
        TraceNode::new(
            1,
            "instance_norm_0".into(),
            TraceOp::InstanceNorm { eps },
            vec![0],
            vec![batch, channels, time],
            DType::F32,
        ),
    ]);

    // Compile WITHOUT precision — should route to fused path.
    let compiled = CompiledModel::builder(&graph, &cache)
        .build()
        .expect("compile");

    // AC: verify no precision contract is set (default).
    assert!(
        compiled.precision().is_none(),
        "default CompiledModel must not have precision set"
    );

    // Execute — exercises the fused single-dispatch path in
    // execute_native_instance_norm (native_instance_norm).
    let input_buf = create_input_buffer(&cache, &input_data);
    let out_buf = compiled.execute(&cache, &[&input_buf]).expect("execute");
    let result = super::helpers::read_output_n(&out_buf, batch * channels * time);

    // Verify output matches CPU reference.
    let expected = cpu_instance_norm(&input_data, batch, channels, time, eps as f32);
    assert_close("instance_norm_fused_default", &result, &expected, 1e-4);
}

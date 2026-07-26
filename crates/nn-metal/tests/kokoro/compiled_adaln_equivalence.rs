// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Fused vs decomposed equivalence tests for AdaLayerNorm NativeOp.
//!
//! Verifies that the fused `NativeOpKind::AdaLayerNorm` GPU kernel produces
//! identical output to the CPU reference path (within tolerance), and that
//! the trace compiler emits a single NativeOp instead of ~6-7 IR dispatches.
//!
//! AC from #2482:
//! - GPU parity test (fused vs decomposed)
//! - Dispatch count verification on AdaLayerNorm trace
//!
//! Part of #2218 (Kokoro epic).

use super::helpers;

use std::collections::HashMap;

use nn_core::dyn_tensor::trace::{ComputationGraph, KokoroFusedOp, TraceNode, TraceOp, WeightRef};
use nn_core::dyn_tensor::DynTensor;
use nn_core::{DType, Device, VarBuilder};
use nn_metal::compiled_model::CompiledModel;
use nn_models::kokoro_error::KokoroError;
use nn_models::kokoro_tts::AdaLayerNorm;

use helpers::{assert_close, compile_and_run, create_input_buffer, input_node};

// -- CPU reference ------------------------------------------------------------

/// CPU reference for AdaLayerNorm: `(1 + gamma) * LayerNorm(x, w, b, eps) + beta`.
///
/// x: `[B, T, C]`, gamma/beta: `[B, 1, C]` (broadcast over T).
/// norm_weight/norm_bias: `[C]` (LayerNorm learnable params).
fn cpu_ada_layer_norm(
    x: &[f32],
    gamma: &[f32],
    beta: &[f32],
    norm_weight: &[f32],
    norm_bias: &[f32],
    batch: usize,
    time: usize,
    channels: usize,
    eps: f32,
) -> Vec<f32> {
    let mut output = vec![0.0_f32; batch * time * channels];
    for b in 0..batch {
        for t in 0..time {
            let row_offset = (b * time + t) * channels;
            let row = &x[row_offset..row_offset + channels];
            let mean: f32 = row.iter().sum::<f32>() / channels as f32;
            let var: f32 =
                row.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / channels as f32;
            let inv_std = 1.0 / (var + eps).sqrt();
            let gb_offset = b * channels;
            for c in 0..channels {
                let normed = (row[c] - mean) * inv_std;
                let affine_normed = normed * norm_weight[c] + norm_bias[c];
                output[row_offset + c] =
                    (1.0 + gamma[gb_offset + c]) * affine_normed + beta[gb_offset + c];
            }
        }
    }
    output
}

// -- Shared test harness ------------------------------------------------------

/// Test parameters for a single AdaLayerNorm GPU parity test.
struct AdalnTestCase {
    batch: usize,
    time: usize,
    channels: usize,
    seed_base: u64,
    x_range: (f32, f32),
    gamma_range: (f32, f32),
    beta_range: (f32, f32),
    nw_range: (f32, f32),
    nb_range: (f32, f32),
}

/// Build a trace graph, compile, execute on GPU, and compare with CPU reference.
fn run_adaln_parity(label: &str, tc: &AdalnTestCase) {
    super::test_utils::gpu_init();
    let cache = super::test_utils::metal_setup();

    let (b, t, c) = (tc.batch, tc.time, tc.channels);
    let eps = 1e-5_f64;

    let x_data =
        super::test_utils::rand_f32_vec(tc.seed_base, b * t * c, tc.x_range.0, tc.x_range.1);
    let gamma_data = super::test_utils::rand_f32_vec(
        tc.seed_base + 1,
        b * c,
        tc.gamma_range.0,
        tc.gamma_range.1,
    );
    let beta_data =
        super::test_utils::rand_f32_vec(tc.seed_base + 2, b * c, tc.beta_range.0, tc.beta_range.1);
    let nw_data =
        super::test_utils::rand_f32_vec(tc.seed_base + 3, c, tc.nw_range.0, tc.nw_range.1);
    let nb_data =
        super::test_utils::rand_f32_vec(tc.seed_base + 4, c, tc.nb_range.0, tc.nb_range.1);

    let norm_weight = WeightRef::new(nw_data.clone(), vec![c]).expect("norm_weight");
    let norm_bias = WeightRef::new(nb_data.clone(), vec![c]).expect("norm_bias");

    let graph = ComputationGraph::from_nodes(vec![
        input_node(0, &[b, t, c]),
        input_node(1, &[b, 1, c]),
        input_node(2, &[b, 1, c]),
        TraceNode::new(
            3,
            "ada_layer_norm_0".into(),
            TraceOp::KokoroFused(KokoroFusedOp::AdaLayerNorm {
                norm_weight,
                norm_bias,
                eps,
            }),
            vec![0, 1, 2],
            vec![b, t, c],
            DType::F32,
        ),
    ]);

    let x_buf = create_input_buffer(&cache, &x_data);
    let gamma_buf = create_input_buffer(&cache, &gamma_data);
    let beta_buf = create_input_buffer(&cache, &beta_data);

    let result = compile_and_run(&cache, graph, &[&x_buf, &gamma_buf, &beta_buf], b * t * c);
    let expected = cpu_ada_layer_norm(
        &x_data,
        &gamma_data,
        &beta_data,
        &nw_data,
        &nb_data,
        b,
        t,
        c,
        eps as f32,
    );
    assert_close(label, &result, &expected, 1e-4);
}

// -- Test: AdaLayerNorm NativeOp via manual trace graph -----------------------

/// [1, 8, 16] -> AdaLayerNorm(eps=1e-5): fused single-dispatch GPU kernel.
#[test]
fn test_compiled_adaln_nativeop() {
    run_adaln_parity(
        "adaln_nativeop",
        &AdalnTestCase {
            batch: 1,
            time: 8,
            channels: 16,
            seed_base: 0xADA0_0001,
            x_range: (-1.0, 1.0),
            gamma_range: (-0.3, 0.3),
            beta_range: (-0.2, 0.2),
            nw_range: (0.8, 1.2),
            nb_range: (-0.1, 0.1),
        },
    );
}

/// [2, 16, 32] -> AdaLayerNorm: batched case, batch-index verification.
#[test]
fn test_compiled_adaln_nativeop_batched() {
    run_adaln_parity(
        "adaln_nativeop_batched",
        &AdalnTestCase {
            batch: 2,
            time: 16,
            channels: 32,
            seed_base: 0xADA0_0010,
            x_range: (-2.0, 2.0),
            gamma_range: (-0.5, 0.5),
            beta_range: (-0.3, 0.3),
            nw_range: (0.8, 1.2),
            nb_range: (-0.1, 0.1),
        },
    );
}

/// [1, 4, 512] -> AdaLayerNorm: hidden_dim > threadgroup size (256).
#[test]
fn test_compiled_adaln_nativeop_large_hidden() {
    run_adaln_parity(
        "adaln_nativeop_large_hidden",
        &AdalnTestCase {
            batch: 1,
            time: 4,
            channels: 512,
            seed_base: 0xADA0_0020,
            x_range: (-1.0, 1.0),
            gamma_range: (-0.3, 0.3),
            beta_range: (-0.2, 0.2),
            nw_range: (0.9, 1.1),
            nb_range: (-0.05, 0.05),
        },
    );
}

// -- Test: Dispatch count via compile_forward ---------------------------------

/// Build an AdaLayerNorm module with deterministic weights.
fn build_adaln_module(channels: usize, style_dim: usize) -> AdaLayerNorm {
    let cpu = Device::Cpu;
    let fill = |shape: &[usize]| -> DynTensor {
        let n: usize = shape.iter().product();
        let data: Vec<f32> = (0..n).map(|i| 0.01 * (i as f32 + 1.0)).collect();
        DynTensor::from_vec(data, shape, &cpu).unwrap()
    };
    let mut m = HashMap::new();
    m.insert("norm.weight".into(), fill(&[channels]));
    m.insert("norm.bias".into(), fill(&[channels]));
    m.insert("fc.weight".into(), fill(&[2 * channels, style_dim]));
    m.insert("fc.bias".into(), fill(&[2 * channels]));
    let vb = VarBuilder::from_tensors(m, DType::F32, &cpu);
    AdaLayerNorm::load(&vb, channels, style_dim).expect("load AdaLayerNorm")
}

/// AdaLayerNorm traced via `compile_forward` produces exactly 1 NativeOp.
///
/// Verifies dispatch count reduction from #2482 and GPU-vs-CPU parity.
#[test]
#[allow(deprecated)]
fn test_adaln_compile_forward_dispatch_count() {
    super::test_utils::gpu_init();
    let cache = super::test_utils::metal_setup();

    let (channels, style_dim) = (16, 8);
    let adaln = build_adaln_module(channels, style_dim);

    let cpu = Device::Cpu;
    let fill = |shape: &[usize]| -> DynTensor {
        let n: usize = shape.iter().product();
        let data: Vec<f32> = (0..n).map(|i| 0.01 * (i as f32 + 1.0)).collect();
        DynTensor::from_vec(data, shape, &cpu).unwrap()
    };
    let x = fill(&[1, 8, channels]);
    let style = fill(&[1, style_dim]);

    let compiled = CompiledModel::compile_forward(
        &[&x, &style],
        |inputs| {
            adaln
                .forward(&inputs[0], &inputs[1])
                .map_err(KokoroError::into_tensor_error)
        },
        &cache,
    )
    .expect("compile_forward");

    let native_ops = compiled.num_native_ops();
    assert_eq!(
        native_ops, 1,
        "AdaLayerNorm should compile to 1 NativeOp, got {native_ops}"
    );

    eprintln!(
        "AdaLayerNorm: {} total dispatches ({native_ops} native + {} IR)",
        compiled.num_dispatches(),
        compiled.num_ir_dispatches()
    );

    // Verify GPU output matches CPU reference.
    let cpu_out = adaln.forward(&x, &style).expect("CPU forward");
    let cpu_vals = cpu_out.to_flat_vec::<f32>().unwrap();

    let x_gpu = x.to_device(&Device::metal()).unwrap();
    let style_gpu = style.to_device(&Device::metal()).unwrap();
    let gpu_out = compiled
        .execute_dyn(&cache, &[&x_gpu, &style_gpu])
        .expect("GPU execute");
    let gpu_vals = gpu_out
        .to_device(&cpu)
        .unwrap()
        .to_flat_vec::<f32>()
        .unwrap();

    assert_eq!(cpu_vals.len(), gpu_vals.len(), "output length mismatch");
    let max_diff = cpu_vals
        .iter()
        .zip(gpu_vals.iter())
        .map(|(c, g)| (c - g).abs())
        .fold(0.0_f32, f32::max);
    eprintln!(
        "AdaLayerNorm parity: {} elems, max_diff={max_diff:.6e}",
        cpu_vals.len()
    );
    assert!(
        max_diff < 1e-3,
        "Fused AdaLayerNorm max_diff={max_diff:.6e} > 1e-3"
    );
}

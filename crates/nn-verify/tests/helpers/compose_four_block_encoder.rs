// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Integration tests: 4-block Demucs encoder composition.
//!
//! Validates that a full Demucs encoder stack (4 Conv1d + Snake + InstanceNorm
//! blocks) translates through `tensor_kernel_to_graph` and produces a single
//! NY `GraphNetwork` where IBP and CROWN bounds propagate end-to-end.
//!
//! Extends `compose_tensor_chain_two_layer.rs` to the full 4-block depth
//! matching dvoice's Demucs encoder: 1→48→96→192→384 channels.
//!
//! Part of #684 AC1.

use super::common::{assert_bounds_valid, assert_crown_tighter_when_not_fallback, conv1d_out_len};
use nn_dsl::adain::build_snake_scalar_kernel;
use nn_dsl::tensor_block_builder::TensorBlockBuilder;
use nn_verify::{tensor_kernel_to_graph, BoundedTensor, TensorParamBinding};
use ndarray::{ArrayD, IxDyn};

// ---------------------------------------------------------------------------
// 4-block Demucs encoder builder
// ---------------------------------------------------------------------------

/// Channel widths per block. Block i maps channels[i] → channels[i+1].
struct EncoderConfig {
    channels: Vec<usize>,
    kernel_size: usize,
    stride: usize,
    padding: usize,
    in_length: usize,
}

/// Build a 4-block Demucs encoder using TensorBlockBuilder.
///
/// Each block: Conv1d(ch[i]→ch[i+1], k, stride, pad) → Snake → InstanceNorm
/// All blocks share alpha and eps parameters (typical in Demucs).
///
/// Returns (TensorKernelDef, per-block output shapes).
fn build_four_block_encoder(
    cfg: &EncoderConfig,
) -> (nn_dsl::tensor_ir::TensorKernelDef, Vec<Vec<usize>>) {
    assert_eq!(cfg.channels.len(), 5, "need 5 channel widths for 4 blocks");

    // Pre-compute output lengths.
    let mut lengths = Vec::with_capacity(5);
    lengths.push(cfg.in_length);
    for _ in 0..4 {
        let prev = *lengths.last().unwrap();
        lengths.push(conv1d_out_len(
            prev,
            cfg.kernel_size,
            cfg.stride,
            cfg.padding,
        ));
    }

    let mut b = TensorBlockBuilder::new("demucs_4block_encoder");

    // Inputs: data + 4 conv weights + alpha + eps
    let data = b.add_input("data", &[cfg.channels[0], cfg.in_length]);
    let weights: Vec<_> = (0..4)
        .map(|i| {
            b.add_input(
                &format!("weight{}", i + 1),
                &[cfg.channels[i + 1], cfg.channels[i], cfg.kernel_size],
            )
        })
        .collect();
    let alpha = b.add_input("alpha", &[1]);
    let eps = b.add_input("eps", &[1]);

    let mut prev_output = data;
    let mut block_shapes = Vec::with_capacity(4);

    for i in 0..4 {
        let out_ch = cfg.channels[i + 1];
        let out_len = lengths[i + 1];
        let out_shape = [out_ch, out_len];
        block_shapes.push(out_shape.to_vec());

        let snake = build_snake_scalar_kernel().expect("snake kernel");

        let conv = b.add_conv1d(
            prev_output,
            weights[i],
            None,
            cfg.stride,
            cfg.padding,
            &out_shape,
        );
        let alpha_bc = b.add_broadcast(alpha, &out_shape);
        let act = b.add_elementwise(snake, &[conv, alpha_bc], &out_shape);
        let norm = b.add_instance_norm(act, eps, 1, None, None, &out_shape);

        prev_output = norm;
    }

    let def = b.build(prev_output).expect("valid graph");
    (def, block_shapes)
}

// ---------------------------------------------------------------------------
// Small-scale tests (fast)
// ---------------------------------------------------------------------------

/// Helper: small config with channels [1, 2, 4, 8, 16], stride=1.
fn small_config() -> EncoderConfig {
    EncoderConfig {
        channels: vec![1, 2, 4, 8, 16],
        kernel_size: 3,
        stride: 1,
        padding: 1,
        in_length: 16,
    }
}

/// 4-block encoder graph builds and translates.
#[test]
fn test_four_block_encoder_graph_builds() {
    let cfg = small_config();
    let (def, block_shapes) = build_four_block_encoder(&cfg);

    // Verify output shape of final block.
    assert_eq!(def.nodes.last().unwrap().shape, block_shapes[3]);

    let bindings = small_bindings(&cfg);
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("4-block encoder graph");
    assert!(
        graph.num_nodes() >= 12,
        "4-block graph should have >= 12 nodes (3 per block × 4)"
    );
}

/// IBP bounds propagate through all 4 blocks.
#[test]
fn test_four_block_encoder_ibp_propagates() {
    let cfg = small_config();
    let (def, block_shapes) = build_four_block_encoder(&cfg);

    let bindings = small_bindings(&cfg);
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");

    let lower = ArrayD::from_elem(IxDyn(&[cfg.channels[0], cfg.in_length]), -1.0f32);
    let upper = ArrayD::from_elem(IxDyn(&[cfg.channels[0], cfg.in_length]), 1.0f32);
    let input = BoundedTensor::new(lower, upper).expect("bounds");

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through 4-block encoder");
    let (lo, _hi) = output.lower_upper();

    assert_eq!(lo.shape(), block_shapes[3].as_slice());
    assert_bounds_valid(&output);
}

/// CROWN propagation through 4-block encoder.
///
/// Uses `assert_crown_tighter_when_not_fallback` to verify CROWN produces
/// tighter bounds than IBP when CROWN succeeds.
#[test]
fn test_four_block_encoder_crown_propagates() {
    let cfg = small_config();
    let (def, block_shapes) = build_four_block_encoder(&cfg);

    let bindings = small_bindings(&cfg);
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");

    let lower = ArrayD::from_elem(IxDyn(&[cfg.channels[0], cfg.in_length]), -1.0f32);
    let upper = ArrayD::from_elem(IxDyn(&[cfg.channels[0], cfg.in_length]), 1.0f32);
    let input = BoundedTensor::new(lower, upper).expect("bounds");

    let (method, output, fallback_reason) = assert_crown_tighter_when_not_fallback(&graph, &input);
    let (lo, _) = output.lower_upper();

    assert_eq!(lo.shape(), block_shapes[3].as_slice());
    assert_bounds_valid(&output);

    eprintln!("4-block encoder (small): method={method:?}");
    if let Some(reason) = &fallback_reason {
        eprintln!("CROWN fallback reason: {reason}");
    }
}

// ---------------------------------------------------------------------------
// Dvoice-scale tests (AC1 + AC4)
// ---------------------------------------------------------------------------

/// Dvoice-scale encoder config: 1→48→96→192→384, k=8, stride=4, pad=2.
///
/// Shape progression (in_length=1024):
///   Block 1: [1, 1024] → [48, 256]
///   Block 2: [48, 256] → [96, 64]
///   Block 3: [96, 64]  → [192, 16]
///   Block 4: [192, 16] → [384, 4]
fn dvoice_config() -> EncoderConfig {
    EncoderConfig {
        channels: vec![1, 48, 96, 192, 384],
        kernel_size: 8,
        stride: 4,
        padding: 2,
        in_length: 1024,
    }
}

/// Dvoice-scale 4-block encoder: IBP bounds propagate end-to-end.
#[test]
fn test_four_block_encoder_dvoice_ibp() {
    let cfg = dvoice_config();
    let (def, block_shapes) = build_four_block_encoder(&cfg);

    let bindings = dvoice_bindings(&cfg);
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("dvoice 4-block graph");

    let lower = ArrayD::from_elem(IxDyn(&[1, 1024]), -1.0f32);
    let upper = ArrayD::from_elem(IxDyn(&[1, 1024]), 1.0f32);
    let input = BoundedTensor::new(lower, upper).expect("bounds");

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through dvoice 4-block encoder");
    let (lo, _hi) = output.lower_upper();

    // Final output: [384, 4]
    assert_eq!(lo.shape(), &[384, 4]);
    assert_eq!(lo.shape(), block_shapes[3].as_slice());

    assert_bounds_valid(&output);
}

/// Dvoice-scale CROWN propagation through full 4-block encoder.
///
/// Uses `assert_crown_tighter_when_not_fallback` to verify CROWN produces
/// tighter bounds than IBP when CROWN succeeds.
#[test]
fn test_four_block_encoder_dvoice_crown() {
    let cfg = dvoice_config();
    let (def, block_shapes) = build_four_block_encoder(&cfg);

    let bindings = dvoice_bindings(&cfg);
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("dvoice 4-block graph");

    let lower = ArrayD::from_elem(IxDyn(&[1, 1024]), -1.0f32);
    let upper = ArrayD::from_elem(IxDyn(&[1, 1024]), 1.0f32);
    let input = BoundedTensor::new(lower, upper).expect("bounds");

    let (method, output, fallback_reason) = assert_crown_tighter_when_not_fallback(&graph, &input);
    let (lo, _) = output.lower_upper();

    assert_eq!(lo.shape(), &[384, 4]);
    assert_eq!(lo.shape(), block_shapes[3].as_slice());

    assert_bounds_valid(&output);

    eprintln!("4-block encoder (dvoice): method={method:?}");
    if let Some(reason) = &fallback_reason {
        eprintln!("CROWN fallback reason: {reason}");
    }
}

// ---------------------------------------------------------------------------
// Binding constructors
// ---------------------------------------------------------------------------

/// Small-scale bindings with small uniform weights.
fn small_bindings(cfg: &EncoderConfig) -> Vec<TensorParamBinding> {
    let mut bindings = vec![TensorParamBinding::Variable]; // data
    for i in 0..4 {
        let w = ArrayD::from_elem(
            IxDyn(&[cfg.channels[i + 1], cfg.channels[i], cfg.kernel_size]),
            0.1f32,
        );
        bindings.push(TensorParamBinding::ConstantTensor(w));
    }
    bindings.push(TensorParamBinding::ConstantScalar(1.0)); // alpha
    bindings.push(TensorParamBinding::ConstantScalar(1e-5)); // eps
    bindings
}

/// Dvoice-scale bindings with small weights (to keep bounds reasonable).
fn dvoice_bindings(cfg: &EncoderConfig) -> Vec<TensorParamBinding> {
    let mut bindings = vec![TensorParamBinding::Variable]; // data
    for i in 0..4 {
        // Use smaller weights for deeper layers to reduce bound widening.
        let scale = 0.01 / ((i + 1) as f32).sqrt();
        let w = ArrayD::from_elem(
            IxDyn(&[cfg.channels[i + 1], cfg.channels[i], cfg.kernel_size]),
            scale,
        );
        bindings.push(TensorParamBinding::ConstantTensor(w));
    }
    bindings.push(TensorParamBinding::ConstantScalar(1.0)); // alpha
    bindings.push(TensorParamBinding::ConstantScalar(1e-5)); // eps
    bindings
}

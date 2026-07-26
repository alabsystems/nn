// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Integration tests: 4-block Demucs decoder composition with skip connections.
//!
//! Validates that a full Demucs decoder stack (4 blocks of BinaryAdd(skip) →
//! ConvTranspose1d → center_trim → GLU → GELU) translates through
//! `tensor_kernel_to_graph` and produces a single NY `GraphNetwork`
//! where IBP bounds propagate end-to-end.
//!
//! Extends `compose_decoder_conv_transpose.rs` to the full 4-block depth
//! with multi-variable skip connection inputs from each encoder block.
//!
//! Part of #684 AC2.

use super::common::{assert_bounds_valid, assert_crown_tighter_than_ibp, conv_transpose_out_len};
use nn_dsl::tensor_block_builder::TensorBlockBuilder;
use nn_verify::{
    propagate_with_crown_fallback, tensor_kernel_to_graph, BoundedTensor, TensorParamBinding,
};
use ndarray::{ArrayD, IxDyn};

// ---------------------------------------------------------------------------
// 4-block Demucs decoder builder
// ---------------------------------------------------------------------------

/// Decoder configuration mirroring the encoder.
struct DecoderConfig {
    /// Channel widths: [bottleneck, dec1_out, dec2_out, dec3_out, dec4_out].
    /// Each decoder block reduces channels by ~half via GLU split.
    channels: Vec<usize>,
    /// Time lengths at each decoder stage input.
    time_lengths: Vec<usize>,
    kernel_size: usize,
    stride: usize,
    padding: usize,
    trim: usize,
}

/// Build a 4-block Demucs decoder using TensorBlockBuilder.
///
/// Each block: BinaryAdd(skip) → ConvTranspose1d → center_trim → GLU → GELU.
///
/// Variable inputs: decoder_data + 4 skip connections (5 total).
/// The skip connections come from the corresponding encoder blocks in reverse
/// order (skip_i connects to decoder block i).
///
/// Returns (TensorKernelDef, per-block output shapes, num_variables=5).
fn build_four_block_decoder(
    cfg: &DecoderConfig,
) -> (nn_dsl::tensor_ir::TensorKernelDef, Vec<Vec<usize>>) {
    assert_eq!(cfg.channels.len(), 5, "need 5 channel widths for 4 blocks");
    assert_eq!(
        cfg.time_lengths.len(),
        5,
        "need 5 time lengths for 4 blocks"
    );

    let mut b = TensorBlockBuilder::new("demucs_4block_decoder");

    // Variable inputs: decoder data (from bottleneck) + 4 skip connections.
    let data = b.add_input("data", &[cfg.channels[0], cfg.time_lengths[0]]);
    let skips: Vec<_> = (0..4)
        .map(|i| {
            b.add_input(
                &format!("skip{}", i + 1),
                &[cfg.channels[i], cfg.time_lengths[i]],
            )
        })
        .collect();

    // Constant inputs: 4 ConvTranspose1d weights.
    let weights: Vec<_> = (0..4)
        .map(|i| {
            // ConvTranspose1d weight shape: [in_ch, out_ch, kernel_size]
            // where out_ch = 2 * cfg.channels[i+1] (doubled for GLU split)
            let doubled_out = cfg.channels[i + 1] * 2;
            b.add_input(
                &format!("weight{}", i + 1),
                &[cfg.channels[i], doubled_out, cfg.kernel_size],
            )
        })
        .collect();

    let mut prev_output = data;
    let mut block_shapes = Vec::with_capacity(4);

    for i in 0..4 {
        let c_in = cfg.channels[i];
        let c_out = cfg.channels[i + 1];
        let t_in = cfg.time_lengths[i];
        let doubled_out = c_out * 2;

        let t_up = conv_transpose_out_len(t_in, cfg.stride, cfg.kernel_size, cfg.padding);
        let t_trimmed = t_up - 2 * cfg.trim;

        // Step 1: Skip connection add
        let added = b.add_binary_add(prev_output, skips[i], &[c_in, t_in]);

        // Step 2: ConvTranspose1d (upsample)
        let deconv = b.add_conv_transpose_1d(
            added,
            weights[i],
            None,
            cfg.stride,
            cfg.padding,
            1, // dilation
            1, // groups
            0, // output_padding
            &[doubled_out, t_up],
        );

        // Step 3: Center trim (Narrow on time axis)
        let trimmed = b.add_narrow(deconv, 1, cfg.trim, t_trimmed, &[doubled_out, t_trimmed]);

        // Step 4: GLU (Narrow + Sigmoid + BinaryMul on channel axis)
        let glu_out = b
            .add_glu(trimmed, 0, &[doubled_out, t_trimmed])
            .expect("even dim");

        // Step 5: GELU activation
        let gelu_out = b.add_gelu(glu_out, &[c_out, t_trimmed]);

        block_shapes.push(vec![c_out, t_trimmed]);
        prev_output = gelu_out;
    }

    let def = b.build(prev_output).expect("valid graph");
    (def, block_shapes)
}

/// Create BoundedTensor input for N variable inputs stacked on axis 0.
///
/// For the 4-block decoder, we have 5 variables with different shapes.
/// Since NY multi-variable inputs require uniform shape (stacked on
/// axis 0), we need all variable inputs to have the same [C, T] shape.
///
/// For the uniform-shape decoder test, all skip connections share the
/// bottleneck shape (simplification for testing graph construction).
fn multi_var_uniform_input(
    num_vars: usize,
    channels: usize,
    length: usize,
    lo: f32,
    hi: f32,
) -> BoundedTensor {
    let lower = ArrayD::from_elem(IxDyn(&[num_vars, channels, length]), lo);
    let upper = ArrayD::from_elem(IxDyn(&[num_vars, channels, length]), hi);
    BoundedTensor::new(lower, upper).expect("valid bounds")
}

// ---------------------------------------------------------------------------
// Small-scale tests
// ---------------------------------------------------------------------------

/// Small decoder config: uniform shapes for testing graph construction.
/// All blocks use the same channel/time dimensions (simplification).
fn small_uniform_decoder_config() -> DecoderConfig {
    // Use uniform shapes: all blocks are [16, 8] so skip connections match.
    // This tests the graph topology (5-variable wiring, 4-block chaining)
    // without requiring shape-varying multi-variable inputs.
    DecoderConfig {
        channels: vec![16, 16, 16, 16, 16],
        time_lengths: vec![8, 8, 8, 8, 8],
        kernel_size: 3,
        stride: 1,
        padding: 1,
        trim: 0,
    }
}

/// 4-block decoder graph builds with 5 variable inputs.
#[test]
fn test_four_block_decoder_graph_builds() {
    let cfg = small_uniform_decoder_config();
    let (def, _) = build_four_block_decoder(&cfg);

    def.validate().expect("4-block decoder should validate");

    let mut bindings: Vec<TensorParamBinding> = Vec::new();
    // 5 variable inputs: data + 4 skips
    for _ in 0..5 {
        bindings.push(TensorParamBinding::Variable);
    }
    // 4 constant weights
    for i in 0..4 {
        let doubled_out = cfg.channels[i + 1] * 2;
        let w = ArrayD::from_elem(
            IxDyn(&[cfg.channels[i], doubled_out, cfg.kernel_size]),
            0.1f32,
        );
        bindings.push(TensorParamBinding::ConstantTensor(w));
    }

    let graph = tensor_kernel_to_graph(&def, &bindings).expect("4-block decoder graph");
    assert!(
        graph.num_nodes() >= 20,
        "4-block decoder graph should have >= 20 nodes (5 per block × 4)"
    );
}

/// IBP bounds propagate through all 4 decoder blocks with skip connections.
#[test]
fn test_four_block_decoder_ibp_propagates() {
    let cfg = small_uniform_decoder_config();
    let (def, block_shapes) = build_four_block_decoder(&cfg);

    let mut bindings: Vec<TensorParamBinding> = Vec::new();
    for _ in 0..5 {
        bindings.push(TensorParamBinding::Variable);
    }
    for i in 0..4 {
        let doubled_out = cfg.channels[i + 1] * 2;
        let w = ArrayD::from_elem(
            IxDyn(&[cfg.channels[i], doubled_out, cfg.kernel_size]),
            0.1f32,
        );
        bindings.push(TensorParamBinding::ConstantTensor(w));
    }

    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");

    // 5 variables, each [16, 8], stacked → [5, 16, 8]
    let input = multi_var_uniform_input(5, 16, 8, -1.0, 1.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through 4-block decoder");
    let (lo, _hi) = output.lower_upper();

    // Each variable enters its subgraph at its TRUE declared rank (#358 flat
    // per-variable Slice+Reshape harness), so the output keeps its natural
    // [C, T] shape with no leading stacking axis.
    assert_eq!(lo.shape(), block_shapes[3].as_slice());
    assert_bounds_valid(&output);
}

// ---------------------------------------------------------------------------
// Dvoice-scale decoder (uniform channels at production width)
// ---------------------------------------------------------------------------

/// Dvoice-scale decoder config.
///
/// Uses uniform channels (96) at dvoice-relevant width. The real Demucs
/// decoder has varying channels (384→192→96→48→24), but multi-variable
/// stacking requires all Variable inputs to share a shape. Uniform channels
/// test the full 4-block topology at production channel width.
///
/// Time progression (stride=1, pad=1, k=3, no trim):
///   All blocks: [96, 4] → ConvTranspose1d → [192, 4] → GLU → [96, 4]
fn dvoice_decoder_config() -> DecoderConfig {
    DecoderConfig {
        channels: vec![96, 96, 96, 96, 96],
        time_lengths: vec![4, 4, 4, 4, 4],
        kernel_size: 3,
        stride: 1,
        padding: 1,
        trim: 0,
    }
}

/// Dvoice-scale 4-block decoder: IBP bounds propagate end-to-end.
#[test]
fn test_four_block_decoder_dvoice_ibp() {
    let cfg = dvoice_decoder_config();
    let (def, block_shapes) = build_four_block_decoder(&cfg);

    let mut bindings: Vec<TensorParamBinding> = Vec::new();
    // 5 variable inputs: data + 4 skips
    for _ in 0..5 {
        bindings.push(TensorParamBinding::Variable);
    }
    // 4 constant weights (use small values to keep bounds reasonable)
    for i in 0..4 {
        let doubled_out = cfg.channels[i + 1] * 2;
        let scale = 0.01 / ((i + 1) as f32).sqrt();
        let w = ArrayD::from_elem(
            IxDyn(&[cfg.channels[i], doubled_out, cfg.kernel_size]),
            scale,
        );
        bindings.push(TensorParamBinding::ConstantTensor(w));
    }

    let graph = tensor_kernel_to_graph(&def, &bindings).expect("dvoice 4-block decoder graph");

    // 5 variables, all [96, 4] (uniform shape for multi-variable stacking).
    let input = multi_var_uniform_input(5, 96, 4, -1.0, 1.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through dvoice 4-block decoder");
    let (lo, _hi) = output.lower_upper();

    // Each variable enters its subgraph at its TRUE declared rank (#358 flat
    // per-variable Slice+Reshape harness), so the output keeps its natural
    // [C, T] shape with no leading stacking axis.
    assert_eq!(lo.shape(), block_shapes[3].as_slice());
    assert_bounds_valid(&output);
}

// ---------------------------------------------------------------------------
// CROWN propagation tests
// ---------------------------------------------------------------------------

/// CROWN produces tighter-or-equal bounds than IBP on 4-block decoder.
#[test]
fn test_four_block_decoder_crown_tighter_than_ibp() {
    let cfg = small_uniform_decoder_config();
    let (def, _) = build_four_block_decoder(&cfg);

    let mut bindings: Vec<TensorParamBinding> = Vec::new();
    for _ in 0..5 {
        bindings.push(TensorParamBinding::Variable);
    }
    for i in 0..4 {
        let doubled_out = cfg.channels[i + 1] * 2;
        let w = ArrayD::from_elem(
            IxDyn(&[cfg.channels[i], doubled_out, cfg.kernel_size]),
            0.1f32,
        );
        bindings.push(TensorParamBinding::ConstantTensor(w));
    }

    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");

    // 5 variables, each [16, 8], stacked → [5, 16, 8]
    let input = multi_var_uniform_input(5, 16, 8, -1.0, 1.0);

    let ibp_output = graph
        .propagate_ibp(&input)
        .expect("IBP through 4-block decoder");
    let (_, crown_output, _) =
        propagate_with_crown_fallback(&graph, &input).expect("CROWN through 4-block decoder");

    assert_crown_tighter_than_ibp(&crown_output, &ibp_output);
}

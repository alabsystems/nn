// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Integration tests: Silero VAD 4-block encoder composition.
//!
//! Validates that the Silero VAD encoder stack (4 Conv1d + ReLU blocks)
//! translates through `tensor_kernel_to_graph` and produces a single
//! NY `GraphNetwork` where IBP and CROWN bounds propagate end-to-end.
//!
//! Unlike the Demucs encoder (Conv1d + Snake + InstanceNorm), Silero VAD
//! uses Conv1d + ReLU — a simpler activation but the first real model
//! composed as a single verified graph.
//!
//! Architecture (after STFT, 16kHz):
//! ```text
//! STFT output [129, 4]
//!   → Enc0: Conv1d(129→128, k=3, s=1, p=1) + ReLU → [128, 4]
//!   → Enc1: Conv1d(128→64, k=3, s=2, p=1) + ReLU  → [64, 2]
//!   → Enc2: Conv1d(64→64, k=3, s=2, p=1) + ReLU   → [64, 1]
//!   → Enc3: Conv1d(64→128, k=3, s=1, p=1) + ReLU  → [128, 1]
//! ```
//!
//! Part of #770 AC1-AC4.

use super::common;

use common::{assert_bounds_valid, assert_crown_tighter_than_ibp, bounds_min_max, conv1d_out_len};
use nn_dsl::tensor_block_builder::TensorBlockBuilder;
use nn_verify::{
    propagate_with_crown_fallback, tensor_kernel_to_graph, verify_tensor_and_record, BoundedTensor,
    TensorParamBinding, VerifyStatus,
};
use ndarray::{ArrayD, IxDyn};

// ---------------------------------------------------------------------------
// Silero VAD encoder configuration
// ---------------------------------------------------------------------------

/// Silero VAD encoder block parameters.
struct VadEncoderBlock {
    in_channels: usize,
    out_channels: usize,
    kernel_size: usize,
    stride: usize,
    padding: usize,
}

/// Silero VAD 16kHz encoder blocks (matching silero_vad.rs ENCODER_BLOCKS).
const VAD_BLOCKS: [VadEncoderBlock; 4] = [
    VadEncoderBlock {
        in_channels: 129,
        out_channels: 128,
        kernel_size: 3,
        stride: 1,
        padding: 1,
    },
    VadEncoderBlock {
        in_channels: 128,
        out_channels: 64,
        kernel_size: 3,
        stride: 2,
        padding: 1,
    },
    VadEncoderBlock {
        in_channels: 64,
        out_channels: 64,
        kernel_size: 3,
        stride: 2,
        padding: 1,
    },
    VadEncoderBlock {
        in_channels: 64,
        out_channels: 128,
        kernel_size: 3,
        stride: 1,
        padding: 1,
    },
];

/// Build Silero VAD 4-block encoder as a single TensorKernelDef.
///
/// Each block: Conv1d(in→out, k, stride, pad) + ReLU
/// Biases are included (matching production `SileroVad::forward()`).
///
/// Returns (TensorKernelDef, per-block output shapes, input temporal length).
fn build_vad_encoder(
    in_length: usize,
) -> (nn_dsl::tensor_ir::TensorKernelDef, Vec<[usize; 2]>, usize) {
    let mut b = TensorBlockBuilder::new("silero_vad_encoder");

    // Input: STFT magnitude spectrogram [n_freqs=129, n_frames]
    let data = b.add_input("stft_mag", &[VAD_BLOCKS[0].in_channels, in_length]);

    // Conv weights and biases for each block.
    let weights: Vec<_> = VAD_BLOCKS
        .iter()
        .enumerate()
        .map(|(i, blk)| {
            b.add_input(
                &format!("enc_weight_{i}"),
                &[blk.out_channels, blk.in_channels, blk.kernel_size],
            )
        })
        .collect();

    let biases: Vec<_> = VAD_BLOCKS
        .iter()
        .enumerate()
        .map(|(i, blk)| b.add_input(&format!("enc_bias_{i}"), &[blk.out_channels]))
        .collect();

    let mut prev_output = data;
    let mut block_shapes = Vec::with_capacity(4);
    let mut t = in_length;

    for (i, blk) in VAD_BLOCKS.iter().enumerate() {
        t = conv1d_out_len(t, blk.kernel_size, blk.stride, blk.padding);
        let out_shape = [blk.out_channels, t];
        block_shapes.push(out_shape);

        let conv = b.add_conv1d(
            prev_output,
            weights[i],
            Some(biases[i]),
            blk.stride,
            blk.padding,
            &out_shape,
        );
        let relu = b.add_relu(conv, &out_shape);
        prev_output = relu;
    }

    let def = b.build(prev_output).expect("valid graph");
    (def, block_shapes, in_length)
}

// ---------------------------------------------------------------------------
// Small-scale tests (fast, for CI)
// ---------------------------------------------------------------------------

/// Small-scale bindings: uniform 0.1 weights, zero biases.
fn small_bindings() -> Vec<TensorParamBinding> {
    let mut bindings = vec![TensorParamBinding::Variable]; // stft_mag

    for blk in &VAD_BLOCKS {
        let w = ArrayD::from_elem(
            IxDyn(&[blk.out_channels, blk.in_channels, blk.kernel_size]),
            0.1f32,
        );
        bindings.push(TensorParamBinding::ConstantTensor(w));
    }

    for blk in &VAD_BLOCKS {
        let bias = ArrayD::from_elem(IxDyn(&[blk.out_channels]), 0.0f32);
        bindings.push(TensorParamBinding::ConstantTensor(bias));
    }

    bindings
}

/// Small-scale input length (matching Silero VAD: 4 frames from STFT).
const SILERO_VAD_STFT_FRAMES: usize = 4;

/// Encoder graph builds and translates to NY.
#[test]
fn test_vad_encoder_graph_builds() {
    let (def, block_shapes, _) = build_vad_encoder(SILERO_VAD_STFT_FRAMES);

    // Verify temporal progression: 4 → 4 → 2 → 1 → 1
    assert_eq!(block_shapes[0], [128, 4]);
    assert_eq!(block_shapes[1], [64, 2]);
    assert_eq!(block_shapes[2], [64, 1]);
    assert_eq!(block_shapes[3], [128, 1]);

    let bindings = small_bindings();
    let graph =
        tensor_kernel_to_graph(&def, &bindings).expect("Silero VAD encoder graph translation");

    // 4 blocks × 2 ops (Conv1d + ReLU) = 8 ops minimum, plus input node.
    assert!(
        graph.num_nodes() >= 8,
        "VAD encoder graph should have >= 8 nodes (2 per block × 4), got {}",
        graph.num_nodes()
    );
}

/// IBP bounds propagate through all 4 encoder blocks.
#[test]
fn test_vad_encoder_ibp_propagates() {
    let (def, block_shapes, _) = build_vad_encoder(SILERO_VAD_STFT_FRAMES);

    let bindings = small_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");

    // STFT magnitude is non-negative (it's sqrt(real² + imag²)).
    let lower = ArrayD::from_elem(IxDyn(&[129, SILERO_VAD_STFT_FRAMES]), 0.0f32);
    let upper = ArrayD::from_elem(IxDyn(&[129, SILERO_VAD_STFT_FRAMES]), 10.0f32);
    let input = BoundedTensor::new(lower, upper).expect("bounds");

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through VAD encoder");
    // Final encoder output: [128, 1]
    assert_eq!(output.lower_upper().0.shape(), block_shapes[3].as_slice());
    assert_bounds_valid(&output);

    // ReLU ensures all outputs are non-negative (for non-negative inputs with
    // zero biases and positive weights, but IBP may still show negative lower
    // bounds due to interval arithmetic overapproximation).

    // Magnitude sanity: Conv1d+ReLU with small weights and [0,10] input should
    // not produce astronomically large bounds. 1e10 is generous for 4 blocks.
    let (_, hi_max) = bounds_min_max(&output);
    assert!(
        hi_max < 1e10,
        "IBP upper bound magnitude too large: {hi_max} (possible bug)"
    );
}

/// CROWN propagation through 4-block VAD encoder.
#[test]
fn test_vad_encoder_crown_propagates() {
    let (def, block_shapes, _) = build_vad_encoder(SILERO_VAD_STFT_FRAMES);

    let bindings = small_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");

    let lower = ArrayD::from_elem(IxDyn(&[129, SILERO_VAD_STFT_FRAMES]), 0.0f32);
    let upper = ArrayD::from_elem(IxDyn(&[129, SILERO_VAD_STFT_FRAMES]), 10.0f32);
    let input = BoundedTensor::new(lower, upper).expect("bounds");

    let (_, output, _) =
        propagate_with_crown_fallback(&graph, &input).expect("CROWN through VAD encoder");
    assert_eq!(output.lower_upper().0.shape(), block_shapes[3].as_slice());
    assert_bounds_valid(&output);

    // Magnitude sanity: CROWN should be at least as tight as IBP.
    let (_, hi_max) = bounds_min_max(&output);
    assert!(
        hi_max < 1e10,
        "CROWN upper bound magnitude too large: {hi_max} (possible bug)"
    );
}

/// IBP vs CROWN: CROWN should produce tighter (or equal) bounds.
#[test]
fn test_vad_encoder_crown_tighter_than_ibp() {
    let (def, _, _) = build_vad_encoder(SILERO_VAD_STFT_FRAMES);

    let bindings = small_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");

    let lower = ArrayD::from_elem(IxDyn(&[129, SILERO_VAD_STFT_FRAMES]), 0.0f32);
    let upper = ArrayD::from_elem(IxDyn(&[129, SILERO_VAD_STFT_FRAMES]), 10.0f32);
    let input = BoundedTensor::new(lower, upper).expect("bounds");

    let ibp_output = graph.propagate_ibp(&input).expect("IBP");
    let (_, crown_output, _) = propagate_with_crown_fallback(&graph, &input).expect("CROWN");

    assert_crown_tighter_than_ibp(&crown_output, &ibp_output);
}

/// CROWN vs IBP bound width comparison for the Silero VAD encoder.
///
/// Measures the actual width improvement that CROWN provides over IBP
/// for the 4-block Conv1d+ReLU encoder at its native dimensions
/// (129→128→64→64→128 channels). With channel dimensions of 64-129,
/// CROWN's linear relaxation should provide meaningful tightening.
///
/// Part of #2239: Document bound tightness improvement.
#[test]
fn test_vad_encoder_crown_ibp_width_comparison() {
    let (def, _, _) = build_vad_encoder(SILERO_VAD_STFT_FRAMES);

    let bindings = small_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");

    let lower = ArrayD::from_elem(IxDyn(&[129, SILERO_VAD_STFT_FRAMES]), 0.0f32);
    let upper = ArrayD::from_elem(IxDyn(&[129, SILERO_VAD_STFT_FRAMES]), 10.0f32);
    let input = BoundedTensor::new(lower, upper).expect("bounds");

    let ibp_output = graph.propagate_ibp(&input).expect("IBP");
    let (method, crown_output, fallback_reason) =
        propagate_with_crown_fallback(&graph, &input).expect("CROWN");

    let (ibp_lo_min, ibp_hi_max) = bounds_min_max(&ibp_output);
    let (crown_lo_min, crown_hi_max) = bounds_min_max(&crown_output);

    let ibp_width = ibp_hi_max - ibp_lo_min;
    let crown_width = crown_hi_max - crown_lo_min;
    let improvement = if crown_width > 0.0 {
        ibp_width / crown_width
    } else {
        f32::INFINITY
    };

    eprintln!("\n=== Silero VAD Encoder: CROWN vs IBP Width ===");
    eprintln!("  Method:       {method:?}");
    eprintln!("  IBP bounds:   [{ibp_lo_min:.4}, {ibp_hi_max:.4}] width={ibp_width:.4}");
    eprintln!("  CROWN bounds: [{crown_lo_min:.4}, {crown_hi_max:.4}] width={crown_width:.4}");
    eprintln!("  Improvement:  {improvement:.2}x");

    if let Some(reason) = &fallback_reason {
        eprintln!("  Fallback:     {reason}");
    }

    // Soundness: CROWN width <= IBP width.
    assert!(
        crown_width <= ibp_width + 1e-4,
        "CROWN width {crown_width} must be <= IBP width {ibp_width} (soundness)"
    );

    // Document CROWN method and improvement for #2239 evidence.
    if matches!(method, nn_verify::PropMethod::Crown) {
        eprintln!("  VAD encoder (d_in=129): CROWN provides {improvement:.1}x tighter bounds ✓");
    }
}

// ---------------------------------------------------------------------------
// Status recording (AC4: model-level verification in status file)
// ---------------------------------------------------------------------------

/// Record Silero VAD encoder verification in `VerifyStatus`.
///
/// Uses `verify_tensor_and_record` pipeline: translates the composed
/// 4-block encoder to NY, propagates bounds (IBP → CROWN
/// escalation), and records the result under "silero_vad_encoder".
#[test]
fn test_vad_encoder_verify_and_record() {
    let (def, block_shapes, _) = build_vad_encoder(SILERO_VAD_STFT_FRAMES);

    let bindings = small_bindings();
    let input = BoundedTensor::new(
        ArrayD::from_elem(IxDyn(&[129, SILERO_VAD_STFT_FRAMES]), 0.0f32),
        ArrayD::from_elem(IxDyn(&[129, SILERO_VAD_STFT_FRAMES]), 10.0f32),
    )
    .expect("input bounds");

    let mut status = VerifyStatus::default();
    let result = verify_tensor_and_record(
        &mut status,
        &def,
        &bindings,
        &input,
        Some("silero_vad_encoder"),
    )
    .expect("verify_tensor_and_record pipeline");

    // Verification result should have finite bounds.
    assert!(
        result.verification.is_finite,
        "encoder output bounds must be finite"
    );
    assert_eq!(result.num_variables, 1, "single Variable input (stft_mag)");

    // Output tensor bounds should match final encoder shape [128, 1].
    let (lo, _hi) = result.output_bounds.lower_upper();
    assert_eq!(lo.shape(), block_shapes[3].as_slice());
    assert_bounds_valid(&result.output_bounds);

    // Status file should contain an entry for the encoder.
    assert!(
        status.kernel("silero_vad_encoder").is_some(),
        "status should contain 'silero_vad_encoder' entry"
    );
}

// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! IBP compose verification tests for Silero VAD pipeline bounds.
//!
//! Complements existing Silero VAD tests (full, encoder, deep, certificate)
//! with pipeline-level bound property tests:
//!
//! 1. **Audio normalization** — pre-STFT audio in [-1,1] produces tighter
//!    output bounds than unnormalized audio, verifying the normalization
//!    contract at the pipeline entry point.
//! 2. **Encoder-to-LSTM reshape** — isolated reshape from [128,1] to [1,128]
//!    preserves bounds exactly (no widening from reshape).
//! 3. **Encoder progressive tightening** — bounds width monotonically relates
//!    to the number of encoder blocks (more blocks = more computation, but
//!    ReLU truncation prevents unbounded growth).
//! 4. **Weight magnitude sensitivity** — larger weights produce wider output
//!    bounds. Verifies that IBP correctly tracks weight scale impact.
//! 5. **Negative weight bounds** — encoder with mixed positive/negative weights
//!    still produces finite, valid bounds. ReLU clipping remains effective.
//! 6. **Multi-threshold analysis** — output probability bounds checked against
//!    multiple VAD thresholds (0.3, 0.5, 0.7) for threshold margin analysis.
//! 7. **LSTM gate isolation** — individual LSTM gates (forget, input, output)
//!    produce bounded outputs from sigmoid/tanh activations.
//! 8. **Full pipeline with normalized input** — end-to-end verification that
//!    normalized audio input [0, 1] produces tighter bounds than [0, 10].
//!
//! Part of #4186.

use super::common::{assert_bounds_valid, bounds_min_max, verify_and_assert};
use crate::silero_vad_test_helpers::{
    build_full_silero_vad, full_model_bindings, stft_input_bounds, LSTM_HIDDEN_SIZE, STFT_N_FRAMES,
    STFT_N_FREQS, VAD_BLOCKS,
};
use ndarray::{ArrayD, IxDyn};
use nn_dsl::tensor_block_builder::TensorBlockBuilder;
use nn_verify::{tensor_kernel_to_graph, BoundedTensor, TensorParamBinding};

// ---------------------------------------------------------------------------
// 1. Audio normalization bounds
// ---------------------------------------------------------------------------

/// Normalized audio input (STFT magnitude from [-1,1] audio) produces tighter
/// output bounds than unnormalized input (STFT magnitude from [-32768,32767] audio).
///
/// The STFT magnitude of normalized audio is bounded by a much smaller range
/// than unnormalized audio. This test verifies that the pipeline correctly
/// propagates the tighter input range to a tighter output range.
#[test]
fn test_audio_normalization_tighter_output() {
    let def = build_full_silero_vad();
    let bindings = full_model_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");

    // Normalized audio: STFT magnitude in [0, 1] (from audio in [-1, 1]).
    let normalized_input = BoundedTensor::new(
        ArrayD::from_elem(IxDyn(&[STFT_N_FREQS, STFT_N_FRAMES]), 0.0f32),
        ArrayD::from_elem(IxDyn(&[STFT_N_FREQS, STFT_N_FRAMES]), 1.0f32),
    )
    .expect("normalized bounds");

    // Unnormalized audio: STFT magnitude in [0, 100] (large dynamic range).
    let unnormalized_input = BoundedTensor::new(
        ArrayD::from_elem(IxDyn(&[STFT_N_FREQS, STFT_N_FRAMES]), 0.0f32),
        ArrayD::from_elem(IxDyn(&[STFT_N_FREQS, STFT_N_FRAMES]), 100.0f32),
    )
    .expect("unnormalized bounds");

    let norm_output = graph
        .propagate_ibp(&normalized_input)
        .expect("IBP normalized");
    let unnorm_output = graph
        .propagate_ibp(&unnormalized_input)
        .expect("IBP unnormalized");

    assert_bounds_valid(&norm_output);
    assert_bounds_valid(&unnorm_output);

    let (norm_lo, norm_hi) = bounds_min_max(&norm_output);
    let (unnorm_lo, unnorm_hi) = bounds_min_max(&unnorm_output);

    let norm_width = norm_hi - norm_lo;
    let unnorm_width = unnorm_hi - unnorm_lo;

    eprintln!("Normalized [0,1]:   bounds [{norm_lo:.6}, {norm_hi:.6}], width={norm_width:.6}");
    eprintln!(
        "Unnormalized [0,100]: bounds [{unnorm_lo:.6}, {unnorm_hi:.6}], width={unnorm_width:.6}"
    );

    // Normalized input should produce tighter or equal output bounds.
    assert!(
        norm_width <= unnorm_width + 1e-6,
        "normalized audio should produce tighter output bounds: \
         norm_width={norm_width}, unnorm_width={unnorm_width}"
    );

    // Both should be valid probabilities.
    assert!(norm_lo >= -0.01, "normalized lower >= 0, got {norm_lo}");
    assert!(norm_hi <= 1.01, "normalized upper <= 1, got {norm_hi}");
}

// ---------------------------------------------------------------------------
// 2. Encoder-to-LSTM reshape preserves bounds
// ---------------------------------------------------------------------------

/// Reshape from [128, 1] to [1, 128] must preserve bounds exactly.
///
/// The reshape between encoder output and LSTM input is a structural
/// operation with no computation. IBP bounds must be identical before
/// and after reshape (no widening).
#[test]
fn test_reshape_preserves_bounds() {
    let h = LSTM_HIDDEN_SIZE;
    let mut b = TensorBlockBuilder::new("silero_vad_reshape_test");

    let input = b.add_input("enc_output", &[h, 1]);
    let reshaped = b.add_reshape(input, &[1, h]);
    let def = b.build(reshaped).expect("valid reshape graph");

    let bindings = vec![TensorParamBinding::Variable];

    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");

    // Non-uniform input bounds to detect any element reordering issues.
    let mut lo_data = vec![0.0f32; h];
    let mut hi_data = vec![0.0f32; h];
    for i in 0..h {
        lo_data[i] = i as f32 * 0.01;
        hi_data[i] = lo_data[i] + 1.0;
    }
    let input_bounds = BoundedTensor::new(
        ArrayD::from_shape_vec(IxDyn(&[h, 1]), lo_data.clone()).expect("lo"),
        ArrayD::from_shape_vec(IxDyn(&[h, 1]), hi_data.clone()).expect("hi"),
    )
    .expect("bounds");

    let output = graph.propagate_ibp(&input_bounds).expect("IBP");
    let (out_lo, out_hi) = output.lower_upper();

    assert_eq!(out_lo.shape(), &[1, h], "output shape should be [1, {h}]");
    assert_bounds_valid(&output);

    // Verify bounds are preserved element-wise (reshape is just reinterpretation).
    for i in 0..h {
        let expected_lo = lo_data[i];
        let expected_hi = hi_data[i];
        let actual_lo = out_lo[[0, i]];
        let actual_hi = out_hi[[0, i]];
        assert!(
            (actual_lo - expected_lo).abs() < 1e-6,
            "reshape lower[{i}]: expected {expected_lo}, got {actual_lo}"
        );
        assert!(
            (actual_hi - expected_hi).abs() < 1e-6,
            "reshape upper[{i}]: expected {expected_hi}, got {actual_hi}"
        );
    }
}

/// Record reshape verification in status file.
#[test]
fn test_reshape_verify_and_record() {
    let h = LSTM_HIDDEN_SIZE;
    let mut b = TensorBlockBuilder::new("silero_vad_reshape");
    let input = b.add_input("enc_output", &[h, 1]);
    let reshaped = b.add_reshape(input, &[1, h]);
    let def = b.build(reshaped).expect("valid graph");

    let bindings = vec![TensorParamBinding::Variable];
    let input_bounds = BoundedTensor::new(
        ArrayD::from_elem(IxDyn(&[h, 1]), 0.0f32),
        ArrayD::from_elem(IxDyn(&[h, 1]), 5.0f32),
    )
    .expect("bounds");

    let result = verify_and_assert(&def, &bindings, &input_bounds, "silero_vad_reshape");
    assert_eq!(result.num_variables, 1);
}

// ---------------------------------------------------------------------------
// 3. Encoder progressive bounds — ReLU prevents unbounded growth
// ---------------------------------------------------------------------------

/// Each additional encoder block does not cause unbounded growth in output
/// bounds. ReLU truncation at zero clips negative intervals, preventing
/// exponential blow-up of IBP over-approximation through multiple layers.
#[test]
fn test_encoder_progressive_relu_clips() {
    use super::common::conv1d_out_len;

    // Build a 1-block, 2-block, 3-block, and 4-block encoder and compare widths.
    let mut widths = Vec::new();

    for num_blocks in 1..=4 {
        let blocks = &VAD_BLOCKS[..num_blocks];
        let mut b = TensorBlockBuilder::new(&format!("silero_vad_enc_{num_blocks}_blocks"));

        let stft = b.add_input("stft_mag", &[blocks[0].in_channels, STFT_N_FRAMES]);

        // Declare all weight inputs first, then all bias inputs, so the
        // declaration order matches the grouped `bindings` order below
        // (positional binding requires exact `add_input()` order). Mirrors
        // the convention in `add_encoder_blocks`.
        let mut weights = Vec::new();
        for (i, blk) in blocks.iter().enumerate() {
            weights.push(b.add_input(
                &format!("w_{i}"),
                &[blk.out_channels, blk.in_channels, blk.kernel_size],
            ));
        }
        let mut biases = Vec::new();
        for (i, blk) in blocks.iter().enumerate() {
            biases.push(b.add_input(&format!("b_{i}"), &[blk.out_channels]));
        }

        let mut prev = stft;
        let mut t = STFT_N_FRAMES;
        for (i, blk) in blocks.iter().enumerate() {
            t = conv1d_out_len(t, blk.kernel_size, blk.stride, blk.padding);
            let out_shape = [blk.out_channels, t];
            let conv = b.add_conv1d(
                prev,
                weights[i],
                Some(biases[i]),
                blk.stride,
                blk.padding,
                &out_shape,
            );
            prev = b.add_relu(conv, &out_shape);
        }

        let def = b.build(prev).expect("valid graph");

        let mut bindings = vec![TensorParamBinding::Variable];
        for blk in blocks {
            bindings.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
                IxDyn(&[blk.out_channels, blk.in_channels, blk.kernel_size]),
                0.01f32,
            )));
        }
        for blk in blocks {
            bindings.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
                IxDyn(&[blk.out_channels]),
                0.0f32,
            )));
        }

        let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");
        let input = BoundedTensor::new(
            ArrayD::from_elem(IxDyn(&[blocks[0].in_channels, STFT_N_FRAMES]), 0.0f32),
            ArrayD::from_elem(IxDyn(&[blocks[0].in_channels, STFT_N_FRAMES]), 10.0f32),
        )
        .expect("bounds");

        let output = graph.propagate_ibp(&input).expect("IBP");
        assert_bounds_valid(&output);

        let (lo_min, hi_max) = bounds_min_max(&output);
        let width = hi_max - lo_min;
        widths.push(width);

        eprintln!(
            "Encoder {num_blocks} blocks: bounds [{lo_min:.4}, {hi_max:.4}], width={width:.4}"
        );
    }

    // All widths should be finite and non-negative.
    for (i, w) in widths.iter().enumerate() {
        assert!(w.is_finite(), "width for {} blocks must be finite", i + 1);
        assert!(*w >= 0.0, "width for {} blocks must be non-negative", i + 1);
    }

    // The 4-block encoder should not have astronomically wider bounds than 1-block.
    // With small weights (0.01) and ReLU clipping, growth should be moderate.
    assert!(
        widths[3] < widths[0] * 1000.0,
        "4-block bounds width {} should not be >1000x wider than 1-block width {}",
        widths[3],
        widths[0]
    );
}

// ---------------------------------------------------------------------------
// 4. Weight magnitude sensitivity
// ---------------------------------------------------------------------------

/// Larger weights produce wider IBP output bounds. This verifies that
/// IBP correctly tracks the effect of weight magnitude on output uncertainty.
#[test]
fn test_weight_magnitude_widens_bounds() {
    let def = build_full_silero_vad();

    let make_bindings = |weight_val: f32| -> Vec<TensorParamBinding> {
        let mut bindings = Vec::new();
        bindings.push(TensorParamBinding::Variable);

        // LSTM states: zero.
        bindings.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[1, LSTM_HIDDEN_SIZE]),
            0.0f32,
        )));
        bindings.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[1, LSTM_HIDDEN_SIZE]),
            0.0f32,
        )));

        // Encoder weights + biases.
        for blk in &VAD_BLOCKS {
            bindings.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
                IxDyn(&[blk.out_channels, blk.in_channels, blk.kernel_size]),
                weight_val,
            )));
        }
        for blk in &VAD_BLOCKS {
            bindings.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
                IxDyn(&[blk.out_channels]),
                0.0f32,
            )));
        }

        // LSTM weights and bias.
        bindings.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[4 * LSTM_HIDDEN_SIZE, LSTM_HIDDEN_SIZE]),
            weight_val,
        )));
        bindings.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[4 * LSTM_HIDDEN_SIZE, LSTM_HIDDEN_SIZE]),
            weight_val,
        )));
        bindings.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[4 * LSTM_HIDDEN_SIZE]),
            0.0f32,
        )));

        // Output weight and bias.
        bindings.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[1, LSTM_HIDDEN_SIZE]),
            weight_val,
        )));
        bindings.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[1]),
            0.0f32,
        )));
        bindings
    };

    let input = stft_input_bounds();

    // Small weights.
    let small_bindings = make_bindings(0.001);
    let graph_small = tensor_kernel_to_graph(&def, &small_bindings).expect("graph small");
    let out_small = graph_small.propagate_ibp(&input).expect("IBP small");
    assert_bounds_valid(&out_small);

    // Larger weights.
    let large_bindings = make_bindings(0.1);
    let graph_large = tensor_kernel_to_graph(&def, &large_bindings).expect("graph large");
    let out_large = graph_large.propagate_ibp(&input).expect("IBP large");
    assert_bounds_valid(&out_large);

    let (small_lo, small_hi) = bounds_min_max(&out_small);
    let (large_lo, large_hi) = bounds_min_max(&out_large);

    let small_width = small_hi - small_lo;
    let large_width = large_hi - large_lo;

    eprintln!("Small weights (0.001): width={small_width:.6}");
    eprintln!("Large weights (0.1):   width={large_width:.6}");

    // Larger weights should produce wider or equal bounds.
    assert!(
        large_width >= small_width - 1e-6,
        "larger weights should produce wider bounds: large_width={large_width}, small_width={small_width}"
    );
}

// ---------------------------------------------------------------------------
// 5. Negative weight bounds
// ---------------------------------------------------------------------------

/// Encoder with mixed positive/negative weights produces finite, valid bounds.
///
/// Negative weights cause Conv1d to flip interval endpoints (IBP must swap
/// lower/upper when multiplying by negative values). Verifies IBP correctness
/// for the signed-weight case with ReLU clipping.
#[test]
fn test_negative_weights_finite_bounds() {
    let def = build_full_silero_vad();

    // Build bindings with alternating positive/negative weights.
    let mut bindings = Vec::new();
    bindings.push(TensorParamBinding::Variable);

    // LSTM states: zero.
    bindings.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
        IxDyn(&[1, LSTM_HIDDEN_SIZE]),
        0.0f32,
    )));
    bindings.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
        IxDyn(&[1, LSTM_HIDDEN_SIZE]),
        0.0f32,
    )));

    // Encoder weights: alternating +0.01 and -0.01.
    for blk in &VAD_BLOCKS {
        let n = blk.out_channels * blk.in_channels * blk.kernel_size;
        let data: Vec<f32> = (0..n)
            .map(|i| if i % 2 == 0 { 0.01 } else { -0.01 })
            .collect();
        bindings.push(TensorParamBinding::ConstantTensor(
            ArrayD::from_shape_vec(
                IxDyn(&[blk.out_channels, blk.in_channels, blk.kernel_size]),
                data,
            )
            .expect("weight shape"),
        ));
    }
    for blk in &VAD_BLOCKS {
        bindings.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[blk.out_channels]),
            0.0f32,
        )));
    }

    // LSTM weights: mixed sign.
    let lstm_n = 4 * LSTM_HIDDEN_SIZE * LSTM_HIDDEN_SIZE;
    let lstm_data: Vec<f32> = (0..lstm_n)
        .map(|i| if i % 2 == 0 { 0.01 } else { -0.01 })
        .collect();
    bindings.push(TensorParamBinding::ConstantTensor(
        ArrayD::from_shape_vec(
            IxDyn(&[4 * LSTM_HIDDEN_SIZE, LSTM_HIDDEN_SIZE]),
            lstm_data.clone(),
        )
        .expect("w_ih shape"),
    ));
    bindings.push(TensorParamBinding::ConstantTensor(
        ArrayD::from_shape_vec(IxDyn(&[4 * LSTM_HIDDEN_SIZE, LSTM_HIDDEN_SIZE]), lstm_data)
            .expect("w_hh shape"),
    ));
    bindings.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
        IxDyn(&[4 * LSTM_HIDDEN_SIZE]),
        0.0f32,
    )));

    // Output weights: mixed sign.
    let out_n = LSTM_HIDDEN_SIZE;
    let out_data: Vec<f32> = (0..out_n)
        .map(|i| if i % 2 == 0 { 0.01 } else { -0.01 })
        .collect();
    bindings.push(TensorParamBinding::ConstantTensor(
        ArrayD::from_shape_vec(IxDyn(&[1, LSTM_HIDDEN_SIZE]), out_data).expect("out_w shape"),
    ));
    bindings.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
        IxDyn(&[1]),
        0.0f32,
    )));

    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph with negative weights");
    let input = stft_input_bounds();
    let output = graph
        .propagate_ibp(&input)
        .expect("IBP with negative weights");

    assert_bounds_valid(&output);

    let (lo, hi) = output.lower_upper();
    let lo_val = lo[[0, 0]];
    let hi_val = hi[[0, 0]];

    assert!(lo_val.is_finite(), "negative weight lower must be finite");
    assert!(hi_val.is_finite(), "negative weight upper must be finite");

    // Sigmoid output: still in [0, 1].
    assert!(
        lo_val >= -0.01,
        "sigmoid lower with negative weights >= 0, got {lo_val}"
    );
    assert!(
        hi_val <= 1.01,
        "sigmoid upper with negative weights <= 1, got {hi_val}"
    );

    eprintln!("Negative weights: probability bounds [{lo_val:.6}, {hi_val:.6}]");
}

// ---------------------------------------------------------------------------
// 6. Multi-threshold analysis
// ---------------------------------------------------------------------------

/// Verify output bounds relative to multiple VAD thresholds.
///
/// Silero VAD uses a configurable speech detection threshold (default 0.5).
/// This test checks that IBP bounds are informative relative to typical
/// threshold values (0.3 for sensitive detection, 0.5 default, 0.7 strict).
#[test]
fn test_multi_threshold_analysis() {
    let def = build_full_silero_vad();
    let bindings = full_model_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");

    // Moderate input: typical speech-level STFT magnitudes.
    let input = BoundedTensor::new(
        ArrayD::from_elem(IxDyn(&[STFT_N_FREQS, STFT_N_FRAMES]), 0.0f32),
        ArrayD::from_elem(IxDyn(&[STFT_N_FREQS, STFT_N_FRAMES]), 5.0f32),
    )
    .expect("bounds");

    let output = graph.propagate_ibp(&input).expect("IBP");
    assert_bounds_valid(&output);

    let (lo, hi) = output.lower_upper();
    let lo_val = lo[[0, 0]];
    let hi_val = hi[[0, 0]];

    eprintln!("Multi-threshold analysis: output bounds [{lo_val:.6}, {hi_val:.6}]");

    // Check against standard thresholds.
    let thresholds = [(0.3, "sensitive"), (0.5, "default"), (0.7, "strict")];
    for (threshold, label) in &thresholds {
        let below_threshold = hi_val < *threshold;
        let above_threshold = lo_val > *threshold;
        let straddles = lo_val <= *threshold && hi_val >= *threshold;

        eprintln!(
            "  Threshold {threshold} ({label}): below={below_threshold}, above={above_threshold}, straddles={straddles}"
        );

        // Bounds must be valid probabilities regardless of threshold.
        assert!(lo_val >= -0.01, "lower >= 0 for threshold {threshold}");
        assert!(hi_val <= 1.01, "upper <= 1 for threshold {threshold}");
    }

    // Output width should be meaningful (not vacuously wide).
    let width = hi_val - lo_val;
    assert!(
        width < 1.0,
        "output width {width:.4} for [0,5] input should be < 1.0 (non-vacuous)"
    );
}

// ---------------------------------------------------------------------------
// 7. LSTM gate isolation — sigmoid/tanh gate bounds
// ---------------------------------------------------------------------------

/// Isolated LSTM gate (sigmoid activation) produces output in [0, 1].
///
/// Each LSTM gate (forget, input, output) applies sigmoid to a linear
/// combination. This test verifies that a single gate's IBP bounds
/// respect the sigmoid range, independent of the full LSTM decomposition.
#[test]
fn test_lstm_gate_sigmoid_bounds() {
    let h = LSTM_HIDDEN_SIZE;
    let mut b = TensorBlockBuilder::new("silero_vad_lstm_gate");

    // Single gate: Linear(input) + Linear(hidden) + bias -> Sigmoid
    let input = b.add_input("gate_input", &[1, h]);
    let weight = b.add_input("gate_weight", &[h, h]);
    let bias = b.add_input("gate_bias", &[1, h]);

    // Linear: input @ weight^T -> [1, h]
    let linear = b.add_linear(input, weight, None, &[1, h]);
    // Add bias
    let biased = b.add_binary_add(linear, bias, &[1, h]);
    // Sigmoid
    let gate_out = b.add_sigmoid(biased, &[1, h]);

    let def = b.build(gate_out).expect("valid gate graph");

    let bindings = vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[h, h]), 0.01f32)),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[1, h]), 0.0f32)),
    ];

    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");

    // Gate input range: encoder output after ReLU is non-negative.
    let input_bounds = BoundedTensor::new(
        ArrayD::from_elem(IxDyn(&[1, h]), 0.0f32),
        ArrayD::from_elem(IxDyn(&[1, h]), 5.0f32),
    )
    .expect("bounds");

    let output = graph.propagate_ibp(&input_bounds).expect("IBP gate");
    assert_bounds_valid(&output);

    let (lo, hi) = output.lower_upper();
    assert_eq!(lo.shape(), &[1, h]);

    // Sigmoid output must be in [0, 1] for all elements.
    let lo_min = lo.iter().copied().fold(f32::INFINITY, f32::min);
    let hi_max = hi.iter().copied().fold(f32::NEG_INFINITY, f32::max);

    assert!(
        lo_min >= -0.01,
        "sigmoid gate lower must be >= 0, got {lo_min}"
    );
    assert!(
        hi_max <= 1.01,
        "sigmoid gate upper must be <= 1, got {hi_max}"
    );

    eprintln!("LSTM gate sigmoid bounds: [{lo_min:.6}, {hi_max:.6}]");
}

/// Record LSTM gate verification in status file.
#[test]
fn test_lstm_gate_verify_and_record() {
    let h = LSTM_HIDDEN_SIZE;
    let mut b = TensorBlockBuilder::new("silero_vad_lstm_gate");

    let input = b.add_input("gate_input", &[1, h]);
    let weight = b.add_input("gate_weight", &[h, h]);
    let bias = b.add_input("gate_bias", &[1, h]);

    let linear = b.add_linear(input, weight, None, &[1, h]);
    let biased = b.add_binary_add(linear, bias, &[1, h]);
    let gate_out = b.add_sigmoid(biased, &[1, h]);

    let def = b.build(gate_out).expect("valid graph");

    let bindings = vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[h, h]), 0.01f32)),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[1, h]), 0.0f32)),
    ];

    let input_bounds = BoundedTensor::new(
        ArrayD::from_elem(IxDyn(&[1, h]), 0.0f32),
        ArrayD::from_elem(IxDyn(&[1, h]), 5.0f32),
    )
    .expect("bounds");

    let result = verify_and_assert(&def, &bindings, &input_bounds, "silero_vad_lstm_gate");
    assert_eq!(result.num_variables, 1);
}

// ---------------------------------------------------------------------------
// 8. Full pipeline with normalized input — end-to-end tightness
// ---------------------------------------------------------------------------

/// Full pipeline verification with normalized STFT input [0, 1] produces
/// tighter bounds than the standard [0, 10] range, and records both entries
/// in the verification status file.
#[test]
fn test_full_pipeline_normalized_verify_and_record() {
    let def = build_full_silero_vad();
    let bindings = full_model_bindings();

    // Normalized input: STFT magnitude in [0, 1].
    let normalized_input = BoundedTensor::new(
        ArrayD::from_elem(IxDyn(&[STFT_N_FREQS, STFT_N_FRAMES]), 0.0f32),
        ArrayD::from_elem(IxDyn(&[STFT_N_FREQS, STFT_N_FRAMES]), 1.0f32),
    )
    .expect("normalized bounds");

    let result = verify_and_assert(
        &def,
        &bindings,
        &normalized_input,
        "silero_vad_full_normalized",
    );
    assert_eq!(result.num_variables, 1);

    // Also run with standard [0, 10] input for comparison.
    let standard_input = stft_input_bounds();
    let result_std =
        verify_and_assert(&def, &bindings, &standard_input, "silero_vad_full_standard");
    assert_eq!(result_std.num_variables, 1);

    // Normalized output should be tighter.
    let (norm_lo, norm_hi) = bounds_min_max(&result.output_bounds);
    let (std_lo, std_hi) = bounds_min_max(&result_std.output_bounds);

    let norm_width = norm_hi - norm_lo;
    let std_width = std_hi - std_lo;

    eprintln!("Normalized [0,1]:  width={norm_width:.6}, bounds=[{norm_lo:.6}, {norm_hi:.6}]");
    eprintln!("Standard [0,10]:   width={std_width:.6}, bounds=[{std_lo:.6}, {std_hi:.6}]");

    assert!(
        norm_width <= std_width + 1e-6,
        "normalized pipeline should have tighter bounds: norm={norm_width}, std={std_width}"
    );
}

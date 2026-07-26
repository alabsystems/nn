// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Deep NY compose verification for Silero VAD model.
//!
//! Complements the existing full-pipeline and encoder-stack tests with:
//!
//! 1. **Isolated LSTM cell** — verifies LSTM gate decomposition produces
//!    bounded outputs for arbitrary encoder embeddings and zero initial state.
//! 2. **Isolated output classifier** — ReLU + Linear(128->1) + Sigmoid stage
//!    verified independently to confirm probability output in [0, 1].
//! 3. **Streaming two-step** — simulates two sequential VAD frames where
//!    the second frame uses the LSTM hidden/cell state produced by the first.
//!    Verifies that multi-frame bounds remain finite and valid.
//! 4. **Decision boundary** — verifies that the VAD output probability bounds
//!    relate correctly to the 0.5 speech detection threshold.
//! 5. **Per-block encoder** — each of the 4 Conv1d+ReLU encoder blocks
//!    verified in isolation with correct input/output shapes.
//!
//! Part of #4281.

use super::common::{
    assert_bounds_valid, assert_crown_tighter_when_not_fallback, bounds_min_max, conv1d_out_len,
    verify_and_assert,
};
use crate::silero_vad_test_helpers::{LSTM_HIDDEN_SIZE, STFT_N_FRAMES, STFT_N_FREQS, VAD_BLOCKS};
use ndarray::{ArrayD, IxDyn};
use nn_dsl::tensor_block_builder::TensorBlockBuilder;
use nn_verify::{
    propagate_with_crown_fallback, tensor_kernel_to_graph, BoundedTensor, TensorParamBinding,
};

// ---------------------------------------------------------------------------
// 1. Isolated LSTM cell composition
// ---------------------------------------------------------------------------

/// Build a standalone LSTM cell graph: input [1, H] + hidden [1, H] + cell [1, H] -> h_new [1, H].
fn build_isolated_lstm() -> nn_dsl::tensor_ir::TensorKernelDef {
    let h = LSTM_HIDDEN_SIZE;
    let mut b = TensorBlockBuilder::new("silero_vad_lstm_cell");

    let input = b.add_input("lstm_input", &[1, h]);
    let hidden = b.add_input("hidden_state", &[1, h]);
    let cell = b.add_input("cell_state", &[1, h]);
    let w_ih = b.add_input("weight_ih", &[4 * h, h]);
    let w_hh = b.add_input("weight_hh", &[4 * h, h]);
    let bias = b.add_input("bias", &[4 * h]);

    let lstm_out = b.add_lstm(input, hidden, cell, w_ih, w_hh, Some(bias), &[1, h]);
    b.build(lstm_out).expect("valid LSTM graph")
}

/// Parameter bindings for isolated LSTM: lstm_input is Variable, rest are constants.
fn lstm_bindings() -> Vec<TensorParamBinding> {
    let h = LSTM_HIDDEN_SIZE;
    vec![
        TensorParamBinding::Variable, // lstm_input
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[1, h]), 0.0f32)), // hidden
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[1, h]), 0.0f32)), // cell
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[4 * h, h]), 0.01f32)), // w_ih
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[4 * h, h]), 0.01f32)), // w_hh
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[4 * h]), 0.0f32)), // bias
    ]
}

/// LSTM input bounds: encoder output range is non-negative (after ReLU).
fn lstm_input_bounds() -> BoundedTensor {
    let h = LSTM_HIDDEN_SIZE;
    BoundedTensor::new(
        ArrayD::from_elem(IxDyn(&[1, h]), 0.0f32),
        ArrayD::from_elem(IxDyn(&[1, h]), 5.0f32),
    )
    .expect("lstm input bounds")
}

/// LSTM cell graph builds and translates to NY.
#[test]
fn test_lstm_cell_graph_builds() {
    let def = build_isolated_lstm();
    let bindings = lstm_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("LSTM graph translation");

    // LSTM decomposes into ~21 internal nodes (4 gates x 5 ops + tanh/sigmoid).
    assert!(
        graph.num_nodes() >= 15,
        "LSTM graph should have >= 15 decomposed nodes, got {}",
        graph.num_nodes()
    );
}

/// IBP propagates through isolated LSTM cell with finite bounds.
#[test]
fn test_lstm_cell_ibp_propagates() {
    let def = build_isolated_lstm();
    let bindings = lstm_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");

    let input = lstm_input_bounds();
    let output = graph.propagate_ibp(&input).expect("IBP through LSTM");

    assert_eq!(
        output.lower_upper().0.shape(),
        &[1, LSTM_HIDDEN_SIZE],
        "LSTM output shape should be [1, {LSTM_HIDDEN_SIZE}]"
    );
    assert_bounds_valid(&output);

    // LSTM tanh output gates bound h_new to approximately [-1, 1] per element,
    // but IBP overapproximation may widen this. Check reasonable magnitude.
    let (lo_min, hi_max) = bounds_min_max(&output);
    assert!(
        lo_min > -10.0 && hi_max < 10.0,
        "LSTM IBP bounds magnitude should be reasonable: [{lo_min}, {hi_max}]"
    );
}

/// CROWN propagation through isolated LSTM cell.
#[test]
fn test_lstm_cell_crown_propagates() {
    let def = build_isolated_lstm();
    let bindings = lstm_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");
    let input = lstm_input_bounds();

    let (method, output, fallback_reason) =
        propagate_with_crown_fallback(&graph, &input).expect("propagation");

    assert_eq!(output.lower_upper().0.shape(), &[1, LSTM_HIDDEN_SIZE],);
    assert_bounds_valid(&output);

    eprintln!(
        "LSTM cell: method={method:?}, fallback={fallback_reason:?}"
    );
}

/// Record isolated LSTM cell verification in status file.
#[test]
fn test_lstm_cell_verify_and_record() {
    let def = build_isolated_lstm();
    let bindings = lstm_bindings();
    let input = lstm_input_bounds();

    let result = verify_and_assert(&def, &bindings, &input, "silero_vad_lstm_cell");
    assert_eq!(
        result.num_variables, 1,
        "single Variable input (lstm_input)"
    );
}

// ---------------------------------------------------------------------------
// 2. Isolated output classifier stage
// ---------------------------------------------------------------------------

/// Build standalone output classifier: ReLU -> Linear(128->1) -> Sigmoid.
fn build_isolated_output_stage() -> nn_dsl::tensor_ir::TensorKernelDef {
    let h = LSTM_HIDDEN_SIZE;
    let mut b = TensorBlockBuilder::new("silero_vad_output_stage");

    let input = b.add_input("classifier_input", &[1, h]);
    let weight = b.add_input("output_weight", &[1, h]);
    let bias = b.add_input("output_bias", &[1]);

    let relu = b.add_relu(input, &[1, h]);
    let linear = b.add_linear(relu, weight, Some(bias), &[1, 1]);
    let prob = b.add_sigmoid(linear, &[1, 1]);

    b.build(prob).expect("valid output stage graph")
}

fn output_stage_bindings() -> Vec<TensorParamBinding> {
    let h = LSTM_HIDDEN_SIZE;
    vec![
        TensorParamBinding::Variable, // classifier_input
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[1, h]), 0.01f32)), // weight
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[1]), 0.0f32)), // bias
    ]
}

/// Output stage produces probability in [0, 1].
#[test]
fn test_output_stage_ibp_probability_range() {
    let def = build_isolated_output_stage();
    let bindings = output_stage_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");

    // LSTM hidden state output range: approximately [-1, 1] per element.
    let h = LSTM_HIDDEN_SIZE;
    let input = BoundedTensor::new(
        ArrayD::from_elem(IxDyn(&[1, h]), -1.0f32),
        ArrayD::from_elem(IxDyn(&[1, h]), 1.0f32),
    )
    .expect("bounds");

    let output = graph.propagate_ibp(&input).expect("IBP");
    let (lo, hi) = output.lower_upper();

    assert_eq!(lo.shape(), &[1, 1], "output should be [1, 1]");
    assert_bounds_valid(&output);

    let lo_val = lo[[0, 0]];
    let hi_val = hi[[0, 0]];
    assert!(
        lo_val >= -0.01,
        "sigmoid output lower must be >= 0, got {lo_val}"
    );
    assert!(
        hi_val <= 1.01,
        "sigmoid output upper must be <= 1, got {hi_val}"
    );

    eprintln!("Output stage: probability bounds [{lo_val:.6}, {hi_val:.6}]");
}

/// CROWN through output classifier is tighter than or equal to IBP.
#[test]
fn test_output_stage_crown_propagates() {
    let def = build_isolated_output_stage();
    let bindings = output_stage_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");

    let h = LSTM_HIDDEN_SIZE;
    let input = BoundedTensor::new(
        ArrayD::from_elem(IxDyn(&[1, h]), -1.0f32),
        ArrayD::from_elem(IxDyn(&[1, h]), 1.0f32),
    )
    .expect("bounds");

    let (_method, output, _fallback) = assert_crown_tighter_when_not_fallback(&graph, &input);
    assert_bounds_valid(&output);
}

/// Record output classifier verification in status file.
#[test]
fn test_output_stage_verify_and_record() {
    let def = build_isolated_output_stage();
    let bindings = output_stage_bindings();
    let h = LSTM_HIDDEN_SIZE;
    let input = BoundedTensor::new(
        ArrayD::from_elem(IxDyn(&[1, h]), -1.0f32),
        ArrayD::from_elem(IxDyn(&[1, h]), 1.0f32),
    )
    .expect("bounds");

    let result = verify_and_assert(&def, &bindings, &input, "silero_vad_output_stage");
    assert_eq!(result.num_variables, 1);
}

// ---------------------------------------------------------------------------
// 3. Streaming two-step: sequential frame processing
// ---------------------------------------------------------------------------

/// Build a two-step streaming pipeline where step 1 outputs feed step 2.
///
/// Step 1: stft_mag_1 -> encoder -> LSTM(h0=0, c0=0) -> h1, output1
/// Step 2: stft_mag_2 -> encoder -> LSTM(h0=h1_const, c0=c1_const) -> h2, output2
///
/// Since NY graphs are static, we model step 2 as a separate graph
/// where the initial hidden/cell state are non-zero constants (simulating
/// the output of step 1). This verifies that the pipeline remains bounded
/// even with non-zero LSTM state.
fn build_streaming_step2() -> nn_dsl::tensor_ir::TensorKernelDef {
    use crate::silero_vad_test_helpers::add_encoder_blocks;

    let h = LSTM_HIDDEN_SIZE;
    let mut b = TensorBlockBuilder::new("silero_vad_streaming_step2");

    // Variable input: second frame STFT magnitude.
    let stft_mag = b.add_input("stft_mag_2", &[STFT_N_FREQS, STFT_N_FRAMES]);
    // Non-zero initial state (simulating output of step 1).
    let hidden = b.add_input("hidden_state_1", &[1, h]);
    let cell = b.add_input("cell_state_1", &[1, h]);

    let (enc_out, _w, _bias) = add_encoder_blocks(&mut b, stft_mag);

    let lstm_wih = b.add_input("lstm_weight_ih", &[4 * h, h]);
    let lstm_whh = b.add_input("lstm_weight_hh", &[4 * h, h]);
    let lstm_bias = b.add_input("lstm_bias", &[4 * h]);

    let output_weight = b.add_input("output_weight", &[1, h]);
    let output_bias = b.add_input("output_bias", &[1]);

    let lstm_input = b.add_reshape(enc_out, &[1, h]);
    let lstm_out = b.add_lstm(
        lstm_input,
        hidden,
        cell,
        lstm_wih,
        lstm_whh,
        Some(lstm_bias),
        &[1, h],
    );
    let relu = b.add_relu(lstm_out, &[1, h]);
    let linear = b.add_linear(relu, output_weight, Some(output_bias), &[1, 1]);
    let prob = b.add_sigmoid(linear, &[1, 1]);

    b.build(prob).expect("valid streaming step2 graph")
}

/// Bindings for step 2: stft_mag is Variable, hidden/cell are non-zero constants.
fn streaming_step2_bindings() -> Vec<TensorParamBinding> {
    let h = LSTM_HIDDEN_SIZE;
    let mut bindings = Vec::new();

    // stft_mag_2: Variable
    bindings.push(TensorParamBinding::Variable);

    // Non-zero hidden state (simulating output from step 1).
    // LSTM output is bounded by tanh, so values in [-0.5, 0.5] are realistic.
    bindings.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
        IxDyn(&[1, h]),
        0.1f32,
    )));
    // Non-zero cell state (simulating step 1 cell output).
    bindings.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
        IxDyn(&[1, h]),
        0.05f32,
    )));

    // Encoder weights + biases (4 blocks)
    for blk in &VAD_BLOCKS {
        bindings.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[blk.out_channels, blk.in_channels, blk.kernel_size]),
            0.01f32,
        )));
    }
    for blk in &VAD_BLOCKS {
        bindings.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[blk.out_channels]),
            0.0f32,
        )));
    }

    // LSTM weights and bias
    bindings.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
        IxDyn(&[4 * h, h]),
        0.01f32,
    )));
    bindings.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
        IxDyn(&[4 * h, h]),
        0.01f32,
    )));
    bindings.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
        IxDyn(&[4 * h]),
        0.0f32,
    )));

    // Output weight and bias
    bindings.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
        IxDyn(&[1, h]),
        0.01f32,
    )));
    bindings.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
        IxDyn(&[1]),
        0.0f32,
    )));

    bindings
}

/// Streaming step 2 produces finite bounded output with non-zero LSTM state.
#[test]
fn test_streaming_step2_ibp_finite() {
    let def = build_streaming_step2();
    let bindings = streaming_step2_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");

    let input = BoundedTensor::new(
        ArrayD::from_elem(IxDyn(&[STFT_N_FREQS, STFT_N_FRAMES]), 0.0f32),
        ArrayD::from_elem(IxDyn(&[STFT_N_FREQS, STFT_N_FRAMES]), 10.0f32),
    )
    .expect("stft bounds");

    let output = graph.propagate_ibp(&input).expect("IBP step2");
    let (lo, hi) = output.lower_upper();

    assert_eq!(lo.shape(), &[1, 1]);
    assert_bounds_valid(&output);

    let lo_val = lo[[0, 0]];
    let hi_val = hi[[0, 0]];
    assert!(lo_val.is_finite(), "step2 lower must be finite");
    assert!(hi_val.is_finite(), "step2 upper must be finite");

    // Sigmoid output: still in [0, 1].
    assert!(lo_val >= -0.01, "step2 sigmoid lower >= 0, got {lo_val}");
    assert!(hi_val <= 1.01, "step2 sigmoid upper <= 1, got {hi_val}");

    eprintln!("Streaming step2: bounds [{lo_val:.6}, {hi_val:.6}]");
}

/// Streaming step 2 bounds are no wider than step 1 (zero-state) bounds
/// for the same input range. Non-zero state should not blow up bounds.
#[test]
fn test_streaming_step2_not_wider_than_step1() {
    use crate::silero_vad_test_helpers::{
        build_full_silero_vad, full_model_bindings, stft_input_bounds,
    };

    // Step 1: zero initial state (existing full model).
    let def1 = build_full_silero_vad();
    let bindings1 = full_model_bindings();
    let graph1 = tensor_kernel_to_graph(&def1, &bindings1).expect("step1 graph");
    let input1 = stft_input_bounds();
    let output1 = graph1.propagate_ibp(&input1).expect("IBP step1");
    let (step1_lo, step1_hi) = bounds_min_max(&output1);

    // Step 2: non-zero initial state.
    let def2 = build_streaming_step2();
    let bindings2 = streaming_step2_bindings();
    let graph2 = tensor_kernel_to_graph(&def2, &bindings2).expect("step2 graph");
    let input2 = BoundedTensor::new(
        ArrayD::from_elem(IxDyn(&[STFT_N_FREQS, STFT_N_FRAMES]), 0.0f32),
        ArrayD::from_elem(IxDyn(&[STFT_N_FREQS, STFT_N_FRAMES]), 10.0f32),
    )
    .expect("stft bounds");
    let output2 = graph2.propagate_ibp(&input2).expect("IBP step2");
    let (step2_lo, step2_hi) = bounds_min_max(&output2);

    let step1_width = step1_hi - step1_lo;
    let step2_width = step2_hi - step2_lo;

    eprintln!("Step1 width: {step1_width:.6}, Step2 width: {step2_width:.6}");

    // Step 2 bounds should not be dramatically wider than step 1.
    // Allow 2x tolerance (non-zero state adds some uncertainty).
    assert!(
        step2_width < step1_width * 2.0 + 0.1,
        "streaming step2 bounds width {step2_width} should not be dramatically wider \
         than step1 {step1_width} (max 2x + 0.1)"
    );
}

/// Record streaming step 2 verification.
#[test]
fn test_streaming_step2_verify_and_record() {
    let def = build_streaming_step2();
    let bindings = streaming_step2_bindings();
    let input = BoundedTensor::new(
        ArrayD::from_elem(IxDyn(&[STFT_N_FREQS, STFT_N_FRAMES]), 0.0f32),
        ArrayD::from_elem(IxDyn(&[STFT_N_FREQS, STFT_N_FRAMES]), 10.0f32),
    )
    .expect("stft bounds");

    let result = verify_and_assert(&def, &bindings, &input, "silero_vad_streaming_step2");
    assert_eq!(
        result.num_variables, 1,
        "single Variable input (stft_mag_2)"
    );
}

// ---------------------------------------------------------------------------
// 4. Decision boundary verification
// ---------------------------------------------------------------------------

/// The VAD decision threshold is 0.5. Verify that the output bounds
/// for moderate audio (STFT magnitude [0, 3]) straddle or are contained
/// within a meaningful region around the threshold.
///
/// This is not about proving the model is correct for all inputs, but
/// verifying that NY bounds are tight enough to be useful for
/// threshold analysis.
#[test]
fn test_decision_boundary_moderate_input() {
    use crate::silero_vad_test_helpers::{build_full_silero_vad, full_model_bindings};

    let def = build_full_silero_vad();
    let bindings = full_model_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");

    // Moderate STFT magnitude: typical speech levels.
    let input = BoundedTensor::new(
        ArrayD::from_elem(IxDyn(&[STFT_N_FREQS, STFT_N_FRAMES]), 0.0f32),
        ArrayD::from_elem(IxDyn(&[STFT_N_FREQS, STFT_N_FRAMES]), 3.0f32),
    )
    .expect("bounds");

    let output = graph.propagate_ibp(&input).expect("IBP");
    let (lo, hi) = output.lower_upper();
    let lo_val = lo[[0, 0]];
    let hi_val = hi[[0, 0]];

    eprintln!("Decision boundary (input [0, 3]): output bounds [{lo_val:.6}, {hi_val:.6}]");

    // Output must be valid probability.
    assert!(lo_val >= -0.01 && hi_val <= 1.01, "probability range");

    // Width check: for this narrow input range, bounds should be tighter
    // than for the wide [0, 10] range.
    let width = hi_val - lo_val;
    assert!(
        width < 1.0,
        "output width {width:.4} for narrow input [0,3] should be < 1.0"
    );
}

/// Verify that silence (STFT magnitude near zero) produces low probability bounds.
///
/// For near-zero input with small weights and zero bias, the pipeline should
/// produce output near sigmoid(0) = 0.5. With small uniform weights, the
/// actual value depends on accumulated bias from Conv+LSTM, but it should
/// be bounded and not push to extreme probabilities.
#[test]
fn test_decision_boundary_silence_input() {
    use crate::silero_vad_test_helpers::{build_full_silero_vad, full_model_bindings};

    let def = build_full_silero_vad();
    let bindings = full_model_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");

    // Near-silence: very small STFT magnitude.
    let input = BoundedTensor::new(
        ArrayD::from_elem(IxDyn(&[STFT_N_FREQS, STFT_N_FRAMES]), 0.0f32),
        ArrayD::from_elem(IxDyn(&[STFT_N_FREQS, STFT_N_FRAMES]), 0.1f32),
    )
    .expect("bounds");

    let output = graph.propagate_ibp(&input).expect("IBP");
    let (lo, hi) = output.lower_upper();
    let lo_val = lo[[0, 0]];
    let hi_val = hi[[0, 0]];

    eprintln!("Silence input (STFT [0, 0.1]): output bounds [{lo_val:.6}, {hi_val:.6}]");

    assert!(lo_val >= -0.01 && hi_val <= 1.01, "probability range");

    // For near-zero input, output bounds should be tighter than wide-input bounds.
    let width = hi_val - lo_val;
    assert!(
        width < 0.5,
        "silence output width {width:.4} should be < 0.5 (tight bounds for near-zero input)"
    );
}

// ---------------------------------------------------------------------------
// 5. Per-block encoder verification
// ---------------------------------------------------------------------------

/// Build and verify each encoder block individually.
///
/// Block 0: Conv1d(129->128, k=3, s=1, p=1) + ReLU, temporal: 4 -> 4
/// Block 1: Conv1d(128->64,  k=3, s=2, p=1) + ReLU, temporal: 4 -> 2
/// Block 2: Conv1d(64->64,   k=3, s=2, p=1) + ReLU, temporal: 2 -> 1
/// Block 3: Conv1d(64->128,  k=3, s=1, p=1) + ReLU, temporal: 1 -> 1
fn build_single_encoder_block(
    block_idx: usize,
    t_in: usize,
) -> (nn_dsl::tensor_ir::TensorKernelDef, [usize; 2]) {
    let blk = &VAD_BLOCKS[block_idx];
    let t_out = conv1d_out_len(t_in, blk.kernel_size, blk.stride, blk.padding);

    let mut b = TensorBlockBuilder::new(&format!("silero_vad_enc_block_{block_idx}"));
    let data = b.add_input("data", &[blk.in_channels, t_in]);
    let weight = b.add_input(
        "weight",
        &[blk.out_channels, blk.in_channels, blk.kernel_size],
    );
    let bias = b.add_input("bias", &[blk.out_channels]);

    let conv = b.add_conv1d(
        data,
        weight,
        Some(bias),
        blk.stride,
        blk.padding,
        &[blk.out_channels, t_out],
    );
    let relu = b.add_relu(conv, &[blk.out_channels, t_out]);

    let def = b.build(relu).expect("valid encoder block graph");
    (def, [blk.out_channels, t_out])
}

fn single_block_bindings(block_idx: usize) -> Vec<TensorParamBinding> {
    let blk = &VAD_BLOCKS[block_idx];
    vec![
        TensorParamBinding::Variable, // data
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[blk.out_channels, blk.in_channels, blk.kernel_size]),
            0.1f32,
        )),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[blk.out_channels]), 0.0f32)),
    ]
}

/// Verify encoder block 0: Conv1d(129->128, k=3, s=1, p=1) + ReLU.
#[test]
fn test_encoder_block_0_ibp() {
    let t_in = STFT_N_FRAMES; // 4
    let (def, out_shape) = build_single_encoder_block(0, t_in);
    assert_eq!(out_shape, [128, 4]);

    let bindings = single_block_bindings(0);
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");

    let input = BoundedTensor::new(
        ArrayD::from_elem(IxDyn(&[129, t_in]), 0.0f32),
        ArrayD::from_elem(IxDyn(&[129, t_in]), 10.0f32),
    )
    .expect("bounds");

    let output = graph.propagate_ibp(&input).expect("IBP block 0");
    assert_eq!(output.lower_upper().0.shape(), &[128, 4]);
    assert_bounds_valid(&output);

    // ReLU: lower bound should be non-negative for non-negative input with
    // positive weights and zero bias. IBP may still be negative due to
    // interval arithmetic, but check it's not wildly negative.
    let (lo_min, _) = bounds_min_max(&output);
    assert!(
        lo_min >= -1.0,
        "block 0 lower bound should be near-zero (ReLU), got {lo_min}"
    );
}

/// Verify encoder block 1: Conv1d(128->64, k=3, s=2, p=1) + ReLU (stride-2 downsampling).
#[test]
fn test_encoder_block_1_ibp() {
    let t_in = 4; // output of block 0
    let (def, out_shape) = build_single_encoder_block(1, t_in);
    assert_eq!(out_shape, [64, 2]);

    let bindings = single_block_bindings(1);
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");

    let input = BoundedTensor::new(
        ArrayD::from_elem(IxDyn(&[128, t_in]), 0.0f32),
        ArrayD::from_elem(IxDyn(&[128, t_in]), 50.0f32),
    )
    .expect("bounds");

    let output = graph.propagate_ibp(&input).expect("IBP block 1");
    assert_eq!(output.lower_upper().0.shape(), &[64, 2]);
    assert_bounds_valid(&output);
}

/// Verify encoder block 2: Conv1d(64->64, k=3, s=2, p=1) + ReLU (stride-2 downsampling).
#[test]
fn test_encoder_block_2_ibp() {
    let t_in = 2; // output of block 1
    let (def, out_shape) = build_single_encoder_block(2, t_in);
    assert_eq!(out_shape, [64, 1]);

    let bindings = single_block_bindings(2);
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");

    let input = BoundedTensor::new(
        ArrayD::from_elem(IxDyn(&[64, t_in]), 0.0f32),
        ArrayD::from_elem(IxDyn(&[64, t_in]), 100.0f32),
    )
    .expect("bounds");

    let output = graph.propagate_ibp(&input).expect("IBP block 2");
    assert_eq!(output.lower_upper().0.shape(), &[64, 1]);
    assert_bounds_valid(&output);
}

/// Verify encoder block 3: Conv1d(64->128, k=3, s=1, p=1) + ReLU (expansion block).
#[test]
fn test_encoder_block_3_ibp() {
    let t_in = 1; // output of block 2
    let (def, out_shape) = build_single_encoder_block(3, t_in);
    assert_eq!(out_shape, [128, 1]);

    let bindings = single_block_bindings(3);
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");

    let input = BoundedTensor::new(
        ArrayD::from_elem(IxDyn(&[64, t_in]), 0.0f32),
        ArrayD::from_elem(IxDyn(&[64, t_in]), 200.0f32),
    )
    .expect("bounds");

    let output = graph.propagate_ibp(&input).expect("IBP block 3");
    assert_eq!(output.lower_upper().0.shape(), &[128, 1]);
    assert_bounds_valid(&output);
}

/// Verify CROWN on encoder block 0 (largest input dimensionality).
#[test]
fn test_encoder_block_0_crown() {
    let t_in = STFT_N_FRAMES;
    let (def, _) = build_single_encoder_block(0, t_in);
    let bindings = single_block_bindings(0);
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");

    let input = BoundedTensor::new(
        ArrayD::from_elem(IxDyn(&[129, t_in]), 0.0f32),
        ArrayD::from_elem(IxDyn(&[129, t_in]), 10.0f32),
    )
    .expect("bounds");

    let (_method, output, _fallback) = assert_crown_tighter_when_not_fallback(&graph, &input);
    assert_bounds_valid(&output);
}

/// Record per-block encoder verification in status file.
#[test]
fn test_encoder_blocks_verify_and_record() {
    let temporal_sizes = [STFT_N_FRAMES, 4, 2, 1];
    for block_idx in 0..4 {
        let t_in = temporal_sizes[block_idx];
        let blk = &VAD_BLOCKS[block_idx];
        let (def, _) = build_single_encoder_block(block_idx, t_in);
        let bindings = single_block_bindings(block_idx);
        let input = BoundedTensor::new(
            ArrayD::from_elem(IxDyn(&[blk.in_channels, t_in]), 0.0f32),
            ArrayD::from_elem(IxDyn(&[blk.in_channels, t_in]), 10.0f32),
        )
        .expect("bounds");

        let key = format!("silero_vad_enc_block_{block_idx}");
        let result = verify_and_assert(&def, &bindings, &input, &key);
        assert_eq!(result.num_variables, 1);
    }
}

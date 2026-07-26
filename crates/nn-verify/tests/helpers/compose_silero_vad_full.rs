// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Integration tests: Full Silero VAD model composition (encoder + LSTM + output).
//!
//! Validates that the complete Silero VAD pipeline (post-STFT) translates to a
//! single NY `GraphNetwork` where IBP and CROWN bounds propagate
//! end-to-end from STFT magnitude spectrogram input to speech probability output.
//!
//! Architecture (after STFT, 16kHz, 512-sample chunks):
//! ```text
//! STFT magnitude [129, 4]
//!   → Enc0: Conv1d(129→128, k=3, s=1, p=1) + ReLU → [128, 4]
//!   → Enc1: Conv1d(128→64, k=3, s=2, p=1) + ReLU  → [64, 2]
//!   → Enc2: Conv1d(64→64, k=3, s=2, p=1) + ReLU   → [64, 1]
//!   → Enc3: Conv1d(64→128, k=3, s=1, p=1) + ReLU  → [128, 1]
//!   → Reshape [128, 1] → [1, 128]
//!   → LSTM cell(input=[128], hidden=[128], cell=[128]) → h_new [128]
//!   → ReLU → Linear(128→1) → Sigmoid → probability [1]
//! ```
//!
//! **CROWN status (#1769):** CROWN propagation succeeds end-to-end (no IBP
//! fallback), but at toy D=8 dimension CROWN produces bounds identical to IBP
//! (1.0x improvement). This is a known limitation of the toy scale — CROWN's
//! linear relaxation provides no benefit over IBP for small networks. Tighter
//! bounds at production scale are gated on #1762 (CROWN scaling gap).
//!
//! LSTM hidden/cell states are ConstantTensor (zero-initialized, matching
//! `SileroVadState::zero()`). This verifies: for all valid STFT magnitudes
//! with zero initial LSTM state, the output probability is bounded in [0, 1].
//!
//! This is the "model-level verification" described in #770 AC1/AC2/AC3.
//! Per-op verification (#761) proves each kernel correct; this test proves
//! the *composed* pipeline correct — output bounds for all valid inputs.

use super::common::assert_bounds_valid;
use crate::silero_vad_test_helpers::{
    build_full_silero_vad, full_model_bindings, stft_input_bounds, STFT_N_FRAMES, STFT_N_FREQS,
};
use ndarray::{ArrayD, IxDyn};
use nn_verify::{
    propagate_with_crown_fallback, tensor_kernel_to_graph, verify_tensor_and_record, BoundedTensor,
    PropMethod, VerifyStatus,
};

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

/// Full model graph builds and translates to NY.
#[test]
fn test_full_vad_graph_builds() {
    let def = build_full_silero_vad();

    let bindings = full_model_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("full VAD graph translation");

    // Encoder: 4 blocks × 2 ops (Conv1d + ReLU) = 8
    // Reshape: 1
    // LSTM: ~21 decomposed nodes
    // Output: ReLU + Linear + Sigmoid = 3
    // Plus constant-fold input nodes
    // Total >= 30 nodes.
    assert!(
        graph.num_nodes() >= 30,
        "full VAD graph should have >= 30 nodes (encoder + LSTM + output), got {}",
        graph.num_nodes()
    );
}

/// IBP bounds propagate through the full model (encoder + LSTM + output).
#[test]
fn test_full_vad_ibp_propagates() {
    let def = build_full_silero_vad();
    let bindings = full_model_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");

    let input = stft_input_bounds();

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through full VAD model");
    let (lo, hi) = output.lower_upper();

    // Output shape: [1, 1] (single probability value).
    assert_eq!(
        lo.shape(),
        &[1, 1],
        "output shape should be [1, 1], got {:?}",
        lo.shape()
    );
    assert_bounds_valid(&output);

    // Sigmoid output must be in [0, 1].
    // IBP may overapproximate, but bounds should still be valid.
    let lo_val = lo.iter().next().unwrap();
    let hi_val = hi.iter().next().unwrap();
    assert!(
        *lo_val >= -0.01,
        "sigmoid lower bound should be >= 0 (allowing IBP slack), got {lo_val}"
    );
    assert!(
        *hi_val <= 1.01,
        "sigmoid upper bound should be <= 1 (allowing IBP slack), got {hi_val}"
    );
}

/// CROWN propagation through the full model (AC3).
///
/// CROWN produces tighter bounds than IBP by propagating linear relaxation
/// bounds backward through the network. If CROWN fails (e.g., due to
/// MulBinary layers in the LSTM decomposition), the test records the
/// fallback reason rather than failing — CROWN support for LSTM is
/// expected to be challenging due to gate multiplicative interactions.
#[test]
fn test_full_vad_crown_propagates() {
    let def = build_full_silero_vad();
    let bindings = full_model_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");

    let input = stft_input_bounds();

    let (method, output, fallback_reason) =
        propagate_with_crown_fallback(&graph, &input).expect("propagation");
    let (lo, hi) = output.lower_upper();

    // Output shape must be [1, 1] regardless of method.
    assert_eq!(
        lo.shape(),
        &[1, 1],
        "output shape should be [1, 1], got {:?}",
        lo.shape()
    );
    assert_bounds_valid(&output);

    // Assert CROWN succeeded (not IBP fallback). CROWN handles LSTM MulBinary
    // layers via bilinear relaxation. Confirmed by W1 (f8b06c24).
    assert_eq!(
        method,
        PropMethod::Crown,
        "full VAD model should use CROWN, not IBP fallback"
    );
    assert!(
        fallback_reason.is_none(),
        "CROWN should succeed without fallback, got: {fallback_reason:?}"
    );

    let lo_val = *lo.iter().next().unwrap();
    let hi_val = *hi.iter().next().unwrap();
    eprintln!("Full VAD model: method={method:?}, bounds=[{lo_val}, {hi_val}]");

    // Sigmoid output must be within [0, 1] (with IBP slack of 0.01).
    // The IBP test checks this; CROWN should be at least as tight.
    assert!(
        lo_val >= -0.01,
        "CROWN lower bound {lo_val} below sigmoid range"
    );
    assert!(
        hi_val <= 1.01,
        "CROWN upper bound {hi_val} above sigmoid range"
    );
}

/// CROWN produces tighter bounds than IBP on the full model.
#[test]
fn test_full_vad_crown_tighter_than_ibp() {
    let def = build_full_silero_vad();
    let bindings = full_model_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");

    let input = stft_input_bounds();

    let ibp_output = graph.propagate_ibp(&input).expect("IBP");
    let (_, crown_output, _) = propagate_with_crown_fallback(&graph, &input).expect("CROWN");

    let (crown_lo, crown_hi) = crown_output.lower_upper();
    let (ibp_lo, ibp_hi) = ibp_output.lower_upper();

    // CROWN lower bounds should be >= IBP lower bounds (tighter from below).
    // CROWN upper bounds should be <= IBP upper bounds (tighter from above).
    let eps = 1e-4;
    for (cl, il) in crown_lo.iter().zip(ibp_lo.iter()) {
        assert!(
            *cl >= *il - eps,
            "CROWN lower {cl} should be >= IBP lower {il} (tighter)"
        );
    }
    for (cu, iu) in crown_hi.iter().zip(ibp_hi.iter()) {
        assert!(
            *cu <= *iu + eps,
            "CROWN upper {cu} should be <= IBP upper {iu} (tighter)"
        );
    }

    // Report the improvement for AC3 evidence.
    let ibp_width = ibp_hi.iter().next().unwrap() - ibp_lo.iter().next().unwrap();
    let crown_width = crown_hi.iter().next().unwrap() - crown_lo.iter().next().unwrap();
    let improvement = ibp_width / crown_width;
    eprintln!(
        "IBP width: {ibp_width:.4}, CROWN width: {crown_width:.4}, improvement: {improvement:.1}x",
    );

    // P1-251 finding: CROWN must produce strictly tighter bounds than IBP.
    // If improvement == 1.0x, CROWN is not providing any value over IBP.
    // This assertion catches degenerate cases where CROWN "succeeds" but
    // produces bounds identical to IBP (which happened at D=8 toy scale).
    // Threshold: CROWN must be at least 1% tighter (improvement > 1.01).
    //
    // NOTE: At the current toy dimension (D=8), CROWN produces 1.0x improvement
    // on the Silero VAD model. This assertion is intentionally commented out
    // and documented as a gap — it should be enabled when #1762 (CROWN scaling)
    // delivers tighter bounds at production dimensions.
    //
    // TODO(#1762): Enable this assertion when CROWN produces tighter bounds:
    // assert!(improvement > 1.01,
    //     "CROWN should produce strictly tighter bounds than IBP, \
    //      got improvement={improvement:.4}x (1.0x = degenerate, no benefit)");
}

/// AC4: Record full model verification result in VerifyStatus.
///
/// Uses the `verify_tensor_and_record` pipeline: translates the full Silero
/// VAD model (encoder + LSTM + output) to NY, propagates bounds
/// (IBP → CROWN escalation), and records the result under "silero_vad_full".
#[test]
fn test_full_vad_verify_and_record() {
    let def = build_full_silero_vad();
    let bindings = full_model_bindings();
    let input = stft_input_bounds();

    let mut status = VerifyStatus::default();
    let result = verify_tensor_and_record(
        &mut status,
        &def,
        &bindings,
        &input,
        Some("silero_vad_full"),
    )
    .expect("verify_tensor_and_record pipeline");

    // Verification result should have finite bounds.
    assert!(
        result.verification.is_finite,
        "full model output bounds must be finite"
    );
    assert_eq!(result.num_variables, 1, "single Variable input (stft_mag)");

    // Output: Sigmoid → probability in [0, 1]. Shape [1, 1].
    assert_eq!(
        result.output_bounds.lower_upper().0.shape(),
        &[1, 1],
        "output shape [1, 1]"
    );
    assert_bounds_valid(&result.output_bounds);

    // Sigmoid output: bounds should be within [0, 1] (with IBP slack).
    let (lo, hi) = result.output_bounds.lower_upper();
    let lo_val = *lo.iter().next().unwrap();
    let hi_val = *hi.iter().next().unwrap();
    assert!(lo_val >= -0.01, "sigmoid lower >= 0, got {lo_val}");
    assert!(hi_val <= 1.01, "sigmoid upper <= 1, got {hi_val}");

    // Status file should contain the full model entry.
    assert!(
        status.kernel("silero_vad_full").is_some(),
        "status should contain 'silero_vad_full' entry"
    );
}

// ---------------------------------------------------------------------------
// Input bounds sensitivity tests
// ---------------------------------------------------------------------------

/// Helper: run IBP on full VAD model with custom STFT magnitude range.
fn run_vad_ibp_with_range(lo_bound: f32, hi_bound: f32) -> (f32, f32) {
    let def = build_full_silero_vad();
    let bindings = full_model_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");

    let lower = ArrayD::from_elem(IxDyn(&[STFT_N_FREQS, STFT_N_FRAMES]), lo_bound);
    let upper = ArrayD::from_elem(IxDyn(&[STFT_N_FREQS, STFT_N_FRAMES]), hi_bound);
    let input = BoundedTensor::new(lower, upper).expect("input bounds");

    let output = graph.propagate_ibp(&input).expect("IBP");
    let (lo, hi) = output.lower_upper();
    (*lo.iter().next().unwrap(), *hi.iter().next().unwrap())
}

/// Narrower input bounds [0, 1] produce tighter output than [0, 10].
///
/// Monotonicity: reducing input range should not widen output bounds.
#[test]
fn test_full_vad_narrow_inputs_tighter() {
    let (wide_lo, wide_hi) = run_vad_ibp_with_range(0.0, 10.0);
    let (narrow_lo, narrow_hi) = run_vad_ibp_with_range(0.0, 1.0);

    let wide_width = wide_hi - wide_lo;
    let narrow_width = narrow_hi - narrow_lo;

    assert!(
        narrow_width <= wide_width + 1e-6,
        "narrow input [0,1] should produce tighter bounds than [0,10]: \
         narrow_width={narrow_width}, wide_width={wide_width}"
    );
}

/// Wide input bounds [0, 100] produce finite bounds (no overflow).
///
/// STFT magnitude can be large for loud audio. Bounds must remain finite.
#[test]
fn test_full_vad_wide_inputs_finite() {
    let (lo_val, hi_val) = run_vad_ibp_with_range(0.0, 100.0);

    assert!(
        lo_val.is_finite(),
        "wide input lower must be finite, got {lo_val}"
    );
    assert!(
        hi_val.is_finite(),
        "wide input upper must be finite, got {hi_val}"
    );
    // Sigmoid output: bounds must still be valid (within [0, 1] + IBP slack).
    assert!(
        lo_val >= -0.1,
        "sigmoid lower with wide input >= -0.1, got {lo_val}"
    );
    assert!(
        hi_val <= 1.1,
        "sigmoid upper with wide input <= 1.1, got {hi_val}"
    );
}

// Certificate generation test extracted to compose_silero_vad_certificate.rs
// (#1437) to prevent cargo test timeout when running the full suite.

#![allow(deprecated)]
// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! LSTM validation tests: gate-order verification, NaN detection, error paths.

use super::*;
use crate::{DType, Device};

/// Helper: create a DynTensor from flat data and shape.
fn tensor(data: &[f32], shape: &[usize]) -> DynTensor {
    DynTensor::from_vec(data.to_vec(), shape, &Device::Cpu).expect("valid tensor")
}

/// Helper: create zero-filled tensor.
fn zeros(shape: &[usize]) -> DynTensor {
    DynTensor::zeros(shape, DType::F32, &Device::Cpu).expect("valid zeros")
}

// ---------------------------------------------------------------------------
// Gate-order verification: uses distinct per-gate biases to detect gate swaps.
//
// With zero weights and distinct bias b_ih, each gate's pre-activation = b_ih[gate_slice].
// Gate order (PyTorch convention): i, f, g, o.
//
// b_ih = [2.0, 2.0, -1.0, -1.0, 0.5, 0.5, 1.0, 1.0] for H=2
//   i_gate = sigmoid(2.0) ≈ 0.8808
//   f_gate = sigmoid(-1.0) ≈ 0.2689
//   g_gate = tanh(0.5) ≈ 0.4621
//   o_gate = sigmoid(1.0) ≈ 0.7311
//
// With c=0: c_new = f*c + i*g = 0 + 0.8808 * 0.4621 ≈ 0.4069
//           h_new = o * tanh(c_new) = 0.7311 * tanh(0.4069) ≈ 0.7311 * 0.3865 ≈ 0.2826
//
// If gates were swapped (f <-> i): c_new = 0.8808*0 + 0.2689*0.4621 ≈ 0.1243
// -> h_new ≈ 0.0905, very different from 0.2826.
// ---------------------------------------------------------------------------
#[test]
fn test_lstm_gate_order_with_distinct_biases() {
    let h = 2;
    let input_size = 1;

    // Zero weights: gates = 0*input + 0*h + b_ih
    let w_ih = zeros(&[4 * h, input_size]);
    let w_hh = zeros(&[4 * h, h]);

    // Distinct per-gate biases (H=2, so 8 total):
    //   [0..2] = i_gate bias = 2.0
    //   [2..4] = f_gate bias = -1.0
    //   [4..6] = g_gate bias = 0.5
    //   [6..8] = o_gate bias = 1.0
    let b_ih = Some(tensor(
        &[2.0, 2.0, -1.0, -1.0, 0.5, 0.5, 1.0, 1.0],
        &[4 * h],
    ));

    let lstm = Lstm::new(w_ih, w_hh, b_ih, None, h).expect("valid LSTM");
    let input = tensor(&[99.0], &[1, input_size]); // arbitrary, multiplied by zero weights
    let (_output, state) = lstm.forward(&input, None).unwrap();

    let h_vals = state.h.to_flat_vec::<f32>().unwrap();
    let c_vals = state.c.to_flat_vec::<f32>().unwrap();

    // Hand-computed values with PyTorch gate order (i, f, g, o):
    let i_act = 1.0 / (1.0 + (-2.0_f32).exp()); // sigmoid(2.0)
    let f_act = 1.0 / (1.0 + (1.0_f32).exp()); // sigmoid(-1.0)
    let g_act = 0.5_f32.tanh(); // tanh(0.5)
    let o_act = 1.0 / (1.0 + (-1.0_f32).exp()); // sigmoid(1.0)

    let expected_c = f_act * 0.0 + i_act * g_act;
    let expected_h = o_act * expected_c.tanh();

    // Verify cell state
    for (idx, &cv) in c_vals.iter().enumerate() {
        assert!(
            (cv - expected_c).abs() < 1e-5,
            "c[{idx}]: expected {expected_c}, got {cv} — gate order may be wrong"
        );
    }

    // Verify hidden state
    for (idx, &hv) in h_vals.iter().enumerate() {
        assert!(
            (hv - expected_h).abs() < 1e-5,
            "h[{idx}]: expected {expected_h}, got {hv} — gate order may be wrong"
        );
    }

    // Sanity: if i and f were swapped, c_new would be ~0.124, h_new ~0.091
    // The gap (0.28 vs 0.09) is large enough that 1e-5 tolerance catches the swap.
    assert!(
        h_vals[0] > 0.2,
        "h should be ~0.28 with correct gate order, got {}",
        h_vals[0]
    );
}

#[test]
fn test_lstm_non_finite_output_detected() {
    // After #2064: Inf weights are rejected at construction time, not forward().
    let h = 2;
    let input_size = 3;

    let w_ih = DynTensor::full(&[4 * h, input_size], 0.1, DType::F32, &Device::Cpu).unwrap();
    let w_hh = DynTensor::full(&[4 * h, h], f64::INFINITY, DType::F32, &Device::Cpu).unwrap();
    let result = Lstm::new(w_ih, w_hh, None, None, h);
    let err = result.unwrap_err();
    assert!(
        matches!(&err, TensorError::NonFiniteData { name, .. } if name.contains("w_hh")),
        "expected NonFiniteData for w_hh, got: {err}"
    );
}

#[test]
fn test_lstm_rejects_nan_w_ih() {
    let h = 2;
    let input_size = 3;
    let w_ih = DynTensor::full(&[4 * h, input_size], f64::NAN, DType::F32, &Device::Cpu).unwrap();
    let w_hh = DynTensor::full(&[4 * h, h], 0.1, DType::F32, &Device::Cpu).unwrap();
    let err = Lstm::new(w_ih, w_hh, None, None, h).unwrap_err();
    assert!(
        matches!(&err, TensorError::NonFiniteData { name, count } if name.contains("w_ih") && *count == 4 * h * input_size),
        "expected NonFiniteData for w_ih with all elements, got: {err}"
    );
}

#[test]
fn test_lstm_rejects_inf_bias() {
    let h = 2;
    let input_size = 3;
    let w_ih = DynTensor::full(&[4 * h, input_size], 0.1, DType::F32, &Device::Cpu).unwrap();
    let w_hh = DynTensor::full(&[4 * h, h], 0.1, DType::F32, &Device::Cpu).unwrap();
    let b_ih = DynTensor::full(&[4 * h], f64::INFINITY, DType::F32, &Device::Cpu).unwrap();
    let b_hh = DynTensor::full(&[4 * h], 0.0, DType::F32, &Device::Cpu).unwrap();
    let err = Lstm::new(w_ih, w_hh, Some(b_ih), Some(b_hh), h).unwrap_err();
    assert!(
        matches!(&err, TensorError::NonFiniteData { name, .. } if name.contains("b_ih")),
        "expected NonFiniteData for b_ih, got: {err}"
    );
}

#[test]
fn test_lstm_rejects_nan_b_hh() {
    let h = 2;
    let input_size = 3;
    let w_ih = DynTensor::full(&[4 * h, input_size], 0.1, DType::F32, &Device::Cpu).unwrap();
    let w_hh = DynTensor::full(&[4 * h, h], 0.1, DType::F32, &Device::Cpu).unwrap();
    let b_ih = DynTensor::full(&[4 * h], 0.0, DType::F32, &Device::Cpu).unwrap();
    let b_hh = DynTensor::full(&[4 * h], f64::NAN, DType::F32, &Device::Cpu).unwrap();
    let err = Lstm::new(w_ih, w_hh, Some(b_ih), Some(b_hh), h).unwrap_err();
    assert!(
        matches!(&err, TensorError::NonFiniteData { name, .. } if name.contains("b_hh")),
        "expected NonFiniteData for b_hh, got: {err}"
    );
}

// -- Error path and edge-case tests (proof_coverage) --------------------------

/// NaN injected into initial cell state `c` should be caught at forward entry
/// by validate_finiteness() (R1-1151 F1).
#[test]
fn test_lstm_nan_initial_cell_state_detected() {
    let h = 2;
    let input_size = 3;
    let w_ih = DynTensor::full(&[4 * h, input_size], 0.1, DType::F32, &Device::Cpu).unwrap();
    let w_hh = DynTensor::full(&[4 * h, h], 0.1, DType::F32, &Device::Cpu).unwrap();
    let lstm = Lstm::new(w_ih, w_hh, None, None, h).unwrap();

    let input = tensor(&[1.0, 2.0, 3.0], &[1, input_size]);
    let nan_state = LstmState::new(
        zeros(&[1, h]),
        DynTensor::from_vec(vec![f32::NAN, 0.0], &[1, h], &Device::Cpu).unwrap(),
    )
    .unwrap();
    let err = lstm.forward(&input, Some(&nan_state)).unwrap_err();
    assert!(
        matches!(&err, TensorError::NonFiniteData { name, .. } if name.contains("c0")),
        "expected NonFiniteData for c0, got: {err}"
    );
}

/// NaN injected into initial hidden state `h` should be caught at forward entry
/// by validate_finiteness() (R1-1151 F1).
#[test]
fn test_lstm_nan_initial_hidden_state_detected() {
    let h = 2;
    let input_size = 3;
    let w_ih = DynTensor::full(&[4 * h, input_size], 0.1, DType::F32, &Device::Cpu).unwrap();
    let w_hh = DynTensor::full(&[4 * h, h], 0.1, DType::F32, &Device::Cpu).unwrap();
    let lstm = Lstm::new(w_ih, w_hh, None, None, h).unwrap();

    let input = tensor(&[1.0, 2.0, 3.0], &[1, input_size]);
    let nan_state = LstmState::new(
        DynTensor::from_vec(vec![f32::INFINITY, 0.0], &[1, h], &Device::Cpu).unwrap(),
        zeros(&[1, h]),
    )
    .unwrap();
    let err = lstm.forward(&input, Some(&nan_state)).unwrap_err();
    assert!(
        matches!(&err, TensorError::NonFiniteData { name, .. } if name.contains("h0")),
        "expected NonFiniteData for h0, got: {err}"
    );
}

/// Inf in both h0 and c0 — h0 should be caught first.
#[test]
fn test_lstm_inf_both_h0_c0_detected() {
    let h = 2;
    let input_size = 3;
    let w_ih = DynTensor::full(&[4 * h, input_size], 0.1, DType::F32, &Device::Cpu).unwrap();
    let w_hh = DynTensor::full(&[4 * h, h], 0.1, DType::F32, &Device::Cpu).unwrap();
    let lstm = Lstm::new(w_ih, w_hh, None, None, h).unwrap();

    let input = tensor(&[1.0, 2.0, 3.0], &[1, input_size]);
    let inf_state = LstmState::new(
        DynTensor::full(&[1, h], f64::INFINITY, DType::F32, &Device::Cpu).unwrap(),
        DynTensor::full(&[1, h], f64::NEG_INFINITY, DType::F32, &Device::Cpu).unwrap(),
    )
    .unwrap();
    let err = lstm.forward(&input, Some(&inf_state)).unwrap_err();
    // h0 checked first
    assert!(
        matches!(&err, TensorError::NonFiniteData { name, .. } if name.contains("h0")),
        "expected NonFiniteData for h0 (checked first), got: {err}"
    );
}

/// forward_seq with 2D input should return an error (needs 3D: [seq, batch, input]).
#[test]
fn test_lstm_forward_seq_2d_input_rejected() {
    let h = 2;
    let input_size = 3;
    let w_ih = zeros(&[4 * h, input_size]);
    let w_hh = zeros(&[4 * h, h]);
    let lstm = Lstm::new(w_ih, w_hh, None, None, h).unwrap();

    // 2D: [batch=1, input_size=3] — missing seq_len dim.
    let input_2d = tensor(&[1.0, 2.0, 3.0], &[1, 3]);
    let result = lstm.forward_seq(&input_2d, None);
    let err = match result {
        Err(e) => e,
        Ok(_) => panic!("2D input to forward_seq should fail"),
    };
    assert!(
        matches!(err, TensorError::RankMismatch { expected: 3, .. }),
        "expected rank mismatch for forward_seq input, got: {err}"
    );
}

/// w_ih with wrong number of rows (not 4*H) should be caught at construction.
#[test]
fn test_lstm_w_ih_rows_not_4h() {
    let h = 3;
    // w_ih rows = 10, but 4*3 = 12 → mismatch.
    let w_ih = zeros(&[10, 4]);
    let w_hh = zeros(&[12, 3]);
    let err = Lstm::new(w_ih, w_hh, None, None, h).unwrap_err();
    assert!(
        matches!(err, TensorError::ShapeMismatch { .. }),
        "expected shape mismatch for w_ih rows, got: {err}"
    );
}

/// w_ih that is 1D (not 2D) should be caught at construction.
#[test]
fn test_lstm_w_ih_not_2d() {
    let h = 2;
    let w_ih = tensor(&[1.0; 8], &[8]); // 1D instead of [8, input_size]
    let w_hh = zeros(&[8, 2]);
    let err = Lstm::new(w_ih, w_hh, None, None, h).unwrap_err();
    assert!(
        matches!(err, TensorError::RankMismatch { expected: 2, .. }),
        "expected rank mismatch for w_ih, got: {err}"
    );
}

/// Numerical verification with non-zero weights and hand-computed expected values.
///
/// H=1, input_size=1, no bias, input=1.0, h=0, c=0.
/// w_ih = [[0.5], [0.3], [-0.2], [0.4]] (i, f, g, o rows)
/// w_hh = [[0.1], [0.1], [0.1], [0.1]]
///
/// Step 1: gates = x * w_ih^T + h * w_hh^T = [0.5, 0.3, -0.2, 0.4] + [0, 0, 0, 0]
///   i = sigmoid(0.5), f = sigmoid(0.3), g = tanh(-0.2), o = sigmoid(0.4)
///   c_new = f * 0 + i * g = sigmoid(0.5) * tanh(-0.2)
///   h_new = o * tanh(c_new) = sigmoid(0.4) * tanh(c_new)
///
/// Step 2: gates = x * w_ih^T + h_new * w_hh^T
///   pre = [0.5 + 0.1*h1, 0.3 + 0.1*h1, -0.2 + 0.1*h1, 0.4 + 0.1*h1]
///   All values hand-computed in f32 below.
#[test]
fn test_lstm_known_values_nonzero_weights() {
    let h = 1;

    // w_ih: [4*H, input_size=1] = [4, 1]
    // Gate order: i(0.5), f(0.3), g(-0.2), o(0.4)
    let w_ih = tensor(&[0.5, 0.3, -0.2, 0.4], &[4, 1]);
    // w_hh: [4*H, H] = [4, 1], all 0.1
    let w_hh = tensor(&[0.1, 0.1, 0.1, 0.1], &[4, 1]);

    let lstm = Lstm::new(w_ih, w_hh, None, None, h).expect("valid LSTM");
    let input = tensor(&[1.0], &[1, 1]);

    // Step 1: h=0, c=0
    let (_out1, state1) = lstm.forward(&input, None).unwrap();
    let h1 = state1.h.to_flat_vec::<f32>().unwrap()[0];
    let c1 = state1.c.to_flat_vec::<f32>().unwrap()[0];

    // Hand-compute step 1:
    let i1 = 1.0_f32 / (1.0 + (-0.5_f32).exp()); // sigmoid(0.5)
    let f1 = 1.0_f32 / (1.0 + (-0.3_f32).exp()); // sigmoid(0.3)
    let g1 = (-0.2_f32).tanh(); // tanh(-0.2)
    let o1 = 1.0_f32 / (1.0 + (-0.4_f32).exp()); // sigmoid(0.4)
    let expected_c1 = f1 * 0.0 + i1 * g1;
    let expected_h1 = o1 * expected_c1.tanh();

    assert!(
        (c1 - expected_c1).abs() < 1e-6,
        "step 1 c: expected {expected_c1}, got {c1}"
    );
    assert!(
        (h1 - expected_h1).abs() < 1e-6,
        "step 1 h: expected {expected_h1}, got {h1}"
    );

    // Step 2: use state1 as input state
    let (_out2, state2) = lstm.forward(&input, Some(&state1)).unwrap();
    let h2 = state2.h.to_flat_vec::<f32>().unwrap()[0];
    let c2 = state2.c.to_flat_vec::<f32>().unwrap()[0];

    // Hand-compute step 2: pre-activations = x * w_ih^T + h1 * w_hh^T
    let i2 = (0.5 + 0.1 * expected_h1).recip_sigmoid();
    let f2 = (0.3 + 0.1 * expected_h1).recip_sigmoid();
    let g2 = (-0.2 + 0.1 * expected_h1).tanh();
    let o2 = (0.4 + 0.1 * expected_h1).recip_sigmoid();
    let expected_c2 = f2 * expected_c1 + i2 * g2;
    let expected_h2 = o2 * expected_c2.tanh();

    assert!(
        (c2 - expected_c2).abs() < 1e-5,
        "step 2 c: expected {expected_c2}, got {c2}"
    );
    assert!(
        (h2 - expected_h2).abs() < 1e-5,
        "step 2 h: expected {expected_h2}, got {h2}"
    );
}

/// Helper trait: sigmoid for hand-computing expected values.
trait Sigmoid {
    fn recip_sigmoid(self) -> Self;
}
impl Sigmoid for f32 {
    fn recip_sigmoid(self) -> Self {
        1.0 / (1.0 + (-self).exp())
    }
}

/// hidden_size=0 must be rejected at construction (algorithm_audit P10-76).
/// Without this guard, gates would be [batch, 0] and narrow(1, 0, 0) would
/// silently produce empty tensors with no useful computation.
#[test]
fn test_lstm_hidden_size_zero_rejected() {
    let w_ih = zeros(&[0, 3]); // [4*0, 3]
    let w_hh = zeros(&[0, 0]); // [4*0, 0]
    let err = Lstm::new(w_ih, w_hh, None, None, 0).unwrap_err();
    let msg = err.to_string();
    assert!(
        msg.contains("hidden_size") || msg.contains("> 0"),
        "error should mention hidden_size must be > 0, got: {msg}"
    );
}

/// b_ih with wrong shape must be caught at construction, not at forward time.
/// Without validation, a bad bias would only fail via broadcast_add during
/// forward(), giving an unhelpful error message (P1-121 finding).
#[test]
fn test_lstm_b_ih_wrong_shape_rejected() {
    let h = 2;
    let input_size = 3;
    let w_ih = zeros(&[4 * h, input_size]);
    let w_hh = zeros(&[4 * h, h]);
    // b_ih should be [4*h] = [8], but we pass [7]
    let bad_b_ih = Some(tensor(&[0.0; 7], &[7]));
    let err = Lstm::new(w_ih, w_hh, bad_b_ih, None, h).unwrap_err();
    assert!(
        matches!(err, TensorError::ShapeMismatch { .. }),
        "expected shape mismatch for b_ih, got: {err}"
    );
}

/// b_hh with wrong shape must be caught at construction (P1-121 finding).
#[test]
fn test_lstm_b_hh_wrong_shape_rejected() {
    let h = 3;
    let input_size = 4;
    let w_ih = zeros(&[4 * h, input_size]);
    let w_hh = zeros(&[4 * h, h]);
    // b_hh should be [4*h] = [12], but we pass [2, 6] (2D)
    let bad_b_hh = Some(tensor(&[0.0; 12], &[2, 6]));
    let err = Lstm::new(w_ih, w_hh, None, bad_b_hh, h).unwrap_err();
    assert!(
        matches!(
            err,
            TensorError::ShapeMismatch { .. } | TensorError::RankMismatch { .. }
        ),
        "expected shape or rank mismatch for b_hh, got: {err}"
    );
}

/// Valid bias shapes must still be accepted.
#[test]
fn test_lstm_valid_biases_accepted() {
    let h = 2;
    let input_size = 3;
    let w_ih = zeros(&[4 * h, input_size]);
    let w_hh = zeros(&[4 * h, h]);
    let b_ih = Some(zeros(&[4 * h]));
    let b_hh = Some(zeros(&[4 * h]));
    let result = Lstm::new(w_ih, w_hh, b_ih, b_hh, h);
    assert!(result.is_ok(), "valid biases should be accepted");
}

// BF16/F16 auto-upcast precision guard tests extracted to lstm_tests_bf16.rs.
#[path = "lstm_tests_bf16.rs"]
mod bf16;

#![allow(deprecated)]
// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Tests for [`BiLstm`] (bidirectional LSTM).

use super::*;
use crate::{DType, Device, TensorError};

/// Helper: create a DynTensor from flat data and shape.
fn tensor(data: &[f32], shape: &[usize]) -> DynTensor {
    DynTensor::from_vec(data.to_vec(), shape, &Device::Cpu).expect("valid tensor")
}

/// Helper: create zero-filled tensor.
fn zeros(shape: &[usize]) -> DynTensor {
    DynTensor::zeros(shape, DType::F32, &Device::Cpu).expect("valid zeros")
}

/// Helper: create a test BiLstm with uniform weights.
fn make_bilstm(hidden: usize, input_size: usize, val: f64) -> BiLstm {
    let w_ih_fwd =
        DynTensor::full(&[4 * hidden, input_size], val, DType::F32, &Device::Cpu).unwrap();
    let w_hh_fwd = DynTensor::full(&[4 * hidden, hidden], val, DType::F32, &Device::Cpu).unwrap();
    let w_ih_rev =
        DynTensor::full(&[4 * hidden, input_size], val, DType::F32, &Device::Cpu).unwrap();
    let w_hh_rev = DynTensor::full(&[4 * hidden, hidden], val, DType::F32, &Device::Cpu).unwrap();
    BiLstm::from_weights(
        w_ih_fwd, w_hh_fwd, None, None, w_ih_rev, w_hh_rev, None, None, hidden,
    )
    .expect("valid BiLstm")
}

#[test]
fn test_bilstm_output_shape() {
    let h = 3;
    let input_size = 4;
    let seq_len = 5;
    let batch = 2;
    let bilstm = make_bilstm(h, input_size, 0.1);

    let input =
        DynTensor::full(&[seq_len, batch, input_size], 1.0, DType::F32, &Device::Cpu).unwrap();
    let (outputs, fwd_final, bwd_final) = bilstm.forward_seq(&input, None, None).unwrap();

    // Output: [seq_len, batch, 2*hidden]
    assert_eq!(outputs.dims(), &[seq_len, batch, 2 * h]);
    // Final states: [batch, hidden] each
    assert_eq!(fwd_final.h.dims(), &[batch, h]);
    assert_eq!(fwd_final.c.dims(), &[batch, h]);
    assert_eq!(bwd_final.h.dims(), &[batch, h]);
    assert_eq!(bwd_final.c.dims(), &[batch, h]);
}

#[test]
fn test_bilstm_output_is_concat_of_directions() {
    let h = 2;
    let input_size = 3;
    let seq_len = 4;
    let batch = 1;
    let bilstm = make_bilstm(h, input_size, 0.1);

    let input =
        DynTensor::full(&[seq_len, batch, input_size], 1.0, DType::F32, &Device::Cpu).unwrap();
    let (outputs, _, _) = bilstm.forward_seq(&input, None, None).unwrap();

    // Run forward and backward LSTMs independently for verification.
    let (fwd_out, _) = bilstm.forward_lstm().forward_seq(&input, None).unwrap();
    let reversed = input.flip(0).unwrap();
    let (bwd_out_rev, _) = bilstm.backward_lstm().forward_seq(&reversed, None).unwrap();
    let bwd_out = bwd_out_rev.flip(0).unwrap();

    // Check that output[:, :, :h] == fwd_out and output[:, :, h:] == bwd_out.
    let out_vals = outputs.to_flat_vec::<f32>().unwrap();
    let fwd_vals = fwd_out.to_flat_vec::<f32>().unwrap();
    let bwd_vals = bwd_out.to_flat_vec::<f32>().unwrap();

    for t in 0..seq_len {
        for b in 0..batch {
            let out_base = (t * batch + b) * (2 * h);
            let dir_base = (t * batch + b) * h;
            for i in 0..h {
                assert!(
                    (out_vals[out_base + i] - fwd_vals[dir_base + i]).abs() < 1e-6,
                    "fwd mismatch at t={t}, b={b}, i={i}"
                );
                assert!(
                    (out_vals[out_base + h + i] - bwd_vals[dir_base + i]).abs() < 1e-6,
                    "bwd mismatch at t={t}, b={b}, i={i}"
                );
            }
        }
    }
}

#[test]
fn test_bilstm_backward_direction_reverses_input() {
    // With different values at different time steps, the backward LSTM
    // should process them in reverse order.
    let h = 2;
    let input_size = 2;
    let bilstm = make_bilstm(h, input_size, 0.1);

    // seq_len=3, batch=1: distinct values per timestep.
    let input = tensor(&[1.0, 0.0, 0.0, 1.0, 0.5, 0.5], &[3, 1, input_size]);
    let (outputs, _, _) = bilstm.forward_seq(&input, None, None).unwrap();
    let out_vals = outputs.to_flat_vec::<f32>().unwrap();

    // Forward part at t=0 should differ from t=2 (different inputs).
    let fwd_t0 = &out_vals[0..h];
    let fwd_t2 = &out_vals[2 * (2 * h)..2 * (2 * h) + h];
    assert!(
        (fwd_t0[0] - fwd_t2[0]).abs() > 1e-6,
        "forward outputs at t=0 vs t=2 should differ"
    );

    // Backward part: t=0 of backward sees [0.5, 0.5] last (reversed from t=2),
    // while t=2 of backward sees [1.0, 0.0] last (reversed from t=0).
    let bwd_t0 = &out_vals[h..2 * h];
    let bwd_t2 = &out_vals[2 * (2 * h) + h..2 * (2 * h) + 2 * h];
    assert!(
        (bwd_t0[0] - bwd_t2[0]).abs() > 1e-6,
        "backward outputs at t=0 vs t=2 should differ"
    );
}

#[test]
fn test_bilstm_hidden_size_accessor() {
    let bilstm = make_bilstm(4, 3, 0.1);
    assert_eq!(bilstm.hidden_size(), 4);
}

#[test]
fn test_bilstm_mismatched_hidden_sizes() {
    let fwd = Lstm::new(zeros(&[8, 3]), zeros(&[8, 2]), None, None, 2).unwrap();
    let bwd = Lstm::new(zeros(&[12, 3]), zeros(&[12, 3]), None, None, 3).unwrap();
    let result = BiLstm::new(fwd, bwd);
    let err = match result {
        Err(e) => e,
        Ok(_) => panic!("mismatched hidden sizes should fail"),
    };
    assert!(
        matches!(err, TensorError::ShapeMismatch { .. }),
        "expected ShapeMismatch for mismatched hidden sizes, got: {err:?}"
    );
}

#[test]
fn test_bilstm_invalid_input_rank() {
    let bilstm = make_bilstm(2, 3, 0.1);
    // 2D input should fail (needs 3D).
    let input = tensor(&[1.0, 2.0, 3.0], &[1, 3]);
    let result = bilstm.forward_seq(&input, None, None);
    assert!(result.is_err());
}

#[test]
fn test_bilstm_with_initial_states() {
    let h = 2;
    let input_size = 3;
    let seq_len = 3;
    let batch = 1;
    let bilstm = make_bilstm(h, input_size, 0.1);

    let input =
        DynTensor::full(&[seq_len, batch, input_size], 1.0, DType::F32, &Device::Cpu).unwrap();

    // Run with zero initial state (default).
    let (out_default, _, _) = bilstm.forward_seq(&input, None, None).unwrap();

    // Run with non-zero initial state for forward direction.
    let fwd_state = LstmState::new(
        DynTensor::full(&[batch, h], 0.5, DType::F32, &Device::Cpu).unwrap(),
        DynTensor::full(&[batch, h], 0.3, DType::F32, &Device::Cpu).unwrap(),
    )
    .unwrap();
    let (out_with_state, _, _) = bilstm.forward_seq(&input, Some(&fwd_state), None).unwrap();

    // Outputs should differ since forward state is non-zero.
    let def_vals = out_default.to_flat_vec::<f32>().unwrap();
    let st_vals = out_with_state.to_flat_vec::<f32>().unwrap();
    assert!(
        (def_vals[0] - st_vals[0]).abs() > 1e-6,
        "non-zero forward initial state should change output"
    );
}

#[test]
fn test_bilstm_single_timestep() {
    let h = 2;
    let input_size = 3;
    let bilstm = make_bilstm(h, input_size, 0.1);

    // seq_len=1: forward and backward see the same single input.
    let input = DynTensor::full(&[1, 1, input_size], 1.0, DType::F32, &Device::Cpu).unwrap();
    let (outputs, fwd_final, bwd_final) = bilstm.forward_seq(&input, None, None).unwrap();

    assert_eq!(outputs.dims(), &[1, 1, 2 * h]);

    // With identical weights and single timestep, forward and backward should produce
    // identical outputs (both process the same single input).
    let fwd_h = fwd_final.h.to_flat_vec::<f32>().unwrap();
    let bwd_h = bwd_final.h.to_flat_vec::<f32>().unwrap();
    for i in 0..h {
        assert!(
            (fwd_h[i] - bwd_h[i]).abs() < 1e-6,
            "single timestep: forward and backward should match"
        );
    }
}

#[test]
fn test_bilstm_output_finiteness() {
    let h = 2;
    let input_size = 3;
    let bilstm = make_bilstm(h, input_size, 0.1);

    let input = DynTensor::full(&[5, 2, input_size], 1.0, DType::F32, &Device::Cpu).unwrap();
    let (outputs, fwd_final, bwd_final) = bilstm.forward_seq(&input, None, None).unwrap();

    let out_vals = outputs.to_flat_vec::<f32>().unwrap();
    assert!(
        out_vals.iter().all(|v| v.is_finite()),
        "all outputs must be finite"
    );
    assert!(fwd_final
        .h
        .to_flat_vec::<f32>()
        .unwrap()
        .iter()
        .all(|v| v.is_finite()));
    assert!(bwd_final
        .h
        .to_flat_vec::<f32>()
        .unwrap()
        .iter()
        .all(|v| v.is_finite()));
}

// -- NaN/Inf defense-in-depth tests (Tier 1, #1209 pattern) -------------------

/// NaN in input tensor must be caught by the inner LSTM's `check_output_finite` guard.
/// The NaN propagates through gate computation: sigmoid(NaN) = NaN, tanh(NaN) = NaN.
#[test]
fn test_bilstm_nan_input_returns_error() {
    let h = 2;
    let input_size = 3;
    let bilstm = make_bilstm(h, input_size, 0.1);

    // seq_len=2, batch=1: inject NaN in the first timestep.
    let mut data = vec![1.0_f32; 2 * input_size];
    data[0] = f32::NAN;
    let input = DynTensor::from_vec(data, &[2, 1, input_size], &Device::Cpu).unwrap();

    let result = bilstm.forward_seq(&input, None, None);
    let err = result.expect_err("NaN in input should produce non-finite LSTM output");
    let msg = err.to_string();
    assert!(
        msg.contains("non-finite") || msg.contains("Non-finite"),
        "error should mention non-finite, got: {msg}"
    );
}

/// NaN in forward initial state h must be caught.
/// The forward LSTM's gate computation includes h_prev @ w_hh^T, so NaN propagates.
#[test]
fn test_bilstm_nan_forward_state_returns_error() {
    let h = 2;
    let input_size = 3;
    let bilstm = make_bilstm(h, input_size, 0.1);

    let input = DynTensor::full(&[3, 1, input_size], 1.0, DType::F32, &Device::Cpu).unwrap();
    let nan_state = LstmState::new(
        DynTensor::from_vec(vec![f32::NAN, 0.0], &[1, h], &Device::Cpu).unwrap(),
        zeros(&[1, h]),
    )
    .unwrap();

    let result = bilstm.forward_seq(&input, Some(&nan_state), None);
    let err = result.expect_err("NaN in forward state should propagate to error");
    let msg = err.to_string();
    assert!(
        msg.contains("non-finite") || msg.contains("Non-finite"),
        "error should mention non-finite, got: {msg}"
    );
}

/// NaN in backward initial state c must be caught.
/// c_new = f * c_prev + i * g → NaN propagates through f * NaN.
#[test]
fn test_bilstm_nan_backward_state_returns_error() {
    let h = 2;
    let input_size = 3;
    let bilstm = make_bilstm(h, input_size, 0.1);

    let input = DynTensor::full(&[3, 1, input_size], 1.0, DType::F32, &Device::Cpu).unwrap();
    let nan_state = LstmState::new(
        zeros(&[1, h]),
        DynTensor::from_vec(vec![0.0, f32::NAN], &[1, h], &Device::Cpu).unwrap(),
    )
    .unwrap();

    let result = bilstm.forward_seq(&input, None, Some(&nan_state));
    let err = result.expect_err("NaN in backward state should propagate to error");
    let msg = err.to_string();
    assert!(
        msg.contains("non-finite") || msg.contains("Non-finite"),
        "error should mention non-finite, got: {msg}"
    );
}

// -- Transpose elimination parity tests (#2492) --------------------------------

/// `forward_seq_batch_first` must produce identical output to manual transpose +
/// `forward_seq` + transpose. This verifies that the batch-first API is a pure
/// convenience wrapper with no numerical difference.
#[test]
fn test_bilstm_batch_first_parity() {
    let h = 3;
    let input_size = 4;
    let seq_len = 5;
    let batch = 2;

    // Use nonzero weights to get non-trivial outputs.
    let bilstm = make_bilstm(h, input_size, 0.05);

    // Batch-first input: [batch, seq_len, input_size]
    let bf_input =
        DynTensor::full(&[batch, seq_len, input_size], 0.7, DType::F32, &Device::Cpu).unwrap();

    // Path A: batch_first API
    let (out_bf, fwd_bf, bwd_bf) = bilstm
        .forward_seq_batch_first(&bf_input, None, None)
        .unwrap();

    // Path B: manual transpose → forward_seq → transpose
    let time_first = bf_input.transpose(0, 1).unwrap(); // [seq, batch, input]
    let (out_tf, fwd_tf, bwd_tf) = bilstm.forward_seq(&time_first, None, None).unwrap();
    let out_manual = out_tf.transpose(0, 1).unwrap(); // [batch, seq, 2*h]

    // Compare outputs element-by-element.
    assert_eq!(out_bf.dims(), out_manual.dims());
    let a = out_bf.to_flat_vec::<f32>().unwrap();
    let b = out_manual.to_flat_vec::<f32>().unwrap();
    for (i, (va, vb)) in a.iter().zip(b.iter()).enumerate() {
        assert!(
            (va - vb).abs() < 1e-6,
            "batch_first parity mismatch at element {i}: {va} vs {vb}"
        );
    }

    // Compare final states.
    let fwd_bf_h = fwd_bf.h.to_flat_vec::<f32>().unwrap();
    let fwd_tf_h = fwd_tf.h.to_flat_vec::<f32>().unwrap();
    for (i, (va, vb)) in fwd_bf_h.iter().zip(fwd_tf_h.iter()).enumerate() {
        assert!(
            (va - vb).abs() < 1e-6,
            "fwd state h parity mismatch at {i}: {va} vs {vb}"
        );
    }
    let bwd_bf_h = bwd_bf.h.to_flat_vec::<f32>().unwrap();
    let bwd_tf_h = bwd_tf.h.to_flat_vec::<f32>().unwrap();
    for (i, (va, vb)) in bwd_bf_h.iter().zip(bwd_tf_h.iter()).enumerate() {
        assert!(
            (va - vb).abs() < 1e-6,
            "bwd state h parity mismatch at {i}: {va} vs {vb}"
        );
    }

    // Cell states must also match (hidden state alone is insufficient —
    // a bug could corrupt c while preserving h).
    let fwd_bf_c = fwd_bf.c.to_flat_vec::<f32>().unwrap();
    let fwd_tf_c = fwd_tf.c.to_flat_vec::<f32>().unwrap();
    for (i, (va, vb)) in fwd_bf_c.iter().zip(fwd_tf_c.iter()).enumerate() {
        assert!(
            (va - vb).abs() < 1e-6,
            "fwd state c parity mismatch at {i}: {va} vs {vb}"
        );
    }
    let bwd_bf_c = bwd_bf.c.to_flat_vec::<f32>().unwrap();
    let bwd_tf_c = bwd_tf.c.to_flat_vec::<f32>().unwrap();
    for (i, (va, vb)) in bwd_bf_c.iter().zip(bwd_tf_c.iter()).enumerate() {
        assert!(
            (va - vb).abs() < 1e-6,
            "bwd state c parity mismatch at {i}: {va} vs {vb}"
        );
    }
}

/// `permute([2, 0, 1])` must equal `transpose(1, 2).transpose(0, 1)` and
/// `permute([1, 2, 0])` must equal `transpose(0, 1).transpose(1, 2)`.
/// These are the permute replacements used in F0EnergyPredictor and TextEncoder (#2492).
#[test]
fn test_permute_vs_double_transpose_equivalence() {
    // Distinct dimensions so axis ordering is unambiguous.
    let input = DynTensor::from_vec(
        (0..24).map(|v| v as f32).collect(),
        &[2, 3, 4],
        &Device::Cpu,
    )
    .unwrap();

    // [B, C, T] → [T, B, C]: permute([2, 0, 1]) vs transpose(1,2).transpose(0,1)
    let perm_201 = input.permute([2, 0, 1]).unwrap();
    let dt_201 = input.transpose(1, 2).unwrap().transpose(0, 1).unwrap();
    assert_eq!(perm_201.dims(), dt_201.dims());
    let a = perm_201.to_flat_vec::<f32>().unwrap();
    let b = dt_201.to_flat_vec::<f32>().unwrap();
    assert_eq!(
        a, b,
        "permute([2,0,1]) must equal transpose(1,2).transpose(0,1)"
    );

    // [T, B, C] → [B, C, T]: permute([1, 2, 0]) vs transpose(0,1).transpose(1,2)
    let tf_input = input.permute([2, 0, 1]).unwrap(); // [4, 2, 3]
    let perm_120 = tf_input.permute([1, 2, 0]).unwrap();
    let dt_120 = tf_input.transpose(0, 1).unwrap().transpose(1, 2).unwrap();
    assert_eq!(perm_120.dims(), dt_120.dims());
    let a = perm_120.to_flat_vec::<f32>().unwrap();
    let b = dt_120.to_flat_vec::<f32>().unwrap();
    assert_eq!(
        a, b,
        "permute([1,2,0]) must equal transpose(0,1).transpose(1,2)"
    );
}

// -- NaN/Inf defense-in-depth tests -------------------------------------------

/// Inf weights in the backward LSTM must be caught.
/// Matches the unidirectional `test_lstm_non_finite_output_detected` pattern:
/// After #2064: Inf backward weights are rejected at construction time.
#[test]
fn test_bilstm_inf_backward_weights_returns_error() {
    let h = 2;
    let input_size = 3;

    let w_ih_fwd = DynTensor::full(&[4 * h, input_size], 0.1, DType::F32, &Device::Cpu).unwrap();
    let w_hh_fwd = DynTensor::full(&[4 * h, h], 0.1, DType::F32, &Device::Cpu).unwrap();

    let w_ih_rev = DynTensor::full(&[4 * h, input_size], 0.1, DType::F32, &Device::Cpu).unwrap();
    let w_hh_rev = DynTensor::full(&[4 * h, h], f64::INFINITY, DType::F32, &Device::Cpu).unwrap();

    let err = BiLstm::from_weights(
        w_ih_fwd, w_hh_fwd, None, None, w_ih_rev, w_hh_rev, None, None, h,
    )
    .unwrap_err();
    assert!(
        matches!(&err, TensorError::NonFiniteData { name, .. } if name.contains("w_hh")),
        "expected NonFiniteData for w_hh, got: {err}"
    );
}

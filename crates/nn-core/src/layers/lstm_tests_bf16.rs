#![allow(deprecated)]
// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! LSTM BF16/F16 auto-upcast precision guard tests (#1990).

use super::*;
use crate::{DType, Device};

/// Helper: create a DynTensor from flat data and shape.
fn tensor(data: &[f32], shape: &[usize]) -> DynTensor {
    DynTensor::from_vec(data.to_vec(), shape, &Device::Cpu).expect("valid tensor")
}

/// BF16 input to forward() should produce BF16 output with finite values.
/// The auto-upcast guard converts BF16→F32 internally for numerical stability,
/// then downcasts results back to BF16.
#[test]
fn test_lstm_bf16_forward_preserves_dtype() {
    let h = 2;
    let input_size = 3;
    let w_ih = DynTensor::full(&[4 * h, input_size], 0.1, DType::F32, &Device::Cpu).unwrap();
    let w_hh = DynTensor::full(&[4 * h, h], 0.1, DType::F32, &Device::Cpu).unwrap();
    let lstm = Lstm::new(w_ih, w_hh, None, None, h).unwrap();

    // Create BF16 input by converting from F32.
    let f32_input = DynTensor::full(&[1, input_size], 1.0, DType::F32, &Device::Cpu).unwrap();
    let bf16_input = f32_input.to_dtype(DType::BF16).unwrap();
    assert_eq!(bf16_input.dtype(), DType::BF16);

    let (output, state) = lstm.forward(&bf16_input, None).unwrap();

    // Output dtype must match input dtype (BF16).
    assert_eq!(output.dtype(), DType::BF16, "output should be BF16");
    assert_eq!(state.h.dtype(), DType::BF16, "state h should be BF16");
    assert_eq!(state.c.dtype(), DType::BF16, "state c should be BF16");

    // Values must be finite and non-zero.
    let h_vals = output
        .to_dtype(DType::F32)
        .unwrap()
        .to_flat_vec::<f32>()
        .unwrap();
    for &v in &h_vals {
        assert!(v.is_finite(), "BF16 LSTM output must be finite, got {v}");
        assert!(v.abs() > 1e-10, "BF16 LSTM output should be non-zero");
    }
}

/// BF16 forward should produce results close to F32 forward (within BF16 precision).
#[test]
fn test_lstm_bf16_matches_f32_within_tolerance() {
    let h = 2;
    let input_size = 3;
    let w_ih = DynTensor::full(&[4 * h, input_size], 0.1, DType::F32, &Device::Cpu).unwrap();
    let w_hh = DynTensor::full(&[4 * h, h], 0.1, DType::F32, &Device::Cpu).unwrap();
    let lstm = Lstm::new(w_ih, w_hh, None, None, h).unwrap();

    let f32_input = tensor(&[1.0, 2.0, 3.0], &[1, 3]);
    let bf16_input = f32_input.to_dtype(DType::BF16).unwrap();

    let (f32_out, _) = lstm.forward(&f32_input, None).unwrap();
    let (bf16_out, _) = lstm.forward(&bf16_input, None).unwrap();

    let f32_vals = f32_out.to_flat_vec::<f32>().unwrap();
    let bf16_vals = bf16_out
        .to_dtype(DType::F32)
        .unwrap()
        .to_flat_vec::<f32>()
        .unwrap();

    // BF16 has ~7 bits of mantissa → tolerance ~1e-2 is reasonable.
    for i in 0..h {
        assert!(
            (f32_vals[i] - bf16_vals[i]).abs() < 0.01,
            "BF16 vs F32 mismatch at [{i}]: f32={}, bf16={}",
            f32_vals[i],
            bf16_vals[i],
        );
    }
}

/// BF16 input to forward_seq() should produce BF16 output with correct shapes.
#[test]
fn test_lstm_bf16_forward_seq_preserves_dtype() {
    let h = 2;
    let input_size = 3;
    let seq_len = 3;
    let batch = 1;

    let w_ih = DynTensor::full(&[4 * h, input_size], 0.1, DType::F32, &Device::Cpu).unwrap();
    let w_hh = DynTensor::full(&[4 * h, h], 0.1, DType::F32, &Device::Cpu).unwrap();
    let lstm = Lstm::new(w_ih, w_hh, None, None, h).unwrap();

    let f32_seq =
        DynTensor::full(&[seq_len, batch, input_size], 1.0, DType::F32, &Device::Cpu).unwrap();
    let bf16_seq = f32_seq.to_dtype(DType::BF16).unwrap();

    let (outputs, final_state) = lstm.forward_seq(&bf16_seq, None).unwrap();

    assert_eq!(outputs.dtype(), DType::BF16, "seq outputs should be BF16");
    assert_eq!(outputs.dims(), &[seq_len, batch, h]);
    assert_eq!(final_state.h.dtype(), DType::BF16);
    assert_eq!(final_state.c.dtype(), DType::BF16);

    // Verify finiteness.
    let out_vals = outputs
        .to_dtype(DType::F32)
        .unwrap()
        .to_flat_vec::<f32>()
        .unwrap();
    assert!(
        out_vals.iter().all(|v| v.is_finite()),
        "BF16 forward_seq outputs must all be finite"
    );
}

/// F16 input to forward() should also be auto-upcast (same guard as BF16).
#[test]
fn test_lstm_f16_forward_preserves_dtype() {
    let h = 2;
    let input_size = 3;
    let w_ih = DynTensor::full(&[4 * h, input_size], 0.1, DType::F32, &Device::Cpu).unwrap();
    let w_hh = DynTensor::full(&[4 * h, h], 0.1, DType::F32, &Device::Cpu).unwrap();
    let lstm = Lstm::new(w_ih, w_hh, None, None, h).unwrap();

    let f32_input = DynTensor::full(&[1, input_size], 1.0, DType::F32, &Device::Cpu).unwrap();
    let f16_input = f32_input.to_dtype(DType::F16).unwrap();
    assert_eq!(f16_input.dtype(), DType::F16);

    let (output, state) = lstm.forward(&f16_input, None).unwrap();

    assert_eq!(output.dtype(), DType::F16, "output should be F16");
    assert_eq!(state.h.dtype(), DType::F16, "state h should be F16");
    assert_eq!(state.c.dtype(), DType::F16, "state c should be F16");

    let h_vals = output
        .to_dtype(DType::F32)
        .unwrap()
        .to_flat_vec::<f32>()
        .unwrap();
    for &v in &h_vals {
        assert!(v.is_finite(), "F16 LSTM output must be finite, got {v}");
    }
}

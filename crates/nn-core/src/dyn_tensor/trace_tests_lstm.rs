// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! LSTM tracing tests — verifies fix for #2189.
//!
//! LSTM trace recording must capture all 3 inputs (x, h_state, c_state),
//! not just the input tensor. Without this, compile_trace fails with
//! MissingInputNode at index 1.

use super::*;
use crate::dyn_tensor::DynTensor;
use crate::{DType, Device};

fn cpu() -> Device {
    Device::Cpu
}

#[test]
fn test_trace_lstm_records_3_inputs() {
    use crate::layers::Lstm;

    // hidden_size=2, input_size=3
    // w_ih: [4*H, I] = [8, 3], w_hh: [4*H, H] = [8, 2]
    let w_ih = DynTensor::new(&[0.1f32; 24], &[8, 3], &cpu()).expect("valid w_ih");
    let w_hh = DynTensor::new(&[0.1f32; 16], &[8, 2], &cpu()).expect("valid w_hh");
    let b_ih = DynTensor::new(&[0.0f32; 8], &[8], &cpu()).expect("valid b_ih");
    let b_hh = DynTensor::new(&[0.0f32; 8], &[8], &cpu()).expect("valid b_hh");
    let lstm = Lstm::new(w_ih, w_hh, Some(b_ih), Some(b_hh), 2).expect("valid lstm");

    let x = DynTensor::new(&[1.0, 2.0, 3.0], &[1, 3], &cpu()).expect("valid input");

    let (result, graph) = trace_graph(|| {
        let mut x = x.clone();
        let id = record_input(&[1, 3], DType::F32).expect("record input");
        x.set_trace_id(id);
        let (output, _state) = lstm.forward(&x, None)?;
        Ok(output)
    })
    .expect("trace_graph");

    // Graph: 1 input (x) + 2 auto-registered inputs (h, c) + 1 LSTM op = 4 nodes
    assert_eq!(
        graph.len(),
        4,
        "expected 4 nodes: input + h_state + c_state + lstm"
    );

    let output = graph.output_node().expect("has output node");
    assert!(
        matches!(output.op(), TraceOp::Lstm { hidden_size: 2, .. }),
        "expected Lstm op, got {:?}",
        output.op()
    );
    if let TraceOp::Lstm {
        weight_ih,
        weight_hh,
        ..
    } = output.op()
    {
        assert_eq!(weight_ih.shape(), &[8, 3]);
        assert_eq!(weight_hh.shape(), &[8, 2]);
    }

    // LSTM node must have exactly 3 inputs (x, h, c) — the bug was only 1
    assert_eq!(
        output.inputs().len(),
        3,
        "LSTM must record 3 inputs: x, h_state, c_state"
    );

    // Output shape: [batch=1, hidden_size=2]
    assert_eq!(result.dims(), &[1, 2]);
}

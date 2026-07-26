// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! LSTM validation / error-path integration tests.
//!
//! Tests that `TensorKernelDef::validate()` catches shape mismatches in
//! manually-constructed LSTM kernels. Split from `graph_translate_lstm.rs`
//! to stay under 500 lines (#795).

use nn_dsl::tensor_ir::{TensorKernelDef, TensorNode, TensorNodeId, TensorOpKind};

use super::build_lstm_kernel;

/// Build an LSTM kernel manually (bypassing builder debug_assert) for negative tests.
fn manual_lstm_kernel(
    input_shape: &[usize],
    hidden_shape: &[usize],
    cell_shape: &[usize],
    wih_shape: &[usize],
    whh_shape: &[usize],
    bias_shape: Option<&[usize]>,
    out_shape: &[usize],
) -> TensorKernelDef {
    let mut nodes = vec![
        TensorNode::new(
            TensorNodeId::new(0),
            TensorOpKind::Input {
                name: "input".into(),
                shape: input_shape.to_vec(),
            },
            input_shape.to_vec(),
        ),
        TensorNode::new(
            TensorNodeId::new(1),
            TensorOpKind::Input {
                name: "hidden".into(),
                shape: hidden_shape.to_vec(),
            },
            hidden_shape.to_vec(),
        ),
        TensorNode::new(
            TensorNodeId::new(2),
            TensorOpKind::Input {
                name: "cell".into(),
                shape: cell_shape.to_vec(),
            },
            cell_shape.to_vec(),
        ),
        TensorNode::new(
            TensorNodeId::new(3),
            TensorOpKind::Input {
                name: "weight_ih".into(),
                shape: wih_shape.to_vec(),
            },
            wih_shape.to_vec(),
        ),
        TensorNode::new(
            TensorNodeId::new(4),
            TensorOpKind::Input {
                name: "weight_hh".into(),
                shape: whh_shape.to_vec(),
            },
            whh_shape.to_vec(),
        ),
    ];

    let bias = if let Some(bs) = bias_shape {
        nodes.push(TensorNode::new(
            TensorNodeId::new(5),
            TensorOpKind::Input {
                name: "bias".into(),
                shape: bs.to_vec(),
            },
            bs.to_vec(),
        ));
        Some(TensorNodeId::new(5))
    } else {
        None
    };

    let lstm_id = TensorNodeId::new(nodes.len());
    nodes.push(TensorNode::new(
        lstm_id,
        TensorOpKind::Lstm {
            input: TensorNodeId::new(0),
            hidden_state: TensorNodeId::new(1),
            cell_state: TensorNodeId::new(2),
            weight_ih: TensorNodeId::new(3),
            weight_hh: TensorNodeId::new(4),
            bias,
        },
        out_shape.to_vec(),
    ));

    TensorKernelDef::new("test_lstm", nodes, lstm_id)
}

#[test]
fn test_lstm_valid_kernel_passes_validation() {
    let kernel = build_lstm_kernel("lstm_valid", 4, 3, true);
    let result = kernel.validate();
    assert!(
        result.is_ok(),
        "valid LSTM kernel should pass validation: {result:?}"
    );
}

#[test]
fn test_lstm_hidden_cell_shape_mismatch_rejected() {
    // hidden=[3], cell=[4] — shape mismatch
    let kernel = manual_lstm_kernel(
        &[4],     // input
        &[3],     // hidden
        &[4],     // cell (mismatched!)
        &[12, 4], // weight_ih (4*3, 4)
        &[12, 3], // weight_hh (4*3, 3)
        None,
        &[3], // output
    );
    let result = kernel.validate();
    assert!(
        result.is_err(),
        "hidden/cell mismatch should fail validation"
    );
    let err = result.unwrap_err().to_string();
    assert!(
        err.contains("hidden_state shape") || err.contains("cell_state"),
        "error should mention hidden/cell mismatch: {err}"
    );
}

#[test]
fn test_lstm_weight_ih_shape_mismatch_rejected() {
    // weight_ih should be [4*3, 4] = [12, 4], but we give [8, 4]
    let kernel = manual_lstm_kernel(
        &[4],
        &[3],
        &[3],
        &[8, 4], // wrong: should be [12, 4]
        &[12, 3],
        None,
        &[3],
    );
    let result = kernel.validate();
    assert!(
        result.is_err(),
        "weight_ih shape mismatch should fail validation"
    );
    let err = result.unwrap_err().to_string();
    assert!(
        err.contains("weight_ih"),
        "error should mention weight_ih: {err}"
    );
}

#[test]
fn test_lstm_weight_hh_shape_mismatch_rejected() {
    // weight_hh should be [4*3, 3] = [12, 3], but we give [12, 5]
    let kernel = manual_lstm_kernel(
        &[4],
        &[3],
        &[3],
        &[12, 4],
        &[12, 5], // wrong: should be [12, 3]
        None,
        &[3],
    );
    let result = kernel.validate();
    assert!(
        result.is_err(),
        "weight_hh shape mismatch should fail validation"
    );
    let err = result.unwrap_err().to_string();
    assert!(
        err.contains("weight_hh"),
        "error should mention weight_hh: {err}"
    );
}

#[test]
fn test_lstm_bias_shape_mismatch_rejected() {
    // bias should be [12], but we give [8]
    let kernel = manual_lstm_kernel(
        &[4],
        &[3],
        &[3],
        &[12, 4],
        &[12, 3],
        Some(&[8]), // wrong: should be [12]
        &[3],
    );
    let result = kernel.validate();
    assert!(
        result.is_err(),
        "bias shape mismatch should fail validation"
    );
    let err = result.unwrap_err().to_string();
    assert!(err.contains("bias"), "error should mention bias: {err}");
}

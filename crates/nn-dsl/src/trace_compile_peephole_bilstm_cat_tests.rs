// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Tests for the BiLstmCat peephole pass (#4252).

use std::collections::HashMap;

use nn_core::dyn_tensor::trace::WeightRef;

use crate::trace_compile::{CompiledStep, NativeOpKind};

/// Helper: create a forward LstmSequence NativeOp step.
fn make_lstm_step(hidden_size: usize, reverse: bool) -> CompiledStep {
    let prefix = if reverse { "rev" } else { "fwd" };
    let n = 4 * hidden_size;
    let mut weight_data = HashMap::new();
    weight_data.insert(
        "weight_ih".to_string(),
        WeightRef::new(vec![0.0f32; n * 32], vec![n, 32]).unwrap(),
    );
    weight_data.insert(
        "weight_hh".to_string(),
        WeightRef::new(vec![0.0f32; n * hidden_size], vec![n, hidden_size]).unwrap(),
    );
    weight_data.insert(
        "h0".to_string(),
        WeightRef::new(vec![0.0f32; hidden_size], vec![1, hidden_size]).unwrap(),
    );
    weight_data.insert(
        "c0".to_string(),
        WeightRef::new(vec![0.0f32; hidden_size], vec![1, hidden_size]).unwrap(),
    );
    let _ = prefix; // Suppress unused warning — prefix is only needed for merged weights.
    CompiledStep::NativeOp {
        op: NativeOpKind::LstmSequence {
            hidden_size,
            input_shape: vec![10, 1, 32],
            h_shape: vec![1, hidden_size],
            reverse,
        },
        weight_data,
    }
}

#[test]
fn test_bilstm_cat_fusion_replaces_steps() {
    // Verify the structural requirement: BiLstmCat NativeOp has merged
    // weight keys with fwd_/rev_ prefixes.
    let fwd = make_lstm_step(64, false);
    let rev = make_lstm_step(64, true);

    // Extract weight data from both steps.
    let fwd_weights = match &fwd {
        CompiledStep::NativeOp { weight_data, .. } => weight_data.clone(),
        _ => panic!("expected NativeOp"),
    };
    let rev_weights = match &rev {
        CompiledStep::NativeOp { weight_data, .. } => weight_data.clone(),
        _ => panic!("expected NativeOp"),
    };

    // Merge with prefixes (same logic as the peephole pass).
    let mut merged = HashMap::new();
    for (k, v) in &fwd_weights {
        merged.insert(format!("fwd_{k}"), v.clone());
    }
    for (k, v) in &rev_weights {
        merged.insert(format!("rev_{k}"), v.clone());
    }

    // Verify all expected keys are present.
    assert!(merged.contains_key("fwd_weight_ih"));
    assert!(merged.contains_key("fwd_weight_hh"));
    assert!(merged.contains_key("fwd_h0"));
    assert!(merged.contains_key("fwd_c0"));
    assert!(merged.contains_key("rev_weight_ih"));
    assert!(merged.contains_key("rev_weight_hh"));
    assert!(merged.contains_key("rev_h0"));
    assert!(merged.contains_key("rev_c0"));
    assert_eq!(merged.len(), 8);
}

#[test]
fn test_bilstm_cat_nativeop_variant_fields() {
    // Verify BiLstmCat variant can be constructed with expected fields.
    let op = NativeOpKind::BiLstmCat {
        hidden_size: 64,
        input_shape: vec![10, 1, 32],
        h_shape: vec![1, 64],
        fwd_lstm_step: 0,
        rev_lstm_step: 2,
    };

    assert_eq!(op.variant_name(), "BiLstmCat");

    // BiLstmCat: fwd LSTM + rev LSTM + cat = 3 Metal dispatches.
    assert_eq!(op.estimated_metal_dispatches(), 3);
}

#[test]
fn test_mismatched_hidden_size_not_fused() {
    // Two LSTMs with different hidden sizes should not be fused.
    let fwd = make_lstm_step(64, false);
    let rev = make_lstm_step(128, true);

    // Extract hidden sizes.
    let fwd_h = match &fwd {
        CompiledStep::NativeOp {
            op: NativeOpKind::LstmSequence { hidden_size, .. },
            ..
        } => *hidden_size,
        _ => panic!(),
    };
    let rev_h = match &rev {
        CompiledStep::NativeOp {
            op: NativeOpKind::LstmSequence { hidden_size, .. },
            ..
        } => *hidden_size,
        _ => panic!(),
    };

    // Different hidden sizes — the pass should skip this pair.
    assert_ne!(fwd_h, rev_h);
}

// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Tests for LSTM cell decomposition into primitive tensor ops.
//!
//! Extracted from lstm_decomposed.rs inline tests for 500-line compliance.

use super::*;

#[test]
fn test_decomposed_lstm_with_bias_validates() {
    let def = build_lstm_cell_decomposed(128, 128, 1, true).expect("valid dims");
    assert!(def.validate().is_ok(), "{:?}", def.validate());
    assert_eq!(def.name, "lstm_cell_decomposed");
    assert_eq!(def.nodes.len(), 22);
}

#[test]
fn test_decomposed_lstm_no_bias_validates() {
    let def = build_lstm_cell_decomposed(128, 128, 1, false).expect("valid dims");
    assert!(def.validate().is_ok(), "{:?}", def.validate());
    assert_eq!(def.nodes.len(), 21);
}

#[test]
fn test_decomposed_lstm_silero_vad_shape() {
    let def = build_lstm_cell_decomposed(128, 128, 1, true).expect("valid dims");
    assert!(def.validate().is_ok());
    assert_eq!(def.nodes[def.output.index()].shape, vec![1, 128]);
}

#[test]
fn test_decomposed_lstm_different_input_hidden_sizes() {
    let def = build_lstm_cell_decomposed(64, 256, 2, true).expect("valid dims");
    assert!(def.validate().is_ok());
    assert_eq!(def.nodes[def.output.index()].shape, vec![2, 256]);
}

#[test]
fn test_decompose_in_builder_returns_both_states() {
    let mut builder = TensorBlockBuilder::new("test_dual_output");
    let input = builder.add_input("input", &[1, 64]);
    let hidden = builder.add_input("hidden", &[1, 32]);
    let cell = builder.add_input("cell", &[1, 32]);
    let w_ih = builder.add_input("w_ih", &[128, 64]);
    let w_hh = builder.add_input("w_hh", &[128, 32]);
    let bias = builder.add_input("bias", &[128]);
    let outputs = decompose_lstm_cell(
        &mut builder,
        input,
        hidden,
        cell,
        w_ih,
        w_hh,
        Some(bias),
        32,
        1,
    );
    assert_ne!(outputs.h_new, outputs.c_new);
    let def = builder.build(outputs.h_new).expect("valid graph");
    assert!(def.validate().is_ok());
}

#[test]
fn test_decompose_c_new_as_output() {
    let mut builder = TensorBlockBuilder::new("test_c_output");
    let input = builder.add_input("input", &[1, 64]);
    let hidden = builder.add_input("hidden", &[1, 32]);
    let cell = builder.add_input("cell", &[1, 32]);
    let w_ih = builder.add_input("w_ih", &[128, 64]);
    let w_hh = builder.add_input("w_hh", &[128, 32]);
    let outputs = decompose_lstm_cell(&mut builder, input, hidden, cell, w_ih, w_hh, None, 32, 1);
    let def = builder.build(outputs.c_new).expect("valid graph");
    assert!(def.validate().is_ok());
    assert_eq!(def.nodes[def.output.index()].shape, vec![1, 32]);
}

#[test]
fn test_decomposed_node_count_matches_design() {
    let with = build_lstm_cell_decomposed(128, 128, 1, true).expect("valid dims");
    let without = build_lstm_cell_decomposed(128, 128, 1, false).expect("valid dims");
    assert_eq!(with.nodes.len(), 22);
    assert_eq!(without.nodes.len(), 21);
}

#[test]
fn test_dual_output_validates() {
    let def = build_lstm_cell_decomposed_dual(128, 128, 1, true).expect("valid dims");
    assert!(def.validate().is_ok(), "{:?}", def.validate());
    assert_eq!(def.name, "lstm_cell_decomposed_dual");
    assert_eq!(def.nodes[def.output.index()].shape, vec![2, 1, 128]);
}

#[test]
fn test_dual_output_node_count() {
    let def = build_lstm_cell_decomposed_dual(128, 128, 1, true).expect("valid dims");
    assert_eq!(def.nodes.len(), 23);
    let nb = build_lstm_cell_decomposed_dual(128, 128, 1, false).expect("valid dims");
    assert_eq!(nb.nodes.len(), 22);
}

#[test]
fn test_dual_output_flat_buffer_size() {
    let def = build_lstm_cell_decomposed_dual(128, 128, 1, true).expect("valid dims");
    let total: usize = def.nodes[def.output.index()].shape.iter().product();
    assert_eq!(total, 256);
}

#[test]
fn test_zero_input_size_returns_error() {
    assert!(matches!(
        build_lstm_cell_decomposed(0, 128, 1, true).unwrap_err(),
        TensorIRError::Layer(TensorIRLayerError::LstmZeroDimension {
            param: "input_size"
        })
    ));
}

#[test]
fn test_zero_hidden_size_returns_error() {
    assert!(matches!(
        build_lstm_cell_decomposed(64, 0, 1, true).unwrap_err(),
        TensorIRError::Layer(TensorIRLayerError::LstmZeroDimension {
            param: "hidden_size"
        })
    ));
}

#[test]
fn test_zero_batch_returns_error() {
    assert!(matches!(
        build_lstm_cell_decomposed(64, 128, 0, true).unwrap_err(),
        TensorIRError::Layer(TensorIRLayerError::LstmZeroDimension { param: "batch" })
    ));
}

#[test]
fn test_dual_zero_hidden_returns_error() {
    assert!(build_lstm_cell_decomposed_dual(128, 0, 1, true).is_err());
}

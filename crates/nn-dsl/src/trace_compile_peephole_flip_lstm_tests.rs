// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Tests for peephole pass 10: Flip + LstmSequence + Flip → reverse LSTM.
//! Part of #1815.

use std::collections::HashMap;

use nn_core::dyn_tensor::trace::{ComputationGraph, TraceNode, TraceOp, WeightRef};
use nn_core::DType;

use super::super::super::{CompiledKernel, CompiledStep, NativeOpKind};

fn test_node(id: u64, name: &str, inputs: Vec<u64>, shape: Vec<usize>) -> TraceNode {
    TraceNode::new(
        id,
        name.to_string(),
        TraceOp::Relu,
        inputs,
        shape,
        DType::F32,
    )
}

fn test_input_node(id: u64, shape: Vec<usize>) -> TraceNode {
    TraceNode::new(
        id,
        format!("input_{id}"),
        TraceOp::Input,
        vec![],
        shape,
        DType::F32,
    )
}

/// Build a Dispatch{flip} step matching the production `compile_flip` output.
///
/// Uses kernel name "flip" (not "flip_0") with an IndexSelect on dim 0,
/// matching what `trace_compile_misc.rs::compile_flip` produces.
fn make_flip_dispatch(input_shape: &[usize]) -> CompiledStep {
    use crate::tensor_block_builder::TensorBlockBuilder;

    let n = input_shape[0]; // flip along dim 0
    let reversed_indices: Vec<f32> = (0..n).rev().map(|i| i as f32).collect();
    let idx_weight = WeightRef::new(reversed_indices, vec![n]).expect("valid");

    let mut b = TensorBlockBuilder::new("flip");
    let input = b.add_input("input_0", input_shape);
    let indices = b.add_input("flip_indices", &[n]);
    let output = b.add_index_select(input, indices, 0, input_shape);
    let def = b.build(output).expect("valid flip IR");

    let mut weight_data = HashMap::new();
    weight_data.insert("flip_indices".to_string(), idx_weight);

    CompiledStep::Dispatch {
        kernel: CompiledKernel::new(def),
        weight_data,
        external_node_ids: None,
    }
}

/// Build a LstmSequence NativeOp step.
fn make_lstm_native(hidden_size: usize, input_shape: &[usize], reverse: bool) -> CompiledStep {
    let h_shape = vec![input_shape[1], hidden_size];
    let weight_ih_shape = vec![4 * hidden_size, input_shape[2]];
    let weight_hh_shape = vec![4 * hidden_size, hidden_size];
    let bias_shape = vec![4 * hidden_size];

    let mut weight_data = HashMap::new();
    weight_data.insert(
        "weight_ih".to_string(),
        WeightRef::new(
            vec![0.0f32; weight_ih_shape.iter().product()],
            weight_ih_shape,
        )
        .expect("valid"),
    );
    weight_data.insert(
        "weight_hh".to_string(),
        WeightRef::new(
            vec![0.0f32; weight_hh_shape.iter().product()],
            weight_hh_shape,
        )
        .expect("valid"),
    );
    weight_data.insert(
        "bias".to_string(),
        WeightRef::new(vec![0.0f32; bias_shape[0]], bias_shape).expect("valid"),
    );
    weight_data.insert(
        "h0".to_string(),
        WeightRef::new(vec![0.0f32; h_shape.iter().product()], h_shape.clone()).expect("valid"),
    );
    weight_data.insert(
        "c0".to_string(),
        WeightRef::new(vec![0.0f32; h_shape.iter().product()], h_shape.clone()).expect("valid"),
    );

    CompiledStep::NativeOp {
        op: NativeOpKind::LstmSequence {
            hidden_size,
            input_shape: input_shape.to_vec(),
            h_shape,
            reverse,
        },
        weight_data,
    }
}

// -- Flip + LSTM + Flip absorption --------------------------------------------

#[test]
fn test_peephole_absorbs_flip_lstm_flip() {
    let seq_shape = vec![10, 1, 64]; // [seq, batch, input]
    let out_shape = vec![10, 1, 128]; // [seq, batch, hidden]
    let hidden_size = 128;

    let mut steps = vec![
        CompiledStep::InputForward,                       // 0: input
        make_flip_dispatch(&seq_shape),                   // 1: flip_in
        make_lstm_native(hidden_size, &seq_shape, false), // 2: LSTM(forward)
        make_flip_dispatch(&out_shape),                   // 3: flip_out
    ];

    let graph = ComputationGraph::from_nodes(vec![
        test_input_node(0, seq_shape.clone()),
        test_node(1, "flip_in", vec![0], seq_shape),
        test_node(2, "lstm_bwd", vec![1], out_shape.clone()),
        test_node(3, "flip_out", vec![2], out_shape),
    ]);

    super::super::apply_peephole(&mut steps, &graph);

    // flip_in should be absorbed → IdentityPassthrough.
    assert!(
        matches!(&steps[1], CompiledStep::IdentityPassthrough),
        "flip_in should be IdentityPassthrough, got {:?}",
        std::mem::discriminant(&steps[1])
    );

    // LSTM should now have reverse=true.
    match &steps[2] {
        CompiledStep::NativeOp {
            op: NativeOpKind::LstmSequence { reverse, .. },
            ..
        } => {
            assert!(*reverse, "LSTM should have reverse=true after absorption");
        }
        other => panic!(
            "expected LstmSequence at step 2, got: {:?}",
            std::mem::discriminant(other)
        ),
    }

    // flip_out should be absorbed → IdentityPassthrough.
    assert!(
        matches!(&steps[3], CompiledStep::IdentityPassthrough),
        "flip_out should be IdentityPassthrough, got {:?}",
        std::mem::discriminant(&steps[3])
    );
}

// -- Fan-out prevents absorption: flip_in has 2 consumers ---------------------

#[test]
fn test_peephole_no_absorb_flip_fanout() {
    let seq_shape = vec![10, 1, 64];
    let out_shape = vec![10, 1, 128];
    let hidden_size = 128;

    let mut steps = vec![
        CompiledStep::InputForward,                       // 0: input
        make_flip_dispatch(&seq_shape),                   // 1: flip_in (2 consumers)
        make_lstm_native(hidden_size, &seq_shape, false), // 2: LSTM
        make_flip_dispatch(&out_shape),                   // 3: flip_out
        CompiledStep::IdentityPassthrough,                // 4: second consumer of flip_in
    ];

    let graph = ComputationGraph::from_nodes(vec![
        test_input_node(0, seq_shape.clone()),
        test_node(1, "flip_in", vec![0], seq_shape.clone()),
        test_node(2, "lstm_bwd", vec![1], out_shape.clone()),
        test_node(3, "flip_out", vec![2], out_shape),
        test_node(4, "other", vec![1], seq_shape), // second consumer of flip
    ]);

    super::super::apply_peephole(&mut steps, &graph);

    // Should NOT absorb — flip_in has fan-out > 1.
    assert!(
        matches!(&steps[1], CompiledStep::Dispatch { .. }),
        "flip_in should remain unfused with fan-out > 1"
    );
}

// -- Already-reverse LSTM is skipped ------------------------------------------

#[test]
fn test_peephole_no_absorb_already_reverse() {
    let seq_shape = vec![10, 1, 64];
    let out_shape = vec![10, 1, 128];
    let hidden_size = 128;

    let mut steps = vec![
        CompiledStep::InputForward,
        make_flip_dispatch(&seq_shape),
        make_lstm_native(hidden_size, &seq_shape, true), // already reverse
        make_flip_dispatch(&out_shape),
    ];

    let graph = ComputationGraph::from_nodes(vec![
        test_input_node(0, seq_shape.clone()),
        test_node(1, "flip_in", vec![0], seq_shape),
        test_node(2, "lstm_bwd", vec![1], out_shape.clone()),
        test_node(3, "flip_out", vec![2], out_shape),
    ]);

    super::super::apply_peephole(&mut steps, &graph);

    // Should NOT absorb — LSTM is already in reverse mode.
    assert!(
        matches!(&steps[1], CompiledStep::Dispatch { .. }),
        "flip_in should remain when LSTM already reversed"
    );
}

// -- Non-flip dispatch between flip_in and LSTM blocks absorption -------------

#[test]
fn test_peephole_no_absorb_non_flip_intervening() {
    let seq_shape = vec![10, 1, 64];
    let out_shape = vec![10, 1, 128];
    let hidden_size = 128;

    // flip_in → relu (non-flip) → LSTM → flip_out: no triple match.
    let mut steps = vec![
        CompiledStep::InputForward,
        make_flip_dispatch(&seq_shape),                   // 1: flip
        CompiledStep::IdentityPassthrough,                // 2: some other step
        make_lstm_native(hidden_size, &seq_shape, false), // 3: LSTM
        make_flip_dispatch(&out_shape),                   // 4: flip
    ];

    let graph = ComputationGraph::from_nodes(vec![
        test_input_node(0, seq_shape.clone()),
        test_node(1, "flip_in", vec![0], seq_shape.clone()),
        test_node(2, "relu", vec![1], seq_shape),
        test_node(3, "lstm_bwd", vec![2], out_shape.clone()),
        test_node(4, "flip_out", vec![3], out_shape),
    ]);

    super::super::apply_peephole(&mut steps, &graph);

    // flip_in should remain — LSTM is not adjacent.
    assert!(
        matches!(&steps[1], CompiledStep::Dispatch { .. }),
        "flip_in should remain when LSTM is not adjacent"
    );
    // LSTM should still be forward.
    match &steps[3] {
        CompiledStep::NativeOp {
            op: NativeOpKind::LstmSequence { reverse, .. },
            ..
        } => assert!(!*reverse, "LSTM should remain forward"),
        _ => panic!("expected LstmSequence at step 3"),
    }
}

// -- Multiple consecutive BiLSTM layers each absorb independently -------------

#[test]
fn test_peephole_absorbs_two_bilstm_layers() {
    let seq_shape = vec![10, 1, 64];
    let out_shape = vec![10, 1, 128];
    let hidden_size = 128;

    // Two back-to-back BiLSTM backward layers:
    // flip → LSTM → flip → flip → LSTM → flip
    let mut steps = vec![
        CompiledStep::InputForward,                       // 0
        make_flip_dispatch(&seq_shape),                   // 1: flip_in_1
        make_lstm_native(hidden_size, &seq_shape, false), // 2: LSTM_1
        make_flip_dispatch(&out_shape),                   // 3: flip_out_1
        make_flip_dispatch(&out_shape),                   // 4: flip_in_2
        make_lstm_native(hidden_size, &out_shape, false), // 5: LSTM_2
        make_flip_dispatch(&out_shape),                   // 6: flip_out_2
    ];

    let graph = ComputationGraph::from_nodes(vec![
        test_input_node(0, seq_shape.clone()),
        test_node(1, "flip_in_1", vec![0], seq_shape),
        test_node(2, "lstm_1", vec![1], out_shape.clone()),
        test_node(3, "flip_out_1", vec![2], out_shape.clone()),
        test_node(4, "flip_in_2", vec![3], out_shape.clone()),
        test_node(5, "lstm_2", vec![4], out_shape.clone()),
        test_node(6, "flip_out_2", vec![5], out_shape),
    ]);

    super::super::apply_peephole(&mut steps, &graph);

    // Both triples should be absorbed.
    assert!(
        matches!(&steps[1], CompiledStep::IdentityPassthrough),
        "flip_in_1 should be absorbed"
    );
    match &steps[2] {
        CompiledStep::NativeOp {
            op: NativeOpKind::LstmSequence { reverse, .. },
            ..
        } => assert!(*reverse, "LSTM_1 should be reverse"),
        _ => panic!("expected LstmSequence at step 2"),
    }
    assert!(
        matches!(&steps[3], CompiledStep::IdentityPassthrough),
        "flip_out_1 should be absorbed"
    );
    assert!(
        matches!(&steps[4], CompiledStep::IdentityPassthrough),
        "flip_in_2 should be absorbed"
    );
    match &steps[5] {
        CompiledStep::NativeOp {
            op: NativeOpKind::LstmSequence { reverse, .. },
            ..
        } => assert!(*reverse, "LSTM_2 should be reverse"),
        _ => panic!("expected LstmSequence at step 5"),
    }
    assert!(
        matches!(&steps[6], CompiledStep::IdentityPassthrough),
        "flip_out_2 should be absorbed"
    );
}

// -- Dispatch count unchanged -------------------------------------------------

#[test]
fn test_lstm_reverse_single_dispatch() {
    let op = NativeOpKind::LstmSequence {
        hidden_size: 256,
        input_shape: vec![50, 1, 512],
        h_shape: vec![1, 256],
        reverse: true,
    };
    assert_eq!(
        op.estimated_metal_dispatches(),
        1,
        "Reverse LSTM should be 1 dispatch"
    );
}

// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Tests for multi-node expansion: LSTM, BiLSTM, repeat_interleave.

use std::collections::HashMap;

use nn_core::dyn_tensor::trace::TraceOp;

use super::*;
use crate::parse::{
    Argument, ArgumentFloat, ArgumentInt, ArgumentTensor, NamedArgument, Node, TensorArgument,
    TensorMeta,
};

fn tensor_arg(name: &str) -> Argument {
    Argument::Tensor(ArgumentTensor {
        as_tensor: TensorArgument {
            name: name.to_string(),
        },
    })
}

fn named(name: &str, arg: Argument) -> NamedArgument {
    NamedArgument {
        name: name.to_string(),
        arg,
        kind: None,
    }
}

fn int_arg(val: i64) -> Argument {
    Argument::Int(ArgumentInt { as_int: val })
}

fn tensors_arg(names: &[&str]) -> Argument {
    Argument::Tensors(crate::parse::ArgumentTensors {
        as_tensors: names
            .iter()
            .map(|n| TensorArgument {
                name: n.to_string(),
            })
            .collect(),
    })
}

fn bool_arg(val: bool) -> Argument {
    Argument::Bool(crate::parse::ArgumentBool { as_bool: val })
}

fn float_arg(val: f64) -> Argument {
    Argument::Float(ArgumentFloat { as_float: val })
}

fn empty_ctx() -> OpMapContext<'static> {
    let meta: &'static HashMap<String, TensorMeta> = Box::leak(Box::default());
    let weights: &'static HashMap<String, ResolvedWeight> = Box::leak(Box::default());
    OpMapContext {
        tensor_meta: meta,
        weights,
    }
}

// --- #2354: LSTM op mapping for Kokoro ---

fn lstm_ctx_with_weights(has_biases: bool) -> OpMapContext<'static> {
    let meta: &'static HashMap<String, TensorMeta> = Box::leak(Box::default());
    let mut weights = HashMap::new();
    // hidden_size=4, input_size=3: w_ih=[16, 3], w_hh=[16, 4]
    weights.insert(
        "p_lstm_w_ih".to_string(),
        ResolvedWeight::new(vec![0.1; 48], vec![16, 3]),
    );
    weights.insert(
        "p_lstm_w_hh".to_string(),
        ResolvedWeight::new(vec![0.1; 64], vec![16, 4]),
    );
    if has_biases {
        weights.insert(
            "p_lstm_b_ih".to_string(),
            ResolvedWeight::new(vec![0.0; 16], vec![16]),
        );
        weights.insert(
            "p_lstm_b_hh".to_string(),
            ResolvedWeight::new(vec![0.0; 16], vec![16]),
        );
    }
    let weights: &'static HashMap<String, ResolvedWeight> = Box::leak(Box::new(weights));
    OpMapContext {
        tensor_meta: meta,
        weights,
    }
}

#[test]
fn test_map_lstm_with_biases() {
    let ctx = lstm_ctx_with_weights(true);
    let node = Node {
        target: "torch.ops.aten.lstm.input".to_string(),
        inputs: vec![
            named("input", tensor_arg("x")),
            named("hx", tensors_arg(&["h_0", "c_0"])),
            named(
                "params",
                tensors_arg(&["p_lstm_w_ih", "p_lstm_w_hh", "p_lstm_b_ih", "p_lstm_b_hh"]),
            ),
            named("has_biases", bool_arg(true)),
            named("num_layers", int_arg(1)),
            named("dropout", float_arg(0.0)),
            named("train", bool_arg(false)),
            named("bidirectional", bool_arg(false)),
            named("batch_first", bool_arg(true)),
        ],
        outputs: vec![],
        metadata: HashMap::new(),
    };
    let (op, inputs) = map_node_to_trace_op(&node, &ctx, 0).unwrap();
    assert!(
        matches!(op, TraceOp::Lstm { hidden_size: 4, .. }),
        "expected Lstm with hidden_size=4, got: {op:?}"
    );
    assert_eq!(inputs, vec!["x", "h_0", "c_0"]);
    if let TraceOp::Lstm {
        bias_ih, bias_hh, ..
    } = &op
    {
        assert!(bias_ih.is_some(), "expected bias_ih");
        assert!(bias_hh.is_some(), "expected bias_hh");
    }
}

#[test]
fn test_map_lstm_without_biases() {
    let ctx = lstm_ctx_with_weights(false);
    let node = Node {
        target: "torch.ops.aten.lstm.input".to_string(),
        inputs: vec![
            named("input", tensor_arg("x")),
            named("hx", tensors_arg(&["h_0", "c_0"])),
            named("params", tensors_arg(&["p_lstm_w_ih", "p_lstm_w_hh"])),
            named("has_biases", bool_arg(false)),
            named("num_layers", int_arg(1)),
            named("dropout", float_arg(0.0)),
            named("train", bool_arg(false)),
            named("bidirectional", bool_arg(false)),
            named("batch_first", bool_arg(true)),
        ],
        outputs: vec![],
        metadata: HashMap::new(),
    };
    let (op, inputs) = map_node_to_trace_op(&node, &ctx, 0).unwrap();
    assert!(
        matches!(
            op,
            TraceOp::Lstm {
                hidden_size: 4,
                bias_ih: None,
                bias_hh: None,
                ..
            }
        ),
        "expected Lstm without biases, got: {op:?}"
    );
    assert_eq!(inputs, vec!["x", "h_0", "c_0"]);
}

#[test]
fn test_map_lstm_multi_layer_rejected() {
    let ctx = lstm_ctx_with_weights(true);
    let node = Node {
        target: "torch.ops.aten.lstm.input".to_string(),
        inputs: vec![
            named("input", tensor_arg("x")),
            named("hx", tensors_arg(&["h_0", "c_0"])),
            named(
                "params",
                tensors_arg(&["p_lstm_w_ih", "p_lstm_w_hh", "p_lstm_b_ih", "p_lstm_b_hh"]),
            ),
            named("has_biases", bool_arg(true)),
            named("num_layers", int_arg(2)),
            named("dropout", float_arg(0.0)),
            named("train", bool_arg(false)),
            named("bidirectional", bool_arg(false)),
            named("batch_first", bool_arg(true)),
        ],
        outputs: vec![],
        metadata: HashMap::new(),
    };
    let err = map_node_to_trace_op(&node, &ctx, 0).unwrap_err();
    assert!(
        format!("{err:?}").contains("num_layers=2"),
        "expected multi-layer rejection, got: {err:?}"
    );
}

#[test]
fn test_map_lstm_bidirectional_rejected() {
    let ctx = lstm_ctx_with_weights(true);
    let node = Node {
        target: "torch.ops.aten.lstm.input".to_string(),
        inputs: vec![
            named("input", tensor_arg("x")),
            named("hx", tensors_arg(&["h_0", "c_0"])),
            named(
                "params",
                tensors_arg(&["p_lstm_w_ih", "p_lstm_w_hh", "p_lstm_b_ih", "p_lstm_b_hh"]),
            ),
            named("has_biases", bool_arg(true)),
            named("num_layers", int_arg(1)),
            named("dropout", float_arg(0.0)),
            named("train", bool_arg(false)),
            named("bidirectional", bool_arg(true)),
            named("batch_first", bool_arg(true)),
        ],
        outputs: vec![],
        metadata: HashMap::new(),
    };
    let err = map_node_to_trace_op(&node, &ctx, 0).unwrap_err();
    assert!(
        format!("{err:?}").contains("bidirectional"),
        "expected bidirectional rejection, got: {err:?}"
    );
}

// --- repeat_interleave mapping ---

#[test]
fn test_map_repeat_interleave() {
    let ctx = empty_ctx();
    let node = Node {
        target: "torch.ops.aten.repeat_interleave.self_Tensor".to_string(),
        inputs: vec![
            named("self", tensor_arg("x")),
            named("repeats", tensor_arg("durations")),
            named("dim", int_arg(1)),
        ],
        outputs: vec![],
        metadata: HashMap::new(),
    };
    let (op, inputs) = map_node_to_trace_op(&node, &ctx, 0).unwrap();
    assert!(
        matches!(op, TraceOp::RepeatInterleave { dim: 1 }),
        "expected RepeatInterleave {{ dim: 1 }}, got: {op:?}"
    );
    assert_eq!(inputs, vec!["x", "durations"]);
}

#[test]
fn test_map_repeat_interleave_dim0() {
    let ctx = empty_ctx();
    let node = Node {
        target: "torch.ops.aten.repeat_interleave.self_Tensor".to_string(),
        inputs: vec![
            named("self", tensor_arg("features")),
            named("repeats", tensor_arg("counts")),
            named("dim", int_arg(0)),
        ],
        outputs: vec![],
        metadata: HashMap::new(),
    };
    let (op, inputs) = map_node_to_trace_op(&node, &ctx, 0).unwrap();
    assert!(matches!(op, TraceOp::RepeatInterleave { dim: 0 }));
    assert_eq!(inputs, vec!["features", "counts"]);
}

#[test]
fn test_map_repeat_interleave_missing_dim() {
    let ctx = empty_ctx();
    let node = Node {
        target: "torch.ops.aten.repeat_interleave.self_Tensor".to_string(),
        inputs: vec![
            named("self", tensor_arg("x")),
            named("repeats", tensor_arg("r")),
        ],
        outputs: vec![],
        metadata: HashMap::new(),
    };
    let err = map_node_to_trace_op(&node, &ctx, 0).unwrap_err();
    assert!(
        format!("{err:?}").contains("dim"),
        "expected dim-related error, got: {err:?}"
    );
}

// --- BiLSTM expansion ---

fn bilstm_ctx(num_layers: usize, has_biases: bool) -> OpMapContext<'static> {
    let meta: &'static HashMap<String, TensorMeta> = Box::leak(Box::default());
    let mut weights = HashMap::new();
    // hidden_size=4, input_size=3 for layer 0, input_size=8 (2*H) for layers 1+
    for layer in 0..num_layers {
        let in_sz = if layer == 0 { 3 } else { 8 }; // 2*hidden_size
        for (dir, suffix) in [(0, ""), (1, "_reverse")] {
            let _ = dir;
            let prefix = format!("p_lstm_w_ih_l{layer}{suffix}");
            weights.insert(
                prefix,
                ResolvedWeight::new(vec![0.1; 16 * in_sz], vec![16, in_sz]),
            );
            let prefix = format!("p_lstm_w_hh_l{layer}{suffix}");
            weights.insert(prefix, ResolvedWeight::new(vec![0.1; 64], vec![16, 4]));
            if has_biases {
                let prefix = format!("p_lstm_b_ih_l{layer}{suffix}");
                weights.insert(prefix, ResolvedWeight::new(vec![0.0; 16], vec![16]));
                let prefix = format!("p_lstm_b_hh_l{layer}{suffix}");
                weights.insert(prefix, ResolvedWeight::new(vec![0.0; 16], vec![16]));
            }
        }
    }
    let weights: &'static HashMap<String, ResolvedWeight> = Box::leak(Box::new(weights));
    OpMapContext {
        tensor_meta: meta,
        weights,
    }
}

fn bilstm_param_names(num_layers: usize, has_biases: bool) -> Vec<String> {
    let mut names = Vec::new();
    for layer in 0..num_layers {
        for suffix in ["", "_reverse"] {
            names.push(format!("p_lstm_w_ih_l{layer}{suffix}"));
            names.push(format!("p_lstm_w_hh_l{layer}{suffix}"));
            if has_biases {
                names.push(format!("p_lstm_b_ih_l{layer}{suffix}"));
                names.push(format!("p_lstm_b_hh_l{layer}{suffix}"));
            }
        }
    }
    names
}

#[test]
fn test_expand_bilstm_single_layer() {
    let ctx = bilstm_ctx(1, true);
    let param_names = bilstm_param_names(1, true);
    let param_refs: Vec<&str> = param_names.iter().map(String::as_str).collect();
    let node = Node {
        target: "torch.ops.aten.lstm.input".to_string(),
        inputs: vec![
            named("input", tensor_arg("x")),
            named("hx", tensors_arg(&["h_0", "c_0"])),
            named("params", tensors_arg(&param_refs)),
            named("has_biases", bool_arg(true)),
            named("num_layers", int_arg(1)),
            named("dropout", float_arg(0.0)),
            named("train", bool_arg(false)),
            named("bidirectional", bool_arg(true)),
            named("batch_first", bool_arg(true)),
        ],
        outputs: vec![tensor_arg("lstm_out")],
        metadata: HashMap::new(),
    };
    let expanded = try_expand_node(&node, &ctx, "lstm_out", &[1, 10, 3]).unwrap();
    let expanded = expanded.expect("should expand BiLSTM");
    // 1 layer: 4 constants + 1 fwd_lstm + 1 flip_in + 1 bwd_lstm + 1 flip_out + 1 cat = 9
    assert_eq!(
        expanded.len(),
        9,
        "expected 9 nodes for single-layer BiLSTM"
    );
    // Last node should be cat with output name matching original
    assert_eq!(expanded.last().unwrap().name, "lstm_out");
    assert!(
        matches!(
            expanded.last().unwrap().op,
            TraceOp::Cat {
                dim: 2,
                num_inputs: 2
            }
        ),
        "last node should be Cat on dim 2"
    );
    // Check output shape: [1, 10, 8] (2 * hidden_size=4)
    assert_eq!(expanded.last().unwrap().output_shape, vec![1, 10, 8]);
    // Forward LSTM should be at index 4
    assert!(
        matches!(expanded[4].op, TraceOp::Lstm { hidden_size: 4, .. }),
        "node 4 should be forward LSTM, got: {:?}",
        expanded[4].op
    );
}

#[test]
fn test_expand_bilstm_multi_layer() {
    let ctx = bilstm_ctx(3, true);
    let param_names = bilstm_param_names(3, true);
    let param_refs: Vec<&str> = param_names.iter().map(String::as_str).collect();
    let node = Node {
        target: "torch.ops.aten.lstm.input".to_string(),
        inputs: vec![
            named("input", tensor_arg("x")),
            named("hx", tensors_arg(&["h_0", "c_0"])),
            named("params", tensors_arg(&param_refs)),
            named("has_biases", bool_arg(true)),
            named("num_layers", int_arg(3)),
            named("dropout", float_arg(0.0)),
            named("train", bool_arg(false)),
            named("bidirectional", bool_arg(true)),
            named("batch_first", bool_arg(true)),
        ],
        outputs: vec![tensor_arg("bilstm_out")],
        metadata: HashMap::new(),
    };
    let expanded = try_expand_node(&node, &ctx, "bilstm_out", &[1, 10, 3]).unwrap();
    let expanded = expanded.expect("should expand multi-layer BiLSTM");
    // 3 layers * 9 nodes = 27
    assert_eq!(expanded.len(), 27, "expected 27 nodes for 3-layer BiLSTM");
    // Last node should be the final cat with the original output name
    assert_eq!(expanded.last().unwrap().name, "bilstm_out");
    // Final output shape: [1, 10, 8] (2 * hidden_size=4)
    assert_eq!(expanded.last().unwrap().output_shape, vec![1, 10, 8]);
}

#[test]
fn test_expand_bilstm_not_batch_first() {
    let ctx = bilstm_ctx(1, true);
    let param_names = bilstm_param_names(1, true);
    let param_refs: Vec<&str> = param_names.iter().map(String::as_str).collect();
    let node = Node {
        target: "torch.ops.aten.lstm.input".to_string(),
        inputs: vec![
            named("input", tensor_arg("x")),
            named("hx", tensors_arg(&["h_0", "c_0"])),
            named("params", tensors_arg(&param_refs)),
            named("has_biases", bool_arg(true)),
            named("num_layers", int_arg(1)),
            named("dropout", float_arg(0.0)),
            named("train", bool_arg(false)),
            named("bidirectional", bool_arg(true)),
            named("batch_first", bool_arg(false)),
        ],
        outputs: vec![tensor_arg("out")],
        metadata: HashMap::new(),
    };
    let expanded = try_expand_node(&node, &ctx, "out", &[10, 1, 3]).unwrap();
    let expanded = expanded.expect("should expand BiLSTM");
    // Flip should be on dim 0 (seq_dim for non-batch_first)
    let flip_in = &expanded[5]; // flip_in is at index 5 (after 4 constants + fwd_lstm)
    assert!(
        matches!(flip_in.op, TraceOp::Flip { dim: 0 }),
        "flip should be dim 0 for non-batch_first, got: {:?}",
        flip_in.op
    );
}

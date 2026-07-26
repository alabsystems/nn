// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! LSTM op expansion into decomposed primitives for MSL codegen.
//!
//! Decomposes `TensorOpKind::Lstm` into primitive ops: Linear, BinaryAdd,
//! Narrow, Sigmoid, Tanh, BinaryMul. For 3-D sequence inputs (`[S, B, H]`),
//! broadcast nodes are inserted for the initial hidden/cell states.
//!
//! Every primitive op has a working `DispatchStep` and MSL codegen path.
//! Part of #2306 — wiring decomposed LSTM into the compiled model pipeline.

use crate::tensor_builders::broadcast_node;
use crate::tensor_ir::{BroadcastAlignment, TensorNode, TensorNodeId, TensorOpKind};

use super::ExpandState;

/// Expand a single LSTM cell into decomposed primitive ops.
///
/// Supports both 2-D `[B, H]` and 3-D `[S, B, H]` output shapes. For the
/// sequence case (3-D), the hidden/cell state inputs (`[1, B, H]`) are
/// broadcast to `[S, B, H]` before element-wise ops.
///
/// Gate decomposition:
/// ```text
/// gates = Linear(input, weight_ih, bias) + Linear(hidden, weight_hh, None)
/// i = sigmoid(narrow(gates, last_axis, 0,   H))
/// f = sigmoid(narrow(gates, last_axis, H,   H))
/// g = tanh(narrow(gates, last_axis, 2*H, H))
/// o = sigmoid(narrow(gates, last_axis, 3*H, H))
/// c_new = f * cell_state + i * g
/// h_new = o * tanh(c_new)
/// ```
///
/// Returns the node ID of `h_new`.
pub(super) fn emit_lstm_cell(
    st: &mut ExpandState,
    input: usize,
    hidden_state: usize,
    cell_state: usize,
    weight_ih: usize,
    weight_hh: usize,
    bias: Option<usize>,
    out_shape: &[usize],
) -> usize {
    let rank = out_shape.len();
    let hidden_size = out_shape[rank - 1];
    let gate_size = 4 * hidden_size;
    let narrow_axis = rank - 1;

    // Gate shape: output shape with last dim = 4*H.
    let mut gate_shape = out_shape.to_vec();
    gate_shape[rank - 1] = gate_size;

    // Hidden-state gate shape: h_state's leading dims + 4*H.
    let mut hh_gate_shape = st.node_shape(hidden_state).to_vec();
    *hh_gate_shape.last_mut().expect("non-empty") = gate_size;

    // Step 1: ih_out = Linear(input, weight_ih, bias)
    let ih_out = st.alloc();
    st.push(TensorNode::new(
        TensorNodeId::new(ih_out),
        TensorOpKind::Linear {
            input: TensorNodeId::new(input),
            weight: TensorNodeId::new(weight_ih),
            bias: bias.map(TensorNodeId::new),
        },
        gate_shape.clone(),
    ));

    // Step 2: hh_out = Linear(hidden_state, weight_hh, None)
    let hh_out_raw = st.alloc();
    st.push(TensorNode::new(
        TensorNodeId::new(hh_out_raw),
        TensorOpKind::Linear {
            input: TensorNodeId::new(hidden_state),
            weight: TensorNodeId::new(weight_hh),
            bias: None,
        },
        hh_gate_shape.clone(),
    ));

    // Broadcast hh_out to gate_shape if needed ([1,B,4H] → [S,B,4H]).
    let hh_out = if hh_gate_shape != gate_shape {
        let bc = st.alloc();
        st.push(broadcast_node(
            bc,
            hh_out_raw,
            &gate_shape,
            BroadcastAlignment::Right,
        ));
        bc
    } else {
        hh_out_raw
    };

    // Step 3: gates = ih_out + hh_out
    let gates = st.alloc();
    st.push(TensorNode::new(
        TensorNodeId::new(gates),
        TensorOpKind::BinaryAdd {
            left: TensorNodeId::new(ih_out),
            right: TensorNodeId::new(hh_out),
        },
        gate_shape,
    ));

    // Step 4-5: Narrow + activations for each gate
    let i_pre = emit_narrow(st, gates, narrow_axis, 0, hidden_size, out_shape);
    let f_pre = emit_narrow(st, gates, narrow_axis, hidden_size, hidden_size, out_shape);
    let g_pre = emit_narrow(
        st,
        gates,
        narrow_axis,
        2 * hidden_size,
        hidden_size,
        out_shape,
    );
    let o_pre = emit_narrow(
        st,
        gates,
        narrow_axis,
        3 * hidden_size,
        hidden_size,
        out_shape,
    );

    let i_gate = emit_sigmoid(st, i_pre, out_shape);
    let f_gate = emit_sigmoid(st, f_pre, out_shape);
    let g_gate = emit_tanh(st, g_pre, out_shape);
    let o_gate = emit_sigmoid(st, o_pre, out_shape);

    // Broadcast cell_state to out_shape if needed ([1,B,H] → [S,B,H]).
    let c_state_shape = st.node_shape(cell_state).to_vec();
    let cell_bc = if c_state_shape[..] != *out_shape {
        let bc = st.alloc();
        st.push(broadcast_node(
            bc,
            cell_state,
            out_shape,
            BroadcastAlignment::Right,
        ));
        bc
    } else {
        cell_state
    };

    // Step 6: c_new = f * cell_state + i * g
    let fc = emit_binary_mul(st, f_gate, cell_bc, out_shape);
    let ig = emit_binary_mul(st, i_gate, g_gate, out_shape);
    let c_new = emit_binary_add(st, fc, ig, out_shape);

    // Step 7: h_new = o * tanh(c_new)
    let c_new_tanh = emit_tanh(st, c_new, out_shape);
    emit_binary_mul(st, o_gate, c_new_tanh, out_shape)
}

fn emit_narrow(
    st: &mut ExpandState,
    input: usize,
    axis: usize,
    start: usize,
    length: usize,
    shape: &[usize],
) -> usize {
    let id = st.alloc();
    st.push(TensorNode::new(
        TensorNodeId::new(id),
        TensorOpKind::Narrow {
            input: TensorNodeId::new(input),
            axis,
            start,
            length,
        },
        shape.to_vec(),
    ));
    id
}

fn emit_sigmoid(st: &mut ExpandState, input: usize, shape: &[usize]) -> usize {
    let id = st.alloc();
    st.push(TensorNode::new(
        TensorNodeId::new(id),
        TensorOpKind::Sigmoid {
            input: TensorNodeId::new(input),
        },
        shape.to_vec(),
    ));
    id
}

fn emit_tanh(st: &mut ExpandState, input: usize, shape: &[usize]) -> usize {
    let id = st.alloc();
    st.push(TensorNode::new(
        TensorNodeId::new(id),
        TensorOpKind::Tanh {
            input: TensorNodeId::new(input),
        },
        shape.to_vec(),
    ));
    id
}

fn emit_binary_mul(st: &mut ExpandState, left: usize, right: usize, shape: &[usize]) -> usize {
    let id = st.alloc();
    st.push(TensorNode::new(
        TensorNodeId::new(id),
        TensorOpKind::BinaryMul {
            left: TensorNodeId::new(left),
            right: TensorNodeId::new(right),
        },
        shape.to_vec(),
    ));
    id
}

fn emit_binary_add(st: &mut ExpandState, left: usize, right: usize, shape: &[usize]) -> usize {
    let id = st.alloc();
    st.push(TensorNode::new(
        TensorNodeId::new(id),
        TensorOpKind::BinaryAdd {
            left: TensorNodeId::new(left),
            right: TensorNodeId::new(right),
        },
        shape.to_vec(),
    ));
    id
}

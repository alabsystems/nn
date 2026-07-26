// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! LSTM cell decomposition into primitive tensor ops.
//!
//! Decomposes a single-timestep LSTM cell into: 2× Linear, 1× BinaryAdd,
//! 4× Narrow, 3× Sigmoid, 2× Tanh, 3× BinaryMul, 1× BinaryAdd = 16 ops.
//! Every primitive op has a working Metal dispatch path, so the decomposed
//! LSTM runs on GPU without a monolithic LSTM kernel.
//!
//! PyTorch `nn.LSTMCell` gate ordering: `[i, f, g, o]` (input, forget,
//! cell-candidate, output). Weight shapes: `weight_ih: [4*H, I]`,
//! `weight_hh: [4*H, H]`, `bias: [4*H]` (optional).
//!
//! Part of #761 — Direction 3 (LSTM decomposition for Silero VAD).

use crate::tensor_block_builder::TensorBlockBuilder;
use crate::tensor_ir::{TensorIRError, TensorIRLayerError, TensorKernelDef, TensorNodeId};

/// Result of an LSTM cell decomposition: both output states.
#[derive(Debug, Clone, Copy)]
pub struct LstmCellOutputs {
    /// New hidden state h_new = o * tanh(c_new). Shape: [batch, hidden_size].
    pub h_new: TensorNodeId,
    /// New cell state c_new = f * c + i * g. Shape: [batch, hidden_size].
    pub c_new: TensorNodeId,
}

fn validate_lstm_dimensions(
    input_size: usize,
    hidden_size: usize,
    batch: usize,
) -> Result<(), TensorIRError> {
    if input_size == 0 {
        return Err(TensorIRLayerError::LstmZeroDimension {
            param: "input_size",
        }
        .into());
    }
    if hidden_size == 0 {
        return Err(TensorIRLayerError::LstmZeroDimension {
            param: "hidden_size",
        }
        .into());
    }
    if batch == 0 {
        return Err(TensorIRLayerError::LstmZeroDimension { param: "batch" }.into());
    }
    Ok(())
}

/// Decompose an LSTM cell into primitive ops within an existing builder.
///
/// Takes input, hidden state, cell state, and weight/bias node IDs that
/// have already been added to the builder via `add_input`. Returns node
/// IDs for both output states (h_new, c_new).
///
/// # Gate decomposition
///
/// ```text
/// gates = Linear(input, weight_ih, bias) + Linear(hidden, weight_hh, None)
/// i = sigmoid(narrow(gates, axis=1, 0,   H))   // input gate
/// f = sigmoid(narrow(gates, axis=1, H,   H))   // forget gate
/// g = tanh(narrow(gates, axis=1, 2*H, H))      // cell candidate
/// o = sigmoid(narrow(gates, axis=1, 3*H, H))   // output gate
/// c_new = f * cell_state + i * g
/// h_new = o * tanh(c_new)
/// ```
pub(crate) fn decompose_lstm_cell(
    builder: &mut TensorBlockBuilder,
    input: TensorNodeId,
    hidden_state: TensorNodeId,
    cell_state: TensorNodeId,
    weight_ih: TensorNodeId,
    weight_hh: TensorNodeId,
    bias: Option<TensorNodeId>,
    hidden_size: usize,
    batch: usize,
) -> LstmCellOutputs {
    let gate_size = 4 * hidden_size;
    let gate_shape = [batch, gate_size];
    let h_shape = [batch, hidden_size];

    let ih_out = builder.add_linear(input, weight_ih, bias, &gate_shape);
    let hh_out = builder.add_linear(hidden_state, weight_hh, None, &gate_shape);
    let gates = builder.add_binary_add(ih_out, hh_out, &gate_shape);

    let i_pre = builder.add_narrow(gates, 1, 0, hidden_size, &h_shape);
    let f_pre = builder.add_narrow(gates, 1, hidden_size, hidden_size, &h_shape);
    let g_pre = builder.add_narrow(gates, 1, 2 * hidden_size, hidden_size, &h_shape);
    let o_pre = builder.add_narrow(gates, 1, 3 * hidden_size, hidden_size, &h_shape);

    let i = builder.add_sigmoid(i_pre, &h_shape);
    let f = builder.add_sigmoid(f_pre, &h_shape);
    let g = builder.add_tanh(g_pre, &h_shape);
    let o = builder.add_sigmoid(o_pre, &h_shape);

    let fc = builder.add_binary_mul(f, cell_state, &h_shape);
    let ig = builder.add_binary_mul(i, g, &h_shape);
    let c_new = builder.add_binary_add(fc, ig, &h_shape);

    let c_new_tanh = builder.add_tanh(c_new, &h_shape);
    let h_new = builder.add_binary_mul(o, c_new_tanh, &h_shape);

    LstmCellOutputs { h_new, c_new }
}

/// Build a complete decomposed LSTM cell as a standalone `TensorKernelDef`.
///
/// # Errors
///
/// Returns `TensorIRLayerError::LstmZeroDimension` if any dimension is zero.
#[allow(dead_code)] // Called from #[cfg(test)] only
pub(crate) fn build_lstm_cell_decomposed(
    input_size: usize,
    hidden_size: usize,
    batch: usize,
    with_bias: bool,
) -> Result<TensorKernelDef, TensorIRError> {
    validate_lstm_dimensions(input_size, hidden_size, batch)?;
    let mut builder = TensorBlockBuilder::new("lstm_cell_decomposed");
    let input = builder.add_input(crate::input_names::DATA, &[batch, input_size]);
    let hidden = builder.add_input(crate::input_names::HIDDEN_STATE, &[batch, hidden_size]);
    let cell = builder.add_input(crate::input_names::CELL_STATE, &[batch, hidden_size]);
    let w_ih = builder.add_input(
        crate::input_names::WEIGHT_IH,
        &[4 * hidden_size, input_size],
    );
    let w_hh = builder.add_input(
        crate::input_names::WEIGHT_HH,
        &[4 * hidden_size, hidden_size],
    );
    let bias = if with_bias {
        Some(builder.add_input(crate::input_names::BIAS, &[4 * hidden_size]))
    } else {
        None
    };
    let outputs = decompose_lstm_cell(
        &mut builder,
        input,
        hidden,
        cell,
        w_ih,
        w_hh,
        bias,
        hidden_size,
        batch,
    );
    builder.build(outputs.h_new)
}

/// Build a decomposed LSTM cell that outputs both h_new and c_new.
///
/// Stacks `[h_new, c_new]` along axis 0 to produce a single output tensor
/// of shape `[2, batch, hidden_size]`.
///
/// Axis-0 stacking enables zero-copy narrow for both h and c extraction
/// regardless of batch size — dim-0 narrow is always contiguous in row-major
/// layout, so `MetalBuffer::alias()` with byte offset suffices (no GPU kernel).
///
/// # Errors
///
/// Returns `TensorIRLayerError::LstmZeroDimension` if any dimension is zero.
pub fn build_lstm_cell_decomposed_dual(
    input_size: usize,
    hidden_size: usize,
    batch: usize,
    with_bias: bool,
) -> Result<TensorKernelDef, TensorIRError> {
    validate_lstm_dimensions(input_size, hidden_size, batch)?;
    let mut builder = TensorBlockBuilder::new("lstm_cell_decomposed_dual");
    let input = builder.add_input(crate::input_names::DATA, &[batch, input_size]);
    let hidden = builder.add_input(crate::input_names::HIDDEN_STATE, &[batch, hidden_size]);
    let cell = builder.add_input(crate::input_names::CELL_STATE, &[batch, hidden_size]);
    let w_ih = builder.add_input(
        crate::input_names::WEIGHT_IH,
        &[4 * hidden_size, input_size],
    );
    let w_hh = builder.add_input(
        crate::input_names::WEIGHT_HH,
        &[4 * hidden_size, hidden_size],
    );
    let bias = if with_bias {
        Some(builder.add_input(crate::input_names::BIAS, &[4 * hidden_size]))
    } else {
        None
    };
    let outputs = decompose_lstm_cell(
        &mut builder,
        input,
        hidden,
        cell,
        w_ih,
        w_hh,
        bias,
        hidden_size,
        batch,
    );
    let stacked = builder.add_stack(&[outputs.h_new, outputs.c_new], 0, &[2, batch, hidden_size]);
    builder.build(stacked)
}

#[cfg(test)]
#[path = "lstm_decomposed_tests.rs"]
mod tests;

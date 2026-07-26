// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! LSTM trace compilation helper.
//!
//! Extracted from `trace_compile_ops.rs` to stay within the 450-line limit.
//! Handles both 3-input (explicit h/c states) and 1-input (zero-initialized
//! states from `forward_seq` with `None`) cases.

use std::collections::HashMap;

use nn_core::dyn_tensor::trace::{ComputationGraph, TraceNode, WeightRef};

use crate::tensor_block_builder::TensorBlockBuilder;
use crate::tensor_ir::{TensorIRError, TensorNodeId};

use super::super::{resolve_input_shape, CompiledKernel, CompiledStep, NativeOpKind};
use super::{add_weight, build_op_with_weights};

pub(in crate::trace_compile) fn compile_lstm(
    node: &TraceNode,
    graph: &ComputationGraph,
    weight_ih: &WeightRef,
    weight_hh: &WeightRef,
    bias_ih: &Option<WeightRef>,
    bias_hh: &Option<WeightRef>,
    hidden_size: usize,
) -> Result<CompiledStep, TensorIRError> {
    let input_shape = resolve_input_shape(node, 0, graph)?;

    // 3D input = sequence LSTM → delegate to fused Metal kernel via NativeOp.
    // The IR expansion path (`emit_lstm_cell`) only handles 2D [batch, hidden]
    // single-timestep inputs. For 3D [seq_len, batch, input_size] sequences,
    // NativeOp dispatches to the existing `gpu_lstm_sequence` kernel which
    // processes the full sequence in a single Metal compute dispatch.
    if input_shape.len() == 3 {
        return compile_lstm_sequence(
            input_shape,
            weight_ih,
            weight_hh,
            bias_ih,
            bias_hh,
            hidden_size,
        );
    }

    let num_inputs = node.inputs().len();

    // When LSTM is traced with None initial states (forward_seq with no
    // state), the trace only records 1 input (data). Synthesize zero h/c
    // as weight data. When all 3 inputs are present, use graph shapes.
    let zero_state: Option<WeightRef> = if num_inputs < 3 {
        let batch = if input_shape.len() >= 2 {
            input_shape[input_shape.len() - 2]
        } else {
            1
        };
        let shape = vec![batch, hidden_size];
        let data = vec![0.0f32; batch * hidden_size];
        Some(
            WeightRef::new(data, shape).map_err(|_| TensorIRError::UnsupportedTraceOp {
                name: "lstm: invalid zero-state shape".into(),
            })?,
        )
    } else {
        None
    };

    // Resolve explicit state shapes outside the closure (where `?` works).
    let explicit_shapes = if zero_state.is_none() {
        Some((
            resolve_input_shape(node, 1, graph)?,
            resolve_input_shape(node, 2, graph)?,
        ))
    } else {
        None
    };

    let (def, weight_data) = build_op_with_weights("lstm", node, |b, wd| {
        let input = b.add_input("input_0", input_shape);

        let (h_state, c_state) = if let Some(ref wref) = zero_state {
            (
                add_weight(b, wd, "zero_h", wref),
                add_weight(b, wd, "zero_c", wref),
            )
        } else {
            // SAFETY: explicit_shapes is Some when zero_state is None (set above).
            let (h_shape, c_shape) = explicit_shapes
                .as_ref()
                .expect("invariant: explicit_shapes set when zero_state is None");
            (
                b.add_input("hidden_state", h_shape),
                b.add_input("cell_state", c_shape),
            )
        };

        let wih = add_weight(b, wd, "weight_ih", weight_ih);
        let whh = add_weight(b, wd, "weight_hh", weight_hh);
        let combined_bias = combine_lstm_biases(b, wd, bias_ih, bias_hh);
        b.add_lstm(
            input,
            h_state,
            c_state,
            wih,
            whh,
            combined_bias,
            node.output_shape(),
        )
    })?;
    Ok(CompiledStep::Dispatch {
        kernel: CompiledKernel::new(def),
        weight_data,
        external_node_ids: super::graph_input_ids(node, num_inputs.min(3)),
    })
}

/// Compile a 3D sequence LSTM as a `NativeOp` that delegates to the
/// existing fused `gpu_lstm_sequence` Metal kernel.
///
/// Input shape: `[seq_len, batch, input_size]`
/// Output shape: `[seq_len, batch, hidden_size]` (from the trace node)
fn compile_lstm_sequence(
    input_shape: &[usize],
    weight_ih: &WeightRef,
    weight_hh: &WeightRef,
    bias_ih: &Option<WeightRef>,
    bias_hh: &Option<WeightRef>,
    hidden_size: usize,
) -> Result<CompiledStep, TensorIRError> {
    let batch = input_shape[1];
    let h_shape = vec![batch, hidden_size];

    let input_size = input_shape[2];
    let mut weight_data = HashMap::new();
    weight_data.insert("weight_ih".to_string(), weight_ih.clone());
    weight_data.insert("weight_hh".to_string(), weight_hh.clone());

    // Pre-transpose weight_ih from [4*H, I] to [I, 4*H] at compile time.
    // The precomputed LSTM path uses `encode_simdgroup_matmul_into_batch`
    // which requires physically transposed data (no stride/view support).
    // Part of #2981 (LSTM GEMM in-plan dispatch), restored in #3491.
    let rows = 4 * hidden_size;
    let cols = input_size;
    let src = weight_ih.data();
    if src.len() == rows * cols {
        let mut transposed = vec![0.0f32; cols * rows];
        for r in 0..rows {
            for c in 0..cols {
                transposed[c * rows + r] = src[r * cols + c];
            }
        }
        let w_ih_t = WeightRef::new(transposed, vec![cols, rows]).map_err(|_| {
            TensorIRError::UnsupportedTraceOp {
                name: "lstm_sequence: invalid weight_ih_t shape".into(),
            }
        })?;
        weight_data.insert("weight_ih_t".to_string(), w_ih_t);
    }

    // Pre-combine biases at compile time to avoid a per-dispatch GPU add.
    // Previously stored separately ("bias_ih", "bias_hh") and added at
    // dispatch time — 1 extra GPU dispatch per LSTM call (8 in Kokoro).
    // Part of #2981, restored in #3491.
    if let (Some(bih), Some(bhh)) = (bias_ih, bias_hh) {
        let combined_data: Vec<f32> = bih
            .data()
            .iter()
            .zip(bhh.data().iter())
            .map(|(a, b)| a + b)
            .collect();
        let combined = WeightRef::new(combined_data, bih.shape().to_vec()).map_err(|_| {
            TensorIRError::UnsupportedTraceOp {
                name: "lstm_sequence: invalid combined bias shape".into(),
            }
        })?;
        weight_data.insert("bias".to_string(), combined);
    } else if let Some(b) = bias_ih.as_ref().or(bias_hh.as_ref()) {
        weight_data.insert("bias".to_string(), b.clone());
    }

    // Synthesize zero initial states as weight data.
    let zero_h =
        WeightRef::new(vec![0.0f32; batch * hidden_size], h_shape.clone()).map_err(|_| {
            TensorIRError::UnsupportedTraceOp {
                name: "lstm_sequence: invalid zero-state shape".into(),
            }
        })?;
    let zero_c =
        WeightRef::new(vec![0.0f32; batch * hidden_size], h_shape.clone()).map_err(|_| {
            TensorIRError::UnsupportedTraceOp {
                name: "lstm_sequence: invalid zero-state shape".into(),
            }
        })?;
    weight_data.insert("h0".to_string(), zero_h);
    weight_data.insert("c0".to_string(), zero_c);

    Ok(CompiledStep::NativeOp {
        op: NativeOpKind::LstmSequence {
            hidden_size,
            input_shape: input_shape.to_vec(),
            h_shape,
            reverse: false,
        },
        weight_data,
    })
}

fn combine_lstm_biases(
    b: &mut TensorBlockBuilder,
    wd: &mut HashMap<String, WeightRef>,
    bias_ih: &Option<WeightRef>,
    bias_hh: &Option<WeightRef>,
) -> Option<TensorNodeId> {
    match (bias_ih, bias_hh) {
        (Some(bih), Some(bhh)) => {
            let bih_id = add_weight(b, wd, "bias_ih", bih);
            let bhh_id = add_weight(b, wd, "bias_hh", bhh);
            Some(b.add_binary_add(bih_id, bhh_id, bih.shape()))
        }
        (Some(bih), None) => Some(add_weight(b, wd, "bias_ih", bih)),
        (None, Some(bhh)) => Some(add_weight(b, wd, "bias_hh", bhh)),
        (None, None) => None,
    }
}

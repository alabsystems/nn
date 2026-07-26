// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! `compile_node()` dispatch: maps each `TraceOp` to a `CompiledStep`.
//!
//! Delegates to category dispatchers so that adding a new op only requires
//! touching the relevant category file, not this shared hub. See #2305.
//!
//! Categories:
//! - **elementwise**: binary, unary, min/max, activations, reductions, powf
//! - **structured**: linear, matmul, conv, norm, pool, embedding, LSTM, softmax
//! - **shape**: narrow, cat, transpose, permute, expand, cumsum, flip, clamp
//! - **composite**: attention, selection, unfold, upsample, pixel shuffle

use std::collections::HashMap;

use nn_core::dyn_tensor::trace::{ComputationGraph, TraceNode, TraceOp, WeightRef};

use crate::tensor_ir::TensorIRError;

use super::CompiledStep;

#[path = "trace_compile_dispatch_composite.rs"]
mod composite;
#[path = "trace_compile_dispatch_elementwise.rs"]
mod elementwise;
#[path = "trace_compile_dispatch_shape.rs"]
mod shape;
#[path = "trace_compile_dispatch_structured.rs"]
mod structured;

/// Compile a single trace node into a `CompiledStep`.
///
/// Identity and passthrough ops are handled inline. All other ops delegate
/// to category dispatchers that return `Option<Result<...>>`.
pub(super) fn compile_node(
    node: &TraceNode,
    graph: &ComputationGraph,
) -> Result<CompiledStep, TensorIRError> {
    // -- Identity ops (no dispatch needed) ------------------------------------
    match node.op() {
        TraceOp::Input => return Ok(CompiledStep::InputForward),
        TraceOp::Constant { value } => {
            return Ok(CompiledStep::ConstantValue {
                value: *value,
                shape: node.output_shape().to_vec(),
            });
        }
        TraceOp::ConstantWeight { weight } => {
            let mut weight_data = HashMap::new();
            weight_data.insert("constant_weight".to_string(), weight.clone());
            return Ok(CompiledStep::NativeOp {
                op: super::NativeOpKind::ConstantWeight {
                    name: "constant_weight".into(),
                    shape: node.output_shape().to_vec(),
                },
                weight_data,
            });
        }
        TraceOp::Dropout => return Ok(CompiledStep::IdentityPassthrough),
        TraceOp::Arange { start, end, step } => {
            return compile_arange(*start, *end, *step, node.output_shape());
        }

        // -- Shape-only passthroughs (metadata manipulation) ------------------
        TraceOp::Reshape { target_shape } => {
            return Ok(CompiledStep::Passthrough {
                op_name: "reshape".into(),
                output_shape: target_shape.clone(),
            });
        }
        TraceOp::Unsqueeze { .. } | TraceOp::Squeeze { .. } => {
            return Ok(CompiledStep::Passthrough {
                op_name: node.op().canonical_name().to_string(),
                output_shape: node.output_shape().to_vec(),
            });
        }
        TraceOp::ToDtype { .. } => {
            return Ok(CompiledStep::Passthrough {
                op_name: "to_dtype".into(),
                output_shape: node.output_shape().to_vec(),
            });
        }
        _ => {}
    }

    // -- Category dispatchers (each in its own file) --------------------------
    if let Some(result) = elementwise::try_compile(node, graph) {
        return result;
    }
    if let Some(result) = structured::try_compile(node, graph) {
        return result;
    }
    if let Some(result) = shape::try_compile(node, graph) {
        return result;
    }
    if let Some(result) = composite::try_compile(node, graph) {
        return result;
    }

    // -- Everything else is unsupported for now -------------------------------
    Err(TensorIRError::UnsupportedTraceOp {
        name: node.op().canonical_name().to_string(),
    })
}

/// Compile `Arange { start, end, step }` as a pre-computed constant weight.
///
/// All parameters are compile-time constants, so we compute the full
/// output vector and embed it as a weight reference.
fn compile_arange(
    start: f64,
    end: f64,
    step: f64,
    output_shape: &[usize],
) -> Result<CompiledStep, TensorIRError> {
    if !start.is_finite() || !end.is_finite() || !step.is_finite() || step == 0.0 {
        return Err(TensorIRError::NonFiniteConstant {
            name: "arange parameters".into(),
            value: if !start.is_finite() {
                start
            } else if !end.is_finite() {
                end
            } else {
                step
            },
        });
    }

    // Derive element count from the trace graph's output shape when available.
    // The ceil formula can disagree with output_shape due to floating-point
    // rounding at boundaries (e.g., arange(0, 1, 0.1) → ceil might give 11
    // but trace says [10]). The trace shape is authoritative.
    let n = if output_shape.is_empty() {
        ((end - start) / step).ceil().max(0.0) as usize
    } else {
        output_shape
            .iter()
            .try_fold(1usize, |acc, &d| acc.checked_mul(d))
            .ok_or_else(|| TensorIRError::ShapeOverflow {
                shape: output_shape.to_vec(),
            })?
    };
    let data: Vec<f32> = (0..n).map(|i| (start + (i as f64) * step) as f32).collect();
    let shape = if output_shape.is_empty() {
        vec![n]
    } else {
        output_shape.to_vec()
    };

    let weight =
        WeightRef::new(data, shape.clone()).map_err(|_| TensorIRError::NonFiniteConstant {
            name: "arange output".into(),
            value: start,
        })?;

    let mut weight_data = HashMap::new();
    weight_data.insert("arange_data".to_string(), weight);

    Ok(CompiledStep::NativeOp {
        op: super::NativeOpKind::ConstantWeight {
            name: "arange".into(),
            shape,
        },
        weight_data,
    })
}

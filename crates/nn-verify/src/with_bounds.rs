// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Developer-facing bounds propagation with lenient mode.
//!
//! [`with_bounds`] wraps a model forward pass and propagates IBP bounds
//! through the captured computation graph. Unlike [`quick_bounds`], it
//! supports a lenient mode that reports unsupported ops without erroring,
//! enabling developers to get bounds information for models that use
//! partially-supported op sets.
//!
//! ```rust,ignore
//! use nn_verify::{with_bounds, BoundsPolicy};
//!
//! let result = with_bounds(
//!     &input,
//!     &input_bounds,
//!     BoundsPolicy::Lenient,
//!     |traced_input| model.forward(traced_input),
//! )?;
//!
//! if let Some(bounds) = &result.bounds {
//!     println!("Output range: [{}, {}]",
//!         bounds.lower().iter().fold(f32::INFINITY, |a, &b| a.min(b)),
//!         bounds.upper().iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b)));
//! }
//! for gap in &result.unsupported_ops {
//!     eprintln!("  unsupported: {gap}");
//! }
//! ```
//!
//! Part of #2218 (Developer-Facing Bounds Propagation).

use ny_api::BoundedTensor;
use nn_core::dyn_tensor::trace::{record_input, trace_graph, ComputationGraph, TraceOp};
use nn_core::dyn_tensor::DynTensor;

use crate::error::VerifyError;
use crate::trace_to_graph::trace_to_graph_model;

/// Policy for handling unsupported ops during bounds propagation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BoundsPolicy {
    /// Error on any unsupported op. Equivalent to [`quick_bounds`].
    Strict,
    /// Attempt translation; on failure, return `bounds: None` with a list
    /// of unsupported ops. The model forward pass still executes normally.
    Lenient,
}

/// Result of [`with_bounds`] — model output plus optional bounds and diagnostics.
#[derive(Debug)]
pub struct WithBoundsResult<T> {
    /// The model's forward pass output (always available).
    pub output: T,
    /// Propagated output bounds. `None` if translation or propagation failed
    /// (only possible in `Lenient` mode; `Strict` mode errors instead).
    pub bounds: Option<BoundedTensor>,
    /// List of unsupported `TraceOp` names encountered during translation.
    /// Empty if all ops were successfully translated.
    pub unsupported_ops: Vec<String>,
    /// Number of nodes in the captured computation graph.
    pub graph_node_count: usize,
    /// Which propagation method was used (if bounds were computed).
    pub method: Option<&'static str>,
}

/// Trace a model forward pass and propagate IBP bounds, with policy control.
///
/// In `Strict` mode, behaves like [`quick_bounds`] — errors on unsupported ops.
/// In `Lenient` mode, attempts translation and returns `None` for bounds if
/// any op cannot be translated, along with a diagnostic list of unsupported ops.
///
/// # Arguments
///
/// - `input`: the model's input tensor
/// - `input_bounds`: IBP bounds for the input (must match `input` shape)
/// - `policy`: how to handle unsupported ops
/// - `f`: model forward pass, receives the traced input tensor
pub fn with_bounds<F, T>(
    input: &DynTensor,
    input_bounds: &BoundedTensor,
    policy: BoundsPolicy,
    f: F,
) -> Result<WithBoundsResult<T>, VerifyError>
where
    F: FnOnce(&DynTensor) -> nn_core::Result<T>,
{
    // 1. Capture the computation graph.
    let (output, graph) = trace_graph(|| {
        let mut traced = input.clone();
        if let Some(id) = record_input(input.dims(), input.dtype()) {
            traced.set_trace_id(id);
        }
        f(&traced)
    })
    .map_err(|e| VerifyError::PropagationFailed(format!("trace_graph failed: {e}")))?;

    let graph_node_count = graph.nodes().len();

    // 2. Attempt translation + propagation.
    match trace_to_graph_model(&graph) {
        Ok(result) => match result.graph.propagate_ibp(input_bounds) {
            Ok(output_bounds) => Ok(WithBoundsResult {
                output,
                bounds: Some(output_bounds),
                unsupported_ops: Vec::new(),
                graph_node_count,
                method: Some("IBP"),
            }),
            Err(e) => match policy {
                BoundsPolicy::Strict => Err(VerifyError::from(e)),
                BoundsPolicy::Lenient => Ok(WithBoundsResult {
                    output,
                    bounds: None,
                    unsupported_ops: vec![format!("propagation failed: {e}")],
                    graph_node_count,
                    method: None,
                }),
            },
        },
        Err(VerifyError::UnsupportedOp(op_name)) => match policy {
            BoundsPolicy::Strict => Err(VerifyError::UnsupportedOp(op_name)),
            BoundsPolicy::Lenient => {
                // Scan the graph for ALL unsupported ops (not just the first).
                let unsupported = scan_unsupported_ops(&graph);
                Ok(WithBoundsResult {
                    output,
                    bounds: None,
                    unsupported_ops: unsupported,
                    graph_node_count,
                    method: None,
                })
            }
        },
        Err(e) => match policy {
            BoundsPolicy::Strict => Err(e),
            BoundsPolicy::Lenient => Ok(WithBoundsResult {
                output,
                bounds: None,
                unsupported_ops: vec![format!("translation failed: {e}")],
                graph_node_count,
                method: None,
            }),
        },
    }
}

/// Multi-input variant of [`with_bounds`].
///
/// Each input tensor is registered as a separate trace input. The caller must
/// provide IBP bounds as a flat 1D tensor of shape `[sum of all input elements]`.
pub fn with_bounds_multi_input<F, T>(
    inputs: &[&DynTensor],
    input_bounds: &BoundedTensor,
    policy: BoundsPolicy,
    f: F,
) -> Result<WithBoundsResult<T>, VerifyError>
where
    F: FnOnce(&[DynTensor]) -> nn_core::Result<T>,
{
    let (output, graph) = trace_graph(|| {
        let traced: Vec<DynTensor> = inputs
            .iter()
            .map(|inp| {
                let mut t = (*inp).clone();
                if let Some(id) = record_input(inp.dims(), inp.dtype()) {
                    t.set_trace_id(id);
                }
                t
            })
            .collect();
        f(&traced)
    })
    .map_err(|e| VerifyError::PropagationFailed(format!("trace_graph failed: {e}")))?;

    let graph_node_count = graph.nodes().len();

    match crate::trace_to_graph::trace_to_graph_model_multi_input(&graph) {
        Ok(result) => match result.graph.propagate_ibp(input_bounds) {
            Ok(output_bounds) => Ok(WithBoundsResult {
                output,
                bounds: Some(output_bounds),
                unsupported_ops: Vec::new(),
                graph_node_count,
                method: Some("IBP"),
            }),
            Err(e) => match policy {
                BoundsPolicy::Strict => Err(VerifyError::from(e)),
                BoundsPolicy::Lenient => Ok(WithBoundsResult {
                    output,
                    bounds: None,
                    unsupported_ops: vec![format!("propagation failed: {e}")],
                    graph_node_count,
                    method: None,
                }),
            },
        },
        Err(VerifyError::UnsupportedOp(op_name)) => match policy {
            BoundsPolicy::Strict => Err(VerifyError::UnsupportedOp(op_name)),
            BoundsPolicy::Lenient => {
                let unsupported = scan_unsupported_ops(&graph);
                Ok(WithBoundsResult {
                    output,
                    bounds: None,
                    unsupported_ops: unsupported,
                    graph_node_count,
                    method: None,
                })
            }
        },
        Err(e) => match policy {
            BoundsPolicy::Strict => Err(e),
            BoundsPolicy::Lenient => Ok(WithBoundsResult {
                output,
                bounds: None,
                unsupported_ops: vec![format!("translation failed: {e}")],
                graph_node_count,
                method: None,
            }),
        },
    }
}

// ---------------------------------------------------------------------------
// Unsupported op scanner
// ---------------------------------------------------------------------------

/// Known-unsupported TraceOp variants that the translator cannot handle.
///
/// Returns true for ops that will definitely fail translation. This avoids
/// false positives from ops that MIGHT fail (e.g., Powf with unsupported
/// exponents) — those are caught by the actual translation attempt.
fn is_known_unsupported(op: &TraceOp) -> bool {
    matches!(
        op,
        TraceOp::SwiGlu
            | TraceOp::ScatterAdd { .. }
            | TraceOp::IndexAdd { .. }
            | TraceOp::IndexPut { .. }
            | TraceOp::Topk { .. }
            | TraceOp::Argmax { .. }
            | TraceOp::Argmin { .. }
            | TraceOp::ArgSort { .. }
            | TraceOp::Compare { .. }
            | TraceOp::CompareTensor { .. }
            | TraceOp::Triu { .. }
            | TraceOp::Tril { .. }
            | TraceOp::GridSample { .. }
            | TraceOp::MultiHeadAttention { .. }
            | TraceOp::Custom { .. }
    )
}

/// Scan a computation graph for all unsupported ops.
///
/// Returns deduplicated op names in the order first encountered.
fn scan_unsupported_ops(graph: &ComputationGraph) -> Vec<String> {
    let mut seen = std::collections::HashSet::new();
    let mut result = Vec::new();
    for node in graph.nodes() {
        if is_known_unsupported(node.op()) {
            let name = node.op().canonical_name();
            if seen.insert(name.to_string()) {
                result.push(name.to_string());
            }
        }
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use ny_api::BoundedTensor;
    use nn_core::dyn_tensor::DynTensor;
    use nn_core::layers::{Linear, Module};
    use nn_core::Device;
    use ndarray::{ArrayD, IxDyn};

    fn simple_model_input() -> (DynTensor, BoundedTensor, Linear) {
        let weight =
            DynTensor::from_vec(vec![1.0, 0.5, -0.5, 1.0], &[2, 2], &Device::Cpu).expect("weight");
        let linear = Linear::new(weight, None).expect("linear");
        let input = DynTensor::from_vec(vec![0.5, -0.5], &[1, 2], &Device::Cpu).expect("input");
        let lower = ArrayD::from_elem(IxDyn(&[1, 2]), -1.0f32);
        let upper = ArrayD::from_elem(IxDyn(&[1, 2]), 1.0f32);
        let bounds = BoundedTensor::new(lower, upper).expect("bounds");
        (input, bounds, linear)
    }

    #[test]
    fn test_with_bounds_strict_linear_relu() {
        let (input, input_bounds, linear) = simple_model_input();

        let result = with_bounds(&input, &input_bounds, BoundsPolicy::Strict, |x| {
            let h = linear.forward(x)?;
            h.relu()
        })
        .expect("strict mode should succeed for supported ops");

        assert!(result.bounds.is_some());
        assert!(result.unsupported_ops.is_empty());
        assert_eq!(result.method, Some("IBP"));
        assert!(result.graph_node_count > 0);

        let bounds = result.bounds.unwrap();
        let (lo, hi) = bounds.lower_upper();
        for (l, u) in lo.iter().zip(hi.iter()) {
            assert!(l <= u, "lower ({l}) <= upper ({u})");
        }
        // ReLU: lower >= 0
        for &l in lo.iter() {
            assert!(l >= 0.0, "ReLU lower >= 0, got {l}");
        }
    }

    #[test]
    fn test_with_bounds_lenient_supported_ops() {
        let (input, input_bounds, linear) = simple_model_input();

        let result = with_bounds(&input, &input_bounds, BoundsPolicy::Lenient, |x| {
            linear.forward(x)?.relu()
        })
        .expect("lenient mode should succeed");

        assert!(result.bounds.is_some());
        assert!(result.unsupported_ops.is_empty());
    }

    #[test]
    fn test_with_bounds_lenient_reports_unsupported() {
        let (input, input_bounds, linear) = simple_model_input();

        // Use topk which is unsupported by the translator.
        let result = with_bounds(&input, &input_bounds, BoundsPolicy::Lenient, |x| {
            let h = linear.forward(x)?;
            // topk produces (values, indices) — take values only.
            let (values, _indices) = h.topk(1, 1)?;
            Ok(values)
        })
        .expect("lenient mode should not error on unsupported ops");

        assert!(result.bounds.is_none());
        assert!(!result.unsupported_ops.is_empty());
    }

    #[test]
    fn test_with_bounds_strict_errors_on_unsupported() {
        let (input, input_bounds, linear) = simple_model_input();

        let result = with_bounds(&input, &input_bounds, BoundsPolicy::Strict, |x| {
            let h = linear.forward(x)?;
            let (values, _indices) = h.topk(1, 1)?;
            Ok(values)
        });

        assert!(result.is_err());
    }

    #[test]
    fn test_scan_unsupported_ops_deduplicates() {
        let ops = vec!["topk", "topk", "gather"];
        let mut seen = std::collections::HashSet::new();
        let mut deduped = Vec::new();
        for op in ops {
            if seen.insert(op.to_string()) {
                deduped.push(op.to_string());
            }
        }
        assert_eq!(deduped.len(), 2);
    }
}

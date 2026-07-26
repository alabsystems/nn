// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Auto-verify a pre-traced `ComputationGraph`.
//!
//! Takes an already-captured `ComputationGraph` and runs IBP bound
//! propagation through the translated NY `GraphNetwork`.
//!
//! This is the primary verification API for traced models:
//!
//! ```rust,ignore
//! use nn_core::dyn_tensor::trace::trace_graph;
//! use nn_verify::verify_trace;
//! use ny_api::BoundedTensor;
//!
//! let (_output, graph) = trace_graph(|| model.forward(&input))?;
//! let bounds = BoundedTensor::from_epsilon(&input_values, 0.01)?;
//! let result = verify_trace(&graph, &bounds)?;
//! println!("IBP width: {}", result.ibp_width);
//! ```
//!
//! Unlike [`quick_bounds`](super::quick_bounds) which traces a new forward
//! pass, this function works on pre-existing graphs — useful for the Kokoro
//! compiled pipeline where segments are traced during compilation.
//!
//! Auto-detects single vs multi-input graphs. For multi-input graphs,
//! `input_bounds` must be a flat 1D tensor of shape `[sum of all input elements]`.
//!
//! Part of #3029, #2218.

use ny_api::BoundedTensor;
use ny_propagate::GraphNetwork;
use nn_core::dyn_tensor::trace::{ComputationGraph, TraceOp};

use crate::error::VerifyError;
use crate::trace_to_graph::trace_to_graph_model_multi_input;

/// Result of verifying a traced computation graph.
///
/// Contains the IBP output bounds, the translated `GraphNetwork` (for
/// optional CROWN tightening by the caller), and diagnostic metrics.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct VerifyTraceResult {
    /// The NY `GraphNetwork` translated from the trace.
    ///
    /// Callers can use this for CROWN tightening:
    /// ```rust,ignore
    /// let crown = result.network.propagate_crown(&input_bounds)?;
    /// ```
    pub network: GraphNetwork,
    /// IBP output bounds (always computed).
    pub ibp_bounds: BoundedTensor,
    /// Number of nodes in the computation graph.
    pub node_count: usize,
    /// Number of `TraceOp::Input` nodes in the graph.
    pub input_count: usize,
    /// IBP bound width: max(upper - lower) across all output elements.
    ///
    /// Lower is better. Width 0.0 means the output is provably constant.
    /// Width > 1e6 suggests vacuous bounds (common with CROWN through norms).
    pub ibp_width: f32,
}

impl VerifyTraceResult {
    /// Whether the IBP bounds are tight enough to be useful (width < 100.0).
    ///
    /// Vacuous bounds (width > 1e6) provide no verification value.
    /// This threshold matches the "vacuous" classification in the
    /// verification completeness roadmap.
    #[must_use]
    pub fn is_tight(&self) -> bool {
        self.ibp_width < 100.0
    }
}

/// Verify a pre-traced `ComputationGraph` by translating it to a
/// NY `GraphNetwork` and propagating IBP bounds.
///
/// # Auto-detection
///
/// Uses `trace_to_graph_model_multi_input` which auto-detects:
/// - **Single input:** bounds shape matches the input tensor shape.
/// - **Multi input:** bounds are a flat 1D tensor `[sum of all input elements]`.
///
/// Weight-only `Input` nodes (consumed only by composite ops like Conv1d/Linear)
/// are automatically excluded from the variable input set during translation.
/// Note: the returned [`VerifyTraceResult::input_count`] counts *all*
/// `TraceOp::Input` nodes in the graph, not just variable ones.
///
/// # Errors
///
/// - [`VerifyError::UnsupportedOp`] if any trace op cannot be translated
/// - [`VerifyError::PropagationFailed`] if IBP propagation fails
/// - [`VerifyError::Structural`] if the graph is empty or topologically invalid
pub fn verify_trace(
    graph: &ComputationGraph,
    input_bounds: &BoundedTensor,
) -> Result<VerifyTraceResult, VerifyError> {
    let node_count = graph.nodes().len();
    let input_count = graph
        .nodes()
        .iter()
        .filter(|n| matches!(n.op(), TraceOp::Input))
        .count();

    let result = trace_to_graph_model_multi_input(graph)?;
    let network = result.graph;
    let ibp_bounds = network.propagate_ibp(input_bounds)?;

    let ibp_width = compute_width(&ibp_bounds);

    Ok(VerifyTraceResult {
        network,
        ibp_bounds,
        node_count,
        input_count,
        ibp_width,
    })
}

/// Compute the max element-wise width (upper - lower) of a BoundedTensor.
///
/// Uses NaN-propagating fold: if any bound element is NaN, the result is NaN.
/// Without this, `f32::max` drops NaN operands and a vacuous proof with NaN
/// bounds would silently report `ibp_width: 0.0`. See #3196.
fn compute_width(bounds: &BoundedTensor) -> f32 {
    let (lo, hi) = bounds.lower_upper();
    lo.iter()
        .zip(hi.iter())
        .map(|(l, h)| h - l)
        .fold(0.0f32, |acc, w| {
            if w.is_nan() || acc.is_nan() {
                f32::NAN
            } else {
                acc.max(w)
            }
        })
}

#[cfg(test)]
mod tests {
    use super::*;
    use ny_api::BoundedTensor;
    use nn_core::dyn_tensor::trace::{record_input, trace_graph};
    use nn_core::dyn_tensor::DynTensor;
    use nn_core::layers::{Linear, Module};
    use nn_core::Device;
    use ndarray::{ArrayD, IxDyn};

    #[test]
    fn test_verify_trace_linear_relu() {
        // Build a simple Linear(2→2) + ReLU model, trace it, then verify.
        let weight = DynTensor::from_vec(vec![1.0, 0.0, 0.0, 1.0], &[2, 2], &Device::Cpu).unwrap();
        let linear = Linear::new(weight, None).unwrap();
        let input = DynTensor::from_vec(vec![0.5, -0.5], &[1, 2], &Device::Cpu).unwrap();

        // Trace the forward pass.
        let (_output, graph) = trace_graph(|| {
            let mut traced = input.clone();
            if let Some(id) = record_input(input.dims(), input.dtype()) {
                traced.set_trace_id(id);
            }
            let h = linear.forward(&traced)?;
            h.relu()
        })
        .unwrap();

        // Construct input bounds: each element in [-1, 1].
        let lower = ArrayD::from_elem(IxDyn(&[1, 2]), -1.0f32);
        let upper = ArrayD::from_elem(IxDyn(&[1, 2]), 1.0f32);
        let input_bounds = BoundedTensor::new(lower, upper).unwrap();

        // Verify the traced graph.
        let result = verify_trace(&graph, &input_bounds).unwrap();

        // Diagnostics.
        assert!(result.node_count > 0);
        assert_eq!(result.input_count, 1);
        assert!(result.ibp_width.is_finite());
        // Identity matrix + ReLU with input [-1, 1]: per-element output is [0, 1],
        // so max width should be 1.0. IBP is exact for affine+ReLU.
        assert!(
            (result.ibp_width - 1.0).abs() < 0.1,
            "expected ibp_width ≈ 1.0 for identity+ReLU, got {}",
            result.ibp_width
        );

        // ReLU output with identity matrix and input [-1, 1]:
        // lower bound >= 0 (ReLU clamps negatives), upper bound <= 1.0.
        let (lo, hi) = result.ibp_bounds.lower_upper();
        for &l in lo.iter() {
            assert!(l >= -1e-6, "ReLU lower bound should be >= 0, got {l}");
        }
        for &h in hi.iter() {
            assert!(
                h <= 1.0 + 1e-6,
                "identity+ReLU upper bound should be <= 1.0, got {h}"
            );
        }

        assert!(result.is_tight());
    }

    #[test]
    fn test_verify_trace_identity() {
        // Identity model: output = input. Width should match input width.
        let input = DynTensor::from_vec(vec![1.0f32], &[1], &Device::Cpu).unwrap();

        let (_output, graph) = trace_graph(|| {
            let mut traced = input.clone();
            if let Some(id) = record_input(input.dims(), input.dtype()) {
                traced.set_trace_id(id);
            }
            Ok(traced)
        })
        .unwrap();

        let lower = ArrayD::from_elem(IxDyn(&[1]), 0.0f32);
        let upper = ArrayD::from_elem(IxDyn(&[1]), 2.0f32);
        let bounds = BoundedTensor::new(lower, upper).unwrap();

        let result = verify_trace(&graph, &bounds).unwrap();
        // Identity: output width should equal input width (2.0).
        assert!((result.ibp_width - 2.0).abs() < 1e-6);
        assert_eq!(result.input_count, 1);
    }

    #[test]
    fn test_verify_trace_empty_graph_errors() {
        let graph = ComputationGraph::from_nodes(vec![]);
        let bounds = BoundedTensor::new(
            ArrayD::from_elem(IxDyn(&[1]), 0.0f32),
            ArrayD::from_elem(IxDyn(&[1]), 1.0f32),
        )
        .unwrap();

        let err = verify_trace(&graph, &bounds).unwrap_err();
        // Empty graph should produce a structural or empty-graph error, not
        // silently succeed or produce a propagation error.
        let msg = err.to_string();
        assert!(
            msg.contains("no nodes")
                || msg.contains("empty")
                || msg.contains("structural")
                || msg.contains("no output")
                || msg.contains("no variable"),
            "expected structural/empty-graph error for empty graph, got: {msg}"
        );
    }

    #[test]
    fn test_compute_width() {
        let lower = ArrayD::from_elem(IxDyn(&[3]), -1.0f32);
        let upper = ArrayD::from_shape_vec(IxDyn(&[3]), vec![1.0, 3.0, 0.5]).unwrap();
        let bounds = BoundedTensor::new(lower, upper).unwrap();
        let width = compute_width(&bounds);
        // max width: 3.0 - (-1.0) = 4.0
        assert!((width - 4.0).abs() < 1e-6);
    }

    /// Build a `VerifyTraceResult` with an overridden `ibp_width` for threshold tests.
    fn result_with_width(width: f32) -> VerifyTraceResult {
        let input = DynTensor::from_vec(vec![1.0f32], &[1], &Device::Cpu).unwrap();
        let (_output, graph) = trace_graph(|| {
            let mut traced = input.clone();
            if let Some(id) = record_input(input.dims(), input.dtype()) {
                traced.set_trace_id(id);
            }
            Ok(traced)
        })
        .unwrap();
        let bounds = BoundedTensor::new(
            ArrayD::from_elem(IxDyn(&[1]), 0.0f32),
            ArrayD::from_elem(IxDyn(&[1]), 1.0f32),
        )
        .unwrap();
        let mut r = verify_trace(&graph, &bounds).unwrap();
        r.ibp_width = width;
        r
    }

    #[test]
    fn test_is_tight_below_threshold() {
        let r = result_with_width(99.9);
        assert!(r.is_tight(), "width 99.9 should be tight (< 100.0)");
    }

    #[test]
    fn test_is_tight_at_threshold() {
        let r = result_with_width(100.0);
        assert!(!r.is_tight(), "width 100.0 should NOT be tight (strict <)");
    }

    #[test]
    fn test_is_tight_above_threshold() {
        let r = result_with_width(100.1);
        assert!(!r.is_tight(), "width 100.1 should NOT be tight");
    }
}

#[cfg(test)]
#[path = "verify_trace_tests.rs"]
mod tests_nan;

// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! One-call bounds propagation for DynTensor models.
//!
//! Wraps the 4-step verification pipeline (trace → translate → build → propagate)
//! into a single function call:
//!
//! ```rust,ignore
//! use nn_verify::{quick_bounds, BoundedTensor};
//!
//! let input_bounds = BoundedTensor::from_epsilon(&input_values, 0.01)?;
//! let (output, output_bounds) = quick_bounds(
//!     &input,
//!     &input_bounds,
//!     |traced_input| model.forward(traced_input),
//! )?;
//! ```

use ny_api::BoundedTensor;
use nn_core::dyn_tensor::trace::{record_input, trace_graph};
use nn_core::dyn_tensor::DynTensor;

use crate::error::VerifyError;
use crate::trace_to_graph::trace_to_graph_model;

/// Trace a model forward pass and propagate IBP bounds through the captured graph.
///
/// Handles trace ID setup for the input tensor automatically — the closure
/// receives a traced clone of `input` that is registered as the graph input.
///
/// This is a convenience wrapper that combines:
/// 1. `trace_graph(f)` — capture the computation graph
/// 2. `trace_to_graph_model(&graph)` — translate to NY GraphNetwork
/// 3. `network.propagate_ibp(input_bounds)` — propagate interval bounds
///
/// # Arguments
///
/// - `input`: the model's input tensor (used for shape/dtype and as concrete value)
/// - `input_bounds`: IBP bounds for the input (must match `input` shape)
/// - `f`: model forward pass, receives the traced input tensor
///
/// # Errors
///
/// - `VerifyError::UnsupportedOp` if any op in the model can't be translated
/// - `VerifyError::Ny` if bounds propagation fails
/// - Propagates any error from the model forward pass
pub fn quick_bounds<F, T>(
    input: &DynTensor,
    input_bounds: &BoundedTensor,
    f: F,
) -> Result<(T, BoundedTensor), VerifyError>
where
    F: FnOnce(&DynTensor) -> nn_core::Result<T>,
{
    let (output, graph) = trace_graph(|| {
        let mut traced = input.clone();
        if let Some(id) = record_input(input.dims(), input.dtype()) {
            traced.set_trace_id(id);
        }
        f(&traced)
    })
    .map_err(|e| VerifyError::PropagationFailed(format!("trace_graph failed: {e}")))?;

    let network = trace_to_graph_model(&graph)?.graph;
    let output_bounds = network.propagate_ibp(input_bounds)?;
    Ok((output, output_bounds))
}

/// Like [`quick_bounds`] but for models with multiple independent inputs.
///
/// Each input tensor is registered as a separate trace input. The caller must
/// provide IBP bounds as a flat 1D tensor of shape `[sum of all input elements]`.
pub fn quick_bounds_multi_input<F, T>(
    inputs: &[&DynTensor],
    input_bounds: &BoundedTensor,
    f: F,
) -> Result<(T, BoundedTensor), VerifyError>
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

    let network = crate::trace_to_graph::trace_to_graph_model_multi_input(&graph)?.graph;
    let output_bounds = network.propagate_ibp(input_bounds)?;
    Ok((output, output_bounds))
}

#[cfg(test)]
mod tests {
    use super::*;
    use ny_api::BoundedTensor;
    use nn_core::dyn_tensor::DynTensor;
    use nn_core::layers::{Linear, Module};
    use nn_core::Device;
    use ndarray::{ArrayD, IxDyn};

    #[test]
    fn test_quick_bounds_linear_relu() {
        // Simple Linear(2→2) + ReLU model
        let weight =
            DynTensor::from_vec(vec![1.0, 0.0, 0.0, 1.0], &[2, 2], &Device::Cpu).expect("weight");
        let linear = Linear::new(weight, None).expect("linear");
        let input = DynTensor::from_vec(vec![0.5, -0.5], &[1, 2], &Device::Cpu).expect("input");

        // Input bounds: each element in [-1, 1]
        let lower = ArrayD::from_elem(IxDyn(&[1, 2]), -1.0f32);
        let upper = ArrayD::from_elem(IxDyn(&[1, 2]), 1.0f32);
        let input_bounds = BoundedTensor::new(lower, upper).expect("bounds");

        let (output, output_bounds) = quick_bounds(&input, &input_bounds, |traced_input| {
            let h = linear.forward(traced_input)?;
            h.relu()
        })
        .expect("quick_bounds");

        // Output should be concrete value
        let out_data = output.to_flat_vec::<f32>().expect("to_vec");
        assert_eq!(out_data.len(), 2);

        // Output bounds should be valid (lower <= upper)
        let (lo, hi) = output_bounds.lower_upper();
        for (l, u) in lo.iter().zip(hi.iter()) {
            assert!(l <= u, "lower ({l}) should be <= upper ({u})");
        }

        // ReLU clips negative values, so lower bound should be >= 0
        for &l in lo.iter() {
            assert!(l >= 0.0, "ReLU output lower bound should be >= 0, got {l}");
        }
    }
}

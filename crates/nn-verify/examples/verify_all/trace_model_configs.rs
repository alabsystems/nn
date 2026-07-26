// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Trace-based model verification configurations for `verify_all`.
//!
//! These configs use `trace_graph()` + `trace_to_graph_model()` to translate
//! DynTensor imperative forward passes into NY `GraphNetwork`s for
//! formal verification. This validates the trace-to-graph pipeline in
//! production (#2074 AC0).
//!
//! Unlike `model_configs.rs` (which manually builds `TensorKernelDef` via
//! `TensorBlockBuilder`), these configs exercise the automated DynTensor
//! tracing path that real model consumers use.

use ny_propagate::GraphNetwork;
use nn_core::dyn_tensor::trace::{record_input, trace_graph};
use nn_core::dyn_tensor::DynTensor;
use nn_core::layers::{Conv1d, Conv1dConfig, Linear, Module};
use nn_core::{DType, Device};
use nn_verify::BoundedTensor;
use ndarray::{ArrayD, IxDyn};

/// A trace-based model verification configuration.
///
/// Holds a pre-built `GraphNetwork` (produced by `trace_to_graph_model()`)
/// and the input bounds for verification.
pub(crate) struct TraceModelConfig {
    pub name: &'static str,
    /// Pre-built NY graph from DynTensor trace.
    pub graph: GraphNetwork,
    /// Input bounds for the traced variable.
    pub input_bounds: BoundedTensor,
    /// Scalar lower bound (for certificate metadata).
    pub input_lower: f32,
    /// Scalar upper bound (for certificate metadata).
    pub input_upper: f32,
}

fn cpu() -> Device {
    Device::Cpu
}

/// Build a traced Conv1d → ReLU → Linear → Sigmoid model.
///
/// This is a simplified Silero-VAD-like architecture exercising the key
/// trace-to-graph translation paths: convolution, activation, linear, and
/// sigmoid. Uses tiny dimensions for NY tractability.
fn build_traced_conv_relu_linear_sigmoid() -> Result<(GraphNetwork, BoundedTensor), String> {
    let in_channels = 2;
    let out_channels = 4;
    let kernel_size = 3;
    let time_steps = 8;
    let hidden_dim = out_channels * (time_steps - kernel_size + 1); // 4 * 6 = 24
    let output_dim = 1;

    // Build nn layers with small synthetic weights.
    let conv_weight = DynTensor::new(
        &vec![0.01_f32; out_channels * in_channels * kernel_size],
        &[out_channels, in_channels, kernel_size],
        &cpu(),
    )
    .map_err(|e| format!("conv weight: {e}"))?;
    let conv_bias = DynTensor::new(&vec![0.0_f32; out_channels], &[out_channels], &cpu())
        .map_err(|e| format!("conv bias: {e}"))?;
    let conv = Conv1d::new(conv_weight, Some(conv_bias), Conv1dConfig::default())
        .map_err(|e| format!("conv1d: {e}"))?;

    let linear_weight = DynTensor::new(
        &vec![0.01_f32; output_dim * hidden_dim],
        &[output_dim, hidden_dim],
        &cpu(),
    )
    .map_err(|e| format!("linear weight: {e}"))?;
    let linear_bias = DynTensor::new(&vec![0.0_f32; output_dim], &[output_dim], &cpu())
        .map_err(|e| format!("linear bias: {e}"))?;
    let linear =
        Linear::new(linear_weight, Some(linear_bias)).map_err(|e| format!("linear: {e}"))?;

    // Trace the forward pass.
    let x = DynTensor::new(
        &vec![1.0_f32; in_channels * time_steps],
        &[1, in_channels, time_steps],
        &cpu(),
    )
    .map_err(|e| format!("input: {e}"))?;

    let (_result, graph) = trace_graph(|| {
        let mut x = x.clone();
        let id = record_input(&[1, in_channels, time_steps], DType::F32)
            .ok_or_else(|| nn_core::TensorError::InvalidShape("trace not active".into()))?;
        x.set_trace_id(id);
        let h = conv.forward(&x)?;
        let h = h.relu()?;
        let h = h.reshape([1, hidden_dim])?;
        let h = linear.forward(&h)?;
        let y = h.sigmoid()?;
        Ok(y)
    })
    .map_err(|e| format!("trace: {e}"))?;

    // Translate to GraphNetwork.
    let gn = nn_verify::trace_to_graph_model(&graph)
        .map_err(|e| format!("trace_to_graph_model: {e}"))?
        .graph;

    // Input bounds: [1, in_channels, time_steps] in [-1, 1].
    let lower = ArrayD::from_elem(IxDyn(&[1, in_channels, time_steps]), -1.0_f32);
    let upper = ArrayD::from_elem(IxDyn(&[1, in_channels, time_steps]), 1.0_f32);
    let bounds = BoundedTensor::new(lower, upper).map_err(|e| format!("bounds: {e}"))?;

    Ok((gn, bounds))
}

/// Build all trace-based model verification configurations.
///
/// Each config traces a DynTensor forward pass and translates it to a
/// NY `GraphNetwork` via `trace_to_graph_model()`.
pub(crate) fn build_trace_model_configs() -> Vec<TraceModelConfig> {
    let mut configs = Vec::new();

    match build_traced_conv_relu_linear_sigmoid() {
        Ok((graph, bounds)) => {
            configs.push(TraceModelConfig {
                name: "trace_conv_relu_linear_sigmoid",
                graph,
                input_bounds: bounds,
                input_lower: -1.0,
                input_upper: 1.0,
            });
        }
        Err(e) => {
            eprintln!("trace_conv_relu_linear_sigmoid  BUILD_ERR {e}");
        }
    }

    configs
}

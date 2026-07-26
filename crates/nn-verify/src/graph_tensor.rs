// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Tensor-level IR → NY GraphNetwork translation.
//!
//! Maps each `TensorOpKind` variant to a NY layer via a submodule
//! in `graph_tensor_*.rs`. See `designs/2026-02-26-kernelir-tensor-ops.md`.

use std::collections::HashMap;

use ny_propagate::layers::AddConstantLayer;
use ny_propagate::{GraphNetwork, GraphNode, Layer, NETWORK_INPUT};
use nn_dsl::tensor_ir::{TensorKernelDef, TensorOpKind};

use crate::error::{StructuralError, VerifyError};
use crate::graph::{scalar_array, FiniteF32};
use crate::util::get_value;

#[path = "graph_tensor_exp.rs"]
mod exp;
#[path = "graph_tensor_relu.rs"]
mod relu;
#[path = "graph_tensor_sigmoid.rs"]
mod sigmoid;
#[path = "graph_tensor_silu.rs"]
mod silu;
#[path = "graph_tensor_softplus.rs"]
mod softplus;
#[path = "graph_tensor_tanh.rs"]
mod tanh;

/// Convert `usize` to `i32` for NY axis parameters.
/// Returns `StructuralError::ShapeConstraint` if value exceeds `i32::MAX`.
pub(crate) fn axis_as_i32(val: usize, context: &str) -> Result<i32, VerifyError> {
    i32::try_from(val).map_err(|_| {
        StructuralError::ShapeConstraint {
            context: format!("{context}: axis value {val} exceeds i32::MAX"),
        }
        .into()
    })
}

/// Convert `usize` to `i64` for NY shape/axis parameters.
/// Returns `StructuralError::ShapeConstraint` if value exceeds `i64::MAX`.
pub(crate) fn dim_as_i64(val: usize, context: &str) -> Result<i64, VerifyError> {
    i64::try_from(val).map_err(|_| {
        StructuralError::ShapeConstraint {
            context: format!("{context}: dimension value {val} exceeds i64::MAX"),
        }
        .into()
    })
}

/// Convert `i64` shape dimension to `usize` with non-negativity validation.
///
/// Rejects negative values that would wrap to huge `usize` values, producing
/// incorrect graph topologies or enormous buffer allocations. This is the
/// inverse of [`dim_as_i64`].
///
/// Runtime consumers (the legacy in-crate trace translator) were deleted at
/// the ny-trace-bridge cutover; only the Kani proofs remain.
#[cfg(kani)]
pub(crate) fn checked_i64_to_usize(val: i64, context: &str) -> Result<usize, VerifyError> {
    usize::try_from(val).map_err(|_| {
        StructuralError::ShapeConstraint {
            context: format!("{context}: i64 value {val} is negative (cannot convert to usize)"),
        }
        .into()
    })
}

/// Convert `f64` scale/threshold to `usize` with finiteness and non-negativity checks.
///
/// Rejects NaN (Rust saturates to 0), negative values, and non-integral values
/// that would silently truncate. Used for scale factors and threshold counts.
///
/// Runtime consumers (the legacy in-crate trace translator) were deleted at
/// the ny-trace-bridge cutover; only the Kani proofs remain.
#[cfg(kani)]
pub(crate) fn checked_f64_to_usize(val: f64, context: &str) -> Result<usize, VerifyError> {
    if !val.is_finite() {
        return Err(StructuralError::ShapeConstraint {
            context: format!("{context}: f64 value {val} is non-finite"),
        }
        .into());
    }
    if val < 0.0 {
        return Err(StructuralError::ShapeConstraint {
            context: format!("{context}: f64 value {val} is negative"),
        }
        .into());
    }
    let rounded = val.round();
    if (rounded - val).abs() > 1e-6 {
        return Err(StructuralError::ShapeConstraint {
            context: format!("{context}: f64 value {val} is not integral"),
        }
        .into());
    }
    // Safe: rounded is finite, non-negative, and integral. For practical tensor
    // shapes (< 2^53), f64 → usize cast is exact. Values > 2^53 would saturate.
    Ok(rounded as usize)
}

/// How a tensor input is treated during tensor-level verification.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub enum TensorParamBinding {
    /// Variable tensor — bounds are provided via BoundedTensor.
    Variable,
    /// Fixed constant scalar (broadcast to tensor shape at verification time).
    ConstantScalar(f32),
    /// Fixed weight/bias tensor (e.g., Conv1d kernel). Not verified as a variable;
    /// treated as a constant parameter during bound propagation.
    ConstantTensor(ndarray::ArrayD<f32>),
}

/// Value of a tensor IR node during translation: NY node reference,
/// known constant scalar, or constant tensor (for Conv1d weights).
#[derive(Clone, Debug)]
pub(crate) enum TensorNodeValue {
    Variable(String),
    Constant(FiniteF32),
    /// A fixed weight tensor (e.g., Conv1d kernel/bias). Not added to the graph
    /// as a node — consumed directly by tensor ops like Conv1d during translation.
    WeightTensor(ndarray::ArrayD<f32>),
}

/// Tensor-equivalent of [`checked_constant`](crate::graph_translate::checked_constant).
///
/// Validates finiteness and wraps in `TensorNodeValue::Constant`.
fn checked_tensor_constant(value: f32, context: &str) -> Result<TensorNodeValue, VerifyError> {
    let finite = FiniteF32::new(value).map_err(|_| VerifyError::NonFiniteConstant {
        value,
        context: context.to_string(),
    })?;
    Ok(TensorNodeValue::Constant(finite))
}

/// Translate a `TensorKernelDef` into a NY `GraphNetwork`.
///
/// Uses default `NormBoundsMode::ForwardMode` for normalization layers.
/// For strict soundness, use [`tensor_kernel_to_graph_with_norm_mode`]
/// with [`NormBoundsMode::Conservative`].
#[must_use = "graph should be used for verification"]
pub fn tensor_kernel_to_graph(
    kernel: &TensorKernelDef,
    input_bindings: &[TensorParamBinding],
) -> Result<GraphNetwork, VerifyError> {
    tensor_kernel_to_graph_with_norm_mode(
        kernel,
        input_bindings,
        crate::verify_types::NormBoundsMode::ForwardMode,
    )
}

/// Translate a `TensorKernelDef` into a NY `GraphNetwork` using
/// named input bindings.
///
/// Instead of a positional `&[TensorParamBinding]` that must match
/// `add_input()` order exactly, accepts a `HashMap<&str, TensorParamBinding>`
/// keyed by the input name passed to `TensorBlockBuilder::add_input()`.
///
/// Inputs not present in the map default to [`TensorParamBinding::Variable`].
#[must_use = "graph should be used for verification"]
pub fn model_to_graph_network(
    kernel: &TensorKernelDef,
    named_bindings: &HashMap<&str, TensorParamBinding>,
) -> Result<GraphNetwork, VerifyError> {
    model_to_graph_network_with_norm_mode(
        kernel,
        named_bindings,
        crate::verify_types::NormBoundsMode::ForwardMode,
    )
}

/// Translate a `TensorKernelDef` into a NY `GraphNetwork` using
/// named input bindings with configurable normalization bounds mode.
///
/// See [`model_to_graph_network`] for named-binding semantics.
/// See [`NormBoundsMode`] for normalization options.
#[must_use = "graph should be used for verification"]
pub fn model_to_graph_network_with_norm_mode(
    kernel: &TensorKernelDef,
    named_bindings: &HashMap<&str, TensorParamBinding>,
    norm_mode: crate::verify_types::NormBoundsMode,
) -> Result<GraphNetwork, VerifyError> {
    let positional = resolve_named_bindings(kernel, named_bindings)?;
    tensor_kernel_to_graph_with_norm_mode(kernel, &positional, norm_mode)
}

/// Resolve a named-binding map into a positional `Vec<TensorParamBinding>`
/// matching `TensorKernelDef` input order.
///
/// Inputs not present in the map default to `TensorParamBinding::Variable`.
fn resolve_named_bindings(
    kernel: &TensorKernelDef,
    named: &HashMap<&str, TensorParamBinding>,
) -> Result<Vec<TensorParamBinding>, VerifyError> {
    let mut positional = Vec::new();
    for node in &kernel.nodes {
        if let TensorOpKind::Input { name, .. } = &node.kind {
            let binding = named
                .get(name.as_str())
                .cloned()
                .unwrap_or(TensorParamBinding::Variable);
            positional.push(binding);
        }
    }
    Ok(positional)
}

/// Translate a `TensorKernelDef` into a NY `GraphNetwork` with
/// configurable normalization layer bounds mode.
///
/// `norm_mode` controls `forward_mode` and `crown_mode` on InstanceNorm,
/// RmsNorm, LayerNorm, and AdaIN layers. See [`NormBoundsMode`] for options.
///
/// Use [`NormBoundsMode::ForwardMode`] for dramatically tighter bounds through
/// normalization layers (~50x width vs ~1e10x with `Conservative`). See #744.
#[must_use = "graph should be used for verification"]
pub fn tensor_kernel_to_graph_with_norm_mode(
    kernel: &TensorKernelDef,
    input_bindings: &[TensorParamBinding],
    norm_mode: crate::verify_types::NormBoundsMode,
) -> Result<GraphNetwork, VerifyError> {
    kernel.validate()?;

    // Try native RoPE path (collapses 10-node rope_rotate → single RopeLayer).
    if let Some(graph) = try_native_rope(kernel, input_bindings)? {
        return Ok(graph);
    }

    let input_count = kernel
        .nodes
        .iter()
        .filter(|n| matches!(n.kind, TensorOpKind::Input { .. }))
        .count();
    if input_bindings.len() != input_count {
        return Err(VerifyError::ParamCountMismatch {
            ir_count: input_count,
            provided: input_bindings.len(),
        });
    }

    let mut graph = GraphNetwork::new();
    let variable_shapes = variable_input_shapes(input_bindings, &kernel.nodes)?;
    let (_num_variables, input_node_names) =
        setup_multi_variable_inputs(input_bindings, &variable_shapes, &mut graph)?;
    // Each variable enters its subgraph at its TRUE declared rank (via the flat
    // Slice+Reshape split in `setup_multi_variable_inputs`), so ops use NATURAL
    // axes with no stacking-dimension offset. Always 0 (#358 multi-variable fix).
    let axis_offset: usize = 0;

    let mut node_values: Vec<TensorNodeValue> = Vec::with_capacity(kernel.nodes.len());
    let mut input_idx: usize = 0;
    let tensor_ctx = TensorTranslationContext {
        input_bindings,
        input_node_names: &input_node_names,
        axis_offset,
        all_nodes: &kernel.nodes,
        norm_mode,
    };

    for node in &kernel.nodes {
        let value = translate_tensor_node(
            &tensor_ctx,
            node.id,
            &node.kind,
            &node_values,
            &mut input_idx,
            &mut graph,
        )?;
        node_values.push(value);
    }

    // Set output. Wrap bare NETWORK_INPUT in identity layer (see #477, #727).
    match get_value(&node_values, kernel.output.index(), "tensor output")? {
        TensorNodeValue::Variable(name) if name == NETWORK_INPUT => {
            let identity_name = format!("t{}_identity", kernel.output.index());
            graph.add_node(GraphNode::from_input(
                identity_name.clone(),
                Layer::AddConstant(AddConstantLayer::new(scalar_array(0.0)?)),
            ));
            graph.set_output(identity_name);
        }
        TensorNodeValue::Variable(name) => graph.set_output(name.clone()),
        TensorNodeValue::Constant(val) => {
            let name = "t_out".to_string();
            graph.add_node(GraphNode::from_input(
                name.clone(),
                Layer::AddConstant(AddConstantLayer::new(scalar_array(val.get())?)),
            ));
            graph.set_output(name);
        }
        TensorNodeValue::WeightTensor(_) => {
            return Err(VerifyError::UnsupportedOp(
                "weight tensor cannot be used as graph output".into(),
            ));
        }
    }

    Ok(graph)
}

/// Immutable context shared across all tensor node translations.
pub(crate) struct TensorTranslationContext<'a> {
    pub(crate) input_bindings: &'a [TensorParamBinding],
    pub(crate) input_node_names: &'a [Option<String>],
    pub(crate) axis_offset: usize,
    pub(crate) all_nodes: &'a [nn_dsl::tensor_ir::TensorNode],
    /// Controls `forward_mode` and `crown_mode` on normalization layers.
    pub(crate) norm_mode: crate::verify_types::NormBoundsMode,
}

#[path = "graph_tensor_dispatch.rs"]
mod dispatch;
use dispatch::translate_tensor_node;

// Per-op translation submodules — accessed by graph_tensor_dispatch.rs via super::.
#[path = "graph_tensor_binary.rs"]
mod binary;
#[path = "graph_tensor_elementwise.rs"]
mod elementwise;
#[path = "graph_tensor_group_norm_fusion.rs"]
mod group_norm_fusion;
#[path = "graph_tensor_reduce.rs"]
mod reduce;
use reduce::{setup_multi_variable_inputs, variable_input_shapes};
#[path = "graph_tensor_conv1d.rs"]
mod conv1d;
#[path = "graph_tensor_conv2d.rs"]
mod conv2d;
#[path = "graph_tensor_conv_transpose_1d.rs"]
mod conv_transpose_1d;
#[path = "graph_tensor_instance_norm.rs"]
mod instance_norm;
#[path = "graph_tensor_norm_util.rs"]
mod norm_util;
#[path = "graph_tensor_rope.rs"]
mod rope;
#[path = "graph_tensor_structural.rs"]
mod structural;
use rope::try_native_rope;

#[path = "graph_tensor_compose.rs"]
mod compose;
pub use compose::{chain_graphs, tensor_kernels_to_grouped_graph};

#[path = "graph_tensor_adain.rs"]
mod adain;
#[path = "graph_tensor_attention.rs"]
mod attention;
#[path = "graph_tensor_attention_defuse.rs"]
mod attention_defuse;
#[path = "graph_tensor_batch_norm.rs"]
mod batch_norm;
#[path = "graph_tensor_embedding.rs"]
mod embedding;
#[path = "graph_tensor_gated_delta_net.rs"]
mod gated_delta_net;
#[path = "graph_tensor_gelu.rs"]
mod gelu;
#[path = "graph_tensor_layer_norm.rs"]
mod layer_norm;
#[path = "graph_tensor_leaky_relu.rs"]
mod leaky_relu;
#[path = "graph_tensor_linear.rs"]
mod linear;
#[path = "graph_tensor_lstm.rs"]
mod lstm;
#[path = "graph_tensor_matmul.rs"]
mod matmul;
#[path = "graph_tensor_rms_norm.rs"]
mod rms_norm;
#[path = "graph_tensor_softmax.rs"]
mod softmax;
#[path = "graph_tensor_transpose.rs"]
mod transpose;
#[path = "graph_tensor_zero_pad.rs"]
mod zero_pad;

#[cfg(kani)]
#[path = "graph_tensor_kani.rs"]
mod kani_proofs;

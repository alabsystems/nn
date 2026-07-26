// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Builds an nn `ComputationGraph` from a parsed torch.export graph.
//!
//! Walks the parsed graph topologically, resolves tensor names to `NodeId`s,
//! maps each aten op to a `TraceOp`, and assembles `TraceNode`s.

use std::collections::HashMap;

use nn_core::dyn_tensor::trace::{ComputationGraph, NodeId, TraceNode, TraceOp};
use nn_core::DType;

use crate::error::ImportError;
use crate::op_map::{map_node_to_trace_op, try_expand_node, OpMapContext, ResolvedWeight};
use crate::parse::{ExportedProgram, InputSpec, OutputSpec, TensorMeta};

/// Result of building a computation graph from a torch.export program.
#[derive(Debug)]
#[non_exhaustive]
pub struct ImportedGraph {
    /// The computation graph ready for compilation.
    pub graph: ComputationGraph,
    /// Number of user inputs (runtime tensors, not parameters).
    pub num_user_inputs: usize,
    /// Ordered list of user input tensor names.
    pub user_input_names: Vec<String>,
    /// Ordered list of output tensor names.
    pub output_names: Vec<String>,
}

impl ImportedGraph {
    /// Create a new imported graph.
    pub fn new(
        graph: ComputationGraph,
        num_user_inputs: usize,
        user_input_names: Vec<String>,
        output_names: Vec<String>,
    ) -> Self {
        Self {
            graph,
            num_user_inputs,
            user_input_names,
            output_names,
        }
    }
}

/// Build a `ComputationGraph` from a parsed `ExportedProgram` and weight data.
///
/// `weights` maps the **graph placeholder name** (e.g., `"p_linear_weight"`)
/// to resolved flat f32 data + shape. The caller is responsible for loading
/// weights from safetensors and mapping parameter FQNs to graph names.
pub fn build_graph(
    program: &ExportedProgram,
    weights: &HashMap<String, ResolvedWeight>,
) -> Result<ImportedGraph, ImportError> {
    let graph = &program.graph_module.graph;
    let signature = &program.graph_module.signature;

    // Build the op-mapping context.
    let ctx = OpMapContext {
        tensor_meta: &graph.tensor_values,
        weights,
    };

    // Phase 1: Classify graph inputs (parameters, buffers, user inputs).
    let mut param_names: HashMap<String, String> = HashMap::new(); // graph_name -> fqn
    let mut buffer_names: HashMap<String, String> = HashMap::new();
    let mut user_input_names: Vec<String> = Vec::new();

    for spec in &signature.input_specs {
        match spec {
            InputSpec::Parameter(p) => {
                param_names.insert(
                    p.parameter.arg.name.clone(),
                    p.parameter.parameter_name.clone(),
                );
            }
            InputSpec::Buffer(b) => {
                buffer_names.insert(b.buffer.arg.name.clone(), b.buffer.buffer_name.clone());
            }
            InputSpec::UserInput(u) => {
                if let Some(name) = u.user_input.arg.as_tensor_name() {
                    user_input_names.push(name.to_string());
                }
            }
            _ => {} // token, constant_input, etc. — skip
        }
    }

    // Phase 2: Create input TraceNodes and build the name -> NodeId map.
    let mut nodes: Vec<TraceNode> = Vec::new();
    let mut name_to_id: HashMap<String, NodeId> = HashMap::new();
    let mut next_id: NodeId = 0;

    // Register user inputs as Input nodes.
    for name in &user_input_names {
        let id = next_id;
        next_id += 1;
        let (shape, dtype) = tensor_shape_dtype(name, &graph.tensor_values)?;
        nodes.push(TraceNode::new(
            id,
            name.clone(),
            TraceOp::Input,
            vec![],
            shape,
            dtype,
        ));
        name_to_id.insert(name.clone(), id);
    }

    // Parameters and buffers are NOT input nodes — they are embedded in TraceOp
    // variants (Linear { weight: WeightRef }, etc.) via the op mapper. We still
    // register them in name_to_id so they can be resolved when aten ops reference
    // them as tensor inputs. We use a Constant(0.0) placeholder that won't be
    // in the final output dependency chain.
    for name in param_names.keys().chain(buffer_names.keys()) {
        let id = next_id;
        next_id += 1;
        let (shape, dtype) =
            tensor_shape_dtype(name, &graph.tensor_values).unwrap_or_else(|_| (vec![], DType::F32));
        nodes.push(TraceNode::new(
            id,
            name.clone(),
            TraceOp::Constant { value: 0.0 },
            vec![],
            shape,
            dtype,
        ));
        name_to_id.insert(name.clone(), id);
    }

    // Phase 3: Walk computation nodes and build the graph.
    for node in &graph.nodes {
        // Determine output name (first output tensor).
        let output_name = node
            .outputs
            .first()
            .and_then(|a| a.as_tensor_name())
            .unwrap_or(&node.target);

        let (out_shape, out_dtype) = tensor_shape_dtype(output_name, &graph.tensor_values)
            .unwrap_or_else(|_| (vec![], DType::F32));

        // Handle `getitem` (Python built-in for tuple unpacking, e.g., LSTM output).
        // getitem[0] on an LSTM output → alias for the expanded bilstm output.
        // getitem[1,2] → dummy constant (hidden/cell states, usually unused).
        if node.target.contains("getitem") {
            let source_name = node.inputs.first().and_then(|na| na.arg.as_tensor_name());
            if let Some(source) = source_name {
                if let Some(&source_id) = name_to_id.get(source) {
                    // For index 0, alias to the source node.
                    // For other indices, create a constant placeholder.
                    let index = node
                        .inputs
                        .get(1)
                        .and_then(|na| na.arg.as_int())
                        .unwrap_or(0);
                    if index == 0 {
                        name_to_id.insert(output_name.to_string(), source_id);
                    } else {
                        let id = next_id;
                        next_id += 1;
                        nodes.push(TraceNode::new(
                            id,
                            output_name.to_string(),
                            TraceOp::Constant { value: 0.0 },
                            vec![],
                            out_shape,
                            out_dtype,
                        ));
                        name_to_id.insert(output_name.to_string(), id);
                    }
                    continue;
                }
            }
        }

        // Check for multi-node expansion (e.g., bidirectional LSTM → multiple
        // unidirectional LSTMs + flip + cat). The input shape is needed for
        // computing intermediate shapes in the expansion.
        let input_shape = node
            .inputs
            .first()
            .and_then(|na| na.arg.as_tensor_name())
            .and_then(|n| graph.tensor_values.get(n))
            .and_then(TensorMeta::concrete_shape)
            .unwrap_or_default();

        if let Some(expanded) = try_expand_node(node, &ctx, output_name, &input_shape)? {
            for en in expanded {
                let input_ids: Vec<NodeId> = en
                    .input_names
                    .iter()
                    .map(|name| {
                        name_to_id.get(name.as_str()).copied().ok_or_else(|| {
                            ImportError::TopologyError {
                                node_name: en.name.clone(),
                                ref_name: name.clone(),
                            }
                        })
                    })
                    .collect::<Result<_, _>>()?;
                let id = next_id;
                next_id += 1;
                nodes.push(TraceNode::new(
                    id,
                    en.name.clone(),
                    en.op,
                    input_ids,
                    en.output_shape,
                    en.output_dtype,
                ));
                name_to_id.insert(en.name, id);
            }
            continue;
        }

        // Standard single-op path.
        let (op, input_names) = map_node_to_trace_op(node, &ctx, input_shape.len())?;

        let input_ids: Vec<NodeId> = input_names
            .iter()
            .map(|name| {
                name_to_id
                    .get(name.as_str())
                    .copied()
                    .ok_or_else(|| ImportError::TopologyError {
                        node_name: node.target.clone(),
                        ref_name: name.clone(),
                    })
            })
            .collect::<Result<_, _>>()?;

        let id = next_id;
        next_id += 1;
        nodes.push(TraceNode::new(
            id,
            output_name.to_string(),
            op,
            input_ids,
            out_shape,
            out_dtype,
        ));
        name_to_id.insert(output_name.to_string(), id);
    }

    // Phase 4: Mark output nodes.
    let mut output_names: Vec<String> = Vec::new();
    let mut comp_graph = ComputationGraph::from_nodes(nodes);

    for spec in &signature.output_specs {
        if let OutputSpec::UserOutput(uo) = spec {
            if let Some(name) = uo.user_output.arg.as_tensor_name() {
                output_names.push(name.to_string());
                if let Some(&id) = name_to_id.get(name) {
                    let _ = comp_graph.mark_output(id);
                }
            }
        }
    }

    // Validate topology.
    comp_graph.validate_topology().map_err(|e| match e {
        nn_core::TensorError::TopologyError {
            node_name,
            missing_input,
            ..
        } => ImportError::TopologyError {
            node_name,
            ref_name: format!("node_id_{missing_input}"),
        },
        other => ImportError::Tensor(other),
    })?;

    let num_inputs = user_input_names.len();
    Ok(ImportedGraph::new(
        comp_graph,
        num_inputs,
        user_input_names,
        output_names,
    ))
}

/// Helper: Build the weight map from a safetensors-loaded weight map.
///
/// `param_map` maps graph placeholder names to parameter FQNs.
/// `weight_data` maps parameter FQNs to (data, shape) tuples.
pub fn build_weight_map(
    input_specs: &[InputSpec],
    weight_data: &HashMap<String, (Vec<f32>, Vec<usize>)>,
) -> HashMap<String, ResolvedWeight> {
    let mut result = HashMap::new();
    for spec in input_specs {
        match spec {
            InputSpec::Parameter(p) => {
                if let Some((data, shape)) = weight_data.get(&p.parameter.parameter_name) {
                    result.insert(
                        p.parameter.arg.name.clone(),
                        ResolvedWeight::new(data.clone(), shape.clone()),
                    );
                }
            }
            InputSpec::Buffer(b) => {
                if let Some((data, shape)) = weight_data.get(&b.buffer.buffer_name) {
                    result.insert(
                        b.buffer.arg.name.clone(),
                        ResolvedWeight::new(data.clone(), shape.clone()),
                    );
                }
            }
            _ => {}
        }
    }
    result
}

/// Extract shape and dtype from tensor_values for a given tensor name.
fn tensor_shape_dtype(
    name: &str,
    tensor_values: &HashMap<String, TensorMeta>,
) -> Result<(Vec<usize>, DType), ImportError> {
    let meta = tensor_values
        .get(name)
        .ok_or_else(|| ImportError::UnknownTensor {
            name: name.to_string(),
        })?;
    let shape = meta
        .concrete_shape()
        .ok_or_else(|| ImportError::UnknownTensor {
            name: format!("{name} (has dynamic dimensions)"),
        })?;
    let dtype = meta.to_dtype().unwrap_or(DType::F32);
    Ok((shape, dtype))
}

#[cfg(test)]
#[path = "graph_build_tests.rs"]
mod tests;

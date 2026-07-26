// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Sequential kernel composition into a single NY `GraphNetwork`.
//!
//! Chains two kernel subgraphs so kernel A's output feeds kernel B's input,
//! enabling NY to propagate bounds through the full composition.
//!
//! This is the building block for multi-layer model verification (#534).
//! For fusion *equivalence* proofs (fused vs sequential), see [`super::fusion`].

use ny_propagate::layers::SliceLayer;
use ny_propagate::{GraphNetwork, GraphNode, Layer};
use nn_dsl::ir::KernelDef;

use crate::error::VerifyError;
use crate::fusion::translate_kernel_path;

/// Specification for sequential composition of two kernels: A → B.
///
/// Kernel A produces an output that feeds one of kernel B's parameters.
/// All other parameters for both kernels are provided as constants.
///
/// Use [`SequentialSpec::new()`] to construct with upfront validation.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct SequentialSpec<'a> {
    /// First kernel (output feeds into `second`).
    pub first: &'a KernelDef,
    /// Second kernel (receives `first`'s output as input).
    pub second: &'a KernelDef,
    /// Constant values for `first`'s non-variable parameters.
    /// Entry `i` corresponds to `first.params[i+1]` (param 0 is the variable input).
    pub first_constants: &'a [f32],
    /// Constant values for `second`'s non-variable parameters.
    /// Entries correspond to non-chained parameters of `second`, in order.
    pub second_constants: &'a [f32],
    /// Which parameter index of `second` receives `first`'s output.
    /// Typically 0 (the main input).
    pub chain_param: usize,
}

impl<'a> SequentialSpec<'a> {
    /// Construct a validated `SequentialSpec`.
    ///
    /// Checks parameter count correctness and constant finiteness upfront,
    /// matching the pattern established by [`super::fusion_spec::FusionSpec::new()`].
    ///
    /// # Errors
    ///
    /// Returns [`VerifyError::ParamCountMismatch`] if constant counts don't match
    /// kernel parameter counts, or [`VerifyError::NonFiniteInputMetadata`] if any
    /// constant is NaN or infinite.
    pub fn new(
        first: &'a KernelDef,
        second: &'a KernelDef,
        first_constants: &'a [f32],
        second_constants: &'a [f32],
        chain_param: usize,
    ) -> Result<Self, VerifyError> {
        let spec = Self {
            first,
            second,
            first_constants,
            second_constants,
            chain_param,
        };
        validate_spec(&spec)?;
        Ok(spec)
    }
}

/// Build a `GraphNetwork` that chains kernel A → kernel B.
///
/// The composed graph has a single scalar input that feeds kernel A.
/// Kernel A's output feeds into kernel B at `spec.chain_param`.
/// All other parameters are set to their constant values.
///
/// # Errors
///
/// Returns `VerifyError` if either kernel fails validation or parameter
/// counts don't match the provided constants.
pub fn compose_sequential(spec: &SequentialSpec<'_>) -> Result<GraphNetwork, VerifyError> {
    validate_spec(spec)?;

    let first_expected_constants = spec.first.params.len().saturating_sub(1);
    let mut graph = GraphNetwork::new();

    // Single shared input: SliceLayer extracting element [0,1) from NETWORK_INPUT.
    let input_name = "in_0".to_string();
    graph.add_node(GraphNode::from_input(
        input_name.clone(),
        Layer::Slice(SliceLayer::new(0, 0, 1)),
    ));

    // First kernel (prefix "a_"): param 0 = variable input, rest = constants.
    let mut first_names: Vec<Option<String>> = std::iter::once(Some(input_name))
        .chain(std::iter::repeat_n(None, first_expected_constants))
        .collect();
    for (i, &val) in spec.first_constants.iter().enumerate() {
        let name = add_constant_node(&mut graph, &format!("a_const_{i}"), val)?;
        first_names[i + 1] = Some(name);
    }
    let first_out = translate_kernel_path("a_", spec.first, &first_names, &mut graph)?;

    // Second kernel (prefix "b_"): chain_param receives first's output, rest = constants.
    let second_names = build_second_param_names(spec, &first_out, &mut graph)?;
    let second_out = translate_kernel_path("b_", spec.second, &second_names, &mut graph)?;

    graph.set_output(second_out);
    Ok(graph)
}

/// Validate a `SequentialSpec` for parameter count correctness and constant finiteness.
fn validate_spec(spec: &SequentialSpec<'_>) -> Result<(), VerifyError> {
    spec.first.validate()?;
    spec.second.validate()?;

    let first_expected = spec.first.params.len().saturating_sub(1);
    if spec.first_constants.len() != first_expected {
        return Err(VerifyError::ParamCountMismatch {
            ir_count: spec.first.params.len(),
            provided: spec.first_constants.len() + 1,
        });
    }
    if spec.chain_param >= spec.second.params.len() {
        return Err(VerifyError::ParamCountMismatch {
            ir_count: spec.second.params.len(),
            provided: spec.chain_param + 1,
        });
    }
    let second_expected = spec.second.params.len().saturating_sub(1);
    if spec.second_constants.len() != second_expected {
        return Err(VerifyError::ParamCountMismatch {
            ir_count: spec.second.params.len(),
            provided: spec.second_constants.len() + 1,
        });
    }

    // Validate constant finiteness (defense-in-depth: scalar_array panics on
    // NaN/Inf, but returning Result is better than a runtime panic).
    for (i, &val) in spec.first_constants.iter().enumerate() {
        if !val.is_finite() {
            return Err(VerifyError::NonFiniteInputMetadata {
                context: format!("first_constants[{i}] = {val}"),
            });
        }
    }
    for (i, &val) in spec.second_constants.iter().enumerate() {
        if !val.is_finite() {
            return Err(VerifyError::NonFiniteInputMetadata {
                context: format!("second_constants[{i}] = {val}"),
            });
        }
    }
    Ok(())
}

/// Add a constant-value node to the graph: `MulConstant(0)` → `AddConstant(val)`.
///
/// Returns the name of the output node (the `AddConstant` node).
fn add_constant_node(
    graph: &mut GraphNetwork,
    name: &str,
    val: f32,
) -> Result<String, VerifyError> {
    let zero_name = format!("{name}_zero");
    graph.add_node(GraphNode::from_input(
        zero_name.clone(),
        Layer::MulConstant(ny_propagate::layers::MulConstantLayer::scalar(0.0)),
    ));
    graph.add_node(GraphNode::new(
        name.to_string(),
        Layer::AddConstant(ny_propagate::layers::AddConstantLayer::new(
            crate::graph::scalar_array(val)?,
        )),
        vec![zero_name],
    ));
    Ok(name.to_string())
}

/// Build parameter name mappings for the second kernel in a sequential composition.
///
/// `chain_param` receives the first kernel's output; all other params get constant nodes.
fn build_second_param_names(
    spec: &SequentialSpec<'_>,
    first_out: &str,
    graph: &mut GraphNetwork,
) -> Result<Vec<Option<String>>, VerifyError> {
    let mut names: Vec<Option<String>> = Vec::with_capacity(spec.second.params.len());
    let mut const_idx = 0;
    for i in 0..spec.second.params.len() {
        if i == spec.chain_param {
            names.push(Some(first_out.to_string()));
        } else {
            let val = spec.second_constants[const_idx];
            let name = add_constant_node(graph, &format!("b_const_{const_idx}"), val)?;
            names.push(Some(name));
            const_idx += 1;
        }
    }
    Ok(names)
}

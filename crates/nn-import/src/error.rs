// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Error types for `nn-import`.

/// Errors arising from torch.export graph parsing and model import.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum ImportError {
    /// JSON deserialization failed.
    #[error("failed to parse torch.export JSON: {0}")]
    JsonParse(#[from] serde_json::Error),

    /// Schema version is not supported.
    #[error("unsupported schema version {major}.{minor} (expected major=8)")]
    UnsupportedSchema { major: u64, minor: u64 },

    /// An aten op target string has no mapping to a TraceOp.
    #[error("unsupported aten op: {target}")]
    UnsupportedOp { target: String },

    /// A required argument is missing from an op node.
    #[error("missing argument '{arg_name}' for op '{op_target}'")]
    MissingArgument { op_target: String, arg_name: String },

    /// An argument has the wrong type.
    #[error("argument '{arg_name}' for op '{op_target}' has wrong type: expected {expected}, got {actual}")]
    WrongArgumentType {
        op_target: String,
        arg_name: String,
        expected: &'static str,
        actual: String,
    },

    /// A tensor name referenced by a node was not found in `tensor_values`.
    #[error("tensor '{name}' not found in graph tensor_values")]
    UnknownTensor { name: String },

    /// A weight parameter referenced in `input_specs` has no data.
    #[error("weight '{fqn}' referenced in graph but not found in safetensors")]
    MissingWeight { fqn: String },

    /// Weight data length does not match shape product.
    #[error(
        "weight '{name}' has shape {shape:?} ({expected} elements) but data has {actual} values"
    )]
    WeightShapeMismatch {
        name: String,
        shape: Vec<usize>,
        expected: usize,
        actual: usize,
    },

    /// A negative dimension/size value was provided where only non-negative is valid.
    ///
    /// torch.export normalizes negative dimension indices to positive, so a
    /// negative value here (other than the -1 sentinel in reshape/expand)
    /// indicates a graph format issue or unsupported pattern.
    #[error("negative value {value} for argument '{arg_name}' in op '{op_target}'")]
    NegativeDimension {
        op_target: String,
        arg_name: String,
        value: i64,
    },

    /// Multi-axis reduction/flip not yet supported.
    ///
    /// The TraceOp variants currently take a single `dim: usize`. Multi-axis
    /// reductions require changing TraceOp to `dims: Vec<usize>` (cross-crate).
    #[error("multi-axis {op_kind} on dims {dims:?} for op '{op_target}' not yet supported (single dim only)")]
    MultiAxisNotSupported {
        op_target: String,
        op_kind: &'static str,
        dims: Vec<i64>,
    },

    /// Graph topology error (forward reference detected).
    #[error("topology error: node '{node_name}' references unknown tensor '{ref_name}'")]
    TopologyError { node_name: String, ref_name: String },

    /// File I/O error (distinct from MissingWeight for semantic correctness).
    #[error("I/O error reading '{path}': {detail}")]
    Io { path: String, detail: String },

    /// Safetensors dtype not supported for conversion to f32.
    #[error("unsupported safetensors dtype {dtype} for weight '{name}'")]
    UnsupportedDtype { name: String, dtype: String },

    /// Required weight groups are missing from the safetensors file.
    #[error("missing Kokoro weight groups: {missing_prefixes}")]
    MissingWeightGroups { missing_prefixes: String },

    /// The compiled model backend failed to load the safetensors file.
    #[error("compiled model load failed for '{path}': {detail}")]
    CompiledModelLoad { path: String, detail: String },

    /// Core tensor error from nn-core.
    #[error("tensor error: {0}")]
    Tensor(#[from] nn_core::TensorError),
}

// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Trace node and node identifier types.

use crate::DType;

use super::TraceOp;

/// A unique identifier for a traced tensor.
pub type NodeId = u64;

/// A node in the computation graph captured during tracing.
#[derive(Debug, Clone)]
pub struct TraceNode {
    /// Unique node identifier.
    pub(super) id: NodeId,
    /// Human-readable name (e.g., "linear_0", "relu_1").
    pub(super) name: String,
    /// The operation this node performs.
    pub(super) op: TraceOp,
    /// IDs of input nodes.
    pub(super) inputs: Vec<NodeId>,
    /// Output tensor shape.
    pub(super) output_shape: Vec<usize>,
    /// Output tensor dtype.
    pub(super) output_dtype: DType,
}

impl TraceNode {
    /// Create a new trace node with explicit fields.
    ///
    /// Used by test code and the trace compiler to construct graphs
    /// outside the thread-local recorder.
    pub fn new(
        id: NodeId,
        name: String,
        op: TraceOp,
        inputs: Vec<NodeId>,
        output_shape: Vec<usize>,
        output_dtype: DType,
    ) -> Self {
        Self {
            id,
            name,
            op,
            inputs,
            output_shape,
            output_dtype,
        }
    }

    /// Returns the node's unique ID.
    pub fn id(&self) -> NodeId {
        self.id
    }

    /// Returns the node's name.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Returns the operation.
    pub fn op(&self) -> &TraceOp {
        &self.op
    }

    /// Returns input node IDs.
    pub fn inputs(&self) -> &[NodeId] {
        &self.inputs
    }

    /// Returns the output shape.
    pub fn output_shape(&self) -> &[usize] {
        &self.output_shape
    }

    /// Returns the output dtype.
    pub fn output_dtype(&self) -> DType {
        self.output_dtype
    }
}

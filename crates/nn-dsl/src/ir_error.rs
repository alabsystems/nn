// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Error types for kernel IR construction and validation.

use thiserror::Error;

use super::{NodeId, ValueType};

/// Errors that can occur during IR construction or validation.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum IRError {
    /// A node references a `NodeId` beyond the bounds of the node list.
    #[error("node {0:?} references out-of-bounds node")]
    InvalidNodeRef(NodeId),
    /// A node references itself or a later node (violates topological order).
    #[error(
        "node {0:?} has forward or self reference to {1:?} (must reference strictly earlier nodes)"
    )]
    ForwardRef(NodeId, NodeId),
    /// A `Param` node references a parameter index that does not exist.
    #[error("parameter index {0} out of bounds (have {1} params)")]
    InvalidParamRef(usize, usize),
    /// A `SumReduce` node has an empty input list.
    #[error("sum-reduce node {0:?} must have at least one input")]
    EmptySumReduce(NodeId),
    /// A `Literal` node contains NaN or Inf (not valid in MSL).
    #[error("literal node {0:?} has non-finite value {1} (NaN/Inf not valid in MSL)")]
    NonFiniteLiteral(NodeId, f64),
    /// A node's `id` field does not match its position in the node vector.
    #[error(
        "node at index {expected_index} has id {found:?} (must equal NodeId::new({expected_index}))"
    )]
    MismatchedNodeId {
        found: NodeId,
        expected_index: usize,
    },
    /// A `Select` node's condition operand is not boolean.
    #[error("select node {node:?} requires bool condition, found {found:?}")]
    SelectCondNotBool { node: NodeId, found: ValueType },
    /// A `Select` node's then/else branches have different types.
    #[error("select node {node:?} branch type mismatch: then={then_type:?}, else={else_type:?}")]
    SelectBranchTypeMismatch {
        node: NodeId,
        then_type: ValueType,
        else_type: ValueType,
    },
    /// Binary operation operands have mismatched types.
    #[error("node {node:?} operand type mismatch: {lhs_type:?} vs {rhs_type:?}")]
    OperandTypeMismatch {
        node: NodeId,
        lhs_type: ValueType,
        rhs_type: ValueType,
    },
    /// An arithmetic node received a boolean operand.
    #[error("node {node:?} requires numeric operand at {operand:?}, found {found:?}")]
    NonNumericOperand {
        node: NodeId,
        operand: NodeId,
        found: ValueType,
    },
    /// Output node type does not match the kernel's declared return type.
    #[error("output node type {found:?} does not match declared return type {expected:?}")]
    OutputTypeMismatch {
        found: ValueType,
        expected: ValueType,
    },
    /// Powi exponent exceeds the configured maximum.
    #[error("powi node {node:?} exponent {exp} exceeds maximum |{exp}| > {max} (would generate degenerate MSL)")]
    PowiExponentTooLarge { node: NodeId, exp: i32, max: u32 },

    /// Type conversion is not supported by the kernel subset.
    #[error("unsupported type conversion: {0}")]
    UnsupportedType(String),

    /// Kernel parameter count exceeds Metal's buffer slot limit.
    #[error("Metal buffer limit exceeded: kernel requires {required} buffers but Metal allows at most {max} (buffer indices 0..={max_index})")]
    BufferLimitExceeded {
        required: usize,
        max: usize,
        max_index: usize,
    },

    /// Kernel or parameter name is not a valid identifier.
    #[error("invalid identifier `{name}` ({context}): {reason}")]
    InvalidIdentifier {
        name: String,
        context: &'static str,
        reason: &'static str,
    },
}

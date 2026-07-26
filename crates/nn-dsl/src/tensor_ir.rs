// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Tensor-level Kernel IR for multi-element ops (reductions, norms, attention).
//! See `designs/2026-02-26-kernelir-tensor-ops.md`.

#[path = "tensor_ir_types.rs"]
mod types;
pub use types::{AttentionMask, BroadcastAlignment, Pool2dParams, ReduceOp, TensorNodeId};

#[path = "tensor_ir_ops.rs"]
mod ops;
pub use ops::TensorOpKind;

#[path = "tensor_ir_error_layer.rs"]
mod error_layer;
pub use error_layer::{TensorIRConvError, TensorIRLayerError};

#[path = "tensor_ir_error.rs"]
mod error;
pub use error::TensorIRError;

#[path = "tensor_ir_pretty.rs"]
mod tensor_ir_pretty;

#[path = "tensor_ir_broadcast.rs"]
mod broadcast;

#[path = "tensor_ir_validate.rs"]
mod validate;

pub use broadcast::infer_broadcast_alignment;
use broadcast::validate_broadcast_alignment;
pub use tensor_ir_pretty::tensor_ir_pretty_print;

/// A node in the tensor-level IR graph.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
#[non_exhaustive]
pub struct TensorNode {
    pub id: TensorNodeId,
    pub kind: TensorOpKind,
    /// Output shape of this node.
    pub shape: Vec<usize>,
}

impl TensorNode {
    /// Create a new tensor IR node.
    #[must_use]
    pub fn new(id: TensorNodeId, kind: TensorOpKind, shape: Vec<usize>) -> Self {
        Self { id, kind, shape }
    }
}

/// Complete tensor-level kernel definition.
///
/// Topologically ordered: each node references only earlier nodes.
/// The output node's shape is the kernel output shape.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
#[non_exhaustive]
pub struct TensorKernelDef {
    pub name: String,
    pub nodes: Vec<TensorNode>,
    pub output: TensorNodeId,
}

impl TensorKernelDef {
    /// Create a new tensor-level kernel definition.
    #[must_use]
    pub fn new(name: impl Into<String>, nodes: Vec<TensorNode>, output: TensorNodeId) -> Self {
        Self {
            name: name.into(),
            nodes,
            output,
        }
    }
}

#[path = "tensor_ir_remap.rs"]
mod remap;

#[cfg(test)]
#[path = "tensor_ir_remap_tests.rs"]
mod remap_tests;

#[cfg(test)]
#[path = "tensor_ir_validate_tests.rs"]
mod validate_tests;

#[cfg(kani)]
#[path = "tensor_ir_kani.rs"]
mod kani_proofs;

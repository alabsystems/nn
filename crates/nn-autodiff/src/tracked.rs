// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Tracked tensor for recording computation graphs.
//!
//! [`TrackedTensor`] wraps a [`DynTensor`] with operation tracking for the
//! backward pass. Inference code uses `DynTensor` directly (zero overhead);
//! training code uses `TrackedTensor` to record the computation graph.

use crate::error::Result;
use crate::op::Op;
use crate::var::{Var, VarId};
use nn_core::dyn_tensor::DynTensor;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

/// Unique node identifier for topological sort during backward.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct NodeId(u64);

static NEXT_NODE_ID: AtomicU64 = AtomicU64::new(0);

impl NodeId {
    fn next() -> Self {
        Self(NEXT_NODE_ID.fetch_add(1, Ordering::Relaxed))
    }

    /// Raw ID value (for topo sort visited set).
    #[must_use]
    pub fn as_u64(self) -> u64 {
        self.0
    }
}

/// A tensor that records its computation history for the backward pass.
///
/// `TrackedTensor` = `DynTensor` + `Op` tracking. The computation graph is
/// implicit: each `TrackedTensor` stores an optional `Op` describing how it
/// was produced from other `TrackedTensor`s.
///
/// Leaf nodes are either:
/// - **Variables** (`from_var`): receive gradients during backward
/// - **Constants** (`from_tensor`): no gradients flow through
///
/// Intermediate nodes are created by arithmetic methods that record their `Op`.
pub struct TrackedTensor {
    data: DynTensor,
    op: Option<Op>,
    is_var: bool,
    var_id: Option<VarId>,
    pub(crate) node_id: NodeId,
}

impl TrackedTensor {
    /// Create a tracked tensor from a trainable variable (leaf node).
    ///
    /// Gradients will accumulate for this node during backward.
    pub fn from_var(var: &Var) -> Result<Self> {
        Ok(Self {
            data: var.data()?,
            op: None,
            is_var: true,
            var_id: Some(var.id()),
            node_id: NodeId::next(),
        })
    }

    /// Create a tracked tensor from a plain `DynTensor` (constant leaf).
    ///
    /// No gradients flow through constants. This is used for input data,
    /// frozen weights, and other non-trainable tensors.
    pub fn from_tensor(data: DynTensor) -> Self {
        Self {
            data,
            op: None,
            is_var: false,
            var_id: None,
            node_id: NodeId::next(),
        }
    }

    /// Create an intermediate tracked tensor from a computed result.
    pub(crate) fn from_op(data: DynTensor, op: Op) -> Self {
        Self {
            data,
            op: Some(op),
            is_var: false,
            var_id: None,
            node_id: NodeId::next(),
        }
    }

    /// Get the underlying `DynTensor`.
    #[must_use]
    pub fn tensor(&self) -> &DynTensor {
        &self.data
    }

    /// Unwrap to the underlying `DynTensor`, discarding op tracking.
    ///
    /// Takes ownership of `self.op` first (making our custom `Drop` impl a
    /// no-op), then replaces `self.data` with a zero-element dummy tensor.
    pub fn into_tensor(mut self) -> Result<DynTensor> {
        // Take the op first so our Drop impl's iterative cleanup is trivial.
        let _op = self.op.take();
        // Replace data with a dummy; the dummy will be dropped by our Drop impl
        // (which is a no-op since op is now None).
        Ok(std::mem::replace(
            &mut self.data,
            // Zero-element dummy tensor — minimal allocation.
            DynTensor::zeros(&[], nn_core::DType::F32, &nn_core::Device::Cpu)?,
        ))
    }

    /// Whether this node is a trainable variable.
    #[must_use]
    pub fn is_var(&self) -> bool {
        self.is_var
    }

    /// The `VarId` if this is a variable leaf node.
    #[must_use]
    pub fn var_id(&self) -> Option<VarId> {
        self.var_id
    }

    /// The operation that produced this tensor (None for leaves).
    ///
    /// Used by the backward pass to traverse the computation graph,
    /// and by users to inspect how a tensor was computed.
    #[must_use]
    pub fn op(&self) -> Option<&Op> {
        self.op.as_ref()
    }

    /// Unique node identifier for graph traversal.
    #[must_use]
    pub fn node_id(&self) -> NodeId {
        self.node_id
    }

    /// Sum over a dimension, keeping dimension.
    pub fn sum_keepdim(self: &Arc<Self>, dim: usize) -> Result<Arc<Self>> {
        let data = self.data.sum_keepdim(dim)?;
        Ok(Arc::new(Self::from_op(
            data,
            Op::SumKeepDim(Arc::clone(self), dim),
        )))
    }

    /// Mean over a dimension, keeping dimension.
    pub fn mean_keepdim(self: &Arc<Self>, dim: usize) -> Result<Arc<Self>> {
        let data = self.data.mean_keepdim(dim)?;
        Ok(Arc::new(Self::from_op(
            data,
            Op::MeanKeepDim(Arc::clone(self), dim),
        )))
    }

    /// Reshape to new dimensions. Records original shape for backward.
    pub fn reshape(self: &Arc<Self>, new_dims: &[usize]) -> Result<Arc<Self>> {
        let original_shape = self.data.dims().to_vec();
        let data = self.data.reshape(new_dims)?;
        Ok(Arc::new(Self::from_op(
            data,
            Op::Reshape(Arc::clone(self), original_shape),
        )))
    }

    /// Transpose two dimensions.
    pub fn transpose(self: &Arc<Self>, d1: usize, d2: usize) -> Result<Arc<Self>> {
        let data = self.data.transpose(d1, d2)?;
        Ok(Arc::new(Self::from_op(
            data,
            Op::Transpose(Arc::clone(self), d1, d2),
        )))
    }

    /// Insert a dimension of size 1.
    pub fn unsqueeze(self: &Arc<Self>, dim: usize) -> Result<Arc<Self>> {
        let data = self.data.unsqueeze(dim)?;
        Ok(Arc::new(Self::from_op(
            data,
            Op::Unsqueeze(Arc::clone(self), dim),
        )))
    }

    /// Remove a dimension of size 1.
    pub fn squeeze(self: &Arc<Self>, dim: usize) -> Result<Arc<Self>> {
        let data = self.data.squeeze(dim)?;
        Ok(Arc::new(Self::from_op(
            data,
            Op::Squeeze(Arc::clone(self), dim),
        )))
    }

    /// Broadcast expand to the given shape.
    ///
    /// Records the original shape for backward reduction.
    pub fn broadcast_as(self: &Arc<Self>, shape: &[usize]) -> Result<Arc<Self>> {
        let original_shape = self.data.dims().to_vec();
        let data = self.data.expand(shape)?;
        Ok(Arc::new(Self::from_op(
            data,
            Op::Broadcast(Arc::clone(self), original_shape),
        )))
    }

    /// Narrow (slice) along a dimension.
    ///
    /// Records the original dimension size for backward zero-padding.
    pub fn narrow(self: &Arc<Self>, dim: usize, start: usize, len: usize) -> Result<Arc<Self>> {
        let orig_dim_size = self.data.dims()[dim];
        let data = self.data.narrow(dim, start, len)?;
        Ok(Arc::new(Self::from_op(
            data,
            Op::Narrow(Arc::clone(self), dim, start, orig_dim_size),
        )))
    }

    /// Unfold (sliding window extraction) along a dimension.
    ///
    /// Replaces O(n_frames) narrow() + cat() with a single operation.
    /// For `[..., T, ...]` with `unfold(dim, size, step)`, output shape is
    /// `[..., n_windows, ..., size]` where `n_windows = (T - size) / step + 1`.
    pub fn unfold(self: &Arc<Self>, dim: usize, size: usize, step: usize) -> Result<Arc<Self>> {
        let data = self.data.unfold(dim, size, step)?;
        Ok(Arc::new(Self::from_op(
            data,
            Op::Unfold(Arc::clone(self), dim, size, step),
        )))
    }

    /// Total number of elements.
    #[must_use]
    pub fn numel(&self) -> usize {
        self.data.numel()
    }

    /// Shape of the underlying tensor.
    #[must_use]
    pub fn dims(&self) -> &[usize] {
        self.data.dims()
    }

    /// Detach from the computation graph.
    ///
    /// Returns a new `TrackedTensor` with the same data but no recorded
    /// operation. Gradients will not flow backward through this tensor,
    /// matching PyTorch's `.detach()` semantics.
    ///
    /// Use cases: stop-gradient patterns, target networks, frozen sub-networks.
    pub fn detach(self: &Arc<Self>) -> Arc<Self> {
        Arc::new(Self::from_tensor(self.data.clone()))
    }
}

// Iterative Drop impl to prevent stack overflow on deep computation graphs.
// Extracted to tracked_drop.rs via #[path] submodule.
#[path = "tracked_drop.rs"]
mod drop_impl;

impl std::fmt::Debug for TrackedTensor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TrackedTensor")
            .field("dims", &self.data.dims())
            .field("is_var", &self.is_var)
            .field("has_op", &self.op.is_some())
            .field("node_id", &self.node_id)
            .finish_non_exhaustive()
    }
}

#[path = "tracked_ops.rs"]
mod ops;

#[path = "tracked_composite_ops.rs"]
mod composite_ops;

#[cfg(test)]
#[path = "tracked_tests.rs"]
mod tests;

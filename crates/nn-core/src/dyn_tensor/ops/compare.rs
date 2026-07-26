// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Tensor-vs-tensor comparison convenience methods for [`DynTensor`].
//!
//! These are short-named aliases for the `broadcast_*` comparison methods
//! defined in `selection/compare.rs`. The `broadcast_*` methods handle
//! NumPy-style broadcasting, GPU dispatch, and tracing. These `*_tensor`
//! methods delegate directly.
//!
//! Scalar comparison (`eq(f64)`, `ne(f64)`, etc.) lives in
//! `selection/compare.rs`.
//!
//! `where_cond` (ternary select) lives in `selection/where_cond.rs` and
//! is already a method on `DynTensor`.

use super::super::DynTensor;
use crate::Result;

impl DynTensor {
    /// Element-wise equality comparison with another tensor.
    ///
    /// Returns a U8 tensor (CPU) or F32 tensor (GPU) with 1 where elements
    /// are equal and 0 elsewhere. Supports NumPy-style broadcasting.
    ///
    /// Alias for [`broadcast_eq`](Self::broadcast_eq).
    pub fn eq_tensor(&self, rhs: &Self) -> Result<Self> {
        self.broadcast_eq(rhs)
    }

    /// Element-wise not-equal comparison with another tensor.
    ///
    /// Returns a U8 tensor (CPU) or F32 tensor (GPU) with 1 where elements
    /// differ and 0 where they are equal. Supports NumPy-style broadcasting.
    ///
    /// Alias for [`broadcast_ne`](Self::broadcast_ne).
    pub fn ne_tensor(&self, rhs: &Self) -> Result<Self> {
        self.broadcast_ne(rhs)
    }

    /// Element-wise less-than comparison with another tensor.
    ///
    /// Returns a U8 tensor (CPU) or F32 tensor (GPU) with 1 where
    /// `self[i] < rhs[i]` and 0 elsewhere. Supports NumPy-style broadcasting.
    ///
    /// Alias for [`broadcast_lt`](Self::broadcast_lt).
    pub fn lt_tensor(&self, rhs: &Self) -> Result<Self> {
        self.broadcast_lt(rhs)
    }

    /// Element-wise less-than-or-equal comparison with another tensor.
    ///
    /// Returns a U8 tensor (CPU) or F32 tensor (GPU) with 1 where
    /// `self[i] <= rhs[i]` and 0 elsewhere. Supports NumPy-style broadcasting.
    ///
    /// Alias for [`broadcast_le`](Self::broadcast_le).
    pub fn le_tensor(&self, rhs: &Self) -> Result<Self> {
        self.broadcast_le(rhs)
    }

    /// Element-wise greater-than comparison with another tensor.
    ///
    /// Returns a U8 tensor (CPU) or F32 tensor (GPU) with 1 where
    /// `self[i] > rhs[i]` and 0 elsewhere. Supports NumPy-style broadcasting.
    ///
    /// Alias for [`broadcast_gt`](Self::broadcast_gt).
    pub fn gt_tensor(&self, rhs: &Self) -> Result<Self> {
        self.broadcast_gt(rhs)
    }

    /// Element-wise greater-than-or-equal comparison with another tensor.
    ///
    /// Returns a U8 tensor (CPU) or F32 tensor (GPU) with 1 where
    /// `self[i] >= rhs[i]` and 0 elsewhere. Supports NumPy-style broadcasting.
    ///
    /// Alias for [`broadcast_ge`](Self::broadcast_ge).
    pub fn ge_tensor(&self, rhs: &Self) -> Result<Self> {
        self.broadcast_ge(rhs)
    }
}

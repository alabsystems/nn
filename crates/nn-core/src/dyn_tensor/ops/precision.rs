// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Precision-aware DynTensor operations.
//!
//! These methods accept a [`MixedPrecisionPolicy`] and cast inputs/outputs
//! to the appropriate dtype before and after computation. They are opt-in —
//! existing `matmul()` / `softmax()` / `layer_norm()` are unchanged.

use super::super::DynTensor;
use crate::mixed_precision::{MixedPrecisionPolicy, OpDTypeCategory};
use crate::Result;

impl DynTensor {
    /// Matrix multiplication with precision-aware dtype selection.
    ///
    /// When a `MixedPrecisionPolicy` is active:
    /// 1. Casts inputs to `compute_dtype` (e.g., F16 on Apple Silicon) before matmul
    /// 2. Performs matmul in `compute_dtype`
    /// 3. Returns the result in `compute_dtype` (consumer casts if needed)
    ///
    /// Without a policy: use [`matmul()`](Self::matmul) directly (current behavior).
    pub fn matmul_with_policy(
        &self,
        rhs: &Self,
        policy: &MixedPrecisionPolicy,
    ) -> Result<Self> {
        let compute_dt = policy.dtype_for_op(OpDTypeCategory::Compute);
        let lhs = if self.dtype() != compute_dt {
            self.to_dtype(compute_dt)?
        } else {
            self.clone()
        };
        let rhs = if rhs.dtype() != compute_dt {
            rhs.to_dtype(compute_dt)?
        } else {
            rhs.clone()
        };
        lhs.matmul(&rhs)
    }

    /// Softmax with precision-aware dtype selection.
    ///
    /// Softmax is classified as `Accumulate` — numerically sensitive, requires
    /// full precision. Casts to `accumulate_dtype` (F32) before computation.
    pub fn softmax_with_policy(
        &self,
        dim: impl crate::dyn_tensor::Dim,
        policy: &MixedPrecisionPolicy,
    ) -> Result<Self> {
        let acc_dt = policy.dtype_for_op(OpDTypeCategory::Accumulate);
        let input = if self.dtype() != acc_dt {
            self.to_dtype(acc_dt)?
        } else {
            self.clone()
        };
        input.softmax(dim)
    }

    /// Layer normalization with precision-aware dtype selection.
    ///
    /// LayerNorm is classified as `Accumulate` — numerically sensitive.
    /// Casts input to `accumulate_dtype` (F32) before normalization.
    ///
    /// Note: This wraps the mean/var/normalize sequence, not `layers::LayerNorm`
    /// which handles its own weight loading.
    pub fn layer_norm_with_policy(
        &self,
        dim: impl crate::dyn_tensor::Dim,
        eps: f64,
        policy: &MixedPrecisionPolicy,
    ) -> Result<Self> {
        let acc_dt = policy.dtype_for_op(OpDTypeCategory::Accumulate);
        let input = if self.dtype() != acc_dt {
            self.to_dtype(acc_dt)?
        } else {
            self.clone()
        };
        // Manual layer norm: (x - mean) / sqrt(var + eps)
        let resolved = dim.to_index(input.rank())?;
        let mean = input.mean_keepdim(resolved)?;
        let centered = input.broadcast_sub(&mean)?;
        let var = centered.sqr()?.mean_keepdim(resolved)?;
        let std = var.add_scalar(eps)?.sqrt()?;
        centered.broadcast_div(&std)
    }
}

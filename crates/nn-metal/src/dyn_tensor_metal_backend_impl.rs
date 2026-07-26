// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! GPU backend trait implementations for `MetalDynBackend`.
//!
//! The 4 GPU sub-traits ([`GpuBackend`], [`GpuShapeOps`], [`GpuNnOps`],
//! [`GpuSelectionOps`]) are implemented separately, decomposing the former
//! monolithic 37-method impl into focused blocks (#1917).
//!
//! Delegates each operation to static methods on `MetalDynBackend` defined in
//! the `ops`, `ops_reduce`, `matmul`, `shape_ops`, `conv_ops`, `norm_ops`,
//! `data_ops`, `scatter_ops`, `argreduce_ops`, and `cumsum_ops` submodules.

use nn_core::dyn_tensor::{BinaryOp, DynTensor, GpuBackend, ReduceOp, UnaryOp};
use nn_core::Result;

use super::MetalDynBackend;

// Helper functions extracted to dyn_tensor_metal_backend_helpers.rs (500-line limit).
#[path = "dyn_tensor_metal_backend_helpers.rs"]
mod helpers;

// Sub-trait implementations extracted to separate files (#1917).
#[path = "dyn_tensor_metal_backend_nn.rs"]
mod backend_nn;
#[path = "dyn_tensor_metal_backend_selection.rs"]
mod backend_selection;
#[path = "dyn_tensor_metal_backend_shape.rs"]
mod backend_shape;

// ---------------------------------------------------------------------------
// GpuBackend: 8 core methods (binary, unary, reduce, matmul, transfer, NaN)
// ---------------------------------------------------------------------------

impl GpuBackend for MetalDynBackend {
    fn binary_op(&self, op: BinaryOp, lhs: &DynTensor, rhs: &DynTensor) -> Result<DynTensor> {
        Self::gpu_binary(op, lhs, rhs)
    }

    fn unary_op(&self, op: UnaryOp, x: &DynTensor) -> Result<DynTensor> {
        match op {
            UnaryOp::Silu => Self::gpu_silu(x),
            _ => Self::gpu_unary(op, x),
        }
    }

    fn reduce_op(
        &self,
        op: ReduceOp,
        x: &DynTensor,
        dim: usize,
        keepdim: bool,
    ) -> Result<DynTensor> {
        match op {
            ReduceOp::Sum | ReduceOp::Mean | ReduceOp::Max | ReduceOp::Min => {
                if x.rank() > 0 && dim + 1 != x.rank() {
                    // Non-last-axis: transpose→reduce(last)→transpose on GPU.
                    Self::gpu_reduce_via_transpose(op, x, dim, keepdim)
                } else {
                    Self::gpu_reduce(op, x, dim, keepdim)
                }
            }
            other => Err(nn_core::TensorError::Unsupported(format!(
                "Metal GPU reduce_op: unsupported ReduceOp variant {other:?}"
            ))),
        }
    }

    fn reduce_op_compensated(
        &self,
        op: ReduceOp,
        x: &DynTensor,
        dim: usize,
        keepdim: bool,
    ) -> Result<DynTensor> {
        Self::gpu_reduce_compensated(op, x, dim, keepdim)
    }

    fn matmul(&self, lhs: &DynTensor, rhs: &DynTensor) -> Result<DynTensor> {
        Self::gpu_matmul(lhs, rhs)
    }

    fn to_gpu(&self, x: &DynTensor) -> Result<DynTensor> {
        Self::cpu_to_gpu(x)
    }

    fn to_cpu(&self, x: &DynTensor) -> Result<DynTensor> {
        Self::gpu_to_cpu(x)
    }

    fn count_non_finite(&self, x: &DynTensor) -> Option<Result<usize>> {
        Some(Self::gpu_count_non_finite(x))
    }

    fn cast_dtype(
        &self,
        x: &DynTensor,
        target_dtype: nn_core::DType,
    ) -> Option<Result<DynTensor>> {
        Self::gpu_cast_dtype(x, target_dtype)
    }

    fn backend_name(&self) -> &'static str {
        "metal"
    }

    fn flush_pending(&self) -> Result<()> {
        crate::gpu_scope::flush()
    }
}

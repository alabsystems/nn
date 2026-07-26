// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! CPU arithmetic, reduction, and matmul operations for [`DynTensor`].
//!
//! Unary math functions (exp, log, sqrt, activations, clamp) are in
//! the `math` submodule. GPU tensors dispatch through the registered
//! [`GpuBackend`]. CPU tensors use ndarray directly.

use std::sync::Arc;

use super::{
    gpu_backend, gpu_backend_dispatch, trace, BinaryOp, DynTensor, FloatStorage, TensorStorage,
};
use crate::dyn_tensor::trace::TraceOp;
use crate::{Result, TensorError};

mod binary;
mod compare;
mod indexing;
mod math;
mod matmul;
mod pad;
pub mod scatter_gather;
mod topk;

pub(crate) use binary::{broadcast_output_shape, check_same_device};
pub use scatter_gather::{gather, index_select, scatter, scatter_add};

// -- Binary arithmetic --------------------------------------------------------
//
// **Named methods broadcast (matching candle):** `.add()`, `.sub()`, `.mul()`,
// `.div()` all use NumPy-style right-aligned broadcasting and accept compatible
// shapes. This matches candle's `Tensor::add/sub/mul/div` behavior for seamless
// candle→nn migration. The `broadcast_*` variants are aliases.
//
// **Strict variants:** `.strict_add()`, `.strict_sub()`, `.strict_mul()`,
// `.strict_div()` require exact shape match and return an error on mismatch.
// Use these when you want to enforce shape equality.
//
// Operator overloads (`+`, `-`, `*`, `/`) also delegate to broadcast variants.

impl DynTensor {
    /// In-place element-wise addition: `self += rhs`.
    ///
    /// When the tensor's internal storage has a reference count of 1 (sole
    /// owner), mutates the buffer directly — zero allocation. This is the
    /// common case in gradient accumulation where each gradient is owned only
    /// by the [`GradStore`](nn_autodiff::GradStore).
    ///
    /// When the reference count is >1 (shared storage, e.g. from `.clone()`
    /// or zero-copy `narrow`), falls back to allocating `self.add(rhs)`.
    ///
    /// Requires identical shape and dtype (no broadcasting). For broadcast
    /// addition, use [`add`](Self::add).
    pub fn add_assign(&mut self, rhs: &Self) -> Result<()> {
        if self.dims != rhs.dims {
            return Err(TensorError::shape_mismatch(
                self.dims.clone(),
                rhs.dims().to_vec(),
            ));
        }
        if self.dtype != rhs.dtype {
            return Err(TensorError::dtype_mismatch(self.dtype, rhs.dtype));
        }
        if self.device() != rhs.device() {
            return Err(TensorError::Unsupported("add_assign: mixed devices".into()));
        }
        // CPU in-place path: try to get mutable access to the Arc'd storage.
        // Succeeds when refcount == 1 (common in gradient accumulation where
        // each gradient entry is uniquely owned by the GradStore).
        if let TensorStorage::Cpu(ref mut arc) = self.storage {
            if let Some(any_mut) = Arc::get_mut(arc) {
                if let Some(fs) = any_mut.downcast_mut::<FloatStorage>() {
                    if let TensorStorage::Cpu(rhs_arc) = &rhs.storage {
                        if let Some(rhs_fs) = rhs_arc.downcast_ref::<FloatStorage>() {
                            return fs.add_assign(rhs_fs);
                        }
                    }
                    // rhs uses legacy storage (raw ArrayD<f32>) — fall through
                    // to allocating add which handles both storage types.
                }
            }
        }
        // Fallback: allocating add (shared storage, GPU, or non-FloatStorage).
        *self = self.add(rhs)?;
        Ok(())
    }

    /// Element-wise addition with NumPy-style broadcasting.
    ///
    /// Compatible shapes are broadcast automatically (e.g., `[3,1] + [1,4]` → `[3,4]`).
    /// Matches candle's `Tensor::add()` behavior. For strict same-shape requirement,
    /// use [`strict_add`](Self::strict_add).
    pub fn add(&self, rhs: &Self) -> Result<Self> {
        self.broadcast_binary_op(BinaryOp::Add, rhs)
    }

    /// Element-wise subtraction with NumPy-style broadcasting.
    ///
    /// Matches candle's `Tensor::sub()` behavior. For strict same-shape requirement,
    /// use [`strict_sub`](Self::strict_sub).
    pub fn sub(&self, rhs: &Self) -> Result<Self> {
        self.broadcast_binary_op(BinaryOp::Sub, rhs)
    }

    /// Element-wise multiplication with NumPy-style broadcasting.
    ///
    /// Matches candle's `Tensor::mul()` behavior. For strict same-shape requirement,
    /// use [`strict_mul`](Self::strict_mul).
    pub fn mul(&self, rhs: &Self) -> Result<Self> {
        self.broadcast_binary_op(BinaryOp::Mul, rhs)
    }

    /// Element-wise division with NumPy-style broadcasting.
    ///
    /// Matches candle's `Tensor::div()` behavior. For strict same-shape requirement,
    /// use [`strict_div`](Self::strict_div).
    pub fn div(&self, rhs: &Self) -> Result<Self> {
        self.broadcast_binary_op(BinaryOp::Div, rhs)
    }

    /// Element-wise addition (**strict**: requires identical shapes).
    pub fn strict_add(&self, rhs: &Self) -> Result<Self> {
        self.binary_op_impl(BinaryOp::Add, rhs)
    }

    /// Element-wise subtraction (**strict**: requires identical shapes).
    pub fn strict_sub(&self, rhs: &Self) -> Result<Self> {
        self.binary_op_impl(BinaryOp::Sub, rhs)
    }

    /// Element-wise multiplication (**strict**: requires identical shapes).
    pub fn strict_mul(&self, rhs: &Self) -> Result<Self> {
        self.binary_op_impl(BinaryOp::Mul, rhs)
    }

    /// Element-wise division (**strict**: requires identical shapes).
    pub fn strict_div(&self, rhs: &Self) -> Result<Self> {
        self.binary_op_impl(BinaryOp::Div, rhs)
    }

    /// Broadcast addition (alias for [`add`](Self::add)).
    pub fn broadcast_add(&self, rhs: &Self) -> Result<Self> {
        self.broadcast_binary_op(BinaryOp::Add, rhs)
    }

    /// Broadcast subtraction (alias for [`sub`](Self::sub)).
    pub fn broadcast_sub(&self, rhs: &Self) -> Result<Self> {
        self.broadcast_binary_op(BinaryOp::Sub, rhs)
    }

    /// Broadcast multiplication (alias for [`mul`](Self::mul)).
    pub fn broadcast_mul(&self, rhs: &Self) -> Result<Self> {
        self.broadcast_binary_op(BinaryOp::Mul, rhs)
    }

    /// Broadcast division (alias for [`div`](Self::div)).
    pub fn broadcast_div(&self, rhs: &Self) -> Result<Self> {
        self.broadcast_binary_op(BinaryOp::Div, rhs)
    }

    /// Create a scalar tensor on the same device as `self`.
    fn scalar_like(&self, val: f64) -> Result<Self> {
        Self::full(&[], val, self.dtype, &self.device())
    }

    /// Affine transform: `self * mul + add` (element-wise).
    ///
    /// GPU tensors use fused scalar kernels when available (#3230 Gap 2),
    /// eliminating scalar_like() CPU alloc + GPU transfer overhead.
    pub fn affine(&self, mul_val: f64, add_val: f64) -> Result<Self> {
        // Fused GPU path bypasses trace recording — skip when tracing.
        if self.device().is_gpu() && !trace::is_tracing() {
            if let Some(mul_result) =
                gpu_backend_dispatch(|b| b.scalar_binary_op(BinaryOp::Mul, self, mul_val))
            {
                let intermediate = mul_result?;
                if let Some(add_result) = gpu_backend_dispatch(|b| {
                    b.scalar_binary_op(BinaryOp::Add, &intermediate, add_val)
                }) {
                    return add_result;
                }
            }
        }
        let m = self.scalar_like(mul_val)?;
        let a = self.scalar_like(add_val)?;
        self.broadcast_mul(&m)?.broadcast_add(&a)
    }

    // -- Scalar arithmetic (tensor op scalar) ---------------------------------

    /// Add a scalar to every element.
    ///
    /// GPU tensors use a fused scalar kernel when available (#3230 Gap 2),
    /// avoiding scalar_like() alloc + broadcast.
    pub fn add_scalar(&self, val: f64) -> Result<Self> {
        // Fused GPU path bypasses trace recording — skip when tracing.
        if self.device().is_gpu() && !trace::is_tracing() {
            if let Some(result) =
                gpu_backend_dispatch(|b| b.scalar_binary_op(BinaryOp::Add, self, val))
            {
                return result;
            }
        }
        let s = self.scalar_like(val)?;
        self.broadcast_add(&s)
    }

    /// Multiply every element by a scalar.
    ///
    /// GPU tensors use a fused scalar kernel when available (#3230 Gap 2),
    /// avoiding scalar_like() alloc + broadcast.
    pub fn mul_scalar(&self, val: f64) -> Result<Self> {
        // Fused GPU path bypasses trace recording — skip when tracing.
        if self.device().is_gpu() && !trace::is_tracing() {
            if let Some(result) =
                gpu_backend_dispatch(|b| b.scalar_binary_op(BinaryOp::Mul, self, val))
            {
                return result;
            }
        }
        let s = self.scalar_like(val)?;
        self.broadcast_mul(&s)
    }

    /// Subtract a scalar from every element.
    ///
    /// GPU tensors use a fused scalar kernel when available (#3230 Gap 2),
    /// avoiding scalar_like() alloc + broadcast.
    pub fn sub_scalar(&self, val: f64) -> Result<Self> {
        // Fused GPU path bypasses trace recording — skip when tracing.
        if self.device().is_gpu() && !trace::is_tracing() {
            if let Some(result) =
                gpu_backend_dispatch(|b| b.scalar_binary_op(BinaryOp::Sub, self, val))
            {
                return result;
            }
        }
        self.add_scalar(-val)
    }

    /// Divide every element by a scalar.
    ///
    /// Returns `ZeroDivisor` error if `val` is exactly zero. For near-zero
    /// values, the division proceeds with IEEE 754 semantics (may produce Inf).
    ///
    /// GPU tensors use a fused scalar kernel when available (#3230 Gap 2),
    /// avoiding scalar_like() alloc + broadcast.
    pub fn div_scalar(&self, val: f64) -> Result<Self> {
        if val == 0.0 {
            return Err(TensorError::Unsupported(
                "div_scalar: divisor is zero".into(),
            ));
        }
        // Fused GPU path bypasses trace recording — skip when tracing.
        if self.device().is_gpu() && !trace::is_tracing() {
            if let Some(result) =
                gpu_backend_dispatch(|b| b.scalar_binary_op(BinaryOp::Div, self, val))
            {
                return result;
            }
        }
        let s = self.scalar_like(val)?;
        self.broadcast_div(&s)
    }
}

// -- Reductions ---------------------------------------------------------------
mod reduce;

// -- MatMul -------------------------------------------------------------------

impl DynTensor {
    /// Matrix multiplication.
    ///
    /// Supports:
    /// - 2D × 2D: [M, K] × [K, N] → [M, N]
    /// - 3D × 3D: [B, M, K] × [B, K, N] → [B, M, N]
    /// - 3D × 2D: [B, M, K] × [K, N] → [B, M, N] (broadcast)
    /// - 4D × 4D: [B, H, M, K] × [B, H, K, N] → [B, H, M, N]
    /// - 4D × 2D: [B, H, M, K] × [K, N] → [B, H, M, N] (broadcast)
    pub fn matmul(&self, rhs: &Self) -> Result<Self> {
        // Auto-dequantize quantized operands before matmul dispatch.
        if self.is_quantized() || rhs.is_quantized() {
            let lhs_deq = self.auto_dequantize()?;
            let rhs_deq = rhs.auto_dequantize()?;
            return lhs_deq.matmul(&rhs_deq);
        }
        check_same_device(self, rhs)?;
        let mut result = match (&self.storage, &rhs.storage) {
            (TensorStorage::Cpu(_), TensorStorage::Cpu(_)) => self.cpu_matmul(rhs),
            (TensorStorage::Gpu { .. }, TensorStorage::Gpu { .. }) => {
                let backend = gpu_backend()?;
                backend.matmul(self, rhs)
            }
            _ => Err(TensorError::Unsupported("mixed CPU/GPU storage".into())),
        }?;
        if trace::is_tracing() {
            let input_ids = Self::trace_input_ids(&[self, rhs])?;
            if let Some(id) =
                trace::record_op(TraceOp::MatMul, &input_ids, result.dims(), result.dtype())
            {
                result.set_trace_id(id);
            }
        }
        Ok(result)
    }

    fn cpu_matmul(&self, rhs: &Self) -> Result<Self> {
        matmul::cpu_matmul(self, rhs)
    }
}

// -- Einsum -------------------------------------------------------------------
pub mod einsum;
pub use einsum::{einsum, EinsumNotation};

// -- Softmax ------------------------------------------------------------------
pub(crate) mod softmax;
#[allow(deprecated)]
pub use softmax::softmax_last_dim;

// -- Precision-aware ops (opt-in, with MixedPrecisionPolicy) ------------------
mod precision;

// -- Operator overloads (std::ops impls) --------------------------------------
mod overloads;

#[cfg(test)]
mod tests;

#[cfg(test)]
mod tests_compare;

#[cfg(test)]
mod tests_pad_indexing;

#[cfg(test)]
mod tests_topk;

#[cfg(test)]
mod tests_precision;

#[cfg(test)]
mod tests_reduce;

#[cfg(test)]
mod tests_shape_ops;

#[cfg(test)]
mod tests_compare_reduce;

#[cfg(kani)]
#[path = "kani_binary_broadcast_proofs.rs"]
mod kani_binary_broadcast_proofs;
#[cfg(kani)]
#[path = "kani_binary_proofs.rs"]
mod kani_binary_proofs;
#[cfg(kani)]
#[path = "kani_dpdf_log_softmax_proofs.rs"]
mod kani_dpdf_log_softmax_proofs;
#[cfg(kani)]
#[path = "kani_math_compound.rs"]
mod kani_math_compound;
#[cfg(kani)]
#[path = "kani_math_compound_extended.rs"]
mod kani_math_compound_extended;
#[cfg(kani)]
#[path = "kani_matmul_proofs.rs"]
mod kani_matmul_proofs;
#[cfg(kani)]
#[path = "kani_precision_proofs.rs"]
mod kani_precision_proofs;
#[cfg(kani)]
#[path = "kani_reduce.rs"]
mod kani_reduce;
#[cfg(kani)]
#[path = "kani_reduce_extended.rs"]
mod kani_reduce_extended;
#[cfg(kani)]
#[path = "kani_reduce_ops_proofs.rs"]
mod kani_reduce_ops_proofs;
#[cfg(kani)]
#[path = "kani_softmax.rs"]
mod kani_softmax;
#[cfg(kani)]
#[path = "kani_softmax_edge_cases.rs"]
mod kani_softmax_edge_cases;
#[cfg(kani)]
#[path = "kani_unary_math_proofs.rs"]
mod kani_unary_math_proofs;

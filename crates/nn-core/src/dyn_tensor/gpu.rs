// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! GPU backend traits and operation enums for DynTensor dispatch.
//!
//! The monolithic `GpuBackend` trait (41 methods) is decomposed into 4 sub-traits:
//!
//! - [`GpuBackend`]: 8 core required methods + 3 optional (binary_op, unary_op,
//!   reduce_op, reduce_op_compensated, matmul, to_gpu, to_cpu, count_non_finite,
//!   to_ane, ane_to_gpu, backend_name)
//! - [`GpuShapeOps`]: 7 shape methods (narrow, transpose, permute, cat, expand,
//!   unfold, slice_set)
//! - [`GpuNnOps`]: 11 NN methods (softmax, log_softmax, conv1d, conv2d, conv_transpose1d,
//!   layer_norm, group_norm, rms_norm, rope, lstm_cell, lstm_sequence)
//! - [`GpuSelectionOps`]: 12 selection methods (index_select, gather, compare,
//!   compare_tensor, where_cond, index_add, scatter_add, cumsum, repeat_interleave,
//!   argmax, argmin, topk)
//!
//! [`GpuFullBackend`] is the alias trait combining all four. The global singleton
//! stores `Box<dyn GpuFullBackend>` and all dispatch functions operate on it.

use super::DynTensor;
use crate::{Result, TensorError};
use std::sync::OnceLock;

// Sub-trait modules.
#[path = "gpu_nn.rs"]
mod gpu_nn;
#[path = "gpu_selection.rs"]
mod gpu_selection;
#[path = "gpu_shape.rs"]
mod gpu_shape;

pub use gpu_nn::GpuNnOps;
pub use gpu_selection::GpuSelectionOps;
pub use gpu_shape::GpuShapeOps;

/// Core GPU backend trait: 8 fundamental methods required for GPU dispatch.
///
/// Implemented by nn-metal (and future nn-cuda, nn-vulkan). Backend types
/// also implement [`GpuShapeOps`], [`GpuNnOps`], and [`GpuSelectionOps`] to
/// provide the full 41-method surface via [`GpuFullBackend`].
///
/// Registered once via [`register_gpu_backend`].
pub trait GpuBackend: Send + Sync {
    /// Element-wise binary operation.
    fn binary_op(&self, op: BinaryOp, lhs: &DynTensor, rhs: &DynTensor) -> Result<DynTensor>;
    /// Element-wise unary operation.
    fn unary_op(&self, op: UnaryOp, x: &DynTensor) -> Result<DynTensor>;
    /// Reduction along a dimension with keepdim control.
    fn reduce_op(
        &self,
        op: ReduceOp,
        x: &DynTensor,
        dim: usize,
        keepdim: bool,
    ) -> Result<DynTensor>;

    /// Compensated (Kahan) reduction for near-f64 precision from f32 (#1814).
    ///
    /// Default: falls back to [`reduce_op`](Self::reduce_op). Backends that
    /// support `PrecisionTier::Strict` (e.g., Metal) override this to emit
    /// Kahan-compensated MSL kernels.
    fn reduce_op_compensated(
        &self,
        op: ReduceOp,
        x: &DynTensor,
        dim: usize,
        keepdim: bool,
    ) -> Result<DynTensor> {
        self.reduce_op(op, x, dim, keepdim)
    }

    /// Matrix multiplication.
    fn matmul(&self, lhs: &DynTensor, rhs: &DynTensor) -> Result<DynTensor>;
    /// Transfer tensor to this GPU device.
    fn to_gpu(&self, x: &DynTensor) -> Result<DynTensor>;
    /// Transfer tensor from GPU to CPU.
    fn to_cpu(&self, x: &DynTensor) -> Result<DynTensor>;

    /// Count non-finite (NaN/Inf) elements in a GPU tensor.
    ///
    /// Returns the number of non-finite elements without constructing a full
    /// CPU tensor. Used by [`crate::layers::check_output_finite`] to enable Tier 1
    /// per-layer NaN guards on GPU tensors (#1320).
    ///
    /// Default implementation returns `None` (caller falls back to CPU round-trip
    /// or skips the check). Metal backend reads the GPU buffer directly.
    fn count_non_finite(&self, _x: &DynTensor) -> Option<Result<usize>> {
        None
    }

    /// Transfer a tensor to the Apple Neural Engine (via CoreML).
    ///
    /// Returns `None` if the backend does not support ANE transfers.
    /// Used by dvoice for Kokoro model dispatch to ANE (#1954).
    fn to_ane(&self, _x: &DynTensor) -> Option<Result<DynTensor>> {
        None
    }

    /// Transfer a tensor from the Apple Neural Engine back to this backend.
    ///
    /// Returns `None` if the backend does not support ANE transfers.
    /// Used by dvoice for Kokoro model dispatch from ANE (#1954).
    fn ane_to_gpu(&self, _x: &DynTensor) -> Option<Result<DynTensor>> {
        None
    }

    /// Cast a GPU tensor to a different float dtype without CPU round-trip.
    ///
    /// Handles cross-byte-width conversions (F32↔F16, F32↔BF16) that cannot
    /// use zero-copy relabel. Returns `None` if the backend doesn't support
    /// GPU-native dtype casting (caller falls back to CPU round-trip).
    ///
    /// The default implementation returns `None`. Metal backend overrides this
    /// with a raw MSL kernel that reads one type and writes the other.
    fn cast_dtype(&self, _x: &DynTensor, _target_dtype: crate::DType) -> Option<Result<DynTensor>> {
        None
    }

    /// Human-readable name of this backend (e.g., `"metal"`, `"cuda"`).
    ///
    /// Used for logging, error messages, and backend selection by consumers.
    /// Default returns `"unknown"`.
    fn backend_name(&self) -> &'static str {
        "unknown"
    }

    /// Flush pending GPU work without CPU readback.
    ///
    /// For lazy-batching backends (Metal), this commits the current command
    /// buffer and starts a new one. Used to prevent command buffer depth
    /// pathology on deep models (e.g., SigLip2 with 200+ dispatches).
    /// Source: #4319
    fn flush_pending(&self) -> Result<()> {
        Ok(())
    }
}

/// Alias trait combining all 4 GPU sub-traits into the full 41-method surface.
///
/// This is the type stored in the global backend registry. Backend crates
/// implement `GpuBackend + GpuShapeOps + GpuNnOps + GpuSelectionOps`
/// and the blanket impl provides `GpuFullBackend` automatically.
pub trait GpuFullBackend: GpuBackend + GpuShapeOps + GpuNnOps + GpuSelectionOps {}

/// Blanket impl: any type implementing all 4 sub-traits is a `GpuFullBackend`.
impl<T: GpuBackend + GpuShapeOps + GpuNnOps + GpuSelectionOps> GpuFullBackend for T {}

// GPU operation discriminant enums extracted to gpu_ops.rs (#1575).
#[path = "gpu_ops.rs"]
mod ops;
pub use ops::{BinaryOp, CompareOp, ReduceOp, UnaryOp};

static GPU_BACKEND: OnceLock<Box<dyn GpuFullBackend>> = OnceLock::new();

/// Register a GPU backend for `DynTensor` operations. Called once at startup
/// by the backend crate (e.g., `nn_metal::register_metal_backend()`).
pub fn register_gpu_backend(backend: Box<dyn GpuFullBackend>) {
    GPU_BACKEND.set(backend).ok();
}

/// Get the registered GPU backend, or error if none registered.
pub(crate) fn gpu_backend() -> Result<&'static dyn GpuFullBackend> {
    GPU_BACKEND
        .get()
        .map(AsRef::as_ref)
        .ok_or_else(|| TensorError::Unsupported("no GPU backend registered".into()))
}

/// Try to dispatch an optional GPU op. Returns `None` if no backend
/// is registered or the backend doesn't support this op.
pub(crate) fn gpu_backend_dispatch(
    f: impl FnOnce(&dyn GpuFullBackend) -> Option<Result<DynTensor>>,
) -> Option<Result<DynTensor>> {
    let backend = GPU_BACKEND.get()?.as_ref();
    f(backend)
}

/// Try to dispatch a GPU op returning a pair of tensors (e.g., LSTM cell h/c).
/// Returns `None` if no backend is registered or the backend doesn't support this op.
pub(crate) fn gpu_backend_dispatch_pair(
    f: impl FnOnce(&dyn GpuFullBackend) -> Option<Result<(DynTensor, DynTensor)>>,
) -> Option<Result<(DynTensor, DynTensor)>> {
    let backend = GPU_BACKEND.get()?.as_ref();
    f(backend)
}

/// Try to dispatch a GPU op returning a triple of tensors (e.g., LSTM sequence).
/// Returns `None` if no backend is registered or the backend doesn't support this op.
pub(crate) fn gpu_backend_dispatch_triple(
    f: impl FnOnce(&dyn GpuFullBackend) -> Option<Result<(DynTensor, DynTensor, DynTensor)>>,
) -> Option<Result<(DynTensor, DynTensor, DynTensor)>> {
    let backend = GPU_BACKEND.get()?.as_ref();
    f(backend)
}

/// Try to count non-finite elements in a GPU tensor via the registered backend.
/// Returns `None` if no backend is registered or it doesn't support the op.
pub(crate) fn gpu_backend_dispatch_count_non_finite(x: &DynTensor) -> Option<Result<usize>> {
    let backend = GPU_BACKEND.get()?.as_ref();
    backend.count_non_finite(x)
}

/// Flush pending GPU work without CPU readback.
///
/// For lazy-batching backends (Metal), commits the current command buffer
/// to prevent deep dependency chains from causing GPU timeout. No-op if
/// no backend is registered or the backend doesn't support lazy batching.
/// Source: #4319
pub fn gpu_backend_flush() -> Result<()> {
    if let Some(backend) = GPU_BACKEND.get() {
        backend.flush_pending()?;
    }
    Ok(())
}

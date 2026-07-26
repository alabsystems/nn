// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! GPU-resident LoRA overlay for live adapter application on Metal.
//!
//! [`LoraGpuOverlay`] computes effective weights `W + scaling * (B @ A)` entirely
//! on GPU using existing DynTensor matmul + mul_scalar + add operations. The base
//! weight is never mutated — `apply()` returns a new GPU tensor.
//!
//! This is the inference-side complement to `nn_optim::LoraLinear` (training-side).
//! dvoice uses this for live adapter swapping (voice identity switching).

use nn_core::{DynTensor, Result, TensorError};

/// A LoRA overlay that applies rank-r updates on GPU without mutating base weights.
///
/// Holds GPU-resident A and B matrices plus a scaling factor. `apply()` computes
/// `W + scaling * (B @ A)` using 3 GPU dispatches (matmul, mul_scalar, add).
///
/// # Usage
///
/// ```rust,no_run
/// # use nn_core::DynTensor;
/// // Create overlay from GPU-resident A and B tensors
/// # fn example(a_gpu: DynTensor, b_gpu: DynTensor, base_weight: DynTensor) -> nn_core::Result<()> {
/// use nn_metal::lora_overlay::LoraGpuOverlay;
/// let overlay = LoraGpuOverlay::from_tensors(a_gpu, b_gpu, 1.0)?;
/// let w_eff = overlay.apply(&base_weight)?;
/// // w_eff is a new GPU tensor — base_weight is unchanged.
/// // To remove the adapter: drop w_eff and use base_weight directly.
/// # Ok(()) }
/// ```
#[derive(Debug)]
pub struct LoraGpuOverlay {
    /// A matrix `[rank, in_features]` on GPU.
    a: DynTensor,
    /// B matrix `[out_features, rank]` on GPU.
    b: DynTensor,
    /// Scaling factor (`alpha / rank`).
    scaling: f32,
}

impl LoraGpuOverlay {
    /// Create from GPU-resident A and B tensors.
    ///
    /// Validates:
    /// - `scaling` is finite (rejects NaN/Inf)
    /// - A and B are 2D matrices
    /// - Inner dimensions match: A is `[rank, in]`, B is `[out, rank]`
    /// - Rank > 0
    pub fn from_tensors(a: DynTensor, b: DynTensor, scaling: f32) -> Result<Self> {
        if !scaling.is_finite() {
            return Err(TensorError::InvalidShape(
                "LoraGpuOverlay: scaling must be finite".into(),
            ));
        }
        if a.rank() != 2 {
            return Err(TensorError::InvalidShape(format!(
                "LoraGpuOverlay: A must be 2D, got rank {}",
                a.rank()
            )));
        }
        if b.rank() != 2 {
            return Err(TensorError::InvalidShape(format!(
                "LoraGpuOverlay: B must be 2D, got rank {}",
                b.rank()
            )));
        }
        let rank_a = a.dims()[0]; // A is [rank, in_features]
        let rank_b = b.dims()[1]; // B is [out_features, rank]
        if rank_a != rank_b {
            return Err(TensorError::InvalidShape(format!(
                "LoraGpuOverlay: inner rank mismatch: A[0]={rank_a} != B[1]={rank_b}"
            )));
        }
        if rank_a == 0 {
            return Err(TensorError::InvalidShape(
                "LoraGpuOverlay: rank must be > 0".into(),
            ));
        }
        Ok(Self { a, b, scaling })
    }

    /// Compute effective weight: `W_eff = W + scaling * (B @ A)`.
    ///
    /// Returns a **new** DynTensor. The base weight `w` is not mutated.
    /// Uses GPU matmul (simdgroup for conforming shapes) and GPU element-wise ops.
    pub fn apply(&self, w: &DynTensor) -> Result<DynTensor> {
        if w.rank() != 2 {
            return Err(TensorError::InvalidShape(format!(
                "LoraGpuOverlay::apply: base weight must be 2D, got rank {}",
                w.rank()
            )));
        }
        let expected_out = self.b.dims()[0]; // [out_features, rank]
        let expected_in = self.a.dims()[1]; // [rank, in_features]
        if w.dims()[0] != expected_out || w.dims()[1] != expected_in {
            return Err(TensorError::shape_mismatch(
                vec![expected_out, expected_in],
                w.dims().to_vec(),
            ));
        }

        // B @ A: [out, rank] @ [rank, in] → [out, in]
        let ba = self.b.matmul(&self.a)?;
        // scaling * (B @ A)
        let scaled = ba.mul_scalar(f64::from(self.scaling))?;
        // W + scaling * (B @ A)
        w.add(&scaled)
    }

    /// Swap adapter: compute effective weight with a new overlay on the same base.
    ///
    /// Equivalent to `new_overlay.apply(base_w)` — the old effective weight is
    /// simply dropped. Since `apply()` creates a new buffer and base weights are
    /// immutable, swapping is just calling `apply()` with a different overlay.
    pub fn swap(new_overlay: &Self, base_w: &DynTensor) -> Result<DynTensor> {
        new_overlay.apply(base_w)
    }

    /// Rank of the low-rank update (inner dimension of A and B).
    #[must_use]
    pub fn rank(&self) -> usize {
        self.a.dims()[0]
    }

    /// Scaling factor (`alpha / rank`).
    #[must_use]
    pub fn scaling(&self) -> f32 {
        self.scaling
    }

    /// Reference to the A matrix `[rank, in_features]`.
    #[must_use]
    pub fn a(&self) -> &DynTensor {
        &self.a
    }

    /// Reference to the B matrix `[out_features, rank]`.
    #[must_use]
    pub fn b(&self) -> &DynTensor {
        &self.b
    }
}

#[cfg(test)]
#[path = "lora_overlay_tests.rs"]
mod tests;

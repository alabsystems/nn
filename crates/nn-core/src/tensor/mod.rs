// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Core tensor type for nn with compile-time rank checking.
//!
//! [`Tensor<D, T, B>`] encodes the number of dimensions (rank) in the type
//! system. Rank mismatches (e.g., passing a 2D tensor where 3D is expected)
//! are caught at compile time. Dimension sizes remain runtime values stored
//! in `[usize; D]`.
//!
//! This is the burn-style hybrid approach: compile-time rank, runtime sizes.
//! See `designs/2026-02-26-shape-strategy.md` for the full design rationale.
//!
//! # Example
//!
//! ```rust
//! use nn_core::Tensor;
//!
//! // Create a 2D tensor — rank is part of the type.
//! let t: Tensor<2> = Tensor::zeros([2, 3]).expect("CPU allocation");
//! assert_eq!(t.dims(), &[2, 3]);
//! assert_eq!(t.ndim(), 2);
//!
//! // Rank mismatches are compile errors:
//! // let t3: Tensor<3> = t; // ERROR: expected Tensor<3>, found Tensor<2>
//! ```

use crate::backend::{Backend, CpuBackend};
use crate::{DType, Device, IntervalBounds, Result, TensorError};
use ndarray::ArrayD;
use std::sync::Arc;

// -- TensorElement trait -----------------------------------------------------------

/// Type constraint for tensor elements.
///
/// All types stored in a [`Tensor`] must implement this trait, which ensures
/// they can be zero/one initialized, copied efficiently, and mapped to a
/// [`DType`] discriminant.
pub trait TensorElement:
    Clone + Copy + Default + Send + Sync + num_traits::Zero + num_traits::One + 'static
{
    /// The data type discriminant for this element type.
    #[must_use]
    fn dtype() -> DType;
}

impl TensorElement for f32 {
    fn dtype() -> DType {
        DType::F32
    }
}

impl TensorElement for f64 {
    fn dtype() -> DType {
        DType::F64
    }
}

impl TensorElement for i32 {
    fn dtype() -> DType {
        DType::I32
    }
}

impl TensorElement for i64 {
    fn dtype() -> DType {
        DType::I64
    }
}

impl TensorElement for u8 {
    fn dtype() -> DType {
        DType::U8
    }
}

impl TensorElement for half::f16 {
    fn dtype() -> DType {
        DType::F16
    }
}

impl TensorElement for half::bf16 {
    fn dtype() -> DType {
        DType::BF16
    }
}

// Note: `bool` has DType::Bool but no TensorElement impl because `bool` does not
// implement `num_traits::Zero` or `num_traits::One`. A Bool tensor type would
// require either a newtype wrapper or a separate trait. Tracked as a known gap.

// -- Dimension arithmetic ----------------------------------------------------------

/// Compute the product of dimension sizes using checked arithmetic.
///
/// Returns an error if the product overflows `usize`. This prevents silent
/// wrap-around when large dimension sizes are provided (e.g., `[usize::MAX, 2]`).
pub fn checked_dim_product(dims: &[usize]) -> Result<usize> {
    dims.iter().try_fold(1usize, |acc, &d| {
        acc.checked_mul(d)
            .ok_or_else(|| TensorError::DimensionOverflow {
                dims: dims.to_vec(),
            })
    })
}

// -- Tensor struct -----------------------------------------------------------------

/// Device-agnostic tensor with compile-time rank and verification support.
///
/// - `D`: Number of dimensions (rank). Enforced at compile time.
/// - `T`: Element type. Defaults to `f32`.
/// - `B`: Backend for storage. Defaults to [`CpuBackend`].
///
/// Dimension sizes are runtime values in `[usize; D]`. The rank `D` is a
/// compile-time constant, so rank mismatches are type errors.
pub struct Tensor<const D: usize, T: TensorElement = f32, B: Backend = CpuBackend> {
    dims: [usize; D],
    storage: B::TensorPrimitive<T>,
    bounds: Option<IntervalBounds>,
}

// -- Generic impls (all backends) --------------------------------------------------

impl<const D: usize, T: TensorElement, B: Backend> Tensor<D, T, B> {
    /// Construct a tensor from pre-allocated backend storage.
    ///
    /// Backend crates use this to create tensors from device-specific storage
    /// (e.g., `MetalTensorStorage`). The caller is responsible for ensuring
    /// `storage` has at least `dims.iter().product()` elements.
    ///
    /// Validates that the dimension product doesn't overflow.
    #[must_use = "tensor constructor returns a Result that must be used"]
    pub fn from_storage(dims: [usize; D], storage: B::TensorPrimitive<T>) -> Result<Self> {
        checked_dim_product(&dims)?;
        Ok(Self {
            dims,
            storage,
            bounds: None,
        })
    }

    /// Get a reference to the underlying backend storage.
    #[must_use]
    pub fn storage(&self) -> &B::TensorPrimitive<T> {
        &self.storage
    }

    /// Create a zero-filled tensor with the given dimensions.
    #[must_use = "tensor constructor returns a Result that must be used"]
    pub fn zeros(dims: [usize; D]) -> Result<Self> {
        // Validate dimension product won't overflow before backend allocation.
        checked_dim_product(&dims)?;
        let storage = B::zeros::<D, T>(dims)?;
        Ok(Self {
            dims,
            storage,
            bounds: None,
        })
    }

    /// Create a one-filled tensor with the given dimensions.
    #[must_use = "tensor constructor returns a Result that must be used"]
    pub fn ones(dims: [usize; D]) -> Result<Self> {
        // Validate dimension product won't overflow before backend allocation.
        checked_dim_product(&dims)?;
        let storage = B::ones::<D, T>(dims)?;
        Ok(Self {
            dims,
            storage,
            bounds: None,
        })
    }

    /// Dimension sizes as a fixed-size array.
    #[must_use]
    pub fn dims(&self) -> &[usize; D] {
        &self.dims
    }

    /// Number of dimensions (rank). Compile-time constant.
    #[must_use]
    pub const fn ndim(&self) -> usize {
        D
    }

    /// Total number of elements (product of all dimensions).
    ///
    /// All construction paths (`zeros`, `ones`, `from_vec`, `from_ndarray`)
    /// validate dimensions via `checked_dim_product`, so this cannot overflow
    /// on a validly constructed `Tensor`. The `assert!` is defense-in-depth
    /// that remains active in release builds (unlike `debug_assert!`).
    #[must_use]
    pub fn numel(&self) -> usize {
        assert!(
            self.dims
                .iter()
                .try_fold(1usize, |a, &d| a.checked_mul(d))
                .is_some(),
            "numel overflow: dims {:?}",
            self.dims,
        );
        self.dims.iter().product()
    }

    /// The device this tensor's backend targets.
    #[must_use]
    pub fn device(&self) -> Device {
        B::device()
    }

    /// Element data type.
    #[must_use]
    pub fn dtype(&self) -> DType {
        T::dtype()
    }

    /// Attach interval bounds for verification.
    ///
    /// Bounds shape must match the tensor dimensions exactly.
    #[must_use = "with_bounds returns a new tensor; the original is consumed"]
    pub fn with_bounds(mut self, bounds: IntervalBounds) -> Result<Self> {
        let expected = self.dims.as_slice();
        let actual = bounds.shape();
        if expected != actual {
            return Err(TensorError::shape_mismatch(
                expected.to_vec(),
                actual.to_vec(),
            ));
        }
        self.bounds = Some(bounds);
        Ok(self)
    }

    /// Get interval bounds (for NY integration).
    #[must_use]
    pub fn bounds(&self) -> Option<&IntervalBounds> {
        self.bounds.as_ref()
    }
}

// -- CPU-specific impls ------------------------------------------------------------

impl<const D: usize, T: TensorElement> Tensor<D, T, CpuBackend> {
    /// Create a tensor from a flat `Vec<T>` with explicit dimensions.
    ///
    /// Returns an error if `data.len()` doesn't match the product of `dims`.
    #[must_use = "returns a Result that may contain an error"]
    pub fn from_vec(dims: [usize; D], data: Vec<T>) -> Result<Self> {
        let expected = checked_dim_product(&dims)?;
        if data.len() != expected {
            return Err(TensorError::DataLengthMismatch {
                expected,
                actual: data.len(),
            });
        }
        let shape: Vec<usize> = dims.to_vec();
        let arr = ArrayD::from_shape_vec(ndarray::IxDyn(&shape), data)?;
        Ok(Self {
            dims,
            storage: Arc::new(arr),
            bounds: None,
        })
    }

    /// Create a tensor from a dynamic ndarray.
    ///
    /// Returns an error if the ndarray's rank doesn't match `D`.
    #[must_use = "returns a Result that may contain an error"]
    pub fn from_ndarray(arr: ArrayD<T>) -> Result<Self> {
        let shape = arr.shape();
        if shape.len() != D {
            return Err(TensorError::RankMismatch {
                expected: D,
                actual: shape.len(),
            });
        }
        let mut dims = [0usize; D];
        dims.copy_from_slice(shape);
        // Validate dimension product won't overflow (consistent with zeros/ones/from_vec).
        checked_dim_product(&dims)?;
        Ok(Self {
            dims,
            storage: Arc::new(arr),
            bounds: None,
        })
    }

    /// Get a reference to the underlying ndarray (CPU only).
    #[must_use]
    pub fn as_ndarray(&self) -> &ArrayD<T> {
        self.storage.as_ref()
    }
}

// -- Trait impls -------------------------------------------------------------------

impl<const D: usize, T: TensorElement, B: Backend> Clone for Tensor<D, T, B> {
    fn clone(&self) -> Self {
        Self {
            dims: self.dims,
            storage: self.storage.clone(),
            bounds: self.bounds.clone(),
        }
    }
}

impl<const D: usize, T: TensorElement, B: Backend> std::fmt::Debug for Tensor<D, T, B> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Tensor")
            .field("dims", &self.dims.as_slice())
            .field("rank", &D)
            .field("dtype", &T::dtype())
            .field("device", &B::device())
            .field("has_bounds", &self.bounds.is_some())
            .finish_non_exhaustive()
    }
}

#[cfg(test)]
mod tests;

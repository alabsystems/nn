// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Typed backing array for float [`DynTensor`] storage.
//!
//! Integer storage (U32, U8, I64) remains in the existing [`TensorStorage::Cpu`]
//! path as `ArrayD<T>` behind `dyn Any`. Float storage uses [`FloatStorage`] to
//! avoid f32 materialization for half-precision types.
//!
//! This module is the foundation for native bf16/f16 DynTensor support (#1646).

use crate::{DType, Result, TensorError};
use half::{bf16, f16};
use ndarray::{ArrayD, ArrayViewD, IxDyn};

/// Typed backing array for float DynTensors.
///
/// Each variant stores a native ndarray with the corresponding element type.
/// CPU operations dispatch via match on the variant. The `half` crate's
/// `num-traits` feature gives ndarray `Zero`, `One`, and basic arithmetic
/// for f16/bf16 (software-emulated via f32 round-trip — acceptable for CPU
/// fallback since GPU is the primary compute path).
#[derive(Clone, Debug)]
pub(crate) enum FloatStorage {
    F32(ArrayD<f32>),
    F16(ArrayD<f16>),
    BF16(ArrayD<bf16>),
}

impl FloatStorage {
    /// Corresponding [`DType`] discriminant.
    #[must_use]
    pub(crate) fn dtype(&self) -> DType {
        match self {
            Self::F32(_) => DType::F32,
            Self::F16(_) => DType::F16,
            Self::BF16(_) => DType::BF16,
        }
    }

    /// Zero-copy view for F32. Returns error for f16/bf16.
    pub(crate) fn as_f32_view(&self) -> Result<ArrayViewD<'_, f32>> {
        match self {
            Self::F32(a) => Ok(a.view()),
            Self::F16(_) => Err(TensorError::dtype_mismatch(DType::F32, DType::F16)),
            Self::BF16(_) => Err(TensorError::dtype_mismatch(DType::F32, DType::BF16)),
        }
    }

    /// Zero-copy view for F16. Returns error for f32/bf16.
    pub(crate) fn as_f16_view(&self) -> Result<ArrayViewD<'_, f16>> {
        match self {
            Self::F16(a) => Ok(a.view()),
            Self::F32(_) => Err(TensorError::dtype_mismatch(DType::F16, DType::F32)),
            Self::BF16(_) => Err(TensorError::dtype_mismatch(DType::F16, DType::BF16)),
        }
    }

    /// Zero-copy view for BF16. Returns error for f32/f16.
    pub(crate) fn as_bf16_view(&self) -> Result<ArrayViewD<'_, bf16>> {
        match self {
            Self::BF16(a) => Ok(a.view()),
            Self::F32(_) => Err(TensorError::dtype_mismatch(DType::BF16, DType::F32)),
            Self::F16(_) => Err(TensorError::dtype_mismatch(DType::BF16, DType::F16)),
        }
    }

    /// Convert to an owned `ArrayD<f32>`. Clones for F32, converts for f16/bf16.
    pub(crate) fn to_f32_array(&self) -> ArrayD<f32> {
        match self {
            Self::F32(a) => a.clone(),
            Self::F16(a) => a.mapv(f16::to_f32),
            Self::BF16(a) => a.mapv(bf16::to_f32),
        }
    }

    /// Create a `FloatStorage` from an `ArrayD<f32>` result, converting to the
    /// target dtype. Used when an operation computes in f32 and the result must
    /// match the original dtype (e.g., matmul, reductions).
    pub(crate) fn from_f32_array(arr: ArrayD<f32>, target: DType) -> Self {
        match target {
            DType::F16 => Self::F16(arr.mapv(f16::from_f32)),
            DType::BF16 => Self::BF16(arr.mapv(bf16::from_f32)),
            DType::F32 | DType::F64 => Self::F32(arr),
            // Non-float dtypes should not reach FloatStorage. Treat as f32 for
            // backward compatibility — callers should validate dtype upstream.
            DType::U32 | DType::U8 | DType::I64 | DType::I32 | DType::Bool => Self::F32(arr),
        }
    }

    /// Create zeros with the given shape and float dtype.
    pub(crate) fn zeros(shape: &[usize], dtype: DType) -> Result<Self> {
        use ndarray::Array;
        match dtype {
            DType::F32 => Ok(Self::F32(Array::zeros(IxDyn(shape)))),
            DType::F16 => Ok(Self::F16(Array::from_elem(IxDyn(shape), f16::ZERO))),
            DType::BF16 => Ok(Self::BF16(Array::from_elem(IxDyn(shape), bf16::ZERO))),
            other => Err(TensorError::Unsupported(format!(
                "FloatStorage::zeros: {other} is not a float dtype"
            ))),
        }
    }

    /// Create ones with the given shape and float dtype.
    pub(crate) fn ones(shape: &[usize], dtype: DType) -> Result<Self> {
        use ndarray::Array;
        match dtype {
            DType::F32 => Ok(Self::F32(Array::ones(IxDyn(shape)))),
            DType::F16 => Ok(Self::F16(Array::from_elem(IxDyn(shape), f16::ONE))),
            DType::BF16 => Ok(Self::BF16(Array::from_elem(IxDyn(shape), bf16::ONE))),
            other => Err(TensorError::Unsupported(format!(
                "FloatStorage::ones: {other} is not a float dtype"
            ))),
        }
    }

    /// In-place element-wise addition: `self += rhs`.
    ///
    /// Requires identical shape and dtype. Modifies `self` directly without
    /// allocating a new array. Used by gradient accumulation to avoid
    /// temporary tensor allocation on every fan-in add.
    pub(crate) fn add_assign(&mut self, rhs: &Self) -> Result<()> {
        match (self, rhs) {
            (Self::F32(a), Self::F32(b)) => {
                if a.shape() != b.shape() {
                    return Err(TensorError::shape_mismatch(
                        a.shape().to_vec(),
                        b.shape().to_vec(),
                    ));
                }
                *a += b;
                Ok(())
            }
            (Self::F16(a), Self::F16(b)) => {
                if a.shape() != b.shape() {
                    return Err(TensorError::shape_mismatch(
                        a.shape().to_vec(),
                        b.shape().to_vec(),
                    ));
                }
                ndarray::Zip::from(a.view_mut())
                    .and(b.view())
                    .for_each(|dst, &src| {
                        *dst = f16::from_f32(dst.to_f32() + src.to_f32());
                    });
                Ok(())
            }
            (Self::BF16(a), Self::BF16(b)) => {
                if a.shape() != b.shape() {
                    return Err(TensorError::shape_mismatch(
                        a.shape().to_vec(),
                        b.shape().to_vec(),
                    ));
                }
                ndarray::Zip::from(a.view_mut())
                    .and(b.view())
                    .for_each(|dst, &src| {
                        *dst = bf16::from_f32(dst.to_f32() + src.to_f32());
                    });
                Ok(())
            }
            (s, o) => Err(TensorError::dtype_mismatch(s.dtype(), o.dtype())),
        }
    }

    /// Create a constant-filled tensor with the given shape, value, and float dtype.
    ///
    /// The f64 value is converted to the target dtype. Returns error if the
    /// value overflows the target type's range (f32, f16, or bf16).
    pub(crate) fn full(shape: &[usize], val: f64, dtype: DType) -> Result<Self> {
        use ndarray::Array;
        match dtype {
            DType::F32 => {
                let val_f32 = super::checked_f64_to_f32(val, "FloatStorage::full()")?;
                Ok(Self::F32(Array::from_elem(IxDyn(shape), val_f32)))
            }
            DType::F16 => {
                let val_f16 = f16::from_f64(val);
                if !val_f16.is_finite() && val.is_finite() {
                    return Err(TensorError::InvalidBounds(format!(
                        "FloatStorage::full(): value {val} overflows f16 (becomes {val_f16})"
                    )));
                }
                Ok(Self::F16(Array::from_elem(IxDyn(shape), val_f16)))
            }
            DType::BF16 => {
                let val_bf16 = bf16::from_f64(val);
                if !val_bf16.is_finite() && val.is_finite() {
                    return Err(TensorError::InvalidBounds(format!(
                        "FloatStorage::full(): value {val} overflows bf16 (becomes {val_bf16})"
                    )));
                }
                Ok(Self::BF16(Array::from_elem(IxDyn(shape), val_bf16)))
            }
            other => Err(TensorError::Unsupported(format!(
                "FloatStorage::full: {other} is not a float dtype"
            ))),
        }
    }
}

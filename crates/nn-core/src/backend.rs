// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Backend abstraction for device-specific tensor storage.
//!
//! The [`Backend`] trait defines the interface for tensor storage primitives.
//! Each backend (CPU, Metal, CUDA, Vulkan) provides its own storage type
//! and allocation methods. The tensor type is generic over the backend,
//! enabling compile-time device dispatch.

use crate::tensor::checked_dim_product;
use crate::{Device, Result, TensorElement};
use ndarray::{ArrayD, IxDyn};
use std::sync::Arc;

/// Backend trait for device-specific tensor storage.
///
/// Backends provide a generic associated storage type and allocation
/// methods. GPU backends will return errors when device resources are
/// unavailable; [`CpuBackend`] allocation is infallible (barring OOM).
pub trait Backend: Clone + Send + Sync + 'static {
    /// Device-specific tensor storage, generic over element type.
    ///
    /// Must be `Send + Sync` so tensors can be shared across threads
    /// (e.g., data loaders, multi-threaded training).
    type TensorPrimitive<T: TensorElement>: Clone + Send + Sync;

    /// The device this backend targets.
    fn device() -> Device;

    /// Allocate zero-filled storage with the given dimensions.
    fn zeros<const D: usize, T: TensorElement>(
        dims: [usize; D],
    ) -> Result<Self::TensorPrimitive<T>>;

    /// Allocate one-filled storage with the given dimensions.
    fn ones<const D: usize, T: TensorElement>(dims: [usize; D])
        -> Result<Self::TensorPrimitive<T>>;
}

/// CPU backend using ndarray for storage.
///
/// This is the default backend. All tensors are stored as `Arc<ArrayD<T>>`
/// for shared-ownership CPU arrays.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct CpuBackend;

impl Backend for CpuBackend {
    type TensorPrimitive<T: TensorElement> = Arc<ArrayD<T>>;

    fn device() -> Device {
        Device::Cpu
    }

    fn zeros<const D: usize, T: TensorElement>(
        dims: [usize; D],
    ) -> Result<Self::TensorPrimitive<T>> {
        checked_dim_product(&dims)?;
        let arr = ArrayD::zeros(IxDyn(&dims));
        Ok(Arc::new(arr))
    }

    fn ones<const D: usize, T: TensorElement>(
        dims: [usize; D],
    ) -> Result<Self::TensorPrimitive<T>> {
        checked_dim_product(&dims)?;
        let arr = ArrayD::ones(IxDyn(&dims));
        Ok(Arc::new(arr))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cpu_backend_device() {
        assert_eq!(CpuBackend::device(), Device::Cpu);
    }

    #[test]
    fn test_cpu_backend_zeros_f32() {
        let storage = CpuBackend::zeros::<2, f32>([3, 4]).expect("CPU allocation");
        assert_eq!(storage.shape(), &[3, 4]);
        assert!(storage.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn test_cpu_backend_ones_f32() {
        let storage = CpuBackend::ones::<2, f32>([3, 4]).expect("CPU allocation");
        assert_eq!(storage.shape(), &[3, 4]);
        assert!(storage.iter().all(|&v| v == 1.0));
    }

    #[test]
    fn test_cpu_backend_zeros_i32() {
        let storage = CpuBackend::zeros::<1, i32>([5]).expect("CPU allocation");
        assert_eq!(storage.shape(), &[5]);
        assert!(storage.iter().all(|&v| v == 0));
    }

    #[test]
    fn test_cpu_backend_scalar() {
        let storage = CpuBackend::zeros::<0, f32>([]).expect("CPU allocation");
        assert_eq!(storage.shape(), &[] as &[usize]);
        assert_eq!(storage.len(), 1); // scalar has 1 element
    }

    #[test]
    fn test_cpu_backend_zeros_overflow_returns_error() {
        let result = CpuBackend::zeros::<2, f32>([usize::MAX, 2]);
        assert!(
            result.is_err(),
            "zeros with overflowing dims should return error, not panic"
        );
    }

    #[test]
    fn test_cpu_backend_ones_overflow_returns_error() {
        let result = CpuBackend::ones::<2, f32>([usize::MAX, 2]);
        assert!(
            result.is_err(),
            "ones with overflowing dims should return error, not panic"
        );
    }
}

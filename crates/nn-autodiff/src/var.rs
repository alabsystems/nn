// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Trainable variable (Var) type for automatic differentiation.
//!
//! `Var` wraps a `DynTensor` with interior mutability (for optimizer weight updates)
//! and a unique ID (for gradient accumulation). Matches candle's `Var` concept.

use crate::error::Result;
use nn_core::dyn_tensor::DynTensor;
use nn_core::{DType, Device};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, RwLock};

/// Unique identifier for each trainable variable.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct VarId(u64);

static NEXT_VAR_ID: AtomicU64 = AtomicU64::new(0);

impl VarId {
    fn next() -> Self {
        Self(NEXT_VAR_ID.fetch_add(1, Ordering::Relaxed))
    }
}

/// Trainable variable wrapping a `DynTensor`.
///
/// Like candle's `Var`: supports in-place weight updates from optimizers
/// and carries a unique ID for gradient accumulation in `GradStore`.
///
/// # Interior mutability
///
/// `Var::set()` replaces the underlying tensor data (used by optimizers).
/// The `RwLock` allows multiple readers during forward pass and exclusive
/// write access during optimizer step.
///
/// # Example
/// ```no_run
/// use nn_autodiff::Var;
/// use nn_core::{DType, Device};
///
/// # fn main() -> std::result::Result<(), nn_autodiff::AutodiffError> {
/// let var = Var::zeros(&[3, 4], DType::F32, &Device::Cpu)?;
/// let data = var.data()?;
/// assert_eq!(data.dims(), &[3, 4]);
/// # Ok(())
/// # }
/// ```
#[derive(Clone)]
pub struct Var {
    id: VarId,
    data: Arc<RwLock<DynTensor>>,
}

impl Var {
    /// Create a new trainable variable from an existing tensor.
    pub fn new(data: DynTensor) -> Self {
        Self {
            id: VarId::next(),
            data: Arc::new(RwLock::new(data)),
        }
    }

    /// Create a zero-initialized trainable variable.
    pub fn zeros(dims: &[usize], dtype: DType, device: &Device) -> Result<Self> {
        Ok(Self::new(DynTensor::zeros(dims, dtype, device)?))
    }

    /// Create a trainable variable from a tensor (clones the data).
    pub fn from_tensor(t: &DynTensor) -> Self {
        Self::new(t.clone())
    }

    /// Get current tensor data (clones the underlying tensor).
    ///
    /// Returns an error if the RwLock is poisoned (a thread panicked while
    /// holding it).
    pub fn data(&self) -> Result<DynTensor> {
        Ok(self
            .data
            .read()
            .map_err(|_| crate::AutodiffError::LockPoisoned {
                context: "Var::data() read",
            })?
            .clone())
    }

    /// Replace tensor data in-place (used by optimizers).
    ///
    /// Returns an error if the new data has a different shape or if the
    /// RwLock is poisoned.
    pub fn set(&self, new_data: &DynTensor) -> Result<()> {
        let mut guard = self
            .data
            .write()
            .map_err(|_| crate::AutodiffError::LockPoisoned {
                context: "Var::set() write",
            })?;
        if guard.dims() != new_data.dims() {
            return Err(nn_core::TensorError::shape_mismatch(
                guard.dims().to_vec(),
                new_data.dims().to_vec(),
            )
            .into());
        }
        if guard.dtype() != new_data.dtype() {
            return Err(
                nn_core::TensorError::dtype_mismatch(guard.dtype(), new_data.dtype()).into(),
            );
        }
        *guard = new_data.clone();
        Ok(())
    }

    /// Unique identifier for this variable.
    #[must_use]
    pub fn id(&self) -> VarId {
        self.id
    }

    /// Shape of the underlying tensor.
    ///
    /// Returns an error if the RwLock is poisoned.
    pub fn dims(&self) -> Result<Vec<usize>> {
        Ok(self
            .data
            .read()
            .map_err(|_| crate::AutodiffError::LockPoisoned {
                context: "Var::dims() read",
            })?
            .dims()
            .to_vec())
    }

    /// Data type of the underlying tensor.
    ///
    /// Returns an error if the RwLock is poisoned.
    pub fn dtype(&self) -> Result<DType> {
        Ok(self
            .data
            .read()
            .map_err(|_| crate::AutodiffError::LockPoisoned {
                context: "Var::dtype() read",
            })?
            .dtype())
    }

    /// Device of the underlying tensor.
    ///
    /// Returns an error if the RwLock is poisoned.
    pub fn device(&self) -> Result<Device> {
        Ok(self
            .data
            .read()
            .map_err(|_| crate::AutodiffError::LockPoisoned {
                context: "Var::device() read",
            })?
            .device())
    }
}

impl std::fmt::Debug for Var {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut s = f.debug_struct("Var");
        s.field("id", &self.id);
        match self.dims() {
            Ok(dims) => {
                s.field("dims", &dims);
            }
            Err(_) => {
                s.field("dims", &"<lock poisoned>");
            }
        }
        match self.dtype() {
            Ok(dtype) => {
                s.field("dtype", &dtype);
            }
            Err(_) => {
                s.field("dtype", &"<lock poisoned>");
            }
        }
        s.finish_non_exhaustive()
    }
}

#[cfg(test)]
#[path = "var_tests.rs"]
mod tests;

// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! `Dim` trait for dimension resolution — accepts `usize`, [`D`], and `i32`.
//!
//! All DynTensor methods that take a dimension parameter accept `impl Dim`,
//! so callers can pass `0usize`, `D::Minus1`, or `-1i32` (PyTorch-style).

use super::D;
use crate::{check_dim, Result, TensorError};

/// Trait for dimension resolution. Accepts both `usize`, [`D`], and `i32`.
pub trait Dim {
    /// Resolve to a concrete dimension index given the tensor rank.
    fn to_index(&self, rank: usize) -> Result<usize>;
}

impl Dim for usize {
    fn to_index(&self, rank: usize) -> Result<usize> {
        check_dim(*self, rank)?;
        Ok(*self)
    }
}

impl Dim for D {
    fn to_index(&self, rank: usize) -> Result<usize> {
        self.resolve(rank)
    }
}

impl Dim for i32 {
    fn to_index(&self, rank: usize) -> Result<usize> {
        if *self >= 0 {
            let d = *self as usize;
            check_dim(d, rank)?;
            Ok(d)
        } else {
            let neg = self.unsigned_abs() as usize;
            if neg > rank {
                return Err(TensorError::DimensionOutOfRange { dim: neg, rank });
            }
            Ok(rank - neg)
        }
    }
}

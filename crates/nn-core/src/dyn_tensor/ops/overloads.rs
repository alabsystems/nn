// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Operator overloads for [`DynTensor`] (`+`, `-`, `*`, `/` for references and scalars).
//!
//! candle convention: `Output = Result<DynTensor>`.
//!
//! **Important:** All operator overloads delegate to the **broadcast** variants
//! (`broadcast_add`, `broadcast_sub`, etc.), matching candle's behavior. This
//! means `(&a + &b)?` uses NumPy-style broadcasting and succeeds on compatible
//! but different shapes. The named methods (`.add()`, `.sub()`) require exact
//! shape match. See the ops module docs for details.

use super::DynTensor;
use crate::Result;

impl std::ops::Add for &DynTensor {
    type Output = Result<DynTensor>;
    fn add(self, rhs: &DynTensor) -> Self::Output {
        self.broadcast_add(rhs)
    }
}

impl std::ops::Sub for &DynTensor {
    type Output = Result<DynTensor>;
    fn sub(self, rhs: &DynTensor) -> Self::Output {
        self.broadcast_sub(rhs)
    }
}

impl std::ops::Mul for &DynTensor {
    type Output = Result<DynTensor>;
    fn mul(self, rhs: &DynTensor) -> Self::Output {
        self.broadcast_mul(rhs)
    }
}

impl std::ops::Div for &DynTensor {
    type Output = Result<DynTensor>;
    fn div(self, rhs: &DynTensor) -> Self::Output {
        self.broadcast_div(rhs)
    }
}

impl std::ops::Add<f64> for &DynTensor {
    type Output = Result<DynTensor>;
    fn add(self, rhs: f64) -> Self::Output {
        self.add_scalar(rhs)
    }
}

impl std::ops::Mul<f64> for &DynTensor {
    type Output = Result<DynTensor>;
    fn mul(self, rhs: f64) -> Self::Output {
        self.mul_scalar(rhs)
    }
}

impl std::ops::Sub<f64> for &DynTensor {
    type Output = Result<DynTensor>;
    fn sub(self, rhs: f64) -> Self::Output {
        self.sub_scalar(rhs)
    }
}

impl std::ops::Div<f64> for &DynTensor {
    type Output = Result<DynTensor>;
    fn div(self, rhs: f64) -> Self::Output {
        self.div_scalar(rhs)
    }
}

impl std::ops::Neg for &DynTensor {
    type Output = Result<DynTensor>;
    fn neg(self) -> Self::Output {
        DynTensor::neg(self)
    }
}

// --- Owned × Owned ---

impl std::ops::Add for DynTensor {
    type Output = Result<Self>;
    fn add(self, rhs: Self) -> Self::Output {
        self.broadcast_add(&rhs)
    }
}

impl std::ops::Sub for DynTensor {
    type Output = Result<Self>;
    fn sub(self, rhs: Self) -> Self::Output {
        self.broadcast_sub(&rhs)
    }
}

impl std::ops::Mul for DynTensor {
    type Output = Result<Self>;
    fn mul(self, rhs: Self) -> Self::Output {
        self.broadcast_mul(&rhs)
    }
}

impl std::ops::Div for DynTensor {
    type Output = Result<Self>;
    fn div(self, rhs: Self) -> Self::Output {
        self.broadcast_div(&rhs)
    }
}

// --- Ref × Owned ---

impl std::ops::Add<DynTensor> for &DynTensor {
    type Output = Result<DynTensor>;
    fn add(self, rhs: DynTensor) -> Self::Output {
        self.broadcast_add(&rhs)
    }
}

impl std::ops::Sub<DynTensor> for &DynTensor {
    type Output = Result<DynTensor>;
    fn sub(self, rhs: DynTensor) -> Self::Output {
        self.broadcast_sub(&rhs)
    }
}

impl std::ops::Mul<DynTensor> for &DynTensor {
    type Output = Result<DynTensor>;
    fn mul(self, rhs: DynTensor) -> Self::Output {
        self.broadcast_mul(&rhs)
    }
}

impl std::ops::Div<DynTensor> for &DynTensor {
    type Output = Result<DynTensor>;
    fn div(self, rhs: DynTensor) -> Self::Output {
        self.broadcast_div(&rhs)
    }
}

// --- Owned × Ref ---

impl std::ops::Add<&Self> for DynTensor {
    type Output = Result<Self>;
    fn add(self, rhs: &Self) -> Self::Output {
        self.broadcast_add(rhs)
    }
}

impl std::ops::Sub<&Self> for DynTensor {
    type Output = Result<Self>;
    fn sub(self, rhs: &Self) -> Self::Output {
        self.broadcast_sub(rhs)
    }
}

impl std::ops::Mul<&Self> for DynTensor {
    type Output = Result<Self>;
    fn mul(self, rhs: &Self) -> Self::Output {
        self.broadcast_mul(rhs)
    }
}

impl std::ops::Div<&Self> for DynTensor {
    type Output = Result<Self>;
    fn div(self, rhs: &Self) -> Self::Output {
        self.broadcast_div(rhs)
    }
}

// --- Owned × Scalar ---

impl std::ops::Add<f64> for DynTensor {
    type Output = Result<Self>;
    fn add(self, rhs: f64) -> Self::Output {
        self.add_scalar(rhs)
    }
}

impl std::ops::Sub<f64> for DynTensor {
    type Output = Result<Self>;
    fn sub(self, rhs: f64) -> Self::Output {
        self.sub_scalar(rhs)
    }
}

impl std::ops::Mul<f64> for DynTensor {
    type Output = Result<Self>;
    fn mul(self, rhs: f64) -> Self::Output {
        self.mul_scalar(rhs)
    }
}

impl std::ops::Div<f64> for DynTensor {
    type Output = Result<Self>;
    fn div(self, rhs: f64) -> Self::Output {
        self.div_scalar(rhs)
    }
}

// --- Scalar × Tensor (reverse) ---

impl std::ops::Add<DynTensor> for f64 {
    type Output = Result<DynTensor>;
    fn add(self, rhs: DynTensor) -> Self::Output {
        rhs.add_scalar(self)
    }
}

impl std::ops::Add<&DynTensor> for f64 {
    type Output = Result<DynTensor>;
    fn add(self, rhs: &DynTensor) -> Self::Output {
        rhs.add_scalar(self)
    }
}

impl std::ops::Mul<DynTensor> for f64 {
    type Output = Result<DynTensor>;
    fn mul(self, rhs: DynTensor) -> Self::Output {
        rhs.mul_scalar(self)
    }
}

impl std::ops::Mul<&DynTensor> for f64 {
    type Output = Result<DynTensor>;
    fn mul(self, rhs: &DynTensor) -> Self::Output {
        rhs.mul_scalar(self)
    }
}

impl std::ops::Sub<DynTensor> for f64 {
    type Output = Result<DynTensor>;
    fn sub(self, rhs: DynTensor) -> Self::Output {
        rhs.neg()?.add_scalar(self)
    }
}

impl std::ops::Sub<&DynTensor> for f64 {
    type Output = Result<DynTensor>;
    fn sub(self, rhs: &DynTensor) -> Self::Output {
        DynTensor::neg(rhs)?.add_scalar(self)
    }
}

impl std::ops::Div<DynTensor> for f64 {
    type Output = Result<DynTensor>;
    fn div(self, rhs: DynTensor) -> Self::Output {
        rhs.recip()?.mul_scalar(self)
    }
}

impl std::ops::Div<&DynTensor> for f64 {
    type Output = Result<DynTensor>;
    fn div(self, rhs: &DynTensor) -> Self::Output {
        rhs.recip()?.mul_scalar(self)
    }
}

// --- Owned Neg ---

impl std::ops::Neg for DynTensor {
    type Output = Result<Self>;
    fn neg(self) -> Self::Output {
        Self::neg(&self)
    }
}

// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Type conversions and Display impls for kernel IR types.
//!
//! Extracted from `ir.rs` to keep it under the 500-line limit (#557).

use super::{BinOpKind, CompareOpKind, IRError, ScalarType, UnaryFnKind, ValueType};

impl From<ScalarType> for nn_core::DType {
    fn from(st: ScalarType) -> Self {
        match st {
            ScalarType::F32 => Self::F32,
            ScalarType::F16 => Self::F16,
            ScalarType::BF16 => Self::BF16,
        }
    }
}

impl TryFrom<nn_core::DType> for ScalarType {
    type Error = IRError;
    fn try_from(dt: nn_core::DType) -> Result<Self, IRError> {
        match dt {
            nn_core::DType::F32 => Ok(Self::F32),
            nn_core::DType::F16 => Ok(Self::F16),
            nn_core::DType::BF16 => Ok(Self::BF16),
            other => Err(IRError::UnsupportedType(format!(
                "DType::{other} has no ScalarType equivalent"
            ))),
        }
    }
}

impl TryFrom<nn_core::DType> for ValueType {
    type Error = IRError;
    fn try_from(dt: nn_core::DType) -> Result<Self, IRError> {
        match dt {
            nn_core::DType::F32 => Ok(Self::F32),
            nn_core::DType::F16 => Ok(Self::F16),
            nn_core::DType::BF16 => Ok(Self::BF16),
            nn_core::DType::Bool => Ok(Self::Bool),
            other => Err(IRError::UnsupportedType(format!(
                "DType::{other} has no ValueType equivalent"
            ))),
        }
    }
}

impl std::fmt::Display for ScalarType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.type_name())
    }
}

impl std::fmt::Display for BinOpKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Add => write!(f, "+"),
            Self::Sub => write!(f, "-"),
            Self::Mul => write!(f, "*"),
            Self::Div => write!(f, "/"),
        }
    }
}

impl std::fmt::Display for CompareOpKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Eq => write!(f, "=="),
            Self::Ne => write!(f, "!="),
            Self::Lt => write!(f, "<"),
            Self::Le => write!(f, "<="),
            Self::Gt => write!(f, ">"),
            Self::Ge => write!(f, ">="),
        }
    }
}

impl std::fmt::Display for UnaryFnKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.method_name())
    }
}

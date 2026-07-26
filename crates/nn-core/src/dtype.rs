// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Data types for nn

/// Tensor data types
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum DType {
    /// 32-bit float
    F32,
    /// 16-bit float (IEEE 754)
    F16,
    /// 16-bit bfloat
    BF16,
    /// 64-bit float
    F64,
    /// 32-bit signed integer
    I32,
    /// 64-bit signed integer
    I64,
    /// 32-bit unsigned integer
    U32,
    /// 8-bit unsigned integer
    U8,
    /// Boolean
    Bool,
}

impl DType {
    /// Size in bytes
    #[must_use]
    pub fn size_bytes(&self) -> usize {
        match self {
            Self::F32 => 4,
            Self::F16 => 2,
            Self::BF16 => 2,
            Self::F64 => 8,
            Self::I32 => 4,
            Self::I64 => 8,
            Self::U32 => 4,
            Self::U8 => 1,
            Self::Bool => 1,
        }
    }

    /// Check if floating point
    #[must_use]
    pub fn is_float(&self) -> bool {
        match self {
            Self::F32 | Self::F16 | Self::BF16 | Self::F64 => true,
            Self::I32 | Self::I64 | Self::U32 | Self::U8 | Self::Bool => false,
        }
    }

    /// Check if integer
    #[must_use]
    pub fn is_int(&self) -> bool {
        match self {
            Self::I32 | Self::I64 | Self::U32 | Self::U8 => true,
            Self::F32 | Self::F16 | Self::BF16 | Self::F64 | Self::Bool => false,
        }
    }
}

impl std::fmt::Display for DType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::F32 => write!(f, "f32"),
            Self::F16 => write!(f, "f16"),
            Self::BF16 => write!(f, "bf16"),
            Self::F64 => write!(f, "f64"),
            Self::I32 => write!(f, "i32"),
            Self::I64 => write!(f, "i64"),
            Self::U32 => write!(f, "u32"),
            Self::U8 => write!(f, "u8"),
            Self::Bool => write!(f, "bool"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_size_bytes_all_variants() {
        assert_eq!(DType::F32.size_bytes(), 4);
        assert_eq!(DType::F16.size_bytes(), 2);
        assert_eq!(DType::BF16.size_bytes(), 2);
        assert_eq!(DType::F64.size_bytes(), 8);
        assert_eq!(DType::I32.size_bytes(), 4);
        assert_eq!(DType::I64.size_bytes(), 8);
        assert_eq!(DType::U32.size_bytes(), 4);
        assert_eq!(DType::U8.size_bytes(), 1);
        assert_eq!(DType::Bool.size_bytes(), 1);
    }

    #[test]
    fn test_size_bytes_all_nonzero() {
        let all = [
            DType::F32,
            DType::F16,
            DType::BF16,
            DType::F64,
            DType::I32,
            DType::I64,
            DType::U32,
            DType::U8,
            DType::Bool,
        ];
        for dt in all {
            assert!(dt.size_bytes() > 0, "{dt} should have nonzero size");
        }
    }

    #[test]
    fn test_is_float() {
        assert!(DType::F32.is_float());
        assert!(DType::F16.is_float());
        assert!(DType::BF16.is_float());
        assert!(DType::F64.is_float());
        assert!(!DType::I32.is_float());
        assert!(!DType::I64.is_float());
        assert!(!DType::U32.is_float());
        assert!(!DType::U8.is_float());
        assert!(!DType::Bool.is_float());
    }

    #[test]
    fn test_is_int() {
        assert!(!DType::F32.is_int());
        assert!(!DType::F16.is_int());
        assert!(!DType::BF16.is_int());
        assert!(!DType::F64.is_int());
        assert!(DType::I32.is_int());
        assert!(DType::I64.is_int());
        assert!(DType::U32.is_int());
        assert!(DType::U8.is_int());
        assert!(!DType::Bool.is_int());
    }

    #[test]
    fn test_bool_is_neither_float_nor_int() {
        assert!(!DType::Bool.is_float());
        assert!(!DType::Bool.is_int());
    }

    #[test]
    fn test_display_all_variants() {
        assert_eq!(format!("{}", DType::F32), "f32");
        assert_eq!(format!("{}", DType::F16), "f16");
        assert_eq!(format!("{}", DType::BF16), "bf16");
        assert_eq!(format!("{}", DType::F64), "f64");
        assert_eq!(format!("{}", DType::I32), "i32");
        assert_eq!(format!("{}", DType::I64), "i64");
        assert_eq!(format!("{}", DType::U32), "u32");
        assert_eq!(format!("{}", DType::U8), "u8");
        assert_eq!(format!("{}", DType::Bool), "bool");
    }
}

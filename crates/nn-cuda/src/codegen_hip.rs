// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! HIP codegen helper utilities: type mapping, prelude, and formatting.
//!
//! Parallel to `nn-dsl::codegen_msl_helpers` — adapted for HIP/ROCm conventions.

use nn_dsl::ScalarType;

/// HIP C++ prelude included at the top of every generated kernel file.
pub const HIP_PRELUDE: &str = "\
#include <hip/hip_runtime.h>\n\
#include <hip/hip_fp16.h>\n\
#include <hip/hip_bfloat16.h>\n\n";

/// Map `ScalarType` to HIP C++ type name.
///
/// AMD HIP distinguishes `half` (IEEE fp16) from `hip_bfloat16` (bf16).
/// Metal maps both F16 and BF16 to `half` because Metal lacks native bf16,
/// but HIP has proper bf16 support via `hip_bfloat16` from `hip/hip_bfloat16.h`.
pub fn hip_type(dtype: ScalarType) -> Result<&'static str, crate::HipCodegenError> {
    match dtype {
        ScalarType::F32 => Ok("float"),
        ScalarType::F16 => Ok("half"),
        ScalarType::BF16 => Ok("hip_bfloat16"),
        _ => Err(crate::HipCodegenError::UnsupportedIRVariant {
            variant_desc: "ScalarType",
        }),
    }
}

/// Accumulator type for dot products — always f32 to avoid precision loss.
pub fn hip_accumulator_type(_dtype: ScalarType) -> &'static str {
    "float"
}

/// Validate that a `usize` fits in a 32-bit unsigned int and return as string.
///
/// HIP kernels use `unsigned int` for indexing (32-bit). Values exceeding
/// `u32::MAX` cannot be represented.
pub fn safe_hip_uint(val: usize) -> Result<String, crate::HipCodegenError> {
    if val > u32::MAX as usize {
        return Err(crate::HipCodegenError::ShapeProductOverflow { shape: vec![val] });
    }
    Ok(val.to_string())
}

/// Default thread block size for elementwise kernels.
pub const HIP_BLOCK_SIZE: usize = 256;

/// Default thread block size for reduction kernels (matches MSL threadgroup size).
pub const REDUCE_BLOCK_SIZE: usize = 256;

/// Format a float literal for HIP C++ source (avoids scientific notation issues).
pub fn format_float(val: f32) -> String {
    if val == f32::INFINITY {
        "HUGE_VALF".to_string()
    } else if val == f32::NEG_INFINITY {
        "(-HUGE_VALF)".to_string()
    } else if val.is_nan() {
        "nanf(\"\")".to_string()
    } else {
        // Use enough precision to round-trip f32.
        format!("{val:.8}")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hip_type_mapping() {
        assert_eq!(hip_type(ScalarType::F32).unwrap(), "float");
        assert_eq!(hip_type(ScalarType::F16).unwrap(), "half");
        assert_eq!(hip_type(ScalarType::BF16).unwrap(), "hip_bfloat16");
    }

    #[test]
    fn test_safe_hip_uint_valid() {
        assert_eq!(safe_hip_uint(0).unwrap(), "0");
        assert_eq!(safe_hip_uint(1024).unwrap(), "1024");
        assert_eq!(safe_hip_uint(u32::MAX as usize).unwrap(), "4294967295");
    }

    #[test]
    fn test_safe_hip_uint_overflow() {
        let result = safe_hip_uint(u32::MAX as usize + 1);
        assert!(result.is_err());
    }

    #[test]
    fn test_format_float() {
        assert_eq!(format_float(f32::INFINITY), "HUGE_VALF");
        assert_eq!(format_float(f32::NEG_INFINITY), "(-HUGE_VALF)");
        assert!(format_float(1.5).contains("1.5"));
    }
}

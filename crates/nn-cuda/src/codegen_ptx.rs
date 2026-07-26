// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! PTX codegen helper utilities: type mapping, prelude, and formatting.
//!
//! Parallel to [`codegen_hip`](super::codegen_hip) — adapted for NVIDIA PTX ISA conventions.
//! PTX (Parallel Thread Execution) is NVIDIA's virtual ISA that compiles to
//! SASS machine code via `ptxas` or the CUDA driver's JIT compiler.

use nn_dsl::ScalarType;

/// PTX version target (6.5 supports sm_70+, tensor cores on Volta/Ampere/Hopper).
pub const PTX_VERSION: &str = "6.5";

/// Default SM target for PTX compilation.
pub const DEFAULT_SM_TARGET: &str = "sm_80";

/// PTX prelude included at the top of every generated PTX kernel.
///
/// Declares PTX version and target architecture. The `.address_size 64`
/// directive enables 64-bit addressing for buffer pointers.
pub fn ptx_prelude(sm_target: &str) -> String {
    format!(
        ".version {PTX_VERSION}\n\
         .target {sm_target}\n\
         .address_size 64\n\n"
    )
}

/// Map `ScalarType` to PTX register type.
///
/// PTX uses `.f32` for 32-bit float, `.f16` for IEEE fp16, `.b16` for bf16
/// (bf16 is treated as a 16-bit bitfield with special conversion instructions).
pub fn ptx_type(dtype: ScalarType) -> Result<&'static str, PtxCodegenError> {
    match dtype {
        ScalarType::F32 => Ok(".f32"),
        ScalarType::F16 => Ok(".f16"),
        ScalarType::BF16 => Ok(".b16"),
        _ => Err(PtxCodegenError::UnsupportedType {
            type_desc: "non-float ScalarType",
        }),
    }
}

/// Map `ScalarType` to PTX register size suffix for `mov`, `ld`, `st`.
///
/// bf16 is stored as 16-bit values but loaded/stored with `.b16` (bitwise).
pub fn ptx_reg_type(dtype: ScalarType) -> Result<&'static str, PtxCodegenError> {
    match dtype {
        ScalarType::F32 => Ok(".f32"),
        ScalarType::F16 => Ok(".f16"),
        ScalarType::BF16 => Ok(".b16"),
        _ => Err(PtxCodegenError::UnsupportedType {
            type_desc: "non-float ScalarType",
        }),
    }
}

/// Map `ScalarType` to byte size.
pub fn ptx_type_bytes(dtype: ScalarType) -> Result<usize, PtxCodegenError> {
    match dtype {
        ScalarType::F32 => Ok(4),
        ScalarType::F16 | ScalarType::BF16 => Ok(2),
        _ => Err(PtxCodegenError::UnsupportedType {
            type_desc: "non-float ScalarType",
        }),
    }
}

/// Map `ScalarType` to CUDA C++ type name (for host-side wrappers).
pub fn cuda_type(dtype: ScalarType) -> Result<&'static str, PtxCodegenError> {
    match dtype {
        ScalarType::F32 => Ok("float"),
        ScalarType::F16 => Ok("__half"),
        ScalarType::BF16 => Ok("__nv_bfloat16"),
        _ => Err(PtxCodegenError::UnsupportedType {
            type_desc: "non-float ScalarType",
        }),
    }
}

/// Accumulator type for dot products in PTX — always f32 for precision.
pub fn ptx_accumulator_type(_dtype: ScalarType) -> &'static str {
    ".f32"
}

/// Default thread block size for elementwise kernels (256 threads).
pub const PTX_BLOCK_SIZE: usize = 256;

/// Default thread block size for reduction kernels.
pub const REDUCE_BLOCK_SIZE: usize = 256;

/// Warp size on all NVIDIA GPUs.
pub const WARP_SIZE: usize = 32;

/// Format a float literal for PTX assembly.
///
/// PTX uses IEEE 754 hex float representation for exact constants.
/// This avoids decimal-to-binary conversion imprecision.
pub fn format_ptx_float(val: f32) -> String {
    if val == f32::INFINITY {
        "0x7F800000".to_string() // +inf in IEEE 754
    } else if val == f32::NEG_INFINITY {
        "0xFF800000".to_string() // -inf in IEEE 754
    } else if val.is_nan() {
        "0x7FC00000".to_string() // quiet NaN
    } else {
        // PTX accepts IEEE 754 hex float: 0fXXXXXXXX
        format!("0f{:08X}", val.to_bits())
    }
}

/// Validate that a `usize` fits in a 32-bit unsigned int.
///
/// PTX kernels use 32-bit indexing for thread IDs and grid dimensions.
pub fn safe_ptx_uint(val: usize) -> Result<String, PtxCodegenError> {
    if val > u32::MAX as usize {
        return Err(PtxCodegenError::ValueExceedsU32 {
            value: val,
            max: u32::MAX,
        });
    }
    Ok(val.to_string())
}

/// Errors from PTX code generation.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum PtxCodegenError {
    #[error("unsupported type for PTX codegen: {type_desc}")]
    UnsupportedType { type_desc: &'static str },

    #[error("value {value} exceeds u32::MAX ({max}) for PTX 32-bit indexing")]
    ValueExceedsU32 { value: usize, max: u32 },

    #[error("shape product overflow: {shape:?}")]
    ShapeProductOverflow { shape: Vec<usize> },

    #[error("unsupported dispatch step for PTX codegen: {step_name}")]
    UnsupportedStep { step_name: &'static str },

    #[error("invalid parameter: {0}")]
    InvalidParameter(String),

    #[error("axis {axis} out of bounds for shape with rank {rank}")]
    AxisOutOfBounds { axis: usize, rank: usize },
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ptx_type_mapping() {
        assert_eq!(ptx_type(ScalarType::F32).unwrap(), ".f32");
        assert_eq!(ptx_type(ScalarType::F16).unwrap(), ".f16");
        assert_eq!(ptx_type(ScalarType::BF16).unwrap(), ".b16");
    }

    #[test]
    fn test_cuda_type_mapping() {
        assert_eq!(cuda_type(ScalarType::F32).unwrap(), "float");
        assert_eq!(cuda_type(ScalarType::F16).unwrap(), "__half");
        assert_eq!(cuda_type(ScalarType::BF16).unwrap(), "__nv_bfloat16");
    }

    #[test]
    fn test_ptx_type_bytes() {
        assert_eq!(ptx_type_bytes(ScalarType::F32).unwrap(), 4);
        assert_eq!(ptx_type_bytes(ScalarType::F16).unwrap(), 2);
        assert_eq!(ptx_type_bytes(ScalarType::BF16).unwrap(), 2);
    }

    #[test]
    fn test_ptx_prelude() {
        let prelude = ptx_prelude("sm_80");
        assert!(prelude.contains(".version 6.5"));
        assert!(prelude.contains(".target sm_80"));
        assert!(prelude.contains(".address_size 64"));
    }

    #[test]
    fn test_format_ptx_float() {
        assert_eq!(format_ptx_float(f32::INFINITY), "0x7F800000");
        assert_eq!(format_ptx_float(f32::NEG_INFINITY), "0xFF800000");
        // 1.0f32 = 0x3F800000
        assert_eq!(format_ptx_float(1.0), "0f3F800000");
        // 0.0f32 = 0x00000000
        assert_eq!(format_ptx_float(0.0), "0f00000000");
    }

    #[test]
    fn test_safe_ptx_uint_valid() {
        assert_eq!(safe_ptx_uint(0).unwrap(), "0");
        assert_eq!(safe_ptx_uint(1024).unwrap(), "1024");
        assert_eq!(safe_ptx_uint(u32::MAX as usize).unwrap(), "4294967295");
    }

    #[test]
    fn test_safe_ptx_uint_overflow() {
        let result = safe_ptx_uint(u32::MAX as usize + 1);
        assert!(result.is_err());
    }
}

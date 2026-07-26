// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Dtype conversion operations for [`DynTensor`].
//!
//! Extracted from `compare.rs` for file-size compliance.
//! Contains `to_dtype` and validated type-conversion helpers.

use crate::dyn_tensor::trace::{self, TraceOp};
use crate::dyn_tensor::DynTensor;
use crate::{DType, Device, Result, TensorError};

/// Check whether two float dtypes share the same GPU buffer byte width.
///
/// BF16 and F16 both map to 2-byte Metal `half` buffers (#1646 D7/D8).
/// F32 (and F64, stored as f32) uses 4-byte buffers.
/// Only same-width conversions can use zero-copy `gpu_relabel_dtype`;
/// cross-width conversions must round-trip through CPU.
fn same_gpu_byte_width(a: DType, b: DType) -> bool {
    gpu_float_bytes(a) == gpu_float_bytes(b)
}

fn gpu_float_bytes(dt: DType) -> usize {
    match dt {
        DType::BF16 | DType::F16 => 2,
        DType::F32 | DType::F64 => 4,
        // Non-float dtypes: return sentinel that never matches, forcing CPU path.
        // Exhaustive listing per #1409 — adding a new float DType triggers compile error.
        DType::I32 | DType::I64 | DType::U32 | DType::U8 | DType::Bool => 0,
    }
}

impl DynTensor {
    /// Convert tensor to a different dtype. GPU tensors auto-transfer for conversion.
    ///
    /// Float→float conversions produce native storage: F32→BF16 creates `FloatStorage::BF16`,
    /// F32→F16 creates `FloatStorage::F16`, BF16/F16→F32 converts to `FloatStorage::F32`.
    ///
    /// Integer conversions (F32↔U32, F32↔U8, F32↔I64, etc.) produce correctly-typed storage.
    /// BF16/F16 → integer conversions go through an f32 intermediate.
    pub fn to_dtype(&self, dtype: DType) -> Result<Self> {
        if dtype == self.dtype {
            return Ok(self.clone());
        }
        if self.device().is_gpu() {
            // BF16↔F16 share 2-byte Metal buffers: zero-copy relabel is safe.
            // F32↔BF16 and F32↔F16 cross byte widths (4-byte vs 2-byte):
            // relabeling would cause dispatch to misinterpret buffer data.
            // Route through CPU for actual data conversion.
            if self.dtype.is_float() && dtype.is_float() && same_gpu_byte_width(self.dtype, dtype) {
                let mut result = self.gpu_relabel_dtype(dtype)?;
                self.record_to_dtype_trace(dtype, &mut result)?;
                return Ok(result);
            }
            // Try GPU-native dtype cast (avoids GPU→CPU→GPU round-trip).
            if self.dtype.is_float() && dtype.is_float() {
                if let Some(result) =
                    crate::dyn_tensor::gpu::gpu_backend_dispatch(|b| b.cast_dtype(self, dtype))
                {
                    let mut result = result?;
                    self.record_to_dtype_trace(dtype, &mut result)?;
                    return Ok(result);
                }
            }
            // Fallback: round-trip through CPU for conversion.
            let cpu = self.to_device(&Device::Cpu)?;
            let mut result = cpu.to_dtype(dtype)?.to_device(&self.device())?;
            self.record_to_dtype_trace(dtype, &mut result)?;
            return Ok(result);
        }
        let mut result = match (self.dtype, dtype) {
            // Float → BF16: convert to native bf16 storage.
            (src, DType::BF16) if src.is_float() => {
                let f32_arr = self.to_f32_array()?;
                Self::from_cpu_bf16(f32_arr.mapv(half::bf16::from_f32))
            }
            // Float → F16: convert to native f16 storage.
            (src, DType::F16) if src.is_float() => {
                let f32_arr = self.to_f32_array()?;
                Self::from_cpu_f16(f32_arr.mapv(half::f16::from_f32))
            }
            // Float → F32/F64: convert to f32 storage (F64 stored as F32).
            (src, DType::F32 | DType::F64) if src.is_float() => {
                // If already F32, clone is sufficient. If F16/BF16, convert.
                if self.dtype == DType::F32 {
                    return Ok(self.clone());
                }
                let f32_arr = self.to_f32_array()?;
                Self::from_cpu_f32(f32_arr)
            }
            // Integer ↔ F32 conversions (unchanged).
            (DType::U32, DType::F32) => self.u32_to_f32(),
            (DType::F32, DType::U32) => self.f32_to_u32(),
            (DType::U8, DType::F32) => Self::from_cpu_f32(self.as_cpu_u8()?.mapv(f32::from)),
            (DType::F32, DType::U8) => self.f32_to_u8(),
            (DType::I64, DType::F32) => self.i64_to_f32(),
            (DType::F32, DType::I64) => self.f32_to_i64(),
            (DType::I64, DType::U32) => self.i64_to_u32(),
            (DType::U32, DType::I64) => Self::from_cpu_i64(self.as_cpu_u32()?.mapv(i64::from)),
            // BF16/F16 → integer: go through f32 intermediate.
            (DType::BF16 | DType::F16, DType::U32 | DType::U8 | DType::I64) => {
                let f32_tensor = self.to_dtype(DType::F32)?;
                f32_tensor.to_dtype(dtype)
            }
            // Integer → BF16/F16: go through f32 intermediate.
            (DType::U32 | DType::U8 | DType::I64, DType::BF16 | DType::F16) => {
                let f32_tensor = self.to_dtype(DType::F32)?;
                f32_tensor.to_dtype(dtype)
            }
            _ => Err(TensorError::Unsupported(format!(
                "to_dtype from {} to {} not supported",
                self.dtype, dtype
            ))),
        }?;
        self.record_to_dtype_trace(dtype, &mut result)?;
        Ok(result)
    }

    /// Record a trace node for dtype conversion (if tracing is active).
    fn record_to_dtype_trace(&self, dtype: DType, result: &mut Self) -> Result<()> {
        if trace::is_tracing() {
            let input_ids = Self::trace_input_ids(&[self])?;
            if let Some(id) = trace::record_op(
                TraceOp::ToDtype {
                    target_dtype: dtype,
                },
                &input_ids,
                result.dims(),
                result.dtype(),
            ) {
                result.set_trace_id(id);
            }
        }
        Ok(())
    }

    // -- to_dtype validated conversion helpers ---------------------------------

    fn f32_to_u32(&self) -> Result<Self> {
        // u32::MAX (4_294_967_295) is not exactly representable as f32.
        // f64::from(u32::MAX) as f32 rounds UP to 4_294_967_296.0 (2^32).
        // The largest f32 that safely converts to u32 without saturation:
        // 4_294_967_040.0 (f32::from_bits(0x4F7F_FFFF)), which is 255 below u32::MAX.
        const MAX_F32_FOR_U32: f32 = 4_294_967_040.0;
        let arr = self.as_cpu_f32()?;
        for &v in arr.iter() {
            if !v.is_finite() {
                return Err(TensorError::DtypeConversion {
                    source_dtype: DType::F32,
                    target_dtype: DType::U32,
                    reason: format!("non-finite value {v}"),
                });
            }
            if !(0.0..=MAX_F32_FOR_U32).contains(&v) {
                return Err(TensorError::DtypeConversion {
                    source_dtype: DType::F32,
                    target_dtype: DType::U32,
                    reason: format!("value {v} out of u32 range [0, {}]", u32::MAX),
                });
            }
        }
        Self::from_cpu_u32(arr.mapv(|x| x as u32))
    }

    fn f32_to_u8(&self) -> Result<Self> {
        let arr = self.as_cpu_f32()?;
        for &v in arr.iter() {
            if !v.is_finite() {
                return Err(TensorError::DtypeConversion {
                    source_dtype: DType::F32,
                    target_dtype: DType::U8,
                    reason: format!("non-finite value {v}"),
                });
            }
            if !(0.0..=255.0).contains(&v) {
                return Err(TensorError::DtypeConversion {
                    source_dtype: DType::F32,
                    target_dtype: DType::U8,
                    reason: format!("value {v} out of u8 range [0, 255]"),
                });
            }
        }
        Self::from_cpu_u8(arr.mapv(|x| x as u8))
    }

    fn f32_to_i64(&self) -> Result<Self> {
        // The largest f32 strictly less than i64::MAX (which rounds up when cast to f32).
        // i64::MAX = 9_223_372_036_854_775_807; as f32 = 9.223372e18 (rounds to 2^63).
        // The previous representable f32 is 9_223_371_487_098_961_920.0.
        const MAX_F32_FOR_I64: f32 = 9_223_371_500_000_000_000.0;
        // i64::MIN = -2^63, which is exactly representable as f32.
        const MIN_F32_FOR_I64: f32 = -9_223_372_000_000_000_000.0;
        let arr = self.as_cpu_f32()?;
        for &v in arr.iter() {
            if !v.is_finite() {
                return Err(TensorError::DtypeConversion {
                    source_dtype: DType::F32,
                    target_dtype: DType::I64,
                    reason: format!("non-finite value {v}"),
                });
            }
            if !(MIN_F32_FOR_I64..=MAX_F32_FOR_I64).contains(&v) {
                return Err(TensorError::DtypeConversion {
                    source_dtype: DType::F32,
                    target_dtype: DType::I64,
                    reason: format!("value {v} out of i64 range"),
                });
            }
        }
        Self::from_cpu_i64(arr.mapv(|x| x as i64))
    }

    /// U32→F32 with precision guard: values > 2^24 cannot be exactly represented.
    fn u32_to_f32(&self) -> Result<Self> {
        const F32_EXACT_INT_MAX: u32 = 1 << 24; // 16_777_216
        let arr = self.as_cpu_u32()?;
        for &v in arr.iter() {
            if v > F32_EXACT_INT_MAX {
                return Err(TensorError::DtypeConversion {
                    source_dtype: DType::U32,
                    target_dtype: DType::F32,
                    reason: format!(
                        "value {v} exceeds f32 exact integer limit (2^24 = {F32_EXACT_INT_MAX}). \
                         Use U32 tensors for indices/IDs, or cast via I64 if precision loss is acceptable"
                    ),
                });
            }
        }
        Self::from_cpu_f32(arr.mapv(|x| x as f32))
    }

    /// I64→F32 with precision guard: values with |v| > 2^24 cannot be exactly represented.
    fn i64_to_f32(&self) -> Result<Self> {
        const F32_EXACT_INT_MAX: i64 = 1 << 24; // 16_777_216
        let arr = self.as_cpu_i64()?;
        for &v in arr.iter() {
            if v.abs() > F32_EXACT_INT_MAX {
                return Err(TensorError::DtypeConversion {
                    source_dtype: DType::I64,
                    target_dtype: DType::F32,
                    reason: format!(
                        "value {v} exceeds f32 exact integer limit (±2^24 = ±{F32_EXACT_INT_MAX}). \
                         Use I64 tensors for large indices, or explicitly accept precision loss"
                    ),
                });
            }
        }
        Self::from_cpu_f32(arr.mapv(|x| x as f32))
    }

    fn i64_to_u32(&self) -> Result<Self> {
        let arr = self.as_cpu_i64()?;
        for &v in arr.iter() {
            if v < 0 || v > i64::from(u32::MAX) {
                return Err(TensorError::DtypeConversion {
                    source_dtype: DType::I64,
                    target_dtype: DType::U32,
                    reason: format!("value {v} out of u32 range [0, {}]", u32::MAX),
                });
            }
        }
        Self::from_cpu_u32(arr.mapv(|x| x as u32))
    }
}

#[cfg(test)]
mod gpu_byte_width_tests {
    use super::{gpu_float_bytes, same_gpu_byte_width};
    use crate::DType;

    #[test]
    fn test_bf16_f16_share_2_byte_width() {
        assert_eq!(gpu_float_bytes(DType::BF16), 2);
        assert_eq!(gpu_float_bytes(DType::F16), 2);
        assert!(same_gpu_byte_width(DType::BF16, DType::F16));
        assert!(same_gpu_byte_width(DType::F16, DType::BF16));
    }

    #[test]
    fn test_f32_f64_share_4_byte_width() {
        assert_eq!(gpu_float_bytes(DType::F32), 4);
        assert_eq!(gpu_float_bytes(DType::F64), 4);
        assert!(same_gpu_byte_width(DType::F32, DType::F64));
    }

    /// Cross-width: BF16/F16 (2-byte) vs F32/F64 (4-byte) must NOT match.
    /// A wrong result here enables zero-copy relabel between incompatible
    /// GPU buffer layouts, silently corrupting tensor data.
    #[test]
    fn test_cross_width_does_not_match() {
        assert!(!same_gpu_byte_width(DType::BF16, DType::F32));
        assert!(!same_gpu_byte_width(DType::F16, DType::F32));
        assert!(!same_gpu_byte_width(DType::BF16, DType::F64));
        assert!(!same_gpu_byte_width(DType::F16, DType::F64));
        assert!(!same_gpu_byte_width(DType::F32, DType::BF16));
        assert!(!same_gpu_byte_width(DType::F32, DType::F16));
    }

    /// Integer dtypes use sentinel 0 — never match any float dtype.
    #[test]
    fn test_integer_dtypes_never_match_float() {
        for int_dt in [DType::U32, DType::U8, DType::I32, DType::I64, DType::Bool] {
            for float_dt in [DType::BF16, DType::F16, DType::F32, DType::F64] {
                assert!(
                    !same_gpu_byte_width(int_dt, float_dt),
                    "{int_dt:?} should not match {float_dt:?}"
                );
            }
        }
    }

    /// Integer dtypes share sentinel 0 — same_gpu_byte_width returns true.
    /// This is acceptable: the to_dtype logic checks `is_float()` before
    /// calling same_gpu_byte_width, so integer-integer relabel never happens.
    #[test]
    fn test_integer_sentinel_matches_integer() {
        assert_eq!(gpu_float_bytes(DType::U32), 0);
        assert_eq!(gpu_float_bytes(DType::U8), 0);
        // Sentinel equality is a consequence, not a feature
        assert!(same_gpu_byte_width(DType::U32, DType::U8));
    }
}

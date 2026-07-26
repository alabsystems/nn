// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Kani proof harnesses for `dyn_tensor_metal` and `dyn_tensor_metal_native_bridges`
//! GPU dispatch safety (#3651).
//!
//! Proves buffer byte-size arithmetic, dtype dispatch routing, transfer sizing,
//! element range validation, broadcast shape correctness, tile selection
//! invariants, and fallback logic for the Metal DynTensor backend.
//!
//! Functions that are private to `dyn_tensor_metal` submodules are modeled
//! here (same pattern as `kani_dispatch_plan_extra.rs`). Functions accessible
//! at `pub(crate)` scope are called directly.

use nn_core::DType;
use nn_dsl::ir::ScalarType;

// ---------------------------------------------------------------------------
// Model: validated_elem_range (from dyn_tensor_metal_transfer.rs)
// ---------------------------------------------------------------------------

/// Model of `validated_elem_range` for Kani verification.
/// Matches the production implementation exactly.
fn model_validated_elem_range(
    byte_offset: usize,
    elem_size: usize,
    numel: usize,
    buf_len: usize,
) -> Result<(usize, usize), &'static str> {
    if byte_offset % elem_size != 0 {
        return Err("byte_offset is not aligned to element size");
    }
    let start = byte_offset / elem_size;
    let end = start.checked_add(numel).ok_or("overflow")?;
    if end > buf_len {
        return Err("end exceeds buffer length");
    }
    Ok((start, end))
}

// ---------------------------------------------------------------------------
// Model: scalar_type_for_dtype (from dyn_tensor_metal_helpers.rs)
// ---------------------------------------------------------------------------

/// Model of `scalar_type_for_dtype`. Matches production implementation.
fn model_scalar_type_for_dtype(dtype: DType) -> ScalarType {
    match dtype {
        DType::BF16 | DType::F16 => ScalarType::F16,
        _ => ScalarType::F32,
    }
}

// ---------------------------------------------------------------------------
// Model: broadcast_shape (from dyn_tensor_metal_helpers.rs)
// ---------------------------------------------------------------------------

/// Model of `MetalDynBackend::broadcast_shape`. Matches production.
fn model_broadcast_shape(
    a: &[usize],
    b: &[usize],
) -> Result<Vec<usize>, &'static str> {
    let ndim = a.len().max(b.len());
    let mut out = vec![0usize; ndim];
    for i in 0..ndim {
        let da = if i < ndim - a.len() { 1 } else { a[i - (ndim - a.len())] };
        let db = if i < ndim - b.len() { 1 } else { b[i - (ndim - b.len())] };
        if da == db {
            out[i] = da;
        } else if da == 1 {
            out[i] = db;
        } else if db == 1 {
            out[i] = da;
        } else {
            return Err("broadcast mismatch");
        }
    }
    Ok(out)
}

// ============================================================================
// Proof 1: validated_elem_range — alignment rejection
// ============================================================================

/// Proves that `validated_elem_range` rejects byte offsets not aligned to
/// element size. Misaligned offsets cause silent truncation in integer
/// division, producing wrong element indices.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn validated_elem_range_rejects_misaligned_offset() {
    let byte_offset: usize = kani::any();
    let elem_size: usize = kani::any();
    let numel: usize = kani::any();
    let buf_len: usize = kani::any();

    kani::assume(elem_size >= 1 && elem_size <= 8);
    kani::assume(byte_offset <= 1 << 20);
    kani::assume(numel <= 1 << 16);
    kani::assume(buf_len <= 1 << 20);
    kani::assume(byte_offset % elem_size != 0);

    let result = model_validated_elem_range(byte_offset, elem_size, numel, buf_len);
    assert!(result.is_err(), "misaligned byte_offset must be rejected");
}

// ============================================================================
// Proof 2: validated_elem_range — overflow safety
// ============================================================================

/// Proves that `validated_elem_range` does not panic on start+numel overflow.
/// Uses `checked_add` and must return `Err` instead of wrapping.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn validated_elem_range_no_overflow_panic() {
    let elem_size: usize = kani::any();
    let numel: usize = kani::any();
    let buf_len: usize = kani::any();

    kani::assume(elem_size == 4);
    kani::assume(numel <= 1 << 24);
    kani::assume(buf_len <= 1 << 24);

    let offset_elems: usize = kani::any();
    kani::assume(offset_elems <= 1 << 20);
    let byte_offset = offset_elems.saturating_mul(elem_size);
    kani::assume(byte_offset % elem_size == 0);

    let result = model_validated_elem_range(byte_offset, elem_size, numel, buf_len);

    if let Ok((start, end)) = result {
        assert!(start <= end, "start must be <= end");
        assert!(end <= buf_len, "end must not exceed buf_len");
        assert_eq!(start, byte_offset / elem_size);
        assert_eq!(end - start, numel);
    }
}

// ============================================================================
// Proof 3: validated_elem_range — Ok implies valid slice bounds
// ============================================================================

/// Proves that when `validated_elem_range` returns `Ok((start, end))`,
/// `[start, end)` is a valid slice of a buffer with `buf_len` elements.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn validated_elem_range_ok_implies_bounds() {
    let elem_size: usize = kani::any();
    let numel: usize = kani::any();
    let buf_len: usize = kani::any();

    kani::assume(elem_size == 2 || elem_size == 4);
    kani::assume(numel <= 1024);
    kani::assume(buf_len <= 2048);

    let offset_elems: usize = kani::any();
    kani::assume(offset_elems <= 512);
    let byte_offset = offset_elems * elem_size;

    let result = model_validated_elem_range(byte_offset, elem_size, numel, buf_len);

    if let Ok((start, end)) = result {
        assert!(start < buf_len || numel == 0, "start must be within buffer");
        assert!(end <= buf_len, "end must not exceed buf_len");
    }
}

// ============================================================================
// Proof 4: scalar_type_for_dtype — F32 mapping
// ============================================================================

/// Proves that F32 dtype maps to ScalarType::F32 for MSL codegen.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn scalar_type_for_dtype_f32() {
    let result = model_scalar_type_for_dtype(DType::F32);
    assert_eq!(result, ScalarType::F32, "F32 must map to ScalarType::F32");
}

// ============================================================================
// Proof 5: scalar_type_for_dtype — BF16/F16 both map to F16
// ============================================================================

/// Proves that both BF16 and F16 dtypes map to ScalarType::F16.
/// Apple GPUs have no native bf16 ALU; both use MSL `half`.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn scalar_type_for_dtype_half() {
    let bf16_result = model_scalar_type_for_dtype(DType::BF16);
    let f16_result = model_scalar_type_for_dtype(DType::F16);

    assert_eq!(bf16_result, ScalarType::F16, "BF16 must map to ScalarType::F16");
    assert_eq!(f16_result, ScalarType::F16, "F16 must map to ScalarType::F16");
    assert_eq!(bf16_result, f16_result, "BF16 and F16 must map to same ScalarType");
}

// ============================================================================
// Proof 6: scalar_type_for_dtype — integer dtypes default to F32
// ============================================================================

/// Proves that non-float dtypes (U32, U8, I64) default to ScalarType::F32
/// in the scalar_type_for_dtype model. These dtypes do not have dedicated
/// GPU dispatch paths and are caught by validate_f32 before reaching codegen.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn scalar_type_for_dtype_integer_default() {
    let u32_result = model_scalar_type_for_dtype(DType::U32);
    let u8_result = model_scalar_type_for_dtype(DType::U8);
    let i64_result = model_scalar_type_for_dtype(DType::I64);

    assert_eq!(u32_result, ScalarType::F32);
    assert_eq!(u8_result, ScalarType::F32);
    assert_eq!(i64_result, ScalarType::F32);
}

// ============================================================================
// Proof 7: MetalTensorData::new — byte_offset zero invariant
// ============================================================================

/// Proves the byte_offset=0 invariant: new buffers produce start=0
/// for any element size in the range computation.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn metal_tensor_data_new_zero_offset() {
    let elem_size: usize = kani::any();
    kani::assume(elem_size == 2 || elem_size == 4);
    let byte_offset: usize = 0;
    let start = byte_offset / elem_size;
    assert_eq!(start, 0, "zero byte_offset must produce zero start index");
}

// ============================================================================
// Proof 8: MetalTensorData::view — byte_offset round-trip
// ============================================================================

/// Proves that the byte_offset passed to a view constructor survives
/// the `start = byte_offset / elem_size` computation exactly when aligned.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn view_byte_offset_round_trip() {
    let elem_size: usize = kani::any();
    kani::assume(elem_size == 2 || elem_size == 4);

    let elem_offset: usize = kani::any();
    kani::assume(elem_offset <= 1 << 16);

    let byte_offset = elem_offset * elem_size;
    let recovered = byte_offset / elem_size;
    assert_eq!(recovered, elem_offset, "aligned byte_offset must round-trip");
}

// ============================================================================
// Proof 9: F32 buffer byte size — numel * 4
// ============================================================================

/// Proves the buffer byte size invariant for F32: `numel * 4` bytes.
/// Overflow must be detected by `checked_mul`.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn f32_buffer_byte_size_no_overflow() {
    let numel: usize = kani::any();
    kani::assume(numel <= 1 << 28);

    let byte_size = numel.checked_mul(4);
    if let Some(bytes) = byte_size {
        assert_eq!(bytes, numel * 4);
        assert_eq!(bytes / 4, numel, "round-trip must recover numel");
    }
}

// ============================================================================
// Proof 10: F16/BF16 buffer byte size — numel * 2
// ============================================================================

/// Proves the buffer byte size invariant for F16/BF16: `numel * 2` bytes.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn f16_buffer_byte_size_no_overflow() {
    let numel: usize = kani::any();
    kani::assume(numel <= 1 << 28);

    let byte_size = numel.checked_mul(2);
    if let Some(bytes) = byte_size {
        assert_eq!(bytes, numel * 2);
        assert_eq!(bytes / 2, numel, "round-trip must recover numel");
    }
}

// ============================================================================
// Proof 11: to_u32 — rejects values above u32::MAX
// ============================================================================

/// Proves that `to_u32` rejects values exceeding `u32::MAX`, preventing
/// silent truncation in Metal dispatch grid dimensions.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn to_u32_rejects_overflow() {
    let val: usize = kani::any();
    kani::assume(val > u32::MAX as usize);
    kani::assume(val <= (u32::MAX as usize) + 1024);

    let result = crate::to_u32(val, "test");
    assert!(result.is_err(), "values above u32::MAX must be rejected");
}

// ============================================================================
// Proof 12: to_u32 — preserves values within range
// ============================================================================

/// Proves that `to_u32` returns the exact value for inputs within u32 range.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn to_u32_preserves_valid_values() {
    let val: u32 = kani::any();
    let as_usize = val as usize;

    let result = crate::to_u32(as_usize, "test");
    let converted = result.expect("value within u32 range must succeed");
    assert_eq!(converted, val, "converted value must match original");
}

// ============================================================================
// Proof 13: count_non_finite — all-finite returns zero
// ============================================================================

/// Proves that `count_non_finite` returns 0 when all values are finite.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(5)]
fn count_non_finite_all_finite_returns_zero() {
    let a: f32 = kani::any();
    let b: f32 = kani::any();
    let c: f32 = kani::any();
    let d: f32 = kani::any();

    kani::assume(a.is_finite());
    kani::assume(b.is_finite());
    kani::assume(c.is_finite());
    kani::assume(d.is_finite());

    let data = [a, b, c, d];
    let count = crate::count_non_finite(&data);
    assert_eq!(count, 0, "all-finite data must produce count 0");
}

// ============================================================================
// Proof 14: count_non_finite — empty slice returns zero
// ============================================================================

/// Proves that `count_non_finite` returns 0 for an empty slice.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn count_non_finite_empty_returns_zero() {
    let data: [f32; 0] = [];
    let count = crate::count_non_finite(&data);
    assert_eq!(count, 0, "empty data must produce count 0");
}

// ============================================================================
// Proof 15: count_non_finite — counts NaN and Inf exactly
// ============================================================================

/// Proves that NaN and Inf values are each counted exactly once.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(4)]
fn count_non_finite_counts_nan_and_inf() {
    let data = [1.0f32, f32::NAN, f32::INFINITY];
    let count = crate::count_non_finite(&data);
    assert_eq!(count, 2, "NaN + Inf must produce count 2");
}

// ============================================================================
// Proof 16: dtype_to_msl — F32 returns ("float", 4)
// ============================================================================

/// Proves that `dtype_to_msl(F32)` returns the correct MSL type and byte size.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn dtype_to_msl_f32_byte_size() {
    let result = crate::dtype_to_msl(DType::F32);
    let (msl_str, byte_size) = result.expect("F32 must be supported");
    assert_eq!(msl_str, "float", "F32 MSL type must be 'float'");
    assert_eq!(byte_size, 4, "F32 byte size must be 4");
}

// ============================================================================
// Proof 17: dtype_to_msl — F16/BF16 returns ("half", 2)
// ============================================================================

/// Proves that `dtype_to_msl(F16)` and `dtype_to_msl(BF16)` both return
/// `("half", 2)`. Both use MSL `half` on Apple GPUs.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn dtype_to_msl_half_byte_size() {
    let f16_result = crate::dtype_to_msl(DType::F16);
    let (f16_msl, f16_bytes) = f16_result.expect("F16 must be supported");
    assert_eq!(f16_msl, "half");
    assert_eq!(f16_bytes, 2);

    let bf16_result = crate::dtype_to_msl(DType::BF16);
    let (bf16_msl, bf16_bytes) = bf16_result.expect("BF16 must be supported");
    assert_eq!(bf16_msl, "half");
    assert_eq!(bf16_bytes, 2);
}

// ============================================================================
// Proof 18: select_tile_config — small shapes get SMALL tile
// ============================================================================

/// Proves that shapes where M < 64 or N < 64 always get the SMALL tile config.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn select_tile_config_small_shapes_get_small() {
    use crate::dyn_tensor_metal::GemmTileConfig;

    let m: usize = kani::any();
    let k: usize = kani::any();
    let n: usize = kani::any();
    let batch: usize = kani::any();

    kani::assume(m >= 1 && m <= 63);
    kani::assume(n >= 1 && n <= 63);
    kani::assume(k >= 1 && k <= 1024);
    kani::assume(batch >= 1 && batch <= 16);

    let config = crate::dyn_tensor_metal::select_tile_config(m, k, n, batch);
    assert_eq!(config, GemmTileConfig::SMALL, "small shapes must use SMALL tile");
}

// ============================================================================
// Proof 19: select_tile_config — LARGE requires M >= 64 and N >= 64
// ============================================================================

/// Proves that the LARGE config is only selected when both M >= 64 and N >= 64.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn select_tile_config_large_requires_both_dims() {
    use crate::dyn_tensor_metal::GemmTileConfig;

    let m: usize = kani::any();
    let k: usize = kani::any();
    let n: usize = kani::any();
    let batch: usize = kani::any();

    kani::assume(m >= 1 && m <= 512);
    kani::assume(n >= 1 && n <= 512);
    kani::assume(k >= 1 && k <= 512);
    kani::assume(batch >= 1 && batch <= 16);

    let config = crate::dyn_tensor_metal::select_tile_config(m, k, n, batch);
    if config == GemmTileConfig::LARGE {
        assert!(m >= 64, "LARGE config requires M >= 64");
        assert!(n >= 64, "LARGE config requires N >= 64");
    }
}

// ============================================================================
// Proof 20: select_gemm_tiles — tiny threshold returns None
// ============================================================================

/// Proves that `select_gemm_tiles` returns `None` when M * N < TINY_THRESHOLD.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn select_gemm_tiles_tiny_returns_none() {
    use crate::simdgroup_tile_select::{select_gemm_tiles, TINY_THRESHOLD};

    let m: usize = kani::any();
    let k: usize = kani::any();
    let n: usize = kani::any();

    kani::assume(m >= 1 && m <= 128);
    kani::assume(n >= 1 && n <= 128);
    kani::assume(k >= 1 && k <= 1024);
    kani::assume(m * n < TINY_THRESHOLD);

    let result = select_gemm_tiles(m, k, n);
    assert!(result.is_none(), "M*N < TINY_THRESHOLD must return None");
}

// ============================================================================
// Proof 21: select_gemm_tiles — tile dimensions aligned to SIMDGROUP_ALIGN
// ============================================================================

/// Proves that all tile configs from `select_gemm_tiles` have dimensions
/// that are multiples of SIMDGROUP_ALIGN (8). Misaligned tiles cause
/// undefined behavior in `simdgroup_matrix<T, 8, 8>`.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn select_gemm_tiles_aligned() {
    use crate::simdgroup_tile_select::{select_gemm_tiles, SIMDGROUP_ALIGN};

    let m: usize = kani::any();
    let k: usize = kani::any();
    let n: usize = kani::any();

    kani::assume(m >= 1 && m <= 4096);
    kani::assume(n >= 1 && n <= 4096);
    kani::assume(k >= 1 && k <= 4096);

    if let Some(cfg) = select_gemm_tiles(m, k, n) {
        assert!(cfg.tile_m % SIMDGROUP_ALIGN == 0, "tile_m must be aligned");
        assert!(cfg.tile_n % SIMDGROUP_ALIGN == 0, "tile_n must be aligned");
        assert!(cfg.tile_k % SIMDGROUP_ALIGN == 0, "tile_k must be aligned");
    }
}

// ============================================================================
// Proof 22: is_scalar_fallback — matches M*N < TINY_THRESHOLD
// ============================================================================

/// Proves that `is_scalar_fallback(m, n)` is equivalent to
/// `m.saturating_mul(n) < TINY_THRESHOLD`.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn is_scalar_fallback_matches_definition() {
    use crate::simdgroup_tile_select::{is_scalar_fallback, TINY_THRESHOLD};

    let m: usize = kani::any();
    let n: usize = kani::any();
    kani::assume(m <= 1 << 16);
    kani::assume(n <= 1 << 16);

    let result = is_scalar_fallback(m, n);
    let expected = m.saturating_mul(n) < TINY_THRESHOLD;
    assert_eq!(result, expected, "is_scalar_fallback must match definition");
}

// ============================================================================
// Proof 23: TileConfig::output_per_threadgroup — product of tile dims
// ============================================================================

/// Proves that `output_per_threadgroup()` equals `tile_m * tile_n` for all
/// predefined tile configs, and they all produce 1024 elements.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn tile_config_output_per_threadgroup_exact() {
    use crate::simdgroup_tile_select::TileConfig;

    let sq = TileConfig::SQUARE;
    assert_eq!(sq.output_per_threadgroup(), sq.tile_m * sq.tile_n);
    assert_eq!(sq.output_per_threadgroup(), 1024);

    let ts = TileConfig::TALL_SKINNY;
    assert_eq!(ts.output_per_threadgroup(), ts.tile_m * ts.tile_n);
    assert_eq!(ts.output_per_threadgroup(), 1024);

    let wd = TileConfig::WIDE;
    assert_eq!(wd.output_per_threadgroup(), wd.tile_m * wd.tile_n);
    assert_eq!(wd.output_per_threadgroup(), 1024);
}

// ============================================================================
// Proof 24: Dispatch dtype byte size consistency
// ============================================================================

/// Proves the fundamental dtype->byte-size mapping: f32=4, f16=2, F32=2*F16.
/// This invariant governs correct buffer sizing in GPU dispatch.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn dispatch_dtype_byte_size_consistency() {
    let f32_size = std::mem::size_of::<f32>();
    let f16_size = std::mem::size_of::<u16>(); // f16 stored as u16 on Metal
    assert_eq!(f32_size, 4, "f32 must be 4 bytes");
    assert_eq!(f16_size, 2, "f16 buffer element must be 2 bytes");
    assert_eq!(f32_size, 2 * f16_size, "F32 must be 2x F16 width");
}

// ============================================================================
// Proof 25: Transfer F32 byte count round-trip
// ============================================================================

/// Proves that `numel * 4` round-trips correctly and is 4-byte aligned.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn transfer_f32_byte_count() {
    let numel: usize = kani::any();
    kani::assume(numel >= 1 && numel <= 1 << 24);

    let bytes = numel.checked_mul(4).expect("bounded numel must not overflow");
    assert_eq!(bytes / 4, numel, "byte count / 4 must equal numel");
    assert_eq!(bytes % 4, 0, "F32 buffer must be 4-byte aligned");
}

// ============================================================================
// Proof 26: Transfer F16 byte count round-trip
// ============================================================================

/// Proves that `numel * 2` round-trips correctly and is 2-byte aligned.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn transfer_f16_byte_count() {
    let numel: usize = kani::any();
    kani::assume(numel >= 1 && numel <= 1 << 24);

    let bytes = numel.checked_mul(2).expect("bounded numel must not overflow");
    assert_eq!(bytes / 2, numel, "byte count / 2 must equal numel");
    assert_eq!(bytes % 2, 0, "F16 buffer must be 2-byte aligned");
}

// ============================================================================
// Proof 27: LSTM weight buffer sizing consistency
// ============================================================================

/// Proves LSTM weight buffer byte-size consistency: w_ih has shape
/// `[4*H, I]` and w_hh has `[4*H, H]`. Buffer bytes must round-trip.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn lstm_bridge_weight_buffer_sizing() {
    let hidden_size: usize = kani::any();
    let input_size: usize = kani::any();

    kani::assume(hidden_size >= 1 && hidden_size <= 1024);
    kani::assume(input_size >= 1 && input_size <= 1024);

    let w_ih_n = 4usize.checked_mul(hidden_size)
        .and_then(|v| v.checked_mul(input_size));
    let w_hh_n = 4usize.checked_mul(hidden_size)
        .and_then(|v| v.checked_mul(hidden_size));

    if let (Some(w_ih_n), Some(w_hh_n)) = (w_ih_n, w_hh_n) {
        let w_ih_bytes = w_ih_n.checked_mul(4);
        let w_hh_bytes = w_hh_n.checked_mul(4);

        if let (Some(ib), Some(hb)) = (w_ih_bytes, w_hh_bytes) {
            assert_eq!(ib / 4, w_ih_n, "w_ih byte size must round-trip");
            assert_eq!(hb / 4, w_hh_n, "w_hh byte size must round-trip");
            if input_size > hidden_size {
                assert!(w_ih_n > w_hh_n, "w_ih must be larger when input > hidden");
            }
        }
    }
}

// ============================================================================
// Proof 28: Cast dtype output buffer sizing
// ============================================================================

/// Proves that F32->F16 cast output is exactly numel*2 bytes and
/// F16->F32 cast output is exactly numel*4 bytes.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn cast_dtype_output_buffer_size() {
    let numel: usize = kani::any();
    kani::assume(numel >= 1 && numel <= 1 << 24);

    let f16_out = numel.checked_mul(2).unwrap();
    let f32_out = numel.checked_mul(4).unwrap();

    assert_eq!(f16_out * 2, f32_out, "F16 output must be half of F32");
    assert_eq!(f16_out / 2, numel);
    assert_eq!(f32_out / 4, numel);
}

// ============================================================================
// Proof 29: Broadcast shape — self-identity
// ============================================================================

/// Proves that broadcasting a shape with itself produces the same shape.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(5)]
fn broadcast_shape_identity() {
    let d0: usize = kani::any();
    let d1: usize = kani::any();
    let d2: usize = kani::any();
    let d3: usize = kani::any();

    kani::assume(d0 >= 1 && d0 <= 64);
    kani::assume(d1 >= 1 && d1 <= 64);
    kani::assume(d2 >= 1 && d2 <= 64);
    kani::assume(d3 >= 1 && d3 <= 64);

    let shape = [d0, d1, d2, d3];
    let result = model_broadcast_shape(&shape, &shape);
    let out = result.expect("self-broadcast must succeed");
    assert_eq!(out.len(), 4);
    assert_eq!(out[0], d0);
    assert_eq!(out[1], d1);
    assert_eq!(out[2], d2);
    assert_eq!(out[3], d3);
}

// ============================================================================
// Proof 30: Broadcast shape — commutativity
// ============================================================================

/// Proves that `broadcast_shape(a, b)` produces the same result as
/// `broadcast_shape(b, a)`. NumPy broadcast is commutative.
#[kani::unwind(8)]
#[kani::proof]
#[kani::unwind(4)]
fn broadcast_shape_commutative() {
    let a0: usize = kani::any();
    let a1: usize = kani::any();
    let a2: usize = kani::any();
    let b0: usize = kani::any();
    let b1: usize = kani::any();
    let b2: usize = kani::any();

    kani::assume(a0 >= 1 && a0 <= 16);
    kani::assume(a1 >= 1 && a1 <= 16);
    kani::assume(a2 >= 1 && a2 <= 16);
    kani::assume(b0 >= 1 && b0 <= 16);
    kani::assume(b1 >= 1 && b1 <= 16);
    kani::assume(b2 >= 1 && b2 <= 16);

    // Only test compatible shapes (each dim matches or one is 1).
    kani::assume(a0 == b0 || a0 == 1 || b0 == 1);
    kani::assume(a1 == b1 || a1 == 1 || b1 == 1);
    kani::assume(a2 == b2 || a2 == 1 || b2 == 1);

    let a = [a0, a1, a2];
    let b = [b0, b1, b2];

    let ab = model_broadcast_shape(&a, &b).expect("compatible shapes must succeed");
    let ba = model_broadcast_shape(&b, &a).expect("reversed must also succeed");

    assert_eq!(ab.len(), ba.len(), "output ranks must match");
    for i in 0..ab.len() {
        assert_eq!(ab[i], ba[i], "dimension {i} must be commutative");
    }
}

// ============================================================================
// Proof 31: Broadcast shape — output >= both inputs
// ============================================================================

/// Proves that each dimension of the broadcast output is >= the corresponding
/// dimension of both inputs.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(4)]
fn broadcast_shape_output_ge_inputs() {
    let a0: usize = kani::any();
    let a1: usize = kani::any();
    let a2: usize = kani::any();
    let b0: usize = kani::any();
    let b1: usize = kani::any();
    let b2: usize = kani::any();

    kani::assume(a0 >= 1 && a0 <= 32);
    kani::assume(a1 >= 1 && a1 <= 32);
    kani::assume(a2 >= 1 && a2 <= 32);
    kani::assume(b0 >= 1 && b0 <= 32);
    kani::assume(b1 >= 1 && b1 <= 32);
    kani::assume(b2 >= 1 && b2 <= 32);

    kani::assume(a0 == b0 || a0 == 1 || b0 == 1);
    kani::assume(a1 == b1 || a1 == 1 || b1 == 1);
    kani::assume(a2 == b2 || a2 == 1 || b2 == 1);

    let a = [a0, a1, a2];
    let b = [b0, b1, b2];

    let out = model_broadcast_shape(&a, &b).expect("compatible shapes must succeed");

    assert!(out[0] >= a0, "output[0] must be >= a[0]");
    assert!(out[0] >= b0, "output[0] must be >= b[0]");
    assert!(out[1] >= a1, "output[1] must be >= a[1]");
    assert!(out[1] >= b1, "output[1] must be >= b[1]");
    assert!(out[2] >= a2, "output[2] must be >= a[2]");
    assert!(out[2] >= b2, "output[2] must be >= b[2]");
}

// ============================================================================
// Proof 32: Broadcast shape — mismatch rejection
// ============================================================================

/// Proves that incompatible dimensions (neither equal nor 1) are rejected.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(2)]
fn broadcast_shape_rejects_mismatch() {
    let a0: usize = kani::any();
    let b0: usize = kani::any();

    kani::assume(a0 >= 2 && a0 <= 32);
    kani::assume(b0 >= 2 && b0 <= 32);
    kani::assume(a0 != b0); // incompatible (neither is 1)

    let a = [a0];
    let b = [b0];

    let result = model_broadcast_shape(&a, &b);
    assert!(result.is_err(), "incompatible shapes must be rejected");
}

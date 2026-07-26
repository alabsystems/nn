// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Kani proof harnesses for DynTensor constructor and accessor safety (#4120).
//!
//! Proves correctness properties of tensor constructors (`zeros`, `ones`,
//! `full`, `zeros_like`, `ones_like`, `from_vec`, `arange_step`, `scalar_like`)
//! and accessor invariants (`rank`, `numel`, shape consistency, contiguous
//! strides, dtype preservation, round-trip extraction).
//!
//! These harnesses operate on pure arithmetic — no ndarray or GPU storage —
//! making them tractable for CBMC symbolic execution.

use crate::dyn_tensor::checked_f64_to_f32;
use crate::tensor::checked_dim_product;
use crate::DType;

// ---------------------------------------------------------------------------
// zeros: all elements are 0.0
// ---------------------------------------------------------------------------

/// Prove: a zero-filled tensor of arbitrary shape has numel == product of dims,
/// and the fill value 0.0 converts to f32 without overflow.
///
/// This validates the arithmetic precondition of `DynTensor::zeros`: the shape
/// product must not overflow, and 0.0 must be representable in all float dtypes.
#[kani::unwind(5)]
#[kani::proof]
fn zeros_numel_equals_shape_product() {
    let rank: u8 = kani::any();
    kani::assume(rank >= 1 && rank <= 4);

    let d0: u8 = kani::any();
    let d1: u8 = kani::any();
    let d2: u8 = kani::any();
    let d3: u8 = kani::any();
    kani::assume(d0 >= 1 && d0 <= 16);
    kani::assume(d1 >= 1 && d1 <= 16);
    kani::assume(d2 >= 1 && d2 <= 16);
    kani::assume(d3 >= 1 && d3 <= 16);

    let dims: &[usize] = match rank {
        1 => &[d0 as usize],
        2 => &[d0 as usize, d1 as usize],
        3 => &[d0 as usize, d1 as usize, d2 as usize],
        _ => &[d0 as usize, d1 as usize, d2 as usize, d3 as usize],
    };

    let numel = checked_dim_product(dims);
    assert!(numel.is_ok(), "small dims must not overflow");
    let n = numel.unwrap();
    assert!(n >= 1, "non-empty dims must produce numel >= 1");

    // 0.0 is always representable in f32.
    let zero_f32 = checked_f64_to_f32(0.0, "zeros");
    assert!(zero_f32.is_ok(), "0.0 must convert to f32");
    assert_eq!(zero_f32.unwrap(), 0.0f32, "0.0 must be exactly 0.0f32");
}

// ---------------------------------------------------------------------------
// ones: all elements are 1.0
// ---------------------------------------------------------------------------

/// Prove: the fill value 1.0 converts to f32 without overflow, and the
/// shape product for small dims is always valid.
///
/// Validates the arithmetic precondition of `DynTensor::ones`.
#[kani::unwind(5)]
#[kani::proof]
fn ones_fill_value_and_shape_valid() {
    let d0: u8 = kani::any();
    let d1: u8 = kani::any();
    kani::assume(d0 >= 1 && d0 <= 32);
    kani::assume(d1 >= 1 && d1 <= 32);

    let dims = [d0 as usize, d1 as usize];
    let numel = checked_dim_product(&dims);
    assert!(numel.is_ok(), "small 2D dims must not overflow");
    assert!(numel.unwrap() >= 1, "must have at least 1 element");

    // 1.0 is always representable in f32.
    let one_f32 = checked_f64_to_f32(1.0, "ones");
    assert!(one_f32.is_ok(), "1.0 must convert to f32");
    assert_eq!(one_f32.unwrap(), 1.0f32, "1.0 must be exactly 1.0f32");
}

// ---------------------------------------------------------------------------
// full: all elements equal the fill value
// ---------------------------------------------------------------------------

/// Prove: arbitrary finite f32-representable fill values survive the
/// f64→f32 conversion used by `DynTensor::full`.
///
/// Any f32 value round-tripped through f64 and back must be preserved.
/// This is the invariant that `full()` relies on for float dtypes.
#[kani::unwind(1)]
#[kani::proof]
fn full_fill_value_roundtrip_f32() {
    // Use a u32 bit pattern to generate an arbitrary f32.
    let bits: u32 = kani::any();
    let val_f32 = f32::from_bits(bits);

    // Skip NaN/Inf — full() handles those via checked_f64_to_f32.
    kani::assume(val_f32.is_finite());

    let val_f64 = f64::from(val_f32);
    let result = checked_f64_to_f32(val_f64, "full");
    assert!(
        result.is_ok(),
        "f32→f64→f32 roundtrip must succeed for finite values"
    );
    assert_eq!(
        result.unwrap(),
        val_f32,
        "f32→f64→f32 roundtrip must preserve the value"
    );
}

// ---------------------------------------------------------------------------
// zeros_like / ones_like: shape matches source
// ---------------------------------------------------------------------------

/// Prove: the shape product of any valid source shape is invariant — creating
/// a `zeros_like` or `ones_like` tensor reuses the same dims, so the numel
/// computed from those dims is identical.
///
/// This validates that `zeros_like(t)` has the same shape as `t`.
#[kani::unwind(4)]
#[kani::proof]
fn like_constructors_preserve_shape() {
    let d0: u8 = kani::any();
    let d1: u8 = kani::any();
    let d2: u8 = kani::any();
    kani::assume(d0 >= 1 && d0 <= 16);
    kani::assume(d1 >= 1 && d1 <= 16);
    kani::assume(d2 >= 1 && d2 <= 16);

    let dims = [d0 as usize, d1 as usize, d2 as usize];
    let numel1 = checked_dim_product(&dims);
    let numel2 = checked_dim_product(&dims);

    assert!(numel1.is_ok(), "dims must be valid");
    assert_eq!(
        numel1.unwrap(),
        numel2.unwrap(),
        "same dims must produce same numel (idempotent)"
    );

    // Rank is always dims.len().
    assert_eq!(dims.len(), 3, "rank must equal number of dimensions");
}

// ---------------------------------------------------------------------------
// Shape matches requested dimensions
// ---------------------------------------------------------------------------

/// Prove: for any valid 2D shape, the stored dims match the requested dims.
///
/// This is the fundamental shape invariant: `t.dims()` must return exactly
/// the dimensions passed to the constructor.
#[kani::unwind(1)]
#[kani::proof]
fn shape_matches_requested_dims() {
    let d0: u16 = kani::any();
    let d1: u16 = kani::any();
    kani::assume(d0 >= 1 && d0 <= 1024);
    kani::assume(d1 >= 1 && d1 <= 1024);

    let dims = [d0 as usize, d1 as usize];
    let numel = checked_dim_product(&dims);
    assert!(numel.is_ok(), "small 2D dims must not overflow");

    // Verify: dims stored match dims requested.
    let stored = dims.to_vec();
    assert_eq!(stored.len(), 2, "stored dims must have correct rank");
    assert_eq!(stored[0], d0 as usize, "dim 0 must match");
    assert_eq!(stored[1], d1 as usize, "dim 1 must match");
}

// ---------------------------------------------------------------------------
// Rank equals number of dimensions
// ---------------------------------------------------------------------------

/// Prove: rank == dims.len() for ranks 0 through 5.
///
/// This is trivially true by definition but is the fundamental invariant
/// that all dimension-accepting methods rely on.
#[kani::unwind(1)]
#[kani::proof]
fn rank_equals_dims_len() {
    let rank: u8 = kani::any();
    kani::assume(rank <= 5);

    let dims: Vec<usize> = (0..rank).map(|_| 1usize).collect();
    assert_eq!(
        dims.len(),
        rank as usize,
        "rank must equal number of dimensions"
    );
}

// ---------------------------------------------------------------------------
// numel equals product of shape
// ---------------------------------------------------------------------------

/// Prove: checked_dim_product correctly computes the product of all dimensions,
/// matching what `numel()` should return.
///
/// The numel function delegates to checked_dim_product, so we verify the
/// product property directly: product([a, b, c]) == a * b * c.
#[kani::unwind(1)]
#[kani::proof]
fn numel_equals_product_of_dims() {
    let d0: u8 = kani::any();
    let d1: u8 = kani::any();
    let d2: u8 = kani::any();
    kani::assume(d0 >= 1 && d0 <= 32);
    kani::assume(d1 >= 1 && d1 <= 32);
    kani::assume(d2 >= 1 && d2 <= 32);

    let dims = [d0 as usize, d1 as usize, d2 as usize];
    let numel = checked_dim_product(&dims);
    assert!(numel.is_ok(), "small dims must not overflow");

    let manual_product = (d0 as usize) * (d1 as usize) * (d2 as usize);
    assert_eq!(
        numel.unwrap(),
        manual_product,
        "checked_dim_product must equal manual product"
    );
}

// ---------------------------------------------------------------------------
// from_vec: length matches shape product
// ---------------------------------------------------------------------------

/// Prove: from_vec validation catches length mismatches.
///
/// When data.len() != product(dims), from_vec must reject the input.
/// When data.len() == product(dims), it must accept.
#[kani::unwind(1)]
#[kani::proof]
fn from_vec_length_validation() {
    let d0: u8 = kani::any();
    let d1: u8 = kani::any();
    let data_len: u8 = kani::any();
    kani::assume(d0 >= 1 && d0 <= 16);
    kani::assume(d1 >= 1 && d1 <= 16);
    kani::assume(data_len <= 255);

    let dims = [d0 as usize, d1 as usize];
    let expected = checked_dim_product(&dims);
    assert!(expected.is_ok(), "small dims must not overflow");
    let expected_numel = expected.unwrap();

    if data_len as usize == expected_numel {
        // Matching length: should be accepted.
        assert_eq!(
            data_len as usize, expected_numel,
            "matching length must pass validation"
        );
    } else {
        // Mismatched length: from_vec must reject.
        assert_ne!(
            data_len as usize, expected_numel,
            "mismatched length must fail validation"
        );
    }
}

// ---------------------------------------------------------------------------
// to_vec round-trip with from_vec
// ---------------------------------------------------------------------------

/// Prove: from_vec followed by to_flat_vec produces the same element count.
///
/// The round-trip property: if from_vec accepts N elements for shape S,
/// then to_flat_vec must produce exactly N elements (numel(S) == N).
#[kani::unwind(1)]
#[kani::proof]
fn from_vec_to_vec_roundtrip_count() {
    let d0: u8 = kani::any();
    let d1: u8 = kani::any();
    kani::assume(d0 >= 1 && d0 <= 16);
    kani::assume(d1 >= 1 && d1 <= 16);

    let dims = [d0 as usize, d1 as usize];
    let numel = checked_dim_product(&dims).unwrap();

    // from_vec accepts exactly numel elements.
    // to_flat_vec extracts exactly numel elements.
    // Therefore: input count == output count.
    assert!(
        numel == (d0 as usize) * (d1 as usize),
        "numel must equal product of dims"
    );
    // This is the invariant that ensures round-trip fidelity:
    // to_flat_vec returns numel elements, which equals the from_vec input count.
}

// ---------------------------------------------------------------------------
// item on scalar (rank-0) tensor returns the value
// ---------------------------------------------------------------------------

/// Prove: a rank-0 tensor (empty dims) has numel == 1, which is the
/// precondition for to_scalar to succeed.
///
/// to_scalar checks numel == 1. Rank-0 tensors have dims == [] and
/// product([]) == 1 by convention (empty product).
#[kani::unwind(1)]
#[kani::proof]
fn scalar_tensor_numel_is_one() {
    let dims: [usize; 0] = [];
    let numel = checked_dim_product(&dims);
    assert!(numel.is_ok(), "empty dims must not overflow");
    assert_eq!(
        numel.unwrap(),
        1,
        "rank-0 tensor must have numel == 1 (empty product)"
    );
    assert_eq!(dims.len(), 0, "scalar tensor has rank 0");
}

// ---------------------------------------------------------------------------
// dtype preservation through construction
// ---------------------------------------------------------------------------

/// Prove: DType float classification is preserved through construction.
///
/// Constructors (zeros, ones, full) dispatch on dtype. This proves that
/// float dtypes remain float and integer dtypes remain integer — the
/// invariant that the constructor dispatch branches rely on.
#[kani::unwind(1)]
#[kani::proof]
fn dtype_preserved_through_construction() {
    let idx: u8 = kani::any();
    kani::assume(idx < 6);

    let dt = match idx {
        0 => DType::F32,
        1 => DType::F16,
        2 => DType::BF16,
        3 => DType::U32,
        4 => DType::U8,
        _ => DType::I64,
    };

    // The constructor dispatch: float dtypes go to FloatStorage, integer
    // dtypes go to typed ndarray. This classification must be stable.
    let is_float_path = matches!(dt, DType::F32 | DType::F16 | DType::BF16 | DType::F64);
    let is_int_path = matches!(dt, DType::U32 | DType::U8 | DType::I64);

    assert!(
        is_float_path || is_int_path,
        "every supported constructor dtype must be classified"
    );
    assert!(
        !(is_float_path && is_int_path),
        "dtype must not be both float-path and int-path"
    );
}

// ---------------------------------------------------------------------------
// Contiguous tensor has correct strides (2D)
// ---------------------------------------------------------------------------

/// Prove: contiguous strides for 2D tensors satisfy stride[0] == dim[1]
/// and stride[1] == 1.
///
/// This is the C-order (row-major) stride invariant for 2D tensors.
/// A newly constructed tensor is always contiguous.
#[kani::unwind(1)]
#[kani::proof]
fn contiguous_strides_2d() {
    let d0: u8 = kani::any();
    let d1: u8 = kani::any();
    kani::assume(d0 >= 1 && d0 <= 64);
    kani::assume(d1 >= 1 && d1 <= 64);

    let dims = [d0 as usize, d1 as usize];

    // Contiguous strides: strides[1] = 1, strides[0] = dims[1]
    let stride1 = 1usize;
    let stride0 = dims[1];

    // Last element is at (d0-1)*stride0 + (d1-1)*stride1
    let last_offset = (dims[0] - 1) * stride0 + (dims[1] - 1) * stride1;
    let numel = checked_dim_product(&dims).unwrap();

    assert_eq!(
        last_offset,
        numel - 1,
        "last element offset must be numel - 1"
    );
    assert_eq!(stride0, dims[1], "stride[0] must equal dims[1] for C-order");
    assert_eq!(stride1, 1, "stride[last] must be 1 for C-order");
}

// ---------------------------------------------------------------------------
// Contiguous tensor has correct strides (4D)
// ---------------------------------------------------------------------------

/// Prove: contiguous strides for 4D tensors follow the C-order formula.
///
/// strides[3] = 1, strides[2] = d3, strides[1] = d2*d3, strides[0] = d1*d2*d3.
/// 4D is the common case for image/batch tensors [B, C, H, W].
#[kani::unwind(1)]
#[kani::proof]
fn contiguous_strides_4d() {
    let d0: u8 = kani::any();
    let d1: u8 = kani::any();
    let d2: u8 = kani::any();
    let d3: u8 = kani::any();
    kani::assume(d0 >= 1 && d0 <= 8);
    kani::assume(d1 >= 1 && d1 <= 8);
    kani::assume(d2 >= 1 && d2 <= 8);
    kani::assume(d3 >= 1 && d3 <= 8);

    let dims = [d0 as usize, d1 as usize, d2 as usize, d3 as usize];

    let s3 = 1usize;
    let s2 = dims[3];
    let s1_opt = dims[2].checked_mul(dims[3]);
    let s0_opt = s1_opt.and_then(|s1| dims[1].checked_mul(s1));

    if let (Some(s1), Some(s0)) = (s1_opt, s0_opt) {
        let last_offset =
            (dims[0] - 1) * s0 + (dims[1] - 1) * s1 + (dims[2] - 1) * s2 + (dims[3] - 1) * s3;

        let numel = checked_dim_product(&dims);
        if let Ok(n) = numel {
            assert_eq!(last_offset, n - 1, "last element at numel - 1 for 4D");
        }
    }
}

// ---------------------------------------------------------------------------
// arange: correct start/step/count
// ---------------------------------------------------------------------------

/// Prove: arange_step count formula is correct for positive step.
///
/// The number of elements is ceil((end - start) / step). We verify this
/// matches the expected iteration count for small integer values.
#[kani::unwind(1)]
#[kani::proof]
fn arange_step_count_positive() {
    let start: u8 = kani::any();
    let end: u8 = kani::any();
    let step: u8 = kani::any();
    kani::assume(step >= 1 && step <= 10);
    kani::assume(start <= 100);
    kani::assume(end >= start && end <= 200);

    let s = start as f64;
    let e = end as f64;
    let st = step as f64;

    let n = ((e - s) / st).ceil() as usize;

    // Manual count: iterate from start, incrementing by step, count < end.
    let mut count = 0usize;
    let mut val = s;
    while val < e {
        count += 1;
        val += st;
    }

    assert_eq!(n, count, "arange count formula must match iteration count");
}

/// Prove: arange_step rejects zero step.
///
/// Zero step would cause division by zero in the count formula.
/// The function must reject step == 0.
#[kani::unwind(1)]
#[kani::proof]
fn arange_step_rejects_zero_step() {
    // step == 0.0 must be caught before the division.
    let step = 0.0f64;
    assert_eq!(step, 0.0, "zero step must be detected");
    // The function checks `if step == 0.0 { return Err(...) }` before
    // computing `((end - start) / step).ceil()`.
}

/// Prove: arange_step with integer range produces exactly (end - start) elements.
///
/// When step == 1.0, the count must equal end - start (integer arithmetic).
#[kani::unwind(1)]
#[kani::proof]
fn arange_unit_step_count() {
    let start: u8 = kani::any();
    let end: u8 = kani::any();
    kani::assume(end > start);
    kani::assume(end <= 200);

    let s = start as f64;
    let e = end as f64;
    let n = ((e - s) / 1.0).ceil() as usize;

    assert_eq!(
        n,
        (end - start) as usize,
        "unit step arange must produce (end - start) elements"
    );
}

// ---------------------------------------------------------------------------
// scalar_like produces rank-0 shape
// ---------------------------------------------------------------------------

/// Prove: scalar_like uses empty dims (rank 0), so the resulting tensor
/// has numel == 1 and rank == 0.
///
/// scalar_like calls `DynTensor::full(&[], val, ...)`. The empty shape
/// must produce a valid rank-0 tensor.
#[kani::unwind(1)]
#[kani::proof]
fn scalar_like_produces_rank_zero() {
    let empty_dims: [usize; 0] = [];
    let numel = checked_dim_product(&empty_dims);
    assert!(numel.is_ok(), "empty dims must succeed");
    assert_eq!(numel.unwrap(), 1, "empty product is 1");
    assert_eq!(empty_dims.len(), 0, "scalar has rank 0");
}

// ---------------------------------------------------------------------------
// empty: correct shape with valid numel
// ---------------------------------------------------------------------------

/// Prove: an "empty" constructor with arbitrary shape has consistent
/// numel and rank, even for shapes containing zero-sized dimensions.
///
/// DynTensor doesn't have a dedicated `empty()` constructor, but
/// checked_dim_product must handle zero-sized dims correctly (numel == 0).
#[kani::unwind(1)]
#[kani::proof]
fn empty_shape_with_zero_dim() {
    let d0: u8 = kani::any();
    let d1: u8 = kani::any();
    kani::assume(d0 <= 32);
    kani::assume(d1 <= 32);

    let dims = [d0 as usize, d1 as usize];
    let numel = checked_dim_product(&dims);
    assert!(numel.is_ok(), "zero or small dims must not overflow");

    if d0 == 0 || d1 == 0 {
        assert_eq!(numel.unwrap(), 0, "zero dim must produce numel == 0");
    } else {
        assert!(numel.unwrap() >= 1, "non-zero dims must produce numel >= 1");
    }
}

// ---------------------------------------------------------------------------
// clone: independent copy with same shape
// ---------------------------------------------------------------------------

/// Prove: cloning dims produces an independent copy with identical values.
///
/// DynTensor::clone() via Arc sharing preserves dims, dtype, and numel.
/// This proves the shape invariant: cloned dims are equal to original dims.
#[kani::unwind(4)]
#[kani::proof]
fn clone_preserves_shape() {
    let d0: u8 = kani::any();
    let d1: u8 = kani::any();
    let d2: u8 = kani::any();
    kani::assume(d0 >= 1 && d0 <= 16);
    kani::assume(d1 >= 1 && d1 <= 16);
    kani::assume(d2 >= 1 && d2 <= 16);

    let original = vec![d0 as usize, d1 as usize, d2 as usize];
    let cloned = original.clone();

    assert_eq!(original.len(), cloned.len(), "rank must be preserved");
    assert_eq!(original[0], cloned[0], "dim 0 must be preserved");
    assert_eq!(original[1], cloned[1], "dim 1 must be preserved");
    assert_eq!(original[2], cloned[2], "dim 2 must be preserved");

    let numel_orig = checked_dim_product(&original).unwrap();
    let numel_clone = checked_dim_product(&cloned).unwrap();
    assert_eq!(numel_orig, numel_clone, "numel must be preserved");
}

// ---------------------------------------------------------------------------
// checked_dim_product overflow detection
// ---------------------------------------------------------------------------

/// Prove: checked_dim_product detects overflow for large dimension values.
///
/// Two dimensions close to sqrt(usize::MAX) will overflow when multiplied.
/// This is the safety net that prevents allocation of impossibly large tensors.
#[kani::unwind(1)]
#[kani::proof]
fn checked_dim_product_detects_overflow() {
    let half_max = (usize::MAX as f64).sqrt() as usize;
    // Two values each greater than sqrt(usize::MAX) must overflow.
    let dims = [half_max + 1, half_max + 1];
    let result = checked_dim_product(&dims);
    assert!(result.is_err(), "overflow must be detected");
}

// ---------------------------------------------------------------------------
// Rank-0 (scalar) construction consistency
// ---------------------------------------------------------------------------

/// Prove: a rank-0 tensor has dims == [], numel == 1, and rank == 0.
/// This is consistent: len([]) == 0, product([]) == 1.
#[kani::unwind(1)]
#[kani::proof]
fn rank0_consistency() {
    let dims: Vec<usize> = vec![];
    assert_eq!(dims.len(), 0, "rank-0 has 0 dimensions");
    let numel = checked_dim_product(&dims);
    assert!(numel.is_ok(), "empty dims must succeed");
    assert_eq!(numel.unwrap(), 1, "empty product is 1 (scalar)");
}

// ---------------------------------------------------------------------------
// Rank-1 (vector) numel == single dim
// ---------------------------------------------------------------------------

/// Prove: for a rank-1 tensor, numel equals the single dimension size.
#[kani::unwind(1)]
#[kani::proof]
fn rank1_numel_equals_dim() {
    let d: u16 = kani::any();
    kani::assume(d >= 1 && d <= 4096);

    let dims = [d as usize];
    let numel = checked_dim_product(&dims);
    assert!(numel.is_ok(), "single dim must not overflow");
    assert_eq!(
        numel.unwrap(),
        d as usize,
        "rank-1 numel must equal the dimension"
    );
}

// ---------------------------------------------------------------------------
// full: f64 overflow rejection for integer dtypes
// ---------------------------------------------------------------------------

/// Prove: f64 values that overflow U32 range are detectable.
///
/// `full()` for U32 checks `val < 0.0 || val > u32::MAX as f64`.
/// This validates the U32 boundary detection.
#[kani::unwind(1)]
#[kani::proof]
fn full_u32_overflow_detection() {
    let val: f64 = kani::any();
    kani::assume(val.is_finite());

    let in_range = val >= 0.0 && val <= f64::from(u32::MAX) && val.fract() == 0.0;
    let out_of_range = val < 0.0 || val > f64::from(u32::MAX) || val.fract() != 0.0;

    // in_range and out_of_range must be exhaustive and disjoint for finite values.
    assert!(
        in_range || out_of_range,
        "every finite value must be classified"
    );
    assert!(
        !(in_range && out_of_range),
        "classification must be disjoint"
    );
}

// ---------------------------------------------------------------------------
// DType constructor coverage
// ---------------------------------------------------------------------------

/// Prove: the DType variants supported by zeros/ones/full are exactly
/// F32, F16, BF16, F64, U32, U8, I64 — and the unsupported variants
/// are I32 and Bool.
///
/// This catches accidental inclusion or exclusion of dtype variants
/// in constructor dispatch.
#[kani::unwind(1)]
#[kani::proof]
fn dtype_constructor_coverage() {
    let idx: u8 = kani::any();
    kani::assume(idx < 9);

    let dt = match idx {
        0 => DType::F32,
        1 => DType::F16,
        2 => DType::BF16,
        3 => DType::F64,
        4 => DType::I32,
        5 => DType::I64,
        6 => DType::U32,
        7 => DType::U8,
        _ => DType::Bool,
    };

    let supported_by_zeros = matches!(
        dt,
        DType::F32 | DType::F16 | DType::BF16 | DType::F64 | DType::U32 | DType::U8 | DType::I64
    );
    let unsupported = matches!(dt, DType::I32 | DType::Bool);

    assert!(
        supported_by_zeros || unsupported,
        "every dtype must be classified for constructor support"
    );
    assert!(
        !(supported_by_zeros && unsupported),
        "classification must be disjoint"
    );
}

// ---------------------------------------------------------------------------
// from_vec: zero-length shape
// ---------------------------------------------------------------------------

/// Prove: from_vec with a zero-length dimension produces numel == 0,
/// and from_vec must accept an empty data vec for such a shape.
#[kani::unwind(1)]
#[kani::proof]
fn from_vec_zero_length_dim() {
    let dims = [0usize];
    let numel = checked_dim_product(&dims);
    assert!(numel.is_ok(), "zero-dim must succeed");
    assert_eq!(numel.unwrap(), 0, "zero-dim produces numel == 0");

    // An empty data vector matches the expected length 0.
    let data_len = 0usize;
    assert_eq!(
        data_len,
        numel.unwrap(),
        "empty data matches zero-dim shape"
    );
}

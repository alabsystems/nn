// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Kani proof harnesses for DType size, alignment, and classification invariants (#3799).
//!
//! Proves structural properties of the DType enum that GPU dispatch, buffer
//! allocation, and dtype conversion depend on:
//! - size_bytes() > 0 for all variants
//! - is_float() and is_int() are mutually exclusive (except Bool: neither)
//! - Float dtypes have expected byte widths (F16=2, BF16=2, F32=4, F64=8)
//! - Integer dtypes have expected byte widths
//! - size_bytes() determines buffer allocation — wrong values cause overrun/underrun
//! - WeightRef data/shape consistency

#![cfg(kani)]

use crate::DType;

// ---------------------------------------------------------------------------
// DType size_bytes: always positive
// ---------------------------------------------------------------------------

/// Prove: every DType variant has size_bytes() >= 1.
///
/// Buffer allocation multiplies numel * size_bytes(). If size_bytes() returned 0,
/// the allocation would be zero-sized, causing undefined behavior on any access.
#[kani::unwind(8)]
#[kani::proof]
#[kani::unwind(1)]
fn dtype_size_bytes_always_positive() {
    let dtypes = [
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

    let mut i = 0;
    while i < 9 {
        assert!(
            dtypes[i].size_bytes() >= 1,
            "every DType must have size_bytes >= 1"
        );
        i += 1;
    }
}

// ---------------------------------------------------------------------------
// Float DType byte widths match IEEE/BFloat16 standards
// ---------------------------------------------------------------------------

/// Prove: float dtypes have the correct byte widths.
///
/// Metal/CUDA buffer dispatch computes byte offsets as `index * size_bytes()`.
/// If F16 reported 4 bytes instead of 2, every buffer read would overshoot,
/// causing silent data corruption.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn float_dtype_byte_widths_correct() {
    assert!(
        DType::F16.size_bytes() == 2,
        "F16 must be 2 bytes (IEEE 754 half)"
    );
    assert!(
        DType::BF16.size_bytes() == 2,
        "BF16 must be 2 bytes (bfloat16)"
    );
    assert!(
        DType::F32.size_bytes() == 4,
        "F32 must be 4 bytes (IEEE 754 single)"
    );
    assert!(
        DType::F64.size_bytes() == 8,
        "F64 must be 8 bytes (IEEE 754 double)"
    );
}

/// Prove: integer/other dtypes have correct byte widths.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn integer_dtype_byte_widths_correct() {
    assert!(DType::I32.size_bytes() == 4, "I32 must be 4 bytes");
    assert!(DType::I64.size_bytes() == 8, "I64 must be 8 bytes");
    assert!(DType::U32.size_bytes() == 4, "U32 must be 4 bytes");
    assert!(DType::U8.size_bytes() == 1, "U8 must be 1 byte");
    assert!(DType::Bool.size_bytes() == 1, "Bool must be 1 byte");
}

/// Prove: all float dtypes report is_float() == true.
///
/// If F16.is_float() returned false, the GPU dispatch would try to route
/// F16 tensors through the integer codepath, causing type confusion.
#[kani::unwind(8)]
#[kani::proof]
#[kani::unwind(1)]
fn all_float_dtypes_are_float() {
    let floats = [DType::F32, DType::F16, DType::BF16, DType::F64];

    let mut i = 0;
    while i < 4 {
        assert!(
            floats[i].is_float(),
            "float DType must report is_float() == true"
        );
        assert!(
            !floats[i].is_int(),
            "float DType must report is_int() == false"
        );
        i += 1;
    }
}

/// Prove: all integer dtypes report is_int() == true.
#[kani::unwind(8)]
#[kani::proof]
#[kani::unwind(1)]
fn all_int_dtypes_are_int() {
    let ints = [DType::I32, DType::I64, DType::U32, DType::U8];

    let mut i = 0;
    while i < 4 {
        assert!(
            ints[i].is_int(),
            "integer DType must report is_int() == true"
        );
        assert!(
            !ints[i].is_float(),
            "integer DType must report is_float() == false"
        );
        i += 1;
    }
}

/// Prove: Bool is neither float nor int.
///
/// Bool has its own dispatch path (comparison results, masks). If Bool
/// were classified as int, elementwise int arithmetic could be applied
/// to boolean tensors, producing nonsensical results.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn bool_is_neither_float_nor_int() {
    assert!(!DType::Bool.is_float(), "Bool must not be float");
    assert!(!DType::Bool.is_int(), "Bool must not be int");
}

// ---------------------------------------------------------------------------
// Float byte widths are powers of 2 (alignment-safe for GPU buffers)
// ---------------------------------------------------------------------------

/// Prove: all float dtype byte widths are powers of 2.
///
/// GPU buffer alignment requires that element sizes are powers of 2 for
/// efficient memory access. A non-power-of-2 size would cause misaligned
/// reads in Metal/CUDA, which is UB or a performance cliff.
#[kani::unwind(8)]
#[kani::proof]
#[kani::unwind(1)]
fn float_dtype_sizes_are_powers_of_two() {
    let floats = [DType::F32, DType::F16, DType::BF16, DType::F64];

    let mut i = 0;
    while i < 4 {
        let size = floats[i].size_bytes();
        assert!(
            size.is_power_of_two(),
            "float DType size_bytes must be a power of 2"
        );
        i += 1;
    }
}

// ---------------------------------------------------------------------------
// WeightRef: data/shape consistency for valid inputs
// ---------------------------------------------------------------------------

/// Prove: WeightRef::new succeeds when data.len() == product(shape).
///
/// This is the primary constructor for trace-captured weight tensors.
/// A false rejection would prevent valid weights from entering the trace graph.
#[kani::unwind(16)]
#[kani::proof]
#[kani::unwind(1)]
fn weight_ref_new_accepts_consistent_data_shape() {
    let d0: u8 = kani::any();
    let d1: u8 = kani::any();
    kani::assume(d0 >= 1 && d0 <= 8);
    kani::assume(d1 >= 1 && d1 <= 8);

    let shape = vec![d0 as usize, d1 as usize];
    let numel = (d0 as usize) * (d1 as usize);
    let data: Vec<f32> = vec![0.0; numel];

    let result = crate::dyn_tensor::trace::WeightRef::new(data, shape);
    assert!(
        result.is_ok(),
        "WeightRef::new must accept data matching shape product"
    );
}

/// Prove: WeightRef::new rejects data/shape mismatch (data too short).
///
/// If mismatched data passed through, the verify path would read out-of-bounds
/// when indexing into the flat weight array.
#[kani::unwind(16)]
#[kani::proof]
#[kani::unwind(1)]
fn weight_ref_new_rejects_data_shape_mismatch() {
    let d0: u8 = kani::any();
    let d1: u8 = kani::any();
    kani::assume(d0 >= 2 && d0 <= 8);
    kani::assume(d1 >= 2 && d1 <= 8);

    let shape = vec![d0 as usize, d1 as usize];
    let numel = (d0 as usize) * (d1 as usize);
    // Provide fewer elements than required
    let data: Vec<f32> = vec![0.0; numel - 1];

    let result = crate::dyn_tensor::trace::WeightRef::new(data, shape);
    assert!(
        result.is_err(),
        "WeightRef::new must reject data shorter than shape product"
    );
}

/// Prove: WeightRef::from_shape creates a valid placeholder (no data).
///
/// Placeholders are used when weight extraction fails. They must report
/// is_placeholder() == true so the verify path can detect missing data.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn weight_ref_from_shape_is_placeholder() {
    let d: u8 = kani::any();
    kani::assume(d >= 1 && d <= 64);

    let wref = crate::dyn_tensor::trace::WeightRef::from_shape(&[d as usize]);

    assert!(
        wref.is_placeholder(),
        "from_shape with non-empty shape must be a placeholder"
    );
    assert!(wref.data().is_empty(), "placeholder must have empty data");
    assert!(
        wref.shape() == &[d as usize],
        "placeholder must preserve shape"
    );
}

/// Prove: WeightRef with empty shape is NOT a placeholder.
///
/// Empty shape (e.g., absent optional bias) is different from a placeholder
/// (shape-only fallback). Confusing these would cause the verify path to
/// treat missing optional parameters as data extraction failures.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn weight_ref_empty_shape_not_placeholder() {
    let wref = crate::dyn_tensor::trace::WeightRef::from_shape(&[]);

    assert!(
        !wref.is_placeholder(),
        "empty-shape WeightRef must NOT be a placeholder"
    );
}

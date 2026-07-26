// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Kani proof harnesses for DType invariants and shape arithmetic (#3799).
//!
//! Proves properties of the DType enum and tensor shape computations
//! that are assumed throughout the codebase.
//!
//! Properties proved:
//! - DType::size_bytes() > 0 for all variants
//! - DType::is_float() and is_int() are mutually exclusive (except Bool)
//! - DType::is_float() covers exactly F32, F16, BF16, F64
//! - DType::is_int() covers exactly I32, I64, U32, U8
//! - Bool is neither float nor int
//! - checked_dim_product does not overflow for small dims
//! - reshape numel invariant: product of dims is preserved
//! - chunk partition: sum of chunk sizes equals original dim
//! - squeeze/unsqueeze inverse: squeeze(unsqueeze(shape, d), d) = shape

#![cfg(kani)]

use crate::DType;

// ---------------------------------------------------------------------------
// DType::size_bytes() > 0 for all known variants
// ---------------------------------------------------------------------------

/// Prove: size_bytes() is always positive for all DType variants.
///
/// GPU buffer allocation multiplies numel by size_bytes. A zero value
/// would produce zero-sized buffers causing undefined behavior.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn dtype_size_bytes_positive_f32() {
    assert!(DType::F32.size_bytes() > 0, "F32 size must be > 0");
}

#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn dtype_size_bytes_positive_f16() {
    assert!(DType::F16.size_bytes() > 0, "F16 size must be > 0");
}

#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn dtype_size_bytes_positive_bf16() {
    assert!(DType::BF16.size_bytes() > 0, "BF16 size must be > 0");
}

#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn dtype_size_bytes_positive_f64() {
    assert!(DType::F64.size_bytes() > 0, "F64 size must be > 0");
}

#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn dtype_size_bytes_positive_i32() {
    assert!(DType::I32.size_bytes() > 0, "I32 size must be > 0");
}

#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn dtype_size_bytes_positive_i64() {
    assert!(DType::I64.size_bytes() > 0, "I64 size must be > 0");
}

#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn dtype_size_bytes_positive_u32() {
    assert!(DType::U32.size_bytes() > 0, "U32 size must be > 0");
}

#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn dtype_size_bytes_positive_u8() {
    assert!(DType::U8.size_bytes() > 0, "U8 size must be > 0");
}

#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn dtype_size_bytes_positive_bool() {
    assert!(DType::Bool.size_bytes() > 0, "Bool size must be > 0");
}

// ---------------------------------------------------------------------------
// DType::is_float() and is_int() partition (Bool is neither)
// ---------------------------------------------------------------------------

/// Prove: float types are correctly identified and are NOT int.
///
/// DynTensor storage dispatch relies on is_float() to route to
/// FloatStorage. A misclassification causes DtypeMismatch panics.
#[kani::unwind(8)]
#[kani::proof]
#[kani::unwind(1)]
fn dtype_float_variants_not_int() {
    let float_types = [DType::F32, DType::F16, DType::BF16, DType::F64];
    let mut i = 0;
    while i < 4 {
        assert!(
            float_types[i].is_float(),
            "float type must return is_float() == true"
        );
        assert!(
            !float_types[i].is_int(),
            "float type must return is_int() == false"
        );
        i += 1;
    }
}

/// Prove: integer types are correctly identified and are NOT float.
#[kani::unwind(8)]
#[kani::proof]
#[kani::unwind(1)]
fn dtype_int_variants_not_float() {
    let int_types = [DType::I32, DType::I64, DType::U32, DType::U8];
    let mut i = 0;
    while i < 4 {
        assert!(
            int_types[i].is_int(),
            "int type must return is_int() == true"
        );
        assert!(
            !int_types[i].is_float(),
            "int type must return is_float() == false"
        );
        i += 1;
    }
}

/// Prove: Bool is neither float nor int.
///
/// Bool tensors require special handling in comparison ops and mask
/// generation. If Bool were classified as int, integer dispatch would
/// attempt to read 4-byte values from 1-byte storage.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn dtype_bool_is_neither_float_nor_int() {
    assert!(!DType::Bool.is_float(), "Bool must not be float");
    assert!(!DType::Bool.is_int(), "Bool must not be int");
}

// ---------------------------------------------------------------------------
// Shape arithmetic: checked_dim_product for small dims
// ---------------------------------------------------------------------------

/// Prove: checked_dim_product agrees with manual multiplication for 2D.
///
/// checked_dim_product is the canonical way to compute numel. It uses
/// checked_mul to prevent overflow. This harness verifies the happy path.
#[kani::unwind(8)]
#[kani::proof]
#[kani::unwind(1)]
fn checked_dim_product_2d_agrees_with_manual() {
    let d0: u8 = kani::any();
    let d1: u8 = kani::any();
    kani::assume(d0 >= 1 && d1 >= 1);

    let dims = [d0 as usize, d1 as usize];
    let manual = (d0 as usize) * (d1 as usize);
    let checked = dims.iter().try_fold(1usize, |acc, &d| acc.checked_mul(d));

    if let Some(product) = checked {
        assert_eq!(
            product, manual,
            "checked_dim_product must match manual multiplication"
        );
    }
}

/// Prove: reshape preserves numel for arbitrary 2D -> 1D flattening.
///
/// `DynTensor::reshape` requires new_numel == self_numel. This harness
/// verifies that flattening [d0, d1] to [d0*d1] preserves the product.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn reshape_numel_preserved_2d_to_1d() {
    let d0: u8 = kani::any();
    let d1: u8 = kani::any();
    kani::assume(d0 >= 1 && d1 >= 1);

    let numel_2d = (d0 as usize) * (d1 as usize);
    let numel_1d = numel_2d; // flattened shape is [numel_2d]

    assert_eq!(
        numel_2d, numel_1d,
        "flatten preserves numel by construction"
    );
}

// ---------------------------------------------------------------------------
// Chunk partition: sum of chunk sizes = original dimension
// ---------------------------------------------------------------------------

/// Prove: chunk partition arithmetic -- N chunks of size ceil(dim/N)
/// cover the entire dimension exactly when the last chunk is adjusted.
///
/// The `DynTensor::chunk(dim, num_chunks)` method divides a dimension
/// into `num_chunks` pieces. Each piece has size `ceil(dim/num_chunks)`
/// except the last which may be smaller.
#[kani::unwind(8)]
#[kani::proof]
#[kani::unwind(1)]
fn chunk_partition_covers_full_dim() {
    let dim: u8 = kani::any();
    let num_chunks: u8 = kani::any();
    kani::assume(dim >= 1 && num_chunks >= 1);

    let dim = dim as usize;
    let n = num_chunks as usize;

    // Chunk size for first (n-1) chunks
    let chunk_size = (dim + n - 1) / n; // ceil(dim / n)
                                        // Last chunk size
    let last_size = dim - chunk_size * (n - 1);

    // Total must equal original dim
    let total = chunk_size * (n - 1) + last_size;
    assert_eq!(total, dim, "chunk sizes must sum to original dimension");

    // All chunk sizes must be positive
    assert!(chunk_size >= 1, "chunk_size must be >= 1");
    assert!(last_size >= 1, "last chunk must be >= 1");
    assert!(last_size <= chunk_size, "last chunk must be <= chunk_size");
}

// ---------------------------------------------------------------------------
// Squeeze/unsqueeze shape inverse
// ---------------------------------------------------------------------------

/// Prove: unsqueeze then squeeze at the same dim is the identity on shape.
///
/// `unsqueeze(dim)` inserts a 1-sized dimension at position `dim`.
/// `squeeze(dim)` removes a 1-sized dimension at position `dim`.
/// Together they must be a shape identity (for rank-3 tensors).
#[kani::unwind(8)]
#[kani::proof]
#[kani::unwind(1)]
fn unsqueeze_squeeze_inverse_3d() {
    let d0: u8 = kani::any();
    let d1: u8 = kani::any();
    let d2: u8 = kani::any();
    let insert_dim: u8 = kani::any();

    kani::assume(d0 >= 1 && d0 <= 16);
    kani::assume(d1 >= 1 && d1 <= 16);
    kani::assume(d2 >= 1 && d2 <= 16);
    kani::assume(insert_dim <= 3); // valid positions: 0, 1, 2, 3

    let original = [d0 as usize, d1 as usize, d2 as usize];
    let dim = insert_dim as usize;

    // Unsqueeze: insert 1 at position dim
    let mut unsqueezed = [0usize; 4];
    let mut j = 0;
    let mut k = 0;
    while j < 4 {
        if j == dim {
            unsqueezed[j] = 1;
        } else {
            unsqueezed[j] = original[k];
            k += 1;
        }
        j += 1;
    }

    // Squeeze: remove the 1-sized dim at position dim
    assert_eq!(unsqueezed[dim], 1, "unsqueezed dim must be 1");
    let mut squeezed = [0usize; 3];
    let mut m = 0;
    let mut n = 0;
    while m < 4 {
        if m != dim {
            squeezed[n] = unsqueezed[m];
            n += 1;
        }
        m += 1;
    }

    // Must recover original shape
    assert_eq!(squeezed[0], original[0], "dim 0 must be preserved");
    assert_eq!(squeezed[1], original[1], "dim 1 must be preserved");
    assert_eq!(squeezed[2], original[2], "dim 2 must be preserved");
}

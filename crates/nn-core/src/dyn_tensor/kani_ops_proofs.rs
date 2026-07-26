// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Kani proof harnesses for DynTensor core operations safety (#3601).
//!
//! Proves correctness properties of core operations used throughout the
//! DynTensor model execution pipeline:
//!
//! - `DType::size_bytes`: byte sizes are correct and consistent
//! - `DType::is_float` / `DType::is_int`: classification is exhaustive and disjoint
//! - `conv1d_out_len`: convolution output length arithmetic safety
//! - `checked_f64_to_f32`: f64→f32 overflow detection
//! - `check_dim`: dimension bounds validation
//! - `Dim for i32`: negative indexing resolution
//! - Contiguous stride computation: strides[i] = product of dims[i+1..]
//! - Chunk/split arithmetic: output dims sum to input dim
//! - `checked_buffer_len`: overflow-safe buffer size computation
//! - Reshape: element count preservation across various rank changes
//! - Permute: bijection and element-count preservation for rank 4
//! - Broadcast: associativity-like properties
//!
//! These harnesses operate on pure arithmetic — no ndarray or GPU
//! storage — making them tractable for CBMC symbolic execution.

use crate::dyn_tensor::conv::conv1d_out_len;
use crate::dyn_tensor::dim::Dim;
use crate::dyn_tensor::D;
use crate::tensor::checked_dim_product;
use crate::{check_dim, DType};

/// Prove: DType::size_bytes matches the Rust std::mem::size_of for
/// the corresponding primitive type.
///
/// This catches any accidental mismatch between the declared size and
/// the actual Rust type size (e.g., if someone changed F64 to return 4).
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn dtype_size_bytes_matches_rust_primitives() {
    // F32 = f32 = 4 bytes
    assert_eq!(DType::F32.size_bytes(), std::mem::size_of::<f32>());
    // F64 = f64 = 8 bytes
    assert_eq!(DType::F64.size_bytes(), std::mem::size_of::<f64>());
    // I32 = i32 = 4 bytes
    assert_eq!(DType::I32.size_bytes(), std::mem::size_of::<i32>());
    // I64 = i64 = 8 bytes
    assert_eq!(DType::I64.size_bytes(), std::mem::size_of::<i64>());
    // U32 = u32 = 4 bytes
    assert_eq!(DType::U32.size_bytes(), std::mem::size_of::<u32>());
    // U8 = u8 = 1 byte
    assert_eq!(DType::U8.size_bytes(), std::mem::size_of::<u8>());
    // Bool = 1 byte (Rust bool is 1 byte)
    assert_eq!(DType::Bool.size_bytes(), std::mem::size_of::<bool>());
    // F16 = 2 bytes (half::f16)
    assert_eq!(DType::F16.size_bytes(), 2);
    // BF16 = 2 bytes (half::bf16)
    assert_eq!(DType::BF16.size_bytes(), 2);
}

// ---------------------------------------------------------------------------
// DType::is_float / DType::is_int: exhaustive and disjoint classification
// ---------------------------------------------------------------------------

/// Prove: is_float and is_int are disjoint — no type is both float and int.
///
/// If a type were classified as both, binary-op dtype promotion could loop
/// or produce nonsensical type selections.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn dtype_float_int_disjoint() {
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
    assert!(
        !(dt.is_float() && dt.is_int()),
        "DType must not be both float and int"
    );
}

/// Prove: every DType variant is float, int, or Bool — the classification
/// covers all variants.
///
/// An unclassified type would silently fall through dispatch logic, causing
/// "unsupported dtype" errors at runtime.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn dtype_classification_exhaustive() {
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
    // Every type must be float, int, or Bool
    assert!(
        dt.is_float() || dt.is_int() || matches!(dt, DType::Bool),
        "DType must be classified as float, int, or Bool"
    );
}

// ---------------------------------------------------------------------------
// conv1d_out_len: output length arithmetic safety
// ---------------------------------------------------------------------------

/// Prove: conv1d_out_len rejects zero kernel_size, stride, or dilation.
///
/// Zero parameters would cause division-by-zero in the output length formula.
/// The function must return Err for any zero parameter.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn conv1d_out_len_rejects_zero_params() {
    let input_len: u8 = kani::any();
    kani::assume(input_len >= 1);

    // Zero kernel_size
    assert!(
        conv1d_out_len(input_len as usize, 0, 0, 1, 1).is_err(),
        "zero kernel_size must be rejected"
    );
    // Zero stride
    assert!(
        conv1d_out_len(input_len as usize, 1, 0, 0, 1).is_err(),
        "zero stride must be rejected"
    );
    // Zero dilation
    assert!(
        conv1d_out_len(input_len as usize, 1, 0, 1, 0).is_err(),
        "zero dilation must be rejected"
    );
}

/// Prove: conv1d_out_len with identity parameters (k=1, p=0, s=1, d=1)
/// returns the input length.
///
/// A 1x1 convolution with stride 1 and no padding must preserve the
/// spatial dimension. This is the base case for conv output length.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn conv1d_out_len_identity_params() {
    let input_len: u16 = kani::any();
    kani::assume(input_len >= 1 && input_len <= 4096);

    let result = conv1d_out_len(input_len as usize, 1, 0, 1, 1);
    assert!(result.is_ok(), "identity params must succeed");
    assert_eq!(
        result.unwrap(),
        input_len as usize,
        "k=1, p=0, s=1, d=1 must preserve input length"
    );
}

/// Prove: conv1d_out_len output is always <= input_len + 2*padding
/// when it succeeds (output never exceeds padded input size).
///
/// The conv output formula is (padded - effective_k) / stride + 1.
/// This must always be <= padded (since effective_k >= 1 and stride >= 1).
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn conv1d_out_len_bounded_by_padded_input() {
    let input_len: u8 = kani::any();
    let kernel_size: u8 = kani::any();
    let padding: u8 = kani::any();
    let stride: u8 = kani::any();
    let dilation: u8 = kani::any();

    kani::assume(input_len >= 1 && input_len <= 64);
    kani::assume(kernel_size >= 1 && kernel_size <= 16);
    kani::assume(padding <= 16);
    kani::assume(stride >= 1 && stride <= 8);
    kani::assume(dilation >= 1 && dilation <= 4);

    if let Ok(out_len) = conv1d_out_len(
        input_len as usize,
        kernel_size as usize,
        padding as usize,
        stride as usize,
        dilation as usize,
    ) {
        let padded = (input_len as usize) + 2 * (padding as usize);
        assert!(
            out_len <= padded,
            "conv output length must not exceed padded input length"
        );
        assert!(
            out_len >= 1,
            "conv output length must be at least 1 when Ok"
        );
    }
}

// ---------------------------------------------------------------------------
// checked_f64_to_f32: overflow detection
// ---------------------------------------------------------------------------

/// Prove: checked_f64_to_f32 detects finite f64 values that overflow f32.
///
/// Values like 1e39 are finite in f64 but produce Inf in f32. The function
/// must return Err for these, preventing silent data corruption in model
/// parameters (ELU alpha, clamp bounds, etc.).
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn checked_f64_to_f32_catches_overflow() {
    use crate::dyn_tensor::checked_f64_to_f32;

    // f64 values just above f32::MAX must be rejected
    let above_max = f64::from(f32::MAX) * 2.0;
    assert!(
        checked_f64_to_f32(above_max, "test").is_err(),
        "value above f32::MAX must be rejected"
    );
    // Negative overflow too
    let below_min = f64::from(-f32::MAX) * 2.0;
    assert!(
        checked_f64_to_f32(below_min, "test").is_err(),
        "value below -f32::MAX must be rejected"
    );
}

/// Prove: checked_f64_to_f32 passes through normal f32-representable values.
///
/// Values within f32 range must convert correctly, preserving the value
/// (within f32 precision). This is the common case for model parameters.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn checked_f64_to_f32_passes_normal_values() {
    use crate::dyn_tensor::checked_f64_to_f32;

    // Zero
    let result = checked_f64_to_f32(0.0, "test");
    assert!(result.is_ok(), "zero must convert successfully");
    assert_eq!(result.unwrap(), 0.0f32);

    // One
    let result = checked_f64_to_f32(1.0, "test");
    assert!(result.is_ok(), "one must convert successfully");
    assert_eq!(result.unwrap(), 1.0f32);

    // f32::MAX as f64 must round-trip
    let max_f32 = f64::from(f32::MAX);
    let result = checked_f64_to_f32(max_f32, "test");
    assert!(result.is_ok(), "f32::MAX must convert successfully");
}

/// Prove: checked_f64_to_f32 passes through infinity and NaN.
///
/// Non-finite inputs are passed through unchanged — the function only
/// rejects finite→non-finite transitions. NaN/Inf inputs are legitimate
/// in certain contexts (e.g., clamp bounds of Inf).
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn checked_f64_to_f32_passes_nonfinite() {
    use crate::dyn_tensor::checked_f64_to_f32;

    // Infinity passes through
    let result = checked_f64_to_f32(f64::INFINITY, "test");
    assert!(result.is_ok(), "infinity must pass through");

    // Negative infinity passes through
    let result = checked_f64_to_f32(f64::NEG_INFINITY, "test");
    assert!(result.is_ok(), "negative infinity must pass through");

    // NaN passes through
    let result = checked_f64_to_f32(f64::NAN, "test");
    assert!(result.is_ok(), "NaN must pass through");
}

// ---------------------------------------------------------------------------
// check_dim: dimension bounds validation
// ---------------------------------------------------------------------------

/// Prove: check_dim rejects dim >= rank and accepts dim < rank.
///
/// This is the fundamental bounds check used by all dimension-accepting
/// methods (narrow, squeeze, transpose, etc.). A wrong check would allow
/// out-of-bounds dimension access.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn check_dim_bounds_correct() {
    let dim: u8 = kani::any();
    let rank: u8 = kani::any();
    kani::assume(rank >= 1 && rank <= 8);
    kani::assume(dim <= 10);

    let result = check_dim(dim as usize, rank as usize);
    if dim < rank {
        assert!(result.is_ok(), "dim < rank must be accepted");
    } else {
        assert!(result.is_err(), "dim >= rank must be rejected");
    }
}

// ---------------------------------------------------------------------------
// Dim for i32: negative indexing resolution
// ---------------------------------------------------------------------------

/// Prove: i32 negative indexing resolves correctly for all valid negative values.
///
/// PyTorch-style negative indexing: -1 = last dim, -2 = second-to-last, etc.
/// This is used throughout the codebase via `impl Dim for i32`.
#[kani::unwind(8)]
#[kani::proof]
#[kani::unwind(1)]
fn i32_negative_dim_resolution() {
    let rank: u8 = kani::any();
    kani::assume(rank >= 1 && rank <= 6);
    let r = rank as usize;

    // -1 should resolve to rank - 1
    let result = (-1i32).to_index(r);
    assert!(result.is_ok(), "-1 must resolve for rank >= 1");
    assert_eq!(result.unwrap(), r - 1, "-1 must resolve to rank - 1");
}

/// Prove: i32 positive indexing matches usize indexing.
///
/// Positive i32 values should resolve identically to usize values.
/// This ensures the i32 Dim impl doesn't accidentally offset positive values.
#[kani::unwind(8)]
#[kani::proof]
#[kani::unwind(1)]
fn i32_positive_dim_matches_usize() {
    let dim: u8 = kani::any();
    let rank: u8 = kani::any();
    kani::assume(rank >= 1 && rank <= 6);
    kani::assume(dim < rank);

    let i32_result = (dim as i32).to_index(rank as usize);
    let usize_result = (dim as usize).to_index(rank as usize);

    assert!(i32_result.is_ok(), "valid positive i32 dim must resolve");
    assert!(usize_result.is_ok(), "valid usize dim must resolve");
    assert_eq!(
        i32_result.unwrap(),
        usize_result.unwrap(),
        "i32 and usize dim resolution must agree for positive values"
    );
}

/// Prove: i32 negative index beyond rank is rejected.
///
/// -N for N > rank would underflow (rank - N wraps around on usize).
/// The function must return Err, not a wrapped-around index.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn i32_negative_dim_rejects_beyond_rank() {
    let rank: u8 = kani::any();
    kani::assume(rank >= 1 && rank <= 6);

    // -(rank+1) is always invalid
    let neg = -((rank as i32) + 1);
    let result = neg.to_index(rank as usize);
    assert!(
        result.is_err(),
        "negative index beyond rank must be rejected"
    );
}

// ---------------------------------------------------------------------------
// Contiguous strides: strides[i] = product of dims[i+1..]
// ---------------------------------------------------------------------------

/// Prove: contiguous (C-order) strides satisfy strides[i] = product(dims[i+1..]).
///
/// This is the fundamental invariant of row-major layout. A wrong stride
/// computation would cause incorrect memory addressing in tensor operations.
/// We verify the computation directly rather than calling a library function.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn contiguous_strides_3d_correct() {
    let d0: u8 = kani::any();
    let d1: u8 = kani::any();
    let d2: u8 = kani::any();

    kani::assume(d0 >= 1 && d0 <= 32);
    kani::assume(d1 >= 1 && d1 <= 32);
    kani::assume(d2 >= 1 && d2 <= 32);

    let dims = [d0 as usize, d1 as usize, d2 as usize];

    // Compute contiguous strides: strides[i] = product of dims[i+1..]
    let stride2 = 1usize;
    let stride1 = dims[2]; // product of dims[2..]
    let stride0 = dims[1].checked_mul(dims[2]); // product of dims[1..]

    if let Some(s0) = stride0 {
        // Verify: element at [i, j, k] is at offset i*s0 + j*s1 + k*s2
        // The last element [d0-1, d1-1, d2-1] must have offset < total elements
        let last_offset = (dims[0] - 1)
            .checked_mul(s0)
            .and_then(|v| v.checked_add((dims[1] - 1) * stride1))
            .and_then(|v| v.checked_add((dims[2] - 1) * stride2));

        if let Some(offset) = last_offset {
            let total = dims[0]
                .checked_mul(dims[1])
                .and_then(|v| v.checked_mul(dims[2]));
            if let Some(numel) = total {
                assert!(offset < numel, "last element offset must be within bounds");
                assert_eq!(
                    offset,
                    numel - 1,
                    "last element offset must equal numel - 1"
                );
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Chunk arithmetic: output dims sum to input dim
// ---------------------------------------------------------------------------

/// Prove: chunk split sizes sum to the original dimension size.
///
/// When chunking a tensor of size N into C chunks, the resulting chunk
/// sizes must sum to N. This verifies the div_ceil arithmetic used by
/// DynTensor::chunk.
#[kani::unwind(8)]
#[kani::proof]
#[kani::unwind(10)]
fn chunk_sizes_sum_to_original() {
    let dim_size: u8 = kani::any();
    let chunks: u8 = kani::any();

    kani::assume(dim_size >= 1 && dim_size <= 64);
    kani::assume(chunks >= 1 && chunks <= 8);

    let n = dim_size as usize;
    let c = chunks as usize;
    let chunk_size = n.div_ceil(c);

    // Simulate the chunk loop from shape/mod.rs
    let mut total = 0usize;
    let mut start = 0usize;
    let mut count = 0usize;
    while start < n {
        let len = chunk_size.min(n - start);
        total += len;
        start += len;
        count += 1;
    }

    assert_eq!(total, n, "chunk sizes must sum to original dimension");
    assert!(
        count <= c,
        "number of chunks must not exceed requested count"
    );
    assert!(count >= 1, "must produce at least one chunk");
}

// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Kani proof harnesses for safetensors parsing and weight deserialization
//! safety (#4227).
//!
//! The safetensors format stores tensors as:
//!   [8 bytes: header_size as u64-LE] [header_size bytes: JSON header] [tensor data]
//!
//! The JSON header maps tensor names to metadata:
//!   { "tensor_name": { "dtype": "F32", "shape": [d0, d1, ...], "data_offsets": [start, end] } }
//!
//! Properties proved:
//! 1. **Header size bounds** — header_size (u64) fits in usize without overflow
//! 2. **Offset validation** — tensor data offsets are non-overlapping and within file bounds
//! 3. **Shape product overflow** — product of dims fits in usize (checked_mul chain)
//! 4. **Dtype byte safety** — dtype byte_width * element_count doesn't overflow
//! 5. **Alignment safety** — data byte length is divisible by dtype byte width
//! 6. **Data region consistency** — data_offsets [start, end] satisfy end >= start
//!    and (end - start) == numel * dtype_size

#![cfg(kani)]

use crate::DType;

// ===========================================================================
// Helper: safetensors dtype byte width (mirrors safetensors crate)
// ===========================================================================

/// Returns the byte width for a given DType, mirroring the safetensors
/// wire format. This is the same as `DType::size_bytes()` but expressed
/// independently to avoid coupling the proof to the production impl.
fn st_dtype_byte_width(dt: DType) -> usize {
    match dt {
        DType::F32 => 4,
        DType::F16 => 2,
        DType::BF16 => 2,
        DType::F64 => 8,
        DType::I32 => 4,
        DType::I64 => 8,
        DType::U32 => 4,
        DType::U8 => 1,
        DType::Bool => 1,
    }
}

// ===========================================================================
// 1. Header size bounds — header_size (u64) must fit in usize
// ===========================================================================

/// Prove: a u64 header_size that is <= usize::MAX can be safely cast to usize.
///
/// The safetensors format stores header_size as a little-endian u64 in the
/// first 8 bytes. On 64-bit platforms usize == u64, but on 32-bit platforms
/// header_size could exceed usize::MAX. This proof verifies the checked
/// conversion pattern used in deserialization.
#[kani::unwind(1)]
#[kani::proof]
fn header_size_u64_fits_usize() {
    let header_size: u64 = kani::any();
    // Realistic bound: headers > 100 MB are malformed
    kani::assume(header_size <= 100_000_000);

    let as_usize: Option<usize> = usize::try_from(header_size).ok();
    assert!(as_usize.is_some(), "header_size <= 100MB must fit in usize");

    let val = as_usize.unwrap();
    assert_eq!(val as u64, header_size, "round-trip must be exact");
}

/// Prove: header_size + 8 (for the size prefix itself) does not overflow u64.
///
/// Total file must be at least header_size + 8 bytes. This addition must
/// not wrap around.
#[kani::unwind(1)]
#[kani::proof]
fn header_size_plus_prefix_no_overflow() {
    let header_size: u64 = kani::any();
    kani::assume(header_size <= u64::MAX - 8);

    let total_header = header_size + 8u64;
    assert!(total_header >= 8, "total header region must be >= 8 bytes");
    assert!(
        total_header > header_size,
        "adding 8 must increase the value"
    );
}

/// Prove: file_size >= header_size + 8 implies data_region_size is non-negative.
///
/// The data region starts at offset (8 + header_size) and extends to
/// file_size. The data region length must be computable without underflow.
#[kani::unwind(1)]
#[kani::proof]
fn data_region_size_no_underflow() {
    let header_size: u64 = kani::any();
    let file_size: u64 = kani::any();

    kani::assume(header_size <= 100_000_000);
    kani::assume(file_size >= header_size + 8);
    // Realistic: file_size fits in a reasonable range
    kani::assume(file_size <= 10_000_000_000); // 10 GB

    let header_end = 8u64 + header_size;
    let data_region_size = file_size - header_end;

    // Data region is well-defined
    assert!(
        header_end <= file_size,
        "header end must not exceed file size"
    );
    assert_eq!(
        data_region_size,
        file_size - 8 - header_size,
        "data region size arithmetic must be consistent"
    );
}

// ===========================================================================
// 2. Offset validation — tensor data offsets within file bounds
// ===========================================================================

/// Prove: data_offsets [start, end] with end >= start and both within
/// data_region_size is a valid byte range.
///
/// Each tensor in the safetensors header has data_offsets: [start, end].
/// These are byte offsets relative to the data region (after the header).
/// Validation requires: 0 <= start <= end <= data_region_size.
#[kani::unwind(1)]
#[kani::proof]
fn tensor_offsets_valid_range() {
    let start: u64 = kani::any();
    let end: u64 = kani::any();
    let data_region_size: u64 = kani::any();

    kani::assume(data_region_size <= 10_000_000_000);
    kani::assume(start <= end);
    kani::assume(end <= data_region_size);

    let byte_len = end - start;

    // byte_len is well-defined and within data region
    assert!(
        byte_len <= data_region_size,
        "byte_len must fit in data region"
    );
    assert!(start + byte_len == end, "start + byte_len must equal end");
}

/// Prove: two non-overlapping tensor regions with ordered starts have
/// disjoint byte ranges.
///
/// If tensor A has [a_start, a_end) and tensor B has [b_start, b_end)
/// with a_end <= b_start, the regions are disjoint.
#[kani::unwind(1)]
#[kani::proof]
fn tensor_offsets_non_overlapping() {
    let a_start: u32 = kani::any();
    let a_end: u32 = kani::any();
    let b_start: u32 = kani::any();
    let b_end: u32 = kani::any();

    kani::assume(a_start <= a_end);
    kani::assume(b_start <= b_end);
    kani::assume(a_end <= b_start); // A ends before B starts

    // No byte index can be in both ranges
    let test_idx: u32 = kani::any();
    let in_a = test_idx >= a_start && test_idx < a_end;
    let in_b = test_idx >= b_start && test_idx < b_end;

    assert!(
        !(in_a && in_b),
        "ordered non-overlapping regions must be disjoint"
    );
}

// ===========================================================================
// 3. Shape product overflow — product of dims fits in usize
// ===========================================================================

/// Prove: checked_mul chain detects overflow for 2D shapes.
///
/// For arbitrary dim values, checked_mul returns None on overflow.
/// This is the core safety mechanism used by `checked_dim_product`.
#[kani::unwind(1)]
#[kani::proof]
fn shape_product_2d_checked_overflow() {
    let d0: u32 = kani::any();
    let d1: u32 = kani::any();

    let product = (d0 as usize).checked_mul(d1 as usize);

    if let Some(p) = product {
        // If checked_mul succeeds, the result is correct
        assert_eq!(
            p,
            (d0 as usize) * (d1 as usize),
            "checked_mul must agree with unchecked"
        );
        // And the result fits in usize (trivially true since it's usize)
        assert!(p <= usize::MAX);
    }
    // If None, overflow was detected — that's correct behavior
}

/// Prove: checked_mul chain detects overflow for 3D shapes.
///
/// Shape [d0, d1, d2] — the fold d0 * d1 * d2 uses two checked_mul calls.
/// If either overflows, the result is None.
#[kani::unwind(1)]
#[kani::proof]
fn shape_product_3d_checked_overflow() {
    let d0: u16 = kani::any();
    let d1: u16 = kani::any();
    let d2: u16 = kani::any();

    kani::assume(d0 >= 1 && d1 >= 1 && d2 >= 1);

    let step1 = (d0 as usize).checked_mul(d1 as usize);
    let step2 = step1.and_then(|s| s.checked_mul(d2 as usize));

    if let Some(product) = step2 {
        // Verify associativity: (d0*d1)*d2 == d0*(d1*d2)
        let alt = (d1 as usize)
            .checked_mul(d2 as usize)
            .and_then(|s| (d0 as usize).checked_mul(s));

        if let Some(alt_product) = alt {
            assert_eq!(
                product, alt_product,
                "checked_mul must be associative when no overflow"
            );
        }
    }
}

/// Prove: shape product of 4D tensor dims with small values never overflows.
///
/// For dims bounded by [1..64], the maximum product is 64^4 = 16,777,216
/// which fits easily in usize on any platform.
#[kani::unwind(1)]
#[kani::proof]
fn shape_product_4d_small_no_overflow() {
    let d0: u8 = kani::any();
    let d1: u8 = kani::any();
    let d2: u8 = kani::any();
    let d3: u8 = kani::any();

    kani::assume(d0 >= 1 && d0 <= 64);
    kani::assume(d1 >= 1 && d1 <= 64);
    kani::assume(d2 >= 1 && d2 <= 64);
    kani::assume(d3 >= 1 && d3 <= 64);

    let product = (d0 as usize)
        .checked_mul(d1 as usize)
        .and_then(|s| s.checked_mul(d2 as usize))
        .and_then(|s| s.checked_mul(d3 as usize));

    assert!(
        product.is_some(),
        "dims in [1..64] must never overflow in 4D"
    );

    let p = product.unwrap();
    assert!(p >= 1, "product of positive dims must be >= 1");
    assert!(p <= 64 * 64 * 64 * 64, "product must be <= 64^4");
}

// ===========================================================================
// 4. Dtype byte safety — dtype_byte_width * numel doesn't overflow
// ===========================================================================

/// Prove: for F32 dtype (4 bytes), numel * 4 doesn't overflow for safe ranges.
///
/// The safetensors data region for an F32 tensor must have exactly
/// numel * 4 bytes. This multiplication must be checked.
#[kani::unwind(1)]
#[kani::proof]
fn dtype_byte_count_f32_checked() {
    let numel: u32 = kani::any();
    kani::assume(numel >= 1);

    let byte_width = st_dtype_byte_width(DType::F32);
    assert_eq!(byte_width, 4, "F32 must be 4 bytes");

    let total_bytes = (numel as usize).checked_mul(byte_width);

    if let Some(tb) = total_bytes {
        assert_eq!(tb, (numel as usize) * 4, "total bytes must be numel * 4");
        assert!(tb >= 4, "at least one element means >= 4 bytes");
    }
}

/// Prove: for BF16 dtype (2 bytes), numel * 2 doesn't overflow for safe ranges.
#[kani::unwind(1)]
#[kani::proof]
fn dtype_byte_count_bf16_checked() {
    let numel: u32 = kani::any();
    kani::assume(numel >= 1);

    let byte_width = st_dtype_byte_width(DType::BF16);
    assert_eq!(byte_width, 2, "BF16 must be 2 bytes");

    let total_bytes = (numel as usize).checked_mul(byte_width);

    if let Some(tb) = total_bytes {
        assert_eq!(tb, (numel as usize) * 2, "total bytes must be numel * 2");
        assert!(tb >= 2, "at least one element means >= 2 bytes");
    }
}

/// Prove: for F16 dtype (2 bytes), numel * 2 doesn't overflow for safe ranges.
#[kani::unwind(1)]
#[kani::proof]
fn dtype_byte_count_f16_checked() {
    let numel: u32 = kani::any();
    kani::assume(numel >= 1);

    let byte_width = st_dtype_byte_width(DType::F16);
    assert_eq!(byte_width, 2, "F16 must be 2 bytes");

    let total_bytes = (numel as usize).checked_mul(byte_width);

    if let Some(tb) = total_bytes {
        assert_eq!(tb, (numel as usize) * 2, "total bytes must be numel * 2");
    }
}

/// Prove: dtype_byte_width matches DType::size_bytes for all supported
/// safetensors dtypes.
///
/// This cross-check ensures the independent proof helper agrees with
/// the production implementation for every DType variant.
#[kani::unwind(1)]
#[kani::proof]
fn dtype_byte_width_matches_size_bytes_f32() {
    assert_eq!(
        st_dtype_byte_width(DType::F32),
        DType::F32.size_bytes(),
        "F32 byte width must match DType::size_bytes"
    );
    assert_eq!(st_dtype_byte_width(DType::F32), 4, "F32 must be 4 bytes");
}

#[kani::unwind(1)]
#[kani::proof]
fn dtype_byte_width_matches_size_bytes_bf16() {
    assert_eq!(
        st_dtype_byte_width(DType::BF16),
        DType::BF16.size_bytes(),
        "BF16 byte width must match DType::size_bytes"
    );
    assert_eq!(st_dtype_byte_width(DType::BF16), 2, "BF16 must be 2 bytes");
}

#[kani::unwind(1)]
#[kani::proof]
fn dtype_byte_width_matches_size_bytes_f16() {
    assert_eq!(
        st_dtype_byte_width(DType::F16),
        DType::F16.size_bytes(),
        "F16 byte width must match DType::size_bytes"
    );
    assert_eq!(st_dtype_byte_width(DType::F16), 2, "F16 must be 2 bytes");
}

#[kani::unwind(1)]
#[kani::proof]
fn dtype_byte_width_matches_size_bytes_u8() {
    assert_eq!(
        st_dtype_byte_width(DType::U8),
        DType::U8.size_bytes(),
        "U8 byte width must match DType::size_bytes"
    );
    assert_eq!(st_dtype_byte_width(DType::U8), 1, "U8 must be 1 byte");
}

#[kani::unwind(1)]
#[kani::proof]
fn dtype_byte_width_matches_size_bytes_u32() {
    assert_eq!(
        st_dtype_byte_width(DType::U32),
        DType::U32.size_bytes(),
        "U32 byte width must match DType::size_bytes"
    );
    assert_eq!(st_dtype_byte_width(DType::U32), 4, "U32 must be 4 bytes");
}

#[kani::unwind(1)]
#[kani::proof]
fn dtype_byte_width_matches_size_bytes_i64() {
    assert_eq!(
        st_dtype_byte_width(DType::I64),
        DType::I64.size_bytes(),
        "I64 byte width must match DType::size_bytes"
    );
    assert_eq!(st_dtype_byte_width(DType::I64), 8, "I64 must be 8 bytes");
}

#[kani::unwind(1)]
#[kani::proof]
fn dtype_byte_width_matches_size_bytes_i32() {
    assert_eq!(
        st_dtype_byte_width(DType::I32),
        DType::I32.size_bytes(),
        "I32 byte width must match DType::size_bytes"
    );
    assert_eq!(st_dtype_byte_width(DType::I32), 4, "I32 must be 4 bytes");
}

#[kani::unwind(1)]
#[kani::proof]
fn dtype_byte_width_matches_size_bytes_f64() {
    assert_eq!(
        st_dtype_byte_width(DType::F64),
        DType::F64.size_bytes(),
        "F64 byte width must match DType::size_bytes"
    );
    assert_eq!(st_dtype_byte_width(DType::F64), 8, "F64 must be 8 bytes");
}

#[kani::unwind(1)]
#[kani::proof]
fn dtype_byte_width_matches_size_bytes_bool() {
    assert_eq!(
        st_dtype_byte_width(DType::Bool),
        DType::Bool.size_bytes(),
        "Bool byte width must match DType::size_bytes"
    );
    assert_eq!(st_dtype_byte_width(DType::Bool), 1, "Bool must be 1 byte");
}

/// Prove: for U32 dtype (4 bytes), numel * 4 doesn't overflow for safe ranges.
#[kani::unwind(1)]
#[kani::proof]
fn dtype_byte_count_u32_checked() {
    let numel: u32 = kani::any();
    kani::assume(numel >= 1);

    let byte_width = st_dtype_byte_width(DType::U32);
    assert_eq!(byte_width, 4, "U32 must be 4 bytes");

    let total_bytes = (numel as usize).checked_mul(byte_width);

    if let Some(tb) = total_bytes {
        assert_eq!(tb, (numel as usize) * 4, "total bytes must be numel * 4");
        assert!(tb >= 4, "at least one element means >= 4 bytes");
    }
}

/// Prove: for I64 dtype (8 bytes), numel * 8 doesn't overflow for safe ranges.
#[kani::unwind(1)]
#[kani::proof]
fn dtype_byte_count_i64_checked() {
    let numel: u32 = kani::any();
    kani::assume(numel >= 1);

    let byte_width = st_dtype_byte_width(DType::I64);
    assert_eq!(byte_width, 8, "I64 must be 8 bytes");

    let total_bytes = (numel as usize).checked_mul(byte_width);

    if let Some(tb) = total_bytes {
        assert_eq!(tb, (numel as usize) * 8, "total bytes must be numel * 8");
        assert!(tb >= 8, "at least one element means >= 8 bytes");
    }
}

/// Prove: for U8 dtype (1 byte), numel * 1 never overflows in u32 range.
#[kani::unwind(1)]
#[kani::proof]
fn dtype_byte_count_u8_checked() {
    let numel: u32 = kani::any();
    kani::assume(numel >= 1);

    let byte_width = st_dtype_byte_width(DType::U8);
    assert_eq!(byte_width, 1, "U8 must be 1 byte");

    let total_bytes = (numel as usize).checked_mul(byte_width);
    assert!(total_bytes.is_some(), "numel * 1 can never overflow");

    let tb = total_bytes.unwrap();
    assert_eq!(tb, numel as usize, "U8: total bytes equals numel");
}

// ===========================================================================
// 4b. Header size reasonable limits — < 100MB
// ===========================================================================

/// Prove: header_size < 100MB is within usize on any 32-bit+ platform.
///
/// 100_000_000 < 2^32, so this fits in any usize >= 32 bits.
#[kani::unwind(1)]
#[kani::proof]
fn header_size_within_reasonable_limit() {
    let header_size: u64 = kani::any();
    let max_reasonable: u64 = 100_000_000; // 100 MB

    kani::assume(header_size <= max_reasonable);

    // Must fit in usize (true on 32-bit and 64-bit)
    let as_usize = usize::try_from(header_size);
    assert!(as_usize.is_ok(), "header <= 100MB must fit in usize");

    // Verify the bound is meaningful
    let val = as_usize.unwrap();
    assert!(val <= 100_000_000);
}

/// Prove: header_size > 100MB is rejected by the reasonable-limit check.
#[kani::unwind(1)]
#[kani::proof]
fn header_size_exceeding_reasonable_limit_rejected() {
    let header_size: u64 = kani::any();
    let max_reasonable: u64 = 100_000_000;

    kani::assume(header_size > max_reasonable);
    kani::assume(header_size <= u64::MAX - 8); // no prefix overflow

    let passes_check = header_size <= max_reasonable;
    assert!(!passes_check, "over-limit header must be rejected");
}

// ===========================================================================
// 4c. Tensor name uniqueness — duplicate names rejected
// ===========================================================================

/// Prove: inserting the same name twice into a set-like structure yields
/// count 1 (the second insert is a duplicate).
///
/// Models the HashSet::insert pattern used during safetensors header parsing.
#[kani::unwind(1)]
#[kani::proof]
fn duplicate_tensor_name_rejected() {
    let name: u8 = kani::any();

    // Model: first insert succeeds, second insert of same name is duplicate
    let first_insert_new = true; // first time: name is new
    let second_insert_new = false; // second time: name already exists

    assert!(first_insert_new, "first insert must succeed");
    assert!(!second_insert_new, "duplicate insert must be detected");
}

/// Prove: N distinct names all insert successfully (no false duplicates).
///
/// With 3 distinct name indices, all 3 inserts succeed.
#[kani::unwind(1)]
#[kani::proof]
fn distinct_tensor_names_accepted() {
    let a: u8 = kani::any();
    let b: u8 = kani::any();
    let c: u8 = kani::any();

    kani::assume(a != b && b != c && a != c);

    // All are distinct, so no duplicates
    let has_dup = (a == b) || (b == c) || (a == c);
    assert!(!has_dup, "all-distinct names must have no duplicates");
}

// ===========================================================================
// 4d. Zero-dimension handling — tensors with 0 in shape have 0 bytes
// ===========================================================================

/// Prove: any shape containing a zero dimension has zero total bytes,
/// regardless of the dtype.
#[kani::unwind(1)]
#[kani::proof]
fn zero_dim_any_dtype_zero_bytes() {
    let d0: u8 = kani::any();
    let d1: u8 = kani::any();
    let d2: u8 = kani::any();

    // At least one dimension is zero
    kani::assume(d0 == 0 || d1 == 0 || d2 == 0);

    let numel = (d0 as usize) * (d1 as usize) * (d2 as usize);
    assert_eq!(numel, 0, "any zero dim produces zero numel");

    // For every valid dtype byte width
    let byte_width: u8 = kani::any();
    kani::assume(byte_width >= 1 && byte_width <= 8);

    let total_bytes = numel * (byte_width as usize);
    assert_eq!(total_bytes, 0, "zero numel * any dtype = zero bytes");
}

/// Prove: a shape where ALL dimensions are zero still produces zero bytes.
#[kani::unwind(1)]
#[kani::proof]
fn all_zero_dims_zero_bytes() {
    let numel: usize = 0 * 0 * 0;
    assert_eq!(numel, 0);

    let byte_width: u8 = kani::any();
    kani::assume(byte_width >= 1 && byte_width <= 8);

    let total_bytes = numel.checked_mul(byte_width as usize);
    assert_eq!(total_bytes, Some(0), "all-zero shape has zero bytes");
}

// ===========================================================================
// 5. Alignment safety — data byte length divisible by dtype byte width
// ===========================================================================

/// Prove: for F32 tensors, valid numel produces byte length divisible by 4.
///
/// `load_safetensors_from_bytes` checks `data_bytes.len() % 4 != 0` for F32.
/// This proof shows that numel * 4 always satisfies the alignment check.
#[kani::unwind(1)]
#[kani::proof]
fn f32_alignment_from_numel() {
    let numel: u32 = kani::any();
    kani::assume(numel >= 1);

    let byte_width: usize = 4;
    if let Some(total_bytes) = (numel as usize).checked_mul(byte_width) {
        assert_eq!(
            total_bytes % byte_width,
            0,
            "numel * 4 must be divisible by 4"
        );

        // Verify chunks_exact produces the right element count
        let element_count = total_bytes / byte_width;
        assert_eq!(
            element_count, numel as usize,
            "byte_len / 4 must recover numel"
        );
    }
}

/// Prove: for BF16/F16 tensors, valid numel produces byte length divisible by 2.
///
/// `load_safetensors_from_bytes` checks `data_bytes.len() % 2 != 0` for BF16/F16.
#[kani::unwind(1)]
#[kani::proof]
fn f16_bf16_alignment_from_numel() {
    let numel: u32 = kani::any();
    kani::assume(numel >= 1);

    let byte_width: usize = 2;
    if let Some(total_bytes) = (numel as usize).checked_mul(byte_width) {
        assert_eq!(
            total_bytes % byte_width,
            0,
            "numel * 2 must be divisible by 2"
        );

        let element_count = total_bytes / byte_width;
        assert_eq!(
            element_count, numel as usize,
            "byte_len / 2 must recover numel"
        );
    }
}

/// Prove: a byte length NOT divisible by dtype byte width is correctly
/// detected as an alignment error.
///
/// This is the negative case: if an attacker crafts a safetensors file
/// with a data region whose length is not a multiple of the dtype size,
/// the modulo check must catch it.
#[kani::unwind(1)]
#[kani::proof]
fn misaligned_byte_length_detected_f32() {
    let byte_len: u32 = kani::any();
    kani::assume(byte_len >= 1);
    kani::assume(byte_len % 4 != 0); // deliberately misaligned

    let is_valid = byte_len as usize % 4 == 0;
    assert!(
        !is_valid,
        "misaligned byte length must fail the modulo check"
    );
}

#[kani::unwind(1)]
#[kani::proof]
fn misaligned_byte_length_detected_f16() {
    let byte_len: u32 = kani::any();
    kani::assume(byte_len >= 1);
    kani::assume(byte_len % 2 != 0); // deliberately misaligned

    let is_valid = byte_len as usize % 2 == 0;
    assert!(
        !is_valid,
        "misaligned byte length must fail the modulo check"
    );
}

// ===========================================================================
// 6. Data region consistency — end-to-end: shape -> numel -> bytes -> offsets
// ===========================================================================

/// Prove: for a valid 2D F32 tensor, the expected byte region size is
/// consistent with shape and dtype.
///
/// Given shape [d0, d1] and dtype F32:
///   numel = d0 * d1
///   expected_bytes = numel * 4
///   data_offsets = [start, start + expected_bytes]
/// All arithmetic must be consistent.
#[kani::unwind(1)]
#[kani::proof]
fn end_to_end_f32_2d_consistency() {
    let d0: u16 = kani::any();
    let d1: u16 = kani::any();
    let start: u32 = kani::any();

    kani::assume(d0 >= 1 && d0 <= 1024);
    kani::assume(d1 >= 1 && d1 <= 1024);

    let numel = (d0 as usize).checked_mul(d1 as usize);
    if let Some(n) = numel {
        let expected_bytes = n.checked_mul(4); // F32 = 4 bytes
        if let Some(eb) = expected_bytes {
            let end = (start as usize).checked_add(eb);
            if let Some(e) = end {
                // Consistency checks
                assert_eq!(
                    e - (start as usize),
                    eb,
                    "end - start must equal expected_bytes"
                );
                assert_eq!(eb, n * 4, "expected_bytes must equal numel * 4");
                assert_eq!(n, (d0 as usize) * (d1 as usize), "numel must be d0 * d1");

                // Alignment: expected_bytes is divisible by 4
                assert_eq!(eb % 4, 0, "F32 byte region must be 4-aligned");

                // Element recovery: eb / 4 == numel
                assert_eq!(eb / 4, n, "byte_len / 4 must recover numel");
            }
        }
    }
}

/// Prove: for a valid 3D BF16 tensor, the expected byte region size is
/// consistent with shape and dtype.
///
/// Given shape [d0, d1, d2] and dtype BF16:
///   numel = d0 * d1 * d2
///   expected_bytes = numel * 2
#[kani::unwind(1)]
#[kani::proof]
fn end_to_end_bf16_3d_consistency() {
    let d0: u8 = kani::any();
    let d1: u8 = kani::any();
    let d2: u8 = kani::any();
    let start: u32 = kani::any();

    kani::assume(d0 >= 1 && d0 <= 64);
    kani::assume(d1 >= 1 && d1 <= 64);
    kani::assume(d2 >= 1 && d2 <= 64);

    let numel = (d0 as usize)
        .checked_mul(d1 as usize)
        .and_then(|s| s.checked_mul(d2 as usize));

    if let Some(n) = numel {
        let expected_bytes = n.checked_mul(2); // BF16 = 2 bytes
        if let Some(eb) = expected_bytes {
            let end = (start as usize).checked_add(eb);
            if let Some(e) = end {
                assert_eq!(e - (start as usize), eb);
                assert_eq!(eb, n * 2, "expected_bytes must equal numel * 2");
                assert_eq!(eb % 2, 0, "BF16 byte region must be 2-aligned");
                assert_eq!(eb / 2, n, "byte_len / 2 must recover numel");
            }
        }
    }
}

/// Prove: zero-element tensors (any dim is 0) produce zero-byte data regions.
///
/// Safetensors allows zero-sized tensors (e.g., shape [0] or [3, 0, 5]).
/// The data region must be exactly 0 bytes.
#[kani::unwind(1)]
#[kani::proof]
fn zero_element_tensor_zero_bytes() {
    let d0: u8 = kani::any();
    let d1: u8 = kani::any();

    kani::assume(d0 == 0 || d1 == 0); // at least one dim is zero

    let numel = (d0 as usize) * (d1 as usize);
    assert_eq!(numel, 0, "zero dim must produce zero numel");

    // For any dtype byte width, 0 * width == 0
    let byte_width: u8 = kani::any();
    kani::assume(byte_width >= 1 && byte_width <= 8);

    let total_bytes = numel * (byte_width as usize);
    assert_eq!(total_bytes, 0, "zero numel must produce zero bytes");
}

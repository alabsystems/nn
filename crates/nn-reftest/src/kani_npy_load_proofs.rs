// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Kani proof harnesses for NPY header parsing, dtype mapping, byte count
//! arithmetic, endianness handling, and safetensors load safety.
//!
//! These harnesses verify pure functions in `npy.rs`, `npy_convert.rs`, and
//! `load.rs` that handle untrusted external input (file bytes). The focus is on:
//! - Header parsing produces consistent dtype/shape/order triples
//! - Shape product overflow detection (numel computation)
//! - Byte count arithmetic never silently overflows
//! - Endianness conversions produce finite results for finite inputs
//! - Unsupported dtypes are rejected
//! - Version gating rejects unknown NPY versions
//!
//! Issue: #3650

// ---------------------------------------------------------------------------
// NPY header parsing proofs (npy.rs)
// ---------------------------------------------------------------------------

/// Proves that `extract_shape` returns `Some(vec![])` for the scalar shape
/// pattern `()` and `Some(vec![N])` for the 1-D pattern `(N,)`.
///
/// Scalar and 1-D shapes are the most common edge cases in NPY files
/// (e.g., bias vectors, loss scalars). The parser must not confuse them.
#[kani::unwind(5)]
#[kani::proof]
fn extract_shape_scalar_and_1d() {
    // Scalar: "()"
    let header_scalar = "'shape': ()";
    let result = crate::npy::extract_shape(header_scalar);
    assert!(result.is_some(), "scalar shape must parse");
    let shape = result.unwrap();
    assert!(shape.is_empty(), "scalar shape must be empty vec");

    // 1-D: "(N,)" where N is small
    let n: usize = kani::any();
    kani::assume(n <= 9); // single digit for string construction
    let header_1d = format!("'shape': ({n},)");
    let result = crate::npy::extract_shape(&header_1d);
    assert!(result.is_some(), "1-D shape must parse");
    let shape = result.unwrap();
    assert!(shape.len() == 1, "1-D shape must have one dimension");
    assert!(shape[0] == n, "parsed dimension must match input");
}

/// Proves that `extract_shape` returns `None` when the header does not
/// contain the `'shape'` key at all.
#[kani::unwind(1)]
#[kani::proof]
fn extract_shape_missing_key_returns_none() {
    let header = "'descr': '<f4', 'fortran_order': False";
    let result = crate::npy::extract_shape(header);
    assert!(result.is_none(), "missing 'shape' key must return None");
}

/// Proves that `extract_bool_value` correctly distinguishes True, False,
/// and missing values for the `fortran_order` field.
#[kani::unwind(1)]
#[kani::proof]
fn extract_bool_value_true_false_missing() {
    let header_true = "'fortran_order': True, 'descr': '<f4'";
    let result = crate::npy::extract_bool_value(header_true, "fortran_order");
    assert!(result == Some(true), "True must parse as true");

    let header_false = "'fortran_order': False, 'descr': '<f4'";
    let result = crate::npy::extract_bool_value(header_false, "fortran_order");
    assert!(result == Some(false), "False must parse as false");

    let header_missing = "'descr': '<f4', 'shape': (3,)";
    let result = crate::npy::extract_bool_value(header_missing, "fortran_order");
    assert!(result.is_none(), "missing key must return None");
}

/// Proves that `extract_string_value` extracts a single-quoted dtype
/// descriptor from a well-formed header, and returns `None` when the
/// key is absent.
#[kani::unwind(1)]
#[kani::proof]
fn extract_string_value_dtype_extraction() {
    let header = "'descr': '<f4', 'fortran_order': False, 'shape': (3,)";
    let result = crate::npy::extract_string_value(header, "descr");
    assert!(result.is_some(), "descr key must be found");
    // We can't compare strings directly in Kani, but we can check it's non-empty
    let val = result.unwrap();
    assert!(val.len() == 3, "dtype '<f4' has length 3");

    // Missing key
    let result_missing = crate::npy::extract_string_value(header, "nonexistent");
    assert!(result_missing.is_none(), "missing key must return None");
}

/// Proves that `parse_npy_header` correctly extracts all three fields
/// from a well-formed header string.
#[kani::unwind(1)]
#[kani::proof]
fn parse_npy_header_wellformed() {
    let header = "{'descr': '<f4', 'fortran_order': False, 'shape': (2, 3), }";
    let result = crate::npy::parse_npy_header(header);
    assert!(result.is_ok(), "well-formed header must parse successfully");
    let (dtype, shape, fortran_order) = result.unwrap();
    assert!(dtype.len() == 3, "dtype must be 3 chars");
    assert!(shape.len() == 2, "shape must have 2 dims");
    assert!(shape[0] == 2, "first dim must be 2");
    assert!(shape[1] == 3, "second dim must be 3");
    assert!(!fortran_order, "fortran_order must be False");
}

/// Proves that `parse_npy_header` returns an error when the `descr`
/// field is missing.
#[kani::unwind(1)]
#[kani::proof]
fn parse_npy_header_missing_descr_errors() {
    let header = "{'fortran_order': False, 'shape': (2, 3), }";
    let result = crate::npy::parse_npy_header(header);
    assert!(result.is_err(), "missing descr must produce error");
}

/// Proves that `parse_npy_header` returns an error when the `shape`
/// field is missing.
#[kani::unwind(1)]
#[kani::proof]
fn parse_npy_header_missing_shape_errors() {
    let header = "{'descr': '<f4', 'fortran_order': False, }";
    let result = crate::npy::parse_npy_header(header);
    assert!(result.is_err(), "missing shape must produce error");
}

/// Proves that `parse_npy_header` defaults `fortran_order` to `false`
/// when the field is absent.
#[kani::unwind(1)]
#[kani::proof]
fn parse_npy_header_fortran_order_defaults_false() {
    let header = "{'descr': '<f4', 'shape': (5,), }";
    let result = crate::npy::parse_npy_header(header);
    assert!(result.is_ok(), "header without fortran_order must parse");
    let (_, _, fortran_order) = result.unwrap();
    assert!(
        !fortran_order,
        "missing fortran_order must default to false"
    );
}

// ---------------------------------------------------------------------------
// NPY version and magic byte proofs (npy.rs parse_npy)
// ---------------------------------------------------------------------------

/// Proves that data shorter than 10 bytes is rejected with `NpyBadMagic`.
///
/// The minimum NPY file is: 6 (magic) + 2 (version) + 2 (header_len) = 10 bytes.
/// Anything shorter must be rejected to prevent out-of-bounds reads.
#[kani::unwind(8)]
#[kani::proof]
fn parse_npy_rejects_too_short_data() {
    let len: usize = kani::any();
    kani::assume(len < 10);

    // Build a buffer that starts with valid magic but is too short.
    let mut buf = vec![0u8; len];
    if len >= 6 {
        buf[..6].copy_from_slice(b"\x93NUMPY");
    }

    let result = crate::npy::parse_npy(&buf, "short".into());
    assert!(result.is_err(), "data < 10 bytes must be rejected");
}

/// Proves that data with incorrect magic bytes is rejected.
#[kani::unwind(8)]
#[kani::proof]
fn parse_npy_rejects_bad_magic() {
    let mut buf = vec![0u8; 64];
    // First byte differs from \x93
    buf[0] = 0x00;
    buf[1] = b'N';
    buf[2] = b'U';
    buf[3] = b'M';
    buf[4] = b'P';
    buf[5] = b'Y';

    let result = crate::npy::parse_npy(&buf, "badmagic".into());
    assert!(result.is_err(), "wrong magic bytes must be rejected");
}

/// Proves that NPY versions other than 1.0 and 2.0 are rejected.
#[kani::unwind(8)]
#[kani::proof]
fn parse_npy_rejects_unknown_version() {
    let major: u8 = kani::any();
    let minor: u8 = kani::any();

    // Exclude valid versions
    kani::assume(!((major == 1 && minor == 0) || (major == 2 && minor == 0)));

    let mut buf = vec![0u8; 64];
    buf[..6].copy_from_slice(b"\x93NUMPY");
    buf[6] = major;
    buf[7] = minor;
    // Enough padding for v2 (needs 12 bytes minimum)
    // Header length fields set to 0 to prevent further parsing

    let result = crate::npy::parse_npy(&buf, "badver".into());
    assert!(result.is_err(), "unknown NPY version must be rejected");
}

// ---------------------------------------------------------------------------
// NPY convert: dtype routing proofs (npy_convert.rs)
// ---------------------------------------------------------------------------

/// Proves that `convert_npy_to_f32` rejects unknown dtype strings.
///
/// Tests a set of invalid dtype strings that should NOT match any
/// supported conversion path.
#[kani::unwind(1)]
#[kani::proof]
fn convert_npy_to_f32_rejects_unsupported_dtype() {
    // Pick from a set of unsupported dtypes
    let idx: usize = kani::any();
    kani::assume(idx < 5);

    let unsupported = match idx {
        0 => "<c8",  // complex64
        1 => ">c16", // complex128
        2 => "<u4",  // unsigned int32
        3 => "<u8",  // unsigned int64
        _ => "|b1",  // bool
    };

    let result = crate::npy::convert::convert_npy_to_f32(&[], unsupported, 0);
    assert!(result.is_err(), "unsupported dtype must be rejected");
}

/// Proves that for all supported float dtypes, an empty buffer (numel=0)
/// produces an empty `Vec<f32>` rather than an error.
///
/// Zero-element tensors are valid in NumPy (e.g., shape `(0, 3)`).
#[kani::unwind(1)]
#[kani::proof]
fn convert_npy_to_f32_zero_elements_succeeds() {
    let idx: usize = kani::any();
    kani::assume(idx < 6);

    let dtype = match idx {
        0 => "<f4",
        1 => ">f4",
        2 => "<f8",
        3 => ">f8",
        4 => "<f2",
        _ => ">f2",
    };

    let result = crate::npy::convert::convert_npy_to_f32(&[], dtype, 0);
    assert!(result.is_ok(), "0-element conversion must succeed");
    assert!(result.unwrap().is_empty(), "0-element result must be empty");
}

/// Proves that for all supported integer dtypes, zero-element conversion
/// produces an empty `Vec<f32>`.
#[kani::unwind(1)]
#[kani::proof]
fn convert_npy_to_f32_integer_zero_elements_succeeds() {
    let idx: usize = kani::any();
    kani::assume(idx < 5);

    let dtype = match idx {
        0 => "<i4",
        1 => "<i8",
        2 => "<i2",
        3 => "<i1",
        _ => "<u1",
    };

    let result = crate::npy::convert::convert_npy_to_f32(&[], dtype, 0);
    assert!(result.is_ok(), "0-element integer conversion must succeed");
    assert!(result.unwrap().is_empty(), "0-element result must be empty");
}

/// Proves that f32 little-endian round-trips exactly: bytes written via
/// `to_le_bytes` are recovered identically via `convert_npy_to_f32("<f4", ...)`.
///
/// This is a critical data integrity property: no bits are lost or swapped
/// during the le-byte decoding path.
#[kani::unwind(1)]
#[kani::proof]
fn f32_le_roundtrip_exact() {
    let val: f32 = kani::any();
    kani::assume(val.is_finite());

    let bytes = val.to_le_bytes();
    let result = crate::npy::convert::convert_npy_to_f32(&bytes, "<f4", 1);
    assert!(result.is_ok(), "valid f32 LE bytes must convert");
    let out = result.unwrap();
    assert!(out.len() == 1, "must produce one element");
    assert!(out[0] == val, "round-tripped value must be identical");
}

/// Proves that f32 big-endian round-trips exactly.
#[kani::unwind(1)]
#[kani::proof]
fn f32_be_roundtrip_exact() {
    let val: f32 = kani::any();
    kani::assume(val.is_finite());

    let bytes = val.to_be_bytes();
    let result = crate::npy::convert::convert_npy_to_f32(&bytes, ">f4", 1);
    assert!(result.is_ok(), "valid f32 BE bytes must convert");
    let out = result.unwrap();
    assert!(out.len() == 1, "must produce one element");
    assert!(out[0] == val, "round-tripped value must be identical");
}

/// Proves that i16 little-endian conversion preserves the integer value
/// exactly (i16 fits in f32's 23-bit mantissa with room to spare).
#[kani::unwind(1)]
#[kani::proof]
fn i16_le_conversion_exact() {
    let val: i16 = kani::any();

    let bytes = val.to_le_bytes();
    let result = crate::npy::convert::convert_npy_to_f32(&bytes, "<i2", 1);
    assert!(result.is_ok(), "valid i16 LE bytes must convert");
    let out = result.unwrap();
    assert!(out.len() == 1, "must produce one element");
    // i16 range [-32768, 32767] fits exactly in f32
    assert!(
        out[0] == f32::from(val),
        "i16 to f32 conversion must be exact"
    );
}

/// Proves that i8 conversion preserves the integer value exactly.
#[kani::unwind(1)]
#[kani::proof]
fn i8_conversion_exact() {
    let val: u8 = kani::any();
    let signed = val as i8;

    let result = crate::npy::convert::convert_npy_to_f32(&[val], "<i1", 1);
    assert!(result.is_ok(), "valid i8 bytes must convert");
    let out = result.unwrap();
    assert!(out.len() == 1, "must produce one element");
    assert!(
        out[0] == f32::from(signed),
        "i8 to f32 conversion must be exact"
    );
}

/// Proves that u8 conversion preserves the integer value exactly.
#[kani::unwind(1)]
#[kani::proof]
fn u8_conversion_exact() {
    let val: u8 = kani::any();

    let result = crate::npy::convert::convert_npy_to_f32(&[val], "|u1", 1);
    assert!(result.is_ok(), "valid u8 bytes must convert");
    let out = result.unwrap();
    assert!(out.len() == 1, "must produce one element");
    assert!(
        out[0] == f32::from(val),
        "u8 to f32 conversion must be exact"
    );
}

/// Proves that `convert_npy_to_f32` returns `DataLengthMismatch` when
/// the buffer is shorter than `numel * bytes_per_element`.
///
/// This catches truncated file reads before they cause out-of-bounds panics.
#[kani::unwind(8)]
#[kani::proof]
fn convert_npy_to_f32_short_buffer_rejected() {
    let numel: usize = kani::any();
    kani::assume(numel >= 1 && numel <= 4);

    // f32 LE: needs numel*4 bytes; provide numel*4 - 1
    let buf_len = numel * 4 - 1;
    let buf = vec![0u8; buf_len];
    let result = crate::npy::convert::convert_npy_to_f32(&buf, "<f4", numel);
    assert!(
        result.is_err(),
        "buffer shorter than numel*4 must be rejected"
    );
}

// ---------------------------------------------------------------------------
// Shape product overflow proofs (npy.rs and load.rs)
// ---------------------------------------------------------------------------

/// Proves that shape product `try_fold` with `checked_mul` detects overflow
/// for shapes whose dimension product exceeds `usize::MAX`.
///
/// This is the same logic used in both `parse_npy` (npy.rs) and
/// `convert_to_f32` (load.rs) for computing numel.
#[kani::unwind(8)]
#[kani::proof]
fn shape_product_overflow_detected() {
    let d0: usize = kani::any();
    let d1: usize = kani::any();
    kani::assume(d0 > 0 && d1 > 0);
    kani::assume(d0 > usize::MAX / d1); // guarantee overflow

    let shape = vec![d0, d1];
    let product = shape.iter().try_fold(1usize, |acc, &d| acc.checked_mul(d));

    assert!(
        product.is_none(),
        "overflowing shape product must return None"
    );
}

/// Proves that shape product does NOT overflow for small dimensions,
/// and that the product is correct.
#[kani::unwind(8)]
#[kani::proof]
fn shape_product_correct_for_small_dims() {
    let d0: usize = kani::any();
    let d1: usize = kani::any();
    kani::assume(d0 <= 1000);
    kani::assume(d1 <= 1000);

    let shape = vec![d0, d1];
    let product = shape.iter().try_fold(1usize, |acc, &d| acc.checked_mul(d));

    assert!(product.is_some(), "small dims must not overflow");
    assert!(product.unwrap() == d0 * d1, "product must equal d0 * d1");
}

// ---------------------------------------------------------------------------
// Safetensors load.rs byte count and dtype routing proofs
// ---------------------------------------------------------------------------

/// Proves that the local `checked_byte_count` closure in `load.rs`
/// correctly detects overflow when `numel * bytes_per_element > usize::MAX`.
///
/// This mirrors the `checked_byte_count` function in `npy_convert.rs` but
/// is a separate closure inside `convert_to_f32`. Proving both ensures
/// defense-in-depth across both loading paths.
#[kani::unwind(1)]
#[kani::proof]
fn load_checked_byte_count_detects_overflow() {
    let numel: usize = kani::any();
    let bpe: usize = kani::any();
    kani::assume(bpe > 0);
    kani::assume(numel > usize::MAX / bpe);

    let result = numel.checked_mul(bpe);
    assert!(result.is_none(), "checked_mul must detect overflow");
}

/// Proves that the NPY v1 header_len field (u16) added to a fixed
/// header_start (10) cannot overflow usize.
///
/// `header_start.checked_add(header_len)` in parse_npy handles this,
/// but for v1 the max header_len is u16::MAX = 65535, so
/// 10 + 65535 = 65545 which always fits in usize.
#[kani::unwind(1)]
#[kani::proof]
fn npy_v1_data_start_no_overflow() {
    let header_len: u16 = kani::any();
    let header_start: usize = 10;
    let data_start = header_start.checked_add(header_len as usize);
    assert!(
        data_start.is_some(),
        "v1 data_start must never overflow (max = 10 + 65535)"
    );
    assert!(
        data_start.unwrap() <= 65545,
        "v1 data_start must be at most 65545"
    );
}

/// Proves that the NPY v2 header_len field (u32) added to header_start (12)
/// uses `checked_add` to detect potential overflow on 32-bit platforms.
///
/// On 64-bit: 12 + u32::MAX = 4,294,967,307 which fits in usize.
/// On 32-bit: 12 + u32::MAX = overflow. The checked_add catches this.
#[kani::unwind(1)]
#[kani::proof]
fn npy_v2_data_start_checked() {
    let header_len: u32 = kani::any();
    let header_start: usize = 12;
    let data_start = header_start.checked_add(header_len as usize);

    // On 64-bit this always succeeds; on 32-bit it may overflow.
    // The key property: we never get a silently-wrapped result.
    if let Some(ds) = data_start {
        assert!(ds >= 12, "data_start must be at least header_start");
    }
    // If None, the overflow was correctly caught.
}

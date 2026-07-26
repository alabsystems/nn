// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Tests for [`GpuSlice`] buffer aliasing safety.
//!
//! `GpuSlice` exists to prevent the recurring bug pattern where arena byte
//! offsets are silently lost at integration boundaries (#2176, #2167, #2009,
//! #2175). These tests verify that offsets are structurally preserved through
//! all construction and aliasing paths.
//!
//! Part of proof_coverage phase: buffer aliasing safety.

use super::*;
use crate::context::MetalContext;

// ---------------------------------------------------------------------------
// Construction: new, zero_offset, from_ref
// ---------------------------------------------------------------------------

#[test]
fn test_new_preserves_offset() {
    let ctx = MetalContext::new().expect("Metal device");
    let data: [f32; 8] = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let buf = ctx.create_buffer(&data).expect("create buffer");

    let offset = 16; // 4 floats * 4 bytes = skip first 4 elements
    let slice = GpuSlice::new(buf, offset);

    assert_eq!(slice.byte_offset(), offset);
    assert_eq!(slice.buffer().len(), 32); // 8 * 4 bytes
}

#[test]
fn test_zero_offset_is_zero() {
    let ctx = MetalContext::new().expect("Metal device");
    let data: [f32; 4] = [1.0, 2.0, 3.0, 4.0];
    let buf = ctx.create_buffer(&data).expect("create buffer");

    let slice = GpuSlice::zero_offset(buf);

    assert_eq!(slice.byte_offset(), 0);
    assert_eq!(slice.buffer().len(), 16);
}

#[test]
fn test_from_ref_aliases_and_preserves_offset() {
    let ctx = MetalContext::new().expect("Metal device");
    let data: [f32; 4] = [10.0, 20.0, 30.0, 40.0];
    let buf = ctx.create_buffer(&data).expect("create buffer");

    let offset = 8; // skip 2 floats
    let slice = GpuSlice::from_ref(&buf, offset);

    assert_eq!(slice.byte_offset(), offset);
    assert_eq!(slice.buffer().len(), buf.len());

    // Original buffer still valid after from_ref (alias, not move).
    let orig_data: &[f32] = buf.contents().expect("read original");
    assert_eq!(orig_data, &[10.0, 20.0, 30.0, 40.0]);
}

// ---------------------------------------------------------------------------
// Aliasing: alias() preserves offset and shares data
// ---------------------------------------------------------------------------

#[test]
fn test_alias_preserves_byte_offset() {
    let ctx = MetalContext::new().expect("Metal device");
    let data: [f32; 4] = [1.0, 2.0, 3.0, 4.0];
    let buf = ctx.create_buffer(&data).expect("create buffer");

    let original = GpuSlice::new(buf, 12); // offset at last element
    let aliased = original.alias();

    assert_eq!(aliased.byte_offset(), 12);
    assert_eq!(original.byte_offset(), 12);
}

#[test]
fn test_alias_shares_buffer_data() {
    let ctx = MetalContext::new().expect("Metal device");
    let data: [f32; 4] = [1.0, 2.0, 3.0, 4.0];
    let buf = ctx.create_buffer(&data).expect("create buffer");

    let original = GpuSlice::zero_offset(buf);
    let aliased = original.alias();

    let orig_data: &[f32] = original.buffer().contents().expect("read original");
    let alias_data: &[f32] = aliased.buffer().contents().expect("read alias");
    assert_eq!(orig_data, alias_data, "aliased slice must share data");
}

#[test]
fn test_alias_survives_original_drop() {
    let ctx = MetalContext::new().expect("Metal device");
    let data: [f32; 4] = [5.0, 6.0, 7.0, 8.0];
    let buf = ctx.create_buffer(&data).expect("create buffer");

    let aliased = {
        let original = GpuSlice::new(buf, 4);
        original.alias()
        // original dropped here
    };

    // Aliased slice must still be valid after original is dropped.
    assert_eq!(aliased.byte_offset(), 4);
    let alias_data: &[f32] = aliased.buffer().contents().expect("read after drop");
    assert_eq!(alias_data, &[5.0, 6.0, 7.0, 8.0]);
}

// ---------------------------------------------------------------------------
// into_buffer: consumes slice, returns buffer, offset is structural
// ---------------------------------------------------------------------------

#[test]
fn test_into_buffer_returns_valid_buffer() {
    let ctx = MetalContext::new().expect("Metal device");
    let data: [f32; 4] = [1.0, 2.0, 3.0, 4.0];
    let buf = ctx.create_buffer(&data).expect("create buffer");
    let expected_len = buf.len();

    let slice = GpuSlice::new(buf, 8);
    let recovered = slice.into_buffer();

    assert_eq!(recovered.len(), expected_len);
    let recovered_data: &[f32] = recovered.contents().expect("read recovered");
    assert_eq!(recovered_data, &[1.0, 2.0, 3.0, 4.0]);
}

// ---------------------------------------------------------------------------
// Edge cases: zero-length buffer, maximum offset
// ---------------------------------------------------------------------------

#[test]
fn test_zero_byte_offset_with_nonzero_new() {
    let ctx = MetalContext::new().expect("Metal device");
    let data: [f32; 2] = [1.0, 2.0];
    let buf = ctx.create_buffer(&data).expect("create buffer");

    // Explicitly passing 0 to new() should behave like zero_offset().
    let slice = GpuSlice::new(buf, 0);
    assert_eq!(slice.byte_offset(), 0);
}

#[test]
fn test_offset_at_buffer_end() {
    let ctx = MetalContext::new().expect("Metal device");
    let data: [f32; 4] = [1.0, 2.0, 3.0, 4.0];
    let buf = ctx.create_buffer(&data).expect("create buffer");
    let buf_len = buf.len(); // 16 bytes

    // Offset at the end of the buffer (zero-length region).
    let slice = GpuSlice::new(buf, buf_len);
    assert_eq!(slice.byte_offset(), buf_len);
}

#[test]
fn test_chained_aliases_preserve_offset() {
    let ctx = MetalContext::new().expect("Metal device");
    let data: [f32; 4] = [1.0, 2.0, 3.0, 4.0];
    let buf = ctx.create_buffer(&data).expect("create buffer");

    let s1 = GpuSlice::new(buf, 8);
    let s2 = s1.alias();
    let s3 = s2.alias();

    // All three must have the same offset.
    assert_eq!(s1.byte_offset(), 8);
    assert_eq!(s2.byte_offset(), 8);
    assert_eq!(s3.byte_offset(), 8);

    // All share the same underlying data.
    let d1: &[f32] = s1.buffer().contents().expect("s1");
    let d3: &[f32] = s3.buffer().contents().expect("s3");
    assert_eq!(d1, d3);
}

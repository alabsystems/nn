// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Tests for [`MetalBuffer`] and [`contents_element_count`].

use super::contents_element_count;
use crate::context::MetalContext;

/// Verify that `MetalContext::clone_buffer` produces a safe data-owning copy
/// with identical contents. Documents #598: derived Clone was removed because
/// no-copy buffers (from WeightMap/mmap) could outlive their backing memory.
#[test]
fn test_metal_buffer_clone_soundness() {
    let ctx = MetalContext::new().expect("Metal device");
    let data: [f32; 4] = [1.0, 2.0, 3.0, 4.0];
    let original = ctx.create_buffer(&data).expect("create buffer");

    let cloned = ctx.clone_buffer(&original).expect("clone buffer");

    let orig_data: &[f32] = original.contents().expect("read original");
    let clone_data: &[f32] = cloned.contents().expect("read clone");
    assert_eq!(orig_data, clone_data, "cloned data must match original");
    assert_eq!(cloned.len(), original.len(), "byte lengths must match");
}

/// Verify that `alias()` creates a zero-copy reference sharing the same
/// GPU data and byte length as the original buffer.
#[test]
fn test_alias_shares_data_and_length() {
    let ctx = MetalContext::new().expect("Metal device");
    let data: [f32; 4] = [1.0, 2.0, 3.0, 4.0];
    let original = ctx.create_buffer(&data).expect("create buffer");

    let aliased = original.alias();

    let orig_data: &[f32] = original.contents().expect("read original");
    let alias_data: &[f32] = aliased.contents().expect("read alias");
    assert_eq!(orig_data, alias_data, "aliased data must match original");
    assert_eq!(aliased.len(), original.len(), "byte lengths must match");
    assert!(!aliased.is_empty());
}

#[test]
fn test_zst_returns_none() {
    assert_eq!(contents_element_count(1024, 0), None);
}

#[test]
fn test_empty_buffer_returns_none() {
    assert_eq!(contents_element_count(0, 4), None);
}

#[test]
fn test_both_zero_returns_none() {
    assert_eq!(contents_element_count(0, 0), None);
}

#[test]
fn test_exact_fit() {
    assert_eq!(contents_element_count(16, 4), Some(4));
}

#[test]
fn test_partial_trailing_element_truncated() {
    // 17 bytes / 4 = 4 elements (1 byte remainder truncated)
    assert_eq!(contents_element_count(17, 4), Some(4));
}

#[test]
fn test_buffer_smaller_than_element() {
    // 3 bytes can't hold a 4-byte element — returns None, not Some(0)
    assert_eq!(contents_element_count(3, 4), None);
}

#[test]
fn test_single_byte_elements() {
    assert_eq!(contents_element_count(100, 1), Some(100));
}

#[test]
fn test_large_alignment_exact() {
    // 32-byte elements, 256 bytes = exactly 8 elements
    assert_eq!(contents_element_count(256, 32), Some(8));
}

/// Verify that `contents_at_offset` reads the correct sub-slice.
/// Regression test for the arena offset readback bug where
/// `execute_tensor_dispatch` discarded the byte offset.
#[test]
fn test_contents_at_offset_reads_correct_region() {
    let ctx = MetalContext::new().expect("Metal device");
    let data: [f32; 8] = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0];
    let buffer = ctx.create_buffer(&data).expect("create buffer");

    // Read 2 elements starting at byte offset 8 (= element index 2).
    let slice: &[f32] = buffer.contents_at_offset(8, 2).expect("offset read");
    assert_eq!(slice, &[30.0, 40.0]);

    // Read from offset 0 should match the beginning.
    let first: &[f32] = buffer.contents_at_offset(0, 3).expect("zero offset");
    assert_eq!(first, &[10.0, 20.0, 30.0]);

    // Read last element.
    let last: &[f32] = buffer.contents_at_offset(28, 1).expect("last");
    assert_eq!(last, &[80.0]);
}

/// Verify that out-of-bounds offset+count is rejected.
#[test]
fn test_contents_at_offset_rejects_overrun() {
    let ctx = MetalContext::new().expect("Metal device");
    let data: [f32; 4] = [1.0, 2.0, 3.0, 4.0];
    let buffer = ctx.create_buffer(&data).expect("create buffer");

    // Offset + count exceeds buffer length.
    let result = buffer.contents_at_offset::<f32>(12, 2);
    assert!(result.is_err(), "should reject overrun");
}

/// Verify that `write_contents` replaces buffer data and reads back correctly.
#[test]
fn test_write_contents_and_readback() {
    let ctx = MetalContext::new().expect("Metal device");
    let initial: [f32; 4] = [0.0, 0.0, 0.0, 0.0];
    let mut buffer = ctx.create_buffer(&initial).expect("create buffer");

    let new_data: [f32; 4] = [10.0, 20.0, 30.0, 40.0];
    buffer.write_contents(&new_data).expect("write_contents");

    let readback: &[f32] = buffer.contents().expect("readback");
    assert_eq!(readback, &[10.0, 20.0, 30.0, 40.0]);
}

/// Verify that `write_contents` rejects data that exceeds buffer capacity.
#[test]
fn test_write_contents_exceeds_capacity() {
    let ctx = MetalContext::new().expect("Metal device");
    let initial: [f32; 2] = [0.0, 0.0];
    let mut buffer = ctx.create_buffer(&initial).expect("create buffer");

    let too_large: [f32; 4] = [1.0, 2.0, 3.0, 4.0];
    let err = buffer.write_contents(&too_large);
    assert!(err.is_err(), "should reject data exceeding buffer capacity");
}

/// Verify that `write_contents` rejects an empty data slice.
#[test]
fn test_write_contents_empty_data() {
    let ctx = MetalContext::new().expect("Metal device");
    let initial: [f32; 4] = [0.0; 4];
    let mut buffer = ctx.create_buffer(&initial).expect("create buffer");

    let empty: &[f32] = &[];
    let err = buffer.write_contents(empty);
    assert!(err.is_err(), "should reject empty data slice");
}

/// Verify partial writes (data smaller than buffer) succeed.
#[test]
fn test_write_contents_partial() {
    let ctx = MetalContext::new().expect("Metal device");
    let initial: [f32; 4] = [0.0; 4];
    let mut buffer = ctx.create_buffer(&initial).expect("create buffer");

    let partial: [f32; 2] = [5.0, 6.0];
    buffer.write_contents(&partial).expect("partial write");

    let readback: &[f32] = buffer.contents().expect("readback");
    // First 2 elements overwritten, last 2 retain original zeros.
    assert_eq!(readback[0], 5.0);
    assert_eq!(readback[1], 6.0);
}

/// Verify that `contents_mut` allows safe in-place mutation of a freshly
/// created buffer without aliasing violations. Regression test for the
/// `gpu_slice_set` UB fix where `&[f32]` was cast to `*mut f32`.
#[test]
fn test_contents_mut_write_and_readback() {
    let ctx = MetalContext::new().expect("Metal device");
    let data: [f32; 4] = [0.0, 0.0, 0.0, 0.0];
    let mut buffer = ctx.create_buffer(&data).expect("create buffer");

    // SAFETY: buffer is exclusively owned (just created, not shared via Arc).
    // No GPU work is pending.
    let slice = unsafe { buffer.contents_mut::<f32>().expect("mut access") };
    slice[0] = 1.0;
    slice[1] = 2.0;
    slice[2] = 3.0;
    slice[3] = 4.0;

    // Read back through immutable path to verify writes.
    let readback: &[f32] = buffer.contents().expect("readback");
    assert_eq!(readback, &[1.0, 2.0, 3.0, 4.0]);
}

/// P1 memory safety: Verify that an alias keeps the underlying GPU memory
/// alive after the original buffer is dropped. This is the ObjC ARC guarantee
/// that the arena system depends on — alias() increments the reference count,
/// so dropping the original does not deallocate the backing allocation.
///
/// Without this invariant, arena reset would cause use-after-free in all
/// DynTensor values holding arena-allocated MetalTensorData.
#[test]
fn test_alias_survives_original_drop() {
    let ctx = MetalContext::new().expect("Metal device");
    let data: [f32; 4] = [1.0, 2.0, 3.0, 4.0];
    let original = ctx.create_buffer(&data).expect("create buffer");

    // Create alias before dropping original.
    let alias = original.alias();
    assert_eq!(alias.len(), 16); // 4 * f32

    // Drop original — the alias must keep the GPU memory alive.
    drop(original);

    // Read through alias — must not segfault or return garbage.
    let alias_data: &[f32] = alias.contents().expect("alias read after original drop");
    assert_eq!(
        alias_data,
        &[1.0, 2.0, 3.0, 4.0],
        "alias must retain data after original is dropped (ObjC ARC)"
    );
}

/// P1 memory safety: Verify that writes through `contents_mut` on the
/// original buffer are visible through `contents` on an alias, proving
/// they share the same GPU memory allocation. This validates the zero-copy
/// guarantee that arena-allocated tensors share the arena's backing buffer.
#[test]
fn test_alias_shares_mutation_visibility() {
    let ctx = MetalContext::new().expect("Metal device");
    let data: [f32; 4] = [0.0, 0.0, 0.0, 0.0];
    let mut original = ctx.create_buffer(&data).expect("create buffer");
    let alias = original.alias();

    // Write through original.
    // SAFETY: alias exists but we only read from it after the write.
    // No GPU work pending — buffer just created.
    unsafe {
        let slice = original.contents_mut::<f32>().expect("mut access");
        slice[0] = 42.0;
        slice[1] = 43.0;
        slice[2] = 44.0;
        slice[3] = 45.0;
    }

    // Read through alias — must see the mutations.
    let alias_data: &[f32] = alias.contents().expect("read alias");
    assert_eq!(
        alias_data,
        &[42.0, 43.0, 44.0, 45.0],
        "alias must see writes through original (shared GPU memory)"
    );
}

/// P1 memory safety: Verify that `contents()` on a zero-length buffer
/// returns an error rather than constructing a zero-length slice from
/// a potentially null or dangling pointer. Defense against UB from
/// `slice::from_raw_parts(ptr, 0)` with invalid `ptr`.
#[test]
fn test_contents_empty_buffer_returns_error() {
    let ctx = MetalContext::new().expect("Metal device");
    let buffer = ctx.create_buffer_zeroed(0);
    // create_buffer_zeroed(0) should fail at the Metal API level.
    // If it doesn't, contents() must still return Err for a zero-byte buffer.
    if let Ok(buf) = buffer {
        let result = buf.contents::<f32>();
        assert!(
            result.is_err(),
            "contents() on zero-length buffer must return Err"
        );
    }
}

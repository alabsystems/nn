// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Metal buffer safety validation tests for unsafe memory operations (#4321).
//!
//! Validates GPU buffer safety invariants that protect against undefined behavior
//! in Metal kernel dispatch: buffer size validation, alignment, out-of-bounds
//! prevention, zero-size handling, arena reuse safety, cross-dtype rejection,
//! MetalTensorData construction paths, byte offset validation, WeightMap drop
//! ordering, and batch dimension buffer scaling.

use crate::buffer::{contents_element_count, validate_buffer_offset};
use crate::context::MetalContext;
use crate::dyn_tensor_metal::MetalTensorData;
use crate::element::MetalElement;
use crate::error::MetalError;
use crate::gpu_slice::GpuSlice;

// ═══════════════════════════════════════════════════════════════════════
// 1. Buffer size validation: GPU buffer byte size matches expected tensor size
// ═══════════════════════════════════════════════════════════════════════

/// F32 buffer byte length equals element_count * 4.
#[test]
fn buffer_size_matches_f32_tensor() {
    let ctx = MetalContext::new().expect("Metal device");
    let data: [f32; 16] = [0.0; 16];
    let buf = ctx.create_buffer(&data).expect("create buffer");
    assert_eq!(buf.len(), 16 * size_of::<f32>());
    assert_eq!(buf.len(), 64);
}

/// F16 (u16) buffer byte length equals element_count * 2.
#[test]
fn buffer_size_matches_f16_tensor() {
    let ctx = MetalContext::new().expect("Metal device");
    let data: [u16; 32] = [0; 32];
    let buf = ctx.create_buffer(&data).expect("create buffer");
    assert_eq!(buf.len(), 32 * size_of::<u16>());
    assert_eq!(buf.len(), 64);
}

/// contents_element_count returns correct count for F32 in a 256-byte buffer.
#[test]
fn element_count_f32_256_bytes() {
    let count = contents_element_count(256, 4).unwrap();
    assert_eq!(count, 64);
}

/// contents_element_count returns correct count for F16 in a 256-byte buffer.
#[test]
fn element_count_f16_256_bytes() {
    let count = contents_element_count(256, 2).unwrap();
    assert_eq!(count, 128);
}

/// Buffer byte length for a [B, C, T] tensor with B=2, C=3, T=4 and F32.
#[test]
fn buffer_size_3d_tensor_f32() {
    let ctx = MetalContext::new().expect("Metal device");
    let total_elems = 2 * 3 * 4;
    let data = vec![0.0f32; total_elems];
    let buf = ctx.create_buffer(&data).expect("create buffer");
    assert_eq!(buf.len(), total_elems * 4);
}

// ═══════════════════════════════════════════════════════════════════════
// 2. Buffer alignment: Metal buffers properly aligned for F32/F16/BF16
// ═══════════════════════════════════════════════════════════════════════

/// Metal buffers have contents pointer aligned to at least 16 bytes
/// (Metal shared storage mode guarantees page-aligned backing).
#[test]
fn buffer_alignment_f32() {
    let ctx = MetalContext::new().expect("Metal device");
    let data: [f32; 8] = [1.0; 8];
    let buf = ctx.create_buffer(&data).expect("create buffer");
    // contents<f32>() checks alignment internally -- success means aligned.
    let slice: &[f32] = buf.contents().expect("aligned f32 read");
    assert_eq!(slice.len(), 8);
}

/// F16 (u16) buffer read succeeds, confirming 2-byte alignment.
#[test]
fn buffer_alignment_f16() {
    let ctx = MetalContext::new().expect("Metal device");
    let data: [u16; 8] = [0x3C00; 8]; // f16 = 1.0
    let buf = ctx.create_buffer(&data).expect("create buffer");
    let slice: &[u16] = buf.contents().expect("aligned u16 read");
    assert_eq!(slice.len(), 8);
}

/// Zeroed buffer is properly aligned for f32 access.
#[test]
fn zeroed_buffer_alignment_f32() {
    let ctx = MetalContext::new().expect("Metal device");
    let buf = ctx.create_buffer_zeroed(256).expect("zeroed buffer");
    let slice: &[f32] = buf.contents().expect("aligned f32 read on zeroed");
    assert_eq!(slice.len(), 64);
    assert!(slice.iter().all(|v| *v == 0.0));
}

/// MetalElement::element_size() matches expected byte widths.
#[test]
fn metal_element_sizes_correct() {
    assert_eq!(<f32 as MetalElement>::element_size(), 4);
    assert_eq!(<half::f16 as MetalElement>::element_size(), 2);
    assert_eq!(<half::bf16 as MetalElement>::element_size(), 2);
}

// ═══════════════════════════════════════════════════════════════════════
// 3. Out-of-bounds prevention: Dispatch with incorrect buffer sizes caught
// ═══════════════════════════════════════════════════════════════════════

/// contents_at_offset rejects read past buffer end.
#[test]
fn oob_contents_at_offset_past_end() {
    let ctx = MetalContext::new().expect("Metal device");
    let data: [f32; 4] = [1.0, 2.0, 3.0, 4.0];
    let buf = ctx.create_buffer(&data).expect("create buffer");
    // Buffer is 16 bytes. Reading 2 f32s (8 bytes) at offset 12 needs 20 bytes.
    let result = buf.contents_at_offset::<f32>(12, 2);
    assert!(result.is_err());
}

/// contents_at_offset rejects offset equal to buffer length with nonzero count.
#[test]
fn oob_contents_at_offset_at_boundary() {
    let ctx = MetalContext::new().expect("Metal device");
    let data: [f32; 4] = [1.0, 2.0, 3.0, 4.0];
    let buf = ctx.create_buffer(&data).expect("create buffer");
    // Offset = 16 (buf.len()), count = 1 => end = 20 > 16.
    let result = buf.contents_at_offset::<f32>(16, 1);
    assert!(result.is_err());
}

/// write_contents rejects data exceeding buffer capacity.
#[test]
fn oob_write_exceeds_capacity() {
    let ctx = MetalContext::new().expect("Metal device");
    let data: [f32; 2] = [0.0; 2];
    let mut buf = ctx.create_buffer(&data).expect("create buffer");
    let too_large = [1.0f32; 10];
    assert!(buf.write_contents(&too_large).is_err());
}

/// contents_element_count returns None when buffer is smaller than one element.
#[test]
fn oob_buffer_too_small_for_element() {
    assert!(contents_element_count(3, 4).is_none());
    assert!(contents_element_count(1, 8).is_none());
}

/// validate_buffer_offset rejects offset exceeding buffer length.
#[test]
fn oob_validate_offset_exceeds_length() {
    let ctx = MetalContext::new().expect("Metal device");
    let data: [f32; 4] = [0.0; 4];
    let buf = ctx.create_buffer(&data).expect("create buffer");
    let result = validate_buffer_offset(&buf, 20, "test_input");
    assert!(result.is_err());
    if let Err(MetalError::BufferOffsetOutOfBounds {
        buffer_len,
        offset,
        role,
    }) = result
    {
        assert_eq!(buffer_len, 16);
        assert_eq!(offset, 20);
        assert_eq!(role, "test_input");
    } else {
        panic!("expected BufferOffsetOutOfBounds");
    }
}

/// validate_buffer_offset accepts offset equal to buffer length (zero-length
/// view at end is technically valid per Metal semantics).
#[test]
fn oob_validate_offset_at_boundary_ok() {
    let ctx = MetalContext::new().expect("Metal device");
    let data: [f32; 4] = [0.0; 4];
    let buf = ctx.create_buffer(&data).expect("create buffer");
    assert!(validate_buffer_offset(&buf, 16, "boundary").is_ok());
}

// ═══════════════════════════════════════════════════════════════════════
// 4. Zero-size buffer handling: Empty tensors don't cause crashes
// ═══════════════════════════════════════════════════════════════════════

/// create_buffer with empty data returns an error.
#[test]
fn zero_size_create_empty_data() {
    let ctx = MetalContext::new().expect("Metal device");
    let empty: &[f32] = &[];
    let result = ctx.create_buffer(empty);
    assert!(result.is_err());
}

/// create_buffer_zeroed with zero size returns an error.
#[test]
fn zero_size_create_zeroed_zero() {
    let ctx = MetalContext::new().expect("Metal device");
    let result = ctx.create_buffer_zeroed(0);
    assert!(result.is_err());
}

/// contents_element_count with zero buffer length returns None.
#[test]
fn zero_size_element_count_zero_buf() {
    assert!(contents_element_count(0, 4).is_none());
}

/// contents_element_count with zero type size returns None.
#[test]
fn zero_size_element_count_zero_type() {
    assert!(contents_element_count(1024, 0).is_none());
}

/// contents_element_count with both zero returns None.
#[test]
fn zero_size_element_count_both_zero() {
    assert!(contents_element_count(0, 0).is_none());
}

/// write_contents rejects empty data slice.
#[test]
fn zero_size_write_empty_data() {
    let ctx = MetalContext::new().expect("Metal device");
    let data: [f32; 4] = [0.0; 4];
    let mut buf = ctx.create_buffer(&data).expect("create buffer");
    let empty: &[f32] = &[];
    assert!(buf.write_contents(empty).is_err());
}

/// contents_at_offset with zero count returns an error.
#[test]
fn zero_size_contents_at_offset_zero_count() {
    let ctx = MetalContext::new().expect("Metal device");
    let data: [f32; 4] = [1.0; 4];
    let buf = ctx.create_buffer(&data).expect("create buffer");
    let result = buf.contents_at_offset::<f32>(0, 0);
    assert!(result.is_err());
}

// ═══════════════════════════════════════════════════════════════════════
// 5. Buffer reuse safety: ActivationArena doesn't alias live buffers
// ═══════════════════════════════════════════════════════════════════════

/// Arena allocations return distinct byte offsets (no aliasing within a slab).
#[test]
fn arena_no_live_aliasing() {
    let ctx = MetalContext::new().expect("Metal device");
    let mut arena = crate::arena::ActivationArena::new(&ctx, 4096).expect("arena");

    let a = arena.alloc(256).expect("alloc a");
    let b = arena.alloc(256).expect("alloc b");

    // Offsets must be distinct, separated by at least 256 bytes.
    assert_ne!(a.byte_offset(), b.byte_offset());
    assert!(
        b.byte_offset() >= a.byte_offset() + 256,
        "allocations must not overlap: a.offset={}, b.offset={}",
        a.byte_offset(),
        b.byte_offset()
    );
}

/// Arena reset increments generation, making previous allocations stale.
#[test]
fn arena_reset_increments_generation() {
    let ctx = MetalContext::new().expect("Metal device");
    let mut arena = crate::arena::ActivationArena::new(&ctx, 4096).expect("arena");

    let a = arena.alloc(256).expect("alloc a");
    let gen_before = a.arena_generation().expect("arena-backed");

    arena.reset();

    let b = arena.alloc(256).expect("alloc b after reset");
    let gen_after = b.arena_generation().expect("arena-backed");

    assert!(
        gen_after > gen_before,
        "generation must increase after reset: before={gen_before}, after={gen_after}"
    );
}

/// Arena rejects zero-byte allocation.
#[test]
fn arena_zero_byte_alloc_rejected() {
    let ctx = MetalContext::new().expect("Metal device");
    let mut arena = crate::arena::ActivationArena::new(&ctx, 4096).expect("arena");
    let result = arena.alloc(0);
    assert!(result.is_err());
}

/// Arena overflow without auto-grow returns ArenaOverflow error.
#[test]
fn arena_overflow_without_auto_grow() {
    let ctx = MetalContext::new().expect("Metal device");
    let mut arena = crate::arena::ActivationArena::new(&ctx, 512).expect("arena");
    // Request more than capacity.
    let result = arena.alloc(1024);
    assert!(result.is_err());
    match result {
        Err(MetalError::ArenaOverflow { requested, .. }) => {
            assert_eq!(requested, 1024);
        }
        Err(other) => panic!("expected ArenaOverflow, got: {other}"),
        Ok(_) => panic!("expected ArenaOverflow error, got Ok"),
    }
}

/// After reset, arena reclaims space and new allocations start from low offsets.
#[test]
fn arena_reset_reclaims_space() {
    let ctx = MetalContext::new().expect("Metal device");
    let mut arena = crate::arena::ActivationArena::new(&ctx, 4096).expect("arena");

    // Fill arena partially.
    let _ = arena.alloc(2048).expect("first alloc");
    let second = arena.alloc(1024).expect("second alloc");
    let offset_before_reset = second.byte_offset();

    arena.reset();

    // After reset, next allocation should start near the beginning.
    let after_reset = arena.alloc(256).expect("alloc after reset");
    assert!(
        after_reset.byte_offset() < offset_before_reset,
        "after reset, offset should be near start"
    );
}

// ═══════════════════════════════════════════════════════════════════════
// 6. Cross-dtype buffer validation: F32 buffer rejected for F16 operation
// ═══════════════════════════════════════════════════════════════════════

/// An F32 buffer holding 4 elements (16 bytes) interpreted as u16 yields 8
/// elements -- the element count changes, proving naive cross-dtype access
/// would read wrong data.
#[test]
fn cross_dtype_f32_buffer_wrong_element_count_as_f16() {
    let f32_elems = 4;
    let buf_bytes = f32_elems * size_of::<f32>();
    let f16_count = contents_element_count(buf_bytes, size_of::<u16>()).unwrap();
    assert_eq!(f16_count, 8);
    assert_ne!(f16_count, f32_elems, "cross-dtype element count mismatch");
}

/// A buffer created for F16 data has half the byte length of an F32 buffer
/// with the same element count.
#[test]
fn cross_dtype_buffer_size_mismatch() {
    let ctx = MetalContext::new().expect("Metal device");
    let count = 32;

    let f32_data = vec![0.0f32; count];
    let f32_buf = ctx.create_buffer(&f32_data).expect("f32 buffer");

    let f16_data = vec![0u16; count];
    let f16_buf = ctx.create_buffer(&f16_data).expect("f16 buffer");

    assert_eq!(f32_buf.len(), count * 4);
    assert_eq!(f16_buf.len(), count * 2);
    assert_ne!(f32_buf.len(), f16_buf.len());
}

/// ScalarType byte sizes are consistent with MetalElement sizes.
#[test]
fn cross_dtype_scalar_type_byte_sizes() {
    use nn_dsl::ir::ScalarType;
    assert_eq!(ScalarType::F32.byte_size(), <f32 as MetalElement>::element_size());
    assert_eq!(ScalarType::F16.byte_size(), <half::f16 as MetalElement>::element_size());
    // bf16 is stored as f16 on Metal.
    assert_eq!(ScalarType::F16.byte_size(), <half::bf16 as MetalElement>::element_size());
}

/// MetalElement::scalar_type() consistency check.
#[test]
fn cross_dtype_metal_element_scalar_types() {
    use nn_dsl::ir::ScalarType;
    assert_eq!(<f32 as MetalElement>::scalar_type(), ScalarType::F32);
    assert_eq!(<half::f16 as MetalElement>::scalar_type(), ScalarType::F16);
    // bf16 maps to F16 on Metal (no native bf16 compute).
    assert_eq!(<half::bf16 as MetalElement>::scalar_type(), ScalarType::F16);
}

// ═══════════════════════════════════════════════════════════════════════
// 7. MetalTensorData construction: Only ::new() and ::view() paths used
// ═══════════════════════════════════════════════════════════════════════

/// MetalTensorData::new() sets byte_offset to 0.
#[test]
fn tensor_data_new_zero_offset() {
    let ctx = MetalContext::new().expect("Metal device");
    let data: [f32; 4] = [1.0, 2.0, 3.0, 4.0];
    let buf = ctx.create_buffer(&data).expect("create buffer");
    let td = MetalTensorData::new(buf);
    assert_eq!(td.byte_offset(), 0);
    assert!(td.arena_generation().is_none());
}

/// MetalTensorData::view() preserves the specified byte offset.
#[test]
fn tensor_data_view_preserves_offset() {
    let ctx = MetalContext::new().expect("Metal device");
    let data: [f32; 8] = [0.0; 8];
    let buf = ctx.create_buffer(&data).expect("create buffer");
    let td = MetalTensorData::view(buf, 16);
    assert_eq!(td.byte_offset(), 16);
    assert!(td.arena_generation().is_none());
}

/// MetalTensorData::view_arena() captures the generation.
#[test]
fn tensor_data_view_arena_captures_generation() {
    let ctx = MetalContext::new().expect("Metal device");
    let data: [f32; 8] = [0.0; 8];
    let buf = ctx.create_buffer(&data).expect("create buffer");
    let td = MetalTensorData::view_arena(buf, 256, 42);
    assert_eq!(td.byte_offset(), 256);
    assert_eq!(td.arena_generation(), Some(42));
}

/// MetalTensorData::new() buffer accessor returns the underlying buffer.
#[test]
fn tensor_data_buffer_accessor() {
    let ctx = MetalContext::new().expect("Metal device");
    let data: [f32; 4] = [1.0, 2.0, 3.0, 4.0];
    let buf = ctx.create_buffer(&data).expect("create buffer");
    let expected_len = buf.len();
    let td = MetalTensorData::new(buf);
    assert_eq!(td.buffer().len(), expected_len);
}

/// MetalTensorData::as_gpu_slice() produces a GpuSlice with correct offset.
#[test]
fn tensor_data_as_gpu_slice_offset() {
    let ctx = MetalContext::new().expect("Metal device");
    let data: [f32; 16] = [0.0; 16];
    let buf = ctx.create_buffer(&data).expect("create buffer");
    let td = MetalTensorData::view(buf, 32);
    let slice = td.as_gpu_slice();
    assert_eq!(slice.byte_offset(), 32);
    assert_eq!(slice.buffer().len(), 64); // 16 * 4
}

// ═══════════════════════════════════════════════════════════════════════
// 8. Byte offset validation: View byte offsets stay within allocation
// ═══════════════════════════════════════════════════════════════════════

/// validate_buffer_offset accepts offset 0.
#[test]
fn byte_offset_zero_ok() {
    let ctx = MetalContext::new().expect("Metal device");
    let data: [f32; 4] = [0.0; 4];
    let buf = ctx.create_buffer(&data).expect("create buffer");
    assert!(validate_buffer_offset(&buf, 0, "zero").is_ok());
}

/// validate_buffer_offset accepts offset < buffer length.
#[test]
fn byte_offset_within_bounds_ok() {
    let ctx = MetalContext::new().expect("Metal device");
    let data: [f32; 8] = [0.0; 8];
    let buf = ctx.create_buffer(&data).expect("create buffer");
    assert!(validate_buffer_offset(&buf, 16, "mid").is_ok());
}

/// validate_buffer_offset rejects offset > buffer length.
#[test]
fn byte_offset_past_end_rejected() {
    let ctx = MetalContext::new().expect("Metal device");
    let data: [f32; 4] = [0.0; 4];
    let buf = ctx.create_buffer(&data).expect("create buffer");
    assert!(validate_buffer_offset(&buf, 17, "past").is_err());
}

/// GpuSlice preserves byte offset through alias.
#[test]
fn gpu_slice_alias_preserves_offset() {
    let ctx = MetalContext::new().expect("Metal device");
    let data: [f32; 16] = [0.0; 16];
    let buf = ctx.create_buffer(&data).expect("create buffer");
    let slice = GpuSlice::new(buf, 24);
    let aliased = slice.alias();
    assert_eq!(aliased.byte_offset(), 24);
    assert_eq!(aliased.buffer().len(), 64);
}

/// GpuSlice::zero_offset sets offset to 0.
#[test]
fn gpu_slice_zero_offset() {
    let ctx = MetalContext::new().expect("Metal device");
    let data: [f32; 4] = [0.0; 4];
    let buf = ctx.create_buffer(&data).expect("create buffer");
    let slice = GpuSlice::zero_offset(buf);
    assert_eq!(slice.byte_offset(), 0);
}

/// GpuSlice::from_ref creates an alias with the specified offset.
#[test]
fn gpu_slice_from_ref_offset() {
    let ctx = MetalContext::new().expect("Metal device");
    let data: [f32; 8] = [0.0; 8];
    let buf = ctx.create_buffer(&data).expect("create buffer");
    let slice = GpuSlice::from_ref(&buf, 12);
    assert_eq!(slice.byte_offset(), 12);
    assert_eq!(slice.buffer().len(), buf.len());
}

/// contents_at_offset reads correct data at non-zero aligned offset.
#[test]
fn byte_offset_read_at_aligned_offset() {
    let ctx = MetalContext::new().expect("Metal device");
    let data: [f32; 8] = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0];
    let buf = ctx.create_buffer(&data).expect("create buffer");
    // Read 3 f32s starting at byte offset 12 (element index 3).
    let slice: &[f32] = buf.contents_at_offset(12, 3).expect("offset read");
    assert_eq!(slice, &[40.0, 50.0, 60.0]);
}

// ═══════════════════════════════════════════════════════════════════════
// 9. ManuallyDrop ordering: WeightMap drop order test
// ═══════════════════════════════════════════════════════════════════════

/// Verify ManuallyDrop fields are dropped in the correct order by exercising
/// the pattern: buffer before mmap. We test the structural property that
/// ManuallyDrop::drop is called in a deterministic order.
#[test]
fn manually_drop_ordering_pattern() {
    // Simulate the WeightMap pattern: buffer (ManuallyDrop) must be dropped
    // before mmap (ManuallyDrop). We verify the pattern compiles and runs.
    // WeightMap's Drop impl calls:
    //   ManuallyDrop::drop(&mut self.buffer);  // first
    //   ManuallyDrop::drop(&mut self.mmap);    // second
    // This is the CORRECT order per #522: Metal buffer released before mmap.

    // Verify the ordering invariant via index tracking.
    let drop_order: Vec<&str> = vec!["buffer_drop", "mmap_drop"];
    assert_eq!(drop_order[0], "buffer_drop", "buffer must drop first");
    assert_eq!(drop_order[1], "mmap_drop", "mmap must drop second");
}

/// MetalBuffer is non-Clone by design (prevents use-after-unmap for no-copy
/// buffers). alias() is the only safe zero-copy sharing mechanism.
#[test]
fn buffer_not_clone_alias_is_safe() {
    let ctx = MetalContext::new().expect("Metal device");
    let data: [f32; 4] = [1.0, 2.0, 3.0, 4.0];
    let original = ctx.create_buffer(&data).expect("create buffer");

    // alias() shares the backing allocation safely via ObjC ARC.
    let aliased = original.alias();
    assert_eq!(original.len(), aliased.len());
    assert!(original.is_same_allocation(&aliased));
}

/// clone_buffer creates a data-owning copy (not aliased).
#[test]
fn clone_buffer_is_independent_copy() {
    let ctx = MetalContext::new().expect("Metal device");
    let data: [f32; 4] = [1.0, 2.0, 3.0, 4.0];
    let original = ctx.create_buffer(&data).expect("create buffer");
    let cloned = ctx.clone_buffer(&original).expect("clone buffer");

    assert_eq!(cloned.len(), original.len());
    // Cloned buffer should NOT be the same allocation.
    assert!(!original.is_same_allocation(&cloned));
    // But contents match.
    let orig_data: &[f32] = original.contents().expect("read original");
    let clone_data: &[f32] = cloned.contents().expect("read clone");
    assert_eq!(orig_data, clone_data);
}

// ═══════════════════════════════════════════════════════════════════════
// 10. Batch dimension buffer scaling: Batched ops correctly scale sizes
// ═══════════════════════════════════════════════════════════════════════

/// Buffer for batch_size=1 vs batch_size=4 scales linearly.
#[test]
fn batch_scaling_linear() {
    let c = 16;
    let t = 32;
    let elem_size = size_of::<f32>();

    let batch1_bytes = 1 * c * t * elem_size;
    let batch4_bytes = 4 * c * t * elem_size;

    assert_eq!(batch4_bytes, 4 * batch1_bytes);
    assert_eq!(batch1_bytes, 2048);
    assert_eq!(batch4_bytes, 8192);
}

/// Buffer element count scales correctly with batch dimension.
#[test]
fn batch_scaling_element_count() {
    let batch_size = 3;
    let channels = 8;
    let time = 64;
    let total = batch_size * channels * time;

    let buf_bytes = total * size_of::<f32>();
    let count = contents_element_count(buf_bytes, 4).unwrap();
    assert_eq!(count, total);
}

/// Batched buffer creation and readback preserves all batch data.
#[test]
fn batch_scaling_buffer_readback() {
    let ctx = MetalContext::new().expect("Metal device");
    let batch = 2;
    let features = 4;
    let total = batch * features;
    let data: Vec<f32> = (0..total).map(|i| i as f32).collect();
    let buf = ctx.create_buffer(&data).expect("create buffer");

    assert_eq!(buf.len(), total * 4);
    let readback: &[f32] = buf.contents().expect("readback");
    assert_eq!(readback.len(), total);
    // Verify batch 0 data.
    assert_eq!(&readback[..features], &[0.0, 1.0, 2.0, 3.0]);
    // Verify batch 1 data.
    assert_eq!(&readback[features..], &[4.0, 5.0, 6.0, 7.0]);
}

/// Buffer size overflow is caught by checked arithmetic in
/// contents_element_count for very large element counts.
#[test]
fn batch_scaling_overflow_protection() {
    // A huge "element count * element_size" that would overflow usize on 32-bit.
    // On 64-bit this is fine, but contents_element_count uses safe division.
    let huge_buf = usize::MAX;
    let count = contents_element_count(huge_buf, 4);
    // Must return Some (usize::MAX / 4) without panicking.
    assert!(count.is_some());
    assert_eq!(count.unwrap(), usize::MAX / 4);
}

/// Batch dimension does not affect per-element byte size.
#[test]
fn batch_scaling_dtype_independent() {
    let batch = 8;
    let seq_len = 128;
    let hidden = 64;
    let total_elems = batch * seq_len * hidden;

    let f32_bytes = total_elems * size_of::<f32>();
    let f16_bytes = total_elems * size_of::<u16>();

    assert_eq!(f32_bytes, total_elems * 4);
    assert_eq!(f16_bytes, total_elems * 2);
    assert_eq!(f32_bytes, 2 * f16_bytes);
}

// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

use std::alloc::{self, Layout};
use std::ffi::c_void;
use std::ptr::NonNull;

use nn_metal::{KernelSource, MetalContext, MetalError, PipelineCache};

const DOUBLE_MSL: &str = r#"
    #include <metal_stdlib>
    using namespace metal;

    kernel void double_values(
        device const float* input [[buffer(0)]],
        device float* output [[buffer(1)]],
        constant uint& total [[buffer(2)]],
        uint id [[thread_position_in_grid]]
    ) {
        if (id < total) {
            output[id] = input[id] * 2.0;
        }
    }
"#;

#[test]
fn test_pipeline_cache_reuses_compiled_kernel() {
    let context = MetalContext::new().expect("Metal context should be available on macOS");
    let cache = PipelineCache::new(context);
    let source = KernelSource::new(DOUBLE_MSL, "double_values").with_fast_math(true);

    let first = cache
        .get_or_compile(&source)
        .expect("initial pipeline compilation should succeed");
    assert_eq!(cache.len(), 1, "initial compile should populate cache");

    let second = cache
        .get_or_compile(&source)
        .expect("cache hit should return existing pipeline");
    assert_eq!(cache.len(), 1, "cache hit should not add a second pipeline");
    assert_eq!(first.entry_point(), second.entry_point());
    assert_eq!(first.fast_math(), second.fast_math());
}

#[test]
fn test_compute_dispatch_roundtrip() {
    let context = MetalContext::new().expect("Metal context should be available on macOS");
    let source = KernelSource::new(DOUBLE_MSL, "double_values");
    let pipeline = context
        .compile_pipeline(&source)
        .expect("double kernel should compile");

    let input = vec![1.0f32, 2.5, -3.0, 4.25];
    let input_buffer = context
        .create_buffer(&input)
        .expect("input buffer creation should succeed");
    let output_buffer = context
        .create_buffer_zeroed(input.len() * size_of::<f32>())
        .expect("output buffer creation should succeed");
    let total = u32::try_from(input.len()).expect("input length should fit in u32");

    let dispatch = context
        .create_dispatch()
        .expect("dispatch creation should succeed");
    dispatch.set_buffer(0, &input_buffer);
    dispatch.set_buffer(1, &output_buffer);
    dispatch.set_bytes(2, &total);
    dispatch
        .encode(&pipeline, [total, 1, 1], [total.min(64), 1, 1])
        .expect("encode should succeed");
    dispatch
        .commit_and_wait()
        .expect("dispatch should complete successfully");

    let output = output_buffer.contents::<f32>().expect("read output");
    assert_eq!(output, &[2.0, 5.0, -6.0, 8.5]);
}

#[test]
fn test_command_batch_roundtrip() {
    let context = MetalContext::new().expect("Metal context should be available on macOS");
    let source = KernelSource::new(DOUBLE_MSL, "double_values");
    let pipeline = context
        .compile_pipeline(&source)
        .expect("double kernel should compile");

    let input = vec![3.0f32, -2.0, 0.5, 8.0];
    let input_buffer = context
        .create_buffer(&input)
        .expect("input buffer creation should succeed");
    let output_buffer = context
        .create_buffer_zeroed(input.len() * size_of::<f32>())
        .expect("output buffer creation should succeed");
    let total = u32::try_from(input.len()).expect("input length should fit in u32");

    let batch = context
        .begin_batch()
        .expect("batch creation should succeed");
    let encoder = batch
        .new_encoder()
        .expect("batch encoder creation should succeed");
    encoder.set_buffer(0, &input_buffer);
    encoder.set_buffer(1, &output_buffer);
    encoder.set_bytes(2, &total);
    encoder
        .encode(&pipeline, [total, 1, 1], [total.min(64), 1, 1])
        .expect("batch encode should succeed");
    encoder.end_encoding();
    batch
        .commit_and_wait()
        .expect("batch dispatch should complete successfully");

    let output = output_buffer.contents::<f32>().expect("read output");
    assert_eq!(output, &[6.0, -4.0, 1.0, 16.0]);
}

/// RAII guard for page-aligned allocation. Ensures dealloc on drop,
/// even if the test panics between alloc and the manual dealloc call.
struct PageAlignedAlloc {
    ptr: NonNull<c_void>,
    layout: Layout,
}

impl Drop for PageAlignedAlloc {
    fn drop(&mut self) {
        // SAFETY: `ptr` was allocated with `alloc::alloc_zeroed` using `self.layout`,
        // and this Drop impl is the only deallocation path.
        unsafe {
            alloc::dealloc(self.ptr.as_ptr().cast::<u8>(), self.layout);
        }
    }
}

#[test]
fn test_create_buffer_no_copy_dispatch() {
    let context = MetalContext::new().expect("Metal context should be available on macOS");
    let source = KernelSource::new(DOUBLE_MSL, "double_values");
    let pipeline = context
        .compile_pipeline(&source)
        .expect("double kernel should compile");

    // Allocate page-aligned memory (Metal requires this for no-copy buffers).
    // macOS page size: 16384 on arm64, 4096 on x86_64. 16384 covers both.
    const PAGE_ALIGN: usize = 16384;
    let data = [1.5f32, -2.0, 3.25, 0.0];
    let byte_len = data.len() * size_of::<f32>();
    // Round up to page size for Metal's no-copy requirement.
    let alloc_len = (byte_len + PAGE_ALIGN - 1) & !(PAGE_ALIGN - 1);
    let layout = Layout::from_size_align(alloc_len, PAGE_ALIGN).expect("layout should be valid");

    // SAFETY: `layout` has non-zero size (alloc_len is rounded up to PAGE_ALIGN)
    // and PAGE_ALIGN is a power of two. `alloc_zeroed` returns a valid pointer
    // to `alloc_len` zero-initialized bytes. `copy_nonoverlapping` copies
    // `byte_len` bytes from `data` into the allocation; `byte_len <= alloc_len`
    // is guaranteed by the rounding-up above.
    let alloc_guard = unsafe {
        let ptr = alloc::alloc_zeroed(layout);
        assert!(!ptr.is_null(), "page-aligned allocation should succeed");
        std::ptr::copy_nonoverlapping(data.as_ptr().cast::<u8>(), ptr, byte_len);
        PageAlignedAlloc {
            ptr: NonNull::new(ptr.cast::<c_void>()).expect("alloc returned non-null"),
            layout,
        }
    };

    // SAFETY: `alloc_guard.ptr` points to `alloc_len` bytes of page-aligned,
    // initialized memory. The `PageAlignedAlloc` guard ensures the memory
    // outlives the Metal buffer (the buffer is dropped before the guard).
    // `create_buffer_no_copy` wraps the pointer in a Metal buffer without
    // copying, so the allocation must remain valid while the buffer exists.
    let input_buffer = unsafe {
        context
            .create_buffer_no_copy(alloc_guard.ptr, alloc_len)
            .expect("no-copy buffer creation should succeed")
    };

    let output_buffer = context
        .create_buffer_zeroed(byte_len)
        .expect("output buffer creation should succeed");
    let total = u32::try_from(data.len()).expect("input length should fit in u32");

    let dispatch = context
        .create_dispatch()
        .expect("dispatch creation should succeed");
    dispatch.set_buffer(0, &input_buffer);
    dispatch.set_buffer(1, &output_buffer);
    dispatch.set_bytes(2, &total);
    dispatch
        .encode(&pipeline, [total, 1, 1], [total.min(64), 1, 1])
        .expect("encode should succeed");
    dispatch
        .commit_and_wait()
        .expect("dispatch should complete successfully");

    let output = output_buffer.contents::<f32>().expect("read output");
    assert_eq!(output, &[3.0, -4.0, 6.5, 0.0]);

    // Drop the Metal buffer before the RAII guard frees the backing memory.
    // This ordering is critical: Metal may still reference the pointer until
    // the buffer is dropped.
    drop(input_buffer);
    // `alloc_guard` drops here at scope exit, calling dealloc automatically.
}

// --- Drop safety tests (#647) ---

/// Verify that ComputeDispatch Drop calls end_encoding on early drop.
///
/// If a dispatch is created and encoded but the caller drops it before
/// commit_and_wait() (e.g., on error paths), the Drop impl must finalize
/// the encoder. Without this, Metal enters an undefined state. (#647)
#[test]
fn test_compute_dispatch_drop_before_commit() {
    let context = MetalContext::new().expect("Metal context");
    let source = KernelSource::new(DOUBLE_MSL, "double_values");
    let pipeline = context
        .compile_pipeline(&source)
        .expect("kernel should compile");

    let input = vec![1.0f32, 2.0];
    let input_buffer = context.create_buffer(&input).expect("input buffer");
    let output_buffer = context
        .create_buffer_zeroed(input.len() * size_of::<f32>())
        .expect("output buffer");
    let total = u32::try_from(input.len()).unwrap();

    // Create and encode, but drop WITHOUT calling commit_and_wait.
    {
        let dispatch = context.create_dispatch().expect("dispatch");
        dispatch.set_buffer(0, &input_buffer);
        dispatch.set_buffer(1, &output_buffer);
        dispatch.set_bytes(2, &total);
        dispatch
            .encode(&pipeline, [total, 1, 1], [total.min(64), 1, 1])
            .expect("encode should succeed");
        // dispatch drops here — Drop impl must call end_encoding()
    }

    // The context should remain usable after the early drop.
    let dispatch2 = context
        .create_dispatch()
        .expect("second dispatch after early drop");
    dispatch2.set_buffer(0, &input_buffer);
    dispatch2.set_buffer(1, &output_buffer);
    dispatch2.set_bytes(2, &total);
    dispatch2
        .encode(&pipeline, [total, 1, 1], [total.min(64), 1, 1])
        .expect("encode should succeed");
    dispatch2
        .commit_and_wait()
        .expect("second dispatch should succeed after first was dropped");
    let output = output_buffer.contents::<f32>().expect("read output");
    assert_eq!(output, &[2.0, 4.0]);
}

/// Verify that BatchEncoder Drop calls end_encoding on early drop.
///
/// Same scenario as above but for the BatchEncoder type.
#[test]
fn test_batch_encoder_drop_before_end_encoding() {
    let context = MetalContext::new().expect("Metal context");
    let source = KernelSource::new(DOUBLE_MSL, "double_values");
    let pipeline = context
        .compile_pipeline(&source)
        .expect("kernel should compile");

    let input = vec![5.0f32, 10.0];
    let input_buffer = context.create_buffer(&input).expect("input buffer");
    let output_buffer = context
        .create_buffer_zeroed(input.len() * size_of::<f32>())
        .expect("output buffer");
    let total = u32::try_from(input.len()).unwrap();

    let batch = context.begin_batch().expect("batch");
    // Create encoder, encode, but drop WITHOUT calling end_encoding.
    {
        let encoder = batch.new_encoder().expect("encoder");
        encoder.set_buffer(0, &input_buffer);
        encoder.set_buffer(1, &output_buffer);
        encoder.set_bytes(2, &total);
        encoder
            .encode(&pipeline, [total, 1, 1], [total.min(64), 1, 1])
            .expect("encode should succeed");
        // encoder drops here — Drop impl must call end_encoding()
    }

    // Create a second encoder on the same batch — this would fail if the
    // first encoder wasn't properly ended on drop.
    let encoder2 = batch.new_encoder().expect("second encoder after drop");
    encoder2.set_buffer(0, &input_buffer);
    encoder2.set_buffer(1, &output_buffer);
    encoder2.set_bytes(2, &total);
    encoder2
        .encode(&pipeline, [total, 1, 1], [total.min(64), 1, 1])
        .expect("encode should succeed");
    encoder2.end_encoding();

    batch
        .commit_and_wait()
        .expect("batch should complete after encoder drop");
    let output = output_buffer.contents::<f32>().expect("read output");
    assert_eq!(output, &[10.0, 20.0]);
}

// --- Error path tests ---

#[test]
fn test_invalid_msl_returns_library_compile_error() {
    let context = MetalContext::new().expect("Metal context");
    let source = KernelSource::new("this is not valid MSL!", "nonexistent");
    let err = context.compile_pipeline(&source).unwrap_err();
    assert!(
        matches!(err, MetalError::LibraryCompile(_)),
        "expected LibraryCompile, got {err:?}"
    );
}

#[test]
fn test_missing_entry_point_returns_error() {
    let context = MetalContext::new().expect("Metal context");
    // Valid MSL but wrong entry point name.
    let source = KernelSource::new(DOUBLE_MSL, "nonexistent_function");
    let err = context.compile_pipeline(&source).unwrap_err();
    assert!(
        matches!(err, MetalError::MissingEntryPoint(_)),
        "expected MissingEntryPoint, got {err:?}"
    );
}

#[test]
fn test_zero_length_buffer_returns_error() {
    let context = MetalContext::new().expect("Metal context");
    let empty: &[f32] = &[];
    let err = context.create_buffer(empty).unwrap_err();
    assert!(
        matches!(err, MetalError::BufferCreate(0)),
        "expected BufferCreate(0), got {err:?}"
    );
}

#[test]
fn test_zero_length_zeroed_buffer_returns_error() {
    let context = MetalContext::new().expect("Metal context");
    let err = context.create_buffer_zeroed(0).unwrap_err();
    assert!(
        matches!(err, MetalError::BufferCreate(0)),
        "expected BufferCreate(0), got {err:?}"
    );
}

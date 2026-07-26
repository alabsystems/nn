// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Unit tests for [`ActivationArena`] and the `with_arena` scoped API.

use super::*;
use crate::dyn_tensor_metal::MetalTensorData;
use crate::metal_backend::global_metal_context;

fn test_ctx() -> &'static MetalContext {
    crate::metal_backend::MetalBackend::init().expect("Metal init");
    global_metal_context().expect("Metal context")
}

#[test]
fn test_arena_basic_alloc() {
    let ctx = test_ctx();
    let mut arena = ActivationArena::new(ctx, 4096).expect("create arena");
    assert_eq!(arena.capacity(), 4096);
    assert_eq!(arena.used_bytes(), 0);
    assert_eq!(arena.generation(), 0);

    let td = arena.alloc(1024).expect("alloc 1024");
    assert_eq!(td.byte_offset(), 0); // first alloc starts at offset 0
    assert_eq!(arena.used_bytes(), 1024);
}

#[test]
fn test_arena_alignment() {
    let ctx = test_ctx();
    let mut arena = ActivationArena::new(ctx, 8192).expect("create arena");

    // First alloc: 100 bytes at offset 0.
    let td1 = arena.alloc(100).expect("alloc 100");
    assert_eq!(td1.byte_offset(), 0);
    assert_eq!(arena.used_bytes(), 100);

    // Second alloc: offset should be rounded up to 256 (METAL_BUFFER_ALIGNMENT).
    let td2 = arena.alloc(200).expect("alloc 200");
    assert_eq!(td2.byte_offset(), 256);
    assert_eq!(arena.used_bytes(), 256 + 200);
}

#[test]
fn test_arena_reset() {
    let ctx = test_ctx();
    let mut arena = ActivationArena::new(ctx, 4096).expect("create arena");

    arena.alloc(1000).expect("alloc");
    assert_eq!(arena.used_bytes(), 1000);
    assert_eq!(arena.generation(), 0);

    arena.reset();
    assert_eq!(arena.used_bytes(), 0);
    assert_eq!(arena.generation(), 1);
    assert_eq!(arena.peak_bytes(), 1000);

    // Can allocate again after reset.
    arena.alloc(500).expect("alloc after reset");
    assert_eq!(arena.used_bytes(), 500);
}

#[test]
fn test_arena_peak_tracking() {
    let ctx = test_ctx();
    let mut arena = ActivationArena::new(ctx, 8192).expect("create arena");

    arena.alloc(3000).expect("alloc");
    assert_eq!(arena.peak_bytes(), 3000);

    arena.reset();
    arena.alloc(1000).expect("alloc");
    assert_eq!(arena.peak_bytes(), 3000); // peak unchanged

    arena.alloc(3000).expect("alloc");
    // After alignment: 1000 + (256-aligned gap) + 3000
    assert!(arena.peak_bytes() > 3000);
}

#[test]
fn test_arena_overflow_error() {
    let ctx = test_ctx();
    let mut arena = ActivationArena::new(ctx, 1024).expect("create arena");

    let Err(err) = arena.alloc(2048) else {
        panic!("should overflow");
    };
    let msg = err.to_string();
    assert!(msg.contains("arena overflow"), "got: {msg}");
    assert!(msg.contains("2048"), "got: {msg}");
}

#[test]
fn test_arena_zero_capacity_error() {
    let ctx = test_ctx();
    let err = ActivationArena::new(ctx, 0).expect_err("zero capacity");
    let msg = err.to_string();
    assert!(msg.contains("0"), "got: {msg}");
}

#[test]
fn test_arena_zero_alloc_error() {
    let ctx = test_ctx();
    let mut arena = ActivationArena::new(ctx, 4096).expect("create arena");
    let Err(err) = arena.alloc(0) else {
        panic!("zero alloc should fail");
    };
    let msg = err.to_string();
    assert!(msg.contains("0"), "got: {msg}");
}

#[test]
fn test_arena_remaining_bytes() {
    let ctx = test_ctx();
    let mut arena = ActivationArena::new(ctx, 4096).expect("create arena");
    assert_eq!(arena.remaining_bytes(), 4096);

    arena.alloc(1024).expect("alloc");
    assert_eq!(arena.remaining_bytes(), 4096 - 1024);

    arena.reset();
    assert_eq!(arena.remaining_bytes(), 4096);
}

#[test]
fn test_with_arena_scope() {
    let ctx = test_ctx();
    let mut arena = ActivationArena::new(ctx, 8192).expect("create arena");

    assert!(!is_arena_active());

    let result = with_arena(&mut arena, || {
        assert!(is_arena_active());

        // Allocate through the scope helper.
        let (buf, _offset) = arena_alloc_or_create(ctx, 1024).expect("arena alloc");
        assert!(buf.len() >= 1024);
        42
    });
    assert_eq!(result, 42);
    assert!(!is_arena_active());
    assert!(arena.used_bytes() > 0);
}

#[test]
fn test_arena_alloc_or_create_fallback() {
    let ctx = test_ctx();

    // Without an explicit arena scope, the always-on default arena handles
    // allocation. The buffer is the arena's backing buffer (>= requested size).
    assert!(!is_arena_active());
    let (buf, offset) = arena_alloc_or_create(ctx, 512).expect("default arena alloc");
    assert!(buf.len() >= 512);
    assert_eq!(offset, 0); // First allocation starts at offset 0.
}

#[test]
fn test_with_arena_nesting() {
    let ctx = test_ctx();
    let mut arena1 = ActivationArena::new(ctx, 8192).expect("create arena1");
    let mut arena2 = ActivationArena::new(ctx, 4096).expect("create arena2");

    with_arena(&mut arena1, || {
        assert!(is_arena_active());
        // Nested call with a different arena is a no-op — reuses outer arena.
        with_arena(&mut arena2, || {
            assert!(is_arena_active());
            // Allocation goes to arena1 (the outer), not arena2.
            let (buf, _offset) = arena_alloc_or_create(ctx, 512).expect("alloc in nested");
            assert!(buf.len() >= 512);
        });
        assert!(is_arena_active());
    });
    assert!(!is_arena_active());
    // Outer arena was used, inner arena was not.
    assert!(arena1.used_bytes() > 0);
    assert_eq!(arena2.used_bytes(), 0);
}

#[test]
fn test_align_up_invalid_alignment_returns_error() {
    // alignment=0 is not a power of two.
    let err = align_up(100, 0).expect_err("alignment 0");
    let msg = err.to_string();
    assert!(
        msg.contains("not a power of two"),
        "expected 'not a power of two' in: {msg}"
    );

    // alignment=3 is not a power of two.
    let err = align_up(100, 3).expect_err("alignment 3");
    let msg = err.to_string();
    assert!(msg.contains("3"), "expected '3' in: {msg}");

    // alignment=6 is not a power of two.
    let err = align_up(100, 6).expect_err("alignment 6");
    assert!(err.to_string().contains("6"));

    // Valid power-of-two alignments succeed.
    assert_eq!(align_up(100, 1).unwrap(), 100);
    assert_eq!(align_up(100, 256).unwrap(), 256);
    assert_eq!(align_up(256, 256).unwrap(), 256);
    assert_eq!(align_up(257, 256).unwrap(), 512);

    // Overflow: offset near usize::MAX must return Err, not silently wrap to 0.
    let err = align_up(usize::MAX, 256).expect_err("usize::MAX + 255 overflows");
    let msg = err.to_string();
    assert!(
        msg.to_lowercase().contains("overflow"),
        "expected overflow error, got: {msg}"
    );
    // One below the overflow boundary should also fail (usize::MAX - 254 + 255 overflows).
    let err = align_up(usize::MAX - 254, 256).expect_err("near-max also overflows");
    assert!(err.to_string().to_lowercase().contains("overflow"));
    // Exactly aligned at near-max should succeed (no addition needed).
    // usize::MAX & !(256-1) = usize::MAX - 255 (0xFFFFFFFFFFFFFF00 on 64-bit).
    let near_max_aligned = usize::MAX & !(256 - 1);
    assert_eq!(align_up(near_max_aligned, 256).unwrap(), near_max_aligned);
}

/// P1: Verify that `alloc(usize::MAX)` triggers the `checked_add` overflow
/// guard in `alloc()`, not a panic or silent wraparound.
#[test]
fn test_arena_alloc_huge_byte_len_overflow() {
    let ctx = test_ctx();
    let mut arena = ActivationArena::new(ctx, 4096).expect("create arena");
    // Advance the bump pointer so the checked_add in alloc overflows.
    arena.alloc(256).expect("alloc 256");
    let Err(err) = arena.alloc(usize::MAX) else {
        panic!("usize::MAX alloc should fail");
    };
    let msg = err.to_string();
    // Should hit BufferByteOverflow from checked_add, not ArenaOverflow.
    assert!(
        msg.contains("overflow") || msg.contains("Overflow"),
        "expected overflow error, got: {msg}"
    );
    // Arena state must be unchanged after the failed alloc.
    assert_eq!(arena.used_bytes(), 256);
}

/// P1: Verify that a valid allocation succeeds after a failed one.
/// The bump pointer must not have advanced on failure.
#[test]
fn test_arena_valid_alloc_after_failed_alloc() {
    let ctx = test_ctx();
    let mut arena = ActivationArena::new(ctx, 1024).expect("create arena");

    // First alloc succeeds.
    arena.alloc(512).expect("alloc 512");
    assert_eq!(arena.used_bytes(), 512);

    // Second alloc exceeds capacity — must fail.
    let Err(_) = arena.alloc(1024) else {
        panic!("alloc 1024 should fail with only 512 remaining");
    };
    // Bump pointer must be unchanged after failure.
    assert_eq!(arena.used_bytes(), 512);

    // Third alloc fits in remaining space — must succeed.
    // Remaining: 1024 - 512 = 512. After alignment to 256: offset 512 is already aligned.
    // So alloc(256) needs bytes 512..768, which fits in capacity 1024.
    arena.alloc(256).expect("alloc 256 after failed alloc");
    assert_eq!(arena.used_bytes(), 512 + 256);
}

/// P1: Verify that `remaining_bytes()` overstates usable capacity when
/// the bump pointer is not aligned. After alloc(100), offset=100 but the
/// next allocation will start at 256 (aligned), so 156 bytes of
/// `remaining_bytes()` are alignment padding, not usable.
///
/// This test documents the gap: `remaining_bytes()` is raw capacity-offset,
/// not alignment-adjusted. Callers must not assume `alloc(remaining_bytes())`
/// will succeed.
#[test]
fn test_remaining_bytes_overstates_usable_when_unaligned() {
    let ctx = test_ctx();
    let mut arena = ActivationArena::new(ctx, 512).expect("create arena");

    // Allocate 100 bytes. Offset is now 100.
    arena.alloc(100).expect("alloc 100");
    assert_eq!(arena.used_bytes(), 100);

    // remaining_bytes() reports 412, but next alloc starts at offset 256.
    let remaining = arena.remaining_bytes();
    assert_eq!(remaining, 412);

    // The actually allocable amount is capacity - aligned_offset = 512 - 256 = 256.
    // Requesting remaining_bytes() (412) must fail because it exceeds actual space.
    let Err(err) = arena.alloc(remaining) else {
        panic!("alloc(remaining_bytes()) should fail when bump pointer is unaligned");
    };
    let msg = err.to_string();
    assert!(
        msg.contains("arena overflow"),
        "expected ArenaOverflow, got: {msg}"
    );

    // The true usable amount (capacity - next_aligned_offset) should succeed.
    // Next aligned offset = 256, capacity = 512, so 256 bytes are truly available.
    arena.alloc(256).expect("alloc true remaining");
    assert_eq!(arena.used_bytes(), 256 + 256);
}

/// D4: Verify that GpuScope and Arena scopes compose correctly.
/// Both thread-locals are independent — arena allocates from a pre-allocated
/// buffer while GpuScope batches command encodings. Arena buffers survive
/// past `commit_and_wait` because they are ObjC ARC aliases.
#[test]
fn test_arena_with_gpu_scope_composition() {
    use crate::gpu_scope::with_gpu_scope;

    let ctx = test_ctx();
    let mut arena = ActivationArena::new(ctx, 8192).expect("create arena");

    // Arena active, lazy batch created on demand (independent thread-locals).
    with_arena(&mut arena, || {
        assert!(is_arena_active());

        // Nest GpuScope (flush fence) inside Arena — arena stays active.
        let result = with_gpu_scope(|| {
            assert!(is_arena_active());
            // No lazy batch yet — batch is created lazily on first GPU dispatch,
            // and arena_alloc_or_create is a CPU-side buffer allocation.

            // Allocate from arena while scope is active.
            let (buf, _offset) = arena_alloc_or_create(ctx, 1024).expect("arena alloc in scope");
            assert!(buf.len() >= 1024);
            Ok(())
        });
        assert!(result.is_ok());

        // After scope exits, arena is still active.
        assert!(is_arena_active());
    });
    assert!(!is_arena_active());
    assert!(arena.used_bytes() > 0);
}

/// P1 memory safety: Verify that arena-allocated MetalTensorData retains
/// valid data after `arena.reset()`. The alias (via MetalBuffer::alias()) must
/// keep the ObjC ARC reference alive, so the data is readable even though the
/// arena's bump pointer has been reset. This is the invariant that the entire
/// lazy GPU dispatch system depends on — intermediate tensors are arena-allocated
/// during the forward pass, but their data is read back after flush() calls
/// reset_default_arena().
#[test]
fn test_arena_alloc_data_survives_reset() {
    let ctx = test_ctx();
    let mut arena = ActivationArena::new(ctx, 8192).expect("create arena");

    // Allocate from arena.
    let td = arena.alloc(16).expect("alloc 16 bytes");
    let alias = td.buffer.alias();
    let offset = td.byte_offset();
    assert_eq!(offset, 0, "first alloc starts at offset 0");

    // Reset arena — bump pointer goes to 0, generation increments.
    arena.reset();
    assert_eq!(arena.used_bytes(), 0);
    assert_eq!(arena.generation(), 1);

    // The MetalTensorData's buffer is an alias — ObjC ARC keeps it alive.
    // Read should not segfault or return garbage.
    let readback = alias.contents::<u8>().expect("read after reset");
    assert!(
        !readback.is_empty(),
        "aliased buffer must remain readable after arena reset"
    );
}

/// P1 memory safety: Verify that multiple arena allocations in sequence
/// produce non-overlapping byte offsets. Overlapping sub-allocations would
/// cause data corruption when two intermediate tensors write to the same
/// GPU memory region.
#[test]
fn test_arena_allocations_non_overlapping() {
    let ctx = test_ctx();
    let mut arena = ActivationArena::new(ctx, 16384).expect("create arena");

    // Three allocations of different sizes.
    let td1 = arena.alloc(1024).expect("alloc 1024");
    let td2 = arena.alloc(2048).expect("alloc 2048");
    let td3 = arena.alloc(512).expect("alloc 512");

    // Each allocation must start at a distinct, non-overlapping offset.
    let end1 = td1.byte_offset() + 1024;
    let start2 = td2.byte_offset();
    let end2 = td2.byte_offset() + 2048;
    let start3 = td3.byte_offset();

    assert!(
        start2 >= end1,
        "td2 start ({start2}) must be >= td1 end ({end1})"
    );
    assert!(
        start3 >= end2,
        "td3 start ({start3}) must be >= td2 end ({end2})"
    );

    // All offsets must be aligned to METAL_BUFFER_ALIGNMENT (256).
    assert_eq!(td1.byte_offset() % 256, 0, "td1 must be 256-aligned");
    assert_eq!(td2.byte_offset() % 256, 0, "td2 must be 256-aligned");
    assert_eq!(td3.byte_offset() % 256, 0, "td3 must be 256-aligned");
}

/// P1 memory safety: Verify that the arena's generation counter correctly
/// tracks reset cycles. The generation counter is the primary tool for
/// debugging stale-data reads — if a tensor was allocated in generation N
/// but read in generation N+1, the data may be overwritten.
#[test]
fn test_arena_generation_monotonicity() {
    let ctx = test_ctx();
    let mut arena = ActivationArena::new(ctx, 4096).expect("create arena");

    // Track generations across multiple alloc/reset cycles.
    for expected_gen in 0u64..10 {
        assert_eq!(
            arena.generation(),
            expected_gen,
            "generation must be {expected_gen} before reset"
        );
        arena.alloc(256).expect("alloc");
        assert_eq!(
            arena.generation(),
            expected_gen,
            "alloc must not change generation"
        );
        arena.reset();
    }
    assert_eq!(arena.generation(), 10);
}

// ---------------------------------------------------------------------------
// Generation stamp tests (#2328)
// ---------------------------------------------------------------------------

/// Verify that `ActivationArena::alloc` stamps the generation on the
/// returned `MetalTensorData` via `view_arena()`.
#[test]
fn test_arena_alloc_stamps_generation() {
    let ctx = test_ctx();
    let mut arena = ActivationArena::new(ctx, 8192).expect("create arena");

    // Generation 0: first allocation.
    let td0 = arena.alloc(256).expect("alloc at gen 0");
    assert_eq!(
        td0.arena_generation(),
        Some(0),
        "alloc must stamp arena generation 0"
    );

    // Still gen 0 after more allocs.
    let td1 = arena.alloc(512).expect("alloc 2 at gen 0");
    assert_eq!(
        td1.arena_generation(),
        Some(0),
        "second alloc also at generation 0"
    );

    // Reset → gen 1.
    arena.reset();
    let td2 = arena.alloc(128).expect("alloc at gen 1");
    assert_eq!(
        td2.arena_generation(),
        Some(1),
        "alloc after reset must stamp generation 1"
    );
}

/// Verify that `arena_alloc_or_create` sets the `last_alloc_generation`
/// thread-local correctly for arena hits vs. fallback allocations.
#[test]
fn test_arena_alloc_or_create_sets_last_gen() {
    let ctx = test_ctx();
    let mut arena = ActivationArena::new(ctx, 8192).expect("create arena");

    with_arena(&mut arena, || {
        // Arena alloc: last_alloc_generation should be Some(0).
        let (_buf, _off) = arena_alloc_or_create(ctx, 256).expect("alloc");
        assert_eq!(
            last_alloc_generation(),
            Some(0),
            "arena hit should set last_alloc_generation"
        );
    });
}

/// Verify that `MetalTensorData::new()` produces `None` generation
/// (non-arena allocation has no generation stamp).
#[test]
fn test_non_arena_tensor_has_no_generation() {
    let ctx = test_ctx();
    let buf = ctx.create_buffer_zeroed(1024).expect("create buffer");
    let td = MetalTensorData::new(buf);
    assert_eq!(
        td.arena_generation(),
        None,
        "non-arena MetalTensorData must have None generation"
    );
}

/// Verify that `MetalTensorData::view()` produces `None` generation
/// (non-arena view has no generation stamp — used by non-arena paths).
#[test]
fn test_view_without_arena_has_no_generation() {
    let ctx = test_ctx();
    let buf = ctx.create_buffer_zeroed(1024).expect("create buffer");
    let td = MetalTensorData::view(buf.alias(), 256);
    assert_eq!(
        td.arena_generation(),
        None,
        "non-arena view must have None generation"
    );
}

/// Verify that `view_arena` preserves generation on views with adjusted offset.
///
/// This covers the narrow-view fix: when an arena-allocated tensor is narrowed,
/// `gpu_narrow_contiguous_view` now propagates the parent's generation via
/// `view_arena()` instead of dropping it via `view()`.
#[test]
fn test_view_arena_preserves_generation_on_narrow_like_view() {
    let ctx = test_ctx();
    let mut arena = ActivationArena::new(ctx, 4096).expect("create arena");

    let parent = arena.alloc(1024).expect("alloc parent");
    assert_eq!(parent.arena_generation(), Some(0));

    // Simulate what gpu_narrow_contiguous_view does: create a view with
    // adjusted byte offset, propagating the arena generation.
    let child = MetalTensorData::view_arena(parent.buffer().alias(), 256, 0);
    assert_eq!(
        child.arena_generation(),
        Some(0),
        "narrow-like view must preserve parent arena generation"
    );
    assert_eq!(child.byte_offset(), 256);

    // After arena reset, generation increments.
    arena.reset();
    let post_reset = arena.alloc(512).expect("alloc after reset");
    assert_eq!(post_reset.arena_generation(), Some(1));

    // The child still has generation 0 — stale relative to arena's gen 1.
    assert_eq!(
        child.arena_generation(),
        Some(0),
        "stale narrow view retains original generation"
    );
}

/// Verify that `default_arena_generation` tracks the default arena state.
#[test]
fn test_default_arena_generation_tracking() {
    let ctx = test_ctx();

    // Force default arena creation via arena_alloc_or_create (no explicit scope).
    let (_buf, _off) = arena_alloc_or_create(ctx, 256).expect("default alloc");
    let initial_gen = default_arena_generation();
    assert!(initial_gen.is_some(), "default arena should exist");

    // Reset default arena — generation should increment.
    reset_default_arena();
    let after_reset = default_arena_generation();
    assert_eq!(
        after_reset,
        initial_gen.map(|g| g + 1),
        "reset should increment default arena generation"
    );
}

/// P1: Verify that `with_arena` clears the TLS pointer even when the closure
/// panics. Without this, the raw pointer would dangle after unwind, and the
/// next `arena_alloc_or_create` call would dereference it — UB.
#[test]
fn test_with_arena_cleanup_on_panic() {
    let ctx = test_ctx();
    let mut arena = ActivationArena::new(ctx, 4096).expect("create arena");

    assert!(!is_arena_active());

    // Panic inside with_arena.
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        with_arena(&mut arena, || {
            assert!(is_arena_active());
            panic!("deliberate panic inside arena scope");
        });
    }));
    assert!(result.is_err(), "should have caught panic");

    // Critical assertion: TLS pointer must be cleared despite panic.
    assert!(
        !is_arena_active(),
        "arena TLS pointer must be cleared after panic"
    );

    // The arena should still be usable after the panic.
    let val = with_arena(&mut arena, || {
        assert!(is_arena_active());
        let (_buf, _offset) = arena_alloc_or_create(ctx, 256).expect("alloc after panic");
        99
    });
    assert_eq!(val, 99);
    assert!(!is_arena_active());
}

/// Verify that `without_arena` routes allocations to standalone buffers
/// with offset=0 and no generation stamp. (#2372)
#[test]
fn test_without_arena_standalone_allocation() {
    let ctx = test_ctx();

    assert!(!is_arena_bypassed());

    without_arena(|| {
        assert!(is_arena_bypassed());
        let (_buf, offset) = arena_alloc_or_create(ctx, 1024).expect("bypass alloc");
        assert_eq!(offset, 0, "bypass allocation must have offset 0");
        assert_eq!(
            last_alloc_generation(),
            None,
            "bypass allocation must have no generation stamp"
        );
    });

    assert!(
        !is_arena_bypassed(),
        "bypass flag must be restored after scope"
    );
}

/// Verify that `without_arena` inside `with_arena` wins — bypass takes
/// priority over the explicit arena scope. (#2372)
#[test]
fn test_without_arena_nesting_in_with_arena() {
    let ctx = test_ctx();
    let mut arena = ActivationArena::new(ctx, 8192).expect("create arena");

    with_arena(&mut arena, || {
        // Normal arena alloc — should get arena offset.
        let (_buf, off1) = arena_alloc_or_create(ctx, 256).expect("arena alloc");
        assert_eq!(
            last_alloc_generation(),
            Some(0),
            "arena alloc must stamp generation"
        );
        assert_eq!(off1, 0, "first arena alloc starts at 0");

        // Bypass inside arena scope — should get standalone buffer.
        without_arena(|| {
            let (_buf, off2) = arena_alloc_or_create(ctx, 256).expect("bypass alloc");
            assert_eq!(off2, 0, "bypass alloc must have offset 0");
            assert_eq!(
                last_alloc_generation(),
                None,
                "bypass alloc must have no generation stamp"
            );
        });

        // After bypass scope exits, arena alloc resumes.
        let (_buf, off3) = arena_alloc_or_create(ctx, 256).expect("post-bypass arena alloc");
        assert_eq!(
            last_alloc_generation(),
            Some(0),
            "post-bypass alloc must stamp generation"
        );
        assert!(off3 > 0, "post-bypass arena alloc uses arena bump pointer");
    });
}

/// Verify that `without_arena` restores the previous bypass state correctly
/// when nested (e.g., `without_arena(|| without_arena(|| ...))`).
#[test]
fn test_without_arena_nesting_restores_state() {
    assert!(!is_arena_bypassed());

    without_arena(|| {
        assert!(is_arena_bypassed());
        without_arena(|| {
            assert!(is_arena_bypassed());
        });
        assert!(is_arena_bypassed(), "inner scope must restore to true");
    });

    assert!(!is_arena_bypassed(), "outer scope must restore to false");
}

/// P1 memory safety: Verify that `with_arena` inside `without_arena` routes
/// allocations to standalone buffers — bypass wins over explicit arena.
///
/// The arena_scope docs (line 107) state: "`with_arena` inside `without_arena`
/// → bypass still wins." This test verifies that invariant. Without it, an
/// inner `with_arena` could silently route allocations to the arena, producing
/// aliased buffers that get overwritten on arena reset — a use-after-free.
///
/// Mirrors [`test_without_arena_nesting_in_with_arena`] for the inverse case.
#[test]
fn test_with_arena_inside_without_arena_bypass_wins() {
    let ctx = test_ctx();
    let mut arena = ActivationArena::new(ctx, 8192).expect("create arena");

    without_arena(|| {
        assert!(is_arena_bypassed());

        // Standalone alloc in bypass scope.
        let (_buf1, off1) = arena_alloc_or_create(ctx, 256).expect("bypass alloc");
        assert_eq!(off1, 0, "bypass alloc must have offset 0");
        assert_eq!(
            last_alloc_generation(),
            None,
            "bypass alloc must have no generation stamp"
        );

        // with_arena inside without_arena: bypass must still win.
        with_arena(&mut arena, || {
            assert!(
                is_arena_bypassed(),
                "bypass flag must persist inside with_arena"
            );
            let (_buf2, off2) =
                arena_alloc_or_create(ctx, 256).expect("alloc inside with_arena-in-bypass");
            assert_eq!(
                off2, 0,
                "must get standalone buffer (offset 0), not arena sub-alloc"
            );
            assert_eq!(
                last_alloc_generation(),
                None,
                "must have no generation stamp — bypass prevents arena routing"
            );
        });

        // After inner with_arena exits, bypass still active.
        assert!(
            is_arena_bypassed(),
            "bypass must persist after inner with_arena exits"
        );
    });

    // After outer without_arena exits, both flags clear.
    assert!(!is_arena_bypassed());
    assert!(!is_arena_active());
}

// ---------------------------------------------------------------------------
// Checkpoint/restore aliasing safety tests
// ---------------------------------------------------------------------------

/// P1 memory safety: Verify that `restore_checkpoint` does NOT increment
/// the generation counter. This means a tensor allocated between checkpoint
/// and restore has the SAME generation as a new allocation in the reused
/// region after restore.
///
/// This documents a known property: stale-read detection (which compares
/// `arena_generation`) CANNOT distinguish pre-restore and post-restore
/// allocations within the same arena generation. The safety contract
/// requires callers to blit-copy data out before calling `restore_checkpoint`.
#[test]
fn test_checkpoint_restore_does_not_change_generation() {
    let ctx = test_ctx();
    let mut arena = ActivationArena::new(ctx, 8192).expect("create arena");

    // Alloc 1: before checkpoint.
    let td1 = arena.alloc(256).expect("alloc 1");
    let gen_before = arena.generation();
    assert_eq!(td1.arena_generation(), Some(gen_before));

    // Save checkpoint.
    let cp = arena.checkpoint();
    assert_eq!(cp, 256);

    // Alloc 2: temporary, will be "freed" by restore.
    let td2 = arena.alloc(512).expect("alloc 2");
    assert_eq!(td2.arena_generation(), Some(gen_before));

    // Restore — generation must NOT change.
    arena.restore_checkpoint(cp).unwrap();
    assert_eq!(
        arena.generation(),
        gen_before,
        "restore_checkpoint must not change generation"
    );

    // Alloc 3: reuses td2's memory region.
    let td3 = arena.alloc(512).expect("alloc 3");
    assert_eq!(
        td3.arena_generation(),
        Some(gen_before),
        "post-restore alloc has same generation as pre-restore alloc"
    );

    // The key aliasing property: td2 and td3 have the same byte offset
    // and the same generation — stale-read detection cannot tell them apart.
    assert_eq!(
        td2.byte_offset(),
        td3.byte_offset(),
        "post-restore alloc reuses the same byte offset"
    );
    assert_eq!(
        td2.arena_generation(),
        td3.arena_generation(),
        "generation cannot distinguish pre/post-restore allocations"
    );
}

/// AC1 of #3114: Stale-read detection is blind to checkpoint-based reuse.
///
/// After restore_checkpoint, the stale-read check (generation comparison)
/// CANNOT distinguish td2 (freed by restore) from td3 (reuses td2's memory).
/// Both have the same arena_generation and overlapping byte ranges.
///
/// This is the documented blindspot. Safety depends on callers blit-copying
/// data out before restore, not on stale-read detection.
#[test]
fn test_checkpoint_restore_stale_read_blindspot() {
    let ctx = test_ctx();
    let mut arena = ActivationArena::new(ctx, 8192).expect("create arena");
    let generation = arena.generation();

    // Alloc and write known pattern to td1 (survives restore).
    let td1 = arena.alloc(256).expect("alloc td1");
    assert_eq!(td1.arena_generation(), Some(generation));

    let cp = arena.checkpoint();

    // Alloc td2 in the temporary region.
    let td2 = arena.alloc(512).expect("alloc td2");
    assert_eq!(td2.arena_generation(), Some(generation));
    let td2_offset = td2.byte_offset();

    // Restore checkpoint — td2's region is logically freed.
    arena.restore_checkpoint(cp).unwrap();

    // Alloc td3 — reuses td2's region.
    let td3 = arena.alloc(512).expect("alloc td3");
    assert_eq!(td3.byte_offset(), td2_offset, "td3 reuses td2's memory");
    assert_eq!(td3.arena_generation(), Some(generation), "same generation");

    // The blindspot: td2 and td3 are indistinguishable to stale-read detection.
    // Both have:
    //   - Same arena_generation
    //   - Same byte_offset
    //   - Overlapping byte ranges
    // A stale-read check on td2 would compare generation == generation → NOT stale.
    // This is correct: the safety contract is enforced by blit-copy, not by
    // generation tracking.
    assert_eq!(
        td2.arena_generation(),
        td3.arena_generation(),
        "stale-read detection cannot distinguish td2 from td3 — this is the checkpoint blindspot"
    );

    // td1 (before checkpoint) is unaffected and valid.
    assert_eq!(td1.byte_offset(), 0);
    assert_eq!(td1.arena_generation(), Some(generation));
}

/// AC2 of #3114: The safe pattern — blit-copy to a planned buffer before
/// checkpoint restore, mirroring compiled model execution.
///
/// This test verifies the pattern used in compiled_model_execute.rs:
/// 1. Allocate step output from arena
/// 2. Blit-copy output to a contiguous planned buffer
/// 3. Restore arena checkpoint (reclaim step output memory)
/// 4. Verify planned buffer data is intact (not affected by restore)
///
/// The planned buffer is NOT arena-allocated — it's a standalone Metal buffer.
/// This makes it immune to arena restore operations.
#[test]
fn test_relocate_then_restore_pattern() {
    let ctx = test_ctx();
    let mut arena = ActivationArena::new(ctx, 8192).expect("create arena");

    // Simulate: pre-existing allocation (like an earlier step's output).
    let _base = arena.alloc(256).expect("base alloc");

    // Save checkpoint before temporary step output.
    let cp = arena.checkpoint();
    assert_eq!(cp, 256);

    // Step output: temporary allocation that will be relocated.
    let step_out = arena.alloc(1024).expect("step output");
    assert_eq!(step_out.byte_offset(), 256);
    assert_eq!(arena.used_bytes(), 256 + 1024);

    // Create a planned buffer (standalone, NOT from arena).
    let planned_buf = ctx.create_buffer_zeroed(4096).expect("planned buffer");

    // Blit-copy step output to planned buffer at a specific offset.
    // In compiled_model_execute, this is `relocate_to_planned_buffer`.
    let dst_offset = 512_usize;
    let copy_size = 1024_usize;
    crate::gpu_scope::with_gpu_scope(|| {
        crate::gpu_scope::ensure_batch_for_blit()?;
        crate::gpu_scope::encode_into_lazy_batch(|batch| {
            batch.blit_copy(
                step_out.buffer(),
                step_out.byte_offset(),
                &planned_buf,
                dst_offset,
                copy_size,
            )
        })
        .expect("scope")
        .expect("blit_copy");
        Ok(())
    })
    .expect("gpu scope");

    // Restore checkpoint — step_out's arena region is freed.
    arena.restore_checkpoint(cp).unwrap();
    assert_eq!(arena.used_bytes(), 256, "arena reclaimed step output");

    // New allocation reuses the freed region.
    let new_alloc = arena.alloc(1024).expect("new alloc in freed region");
    assert_eq!(
        new_alloc.byte_offset(),
        step_out.byte_offset(),
        "new alloc reuses step_out's region"
    );

    // The planned buffer is unaffected — it's a standalone allocation.
    // Its data (the blit-copied step output) survives arena restore.
    assert_eq!(planned_buf.len(), 4096, "planned buffer intact");
    // Note: We cannot verify GPU data contents without a readback+compute
    // test. The structural properties (buffer length, arena state) confirm
    // the pattern's safety: arena restore only affects arena-allocated
    // memory, not standalone buffers.
}

/// P1 memory safety: Verify that `restore_checkpoint` returns `Err` when
/// the saved offset is ahead of the current bump pointer (restoring to
/// a "future" state is always a bug).
#[test]
fn test_restore_checkpoint_rejects_future_offset() {
    let ctx = test_ctx();
    let mut arena = ActivationArena::new(ctx, 8192).expect("create arena");

    arena.alloc(256).expect("alloc");
    let current = arena.used_bytes();
    assert_eq!(current, 256);

    // Attempting to restore to an offset > current must return Err.
    let result = arena.restore_checkpoint(current + 1);
    assert!(
        result.is_err(),
        "restore_checkpoint(future) must return Err"
    );

    // Arena state must be unchanged after the rejected restore.
    assert_eq!(arena.used_bytes(), 256);
    assert_eq!(arena.generation(), 0);
}

/// P1 memory safety: Verify that `restore_checkpoint(0)` effectively resets
/// the bump pointer without incrementing generation (unlike `reset()`).
#[test]
fn test_restore_checkpoint_zero_vs_reset() {
    let ctx = test_ctx();
    let mut arena = ActivationArena::new(ctx, 4096).expect("create arena");

    arena.alloc(1024).expect("alloc");
    let gen_before = arena.generation();

    // Restore to 0: same effect as reset on bump pointer, but NO generation change.
    arena.restore_checkpoint(0).unwrap();
    assert_eq!(arena.used_bytes(), 0);
    assert_eq!(
        arena.generation(),
        gen_before,
        "restore_checkpoint(0) must not change generation"
    );

    // Contrast with reset(): generation increments.
    arena.alloc(512).expect("alloc");
    arena.reset();
    assert_eq!(arena.used_bytes(), 0);
    assert_eq!(
        arena.generation(),
        gen_before + 1,
        "reset() must increment generation"
    );
}

// ---------------------------------------------------------------------------
// Coverage gap tests (P10 review)
// ---------------------------------------------------------------------------

/// Verify that `arena_alloc_or_create` falls back to a standalone buffer
/// when the explicit `with_arena` arena overflows, without propagating
/// ArenaOverflow as a hard error to the caller.
///
/// This tests the `try_arena_alloc` overflow → fresh buffer fallback path
/// within Priority 1 (explicit with_arena scope). Without this test, a
/// regression in `try_arena_alloc`'s ArenaOverflow match arm could cause
/// GPU dispatch to fail on large intermediate tensors.
#[test]
fn test_with_arena_overflow_fallback_to_standalone() {
    let ctx = test_ctx();
    // Small arena: only 512 bytes.
    let mut arena = ActivationArena::new(ctx, 512).expect("create arena");

    reset_arena_stats();

    with_arena(&mut arena, || {
        // First alloc fits in arena.
        let (buf1, off1) = arena_alloc_or_create(ctx, 256).expect("fits in arena");
        assert!(buf1.len() >= 256);
        assert_eq!(off1, 0);

        let stats1 = arena_stats();
        assert_eq!(stats1.hits, 1, "first alloc should be arena hit");
        assert_eq!(stats1.misses, 0);

        // Second alloc exceeds arena capacity — must fall back to standalone
        // buffer without returning an error.
        let (buf2, off2) = arena_alloc_or_create(ctx, 1024).expect("overflow must not error");
        assert!(buf2.len() >= 1024);
        assert_eq!(off2, 0, "standalone fallback must have offset 0");

        let stats2 = arena_stats();
        assert_eq!(stats2.hits, 1, "overflow should not increment hits");
        assert_eq!(stats2.misses, 1, "overflow should increment misses");

        // Verify fallback has no generation stamp.
        assert_eq!(
            last_alloc_generation(),
            None,
            "overflow fallback must have no generation stamp"
        );
    });
}

/// Verify that `checkpoint_default_arena` and `restore_default_arena`
/// correctly save and restore the default arena's bump pointer.
///
/// The compiled model pipeline uses these to reclaim temporary GPU buffers
/// within a single forward pass without a full arena reset.
#[test]
fn test_checkpoint_restore_default_arena() {
    let ctx = test_ctx();

    // Force default arena creation.
    let (_buf, _off) = arena_alloc_or_create(ctx, 256).expect("init default arena");
    let used_before = default_arena_used_bytes().expect("default arena exists");
    assert!(used_before > 0);

    // Save checkpoint.
    let cp = checkpoint_default_arena();
    assert!(
        cp.is_some(),
        "checkpoint must return Some when arena exists"
    );
    let (cp_offset, _cp_gen) = cp.unwrap();
    assert_eq!(cp_offset, used_before);

    // Allocate more from default arena.
    let (_buf2, _off2) = arena_alloc_or_create(ctx, 512).expect("alloc more");
    let used_after = default_arena_used_bytes().unwrap();
    assert!(
        used_after > used_before,
        "second alloc must advance bump pointer"
    );

    // Restore checkpoint — bump pointer goes back.
    restore_default_arena(cp);
    let used_restored = default_arena_used_bytes().unwrap();
    assert_eq!(
        used_restored, used_before,
        "restore must return bump pointer to checkpoint"
    );

    // Generation must NOT have changed (restore != reset).
    let gen_after_restore = default_arena_generation().unwrap();
    // Do a reset to compare.
    reset_default_arena();
    let gen_after_reset = default_arena_generation().unwrap();
    assert_eq!(
        gen_after_reset,
        gen_after_restore + 1,
        "reset increments generation but restore does not"
    );
}

/// Verify `ArenaStats::hit_rate()` calculation including the zero-total
/// edge case (no allocations → 0.0, not NaN or panic).
#[test]
fn test_arena_stats_hit_rate() {
    // Zero allocations: hit_rate must be 0.0, not NaN or division-by-zero.
    let empty = ArenaStats {
        hits: 0,
        misses: 0,
        pool: PoolStats::default(),
        growth_count: 0,
        total_growth_count: 0,
        overflow_count: 0,
        total_overflow_count: 0,
        overflow_bytes: 0,
        total_overflow_bytes: 0,
    };
    assert_eq!(empty.hit_rate(), 0.0);
    assert!(
        !empty.hit_rate().is_nan(),
        "zero-total must not produce NaN"
    );

    // All hits.
    let all_hits = ArenaStats {
        hits: 100,
        misses: 0,
        pool: PoolStats::default(),
        growth_count: 0,
        total_growth_count: 0,
        overflow_count: 0,
        total_overflow_count: 0,
        overflow_bytes: 0,
        total_overflow_bytes: 0,
    };
    assert_eq!(all_hits.hit_rate(), 1.0);

    // All misses.
    let all_misses = ArenaStats {
        hits: 0,
        misses: 50,
        pool: PoolStats {
            hits: 10,
            pooled_buffers: 3,
            ..PoolStats::default()
        },
        growth_count: 0,
        total_growth_count: 0,
        overflow_count: 0,
        total_overflow_count: 0,
        overflow_bytes: 0,
        total_overflow_bytes: 0,
    };
    assert_eq!(all_misses.hit_rate(), 0.0);

    // Mixed.
    let mixed = ArenaStats {
        hits: 75,
        misses: 25,
        pool: PoolStats {
            hits: 5,
            pooled_buffers: 2,
            ..PoolStats::default()
        },
        growth_count: 0,
        total_growth_count: 0,
        overflow_count: 0,
        total_overflow_count: 0,
        overflow_bytes: 0,
        total_overflow_bytes: 0,
    };
    assert!((mixed.hit_rate() - 0.75).abs() < f64::EPSILON);
}

/// Verify that `restore_default_arena` skips restore when a reset (flush)
/// occurred between checkpoint and restore — no panic.
///
/// Models the #3133 bug: `flush()` resets the arena (offset=0, gen++)
/// between `checkpoint_default_arena()` and `restore_default_arena()`.
/// Without the generation guard, `restore_checkpoint` panics because
/// `saved_offset > current_offset`.
#[test]
fn test_checkpoint_restore_skips_after_reset() {
    let ctx = test_ctx();

    // Force default arena creation with some initial allocation.
    let (_buf, _off) = arena_alloc_or_create(ctx, 1024).expect("init default arena");
    let used_before = default_arena_used_bytes().expect("default arena exists");
    assert!(
        used_before > 0,
        "must have non-zero offset for test to be meaningful"
    );

    // Save checkpoint at non-zero offset.
    let cp = checkpoint_default_arena();
    assert!(cp.is_some());

    // Simulate flush: reset the arena (offset=0, generation++).
    reset_default_arena();
    let used_after_reset = default_arena_used_bytes().unwrap();
    assert_eq!(used_after_reset, 0, "reset clears bump pointer");

    // Restore MUST NOT panic — generation mismatch causes skip.
    // Before the #3133 fix, this would panic:
    //   "arena restore_checkpoint: saved N > current 0"
    restore_default_arena(cp);

    // Verify arena is still at offset 0 (restore was skipped).
    let used_final = default_arena_used_bytes().unwrap();
    assert_eq!(
        used_final, 0,
        "restore must be skipped when generation changed (arena was reset)"
    );
}

// ---------------------------------------------------------------------------
// without_arena panic safety tests
// ---------------------------------------------------------------------------

/// P1 memory safety: Verify that `without_arena` clears the ARENA_BYPASS
/// flag even when the closure panics. Without this, the bypass flag would
/// remain `true` after unwind, causing ALL subsequent GPU allocations on
/// this thread to skip the arena — silently defeating arena-based memory
/// reuse and causing RSS regression.
///
/// This mirrors `test_with_arena_cleanup_on_panic` for the bypass path.
#[test]
fn test_without_arena_cleanup_on_panic() {
    assert!(!is_arena_bypassed());

    // Panic inside without_arena.
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        without_arena(|| {
            assert!(is_arena_bypassed());
            panic!("deliberate panic inside bypass scope");
        });
    }));
    assert!(result.is_err(), "should have caught panic");

    // Critical assertion: bypass flag must be restored to false.
    assert!(
        !is_arena_bypassed(),
        "ARENA_BYPASS must be restored after panic in without_arena"
    );
}

/// P1 memory safety: Verify that a panic inside `without_arena` nested
/// within `with_arena` cleans up BOTH the bypass flag AND the arena TLS
/// pointer. This is the worst-case combined scope cleanup scenario.
#[test]
fn test_without_arena_panic_inside_with_arena() {
    let ctx = test_ctx();
    let mut arena = ActivationArena::new(ctx, 8192).expect("create arena");

    assert!(!is_arena_active());
    assert!(!is_arena_bypassed());

    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        with_arena(&mut arena, || {
            assert!(is_arena_active());
            without_arena(|| {
                assert!(is_arena_bypassed());
                panic!("deliberate panic in bypass-inside-arena");
            });
        });
    }));
    assert!(result.is_err(), "should have caught panic");

    // Both scopes must be cleaned up.
    assert!(
        !is_arena_bypassed(),
        "bypass flag must be cleared after nested panic"
    );
    assert!(
        !is_arena_active(),
        "arena TLS pointer must be cleared after nested panic"
    );

    // Arena should be usable after the double-unwind cleanup.
    let val = with_arena(&mut arena, || {
        assert!(is_arena_active());
        assert!(!is_arena_bypassed());
        let (_buf, _off) = arena_alloc_or_create(ctx, 256).expect("alloc after nested panic");
        77
    });
    assert_eq!(val, 77);
}

/// Verify that arena overflow fallback routes through the buffer pool
/// and updates pool stats correctly. `arena_stats().pool` includes
/// full `PoolStats` — verify `pooled_buffers` increments on overflow.
#[test]
fn test_arena_overflow_updates_pool_stats() {
    let ctx = test_ctx();
    let mut arena = ActivationArena::new(ctx, 512).expect("create small arena");

    reset_arena_stats();

    with_arena(&mut arena, || {
        // First alloc fits in arena.
        let (_buf, _off) = arena_alloc_or_create(ctx, 256).expect("fits in arena");

        let stats1 = arena_stats();
        assert_eq!(stats1.hits, 1, "first alloc is arena hit");
        let pool_buffers_before = stats1.pool.pooled_buffers;

        // Second alloc overflows arena — routed through pool_acquire.
        let (_buf2, off2) = arena_alloc_or_create(ctx, 2048).expect("overflow to pool");
        assert_eq!(off2, 0, "pool allocation has offset 0");

        let stats2 = arena_stats();
        assert_eq!(stats2.hits, 1, "overflow should not increment arena hits");
        assert_eq!(stats2.misses, 1, "overflow should increment misses");
        assert!(
            stats2.pool.pooled_buffers > pool_buffers_before,
            "pool must gain entry on overflow (before={pool_buffers_before}, after={})",
            stats2.pool.pooled_buffers
        );
    });
}

// -- fresh_allocs and arena_capacity tests (#3079 D4) --

/// ArenaStats::fresh_allocs() returns misses that bypassed the pool.
#[test]
fn test_fresh_allocs_computation() {
    let stats = ArenaStats {
        hits: 100,
        misses: 20,
        pool: PoolStats {
            hits: 15,
            pooled_buffers: 5,
            pooled_bytes: 320 * 1024,
            ..PoolStats::default()
        },
        growth_count: 0,
        total_growth_count: 0,
        overflow_count: 0,
        total_overflow_count: 0,
        overflow_bytes: 0,
        total_overflow_bytes: 0,
    };
    assert_eq!(
        stats.fresh_allocs(),
        5,
        "20 misses - 15 pool.hits = 5 fresh"
    );

    let all_pooled = ArenaStats {
        hits: 50,
        misses: 10,
        pool: PoolStats {
            hits: 10,
            pooled_buffers: 3,
            pooled_bytes: 192 * 1024,
            ..PoolStats::default()
        },
        growth_count: 0,
        total_growth_count: 0,
        overflow_count: 0,
        total_overflow_count: 0,
        overflow_bytes: 0,
        total_overflow_bytes: 0,
    };
    assert_eq!(all_pooled.fresh_allocs(), 0, "all misses served by pool");

    let no_misses = ArenaStats {
        hits: 50,
        misses: 0,
        pool: PoolStats::default(),
        growth_count: 0,
        total_growth_count: 0,
        overflow_count: 0,
        total_overflow_count: 0,
        overflow_bytes: 0,
        total_overflow_bytes: 0,
    };
    assert_eq!(no_misses.fresh_allocs(), 0, "no misses = no fresh allocs");
}

/// arena_capacity() returns the default arena capacity constant.
#[test]
fn test_arena_capacity_returns_64mb() {
    let cap = arena_capacity();
    assert_eq!(
        cap,
        64 * 1024 * 1024,
        "default arena capacity should be 64 MB"
    );
}

// ---------------------------------------------------------------------------
// from_arena_alloc dispatch bridge tests
// ---------------------------------------------------------------------------

/// P1 memory safety: Verify that `MetalTensorData::from_arena_alloc` captures
/// the arena generation from `last_alloc_generation()` when inside an arena scope.
///
/// `from_arena_alloc` is the critical bridge between `arena_alloc_or_create`
/// (which sets the thread-local generation) and `MetalTensorData` construction.
/// It's used at 19+ call sites in GPU dispatch helpers. A bug here (e.g.,
/// failing to capture generation) would silently disable stale-read detection
/// for all arena-backed tensors.
#[test]
fn test_from_arena_alloc_captures_generation_in_arena_scope() {
    let ctx = test_ctx();
    let mut arena = ActivationArena::new(ctx, 8192).expect("create arena");

    with_arena(&mut arena, || {
        // arena_alloc_or_create sets last_alloc_generation to Some(0).
        let (buf, offset) = arena_alloc_or_create(ctx, 256).expect("arena alloc");
        assert_eq!(
            last_alloc_generation(),
            Some(0),
            "arena alloc must set last_alloc_generation"
        );

        // from_arena_alloc reads last_alloc_generation and creates view_arena.
        let td = MetalTensorData::from_arena_alloc(buf, offset);
        assert_eq!(
            td.arena_generation(),
            Some(0),
            "from_arena_alloc must capture generation from thread-local"
        );
        assert_eq!(td.byte_offset(), offset);
    });
}

/// P1 memory safety: Verify that `from_arena_alloc` produces `None` generation
/// when arena is bypassed (without_arena scope). Pool-acquired buffers must NOT
/// have a generation stamp — they are standalone and immune to arena resets.
#[test]
fn test_from_arena_alloc_no_generation_in_bypass() {
    let ctx = test_ctx();

    without_arena(|| {
        let (buf, offset) = arena_alloc_or_create(ctx, 256).expect("bypass alloc");
        assert_eq!(offset, 0, "bypass alloc has offset 0");
        assert_eq!(
            last_alloc_generation(),
            None,
            "bypass alloc must not set generation"
        );

        let td = MetalTensorData::from_arena_alloc(buf, offset);
        assert_eq!(
            td.arena_generation(),
            None,
            "from_arena_alloc must produce None generation for bypass allocs"
        );
    });
}

/// Verify that `from_arena_alloc` preserves generation across multiple arena
/// allocs within the same generation — the generation stamp is the same for all.
#[test]
fn test_from_arena_alloc_consistent_generation_across_allocs() {
    let ctx = test_ctx();
    let mut arena = ActivationArena::new(ctx, 16384).expect("create arena");

    with_arena(&mut arena, || {
        // First alloc.
        let (buf1, off1) = arena_alloc_or_create(ctx, 256).expect("alloc 1");
        let td1 = MetalTensorData::from_arena_alloc(buf1, off1);

        // Second alloc — same generation.
        let (buf2, off2) = arena_alloc_or_create(ctx, 512).expect("alloc 2");
        let td2 = MetalTensorData::from_arena_alloc(buf2, off2);

        assert_eq!(td1.arena_generation(), Some(0));
        assert_eq!(td2.arena_generation(), Some(0));
        assert_eq!(
            td1.arena_generation(),
            td2.arena_generation(),
            "all allocs within the same generation must have the same stamp"
        );
        // Offsets must differ (non-overlapping).
        assert_ne!(
            off1, off2,
            "sequential arena allocs must have different offsets"
        );
    });
}

// ---------------------------------------------------------------------------
// Buffer pool acquire-reclaim cycle tests (P10 gap coverage)
// ---------------------------------------------------------------------------

/// P1 buffer aliasing safety: Verify the full pool_acquire → pool_reclaim →
/// pool_acquire cycle produces a hit (buffer reuse) on the second acquire.
///
/// This is the core reuse invariant of the buffer pool. Without this test,
/// a regression in reclaim_all() or the available-flag scan in acquire()
/// could silently disable buffer reuse, causing RSS regression.
#[test]
fn test_pool_acquire_reclaim_reuse_cycle() {
    let ctx = test_ctx();

    // Reset pool stats to isolate this test.
    pool::reset_pool_stats();

    // First acquire: must be a miss (no available entries).
    let (buf1, off1) = pool::pool_acquire(ctx, 64 * 1024).expect("first acquire");
    assert_eq!(off1, 0, "pool acquire always returns offset 0");
    assert!(buf1.len() >= 64 * 1024);

    let stats1 = pool::pool_stats();
    assert_eq!(stats1.acquisitions, 1);
    assert_eq!(stats1.hits, 0, "first acquire must be a miss");
    assert_eq!(
        stats1.misses, 1,
        "first acquire creates a new pooled buffer"
    );
    assert_eq!(stats1.pooled_buffers, 1, "one buffer now in pool");

    // Reclaim: mark all pool entries as available.
    pool_reclaim();

    // Second acquire with same size class: must be a hit (reuse).
    let (buf2, off2) = pool::pool_acquire(ctx, 64 * 1024).expect("second acquire");
    assert_eq!(off2, 0);
    assert!(buf2.len() >= 64 * 1024);

    let stats2 = pool::pool_stats();
    assert_eq!(stats2.acquisitions, 2);
    assert_eq!(
        stats2.hits, 1,
        "second acquire must hit the reclaimed entry"
    );
    assert_eq!(stats2.misses, 1, "no new miss on reuse");
    assert_eq!(
        stats2.pooled_buffers, 1,
        "pool size unchanged — reused existing entry"
    );

    // Invariant: acquisitions == hits + misses + discards
    assert_eq!(
        stats2.acquisitions,
        stats2.hits + stats2.misses + stats2.discards,
        "stats invariant must hold after acquire-reclaim-acquire cycle"
    );
}

/// Verify that pool_reclaim() (the public thread-local API) makes previously
/// acquired entries available for reuse across multiple size classes.
#[test]
fn test_pool_reclaim_multi_class_reuse() {
    let ctx = test_ctx();
    pool::reset_pool_stats();

    // Acquire buffers in two different size classes.
    let (_buf_64k, _) = pool::pool_acquire(ctx, 64 * 1024).expect("acquire 64KB");
    let (_buf_1m, _) = pool::pool_acquire(ctx, 1024 * 1024).expect("acquire 1MB");

    let stats_before = pool::pool_stats();
    assert_eq!(
        stats_before.pooled_buffers, 2,
        "two entries across two classes"
    );
    assert_eq!(stats_before.misses, 2, "both were misses");

    // Reclaim all.
    pool_reclaim();

    // Re-acquire both — should both be hits.
    let (_buf_64k_2, _) = pool::pool_acquire(ctx, 64 * 1024).expect("re-acquire 64KB");
    let (_buf_1m_2, _) = pool::pool_acquire(ctx, 1024 * 1024).expect("re-acquire 1MB");

    let stats_after = pool::pool_stats();
    assert_eq!(stats_after.hits, 2, "both re-acquires must be hits");
    assert_eq!(
        stats_after.pooled_buffers, 2,
        "pool size unchanged after reuse"
    );
}

/// Verify that `default_arena_peak_bytes()` tracks peak usage of the
/// default (always-on, 64 MB) arena.
///
/// The default arena is lazily initialized on first `arena_alloc_or_create`
/// without an explicit `with_arena` scope. `default_arena_peak_bytes()`
/// exposes peak tracking for diagnostics.
#[test]
fn test_default_arena_peak_bytes_tracking() {
    let ctx = test_ctx();

    // Force default arena creation with a known allocation.
    let (_buf, _off) = arena_alloc_or_create(ctx, 4096).expect("init default arena");

    // Peak must be >= the allocation size.
    let peak = default_arena_peak_bytes();
    assert!(
        peak.is_some(),
        "default_arena_peak_bytes() must return Some after allocation"
    );
    assert!(
        peak.unwrap() >= 4096,
        "peak ({}) must be >= allocated 4096 bytes",
        peak.unwrap()
    );

    // Allocate more — peak should increase.
    let (_buf2, _off2) = arena_alloc_or_create(ctx, 8192).expect("second alloc");
    let peak2 = default_arena_peak_bytes().unwrap();
    assert!(
        peak2 >= peak.unwrap(),
        "peak must be monotonically non-decreasing: {} >= {}",
        peak2,
        peak.unwrap()
    );

    // Reset default arena — peak should be preserved (peak survives reset).
    reset_default_arena();
    let peak_after_reset = default_arena_peak_bytes().unwrap();
    assert_eq!(peak_after_reset, peak2, "peak must survive arena reset");
}

// ── Planned redirect tests (#3448) ──────────────────────────────────────

#[test]
fn test_planned_redirect_size_match() {
    let ctx = test_ctx();
    let buf = ctx.create_buffer_zeroed(4096).expect("create buffer");
    let offset = 128;
    let expected_bytes = 1024;

    // Arm the redirect.
    set_planned_redirect(&buf, offset, expected_bytes);

    // Allocate with matching size — should return the planned buffer.
    let (alloc_buf, alloc_off) =
        arena_alloc_or_create(ctx, expected_bytes).expect("alloc matching size");
    assert!(
        alloc_buf.is_same_allocation(&buf),
        "matching allocation should return the planned buffer"
    );
    assert_eq!(alloc_off, offset, "offset should match the planned offset");

    // Subsequent alloc of same size should NOT return planned buffer (consumed).
    let (alloc2_buf, _) =
        arena_alloc_or_create(ctx, expected_bytes).expect("alloc after consumed");
    assert!(
        !alloc2_buf.is_same_allocation(&buf),
        "redirect should be consumed after first match"
    );
    // Clean up default arena state.
    reset_default_arena();
}

#[test]
fn test_planned_redirect_size_mismatch() {
    let ctx = test_ctx();
    let buf = ctx.create_buffer_zeroed(4096).expect("create buffer");
    set_planned_redirect(&buf, 0, 1024);

    // Allocate with different size — should NOT return planned buffer.
    let (alloc_buf, _) = arena_alloc_or_create(ctx, 2048).expect("alloc mismatched size");
    assert!(
        !alloc_buf.is_same_allocation(&buf),
        "mismatched size should bypass the redirect"
    );

    // Clean up: redirect is still armed, clear it.
    clear_planned_redirect();
    reset_default_arena();
}

#[test]
fn test_planned_redirect_guard_auto_clears() {
    let ctx = test_ctx();
    let buf = ctx.create_buffer_zeroed(4096).expect("create buffer");

    {
        let _guard = arm_planned_redirect_guard(&buf, 64, 512);
        // Guard is alive — redirect is armed.
        // Drop guard at end of scope.
    }

    // After guard drop, redirect should be cleared.
    // Allocating 512 bytes should NOT return the planned buffer.
    let (alloc_buf, _) = arena_alloc_or_create(ctx, 512).expect("alloc after guard drop");
    assert!(
        !alloc_buf.is_same_allocation(&buf),
        "guard drop should clear the redirect"
    );
    reset_default_arena();
}

#[test]
fn test_planned_redirect_consumed_then_guard_drop() {
    let ctx = test_ctx();
    let buf = ctx.create_buffer_zeroed(4096).expect("create buffer");

    let _guard = arm_planned_redirect_guard(&buf, 0, 256);

    // Consume the redirect.
    let (alloc_buf, _) = arena_alloc_or_create(ctx, 256).expect("alloc matching");
    assert!(
        alloc_buf.is_same_allocation(&buf),
        "should consume the redirect"
    );

    // Drop guard — should be a no-op since redirect was already consumed.
    drop(_guard);

    // Verify no stale state: next alloc of same size uses arena, not redirect.
    let (alloc2_buf, _) = arena_alloc_or_create(ctx, 256).expect("alloc after drop");
    assert!(
        !alloc2_buf.is_same_allocation(&buf),
        "no stale redirect after guard drop"
    );
    reset_default_arena();
}

// ---------------------------------------------------------------------------
// Decode scope tests (#3359)
// ---------------------------------------------------------------------------

/// Verify that `with_decode_scope` records the arena generation at entry.
#[test]
fn test_decode_scope_sets_generation() {
    let _ctx = test_ctx();
    // Ensure default arena exists.
    let _ = arena_alloc_or_create(_ctx, 256).expect("init arena");

    // Outside decode scope: no scope generation.
    assert!(
        decode_scope_generation().is_none(),
        "no decode scope outside with_decode_scope"
    );

    with_decode_scope(|| {
        let scope_gen = decode_scope_generation();
        assert!(scope_gen.is_some(), "decode scope must be active");
    });

    // After scope exits: cleared.
    assert!(
        decode_scope_generation().is_none(),
        "decode scope must be cleared after exit"
    );
    reset_default_arena();
}

/// Verify that nested `with_decode_scope` preserves the outer scope.
#[test]
fn test_decode_scope_nesting_preserves_outer() {
    let _ctx = test_ctx();
    let _ = arena_alloc_or_create(_ctx, 256).expect("init arena");

    with_decode_scope(|| {
        let outer_gen = decode_scope_generation().expect("outer scope");

        // Advance the arena generation.
        reset_default_arena();

        with_decode_scope(|| {
            let inner_gen = decode_scope_generation().expect("inner scope");
            // Inner scope must preserve the outer (earlier) generation.
            assert_eq!(
                inner_gen, outer_gen,
                "nested decode scope must preserve outer generation"
            );
        });

        // Outer scope still active.
        assert_eq!(
            decode_scope_generation(),
            Some(outer_gen),
            "outer scope must remain after inner exits"
        );
    });
    reset_default_arena();
}

/// Verify that `with_decode_scope` cleans up on panic.
#[test]
fn test_decode_scope_cleanup_on_panic() {
    let _ctx = test_ctx();
    let _ = arena_alloc_or_create(_ctx, 256).expect("init arena");

    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        with_decode_scope(|| {
            assert!(decode_scope_generation().is_some());
            panic!("test panic inside decode scope");
        });
    }));
    assert!(result.is_err(), "panic must propagate");

    assert!(
        decode_scope_generation().is_none(),
        "decode scope must be cleared after panic"
    );
    reset_default_arena();
}

/// Simulate the autoregressive decode pattern: multiple arena resets within
/// a decode scope. Tensors from earlier generations should be considered
/// within scope (alloc_gen >= scope_gen), so the stale check would skip them.
#[test]
fn test_decode_scope_tolerates_multi_generation_advance() {
    let _ctx = test_ctx();
    let _ = arena_alloc_or_create(_ctx, 256).expect("init arena");

    with_decode_scope(|| {
        let scope_gen = decode_scope_generation().expect("scope active");

        // Allocate a tensor in the initial generation.
        let td_early = {
            let (_buf, _off) = arena_alloc_or_create(_ctx, 256).expect("alloc early");
            let alloc_gen = last_alloc_generation();
            (alloc_gen, scope_gen)
        };

        // Simulate multiple decode steps: reset arena several times.
        for _ in 0..5 {
            reset_default_arena();
            let _ = arena_alloc_or_create(_ctx, 256).expect("alloc step");
        }

        let current_gen = default_arena_generation().expect("arena exists");
        assert!(
            current_gen > scope_gen + 1,
            "arena must have advanced multiple generations: current={current_gen}, scope={scope_gen}"
        );

        // The early tensor's alloc_gen >= scope_gen, so it would pass the
        // decode scope check in gpu_to_cpu.
        if let Some(alloc_gen) = td_early.0 {
            assert!(
                alloc_gen >= scope_gen,
                "early tensor must be within decode scope"
            );
            let in_scope = decode_scope_generation()
                .is_some_and(|sg| alloc_gen >= sg);
            assert!(in_scope, "decode scope must protect early tensor from stale check");
        }
    });
    reset_default_arena();
}

// ---------------------------------------------------------------------------
// Edge-case allocation pattern tests (#3555)
// ---------------------------------------------------------------------------

/// Verify that all allocations within a single generation have offsets aligned
/// to METAL_BUFFER_ALIGNMENT, regardless of requested sizes.
#[test]
fn test_all_alloc_offsets_are_metal_aligned() {
    let ctx = test_ctx();
    let mut arena = ActivationArena::new(ctx, 65536).expect("create arena");

    // Allocate a series of odd-sized buffers that are NOT multiples of 256.
    let sizes = [1, 7, 100, 255, 257, 511, 1023, 3];
    let mut offsets = Vec::new();

    for &size in &sizes {
        let td = arena
            .alloc(size)
            .unwrap_or_else(|_| panic!("alloc {size}"));
        let offset = td.byte_offset();
        assert_eq!(
            offset % 256,
            0,
            "alloc({size}) returned offset {offset} which is not 256-aligned"
        );
        offsets.push((offset, size));
    }

    // Verify pairwise non-overlap.
    for i in 0..offsets.len() - 1 {
        let (start_i, len_i) = offsets[i];
        let end_i = start_i + len_i;
        let (start_next, _) = offsets[i + 1];
        assert!(
            start_next >= end_i,
            "alloc {i} [{}..{}) overlaps alloc {} [{}..)",
            start_i, end_i, i + 1, start_next
        );
    }
}

/// Verify that after a full reset cycle, all arena state is consistent:
/// offset=0, generation incremented, capacity unchanged, remaining=capacity.
#[test]
fn test_reset_full_state_consistency() {
    let ctx = test_ctx();
    let cap = 16384;
    let mut arena = ActivationArena::new(ctx, cap).expect("create arena");

    // Allocate several times to advance the bump pointer.
    arena.alloc(1024).expect("alloc 1");
    arena.alloc(2048).expect("alloc 2");
    arena.alloc(512).expect("alloc 3");
    let peak_before = arena.peak_bytes();
    assert!(peak_before > 0);

    let gen_before = arena.generation();

    // Reset.
    arena.reset();

    assert_eq!(arena.used_bytes(), 0, "offset must be 0 after reset");
    assert_eq!(arena.generation(), gen_before + 1, "generation must increment by 1");
    assert_eq!(arena.capacity(), cap, "capacity must not change on reset");
    assert_eq!(arena.remaining_bytes(), cap, "remaining must equal capacity after reset");
    assert_eq!(arena.peak_bytes(), peak_before, "peak must survive reset");
}

/// Verify that peak_bytes tracks the all-time maximum across multiple
/// reset/allocation cycles, including cycles with larger and smaller allocations.
#[test]
fn test_peak_bytes_across_multiple_generations() {
    let ctx = test_ctx();
    let mut arena = ActivationArena::new(ctx, 65536).expect("create arena");

    // Gen 0: allocate 4096 bytes.
    arena.alloc(4096).expect("gen 0 alloc");
    let peak_gen0 = arena.peak_bytes();
    assert!(peak_gen0 >= 4096);

    // Gen 1: smaller allocation — peak must not decrease.
    arena.reset();
    arena.alloc(1024).expect("gen 1 alloc");
    assert_eq!(
        arena.peak_bytes(),
        peak_gen0,
        "peak must not decrease when gen 1 < gen 0"
    );

    // Gen 2: larger allocation — peak must increase.
    arena.reset();
    arena.alloc(8192).expect("gen 2 alloc");
    let peak_gen2 = arena.peak_bytes();
    assert!(
        peak_gen2 > peak_gen0,
        "peak must increase when gen 2 > gen 0: {peak_gen2} > {peak_gen0}"
    );

    // Gen 3: even smaller — peak still at gen 2 level.
    arena.reset();
    arena.alloc(512).expect("gen 3 alloc");
    assert_eq!(arena.peak_bytes(), peak_gen2, "peak must hold at gen 2 high water mark");
}

/// Verify that the arena capacity never changes through any operation
/// (alloc, reset, checkpoint, restore). Capacity is fixed at creation.
#[test]
fn test_capacity_immutable_through_operations() {
    let ctx = test_ctx();
    let cap = 32768;
    let mut arena = ActivationArena::new(ctx, cap).expect("create arena");

    assert_eq!(arena.capacity(), cap);

    // After alloc.
    arena.alloc(1024).expect("alloc");
    assert_eq!(arena.capacity(), cap, "capacity must not change after alloc");

    // After checkpoint.
    let cp = arena.checkpoint();
    assert_eq!(arena.capacity(), cap, "capacity must not change after checkpoint");

    // After more allocs.
    arena.alloc(2048).expect("alloc 2");
    assert_eq!(arena.capacity(), cap, "capacity must not change after second alloc");

    // After restore.
    arena.restore_checkpoint(cp).expect("restore");
    assert_eq!(arena.capacity(), cap, "capacity must not change after restore");

    // After reset.
    arena.reset();
    assert_eq!(arena.capacity(), cap, "capacity must not change after reset");

    // After multiple resets.
    for _ in 0..5 {
        arena.alloc(256).expect("alloc in cycle");
        arena.reset();
        assert_eq!(arena.capacity(), cap, "capacity must not change through cycles");
    }
}

/// Verify that checkpoint/restore to the exact same offset is a no-op.
/// This covers the edge case where no allocations happen between
/// checkpoint and restore.
#[test]
fn test_checkpoint_restore_noop_when_no_allocs() {
    let ctx = test_ctx();
    let mut arena = ActivationArena::new(ctx, 8192).expect("create arena");

    arena.alloc(1024).expect("alloc");
    let used_before = arena.used_bytes();
    let gen_before = arena.generation();

    let cp = arena.checkpoint();
    assert_eq!(cp, used_before);

    // No allocations between checkpoint and restore.
    arena.restore_checkpoint(cp).expect("restore");

    assert_eq!(arena.used_bytes(), used_before, "offset must not change");
    assert_eq!(arena.generation(), gen_before, "restore must not change generation");
}

/// Verify that restoring to offset 0 reclaims all memory (like reset but
/// without generation increment).
#[test]
fn test_restore_to_zero_reclaims_all() {
    let ctx = test_ctx();
    let mut arena = ActivationArena::new(ctx, 8192).expect("create arena");
    let gen_before = arena.generation();

    // Checkpoint at offset 0 (before any allocation).
    let cp = arena.checkpoint();
    assert_eq!(cp, 0);

    // Allocate.
    arena.alloc(2048).expect("alloc");
    assert!(arena.used_bytes() > 0);

    // Restore to 0 — reclaims all memory.
    arena.restore_checkpoint(cp).expect("restore to 0");
    assert_eq!(arena.used_bytes(), 0, "all memory reclaimed");
    assert_eq!(
        arena.generation(),
        gen_before,
        "generation must NOT change on restore (unlike reset)"
    );

    // Can allocate again from offset 0.
    let td = arena.alloc(512).expect("alloc after restore to 0");
    assert_eq!(td.byte_offset(), 0, "first alloc after restore-to-0 starts at 0");
}

/// Verify that `restore_checkpoint` returns an error when the saved offset
/// is strictly greater than the current offset, and that all arena state
/// is completely unchanged after the rejected restore.
#[test]
fn test_restore_checkpoint_rejects_future_offset_state_unchanged() {
    let ctx = test_ctx();
    let mut arena = ActivationArena::new(ctx, 8192).expect("create arena");

    // Allocate a small amount.
    arena.alloc(256).expect("alloc 256");
    let current = arena.used_bytes();
    let generation = arena.generation();
    let peak = arena.peak_bytes();

    // Try to restore to a larger offset — must fail.
    let future_offset = current + 1024;
    let err = arena
        .restore_checkpoint(future_offset)
        .expect_err("future offset must be rejected");
    let msg = err.to_string();
    assert!(
        msg.contains("checkpoint"),
        "error must mention checkpoint: {msg}"
    );

    // ALL arena state must be unchanged after the failed restore.
    assert_eq!(arena.used_bytes(), current, "offset must not change on failed restore");
    assert_eq!(arena.generation(), generation, "generation must not change on failed restore");
    assert_eq!(arena.peak_bytes(), peak, "peak must not change on failed restore");
}

/// Verify that the exact capacity can be fully utilized when the allocation
/// size is a multiple of the alignment.
#[test]
fn test_exact_capacity_utilization_aligned_sizes() {
    let ctx = test_ctx();
    // 4 * 256 = 1024 bytes capacity.
    let mut arena = ActivationArena::new(ctx, 1024).expect("create arena");

    // Allocate in aligned chunks: 256 + 256 + 256 + 256 = 1024.
    arena.alloc(256).expect("alloc 1");
    arena.alloc(256).expect("alloc 2");
    arena.alloc(256).expect("alloc 3");
    arena.alloc(256).expect("alloc 4");

    assert_eq!(arena.used_bytes(), 1024, "must use full capacity");
    assert_eq!(arena.remaining_bytes(), 0, "no remaining bytes");

    // One more byte must fail.
    let err = arena.alloc(1);
    assert!(err.is_err(), "alloc past capacity must fail");
}

/// Verify that alignment padding is correctly accounted for: allocating
/// sizes that are NOT multiples of 256 causes the bump pointer to advance
/// past the useful region due to alignment rounding.
#[test]
fn test_alignment_padding_reduces_effective_capacity() {
    let ctx = test_ctx();
    let mut arena = ActivationArena::new(ctx, 1024).expect("create arena");

    // Alloc 1 byte: uses bytes [0, 1), but bump pointer advances to 1.
    // Next alloc starts at align_up(1, 256) = 256.
    arena.alloc(1).expect("alloc 1 byte");
    assert_eq!(arena.used_bytes(), 1);

    // Next alloc: offset 256. Alloc 1 byte: uses [256, 257).
    // Next alloc would start at 512.
    arena.alloc(1).expect("alloc 1 byte at 256");
    assert_eq!(arena.used_bytes(), 257); // 256 + 1

    // Next alloc: offset 512. Alloc 1 byte: uses [512, 513).
    arena.alloc(1).expect("alloc 1 byte at 512");
    assert_eq!(arena.used_bytes(), 513); // 512 + 1

    // Next alloc: offset 768. Alloc 1 byte: uses [768, 769).
    arena.alloc(1).expect("alloc 1 byte at 768");
    assert_eq!(arena.used_bytes(), 769); // 768 + 1

    // Only 4 bytes of payload in 1024 bytes — 99.6% is alignment padding.
    // Next alloc would start at 1024, which equals capacity, so even 1 byte fails.
    let err = arena.alloc(1);
    assert!(err.is_err(), "5th 1-byte alloc must fail: alignment ate the capacity");
}

// ---------------------------------------------------------------------------
// Multi-step checkpoint/restore cycle (realistic Kokoro pattern) (#3555)
// ---------------------------------------------------------------------------

/// Exercises the compiled model pattern: a persistent allocation followed by
/// repeated checkpoint → temp alloc → restore cycles. Each cycle reuses the
/// same arena region, simulating Kokoro's compiled steps.
///
/// Verifies:
/// - Checkpoint/restore produces deterministic offsets across iterations
/// - Peak bytes tracks the true high-water mark
/// - Generation does NOT change (checkpoint/restore is intra-generation)
/// - The persistent allocation is not corrupted by temp cycles
#[test]
fn test_multi_step_checkpoint_restore_cycle() {
    let ctx = test_ctx();
    // 64 KB arena — enough for a few 256-aligned allocations.
    let mut arena = ActivationArena::new(ctx, 64 * 1024).expect("create arena");

    // Persistent allocation: 4096 bytes at offset 0.
    let persistent = arena.alloc(4096).expect("persistent alloc");
    assert_eq!(persistent.byte_offset(), 0);
    assert_eq!(arena.used_bytes(), 4096);

    let checkpoint_offset = arena.checkpoint();
    assert_eq!(checkpoint_offset, 4096);
    let gen_before = arena.generation();

    // Simulate 10 compiled model steps, each doing checkpoint → temp → restore.
    for step in 0..10 {
        let cp = arena.checkpoint();
        assert_eq!(cp, checkpoint_offset, "checkpoint must be stable at step {step}");

        // Temp allocation: 2048 bytes.
        let temp = arena.alloc(2048).expect("temp alloc");
        // After alignment of 4096 (which is already aligned), temp starts at 4096.
        assert_eq!(temp.byte_offset(), 4096, "temp offset at step {step}");
        assert_eq!(arena.used_bytes(), 4096 + 2048);

        // Restore checkpoint: bump pointer returns to 4096.
        arena
            .restore_checkpoint(checkpoint_offset)
            .expect("restore");
        assert_eq!(arena.used_bytes(), checkpoint_offset);
    }

    // Generation did NOT change (checkpoint/restore is intra-generation).
    assert_eq!(arena.generation(), gen_before);

    // Peak bytes reflects the temp allocation high-water mark.
    assert_eq!(arena.peak_bytes(), 4096 + 2048);

    // Persistent allocation is still valid (offset 0, within arena).
    assert_eq!(persistent.byte_offset(), 0);
}

/// Exercises a full lifecycle: alloc → reset → alloc → checkpoint → restore →
/// reset, verifying all state transitions.
#[test]
fn test_full_arena_lifecycle() {
    let ctx = test_ctx();
    let mut arena = ActivationArena::new(ctx, 16 * 1024).expect("create arena");

    // Phase 1: Generation 0 allocations.
    let a1 = arena.alloc(1024).expect("g0 alloc 1");
    assert_eq!(a1.byte_offset(), 0);
    let a2 = arena.alloc(512).expect("g0 alloc 2");
    assert_eq!(a2.byte_offset(), 1024); // 1024 is already 256-aligned
    assert_eq!(arena.used_bytes(), 1024 + 512);
    assert_eq!(arena.generation(), 0);

    // Phase 2: Reset to generation 1.
    arena.reset();
    assert_eq!(arena.used_bytes(), 0);
    assert_eq!(arena.generation(), 1);
    assert_eq!(arena.remaining_bytes(), 16 * 1024);
    let peak_g0 = arena.peak_bytes();
    assert_eq!(peak_g0, 1024 + 512);

    // Phase 3: Generation 1 — checkpoint/restore pattern.
    let b1 = arena.alloc(2048).expect("g1 alloc");
    assert_eq!(b1.byte_offset(), 0);
    let cp = arena.checkpoint();
    assert_eq!(cp, 2048);

    let b_temp = arena.alloc(4096).expect("g1 temp alloc");
    assert_eq!(b_temp.byte_offset(), 2048); // 2048 is 256-aligned
    assert_eq!(arena.used_bytes(), 2048 + 4096);

    arena.restore_checkpoint(cp).expect("restore");
    assert_eq!(arena.used_bytes(), 2048);
    assert_eq!(arena.generation(), 1); // unchanged

    // Peak reflects g1 high-water (2048+4096 > g0 peak of 1536).
    assert_eq!(arena.peak_bytes(), 2048 + 4096);

    // Phase 4: Reset to generation 2.
    arena.reset();
    assert_eq!(arena.used_bytes(), 0);
    assert_eq!(arena.generation(), 2);
    // Peak preserved across reset.
    assert_eq!(arena.peak_bytes(), 2048 + 4096);

    // Phase 5: Allocations still work in generation 2.
    let c1 = arena.alloc(256).expect("g2 alloc");
    assert_eq!(c1.byte_offset(), 0);
    assert_eq!(arena.used_bytes(), 256);
}

/// Verifies that after checkpoint/restore, the arena correctly aligns the
/// next allocation even when the checkpoint offset is not aligned.
#[test]
fn test_checkpoint_restore_alignment_recovery() {
    let ctx = test_ctx();
    let mut arena = ActivationArena::new(ctx, 8192).expect("create arena");

    // Create a non-aligned checkpoint: alloc 100 bytes → offset = 100.
    let a1 = arena.alloc(100).expect("alloc 100");
    assert_eq!(a1.byte_offset(), 0);
    assert_eq!(arena.used_bytes(), 100);

    let cp = arena.checkpoint();
    assert_eq!(cp, 100); // NOT 256-aligned

    // Temp alloc: starts at 256 (align_up(100, 256) = 256).
    let temp = arena.alloc(300).expect("temp alloc");
    assert_eq!(temp.byte_offset(), 256);

    // Restore to non-aligned checkpoint.
    arena.restore_checkpoint(cp).expect("restore");
    assert_eq!(arena.used_bytes(), 100);

    // Next alloc must be aligned despite the non-aligned offset.
    let a2 = arena.alloc(400).expect("post-restore alloc");
    assert_eq!(a2.byte_offset(), 256); // align_up(100, 256) = 256
    assert_eq!(
        a2.byte_offset() % 256,
        0,
        "post-restore alloc must be Metal-aligned"
    );
}

// ---------------------------------------------------------------------------
// ActivationArena stats tracking tests (Part of #4353)
// ---------------------------------------------------------------------------

/// A fresh arena (just created, no allocations) reports zero for all
/// allocation-related counters.
#[test]
fn test_fresh_arena_reports_zero_stats() {
    let ctx = test_ctx();
    let arena = ActivationArena::new(ctx, 4096).expect("create arena");

    assert_eq!(arena.used_bytes(), 0, "no bytes used on fresh arena");
    assert_eq!(arena.peak_bytes(), 0, "no peak on fresh arena");
    assert_eq!(arena.generation(), 0, "generation starts at 0");
    assert_eq!(arena.remaining_bytes(), 4096, "all capacity is remaining");
    assert_eq!(arena.growth_count(), 0, "no growth events");
    assert_eq!(arena.total_growth_count(), 0, "no total growth events");
    assert!(!arena.is_auto_grow(), "auto-grow off by default");
    assert_eq!(arena.retired_slab_count(), 0, "no retired slabs");
}

/// After `reset_arena_stats()`, the thread-local stats counters are zeroed
/// including pool stats. A subsequent `arena_stats()` call returns all zeros.
#[test]
fn test_reset_arena_stats_clears_all_counters() {
    let ctx = test_ctx();

    // Force some stats by allocating through the default arena.
    let (_buf, _off) = arena_alloc_or_create(ctx, 512).expect("alloc to generate stats");

    reset_arena_stats();

    let stats = arena_stats();
    assert_eq!(stats.hits, 0, "hits cleared after reset");
    assert_eq!(stats.misses, 0, "misses cleared after reset");
    assert_eq!(stats.pool.acquisitions, 0, "pool acquisitions cleared");
    assert_eq!(stats.pool.hits, 0, "pool hits cleared");
    assert_eq!(stats.pool.misses, 0, "pool misses cleared");
    assert_eq!(stats.pool.discards, 0, "pool discards cleared");
    assert_eq!(stats.hit_rate(), 0.0, "hit rate 0.0 after reset");
    assert_eq!(stats.fresh_allocs(), 0, "fresh allocs 0 after reset");
}

/// After allocations within a `with_arena` scope, `arena_stats()` reflects
/// the correct hit count. Each successful arena allocation increments hits.
#[test]
fn test_with_arena_scope_updates_stats_hits() {
    let ctx = test_ctx();
    let mut arena = ActivationArena::new(ctx, 16384).expect("create arena");

    reset_arena_stats();

    with_arena(&mut arena, || {
        // Three allocations that fit in the arena.
        let _ = arena_alloc_or_create(ctx, 256).expect("alloc 1");
        let _ = arena_alloc_or_create(ctx, 512).expect("alloc 2");
        let _ = arena_alloc_or_create(ctx, 1024).expect("alloc 3");
    });

    let stats = arena_stats();
    assert_eq!(stats.hits, 3, "three arena allocations = 3 hits");
    assert_eq!(stats.misses, 0, "no overflows = 0 misses");
    assert_eq!(stats.hit_rate(), 1.0, "100% hit rate when all fit");
}

/// Overflow allocations within `with_arena` increment misses and route
/// through the pool, updating pool stats accordingly.
#[test]
fn test_with_arena_scope_overflow_updates_misses_and_pool() {
    let ctx = test_ctx();
    // Small arena that will overflow on the second allocation.
    let mut arena = ActivationArena::new(ctx, 512).expect("create small arena");

    reset_arena_stats();

    with_arena(&mut arena, || {
        // Fits in arena.
        let _ = arena_alloc_or_create(ctx, 256).expect("fits");
        // Overflows arena — routed to pool.
        let _ = arena_alloc_or_create(ctx, 4096).expect("overflows");
    });

    let stats = arena_stats();
    assert_eq!(stats.hits, 1, "one arena hit");
    assert_eq!(stats.misses, 1, "one overflow miss");
    assert!(
        stats.pool.pooled_buffers > 0 || stats.pool.discards > 0,
        "overflow must route through pool (pooled_buffers={}, discards={})",
        stats.pool.pooled_buffers,
        stats.pool.discards
    );
    assert!(
        (stats.hit_rate() - 0.5).abs() < 1e-12,
        "1 hit / 2 total = 0.5 hit rate"
    );
}

/// Buffer reuse through the pool: after reclaim, a second acquire of the
/// same size class reuses the buffer (pool hit) instead of allocating fresh.
#[test]
fn test_buffer_reuse_same_size_class_pool_hit() {
    let ctx = test_ctx();

    pool::reset_pool_stats();

    // First acquire: cold miss, creates a new pooled buffer.
    let (_buf1, _) = pool::pool_acquire(ctx, 256 * 1024).expect("first acquire 256KB");
    let stats1 = pool::pool_stats();
    assert_eq!(stats1.acquisitions, 1);
    assert_eq!(stats1.hits, 0, "first is cold miss");
    assert_eq!(stats1.pooled_buffers, 1, "one buffer in pool");

    // Reclaim: makes the buffer available.
    pool_reclaim();

    // Second acquire same size class: warm hit, reuses.
    let (_buf2, _) = pool::pool_acquire(ctx, 256 * 1024).expect("second acquire 256KB");
    let stats2 = pool::pool_stats();
    assert_eq!(stats2.acquisitions, 2);
    assert_eq!(stats2.hits, 1, "second must be a pool hit (reuse)");
    assert_eq!(stats2.pooled_buffers, 1, "no new buffer created");

    // Third acquire same size WITHOUT reclaim: no available entry.
    let (_buf3, _) = pool::pool_acquire(ctx, 256 * 1024).expect("third acquire 256KB");
    let stats3 = pool::pool_stats();
    assert_eq!(stats3.acquisitions, 3);
    assert_eq!(stats3.hits, 1, "no reclaim = no hit, still 1");
    assert_eq!(stats3.pooled_buffers, 2, "new buffer created for third");
}

/// Pool stats break down by size class: acquiring buffers of different sizes
/// creates entries in different size classes, reflected in `pooled_buffers`.
#[test]
fn test_pool_stats_multiple_size_classes() {
    let ctx = test_ctx();

    pool::reset_pool_stats();

    // Acquire in three different size classes.
    let (_a, _) = pool::pool_acquire(ctx, 64 * 1024).expect("64KB");       // class 0
    let (_b, _) = pool::pool_acquire(ctx, 256 * 1024).expect("256KB");     // class 1
    let (_c, _) = pool::pool_acquire(ctx, 4 * 1024 * 1024).expect("4MB"); // class 3

    let stats = pool::pool_stats();
    assert_eq!(stats.acquisitions, 3, "three acquires");
    assert_eq!(stats.pooled_buffers, 3, "three entries across three classes");
    assert_eq!(stats.hits, 0, "all cold misses");
    assert_eq!(stats.misses, 3, "three misses (new pooled buffers)");

    // pooled_bytes should reflect the sum of size-class capacities (not
    // request sizes — buffers are rounded up to their class boundary).
    assert!(
        stats.pooled_bytes >= 64 * 1024 + 256 * 1024 + 4 * 1024 * 1024,
        "pooled_bytes ({}) must be at least the sum of class sizes",
        stats.pooled_bytes
    );

    // Reclaim + re-acquire: all three should hit.
    pool_reclaim();
    let _ = pool::pool_acquire(ctx, 64 * 1024).expect("re-64KB");
    let _ = pool::pool_acquire(ctx, 256 * 1024).expect("re-256KB");
    let _ = pool::pool_acquire(ctx, 4 * 1024 * 1024).expect("re-4MB");

    let stats2 = pool::pool_stats();
    assert_eq!(stats2.acquisitions, 6, "six total acquires");
    assert_eq!(stats2.hits, 3, "three hits after reclaim");
    assert_eq!(stats2.pooled_buffers, 3, "still three entries (reused)");
}

/// Auto-grow: when enabled, overflow triggers slab growth instead of
/// `ArenaOverflow`. Growth stats are updated correctly.
#[test]
fn test_auto_grow_updates_growth_stats() {
    let ctx = test_ctx();
    let mut arena = ActivationArena::new(ctx, 512).expect("create small arena");
    arena.set_auto_grow(ctx);
    assert!(arena.is_auto_grow(), "auto-grow enabled");

    // First alloc fits in the initial slab.
    let _ = arena.alloc(256).expect("fits in initial slab");
    assert_eq!(arena.growth_count(), 0, "no growth yet");
    assert_eq!(arena.total_growth_count(), 0);
    assert_eq!(arena.retired_slab_count(), 0);

    // Second alloc overflows — triggers auto-grow.
    let _ = arena.alloc(1024).expect("triggers auto-grow");
    assert_eq!(arena.growth_count(), 1, "one growth event this generation");
    assert_eq!(arena.total_growth_count(), 1, "one total growth event");
    assert_eq!(arena.retired_slab_count(), 1, "old slab retired");
    assert!(
        arena.capacity() >= 1024,
        "new slab capacity ({}) must fit the request",
        arena.capacity()
    );

    // Reset clears growth_count but not total_growth_count.
    arena.reset();
    assert_eq!(arena.growth_count(), 0, "growth_count reset to 0");
    assert_eq!(arena.total_growth_count(), 1, "total_growth_count persists");
    assert_eq!(arena.retired_slab_count(), 0, "retired slabs dropped on reset");

    // Trigger another growth in the new generation.
    let big_alloc = arena.capacity() + 256;
    let _ = arena.alloc(big_alloc).expect("triggers second auto-grow");
    assert_eq!(arena.growth_count(), 1, "one growth in new generation");
    assert_eq!(arena.total_growth_count(), 2, "two total growths across generations");
}

/// ArenaStats growth_count and total_growth_count fields reflect the
/// default arena's growth state when queried via `arena_stats()`.
#[test]
fn test_arena_stats_growth_fields_via_default_arena() {
    // Reset stats before measuring.
    reset_arena_stats();

    let stats = arena_stats();
    // Default arena may or may not exist yet depending on prior tests,
    // but growth_count fields should always be non-negative and consistent.
    assert!(
        stats.total_growth_count >= stats.growth_count,
        "total_growth ({}) must be >= current generation growth ({})",
        stats.total_growth_count,
        stats.growth_count
    );
}

/// `with_arena` scope stats are cumulative: multiple scopes add up.
#[test]
fn test_stats_cumulative_across_with_arena_scopes() {
    let ctx = test_ctx();
    let mut arena = ActivationArena::new(ctx, 16384).expect("create arena");

    reset_arena_stats();

    // First scope: 2 allocations.
    with_arena(&mut arena, || {
        let _ = arena_alloc_or_create(ctx, 256).expect("scope1 alloc1");
        let _ = arena_alloc_or_create(ctx, 256).expect("scope1 alloc2");
    });

    let stats1 = arena_stats();
    assert_eq!(stats1.hits, 2, "2 hits after first scope");

    arena.reset();

    // Second scope: 3 allocations.
    with_arena(&mut arena, || {
        let _ = arena_alloc_or_create(ctx, 256).expect("scope2 alloc1");
        let _ = arena_alloc_or_create(ctx, 256).expect("scope2 alloc2");
        let _ = arena_alloc_or_create(ctx, 256).expect("scope2 alloc3");
    });

    let stats2 = arena_stats();
    assert_eq!(stats2.hits, 5, "cumulative: 2 + 3 = 5 hits");
    assert_eq!(stats2.misses, 0, "no overflows");
}

/// `reset_arena_stats()` is idempotent: calling it twice in a row leaves
/// all counters at zero.
#[test]
fn test_reset_arena_stats_idempotent() {
    reset_arena_stats();
    reset_arena_stats();
    let stats = arena_stats();
    assert_eq!(stats.hits, 0);
    assert_eq!(stats.misses, 0);
    assert_eq!(stats.pool.acquisitions, 0);
    assert_eq!(stats.pool.hits, 0);
}

/// `without_arena` bypass increments misses (not hits), because all
/// allocations are routed to standalone pool buffers.
#[test]
fn test_without_arena_bypass_increments_misses() {
    let ctx = test_ctx();

    reset_arena_stats();

    without_arena(|| {
        let _ = arena_alloc_or_create(ctx, 1024).expect("bypass alloc");
    });

    let stats = arena_stats();
    assert_eq!(stats.hits, 0, "bypass never hits arena");
    assert_eq!(stats.misses, 1, "bypass counted as miss");
}

/// Pre-sizing the arena via `ensure_default_arena_capacity` prevents growth
/// events when subsequent allocations fit within the pre-sized capacity.
/// Part of #4289.
#[test]
fn test_ensure_capacity_prevents_growth() {
    let ctx = test_ctx();
    reset_default_arena();
    reset_arena_stats();

    let target = 4 * 1024 * 1024; // 4 MB
    ensure_default_arena_capacity(ctx, target).expect("ensure_capacity");

    // Allocate 7 x 512 KB = 3.5 MB — fits within 4 MB pre-sized arena.
    let chunk = 512 * 1024;
    for _ in 0..7 {
        let _ = arena_alloc_or_create(ctx, chunk).expect("alloc chunk");
    }

    let stats = arena_stats();
    assert_eq!(stats.growth_count, 0, "no growth events after pre-sizing");
    assert_eq!(stats.misses, 0, "all allocations should be arena hits");
    assert!(stats.hits >= 7, "at least 7 arena hits expected");

    reset_default_arena();
}

/// When an allocation exceeds the pre-sized capacity, auto-grow handles it
/// without falling back to standalone pool buffers (misses stay at 0).
/// Part of #4289.
#[test]
fn test_ensure_capacity_auto_grow_on_excess() {
    let ctx = test_ctx();
    reset_default_arena();
    reset_arena_stats();

    ensure_default_arena_capacity(ctx, 1024).expect("ensure_capacity");

    // Allocate more than pre-sized capacity — auto-grow kicks in.
    let _ = arena_alloc_or_create(ctx, 2048).expect("alloc exceeds pre-size");

    let stats = arena_stats();
    assert_eq!(stats.misses, 0, "auto-grow should handle overflow without pool fallback");
    assert!(stats.hits >= 1, "allocation should be an arena hit");

    reset_default_arena();
}

// ---------------------------------------------------------------------------
// GPU buffer pool / ActivationArena memory reuse tests (Part of #3828)
// ---------------------------------------------------------------------------

/// Auto-grow overflow stats: verify that `overflow_count`, `overflow_bytes`,
/// `total_overflow_count`, and `total_overflow_bytes` are correctly updated
/// when auto-grow triggers slab growth.
#[test]
fn test_auto_grow_overflow_stats_tracking() {
    let ctx = test_ctx();
    let mut arena = ActivationArena::new(ctx, 512).expect("create small arena");
    arena.set_auto_grow(ctx);

    // Fresh arena: all overflow counters at zero.
    assert_eq!(arena.overflow_count(), 0);
    assert_eq!(arena.total_overflow_count(), 0);
    assert_eq!(arena.overflow_bytes(), 0);
    assert_eq!(arena.total_overflow_bytes(), 0);

    // First alloc fits — no overflow.
    let _ = arena.alloc(256).expect("fits in initial slab");
    assert_eq!(arena.overflow_count(), 0);
    assert_eq!(arena.overflow_bytes(), 0);

    // Second alloc overflows the 512-byte slab — triggers auto-grow.
    let _ = arena.alloc(1024).expect("triggers auto-grow");
    assert_eq!(arena.overflow_count(), 1, "one overflow this generation");
    assert_eq!(arena.total_overflow_count(), 1, "one total overflow");
    assert_eq!(arena.overflow_bytes(), 1024, "1024 bytes via overflow");
    assert_eq!(arena.total_overflow_bytes(), 1024, "1024 total overflow bytes");

    // Reset clears per-generation overflow stats but preserves totals.
    arena.reset();
    assert_eq!(arena.overflow_count(), 0, "per-gen overflow_count cleared");
    assert_eq!(arena.overflow_bytes(), 0, "per-gen overflow_bytes cleared");
    assert_eq!(arena.total_overflow_count(), 1, "total persists across reset");
    assert_eq!(arena.total_overflow_bytes(), 1024, "total bytes persists");

    // Trigger another overflow in the new generation.
    let big = arena.capacity() + 512;
    let _ = arena.alloc(big).expect("second auto-grow");
    assert_eq!(arena.overflow_count(), 1, "one overflow in new gen");
    assert_eq!(arena.total_overflow_count(), 2, "two total overflows");
    assert_eq!(arena.overflow_bytes(), big);
    assert_eq!(arena.total_overflow_bytes(), 1024 + big);
}

/// Auto-grow capacity doubling: verify that `grow_and_alloc` at least doubles
/// the previous capacity, and that the new slab fits the request.
#[test]
fn test_auto_grow_capacity_doubling() {
    let ctx = test_ctx();
    let initial_cap = 1024;
    let mut arena = ActivationArena::new(ctx, initial_cap).expect("create arena");
    arena.set_auto_grow(ctx);

    // Fill the initial slab.
    let _ = arena.alloc(initial_cap).expect("fill initial slab");
    assert_eq!(arena.capacity(), initial_cap);

    // Request 256 bytes — overflows but small. New cap = max(2*1024, 256+256) = 2048.
    let _ = arena.alloc(256).expect("small overflow triggers doubling");
    assert!(
        arena.capacity() >= 2 * initial_cap,
        "capacity ({}) must be >= 2 * initial ({})",
        arena.capacity(),
        2 * initial_cap
    );
    assert_eq!(arena.retired_slab_count(), 1, "one retired slab");

    // Request larger than 2 * current capacity — new cap must fit the request.
    let huge = arena.capacity() * 3;
    let _ = arena.alloc(huge).expect("huge alloc triggers big growth");
    assert!(
        arena.capacity() >= huge,
        "capacity ({}) must be >= request ({})",
        arena.capacity(),
        huge
    );
    assert_eq!(arena.retired_slab_count(), 2, "two retired slabs");
}

/// Auto-grow retired slabs are dropped on reset, freeing Metal memory.
#[test]
fn test_auto_grow_retired_slabs_cleared_on_reset() {
    let ctx = test_ctx();
    let mut arena = ActivationArena::new(ctx, 512).expect("create arena");
    arena.set_auto_grow(ctx);

    // Trigger three growth events to accumulate retired slabs.
    for _ in 0..3 {
        let big = arena.capacity() + 256;
        let _ = arena.alloc(big).expect("trigger growth");
    }
    assert_eq!(arena.retired_slab_count(), 3, "three retired slabs before reset");

    // Reset drops all retired slabs.
    arena.reset();
    assert_eq!(arena.retired_slab_count(), 0, "retired slabs cleared on reset");

    // The current (larger) slab is retained — capacity did NOT shrink.
    assert!(arena.capacity() >= 512, "capacity >= initial after reset");
}

/// `try_reset_active_arena`: resets the explicit arena inside `with_arena`.
#[test]
fn test_try_reset_active_arena_explicit_scope() {
    let ctx = test_ctx();
    let mut arena = ActivationArena::new(ctx, 8192).expect("create arena");

    // Cannot read `arena` inside `with_arena` closure (mutable borrow conflict).
    // Verify state before and after the scope instead.
    with_arena(&mut arena, || {
        let _ = arena_alloc_or_create(ctx, 1024).expect("alloc");
        let did_reset = try_reset_active_arena();
        assert!(did_reset, "must return true when arena was reset");
    });
    // After scope: arena was reset inside the closure.
    assert_eq!(arena.generation(), 1, "generation must have incremented");
    assert_eq!(arena.used_bytes(), 0, "bump pointer must be 0 after reset");
}

/// `try_reset_active_arena`: resets the default arena when no explicit scope.
#[test]
fn test_try_reset_active_arena_default_arena() {
    let ctx = test_ctx();

    // Force default arena creation.
    let _ = arena_alloc_or_create(ctx, 256).expect("init default arena");
    let gen_before = default_arena_generation().expect("arena exists");

    let did_reset = try_reset_active_arena();
    assert!(did_reset, "must return true when default arena was reset");

    let gen_after = default_arena_generation().expect("arena exists");
    assert_eq!(gen_after, gen_before + 1, "generation must increment");
}

/// `try_reset_active_arena` returns false when bypass is active.
#[test]
fn test_try_reset_active_arena_false_in_bypass() {
    without_arena(|| {
        let did_reset = try_reset_active_arena();
        assert!(!did_reset, "must return false when bypass is active");
    });
}

/// Buffer pool byte budget enforcement: after filling to MAX_POOLED_BYTES,
/// additional acquire calls for new size classes are discarded.
#[test]
fn test_pool_byte_budget_cap_enforcement() {
    let ctx = test_ctx();
    pool::reset_pool_stats();

    // Acquire 8 buffers of 64 MB each = 512 MB = MAX_POOLED_BYTES.
    for i in 0..8 {
        let (_buf, _) = pool::pool_acquire(ctx, 64 * 1024 * 1024)
            .unwrap_or_else(|e| panic!("acquire {i} failed: {e}"));
    }

    let stats_full = pool::pool_stats();
    assert_eq!(stats_full.pooled_bytes, 512 * 1024 * 1024, "budget full");
    assert_eq!(stats_full.pooled_buffers, 8, "8 buffers pooled");

    // Reclaim all so entries are available.
    pool_reclaim();

    // Request 1 MB (class 2, not in pool). Budget is exhausted, so it is a discard.
    let (_buf_new, _) = pool::pool_acquire(ctx, 1024 * 1024).expect("acquire small");

    let stats_after = pool::pool_stats();
    assert!(
        stats_after.discards > stats_full.discards,
        "over-budget request must be discarded: before={}, after={}",
        stats_full.discards,
        stats_after.discards
    );
}

/// Buffer pool reclaim on populated pool: all entries become available.
#[test]
fn test_pool_reclaim_populated_entries_reused() {
    let ctx = test_ctx();
    pool::reset_pool_stats();

    // Acquire 3 buffers in the same size class (no reclaim between them).
    let (_a, _) = pool::pool_acquire(ctx, 64 * 1024).expect("acquire 1");
    let (_b, _) = pool::pool_acquire(ctx, 64 * 1024).expect("acquire 2");
    let (_c, _) = pool::pool_acquire(ctx, 64 * 1024).expect("acquire 3");

    let stats_before = pool::pool_stats();
    assert_eq!(stats_before.hits, 0, "no hits before reclaim");
    assert_eq!(stats_before.pooled_buffers, 3, "3 entries in pool");

    pool_reclaim();

    // Re-acquire 3 buffers — all should be hits.
    let (_d, _) = pool::pool_acquire(ctx, 64 * 1024).expect("re-acquire 1");
    let (_e, _) = pool::pool_acquire(ctx, 64 * 1024).expect("re-acquire 2");
    let (_f, _) = pool::pool_acquire(ctx, 64 * 1024).expect("re-acquire 3");

    let stats_after = pool::pool_stats();
    assert_eq!(stats_after.hits, 3, "all re-acquires must be hits");
    assert_eq!(stats_after.pooled_buffers, 3, "no new entries added");
}

/// `ArenaEstimate` equality and clone.
#[test]
fn test_arena_estimate_equality_and_clone() {
    let a = ArenaEstimate {
        peak_bytes: 4096,
        total_bytes: 8192,
        step_count: 3,
    };
    let b = a.clone();
    assert_eq!(a, b, "clone must produce equal ArenaEstimate");

    let c = ArenaEstimate {
        peak_bytes: 4096,
        total_bytes: 8192,
        step_count: 4,
    };
    assert_ne!(a, c, "different step_count must be unequal");
}

/// `estimate_arena_peak_from_shapes` with zero-dimension shape: product is 0,
/// so the entry is skipped.
#[test]
fn test_estimate_peak_from_shapes_zero_dim_skipped() {
    let entries: Vec<(&str, &[usize], usize)> = vec![
        ("empty_batch", &[0, 256, 100], 4), // product = 0
        ("valid", &[1, 256, 100], 4),       // 25600 * 4 = 102400
    ];
    let est = estimate_arena_peak_from_shapes(&entries);
    assert_eq!(est.step_count, 1, "zero-dim shape should be skipped");
    assert_eq!(est.total_bytes, 102400);
    assert_eq!(est.peak_bytes, 102400);
}

/// `estimate_arena_peak_bytes` alignment simulation matches real arena behavior.
#[test]
fn test_estimate_peak_matches_real_arena_used_bytes() {
    let ctx = test_ctx();
    let sizes = vec![100, 200, 300, 500, 1024, 7, 4096];

    let est = estimate_arena_peak_bytes(sizes.iter().copied());

    let mut arena = ActivationArena::new(ctx, est.peak_bytes + 4096).expect("arena");
    for &size in &sizes {
        let _ = arena.alloc(size).expect("alloc");
    }

    assert_eq!(
        arena.used_bytes(),
        est.peak_bytes,
        "estimate must match real arena used_bytes for sequential allocs"
    );
}

/// Auto-grow: after growth, new slab allocation starts at offset 0.
#[test]
fn test_auto_grow_new_slab_offset_zero() {
    let ctx = test_ctx();
    let mut arena = ActivationArena::new(ctx, 512).expect("create arena");
    arena.set_auto_grow(ctx);

    let _ = arena.alloc(512).expect("fill slab");
    assert_eq!(arena.used_bytes(), 512);

    let td = arena.alloc(256).expect("alloc in new slab");
    assert_eq!(td.byte_offset(), 0, "new slab starts at offset 0");
    assert_eq!(arena.used_bytes(), 256, "used_bytes reflects new slab only");
}

/// Auto-grow flag persists across reset.
#[test]
fn test_auto_grow_flag_persists_across_reset() {
    let ctx = test_ctx();
    let mut arena = ActivationArena::new(ctx, 1024).expect("create arena");
    assert!(!arena.is_auto_grow(), "off by default");

    arena.set_auto_grow(ctx);
    assert!(arena.is_auto_grow());

    arena.reset();
    assert!(arena.is_auto_grow(), "auto-grow persists across reset");

    let _ = arena.alloc(2048).expect("post-reset auto-grow");
    assert_eq!(arena.growth_count(), 1, "growth works after reset");
}

/// Buffer pool MAX_PER_CLASS: 9th acquire is a discard when bucket is full.
#[test]
fn test_pool_max_per_class_ninth_acquire_discarded() {
    let ctx = test_ctx();
    pool::reset_pool_stats();

    for i in 0..8 {
        let (_buf, _) = pool::pool_acquire(ctx, 64 * 1024)
            .unwrap_or_else(|e| panic!("acquire {i} failed: {e}"));
    }

    let stats_at_cap = pool::pool_stats();
    assert_eq!(stats_at_cap.pooled_buffers, 8, "8 entries at MAX_PER_CLASS");

    let (_buf9, _) = pool::pool_acquire(ctx, 64 * 1024).expect("9th acquire");

    let stats_over = pool::pool_stats();
    assert_eq!(stats_over.pooled_buffers, 8, "no growth beyond MAX_PER_CLASS");
    assert!(
        stats_over.discards > stats_at_cap.discards,
        "9th acquire must discard: before={}, after={}",
        stats_at_cap.discards,
        stats_over.discards
    );
}

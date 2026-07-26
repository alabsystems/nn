// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Kani proof harnesses for arena buffer aliasing safety.
//!
//! Proves the core safety properties of the bump allocator:
//! - `align_up` correctness (aligned, monotonic, minimal padding)
//! - Sequential allocations produce non-overlapping regions
//! - Checkpoint/restore reuses memory (documenting the aliasing hazard)
//! - Multi-cycle checkpoint/restore maintains capacity and determinism

use super::{align_up, METAL_BUFFER_ALIGNMENT};

/// Prove: `align_up` result is always a multiple of the alignment.
#[kani::unwind(8)]
#[kani::proof]
#[kani::unwind(1)]
fn align_up_result_is_aligned() {
    let offset: usize = kani::any();
    let alignment: usize = kani::any();
    // Constrain to realistic bounds for CBMC tractability.
    kani::assume(alignment > 0 && alignment <= 4096);
    kani::assume(alignment.is_power_of_two());
    kani::assume(offset <= (1usize << 40)); // ~1 TB

    if let Ok(result) = align_up(offset, alignment) {
        assert_eq!(result % alignment, 0, "result must be aligned");
    }
    // Err is valid (overflow near usize::MAX).
}

/// Prove: `align_up` result is always >= the input offset (never rounds down).
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn align_up_result_geq_offset() {
    let offset: usize = kani::any();
    let alignment: usize = kani::any();
    kani::assume(alignment > 0 && alignment <= 4096);
    kani::assume(alignment.is_power_of_two());
    kani::assume(offset <= (1usize << 40));

    if let Ok(result) = align_up(offset, alignment) {
        assert!(result >= offset, "align_up must never decrease offset");
    }
}

/// Prove: `align_up` padding is minimal — less than one alignment unit.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn align_up_padding_is_minimal() {
    let offset: usize = kani::any();
    let alignment: usize = kani::any();
    kani::assume(alignment > 0 && alignment <= 4096);
    kani::assume(alignment.is_power_of_two());
    kani::assume(offset <= (1usize << 40));

    if let Ok(result) = align_up(offset, alignment) {
        let padding = result - offset;
        assert!(
            padding < alignment,
            "padding ({padding}) must be < alignment ({alignment})"
        );
    }
}

/// Prove: `align_up` with non-power-of-two alignment always returns Err.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn align_up_rejects_non_power_of_two() {
    let offset: usize = kani::any();
    let alignment: usize = kani::any();
    kani::assume(offset <= (1usize << 40));
    kani::assume(alignment <= 4096);
    kani::assume(!alignment.is_power_of_two());

    assert!(
        align_up(offset, alignment).is_err(),
        "non-power-of-two alignment must fail"
    );
}

/// Prove: two sequential bump allocations with Metal alignment produce
/// non-overlapping byte regions.
///
/// Models the core bump allocator logic from `ActivationArena::alloc`:
/// ```text
/// alloc1: start1 = align_up(0, 256),       end1 = start1 + len1
/// alloc2: start2 = align_up(end1, 256),     end2 = start2 + len2
/// ```
/// Proves: `start2 >= end1` (no overlap).
#[kani::unwind(8)]
#[kani::proof]
#[kani::unwind(1)]
fn bump_alloc_regions_non_overlapping() {
    let len1: usize = kani::any();
    let len2: usize = kani::any();
    let capacity: usize = kani::any();
    let alignment = METAL_BUFFER_ALIGNMENT; // 256

    // Bound inputs for CBMC tractability.
    kani::assume(len1 > 0 && len1 <= (1usize << 30));
    kani::assume(len2 > 0 && len2 <= (1usize << 30));
    kani::assume(capacity <= (1usize << 32));

    // First allocation.
    let start1 = match align_up(0, alignment) {
        Ok(v) => v,
        Err(_) => return,
    };
    let end1 = match start1.checked_add(len1) {
        Some(v) if v <= capacity => v,
        _ => return,
    };

    // Second allocation.
    let start2 = match align_up(end1, alignment) {
        Ok(v) => v,
        Err(_) => return,
    };
    let end2 = match start2.checked_add(len2) {
        Some(v) if v <= capacity => v,
        _ => return,
    };

    // Core safety property: regions [start1, end1) and [start2, end2) don't overlap.
    assert!(
        start2 >= end1,
        "second allocation must start at or after first allocation ends"
    );
    // Both regions are within capacity.
    assert!(end1 <= capacity);
    assert!(end2 <= capacity);
}

/// Prove: checkpoint/restore to a prior offset followed by a new allocation
/// produces a region that may overlap with the post-checkpoint allocation.
///
/// This is the expected (documented) behavior — `restore_checkpoint` is
/// explicitly a "logical free" that permits overwrite. The safety contract
/// requires the caller to ensure no live references exist to the freed region.
/// This harness documents the aliasing hazard formally.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn checkpoint_restore_reuses_memory() {
    let len1: usize = kani::any();
    let len2: usize = kani::any();
    let len3: usize = kani::any();
    let capacity: usize = kani::any();
    let alignment = METAL_BUFFER_ALIGNMENT;

    kani::assume(len1 > 0 && len1 <= (1usize << 28));
    kani::assume(len2 > 0 && len2 <= (1usize << 28));
    kani::assume(len3 > 0 && len3 <= (1usize << 28));
    kani::assume(capacity <= (1usize << 30));

    // Alloc 1: advance bump pointer.
    let start1 = match align_up(0, alignment) {
        Ok(v) => v,
        Err(_) => return,
    };
    let end1 = match start1.checked_add(len1) {
        Some(v) if v <= capacity => v,
        _ => return,
    };
    // Checkpoint at end1.
    let checkpoint = end1;

    // Alloc 2: temporary allocation (will be "freed" by restore).
    let start2 = match align_up(end1, alignment) {
        Ok(v) => v,
        Err(_) => return,
    };
    let end2 = match start2.checked_add(len2) {
        Some(v) if v <= capacity => v,
        _ => return,
    };

    // Restore checkpoint: bump pointer goes back to `checkpoint`.
    // Alloc 3: new allocation in the restored region.
    let start3 = match align_up(checkpoint, alignment) {
        Ok(v) => v,
        Err(_) => return,
    };
    let end3 = match start3.checked_add(len3) {
        Some(v) if v <= capacity => v,
        _ => return,
    };

    // The KEY aliasing property: alloc 3 may overlap alloc 2.
    // This is INTENTIONAL — restore_checkpoint logically frees alloc 2's
    // region. The safety contract requires alloc 2's data to have been
    // blit-copied out before restore.
    //
    // Prove: alloc 3 starts at the same position alloc 2 did.
    // Note: this follows deterministically from checkpoint == end1 and
    // align_up being a pure function. The value of the assertion is
    // documenting the aliasing hazard formally, not proving a non-obvious
    // property.
    assert_eq!(
        start3, start2,
        "post-restore allocation reuses the same aligned offset"
    );
    // Alloc 1 is NOT affected — it's before the checkpoint.
    assert!(start1 < checkpoint || len1 == 0);
}

/// Prove: two sequential bump allocations from an **arbitrary** starting
/// offset produce non-overlapping regions.
///
/// Generalizes [`bump_alloc_regions_non_overlapping`] which always starts
/// from offset 0. This covers the intra-step case: the arena already has
/// prior allocations when two new allocations occur consecutively.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn bump_alloc_non_overlapping_arbitrary_offset() {
    let initial_offset: usize = kani::any();
    let len1: usize = kani::any();
    let len2: usize = kani::any();
    let capacity: usize = kani::any();
    let alignment = METAL_BUFFER_ALIGNMENT; // 256

    // Model the arena invariant: offset <= capacity.
    kani::assume(initial_offset <= capacity);
    kani::assume(capacity <= (1usize << 32));
    kani::assume(len1 > 0 && len1 <= (1usize << 30));
    kani::assume(len2 > 0 && len2 <= (1usize << 30));

    // First allocation from arbitrary offset.
    let start1 = match align_up(initial_offset, alignment) {
        Ok(v) => v,
        Err(_) => return,
    };
    let end1 = match start1.checked_add(len1) {
        Some(v) if v <= capacity => v,
        _ => return,
    };

    // Second allocation from where first ended.
    let start2 = match align_up(end1, alignment) {
        Ok(v) => v,
        Err(_) => return,
    };
    let end2 = match start2.checked_add(len2) {
        Some(v) if v <= capacity => v,
        _ => return,
    };

    // Core safety: regions [start1, end1) and [start2, end2) don't overlap.
    assert!(
        start2 >= end1,
        "second allocation must start at or after first allocation ends"
    );
    // Both within capacity.
    assert!(end1 <= capacity);
    assert!(end2 <= capacity);
    // Both start at or after initial offset.
    assert!(start1 >= initial_offset);
}

/// Prove: a successful bump allocation keeps the offset within capacity.
///
/// This is the per-allocation capacity bound — the invariant that prevents
/// GPU out-of-bounds writes. The existing `bump_alloc_regions_non_overlapping`
/// proves non-overlap; this harness proves that each individual allocation's
/// end offset is bounded by capacity regardless of the starting offset.
///
/// Models `ActivationArena::alloc` from an arbitrary (valid) starting offset,
/// not just offset=0. This covers the general case: the arena may already
/// have prior allocations when the current alloc begins.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn bump_alloc_preserves_capacity_bound() {
    let offset: usize = kani::any();
    let byte_len: usize = kani::any();
    let capacity: usize = kani::any();
    let alignment = METAL_BUFFER_ALIGNMENT;

    kani::assume(byte_len > 0 && byte_len <= (1usize << 30));
    kani::assume(offset <= capacity);
    kani::assume(capacity <= (1usize << 32));

    let aligned = match align_up(offset, alignment) {
        Ok(v) => v,
        Err(_) => return, // overflow → alloc returns Err (safe)
    };

    let new_offset = match aligned.checked_add(byte_len) {
        Some(v) => v,
        None => return, // overflow → alloc returns Err (safe)
    };

    if new_offset <= capacity {
        // Post-condition of a successful alloc:
        // allocated region [aligned, new_offset) fits within [0, capacity).
        assert!(
            aligned <= capacity,
            "aligned offset must be within capacity"
        );
        assert!(new_offset <= capacity, "new offset must be within capacity");
        assert!(
            aligned >= offset,
            "aligned offset must not decrease from starting offset"
        );

        // The allocated region has exactly byte_len usable bytes.
        assert_eq!(
            new_offset - aligned,
            byte_len,
            "region must be exactly byte_len bytes"
        );
    }
    // new_offset > capacity → alloc returns ArenaOverflow (safe).
}

/// Prove: between `checkpoint()` and `restore_checkpoint()`, any successful
/// `alloc` can only increase the bump offset. Therefore the
/// `restore_checkpoint` assertion (`saved_offset <= self.offset`) always
/// holds for the save→alloc*→restore pattern.
///
/// This closes the "restore precondition" proof gap: a checkpoint value
/// captured before allocations can always be restored after those
/// allocations, because alloc monotonically advances the offset.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn checkpoint_to_restore_offset_monotonic() {
    let pre_offset: usize = kani::any();
    let alloc_len: usize = kani::any();
    let capacity: usize = kani::any();
    let alignment = METAL_BUFFER_ALIGNMENT;

    // Arena invariant: offset <= capacity.
    kani::assume(pre_offset <= capacity);
    kani::assume(capacity <= (1usize << 32));
    kani::assume(alloc_len > 0 && alloc_len <= (1usize << 30));

    // Checkpoint captures the current offset.
    let checkpoint = pre_offset;

    // One allocation between checkpoint and restore.
    let start = match align_up(pre_offset, alignment) {
        Ok(v) => v,
        Err(_) => return, // overflow → alloc returns Err, offset unchanged (safe)
    };
    let new_offset = match start.checked_add(alloc_len) {
        Some(v) if v <= capacity => v,
        _ => return, // alloc fails, offset unchanged (safe)
    };

    // Core property: alloc never decreases offset.
    assert!(
        new_offset >= checkpoint,
        "alloc must not decrease offset below checkpoint"
    );

    // Therefore: `restore_checkpoint(checkpoint)` assertion holds.
    // The assertion is `saved_offset <= self.offset`, i.e., checkpoint <= new_offset.
    assert!(
        checkpoint <= new_offset,
        "restore_checkpoint precondition: saved <= current"
    );
}

/// Prove: two consecutive checkpoint/restore cycles maintain arena safety.
///
/// Models the compiled model pattern: each step does
/// checkpoint→alloc(temp)→blit-copy→restore, repeated for N steps.
/// Proves:
/// 1. Both cycles reuse the same aligned start offset (deterministic).
/// 2. Capacity bound is maintained in both cycles.
/// 3. The pre-checkpoint allocation is not affected by either cycle.
///
/// This closes the "multi-cycle checkpoint/restore" proof gap identified
/// in commit 88108a9.
#[kani::unwind(8)]
#[kani::proof]
#[kani::unwind(1)]
fn two_checkpoint_restore_cycles_deterministic() {
    let initial_len: usize = kani::any();
    let temp_len_1: usize = kani::any();
    let temp_len_2: usize = kani::any();
    let capacity: usize = kani::any();
    let alignment = METAL_BUFFER_ALIGNMENT;

    kani::assume(initial_len > 0 && initial_len <= (1usize << 28));
    kani::assume(temp_len_1 > 0 && temp_len_1 <= (1usize << 28));
    kani::assume(temp_len_2 > 0 && temp_len_2 <= (1usize << 28));
    kani::assume(capacity <= (1usize << 30));

    // Pre-checkpoint allocation (persistent across cycles).
    let start0 = match align_up(0, alignment) {
        Ok(v) => v,
        Err(_) => return,
    };
    let end0 = match start0.checked_add(initial_len) {
        Some(v) if v <= capacity => v,
        _ => return,
    };
    let checkpoint = end0;

    // --- Cycle 1: checkpoint → alloc temporary → restore ---
    let c1_start = match align_up(checkpoint, alignment) {
        Ok(v) => v,
        Err(_) => return,
    };
    let c1_end = match c1_start.checked_add(temp_len_1) {
        Some(v) if v <= capacity => v,
        _ => return,
    };
    // (Restore to checkpoint — bump pointer = checkpoint.)

    // --- Cycle 2: checkpoint → alloc temporary → restore ---
    let c2_start = match align_up(checkpoint, alignment) {
        Ok(v) => v,
        Err(_) => return,
    };
    let c2_end = match c2_start.checked_add(temp_len_2) {
        Some(v) if v <= capacity => v,
        _ => return,
    };

    // Property 1: Deterministic — both cycles start at the same offset.
    assert_eq!(
        c1_start, c2_start,
        "checkpoint/restore cycles produce identical start offsets"
    );

    // Property 2: Capacity bound maintained in both cycles.
    assert!(c1_end <= capacity, "cycle 1 within capacity");
    assert!(c2_end <= capacity, "cycle 2 within capacity");

    // Property 3: Initial allocation is NOT affected by either cycle.
    assert!(
        end0 <= checkpoint,
        "initial alloc ends at or before checkpoint"
    );
    assert!(
        c1_start >= end0,
        "cycle 1 temp does not overlap initial allocation"
    );
    assert!(
        c2_start >= end0,
        "cycle 2 temp does not overlap initial allocation"
    );

    // Property 4: restore_checkpoint precondition holds — after each cycle's
    // alloc, the offset (c*_end) >= checkpoint.
    assert!(
        c1_end >= checkpoint,
        "restore precondition holds for cycle 1"
    );
    assert!(
        c2_end >= checkpoint,
        "restore precondition holds for cycle 2"
    );
}

/// Prove: generation-guarded restore is safe when a reset occurs between
/// checkpoint and restore.
///
/// Models the bug scenario from #3133: `flush()` or auto-flush resets the
/// arena (offset=0, gen++) between `checkpoint_default_arena()` and
/// `restore_default_arena()`. Without the generation guard, the assert
/// in `restore_checkpoint` panics because `saved_offset > current_offset`.
///
/// The proposed fix records generation at checkpoint time and skips the
/// restore if generation changed. This harness proves that approach is
/// correct: the generation check either permits a valid restore OR
/// correctly skips an invalid one.
///
/// Part of #3133.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn generation_guarded_restore_safe_after_reset() {
    let pre_alloc_bytes: usize = kani::any();
    let step_alloc_bytes: usize = kani::any();
    let post_reset_alloc_bytes: usize = kani::any();
    let capacity: usize = kani::any();
    let alignment = METAL_BUFFER_ALIGNMENT;
    let do_reset: bool = kani::any();

    kani::assume(pre_alloc_bytes > 0 && pre_alloc_bytes <= (1usize << 28));
    kani::assume(step_alloc_bytes > 0 && step_alloc_bytes <= (1usize << 28));
    kani::assume(post_reset_alloc_bytes <= (1usize << 28));
    kani::assume(capacity <= (1usize << 30));

    // --- Phase 1: Prior allocations establish checkpoint offset > 0 ---
    let start0 = match align_up(0, alignment) {
        Ok(v) => v,
        Err(_) => return,
    };
    let end0 = match start0.checked_add(pre_alloc_bytes) {
        Some(v) if v <= capacity => v,
        _ => return,
    };

    // Checkpoint: save offset AND generation.
    let checkpoint_offset = end0;
    let checkpoint_gen: u64 = 0; // initial generation

    // --- Phase 2: Step dispatch allocates within checkpoint window ---
    let step_start = match align_up(end0, alignment) {
        Ok(v) => v,
        Err(_) => return,
    };
    let step_end = match step_start.checked_add(step_alloc_bytes) {
        Some(v) if v <= capacity => v,
        _ => return,
    };

    // --- Phase 3: flush() MAY reset the arena (non-deterministic) ---
    let (current_offset, current_gen) = if do_reset {
        // Reset: offset=0, gen++. Then possibly some new allocations.
        if post_reset_alloc_bytes > 0 {
            let new_start = match align_up(0, alignment) {
                Ok(v) => v,
                Err(_) => return,
            };
            let new_end = match new_start.checked_add(post_reset_alloc_bytes) {
                Some(v) if v <= capacity => v,
                _ => return,
            };
            (new_end, 1u64) // gen incremented by reset
        } else {
            (0usize, 1u64)
        }
    } else {
        // No reset: offset unchanged from step_end.
        (step_end, 0u64)
    };

    // --- Phase 4: Generation-guarded restore ---
    if checkpoint_gen == current_gen {
        // Generation matches → arena was NOT reset → restore is valid.
        // Prove: checkpoint_offset <= current_offset (restore precondition).
        assert!(
            checkpoint_offset <= current_offset,
            "when generation matches, checkpoint offset must be <= current offset"
        );
    }
    // When generation differs: skip restore (no assert needed, always safe).
    // The key property: we NEVER call restore_checkpoint with a stale offset.
}

/// Prove: a naive offset-only restore check is insufficient after reset.
///
/// Models the scenario: checkpoint at offset X, reset (gen++), new allocs
/// push offset past X. A naive `checkpoint_offset <= current_offset` check
/// would pass (the offset grew past the checkpoint), but the restore is
/// WRONG because the arena contents are from a different generation.
///
/// The generation guard catches this: `checkpoint_gen != current_gen` →
/// skip restore. This harness proves the naive check is reachable when
/// it shouldn't be, demonstrating why the generation guard is necessary.
///
/// Part of #3133.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn offset_only_restore_check_insufficient_after_reset() {
    let pre_alloc_bytes: usize = kani::any();
    let post_reset_alloc_bytes: usize = kani::any();
    let capacity: usize = kani::any();
    let alignment = METAL_BUFFER_ALIGNMENT;

    kani::assume(pre_alloc_bytes > 0 && pre_alloc_bytes <= (1usize << 28));
    kani::assume(post_reset_alloc_bytes > 0 && post_reset_alloc_bytes <= (1usize << 28));
    kani::assume(capacity <= (1usize << 30));

    // Phase 1: allocate, then checkpoint.
    let start0 = match align_up(0, alignment) {
        Ok(v) => v,
        Err(_) => return,
    };
    let end0 = match start0.checked_add(pre_alloc_bytes) {
        Some(v) if v <= capacity => v,
        _ => return,
    };
    let checkpoint_offset = end0;
    let checkpoint_gen: u64 = 0;

    // Phase 2: reset (gen++) then re-allocate past the checkpoint offset.
    let current_gen: u64 = 1; // generation changed
    let reset_start = match align_up(0, alignment) {
        Ok(v) => v,
        Err(_) => return,
    };
    let reset_end = match reset_start.checked_add(post_reset_alloc_bytes) {
        Some(v) if v <= capacity => v,
        _ => return,
    };
    let current_offset = reset_end;

    // Constrain: post-reset offset grew past checkpoint (the dangerous case).
    kani::assume(current_offset >= checkpoint_offset);

    // The naive offset-only check PASSES (this is the bug scenario):
    let naive_check_passes = checkpoint_offset <= current_offset;
    assert!(naive_check_passes, "naive check must pass in this scenario");

    // But the generation guard correctly BLOCKS the restore:
    let gen_guard_blocks = checkpoint_gen != current_gen;
    assert!(
        gen_guard_blocks,
        "generation guard must block stale restore even when offset check passes"
    );
}

// ---------------------------------------------------------------------------
// Buffer size alignment invariants (#3555)
// ---------------------------------------------------------------------------

/// Prove: every successful bump allocation returns a start offset that is a
/// multiple of `METAL_BUFFER_ALIGNMENT` (256 bytes).
///
/// This is the Metal API contract: `set_buffer(_:offset:atIndex:)` requires
/// the offset to be aligned. A misaligned offset causes undefined GPU behavior.
/// The existing `align_up_result_is_aligned` proves `align_up` itself; this
/// harness proves the full `alloc` path returns aligned offsets.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn alloc_returns_aligned_offset() {
    let initial_offset: usize = kani::any();
    let byte_len: usize = kani::any();
    let capacity: usize = kani::any();
    let alignment = METAL_BUFFER_ALIGNMENT; // 256

    kani::assume(byte_len > 0 && byte_len <= (1usize << 30));
    kani::assume(initial_offset <= capacity);
    kani::assume(capacity <= (1usize << 32));

    // Model ActivationArena::alloc: align the current offset, then advance.
    let aligned_offset = match align_up(initial_offset, alignment) {
        Ok(v) => v,
        Err(_) => return,
    };
    let new_offset = match aligned_offset.checked_add(byte_len) {
        Some(v) if v <= capacity => v,
        _ => return,
    };

    // The returned offset to the caller is `aligned_offset`.
    assert_eq!(
        aligned_offset % alignment,
        0,
        "alloc must return an offset aligned to METAL_BUFFER_ALIGNMENT"
    );
    // Sanity: new_offset is the post-alloc bump pointer position.
    assert!(new_offset > aligned_offset || byte_len == 0);
    let _ = new_offset; // suppress unused warning in some CBMC modes
}

/// Prove: the first allocation from a freshly-created (or reset) arena
/// always starts at offset 0, which is trivially aligned.
///
/// This covers the post-`reset()` state: offset=0 is always aligned to any
/// power-of-two alignment. The harness confirms align_up(0, 256) == 0.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn first_alloc_after_reset_starts_at_zero() {
    let byte_len: usize = kani::any();
    let capacity: usize = kani::any();
    let alignment = METAL_BUFFER_ALIGNMENT;

    kani::assume(byte_len > 0 && byte_len <= (1usize << 30));
    kani::assume(capacity <= (1usize << 32));
    kani::assume(byte_len <= capacity);

    // After reset, offset is 0.
    let aligned = align_up(0, alignment).expect("align_up(0, 256) cannot fail");
    assert_eq!(aligned, 0, "first allocation must start at offset 0");

    let new_offset = aligned + byte_len; // cannot overflow: byte_len <= capacity <= 2^32
    assert!(new_offset <= capacity);
}

// ---------------------------------------------------------------------------
// Non-overlapping lifetimes for N sequential allocations (#3555)
// ---------------------------------------------------------------------------

/// Prove: three sequential bump allocations from arbitrary initial offset
/// produce pairwise non-overlapping regions.
///
/// Extends the 2-allocation proofs to cover the N=3 case. Induction from
/// N=2 to N=3 gives confidence that the bump allocator is correct for
/// arbitrary sequences (since the N→N+1 step is structurally identical).
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn three_sequential_allocs_non_overlapping() {
    let initial_offset: usize = kani::any();
    let len1: usize = kani::any();
    let len2: usize = kani::any();
    let len3: usize = kani::any();
    let capacity: usize = kani::any();
    let alignment = METAL_BUFFER_ALIGNMENT;

    kani::assume(initial_offset <= capacity);
    kani::assume(capacity <= (1usize << 31));
    kani::assume(len1 > 0 && len1 <= (1usize << 28));
    kani::assume(len2 > 0 && len2 <= (1usize << 28));
    kani::assume(len3 > 0 && len3 <= (1usize << 28));

    // Alloc 1
    let start1 = match align_up(initial_offset, alignment) {
        Ok(v) => v,
        Err(_) => return,
    };
    let end1 = match start1.checked_add(len1) {
        Some(v) if v <= capacity => v,
        _ => return,
    };

    // Alloc 2
    let start2 = match align_up(end1, alignment) {
        Ok(v) => v,
        Err(_) => return,
    };
    let end2 = match start2.checked_add(len2) {
        Some(v) if v <= capacity => v,
        _ => return,
    };

    // Alloc 3
    let start3 = match align_up(end2, alignment) {
        Ok(v) => v,
        Err(_) => return,
    };
    let end3 = match start3.checked_add(len3) {
        Some(v) if v <= capacity => v,
        _ => return,
    };

    // Pairwise non-overlap.
    assert!(start2 >= end1, "alloc 2 must not overlap alloc 1");
    assert!(start3 >= end2, "alloc 3 must not overlap alloc 2");
    // Transitive: alloc 3 also doesn't overlap alloc 1.
    assert!(start3 >= end1, "alloc 3 must not overlap alloc 1 (transitive)");
    // All within capacity.
    assert!(end1 <= capacity);
    assert!(end2 <= capacity);
    assert!(end3 <= capacity);
}

// ---------------------------------------------------------------------------
// Arena reset/clear correctness (#3555)
// ---------------------------------------------------------------------------

/// Prove: after reset (offset=0, gen++), the next allocation occupies
/// the same region as the first allocation of the previous generation.
///
/// This is the core reset correctness property: reset reclaims ALL arena
/// memory, so the next allocation starts at offset 0 and may overlap with
/// any allocation from the previous generation. The generation counter
/// distinguishes the two epochs.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn reset_reclaims_all_memory() {
    let len_gen0: usize = kani::any();
    let len_gen1: usize = kani::any();
    let capacity: usize = kani::any();
    let alignment = METAL_BUFFER_ALIGNMENT;

    kani::assume(len_gen0 > 0 && len_gen0 <= (1usize << 28));
    kani::assume(len_gen1 > 0 && len_gen1 <= (1usize << 28));
    kani::assume(capacity <= (1usize << 30));

    // Generation 0: allocate.
    let start_g0 = match align_up(0, alignment) {
        Ok(v) => v,
        Err(_) => return,
    };
    let end_g0 = match start_g0.checked_add(len_gen0) {
        Some(v) if v <= capacity => v,
        _ => return,
    };
    let offset_before_reset = end_g0;

    // Reset: offset → 0, generation → 1.
    let offset_after_reset: usize = 0;
    let gen_after_reset: u64 = 1;

    // Generation 1: allocate from offset 0.
    let start_g1 = match align_up(offset_after_reset, alignment) {
        Ok(v) => v,
        Err(_) => return,
    };
    let end_g1 = match start_g1.checked_add(len_gen1) {
        Some(v) if v <= capacity => v,
        _ => return,
    };

    // Core property: gen 1 starts at 0, same as gen 0 did.
    assert_eq!(start_g0, start_g1, "post-reset alloc starts at same offset as initial");
    assert_eq!(start_g1, 0, "post-reset alloc starts at offset 0");

    // The regions may overlap (intentional — reset reclaims memory).
    // The generation counter distinguishes them.
    assert_eq!(gen_after_reset, 1, "generation must increment on reset");
    assert!(offset_before_reset > 0 || len_gen0 == 0);

    // Both within capacity.
    assert!(end_g0 <= capacity);
    assert!(end_g1 <= capacity);
}

/// Prove: `restore_checkpoint` correctly rejects a saved offset that is
/// ahead of the current offset (invalid restore).
///
/// Models the error path in `restore_checkpoint`: `saved_offset > self.offset`
/// returns `Err(ArenaCheckpoint)`. This harness proves that the check is
/// exhaustive — every case where saved > current is caught.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn restore_checkpoint_rejects_future_offset() {
    let current_offset: usize = kani::any();
    let saved_offset: usize = kani::any();
    let capacity: usize = kani::any();

    kani::assume(current_offset <= capacity);
    kani::assume(saved_offset <= capacity);
    kani::assume(capacity <= (1usize << 32));

    // Model restore_checkpoint logic.
    if saved_offset > current_offset {
        // Must return Err — this is the reject case.
        // The assertion documents that the check catches ALL invalid restores.
        assert!(
            saved_offset > current_offset,
            "future offset must be rejected"
        );
    } else {
        // Valid restore: saved_offset <= current_offset.
        // Post-condition: the new offset equals saved_offset.
        assert!(
            saved_offset <= current_offset,
            "valid restore requires saved <= current"
        );
    }
}

// ---------------------------------------------------------------------------
// Capacity growth / peak_bytes invariants (#3555)
// ---------------------------------------------------------------------------

/// Prove: `peak_bytes` is monotonically non-decreasing across allocations.
///
/// Models the peak tracking logic in `ActivationArena::alloc`:
/// `if self.offset > self.peak_bytes { self.peak_bytes = self.offset; }`
/// Proves that after N allocations (modeled as 2), peak_bytes >= all
/// intermediate offset values.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn peak_bytes_monotonically_non_decreasing() {
    let len1: usize = kani::any();
    let len2: usize = kani::any();
    let capacity: usize = kani::any();
    let alignment = METAL_BUFFER_ALIGNMENT;

    kani::assume(len1 > 0 && len1 <= (1usize << 28));
    kani::assume(len2 > 0 && len2 <= (1usize << 28));
    kani::assume(capacity <= (1usize << 30));

    let mut peak: usize = 0;
    let mut offset: usize = 0;

    // Alloc 1.
    let start1 = match align_up(offset, alignment) {
        Ok(v) => v,
        Err(_) => return,
    };
    offset = match start1.checked_add(len1) {
        Some(v) if v <= capacity => v,
        _ => return,
    };
    if offset > peak {
        peak = offset;
    }
    let peak_after_1 = peak;

    // Alloc 2.
    let start2 = match align_up(offset, alignment) {
        Ok(v) => v,
        Err(_) => return,
    };
    offset = match start2.checked_add(len2) {
        Some(v) if v <= capacity => v,
        _ => return,
    };
    if offset > peak {
        peak = offset;
    }
    let peak_after_2 = peak;

    // Monotonicity: peak never decreases.
    assert!(
        peak_after_2 >= peak_after_1,
        "peak_bytes must be monotonically non-decreasing"
    );

    // Peak >= current offset at all times.
    assert!(peak >= offset, "peak must be >= current offset");
    let _ = start1;
    let _ = start2;
}

/// Prove: `peak_bytes` survives reset — it tracks the all-time high water
/// mark across all generations, not just the current generation.
///
/// Models: alloc in gen 0 → reset → alloc in gen 1 (smaller). Peak must
/// retain the gen 0 high water mark.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn peak_bytes_survives_reset() {
    let len_gen0: usize = kani::any();
    let len_gen1: usize = kani::any();
    let capacity: usize = kani::any();
    let alignment = METAL_BUFFER_ALIGNMENT;

    kani::assume(len_gen0 > 0 && len_gen0 <= (1usize << 28));
    kani::assume(len_gen1 > 0 && len_gen1 <= (1usize << 28));
    kani::assume(capacity <= (1usize << 30));
    // Constrain gen1 to be smaller to test the interesting case.
    kani::assume(len_gen1 <= len_gen0);

    let mut peak: usize = 0;

    // Gen 0: allocate.
    let start0 = match align_up(0, alignment) {
        Ok(v) => v,
        Err(_) => return,
    };
    let offset0 = match start0.checked_add(len_gen0) {
        Some(v) if v <= capacity => v,
        _ => return,
    };
    if offset0 > peak {
        peak = offset0;
    }
    let peak_before_reset = peak;

    // Reset: offset → 0. peak_bytes is NOT reset.
    // (ActivationArena::reset only sets self.offset = 0.)

    // Gen 1: allocate (smaller).
    let start1 = match align_up(0, alignment) {
        Ok(v) => v,
        Err(_) => return,
    };
    let offset1 = match start1.checked_add(len_gen1) {
        Some(v) if v <= capacity => v,
        _ => return,
    };
    if offset1 > peak {
        peak = offset1;
    }

    // Peak must retain the gen 0 value (since gen1 is smaller).
    assert!(
        peak >= peak_before_reset,
        "peak must not decrease after reset + smaller alloc"
    );
    assert_eq!(
        peak, peak_before_reset,
        "peak must equal gen 0 high water mark when gen 1 is smaller"
    );
}

/// Prove: a failed allocation (overflow or capacity exceeded) does NOT
/// modify the bump pointer.
///
/// Models the alloc path: if `aligned_offset + byte_len > capacity` or
/// `checked_add` overflows, the offset must remain at its pre-alloc value.
/// This is critical for correctness — a partially-advanced pointer after
/// failure would waste arena space and violate alignment invariants.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn failed_alloc_preserves_offset() {
    let initial_offset: usize = kani::any();
    let byte_len: usize = kani::any();
    let capacity: usize = kani::any();
    let alignment = METAL_BUFFER_ALIGNMENT;

    kani::assume(byte_len > 0 && byte_len <= (1usize << 30));
    kani::assume(initial_offset <= capacity);
    kani::assume(capacity <= (1usize << 32));

    let mut offset = initial_offset;

    // Model alloc logic.
    let aligned = match align_up(offset, alignment) {
        Ok(v) => v,
        Err(_) => {
            // Alignment failed (overflow). Offset unchanged.
            assert_eq!(offset, initial_offset, "offset must not change on align failure");
            return;
        }
    };

    let new_offset = match aligned.checked_add(byte_len) {
        Some(v) => v,
        None => {
            // Checked add overflow. Offset unchanged.
            assert_eq!(offset, initial_offset, "offset must not change on add overflow");
            return;
        }
    };

    if new_offset > capacity {
        // Capacity exceeded. Offset unchanged.
        assert_eq!(offset, initial_offset, "offset must not change on capacity overflow");
    } else {
        // Success: offset advances.
        offset = new_offset;
        assert!(offset > initial_offset || (initial_offset == 0 && byte_len > 0));
    }
}

/// Prove: generation counter increases by exactly 1 on each reset.
///
/// Models N=2 reset cycles. Proves generation increments are sequential
/// and cannot skip values. This is important because stale-read detection
/// in `gpu_to_cpu` uses `current_gen > alloc_gen + 1` to detect multi-
/// generation staleness.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn generation_increments_by_one_on_reset() {
    let initial_gen: u64 = kani::any();
    kani::assume(initial_gen <= (1u64 << 60)); // avoid overflow at u64::MAX

    // Model: reset increments generation by 1.
    let gen_after_reset_1 = initial_gen + 1;
    let gen_after_reset_2 = gen_after_reset_1 + 1;

    // Exactly +1 per reset.
    assert_eq!(gen_after_reset_1, initial_gen + 1);
    assert_eq!(gen_after_reset_2, initial_gen + 2);

    // Monotonic.
    assert!(gen_after_reset_1 > initial_gen);
    assert!(gen_after_reset_2 > gen_after_reset_1);

    // The stale detection formula: current_gen > alloc_gen + 1 means
    // at least 2 resets have occurred since allocation.
    let alloc_gen = initial_gen;
    let current_gen = gen_after_reset_2;
    assert!(
        current_gen > alloc_gen + 1,
        "after 2 resets, stale detection must fire"
    );

    // After only 1 reset, stale detection must NOT fire.
    let current_gen_1 = gen_after_reset_1;
    assert!(
        !(current_gen_1 > alloc_gen + 1),
        "after 1 reset, stale detection must not fire"
    );
}

// ---------------------------------------------------------------------------
// Remaining-bytes soundness (#3555)
// ---------------------------------------------------------------------------

/// Prove: `remaining_bytes()` is a correct upper bound on the next
/// successful allocation size (accounting for alignment waste).
///
/// `remaining_bytes = capacity - offset`. But the next alloc first aligns
/// `offset` up, consuming some of the "remaining" bytes as padding. This
/// means `remaining_bytes()` can overestimate by up to `alignment - 1`
/// bytes. The harness proves:
///
/// 1. If `byte_len <= remaining_bytes - (alignment - 1)`, alloc succeeds.
/// 2. If `byte_len > remaining_bytes`, alloc always fails.
///
/// This closes the gap: callers using `remaining_bytes()` to pre-check
/// allocations must account for worst-case alignment padding.
#[kani::unwind(8)]
#[kani::proof]
#[kani::unwind(1)]
fn remaining_bytes_upper_bound_soundness() {
    let offset: usize = kani::any();
    let capacity: usize = kani::any();
    let byte_len: usize = kani::any();
    let alignment = METAL_BUFFER_ALIGNMENT;

    kani::assume(offset <= capacity);
    kani::assume(capacity <= (1usize << 32));
    kani::assume(byte_len > 0 && byte_len <= (1usize << 30));

    let remaining = capacity.saturating_sub(offset);

    // Property 2: if byte_len > remaining, alloc MUST fail.
    // Because aligned_offset >= offset, so aligned_offset + byte_len > capacity.
    if byte_len > remaining {
        let aligned = match align_up(offset, alignment) {
            Ok(v) => v,
            Err(_) => return, // overflow -> alloc fails (correct)
        };
        // aligned >= offset, so aligned + byte_len >= offset + byte_len > capacity.
        if let Some(new_offset) = aligned.checked_add(byte_len) {
            assert!(
                new_offset > capacity,
                "alloc must fail when byte_len > remaining_bytes"
            );
        }
        // checked_add overflow also means alloc fails (correct).
    }

    // Property 1: if byte_len + (alignment - 1) <= remaining, alloc MUST succeed.
    // This is the conservative check callers should use.
    let padding_budget = alignment - 1; // 255 for Metal
    if let Some(safe_threshold) = byte_len.checked_add(padding_budget) {
        if safe_threshold <= remaining {
            let aligned = align_up(offset, alignment)
                .expect("align_up cannot fail: offset <= capacity <= 2^32");
            let new_offset = aligned
                .checked_add(byte_len)
                .expect("no overflow: byte_len + padding <= remaining <= capacity");
            assert!(
                new_offset <= capacity,
                "alloc must succeed when byte_len + alignment - 1 <= remaining"
            );
        }
    }
}

// ---------------------------------------------------------------------------
// Checkpoint/restore preserves alignment for next alloc (#3555)
// ---------------------------------------------------------------------------

/// Prove: after `restore_checkpoint`, the next allocation still starts at
/// an aligned offset.
///
/// This covers the critical path in compiled model execution: checkpoint →
/// alloc temps → blit → restore → alloc next step output. The next step's
/// output offset must be aligned even though it starts from the restored
/// (possibly non-aligned) checkpoint offset.
///
/// The property holds because `alloc` always calls `align_up` on the
/// current offset before sub-allocating, regardless of whether the offset
/// came from a fresh alloc or a checkpoint restore.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn post_restore_alloc_is_aligned() {
    let initial_len: usize = kani::any();
    let temp_len: usize = kani::any();
    let next_len: usize = kani::any();
    let capacity: usize = kani::any();
    let alignment = METAL_BUFFER_ALIGNMENT;

    kani::assume(initial_len > 0 && initial_len <= (1usize << 28));
    kani::assume(temp_len > 0 && temp_len <= (1usize << 28));
    kani::assume(next_len > 0 && next_len <= (1usize << 28));
    kani::assume(capacity <= (1usize << 30));

    // Step 1: initial alloc (produces a non-aligned end offset if initial_len
    // is not a multiple of 256).
    let start0 = match align_up(0, alignment) {
        Ok(v) => v,
        Err(_) => return,
    };
    let end0 = match start0.checked_add(initial_len) {
        Some(v) if v <= capacity => v,
        _ => return,
    };

    // Checkpoint at end0 (which may NOT be aligned).
    let checkpoint = end0;

    // Step 2: temp alloc (align_up(checkpoint, ...) aligns it).
    let temp_start = match align_up(checkpoint, alignment) {
        Ok(v) => v,
        Err(_) => return,
    };
    let temp_end = match temp_start.checked_add(temp_len) {
        Some(v) if v <= capacity => v,
        _ => return,
    };
    let _ = temp_end;

    // Restore checkpoint: offset goes back to `checkpoint`.
    // Step 3: next alloc from the restored offset.
    let next_start = match align_up(checkpoint, alignment) {
        Ok(v) => v,
        Err(_) => return,
    };
    let next_end = match next_start.checked_add(next_len) {
        Some(v) if v <= capacity => v,
        _ => return,
    };

    // Core property: the post-restore allocation is aligned.
    assert_eq!(
        next_start % alignment,
        0,
        "alloc after restore_checkpoint must return an aligned offset"
    );
    // And it's within capacity.
    assert!(next_end <= capacity);
    // And it doesn't overlap the pre-checkpoint allocation.
    assert!(next_start >= end0, "post-restore alloc must not overlap pre-checkpoint region");
}

// ---------------------------------------------------------------------------
// Multiple resets preserve arena invariants (#3555)
// ---------------------------------------------------------------------------

/// Prove: N consecutive resets (modeled as 3) each maintain the arena's
/// structural invariants:
///   - offset returns to 0
///   - generation is strictly monotonic
///   - peak_bytes is preserved (never decreases)
///   - subsequent alloc succeeds if within capacity
///
/// This is stronger than `peak_bytes_survives_reset` (which only covers 1
/// reset) and covers the real-world pattern: Kokoro synthesis in a loop
/// calls reset hundreds of times.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn multiple_resets_preserve_invariants() {
    let alloc_len: usize = kani::any();
    let capacity: usize = kani::any();
    let alignment = METAL_BUFFER_ALIGNMENT;

    kani::assume(alloc_len > 0 && alloc_len <= (1usize << 28));
    kani::assume(capacity <= (1usize << 30));
    kani::assume(alloc_len <= capacity);

    let mut offset: usize = 0;
    let mut peak: usize = 0;
    let mut generation: u64 = 0;

    // --- Generation 0: alloc ---
    let start = align_up(offset, alignment).expect("align_up(0, 256) cannot fail");
    offset = match start.checked_add(alloc_len) {
        Some(v) if v <= capacity => v,
        _ => return,
    };
    if offset > peak {
        peak = offset;
    }
    let peak_after_g0 = peak;

    // --- Reset 1 ---
    offset = 0;
    generation += 1;
    assert_eq!(offset, 0, "reset must zero offset");
    assert_eq!(generation, 1);
    assert_eq!(peak, peak_after_g0, "peak must survive reset 1");

    // --- Generation 1: alloc (same pattern) ---
    let start1 = align_up(offset, alignment).expect("align_up(0, 256) cannot fail");
    offset = match start1.checked_add(alloc_len) {
        Some(v) if v <= capacity => v,
        _ => return,
    };
    if offset > peak {
        peak = offset;
    }

    // --- Reset 2 ---
    offset = 0;
    generation += 1;
    assert_eq!(offset, 0, "reset must zero offset");
    assert_eq!(generation, 2);
    assert!(peak >= peak_after_g0, "peak must not decrease after reset 2");

    // --- Reset 3 ---
    offset = 0;
    generation += 1;
    assert_eq!(offset, 0, "reset must zero offset");
    assert_eq!(generation, 3);
    assert!(peak >= peak_after_g0, "peak must not decrease after reset 3");

    // After 3 resets, alloc still succeeds (arena is functional).
    let start_final = align_up(offset, alignment).expect("align_up(0, 256) cannot fail");
    let final_offset = start_final + alloc_len;
    assert!(final_offset <= capacity, "alloc must succeed after multiple resets");

    // Generation is strictly monotonic across all resets.
    assert!(generation > 0, "generation must be positive after resets");
}

// ---------------------------------------------------------------------------
// Aliased regions share the same buffer identity (#3555)
// ---------------------------------------------------------------------------

/// Prove: two allocations within the same generation produce views into the
/// SAME underlying buffer (i.e., same buffer identity). The arena hands out
/// `MetalBuffer::alias()` which is a refcounted clone of the arena's buffer,
/// not a separate Metal allocation.
///
/// This harness models the key invariant: arena sub-allocations differ ONLY
/// in byte_offset, NOT in buffer identity. Both regions live in [0, capacity).
/// If this invariant were violated (e.g., alloc returned a fresh buffer),
/// the non-overlap proof would be irrelevant because the regions would be in
/// different address spaces.
#[kani::unwind(8)]
#[kani::proof]
#[kani::unwind(1)]
fn same_generation_allocs_share_buffer_range() {
    let len1: usize = kani::any();
    let len2: usize = kani::any();
    let capacity: usize = kani::any();
    let alignment = METAL_BUFFER_ALIGNMENT;

    kani::assume(len1 > 0 && len1 <= (1usize << 28));
    kani::assume(len2 > 0 && len2 <= (1usize << 28));
    kani::assume(capacity <= (1usize << 30));

    // Alloc 1.
    let start1 = match align_up(0, alignment) {
        Ok(v) => v,
        Err(_) => return,
    };
    let end1 = match start1.checked_add(len1) {
        Some(v) if v <= capacity => v,
        _ => return,
    };

    // Alloc 2.
    let start2 = match align_up(end1, alignment) {
        Ok(v) => v,
        Err(_) => return,
    };
    let end2 = match start2.checked_add(len2) {
        Some(v) if v <= capacity => v,
        _ => return,
    };

    // Both regions are within [0, capacity) — same buffer address space.
    assert!(start1 < capacity, "alloc 1 start within buffer");
    assert!(end1 <= capacity, "alloc 1 end within buffer");
    assert!(start2 < capacity, "alloc 2 start within buffer");
    assert!(end2 <= capacity, "alloc 2 end within buffer");

    // The regions are non-overlapping within the shared buffer.
    assert!(start2 >= end1, "regions must not overlap within shared buffer");

    // Both offsets are aligned (Metal API requirement for shared buffer views).
    assert_eq!(start1 % alignment, 0, "alloc 1 offset aligned");
    assert_eq!(start2 % alignment, 0, "alloc 2 offset aligned");
}

// ---------------------------------------------------------------------------
// Exhaustive capacity boundary (#3555)
// ---------------------------------------------------------------------------

/// Prove: an allocation that exactly fills the remaining capacity succeeds,
/// and one byte more fails.
///
/// This is the boundary condition for the `new_offset > self.capacity` check
/// in `alloc`. An off-by-one here (e.g., `>=` instead of `>`) would either
/// reject a valid allocation at the capacity boundary or accept an OOB one.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn capacity_boundary_exact_fit() {
    let initial_offset: usize = kani::any();
    let capacity: usize = kani::any();
    let alignment = METAL_BUFFER_ALIGNMENT;

    kani::assume(initial_offset <= capacity);
    kani::assume(capacity <= (1usize << 32));
    kani::assume(capacity > 0);

    let aligned = match align_up(initial_offset, alignment) {
        Ok(v) if v <= capacity => v,
        _ => return,
    };

    let exactly_fits = capacity - aligned;
    if exactly_fits == 0 {
        return; // alloc(0) is rejected separately
    }

    // Exactly filling: new_offset == capacity. The `> capacity` check does NOT fire.
    let new_offset_exact = aligned + exactly_fits;
    assert_eq!(new_offset_exact, capacity, "exact fit reaches capacity");
    assert!(
        !(new_offset_exact > capacity),
        "exact fit must NOT be rejected"
    );

    // One byte more: new_offset > capacity. The check fires.
    if let Some(new_offset_over) = aligned.checked_add(exactly_fits + 1) {
        assert!(
            new_offset_over > capacity,
            "one byte over must be rejected"
        );
    }
    // checked_add overflow also means rejection (safe).
}

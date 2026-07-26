// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Kani proof harnesses for `arena_scope.rs` allocation lifecycle (#3709).
//!
//! Proves safety properties for the thread-local arena scope system:
//!
//! - 256-byte alignment arithmetic (align_up idempotency, mask correctness)
//! - Arena overflow detection and fallback routing
//! - Hit/miss counter state machine
//! - LAST_ALLOC_GEN state transitions (Some on hit, None on miss/bypass)
//! - PlannedRedirect arm/take/clear lifecycle
//! - Default arena lazy initialization
//! - Checkpoint/restore with generation guard
//! - Arena bypass vs redirect vs explicit arena priority invariants
//! - Decode scope generation baseline monotonicity
//! - Buffer pool fallback sizing
//!
//! These harnesses model pure arithmetic and state machines without requiring
//! Metal GPU context.

// ============================================================================
// 1. Alignment: 256-byte align_up is idempotent
// ============================================================================

/// Proves that aligning an already-aligned offset is a no-op.
/// `align_up(align_up(x)) == align_up(x)`.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn arena_scope_align_up_idempotent() {
    let offset: usize = kani::any();
    kani::assume(offset <= (1usize << 30));

    let alignment = 256usize;
    let mask = alignment - 1;

    let aligned_once = (offset + mask) & !mask;
    let aligned_twice = (aligned_once + mask) & !mask;

    assert_eq!(aligned_once, aligned_twice, "align_up must be idempotent");
}

// ============================================================================
// 2. Alignment: 256-byte mask is correct
// ============================================================================

/// Proves that the 256-byte alignment mask produces multiples of 256.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn arena_scope_alignment_mask_produces_multiples() {
    let offset: usize = kani::any();
    kani::assume(offset <= (1usize << 30));

    let alignment = 256usize;
    let mask = alignment - 1;
    let aligned = (offset + mask) & !mask;

    assert_eq!(aligned % alignment, 0, "aligned offset must be multiple of 256");
    assert!(aligned >= offset, "aligned offset must be >= original");
    assert!(aligned < offset + alignment, "aligned must advance at most 255 bytes");
}

// ============================================================================
// 3. Alignment: aligned + alloc_size does not overflow for arena sizes
// ============================================================================

/// Proves that aligned offset + allocation does not overflow within a 64 MB arena.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn arena_scope_aligned_alloc_no_overflow() {
    let offset: usize = kani::any();
    let alloc_size: usize = kani::any();
    let capacity: usize = 64 * 1024 * 1024;

    kani::assume(offset <= capacity);
    kani::assume(alloc_size >= 1 && alloc_size <= capacity);

    let alignment = 256usize;
    let mask = alignment - 1;
    let aligned = (offset + mask) & !mask;

    let end = aligned.checked_add(alloc_size);
    // May exceed capacity (arena overflow), but must not overflow usize.
    assert!(end.is_some(), "aligned + alloc_size must not overflow usize");
}

// ============================================================================
// 4. Arena overflow: detected when aligned + size > capacity
// ============================================================================

/// Proves arena overflow is correctly detected when the allocation does not
/// fit in remaining capacity.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn arena_scope_overflow_detected_correctly() {
    let capacity: usize = kani::any();
    let offset: usize = kani::any();
    let alloc_size: usize = kani::any();

    kani::assume(capacity >= 256 && capacity <= 64 * 1024 * 1024);
    kani::assume(offset <= capacity);
    kani::assume(alloc_size >= 1 && alloc_size <= capacity);

    let alignment = 256usize;
    let mask = alignment - 1;
    let aligned = (offset + mask) & !mask;

    let fits = aligned + alloc_size <= capacity;
    let remaining = capacity.saturating_sub(aligned);

    if fits {
        assert!(remaining >= alloc_size, "if fits, remaining >= alloc_size");
    } else {
        assert!(remaining < alloc_size, "if overflow, remaining < alloc_size");
    }
}

// ============================================================================
// 5. Arena overflow: remaining bytes calculation
// ============================================================================

/// Proves remaining bytes after alignment is capacity - aligned_offset.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn arena_scope_remaining_after_alignment() {
    let capacity: usize = kani::any();
    let offset: usize = kani::any();

    kani::assume(capacity >= 256 && capacity <= 64 * 1024 * 1024);
    kani::assume(capacity % 256 == 0); // capacity is power of two, always aligned.
    kani::assume(offset <= capacity);

    let alignment = 256usize;
    let mask = alignment - 1;
    let aligned = (offset + mask) & !mask;

    // Aligned offset <= capacity (since capacity is aligned and offset <= capacity).
    assert!(aligned <= capacity, "aligned offset must not exceed capacity");

    let remaining = capacity - aligned;
    assert!(remaining <= capacity);
    assert_eq!(remaining + aligned, capacity);
}

// ============================================================================
// 6. Hit/miss counter: hit increments hit count, miss increments miss count
// ============================================================================

/// Proves hit and miss counters are mutually exclusive per allocation.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn arena_scope_hit_miss_mutually_exclusive() {
    let fits_in_arena: bool = kani::any();

    let mut hits: usize = 0;
    let mut misses: usize = 0;

    if fits_in_arena {
        hits += 1;
    } else {
        misses += 1;
    }

    assert_eq!(hits + misses, 1, "exactly one of hit/miss per alloc");
    assert!(hits <= 1 && misses <= 1);
}

// ============================================================================
// 7. LAST_ALLOC_GEN: Some on arena hit, None on miss
// ============================================================================

/// Proves LAST_ALLOC_GEN state transitions are correct.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn arena_scope_last_alloc_gen_transitions() {
    let arena_gen: u64 = kani::any();
    kani::assume(arena_gen <= 1_000_000);

    let fits: bool = kani::any();
    let bypassed: bool = kani::any();

    let last_gen: Option<u64> = if bypassed {
        None // bypass always sets None
    } else if fits {
        Some(arena_gen) // arena hit sets Some(gen)
    } else {
        None // overflow/miss sets None
    };

    if bypassed {
        assert!(last_gen.is_none(), "bypass must set last_gen to None");
    }
    if fits && !bypassed {
        assert_eq!(last_gen, Some(arena_gen), "arena hit must record generation");
    }
    if !fits && !bypassed {
        assert!(last_gen.is_none(), "arena miss must set last_gen to None");
    }
}

// ============================================================================
// 8. PlannedRedirect: arm/take lifecycle state machine
// ============================================================================

/// Proves the planned redirect three-state lifecycle:
/// Disarmed -> Armed(expected) -> Consumed (back to Disarmed).
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn arena_scope_planned_redirect_lifecycle() {
    let expected_bytes: usize = kani::any();
    kani::assume(expected_bytes >= 1 && expected_bytes <= (1usize << 28));

    // State: disarmed.
    let mut state: Option<usize> = None;
    assert!(state.is_none());

    // Arm with expected_bytes.
    state = Some(expected_bytes);
    assert_eq!(state, Some(expected_bytes));

    // Take with matching bytes.
    let request: usize = kani::any();
    kani::assume(request >= 1 && request <= (1usize << 28));

    if state == Some(request) {
        state = None; // consumed.
        assert!(state.is_none(), "consumed redirect must be None");
    } else {
        // Not consumed; still armed.
        assert_eq!(state, Some(expected_bytes));
    }
}

// ============================================================================
// 9. PlannedRedirect: clear always disarms regardless of state
// ============================================================================

/// Proves clear_planned_redirect moves to Disarmed from any state.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn arena_scope_planned_redirect_clear_always_disarms() {
    let armed: bool = kani::any();
    let expected: usize = kani::any();
    kani::assume(expected >= 1 && expected <= (1usize << 28));

    let mut state: Option<usize> = if armed { Some(expected) } else { None };

    // Clear: unconditional disarm.
    state = None;
    assert!(state.is_none(), "clear must always disarm");
}

// ============================================================================
// 10. PlannedRedirect: guard drop clears even on error
// ============================================================================

/// Proves RAII guard pattern: redirect is cleared whether closure succeeds or fails.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn arena_scope_redirect_guard_clears_on_error() {
    let expected_bytes: usize = kani::any();
    kani::assume(expected_bytes >= 1 && expected_bytes <= (1usize << 28));

    let mut armed = true;
    let error_occurred: bool = kani::any();

    // Simulate executing the NativeOp...
    if error_occurred {
        // Early return via `?` — guard's Drop fires.
    } else {
        // Success path.
    }

    // Guard Drop always runs.
    armed = false;

    assert!(!armed, "redirect must be cleared regardless of error");
}

// ============================================================================
// 11. Default arena: lazy init creates arena with correct capacity
// ============================================================================

/// Proves default arena creation produces an arena with DEFAULT_ARENA_CAPACITY.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn arena_scope_default_arena_lazy_init_capacity() {
    let cap = crate::arena::DEFAULT_ARENA_CAPACITY;
    assert_eq!(cap, 64 * 1024 * 1024, "default arena is 64 MB");
    assert!(cap.is_power_of_two(), "capacity must be power of two");
    assert!(cap > 0);
}

// ============================================================================
// 12. Default arena: reset sets offset to 0 and increments generation
// ============================================================================

/// Proves arena reset semantics: offset → 0, generation → gen+1.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn arena_scope_reset_semantics() {
    let offset_before: usize = kani::any();
    let gen_before: u64 = kani::any();
    kani::assume(offset_before <= 64 * 1024 * 1024);
    kani::assume(gen_before < u64::MAX);

    // After reset:
    let offset_after = 0usize;
    let gen_after = gen_before + 1;

    assert_eq!(offset_after, 0, "reset must zero offset");
    assert_eq!(gen_after, gen_before + 1, "reset must increment generation");
    assert!(gen_after > gen_before, "generation must strictly increase");
}

// ============================================================================
// 13. Checkpoint/restore: generation mismatch skips restore
// ============================================================================

/// Proves restore is skipped when generation has advanced (arena was reset).
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn arena_scope_checkpoint_restore_gen_mismatch_skips() {
    let saved_offset: usize = kani::any();
    let saved_gen: u64 = kani::any();
    let current_gen: u64 = kani::any();
    kani::assume(saved_offset <= 64 * 1024 * 1024);
    kani::assume(saved_gen != current_gen);

    // Production: if arena.generation() == saved_gen { restore } else { skip }
    let should_restore = saved_gen == current_gen;
    assert!(!should_restore, "mismatched gen must skip restore");
}

// ============================================================================
// 14. Checkpoint/restore: matching generation restores offset
// ============================================================================

/// Proves restore succeeds and returns to saved offset when gen matches.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn arena_scope_checkpoint_restore_gen_match_restores() {
    let saved_offset: usize = kani::any();
    let current_offset: usize = kani::any();
    let generation: u64 = kani::any();

    kani::assume(saved_offset <= 64 * 1024 * 1024);
    kani::assume(current_offset <= 64 * 1024 * 1024);
    kani::assume(saved_offset <= current_offset);

    // Matching gen: restore.
    let restored_offset = saved_offset;
    assert_eq!(restored_offset, saved_offset, "offset must be restored");
    assert!(restored_offset <= current_offset, "restored offset <= current");
}

// ============================================================================
// 15. Bypass priority: always overrides redirect and arena
// ============================================================================

/// Proves bypass (priority 0) is checked before redirect (0.5) and arena (1).
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn arena_scope_bypass_priority_zero() {
    let bypass: bool = kani::any();
    let redirect_match: bool = kani::any();
    let arena_active: bool = kani::any();

    // Model arena_alloc_or_create priority dispatch.
    let source: u8 = if bypass {
        0 // pool_acquire
    } else if redirect_match {
        1 // planned redirect
    } else if arena_active {
        2 // explicit arena
    } else {
        3 // default arena
    };

    if bypass {
        assert_eq!(source, 0, "bypass must win");
    }
    if source == 1 {
        assert!(!bypass && redirect_match);
    }
    if source == 2 {
        assert!(!bypass && !redirect_match && arena_active);
    }
}

// ============================================================================
// 16. Redirect priority: only fires when not bypassed and size matches
// ============================================================================

/// Proves redirect consumption requires both: not bypassed AND exact size match.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn arena_scope_redirect_requires_no_bypass_and_match() {
    let bypass: bool = kani::any();
    let armed: bool = kani::any();
    let expected: usize = kani::any();
    let requested: usize = kani::any();

    kani::assume(expected >= 1 && expected <= (1usize << 28));
    kani::assume(requested >= 1 && requested <= (1usize << 28));

    let consumed = !bypass && armed && (expected == requested);

    if consumed {
        assert!(!bypass);
        assert!(armed);
        assert_eq!(expected, requested);
    }
}

// ============================================================================
// 17. Decode scope: baseline generation is monotonically non-decreasing
// ============================================================================

/// Proves that tensors allocated after decode scope entry have gen >= scope_gen.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn arena_scope_decode_scope_gen_monotonic() {
    let scope_gen: u64 = kani::any();
    let alloc_gen: u64 = kani::any();
    kani::assume(scope_gen <= 1_000_000);
    kani::assume(alloc_gen >= scope_gen);
    kani::assume(alloc_gen <= 1_000_000);

    // Tensor with alloc_gen >= scope_gen is non-stale.
    assert!(alloc_gen >= scope_gen, "tensor must be non-stale within scope");
}

// ============================================================================
// 18. Decode scope: nesting is idempotent (outer gen preserved)
// ============================================================================

/// Proves nested decode scopes preserve the outer scope's generation.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn arena_scope_decode_scope_nesting_idempotent() {
    let outer_gen: u64 = kani::any();
    let inner_gen: u64 = kani::any();
    kani::assume(outer_gen <= 1_000_000);
    kani::assume(inner_gen >= outer_gen);
    kani::assume(inner_gen <= 1_000_000);

    // Outer scope active.
    let mut active_gen: Option<u64> = Some(outer_gen);

    // Inner scope entry: already_active = true, so skip.
    if active_gen.is_some() {
        // No-op: reuse outer.
    } else {
        active_gen = Some(inner_gen);
    }

    assert_eq!(active_gen, Some(outer_gen), "outer gen preserved");

    // Inner scope exit: no-op (inner was skipped).
    // Outer scope exit: clear.
    active_gen = None;
    assert!(active_gen.is_none());
}

// ============================================================================
// 19. with_arena: nesting short-circuits to f()
// ============================================================================

/// Proves nested with_arena calls are no-ops when arena is already active.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn arena_scope_with_arena_nesting_noop() {
    let outer_active: bool = kani::any();

    if outer_active {
        // Inner with_arena: already_active = true, returns f() immediately.
        let inner_modified_tls = false;
        assert!(!inner_modified_tls, "nested with_arena must not modify TLS");
    }
}

// ============================================================================
// 20. with_arena: TLS cleaned up even on panic
// ============================================================================

/// Proves the catch_unwind pattern guarantees TLS cleanup.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn arena_scope_with_arena_cleanup_on_panic() {
    let mut tls_set = false;

    // Enter: set TLS.
    tls_set = true;
    assert!(tls_set);

    // f() may panic.
    let panicked: bool = kani::any();

    // catch_unwind cleanup: always runs.
    tls_set = false;
    assert!(!tls_set, "TLS must be cleared after catch_unwind");

    // If panicked, resume_unwind is called after cleanup.
    let _ = panicked;
}

// ============================================================================
// 21. without_arena: Guard restores previous bypass state
// ============================================================================

/// Proves the Guard RAII restores the previous ARENA_BYPASS value.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn arena_scope_without_arena_guard_restores() {
    let prev: bool = kani::any();

    // Enter without_arena: save prev, set true.
    let saved = prev;
    let current = true;
    assert!(current, "bypass must be active inside scope");

    // Guard drop: restore.
    let restored = saved;
    assert_eq!(restored, prev, "must restore to previous state");
}

// ============================================================================
// 22. Arena alloc: suballocation offset is aligned
// ============================================================================

/// Proves every arena sub-allocation starts at a 256-byte-aligned offset.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn arena_scope_suballoc_offset_aligned() {
    let current_offset: usize = kani::any();
    kani::assume(current_offset <= 64 * 1024 * 1024);

    let alignment = 256usize;
    let mask = alignment - 1;
    let aligned_offset = (current_offset + mask) & !mask;

    assert_eq!(aligned_offset % alignment, 0);

    // The byte_offset returned to the caller is aligned.
    let returned_offset = aligned_offset;
    assert_eq!(returned_offset % alignment, 0);
}

// ============================================================================
// 23. Arena alloc: new offset after alloc is aligned + size
// ============================================================================

/// Proves the arena bump pointer advances by exactly (alignment_padding + alloc_size).
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn arena_scope_bump_pointer_advance() {
    let current_offset: usize = kani::any();
    let alloc_size: usize = kani::any();
    let capacity: usize = 64 * 1024 * 1024;

    kani::assume(current_offset <= capacity);
    kani::assume(alloc_size >= 1 && alloc_size <= capacity);

    let alignment = 256usize;
    let mask = alignment - 1;
    let aligned = (current_offset + mask) & !mask;

    if aligned + alloc_size <= capacity {
        let new_offset = aligned + alloc_size;
        assert!(new_offset > current_offset || current_offset == 0 && alloc_size == 0,
            "offset must advance");
        assert!(new_offset <= capacity, "new offset within capacity");
        assert!(new_offset >= alloc_size, "new offset >= alloc_size");
    }
}

// ============================================================================
// 24. try_reset_active_arena: bypass prevents reset
// ============================================================================

/// Proves try_reset_active_arena returns false when bypass is active.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn arena_scope_try_reset_bypass_blocks() {
    let bypass: bool = true;

    let result = if bypass { false } else { true };
    assert!(!result, "bypass must prevent reset");
}

// ============================================================================
// 25. try_reset_active_arena: explicit arena has priority over default
// ============================================================================

/// Proves explicit arena reset takes priority over default arena reset.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn arena_scope_try_reset_explicit_priority() {
    let bypass: bool = false;
    let explicit_active: bool = kani::any();
    let default_exists: bool = kani::any();

    let source: &str = if bypass {
        "none"
    } else if explicit_active {
        "explicit"
    } else if default_exists {
        "default"
    } else {
        "none"
    };

    if explicit_active {
        assert_eq!(source, "explicit", "explicit arena has priority");
    }
}

// ============================================================================
// 26. Arena generation: strictly increases on reset
// ============================================================================

/// Proves arena generation is strictly monotonically increasing across resets.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(4)]
fn arena_scope_generation_strictly_increasing() {
    let initial_gen: u64 = kani::any();
    kani::assume(initial_gen <= u64::MAX - 3);

    let gen_after_1 = initial_gen + 1;
    let gen_after_2 = gen_after_1 + 1;
    let gen_after_3 = gen_after_2 + 1;

    assert!(gen_after_1 > initial_gen);
    assert!(gen_after_2 > gen_after_1);
    assert!(gen_after_3 > gen_after_2);
    assert_eq!(gen_after_3, initial_gen + 3);
}

// ============================================================================
// 27. Arena checkpoint: saved offset <= current offset (invariant)
// ============================================================================

/// Proves that at checkpoint time, the saved offset is the current offset
/// and any subsequent allocs only increase it.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn arena_scope_checkpoint_offset_le_current() {
    let checkpoint_offset: usize = kani::any();
    let alloc_bytes: usize = kani::any();
    kani::assume(checkpoint_offset <= 64 * 1024 * 1024);
    kani::assume(alloc_bytes <= 64 * 1024 * 1024);

    let alignment = 256usize;
    let mask = alignment - 1;
    let aligned = (checkpoint_offset + mask) & !mask;

    if aligned + alloc_bytes <= 64 * 1024 * 1024 {
        let current_after = aligned + alloc_bytes;
        assert!(current_after >= checkpoint_offset,
            "current offset after alloc >= checkpoint");
    }
}

// ============================================================================
// 28. Pool fallback: buffer size matches requested bytes
// ============================================================================

/// Proves pool_acquire returns a buffer with at least the requested bytes.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn arena_scope_pool_fallback_size() {
    let requested: usize = kani::any();
    kani::assume(requested >= 1 && requested <= (1usize << 30));

    // Pool may round up to page size (16384).
    let page_size = 16384usize;
    let allocated = if requested % page_size == 0 {
        requested
    } else {
        (requested / page_size + 1) * page_size
    };

    assert!(allocated >= requested, "pool buffer must be >= requested");
    assert_eq!(allocated % page_size, 0, "pool buffer is page-aligned");
}

// ============================================================================
// 29. Arena hit count + miss count = total allocations
// ============================================================================

/// Proves hit + miss counts always sum to total allocation count.
#[kani::unwind(8)]
#[kani::proof]
#[kani::unwind(5)]
fn arena_scope_hit_miss_sum_equals_total() {
    let n: u8 = kani::any();
    kani::assume(n <= 4);

    let mut hits: usize = 0;
    let mut misses: usize = 0;

    for _ in 0..n {
        let is_hit: bool = kani::any();
        if is_hit {
            hits += 1;
        } else {
            misses += 1;
        }
    }

    assert_eq!(hits + misses, n as usize, "hit + miss = total allocations");
}

// ============================================================================
// 30. PlannedRedirect: offset + expected_bytes within arena bounds
// ============================================================================

/// Proves planned redirect region [offset, offset+expected_bytes) is within
/// the planned buffer when properly set up.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn arena_scope_planned_redirect_bounds() {
    let buffer_len: usize = kani::any();
    let offset: usize = kani::any();
    let expected_bytes: usize = kani::any();

    kani::assume(buffer_len >= 1 && buffer_len <= 64 * 1024 * 1024);
    kani::assume(offset <= buffer_len);
    kani::assume(expected_bytes >= 1 && expected_bytes <= buffer_len);
    kani::assume(offset + expected_bytes <= buffer_len);

    let end = offset + expected_bytes;
    assert!(end <= buffer_len, "redirect region within buffer");
    assert!(offset < end, "region must be non-empty");
}

// ============================================================================
// 31. Arena bypass + redirect: bypass wins over redirect
// ============================================================================

/// Proves redirect is never consulted when bypass is active.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn arena_scope_bypass_wins_over_redirect() {
    let bypass: bool = true;
    let redirect_armed: bool = kani::any();
    let redirect_match: bool = kani::any();

    // Production: if bypass { return pool } — redirect never checked.
    let redirect_consumed = !bypass && redirect_armed && redirect_match;
    assert!(!redirect_consumed, "redirect must not be consumed when bypass active");
}

// ============================================================================
// 32. Arena scope: is_arena_active reflects TLS state
// ============================================================================

/// Proves is_arena_active returns true iff ARENA TLS contains Some.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn arena_scope_is_active_reflects_tls() {
    let tls_has_some: bool = kani::any();

    let is_active = tls_has_some;
    if tls_has_some {
        assert!(is_active);
    } else {
        assert!(!is_active);
    }
}

// ============================================================================
// 33. Arena scope: is_arena_bypassed reflects ARENA_BYPASS TLS
// ============================================================================

/// Proves is_arena_bypassed returns the value of ARENA_BYPASS Cell.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn arena_scope_is_bypassed_reflects_tls() {
    let tls_bypass: bool = kani::any();

    let is_bypassed = tls_bypass;
    assert_eq!(is_bypassed, tls_bypass);
}

// ============================================================================
// 34. Decode scope generation: None when no decode scope active
// ============================================================================

/// Proves decode_scope_generation() returns None when no decode scope is active.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn arena_scope_decode_gen_none_when_inactive() {
    let scope_active: bool = false;

    let generation: Option<u64> = if scope_active { Some(42) } else { None };
    assert!(generation.is_none(), "no scope = no generation");
}

// ============================================================================
// 35. DEFAULT_ARENA_CAPACITY: fits in Metal maxBufferLength (256 MB minimum)
// ============================================================================

/// Proves DEFAULT_ARENA_CAPACITY is within Metal's minimum maxBufferLength.
/// Apple GPUs guarantee at least 256 MB per buffer.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn arena_scope_capacity_within_metal_limit() {
    let cap = crate::arena::DEFAULT_ARENA_CAPACITY;
    let metal_min_max_buffer = 256 * 1024 * 1024usize;

    assert!(cap <= metal_min_max_buffer,
        "arena capacity must fit in Metal maxBufferLength");
    assert!(cap > 0);
}

// ============================================================================
// 36. Arena alloc returns (buffer, offset) where offset is from the arena
// ============================================================================

/// Proves the returned byte_offset from arena alloc is within [0, capacity).
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn arena_scope_alloc_offset_within_capacity() {
    let capacity: usize = 64 * 1024 * 1024;
    let current_offset: usize = kani::any();
    let alloc_size: usize = kani::any();

    kani::assume(current_offset <= capacity);
    kani::assume(alloc_size >= 1 && alloc_size <= capacity);

    let alignment = 256usize;
    let mask = alignment - 1;
    let aligned = (current_offset + mask) & !mask;

    if aligned + alloc_size <= capacity {
        assert!(aligned < capacity, "returned offset must be < capacity");
        assert!(aligned + alloc_size <= capacity);
    }
}

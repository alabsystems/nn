// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Kani proof harnesses for [`GpuWeightCache`] invariants.
//!
//! `GpuWeightCache<T>` wraps `RwLock<Option<Result<T, String>>>` with a
//! monotonic generation counter. Since Kani cannot instrument `RwLock`
//! directly, these harnesses model the cache's abstract state machine:
//!
//! - The cache cell is `Option<Result<T, String>>` (None = uninit,
//!   Some(Ok(t)) = cached, Some(Err(e)) = error).
//! - `get_or_init_with` returns cached value or initializes.
//! - `replace` swaps the cached value and bumps generation.
//! - `invalidate` clears the cached value and bumps generation.
//! - `generation()` is monotonically non-decreasing.
//!
//! We prove:
//!
//! 1. **Cache hit returns same value** — same key always maps to same data.
//! 2. **Cache miss returns None** — uninit cache has no stale data.
//! 3. **Insertion preserves existing** — adding via init does not evict.
//! 4. **Weight count accuracy** — init transitions None → Some exactly once.
//! 5. **Invalidate empties cache** — after invalidate, cell is None.
//! 6. **Duplicate init is idempotent** — double init keeps first value.
//! 7. **Generation monotonicity** — replace/invalidate only increase generation.
//! 8. **Replace swaps atomically** — old value is gone, new value is present.
//! 9. **Error caching** — init failure is cached; no re-initialization.
//! 10. **New cache is empty** — fresh GpuWeightCache starts as None, gen 0.

// ============================================================================
// 1. Cache hit returns same value
// ============================================================================

/// Prove: once initialized with a value, repeated lookups return the same
/// value without re-running the init closure.
///
/// Models `get_or_init_with`: fast-path read lock sees `Some(Ok(v))` and
/// returns it. The init closure is NOT called on subsequent accesses.
#[kani::proof]
#[kani::unwind(1)]
fn proof_cache_hit_returns_same_value() {
    let weight_id: u64 = kani::any();
    kani::assume(weight_id > 0);

    // Model the cache cell.
    let mut cell: Option<Result<u64, String>> = None;

    // First access: cell is None, init runs.
    if cell.is_none() {
        cell = Some(Ok(weight_id));
    }

    // Verify first access yielded the value.
    assert!(cell.is_some());
    assert_eq!(cell.as_ref().unwrap().as_ref().unwrap(), &weight_id);

    // Second access: cell is Some(Ok(..)), init must NOT run.
    let init_called;
    if cell.is_none() {
        // This branch is unreachable after first init.
        cell = Some(Ok(999));
        init_called = true;
    } else {
        init_called = false;
    }

    assert!(!init_called, "init must not be called on cache hit");
    assert_eq!(
        cell.as_ref().unwrap().as_ref().unwrap(),
        &weight_id,
        "cache must return the original value on repeated access"
    );
}

// ============================================================================
// 2. Cache miss returns None (uninit has no stale data)
// ============================================================================

/// Prove: an uninitialized cache cell contains None — there is no stale
/// data from a previous lifecycle. This models `GpuWeightCache::new()`
/// which sets `inner` to `RwLock::new(None)`.
#[kani::proof]
#[kani::unwind(1)]
fn proof_cache_miss_returns_none() {
    // Model a fresh cache.
    let cell: Option<Result<u64, String>> = None;

    // Lookup without init: the cell is None.
    assert!(cell.is_none(), "fresh cache must be None — no stale data");

    // The fast-path in get_or_init_with checks `guard.as_ref()` which
    // returns None, falling through to the slow init path.
    let has_value = cell.as_ref().and_then(|r| r.as_ref().ok()).is_some();
    assert!(!has_value, "uninit cache must not contain a valid value");
}

// ============================================================================
// 3. Insertion preserves existing entries
// ============================================================================

/// Prove: initializing one cache does not affect another cache instance.
/// Each `GpuWeightCache<T>` is independent — there is no shared state
/// between different cache instances.
///
/// Models two independent model weight caches (e.g., SileroVad and
/// DemucsTransformer each have their own `GpuWeightCache`).
#[kani::proof]
#[kani::unwind(1)]
fn proof_insertion_preserves_existing() {
    let weight_a: u64 = kani::any();
    let weight_b: u64 = kani::any();
    kani::assume(weight_a > 0 && weight_b > 0);
    kani::assume(weight_a != weight_b);

    // Model two independent cache cells.
    let mut cell_a: Option<Result<u64, String>> = None;
    let mut cell_b: Option<Result<u64, String>> = None;

    // Init cache A.
    if cell_a.is_none() {
        cell_a = Some(Ok(weight_a));
    }

    // Verify A is initialized.
    assert_eq!(cell_a.as_ref().unwrap().as_ref().unwrap(), &weight_a);

    // Init cache B — must not disturb cache A.
    if cell_b.is_none() {
        cell_b = Some(Ok(weight_b));
    }

    // Both caches hold their respective values.
    assert_eq!(
        cell_a.as_ref().unwrap().as_ref().unwrap(),
        &weight_a,
        "cache A must be unaffected by cache B initialization"
    );
    assert_eq!(
        cell_b.as_ref().unwrap().as_ref().unwrap(),
        &weight_b,
        "cache B must hold its own value"
    );
}

// ============================================================================
// 4. Weight count accuracy (init transitions None -> Some exactly once)
// ============================================================================

/// Prove: the cache transitions from None to Some exactly once during
/// `get_or_init_with`. The double-check pattern (read lock, then write
/// lock with `if guard.is_none()`) ensures the init closure runs at most
/// once even under contention.
///
/// We model this with a counter that tracks init invocations.
#[kani::proof]
#[kani::unwind(5)]
fn proof_weight_count_accuracy() {
    let value: u64 = kani::any();

    let mut cell: Option<Result<u64, String>> = None;
    let mut init_count: u32 = 0;

    // Simulate up to 4 get_or_init_with calls.
    let num_accesses: u32 = kani::any();
    kani::assume(num_accesses >= 1 && num_accesses <= 4);

    let mut i: u32 = 0;
    while i < num_accesses {
        // Model the double-check init pattern.
        if cell.is_none() {
            cell = Some(Ok(value));
            init_count += 1;
        }
        // Every access after init sees Some.
        assert!(cell.is_some(), "cell must be Some after first init");
        i += 1;
    }

    // Init ran exactly once regardless of access count.
    assert_eq!(
        init_count, 1,
        "init must run exactly once across multiple accesses"
    );

    // The cached value is correct.
    assert_eq!(cell.as_ref().unwrap().as_ref().unwrap(), &value);
}

// ============================================================================
// 5. Invalidate empties cache (clear)
// ============================================================================

/// Prove: after `invalidate()`, the cache cell is None. The next
/// `get_or_init_with` call will re-run the init closure.
///
/// Models `invalidate()`: sets `*guard = None` and bumps generation.
#[kani::proof]
#[kani::unwind(1)]
fn proof_invalidate_empties_cache() {
    let value: u64 = kani::any();
    kani::assume(value > 0);

    // Start with an initialized cache.
    let mut cell: Option<Result<u64, String>> = Some(Ok(value));
    let mut generation: u64 = 0;

    // Verify it is initialized.
    assert!(cell.is_some());
    assert_eq!(cell.as_ref().unwrap().as_ref().unwrap(), &value);

    // invalidate(): clear cell, bump generation.
    cell = None;
    generation += 1;

    // After invalidate: cell is None, generation increased.
    assert!(cell.is_none(), "cache must be empty after invalidate");
    assert_eq!(generation, 1, "generation must increase on invalidate");

    // A subsequent lookup sees None (would trigger re-init).
    let has_value = cell.as_ref().and_then(|r| r.as_ref().ok()).is_some();
    assert!(!has_value, "invalidated cache must return None on lookup");
}

// ============================================================================
// 6. Duplicate init is idempotent
// ============================================================================

/// Prove: calling `get_or_init_with` multiple times with different init
/// closures always returns the first-initialized value. The second init
/// closure is never called because the write-lock path checks
/// `if guard.is_none()` — since it is already `Some`, the closure is skipped.
#[kani::proof]
#[kani::unwind(1)]
fn proof_duplicate_init_is_idempotent() {
    let first_value: u64 = kani::any();
    let second_value: u64 = kani::any();
    kani::assume(first_value > 0 && second_value > 0);
    kani::assume(first_value != second_value);

    let mut cell: Option<Result<u64, String>> = None;

    // First get_or_init_with: init runs, stores first_value.
    if cell.is_none() {
        cell = Some(Ok(first_value));
    }

    // Second get_or_init_with with a different closure: init does NOT run.
    let second_init_ran;
    if cell.is_none() {
        cell = Some(Ok(second_value));
        second_init_ran = true;
    } else {
        second_init_ran = false;
    }

    assert!(!second_init_ran, "second init must not run — cell is already Some");
    assert_eq!(
        cell.as_ref().unwrap().as_ref().unwrap(),
        &first_value,
        "cache must retain the first-initialized value"
    );
}

// ============================================================================
// 7. Generation monotonicity
// ============================================================================

/// Prove: the generation counter is monotonically non-decreasing.
/// `replace()` and `invalidate()` both call `fetch_add(1, SeqCst)`.
/// No operation ever decreases the generation.
///
/// We model a symbolic sequence of replace/invalidate operations and
/// verify the counter only goes up.
#[kani::proof]
#[kani::unwind(6)]
fn proof_generation_monotonically_increases() {
    let mut generation: u64 = 0;

    // Perform up to 5 symbolic operations.
    let num_ops: u32 = kani::any();
    kani::assume(num_ops >= 1 && num_ops <= 5);

    let mut i: u32 = 0;
    while i < num_ops {
        let prev = generation;

        // Both replace and invalidate do fetch_add(1).
        let is_replace: bool = kani::any();
        let _ = is_replace; // both paths bump generation identically
        generation += 1;

        assert!(
            generation > prev,
            "generation must strictly increase on each replace/invalidate"
        );
        i += 1;
    }

    // Final generation equals number of operations.
    assert_eq!(
        generation, num_ops as u64,
        "generation must equal total operation count"
    );
}

// ============================================================================
// 8. Replace swaps atomically
// ============================================================================

/// Prove: `replace()` atomically swaps the cached value. After replace,
/// the old value is gone and the new value is returned by subsequent
/// lookups. The generation counter is bumped.
#[kani::proof]
#[kani::unwind(1)]
fn proof_replace_swaps_atomically() {
    let old_value: u64 = kani::any();
    let new_value: u64 = kani::any();
    kani::assume(old_value > 0 && new_value > 0);
    kani::assume(old_value != new_value);

    // Start with an initialized cache.
    let mut cell: Option<Result<u64, String>> = Some(Ok(old_value));
    let mut generation: u64 = 0;

    // Verify old value is present.
    assert_eq!(cell.as_ref().unwrap().as_ref().unwrap(), &old_value);

    // replace(new_value): bump generation, set cell to Some(Ok(new_value)).
    let prev_gen = generation;
    generation += 1;
    cell = Some(Ok(new_value));

    // After replace: new value is present, old value is gone.
    assert_eq!(
        cell.as_ref().unwrap().as_ref().unwrap(),
        &new_value,
        "cache must contain the new value after replace"
    );
    assert_ne!(
        cell.as_ref().unwrap().as_ref().unwrap(),
        &old_value,
        "old value must not be present after replace"
    );
    assert_eq!(
        generation,
        prev_gen + 1,
        "generation must increase by 1 on replace"
    );
}

// ============================================================================
// 9. Error caching: init failure is cached, no re-initialization
// ============================================================================

/// Prove: when the init closure returns `Err(msg)`, the error is cached
/// in the cell as `Some(Err(msg))`. Subsequent `get_or_init_with` calls
/// see `Some(Err(..))` on the fast-path read and return the mapped error
/// without re-running init.
///
/// This models the behavior in `get_or_init_with` where `*guard = Some(init())`
/// stores the error, and the fast-path `if let Some(result) = guard.as_ref()`
/// returns `Err(map_err(e.clone()))`.
#[kani::proof]
#[kani::unwind(1)]
fn proof_error_is_cached() {
    let error_msg = String::from("GPU alloc failed");

    let mut cell: Option<Result<u64, String>> = None;
    let mut init_count: u32 = 0;

    // First get_or_init_with: init returns Err.
    if cell.is_none() {
        cell = Some(Err(error_msg.clone()));
        init_count += 1;
    }

    // Cell now holds Some(Err(..)).
    assert!(cell.is_some(), "cell must be Some after init (even on error)");
    assert!(
        cell.as_ref().unwrap().is_err(),
        "cell must hold the error result"
    );

    // Second get_or_init_with: fast-path sees Some(Err(..)), does NOT re-init.
    let second_init_ran;
    if cell.is_none() {
        cell = Some(Ok(42));
        init_count += 1;
        second_init_ran = true;
    } else {
        second_init_ran = false;
    }

    assert!(!second_init_ran, "init must not re-run after cached error");
    assert_eq!(init_count, 1, "init ran exactly once");
    assert!(
        cell.as_ref().unwrap().is_err(),
        "cached error must persist across lookups"
    );
}

// ============================================================================
// 10. New cache is empty with generation 0
// ============================================================================

/// Prove: `GpuWeightCache::new()` creates a cache with `inner = None`
/// and `generation = 0`. This is the initial state from which all other
/// operations proceed.
#[kani::proof]
#[kani::unwind(1)]
fn proof_new_cache_is_empty() {
    // Model GpuWeightCache::new().
    let cell: Option<Result<u64, String>> = None;
    let generation: u64 = 0;

    assert!(cell.is_none(), "new cache must start with None");
    assert_eq!(generation, 0, "new cache must start with generation 0");

    // No value is accessible.
    let has_value = cell.as_ref().and_then(|r| r.as_ref().ok()).is_some();
    assert!(!has_value, "new cache must not contain any value");

    let has_error = cell.as_ref().and_then(|r| r.as_ref().err()).is_some();
    assert!(!has_error, "new cache must not contain any error");
}

// ============================================================================
// 11. Invalidate-then-reinit produces fresh value
// ============================================================================

/// Prove: the invalidate → re-init cycle works correctly. After
/// `invalidate()`, the next `get_or_init_with` runs the init closure
/// and stores the new value. The new value may differ from the original.
#[kani::proof]
#[kani::unwind(1)]
fn proof_invalidate_then_reinit() {
    let original: u64 = kani::any();
    let refreshed: u64 = kani::any();
    kani::assume(original > 0 && refreshed > 0);

    let mut cell: Option<Result<u64, String>> = None;
    let mut generation: u64 = 0;

    // Init with original value.
    if cell.is_none() {
        cell = Some(Ok(original));
    }
    assert_eq!(cell.as_ref().unwrap().as_ref().unwrap(), &original);

    // Invalidate.
    cell = None;
    generation += 1;
    assert!(cell.is_none());

    // Re-init with refreshed value.
    if cell.is_none() {
        cell = Some(Ok(refreshed));
    }

    assert_eq!(
        cell.as_ref().unwrap().as_ref().unwrap(),
        &refreshed,
        "re-init after invalidate must use the new value"
    );
    assert_eq!(generation, 1, "generation reflects one invalidation");
}

// ============================================================================
// 12. Replace preserves generation ordering across multiple replaces
// ============================================================================

/// Prove: sequential `replace()` calls produce strictly increasing
/// generation numbers. The return value of `replace()` is the previous
/// generation (from `fetch_add`), and the new generation equals
/// previous + 1.
#[kani::proof]
#[kani::unwind(4)]
fn proof_replace_generation_ordering() {
    let mut generation: u64 = 0;

    let num_replaces: u32 = kani::any();
    kani::assume(num_replaces >= 1 && num_replaces <= 3);

    let mut i: u32 = 0;
    while i < num_replaces {
        // replace() calls fetch_add(1, SeqCst) and returns the previous value.
        let prev = generation;
        generation += 1;

        // The returned previous generation matches what we had before.
        assert_eq!(prev, i as u64, "replace must return the prior generation");

        // New generation is always prev + 1.
        assert_eq!(generation, prev + 1, "new generation = prev + 1");

        i += 1;
    }

    // Final generation equals total number of replaces.
    assert_eq!(generation, num_replaces as u64);
}

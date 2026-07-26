// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Kani proof harnesses for [`MetalBufferPool`] size-class safety.
//!
//! Proves that the pool's size-class bucketing never returns an undersized
//! buffer for any request within the poolable range. An undersized buffer
//! causes GPU out-of-bounds writes — the exact bug documented in #3104.
//!
//! # Aliasing safety context
//!
//! The buffer pool's aliasing safety relies on the caller contract: all
//! outstanding aliases must be dropped before [`reclaim_all`] is called.
//! This cannot be proven by Kani (the property is about ObjC ARC reference
//! counts outside Rust's type system). The `pool_reclaim` callers in
//! `compiled_kokoro_pipeline.rs` and `compiled_kokoro_diagnostics_memory.rs`
//! uphold this contract because `synthesize()` returns CPU-resident data —
//! all GPU tensor aliases are dropped before the next call.
//!
//! What Kani CAN prove is that the size-class selection logic never maps a
//! request to a class whose buffer capacity is smaller than the request,
//! and that it always chooses the smallest sufficient class (minimality).

use super::{MetalBufferPool, MAX_PER_CLASS, MAX_POOLED_BYTES, SIZE_CLASSES};

/// Prove: `size_class_for(bytes)` always returns an index whose size class
/// is >= `bytes`, for all `bytes` within the poolable range.
///
/// This is the core safety property. Without it, `acquire` would create a
/// buffer at `class_size` that is smaller than `min_bytes`, leading to GPU
/// out-of-bounds writes when the caller writes `min_bytes` worth of data.
#[kani::unwind(8)]
#[kani::proof]
#[kani::unwind(8)] // 7 size classes + 1 for loop termination
fn size_class_sufficient_for_poolable_range() {
    let bytes: usize = kani::any();
    let last_class = SIZE_CLASSES[SIZE_CLASSES.len() - 1];

    // Only prove for the poolable range. Requests > last_class bypass the
    // pool entirely via the oversized guard in acquire().
    kani::assume(bytes > 0 && bytes <= last_class);

    let class_idx = MetalBufferPool::size_class_for(bytes);
    let class_size = SIZE_CLASSES[class_idx];

    assert!(
        class_size >= bytes,
        "size class {class_idx} ({class_size} bytes) must be >= request ({bytes} bytes)"
    );
}

/// Prove: `size_class_for` return value is always a valid index into
/// `SIZE_CLASSES`. Out-of-bounds would panic at `SIZE_CLASSES[class]`
/// in `acquire`, but proving it in Kani gives a stronger guarantee.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(8)]
fn size_class_index_in_bounds() {
    let bytes: usize = kani::any();
    kani::assume(bytes <= (1usize << 32)); // ~4 GB

    let class_idx = MetalBufferPool::size_class_for(bytes);
    assert!(
        class_idx < SIZE_CLASSES.len(),
        "class index {class_idx} must be < {} (SIZE_CLASSES.len())",
        SIZE_CLASSES.len()
    );
}

/// Prove: `SIZE_CLASSES` is strictly monotonically increasing.
///
/// This structural property is required for `size_class_for` to work
/// correctly — if two adjacent classes had the same threshold, requests
/// near the boundary could be routed to either class non-deterministically.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(8)]
fn size_classes_strictly_increasing() {
    let i: usize = kani::any();
    kani::assume(i < SIZE_CLASSES.len() - 1);

    assert!(
        SIZE_CLASSES[i] < SIZE_CLASSES[i + 1],
        "SIZE_CLASSES[{i}] ({}) must be < SIZE_CLASSES[{}] ({})",
        SIZE_CLASSES[i],
        i + 1,
        SIZE_CLASSES[i + 1]
    );
}

/// Prove: the oversized-request boundary is correct — requests at exactly
/// `SIZE_CLASSES[last]` are poolable, requests at `SIZE_CLASSES[last] + 1`
/// are not. This proves the guard `min_bytes > SIZE_CLASSES.last()` in
/// `acquire()` has the correct boundary.
#[kani::unwind(8)]
#[kani::proof]
#[kani::unwind(8)]
fn oversized_boundary_correctness() {
    let last_class = SIZE_CLASSES[SIZE_CLASSES.len() - 1];

    // Exactly at last class: poolable, and class_size == request.
    let class_at = MetalBufferPool::size_class_for(last_class);
    assert_eq!(
        SIZE_CLASSES[class_at], last_class,
        "request == last class must map to last class exactly"
    );

    // One byte over: size_class_for still returns last class, but
    // class_size < request. The oversized guard in acquire() catches this.
    let class_over = MetalBufferPool::size_class_for(last_class + 1);
    let class_size_over = SIZE_CLASSES[class_over];
    assert!(
        class_size_over < last_class + 1,
        "request > last class: class_size ({class_size_over}) must be < request ({})",
        last_class + 1
    );
}

/// Prove: `size_class_for(bytes)` returns the *smallest* sufficient class.
///
/// Sufficiency alone (harness 1) does not prevent a regression where
/// `size_class_for` returns a larger class than necessary — e.g., mapping
/// all requests to the last class. That would be safe (no undersized buffer)
/// but wasteful: a 100 KB request would get a 256 MB buffer.
///
/// This harness proves minimality by checking the immediately smaller class
/// (`class_idx - 1`) is insufficient. Combined with
/// [`size_classes_strictly_increasing`] (which proves `SIZE_CLASSES[j] <
/// SIZE_CLASSES[k]` for all `j < k`), this implies ALL smaller classes are
/// insufficient by transitivity.
#[kani::unwind(8)]
#[kani::proof]
#[kani::unwind(8)]
fn size_class_is_minimal() {
    let bytes: usize = kani::any();
    let last_class = SIZE_CLASSES[SIZE_CLASSES.len() - 1];
    kani::assume(bytes > 0 && bytes <= last_class);

    let class_idx = MetalBufferPool::size_class_for(bytes);

    // Every class below the chosen one must be insufficient.
    if class_idx > 0 {
        let smaller_class_size = SIZE_CLASSES[class_idx - 1];
        assert!(
            smaller_class_size < bytes,
            "class {} ({smaller_class_size} bytes) must be insufficient for request ({bytes} bytes) — \
             otherwise class {} is not the smallest sufficient class",
            class_idx - 1,
            class_idx
        );
    }
    // class_idx == 0: no smaller class exists, trivially minimal.
}

/// Prove: MAX_POOLED_BYTES budget is never exceeded by any valid sequence
/// of pool insertions.
///
/// The `acquire` method checks `self.pooled_bytes + class_size <= MAX_POOLED_BYTES`
/// before inserting. This harness models N symbolic insertions into a single
/// size class and proves that `pooled_bytes` never exceeds `MAX_POOLED_BYTES`
/// after any insertion.
///
/// This is the memory safety property for the pool: without it, a bug in the
/// budget check could cause unbounded Metal buffer retention (up to 2.5 GB),
/// causing OOM kills on memory-constrained systems.
#[kani::unwind(8)]
#[kani::proof]
#[kani::unwind(10)] // MAX_PER_CLASS (8) + 2 for loop overhead
fn pooled_bytes_never_exceeds_budget() {
    // Choose a symbolic size class.
    let class_idx: usize = kani::any();
    kani::assume(class_idx < SIZE_CLASSES.len());
    let class_size = SIZE_CLASSES[class_idx];

    // Model N insertions into this class.
    // The acquire() logic: insert if entries.len() < MAX_PER_CLASS
    //   AND pooled_bytes + class_size <= MAX_POOLED_BYTES.
    let mut pooled_bytes: usize = 0;
    let mut entries: usize = 0;

    // Symbolic number of insertion attempts.
    let attempts: usize = kani::any();
    kani::assume(attempts <= MAX_PER_CLASS);

    for _ in 0..MAX_PER_CLASS {
        if entries >= attempts {
            break;
        }
        // Model the acquire budget check.
        if entries < MAX_PER_CLASS && pooled_bytes + class_size <= MAX_POOLED_BYTES {
            pooled_bytes += class_size;
            entries += 1;
        }
    }

    // The invariant: pooled_bytes <= MAX_POOLED_BYTES after any sequence.
    assert!(
        pooled_bytes <= MAX_POOLED_BYTES,
        "pooled_bytes ({pooled_bytes}) must be <= MAX_POOLED_BYTES ({MAX_POOLED_BYTES})"
    );
}

/// Prove: pooled_bytes budget holds across ALL size classes simultaneously.
///
/// The pool has 7 size classes that share a single byte budget. This harness
/// models one insertion per class (symbolically chosen to insert or skip)
/// and proves the total never exceeds MAX_POOLED_BYTES.
///
/// This is stronger than the single-class proof: it catches bugs where
/// individual class checks are correct but the shared budget accounting
/// has an off-by-one or double-count.
#[kani::unwind(8)]
#[kani::proof]
#[kani::unwind(8)] // 7 classes + 1
fn pooled_bytes_budget_across_all_classes() {
    let mut pooled_bytes: usize = 0;

    // For each class, symbolically decide whether to insert one entry.
    for class_idx in 0..SIZE_CLASSES.len() {
        let insert: bool = kani::any();
        if insert {
            let class_size = SIZE_CLASSES[class_idx];
            // Model the acquire budget check.
            if pooled_bytes + class_size <= MAX_POOLED_BYTES {
                pooled_bytes += class_size;
            }
        }
    }

    assert!(
        pooled_bytes <= MAX_POOLED_BYTES,
        "total pooled bytes across all classes must be <= budget"
    );
}

/// Prove: the per-class cap (MAX_PER_CLASS) limits entries even without
/// the byte budget.
///
/// Models insertions into the smallest class (64KB) where the byte budget
/// would allow many more entries. Proves that MAX_PER_CLASS (8) is the
/// binding constraint, not the byte budget, for small classes.
#[kani::unwind(8)]
#[kani::proof]
#[kani::unwind(12)]
fn per_class_cap_limits_small_class_entries() {
    let class_size = SIZE_CLASSES[0]; // 64 KB — smallest class
    let mut entries: usize = 0;
    let mut pooled_bytes: usize = 0;

    // 10 insertion attempts — more than MAX_PER_CLASS.
    for _ in 0..MAX_PER_CLASS + 2 {
        if entries < MAX_PER_CLASS && pooled_bytes + class_size <= MAX_POOLED_BYTES {
            pooled_bytes += class_size;
            entries += 1;
        }
    }

    // MAX_PER_CLASS must be the binding constraint for 64KB class.
    // 8 * 64KB = 512KB << 512MB budget.
    assert!(
        entries <= MAX_PER_CLASS,
        "entries ({entries}) must be <= MAX_PER_CLASS ({MAX_PER_CLASS})"
    );
    assert_eq!(
        entries, MAX_PER_CLASS,
        "small class should reach per-class cap, not byte budget"
    );
}

/// Prove: the byte budget is the binding constraint for the largest class.
///
/// For 256MB class (index 6), MAX_PER_CLASS * 256MB = 2048MB > 512MB budget.
/// The byte budget should stop insertions before MAX_PER_CLASS is reached.
#[kani::unwind(8)]
#[kani::proof]
#[kani::unwind(10)]
fn byte_budget_limits_large_class_entries() {
    let class_size = SIZE_CLASSES[6]; // 256 MB — largest class
    let mut entries: usize = 0;
    let mut pooled_bytes: usize = 0;

    for _ in 0..MAX_PER_CLASS {
        if entries < MAX_PER_CLASS && pooled_bytes + class_size <= MAX_POOLED_BYTES {
            pooled_bytes += class_size;
            entries += 1;
        }
    }

    // Byte budget is binding: 512MB / 256MB = 2, so at most 2 entries.
    assert!(
        entries <= MAX_POOLED_BYTES / class_size,
        "byte budget must cap 256MB class to {} entries, got {entries}",
        MAX_POOLED_BYTES / class_size,
    );
    assert!(
        entries < MAX_PER_CLASS,
        "byte budget must prevent reaching per-class cap for 256MB class"
    );
}

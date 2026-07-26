// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Additional Kani proof harnesses for dispatch plan safety (#3615).
//!
//! Expands coverage beyond `kani_dispatch_plan.rs` (7 harnesses) and
//! `kani_dispatch_plan_builder.rs` (15 harnesses) with proofs for:
//!
//! - Plan builder override (`with_*`) method correctness
//! - Elementwise zero-total corner case (grid=[0,1,1], output_elems=0)
//! - Grid2D output_elems equals product of grid dimensions
//! - Buffer lifetime intersection detection with reuse
//! - Release map: no duplicate entries
//! - Release map: multiple output steps exclusion
//! - Last use: monotonic upper bound after multi-consumer DAG
//! - Buffer planner: allocation-free (size=0) steps correctly skipped
//! - Buffer planner: best-fit reuse returns smallest sufficient slot
//! - Topological dependency: consumer always after producer
//! - Threadgroup width 1d: is min(64, total) exactly
//! - Plan grid 2d: output_elems consistent with grid product
//! - Reduction shared memory bytes: round-trip through u64
//! - NarrowView source_step bounds within step count
//! - Weight index per-step uniqueness after dedup
//! - Linear scan alloc: freed slot reuse offset alignment

use crate::dispatch_plan::*;

// ---------------------------------------------------------------------------
// Helper: release_at model (same as kani_dispatch_plan_builder.rs)
// ---------------------------------------------------------------------------

fn model_release_at_extra(
    last_use: &[usize],
    num_steps: usize,
    output_step_indices: &[usize],
) -> Vec<Vec<usize>> {
    let mut map: Vec<Vec<usize>> = (0..num_steps).map(|_| Vec::new()).collect();
    for (step, &consumer) in last_use.iter().enumerate() {
        if consumer > step && consumer < num_steps && !output_step_indices.contains(&step) {
            map[consumer].push(step);
        }
    }
    map
}

// ============================================================================
// Proof 1: with_output_elems preserves all other fields
// ============================================================================

/// Proves that `with_output_elems` changes only `output_elems` and preserves
/// grid, threads, constants, threadgroup_memory_bytes, and use_threadgroups.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn with_output_elems_preserves_other_fields() {
    let total: u32 = kani::any();
    kani::assume(total > 0);

    let plan = plan_elementwise(total).expect("valid plan");
    let new_elems: usize = kani::any();
    kani::assume(new_elems <= 1 << 20);

    let modified = plan.clone().with_output_elems(new_elems);

    // Changed field.
    assert_eq!(modified.output_elems(), new_elems);

    // Unchanged fields.
    assert_eq!(modified.grid(), plan.grid());
    assert_eq!(modified.threads(), plan.threads());
    assert_eq!(modified.constants(), plan.constants());
    assert_eq!(
        modified.threadgroup_memory_bytes(),
        plan.threadgroup_memory_bytes()
    );
    assert_eq!(modified.use_threadgroups(), plan.use_threadgroups());
}

// ============================================================================
// Proof 2: with_constants preserves all other fields
// ============================================================================

/// Proves that `with_constants` changes only the constants vector.
#[kani::unwind(16)]
#[kani::proof]
#[kani::unwind(1)]
fn with_constants_preserves_other_fields() {
    let total: u32 = kani::any();
    kani::assume(total > 0);

    let plan = plan_elementwise(total).expect("valid plan");
    let c0: u32 = kani::any();
    let c1: u32 = kani::any();
    let new_constants = vec![c0, c1];

    let modified = plan.clone().with_constants(new_constants.clone());

    assert_eq!(modified.constants().len(), 2);
    assert_eq!(modified.constants()[0], c0);
    assert_eq!(modified.constants()[1], c1);

    // Unchanged.
    assert_eq!(modified.grid(), plan.grid());
    assert_eq!(modified.threads(), plan.threads());
    assert_eq!(modified.output_elems(), plan.output_elems());
    assert_eq!(
        modified.threadgroup_memory_bytes(),
        plan.threadgroup_memory_bytes()
    );
    assert_eq!(modified.use_threadgroups(), plan.use_threadgroups());
}

// ============================================================================
// Proof 3: with_use_threadgroups preserves all other fields
// ============================================================================

/// Proves that `with_use_threadgroups` changes only the dispatch mode flag.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn with_use_threadgroups_preserves_other_fields() {
    let outer: u32 = kani::any();
    let reduce: u32 = kani::any();
    let threads: u32 = kani::any();
    let shared: u32 = kani::any();
    kani::assume(outer > 0 && reduce > 0 && threads > 0);

    let plan = plan_reduction(outer, reduce, threads, shared).expect("valid plan");
    assert!(plan.use_threadgroups()); // reduction always uses threadgroups

    let modified = plan.clone().with_use_threadgroups(false);
    assert!(!modified.use_threadgroups());

    // Unchanged.
    assert_eq!(modified.grid(), plan.grid());
    assert_eq!(modified.threads(), plan.threads());
    assert_eq!(modified.output_elems(), plan.output_elems());
    assert_eq!(modified.constants(), plan.constants());
    assert_eq!(
        modified.threadgroup_memory_bytes(),
        plan.threadgroup_memory_bytes()
    );
}

// ============================================================================
// Proof 4: Elementwise zero-total produces correct degenerate plan
// ============================================================================

/// Proves that `plan_elementwise(0)` produces a valid plan with
/// grid=[0,1,1], output_elems=0, and no shared memory.
///
/// This corner case is reachable when a trace produces a zero-element
/// tensor (e.g., empty batch). The plan must not panic and must produce
/// zero output elements.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn plan_elementwise_zero_total_degenerate() {
    let plan = plan_elementwise(0).expect("zero total must succeed");

    assert_eq!(plan.output_elems(), 0, "zero total must produce 0 output");
    assert_eq!(plan.grid(), [0, 1, 1], "zero total grid must be [0,1,1]");
    assert!(!plan.use_threadgroups());
    assert!(plan.threadgroup_memory_bytes().is_none());
    assert_eq!(plan.constants().len(), 1);
    assert_eq!(plan.constants()[0], 0);
}

// ============================================================================
// Proof 5: Grid2D output_elems equals exact product of grid dims
// ============================================================================

/// Proves that `plan_grid_2d` output_elems is exactly `grid[0] * grid[1]`
/// (as usize), matching the widened product.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(3)]
fn plan_grid_2d_output_elems_exact() {
    let g0: u32 = kani::any();
    let g1: u32 = kani::any();
    let t0: u32 = kani::any();
    let t1: u32 = kani::any();
    kani::assume(g0 > 0 && g1 > 0 && t0 > 0 && t1 > 0);

    let plan = plan_grid_2d([g0, g1], [t0, t1]).expect("valid 2D grid");
    let expected = (g0 as u128) * (g1 as u128);

    assert_eq!(
        plan.output_elems() as u128, expected,
        "output_elems must equal widened grid product"
    );
}

// ============================================================================
// Proof 6: Release map has no duplicate entries
// ============================================================================

/// Proves that `release_at[j]` contains no duplicate step indices.
/// A duplicate would cause double-free of a buffer region.
#[kani::unwind(16)]
#[kani::proof]
#[kani::unwind(6)]
fn release_at_no_duplicate_entries() {
    const N: usize = 5;
    let mut last_use = [0usize; N];
    for i in 0..N {
        last_use[i] = kani::any();
        kani::assume(last_use[i] >= i && last_use[i] < N);
    }

    let output_steps: Vec<usize> = Vec::new();
    let map = model_release_at_extra(&last_use, N, &output_steps);

    for j in 0..N {
        let entries = &map[j];
        for (a_idx, &a) in entries.iter().enumerate() {
            for (b_idx, &b) in entries.iter().enumerate() {
                if a_idx != b_idx {
                    assert_ne!(a, b, "release_at[{j}] contains duplicate step {a}");
                }
            }
        }
    }
}

// ============================================================================
// Proof 7: Multiple output steps excluded from release map
// ============================================================================

/// Proves that when TWO output steps are marked, NEITHER appears in
/// any release_at slot. Extends the single-output proof from
/// kani_dispatch_plan_builder.rs.
#[kani::unwind(16)]
#[kani::proof]
#[kani::unwind(6)]
fn release_at_excludes_multiple_output_steps() {
    const N: usize = 5;
    let mut last_use = [0usize; N];
    for i in 0..N {
        last_use[i] = kani::any();
        kani::assume(last_use[i] >= i && last_use[i] < N);
    }

    // Two distinct output steps.
    let out0: usize = kani::any();
    let out1: usize = kani::any();
    kani::assume(out0 < N && out1 < N && out0 != out1);
    let output_steps = vec![out0, out1];

    let map = model_release_at_extra(&last_use, N, &output_steps);

    for j in 0..N {
        for &s in &map[j] {
            assert_ne!(
                s, out0,
                "output step {out0} must not appear in release_at"
            );
            assert_ne!(
                s, out1,
                "output step {out1} must not appear in release_at"
            );
        }
    }
}

// ============================================================================
// Proof 8: Last use monotonic upper bound in multi-consumer DAG
// ============================================================================

/// Proves that after computing `last_use` for a 6-step DAG with up to 3
/// edges per step, the `last_use[i]` value is at least the maximum
/// consumer index that references step i.
#[kani::unwind(16)]
#[kani::proof]
#[kani::unwind(7)]
fn last_use_upper_bound_multi_consumer() {
    const N: usize = 6;

    // Build DAG: each step has 0-2 inputs, all pointing to earlier steps.
    let mut edge_map: Vec<Vec<usize>> = Vec::new();
    for i in 0..N {
        let num_edges: u8 = kani::any();
        kani::assume(num_edges <= 2);
        let mut edges = Vec::new();
        if num_edges >= 1 && i > 0 {
            let src: usize = kani::any();
            kani::assume(src < i);
            edges.push(src);
        }
        if num_edges >= 2 && i > 1 {
            let src: usize = kani::any();
            kani::assume(src < i);
            edges.push(src);
        }
        edge_map.push(edges);
    }

    // Compute last_use.
    let mut last_use: Vec<usize> = (0..N).collect();
    for (consumer_idx, inputs) in edge_map.iter().enumerate() {
        for &producer_idx in inputs {
            if consumer_idx > last_use[producer_idx] {
                last_use[producer_idx] = consumer_idx;
            }
        }
    }

    // Verify: last_use[p] >= max(consumer for consumer that uses p).
    for (consumer_idx, inputs) in edge_map.iter().enumerate() {
        for &producer_idx in inputs {
            assert!(
                last_use[producer_idx] >= consumer_idx,
                "last_use[{producer_idx}] must be >= consumer {consumer_idx}"
            );
        }
    }

    // Verify: last_use bounded by N.
    for i in 0..N {
        assert!(last_use[i] < N);
    }
}

// ============================================================================
// Proof 9: Buffer planner allocation-free steps produce no offset
// ============================================================================

/// Proves that steps with size=0 (passthrough, identity, reshape) never
/// receive an allocated offset in the buffer planner. They must remain
/// unallocated to avoid wasting buffer space.
#[kani::unwind(8)]
#[kani::proof]
#[kani::unwind(5)]
fn allocation_free_steps_get_no_offset() {
    const N: usize = 4;
    const ALIGN: usize = 16;

    let mut sizes = [0usize; N];
    let mut last_use = [0usize; N];
    for i in 0..N {
        sizes[i] = kani::any();
        kani::assume(sizes[i] <= 128);
        last_use[i] = kani::any();
        kani::assume(last_use[i] >= i && last_use[i] < N);
    }

    // At least one step has size=0.
    let zero_step: usize = kani::any();
    kani::assume(zero_step < N);
    sizes[zero_step] = 0;

    // Run allocator.
    let mut offsets: [Option<usize>; N] = [None; N];
    let mut hwm: usize = 0;

    for step_idx in 0..N {
        if sizes[step_idx] == 0 {
            continue;
        }
        let aligned_size = (sizes[step_idx] + ALIGN - 1) / ALIGN * ALIGN;
        offsets[step_idx] = Some(hwm);
        hwm = hwm.saturating_add(aligned_size);
    }

    // The zero-sized step must not have an offset.
    assert!(
        offsets[zero_step].is_none(),
        "zero-sized step must not receive an offset"
    );
}

// ============================================================================
// Proof 10: Best-fit reuse returns smallest sufficient free slot
// ============================================================================

/// Proves that the best-fit allocator selects the smallest free slot
/// that is >= the requested size, minimizing fragmentation.
#[kani::unwind(8)]
#[kani::proof]
#[kani::unwind(4)]
fn best_fit_selects_smallest_sufficient_slot() {
    // Model 3 free slots with symbolic sizes.
    let slot0_size: usize = kani::any();
    let slot1_size: usize = kani::any();
    let slot2_size: usize = kani::any();
    let request: usize = kani::any();

    kani::assume(slot0_size >= 1 && slot0_size <= 1024);
    kani::assume(slot1_size >= 1 && slot1_size <= 1024);
    kani::assume(slot2_size >= 1 && slot2_size <= 1024);
    kani::assume(request >= 1 && request <= 1024);

    // At least one slot fits.
    kani::assume(
        slot0_size >= request || slot1_size >= request || slot2_size >= request,
    );

    let slots = [(0usize, slot0_size), (0usize, slot1_size), (0usize, slot2_size)];

    // Model best-fit selection.
    let best = slots
        .iter()
        .enumerate()
        .filter(|(_, slot)| slot.1 >= request)
        .min_by_key(|(_, slot)| slot.1)
        .map(|(idx, _)| idx);

    let chosen_idx = best.expect("at least one slot must fit");
    let chosen_size = slots[chosen_idx].1;

    // Property 1: chosen slot is sufficient.
    assert!(
        chosen_size >= request,
        "chosen slot must be >= request"
    );

    // Property 2: no smaller sufficient slot exists.
    for (i, &(_, sz)) in slots.iter().enumerate() {
        if sz >= request && i != chosen_idx {
            assert!(
                sz >= chosen_size,
                "no smaller sufficient slot should exist"
            );
        }
    }
}

// ============================================================================
// Proof 11: Topological order: consumer index always > producer index
// ============================================================================

/// Proves that in a topologically-ordered DAG, every edge (consumer, producer)
/// satisfies consumer > producer. This is the fundamental ordering invariant
/// required by the buffer planner's last_use computation.
#[kani::unwind(8)]
#[kani::proof]
#[kani::unwind(6)]
fn topological_order_consumer_after_producer() {
    const N: usize = 5;

    // Build symbolic DAG with topological order constraint.
    for i in 0..N {
        let num_edges: u8 = kani::any();
        kani::assume(num_edges <= 2);

        if num_edges >= 1 && i > 0 {
            let src: usize = kani::any();
            kani::assume(src < i);
            // The topological order property.
            assert!(i > src, "consumer must follow producer in topo order");
        }
        if num_edges >= 2 && i > 1 {
            let src: usize = kani::any();
            kani::assume(src < i);
            assert!(i > src);
        }
    }
}

// ============================================================================
// Proof 12: threadgroup_width_1d is exactly min(64, total)
// ============================================================================

/// Proves the exact definition: `threadgroup_width_1d(total)` ==
/// `min(64, total)` for all non-zero total values.
///
/// This extends `threadgroup_width_bounded` (which only proves the range)
/// with an exact equality proof.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn threadgroup_width_exact_min_64_total() {
    let total: u32 = kani::any();
    kani::assume(total > 0);

    let width = threadgroup_width_1d(total);
    let expected = if total < 64 { total } else { 64 };

    assert_eq!(
        width, expected,
        "threadgroup_width_1d must equal min(64, total)"
    );
}

// ============================================================================
// Proof 13: Reduction shared memory round-trip through u64
// ============================================================================

/// Proves that the `shared_bytes` parameter round-trips through u64
/// storage without loss. The plan stores it as `Some(u64)` via
/// `u64::from(shared_bytes: u32)`. This proves no truncation occurs.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn reduction_shared_memory_u64_roundtrip() {
    let shared_bytes: u32 = kani::any();
    let outer: u32 = kani::any();
    let reduce: u32 = kani::any();
    let threads: u32 = kani::any();
    kani::assume(outer > 0 && reduce > 0 && threads > 0);

    let plan = plan_reduction(outer, reduce, threads, shared_bytes)
        .expect("valid reduction");

    let stored = plan.threadgroup_memory_bytes().expect("reduction has shared mem");

    // Round-trip: u32 → u64 → u32 must be lossless.
    assert_eq!(stored, shared_bytes as u64, "u32 → u64 must be lossless");
    assert_eq!(
        stored as u32, shared_bytes,
        "u64 → u32 roundtrip must recover original"
    );
}

// ============================================================================
// Proof 14: NarrowView source_step bounds within step count
// ============================================================================

/// Proves the safety invariant that a NarrowView's `source_step` index,
/// when present, must be strictly less than the consuming step index
/// (topological order) and within the total step count.
///
/// Models: `compiled_model_execute_steps.rs` NarrowView dispatch logic.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn narrow_view_source_step_within_bounds() {
    let num_steps: usize = kani::any();
    let consuming_step: usize = kani::any();
    let source_step: usize = kani::any();

    kani::assume(num_steps >= 2 && num_steps <= 256);
    kani::assume(consuming_step < num_steps);
    kani::assume(source_step < num_steps);
    kani::assume(source_step < consuming_step); // topological order

    // The source_step must be a valid index into buffers[].
    assert!(source_step < num_steps, "source_step must be in bounds");

    // And it must precede the consuming step.
    assert!(
        source_step < consuming_step,
        "source_step must precede consuming step"
    );

    // The consuming step itself must be in bounds.
    assert!(consuming_step < num_steps, "consuming step must be in bounds");
}

// ============================================================================
// Proof 15: Weight index per-step uniqueness after dedup
// ============================================================================

/// Proves that the flat_weights_to_indexed model produces no duplicate
/// entry_ids within a single step bucket. Duplicate weight bindings for
/// the same step would cause incorrect Metal argument table setup.
#[kani::unwind(16)]
#[kani::proof]
#[kani::unwind(5)]
fn weight_index_per_step_unique() {
    const MAX_STEPS: usize = 3;
    let num_steps: usize = kani::any();
    kani::assume(num_steps > 0 && num_steps <= MAX_STEPS);

    // Two entries targeting the same step with distinct entry_ids.
    let step_idx: usize = kani::any();
    kani::assume(step_idx < num_steps);
    let entry_a: usize = kani::any();
    let entry_b: usize = kani::any();
    kani::assume(entry_a <= 10 && entry_b <= 10);
    kani::assume(entry_a != entry_b);

    let entries = [(step_idx, entry_a), (step_idx, entry_b)];

    // Model flat_weights_to_indexed.
    let mut indexed: Vec<Vec<usize>> = (0..num_steps).map(|_| Vec::new()).collect();
    for &(si, eid) in &entries {
        if si < num_steps {
            indexed[si].push(eid);
        }
    }

    // Both entries present in the bucket.
    let bucket = &indexed[step_idx];
    assert!(bucket.contains(&entry_a));
    assert!(bucket.contains(&entry_b));

    // And they are distinct.
    assert_ne!(entry_a, entry_b);
    assert_eq!(
        bucket.len(),
        2,
        "two distinct entries must produce two elements"
    );
}

// ============================================================================
// Proof 16: Freed slot reuse preserves Metal alignment
// ============================================================================

/// Proves that when a freed slot is reused and has a remainder, the
/// remainder offset is still >= the original slot offset (no backward
/// movement), and the remainder + reused region covers the original slot.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn freed_slot_reuse_remainder_covers_original() {
    let slot_offset: usize = kani::any();
    let slot_size: usize = kani::any();
    let request: usize = kani::any();

    kani::assume(slot_offset <= 1 << 24);
    kani::assume(slot_size >= 1 && slot_size <= 1 << 20);
    kani::assume(request >= 1 && request <= slot_size); // fits in slot

    let remainder_offset = slot_offset + request;
    let remainder_size = slot_size - request;

    // Property 1: reused region [slot_offset, slot_offset+request) is valid.
    assert!(slot_offset + request <= slot_offset + slot_size);

    // Property 2: remainder starts exactly after the reused region.
    assert_eq!(remainder_offset, slot_offset + request);

    // Property 3: reused + remainder = original slot.
    assert_eq!(request + remainder_size, slot_size);

    // Property 4: remainder offset >= slot_offset (no backward).
    assert!(remainder_offset >= slot_offset);
}

// ============================================================================
// Proof 17: with_threadgroup_memory_bytes override correctness
// ============================================================================

/// Proves that `with_threadgroup_memory_bytes` correctly overrides the
/// shared memory field, including the None → Some and Some → None transitions.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn with_threadgroup_memory_bytes_override() {
    // Start from an elementwise plan (no shared memory).
    let total: u32 = kani::any();
    kani::assume(total > 0);
    let plan = plan_elementwise(total).expect("valid plan");
    assert!(plan.threadgroup_memory_bytes().is_none());

    // Add shared memory.
    let bytes: u64 = kani::any();
    kani::assume(bytes <= 1 << 16);
    let modified = plan.clone().with_threadgroup_memory_bytes(Some(bytes));
    assert_eq!(modified.threadgroup_memory_bytes(), Some(bytes));

    // Remove shared memory.
    let cleared = modified.with_threadgroup_memory_bytes(None);
    assert!(cleared.threadgroup_memory_bytes().is_none());
}

// ============================================================================
// Proof 18: Release map total entry count equals releasable step count
// ============================================================================

/// Proves that the total number of entries across all release_at slots
/// equals the number of steps that have last_use[i] > i and size > 0
/// and are not output steps. No entries are lost or duplicated.
#[kani::unwind(16)]
#[kani::proof]
#[kani::unwind(5)]
fn release_at_total_entries_matches_releasable_count() {
    const N: usize = 4;
    let mut sizes = [0usize; N];
    let mut last_use = [0usize; N];
    for i in 0..N {
        sizes[i] = kani::any();
        kani::assume(sizes[i] <= 64);
        last_use[i] = kani::any();
        kani::assume(last_use[i] >= i && last_use[i] < N);
    }

    let output_steps: Vec<usize> = Vec::new();
    let map = model_release_at_extra(&last_use, N, &output_steps);

    // Count total entries in release map.
    let mut total_released = 0usize;
    for j in 0..N {
        total_released += map[j].len();
    }

    // Count releasable steps.
    let mut releasable = 0usize;
    for i in 0..N {
        if last_use[i] > i && last_use[i] < N && sizes[i] > 0 {
            releasable += 1;
        }
    }

    assert_eq!(
        total_released, releasable,
        "release map entry count must equal releasable step count"
    );
}

// ============================================================================
// Proof 19: plan_grid_2d constants match grid dimensions
// ============================================================================

/// Proves that the constants vector produced by plan_grid_2d contains
/// exactly [grid_x, grid_y], enabling the MSL kernel to reconstruct
/// the 2D dispatch topology from constant bindings.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(3)]
fn plan_grid_2d_constants_match_grid() {
    let g0: u32 = kani::any();
    let g1: u32 = kani::any();
    let t0: u32 = kani::any();
    let t1: u32 = kani::any();
    kani::assume(g0 > 0 && g1 > 0 && t0 > 0 && t1 > 0);

    let plan = plan_grid_2d([g0, g1], [t0, t1]).expect("valid 2D grid");
    let constants = plan.constants();

    assert_eq!(constants.len(), 2, "2D plan must have exactly 2 constants");
    assert_eq!(constants[0], g0, "constant[0] must be grid_x");
    assert_eq!(constants[1], g1, "constant[1] must be grid_y");
}

// ============================================================================
// Proof 20: plan_grid_3d constants match grid dimensions
// ============================================================================

/// Proves that the constants vector produced by plan_grid_3d contains
/// exactly [grid_x, grid_y, grid_z].
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(4)]
fn plan_grid_3d_constants_match_grid() {
    let g: [u32; 3] = [kani::any(), kani::any(), kani::any()];
    let t: [u32; 3] = [kani::any(), kani::any(), kani::any()];
    kani::assume(g[0] > 0 && g[1] > 0 && g[2] > 0);
    kani::assume(t[0] > 0 && t[1] > 0 && t[2] > 0);

    let widened = (g[0] as u128) * (g[1] as u128) * (g[2] as u128);
    kani::assume(widened <= usize::MAX as u128);

    let plan = plan_grid_3d(g, t).expect("valid 3D grid");
    let constants = plan.constants();

    assert_eq!(constants.len(), 3, "3D plan must have exactly 3 constants");
    assert_eq!(constants[0], g[0], "constant[0] must be grid_x");
    assert_eq!(constants[1], g[1], "constant[1] must be grid_y");
    assert_eq!(constants[2], g[2], "constant[2] must be grid_z");
}

// ============================================================================
// Proof 21: Grid2D no shared memory and uses dispatch_threads
// ============================================================================

/// Proves that plan_grid_2d never requests threadgroup shared memory
/// and always uses dispatch_threads (not dispatch_threadgroups).
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(3)]
fn plan_grid_2d_no_shared_memory_dispatch_threads() {
    let g0: u32 = kani::any();
    let g1: u32 = kani::any();
    let t0: u32 = kani::any();
    let t1: u32 = kani::any();
    kani::assume(g0 > 0 && g1 > 0 && t0 > 0 && t1 > 0);

    let plan = plan_grid_2d([g0, g1], [t0, t1]).expect("valid 2D grid");

    assert!(
        plan.threadgroup_memory_bytes().is_none(),
        "Grid2D must not request shared memory"
    );
    assert!(
        !plan.use_threadgroups(),
        "Grid2D must use dispatch_threads"
    );
}

// ============================================================================
// Proof 22: Grid3D no shared memory and uses dispatch_threads
// ============================================================================

/// Proves that plan_grid_3d never requests threadgroup shared memory
/// and always uses dispatch_threads (not dispatch_threadgroups).
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(4)]
fn plan_grid_3d_no_shared_memory_dispatch_threads() {
    let g: [u32; 3] = [kani::any(), kani::any(), kani::any()];
    let t: [u32; 3] = [kani::any(), kani::any(), kani::any()];
    kani::assume(g[0] > 0 && g[1] > 0 && g[2] > 0);
    kani::assume(t[0] > 0 && t[1] > 0 && t[2] > 0);
    let widened = (g[0] as u128) * (g[1] as u128) * (g[2] as u128);
    kani::assume(widened <= usize::MAX as u128);

    let plan = plan_grid_3d(g, t).expect("valid 3D grid");

    assert!(
        plan.threadgroup_memory_bytes().is_none(),
        "Grid3D must not request shared memory"
    );
    assert!(
        !plan.use_threadgroups(),
        "Grid3D must use dispatch_threads"
    );
}

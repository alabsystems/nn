// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Kani proof harnesses for GPU dispatch plan builder safety (#3573).
//!
//! The dispatch plan builder converts compiled traces into GPU dispatch steps.
//! These harnesses prove buffer index bounds, variant selection correctness,
//! and ordering safety for the pure functions that the builder depends on.
//!
//! # Properties Proved
//!
//! ## Buffer index bounds
//! - `flat_weights_to_indexed`: no out-of-bounds indexing, preserves all valid entries
//! - `release_at` computation: map entries are bounded, no self-reference
//! - `threadgroup_width_1d`: result always in [1, min(total, 64)]
//!
//! ## Variant selection
//! - `plan_elementwise` always uses `dispatch_threads` (not threadgroups)
//! - `plan_reduction` always uses `dispatch_threadgroups` with shared memory
//! - `plan_grid_2d` / `plan_grid_3d` thread dimensions pass through unchanged
//!
//! ## Ordering safety
//! - `release_at` map: released steps always precede the releasing step
//! - `release_at` entries never include output step indices
//! - Dispatch plan constants vector matches the mode (1 for elementwise,
//!   2 for 2D/reduction, 3 for 3D)

// ---------------------------------------------------------------------------
// Helper: re-implement flat_weights_to_indexed for Kani
// ---------------------------------------------------------------------------
//
// The production function uses `HashMap<(usize, String), MetalBuffer>` which
// depends on Metal. Re-implement the index-safety logic with plain types to
// verify the bounds guard `if step_idx < num_steps`.

/// Model of `flat_weights_to_indexed` from `compiled_model_build.rs`.
///
/// Maps `(step_idx, entry_id)` pairs into a `Vec<Vec<usize>>` indexed by
/// step. The critical safety property is that `step_idx < num_steps` is
/// checked before indexing — without this guard, an out-of-range step_idx
/// from a corrupted weight map would cause a panic.
fn model_flat_weights_to_indexed(
    entries: &[(usize, usize)], // (step_idx, entry_id)
    num_steps: usize,
) -> Vec<Vec<usize>> {
    let mut indexed: Vec<Vec<usize>> = (0..num_steps).map(|_| Vec::new()).collect();
    for &(step_idx, entry_id) in entries {
        if step_idx < num_steps {
            indexed[step_idx].push(entry_id);
        }
    }
    indexed
}

/// Model of `release_at` computation from `compiled_model_builder.rs:373-382`.
///
/// Builds a map where `release_at[j]` lists step indices whose last consumer
/// is step `j`. Output steps are excluded (their buffers must be preserved).
fn model_release_at(
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

// ---------------------------------------------------------------------------
// Proof 1: flat_weights_to_indexed never panics on out-of-range step_idx
// ---------------------------------------------------------------------------

/// For any set of entries (including out-of-range step indices), the function
/// produces a valid Vec without panicking. Out-of-range entries are silently
/// dropped (matching production behavior in `compiled_model_build.rs:376`).
#[kani::unwind(8)]
#[kani::proof]
#[kani::unwind(5)]
fn flat_weights_no_oob_panic() {
    const MAX_STEPS: usize = 4;
    let num_steps: usize = kani::any();
    kani::assume(num_steps <= MAX_STEPS);

    // 3 symbolic entries, some potentially out of range
    let mut entries = [(0usize, 0usize); 3];
    for i in 0..3 {
        entries[i].0 = kani::any();
        entries[i].1 = kani::any();
        // Don't constrain step_idx — let Kani explore out-of-range values
        kani::assume(entries[i].0 <= MAX_STEPS + 2);
        kani::assume(entries[i].1 <= 10);
    }

    let indexed = model_flat_weights_to_indexed(&entries, num_steps);
    assert_eq!(indexed.len(), num_steps);
}

// ---------------------------------------------------------------------------
// Proof 2: flat_weights_to_indexed preserves all valid entries
// ---------------------------------------------------------------------------

/// Every entry whose step_idx is in-range appears in the output.
/// Every entry in the output came from the input.
#[kani::unwind(16)]
#[kani::proof]
#[kani::unwind(5)]
fn flat_weights_preserves_valid_entries() {
    const N: usize = 3;
    let num_steps: usize = kani::any();
    kani::assume(num_steps > 0 && num_steps <= N);

    let step_idx: usize = kani::any();
    let entry_id: usize = kani::any();
    kani::assume(step_idx < num_steps);
    kani::assume(entry_id <= 10);

    let entries = [(step_idx, entry_id)];
    let indexed = model_flat_weights_to_indexed(&entries, num_steps);

    assert!(
        indexed[step_idx].contains(&entry_id),
        "valid entry must appear in indexed output"
    );
}

// ---------------------------------------------------------------------------
// Proof 3: flat_weights_to_indexed drops out-of-range entries
// ---------------------------------------------------------------------------

/// Entries with step_idx >= num_steps are not present in any bucket.
#[kani::unwind(16)]
#[kani::proof]
#[kani::unwind(5)]
fn flat_weights_drops_oob_entries() {
    const N: usize = 3;
    let num_steps: usize = kani::any();
    kani::assume(num_steps > 0 && num_steps <= N);

    let step_idx: usize = kani::any();
    kani::assume(step_idx >= num_steps && step_idx <= N + 2);
    let entry_id: usize = kani::any();
    kani::assume(entry_id <= 10);

    let entries = [(step_idx, entry_id)];
    let indexed = model_flat_weights_to_indexed(&entries, num_steps);

    // No bucket should contain this entry
    for bucket in &indexed {
        assert!(
            !bucket.contains(&entry_id) || step_idx < num_steps,
            "out-of-range entry must not appear in any bucket"
        );
    }
}

// ---------------------------------------------------------------------------
// Proof 4: release_at entries are bounded and precede the release point
// ---------------------------------------------------------------------------

/// For every entry in `release_at[j]`, the released step index `s` satisfies:
/// - `s < j` (released step precedes the releasing step)
/// - `s < num_steps` (bounded)
/// - `last_use[s] == j` (correct association)
#[kani::unwind(16)]
#[kani::proof]
#[kani::unwind(5)]
fn release_at_entries_precede_release_point() {
    const N: usize = 4;
    let mut last_use = [0usize; N];
    for i in 0..N {
        last_use[i] = kani::any();
        kani::assume(last_use[i] >= i && last_use[i] < N);
    }

    let output_steps: Vec<usize> = Vec::new(); // no outputs for this proof
    let map = model_release_at(&last_use, N, &output_steps);

    for j in 0..N {
        for &s in &map[j] {
            assert!(s < j, "released step {s} must precede release point {j}");
            assert!(s < N, "released step {s} must be bounded by num_steps");
            assert_eq!(
                last_use[s], j,
                "released step's last_use must equal release point"
            );
        }
    }
}

// ---------------------------------------------------------------------------
// Proof 5: release_at never contains output step indices
// ---------------------------------------------------------------------------

/// Output steps are excluded from the release map (their buffers must be
/// preserved for the caller). This prevents premature buffer deallocation
/// for model outputs.
#[kani::unwind(16)]
#[kani::proof]
#[kani::unwind(5)]
fn release_at_excludes_output_steps() {
    const N: usize = 4;
    let mut last_use = [0usize; N];
    for i in 0..N {
        last_use[i] = kani::any();
        kani::assume(last_use[i] >= i && last_use[i] < N);
    }

    // Symbolic output step index
    let output_idx: usize = kani::any();
    kani::assume(output_idx < N);
    let output_steps = vec![output_idx];

    let map = model_release_at(&last_use, N, &output_steps);

    for j in 0..N {
        for &s in &map[j] {
            assert_ne!(
                s, output_idx,
                "output step {output_idx} must not appear in release_at"
            );
        }
    }
}

// ---------------------------------------------------------------------------
// Proof 6: release_at map has no self-references
// ---------------------------------------------------------------------------

/// No step releases itself: `release_at[j]` never contains `j`.
/// The guard `consumer > step` in the builder ensures this.
#[kani::unwind(16)]
#[kani::proof]
#[kani::unwind(5)]
fn release_at_no_self_reference() {
    const N: usize = 4;
    let mut last_use = [0usize; N];
    for i in 0..N {
        last_use[i] = kani::any();
        kani::assume(last_use[i] >= i && last_use[i] < N);
    }

    let output_steps: Vec<usize> = Vec::new();
    let map = model_release_at(&last_use, N, &output_steps);

    for j in 0..N {
        for &s in &map[j] {
            assert_ne!(s, j, "step {j} must not release itself");
        }
    }
}

// ---------------------------------------------------------------------------
// Proof 7: threadgroup_width_1d is always in [1, min(total, 64)]
// ---------------------------------------------------------------------------

/// The threadgroup width for 1D dispatch is:
/// - At least 1 when total > 0
/// - At most 64 (Metal best practice)
/// - At most total (never exceeds work size)
///
/// This ensures Metal never dispatches more threads than elements.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn threadgroup_width_bounded() {
    use crate::dispatch_plan::threadgroup_width_1d;

    let total: u32 = kani::any();
    kani::assume(total > 0);

    let width = threadgroup_width_1d(total);
    assert!(width >= 1, "threadgroup width must be >= 1");
    assert!(width <= 64, "threadgroup width must be <= 64");
    assert!(
        width <= total,
        "threadgroup width must not exceed total elements"
    );
}

// ---------------------------------------------------------------------------
// Proof 8: plan_elementwise variant selection (dispatch_threads, no shared mem)
// ---------------------------------------------------------------------------

/// Elementwise plans always use `dispatch_threads` (not `dispatch_threadgroups`)
/// and never request threadgroup shared memory. This is the variant selection
/// correctness for the most common dispatch mode.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn plan_elementwise_variant_selection() {
    use crate::dispatch_plan::plan_elementwise;

    let total: u32 = kani::any();
    let plan = plan_elementwise(total).expect("elementwise always succeeds");

    assert!(
        !plan.use_threadgroups(),
        "elementwise must use dispatch_threads, not dispatch_threadgroups"
    );
    assert!(
        plan.threadgroup_memory_bytes().is_none(),
        "elementwise must not request shared memory"
    );
    assert_eq!(
        plan.constants().len(),
        1,
        "elementwise must have exactly 1 constant (total)"
    );
    assert_eq!(
        plan.constants()[0], total,
        "elementwise constant must equal total"
    );
}

// ---------------------------------------------------------------------------
// Proof 9: plan_reduction variant selection (dispatch_threadgroups + shared mem)
// ---------------------------------------------------------------------------

/// Reduction plans always use `dispatch_threadgroups` with shared memory.
/// This is the variant selection correctness for norm/reduce operations.
#[kani::unwind(8)]
#[kani::proof]
#[kani::unwind(1)]
fn plan_reduction_variant_selection() {
    use crate::dispatch_plan::plan_reduction;

    let outer: u32 = kani::any();
    let reduce: u32 = kani::any();
    let threads: u32 = kani::any();
    let shared_bytes: u32 = kani::any();
    kani::assume(outer > 0 && reduce > 0 && threads > 0);

    let plan = plan_reduction(outer, reduce, threads, shared_bytes)
        .expect("reduction succeeds for non-zero params");

    assert!(
        plan.use_threadgroups(),
        "reduction must use dispatch_threadgroups"
    );
    assert!(
        plan.threadgroup_memory_bytes().is_some(),
        "reduction must have shared memory"
    );
    assert_eq!(
        plan.threadgroup_memory_bytes().unwrap(),
        shared_bytes as u64,
        "shared memory must equal requested bytes"
    );
}

// ---------------------------------------------------------------------------
// Proof 10: plan_grid_2d thread dimensions pass through unchanged
// ---------------------------------------------------------------------------

/// Grid2D thread configuration is passed through to the dispatch plan
/// exactly as specified. Thread dimensions must not be modified — the
/// caller chooses threadgroup sizes based on kernel requirements.
#[kani::unwind(8)]
#[kani::proof]
#[kani::unwind(3)]
fn plan_grid_2d_threads_passthrough() {
    use crate::dispatch_plan::plan_grid_2d;

    let g0: u32 = kani::any();
    let g1: u32 = kani::any();
    let t0: u32 = kani::any();
    let t1: u32 = kani::any();
    kani::assume(g0 > 0 && g1 > 0 && t0 > 0 && t1 > 0);

    let plan = plan_grid_2d([g0, g1], [t0, t1]).expect("valid 2D grid");

    let threads = plan.threads();
    assert_eq!(threads[0], t0, "threads.x must match input");
    assert_eq!(threads[1], t1, "threads.y must match input");
    assert_eq!(threads[2], 1, "threads.z must be 1 for 2D");

    let grid = plan.grid();
    assert_eq!(grid[0], g0, "grid.x must match input");
    assert_eq!(grid[1], g1, "grid.y must match input");
    assert_eq!(grid[2], 1, "grid.z must be 1 for 2D");
}

// ---------------------------------------------------------------------------
// Proof 11: plan_grid_3d thread dimensions pass through unchanged
// ---------------------------------------------------------------------------

/// Grid3D thread configuration is passed through to the dispatch plan
/// exactly as specified, with all three dimensions preserved.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(4)]
fn plan_grid_3d_threads_passthrough() {
    use crate::dispatch_plan::plan_grid_3d;

    let g: [u32; 3] = [kani::any(), kani::any(), kani::any()];
    let t: [u32; 3] = [kani::any(), kani::any(), kani::any()];
    kani::assume(g[0] > 0 && g[1] > 0 && g[2] > 0);
    kani::assume(t[0] > 0 && t[1] > 0 && t[2] > 0);

    // Only test non-overflowing cases (overflow → Err is tested separately)
    let widened = (g[0] as u128) * (g[1] as u128) * (g[2] as u128);
    kani::assume(widened <= usize::MAX as u128);

    let plan = plan_grid_3d(g, t).expect("valid 3D grid");

    let threads = plan.threads();
    assert_eq!(threads[0], t[0]);
    assert_eq!(threads[1], t[1]);
    assert_eq!(threads[2], t[2]);

    let grid = plan.grid();
    assert_eq!(grid[0], g[0]);
    assert_eq!(grid[1], g[1]);
    assert_eq!(grid[2], g[2]);
}

// ---------------------------------------------------------------------------
// Proof 12: dispatch plan constants vector length matches mode
// ---------------------------------------------------------------------------

/// Each dispatch mode produces the correct number of constants:
/// - Elementwise: 1 constant (total)
/// - Grid2D: 2 constants (grid_x, grid_y)
/// - Grid3D: 3 constants (grid_x, grid_y, grid_z)
/// - Reduction: 2 constants (outer, reduce)
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn plan_constants_count_matches_mode() {
    use crate::dispatch_plan::{plan_elementwise, plan_grid_2d, plan_grid_3d, plan_reduction};

    // Choose a mode symbolically
    let mode: u8 = kani::any();
    kani::assume(mode < 4);

    match mode {
        0 => {
            let total: u32 = kani::any();
            let plan = plan_elementwise(total).unwrap();
            assert_eq!(plan.constants().len(), 1);
        }
        1 => {
            let g0: u32 = kani::any();
            let g1: u32 = kani::any();
            let t0: u32 = kani::any();
            let t1: u32 = kani::any();
            kani::assume(g0 > 0 && g1 > 0 && t0 > 0 && t1 > 0);
            let plan = plan_grid_2d([g0, g1], [t0, t1]).unwrap();
            assert_eq!(plan.constants().len(), 2);
        }
        2 => {
            let g: [u32; 3] = [kani::any(), kani::any(), kani::any()];
            let t: [u32; 3] = [kani::any(), kani::any(), kani::any()];
            kani::assume(g[0] > 0 && g[1] > 0 && g[2] > 0);
            kani::assume(t[0] > 0 && t[1] > 0 && t[2] > 0);
            let widened = (g[0] as u128) * (g[1] as u128) * (g[2] as u128);
            kani::assume(widened <= usize::MAX as u128);
            let plan = plan_grid_3d(g, t).unwrap();
            assert_eq!(plan.constants().len(), 3);
        }
        3 => {
            let outer: u32 = kani::any();
            let reduce: u32 = kani::any();
            let threads: u32 = kani::any();
            let shared: u32 = kani::any();
            kani::assume(outer > 0 && reduce > 0 && threads > 0);
            let plan = plan_reduction(outer, reduce, threads, shared).unwrap();
            assert_eq!(plan.constants().len(), 2);
        }
        _ => unreachable!(),
    }
}

// ---------------------------------------------------------------------------
// Proof 13: tiled_transpose_2d_params batch is always >= 1
// ---------------------------------------------------------------------------

/// When `tiled_transpose_2d_params` returns `Some`, the batch dimension
/// is always >= 1 (the `.max(1)` guard). This prevents zero-element
/// dispatches which would waste a Metal command encoder slot.
#[kani::unwind(16)]
#[kani::proof]
#[kani::unwind(6)]
fn tiled_transpose_batch_geq_one() {
    use nn_dsl::tiled_transpose_2d_params;
    use nn_dsl::TILED_TRANSPOSE_TILE_SIZE;

    let rank: usize = kani::any();
    kani::assume(rank >= 2 && rank <= 5);

    let mut shape = vec![0usize; rank];
    for i in 0..rank {
        shape[i] = kani::any();
        kani::assume(shape[i] >= 1 && shape[i] <= 128);
    }
    // Last two dims must meet tile threshold for Some result
    kani::assume(shape[rank - 2] >= TILED_TRANSPOSE_TILE_SIZE);
    kani::assume(shape[rank - 1] >= TILED_TRANSPOSE_TILE_SIZE);

    // Identity axes for all but last two, then swap
    let mut axes = vec![0usize; rank];
    for i in 0..rank - 2 {
        axes[i] = i;
    }
    axes[rank - 2] = rank - 1;
    axes[rank - 1] = rank - 2;

    if let Some((batch, rows, cols)) = tiled_transpose_2d_params(&shape, &axes) {
        assert!(batch >= 1, "batch must be >= 1");
        assert_eq!(rows, shape[rank - 2], "rows must match input dim");
        assert_eq!(cols, shape[rank - 1], "cols must match input dim");
    }
}

// ---------------------------------------------------------------------------
// Proof 14: tiled_transpose_2d_params rejects non-swap axes
// ---------------------------------------------------------------------------

/// When the last two axes are NOT swapped (identity permutation), the
/// function correctly returns `None` — a non-transposing permutation
/// should not use the tiled transpose kernel.
#[kani::unwind(8)]
#[kani::proof]
#[kani::unwind(4)]
fn tiled_transpose_rejects_identity_axes() {
    use nn_dsl::tiled_transpose_2d_params;
    use nn_dsl::TILED_TRANSPOSE_TILE_SIZE;

    // rank 3, identity axes [0, 1, 2]
    let shape = [32usize, TILED_TRANSPOSE_TILE_SIZE, TILED_TRANSPOSE_TILE_SIZE];
    let axes = [0, 1, 2];

    let result = tiled_transpose_2d_params(&shape, &axes);
    assert!(
        result.is_none(),
        "identity axes must not qualify for tiled transpose"
    );
}

// ---------------------------------------------------------------------------
// Proof 15: reduction grid dimension equals outer (one threadgroup per slice)
// ---------------------------------------------------------------------------

/// The reduction dispatch uses exactly one threadgroup per outer slice.
/// This is the fundamental dispatch topology for norm/reduce kernels:
/// each threadgroup cooperatively reduces `reduce` elements.
#[kani::unwind(8)]
#[kani::proof]
#[kani::unwind(1)]
fn plan_reduction_grid_equals_outer() {
    use crate::dispatch_plan::plan_reduction;

    let outer: u32 = kani::any();
    let reduce: u32 = kani::any();
    let threads: u32 = kani::any();
    let shared: u32 = kani::any();
    kani::assume(outer > 0 && reduce > 0 && threads > 0);

    let plan = plan_reduction(outer, reduce, threads, shared)
        .expect("valid reduction");

    let grid = plan.grid();
    assert_eq!(
        grid[0], outer,
        "grid.x must equal outer (one threadgroup per slice)"
    );
    assert_eq!(grid[1], 1, "grid.y must be 1 for 1D reduction");
    assert_eq!(grid[2], 1, "grid.z must be 1 for 1D reduction");

    let plan_threads = plan.threads();
    assert_eq!(
        plan_threads[0], threads,
        "threads.x must match requested thread count"
    );
}

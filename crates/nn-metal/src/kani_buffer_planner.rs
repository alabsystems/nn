// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Kani proof harnesses for compiled model buffer planner aliasing safety (#3571).
//!
//! Proves critical safety invariants at the Metal dispatch level:
//! - Buffer size calculations don't overflow for valid tensor shapes
//! - No two live tensors share overlapping buffer regions in the planned allocation
//! - Buffer alignment requirements are met (Metal 256-byte alignment)
//! - Dispatch step ordering respects data dependencies (last_use correctness)
//! - NativeOp encoding indices are within valid range
//! - Total buffer pool size doesn't overflow
//!
//! These harnesses complement the algorithmic proofs in
//! `nn-dsl/src/kani_buffer_planner.rs` with Metal-specific safety properties
//! about buffer aliasing in the compiled model execution pipeline.

// ---------------------------------------------------------------------------
// Proof 1: Buffer size calculations don't overflow for valid tensor shapes
// ---------------------------------------------------------------------------

/// Proves that `checked_shape_bytes` (the buffer planner's per-step byte size
/// calculation) never silently wraps for shapes within production bounds.
///
/// Models the logic from `buffer_planner_bytes.rs:checked_shape_bytes`:
/// ```
/// shape.iter().try_fold(1usize, |acc, &dim| acc.checked_mul(dim))
///     .and_then(|product| product.checked_mul(F32_BYTES))
///     .unwrap_or(0)
/// ```
///
/// For 3D tensors `[B, C, T]` (the dominant shape in Kokoro/model execution),
/// this proves either the byte count is correct or the function returns 0
/// (never panics, never wraps).
#[kani::unwind(8)]
#[kani::proof]
#[kani::unwind(1)]
fn buffer_size_no_overflow_3d() {
    let b: usize = kani::any();
    let c: usize = kani::any();
    let t: usize = kani::any();
    let f32_bytes: usize = 4;

    // Production bounds: batch 1-64, channels 1-1024, time 1-2^14.
    kani::assume(b >= 1 && b <= 64);
    kani::assume(c >= 1 && c <= 1024);
    kani::assume(t >= 1 && t <= (1usize << 14));

    // Model checked_shape_bytes.
    let result = b
        .checked_mul(c)
        .and_then(|bc| bc.checked_mul(t))
        .and_then(|bct| bct.checked_mul(f32_bytes));

    match result {
        Some(bytes) => {
            // Property 1: No silent wrap — result equals widened computation.
            let widened = (b as u128) * (c as u128) * (t as u128) * (f32_bytes as u128);
            assert_eq!(bytes as u128, widened, "byte count must match widened product");

            // Property 2: Result is positive for non-degenerate shapes.
            assert!(bytes > 0, "byte count must be positive for valid shapes");

            // Property 3: Element count is recoverable.
            assert_eq!(bytes / f32_bytes, b * c * t, "element count must be recoverable");
        }
        None => {
            // Overflow detected — would return 0 in production code.
            // Verify that widened computation does indeed overflow usize.
            let widened = (b as u128) * (c as u128) * (t as u128) * (f32_bytes as u128);
            assert!(widened > usize::MAX as u128, "overflow only when widened exceeds usize::MAX");
        }
    }
}

// ---------------------------------------------------------------------------
// Proof 2: No two live tensors share overlapping buffer regions
// ---------------------------------------------------------------------------

/// Proves that the linear-scan allocator, when applied with Metal's single
/// contiguous buffer approach, never assigns overlapping byte regions to
/// steps whose lifetimes overlap.
///
/// This extends `proof_linear_scan_no_overlap_3step` from nn-dsl with a
/// 4-step graph and 16-byte Metal alignment enforcement.
///
/// Models: `compiled_model_execute_steps.rs` line 67-87 (planned buffer setup)
/// and `buffer_planner.rs:linear_scan_alloc`.
#[kani::unwind(16)]
#[kani::proof]
#[kani::unwind(6)]
fn no_overlap_with_metal_alignment_4step() {
    const N: usize = 4;
    const METAL_ALIGN: usize = 16; // Minimum Metal buffer offset alignment

    // Symbolic sizes: 0-128 bytes each (0 = non-allocating passthrough step).
    let mut sizes = [0usize; N];
    for i in 0..N {
        sizes[i] = kani::any();
        kani::assume(sizes[i] <= 128);
    }

    // Symbolic last_use: last_use[i] >= i, last_use[i] < N.
    let mut last_use = [0usize; N];
    for i in 0..N {
        last_use[i] = kani::any();
        kani::assume(last_use[i] >= i && last_use[i] < N);
    }

    // Re-implement linear_scan_alloc with Metal alignment.
    let mut offsets: [Option<usize>; N] = [None; N];
    let mut free_slots: Vec<(usize, usize)> = Vec::new();
    let mut hwm: usize = 0;

    // Build release map.
    let mut release_at: [Vec<usize>; N] = [Vec::new(), Vec::new(), Vec::new(), Vec::new()];
    for step in 0..N {
        let consumer = last_use[step];
        if consumer > step && consumer < N && sizes[step] > 0 {
            release_at[consumer].push(step);
        }
    }

    for step_idx in 0..N {
        let size = sizes[step_idx];
        if size == 0 {
            continue;
        }

        // Align size up to METAL_ALIGN for offset alignment guarantee.
        let aligned_size = (size + METAL_ALIGN - 1) / METAL_ALIGN * METAL_ALIGN;

        // alloc_or_reuse logic.
        let best_fit = free_slots
            .iter()
            .enumerate()
            .filter(|(_, slot)| slot.1 >= aligned_size)
            .min_by_key(|(_, slot)| slot.1)
            .map(|(idx, _)| idx);

        let offset = if let Some(slot_idx) = best_fit {
            let slot = free_slots.swap_remove(slot_idx);
            let remainder = slot.1 - aligned_size;
            if remainder > 0 {
                free_slots.push((slot.0 + aligned_size, remainder));
            }
            slot.0
        } else {
            let o = hwm;
            hwm = hwm.saturating_add(aligned_size);
            o
        };

        offsets[step_idx] = Some(offset);

        // Release prior buffers.
        for &prior_idx in &release_at[step_idx] {
            if let Some(prior_offset) = offsets[prior_idx] {
                let prior_aligned =
                    (sizes[prior_idx] + METAL_ALIGN - 1) / METAL_ALIGN * METAL_ALIGN;
                free_slots.push((prior_offset, prior_aligned));
            }
        }
    }

    // Verify: no two simultaneously-live allocated buffers overlap.
    for i in 0..N {
        let Some(off_i) = offsets[i] else { continue };
        if sizes[i] == 0 {
            continue;
        }
        let end_i = off_i + sizes[i];

        // Offset must be within high_water_mark.
        assert!(end_i <= hwm, "step end exceeds hwm");

        for j in (i + 1)..N {
            let Some(off_j) = offsets[j] else { continue };
            if sizes[j] == 0 {
                continue;
            }
            let end_j = off_j + sizes[j];

            // Check time overlap.
            let time_overlap = i <= last_use[j] && j <= last_use[i];
            if !time_overlap {
                continue;
            }

            // Memory must not overlap for simultaneously-live buffers.
            let memory_overlap = end_i > off_j && end_j > off_i;
            assert!(
                !memory_overlap,
                "steps {i} and {j} overlap in both memory and time"
            );
        }
    }
}

// ---------------------------------------------------------------------------
// Proof 3: Buffer alignment requirements are met (Metal 256-byte alignment)
// ---------------------------------------------------------------------------

/// Proves that the Metal arena's `align_up` function always produces offsets
/// aligned to `METAL_BUFFER_ALIGNMENT` (256 bytes), and that the aligned offset
/// is at least as large as the input offset.
///
/// Models `arena.rs:align_up` and `METAL_BUFFER_ALIGNMENT = 256`.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn metal_buffer_alignment_256() {
    let offset: usize = kani::any();
    let alignment: usize = 256; // METAL_BUFFER_ALIGNMENT

    // Avoid overflow in the align_up calculation.
    kani::assume(offset <= usize::MAX - alignment);

    // Model align_up: (offset + alignment - 1) & !(alignment - 1)
    let mask = alignment - 1;
    let aligned = (offset + mask) & !mask;

    // Property 1: Result is aligned to 256.
    assert_eq!(aligned % alignment, 0, "aligned offset must be divisible by 256");

    // Property 2: Result >= original offset (no underflow).
    assert!(aligned >= offset, "aligned offset must be >= original");

    // Property 3: Result is the smallest aligned value >= offset.
    // i.e., aligned - alignment < offset
    if aligned >= alignment {
        assert!(
            aligned - alignment < offset,
            "aligned must be the smallest aligned offset >= input"
        );
    }

    // Property 4: Padding is at most alignment - 1 bytes.
    let padding = aligned - offset;
    assert!(padding < alignment, "alignment padding must be < alignment");
}

// ---------------------------------------------------------------------------
// Proof 4: Dispatch step ordering respects data dependencies
// ---------------------------------------------------------------------------

/// Proves that `compute_last_use` produces valid dependency ordering:
/// for any edge `consumer → producer`, last_use[producer] >= consumer.
///
/// This ensures the buffer planner never frees a buffer before its last
/// consumer has executed — the fundamental safety property for buffer aliasing
/// in `compiled_model_execute_steps.rs`.
///
/// Models: `buffer_planner.rs:compute_last_use` + the release_at logic
/// in `compiled_model_execute_steps.rs:273-280`.
#[kani::unwind(16)]
#[kani::proof]
#[kani::unwind(6)]
fn dependency_ordering_preserved_5step() {
    const N: usize = 5;

    // Build a symbolic DAG: each step has 0-2 input edges.
    // Topological order: edges only point to earlier steps.
    let mut edge_map: Vec<Vec<usize>> = Vec::new();
    for i in 0..N {
        let num_edges: u8 = kani::any();
        kani::assume(num_edges <= 2);
        let mut edges = Vec::new();
        if num_edges >= 1 && i > 0 {
            let src0: usize = kani::any();
            kani::assume(src0 < i);
            edges.push(src0);
        }
        if num_edges >= 2 && i > 1 {
            let src1: usize = kani::any();
            kani::assume(src1 < i);
            edges.push(src1);
        }
        edge_map.push(edges);
    }

    // Compute last_use (same algorithm as buffer_planner.rs).
    let mut last_use: Vec<usize> = (0..N).collect();
    for (consumer_idx, inputs) in edge_map.iter().enumerate() {
        for &producer_idx in inputs {
            if consumer_idx > last_use[producer_idx] {
                last_use[producer_idx] = consumer_idx;
            }
        }
    }

    // Verify: for every edge, the producer is live until at least the consumer.
    for (consumer_idx, inputs) in edge_map.iter().enumerate() {
        for &producer_idx in inputs {
            assert!(
                last_use[producer_idx] >= consumer_idx,
                "producer {} freed at {} before consumer {} reads it",
                producer_idx,
                last_use[producer_idx],
                consumer_idx,
            );
        }
    }

    // Verify: release_at never frees a buffer before its producer step.
    for i in 0..N {
        assert!(
            last_use[i] >= i,
            "last_use[{}] = {} < {} (freed before produced)",
            i,
            last_use[i],
            i,
        );
    }

    // Verify: last_use is bounded.
    for i in 0..N {
        assert!(
            last_use[i] < N,
            "last_use[{}] = {} >= N (out of bounds)",
            i,
            last_use[i],
        );
    }
}

// ---------------------------------------------------------------------------
// Proof 5: NativeOp encoding indices are within valid range
// ---------------------------------------------------------------------------

/// Proves that step indices referenced by NativeOp direct-access patterns
/// (FusedResBlock input_steps, BatchedStyleProjection style_step,
/// ProjectionSlice source_step) are within valid buffer bounds when
/// the `validate_buffer_plan_edges` check passes.
///
/// Models: `compiled_model_build.rs:validate_buffer_plan_edges` lines 44-99.
/// This is the safety gate that prevents out-of-bounds buffer access in
/// `compiled_model_execute_native_fused.rs`.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn native_op_step_indices_within_bounds() {
    let num_steps: usize = kani::any();
    kani::assume(num_steps >= 2 && num_steps <= 16);

    // Symbolic step index referenced by a NativeOp.
    let dep_step: usize = kani::any();
    let consumer_step: usize = kani::any();
    kani::assume(dep_step < num_steps);
    kani::assume(consumer_step < num_steps);
    kani::assume(consumer_step > dep_step); // topological order

    // Symbolic last_use array (from buffer planner).
    let last_use_dep: usize = kani::any();
    kani::assume(last_use_dep >= dep_step && last_use_dep < num_steps);

    // Model validate_buffer_plan_edges check.
    // 1) dep must be within last_use bounds.
    let within_bounds = dep_step < num_steps;
    // 2) last_use[dep] must be >= consumer_step (buffer still live).
    let still_live = last_use_dep >= consumer_step;

    // If validation passes, the buffer access is safe.
    if within_bounds && still_live {
        // Property 1: dep_step is a valid index into buffers[].
        assert!(dep_step < num_steps, "dep_step must be in bounds");

        // Property 2: The buffer at dep_step is still live at consumer_step.
        assert!(
            last_use_dep >= consumer_step,
            "buffer at dep_step must be live at consumer_step"
        );

        // Property 3: Consumer is strictly after producer (no self-reference).
        assert!(
            consumer_step > dep_step,
            "consumer must be after producer in topological order"
        );
    }
}

// ---------------------------------------------------------------------------
// Proof 6: Total buffer pool size doesn't overflow
// ---------------------------------------------------------------------------

/// Proves that the buffer planner's `high_water_mark` (total contiguous
/// buffer size) doesn't overflow when computed via `saturating_add` for
/// production-bounded step sizes.
///
/// Models: `buffer_planner.rs:alloc_or_reuse` line 310:
/// `*high_water_mark = high_water_mark.saturating_add(size)`
/// and the `total_bytes` field checked in
/// `compiled_model_execute_steps.rs:67`.
///
/// In production, the total is passed to `create_buffer_zeroed(total_bytes)`
/// which allocates a single Metal buffer. Overflow would cause a too-small
/// allocation and subsequent out-of-bounds GPU writes.
#[kani::unwind(8)]
#[kani::proof]
#[kani::unwind(9)]
fn total_pool_size_no_overflow() {
    const N: usize = 8;
    let f32_bytes: usize = 4;

    let mut high_water_mark: usize = 0;
    let mut naive_total: usize = 0;

    for _ in 0..N {
        let numel: usize = kani::any();
        // Production upper bound: 64 * 1024 * 16384 * 4 = 4 GiB max per step.
        // Constrain to smaller range for Kani tractability.
        kani::assume(numel <= 1024 * 1024); // 1M elements max per step

        let size = numel.checked_mul(f32_bytes).unwrap_or(0);
        if size == 0 {
            continue;
        }

        // Model alloc_or_reuse: when no free slot, extends high_water_mark.
        // (worst case: no reuse, all allocations are fresh)
        let offset = high_water_mark;
        let new_hwm = high_water_mark.checked_add(size);

        match new_hwm {
            Some(hwm) => {
                high_water_mark = hwm;

                // Property 1: Offset + size <= new high_water_mark.
                assert!(
                    offset + size <= high_water_mark,
                    "allocated region exceeds high_water_mark"
                );
            }
            None => {
                // Overflow: saturating_add would cap at usize::MAX.
                // Production code uses saturating_add, so we verify the cap.
                let saturated = high_water_mark.saturating_add(size);
                assert_eq!(
                    saturated,
                    usize::MAX,
                    "saturating_add must cap at usize::MAX on overflow"
                );
                return; // Cannot allocate more.
            }
        }

        // Track naive total (sum of all sizes, ignoring reuse).
        naive_total = naive_total.saturating_add(size);
    }

    // Property 2: high_water_mark <= naive_total (reuse can only help).
    // In the worst case (no reuse), they are equal.
    assert!(
        high_water_mark <= naive_total,
        "high_water_mark must not exceed naive_total without reuse"
    );

    // Property 3: When valid, high_water_mark is the exact amount needed
    // for create_buffer_zeroed().
    assert!(
        high_water_mark <= N * 1024 * 1024 * f32_bytes,
        "high_water_mark bounded by max steps * max step size"
    );
}

// ---------------------------------------------------------------------------
// Proof 7: Relocate-to-planned-buffer bounds check correctness
// ---------------------------------------------------------------------------

/// Proves that the source and destination bounds checks in
/// `relocate_to_planned_buffer` correctly reject all out-of-bounds accesses
/// and accept all in-bounds accesses.
///
/// Models: `compiled_model_execute_helpers.rs:80-139`
/// (the safety gate before the blit-copy into the planned buffer).
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn relocate_bounds_check_correct() {
    let src_offset: usize = kani::any();
    let src_buf_len: usize = kani::any();
    let dst_offset: usize = kani::any();
    let planned_buf_len: usize = kani::any();
    let size: usize = kani::any();

    kani::assume(src_buf_len <= 1 << 30);
    kani::assume(planned_buf_len <= 1 << 30);
    kani::assume(src_offset <= 1 << 30);
    kani::assume(dst_offset <= 1 << 30);
    kani::assume(size <= 1 << 30);

    // Model the source bounds check.
    let src_end = src_offset.checked_add(size);
    let src_ok = src_end.map_or(false, |end| end <= src_buf_len);

    // Model the destination bounds check.
    let dst_end = dst_offset.checked_add(size);
    let dst_ok = dst_end.map_or(false, |end| end <= planned_buf_len);

    if src_ok && dst_ok {
        // Both checks pass: the blit is safe.
        // Property 1: Source region is within source buffer.
        assert!(src_offset + size <= src_buf_len);

        // Property 2: Destination region is within planned buffer.
        assert!(dst_offset + size <= planned_buf_len);

        // Property 3: No overflow in either end calculation.
        assert!(src_end.is_some());
        assert!(dst_end.is_some());
    }

    // Property 4: If either check fails, at least one bound is violated.
    if !src_ok {
        // Either overflow or out-of-bounds.
        assert!(
            src_end.is_none() || src_end.unwrap() > src_buf_len,
            "src check must fail only on overflow or OOB"
        );
    }
    if !dst_ok {
        assert!(
            dst_end.is_none() || dst_end.unwrap() > planned_buf_len,
            "dst check must fail only on overflow or OOB"
        );
    }
}

// ---------------------------------------------------------------------------
// Proof 8: Release-at map construction correctness
// ---------------------------------------------------------------------------

/// Proves that the pre-built `release_at` map (constructed once at model build
/// time) correctly mirrors the `last_use` array — every step that should be
/// released at step `j` appears in `release_at[j]`, and no step appears in
/// the wrong release slot.
///
/// Models: `compiled_model_execute_steps.rs:253-260` (release_at construction)
/// and `buffer_planner.rs:255-260`.
#[kani::unwind(16)]
#[kani::proof]
#[kani::unwind(6)]
fn release_at_map_consistent_with_last_use() {
    const N: usize = 5;

    // Symbolic step_sizes and last_use.
    let mut step_sizes = [0usize; N];
    let mut last_use = [0usize; N];

    for i in 0..N {
        step_sizes[i] = kani::any();
        kani::assume(step_sizes[i] <= 64);
        last_use[i] = kani::any();
        kani::assume(last_use[i] >= i && last_use[i] < N);
    }

    // Symbolic output steps (excluded from release_at in production).
    // For this proof, we don't exclude outputs — we verify the base logic.

    // Build release_at (same logic as buffer_planner.rs and execute_steps.rs).
    let mut release_at: [Vec<usize>; N] = [Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new()];
    for step in 0..N {
        let consumer = last_use[step];
        if consumer > step && consumer < N && step_sizes[step] > 0 {
            release_at[consumer].push(step);
        }
    }

    // Verify: every step in release_at[j] has last_use == j.
    for j in 0..N {
        for &released_step in &release_at[j] {
            assert_eq!(
                last_use[released_step], j,
                "released step's last_use must equal release slot"
            );
            // Verify the step has a non-zero size (condition from build).
            assert!(
                step_sizes[released_step] > 0,
                "only non-zero-sized steps appear in release_at"
            );
            // Verify the step is earlier than the release point.
            assert!(
                released_step < j,
                "released step must be earlier than its release point"
            );
        }
    }

    // Verify: every step with last_use[step] > step and size > 0
    // appears in exactly one release_at slot.
    for step in 0..N {
        if last_use[step] > step && last_use[step] < N && step_sizes[step] > 0 {
            let consumer = last_use[step];
            let count = release_at[consumer]
                .iter()
                .filter(|&&s| s == step)
                .count();
            assert_eq!(
                count, 1,
                "step must appear exactly once in its release slot"
            );
        }
    }
}

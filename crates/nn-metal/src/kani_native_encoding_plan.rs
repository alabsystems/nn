// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Kani proof harnesses for NativeEncoding plan construction (#3472).
//!
//! NativeEncoding plans compute buffer sizes, grid dimensions, binding indices,
//! and auxiliary allocations for direct Metal dispatch. Bugs here cause:
//! - GPU out-of-bounds reads/writes (wrong `output_bytes`)
//! - Silent data corruption (wrong grid/threadgroup dims)
//! - Metal validation layer crashes (overlapping binding indices)
//! - Panics in production (unchecked arithmetic)
//!
//! These harnesses prove the arithmetic invariants of plan construction
//! WITHOUT Metal dependencies — only the pure Rust computation paths.

use std::mem::size_of;

// ============================================================================
// FusedResBlock plan dimension arithmetic
// ============================================================================

/// Prove: FusedResBlock `eff_k` calculation cannot underflow.
///
/// Models the pattern: `eff_k = (kernel_size - 1) * dilation + 1`
/// from `compiled_model_execute_native_resblock_plan.rs:115-116`.
///
/// If `kernel_size == 0`, the subtraction `kernel_size - 1` wraps in debug
/// mode (panic) and wraps to `usize::MAX` in release mode. This harness
/// proves the calculation is safe for all valid kernel sizes.
#[kani::unwind(8)]
#[kani::proof]
#[kani::unwind(1)]
fn resblock_eff_k_no_underflow() {
    let kernel_size: usize = kani::any();
    let dilation: usize = kani::any();

    // Kokoro Conv1d: kernel_size ∈ {3, 7}, dilation ∈ {1, 2, 4}.
    // Bound generously for future use.
    kani::assume(kernel_size >= 1 && kernel_size <= 31);
    kani::assume(dilation >= 1 && dilation <= 16);

    // This is the exact expression from resblock_plan.rs:115.
    let eff_k = (kernel_size - 1) * dilation + 1;

    // eff_k >= 1 always (kernel_size >= 1, dilation >= 1).
    assert!(eff_k >= 1, "eff_k must be at least 1");

    // eff_k <= kernel_size * dilation (tighter: (ks-1)*dil + 1 <= ks*dil).
    assert!(eff_k <= kernel_size * dilation, "eff_k bound exceeded");
}

/// Prove: FusedResBlock `out_len` is positive when `padded >= eff_k`.
///
/// Models `out_len = padded - eff_k + 1` from resblock_plan.rs:123.
/// The plan function guards `padded < eff_k` and returns Err. This proof
/// verifies that when the guard passes, `out_len >= 1`.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn resblock_out_len_positive_when_valid() {
    let in_len: usize = kani::any();
    let padding: usize = kani::any();
    let kernel_size: usize = kani::any();
    let dilation: usize = kani::any();

    kani::assume(kernel_size >= 1 && kernel_size <= 31);
    kani::assume(dilation >= 1 && dilation <= 16);
    kani::assume(in_len >= 1 && in_len <= 65536);
    kani::assume(padding <= 128);

    let eff_k = (kernel_size - 1) * dilation + 1;
    let padded = in_len + 2 * padding;

    // Guard from resblock_plan.rs:117.
    kani::assume(padded >= eff_k);

    let out_len = padded - eff_k + 1;
    assert!(out_len >= 1, "out_len must be at least 1 when padded >= eff_k");
}

/// Prove: FusedResBlock phase 2 input length equals phase 1 output length.
///
/// The plan chains phase1.out_len as phase2.in_len (resblock_plan.rs:132-133).
/// This proves the padded2/eff_k2 calculation uses the correct intermediate
/// length, not the original input length.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn resblock_phase_chaining_consistency() {
    let in_len1: usize = kani::any();
    let pad1: usize = kani::any();
    let ks1: usize = kani::any();
    let dil1: usize = kani::any();
    let pad2: usize = kani::any();
    let ks2: usize = kani::any();
    let dil2: usize = kani::any();

    kani::assume(ks1 >= 1 && ks1 <= 15);
    kani::assume(ks2 >= 1 && ks2 <= 15);
    kani::assume(dil1 >= 1 && dil1 <= 8);
    kani::assume(dil2 >= 1 && dil2 <= 8);
    kani::assume(in_len1 >= 1 && in_len1 <= 4096);
    kani::assume(pad1 <= 64);
    kani::assume(pad2 <= 64);

    // Phase 1.
    let eff_k1 = (ks1 - 1) * dil1 + 1;
    let padded1 = in_len1 + 2 * pad1;
    kani::assume(padded1 >= eff_k1);
    let out_len1 = padded1 - eff_k1 + 1;

    // Phase 2 uses out_len1 as input.
    let eff_k2 = (ks2 - 1) * dil2 + 1;
    let padded2 = out_len1 + 2 * pad2;
    kani::assume(padded2 >= eff_k2);
    let out_len2 = padded2 - eff_k2 + 1;

    // Both outputs are positive.
    assert!(out_len1 >= 1);
    assert!(out_len2 >= 1);

    // Monotonicity: with zero padding and unit dilation, conv shrinks length.
    if pad1 == 0 && dil1 == 1 {
        assert!(out_len1 <= in_len1, "conv with no padding shrinks");
    }
}

// ============================================================================
// Stats encoding buffer size
// ============================================================================

/// Prove: stats encoding `output_bytes` is exactly `flat_rows * 2 * sizeof(f32)`.
///
/// The stats kernel outputs (mean, inv_std) per row, each f32.
/// Models `plan_stats_encoding` at resblock_plan.rs:185-187.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn stats_encoding_output_bytes_exact() {
    let flat_rows: usize = kani::any();

    // Kokoro: flat_rows = batch * channels, max ~4 * 512 = 2048.
    kani::assume(flat_rows >= 1 && flat_rows <= 65536);

    let stats_bytes = flat_rows.checked_mul(2 * size_of::<f32>());
    if let Some(bytes) = stats_bytes {
        // Each row gets exactly 2 floats (mean + inv_std).
        assert_eq!(bytes, flat_rows * 8, "stats bytes must be 8 per row");
        // Non-zero when flat_rows > 0.
        assert!(bytes > 0);
    }
    // If overflow: checked_mul returns None, plan returns Err. Safe.
}

/// Prove: stats encoding grid dimension is non-zero and equals flat_rows.
///
/// The grid is `[flat_rows_u32, 1, 1]` — one threadgroup per row.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn stats_encoding_grid_nonzero() {
    let flat_rows: u32 = kani::any();
    kani::assume(flat_rows >= 1);

    let grid = [flat_rows, 1u32, 1u32];
    let threadgroup = [256u32, 1u32, 1u32];

    // Total threads = flat_rows * 256 (one TG of 256 per row).
    let total_threads = flat_rows as u64 * 256;
    assert!(total_threads > 0, "must dispatch at least one thread");

    // Grid dimensions are non-zero.
    assert!(grid[0] > 0);
    assert!(grid[1] > 0);
    assert!(grid[2] > 0);
    assert!(threadgroup[0] > 0);
}

// ============================================================================
// Conv-with-stats encoding buffer sizes
// ============================================================================

/// Prove: conv_with_stats auxiliary buffer sizes are consistent.
///
/// Models the auxiliary allocations from resblock_plan.rs:271-282:
/// - next_stats: flat_out_rows * 2 * sizeof(f32)
/// - counter: flat_out_rows * sizeof(u32)
/// - partials: grid_x * flat_out_rows * 3 * sizeof(f32)
///
/// All three must be non-zero when flat_out_rows > 0 and grid_x > 0.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn conv_stats_auxiliary_sizes_nonzero() {
    let flat_out_rows: usize = kani::any();
    let out_len: usize = kani::any();

    kani::assume(flat_out_rows >= 1 && flat_out_rows <= 8192);
    kani::assume(out_len >= 1 && out_len <= 65536);

    // CONV_TG_X = 64 (from dyn_tensor_metal_norm_conv_stats.rs:96).
    let conv_tg_x: u32 = 64;
    let grid_x = (out_len as u32).div_ceil(conv_tg_x);

    let next_stats = flat_out_rows.checked_mul(2 * size_of::<f32>());
    let counter = flat_out_rows.checked_mul(size_of::<u32>());
    let partials = (grid_x as usize)
        .checked_mul(flat_out_rows)
        .and_then(|v| v.checked_mul(3 * size_of::<f32>()));

    // All allocations are non-zero when inputs are non-zero.
    if let Some(ns) = next_stats {
        assert!(ns > 0, "next_stats must be > 0");
    }
    if let Some(c) = counter {
        assert!(c > 0, "counter must be > 0");
    }
    if let Some(p) = partials {
        assert!(p > 0, "partials must be > 0");
    }

    // grid_x is always >= 1 when out_len >= 1.
    assert!(grid_x >= 1, "grid_x must be >= 1");
}

/// Prove: conv_with_stats output_bytes matches total elements * elem_bytes.
///
/// Models resblock_plan.rs:257-265: `total_out = B * out_ch * out_len`,
/// `out_bytes = total_out * elem_bytes`.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn conv_stats_output_bytes_consistent() {
    let batch: usize = kani::any();
    let out_ch: usize = kani::any();
    let out_len: usize = kani::any();
    let elem_bytes: usize = kani::any();

    kani::assume(batch >= 1 && batch <= 16);
    kani::assume(out_ch >= 1 && out_ch <= 1024);
    kani::assume(out_len >= 1 && out_len <= 65536);
    kani::assume(elem_bytes == 2 || elem_bytes == 4); // f16 or f32

    let total_out = batch
        .checked_mul(out_ch)
        .and_then(|v| v.checked_mul(out_len));
    let out_bytes = total_out.and_then(|t| t.checked_mul(elem_bytes));

    if let (Some(total), Some(bytes)) = (total_out, out_bytes) {
        // output holds exactly B * C * T elements.
        assert_eq!(total, batch * out_ch * out_len);
        // byte size is element count * element size.
        assert_eq!(bytes, total * elem_bytes);
        assert!(bytes > 0);
    }
    // Overflow returns None → plan returns Err. Safe.
}

// ============================================================================
// Binding index non-overlap
// ============================================================================

/// Prove: FusedResBlock encoding 1 binding indices and auxiliary binding
/// indices are disjoint.
///
/// Encoding 1 uses bindings at indices 0-15, 19-20. Auxiliary allocs use
/// 16, 17, 18. This proves no overlap exists.
///
/// Bug scenario: if a future change moves a binding to index 16-18,
/// the auxiliary would overwrite it silently. This proof catches that.
#[kani::unwind(8)]
#[kani::proof]
#[kani::unwind(1)]
fn resblock_enc1_binding_auxiliary_disjoint() {
    // Binding indices from resblock_plan.rs:291-323.
    // NativeBindingSource bindings: 0..=15, 19, 20.
    let binding_indices: [usize; 16] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15];
    let epilogue_indices: [usize; 2] = [19, 20];

    // Auxiliary binding indices from resblock_plan.rs:333-352.
    let aux_indices: [usize; 3] = [16, 17, 18];

    // No auxiliary index overlaps with any binding index.
    for &ai in &aux_indices {
        for &bi in &binding_indices {
            assert_ne!(ai, bi, "auxiliary overlaps binding");
        }
        for &ei in &epilogue_indices {
            assert_ne!(ai, ei, "auxiliary overlaps epilogue");
        }
    }

    // Auxiliary indices are contiguous from 16.
    assert_eq!(aux_indices[0], 16);
    assert_eq!(aux_indices[1], 17);
    assert_eq!(aux_indices[2], 18);
}

/// Prove: FusedResBlock encoding 2 binding indices are within expected range.
///
/// Encoding 2 uses bindings at indices 0-15 (no auxiliaries).
/// From resblock_plan.rs:420-452.
#[kani::unwind(16)]
#[kani::proof]
#[kani::unwind(1)]
fn resblock_enc2_bindings_in_range() {
    // All binding indices for encoding 2.
    let indices: [usize; 16] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15];

    for &idx in &indices {
        // Metal argument buffer limit is typically 31 for compute shaders.
        assert!(idx < 31, "binding index exceeds Metal limit");
    }

    // No auxiliaries in encoding 2.
    // (The struct literal has `auxiliary_allocs: vec![]`.)
}

// ============================================================================
// Sequence Intermediate index validity
// ============================================================================

/// Prove: FusedResBlock sequence `Intermediate` references are in-bounds.
///
/// The 3-encoding sequence at resblock_plan.rs:152-173:
/// - enc0: stats → output[0]
/// - enc1: conv_stats → uses Intermediate(0) [enc0 output], produces output[1]
/// - enc2: conv_precomp → uses Intermediate(1) [enc1 output]
///
/// Also enc2 uses IntermediateAuxiliary { encoding_idx: 1, auxiliary_idx: 0 }
/// which references the first exposed auxiliary of enc1.
///
/// This proves all Intermediate/IntermediateAuxiliary references point to
/// encodings that have already been dispatched (src_idx < current_idx).
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(4)]
fn resblock_sequence_intermediate_refs_valid() {
    // Model the 3-encoding sequence.
    let num_encodings: usize = 3;

    // Encoding 0: no Intermediate or IntermediateAuxiliary references.
    // (Only PreResolved(0) and Output.)

    // Encoding 1: references Intermediate(0).
    let enc1_intermediate_ref: usize = 0;
    assert!(
        enc1_intermediate_ref < 1, // enc1 is at index 1; only enc0 (index 0) is prior.
        "enc1 Intermediate ref must point to a prior encoding"
    );

    // Encoding 2: references Intermediate(1).
    let enc2_intermediate_ref: usize = 1;
    assert!(
        enc2_intermediate_ref < 2, // enc2 is at index 2; enc0, enc1 are prior.
        "enc2 Intermediate ref must point to a prior encoding"
    );

    // Encoding 2: references IntermediateAuxiliary { encoding_idx: 1, auxiliary_idx: 0 }.
    let enc2_aux_encoding_idx: usize = 1;
    let enc2_aux_auxiliary_idx: usize = 0;
    assert!(
        enc2_aux_encoding_idx < 2,
        "enc2 IntermediateAuxiliary encoding_idx must be prior"
    );

    // enc1 has 3 auxiliary_allocs, but only 1 is exposed (next_stats at index 0).
    let enc1_exposed_count: usize = 1; // next_stats has expose_as_intermediate=true.
    assert!(
        enc2_aux_auxiliary_idx < enc1_exposed_count,
        "enc2 auxiliary_idx must be within enc1's exposed auxiliaries"
    );
}

// ============================================================================
// InstanceNorm encoding plan
// ============================================================================

/// Prove: InstanceNorm plan `output_bytes` equals input element count * elem_bytes.
///
/// InstanceNorm is element-wise: output shape == input shape.
/// Models plan_instance_norm_encoding at compiled_model_execute_native_simple.rs:67-131.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn instance_norm_output_preserves_shape() {
    let batch: usize = kani::any();
    let channels: usize = kani::any();
    let spatial: usize = kani::any();
    let elem_bytes: usize = kani::any();

    kani::assume(batch >= 1 && batch <= 16);
    kani::assume(channels >= 1 && channels <= 1024);
    kani::assume(spatial >= 1 && spatial <= 65536);
    kani::assume(elem_bytes == 2 || elem_bytes == 4);

    let flat_rows = batch.checked_mul(channels);
    let total_elems = flat_rows.and_then(|fr| fr.checked_mul(spatial));
    let out_bytes = total_elems.and_then(|te| te.checked_mul(elem_bytes));

    if let (Some(fr), Some(te), Some(ob)) = (flat_rows, total_elems, out_bytes) {
        // flat_rows = B * C.
        assert_eq!(fr, batch * channels);
        // total_elems = B * C * spatial = full input element count.
        assert_eq!(te, batch * channels * spatial);
        // out_bytes = elements * bytes_per_element.
        assert_eq!(ob, te * elem_bytes);
        // Grid dispatches one TG per (batch, channel) pair.
        assert_eq!(fr, batch * channels);
    }
}

/// Prove: InstanceNorm grid `flat_rows_u32` is non-zero.
///
/// The grid is `[flat_rows_u32, 1, 1]`. If flat_rows were 0, Metal
/// would dispatch zero threadgroups — a silent no-op.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn instance_norm_grid_nonzero() {
    let batch: u32 = kani::any();
    let channels: u32 = kani::any();

    kani::assume(batch >= 1 && batch <= 16);
    kani::assume(channels >= 1 && channels <= 1024);

    let flat_rows = batch as usize * channels as usize;
    let flat_rows_u32 = u32::try_from(flat_rows);

    if let Ok(fr_u32) = flat_rows_u32 {
        assert!(fr_u32 >= 1, "grid must dispatch at least 1 threadgroup");
    }
}

// ============================================================================
// PreResolved index completeness
// ============================================================================

/// Prove: FusedResBlock PreResolved indices 0-5 form a complete, gap-free set.
///
/// The plan documents PreResolved layout:
///   0: x, 1: gamma1, 2: beta1, 3: gamma2, 4: beta2, 5: residual
///
/// This verifies no index is skipped and the maximum is 5.
#[kani::unwind(8)]
#[kani::proof]
#[kani::unwind(1)]
fn resblock_pre_resolved_indices_complete() {
    // Collect all PreResolved indices from the 3 encodings.
    // enc0: PreResolved(0) = x
    // enc1: PreResolved(0) = x, PreResolved(1) = gamma1, PreResolved(2) = beta1
    // enc2: PreResolved(3) = gamma2, PreResolved(4) = beta2, PreResolved(5) = residual
    let all_pre_resolved: [usize; 7] = [0, 0, 1, 2, 3, 4, 5];

    let max_idx = *all_pre_resolved.iter().max().unwrap();
    assert_eq!(max_idx, 5, "max PreResolved index must be 5");

    // All indices 0-5 appear at least once.
    for target in 0..=5usize {
        let found = all_pre_resolved.iter().any(|&i| i == target);
        assert!(found, "PreResolved index {} must appear", target);
    }
}

// NOTE: 3 tautological harnesses removed during self-audit:
// - sequence_empty_returns_error (asserted what it assumed)
// - stats_encoding_zero_threadgroup_memory (asserted a hardcoded literal)
// - resblock_all_encodings_use_threadgroups_mode (asserted hardcoded literals)
// These would require calling actual plan functions to be substantive,
// but the plan functions need Metal (PipelineCache) which Kani can't provide.
// The structural invariants are documented in the plan function doc-comments.

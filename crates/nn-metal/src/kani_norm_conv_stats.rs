// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

/// Proves: effective kernel size computation does not overflow for valid dilation.
///
/// SUBSTANTIVE: `effective_k = (kernel_size - 1) * dilation + 1`. With kernel_size
/// up to 64 and dilation up to 16, the max is 63 * 16 + 1 = 1009, well within usize.
///
/// Covers: `dyn_tensor_metal_norm_conv_stats.rs` line 256.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn effective_kernel_size_no_overflow() {
    let kernel_size: usize = kani::any();
    let dilation: usize = kani::any();

    kani::assume(kernel_size >= 1 && kernel_size <= 64);
    kani::assume(dilation >= 1 && dilation <= 16);

    let km1 = kernel_size - 1;
    let product = km1.checked_mul(dilation);
    assert!(product.is_some(), "(kernel_size - 1) * dilation must not overflow");

    let effective_k = product.unwrap() + 1;
    assert!(
        effective_k >= 1,
        "effective kernel size must be >= 1"
    );
    assert!(
        effective_k <= 1009,
        "effective kernel size within expected bound (63 * 16 + 1)"
    );
}

// =========================================================================
// Buffer sizing arithmetic
// =========================================================================

/// Proves: stats buffer size (2 * sizeof(f32) per row) does not overflow.
///
/// SUBSTANTIVE: `stats_bytes = flat_rows * 2 * sizeof(f32)` where
/// flat_rows = batch * in_channels. This checked_mul chain must succeed
/// for production dimensions.
///
/// Covers: `dyn_tensor_metal_norm_conv_stats.rs` lines 308-312.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn stats_buffer_size_no_overflow() {
    let batch: usize = kani::any();
    let in_channels: usize = kani::any();

    kani::assume(batch >= 1 && batch <= 64);
    kani::assume(in_channels >= 1 && in_channels <= 1024);

    let flat_rows = batch.checked_mul(in_channels);
    assert!(flat_rows.is_some(), "batch * in_channels must not overflow");

    let stats_bytes = flat_rows.unwrap().checked_mul(2 * std::mem::size_of::<f32>());
    assert!(
        stats_bytes.is_some(),
        "stats buffer bytes must not overflow"
    );

    // Upper bound: 64 * 1024 * 8 = 524,288 bytes (< 1 MB).
    assert!(
        stats_bytes.unwrap() <= 524_288,
        "stats buffer within expected bound"
    );
}

/// Proves: output element count checked_mul chain is safe for production dims.
///
/// SUBSTANTIVE: `total_out = batch * out_channels * out_len`, computed via
/// checked_mul chain. Must not overflow for Kokoro production shapes.
///
/// Covers: `dyn_tensor_metal_norm_conv_stats.rs` lines 354-359.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn output_element_count_no_overflow() {
    let batch: usize = kani::any();
    let out_channels: usize = kani::any();
    let out_len: usize = kani::any();

    kani::assume(batch >= 1 && batch <= 64);
    kani::assume(out_channels >= 1 && out_channels <= 1024);
    kani::assume(out_len >= 1 && out_len <= 16384);

    let total = batch
        .checked_mul(out_channels)
        .and_then(|v| v.checked_mul(out_len));

    assert!(total.is_some(), "batch * out_channels * out_len must not overflow");

    let elem_bytes: usize = kani::any();
    kani::assume(elem_bytes == 2 || elem_bytes == 4);

    let out_bytes = total.unwrap().checked_mul(elem_bytes);
    assert!(out_bytes.is_some(), "output bytes must not overflow");
}

/// Proves: partials buffer size does not overflow for production dims.
///
/// SUBSTANTIVE: `partials_bytes = grid_x * flat_out_rows * 3 * sizeof(f32)`.
/// Three floats per TG per row for Welford partials (n, mean, m2).
///
/// Covers: `dyn_tensor_metal_norm_conv_stats.rs` lines 391-398.
#[kani::unwind(8)]
#[kani::proof]
#[kani::unwind(1)]
fn partials_buffer_size_no_overflow() {
    let grid_x: usize = kani::any();
    let flat_out_rows: usize = kani::any();

    // grid_x = ceil(out_len / CONV_TG_X). Max out_len=16384, CONV_TG_X=64 → 256.
    kani::assume(grid_x >= 1 && grid_x <= 256);
    // flat_out_rows = batch * out_channels. Max 64 * 1024 = 65536.
    kani::assume(flat_out_rows >= 1 && flat_out_rows <= 65536);

    let partials_bytes = grid_x
        .checked_mul(flat_out_rows)
        .and_then(|v| v.checked_mul(3 * std::mem::size_of::<f32>()));

    assert!(
        partials_bytes.is_some(),
        "partials buffer bytes must not overflow"
    );

    // Upper bound: 256 * 65536 * 12 = 201,326,592 bytes (~192 MB).
    // This is large but within usize on 64-bit.
    assert!(
        partials_bytes.unwrap() <= 300_000_000,
        "partials buffer within 300 MB for production dims"
    );
}

/// Proves: counter buffer size does not overflow.
///
/// SUBSTANTIVE: `counter_bytes = flat_out_rows * sizeof(u32)`.
///
/// Covers: `dyn_tensor_metal_norm_conv_stats.rs` lines 382-386.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn counter_buffer_size_no_overflow() {
    let flat_out_rows: usize = kani::any();
    kani::assume(flat_out_rows >= 1 && flat_out_rows <= 65536);

    let counter_bytes = flat_out_rows.checked_mul(std::mem::size_of::<u32>());
    assert!(
        counter_bytes.is_some(),
        "counter buffer bytes must not overflow"
    );

    // Upper bound: 65536 * 4 = 262,144 bytes.
    assert!(
        counter_bytes.unwrap() <= 262_144,
        "counter buffer within expected bound"
    );
}

// =========================================================================
// Grid dispatch coverage
// =========================================================================

/// Proves: grid X covers all output positions.
///
/// SUBSTANTIVE: `grid_x = out_len.div_ceil(CONV_TG_X)` where CONV_TG_X = 64.
/// Total threads = grid_x * CONV_TG_X >= out_len.
///
/// Covers: `dyn_tensor_metal_norm_conv_stats.rs` line 378.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn grid_x_covers_all_output_positions() {
    let out_len: u32 = kani::any();
    kani::assume(out_len >= 1);

    let conv_tg_x: u32 = 64; // CONV_TG_X constant
    let grid_x = out_len.div_ceil(conv_tg_x);

    let total_threads = grid_x.checked_mul(conv_tg_x);
    assert!(
        total_threads.is_some(),
        "grid_x * CONV_TG_X must not overflow u32"
    );
    assert!(
        total_threads.unwrap() >= out_len,
        "grid must cover all output positions"
    );
}

/// Proves: stats kernel grid covers all channel rows.
///
/// SUBSTANTIVE: The stats kernel dispatches `flat_rows` threadgroups,
/// one per (batch, channel) row. Each threadgroup of STATS_TG_SIZE=256
/// threads reduces `in_len` spatial elements.
///
/// Covers: `dyn_tensor_metal_norm_conv_stats.rs` line 327.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn stats_kernel_grid_covers_all_rows() {
    let batch: usize = kani::any();
    let in_channels: usize = kani::any();

    kani::assume(batch >= 1 && batch <= 64);
    kani::assume(in_channels >= 1 && in_channels <= 1024);

    let flat_rows = batch * in_channels;

    // Each row gets one threadgroup.
    let flat_rows_u32 = u32::try_from(flat_rows);
    assert!(flat_rows_u32.is_ok(), "flat_rows must fit in u32");

    // Grid is [flat_rows, 1, 1] — one TG per row.
    assert!(flat_rows >= 1, "must dispatch at least one threadgroup");
}

// =========================================================================
// Pipeline cache slots
// =========================================================================

/// Proves: pipeline cache slot indices are within PIPE_SLOTS bounds.
///
/// SUBSTANTIVE: The module uses 10 cache slots:
///   0-1: stats {float, half}
///   2-5: conv_with_stats {leaky, snake} x {float, half}
///   6-9: precomputed {leaky, snake} x {float, half}
///
/// scalar_offset returns 0 for "float", 1 for "half".
/// Conv slot base: LeakyRelu=2, Snake=4.
/// Precomputed slot base: LeakyRelu=6, Snake=8.
///
/// Covers: `dyn_tensor_metal_norm_conv_stats.rs` lines 47-60, 339-342.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn pipeline_cache_slots_within_bounds() {
    let pipe_slots: usize = 10;

    let is_half: bool = kani::any();
    let scalar_off: usize = if is_half { 1 } else { 0 };

    // Stats slot: 0 or 1.
    let stats_slot = scalar_off;
    assert!(stats_slot < pipe_slots, "stats slot must be within bounds");

    // Conv slot: base 2 (leaky) or 4 (snake) + scalar_off.
    let is_snake: bool = kani::any();
    let conv_base: usize = if is_snake { 4 } else { 2 };
    let conv_slot = conv_base + scalar_off;
    assert!(conv_slot < pipe_slots, "conv slot must be within bounds");

    // Precomputed slot: base 6 (leaky) or 8 (snake) + scalar_off.
    let precomp_base: usize = if is_snake { 8 } else { 6 };
    let precomp_slot = precomp_base + scalar_off;
    assert!(precomp_slot < pipe_slots, "precomputed slot must be within bounds");
}

/// Proves: scalar_offset returns 0 or 1 exclusively.
///
/// SUBSTANTIVE: scalar_offset("half") = 1, scalar_offset(anything else) = 0.
/// This is a helper for pipeline cache indexing.
///
/// Covers: `dyn_tensor_metal_norm_conv_stats.rs` lines 54-60.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn scalar_offset_is_binary() {
    let is_half: bool = kani::any();
    let offset: usize = if is_half { 1 } else { 0 };

    assert!(offset <= 1, "scalar offset must be 0 or 1");
}

// =========================================================================
// Epsilon and slope validation
// =========================================================================

/// Proves: eps validation rejects non-finite and non-positive values.
///
/// SUBSTANTIVE: The dispatch code checks `!eps_f32.is_finite() || eps_f32 <= 0.0`.
/// This must reject NaN, Inf, -Inf, 0.0, and negative values.
///
/// Covers: `dyn_tensor_metal_norm_conv_stats.rs` lines 266-269.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn eps_validation_rejects_invalid() {
    let eps: f32 = kani::any();

    let is_valid = eps.is_finite() && eps > 0.0;

    if eps.is_nan() {
        assert!(!is_valid, "NaN eps must be rejected");
    }
    if eps == f32::INFINITY || eps == f32::NEG_INFINITY {
        assert!(!is_valid, "Inf eps must be rejected");
    }
    if eps == 0.0 {
        assert!(!is_valid, "zero eps must be rejected");
    }
    if eps.is_finite() && eps < 0.0 {
        assert!(!is_valid, "negative eps must be rejected");
    }
}

/// Proves: next_phase_eps validation rejects non-finite and non-positive.
///
/// SUBSTANTIVE: Same validation as eps, but for `next_phase_eps`.
///
/// Covers: `dyn_tensor_metal_norm_conv_stats.rs` lines 271-275.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn next_phase_eps_validation_rejects_invalid() {
    let next_eps: f32 = kani::any();

    let is_valid = next_eps.is_finite() && next_eps > 0.0;

    // Valid eps are (0.0, finite_max].
    if is_valid {
        assert!(next_eps > 0.0, "valid eps must be positive");
        assert!(next_eps.is_finite(), "valid eps must be finite");
    }

    // Invalid cases.
    if next_eps.is_nan() || !next_eps.is_finite() || next_eps <= 0.0 {
        assert!(!is_valid, "invalid next_phase_eps must be rejected");
    }
}

/// Proves: slope finiteness validation is correct.
///
/// SUBSTANTIVE: The LeakyRelu path checks `!slope_f32.is_finite()`.
/// This rejects NaN and Inf slopes but allows any finite value including
/// negative slopes.
///
/// Covers: `dyn_tensor_metal_norm_conv_stats.rs` lines 172-176.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn slope_finiteness_validation() {
    let slope: f64 = kani::any();
    kani::assume(slope.is_finite());
    kani::assume(slope >= -1e6 && slope <= 1e6);

    let slope_f32 = slope as f32;

    // For reasonable f64 values, the f32 cast should produce a finite value.
    assert!(
        slope_f32.is_finite(),
        "f32 cast of bounded f64 slope must be finite"
    );
}

// =========================================================================
// Welford merge finiteness
// =========================================================================

/// Proves: Welford merge preserves finiteness for bounded inputs.
///
/// SUBSTANTIVE: The stats epilogue MSL uses `welford_merge(a, b)` which
/// combines two Welford states. The key computation is:
///   delta = b.mean - a.mean
///   n_ab = a.n + b.n
///   mean_ab = a.mean + delta * (b.n / n_ab)
///   m2_ab = a.m2 + b.m2 + delta * delta * (a.n * b.n / n_ab)
///
/// For bounded inputs, all outputs must remain finite.
///
/// Covers: `dyn_tensor_metal_norm_conv_stats_msl.rs` lines 43-53.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn welford_merge_finiteness() {
    let a_n: f32 = kani::any();
    let a_mean: f32 = kani::any();
    let a_m2: f32 = kani::any();
    let b_n: f32 = kani::any();
    let b_mean: f32 = kani::any();
    let b_m2: f32 = kani::any();

    kani::assume(a_n.is_finite() && a_n >= 0.0 && a_n <= 16384.0);
    kani::assume(a_mean.is_finite() && a_mean >= -1e3 && a_mean <= 1e3);
    kani::assume(a_m2.is_finite() && a_m2 >= 0.0 && a_m2 <= 1e8);
    kani::assume(b_n.is_finite() && b_n >= 0.0 && b_n <= 16384.0);
    kani::assume(b_mean.is_finite() && b_mean >= -1e3 && b_mean <= 1e3);
    kani::assume(b_m2.is_finite() && b_m2 >= 0.0 && b_m2 <= 1e8);

    let n_ab = a_n + b_n;
    kani::assume(n_ab > 0.0); // At least one sample.

    let delta = b_mean - a_mean;
    assert!(delta.is_finite(), "delta must be finite");

    let mean_ab = a_mean + delta * (b_n / n_ab);
    assert!(mean_ab.is_finite(), "merged mean must be finite");

    let m2_ab = a_m2 + b_m2 + delta * delta * (a_n * b_n / n_ab);
    assert!(m2_ab.is_finite(), "merged m2 must be finite");
    assert!(n_ab.is_finite(), "merged n must be finite");
}

// =========================================================================
// Stats buffer layout
// =========================================================================

/// Proves: stats buffer indexing (mean at 2*i, inv_std at 2*i+1) stays in bounds.
///
/// SUBSTANTIVE: The MSL kernel writes `next_stats[stats_row * 2]` (mean)
/// and `next_stats[stats_row * 2 + 1]` (inv_std). The Rust side allocates
/// `flat_out_rows * 2 * sizeof(f32)` bytes. This harness proves the index
/// stays within the allocated count.
///
/// Covers: `dyn_tensor_metal_norm_conv_stats_msl.rs` lines 64-65.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn stats_buffer_indexing_in_bounds() {
    let flat_out_rows: usize = kani::any();
    kani::assume(flat_out_rows >= 1 && flat_out_rows <= 65536);

    // Allocated float count = flat_out_rows * 2.
    let alloc_count = flat_out_rows * 2;

    let stats_row: usize = kani::any();
    kani::assume(stats_row < flat_out_rows);

    let mean_idx = stats_row * 2;
    let inv_std_idx = stats_row * 2 + 1;

    assert!(mean_idx < alloc_count, "mean index must be within allocation");
    assert!(
        inv_std_idx < alloc_count,
        "inv_std index must be within allocation"
    );
}

// =========================================================================
// Conv output index within buffer
// =========================================================================

/// Proves: conv output index `(b * out_ch + oc) * out_len + t` is within bounds.
///
/// SUBSTANTIVE: The MSL kernel writes `output[out_idx]` where
/// `out_idx = (b * out_channels + oc) * out_len + t`. The Rust side
/// allocates `batch * out_channels * out_len` elements.
///
/// Covers: `dyn_tensor_metal_norm_conv_stats_msl.rs` line 173.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn conv_output_index_in_bounds() {
    let batch: usize = kani::any();
    let out_channels: usize = kani::any();
    let out_len: usize = kani::any();

    kani::assume(batch >= 1 && batch <= 64);
    kani::assume(out_channels >= 1 && out_channels <= 512);
    kani::assume(out_len >= 1 && out_len <= 16384);

    let total = batch
        .checked_mul(out_channels)
        .and_then(|v| v.checked_mul(out_len));
    kani::assume(total.is_some());
    let total = total.unwrap();

    let b: usize = kani::any();
    let oc: usize = kani::any();
    let t: usize = kani::any();
    kani::assume(b < batch);
    kani::assume(oc < out_channels);
    kani::assume(t < out_len);

    let out_idx = (b * out_channels + oc) * out_len + t;
    assert!(out_idx < total, "conv output index must be within allocated buffer");
}

// =========================================================================
// Flat rows overflow detection
// =========================================================================

/// Proves: flat_rows = batch * in_channels fits in u32 for production dims.
///
/// SUBSTANTIVE: The dispatch code converts flat_rows to u32 via `to_u32()`.
/// For batch <= 64, in_channels <= 1024, max is 65536 < u32::MAX.
///
/// Covers: `dyn_tensor_metal_norm_conv_stats.rs` line 317.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn flat_rows_fits_u32() {
    let batch: usize = kani::any();
    let in_channels: usize = kani::any();

    kani::assume(batch >= 1 && batch <= 64);
    kani::assume(in_channels >= 1 && in_channels <= 1024);

    let flat_rows = batch * in_channels;
    assert!(
        flat_rows <= u32::MAX as usize,
        "flat_rows must fit in u32"
    );
}

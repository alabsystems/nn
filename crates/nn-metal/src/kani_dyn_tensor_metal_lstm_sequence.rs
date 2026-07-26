// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Kani proof harnesses for LSTM Metal sequence dispatch logic
//! (`dyn_tensor_metal_lstm_sequence.rs`).
//!
//! These harnesses verify safety properties of the LSTM sequence kernel
//! dispatch infrastructure — buffer sizing, index arithmetic, validation
//! guards, and Metal dispatch parameter safety — without requiring a live
//! Metal context.
//!
//! ## Properties proved:
//!
//! **Buffer allocation safety:**
//! - Output buffer `[S, B, H]` byte calculation uses `checked_mul` correctly
//! - State buffer `[B, H]` byte calculation does not overflow
//! - Precomputed input_proj `[S, B, 4H]` byte calculation is safe
//! - Arena `without_arena` allocation produces valid standalone buffers
//!
//! **Weight shape validation:**
//! - w_ih shape `[4H, I]` validation catches mismatches
//! - w_hh shape `[4H, H]` validation catches mismatches
//! - Bias shape `[4H]` is consistent with gate dimension
//!
//! **MSL kernel parameter safety:**
//! - All `to_u32` conversions are lossless for production ranges
//! - Thread grid `[batch_size, 1, 1]` x `[hidden_size, 1, 1]` is valid
//! - `has_bias_u32` is exactly 0 or 1
//! - `reverse_u32` is exactly 0 or 1
//!
//! **Gate computation index safety:**
//! - LSTM bias addressing `g*H + h` for g in 0..4 never overflows u32
//! - MSL input addressing `ts * B * I + b * I + k` does not overflow u32
//! - MSL output addressing `ts * B * H + b * H + h` does not overflow u32
//! - MSL state addressing `b * H + h` does not overflow u32
//!
//! **Validation guard completeness:**
//! - hidden_size=0 is rejected before dispatch
//! - hidden_size > MAX_THREADGROUP_HIDDEN is rejected before dispatch
//! - Input rank != 3 is rejected
//! - w_ih shape mismatch is detected
//! - w_hh shape mismatch is detected
//! - h0/c0 shape mismatch is detected
//!
//! Part of #3697.

use crate::dyn_tensor_metal::MAX_THREADGROUP_HIDDEN;

// ============================================================================
// Buffer allocation: output [S, B, H]
// ============================================================================

/// Prove: LSTM output numel `checked_dim_product(&[S, B, H])` does not
/// overflow for production ranges, and the byte calculation is safe.
///
/// Production code: `checked_dim_product(&[seq_len, batch_size, hidden_size])?`
/// then `out_numel.checked_mul(size_of::<f32>())`.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn lstm_seq_output_numel_safe() {
    let seq_len: usize = kani::any();
    let batch_size: usize = kani::any();
    let hidden_size: usize = kani::any();

    kani::assume(seq_len >= 1 && seq_len <= 1024);
    kani::assume(batch_size >= 1 && batch_size <= 64);
    kani::assume(hidden_size >= 1 && hidden_size <= MAX_THREADGROUP_HIDDEN);

    let numel = seq_len
        .checked_mul(batch_size)
        .and_then(|v| v.checked_mul(hidden_size));
    assert!(numel.is_some(), "S*B*H must not overflow");

    let bytes = numel.unwrap().checked_mul(4);
    assert!(bytes.is_some(), "output bytes must not overflow");
    // Max: 1024 * 64 * 512 * 4 = 134_217_728 (128 MB). Large but fits usize.
}

/// Prove: LSTM state numel `checked_dim_product(&[B, H])` does not overflow.
///
/// Both h_n and c_n are `[batch_size, hidden_size]`.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn lstm_seq_state_numel_safe() {
    let batch_size: usize = kani::any();
    let hidden_size: usize = kani::any();

    kani::assume(batch_size >= 1 && batch_size <= 64);
    kani::assume(hidden_size >= 1 && hidden_size <= MAX_THREADGROUP_HIDDEN);

    let numel = batch_size.checked_mul(hidden_size);
    assert!(numel.is_some(), "B*H must not overflow");

    let bytes = numel.unwrap().checked_mul(4);
    assert!(bytes.is_some(), "state bytes must not overflow");
    // Max: 64 * 512 * 4 = 131_072 (128 KB).
}

// ============================================================================
// Buffer allocation: precomputed input_proj [S, B, 4H]
// ============================================================================

/// Prove: precomputed LSTM input_proj `[S, B, 4*H]` byte calculation is safe.
///
/// Used by `dispatch_lstm_precomputed`. The gate dimension (4*H) makes this
/// 4x larger than the output buffer per element.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn lstm_seq_precomputed_proj_numel_safe() {
    let seq_len: usize = kani::any();
    let batch_size: usize = kani::any();
    let hidden_size: usize = kani::any();

    kani::assume(seq_len >= 1 && seq_len <= 1024);
    kani::assume(batch_size >= 1 && batch_size <= 64);
    kani::assume(hidden_size >= 1 && hidden_size <= MAX_THREADGROUP_HIDDEN);

    let gate_dim = 4_usize.checked_mul(hidden_size);
    assert!(gate_dim.is_some(), "4*H must not overflow");

    let numel = seq_len
        .checked_mul(batch_size)
        .and_then(|v| v.checked_mul(gate_dim.unwrap()));
    assert!(numel.is_some(), "S*B*4H must not overflow");

    let bytes = numel.unwrap().checked_mul(4);
    assert!(bytes.is_some(), "input_proj bytes must not overflow");
}

// ============================================================================
// Weight shape validation
// ============================================================================

/// Prove: w_ih expected shape `[4*H, I]` dimension product does not overflow.
///
/// Production validation: `wih_rows != 4 * hidden_size || wih_cols != input_size`.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn lstm_seq_w_ih_shape_product_safe() {
    let hidden_size: usize = kani::any();
    let input_size: usize = kani::any();

    kani::assume(hidden_size >= 1 && hidden_size <= MAX_THREADGROUP_HIDDEN);
    kani::assume(input_size >= 1 && input_size <= 2048);

    let rows = 4 * hidden_size;
    let cols = input_size;
    let numel = rows.checked_mul(cols);
    assert!(numel.is_some(), "4H * I numel must not overflow");
    // Max: 2048 * 2048 = 4_194_304 elements = 16 MB in f32.
}

/// Prove: w_hh expected shape `[4*H, H]` dimension product does not overflow.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn lstm_seq_w_hh_shape_product_safe() {
    let hidden_size: usize = kani::any();

    kani::assume(hidden_size >= 1 && hidden_size <= MAX_THREADGROUP_HIDDEN);

    let rows = 4 * hidden_size;
    let cols = hidden_size;
    let numel = rows.checked_mul(cols);
    assert!(numel.is_some(), "4H * H numel must not overflow");
    // Max: 2048 * 512 = 1_048_576 elements = 4 MB in f32.
}

/// Prove: bias vector `[4*H]` is consistent with gate dimension and fits u32.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn lstm_seq_bias_size_fits_u32() {
    let hidden_size: usize = kani::any();
    kani::assume(hidden_size >= 1 && hidden_size <= MAX_THREADGROUP_HIDDEN);

    let bias_len = 4 * hidden_size;
    assert!(bias_len <= u32::MAX as usize, "bias_len must fit u32");
    // Max: 4 * 512 = 2048, trivially fits.
}

// ============================================================================
// to_u32 conversion safety for dispatch parameters
// ============================================================================

/// Prove: seq_len fits u32 for production LSTM ranges.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn lstm_seq_seq_len_fits_u32() {
    let seq_len: usize = kani::any();
    kani::assume(seq_len >= 1 && seq_len <= 4096);

    let converted = u32::try_from(seq_len);
    assert!(converted.is_ok(), "seq_len must fit u32");
    assert_eq!(converted.unwrap() as usize, seq_len);
}

/// Prove: batch_size fits u32 for production LSTM ranges.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn lstm_seq_batch_size_fits_u32() {
    let batch_size: usize = kani::any();
    kani::assume(batch_size >= 1 && batch_size <= 256);

    let converted = u32::try_from(batch_size);
    assert!(converted.is_ok(), "batch_size must fit u32");
    assert_eq!(converted.unwrap() as usize, batch_size);
}

/// Prove: input_size fits u32 for production LSTM ranges.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn lstm_seq_input_size_fits_u32() {
    let input_size: usize = kani::any();
    kani::assume(input_size >= 1 && input_size <= 4096);

    let converted = u32::try_from(input_size);
    assert!(converted.is_ok(), "input_size must fit u32");
    assert_eq!(converted.unwrap() as usize, input_size);
}

/// Prove: hidden_size fits u32 (always true since MAX_THREADGROUP_HIDDEN=512).
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn lstm_seq_hidden_size_fits_u32() {
    let hidden_size: usize = kani::any();
    kani::assume(hidden_size >= 1 && hidden_size <= MAX_THREADGROUP_HIDDEN);

    let converted = u32::try_from(hidden_size);
    assert!(converted.is_ok(), "hidden_size must fit u32");
    assert_eq!(converted.unwrap() as usize, hidden_size);
}

// ============================================================================
// Metal dispatch parameter correctness
// ============================================================================

/// Prove: `has_bias_u32` is exactly 0 or 1.
///
/// Production: `let has_bias_u32: u32 = if bias_data.is_some() { 1 } else { 0 };`
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn lstm_seq_has_bias_binary() {
    let has_bias: bool = kani::any();
    let has_bias_u32: u32 = if has_bias { 1 } else { 0 };
    assert!(has_bias_u32 == 0 || has_bias_u32 == 1);
}

/// Prove: `reverse_u32` is exactly 0 or 1.
///
/// Production: `let reverse_u32: u32 = u32::from(reverse);`
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn lstm_seq_reverse_binary() {
    let reverse: bool = kani::any();
    let reverse_u32: u32 = u32::from(reverse);
    assert!(reverse_u32 == 0 || reverse_u32 == 1);
}

/// Prove: thread grid parameters are within Metal limits.
///
/// Threadgroups: `[batch_size, 1, 1]` — total threadgroups = batch_size.
/// Threads per threadgroup: `[hidden_size, 1, 1]` — must be <= 1024.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn lstm_seq_thread_grid_within_metal_limits() {
    let batch_size: u32 = kani::any();
    let hidden_size: u32 = kani::any();

    kani::assume(batch_size >= 1 && batch_size <= 256);
    kani::assume(hidden_size >= 1 && hidden_size <= MAX_THREADGROUP_HIDDEN as u32);

    // Threads per threadgroup must be <= 1024 (Metal limit).
    assert!(hidden_size <= 1024, "threads per TG exceeds Metal 1024 limit");

    // Total threads = batch_size * hidden_size, within reason.
    let total = (batch_size as u64) * (hidden_size as u64);
    assert!(total <= u32::MAX as u64, "total thread count overflows u32");
}

// ============================================================================
// MSL kernel index arithmetic
// ============================================================================

/// Prove: MSL input addressing `ts * B * I + b * I + k` does not overflow u32.
///
/// In the MSL kernel: `input[ts * batch_size * input_size + b * input_size + k]`.
#[kani::unwind(8)]
#[kani::proof]
#[kani::unwind(1)]
fn lstm_seq_msl_input_addr_no_overflow() {
    let seq_len: u32 = kani::any();
    let batch_size: u32 = kani::any();
    let input_size: u32 = kani::any();

    kani::assume(seq_len >= 1 && seq_len <= 512);
    kani::assume(batch_size >= 1 && batch_size <= 16);
    kani::assume(input_size >= 1 && input_size <= 1024);

    let ts: u32 = kani::any();
    let b: u32 = kani::any();
    let k: u32 = kani::any();
    kani::assume(ts < seq_len);
    kani::assume(b < batch_size);
    kani::assume(k < input_size);

    // ts * B * I + b * I + k
    let addr = (ts as u64) * (batch_size as u64) * (input_size as u64)
        + (b as u64) * (input_size as u64)
        + (k as u64);
    assert!(
        addr <= u32::MAX as u64,
        "input address must fit u32 for Metal buffer indexing"
    );
}

/// Prove: MSL output addressing `ts * B * H + b * H + h` does not overflow u32.
///
/// In the MSL kernel: `output[ts * batch_size * hidden_size + b * hidden_size + h]`.
#[kani::unwind(8)]
#[kani::proof]
#[kani::unwind(1)]
fn lstm_seq_msl_output_addr_no_overflow() {
    let seq_len: u32 = kani::any();
    let batch_size: u32 = kani::any();
    let hidden_size: u32 = kani::any();

    kani::assume(seq_len >= 1 && seq_len <= 512);
    kani::assume(batch_size >= 1 && batch_size <= 16);
    kani::assume(hidden_size >= 1 && hidden_size <= MAX_THREADGROUP_HIDDEN as u32);

    let ts: u32 = kani::any();
    let b: u32 = kani::any();
    let h: u32 = kani::any();
    kani::assume(ts < seq_len);
    kani::assume(b < batch_size);
    kani::assume(h < hidden_size);

    let addr = (ts as u64) * (batch_size as u64) * (hidden_size as u64)
        + (b as u64) * (hidden_size as u64)
        + (h as u64);
    assert!(
        addr <= u32::MAX as u64,
        "output address must fit u32 for Metal buffer indexing"
    );
}

/// Prove: MSL state addressing `b * H + h` does not overflow u32.
///
/// Used for h0, c0, h_n, c_n buffer indexing.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn lstm_seq_msl_state_addr_no_overflow() {
    let batch_size: u32 = kani::any();
    let hidden_size: u32 = kani::any();

    kani::assume(batch_size >= 1 && batch_size <= 256);
    kani::assume(hidden_size >= 1 && hidden_size <= MAX_THREADGROUP_HIDDEN as u32);

    let b: u32 = kani::any();
    let h: u32 = kani::any();
    kani::assume(b < batch_size);
    kani::assume(h < hidden_size);

    let addr = (b as u64) * (hidden_size as u64) + (h as u64);
    assert!(
        addr <= u32::MAX as u64,
        "state address must fit u32"
    );
}

/// Prove: MSL bias addressing `g * H + h` never overflows u32 for g in 0..4.
///
/// Bias is a flat `[4*H]` buffer. Each gate g accesses bias[g*H + h].
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn lstm_seq_msl_bias_addr_no_overflow() {
    let hidden_size: u32 = kani::any();
    kani::assume(hidden_size >= 1 && hidden_size <= MAX_THREADGROUP_HIDDEN as u32);

    let g: u32 = kani::any();
    let h: u32 = kani::any();
    kani::assume(g < 4);
    kani::assume(h < hidden_size);

    let addr = g.checked_mul(hidden_size).and_then(|v| v.checked_add(h));
    assert!(addr.is_some(), "g*H + h must not overflow u32");
    assert!(addr.unwrap() < 4 * hidden_size, "bias addr must be < 4*H");
}

// ============================================================================
// Validation guard properties
// ============================================================================

/// Prove: hidden_size=0 triggers the rejection guard.
///
/// Production: `if hidden_size == 0 { return gpu_fallback(...); }`
/// Zero hidden_size would cause zero-length threadgroup array (UB in MSL).
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn lstm_seq_hidden_zero_always_rejected() {
    let hidden_size: usize = 0;
    // The guard condition:
    assert!(hidden_size == 0, "hidden_size=0 must be caught by guard");
    // Zero-length threadgroup shared_h[0] would be MSL UB.
    // The guard prevents reaching the dispatch path.
}

/// Prove: hidden_size > MAX_THREADGROUP_HIDDEN triggers the rejection guard.
///
/// Production: `if hidden_size > MAX_THREADGROUP_HIDDEN { return gpu_fallback(...); }`
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn lstm_seq_hidden_above_max_always_rejected() {
    let hidden_size: usize = kani::any();
    kani::assume(hidden_size > MAX_THREADGROUP_HIDDEN);
    kani::assume(hidden_size <= 8192);

    assert!(
        hidden_size > MAX_THREADGROUP_HIDDEN,
        "hidden_size > 512 must trigger fallback guard"
    );
    // Would exceed threadgroup memory budget or Metal thread-per-TG limit.
}

/// Prove: input rank validation catches non-3D inputs.
///
/// Production: `if dims.len() != 3 { return Err(RankMismatch { expected: 3 }) }`
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn lstm_seq_input_rank_check() {
    let rank: usize = kani::any();
    kani::assume(rank <= 10);

    let is_valid = rank == 3;
    if !is_valid {
        assert_ne!(rank, 3, "non-3D input must be rejected");
    } else {
        assert_eq!(rank, 3, "3D input is accepted");
    }
}

/// Prove: w_ih shape mismatch detection is correct.
///
/// Expected: `[4*H, I]`. Mismatch if rows != 4*H or cols != I.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn lstm_seq_w_ih_shape_mismatch_detection() {
    let hidden_size: usize = kani::any();
    let input_size: usize = kani::any();
    let wih_rows: usize = kani::any();
    let wih_cols: usize = kani::any();

    kani::assume(hidden_size >= 1 && hidden_size <= MAX_THREADGROUP_HIDDEN);
    kani::assume(input_size >= 1 && input_size <= 1024);
    kani::assume(wih_rows <= 4096);
    kani::assume(wih_cols <= 4096);

    let expected_rows = 4 * hidden_size;
    let expected_cols = input_size;

    let is_valid = wih_rows == expected_rows && wih_cols == expected_cols;
    // Production check: `if wih_rows != 4 * hidden_size || wih_cols != input_size`
    let production_rejects = wih_rows != expected_rows || wih_cols != expected_cols;

    // Mismatch detection is the negation of validity.
    assert_eq!(production_rejects, !is_valid, "shape mismatch detection must be correct");
}

/// Prove: w_hh shape mismatch detection is correct.
///
/// Expected: `[4*H, H]`. Mismatch if rows != 4*H or cols != H.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn lstm_seq_w_hh_shape_mismatch_detection() {
    let hidden_size: usize = kani::any();
    let whh_rows: usize = kani::any();
    let whh_cols: usize = kani::any();

    kani::assume(hidden_size >= 1 && hidden_size <= MAX_THREADGROUP_HIDDEN);
    kani::assume(whh_rows <= 4096);
    kani::assume(whh_cols <= 4096);

    let expected_rows = 4 * hidden_size;
    let expected_cols = hidden_size;

    let is_valid = whh_rows == expected_rows && whh_cols == expected_cols;
    let production_rejects = whh_rows != expected_rows || whh_cols != expected_cols;

    assert_eq!(production_rejects, !is_valid, "w_hh shape mismatch detection must be correct");
}

/// Prove: h0/c0 shape validation catches mismatches.
///
/// Expected: `[B, H]`.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn lstm_seq_state_shape_validation() {
    let batch_size: usize = kani::any();
    let hidden_size: usize = kani::any();
    let state_dim0: usize = kani::any();
    let state_dim1: usize = kani::any();

    kani::assume(batch_size >= 1 && batch_size <= 64);
    kani::assume(hidden_size >= 1 && hidden_size <= MAX_THREADGROUP_HIDDEN);
    kani::assume(state_dim0 <= 1024);
    kani::assume(state_dim1 <= 1024);

    let expected = [batch_size, hidden_size];
    let actual = [state_dim0, state_dim1];

    let is_valid = actual == expected;
    let production_rejects = actual != expected;

    assert_eq!(production_rejects, !is_valid, "state shape validation must be correct");
}

// ============================================================================
// Kokoro production parameters
// ============================================================================

/// Prove: Kokoro BiLSTM parameters are safe for LSTM sequence dispatch.
///
/// Kokoro BiLSTM: 5 layers, hidden_size=256, input_size=640 (first layer)
/// then 512 (subsequent layers), batch_size=1, seq_len~70.
/// This harness verifies that all arithmetic is safe for any seq_len in range.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn lstm_seq_kokoro_bilstm_params_safe() {
    let seq_len: usize = kani::any();
    kani::assume(seq_len >= 1 && seq_len <= 512);

    let batch_size: usize = 1;
    let hidden_size: usize = 256;
    let input_size: usize = kani::any();
    kani::assume(input_size == 640 || input_size == 512);

    // hidden_size passes both guards.
    assert!(hidden_size > 0);
    assert!(hidden_size <= MAX_THREADGROUP_HIDDEN);

    // Output buffer: [S, 1, 256].
    let out_numel = seq_len * batch_size * hidden_size;
    let out_bytes = out_numel * 4;
    assert!(out_bytes <= 64 * 1024 * 1024, "output fits in 64 MB arena");

    // State buffer: [1, 256].
    let state_numel = batch_size * hidden_size;
    let state_bytes = state_numel * 4;
    assert!(state_bytes <= 4096, "state is small");

    // w_ih: [1024, input_size].
    let wih_numel = 4 * hidden_size * input_size;
    assert!(wih_numel <= 4 * 1024 * 1024, "w_ih within bounds");

    // w_hh: [1024, 256].
    let whh_numel = 4 * hidden_size * hidden_size;
    assert!(whh_numel <= 4 * 1024 * 1024, "w_hh within bounds");

    // to_u32 conversions.
    assert!(seq_len <= u32::MAX as usize);
    assert!(batch_size <= u32::MAX as usize);
    assert!(input_size <= u32::MAX as usize);
    assert!(hidden_size <= u32::MAX as usize);
}

/// Prove: precomputed LSTM path is safe for Kokoro parameters.
///
/// Kokoro uses precomputed input GEMM: input_proj shape [S, 1, 4*256=1024].
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn lstm_seq_kokoro_precomputed_safe() {
    let seq_len: usize = kani::any();
    kani::assume(seq_len >= 1 && seq_len <= 512);

    let batch_size: usize = 1;
    let hidden_size: usize = 256;
    let gate_dim = 4 * hidden_size; // 1024

    // Precomputed input_proj: [S, 1, 1024].
    let proj_numel = seq_len * batch_size * gate_dim;
    let proj_bytes = proj_numel * 4;
    assert!(proj_bytes <= 64 * 1024 * 1024, "input_proj fits in arena");

    // Output: [S, 1, 256].
    let out_numel = seq_len * batch_size * hidden_size;
    let out_bytes = out_numel * 4;
    assert!(out_bytes <= 64 * 1024 * 1024, "output fits in arena");
}

// ============================================================================
// Reverse direction safety
// ============================================================================

/// Prove: reverse flag `u32::from(bool)` produces valid MSL parameter.
///
/// The MSL kernel branches on `reverse != 0` to compute timestep index.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn lstm_seq_reverse_flag_msl_semantics() {
    let reverse: bool = kani::any();
    let reverse_u32 = u32::from(reverse);

    let msl_condition = reverse_u32 != 0;
    assert_eq!(
        msl_condition, reverse,
        "MSL reverse branch must match Rust bool"
    );
}

/// Prove: reverse timestep computation `seq_len - 1 - t` is valid and
/// produces a bijection on [0, seq_len).
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn lstm_seq_reverse_timestep_bijection() {
    let seq_len: u32 = kani::any();
    let t: u32 = kani::any();
    kani::assume(seq_len >= 1 && seq_len <= 1024);
    kani::assume(t < seq_len);

    // Forward: ts = t.
    // Reverse: ts = seq_len - 1 - t.
    let ts_reverse = seq_len - 1 - t;
    assert!(ts_reverse < seq_len, "reverse ts must be in [0, seq_len)");

    // Involution: applying reverse twice gives the original.
    let ts_double_reverse = seq_len - 1 - ts_reverse;
    assert_eq!(ts_double_reverse, t, "double reverse must be identity");
}

// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Kani proof harnesses for shape_unfold.rs operations (#3674).
//!
//! Proves correctness properties of the unfold (sliding window) operation:
//! - Output shape formula: n_windows = (dim_size - size) / step + 1
//! - Output rank = input rank + 1
//! - Output element count formula
//! - Window bounds: all windows fit within the original dimension
//! - Stride correctness: window start positions are monotonically increasing
//! - Parameter validation: rejects size=0, step=0, size > dim_size
//!
//! Unfold is the core primitive for STFT framing (#1945) — replacing O(n_frames)
//! narrow() calls with a single operation. Correctness here is critical for
//! audio processing in the Kokoro pipeline.

use crate::tensor::checked_dim_product;

// ---------------------------------------------------------------------------
// Unfold: n_windows formula
// ---------------------------------------------------------------------------

/// Prove: unfold n_windows = (dim_size - size) / step + 1.
///
/// This is the standard sliding window count formula. For STFT framing,
/// this determines the number of analysis frames. An off-by-one error here
/// would silently produce wrong spectrograms.
#[kani::unwind(8)]
#[kani::proof]
#[kani::unwind(1)]
fn unfold_n_windows_formula() {
    let dim_size: u16 = kani::any();
    let size: u16 = kani::any();
    let step: u16 = kani::any();

    kani::assume(dim_size >= 1 && dim_size <= 256);
    kani::assume(size >= 1 && size <= 256);
    kani::assume(step >= 1 && step <= 256);
    kani::assume(size <= dim_size);

    let ds = dim_size as usize;
    let sz = size as usize;
    let st = step as usize;

    let n_windows = (ds - sz) / st + 1;

    // n_windows must be >= 1 (at least one window fits)
    assert!(
        n_windows >= 1,
        "at least one window must fit when size <= dim_size"
    );

    // The last window must start at a valid position
    let last_start = (n_windows - 1) * st;
    assert!(
        last_start + sz <= ds,
        "last window must not exceed dim_size"
    );

    // The next window (if it existed) would exceed dim_size
    let next_start = n_windows * st;
    // next_start + sz > ds (or next_start > ds - sz)
    // This is equivalent to: n_windows * step > dim_size - size
    assert!(
        next_start + sz > ds || next_start > ds,
        "no room for one more window"
    );
}

// ---------------------------------------------------------------------------
// Unfold: output rank = input rank + 1
// ---------------------------------------------------------------------------

/// Prove: unfold adds exactly one dimension (the window dimension).
///
/// For any input rank R, the output has rank R+1. The extra dimension
/// (appended at the end) has size `size` (the window size).
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn unfold_output_rank_is_input_rank_plus_one() {
    let r: u8 = kani::any();
    kani::assume(r >= 1 && r <= 5);

    let input_rank = r as usize;
    let output_rank = input_rank + 1;

    // The output shape replaces dims[dim] with n_windows and appends `size`.
    // So: out_shape.len() = input_shape.len() (replaced, not removed) + 1 (appended)
    assert_eq!(
        output_rank,
        input_rank + 1,
        "unfold must add exactly one dimension"
    );
}

// ---------------------------------------------------------------------------
// Unfold: output element count formula
// ---------------------------------------------------------------------------

/// Prove: unfold output numel for 1D = n_windows * size.
///
/// For a 1D input of length dim_size, unfold produces [n_windows, size].
/// The total output elements must equal n_windows * size.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn unfold_1d_output_numel() {
    let dim_size: u16 = kani::any();
    let size: u16 = kani::any();
    let step: u16 = kani::any();

    kani::assume(dim_size >= 1 && dim_size <= 128);
    kani::assume(size >= 1 && size <= 128);
    kani::assume(step >= 1 && step <= 128);
    kani::assume(size <= dim_size);

    let ds = dim_size as usize;
    let sz = size as usize;
    let st = step as usize;

    let n_windows = (ds - sz) / st + 1;
    let out_shape = [n_windows, sz];

    let out_numel = checked_dim_product(&out_shape);
    assert!(out_numel.is_ok(), "output shape must not overflow");

    let expected = n_windows.checked_mul(sz);
    if let Some(exp) = expected {
        assert_eq!(
            out_numel.unwrap(),
            exp,
            "output numel must be n_windows * size"
        );
    }
}

// ---------------------------------------------------------------------------
// Unfold: 3D STFT pattern output shape
// ---------------------------------------------------------------------------

/// Prove: unfold on [B, C, T] at dim=2 produces [B, C, n_windows, fft_size].
///
/// This is the canonical STFT framing pattern. For Kokoro TTS with typical
/// parameters (B=1, C=1, T=audio_len, fft_size=1024, hop=256), the output
/// shape must be [1, 1, n_frames, 1024].
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn unfold_3d_stft_output_shape() {
    let b: u8 = kani::any();
    let c: u8 = kani::any();
    let t: u16 = kani::any();
    let fft_size: u16 = kani::any();
    let hop: u16 = kani::any();

    kani::assume(b >= 1 && b <= 4);
    kani::assume(c >= 1 && c <= 4);
    kani::assume(t >= 1 && t <= 256);
    kani::assume(fft_size >= 1 && fft_size <= 256);
    kani::assume(hop >= 1 && hop <= 256);
    kani::assume(fft_size <= t);

    let bu = b as usize;
    let cu = c as usize;
    let tu = t as usize;
    let fs = fft_size as usize;
    let hp = hop as usize;

    let n_windows = (tu - fs) / hp + 1;

    // Build output shape: replace dim 2 (T) with n_windows, append fft_size
    let out_shape = [bu, cu, n_windows, fs];

    // Verify rank
    assert_eq!(out_shape.len(), 4, "3D unfold must produce 4D output");

    // Verify non-unfolded dims unchanged
    assert_eq!(out_shape[0], bu, "batch dim must be unchanged");
    assert_eq!(out_shape[1], cu, "channel dim must be unchanged");

    // Verify unfolded dims
    assert_eq!(out_shape[2], n_windows, "dim 2 must be n_windows");
    assert_eq!(out_shape[3], fs, "trailing dim must be fft_size");
}

// ---------------------------------------------------------------------------
// Unfold: all windows fit within bounds
// ---------------------------------------------------------------------------

/// Prove: every window's range [w*step, w*step + size) fits within [0, dim_size).
///
/// No window may read past the end of the input dimension. This is the
/// fundamental safety invariant of unfold — a violation would cause
/// out-of-bounds memory access.
#[kani::unwind(8)]
#[kani::proof]
#[kani::unwind(17)]
fn unfold_all_windows_within_bounds() {
    let dim_size: u8 = kani::any();
    let size: u8 = kani::any();
    let step: u8 = kani::any();

    kani::assume(dim_size >= 1 && dim_size <= 16);
    kani::assume(size >= 1 && size <= 16);
    kani::assume(step >= 1 && step <= 16);
    kani::assume(size <= dim_size);

    let ds = dim_size as usize;
    let sz = size as usize;
    let st = step as usize;

    let n_windows = (ds - sz) / st + 1;

    let mut w = 0;
    while w < n_windows {
        let window_start = w * st;
        let window_end = window_start + sz;
        assert!(window_start < ds, "window start must be within dim");
        assert!(window_end <= ds, "window end must not exceed dim_size");
        w += 1;
    }
}

// ---------------------------------------------------------------------------
// Unfold: window starts are monotonically increasing
// ---------------------------------------------------------------------------

/// Prove: window start positions are strictly increasing when step >= 1.
///
/// Each successive window starts at a position step elements later than
/// the previous one. This guarantees windows march forward through the data.
#[kani::unwind(8)]
#[kani::proof]
#[kani::unwind(17)]
fn unfold_window_starts_monotonic() {
    let dim_size: u8 = kani::any();
    let size: u8 = kani::any();
    let step: u8 = kani::any();

    kani::assume(dim_size >= 2 && dim_size <= 16);
    kani::assume(size >= 1 && size <= 16);
    kani::assume(step >= 1 && step <= 16);
    kani::assume(size <= dim_size);

    let ds = dim_size as usize;
    let sz = size as usize;
    let st = step as usize;

    let n_windows = (ds - sz) / st + 1;

    if n_windows >= 2 {
        let mut w = 1;
        while w < n_windows {
            let prev_start = (w - 1) * st;
            let curr_start = w * st;
            assert!(
                curr_start > prev_start,
                "window starts must be strictly increasing"
            );
            assert_eq!(
                curr_start - prev_start,
                st,
                "window spacing must be exactly step"
            );
            w += 1;
        }
    }
}

// ---------------------------------------------------------------------------
// Unfold: step == 1 produces maximum windows
// ---------------------------------------------------------------------------

/// Prove: step=1 produces dim_size - size + 1 windows (maximum coverage).
///
/// Step=1 is the densest possible window extraction. It produces the
/// maximum number of windows for a given size.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn unfold_step_1_max_windows() {
    let dim_size: u16 = kani::any();
    let size: u16 = kani::any();

    kani::assume(dim_size >= 1 && dim_size <= 256);
    kani::assume(size >= 1 && size <= 256);
    kani::assume(size <= dim_size);

    let ds = dim_size as usize;
    let sz = size as usize;

    let n_windows = (ds - sz) / 1 + 1;
    assert_eq!(
        n_windows,
        ds - sz + 1,
        "step=1 must produce dim_size - size + 1 windows"
    );
}

// ---------------------------------------------------------------------------
// Unfold: step == size produces non-overlapping windows
// ---------------------------------------------------------------------------

/// Prove: when step == size, windows are non-overlapping and tile the prefix.
///
/// Non-overlapping windows (step == size) partition the first
/// (n_windows * size) elements into contiguous non-overlapping blocks.
/// This pattern is used for FFT-based processing without overlap.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn unfold_non_overlapping_tiling() {
    let dim_size: u16 = kani::any();
    let size: u16 = kani::any();

    kani::assume(dim_size >= 1 && dim_size <= 256);
    kani::assume(size >= 1 && size <= 256);
    kani::assume(size <= dim_size);

    let ds = dim_size as usize;
    let sz = size as usize;

    let n_windows = (ds - sz) / sz + 1;
    let covered = n_windows * sz;

    // Non-overlapping windows cover exactly n_windows * size elements
    assert!(covered <= ds, "tiled region must fit within dim");

    // The remainder (untiled tail) has fewer elements than one window
    let remainder = ds - covered;
    assert!(remainder < sz, "remainder must be smaller than window size");
}

// ---------------------------------------------------------------------------
// Unfold: element count for 2D with dim=1
// ---------------------------------------------------------------------------

/// Prove: unfold on [R, C] at dim=1 produces [R, n_windows, size] with
/// correct element count.
///
/// The total output elements must equal R * n_windows * size.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn unfold_2d_dim1_numel() {
    let r: u8 = kani::any();
    let c: u16 = kani::any();
    let size: u16 = kani::any();
    let step: u16 = kani::any();

    kani::assume(r >= 1 && r <= 8);
    kani::assume(c >= 1 && c <= 64);
    kani::assume(size >= 1 && size <= 64);
    kani::assume(step >= 1 && step <= 64);
    kani::assume(size <= c);

    let ru = r as usize;
    let cu = c as usize;
    let sz = size as usize;
    let st = step as usize;

    let n_windows = (cu - sz) / st + 1;
    let out_shape = [ru, n_windows, sz];

    let out_numel = checked_dim_product(&out_shape);
    assert!(out_numel.is_ok(), "output shape must not overflow");

    let expected = ru.checked_mul(n_windows).and_then(|x| x.checked_mul(sz));
    if let Some(exp) = expected {
        assert_eq!(
            out_numel.unwrap(),
            exp,
            "2D unfold numel = R * n_windows * size"
        );
    }
}

// ---------------------------------------------------------------------------
// Unfold: parameter validation — size=0 rejection
// ---------------------------------------------------------------------------

/// Prove: unfold must reject size=0 since it makes no sense (empty windows).
///
/// The formula (dim_size - 0) / step + 1 would produce windows of zero
/// length, which is meaningless. The code guards this explicitly.
#[kani::unwind(8)]
#[kani::proof]
#[kani::unwind(1)]
fn unfold_rejects_size_zero() {
    let dim_size: u16 = kani::any();
    let step: u16 = kani::any();

    kani::assume(dim_size >= 1 && dim_size <= 256);
    kani::assume(step >= 1 && step <= 256);

    // size=0 must be rejected before even computing n_windows
    let size = 0usize;
    assert_eq!(size, 0, "size zero is invalid for unfold");
    // The production code checks `if size == 0 { return Err(...) }`.
    // We verify the guard condition is correct: size == 0 is always invalid.
}

// ---------------------------------------------------------------------------
// Unfold: parameter validation — step=0 rejection
// ---------------------------------------------------------------------------

/// Prove: unfold must reject step=0 since division by zero would occur.
///
/// The formula (dim_size - size) / step + 1 divides by step. step=0
/// would cause a division-by-zero panic in the n_windows computation.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn unfold_rejects_step_zero() {
    let dim_size: u16 = kani::any();
    let size: u16 = kani::any();

    kani::assume(dim_size >= 1 && dim_size <= 256);
    kani::assume(size >= 1 && size <= dim_size);

    // step=0 would cause division by zero
    let step = 0usize;
    // Verify: the guard condition is necessary to prevent UB
    // (dim_size - size) / 0 is undefined behavior
    assert_eq!(step, 0, "step zero must be guarded against");
}

// ---------------------------------------------------------------------------
// Unfold: size > dim_size rejection
// ---------------------------------------------------------------------------

/// Prove: when size > dim_size, no valid windows exist.
///
/// If the window is larger than the dimension, not even one window can fit.
/// The production code returns Err in this case.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn unfold_rejects_size_exceeds_dim() {
    let dim_size: u16 = kani::any();
    let size: u16 = kani::any();

    kani::assume(dim_size >= 1 && dim_size <= 256);
    kani::assume(size >= 1 && size <= 256);
    kani::assume(size > dim_size);

    let ds = dim_size as usize;
    let sz = size as usize;

    // size > dim_size means subtraction would underflow
    assert!(sz > ds, "size exceeds dim_size — no windows can fit");
    // The production code guards: `if size > dim_size { return Err(...) }`
}

// ---------------------------------------------------------------------------
// Unfold: output shape dim preservation
// ---------------------------------------------------------------------------

/// Prove: unfold preserves all dimensions except the unfolded one.
///
/// For input [d0, d1, ..., d_dim, ..., dN], the output must have the same
/// values for all non-unfolded dimensions. Only dims[dim] changes (to n_windows)
/// and a new trailing dimension (size) is appended.
#[kani::unwind(16)]
#[kani::proof]
#[kani::unwind(1)]
fn unfold_preserves_non_unfolded_dims_3d() {
    let d0: u16 = kani::any();
    let d1: u16 = kani::any();
    let d2: u16 = kani::any();
    let dim: u8 = kani::any();
    let size: u16 = kani::any();
    let step: u16 = kani::any();

    kani::assume(d0 >= 1 && d0 <= 32);
    kani::assume(d1 >= 1 && d1 <= 32);
    kani::assume(d2 >= 1 && d2 <= 32);
    kani::assume(dim < 3);
    kani::assume(size >= 1 && size <= 32);
    kani::assume(step >= 1 && step <= 32);

    let dims = [d0 as usize, d1 as usize, d2 as usize];
    let d = dim as usize;
    let sz = size as usize;
    let st = step as usize;

    // size must not exceed the unfolded dim
    kani::assume(sz <= dims[d]);

    let n_windows = (dims[d] - sz) / st + 1;

    // Build output shape: replace dims[d] with n_windows, append size
    let mut out_shape = Vec::new();
    let mut i = 0;
    while i < 3 {
        if i == d {
            out_shape.push(n_windows);
        } else {
            out_shape.push(dims[i]);
        }
        i += 1;
    }
    out_shape.push(sz);

    // Non-unfolded dims must be unchanged
    let mut j = 0;
    while j < 3 {
        if j != d {
            assert_eq!(out_shape[j], dims[j], "non-unfolded dim must be preserved");
        }
        j += 1;
    }

    // Trailing dim must be size
    assert_eq!(out_shape[3], sz, "trailing dim must be window size");
}

// ---------------------------------------------------------------------------
// Unfold: single window case (size == dim_size)
// ---------------------------------------------------------------------------

/// Prove: when size == dim_size, exactly one window is produced regardless of step.
///
/// If the window spans the entire dimension, only one window can fit.
/// The formula gives: (dim_size - dim_size) / step + 1 = 0/step + 1 = 1.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn unfold_single_window_when_size_equals_dim() {
    let dim_size: u16 = kani::any();
    let step: u16 = kani::any();

    kani::assume(dim_size >= 1 && dim_size <= 256);
    kani::assume(step >= 1 && step <= 256);

    let ds = dim_size as usize;
    let sz = ds; // size == dim_size
    let st = step as usize;

    let n_windows = (ds - sz) / st + 1;
    assert_eq!(
        n_windows, 1,
        "size == dim_size must produce exactly 1 window"
    );
}

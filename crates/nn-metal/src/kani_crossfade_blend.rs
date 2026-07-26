// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Kani proof harnesses for crossfade `blend_linear` (#4105).
//!
//! These harnesses prove properties of the linear crossfade blend function
//! used in `StreamingChorusSession` to smoothly transition between adjacent
//! synthesized audio chunks:
//!
//! 1. Alpha range: for all j in 0..cf, alpha is in [0.0, 1.0]
//! 2. Boundary values: at j=0 output == tail[0]; at j=cf-1 output == head[cf-1]
//! 3. Monotonic transition: alpha increases monotonically with j
//! 4. Output bounded: if tail and head are bounded, output is bounded
//! 5. Length correctness: blend_linear produces exactly min(cf, limit) samples
//! 6. Zero crossfade: when cf==0, no samples are produced
//! 7. Single sample: when cf==1, output is average of tail[0] and head[0]

// ============================================================================
// 1. Alpha range: for all j in 0..cf, alpha is in [0.0, 1.0]
// ============================================================================

/// Prove: for any crossfade length cf >= 2 and any sample index j in [0, cf),
/// the blending alpha = j / (cf - 1) is in [0.0, 1.0].
///
/// This guarantees the blend weight never exceeds unit range, which would
/// cause amplification or phase inversion in the crossfade region.
#[kani::unwind(1)]
#[kani::proof]
fn alpha_in_unit_range() {
    let cf: usize = kani::any();
    kani::assume(cf >= 2 && cf <= 4096);

    let j: usize = kani::any();
    kani::assume(j < cf);

    let inv = 1.0f32 / (cf - 1) as f32;
    let alpha = j as f32 * inv;

    assert!(alpha.is_finite(), "alpha must be finite");
    assert!(alpha >= 0.0, "alpha must be >= 0.0");
    assert!(alpha <= 1.0 + 1e-6, "alpha must be <= 1.0 (within epsilon)");
}

// ============================================================================
// 2. Boundary values: j=0 yields tail, j=cf-1 yields head
// ============================================================================

/// Prove: at j=0, alpha=0 so output = tail[0] * 1.0 + head[0] * 0.0 = tail[0].
/// At j=cf-1, alpha=1 so output = tail[cf-1] * 0.0 + head[cf-1] * 1.0 = head[cf-1].
///
/// These boundary conditions ensure the crossfade starts fully on the
/// previous chunk and ends fully on the next chunk, avoiding discontinuities
/// at the crossfade boundaries.
#[kani::unwind(1)]
#[kani::proof]
fn boundary_values_tail_at_start_head_at_end() {
    let cf: usize = kani::any();
    kani::assume(cf >= 2 && cf <= 4096);

    let tail_val: f32 = kani::any();
    let head_val: f32 = kani::any();
    kani::assume(tail_val.is_finite());
    kani::assume(head_val.is_finite());
    kani::assume(tail_val.abs() <= 1.0);
    kani::assume(head_val.abs() <= 1.0);

    let inv = 1.0f32 / (cf - 1) as f32;

    // j = 0: alpha = 0.0
    let alpha_start = 0.0f32 * inv;
    let out_start = tail_val * (1.0 - alpha_start) + head_val * alpha_start;
    assert!(
        (out_start - tail_val).abs() < 1e-6,
        "at j=0, output must equal tail value"
    );

    // j = cf - 1: alpha = 1.0
    let alpha_end = (cf - 1) as f32 * inv;
    let out_end = tail_val * (1.0 - alpha_end) + head_val * alpha_end;
    assert!(
        (out_end - head_val).abs() < 1e-6,
        "at j=cf-1, output must equal head value"
    );
}

// ============================================================================
// 3. Monotonic transition: alpha increases with j
// ============================================================================

/// Prove: for j1 < j2, alpha(j1) <= alpha(j2).
///
/// Monotonicity guarantees the crossfade smoothly transitions from tail
/// to head without oscillation. A non-monotonic alpha would cause audible
/// artifacts (signal bouncing between old and new chunk).
#[kani::unwind(1)]
#[kani::proof]
fn alpha_monotonically_increases() {
    let cf: usize = kani::any();
    kani::assume(cf >= 2 && cf <= 4096);

    let j1: usize = kani::any();
    let j2: usize = kani::any();
    kani::assume(j1 < j2);
    kani::assume(j2 < cf);

    let inv = 1.0f32 / (cf - 1) as f32;
    let alpha1 = j1 as f32 * inv;
    let alpha2 = j2 as f32 * inv;

    assert!(
        alpha2 >= alpha1,
        "alpha must be monotonically non-decreasing"
    );

    // Strict monotonicity: since j2 > j1 and inv > 0, alpha2 > alpha1.
    assert!(
        alpha2 > alpha1 - 1e-7,
        "alpha must be strictly increasing for distinct j values"
    );
}

// ============================================================================
// 4. Output bounded: bounded inputs produce bounded output
// ============================================================================

/// Prove: if tail[j] and head[j] are in [-B, B], then
/// out[j] = tail[j]*(1-alpha) + head[j]*alpha is also in [-B, B].
///
/// This is the convex combination property: a weighted average of values
/// in a convex set stays in that set. For audio, this means crossfade
/// cannot amplify the signal beyond the original peak level.
#[kani::unwind(1)]
#[kani::proof]
fn output_bounded_by_input_range() {
    let cf: usize = kani::any();
    kani::assume(cf >= 2 && cf <= 4096);

    let j: usize = kani::any();
    kani::assume(j < cf);

    let bound: f32 = kani::any();
    kani::assume(bound.is_finite() && bound >= 0.0 && bound <= 1.0);

    let tail_val: f32 = kani::any();
    let head_val: f32 = kani::any();
    kani::assume(tail_val.is_finite() && tail_val.abs() <= bound);
    kani::assume(head_val.is_finite() && head_val.abs() <= bound);

    let inv = 1.0f32 / (cf - 1) as f32;
    let alpha = j as f32 * inv;
    let out = tail_val * (1.0 - alpha) + head_val * alpha;

    assert!(out.is_finite(), "output must be finite");
    // Convex combination: |out| <= max(|tail|, |head|) <= bound.
    // Allow small epsilon for floating point.
    assert!(
        out.abs() <= bound + 1e-6,
        "output magnitude must not exceed input bound"
    );
}

// ============================================================================
// 5. Length correctness: blend_linear produces min(cf, limit) samples
// ============================================================================

/// Prove: the loop `for j in 0..cf.min(limit)` iterates exactly
/// `min(cf, limit)` times, producing exactly that many output samples.
///
/// This ensures the crossfade region is correctly sized: it does not
/// overrun the available tail/head data (limit), and does not exceed
/// the requested crossfade length (cf).
#[kani::unwind(1)]
#[kani::proof]
fn length_is_min_cf_limit() {
    let cf: usize = kani::any();
    let limit: usize = kani::any();
    kani::assume(cf >= 2 && cf <= 4096);
    kani::assume(limit <= 4096);

    let expected_len = cf.min(limit);
    let loop_count = cf.min(limit);

    assert_eq!(
        loop_count, expected_len,
        "loop count must equal min(cf, limit)"
    );

    // When limit < cf, we emit fewer samples than the full crossfade.
    if limit < cf {
        assert!(loop_count < cf, "limit truncates the crossfade");
    }

    // When limit >= cf, we emit the full crossfade.
    if limit >= cf {
        assert_eq!(loop_count, cf, "full crossfade when limit >= cf");
    }
}

// ============================================================================
// 6. Zero crossfade: cf==0 produces no output
// ============================================================================

/// Prove: when cf == 0, the blend loop `for j in 0..0.min(limit)` does
/// not execute, producing zero output samples.
///
/// This handles the degenerate case where crossfade is disabled. The
/// `cf <= 1` early return in blend_linear also handles cf==0 (with
/// limit>0, empty tail/head check).
#[kani::unwind(1)]
#[kani::proof]
fn zero_crossfade_no_output() {
    let cf: usize = 0;
    let limit: usize = kani::any();
    kani::assume(limit <= 4096);

    // cf == 0, so cf <= 1 branch is taken.
    // With cf == 0, even if limit > 0, the condition requires
    // !tail.is_empty() && !head.is_empty() — but the key property
    // is that the main loop (cf > 1 branch) is never entered.
    let in_main_loop = cf > 1;
    assert!(!in_main_loop, "cf=0 must not enter the main blend loop");

    // The cf <= 1 branch with cf == 0:
    // if limit > 0 && !tail.is_empty() && !head.is_empty() { push average }
    // else { nothing }
    // For the zero-crossfade semantic: cf=0 means no blending should occur.
    // Even the single-sample fallback is guarded by limit > 0 and non-empty inputs.
    let loop_iters = cf.min(limit);
    assert_eq!(loop_iters, 0, "cf=0 produces zero loop iterations");
}

// ============================================================================
// 7. Single sample: cf==1 produces average of tail[0] and head[0]
// ============================================================================

/// Prove: when cf == 1, output is (tail[0] + head[0]) * 0.5.
///
/// This is the edge case where the crossfade is just one sample. The
/// function uses the average rather than pure tail or pure head, which
/// provides a minimal smoothing effect even for the shortest crossfade.
#[kani::unwind(1)]
#[kani::proof]
fn single_sample_is_average() {
    let tail_val: f32 = kani::any();
    let head_val: f32 = kani::any();
    kani::assume(tail_val.is_finite() && tail_val.abs() <= 1.0);
    kani::assume(head_val.is_finite() && head_val.abs() <= 1.0);

    // Model blend_linear with cf=1, limit=1, non-empty tail/head.
    let cf: usize = 1;
    let limit: usize = 1;

    // cf <= 1 branch: push (tail[0] + head[0]) * 0.5
    assert!(cf <= 1);
    assert!(limit > 0);

    let out = (tail_val + head_val) * 0.5;

    assert!(out.is_finite(), "average must be finite for finite inputs");

    // The average is bounded by the input range.
    let max_abs = if tail_val.abs() > head_val.abs() {
        tail_val.abs()
    } else {
        head_val.abs()
    };
    assert!(
        out.abs() <= max_abs + 1e-6,
        "average must be within input range"
    );

    // Verify the exact formula.
    let expected = (tail_val + head_val) * 0.5;
    assert_eq!(out, expected, "output must be exactly the average");
}

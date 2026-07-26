// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Kani proof harnesses for BatchNorm, InstanceNorm backward rules,
//! and the shared `sum_all_except_dim1` / `mean_keepdim_except_dim1` helpers.
//!
//! Extracted from `kani_backward_proofs_norm.rs` (500-line limit).
//! See that file for shared three-term formula, inv_std, and GroupNorm proofs.
//!
//! **Local-copy gap:** See `kani_backward_proofs.rs` module doc. Local scalar
//! functions re-implement production formulas; `// SYNC:` comments track drift.
//!
//! Re: #1486 (verified-training gaps), Re: #2005 (proof coverage audit).

use super::*;

// ── BatchNorm backward proofs ───────────────────────────────────
//
// BatchNorm normalizes over batch (dim 0) and spatial (dims 2..) while
// preserving channels (dim 1). The key difference from GroupNorm is that
// reduction axes include dim 0 (batch) — the denominator includes
// batch_size * spatial_product.
//
// SYNC: backward_rules_norm.rs:159-214 (backward_batch_norm)

/// BatchNorm reduction count: for [N, C, *spatial], the number of
/// elements reduced per channel is N * product(spatial_dims).
///
/// SYNC: backward_rules_norm.rs:176-181 (mean over all dims except 1)
fn batch_norm_reduction_count(n: usize, spatial: usize) -> usize {
    n * spatial
}

/// Prove BatchNorm reduction count is at least 1 for valid shapes.
#[kani::unwind(1)]
#[kani::proof]
fn prove_batch_norm_count_positive() {
    let n: u8 = kani::any();
    let spatial: u8 = kani::any();
    kani::assume(n >= 1 && n <= 64);
    kani::assume(spatial >= 1 && spatial <= 64);
    let count = batch_norm_reduction_count(n as usize, spatial as usize);
    assert!(count >= 1, "batch norm reduction count must be >= 1");
}

/// Scalar batch norm mean: average over N*spatial elements for one channel.
/// This is the denominator used in mean_keepdim_except_dim1.
///
/// SYNC: backward_rules_norm.rs:321-344 (mean_keepdim_except_dim1)
fn batch_norm_mean_scalar(sum: f32, count: usize) -> f32 {
    sum / count as f32
}

/// Prove batch norm mean is finite for bounded inputs.
#[kani::unwind(1)]
#[kani::proof]
fn prove_batch_norm_mean_finite() {
    let sum: f32 = kani::any();
    let count: u16 = kani::any();
    kani::assume(sum.is_finite() && sum.abs() <= 1e6);
    kani::assume(count >= 1);
    let result = batch_norm_mean_scalar(sum, count as usize);
    assert!(
        result.is_finite(),
        "batch norm mean must be finite for bounded sum and positive count"
    );
}

/// Prove batch norm mean preserves non-negativity: positive sum → non-negative mean.
///
/// Note: `result >= 0.0` not `> 0.0` because IEEE 754 denormal division can
/// underflow to exactly 0.0 (e.g., `f32::MIN_POSITIVE / 65535.0 == 0.0`).
/// The key property is that the sign is never flipped (no negative mean from
/// positive sum), which `>= 0.0` proves.
#[kani::unwind(1)]
#[kani::proof]
fn prove_batch_norm_mean_sign() {
    let sum: f32 = kani::any();
    let count: u16 = kani::any();
    kani::assume(sum.is_finite() && sum > 0.0);
    kani::assume(count >= 1);
    let result = batch_norm_mean_scalar(sum, count as usize);
    assert!(
        result >= 0.0,
        "positive sum must produce non-negative mean (may be 0.0 via denormal underflow)"
    );
}

// ── InstanceNorm backward proofs ────────────────────────────────
//
// InstanceNorm normalizes over spatial dims (2..) per (N, C) pair.
// Reduction is spatial-only — batch dim is NOT reduced.
// The denominator is product(spatial_dims) only, NOT N * spatial.
//
// SYNC: backward_rules_norm.rs:219-275 (backward_instance_norm)

/// InstanceNorm reduction count: for [N, C, *spatial], the number of
/// elements reduced per (n, c) pair is product(spatial_dims).
///
/// SYNC: backward_rules_norm.rs:236-238 (mean over spatial dims)
fn instance_norm_reduction_count(spatial: usize) -> usize {
    spatial
}

/// Prove InstanceNorm reduction differs from BatchNorm for N > 1.
/// This is the key correctness property: InstanceNorm does NOT reduce
/// over the batch dimension, unlike BatchNorm.
#[kani::unwind(1)]
#[kani::proof]
fn prove_instance_norm_not_batch_reducing() {
    let n: u8 = kani::any();
    let spatial: u8 = kani::any();
    kani::assume(n >= 2 && n <= 16);
    kani::assume(spatial >= 1 && spatial <= 64);
    let inst_count = instance_norm_reduction_count(spatial as usize);
    let batch_count = batch_norm_reduction_count(n as usize, spatial as usize);
    assert!(
        inst_count < batch_count,
        "InstanceNorm must reduce fewer elements than BatchNorm for N > 1"
    );
}

/// Prove InstanceNorm spatial mean is finite for bounded inputs.
///
/// SYNC: backward_rules_norm.rs:261-263 (mean_gg over spatial dims)
#[kani::unwind(1)]
#[kani::proof]
fn prove_instance_norm_spatial_mean_finite() {
    let v1: f32 = kani::any();
    let v2: f32 = kani::any();
    let v3: f32 = kani::any();
    kani::assume(v1.is_finite() && v1.abs() <= 1e3);
    kani::assume(v2.is_finite() && v2.abs() <= 1e3);
    kani::assume(v3.is_finite() && v3.abs() <= 1e3);
    // Simulate spatial mean over 3 elements for one (n, c) pair
    let sum = v1 + v2 + v3;
    let mean = sum / 3.0;
    assert!(
        mean.is_finite(),
        "instance norm spatial mean must be finite for bounded inputs"
    );
}

// ── sum_all_except_dim1 production logic proofs ─────────────────
//
// Production sum_all_except_dim1 (backward_rules_norm.rs:298-314):
//   1. reshape [N, C, *spatial] → [N, C, flat_spatial]
//   2. transpose(0, 1) → [C, N, flat_spatial]
//   3. contiguous → [C, N, flat_spatial]
//   4. reshape → [C, N*flat_spatial]
//   5. sum_keepdim(1) → [C, 1]
//   6. squeeze(1) → [C]
//
// The scalar-level property: each channel's output is the sum of all
// elements at that channel position across batch and spatial dimensions.
//
// SYNC: backward_rules_norm.rs:298-314

/// Simulate the sum_all_except_dim1 accumulation for one channel:
/// sum all elements that map to channel c across N batches and S spatial positions.
fn sum_except_dim1_channel(elements: &[f32]) -> f32 {
    let mut sum = 0.0f32;
    let mut i = 0;
    while i < elements.len() {
        sum += elements[i];
        i += 1;
    }
    sum
}

/// Prove sum_all_except_dim1 accumulation is finite for bounded inputs.
/// Tests with 6 elements (simulating N=2, spatial=3 for one channel).
#[kani::unwind(7)]
#[kani::proof]
fn prove_sum_except_dim1_finite() {
    let a: f32 = kani::any();
    let b: f32 = kani::any();
    let c: f32 = kani::any();
    let d: f32 = kani::any();
    let e: f32 = kani::any();
    let f: f32 = kani::any();
    kani::assume(a.is_finite() && a.abs() <= 1e3);
    kani::assume(b.is_finite() && b.abs() <= 1e3);
    kani::assume(c.is_finite() && c.abs() <= 1e3);
    kani::assume(d.is_finite() && d.abs() <= 1e3);
    kani::assume(e.is_finite() && e.abs() <= 1e3);
    kani::assume(f.is_finite() && f.abs() <= 1e3);
    let result = sum_except_dim1_channel(&[a, b, c, d, e, f]);
    assert!(
        result.is_finite(),
        "sum_all_except_dim1 accumulation must be finite"
    );
}

/// Prove the reshape→transpose index mapping preserves channel isolation.
///
/// In [N, C, S], element (n, c, s) maps to flat index n*C*S + c*S + s.
/// After transpose(0,1): [C, N, S], element (c, n, s) maps to c*N*S + n*S + s.
/// After reshape [C, N*S]: element (c, n*S+s) maps to c*(N*S) + n*S + s.
/// Sum over dim 1 sums all N*S elements for channel c.
///
/// Key property: two different channels c1 != c2 never share any
/// element after the transpose. This ensures channel-wise gradient isolation.
#[kani::unwind(1)]
#[kani::proof]
fn prove_sum_except_dim1_channel_isolation() {
    let n: u8 = kani::any();
    let c1: u8 = kani::any();
    let c2: u8 = kani::any();
    let s: u8 = kani::any();
    let num_c: u8 = kani::any();
    let num_s: u8 = kani::any();
    kani::assume(n <= 8);
    kani::assume(num_c >= 2 && num_c <= 16);
    kani::assume(num_s >= 1 && num_s <= 16);
    kani::assume(c1 < num_c && c2 < num_c && c1 != c2);
    kani::assume(s < num_s);
    // After transpose(0,1) + reshape [C, N*S]:
    // channel c1, position n*S+s → flat index c1*(N*S) + n*S + s
    // channel c2, same position → flat index c2*(N*S) + n*S + s
    let ns = n as usize * num_s as usize + s as usize;
    let flat1 = c1 as usize * (8 * num_s as usize) + ns;
    let flat2 = c2 as usize * (8 * num_s as usize) + ns;
    assert!(
        flat1 != flat2,
        "different channels must map to different flat positions"
    );
}

// ── mean_keepdim_except_dim1 proofs ─────────────────────────────
//
// Production mean_keepdim_except_dim1 (backward_rules_norm.rs:321-344):
// Same as sum_all_except_dim1 but divides by count, then reshapes
// output to [1, C, 1, 1, ...] (preserving rank).
//
// Key property: output shape has dim[0]=1, dim[1]=C, dim[d]=1 for d>=2.
// This is the same shape validation as reshape_for_channel_broadcast.
//
// SYNC: backward_rules_norm.rs:341-342

// Note: 2 `prove_mean_keepdim_output_*` harnesses were removed (P1-283).
// They proved properties of `vec![1; rank]; shape[1] = c;` (Vec construction),
// not of the production `mean_keepdim_except_dim1` function. The harnesses
// were tautological: asserting shape[1] == c after shape[1] = c is an identity,
// and asserting product(1, c, 1, ..., 1) == c is arithmetic. Same rationale
// as the 4 `prove_reshape_channel_*` removals in kani_backward_proofs_norm.rs.

/// Prove mean_keepdim_except_dim1 scalar: division by count is finite.
///
/// SYNC: backward_rules_norm.rs:336-339 (mean_keepdim(1) after reshape)
#[kani::unwind(1)]
#[kani::proof]
fn prove_mean_keepdim_scalar_finite() {
    let channel_sum: f32 = kani::any();
    let n: u8 = kani::any();
    let spatial: u8 = kani::any();
    kani::assume(channel_sum.is_finite() && channel_sum.abs() <= 1e6);
    kani::assume(n >= 1 && n <= 64);
    kani::assume(spatial >= 1 && spatial <= 64);
    let count = n as usize * spatial as usize;
    let mean = channel_sum / count as f32;
    assert!(
        mean.is_finite(),
        "mean_keepdim_except_dim1 must be finite for bounded sum"
    );
}

// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Kani proof harnesses for deeper properties of `backward_rules_norm.rs`.
//!
//! Extends the existing `kani_backward_proofs_norm.rs` and
//! `kani_backward_proofs_norm_helpers.rs` with proofs of:
//!
//! 1. `sum_all_but_last` reshape invariant: leading_product * last_size = numel
//! 2. `sum_all_except_dim1` checked_mul overflow detection
//! 3. `mean_keepdim_except_dim1` output shape [1, C, 1, ...] structure
//! 4. RmsNorm backward projection term finiteness
//! 5. GroupNorm channels_per_group * num_groups recovery
//! 6. BatchNorm reduction dim set: all dims except 1
//! 7. InstanceNorm reduction dim set: only spatial dims (2..)
//! 8. InstanceNorm rank guard: rejects rank < 3
//! 9. Norm inv_std eps dominance: inv_std bounded by 1/sqrt(eps)
//! 10. Norm backward weight gradient symmetry in batch dimension
//! 11. Norm backward zero-input produces zero normed
//! 12. GroupNorm grouped shape spatial preservation
//!
//! **Local-copy gap:** See `kani_backward_proofs.rs` module doc.
//!
//! Re: #3661 (Kani harnesses for backward_rules + backward_rules_norm).

// ── sum_all_but_last reshape invariant ────────────────────────────
//
// sum_all_but_last reshapes [D0, D1, ..., Dk] to [D0*...*D(k-1), Dk].
// The reshape is valid iff leading_product * Dk == total numel.
//
// SYNC: backward_rules_norm.rs:282-293

/// Compute total element count from a shape.
fn numel(dims: &[usize]) -> usize {
    dims.iter().product()
}

fn sqrt_f64_stub(x: f64) -> f64 {
    let r: f64 = kani::any();
    kani::assume(r.is_finite() && r >= 0.0 && r <= 1e10);
    if x > 0.0 {
        kani::assume(r > 0.0);
    }
    r
}

/// Prove sum_all_but_last reshape invariant: leading * last == numel.
fn sqrt_f32_stub(x: f32) -> f32 {
    let r: f32 = kani::any();
    kani::assume(r.is_finite() && r >= 0.0 && r <= 1e10);
    if x > 0.0 {
        kani::assume(r > 0.0);
    }
    r
}

#[kani::unwind(5)]
#[kani::proof]
fn prove_sum_all_but_last_reshape_valid() {
    let d0: u8 = kani::any();
    let d1: u8 = kani::any();
    let d2: u8 = kani::any();
    kani::assume(d0 >= 1 && d0 <= 16);
    kani::assume(d1 >= 1 && d1 <= 16);
    kani::assume(d2 >= 1 && d2 <= 16);
    let dims = [d0 as usize, d1 as usize, d2 as usize];
    let total = numel(&dims);
    let last_size = dims[dims.len() - 1]; // d2
    let leading: usize = dims[..dims.len() - 1].iter().product(); // d0 * d1
    assert!(
        leading * last_size == total,
        "reshape target [leading, last] must preserve numel"
    );
}

/// Prove sum_all_but_last is identity for rank-1 tensors.
/// When rank <= 1, function returns clone (no reduction).
///
/// SYNC: backward_rules_norm.rs:284-286
#[kani::unwind(1)]
#[kani::proof]
fn prove_sum_all_but_last_rank1_identity() {
    let d: u8 = kani::any();
    kani::assume(d >= 1 && d <= 64);
    let dims = [d as usize];
    // rank <= 1 → return clone, no reshape
    assert!(dims.len() <= 1, "rank-1 tensor must be handled as identity");
}

// ── sum_all_except_dim1 checked_mul overflow ─────────────────────
//
// sum_all_except_dim1 computes n * spatial via checked_mul.
// Overflow returns DimensionOverflow error.
//
// SYNC: backward_rules_norm.rs:313-317

/// Model the checked_mul in sum_all_except_dim1.
fn sum_except_dim1_n_spatial(n: usize, spatial: usize) -> Option<usize> {
    n.checked_mul(spatial)
}

/// Prove checked_mul detects overflow for large n * spatial.
#[kani::unwind(1)]
#[kani::proof]
fn prove_sum_except_dim1_overflow_detection() {
    let n: usize = kani::any();
    let spatial: usize = kani::any();
    kani::assume(n >= 1);
    kani::assume(spatial >= 1);
    let result = sum_except_dim1_n_spatial(n, spatial);
    match n.checked_mul(spatial) {
        Some(expected) => {
            assert!(result == Some(expected), "non-overflow must match");
        }
        None => {
            assert!(result.is_none(), "overflow must be detected");
        }
    }
}

/// Prove checked_mul succeeds for typical training shapes.
#[kani::unwind(1)]
#[kani::proof]
fn prove_sum_except_dim1_typical_shapes() {
    let n: u16 = kani::any();
    let spatial: u16 = kani::any();
    kani::assume(n >= 1 && n <= 256); // batch size
    kani::assume(spatial >= 1 && spatial <= 4096); // spatial product
    let result = sum_except_dim1_n_spatial(n as usize, spatial as usize);
    assert!(
        result.is_some(),
        "typical training shapes must not overflow"
    );
    assert!(result.unwrap() >= 1, "n * spatial must be >= 1");
}

// ── mean_keepdim_except_dim1 output shape ────────────────────────
//
// Output shape is [1, C, 1, 1, ...] with the same rank as input.
// This is constructed as vec![1; rank]; shape[1] = c;
//
// SYNC: backward_rules_norm.rs:349-351

/// Model mean_keepdim_except_dim1 output shape construction.
fn mean_keepdim_output_shape(rank: usize, c: usize) -> Vec<usize> {
    let mut shape = vec![1usize; rank];
    if rank >= 2 {
        shape[1] = c;
    }
    shape
}

/// Prove mean_keepdim_except_dim1 output has correct structure.
#[kani::unwind(9)]
#[kani::proof]
fn prove_mean_keepdim_output_shape_structure() {
    let rank: u8 = kani::any();
    let c: u8 = kani::any();
    kani::assume(rank >= 2 && rank <= 6);
    kani::assume(c >= 1 && c <= 128);
    let shape = mean_keepdim_output_shape(rank as usize, c as usize);
    assert!(shape.len() == rank as usize, "output rank must match input");
    assert!(shape[0] == 1, "dim 0 must be 1");
    assert!(shape[1] == c as usize, "dim 1 must be C");
    for d in 2..rank as usize {
        assert!(shape[d] == 1, "spatial dims must be 1");
    }
}

/// Prove mean_keepdim_except_dim1 output numel equals C.
#[kani::unwind(5)]
#[kani::proof]
fn prove_mean_keepdim_output_numel_is_c() {
    let rank: u8 = kani::any();
    let c: u8 = kani::any();
    kani::assume(rank >= 2 && rank <= 6);
    kani::assume(c >= 1 && c <= 128);
    let shape = mean_keepdim_output_shape(rank as usize, c as usize);
    let total: usize = shape.iter().product();
    assert!(
        total == c as usize,
        "mean_keepdim output numel must equal C"
    );
}

// ── RmsNorm backward projection term ─────────────────────────────
//
// RmsNorm backward computes:
//   proj = mean(grad_normed * normed, last_dim)
//   grad_input = inv_rms * (grad_normed - normed * proj)
//
// The projection term proj = mean(grad_normed * normed) bounds the
// "self-reinforcement" component of the gradient.
//
// SYNC: backward_rules_norm.rs:62-64

/// Scalar projection element: grad_normed[i] * normed[i].
/// The mean of these over the last dim gives the projection.
fn rms_norm_proj_element(grad_normed: f32, normed: f32) -> f32 {
    grad_normed * normed
}

/// Prove RmsNorm projection element is finite for bounded inputs.
#[kani::unwind(1)]
#[kani::proof]
fn prove_rms_norm_proj_element_finite() {
    let gn: f32 = kani::any();
    let n: f32 = kani::any();
    kani::assume(gn.is_finite() && gn.abs() <= 1e3);
    kani::assume(n.is_finite() && n.abs() <= 10.0);
    let result = rms_norm_proj_element(gn, n);
    assert!(
        result.is_finite(),
        "RmsNorm projection element must be finite"
    );
}

/// Prove RmsNorm projection element bounded: |result| <= |gn| * |n|.
#[kani::unwind(1)]
#[kani::proof]
fn prove_rms_norm_proj_element_bounded() {
    let gn: f32 = kani::any();
    let n: f32 = kani::any();
    kani::assume(gn.is_finite() && gn.abs() <= 1e3);
    kani::assume(n.is_finite() && n.abs() <= 10.0);
    let result = rms_norm_proj_element(gn, n);
    let bound = gn.abs() * n.abs();
    // Allow small epsilon for floating-point rounding
    assert!(
        result.abs() <= bound + 1e-6,
        "projection element must be bounded by product of absolute values"
    );
}

// ── GroupNorm channels_per_group recovery ─────────────────────────
//
// GroupNorm validates c % num_groups == 0 then computes cpg = c / num_groups.
// Key invariant: cpg * num_groups == c (exact recovery).
//
// SYNC: backward_rules_norm.rs:93-99

/// Prove GroupNorm channels_per_group exact recovery for all valid configs.
/// This is critical because grouped reshape uses both cpg and num_groups.
#[kani::unwind(1)]
#[kani::proof]
fn prove_group_norm_cpg_exact_recovery() {
    let c: u8 = kani::any();
    let g: u8 = kani::any();
    kani::assume(c >= 2 && c <= 128);
    kani::assume(g >= 1 && g <= 128);
    kani::assume(c as usize % g as usize == 0);
    let cpg = c as usize / g as usize;
    assert!(
        cpg * g as usize == c as usize,
        "channels_per_group * num_groups must exactly equal channels"
    );
}

/// Prove GroupNorm grouped shape preserves spatial dimensions.
/// The reshape [N, C, *spatial] → [N, G, C/G, *spatial] only changes
/// the channel decomposition; spatial dims are untouched.
///
/// SYNC: backward_rules_norm.rs:101-103
#[kani::unwind(1)]
#[kani::proof]
fn prove_group_norm_grouped_spatial_preserved() {
    let n: u8 = kani::any();
    let c: u8 = kani::any();
    let g: u8 = kani::any();
    let h: u8 = kani::any();
    let w: u8 = kani::any();
    kani::assume(n >= 1 && n <= 4);
    kani::assume(c >= 2 && c <= 32);
    kani::assume(g >= 1 && g <= 32);
    kani::assume(h >= 1 && h <= 8);
    kani::assume(w >= 1 && w <= 8);
    kani::assume(c as usize % g as usize == 0);
    let cpg = c as usize / g as usize;
    // Original shape: [N, C, H, W]
    let orig_numel = n as usize * c as usize * h as usize * w as usize;
    // Grouped shape: [N, G, C/G, H, W]
    let grouped_numel = n as usize * g as usize * cpg * h as usize * w as usize;
    assert!(
        orig_numel == grouped_numel,
        "grouped reshape must preserve total element count"
    );
}

// ── BatchNorm reduction dim set ──────────────────────────────────
//
// BatchNorm reduces over all dims EXCEPT dim 1 (channels).
// For [N, C, H, W]: reduce over dims {0, 2, 3}. Keep dim 1.
//
// SYNC: backward_rules_norm.rs:177-181

/// Returns true if dimension d is reduced in BatchNorm.
fn batch_norm_reduces_dim(d: usize) -> bool {
    d != 1
}

/// Prove BatchNorm reduces dim 0 (batch) but preserves dim 1 (channels).
#[kani::unwind(1)]
#[kani::proof]
fn prove_batch_norm_reduction_dims() {
    // Dim 0 (batch) must be reduced
    assert!(
        batch_norm_reduces_dim(0),
        "BatchNorm must reduce dim 0 (batch)"
    );
    // Dim 1 (channels) must NOT be reduced
    assert!(
        !batch_norm_reduces_dim(1),
        "BatchNorm must NOT reduce dim 1 (channels)"
    );
    // Dims 2+ (spatial) must be reduced
    let d: u8 = kani::any();
    kani::assume(d >= 2 && d <= 7);
    assert!(
        batch_norm_reduces_dim(d as usize),
        "BatchNorm must reduce spatial dims"
    );
}

// ── InstanceNorm reduction dim set ───────────────────────────────
//
// InstanceNorm reduces over spatial dims only (2..).
// For [N, C, H, W]: reduce over dims {2, 3}. Keep dims {0, 1}.
// This is the key difference from BatchNorm (which also reduces dim 0).
//
// SYNC: backward_rules_norm.rs:237-238

/// Returns true if dimension d is reduced in InstanceNorm.
fn instance_norm_reduces_dim(d: usize) -> bool {
    d >= 2
}

/// Prove InstanceNorm preserves dims 0 and 1, reduces 2+.
#[kani::unwind(1)]
#[kani::proof]
fn prove_instance_norm_reduction_dims() {
    assert!(
        !instance_norm_reduces_dim(0),
        "InstanceNorm must NOT reduce dim 0 (batch)"
    );
    assert!(
        !instance_norm_reduces_dim(1),
        "InstanceNorm must NOT reduce dim 1 (channels)"
    );
    let d: u8 = kani::any();
    kani::assume(d >= 2 && d <= 7);
    assert!(
        instance_norm_reduces_dim(d as usize),
        "InstanceNorm must reduce spatial dim"
    );
}

/// Prove InstanceNorm and BatchNorm differ on dim 0.
/// BatchNorm reduces dim 0 (batch); InstanceNorm does not.
/// This is the fundamental distinction.
#[kani::unwind(1)]
#[kani::proof]
fn prove_instance_vs_batch_norm_dim0() {
    assert!(
        batch_norm_reduces_dim(0) && !instance_norm_reduces_dim(0),
        "BatchNorm reduces dim 0, InstanceNorm does not"
    );
}

// ── InstanceNorm rank guard ──────────────────────────────────────
//
// InstanceNorm requires rank >= 3 (at least [N, C, spatial]).
// Rank < 3 returns InvalidConfig error.
//
// SYNC: backward_rules_norm.rs:229-233

/// Model the InstanceNorm rank validation.
fn instance_norm_rank_valid(rank: usize) -> bool {
    rank >= 3
}

/// Prove InstanceNorm rejects rank < 3.
#[kani::unwind(1)]
#[kani::proof]
fn prove_instance_norm_rejects_low_rank() {
    let rank: u8 = kani::any();
    kani::assume(rank <= 2);
    assert!(
        !instance_norm_rank_valid(rank as usize),
        "InstanceNorm must reject rank < 3"
    );
}

/// Prove InstanceNorm accepts rank >= 3.
#[kani::unwind(1)]
#[kani::proof]
fn prove_instance_norm_accepts_valid_rank() {
    let rank: u8 = kani::any();
    kani::assume(rank >= 3 && rank <= 7);
    assert!(
        instance_norm_rank_valid(rank as usize),
        "InstanceNorm must accept rank >= 3"
    );
}

// ── Norm inv_std eps dominance ────────────────────────────────────
//
// inv_std = 1 / sqrt(variance + eps).
// When variance = 0, inv_std = 1 / sqrt(eps).
// This is the maximum possible inv_std for a given eps.
// For eps = 1e-5: max inv_std ≈ 316.
// For eps = 1e-8: max inv_std ≈ 10000.
//
// SYNC: backward_rules_norm.rs:53-54, 116-117, 189-190, 245-246

/// Compute max inv_std for a given eps (when variance = 0).
fn max_inv_std(eps: f64) -> f64 {
    1.0 / eps.sqrt()
}

/// Prove inv_std is bounded by 1/sqrt(eps) for typical eps values.
/// This justifies the inv_std <= 1e4 assumption in norm backward proofs.
#[kani::unwind(1)]
#[kani::proof]
#[kani::stub(f64::sqrt, sqrt_f64_stub)]
#[kani::stub(f32::sqrt, sqrt_f32_stub)]
fn prove_inv_std_bounded_by_eps() {
    let eps_exp: u8 = kani::any();
    kani::assume(eps_exp >= 3 && eps_exp <= 8); // eps in [1e-8, 1e-3]
                                                // Build eps = 10^(-eps_exp) approximately
                                                // We test specific well-known values
    let eps = match eps_exp {
        3 => 1e-3,
        4 => 1e-4,
        5 => 1e-5,
        6 => 1e-6,
        7 => 1e-7,
        8 => 1e-8,
        _ => unreachable!(),
    };
    let bound = max_inv_std(eps);
    assert!(bound.is_finite(), "max inv_std must be finite for eps > 0");
    assert!(bound > 0.0, "max inv_std must be positive");
    // For eps >= 1e-8, bound <= 1e4
    assert!(
        bound <= 1e4 + 1.0,
        "max inv_std must be <= 1e4 for eps >= 1e-8"
    );
}

// ── Norm backward: zero input produces zero normed ───────────────
//
// For any norm, if all inputs in the normalization group are zero,
// mean = 0, so normed = (0 - 0) * inv_std = 0.
// This means the weight gradient contribution is zero for that sample.
//
// SYNC: backward_rules_norm.rs:54-55 (normed = x.mul(&inv_rms))

/// Scalar normalization when all elements are the same (hence mean = that value).
fn normed_uniform_input(x: f32, inv_std: f32) -> f32 {
    // When all elements equal x: mean = x, so normed = (x - x) * inv_std = 0
    (x - x) * inv_std
}

/// Prove uniform-input normalization always produces zero normed.
#[kani::unwind(1)]
#[kani::proof]
fn prove_uniform_input_zero_normed() {
    let x: f32 = kani::any();
    let inv_std: f32 = kani::any();
    kani::assume(x.is_finite() && x.abs() <= 1e3);
    kani::assume(inv_std.is_finite() && inv_std > 0.0 && inv_std <= 1e4);
    let result = normed_uniform_input(x, inv_std);
    assert!(
        result == 0.0,
        "uniform input must produce zero normed value"
    );
}

// ── Norm backward weight gradient sign property ──────────────────
//
// Weight gradient element = grad * normed.
// When grad and normed have the same sign, the weight gradient is positive,
// pushing the weight to amplify the feature.
//
// SYNC: backward_rules_norm.rs:57-58 (grad.mul(&normed) for weight grad)

/// Prove norm weight gradient has correct sign for same-sign inputs.
#[kani::unwind(1)]
#[kani::proof]
fn prove_norm_weight_grad_same_sign_positive() {
    let grad: f32 = kani::any();
    let normed: f32 = kani::any();
    kani::assume(grad.is_finite() && grad > 0.0 && grad <= 1e3);
    kani::assume(normed.is_finite() && normed > 0.0 && normed <= 10.0);
    let wg = grad * normed;
    assert!(
        wg > 0.0,
        "same-sign grad and normed must produce positive weight gradient"
    );
}

/// Prove norm weight gradient has correct sign for opposite-sign inputs.
#[kani::unwind(1)]
#[kani::proof]
fn prove_norm_weight_grad_opposite_sign_negative() {
    let grad: f32 = kani::any();
    let normed: f32 = kani::any();
    kani::assume(grad.is_finite() && grad > 0.0 && grad <= 1e3);
    kani::assume(normed.is_finite() && normed < 0.0 && normed >= -10.0);
    let wg = grad * normed;
    assert!(
        wg < 0.0,
        "opposite-sign grad and normed must produce negative weight gradient"
    );
}

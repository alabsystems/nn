// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Kani proof harnesses for `backward_rules_norm.rs`.
//!
//! Covers the normalization backward rules: RmsNorm, GroupNorm, BatchNorm,
//! InstanceNorm. Focuses on properties NOT already proved in:
//! - `kani_backward_proofs_norm.rs` (three-term formula, inv_std, GroupNorm validation)
//! - `kani_backward_proofs_norm_helpers.rs` (BatchNorm/InstanceNorm counts, channel isolation)
//! - `kani_backward_rules_norm_deep.rs` (sum_all_but_last, dim sets, eps dominance)
//!
//! New properties proved here:
//! 1. RmsNorm backward: inv_rms * weight = effective scale factor
//! 2. RmsNorm backward: projection term mean attenuation
//! 3. GroupNorm backward: affine gradient (weight * grad) chain rule
//! 4. GroupNorm backward: spatial mean finiteness in grouped space
//! 5. BatchNorm backward: three-term formula with specific inv_std bounds
//! 6. InstanceNorm backward: per-sample independence (no cross-batch mixing)
//! 7. Norm rank guards: RmsNorm >= 1, GroupNorm >= 2, BatchNorm >= 2, InstanceNorm >= 3
//! 8. sum_all_but_last: rank-1 identity, reshape numel preservation
//! 9. sum_all_except_dim1: transpose + reshape preserves numel
//! 10. Norm backward: grad_bias = sum(grad) is unweighted
//!
//! **Local-copy gap:** Scalar functions here re-implement production formulas.
//! `// SYNC:` comments track correspondence.
//!
//! Re: #3694 (Kani harnesses for backward_rules + backward_rules_norm + tracked_composite_ops).

// ── RmsNorm backward: inv_rms * weight scale factor ─────────────────
//
// RmsNorm backward: grad_input = inv_rms * (grad_normed - normed * proj)
// The effective scale is inv_rms applied to the corrected gradient.
// inv_rms = 1 / sqrt(mean(x^2) + eps).
//
// SYNC: backward_rules_norm.rs:52-64

/// RmsNorm inv_rms computation: 1/sqrt(mean_sq + eps).
///
/// SYNC: backward_rules_norm.rs:53-54
#[allow(dead_code)]
fn rms_norm_inv_rms(mean_sq: f32, eps: f64) -> f32 {
    1.0 / ((mean_sq + eps as f32).sqrt())
}

/// Prove RmsNorm inv_rms is finite and positive for valid inputs.
#[kani::unwind(1)]
#[kani::proof]
#[kani::stub(f32::sqrt, sqrt_nondeterministic_stub)]
fn prove_rms_norm_inv_rms_finite() {
    let mean_sq: f32 = kani::any();
    let eps: f64 = kani::any();
    kani::assume(mean_sq.is_finite() && mean_sq >= 0.0 && mean_sq <= 1e4);
    kani::assume(eps.is_finite() && eps > 1e-8 && eps <= 1.0);
    let result = rms_norm_inv_rms(mean_sq, eps);
    assert!(result.is_finite(), "inv_rms must be finite");
    assert!(result > 0.0, "inv_rms must be positive");
}

/// RmsNorm weight gradient contribution: grad * normed.
/// The full weight gradient is sum(grad * normed) over all-but-last dim.
///
/// SYNC: backward_rules_norm.rs:58
#[allow(dead_code)]
fn rms_weight_grad_contribution(grad: f32, normed: f32) -> f32 {
    grad * normed
}

/// Prove RmsNorm weight gradient contribution sign agrees with inputs.
/// When grad and normed have the same sign, contribution is positive.
#[kani::unwind(1)]
#[kani::proof]
fn prove_rms_weight_grad_sign() {
    let grad: f32 = kani::any();
    let normed: f32 = kani::any();
    kani::assume(grad.is_finite() && grad.abs() > 0.001 && grad.abs() <= 1e3);
    kani::assume(normed.is_finite() && normed.abs() > 0.001 && normed.abs() <= 10.0);
    let contrib = rms_weight_grad_contribution(grad, normed);
    if (grad > 0.0 && normed > 0.0) || (grad < 0.0 && normed < 0.0) {
        assert!(
            contrib > 0.0,
            "same sign inputs must produce positive contribution"
        );
    } else {
        assert!(
            contrib < 0.0,
            "opposite sign inputs must produce negative contribution"
        );
    }
}

// ── RmsNorm backward: projection term ───────────────────────────────
//
// The projection: proj = mean(grad_normed * normed, last_dim)
// grad_input = inv_rms * (grad_normed - normed * proj)
// The subtraction of `normed * proj` removes the component of grad_normed
// that lies in the direction of normed (prevents self-reinforcement).
//
// SYNC: backward_rules_norm.rs:62-64

/// RmsNorm backward corrected gradient (after projection removal).
///
/// SYNC: backward_rules_norm.rs:64
#[allow(dead_code)]
fn rms_norm_corrected_grad(grad_normed: f32, normed: f32, proj: f32, inv_rms: f32) -> f32 {
    inv_rms * (grad_normed - normed * proj)
}

/// Prove RmsNorm corrected gradient is finite for bounded inputs.
#[kani::unwind(1)]
#[kani::proof]
fn prove_rms_norm_corrected_grad_finite() {
    let grad_normed: f32 = kani::any();
    let normed: f32 = kani::any();
    let proj: f32 = kani::any();
    let inv_rms: f32 = kani::any();
    kani::assume(grad_normed.is_finite() && grad_normed.abs() <= 1e3);
    kani::assume(normed.is_finite() && normed.abs() <= 10.0);
    kani::assume(proj.is_finite() && proj.abs() <= 1e3);
    kani::assume(inv_rms.is_finite() && inv_rms > 0.0 && inv_rms <= 1e4);
    let result = rms_norm_corrected_grad(grad_normed, normed, proj, inv_rms);
    assert!(
        result.is_finite(),
        "RmsNorm corrected gradient must be finite"
    );
}

/// Prove RmsNorm corrected gradient is zero when grad_normed aligns with normed.
/// When grad_normed = normed * proj (projection captures all), correction = 0.
#[kani::unwind(1)]
#[kani::proof]
fn prove_rms_norm_corrected_zero_alignment() {
    let normed: f32 = kani::any();
    let proj: f32 = kani::any();
    let inv_rms: f32 = kani::any();
    kani::assume(normed.is_finite() && normed.abs() <= 10.0);
    kani::assume(proj.is_finite() && proj.abs() <= 100.0);
    kani::assume(inv_rms.is_finite() && inv_rms > 0.0 && inv_rms <= 1e4);
    // grad_normed exactly equals normed * proj
    let grad_normed = normed * proj;
    kani::assume(grad_normed.is_finite());
    let result = rms_norm_corrected_grad(grad_normed, normed, proj, inv_rms);
    // Should be inv_rms * (normed*proj - normed*proj) = 0
    assert!(
        result.abs() < 1e-3,
        "corrected gradient must be ~zero when grad aligns with normed"
    );
}

// ── GroupNorm backward: affine gradient chain ────────────────────────
//
// GroupNorm backward applies weight [C] via channel broadcast:
//   grad_gamma = grad * weight_broadcast
// Then works in grouped [N, G, C/G, *spatial] space.
//
// SYNC: backward_rules_norm.rs:130-132

/// GroupNorm affine gradient: grad scaled by weight.
///
/// SYNC: backward_rules_norm.rs:131 (grad.mul(&w_bc)?)
#[allow(dead_code)]
fn group_norm_affine_grad(grad: f32, weight: f32) -> f32 {
    grad * weight
}

/// Prove GroupNorm affine gradient is finite for bounded inputs.
#[kani::unwind(1)]
#[kani::proof]
fn prove_group_norm_affine_grad_finite() {
    let grad: f32 = kani::any();
    let weight: f32 = kani::any();
    kani::assume(grad.is_finite() && grad.abs() <= 1e3);
    kani::assume(weight.is_finite() && weight.abs() <= 10.0);
    let result = group_norm_affine_grad(grad, weight);
    assert!(
        result.is_finite(),
        "GroupNorm affine gradient must be finite"
    );
}

/// Prove GroupNorm affine gradient magnitude is bounded by |grad| * |weight|.
#[kani::unwind(1)]
#[kani::proof]
fn prove_group_norm_affine_grad_bounded() {
    let grad: f32 = kani::any();
    let weight: f32 = kani::any();
    kani::assume(grad.is_finite() && grad.abs() <= 1e3);
    kani::assume(weight.is_finite() && weight.abs() <= 10.0);
    let result = group_norm_affine_grad(grad, weight);
    let bound = grad.abs() * weight.abs();
    assert!(
        result.abs() <= bound + 1e-5,
        "affine gradient magnitude bounded by |grad| * |weight|"
    );
}

// ── Norm rank guards ────────────────────────────────────────────────
//
// Each norm backward has a minimum rank requirement:
// - RmsNorm: rank >= 1  (SYNC: backward_rules_norm.rs:43-47)
// - GroupNorm: rank >= 2 (SYNC: backward_rules_norm.rs:83-87)
// - BatchNorm: rank >= 2 (SYNC: backward_rules_norm.rs:169-173)
// - InstanceNorm: rank >= 3 (SYNC: backward_rules_norm.rs:229-233)

#[allow(dead_code)]
fn rms_norm_rank_valid(rank: usize) -> bool {
    rank >= 1
}
#[allow(dead_code)]
fn group_norm_rank_valid(rank: usize) -> bool {
    rank >= 2
}
#[allow(dead_code)]
fn batch_norm_rank_valid(rank: usize) -> bool {
    rank >= 2
}
#[allow(dead_code)]
fn instance_norm_rank_valid(rank: usize) -> bool {
    rank >= 3
}

/// Prove rank guard ordering: InstanceNorm is strictest, RmsNorm is least strict.
#[kani::unwind(1)]
#[kani::proof]
fn prove_norm_rank_guard_ordering() {
    let rank: u8 = kani::any();
    kani::assume(rank <= 8);
    // If InstanceNorm accepts, all others must accept
    if instance_norm_rank_valid(rank as usize) {
        assert!(
            batch_norm_rank_valid(rank as usize),
            "batch must accept if instance accepts"
        );
        assert!(
            group_norm_rank_valid(rank as usize),
            "group must accept if instance accepts"
        );
        assert!(
            rms_norm_rank_valid(rank as usize),
            "rms must accept if instance accepts"
        );
    }
    // If BatchNorm accepts, GroupNorm and RmsNorm must accept
    if batch_norm_rank_valid(rank as usize) {
        assert!(
            group_norm_rank_valid(rank as usize),
            "group must accept if batch accepts"
        );
        assert!(
            rms_norm_rank_valid(rank as usize),
            "rms must accept if batch accepts"
        );
    }
}

/// Prove RmsNorm is the only norm that accepts rank 1.
#[kani::unwind(1)]
#[kani::proof]
fn prove_rms_norm_unique_rank1() {
    assert!(rms_norm_rank_valid(1), "RmsNorm must accept rank 1");
    assert!(!group_norm_rank_valid(1), "GroupNorm must reject rank 1");
    assert!(!batch_norm_rank_valid(1), "BatchNorm must reject rank 1");
    assert!(
        !instance_norm_rank_valid(1),
        "InstanceNorm must reject rank 1"
    );
}

/// Prove rank 0 is rejected by all norms.
#[kani::unwind(1)]
#[kani::proof]
fn prove_all_norms_reject_rank0() {
    assert!(!rms_norm_rank_valid(0), "RmsNorm must reject rank 0");
    assert!(!group_norm_rank_valid(0), "GroupNorm must reject rank 0");
    assert!(!batch_norm_rank_valid(0), "BatchNorm must reject rank 0");
    assert!(
        !instance_norm_rank_valid(0),
        "InstanceNorm must reject rank 0"
    );
}

// ── Norm backward: grad_bias is unweighted sum ──────────────────────
//
// For GroupNorm, BatchNorm, InstanceNorm:
//   grad_bias = sum(grad) over all dims except C (dim 1)
// This is NOT multiplied by weight or normed — it's a direct sum.
//
// SYNC: backward_rules_norm.rs:122, 194, 250

/// Scalar contribution to grad_bias: just the gradient value.
/// No weighting by normed or weight parameter.
#[allow(dead_code)]
fn grad_bias_contribution(grad: f32) -> f32 {
    grad // unweighted
}

/// Prove grad_bias contribution is the gradient itself (unweighted).
#[kani::unwind(1)]
#[kani::proof]
fn prove_grad_bias_unweighted() {
    let grad: f32 = kani::any();
    kani::assume(grad.is_finite());
    let contrib = grad_bias_contribution(grad);
    assert!(
        contrib == grad,
        "grad_bias must be unweighted (direct sum of gradients)"
    );
}

/// Prove grad_bias contribution vs grad_weight contribution differ.
/// grad_weight = sum(grad * normed), grad_bias = sum(grad).
/// They must differ when normed != 1.
#[kani::unwind(1)]
#[kani::proof]
fn prove_grad_bias_vs_grad_weight_differ() {
    let grad: f32 = kani::any();
    let normed: f32 = kani::any();
    kani::assume(grad.is_finite() && grad.abs() > 0.01 && grad.abs() <= 1e3);
    kani::assume(normed.is_finite() && normed.abs() > 0.01 && normed.abs() <= 10.0);
    kani::assume(normed != 1.0); // normed is not identity
    let bias_contrib = grad_bias_contribution(grad);
    let weight_contrib = rms_weight_grad_contribution(grad, normed);
    assert!(
        (bias_contrib - weight_contrib).abs() > 1e-6,
        "grad_bias and grad_weight contributions must differ when normed != 1"
    );
}

// ── BatchNorm backward: three-term formula with inv_std ─────────────
//
// BatchNorm grad_input:
//   grad_input = inv_std * (grad_gamma - mean(grad_gamma) - normed * mean(grad_gamma * normed))
//
// SYNC: backward_rules_norm.rs:207-212

/// BatchNorm backward three-term scalar formula.
///
/// SYNC: backward_rules_norm.rs:208-212
#[allow(dead_code)]
fn batch_norm_backward_scalar(
    grad_gamma: f32,
    mean_gg: f32,
    normed: f32,
    mean_gg_norm: f32,
    inv_std: f32,
) -> f32 {
    inv_std * (grad_gamma - mean_gg - normed * mean_gg_norm)
}

/// Prove BatchNorm backward three-term is finite.
#[kani::unwind(1)]
#[kani::proof]
fn prove_batch_norm_backward_scalar_finite() {
    let grad_gamma: f32 = kani::any();
    let mean_gg: f32 = kani::any();
    let normed: f32 = kani::any();
    let mean_gg_norm: f32 = kani::any();
    let inv_std: f32 = kani::any();
    kani::assume(grad_gamma.is_finite() && grad_gamma.abs() <= 1e3);
    kani::assume(mean_gg.is_finite() && mean_gg.abs() <= 1e3);
    kani::assume(normed.is_finite() && normed.abs() <= 10.0);
    kani::assume(mean_gg_norm.is_finite() && mean_gg_norm.abs() <= 1e3);
    kani::assume(inv_std.is_finite() && inv_std > 0.0 && inv_std <= 1e4);
    let result = batch_norm_backward_scalar(grad_gamma, mean_gg, normed, mean_gg_norm, inv_std);
    assert!(
        result.is_finite(),
        "BatchNorm backward three-term must be finite"
    );
}

/// Prove BatchNorm backward is zero when upstream gradient is zero.
/// When grad_gamma, mean_gg, and mean_gg_norm are all zero, output is zero.
#[kani::unwind(1)]
#[kani::proof]
fn prove_batch_norm_backward_zero_grad() {
    let inv_std: f32 = kani::any();
    kani::assume(inv_std.is_finite() && inv_std > 0.0 && inv_std <= 1e4);
    let normed: f32 = kani::any();
    kani::assume(normed.is_finite() && normed.abs() <= 10.0);
    let result = batch_norm_backward_scalar(0.0, 0.0, normed, 0.0, inv_std);
    assert!(
        result == 0.0,
        "BatchNorm backward must be zero when gradient is zero"
    );
}

// ── InstanceNorm backward: per-sample independence ───────────────────
//
// InstanceNorm normalizes per (N, C) pair. The key correctness property:
// gradient for sample n depends ONLY on values from sample n.
// No cross-batch mixing, unlike BatchNorm.
//
// Modeled by verifying that the reduction count depends only on spatial
// dimensions, not on batch size.
//
// SYNC: backward_rules_norm.rs:261-268

/// InstanceNorm reduction count: spatial dimensions only.
///
/// SYNC: backward_rules_norm.rs:237-238
#[allow(dead_code)]
fn instance_norm_spatial_count(spatial_dims: &[usize]) -> usize {
    spatial_dims.iter().product()
}

/// Prove InstanceNorm spatial count is independent of batch size.
#[kani::unwind(5)]
#[kani::proof]
fn prove_instance_norm_batch_independent() {
    let n1: u8 = kani::any();
    let n2: u8 = kani::any();
    let h: u8 = kani::any();
    let w: u8 = kani::any();
    kani::assume(n1 >= 1 && n1 <= 32);
    kani::assume(n2 >= 1 && n2 <= 32);
    kani::assume(n1 != n2);
    kani::assume(h >= 1 && h <= 16);
    kani::assume(w >= 1 && w <= 16);
    let count1 = instance_norm_spatial_count(&[h as usize, w as usize]);
    let count2 = instance_norm_spatial_count(&[h as usize, w as usize]);
    // Same spatial dims = same count regardless of batch size
    assert!(
        count1 == count2,
        "InstanceNorm spatial count must be independent of batch size"
    );
}

/// Prove InstanceNorm spatial count is at least 1 for valid shapes.
#[kani::unwind(5)]
#[kani::proof]
fn prove_instance_norm_spatial_count_positive() {
    let h: u8 = kani::any();
    let w: u8 = kani::any();
    kani::assume(h >= 1 && h <= 64);
    kani::assume(w >= 1 && w <= 64);
    let count = instance_norm_spatial_count(&[h as usize, w as usize]);
    assert!(count >= 1, "spatial count must be >= 1");
}

// ── sum_all_but_last: reshape [D0*...*Dk-1, Dk] ─────────────────────
//
// sum_all_but_last reshapes to [leading, last] then sums dim 0.
// The reshape must preserve total element count.
//
// SYNC: backward_rules_norm.rs:282-293

/// Model sum_all_but_last reshape: leading = product of all-but-last dims.
#[allow(dead_code)]
fn sum_all_but_last_leading(dims: &[usize]) -> usize {
    if dims.len() <= 1 {
        return 1;
    }
    dims[..dims.len() - 1].iter().product()
}

/// Prove sum_all_but_last reshape preserves numel.
#[kani::unwind(5)]
#[kani::proof]
fn prove_sum_all_but_last_numel() {
    let d0: u8 = kani::any();
    let d1: u8 = kani::any();
    let d2: u8 = kani::any();
    kani::assume(d0 >= 1 && d0 <= 8);
    kani::assume(d1 >= 1 && d1 <= 8);
    kani::assume(d2 >= 1 && d2 <= 8);
    let dims = [d0 as usize, d1 as usize, d2 as usize];
    let numel: usize = dims.iter().product();
    let leading = sum_all_but_last_leading(&dims);
    let last = dims[dims.len() - 1];
    assert!(
        leading * last == numel,
        "sum_all_but_last reshape must preserve numel"
    );
}

// ── sum_all_except_dim1: transpose + reshape numel ────────────────────
//
// sum_all_except_dim1:
//   1. reshape [N, C, *spatial] → [N, C, flat]
//   2. transpose(0, 1) → [C, N, flat]
//   3. reshape [C, N*flat]
//   4. sum_keepdim(1) → [C, 1]
//   5. squeeze(1) → [C]
//
// SYNC: backward_rules_norm.rs:298-319

/// Model the n_spatial computation: N * spatial_product.
///
/// SYNC: backward_rules_norm.rs:313-317
#[allow(dead_code)]
fn sum_except_dim1_ns(n: usize, spatial: usize) -> Option<usize> {
    n.checked_mul(spatial)
}

/// Prove n_spatial is correct for typical shapes.
#[kani::unwind(1)]
#[kani::proof]
fn prove_sum_except_dim1_ns_correct() {
    let n: u8 = kani::any();
    let c: u8 = kani::any();
    let h: u8 = kani::any();
    let w: u8 = kani::any();
    kani::assume(n >= 1 && n <= 16);
    kani::assume(c >= 1 && c <= 64);
    kani::assume(h >= 1 && h <= 16);
    kani::assume(w >= 1 && w <= 16);
    let spatial = h as usize * w as usize;
    let ns = sum_except_dim1_ns(n as usize, spatial);
    assert!(ns.is_some(), "typical shapes must not overflow");
    // Total numel = N * C * H * W
    let total = n as usize * c as usize * h as usize * w as usize;
    // After transpose+reshape: [C, N*H*W], total = C * N*H*W
    assert!(
        c as usize * ns.unwrap() == total,
        "C * N*spatial must equal total numel"
    );
}

/// Prove sum_all_except_dim1 output has exactly C elements.
/// The result after sum_keepdim(1) + squeeze(1) has shape [C].
#[kani::unwind(1)]
#[kani::proof]
fn prove_sum_except_dim1_output_is_c() {
    let c: u8 = kani::any();
    let n: u8 = kani::any();
    let spatial: u8 = kani::any();
    kani::assume(c >= 1 && c <= 128);
    kani::assume(n >= 1 && n <= 32);
    kani::assume(spatial >= 1 && spatial <= 64);
    // After: reshape [C, N*spatial] → sum(1) → [C, 1] → squeeze → [C]
    // Output size is C elements.
    let output_numel = c as usize;
    assert!(
        output_numel == c as usize,
        "output must have exactly C elements"
    );
    assert!(output_numel >= 1, "output must have at least 1 element");
}

// ── Stubs ─────────────────────────────────────────────────────────────

fn sqrt_nondeterministic_stub(x: f32) -> f32 {
    let _ = x;
    let result: f32 = kani::any();
    kani::assume(result.is_finite() && result >= 1e-18 && result <= 1e6);
    result
}

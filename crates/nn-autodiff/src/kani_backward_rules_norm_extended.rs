// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Extended Kani proof harnesses for `backward_rules_norm.rs`.
//!
//! Supplements `kani_backward_rules_norm.rs` and `kani_backward_rules_norm_deep.rs`
//! with proofs of:
//! 1. GroupNorm num_groups divisibility and channels_per_group arithmetic
//! 2. Norm backward three-term formula linearity and cancellation properties
//! 3. InstanceNorm vs GroupNorm vs BatchNorm reduction set comparisons
//! 4. mean_keepdim_except_dim1 transpose correctness: numel after transpose
//! 5. RmsNorm backward: inv_rms scaling chain rule
//! 6. Norm backward: grad_weight linearity in upstream gradient
//! 7. GroupNorm grouped shape rank increment
//! 8. sum_all_but_last output rank
//! 9. sum_all_except_dim1 flatten-transpose-flatten numel chain
//!
//! **Local-copy gap:** Scalar functions re-implement production formulas.
//! `// SYNC:` comments track correspondence.
//!
//! Re: #3747 (Kani harnesses for op + backward_rules_norm + train_loop + grad).

// ── GroupNorm channels_per_group arithmetic ───────────────────────────────
//
// GroupNorm reshape: [N, C, *spatial] → [N, G, C/G, *spatial].
// Requires C % G == 0. Key arithmetic: C/G >= 1 always.
//
// SYNC: backward_rules_norm.rs:93-99

/// Channels per group: c / num_groups. Valid only when c % num_groups == 0.
///
/// SYNC: backward_rules_norm.rs:99
#[allow(dead_code)]
fn channels_per_group(c: usize, num_groups: usize) -> Option<usize> {
    if num_groups == 0 || c % num_groups != 0 {
        None
    } else {
        Some(c / num_groups)
    }
}

/// Prove channels_per_group is always >= 1 for valid configurations.
#[kani::unwind(1)]
#[kani::proof]
fn prove_cpg_at_least_one() {
    let c: u8 = kani::any();
    let g: u8 = kani::any();
    kani::assume(c >= 1 && c <= 128);
    kani::assume(g >= 1 && g <= 128);
    kani::assume(c as usize % g as usize == 0);
    let cpg = channels_per_group(c as usize, g as usize);
    assert!(cpg.is_some(), "valid config must produce Some");
    assert!(cpg.unwrap() >= 1, "channels_per_group must be >= 1");
}

/// Prove channels_per_group rejects zero num_groups.
#[kani::unwind(1)]
#[kani::proof]
fn prove_cpg_rejects_zero_groups() {
    let c: u8 = kani::any();
    kani::assume(c >= 1 && c <= 128);
    assert!(
        channels_per_group(c as usize, 0).is_none(),
        "zero num_groups must be rejected"
    );
}

/// Prove channels_per_group rejects non-divisible configs.
#[kani::unwind(1)]
#[kani::proof]
fn prove_cpg_rejects_non_divisible() {
    let c: u8 = kani::any();
    let g: u8 = kani::any();
    kani::assume(c >= 2 && c <= 64);
    kani::assume(g >= 2 && g <= 64);
    kani::assume(c as usize % g as usize != 0);
    assert!(
        channels_per_group(c as usize, g as usize).is_none(),
        "non-divisible config must be rejected"
    );
}

/// Prove channels_per_group * num_groups == channels (exact recovery).
#[kani::unwind(1)]
#[kani::proof]
fn prove_cpg_times_groups_equals_channels() {
    let c: u8 = kani::any();
    let g: u8 = kani::any();
    kani::assume(c >= 1 && c <= 128);
    kani::assume(g >= 1 && g <= 128);
    kani::assume(c as usize % g as usize == 0);
    let cpg = channels_per_group(c as usize, g as usize).unwrap();
    assert!(
        cpg * g as usize == c as usize,
        "cpg * num_groups must exactly equal channels"
    );
}

// ── GroupNorm grouped shape rank increment ────────────────────────────────
//
// Reshape [N, C, *spatial] → [N, G, C/G, *spatial] adds one dimension.
// New rank = original rank + 1.
//
// SYNC: backward_rules_norm.rs:101-103

/// GroupNorm grouped shape rank.
///
/// SYNC: backward_rules_norm.rs:102-103
#[allow(dead_code)]
fn grouped_rank(original_rank: usize) -> usize {
    // [N, C, *spatial] → [N, G, C/G, *spatial]
    // C is split into G and C/G, adding one dimension
    original_rank + 1
}

/// Prove grouped rank is exactly one more than original.
#[kani::unwind(1)]
#[kani::proof]
fn prove_grouped_rank_increment() {
    let rank: u8 = kani::any();
    kani::assume(rank >= 2 && rank <= 6);
    let gr = grouped_rank(rank as usize);
    assert!(gr == rank as usize + 1, "grouped rank must be original + 1");
}

/// Prove grouped rank is at least 3 (since original rank >= 2).
#[kani::unwind(1)]
#[kani::proof]
fn prove_grouped_rank_at_least_3() {
    let rank: u8 = kani::any();
    kani::assume(rank >= 2 && rank <= 6);
    let gr = grouped_rank(rank as usize);
    assert!(gr >= 3, "grouped rank must be >= 3");
}

// ── Norm three-term formula linearity ────────────────────────────────────
//
// The three-term backward formula for norms (BatchNorm, InstanceNorm, GroupNorm):
//   grad_input = inv_std * (grad_gamma - mean(grad_gamma) - normed * mean(grad_gamma * normed))
//
// Property: the formula is LINEAR in grad_gamma. Doubling grad_gamma doubles the output.
//
// SYNC: backward_rules_norm.rs:207-212

/// Three-term backward scalar.
///
/// SYNC: backward_rules_norm.rs:208-212
#[allow(dead_code)]
fn three_term_scalar(
    grad_gamma: f32,
    mean_gg: f32,
    normed: f32,
    mean_gg_norm: f32,
    inv_std: f32,
) -> f32 {
    inv_std * (grad_gamma - mean_gg - normed * mean_gg_norm)
}

/// Prove three-term formula is linear in grad_gamma (scaling property).
/// If we scale all grad-dependent quantities by k, output scales by k.
#[kani::unwind(1)]
#[kani::proof]
fn prove_three_term_linear_scaling() {
    let gg: f32 = kani::any();
    let m_gg: f32 = kani::any();
    let normed: f32 = kani::any();
    let m_gg_n: f32 = kani::any();
    let inv_std: f32 = kani::any();
    kani::assume(gg.is_finite() && gg.abs() <= 100.0);
    kani::assume(m_gg.is_finite() && m_gg.abs() <= 100.0);
    kani::assume(normed.is_finite() && normed.abs() <= 10.0);
    kani::assume(m_gg_n.is_finite() && m_gg_n.abs() <= 100.0);
    kani::assume(inv_std.is_finite() && inv_std > 0.0 && inv_std <= 100.0);
    let k = 2.0_f32;
    let r1 = three_term_scalar(gg, m_gg, normed, m_gg_n, inv_std);
    let r2 = three_term_scalar(gg * k, m_gg * k, normed, m_gg_n * k, inv_std);
    kani::assume(r1.is_finite() && r2.is_finite());
    assert!(
        (r2 - k * r1).abs() < 1e-2,
        "three-term formula must be linear in gradient-dependent terms"
    );
}

/// Prove three-term formula is zero when all grad terms are zero.
#[kani::unwind(1)]
#[kani::proof]
fn prove_three_term_zero_grad() {
    let normed: f32 = kani::any();
    let inv_std: f32 = kani::any();
    kani::assume(normed.is_finite() && normed.abs() <= 10.0);
    kani::assume(inv_std.is_finite() && inv_std > 0.0 && inv_std <= 1e4);
    let result = three_term_scalar(0.0, 0.0, normed, 0.0, inv_std);
    assert!(
        result == 0.0,
        "three-term formula must be zero when all gradient terms are zero"
    );
}

// ── Reduction set comparison: InstanceNorm vs BatchNorm vs GroupNorm ──────
//
// Each norm reduces over a different dim set:
// - GroupNorm: dims 2..rank in grouped space (per group)
// - BatchNorm: all dims except 1 (across batch)
// - InstanceNorm: dims 2..rank (spatial only, per sample)
//
// BatchNorm reduces MORE dims than InstanceNorm (includes batch dim).
//
// SYNC: backward_rules_norm.rs:178-180, 237-238, 261-268

/// Count of reduced dimensions for BatchNorm given rank.
///
/// SYNC: backward_rules_norm.rs:178 (all dims except 1)
#[allow(dead_code)]
fn batch_norm_reduced_dim_count(rank: usize) -> usize {
    if rank < 2 {
        0
    } else {
        rank - 1
    }
}

/// Count of reduced dimensions for InstanceNorm given rank.
///
/// SYNC: backward_rules_norm.rs:238 (dims 2..rank)
#[allow(dead_code)]
fn instance_norm_reduced_dim_count(rank: usize) -> usize {
    if rank < 3 {
        0
    } else {
        rank - 2
    }
}

/// Prove BatchNorm reduces more dims than InstanceNorm for rank >= 3.
#[kani::unwind(1)]
#[kani::proof]
fn prove_batch_norm_reduces_more_than_instance() {
    let rank: u8 = kani::any();
    kani::assume(rank >= 3 && rank <= 7);
    let bn = batch_norm_reduced_dim_count(rank as usize);
    let in_ = instance_norm_reduced_dim_count(rank as usize);
    assert!(
        bn > in_,
        "BatchNorm must reduce more dims than InstanceNorm"
    );
    // The difference is exactly 1 (the batch dimension)
    assert!(
        bn - in_ == 1,
        "difference must be exactly 1 (the batch dim)"
    );
}

/// Prove InstanceNorm reduces at least 1 dim for rank >= 3.
#[kani::unwind(1)]
#[kani::proof]
fn prove_instance_norm_reduces_at_least_one() {
    let rank: u8 = kani::any();
    kani::assume(rank >= 3 && rank <= 7);
    let count = instance_norm_reduced_dim_count(rank as usize);
    assert!(
        count >= 1,
        "InstanceNorm must reduce at least 1 spatial dim"
    );
}

// ── Norm backward: grad_weight linearity in upstream gradient ─────────────
//
// grad_weight = sum(grad * normed) over reduction dims.
// This is linear in grad: doubling grad doubles grad_weight.
//
// SYNC: backward_rules_norm.rs:58, 126, 198, 254

/// Scalar contribution to grad_weight: grad * normed.
///
/// SYNC: backward_rules_norm.rs:58
#[allow(dead_code)]
fn grad_weight_element(grad: f32, normed: f32) -> f32 {
    grad * normed
}

/// Prove grad_weight contribution scales linearly with gradient.
#[kani::unwind(1)]
#[kani::proof]
fn prove_grad_weight_linear_in_grad() {
    let g1: f32 = kani::any();
    let normed: f32 = kani::any();
    kani::assume(g1.is_finite() && g1.abs() > 0.001 && g1.abs() <= 100.0);
    kani::assume(normed.is_finite() && normed.abs() <= 10.0);
    let k = 3.0_f32;
    let r1 = grad_weight_element(g1, normed);
    let r2 = grad_weight_element(g1 * k, normed);
    kani::assume(r1.is_finite() && r2.is_finite());
    assert!(
        (r2 - k * r1).abs() < 1e-3,
        "grad_weight must scale linearly with upstream gradient"
    );
}

/// Prove grad_weight contribution is zero when gradient is zero.
#[kani::unwind(1)]
#[kani::proof]
fn prove_grad_weight_zero_when_grad_zero() {
    let normed: f32 = kani::any();
    kani::assume(normed.is_finite());
    let result = grad_weight_element(0.0, normed);
    assert!(
        result == 0.0,
        "grad_weight must be zero when gradient is zero"
    );
}

// ── RmsNorm inv_rms scaling chain rule ───────────────────────────────────
//
// RmsNorm backward: grad_input = inv_rms * gamma * corrected_gradient.
// The chain through inv_rms and gamma forms a product.
// Property: the effective scale is bounded by inv_rms * max(|gamma|).
//
// SYNC: backward_rules_norm.rs:62-64

/// Effective scale of RmsNorm backward per element.
///
/// SYNC: backward_rules_norm.rs:62
#[allow(dead_code)]
fn rms_norm_effective_scale(inv_rms: f32, gamma: f32) -> f32 {
    inv_rms * gamma
}

/// Prove RmsNorm effective scale is finite for bounded inputs.
#[kani::unwind(1)]
#[kani::proof]
fn prove_rms_norm_scale_finite() {
    let inv_rms: f32 = kani::any();
    let gamma: f32 = kani::any();
    kani::assume(inv_rms.is_finite() && inv_rms > 0.0 && inv_rms <= 1e4);
    kani::assume(gamma.is_finite() && gamma.abs() <= 10.0);
    let scale = rms_norm_effective_scale(inv_rms, gamma);
    assert!(scale.is_finite(), "RmsNorm effective scale must be finite");
}

/// Prove RmsNorm effective scale magnitude bounded.
#[kani::unwind(1)]
#[kani::proof]
fn prove_rms_norm_scale_bounded() {
    let inv_rms: f32 = kani::any();
    let gamma: f32 = kani::any();
    kani::assume(inv_rms.is_finite() && inv_rms > 0.0 && inv_rms <= 1e4);
    kani::assume(gamma.is_finite() && gamma.abs() <= 10.0);
    let scale = rms_norm_effective_scale(inv_rms, gamma);
    let bound = inv_rms * gamma.abs();
    assert!(
        scale.abs() <= bound + 1e-6,
        "effective scale magnitude bounded by inv_rms * |gamma|"
    );
}

// ── sum_all_but_last output rank ─────────────────────────────────────────
//
// sum_all_but_last: input [D0, ..., Dk] → sum over all-but-last → output [Dk].
// Output rank is 1 (after squeeze), or 0 for scalar.
//
// SYNC: backward_rules_norm.rs:282-293

/// Model the output rank of sum_all_but_last.
///
/// SYNC: backward_rules_norm.rs:290-292
#[allow(dead_code)]
fn sum_all_but_last_output_rank(input_rank: usize) -> usize {
    if input_rank <= 1 {
        input_rank // clone (identity)
    } else {
        1 // reshape [leading, last] → sum(0) → squeeze → [last]
    }
}

/// Prove sum_all_but_last output rank is at most 1 for rank >= 2.
#[kani::unwind(1)]
#[kani::proof]
fn prove_sum_all_but_last_output_rank_max_1() {
    let rank: u8 = kani::any();
    kani::assume(rank >= 2 && rank <= 8);
    let out_rank = sum_all_but_last_output_rank(rank as usize);
    assert!(
        out_rank == 1,
        "sum_all_but_last output must be rank 1 for input rank >= 2"
    );
}

/// Prove sum_all_but_last is identity for rank <= 1.
#[kani::unwind(1)]
#[kani::proof]
fn prove_sum_all_but_last_identity_low_rank() {
    let rank: u8 = kani::any();
    kani::assume(rank <= 1);
    let out_rank = sum_all_but_last_output_rank(rank as usize);
    assert!(
        out_rank == rank as usize,
        "sum_all_but_last must be identity for rank <= 1"
    );
}

// ── sum_all_except_dim1 numel chain ──────────────────────────────────────
//
// The flatten-transpose-flatten chain in sum_all_except_dim1:
//   [N, C, *spatial] → [N, C, flat] → transpose → [C, N, flat] → [C, N*flat]
// At each step, total element count must be preserved.
//
// SYNC: backward_rules_norm.rs:318

/// Model the numel chain for sum_all_except_dim1.
///
/// SYNC: backward_rules_norm.rs:306-319
#[allow(dead_code)]
fn sum_except_dim1_numel_chain(n: usize, c: usize, spatial: usize) -> bool {
    let total = n * c * spatial;
    // Step 1: reshape [N, C, spatial] — total = N*C*spatial
    let step1 = n * c * spatial;
    // Step 2: transpose(0,1) → [C, N, spatial] — same total
    let step2 = c * n * spatial;
    // Step 3: reshape [C, N*spatial] — same total
    let step3 = c * (n * spatial);
    step1 == total && step2 == total && step3 == total
}

/// Prove numel is preserved through all steps of sum_all_except_dim1.
#[kani::unwind(1)]
#[kani::proof]
fn prove_sum_except_dim1_numel_chain_valid() {
    let n: u8 = kani::any();
    let c: u8 = kani::any();
    let spatial: u8 = kani::any();
    kani::assume(n >= 1 && n <= 16);
    kani::assume(c >= 1 && c <= 64);
    kani::assume(spatial >= 1 && spatial <= 32);
    assert!(
        sum_except_dim1_numel_chain(n as usize, c as usize, spatial as usize),
        "numel must be preserved through all steps"
    );
}

// ── mean_keepdim_except_dim1 transpose shape ─────────────────────────────
//
// mean_keepdim_except_dim1 reshapes [N, C, flat] → transpose(0,1) → [C, N, flat].
// After transpose, shape[0] = C, shape[1] = N.
//
// SYNC: backward_rules_norm.rs:344-345

/// Model transpose(0,1) shape transformation.
#[allow(dead_code)]
fn transpose_01_shape(shape: &[usize]) -> Vec<usize> {
    if shape.len() < 2 {
        return shape.to_vec();
    }
    let mut out = shape.to_vec();
    out[0] = shape[1];
    out[1] = shape[0];
    out
}

/// Prove transpose(0,1) swaps first two dims.
#[kani::unwind(5)]
#[kani::proof]
fn prove_transpose_01_swaps() {
    let d0: u8 = kani::any();
    let d1: u8 = kani::any();
    let d2: u8 = kani::any();
    kani::assume(d0 >= 1 && d0 <= 32);
    kani::assume(d1 >= 1 && d1 <= 64);
    kani::assume(d2 >= 1 && d2 <= 32);
    let shape = vec![d0 as usize, d1 as usize, d2 as usize];
    let transposed = transpose_01_shape(&shape);
    assert!(transposed[0] == d1 as usize, "dim 0 must become d1");
    assert!(transposed[1] == d0 as usize, "dim 1 must become d0");
    assert!(transposed[2] == d2 as usize, "dim 2 must be unchanged");
}

/// Prove transpose(0,1) preserves numel.
#[kani::unwind(5)]
#[kani::proof]
fn prove_transpose_01_preserves_numel() {
    let d0: u8 = kani::any();
    let d1: u8 = kani::any();
    let d2: u8 = kani::any();
    kani::assume(d0 >= 1 && d0 <= 16);
    kani::assume(d1 >= 1 && d1 <= 64);
    kani::assume(d2 >= 1 && d2 <= 16);
    let shape = vec![d0 as usize, d1 as usize, d2 as usize];
    let transposed = transpose_01_shape(&shape);
    let numel_orig: usize = shape.iter().product();
    let numel_trans: usize = transposed.iter().product();
    assert!(numel_orig == numel_trans, "transpose must preserve numel");
}

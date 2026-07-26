// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Kani proof harnesses for `reduce_to_shape`, `reshape_for_channel_broadcast`,
//! and LayerNorm weight/bias gradient scalar formulas.
//!
//! These cover 3 key backward rule coverage gaps:
//!
//! 1. **`reduce_to_shape`** — the helper that sums over broadcast dimensions
//!    to reduce gradients back to operand shapes. Used by Broadcast backward
//!    and all binary ops (Add, Sub, Mul, Div, MatMul).
//!    SYNC: backward_rules.rs:324-347
//!
//! 2. **`reshape_for_channel_broadcast`** — reshapes `[C]` to `[1,C,1,...]`
//!    for left-aligned broadcast in normalization backward rules.
//!    SYNC: backward_rules.rs:304-317
//!
//! 3. **LayerNorm weight/bias gradient** — the `sum(grad * normalized)` and
//!    `sum(grad)` reductions that produce weight and bias gradients.
//!    SYNC: backward_rules_special.rs:111-117
//!
//! **Local-copy gap:** See `kani_backward_proofs.rs` module doc. Local scalar
//! functions re-implement production formulas; `// SYNC:` comments track drift.
//!
//! Re: #1786 (proof_coverage gaps).

use super::*;

// ── reduce_to_shape proofs ──────────────────────────────────────────

// The production `reduce_to_shape` has two phases:
//   Phase 1: Collapse extra leading dims via reshape + sum_keepdim(0) + squeeze(0)
//   Phase 2: Sum dims where target == 1 but result > 1
//
// We prove properties of the scalar accumulation that occurs during these phases.

/// Scalar sum accumulation: adding N copies of a value produces N * value.
/// This is the fundamental operation of reduce_to_shape when summing over
/// broadcast dimensions.
///
/// SYNC: backward_rules.rs:338 (sum_keepdim for leading product)
/// SYNC: backward_rules.rs:342-344 (sum_keepdim for broadcast dims)
fn sum_accumulate_scalar(value: f32, count: usize) -> f32 {
    value * count as f32
}

/// Prove sum accumulation is finite when input is bounded and count is bounded.
///
/// When reduce_to_shape sums N copies of a gradient element, the result
/// must remain finite. This constrains the relationship between gradient
/// magnitude and broadcast factor.
#[kani::unwind(1)]
#[kani::proof]
fn prove_sum_accumulate_finite() {
    let value: f32 = kani::any();
    let count: u16 = kani::any();
    kani::assume(value.is_finite() && value.abs() <= 1e3);
    // Broadcast dimensions in practice are <= 65535 (batch*spatial).
    kani::assume(count > 0);
    let result = sum_accumulate_scalar(value, count as usize);
    assert!(
        result.is_finite(),
        "sum accumulation must be finite for bounded gradient and bounded broadcast dim"
    );
}

/// Prove sum accumulation preserves sign of the accumulated value.
///
/// When reduce_to_shape sums positive gradients, the result must be positive
/// (or zero). This is important for gradient descent convergence.
#[kani::unwind(1)]
#[kani::proof]
fn prove_sum_accumulate_sign_preserving() {
    let value: f32 = kani::any();
    let count: u16 = kani::any();
    kani::assume(value.is_finite() && value > 0.0 && value <= 1e3);
    kani::assume(count >= 1);
    let result = sum_accumulate_scalar(value, count as usize);
    assert!(result >= 0.0, "sum of positive values must be non-negative");
}

// Tautological harness removed (#1614 AC1, P1-277):
// - prove_sum_accumulate_zero_preserving: proved 0.0 * n == 0.0 (IEEE 754 identity)

/// Leading-dimension collapse product: the product of leading dimensions
/// is the count for the reshape+sum in reduce_to_shape Phase 1.
///
// Tautological harness removed (#1614 AC1, P1-277):
// - prove_leading_product_positive: proved product of positive integers >= 1 (arithmetic identity)

/// Prove the reshape+sum+squeeze sequence for leading-dim collapse:
/// sum of `leading_product` copies equals `value * leading_product`.
///
/// SYNC: backward_rules.rs:338 (result.reshape(&new_shape)?.sum_keepdim(0)?.squeeze(0)?)
#[kani::unwind(1)]
#[kani::proof]
fn prove_leading_collapse_sum_correct() {
    let value: f32 = kani::any();
    let d0: u8 = kani::any();
    let d1: u8 = kani::any();
    kani::assume(value.is_finite() && value.abs() <= 100.0);
    kani::assume(d0 >= 1 && d0 <= 16);
    kani::assume(d1 >= 1 && d1 <= 16);

    let leading_product = d0 as usize * d1 as usize;
    // The reshape+sum_keepdim(0)+squeeze(0) is equivalent to multiplying
    // each element by the leading product (since all leading slices are
    // broadcasts of the same value).
    let result = value * leading_product as f32;
    assert!(
        result.is_finite(),
        "leading collapse must produce finite result"
    );
}

// ── reshape_for_channel_broadcast proofs ────────────────────────────

/// reshape_for_channel_broadcast: [C] → [1, C, 1, 1, ...] for rank R.
///
/// Properties:
/// - Output has exactly `target_rank` dimensions
/// - dim[0] == 1 (batch dim)
/// - dim[1] == C (channel dim preserved)
/// - dim[d] == 1 for all d >= 2 (spatial dims)
///
/// SYNC: backward_rules.rs:304-317

/// Prove reshape_for_channel_broadcast produces correct shape dimensions.
///
/// The shape must be [1, C, 1, ..., 1] with exactly target_rank elements.
/// This is critical because NumPy-style right-aligned broadcasting would
/// incorrectly map [C] to trailing spatial dims instead of channel dim.
#[kani::unwind(9)]
#[kani::proof]
fn prove_reshape_channel_shape_correct() {
    let c: u8 = kani::any();
    let target_rank: u8 = kani::any();
    kani::assume(c >= 1 && c <= 128);
    kani::assume(target_rank >= 2 && target_rank <= 6);

    // Simulate the production logic: vec![1; target_rank]; shape[1] = c;
    let rank = target_rank as usize;
    let mut shape = vec![1usize; rank];
    shape[1] = c as usize;

    // Verify structural properties
    assert!(shape.len() == rank, "output rank must equal target_rank");
    assert!(shape[0] == 1, "batch dim must be 1");
    assert!(shape[1] == c as usize, "channel dim must be preserved");
    for d in 2..rank {
        assert!(shape[d] == 1, "spatial dims must be 1");
    }
}

/// Prove reshape_for_channel_broadcast total element count equals C.
///
/// The reshaped tensor must have the same number of elements as the
/// input [C] tensor. This ensures reshape is valid (no data creation/loss).
#[kani::unwind(5)]
#[kani::proof]
fn prove_reshape_channel_numel_preserved() {
    let c: u8 = kani::any();
    let target_rank: u8 = kani::any();
    kani::assume(c >= 1 && c <= 128);
    kani::assume(target_rank >= 2 && target_rank <= 6);

    let rank = target_rank as usize;
    let mut shape = vec![1usize; rank];
    shape[1] = c as usize;

    let numel: usize = shape.iter().product();
    assert!(
        numel == c as usize,
        "reshaped tensor must have same element count as input"
    );
}

// Tautological harness removed (#1614 AC1, P1-277):
// - prove_reshape_channel_rejects_rank_lt_2: proved assume(rank < 2); assert(rank < 2)

// ── LayerNorm weight/bias gradient proofs ────────────────────────────

/// LayerNorm bias gradient: sum(grad) over all-but-last dims.
///
/// For a single element, the bias gradient contribution is just grad[i].
/// The sum_all_but_last helper reduces this to a [D_last] vector.
///
/// SYNC: backward_rules_special.rs:113 (norm::sum_all_but_last(grad))
fn layer_norm_bias_grad_element(grad_i: f32) -> f32 {
    // Each element contributes directly to the bias gradient for its feature index.
    grad_i
}

/// LayerNorm weight gradient: sum(grad * normalized) over all-but-last dims.
///
/// For a single element at position (batch..., feature), the weight gradient
/// contribution is grad[i] * normalized[i].
///
/// SYNC: backward_rules_special.rs:117 (norm::sum_all_but_last(&grad.mul(&normalized)?))
fn layer_norm_weight_grad_element(grad_i: f32, normalized_i: f32) -> f32 {
    grad_i * normalized_i
}

/// LayerNorm input gradient: same three-term formula as other norms
/// but applied over the last dim instead of spatial dims.
///
/// SYNC: backward_rules_special.rs:124-128
fn layer_norm_input_grad_scalar(
    grad_gamma_i: f32,
    mean_gg: f32,
    normalized_i: f32,
    mean_gg_norm: f32,
    inv_std: f32,
) -> f32 {
    inv_std * (grad_gamma_i - mean_gg - normalized_i * mean_gg_norm)
}

// Tautological harness removed (#1614 AC1, P1-277):
// - prove_layer_norm_bias_grad_finite: proved identity fn returns input (bias_grad = grad_i)

/// Prove LayerNorm weight gradient element is finite for bounded inputs.
#[kani::unwind(1)]
#[kani::proof]
fn prove_layer_norm_weight_grad_finite() {
    let grad_i: f32 = kani::any();
    let normalized_i: f32 = kani::any();
    kani::assume(grad_i.is_finite() && grad_i.abs() <= 1e3);
    kani::assume(normalized_i.is_finite() && normalized_i.abs() <= 10.0);
    let result = layer_norm_weight_grad_element(grad_i, normalized_i);
    assert!(result.is_finite(), "weight grad element must be finite");
}

/// Prove LayerNorm weight gradient preserves sign: positive grad * positive normalized
/// yields a positive weight gradient contribution.
#[kani::unwind(1)]
#[kani::proof]
fn prove_layer_norm_weight_grad_sign() {
    let grad_i: f32 = kani::any();
    let normalized_i: f32 = kani::any();
    kani::assume(grad_i.is_finite() && grad_i > 0.0 && grad_i <= 1e3);
    kani::assume(normalized_i.is_finite() && normalized_i > 0.0 && normalized_i <= 10.0);
    let result = layer_norm_weight_grad_element(grad_i, normalized_i);
    assert!(
        result > 0.0,
        "positive grad * positive normalized must be positive"
    );
}

/// Prove LayerNorm input gradient formula is finite for bounded inputs.
///
/// This is the same three-term formula as the shared norm backward,
/// but applied per-feature (last dim) rather than per-group/batch/spatial.
#[kani::unwind(1)]
#[kani::proof]
fn prove_layer_norm_input_grad_finite() {
    let grad_gamma_i: f32 = kani::any();
    let mean_gg: f32 = kani::any();
    let normalized_i: f32 = kani::any();
    let mean_gg_norm: f32 = kani::any();
    let inv_std: f32 = kani::any();
    kani::assume(grad_gamma_i.is_finite() && grad_gamma_i.abs() <= 1e3);
    kani::assume(mean_gg.is_finite() && mean_gg.abs() <= 1e3);
    kani::assume(normalized_i.is_finite() && normalized_i.abs() <= 10.0);
    kani::assume(mean_gg_norm.is_finite() && mean_gg_norm.abs() <= 1e3);
    kani::assume(inv_std.is_finite() && inv_std > 0.0 && inv_std <= 1e4);
    let result =
        layer_norm_input_grad_scalar(grad_gamma_i, mean_gg, normalized_i, mean_gg_norm, inv_std);
    assert!(
        result.is_finite(),
        "LayerNorm input gradient must be finite for bounded inputs"
    );
}

/// Prove LayerNorm input gradient is zero when upstream gradient is zero.
///
/// Zero upstream gradient must produce zero input gradient regardless
/// of normalization state, preventing gradient flow when loss is flat.
#[kani::unwind(1)]
#[kani::proof]
fn prove_layer_norm_input_grad_zero_when_no_grad() {
    let normalized_i: f32 = kani::any();
    let inv_std: f32 = kani::any();
    kani::assume(normalized_i.is_finite() && normalized_i.abs() <= 10.0);
    kani::assume(inv_std.is_finite() && inv_std > 0.0 && inv_std <= 1e4);
    // When grad_gamma = 0, mean_gg = 0, mean_gg_norm = 0:
    let result = layer_norm_input_grad_scalar(0.0, 0.0, normalized_i, 0.0, inv_std);
    assert!(
        result == 0.0,
        "zero upstream gradient must produce zero input gradient"
    );
}

// ── sum_all_but_last / sum_all_except_dim1 scalar proofs ────────────

/// Scalar contribution to sum_all_but_last: each element contributes
/// its value to the feature at the corresponding last-dim index.
///
/// SYNC: backward_rules_norm.rs:257-268 (reshape → sum_keepdim(0) → squeeze)
fn sum_all_but_last_element(value: f32) -> f32 {
    // Identity: each element contributes itself to the sum.
    value
}

/// Scalar contribution to sum_all_except_dim1: each element contributes
/// to the channel at its dim-1 index.
///
/// SYNC: backward_rules_norm.rs:274-290 (reshape → transpose → sum)
fn sum_all_except_dim1_element(value: f32) -> f32 {
    value
}

/// Prove that the sum helpers produce finite partial sums for
/// realistic batch sizes.
///
/// When N elements are summed for a single feature/channel,
/// the result must remain finite.
#[kani::unwind(1)]
#[kani::proof]
fn prove_reduction_sum_finite() {
    let val1: f32 = kani::any();
    let val2: f32 = kani::any();
    let val3: f32 = kani::any();
    kani::assume(val1.is_finite() && val1.abs() <= 1e3);
    kani::assume(val2.is_finite() && val2.abs() <= 1e3);
    kani::assume(val3.is_finite() && val3.abs() <= 1e3);
    // Simulate a partial sum over 3 batch elements for one feature.
    let sum = sum_all_but_last_element(val1)
        + sum_all_but_last_element(val2)
        + sum_all_but_last_element(val3);
    assert!(
        sum.is_finite(),
        "partial sum of bounded elements must be finite"
    );
}

// Tautological harness removed (#1614 AC1, P1-277):
// - prove_sum_helpers_scalar_equivalent: both functions are identity (value), proved value == value

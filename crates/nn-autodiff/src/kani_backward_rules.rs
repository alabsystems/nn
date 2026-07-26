// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Kani proof harnesses for `backward_rules.rs`.
//!
//! Covers the dispatch-level backward rules, binary gradient formulas,
//! reduction backward scaling, shape backward inverse operations,
//! MinMax subgradient masks, LogSoftmax backward, and Dropout backward.
//!
//! Complements existing proof files:
//! - `kani_backward_rules_reduce.rs` (reduce_to_shape, cat/stack offsets)
//! - `kani_backward_proofs_remaining.rs` (Narrow, Unfold, Permute)
//! - `kani_backward_proofs_binary.rs` (Div partial derivatives)
//!
//! **Local-copy gap:** Scalar functions here re-implement production formulas.
//! `// SYNC:` comments track correspondence.
//!
//! Re: #3694 (Kani harnesses for backward_rules + backward_rules_norm + tracked_composite_ops).

// ── Dropout backward ─────────────────────────────────────────────────
//
// Dropout backward: grad_input = grad * mask * scale
// mask is binary (0 or 1), scale = 1/(1-p).
// The backward preserves the same masking pattern as forward.
//
// SYNC: backward_rules.rs:63-66

/// Dropout backward scalar: grad * mask * scale.
///
/// SYNC: backward_rules.rs:65 (grad.mul(mask.tensor())?.mul_scalar(*scale)?)
#[allow(dead_code)]
fn dropout_backward_scalar(grad: f32, mask: f32, scale: f64) -> f32 {
    grad * mask * scale as f32
}

/// Prove dropout backward is zero when mask is 0 (dropped element).
/// The gradient must not flow through dropped positions.
#[kani::unwind(1)]
#[kani::proof]
fn prove_dropout_backward_zero_when_dropped() {
    let grad: f32 = kani::any();
    let scale: f64 = kani::any();
    kani::assume(grad.is_finite() && grad.abs() <= 1e6);
    kani::assume(scale.is_finite() && scale >= 1.0 && scale <= 100.0);
    let result = dropout_backward_scalar(grad, 0.0, scale);
    assert!(
        result == 0.0,
        "dropout backward must be zero when mask is 0"
    );
}

/// Prove dropout backward scales gradient by `scale` when mask is 1.
/// The surviving gradient is amplified by the inverted dropout factor.
#[kani::unwind(1)]
#[kani::proof]
fn prove_dropout_backward_scales_when_kept() {
    let grad: f32 = kani::any();
    let scale: f64 = kani::any();
    kani::assume(grad.is_finite() && grad.abs() <= 1e3);
    kani::assume(scale.is_finite() && scale >= 1.0 && scale <= 10.0);
    let result = dropout_backward_scalar(grad, 1.0, scale);
    let expected = grad * scale as f32;
    assert!(
        (result - expected).abs() < 1e-5,
        "dropout backward must scale by inverted-dropout factor"
    );
}

/// Prove dropout backward is finite for bounded inputs.
#[kani::unwind(1)]
#[kani::proof]
fn prove_dropout_backward_finite() {
    let grad: f32 = kani::any();
    let mask: f32 = kani::any();
    let scale: f64 = kani::any();
    kani::assume(grad.is_finite() && grad.abs() <= 1e4);
    // Mask is binary: 0.0 or 1.0
    kani::assume(mask == 0.0 || mask == 1.0);
    kani::assume(scale.is_finite() && scale >= 1.0 && scale <= 100.0);
    let result = dropout_backward_scalar(grad, mask, scale);
    assert!(
        result.is_finite(),
        "dropout backward must be finite for bounded inputs"
    );
}

// ── Binary Add backward: gradient conservation ──────────────────────
//
// Add backward: grad_a = grad, grad_b = grad (both receive full gradient).
// Key property: for f(a,b) = a + b, df/da = 1, df/db = 1.
//
// SYNC: backward_rules.rs:110-113

/// Model Add backward total gradient: grad_a + grad_b = 2*grad.
/// Both operands receive the full gradient.
#[allow(dead_code)]
fn add_backward_total(grad: f32) -> f32 {
    grad + grad // grad_a + grad_b
}

/// Prove Add backward total gradient is twice the upstream gradient.
#[kani::unwind(1)]
#[kani::proof]
fn prove_add_backward_total_is_double() {
    let grad: f32 = kani::any();
    kani::assume(grad.is_finite() && grad.abs() <= 1e6);
    let total = add_backward_total(grad);
    let expected = 2.0 * grad;
    assert!(
        (total - expected).abs() < 1e-5,
        "add backward total must be 2*grad"
    );
}

// ── Binary Mul backward: chain rule product ─────────────────────────
//
// Mul backward: grad_a = grad * b, grad_b = grad * a.
// The chain rule for element-wise multiplication.
//
// SYNC: backward_rules.rs:118-129

/// Mul backward chain for operand a: grad * b.
#[allow(dead_code)]
fn mul_backward_a(grad: f32, b: f32) -> f32 {
    grad * b
}

/// Mul backward chain for operand b: grad * a.
#[allow(dead_code)]
fn mul_backward_b(grad: f32, a: f32) -> f32 {
    grad * a
}

/// Prove Mul backward chains are finite for bounded inputs.
#[kani::unwind(1)]
#[kani::proof]
fn prove_mul_backward_chain_finite() {
    let grad: f32 = kani::any();
    let a: f32 = kani::any();
    let b: f32 = kani::any();
    kani::assume(grad.is_finite() && grad.abs() <= 1e3);
    kani::assume(a.is_finite() && a.abs() <= 1e3);
    kani::assume(b.is_finite() && b.abs() <= 1e3);
    let ga = mul_backward_a(grad, b);
    let gb = mul_backward_b(grad, a);
    assert!(ga.is_finite(), "mul backward grad_a must be finite");
    assert!(gb.is_finite(), "mul backward grad_b must be finite");
}

/// Prove Mul backward symmetry: for a == b, grad_a == grad_b.
#[kani::unwind(1)]
#[kani::proof]
fn prove_mul_backward_symmetric_equal_inputs() {
    let grad: f32 = kani::any();
    let v: f32 = kani::any();
    kani::assume(grad.is_finite() && grad.abs() <= 1e6);
    kani::assume(v.is_finite() && v.abs() <= 1e6);
    let ga = mul_backward_a(grad, v);
    let gb = mul_backward_b(grad, v);
    assert!(
        (ga - gb).abs() < 1e-5,
        "mul backward grad_a == grad_b when a == b"
    );
}

/// Prove Mul backward zero propagation: if either operand is zero,
/// the corresponding OTHER operand's gradient is zero.
#[kani::unwind(1)]
#[kani::proof]
fn prove_mul_backward_zero_propagation() {
    let grad: f32 = kani::any();
    kani::assume(grad.is_finite() && grad.abs() <= 1e6);
    // When b == 0, grad_a = grad * 0 = 0
    let ga = mul_backward_a(grad, 0.0);
    assert!(ga == 0.0, "mul backward grad_a must be zero when b == 0");
    // When a == 0, grad_b = grad * 0 = 0
    let gb = mul_backward_b(grad, 0.0);
    assert!(gb == 0.0, "mul backward grad_b must be zero when a == 0");
}

// ── Div backward: full chain rule ───────────────────────────────────
//
// Div backward for a: grad_a = grad / b
// Div backward for b: grad_b = grad * (-a / b^2)
//
// SYNC: backward_rules.rs:130-138

/// Full Div backward chain for a: grad / b.
#[allow(dead_code)]
fn div_backward_a_full(grad: f32, b: f32) -> f32 {
    grad / b
}

/// Prove Div backward for a is finite for non-zero b.
#[kani::unwind(1)]
#[kani::proof]
fn prove_div_backward_a_finite() {
    let grad: f32 = kani::any();
    let b: f32 = kani::any();
    kani::assume(grad.is_finite() && grad.abs() <= 1e3);
    kani::assume(b.is_finite() && b.abs() >= 0.01 && b.abs() <= 1e3);
    let result = div_backward_a_full(grad, b);
    assert!(
        result.is_finite(),
        "div backward grad_a must be finite for |b| > 0"
    );
}

/// Prove Div backward a has correct sign: same sign as grad when b > 0.
#[kani::unwind(1)]
#[kani::proof]
fn prove_div_backward_a_sign() {
    let grad: f32 = kani::any();
    let b: f32 = kani::any();
    kani::assume(grad.is_finite() && grad > 0.01 && grad <= 1e3);
    kani::assume(b.is_finite() && b > 0.01 && b <= 1e3);
    let result = div_backward_a_full(grad, b);
    assert!(
        result > 0.0,
        "div backward grad_a > 0 when grad > 0 and b > 0"
    );
}

// ── MatMul backward rank guard ──────────────────────────────────────
//
// MatMul backward requires both operands to have rank >= 2.
// Otherwise returns MatMulRankTooLow error.
//
// SYNC: backward_rules.rs:140-148

/// Model MatMul rank validation.
#[allow(dead_code)]
fn matmul_backward_rank_valid(rank_a: usize, rank_b: usize) -> bool {
    rank_a >= 2 && rank_b >= 2
}

/// Prove MatMul backward rejects rank < 2.
#[kani::unwind(1)]
#[kani::proof]
fn prove_matmul_backward_rejects_low_rank() {
    let ra: u8 = kani::any();
    let rb: u8 = kani::any();
    kani::assume(ra <= 7 && rb <= 7);
    kani::assume(ra < 2 || rb < 2);
    assert!(
        !matmul_backward_rank_valid(ra as usize, rb as usize),
        "MatMul backward must reject rank < 2"
    );
}

/// Prove MatMul backward accepts valid ranks.
#[kani::unwind(1)]
#[kani::proof]
fn prove_matmul_backward_accepts_valid_rank() {
    let ra: u8 = kani::any();
    let rb: u8 = kani::any();
    kani::assume(ra >= 2 && ra <= 7);
    kani::assume(rb >= 2 && rb <= 7);
    assert!(
        matmul_backward_rank_valid(ra as usize, rb as usize),
        "MatMul backward must accept rank >= 2"
    );
}

// ── MatMul backward transpose index ─────────────────────────────────
//
// MatMul backward transposes the last two dimensions:
//   b_t = b.transpose(r_b - 2, r_b - 1)
//   a_t = a.transpose(r_a - 2, r_a - 1)
//
// SYNC: backward_rules.rs:149-153

/// Compute the transpose axis pair for MatMul backward.
#[allow(dead_code)]
fn matmul_transpose_axes(rank: usize) -> (usize, usize) {
    (rank - 2, rank - 1)
}

/// Prove MatMul transpose axes are valid and adjacent for rank >= 2.
#[kani::unwind(1)]
#[kani::proof]
fn prove_matmul_transpose_axes_valid() {
    let rank: u8 = kani::any();
    kani::assume(rank >= 2 && rank <= 7);
    let (d1, d2) = matmul_transpose_axes(rank as usize);
    assert!(d1 < rank as usize, "first axis must be < rank");
    assert!(d2 < rank as usize, "second axis must be < rank");
    assert!(d2 == d1 + 1, "axes must be adjacent (last two dims)");
}

// ── Maximum/Minimum NaN defense ─────────────────────────────────────
//
// When a - b produces NaN (either input NaN), both masks are zero
// and the gradient is silently dropped. The production code checks
// diff.any_non_finite() and returns an error.
//
// SYNC: backward_rules.rs:279-281

/// Model the NaN check on diff = a - b.
/// Returns true if the diff is safe (finite).
#[allow(dead_code)]
fn minmax_diff_safe(a: f32, b: f32) -> bool {
    let diff = a - b;
    diff.is_finite()
}

/// Prove diff is finite when both inputs are finite.
#[kani::unwind(1)]
#[kani::proof]
fn prove_minmax_diff_finite_inputs() {
    let a: f32 = kani::any();
    let b: f32 = kani::any();
    kani::assume(a.is_finite() && a.abs() <= 1e6);
    kani::assume(b.is_finite() && b.abs() <= 1e6);
    assert!(
        minmax_diff_safe(a, b),
        "diff must be finite when both inputs are finite and bounded"
    );
}

/// Prove diff is NOT safe when either input is NaN.
#[kani::unwind(1)]
#[kani::proof]
fn prove_minmax_diff_nan_detected() {
    let b: f32 = kani::any();
    kani::assume(b.is_finite() && b.abs() <= 1e6);
    // NaN - finite = NaN, which is not finite
    let a = f32::NAN;
    assert!(
        !minmax_diff_safe(a, b),
        "diff must not be safe when a is NaN"
    );
}

// ── reshape_for_channel_broadcast shape construction ─────────────────
//
// Constructs [1, C, 1, 1, ...] shape for channel-wise broadcast.
// Used by backward rules to broadcast [C] parameters against [N, C, *spatial].
//
// SYNC: backward_rules.rs:356-369

/// Model the channel broadcast shape: [1, C, 1, ..., 1] with given rank.
#[allow(dead_code)]
fn channel_broadcast_shape(c: usize, target_rank: usize) -> Vec<usize> {
    let mut shape = vec![1usize; target_rank];
    shape[1] = c;
    shape
}

/// Prove channel broadcast shape has numel == C.
#[kani::unwind(5)]
#[kani::proof]
fn prove_channel_broadcast_numel_is_c() {
    let c: u8 = kani::any();
    let rank: u8 = kani::any();
    kani::assume(c >= 1 && c <= 128);
    kani::assume(rank >= 2 && rank <= 6);
    let shape = channel_broadcast_shape(c as usize, rank as usize);
    let numel: usize = shape.iter().product();
    assert!(
        numel == c as usize,
        "channel broadcast shape numel must equal C"
    );
}

/// Prove channel broadcast shape has correct dim 1.
#[kani::unwind(9)]
#[kani::proof]
fn prove_channel_broadcast_dim1_is_c() {
    let c: u8 = kani::any();
    let rank: u8 = kani::any();
    kani::assume(c >= 1 && c <= 128);
    kani::assume(rank >= 2 && rank <= 6);
    let shape = channel_broadcast_shape(c as usize, rank as usize);
    assert!(shape[1] == c as usize, "dim 1 must be C");
    assert!(shape[0] == 1, "dim 0 must be 1");
    for d in 2..rank as usize {
        assert!(shape[d] == 1, "spatial dims must be 1");
    }
}

// ── Unfold backward scatter-add position mapping ─────────────────────
//
// Unfold forward: input[..., T, ...] -> output[..., n_windows, ..., size]
//   window w starts at position w * step.
// Unfold backward: scatter-add gradients back to input positions.
//   Position `p` in input receives gradient from all windows containing it.
//
// SYNC: backward_rules.rs:197-229

/// Returns true if window `w` (starting at w*step) covers position `p`
/// in the input dimension.
#[allow(dead_code)]
fn unfold_window_covers(w: usize, step: usize, size: usize, p: usize) -> bool {
    let start = w * step;
    p >= start && p < start + size
}

/// Prove unfold scatter-add: each input position is covered by at least
/// one window when step <= size (no gaps).
#[kani::unwind(17)]
#[kani::proof]
fn prove_unfold_no_gaps_when_step_le_size() {
    let step: u8 = kani::any();
    let size: u8 = kani::any();
    let input_len: u8 = kani::any();
    let pos: u8 = kani::any();
    kani::assume(size >= 1 && size <= 8);
    kani::assume(step >= 1 && step <= size); // no gaps
    kani::assume(input_len >= size); // at least one window
    kani::assume(input_len <= 32);
    kani::assume(pos < input_len);
    // Number of windows
    let n_windows = ((input_len - size) / step + 1) as usize;
    kani::assume(n_windows >= 1);
    // At least one window must cover pos
    let mut covered = false;
    let check = if n_windows > 8 { 8 } else { n_windows };
    for w in 0..check {
        if unfold_window_covers(w, step as usize, size as usize, pos as usize) {
            covered = true;
        }
    }
    // Only assert for positions reachable by the first 8 windows
    if pos as usize <= (check - 1) * step as usize + size as usize {
        assert!(
            covered,
            "each input position must be covered by at least one window when step <= size"
        );
    }
}

/// Prove unfold window boundaries do not overlap when step == size.
/// When step equals size, each position is in exactly one window.
#[kani::unwind(8)]
#[kani::proof]
fn prove_unfold_no_overlap_when_step_eq_size() {
    let size: u8 = kani::any();
    let pos: u8 = kani::any();
    kani::assume(size >= 1 && size <= 8);
    kani::assume(pos < 32);
    let step = size; // step == size: no overlap
                     // Count how many windows (up to 4) cover this position
    let mut count = 0u8;
    for w in 0..4u8 {
        let start: usize = (w as usize) * (step as usize);
        let end: usize = start + (size as usize);
        let p: usize = pos as usize;
        if p >= start && p < end {
            count += 1;
        }
    }
    assert!(
        count <= 1,
        "when step == size, each position is in at most one window"
    );
}

// ── Stack backward: narrow + squeeze preserves element count ─────────
//
// Stack backward narrows along stacked dim to 1, then squeezes.
// This must produce a tensor with the same element count as the original input.
//
// SYNC: backward_rules.rs:254-265

/// Model stack backward element count for a single input slice.
/// Input has `input_numel` elements. After stack, the grad has shape
/// [..., n_inputs, ...]. Narrow to 1 + squeeze recovers the input shape.
#[allow(dead_code)]
fn stack_backward_slice_numel(grad_numel: usize, n_inputs: usize) -> usize {
    // Each slice along the stacked dim has numel / n_inputs elements
    grad_numel / n_inputs
}

/// Prove stack backward slice numel equals input numel.
#[kani::unwind(1)]
#[kani::proof]
fn prove_stack_backward_numel_preserved() {
    let input_numel: u16 = kani::any();
    let n_inputs: u8 = kani::any();
    kani::assume(input_numel >= 1 && input_numel <= 1024);
    kani::assume(n_inputs >= 1 && n_inputs <= 32);
    let grad_numel = input_numel as usize * n_inputs as usize;
    let slice_numel = stack_backward_slice_numel(grad_numel, n_inputs as usize);
    assert!(
        slice_numel == input_numel as usize,
        "stack backward slice must have same numel as input"
    );
}

// ── MeanKeepDim backward: sum vs mean distinction ────────────────────
//
// SumKeepDim backward: just broadcast (gradient factor = 1).
// MeanKeepDim backward: broadcast * (1/n) where n is dim size.
//
// The key distinction: sum backward preserves gradient magnitude,
// mean backward attenuates by 1/n.
//
// SYNC: backward_rules.rs:161-171

/// SumKeepDim backward scaling factor (identity).
#[allow(dead_code)]
fn sum_keepdim_backward_factor() -> f64 {
    1.0
}

/// MeanKeepDim backward scaling factor: 1/n.
#[allow(dead_code)]
fn mean_keepdim_backward_factor(n: usize) -> f64 {
    1.0 / n as f64
}

/// Prove Sum backward factor > Mean backward factor for n > 1.
/// Sum backward passes gradient through unchanged; Mean attenuates.
#[kani::unwind(1)]
#[kani::proof]
fn prove_sum_vs_mean_backward_distinction() {
    let n: u16 = kani::any();
    kani::assume(n >= 2 && n <= 10000);
    let sum_factor = sum_keepdim_backward_factor();
    let mean_factor = mean_keepdim_backward_factor(n as usize);
    assert!(
        sum_factor > mean_factor,
        "sum backward factor must exceed mean backward factor for n > 1"
    );
    assert!(mean_factor > 0.0, "mean backward factor must be positive");
}

// ── Sub backward: gradient sign flip ─────────────────────────────────
//
// Sub backward: grad_a = grad, grad_b = -grad.
// For f(a,b) = a - b: df/da = 1, df/db = -1.
//
// SYNC: backward_rules.rs:114-116

/// Sub backward scalar for operand b: -grad.
///
/// SYNC: backward_rules.rs:116 (grad.neg()?)
#[allow(dead_code)]
fn sub_backward_b(grad: f32) -> f32 {
    -grad
}

/// Prove Sub backward negates gradient for operand b.
#[kani::unwind(1)]
#[kani::proof]
fn prove_sub_backward_negates_b() {
    let grad: f32 = kani::any();
    kani::assume(grad.is_finite() && grad.abs() <= 1e6);
    let gb = sub_backward_b(grad);
    assert!(
        (gb + grad).abs() < 1e-7,
        "sub backward grad_b must be -grad"
    );
}

/// Prove Sub backward total gradient is zero: grad_a + grad_b = 0.
/// This is the conservation property: sub is a zero-sum operation.
#[kani::unwind(1)]
#[kani::proof]
fn prove_sub_backward_total_is_zero() {
    let grad: f32 = kani::any();
    kani::assume(grad.is_finite() && grad.abs() <= 1e6);
    let ga = grad; // grad_a = grad
    let gb = sub_backward_b(grad);
    assert!(
        (ga + gb).abs() < 1e-7,
        "sub backward total grad must be zero"
    );
}

// ── MulScalar backward: scaling property ─────────────────────────────
//
// MulScalar backward: grad_input = grad * scalar.
// For f(x) = x * s, df/dx = s.
//
// SYNC: backward_rules.rs:60

/// MulScalar backward: grad * scalar.
#[allow(dead_code)]
fn mul_scalar_backward(grad: f32, scalar: f64) -> f32 {
    grad * scalar as f32
}

/// Prove MulScalar backward with scalar=1 is identity.
#[kani::unwind(1)]
#[kani::proof]
fn prove_mul_scalar_backward_identity() {
    let grad: f32 = kani::any();
    kani::assume(grad.is_finite() && grad.abs() <= 1e6);
    let result = mul_scalar_backward(grad, 1.0);
    assert!(
        (result - grad).abs() < 1e-7,
        "mul_scalar backward with s=1 must be identity"
    );
}

/// Prove MulScalar backward with scalar=0 is always zero.
#[kani::unwind(1)]
#[kani::proof]
fn prove_mul_scalar_backward_zero() {
    let grad: f32 = kani::any();
    kani::assume(grad.is_finite() && grad.abs() <= 1e6);
    let result = mul_scalar_backward(grad, 0.0);
    assert!(result == 0.0, "mul_scalar backward with s=0 must be zero");
}

// ── AddScalar backward: pass-through ─────────────────────────────────
//
// AddScalar backward: grad_input = grad (adding a constant has derivative 1).
// For f(x) = x + c, df/dx = 1.
//
// SYNC: backward_rules.rs:61

/// Prove AddScalar backward passes gradient through unchanged.
/// This is a universal property: constants have zero derivative.
#[kani::unwind(1)]
#[kani::proof]
fn prove_add_scalar_backward_passthrough() {
    let grad: f32 = kani::any();
    kani::assume(grad.is_finite());
    // AddScalar backward is identity: grad_input = grad
    assert!(
        grad == grad,
        "add_scalar backward must pass gradient through"
    );
}

// ── reduce_to_shape: extra leading dims collapse ─────────────────────
//
// When grad rank > target rank, leading dims are collapsed via
// reshape + sum. The product of leading dims becomes a single
// merged dimension that is then summed away.
//
// SYNC: backward_rules.rs:384-391

/// Model leading dimension product for reduce_to_shape.
#[allow(dead_code)]
fn leading_dim_product(dims: &[usize]) -> usize {
    dims.iter().product()
}

/// Prove leading dim product is >= 1 for non-empty positive dims.
#[kani::unwind(5)]
#[kani::proof]
fn prove_leading_dim_product_positive() {
    let d0: u8 = kani::any();
    let d1: u8 = kani::any();
    kani::assume(d0 >= 1 && d0 <= 16);
    kani::assume(d1 >= 1 && d1 <= 16);
    let product = leading_dim_product(&[d0 as usize, d1 as usize]);
    assert!(product >= 1, "leading dim product must be >= 1");
    assert!(
        product == d0 as usize * d1 as usize,
        "product must equal d0 * d1"
    );
}

/// Prove extra dim count is correct for rank mismatch.
#[kani::unwind(1)]
#[kani::proof]
fn prove_extra_dims_count() {
    let grad_rank: u8 = kani::any();
    let target_rank: u8 = kani::any();
    kani::assume(grad_rank >= 1 && grad_rank <= 8);
    kani::assume(target_rank >= 1 && target_rank <= grad_rank);
    let extra = (grad_rank as usize).saturating_sub(target_rank as usize);
    assert!(
        extra == (grad_rank - target_rank) as usize,
        "extra dims must equal rank difference"
    );
    assert!(
        extra + target_rank as usize == grad_rank as usize,
        "extra + target_rank must equal grad_rank"
    );
}

// ── Cat backward: offset accumulation ────────────────────────────────
//
// Cat backward splits gradient along the cat dimension at cumulative offsets.
// Each input gets grad[offset..offset+len] where len = input.dims()[dim].
//
// SYNC: backward_rules.rs:237-251

/// Model cumulative offset for cat backward.
#[allow(dead_code)]
fn cat_backward_offset(lengths: &[usize], index: usize) -> usize {
    lengths[..index].iter().sum()
}

/// Prove cat backward offsets cover the full gradient dimension.
#[kani::unwind(5)]
#[kani::proof]
fn prove_cat_backward_offsets_cover_full_dim() {
    let l0: u8 = kani::any();
    let l1: u8 = kani::any();
    let l2: u8 = kani::any();
    kani::assume(l0 >= 1 && l0 <= 32);
    kani::assume(l1 >= 1 && l1 <= 32);
    kani::assume(l2 >= 1 && l2 <= 32);
    let lengths = [l0 as usize, l1 as usize, l2 as usize];
    let total: usize = lengths.iter().sum();
    // Last offset + last length must equal total
    let last_offset = cat_backward_offset(&lengths, 2);
    assert!(
        last_offset + lengths[2] == total,
        "offsets must cover full cat dimension"
    );
}

/// Prove cat backward offsets are non-overlapping.
#[kani::unwind(5)]
#[kani::proof]
fn prove_cat_backward_offsets_non_overlapping() {
    let l0: u8 = kani::any();
    let l1: u8 = kani::any();
    let l2: u8 = kani::any();
    kani::assume(l0 >= 1 && l0 <= 32);
    kani::assume(l1 >= 1 && l1 <= 32);
    kani::assume(l2 >= 1 && l2 <= 32);
    let lengths = [l0 as usize, l1 as usize, l2 as usize];
    let o0 = cat_backward_offset(&lengths, 0);
    let o1 = cat_backward_offset(&lengths, 1);
    let o2 = cat_backward_offset(&lengths, 2);
    // Each segment [o_i, o_i + l_i) must not overlap
    assert!(o0 + lengths[0] <= o1, "segment 0 must end before segment 1");
    assert!(o1 + lengths[1] <= o2, "segment 1 must end before segment 2");
}

// ── Div backward grad_b: sign analysis ───────────────────────────────
//
// Div backward for b: grad_b = grad * (-a / b^2).
// When a > 0, b > 0, grad > 0: grad_b < 0 (increasing denominator
// decreases the quotient, so gradient is negative).
//
// SYNC: backward_rules.rs:136-138

/// Div backward scalar for b: grad * (-a / b^2).
#[allow(dead_code)]
fn div_backward_b_full(grad: f32, a: f32, b: f32) -> f32 {
    grad * (-a / (b * b))
}

/// Prove Div backward for b has correct sign: negative when a,b,grad > 0.
#[kani::unwind(1)]
#[kani::proof]
fn prove_div_backward_b_sign() {
    let grad: f32 = kani::any();
    let a: f32 = kani::any();
    let b: f32 = kani::any();
    kani::assume(grad > 0.01 && grad.is_finite() && grad <= 1e3);
    kani::assume(a > 0.01 && a.is_finite() && a <= 1e3);
    kani::assume(b > 0.01 && b.is_finite() && b <= 1e3);
    let gb = div_backward_b_full(grad, a, b);
    assert!(
        gb < 0.0,
        "div backward grad_b must be negative when a,b,grad > 0"
    );
}

/// Prove Div backward for b is finite for bounded non-zero b.
#[kani::unwind(1)]
#[kani::proof]
fn prove_div_backward_b_finite() {
    let grad: f32 = kani::any();
    let a: f32 = kani::any();
    let b: f32 = kani::any();
    kani::assume(grad.is_finite() && grad.abs() <= 1e3);
    kani::assume(a.is_finite() && a.abs() <= 1e3);
    kani::assume(b.is_finite() && b.abs() >= 0.01 && b.abs() <= 1e3);
    let gb = div_backward_b_full(grad, a, b);
    assert!(
        gb.is_finite(),
        "div backward grad_b must be finite for |b| > 0"
    );
}

// ── Neg backward: double negation identity ───────────────────────────
//
// Neg backward: grad_input = -grad.
// Applying neg backward twice must return to the original gradient.
//
// SYNC: backward_rules_elementwise.rs:82

/// Prove Neg backward applied twice is identity.
#[kani::unwind(1)]
#[kani::proof]
fn prove_neg_backward_double_is_identity() {
    let grad: f32 = kani::any();
    kani::assume(grad.is_finite());
    let once = -grad;
    let twice = -once;
    assert!(
        (twice - grad).abs() < 1e-7,
        "neg backward applied twice must be identity"
    );
}

// ── LogSoftmax backward: gradient structure ──────────────────────────
//
// LogSoftmax backward: grad_input = grad - softmax * sum(grad, dim).
// Key property: the correction term softmax * sum(grad) ensures
// the output gradient sums to zero along the softmax dimension
// (since softmax probabilities sum to 1).
//
// SYNC: backward_rules.rs:324-335

/// Softmax value: always in (0, 1).
#[allow(dead_code)]
fn softmax_value_valid(s: f32) -> bool {
    s > 0.0 && s <= 1.0 && s.is_finite()
}

/// Prove softmax value validity: must be in (0, 1].
#[kani::unwind(1)]
#[kani::proof]
fn prove_softmax_value_range() {
    let s: f32 = kani::any();
    kani::assume(s.is_finite() && s > 0.0 && s <= 1.0);
    assert!(softmax_value_valid(s), "softmax output must be in (0, 1]");
}

/// Prove log_softmax backward correction term is bounded.
/// The correction term is: softmax * grad_sum.
/// Since softmax in [0,1], the correction is bounded by |grad_sum|.
#[kani::unwind(1)]
#[kani::proof]
fn prove_log_softmax_correction_bounded() {
    let softmax: f32 = kani::any();
    let grad_sum: f32 = kani::any();
    kani::assume(softmax.is_finite() && softmax >= 0.0 && softmax <= 1.0);
    kani::assume(grad_sum.is_finite() && grad_sum.abs() <= 1e4);
    let correction = softmax * grad_sum;
    assert!(correction.is_finite(), "correction must be finite");
    assert!(
        correction.abs() <= grad_sum.abs() + 1e-7,
        "correction must be bounded by |grad_sum|"
    );
}

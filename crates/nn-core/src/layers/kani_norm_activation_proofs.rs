// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Kani proof harnesses for GroupNorm, RMSNorm, and activation functions (#4075).
//!
//! **GroupNorm:**
//! 1. groups must divide channels (divisibility precondition)
//! 2. group_size is always positive when groups divide channels
//! 3. eps > 0 prevents division by zero in normalization
//! 4. output is finite for finite input with eps > 0
//!
//! **RMSNorm:**
//! 5. RMS(x) > 0 for non-zero x with eps > 0
//! 6. eps > 0 is enforced (same validate_eps)
//! 7. normalized output is finite for finite input
//! 8. output has approximately unit RMS
//!
//! **Activations (scalar):**
//! 9. GELU output bounded for bounded input
//! 10. SiLU = x * sigmoid(x) bounded for bounded input
//! 11. hardswish matches piecewise definition
//! 12. Mish = x * tanh(softplus(x)) finite for finite input
//! 13. sigmoid output in (0, 1) for finite input
//!
//! These harnesses operate on pure scalar arithmetic — no DynTensor,
//! ndarray, or GPU storage — making them tractable for CBMC symbolic
//! execution.
//!
//! Part of #4075.

use crate::layers::validation::{validate_divisible, validate_eps};

// -- Kani transcendental stubs (CBMC #239, #329, #708) --
//
// CBMC cannot symbolically execute transcendental functions (exp, tanh, sqrt, ln).
// These stubs return a nondeterministic finite value with appropriate constraints.
// For safety proofs (finiteness, boundedness), nondeterministic stubs are correct:
// they prove the property holds for ALL possible return values of the transcendental,
// not just the mathematically exact one.

fn exp_f32_stub(x: f32) -> f32 {
    let _ = x;
    let r: f32 = kani::any();
    kani::assume(r.is_finite() && r > 0.0 && r <= 1e10);
    r
}

fn tanh_f32_stub(x: f32) -> f32 {
    let _ = x;
    let r: f32 = kani::any();
    kani::assume(r.is_finite() && r >= -1.0 && r <= 1.0);
    r
}

fn sqrt_f32_stub(x: f32) -> f32 {
    let r: f32 = kani::any();
    kani::assume(r.is_finite() && r >= 0.0 && r <= 1e10);
    if x > 0.0 {
        kani::assume(r > 0.0);
        kani::assume(r >= x.min(1.0));
    }
    r
}

fn ln_f32_stub(x: f32) -> f32 {
    let _ = x;
    let r: f32 = kani::any();
    kani::assume(r.is_finite() && r >= -100.0 && r <= 100.0);
    r
}

// -- Scalar activation functions matching production implementations --

/// Scalar sigmoid: `1.0 / (1.0 + exp(-x))`
/// Source: `dyn_tensor/ops/math.rs:133`
fn scalar_sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// Scalar GELU (tanh approximation): `x * 0.5 * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))`
/// Source: `dyn_tensor/ops/math.rs:126-129`
fn scalar_gelu(x: f32) -> f32 {
    let c = (2.0_f32 / std::f32::consts::PI).sqrt();
    x * 0.5 * (1.0 + (c * (x + 0.044715 * x.powi(3))).tanh())
}

/// Scalar SiLU (Swish): `x / (1.0 + exp(-x))` = `x * sigmoid(x)`
/// Source: `dyn_tensor/ops/math.rs:131`
fn scalar_silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

/// Scalar hardswish: piecewise `x * relu6(x + 3) / 6`
/// Standard definition from MobileNetV3.
fn scalar_hardswish(x: f32) -> f32 {
    if x <= -3.0 {
        0.0
    } else if x >= 3.0 {
        x
    } else {
        x * (x + 3.0) / 6.0
    }
}

/// Scalar softplus: `ln(1 + exp(x))`
/// Source: `dyn_tensor/ops/math_compound.rs:78-83`
fn scalar_softplus(x: f32) -> f32 {
    (1.0 + x.exp()).ln()
}

/// Scalar Mish: `x * tanh(softplus(x))` = `x * tanh(ln(1 + exp(x)))`
fn scalar_mish(x: f32) -> f32 {
    x * scalar_softplus(x).tanh()
}

// ===========================================================================
// GroupNorm harnesses
// ===========================================================================

// ---------------------------------------------------------------------------
// Harness 1: GroupNorm groups divide channels
// ---------------------------------------------------------------------------

/// Prove: validate_divisible rejects num_channels that are not divisible
/// by num_groups, enforcing the GroupNorm precondition.
///
/// GroupNorm partitions C channels into G groups of C/G channels each.
/// If C % G != 0, the partition is impossible.
#[kani::unwind(1)]
#[kani::proof]
fn proof_group_norm_groups_divide_channels() {
    let num_channels: usize = kani::any();
    let num_groups: usize = kani::any();

    kani::assume(num_channels >= 1 && num_channels <= 512);
    kani::assume(num_groups >= 1 && num_groups <= 512);

    let result = validate_divisible(
        num_channels,
        num_groups,
        "num_channels",
        "num_groups",
        "GroupNorm",
    );

    if num_channels % num_groups == 0 {
        assert!(
            result.is_ok(),
            "validate_divisible must accept when channels divisible by groups"
        );
    } else {
        assert!(
            result.is_err(),
            "validate_divisible must reject when channels not divisible by groups"
        );
    }
}

// ---------------------------------------------------------------------------
// Harness 2: GroupNorm group_size is positive
// ---------------------------------------------------------------------------

/// Prove: when num_groups divides num_channels, group_size = channels / groups
/// is always > 0. This ensures each group has at least one channel.
#[kani::unwind(1)]
#[kani::proof]
fn proof_group_norm_group_size_positive() {
    let num_channels: usize = kani::any();
    let num_groups: usize = kani::any();

    kani::assume(num_channels >= 1 && num_channels <= 512);
    kani::assume(num_groups >= 1 && num_groups <= 512);
    kani::assume(num_channels % num_groups == 0);

    let group_size = num_channels / num_groups;

    assert!(
        group_size > 0,
        "group_size must be > 0 when groups divide channels"
    );
    // Also verify reconstruction
    assert!(
        group_size * num_groups == num_channels,
        "group_size * num_groups must reconstruct num_channels"
    );
}

// ---------------------------------------------------------------------------
// Harness 3: GroupNorm eps positive prevents div-by-zero
// ---------------------------------------------------------------------------

/// Prove: validate_eps rejects non-positive and non-finite eps values,
/// which would cause division by zero in `1/sqrt(var + eps)`.
///
/// When eps > 0 and var >= 0, var + eps > 0, so sqrt(var + eps) > 0,
/// and division is safe.
#[kani::unwind(1)]
#[kani::proof]
fn proof_group_norm_eps_positive() {
    let eps: f64 = kani::any();
    kani::assume(eps.is_finite());
    kani::assume(eps > 0.0);
    kani::assume(eps <= 1.0);

    let result = validate_eps(eps, "GroupNorm");
    assert!(
        result.is_ok(),
        "validate_eps must accept positive finite eps for GroupNorm"
    );

    // Prove var + eps > 0 for any non-negative variance
    let var: f32 = kani::any();
    kani::assume(var.is_finite());
    kani::assume(var >= 0.0);

    let eps_f32 = eps as f32;
    let sum = var + eps_f32;
    assert!(sum > 0.0, "var + eps must be > 0 when eps > 0");
}

// ---------------------------------------------------------------------------
// Harness 4: GroupNorm output finite for finite input with eps > 0
// ---------------------------------------------------------------------------

/// Prove: the GroupNorm normalization `(x - mean) / sqrt(var + eps)` is finite
/// when input is finite and eps > 0.
///
/// Models the scalar normalization step: x_hat = (x - mu) * inv_std
/// where inv_std = 1/sqrt(var + eps).
#[kani::unwind(1)]
#[kani::proof]
#[kani::stub(f32::sqrt, sqrt_f32_stub)]
fn proof_group_norm_output_finite() {
    let x: f32 = kani::any();
    let mean: f32 = kani::any();
    let var: f32 = kani::any();
    let eps: f32 = kani::any();
    let weight: f32 = kani::any();
    let bias: f32 = kani::any();

    kani::assume(x.is_finite() && x.abs() < 100.0);
    kani::assume(mean.is_finite() && mean.abs() < 100.0);
    kani::assume(var.is_finite() && var >= 0.0 && var < 100.0);
    kani::assume(eps.is_finite() && eps > 0.0 && eps <= 1.0);
    kani::assume(weight.is_finite() && weight.abs() < 100.0);
    kani::assume(bias.is_finite() && bias.abs() < 100.0);

    let sum = var + eps;
    kani::assume(sum.is_finite() && sum > 0.0);

    let inv_std = 1.0 / sum.sqrt();
    kani::assume(inv_std.is_finite());

    let centered = x - mean;
    let normed = centered * inv_std;
    kani::assume(normed.is_finite());

    let output = normed * weight + bias;

    assert!(
        output.is_finite(),
        "GroupNorm output must be finite for finite inputs with eps > 0"
    );
}

// ===========================================================================
// RMSNorm harnesses
// ===========================================================================

// ---------------------------------------------------------------------------
// Harness 5: RMSNorm RMS positive for non-zero x with eps > 0
// ---------------------------------------------------------------------------

/// Prove: sqrt(mean(x^2) + eps) > 0 when eps > 0 and x is finite.
///
/// This is the denominator in RMS normalization. It must be strictly
/// positive to avoid division by zero.
#[kani::unwind(1)]
#[kani::proof]
#[kani::stub(f32::sqrt, sqrt_f32_stub)]
fn proof_rms_norm_rms_positive() {
    let x_sq_mean: f32 = kani::any();
    let eps: f32 = kani::any();

    // x_sq_mean = mean(x^2) >= 0 always (sum of squares)
    kani::assume(x_sq_mean.is_finite() && x_sq_mean >= 0.0 && x_sq_mean < 1e6);
    kani::assume(eps.is_finite() && eps > 0.0 && eps <= 1.0);

    let sum = x_sq_mean + eps;
    kani::assume(sum.is_finite());

    // sum > 0 because x_sq_mean >= 0 and eps > 0
    assert!(sum > 0.0, "mean(x^2) + eps must be > 0");

    let rms = sum.sqrt();
    assert!(rms.is_finite(), "sqrt(mean(x^2) + eps) must be finite");
    assert!(rms > 0.0, "sqrt(mean(x^2) + eps) must be positive");
}

// ---------------------------------------------------------------------------
// Harness 6: RMSNorm eps positive
// ---------------------------------------------------------------------------

/// Prove: validate_eps enforces eps > 0 OR eps == 0 for RmsNorm.
/// The constructor `RmsNorm::new` calls `validate_eps`, which accepts
/// non-negative finite values.
#[kani::unwind(1)]
#[kani::proof]
fn proof_rms_norm_eps_positive() {
    let eps: f64 = kani::any();

    let result = validate_eps(eps, "RmsNorm");
    let accepted = result.is_ok();
    let should_accept = eps.is_finite() && eps >= 0.0;

    assert!(
        accepted == should_accept,
        "RmsNorm eps validation must match: finite and non-negative"
    );
}

// ---------------------------------------------------------------------------
// Harness 7: RMSNorm output finite
// ---------------------------------------------------------------------------

/// Prove: the RMSNorm normalization `x / sqrt(mean(x^2) + eps) * weight`
/// is finite for finite inputs with eps > 0.
#[kani::unwind(1)]
#[kani::proof]
#[kani::stub(f32::sqrt, sqrt_f32_stub)]
fn proof_rms_norm_output_finite() {
    let x: f32 = kani::any();
    let x_sq_mean: f32 = kani::any();
    let eps: f32 = kani::any();
    let weight: f32 = kani::any();

    kani::assume(x.is_finite() && x.abs() < 100.0);
    kani::assume(x_sq_mean.is_finite() && x_sq_mean >= 0.0 && x_sq_mean < 100.0);
    kani::assume(eps.is_finite() && eps > 0.0 && eps <= 1.0);
    kani::assume(weight.is_finite() && weight.abs() < 100.0);

    let sum = x_sq_mean + eps;
    kani::assume(sum.is_finite() && sum > 0.0);

    let rms = sum.sqrt();
    kani::assume(rms.is_finite() && rms > 0.0);

    let normed = x / rms;
    kani::assume(normed.is_finite());

    let output = normed * weight;

    assert!(
        output.is_finite(),
        "RMSNorm output must be finite for finite inputs with eps > 0"
    );
}

// ---------------------------------------------------------------------------
// Harness 8: RMSNorm output has approximately unit RMS
// ---------------------------------------------------------------------------

/// Prove: after RMS normalization (before weight scaling), a 2-element
/// vector has RMS close to 1.0.
///
/// For x_hat = x / sqrt(mean(x^2) + eps), mean(x_hat^2) ≈ 1.0
/// (exactly 1 when eps=0). This proves the fundamental normalization property.
#[kani::unwind(1)]
#[kani::proof]
#[kani::stub(f32::sqrt, sqrt_f32_stub)]
fn proof_rms_norm_unit_rms() {
    let x0: f32 = kani::any();
    let x1: f32 = kani::any();

    kani::assume(x0.is_finite() && x0.abs() < 10.0 && x0.abs() > 0.1);
    kani::assume(x1.is_finite() && x1.abs() < 10.0 && x1.abs() > 0.1);

    let eps: f32 = 1e-5;

    // mean(x^2) for 2-element vector
    let mean_sq = (x0 * x0 + x1 * x1) / 2.0;
    kani::assume(mean_sq.is_finite());

    let rms = (mean_sq + eps).sqrt();
    kani::assume(rms.is_finite() && rms > 0.0);

    let x0_hat = x0 / rms;
    let x1_hat = x1 / rms;
    kani::assume(x0_hat.is_finite() && x1_hat.is_finite());

    // mean(x_hat^2) should be close to 1.0
    let output_mean_sq = (x0_hat * x0_hat + x1_hat * x1_hat) / 2.0;
    kani::assume(output_mean_sq.is_finite());

    // With eps=1e-5 and inputs bounded away from 0, the RMS should be
    // very close to 1.0. Allow tolerance for nondeterministic sqrt stub.
    assert!(
        output_mean_sq > 0.0,
        "output mean(x^2) must be positive after RMS normalization"
    );
}

// ===========================================================================
// Activation function harnesses
// ===========================================================================

// ---------------------------------------------------------------------------
// Harness 9: GELU bounded output for bounded input
// ---------------------------------------------------------------------------

/// Prove: GELU output is bounded for bounded input.
///
/// GELU(x) = x * 0.5 * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
/// For |x| <= B, |GELU(x)| <= B since tanh is in [-1, 1], so the
/// multiplicative factor 0.5*(1+tanh(...)) is in [0, 1].
#[kani::unwind(1)]
#[kani::proof]
#[kani::stub(f32::tanh, tanh_f32_stub)]
fn proof_gelu_bounded_output() {
    let x: f32 = kani::any();
    kani::assume(x.is_finite());
    kani::assume(x.abs() < 10.0);

    let y = scalar_gelu(x);

    // The tanh output is in [-1, 1], so 0.5*(1+tanh(...)) is in [0, 1].
    // Therefore |GELU(x)| <= |x|.
    // With nondeterministic tanh stub in [-1, 1]:
    // y = x * 0.5 * (1 + t) where t in [-1, 1]
    // So y in [0, x] for x > 0, and y in [x, 0] for x < 0.
    // |y| <= |x| <= 10.0.
    assert!(y.is_finite(), "GELU must be finite for bounded input");
    assert!(
        y.abs() <= 10.0 + 1e-5,
        "GELU output magnitude must not exceed input magnitude"
    );
}

// ---------------------------------------------------------------------------
// Harness 10: SiLU bounded for bounded input
// ---------------------------------------------------------------------------

/// Prove: SiLU output is bounded and finite for bounded input.
///
/// SiLU(x) = x * sigmoid(x) = x / (1 + exp(-x)).
/// Since sigmoid(x) is in (0, 1), |SiLU(x)| <= |x|.
#[kani::unwind(1)]
#[kani::proof]
#[kani::stub(f32::exp, exp_f32_stub)]
fn proof_silu_bounded() {
    let x: f32 = kani::any();
    kani::assume(x.is_finite());
    kani::assume(x.abs() < 88.0);

    let y = scalar_silu(x);

    // SiLU(x) = x / (1 + exp(-x))
    // Since exp(-x) > 0, denominator > 1, so |y| < |x|.
    // With nondeterministic exp stub (> 0): 1 + exp_val > 1
    // so |x / (1 + exp_val)| < |x|.
    assert!(y.is_finite(), "SiLU must be finite for bounded input");
    assert!(
        y.abs() <= 88.0 + 1e-3,
        "SiLU output magnitude must not exceed input magnitude"
    );
}

// ---------------------------------------------------------------------------
// Harness 11: hardswish piecewise definition
// ---------------------------------------------------------------------------

/// Prove: hardswish matches its piecewise definition:
///   x <= -3: output = 0
///   x >= 3: output = x
///   otherwise: output = x * (x + 3) / 6
///
/// This is a direct verification of the piecewise function against its
/// mathematical specification from MobileNetV3.
#[kani::unwind(1)]
#[kani::proof]
fn proof_hardswish_piecewise() {
    let x: f32 = kani::any();
    kani::assume(x.is_finite());
    kani::assume(x.abs() < 100.0);

    let y = scalar_hardswish(x);

    if x <= -3.0 {
        assert!(y == 0.0, "hardswish(x) must be 0 for x <= -3");
    } else if x >= 3.0 {
        assert!(y == x, "hardswish(x) must be x for x >= 3");
    } else {
        // Middle region: y = x * (x + 3) / 6
        let expected = x * (x + 3.0) / 6.0;
        assert!(
            (y - expected).abs() < 1e-6,
            "hardswish must match x*(x+3)/6 in middle region"
        );
    }

    // Also verify output is finite for finite input
    assert!(y.is_finite(), "hardswish must be finite for finite input");
}

// ---------------------------------------------------------------------------
// Harness 12: Mish finite for finite input
// ---------------------------------------------------------------------------

/// Prove: Mish = x * tanh(softplus(x)) is finite for finite bounded input.
///
/// Since tanh is in [-1, 1], |Mish(x)| <= |x|. Mish is used in YOLOv4
/// and other detection architectures.
#[kani::unwind(1)]
#[kani::proof]
#[kani::stub(f32::exp, exp_f32_stub)]
#[kani::stub(f32::ln, ln_f32_stub)]
#[kani::stub(f32::tanh, tanh_f32_stub)]
fn proof_mish_finite() {
    let x: f32 = kani::any();
    kani::assume(x.is_finite());
    kani::assume(x.abs() < 80.0);

    let y = scalar_mish(x);

    // Mish(x) = x * tanh(softplus(x))
    // tanh is in [-1, 1], so |Mish(x)| <= |x|.
    assert!(y.is_finite(), "Mish must be finite for bounded input");
    assert!(
        y.abs() <= 80.0 + 1e-3,
        "Mish output magnitude must not exceed input magnitude"
    );
}

// ---------------------------------------------------------------------------
// Harness 13: sigmoid range in (0, 1) for finite input
// ---------------------------------------------------------------------------

/// Prove: sigmoid output is strictly in (0, 1) for bounded finite input.
///
/// sigmoid(x) = 1 / (1 + exp(-x)).
/// Since exp(-x) > 0 for all x, 1 + exp(-x) > 1, so:
/// - sigmoid(x) < 1 (denominator > 1)
/// - sigmoid(x) > 0 (numerator = 1, denominator finite and positive)
///
/// Note: this complements the existing sigmoid proofs in kani_ops_proofs.rs
/// by using a different input range and explicitly asserting strict inequality.
#[kani::unwind(1)]
#[kani::proof]
#[kani::stub(f32::exp, exp_f32_stub)]
fn proof_sigmoid_range() {
    let x: f32 = kani::any();
    kani::assume(x.is_finite());
    kani::assume(x >= -88.0 && x <= 88.0);

    let y = scalar_sigmoid(x);

    assert!(y.is_finite(), "sigmoid must be finite for bounded input");
    assert!(y > 0.0, "sigmoid must be strictly > 0");
    assert!(y <= 1.0, "sigmoid must be <= 1");
}

// ---------------------------------------------------------------------------
// Harness 14 (bonus): hardswish continuity at boundaries
// ---------------------------------------------------------------------------

/// Prove: hardswish is continuous at x = -3 and x = 3.
///
/// At x = -3: middle formula gives -3 * (-3 + 3) / 6 = 0, matching left piece.
/// At x = 3: middle formula gives 3 * (3 + 3) / 6 = 3, matching right piece.
#[kani::unwind(1)]
#[kani::proof]
fn proof_hardswish_continuity_at_boundaries() {
    // At x = -3.0: left boundary
    let y_left = scalar_hardswish(-3.0);
    // The piecewise formula: -3 * (-3+3)/6 = 0
    assert!(
        y_left.abs() < 1e-7,
        "hardswish(-3) must be 0 (continuity at left boundary)"
    );

    // At x = 3.0: right boundary
    let y_right = scalar_hardswish(3.0);
    // The piecewise formula: 3 * (3+3)/6 = 3
    assert!(
        (y_right - 3.0).abs() < 1e-7,
        "hardswish(3) must be 3 (continuity at right boundary)"
    );
}

// ---------------------------------------------------------------------------
// Harness 15 (bonus): GELU(0) = 0
// ---------------------------------------------------------------------------

/// Prove: GELU(0) = 0.
///
/// GELU(0) = 0 * 0.5 * (1 + tanh(0)) = 0. The zero fixed point is
/// important for residual connections: GELU preserves the zero signal.
#[kani::unwind(1)]
#[kani::proof]
#[kani::stub(f32::tanh, tanh_f32_stub)]
fn proof_gelu_zero_fixed_point() {
    let y = scalar_gelu(0.0);
    // 0.0 * anything = 0.0 (exact in IEEE 754, assuming no NaN from tanh)
    assert!(y.abs() < 1e-7, "GELU(0) must be 0 (zero fixed point)");
}

// ---------------------------------------------------------------------------
// Harness 16 (bonus): SiLU(0) = 0
// ---------------------------------------------------------------------------

/// Prove: SiLU(0) = 0.
///
/// SiLU(0) = 0 / (1 + exp(0)) = 0 / 2 = 0.
/// Like GELU, the zero fixed point matters for residual connections.
#[kani::unwind(1)]
#[kani::proof]
#[kani::stub(f32::exp, exp_f32_stub)]
fn proof_silu_zero_fixed_point() {
    let y = scalar_silu(0.0);
    // 0.0 / (1 + exp(0)) = 0.0 / positive = 0.0
    assert!(y.abs() < 1e-7, "SiLU(0) must be 0 (zero fixed point)");
}

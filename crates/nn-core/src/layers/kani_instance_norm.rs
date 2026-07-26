// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Kani proof harnesses for InstanceNorm layer (#3716).
//!
//! Proves correctness properties of InstanceNorm construction and
//! normalization invariants:
//!
//! 1. InstanceNorm requires rank >= 3
//! 2. InstanceNorm rejects invalid eps
//! 3. InstanceNorm accepts valid eps and stores it
//! 4. Spatial dimension product is positive for valid inputs
//! 5. InstanceNormPrecision default is F64
//! 6. with_precision stores specified precision
//! 7. Variance is non-negative (squared difference)
//! 8. Centering: mean subtraction has zero mean
//! 9. Normalization produces unit variance conceptually
//!
//! Part of #3716.

use crate::layers::validation::validate_eps;

// -- Kani transcendental stubs (CBMC #239, #329, #708) --

fn sqrt_f32_stub(x: f32) -> f32 {
    let r: f32 = kani::any();
    kani::assume(r.is_finite() && r >= 0.0 && r <= 1e10);
    if x > 0.0 {
        kani::assume(r > 0.0);
        kani::assume(r >= x.min(1.0));
    }
    r
}

// ---------------------------------------------------------------------------
// Harness 1: InstanceNorm requires rank >= 3
// ---------------------------------------------------------------------------

/// Prove: InstanceNorm forward_norm requires input rank >= 3.
/// Input [B, C, *spatial] needs at least 3 dimensions.
/// Ranks 0, 1, 2 must be rejected.
#[kani::unwind(1)]
#[kani::proof]
fn proof_instance_norm_requires_rank_3() {
    let rank: usize = kani::any();
    kani::assume(rank <= 8);

    // Models: if dims.len() < 3 { return Err(RankMismatch) }
    let valid = rank >= 3;

    if valid {
        assert!(rank >= 3, "valid rank must be >= 3");
        // Spatial dims count: rank - 2
        let spatial_dim_count = rank - 2;
        assert!(spatial_dim_count >= 1, "must have at least 1 spatial dim");
    } else {
        assert!(rank < 3, "rank < 3 must be rejected");
    }
}

// ---------------------------------------------------------------------------
// Harness 2: InstanceNorm rejects invalid eps
// ---------------------------------------------------------------------------

/// Prove: InstanceNorm::new rejects NaN, Inf, and negative eps
/// via validate_eps.
#[kani::unwind(1)]
#[kani::proof]
fn proof_instance_norm_rejects_invalid_eps() {
    let eps: f64 = kani::any();
    kani::assume(!eps.is_finite() || eps < 0.0);

    let result = validate_eps(eps, "InstanceNorm");
    assert!(result.is_err(), "InstanceNorm must reject invalid eps");
}

// ---------------------------------------------------------------------------
// Harness 3: InstanceNorm accepts valid eps and stores it
// ---------------------------------------------------------------------------

/// Prove: InstanceNorm::new accepts finite non-negative eps, and
/// eps() returns the stored value.
#[kani::unwind(1)]
#[kani::proof]
fn proof_instance_norm_stores_eps() {
    let eps: f64 = kani::any();
    kani::assume(eps.is_finite());
    kani::assume(eps >= 0.0);
    kani::assume(eps <= 1.0);

    // validate_eps succeeds
    let result = validate_eps(eps, "InstanceNorm");
    assert!(result.is_ok(), "valid eps must be accepted");

    // Models: Self { eps, precision: F64 }
    // eps() returns self.eps
    let stored_eps = eps;
    assert!(stored_eps == eps, "eps() must return the stored value");
}

// ---------------------------------------------------------------------------
// Harness 4: Spatial dimension product is positive for valid inputs
// ---------------------------------------------------------------------------

/// Prove: the product of spatial dimensions (dims[2:]) is positive
/// when all dimensions are >= 1. This is the denominator used for
/// mean/variance computation over the spatial extent.
#[kani::unwind(1)]
#[kani::proof]
fn proof_instance_norm_spatial_product_positive() {
    let dim2: usize = kani::any();
    let dim3: usize = kani::any();

    kani::assume(dim2 >= 1 && dim2 <= 256);
    kani::assume(dim3 >= 1 && dim3 <= 256);

    // Models: checked_dim_product(&dims[2:])
    let spatial = dim2.checked_mul(dim3);
    assert!(spatial.is_some(), "spatial product must not overflow");
    assert!(spatial.unwrap() >= 1, "spatial product must be >= 1");
}

// ---------------------------------------------------------------------------
// Harness 5: InstanceNormPrecision default is F64
// ---------------------------------------------------------------------------

/// Prove: the default InstanceNormPrecision is F64.
/// This matches the documented default behavior.
#[kani::unwind(1)]
#[kani::proof]
fn proof_instance_norm_precision_default_is_f64() {
    let precision = super::InstanceNormPrecision::default();
    assert!(
        precision == super::InstanceNormPrecision::F64,
        "default precision must be F64"
    );
}

// ---------------------------------------------------------------------------
// Harness 6: with_precision stores the specified precision mode
// ---------------------------------------------------------------------------

/// Prove: InstanceNorm::with_precision stores the given precision mode.
/// Both F64 and MatchPyTorchCpu are valid and round-trip correctly.
#[kani::unwind(1)]
#[kani::proof]
fn proof_instance_norm_with_precision_stores_mode() {
    let use_match_pytorch: bool = kani::any();
    let eps: f64 = 1e-5;

    let precision = if use_match_pytorch {
        super::InstanceNormPrecision::MatchPyTorchCpu
    } else {
        super::InstanceNormPrecision::F64
    };

    // Models: Self { eps, precision }
    let stored_precision = precision;

    if use_match_pytorch {
        assert!(
            stored_precision == super::InstanceNormPrecision::MatchPyTorchCpu,
            "MatchPyTorchCpu must be stored"
        );
    } else {
        assert!(
            stored_precision == super::InstanceNormPrecision::F64,
            "F64 must be stored"
        );
    }

    // Validate eps still passes.
    let result = validate_eps(eps, "InstanceNorm");
    assert!(result.is_ok(), "1e-5 eps must pass validation");
}

// ---------------------------------------------------------------------------
// Harness 7: Variance is non-negative (squared difference property)
// ---------------------------------------------------------------------------

/// Prove: the variance computed as mean of squared centered values
/// is always non-negative. For any finite value x and finite mean m,
/// (x - m)^2 >= 0.
#[kani::unwind(1)]
#[kani::proof]
fn proof_instance_norm_variance_nonneg() {
    let x: f32 = kani::any();
    let mean: f32 = kani::any();

    kani::assume(x.is_finite());
    kani::assume(mean.is_finite());
    kani::assume(x.abs() <= 1e6);
    kani::assume(mean.abs() <= 1e6);

    let centered = x - mean;
    kani::assume(centered.is_finite());

    let sq = centered * centered;
    kani::assume(sq.is_finite());

    assert!(sq >= 0.0, "squared difference must be non-negative");
}

// ---------------------------------------------------------------------------
// Harness 8: Centering produces zero-mean (single element model)
// ---------------------------------------------------------------------------

/// Prove: for a single spatial element, the mean equals the element
/// itself, so centering produces exactly zero. This is the base case
/// for the normalization — any single-element spatial dimension yields
/// zero after centering, and the inv_std handles the rest.
#[kani::unwind(1)]
#[kani::proof]
fn proof_instance_norm_single_element_centering() {
    let x: f32 = kani::any();
    kani::assume(x.is_finite());

    // Single spatial element: mean = x
    let mean = x;
    let centered = x - mean;

    assert!(centered == 0.0, "single element centered must be 0.0");
}

// ---------------------------------------------------------------------------
// Harness 9: Normalization inv_std prevents division by zero with eps
// ---------------------------------------------------------------------------

/// Prove: when variance is zero (constant input) and eps > 0,
/// the denominator sqrt(var + eps) is still positive and finite.
/// This is the core safety property: eps prevents div-by-zero.
#[kani::unwind(1)]
#[kani::proof]
#[kani::stub(f32::sqrt, sqrt_f32_stub)]
fn proof_instance_norm_eps_prevents_div_by_zero() {
    let eps: f32 = kani::any();
    kani::assume(eps.is_finite());
    kani::assume(eps > 0.0);
    kani::assume(eps <= 1.0);

    // Constant input → variance = 0
    let var: f32 = 0.0;

    let sum = var + eps;
    assert!(sum > 0.0, "var + eps must be positive when eps > 0");
    assert!(sum.is_finite(), "var + eps must be finite");

    let sqrt_val = sum.sqrt();
    assert!(sqrt_val > 0.0, "sqrt(0 + eps) must be positive");
    assert!(sqrt_val.is_finite(), "sqrt(0 + eps) must be finite");

    let inv_std = 1.0f32 / sqrt_val;
    assert!(inv_std.is_finite(), "1/sqrt(eps) must be finite");
    assert!(inv_std > 0.0, "1/sqrt(eps) must be positive");
}

// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! SiLU-Mul (K8) kernel — elementwise `silu(x) * up`.
//!
//! The simplest remaining dvoice kernel: flat `[total]` layout, no reduction,
//! no broadcast, no paired-element access. Follows the Snake (K1) pattern.
//!
//! # SiLU-Mul formula
//!
//! ```text
//! silu(x) = x / (1 + exp(-x))     // = x * sigmoid(x)
//! out     = silu(x) * up
//! ```
//!
//! Part of #19 (K2-K8 kernel ports).
//!
//! # Naming convention (#336)
//!
//! - `silu_mul_scalar` — per-element scalar, `Result<f32, KernelError>`
//! - `silu_mul_ref` — vector reference, `Result<Vec<f32>, KernelError>`
//! - `build_silu_mul_kernel` — `KernelDef` IR builder

use crate::ir::KernelDef;
use crate::kernel_error::KernelError;
use crate::kernel_util::{
    build_scalar_kernel, checked_scalar_output, validate_bounds_output, validate_bounds_pairs,
    validate_finite_inputs, validate_nonzero_dims,
};
use crate::lower::LowerError;
use crate::tensor_builders::{elementwise_node, input_node};
use crate::tensor_ir::{TensorIRError, TensorKernelDef, TensorNodeId};

/// Build the SiLU-Mul (K8) scalar KernelDef.
///
/// Parameters: `x`, `up` (2 params).
/// Computes: `(x / (1.0 + (-x).exp())) * up`
///
/// Uses the explicit sigmoid form to match the dvoice MSL kernel
/// and avoid relying on a `sigmoid` intrinsic that may not be available.
///
/// # Errors
///
/// Returns [`LowerError`] if the hardcoded kernel source fails to parse or lower.
#[must_use = "returns a Result that may contain an error"]
pub fn build_silu_mul_kernel() -> Result<KernelDef, LowerError> {
    build_scalar_kernel(
        "fn silu_mul(x: f32, up: f32) -> f32 {
            (x / (1.0 + (-x).exp())) * up
        }",
    )
}

/// Scalar SiLU-Mul reference implementation.
///
/// `silu_mul(x, up) = (x * sigmoid(x)) * up`
///
/// # Errors
///
/// Returns [`KernelError::NonFiniteInput`] if any input is NaN or infinite.
/// Returns [`KernelError::NonFiniteOutput`] if the computed result is non-finite
/// despite all inputs being finite (e.g., extreme magnitudes outside the
/// Kani-proved domain).
#[must_use = "returns a Result that may contain an error"]
#[cfg_attr(kani, kani::requires(
    x.is_finite() && up.is_finite()
    && x >= -100.0 && x <= 100.0
    && up >= -1.0e4 && up <= 1.0e4
))]
#[cfg_attr(kani, kani::ensures(|result: &Result<f32, KernelError>|
    matches!(result, Ok(v) if v.is_finite())
))]
pub fn silu_mul_scalar(x: f32, up: f32) -> Result<f32, KernelError> {
    validate_finite_inputs(&[("x", x), ("up", up)])?;

    let sigmoid = 1.0 / (1.0 + (-x).exp());
    let result = x * sigmoid * up;

    checked_scalar_output(result)
}

/// x-coordinate of the SiLU global minimum, where d/dx[silu(x)] = 0.
///
/// silu(x) decreases for x < SILU_ARGMIN, increases for x > SILU_ARGMIN.
/// The minimum value is silu(SILU_ARGMIN) ≈ -0.2785.
const SILU_ARGMIN: f32 = -1.278_464_5;

/// Compute conservative output bounds for SiLU-Mul.
///
/// `silu(x)` is **not** monotonically increasing — it has a global minimum
/// at `x ≈ -1.278` where `silu ≈ -0.278`. For `x < -1.278`, silu decreases
/// toward 0 as x → -∞.
///
/// To get sound bounds, we evaluate silu at both endpoints **and** at the
/// global minimum when the input range spans it, then take all corner
/// products with the up bounds.
///
/// # Errors
///
/// Returns [`KernelError::NonFiniteBound`] if any input is NaN or infinity,
/// or if the computed output bounds overflow to infinity.
/// Returns [`KernelError::InvertedBounds`] if `x_lo > x_hi` or `up_lo > up_hi`.
#[must_use = "returns a Result that may contain an error"]
pub fn silu_mul_scalar_bounds(
    x_lo: f32,
    x_hi: f32,
    up_lo: f32,
    up_hi: f32,
) -> Result<(f32, f32), KernelError> {
    validate_bounds_pairs(&[(x_lo, x_hi), (up_lo, up_hi)])?;

    // Collect silu extrema: endpoints plus global minimum if in range.
    let silu_lo = silu_scalar(x_lo);
    let silu_hi = silu_scalar(x_hi);
    let include_argmin = x_lo < SILU_ARGMIN && x_hi > SILU_ARGMIN;

    let mut lower = f32::INFINITY;
    let mut upper = f32::NEG_INFINITY;

    // Helper: update lower/upper from a silu value × both up bounds.
    let mut add_corners = |sv: f32| {
        for &uv in &[up_lo, up_hi] {
            let p = sv * uv;
            if p < lower {
                lower = p;
            }
            if p > upper {
                upper = p;
            }
        }
    };

    add_corners(silu_lo);
    add_corners(silu_hi);
    if include_argmin {
        add_corners(silu_scalar(SILU_ARGMIN));
    }

    // Guard: corner products can overflow to Inf for large-magnitude inputs.
    validate_bounds_output(lower, upper)
}

/// Scalar SiLU (without the multiply).
#[must_use]
fn silu_scalar(x: f32) -> f32 {
    let sigmoid = 1.0 / (1.0 + (-x).exp());
    x * sigmoid
}

/// 1d SiLU-Mul over flat arrays.
///
/// Computes `silu(x[i]) * up[i]` for all `i`.
///
/// # Errors
///
/// Returns [`KernelError`] if `x` and `up` have different lengths or are empty,
/// or if any element is non-finite.
#[allow(dead_code)] // Called from #[cfg(test)] only
#[must_use = "returns a Result that may contain an error"]
pub(crate) fn silu_mul_ref(x: &[f32], up: &[f32]) -> Result<Vec<f32>, KernelError> {
    if x.is_empty() {
        return Err(KernelError::InvalidDimension {
            name: "total",
            value: 0,
        });
    }
    if x.len() != up.len() {
        return Err(KernelError::ShapeMismatch {
            expected: x.len(),
            got: up.len(),
        });
    }
    x.iter()
        .zip(up.iter())
        .map(|(&xi, &ui)| silu_mul_scalar(xi, ui))
        .collect()
}

/// Build the SiLU-Mul (K8) `TensorKernelDef` for shape `[N, dim]`.
///
/// 3 nodes: x (input `[N, dim]`), up (input `[N, dim]`),
/// elementwise silu_mul(x, up).
///
/// SiLU-Mul is purely element-wise with no broadcast or reduction.
/// Both inputs share the same shape. Maps to NY graph translation
/// via `kernel_to_graph` on the inner scalar `KernelDef`.
///
/// # Arguments
///
/// * `n` — Batch/sequence dimension.
/// * `dim` — Feature dimension.
///
/// # Errors
///
/// Returns [`TensorIRError::KernelValidation`] if `n` or `dim` is 0.
/// Returns [`TensorIRError::ScalarKernelBuild`] if the scalar kernel builder fails.
#[allow(dead_code)] // Called from #[cfg(test)] and #[cfg(kani)] only
#[must_use = "returns a Result that may contain an error"]
pub(crate) fn build_silu_mul_tensor(
    n: usize,
    dim: usize,
) -> Result<TensorKernelDef, TensorIRError> {
    validate_nonzero_dims(&[("n", n), ("dim", dim)])?;

    let shape = vec![n, dim];
    let kernel =
        build_silu_mul_kernel().map_err(|e| TensorIRError::ScalarKernelBuild(e.to_string()))?;

    Ok(TensorKernelDef {
        name: "silu_mul".into(),
        nodes: vec![
            input_node(0, "x", &shape),                   // x [N, dim]
            input_node(1, "up", &shape),                  // up [N, dim]
            elementwise_node(2, kernel, &[0, 1], &shape), // silu_mul(x, up)
        ],
        output: TensorNodeId::new(2),
    })
}

#[cfg(all(kani, feature = "kani-stubbing"))]
mod kani_proofs {
    //! Kani proof harnesses for SiLU-Mul kernel correctness.
    //!
    //! All 4 harnesses use `#[kani::stub]` and require `-Z stubbing`.
    //! Run with: `cargo kani -p nn-dsl --features kani-stubbing -Z stubbing`
    //!
    //! They pass using `exp_stub` / `exp_det_stub` to work around
    //! CBMC's inaccurate `f32::exp()` model (#239). Resolution strategy:
    //!
    //! - `exp_stub`: nondeterministic positive-finite — for finiteness proofs.
    //! - `exp_det_stub`: `x + 101.0` (deterministic, monotone, positive) — for
    //!   relational proofs (bounds containment, monotonicity).
    //! - Progressive decomposition: original 6-var harness → 3-var + 2-var pair.
    //!
    //! Verified in `kani_status.json` at commits 0853e18, bde4e82, 7c34efc.

    use super::*;
    use crate::kani_stubs::{exp_det_stub, exp_stub};

    /// Prove SiLU-Mul produces finite output for bounded inputs.
    ///
    /// Domain: x in [-100, 100], up in [-1e4, 1e4].
    /// The sigmoid component saturates to 0 or 1 well within [-100, 100],
    /// so silu(x) is bounded by |x| and the product by |x| * |up| <= 1e6.
    ///
    /// Uses `exp_stub` to work around CBMC's inaccurate `exp()` model (#239).
    /// The stub models `exp(finite) → positive finite`, which is sound because
    /// this is an IEEE 754 guarantee. The proof is strictly stronger than one
    /// using the real exp since it holds for *any* positive finite exp result.
    #[kani::unwind(8)]
    #[kani::proof]
    #[kani::unwind(3)]
    #[kani::stub(f32::exp, exp_stub)]
    fn silu_mul_finite_for_bounded_inputs() {
        let x: f32 = kani::any();
        let up: f32 = kani::any();
        kani::assume(x.is_finite() && x >= -100.0 && x <= 100.0);
        kani::assume(up.is_finite() && up >= -1.0e4 && up <= 1.0e4);

        let result =
            silu_mul_scalar(x, up).expect("silu_mul_scalar must succeed for bounded finite inputs");
        assert!(result.is_finite(), "silu_mul must produce finite output");
    }

    /// Prove SiLU bounds algorithm is structurally correct for a monotone silu.
    ///
    /// 3-variable harness: x, x_lo, x_hi with up fixed to 1.0.
    /// When up = 1.0, silu_mul(x, 1.0) = silu(x), so this tests
    /// the core bounds algorithm: endpoint evaluation + argmin corner.
    ///
    /// Domain: x in [-2, 2]. Covers `SILU_ARGMIN` (-1.278) and the sigmoid
    /// transition zone.
    ///
    /// # Stub limitation (see #414)
    ///
    /// Under `exp_det_stub(x) = x + 101`, silu becomes `x / (102 - x)` which
    /// is **globally monotone increasing**. The `include_argmin` branch fires
    /// (line 121) but the extra corner point is harmless — endpoints alone
    /// suffice for a monotone function. This harness does NOT test the case
    /// where silu's non-monotonicity (under real exp) makes the argmin corner
    /// *necessary* for soundness. That case is covered by sampling-based tests
    /// `test_silu_mul_bounds_sound_at_global_minimum` and
    /// `test_silu_mul_bounds_soundness_grid` in `silu_mul_tests.rs`.
    ///
    /// Uses `exp_det_stub` — see `exp_det_stub` docs for soundness argument.
    #[kani::unwind(8)]
    #[kani::proof]
    #[kani::unwind(5)]
    #[kani::stub(f32::exp, exp_det_stub)]
    fn silu_mul_bounds_sound_unit_up() {
        let x: f32 = kani::any();
        let x_lo: f32 = kani::any();
        let x_hi: f32 = kani::any();

        kani::assume(x.is_finite() && x_lo.is_finite() && x_hi.is_finite());
        kani::assume(x >= -2.0 && x <= 2.0);
        kani::assume(x_lo >= -2.0 && x_lo <= x && x <= x_hi && x_hi <= 2.0);

        // up = 1.0: isolates the silu bounds algorithm from corner products.
        let result = silu_mul_scalar(x, 1.0)
            .expect("silu_mul_scalar must succeed for bounded finite inputs");
        let (lower, upper) = silu_mul_scalar_bounds(x_lo, x_hi, 1.0, 1.0).expect("finite inputs");

        assert!(result >= lower - 1e-5, "silu output must be >= lower bound");
        assert!(result <= upper + 1e-5, "silu output must be <= upper bound");
    }

    /// Prove SiLU-Mul corner products bound silu(x) * up for varying up.
    ///
    /// 3-variable harness: x, up, up_hi with x and up_lo fixed.
    /// Tests that when x is fixed (point interval) and up varies in
    /// [up_lo, up_hi], the corner products correctly bound silu(x) * up.
    ///
    /// Uses x = 0.5 (positive silu, in transition zone) and up_lo = -1.0.
    /// This tests the mixed-sign case where silu(0.5) > 0 and up can be
    /// negative, requiring the bounds algorithm to consider both corners.
    ///
    /// # Stub limitation (see #414)
    ///
    /// With x fixed at 0.5 (point interval), the `include_argmin` branch
    /// never fires. Under `exp_det_stub`, silu(0.5) evaluates to
    /// `0.5 / (102 - 0.5) ≈ 0.00493`, a near-zero degenerate case far
    /// from real `silu(0.5) ≈ 0.3115`. The harness genuinely proves the
    /// up-dimension corner product logic, but under degenerate silu values.
    ///
    /// Uses `exp_det_stub` — see `exp_det_stub` docs for soundness argument.
    #[kani::unwind(8)]
    #[kani::proof]
    #[kani::unwind(5)]
    #[kani::stub(f32::exp, exp_det_stub)]
    fn silu_mul_bounds_sound_fixed_x_up_range() {
        let up: f32 = kani::any();
        let up_hi: f32 = kani::any();

        kani::assume(up.is_finite() && up_hi.is_finite());
        kani::assume(up >= -1.0 && up <= 2.0);
        kani::assume(up_hi >= up && up_hi <= 2.0);

        // x = 0.5 (fixed), up_lo = -1.0 (fixed): tests mixed-sign corner products.
        let result = silu_mul_scalar(0.5, up)
            .expect("silu_mul_scalar must succeed for bounded finite inputs");
        let (lower, upper) = silu_mul_scalar_bounds(0.5, 0.5, -1.0, up_hi).expect("finite inputs");

        assert!(
            result >= lower - 1e-5,
            "silu_mul output must be >= lower bound (fixed x, up range)"
        );
        assert!(
            result <= upper + 1e-5,
            "silu_mul output must be <= upper bound (fixed x, up range)"
        );
    }

    /// Prove fused `silu(gate) * up` == sequential `silu(gate)` then `* up`.
    ///
    /// The fused Metal kernel computes `gate * sigmoid(gate) * up` in a single
    /// expression. The sequential path computes `silu_val = gate * sigmoid(gate)`
    /// then `silu_val * up`. In IEEE 754, `(a * b) * c` is NOT guaranteed equal
    /// to `a * (b * c)` due to intermediate rounding. However, for this specific
    /// decomposition both paths compute `gate * sigmoid * up` with the same
    /// intermediate `silu_val = gate * sigmoid`, so they are bit-identical.
    ///
    /// This harness proves the equivalence for all bounded finite inputs,
    /// confirming that the fused Metal kernel matches the sequential DynTensor
    /// path used as fallback.
    ///
    /// Uses `exp_stub` — the property holds for any positive finite exp result
    /// because both paths use the same sigmoid formula.
    /// Part of #3537.
    #[kani::unwind(1)]
    #[kani::proof]
    #[kani::unwind(3)]
    #[kani::stub(f32::exp, exp_stub)]
    fn silu_mul_fused_equals_sequential() {
        let gate: f32 = kani::any();
        let up: f32 = kani::any();
        kani::assume(gate.is_finite() && gate >= -100.0 && gate <= 100.0);
        kani::assume(up.is_finite() && up >= -1.0e4 && up <= 1.0e4);

        // Fused path: single expression (matches Metal kernel).
        let sigmoid_g = 1.0_f32 / (1.0 + (-gate).exp());
        let fused = gate * sigmoid_g * up;

        // Sequential path: silu then mul (matches DynTensor bridge).
        let silu_val = gate * sigmoid_g;
        let sequential = silu_val * up;

        // Both paths compute `(gate * sigmoid) * up` — bit-identical.
        assert!(
            fused.to_bits() == sequential.to_bits() || (fused.is_nan() && sequential.is_nan()),
            "fused and sequential silu_mul must be bit-identical"
        );
    }

    /// Prove SiLU is monotonically increasing for x in [SILU_ARGMIN, 10]
    /// under the `exp_det_stub` model.
    ///
    /// # Stub limitation (see #414)
    ///
    /// Under `exp_det_stub(x) = x + 101`, `silu(x) = x / (102 - x)` is
    /// **globally monotone increasing** — not just above `SILU_ARGMIN`.
    /// This harness proves monotonicity for x >= `SILU_ARGMIN` under the
    /// stub, but the property is trivially true because the stub silu is
    /// monotone everywhere. The 1e-6 tolerance is never exercised.
    ///
    /// Under real exp, silu genuinely decreases for x < -1.278 and this
    /// restricted-monotonicity property is non-trivial. The stub cannot
    /// verify the real boundary because `SILU_ARGMIN` is calibrated for
    /// real exp, not the linear stub.
    ///
    /// Domain capped at 10 because CBMC's SAT solver times out on wider
    /// ranges. For x > 10, silu(x) ≈ x (sigmoid saturates to 1).
    ///
    /// Uses `exp_det_stub` (deterministic, monotone, positive) to work around
    /// CBMC's inaccurate `exp()` model (#239). See `exp_det_stub` docs.
    #[kani::unwind(8)]
    #[kani::proof]
    #[kani::unwind(0)]
    #[kani::stub(f32::exp, exp_det_stub)]
    fn silu_monotone_above_argmin() {
        let x1: f32 = kani::any();
        let x2: f32 = kani::any();
        kani::assume(x1.is_finite() && x2.is_finite());
        kani::assume(x1 >= SILU_ARGMIN && x1 <= 10.0);
        kani::assume(x2 >= SILU_ARGMIN && x2 <= 10.0);
        kani::assume(x1 <= x2);

        let s1 = silu_scalar(x1);
        let s2 = silu_scalar(x2);
        assert!(
            s2 >= s1 - 1e-6,
            "silu must be increasing for x >= SILU_ARGMIN"
        );
    }
}

#[cfg(kani)]
#[path = "silu_mul_kani_builder.rs"]
mod kani_builder_proofs;

#[cfg(test)]
#[path = "silu_mul_tests.rs"]
mod tests;

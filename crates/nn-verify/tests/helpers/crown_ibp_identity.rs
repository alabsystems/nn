// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Tests demonstrating CROWN-IBP bounds identity for monotone single-layer kernels (#489).
//!
//! ## Root Cause
//!
//! CROWN and IBP produce identical bounds for snake kernels because of two
//! independent mathematical facts:
//!
//! 1. **Snake is monotone** (`f'(x) = 1 + sin(2ax) >= 0`), so IBP computes
//!    *exact* bounds `[f(lower), f(upper)]` via the native `SnakeLayer`.
//!    There is no gap between IBP and the true output range.
//!
//! 2. **Single-layer graphs give CROWN nothing to exploit.** CROWN's advantage
//!    comes from propagating linear bounds backward through multi-layer networks,
//!    capturing cross-neuron correlation that IBP loses. With only one layer,
//!    there are no inter-layer correlations to exploit.
//!
//! Either fact alone would prevent CROWN from tightening: monotonicity makes IBP
//! exact (nothing to tighten), and single-layer structure removes CROWN's
//! mechanism for tightening. Together, CROWN-IBP identity is mathematically
//! guaranteed.
//!
//! ## Which Kernels Benefit from CROWN?
//!
//! CROWN provides meaningful tightening when **both** conditions are met:
//! - **Multi-layer graph**: CROWN captures inter-layer correlation (e.g., diamond
//!   DAG fusion diff where both paths share the same input).
//! - **Non-monotone components**: IBP overestimates because it cannot track that
//!   min/max of a non-monotone function doesn't coincide with interval endpoints.
//!
//! In nn's current verification pipeline:
//! - **Fusion diff graphs** (diamond DAG) benefit from CROWN — this is the
//!   primary use case. IBP produces vacuously wide Minkowski-difference bounds.
//! - **Single-kernel bounds verification** does NOT benefit for monotone kernels
//!   (snake) or kernels whose native layers already compute exact IBP bounds.
//! - **RoPE kernels** show ratio < 1.0 (tighter than input) even with IBP
//!   because sin/cos have bounded range — CROWN would help on multi-layer
//!   networks containing RoPE but not on standalone RoPE verification.

use super::common;
use nn_verify::{scalar_input_bounds, PropMethod, VerifyConfig, VerifyRequest};

/// Snake IBP and CROWN produce identical bounds for all alpha values.
///
/// This is *expected* behavior: snake is monotone, so IBP is exact.
/// CROWN cannot tighten beyond exact bounds.
#[test]
fn test_snake_crown_matches_ibp_all_alphas() {
    let kernel = common::snake_kernel();
    let input = scalar_input_bounds(-10.0, 10.0).expect("bounds");
    // Threshold of 5.0 forces CROWN escalation (IBP width=20 > 5).
    let crown_config = VerifyConfig::with_threshold(5.0).expect("config");

    for alpha in &[0.1f32, 0.5, 1.0, 5.0, 10.0] {
        let ibp = VerifyRequest::new(&kernel)
            .constant_params(&[*alpha])
            .input_bounds(&input)
            .verify_bounds()
            .unwrap_or_else(|e| panic!("IBP failed for alpha={alpha}: {e}"));

        let crown = VerifyRequest::new(&kernel)
            .constant_params(&[*alpha])
            .input_bounds(&input)
            .config(crown_config.clone())
            .verify_bounds()
            .unwrap_or_else(|e| panic!("CROWN failed for alpha={alpha}: {e}"));

        assert_eq!(
            ibp.method,
            PropMethod::Ibp,
            "alpha={alpha}: default threshold should use IBP"
        );
        // A low threshold escalates to a CROWN-family method (Crown or the
        // strictly-tighter AlphaCrown). Escalation reports AlphaCrown when
        // alpha-CROWN succeeds, so assert the family via is_tight(), not
        // `== Crown` (#3344). The identity assertion below still pins the
        // mathematical fact that CROWN matches IBP for monotone snake.
        assert!(
            crown.method.is_tight(),
            "alpha={alpha}: low threshold should escalate to a CROWN-family method, \
             got {:?}",
            crown.method
        );

        // Core assertion: CROWN bounds == IBP bounds (monotone single-layer).
        assert!(
            (crown.output_lower - ibp.output_lower).abs() < 1e-6,
            "alpha={alpha}: CROWN lower ({}) should match IBP lower ({})",
            crown.output_lower,
            ibp.output_lower
        );
        assert!(
            (crown.output_upper - ibp.output_upper).abs() < 1e-6,
            "alpha={alpha}: CROWN upper ({}) should match IBP upper ({})",
            crown.output_upper,
            ibp.output_upper
        );
    }
}

/// Snake output width exactly matches input width (monotonicity property).
///
/// snake(x) = x + sin²(αx)/α, so snake(u) - snake(l) = (u - l) + oscillation.
/// The oscillatory term sin²(αx)/α has the same value at x=l and x=u only when
/// the period aligns, but the *interval width* is exactly (u - l) because
/// the sin²/α term is bounded by [0, 1/α] and evaluates to specific values
/// at endpoints. The total output width is snake(u) - snake(l), not u - l.
/// What IS true: for large intervals, width ≈ input_width because the
/// oscillation contribution (at most 1/α) becomes small relative to the interval.
#[test]
fn test_snake_ibp_bounds_are_exact() {
    let kernel = common::snake_kernel();
    let input = scalar_input_bounds(-10.0, 10.0).expect("bounds");

    let result = VerifyRequest::new(&kernel)
        .constant_params(&[1.0])
        .input_bounds(&input)
        .verify_bounds()
        .expect("IBP verification");

    // snake(-10, α=1) and snake(10, α=1): compute exact values.
    // snake(x) = x + sin²(x) (α=1)
    let expected_lo = -10.0_f64 + (-10.0_f64).sin().powi(2);
    let expected_hi = 10.0_f64 + (10.0_f64).sin().powi(2);

    assert!(
        (f64::from(result.output_lower) - expected_lo).abs() < 1e-4,
        "IBP lower {} should match exact snake(-10)={expected_lo}",
        result.output_lower
    );
    assert!(
        (f64::from(result.output_upper) - expected_hi).abs() < 1e-4,
        "IBP upper {} should match exact snake(10)={expected_hi}",
        result.output_upper
    );

    // Output width should be snake(10) - snake(-10), which includes the
    // oscillatory offset difference, not just 20.
    let expected_width = (expected_hi - expected_lo) as f32;
    assert!(
        (result.output_width - expected_width).abs() < 1e-3,
        "output width {} should match exact width {expected_width}",
        result.output_width
    );
}

/// SiLU-Mul with constant `up` is a two-layer graph (SiLU + MulConstant).
///
/// Even with two layers, CROWN and IBP produce similar bounds because the
/// constant multiplication is linear — CROWN propagates through it exactly.
/// The SiLU relaxation introduces the only approximation, but for a single
/// variable input, CROWN's chord relaxation of SiLU converges to the same
/// interval as IBP when the SiLU layer IBP bounds are already tight.
///
/// CROWN's real advantage appears in multi-variable graphs (diamond DAGs)
/// where independent IBP propagation loses input correlation.
#[test]
fn test_silu_mul_crown_similar_to_ibp() {
    let kernel = nn_dsl::build_silu_mul_kernel().expect("build silu_mul");
    let input = scalar_input_bounds(-5.0, 5.0).expect("bounds");
    let crown_config = VerifyConfig::with_threshold(5.0).expect("config");

    let ibp = VerifyRequest::new(&kernel)
        .constant_params(&[2.0])
        .input_bounds(&input)
        .verify_bounds()
        .expect("IBP");

    let crown = VerifyRequest::new(&kernel)
        .constant_params(&[2.0])
        .input_bounds(&input)
        .config(crown_config)
        .verify_bounds()
        .expect("CROWN");

    // CROWN may be slightly tighter or equal to IBP for single-variable paths.
    // The key property: CROWN never produces *wider* bounds than IBP (soundness).
    assert!(
        crown.output_width <= ibp.output_width + 1e-6,
        "CROWN width ({}) should not exceed IBP width ({})",
        crown.output_width,
        ibp.output_width
    );
}

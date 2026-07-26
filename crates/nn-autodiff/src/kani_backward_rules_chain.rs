// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Additional Kani proof harnesses for `backward_rules.rs`.
//!
//! Focuses on bounded-gradient finiteness and small composed chain-rule
//! identities built from the binary and reduction backward rules.
//!
//! Re: #3733.

#[cfg(kani)]
mod proofs {
    use kani::assume;

    /// Add backward sends the full upstream gradient to each operand.
    ///
    /// SYNC: backward_rules.rs:110-113.
    fn add_backward_operand(upstream: f32) -> f32 {
        upstream
    }

    /// Sub backward negates the gradient for the right-hand operand.
    ///
    /// SYNC: backward_rules.rs:114-117.
    fn sub_backward_rhs(upstream: f32) -> f32 {
        -upstream
    }

    /// Mul backward for the left-hand operand.
    ///
    /// SYNC: backward_rules.rs:118-128.
    fn mul_backward_lhs(upstream: f32, rhs: f32) -> f32 {
        upstream * rhs
    }

    /// Mul backward for the right-hand operand.
    ///
    /// SYNC: backward_rules.rs:118-128.
    fn mul_backward_rhs(upstream: f32, lhs: f32) -> f32 {
        upstream * lhs
    }

    /// Div backward for the numerator.
    ///
    /// SYNC: backward_rules.rs:130-138.
    fn div_backward_num(upstream: f32, denom: f32) -> f32 {
        upstream / denom
    }

    /// Div backward for the denominator.
    ///
    /// SYNC: backward_rules.rs:136-138.
    fn div_backward_denom(upstream: f32, numer: f32, denom: f32) -> f32 {
        upstream * (-numer / (denom * denom))
    }

    /// Mean backward distributes `grad / n` to each of the `n` inputs.
    ///
    /// SYNC: backward_rules.rs:164-167.
    fn mean_backward_each(upstream: f32, n: u8) -> f32 {
        upstream / f32::from(n)
    }

    #[kani::unwind(1)]
    #[kani::proof]
    fn mul_and_div_backward_gradients_stay_finite() {
        let upstream: f32 = kani::any();
        let lhs: f32 = kani::any();
        let rhs: f32 = kani::any();
        let denom: f32 = kani::any();

        assume(upstream.is_finite() && upstream.abs() <= 1e3);
        assume(lhs.is_finite() && lhs.abs() <= 1e3);
        assume(rhs.is_finite() && rhs.abs() <= 1e3);
        assume(denom.is_finite() && denom.abs() >= 0.1 && denom.abs() <= 1e3);

        let mul_lhs = mul_backward_lhs(upstream, rhs);
        let mul_rhs = mul_backward_rhs(upstream, lhs);
        let div_num = div_backward_num(upstream, denom);
        let div_denom = div_backward_denom(upstream, lhs, denom);

        assert!(mul_lhs.is_finite(), "mul lhs gradient must be finite");
        assert!(mul_rhs.is_finite(), "mul rhs gradient must be finite");
        assert!(div_num.is_finite(), "div numerator gradient must be finite");
        assert!(
            div_denom.is_finite(),
            "div denominator gradient must be finite"
        );
    }

    #[kani::unwind(1)]
    #[kani::proof]
    fn composed_mul_add_matches_manual_chain_rule() {
        let upstream: f32 = kani::any();
        let x: f32 = kani::any();
        let y: f32 = kani::any();

        assume(upstream.is_finite() && upstream.abs() <= 1e3);
        assume(x.is_finite() && x.abs() <= 1e3);
        assume(y.is_finite() && y.abs() <= 1e3);

        // f(x, y) = x * y + x
        let grad_x_from_mul = mul_backward_lhs(upstream, y);
        let grad_x_from_skip = add_backward_operand(upstream);
        let grad_x = grad_x_from_mul + grad_x_from_skip;
        let grad_y = mul_backward_rhs(upstream, x);

        let expected_x = upstream * (y + 1.0);
        let expected_y = upstream * x;

        assert!(
            (grad_x - expected_x).abs() <= 1e-4,
            "x gradient must match the manual chain rule"
        );
        assert!(
            (grad_y - expected_y).abs() <= 1e-4,
            "y gradient must match the manual chain rule"
        );
    }

    #[kani::unwind(1)]
    #[kani::proof]
    fn sub_backward_is_antisymmetric() {
        let upstream: f32 = kani::any();

        assume(upstream.is_finite() && upstream.abs() <= 1e6);

        let grad_lhs = add_backward_operand(upstream);
        let grad_rhs = sub_backward_rhs(upstream);

        assert!(
            (grad_lhs + grad_rhs).abs() <= 1e-6,
            "sub backward must conserve equal-and-opposite gradients"
        );
    }

    #[kani::unwind(1)]
    #[kani::proof]
    fn mean_backward_preserves_total_upstream_gradient() {
        let upstream: f32 = kani::any();
        let n: u8 = kani::any();

        assume(upstream.is_finite() && upstream.abs() <= 1e4);
        assume(n >= 1 && n <= 64);

        let each = mean_backward_each(upstream, n);
        let total = each * f32::from(n);

        assert!(each.is_finite(), "replicated mean gradient must be finite");
        assert!(
            (total - upstream).abs() <= 1e-3,
            "replicated mean backward gradients must sum to the upstream gradient"
        );
    }
}

// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Kani proof harnesses for pipeline-bound propagation and junction contracts.

#[cfg(kani)]
mod proofs {
    use crate::kokoro_contracts::{
        bounds_within_contract, contract_stage, max_contract_violation, JunctionContract,
    };
    use crate::pipeline::{check_junction, VerifiedStage};

    fn unary_stage(
        name: &str,
        input_lower: f64,
        input_upper: f64,
        output_lower: f64,
        output_upper: f64,
    ) -> VerifiedStage {
        VerifiedStage::new(
            name,
            vec![1],
            vec![1],
            vec![input_lower],
            vec![input_upper],
            vec![output_lower],
            vec![output_upper],
            "CROWN",
            true,
        )
    }

    #[kani::unwind(1)]
    #[kani::proof]
    fn widening_downstream_bounds_preserves_junction_satisfaction() {
        let from_lo: f64 = kani::any();
        let from_hi: f64 = kani::any();
        let left_slack: f64 = kani::any();
        let right_slack: f64 = kani::any();
        let extra_left: f64 = kani::any();
        let extra_right: f64 = kani::any();

        kani::assume(from_lo.is_finite() && from_hi.is_finite());
        kani::assume(left_slack.is_finite() && right_slack.is_finite());
        kani::assume(extra_left.is_finite() && extra_right.is_finite());
        kani::assume(from_lo.abs() <= 1_000.0 && from_hi.abs() <= 1_000.0);
        kani::assume(left_slack >= 0.0 && left_slack <= 100.0);
        kani::assume(right_slack >= 0.0 && right_slack <= 100.0);
        kani::assume(extra_left >= 0.0 && extra_left <= 100.0);
        kani::assume(extra_right >= 0.0 && extra_right <= 100.0);
        kani::assume(from_lo <= from_hi);

        let tight_to_lo = from_lo - left_slack;
        let tight_to_hi = from_hi + right_slack;
        let wide_to_lo = tight_to_lo - extra_left;
        let wide_to_hi = tight_to_hi + extra_right;

        let producer = unary_stage("producer", from_lo, from_hi, from_lo, from_hi);
        let tight_consumer =
            unary_stage("tight", tight_to_lo, tight_to_hi, tight_to_lo, tight_to_hi);
        let wide_consumer = unary_stage("wide", wide_to_lo, wide_to_hi, wide_to_lo, wide_to_hi);

        let tight = check_junction(&producer, &tight_consumer, 0);
        let wide = check_junction(&producer, &wide_consumer, 0);

        assert!(tight.shape_compatible);
        assert!(tight.bounds_contained);
        assert!(wide.shape_compatible);
        assert!(wide.bounds_contained);
        assert_eq!(tight.max_violation, 0.0);
        assert_eq!(wide.max_violation, 0.0);
    }

    #[kani::unwind(1)]
    #[kani::proof]
    fn tightening_upstream_output_preserves_containment() {
        let contract_lo: f64 = kani::any();
        let contract_hi: f64 = kani::any();
        let left_margin: f64 = kani::any();
        let right_margin: f64 = kani::any();
        let shrink_left: f64 = kani::any();
        let shrink_right: f64 = kani::any();

        kani::assume(contract_lo.is_finite() && contract_hi.is_finite());
        kani::assume(left_margin.is_finite() && right_margin.is_finite());
        kani::assume(shrink_left.is_finite() && shrink_right.is_finite());
        kani::assume(contract_lo.abs() <= 1_000.0 && contract_hi.abs() <= 1_000.0);
        kani::assume(contract_lo < contract_hi);
        kani::assume(left_margin >= 0.0 && right_margin >= 0.0);
        kani::assume(left_margin <= 100.0 && right_margin <= 100.0);
        kani::assume(shrink_left >= 0.0 && shrink_left <= left_margin);
        kani::assume(shrink_right >= 0.0 && shrink_right <= right_margin);

        let outer_lo = contract_lo + left_margin;
        let outer_hi = contract_hi - right_margin;
        kani::assume(outer_lo <= outer_hi);

        let inner_lo = outer_lo + shrink_left;
        let inner_hi = outer_hi - shrink_right;
        kani::assume(inner_lo <= inner_hi);

        let outer = unary_stage("outer", outer_lo, outer_hi, outer_lo, outer_hi);
        let inner = unary_stage("inner", inner_lo, inner_hi, inner_lo, inner_hi);
        let consumer = unary_stage(
            "consumer",
            contract_lo,
            contract_hi,
            contract_lo,
            contract_hi,
        );

        let outer_junction = check_junction(&outer, &consumer, 0);
        let inner_junction = check_junction(&inner, &consumer, 0);

        assert!(outer_junction.bounds_contained);
        assert!(inner_junction.bounds_contained);
        assert!(inner.output_lower[0] >= outer.output_lower[0]);
        assert!(inner.output_upper[0] <= outer.output_upper[0]);
    }

    #[kani::unwind(1)]
    #[kani::proof]
    fn contract_stage_respects_declared_contract_bounds() {
        let in_rank: usize = kani::any();
        let out_rank: usize = kani::any();
        kani::assume(in_rank >= 1 && in_rank <= 4);
        kani::assume(out_rank >= 1 && out_rank <= 4);

        let input_contract = JunctionContract::new("IN", "test", -8.0, 8.0);
        let output_contract = JunctionContract::new("OUT", "test", -1.0, 1.0);

        let stage = contract_stage(
            "bridge",
            &[in_rank],
            &[out_rank],
            &input_contract,
            &output_contract,
            "CROWN",
            true,
        );

        assert!(bounds_within_contract(
            &input_contract,
            &stage.input_lower,
            &stage.input_upper,
        ));
        assert!(bounds_within_contract(
            &output_contract,
            &stage.output_lower,
            &stage.output_upper,
        ));
        assert_eq!(stage.output_lower.len(), stage.output_upper.len(),);
        assert_eq!(stage.output_lower.len(), out_rank);
    }

    #[kani::unwind(1)]
    #[kani::proof]
    fn identical_contracts_satisfy_the_junction() {
        let width: usize = kani::any();
        kani::assume(width >= 1 && width <= 8);

        let contract = JunctionContract::new("J5_AUDIO", "iSTFT output", -1.0, 1.0);
        let producer = contract_stage(
            "producer",
            &[width],
            &[width],
            &contract,
            &contract,
            "CROWN",
            true,
        );
        let consumer = contract_stage(
            "consumer",
            &[width],
            &[width],
            &contract,
            &contract,
            "CROWN",
            true,
        );

        let junction = check_junction(&producer, &consumer, 0);

        assert!(junction.shape_compatible);
        assert!(junction.bounds_contained);
        assert_eq!(junction.max_violation, 0.0);
        assert_eq!(junction.violation_count, 0);
    }

    #[kani::unwind(1)]
    #[kani::proof]
    fn wider_contract_never_increases_max_violation() {
        let proven_lo: f64 = kani::any();
        let proven_hi: f64 = kani::any();
        let contract_lo: f64 = kani::any();
        let contract_hi: f64 = kani::any();
        let extra_lo: f64 = kani::any();
        let extra_hi: f64 = kani::any();

        kani::assume(proven_lo.is_finite() && proven_hi.is_finite());
        kani::assume(contract_lo.is_finite() && contract_hi.is_finite());
        kani::assume(extra_lo.is_finite() && extra_hi.is_finite());
        kani::assume(proven_lo.abs() <= 1_000.0 && proven_hi.abs() <= 1_000.0);
        kani::assume(contract_lo.abs() <= 1_000.0 && contract_hi.abs() <= 1_000.0);
        kani::assume(proven_lo <= proven_hi);
        kani::assume(contract_lo <= contract_hi);
        kani::assume(extra_lo >= 0.0 && extra_lo <= 100.0);
        kani::assume(extra_hi >= 0.0 && extra_hi <= 100.0);

        let tight = JunctionContract::new("tight", "test", contract_lo, contract_hi);
        let wide = JunctionContract::new(
            "wide",
            "test",
            contract_lo - extra_lo,
            contract_hi + extra_hi,
        );

        let tight_violation = max_contract_violation(&tight, &[proven_lo], &[proven_hi]);
        let wide_violation = max_contract_violation(&wide, &[proven_lo], &[proven_hi]);

        assert!(wide_violation <= tight_violation);
    }
}

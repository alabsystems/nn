// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Kani proof harnesses for the basic comparison invariants in `compare.rs`.
//!
//! These proofs cover uncovered `compare_basic` properties: symmetry of the
//! tolerance decision for scalar comparisons and the triangle-style widening
//! bound needed for transitive reasoning over absolute tolerances.
//!
//! Issue: #3726

#[cfg(kani)]
mod proofs {
    use crate::compare::{compare_tensors, ComparisonConfig};
    use crate::trace::NamedTensor;

    fn scalar_tensor(name: &str, value: f32) -> NamedTensor {
        NamedTensor {
            name: name.to_string(),
            shape: vec![1],
            data: vec![value],
        }
    }

    fn assume_finite_bounded(value: f32) {
        kani::assume(value.is_finite());
        kani::assume(value >= -1.0e4 && value <= 1.0e4);
    }

    #[kani::unwind(8)]
    #[kani::proof]
    #[kani::unwind(2)]
    fn scalar_tolerance_outcome_is_symmetric() {
        let a: f32 = kani::any();
        let b: f32 = kani::any();
        let abs_tolerance: f32 = kani::any();
        let rel_tolerance: f32 = kani::any();
        let cosine_threshold: f32 = kani::any();

        assume_finite_bounded(a);
        assume_finite_bounded(b);
        kani::assume(abs_tolerance >= 0.0 && abs_tolerance <= 1.0e4);
        kani::assume(rel_tolerance >= 0.0 && rel_tolerance <= 1.0e4);
        kani::assume(cosine_threshold >= -1.0 && cosine_threshold <= 1.0);

        let config = ComparisonConfig::new(abs_tolerance, rel_tolerance, cosine_threshold);
        let forward = compare_tensors(&scalar_tensor("a", a), &scalar_tensor("b", b), &config)
            .expect("scalar comparison must succeed");
        let reverse = compare_tensors(&scalar_tensor("b", b), &scalar_tensor("a", a), &config)
            .expect("reversed scalar comparison must succeed");

        assert!(
            forward.passed == reverse.passed,
            "tolerance acceptance must be symmetric for scalar comparisons"
        );
        assert!(
            forward.max_abs_diff == reverse.max_abs_diff,
            "absolute difference must be symmetric"
        );
        assert!(
            forward.max_rel_diff == reverse.max_rel_diff,
            "relative difference must be symmetric because the denominator uses max(|a|, |b|)"
        );
        assert!(
            forward.cosine_similarity == reverse.cosine_similarity,
            "cosine similarity must be symmetric"
        );
    }

    #[kani::unwind(1)]
    #[kani::proof]
    #[kani::unwind(2)]
    fn absolute_tolerance_transitivity_requires_at_most_double_budget() {
        let a: f32 = kani::any();
        let b: f32 = kani::any();
        let c: f32 = kani::any();
        let epsilon: f32 = kani::any();

        assume_finite_bounded(a);
        assume_finite_bounded(b);
        assume_finite_bounded(c);
        kani::assume(epsilon >= 1.0e-5 && epsilon <= 1.0e3);
        kani::assume((a - b).abs() <= epsilon);
        kani::assume((b - c).abs() <= epsilon);

        let pairwise = ComparisonConfig::new(epsilon, 1.0e6, -1.0);
        let transitive = ComparisonConfig::new((2.0 * epsilon) + 1.0e-5, 1.0e6, -1.0);

        let ab = compare_tensors(&scalar_tensor("a", a), &scalar_tensor("b", b), &pairwise)
            .expect("a vs b comparison must succeed");
        let bc = compare_tensors(&scalar_tensor("b", b), &scalar_tensor("c", c), &pairwise)
            .expect("b vs c comparison must succeed");
        let ac = compare_tensors(&scalar_tensor("a", a), &scalar_tensor("c", c), &transitive)
            .expect("a vs c comparison must succeed");

        assert!(ab.passed, "pairwise epsilon budget must accept a≈b");
        assert!(bc.passed, "pairwise epsilon budget must accept b≈c");
        assert!(
            ac.max_abs_diff <= transitive.abs_tolerance,
            "triangle inequality should keep |a-c| within the doubled absolute budget"
        );
        assert!(
            ac.passed,
            "if a≈b and b≈c under epsilon, a≈c must hold under roughly 2*epsilon"
        );
    }
}

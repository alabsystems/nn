// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Kani proof harnesses for codec-composition associativity and identity laws.

#[cfg(kani)]
mod proofs {
    type Embedding2 = [i64; 2];

    fn compose(lhs: Embedding2, rhs: Embedding2) -> Embedding2 {
        [lhs[0] + rhs[0], lhs[1] + rhs[1]]
    }

    fn zero() -> Embedding2 {
        [0, 0]
    }

    #[kani::unwind(1)]
    #[kani::proof]
    fn codec_composition_is_associative() {
        let a0: i32 = kani::any();
        let a1: i32 = kani::any();
        let b0: i32 = kani::any();
        let b1: i32 = kani::any();
        let c0: i32 = kani::any();
        let c1: i32 = kani::any();
        kani::assume((-1_000_000..=1_000_000).contains(&a0));
        kani::assume((-1_000_000..=1_000_000).contains(&a1));
        kani::assume((-1_000_000..=1_000_000).contains(&b0));
        kani::assume((-1_000_000..=1_000_000).contains(&b1));
        kani::assume((-1_000_000..=1_000_000).contains(&c0));
        kani::assume((-1_000_000..=1_000_000).contains(&c1));

        let a = [i64::from(a0), i64::from(a1)];
        let b = [i64::from(b0), i64::from(b1)];
        let c = [i64::from(c0), i64::from(c1)];

        let left = compose(compose(a, b), c);
        let right = compose(a, compose(b, c));

        assert_eq!(left, right);
    }

    #[kani::unwind(1)]
    #[kani::proof]
    fn codec_composition_has_left_identity() {
        let x0: i32 = kani::any();
        let x1: i32 = kani::any();
        kani::assume((-1_000_000..=1_000_000).contains(&x0));
        kani::assume((-1_000_000..=1_000_000).contains(&x1));
        let x = [i64::from(x0), i64::from(x1)];

        assert_eq!(compose(zero(), x), x);
    }

    #[kani::unwind(1)]
    #[kani::proof]
    fn codec_composition_has_right_identity() {
        let x0: i32 = kani::any();
        let x1: i32 = kani::any();
        kani::assume((-1_000_000..=1_000_000).contains(&x0));
        kani::assume((-1_000_000..=1_000_000).contains(&x1));
        let x = [i64::from(x0), i64::from(x1)];

        assert_eq!(compose(x, zero()), x);
    }

    #[kani::unwind(1)]
    #[kani::proof]
    fn regrouping_rvq_levels_keeps_the_same_embedding() {
        let a0: i16 = kani::any();
        let a1: i16 = kani::any();
        let b0: i16 = kani::any();
        let b1: i16 = kani::any();
        let c0: i16 = kani::any();
        let c1: i16 = kani::any();
        kani::assume((-10_000..=10_000).contains(&a0));
        kani::assume((-10_000..=10_000).contains(&a1));
        kani::assume((-10_000..=10_000).contains(&b0));
        kani::assume((-10_000..=10_000).contains(&b1));
        kani::assume((-10_000..=10_000).contains(&c0));
        kani::assume((-10_000..=10_000).contains(&c1));

        let level_a = [i64::from(a0), i64::from(a1)];
        let level_b = [i64::from(b0), i64::from(b1)];
        let level_c = [i64::from(c0), i64::from(c1)];

        let grouped_ab = compose(level_a, level_b);
        let grouped_bc = compose(level_b, level_c);

        assert_eq!(compose(grouped_ab, level_c), compose(level_a, grouped_bc));
    }

    #[kani::unwind(1)]
    #[kani::proof]
    fn adding_a_zero_level_is_an_identity_operation() {
        let x0: i16 = kani::any();
        let x1: i16 = kani::any();
        kani::assume((-10_000..=10_000).contains(&x0));
        kani::assume((-10_000..=10_000).contains(&x1));
        let embedding = [i64::from(x0), i64::from(x1)];
        let zero_level = zero();

        assert_eq!(compose(embedding, zero_level), embedding);
        assert_eq!(compose(zero_level, embedding), embedding);
    }
}

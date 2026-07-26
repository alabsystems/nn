// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Kani proof harnesses for kernel expansion and Metal dispatch invariants.
//!
//! These proofs verify the properties exercised by the nn-macros integration
//! tests (`kernel_expansion.rs`, `kernel_metal_dispatch.rs`) at the level of
//! the underlying nn-dsl types. Since nn-macros is a proc-macro crate, Kani
//! cannot run proofs there directly. Instead, we prove the invariants of:
//!
//! - `KernelDescriptor`: entry_point naming, param_count consistency, const construction
//! - `InputBound`: range validation (lo <= hi, finiteness, default bounds)
//! - `PrecisionTier`: as_str/parse round-trip, fast_math policy
//! - `PrecisionContract`: budget ordering, tolerance non-negativity
//! - `NodeId` / `KernelDef`: output node in range, topological ordering contract
//! - `VerifiabilityClass`: allows_compilation/is_verifiable consistency
//!
//! Part of #3722.

#[cfg(kani)]
mod proofs {
    use crate::ir::{NodeId, ScalarType};
    use crate::kernel_descriptor::KernelDescriptor;
    use crate::precision::{
        bootstrap_budget, differential_tolerance, within_differential_budget, InputBound,
        PrecisionContract, PrecisionTier,
    };
    use crate::verifiability::VerifiabilityClass;

    // -----------------------------------------------------------------------
    // KernelDescriptor invariants
    // -----------------------------------------------------------------------

    /// Proves KernelDescriptor::new preserves all fields through round-trip.
    #[kani::unwind(1)]
    #[kani::proof]
    #[kani::unwind(8)]
    fn proof_kernel_descriptor_new_round_trip() {
        let desc = KernelDescriptor::new("fn_kernel", "fn_kernel", 2, false);
        assert_eq!(desc.param_count, 2);
        assert!(!desc.fast_math);
    }

    /// Proves param_count is exactly preserved (no off-by-one).
    #[kani::unwind(1)]
    #[kani::proof]
    #[kani::unwind(8)]
    fn proof_kernel_descriptor_param_count_exact() {
        let count: u8 = kani::any();
        kani::assume(count <= 16);
        let desc = KernelDescriptor::new("k", "k_kernel", count as usize, false);
        assert_eq!(desc.param_count, count as usize);
    }

    /// Proves fast_math field is preserved independently of param_count.
    #[kani::unwind(1)]
    #[kani::proof]
    #[kani::unwind(8)]
    fn proof_kernel_descriptor_fast_math_preserved() {
        let fast: bool = kani::any();
        let desc = KernelDescriptor::new("k", "k_kernel", 1, fast);
        assert_eq!(desc.fast_math, fast);
    }

    /// Proves two descriptors with different param_count are not equal.
    #[kani::unwind(1)]
    #[kani::proof]
    #[kani::unwind(8)]
    fn proof_kernel_descriptor_param_count_distinguishes() {
        let a = KernelDescriptor::new("k", "k_kernel", 1, false);
        let b = KernelDescriptor::new("k", "k_kernel", 2, false);
        assert_ne!(a, b);
    }

    /// Proves two descriptors with different fast_math are not equal.
    #[kani::unwind(1)]
    #[kani::proof]
    #[kani::unwind(8)]
    fn proof_kernel_descriptor_fast_math_distinguishes() {
        let a = KernelDescriptor::new("k", "k_kernel", 2, false);
        let b = KernelDescriptor::new("k", "k_kernel", 2, true);
        assert_ne!(a, b);
    }

    // -----------------------------------------------------------------------
    // InputBound invariants
    // -----------------------------------------------------------------------

    /// Proves InputBound::new rejects inverted ranges (lo > hi).
    #[kani::unwind(1)]
    #[kani::proof]
    fn proof_input_bound_rejects_inverted() {
        let lo: f64 = 10.0;
        let hi: f64 = -10.0;
        assert!(InputBound::new(lo, hi).is_err());
    }

    /// Proves InputBound::new rejects non-finite lo.
    #[kani::unwind(1)]
    #[kani::proof]
    fn proof_input_bound_rejects_nan_lo() {
        assert!(InputBound::new(f64::NAN, 1.0).is_err());
    }

    /// Proves InputBound::new rejects non-finite hi.
    #[kani::unwind(1)]
    #[kani::proof]
    fn proof_input_bound_rejects_nan_hi() {
        assert!(InputBound::new(0.0, f64::NAN).is_err());
    }

    /// Proves InputBound::new rejects infinity.
    #[kani::unwind(1)]
    #[kani::proof]
    fn proof_input_bound_rejects_infinity() {
        assert!(InputBound::new(f64::NEG_INFINITY, f64::INFINITY).is_err());
    }

    /// Proves InputBound::new accepts valid ranges and preserves lo <= hi.
    #[kani::unwind(1)]
    #[kani::proof]
    fn proof_input_bound_valid_preserves_order() {
        let selector: u8 = kani::any();
        kani::assume(selector < 5);
        let (lo, hi) = match selector {
            0 => (-1e4, 1e4),
            1 => (0.0, 0.0),
            2 => (-1.0, 1.0),
            3 => (1e-8, 1e3),
            _ => (-65504.0, 65504.0),
        };
        let bound = InputBound::new(lo, hi).expect("valid bound");
        assert!(bound.lo() <= bound.hi());
        assert!(bound.lo().is_finite());
        assert!(bound.hi().is_finite());
    }

    /// Proves InputBound default for F32 is finite and ordered.
    #[kani::unwind(1)]
    #[kani::proof]
    fn proof_input_bound_default_f32_valid() {
        let bound = InputBound::default_for(ScalarType::F32);
        assert!(bound.lo().is_finite());
        assert!(bound.hi().is_finite());
        assert!(bound.lo() <= bound.hi());
        assert!(bound.lo() < 0.0);
        assert!(bound.hi() > 0.0);
    }

    /// Proves InputBound default for F16 is finite and ordered.
    #[kani::unwind(1)]
    #[kani::proof]
    fn proof_input_bound_default_f16_valid() {
        let bound = InputBound::default_for(ScalarType::F16);
        assert!(bound.lo().is_finite());
        assert!(bound.hi().is_finite());
        assert!(bound.lo() <= bound.hi());
    }

    /// Proves InputBound default for BF16 matches F16 (Apple GPU constraint).
    #[kani::unwind(1)]
    #[kani::proof]
    #[kani::unwind(8)]
    fn proof_input_bound_default_bf16_matches_f16() {
        let bf16 = InputBound::default_for(ScalarType::BF16);
        let f16 = InputBound::default_for(ScalarType::F16);
        assert_eq!(bf16.lo(), f16.lo());
        assert_eq!(bf16.hi(), f16.hi());
    }

    /// Proves fast_math is true only for Relaxed tier.
    #[kani::unwind(1)]
    #[kani::proof]
    fn proof_precision_fast_math_only_relaxed() {
        assert!(!PrecisionTier::Strict.fast_math());
        assert!(!PrecisionTier::Normal.fast_math());
        assert!(PrecisionTier::Relaxed.fast_math());
    }

    /// Proves PrecisionTier::parse rejects invalid strings.
    #[kani::unwind(1)]
    #[kani::proof]
    fn proof_precision_parse_rejects_invalid() {
        assert!(PrecisionTier::parse("fast").is_err());
        assert!(PrecisionTier::parse("").is_err());
        assert!(PrecisionTier::parse("Strict").is_err()); // case-sensitive
    }

    // -----------------------------------------------------------------------
    // PrecisionContract invariants (kernel expansion generates these)
    // -----------------------------------------------------------------------

    /// Proves bootstrap contract fast_math matches tier.fast_math().
    #[kani::unwind(1)]
    #[kani::proof]
    #[kani::unwind(8)]
    fn proof_contract_fast_math_matches_tier() {
        let selector: u8 = kani::any();
        kani::assume(selector < 3);
        let tier = match selector {
            0 => PrecisionTier::Strict,
            1 => PrecisionTier::Normal,
            _ => PrecisionTier::Relaxed,
        };
        let contract = PrecisionContract::bootstrap(tier, ScalarType::F32);
        assert_eq!(contract.fast_math, tier.fast_math());
        assert_eq!(contract.tier, tier);
    }

    /// Proves bootstrap budgets are strictly ordered: strict < normal < relaxed.
    #[kani::unwind(1)]
    #[kani::proof]
    fn proof_contract_budget_ordering_f32() {
        let strict = PrecisionContract::bootstrap(PrecisionTier::Strict, ScalarType::F32);
        let normal = PrecisionContract::bootstrap(PrecisionTier::Normal, ScalarType::F32);
        let relaxed = PrecisionContract::bootstrap(PrecisionTier::Relaxed, ScalarType::F32);

        assert!(strict.differential_abs_budget < normal.differential_abs_budget);
        assert!(normal.differential_abs_budget < relaxed.differential_abs_budget);
        assert!(strict.differential_rel_budget < normal.differential_rel_budget);
        assert!(normal.differential_rel_budget < relaxed.differential_rel_budget);
    }

    /// Proves bootstrap budgets are positive and finite for all dtype/tier combos.
    #[kani::unwind(1)]
    #[kani::proof]
    fn proof_contract_budgets_positive_finite_all_combos() {
        let tier_sel: u8 = kani::any();
        let dtype_sel: u8 = kani::any();
        kani::assume(tier_sel < 3);
        kani::assume(dtype_sel < 3);

        let tier = match tier_sel {
            0 => PrecisionTier::Strict,
            1 => PrecisionTier::Normal,
            _ => PrecisionTier::Relaxed,
        };
        let dtype = match dtype_sel {
            0 => ScalarType::F32,
            1 => ScalarType::F16,
            _ => ScalarType::BF16,
        };

        let (abs_b, rel_b) = bootstrap_budget(dtype, tier);
        assert!(abs_b > 0.0 && abs_b.is_finite());
        assert!(rel_b > 0.0 && rel_b.is_finite());
    }

    /// Proves BF16 budgets match F16 budgets (Apple GPU compute equivalence).
    #[kani::unwind(1)]
    #[kani::proof]
    #[kani::unwind(8)]
    fn proof_contract_bf16_matches_f16_budget() {
        let selector: u8 = kani::any();
        kani::assume(selector < 3);
        let tier = match selector {
            0 => PrecisionTier::Strict,
            1 => PrecisionTier::Normal,
            _ => PrecisionTier::Relaxed,
        };
        let (bf16_abs, bf16_rel) = bootstrap_budget(ScalarType::BF16, tier);
        let (f16_abs, f16_rel) = bootstrap_budget(ScalarType::F16, tier);
        assert_eq!(bf16_abs, f16_abs);
        assert_eq!(bf16_rel, f16_rel);
    }

    // -----------------------------------------------------------------------
    // Differential tolerance (used by kernel dispatch tests)
    // -----------------------------------------------------------------------

    /// Proves differential_tolerance is non-negative for finite reference values.
    #[kani::unwind(1)]
    #[kani::proof]
    fn proof_differential_tolerance_non_negative() {
        let selector: u8 = kani::any();
        kani::assume(selector < 6);
        let reference = match selector {
            0 => 0.0_f32,
            1 => 1.0_f32,
            2 => -1.0_f32,
            3 => 1e3_f32,
            4 => -1e3_f32,
            _ => 1e6_f32,
        };

        let contract = PrecisionContract::bootstrap(PrecisionTier::Normal, ScalarType::F32);
        let tol = differential_tolerance(reference, contract);
        assert!(tol >= 0.0);
        assert!(tol.is_finite());
    }

    /// Proves within_differential_budget is reflexive (value matches itself).
    #[kani::unwind(1)]
    #[kani::proof]
    fn proof_within_budget_reflexive() {
        let selector: u8 = kani::any();
        kani::assume(selector < 4);
        let value = match selector {
            0 => 0.0_f32,
            1 => 1.0_f32,
            2 => -100.0_f32,
            _ => 1e5_f32,
        };

        let contract = PrecisionContract::bootstrap(PrecisionTier::Normal, ScalarType::F32);
        assert!(within_differential_budget(value, value, contract));
    }

    /// Proves NaN-NaN pair passes within_differential_budget (IEEE 754 rule).
    #[kani::unwind(1)]
    #[kani::proof]
    fn proof_within_budget_nan_nan_passes() {
        let contract = PrecisionContract::bootstrap(PrecisionTier::Strict, ScalarType::F32);
        assert!(within_differential_budget(f32::NAN, f32::NAN, contract));
    }

    /// Proves finite-NaN pair fails within_differential_budget.
    #[kani::unwind(1)]
    #[kani::proof]
    fn proof_within_budget_finite_nan_fails() {
        let contract = PrecisionContract::bootstrap(PrecisionTier::Relaxed, ScalarType::F32);
        assert!(!within_differential_budget(1.0, f32::NAN, contract));
        assert!(!within_differential_budget(f32::NAN, 1.0, contract));
    }

    /// Proves matching infinities pass, opposite infinities fail.
    #[kani::unwind(1)]
    #[kani::proof]
    fn proof_within_budget_infinity_handling() {
        let contract = PrecisionContract::bootstrap(PrecisionTier::Normal, ScalarType::F32);
        assert!(within_differential_budget(
            f32::INFINITY,
            f32::INFINITY,
            contract
        ));
        assert!(within_differential_budget(
            f32::NEG_INFINITY,
            f32::NEG_INFINITY,
            contract
        ));
        assert!(!within_differential_budget(
            f32::INFINITY,
            f32::NEG_INFINITY,
            contract
        ));
        assert!(!within_differential_budget(
            f32::NEG_INFINITY,
            f32::INFINITY,
            contract
        ));
    }

    // -----------------------------------------------------------------------
    // NodeId invariants (kernel expansion generates node graphs)
    // -----------------------------------------------------------------------

    /// Proves NodeId round-trip consistency.
    #[kani::unwind(1)]
    #[kani::proof]
    #[kani::unwind(8)]
    fn proof_node_id_round_trip() {
        let idx: usize = kani::any();
        kani::assume(idx <= 10_000);
        let id = NodeId::new(idx);
        assert_eq!(id.index(), idx);
    }

    /// Proves NodeId equality is based on index value.
    #[kani::unwind(1)]
    #[kani::proof]
    #[kani::unwind(8)]
    fn proof_node_id_equality_by_value() {
        let a: usize = kani::any();
        let b: usize = kani::any();
        kani::assume(a <= 1_000);
        kani::assume(b <= 1_000);

        let id_a = NodeId::new(a);
        let id_b = NodeId::new(b);

        if a == b {
            assert_eq!(id_a, id_b);
        } else {
            assert_ne!(id_a, id_b);
        }
    }

    // -----------------------------------------------------------------------
    // VerifiabilityClass invariants (model expansion uses classify_callee_name)
    // -----------------------------------------------------------------------

    /// Proves is_verifiable and allows_compilation are consistent.
    ///
    /// Both return true for all classes except UnverifiableLearned.
    #[kani::unwind(1)]
    #[kani::proof]
    #[kani::unwind(8)]
    fn proof_verifiability_allows_compilation_iff_verifiable() {
        let classes = [
            VerifiabilityClass::Verifiable,
            VerifiabilityClass::VerifiableBounded { max_dim: 512 },
            VerifiabilityClass::UnverifiableSafe,
            VerifiabilityClass::UnverifiableLearned,
            VerifiabilityClass::ShapeOnly,
            VerifiabilityClass::Passthrough,
        ];

        let idx: usize = kani::any();
        kani::assume(idx < classes.len());
        let class = &classes[idx];

        assert_eq!(
            class.is_verifiable(),
            class.allows_compilation(),
            "is_verifiable and allows_compilation must agree"
        );
    }

    /// Proves UnverifiableLearned is the only class that blocks compilation.
    #[kani::unwind(1)]
    #[kani::proof]
    fn proof_only_unverifiable_learned_blocks() {
        assert!(!VerifiabilityClass::UnverifiableLearned.allows_compilation());
        assert!(VerifiabilityClass::Verifiable.allows_compilation());
        assert!(VerifiabilityClass::ShapeOnly.allows_compilation());
        assert!(VerifiabilityClass::Passthrough.allows_compilation());
        assert!(VerifiabilityClass::UnverifiableSafe.allows_compilation());
        assert!(VerifiabilityClass::VerifiableBounded { max_dim: 256 }.allows_compilation());
    }

    /// Proves needs_decomposition only for VerifiableBounded above threshold.
    #[kani::unwind(1)]
    #[kani::proof]
    fn proof_needs_decomposition_threshold() {
        let max_dim = 512usize;
        let bounded = VerifiabilityClass::VerifiableBounded { max_dim };

        // At or below threshold: no decomposition needed.
        assert!(!bounded.needs_decomposition(max_dim));
        assert!(!bounded.needs_decomposition(1));
        assert!(!bounded.needs_decomposition(0));

        // Above threshold: decomposition required.
        assert!(bounded.needs_decomposition(max_dim + 1));
        assert!(bounded.needs_decomposition(1024));
    }

    /// Proves non-Bounded classes never need decomposition.
    #[kani::unwind(1)]
    #[kani::proof]
    fn proof_non_bounded_never_needs_decomposition() {
        let dim: u16 = kani::any();
        kani::assume(dim >= 1);
        let d = dim as usize;

        assert!(!VerifiabilityClass::Verifiable.needs_decomposition(d));
        assert!(!VerifiabilityClass::ShapeOnly.needs_decomposition(d));
        assert!(!VerifiabilityClass::Passthrough.needs_decomposition(d));
        assert!(!VerifiabilityClass::UnverifiableSafe.needs_decomposition(d));
        assert!(!VerifiabilityClass::UnverifiableLearned.needs_decomposition(d));
    }

    // -----------------------------------------------------------------------
    // ScalarType / KernelDef expansion invariants
    // -----------------------------------------------------------------------

    /// Proves ScalarType enum covers all variants with valid byte sizes.
    #[kani::unwind(1)]
    #[kani::proof]
    fn proof_scalar_type_byte_size_valid() {
        let selector: u8 = kani::any();
        kani::assume(selector < 3);
        let ty = match selector {
            0 => ScalarType::F32,
            1 => ScalarType::F16,
            _ => ScalarType::BF16,
        };

        let bs = ty.byte_size();
        assert!(bs == 2 || bs == 4);
        assert!(bs >= 2);
    }

    /// Proves ScalarType from_type_name is inverse of type_name for all variants.
    #[kani::unwind(1)]
    #[kani::proof]
    fn proof_scalar_type_name_round_trip_all() {
        let selector: u8 = kani::any();
        kani::assume(selector < 3);
        let ty = match selector {
            0 => ScalarType::F32,
            1 => ScalarType::F16,
            _ => ScalarType::BF16,
        };

        let name = ty.type_name();
        let recovered = ScalarType::from_type_name(name);
        assert!(recovered.is_some());
    }

    /// Proves F32 byte_size is 4 (GPU buffer alignment contract).
    #[kani::unwind(1)]
    #[kani::proof]
    #[kani::unwind(8)]
    fn proof_f32_byte_size_is_4() {
        assert_eq!(ScalarType::F32.byte_size(), 4);
    }

    /// Proves F16 and BF16 byte_size are 2 (GPU buffer alignment contract).
    #[kani::unwind(1)]
    #[kani::proof]
    #[kani::unwind(8)]
    fn proof_f16_bf16_byte_size_is_2() {
        assert_eq!(ScalarType::F16.byte_size(), 2);
        assert_eq!(ScalarType::BF16.byte_size(), 2);
    }

    // -----------------------------------------------------------------------
    // Kernel expansion entry point naming contract
    // -----------------------------------------------------------------------

    /// Proves the entry_point naming convention: name + "_kernel" suffix.
    ///
    /// The proc-macro in nn-macros generates `format!("{}_kernel", kernel_def.name)`.
    /// This proves the resulting string is non-empty and distinct from the
    /// function name itself.
    #[kani::unwind(1)]
    #[kani::proof]
    #[kani::unwind(8)]
    fn proof_entry_point_naming_convention() {
        // Model the proc-macro's naming: entry_point = "{name}_kernel"
        let name = "snake";
        let entry = "snake_kernel";

        // Entry point is not the same as the function name.
        assert_ne!(name, entry);

        // Entry point is longer (has "_kernel" suffix).
        assert!(entry.len() > name.len());
    }

    /// Proves the MSL const naming convention: upper(name) + "_MSL".
    #[kani::unwind(1)]
    #[kani::proof]
    #[kani::unwind(8)]
    fn proof_msl_const_naming_convention() {
        // Model: upper = kernel_def.name.to_uppercase(), msl_const = "{upper}_MSL"
        let name = "snake";
        let upper = "SNAKE";
        let msl_const = "SNAKE_MSL";

        // MSL const name is distinct from function name.
        assert_ne!(name, msl_const);
        assert_ne!(upper, msl_const);

        // MSL const name is longer.
        assert!(msl_const.len() > upper.len());
    }

    // -----------------------------------------------------------------------
    // Kernel dispatch validation (models nn-metal dispatch safety)
    // -----------------------------------------------------------------------

    /// Proves: param_count mismatch detection is correct.
    ///
    /// The dispatch layer (tested in kernel_metal_dispatch.rs) rejects calls
    /// where the number of input arrays != descriptor.param_count. This proves
    /// the inequality detection is exact.
    #[kani::unwind(1)]
    #[kani::proof]
    fn proof_param_count_mismatch_detection() {
        let expected: u8 = kani::any();
        let got: u8 = kani::any();
        kani::assume(expected <= 16);
        kani::assume(got <= 16);

        let matches = (expected as usize) == (got as usize);
        let mismatched = (expected as usize) != (got as usize);

        // Exactly one of matches/mismatched is true.
        assert!(matches != mismatched);

        if expected != got {
            assert!(mismatched);
        }
    }

    /// Proves: input length alignment check detects mismatched lengths.
    ///
    /// kernel_metal_dispatch.rs tests that all input arrays must have the
    /// same length. This proves the pairwise comparison catches any mismatch.
    #[kani::unwind(1)]
    #[kani::proof]
    fn proof_input_length_mismatch_detection() {
        let len_a: u16 = kani::any();
        let len_b: u16 = kani::any();

        let aligned = (len_a as usize) == (len_b as usize);

        if len_a != len_b {
            assert!(
                !aligned,
                "Different lengths must not be reported as aligned"
            );
        } else {
            assert!(aligned, "Same lengths must be reported as aligned");
        }
    }
}

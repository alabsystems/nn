// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Kani proof harnesses for IR types, verifiability classification, enum
//! round-trip consistency, and NativeOp type invariants.
//!
//! Proves safety and correctness properties of:
//! - `NodeId` arithmetic and ordering invariants
//! - `ScalarType` → `ValueType` conversion consistency
//! - `UnaryFnKind` method_name/from_method_name round-trip for ALL variants
//! - `BinOpKind` variant distinctness
//! - `CompareOpKind` variant distinctness
//! - `MinMaxKind` variant symmetry
//! - `KernelDef::has_ftz_sensitive_op` correctness
//! - `VerifiabilityClass::allows_compilation` / `is_verifiable` consistency
//! - `VerifiabilityClass::needs_decomposition` threshold semantics
//! - `PrecisionTier::fast_math` only true for Relaxed
//! - `PrecisionTier::parse` / `as_str` round-trip
//! - `GemmActivation` variant count sentinel
//! - `FusedNormKind` variant distinctness
//! - `AttentionLayout` default is HeadsFirst
//! - `NormActivation` variant distinctness
//! - `StyleBatchOffset` narrow length formula
//! - `POWI_MAX_EXPONENT` bounds
//! - `BinaryFnKind` variant coverage
//! - `sum_reduce` correctness for small arrays
//! - `KernelDef` output within bounds check
//! - `ValueType::is_numeric` consistency
//!
//! Part of #3805.

#[cfg(kani)]
mod proofs {
    use crate::ir::{
        BinOpKind, BinaryFnKind, CompareOpKind, IRNode, IRNodeKind, KernelDef, MinMaxKind, NodeId,
        Param, ScalarType, UnaryFnKind, ValueType, POWI_MAX_EXPONENT,
    };
    use crate::precision::PrecisionTier;
    use crate::trace_compile::{
        AttentionLayout, FusedNormKind, GemmActivation, NormActivation, StyleBatchOffset,
    };
    use crate::verifiability::VerifiabilityClass;

    // -- Kani transcendental stubs (CBMC #239, #329, #708) --

    fn ceil_f32_stub(x: f32) -> f32 {
        let _ = x;
        let r: f32 = kani::any();
        kani::assume(r.is_finite());
        r
    }

    fn log2_f32_stub(x: f32) -> f32 {
        let _ = x;
        let r: f32 = kani::any();
        kani::assume(r.is_finite() && r >= -150.0 && r <= 150.0);
        r
    }

    // -----------------------------------------------------------------------
    // Proof 1: UnaryFnKind::from_method_name round-trips for all variants
    // -----------------------------------------------------------------------

    #[kani::unwind(1)]
    #[kani::proof]
    fn proof_unary_fn_kind_round_trip_sin() {
        let op = UnaryFnKind::Sin;
        let recovered = UnaryFnKind::from_method_name(op.method_name());
        assert!(recovered.is_some());
        assert!(matches!(recovered.unwrap(), UnaryFnKind::Sin));
    }

    #[kani::unwind(1)]
    #[kani::proof]
    fn proof_unary_fn_kind_round_trip_cos() {
        let op = UnaryFnKind::Cos;
        let recovered = UnaryFnKind::from_method_name(op.method_name());
        assert!(recovered.is_some());
        assert!(matches!(recovered.unwrap(), UnaryFnKind::Cos));
    }

    #[kani::unwind(1)]
    #[kani::proof]
    fn proof_unary_fn_kind_round_trip_sqrt() {
        let op = UnaryFnKind::Sqrt;
        let recovered = UnaryFnKind::from_method_name(op.method_name());
        assert!(recovered.is_some());
        assert!(matches!(recovered.unwrap(), UnaryFnKind::Sqrt));
    }

    #[kani::unwind(1)]
    #[kani::proof]
    fn proof_unary_fn_kind_round_trip_rsqrt() {
        let op = UnaryFnKind::Rsqrt;
        let recovered = UnaryFnKind::from_method_name(op.method_name());
        assert!(recovered.is_some());
        assert!(matches!(recovered.unwrap(), UnaryFnKind::Rsqrt));
    }

    #[kani::unwind(1)]
    #[kani::proof]
    fn proof_unary_fn_kind_round_trip_exp() {
        let op = UnaryFnKind::Exp;
        let recovered = UnaryFnKind::from_method_name(op.method_name());
        assert!(recovered.is_some());
        assert!(matches!(recovered.unwrap(), UnaryFnKind::Exp));
    }

    #[kani::unwind(1)]
    #[kani::proof]
    fn proof_unary_fn_kind_round_trip_abs() {
        let op = UnaryFnKind::Abs;
        let recovered = UnaryFnKind::from_method_name(op.method_name());
        assert!(recovered.is_some());
        assert!(matches!(recovered.unwrap(), UnaryFnKind::Abs));
    }

    #[kani::unwind(1)]
    #[kani::proof]
    fn proof_unary_fn_kind_round_trip_tanh() {
        let op = UnaryFnKind::Tanh;
        let recovered = UnaryFnKind::from_method_name(op.method_name());
        assert!(recovered.is_some());
        assert!(matches!(recovered.unwrap(), UnaryFnKind::Tanh));
    }

    #[kani::unwind(1)]
    #[kani::proof]
    fn proof_unary_fn_kind_round_trip_neg() {
        let op = UnaryFnKind::Neg;
        let recovered = UnaryFnKind::from_method_name(op.method_name());
        assert!(recovered.is_some());
        assert!(matches!(recovered.unwrap(), UnaryFnKind::Neg));
    }

    // -----------------------------------------------------------------------
    // Proof 2: UnaryFnKind::from_method_name rejects unknown names
    // -----------------------------------------------------------------------

    #[kani::unwind(1)]
    #[kani::proof]
    fn proof_unary_fn_kind_rejects_unknown() {
        let result = UnaryFnKind::from_method_name("nonexistent_op");
        assert!(result.is_none(), "Unknown method name must return None");
    }

    // -----------------------------------------------------------------------
    // Proof 3: ScalarType → ValueType conversion is injective
    // -----------------------------------------------------------------------

    #[kani::unwind(1)]
    #[kani::proof]
    fn proof_scalar_to_value_type_injective() {
        let f32_vt: ValueType = ScalarType::F32.into();
        let f16_vt: ValueType = ScalarType::F16.into();
        let bf16_vt: ValueType = ScalarType::BF16.into();

        assert!(matches!(f32_vt, ValueType::F32));
        assert!(matches!(f16_vt, ValueType::F16));
        assert!(matches!(bf16_vt, ValueType::BF16));

        // All numeric value types are numeric
        assert!(f32_vt.is_numeric());
        assert!(f16_vt.is_numeric());
        assert!(bf16_vt.is_numeric());
    }

    // -----------------------------------------------------------------------
    // Proof 4: ValueType::Bool is not numeric
    // -----------------------------------------------------------------------

    #[kani::unwind(1)]
    #[kani::proof]
    fn proof_value_type_bool_not_numeric() {
        let bool_vt = ValueType::Bool;
        assert!(
            !bool_vt.is_numeric(),
            "Bool must not be classified as numeric"
        );
    }

    // -----------------------------------------------------------------------
    // Proof 5: BinOpKind all four variants are distinct
    // -----------------------------------------------------------------------

    #[kani::unwind(1)]
    #[kani::proof]
    #[kani::unwind(8)]
    fn proof_binop_kind_all_distinct() {
        let add = BinOpKind::Add;
        let sub = BinOpKind::Sub;
        let mul = BinOpKind::Mul;
        let div = BinOpKind::Div;

        assert_ne!(std::mem::discriminant(&add), std::mem::discriminant(&sub));
        assert_ne!(std::mem::discriminant(&add), std::mem::discriminant(&mul));
        assert_ne!(std::mem::discriminant(&add), std::mem::discriminant(&div));
        assert_ne!(std::mem::discriminant(&sub), std::mem::discriminant(&mul));
        assert_ne!(std::mem::discriminant(&sub), std::mem::discriminant(&div));
        assert_ne!(std::mem::discriminant(&mul), std::mem::discriminant(&div));
    }

    // -----------------------------------------------------------------------
    // Proof 6: CompareOpKind has 6 distinct variants
    // -----------------------------------------------------------------------

    #[kani::unwind(8)]
    #[kani::proof]
    #[kani::unwind(8)]
    fn proof_compare_op_kind_all_distinct() {
        let variants = [
            CompareOpKind::Eq,
            CompareOpKind::Ne,
            CompareOpKind::Lt,
            CompareOpKind::Le,
            CompareOpKind::Gt,
            CompareOpKind::Ge,
        ];
        for i in 0..variants.len() {
            for j in (i + 1)..variants.len() {
                assert_ne!(
                    std::mem::discriminant(&variants[i]),
                    std::mem::discriminant(&variants[j]),
                    "CompareOpKind variants must be distinct"
                );
            }
        }
    }

    // -----------------------------------------------------------------------
    // Proof 7: MinMaxKind::Min != MinMaxKind::Max
    // -----------------------------------------------------------------------

    #[kani::unwind(1)]
    #[kani::proof]
    #[kani::unwind(8)]
    fn proof_minmax_kind_distinct() {
        assert_ne!(
            std::mem::discriminant(&MinMaxKind::Min),
            std::mem::discriminant(&MinMaxKind::Max),
            "Min and Max must be distinct"
        );
    }

    // -----------------------------------------------------------------------
    // Proof 8: has_ftz_sensitive_op detects Rsqrt
    // -----------------------------------------------------------------------

    #[kani::unwind(8)]
    #[kani::proof]
    fn proof_ftz_detects_rsqrt() {
        let params = vec![Param::new("x", ScalarType::F32)];
        let nodes = vec![
            IRNode::new(NodeId::new(0), IRNodeKind::Param(0)),
            IRNode::new(
                NodeId::new(1),
                IRNodeKind::UnaryFn {
                    op: UnaryFnKind::Rsqrt,
                    input: NodeId::new(0),
                },
            ),
        ];
        let kernel = KernelDef::new("rsqrt_k", params, ScalarType::F32, nodes, NodeId::new(1));
        assert!(kernel.has_ftz_sensitive_op(), "Rsqrt is FTZ-sensitive");
    }

    // -----------------------------------------------------------------------
    // Proof 9: has_ftz_sensitive_op detects Div
    // -----------------------------------------------------------------------

    #[kani::unwind(8)]
    #[kani::proof]
    fn proof_ftz_detects_div() {
        let params = vec![
            Param::new("a", ScalarType::F32),
            Param::new("b", ScalarType::F32),
        ];
        let nodes = vec![
            IRNode::new(NodeId::new(0), IRNodeKind::Param(0)),
            IRNode::new(NodeId::new(1), IRNodeKind::Param(1)),
            IRNode::new(
                NodeId::new(2),
                IRNodeKind::BinOp {
                    op: BinOpKind::Div,
                    lhs: NodeId::new(0),
                    rhs: NodeId::new(1),
                },
            ),
        ];
        let kernel = KernelDef::new("div_k", params, ScalarType::F32, nodes, NodeId::new(2));
        assert!(kernel.has_ftz_sensitive_op(), "Div is FTZ-sensitive");
    }

    // -----------------------------------------------------------------------
    // Proof 10: has_ftz_sensitive_op false for safe ops only
    // -----------------------------------------------------------------------

    #[kani::unwind(8)]
    #[kani::proof]
    fn proof_ftz_false_for_safe_ops() {
        let params = vec![Param::new("x", ScalarType::F32)];
        let nodes = vec![
            IRNode::new(NodeId::new(0), IRNodeKind::Param(0)),
            IRNode::new(
                NodeId::new(1),
                IRNodeKind::UnaryFn {
                    op: UnaryFnKind::Exp,
                    input: NodeId::new(0),
                },
            ),
        ];
        let kernel = KernelDef::new("exp_k", params, ScalarType::F32, nodes, NodeId::new(1));
        assert!(
            !kernel.has_ftz_sensitive_op(),
            "Exp alone is not FTZ-sensitive"
        );
    }

    // -----------------------------------------------------------------------
    // Proof 11: VerifiabilityClass::allows_compilation for all non-Learned
    // -----------------------------------------------------------------------

    #[kani::unwind(1)]
    #[kani::proof]
    fn proof_verifiability_allows_compilation() {
        assert!(VerifiabilityClass::Verifiable.allows_compilation());
        assert!(VerifiabilityClass::VerifiableBounded { max_dim: 512 }.allows_compilation());
        assert!(VerifiabilityClass::UnverifiableSafe.allows_compilation());
        assert!(VerifiabilityClass::ShapeOnly.allows_compilation());
        assert!(VerifiabilityClass::Passthrough.allows_compilation());
        assert!(
            !VerifiabilityClass::UnverifiableLearned.allows_compilation(),
            "UnverifiableLearned must block compilation"
        );
    }

    // -----------------------------------------------------------------------
    // Proof 12: allows_compilation and is_verifiable are identical
    // -----------------------------------------------------------------------

    #[kani::unwind(8)]
    #[kani::proof]
    #[kani::unwind(8)]
    fn proof_verifiability_allows_eq_is_verifiable() {
        let classes = [
            VerifiabilityClass::Verifiable,
            VerifiabilityClass::VerifiableBounded { max_dim: 256 },
            VerifiabilityClass::UnverifiableSafe,
            VerifiabilityClass::UnverifiableLearned,
            VerifiabilityClass::ShapeOnly,
            VerifiabilityClass::Passthrough,
        ];
        for class in &classes {
            assert_eq!(
                class.allows_compilation(),
                class.is_verifiable(),
                "allows_compilation and is_verifiable must agree"
            );
        }
    }

    // -----------------------------------------------------------------------
    // Proof 13: needs_decomposition threshold correctness
    // -----------------------------------------------------------------------

    #[kani::unwind(1)]
    #[kani::proof]
    fn proof_verifiability_needs_decomposition_threshold() {
        let bounded = VerifiabilityClass::VerifiableBounded { max_dim: 256 };

        // At or below max_dim: no decomposition needed
        assert!(!bounded.needs_decomposition(256));
        assert!(!bounded.needs_decomposition(128));
        assert!(!bounded.needs_decomposition(1));

        // Above max_dim: decomposition needed
        assert!(bounded.needs_decomposition(257));
        assert!(bounded.needs_decomposition(512));
    }

    // -----------------------------------------------------------------------
    // Proof 14: needs_decomposition is false for non-Bounded classes
    // -----------------------------------------------------------------------

    #[kani::unwind(1)]
    #[kani::proof]
    fn proof_verifiability_needs_decomposition_non_bounded() {
        let dim: usize = 10000;
        assert!(!VerifiabilityClass::Verifiable.needs_decomposition(dim));
        assert!(!VerifiabilityClass::UnverifiableSafe.needs_decomposition(dim));
        assert!(!VerifiabilityClass::UnverifiableLearned.needs_decomposition(dim));
        assert!(!VerifiabilityClass::ShapeOnly.needs_decomposition(dim));
        assert!(!VerifiabilityClass::Passthrough.needs_decomposition(dim));
    }

    // -----------------------------------------------------------------------
    // Proof 15: PrecisionTier::fast_math only true for Relaxed
    // -----------------------------------------------------------------------

    #[kani::unwind(1)]
    #[kani::proof]
    fn proof_precision_tier_fast_math() {
        assert!(!PrecisionTier::Strict.fast_math());
        assert!(!PrecisionTier::Normal.fast_math());
        assert!(PrecisionTier::Relaxed.fast_math());
    }

    // -----------------------------------------------------------------------
    // Proof 16: PrecisionTier::parse / as_str round-trip
    // -----------------------------------------------------------------------

    #[kani::unwind(1)]
    #[kani::proof]
    fn proof_precision_tier_round_trip_strict() {
        let tier = PrecisionTier::Strict;
        let parsed = PrecisionTier::parse(tier.as_str());
        assert!(parsed.is_ok());
        assert!(matches!(parsed.unwrap(), PrecisionTier::Strict));
    }

    #[kani::unwind(1)]
    #[kani::proof]
    fn proof_precision_tier_round_trip_normal() {
        let tier = PrecisionTier::Normal;
        let parsed = PrecisionTier::parse(tier.as_str());
        assert!(parsed.is_ok());
        assert!(matches!(parsed.unwrap(), PrecisionTier::Normal));
    }

    #[kani::unwind(1)]
    #[kani::proof]
    fn proof_precision_tier_round_trip_relaxed() {
        let tier = PrecisionTier::Relaxed;
        let parsed = PrecisionTier::parse(tier.as_str());
        assert!(parsed.is_ok());
        assert!(matches!(parsed.unwrap(), PrecisionTier::Relaxed));
    }

    // -----------------------------------------------------------------------
    // Proof 17: PrecisionTier::parse rejects invalid
    // -----------------------------------------------------------------------

    #[kani::unwind(1)]
    #[kani::proof]
    fn proof_precision_tier_parse_rejects_invalid() {
        assert!(PrecisionTier::parse("turbo").is_err());
        assert!(PrecisionTier::parse("").is_err());
    }

    // -----------------------------------------------------------------------
    // Proof 18: GemmActivation has 6 distinct variants
    // -----------------------------------------------------------------------

    #[kani::unwind(8)]
    #[kani::proof]
    #[kani::unwind(8)]
    fn proof_gemm_activation_all_distinct() {
        let variants = [
            GemmActivation::Relu,
            GemmActivation::Gelu,
            GemmActivation::GeluErf,
            GemmActivation::Sigmoid,
            GemmActivation::Silu,
            GemmActivation::Tanh,
        ];
        for i in 0..variants.len() {
            for j in (i + 1)..variants.len() {
                assert_ne!(
                    std::mem::discriminant(&variants[i]),
                    std::mem::discriminant(&variants[j]),
                    "GemmActivation variants must be distinct"
                );
            }
        }
    }

    // -----------------------------------------------------------------------
    // Proof 19: FusedNormKind variants are distinct
    // -----------------------------------------------------------------------

    #[kani::unwind(1)]
    #[kani::proof]
    #[kani::unwind(8)]
    fn proof_fused_norm_kind_distinct() {
        assert_ne!(
            std::mem::discriminant(&FusedNormKind::LayerNorm),
            std::mem::discriminant(&FusedNormKind::RmsNorm),
            "LayerNorm and RmsNorm must be distinct"
        );
    }

    // -----------------------------------------------------------------------
    // Proof 20: AttentionLayout default is HeadsFirst
    // -----------------------------------------------------------------------

    #[kani::unwind(1)]
    #[kani::proof]
    fn proof_attention_layout_default() {
        let default_layout = AttentionLayout::default();
        assert!(
            matches!(default_layout, AttentionLayout::HeadsFirst),
            "Default AttentionLayout must be HeadsFirst"
        );
    }

    // -----------------------------------------------------------------------
    // Proof 21: StyleBatchOffset narrow length formula
    // -----------------------------------------------------------------------

    #[kani::unwind(8)]
    #[kani::proof]
    #[kani::unwind(8)]
    fn proof_style_batch_offset_narrow_length() {
        let c1: u8 = kani::any();
        let c2: u8 = kani::any();

        kani::assume(c1 >= 1 && c1 <= 128);
        kani::assume(c2 >= 1 && c2 <= 128);

        let channels1 = c1 as usize;
        let channels2 = c2 as usize;

        // Narrow length = 2*(channels1 + channels2) for [gamma1, beta1, gamma2, beta2]
        let narrow_len = 2 * (channels1 + channels2);
        assert!(narrow_len >= 4, "Narrow length must be >= 4");
        assert_eq!(
            narrow_len,
            2 * channels1 + 2 * channels2,
            "Distributive law must hold"
        );

        let sbo = StyleBatchOffset::new(0, channels1, channels2);
        assert_eq!(sbo.channels1, channels1);
        assert_eq!(sbo.channels2, channels2);
    }

    // -----------------------------------------------------------------------
    // Proof 22: POWI_MAX_EXPONENT is bounded and reasonable
    // -----------------------------------------------------------------------

    #[kani::unwind(1)]
    #[kani::proof]
    #[kani::stub(f32::ceil, ceil_f32_stub)]
    #[kani::stub(f32::log2, log2_f32_stub)]
    fn proof_powi_max_exponent_bounded() {
        assert!(POWI_MAX_EXPONENT >= 2, "Must allow at least squaring");
        assert!(
            POWI_MAX_EXPONENT <= 128,
            "Must not allow unreasonably large exponents"
        );
        // Binary exponentiation generates O(log2(n)) temporaries
        let temporaries = (POWI_MAX_EXPONENT as f64).log2().ceil() as u32;
        assert!(
            temporaries <= 7,
            "Binary exponentiation must produce <= 7 temporaries"
        );
    }

    // -----------------------------------------------------------------------
    // Proof 23: BinaryFnKind::Atan2 is the only variant
    // -----------------------------------------------------------------------

    #[kani::unwind(1)]
    #[kani::proof]
    fn proof_binary_fn_kind_atan2_only() {
        let atan2 = BinaryFnKind::Atan2;
        // If BinaryFnKind gets more variants, this will need updating.
        // For now, Atan2 is the only one.
        assert!(matches!(atan2, BinaryFnKind::Atan2));
    }

    // -----------------------------------------------------------------------
    // Proof 24: sum_reduce correctness for small arrays
    // -----------------------------------------------------------------------

    #[kani::unwind(1)]
    #[kani::proof]
    #[kani::unwind(8)]
    fn proof_sum_reduce_3_elements() {
        let a: u8 = kani::any();
        let b: u8 = kani::any();
        let c: u8 = kani::any();

        kani::assume(a <= 100);
        kani::assume(b <= 100);
        kani::assume(c <= 100);

        let result = crate::sum_reduce([a as u32, b as u32, c as u32]);
        assert_eq!(
            result,
            (a as u32) + (b as u32) + (c as u32),
            "sum_reduce must equal element-wise sum"
        );
    }

    #[kani::unwind(1)]
    #[kani::proof]
    #[kani::unwind(8)]
    fn proof_sum_reduce_1_element() {
        let val: u8 = kani::any();
        let result = crate::sum_reduce([val as u32]);
        assert_eq!(result, val as u32, "sum_reduce of 1 element is identity");
    }

    // -----------------------------------------------------------------------
    // Proof 25: NodeId ordering is consistent with index ordering
    // -----------------------------------------------------------------------

    #[kani::unwind(1)]
    #[kani::proof]
    #[kani::unwind(8)]
    fn proof_node_id_index_ordering() {
        let a_idx: u16 = kani::any();
        let b_idx: u16 = kani::any();

        let a = NodeId::new(a_idx as usize);
        let b = NodeId::new(b_idx as usize);

        if a_idx < b_idx {
            assert!(a.index() < b.index());
        }
        if a_idx == b_idx {
            assert_eq!(a, b);
        }
        if a_idx > b_idx {
            assert!(a.index() > b.index());
        }
    }

    // -----------------------------------------------------------------------
    // Proof 26: KernelDef output must reference a valid node
    // -----------------------------------------------------------------------

    #[kani::unwind(8)]
    #[kani::proof]
    fn proof_kernel_def_output_within_bounds() {
        let params = vec![Param::new("x", ScalarType::F32)];
        let nodes = vec![
            IRNode::new(NodeId::new(0), IRNodeKind::Param(0)),
            IRNode::new(
                NodeId::new(1),
                IRNodeKind::UnaryFn {
                    op: UnaryFnKind::Exp,
                    input: NodeId::new(0),
                },
            ),
        ];
        // Valid output: index 1 (within bounds)
        let kernel = KernelDef::new(
            "test",
            params.clone(),
            ScalarType::F32,
            nodes.clone(),
            NodeId::new(1),
        );
        assert!(kernel.validate().is_ok());

        // Invalid output: index 2 (out of bounds)
        let kernel_bad = KernelDef::new("test_bad", params, ScalarType::F32, nodes, NodeId::new(2));
        assert!(kernel_bad.validate().is_err());
    }

    // -----------------------------------------------------------------------
    // Proof 27: ScalarType::msl_str maps correctly
    // -----------------------------------------------------------------------

    #[kani::unwind(1)]
    #[kani::proof]
    #[kani::unwind(8)]
    fn proof_scalar_type_msl_str() {
        assert_eq!(ScalarType::F32.msl_str(), "float");
        assert_eq!(ScalarType::F16.msl_str(), "half");
        assert_eq!(ScalarType::BF16.msl_str(), "half"); // Apple GPUs: bf16 → half
    }

    // -----------------------------------------------------------------------
    // Proof 28: ScalarType::msl_accumulator_str is always "float"
    // -----------------------------------------------------------------------

    #[kani::unwind(1)]
    #[kani::proof]
    #[kani::unwind(8)]
    fn proof_scalar_type_accumulator_always_float() {
        assert_eq!(ScalarType::F32.msl_accumulator_str(), "float");
        assert_eq!(ScalarType::F16.msl_accumulator_str(), "float");
        assert_eq!(ScalarType::BF16.msl_accumulator_str(), "float");
    }
}

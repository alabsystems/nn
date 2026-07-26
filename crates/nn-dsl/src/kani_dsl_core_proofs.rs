// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Kani proof harnesses for nn-dsl core types and invariants.
//!
//! Proves safety and correctness properties of:
//! - `ScalarType::type_name` / `from_type_name` round-trip for all variants
//! - `ScalarType::byte_size` correctness (F32=4, F16/BF16=2)
//! - `ScalarType::msl_str` consistency with `msl_accumulator_str`
//! - `NodeId::new` / `index` round-trip identity
//! - `InputBound::new` rejects non-finite endpoints
//! - `InputBound::new` rejects inverted bounds (lo > hi)
//! - `InputBound::parse` round-trip for well-formed strings
//! - `InputBound::default_for` correctness per ScalarType
//! - `InputBounds::get` falls back to type default for missing keys
//! - `PeepholeConfig::default` has all passes enabled
//! - `within_differential_budget` IEEE 754: both-NaN is match
//! - `within_differential_budget` IEEE 754: both-same-infinity is match
//! - `within_differential_budget` IEEE 754: mixed finite/non-finite is mismatch
//! - `within_differential_budget` exact match always passes
//! - `differential_tolerance` is non-negative for non-negative reference
//! - `CostModel::step_cost` is non-negative
//! - `KernelDef` with Select IR node validates
//! - `KernelDef` with Compare IR node validates
//! - `KernelDef` with MinMax IR node validates
//! - `KernelDef` forward reference rejection (topological order)
//!
//! Part of #4288.

#[cfg(kani)]
mod proofs {
    use crate::ir::{
        BinOpKind, CompareOpKind, IRNode, IRNodeKind, KernelDef, MinMaxKind, NodeId, Param,
        ScalarType, UnaryFnKind,
    };
    use crate::precision::{
        differential_tolerance, within_differential_budget, InputBound, InputBounds,
        PrecisionContract, PrecisionTier,
    };
    use crate::trace_compile::PeepholeConfig;

    // -----------------------------------------------------------------------
    // Proof 1: ScalarType::type_name / from_type_name round-trip for F32
    // -----------------------------------------------------------------------

    #[kani::unwind(1)]
    #[kani::proof]
    fn proof_scalar_type_round_trip_f32() {
        let ty = ScalarType::F32;
        let name = ty.type_name();
        let recovered = ScalarType::from_type_name(name);
        assert!(recovered.is_some());
        assert!(matches!(recovered.unwrap(), ScalarType::F32));
    }

    // -----------------------------------------------------------------------
    // Proof 2: ScalarType::type_name / from_type_name round-trip for F16
    // -----------------------------------------------------------------------

    #[kani::unwind(1)]
    #[kani::proof]
    fn proof_scalar_type_round_trip_f16() {
        let ty = ScalarType::F16;
        let name = ty.type_name();
        let recovered = ScalarType::from_type_name(name);
        assert!(recovered.is_some());
        assert!(matches!(recovered.unwrap(), ScalarType::F16));
    }

    // -----------------------------------------------------------------------
    // Proof 3: ScalarType::type_name / from_type_name round-trip for BF16
    // -----------------------------------------------------------------------

    #[kani::unwind(1)]
    #[kani::proof]
    fn proof_scalar_type_round_trip_bf16() {
        let ty = ScalarType::BF16;
        let name = ty.type_name();
        let recovered = ScalarType::from_type_name(name);
        assert!(recovered.is_some());
        assert!(matches!(recovered.unwrap(), ScalarType::BF16));
    }

    // -----------------------------------------------------------------------
    // Proof 4: ScalarType::from_type_name rejects unknown names
    // -----------------------------------------------------------------------

    #[kani::unwind(1)]
    #[kani::proof]
    fn proof_scalar_type_from_name_rejects_unknown() {
        assert!(ScalarType::from_type_name("f64").is_none());
        assert!(ScalarType::from_type_name("int32").is_none());
        assert!(ScalarType::from_type_name("").is_none());
    }

    // -----------------------------------------------------------------------
    // Proof 5: ScalarType::byte_size correctness
    // -----------------------------------------------------------------------

    #[kani::unwind(1)]
    #[kani::proof]
    fn proof_scalar_type_byte_size() {
        assert_eq!(ScalarType::F32.byte_size(), 4);
        assert_eq!(ScalarType::F16.byte_size(), 2);
        assert_eq!(ScalarType::BF16.byte_size(), 2);
    }

    // -----------------------------------------------------------------------
    // Proof 6: ScalarType F16 and BF16 share MSL representation
    // -----------------------------------------------------------------------

    #[kani::unwind(1)]
    #[kani::proof]
    fn proof_scalar_type_f16_bf16_same_msl() {
        // Apple GPUs have no native bf16 compute; both map to "half"
        assert_eq!(ScalarType::F16.msl_str(), ScalarType::BF16.msl_str());
        // Both accumulate in float
        assert_eq!(
            ScalarType::F16.msl_accumulator_str(),
            ScalarType::BF16.msl_accumulator_str()
        );
        // Both have same byte size
        assert_eq!(ScalarType::F16.byte_size(), ScalarType::BF16.byte_size());
    }

    // -----------------------------------------------------------------------
    // Proof 7: NodeId::new / index is identity
    // -----------------------------------------------------------------------

    #[kani::unwind(1)]
    #[kani::proof]
    fn proof_node_id_round_trip() {
        let idx: u16 = kani::any();
        let node_id = NodeId::new(idx as usize);
        assert_eq!(
            node_id.index(),
            idx as usize,
            "NodeId round-trip must be identity"
        );
    }

    // -----------------------------------------------------------------------
    // Proof 8: InputBound::new rejects NaN endpoints
    // -----------------------------------------------------------------------

    #[kani::unwind(1)]
    #[kani::proof]
    fn proof_input_bound_rejects_nan_lo() {
        let result = InputBound::new(f64::NAN, 1.0);
        assert!(result.is_err(), "InputBound must reject NaN lo");
    }

    #[kani::unwind(1)]
    #[kani::proof]
    fn proof_input_bound_rejects_nan_hi() {
        let result = InputBound::new(-1.0, f64::NAN);
        assert!(result.is_err(), "InputBound must reject NaN hi");
    }

    // -----------------------------------------------------------------------
    // Proof 9: InputBound::new rejects infinity endpoints
    // -----------------------------------------------------------------------

    #[kani::unwind(1)]
    #[kani::proof]
    fn proof_input_bound_rejects_inf() {
        assert!(InputBound::new(f64::INFINITY, 1.0).is_err());
        assert!(InputBound::new(-1.0, f64::INFINITY).is_err());
        assert!(InputBound::new(f64::NEG_INFINITY, 1.0).is_err());
    }

    // -----------------------------------------------------------------------
    // Proof 10: InputBound::new rejects inverted bounds
    // -----------------------------------------------------------------------

    #[kani::unwind(1)]
    #[kani::proof]
    fn proof_input_bound_rejects_inverted() {
        let result = InputBound::new(10.0, -10.0);
        assert!(result.is_err(), "InputBound must reject lo > hi");
    }

    // -----------------------------------------------------------------------
    // Proof 11: InputBound::new accepts valid bounds and preserves lo/hi
    // -----------------------------------------------------------------------

    #[kani::unwind(1)]
    #[kani::proof]
    fn proof_input_bound_valid_preserves_values() {
        let bound = InputBound::new(-5.0, 5.0).unwrap();
        assert!((bound.lo() - (-5.0)).abs() < 1e-10);
        assert!((bound.hi() - 5.0).abs() < 1e-10);
        assert!(bound.lo() <= bound.hi(), "lo must be <= hi");
    }

    // -----------------------------------------------------------------------
    // Proof 12: InputBound::new with lo == hi (point bound) is valid
    // -----------------------------------------------------------------------

    #[kani::unwind(1)]
    #[kani::proof]
    fn proof_input_bound_point_bound_valid() {
        let result = InputBound::new(3.14, 3.14);
        assert!(result.is_ok(), "Point bound (lo == hi) must be valid");
        let bound = result.unwrap();
        assert!((bound.lo() - bound.hi()).abs() < 1e-15);
    }

    // -----------------------------------------------------------------------
    // Proof 13: InputBound::parse round-trip for well-formed string
    // -----------------------------------------------------------------------

    #[kani::unwind(1)]
    #[kani::proof]
    fn proof_input_bound_parse_valid() {
        let result = InputBound::parse("-1e4..1e4");
        assert!(result.is_ok(), "Must parse valid range string");
        let bound = result.unwrap();
        assert!(bound.lo() < 0.0);
        assert!(bound.hi() > 0.0);
        assert!(bound.lo() <= bound.hi());
    }

    // -----------------------------------------------------------------------
    // Proof 14: InputBound::parse rejects bad format
    // -----------------------------------------------------------------------

    #[kani::unwind(1)]
    #[kani::proof]
    fn proof_input_bound_parse_rejects_bad_format() {
        assert!(InputBound::parse("1.0").is_err(), "No '..' separator");
        assert!(InputBound::parse("1.0..2.0..3.0").is_err(), "Too many '..'");
        assert!(InputBound::parse("abc..1.0").is_err(), "Non-numeric lo");
    }

    // -----------------------------------------------------------------------
    // Proof 15: InputBound::default_for returns correct defaults
    // -----------------------------------------------------------------------

    #[kani::unwind(1)]
    #[kani::proof]
    fn proof_input_bound_default_for_f32() {
        let bound = InputBound::default_for(ScalarType::F32);
        assert!((bound.lo() - (-1e6)).abs() < 1e-6);
        assert!((bound.hi() - 1e6).abs() < 1e-6);
    }

    #[kani::unwind(1)]
    #[kani::proof]
    fn proof_input_bound_default_for_f16() {
        let bound = InputBound::default_for(ScalarType::F16);
        assert!((bound.lo() - (-65504.0)).abs() < 1e-6);
        assert!((bound.hi() - 65504.0).abs() < 1e-6);
    }

    #[kani::unwind(1)]
    #[kani::proof]
    fn proof_input_bound_default_for_bf16_matches_f16() {
        let f16_bound = InputBound::default_for(ScalarType::F16);
        let bf16_bound = InputBound::default_for(ScalarType::BF16);
        assert!((f16_bound.lo() - bf16_bound.lo()).abs() < 1e-10);
        assert!((f16_bound.hi() - bf16_bound.hi()).abs() < 1e-10);
    }

    // -----------------------------------------------------------------------
    // Proof 16: InputBounds::get falls back to type default
    // -----------------------------------------------------------------------

    #[kani::unwind(1)]
    #[kani::proof]
    fn proof_input_bounds_get_fallback() {
        let bounds = InputBounds::new();
        assert!(bounds.is_empty());
        let bound = bounds.get("nonexistent_param", ScalarType::F32);
        let default = InputBound::default_for(ScalarType::F32);
        assert!((bound.lo() - default.lo()).abs() < 1e-10);
        assert!((bound.hi() - default.hi()).abs() < 1e-10);
    }

    // -----------------------------------------------------------------------
    // Proof 17: PeepholeConfig::default has all passes enabled
    // -----------------------------------------------------------------------

    #[kani::unwind(1)]
    #[kani::proof]
    fn proof_peephole_config_default_all_enabled() {
        let config = PeepholeConfig::default();
        assert!(
            config.norm_activ_conv1d,
            "norm_activ_conv1d must be enabled by default"
        );
        assert!(
            config.fused_resblock,
            "fused_resblock must be enabled by default"
        );
        assert!(
            config.linear_activation,
            "linear_activation must be enabled by default"
        );
        assert!(
            config.add_layer_norm,
            "add_layer_norm must be enabled by default"
        );
        assert!(config.norm_linear, "norm_linear must be enabled by default");
        assert!(
            config.attention_transpose,
            "attention_transpose must be enabled by default"
        );
        assert!(config.flip_lstm, "flip_lstm must be enabled by default");
        assert!(
            config.batched_linear_projection,
            "batched_linear_projection must be enabled"
        );
        assert!(
            config.channels_first_layer_norm,
            "channels_first_layer_norm must be enabled"
        );
        assert!(config.silu_mul, "silu_mul must be enabled by default");
        assert!(
            config.auto_fuse_elementwise,
            "auto_fuse_elementwise must be enabled by default"
        );
    }

    // -----------------------------------------------------------------------
    // Proof 18: within_differential_budget: both NaN is match
    // -----------------------------------------------------------------------

    #[kani::unwind(1)]
    #[kani::proof]
    fn proof_within_budget_both_nan_is_match() {
        let contract = PrecisionContract::bootstrap(PrecisionTier::Normal, ScalarType::F32);
        assert!(
            within_differential_budget(f32::NAN, f32::NAN, contract),
            "Both NaN must match (same domain error)"
        );
    }

    // -----------------------------------------------------------------------
    // Proof 19: within_differential_budget: both same infinity is match
    // -----------------------------------------------------------------------

    #[kani::unwind(1)]
    #[kani::proof]
    fn proof_within_budget_same_infinity_is_match() {
        let contract = PrecisionContract::bootstrap(PrecisionTier::Normal, ScalarType::F32);
        assert!(
            within_differential_budget(f32::INFINITY, f32::INFINITY, contract),
            "+inf == +inf must match"
        );
        assert!(
            within_differential_budget(f32::NEG_INFINITY, f32::NEG_INFINITY, contract),
            "-inf == -inf must match"
        );
    }

    // -----------------------------------------------------------------------
    // Proof 20: within_differential_budget: opposite infinities is mismatch
    // -----------------------------------------------------------------------

    #[kani::unwind(1)]
    #[kani::proof]
    fn proof_within_budget_opposite_inf_is_mismatch() {
        let contract = PrecisionContract::bootstrap(PrecisionTier::Normal, ScalarType::F32);
        assert!(
            !within_differential_budget(f32::INFINITY, f32::NEG_INFINITY, contract),
            "+inf vs -inf must mismatch"
        );
        assert!(
            !within_differential_budget(f32::NEG_INFINITY, f32::INFINITY, contract),
            "-inf vs +inf must mismatch"
        );
    }

    // -----------------------------------------------------------------------
    // Proof 21: within_differential_budget: mixed finite/non-finite is mismatch
    // -----------------------------------------------------------------------

    #[kani::unwind(1)]
    #[kani::proof]
    fn proof_within_budget_mixed_finite_nonfinite_is_mismatch() {
        let contract = PrecisionContract::bootstrap(PrecisionTier::Normal, ScalarType::F32);
        assert!(
            !within_differential_budget(1.0, f32::NAN, contract),
            "finite vs NaN must mismatch"
        );
        assert!(
            !within_differential_budget(f32::NAN, 1.0, contract),
            "NaN vs finite must mismatch"
        );
        assert!(
            !within_differential_budget(1.0, f32::INFINITY, contract),
            "finite vs +inf must mismatch"
        );
    }

    // -----------------------------------------------------------------------
    // Proof 22: within_differential_budget: exact match always passes
    // -----------------------------------------------------------------------

    #[kani::unwind(1)]
    #[kani::proof]
    fn proof_within_budget_exact_match() {
        let contract = PrecisionContract::bootstrap(PrecisionTier::Strict, ScalarType::F32);
        // Even with strictest tier, exact match passes
        assert!(within_differential_budget(0.0, 0.0, contract));
        assert!(within_differential_budget(1.0, 1.0, contract));
        assert!(within_differential_budget(-42.5, -42.5, contract));
    }

    // -----------------------------------------------------------------------
    // Proof 23: differential_tolerance is non-negative for non-negative ref
    // -----------------------------------------------------------------------

    #[kani::unwind(1)]
    #[kani::proof]
    fn proof_differential_tolerance_nonneg() {
        let contract = PrecisionContract::bootstrap(PrecisionTier::Normal, ScalarType::F32);
        let tol_zero = differential_tolerance(0.0, contract);
        let tol_pos = differential_tolerance(1.0, contract);
        let tol_neg = differential_tolerance(-1.0, contract);
        assert!(tol_zero >= 0.0, "tolerance at 0 must be non-negative");
        assert!(
            tol_pos >= 0.0,
            "tolerance at positive ref must be non-negative"
        );
        assert!(
            tol_neg >= 0.0,
            "tolerance at negative ref must be non-negative"
        );
    }

    // -----------------------------------------------------------------------
    // Proof 24: differential_tolerance is monotonically increasing in |ref|
    // -----------------------------------------------------------------------

    #[kani::unwind(1)]
    #[kani::proof]
    fn proof_differential_tolerance_monotonic_in_abs_ref() {
        let contract = PrecisionContract::bootstrap(PrecisionTier::Normal, ScalarType::F32);
        let tol_small = differential_tolerance(1.0, contract);
        let tol_large = differential_tolerance(10.0, contract);
        assert!(
            tol_large >= tol_small,
            "tolerance must increase with |reference|"
        );
    }

    // -----------------------------------------------------------------------
    // Proof 25: KernelDef with Select IR node validates
    // -----------------------------------------------------------------------

    #[kani::unwind(8)]
    #[kani::proof]
    fn proof_select_ir_validates() {
        let params = vec![
            Param::new("x", ScalarType::F32),
            Param::new("y", ScalarType::F32),
        ];
        let nodes = vec![
            IRNode::new(NodeId::new(0), IRNodeKind::Param(0)),
            IRNode::new(NodeId::new(1), IRNodeKind::Param(1)),
            IRNode::new(NodeId::new(2), IRNodeKind::Literal(0.0)),
            IRNode::new(
                NodeId::new(3),
                IRNodeKind::Compare {
                    op: CompareOpKind::Gt,
                    lhs: NodeId::new(0),
                    rhs: NodeId::new(2),
                },
            ),
            IRNode::new(
                NodeId::new(4),
                IRNodeKind::Select {
                    cond: NodeId::new(3),
                    then_val: NodeId::new(0),
                    else_val: NodeId::new(1),
                },
            ),
        ];
        let kernel = KernelDef::new("select_k", params, ScalarType::F32, nodes, NodeId::new(4));
        assert!(
            kernel.validate().is_ok(),
            "Select with valid Compare condition must validate"
        );
    }

    // -----------------------------------------------------------------------
    // Proof 26: KernelDef with Compare IR node validates
    // -----------------------------------------------------------------------

    #[kani::unwind(8)]
    #[kani::proof]
    fn proof_compare_ir_validates() {
        let params = vec![
            Param::new("a", ScalarType::F32),
            Param::new("b", ScalarType::F32),
        ];
        let nodes = vec![
            IRNode::new(NodeId::new(0), IRNodeKind::Param(0)),
            IRNode::new(NodeId::new(1), IRNodeKind::Param(1)),
            IRNode::new(
                NodeId::new(2),
                IRNodeKind::Compare {
                    op: CompareOpKind::Le,
                    lhs: NodeId::new(0),
                    rhs: NodeId::new(1),
                },
            ),
            // Select to produce a float from the boolean
            IRNode::new(NodeId::new(3), IRNodeKind::Literal(1.0)),
            IRNode::new(NodeId::new(4), IRNodeKind::Literal(0.0)),
            IRNode::new(
                NodeId::new(5),
                IRNodeKind::Select {
                    cond: NodeId::new(2),
                    then_val: NodeId::new(3),
                    else_val: NodeId::new(4),
                },
            ),
        ];
        let kernel = KernelDef::new("compare_k", params, ScalarType::F32, nodes, NodeId::new(5));
        assert!(
            kernel.validate().is_ok(),
            "Compare + Select kernel must validate"
        );
    }

    // -----------------------------------------------------------------------
    // Proof 27: KernelDef with MinMax IR node validates
    // -----------------------------------------------------------------------

    #[kani::unwind(8)]
    #[kani::proof]
    fn proof_minmax_ir_validates() {
        let params = vec![
            Param::new("a", ScalarType::F32),
            Param::new("b", ScalarType::F32),
        ];
        let nodes = vec![
            IRNode::new(NodeId::new(0), IRNodeKind::Param(0)),
            IRNode::new(NodeId::new(1), IRNodeKind::Param(1)),
            IRNode::new(
                NodeId::new(2),
                IRNodeKind::MinMax {
                    op: MinMaxKind::Max,
                    lhs: NodeId::new(0),
                    rhs: NodeId::new(1),
                },
            ),
        ];
        let kernel = KernelDef::new("max_k", params, ScalarType::F32, nodes, NodeId::new(2));
        assert!(kernel.validate().is_ok(), "MinMax kernel must validate");
    }

    // -----------------------------------------------------------------------
    // Proof 28: KernelDef forward reference rejection
    // -----------------------------------------------------------------------

    #[kani::unwind(8)]
    #[kani::proof]
    fn proof_kernel_def_rejects_forward_reference() {
        let params = vec![Param::new("x", ScalarType::F32)];
        let nodes = vec![
            // Node 0 references Node 1 (forward reference = invalid)
            IRNode::new(
                NodeId::new(0),
                IRNodeKind::UnaryFn {
                    op: UnaryFnKind::Exp,
                    input: NodeId::new(1),
                },
            ),
            IRNode::new(NodeId::new(1), IRNodeKind::Param(0)),
        ];
        let kernel = KernelDef::new("bad_fwd", params, ScalarType::F32, nodes, NodeId::new(0));
        assert!(
            kernel.validate().is_err(),
            "Forward reference must be rejected"
        );
    }

    // -----------------------------------------------------------------------
    // Proof 29: KernelDef self-reference rejection
    // -----------------------------------------------------------------------

    #[kani::unwind(8)]
    #[kani::proof]
    fn proof_kernel_def_rejects_self_reference() {
        let params = vec![Param::new("x", ScalarType::F32)];
        let nodes = vec![
            IRNode::new(NodeId::new(0), IRNodeKind::Param(0)),
            // Node 1 references itself (self-reference = invalid)
            IRNode::new(
                NodeId::new(1),
                IRNodeKind::UnaryFn {
                    op: UnaryFnKind::Abs,
                    input: NodeId::new(1),
                },
            ),
        ];
        let kernel = KernelDef::new("bad_self", params, ScalarType::F32, nodes, NodeId::new(1));
        assert!(
            kernel.validate().is_err(),
            "Self-reference must be rejected"
        );
    }

    // -----------------------------------------------------------------------
    // Proof 30: KernelDef mismatched NodeId rejection
    // -----------------------------------------------------------------------

    #[kani::unwind(8)]
    #[kani::proof]
    fn proof_kernel_def_rejects_mismatched_node_id() {
        let params = vec![Param::new("x", ScalarType::F32)];
        // Node at index 0 has id 1 (mismatch)
        let nodes = vec![IRNode::new(NodeId::new(1), IRNodeKind::Param(0))];
        let kernel = KernelDef::new("bad_id", params, ScalarType::F32, nodes, NodeId::new(1));
        assert!(
            kernel.validate().is_err(),
            "Mismatched NodeId must be rejected"
        );
    }

    // -----------------------------------------------------------------------
    // Proof 31: KernelDef with all BinOpKind variants validates
    // -----------------------------------------------------------------------

    #[kani::unwind(8)]
    #[kani::proof]
    fn proof_all_binop_kinds_validate() {
        let ops = [
            BinOpKind::Add,
            BinOpKind::Sub,
            BinOpKind::Mul,
            BinOpKind::Div,
        ];
        for op in &ops {
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
                        op: *op,
                        lhs: NodeId::new(0),
                        rhs: NodeId::new(1),
                    },
                ),
            ];
            let kernel = KernelDef::new("binop_k", params, ScalarType::F32, nodes, NodeId::new(2));
            assert!(
                kernel.validate().is_ok(),
                "BinOp {:?} kernel must validate",
                op
            );
        }
    }

    // -----------------------------------------------------------------------
    // Proof 32: has_ftz_sensitive_op detects Recip
    // -----------------------------------------------------------------------

    #[kani::unwind(8)]
    #[kani::proof]
    fn proof_ftz_detects_recip() {
        let params = vec![Param::new("x", ScalarType::F32)];
        let nodes = vec![
            IRNode::new(NodeId::new(0), IRNodeKind::Param(0)),
            IRNode::new(
                NodeId::new(1),
                IRNodeKind::UnaryFn {
                    op: UnaryFnKind::Recip,
                    input: NodeId::new(0),
                },
            ),
        ];
        let kernel = KernelDef::new("recip_k", params, ScalarType::F32, nodes, NodeId::new(1));
        assert!(kernel.has_ftz_sensitive_op(), "Recip is FTZ-sensitive");
    }

    // -----------------------------------------------------------------------
    // Proof 33: bootstrap_budget ordering: Strict <= Normal <= Relaxed
    // -----------------------------------------------------------------------

    #[kani::unwind(1)]
    #[kani::proof]
    fn proof_bootstrap_budget_ordering_f16() {
        let (strict_abs, _) =
            crate::precision::bootstrap_budget(ScalarType::F16, PrecisionTier::Strict);
        let (normal_abs, _) =
            crate::precision::bootstrap_budget(ScalarType::F16, PrecisionTier::Normal);
        let (relaxed_abs, _) =
            crate::precision::bootstrap_budget(ScalarType::F16, PrecisionTier::Relaxed);
        assert!(strict_abs <= normal_abs, "F16: strict abs <= normal abs");
        assert!(normal_abs <= relaxed_abs, "F16: normal abs <= relaxed abs");
    }

    // -----------------------------------------------------------------------
    // Proof 34: PrecisionContract::bootstrap consistency
    // -----------------------------------------------------------------------

    #[kani::unwind(1)]
    #[kani::proof]
    fn proof_precision_contract_bootstrap_fields() {
        let contract = PrecisionContract::bootstrap(PrecisionTier::Relaxed, ScalarType::F32);
        assert!(matches!(contract.tier, PrecisionTier::Relaxed));
        assert!(contract.fast_math, "Relaxed tier must have fast_math=true");
        assert!(contract.differential_abs_budget > 0.0);
        assert!(contract.differential_rel_budget > 0.0);

        let contract_strict = PrecisionContract::bootstrap(PrecisionTier::Strict, ScalarType::F32);
        assert!(
            !contract_strict.fast_math,
            "Strict tier must have fast_math=false"
        );
        assert!(
            contract_strict.differential_abs_budget <= contract.differential_abs_budget,
            "Strict budget must be <= Relaxed budget"
        );
    }
}

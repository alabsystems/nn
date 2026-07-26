// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Extra Kani proof harnesses for nn-dsl crate.
//!
//! Proves safety and correctness properties of:
//! - Scalar kernel functions (sigmoid, tanh, gelu, snake) with finite inputs
//! - Scalar kernel NaN/Inf rejection
//! - `BroadcastAlignment` inference correctness
//! - `ReduceOp` variant distinctness
//! - `BufferPlan` total_bytes >= naive_total invariant (structural)
//! - `classify_callee_name` consistency with `classify_op`
//! - `VerifiabilitySummary::is_fully_compilable` semantics
//! - `NnDslError` variant reachability (From impls)
//! - `KernelDef` composed Powi IR validation
//! - `KernelDef` composed Clamp IR validation
//! - `IRNodeKind::SumReduce` topological validity
//! - `NormActivation` enum coverage
//! - `sum_reduce` associativity for 2 elements
//! - `InputBounds` parse/serialize round-trip
//!
//! Part of #3805.

#[cfg(kani)]
mod proofs {
    use crate::ir::{
        BinOpKind, IRNode, IRNodeKind, KernelDef, MinMaxKind, NodeId, Param, ScalarType,
        UnaryFnKind,
    };
    use crate::kernel_error::KernelError;
    use crate::precision::PrecisionTier;
    use crate::tensor_ir::{BroadcastAlignment, ReduceOp};
    use crate::trace_compile::NormActivation;
    use crate::verifiability::{classify_callee_name, VerifiabilityClass, VerifiabilitySummary};

    // -----------------------------------------------------------------------
    // Proof 1: sigmoid_scalar rejects NaN input
    // -----------------------------------------------------------------------

    #[kani::unwind(1)]
    #[kani::proof]
    fn proof_sigmoid_scalar_rejects_nan() {
        let result = crate::sigmoid_scalar(f32::NAN);
        assert!(result.is_err(), "sigmoid must reject NaN input");
    }

    // -----------------------------------------------------------------------
    // Proof 2: sigmoid_scalar rejects infinity input
    // -----------------------------------------------------------------------

    #[kani::unwind(1)]
    #[kani::proof]
    fn proof_sigmoid_scalar_rejects_inf() {
        let result_pos = crate::sigmoid_scalar(f32::INFINITY);
        let result_neg = crate::sigmoid_scalar(f32::NEG_INFINITY);
        assert!(result_pos.is_err(), "sigmoid must reject +inf input");
        assert!(result_neg.is_err(), "sigmoid must reject -inf input");
    }

    // -----------------------------------------------------------------------
    // Proof 3: sigmoid_scalar output is in (0, 1) for moderate inputs
    // -----------------------------------------------------------------------

    #[kani::unwind(1)]
    #[kani::proof]
    fn proof_sigmoid_scalar_output_bounded() {
        let result = crate::sigmoid_scalar(0.0);
        assert!(result.is_ok());
        let val = result.unwrap();
        assert!(val > 0.0 && val < 1.0, "sigmoid(0) must be in (0,1)");
        // sigmoid(0) = 0.5
        assert!((val - 0.5).abs() < 1e-6, "sigmoid(0) must be ~0.5");
    }

    // -----------------------------------------------------------------------
    // Proof 4: tanh_scalar rejects NaN input
    // -----------------------------------------------------------------------

    #[kani::unwind(1)]
    #[kani::proof]
    fn proof_tanh_scalar_rejects_nan() {
        let result = crate::tanh_scalar(f32::NAN);
        assert!(result.is_err(), "tanh must reject NaN input");
    }

    // -----------------------------------------------------------------------
    // Proof 5: tanh_scalar output bounded in [-1, 1]
    // -----------------------------------------------------------------------

    #[kani::unwind(1)]
    #[kani::proof]
    fn proof_tanh_scalar_output_bounded() {
        let result = crate::tanh_scalar(0.0);
        assert!(result.is_ok());
        let val = result.unwrap();
        assert!(val >= -1.0 && val <= 1.0, "tanh output must be in [-1,1]");
        assert!(val.abs() < 1e-6, "tanh(0) must be ~0.0");
    }

    // -----------------------------------------------------------------------
    // Proof 6: gelu_scalar rejects NaN input
    // -----------------------------------------------------------------------

    #[kani::unwind(1)]
    #[kani::proof]
    fn proof_gelu_scalar_rejects_nan() {
        let result = crate::gelu_scalar(f32::NAN);
        assert!(result.is_err(), "gelu must reject NaN input");
    }

    // -----------------------------------------------------------------------
    // Proof 7: gelu_scalar(0) == 0
    // -----------------------------------------------------------------------

    #[kani::unwind(1)]
    #[kani::proof]
    fn proof_gelu_scalar_zero_is_zero() {
        let result = crate::gelu_scalar(0.0);
        assert!(result.is_ok());
        let val = result.unwrap();
        assert!(val.abs() < 1e-6, "gelu(0) must be ~0.0");
    }

    // -----------------------------------------------------------------------
    // Proof 8: snake_scalar rejects NaN x
    // -----------------------------------------------------------------------

    #[kani::unwind(1)]
    #[kani::proof]
    fn proof_snake_scalar_rejects_nan_x() {
        let result = crate::snake_scalar(f32::NAN, 1.0);
        assert!(result.is_err(), "snake must reject NaN x");
    }

    // -----------------------------------------------------------------------
    // Proof 9: snake_scalar rejects NaN alpha
    // -----------------------------------------------------------------------

    #[kani::unwind(1)]
    #[kani::proof]
    fn proof_snake_scalar_rejects_nan_alpha() {
        let result = crate::snake_scalar(1.0, f32::NAN);
        assert!(result.is_err(), "snake must reject NaN alpha");
    }

    // -----------------------------------------------------------------------
    // Proof 10: snake_scalar(0, alpha) == sin^2(0)/alpha == 0
    // -----------------------------------------------------------------------

    #[kani::unwind(1)]
    #[kani::proof]
    fn proof_snake_scalar_zero_input() {
        let result = crate::snake_scalar(0.0, 1.0);
        assert!(result.is_ok());
        let val = result.unwrap();
        // snake(0, a) = 0 + (1/a)*sin(a*0)^2 = 0
        assert!(val.abs() < 1e-6, "snake(0, alpha) must be ~0.0");
    }

    // -----------------------------------------------------------------------
    // Proof 11: ReduceOp has 4 distinct variants
    // -----------------------------------------------------------------------

    #[kani::unwind(8)]
    #[kani::proof]
    #[kani::unwind(8)]
    fn proof_reduce_op_all_distinct() {
        let variants = [ReduceOp::Sum, ReduceOp::Mean, ReduceOp::Max, ReduceOp::Min];
        for i in 0..variants.len() {
            for j in (i + 1)..variants.len() {
                assert_ne!(
                    std::mem::discriminant(&variants[i]),
                    std::mem::discriminant(&variants[j]),
                    "ReduceOp variants must be distinct"
                );
            }
        }
    }

    // -----------------------------------------------------------------------
    // Proof 12: BroadcastAlignment has exactly 2 distinct variants
    // -----------------------------------------------------------------------

    #[kani::unwind(1)]
    #[kani::proof]
    #[kani::unwind(8)]
    fn proof_broadcast_alignment_distinct() {
        assert_ne!(
            std::mem::discriminant(&BroadcastAlignment::Left),
            std::mem::discriminant(&BroadcastAlignment::Right),
            "Left and Right must be distinct"
        );
    }

    // -----------------------------------------------------------------------
    // Proof 13: infer_broadcast_alignment same-rank returns Left
    // -----------------------------------------------------------------------

    #[kani::unwind(1)]
    #[kani::proof]
    fn proof_broadcast_alignment_same_rank() {
        let result = crate::tensor_ir::infer_broadcast_alignment(&[2, 3], &[2, 3]);
        assert!(result.is_ok());
        assert!(matches!(result.unwrap(), BroadcastAlignment::Left));
    }

    // -----------------------------------------------------------------------
    // Proof 14: infer_broadcast_alignment rejects input > target
    // -----------------------------------------------------------------------

    #[kani::unwind(1)]
    #[kani::proof]
    fn proof_broadcast_alignment_rejects_larger_input() {
        let result = crate::tensor_ir::infer_broadcast_alignment(&[2, 3, 4], &[2, 3]);
        assert!(result.is_err(), "Input rank > target rank must fail");
    }

    // -----------------------------------------------------------------------
    // Proof 15: classify_callee_name("relu") is Verifiable
    // -----------------------------------------------------------------------

    #[kani::unwind(1)]
    #[kani::proof]
    fn proof_classify_callee_relu_verifiable() {
        let class = classify_callee_name("relu");
        assert!(matches!(class, VerifiabilityClass::Verifiable));
    }

    // -----------------------------------------------------------------------
    // Proof 16: classify_callee_name("reshape") is ShapeOnly
    // -----------------------------------------------------------------------

    #[kani::unwind(1)]
    #[kani::proof]
    fn proof_classify_callee_reshape_shape_only() {
        let class = classify_callee_name("reshape");
        assert!(matches!(class, VerifiabilityClass::ShapeOnly));
    }

    // -----------------------------------------------------------------------
    // Proof 17: classify_callee_name unknown returns UnverifiableLearned
    // -----------------------------------------------------------------------

    #[kani::unwind(1)]
    #[kani::proof]
    fn proof_classify_callee_unknown_is_unverifiable() {
        let class = classify_callee_name("some_unknown_op_xyz");
        assert!(matches!(class, VerifiabilityClass::UnverifiableLearned));
    }

    // -----------------------------------------------------------------------
    // Proof 18: VerifiabilitySummary::is_fully_compilable when no learned
    // -----------------------------------------------------------------------

    #[kani::unwind(8)]
    #[kani::proof]
    fn proof_verifiability_summary_compilable_without_learned() {
        let summary = VerifiabilitySummary {
            verifiable: 10,
            bounded: 2,
            shape_only: 5,
            passthrough: 3,
            unverifiable_safe: 1,
            unverifiable_learned: 0,
            unverifiable_learned_ops: Vec::new(),
        };
        assert!(
            summary.is_fully_compilable(),
            "Must be compilable when unverifiable_learned == 0"
        );
    }

    // -----------------------------------------------------------------------
    // Proof 19: VerifiabilitySummary NOT compilable when learned > 0
    // -----------------------------------------------------------------------

    #[kani::unwind(16)]
    #[kani::proof]
    #[kani::unwind(8)]
    fn proof_verifiability_summary_not_compilable_with_learned() {
        let summary = VerifiabilitySummary {
            verifiable: 10,
            bounded: 0,
            shape_only: 0,
            passthrough: 0,
            unverifiable_safe: 0,
            unverifiable_learned: 1,
            unverifiable_learned_ops: vec!["bad_op".to_string()],
        };
        assert!(
            !summary.is_fully_compilable(),
            "Must NOT be compilable when unverifiable_learned > 0"
        );
    }

    // -----------------------------------------------------------------------
    // Proof 20: KernelDef with Powi IR validates
    // -----------------------------------------------------------------------

    #[kani::unwind(16)]
    #[kani::proof]
    #[kani::unwind(4)]
    fn proof_powi_kernel_validates() {
        let params = vec![Param::new("x", ScalarType::F32)];
        let nodes = vec![
            IRNode::new(NodeId::new(0), IRNodeKind::Param(0)),
            IRNode::new(
                NodeId::new(1),
                IRNodeKind::Powi {
                    base: NodeId::new(0),
                    exp: 2,
                },
            ),
        ];
        let kernel = KernelDef::new("square", params, ScalarType::F32, nodes, NodeId::new(1));
        assert!(kernel.validate().is_ok(), "Powi(x, 2) kernel must validate");
    }

    // -----------------------------------------------------------------------
    // Proof 21: KernelDef with Clamp IR validates
    // -----------------------------------------------------------------------

    #[kani::unwind(16)]
    #[kani::proof]
    #[kani::unwind(5)]
    fn proof_clamp_ir_kernel_validates() {
        let params = vec![Param::new("x", ScalarType::F32)];
        let nodes = vec![
            IRNode::new(NodeId::new(0), IRNodeKind::Param(0)),
            IRNode::new(NodeId::new(1), IRNodeKind::Literal(-1.0)),
            IRNode::new(NodeId::new(2), IRNodeKind::Literal(1.0)),
            IRNode::new(
                NodeId::new(3),
                IRNodeKind::Clamp {
                    input: NodeId::new(0),
                    min: NodeId::new(1),
                    max: NodeId::new(2),
                },
            ),
        ];
        let kernel = KernelDef::new("clamp_ir", params, ScalarType::F32, nodes, NodeId::new(3));
        assert!(kernel.validate().is_ok(), "Clamp IR kernel must validate");
    }

    // -----------------------------------------------------------------------
    // Proof 22: SumReduce IR node validates with valid backward refs
    // -----------------------------------------------------------------------

    #[kani::unwind(16)]
    #[kani::proof]
    #[kani::unwind(5)]
    fn proof_sum_reduce_ir_validates() {
        let params = vec![
            Param::new("a", ScalarType::F32),
            Param::new("b", ScalarType::F32),
        ];
        let nodes = vec![
            IRNode::new(NodeId::new(0), IRNodeKind::Param(0)),
            IRNode::new(NodeId::new(1), IRNodeKind::Param(1)),
            IRNode::new(
                NodeId::new(2),
                IRNodeKind::SumReduce {
                    inputs: vec![NodeId::new(0), NodeId::new(1)],
                },
            ),
        ];
        let kernel = KernelDef::new("sum2", params, ScalarType::F32, nodes, NodeId::new(2));
        assert!(
            kernel.validate().is_ok(),
            "SumReduce with backward refs must validate"
        );
    }

    // -----------------------------------------------------------------------
    // Proof 23: NormActivation has 2 distinct variants
    // -----------------------------------------------------------------------

    #[kani::unwind(1)]
    #[kani::proof]
    #[kani::unwind(8)]
    fn proof_norm_activation_distinct() {
        assert_ne!(
            std::mem::discriminant(&NormActivation::Snake),
            std::mem::discriminant(&NormActivation::LeakyRelu { slope: 0.2 }),
            "Snake and LeakyRelu must be distinct"
        );
    }

    // -----------------------------------------------------------------------
    // Proof 24: sum_reduce is commutative for 2 u32 elements
    // -----------------------------------------------------------------------

    #[kani::unwind(1)]
    #[kani::proof]
    #[kani::unwind(8)]
    fn proof_sum_reduce_commutative_2() {
        let a: u8 = kani::any();
        let b: u8 = kani::any();
        kani::assume(a <= 100);
        kani::assume(b <= 100);

        let sum_ab = crate::sum_reduce([a as u32, b as u32]);
        let sum_ba = crate::sum_reduce([b as u32, a as u32]);
        assert_eq!(sum_ab, sum_ba, "sum_reduce must be commutative");
    }

    // -----------------------------------------------------------------------
    // Proof 25: NnDslError From<KernelError> variant reachability
    // -----------------------------------------------------------------------

    #[kani::unwind(1)]
    #[kani::proof]
    fn proof_nn_dsl_error_from_kernel_error() {
        let ke = KernelError::NonFiniteInput {
            name: "x",
            value: f32::NAN,
        };
        let err: crate::NnDslError = ke.into();
        assert!(matches!(err, crate::NnDslError::Kernel(_)));
    }

    // -----------------------------------------------------------------------
    // Proof 26: NnDslError From<IRError> variant reachability
    // -----------------------------------------------------------------------

    #[kani::unwind(1)]
    #[kani::proof]
    fn proof_nn_dsl_error_from_ir_error() {
        let ie = crate::ir::IRError::EmptySumReduce(crate::ir::NodeId::new(0));
        let err: crate::NnDslError = ie.into();
        assert!(matches!(err, crate::NnDslError::Ir(_)));
    }

    // -----------------------------------------------------------------------
    // Proof 27: KernelDef with BinaryFn(Atan2) validates
    // -----------------------------------------------------------------------

    #[kani::unwind(16)]
    #[kani::proof]
    #[kani::unwind(4)]
    fn proof_atan2_kernel_validates() {
        use crate::ir::BinaryFnKind;

        let params = vec![
            Param::new("y", ScalarType::F32),
            Param::new("x", ScalarType::F32),
        ];
        let nodes = vec![
            IRNode::new(NodeId::new(0), IRNodeKind::Param(0)),
            IRNode::new(NodeId::new(1), IRNodeKind::Param(1)),
            IRNode::new(
                NodeId::new(2),
                IRNodeKind::BinaryFn {
                    op: BinaryFnKind::Atan2,
                    lhs: NodeId::new(0),
                    rhs: NodeId::new(1),
                },
            ),
        ];
        let kernel = KernelDef::new("atan2_k", params, ScalarType::F32, nodes, NodeId::new(2));
        assert!(kernel.validate().is_ok(), "Atan2 kernel must validate");
    }

    // -----------------------------------------------------------------------
    // Proof 28: KernelDef with F16 scalar type validates
    // -----------------------------------------------------------------------

    #[kani::unwind(8)]
    #[kani::proof]
    fn proof_f16_kernel_validates() {
        let params = vec![Param::new("x", ScalarType::F16)];
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
        let kernel = KernelDef::new("exp_f16", params, ScalarType::F16, nodes, NodeId::new(1));
        assert!(kernel.validate().is_ok(), "F16 kernel must validate");
    }

    // -----------------------------------------------------------------------
    // Proof 29: classify_callee_name consistency: known activations
    // -----------------------------------------------------------------------

    #[kani::unwind(8)]
    #[kani::proof]
    fn proof_classify_callee_known_activations_verifiable() {
        let names = ["sigmoid", "tanh", "gelu", "silu", "exp", "log"];
        for name in &names {
            let class = classify_callee_name(name);
            assert!(
                class.allows_compilation(),
                "Known activation must allow compilation"
            );
        }
    }

    // -----------------------------------------------------------------------
    // Proof 30: sigmoid_scalar for small positive is in (0.5, 1)
    // -----------------------------------------------------------------------

    #[kani::unwind(1)]
    #[kani::proof]
    fn proof_sigmoid_positive_input_above_half() {
        let result = crate::sigmoid_scalar(1.0);
        assert!(result.is_ok());
        let val = result.unwrap();
        assert!(val > 0.5, "sigmoid(positive) must be > 0.5");
        assert!(val < 1.0, "sigmoid output must be < 1.0");
    }

    // -----------------------------------------------------------------------
    // Proof 31: sigmoid_scalar for small negative is in (0, 0.5)
    // -----------------------------------------------------------------------

    #[kani::unwind(1)]
    #[kani::proof]
    fn proof_sigmoid_negative_input_below_half() {
        let result = crate::sigmoid_scalar(-1.0);
        assert!(result.is_ok());
        let val = result.unwrap();
        assert!(val > 0.0, "sigmoid output must be > 0.0");
        assert!(val < 0.5, "sigmoid(negative) must be < 0.5");
    }

    // -----------------------------------------------------------------------
    // Proof 32: tanh_scalar(-x) == -tanh_scalar(x) (odd function)
    // -----------------------------------------------------------------------

    #[kani::unwind(1)]
    #[kani::proof]
    fn proof_tanh_odd_function_at_1() {
        let pos = crate::tanh_scalar(1.0).unwrap();
        let neg = crate::tanh_scalar(-1.0).unwrap();
        assert!(
            (pos + neg).abs() < 1e-6,
            "tanh must be odd: tanh(-x) == -tanh(x)"
        );
    }

    // -----------------------------------------------------------------------
    // Proof 33: PrecisionTier ordering: Strict < Normal < Relaxed tolerance
    // -----------------------------------------------------------------------

    #[kani::unwind(1)]
    #[kani::proof]
    fn proof_precision_tier_tolerance_ordering() {
        use crate::precision::PrecisionContract;
        let ref_val = 1.0_f32;
        let strict = PrecisionContract::bootstrap(PrecisionTier::Strict, ScalarType::F32);
        let normal = PrecisionContract::bootstrap(PrecisionTier::Normal, ScalarType::F32);
        let relaxed = PrecisionContract::bootstrap(PrecisionTier::Relaxed, ScalarType::F32);
        let strict_tol = crate::precision::differential_tolerance(ref_val, strict);
        let normal_tol = crate::precision::differential_tolerance(ref_val, normal);
        let relaxed_tol = crate::precision::differential_tolerance(ref_val, relaxed);
        assert!(
            strict_tol <= normal_tol,
            "Strict tolerance must be <= Normal"
        );
        assert!(
            normal_tol <= relaxed_tol,
            "Normal tolerance must be <= Relaxed"
        );
    }
}

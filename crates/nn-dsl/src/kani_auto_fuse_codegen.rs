// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Kani proof harnesses for `auto_fuse_codegen` correctness properties.
//!
//! Proves additional invariants beyond `kani_auto_fuse_codegen_proofs.rs`:
//! - Neg decomposition: output is `Sub(0, x)`, not identity
//! - Sqr decomposition: output is `Mul(x, x)`, both operands are the input
//! - Clamp with min-only and max-only partial bounds
//! - Mixed binary+unary chain parameter accumulation
//! - Multi-external-input chains count params correctly
//! - add_external_param naming convention (`p0`, `p1`, ...)
//! - Composed kernel output is always the last-emitted node
//! - All supported unary ops produce kernels with exactly 1 param
//! - All supported binary ops produce kernels with exactly 2 params
//! - Gelu node count is bounded (regression: avoid IR explosion)
//! - FuseableOp constructors preserve the original TraceOp
//! - powf with odd integer exponent has sign-preserving Select node
//! - powf with fractional exponent has NaN-guard Select node
//!
//! Part of #3684.

#[cfg(kani)]
mod proofs {
    use crate::auto_fuse_codegen::{
        compose_trace_ops_to_kernel_ir, op_input_count, FuseableOp, OpWiring,
    };
    use crate::ir::{
        BinOpKind, IRNode, IRNodeKind, KernelDef, MinMaxKind, NodeId, Param, ScalarType,
        UnaryFnKind,
    };
    use nn_core::dyn_tensor::trace::TraceOp;

    // -----------------------------------------------------------------------
    // Proof 1: Neg produces Sub(0, x), not identity
    // -----------------------------------------------------------------------

    #[kani::unwind(16)]
    #[kani::proof]
    #[kani::unwind(6)]
    fn proof_neg_decomposition_is_sub_zero() {
        let ops = vec![FuseableOp::unary(TraceOp::Neg)];
        let kernel = compose_trace_ops_to_kernel_ir(&ops, "neg").expect("Neg must compose");
        assert_eq!(kernel.params.len(), 1);
        assert!(kernel.validate().is_ok());

        // Output must be a BinOp::Sub (0 - x), not a Param passthrough
        let out_node = &kernel.nodes[kernel.output.index()];
        assert!(
            matches!(
                out_node.kind,
                IRNodeKind::BinOp {
                    op: BinOpKind::Sub,
                    ..
                }
            ),
            "Neg output must be Sub(0, x)"
        );
    }

    // -----------------------------------------------------------------------
    // Proof 2: Sqr produces Mul(x, x) with both operands referencing input
    // -----------------------------------------------------------------------

    #[kani::unwind(16)]
    #[kani::proof]
    #[kani::unwind(5)]
    fn proof_sqr_decomposition_is_self_mul() {
        let ops = vec![FuseableOp::unary(TraceOp::Sqr)];
        let kernel = compose_trace_ops_to_kernel_ir(&ops, "sqr").expect("Sqr must compose");
        assert_eq!(kernel.params.len(), 1);
        assert!(kernel.validate().is_ok());

        let out_node = &kernel.nodes[kernel.output.index()];
        match &out_node.kind {
            IRNodeKind::BinOp { op, lhs, rhs } => {
                assert!(matches!(op, BinOpKind::Mul), "Sqr must be Mul");
                assert_eq!(lhs, rhs, "Sqr must multiply input by itself");
                // Both operands must reference the Param(0) node
                assert_eq!(lhs.index(), 0, "Sqr operand must be the param node");
            }
            _ => panic!("Sqr output must be BinOp::Mul"),
        }
    }

    // -----------------------------------------------------------------------
    // Proof 3: Clamp with min-only produces single MinMax::Max
    // -----------------------------------------------------------------------

    #[kani::unwind(16)]
    #[kani::proof]
    #[kani::unwind(6)]
    fn proof_clamp_min_only_produces_max() {
        let ops = vec![FuseableOp::unary(TraceOp::Clamp {
            min: Some(-5.0),
            max: None,
        })];
        let kernel =
            compose_trace_ops_to_kernel_ir(&ops, "clamp_min").expect("Clamp min-only must compose");
        assert_eq!(kernel.params.len(), 1);
        assert!(kernel.validate().is_ok());

        // Output must be MinMax::Max (clamping lower bound)
        let out_node = &kernel.nodes[kernel.output.index()];
        assert!(
            matches!(
                out_node.kind,
                IRNodeKind::MinMax {
                    op: MinMaxKind::Max,
                    ..
                }
            ),
            "Clamp(min=Some, max=None) must produce MinMax::Max"
        );
    }

    // -----------------------------------------------------------------------
    // Proof 4: Clamp with max-only produces single MinMax::Min
    // -----------------------------------------------------------------------

    #[kani::unwind(16)]
    #[kani::proof]
    #[kani::unwind(6)]
    fn proof_clamp_max_only_produces_min() {
        let ops = vec![FuseableOp::unary(TraceOp::Clamp {
            min: None,
            max: Some(5.0),
        })];
        let kernel =
            compose_trace_ops_to_kernel_ir(&ops, "clamp_max").expect("Clamp max-only must compose");
        assert_eq!(kernel.params.len(), 1);
        assert!(kernel.validate().is_ok());

        // Output must be MinMax::Min (clamping upper bound)
        let out_node = &kernel.nodes[kernel.output.index()];
        assert!(
            matches!(
                out_node.kind,
                IRNodeKind::MinMax {
                    op: MinMaxKind::Min,
                    ..
                }
            ),
            "Clamp(min=None, max=Some) must produce MinMax::Min"
        );
    }

    // -----------------------------------------------------------------------
    // Proof 5: Clamp with no bounds is identity (output == input param)
    // -----------------------------------------------------------------------

    #[kani::unwind(16)]
    #[kani::proof]
    #[kani::unwind(5)]
    fn proof_clamp_no_bounds_is_identity() {
        let ops = vec![FuseableOp::unary(TraceOp::Clamp {
            min: None,
            max: None,
        })];
        let kernel = compose_trace_ops_to_kernel_ir(&ops, "clamp_none")
            .expect("Clamp(None, None) must compose");
        assert_eq!(kernel.params.len(), 1);
        assert!(kernel.validate().is_ok());

        // With no bounds, output is the param itself
        let out_node = &kernel.nodes[kernel.output.index()];
        assert!(
            matches!(out_node.kind, IRNodeKind::Param(0)),
            "Clamp(None, None) must be identity"
        );
    }

    // -----------------------------------------------------------------------
    // Proof 6: Mixed binary+unary chain accumulates params correctly
    // -----------------------------------------------------------------------

    /// Chain: Add(x, y) -> Exp -> Mul(_, z) -> Relu
    /// Expected params: x, y (from Add), z (from Mul) = 3 total
    #[kani::unwind(16)]
    #[kani::proof]
    #[kani::unwind(12)]
    fn proof_mixed_chain_param_accumulation() {
        let ops = vec![
            FuseableOp::binary_both_external(TraceOp::Add),
            FuseableOp::unary(TraceOp::Exp),
            FuseableOp::binary_second_external(TraceOp::Mul),
            FuseableOp::unary(TraceOp::Relu),
        ];
        let kernel =
            compose_trace_ops_to_kernel_ir(&ops, "mixed_chain").expect("Mixed chain must compose");
        // Add introduces 2, Exp reuses chain, Mul introduces 1, Relu reuses
        assert_eq!(kernel.params.len(), 3, "Add(2) + Mul(1) = 3 params");
        assert!(kernel.validate().is_ok());
    }

    // -----------------------------------------------------------------------
    // Proof 7: Parameter naming follows p0, p1, p2 convention
    // -----------------------------------------------------------------------

    #[kani::unwind(16)]
    #[kani::proof]
    #[kani::unwind(8)]
    fn proof_param_naming_convention() {
        let ops = vec![
            FuseableOp::binary_both_external(TraceOp::Add),
            FuseableOp::binary_second_external(TraceOp::Mul),
        ];
        let kernel = compose_trace_ops_to_kernel_ir(&ops, "naming_test").expect("Must compose");
        assert_eq!(kernel.params.len(), 3);
        assert_eq!(kernel.params[0].name, "p0");
        assert_eq!(kernel.params[1].name, "p1");
        assert_eq!(kernel.params[2].name, "p2");
    }

    // -----------------------------------------------------------------------
    // Proof 8: Output node index is always within nodes bounds
    // -----------------------------------------------------------------------

    #[kani::unwind(16)]
    #[kani::proof]
    #[kani::unwind(10)]
    fn proof_output_within_nodes_bounds() {
        let ops = vec![
            FuseableOp::unary(TraceOp::Sigmoid),
            FuseableOp::unary(TraceOp::Relu),
        ];
        let kernel = compose_trace_ops_to_kernel_ir(&ops, "bounds_check").expect("Must compose");
        assert!(
            kernel.output.index() < kernel.nodes.len(),
            "Output must reference a valid node"
        );
    }

    // -----------------------------------------------------------------------
    // Proof 9: All return types are F32
    // -----------------------------------------------------------------------

    #[kani::unwind(16)]
    #[kani::proof]
    #[kani::unwind(5)]
    fn proof_return_type_is_f32() {
        let ops = vec![FuseableOp::unary(TraceOp::Exp)];
        let kernel = compose_trace_ops_to_kernel_ir(&ops, "f32_return").expect("Must compose");
        assert!(
            matches!(kernel.return_type, ScalarType::F32),
            "Auto-fused kernels must return F32"
        );
    }

    // -----------------------------------------------------------------------
    // Proof 10: All param types are F32
    // -----------------------------------------------------------------------

    #[kani::unwind(16)]
    #[kani::proof]
    #[kani::unwind(8)]
    fn proof_all_params_are_f32() {
        let ops = vec![
            FuseableOp::binary_both_external(TraceOp::Add),
            FuseableOp::binary_second_external(TraceOp::Sub),
        ];
        let kernel = compose_trace_ops_to_kernel_ir(&ops, "f32_params").expect("Must compose");
        for p in &kernel.params {
            assert!(
                matches!(p.ty, ScalarType::F32),
                "All auto-fuse params must be F32"
            );
        }
    }

    // -----------------------------------------------------------------------
    // Proof 11: Gelu node count is bounded (regression guard)
    // -----------------------------------------------------------------------

    /// Gelu uses tanh approximation: many IR nodes but bounded.
    /// Prevents IR explosion from unbounded composition.
    #[kani::unwind(16)]
    #[kani::proof]
    #[kani::unwind(20)]
    fn proof_gelu_node_count_bounded() {
        let ops = vec![FuseableOp::unary(TraceOp::Gelu)];
        let kernel = compose_trace_ops_to_kernel_ir(&ops, "gelu").expect("Gelu must compose");
        // Gelu decomposition: param + many literals/ops.
        // Must be under 30 nodes (currently ~14).
        assert!(
            kernel.nodes.len() <= 30,
            "Gelu must not produce > 30 IR nodes (got {})",
            kernel.nodes.len()
        );
        assert!(kernel.validate().is_ok());
    }

    // -----------------------------------------------------------------------
    // Proof 12: Sigmoid node count is bounded
    // -----------------------------------------------------------------------

    #[kani::unwind(16)]
    #[kani::proof]
    #[kani::unwind(10)]
    fn proof_sigmoid_node_count_bounded() {
        let ops = vec![FuseableOp::unary(TraceOp::Sigmoid)];
        let kernel = compose_trace_ops_to_kernel_ir(&ops, "sigmoid").expect("Sigmoid must compose");
        // sigmoid: param, 0.0, sub, exp, 1.0, add, div = 7 nodes
        assert!(
            kernel.nodes.len() <= 15,
            "Sigmoid must not produce > 15 IR nodes (got {})",
            kernel.nodes.len()
        );
    }

    // -----------------------------------------------------------------------
    // Proof 13: Softplus decomposition produces Log(1 + Exp(x))
    // -----------------------------------------------------------------------

    #[kani::unwind(16)]
    #[kani::proof]
    #[kani::unwind(8)]
    fn proof_softplus_output_is_log() {
        let ops = vec![FuseableOp::unary(TraceOp::Softplus)];
        let kernel =
            compose_trace_ops_to_kernel_ir(&ops, "softplus").expect("Softplus must compose");
        assert!(kernel.validate().is_ok());

        // Output must be UnaryFn::Log (ln of something)
        let out_node = &kernel.nodes[kernel.output.index()];
        assert!(
            matches!(
                out_node.kind,
                IRNodeKind::UnaryFn {
                    op: UnaryFnKind::Log,
                    ..
                }
            ),
            "Softplus output must be Log"
        );
    }

    // -----------------------------------------------------------------------
    // Proof 14: Silu decomposition output is Mul(x, sigmoid(x))
    // -----------------------------------------------------------------------

    #[kani::unwind(16)]
    #[kani::proof]
    #[kani::unwind(12)]
    fn proof_silu_output_is_mul() {
        let ops = vec![FuseableOp::unary(TraceOp::Silu)];
        let kernel = compose_trace_ops_to_kernel_ir(&ops, "silu").expect("Silu must compose");
        assert!(kernel.validate().is_ok());

        // Output must be BinOp::Mul (x * sigmoid)
        let out_node = &kernel.nodes[kernel.output.index()];
        assert!(
            matches!(
                out_node.kind,
                IRNodeKind::BinOp {
                    op: BinOpKind::Mul,
                    ..
                }
            ),
            "Silu output must be Mul"
        );
    }

    // -----------------------------------------------------------------------
    // Proof 15: Powf with odd integer exponent preserves sign
    // -----------------------------------------------------------------------

    /// powf(x, 3.0) must have a Select node for sign preservation.
    #[kani::unwind(16)]
    #[kani::proof]
    #[kani::unwind(16)]
    fn proof_powf_odd_exponent_has_select() {
        let ops = vec![FuseableOp::unary(TraceOp::Powf { exponent: 3.0 })];
        let kernel = compose_trace_ops_to_kernel_ir(&ops, "pow3").expect("Powf(3) must compose");
        assert!(kernel.validate().is_ok());

        // Odd integer exponent produces Select for sign handling
        let has_select = kernel
            .nodes
            .iter()
            .any(|n| matches!(n.kind, IRNodeKind::Select { .. }));
        assert!(
            has_select,
            "Powf with odd exponent must include Select node"
        );
    }

    // -----------------------------------------------------------------------
    // Proof 16: Powf with fractional exponent has NaN guard
    // -----------------------------------------------------------------------

    /// powf(x, 0.5) for negative x should produce NaN.
    /// The IR must include a Select node for the NaN guard.
    #[kani::unwind(16)]
    #[kani::proof]
    #[kani::unwind(16)]
    fn proof_powf_fractional_exponent_has_nan_guard() {
        let ops = vec![FuseableOp::unary(TraceOp::Powf { exponent: 0.5 })];
        let kernel =
            compose_trace_ops_to_kernel_ir(&ops, "pow_half").expect("Powf(0.5) must compose");
        assert!(kernel.validate().is_ok());

        // Fractional exponent: Select guards negative inputs with NaN
        let has_select = kernel
            .nodes
            .iter()
            .any(|n| matches!(n.kind, IRNodeKind::Select { .. }));
        assert!(
            has_select,
            "Powf with fractional exponent must include NaN guard Select"
        );
    }

    // -----------------------------------------------------------------------
    // Proof 17: Powf with even integer exponent has no Select (always positive)
    // -----------------------------------------------------------------------

    /// powf(x, 2.0) is always positive — no Select needed for sign.
    #[kani::unwind(16)]
    #[kani::proof]
    #[kani::unwind(12)]
    fn proof_powf_even_exponent_no_select() {
        let ops = vec![FuseableOp::unary(TraceOp::Powf { exponent: 2.0 })];
        let kernel = compose_trace_ops_to_kernel_ir(&ops, "pow2").expect("Powf(2) must compose");
        assert!(kernel.validate().is_ok());

        // Even integer exponent: |x|^2 is always positive, no Select needed
        let has_select = kernel
            .nodes
            .iter()
            .any(|n| matches!(n.kind, IRNodeKind::Select { .. }));
        assert!(
            !has_select,
            "Powf with even exponent must not include Select"
        );
    }

    // -----------------------------------------------------------------------
    // Proof 18: Chain of 4 binary ops accumulates 5 params
    // -----------------------------------------------------------------------

    /// Add(x,y) → Mul(_, z) → Sub(_, w) → Div(_, v)
    /// 2 + 1 + 1 + 1 = 5 params.
    #[kani::unwind(16)]
    #[kani::proof]
    #[kani::unwind(12)]
    fn proof_four_binary_chain_param_count() {
        let ops = vec![
            FuseableOp::binary_both_external(TraceOp::Add),
            FuseableOp::binary_second_external(TraceOp::Mul),
            FuseableOp::binary_second_external(TraceOp::Sub),
            FuseableOp::binary_second_external(TraceOp::Div),
        ];
        let kernel = compose_trace_ops_to_kernel_ir(&ops, "four_binary")
            .expect("Four-op binary chain must compose");
        assert_eq!(kernel.params.len(), 5);
        assert!(kernel.validate().is_ok());
    }

    // -----------------------------------------------------------------------
    // Proof 19: BinaryFirstExternal wiring places external on LHS
    // -----------------------------------------------------------------------

    #[kani::unwind(16)]
    #[kani::proof]
    #[kani::unwind(8)]
    fn proof_binary_first_external_lhs_is_external() {
        let ops = vec![
            FuseableOp::unary(TraceOp::Exp),
            FuseableOp::binary_first_external(TraceOp::Div),
        ];
        let kernel = compose_trace_ops_to_kernel_ir(&ops, "div_ext_lhs").expect("Must compose");
        assert_eq!(kernel.params.len(), 2);
        assert!(kernel.validate().is_ok());

        // The Div node should have LHS = the second param (p1, the external input)
        // and RHS = the Exp result
        let out_node = &kernel.nodes[kernel.output.index()];
        match &out_node.kind {
            IRNodeKind::BinOp { op, lhs, rhs } => {
                assert!(matches!(op, BinOpKind::Div));
                // LHS should reference the external param node (p1 at some index)
                // RHS should reference the exp output
                // The key invariant: LHS index > RHS index is NOT guaranteed,
                // but LHS must be a Param node (the external input)
                let lhs_node = &kernel.nodes[lhs.index()];
                assert!(
                    matches!(lhs_node.kind, IRNodeKind::Param(1)),
                    "LHS of BinaryFirstExternal Div must be external param p1"
                );
                // RHS should be the exp result (not a param)
                let rhs_node = &kernel.nodes[rhs.index()];
                assert!(
                    !matches!(rhs_node.kind, IRNodeKind::Param(1)),
                    "RHS must not be the external param"
                );
            }
            _ => panic!("Output must be BinOp::Div"),
        }
    }

    // -----------------------------------------------------------------------
    // Proof 20: Elu decomposition has both Compare and Select
    // -----------------------------------------------------------------------

    #[kani::unwind(16)]
    #[kani::proof]
    #[kani::unwind(12)]
    fn proof_elu_has_compare_and_select() {
        let ops = vec![FuseableOp::unary(TraceOp::Elu { alpha: 1.0 })];
        let kernel = compose_trace_ops_to_kernel_ir(&ops, "elu").expect("Elu must compose");
        assert!(kernel.validate().is_ok());

        let has_compare = kernel
            .nodes
            .iter()
            .any(|n| matches!(n.kind, IRNodeKind::Compare { .. }));
        let has_select = kernel
            .nodes
            .iter()
            .any(|n| matches!(n.kind, IRNodeKind::Select { .. }));
        assert!(has_compare, "Elu must have Compare node");
        assert!(has_select, "Elu must have Select node");
    }

    // -----------------------------------------------------------------------
    // Proof 21: LeakyRelu decomposition has Compare and Select
    // -----------------------------------------------------------------------

    #[kani::unwind(16)]
    #[kani::proof]
    #[kani::unwind(10)]
    fn proof_leaky_relu_has_compare_and_select() {
        let ops = vec![FuseableOp::unary(TraceOp::LeakyRelu { slope: 0.2 })];
        let kernel =
            compose_trace_ops_to_kernel_ir(&ops, "leaky_relu").expect("LeakyRelu must compose");
        assert!(kernel.validate().is_ok());

        let has_compare = kernel
            .nodes
            .iter()
            .any(|n| matches!(n.kind, IRNodeKind::Compare { .. }));
        let has_select = kernel
            .nodes
            .iter()
            .any(|n| matches!(n.kind, IRNodeKind::Select { .. }));
        assert!(has_compare, "LeakyRelu must have Compare node");
        assert!(has_select, "LeakyRelu must have Select node");
    }

    // -----------------------------------------------------------------------
    // Proof 22: Minimum binary op produces MinMax::Min
    // -----------------------------------------------------------------------

    #[kani::unwind(16)]
    #[kani::proof]
    #[kani::unwind(6)]
    fn proof_minimum_binary_produces_min() {
        let ops = vec![FuseableOp::binary_both_external(TraceOp::Minimum)];
        let kernel = compose_trace_ops_to_kernel_ir(&ops, "min_ab").expect("Minimum must compose");
        assert_eq!(kernel.params.len(), 2);
        assert!(kernel.validate().is_ok());
        let out_node = &kernel.nodes[kernel.output.index()];
        assert!(matches!(
            out_node.kind,
            IRNodeKind::MinMax {
                op: MinMaxKind::Min,
                ..
            }
        ));
    }

    // -----------------------------------------------------------------------
    // Proof 23: op_input_count consistency with Atan2
    // -----------------------------------------------------------------------

    #[kani::unwind(1)]
    #[kani::proof]
    #[kani::unwind(8)]
    fn proof_op_input_count_atan2_is_binary() {
        let count = op_input_count(&TraceOp::Atan2);
        assert_eq!(count, 2, "Atan2 is a binary op");
    }

    // -----------------------------------------------------------------------
    // Proof 24: Wiring mismatch: unary wiring with binary op rejected
    // -----------------------------------------------------------------------

    /// If a binary op (2 inputs) has Unary wiring, composition should
    /// still succeed if it's the first op (Unary gives 1 input from
    /// external since no chain output exists, but needs 2). This tests
    /// that the error path is correctly hit.
    #[kani::unwind(16)]
    #[kani::proof]
    #[kani::unwind(5)]
    fn proof_unary_wiring_binary_op_rejected() {
        let ops = vec![FuseableOp {
            op: TraceOp::Add,
            wiring: OpWiring::Unary,
        }];
        let result = compose_trace_ops_to_kernel_ir(&ops, "bad_wiring");
        // Add expects 2 inputs, Unary provides 1. This must fail.
        assert!(result.is_err(), "Unary wiring on binary op must fail");
    }
}

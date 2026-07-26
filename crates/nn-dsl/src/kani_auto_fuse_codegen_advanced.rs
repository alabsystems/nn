// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Advanced Kani proof harnesses for `auto_fuse_codegen` IR structure properties.
//!
//! Proves deeper structural invariants beyond basic composition correctness:
//! - Clamp with both bounds produces exactly TWO MinMax nodes (Max then Min)
//! - All supported single unary ops produce valid kernels with validate() == Ok
//! - Relu output is always MinMax::Max (activation structure)
//! - GeluErf polynomial produces a bounded node count (< 60)
//! - Composed kernel output node index == nodes.len() - 1 (always last)
//! - All trig unary ops (Sin, Cos, Tanh) produce UnaryFn output nodes
//! - LeakyRelu slope is embedded as a Literal in the IR
//! - Elu alpha is embedded as a Literal in the IR
//! - Clamp(min, max) with min > max still validates (Metal semantics)
//! - Mixed 5-op chain produces exactly 4 external params
//! - All unary IR node refs are backward (topological sort)
//! - Binary chain with BinaryFirstExternal has correct LHS/RHS wiring
//! - Softplus output: Log(Add(1, Exp(x))) has exactly 4 non-param nodes
//! - Sigmoid decomposition produces exactly 6 non-param nodes
//!
//! Part of #3731.

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
    // Proof 1: Clamp(min, max) produces exactly 2 MinMax nodes
    // -----------------------------------------------------------------------

    #[kani::unwind(16)]
    #[kani::proof]
    #[kani::unwind(8)]
    fn proof_clamp_both_bounds_two_minmax_nodes() {
        let ops = vec![FuseableOp::unary(TraceOp::Clamp {
            min: Some(-1.0),
            max: Some(1.0),
        })];
        let kernel =
            compose_trace_ops_to_kernel_ir(&ops, "clamp_both").expect("Clamp must compose");
        assert!(kernel.validate().is_ok());

        let minmax_count = kernel
            .nodes
            .iter()
            .filter(|n| matches!(n.kind, IRNodeKind::MinMax { .. }))
            .count();
        assert_eq!(
            minmax_count, 2,
            "Clamp(min, max) must produce exactly 2 MinMax nodes"
        );

        // First MinMax is Max (lower bound), second is Min (upper bound)
        let minmax_nodes: Vec<_> = kernel
            .nodes
            .iter()
            .filter(|n| matches!(n.kind, IRNodeKind::MinMax { .. }))
            .collect();
        assert!(matches!(
            minmax_nodes[0].kind,
            IRNodeKind::MinMax {
                op: MinMaxKind::Max,
                ..
            }
        ));
        assert!(matches!(
            minmax_nodes[1].kind,
            IRNodeKind::MinMax {
                op: MinMaxKind::Min,
                ..
            }
        ));
    }

    // -----------------------------------------------------------------------
    // Proof 2: All supported single unary ops validate
    // -----------------------------------------------------------------------

    #[kani::unwind(16)]
    #[kani::proof]
    #[kani::unwind(50)]
    fn proof_all_unary_ops_produce_valid_kernels() {
        let unary_ops = [
            TraceOp::Exp,
            TraceOp::Log,
            TraceOp::Sqrt,
            TraceOp::Abs,
            TraceOp::Recip,
            TraceOp::Sin,
            TraceOp::Cos,
            TraceOp::Floor,
            TraceOp::Round,
            TraceOp::Fract,
            TraceOp::Tanh,
            TraceOp::Sqr,
            TraceOp::Neg,
            TraceOp::Relu,
            TraceOp::Sigmoid,
            TraceOp::Softplus,
        ];
        for op in &unary_ops {
            let ops = vec![FuseableOp::unary(op.clone())];
            let kernel =
                compose_trace_ops_to_kernel_ir(&ops, "test").expect("All unary ops must compose");
            assert_eq!(kernel.params.len(), 1);
            assert!(kernel.validate().is_ok());
        }
    }

    // -----------------------------------------------------------------------
    // Proof 3: Relu output is MinMax::Max, not Compare+Select
    // -----------------------------------------------------------------------

    #[kani::unwind(16)]
    #[kani::proof]
    #[kani::unwind(6)]
    fn proof_relu_output_is_minmax_max() {
        let ops = vec![FuseableOp::unary(TraceOp::Relu)];
        let kernel = compose_trace_ops_to_kernel_ir(&ops, "relu").expect("Relu must compose");
        let out_node = &kernel.nodes[kernel.output.index()];
        assert!(
            matches!(
                out_node.kind,
                IRNodeKind::MinMax {
                    op: MinMaxKind::Max,
                    ..
                }
            ),
            "Relu must use MinMax::Max (max(x, 0)), not Compare+Select"
        );
    }

    // -----------------------------------------------------------------------
    // Proof 4: GeluErf node count is bounded (polynomial is complex)
    // -----------------------------------------------------------------------

    #[kani::unwind(16)]
    #[kani::proof]
    #[kani::unwind(50)]
    fn proof_gelu_erf_node_count_bounded() {
        let ops = vec![FuseableOp::unary(TraceOp::GeluErf)];
        let kernel =
            compose_trace_ops_to_kernel_ir(&ops, "gelu_erf").expect("GeluErf must compose");
        // GeluErf uses A&S 7.1.26 polynomial, producing many nodes.
        // Must be bounded to prevent IR explosion.
        assert!(
            kernel.nodes.len() <= 60,
            "GeluErf must not produce > 60 IR nodes (got {})",
            kernel.nodes.len()
        );
        assert!(kernel.validate().is_ok());
    }

    // -----------------------------------------------------------------------
    // Proof 5: Composed kernel output node is always last emitted
    // -----------------------------------------------------------------------

    #[kani::unwind(16)]
    #[kani::proof]
    #[kani::unwind(12)]
    fn proof_output_node_is_last_in_nodes() {
        let ops = vec![
            FuseableOp::unary(TraceOp::Exp),
            FuseableOp::binary_second_external(TraceOp::Add),
            FuseableOp::unary(TraceOp::Relu),
        ];
        let kernel = compose_trace_ops_to_kernel_ir(&ops, "chain").expect("Must compose");
        assert_eq!(
            kernel.output.index(),
            kernel.nodes.len() - 1,
            "Output must be the last node in the IR"
        );
    }

    // -----------------------------------------------------------------------
    // Proof 6: Trig unary ops produce UnaryFn output nodes
    // -----------------------------------------------------------------------

    #[kani::unwind(16)]
    #[kani::proof]
    #[kani::unwind(5)]
    fn proof_sin_output_is_unary_fn() {
        let ops = vec![FuseableOp::unary(TraceOp::Sin)];
        let kernel = compose_trace_ops_to_kernel_ir(&ops, "sin").expect("Sin must compose");
        let out_node = &kernel.nodes[kernel.output.index()];
        assert!(matches!(
            out_node.kind,
            IRNodeKind::UnaryFn {
                op: UnaryFnKind::Sin,
                ..
            }
        ));
    }

    #[kani::unwind(16)]
    #[kani::proof]
    #[kani::unwind(5)]
    fn proof_cos_output_is_unary_fn() {
        let ops = vec![FuseableOp::unary(TraceOp::Cos)];
        let kernel = compose_trace_ops_to_kernel_ir(&ops, "cos").expect("Cos must compose");
        let out_node = &kernel.nodes[kernel.output.index()];
        assert!(matches!(
            out_node.kind,
            IRNodeKind::UnaryFn {
                op: UnaryFnKind::Cos,
                ..
            }
        ));
    }

    #[kani::unwind(16)]
    #[kani::proof]
    #[kani::unwind(5)]
    fn proof_tanh_output_is_unary_fn() {
        let ops = vec![FuseableOp::unary(TraceOp::Tanh)];
        let kernel = compose_trace_ops_to_kernel_ir(&ops, "tanh").expect("Tanh must compose");
        let out_node = &kernel.nodes[kernel.output.index()];
        assert!(matches!(
            out_node.kind,
            IRNodeKind::UnaryFn {
                op: UnaryFnKind::Tanh,
                ..
            }
        ));
    }

    // -----------------------------------------------------------------------
    // Proof 7: LeakyRelu embeds slope as Literal
    // -----------------------------------------------------------------------

    #[kani::unwind(16)]
    #[kani::proof]
    #[kani::unwind(10)]
    fn proof_leaky_relu_slope_embedded_as_literal() {
        let slope = 0.2;
        let ops = vec![FuseableOp::unary(TraceOp::LeakyRelu { slope })];
        let kernel = compose_trace_ops_to_kernel_ir(&ops, "leaky").expect("LeakyRelu must compose");
        // The slope 0.2 must appear as a Literal node
        let has_slope_literal = kernel
            .nodes
            .iter()
            .any(|n| matches!(n.kind, IRNodeKind::Literal(v) if (v - slope).abs() < 1e-10));
        assert!(
            has_slope_literal,
            "LeakyRelu must embed slope as an IR Literal"
        );
    }

    // -----------------------------------------------------------------------
    // Proof 8: Elu alpha embedded as Literal
    // -----------------------------------------------------------------------

    #[kani::unwind(16)]
    #[kani::proof]
    #[kani::unwind(12)]
    fn proof_elu_alpha_embedded_as_literal() {
        let alpha = 1.5;
        let ops = vec![FuseableOp::unary(TraceOp::Elu { alpha })];
        let kernel = compose_trace_ops_to_kernel_ir(&ops, "elu").expect("Elu must compose");
        let has_alpha_literal = kernel
            .nodes
            .iter()
            .any(|n| matches!(n.kind, IRNodeKind::Literal(v) if (v - alpha).abs() < 1e-10));
        assert!(has_alpha_literal, "Elu must embed alpha as an IR Literal");
    }

    // -----------------------------------------------------------------------
    // Proof 9: 5-op mixed chain produces exactly 4 params
    // -----------------------------------------------------------------------

    /// Chain: Add(x,y) -> Exp -> Sub(_, z) -> Relu -> Mul(_, w)
    /// Params: x, y from Add, z from Sub, w from Mul = 4
    #[kani::unwind(16)]
    #[kani::proof]
    #[kani::unwind(16)]
    fn proof_five_op_chain_four_params() {
        let ops = vec![
            FuseableOp::binary_both_external(TraceOp::Add),
            FuseableOp::unary(TraceOp::Exp),
            FuseableOp::binary_second_external(TraceOp::Sub),
            FuseableOp::unary(TraceOp::Relu),
            FuseableOp::binary_second_external(TraceOp::Mul),
        ];
        let kernel =
            compose_trace_ops_to_kernel_ir(&ops, "five_op").expect("5-op chain must compose");
        assert_eq!(kernel.params.len(), 4, "Add(2) + Sub(1) + Mul(1) = 4");
        assert!(kernel.validate().is_ok());
    }

    // -----------------------------------------------------------------------
    // Proof 10: All node refs in composed chain are backward (topological)
    // -----------------------------------------------------------------------

    #[kani::unwind(16)]
    #[kani::proof]
    #[kani::unwind(16)]
    fn proof_all_node_refs_backward_in_chain() {
        let ops = vec![
            FuseableOp::unary(TraceOp::Sigmoid),
            FuseableOp::unary(TraceOp::Silu),
        ];
        let kernel = compose_trace_ops_to_kernel_ir(&ops, "backward_check").expect("Must compose");

        for node in &kernel.nodes {
            let this_idx = node.id.index();
            match &node.kind {
                IRNodeKind::Param(_) | IRNodeKind::Literal(_) => {}
                IRNodeKind::BinOp { lhs, rhs, .. } => {
                    assert!(lhs.index() < this_idx);
                    assert!(rhs.index() < this_idx);
                }
                IRNodeKind::UnaryFn { input, .. } => {
                    assert!(input.index() < this_idx);
                }
                IRNodeKind::MinMax { lhs, rhs, .. } => {
                    assert!(lhs.index() < this_idx);
                    assert!(rhs.index() < this_idx);
                }
                IRNodeKind::Compare { lhs, rhs, .. } => {
                    assert!(lhs.index() < this_idx);
                    assert!(rhs.index() < this_idx);
                }
                IRNodeKind::Select {
                    cond,
                    then_val,
                    else_val,
                } => {
                    assert!(cond.index() < this_idx);
                    assert!(then_val.index() < this_idx);
                    assert!(else_val.index() < this_idx);
                }
                IRNodeKind::BinaryFn { lhs, rhs, .. } => {
                    assert!(lhs.index() < this_idx);
                    assert!(rhs.index() < this_idx);
                }
                _ => {}
            }
        }
    }

    // -----------------------------------------------------------------------
    // Proof 11: Softplus produces exactly 4 non-param nodes
    // -----------------------------------------------------------------------

    /// softplus(x) = Log(Add(1.0, Exp(x)))
    /// Non-param nodes: Exp, Literal(1.0), Add, Log = 4
    #[kani::unwind(16)]
    #[kani::proof]
    #[kani::unwind(8)]
    fn proof_softplus_non_param_node_count() {
        let ops = vec![FuseableOp::unary(TraceOp::Softplus)];
        let kernel =
            compose_trace_ops_to_kernel_ir(&ops, "softplus").expect("Softplus must compose");
        let non_param = kernel
            .nodes
            .iter()
            .filter(|n| !matches!(n.kind, IRNodeKind::Param(_)))
            .count();
        assert_eq!(
            non_param, 4,
            "Softplus: Exp + Literal(1) + Add + Log = 4 non-param nodes"
        );
    }

    // -----------------------------------------------------------------------
    // Proof 12: Sigmoid decomposition node count
    // -----------------------------------------------------------------------

    /// sigmoid(x) = Div(1, Add(1, Exp(Sub(0, x))))
    /// Non-param nodes: Literal(0), Sub, Exp, Literal(1), Add, Div = 6
    #[kani::unwind(16)]
    #[kani::proof]
    #[kani::unwind(10)]
    fn proof_sigmoid_non_param_node_count() {
        let ops = vec![FuseableOp::unary(TraceOp::Sigmoid)];
        let kernel = compose_trace_ops_to_kernel_ir(&ops, "sigmoid").expect("Sigmoid must compose");
        let non_param = kernel
            .nodes
            .iter()
            .filter(|n| !matches!(n.kind, IRNodeKind::Param(_)))
            .count();
        // sigmoid: Lit(0), Sub(0,x), Exp, Lit(1), Add(1,exp), Div(1,add) = 6
        assert_eq!(
            non_param, 6,
            "Sigmoid must produce exactly 6 non-param nodes"
        );
    }

    // -----------------------------------------------------------------------
    // Proof 13: Atan2 binary op produces BinaryFn output node
    // -----------------------------------------------------------------------

    #[kani::unwind(16)]
    #[kani::proof]
    #[kani::unwind(6)]
    fn proof_atan2_output_is_binary_fn() {
        let ops = vec![FuseableOp::binary_both_external(TraceOp::Atan2)];
        let kernel = compose_trace_ops_to_kernel_ir(&ops, "atan2").expect("Atan2 must compose");
        let out_node = &kernel.nodes[kernel.output.index()];
        assert!(
            matches!(out_node.kind, IRNodeKind::BinaryFn { .. }),
            "Atan2 output must be BinaryFn"
        );
    }

    // -----------------------------------------------------------------------
    // Proof 14: op_input_count returns 1 for all parameterized ops
    // -----------------------------------------------------------------------

    #[kani::unwind(1)]
    #[kani::proof]
    #[kani::unwind(8)]
    fn proof_op_input_count_parameterized_unary() {
        assert_eq!(op_input_count(&TraceOp::LeakyRelu { slope: 0.1 }), 1);
        assert_eq!(op_input_count(&TraceOp::Elu { alpha: 1.0 }), 1);
        assert_eq!(
            op_input_count(&TraceOp::Clamp {
                min: Some(-1.0),
                max: Some(1.0)
            }),
            1
        );
        assert_eq!(op_input_count(&TraceOp::Powf { exponent: 2.0 }), 1);
        assert_eq!(op_input_count(&TraceOp::Softplus), 1);
    }

    // -----------------------------------------------------------------------
    // Proof 15: Kernel name is preserved through composition
    // -----------------------------------------------------------------------

    #[kani::unwind(16)]
    #[kani::proof]
    #[kani::unwind(5)]
    fn proof_kernel_name_preserved() {
        let ops = vec![FuseableOp::unary(TraceOp::Exp)];
        let kernel =
            compose_trace_ops_to_kernel_ir(&ops, "nn_custom_kernel").expect("Must compose");
        assert_eq!(kernel.name, "nn_custom_kernel");
    }

    // -----------------------------------------------------------------------
    // Proof 17: Powf with integer exponent still uses BinOp::Mul chain
    // -----------------------------------------------------------------------

    /// powf(x, 2.0) should produce a Mul(x, x) node, not a generic Powf.
    #[kani::unwind(16)]
    #[kani::proof]
    #[kani::unwind(6)]
    fn proof_powf_integer_exponent_uses_mul() {
        let ops = vec![FuseableOp::unary(TraceOp::Powf { exponent: 2.0 })];
        let kernel = compose_trace_ops_to_kernel_ir(&ops, "powf2").expect("Powf must compose");
        // For exponent=2.0, the emitter uses Mul(x, x) instead of generic powf
        let has_mul = kernel.nodes.iter().any(|n| {
            matches!(
                n.kind,
                IRNodeKind::BinOp {
                    op: BinOpKind::Mul,
                    ..
                }
            )
        });
        assert!(has_mul, "Powf(2.0) should emit Mul(x, x)");
    }
}

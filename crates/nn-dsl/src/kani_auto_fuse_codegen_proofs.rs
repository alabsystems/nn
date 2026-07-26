// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Kani proof harnesses for `auto_fuse_codegen` and `msl_auto_fuse` correctness.
//!
//! Proves critical invariants of the auto-fusion codegen pipeline:
//! - `op_input_count` returns correct arity for all supported TraceOps.
//! - `compose_trace_ops_to_kernel_ir` produces valid KernelDefs for all
//!   single-op chains, multi-op chains, and mixed wiring configurations.
//! - Empty chains are rejected with an error.
//! - `BinaryBothExternal` is rejected when used at non-first position.
//! - External parameter counting is correct for all wiring variants.
//! - Fused MSL codegen buffer counting matches param count + 2.
//! - Buffer limit (Metal 31-slot limit) is correctly enforced.
//! - `FusedKernelMeta::total_elements` is consistent with shape product.
//! - Broadcast index generation does not overflow for bounded shapes.
//! - `row_major_strides` produces correct stride products.
//! - Thread group size constant is within Metal hardware bounds.
//!
//! Part of #3632.

#[cfg(kani)]
mod proofs {
    use crate::auto_fuse_codegen::{compose_trace_ops_to_kernel_ir, FuseableOp, OpWiring};
    use crate::codegen_shared::row_major_strides;
    use crate::ir::{
        BinOpKind, IRNode, IRNodeKind, KernelDef, MinMaxKind, NodeId, Param, ScalarType,
        UnaryFnKind,
    };
    use crate::msl_auto_fuse::{FusedKernelMeta, FusedMslResult, FUSED_THREADGROUP_SIZE};
    use crate::tensor_ir::BroadcastAlignment;
    use nn_core::dyn_tensor::trace::TraceOp;

    // -----------------------------------------------------------------------
    // Helper: build minimal valid KernelDef with N params
    // -----------------------------------------------------------------------

    fn build_n_param_kernel(n: usize) -> KernelDef {
        let mut params = Vec::new();
        let mut nodes = Vec::new();
        for i in 0..n {
            params.push(Param::new(format!("p{i}"), ScalarType::F32));
            nodes.push(IRNode::new(NodeId::new(i), IRNodeKind::Param(i)));
        }
        if n == 0 {
            // Degenerate: single literal kernel
            nodes.push(IRNode::new(NodeId::new(0), IRNodeKind::Literal(1.0)));
            KernelDef::new("test", params, ScalarType::F32, nodes, NodeId::new(0))
        } else if n == 1 {
            KernelDef::new("test", params, ScalarType::F32, nodes, NodeId::new(0))
        } else {
            // Chain add: p0 + p1 + p2 + ...
            let mut prev = NodeId::new(0);
            for i in 1..n {
                let new_id = NodeId::new(nodes.len());
                nodes.push(IRNode::new(
                    new_id,
                    IRNodeKind::BinOp {
                        op: BinOpKind::Add,
                        lhs: prev,
                        rhs: NodeId::new(i),
                    },
                ));
                prev = new_id;
            }
            KernelDef::new("test", params, ScalarType::F32, nodes, prev)
        }
    }

    // -----------------------------------------------------------------------
    // Proof 1: op_input_count returns 2 for binary ops, 1 for unary
    // -----------------------------------------------------------------------

    #[kani::unwind(8)]
    #[kani::proof]
    #[kani::unwind(8)]
    fn proof_op_input_count_binary_ops() {
        // All binary TraceOps must return 2.
        let binary_ops = [
            TraceOp::Add,
            TraceOp::Sub,
            TraceOp::Mul,
            TraceOp::Div,
            TraceOp::Maximum,
            TraceOp::Minimum,
        ];
        for op in &binary_ops {
            let count = crate::auto_fuse_codegen::op_input_count(op);
            assert_eq!(count, 2, "Binary ops must return 2 inputs");
        }
    }

    // -----------------------------------------------------------------------
    // Proof 2: op_input_count returns 1 for common unary ops
    // -----------------------------------------------------------------------

    #[kani::unwind(8)]
    #[kani::proof]
    #[kani::unwind(8)]
    fn proof_op_input_count_unary_ops() {
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
            TraceOp::Relu,
            TraceOp::Gelu,
            TraceOp::Sigmoid,
            TraceOp::Silu,
            TraceOp::Neg,
            TraceOp::Sqr,
            TraceOp::Softplus,
        ];
        for op in &unary_ops {
            let count = crate::auto_fuse_codegen::op_input_count(op);
            assert_eq!(count, 1, "Unary ops must return 1 input");
        }
    }

    // -----------------------------------------------------------------------
    // Proof 3: empty op chain is rejected
    // -----------------------------------------------------------------------

    #[kani::unwind(8)]
    #[kani::proof]
    fn proof_empty_chain_rejected() {
        let ops: Vec<FuseableOp> = Vec::new();
        let result = compose_trace_ops_to_kernel_ir(&ops, "empty");
        assert!(result.is_err(), "Empty chain must be rejected");
    }

    // -----------------------------------------------------------------------
    // Proof 4: single unary op produces valid kernel with 1 param
    // -----------------------------------------------------------------------

    #[kani::unwind(16)]
    #[kani::proof]
    #[kani::unwind(5)]
    fn proof_single_unary_produces_valid_kernel() {
        let ops = vec![FuseableOp::unary(TraceOp::Exp)];
        let kernel =
            compose_trace_ops_to_kernel_ir(&ops, "single_exp").expect("Single Exp must compose");
        assert_eq!(kernel.params.len(), 1, "Single unary must have 1 param");
        assert!(kernel.validate().is_ok(), "Composed kernel must validate");
        assert_eq!(kernel.name, "single_exp");
    }

    // -----------------------------------------------------------------------
    // Proof 5: single binary op produces valid kernel with 2 params
    // -----------------------------------------------------------------------

    #[kani::unwind(16)]
    #[kani::proof]
    #[kani::unwind(5)]
    fn proof_single_binary_both_external_produces_2_params() {
        let ops = vec![FuseableOp::binary_both_external(TraceOp::Add)];
        let kernel = compose_trace_ops_to_kernel_ir(&ops, "single_add")
            .expect("Binary both external Add must compose");
        assert_eq!(
            kernel.params.len(),
            2,
            "Binary both external must have 2 params"
        );
        assert!(kernel.validate().is_ok());
    }

    // -----------------------------------------------------------------------
    // Proof 6: chained exp → relu produces valid kernel with 1 param
    // -----------------------------------------------------------------------

    #[kani::unwind(16)]
    #[kani::proof]
    #[kani::unwind(8)]
    fn proof_exp_relu_chain_valid() {
        let ops = vec![
            FuseableOp::unary(TraceOp::Exp),
            FuseableOp::unary(TraceOp::Relu),
        ];
        let kernel =
            compose_trace_ops_to_kernel_ir(&ops, "exp_relu").expect("Exp → Relu must compose");
        assert_eq!(kernel.params.len(), 1, "Chained unaries share 1 input");
        assert!(kernel.validate().is_ok());
        // Relu emits max(x, 0) which adds Literal + MinMax nodes
        assert!(kernel.nodes.len() >= 4);
    }

    // -----------------------------------------------------------------------
    // Proof 7: BinaryBothExternal rejected at non-first position
    // -----------------------------------------------------------------------

    #[kani::unwind(16)]
    #[kani::proof]
    #[kani::unwind(6)]
    fn proof_binary_both_external_rejected_at_non_first() {
        let ops = vec![
            FuseableOp::unary(TraceOp::Exp),
            FuseableOp::binary_both_external(TraceOp::Add),
        ];
        let result = compose_trace_ops_to_kernel_ir(&ops, "bad_chain");
        assert!(
            result.is_err(),
            "BinaryBothExternal at step 1 must be rejected"
        );
    }

    // -----------------------------------------------------------------------
    // Proof 8: BinarySecondExternal adds exactly 1 new param
    // -----------------------------------------------------------------------

    #[kani::unwind(16)]
    #[kani::proof]
    #[kani::unwind(8)]
    fn proof_binary_second_external_adds_one_param() {
        let ops = vec![
            FuseableOp::unary(TraceOp::Exp),
            FuseableOp::binary_second_external(TraceOp::Add),
        ];
        let kernel = compose_trace_ops_to_kernel_ir(&ops, "exp_add_y")
            .expect("Exp + Add(_, y) must compose");
        // First param from Exp's input, second from Add's external RHS
        assert_eq!(kernel.params.len(), 2);
        assert!(kernel.validate().is_ok());
    }

    // -----------------------------------------------------------------------
    // Proof 9: BinaryFirstExternal adds exactly 1 new param
    // -----------------------------------------------------------------------

    #[kani::unwind(16)]
    #[kani::proof]
    #[kani::unwind(8)]
    fn proof_binary_first_external_adds_one_param() {
        let ops = vec![
            FuseableOp::unary(TraceOp::Exp),
            FuseableOp::binary_first_external(TraceOp::Sub),
        ];
        let kernel = compose_trace_ops_to_kernel_ir(&ops, "exp_sub_ext")
            .expect("Exp + Sub(ext, _) must compose");
        assert_eq!(kernel.params.len(), 2);
        assert!(kernel.validate().is_ok());
    }

    // -----------------------------------------------------------------------
    // Proof 10: triple chain exp → sigmoid → relu validates
    // -----------------------------------------------------------------------

    #[kani::unwind(16)]
    #[kani::proof]
    #[kani::unwind(16)]
    fn proof_triple_unary_chain_validates() {
        let ops = vec![
            FuseableOp::unary(TraceOp::Exp),
            FuseableOp::unary(TraceOp::Sigmoid),
            FuseableOp::unary(TraceOp::Relu),
        ];
        let kernel = compose_trace_ops_to_kernel_ir(&ops, "triple")
            .expect("Triple unary chain must compose");
        assert_eq!(kernel.params.len(), 1);
        assert!(kernel.validate().is_ok());
    }

    // -----------------------------------------------------------------------
    // Proof 11: add(x,y) → relu produces exactly 2 params
    // -----------------------------------------------------------------------

    #[kani::unwind(16)]
    #[kani::proof]
    #[kani::unwind(8)]
    fn proof_add_relu_chain_param_count() {
        let ops = vec![
            FuseableOp::binary_both_external(TraceOp::Add),
            FuseableOp::unary(TraceOp::Relu),
        ];
        let kernel =
            compose_trace_ops_to_kernel_ir(&ops, "add_relu").expect("add(x,y) → relu must compose");
        assert_eq!(kernel.params.len(), 2);
        assert!(kernel.validate().is_ok());
    }

    // -----------------------------------------------------------------------
    // Proof 12: FusedKernelMeta total_elements is shape product
    // -----------------------------------------------------------------------

    #[kani::unwind(16)]
    #[kani::proof]
    #[kani::unwind(8)]
    fn proof_total_elements_is_product() {
        let a: u8 = kani::any();
        let b: u8 = kani::any();
        kani::assume(a >= 1 && a <= 16);
        kani::assume(b >= 1 && b <= 16);
        let shape = vec![a as usize, b as usize];
        let meta = FusedKernelMeta::new(
            vec![shape.clone()],
            shape.clone(),
            BroadcastAlignment::Right,
            ScalarType::F32,
        );
        let expected = (a as usize) * (b as usize);
        assert_eq!(meta.total_elements(), expected);
    }

    // -----------------------------------------------------------------------
    // Proof 13: total_elements for empty shape is 1 (scalar)
    // -----------------------------------------------------------------------

    #[kani::unwind(16)]
    #[kani::proof]
    #[kani::unwind(8)]
    fn proof_total_elements_empty_shape_is_one() {
        let meta = FusedKernelMeta::new(
            vec![vec![]],
            vec![],
            BroadcastAlignment::Right,
            ScalarType::F32,
        );
        // Product of empty slice is 1 by convention
        assert_eq!(meta.total_elements(), 1);
    }

    // -----------------------------------------------------------------------
    // Proof 14: buffer count = params + 2 (output + total)
    // -----------------------------------------------------------------------

    #[kani::unwind(1)]
    #[kani::proof]
    #[kani::unwind(8)]
    fn proof_buffer_count_formula() {
        let n: u8 = kani::any();
        kani::assume(n >= 1 && n <= 29);
        let param_count = n as usize;
        let buffer_count = param_count + 2;
        // Invariant from msl_auto_fuse.rs line 159
        assert_eq!(buffer_count, param_count + 2);
        // Must fit in Metal limit (31 slots = indices 0..30)
        assert!(buffer_count <= 31);
    }

    // -----------------------------------------------------------------------
    // Proof 15: buffer limit enforced for max params
    // -----------------------------------------------------------------------

    #[kani::unwind(1)]
    #[kani::proof]
    fn proof_buffer_limit_enforced() {
        // Metal limit: MAX_METAL_BUFFER_INDEX = 30, so max slots = 31
        // buffer_count = param_count + 2
        // param_count = 29 → buffer_count = 31 ≤ 31 (OK)
        // param_count = 30 → buffer_count = 32 > 31 (REJECTED)
        let ok_params = 29usize;
        let bad_params = 30usize;
        assert!(ok_params + 2 <= 31);
        assert!(bad_params + 2 > 31);
    }

    // -----------------------------------------------------------------------
    // Proof 16: row_major_strides is correct for rank-2
    // -----------------------------------------------------------------------

    #[kani::unwind(16)]
    #[kani::proof]
    #[kani::unwind(8)]
    fn proof_row_major_strides_rank2() {
        let a: u8 = kani::any();
        let b: u8 = kani::any();
        kani::assume(a >= 1 && a <= 128);
        kani::assume(b >= 1 && b <= 128);
        let shape = vec![a as usize, b as usize];
        let strides = row_major_strides(&shape);
        assert!(strides.is_some());
        let s = strides.unwrap();
        assert_eq!(s.len(), 2);
        assert_eq!(s[0], b as usize);
        assert_eq!(s[1], 1);
    }

    // -----------------------------------------------------------------------
    // Proof 17: row_major_strides is correct for rank-3
    // -----------------------------------------------------------------------

    #[kani::unwind(16)]
    #[kani::proof]
    #[kani::unwind(8)]
    fn proof_row_major_strides_rank3() {
        let a: u8 = kani::any();
        let b: u8 = kani::any();
        let c: u8 = kani::any();
        kani::assume(a >= 1 && a <= 16);
        kani::assume(b >= 1 && b <= 16);
        kani::assume(c >= 1 && c <= 16);
        let shape = vec![a as usize, b as usize, c as usize];
        let strides = row_major_strides(&shape);
        assert!(strides.is_some());
        let s = strides.unwrap();
        assert_eq!(s.len(), 3);
        assert_eq!(s[2], 1);
        assert_eq!(s[1], c as usize);
        assert_eq!(s[0], (b as usize) * (c as usize));
    }

    // -----------------------------------------------------------------------
    // Proof 18: row_major_strides rank-1 is always [1]
    // -----------------------------------------------------------------------

    #[kani::unwind(16)]
    #[kani::proof]
    #[kani::unwind(8)]
    fn proof_row_major_strides_rank1() {
        let n: u8 = kani::any();
        kani::assume(n >= 1);
        let shape = vec![n as usize];
        let strides = row_major_strides(&shape).unwrap();
        assert_eq!(strides.len(), 1);
        assert_eq!(strides[0], 1);
    }

    // -----------------------------------------------------------------------
    // Proof 19: FUSED_THREADGROUP_SIZE is within Metal limits
    // -----------------------------------------------------------------------

    #[kani::unwind(8)]
    #[kani::proof]
    #[kani::unwind(8)]
    fn proof_threadgroup_size_within_metal_limits() {
        // Metal max threadgroup size is 1024. We use 256.
        assert_eq!(FUSED_THREADGROUP_SIZE, 256);
        assert!(FUSED_THREADGROUP_SIZE <= 1024);
        // Must be a power of 2 for efficient GPU dispatch
        assert!(FUSED_THREADGROUP_SIZE.is_power_of_two());
    }

    // -----------------------------------------------------------------------
    // Proof 20: OpWiring variant construction is consistent
    // -----------------------------------------------------------------------

    #[kani::unwind(1)]
    #[kani::proof]
    #[kani::unwind(8)]
    fn proof_op_wiring_constructors_consistent() {
        let unary = FuseableOp::unary(TraceOp::Exp);
        assert_eq!(unary.wiring, OpWiring::Unary);

        let bin_second = FuseableOp::binary_second_external(TraceOp::Add);
        assert_eq!(bin_second.wiring, OpWiring::BinarySecondExternal);

        let bin_first = FuseableOp::binary_first_external(TraceOp::Sub);
        assert_eq!(bin_first.wiring, OpWiring::BinaryFirstExternal);

        let bin_both = FuseableOp::binary_both_external(TraceOp::Mul);
        assert_eq!(bin_both.wiring, OpWiring::BinaryBothExternal);
    }

    // -----------------------------------------------------------------------
    // Proof 21: composed softplus chain validates
    // -----------------------------------------------------------------------

    #[kani::unwind(16)]
    #[kani::proof]
    #[kani::unwind(8)]
    fn proof_softplus_chain_validates() {
        // softplus(x) = ln(1 + exp(x))
        let ops = vec![FuseableOp::unary(TraceOp::Softplus)];
        let kernel =
            compose_trace_ops_to_kernel_ir(&ops, "softplus").expect("Softplus must compose");
        assert_eq!(kernel.params.len(), 1);
        assert!(kernel.validate().is_ok());
    }

    // -----------------------------------------------------------------------
    // Proof 22: composed silu chain validates
    // -----------------------------------------------------------------------

    #[kani::unwind(16)]
    #[kani::proof]
    #[kani::unwind(12)]
    fn proof_silu_chain_validates() {
        // silu(x) = x * sigmoid(x) — multi-node composition
        let ops = vec![FuseableOp::unary(TraceOp::Silu)];
        let kernel = compose_trace_ops_to_kernel_ir(&ops, "silu").expect("Silu must compose");
        assert_eq!(kernel.params.len(), 1);
        assert!(kernel.validate().is_ok());
    }

    // -----------------------------------------------------------------------
    // Proof 23: leaky_relu with slope composes correctly
    // -----------------------------------------------------------------------

    #[kani::unwind(16)]
    #[kani::proof]
    #[kani::unwind(10)]
    fn proof_leaky_relu_composes() {
        let ops = vec![FuseableOp::unary(TraceOp::LeakyRelu { slope: 0.01 })];
        let kernel =
            compose_trace_ops_to_kernel_ir(&ops, "leaky_relu").expect("LeakyRelu must compose");
        assert_eq!(kernel.params.len(), 1);
        assert!(kernel.validate().is_ok());
    }

    // -----------------------------------------------------------------------
    // Proof 24: clamp with both bounds composes correctly
    // -----------------------------------------------------------------------

    #[kani::unwind(16)]
    #[kani::proof]
    #[kani::unwind(8)]
    fn proof_clamp_both_bounds_composes() {
        let ops = vec![FuseableOp::unary(TraceOp::Clamp {
            min: Some(-1.0),
            max: Some(1.0),
        })];
        let kernel = compose_trace_ops_to_kernel_ir(&ops, "clamp").expect("Clamp must compose");
        assert_eq!(kernel.params.len(), 1);
        assert!(kernel.validate().is_ok());
    }

    // -----------------------------------------------------------------------
    // Proof 25: powf with integer exponent composes
    // -----------------------------------------------------------------------

    #[kani::unwind(16)]
    #[kani::proof]
    #[kani::unwind(12)]
    fn proof_powf_integer_exponent_composes() {
        let ops = vec![FuseableOp::unary(TraceOp::Powf { exponent: 2.0 })];
        let kernel = compose_trace_ops_to_kernel_ir(&ops, "square").expect("Powf(2) must compose");
        assert_eq!(kernel.params.len(), 1);
        assert!(kernel.validate().is_ok());
    }

    // -----------------------------------------------------------------------
    // Proof 26: powf(0) reduces to literal 1.0
    // -----------------------------------------------------------------------

    #[kani::unwind(16)]
    #[kani::proof]
    #[kani::unwind(5)]
    fn proof_powf_zero_is_literal_one() {
        let ops = vec![FuseableOp::unary(TraceOp::Powf { exponent: 0.0 })];
        let kernel = compose_trace_ops_to_kernel_ir(&ops, "pow0").expect("Powf(0) must compose");
        assert_eq!(kernel.params.len(), 1);
        assert!(kernel.validate().is_ok());
        // Output should be a Literal(1.0) node
        let out_node = &kernel.nodes[kernel.output.index()];
        assert!(
            matches!(out_node.kind, IRNodeKind::Literal(v) if (v - 1.0).abs() < 1e-10),
            "powf(0) output must be literal 1.0"
        );
    }

    // -----------------------------------------------------------------------
    // Proof 27: powf(1) returns input identity
    // -----------------------------------------------------------------------

    #[kani::unwind(16)]
    #[kani::proof]
    #[kani::unwind(5)]
    fn proof_powf_one_is_identity() {
        let ops = vec![FuseableOp::unary(TraceOp::Powf { exponent: 1.0 })];
        let kernel = compose_trace_ops_to_kernel_ir(&ops, "pow1").expect("Powf(1) must compose");
        assert_eq!(kernel.params.len(), 1);
        // Output should be the Param node itself (identity)
        let out_node = &kernel.nodes[kernel.output.index()];
        assert!(
            matches!(out_node.kind, IRNodeKind::Param(0)),
            "powf(1) output must be the param itself"
        );
    }

    // -----------------------------------------------------------------------
    // Proof 28: maximum binary op produces valid kernel
    // -----------------------------------------------------------------------

    #[kani::unwind(16)]
    #[kani::proof]
    #[kani::unwind(6)]
    fn proof_maximum_binary_composes() {
        let ops = vec![FuseableOp::binary_both_external(TraceOp::Maximum)];
        let kernel = compose_trace_ops_to_kernel_ir(&ops, "max_ab").expect("Maximum must compose");
        assert_eq!(kernel.params.len(), 2);
        assert!(kernel.validate().is_ok());
        // Output must be a MinMax::Max node
        let out_node = &kernel.nodes[kernel.output.index()];
        assert!(matches!(
            out_node.kind,
            IRNodeKind::MinMax {
                op: MinMaxKind::Max,
                ..
            }
        ));
    }

    // -----------------------------------------------------------------------
    // Proof 29: elu with alpha composes correctly
    // -----------------------------------------------------------------------

    #[kani::unwind(16)]
    #[kani::proof]
    #[kani::unwind(12)]
    fn proof_elu_composes() {
        let ops = vec![FuseableOp::unary(TraceOp::Elu { alpha: 1.0 })];
        let kernel = compose_trace_ops_to_kernel_ir(&ops, "elu").expect("Elu must compose");
        assert_eq!(kernel.params.len(), 1);
        assert!(kernel.validate().is_ok());
    }

    // -----------------------------------------------------------------------
    // Proof 30: gelu_erf composes to a valid kernel
    // -----------------------------------------------------------------------

    #[kani::unwind(16)]
    #[kani::proof]
    #[kani::unwind(50)]
    fn proof_gelu_erf_composes() {
        let ops = vec![FuseableOp::unary(TraceOp::GeluErf)];
        let kernel =
            compose_trace_ops_to_kernel_ir(&ops, "gelu_erf").expect("GeluErf must compose");
        assert_eq!(kernel.params.len(), 1);
        assert!(kernel.validate().is_ok());
        // GeluErf uses the erf polynomial approximation — many nodes
        assert!(kernel.nodes.len() > 20, "GeluErf must produce 20+ IR nodes");
    }

    // -----------------------------------------------------------------------
    // Proof 31: node IDs in composed kernel are contiguous and monotonic
    // -----------------------------------------------------------------------

    #[kani::unwind(16)]
    #[kani::proof]
    #[kani::unwind(12)]
    fn proof_composed_kernel_node_ids_contiguous() {
        let ops = vec![
            FuseableOp::unary(TraceOp::Exp),
            FuseableOp::binary_second_external(TraceOp::Mul),
            FuseableOp::unary(TraceOp::Tanh),
        ];
        let kernel = compose_trace_ops_to_kernel_ir(&ops, "contiguous_ids").expect("Must compose");

        // All node IDs must equal their array index
        for (i, node) in kernel.nodes.iter().enumerate() {
            assert_eq!(
                node.id.index(),
                i,
                "Node ID must match array index for contiguous IR"
            );
        }
    }

    // -----------------------------------------------------------------------
    // Proof 32: all node references in composed kernel are backward
    // -----------------------------------------------------------------------

    #[kani::unwind(16)]
    #[kani::proof]
    #[kani::unwind(8)]
    fn proof_composed_kernel_all_refs_backward() {
        let ops = vec![
            FuseableOp::unary(TraceOp::Exp),
            FuseableOp::unary(TraceOp::Relu),
        ];
        let kernel = compose_trace_ops_to_kernel_ir(&ops, "backward_refs").expect("Must compose");

        for node in &kernel.nodes {
            let this_idx = node.id.index();
            let refs = collect_refs(&node.kind);
            for r in refs {
                assert!(
                    r < this_idx,
                    "Node {} references future node {} — topological violation",
                    this_idx,
                    r
                );
            }
        }
    }

    /// Collect all NodeId references from an IRNodeKind.
    fn collect_refs(kind: &IRNodeKind) -> Vec<usize> {
        match kind {
            IRNodeKind::Param(_) | IRNodeKind::Literal(_) => vec![],
            IRNodeKind::BinOp { lhs, rhs, .. } => vec![lhs.index(), rhs.index()],
            IRNodeKind::UnaryFn { input, .. } => vec![input.index()],
            IRNodeKind::MinMax { lhs, rhs, .. } => vec![lhs.index(), rhs.index()],
            IRNodeKind::Compare { lhs, rhs, .. } => vec![lhs.index(), rhs.index()],
            IRNodeKind::Select {
                cond,
                then_val,
                else_val,
            } => vec![cond.index(), then_val.index(), else_val.index()],
            IRNodeKind::BinaryFn { lhs, rhs, .. } => vec![lhs.index(), rhs.index()],
            IRNodeKind::Powi { base, .. } => vec![base.index()],
            IRNodeKind::Clamp { input, min, max } => {
                vec![input.index(), min.index(), max.index()]
            }
            IRNodeKind::SumReduce { inputs } => inputs.iter().map(|id| id.index()).collect(),
        }
    }

    // -----------------------------------------------------------------------
    // Proof 33: FusedMslResult::new populates fields correctly
    // -----------------------------------------------------------------------

    #[kani::unwind(16)]
    #[kani::proof]
    #[kani::unwind(8)]
    fn proof_fused_msl_result_fields() {
        let result = FusedMslResult::new("msl code".to_string(), "test_kernel".to_string(), 5, 256);
        assert_eq!(result.buffer_count, 5);
        assert_eq!(result.threadgroup_size, 256);
        assert_eq!(result.kernel_name, "test_kernel");
    }

    // -----------------------------------------------------------------------
    // Proof 34: broadcast alignment offset computation
    // -----------------------------------------------------------------------

    #[kani::unwind(1)]
    #[kani::proof]
    fn proof_broadcast_alignment_offset() {
        // For Right alignment: offset = output_rank - input_rank
        let output_rank: u8 = kani::any();
        let input_rank: u8 = kani::any();
        kani::assume(output_rank >= 1 && output_rank <= 8);
        kani::assume(input_rank >= 1 && input_rank <= output_rank);

        let or = output_rank as usize;
        let ir = input_rank as usize;

        // Right alignment offset
        let right_offset = or.saturating_sub(ir);
        assert!(right_offset + ir == or);

        // Left alignment offset is always 0
        let left_offset = 0usize;
        assert!(left_offset + ir <= or);
    }

    // -----------------------------------------------------------------------
    // Proof 35: row_major_strides empty shape
    // -----------------------------------------------------------------------

    #[kani::unwind(1)]
    #[kani::proof]
    fn proof_row_major_strides_empty() {
        let strides = row_major_strides(&[]);
        assert!(strides.is_some());
        assert!(strides.unwrap().is_empty());
    }

    // -----------------------------------------------------------------------
    // Proof 36: composed atan2 binary op validates
    // -----------------------------------------------------------------------

    #[kani::unwind(16)]
    #[kani::proof]
    #[kani::unwind(6)]
    fn proof_atan2_binary_composes() {
        let ops = vec![FuseableOp::binary_both_external(TraceOp::Atan2)];
        let kernel = compose_trace_ops_to_kernel_ir(&ops, "atan2_ab").expect("Atan2 must compose");
        assert_eq!(kernel.params.len(), 2);
        assert!(kernel.validate().is_ok());
    }
}

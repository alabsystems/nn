// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

#[cfg(kani)]
mod proofs {
    use kani::assume;

    use crate::auto_fuse_codegen::{compose_trace_ops_to_kernel_ir, FuseableOp};
    use crate::ir::{IRNodeKind, KernelDef, NodeId};
    use nn_core::dyn_tensor::trace::TraceOp;

    fn assert_ref_precedes(current_idx: usize, id: NodeId) {
        assert!(
            id.index() < current_idx,
            "IR node {current_idx} must not reference itself or a future node"
        );
    }

    fn assert_kernel_wiring_consistent(kernel: &KernelDef) {
        assert_eq!(
            kernel.output.index(),
            kernel.nodes.len() - 1,
            "composed auto-fuse kernels should return the last emitted node"
        );

        for (idx, param) in kernel.params.iter().enumerate() {
            assert_eq!(param.name, format!("p{idx}"));
        }

        for (idx, node) in kernel.nodes.iter().enumerate() {
            assert_eq!(node.id.index(), idx, "node ids must stay dense");

            match &node.kind {
                IRNodeKind::Param(param_idx) => {
                    assert!(
                        *param_idx < kernel.params.len(),
                        "param node must reference a declared kernel param"
                    );
                }
                IRNodeKind::Literal(_) => {}
                IRNodeKind::BinOp { lhs, rhs, .. }
                | IRNodeKind::Compare { lhs, rhs, .. }
                | IRNodeKind::MinMax { lhs, rhs, .. }
                | IRNodeKind::BinaryFn { lhs, rhs, .. } => {
                    assert_ref_precedes(idx, *lhs);
                    assert_ref_precedes(idx, *rhs);
                }
                IRNodeKind::UnaryFn { input, .. } => {
                    assert_ref_precedes(idx, *input);
                }
                IRNodeKind::Powi { base, .. } => {
                    assert_ref_precedes(idx, *base);
                }
                IRNodeKind::Clamp { input, min, max } => {
                    assert_ref_precedes(idx, *input);
                    assert_ref_precedes(idx, *min);
                    assert_ref_precedes(idx, *max);
                }
                IRNodeKind::Select {
                    cond,
                    then_val,
                    else_val,
                } => {
                    assert_ref_precedes(idx, *cond);
                    assert_ref_precedes(idx, *then_val);
                    assert_ref_precedes(idx, *else_val);
                }
                IRNodeKind::SumReduce { inputs } => {
                    for input in inputs {
                        assert_ref_precedes(idx, *input);
                    }
                }
            }
        }
    }

    #[kani::unwind(16)]
    #[kani::proof]
    #[kani::unwind(32)]
    fn proof_issue3731_auto_fuse_ir_validity_and_wiring_consistency() {
        let second_step: u8 = kani::any();
        assume(second_step <= 2);

        let second = match second_step {
            0 => FuseableOp::unary(TraceOp::Sigmoid),
            1 => FuseableOp::binary_second_external(TraceOp::Mul),
            _ => FuseableOp::binary_first_external(TraceOp::Sub),
        };
        let ops = vec![FuseableOp::unary(TraceOp::Exp), second];

        let kernel = compose_trace_ops_to_kernel_ir(&ops, "issue3731_fused")
            .expect("supported bounded chain should compose");

        assert!(kernel.validate().is_ok(), "composed kernel must validate");
        assert_eq!(
            kernel.params.len(),
            if second_step == 0 { 1 } else { 2 },
            "unary tails must reuse the same input; binary tails add one external buffer"
        );
        assert_kernel_wiring_consistent(&kernel);
    }

    #[kani::unwind(16)]
    #[kani::proof]
    #[kani::unwind(32)]
    fn proof_issue3731_binary_both_external_wires_inputs_in_order() {
        let tail_kind: u8 = kani::any();
        assume(tail_kind <= 1);

        let mut ops = vec![FuseableOp::binary_both_external(TraceOp::Add)];
        if tail_kind == 0 {
            ops.push(FuseableOp::unary(TraceOp::Relu));
        } else {
            ops.push(FuseableOp::binary_second_external(TraceOp::Mul));
        }

        let kernel = compose_trace_ops_to_kernel_ir(&ops, "issue3731_both_external")
            .expect("first-op BinaryBothExternal should remain valid");

        assert!(kernel.validate().is_ok());
        assert_eq!(
            kernel.params.len(),
            if tail_kind == 0 { 2 } else { 3 },
            "the first binary step consumes exactly two external buffers"
        );
        assert_kernel_wiring_consistent(&kernel);

        match &kernel.nodes[0].kind {
            IRNodeKind::Param(0) => {}
            other => assert!(false, "expected first node to be p0, got {other:?}"),
        }
        match &kernel.nodes[1].kind {
            IRNodeKind::Param(1) => {}
            other => assert!(false, "expected second node to be p1, got {other:?}"),
        }
        match &kernel.nodes[2].kind {
            IRNodeKind::BinOp { lhs, rhs, .. } => {
                assert_eq!(*lhs, NodeId::new(0));
                assert_eq!(*rhs, NodeId::new(1));
            }
            other => assert!(false, "expected add node at step 2, got {other:?}"),
        }
    }
}

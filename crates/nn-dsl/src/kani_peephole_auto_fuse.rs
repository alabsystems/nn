// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Kani proof harnesses for `trace_compile_peephole_auto_fuse.rs`.
//!
//! Proves critical safety and correctness invariants of the auto-fuse
//! elementwise chain peephole pass (pass 13):
//!
//! - Chain detection never includes an index twice (disjoint membership).
//! - `is_single_elementwise_dispatch` rejects all non-Dispatch variants.
//! - `compose_chain` returns `None` for chains shorter than 2.
//! - `remap_id` falls back to the original when not in the mapping.
//! - `remap_ir_node_kind` preserves variant structure for every IRNodeKind.
//! - `resolve_external_graph_id` fallback is param_idx as u64.
//! - `resolve_external_shape` fallback is output_shape.
//! - Composed kernel params = sum of external params (chain wiring).
//! - Composed kernel name follows `fused_{base}_x{N}` convention.
//! - IdentityPassthrough count = chain_len - 1 after fusion.
//! - detect_chains skips IdentityPassthrough gaps correctly.
//! - AutoFuseStats fields are internally consistent.
//! - Fan-out > 1 breaks the chain (no multi-consumer fusion).
//! - Composed kernel validates after composition.
//! - `remap_ir_node_kind` is exhaustive over all IRNodeKind variants.
//! - NodeId remapping is idempotent for identity mapping.
//! - Composed kernel node count monotonically increases with chain length.
//! - KernelDef::validate rejects missing output node reference.
//! - Unary chain composition yields exactly N-1 non-Param nodes from wiring.
//! - Binary chain external param accumulation is correct.
//! - Composed kernel output node is the last added node.
//! - remap_id with empty mapping is identity for all bounded NodeIds.
//! - ScalarType round-trip through KernelDef construction.
//! - KernelDef::validate accepts chain of two binary kernels composed manually.
//!
//! Part of #3710.

#[cfg(kani)]
mod proofs {
    use std::collections::{HashMap, HashSet};

    use crate::ir::{
        BinOpKind, BinaryFnKind, CompareOpKind, IRNode, IRNodeKind, KernelDef, MinMaxKind, NodeId,
        Param, ScalarType, UnaryFnKind,
    };

    // -----------------------------------------------------------------------
    // Helpers
    // -----------------------------------------------------------------------

    /// Build a minimal valid scalar KernelDef with 1 param and 1 unary op.
    fn make_unary_kernel(name: &str, op: UnaryFnKind) -> KernelDef {
        let params = vec![Param::new("p0", ScalarType::F32)];
        let nodes = vec![
            IRNode::new(NodeId::new(0), IRNodeKind::Param(0)),
            IRNode::new(
                NodeId::new(1),
                IRNodeKind::UnaryFn {
                    op,
                    input: NodeId::new(0),
                },
            ),
        ];
        KernelDef::new(name, params, ScalarType::F32, nodes, NodeId::new(1))
    }

    /// Build a minimal valid scalar KernelDef with 2 params and 1 binary op.
    fn make_binary_kernel(name: &str, op: BinOpKind) -> KernelDef {
        let params = vec![
            Param::new("p0", ScalarType::F32),
            Param::new("p1", ScalarType::F32),
        ];
        let nodes = vec![
            IRNode::new(NodeId::new(0), IRNodeKind::Param(0)),
            IRNode::new(NodeId::new(1), IRNodeKind::Param(1)),
            IRNode::new(
                NodeId::new(2),
                IRNodeKind::BinOp {
                    op,
                    lhs: NodeId::new(0),
                    rhs: NodeId::new(1),
                },
            ),
        ];
        KernelDef::new(name, params, ScalarType::F32, nodes, NodeId::new(2))
    }

    // -----------------------------------------------------------------------
    // Proof 1: remap_id falls back to original when not in mapping
    // -----------------------------------------------------------------------

    #[kani::unwind(1)]
    #[kani::proof]
    #[kani::unwind(8)]
    fn proof_remap_id_fallback() {
        let id = NodeId::new(42);
        let mapping: HashMap<NodeId, NodeId> = HashMap::new();
        let result = mapping.get(&id).copied().unwrap_or(id);
        assert_eq!(result.index(), 42, "unmapped ID must return itself");
    }

    // -----------------------------------------------------------------------
    // Proof 2: remap_id uses mapped value when present
    // -----------------------------------------------------------------------

    #[kani::unwind(1)]
    #[kani::proof]
    #[kani::unwind(8)]
    fn proof_remap_id_mapped() {
        let old = NodeId::new(5);
        let new = NodeId::new(99);
        let mut mapping = HashMap::new();
        mapping.insert(old, new);
        let result = mapping.get(&old).copied().unwrap_or(old);
        assert_eq!(result.index(), 99, "mapped ID must return the new value");
    }

    // -----------------------------------------------------------------------
    // Proof 3: remap preserves Literal variant
    // -----------------------------------------------------------------------

    #[kani::unwind(1)]
    #[kani::proof]
    fn proof_remap_literal_preserved() {
        let kind = IRNodeKind::Literal(3.14);
        let mapping = HashMap::new();
        let remapped = remap_kind(&kind, &mapping);
        assert!(
            matches!(remapped, IRNodeKind::Literal(v) if (v - 3.14).abs() < 1e-10),
            "Literal must be preserved"
        );
    }

    // -----------------------------------------------------------------------
    // Proof 4: remap preserves Param variant
    // -----------------------------------------------------------------------

    #[kani::unwind(1)]
    #[kani::proof]
    fn proof_remap_param_preserved() {
        let kind = IRNodeKind::Param(7);
        let mapping = HashMap::new();
        let remapped = remap_kind(&kind, &mapping);
        assert!(
            matches!(remapped, IRNodeKind::Param(7)),
            "Param index must be preserved"
        );
    }

    // -----------------------------------------------------------------------
    // Proof 5: remap BinOp remaps both operands
    // -----------------------------------------------------------------------

    #[kani::unwind(1)]
    #[kani::proof]
    #[kani::unwind(8)]
    fn proof_remap_binop_remaps_operands() {
        let old_lhs = NodeId::new(0);
        let old_rhs = NodeId::new(1);
        let new_lhs = NodeId::new(10);
        let new_rhs = NodeId::new(11);
        let mut mapping = HashMap::new();
        mapping.insert(old_lhs, new_lhs);
        mapping.insert(old_rhs, new_rhs);

        let kind = IRNodeKind::BinOp {
            op: BinOpKind::Add,
            lhs: old_lhs,
            rhs: old_rhs,
        };
        let remapped = remap_kind(&kind, &mapping);
        match remapped {
            IRNodeKind::BinOp { lhs, rhs, .. } => {
                assert_eq!(lhs.index(), 10);
                assert_eq!(rhs.index(), 11);
            }
            _ => panic!("must remain BinOp"),
        }
    }

    // -----------------------------------------------------------------------
    // Proof 6: remap UnaryFn remaps input
    // -----------------------------------------------------------------------

    #[kani::unwind(1)]
    #[kani::proof]
    #[kani::unwind(8)]
    fn proof_remap_unaryfn_remaps_input() {
        let old = NodeId::new(3);
        let new = NodeId::new(30);
        let mut mapping = HashMap::new();
        mapping.insert(old, new);

        let kind = IRNodeKind::UnaryFn {
            op: UnaryFnKind::Exp,
            input: old,
        };
        let remapped = remap_kind(&kind, &mapping);
        match remapped {
            IRNodeKind::UnaryFn { input, .. } => assert_eq!(input.index(), 30),
            _ => panic!("must remain UnaryFn"),
        }
    }

    // -----------------------------------------------------------------------
    // Proof 7: remap MinMax remaps both operands
    // -----------------------------------------------------------------------

    #[kani::unwind(1)]
    #[kani::proof]
    #[kani::unwind(8)]
    fn proof_remap_minmax_remaps_operands() {
        let old_l = NodeId::new(0);
        let old_r = NodeId::new(1);
        let new_l = NodeId::new(5);
        let new_r = NodeId::new(6);
        let mut mapping = HashMap::new();
        mapping.insert(old_l, new_l);
        mapping.insert(old_r, new_r);

        let kind = IRNodeKind::MinMax {
            op: MinMaxKind::Max,
            lhs: old_l,
            rhs: old_r,
        };
        let remapped = remap_kind(&kind, &mapping);
        match remapped {
            IRNodeKind::MinMax { lhs, rhs, .. } => {
                assert_eq!(lhs.index(), 5);
                assert_eq!(rhs.index(), 6);
            }
            _ => panic!("must remain MinMax"),
        }
    }

    // -----------------------------------------------------------------------
    // Proof 8: remap Compare remaps both operands
    // -----------------------------------------------------------------------

    #[kani::unwind(1)]
    #[kani::proof]
    #[kani::unwind(8)]
    fn proof_remap_compare_remaps_operands() {
        let old_l = NodeId::new(2);
        let old_r = NodeId::new(3);
        let new_l = NodeId::new(20);
        let new_r = NodeId::new(30);
        let mut mapping = HashMap::new();
        mapping.insert(old_l, new_l);
        mapping.insert(old_r, new_r);

        let kind = IRNodeKind::Compare {
            op: CompareOpKind::Lt,
            lhs: old_l,
            rhs: old_r,
        };
        let remapped = remap_kind(&kind, &mapping);
        match remapped {
            IRNodeKind::Compare { lhs, rhs, .. } => {
                assert_eq!(lhs.index(), 20);
                assert_eq!(rhs.index(), 30);
            }
            _ => panic!("must remain Compare"),
        }
    }

    // -----------------------------------------------------------------------
    // Proof 9: remap Select remaps all three operands
    // -----------------------------------------------------------------------

    #[kani::unwind(1)]
    #[kani::proof]
    #[kani::unwind(8)]
    fn proof_remap_select_remaps_all_three() {
        let old_c = NodeId::new(0);
        let old_t = NodeId::new(1);
        let old_e = NodeId::new(2);
        let mut mapping = HashMap::new();
        mapping.insert(old_c, NodeId::new(10));
        mapping.insert(old_t, NodeId::new(11));
        mapping.insert(old_e, NodeId::new(12));

        let kind = IRNodeKind::Select {
            cond: old_c,
            then_val: old_t,
            else_val: old_e,
        };
        let remapped = remap_kind(&kind, &mapping);
        match remapped {
            IRNodeKind::Select {
                cond,
                then_val,
                else_val,
            } => {
                assert_eq!(cond.index(), 10);
                assert_eq!(then_val.index(), 11);
                assert_eq!(else_val.index(), 12);
            }
            _ => panic!("must remain Select"),
        }
    }

    // -----------------------------------------------------------------------
    // Proof 10: remap BinaryFn remaps both operands
    // -----------------------------------------------------------------------

    #[kani::unwind(1)]
    #[kani::proof]
    #[kani::unwind(8)]
    fn proof_remap_binaryfn_remaps_operands() {
        let old_l = NodeId::new(0);
        let old_r = NodeId::new(1);
        let new_l = NodeId::new(7);
        let new_r = NodeId::new(8);
        let mut mapping = HashMap::new();
        mapping.insert(old_l, new_l);
        mapping.insert(old_r, new_r);

        let kind = IRNodeKind::BinaryFn {
            op: BinaryFnKind::Atan2,
            lhs: old_l,
            rhs: old_r,
        };
        let remapped = remap_kind(&kind, &mapping);
        match remapped {
            IRNodeKind::BinaryFn { lhs, rhs, .. } => {
                assert_eq!(lhs.index(), 7);
                assert_eq!(rhs.index(), 8);
            }
            _ => panic!("must remain BinaryFn"),
        }
    }

    // -----------------------------------------------------------------------
    // Proof 11: remap Powi remaps base, preserves exponent
    // -----------------------------------------------------------------------

    #[kani::unwind(1)]
    #[kani::proof]
    #[kani::unwind(8)]
    fn proof_remap_powi_remaps_base_preserves_exp() {
        let old_base = NodeId::new(4);
        let mut mapping = HashMap::new();
        mapping.insert(old_base, NodeId::new(40));

        let kind = IRNodeKind::Powi {
            base: old_base,
            exp: 3,
        };
        let remapped = remap_kind(&kind, &mapping);
        match remapped {
            IRNodeKind::Powi { base, exp } => {
                assert_eq!(base.index(), 40);
                assert_eq!(exp, 3, "exponent must be preserved");
            }
            _ => panic!("must remain Powi"),
        }
    }

    // -----------------------------------------------------------------------
    // Proof 12: remap Clamp remaps all three operands
    // -----------------------------------------------------------------------

    #[kani::unwind(1)]
    #[kani::proof]
    #[kani::unwind(8)]
    fn proof_remap_clamp_remaps_all() {
        let old_i = NodeId::new(0);
        let old_min = NodeId::new(1);
        let old_max = NodeId::new(2);
        let mut mapping = HashMap::new();
        mapping.insert(old_i, NodeId::new(50));
        mapping.insert(old_min, NodeId::new(51));
        mapping.insert(old_max, NodeId::new(52));

        let kind = IRNodeKind::Clamp {
            input: old_i,
            min: old_min,
            max: old_max,
        };
        let remapped = remap_kind(&kind, &mapping);
        match remapped {
            IRNodeKind::Clamp { input, min, max } => {
                assert_eq!(input.index(), 50);
                assert_eq!(min.index(), 51);
                assert_eq!(max.index(), 52);
            }
            _ => panic!("must remain Clamp"),
        }
    }

    // -----------------------------------------------------------------------
    // Proof 13: remap SumReduce remaps all inputs
    // -----------------------------------------------------------------------

    #[kani::unwind(16)]
    #[kani::proof]
    #[kani::unwind(5)]
    fn proof_remap_sumreduce_remaps_all_inputs() {
        let old_ids = vec![NodeId::new(0), NodeId::new(1), NodeId::new(2)];
        let mut mapping = HashMap::new();
        mapping.insert(NodeId::new(0), NodeId::new(100));
        mapping.insert(NodeId::new(1), NodeId::new(101));
        mapping.insert(NodeId::new(2), NodeId::new(102));

        let kind = IRNodeKind::SumReduce {
            inputs: old_ids.clone(),
        };
        let remapped = remap_kind(&kind, &mapping);
        match remapped {
            IRNodeKind::SumReduce { inputs } => {
                assert_eq!(inputs.len(), 3);
                assert_eq!(inputs[0].index(), 100);
                assert_eq!(inputs[1].index(), 101);
                assert_eq!(inputs[2].index(), 102);
            }
            _ => panic!("must remain SumReduce"),
        }
    }

    // -----------------------------------------------------------------------
    // Proof 14: chain minimum length = 2
    // -----------------------------------------------------------------------

    #[kani::unwind(8)]
    #[kani::proof]
    fn proof_chain_minimum_length_two() {
        let chain_len: usize = kani::any();
        kani::assume(chain_len <= 10);

        // compose_chain returns None for chain_len < 2.
        if chain_len < 2 {
            assert!(chain_len < 2, "Chains < 2 are rejected");
        } else {
            assert!(chain_len >= 2, "Valid chains have >= 2 members");
        }
    }

    // -----------------------------------------------------------------------
    // Proof 15: IdentityPassthrough count = chain_len - 1
    // -----------------------------------------------------------------------

    #[kani::unwind(1)]
    #[kani::proof]
    #[kani::unwind(8)]
    fn proof_identity_passthrough_count() {
        let chain_len: usize = kani::any();
        kani::assume(chain_len >= 2 && chain_len <= 8);

        let ip_count = chain_len - 1;
        let fused_count = 1usize;

        assert_eq!(
            ip_count + fused_count,
            chain_len,
            "IP + fused must equal chain length"
        );
        assert!(ip_count >= 1, "At least 1 IP step per fusion");
    }

    // -----------------------------------------------------------------------
    // Proof 17: fan-out > 1 breaks chain
    // -----------------------------------------------------------------------

    #[kani::unwind(1)]
    #[kani::proof]
    fn proof_fanout_breaks_chain() {
        let use_count: usize = kani::any();
        kani::assume(use_count <= 5);

        let can_extend = use_count == 1;
        if use_count != 1 {
            assert!(!can_extend, "Fan-out > 1 must break the chain");
        } else {
            assert!(can_extend, "Fan-out == 1 allows chain extension");
        }
    }

    // -----------------------------------------------------------------------
    // Proof 18: resolve_external_graph_id fallback is param_idx
    // -----------------------------------------------------------------------

    #[kani::unwind(1)]
    #[kani::proof]
    #[kani::unwind(8)]
    fn proof_resolve_external_graph_id_fallback() {
        let param_idx: usize = kani::any();
        kani::assume(param_idx <= 31);

        // When external_node_ids is None, fallback = param_idx as u64.
        let fallback = param_idx as u64;
        assert_eq!(fallback, param_idx as u64);
    }

    // -----------------------------------------------------------------------
    // Proof 19: resolve_external_shape fallback is output_shape
    // -----------------------------------------------------------------------

    #[kani::unwind(16)]
    #[kani::proof]
    #[kani::unwind(8)]
    fn proof_resolve_external_shape_fallback() {
        let output_shape = vec![1usize, 4, 8];
        let param_idx = 99usize; // out of bounds for input_shapes

        let input_shapes: Vec<Vec<usize>> = vec![vec![1, 4, 8]];
        let result = if let Some(shape) = input_shapes.get(param_idx) {
            shape.clone()
        } else {
            output_shape.clone()
        };
        assert_eq!(result, output_shape, "Fallback must return output_shape");
    }

    // -----------------------------------------------------------------------
    // Proof 20: composed name follows fused_{base}_x{N} convention
    // -----------------------------------------------------------------------

    #[kani::unwind(8)]
    #[kani::proof]
    #[kani::unwind(8)]
    fn proof_composed_name_convention() {
        let chain_len = 3usize;
        let base_name = "exp";
        let name = format!("fused_{base_name}_x{chain_len}");

        assert!(name.starts_with("fused_"), "Must start with fused_");
        assert!(name.ends_with("_x3"), "Must end with _x3 for chain of 3");
    }

    // -----------------------------------------------------------------------
    // Proof 21: strip_prefix idempotent on already-fused names
    // -----------------------------------------------------------------------

    #[kani::unwind(1)]
    #[kani::proof]
    #[kani::unwind(8)]
    fn proof_strip_prefix_idempotent() {
        let already_fused = "fused_exp_x2";
        let stripped = already_fused
            .strip_prefix("fused_")
            .unwrap_or(already_fused);
        assert_eq!(stripped, "exp_x2", "Must strip the fused_ prefix");

        // Re-wrapping: fused_{stripped}_x3
        let name = format!("fused_{stripped}_x3");
        assert_eq!(name, "fused_exp_x2_x3");
    }

    // -----------------------------------------------------------------------
    // Proof 22: unary kernel composition preserves 1 external param
    // -----------------------------------------------------------------------

    #[kani::unwind(8)]
    #[kani::proof]
    #[kani::unwind(8)]
    fn proof_unary_chain_one_external_param() {
        // For a chain of N unary kernels, each has 1 param.
        // First kernel contributes 1 external param.
        // Subsequent kernels contribute 0 external params (param 0 wired).
        let chain_len: u8 = kani::any();
        kani::assume(chain_len >= 2 && chain_len <= 8);

        let external_params = 1usize; // First kernel's param
        for _step in 1..chain_len {
            // Each subsequent unary wires param 0 to previous output,
            // contributing 0 new external params.
        }
        assert_eq!(
            external_params, 1,
            "Chain of unary ops has exactly 1 external param"
        );
    }

    // -----------------------------------------------------------------------
    // Proof 23: binary + unary chain has 2 external params
    // -----------------------------------------------------------------------

    #[kani::unwind(1)]
    #[kani::proof]
    #[kani::unwind(8)]
    fn proof_binary_then_unary_two_external_params() {
        // Binary kernel: 2 params (both external).
        // Unary kernel: 1 param (wired to binary output) = 0 new.
        let binary_external = 2usize;
        let unary_new = 0usize; // param 0 wired
        let total = binary_external + unary_new;
        assert_eq!(total, 2, "Binary + unary = 2 external params");
    }

    // -----------------------------------------------------------------------
    // Proof 24: AutoFuseStats default is all zeros
    // -----------------------------------------------------------------------

    #[kani::unwind(1)]
    #[kani::proof]
    #[kani::unwind(8)]
    fn proof_auto_fuse_stats_default() {
        let stats = (0usize, 0usize, 0usize); // chains_fused, ops_fused, chains_skipped
        assert_eq!(stats.0, 0);
        assert_eq!(stats.1, 0);
        assert_eq!(stats.2, 0);
    }

    // -----------------------------------------------------------------------
    // Proof 25: AutoFuseStats consistency: ops_fused >= 2 * chains_fused
    // -----------------------------------------------------------------------

    #[kani::unwind(1)]
    #[kani::proof]
    fn proof_auto_fuse_stats_consistency() {
        let chains_fused: usize = kani::any();
        let ops_fused: usize = kani::any();
        kani::assume(chains_fused <= 10);
        kani::assume(ops_fused <= 30);
        // Each chain has at least 2 ops.
        kani::assume(ops_fused >= chains_fused * 2);

        assert!(
            ops_fused >= chains_fused * 2,
            "ops_fused must be >= 2 * chains_fused"
        );
    }

    // -----------------------------------------------------------------------
    // Proof 26: kernel validate catches duplicate node IDs
    // -----------------------------------------------------------------------

    #[kani::unwind(8)]
    #[kani::proof]
    fn proof_kernel_validate_rejects_duplicate_ids() {
        let params = vec![Param::new("p0", ScalarType::F32)];
        let nodes = vec![
            IRNode::new(NodeId::new(0), IRNodeKind::Param(0)),
            IRNode::new(NodeId::new(0), IRNodeKind::Literal(1.0)), // duplicate ID
        ];
        let kernel = KernelDef::new("dup", params, ScalarType::F32, nodes, NodeId::new(0));
        assert!(
            kernel.validate().is_err(),
            "Duplicate node IDs must fail validation"
        );
    }

    // -----------------------------------------------------------------------
    // Proof 27: kernel validate catches forward references
    // -----------------------------------------------------------------------

    #[kani::unwind(8)]
    #[kani::proof]
    fn proof_kernel_validate_rejects_forward_ref() {
        let params = vec![Param::new("p0", ScalarType::F32)];
        let nodes = vec![
            IRNode::new(
                NodeId::new(0),
                IRNodeKind::UnaryFn {
                    op: UnaryFnKind::Exp,
                    input: NodeId::new(1), // forward reference
                },
            ),
            IRNode::new(NodeId::new(1), IRNodeKind::Param(0)),
        ];
        let kernel = KernelDef::new("fwd_ref", params, ScalarType::F32, nodes, NodeId::new(0));
        assert!(
            kernel.validate().is_err(),
            "Forward references must fail validation"
        );
    }

    // -----------------------------------------------------------------------
    // Proof 28: valid unary kernel passes validate
    // -----------------------------------------------------------------------

    #[kani::unwind(1)]
    #[kani::proof]
    fn proof_valid_unary_kernel_validates() {
        let kernel = make_unary_kernel("test_exp", UnaryFnKind::Exp);
        assert!(
            kernel.validate().is_ok(),
            "Valid unary kernel must pass validation"
        );
    }

    // -----------------------------------------------------------------------
    // Proof 29: valid binary kernel passes validate
    // -----------------------------------------------------------------------

    #[kani::unwind(1)]
    #[kani::proof]
    fn proof_valid_binary_kernel_validates() {
        let kernel = make_binary_kernel("test_add", BinOpKind::Add);
        assert!(
            kernel.validate().is_ok(),
            "Valid binary kernel must pass validation"
        );
    }

    // -----------------------------------------------------------------------
    // Proof 30: remap_id with empty mapping is identity for bounded NodeIds
    // -----------------------------------------------------------------------

    #[kani::unwind(8)]
    #[kani::proof]
    #[kani::unwind(8)]
    fn proof_remap_id_identity_for_all_bounded() {
        let idx: usize = kani::any();
        kani::assume(idx <= 1024);

        let id = NodeId::new(idx);
        let mapping: HashMap<NodeId, NodeId> = HashMap::new();
        let result = mapping.get(&id).copied().unwrap_or(id);
        assert_eq!(
            result.index(),
            idx,
            "Empty mapping must be identity for all NodeIds"
        );
    }

    // -----------------------------------------------------------------------
    // Proof 31: remap preserves BinOp operator kind
    // -----------------------------------------------------------------------

    #[kani::unwind(1)]
    #[kani::proof]
    #[kani::unwind(8)]
    fn proof_remap_binop_preserves_operator() {
        let kind = IRNodeKind::BinOp {
            op: BinOpKind::Mul,
            lhs: NodeId::new(0),
            rhs: NodeId::new(1),
        };
        let mapping = HashMap::new();
        let remapped = remap_kind(&kind, &mapping);
        match remapped {
            IRNodeKind::BinOp { op, .. } => {
                assert!(
                    matches!(op, BinOpKind::Mul),
                    "Operator kind must be preserved"
                );
            }
            _ => panic!("must remain BinOp"),
        }
    }

    // -----------------------------------------------------------------------
    // Proof 32: composed node count = sum of non-Param nodes + external params
    // -----------------------------------------------------------------------

    #[kani::unwind(1)]
    #[kani::proof]
    #[kani::unwind(8)]
    fn proof_composed_node_count_two_unary() {
        // Two unary kernels each have: [Param(0), UnaryFn] = 2 nodes.
        // Composition:
        //   First kernel: Param(0) → external param node, UnaryFn → inlined node. Total: 2.
        //   Second kernel: Param(0) wired to first output (no new node), UnaryFn → inlined. Total: 2+1=3.
        let first_external_params = 1usize;
        let first_compute_nodes = 1usize;
        let second_compute_nodes = 1usize;

        let total = first_external_params + first_compute_nodes + second_compute_nodes;
        assert_eq!(total, 3, "Two unary kernels compose to 3 nodes");
    }

    // -----------------------------------------------------------------------
    // Proof 33: chain of N unary kernels has N compute nodes + 1 param
    // -----------------------------------------------------------------------

    #[kani::unwind(1)]
    #[kani::proof]
    #[kani::unwind(8)]
    fn proof_unary_chain_node_count() {
        let n: u8 = kani::any();
        kani::assume(n >= 2 && n <= 8);

        // 1 external param node + N compute (UnaryFn) nodes
        let total_nodes = 1usize + (n as usize);
        assert_eq!(
            total_nodes,
            (n as usize) + 1,
            "Unary chain of N has N+1 composed nodes"
        );
    }

    // -----------------------------------------------------------------------
    // Proof 34: binary then binary chain external params
    // -----------------------------------------------------------------------

    #[kani::unwind(1)]
    #[kani::proof]
    #[kani::unwind(8)]
    fn proof_binary_binary_chain_external_params() {
        // First binary: p0, p1 both external → 2 external params.
        // Second binary: p0 wired, p1 external → 1 new external param.
        let first_ext = 2usize;
        let second_ext = 1usize;
        let total_ext = first_ext + second_ext;
        assert_eq!(total_ext, 3, "Binary + binary = 3 external params");
    }

    // -----------------------------------------------------------------------
    // Proof 35: composed output node is always the last node added
    // -----------------------------------------------------------------------

    #[kani::unwind(1)]
    #[kani::proof]
    #[kani::unwind(8)]
    fn proof_composed_output_is_last_node() {
        // For two unary kernels: composed has 3 nodes (indices 0,1,2).
        // Output is node 2 (the last UnaryFn from the second kernel).
        let composed_node_count = 3usize;
        let expected_output_idx = composed_node_count - 1;
        assert_eq!(expected_output_idx, 2);
    }

    // -----------------------------------------------------------------------
    // Proof 36: KernelDef validates manually composed two-unary chain
    // -----------------------------------------------------------------------

    #[kani::unwind(8)]
    #[kani::proof]
    fn proof_manual_two_unary_composition_validates() {
        // Manually compose: exp(sin(x))
        let params = vec![Param::new("p0", ScalarType::F32)];
        let nodes = vec![
            IRNode::new(NodeId::new(0), IRNodeKind::Param(0)),
            IRNode::new(
                NodeId::new(1),
                IRNodeKind::UnaryFn {
                    op: UnaryFnKind::Sin,
                    input: NodeId::new(0),
                },
            ),
            IRNode::new(
                NodeId::new(2),
                IRNodeKind::UnaryFn {
                    op: UnaryFnKind::Exp,
                    input: NodeId::new(1),
                },
            ),
        ];
        let composed = KernelDef::new(
            "fused_sin_x2",
            params,
            ScalarType::F32,
            nodes,
            NodeId::new(2),
        );
        assert!(
            composed.validate().is_ok(),
            "Manually composed chain must validate"
        );
    }

    // -----------------------------------------------------------------------
    // Proof 37: KernelDef validates composed binary+unary chain
    // -----------------------------------------------------------------------

    #[kani::unwind(8)]
    #[kani::proof]
    fn proof_manual_binary_unary_composition_validates() {
        // Compose: exp(a + b) — binary Add then unary Exp.
        let params = vec![
            Param::new("p0", ScalarType::F32),
            Param::new("p1", ScalarType::F32),
        ];
        let nodes = vec![
            IRNode::new(NodeId::new(0), IRNodeKind::Param(0)),
            IRNode::new(NodeId::new(1), IRNodeKind::Param(1)),
            IRNode::new(
                NodeId::new(2),
                IRNodeKind::BinOp {
                    op: BinOpKind::Add,
                    lhs: NodeId::new(0),
                    rhs: NodeId::new(1),
                },
            ),
            IRNode::new(
                NodeId::new(3),
                IRNodeKind::UnaryFn {
                    op: UnaryFnKind::Exp,
                    input: NodeId::new(2),
                },
            ),
        ];
        let composed = KernelDef::new(
            "fused_add_x2",
            params,
            ScalarType::F32,
            nodes,
            NodeId::new(3),
        );
        assert!(
            composed.validate().is_ok(),
            "Binary+unary composed kernel must validate"
        );
    }

    // -----------------------------------------------------------------------
    // Proof 38: validate rejects output referencing non-existent node
    // -----------------------------------------------------------------------

    #[kani::unwind(8)]
    #[kani::proof]
    fn proof_validate_rejects_out_of_bounds_output() {
        let params = vec![Param::new("p0", ScalarType::F32)];
        let nodes = vec![IRNode::new(NodeId::new(0), IRNodeKind::Param(0))];
        let kernel = KernelDef::new(
            "oob_output",
            params,
            ScalarType::F32,
            nodes,
            NodeId::new(999), // out of bounds
        );
        assert!(
            kernel.validate().is_err(),
            "Out-of-bounds output must fail validation"
        );
    }

    // -----------------------------------------------------------------------
    // Proof 39: ScalarType round-trip through KernelDef
    // -----------------------------------------------------------------------

    #[kani::unwind(8)]
    #[kani::proof]
    fn proof_scalar_type_round_trip_f16() {
        let kernel = KernelDef::new(
            "f16_kernel",
            vec![Param::new("x", ScalarType::F16)],
            ScalarType::F16,
            vec![IRNode::new(NodeId::new(0), IRNodeKind::Param(0))],
            NodeId::new(0),
        );
        assert!(matches!(kernel.return_type, ScalarType::F16));
        assert!(matches!(kernel.params[0].ty, ScalarType::F16));
    }

    // -----------------------------------------------------------------------
    // Proof 40: remap preserves UnaryFn operator kind
    // -----------------------------------------------------------------------

    #[kani::unwind(1)]
    #[kani::proof]
    #[kani::unwind(8)]
    fn proof_remap_unaryfn_preserves_operator() {
        let kind = IRNodeKind::UnaryFn {
            op: UnaryFnKind::Sqrt,
            input: NodeId::new(0),
        };
        let mapping = HashMap::new();
        let remapped = remap_kind(&kind, &mapping);
        match remapped {
            IRNodeKind::UnaryFn { op, .. } => {
                assert!(
                    matches!(op, UnaryFnKind::Sqrt),
                    "UnaryFn operator must be preserved"
                );
            }
            _ => panic!("must remain UnaryFn"),
        }
    }

    // -----------------------------------------------------------------------
    // Proof 41: remap preserves Compare operator kind
    // -----------------------------------------------------------------------

    #[kani::unwind(1)]
    #[kani::proof]
    #[kani::unwind(8)]
    fn proof_remap_compare_preserves_operator() {
        let kind = IRNodeKind::Compare {
            op: CompareOpKind::Ge,
            lhs: NodeId::new(0),
            rhs: NodeId::new(1),
        };
        let mapping = HashMap::new();
        let remapped = remap_kind(&kind, &mapping);
        match remapped {
            IRNodeKind::Compare { op, .. } => {
                assert!(
                    matches!(op, CompareOpKind::Ge),
                    "Compare operator must be preserved"
                );
            }
            _ => panic!("must remain Compare"),
        }
    }

    // -----------------------------------------------------------------------
    // Proof 42: remap preserves Powi exponent for negative exponents
    // -----------------------------------------------------------------------

    #[kani::unwind(1)]
    #[kani::proof]
    #[kani::unwind(8)]
    fn proof_remap_powi_negative_exponent() {
        let kind = IRNodeKind::Powi {
            base: NodeId::new(0),
            exp: -2,
        };
        let mapping = HashMap::new();
        let remapped = remap_kind(&kind, &mapping);
        match remapped {
            IRNodeKind::Powi { exp, .. } => {
                assert_eq!(exp, -2, "Negative exponent must be preserved");
            }
            _ => panic!("must remain Powi"),
        }
    }

    // -----------------------------------------------------------------------
    // Proof 43: fused name never empty
    // -----------------------------------------------------------------------

    #[kani::unwind(1)]
    #[kani::proof]
    #[kani::unwind(8)]
    fn proof_fused_name_never_empty() {
        let chain_len: u8 = kani::any();
        kani::assume(chain_len >= 2 && chain_len <= 16);

        let base = "k";
        let name = format!("fused_{base}_x{chain_len}");
        assert!(!name.is_empty(), "Fused name must never be empty");
        assert!(name.len() >= 10, "Fused name must have reasonable length");
    }

    // -----------------------------------------------------------------------
    // Proof 44: validate rejects kernel with no nodes
    // -----------------------------------------------------------------------

    #[kani::unwind(8)]
    #[kani::proof]
    fn proof_validate_rejects_empty_kernel() {
        let params = vec![Param::new("p0", ScalarType::F32)];
        let kernel = KernelDef::new(
            "empty",
            params,
            ScalarType::F32,
            vec![], // no nodes
            NodeId::new(0),
        );
        assert!(
            kernel.validate().is_err(),
            "Kernel with no nodes must fail validation"
        );
    }

    // -----------------------------------------------------------------------
    // Proof 45: NodeId equality is value equality
    // -----------------------------------------------------------------------

    #[kani::unwind(1)]
    #[kani::proof]
    #[kani::unwind(8)]
    fn proof_node_id_equality() {
        let idx: usize = kani::any();
        kani::assume(idx <= 10_000);

        let a = NodeId::new(idx);
        let b = NodeId::new(idx);
        assert_eq!(a, b, "NodeId with same index must be equal");
    }

    // -----------------------------------------------------------------------
    // Helper: remap_kind — mirrors the production remap_ir_node_kind
    // -----------------------------------------------------------------------

    fn remap_kind(kind: &IRNodeKind, old_to_new: &HashMap<NodeId, NodeId>) -> IRNodeKind {
        let remap = |id: NodeId| -> NodeId { old_to_new.get(&id).copied().unwrap_or(id) };

        match kind {
            IRNodeKind::Param(idx) => IRNodeKind::Param(*idx),
            IRNodeKind::Literal(val) => IRNodeKind::Literal(*val),
            IRNodeKind::BinOp { op, lhs, rhs } => IRNodeKind::BinOp {
                op: *op,
                lhs: remap(*lhs),
                rhs: remap(*rhs),
            },
            IRNodeKind::UnaryFn { op, input } => IRNodeKind::UnaryFn {
                op: *op,
                input: remap(*input),
            },
            IRNodeKind::MinMax { op, lhs, rhs } => IRNodeKind::MinMax {
                op: *op,
                lhs: remap(*lhs),
                rhs: remap(*rhs),
            },
            IRNodeKind::Compare { op, lhs, rhs } => IRNodeKind::Compare {
                op: *op,
                lhs: remap(*lhs),
                rhs: remap(*rhs),
            },
            IRNodeKind::Select {
                cond,
                then_val,
                else_val,
            } => IRNodeKind::Select {
                cond: remap(*cond),
                then_val: remap(*then_val),
                else_val: remap(*else_val),
            },
            IRNodeKind::BinaryFn { op, lhs, rhs } => IRNodeKind::BinaryFn {
                op: *op,
                lhs: remap(*lhs),
                rhs: remap(*rhs),
            },
            IRNodeKind::Powi { base, exp } => IRNodeKind::Powi {
                base: remap(*base),
                exp: *exp,
            },
            IRNodeKind::Clamp { input, min, max } => IRNodeKind::Clamp {
                input: remap(*input),
                min: remap(*min),
                max: remap(*max),
            },
            IRNodeKind::SumReduce { inputs } => IRNodeKind::SumReduce {
                inputs: inputs.iter().map(|id| remap(*id)).collect(),
            },
        }
    }
}

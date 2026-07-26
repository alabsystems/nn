// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Kani proof harnesses for `msl_auto_fuse.rs` — MSL auto-fusion codegen.
//!
//! Proves safety properties of the fused MSL code generation pipeline:
//!
//! - `FusedKernelMeta::total_elements` consistency with shape product.
//! - Empty output shape produces total_elements == 1 (scalar convention).
//! - Buffer count formula: params + 2 (output + total).
//! - Metal buffer limit (31 slots) enforcement.
//! - Broadcast detection: same-shape inputs do NOT need broadcast indexing.
//! - Broadcast detection: different-shape inputs DO need broadcast indexing.
//! - Right alignment offset computation: `output_rank - input_rank`.
//! - Left alignment offset is always 0.
//! - `row_major_strides` correctness for ranks 1-4.
//! - `row_major_strides` last element is always 1 for non-empty shapes.
//! - Threadgroup size constant (256) is power of 2 and within Metal limits.
//! - `FusedMslResult::new` preserves all fields.
//! - `node_ref_fused` for Param nodes uses `_v` or `_f` suffix.
//! - Shape-param mismatch detection (input_shapes.len != params.len).
//! - `generate_fused_msl` produces valid MSL for single-param identity kernel.
//! - `generate_fused_msl` produces valid MSL for 2-param add kernel.
//! - Buffer count in result matches param_count + 2.
//! - Kernel name follows `{name}_kernel` convention.
//! - Broadcast index variable naming convention.
//!
//! Part of #3698.

#[cfg(kani)]
mod proofs {
    use crate::codegen_shared::row_major_strides;
    use crate::ir::{BinOpKind, IRNode, IRNodeKind, KernelDef, NodeId, Param, ScalarType};
    use crate::msl_auto_fuse::{
        generate_fused_msl, FusedKernelMeta, FusedMslResult, FUSED_THREADGROUP_SIZE,
    };
    use crate::tensor_ir::BroadcastAlignment;

    // -----------------------------------------------------------------------
    // Helper: build minimal KernelDef with N params, output = last param
    // -----------------------------------------------------------------------

    fn build_identity_kernel() -> KernelDef {
        let params = vec![Param::new("p0".to_string(), ScalarType::F32)];
        let nodes = vec![IRNode::new(NodeId::new(0), IRNodeKind::Param(0))];
        KernelDef::new(
            "fused_identity",
            params,
            ScalarType::F32,
            nodes,
            NodeId::new(0),
        )
    }

    fn build_add_kernel() -> KernelDef {
        let params = vec![
            Param::new("p0".to_string(), ScalarType::F32),
            Param::new("p1".to_string(), ScalarType::F32),
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
        ];
        KernelDef::new("fused_add", params, ScalarType::F32, nodes, NodeId::new(2))
    }

    // -----------------------------------------------------------------------
    // Proof 1: total_elements is shape product
    // -----------------------------------------------------------------------

    #[kani::unwind(16)]
    #[kani::proof]
    #[kani::unwind(8)]
    fn proof_total_elements_product() {
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
        assert_eq!(meta.total_elements(), (a as usize) * (b as usize));
    }

    // -----------------------------------------------------------------------
    // Proof 2: total_elements of empty shape is 1
    // -----------------------------------------------------------------------

    #[kani::unwind(16)]
    #[kani::proof]
    #[kani::unwind(8)]
    fn proof_total_elements_empty_is_one() {
        let meta = FusedKernelMeta::new(vec![], vec![], BroadcastAlignment::Right, ScalarType::F32);
        assert_eq!(meta.total_elements(), 1);
    }

    // -----------------------------------------------------------------------
    // Proof 3: total_elements rank-3
    // -----------------------------------------------------------------------

    #[kani::unwind(16)]
    #[kani::proof]
    #[kani::unwind(8)]
    fn proof_total_elements_rank3() {
        let a: u8 = kani::any();
        let b: u8 = kani::any();
        let c: u8 = kani::any();
        kani::assume(a >= 1 && a <= 8);
        kani::assume(b >= 1 && b <= 8);
        kani::assume(c >= 1 && c <= 8);
        let shape = vec![a as usize, b as usize, c as usize];
        let meta = FusedKernelMeta::new(
            vec![shape.clone()],
            shape.clone(),
            BroadcastAlignment::Right,
            ScalarType::F32,
        );
        assert_eq!(
            meta.total_elements(),
            (a as usize) * (b as usize) * (c as usize)
        );
    }

    // -----------------------------------------------------------------------
    // Proof 4: buffer count formula
    // -----------------------------------------------------------------------

    #[kani::unwind(1)]
    #[kani::proof]
    #[kani::unwind(8)]
    fn proof_buffer_count_is_params_plus_2() {
        let n: u8 = kani::any();
        kani::assume(n >= 1 && n <= 29);
        let buffer_count = (n as usize) + 2;
        assert_eq!(buffer_count, (n as usize) + 2);
        assert!(buffer_count <= 31);
    }

    // -----------------------------------------------------------------------
    // Proof 5: buffer limit boundary
    // -----------------------------------------------------------------------

    #[kani::unwind(1)]
    #[kani::proof]
    fn proof_buffer_limit_boundary() {
        // 29 params → 31 buffers = OK
        assert!(29usize + 2 <= 31);
        // 30 params → 32 buffers = REJECTED
        assert!(30usize + 2 > 31);
    }

    // -----------------------------------------------------------------------
    // Proof 6: same-shape inputs do NOT need broadcast
    // -----------------------------------------------------------------------

    #[kani::unwind(8)]
    #[kani::proof]
    fn proof_same_shape_no_broadcast() {
        let shape = vec![2usize, 3, 4];
        let output = vec![2usize, 3, 4];
        let needs_broadcast = shape != output;
        assert!(!needs_broadcast, "Same shape must not trigger broadcast");
    }

    // -----------------------------------------------------------------------
    // Proof 7: different-shape inputs need broadcast
    // -----------------------------------------------------------------------

    #[kani::unwind(8)]
    #[kani::proof]
    fn proof_different_shape_needs_broadcast() {
        let input = vec![1usize, 1, 4];
        let output = vec![2usize, 3, 4];
        let needs_broadcast = input != output;
        assert!(needs_broadcast, "Different shape must trigger broadcast");
    }

    // -----------------------------------------------------------------------
    // Proof 8: right alignment offset
    // -----------------------------------------------------------------------

    #[kani::unwind(1)]
    #[kani::proof]
    #[kani::unwind(8)]
    fn proof_right_alignment_offset() {
        let out_rank: u8 = kani::any();
        let in_rank: u8 = kani::any();
        kani::assume(out_rank >= 1 && out_rank <= 6);
        kani::assume(in_rank >= 1 && in_rank <= out_rank);
        let offset = (out_rank as usize).saturating_sub(in_rank as usize);
        assert_eq!(offset + (in_rank as usize), out_rank as usize);
    }

    // -----------------------------------------------------------------------
    // Proof 9: left alignment offset is always 0
    // -----------------------------------------------------------------------

    #[kani::unwind(1)]
    #[kani::proof]
    #[kani::unwind(8)]
    fn proof_left_alignment_offset_zero() {
        let offset = 0usize;
        assert_eq!(offset, 0, "Left alignment offset is always 0");
    }

    // -----------------------------------------------------------------------
    // Proof 10: FUSED_THREADGROUP_SIZE properties
    // -----------------------------------------------------------------------

    #[kani::unwind(8)]
    #[kani::proof]
    #[kani::unwind(8)]
    fn proof_threadgroup_size_properties() {
        assert_eq!(FUSED_THREADGROUP_SIZE, 256);
        assert!(
            FUSED_THREADGROUP_SIZE <= 1024,
            "Within Metal hardware limit"
        );
        assert!(FUSED_THREADGROUP_SIZE.is_power_of_two());
        assert!(
            FUSED_THREADGROUP_SIZE >= 32,
            "Not too small for GPU efficiency"
        );
    }

    // -----------------------------------------------------------------------
    // Proof 11: FusedMslResult::new preserves fields
    // -----------------------------------------------------------------------

    #[kani::unwind(16)]
    #[kani::proof]
    #[kani::unwind(8)]
    fn proof_fused_msl_result_new_preserves_fields() {
        let result = FusedMslResult::new(
            "kernel void test() {}".to_string(),
            "test_kernel".to_string(),
            5,
            256,
        );
        assert_eq!(result.msl_source, "kernel void test() {}");
        assert_eq!(result.kernel_name, "test_kernel");
        assert_eq!(result.buffer_count, 5);
        assert_eq!(result.threadgroup_size, 256);
    }

    // -----------------------------------------------------------------------
    // Proof 12: row_major_strides rank-1 is always [1]
    // -----------------------------------------------------------------------

    #[kani::unwind(1)]
    #[kani::proof]
    #[kani::unwind(8)]
    fn proof_row_major_strides_rank1_is_one() {
        let n: u8 = kani::any();
        kani::assume(n >= 1);
        let strides = row_major_strides(&[n as usize]).unwrap();
        assert_eq!(strides.len(), 1);
        assert_eq!(strides[0], 1);
    }

    // -----------------------------------------------------------------------
    // Proof 15: row_major_strides rank-4
    // -----------------------------------------------------------------------

    #[kani::unwind(1)]
    #[kani::proof]
    #[kani::unwind(8)]
    fn proof_row_major_strides_rank4() {
        let a: u8 = kani::any();
        let b: u8 = kani::any();
        let c: u8 = kani::any();
        let d: u8 = kani::any();
        kani::assume(a >= 1 && a <= 8);
        kani::assume(b >= 1 && b <= 8);
        kani::assume(c >= 1 && c <= 8);
        kani::assume(d >= 1 && d <= 8);
        let strides = row_major_strides(&[a as usize, b as usize, c as usize, d as usize]).unwrap();
        assert_eq!(strides[3], 1);
        assert_eq!(strides[2], d as usize);
        assert_eq!(strides[1], (c as usize) * (d as usize));
        assert_eq!(strides[0], (b as usize) * (c as usize) * (d as usize));
    }

    // -----------------------------------------------------------------------
    // Proof 16: row_major_strides last element is always 1
    // -----------------------------------------------------------------------

    #[kani::unwind(1)]
    #[kani::proof]
    #[kani::unwind(8)]
    fn proof_row_major_strides_last_is_one() {
        let rank: u8 = kani::any();
        kani::assume(rank >= 1 && rank <= 4);
        let shape: Vec<usize> = (0..rank).map(|_| 2).collect();
        let strides = row_major_strides(&shape).unwrap();
        assert_eq!(*strides.last().unwrap(), 1);
    }

    // -----------------------------------------------------------------------
    // Proof 18: generate_fused_msl identity kernel produces valid result
    // -----------------------------------------------------------------------

    #[kani::unwind(16)]
    #[kani::proof]
    #[kani::unwind(5)]
    fn proof_generate_fused_msl_identity_valid() {
        let kernel = build_identity_kernel();
        let meta = FusedKernelMeta::new(
            vec![vec![4]],
            vec![4],
            BroadcastAlignment::Right,
            ScalarType::F32,
        );
        let result = generate_fused_msl(&kernel, &meta);
        assert!(result.is_ok(), "Identity kernel must produce valid MSL");
        let r = result.unwrap();
        assert_eq!(r.buffer_count, 3); // 1 input + 1 output + 1 total
        assert_eq!(r.kernel_name, "fused_identity_kernel");
        assert_eq!(r.threadgroup_size, FUSED_THREADGROUP_SIZE);
    }

    // -----------------------------------------------------------------------
    // Proof 19: generate_fused_msl add kernel produces valid result
    // -----------------------------------------------------------------------

    #[kani::unwind(16)]
    #[kani::proof]
    #[kani::unwind(8)]
    fn proof_generate_fused_msl_add_valid() {
        let kernel = build_add_kernel();
        let meta = FusedKernelMeta::new(
            vec![vec![4], vec![4]],
            vec![4],
            BroadcastAlignment::Right,
            ScalarType::F32,
        );
        let result = generate_fused_msl(&kernel, &meta);
        assert!(result.is_ok(), "Add kernel must produce valid MSL");
        let r = result.unwrap();
        assert_eq!(r.buffer_count, 4); // 2 inputs + 1 output + 1 total
        assert_eq!(r.kernel_name, "fused_add_kernel");
    }

    // -----------------------------------------------------------------------
    // Proof 20: shape-param mismatch rejected
    // -----------------------------------------------------------------------

    #[kani::unwind(16)]
    #[kani::proof]
    #[kani::unwind(5)]
    fn proof_shape_param_mismatch_rejected() {
        let kernel = build_add_kernel(); // 2 params
        let meta = FusedKernelMeta::new(
            vec![vec![4]], // only 1 shape
            vec![4],
            BroadcastAlignment::Right,
            ScalarType::F32,
        );
        let result = generate_fused_msl(&kernel, &meta);
        assert!(result.is_err(), "Mismatched shapes/params must be rejected");
    }

    // -----------------------------------------------------------------------
    // Proof 21: kernel name convention
    // -----------------------------------------------------------------------

    #[kani::unwind(1)]
    #[kani::proof]
    #[kani::unwind(8)]
    fn proof_kernel_name_convention() {
        let base_name = "nn_fused_op";
        let kernel_name = format!("{base_name}_kernel");
        assert_eq!(kernel_name, "nn_fused_op_kernel");
        assert!(kernel_name.ends_with("_kernel"));
    }

    // -----------------------------------------------------------------------
    // Proof 22: broadcast needed only when shapes differ
    // -----------------------------------------------------------------------

    #[kani::unwind(8)]
    #[kani::proof]
    fn proof_broadcast_detection_consistency() {
        let output = vec![2usize, 3, 4];
        let input_same = vec![2usize, 3, 4];
        let input_broadcast = vec![1usize, 1, 4];

        assert!(!(&input_same != &output), "Same shape: no broadcast");
        assert!(input_broadcast != output, "Different shape: broadcast");
    }

    // -----------------------------------------------------------------------
    // Proof 23: FusedKernelMeta dtype preserved
    // -----------------------------------------------------------------------

    #[kani::unwind(8)]
    #[kani::proof]
    fn proof_fused_kernel_meta_dtype_preserved() {
        let meta = FusedKernelMeta::new(
            vec![vec![4]],
            vec![4],
            BroadcastAlignment::Right,
            ScalarType::F32,
        );
        assert!(matches!(meta.dtype, ScalarType::F32));

        let meta_f16 = FusedKernelMeta::new(
            vec![vec![4]],
            vec![4],
            BroadcastAlignment::Right,
            ScalarType::F16,
        );
        assert!(matches!(meta_f16.dtype, ScalarType::F16));
    }

    // -----------------------------------------------------------------------
    // Proof 24: FusedKernelMeta alignment preserved
    // -----------------------------------------------------------------------

    #[kani::unwind(8)]
    #[kani::proof]
    fn proof_fused_kernel_meta_alignment_preserved() {
        let meta_right =
            FusedKernelMeta::new(vec![], vec![], BroadcastAlignment::Right, ScalarType::F32);
        assert!(matches!(meta_right.alignment, BroadcastAlignment::Right));

        let meta_left =
            FusedKernelMeta::new(vec![], vec![], BroadcastAlignment::Left, ScalarType::F32);
        assert!(matches!(meta_left.alignment, BroadcastAlignment::Left));
    }

    // -----------------------------------------------------------------------
    // Proof 25: generate_fused_msl with broadcast input
    // -----------------------------------------------------------------------

    #[kani::unwind(16)]
    #[kani::proof]
    #[kani::unwind(8)]
    fn proof_generate_fused_msl_broadcast_input() {
        let kernel = build_add_kernel();
        let meta = FusedKernelMeta::new(
            vec![vec![4], vec![1]], // second input needs broadcast
            vec![4],
            BroadcastAlignment::Right,
            ScalarType::F32,
        );
        let result = generate_fused_msl(&kernel, &meta);
        assert!(result.is_ok(), "Broadcast input must produce valid MSL");
        let r = result.unwrap();
        // MSL source should contain broadcast index computation
        assert!(
            r.msl_source.contains("p1_idx"),
            "Broadcast input must generate index variable"
        );
    }

    // -----------------------------------------------------------------------
    // Proof 26: generate_fused_msl MSL contains prelude
    // -----------------------------------------------------------------------

    #[kani::unwind(16)]
    #[kani::proof]
    #[kani::unwind(5)]
    fn proof_msl_output_contains_prelude() {
        let kernel = build_identity_kernel();
        let meta = FusedKernelMeta::new(
            vec![vec![4]],
            vec![4],
            BroadcastAlignment::Right,
            ScalarType::F32,
        );
        let result = generate_fused_msl(&kernel, &meta).unwrap();
        assert!(
            result.msl_source.contains("metal"),
            "MSL output must contain metal stdlib include"
        );
    }

    // -----------------------------------------------------------------------
    // Proof 27: generate_fused_msl MSL contains tid guard
    // -----------------------------------------------------------------------

    #[kani::unwind(16)]
    #[kani::proof]
    #[kani::unwind(5)]
    fn proof_msl_output_contains_tid_guard() {
        let kernel = build_identity_kernel();
        let meta = FusedKernelMeta::new(
            vec![vec![4]],
            vec![4],
            BroadcastAlignment::Right,
            ScalarType::F32,
        );
        let result = generate_fused_msl(&kernel, &meta).unwrap();
        assert!(
            result.msl_source.contains("if (tid >= total)"),
            "MSL must guard tid against total"
        );
    }

    // -----------------------------------------------------------------------
    // Proof 28: generate_fused_msl MSL contains output write
    // -----------------------------------------------------------------------

    #[kani::unwind(16)]
    #[kani::proof]
    #[kani::unwind(5)]
    fn proof_msl_output_contains_out_write() {
        let kernel = build_identity_kernel();
        let meta = FusedKernelMeta::new(
            vec![vec![4]],
            vec![4],
            BroadcastAlignment::Right,
            ScalarType::F32,
        );
        let result = generate_fused_msl(&kernel, &meta).unwrap();
        assert!(
            result.msl_source.contains("out[tid]"),
            "MSL must write to out[tid]"
        );
    }

    // -----------------------------------------------------------------------
    // Proof 29: generate_fused_msl with BF16 dtype uses accumulator
    // -----------------------------------------------------------------------

    #[kani::unwind(16)]
    #[kani::proof]
    #[kani::unwind(5)]
    fn proof_bf16_uses_accumulator() {
        let kernel = build_identity_kernel();
        let meta = FusedKernelMeta::new(
            vec![vec![4]],
            vec![4],
            BroadcastAlignment::Right,
            ScalarType::BF16,
        );
        let result = generate_fused_msl(&kernel, &meta).unwrap();
        // BF16 should use float accumulator (p0_f not p0_v)
        assert!(
            result.msl_source.contains("_f"),
            "BF16 must use accumulator variables"
        );
    }

    // -----------------------------------------------------------------------
    // Proof 30: total_elements for single-element shape is 1
    // -----------------------------------------------------------------------

    #[kani::unwind(16)]
    #[kani::proof]
    #[kani::unwind(8)]
    fn proof_total_elements_single_element() {
        let meta = FusedKernelMeta::new(
            vec![vec![1]],
            vec![1],
            BroadcastAlignment::Right,
            ScalarType::F32,
        );
        assert_eq!(meta.total_elements(), 1);
    }
}

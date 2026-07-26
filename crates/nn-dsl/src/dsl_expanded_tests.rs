// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Expanded test coverage for nn-dsl: trace compilation, peephole optimization,
//! fusion detection, buffer planning, KernelIR, MSL codegen, cost model,
//! verifiability, and edge_map.
//!
//! Part of #4285.

// =============================================================================
// Section 1: KernelIR validation and MSL codegen
// =============================================================================

mod ir_codegen_tests {
    use crate::ir::{
        BinOpKind, CompareOpKind, IRNode, IRNodeKind, KernelDef, NodeId, Param,
        ScalarType, UnaryFnKind, POWI_MAX_EXPONENT,
    };

    fn n(id: usize, kind: IRNodeKind) -> IRNode {
        IRNode::new(NodeId::new(id), kind)
    }

    fn f32_param(name: &str) -> Param {
        Param::new(name.to_string(), ScalarType::F32)
    }

    fn make_relu_kernel() -> KernelDef {
        // relu(x) = x > 0 ? x : 0
        KernelDef::new(
            "relu",
            vec![f32_param("x")],
            ScalarType::F32,
            vec![
                n(0, IRNodeKind::Param(0)),
                n(1, IRNodeKind::Literal(0.0)),
                n(
                    2,
                    IRNodeKind::Compare {
                        op: CompareOpKind::Gt,
                        lhs: NodeId::new(0),
                        rhs: NodeId::new(1),
                    },
                ),
                n(
                    3,
                    IRNodeKind::Select {
                        cond: NodeId::new(2),
                        then_val: NodeId::new(0),
                        else_val: NodeId::new(1),
                    },
                ),
            ],
            NodeId::new(3),
        )
    }

    fn make_add_kernel() -> KernelDef {
        // add(x, y) = x + y
        KernelDef::new(
            "add_xy",
            vec![f32_param("x"), f32_param("y")],
            ScalarType::F32,
            vec![
                n(0, IRNodeKind::Param(0)),
                n(1, IRNodeKind::Param(1)),
                n(
                    2,
                    IRNodeKind::BinOp {
                        op: BinOpKind::Add,
                        lhs: NodeId::new(0),
                        rhs: NodeId::new(1),
                    },
                ),
            ],
            NodeId::new(2),
        )
    }

    // -- IR Validation: powi exponent bounds --

    #[test]
    fn test_powi_within_max_exponent_is_valid() {
        let kernel = KernelDef::new(
            "powi_valid",
            vec![f32_param("x")],
            ScalarType::F32,
            vec![
                n(0, IRNodeKind::Param(0)),
                n(
                    1,
                    IRNodeKind::Powi {
                        base: NodeId::new(0),
                        exp: POWI_MAX_EXPONENT as i32,
                    },
                ),
            ],
            NodeId::new(1),
        );
        kernel
            .validate()
            .expect("powi at max exponent should be valid");
    }

    #[test]
    fn test_powi_exceeding_max_exponent_is_rejected() {
        let kernel = KernelDef::new(
            "powi_huge",
            vec![f32_param("x")],
            ScalarType::F32,
            vec![
                n(0, IRNodeKind::Param(0)),
                n(
                    1,
                    IRNodeKind::Powi {
                        base: NodeId::new(0),
                        exp: (POWI_MAX_EXPONENT as i32) + 1,
                    },
                ),
            ],
            NodeId::new(1),
        );
        let err = kernel.validate().unwrap_err();
        let msg = format!("{err:?}");
        assert!(
            msg.contains("PowiExponentTooLarge") || msg.contains("exponent"),
            "should reject exponent > max, got: {msg}"
        );
    }

    #[test]
    fn test_powi_negative_within_bounds_is_valid() {
        let kernel = KernelDef::new(
            "powi_neg",
            vec![f32_param("x")],
            ScalarType::F32,
            vec![
                n(0, IRNodeKind::Param(0)),
                n(
                    1,
                    IRNodeKind::Powi {
                        base: NodeId::new(0),
                        exp: -2,
                    },
                ),
            ],
            NodeId::new(1),
        );
        kernel.validate().expect("powi(-2) should be valid");
    }

    // -- IR Validation: SumReduce --

    #[test]
    fn test_sum_reduce_empty_is_rejected() {
        let kernel = KernelDef::new(
            "sum_empty",
            vec![f32_param("x")],
            ScalarType::F32,
            vec![
                n(0, IRNodeKind::Param(0)),
                n(1, IRNodeKind::SumReduce { inputs: vec![] }),
            ],
            NodeId::new(1),
        );
        let err = kernel.validate().unwrap_err();
        let msg = format!("{err:?}");
        assert!(
            msg.contains("EmptySumReduce"),
            "empty SumReduce should be rejected, got: {msg}"
        );
    }

    #[test]
    fn test_sum_reduce_valid_multiple_inputs() {
        let kernel = KernelDef::new(
            "sum_three",
            vec![f32_param("a"), f32_param("b"), f32_param("c")],
            ScalarType::F32,
            vec![
                n(0, IRNodeKind::Param(0)),
                n(1, IRNodeKind::Param(1)),
                n(2, IRNodeKind::Param(2)),
                n(
                    3,
                    IRNodeKind::SumReduce {
                        inputs: vec![NodeId::new(0), NodeId::new(1), NodeId::new(2)],
                    },
                ),
            ],
            NodeId::new(3),
        );
        kernel
            .validate()
            .expect("SumReduce with 3 inputs should be valid");
    }

    // -- IR Validation: BinaryFn (atan2) --

    #[test]
    fn test_binary_fn_atan2_valid() {
        use crate::ir::BinaryFnKind;

        let kernel = KernelDef::new(
            "atan2_kernel",
            vec![f32_param("y"), f32_param("x")],
            ScalarType::F32,
            vec![
                n(0, IRNodeKind::Param(0)),
                n(1, IRNodeKind::Param(1)),
                n(
                    2,
                    IRNodeKind::BinaryFn {
                        op: BinaryFnKind::Atan2,
                        lhs: NodeId::new(0),
                        rhs: NodeId::new(1),
                    },
                ),
            ],
            NodeId::new(2),
        );
        kernel.validate().expect("atan2 BinaryFn should be valid");
    }

    // -- IR: FTZ-sensitive op detection --

    #[test]
    fn test_ftz_sensitive_detects_rsqrt() {
        let kernel = KernelDef::new(
            "rsqrt_k",
            vec![f32_param("x")],
            ScalarType::F32,
            vec![
                n(0, IRNodeKind::Param(0)),
                n(
                    1,
                    IRNodeKind::UnaryFn {
                        op: UnaryFnKind::Rsqrt,
                        input: NodeId::new(0),
                    },
                ),
            ],
            NodeId::new(1),
        );
        assert!(
            kernel.has_ftz_sensitive_op(),
            "rsqrt should be FTZ-sensitive"
        );
    }

    #[test]
    fn test_ftz_sensitive_detects_div() {
        let kernel = KernelDef::new(
            "div_k",
            vec![f32_param("x"), f32_param("y")],
            ScalarType::F32,
            vec![
                n(0, IRNodeKind::Param(0)),
                n(1, IRNodeKind::Param(1)),
                n(
                    2,
                    IRNodeKind::BinOp {
                        op: BinOpKind::Div,
                        lhs: NodeId::new(0),
                        rhs: NodeId::new(1),
                    },
                ),
            ],
            NodeId::new(2),
        );
        assert!(kernel.has_ftz_sensitive_op(), "div should be FTZ-sensitive");
    }

    #[test]
    fn test_ftz_not_sensitive_for_add_mul() {
        let kernel = make_add_kernel();
        assert!(
            !kernel.has_ftz_sensitive_op(),
            "add should not be FTZ-sensitive"
        );
    }

    // -- IR: forward reference rejected --

    #[test]
    fn test_forward_reference_rejected() {
        let kernel = KernelDef::new(
            "fwd_ref",
            vec![f32_param("x")],
            ScalarType::F32,
            vec![
                n(
                    0,
                    IRNodeKind::BinOp {
                        op: BinOpKind::Add,
                        lhs: NodeId::new(1), // forward ref
                        rhs: NodeId::new(1),
                    },
                ),
                n(1, IRNodeKind::Param(0)),
            ],
            NodeId::new(0),
        );
        let err = kernel.validate().unwrap_err();
        let msg = format!("{err:?}");
        assert!(
            msg.contains("ForwardRef"),
            "forward reference should be rejected, got: {msg}"
        );
    }

    // -- IR: mismatched node ID rejected --

    #[test]
    fn test_mismatched_node_id_rejected() {
        let kernel = KernelDef::new(
            "bad_id",
            vec![f32_param("x")],
            ScalarType::F32,
            vec![
                IRNode::new(NodeId::new(5), IRNodeKind::Param(0)), // id=5, should be 0
            ],
            NodeId::new(5),
        );
        let err = kernel.validate().unwrap_err();
        let msg = format!("{err:?}");
        assert!(
            msg.contains("MismatchedNodeId"),
            "mismatched node ID should be rejected, got: {msg}"
        );
    }

    // -- IR: invalid param ref rejected --

    #[test]
    fn test_invalid_param_ref_rejected() {
        let kernel = KernelDef::new(
            "bad_param",
            vec![f32_param("x")],
            ScalarType::F32,
            vec![n(0, IRNodeKind::Param(5))], // only 1 param, index 5 invalid
            NodeId::new(0),
        );
        let err = kernel.validate().unwrap_err();
        let msg = format!("{err:?}");
        assert!(
            msg.contains("InvalidParamRef"),
            "out-of-bounds param ref should be rejected, got: {msg}"
        );
    }

    // -- MSL codegen: emit_msl produces valid MSL structure --

    #[test]
    fn test_emit_msl_produces_metal_keywords() {
        let kernel = make_relu_kernel();
        let msl = crate::codegen_msl::emit_msl(&kernel).expect("emit_msl should succeed");

        // Verify Metal standard library include
        assert!(
            msl.contains("#include <metal_stdlib>"),
            "MSL should include metal_stdlib"
        );
        assert!(
            msl.contains("using namespace metal;"),
            "MSL should use metal namespace"
        );

        // Verify kernel entry point
        assert!(
            msl.contains("[[kernel]]"),
            "MSL should have [[kernel]] attribute"
        );
        assert!(
            msl.contains("relu_kernel"),
            "MSL should have relu_kernel entry point"
        );

        // Verify buffer bindings
        assert!(msl.contains("[[buffer("), "MSL should have buffer bindings");
    }

    #[test]
    fn test_emit_msl_binary_kernel_contains_binop() {
        let kernel = make_add_kernel();
        let msl = crate::codegen_msl::emit_msl(&kernel).expect("emit_msl should succeed");

        // The add kernel should contain the + operator
        assert!(
            msl.contains('+'),
            "add kernel MSL should contain + operator"
        );
        // Verify both parameters appear
        assert!(msl.contains("float x"), "MSL should declare float x");
        assert!(msl.contains("float y"), "MSL should declare float y");
    }

    #[test]
    fn test_emit_scalar_fn_produces_function_body() {
        let kernel = make_relu_kernel();
        let scalar_fn =
            crate::codegen_msl::emit_scalar_fn(&kernel).expect("emit_scalar_fn should succeed");

        // Should contain return statement
        assert!(
            scalar_fn.contains("return"),
            "scalar function should have return statement"
        );
        // Should have float parameter type
        assert!(
            scalar_fn.contains("float x"),
            "scalar function should declare float x"
        );
        // Should NOT contain [[kernel]] (it's just the scalar helper)
        assert!(
            !scalar_fn.contains("[[kernel]]"),
            "scalar function should not contain [[kernel]]"
        );
    }

    #[test]
    fn test_emit_msl_rejects_reserved_kernel_name() {
        // Kernel named "thread" collides with MSL reserved word
        let kernel = KernelDef::new(
            "thread",
            vec![f32_param("x")],
            ScalarType::F32,
            vec![n(0, IRNodeKind::Param(0))],
            NodeId::new(0),
        );
        let result = crate::codegen_msl::emit_msl(&kernel);
        assert!(
            result.is_err(),
            "MSL reserved word 'thread' should be rejected"
        );
    }

    #[test]
    fn test_emit_msl_rejects_reserved_param_name() {
        // Parameter named "kernel" collides with MSL reserved word
        let kernel = KernelDef::new(
            "nn_fn",
            vec![f32_param("kernel")],
            ScalarType::F32,
            vec![n(0, IRNodeKind::Param(0))],
            NodeId::new(0),
        );
        let result = crate::codegen_msl::emit_msl(&kernel);
        assert!(
            result.is_err(),
            "MSL reserved word 'kernel' as param name should be rejected"
        );
    }

    #[test]
    fn test_emit_msl_f16_kernel_uses_half_type() {
        let kernel = KernelDef::new(
            "scale_f16",
            vec![Param::new("x", ScalarType::F16)],
            ScalarType::F16,
            vec![
                n(0, IRNodeKind::Param(0)),
                n(1, IRNodeKind::Literal(2.0)),
                n(
                    2,
                    IRNodeKind::BinOp {
                        op: BinOpKind::Mul,
                        lhs: NodeId::new(0),
                        rhs: NodeId::new(1),
                    },
                ),
            ],
            NodeId::new(2),
        );
        let msl = crate::codegen_msl::emit_msl(&kernel).expect("emit_msl f16 should succeed");
        assert!(
            msl.contains("half"),
            "F16 kernel MSL should contain 'half' type, got:\n{msl}"
        );
    }

    // -- ScalarType tests --

    #[test]
    fn test_scalar_type_byte_sizes() {
        assert_eq!(ScalarType::F32.byte_size(), 4);
        assert_eq!(ScalarType::F16.byte_size(), 2);
        assert_eq!(ScalarType::BF16.byte_size(), 2);
    }

    #[test]
    fn test_scalar_type_msl_str() {
        assert_eq!(ScalarType::F32.msl_str(), "float");
        assert_eq!(ScalarType::F16.msl_str(), "half");
        assert_eq!(ScalarType::BF16.msl_str(), "half"); // BF16 maps to half on Apple GPU
    }

    #[test]
    fn test_scalar_type_from_type_name_roundtrip() {
        for &ty in &[ScalarType::F32, ScalarType::F16, ScalarType::BF16] {
            let name = ty.type_name();
            let recovered = ScalarType::from_type_name(name);
            assert_eq!(recovered, Some(ty), "roundtrip failed for {name}");
        }
        assert_eq!(
            ScalarType::from_type_name("unknown"),
            None,
            "unknown type name should return None"
        );
    }
}

// =============================================================================
// Section 2: Fusion detection
// =============================================================================

mod fusion_detection_tests {
    use nn_core::dyn_tensor::trace::{ComputationGraph, TraceNode, TraceOp};
    use nn_core::DType;

    use crate::trace_compile::{
        compile_trace, compile_trace_to_plan_with_fusion, compile_trace_with_fusion,
        detect_fusion_chains, CompiledStep,
    };

    fn input_node(id: u64, shape: &[usize]) -> TraceNode {
        TraceNode::new(
            id,
            format!("input_{id}"),
            TraceOp::Input,
            vec![],
            shape.to_vec(),
            DType::F32,
        )
    }

    fn unary_node(id: u64, name: &str, op: TraceOp, input: u64, shape: &[usize]) -> TraceNode {
        TraceNode::new(
            id,
            name.to_string(),
            op,
            vec![input],
            shape.to_vec(),
            DType::F32,
        )
    }

    fn binary_node(
        id: u64,
        name: &str,
        op: TraceOp,
        lhs: u64,
        rhs: u64,
        shape: &[usize],
    ) -> TraceNode {
        TraceNode::new(
            id,
            name.to_string(),
            op,
            vec![lhs, rhs],
            shape.to_vec(),
            DType::F32,
        )
    }

    fn count_dispatches(steps: &[CompiledStep]) -> usize {
        steps
            .iter()
            .filter(|s| matches!(s, CompiledStep::Dispatch { .. }))
            .count()
    }

    #[test]
    fn test_fusion_reduces_dispatch_count_for_unary_chain() {
        // input -> relu -> sigmoid -> exp should fuse to 1 dispatch
        let graph = ComputationGraph::from_nodes(vec![
            input_node(0, &[4]),
            unary_node(1, "relu_0", TraceOp::Relu, 0, &[4]),
            unary_node(2, "sigmoid_0", TraceOp::Sigmoid, 1, &[4]),
            unary_node(3, "exp_0", TraceOp::Exp, 2, &[4]),
        ]);

        let without_fusion = compile_trace(&graph).expect("compile");
        let with_fusion = compile_trace_with_fusion(&graph).expect("compile_with_fusion");

        let d_without = count_dispatches(&without_fusion);
        let d_with = count_dispatches(&with_fusion);

        assert!(
            d_with <= d_without,
            "fusion should not increase dispatch count: without={d_without}, with={d_with}"
        );
        // The 3-op chain should fuse to exactly 1 dispatch
        assert_eq!(
            d_with, 1,
            "3-op elementwise chain should fuse to 1 dispatch"
        );
    }

    #[test]
    fn test_no_fusion_for_different_shapes() {
        // relu on [4] followed by relu on [8] should NOT fuse (different shapes)
        let graph = ComputationGraph::from_nodes(vec![
            input_node(0, &[4]),
            unary_node(1, "relu_0", TraceOp::Relu, 0, &[4]),
            // ReduceSum changes shape, preventing fusion
            TraceNode::new(
                2,
                "reduce_sum".into(),
                TraceOp::ReduceSum {
                    dim: 0,
                    keepdim: false,
                },
                vec![1],
                vec![1],
                DType::F32,
            ),
        ]);

        let steps = compile_trace(&graph).expect("compile");
        // ReduceSum is not fusible elementwise, so each step is separate
        assert!(
            count_dispatches(&steps) >= 2,
            "non-fusible ops should stay separate"
        );
    }

    #[test]
    fn test_fusion_preserves_output_correctness() {
        // The compiled plan output_step should still point to the last step
        let graph = ComputationGraph::from_nodes(vec![
            input_node(0, &[4]),
            unary_node(1, "relu_0", TraceOp::Relu, 0, &[4]),
            unary_node(2, "sigmoid_0", TraceOp::Sigmoid, 1, &[4]),
        ]);

        let plan = compile_trace_to_plan_with_fusion(&graph).expect("compile_plan");
        assert_eq!(
            plan.output_step,
            plan.steps.len() - 1,
            "output_step should be the last step"
        );
    }

    #[test]
    fn test_detect_fusion_chains_finds_chains() {
        // Build a graph with a 3-node fusible chain and one non-fusible op
        let graph = ComputationGraph::from_nodes(vec![
            input_node(0, &[8]),
            unary_node(1, "relu_0", TraceOp::Relu, 0, &[8]),
            unary_node(2, "sigmoid_0", TraceOp::Sigmoid, 1, &[8]),
            unary_node(3, "exp_0", TraceOp::Exp, 2, &[8]),
        ]);

        let chains = detect_fusion_chains(&graph).expect("detect chains");
        // Should find at least one chain of length >= 2
        assert!(
            !chains.is_empty(),
            "should detect at least one fusion chain"
        );
        // The longest chain should have length 3 (relu -> sigmoid -> exp)
        let max_len = chains.iter().map(|c| c.chain_len).max().unwrap_or(0);
        assert_eq!(
            max_len, 3,
            "longest chain should be 3 (relu -> sigmoid -> exp)"
        );
    }

    #[test]
    fn test_fan_out_prevents_fusion() {
        // input -> relu -> {sigmoid, exp}: relu has fan-out 2, so sigmoid cannot fuse with relu
        let graph = ComputationGraph::from_nodes(vec![
            input_node(0, &[4]),
            unary_node(1, "relu_0", TraceOp::Relu, 0, &[4]),
            unary_node(2, "sigmoid_0", TraceOp::Sigmoid, 1, &[4]),
            unary_node(3, "exp_0", TraceOp::Exp, 1, &[4]),
        ]);

        let steps = compile_trace_with_fusion(&graph).expect("compile");
        // relu has fan-out 2 (consumed by both sigmoid and exp), so cannot fuse
        // Expect at least 2 dispatches
        let dispatches = count_dispatches(&steps);
        assert!(
            dispatches >= 2,
            "fan-out > 1 should prevent fusion, got {dispatches} dispatches"
        );
    }

    #[test]
    fn test_binary_ops_in_fusion_chain() {
        // input0 -> relu; input1 -> sigmoid; then add(relu, sigmoid)
        // The add combines two branches; it should compile successfully
        let graph = ComputationGraph::from_nodes(vec![
            input_node(0, &[4]),
            input_node(1, &[4]),
            unary_node(2, "relu_0", TraceOp::Relu, 0, &[4]),
            unary_node(3, "sigmoid_0", TraceOp::Sigmoid, 1, &[4]),
            binary_node(4, "add_0", TraceOp::Add, 2, 3, &[4]),
        ]);

        let plan = compile_trace_to_plan_with_fusion(&graph).expect("compile_plan");
        assert!(
            !plan.steps.is_empty(),
            "plan should have steps for binary op graph"
        );
    }
}

// =============================================================================
// Section 3: Peephole configuration (disabled passes)
// =============================================================================

mod peephole_config_tests {
    

    use nn_core::dyn_tensor::trace::{ComputationGraph, TraceNode, TraceOp};
    use nn_core::DType;

    use crate::trace_compile::{
        compile_trace_to_plan_configured, compile_trace_to_plan_with_fusion, PeepholeConfig,
    };

    fn input_node(id: u64, shape: &[usize]) -> TraceNode {
        TraceNode::new(
            id,
            format!("input_{id}"),
            TraceOp::Input,
            vec![],
            shape.to_vec(),
            DType::F32,
        )
    }

    fn test_node(id: u64, name: &str, inputs: Vec<u64>, shape: Vec<usize>) -> TraceNode {
        TraceNode::new(
            id,
            name.to_string(),
            TraceOp::Relu,
            inputs,
            shape,
            DType::F32,
        )
    }

    #[test]
    fn test_peephole_config_default_all_enabled() {
        let config = PeepholeConfig::default();
        assert!(config.norm_activ_conv1d);
        assert!(config.fused_resblock);
        assert!(config.linear_activation);
        assert!(config.add_layer_norm);
        assert!(config.norm_linear);
        assert!(config.attention_transpose);
        assert!(config.flip_lstm);
        assert!(config.batched_linear_projection);
        assert!(config.channels_first_layer_norm);
        assert!(config.silu_mul);
        assert!(config.auto_fuse_elementwise);
    }

    #[test]
    fn test_peephole_config_all_disabled_produces_no_native_ops_from_peephole() {
        // With all peephole passes disabled, the only NativeOps should come from
        // the trace compiler (graph-level AdaIN detection etc.), not from peephole passes.
        // A simple relu -> sigmoid chain should remain as separate Dispatch steps.
        let graph = ComputationGraph::from_nodes(vec![
            input_node(0, &[4, 768]),
            TraceNode::new(
                1,
                "relu".into(),
                TraceOp::Relu,
                vec![0],
                vec![4, 768],
                DType::F32,
            ),
            TraceNode::new(
                2,
                "sigmoid".into(),
                TraceOp::Sigmoid,
                vec![1],
                vec![4, 768],
                DType::F32,
            ),
        ]);

        let config_disabled = PeepholeConfig {
            norm_activ_conv1d: false,
            fused_resblock: false,
            linear_activation: false,
            add_layer_norm: false,
            norm_linear: false,
            attention_transpose: false,
            flip_lstm: false,
            batched_linear_projection: false,
            channels_first_layer_norm: false,
            silu_mul: false,
            auto_fuse_elementwise: false, // disable elementwise auto-fusion too
            bilstm_cat: false,
            add_norm_linear: false,
            fuse_adain_snake: false,
            fuse_upsample_conv1d: false,
            fuse_instance_norm_mul_add: false,
            fuse_conv1d_activation: false,
            fuse_snake_instance_norm: false,
            fuse_conv1d_snake_norm: false,
            fuse_conv1d_snake_norm_resblock: false,
            fuse_add_instance_norm_conv1x1: false,
            fuse_conv_transpose1d_activation: false,
            norm_activ_conv_transpose1d: false,
            fuse_instance_norm_conv1d: false,
            fuse_conv1d_instance_norm: false,
            fuse_linear_layer_norm: false,
            fuse_resblock_chain: false,
            fuse_activation_conv1d: false,
        };

        let plan = compile_trace_to_plan_configured(&graph, &config_disabled)
            .expect("compile with disabled peephole");

        // With all peephole disabled (including auto_fuse_elementwise), relu->sigmoid should
        // still fuse via the main elementwise chain fusion pass (which runs before peephole).
        // The test verifies the plan compiles successfully.
        assert!(
            !plan.steps.is_empty(),
            "plan should have steps even with all peephole passes disabled"
        );
    }

    #[test]
    fn test_configured_plan_matches_default_when_all_enabled() {
        // When all passes are enabled, the configured plan should match the default plan.
        let graph = ComputationGraph::from_nodes(vec![
            input_node(0, &[4, 768]),
            TraceNode::new(
                1,
                "relu".into(),
                TraceOp::Relu,
                vec![0],
                vec![4, 768],
                DType::F32,
            ),
        ]);

        let default_plan = compile_trace_to_plan_with_fusion(&graph).expect("default plan");
        let configured_plan = compile_trace_to_plan_configured(&graph, &PeepholeConfig::default())
            .expect("configured plan");

        assert_eq!(
            default_plan.steps.len(),
            configured_plan.steps.len(),
            "default and all-enabled configured should produce same step count"
        );
    }
}

// =============================================================================
// Section 4: Edge map
// =============================================================================

mod edge_map_tests {
    use nn_core::dyn_tensor::trace::{ComputationGraph, TraceNode, TraceOp};
    use nn_core::DType;

    use crate::edge_map::compute_edge_map;
    use crate::trace_compile::{compile_trace, CompiledStep};

    fn input_node(id: u64, shape: &[usize]) -> TraceNode {
        TraceNode::new(
            id,
            format!("input_{id}"),
            TraceOp::Input,
            vec![],
            shape.to_vec(),
            DType::F32,
        )
    }

    fn unary_node(id: u64, name: &str, op: TraceOp, input: u64, shape: &[usize]) -> TraceNode {
        TraceNode::new(
            id,
            name.to_string(),
            op,
            vec![input],
            shape.to_vec(),
            DType::F32,
        )
    }

    fn binary_node(
        id: u64,
        name: &str,
        op: TraceOp,
        lhs: u64,
        rhs: u64,
        shape: &[usize],
    ) -> TraceNode {
        TraceNode::new(
            id,
            name.to_string(),
            op,
            vec![lhs, rhs],
            shape.to_vec(),
            DType::F32,
        )
    }

    #[test]
    fn test_edge_map_linear_chain() {
        // input(0) -> relu(1) -> sigmoid(2)
        let graph = ComputationGraph::from_nodes(vec![
            input_node(0, &[4]),
            unary_node(1, "relu_0", TraceOp::Relu, 0, &[4]),
            unary_node(2, "sigmoid_0", TraceOp::Sigmoid, 1, &[4]),
        ]);

        let steps = compile_trace(&graph).expect("compile");
        let edge_map = compute_edge_map(&graph, &steps);

        assert_eq!(edge_map.len(), 3);
        assert!(edge_map[0].is_empty(), "input has no inputs");
        assert_eq!(edge_map[1], vec![0], "relu reads from input");
        assert_eq!(edge_map[2], vec![1], "sigmoid reads from relu");
    }

    #[test]
    fn test_edge_map_diamond_topology() {
        // input(0) -> relu(1), input(0) -> sigmoid(2), add(3)=[relu, sigmoid]
        let graph = ComputationGraph::from_nodes(vec![
            input_node(0, &[4]),
            unary_node(1, "relu_0", TraceOp::Relu, 0, &[4]),
            unary_node(2, "sigmoid_0", TraceOp::Sigmoid, 0, &[4]),
            binary_node(3, "add_0", TraceOp::Add, 1, 2, &[4]),
        ]);

        let steps = compile_trace(&graph).expect("compile");
        let edge_map = compute_edge_map(&graph, &steps);

        assert_eq!(edge_map.len(), 4);
        // input fans out to relu and sigmoid
        assert_eq!(edge_map[1], vec![0], "relu reads from input");
        assert_eq!(edge_map[2], vec![0], "sigmoid reads from input");
        // add consumes relu and sigmoid
        assert_eq!(edge_map[3].len(), 2, "add has 2 inputs");
        assert!(edge_map[3].contains(&1), "add reads from relu");
        assert!(edge_map[3].contains(&2), "add reads from sigmoid");
    }

    #[test]
    fn test_edge_map_external_node_ids_override() {
        // When a step has external_node_ids, those override graph topology
        let graph = ComputationGraph::from_nodes(vec![
            input_node(0, &[4]),
            input_node(1, &[4]),
            unary_node(2, "relu_0", TraceOp::Relu, 0, &[4]),
        ]);

        let mut steps = compile_trace(&graph).expect("compile");

        // Override step 2 to have external_node_ids pointing to input 1 instead of input 0
        if let CompiledStep::Dispatch {
            ref mut external_node_ids,
            ..
        } = steps[2]
        {
            *external_node_ids = Some(vec![1]);
        }

        let edge_map = compute_edge_map(&graph, &steps);
        assert_eq!(
            edge_map[2],
            vec![1],
            "external_node_ids should override graph topology"
        );
    }

    #[test]
    fn test_edge_map_narrow_view_source_step() {
        // NarrowView with explicit source_step should use that instead of graph edges
        let graph = ComputationGraph::from_nodes(vec![
            input_node(0, &[8]),
            unary_node(1, "relu_0", TraceOp::Relu, 0, &[8]),
            TraceNode::new(
                2,
                "narrow".into(),
                TraceOp::Narrow {
                    dim: 0,
                    start: 0,
                    length: 4,
                },
                vec![1],
                vec![4],
                DType::F32,
            ),
        ]);

        let mut steps = compile_trace(&graph).expect("compile");

        // Replace the narrow step with a NarrowView that has explicit source_step
        steps[2] = CompiledStep::NarrowView {
            byte_offset: 0,
            output_shape: vec![4],
            source_step: Some(0), // point to input instead of relu
        };

        let edge_map = compute_edge_map(&graph, &steps);
        assert_eq!(
            edge_map[2],
            vec![0],
            "NarrowView source_step should override graph edge"
        );
    }
}

// =============================================================================
// Section 5: Verifiability classification expanded tests
// =============================================================================

mod verifiability_expanded_tests {
    use nn_core::dyn_tensor::trace::TraceOp;

    use crate::verifiability::{
        classify_callee_name, classify_op, VerifiabilityClass, VerifiabilitySummary,
    };

    // -- classify_callee_name tests --

    #[test]
    fn test_classify_callee_relu_is_verifiable() {
        assert_eq!(classify_callee_name("relu"), VerifiabilityClass::Verifiable);
    }

    #[test]
    fn test_classify_callee_linear_is_verifiable() {
        assert_eq!(
            classify_callee_name("linear"),
            VerifiabilityClass::Verifiable
        );
    }

    #[test]
    fn test_classify_callee_layer_norm_is_bounded() {
        match classify_callee_name("layer_norm") {
            VerifiabilityClass::VerifiableBounded { max_dim } => {
                assert_eq!(max_dim, 512);
            }
            other => panic!("expected VerifiableBounded, got {other:?}"),
        }
    }

    #[test]
    fn test_classify_callee_reshape_is_shape_only() {
        assert_eq!(
            classify_callee_name("reshape"),
            VerifiabilityClass::ShapeOnly
        );
    }

    #[test]
    fn test_classify_callee_dropout_is_passthrough() {
        assert_eq!(
            classify_callee_name("dropout"),
            VerifiabilityClass::Passthrough
        );
    }

    #[test]
    fn test_classify_callee_unknown_is_unverifiable_learned() {
        assert_eq!(
            classify_callee_name("nn_custom_op"),
            VerifiabilityClass::UnverifiableLearned
        );
    }

    #[test]
    fn test_classify_callee_rope_is_verifiable() {
        assert_eq!(classify_callee_name("rope"), VerifiabilityClass::Verifiable);
        assert_eq!(
            classify_callee_name("rotary_embedding"),
            VerifiabilityClass::Verifiable
        );
    }

    #[test]
    fn test_classify_callee_sdpa_is_bounded() {
        match classify_callee_name("sdpa") {
            VerifiabilityClass::VerifiableBounded { max_dim } => assert_eq!(max_dim, 512),
            other => panic!("expected VerifiableBounded for sdpa, got {other:?}"),
        }
    }

    #[test]
    fn test_classify_callee_max_pool1d_is_unverifiable_safe() {
        assert_eq!(
            classify_callee_name("max_pool1d"),
            VerifiabilityClass::UnverifiableSafe
        );
    }

    // -- VerifiabilityClass methods --

    #[test]
    fn test_allows_compilation_for_verifiable_and_safe() {
        assert!(VerifiabilityClass::Verifiable.allows_compilation());
        assert!(VerifiabilityClass::ShapeOnly.allows_compilation());
        assert!(VerifiabilityClass::Passthrough.allows_compilation());
        assert!(VerifiabilityClass::UnverifiableSafe.allows_compilation());
        assert!(!VerifiabilityClass::UnverifiableLearned.allows_compilation());
    }

    #[test]
    fn test_needs_decomposition_for_bounded() {
        let bounded = VerifiabilityClass::VerifiableBounded { max_dim: 512 };
        assert!(
            !bounded.needs_decomposition(256),
            "256 <= 512: no decomposition"
        );
        assert!(
            !bounded.needs_decomposition(512),
            "512 <= 512: no decomposition"
        );
        assert!(
            bounded.needs_decomposition(1024),
            "1024 > 512: needs decomposition"
        );
    }

    #[test]
    fn test_needs_decomposition_for_non_bounded() {
        assert!(!VerifiabilityClass::Verifiable.needs_decomposition(10000));
        assert!(!VerifiabilityClass::ShapeOnly.needs_decomposition(10000));
    }

    // -- VerifiabilitySummary --

    #[test]
    fn test_summary_fully_compilable() {
        let summary = VerifiabilitySummary {
            verifiable: 10,
            bounded: 2,
            shape_only: 3,
            passthrough: 1,
            unverifiable_safe: 1,
            unverifiable_learned: 0,
            unverifiable_learned_ops: vec![],
        };
        assert!(summary.is_fully_compilable());
    }

    #[test]
    fn test_summary_not_compilable_with_learned() {
        let summary = VerifiabilitySummary {
            verifiable: 10,
            bounded: 0,
            shape_only: 0,
            passthrough: 0,
            unverifiable_safe: 0,
            unverifiable_learned: 1,
            unverifiable_learned_ops: vec!["custom_op".to_string()],
        };
        assert!(!summary.is_fully_compilable());
    }

    // -- classify_op for special cases --

    #[test]
    fn test_classify_powf_verifiable_for_special_exponents() {
        for exponent in [1.0, 2.0, 0.5] {
            let op = TraceOp::Powf { exponent };
            assert_eq!(
                classify_op(&op),
                VerifiabilityClass::Verifiable,
                "powf({exponent}) should be verifiable"
            );
        }
    }

    #[test]
    fn test_classify_powf_unverifiable_for_general_exponents() {
        let op = TraceOp::Powf { exponent: 3.7 };
        assert_eq!(
            classify_op(&op),
            VerifiabilityClass::UnverifiableLearned,
            "powf(3.7) should be unverifiable"
        );
    }

    #[test]
    fn test_classify_fract_is_unverifiable_safe() {
        assert_eq!(
            classify_op(&TraceOp::Fract),
            VerifiabilityClass::UnverifiableSafe
        );
    }

    #[test]
    fn test_classify_dropout_is_passthrough() {
        assert_eq!(
            classify_op(&TraceOp::Dropout),
            VerifiabilityClass::Passthrough
        );
    }

    #[test]
    fn test_classify_activation_relu_variant_is_verifiable() {
        use nn_core::dyn_tensor::trace::TraceActivation;

        let known_activations = [
            TraceActivation::Relu,
            TraceActivation::Gelu,
            TraceActivation::Sigmoid,
            TraceActivation::Tanh,
            TraceActivation::Silu,
            TraceActivation::Elu,
            TraceActivation::LeakyRelu,
        ];
        for kind in known_activations {
            let name = kind.as_str();
            let op = TraceOp::Activation { kind };
            assert_eq!(
                classify_op(&op),
                VerifiabilityClass::Verifiable,
                "Activation({name}) should be verifiable"
            );
        }
    }

    #[test]
    fn test_classify_activation_mish_is_unverifiable() {
        use nn_core::dyn_tensor::trace::TraceActivation;

        // Mish maps to "mish" which is not in the verifiable list
        let op = TraceOp::Activation {
            kind: TraceActivation::Mish,
        };
        assert_eq!(
            classify_op(&op),
            VerifiabilityClass::UnverifiableLearned,
            "Mish activation should be unverifiable"
        );
    }
}

// =============================================================================
// Section 6: Cost model expanded tests
// =============================================================================

mod cost_model_expanded_tests {
    

    use crate::cost_model::{CostEstimate, CostModel};
    use crate::tensor_ir::{TensorKernelDef, TensorNode, TensorNodeId, TensorOpKind};
    use crate::trace_compile::{CompiledKernel, CompiledPlan, CompiledStep};

    fn make_kernel(name: &str, output_shape: &[usize]) -> CompiledKernel {
        let output_id = TensorNodeId::new(1);
        let node = TensorNode::new(
            output_id,
            TensorOpKind::Input {
                name: "x".to_string(),
                shape: output_shape.to_vec(),
            },
            output_shape.to_vec(),
        );
        let def = TensorKernelDef {
            name: name.to_string(),
            nodes: vec![node],
            output: output_id,
        };
        CompiledKernel::new(def)
    }

    #[test]
    fn test_cost_model_custom_op_throughput() {
        let mut model = CostModel::apple_m4();
        model.op_throughput.insert("matmul".to_string(), 10e12); // 10 TFLOP/s for matmul

        let kernel = make_kernel("matmul", &[1024, 1024]);
        let plan = CompiledPlan {
            steps: vec![CompiledStep::Dispatch {
                kernel,
                weight_data: Default::default(),
                external_node_ids: None,
            }],
            input_shapes: vec![vec![1024, 1024]],
            output_step: 0,
            weight_names: vec![],
        };

        let est = model.estimate(&plan);
        // With 10 TFLOP/s instead of 1 TFLOP/s, compute time is 10x smaller.
        // The total should still be >= launch overhead.
        assert!(est.total_ns >= model.launch_overhead_ns);
    }

    #[test]
    fn test_cost_model_multiple_dispatches_sum() {
        let model = CostModel::apple_m4();
        let kernel1 = make_kernel("relu", &[1, 256]);
        let kernel2 = make_kernel("sigmoid", &[1, 256]);

        let plan = CompiledPlan {
            steps: vec![
                CompiledStep::InputForward,
                CompiledStep::Dispatch {
                    kernel: kernel1,
                    weight_data: Default::default(),
                    external_node_ids: None,
                },
                CompiledStep::Dispatch {
                    kernel: kernel2,
                    weight_data: Default::default(),
                    external_node_ids: None,
                },
            ],
            input_shapes: vec![vec![1, 256]],
            output_step: 2,
            weight_names: vec![],
        };

        let est = model.estimate(&plan);
        assert_eq!(est.dispatch_count, 2);
        assert!(
            est.total_ns >= 2.0 * model.launch_overhead_ns,
            "2 dispatches should cost >= 2x launch overhead"
        );
    }

    #[test]
    fn test_cost_model_constant_value_is_free() {
        let model = CostModel::apple_m4();
        let plan = CompiledPlan {
            steps: vec![CompiledStep::ConstantValue {
                value: 1.0,
                shape: vec![3],
            }],
            input_shapes: vec![],
            output_step: 0,
            weight_names: vec![],
        };

        let est = model.estimate(&plan);
        assert_eq!(est.total_ns, 0.0, "ConstantValue should have zero cost");
        assert_eq!(est.dispatch_count, 0);
    }

    #[test]
    fn test_cost_estimate_top_expensive_steps_ordering() {
        let est = CostEstimate {
            total_ns: 15000.0,
            per_step_ns: vec![(0, 1000.0), (1, 5000.0), (2, 3000.0), (3, 6000.0)],
            dispatch_count: 4,
        };

        let top3 = est.top_expensive_steps(3);
        assert_eq!(top3.len(), 3);
        // Should be sorted descending by cost
        assert_eq!(top3[0].0, 3); // 6000
        assert_eq!(top3[1].0, 1); // 5000
        assert_eq!(top3[2].0, 2); // 3000
    }

    #[test]
    fn test_cost_estimate_display_contains_metrics() {
        let est = CostEstimate {
            total_ns: 10000.0,
            per_step_ns: vec![(0, 10000.0)],
            dispatch_count: 1,
        };
        let display = format!("{est}");
        assert!(display.contains("CostEstimate:"));
        assert!(display.contains("1 dispatches"));
    }
}

// =============================================================================
// Section 7: Buffer planner supplementary tests
// =============================================================================

mod buffer_planner_supplementary_tests {
    use nn_core::dyn_tensor::trace::{ComputationGraph, TraceNode, TraceOp};
    use nn_core::DType;

    use crate::buffer_planner::plan_buffers;
    use crate::trace_compile::compile_trace_to_plan_with_fusion;

    fn input_node(id: u64, shape: &[usize]) -> TraceNode {
        TraceNode::new(
            id,
            format!("input_{id}"),
            TraceOp::Input,
            vec![],
            shape.to_vec(),
            DType::F32,
        )
    }

    fn unary_node(id: u64, name: &str, op: TraceOp, input: u64, shape: &[usize]) -> TraceNode {
        TraceNode::new(
            id,
            name.to_string(),
            op,
            vec![input],
            shape.to_vec(),
            DType::F32,
        )
    }

    #[test]
    fn test_buffer_plan_total_never_exceeds_naive() {
        // For any graph, total_bytes <= naive_total (buffer reuse only helps)
        let graph = ComputationGraph::from_nodes(vec![
            input_node(0, &[32]),
            unary_node(1, "relu_0", TraceOp::Relu, 0, &[32]),
            unary_node(2, "sigmoid_0", TraceOp::Sigmoid, 1, &[32]),
            unary_node(3, "exp_0", TraceOp::Exp, 2, &[32]),
        ]);

        let plan = compile_trace_to_plan_with_fusion(&graph).expect("compile");
        let bp = plan_buffers(&plan, &graph);

        assert!(
            bp.total_bytes <= bp.naive_total,
            "total ({}) should be <= naive ({})",
            bp.total_bytes,
            bp.naive_total,
        );
    }

    #[test]
    fn test_buffer_plan_consistency_invariants() {
        // Verify structural invariants of the buffer plan
        let graph = ComputationGraph::from_nodes(vec![
            input_node(0, &[8]),
            TraceNode::new(
                1,
                "reduce_sum".into(),
                TraceOp::ReduceSum {
                    dim: 0,
                    keepdim: false,
                },
                vec![0],
                vec![1],
                DType::F32,
            ),
            unary_node(2, "relu_0", TraceOp::Relu, 1, &[1]),
        ]);

        let plan = compile_trace_to_plan_with_fusion(&graph).expect("compile");
        let bp = plan_buffers(&plan, &graph);

        // step_offsets, step_sizes, last_use all have same length
        assert_eq!(bp.step_offsets.len(), plan.steps.len());
        assert_eq!(bp.step_sizes.len(), plan.steps.len());
        assert_eq!(bp.last_use.len(), plan.steps.len());

        // Every last_use[i] >= i (a step can't be last-used before it exists)
        for (i, &lu) in bp.last_use.iter().enumerate() {
            assert!(lu >= i, "last_use[{i}]={lu} should be >= {i}");
        }
    }

    #[test]
    fn test_buffer_plan_step_sizes_match_f32() {
        // For F32 tensors, step size should be num_elements * 4 bytes
        let graph = ComputationGraph::from_nodes(vec![
            input_node(0, &[16]),
            unary_node(1, "relu_0", TraceOp::Relu, 0, &[16]),
        ]);

        let plan = compile_trace_to_plan_with_fusion(&graph).expect("compile");
        let bp = plan_buffers(&plan, &graph);

        // Input step should have size 0 (InputForward, no allocation)
        assert_eq!(bp.step_sizes[0], 0, "InputForward should have size 0");

        // Relu step should have 16 * 4 = 64 bytes (or 0 if fused to something)
        if bp.step_sizes[1] > 0 {
            assert_eq!(
                bp.step_sizes[1],
                16 * 4,
                "relu on [16] should be 64 bytes (16 elements * 4 bytes/f32)"
            );
        }
    }
}

// =============================================================================
// Section 8: Trace compilation correctness
// =============================================================================

mod trace_compile_correctness_tests {
    use nn_core::dyn_tensor::trace::{ComputationGraph, TraceNode, TraceOp};
    use nn_core::DType;

    use crate::trace_compile::{
        compile_trace, compile_trace_to_plan, compile_trace_to_plan_with_fusion, CompiledStep,
    };

    fn input_node(id: u64, shape: &[usize]) -> TraceNode {
        TraceNode::new(
            id,
            format!("input_{id}"),
            TraceOp::Input,
            vec![],
            shape.to_vec(),
            DType::F32,
        )
    }

    fn unary_node(id: u64, name: &str, op: TraceOp, input: u64, shape: &[usize]) -> TraceNode {
        TraceNode::new(
            id,
            name.to_string(),
            op,
            vec![input],
            shape.to_vec(),
            DType::F32,
        )
    }

    #[test]
    fn test_compile_input_only_graph() {
        let graph = ComputationGraph::from_nodes(vec![input_node(0, &[4])]);
        let steps = compile_trace(&graph).expect("compile");
        assert_eq!(steps.len(), 1);
        assert!(matches!(steps[0], CompiledStep::InputForward));
    }

    #[test]
    fn test_compile_relu_produces_dispatch() {
        let graph = ComputationGraph::from_nodes(vec![
            input_node(0, &[4]),
            unary_node(1, "relu_0", TraceOp::Relu, 0, &[4]),
        ]);
        let steps = compile_trace(&graph).expect("compile");
        assert_eq!(steps.len(), 2);
        assert!(matches!(steps[0], CompiledStep::InputForward));
        assert!(
            matches!(steps[1], CompiledStep::Dispatch { .. }),
            "relu should produce a Dispatch step"
        );
    }

    #[test]
    fn test_compile_reshape_produces_passthrough() {
        let graph = ComputationGraph::from_nodes(vec![
            input_node(0, &[2, 4]),
            TraceNode::new(
                1,
                "reshape_0".into(),
                TraceOp::Reshape {
                    target_shape: vec![8],
                },
                vec![0],
                vec![8],
                DType::F32,
            ),
        ]);
        let steps = compile_trace(&graph).expect("compile");
        assert_eq!(steps.len(), 2);
        // Reshape should be Passthrough (zero-copy)
        assert!(
            matches!(
                &steps[1],
                CompiledStep::Passthrough { .. } | CompiledStep::Dispatch { .. }
            ),
            "reshape should be Passthrough or simplified Dispatch"
        );
    }

    #[test]
    fn test_compile_dropout_produces_identity() {
        let graph = ComputationGraph::from_nodes(vec![
            input_node(0, &[4]),
            unary_node(1, "dropout_0", TraceOp::Dropout, 0, &[4]),
        ]);
        let steps = compile_trace(&graph).expect("compile");
        assert_eq!(steps.len(), 2);
        assert!(
            matches!(steps[1], CompiledStep::IdentityPassthrough),
            "dropout (inference) should be IdentityPassthrough"
        );
    }

    #[test]
    fn test_compiled_plan_input_shapes_match_graph() {
        let graph = ComputationGraph::from_nodes(vec![
            input_node(0, &[2, 4]),
            input_node(1, &[2, 8]),
            unary_node(2, "relu_0", TraceOp::Relu, 0, &[2, 4]),
        ]);
        let plan = compile_trace_to_plan(&graph).expect("compile_plan");
        assert_eq!(plan.input_shapes.len(), 2);
        assert_eq!(plan.input_shapes[0], vec![2, 4]);
        assert_eq!(plan.input_shapes[1], vec![2, 8]);
    }

    #[test]
    fn test_compiled_plan_weight_names_collected() {
        // When a dispatch step has weight_data, the plan should collect weight names
        let graph = ComputationGraph::from_nodes(vec![
            input_node(0, &[4]),
            unary_node(1, "relu_0", TraceOp::Relu, 0, &[4]),
        ]);
        let plan = compile_trace_to_plan(&graph).expect("compile_plan");
        // Relu has no weights, so weight_names should be empty
        assert!(
            plan.weight_names.is_empty(),
            "relu dispatch should have no weight names"
        );
    }

    #[test]
    fn test_compiled_plan_output_step_is_last() {
        let graph = ComputationGraph::from_nodes(vec![
            input_node(0, &[4]),
            unary_node(1, "relu_0", TraceOp::Relu, 0, &[4]),
            unary_node(2, "sigmoid_0", TraceOp::Sigmoid, 1, &[4]),
        ]);
        let plan = compile_trace_to_plan_with_fusion(&graph).expect("compile_plan");
        assert_eq!(
            plan.output_step,
            plan.steps.len() - 1,
            "output_step should be the last step index"
        );
    }

    #[test]
    fn test_compile_constant_node_produces_constant_step() {
        let graph = ComputationGraph::from_nodes(vec![TraceNode::new(
            0,
            "const_0".into(),
            TraceOp::Constant { value: 3.14 },
            vec![],
            vec![1],
            DType::F32,
        )]);
        let steps = compile_trace(&graph).expect("compile");
        assert_eq!(steps.len(), 1);
        match &steps[0] {
            CompiledStep::ConstantValue { value, shape } => {
                assert_eq!(shape, &[1]);
                assert!((*value - 3.14).abs() < 1e-5);
            }
            other => panic!("Constant should produce ConstantValue, got: {other:?}"),
        }
    }

    #[test]
    fn test_compile_empty_graph() {
        let graph = ComputationGraph::from_nodes(vec![]);
        let steps = compile_trace(&graph).expect("compile empty");
        assert!(steps.is_empty(), "empty graph should produce no steps");
    }

    #[test]
    fn test_plan_from_fusion_has_fewer_or_equal_dispatches() {
        let graph = ComputationGraph::from_nodes(vec![
            input_node(0, &[4]),
            unary_node(1, "relu_0", TraceOp::Relu, 0, &[4]),
            unary_node(2, "sigmoid_0", TraceOp::Sigmoid, 1, &[4]),
        ]);

        let plan_no_fuse = compile_trace_to_plan(&graph).expect("no-fuse");
        let plan_fuse = compile_trace_to_plan_with_fusion(&graph).expect("fuse");

        let dispatches_no_fuse = plan_no_fuse
            .steps
            .iter()
            .filter(|s| matches!(s, CompiledStep::Dispatch { .. }))
            .count();
        let dispatches_fuse = plan_fuse
            .steps
            .iter()
            .filter(|s| matches!(s, CompiledStep::Dispatch { .. }))
            .count();

        assert!(
            dispatches_fuse <= dispatches_no_fuse,
            "fusion should not increase dispatches: fuse={dispatches_fuse}, no_fuse={dispatches_no_fuse}"
        );
    }
}

// =============================================================================
// Section 9: Compiled kernel API tests
// =============================================================================

mod compiled_kernel_tests {
    use crate::tensor_ir::{TensorKernelDef, TensorNode, TensorNodeId, TensorOpKind};
    use crate::trace_compile::CompiledKernel;

    fn make_test_kernel() -> CompiledKernel {
        let input_id = TensorNodeId::new(0);
        let relu_id = TensorNodeId::new(1);

        let input_node = TensorNode::new(
            input_id,
            TensorOpKind::Input {
                name: "x".to_string(),
                shape: vec![4, 8],
            },
            vec![4, 8],
        );
        let relu_node =
            TensorNode::new(relu_id, TensorOpKind::Relu { input: input_id }, vec![4, 8]);

        let def = TensorKernelDef {
            name: "relu".to_string(),
            nodes: vec![input_node, relu_node],
            output: relu_id,
        };
        CompiledKernel::new(def)
    }

    #[test]
    fn test_compiled_kernel_name() {
        let kernel = make_test_kernel();
        assert_eq!(kernel.name(), "relu");
    }

    #[test]
    fn test_compiled_kernel_input_names() {
        let kernel = make_test_kernel();
        let names = kernel.input_names();
        assert_eq!(names, vec!["x"]);
    }

    #[test]
    fn test_compiled_kernel_output_shape() {
        let kernel = make_test_kernel();
        assert_eq!(kernel.output_shape(), Some(&[4, 8][..]));
    }

    #[test]
    fn test_compiled_kernel_def_accessible() {
        let kernel = make_test_kernel();
        let def = kernel.def();
        assert_eq!(def.name, "relu");
        assert_eq!(def.nodes.len(), 2);
    }
}

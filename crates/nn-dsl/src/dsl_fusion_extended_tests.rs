// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Extended tests for nn-dsl fusion detection, graph recording, optimization
//! passes, kernel IR construction, and MSL/PTX codegen configuration.
//!
//! Covers:
//! - KernelDef IR construction and validation edge cases
//! - ScalarType properties and round-tripping
//! - TensorKernelDef construction and tensor IR operations
//! - PeepholeConfig field coverage and default semantics
//! - NativeOpKind variant construction and serialization
//! - FusionBlocker/FusionGap analysis invariants
//! - CompiledStep/CompiledPlan construction patterns
//! - Auto-fuse codegen (FuseableOp, OpWiring) validation
//! - Verifiability classification coverage
//! - CostModel estimation edge cases
//! - Precision tier contract validation
//!
//! Part of #4553.

use std::collections::HashMap;

use nn_core::dyn_tensor::trace::{ComputationGraph, TraceNode, TraceOp};
use nn_core::DType;

use crate::auto_fuse_codegen::{compose_trace_ops_to_kernel_ir, FuseableOp, OpWiring};
use crate::cost_model::CostModel;
use crate::ir::{
    BinOpKind, CompareOpKind, IRNode, IRNodeKind, KernelDef, MinMaxKind, NodeId, Param, ScalarType,
    UnaryFnKind,
};
use crate::precision::{PrecisionContract, PrecisionTier};
use crate::tensor_ir::{ReduceOp, TensorKernelDef, TensorNode, TensorNodeId, TensorOpKind};
use crate::trace_compile::{
    compile_trace_to_plan, compile_trace_to_plan_configured, count_dispatches,
    detect_fusion_chains, CompiledKernel, CompiledPlan, CompiledStep, FusionBlocker, FusionGap,
    FusionGapAnalysis, NativeOpKind, PeepholeConfig,
};
use crate::verifiability::{classify_callee_name, classify_op, VerifiabilityClass};

// ===========================================================================
// Helpers
// ===========================================================================

fn n(id: usize, kind: IRNodeKind) -> IRNode {
    IRNode::new(NodeId::new(id), kind)
}

fn f32_param(name: &str) -> Param {
    Param::new(name.to_string(), ScalarType::F32)
}

fn input_node(id: u64, shape: Vec<usize>) -> TraceNode {
    TraceNode::new(
        id,
        format!("input_{id}"),
        TraceOp::Input,
        vec![],
        shape,
        DType::F32,
    )
}

fn test_node(id: u64, name: &str, op: TraceOp, inputs: Vec<u64>, shape: Vec<usize>) -> TraceNode {
    TraceNode::new(id, name.to_string(), op, inputs, shape, DType::F32)
}

fn tensor_input(id: usize, name: &str, shape: Vec<usize>) -> TensorNode {
    TensorNode::new(
        TensorNodeId::new(id),
        TensorOpKind::Input {
            name: name.into(),
            shape: shape.clone(),
        },
        shape,
    )
}

fn make_simple_dispatch(name: &str, shape: &[usize]) -> CompiledStep {
    let node_id = TensorNodeId::new(0);
    let input_node = TensorNode::new(
        node_id,
        TensorOpKind::Input {
            name: "input_0".into(),
            shape: shape.to_vec(),
        },
        shape.to_vec(),
    );
    let def = TensorKernelDef::new(name, vec![input_node], node_id);
    CompiledStep::Dispatch {
        kernel: CompiledKernel::new(def),
        weight_data: HashMap::new(),
        external_node_ids: None,
    }
}

// ===========================================================================
// Section 1: ScalarType property tests
// ===========================================================================

#[test]
fn test_scalar_type_f32_properties() {
    let ty = ScalarType::F32;
    assert_eq!(ty.type_name(), "f32");
    assert_eq!(ty.msl_str(), "float");
    assert_eq!(ty.msl_accumulator_str(), "float");
    assert_eq!(ty.byte_size(), 4);
}

#[test]
fn test_scalar_type_f16_properties() {
    let ty = ScalarType::F16;
    assert_eq!(ty.type_name(), "f16");
    assert_eq!(ty.msl_str(), "half");
    assert_eq!(ty.msl_accumulator_str(), "float");
    assert_eq!(ty.byte_size(), 2);
}

#[test]
fn test_scalar_type_bf16_maps_to_half_in_msl() {
    let ty = ScalarType::BF16;
    assert_eq!(ty.type_name(), "bf16");
    assert_eq!(ty.msl_str(), "half", "BF16 must map to half on Apple GPUs");
    assert_eq!(ty.byte_size(), 2);
}

#[test]
fn test_scalar_type_from_type_name_round_trip() {
    for ty in &[ScalarType::F32, ScalarType::F16, ScalarType::BF16] {
        let name = ty.type_name();
        let recovered = ScalarType::from_type_name(name);
        assert_eq!(recovered, Some(*ty), "round trip failed for {name}");
    }
}

#[test]
fn test_scalar_type_from_type_name_unknown_returns_none() {
    assert_eq!(ScalarType::from_type_name("i32"), None);
    assert_eq!(ScalarType::from_type_name("float"), None);
    assert_eq!(ScalarType::from_type_name(""), None);
}

#[test]
fn test_scalar_type_all_accumulate_in_f32() {
    for ty in &[ScalarType::F32, ScalarType::F16, ScalarType::BF16] {
        assert_eq!(
            ty.msl_accumulator_str(),
            "float",
            "{ty:?} must accumulate in float"
        );
    }
}

// ===========================================================================
// Section 2: KernelDef IR construction and validation
// ===========================================================================

#[test]
fn test_kernel_def_identity_function_validates() {
    let kernel = KernelDef::new(
        "identity",
        vec![f32_param("x")],
        ScalarType::F32,
        vec![n(0, IRNodeKind::Param(0))],
        NodeId::new(0),
    );
    kernel.validate().expect("identity kernel should validate");
}

#[test]
fn test_kernel_def_literal_only_validates() {
    let kernel = KernelDef::new(
        "constant",
        vec![],
        ScalarType::F32,
        vec![n(0, IRNodeKind::Literal(42.0))],
        NodeId::new(0),
    );
    kernel
        .validate()
        .expect("literal-only kernel should validate");
}

#[test]
fn test_kernel_def_all_binop_kinds() {
    for (i, op) in [
        BinOpKind::Add,
        BinOpKind::Sub,
        BinOpKind::Mul,
        BinOpKind::Div,
    ]
    .iter()
    .enumerate()
    {
        let kernel = KernelDef::new(
            format!("binop_{i}"),
            vec![f32_param("x"), f32_param("y")],
            ScalarType::F32,
            vec![
                n(0, IRNodeKind::Param(0)),
                n(1, IRNodeKind::Param(1)),
                n(
                    2,
                    IRNodeKind::BinOp {
                        op: *op,
                        lhs: NodeId::new(0),
                        rhs: NodeId::new(1),
                    },
                ),
            ],
            NodeId::new(2),
        );
        kernel
            .validate()
            .unwrap_or_else(|e| panic!("binop {i} should validate: {e}"));
    }
}

#[test]
fn test_kernel_def_compare_ops_validate() {
    for (i, op) in [
        CompareOpKind::Lt,
        CompareOpKind::Le,
        CompareOpKind::Gt,
        CompareOpKind::Ge,
        CompareOpKind::Eq,
        CompareOpKind::Ne,
    ]
    .iter()
    .enumerate()
    {
        let kernel = KernelDef::new(
            format!("compare_{i}"),
            vec![f32_param("x"), f32_param("y")],
            ScalarType::F32,
            vec![
                n(0, IRNodeKind::Param(0)),
                n(1, IRNodeKind::Param(1)),
                n(
                    2,
                    IRNodeKind::Compare {
                        op: *op,
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
        );
        kernel
            .validate()
            .unwrap_or_else(|e| panic!("compare {i} should validate: {e}"));
    }
}

#[test]
fn test_kernel_def_unary_fn_sin_validates() {
    let kernel = KernelDef::new(
        "sin_kernel",
        vec![f32_param("x")],
        ScalarType::F32,
        vec![
            n(0, IRNodeKind::Param(0)),
            n(
                1,
                IRNodeKind::UnaryFn {
                    op: UnaryFnKind::Sin,
                    input: NodeId::new(0),
                },
            ),
        ],
        NodeId::new(1),
    );
    kernel.validate().expect("sin kernel should validate");
}

#[test]
fn test_kernel_def_clamp_validates() {
    let kernel = KernelDef::new(
        "clamp_kernel",
        vec![f32_param("x"), f32_param("lo"), f32_param("hi")],
        ScalarType::F32,
        vec![
            n(0, IRNodeKind::Param(0)),
            n(1, IRNodeKind::Param(1)),
            n(2, IRNodeKind::Param(2)),
            n(
                3,
                IRNodeKind::Clamp {
                    input: NodeId::new(0),
                    min: NodeId::new(1),
                    max: NodeId::new(2),
                },
            ),
        ],
        NodeId::new(3),
    );
    kernel.validate().expect("clamp kernel should validate");
}

#[test]
fn test_kernel_def_minmax_validates() {
    for kind in [MinMaxKind::Min, MinMaxKind::Max] {
        let kernel = KernelDef::new(
            "minmax",
            vec![f32_param("a"), f32_param("b")],
            ScalarType::F32,
            vec![
                n(0, IRNodeKind::Param(0)),
                n(1, IRNodeKind::Param(1)),
                n(
                    2,
                    IRNodeKind::MinMax {
                        op: kind,
                        lhs: NodeId::new(0),
                        rhs: NodeId::new(1),
                    },
                ),
            ],
            NodeId::new(2),
        );
        kernel.validate().expect("minmax should validate");
    }
}

#[test]
fn test_kernel_def_sum_reduce_validates() {
    let kernel = KernelDef::new(
        "sum3",
        vec![f32_param("x"), f32_param("y"), f32_param("z")],
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
    kernel.validate().expect("sum_reduce should validate");
}

#[test]
fn test_kernel_def_forward_reference_rejected() {
    let kernel = KernelDef::new(
        "bad",
        vec![f32_param("x")],
        ScalarType::F32,
        vec![
            n(
                0,
                IRNodeKind::UnaryFn {
                    op: UnaryFnKind::Sin,
                    input: NodeId::new(1),
                },
            ),
            n(1, IRNodeKind::Param(0)),
        ],
        NodeId::new(0),
    );
    assert!(
        kernel.validate().is_err(),
        "forward references must be rejected"
    );
}

#[test]
fn test_kernel_def_self_reference_rejected() {
    let kernel = KernelDef::new(
        "self_ref",
        vec![f32_param("x")],
        ScalarType::F32,
        vec![
            n(0, IRNodeKind::Param(0)),
            n(
                1,
                IRNodeKind::UnaryFn {
                    op: UnaryFnKind::Abs,
                    input: NodeId::new(1),
                },
            ),
        ],
        NodeId::new(1),
    );
    assert!(
        kernel.validate().is_err(),
        "self-references must be rejected"
    );
}

#[test]
fn test_kernel_def_has_ftz_sensitive_op_with_div() {
    let kernel = KernelDef::new(
        "div_kernel",
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
    assert!(
        kernel.has_ftz_sensitive_op(),
        "div kernel must be FTZ-sensitive"
    );
}

#[test]
fn test_kernel_def_no_ftz_for_add_only() {
    let kernel = KernelDef::new(
        "add_only",
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
    );
    assert!(
        !kernel.has_ftz_sensitive_op(),
        "add-only kernel should not be FTZ-sensitive"
    );
}

// ===========================================================================
// Section 3: TensorKernelDef construction
// ===========================================================================

#[test]
fn test_tensor_kernel_def_input_only() {
    let def = TensorKernelDef::new(
        "input_test",
        vec![tensor_input(0, "x", vec![2, 8])],
        TensorNodeId::new(0),
    );
    assert_eq!(def.name, "input_test");
    assert_eq!(def.nodes.len(), 1);
    assert_eq!(def.nodes[0].shape, vec![2, 8]);
}

#[test]
fn test_tensor_kernel_def_gelu_construction() {
    let def = TensorKernelDef::new(
        "gelu_test",
        vec![
            tensor_input(0, "x", vec![4, 16]),
            TensorNode::new(
                TensorNodeId::new(1),
                TensorOpKind::Gelu {
                    input: TensorNodeId::new(0),
                },
                vec![4, 16],
            ),
        ],
        TensorNodeId::new(1),
    );
    assert_eq!(def.nodes.len(), 2);
    assert_eq!(def.output, TensorNodeId::new(1));
}

#[test]
fn test_tensor_kernel_def_relu_construction() {
    let def = TensorKernelDef::new(
        "relu_test",
        vec![
            tensor_input(0, "x", vec![1, 256]),
            TensorNode::new(
                TensorNodeId::new(1),
                TensorOpKind::Relu {
                    input: TensorNodeId::new(0),
                },
                vec![1, 256],
            ),
        ],
        TensorNodeId::new(1),
    );
    assert_eq!(def.name, "relu_test");
}

#[test]
fn test_tensor_kernel_def_sigmoid_construction() {
    let def = TensorKernelDef::new(
        "sigmoid_test",
        vec![
            tensor_input(0, "x", vec![2, 32]),
            TensorNode::new(
                TensorNodeId::new(1),
                TensorOpKind::Sigmoid {
                    input: TensorNodeId::new(0),
                },
                vec![2, 32],
            ),
        ],
        TensorNodeId::new(1),
    );
    assert_eq!(def.nodes.len(), 2);
}

#[test]
fn test_tensor_kernel_def_reduce_sum_construction() {
    let def = TensorKernelDef::new(
        "reduce_sum",
        vec![
            tensor_input(0, "x", vec![4, 8]),
            TensorNode::new(
                TensorNodeId::new(1),
                TensorOpKind::Reduce {
                    input: TensorNodeId::new(0),
                    op: ReduceOp::Sum,
                    axis: 1,
                    keepdim: false,
                },
                vec![4],
            ),
        ],
        TensorNodeId::new(1),
    );
    assert_eq!(def.nodes[1].shape, vec![4]);
}

#[test]
fn test_tensor_kernel_def_reduce_max_construction() {
    let def = TensorKernelDef::new(
        "reduce_max",
        vec![
            tensor_input(0, "x", vec![2, 16]),
            TensorNode::new(
                TensorNodeId::new(1),
                TensorOpKind::Reduce {
                    input: TensorNodeId::new(0),
                    op: ReduceOp::Max,
                    axis: 1,
                    keepdim: false,
                },
                vec![2],
            ),
        ],
        TensorNodeId::new(1),
    );
    assert_eq!(def.nodes[1].shape, vec![2]);
}

// ===========================================================================
// Section 4: CompiledKernel / CompiledStep / CompiledPlan
// ===========================================================================

#[test]
fn test_compiled_kernel_name_accessor() {
    let def = TensorKernelDef::new(
        "test_kernel",
        vec![tensor_input(0, "x", vec![4, 8])],
        TensorNodeId::new(0),
    );
    let ck = CompiledKernel::new(def);
    assert_eq!(ck.name(), "test_kernel");
}

#[test]
fn test_compiled_kernel_input_names() {
    let def = TensorKernelDef::new(
        "multi_input",
        vec![
            tensor_input(0, "features", vec![2, 64]),
            tensor_input(1, "weights", vec![64, 32]),
        ],
        TensorNodeId::new(1),
    );
    let ck = CompiledKernel::new(def);
    let names = ck.input_names();
    assert_eq!(names, vec!["features", "weights"]);
}

#[test]
fn test_compiled_kernel_output_shape() {
    let def = TensorKernelDef::new(
        "shaped",
        vec![tensor_input(0, "x", vec![3, 7, 11])],
        TensorNodeId::new(0),
    );
    let ck = CompiledKernel::new(def);
    assert_eq!(ck.output_shape(), Some(&[3, 7, 11][..]));
}

#[test]
fn test_compiled_plan_empty() {
    let plan = CompiledPlan {
        steps: vec![],
        input_shapes: vec![],
        output_step: 0,
        weight_names: vec![],
    };
    assert_eq!(count_dispatches(&plan), 0);
}

#[test]
fn test_compiled_plan_single_dispatch_count() {
    let plan = CompiledPlan {
        steps: vec![make_simple_dispatch("relu", &[1, 256])],
        input_shapes: vec![vec![1, 256]],
        output_step: 0,
        weight_names: vec![],
    };
    assert_eq!(count_dispatches(&plan), 1);
}

#[test]
fn test_compiled_plan_mixed_step_types() {
    let plan = CompiledPlan {
        steps: vec![
            CompiledStep::InputForward,
            make_simple_dispatch("relu", &[1, 64]),
            CompiledStep::Passthrough {
                op_name: "reshape".into(),
                output_shape: vec![1, 8, 8],
            },
            make_simple_dispatch("gelu", &[1, 8, 8]),
            CompiledStep::IdentityPassthrough,
        ],
        input_shapes: vec![vec![1, 64]],
        output_step: 3,
        weight_names: vec![],
    };
    assert_eq!(count_dispatches(&plan), 2);
}

#[test]
fn test_compiled_step_passthrough_preserves_shape() {
    let step = CompiledStep::Passthrough {
        op_name: "squeeze".into(),
        output_shape: vec![4, 8],
    };
    match &step {
        CompiledStep::Passthrough { output_shape, .. } => {
            assert_eq!(output_shape, &vec![4, 8]);
        }
        _ => panic!("expected Passthrough"),
    }
}

#[test]
fn test_compiled_step_constant_value() {
    let step = CompiledStep::ConstantValue {
        value: 1.0,
        shape: vec![1, 512],
    };
    match &step {
        CompiledStep::ConstantValue { value, shape } => {
            assert!((value - 1.0).abs() < 1e-6);
            assert_eq!(shape, &vec![1, 512]);
        }
        _ => panic!("expected ConstantValue"),
    }
}

#[test]
fn test_compiled_step_narrow_view() {
    let step = CompiledStep::NarrowView {
        byte_offset: 1024,
        output_shape: vec![1, 64],
        source_step: Some(3),
    };
    match &step {
        CompiledStep::NarrowView {
            byte_offset,
            source_step,
            ..
        } => {
            assert_eq!(*byte_offset, 1024);
            assert_eq!(*source_step, Some(3));
        }
        _ => panic!("expected NarrowView"),
    }
}

// ===========================================================================
// Section 5: PeepholeConfig tests
// ===========================================================================

#[test]
fn test_peephole_config_default_all_enabled() {
    let cfg = PeepholeConfig::default();
    assert!(cfg.norm_activ_conv1d);
    assert!(cfg.fused_resblock);
    assert!(cfg.linear_activation);
    assert!(cfg.add_layer_norm);
    assert!(cfg.norm_linear);
    assert!(cfg.attention_transpose);
    assert!(cfg.flip_lstm);
    assert!(cfg.batched_linear_projection);
    assert!(cfg.channels_first_layer_norm);
    assert!(cfg.silu_mul);
    assert!(cfg.auto_fuse_elementwise);
    assert!(cfg.bilstm_cat);
    assert!(cfg.add_norm_linear);
    assert!(cfg.fuse_adain_snake);
    assert!(cfg.fuse_upsample_conv1d);
    assert!(cfg.fuse_instance_norm_mul_add);
    assert!(cfg.fuse_conv1d_activation);
}

#[test]
fn test_peephole_config_all_disabled() {
    let cfg = PeepholeConfig {
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
        auto_fuse_elementwise: false,
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
    assert!(!cfg.norm_activ_conv1d);
    assert!(!cfg.fused_resblock);
    assert!(!cfg.linear_activation);
    assert!(!cfg.add_layer_norm);
    assert!(!cfg.norm_linear);
    assert!(!cfg.auto_fuse_elementwise);
}

// ===========================================================================
// Section 6: NativeOpKind construction
// ===========================================================================

#[test]
fn test_native_op_lstm_sequence() {
    let op = NativeOpKind::LstmSequence {
        hidden_size: 128,
        input_shape: vec![50, 1, 64],
        h_shape: vec![1, 128],
        reverse: false,
    };
    match &op {
        NativeOpKind::LstmSequence {
            hidden_size,
            reverse,
            ..
        } => {
            assert_eq!(*hidden_size, 128);
            assert!(!reverse);
        }
        _ => panic!("expected LstmSequence"),
    }
}

#[test]
fn test_native_op_lstm_sequence_reverse() {
    let op = NativeOpKind::LstmSequence {
        hidden_size: 256,
        input_shape: vec![100, 1, 128],
        h_shape: vec![1, 256],
        reverse: true,
    };
    match &op {
        NativeOpKind::LstmSequence { reverse, .. } => {
            assert!(reverse, "reverse flag must be set");
        }
        _ => panic!("expected LstmSequence"),
    }
}

#[test]
fn test_native_op_instance_norm() {
    let op = NativeOpKind::InstanceNorm {
        eps: 1e-5,
        input_shape: vec![1, 64, 1024],
    };
    match &op {
        NativeOpKind::InstanceNorm { eps, input_shape } => {
            assert!((eps - 1e-5).abs() < 1e-8);
            assert_eq!(input_shape, &vec![1, 64, 1024]);
        }
        _ => panic!("expected InstanceNorm"),
    }
}

#[test]
fn test_native_op_layer_norm() {
    let op = NativeOpKind::LayerNorm {
        eps: 1e-5,
        input_shape: vec![1, 32, 768],
        hidden_dim: 768,
    };
    match &op {
        NativeOpKind::LayerNorm { hidden_dim, .. } => {
            assert_eq!(*hidden_dim, 768);
        }
        _ => panic!("expected LayerNorm"),
    }
}

#[test]
fn test_native_op_flash_attention() {
    let op = NativeOpKind::FlashAttention {
        scale: 0.125,
        causal: true,
        q_shape: vec![1, 8, 64, 64],
        k_shape: vec![1, 8, 64, 64],
        output_shape: vec![1, 8, 64, 64],
        input_layout: Default::default(),
    };
    match &op {
        NativeOpKind::FlashAttention { scale, causal, .. } => {
            assert!((scale - 0.125).abs() < 1e-6);
            assert!(causal);
        }
        _ => panic!("expected FlashAttention"),
    }
}

#[test]
fn test_native_op_cumsum() {
    let op = NativeOpKind::Cumsum {
        dim: 1,
        input_shape: vec![1, 256],
    };
    match &op {
        NativeOpKind::Cumsum { dim, .. } => {
            assert_eq!(*dim, 1);
        }
        _ => panic!("expected Cumsum"),
    }
}

#[test]
fn test_native_op_add_layer_norm() {
    let op = NativeOpKind::AddLayerNorm {
        eps: 1e-6,
        input_shape: vec![1, 32, 512],
        hidden_dim: 512,
    };
    match &op {
        NativeOpKind::AddLayerNorm { hidden_dim, .. } => {
            assert_eq!(*hidden_dim, 512);
        }
        _ => panic!("expected AddLayerNorm"),
    }
}

#[test]
fn test_native_op_linear_activation_relu() {
    let op = NativeOpKind::LinearActivation {
        activation: crate::trace_compile::GemmActivation::Relu,
        in_features: 256,
        out_features: 128,
        has_bias: true,
        input_shape: vec![1, 256],
    };
    match &op {
        NativeOpKind::LinearActivation {
            in_features,
            out_features,
            has_bias,
            ..
        } => {
            assert_eq!(*in_features, 256);
            assert_eq!(*out_features, 128);
            assert!(has_bias);
        }
        _ => panic!("expected LinearActivation"),
    }
}

#[test]
fn test_native_op_max_pool1d() {
    let op = NativeOpKind::MaxPool1d {
        kernel_size: 3,
        stride: 2,
        padding: 1,
        input_shape: vec![1, 64, 100],
    };
    match &op {
        NativeOpKind::MaxPool1d {
            kernel_size,
            stride,
            padding,
            ..
        } => {
            assert_eq!(*kernel_size, 3);
            assert_eq!(*stride, 2);
            assert_eq!(*padding, 1);
        }
        _ => panic!("expected MaxPool1d"),
    }
}

#[test]
fn test_native_op_norm_linear() {
    let op = NativeOpKind::NormLinear {
        norm_kind: crate::trace_compile::FusedNormKind::RmsNorm,
        eps: 1e-6,
        input_shape: vec![1, 32, 768],
        hidden_dim: 768,
        out_features: 3072,
        has_bias: false,
    };
    match &op {
        NativeOpKind::NormLinear {
            norm_kind,
            out_features,
            has_bias,
            ..
        } => {
            assert_eq!(*norm_kind, crate::trace_compile::FusedNormKind::RmsNorm);
            assert_eq!(*out_features, 3072);
            assert!(!has_bias);
        }
        _ => panic!("expected NormLinear"),
    }
}

// ===========================================================================
// Section 7: FusionBlocker / FusionGap / FusionGapAnalysis
// ===========================================================================

#[test]
fn test_fusion_blocker_display_all_variants() {
    let variants = [
        FusionBlocker::FanOut,
        FusionBlocker::ShapeMismatch,
        FusionBlocker::NonFusibleOp,
        FusionBlocker::NotDispatch,
        FusionBlocker::AlreadyOptimal,
        FusionBlocker::NoPeepholePattern,
        FusionBlocker::NoDependency,
    ];
    for v in &variants {
        let s = format!("{v}");
        assert!(!s.is_empty(), "Display must produce non-empty string");
    }
}

#[test]
fn test_fusion_gap_construction() {
    let gap = FusionGap {
        step_a: 0,
        step_b: 1,
        kernel_a: "relu".to_string(),
        kernel_b: "gelu".to_string(),
        reason: FusionBlocker::ShapeMismatch,
        savings: 1,
    };
    assert_eq!(gap.step_a, 0);
    assert_eq!(gap.step_b, 1);
    assert_eq!(gap.savings, 1);
}

#[test]
fn test_fusion_gap_analysis_zero_dispatches() {
    let analysis = FusionGapAnalysis {
        gaps: vec![],
        total_dispatches: 0,
        theoretical_minimum: 0,
    };
    assert!(
        (analysis.optimization_opportunity_pct() - 0.0).abs() < 1e-6,
        "zero dispatches should give 0% opportunity"
    );
}

#[test]
fn test_fusion_gap_analysis_with_gaps() {
    let analysis = FusionGapAnalysis {
        gaps: vec![FusionGap {
            step_a: 0,
            step_b: 1,
            kernel_a: "a".into(),
            kernel_b: "b".into(),
            reason: FusionBlocker::FanOut,
            savings: 1,
        }],
        total_dispatches: 10,
        theoretical_minimum: 5,
    };
    // (10 - 5) / 10 * 100 = 50%
    assert!(
        (analysis.optimization_opportunity_pct() - 50.0).abs() < 1e-6,
        "should be 50% opportunity, got {}",
        analysis.optimization_opportunity_pct()
    );
}

#[test]
fn test_fusion_gap_analysis_no_improvement_possible() {
    let analysis = FusionGapAnalysis {
        gaps: vec![],
        total_dispatches: 10,
        theoretical_minimum: 10,
    };
    assert!(
        (analysis.optimization_opportunity_pct() - 0.0).abs() < 1e-6,
        "no gaps means 0% opportunity"
    );
}

// ===========================================================================
// Section 8: Auto-fuse codegen (FuseableOp / OpWiring)
// ===========================================================================

#[test]
fn test_fuseable_op_unary_construction() {
    let op = FuseableOp::unary(TraceOp::Relu);
    assert_eq!(op.wiring, OpWiring::Unary);
}

#[test]
fn test_fuseable_op_binary_second_external() {
    let op = FuseableOp {
        op: TraceOp::Add,
        wiring: OpWiring::BinarySecondExternal,
    };
    assert_eq!(op.wiring, OpWiring::BinarySecondExternal);
}

#[test]
fn test_fuseable_op_binary_first_external() {
    let op = FuseableOp {
        op: TraceOp::Mul,
        wiring: OpWiring::BinaryFirstExternal,
    };
    assert_eq!(op.wiring, OpWiring::BinaryFirstExternal);
}

#[test]
fn test_fuseable_op_binary_both_external() {
    let op = FuseableOp {
        op: TraceOp::Add,
        wiring: OpWiring::BinaryBothExternal,
    };
    assert_eq!(op.wiring, OpWiring::BinaryBothExternal);
}

#[test]
fn test_compose_single_unary_op() {
    let ops = vec![FuseableOp::unary(TraceOp::Relu)];
    let result = compose_trace_ops_to_kernel_ir(&ops, "test_relu");
    assert!(result.is_ok(), "single unary compose should succeed");
    let kernel = result.unwrap();
    kernel.validate().expect("composed kernel should validate");
    assert_eq!(kernel.name, "test_relu");
}

#[test]
fn test_compose_two_unary_ops() {
    let ops = vec![
        FuseableOp::unary(TraceOp::Relu),
        FuseableOp::unary(TraceOp::Sigmoid),
    ];
    let result = compose_trace_ops_to_kernel_ir(&ops, "relu_sigmoid");
    assert!(
        result.is_ok(),
        "relu -> sigmoid compose should succeed: {:?}",
        result.err()
    );
    let kernel = result.unwrap();
    kernel.validate().expect("composed kernel should validate");
    assert_eq!(kernel.params.len(), 1, "chain of unary ops has one input");
}

#[test]
fn test_compose_unary_then_binary() {
    let ops = vec![
        FuseableOp::unary(TraceOp::Exp),
        FuseableOp {
            op: TraceOp::Add,
            wiring: OpWiring::BinarySecondExternal,
        },
    ];
    let result = compose_trace_ops_to_kernel_ir(&ops, "exp_add");
    assert!(result.is_ok(), "exp -> add compose: {:?}", result.err());
    let kernel = result.unwrap();
    kernel.validate().expect("composed kernel should validate");
    // exp(x) + y => 2 params: x, y
    assert_eq!(kernel.params.len(), 2);
}

// ===========================================================================
// Section 9: Verifiability classification
// ===========================================================================

#[test]
fn test_classify_op_relu_is_verifiable() {
    assert!(matches!(
        classify_op(&TraceOp::Relu),
        VerifiabilityClass::Verifiable
    ));
}

#[test]
fn test_classify_op_gelu_is_verifiable() {
    assert!(matches!(
        classify_op(&TraceOp::Gelu),
        VerifiabilityClass::Verifiable
    ));
}

#[test]
fn test_classify_op_sigmoid_is_verifiable() {
    assert!(matches!(
        classify_op(&TraceOp::Sigmoid),
        VerifiabilityClass::Verifiable
    ));
}

#[test]
fn test_classify_op_tanh_is_verifiable() {
    assert!(matches!(
        classify_op(&TraceOp::Tanh),
        VerifiabilityClass::Verifiable
    ));
}

#[test]
fn test_classify_op_exp_is_verifiable() {
    assert!(matches!(
        classify_op(&TraceOp::Exp),
        VerifiabilityClass::Verifiable
    ));
}

#[test]
fn test_classify_op_add_is_verifiable() {
    assert!(matches!(
        classify_op(&TraceOp::Add),
        VerifiabilityClass::Verifiable
    ));
}

#[test]
fn test_classify_callee_relu_is_verifiable() {
    assert!(matches!(
        classify_callee_name("relu"),
        VerifiabilityClass::Verifiable
    ));
}

#[test]
fn test_classify_callee_matmul_is_verifiable() {
    assert!(matches!(
        classify_callee_name("matmul"),
        VerifiabilityClass::Verifiable
    ));
}

#[test]
fn test_classify_callee_unknown_not_verifiable() {
    let cls = classify_callee_name("custom_unknown_op_xyz");
    assert!(
        !matches!(cls, VerifiabilityClass::Verifiable),
        "unknown ops should not be classified as Verifiable"
    );
}

// ===========================================================================
// Section 10: PrecisionTier / PrecisionContract
// ===========================================================================

#[test]
fn test_precision_tier_strict_ordering() {
    let strict = PrecisionContract::bootstrap(PrecisionTier::Strict, ScalarType::F32);
    let normal = PrecisionContract::bootstrap(PrecisionTier::Normal, ScalarType::F32);
    let relaxed = PrecisionContract::bootstrap(PrecisionTier::Relaxed, ScalarType::F32);
    assert!(strict.differential_abs_budget <= normal.differential_abs_budget);
    assert!(normal.differential_abs_budget <= relaxed.differential_abs_budget);
}

#[test]
fn test_precision_contract_strict_has_positive_tolerance() {
    let c = PrecisionContract::bootstrap(PrecisionTier::Strict, ScalarType::F32);
    assert!(
        c.differential_abs_budget > 0.0,
        "strict abs budget must be positive"
    );
    assert!(
        c.differential_rel_budget > 0.0,
        "strict rel budget must be positive"
    );
}

#[test]
fn test_precision_contract_relaxed_has_larger_tolerance() {
    let strict = PrecisionContract::bootstrap(PrecisionTier::Strict, ScalarType::F32);
    let relaxed = PrecisionContract::bootstrap(PrecisionTier::Relaxed, ScalarType::F32);
    assert!(
        relaxed.differential_abs_budget >= strict.differential_abs_budget,
        "relaxed must have >= strict tolerance"
    );
}

// ===========================================================================
// Section 11: CostModel edge cases
// ===========================================================================

#[test]
fn test_cost_model_estimate_passthrough_only() {
    let model = CostModel::apple_m4();
    let plan = CompiledPlan {
        steps: vec![
            CompiledStep::Passthrough {
                op_name: "reshape".into(),
                output_shape: vec![4, 8],
            },
            CompiledStep::Passthrough {
                op_name: "squeeze".into(),
                output_shape: vec![32],
            },
        ],
        input_shapes: vec![],
        output_step: 1,
        weight_names: vec![],
    };
    let est = model.estimate(&plan);
    assert_eq!(est.dispatch_count, 0, "passthroughs have no dispatches");
}

#[test]
fn test_cost_model_multiple_dispatches() {
    let model = CostModel::apple_m4();
    let plan = CompiledPlan {
        steps: vec![
            make_simple_dispatch("relu", &[1, 1024]),
            make_simple_dispatch("gelu", &[1, 1024]),
            make_simple_dispatch("sigmoid", &[1, 1024]),
        ],
        input_shapes: vec![vec![1, 1024]],
        output_step: 2,
        weight_names: vec![],
    };
    let est = model.estimate(&plan);
    assert_eq!(est.dispatch_count, 3);
    assert!(est.total_ns > 0.0, "3 dispatches should have positive cost");
}

// ===========================================================================
// Section 12: Fusion chain detection on graph patterns
// ===========================================================================

#[test]
fn test_detect_fusion_chains_empty_graph() {
    let graph = ComputationGraph::from_nodes(vec![]);
    let chains = detect_fusion_chains(&graph).unwrap();
    assert!(chains.is_empty(), "empty graph has no chains");
}

#[test]
fn test_detect_fusion_chains_single_input() {
    let graph = ComputationGraph::from_nodes(vec![input_node(0, vec![1, 64])]);
    let chains = detect_fusion_chains(&graph).unwrap();
    assert!(chains.is_empty(), "single input has no chains");
}

#[test]
fn test_detect_fusion_chains_tanh_sigmoid() {
    let graph = ComputationGraph::from_nodes(vec![
        input_node(0, vec![1, 128]),
        test_node(1, "tanh", TraceOp::Tanh, vec![0], vec![1, 128]),
        test_node(2, "sigmoid", TraceOp::Sigmoid, vec![1], vec![1, 128]),
    ]);
    let chains = detect_fusion_chains(&graph).unwrap();
    assert!(!chains.is_empty(), "tanh -> sigmoid should form a chain");
    assert_eq!(chains[0].chain_len, 2);
}

#[test]
fn test_detect_fusion_chains_three_unary() {
    let graph = ComputationGraph::from_nodes(vec![
        input_node(0, vec![1, 64]),
        test_node(1, "relu", TraceOp::Relu, vec![0], vec![1, 64]),
        test_node(2, "exp", TraceOp::Exp, vec![1], vec![1, 64]),
        test_node(3, "log", TraceOp::Log, vec![2], vec![1, 64]),
    ]);
    let chains = detect_fusion_chains(&graph).unwrap();
    assert!(!chains.is_empty());
    assert!(chains[0].chain_len >= 3, "should detect a chain of 3");
}

// ===========================================================================
// Section 13: compile_trace_to_plan on simple graphs
// ===========================================================================

#[test]
fn test_compile_trace_to_plan_single_relu() {
    let graph = ComputationGraph::from_nodes(vec![
        input_node(0, vec![1, 256]),
        test_node(1, "relu", TraceOp::Relu, vec![0], vec![1, 256]),
    ]);
    let plan = compile_trace_to_plan(&graph).expect("compile should succeed");
    assert!(!plan.steps.is_empty());
    assert_eq!(plan.input_shapes, vec![vec![1, 256]]);
}

#[test]
fn test_compile_trace_to_plan_input_only() {
    let graph = ComputationGraph::from_nodes(vec![input_node(0, vec![2, 32])]);
    let plan = compile_trace_to_plan(&graph).expect("compile should succeed");
    assert_eq!(plan.input_shapes, vec![vec![2, 32]]);
}

#[test]
fn test_compile_trace_configured_disables_peephole() {
    let graph = ComputationGraph::from_nodes(vec![
        input_node(0, vec![1, 64]),
        test_node(1, "relu", TraceOp::Relu, vec![0], vec![1, 64]),
    ]);
    let disabled_cfg = PeepholeConfig {
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
        auto_fuse_elementwise: false,
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
    let plan =
        compile_trace_to_plan_configured(&graph, &disabled_cfg).expect("compile should succeed");
    assert!(!plan.steps.is_empty());
}

// ===========================================================================
// Section 14: IR node ID consistency
// ===========================================================================

#[test]
fn test_node_id_new_and_index() {
    let id = NodeId::new(42);
    assert_eq!(id.index(), 42);
}

#[test]
fn test_node_id_equality() {
    let a = NodeId::new(7);
    let b = NodeId::new(7);
    let c = NodeId::new(8);
    assert_eq!(a, b);
    assert_ne!(a, c);
}

#[test]
fn test_tensor_node_id_new() {
    let id = TensorNodeId::new(5);
    assert_eq!(id, TensorNodeId::new(5));
}

// ===========================================================================
// Section 15: ValueType properties
// ===========================================================================

#[test]
fn test_value_type_numeric_for_f32() {
    use crate::ir::ValueType;
    assert!(ValueType::F32.is_numeric());
}

#[test]
fn test_value_type_numeric_for_f16() {
    use crate::ir::ValueType;
    assert!(ValueType::F16.is_numeric());
}

#[test]
fn test_value_type_bool_not_numeric() {
    use crate::ir::ValueType;
    assert!(!ValueType::Bool.is_numeric());
}

#[test]
fn test_value_type_from_scalar_type() {
    use crate::ir::ValueType;
    assert_eq!(ValueType::from(ScalarType::F32), ValueType::F32);
    assert_eq!(ValueType::from(ScalarType::F16), ValueType::F16);
    assert_eq!(ValueType::from(ScalarType::BF16), ValueType::BF16);
}

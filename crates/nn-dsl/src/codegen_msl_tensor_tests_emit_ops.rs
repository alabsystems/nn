// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Op-specific MSL emission tests: Conv1d, BinaryAdd, Linear, GeluErf.
//!
//! Extracted from `codegen_msl_tensor_tests_emit.rs` to keep both files under
//! the 500-line limit.

use crate::codegen_msl_tensor::DispatchStep;
use crate::codegen_msl_tensor_emit::emit_tensor_msl;
use crate::ir::ScalarType;
use crate::tensor_ir::{TensorKernelDef, TensorNode, TensorNodeId, TensorOpKind};

// ===========================================================================
// Conv1d MSL emission tests
// ===========================================================================

/// `emit_tensor_msl` emits a Conv1d kernel through the full dispatch plan pipeline.
#[test]
fn test_emit_tensor_msl_conv1d_basic() {
    use crate::conv1d::build_conv1d;
    let def = build_conv1d("conv1d_emit", 4, 2, 3, 8, 1, 0, false).expect("build");
    let msl = emit_tensor_msl(&def, ScalarType::F32).expect("Conv1d MSL emission");
    assert!(msl.starts_with("#include <metal_stdlib>"), "prelude");
    assert!(msl.contains("[[kernel]]"), "kernel attribute");
    assert!(
        msl.contains("conv1d_emit_conv1d_n"),
        "kernel name from dispatch plan"
    );
    assert!(msl.contains("IN_CH_PER_GROUP = 4"), "in_ch_per_group baked");
    assert!(msl.contains("OUT_CHANNELS = 2"), "out_channels baked");
    assert!(msl.contains("KERNEL_SIZE = 3"), "kernel_size baked");
}

/// `emit_tensor_msl` emits Conv1d with bias through the full pipeline.
#[test]
fn test_emit_tensor_msl_conv1d_with_bias() {
    use crate::conv1d::build_conv1d;
    let def = build_conv1d("conv1d_bias_emit", 4, 2, 3, 8, 1, 0, true).expect("build");
    let msl = emit_tensor_msl(&def, ScalarType::F32).expect("Conv1d MSL emission");
    assert!(msl.contains("bias"), "must reference bias in MSL");
    assert!(msl.contains("buffer(2)"), "bias at buffer(2)");
    assert!(msl.contains("buffer(3)"), "output at buffer(3)");
    assert!(msl.contains("buffer(4)"), "total at buffer(4)");
}

// ===========================================================================
// BinaryAdd MSL emission tests (#640)
// ===========================================================================

#[test]
fn test_binary_add_dispatch_plan() {
    use crate::codegen_msl_tensor::build_dispatch_plan;

    let def = TensorKernelDef::new(
        "add_plan",
        vec![
            TensorNode::new(
                TensorNodeId::new(0),
                TensorOpKind::Input {
                    name: "a".into(),
                    shape: vec![2, 4],
                },
                vec![2, 4],
            ),
            TensorNode::new(
                TensorNodeId::new(1),
                TensorOpKind::Input {
                    name: "b".into(),
                    shape: vec![2, 4],
                },
                vec![2, 4],
            ),
            TensorNode::new(
                TensorNodeId::new(2),
                TensorOpKind::BinaryAdd {
                    left: TensorNodeId::new(0),
                    right: TensorNodeId::new(1),
                },
                vec![2, 4],
            ),
        ],
        TensorNodeId::new(2),
    );
    let (plan, _) = build_dispatch_plan(&def, ScalarType::F32).expect("plan");
    assert_eq!(plan.len(), 1, "should have exactly 1 dispatch step");
    assert!(
        matches!(
            &plan[0],
            DispatchStep::BinaryAdd {
                total_elements: 8,
                left,
                right,
                output,
                ..
            } if *left == TensorNodeId::new(0) && *right == TensorNodeId::new(1) && *output == TensorNodeId::new(2)
        ),
        "expected BinaryAdd dispatch step with correct fields, got {:?}",
        &plan[0]
    );
}

#[test]
fn test_binary_add_msl_emission() {
    let def = TensorKernelDef::new(
        "add_emit",
        vec![
            TensorNode::new(
                TensorNodeId::new(0),
                TensorOpKind::Input {
                    name: "a".into(),
                    shape: vec![2, 4],
                },
                vec![2, 4],
            ),
            TensorNode::new(
                TensorNodeId::new(1),
                TensorOpKind::Input {
                    name: "b".into(),
                    shape: vec![2, 4],
                },
                vec![2, 4],
            ),
            TensorNode::new(
                TensorNodeId::new(2),
                TensorOpKind::BinaryAdd {
                    left: TensorNodeId::new(0),
                    right: TensorNodeId::new(1),
                },
                vec![2, 4],
            ),
        ],
        TensorNodeId::new(2),
    );
    let msl = emit_tensor_msl(&def, ScalarType::F32).expect("MSL emission");
    assert!(msl.contains("left"), "MSL must reference 'left' buffer");
    assert!(msl.contains("right"), "MSL must reference 'right' buffer");
    assert!(
        msl.contains("output[tid] = left[tid] + right[tid]"),
        "MSL must contain add expression"
    );
    assert!(msl.contains("8u"), "MSL must contain total_elements guard");
}

// ===========================================================================
// Linear MSL emission tests (#730 Direction 4)
// ===========================================================================

#[test]
fn test_emit_tensor_msl_linear_no_bias() {
    use crate::linear::build_linear;
    let def = build_linear("lin_emit", 4, 2, false).expect("build");
    let msl = emit_tensor_msl(&def, ScalarType::F32).expect("Linear MSL emission");
    assert!(msl.contains("[[kernel]]"), "kernel attribute");
    assert!(msl.contains("lin_emit_linear_n"), "kernel name");
    assert!(msl.contains("IN_FEATURES = 4"), "in_features");
    assert!(msl.contains("OUT_FEATURES = 2"), "out_features");
    assert!(!msl.contains("bias"), "no bias buffer for no-bias variant");
}

#[test]
fn test_emit_tensor_msl_linear_with_bias() {
    use crate::linear::build_linear;
    let def = build_linear("lin_bias", 4, 2, true).expect("build");
    let msl = emit_tensor_msl(&def, ScalarType::F32).expect("Linear MSL emission");
    assert!(msl.contains("bias"), "bias buffer present");
    assert!(msl.contains("buffer(2)"), "bias at buffer(2)");
    assert!(msl.contains("buffer(3)"), "output at buffer(3)");
    assert!(msl.contains("buffer(4)"), "total at buffer(4)");
    assert!(msl.contains("sum += bias[col]"), "bias addition");
}

// ===========================================================================
// GeluErf MSL emission tests (#2519)
// ===========================================================================

/// GeluErf dispatch plan produces a single GeluErf step.
#[test]
fn test_gelu_erf_dispatch_plan() {
    use crate::codegen_msl_tensor::build_dispatch_plan;

    let def = TensorKernelDef::new(
        "gelu_erf_plan",
        vec![
            TensorNode::new(
                TensorNodeId::new(0),
                TensorOpKind::Input {
                    name: "x".into(),
                    shape: vec![2, 8],
                },
                vec![2, 8],
            ),
            TensorNode::new(
                TensorNodeId::new(1),
                TensorOpKind::GeluErf {
                    input: TensorNodeId::new(0),
                },
                vec![2, 8],
            ),
        ],
        TensorNodeId::new(1),
    );
    let (plan, _) = build_dispatch_plan(&def, ScalarType::F32).expect("plan");
    assert_eq!(plan.len(), 1, "should have exactly 1 dispatch step");
    assert!(
        matches!(
            &plan[0],
            DispatchStep::GeluErf {
                total_elements: 16,
                ..
            }
        ),
        "expected GeluErf dispatch step, got {:?}",
        &plan[0]
    );
}

/// GeluErf MSL uses polynomial erf approximation, not `metal::precise::erf`.
///
/// Regression test for #2519: MSL contained `metal::precise::erf` (not a valid
/// Metal function) and `M_SQRT1_2` (a POSIX constant not defined in MSL).
#[test]
fn test_gelu_erf_msl_no_invalid_metal_calls() {
    let def = TensorKernelDef::new(
        "gelu_erf_emit",
        vec![
            TensorNode::new(
                TensorNodeId::new(0),
                TensorOpKind::Input {
                    name: "x".into(),
                    shape: vec![2, 8],
                },
                vec![2, 8],
            ),
            TensorNode::new(
                TensorNodeId::new(1),
                TensorOpKind::GeluErf {
                    input: TensorNodeId::new(0),
                },
                vec![2, 8],
            ),
        ],
        TensorNodeId::new(1),
    );
    let msl = emit_tensor_msl(&def, ScalarType::F32).expect("MSL emission");
    // Must NOT contain the invalid calls from the old codegen
    assert!(
        !msl.contains("metal::precise::erf"),
        "MSL must not call metal::precise::erf (does not exist in MSL)"
    );
    assert!(
        !msl.contains("M_SQRT1_2"),
        "MSL must not use M_SQRT1_2 (POSIX constant, not in MSL)"
    );
    // Must contain the polynomial approximation components
    assert!(
        msl.contains("0.7071067811865476"),
        "MSL must contain 1/sqrt(2) literal"
    );
    assert!(
        msl.contains("0.3275911"),
        "MSL must contain Abramowitz & Stegun p constant"
    );
    assert!(
        msl.contains("metal::precise::exp"),
        "MSL must use metal::precise::exp (valid MSL function)"
    );
    assert!(msl.contains("[[kernel]]"), "kernel attribute");
    assert!(msl.contains("16u"), "total_elements guard");
}

// ===========================================================================
// F16 float accumulator tests (#3250)
// ===========================================================================

/// Helper: build a single unary activation TensorKernelDef.
fn unary_activation_def(name: &str, op: TensorOpKind) -> TensorKernelDef {
    TensorKernelDef::new(
        name,
        vec![
            TensorNode::new(
                TensorNodeId::new(0),
                TensorOpKind::Input {
                    name: "x".into(),
                    shape: vec![2, 8],
                },
                vec![2, 8],
            ),
            TensorNode::new(TensorNodeId::new(1), op, vec![2, 8]),
        ],
        TensorNodeId::new(1),
    )
}

/// F16 sigmoid MSL must use float intermediates, not half.
#[test]
fn test_sigmoid_f16_uses_float_accumulator() {
    let def = unary_activation_def(
        "sig_f16",
        TensorOpKind::Sigmoid {
            input: TensorNodeId::new(0),
        },
    );
    let msl = emit_tensor_msl(&def, ScalarType::F16).expect("MSL emission");
    assert!(
        msl.contains("float x = float("),
        "F16 sigmoid should promote input to float, got:\n{msl}"
    );
    assert!(
        msl.contains("half("),
        "F16 sigmoid should demote output to half, got:\n{msl}"
    );
    assert!(
        msl.contains("device const half*"),
        "F16 sigmoid buffers should be half, got:\n{msl}"
    );
}

/// F16 GELU MSL must use float intermediates for cubic + exp.
#[test]
fn test_gelu_f16_uses_float_accumulator() {
    let def = unary_activation_def(
        "gelu_f16",
        TensorOpKind::Gelu {
            input: TensorNodeId::new(0),
        },
    );
    let msl = emit_tensor_msl(&def, ScalarType::F16).expect("MSL emission");
    assert!(
        msl.contains("float x = float("),
        "F16 GELU should promote input to float"
    );
    assert!(
        msl.contains("float inner ="),
        "F16 GELU intermediates should be float"
    );
}

/// F16 GeluErf MSL must use float intermediates for polynomial approximation.
#[test]
fn test_gelu_erf_f16_uses_float_accumulator() {
    let def = unary_activation_def(
        "gelu_erf_f16",
        TensorOpKind::GeluErf {
            input: TensorNodeId::new(0),
        },
    );
    let msl = emit_tensor_msl(&def, ScalarType::F16).expect("MSL emission");
    assert!(
        msl.contains("float x = float("),
        "F16 GeluErf should promote input to float"
    );
    assert!(
        msl.contains("float u ="),
        "F16 GeluErf intermediates should be float"
    );
    assert!(
        msl.contains("float erf_val ="),
        "F16 GeluErf erf_val should be float"
    );
}

/// F16 ELU MSL must use float intermediates for exp(x) - 1.
#[test]
fn test_elu_f16_uses_float_accumulator() {
    let def = unary_activation_def(
        "elu_f16",
        TensorOpKind::Elu {
            input: TensorNodeId::new(0),
            alpha: 1.0,
        },
    );
    let msl = emit_tensor_msl(&def, ScalarType::F16).expect("MSL emission");
    assert!(
        msl.contains("float x = float("),
        "F16 ELU should promote input to float"
    );
    assert!(
        msl.contains("half(select("),
        "F16 ELU should demote select result to half"
    );
}

/// F16 Softplus MSL must use float intermediates for log(1 + exp(x)).
#[test]
fn test_softplus_f16_uses_float_accumulator() {
    let def = unary_activation_def(
        "softplus_f16",
        TensorOpKind::Softplus {
            input: TensorNodeId::new(0),
        },
    );
    let msl = emit_tensor_msl(&def, ScalarType::F16).expect("MSL emission");
    assert!(
        msl.contains("float x = float("),
        "F16 Softplus should promote input to float"
    );
    assert!(
        msl.contains("half(metal::precise::log("),
        "F16 Softplus should demote log result to half"
    );
}

/// F32 sigmoid MSL is unchanged — both t and acc are "float".
#[test]
fn test_sigmoid_f32_unchanged() {
    let def = unary_activation_def(
        "sig_f32",
        TensorOpKind::Sigmoid {
            input: TensorNodeId::new(0),
        },
    );
    let msl = emit_tensor_msl(&def, ScalarType::F32).expect("MSL emission");
    assert!(
        msl.contains("device const float*"),
        "F32 sigmoid buffers should be float"
    );
    assert!(
        !msl.contains("half"),
        "F32 sigmoid should have no half type"
    );
}

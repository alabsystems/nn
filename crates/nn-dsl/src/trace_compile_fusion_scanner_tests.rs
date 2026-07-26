// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Tests for the data-driven fusion opportunity scanner.

use super::*;
use crate::tensor_ir::{TensorKernelDef, TensorNode, TensorNodeId, TensorOpKind};
use std::collections::HashMap;

use super::super::trace_compile_types::{CompiledKernel, CompiledPlan, CompiledStep};

/// Build a minimal CompiledStep::Dispatch with a given kernel name and shape.
fn make_dispatch(name: &str, shape: &[usize]) -> CompiledStep {
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

fn make_plan(steps: Vec<CompiledStep>) -> CompiledPlan {
    let output_step = if steps.is_empty() { 0 } else { steps.len() - 1 };
    CompiledPlan {
        steps,
        input_shapes: vec![],
        output_step,
        weight_names: vec![],
    }
}

#[test]
fn test_empty_plan() {
    let plan = make_plan(vec![]);
    let result = scan_fusion_opportunities(&plan);
    assert!(result.opportunities.is_empty());
    assert_eq!(result.total_dispatches, 0);
    assert_eq!(result.total_potential_savings, 0);
}

#[test]
fn test_single_dispatch_no_opportunities() {
    let plan = make_plan(vec![make_dispatch("relu", &[1, 4])]);
    let result = scan_fusion_opportunities(&plan);
    assert!(result.opportunities.is_empty());
    assert_eq!(result.total_dispatches, 1);
}

#[test]
fn test_elementwise_chain_detected() {
    let plan = make_plan(vec![
        make_dispatch("add", &[1, 4]),
        make_dispatch("mul", &[1, 4]),
        make_dispatch("exp", &[1, 4]),
    ]);
    let result = scan_fusion_opportunities(&plan);
    assert!(!result.opportunities.is_empty());

    // Should find at least the 2-step patterns and the 3-step pattern.
    let three_step: Vec<_> = result
        .opportunities
        .iter()
        .filter(|o| o.ops.len() == 3)
        .collect();
    assert!(
        !three_step.is_empty(),
        "should detect the 3-step add->mul->exp chain"
    );
    assert_eq!(three_step[0].ops, vec!["add", "mul", "exp"]);
    assert_eq!(three_step[0].instances, 1);
    assert_eq!(three_step[0].dispatch_savings, 2);
    assert_eq!(three_step[0].category, FusionCategory::ElementwiseChain);
}

#[test]
fn test_conv_activation_pattern() {
    let plan = make_plan(vec![
        make_dispatch("conv1d", &[1, 4, 16]),
        make_dispatch("relu", &[1, 4, 16]),
    ]);
    let result = scan_fusion_opportunities(&plan);
    assert_eq!(result.opportunities.len(), 1);
    assert_eq!(
        result.opportunities[0].category,
        FusionCategory::ConvActivation
    );
    assert_eq!(result.opportunities[0].dispatch_savings, 1);
}

#[test]
fn test_native_op_breaks_run() {
    let plan = make_plan(vec![
        make_dispatch("add", &[1, 4]),
        CompiledStep::NativeOp {
            op: super::super::NativeOpKind::InstanceNorm {
                eps: 1e-5,
                input_shape: vec![1, 4],
            },
            weight_data: HashMap::new(),
        },
        make_dispatch("mul", &[1, 4]),
    ]);
    let result = scan_fusion_opportunities(&plan);
    // The NativeOp should break the run, so add+mul are not adjacent.
    assert!(result.opportunities.is_empty());
}

#[test]
fn test_repeated_pattern_counted() {
    let plan = make_plan(vec![
        make_dispatch("conv1d", &[1, 4, 16]),
        make_dispatch("relu", &[1, 4, 16]),
        CompiledStep::IdentityPassthrough, // break
        make_dispatch("conv1d", &[1, 4, 16]),
        make_dispatch("relu", &[1, 4, 16]),
        CompiledStep::IdentityPassthrough, // break
        make_dispatch("conv1d", &[1, 4, 16]),
        make_dispatch("relu", &[1, 4, 16]),
    ]);
    let result = scan_fusion_opportunities(&plan);
    let conv_relu: Vec<_> = result
        .opportunities
        .iter()
        .filter(|o| o.ops == vec!["conv1d", "relu"])
        .collect();
    assert_eq!(conv_relu.len(), 1);
    assert_eq!(conv_relu[0].instances, 3);
    assert_eq!(conv_relu[0].dispatch_savings, 3);
}

#[test]
fn test_priority_ordering() {
    // 3x conv->relu (3 savings) should rank higher than 1x add->mul (1 saving).
    let plan = make_plan(vec![
        make_dispatch("conv1d", &[1, 4, 16]),
        make_dispatch("relu", &[1, 4, 16]),
        CompiledStep::IdentityPassthrough,
        make_dispatch("conv1d", &[1, 4, 16]),
        make_dispatch("relu", &[1, 4, 16]),
        CompiledStep::IdentityPassthrough,
        make_dispatch("conv1d", &[1, 4, 16]),
        make_dispatch("relu", &[1, 4, 16]),
        CompiledStep::IdentityPassthrough,
        make_dispatch("add", &[1, 4]),
        make_dispatch("mul", &[1, 4]),
    ]);
    let result = scan_fusion_opportunities(&plan);
    assert!(!result.opportunities.is_empty());
    // First opportunity should be the higher-priority one.
    assert!(
        result.opportunities[0].priority_score()
            >= result.opportunities.last().unwrap().priority_score()
    );
}

#[test]
fn test_linear_chain_detected() {
    let plan = make_plan(vec![
        make_dispatch("linear", &[1, 64]),
        make_dispatch("linear", &[1, 64]),
    ]);
    let result = scan_fusion_opportunities(&plan);
    assert_eq!(result.opportunities.len(), 1);
    assert_eq!(
        result.opportunities[0].category,
        FusionCategory::LinearChain
    );
}

#[test]
fn test_norm_linear_pattern() {
    let plan = make_plan(vec![
        make_dispatch("layer_norm", &[1, 64]),
        make_dispatch("linear", &[1, 64]),
    ]);
    let result = scan_fusion_opportunities(&plan);
    assert_eq!(result.opportunities.len(), 1);
    assert_eq!(result.opportunities[0].category, FusionCategory::NormLinear);
}

#[test]
fn test_display_formatting() {
    let opp = FusionOpportunity {
        steps: vec![0, 1],
        ops: vec!["conv1d".into(), "relu".into()],
        category: FusionCategory::ConvActivation,
        instances: 5,
        dispatch_savings: 5,
    };
    let display = format!("{opp}");
    assert!(display.contains("ConvActivation"));
    assert!(display.contains("conv1d -> relu"));
    assert!(display.contains("5x"));
    assert!(display.contains("saves 5 dispatches"));
}

#[test]
fn test_summarize() {
    let result = FusionScanResult {
        opportunities: vec![FusionOpportunity {
            steps: vec![0, 1],
            ops: vec!["add".into(), "mul".into()],
            category: FusionCategory::ElementwiseChain,
            instances: 10,
            dispatch_savings: 10,
        }],
        total_dispatches: 100,
        total_potential_savings: 10,
    };
    let summary = result.summarize();
    assert!(summary.contains("100 dispatches"));
    assert!(summary.contains("1 opportunities"));
    assert!(summary.contains("10 potential savings"));
}

#[test]
fn test_passthrough_ignored() {
    let plan = make_plan(vec![
        make_dispatch("add", &[1, 4]),
        CompiledStep::Passthrough {
            op_name: "reshape".into(),
            output_shape: vec![1, 4],
        },
        make_dispatch("mul", &[1, 4]),
    ]);
    let result = scan_fusion_opportunities(&plan);
    // Passthrough breaks the run.
    assert!(result.opportunities.is_empty());
}

#[test]
fn test_four_step_pattern() {
    let plan = make_plan(vec![
        make_dispatch("instance_norm", &[1, 4, 16]),
        make_dispatch("mul", &[1, 4, 16]),
        make_dispatch("add", &[1, 4, 16]),
        make_dispatch("snake_tensor", &[1, 4, 16]),
    ]);
    let result = scan_fusion_opportunities(&plan);
    let four_step: Vec<_> = result
        .opportunities
        .iter()
        .filter(|o| o.ops.len() == 4)
        .collect();
    // norm->mul is fusible, mul->add is fusible, add->activation is fusible.
    assert!(
        !four_step.is_empty(),
        "should detect the 4-step norm->mul->add->snake pattern"
    );
    assert_eq!(four_step[0].dispatch_savings, 3);
}

#[test]
fn test_classify_pattern_conv_norm() {
    let classes = vec![OpClass::Conv, OpClass::Norm];
    assert_eq!(classify_pattern(&classes), FusionCategory::ConvNorm);
}

#[test]
fn test_classify_pattern_activation_conv() {
    let classes = vec![OpClass::Activation, OpClass::Conv];
    assert_eq!(classify_pattern(&classes), FusionCategory::ActivationConv);
}

#[test]
fn test_classify_pattern_three_step_norm_activ_conv() {
    let classes = vec![OpClass::Norm, OpClass::Activation, OpClass::Conv];
    assert_eq!(classify_pattern(&classes), FusionCategory::NormActivConv);
}

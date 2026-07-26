// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Integration tests for `TensorOpKind::Gelu` → NY `GELULayer`.
//!
//! Tests:
//! - Single-variable IBP bounds propagation
//! - Constant-folding
//! - TensorBlockBuilder::add_gelu round-trip
//! - MSL codegen dispatch plan

use nn_dsl::tensor_block_builder::TensorBlockBuilder;
use nn_dsl::tensor_ir::{TensorKernelDef, TensorNode, TensorNodeId, TensorOpKind};
use nn_verify::{tensor_kernel_to_graph, BoundedTensor, TensorParamBinding};
use ndarray::{ArrayD, IxDyn};

/// Build a simple GELU kernel: one input, output = gelu(input).
fn gelu_kernel(name: &str, shape: &[usize]) -> TensorKernelDef {
    TensorKernelDef::new(
        name,
        vec![
            TensorNode::new(
                TensorNodeId::new(0),
                TensorOpKind::Input {
                    name: "x".into(),
                    shape: shape.to_vec(),
                },
                shape.to_vec(),
            ),
            TensorNode::new(
                TensorNodeId::new(1),
                TensorOpKind::Gelu {
                    input: TensorNodeId::new(0),
                },
                shape.to_vec(),
            ),
        ],
        TensorNodeId::new(1),
    )
}

#[test]
fn test_gelu_variable_builds_graph() {
    let def = gelu_kernel("gelu_test", &[4, 32]);
    let graph = tensor_kernel_to_graph(&def, &[TensorParamBinding::Variable]).expect("gelu graph");
    assert!(
        graph.num_nodes() >= 1,
        "graph should have at least the GELU node"
    );
}

#[test]
fn test_gelu_ibp_bounds_positive_range() {
    // GELU is monotonic for x > 0: gelu(1) ~ 0.841, gelu(3) ~ 2.996
    // IBP bounds should be: [gelu(1), gelu(3)]
    let shape = &[2, 4];
    let def = gelu_kernel("ibp_gelu_pos", shape);
    let graph =
        tensor_kernel_to_graph(&def, &[TensorParamBinding::Variable]).expect("build gelu graph");

    let lower = ArrayD::from_elem(IxDyn(shape), 1.0f32);
    let upper = ArrayD::from_elem(IxDyn(shape), 3.0f32);
    let input = BoundedTensor::new(lower, upper).expect("input bounds");
    let output = graph.propagate_ibp(&input).expect("IBP propagation");

    // Observed IBP output: lower=0.841192, upper=2.9963627
    // GELU(1) ≈ 0.8412, GELU(3) ≈ 2.9960 — IBP is near-exact for monotonic range.
    let out_lower = output.lower();
    let out_upper = output.upper();
    for &v in out_lower.iter() {
        assert!(
            v > 0.69 && v < 0.99,
            "expected lower ~0.84 (±0.15), got {v}"
        );
    }
    for &v in out_upper.iter() {
        assert!(v > 2.85 && v < 3.15, "expected upper ~3.0 (±0.15), got {v}");
    }
}

#[test]
fn test_gelu_ibp_bounds_spanning_origin() {
    // GELU has a global minimum at x ~ -0.752.
    // For range [-2, 2], bounds should include the minimum.
    let shape = &[2, 4];
    let def = gelu_kernel("ibp_gelu_span", shape);
    let graph =
        tensor_kernel_to_graph(&def, &[TensorParamBinding::Variable]).expect("build gelu graph");

    let lower = ArrayD::from_elem(IxDyn(shape), -2.0f32);
    let upper = ArrayD::from_elem(IxDyn(shape), 2.0f32);
    let input = BoundedTensor::new(lower, upper).expect("input bounds");
    let output = graph.propagate_ibp(&input).expect("IBP propagation");

    // Observed IBP output: lower=-0.17004077, upper=1.9545977
    // GELU minimum ≈ -0.170 at x ≈ -0.752; GELU(2) ≈ 1.955
    let out_lower = output.lower();
    let out_upper = output.upper();
    // AC1: Lower bound captures GELU minimum region, not just "any negative"
    for &v in out_lower.iter() {
        assert!(
            v > -0.35 && v < 0.0,
            "expected lower in (-0.35, 0.0) capturing GELU minimum ~-0.170, got {v}"
        );
    }
    // AC2: Upper bound window ≤ 0.3 (was 1.0-wide)
    for &v in out_upper.iter() {
        assert!(v > 1.8 && v < 2.1, "expected upper ~1.955 (±0.15), got {v}");
    }
}

#[test]
fn test_gelu_constant_fold() {
    let shape = &[2, 4];
    let def = gelu_kernel("const_gelu", shape);
    let graph = tensor_kernel_to_graph(&def, &[TensorParamBinding::ConstantScalar(0.0)])
        .expect("constant-fold gelu should succeed");

    // GELU(0) = 0
    let lower = ArrayD::from_elem(IxDyn(shape), 0.0f32);
    let upper = ArrayD::from_elem(IxDyn(shape), 0.0f32);
    let input = BoundedTensor::new(lower, upper).expect("input bounds");
    let output = graph.propagate_ibp(&input).expect("IBP propagation");
    for &v in output.lower().iter() {
        assert!(v.abs() < 1e-5, "expected gelu(0) ~ 0.0, got {v}");
    }
}

#[test]
fn test_gelu_builder_round_trip() {
    // Build using TensorBlockBuilder and verify it validates + builds graph.
    let mut b = TensorBlockBuilder::new("gelu_builder");
    let x = b.add_input("x", &[4, 32]);
    let y = b.add_gelu(x, &[4, 32]);
    let def = b.build(y).expect("valid graph");

    assert_eq!(def.name, "gelu_builder");
    assert_eq!(def.nodes.len(), 2);

    let graph = tensor_kernel_to_graph(&def, &[TensorParamBinding::Variable]).expect("build graph");
    assert!(graph.num_nodes() >= 1);
}

#[test]
fn test_gelu_msl_dispatch_plan() {
    use nn_dsl::ir::ScalarType;
    use nn_dsl::{build_dispatch_plan, DispatchStep};

    let def = gelu_kernel("msl_gelu", &[4, 32]);
    let (plan, _) = build_dispatch_plan(&def, ScalarType::F32).expect("dispatch plan");

    assert_eq!(plan.len(), 1, "should have exactly 1 dispatch step");
    match &plan[0] {
        DispatchStep::Gelu {
            total_elements,
            input,
            output,
            ..
        } => {
            assert_eq!(*total_elements, 4 * 32);
            assert_eq!(*input, TensorNodeId::new(0));
            assert_eq!(*output, TensorNodeId::new(1));
        }
        other => panic!("expected DispatchStep::Gelu, got {other:?}"),
    }
}

#[test]
fn test_gelu_msl_emission_uses_exp_form() {
    use nn_dsl::emit_tensor_msl;
    use nn_dsl::ir::ScalarType;

    let def = gelu_kernel("emit_gelu", &[4, 32]);
    let msl = emit_tensor_msl(&def, ScalarType::F32).expect("MSL emission");

    assert!(
        msl.contains("exp("),
        "MSL should use exp() form, not tanh() (#679)"
    );
    assert!(
        !msl.contains("tanh("),
        "MSL should NOT use tanh() — must match scalar reference (#679)"
    );
    assert!(
        msl.contains("0.7978845608"),
        "MSL should contain sqrt(2/pi) constant"
    );
    assert!(
        msl.contains("0.044715"),
        "MSL should contain GELU coefficient"
    );
    assert!(
        msl.contains("[[kernel]]"),
        "MSL should contain kernel attribute"
    );
}

#[test]
fn test_gelu_pretty_print() {
    use nn_dsl::tensor_ir_pretty_print;

    let def = gelu_kernel("pretty_gelu", &[4, 32]);
    let printed = tensor_ir_pretty_print(&def);
    assert!(
        printed.contains("gelu(%0)"),
        "pretty print should show gelu(%0), got: {printed}"
    );
}

// -- GeluErf constant-fold tests (#2311) ---------------------------------------

/// Build a GeluErf kernel: one input, output = gelu_erf(input).
fn gelu_erf_kernel(name: &str, shape: &[usize]) -> TensorKernelDef {
    TensorKernelDef::new(
        name,
        vec![
            TensorNode::new(
                TensorNodeId::new(0),
                TensorOpKind::Input {
                    name: "x".into(),
                    shape: shape.to_vec(),
                },
                shape.to_vec(),
            ),
            TensorNode::new(
                TensorNodeId::new(1),
                TensorOpKind::GeluErf {
                    input: TensorNodeId::new(0),
                },
                shape.to_vec(),
            ),
        ],
        TensorNodeId::new(1),
    )
}

/// GeluErf constant-fold at x=0: gelu_erf(0) = 0.
#[test]
fn test_gelu_erf_constant_fold_zero() {
    let shape = &[2, 4];
    let def = gelu_erf_kernel("const_gelu_erf_zero", shape);
    let graph = tensor_kernel_to_graph(&def, &[TensorParamBinding::ConstantScalar(0.0)])
        .expect("constant-fold gelu_erf(0) should succeed");

    let lower = ArrayD::from_elem(IxDyn(shape), 0.0f32);
    let upper = ArrayD::from_elem(IxDyn(shape), 0.0f32);
    let input = BoundedTensor::new(lower, upper).expect("input bounds");
    let output = graph.propagate_ibp(&input).expect("IBP propagation");
    for &v in output.lower().iter() {
        assert!(v.abs() < 1e-5, "expected gelu_erf(0) ~ 0.0, got {v}");
    }
}

/// GeluErf constant-fold at x=1: gelu_erf(1) ≈ 0.8413.
#[test]
fn test_gelu_erf_constant_fold_positive() {
    let shape = &[2, 4];
    let def = gelu_erf_kernel("const_gelu_erf_pos", shape);
    let graph = tensor_kernel_to_graph(&def, &[TensorParamBinding::ConstantScalar(1.0)])
        .expect("constant-fold gelu_erf(1) should succeed");

    let lower = ArrayD::from_elem(IxDyn(shape), 0.0f32);
    let upper = ArrayD::from_elem(IxDyn(shape), 0.0f32);
    let input = BoundedTensor::new(lower, upper).expect("input bounds");
    let output = graph.propagate_ibp(&input).expect("IBP propagation");
    // gelu_erf(1.0) = 0.5 * 1.0 * (1 + erf(1/sqrt(2))) ≈ 0.5 * 1.0 * 1.6827 ≈ 0.8413
    for &v in output.lower().iter() {
        assert!(
            (v - 0.8413).abs() < 0.01,
            "expected gelu_erf(1.0) ~ 0.8413, got {v}"
        );
    }
}

/// GeluErf constant-fold at x=-1: gelu_erf(-1) ≈ -0.1587.
#[test]
fn test_gelu_erf_constant_fold_negative() {
    let shape = &[2, 4];
    let def = gelu_erf_kernel("const_gelu_erf_neg", shape);
    let graph = tensor_kernel_to_graph(&def, &[TensorParamBinding::ConstantScalar(-1.0)])
        .expect("constant-fold gelu_erf(-1) should succeed");

    let lower = ArrayD::from_elem(IxDyn(shape), 0.0f32);
    let upper = ArrayD::from_elem(IxDyn(shape), 0.0f32);
    let input = BoundedTensor::new(lower, upper).expect("input bounds");
    let output = graph.propagate_ibp(&input).expect("IBP propagation");
    // gelu_erf(-1.0) = 0.5 * (-1.0) * (1 + erf(-1/sqrt(2))) ≈ -0.1587
    for &v in output.lower().iter() {
        assert!(
            (v - (-0.1587)).abs() < 0.01,
            "expected gelu_erf(-1.0) ~ -0.1587, got {v}"
        );
    }
}

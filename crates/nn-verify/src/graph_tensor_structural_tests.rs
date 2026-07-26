// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

use super::*;
use crate::graph::FiniteF32;
use crate::graph_tensor::TensorParamBinding;
use nn_dsl::tensor_ir::{TensorNode, TensorOpKind};

/// Helper: build a minimal TensorTranslationContext for tests.
fn test_ctx<'a>(
    bindings: &'a [TensorParamBinding],
    names: &'a [Option<String>],
    nodes: &'a [TensorNode],
) -> TensorTranslationContext<'a> {
    TensorTranslationContext {
        input_bindings: bindings,
        input_node_names: names,
        axis_offset: 0,
        all_nodes: nodes,
        norm_mode: crate::verify_types::NormBoundsMode::Conservative,
    }
}

/// Stack with a constant input in second position must return
/// UnsupportedOp error, not silently drop the constant (#270).
#[test]
fn test_translate_stack_rejects_constant_input() {
    let nodes = vec![
        TensorNode::new(
            TensorNodeId::new(0),
            TensorOpKind::Input {
                name: "x".into(),
                shape: vec![2, 3],
            },
            vec![2, 3],
        ),
        TensorNode::new(
            TensorNodeId::new(1),
            TensorOpKind::Input {
                name: "y".into(),
                shape: vec![2, 3],
            },
            vec![2, 3],
        ),
    ];
    let bindings = [
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantScalar(1.0),
    ];
    let names = [Some("input_0".to_string()), None];
    let ctx = test_ctx(&bindings, &names, &nodes);

    let node_values = vec![
        TensorNodeValue::Variable("input_0".into()),
        TensorNodeValue::Constant(FiniteF32::new(1.0).expect("finite")),
    ];

    let mut graph = GraphNetwork::new();
    let result = translate_stack(
        &ctx,
        TensorNodeId::new(2),
        &[TensorNodeId::new(0), TensorNodeId::new(1)],
        1,
        &node_values,
        &mut graph,
    );

    assert!(result.is_err(), "Stack with constant input must fail");
    let err_msg = result.unwrap_err().to_string();
    assert!(
        err_msg.contains("constant"),
        "error should mention 'constant', got: {err_msg}"
    );
}

// (Removed test_translate_reshape_empty_shape_returns_error: it exercised the
// old `axis_offset > 0` Reshape stacking branch, which no longer exists after the
// multi-variable harness was switched to per-variable flat Slice+Reshape.)

/// Stack with all-variable inputs succeeds normally.
#[test]
fn test_translate_stack_two_variables_succeeds() {
    let nodes = vec![
        TensorNode::new(
            TensorNodeId::new(0),
            TensorOpKind::Input {
                name: "a".into(),
                shape: vec![2, 3],
            },
            vec![2, 3],
        ),
        TensorNode::new(
            TensorNodeId::new(1),
            TensorOpKind::Input {
                name: "b".into(),
                shape: vec![2, 3],
            },
            vec![2, 3],
        ),
    ];
    let bindings = [TensorParamBinding::Variable, TensorParamBinding::Variable];
    let names = [Some("input_0".to_string()), Some("input_1".to_string())];
    let ctx = test_ctx(&bindings, &names, &nodes);

    let mut graph = GraphNetwork::new();
    let node_values = vec![
        TensorNodeValue::Variable("input_0".into()),
        TensorNodeValue::Variable("input_1".into()),
    ];

    let result = translate_stack(
        &ctx,
        TensorNodeId::new(2),
        &[TensorNodeId::new(0), TensorNodeId::new(1)],
        1,
        &node_values,
        &mut graph,
    );

    assert!(
        result.is_ok(),
        "Stack with 2 variable inputs should succeed: {result:?}"
    );
}

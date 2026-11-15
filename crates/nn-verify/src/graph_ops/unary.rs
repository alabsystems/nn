// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! UnaryFn translation: constant fold or emit NY layer.

use ny_propagate::layers::{
    AbsLayer, CosLayer, ExpLayer, ReciprocalLayer, SinLayer, SqrtLayer, TanhLayer,
};
use ny_propagate::{GraphNetwork, Layer};
use nn_dsl::ir::UnaryFnKind;

use crate::error::VerifyError;
use crate::graph::{add_unary_node, checked_constant, NodeValue};

/// Evaluate a unary function on a constant scalar value.
/// Returns the result, which may be non-finite (e.g., recip(0) = inf, sqrt(-1) = NaN).
fn evaluate_constant_unary(op: UnaryFnKind, val: f32) -> Result<f32, VerifyError> {
    match op {
        UnaryFnKind::Sin => Ok(val.sin()),
        UnaryFnKind::Cos => Ok(val.cos()),
        UnaryFnKind::Sqrt => Ok(val.sqrt()),
        UnaryFnKind::Rsqrt => Ok(1.0 / val.sqrt()),
        UnaryFnKind::Exp => Ok(val.exp()),
        UnaryFnKind::Abs => Ok(val.abs()),
        UnaryFnKind::Recip => Ok(1.0 / val),
        UnaryFnKind::Tanh => Ok(val.tanh()),
        UnaryFnKind::Log => Ok(val.ln()),
        _ => Err(VerifyError::UnsupportedOp(format!("UnaryFn {op:?}"))),
    }
}

/// Translate a `UnaryFn` node to a NY layer or constant fold.
pub(crate) fn translate_unary(
    name: &str,
    op: UnaryFnKind,
    input: &NodeValue,
    graph: &mut GraphNetwork,
) -> Result<NodeValue, VerifyError> {
    match input {
        NodeValue::Constant(v) => {
            let val = v.get();
            let result = evaluate_constant_unary(op, val)?;
            checked_constant(result, &format!("{op:?}({val})"))
        }
        NodeValue::Variable(var_name) => {
            let layer = match op {
                UnaryFnKind::Sin => Layer::Sin(SinLayer::new()),
                UnaryFnKind::Cos => Layer::Cos(CosLayer::new()),
                UnaryFnKind::Sqrt => Layer::Sqrt(SqrtLayer::new()),
                UnaryFnKind::Exp => Layer::Exp(ExpLayer::new()),
                UnaryFnKind::Abs => Layer::Abs(AbsLayer::new()),
                UnaryFnKind::Recip => Layer::Reciprocal(ReciprocalLayer::new()),
                UnaryFnKind::Tanh => Layer::Tanh(TanhLayer::new()),
                UnaryFnKind::Rsqrt => {
                    let sqrt_name = format!("{name}_sqrt");
                    add_unary_node(&sqrt_name, Layer::Sqrt(SqrtLayer::new()), var_name, graph);
                    add_unary_node(
                        name,
                        Layer::Reciprocal(ReciprocalLayer::new()),
                        &sqrt_name,
                        graph,
                    );
                    return Ok(NodeValue::Variable(name.to_string()));
                }
                _ => return Err(VerifyError::UnsupportedOp(format!("UnaryFn {op:?}"))),
            };
            add_unary_node(name, layer, var_name, graph);
            Ok(NodeValue::Variable(name.to_string()))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Part of #665 AC4: verify tanh constant fold at known values.
    #[test]
    fn test_evaluate_constant_unary_tanh_known_values() {
        let result_zero =
            evaluate_constant_unary(UnaryFnKind::Tanh, 0.0).expect("Tanh(0) must succeed");
        assert!(
            (result_zero - 0.0).abs() < 1e-7,
            "tanh(0) must equal 0, got {result_zero}",
        );

        let result_one =
            evaluate_constant_unary(UnaryFnKind::Tanh, 1.0).expect("Tanh(1) must succeed");
        assert!(
            (result_one - 0.7616).abs() < 0.001,
            "tanh(1) must be ~0.7616, got {result_one}",
        );

        let result_neg =
            evaluate_constant_unary(UnaryFnKind::Tanh, -1.0).expect("Tanh(-1) must succeed");
        assert!(
            (result_neg + 0.7616).abs() < 0.001,
            "tanh(-1) must be ~-0.7616, got {result_neg}",
        );
    }
}

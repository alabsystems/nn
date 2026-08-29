// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Native NY layer dispatch for known kernel patterns.
//!
//! When a kernel matches a known activation (Snake, SiLU-Mul, etc.), we emit
//! a single native NY layer instead of decomposing into primitive
//! operations. Native layers produce tighter bounds because they exploit
//! mathematical properties of the composite function.
//!
//! Extracted from graph.rs (#422) to isolate the growing dispatch table.

use nn_dsl::ir::KernelDef;
use ny_propagate::layers::{
    GELULayer, GeluApproximation, MulConstantLayer, SiLULayer, SigmoidLayer, SnakeLayer,
};
use ny_propagate::{GraphNetwork, GraphNode, Layer};

use super::ParamBinding;
use crate::error::VerifyError;

/// Try to replace a kernel with a single native NY layer.
///
/// Returns `Ok(Some(graph))` if the kernel matches a known native layer pattern,
/// `Ok(None)` if decomposition is needed, or `Err` on invalid parameters.
///
/// Native layers produce tighter bounds than decomposition because they exploit
/// mathematical properties of the composite function. For example, Snake's
/// monotonicity (`f'(x) = 1 + sin(2ax) >= 0`) yields exact IBP bounds
/// `[f(l), f(u)]`, while the decomposed Sin→Pow→Mul→Add path loses this
/// because Sin alone is non-monotone.
///
/// ## CROWN-IBP identity for monotone native layers (#489)
///
/// When a native layer produces **exact** IBP bounds (as Snake does via
/// monotonicity), CROWN cannot tighten further. CROWN's advantage comes from
/// capturing inter-layer correlation in multi-layer networks. For a single
/// native layer, there are no inter-layer correlations to exploit, so CROWN
/// converges to the same bounds as IBP. This is mathematically expected, not
/// a bug. CROWN provides value in **multi-layer** contexts like fusion
/// diamond DAGs where IBP loses input correlation across paths.
pub(crate) fn try_native_layer(
    kernel: &KernelDef,
    bindings: &[ParamBinding],
) -> Result<Option<GraphNetwork>, VerifyError> {
    // Snake: f(x, alpha) = x + (1/alpha) * sin²(alpha * x)
    // Requires: 2 params (variable x, constant alpha). The alpha handed to ny
    // is nn's RUNTIME-clamped value (see below); only non-finite alpha falls
    // through to the decomposed path.
    if kernel.name == "snake"
        && bindings.len() == 2
        && matches!(bindings[0], ParamBinding::Variable)
    {
        if let ParamBinding::Constant(alpha) = bindings[1] {
            // nn's runtime kernel clamps alpha (`DynTensor::snake`:
            // `alpha.clamp(1e-8, 1e6)`, matching GPU/DSL `SNAKE_MIN_ALPHA`), so
            // nn NEVER evaluates snake at a raw negative or out-of-range alpha.
            // Verification must model what nn RUNS: hand ny the clamped value.
            //
            // This became load-bearing when ny gained certified negative-alpha
            // Snake (ny 7dea51cce). Before that, `SnakeLayer::new` refused
            // negatives and this fell through to the decomposed path, which
            // applies the IR's own `max(alpha, 1e-8)` node — sound, but only by
            // accident. Passing the raw alpha now would verify
            // `x + (1/-a)sin^2(-a x)` while nn computes the clamped function:
            // a false proof about a different function.
            let runtime_alpha = alpha.clamp(1e-8, 1e6);
            if let Ok(snake) = SnakeLayer::new(runtime_alpha) {
                let mut graph = GraphNetwork::new();
                let node_name = "snake_native";
                graph.add_node(GraphNode::from_input(
                    node_name.to_string(),
                    Layer::Snake(snake),
                ));
                graph.set_output(node_name.to_string());
                return Ok(Some(graph));
            }
            // Non-finite alpha (NaN/inf) survives the clamp and is still
            // refused by ny: fall through to the decomposed path.
        }
    }

    // SiLU-Mul: silu_mul(x, up) = silu(x) * up
    // Requires: 2 params (variable x, constant up).
    // Emits SiLULayer (x * sigmoid(x)) + MulConstant(up).
    // Multi-variable case (both x and up variable) falls through to decomposition.
    if kernel.name == "silu_mul"
        && bindings.len() == 2
        && matches!(bindings[0], ParamBinding::Variable)
    {
        if let ParamBinding::Constant(up) = bindings[1] {
            // Defense-in-depth: reject non-finite `up` even though callers
            // validate bindings (kernel_to_graph_multi checks finiteness).
            if !up.is_finite() {
                return Ok(None);
            }
            let mut graph = GraphNetwork::new();
            let silu_name = "silu_native";
            graph.add_node(GraphNode::from_input(
                silu_name.to_string(),
                Layer::SiLU(SiLULayer::new()),
            ));
            if (up - 1.0).abs() > f32::EPSILON {
                // Multiply by up only when up != 1.0
                let mul_name = "silu_mul_native";
                graph.add_node(GraphNode::new(
                    mul_name.to_string(),
                    Layer::MulConstant(MulConstantLayer::scalar(up)),
                    vec![silu_name.to_string()],
                ));
                graph.set_output(mul_name.to_string());
            } else {
                graph.set_output(silu_name.to_string());
            }
            return Ok(Some(graph));
        }
    }

    // GELU (tanh): gelu(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    // Requires: 1 param (variable x).
    // Emits single GELULayer with Tanh approximation.
    if kernel.name == "gelu" && bindings.len() == 1 && matches!(bindings[0], ParamBinding::Variable)
    {
        let mut graph = GraphNetwork::new();
        let node_name = "gelu_native";
        graph.add_node(GraphNode::from_input(
            node_name.to_string(),
            Layer::GELU(GELULayer::new(GeluApproximation::Tanh)),
        ));
        graph.set_output(node_name.to_string());
        return Ok(Some(graph));
    }

    // GELU (erf): gelu_erf(x) = 0.5 * x * (1 + erf(x / sqrt(2)))
    // Requires: 1 param (variable x).
    // Emits single GELULayer with exact Erf formula (#2247).
    if kernel.name == "gelu_erf"
        && bindings.len() == 1
        && matches!(bindings[0], ParamBinding::Variable)
    {
        let mut graph = GraphNetwork::new();
        let node_name = "gelu_erf_native";
        graph.add_node(GraphNode::from_input(
            node_name.to_string(),
            Layer::GELU(GELULayer::new(GeluApproximation::Erf)),
        ));
        graph.set_output(node_name.to_string());
        return Ok(Some(graph));
    }

    // Sigmoid: sigmoid(x) = 1 / (1 + exp(-x))
    // Requires: 1 param (variable x).
    // Emits single SigmoidLayer. Monotonically increasing → exact IBP bounds.
    if kernel.name == "sigmoid"
        && bindings.len() == 1
        && matches!(bindings[0], ParamBinding::Variable)
    {
        let mut graph = GraphNetwork::new();
        let node_name = "sigmoid_native";
        graph.add_node(GraphNode::from_input(
            node_name.to_string(),
            Layer::Sigmoid(SigmoidLayer::new()),
        ));
        graph.set_output(node_name.to_string());
        return Ok(Some(graph));
    }

    Ok(None)
}

#[cfg(test)]
#[path = "graph_native_tests.rs"]
mod tests;

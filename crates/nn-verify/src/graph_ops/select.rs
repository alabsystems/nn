// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Select/Where translation with activation pattern matching (ReLU, LeakyReLU).

use nn_dsl::ir::{BinOpKind, CompareOpKind, IRNode, IRNodeKind};
use ny_propagate::layers::{
    AddConstantLayer, AddLayer, MulBinaryLayer, MulConstantLayer, ReLULayer, SubLayer, WhereLayer,
};
use ny_propagate::{GraphNetwork, GraphNode, Layer};

use crate::error::VerifyError;
use crate::graph::{add_unary_node, scalar_array, NodeValue};
use crate::util::get_value;

/// Evaluate a constant condition for Select: returns `true` if the then-branch
/// should be taken (positive condition), `false` for else-branch.
fn select_condition_is_true(cond: f32) -> bool {
    cond > 0.0
}

/// Translate a Select node, with pattern matching for known activations.
pub(crate) fn translate_select(
    name: &str,
    cond_idx: usize,
    then_idx: usize,
    else_idx: usize,
    all_nodes: &[IRNode],
    values: &[NodeValue],
    graph: &mut GraphNetwork,
) -> Result<NodeValue, VerifyError> {
    // Try pattern matching for specialized activation layers
    if let Some(result) =
        try_match_activation_pattern(name, cond_idx, then_idx, else_idx, all_nodes, values, graph)
    {
        return result;
    }

    let cond_val = get_value(values, cond_idx, "Select condition")?;
    let then_v = get_value(values, then_idx, "Select then")?;
    let else_v = get_value(values, else_idx, "Select else")?;

    // Constant condition: select branch at translation time.
    //
    // Convention: positive condition means "true" (then-branch). This aligns
    // with compare.rs which folds Ge(5,5) to 1.0 and Gt(5,5) to 0.0.
    //
    // The constant-fold path in compare.rs already handles Ge/Le equality
    // correctly (producing 1.0), so a constant 0.0 here is definitively "false."
    // Variable comparisons produce NodeValue::Variable (subtraction graph node),
    // not NodeValue::Constant, so the equality boundary case is handled by the
    // WhereLayer path below, not here.
    if let NodeValue::Constant(c) = cond_val {
        return if select_condition_is_true(c.get()) {
            Ok(then_v.clone())
        } else {
            Ok(else_v.clone())
        };
    }

    // General case: WhereLayer with 3 inputs [cond, then, else]
    // Constant condition is handled by the early return above. If a future
    // refactor removes that early return, this match produces a clear error
    // instead of a panic.
    let cond_name = match cond_val {
        NodeValue::Variable(v) => v.clone(),
        NodeValue::Constant(_) => {
            return Err(VerifyError::InternalTranslationError {
                context: format!(
                    "Select node `{name}`: constant condition should have been folded"
                ),
            });
        }
    };
    let then_name = ensure_variable_node(&format!("{name}_then"), then_v, graph)?;
    let else_name = ensure_variable_node(&format!("{name}_else"), else_v, graph)?;
    graph.add_node(GraphNode::new(
        name.to_string(),
        Layer::Where(WhereLayer::new()),
        vec![cond_name, then_name, else_name],
    ));
    Ok(NodeValue::Variable(name.to_string()))
}

/// Try to match a Select pattern against known activation functions.
///
/// Matches:
/// - ReLU: `if x > 0 { x } else { 0 }`
/// - LeakyReLU: `if x > 0 { x } else { alpha * x }` — `alpha` may be a
///   translation-time constant or a variable (e.g. a kernel parameter)
///
/// Returns `None` if no pattern matches (caller should fall back to WhereLayer).
fn try_match_activation_pattern(
    name: &str,
    cond_idx: usize,
    then_idx: usize,
    else_idx: usize,
    all_nodes: &[IRNode],
    values: &[NodeValue],
    graph: &mut GraphNetwork,
) -> Option<Result<NodeValue, VerifyError>> {
    // Bounds-checked access returning Some(Err(...)) on out-of-bounds.
    macro_rules! try_get {
        ($slice:expr, $idx:expr, $ctx:expr) => {
            match get_value($slice, $idx, $ctx) {
                Ok(v) => v,
                Err(e) => return Some(Err(e)),
            }
        };
    }

    // Condition must be Compare { op: Gt/Ge, lhs, rhs } where rhs = 0
    let cond_node = try_get!(all_nodes, cond_idx, "Select activation cond node");
    let compare_lhs_idx = match &cond_node.kind {
        IRNodeKind::Compare { op, lhs, rhs }
            if matches!(op, CompareOpKind::Gt | CompareOpKind::Ge)
                && matches!(
                    try_get!(values, rhs.index(), "Select activation rhs value"),
                    NodeValue::Constant(c) if c.get() == 0.0
                ) =>
        {
            lhs.index()
        }
        // SAFETY: IRNodeKind is #[non_exhaustive]. Non-Compare nodes (or Compare
        // with wrong op/rhs) don't match activation patterns. Returning None
        // causes the caller to fall back to WhereLayer (conservative).
        _ => return None,
    };

    // then_val must be the same node as the comparison lhs (x)
    if then_idx != compare_lhs_idx {
        return None;
    }

    // compare lhs must be a variable
    let input_name = match try_get!(values, compare_lhs_idx, "Select activation lhs value") {
        NodeValue::Variable(v) => v.clone(),
        // Constants can't be activation inputs — pattern requires variable.
        NodeValue::Constant(_) => return None,
    };

    // Check else_val for ReLU or LeakyReLU patterns
    let else_val = try_get!(values, else_idx, "Select activation else value");
    if matches!(else_val, NodeValue::Constant(c) if c.get() == 0.0) {
        // ReLU: if x > 0 { x } else { 0 }
        add_unary_node(name, Layer::ReLU(ReLULayer::new()), &input_name, graph);
        return Some(Ok(NodeValue::Variable(name.to_string())));
    }

    // Check for slope * x pattern (LeakyReLU). The slope may be a constant
    // (classic LeakyReLU) or a variable (e.g. the AdaIN+LeakyReLU kernels
    // pass the negative slope as a kernel parameter).
    let else_node = try_get!(all_nodes, else_idx, "Select activation else node");
    let slope = match &else_node.kind {
        IRNodeKind::BinOp {
            op: BinOpKind::Mul,
            lhs,
            rhs,
        } => {
            if lhs.index() == compare_lhs_idx {
                Some(try_get!(values, rhs.index(), "Select LeakyReLU slope (rhs)").clone())
            } else if rhs.index() == compare_lhs_idx {
                Some(try_get!(values, lhs.index(), "Select LeakyReLU slope (lhs)").clone())
            } else {
                None
            }
        }
        // SAFETY: IRNodeKind is #[non_exhaustive]. Non-BinOp::Mul nodes don't
        // match LeakyReLU pattern. None causes `slope?` to return None from
        // the function, falling back to WhereLayer in the caller (conservative).
        _ => None,
    };

    match slope? {
        NodeValue::Constant(alpha) => {
            let alpha = alpha.get();
            // Decompose LeakyReLU(x, alpha) = alpha*x + (1-alpha)*ReLU(x).
            //
            // The historical reason — "NY's LeakyReLULayer returns IBP-wide
            // bounds (#2977)" — is STALE: `leaky_relu_linear_relaxation` is a
            // full chord/tangent relaxation upstream today. The decomposition
            // is kept on MEASURED evidence instead: `Layer::ReLU` is dispatched
            // into NY's per-site ReLU CROWN path and is the only activation
            // α-CROWN optimizes, so the ReLU form's CROWN comes out 1.3-1.7x
            // tighter than a fused `Layer::LeakyReLU` on a shared fixture. The
            // measurement (and the trigger to delete this decomposition once it
            // stops winning) lives in `graph_tensor_leaky_relu.rs`:
            // `tests::decomposition_crown_is_tighter_than_native_leaky_relu`.
            let scale_name = format!("{name}_alpha_x");
            add_unary_node(
                &scale_name,
                Layer::MulConstant(MulConstantLayer::scalar(alpha)),
                &input_name,
                graph,
            );
            let relu_name = format!("{name}_relu");
            add_unary_node(
                &relu_name,
                Layer::ReLU(ReLULayer::new()),
                &input_name,
                graph,
            );
            let relu_scaled_name = format!("{name}_relu_scaled");
            add_unary_node(
                &relu_scaled_name,
                Layer::MulConstant(MulConstantLayer::scalar(1.0 - alpha)),
                &relu_name,
                graph,
            );
            graph.add_node(GraphNode::new(
                name.to_string(),
                Layer::Add(AddLayer),
                vec![scale_name, relu_scaled_name],
            ));
        }
        NodeValue::Variable(slope_name) => {
            // Variable slope: decompose LeakyReLU(x, s) = ReLU(x) - s*ReLU(-x).
            // Exact identity: x >= 0 → ReLU(x)=x, ReLU(-x)=0 → x;
            //                 x <  0 → ReLU(x)=0, ReLU(-x)=-x → s*x.
            // (At x == 0 both select branches agree — s*0 == 0 — so the Gt/Ge
            // boundary distinction is immaterial, same as the constant path.)
            //
            // Previously this shape fell back to a WhereLayer, whose bound
            // propagation takes the branch UNION when the condition is not a
            // translation-time constant. NY's DAG alpha-CROWN currently returns
            // that union even for point inputs (it no longer proves the
            // condition sign from intermediate bounds on this path), which
            // broke fused-vs-sequential equivalence proofs with a
            // (1-slope)*|y| gap. The ReLU/MulBinary decomposition needs no
            // condition reasoning at all and stays exact under CROWN. (#2977)
            //
            // RE-VERIFIED against current NY: `Layer::Where` is still an
            // explicit `BackwardDispatchResult::Unsupported` arm in
            // `network::backward_dispatch::dispatch` (no CROWN backward), and
            // `WhereLayer::propagate_ibp` still returns the per-element union
            // of the two branches for a data-dependent condition. This premise
            // has NOT gone stale — unlike the `#2977` LeakyReLU one above.
            let neg_name = format!("{name}_neg_x");
            add_unary_node(
                &neg_name,
                Layer::MulConstant(MulConstantLayer::scalar(-1.0)),
                &input_name,
                graph,
            );
            let relu_neg_name = format!("{name}_relu_neg");
            add_unary_node(
                &relu_neg_name,
                Layer::ReLU(ReLULayer::new()),
                &neg_name,
                graph,
            );
            let slope_part_name = format!("{name}_slope_neg");
            graph.add_node(GraphNode::binary(
                slope_part_name.clone(),
                Layer::MulBinary(MulBinaryLayer),
                slope_name,
                relu_neg_name,
            ));
            let relu_name = format!("{name}_relu");
            add_unary_node(
                &relu_name,
                Layer::ReLU(ReLULayer::new()),
                &input_name,
                graph,
            );
            graph.add_node(GraphNode::binary(
                name.to_string(),
                Layer::Sub(SubLayer),
                relu_name,
                slope_part_name,
            ));
        }
    }
    Some(Ok(NodeValue::Variable(name.to_string())))
}

/// Ensure a NodeValue is represented as a graph node, creating a constant
/// node if needed. Returns the graph node name.
fn ensure_variable_node(
    fallback_name: &str,
    val: &NodeValue,
    graph: &mut GraphNetwork,
) -> Result<String, VerifyError> {
    match val {
        NodeValue::Variable(var_name) => Ok(var_name.clone()),
        NodeValue::Constant(c) => {
            // Create a constant graph node: MulConstant(0) -> AddConstant(c)
            // IBP bounds: [0,0] -> [c,c] regardless of input.
            let c = c.get();
            let zero_name = format!("{fallback_name}_zero");
            graph.add_node(GraphNode::from_input(
                zero_name.clone(),
                Layer::MulConstant(MulConstantLayer::scalar(0.0)),
            ));
            graph.add_node(GraphNode::new(
                fallback_name.to_string(),
                Layer::AddConstant(AddConstantLayer::new(scalar_array(c)?)),
                vec![zero_name],
            ));
            Ok(fallback_name.to_string())
        }
    }
}

#[cfg(kani)]
mod kani_proofs {
    use super::*;
    use crate::graph_ops::evaluate_constant_compare;
    use nn_dsl::ir::CompareOpKind;

    /// Proves the cross-module Select-Compare convention alignment:
    /// - `evaluate_constant_compare(Ge, x, x)` returns 1.0 for any finite x
    /// - `select_condition_is_true(1.0)` returns true (then-branch)
    /// - `evaluate_constant_compare(Gt, x, x)` returns 0.0 for any finite x
    /// - `select_condition_is_true(0.0)` returns false (else-branch)
    ///
    /// This proves Select correctly interprets Compare's output convention:
    /// Ge(x,x) → then-branch, Gt(x,x) → else-branch.
    #[kani::unwind(1)]
    #[kani::proof]
    fn select_compare_convention_alignment() {
        let x: f32 = kani::any();
        kani::assume(x.is_finite());

        // Ge(x, x) must fold to 1.0 (x >= x is always true)
        let ge_result = evaluate_constant_compare(CompareOpKind::Ge, x, x)
            .expect("Ge on finite values must not fail");
        assert_eq!(
            ge_result.to_bits(),
            1.0f32.to_bits(),
            "Ge(x, x) must return 1.0"
        );

        // Select must interpret 1.0 as then-branch
        assert!(
            select_condition_is_true(ge_result),
            "select must take then-branch on Ge(x, x) result"
        );

        // Gt(x, x) must fold to 0.0 (x > x is always false)
        let gt_result = evaluate_constant_compare(CompareOpKind::Gt, x, x)
            .expect("Gt on finite values must not fail");
        assert_eq!(
            gt_result.to_bits(),
            0.0f32.to_bits(),
            "Gt(x, x) must return 0.0"
        );

        // Select must interpret 0.0 as else-branch
        assert!(
            !select_condition_is_true(gt_result),
            "select must take else-branch on Gt(x, x) result"
        );
    }
}

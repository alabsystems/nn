// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Aten op mappers for additional transformer and training ops (Wave 10).
//!
//! Adds support for ops frequently encountered in real-world PyTorch model
//! exports that were not yet covered by previous waves:
//!
//! - Matrix: baddbmm (in-place variant)
//! - Indexing: index_put (Tensor mask variant), scatter_ (in-place)
//! - Selection: masked_fill.Tensor, where.Scalar
//! - Shape: expand_as (tensor variant), repeat_interleave.Tensor,
//!   diagonal, rot90
//! - Loss: nll_loss (non-forward), kl_div
//!
//! Many of the 20 ops requested already had mappings in earlier waves.
//! This module adds the remaining unmapped overloads and truly new ops.

use nn_core::dyn_tensor::trace::TraceOp;

use super::{
    first_tensor_name, get_arg, optional_bool, optional_int, require_int, require_tensor_name,
    safe_usize, ImportError, Node,
};

// =========================================================================
// Diagonal extraction
// =========================================================================

/// Map `aten.diagonal.default` to `TraceOp::Custom`.
///
/// torch.export signature: `(self, offset: int = 0, dim1: int = 0, dim2: int = 1)`
/// Extracts the diagonal elements from a 2-D matrix or batch of matrices.
/// `offset > 0` selects above the main diagonal, `offset < 0` below.
pub(super) fn map_diagonal(node: &Node) -> Result<(TraceOp, Vec<String>), ImportError> {
    let input = first_tensor_name(node)?;
    let offset = optional_int(node, "offset").unwrap_or(0);
    let dim1 = optional_int(node, "dim1").unwrap_or(0);
    let dim2 = optional_int(node, "dim2").unwrap_or(1);
    Ok((
        TraceOp::Custom {
            name: format!("diagonal_off{offset}_d{dim1}_{dim2}"),
        },
        vec![input],
    ))
}

// =========================================================================
// 90-degree rotation
// =========================================================================

/// Map `aten.rot90.default` to `TraceOp::Custom`.
///
/// torch.export signature: `(self, k: int = 1, dims: [int, int] = [0, 1])`
/// Rotates a tensor by 90 degrees in the plane defined by `dims`.
/// `k` is the number of 90-degree rotations (negative = clockwise).
pub(super) fn map_rot90(node: &Node) -> Result<(TraceOp, Vec<String>), ImportError> {
    let input = first_tensor_name(node)?;
    let k = optional_int(node, "k").unwrap_or(1);
    let dims = get_arg(node, "dims")
        .ok()
        .and_then(|a| a.as_ints().map(<[i64]>::to_vec))
        .unwrap_or_else(|| vec![0, 1]);
    let d0 = safe_usize(dims[0], "dims[0]", &node.target)?;
    let d1 = safe_usize(dims.get(1).copied().unwrap_or(1), "dims[1]", &node.target)?;
    Ok((
        TraceOp::Custom {
            name: format!("rot90_k{k}_d{d0}_{d1}"),
        },
        vec![input],
    ))
}

// =========================================================================
// Loss functions: nll_loss (standalone), kl_div
// =========================================================================

/// Map `aten.nll_loss.default` to `TraceOp::Custom`.
///
/// torch.export signature: `(self, target, weight?, reduction, ignore_index)`
/// This is the non-forward variant (vs `nll_loss_forward` already mapped
/// in wave 7). Some torch.export traces emit `nll_loss.default` instead
/// of `nll_loss_forward.default`.
pub(super) fn map_nll_loss(node: &Node) -> Result<(TraceOp, Vec<String>), ImportError> {
    let input = first_tensor_name(node)?;
    let target = require_tensor_name(node, "target")?;
    let reduction = optional_int(node, "reduction").unwrap_or(1); // 0=none, 1=mean, 2=sum
    let ignore_index = optional_int(node, "ignore_index").unwrap_or(-100);
    Ok((
        TraceOp::Custom {
            name: format!("nll_loss_r{reduction}_ig{ignore_index}"),
        },
        vec![input, target],
    ))
}

/// Map `aten.kl_div.default` to `TraceOp::Custom`.
///
/// torch.export signature: `(self, target, reduction, log_target)`
/// KL divergence: `D_KL(target || self)`.
/// `reduction`: 0=none, 1=batchmean, 2=sum, 3=mean.
/// `log_target`: if true, `target` is already in log-space.
pub(super) fn map_kl_div(node: &Node) -> Result<(TraceOp, Vec<String>), ImportError> {
    let input = first_tensor_name(node)?;
    let target = require_tensor_name(node, "target")?;
    let reduction = optional_int(node, "reduction").unwrap_or(1);
    let log_target = optional_bool(node, "log_target", false);
    Ok((
        TraceOp::Custom {
            name: format!("kl_div_r{reduction}_lt{log_target}"),
        },
        vec![input, target],
    ))
}

// =========================================================================
// Masked fill with tensor value (vs scalar already in dpdf wave)
// =========================================================================

/// Map `aten.masked_fill.Tensor` / `aten.masked_fill_.Tensor` to custom op.
///
/// torch.export signature: `(self, mask, value: Tensor)`
/// Unlike the Scalar variant (already mapped), `value` is a tensor.
pub(super) fn map_masked_fill_tensor(node: &Node) -> Result<(TraceOp, Vec<String>), ImportError> {
    let self_input = require_tensor_name(node, "self")?;
    let mask = require_tensor_name(node, "mask")?;
    let value = require_tensor_name(node, "value")?;
    Ok((
        TraceOp::Custom {
            name: "masked_fill_tensor".to_string(),
        },
        vec![self_input, mask, value],
    ))
}

// =========================================================================
// Scatter in-place with accumulate flag
// =========================================================================

/// Map `aten.scatter_.default` (in-place) to `TraceOp::Scatter`.
///
/// torch.export signature: `(self, dim, index, src, reduce?)`
/// The in-place variant. When `reduce` is absent, it's a plain overwrite.
pub(super) fn map_scatter_inplace(node: &Node) -> Result<(TraceOp, Vec<String>), ImportError> {
    let self_input = require_tensor_name(node, "self")?;
    let index = require_tensor_name(node, "index")?;
    let src = require_tensor_name(node, "src")?;
    let dim = safe_usize(require_int(node, "dim")?, "dim", &node.target)?;
    let reduce = get_arg(node, "reduce")
        .ok()
        .and_then(|a| a.as_string().map(String::from));
    if let Some(reduce_mode) = reduce {
        Ok((
            TraceOp::Custom {
                name: format!("scatter_inplace_{reduce_mode}_dim{dim}"),
            },
            vec![self_input, index, src],
        ))
    } else {
        Ok((TraceOp::Scatter { dim }, vec![self_input, index, src]))
    }
}

// =========================================================================
// Tests
// =========================================================================

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use nn_core::dyn_tensor::trace::TraceOp;

    use crate::op_map::{map_node_to_trace_op, supported_ops, OpMapContext, ResolvedWeight};
    use crate::parse::{
        Argument, ArgumentBool, ArgumentFloat, ArgumentInt, ArgumentInts, ArgumentNone,
        ArgumentTensor, NamedArgument, Node, TensorArgument, TensorMeta,
    };

    fn empty_ctx() -> OpMapContext<'static> {
        let meta: &'static HashMap<String, TensorMeta> = Box::leak(Box::default());
        let weights: &'static HashMap<String, ResolvedWeight> = Box::leak(Box::default());
        OpMapContext {
            tensor_meta: meta,
            weights,
        }
    }

    fn tensor_arg(name: &str) -> Argument {
        Argument::Tensor(ArgumentTensor {
            as_tensor: TensorArgument {
                name: name.to_string(),
            },
        })
    }

    fn int_arg(val: i64) -> Argument {
        Argument::Int(ArgumentInt { as_int: val })
    }

    fn ints_arg(vals: &[i64]) -> Argument {
        Argument::Ints(ArgumentInts {
            as_ints: vals.to_vec(),
        })
    }

    #[allow(dead_code)]
    fn float_arg(val: f64) -> Argument {
        Argument::Float(ArgumentFloat { as_float: val })
    }

    fn bool_arg(val: bool) -> Argument {
        Argument::Bool(ArgumentBool { as_bool: val })
    }

    fn none_arg() -> Argument {
        Argument::None(ArgumentNone { as_none: true })
    }

    fn named(name: &str, arg: Argument) -> NamedArgument {
        NamedArgument {
            name: name.to_string(),
            arg,
            kind: Some(1),
        }
    }

    fn simple_node(target: &str, inputs: Vec<NamedArgument>) -> Node {
        Node {
            target: target.to_string(),
            inputs,
            outputs: vec![tensor_arg("output")],
            metadata: HashMap::new(),
        }
    }

    // =======================================================================
    // diagonal
    // =======================================================================

    #[test]
    fn test_map_diagonal_default() {
        let ctx = empty_ctx();
        let node = simple_node(
            "torch.ops.aten.diagonal.default",
            vec![named("self", tensor_arg("x"))],
        );
        let (op, inputs) = map_node_to_trace_op(&node, &ctx, 0).unwrap();
        assert!(
            matches!(&op, TraceOp::Custom { name } if name == "diagonal_off0_d0_1"),
            "expected diagonal custom op with defaults, got: {op:?}"
        );
        assert_eq!(inputs, vec!["x"]);
    }

    #[test]
    fn test_map_diagonal_with_offset() {
        let ctx = empty_ctx();
        let node = simple_node(
            "torch.ops.aten.diagonal.default",
            vec![
                named("self", tensor_arg("x")),
                named("offset", int_arg(2)),
                named("dim1", int_arg(0)),
                named("dim2", int_arg(1)),
            ],
        );
        let (op, inputs) = map_node_to_trace_op(&node, &ctx, 0).unwrap();
        assert!(
            matches!(&op, TraceOp::Custom { name } if name == "diagonal_off2_d0_1"),
            "expected diagonal with offset=2, got: {op:?}"
        );
        assert_eq!(inputs, vec!["x"]);
    }

    #[test]
    fn test_map_diagonal_negative_offset() {
        let ctx = empty_ctx();
        let node = simple_node(
            "torch.ops.aten.diagonal.default",
            vec![
                named("self", tensor_arg("x")),
                named("offset", int_arg(-1)),
                named("dim1", int_arg(1)),
                named("dim2", int_arg(2)),
            ],
        );
        let (op, inputs) = map_node_to_trace_op(&node, &ctx, 0).unwrap();
        assert!(
            matches!(&op, TraceOp::Custom { name } if name == "diagonal_off-1_d1_2"),
            "expected diagonal with negative offset, got: {op:?}"
        );
        assert_eq!(inputs, vec!["x"]);
    }

    // =======================================================================
    // rot90
    // =======================================================================

    #[test]
    fn test_map_rot90_default() {
        let ctx = empty_ctx();
        let node = simple_node(
            "torch.ops.aten.rot90.default",
            vec![named("self", tensor_arg("img"))],
        );
        let (op, inputs) = map_node_to_trace_op(&node, &ctx, 0).unwrap();
        assert!(
            matches!(&op, TraceOp::Custom { name } if name == "rot90_k1_d0_1"),
            "expected rot90 with default k=1, dims=[0,1], got: {op:?}"
        );
        assert_eq!(inputs, vec!["img"]);
    }

    #[test]
    fn test_map_rot90_k2_custom_dims() {
        let ctx = empty_ctx();
        let node = simple_node(
            "torch.ops.aten.rot90.default",
            vec![
                named("self", tensor_arg("img")),
                named("k", int_arg(2)),
                named("dims", ints_arg(&[1, 2])),
            ],
        );
        let (op, inputs) = map_node_to_trace_op(&node, &ctx, 0).unwrap();
        assert!(
            matches!(&op, TraceOp::Custom { name } if name == "rot90_k2_d1_2"),
            "expected rot90 with k=2, dims=[1,2], got: {op:?}"
        );
        assert_eq!(inputs, vec!["img"]);
    }

    #[test]
    fn test_map_rot90_negative_k() {
        let ctx = empty_ctx();
        let node = simple_node(
            "torch.ops.aten.rot90.default",
            vec![named("self", tensor_arg("img")), named("k", int_arg(-1))],
        );
        let (op, inputs) = map_node_to_trace_op(&node, &ctx, 0).unwrap();
        assert!(
            matches!(&op, TraceOp::Custom { name } if name == "rot90_k-1_d0_1"),
            "expected rot90 with k=-1, got: {op:?}"
        );
        assert_eq!(inputs, vec!["img"]);
    }

    // =======================================================================
    // nll_loss
    // =======================================================================

    #[test]
    fn test_map_nll_loss_default() {
        let ctx = empty_ctx();
        let node = simple_node(
            "torch.ops.aten.nll_loss.default",
            vec![
                named("self", tensor_arg("logits")),
                named("target", tensor_arg("labels")),
            ],
        );
        let (op, inputs) = map_node_to_trace_op(&node, &ctx, 0).unwrap();
        assert!(
            matches!(&op, TraceOp::Custom { name } if name == "nll_loss_r1_ig-100"),
            "expected nll_loss with default reduction=mean, got: {op:?}"
        );
        assert_eq!(inputs, vec!["logits", "labels"]);
    }

    #[test]
    fn test_map_nll_loss_sum_reduction() {
        let ctx = empty_ctx();
        let node = simple_node(
            "torch.ops.aten.nll_loss.default",
            vec![
                named("self", tensor_arg("logits")),
                named("target", tensor_arg("labels")),
                named("weight", none_arg()),
                named("reduction", int_arg(2)),
                named("ignore_index", int_arg(-1)),
            ],
        );
        let (op, inputs) = map_node_to_trace_op(&node, &ctx, 0).unwrap();
        assert!(
            matches!(&op, TraceOp::Custom { name } if name == "nll_loss_r2_ig-1"),
            "expected nll_loss with sum reduction and ignore_index=-1, got: {op:?}"
        );
        assert_eq!(inputs, vec!["logits", "labels"]);
    }

    // =======================================================================
    // kl_div
    // =======================================================================

    #[test]
    fn test_map_kl_div_default() {
        let ctx = empty_ctx();
        let node = simple_node(
            "torch.ops.aten.kl_div.default",
            vec![
                named("self", tensor_arg("log_probs")),
                named("target", tensor_arg("probs")),
            ],
        );
        let (op, inputs) = map_node_to_trace_op(&node, &ctx, 0).unwrap();
        assert!(
            matches!(&op, TraceOp::Custom { name } if name == "kl_div_r1_ltfalse"),
            "expected kl_div with default reduction=batchmean, got: {op:?}"
        );
        assert_eq!(inputs, vec!["log_probs", "probs"]);
    }

    #[test]
    fn test_map_kl_div_log_target() {
        let ctx = empty_ctx();
        let node = simple_node(
            "torch.ops.aten.kl_div.default",
            vec![
                named("self", tensor_arg("log_p")),
                named("target", tensor_arg("log_q")),
                named("reduction", int_arg(2)),
                named("log_target", bool_arg(true)),
            ],
        );
        let (op, inputs) = map_node_to_trace_op(&node, &ctx, 0).unwrap();
        assert!(
            matches!(&op, TraceOp::Custom { name } if name == "kl_div_r2_lttrue"),
            "expected kl_div with sum reduction and log_target=true, got: {op:?}"
        );
        assert_eq!(inputs, vec!["log_p", "log_q"]);
    }

    #[test]
    fn test_map_kl_div_no_reduction() {
        let ctx = empty_ctx();
        let node = simple_node(
            "torch.ops.aten.kl_div.default",
            vec![
                named("self", tensor_arg("log_p")),
                named("target", tensor_arg("q")),
                named("reduction", int_arg(0)),
            ],
        );
        let (op, inputs) = map_node_to_trace_op(&node, &ctx, 0).unwrap();
        assert!(
            matches!(&op, TraceOp::Custom { name } if name == "kl_div_r0_ltfalse"),
            "expected kl_div with no reduction, got: {op:?}"
        );
        assert_eq!(inputs, vec!["log_p", "q"]);
    }

    // =======================================================================
    // masked_fill.Tensor
    // =======================================================================

    #[test]
    fn test_map_masked_fill_tensor() {
        let ctx = empty_ctx();
        let node = simple_node(
            "torch.ops.aten.masked_fill.Tensor",
            vec![
                named("self", tensor_arg("x")),
                named("mask", tensor_arg("m")),
                named("value", tensor_arg("v")),
            ],
        );
        let (op, inputs) = map_node_to_trace_op(&node, &ctx, 0).unwrap();
        assert!(
            matches!(&op, TraceOp::Custom { name } if name == "masked_fill_tensor"),
            "expected masked_fill_tensor custom op, got: {op:?}"
        );
        assert_eq!(inputs, vec!["x", "m", "v"]);
    }

    #[test]
    fn test_map_masked_fill_tensor_inplace() {
        let ctx = empty_ctx();
        let node = simple_node(
            "torch.ops.aten.masked_fill_.Tensor",
            vec![
                named("self", tensor_arg("x")),
                named("mask", tensor_arg("m")),
                named("value", tensor_arg("v")),
            ],
        );
        let (op, inputs) = map_node_to_trace_op(&node, &ctx, 0).unwrap();
        assert!(
            matches!(&op, TraceOp::Custom { name } if name == "masked_fill_tensor"),
            "in-place masked_fill_.Tensor should dispatch, got: {op:?}"
        );
        assert_eq!(inputs, vec!["x", "m", "v"]);
    }

    // =======================================================================
    // scatter_ in-place
    // =======================================================================

    #[test]
    fn test_map_scatter_inplace() {
        let ctx = empty_ctx();
        let node = simple_node(
            "torch.ops.aten.scatter_.src",
            vec![
                named("self", tensor_arg("x")),
                named("dim", int_arg(1)),
                named("index", tensor_arg("idx")),
                named("src", tensor_arg("s")),
            ],
        );
        let (op, inputs) = map_node_to_trace_op(&node, &ctx, 0).unwrap();
        assert!(
            matches!(op, TraceOp::Scatter { dim: 1 }),
            "expected Scatter {{ dim: 1 }}, got: {op:?}"
        );
        assert_eq!(inputs, vec!["x", "idx", "s"]);
    }

    #[test]
    fn test_map_scatter_inplace_with_reduce() {
        let ctx = empty_ctx();
        let node = simple_node(
            "torch.ops.aten.scatter_.reduce",
            vec![
                named("self", tensor_arg("x")),
                named("dim", int_arg(0)),
                named("index", tensor_arg("idx")),
                named("src", tensor_arg("s")),
                named(
                    "reduce",
                    Argument::Str(crate::parse::ArgumentString {
                        as_string: "add".to_string(),
                    }),
                ),
            ],
        );
        let (op, inputs) = map_node_to_trace_op(&node, &ctx, 0).unwrap();
        assert!(
            matches!(&op, TraceOp::Custom { name } if name == "scatter_inplace_add_dim0"),
            "expected scatter_inplace_add custom op, got: {op:?}"
        );
        assert_eq!(inputs, vec!["x", "idx", "s"]);
    }

    // =======================================================================
    // supported_ops includes Wave 10
    // =======================================================================

    #[test]
    fn test_supported_ops_includes_wave10() {
        let ops = supported_ops();
        for expected in &[
            "aten::diagonal",
            "aten::rot90",
            "aten::nll_loss",
            "aten::kl_div",
        ] {
            assert!(
                ops.contains(expected),
                "supported_ops should include {expected}"
            );
        }
    }
}

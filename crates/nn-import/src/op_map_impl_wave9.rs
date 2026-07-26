// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Aten op mappers for commonly missing model patterns (Wave 9).
//!
//! Adds support for ops frequently encountered in real-world PyTorch model
//! exports that were not yet mapped:
//!
//! - Unary math: trunc, expm1, log1p, acos, asin, atan, cosh, sinh
//! - Value testing: isinf, isnan, isfinite
//! - Bitwise: bitwise_not, bitwise_and, bitwise_or
//! - Tensor-arg clamp variants: clamp_min.Tensor, clamp_max.Tensor
//! - Tensor creation: tile, arange.start, eye
//! - Expand variants: expand_as, broadcast_to
//! - Loss functions: binary_cross_entropy_with_logits, cross_entropy_loss
//! - Indexing: index_fill, index_copy, scatter_reduce
//! - Repeat: repeat_interleave.self_int (scalar repeats)

use nn_core::dyn_tensor::trace::TraceOp;

use super::{
    first_tensor_name, get_arg, optional_float, optional_int, require_int,
    require_ints, require_tensor_name, safe_usize, safe_usize_allow_neg1, ImportError, Node,
};

// =========================================================================
// Unary math (map to Custom since no dedicated TraceOp variant)
// =========================================================================

/// Map `aten.trunc.default` to custom op (truncation toward zero).
pub(super) fn map_trunc(node: &Node) -> Result<(TraceOp, Vec<String>), ImportError> {
    let input = first_tensor_name(node)?;
    Ok((
        TraceOp::Custom {
            name: "trunc".to_string(),
        },
        vec![input],
    ))
}

/// Map `aten.expm1.default` to custom op: `exp(x) - 1` (numerically stable).
pub(super) fn map_expm1(node: &Node) -> Result<(TraceOp, Vec<String>), ImportError> {
    let input = first_tensor_name(node)?;
    Ok((
        TraceOp::Custom {
            name: "expm1".to_string(),
        },
        vec![input],
    ))
}

/// Map `aten.log1p.default` to custom op: `log(1 + x)` (numerically stable).
pub(super) fn map_log1p(node: &Node) -> Result<(TraceOp, Vec<String>), ImportError> {
    let input = first_tensor_name(node)?;
    Ok((
        TraceOp::Custom {
            name: "log1p".to_string(),
        },
        vec![input],
    ))
}

/// Map `aten.acos.default` to custom op.
pub(super) fn map_acos(node: &Node) -> Result<(TraceOp, Vec<String>), ImportError> {
    let input = first_tensor_name(node)?;
    Ok((
        TraceOp::Custom {
            name: "acos".to_string(),
        },
        vec![input],
    ))
}

/// Map `aten.asin.default` to custom op.
pub(super) fn map_asin(node: &Node) -> Result<(TraceOp, Vec<String>), ImportError> {
    let input = first_tensor_name(node)?;
    Ok((
        TraceOp::Custom {
            name: "asin".to_string(),
        },
        vec![input],
    ))
}

/// Map `aten.atan.default` to custom op (single-arg arctangent).
pub(super) fn map_atan(node: &Node) -> Result<(TraceOp, Vec<String>), ImportError> {
    let input = first_tensor_name(node)?;
    Ok((
        TraceOp::Custom {
            name: "atan".to_string(),
        },
        vec![input],
    ))
}

/// Map `aten.cosh.default` to custom op.
pub(super) fn map_cosh(node: &Node) -> Result<(TraceOp, Vec<String>), ImportError> {
    let input = first_tensor_name(node)?;
    Ok((
        TraceOp::Custom {
            name: "cosh".to_string(),
        },
        vec![input],
    ))
}

/// Map `aten.sinh.default` to custom op.
pub(super) fn map_sinh(node: &Node) -> Result<(TraceOp, Vec<String>), ImportError> {
    let input = first_tensor_name(node)?;
    Ok((
        TraceOp::Custom {
            name: "sinh".to_string(),
        },
        vec![input],
    ))
}

// =========================================================================
// Value testing (produce boolean mask tensors)
// =========================================================================

/// Map `aten.isinf.default` to custom op producing a boolean mask.
pub(super) fn map_isinf(node: &Node) -> Result<(TraceOp, Vec<String>), ImportError> {
    let input = first_tensor_name(node)?;
    Ok((
        TraceOp::Custom {
            name: "isinf".to_string(),
        },
        vec![input],
    ))
}

/// Map `aten.isnan.default` to custom op producing a boolean mask.
pub(super) fn map_isnan(node: &Node) -> Result<(TraceOp, Vec<String>), ImportError> {
    let input = first_tensor_name(node)?;
    Ok((
        TraceOp::Custom {
            name: "isnan".to_string(),
        },
        vec![input],
    ))
}

/// Map `aten.isfinite.default` to custom op producing a boolean mask.
pub(super) fn map_isfinite(node: &Node) -> Result<(TraceOp, Vec<String>), ImportError> {
    let input = first_tensor_name(node)?;
    Ok((
        TraceOp::Custom {
            name: "isfinite".to_string(),
        },
        vec![input],
    ))
}

// =========================================================================
// Bitwise operations
// =========================================================================

/// Map `aten.bitwise_not.default` to custom op.
pub(super) fn map_bitwise_not(node: &Node) -> Result<(TraceOp, Vec<String>), ImportError> {
    let input = first_tensor_name(node)?;
    Ok((
        TraceOp::Custom {
            name: "bitwise_not".to_string(),
        },
        vec![input],
    ))
}

/// Map `aten.bitwise_and.Tensor` to custom op.
pub(super) fn map_bitwise_and(node: &Node) -> Result<(TraceOp, Vec<String>), ImportError> {
    let lhs = require_tensor_name(node, "self")?;
    let rhs = require_tensor_name(node, "other")?;
    Ok((
        TraceOp::Custom {
            name: "bitwise_and".to_string(),
        },
        vec![lhs, rhs],
    ))
}

/// Map `aten.bitwise_or.Tensor` to custom op.
pub(super) fn map_bitwise_or(node: &Node) -> Result<(TraceOp, Vec<String>), ImportError> {
    let lhs = require_tensor_name(node, "self")?;
    let rhs = require_tensor_name(node, "other")?;
    Ok((
        TraceOp::Custom {
            name: "bitwise_or".to_string(),
        },
        vec![lhs, rhs],
    ))
}

// =========================================================================
// Tensor-arg clamp variants
// =========================================================================

/// Map `aten.clamp_min.Tensor` to `TraceOp::Maximum` (element-wise max).
///
/// `clamp_min(self, min_tensor)` is equivalent to `torch.maximum(self, min_tensor)`.
pub(super) fn map_clamp_min_tensor(node: &Node) -> Result<(TraceOp, Vec<String>), ImportError> {
    let input = require_tensor_name(node, "self")?;
    let min_tensor = require_tensor_name(node, "min")?;
    Ok((TraceOp::Maximum, vec![input, min_tensor]))
}

/// Map `aten.clamp_max.Tensor` to `TraceOp::Minimum` (element-wise min).
///
/// `clamp_max(self, max_tensor)` is equivalent to `torch.minimum(self, max_tensor)`.
pub(super) fn map_clamp_max_tensor(node: &Node) -> Result<(TraceOp, Vec<String>), ImportError> {
    let input = require_tensor_name(node, "self")?;
    let max_tensor = require_tensor_name(node, "max")?;
    Ok((TraceOp::Minimum, vec![input, max_tensor]))
}

// =========================================================================
// Tensor creation: tile, arange.start, eye
// =========================================================================

/// Map `aten.tile.default` to `TraceOp::Expand` (semantically repeat/tile).
///
/// torch.export signature: `(self, dims: [int...])`
/// Tile is a synonym for repeat in PyTorch.
pub(super) fn map_tile(node: &Node) -> Result<(TraceOp, Vec<String>), ImportError> {
    let input = first_tensor_name(node)?;
    let dims = require_ints(node, "dims")?;
    let target_shape: Vec<usize> = dims
        .into_iter()
        .map(|v| safe_usize(v, "dims", &node.target))
        .collect::<Result<_, _>>()?;
    Ok((TraceOp::Expand { target_shape }, vec![input]))
}

/// Map `aten.arange.start` (2-arg form: start, end) to `TraceOp::Arange`.
///
/// torch.export signature: `(start, end, dtype?, ...)`
pub(super) fn map_arange_start(node: &Node) -> Result<(TraceOp, Vec<String>), ImportError> {
    let start = optional_float(node, "start")
        .or_else(|| {
            node.inputs.first().and_then(|na| {
                na.arg
                    .as_float()
                    .or_else(|| na.arg.as_int().map(|i| i as f64))
            })
        })
        .unwrap_or(0.0);
    let end = optional_float(node, "end")
        .or_else(|| {
            node.inputs.get(1).and_then(|na| {
                na.arg
                    .as_float()
                    .or_else(|| na.arg.as_int().map(|i| i as f64))
            })
        })
        .unwrap_or(0.0);
    Ok((
        TraceOp::Arange {
            start,
            end,
            step: 1.0,
        },
        vec![],
    ))
}

/// Map `aten.eye.default` / `aten.eye.m` to custom identity matrix creation.
///
/// torch.export signature: `(n: int, m?: int, ...)`
pub(super) fn map_eye(node: &Node) -> Result<(TraceOp, Vec<String>), ImportError> {
    let n = optional_int(node, "n")
        .or_else(|| node.inputs.first().and_then(|na| na.arg.as_int()))
        .unwrap_or(1);
    let m = optional_int(node, "m").unwrap_or(n);
    Ok((
        TraceOp::Custom {
            name: format!("eye_{n}_{m}"),
        },
        vec![],
    ))
}

// =========================================================================
// Expand variants: expand_as, broadcast_to
// =========================================================================

/// Map `aten.expand_as.default` to `TraceOp::Expand` with empty target shape.
///
/// torch.export signature: `(self, other: Tensor)`
/// Expands self to the shape of other. At import time we don't know the
/// concrete shape of `other`, so we encode as Expand with the target tensor
/// as a dependency and empty target_shape (runtime-resolved).
pub(super) fn map_expand_as(node: &Node) -> Result<(TraceOp, Vec<String>), ImportError> {
    let input = require_tensor_name(node, "self")?;
    let other = require_tensor_name(node, "other")?;
    // Expand with empty target_shape signals "match shape of second input" at runtime.
    Ok((
        TraceOp::Expand {
            target_shape: vec![],
        },
        vec![input, other],
    ))
}

/// Map `aten.broadcast_to.default` to `TraceOp::Expand`.
///
/// torch.export signature: `(self, size: [int...])`
/// Identical semantics to expand.
pub(super) fn map_broadcast_to(node: &Node) -> Result<(TraceOp, Vec<String>), ImportError> {
    let input = first_tensor_name(node)?;
    let size = require_ints(node, "size")?;
    let target_shape: Vec<usize> = size
        .into_iter()
        .map(|v| safe_usize_allow_neg1(v, "size", &node.target))
        .collect::<Result<_, _>>()?;
    Ok((TraceOp::Expand { target_shape }, vec![input]))
}

// =========================================================================
// Loss functions
// =========================================================================

/// Map `aten.binary_cross_entropy_with_logits.default` to custom op.
///
/// torch.export signature: `(self, target, weight?, pos_weight?, reduction)`
/// More numerically stable than BCE + sigmoid separately.
pub(super) fn map_bce_with_logits(node: &Node) -> Result<(TraceOp, Vec<String>), ImportError> {
    let input = first_tensor_name(node)?;
    let target = require_tensor_name(node, "target")?;
    let reduction = optional_int(node, "reduction").unwrap_or(1);
    Ok((
        TraceOp::Custom {
            name: format!("bce_with_logits_r{reduction}"),
        },
        vec![input, target],
    ))
}

/// Map `aten.cross_entropy_loss.default` to custom op.
///
/// torch.export signature: `(self, target, weight?, reduction, ignore_index, label_smoothing)`
pub(super) fn map_cross_entropy_loss(node: &Node) -> Result<(TraceOp, Vec<String>), ImportError> {
    let input = first_tensor_name(node)?;
    let target = require_tensor_name(node, "target")?;
    let reduction = optional_int(node, "reduction").unwrap_or(1);
    Ok((
        TraceOp::Custom {
            name: format!("cross_entropy_loss_r{reduction}"),
        },
        vec![input, target],
    ))
}

// =========================================================================
// Indexing: index_fill, index_copy, scatter_reduce
// =========================================================================

/// Map `aten.index_fill.int_Scalar` / `aten.index_fill_.int_Scalar` to custom op.
///
/// torch.export signature: `(self, dim, index, value)`
/// Fills elements of `self` along `dim` at positions in `index` with scalar `value`.
pub(super) fn map_index_fill(node: &Node) -> Result<(TraceOp, Vec<String>), ImportError> {
    let self_input = require_tensor_name(node, "self")?;
    let index = require_tensor_name(node, "index")?;
    let dim = safe_usize(require_int(node, "dim")?, "dim", &node.target)?;
    let value = optional_float(node, "value").unwrap_or(0.0);
    Ok((
        TraceOp::Custom {
            name: format!("index_fill_dim{dim}_v{value}"),
        },
        vec![self_input, index],
    ))
}

/// Map `aten.index_copy.default` / `aten.index_copy_.default` to custom op.
///
/// torch.export signature: `(self, dim, index, source)`
/// Copies elements of `source` into `self` along `dim` at positions in `index`.
pub(super) fn map_index_copy(node: &Node) -> Result<(TraceOp, Vec<String>), ImportError> {
    let self_input = require_tensor_name(node, "self")?;
    let index = require_tensor_name(node, "index")?;
    let source = require_tensor_name(node, "source")?;
    let dim = safe_usize(require_int(node, "dim")?, "dim", &node.target)?;
    Ok((
        TraceOp::Custom {
            name: format!("index_copy_dim{dim}"),
        },
        vec![self_input, index, source],
    ))
}

/// Map `aten.scatter_reduce.two` to custom op.
///
/// torch.export signature: `(self, dim, index, src, reduce, include_self?)`
/// Reduce modes: "sum", "prod", "mean", "amax", "amin".
pub(super) fn map_scatter_reduce(node: &Node) -> Result<(TraceOp, Vec<String>), ImportError> {
    let self_input = require_tensor_name(node, "self")?;
    let index = require_tensor_name(node, "index")?;
    let src = require_tensor_name(node, "src")?;
    let dim = safe_usize(require_int(node, "dim")?, "dim", &node.target)?;
    let reduce = get_arg(node, "reduce")
        .ok()
        .and_then(|a| a.as_string().map(String::from))
        .unwrap_or_else(|| "sum".to_string());
    Ok((
        TraceOp::Custom {
            name: format!("scatter_reduce_{reduce}_dim{dim}"),
        },
        vec![self_input, index, src],
    ))
}

// =========================================================================
// Repeat: repeat_interleave.self_int (scalar repeats count)
// =========================================================================

/// Map `aten.repeat_interleave.self_int` to `TraceOp::Custom`.
///
/// torch.export signature: `(self, repeats: int, dim?: int, output_size?: int)`
/// Unlike the Tensor variant, `repeats` is a scalar integer (same count for all).
pub(super) fn map_repeat_interleave_int(
    node: &Node,
) -> Result<(TraceOp, Vec<String>), ImportError> {
    let input = first_tensor_name(node)?;
    let repeats = optional_int(node, "repeats")
        .or_else(|| node.inputs.get(1).and_then(|na| na.arg.as_int()))
        .unwrap_or(1);
    let dim = optional_int(node, "dim").unwrap_or(0);
    let dim_u = safe_usize(dim, "dim", &node.target)?;
    Ok((
        TraceOp::Custom {
            name: format!("repeat_interleave_n{repeats}_dim{dim_u}"),
        },
        vec![input],
    ))
}

// =========================================================================
// Miscellaneous: where.ScalarOther, where.ScalarSelf, masked_scatter
// =========================================================================

/// Map `aten.where.ScalarOther` to `TraceOp::WhereCond`.
///
/// `where(condition, self_tensor, scalar_other)` — the `other` argument is a
/// scalar that gets broadcast. We emit WhereCond and let the graph builder
/// handle the constant injection.
pub(super) fn map_where_scalar_other(node: &Node) -> Result<(TraceOp, Vec<String>), ImportError> {
    let cond = require_tensor_name(node, "condition")?;
    let self_ = require_tensor_name(node, "self")?;
    // `other` is a scalar; inject as a Custom op since WhereCond expects 3 tensor inputs.
    let other_val = optional_float(node, "other").unwrap_or(0.0);
    Ok((
        TraceOp::Custom {
            name: format!("where_scalar_other_{other_val}"),
        },
        vec![cond, self_],
    ))
}

/// Map `aten.where.ScalarSelf` to custom op.
///
/// `where(condition, scalar_self, other_tensor)`.
pub(super) fn map_where_scalar_self(node: &Node) -> Result<(TraceOp, Vec<String>), ImportError> {
    let cond = require_tensor_name(node, "condition")?;
    let other = require_tensor_name(node, "other")?;
    let self_val = optional_float(node, "self").unwrap_or(0.0);
    Ok((
        TraceOp::Custom {
            name: format!("where_scalar_self_{self_val}"),
        },
        vec![cond, other],
    ))
}

/// Map `aten.masked_scatter.default` / `aten.masked_scatter_.default` to custom op.
///
/// torch.export signature: `(self, mask, source)`
/// Copies elements from `source` into positions of `self` where `mask` is true.
pub(super) fn map_masked_scatter(node: &Node) -> Result<(TraceOp, Vec<String>), ImportError> {
    let self_input = require_tensor_name(node, "self")?;
    let mask = require_tensor_name(node, "mask")?;
    let source = require_tensor_name(node, "source")?;
    Ok((
        TraceOp::Custom {
            name: "masked_scatter".to_string(),
        },
        vec![self_input, mask, source],
    ))
}

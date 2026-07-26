// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Aten op mappers for advanced tensor manipulation and control flow ops (Wave 13).
//!
//! Adds support for additional overloads and new ops:
//!
//! - Index put: `index_put.hacked_twin`, `index_put_.hacked_twin` (multi-index),
//!   accumulate-mode overloads
//! - Scatter: `scatter_.value_reduce`, `scatter_add_.default` (in-place add)
//! - Gather: `gather.out` (pre-allocated output variant)
//! - Index select: `index_select.out` (pre-allocated output variant)
//! - Masked fill: `masked_fill.Tensor_Scalar` (tensor mask + scalar value combo)
//! - Masked select: `masked_select.default`, `masked_select.out` (NEW)
//! - Nonzero: `nonzero.default`, `nonzero.out` (NEW)
//! - Topk: `topk.values` (pre-allocated output variant)
//! - Sort: `sort.values`, `sort.values_stable` (pre-allocated output variant)
//! - Unique: `unique_dim.default`, `_unique2.default` (NEW)
//! - Unique consecutive: `unique_consecutive.default` (NEW)

use nn_core::dyn_tensor::trace::TraceOp;

use super::{
    first_tensor_name, get_arg, optional_bool, optional_float, optional_int, require_int,
    require_tensor_name, safe_usize, ImportError, Node,
};

// =========================================================================
// Index put: additional overloads with accumulate and hacked_twin
// =========================================================================

/// Map `aten.index_put.hacked_twin` / `aten.index_put_.hacked_twin` to
/// `TraceOp::IndexPut`.
///
/// torch.export signature:
///   `(self, indices: [Tensor?...], values: Tensor, accumulate: bool = False)`
///
/// The "hacked_twin" variant is emitted by some torch.export paths where
/// the indices list contains optional tensors (some may be None). We extract
/// the first non-None index tensor. `accumulate=True` uses addition instead
/// of overwrite.
pub(super) fn map_index_put_hacked_twin(
    node: &Node,
) -> Result<(TraceOp, Vec<String>), ImportError> {
    let self_input = require_tensor_name(node, "self")?;
    let values = require_tensor_name(node, "values")?;
    let accumulate = optional_bool(node, "accumulate", false);

    // indices is a list of optional tensors; extract all non-None tensor names.
    let indices_arg = get_arg(node, "indices")?;
    let index_names: Vec<String> = indices_arg
        .as_tensor_names()
        .map(|names| names.into_iter().map(String::from).collect())
        .unwrap_or_default();

    if index_names.is_empty() {
        return Err(ImportError::MissingArgument {
            op_target: node.target.clone(),
            arg_name: "indices".to_string(),
        });
    }

    if accumulate {
        // Accumulate mode: encoded in custom op name for downstream handling.
        Ok((
            TraceOp::Custom {
                name: format!("index_put_accumulate_dim0_n{}", index_names.len()),
            },
            {
                let mut inputs = vec![self_input];
                inputs.extend(index_names);
                inputs.push(values);
                inputs
            },
        ))
    } else {
        // Standard overwrite mode: use first index tensor.
        Ok((
            TraceOp::IndexPut { dim: 0 },
            vec![self_input, index_names[0].clone(), values],
        ))
    }
}

/// Map `aten.index_put.default` with `accumulate=True` to `TraceOp::Custom`.
///
/// torch.export signature:
///   `(self, indices: [Tensor?...], values: Tensor, accumulate: bool = False)`
///
/// When accumulate is True, values are added rather than overwritten at
/// the indexed positions. This is semantically different from standard
/// index_put and requires special handling.
pub(super) fn map_index_put_accumulate(node: &Node) -> Result<(TraceOp, Vec<String>), ImportError> {
    let self_input = require_tensor_name(node, "self")?;
    let values = require_tensor_name(node, "values")?;

    let indices_arg = get_arg(node, "indices")?;
    let index_names: Vec<String> = indices_arg
        .as_tensor_names()
        .map(|names| names.into_iter().map(String::from).collect())
        .unwrap_or_default();

    if index_names.is_empty() {
        return Err(ImportError::MissingArgument {
            op_target: node.target.clone(),
            arg_name: "indices".to_string(),
        });
    }

    Ok((
        TraceOp::Custom {
            name: format!("index_put_accumulate_dim0_n{}", index_names.len()),
        },
        {
            let mut inputs = vec![self_input];
            inputs.extend(index_names);
            inputs.push(values);
            inputs
        },
    ))
}

// =========================================================================
// Scatter: value_reduce and in-place add variants
// =========================================================================

/// Map `aten.scatter_.value_reduce` to `TraceOp::Custom`.
///
/// torch.export signature:
///   `(self, dim: int, index: Tensor, value: Scalar, reduce: str)`
///
/// In-place scatter with a scalar value and a reduction operation
/// (e.g., "add", "multiply"). The `reduce` parameter controls how the
/// scattered value combines with existing values.
pub(super) fn map_scatter_value_reduce(node: &Node) -> Result<(TraceOp, Vec<String>), ImportError> {
    let self_input = require_tensor_name(node, "self")?;
    let index = require_tensor_name(node, "index")?;
    let dim = safe_usize(require_int(node, "dim")?, "dim", &node.target)?;
    let reduce = get_arg(node, "reduce")
        .ok()
        .and_then(|a| a.as_string().map(String::from))
        .unwrap_or_else(|| "assign".to_string());
    let value = optional_float(node, "value").unwrap_or(0.0);

    Ok((
        TraceOp::Custom {
            name: format!("scatter_value_reduce_dim{dim}_{reduce}_v{value}"),
        },
        vec![self_input, index],
    ))
}

/// Map `aten.scatter_add_.default` (in-place scatter add) to `TraceOp::ScatterAdd`.
///
/// torch.export signature: `(self, dim: int, index: Tensor, src: Tensor)`
///
/// In-place variant of scatter_add. Semantically identical to the non-inplace
/// version already mapped in dpdf wave.
pub(super) fn map_scatter_add_inplace(node: &Node) -> Result<(TraceOp, Vec<String>), ImportError> {
    let self_input = require_tensor_name(node, "self")?;
    let index = require_tensor_name(node, "index")?;
    let src = require_tensor_name(node, "src")?;
    let dim = safe_usize(require_int(node, "dim")?, "dim", &node.target)?;
    Ok((TraceOp::ScatterAdd { dim }, vec![self_input, index, src]))
}

// =========================================================================
// Gather: out variant (pre-allocated output)
// =========================================================================

/// Map `aten.gather.out` to `TraceOp::Gather`.
///
/// torch.export signature:
///   `(self, dim: int, index: Tensor, sparse_grad: bool = False, out: Tensor)`
///
/// Pre-allocated output variant. Semantically identical to `gather.default`.
/// The `out` tensor is ignored since nn handles allocation internally.
pub(super) fn map_gather_out(node: &Node) -> Result<(TraceOp, Vec<String>), ImportError> {
    let input = require_tensor_name(node, "self")?;
    let index = require_tensor_name(node, "index")?;
    let dim = safe_usize(require_int(node, "dim")?, "dim", &node.target)?;
    Ok((TraceOp::Gather { dim }, vec![input, index]))
}

// =========================================================================
// Index select: out variant (pre-allocated output)
// =========================================================================

/// Map `aten.index_select.out` to `TraceOp::IndexSelect`.
///
/// torch.export signature: `(self, dim: int, index: Tensor, out: Tensor)`
///
/// Pre-allocated output variant. Semantically identical to `index_select.default`.
/// The `out` tensor is ignored since nn handles allocation internally.
pub(super) fn map_index_select_out(node: &Node) -> Result<(TraceOp, Vec<String>), ImportError> {
    let input = first_tensor_name(node)?;
    let index = require_tensor_name(node, "index")?;
    let dim = safe_usize(require_int(node, "dim")?, "dim", &node.target)?;
    Ok((TraceOp::IndexSelect { dim }, vec![input, index]))
}

// =========================================================================
// Masked fill: Tensor mask + scalar value combo overload
// =========================================================================

/// Map `aten.masked_fill.Tensor_Scalar` to `TraceOp::Custom`.
///
/// torch.export signature: `(self, mask: Tensor, value: Scalar)`
///
/// Some torch.export paths emit this combined overload name instead of
/// the standard `masked_fill.Scalar`. Semantics are identical.
pub(super) fn map_masked_fill_tensor_scalar(
    node: &Node,
) -> Result<(TraceOp, Vec<String>), ImportError> {
    let self_input = require_tensor_name(node, "self")?;
    let mask = require_tensor_name(node, "mask")?;
    let fill_value = optional_float(node, "value")
        .or_else(|| optional_int(node, "value").map(|i| i as f64))
        .unwrap_or(0.0);

    Ok((
        TraceOp::Custom {
            name: format!("masked_fill_scalar_{fill_value}"),
        },
        vec![self_input, mask],
    ))
}

// =========================================================================
// Masked select: select elements with boolean mask (NEW)
// =========================================================================

/// Map `aten.masked_select.default` to `TraceOp::Custom`.
///
/// torch.export signature: `(self, mask: Tensor)`
///
/// Returns a 1-D tensor of elements from `self` where `mask` is True.
/// Output size is data-dependent (depends on the number of True values
/// in the mask), so this maps to a Custom op.
pub(super) fn map_masked_select(node: &Node) -> Result<(TraceOp, Vec<String>), ImportError> {
    let input = require_tensor_name(node, "self")?;
    let mask = require_tensor_name(node, "mask")?;
    Ok((
        TraceOp::Custom {
            name: "masked_select".to_string(),
        },
        vec![input, mask],
    ))
}

/// Map `aten.masked_select.out` to `TraceOp::Custom`.
///
/// torch.export signature: `(self, mask: Tensor, out: Tensor)`
///
/// Pre-allocated output variant. Same semantics as `masked_select.default`.
pub(super) fn map_masked_select_out(node: &Node) -> Result<(TraceOp, Vec<String>), ImportError> {
    let input = require_tensor_name(node, "self")?;
    let mask = require_tensor_name(node, "mask")?;
    Ok((
        TraceOp::Custom {
            name: "masked_select".to_string(),
        },
        vec![input, mask],
    ))
}

// =========================================================================
// Nonzero: indices of non-zero elements (NEW)
// =========================================================================

/// Map `aten.nonzero.default` to `TraceOp::Custom`.
///
/// torch.export signature: `(self)`
///
/// Returns a 2-D tensor of shape `[N, ndim]` containing the indices of
/// non-zero elements. Output size is data-dependent, so this maps to
/// a Custom op.
pub(super) fn map_nonzero(node: &Node) -> Result<(TraceOp, Vec<String>), ImportError> {
    let input = first_tensor_name(node)?;
    Ok((
        TraceOp::Custom {
            name: "nonzero".to_string(),
        },
        vec![input],
    ))
}

/// Map `aten.nonzero.out` to `TraceOp::Custom`.
///
/// torch.export signature: `(self, out: Tensor)`
///
/// Pre-allocated output variant. Same semantics as `nonzero.default`.
pub(super) fn map_nonzero_out(node: &Node) -> Result<(TraceOp, Vec<String>), ImportError> {
    let input = first_tensor_name(node)?;
    Ok((
        TraceOp::Custom {
            name: "nonzero".to_string(),
        },
        vec![input],
    ))
}

// =========================================================================
// Topk: pre-allocated output variant
// =========================================================================

/// Map `aten.topk.values` to `TraceOp::Topk`.
///
/// torch.export signature:
///   `(self, k: int, dim: int = -1, largest: bool = True, sorted: bool = True,
///    values: Tensor, indices: Tensor)`
///
/// Pre-allocated output variant where `values` and `indices` tensors are
/// provided. Semantically identical to `topk.default`. The `values`/`indices`
/// tensors are ignored since nn handles allocation internally.
pub(super) fn map_topk_values(node: &Node) -> Result<(TraceOp, Vec<String>), ImportError> {
    let input = first_tensor_name(node)?;
    let k = safe_usize(require_int(node, "k")?, "k", &node.target)?;
    let dim = safe_usize(optional_int(node, "dim").unwrap_or(0), "dim", &node.target)?;
    Ok((TraceOp::Topk { k, dim }, vec![input]))
}

// =========================================================================
// Sort: pre-allocated output variants
// =========================================================================

/// Map `aten.sort.values` / `aten.sort.values_stable` to `TraceOp::Sort`.
///
/// torch.export signature:
///   `(self, dim: int = -1, descending: bool = False,
///    values: Tensor, indices: Tensor)`
///
/// Pre-allocated output variant where `values` and `indices` tensors are
/// provided. Semantically identical to `sort.default`/`sort.stable`.
pub(super) fn map_sort_values(node: &Node) -> Result<(TraceOp, Vec<String>), ImportError> {
    let input = first_tensor_name(node)?;
    let dim = safe_usize(optional_int(node, "dim").unwrap_or(0), "dim", &node.target)?;
    let descending = optional_bool(node, "descending", false);
    Ok((TraceOp::Sort { dim, descending }, vec![input]))
}

// =========================================================================
// Unique: return unique elements (NEW)
// =========================================================================

/// Map `aten._unique2.default` to `TraceOp::Custom`.
///
/// torch.export signature:
///   `(self, sorted: bool = True, return_inverse: bool = False,
///    return_counts: bool = False)`
///
/// Returns unique elements of the input tensor. Output size is data-dependent
/// (depends on number of distinct values), so this maps to a Custom op.
/// The `sorted`, `return_inverse`, and `return_counts` flags control which
/// auxiliary tensors are produced.
pub(super) fn map_unique2(node: &Node) -> Result<(TraceOp, Vec<String>), ImportError> {
    let input = first_tensor_name(node)?;
    let sorted = optional_bool(node, "sorted", true);
    let return_inverse = optional_bool(node, "return_inverse", false);
    let return_counts = optional_bool(node, "return_counts", false);
    Ok((
        TraceOp::Custom {
            name: format!("unique_sorted{sorted}_inv{return_inverse}_cnt{return_counts}"),
        },
        vec![input],
    ))
}

/// Map `aten.unique_dim.default` to `TraceOp::Custom`.
///
/// torch.export signature:
///   `(self, dim: int, sorted: bool = True, return_inverse: bool = False,
///    return_counts: bool = False)`
///
/// Returns unique slices along a given dimension. Like `_unique2` but
/// operates along a specific axis.
pub(super) fn map_unique_dim(node: &Node) -> Result<(TraceOp, Vec<String>), ImportError> {
    let input = first_tensor_name(node)?;
    let dim = safe_usize(require_int(node, "dim")?, "dim", &node.target)?;
    let sorted = optional_bool(node, "sorted", true);
    let return_inverse = optional_bool(node, "return_inverse", false);
    let return_counts = optional_bool(node, "return_counts", false);
    Ok((
        TraceOp::Custom {
            name: format!("unique_dim{dim}_sorted{sorted}_inv{return_inverse}_cnt{return_counts}"),
        },
        vec![input],
    ))
}

// =========================================================================
// Unique consecutive: consecutive unique elements (NEW)
// =========================================================================

/// Map `aten.unique_consecutive.default` to `TraceOp::Custom`.
///
/// torch.export signature:
///   `(self, return_inverse: bool = False, return_counts: bool = False,
///    dim: int? = None)`
///
/// Removes consecutive duplicate elements. Unlike `unique`, this only
/// eliminates duplicates that are adjacent. Output size is data-dependent.
/// When `dim` is None, operates on flattened input.
pub(super) fn map_unique_consecutive(node: &Node) -> Result<(TraceOp, Vec<String>), ImportError> {
    let input = first_tensor_name(node)?;
    let return_inverse = optional_bool(node, "return_inverse", false);
    let return_counts = optional_bool(node, "return_counts", false);
    let dim = optional_int(node, "dim");

    let dim_str = match dim {
        Some(d) => format!("dim{d}"),
        None => "flat".to_string(),
    };
    Ok((
        TraceOp::Custom {
            name: format!("unique_consecutive_{dim_str}_inv{return_inverse}_cnt{return_counts}"),
        },
        vec![input],
    ))
}

// =========================================================================
// Tests
// =========================================================================

#[cfg(test)]
#[path = "op_map_impl_wave13_tests.rs"]
mod tests;

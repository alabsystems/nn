// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Aten op mappers for commonly needed PyTorch ops (Wave 14).
//!
//! Adds support for:
//!
//! - Interpolation: `lerp.Scalar`, `lerp.Tensor`
//! - Fused mul-add: `addcmul.default`, `addcdiv.default`
//! - Norm: `linalg_vector_norm.default`
//! - Distance: `cdist.default`
//! - Sampling: `multinomial.default`
//! - Search: `searchsorted.Tensor`, `bucketize.Tensor`
//! - Counting: `count_nonzero.default`, `count_nonzero.dim_IntList`
//! - Cumulative: `cumprod.default`, `cummax.default`, `cummin.default`
//! - Encoding: `one_hot.default`
//! - Activation: `threshold.default`, `threshold_.default`

use nn_core::dyn_tensor::trace::TraceOp;

use super::{
    first_tensor_name, get_arg, optional_bool, optional_float, optional_int, require_int,
    require_ints, require_tensor_name, safe_usize, ImportError, Node,
};

// =========================================================================
// Lerp: linear interpolation
// =========================================================================

/// Map `aten.lerp.Scalar` to `TraceOp::Custom`.
///
/// torch.export signature: `(self, end: Tensor, weight: Scalar)`
///
/// Computes `self + weight * (end - self)` element-wise.
/// The scalar weight controls the interpolation fraction.
pub(super) fn map_lerp_scalar(node: &Node) -> Result<(TraceOp, Vec<String>), ImportError> {
    let self_input = require_tensor_name(node, "self")?;
    let end = require_tensor_name(node, "end")?;
    let weight = optional_float(node, "weight").unwrap_or(0.5);
    Ok((
        TraceOp::Custom {
            name: format!("lerp_scalar_{weight}"),
        },
        vec![self_input, end],
    ))
}

/// Map `aten.lerp.Tensor` to `TraceOp::Custom`.
///
/// torch.export signature: `(self, end: Tensor, weight: Tensor)`
///
/// Computes `self + weight * (end - self)` element-wise.
/// The weight tensor allows per-element interpolation fractions.
pub(super) fn map_lerp_tensor(node: &Node) -> Result<(TraceOp, Vec<String>), ImportError> {
    let self_input = require_tensor_name(node, "self")?;
    let end = require_tensor_name(node, "end")?;
    let weight = require_tensor_name(node, "weight")?;
    Ok((
        TraceOp::Custom {
            name: "lerp_tensor".to_string(),
        },
        vec![self_input, end, weight],
    ))
}

// =========================================================================
// Addcmul / Addcdiv: fused multiply-add/divide
// =========================================================================

/// Map `aten.addcmul.default` to `TraceOp::Custom`.
///
/// torch.export signature:
///   `(self, tensor1: Tensor, tensor2: Tensor, value: Scalar = 1)`
///
/// Computes `self + value * tensor1 * tensor2` element-wise.
pub(super) fn map_addcmul(node: &Node) -> Result<(TraceOp, Vec<String>), ImportError> {
    let self_input = require_tensor_name(node, "self")?;
    let tensor1 = require_tensor_name(node, "tensor1")?;
    let tensor2 = require_tensor_name(node, "tensor2")?;
    let value = optional_float(node, "value").unwrap_or(1.0);
    Ok((
        TraceOp::Custom {
            name: format!("addcmul_v{value}"),
        },
        vec![self_input, tensor1, tensor2],
    ))
}

/// Map `aten.addcdiv.default` to `TraceOp::Custom`.
///
/// torch.export signature:
///   `(self, tensor1: Tensor, tensor2: Tensor, value: Scalar = 1)`
///
/// Computes `self + value * tensor1 / tensor2` element-wise.
/// Common in optimizer update rules (e.g., Adam).
pub(super) fn map_addcdiv(node: &Node) -> Result<(TraceOp, Vec<String>), ImportError> {
    let self_input = require_tensor_name(node, "self")?;
    let tensor1 = require_tensor_name(node, "tensor1")?;
    let tensor2 = require_tensor_name(node, "tensor2")?;
    let value = optional_float(node, "value").unwrap_or(1.0);
    Ok((
        TraceOp::Custom {
            name: format!("addcdiv_v{value}"),
        },
        vec![self_input, tensor1, tensor2],
    ))
}

// =========================================================================
// Linalg vector norm
// =========================================================================

/// Map `aten.linalg_vector_norm.default` to `TraceOp::Custom`.
///
/// torch.export signature:
///   `(self, ord: Scalar = 2, dim: int[]? = None, keepdim: bool = False,
///    dtype: ScalarType? = None)`
///
/// Computes the vector norm of the input tensor along the specified
/// dimensions. The `ord` parameter selects the norm type (1, 2, inf, etc.).
pub(super) fn map_linalg_vector_norm(node: &Node) -> Result<(TraceOp, Vec<String>), ImportError> {
    let input = first_tensor_name(node)?;
    let ord = optional_float(node, "ord").unwrap_or(2.0);
    let keepdim = optional_bool(node, "keepdim", false);

    // dim may be a single int or int list; extract if present.
    let dim_str = match get_arg(node, "dim") {
        Ok(a) if !a.is_none() => {
            if let Some(dims) = a.as_ints() {
                format!(
                    "dim{}",
                    dims.iter()
                        .map(ToString::to_string)
                        .collect::<Vec<_>>()
                        .join("_")
                )
            } else if let Some(d) = a.as_int() {
                format!("dim{d}")
            } else {
                "all".to_string()
            }
        }
        _ => "all".to_string(),
    };

    Ok((
        TraceOp::Custom {
            name: format!("linalg_vector_norm_ord{ord}_{dim_str}_kd{keepdim}"),
        },
        vec![input],
    ))
}

// =========================================================================
// Pairwise distance (cdist)
// =========================================================================

/// Map `aten.cdist.default` to `TraceOp::Custom`.
///
/// torch.export signature:
///   `(x1: Tensor, x2: Tensor, p: float = 2.0, compute_mode: int? = None)`
///
/// Computes batched pairwise distance between rows of x1 and x2.
/// Output shape: `[..., P, R]` where P = x1.size(-2), R = x2.size(-2).
pub(super) fn map_cdist(node: &Node) -> Result<(TraceOp, Vec<String>), ImportError> {
    let x1 = require_tensor_name(node, "x1")?;
    let x2 = require_tensor_name(node, "x2")?;
    let p = optional_float(node, "p").unwrap_or(2.0);
    Ok((
        TraceOp::Custom {
            name: format!("cdist_p{p}"),
        },
        vec![x1, x2],
    ))
}

// =========================================================================
// Multinomial sampling
// =========================================================================

/// Map `aten.multinomial.default` to `TraceOp::Custom`.
///
/// torch.export signature:
///   `(self, num_samples: int, replacement: bool = False, generator: Generator? = None)`
///
/// Draws `num_samples` indices from a multinomial distribution defined by
/// the input probability/weight tensor. Output is data-dependent (stochastic),
/// so this maps to a Custom op.
pub(super) fn map_multinomial(node: &Node) -> Result<(TraceOp, Vec<String>), ImportError> {
    let input = first_tensor_name(node)?;
    let num_samples = safe_usize(
        require_int(node, "num_samples")?,
        "num_samples",
        &node.target,
    )?;
    let replacement = optional_bool(node, "replacement", false);
    Ok((
        TraceOp::Custom {
            name: format!("multinomial_n{num_samples}_repl{replacement}"),
        },
        vec![input],
    ))
}

// =========================================================================
// Searchsorted: binary search in sorted sequence
// =========================================================================

/// Map `aten.searchsorted.Tensor` to `TraceOp::Custom`.
///
/// torch.export signature:
///   `(sorted_sequence: Tensor, self: Tensor, out_int32: bool = False,
///    right: bool = False, side: str? = None, sorter: Tensor? = None)`
///
/// Finds insertion points for values in a sorted sequence.
/// Returns int64 (or int32 if `out_int32=True`) indices.
pub(super) fn map_searchsorted(node: &Node) -> Result<(TraceOp, Vec<String>), ImportError> {
    let sorted_seq = require_tensor_name(node, "sorted_sequence")?;
    let values = require_tensor_name(node, "self")?;
    let right = optional_bool(node, "right", false);
    Ok((
        TraceOp::Custom {
            name: format!("searchsorted_right{right}"),
        },
        vec![sorted_seq, values],
    ))
}

// =========================================================================
// Bucketize: bin elements into buckets
// =========================================================================

/// Map `aten.bucketize.Tensor` to `TraceOp::Custom`.
///
/// torch.export signature:
///   `(self, boundaries: Tensor, out_int32: bool = False, right: bool = False)`
///
/// Finds bucket indices for each element based on sorted boundaries.
/// Equivalent to `searchsorted` on the boundaries tensor.
pub(super) fn map_bucketize(node: &Node) -> Result<(TraceOp, Vec<String>), ImportError> {
    let input = require_tensor_name(node, "self")?;
    let boundaries = require_tensor_name(node, "boundaries")?;
    let right = optional_bool(node, "right", false);
    Ok((
        TraceOp::Custom {
            name: format!("bucketize_right{right}"),
        },
        vec![input, boundaries],
    ))
}

// =========================================================================
// Count nonzero
// =========================================================================

/// Map `aten.count_nonzero.default` to `TraceOp::Custom`.
///
/// torch.export signature: `(self, dim: int? = None)`
///
/// Counts the number of non-zero elements along a dimension.
/// If dim is None, counts all non-zero elements and returns a scalar.
pub(super) fn map_count_nonzero(node: &Node) -> Result<(TraceOp, Vec<String>), ImportError> {
    let input = first_tensor_name(node)?;
    let dim = optional_int(node, "dim");
    let dim_str = match dim {
        Some(d) => format!("dim{d}"),
        None => "all".to_string(),
    };
    Ok((
        TraceOp::Custom {
            name: format!("count_nonzero_{dim_str}"),
        },
        vec![input],
    ))
}

/// Map `aten.count_nonzero.dim_IntList` to `TraceOp::Custom`.
///
/// torch.export signature: `(self, dim: int[])`
///
/// Multi-axis variant: counts non-zero elements along multiple dimensions.
pub(super) fn map_count_nonzero_dims(node: &Node) -> Result<(TraceOp, Vec<String>), ImportError> {
    let input = first_tensor_name(node)?;
    let dims = require_ints(node, "dim")?;
    let dim_str = dims
        .iter()
        .map(ToString::to_string)
        .collect::<Vec<_>>()
        .join("_");
    Ok((
        TraceOp::Custom {
            name: format!("count_nonzero_dim{dim_str}"),
        },
        vec![input],
    ))
}

// =========================================================================
// Cumulative product
// =========================================================================

/// Map `aten.cumprod.default` to `TraceOp::Custom`.
///
/// torch.export signature: `(self, dim: int, dtype: ScalarType? = None)`
///
/// Computes the cumulative product along a dimension.
/// Unlike cumsum (which has a dedicated TraceOp), cumprod maps to Custom
/// since it's less common in inference pipelines.
pub(super) fn map_cumprod(node: &Node) -> Result<(TraceOp, Vec<String>), ImportError> {
    let input = first_tensor_name(node)?;
    let dim = safe_usize(require_int(node, "dim")?, "dim", &node.target)?;
    Ok((
        TraceOp::Custom {
            name: format!("cumprod_dim{dim}"),
        },
        vec![input],
    ))
}

// =========================================================================
// Cumulative max / min
// =========================================================================

/// Map `aten.cummax.default` to `TraceOp::Custom`.
///
/// torch.export signature: `(self, dim: int)`
///
/// Returns a tuple of (values, indices) where values are the cumulative
/// maximum along a dimension. Maps to Custom since output is a tuple.
pub(super) fn map_cummax(node: &Node) -> Result<(TraceOp, Vec<String>), ImportError> {
    let input = first_tensor_name(node)?;
    let dim = safe_usize(require_int(node, "dim")?, "dim", &node.target)?;
    Ok((
        TraceOp::Custom {
            name: format!("cummax_dim{dim}"),
        },
        vec![input],
    ))
}

/// Map `aten.cummin.default` to `TraceOp::Custom`.
///
/// torch.export signature: `(self, dim: int)`
///
/// Returns a tuple of (values, indices) where values are the cumulative
/// minimum along a dimension. Maps to Custom since output is a tuple.
pub(super) fn map_cummin(node: &Node) -> Result<(TraceOp, Vec<String>), ImportError> {
    let input = first_tensor_name(node)?;
    let dim = safe_usize(require_int(node, "dim")?, "dim", &node.target)?;
    Ok((
        TraceOp::Custom {
            name: format!("cummin_dim{dim}"),
        },
        vec![input],
    ))
}

// =========================================================================
// One-hot encoding
// =========================================================================

/// Map `aten.one_hot.default` to `TraceOp::Custom`.
///
/// torch.export signature: `(self, num_classes: int = -1)`
///
/// Converts an integer tensor to one-hot encoding. If `num_classes` is -1,
/// it is inferred from the maximum value in the input.
/// Output shape: `[*input_shape, num_classes]`.
pub(super) fn map_one_hot(node: &Node) -> Result<(TraceOp, Vec<String>), ImportError> {
    let input = first_tensor_name(node)?;
    let num_classes = optional_int(node, "num_classes").unwrap_or(-1);
    Ok((
        TraceOp::Custom {
            name: format!("one_hot_nc{num_classes}"),
        },
        vec![input],
    ))
}

// =========================================================================
// Threshold activation
// =========================================================================

/// Map `aten.threshold.default` / `aten.threshold_.default` to `TraceOp::Custom`.
///
/// torch.export signature: `(self, threshold: Scalar, value: Scalar)`
///
/// Thresholds each element: `x if x > threshold, else value`.
/// This is a generalization of ReLU (threshold=0, value=0).
pub(super) fn map_threshold(node: &Node) -> Result<(TraceOp, Vec<String>), ImportError> {
    let input = first_tensor_name(node)?;
    let threshold = optional_float(node, "threshold").unwrap_or(0.0);
    let value = optional_float(node, "value").unwrap_or(0.0);
    Ok((
        TraceOp::Custom {
            name: format!("threshold_t{threshold}_v{value}"),
        },
        vec![input],
    ))
}

// =========================================================================
// Tests
// =========================================================================

#[cfg(test)]
#[path = "op_map_impl_wave14_tests.rs"]
mod tests;

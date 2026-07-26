// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Aten op mappers for commonly needed PyTorch ops (Wave 15).
//!
//! Adds support for:
//!
//! - Clamp: `clamp.Tensor` (tensor min/max bounds)
//! - Norm: `norm.ScalarOpt_dim` (Lp norm along dims)
//! - Einsum: `einsum.default` (Einstein summation)
//! - Strided view: `as_strided.default` (view with custom strides)
//! - Matrix ops: `addmv.default`, `addr.default`, `outer.default`
//! - Sampling: `bernoulli.default`, `bernoulli_.float`, `randn.default`
//! - Cross product: `cross.default`

use nn_core::dyn_tensor::trace::TraceOp;

use super::{
    first_tensor_name, get_arg, optional_bool, optional_float, optional_int,
    require_ints, require_tensor_name, ImportError, Node,
};

// =========================================================================
// Clamp with tensor bounds
// =========================================================================

/// Map `aten.clamp.Tensor` to `TraceOp::Custom`.
///
/// torch.export signature: `(self, min: Tensor?, max: Tensor?)`
///
/// Clamps each element of `self` to the range `[min, max]` where min and max
/// are tensors (broadcast-compatible). Unlike `clamp.default` which takes
/// scalar bounds, this variant allows per-element clamping ranges.
pub(super) fn map_clamp_tensor(node: &Node) -> Result<(TraceOp, Vec<String>), ImportError> {
    let input = first_tensor_name(node)?;
    let mut inputs = vec![input];

    let has_min = get_arg(node, "min")
        .ok()
        .is_some_and(|a| !a.is_none() && a.as_tensor_name().is_some());
    let has_max = get_arg(node, "max")
        .ok()
        .is_some_and(|a| !a.is_none() && a.as_tensor_name().is_some());

    if has_min {
        inputs.push(require_tensor_name(node, "min")?);
    }
    if has_max {
        inputs.push(require_tensor_name(node, "max")?);
    }

    let suffix = match (has_min, has_max) {
        (true, true) => "min_max",
        (true, false) => "min_only",
        (false, true) => "max_only",
        (false, false) => "noop",
    };

    Ok((
        TraceOp::Custom {
            name: format!("clamp_tensor_{suffix}"),
        },
        inputs,
    ))
}

// =========================================================================
// Norm: Lp norm along dimensions
// =========================================================================

/// Map `aten.norm.ScalarOpt_dim` to `TraceOp::Custom`.
///
/// torch.export signature:
///   `(self, p: Scalar?, dim: int[], keepdim: bool = False)`
///
/// Computes the Lp norm of the input tensor along the specified dimensions.
/// `p` defaults to 2.0 (Frobenius norm). Commonly used in weight
/// normalization and gradient clipping.
pub(super) fn map_norm(node: &Node) -> Result<(TraceOp, Vec<String>), ImportError> {
    let input = first_tensor_name(node)?;
    let p = optional_float(node, "p").unwrap_or(2.0);
    let keepdim = optional_bool(node, "keepdim", false);

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
            name: format!("norm_p{p}_{dim_str}_kd{keepdim}"),
        },
        vec![input],
    ))
}

// =========================================================================
// Einsum: Einstein summation
// =========================================================================

/// Map `aten.einsum.default` to `TraceOp::Custom`.
///
/// torch.export signature:
///   `(equation: str, tensors: Tensor[], path: int[]? = None)`
///
/// Einstein summation convention. Supports arbitrary contraction patterns
/// over multiple input tensors. Common examples:
/// - `"ij,jk->ik"` (matmul)
/// - `"bhqd,bhkd->bhqk"` (attention scores)
/// - `"...ii->...i"` (diagonal extraction)
pub(super) fn map_einsum(node: &Node) -> Result<(TraceOp, Vec<String>), ImportError> {
    // The equation is a string argument.
    let equation = get_arg(node, "equation")
        .ok()
        .and_then(|a| a.as_string())
        .unwrap_or_default()
        .to_string();

    // Collect all tensor inputs from the "tensors" list argument,
    // or fall back to positional tensor inputs after the equation.
    let mut inputs = Vec::new();
    if let Ok(arg) = get_arg(node, "tensors") {
        if let Some(names) = arg.as_tensor_names() {
            inputs.extend(names.into_iter().map(String::from));
        }
    }
    // Fallback: collect any remaining tensor args by position.
    if inputs.is_empty() {
        for na in &node.inputs {
            if let Some(name) = na.arg.as_tensor_name() {
                inputs.push(name.to_string());
            }
        }
    }

    Ok((
        TraceOp::Custom {
            name: format!("einsum_{equation}"),
        },
        inputs,
    ))
}

// =========================================================================
// As-strided view
// =========================================================================

/// Map `aten.as_strided.default` to `TraceOp::Custom`.
///
/// torch.export signature:
///   `(self, size: int[], stride: int[], storage_offset: int? = None)`
///
/// Creates a view of the tensor with the given size and stride.
/// Common in torch.export outputs where view operations are lowered to
/// as_strided for explicit memory layout control.
pub(super) fn map_as_strided(node: &Node) -> Result<(TraceOp, Vec<String>), ImportError> {
    let input = first_tensor_name(node)?;
    let size = require_ints(node, "size")?;
    let stride = require_ints(node, "stride")?;
    let offset = optional_int(node, "storage_offset").unwrap_or(0);

    let size_str = size
        .iter()
        .map(ToString::to_string)
        .collect::<Vec<_>>()
        .join("x");
    let stride_str = stride
        .iter()
        .map(ToString::to_string)
        .collect::<Vec<_>>()
        .join("x");

    Ok((
        TraceOp::Custom {
            name: format!("as_strided_sz{size_str}_st{stride_str}_off{offset}"),
        },
        vec![input],
    ))
}

// =========================================================================
// Matrix-vector multiply-add: addmv
// =========================================================================

/// Map `aten.addmv.default` to `TraceOp::Custom`.
///
/// torch.export signature:
///   `(self, mat: Tensor, vec: Tensor, beta: Scalar = 1, alpha: Scalar = 1)`
///
/// Computes `beta * self + alpha * (mat @ vec)`.
/// Common in linear layers with bias pre-addition.
pub(super) fn map_addmv(node: &Node) -> Result<(TraceOp, Vec<String>), ImportError> {
    let self_input = require_tensor_name(node, "self")?;
    let mat = require_tensor_name(node, "mat")?;
    let vec_input = require_tensor_name(node, "vec")?;
    let beta = optional_float(node, "beta").unwrap_or(1.0);
    let alpha = optional_float(node, "alpha").unwrap_or(1.0);

    Ok((
        TraceOp::Custom {
            name: format!("addmv_b{beta}_a{alpha}"),
        },
        vec![self_input, mat, vec_input],
    ))
}

// =========================================================================
// Additive outer product: addr
// =========================================================================

/// Map `aten.addr.default` to `TraceOp::Custom`.
///
/// torch.export signature:
///   `(self, vec1: Tensor, vec2: Tensor, beta: Scalar = 1, alpha: Scalar = 1)`
///
/// Computes `beta * self + alpha * (vec1 outer vec2)`.
/// Output shape: `[len(vec1), len(vec2)]`.
pub(super) fn map_addr(node: &Node) -> Result<(TraceOp, Vec<String>), ImportError> {
    let self_input = require_tensor_name(node, "self")?;
    let vec1 = require_tensor_name(node, "vec1")?;
    let vec2 = require_tensor_name(node, "vec2")?;
    let beta = optional_float(node, "beta").unwrap_or(1.0);
    let alpha = optional_float(node, "alpha").unwrap_or(1.0);

    Ok((
        TraceOp::Custom {
            name: format!("addr_b{beta}_a{alpha}"),
        },
        vec![self_input, vec1, vec2],
    ))
}

// =========================================================================
// Outer product
// =========================================================================

/// Map `aten.outer.default` to `TraceOp::Custom`.
///
/// torch.export signature: `(self, vec2: Tensor)`
///
/// Computes the outer product of two 1-D tensors.
/// Output shape: `[len(self), len(vec2)]`.
pub(super) fn map_outer(node: &Node) -> Result<(TraceOp, Vec<String>), ImportError> {
    let self_input = require_tensor_name(node, "self")?;
    let vec2 = require_tensor_name(node, "vec2")?;

    Ok((
        TraceOp::Custom {
            name: "outer".to_string(),
        },
        vec![self_input, vec2],
    ))
}

// =========================================================================
// Bernoulli sampling
// =========================================================================

/// Map `aten.bernoulli.default` to `TraceOp::Custom`.
///
/// torch.export signature: `(self, generator: Generator? = None)`
///
/// Draws binary random values from a Bernoulli distribution. Each element
/// of the output is 1 with probability given by the corresponding element
/// of the input, and 0 otherwise. Common in dropout during training.
pub(super) fn map_bernoulli(node: &Node) -> Result<(TraceOp, Vec<String>), ImportError> {
    let input = first_tensor_name(node)?;
    Ok((
        TraceOp::Custom {
            name: "bernoulli".to_string(),
        },
        vec![input],
    ))
}

/// Map `aten.bernoulli_.float` to `TraceOp::Custom`.
///
/// torch.export signature: `(self, p: float = 0.5, generator: Generator? = None)`
///
/// In-place Bernoulli with a scalar probability. Fills the tensor with
/// samples from Bernoulli(p). This variant is common in exported
/// training graphs where dropout uses `bernoulli_`.
pub(super) fn map_bernoulli_float(node: &Node) -> Result<(TraceOp, Vec<String>), ImportError> {
    let input = first_tensor_name(node)?;
    let p = optional_float(node, "p").unwrap_or(0.5);
    Ok((
        TraceOp::Custom {
            name: format!("bernoulli_p{p}"),
        },
        vec![input],
    ))
}

// =========================================================================
// Random normal tensor creation
// =========================================================================

/// Map `aten.randn.default` to `TraceOp::Custom`.
///
/// torch.export signature:
///   `(size: int[], dtype: ScalarType? = None, layout: Layout? = None,
///    device: Device? = None, pin_memory: bool? = None)`
///
/// Creates a tensor filled with random numbers from a standard normal
/// distribution N(0, 1). Common in VAE reparameterization, diffusion
/// models, and noise injection during training.
pub(super) fn map_randn(node: &Node) -> Result<(TraceOp, Vec<String>), ImportError> {
    let size = require_ints(node, "size")?;
    let size_str = size
        .iter()
        .map(ToString::to_string)
        .collect::<Vec<_>>()
        .join("x");

    Ok((
        TraceOp::Custom {
            name: format!("randn_{size_str}"),
        },
        vec![],
    ))
}

// =========================================================================
// Cross product
// =========================================================================

/// Map `aten.cross.default` to `TraceOp::Custom`.
///
/// torch.export signature: `(self, other: Tensor, dim: int? = None)`
///
/// Computes the cross product of two 3-element vectors along the given
/// dimension. Used in 3D geometry operations and physics simulations.
pub(super) fn map_cross(node: &Node) -> Result<(TraceOp, Vec<String>), ImportError> {
    let self_input = require_tensor_name(node, "self")?;
    let other = require_tensor_name(node, "other")?;
    let dim = optional_int(node, "dim");
    let dim_str = match dim {
        Some(d) => format!("dim{d}"),
        None => "auto".to_string(),
    };

    Ok((
        TraceOp::Custom {
            name: format!("cross_{dim_str}"),
        },
        vec![self_input, other],
    ))
}

// =========================================================================
// Tests
// =========================================================================

#[cfg(test)]
#[path = "op_map_impl_wave15_tests.rs"]
mod tests;

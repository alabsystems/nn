// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Aten op mappers for spatial transformer, grid, and shape ops (Wave 11).
//!
//! Adds support for:
//!
//! - Spatial: affine_grid_generator (STN / spatial transformers)
//! - Grid: meshgrid direct mapper (fallback when expand path unavailable)
//! - Stacking: stack direct mapper (fallback when expand path unavailable)
//! - Splitting: split/chunk direct mappers (fallback when expand path unavailable)
//! - Expansion: repeat/expand additional overloads (repeat.Tensor, expand.Scalar)
//! - Masking: masked_fill direct mapper (scalar fallback improvement)
//! - Triangular: triu_/tril_ in-place overloads
//! - Creation: arange.start_stop, linspace.out, affine_grid_generator

use nn_core::dyn_tensor::trace::TraceOp;

use super::{
    first_tensor_name, get_arg, optional_bool, optional_float, optional_int, require_int,
    require_ints, require_tensor_name, safe_usize, ImportError, Node,
};

// =========================================================================
// Affine grid generator (Spatial Transformer Networks)
// =========================================================================

/// Map `aten.affine_grid_generator.default` to `TraceOp::Custom`.
///
/// torch.export signature: `(theta: Tensor, size: [int...], align_corners: bool)`
///
/// Generates a 2D affine sampling grid from an affine transformation matrix
/// `theta` of shape `[N, 2, 3]`. The output grid has shape `[N, H, W, 2]`
/// where `H` and `W` come from the `size` argument `[N, C, H, W]`.
///
/// Used in Spatial Transformer Networks (STN) to produce grid coordinates
/// for `grid_sample`.
pub(super) fn map_affine_grid_generator(
    node: &Node,
) -> Result<(TraceOp, Vec<String>), ImportError> {
    let theta = first_tensor_name(node)?;
    let size = require_ints(node, "size")?;
    let align_corners = optional_bool(node, "align_corners", false);

    // Validate size has 4 elements: [N, C, H, W]
    if size.len() != 4 {
        return Err(ImportError::WrongArgumentType {
            op_target: node.target.clone(),
            arg_name: "size".to_string(),
            expected: "4-element list [N, C, H, W]",
            actual: format!("{}-element list", size.len()),
        });
    }

    let h = safe_usize(size[2], "size[2]", &node.target)?;
    let w = safe_usize(size[3], "size[3]", &node.target)?;

    Ok((
        TraceOp::Custom {
            name: format!("affine_grid_generator_h{h}_w{w}_align{align_corners}"),
        },
        vec![theta],
    ))
}

// =========================================================================
// Meshgrid direct mapper (fallback when shape metadata unavailable)
// =========================================================================

/// Map `aten.meshgrid.default` / `aten.meshgrid.indexing` to `TraceOp::Custom`.
///
/// torch.export signature: `(tensors: [Tensor...], indexing: str = "ij")`
///
/// Creates coordinate grids from 1-D coordinate vectors. When input shapes
/// are available, this is decomposed via `try_expand_node` into Expand +
/// Reshape. This fallback handles the case where shapes are unavailable.
pub(super) fn map_meshgrid(node: &Node) -> Result<(TraceOp, Vec<String>), ImportError> {
    let indexing = get_arg(node, "indexing")
        .ok()
        .and_then(|a| a.as_string().map(String::from))
        .unwrap_or_else(|| "ij".to_string());

    // Collect all tensor inputs
    let mut inputs = Vec::new();
    for na in &node.inputs {
        if let Some(name) = na.arg.as_tensor_name() {
            inputs.push(name.to_string());
        } else if let Some(names) = na.arg.as_tensor_names() {
            inputs.extend(names.iter().map(ToString::to_string));
        }
    }

    if inputs.is_empty() {
        return Err(ImportError::MissingArgument {
            op_target: node.target.clone(),
            arg_name: "tensors".to_string(),
        });
    }

    let num_inputs = inputs.len();
    Ok((
        TraceOp::Custom {
            name: format!("meshgrid_{indexing}_n{num_inputs}"),
        },
        inputs,
    ))
}

// =========================================================================
// Stack direct mapper (fallback when shape metadata unavailable)
// =========================================================================

/// Map `aten.stack.default` to `TraceOp::Custom`.
///
/// torch.export signature: `(tensors: [Tensor...], dim: int = 0)`
///
/// Stacks a sequence of tensors along a new dimension. When input shapes
/// are available, this is decomposed via `try_expand_node` into N Unsqueeze
/// ops + 1 Cat. This fallback handles the case where shapes are unavailable.
pub(super) fn map_stack(node: &Node) -> Result<(TraceOp, Vec<String>), ImportError> {
    let dim = optional_int(node, "dim").unwrap_or(0);

    // Collect all tensor inputs
    let mut inputs = Vec::new();
    for na in &node.inputs {
        if na.name == "dim" {
            continue;
        }
        if let Some(name) = na.arg.as_tensor_name() {
            inputs.push(name.to_string());
        } else if let Some(names) = na.arg.as_tensor_names() {
            inputs.extend(names.iter().map(ToString::to_string));
        }
    }

    if inputs.is_empty() {
        return Err(ImportError::MissingArgument {
            op_target: node.target.clone(),
            arg_name: "tensors".to_string(),
        });
    }

    let num_inputs = inputs.len();
    Ok((
        TraceOp::Custom {
            name: format!("stack_dim{dim}_n{num_inputs}"),
        },
        inputs,
    ))
}

// =========================================================================
// Split / Chunk direct mappers (fallback when shape unavailable)
// =========================================================================

/// Map `aten.split.Tensor` to `TraceOp::Custom` (fallback).
///
/// torch.export signature: `(self, split_size: int, dim: int = 0)`
///
/// Splits a tensor into chunks of `split_size` along `dim`. When input
/// shapes are available, this is decomposed via `try_expand_node` into
/// N Narrow ops. This fallback handles the case where shapes are unavailable.
pub(super) fn map_split(node: &Node) -> Result<(TraceOp, Vec<String>), ImportError> {
    let input = first_tensor_name(node)?;
    let split_size = require_int(node, "split_size")?;
    let dim = optional_int(node, "dim").unwrap_or(0);

    Ok((
        TraceOp::Custom {
            name: format!("split_size{split_size}_dim{dim}"),
        },
        vec![input],
    ))
}

/// Map `aten.split_with_sizes.default` to `TraceOp::Custom` (fallback).
///
/// torch.export signature: `(self, split_sizes: [int...], dim: int = 0)`
///
/// Splits a tensor into chunks with specified sizes along `dim`.
pub(super) fn map_split_with_sizes(node: &Node) -> Result<(TraceOp, Vec<String>), ImportError> {
    let input = first_tensor_name(node)?;
    let sizes = require_ints(node, "split_sizes")?;
    let dim = optional_int(node, "dim").unwrap_or(0);

    let sizes_str: Vec<String> = sizes.iter().map(ToString::to_string).collect();
    Ok((
        TraceOp::Custom {
            name: format!("split_sizes_{}_dim{dim}", sizes_str.join("_")),
        },
        vec![input],
    ))
}

/// Map `aten.chunk.default` to `TraceOp::Custom` (fallback).
///
/// torch.export signature: `(self, chunks: int, dim: int = 0)`
///
/// Splits a tensor into `chunks` approximately equal pieces along `dim`.
/// When input shapes are available, this is decomposed via `try_expand_node`
/// into N Narrow ops. This fallback handles the case where shapes are
/// unavailable.
pub(super) fn map_chunk(node: &Node) -> Result<(TraceOp, Vec<String>), ImportError> {
    let input = first_tensor_name(node)?;
    let chunks = require_int(node, "chunks")?;
    let dim = optional_int(node, "dim").unwrap_or(0);

    Ok((
        TraceOp::Custom {
            name: format!("chunk_n{chunks}_dim{dim}"),
        },
        vec![input],
    ))
}

// =========================================================================
// Repeat / Expand additional overloads
// =========================================================================

/// Map `aten.repeat.default` with tensor `repeats` argument to `TraceOp::Custom`.
///
/// torch.export signature: `(self, repeats: [int...])`
///
/// This handles the case where `repeats` comes as a tensor list rather
/// than an int list. Complements the existing `map_repeat` in dpdf.
pub(super) fn map_repeat_tensor(node: &Node) -> Result<(TraceOp, Vec<String>), ImportError> {
    let input = first_tensor_name(node)?;
    // Try int list first, fall back to generic custom op
    let repeats = get_arg(node, "repeats")
        .ok()
        .and_then(|a| a.as_ints().map(<[i64]>::to_vec));

    if let Some(reps) = repeats {
        let target_shape: Vec<usize> = reps
            .into_iter()
            .map(|v| safe_usize(v, "repeats", &node.target))
            .collect::<Result<_, _>>()?;
        Ok((TraceOp::Expand { target_shape }, vec![input]))
    } else {
        Ok((
            TraceOp::Custom {
                name: "repeat_dynamic".to_string(),
            },
            vec![input],
        ))
    }
}

/// Map `aten.expand.default` with size containing -1 placeholders.
///
/// torch.export signature: `(self, size: [int...], implicit: bool = False)`
///
/// `-1` in `size` means "keep the existing dimension size". This handles
/// the common pattern of expanding only specific dimensions.
pub(super) fn map_expand_with_neg1(node: &Node) -> Result<(TraceOp, Vec<String>), ImportError> {
    let input = first_tensor_name(node)?;
    let size = require_ints(node, "size")?;

    // Convert, treating -1 as usize::MAX (sentinel for "keep dim")
    let target_shape: Vec<usize> = size
        .into_iter()
        .map(|v| {
            if v == -1 {
                Ok(usize::MAX)
            } else {
                safe_usize(v, "size", &node.target)
            }
        })
        .collect::<Result<_, _>>()?;

    Ok((TraceOp::Expand { target_shape }, vec![input]))
}

// =========================================================================
// Masked fill direct mapper (improved scalar fallback)
// =========================================================================

/// Map `aten.masked_fill.Scalar` to `TraceOp::WhereCond` with a constant.
///
/// torch.export signature: `(self, mask: Tensor, value: Scalar)`
///
/// `masked_fill(self, mask, value)` semantics: where mask is True, use `value`;
/// otherwise use `self`. This is equivalent to `where(mask, value, self)`.
///
/// When input shapes are available, `try_expand_node` decomposes this into
/// Constant + WhereCond. This direct mapper handles the fallback case by
/// encoding the fill value into a Custom op for downstream processing.
pub(super) fn map_masked_fill_scalar(node: &Node) -> Result<(TraceOp, Vec<String>), ImportError> {
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
// Triu / Tril in-place overloads
// =========================================================================

/// Map `aten.triu_.default` (in-place) to `TraceOp::Triu`.
///
/// torch.export signature: `(self, diagonal: int = 0)`
/// In-place variant: same semantics as `triu`, modifies self.
pub(super) fn map_triu_inplace(node: &Node) -> Result<(TraceOp, Vec<String>), ImportError> {
    let input = first_tensor_name(node)?;
    let diagonal = optional_int(node, "diagonal").unwrap_or(0);
    Ok((TraceOp::Triu { diagonal }, vec![input]))
}

/// Map `aten.tril_.default` (in-place) to `TraceOp::Tril`.
///
/// torch.export signature: `(self, diagonal: int = 0)`
/// In-place variant: same semantics as `tril`, modifies self.
pub(super) fn map_tril_inplace(node: &Node) -> Result<(TraceOp, Vec<String>), ImportError> {
    let input = first_tensor_name(node)?;
    let diagonal = optional_int(node, "diagonal").unwrap_or(0);
    Ok((TraceOp::Tril { diagonal }, vec![input]))
}

// =========================================================================
// Arange additional overloads
// =========================================================================

/// Map `aten.arange.start_stop` to `TraceOp::Arange`.
///
/// torch.export signature: `(start: Scalar, end: Scalar, ...)`
///
/// Two-argument arange with explicit start and end (step defaults to 1).
/// Complements the existing `arange.default` and `arange.start_step` mappers.
pub(super) fn map_arange_start_stop(node: &Node) -> Result<(TraceOp, Vec<String>), ImportError> {
    let start = optional_float(node, "start")
        .or_else(|| optional_int(node, "start").map(|i| i as f64))
        .unwrap_or(0.0);
    let end = optional_float(node, "end")
        .or_else(|| optional_int(node, "end").map(|i| i as f64))
        .unwrap_or(1.0);
    Ok((
        TraceOp::Arange {
            start,
            end,
            step: 1.0,
        },
        vec![],
    ))
}

// =========================================================================
// Linspace additional overloads
// =========================================================================

/// Map `aten.linspace.out` to `TraceOp::Arange`.
///
/// torch.export signature: `(start, end, steps, out: Tensor)`
///
/// Same as `linspace.default` but writes to an existing output tensor.
/// We ignore the `out` tensor and emit the same Arange decomposition.
pub(super) fn map_linspace_out(node: &Node) -> Result<(TraceOp, Vec<String>), ImportError> {
    let start = optional_float(node, "start")
        .or_else(|| optional_int(node, "start").map(|i| i as f64))
        .unwrap_or(0.0);
    let end = optional_float(node, "end")
        .or_else(|| optional_int(node, "end").map(|i| i as f64))
        .unwrap_or(1.0);
    let steps = optional_int(node, "steps").unwrap_or(100);
    let step = if steps > 1 {
        (end - start) / (steps - 1) as f64
    } else {
        0.0
    };
    Ok((
        TraceOp::Arange {
            start,
            end: end + step * 0.5,
            step,
        },
        vec![],
    ))
}

// =========================================================================
// Tests
// =========================================================================

#[cfg(test)]
#[path = "op_map_impl_wave11_tests.rs"]
mod tests;

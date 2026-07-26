// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Multi-node expansion helpers for `try_expand_node`.
//!
//! Decompose compound aten ops into multiple `ExpandedNode`s:
//! - Scalar binary ops (`add.Scalar`, etc.) → Constant + binary op
//! - `squeeze.default` → Reshape (removes all size-1 dims)
//! - `select.int` → Narrow + Reshape (select index, remove dim)
//! - Multi-axis reductions → sequential single-dim reduces

use nn_core::dyn_tensor::trace::TraceOp;
use nn_core::DType;

use super::{
    first_tensor_name, get_arg, optional_float, optional_int, require_int, safe_usize,
    ExpandedNode, ImportError, Node,
};

/// Expand a scalar binary op (e.g., `add.Scalar`) into a Constant node + binary op.
pub(super) fn expand_scalar_binary(
    node: &Node,
    output_name: &str,
    input_shape: &[usize],
    op: TraceOp,
) -> Result<Vec<ExpandedNode>, ImportError> {
    let input = first_tensor_name(node)?;
    let scalar = optional_float(node, "other")
        .or_else(|| optional_int(node, "other").map(|i| i as f64))
        .ok_or_else(|| ImportError::MissingArgument {
            op_target: node.target.clone(),
            arg_name: "other".to_string(),
        })?;
    let const_name = format!("{output_name}_const");
    Ok(vec![
        ExpandedNode {
            name: const_name.clone(),
            op: TraceOp::Constant { value: scalar },
            input_names: vec![],
            output_shape: vec![],
            output_dtype: DType::F32,
        },
        ExpandedNode {
            name: output_name.to_string(),
            op,
            input_names: vec![input, const_name],
            output_shape: input_shape.to_vec(),
            output_dtype: DType::F32,
        },
    ])
}

/// Expand `squeeze.default` (no dim arg) into a Reshape that removes all size-1 dims.
///
/// torch.export encodes `tensor.squeeze()` as `aten.squeeze.default`. Since we have the
/// concrete input shape from export metadata, we can compute the output shape statically
/// and emit a Reshape — no new TraceOp variant needed.
pub(super) fn expand_squeeze_default(
    node: &Node,
    output_name: &str,
    input_shape: &[usize],
) -> Result<Vec<ExpandedNode>, ImportError> {
    let input = first_tensor_name(node)?;
    let output_shape: Vec<usize> = input_shape.iter().copied().filter(|&s| s != 1).collect();
    Ok(vec![ExpandedNode {
        name: output_name.to_string(),
        op: TraceOp::Reshape {
            target_shape: output_shape.clone(),
        },
        input_names: vec![input],
        output_shape,
        output_dtype: DType::F32,
    }])
}

pub(super) fn make_reduce_sum(dim: usize, keepdim: bool) -> TraceOp {
    TraceOp::ReduceSum { dim, keepdim }
}
pub(super) fn make_reduce_mean(dim: usize, keepdim: bool) -> TraceOp {
    TraceOp::ReduceMean { dim, keepdim }
}
pub(super) fn make_reduce_max(dim: usize, keepdim: bool) -> TraceOp {
    TraceOp::ReduceMax { dim, keepdim }
}
pub(super) fn make_reduce_min(dim: usize, keepdim: bool) -> TraceOp {
    TraceOp::ReduceMin { dim, keepdim }
}

/// Expand a multi-axis reduction (e.g., `sum([1, 2])`) into sequential single-dim reduces.
///
/// All intermediate reduces use `keepdim=true` to preserve dimension indices.
/// If the overall `keepdim` is false, a final Reshape removes the reduced dims.
pub(super) fn expand_multi_axis_reduce(
    node: &Node,
    output_name: &str,
    input_shape: &[usize],
    make_op: fn(usize, bool) -> TraceOp,
    dims: &[i64],
    keepdim: bool,
) -> Result<Vec<ExpandedNode>, ImportError> {
    let input = first_tensor_name(node)?;
    let udims: Vec<usize> = dims
        .iter()
        .map(|&d| safe_usize(d, "dim", &node.target))
        .collect::<Result<_, _>>()?;

    let mut nodes = Vec::new();
    let mut current_shape = input_shape.to_vec();
    let mut current_input = input;

    for (i, &dim) in udims.iter().enumerate() {
        let is_last = i == udims.len() - 1;
        // Final node gets the output name only if keepdim=true (no reshape needed).
        let name = if is_last && keepdim {
            output_name.to_string()
        } else {
            format!("{output_name}_reduce_{i}")
        };

        current_shape[dim] = 1;
        nodes.push(ExpandedNode {
            name: name.clone(),
            op: make_op(dim, true), // all intermediates keepdim=true
            input_names: vec![current_input],
            output_shape: current_shape.clone(),
            output_dtype: DType::F32,
        });
        current_input = name;
    }

    // If keepdim=false, add final Reshape to remove the reduced dimensions.
    if !keepdim {
        let final_shape: Vec<usize> = input_shape
            .iter()
            .enumerate()
            .filter(|(i, _)| !udims.contains(i))
            .map(|(_, &s)| s)
            .collect();
        nodes.push(ExpandedNode {
            name: output_name.to_string(),
            op: TraceOp::Reshape {
                target_shape: final_shape.clone(),
            },
            input_names: vec![current_input],
            output_shape: final_shape,
            output_dtype: DType::F32,
        });
    }

    Ok(nodes)
}

/// Expand `flatten.using_ints(self, start_dim, end_dim)` into a Reshape.
///
/// `torch.flatten(x, start_dim=1, end_dim=-1)` collapses dimensions [start_dim, end_dim]
/// into a single dimension. We compute the flattened shape from the known input shape
/// and emit a Reshape.
pub(super) fn expand_flatten(
    node: &Node,
    output_name: &str,
    input_shape: &[usize],
) -> Result<Vec<ExpandedNode>, ImportError> {
    let input = first_tensor_name(node)?;
    let ndim = input_shape.len();
    let start_raw = optional_int(node, "start_dim").unwrap_or(0);
    let end_raw = optional_int(node, "end_dim").unwrap_or(-1);

    let start = if start_raw < 0 {
        safe_usize(start_raw + ndim as i64, "start_dim", &node.target)?
    } else {
        safe_usize(start_raw, "start_dim", &node.target)?
    };
    let end = if end_raw < 0 {
        safe_usize(end_raw + ndim as i64, "end_dim", &node.target)?
    } else {
        safe_usize(end_raw, "end_dim", &node.target)?
    };

    // Compute flattened shape: dims before start + product(start..=end) + dims after end.
    let mut output_shape = Vec::new();
    for &s in &input_shape[..start] {
        output_shape.push(s);
    }
    let flat_size: usize = input_shape[start..=end].iter().product();
    output_shape.push(flat_size);
    for &s in &input_shape[end + 1..] {
        output_shape.push(s);
    }

    Ok(vec![ExpandedNode {
        name: output_name.to_string(),
        op: TraceOp::Reshape {
            target_shape: output_shape.clone(),
        },
        input_names: vec![input],
        output_shape,
        output_dtype: DType::F32,
    }])
}

/// Expand `chunk(self, chunks, dim)` into N Narrow ops.
///
/// `torch.chunk(x, 2, dim=1)` on shape `[1, 256, T]` produces two `[1, 128, T]` tensors.
/// We decompose into individual Narrow ops, one per output tensor. Output names come from
/// the node's `as_tensors` multi-output list.
pub(super) fn expand_chunk(
    node: &Node,
    _output_name: &str,
    input_shape: &[usize],
) -> Result<Vec<ExpandedNode>, ImportError> {
    let input = first_tensor_name(node)?;
    let chunks = safe_usize(require_int(node, "chunks")?, "chunks", &node.target)?;
    let dim_raw = optional_int(node, "dim").unwrap_or(0);
    let ndim = input_shape.len();
    let dim = if dim_raw < 0 {
        safe_usize(dim_raw + ndim as i64, "dim", &node.target)?
    } else {
        safe_usize(dim_raw, "dim", &node.target)?
    };

    let dim_size = input_shape.get(dim).copied().unwrap_or(0);
    let chunk_size = dim_size.div_ceil(chunks);

    // Extract output tensor names from the multi-output `as_tensors`.
    let output_names: Vec<String> = node
        .outputs
        .first()
        .and_then(super::Argument::as_tensor_names)
        .map(|names| names.into_iter().map(String::from).collect())
        .unwrap_or_default();

    let num_outputs = output_names.len().max(chunks);
    let mut expanded = Vec::new();
    let mut start = 0;

    for i in 0..num_outputs {
        let length = chunk_size.min(dim_size.saturating_sub(start));
        let name = output_names
            .get(i)
            .cloned()
            .unwrap_or_else(|| format!("{_output_name}_chunk_{i}"));

        let mut out_shape = input_shape.to_vec();
        if dim < out_shape.len() {
            out_shape[dim] = length;
        }

        expanded.push(ExpandedNode {
            name,
            op: TraceOp::Narrow { dim, start, length },
            input_names: vec![input.clone()],
            output_shape: out_shape,
            output_dtype: DType::F32,
        });
        start += length;
    }

    Ok(expanded)
}

/// Expand `select.int` (select a single index along a dim) into Narrow + Reshape.
///
/// `select(dim, index)` returns a tensor with dimension `dim` removed. We decompose
/// this into `Narrow(dim, index, 1)` (keeps dim with size 1) followed by a `Reshape`
/// that removes the dim.
pub(super) fn expand_select_int(
    node: &Node,
    output_name: &str,
    input_shape: &[usize],
) -> Result<Vec<ExpandedNode>, ImportError> {
    let input = first_tensor_name(node)?;
    let dim_raw = require_int(node, "dim").or_else(|_| require_int(node, "self.dim()"))?;
    let ndim = input_shape.len();
    let dim = if dim_raw < 0 {
        safe_usize(dim_raw + ndim as i64, "dim", &node.target)?
    } else {
        safe_usize(dim_raw, "dim", &node.target)?
    };
    let index_raw = require_int(node, "index")?;
    let dim_size = input_shape.get(dim).copied().unwrap_or(1);
    let index = if index_raw < 0 {
        safe_usize(index_raw + dim_size as i64, "index", &node.target)?
    } else {
        safe_usize(index_raw, "index", &node.target)?
    };

    let narrow_name = format!("{output_name}_narrow");
    let mut narrow_shape = input_shape.to_vec();
    if dim < narrow_shape.len() {
        narrow_shape[dim] = 1;
    }

    // Output shape: input shape with dim removed entirely.
    let output_shape: Vec<usize> = input_shape
        .iter()
        .enumerate()
        .filter(|&(i, _)| i != dim)
        .map(|(_, &s)| s)
        .collect();

    Ok(vec![
        ExpandedNode {
            name: narrow_name.clone(),
            op: TraceOp::Narrow {
                dim,
                start: index,
                length: 1,
            },
            input_names: vec![input],
            output_shape: narrow_shape,
            output_dtype: DType::F32,
        },
        ExpandedNode {
            name: output_name.to_string(),
            op: TraceOp::Reshape {
                target_shape: output_shape.clone(),
            },
            input_names: vec![narrow_name],
            output_shape,
            output_dtype: DType::F32,
        },
    ])
}

/// Expand `split.Tensor` / `split_with_sizes` into N Narrow ops.
///
/// `split(split_size, dim)` splits a tensor into chunks of `split_size` along `dim`.
/// `split_with_sizes(split_sizes, dim)` uses per-chunk sizes.
/// We decompose into individual Narrow ops, one per output tensor.
pub(super) fn expand_split(
    node: &Node,
    output_name: &str,
    input_shape: &[usize],
) -> Result<Vec<ExpandedNode>, ImportError> {
    let input = first_tensor_name(node)?;
    let dim_raw = optional_int(node, "dim").unwrap_or(0);
    let ndim = input_shape.len();
    let dim = if dim_raw < 0 {
        safe_usize(dim_raw + ndim as i64, "dim", &node.target)?
    } else {
        safe_usize(dim_raw, "dim", &node.target)?
    };

    let dim_size = input_shape.get(dim).copied().unwrap_or(0);

    // Determine split sizes. Can be:
    // - split_size: single int (uniform chunks)
    // - split_sizes: int list (per-chunk sizes, from split_with_sizes)
    let split_sizes: Vec<usize> =
        if let Ok(sizes) = super::require_ints(node, "split_size_or_sections") {
            if sizes.len() == 1 {
                // Uniform split: split_size is a single int.
                let chunk = safe_usize(sizes[0], "split_size", &node.target)?;
                let mut result = Vec::new();
                let mut remaining = dim_size;
                while remaining > 0 {
                    let s = chunk.min(remaining);
                    result.push(s);
                    remaining -= s;
                }
                result
            } else {
                // Per-chunk sizes.
                sizes
                    .into_iter()
                    .map(|v| safe_usize(v, "split_size", &node.target))
                    .collect::<Result<_, _>>()?
            }
        } else if let Ok(sizes) = super::require_ints(node, "split_sizes") {
            sizes
                .into_iter()
                .map(|v| safe_usize(v, "split_sizes", &node.target))
                .collect::<Result<_, _>>()?
        } else if let Ok(split_size) = require_int(node, "split_size") {
            let chunk = safe_usize(split_size, "split_size", &node.target)?;
            let mut result = Vec::new();
            let mut remaining = dim_size;
            while remaining > 0 {
                let s = chunk.min(remaining);
                result.push(s);
                remaining -= s;
            }
            result
        } else {
            return Err(ImportError::MissingArgument {
                op_target: node.target.clone(),
                arg_name: "split_size or split_sizes".to_string(),
            });
        };

    // Extract output tensor names from the multi-output `as_tensors`.
    let output_names: Vec<String> = node
        .outputs
        .first()
        .and_then(super::Argument::as_tensor_names)
        .map(|names| names.into_iter().map(String::from).collect())
        .unwrap_or_default();

    let mut expanded = Vec::new();
    let mut start = 0;

    for (i, &length) in split_sizes.iter().enumerate() {
        let name = output_names
            .get(i)
            .cloned()
            .unwrap_or_else(|| format!("{output_name}_split_{i}"));

        let mut out_shape = input_shape.to_vec();
        if dim < out_shape.len() {
            out_shape[dim] = length;
        }

        expanded.push(ExpandedNode {
            name,
            op: TraceOp::Narrow { dim, start, length },
            input_names: vec![input.clone()],
            output_shape: out_shape,
            output_dtype: DType::F32,
        });
        start += length;
    }

    Ok(expanded)
}

/// Expand `unbind.int` into N (Narrow + Reshape) pairs.
///
/// `unbind(dim)` splits a tensor into `shape[dim]` individual slices, each with
/// dimension `dim` removed. We decompose into N pairs of Narrow(dim, i, 1) + Reshape.
pub(super) fn expand_unbind(
    node: &Node,
    output_name: &str,
    input_shape: &[usize],
) -> Result<Vec<ExpandedNode>, ImportError> {
    let input = first_tensor_name(node)?;
    let dim_raw = optional_int(node, "dim").unwrap_or(0);
    let ndim = input_shape.len();
    let dim = if dim_raw < 0 {
        safe_usize(dim_raw + ndim as i64, "dim", &node.target)?
    } else {
        safe_usize(dim_raw, "dim", &node.target)?
    };

    let dim_size = input_shape.get(dim).copied().unwrap_or(0);

    // Output shape: input shape with dim removed.
    let final_shape: Vec<usize> = input_shape
        .iter()
        .enumerate()
        .filter(|&(i, _)| i != dim)
        .map(|(_, &s)| s)
        .collect();

    // Extract output tensor names.
    let output_names: Vec<String> = node
        .outputs
        .first()
        .and_then(super::Argument::as_tensor_names)
        .map(|names| names.into_iter().map(String::from).collect())
        .unwrap_or_default();

    let mut expanded = Vec::new();
    let mut narrow_shape = input_shape.to_vec();
    if dim < narrow_shape.len() {
        narrow_shape[dim] = 1;
    }

    for i in 0..dim_size {
        let slice_name = output_names
            .get(i)
            .cloned()
            .unwrap_or_else(|| format!("{output_name}_unbind_{i}"));

        let narrow_name = format!("{slice_name}_narrow");

        expanded.push(ExpandedNode {
            name: narrow_name.clone(),
            op: TraceOp::Narrow {
                dim,
                start: i,
                length: 1,
            },
            input_names: vec![input.clone()],
            output_shape: narrow_shape.clone(),
            output_dtype: DType::F32,
        });

        expanded.push(ExpandedNode {
            name: slice_name,
            op: TraceOp::Reshape {
                target_shape: final_shape.clone(),
            },
            input_names: vec![narrow_name],
            output_shape: final_shape.clone(),
            output_dtype: DType::F32,
        });
    }

    Ok(expanded)
}

/// Expand `aten.stack.default` into N Unsqueeze ops + 1 Cat op.
///
/// `stack(tensors, dim)` inserts a new dimension at `dim`, then concatenates.
/// We decompose to: unsqueeze each input at `dim`, then cat along `dim`.
pub(super) fn expand_stack(
    node: &Node,
    output_name: &str,
    input_shape: &[usize],
) -> Result<Vec<ExpandedNode>, ImportError> {
    let tensors_arg = get_arg(node, "tensors")?;
    let tensor_names = tensors_arg
        .as_tensor_names()
        .ok_or_else(|| ImportError::WrongArgumentType {
            op_target: node.target.clone(),
            arg_name: "tensors".to_string(),
            expected: "tensor list",
            actual: "non-tensor-list".to_string(),
        })?
        .into_iter()
        .map(String::from)
        .collect::<Vec<_>>();
    let dim = safe_usize(optional_int(node, "dim").unwrap_or(0), "dim", &node.target)?;

    let num_inputs = tensor_names.len();
    let mut expanded = Vec::with_capacity(num_inputs + 1);
    let mut unsqueezed_names = Vec::with_capacity(num_inputs);

    // Compute the shape after unsqueezing one input at dim.
    let mut unsqueezed_shape = input_shape.to_vec();
    if dim <= unsqueezed_shape.len() {
        unsqueezed_shape.insert(dim, 1);
    }

    for (i, tensor_name) in tensor_names.iter().enumerate() {
        let unsqueeze_name = format!("{output_name}_unsqueeze_{i}");
        expanded.push(ExpandedNode {
            name: unsqueeze_name.clone(),
            op: TraceOp::Unsqueeze { dim },
            input_names: vec![tensor_name.clone()],
            output_shape: unsqueezed_shape.clone(),
            output_dtype: DType::F32,
        });
        unsqueezed_names.push(unsqueeze_name);
    }

    // Compute output shape: same as unsqueezed but dim = num_inputs.
    let mut output_shape = unsqueezed_shape;
    if dim < output_shape.len() {
        output_shape[dim] = num_inputs;
    }

    expanded.push(ExpandedNode {
        name: output_name.to_string(),
        op: TraceOp::Cat { dim, num_inputs },
        input_names: unsqueezed_names,
        output_shape,
        output_dtype: DType::F32,
    });

    Ok(expanded)
}

/// Expand `masked_fill.Scalar` into Constant + WhereCond.
///
/// `masked_fill(self, mask, value)` = `where(mask, value_tensor, self)`.
/// We decompose to: Constant(value) + WhereCond(mask, constant, self).
pub(super) fn expand_masked_fill(
    node: &Node,
    output_name: &str,
    input_shape: &[usize],
) -> Result<Vec<ExpandedNode>, ImportError> {
    let self_input = first_tensor_name(node)?;
    let mask = super::require_tensor_name(node, "mask")?;
    let value = optional_float(node, "value")
        .or_else(|| optional_int(node, "value").map(|i| i as f64))
        .ok_or_else(|| ImportError::MissingArgument {
            op_target: node.target.clone(),
            arg_name: "value".to_string(),
        })?;

    let const_name = format!("{output_name}_fill_val");
    Ok(vec![
        ExpandedNode {
            name: const_name.clone(),
            op: TraceOp::Constant { value },
            input_names: vec![],
            output_shape: vec![],
            output_dtype: DType::F32,
        },
        ExpandedNode {
            name: output_name.to_string(),
            op: TraceOp::WhereCond,
            // WhereCond arity: (condition, true_branch, false_branch)
            // masked_fill: where mask is true, use fill value; otherwise keep self.
            input_names: vec![mask, const_name, self_input],
            output_shape: input_shape.to_vec(),
            output_dtype: DType::F32,
        },
    ])
}

/// Expand `index.Tensor` into IndexSelect for single-index case.
///
/// `x[indices]` where indices is a list of optional tensors. For the common
/// single-index case (one tensor index along dim 0), decompose to IndexSelect.
pub(super) fn expand_index_tensor(
    node: &Node,
    output_name: &str,
    input_shape: &[usize],
) -> Result<Vec<ExpandedNode>, ImportError> {
    let self_input = first_tensor_name(node)?;

    // Extract the indices list. In torch.export, indices is a list of
    // optional tensors. We handle the single-index case (one non-None tensor).
    let indices_arg = get_arg(node, "indices")?;

    // Try to get the list of tensor names from the indices argument.
    let index_names: Vec<String> = indices_arg
        .as_tensor_names()
        .map(|names| names.into_iter().map(String::from).collect())
        .unwrap_or_default();

    // For single tensor index along dim 0, decompose to IndexSelect.
    if index_names.len() == 1 {
        Ok(vec![ExpandedNode {
            name: output_name.to_string(),
            op: TraceOp::IndexSelect { dim: 0 },
            input_names: vec![self_input, index_names[0].clone()],
            output_shape: input_shape.to_vec(),
            output_dtype: DType::F32,
        }])
    } else {
        // Multi-index advanced indexing not yet supported.
        Err(ImportError::UnsupportedOp {
            target: format!(
                "{} (multi-index advanced indexing with {} indices not yet supported)",
                node.target,
                index_names.len()
            ),
        })
    }
}

/// Expand `meshgrid` into Reshape + Expand ops.
///
/// `meshgrid(*tensors, indexing='ij')` creates coordinate grids from 1-D input
/// tensors. For N inputs of sizes [s0, s1, ..., s_{N-1}], it produces N outputs,
/// each of shape [s0, s1, ..., s_{N-1}].
///
/// For input i: reshape to shape with 1s everywhere except dim i (which is s_i),
/// then expand to the full grid shape.
pub(super) fn expand_meshgrid(
    node: &Node,
    output_name: &str,
    input_shape: &[usize],
) -> Result<Vec<ExpandedNode>, ImportError> {
    // meshgrid takes a list of tensors as its first argument.
    let tensors_arg = get_arg(node, "tensors")?;
    let tensor_names = tensors_arg
        .as_tensor_names()
        .ok_or_else(|| ImportError::WrongArgumentType {
            op_target: node.target.clone(),
            arg_name: "tensors".to_string(),
            expected: "tensor list",
            actual: "non-tensor-list".to_string(),
        })?
        .into_iter()
        .map(String::from)
        .collect::<Vec<_>>();

    let num_inputs = tensor_names.len();
    if num_inputs == 0 {
        return Ok(vec![]);
    }

    // We use input_shape as the size of the first tensor.
    // For meshgrid, all inputs are 1-D. The output grid shape is
    // [len(t0), len(t1), ..., len(t_{N-1})].
    // Since we only have input_shape for the first tensor, we use
    // input_shape[0] for all dimensions (common square-grid case).
    // The full output shape computation would need all input shapes,
    // which we don't have in try_expand_node's signature.
    //
    // For the common case where meshgrid is used for 2D coordinate grids
    // with inputs of the same size, this produces the correct decomposition.
    let grid_dim = input_shape.first().copied().unwrap_or(1);
    let grid_shape: Vec<usize> = vec![grid_dim; num_inputs];

    let mut expanded = Vec::new();

    // Extract output tensor names.
    let output_names: Vec<String> = node
        .outputs
        .first()
        .and_then(super::Argument::as_tensor_names)
        .map(|names| names.into_iter().map(String::from).collect())
        .unwrap_or_default();

    for (i, tensor_name) in tensor_names.iter().enumerate() {
        let final_name = output_names
            .get(i)
            .cloned()
            .unwrap_or_else(|| format!("{output_name}_grid_{i}"));

        // Step 1: Reshape to [1, ..., 1, s_i, 1, ..., 1]
        let mut reshape_shape = vec![1usize; num_inputs];
        reshape_shape[i] = grid_dim;
        let reshape_name = format!("{final_name}_reshape");

        expanded.push(ExpandedNode {
            name: reshape_name.clone(),
            op: TraceOp::Reshape {
                target_shape: reshape_shape.clone(),
            },
            input_names: vec![tensor_name.clone()],
            output_shape: reshape_shape,
            output_dtype: DType::F32,
        });

        // Step 2: Expand to the full grid shape.
        expanded.push(ExpandedNode {
            name: final_name,
            op: TraceOp::Expand {
                target_shape: grid_shape.clone(),
            },
            input_names: vec![reshape_name],
            output_shape: grid_shape.clone(),
            output_dtype: DType::F32,
        });
    }

    Ok(expanded)
}

/// Expand `addmm(bias, mat1, mat2)` into MatMul + Add.
///
/// `addmm(self, mat1, mat2, beta=1, alpha=1)` = `self + mat1 @ mat2`.
/// Decomposes to: MatMul(mat1, mat2) → mm_result, then Add(mm_result, self).
pub(super) fn expand_addmm(
    node: &Node,
    output_name: &str,
    input_shape: &[usize],
) -> Result<Vec<ExpandedNode>, ImportError> {
    let bias = super::require_tensor_name(node, "self")?;
    let mat1 = super::require_tensor_name(node, "mat1")?;
    let mat2 = super::require_tensor_name(node, "mat2")?;

    let mm_name = format!("{output_name}_mm");
    Ok(vec![
        ExpandedNode {
            name: mm_name.clone(),
            op: TraceOp::MatMul,
            input_names: vec![mat1, mat2],
            output_shape: input_shape.to_vec(),
            output_dtype: DType::F32,
        },
        ExpandedNode {
            name: output_name.to_string(),
            op: TraceOp::Add,
            input_names: vec![mm_name, bias],
            output_shape: input_shape.to_vec(),
            output_dtype: DType::F32,
        },
    ])
}

/// Expand `baddbmm(self, batch1, batch2, beta, alpha)` into MatMul + Mul + Add.
///
/// General case: `beta * self + alpha * (batch1 @ batch2)`.
/// For beta=0, alpha=1 (common in attention), this is just batch1 @ batch2.
/// We check beta and alpha to optimize.
pub(super) fn expand_baddbmm(
    node: &Node,
    output_name: &str,
    input_shape: &[usize],
) -> Result<Vec<ExpandedNode>, ImportError> {
    let bias = super::require_tensor_name(node, "self")?;
    let batch1 = super::require_tensor_name(node, "batch1")?;
    let batch2 = super::require_tensor_name(node, "batch2")?;
    let beta = optional_float(node, "beta").unwrap_or(0.0);
    let alpha = optional_float(node, "alpha").unwrap_or(1.0);

    let mm_name = format!("{output_name}_bmm");
    let mut nodes = vec![ExpandedNode {
        name: mm_name.clone(),
        op: TraceOp::MatMul,
        input_names: vec![batch1, batch2],
        output_shape: input_shape.to_vec(),
        output_dtype: DType::F32,
    }];

    if beta == 0.0 && alpha == 1.0 {
        // Simple case: just batch1 @ batch2. Rename the mm node.
        nodes[0].name = output_name.to_string();
    } else if beta == 0.0 {
        // alpha * (batch1 @ batch2)
        let alpha_const = format!("{output_name}_alpha");
        nodes.push(ExpandedNode {
            name: alpha_const.clone(),
            op: TraceOp::Constant { value: alpha },
            input_names: vec![],
            output_shape: vec![],
            output_dtype: DType::F32,
        });
        nodes.push(ExpandedNode {
            name: output_name.to_string(),
            op: TraceOp::Mul,
            input_names: vec![mm_name, alpha_const],
            output_shape: input_shape.to_vec(),
            output_dtype: DType::F32,
        });
    } else {
        // General case: beta * self + alpha * (batch1 @ batch2)
        let alpha_const = format!("{output_name}_alpha");
        let beta_const = format!("{output_name}_beta");
        let scaled_mm = format!("{output_name}_scaled_mm");
        let scaled_bias = format!("{output_name}_scaled_bias");

        nodes.push(ExpandedNode {
            name: alpha_const.clone(),
            op: TraceOp::Constant { value: alpha },
            input_names: vec![],
            output_shape: vec![],
            output_dtype: DType::F32,
        });
        nodes.push(ExpandedNode {
            name: scaled_mm.clone(),
            op: TraceOp::Mul,
            input_names: vec![mm_name, alpha_const],
            output_shape: input_shape.to_vec(),
            output_dtype: DType::F32,
        });
        nodes.push(ExpandedNode {
            name: beta_const.clone(),
            op: TraceOp::Constant { value: beta },
            input_names: vec![],
            output_shape: vec![],
            output_dtype: DType::F32,
        });
        nodes.push(ExpandedNode {
            name: scaled_bias.clone(),
            op: TraceOp::Mul,
            input_names: vec![bias, beta_const],
            output_shape: input_shape.to_vec(),
            output_dtype: DType::F32,
        });
        nodes.push(ExpandedNode {
            name: output_name.to_string(),
            op: TraceOp::Add,
            input_names: vec![scaled_mm, scaled_bias],
            output_shape: input_shape.to_vec(),
            output_dtype: DType::F32,
        });
    }

    Ok(nodes)
}

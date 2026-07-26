// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0
//! Individual aten op → TraceOp mapper functions (extracted from `op_map.rs`).

use nn_core::dyn_tensor::trace::TraceOp;

use super::{
    first_tensor_name, get_arg, optional_bool, optional_float, optional_int, optional_weight,
    reduce_params, require_int, require_ints, require_single_dim, require_tensor_name, resolve_dim,
    resolve_weight, safe_usize, safe_usize_allow_neg1, safe_usize_vec, ImportError, Node,
    OpMapContext,
};

pub(super) fn unary_op(node: &Node, op: TraceOp) -> Result<(TraceOp, Vec<String>), ImportError> {
    let input = first_tensor_name(node)?;
    Ok((op, vec![input]))
}

pub(super) fn binary_op(
    node: &Node,
    op: TraceOp,
    rhs_name: &str,
) -> Result<(TraceOp, Vec<String>), ImportError> {
    let lhs = require_tensor_name(node, "self")?;
    let rhs = require_tensor_name(node, rhs_name)?;
    Ok((op, vec![lhs, rhs]))
}

pub(super) fn map_gelu(node: &Node) -> Result<(TraceOp, Vec<String>), ImportError> {
    let input = first_tensor_name(node)?;
    let approximate = get_arg(node, "approximate")
        .ok()
        .and_then(|a| a.as_string())
        .unwrap_or("none");
    let op = if approximate == "tanh" {
        TraceOp::Gelu
    } else {
        TraceOp::GeluErf
    };
    Ok((op, vec![input]))
}

pub(super) fn map_linear(
    node: &Node,
    ctx: &OpMapContext<'_>,
) -> Result<(TraceOp, Vec<String>), ImportError> {
    let input = require_tensor_name(node, "input")?;
    let weight_name = require_tensor_name(node, "weight")?;
    let bias_name = get_arg(node, "bias")
        .ok()
        .and_then(|a| a.as_tensor_name())
        .map(String::from);
    let weight = resolve_weight(&weight_name, ctx)?;
    let bias = optional_weight(bias_name.as_deref(), ctx);
    Ok((TraceOp::Linear { weight, bias }, vec![input]))
}

pub(super) fn map_convolution(
    node: &Node,
    ctx: &OpMapContext<'_>,
) -> Result<(TraceOp, Vec<String>), ImportError> {
    let input = require_tensor_name(node, "input")?;
    let weight_name = require_tensor_name(node, "weight")?;
    let bias_name = get_arg(node, "bias")
        .ok()
        .and_then(|a| a.as_tensor_name())
        .map(String::from);
    let weight = resolve_weight(&weight_name, ctx)?;
    let bias = optional_weight(bias_name.as_deref(), ctx);
    let stride = require_ints(node, "stride")?;
    let padding = require_ints(node, "padding")?;
    let dilation = require_ints(node, "dilation")?;
    let transposed = optional_bool(node, "transposed", false);
    let output_padding = get_arg(node, "output_padding")
        .ok()
        .and_then(|a| a.as_ints())
        .map(<[i64]>::to_vec)
        .unwrap_or_default();
    let groups = optional_int(node, "groups").unwrap_or(1);
    let weight_ndim = weight.shape().len();
    let t = &node.target;
    let groups_u = safe_usize(groups, "groups", t)?;

    let op = if transposed {
        if weight_ndim == 3 {
            TraceOp::ConvTranspose1d {
                weight,
                bias,
                padding: safe_usize(padding.first().copied().unwrap_or(0), "padding", t)?,
                output_padding: safe_usize(
                    output_padding.first().copied().unwrap_or(0),
                    "output_padding",
                    t,
                )?,
                stride: safe_usize(stride.first().copied().unwrap_or(1), "stride", t)?,
                dilation: safe_usize(dilation.first().copied().unwrap_or(1), "dilation", t)?,
                groups: groups_u,
            }
        } else {
            TraceOp::ConvTranspose2d {
                weight,
                bias,
                padding: [
                    safe_usize(padding.first().copied().unwrap_or(0), "padding", t)?,
                    safe_usize(padding.get(1).copied().unwrap_or(0), "padding", t)?,
                ],
                output_padding: [
                    safe_usize(
                        output_padding.first().copied().unwrap_or(0),
                        "output_padding",
                        t,
                    )?,
                    safe_usize(
                        output_padding.get(1).copied().unwrap_or(0),
                        "output_padding",
                        t,
                    )?,
                ],
                stride: [
                    safe_usize(stride.first().copied().unwrap_or(1), "stride", t)?,
                    safe_usize(stride.get(1).copied().unwrap_or(1), "stride", t)?,
                ],
                dilation: [
                    safe_usize(dilation.first().copied().unwrap_or(1), "dilation", t)?,
                    safe_usize(dilation.get(1).copied().unwrap_or(1), "dilation", t)?,
                ],
                groups: groups_u,
            }
        }
    } else if weight_ndim == 3 {
        TraceOp::Conv1d {
            weight,
            bias,
            padding: safe_usize(padding.first().copied().unwrap_or(0), "padding", t)?,
            stride: safe_usize(stride.first().copied().unwrap_or(1), "stride", t)?,
            dilation: safe_usize(dilation.first().copied().unwrap_or(1), "dilation", t)?,
            groups: groups_u,
        }
    } else if weight_ndim == 5 {
        // 3D convolution: weight shape [out_ch, in_ch/groups, kD, kH, kW]
        let pad = [
            safe_usize(padding.first().copied().unwrap_or(0), "padding", t)?,
            safe_usize(padding.get(1).copied().unwrap_or(0), "padding", t)?,
            safe_usize(padding.get(2).copied().unwrap_or(0), "padding", t)?,
        ];
        let str_ = [
            safe_usize(stride.first().copied().unwrap_or(1), "stride", t)?,
            safe_usize(stride.get(1).copied().unwrap_or(1), "stride", t)?,
            safe_usize(stride.get(2).copied().unwrap_or(1), "stride", t)?,
        ];
        let dil = [
            safe_usize(dilation.first().copied().unwrap_or(1), "dilation", t)?,
            safe_usize(dilation.get(1).copied().unwrap_or(1), "dilation", t)?,
            safe_usize(dilation.get(2).copied().unwrap_or(1), "dilation", t)?,
        ];
        TraceOp::Conv3d {
            weight,
            bias,
            padding: pad,
            stride: str_,
            dilation: dil,
            groups: groups_u,
        }
    } else {
        let pad = [
            safe_usize(padding.first().copied().unwrap_or(0), "padding", t)?,
            safe_usize(padding.get(1).copied().unwrap_or(0), "padding", t)?,
        ];
        let str_ = [
            safe_usize(stride.first().copied().unwrap_or(1), "stride", t)?,
            safe_usize(stride.get(1).copied().unwrap_or(1), "stride", t)?,
        ];
        let dil = [
            safe_usize(dilation.first().copied().unwrap_or(1), "dilation", t)?,
            safe_usize(dilation.get(1).copied().unwrap_or(1), "dilation", t)?,
        ];
        TraceOp::Conv2d {
            weight,
            bias,
            padding: pad,
            stride: str_,
            dilation: dil,
            groups: groups_u,
        }
    };
    Ok((op, vec![input]))
}

pub(super) fn map_layer_norm(
    node: &Node,
    ctx: &OpMapContext<'_>,
) -> Result<(TraceOp, Vec<String>), ImportError> {
    let input = require_tensor_name(node, "input")?;
    let weight_name = require_tensor_name(node, "weight")?;
    let bias_name = require_tensor_name(node, "bias")?;
    let weight = resolve_weight(&weight_name, ctx)?;
    let bias = resolve_weight(&bias_name, ctx)?;
    let eps = optional_float(node, "eps").unwrap_or(1e-5);
    Ok((TraceOp::LayerNorm { eps, weight, bias }, vec![input]))
}

pub(super) fn map_group_norm(
    node: &Node,
    ctx: &OpMapContext<'_>,
) -> Result<(TraceOp, Vec<String>), ImportError> {
    let input = require_tensor_name(node, "input")?;
    let weight_name = require_tensor_name(node, "weight")?;
    let bias_name = require_tensor_name(node, "bias")?;
    let num_groups = safe_usize(require_int(node, "num_groups")?, "num_groups", &node.target)?;
    let weight = resolve_weight(&weight_name, ctx)?;
    let bias = resolve_weight(&bias_name, ctx)?;
    let eps = optional_float(node, "eps").unwrap_or(1e-5);
    Ok((
        TraceOp::GroupNorm {
            num_groups,
            eps,
            weight,
            bias,
        },
        vec![input],
    ))
}

pub(super) fn map_batch_norm(
    node: &Node,
    ctx: &OpMapContext<'_>,
) -> Result<(TraceOp, Vec<String>), ImportError> {
    let input = require_tensor_name(node, "input")?;
    let weight = resolve_weight(&require_tensor_name(node, "weight")?, ctx)?;
    let bias = resolve_weight(&require_tensor_name(node, "bias")?, ctx)?;
    let running_mean = resolve_weight(&require_tensor_name(node, "running_mean")?, ctx)?;
    let running_var = resolve_weight(&require_tensor_name(node, "running_var")?, ctx)?;
    let eps = optional_float(node, "eps").unwrap_or(1e-5);
    Ok((
        TraceOp::BatchNorm {
            eps,
            weight,
            bias,
            running_mean,
            running_var,
        },
        vec![input],
    ))
}

pub(super) fn map_instance_norm(node: &Node) -> Result<(TraceOp, Vec<String>), ImportError> {
    let input = first_tensor_name(node)?;
    let eps = optional_float(node, "eps").unwrap_or(1e-5);
    Ok((TraceOp::InstanceNorm { eps }, vec![input]))
}

pub(super) fn map_softmax(
    node: &Node,
    input_ndim: usize,
) -> Result<(TraceOp, Vec<String>), ImportError> {
    let input = first_tensor_name(node)?;
    let dim = resolve_dim(require_int(node, "dim")?, input_ndim, "dim", &node.target)?;
    Ok((TraceOp::Softmax { dim }, vec![input]))
}

pub(super) fn map_log_softmax(
    node: &Node,
    input_ndim: usize,
) -> Result<(TraceOp, Vec<String>), ImportError> {
    let input = first_tensor_name(node)?;
    let dim = resolve_dim(require_int(node, "dim")?, input_ndim, "dim", &node.target)?;
    Ok((TraceOp::LogSoftmax { dim }, vec![input]))
}

pub(super) fn map_sdpa(
    node: &Node,
    ctx: &OpMapContext<'_>,
) -> Result<(TraceOp, Vec<String>), ImportError> {
    let q = require_tensor_name(node, "query")?;
    let k = require_tensor_name(node, "key")?;
    let v = require_tensor_name(node, "value")?;
    let scale = sdpa_scale(node, ctx, &q)?;
    let is_causal = optional_bool(node, "is_causal", false);
    // Check for attn_mask tensor (None means no mask).
    let attn_mask = get_arg(node, "attn_mask").ok().and_then(|a| {
        if a.is_none() {
            None
        } else {
            a.as_tensor_name().map(String::from)
        }
    });
    if is_causal {
        Ok((TraceOp::SdpaCausal { scale }, vec![q, k, v]))
    } else if let Some(mask) = attn_mask {
        Ok((TraceOp::Sdpa { scale }, vec![q, k, v, mask]))
    } else {
        Ok((TraceOp::Sdpa { scale }, vec![q, k, v]))
    }
}

/// Resolve SDPA scale: explicit `scale` arg, or 1/sqrt(head_dim) from query metadata.
fn sdpa_scale(node: &Node, ctx: &OpMapContext<'_>, q_name: &str) -> Result<f64, ImportError> {
    match optional_float(node, "scale") {
        Some(s) => Ok(s),
        None => {
            let head_dim = ctx
                .tensor_meta
                .get(q_name)
                .and_then(|meta| meta.sizes.last())
                .and_then(super::super::parse::SymInt::as_concrete)
                .ok_or_else(|| ImportError::MissingArgument {
                    op_target: node.target.clone(),
                    arg_name: "scale".to_string(),
                })?;
            Ok(1.0 / (head_dim as f64).sqrt())
        }
    }
}

pub(super) fn map_embedding(
    node: &Node,
    ctx: &OpMapContext<'_>,
) -> Result<(TraceOp, Vec<String>), ImportError> {
    let weight_name = require_tensor_name(node, "weight")?;
    let indices = require_tensor_name(node, "indices")?;
    let weight = resolve_weight(&weight_name, ctx)?;
    Ok((TraceOp::Embedding { weight }, vec![indices]))
}

pub(super) fn map_reduce_sum(node: &Node) -> Result<(TraceOp, Vec<String>), ImportError> {
    let (input, dim, keepdim) = reduce_params(node)?;
    Ok((TraceOp::ReduceSum { dim, keepdim }, vec![input]))
}

pub(super) fn map_reduce_mean(node: &Node) -> Result<(TraceOp, Vec<String>), ImportError> {
    let (input, dim, keepdim) = reduce_params(node)?;
    Ok((TraceOp::ReduceMean { dim, keepdim }, vec![input]))
}

pub(super) fn map_reduce_max(node: &Node) -> Result<(TraceOp, Vec<String>), ImportError> {
    let (input, dim, keepdim) = reduce_params(node)?;
    Ok((TraceOp::ReduceMax { dim, keepdim }, vec![input]))
}

pub(super) fn map_reduce_min(node: &Node) -> Result<(TraceOp, Vec<String>), ImportError> {
    let (input, dim, keepdim) = reduce_params(node)?;
    Ok((TraceOp::ReduceMin { dim, keepdim }, vec![input]))
}

pub(super) fn map_reshape(node: &Node) -> Result<(TraceOp, Vec<String>), ImportError> {
    let input = first_tensor_name(node)?;
    let raw = require_ints(node, "size").or_else(|_| require_ints(node, "shape"))?;
    let shape: Vec<usize> = raw
        .into_iter()
        .map(|v| safe_usize_allow_neg1(v, "size", &node.target))
        .collect::<Result<_, _>>()?;
    Ok((
        TraceOp::Reshape {
            target_shape: shape,
        },
        vec![input],
    ))
}

pub(super) fn map_transpose(node: &Node) -> Result<(TraceOp, Vec<String>), ImportError> {
    let input = first_tensor_name(node)?;
    let dim0 = safe_usize(require_int(node, "dim0")?, "dim0", &node.target)?;
    let dim1 = safe_usize(require_int(node, "dim1")?, "dim1", &node.target)?;
    Ok((TraceOp::Transpose { dim0, dim1 }, vec![input]))
}

pub(super) fn map_permute(node: &Node) -> Result<(TraceOp, Vec<String>), ImportError> {
    let input = first_tensor_name(node)?;
    let dims = require_ints(node, "dims")?;
    let axes = safe_usize_vec(dims, "dims", &node.target)?;
    Ok((TraceOp::Permute { axes }, vec![input]))
}

pub(super) fn map_unsqueeze(node: &Node) -> Result<(TraceOp, Vec<String>), ImportError> {
    let input = first_tensor_name(node)?;
    let dim = safe_usize(require_int(node, "dim")?, "dim", &node.target)?;
    Ok((TraceOp::Unsqueeze { dim }, vec![input]))
}

pub(super) fn map_squeeze(node: &Node) -> Result<(TraceOp, Vec<String>), ImportError> {
    let input = first_tensor_name(node)?;
    let dim = safe_usize(require_int(node, "dim")?, "dim", &node.target)?;
    Ok((TraceOp::Squeeze { dim }, vec![input]))
}

/// squeeze.default (no dim arg) — fallback when input shape metadata is missing.
///
/// Normally handled by `try_expand_node` → `expand_squeeze_default` (Reshape-based).
/// This path only triggers when the torch.export graph lacks shape metadata,
/// making it impossible to compute the output shape statically.
pub(super) fn map_squeeze_default(node: &Node) -> Result<(TraceOp, Vec<String>), ImportError> {
    Err(ImportError::UnsupportedOp {
        target: format!(
            "{} (squeeze.default needs input shape metadata for Reshape decomposition)",
            node.target
        ),
    })
}

pub(super) fn map_cat(node: &Node) -> Result<(TraceOp, Vec<String>), ImportError> {
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
    Ok((TraceOp::Cat { dim, num_inputs }, tensor_names))
}

pub(super) fn map_slice(node: &Node) -> Result<(TraceOp, Vec<String>), ImportError> {
    let input = first_tensor_name(node)?;
    let dim = safe_usize(optional_int(node, "dim").unwrap_or(0), "dim", &node.target)?;
    let start = safe_usize(
        optional_int(node, "start").unwrap_or(0),
        "start",
        &node.target,
    )?;
    let end = optional_int(node, "end");
    let length = match end {
        // torch.export encodes open-ended slices (`x[:, 11:, :]`) as end=i64::MAX.
        Some(i64::MAX) => usize::MAX,
        Some(e) => safe_usize(e, "end", &node.target)?.saturating_sub(start),
        None => usize::MAX,
    };
    Ok((TraceOp::Narrow { dim, start, length }, vec![input]))
}

pub(super) fn map_expand(node: &Node) -> Result<(TraceOp, Vec<String>), ImportError> {
    let input = first_tensor_name(node)?;
    let size = require_ints(node, "size")?;
    let target_shape: Vec<usize> = size
        .into_iter()
        .map(|v| safe_usize_allow_neg1(v, "size", &node.target))
        .collect::<Result<_, _>>()?;
    Ok((TraceOp::Expand { target_shape }, vec![input]))
}

pub(super) fn map_flip(node: &Node) -> Result<(TraceOp, Vec<String>), ImportError> {
    let input = first_tensor_name(node)?;
    let dims = require_ints(node, "dims")?;
    let raw_dim = require_single_dim(&dims, "dims", "flip", &node.target)?;
    let dim = safe_usize(raw_dim, "dims", &node.target)?;
    Ok((TraceOp::Flip { dim }, vec![input]))
}

// Pool, activation, comparison, dtype, power, misc, and LSTM ops are in op_map_impl_ext.rs.
// Re-exported below via the ext submodule in op_map.rs.

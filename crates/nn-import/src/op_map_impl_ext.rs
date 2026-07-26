// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Extended aten op mappers: pooling, activation, comparison, type conversion,
//! power, misc, and recurrent (LSTM) ops.
//!
//! Extracted from `op_map_impl.rs` to stay under the 500-line limit.

#[path = "op_map_impl_ext_bilstm.rs"]
mod bilstm;
pub(super) use bilstm::expand_bilstm;

use nn_core::dyn_tensor::trace::TraceOp;

use super::{
    first_tensor_name, get_arg, optional_bool, optional_float, optional_int, parse_pool1d_params,
    parse_pool2d_params, require_int, require_tensor_name, resolve_weight, safe_usize,
    scalar_type_to_name, Argument, ImportError, Node, OpMapContext,
};

// -- Pooling --

pub(super) fn map_max_pool1d(node: &Node) -> Result<(TraceOp, Vec<String>), ImportError> {
    let input = first_tensor_name(node)?;
    let (kernel_size, stride, padding) = parse_pool1d_params(node)?;
    Ok((
        TraceOp::MaxPool1d {
            kernel_size,
            stride,
            padding,
        },
        vec![input],
    ))
}

pub(super) fn map_avg_pool2d(node: &Node) -> Result<(TraceOp, Vec<String>), ImportError> {
    let input = first_tensor_name(node)?;
    let (kernel_size, stride, padding) = parse_pool2d_params(node)?;
    Ok((
        TraceOp::AvgPool2d {
            kernel_size,
            stride,
            padding,
        },
        vec![input],
    ))
}

pub(super) fn map_max_pool2d(node: &Node) -> Result<(TraceOp, Vec<String>), ImportError> {
    let input = first_tensor_name(node)?;
    let (kernel_size, stride, padding) = parse_pool2d_params(node)?;
    Ok((
        TraceOp::MaxPool2d {
            kernel_size,
            stride,
            padding,
        },
        vec![input],
    ))
}

pub(super) fn map_adaptive_avg_pool2d(node: &Node) -> Result<(TraceOp, Vec<String>), ImportError> {
    let input = first_tensor_name(node)?;
    let t = &node.target;
    let output_size = super::require_ints(node, "output_size")?;
    let os = [
        safe_usize(output_size[0], "output_size", t)?,
        safe_usize(output_size[1], "output_size", t)?,
    ];
    Ok((TraceOp::AdaptiveAvgPool2d { output_size: os }, vec![input]))
}

// -- Activation --

pub(super) fn map_elu(node: &Node) -> Result<(TraceOp, Vec<String>), ImportError> {
    let input = first_tensor_name(node)?;
    let alpha = optional_float(node, "alpha").unwrap_or(1.0);
    Ok((TraceOp::Elu { alpha }, vec![input]))
}

pub(super) fn map_leaky_relu(node: &Node) -> Result<(TraceOp, Vec<String>), ImportError> {
    let input = first_tensor_name(node)?;
    let slope = optional_float(node, "negative_slope").unwrap_or(0.01);
    Ok((TraceOp::LeakyRelu { slope }, vec![input]))
}

pub(super) fn map_hardtanh(node: &Node) -> Result<(TraceOp, Vec<String>), ImportError> {
    let input = first_tensor_name(node)?;
    let min_val = optional_float(node, "min_val").unwrap_or(-1.0);
    let max_val = optional_float(node, "max_val").unwrap_or(1.0);
    Ok((
        TraceOp::Clamp {
            min: Some(min_val),
            max: Some(max_val),
        },
        vec![input],
    ))
}

pub(super) fn map_softplus(node: &Node) -> Result<(TraceOp, Vec<String>), ImportError> {
    let input = first_tensor_name(node)?;
    Ok((TraceOp::Softplus, vec![input]))
}

pub(super) fn map_celu(node: &Node) -> Result<(TraceOp, Vec<String>), ImportError> {
    let input = first_tensor_name(node)?;
    let alpha = optional_float(node, "alpha").unwrap_or(1.0);
    Ok((TraceOp::Celu { alpha }, vec![input]))
}

// -- Comparison / Selection --

pub(super) fn map_where_cond(node: &Node) -> Result<(TraceOp, Vec<String>), ImportError> {
    let cond = require_tensor_name(node, "condition")?;
    let self_ = require_tensor_name(node, "self")?;
    let other = require_tensor_name(node, "other")?;
    Ok((TraceOp::WhereCond, vec![cond, self_, other]))
}

pub(super) fn map_clamp(node: &Node) -> Result<(TraceOp, Vec<String>), ImportError> {
    let input = first_tensor_name(node)?;
    let min = optional_float(node, "min");
    let max = optional_float(node, "max");
    Ok((TraceOp::Clamp { min, max }, vec![input]))
}

// -- Type conversion --

pub(super) fn map_to_dtype(node: &Node) -> Result<(TraceOp, Vec<String>), ImportError> {
    let input = first_tensor_name(node)?;
    let dtype_arg = get_arg(node, "dtype")?;
    let target_dtype = match dtype_arg {
        Argument::ScalarType(st) => scalar_type_to_name(st.as_scalar_type, &node.target)?,
        Argument::Int(i) => scalar_type_to_name(i.as_int as i32, &node.target)?,
        other => {
            return Err(ImportError::WrongArgumentType {
                op_target: node.target.clone(),
                arg_name: "dtype".to_string(),
                expected: "ScalarType or Int",
                actual: format!("{other:?}"),
            });
        }
    };
    Ok((TraceOp::ToDtype { target_dtype }, vec![input]))
}

// -- Power --

pub(super) fn map_powf(node: &Node) -> Result<(TraceOp, Vec<String>), ImportError> {
    let input = first_tensor_name(node)?;
    let exponent = optional_float(node, "exponent")
        .or_else(|| node.inputs.get(1).and_then(|na| na.arg.as_float()))
        .unwrap_or(1.0);
    // Rewrite common integer/half-integer exponents to avoid exp(e*log(x))
    // decomposition which produces NaN for negative inputs. (#2751)
    let op = if exponent == 2.0 {
        TraceOp::Sqr
    } else if exponent == 0.5 {
        TraceOp::Sqrt
    } else {
        TraceOp::Powf { exponent }
    };
    Ok((op, vec![input]))
}

// -- Misc --

pub(super) fn map_cumsum(node: &Node) -> Result<(TraceOp, Vec<String>), ImportError> {
    let input = first_tensor_name(node)?;
    let dim = safe_usize(require_int(node, "dim")?, "dim", &node.target)?;
    Ok((TraceOp::Cumsum { dim }, vec![input]))
}

/// Map `aten.repeat_interleave.self_Tensor` to `TraceOp::RepeatInterleave`.
///
/// torch.export signature: `(self, repeats, dim?, output_size?)`
/// - `repeats` is a 1-D tensor of per-element repeat counts
/// - `dim` is the dimension along which to repeat (required for TraceOp)
/// - `output_size` is optional and ignored (pre-allocation hint only)
pub(super) fn map_repeat_interleave(node: &Node) -> Result<(TraceOp, Vec<String>), ImportError> {
    let input = first_tensor_name(node)?;
    let repeats = require_tensor_name(node, "repeats")?;
    let dim = safe_usize(require_int(node, "dim")?, "dim", &node.target)?;
    Ok((TraceOp::RepeatInterleave { dim }, vec![input, repeats]))
}

// -- Zero tensor creation --

/// Map `aten.zeros.default` to `Constant { value: 0.0 }`.
///
/// The output shape comes from `tensor_values` in the graph builder, not from the op args.
/// We return no tensor inputs since this creates a fresh zero tensor.
pub(super) fn map_zeros(node: &Node) -> Result<(TraceOp, Vec<String>), ImportError> {
    let _ = node; // args (size, dtype, device, pin_memory) are metadata; shape from tensor_values
    Ok((TraceOp::Constant { value: 0.0 }, vec![]))
}

/// Map `aten.zeros_like.default` to `Constant { value: 0.0 }`.
///
/// Creates a zero tensor with the same shape/dtype as the input. The output shape
/// comes from `tensor_values` in the graph builder. The input tensor reference is
/// not needed as a graph dependency since we only use its metadata.
pub(super) fn map_zeros_like(node: &Node) -> Result<(TraceOp, Vec<String>), ImportError> {
    let _ = node;
    Ok((TraceOp::Constant { value: 0.0 }, vec![]))
}

// -- Standalone conv1d --

/// Map `aten.conv1d.default` to `TraceOp::Conv1d`.
///
/// Unlike the unified `aten.convolution.default`, `conv1d.default` has a simpler
/// signature without `transposed`, `output_padding`, or `dilation` args. The weight
/// must be a static parameter (not a runtime-computed tensor).
pub(super) fn map_conv1d(
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
    let bias = super::optional_weight(bias_name.as_deref(), ctx);
    let t = &node.target;
    let stride = get_arg(node, "stride")
        .ok()
        .and_then(|a| a.as_ints())
        .map(<[i64]>::to_vec)
        .unwrap_or_else(|| vec![1]);
    let padding = get_arg(node, "padding")
        .ok()
        .and_then(|a| a.as_ints())
        .map(<[i64]>::to_vec)
        .unwrap_or_else(|| vec![0]);
    let dilation = get_arg(node, "dilation")
        .ok()
        .and_then(|a| a.as_ints())
        .map(<[i64]>::to_vec)
        .unwrap_or_else(|| vec![1]);
    let groups = optional_int(node, "groups").unwrap_or(1);
    Ok((
        TraceOp::Conv1d {
            weight,
            bias,
            padding: safe_usize(padding[0], "padding", t)?,
            stride: safe_usize(stride[0], "stride", t)?,
            dilation: safe_usize(dilation[0], "dilation", t)?,
            groups: safe_usize(groups, "groups", t)?,
        },
        vec![input],
    ))
}

// -- Standalone conv2d --

/// Map `aten.conv2d.default` to `TraceOp::Conv2d`.
///
/// Unlike the unified `aten.convolution.default`, `conv2d.default` has a simpler
/// signature without `transposed`, `output_padding` args. Some PyTorch export
/// paths emit this instead of the unified convolution op.
pub(super) fn map_conv2d(
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
    let bias = super::optional_weight(bias_name.as_deref(), ctx);
    let t = &node.target;
    let stride = get_arg(node, "stride")
        .ok()
        .and_then(|a| a.as_ints())
        .map(<[i64]>::to_vec)
        .unwrap_or_else(|| vec![1, 1]);
    let padding = get_arg(node, "padding")
        .ok()
        .and_then(|a| a.as_ints())
        .map(<[i64]>::to_vec)
        .unwrap_or_else(|| vec![0, 0]);
    let dilation = get_arg(node, "dilation")
        .ok()
        .and_then(|a| a.as_ints())
        .map(<[i64]>::to_vec)
        .unwrap_or_else(|| vec![1, 1]);
    let groups = optional_int(node, "groups").unwrap_or(1);
    Ok((
        TraceOp::Conv2d {
            weight,
            bias,
            padding: [
                safe_usize(padding[0], "padding", t)?,
                safe_usize(padding.get(1).copied().unwrap_or(padding[0]), "padding", t)?,
            ],
            stride: [
                safe_usize(stride[0], "stride", t)?,
                safe_usize(stride.get(1).copied().unwrap_or(stride[0]), "stride", t)?,
            ],
            dilation: [
                safe_usize(dilation[0], "dilation", t)?,
                safe_usize(
                    dilation.get(1).copied().unwrap_or(dilation[0]),
                    "dilation",
                    t,
                )?,
            ],
            groups: safe_usize(groups, "groups", t)?,
        },
        vec![input],
    ))
}

// -- Standalone batch_norm --

/// Map `aten.batch_norm.default` to `TraceOp::BatchNorm`.
///
/// Some PyTorch export paths emit `batch_norm.default` instead of the lower-level
/// `native_batch_norm.default` or `_native_batch_norm_legit_no_training.default`.
/// Delegates to the same mapper as the native variants.
pub(super) fn map_batch_norm_standalone(
    node: &Node,
    ctx: &OpMapContext<'_>,
) -> Result<(TraceOp, Vec<String>), ImportError> {
    let input = require_tensor_name(node, "input")?;
    let weight_name = require_tensor_name(node, "weight")?;
    let bias_name = require_tensor_name(node, "bias")?;
    let running_mean_name = require_tensor_name(node, "running_mean")?;
    let running_var_name = require_tensor_name(node, "running_var")?;
    let weight = resolve_weight(&weight_name, ctx)?;
    let bias = resolve_weight(&bias_name, ctx)?;
    let running_mean = resolve_weight(&running_mean_name, ctx)?;
    let running_var = resolve_weight(&running_var_name, ctx)?;
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

// -- Recurrent --

/// Map `aten.lstm.input` to `TraceOp::Lstm`.
///
/// torch.export serializes LSTM as:
///   input: Tensor, hx: [h_0, c_0], params: [w_ih, w_hh, b_ih?, b_hh?],
///   has_biases: bool, num_layers: int, dropout: float, train: bool,
///   bidirectional: bool, batch_first: bool
///
/// Only single-layer, non-bidirectional LSTM is supported (matching TraceOp::Lstm).
pub(super) fn map_lstm(
    node: &Node,
    ctx: &OpMapContext<'_>,
) -> Result<(TraceOp, Vec<String>), ImportError> {
    let input = require_tensor_name(node, "input")?;

    // hx: [h_0, c_0] — initial hidden and cell states.
    let hx_arg = get_arg(node, "hx")?;
    let hx_names = hx_arg
        .as_tensor_names()
        .ok_or_else(|| ImportError::WrongArgumentType {
            op_target: node.target.clone(),
            arg_name: "hx".to_string(),
            expected: "tensor list [h_0, c_0]",
            actual: "non-tensor-list".to_string(),
        })?;
    if hx_names.len() != 2 {
        return Err(ImportError::WrongArgumentType {
            op_target: node.target.clone(),
            arg_name: "hx".to_string(),
            expected: "tensor list of length 2 [h_0, c_0]",
            actual: format!("tensor list of length {}", hx_names.len()),
        });
    }
    let h_0 = hx_names[0].to_string();
    let c_0 = hx_names[1].to_string();

    // Validate single-layer, non-bidirectional.
    let num_layers = optional_int(node, "num_layers").unwrap_or(1);
    if num_layers != 1 {
        return Err(ImportError::UnsupportedOp {
            target: format!(
                "{} (num_layers={num_layers}, only single-layer LSTM supported)",
                node.target
            ),
        });
    }
    let bidirectional = optional_bool(node, "bidirectional", false);
    if bidirectional {
        return Err(ImportError::UnsupportedOp {
            target: format!("{} (bidirectional=true not supported)", node.target),
        });
    }

    // params: flat weight list [w_ih_l0, w_hh_l0, b_ih_l0?, b_hh_l0?].
    let has_biases = optional_bool(node, "has_biases", true);
    let params_arg = get_arg(node, "params")?;
    let param_names =
        params_arg
            .as_tensor_names()
            .ok_or_else(|| ImportError::WrongArgumentType {
                op_target: node.target.clone(),
                arg_name: "params".to_string(),
                expected: "tensor list",
                actual: "non-tensor-list".to_string(),
            })?;

    let expected_len = if has_biases { 4 } else { 2 };
    if param_names.len() < expected_len {
        return Err(ImportError::WrongArgumentType {
            op_target: node.target.clone(),
            arg_name: "params".to_string(),
            expected: if has_biases {
                "tensor list [w_ih, w_hh, b_ih, b_hh]"
            } else {
                "tensor list [w_ih, w_hh]"
            },
            actual: format!("tensor list of length {}", param_names.len()),
        });
    }

    let weight_ih = resolve_weight(param_names[0], ctx)?;
    let weight_hh = resolve_weight(param_names[1], ctx)?;
    let (bias_ih, bias_hh) = if has_biases {
        (
            Some(resolve_weight(param_names[2], ctx)?),
            Some(resolve_weight(param_names[3], ctx)?),
        )
    } else {
        (None, None)
    };

    // hidden_size = w_hh.shape[1] (w_hh is [4*hidden_size, hidden_size]).
    let hidden_size = *weight_hh
        .shape()
        .get(1)
        .ok_or_else(|| ImportError::WrongArgumentType {
            op_target: node.target.clone(),
            arg_name: "params (w_hh)".to_string(),
            expected: "2D weight [4*H, H]",
            actual: format!("shape {:?}", weight_hh.shape()),
        })?;

    Ok((
        TraceOp::Lstm {
            weight_ih,
            weight_hh,
            bias_ih,
            bias_hh,
            hidden_size,
            initial_hidden: None,
            initial_cell: None,
        },
        vec![input, h_0, c_0],
    ))
}

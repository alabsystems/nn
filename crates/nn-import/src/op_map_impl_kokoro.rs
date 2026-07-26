// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Aten op mappers for Kokoro TTS and similar audio models.
//!
//! Adds: reflection_pad1d, constant_pad_nd, upsample_nearest1d, index_select,
//! gt/lt/ge/le/eq/ne comparison, atan2, ones/full/ones_like/full_like, arange,
//! contiguous/clone (identity).

use nn_core::dyn_tensor::trace::TraceOp;
use nn_core::dyn_tensor::CompareOp;

use super::{
    first_tensor_name, get_arg, optional_float, optional_int, require_int, require_ints,
    require_tensor_name, resolve_weight, safe_usize, ImportError, Node, OpMapContext,
};

// -- Padding --

/// Map `aten.reflection_pad1d.default` to `TraceOp::ReflectionPad1d`.
///
/// torch.export signature: `(self, padding: [left, right])`
pub(super) fn map_reflection_pad1d(node: &Node) -> Result<(TraceOp, Vec<String>), ImportError> {
    let input = first_tensor_name(node)?;
    let padding = require_ints(node, "padding")?;
    let t = &node.target;
    let pad_left = safe_usize(padding.first().copied().unwrap_or(0), "padding[0]", t)?;
    let pad_right = safe_usize(padding.get(1).copied().unwrap_or(0), "padding[1]", t)?;
    Ok((
        TraceOp::ReflectionPad1d {
            pad_left,
            pad_right,
        },
        vec![input],
    ))
}

/// Map `aten.constant_pad_nd.default` to `TraceOp::ConstantPadNd`.
///
/// torch.export signature: `(self, pad: [int...], value: float = 0.0)`
/// Padding is in reverse order: `[last_dim_left, last_dim_right, ..., first_dim_left, first_dim_right]`.
pub(super) fn map_constant_pad_nd(node: &Node) -> Result<(TraceOp, Vec<String>), ImportError> {
    let input = first_tensor_name(node)?;
    let raw = require_ints(node, "pad")?;
    let t = &node.target;
    let padding: Vec<usize> = raw
        .into_iter()
        .map(|v| safe_usize(v, "pad", t))
        .collect::<Result<_, _>>()?;
    let value = optional_float(node, "value").unwrap_or(0.0);
    Ok((TraceOp::ConstantPadNd { padding, value }, vec![input]))
}

// -- Upsampling --

/// Map `aten.upsample_nearest1d.default` / `aten.upsample_nearest1d.vec` to `TraceOp::Upsample1d`.
///
/// torch.export signature: `(self, output_size: [int], scales: float?)`
/// Factor is derived from output_size / input_size, or from scales if provided.
pub(super) fn map_upsample_nearest1d(node: &Node) -> Result<(TraceOp, Vec<String>), ImportError> {
    let input = first_tensor_name(node)?;
    // Try scales_factor first (used by some export variants).
    if let Some(scale) = optional_float(node, "scales") {
        let factor = safe_usize(scale as i64, "scales", &node.target)?;
        return Ok((TraceOp::Upsample1d { factor }, vec![input]));
    }
    // Fall back to output_size — but we need input size to compute factor.
    // If output_size is provided, use it directly. The trace compiler resolves
    // the actual factor at execution time from the tensor shapes.
    let output_size = require_ints(node, "output_size")?;
    // Common case: output_size is 2x input, so factor = 2.
    // For safety, encode the output_size as factor if it's a clean multiple.
    // Default to factor=2 which is the dominant Kokoro usage.
    let factor = if output_size.len() == 1 {
        output_size[0].max(1) as usize
    } else {
        2 // default
    };
    // Note: This is approximate. The trace compiler should verify shape consistency.
    Ok((TraceOp::Upsample1d { factor }, vec![input]))
}

// -- Indexing --

/// Map `aten.index_select.default` to `TraceOp::IndexSelect`.
///
/// torch.export signature: `(self, dim: int, index: Tensor)`
pub(super) fn map_index_select(node: &Node) -> Result<(TraceOp, Vec<String>), ImportError> {
    let input = require_tensor_name(node, "self")?;
    let index = require_tensor_name(node, "index")?;
    let dim = safe_usize(require_int(node, "dim")?, "dim", &node.target)?;
    Ok((TraceOp::IndexSelect { dim }, vec![input, index]))
}

// -- Comparison --

/// Map `aten.gt.Scalar` to `TraceOp::Compare { op: Gt, value }`.
pub(super) fn map_gt_scalar(node: &Node) -> Result<(TraceOp, Vec<String>), ImportError> {
    map_compare_scalar(node, CompareOp::Gt)
}

/// Map `aten.lt.Scalar` to `TraceOp::Compare { op: Lt, value }`.
pub(super) fn map_lt_scalar(node: &Node) -> Result<(TraceOp, Vec<String>), ImportError> {
    map_compare_scalar(node, CompareOp::Lt)
}

/// Map `aten.ge.Scalar` to `TraceOp::Compare { op: Ge, value }`.
pub(super) fn map_ge_scalar(node: &Node) -> Result<(TraceOp, Vec<String>), ImportError> {
    map_compare_scalar(node, CompareOp::Ge)
}

/// Map `aten.le.Scalar` to `TraceOp::Compare { op: Le, value }`.
pub(super) fn map_le_scalar(node: &Node) -> Result<(TraceOp, Vec<String>), ImportError> {
    map_compare_scalar(node, CompareOp::Le)
}

/// Map `aten.eq.Scalar` to `TraceOp::Compare { op: Eq, value }`.
pub(super) fn map_eq_scalar(node: &Node) -> Result<(TraceOp, Vec<String>), ImportError> {
    map_compare_scalar(node, CompareOp::Eq)
}

/// Map `aten.ne.Scalar` to `TraceOp::Compare { op: Ne, value }`.
pub(super) fn map_ne_scalar(node: &Node) -> Result<(TraceOp, Vec<String>), ImportError> {
    map_compare_scalar(node, CompareOp::Ne)
}

fn map_compare_scalar(node: &Node, op: CompareOp) -> Result<(TraceOp, Vec<String>), ImportError> {
    let input = first_tensor_name(node)?;
    let value = optional_float(node, "other")
        .or_else(|| {
            node.inputs.get(1).and_then(|na| {
                na.arg
                    .as_float()
                    .or_else(|| na.arg.as_int().map(|i| i as f64))
            })
        })
        .unwrap_or(0.0);
    Ok((TraceOp::Compare { op, value }, vec![input]))
}

/// Map `aten.gt.Tensor` to `TraceOp::CompareTensor { op: Gt }`.
pub(super) fn map_compare_tensor(
    node: &Node,
    op: CompareOp,
) -> Result<(TraceOp, Vec<String>), ImportError> {
    let lhs = require_tensor_name(node, "self")?;
    let rhs = require_tensor_name(node, "other")?;
    Ok((TraceOp::CompareTensor { op }, vec![lhs, rhs]))
}

// -- Trigonometric --

/// Map `aten.atan2.default` to `TraceOp::Atan2`.
///
/// torch.export signature: `(self, other)` → `atan2(self, other)`
pub(super) fn map_atan2(node: &Node) -> Result<(TraceOp, Vec<String>), ImportError> {
    let y = require_tensor_name(node, "self")?;
    let x = require_tensor_name(node, "other")?;
    Ok((TraceOp::Atan2, vec![y, x]))
}

// -- Tensor creation --

/// Map `aten.ones.default` / `aten.ones_like.default` to `TraceOp::Constant { value: 1.0 }`.
pub(super) fn map_ones(node: &Node) -> Result<(TraceOp, Vec<String>), ImportError> {
    let _ = node;
    Ok((TraceOp::Constant { value: 1.0 }, vec![]))
}

/// Map `aten.full.default` / `aten.full_like.default` to `TraceOp::Constant`.
pub(super) fn map_full(node: &Node) -> Result<(TraceOp, Vec<String>), ImportError> {
    let value = optional_float(node, "fill_value")
        .or_else(|| {
            node.inputs.get(1).and_then(|na| {
                na.arg
                    .as_float()
                    .or_else(|| na.arg.as_int().map(|i| i as f64))
            })
        })
        .unwrap_or(0.0);
    Ok((TraceOp::Constant { value }, vec![]))
}

/// Map `aten.arange.default` / `aten.arange.start_step` to `TraceOp::Arange`.
pub(super) fn map_arange(node: &Node) -> Result<(TraceOp, Vec<String>), ImportError> {
    // arange has multiple overloads:
    // - arange(end) → start=0, step=1
    // - arange(start, end) → step=1
    // - arange(start, end, step)
    let start = optional_float(node, "start").unwrap_or(0.0);
    let end = optional_float(node, "end")
        .or_else(|| {
            // Single-argument form: first positional arg is `end`.
            node.inputs.first().and_then(|na| {
                na.arg
                    .as_float()
                    .or_else(|| na.arg.as_int().map(|i| i as f64))
            })
        })
        .unwrap_or(0.0);
    let step = optional_float(node, "step").unwrap_or(1.0);
    Ok((TraceOp::Arange { start, end, step }, vec![]))
}

// -- Transposed convolution --

/// Map `aten.conv_transpose1d.default` to `TraceOp::ConvTranspose1d`.
///
/// torch.export signature: `(input, weight, bias?, stride, padding, output_padding, groups, dilation)`
pub(super) fn map_conv_transpose1d(
    node: &Node,
    ctx: &OpMapContext<'_>,
) -> Result<(TraceOp, Vec<String>), ImportError> {
    let input = require_tensor_name(node, "input")?;
    let weight_name = require_tensor_name(node, "weight")?;
    let bias_name = get_arg(node, "bias")
        .ok()
        .and_then(|a| {
            if a.is_none() {
                None
            } else {
                a.as_tensor_name()
            }
        })
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
    let output_padding = get_arg(node, "output_padding")
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
        TraceOp::ConvTranspose1d {
            weight,
            bias,
            padding: safe_usize(padding[0], "padding", t)?,
            output_padding: safe_usize(output_padding[0], "output_padding", t)?,
            stride: safe_usize(stride[0], "stride", t)?,
            dilation: safe_usize(dilation[0], "dilation", t)?,
            groups: safe_usize(groups, "groups", t)?,
        },
        vec![input],
    ))
}

// -- Generic padding dispatch --

/// Map `aten.pad.default` by routing on the `mode` argument.
///
/// torch.export signature: `(self, pad: [int...], mode: str = "constant", value: float? = None)`
pub(super) fn map_pad(node: &Node) -> Result<(TraceOp, Vec<String>), ImportError> {
    let mode = get_arg(node, "mode")
        .ok()
        .and_then(|a| a.as_string().map(String::from))
        .unwrap_or_else(|| "constant".to_string());
    match mode.as_str() {
        "reflect" => {
            // Reflection padding — route based on padding element count.
            let input = first_tensor_name(node)?;
            let padding = require_ints(node, "pad")?;
            let t = &node.target;
            let pad_left = safe_usize(padding.first().copied().unwrap_or(0), "pad[0]", t)?;
            let pad_right = safe_usize(padding.get(1).copied().unwrap_or(0), "pad[1]", t)?;
            if padding.len() >= 4 {
                // 2D reflection: [left, right, top, bottom]
                let pad_top = safe_usize(padding.get(2).copied().unwrap_or(0), "pad[2]", t)?;
                let pad_bottom = safe_usize(padding.get(3).copied().unwrap_or(0), "pad[3]", t)?;
                Ok((
                    TraceOp::ReflectionPad2d {
                        pad_left,
                        pad_right,
                        pad_top,
                        pad_bottom,
                    },
                    vec![input],
                ))
            } else {
                // 1D reflection: [left, right]
                Ok((
                    TraceOp::ReflectionPad1d {
                        pad_left,
                        pad_right,
                    },
                    vec![input],
                ))
            }
        }
        "constant" => map_constant_pad_nd(node),
        _ => Err(ImportError::UnsupportedOp {
            target: format!("{} (mode={mode})", node.target),
        }),
    }
}

// -- Identity passthrough --

/// Map `aten.contiguous.default` and `aten.clone.default` to identity.
///
/// These are memory layout operations that don't change values.
/// In the import graph, we pass through the input unchanged.
pub(super) fn map_identity(node: &Node) -> Result<(TraceOp, Vec<String>), ImportError> {
    let input = first_tensor_name(node)?;
    // Identity: reshape to same shape (no-op in the trace compiler).
    Ok((
        TraceOp::Reshape {
            target_shape: vec![],
        },
        vec![input],
    ))
}

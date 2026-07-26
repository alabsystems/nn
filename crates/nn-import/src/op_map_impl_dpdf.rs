// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Aten op mappers for dpdf document processing models.
//!
//! Adds support for ops used by the six dpdf model architectures:
//! - DocLayout-YOLO: upsample_nearest2d, hardswish, hardsigmoid, mish,
//!   pixel_shuffle
//! - Granite-Docling: rms_norm, split, triu
//! - PaddleOCR-VL: (most ops already covered by base + ext)
//! - Table Transformer: upsample_bilinear2d, tril
//! - Qwen3-VL: rms_norm (shared with Granite-Docling)
//! - FireRed-OCR: argmax (CTC decoder)

use nn_core::dyn_tensor::trace::{TraceOp, TraceUpsampleMode};

use super::{
    first_tensor_name, get_arg, optional_bool, optional_float, optional_int, require_int,
    require_ints, require_tensor_name, resolve_weight, safe_usize, safe_usize_vec, ImportError,
    Node, OpMapContext,
};

// -- Upsampling 2D --

/// Map `aten.upsample_nearest2d.default` / `aten.upsample_nearest2d.vec` to
/// `TraceOp::Upsample2d { mode: Nearest }`.
///
/// torch.export signature: `(self, output_size: [H, W], scales_h: float?, scales_w: float?)`
pub(super) fn map_upsample_nearest2d(node: &Node) -> Result<(TraceOp, Vec<String>), ImportError> {
    let input = first_tensor_name(node)?;
    let scale_h = optional_float(node, "scales_h").unwrap_or(2.0);
    let scale_w = optional_float(node, "scales_w").unwrap_or(2.0);
    Ok((
        TraceOp::Upsample2d {
            mode: TraceUpsampleMode::Nearest,
            scale_h,
            scale_w,
        },
        vec![input],
    ))
}

/// Map `aten.upsample_bilinear2d.default` / `aten.upsample_bilinear2d.vec` to
/// `TraceOp::Upsample2d { mode: Bilinear }`.
///
/// torch.export signature: `(self, output_size: [H, W], align_corners: bool, scales_h?, scales_w?)`
pub(super) fn map_upsample_bilinear2d(node: &Node) -> Result<(TraceOp, Vec<String>), ImportError> {
    let input = first_tensor_name(node)?;
    let scale_h = optional_float(node, "scales_h").unwrap_or(2.0);
    let scale_w = optional_float(node, "scales_w").unwrap_or(2.0);
    Ok((
        TraceOp::Upsample2d {
            mode: TraceUpsampleMode::Bilinear,
            scale_h,
            scale_w,
        },
        vec![input],
    ))
}

// -- Normalization --

/// Map `aten.rms_norm.default` to `TraceOp::RmsNorm`.
///
/// torch.export signature: `(self, normalized_shape: [int], weight: Tensor, eps: float?)`
///
/// Note: torch.export may decompose rms_norm into elementwise ops. This mapper
/// handles the case where the graph preserves the fused op.
pub(super) fn map_rms_norm(
    node: &Node,
    ctx: &OpMapContext<'_>,
) -> Result<(TraceOp, Vec<String>), ImportError> {
    let input = require_tensor_name(node, "input")?;
    let weight_name = require_tensor_name(node, "weight")?;
    let weight = resolve_weight(&weight_name, ctx)?;
    let eps = optional_float(node, "eps").unwrap_or(1e-6);
    Ok((TraceOp::RmsNorm { eps, weight }, vec![input]))
}

// -- Activation --
// NOTE: hardswish, hardsigmoid, mish, softplus, selu are handled by the main
// activation section (impls/impls_ext). Removed duplicate dpdf implementations.

// -- Mask ops --

/// Map `aten.triu.default` to `TraceOp::Triu`.
///
/// torch.export signature: `(self, diagonal: int = 0)`
pub(super) fn map_triu(node: &Node) -> Result<(TraceOp, Vec<String>), ImportError> {
    let input = first_tensor_name(node)?;
    let diagonal = optional_int(node, "diagonal").unwrap_or(0);
    Ok((TraceOp::Triu { diagonal }, vec![input]))
}

/// Map `aten.tril.default` to `TraceOp::Tril`.
///
/// torch.export signature: `(self, diagonal: int = 0)`
pub(super) fn map_tril(node: &Node) -> Result<(TraceOp, Vec<String>), ImportError> {
    let input = first_tensor_name(node)?;
    let diagonal = optional_int(node, "diagonal").unwrap_or(0);
    Ok((TraceOp::Tril { diagonal }, vec![input]))
}

// -- Selection / Indexing --

/// Map `aten.gather.default` to `TraceOp::Gather`.
///
/// torch.export signature: `(self, dim: int, index: Tensor, sparse_grad: bool = False)`
pub(super) fn map_gather(node: &Node) -> Result<(TraceOp, Vec<String>), ImportError> {
    let input = require_tensor_name(node, "self")?;
    let index = require_tensor_name(node, "index")?;
    let dim = safe_usize(require_int(node, "dim")?, "dim", &node.target)?;
    Ok((TraceOp::Gather { dim }, vec![input, index]))
}

/// Map `aten.argmax.default` to `TraceOp::Argmax`.
///
/// torch.export signature: `(self, dim: int?, keepdim: bool = False)`
pub(super) fn map_argmax(node: &Node) -> Result<(TraceOp, Vec<String>), ImportError> {
    let input = first_tensor_name(node)?;
    let dim = safe_usize(optional_int(node, "dim").unwrap_or(-1), "dim", &node.target).unwrap_or(0);
    Ok((TraceOp::Argmax { dim }, vec![input]))
}

/// Map `aten.argmin.default` to `TraceOp::Argmin`.
pub(super) fn map_argmin(node: &Node) -> Result<(TraceOp, Vec<String>), ImportError> {
    let input = first_tensor_name(node)?;
    let dim = safe_usize(optional_int(node, "dim").unwrap_or(-1), "dim", &node.target).unwrap_or(0);
    Ok((TraceOp::Argmin { dim }, vec![input]))
}

// -- Vision --

/// Map `aten.pixel_shuffle.default` to `TraceOp::PixelShuffle`.
///
/// torch.export signature: `(self, upscale_factor: int)`
pub(super) fn map_pixel_shuffle(node: &Node) -> Result<(TraceOp, Vec<String>), ImportError> {
    let input = first_tensor_name(node)?;
    let factor = safe_usize(
        require_int(node, "upscale_factor")?,
        "upscale_factor",
        &node.target,
    )?;
    Ok((
        TraceOp::PixelShuffle {
            upscale_factor: factor,
        },
        vec![input],
    ))
}

/// Map `aten.pixel_unshuffle.default` to `TraceOp::PixelUnshuffle`.
///
/// torch.export signature: `(self, downscale_factor: int)`
pub(super) fn map_pixel_unshuffle(node: &Node) -> Result<(TraceOp, Vec<String>), ImportError> {
    let input = first_tensor_name(node)?;
    let factor = safe_usize(
        require_int(node, "downscale_factor")?,
        "downscale_factor",
        &node.target,
    )?;
    Ok((
        TraceOp::PixelUnshuffle {
            downscale_factor: factor,
        },
        vec![input],
    ))
}

// -- Repeat --

/// Map `aten.repeat.default` to `TraceOp::Expand` (via repeat semantics).
///
/// torch.export signature: `(self, repeats: [int...])`
/// Repeat tiles the tensor along each dimension. We translate to Expand
/// for the common case where the input has shape 1 along the repeated dims.
/// For general repeat (non-1 input dims), this falls back to Reshape+Expand.
pub(super) fn map_repeat(node: &Node) -> Result<(TraceOp, Vec<String>), ImportError> {
    let input = first_tensor_name(node)?;
    let repeats = require_ints(node, "repeats")?;
    let target_shape: Vec<usize> = repeats
        .into_iter()
        .map(|v| safe_usize(v, "repeats", &node.target))
        .collect::<Result<_, _>>()?;
    // Encode as Expand with the repeat counts as target shape.
    // The trace compiler will handle the semantics based on input shape.
    Ok((TraceOp::Expand { target_shape }, vec![input]))
}

/// Map `aten.unbind.int` via multi-node expansion into N select ops.
///
/// `unbind(dim)` splits a tensor into individual slices along `dim`, each with
/// that dimension removed. We decompose into N (Narrow + Reshape) pairs via
/// `expand_unbind` in `op_map_expand.rs`.
///
/// This single-op path returns an error since unbind always produces multiple
/// outputs and must go through the `try_expand_node` path.
pub(super) fn map_unbind_fallback(node: &Node) -> Result<(TraceOp, Vec<String>), ImportError> {
    Err(ImportError::UnsupportedOp {
        target: format!(
            "{} (unbind must be expanded via try_expand_node; missing input shape metadata)",
            node.target
        ),
    })
}

// -- Interpolate (multi-mode upsample) --

/// Map `aten.interpolate.default` / `aten.interpolate.vec` to
/// `TraceOp::Upsample2d` based on the `mode` argument.
///
/// torch.export signature: `(input, size?, scale_factor?, mode="nearest", ...)`
pub(super) fn map_interpolate(node: &Node) -> Result<(TraceOp, Vec<String>), ImportError> {
    use nn_core::dyn_tensor::trace::TraceUpsampleMode;

    let input = first_tensor_name(node)?;
    let mode = get_arg(node, "mode")
        .ok()
        .and_then(|a| a.as_string().map(String::from))
        .unwrap_or_else(|| "nearest".to_string());
    let scale_h = optional_float(node, "scales_h")
        .or_else(|| optional_float(node, "scale_factor"))
        .unwrap_or(2.0);
    let scale_w = optional_float(node, "scales_w")
        .or_else(|| optional_float(node, "scale_factor"))
        .unwrap_or(2.0);
    let upsample_mode = match mode.as_str() {
        "nearest" => TraceUpsampleMode::Nearest,
        "bilinear" => TraceUpsampleMode::Bilinear,
        "bicubic" => TraceUpsampleMode::Bicubic,
        _ => {
            return Err(ImportError::UnsupportedOp {
                target: format!("{} (mode={mode})", node.target),
            });
        }
    };
    Ok((
        TraceOp::Upsample2d {
            mode: upsample_mode,
            scale_h,
            scale_w,
        },
        vec![input],
    ))
}

// -- Scatter (overwrite) --

/// Map `aten.scatter.src` to `TraceOp::Scatter`.
///
/// torch.export signature: `(self, dim: int, index: Tensor, src: Tensor)`
pub(super) fn map_scatter(node: &Node) -> Result<(TraceOp, Vec<String>), ImportError> {
    let self_input = require_tensor_name(node, "self")?;
    let index = require_tensor_name(node, "index")?;
    let src = require_tensor_name(node, "src")?;
    let dim = safe_usize(require_int(node, "dim")?, "dim", &node.target)?;
    Ok((TraceOp::Scatter { dim }, vec![self_input, index, src]))
}

// -- Reflection pad 2D --

/// Map `aten.reflection_pad2d.default` to `TraceOp::ReflectionPad2d`.
///
/// torch.export signature: `(self, padding: [left, right, top, bottom])`
pub(super) fn map_reflection_pad2d(node: &Node) -> Result<(TraceOp, Vec<String>), ImportError> {
    let input = first_tensor_name(node)?;
    let padding = require_ints(node, "padding")?;
    let t = &node.target;
    let pad_left = safe_usize(padding.first().copied().unwrap_or(0), "padding[0]", t)?;
    let pad_right = safe_usize(padding.get(1).copied().unwrap_or(0), "padding[1]", t)?;
    let pad_top = safe_usize(padding.get(2).copied().unwrap_or(0), "padding[2]", t)?;
    let pad_bottom = safe_usize(padding.get(3).copied().unwrap_or(0), "padding[3]", t)?;
    Ok((
        TraceOp::ReflectionPad2d {
            pad_left,
            pad_right,
            pad_top,
            pad_bottom,
        },
        vec![input],
    ))
}

// -- Clamp max --

/// Map `aten.clamp_max.default` to `TraceOp::Clamp { min: None, max: Some(max) }`.
///
/// torch.export signature: `(self, max: Scalar)`
pub(super) fn map_clamp_max(node: &Node) -> Result<(TraceOp, Vec<String>), ImportError> {
    let input = first_tensor_name(node)?;
    let max = optional_float(node, "max").or_else(|| {
        // Positional fallback: second arg after self.
        node.inputs.get(1).and_then(|a| a.arg.as_float())
    });
    Ok((TraceOp::Clamp { min: None, max }, vec![input]))
}

// -- Narrow --

/// Map `aten.narrow.default` to `TraceOp::Narrow`.
///
/// torch.export signature: `(self, dim: int, start: int|Tensor, length: int)`
pub(super) fn map_narrow(node: &Node) -> Result<(TraceOp, Vec<String>), ImportError> {
    let input = first_tensor_name(node)?;
    let dim = safe_usize(require_int(node, "dim")?, "dim", &node.target)?;
    let start = safe_usize(require_int(node, "start")?, "start", &node.target)?;
    let length = safe_usize(require_int(node, "length")?, "length", &node.target)?;
    Ok((TraceOp::Narrow { dim, start, length }, vec![input]))
}

// -- TopK --

/// Map `aten.topk.default` to `TraceOp::Topk`.
///
/// torch.export signature: `(self, k: int, dim: int = -1, largest: bool = True, sorted: bool = True)`
pub(super) fn map_topk(node: &Node) -> Result<(TraceOp, Vec<String>), ImportError> {
    let input = first_tensor_name(node)?;
    let k = safe_usize(require_int(node, "k")?, "k", &node.target)?;
    let dim = safe_usize(optional_int(node, "dim").unwrap_or(0), "dim", &node.target)?;
    Ok((TraceOp::Topk { k, dim }, vec![input]))
}

// -- Sort --

/// Map `aten.sort.default` to `TraceOp::Sort`.
///
/// torch.export signature: `(self, dim: int = -1, descending: bool = False)`
pub(super) fn map_sort(node: &Node) -> Result<(TraceOp, Vec<String>), ImportError> {
    let input = first_tensor_name(node)?;
    let dim = safe_usize(optional_int(node, "dim").unwrap_or(0), "dim", &node.target)?;
    let descending = optional_bool(node, "descending", false);
    Ok((TraceOp::Sort { dim, descending }, vec![input]))
}

// -- Scatter add --

/// Map `aten.scatter_add.default` to `TraceOp::ScatterAdd`.
///
/// torch.export signature: `(self, dim: int, index: Tensor, src: Tensor)`
pub(super) fn map_scatter_add(node: &Node) -> Result<(TraceOp, Vec<String>), ImportError> {
    let self_input = require_tensor_name(node, "self")?;
    let index = require_tensor_name(node, "index")?;
    let src = require_tensor_name(node, "src")?;
    let dim = safe_usize(require_int(node, "dim")?, "dim", &node.target)?;
    Ok((TraceOp::ScatterAdd { dim }, vec![self_input, index, src]))
}

// -- Scatter value --

/// Map `aten.scatter.value` to `TraceOp::Scatter`.
///
/// torch.export signature: `(self, dim: int, index: Tensor, value: Scalar)`
pub(super) fn map_scatter_value(node: &Node) -> Result<(TraceOp, Vec<String>), ImportError> {
    let self_input = require_tensor_name(node, "self")?;
    let index = require_tensor_name(node, "index")?;
    let dim = safe_usize(require_int(node, "dim")?, "dim", &node.target)?;
    Ok((TraceOp::Scatter { dim }, vec![self_input, index]))
}

// -- Roll --

/// Map `aten.roll.default` to `TraceOp::Roll`.
///
/// torch.export signature: `(self, shifts: [int...], dims: [int...] = [0])`
pub(super) fn map_roll(node: &Node) -> Result<(TraceOp, Vec<String>), ImportError> {
    let input = first_tensor_name(node)?;
    let shifts = require_ints(node, "shifts")?;
    let raw_dims = get_arg(node, "dims")
        .ok()
        .and_then(|a| a.as_ints().map(<[i64]>::to_vec));
    let dims = match raw_dims {
        Some(d) => safe_usize_vec(d, "dims", &node.target)?,
        None => vec![0],
    };
    Ok((TraceOp::Roll { shifts, dims }, vec![input]))
}

// -- Conv3d (standalone) --

/// Map `aten.conv3d.default` to `TraceOp::Conv3d`.
///
/// torch.export signature: `(input, weight, bias?, stride=[1,1,1], padding=[0,0,0], dilation=[1,1,1], groups=1)`
pub(super) fn map_conv3d(
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
    let stride_raw = get_arg(node, "stride")
        .ok()
        .and_then(|a| a.as_ints())
        .map(<[i64]>::to_vec)
        .unwrap_or_else(|| vec![1, 1, 1]);
    let padding_raw = get_arg(node, "padding")
        .ok()
        .and_then(|a| a.as_ints())
        .map(<[i64]>::to_vec)
        .unwrap_or_else(|| vec![0, 0, 0]);
    let dilation_raw = get_arg(node, "dilation")
        .ok()
        .and_then(|a| a.as_ints())
        .map(<[i64]>::to_vec)
        .unwrap_or_else(|| vec![1, 1, 1]);
    let groups = optional_int(node, "groups").unwrap_or(1);
    Ok((
        TraceOp::Conv3d {
            weight,
            bias,
            padding: [
                safe_usize(padding_raw[0], "padding", t)?,
                safe_usize(
                    padding_raw.get(1).copied().unwrap_or(padding_raw[0]),
                    "padding",
                    t,
                )?,
                safe_usize(
                    padding_raw.get(2).copied().unwrap_or(padding_raw[0]),
                    "padding",
                    t,
                )?,
            ],
            stride: [
                safe_usize(stride_raw[0], "stride", t)?,
                safe_usize(
                    stride_raw.get(1).copied().unwrap_or(stride_raw[0]),
                    "stride",
                    t,
                )?,
                safe_usize(
                    stride_raw.get(2).copied().unwrap_or(stride_raw[0]),
                    "stride",
                    t,
                )?,
            ],
            dilation: [
                safe_usize(dilation_raw[0], "dilation", t)?,
                safe_usize(
                    dilation_raw.get(1).copied().unwrap_or(dilation_raw[0]),
                    "dilation",
                    t,
                )?,
                safe_usize(
                    dilation_raw.get(2).copied().unwrap_or(dilation_raw[0]),
                    "dilation",
                    t,
                )?,
            ],
            groups: safe_usize(groups, "groups", t)?,
        },
        vec![input],
    ))
}

// -- Grid sample --

/// Map `aten.grid_sample.default` to `TraceOp::GridSample`.
///
/// torch.export signature:
/// `(input, grid, interpolation_mode=0, padding_mode=0, align_corners=False)`
///
/// interpolation_mode: 0=bilinear, 1=nearest, 2=bicubic (only bilinear supported)
/// padding_mode: 0=zeros, 1=border, 2=reflection (zeros + border supported)
pub(super) fn map_grid_sample(node: &Node) -> Result<(TraceOp, Vec<String>), ImportError> {
    use nn_core::dyn_tensor::GridSamplePaddingMode;

    let input = require_tensor_name(node, "self")?;
    let grid = require_tensor_name(node, "grid")?;
    let interp_mode = optional_int(node, "interpolation_mode").unwrap_or(0);
    if interp_mode != 0 {
        return Err(ImportError::UnsupportedOp {
            target: format!(
                "{} (only bilinear interpolation_mode=0 supported, got {interp_mode})",
                node.target
            ),
        });
    }
    let pad_mode_int = optional_int(node, "padding_mode").unwrap_or(0);
    let padding_mode = match pad_mode_int {
        0 => GridSamplePaddingMode::Zeros,
        1 => GridSamplePaddingMode::Border,
        _ => {
            return Err(ImportError::UnsupportedOp {
                target: format!(
                    "{} (padding_mode={pad_mode_int} not supported, use 0=zeros or 1=border)",
                    node.target
                ),
            });
        }
    };
    let align_corners = optional_bool(node, "align_corners", false);
    Ok((
        TraceOp::GridSample {
            padding_mode,
            align_corners,
        },
        vec![input, grid],
    ))
}

// -- Masked fill (decomposed to WhereCond) --

/// Map `aten.masked_fill.Scalar` / `aten.masked_fill_.Scalar` to
/// a `WhereCond`-based decomposition.
///
/// `masked_fill(self, mask, value)` = `where(mask, value, self)`.
///
/// We decompose this into a Constant node for the fill value and a WhereCond.
/// This is returned as a single-op mapper that goes through the expansion path.
///
/// torch.export signature: `(self, mask: Tensor, value: Scalar)`
pub(super) fn map_masked_fill_fallback(node: &Node) -> Result<(TraceOp, Vec<String>), ImportError> {
    // masked_fill needs to be expanded (Constant + WhereCond), so return
    // an error directing to try_expand_node.
    Err(ImportError::UnsupportedOp {
        target: format!(
            "{} (masked_fill decomposes via try_expand_node into Constant + WhereCond)",
            node.target
        ),
    })
}

// -- Index.Tensor (advanced indexing) --

/// Map `aten.index.Tensor` to a decomposition.
///
/// Advanced tensor indexing `x[indices]` is complex. For the single-index case
/// (one index tensor along dim 0), we decompose to IndexSelect.
/// Multi-index cases are not yet supported.
///
/// torch.export signature: `(self, indices: List[Optional[Tensor]])`
pub(super) fn map_index_tensor_fallback(
    node: &Node,
) -> Result<(TraceOp, Vec<String>), ImportError> {
    Err(ImportError::UnsupportedOp {
        target: format!(
            "{} (index.Tensor decomposes via try_expand_node into IndexSelect/Gather)",
            node.target
        ),
    })
}

// ---------------------------------------------------------------------------
// Vision model ops: conv_transpose2d, pooling, bicubic interpolate
// ---------------------------------------------------------------------------

// -- Standalone ConvTranspose2d --

/// Map `aten.conv_transpose2d.input` to `TraceOp::ConvTranspose2d`.
///
/// Some PyTorch export paths emit `conv_transpose2d.input` instead of the
/// unified `convolution.default` with `transposed=true`. This mapper handles
/// the standalone target for document AI upsampling/decoder paths.
///
/// torch.export signature:
/// `(input, weight, bias?, stride=[1,1], padding=[0,0], output_padding=[0,0],
///  groups=1, dilation=[1,1])`
pub(super) fn map_conv_transpose2d(
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
        .unwrap_or_else(|| vec![1, 1]);
    let padding = get_arg(node, "padding")
        .ok()
        .and_then(|a| a.as_ints())
        .map(<[i64]>::to_vec)
        .unwrap_or_else(|| vec![0, 0]);
    let output_padding = get_arg(node, "output_padding")
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
        TraceOp::ConvTranspose2d {
            weight,
            bias,
            padding: [
                safe_usize(padding[0], "padding", t)?,
                safe_usize(padding.get(1).copied().unwrap_or(padding[0]), "padding", t)?,
            ],
            output_padding: [
                safe_usize(output_padding[0], "output_padding", t)?,
                safe_usize(
                    output_padding.get(1).copied().unwrap_or(output_padding[0]),
                    "output_padding",
                    t,
                )?,
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

// -- Standalone max_pool2d (without indices) --

/// Map `aten.max_pool2d.default` to `TraceOp::MaxPool2d`.
///
/// Some PyTorch export paths emit `max_pool2d.default` instead of
/// `max_pool2d_with_indices.default`. Same mapper logic — indices are
/// unused in inference mode.
pub(super) fn map_max_pool2d_plain(node: &Node) -> Result<(TraceOp, Vec<String>), ImportError> {
    let input = first_tensor_name(node)?;
    let (kernel_size, stride, padding) = super::parse_pool2d_params(node)?;
    Ok((
        TraceOp::MaxPool2d {
            kernel_size,
            stride,
            padding,
        },
        vec![input],
    ))
}

// -- AvgPool1d --

/// Map `aten.avg_pool1d.default` to `TraceOp::AvgPool1d`.
///
/// torch.export signature: `(self, kernel_size, stride?, padding?, ...)`
pub(super) fn map_avg_pool1d(node: &Node) -> Result<(TraceOp, Vec<String>), ImportError> {
    let input = first_tensor_name(node)?;
    let (kernel_size, stride, padding) = super::parse_pool1d_params(node)?;
    Ok((
        TraceOp::AvgPool1d {
            kernel_size,
            stride,
            padding,
        },
        vec![input],
    ))
}

// -- AdaptiveAvgPool1d --

/// Map `aten.adaptive_avg_pool1d.default` to `TraceOp::AdaptiveAvgPool1d`.
///
/// torch.export signature: `(self, output_size: [int])`
pub(super) fn map_adaptive_avg_pool1d(node: &Node) -> Result<(TraceOp, Vec<String>), ImportError> {
    let input = first_tensor_name(node)?;
    let output_size_ints = require_ints(node, "output_size")?;
    let output_size = safe_usize(output_size_ints[0], "output_size", &node.target)?;
    Ok((TraceOp::AdaptiveAvgPool1d { output_size }, vec![input]))
}

// -- AdaptiveMaxPool2d --

/// Map `aten.adaptive_max_pool2d.default` to `TraceOp::AdaptiveMaxPool2d`.
///
/// torch.export signature: `(self, output_size: [H, W])`
/// Used by detection model backbones (e.g., ResNet-FPN).
pub(super) fn map_adaptive_max_pool2d(node: &Node) -> Result<(TraceOp, Vec<String>), ImportError> {
    let input = first_tensor_name(node)?;
    let t = &node.target;
    let output_size = require_ints(node, "output_size")?;
    let os = [
        safe_usize(output_size[0], "output_size", t)?,
        safe_usize(output_size[1], "output_size", t)?,
    ];
    Ok((TraceOp::AdaptiveMaxPool2d { output_size: os }, vec![input]))
}

// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Pretty-print formatting for layer-level `TensorOpKind` nodes.
//!
//! Extracted from `tensor_ir_pretty.rs` to keep both files under 500 lines.
//! Core structural ops (Input, Reshape, AxisSelect, Stack, Concat, Reduce,
//! Elementwise, Broadcast) remain in the parent module.

use super::TensorOpKind;

/// Format a layer-level `TensorOpKind` node as a pretty-print string.
///
/// Returns `Some(formatted)` for layer ops, `None` for core structural ops
/// handled by the caller.
pub(super) fn format_layer_node(kind: &TensorOpKind) -> Option<String> {
    match kind {
        TensorOpKind::InstanceNorm1d {
            input,
            eps,
            axis,
            gamma,
            beta,
        } => {
            let affine = match (gamma, beta) {
                (Some(g), Some(b)) => format!(", gamma=%{}, beta=%{}", g.index(), b.index()),
                _ => String::new(),
            };
            Some(format!(
                "instance_norm_1d(%{}, eps=%{}, axis={}{})",
                input.index(),
                eps.index(),
                axis,
                affine
            ))
        }
        TensorOpKind::Conv1d {
            input,
            weight,
            bias,
            stride,
            padding,
            dilation,
            groups,
        } => {
            let bias_str = match bias {
                Some(b) => format!(", bias=%{}", b.index()),
                None => String::new(),
            };
            let dilation_str = if *dilation != 1 {
                format!(", dilation={dilation}")
            } else {
                String::new()
            };
            let groups_str = if *groups != 1 {
                format!(", groups={groups}")
            } else {
                String::new()
            };
            Some(format!(
                "conv1d(%{}, weight=%{}, stride={}, padding={}{}{}{})",
                input.index(),
                weight.index(),
                stride,
                padding,
                bias_str,
                dilation_str,
                groups_str
            ))
        }
        TensorOpKind::Conv2d {
            input,
            weight,
            bias,
            stride_h,
            stride_w,
            padding_h,
            padding_w,
            dilation_h,
            dilation_w,
            groups,
        } => {
            let bias_str = match bias {
                Some(b) => format!(", bias=%{}", b.index()),
                None => String::new(),
            };
            let dilation_str = if *dilation_h != 1 || *dilation_w != 1 {
                format!(", dilation=({dilation_h},{dilation_w})")
            } else {
                String::new()
            };
            let groups_str = if *groups != 1 {
                format!(", groups={groups}")
            } else {
                String::new()
            };
            Some(format!(
                "conv2d(%{}, weight=%{}, stride=({},{}), padding=({},{}){}{}{})",
                input.index(),
                weight.index(),
                stride_h,
                stride_w,
                padding_h,
                padding_w,
                bias_str,
                dilation_str,
                groups_str
            ))
        }
        TensorOpKind::ConvTranspose1d {
            input,
            weight,
            bias,
            stride,
            padding,
            dilation,
            groups,
            output_padding,
        } => {
            let bias_str = match bias {
                Some(b) => format!(", bias=%{}", b.index()),
                None => String::new(),
            };
            let dilation_str = if *dilation != 1 {
                format!(", dilation={dilation}")
            } else {
                String::new()
            };
            let groups_str = if *groups != 1 {
                format!(", groups={groups}")
            } else {
                String::new()
            };
            let output_padding_str = if *output_padding != 0 {
                format!(", output_padding={output_padding}")
            } else {
                String::new()
            };
            Some(format!(
                "conv_transpose_1d(%{}, weight=%{}, stride={}, padding={}{}{}{}{})",
                input.index(),
                weight.index(),
                stride,
                padding,
                bias_str,
                dilation_str,
                groups_str,
                output_padding_str
            ))
        }
        TensorOpKind::RmsNorm {
            input,
            eps,
            axis,
            weight,
        } => Some(format!(
            "rms_norm(%{}, eps=%{}, axis={}, weight=%{})",
            input.index(),
            eps.index(),
            axis,
            weight.index()
        )),
        TensorOpKind::BinaryAdd { left, right } => {
            Some(format!("binary_add(%{}, %{})", left.index(), right.index()))
        }
        TensorOpKind::BinaryMul { left, right } => {
            Some(format!("binary_mul(%{}, %{})", left.index(), right.index()))
        }
        TensorOpKind::Softmax { input, axis } => {
            Some(format!("softmax(%{}, axis={})", input.index(), axis))
        }
        TensorOpKind::Sigmoid { input } => Some(format!("sigmoid(%{})", input.index())),
        TensorOpKind::Silu { input } => Some(format!("silu(%{})", input.index())),
        TensorOpKind::Gelu { input } => Some(format!("gelu(%{})", input.index())),
        TensorOpKind::GeluErf { input } => Some(format!("gelu_erf(%{})", input.index())),
        TensorOpKind::Relu { input } => Some(format!("relu(%{})", input.index())),
        TensorOpKind::LeakyRelu {
            input,
            negative_slope,
        } => Some(format!(
            "leaky_relu(%{}, slope={})",
            input.index(),
            negative_slope
        )),
        TensorOpKind::Elu { input, alpha } => {
            Some(format!("elu(%{}, alpha={})", input.index(), alpha))
        }
        TensorOpKind::Tanh { input } => Some(format!("tanh(%{})", input.index())),
        TensorOpKind::Softplus { input } => Some(format!("softplus(%{})", input.index())),
        TensorOpKind::Exp { input } => Some(format!("exp(%{})", input.index())),
        TensorOpKind::AdaIN1d {
            input,
            eps,
            axis,
            style_gamma,
            style_beta,
        } => Some(format!(
            "adain_1d(%{}, eps=%{}, axis={}, style_gamma=%{}, style_beta=%{})",
            input.index(),
            eps.index(),
            axis,
            style_gamma.index(),
            style_beta.index()
        )),
        TensorOpKind::Narrow {
            input,
            axis,
            start,
            length,
        } => Some(format!(
            "narrow(%{}, axis={}, start={}, length={})",
            input.index(),
            axis,
            start,
            length
        )),
        TensorOpKind::Linear {
            input,
            weight,
            bias,
        } => {
            let bias_str = match bias {
                Some(b) => format!(", bias=%{}", b.index()),
                None => String::new(),
            };
            Some(format!(
                "linear(%{}, weight=%{}{})",
                input.index(),
                weight.index(),
                bias_str
            ))
        }
        TensorOpKind::MatMul {
            left,
            right,
            transpose_right,
            scale,
        } => {
            let tr_str = if *transpose_right {
                ", transpose_right=true"
            } else {
                ""
            };
            let scale_str = match scale {
                Some(s) => format!(", scale={s}"),
                None => String::new(),
            };
            Some(format!(
                "matmul(%{}, %{}{}{})",
                left.index(),
                right.index(),
                tr_str,
                scale_str
            ))
        }
        TensorOpKind::ZeroPad1d {
            input,
            pad_left,
            pad_right,
        } => Some(format!(
            "zero_pad_1d(%{}, pad_left={}, pad_right={})",
            input.index(),
            pad_left,
            pad_right
        )),
        TensorOpKind::Embedding { input, weight } => Some(format!(
            "embedding(%{}, weight=%{})",
            input.index(),
            weight.index()
        )),
        TensorOpKind::LayerNorm {
            input,
            eps,
            axis,
            weight,
            bias,
        } => Some(format!(
            "layer_norm(%{}, eps=%{}, axis={}, weight=%{}, bias=%{})",
            input.index(),
            eps.index(),
            axis,
            weight.index(),
            bias.index()
        )),
        TensorOpKind::Attention {
            q,
            k,
            v,
            mask,
            scale,
        } => {
            let scale_str = match scale {
                Some(s) => format!(", scale={s}"),
                None => String::new(),
            };
            Some(format!(
                "attention(q=%{}, k=%{}, v=%{}, mask={:?}{})",
                q.index(),
                k.index(),
                v.index(),
                mask,
                scale_str
            ))
        }
        TensorOpKind::Transpose { input, axes } => {
            Some(format!("transpose(%{}, axes={:?})", input.index(), axes))
        }
        TensorOpKind::Lstm {
            input,
            hidden_state,
            cell_state,
            weight_ih,
            weight_hh,
            bias,
        } => {
            let bias_str = match bias {
                Some(b) => format!(", bias=%{}", b.index()),
                None => String::new(),
            };
            Some(format!(
                "lstm(%{}, h=%{}, c=%{}, w_ih=%{}, w_hh=%{}{})",
                input.index(),
                hidden_state.index(),
                cell_state.index(),
                weight_ih.index(),
                weight_hh.index(),
                bias_str
            ))
        }
        TensorOpKind::GatedDeltaNet {
            q,
            k,
            v,
            state,
            gate,
            beta,
            scale,
        } => Some(format!(
            "gated_delta_net(q=%{}, k=%{}, v=%{}, state=%{}, gate=%{}, beta=%{}, scale={})",
            q.index(),
            k.index(),
            v.index(),
            state.index(),
            gate.index(),
            beta.index(),
            scale
        )),
        TensorOpKind::BatchNorm {
            input,
            running_mean,
            running_var,
            weight,
            bias,
            eps,
        } => Some(format!(
            "batch_norm(%{}, mean=%{}, var=%{}, weight=%{}, bias=%{}, eps=%{})",
            input.index(),
            running_mean.index(),
            running_var.index(),
            weight.index(),
            bias.index(),
            eps.index()
        )),
        // Core structural ops — handled by the caller.
        _ => None,
    }
}

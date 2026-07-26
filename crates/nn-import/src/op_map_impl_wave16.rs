// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Aten op mappers for commonly needed PyTorch ops (Wave 16).
//!
//! Adds support for:
//!
//! - In-place activations: `relu_`, `sigmoid_`, `tanh_`, `silu_`, `gelu_`
//! - Native normalization: `native_layer_norm`, `native_group_norm`
//! - GRU recurrent: `gru.input`
//! - Complex/FFT ops: `view_as_real`, `view_as_complex`, `fft_rfft`, `fft_irfft`
//! - Dropout variants: `feature_dropout`, `alpha_dropout`
//!
//! **In-place activations:** torch.export frequently emits in-place activation
//! variants (e.g., `relu_` instead of `relu`). These are semantically identical
//! for inference — the in-place flag is a memory optimization hint. We map them
//! to the same `TraceOp` as their out-of-place counterparts.
//!
//! **Native normalization:** PyTorch's `native_layer_norm` and `native_group_norm`
//! are internal decomposition targets emitted by torch.export when the model uses
//! `nn.LayerNorm` or `nn.GroupNorm`. They have different argument layouts from the
//! public `layer_norm`/`group_norm` ops (e.g., `normalized_shape` instead of named
//! weight/bias args), but produce identical results.

use nn_core::dyn_tensor::trace::TraceOp;

use super::{
    first_tensor_name, get_arg, optional_bool, optional_float, optional_int, require_int,
    require_tensor_name, resolve_weight, safe_usize, ImportError, Node, OpMapContext,
};

// =========================================================================
// In-place activation: relu_
// =========================================================================

/// Map `aten.relu_.default` to `TraceOp::Relu`.
///
/// In-place variant of ReLU. Semantically identical for inference.
/// torch.export signature: `(self) -> Tensor`
pub(super) fn map_relu_inplace(node: &Node) -> Result<(TraceOp, Vec<String>), ImportError> {
    let input = first_tensor_name(node)?;
    Ok((TraceOp::Relu, vec![input]))
}

// =========================================================================
// In-place activation: sigmoid_
// =========================================================================

/// Map `aten.sigmoid_.default` to `TraceOp::Sigmoid`.
///
/// In-place variant of Sigmoid. Semantically identical for inference.
/// torch.export signature: `(self) -> Tensor`
pub(super) fn map_sigmoid_inplace(node: &Node) -> Result<(TraceOp, Vec<String>), ImportError> {
    let input = first_tensor_name(node)?;
    Ok((TraceOp::Sigmoid, vec![input]))
}

// =========================================================================
// In-place activation: tanh_
// =========================================================================

/// Map `aten.tanh_.default` to `TraceOp::Tanh`.
///
/// In-place variant of Tanh. Semantically identical for inference.
/// torch.export signature: `(self) -> Tensor`
pub(super) fn map_tanh_inplace(node: &Node) -> Result<(TraceOp, Vec<String>), ImportError> {
    let input = first_tensor_name(node)?;
    Ok((TraceOp::Tanh, vec![input]))
}

// =========================================================================
// In-place activation: silu_
// =========================================================================

/// Map `aten.silu_.default` to `TraceOp::Silu`.
///
/// In-place variant of SiLU (Swish). Semantically identical for inference.
/// torch.export signature: `(self) -> Tensor`
pub(super) fn map_silu_inplace(node: &Node) -> Result<(TraceOp, Vec<String>), ImportError> {
    let input = first_tensor_name(node)?;
    Ok((TraceOp::Silu, vec![input]))
}

// =========================================================================
// In-place activation: gelu_
// =========================================================================

/// Map `aten.gelu_.default` to `TraceOp::Gelu`.
///
/// In-place variant of GELU. Semantically identical for inference.
/// torch.export signature: `(self, approximate: str = "none") -> Tensor`
pub(super) fn map_gelu_inplace(node: &Node) -> Result<(TraceOp, Vec<String>), ImportError> {
    let input = first_tensor_name(node)?;
    Ok((TraceOp::Gelu, vec![input]))
}

// =========================================================================
// Native layer normalization
// =========================================================================

/// Map `aten.native_layer_norm.default` to `TraceOp::LayerNorm`.
///
/// torch.export signature:
///   `(input: Tensor, normalized_shape: int[], weight: Tensor?, bias: Tensor?,
///    eps: float) -> (Tensor, Tensor, Tensor)`
///
/// PyTorch's internal layer norm implementation. Emitted by torch.export
/// when `nn.LayerNorm` is traced. Returns a tuple of (output, mean, rstd)
/// but only the first output (normalized tensor) is used in the trace graph.
///
/// Argument differences from `aten.layer_norm.default`:
/// - Input arg is named `"input"` (same)
/// - Weight/bias may be None (for non-affine LayerNorm)
/// - Has `normalized_shape` int list (not used in TraceOp, shape is runtime)
/// - eps is a positional float argument
pub(super) fn map_native_layer_norm(
    node: &Node,
    ctx: &OpMapContext<'_>,
) -> Result<(TraceOp, Vec<String>), ImportError> {
    let input = require_tensor_name(node, "input")?;
    let eps = optional_float(node, "eps").unwrap_or(1e-5);

    // Weight and bias may be None for non-affine LayerNorm.
    let weight_name = get_arg(node, "weight").ok().and_then(|a| {
        if a.is_none() {
            None
        } else {
            a.as_tensor_name().map(String::from)
        }
    });
    let bias_name = get_arg(node, "bias").ok().and_then(|a| {
        if a.is_none() {
            None
        } else {
            a.as_tensor_name().map(String::from)
        }
    });

    match (weight_name, bias_name) {
        (Some(w), Some(b)) => {
            let weight = resolve_weight(&w, ctx)?;
            let bias = resolve_weight(&b, ctx)?;
            Ok((TraceOp::LayerNorm { eps, weight, bias }, vec![input]))
        }
        _ => {
            // Non-affine: no weight/bias. Map to Custom since LayerNorm
            // requires WeightRef for weight and bias.
            Ok((
                TraceOp::Custom {
                    name: format!("native_layer_norm_no_affine_eps{eps}"),
                },
                vec![input],
            ))
        }
    }
}

// =========================================================================
// Native group normalization
// =========================================================================

/// Map `aten.native_group_norm.default` to `TraceOp::GroupNorm`.
///
/// torch.export signature:
///   `(input: Tensor, weight: Tensor?, bias: Tensor?,
///    N: int, C: int, HxW: int, group: int, eps: float)
///    -> (Tensor, Tensor, Tensor)`
///
/// PyTorch's internal group norm implementation. Emitted by torch.export.
/// Returns (output, mean, rstd) tuple; only output is used.
///
/// Key difference from `aten.group_norm.default`: the `group` arg (not
/// `num_groups`) and explicit N, C, HxW dimensions.
pub(super) fn map_native_group_norm(
    node: &Node,
    ctx: &OpMapContext<'_>,
) -> Result<(TraceOp, Vec<String>), ImportError> {
    let input = require_tensor_name(node, "input")?;
    let eps = optional_float(node, "eps").unwrap_or(1e-5);
    let num_groups = safe_usize(require_int(node, "group")?, "group", &node.target)?;

    // Weight and bias may be None for non-affine GroupNorm.
    let weight_name = get_arg(node, "weight").ok().and_then(|a| {
        if a.is_none() {
            None
        } else {
            a.as_tensor_name().map(String::from)
        }
    });
    let bias_name = get_arg(node, "bias").ok().and_then(|a| {
        if a.is_none() {
            None
        } else {
            a.as_tensor_name().map(String::from)
        }
    });

    match (weight_name, bias_name) {
        (Some(w), Some(b)) => {
            let weight = resolve_weight(&w, ctx)?;
            let bias = resolve_weight(&b, ctx)?;
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
        _ => {
            // Non-affine GroupNorm.
            Ok((
                TraceOp::Custom {
                    name: format!("native_group_norm_no_affine_g{num_groups}_eps{eps}"),
                },
                vec![input],
            ))
        }
    }
}

// =========================================================================
// GRU recurrent
// =========================================================================

/// Map `aten.gru.input` to `TraceOp::Custom`.
///
/// torch.export signature:
///   `(input: Tensor, hx: Tensor, params: Tensor[],
///    has_biases: bool, num_layers: int, dropout: float,
///    train: bool, bidirectional: bool, batch_first: bool)
///    -> (Tensor, Tensor)`
///
/// GRU (Gated Recurrent Unit) is the second most common RNN after LSTM.
/// Maps to Custom since there is no dedicated `TraceOp::Gru` variant.
/// The params list contains weight matrices interleaved:
///   `[w_ih_l0, w_hh_l0, b_ih_l0, b_hh_l0, ...]` for each layer.
pub(super) fn map_gru(node: &Node) -> Result<(TraceOp, Vec<String>), ImportError> {
    let input = require_tensor_name(node, "input")?;
    let hx = require_tensor_name(node, "hx")?;
    let has_biases = optional_bool(node, "has_biases", true);
    let num_layers = optional_int(node, "num_layers").unwrap_or(1);
    let dropout = optional_float(node, "dropout").unwrap_or(0.0);
    let bidirectional = optional_bool(node, "bidirectional", false);
    let batch_first = optional_bool(node, "batch_first", true);

    Ok((
        TraceOp::Custom {
            name: format!(
                "gru_layers{num_layers}_bias{has_biases}_drop{dropout}_bidir{bidirectional}_bf{batch_first}"
            ),
        },
        vec![input, hx],
    ))
}

// =========================================================================
// Complex tensor view: view_as_real
// =========================================================================

/// Map `aten.view_as_real.default` to `TraceOp::Custom`.
///
/// torch.export signature: `(self) -> Tensor`
///
/// Converts a complex tensor to a real tensor with an extra trailing
/// dimension of size 2, where `[..., 0]` is real and `[..., 1]` is
/// imaginary. Common in FFT/STFT pipelines.
/// Input shape: `[*]` (complex) -> Output shape: `[*, 2]` (real).
pub(super) fn map_view_as_real(node: &Node) -> Result<(TraceOp, Vec<String>), ImportError> {
    let input = first_tensor_name(node)?;
    Ok((
        TraceOp::Custom {
            name: "view_as_real".to_string(),
        },
        vec![input],
    ))
}

// =========================================================================
// Complex tensor view: view_as_complex
// =========================================================================

/// Map `aten.view_as_complex.default` to `TraceOp::Custom`.
///
/// torch.export signature: `(self) -> Tensor`
///
/// Converts a real tensor with trailing dimension of size 2 to a complex
/// tensor. Inverse of `view_as_real`.
/// Input shape: `[*, 2]` (real) -> Output shape: `[*]` (complex).
pub(super) fn map_view_as_complex(node: &Node) -> Result<(TraceOp, Vec<String>), ImportError> {
    let input = first_tensor_name(node)?;
    Ok((
        TraceOp::Custom {
            name: "view_as_complex".to_string(),
        },
        vec![input],
    ))
}

// =========================================================================
// FFT: Real-to-complex FFT (rfft)
// =========================================================================

/// Map `aten.fft_rfft.default` to `TraceOp::Custom`.
///
/// torch.export signature:
///   `(self, n: int? = None, dim: int = -1, norm: str? = None) -> Tensor`
///
/// Computes the one-dimensional discrete Fourier Transform for real input.
/// Output is complex-valued with shape `[..., n//2 + 1]` along the
/// transform dimension. Essential for audio/signal processing models.
pub(super) fn map_fft_rfft(node: &Node) -> Result<(TraceOp, Vec<String>), ImportError> {
    let input = first_tensor_name(node)?;
    let n = optional_int(node, "n");
    let dim = optional_int(node, "dim").unwrap_or(-1);
    let norm = get_arg(node, "norm")
        .ok()
        .and_then(|a| {
            if a.is_none() {
                None
            } else {
                a.as_string().map(String::from)
            }
        })
        .unwrap_or_else(|| "backward".to_string());

    let n_str = match n {
        Some(v) => format!("n{v}"),
        None => "auto".to_string(),
    };

    Ok((
        TraceOp::Custom {
            name: format!("fft_rfft_{n_str}_dim{dim}_{norm}"),
        },
        vec![input],
    ))
}

// =========================================================================
// FFT: Complex-to-real inverse FFT (irfft)
// =========================================================================

/// Map `aten.fft_irfft.default` to `TraceOp::Custom`.
///
/// torch.export signature:
///   `(self, n: int? = None, dim: int = -1, norm: str? = None) -> Tensor`
///
/// Computes the inverse of `rfft`. Input is complex-valued, output is real.
/// Output shape along transform dim is `n` (or `2 * (input_size - 1)` if
/// `n` is None). Essential for audio synthesis and inverse STFT.
pub(super) fn map_fft_irfft(node: &Node) -> Result<(TraceOp, Vec<String>), ImportError> {
    let input = first_tensor_name(node)?;
    let n = optional_int(node, "n");
    let dim = optional_int(node, "dim").unwrap_or(-1);
    let norm = get_arg(node, "norm")
        .ok()
        .and_then(|a| {
            if a.is_none() {
                None
            } else {
                a.as_string().map(String::from)
            }
        })
        .unwrap_or_else(|| "backward".to_string());

    let n_str = match n {
        Some(v) => format!("n{v}"),
        None => "auto".to_string(),
    };

    Ok((
        TraceOp::Custom {
            name: format!("fft_irfft_{n_str}_dim{dim}_{norm}"),
        },
        vec![input],
    ))
}

// =========================================================================
// Dropout variants (all identity at inference)
// =========================================================================

/// Map `aten.feature_dropout.default` to `TraceOp::Dropout`.
///
/// torch.export signature: `(input: Tensor, p: float, train: bool) -> Tensor`
///
/// Feature dropout (drops entire feature maps/channels). At inference time
/// this is a no-op (identity), same as regular dropout.
pub(super) fn map_feature_dropout(node: &Node) -> Result<(TraceOp, Vec<String>), ImportError> {
    let input = first_tensor_name(node)?;
    Ok((TraceOp::Dropout, vec![input]))
}

/// Map `aten.alpha_dropout.default` to `TraceOp::Dropout`.
///
/// torch.export signature: `(input: Tensor, p: float, train: bool) -> Tensor`
///
/// Alpha dropout (used with SELU activation to maintain self-normalizing
/// properties). At inference time this is a no-op (identity).
pub(super) fn map_alpha_dropout(node: &Node) -> Result<(TraceOp, Vec<String>), ImportError> {
    let input = first_tensor_name(node)?;
    Ok((TraceOp::Dropout, vec![input]))
}

// =========================================================================
// Tests
// =========================================================================

#[cfg(test)]
#[path = "op_map_impl_wave16_tests.rs"]
mod tests;
